// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The M1 capture's writer (oot2/capture.h has the switches and the file format; tools/check_capture.py reads it).
//
// Locks, in the one order they are taken: g_frame (the frame boundary, the thread table and the shown ticks, between
// two windows presenting from two threads), then g_file (the file and the spare buffer), then g_data (the buffer the
// lines go to, the one line under construction, and the state the lines carry). The presenting thread takes g_frame and
// g_data, never g_file. Nothing done under g_data blocks, allocates through the runtime or can stop the run (a buffer
// that cannot grow drops the line and counts its bytes), so a fatal raised anywhere can always take g_file and g_data
// for the closing record. g_sub guards the frame's submitters. The closing record reads no kernel or memory state of
// the runtime, whose locks the failing thread may hold.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "oot2/capture.h"
#include "oot2/gfx.h"
#include "oot2/kernel.h"
#include "oot2/nvn_pace.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "plat.h"

#define FORMAT 1
#define TITLE_DEFAULT 1800ull        // presents from the first: the title pass capture's menu at 1,459 and 1,531, plus a margin
#define MAX_THREADS 512              // threads read per frame; a frame with more lists the first ones and is counted
#define MAX_SEEN 2048                // threads described by a "thread" line
#define SUBMITTERS 8                 // distinct submitting threads kept per frame
#define SHOWN_LAG 8                  // presents whose shown tick is asked again before it is given up
#define MAX_NOTES 16
#define SECOND_NS 1000000000ull

typedef struct {
    char *p;
    size_t n, cap;
} Buf;

typedef struct {
    char s[32768];   // a frame line of MAX_THREADS threads fits
    size_t n;
} Line;

typedef struct {
    uint32_t h;
    uint64_t n;
} Submitter;

static int g_started, g_ending, g_ended, g_length_hit;   // atomic
static PlatMutex g_frame = PLAT_MUTEX_INIT, g_file = PLAT_MUTEX_INIT, g_data = PLAT_MUTEX_INIT, g_sub = PLAT_MUTEX_INIT;
static FILE *g_f;                                 // g_file
static Buf g_spare;                               // g_file
static Buf g_buf;                                 // g_data
static Line g_l;                                  // g_data
static uint64_t g_lost;                           // bytes dropped: the buffer could not grow (g_data)
static uint64_t g_origin;                         // plat_now_ns at capture_start
static uint64_t g_title_seq, g_length_ns;
static int g_title_seen, g_have_prev;             // g_data
static uint64_t g_title_t, g_prev_t, g_prev_due, g_last_seq;   // g_data
static struct { char key[64], value[256]; } g_notes[MAX_NOTES];   // before capture_start (g_data)
static int g_nnotes;
static uint32_t g_seen[MAX_SEEN];                 // g_frame
static uint64_t g_seen_entry[MAX_SEEN];
static int g_nseen;
static RtThreadTime g_tt[MAX_THREADS];            // g_frame
static uint64_t g_threads_over;                   // g_frame
static uint64_t g_shown_slot[SHOWN_LAG];          // g_frame: slot seq % SHOWN_LAG holds seq once its shown tick is out
static Submitter g_subs[SUBMITTERS];              // g_sub
static int g_nsubs;
static uint64_t g_counters[CAPTURE_COUNTERS];     // atomic
static uint64_t g_cost_n, g_cost_sum, g_cost_max, g_sampler_cpu;   // atomic

// ---------- lines and the buffer
static int reserve(Buf *b, size_t more) {
    if (b->n + more <= b->cap) return 1;
    size_t cap = b->cap ? b->cap : 1 << 16;
    while (cap < b->n + more) cap *= 2;
    char *p = realloc(b->p, cap);
    if (!p) return 0;
    b->p = p, b->cap = cap;
    return 1;
}

// Text: a tab or a line break inside it becomes a space, so a record stays one line of its fields.
static void l_str(Line *l, const char *s) {
    while (*s && l->n < sizeof l->s - 1) {
        char ch = *s++;
        l->s[l->n++] = ch == '\t' || ch == '\n' || ch == '\r' ? ' ' : ch;
    }
}
static void l_tab(Line *l) {
    if (l->n < sizeof l->s - 1) l->s[l->n++] = '\t';
}
static void l_u(Line *l, uint64_t v) {
    char d[24];
    int k = 0;
    do d[k++] = (char)('0' + v % 10); while (v /= 10);
    if (l->n + (size_t)k + 1 >= sizeof l->s) return;
    while (k) l->s[l->n++] = d[--k];
}
static void l_i(Line *l, int64_t v) {
    if (v < 0) l_str(l, "-");
    l_u(l, v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v);
}
static void l_x(Line *l, uint64_t v) {
    static const char hex[] = "0123456789abcdef";
    char d[16];
    int k = 0;
    do d[k++] = hex[v & 15]; while (v >>= 4);
    if (l->n + (size_t)k + 3 >= sizeof l->s) return;
    l->s[l->n++] = '0', l->s[l->n++] = 'x';
    while (k) l->s[l->n++] = d[--k];
}
static void l_t(Line *l, uint64_t v) { l_tab(l), l_u(l, v); }   // a tab, then the number
static void l_opt(Line *l, uint64_t v) {                       // a tab, then the number, or -1 when unknown
    l_tab(l);
    if (v == UINT64_MAX) l_str(l, "-1");
    else l_u(l, v);
}

// g_data held: the line goes to the buffer whole, or is dropped whole and counted.
static void append(Line *l) {
    l->s[l->n++] = '\n';
    if (reserve(&g_buf, l->n)) {
        memcpy(g_buf.p + g_buf.n, l->s, l->n);
        g_buf.n += l->n;
    } else {
        g_lost += l->n;
    }
    l->n = 0;
}

static void header(const char *key, const char *value) {   // g_data held
    l_str(&g_l, "header"), l_tab(&g_l), l_str(&g_l, key), l_tab(&g_l), l_str(&g_l, value);
    append(&g_l);
}
static void header_u(const char *key, uint64_t v) {
    char s[24];
    snprintf(s, sizeof s, "%llu", (unsigned long long)v);
    header(key, s);
}

// Write what the buffer holds. g_file held, g_data not held.
static void drain(void) {
    plat_mutex_lock(&g_data);
    Buf full = g_buf;
    g_buf = g_spare;
    g_buf.n = 0;
    plat_mutex_unlock(&g_data);
    if (g_f && full.n) fwrite(full.p, 1, full.n, g_f);
    if (g_f) fflush(g_f);
    full.n = 0;
    g_spare = full;
}

static uint64_t since(uint64_t now) { return now > g_origin ? now - g_origin : 0; }

static void counters_now(uint64_t out[CAPTURE_COUNTERS]) {
    for (int i = 0; i < CAPTURE_COUNTERS; i++) out[i] = __atomic_load_n(&g_counters[i], __ATOMIC_RELAXED);
}
static void l_counters(Line *l, const uint64_t c[CAPTURE_COUNTERS]) {
    for (int i = 0; i < CAPTURE_COUNTERS; i++) l_t(l, c[i]);
}
static void l_cost(Line *l) {
    l_t(l, __atomic_load_n(&g_cost_n, __ATOMIC_RELAXED)), l_t(l, __atomic_load_n(&g_cost_sum, __ATOMIC_RELAXED)),
        l_t(l, __atomic_load_n(&g_cost_max, __ATOMIC_RELAXED)), l_t(l, __atomic_load_n(&g_sampler_cpu, __ATOMIC_RELAXED));
}

// What memory the process and the machine hold now.
typedef struct {
    PlatProcMem proc;
    uint64_t total, available, pages;
} MemNow;
static MemNow mem_now(void) {
    MemNow m;
    plat_process_memory(&m.proc);
    plat_system_memory(&m.total, &m.available);
    m.pages = plat_pages_output();
    return m;
}
static void l_mem(Line *l, const MemNow *m) {
    l_t(l, m->proc.working_set), l_t(l, m->proc.private_bytes), l_t(l, m->available), l_opt(l, m->pages);
}

// ---------- the switches
static uint64_t env_count(const char *name, uint64_t dflt) {
    const char *e = getenv(name);
    if (!e || !*e) return dflt;
    char *end;
    unsigned long long v = strtoull(e, &end, 10);
    if (*end || v == 0) rt_fatal("%s=%s: expected a present count of at least 1", name, e);
    return v;
}

static uint64_t env_seconds(const char *name) {
    const char *e = getenv(name);
    if (!e || !*e) return 0;
    char *end;
    double v = strtod(e, &end);
    if (*end || !(v >= 0) || v > 1e6) rt_fatal("%s=%s: expected seconds, 0 to 1000000", name, e);
    return (uint64_t)(v * 1e9);
}

// ---------- the sampling thread: memory, counters and cost once a second, and the buffer to the file
static void sampler_main(void *arg) {
    (void)arg;
    uint64_t next = g_origin + SECOND_NS;
    while (!__atomic_load_n(&g_ending, __ATOMIC_ACQUIRE)) {
        uint64_t now = plat_now_ns();
        if (now < next) {
            uint64_t left = next - now;
            plat_sleep_ns(left < 100000000ull ? left : 100000000ull);   // a tenth of a second at most: the end is seen soon
            continue;
        }
        next += SECOND_NS;
        if (next <= now) next = now + SECOND_NS;   // a late wake-up skips a second, it does not catch up
        MemNow m = mem_now();
        uint64_t c[CAPTURE_COUNTERS];
        counters_now(c);
        plat_mutex_lock(&g_file);
        plat_mutex_lock(&g_data);
        if (!__atomic_load_n(&g_ended, __ATOMIC_ACQUIRE)) {
            uint64_t t = since(now);
            l_str(&g_l, "memory"), l_t(&g_l, g_last_seq), l_t(&g_l, t), l_mem(&g_l, &m);
            append(&g_l);
            l_str(&g_l, "count"), l_t(&g_l, g_last_seq), l_t(&g_l, t), l_counters(&g_l, c);
            append(&g_l);
            l_str(&g_l, "cost"), l_t(&g_l, g_last_seq), l_t(&g_l, t), l_cost(&g_l);
            append(&g_l);
        }
        plat_mutex_unlock(&g_data);
        if (!__atomic_load_n(&g_ended, __ATOMIC_ACQUIRE)) drain();
        plat_mutex_unlock(&g_file);
        __atomic_store_n(&g_sampler_cpu, plat_thread_cpu_ns(NULL), __ATOMIC_RELAXED);
    }
}

// ---------- the closing record
static const char *reason(int code) {
    switch (code) {
    case 0: return __atomic_load_n(&g_length_hit, __ATOMIC_ACQUIRE) ? "length" : "exit";
    case RT_EXIT_FATAL: return "fatal";
    case RT_EXIT_UNIMPLEMENTED: return "unimplemented";
    case RT_EXIT_GUEST_STOP: return "guest-stop";
    case RT_EXIT_CENSUS: return "census-limit";
    default: return "other";
    }
}

void capture_end(int code) {
    if (!__atomic_load_n(&g_started, __ATOMIC_ACQUIRE) || __atomic_exchange_n(&g_ending, 1, __ATOMIC_ACQ_REL)) return;
    MemNow m = mem_now();
    uint64_t c[CAPTURE_COUNTERS];
    counters_now(c);
    uint64_t now = plat_now_ns();
    int fatal = code == RT_EXIT_FATAL || code == RT_EXIT_UNIMPLEMENTED || code == RT_EXIT_GUEST_STOP;
    uint64_t over = __atomic_load_n(&g_threads_over, __ATOMIC_RELAXED);
    plat_mutex_lock(&g_file);
    plat_mutex_lock(&g_data);
    if (g_lost) header_u("lost_bytes", g_lost);   // said before the end, which stays the last line
    if (over) header_u("frames_with_threads_unlisted", over);
    l_str(&g_l, "end"), l_tab(&g_l), l_i(&g_l, code), l_tab(&g_l), l_str(&g_l, reason(code)), l_t(&g_l, g_last_seq),
        l_t(&g_l, since(now)), l_t(&g_l, (uint64_t)fatal), l_counters(&g_l, c), l_mem(&g_l, &m), l_cost(&g_l);
    append(&g_l);
    __atomic_store_n(&g_ended, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&g_data);
    drain();
    if (g_f) {
        int bad = ferror(g_f) | fclose(g_f);
        g_f = NULL;
        if (bad) rt_log("capture: the file could not be finished");
    }
    __atomic_store_n(&g_started, 0, __ATOMIC_RELEASE);
    plat_mutex_unlock(&g_file);
}

static void stop_writer(int code, const char *line) {
    (void)line;   // numbers only: the stop line names game functions, and census.tsv and the log keep it
    capture_end(code);
}

// ---------- start
void capture_start(const char *folder) {
    const char *on = getenv("OOT2_CAPTURE");
    if (!on || strcmp(on, "1")) return;
    if (__atomic_load_n(&g_started, __ATOMIC_ACQUIRE) || __atomic_load_n(&g_ending, __ATOMIC_ACQUIRE))
        rt_fatal("capture_start: called twice");
    g_title_seq = env_count("OOT2_CAPTURE_TITLE", TITLE_DEFAULT);
    g_length_ns = env_seconds("OOT2_CAPTURE_SECONDS");
    uint64_t pages = plat_pages_output();   // opens the counter now, before the timed work
    time_t wall = time(NULL);
    struct tm tm;
#ifdef _WIN32
    gmtime_s(&tm, &wall);
#else
    gmtime_r(&wall, &tm);
#endif
    char stamp[32], iso[32], path[4200], secs[32];
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    strftime(iso, sizeof iso, "%Y-%m-%dT%H:%M:%SZ", &tm);
    snprintf(path, sizeof path, "%s/capture-%s-%u.tsv", folder, stamp, plat_pid());
    snprintf(secs, sizeof secs, "%.3f", (double)g_length_ns / 1e9);
    g_f = fopen(path, "wb");
    if (!g_f) rt_fatal("capture: cannot write %s", path);
    g_origin = plat_now_ns();
    uint64_t total, avail;
    plat_system_memory(&total, &avail);

    plat_mutex_lock(&g_data);
    l_str(&g_l, "oot2-capture"), l_t(&g_l, FORMAT);
    append(&g_l);
    header_u("format", FORMAT);
    header("started_utc", iso);
    header_u("pid", plat_pid());
    header("back_end", gfx_backend()->name);
    if (gfx_backend() == &gfx_null_backend) {   // any other back end notes these itself (capture_note)
        header("driver", "none (null back end)");
        header("display_mode", "none (null back end)");
        header("implicit_vulkan_layers", "none (null back end)");
        header("gpu_timestamps", "none (null back end)");
    }
    header_u("ram_total", total);
    header_u("ram_available", avail);
    header_u("processors", (uint64_t)plat_cpu_count());
    header("pages_output_counter", plat_pages_output_name());
    if (pages == UINT64_MAX) header("pages_output_counter_state", "unreadable on this machine; written as -1");
    header_u("census_mode", (uint64_t)rt_census_on());
    header_u("collect_variants", (uint64_t)(rt_setting(RT_SET_collect_variants) != 0));   // a development run, not a proof
    // A test switch that changes the picture (settings.h): the proof judge refuses a capture whose run had it on, even a run
    // that never reached a movie.
    header("test_movie_blit", rt_setting(RT_SET_test_movie_blit) ? "1 (a test switch that changes the pictures of the movies: not a proof run)" : "0");
    header_u("title_mark_present", g_title_seq);
    header("title_mark_rule", "a present count from the first present (OOT2_CAPTURE_TITLE)");
    header("length_seconds", secs);
    header("times", "ns of the host's monotonic clock since the capture started; ticks of the virtual 60 Hz clock");
    for (int i = 0; i < g_nnotes; i++) header(g_notes[i].key, g_notes[i].value);
    g_nnotes = 0;
    plat_mutex_unlock(&g_data);

    rt_add_stop_writer(stop_writer);
    __atomic_store_n(&g_started, 1, __ATOMIC_RELEASE);
    if (!plat_thread_start(sampler_main, NULL, 1ull << 20)) rt_fatal("capture: cannot start the sampling thread");
    rt_log("capture: writing %s (title mark at present %llu, %s)", path, (unsigned long long)g_title_seq,
           g_length_ns ? "the run ends by itself after the capture's length" : "until the run ends");
}

int capture_on(void) { return __atomic_load_n(&g_started, __ATOMIC_ACQUIRE); }

void capture_note(const char *key, const char *value) {
    plat_mutex_lock(&g_data);
    if (__atomic_load_n(&g_started, __ATOMIC_ACQUIRE)) {
        if (!__atomic_load_n(&g_ended, __ATOMIC_ACQUIRE)) header(key, value);
    } else if (g_nnotes < MAX_NOTES) {
        snprintf(g_notes[g_nnotes].key, sizeof g_notes[g_nnotes].key, "%s", key);
        snprintf(g_notes[g_nnotes].value, sizeof g_notes[g_nnotes].value, "%s", value);
        g_nnotes++;
    }
    plat_mutex_unlock(&g_data);
}

void capture_count(int counter, uint64_t n) {
    if (counter < 0 || counter >= CAPTURE_COUNTERS) rt_fatal("capture_count: counter %d", counter);
    __atomic_fetch_add(&g_counters[counter], n, __ATOMIC_RELAXED);
}

void capture_gpu_pass(uint64_t seq, int pass, uint64_t ns) {
    if (!capture_on()) return;
    plat_mutex_lock(&g_data);
    if (!__atomic_load_n(&g_ended, __ATOMIC_ACQUIRE)) {
        l_str(&g_l, "gpu"), l_t(&g_l, seq), l_tab(&g_l), l_i(&g_l, pass), l_t(&g_l, ns);
        append(&g_l);
    }
    plat_mutex_unlock(&g_data);
}

void capture_submit(void) {
    if (!capture_on()) return;
    uint32_t h = rt_kernel_current();
    plat_mutex_lock(&g_sub);
    int i = 0;
    while (i < g_nsubs && g_subs[i].h != h) i++;
    if (i == g_nsubs && g_nsubs < SUBMITTERS) g_subs[g_nsubs++] = (Submitter){h, 0};
    if (i < g_nsubs) g_subs[i].n++;
    plat_mutex_unlock(&g_sub);
}

// ---------- a present: the frame that ends with it
static int seen(uint32_t h, uint64_t entry) {   // g_frame held
    for (int i = 0; i < g_nseen; i++)
        if (g_seen[i] == h && g_seen_entry[i] == entry) return 1;
    if (g_nseen < MAX_SEEN) g_seen[g_nseen] = h, g_seen_entry[g_nseen++] = entry;
    return 0;
}

// The shown ticks the back end reported since the presents were written, for the last SHOWN_LAG presents before seq.
// g_frame held, g_data not held. The null back end never knows one, so it is not asked.
static void shown_ticks(uint64_t seq, uint64_t shown_now) {
    const GfxBackend *be = gfx_backend();
    if (!be->present_shown || be == &gfx_null_backend) return;
    if (shown_now) g_shown_slot[seq % SHOWN_LAG] = seq;
    for (uint64_t s = seq > SHOWN_LAG ? seq - SHOWN_LAG + 1 : 1; s < seq; s++) {
        NvnPresentRecord o;
        if (g_shown_slot[s % SHOWN_LAG] == s || !nvn_present_record(s, &o) || !o.shown) continue;
        g_shown_slot[s % SHOWN_LAG] = s;
        plat_mutex_lock(&g_data);
        if (!__atomic_load_n(&g_ended, __ATOMIC_ACQUIRE)) {
            l_str(&g_l, "shown"), l_t(&g_l, s), l_t(&g_l, o.shown);
            append(&g_l);
        }
        plat_mutex_unlock(&g_data);
    }
}

void capture_present(uint64_t seq) {
    if (!capture_on()) return;
    uint64_t cpu0 = plat_thread_cpu_ns(NULL);
    plat_mutex_lock(&g_frame);
    uint64_t now = plat_now_ns();
    NvnPresentRecord r;
    if (!nvn_present_record(seq, &r)) memset(&r, 0, sizeof r);
    // The boundary first, then the times: they are then the frame that just ended. Read the other way round they would
    // be the frame before it, and every thread's time would land one frame late.
    rt_kernel_frame();
    int n = rt_kernel_thread_times(g_tt, MAX_THREADS);
    if (n > MAX_THREADS) __atomic_fetch_add(&g_threads_over, 1, __ATOMIC_RELAXED), n = MAX_THREADS;
    uint32_t presenter = rt_kernel_current();
    plat_mutex_lock(&g_sub);
    uint32_t submitter = 0;
    uint64_t submits = 0;
    int submitters = g_nsubs;
    for (int i = 0; i < g_nsubs; i++)
        if (g_subs[i].n > submits) submitter = g_subs[i].h, submits = g_subs[i].n;
    g_nsubs = 0;
    plat_mutex_unlock(&g_sub);
    uint64_t c[CAPTURE_COUNTERS];
    counters_now(c);

    plat_mutex_lock(&g_data);
    int live = !__atomic_load_n(&g_ended, __ATOMIC_ACQUIRE), end_now = 0;
    if (live) {
        uint64_t t = since(now);
        // Ticks lost: this present's due tick against the previous one's plus the interval the gate held for.
        int64_t e = r.requested > r.held ? r.requested : r.held;
        uint64_t late = 0, frame_ns = 0;
        if (g_have_prev) {
            uint64_t want = g_prev_due + (e > 0 ? (uint64_t)e : 0);
            late = e > 0 && r.due > want ? r.due - want : 0;
            frame_ns = now - g_prev_t;
        }
        g_have_prev = 1, g_prev_t = now, g_prev_due = r.due, g_last_seq = seq;
        for (int i = 0; i < n; i++)
            if ((g_tt[i].cpu_ns || g_tt[i].core_ns) && !seen(g_tt[i].handle, g_tt[i].entry)) {
                l_str(&g_l, "thread"), l_tab(&g_l), l_x(&g_l, g_tt[i].handle), l_tab(&g_l), l_x(&g_l, g_tt[i].entry),
                    l_tab(&g_l), l_i(&g_l, g_tt[i].priority);
                append(&g_l);
            }
        l_str(&g_l, "present"), l_t(&g_l, seq), l_t(&g_l, t), l_t(&g_l, r.due), l_t(&g_l, r.shown), l_tab(&g_l),
            l_i(&g_l, r.requested), l_tab(&g_l), l_i(&g_l, r.held), l_t(&g_l, frame_ns), l_t(&g_l, late);
        append(&g_l);
        l_str(&g_l, "frame"), l_t(&g_l, seq), l_tab(&g_l), l_x(&g_l, submitter), l_t(&g_l, submits),
            l_t(&g_l, (uint64_t)submitters), l_tab(&g_l), l_x(&g_l, presenter);
        for (int i = 0; i < n; i++)
            if (g_tt[i].cpu_ns || g_tt[i].core_ns)
                l_tab(&g_l), l_x(&g_l, g_tt[i].handle), l_t(&g_l, g_tt[i].cpu_ns), l_t(&g_l, g_tt[i].core_ns);
        append(&g_l);
        if (!g_title_seen && seq >= g_title_seq) {
            g_title_seen = 1, g_title_t = t;
            l_str(&g_l, "title"), l_t(&g_l, seq), l_t(&g_l, t), l_counters(&g_l, c);
            append(&g_l);
        }
        end_now = g_title_seen && g_length_ns && t - g_title_t >= g_length_ns;
    }
    plat_mutex_unlock(&g_data);
    if (live) shown_ticks(seq, r.shown);
    plat_mutex_unlock(&g_frame);
    if (!live) return;

    uint64_t cost = plat_thread_cpu_ns(NULL) - cpu0;
    __atomic_fetch_add(&g_cost_n, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_cost_sum, cost, __ATOMIC_RELAXED);
    uint64_t m = __atomic_load_n(&g_cost_max, __ATOMIC_RELAXED);
    while (cost > m && !__atomic_compare_exchange_n(&g_cost_max, &m, cost, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
    if (end_now && !__atomic_exchange_n(&g_length_hit, 1, __ATOMIC_ACQ_REL)) {
        char line[160];
        snprintf(line, sizeof line, "the capture ended at its length, %.3f s after the title mark, at present %llu",
                 (double)g_length_ns / 1e9, (unsigned long long)seq);
        rt_log("%s", line);
        rt_census_end(0, line);
    }
}
