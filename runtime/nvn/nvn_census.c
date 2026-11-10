// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the front end counts per present (the null back end boot). Every present logs itself the first time and then every 60th, so a run
// stopped by a time limit still says how many frames it presented. In census mode (OOT2_CENSUS=1, runtime.h) the table of
// host-function calls is also copied at present 1 and at every 60th present, and nvn_census_write turns the copies into
// calls per frame in the NVN2 call census's six columns (phase, frames, slot, function, calls_per_frame, calls): phase "boot" is
// everything before present 1 (frames 0), "wNNN" is one window of presents, "all" the whole run after present 1. The
// slot column is left empty; an offline join to the game's slot map fills it. Nothing here includes the hos layer.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_api.h"
#include "nvn_cmd.h"
#include "oot2/ledger.h"
#include "oot2/memory.h"
#include "oot2/nvn_order.h"
#include "oot2/nvn_pace.h"
#include "oot2/replay_count.h"
#include "oot2/settings.h"
#include "plat.h"

#define WINDOW 60            // presents per window
#define FLUSH_EVERY 600      // a census-mode run rewrites its file this often (a killed run writes nothing at its end)
#define MAX_SNAPS 1024       // 1 + 1023 windows: about 17 minutes at 60 Hz; later windows are not kept

typedef struct {
    uint64_t present, ns;
    uint64_t *calls;
    uint64_t ops[OP_COUNT_];   // records the null back end had replayed, by op (the second draw count)
    NvnPoolStats pools;        // the pools at this snapshot (point 3)
    uint64_t priv;             // the process's private memory at this snapshot, bytes
} Snap;
static Snap g_snap[MAX_SNAPS];
static int g_nsnap, g_nslots, g_full_said;
static PlatMutex g_lock = PLAT_MUTEX_INIT;
static uint64_t g_first_ns;
static NvnStatSource g_sources[NVN_CENSUS_SOURCES];   // a back end's census rows (nvn_census_add_source)
static int g_nsources;   // atomic: a source is written before the count that names it

static uint64_t g_title_present;   // the present of the title line, 0 before it (census mode: its snapshot is taken at that present)

static void take(uint64_t seq) {
    int n = rt_hostfn_count();
    uint64_t *calls = malloc((size_t)n * sizeof *calls);
    if (!calls) rt_fatal("nvn census: out of memory for a snapshot of %d slots", n);
    rt_hostfn_snapshot(calls, n);
    static Snap tmp;   // one thread takes snapshots: the presenting thread
    tmp = (Snap){0};
    gfx_null_op_counts(tmp.ops);
    nvn_pool_stats(&tmp.pools);
    // rt_mem_measure walks the page map and takes tens of milliseconds: every fifth snapshot, and at the title line's window,
    // so the frame it lands in does not stretch every window
    if (g_nsnap % 5 == 0 || seq == g_title_present) {
        RtMemStats ms;
        rt_mem_measure(&ms);
        tmp.priv = ms.process_private;
    }
    plat_mutex_lock(&g_lock);
    if (g_nsnap == MAX_SNAPS) {
        int say = !g_full_said;
        g_full_said = 1;
        plat_mutex_unlock(&g_lock);
        free(calls);
        if (say) rt_log("nvn: %d snapshots kept; later windows are not recorded", MAX_SNAPS);
        return;
    }
    g_nslots = n;
    tmp.present = seq, tmp.ns = plat_now_ns(), tmp.calls = calls;
    g_snap[g_nsnap++] = tmp;
    plat_mutex_unlock(&g_lock);
}

// The title, as the log shows it (the movie player, the title pass capture). The movie's frame is 14 compute dispatches (13 of them tensor
// dispatches of the neural upscaler, one the conversion of the decoded planes to RGBA) and 10 nvnCommandBufferSetRenderTargets
// calls a present; the menu then appears over the looping movie by itself. A window of 60 presents whose calls a present
// reach both counts is a "movie window"; two in a row is the title's movie: the log says so once. The menu's own draws are
// not part of that line: each window's line also carries the draws a present, whose step up over the movie's own marks the
// menu fading in (the movie's full-screen triangle and the window pass may sit in command memory the front end does not
// see, so the draw count is a floor).
#define TITLE_DISPATCHES 14
#define TITLE_TARGETS 10
static uint64_t g_prev_calls[3], g_prev_present;
static int g_movie_windows, g_title_said;

static uint64_t named_calls(const char *name) { return rt_hostfn_calls(nvn_proc_address(name)); }

// The intervals the game asked for over the last window: the present-record ring keeps the newest 512 presents, so it is
// read once a window. "requested 1 x60" is the normal window; any other interval, or a hold in effect, is marked. A
// present made while a fixed frame rate target was set is paced by the target and not by the game's interval, so the
// line names the target (the newest one met) and how many presents of the window it paced.
static void interval_line(uint64_t seq) {
    uint64_t first = seq > WINDOW ? seq - WINDOW + 1 : 1;
    int req[8] = {0}, held[8] = {0}, odd_req = 0, odd_held = 0, n = 0, targeted = 0, target = 0;
    for (uint64_t s = first; s <= seq; s++) {
        NvnPresentRecord r;
        if (!nvn_present_record(s, &r)) continue;
        n++;
        if (r.target) targeted++, target = r.target;
        if (r.requested >= 0 && r.requested < 8) req[r.requested]++;
        else odd_req++;
        if (r.held >= 0 && r.held < 8) held[r.held]++;
        else odd_held++;
    }
    char rq[160] = "", hd[160] = "";
    int marked = odd_req != 0;
    for (int i = 0; i < 8; i++) {
        if (req[i]) snprintf(rq + strlen(rq), sizeof rq - strlen(rq), "%s%d x%d", *rq ? ", " : "", i, req[i]);
        if (req[i] && i != 1) marked = 1;
        if (held[i]) snprintf(hd + strlen(hd), sizeof hd - strlen(hd), "%s%d x%d", *hd ? ", " : "", i, held[i]);
        if (held[i] && i != 0) marked = 1;
    }
    if (odd_req) snprintf(rq + strlen(rq), sizeof rq - strlen(rq), "%sother x%d", *rq ? ", " : "", odd_req);
    if (odd_held) marked = 1;
    char tg[96] = "";
    if (targeted)
        snprintf(tg, sizeof tg, ", frame rate target %d a second on %d (the game's interval is ignored on those)", target,
                 targeted);
    rt_log("nvn: present %llu intervals over %d presents: requested %s, held %s%s%s", (unsigned long long)seq, n,
           *rq ? rq : "none", *hd ? hd : "none", tg, marked ? " (NOT every interval 1 with no hold)" : "");
}

// The calls a present of the last window, once per window, for the log line and the title.
static void window_line(uint64_t seq, double secs) {
    uint64_t cur[3] = {named_calls("nvnCommandBufferSetRenderTargets"), named_calls("nvnCommandBufferDispatchCompute"),
                       named_calls("nvnCommandBufferDrawElementsInstanced")};
    uint64_t frames = seq - g_prev_present;
    if (seq == 1 || !frames) {
        g_prev_present = seq;
        memcpy(g_prev_calls, cur, sizeof cur);
        rt_log("nvn: present %llu, %.2f s after the first", (unsigned long long)seq, secs);
        return;
    }
    double rt = (double)(cur[0] - g_prev_calls[0]) / (double)frames, dp = (double)(cur[1] - g_prev_calls[1]) / (double)frames,
           dr = (double)(cur[2] - g_prev_calls[2]) / (double)frames;
    g_prev_present = seq;
    memcpy(g_prev_calls, cur, sizeof cur);
    rt_log("nvn: present %llu, %.2f s after the first; a present: %.1f SetRenderTargets, %.1f DispatchCompute, %.1f DrawElementsInstanced",
           (unsigned long long)seq, secs, rt, dp, dr);
    interval_line(seq);
    g_movie_windows = (rt >= TITLE_TARGETS && dp >= TITLE_DISPATCHES) ? g_movie_windows + 1 : 0;
    if (g_movie_windows >= 2 && !g_title_said) {
        g_title_said = 1;
        g_title_present = seq;
        rt_log("nvn: title: the movie's frame (%d dispatches and %d SetRenderTargets a present) in two windows in a row, at "
               "present %llu", TITLE_DISPATCHES, TITLE_TARGETS, (unsigned long long)seq);
    }
}

// ---- the census timer's rows (runtime.h, hostfn.c)
// At each present the presenting thread notes, for every thread that made a timed NVN2 call since the last present, what its
// calls cost in exclusive thread CPU time: NVN2 entry points apart from QueueSubmitCommands, QueueSubmitCommands (the null
// back end completes the submission inside it, which is not the Vulkan replay), every other host function, and the guest
// callbacks. At each window it notes every function's calls and exclusive time over the window. census-time.tsv is written by
// nvn_census_extra at each flush and at the end; the capture's frame line, joined by seq, gives the thread's
// whole CPU time per frame.
#define TIME_THREADS 1024
#define TIME_ROW_CAP 4000000
typedef struct { uint64_t seq; uint32_t id, handle; uint64_t ns[RT_TIMER_CLASSES], calls[RT_TIMER_CLASSES]; } TimeRow;
typedef struct { uint64_t seq, calls, ns; uint32_t slot; } FnRow;
static TimeRow *g_trow;
static FnRow *g_frow;
static size_t g_ntrow, g_captrow, g_nfrow, g_capfrow;
static RtTimerThread g_tprev[TIME_THREADS];
static int g_thread_cap_said;
static uint64_t *g_fprev;   // calls and ns of each slot at the last window: 2 * (n + 1)
static int g_fprev_n, g_time_full_said;

static void *grow(void *p, size_t *cap, size_t need, size_t elem) {
    if (need <= *cap) return p;
    size_t c = *cap ? 2 * *cap : 4096;
    void *q = realloc(p, c * elem);
    if (!q) rt_fatal("nvn census: out of memory for the timer rows");
    *cap = c;
    return q;
}

static void timer_present(uint64_t seq) {
    RtTimerThread t[256];
    int n = rt_timer_threads(t, 256);
    int over = n > 256;
    if (over) n = 256;
    plat_mutex_lock(&g_lock);
    for (int i = 0; i < n; i++) {
        uint32_t id = t[i].id;
        if (id >= TIME_THREADS) over = 1;
        if (id >= TIME_THREADS) continue;
        RtTimerThread *p = &g_tprev[id];
        uint64_t dn = t[i].calls[RT_TIMER_NVN] - p->calls[RT_TIMER_NVN], ds = t[i].calls[RT_TIMER_SUBMIT] - p->calls[RT_TIMER_SUBMIT];
        if ((dn || ds) && g_ntrow < TIME_ROW_CAP) {
            g_trow = grow(g_trow, &g_captrow, g_ntrow + 1, sizeof *g_trow);
            TimeRow *r = &g_trow[g_ntrow++];
            r->seq = seq, r->id = id, r->handle = t[i].handle;
            for (int k = 0; k < RT_TIMER_CLASSES; k++) r->ns[k] = t[i].ns[k] - p->ns[k], r->calls[k] = t[i].calls[k] - p->calls[k];
        } else if ((dn || ds) && !g_time_full_said) {
            g_time_full_said = 1;
            rt_log("nvn census: %d timer rows kept; later frames are not recorded", TIME_ROW_CAP);
        }
        *p = t[i];
    }
    if (over && !g_thread_cap_said) {
        g_thread_cap_said = 1;
        rt_log("nvn census: the timer rows cover 256 threads with an id under %d; threads beyond that are not recorded", TIME_THREADS);
    }
    if (seq == 1 || seq % WINDOW == 0) {
        int ns = rt_hostfn_count();
        uint64_t *calls = malloc(2 * (size_t)(ns + 1) * sizeof *calls);
        if (calls) {
            uint64_t *tot = calls + (ns + 1);
            rt_timer_slots(calls, tot, ns);
            if (!g_fprev) g_fprev = calloc(2 * (size_t)(ns + 1), sizeof *g_fprev), g_fprev_n = ns;
            if (g_fprev && g_fprev_n == ns)
                for (int i = 0; i <= ns; i++) {
                    uint64_t dc = calls[i] - g_fprev[i], dt = tot[i] - g_fprev[ns + 1 + i];
                    g_fprev[i] = calls[i], g_fprev[ns + 1 + i] = tot[i];
                    if ((!dc && !dt) || g_nfrow >= TIME_ROW_CAP) continue;
                    g_frow = grow(g_frow, &g_capfrow, g_nfrow + 1, sizeof *g_frow);
                    g_frow[g_nfrow++] = (FnRow){seq, dc, dt, (uint32_t)i};
                }
            free(calls);
        }
    }
    plat_mutex_unlock(&g_lock);
}

static const char *fn_name(uint32_t slot, int *cls) {
    int n = rt_hostfn_count();
    if ((int)slot >= n) return *cls = RT_TIMER_GUEST, "<guest callback>";
    const char *name = rt_hostfn_name(RT_HOSTFN_BASE + 16ull * slot);
    if (!name) name = "?";
    *cls = !strcmp(name, "nvnQueueSubmitCommands") ? RT_TIMER_SUBMIT : !strncmp(name, "nvn", 3) ? RT_TIMER_NVN : RT_TIMER_OTHER;
    return name;
}

static void write_time(FILE *f) {
    typedef unsigned long long ull;
    fprintf(f, "#census-mode OOT2_CENSUS=1 OOT2_CENSUS_TIMER=1: not a default run\n");
    fprintf(f, "#timer clock plat_thread_cpu_ns; nested-call cost subtracted from the parent %llu ns; cost inside a call's own "
               "reading subtracted from every call %llu ns; calls past the depth kept (passed untimed) %llu\n",
            (ull)rt_timer_overhead_ns(), (ull)rt_timer_floor_ns(), (ull)rt_timer_overflows());
    fprintf(f, "#timer a call's time is its own: thread CPU time minus the host calls and guest callbacks nested in it, so a "
               "blocked wait is not charged. Fibres are not followed: SwitchToFiber returns on another host stack, so a "
               "switch's frame takes the time of the fibre it switched to; those names are not NVN2 and sit in the other class.\n");
    fprintf(f, "#frame columns: seq thread_id thread_handle nvn_ns submit_ns other_ns guest_ns nvn_calls submit_calls "
               "other_calls guest_calls (a thread that made a timed NVN2 call since the last present; join by seq with the "
               "capture's frame line)\n");
    fprintf(f, "#fn columns: window_end_seq function class calls exclusive_ns (over the window; class nvn, submit, other, guest)\n");
    static const char *const cls_name[RT_TIMER_CLASSES] = {"nvn", "submit", "other", "guest"};
    plat_mutex_lock(&g_lock);
    for (size_t i = 0; i < g_ntrow; i++) {
        const TimeRow *r = &g_trow[i];
        fprintf(f, "frame\t%llu\t%u\t%u\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n", (ull)r->seq, r->id, r->handle,
                (ull)r->ns[0], (ull)r->ns[1], (ull)r->ns[2], (ull)r->ns[3], (ull)r->calls[0], (ull)r->calls[1],
                (ull)r->calls[2], (ull)r->calls[3]);
    }
    for (size_t i = 0; i < g_nfrow; i++) {
        int cls;
        const char *name = fn_name(g_frow[i].slot, &cls);
        fprintf(f, "fn\t%llu\t%s\t%s\t%llu\t%llu\n", (ull)g_frow[i].seq, name, cls_name[cls], (ull)g_frow[i].calls,
                (ull)g_frow[i].ns);
    }
    plat_mutex_unlock(&g_lock);
}

// The other census-mode files, written beside census-nvn.tsv at each flush and at the end (census.c): the timer's rows
// (census-time.tsv, only with the timer on) and the D11 probe's hits (census-probe.tsv, only with the probe on).
void nvn_census_extra(const char *folder) {
    char path[4200];
    if (rt_timer_on()) {
        snprintf(path, sizeof path, "%s/census-time.tsv", folder);
        FILE *f = fopen(path, "wb");
        if (f) {
            write_time(f);
            fclose(f);
        }
    }
    if (nvn_probe_on()) {
        snprintf(path, sizeof path, "%s/census-probe.tsv", folder);
        FILE *f = fopen(path, "wb");
        if (f) {
            nvn_probe_write(f);
            fclose(f);
        }
    }
}

// Called by nvnQueuePresentTexture after present number seq (1 on) went to the back end.
void nvn_census_present(uint64_t seq) {
    uint64_t now = plat_now_ns();
    if (seq == 1) g_first_ns = now;
    if (seq == 1 || seq % WINDOW == 0) window_line(seq, seq == 1 ? 0.0 : (double)(now - g_first_ns) / 1e9);
    rc_present(seq);   // the replay's counters (OOT2_REPLAY_COUNTERS): a line group when their window is full
    if (!rt_census_on()) return;
    if (seq == 1 || seq % WINDOW == 0) take(seq);
    if (rt_timer_on()) timer_present(seq);
    if (seq % FLUSH_EVERY == 0) rt_census_flush();
    uint64_t limit = rt_census_frames();
    if (limit && seq >= limit) {
        char line[96];
        snprintf(line, sizeof line, "[census mode] ended at its frame limit, present %llu", (unsigned long long)seq);
        rt_log("ended at its frame limit, present %llu", (unsigned long long)seq);
        rt_census_end(RT_EXIT_CENSUS, line);
    }
}

// What the front end counted, as rows of the census (census.c writes them as kind "nvn-stat"): numbers only. emit(ctx, key,
// count, why) is called once per key: recordings, callbacks, waitevents, waitclasses, presents.
void nvn_census_stats(NvnStatRow emit, void *ctx) {
    typedef unsigned long long ull;
    NvnCmdCounts k;
    nvn_cmd_counts(&k);
    uint64_t unsat = 0, waits = gfx_null_wait_events(&unsat);
    char why[640];
    snprintf(why, sizeof why, "inline %llu, slow %llu, ended with a handle %llu, ended empty %llu", (ull)k.begin_inline,
             (ull)k.begin_slow, (ull)k.end_handle, (ull)k.end_empty);
    emit(ctx, "recordings", k.begin_inline + k.begin_slow, why);
    snprintf(why, sizeof why, "memory callbacks: control %llu, command %llu; recordings begun over left-out records %llu",
             (ull)k.callback_control, (ull)k.callback_command, (ull)k.left_out);
    emit(ctx, "callbacks", k.callback_control + k.callback_command, why);
    snprintf(why, sizeof why, "WaitEvents met %llu, unsatisfied at their place %llu", (ull)waits, (ull)unsat);
    emit(ctx, "waitevents", waits, why);
    GfxNullWaitClasses w;
    gfx_null_wait_classes(&w);
    // A wait behind a parked wait of its own queue is judged after that release: it has a class of its own, so that
    // the counts "before" are the ones a signal made before the wait's place, not the ones the release drain reached.
    snprintf(why, sizeof why, "by a GPU signal in order %llu, by a CPU signal before %llu, already satisfied %llu, "
             "reached after a release %llu, unsatisfied %llu (later by GPU %llu, later by CPU %llu, not satisfied by the "
             "end %llu); GPU signals in order: of its own queue %llu, of another %llu", (ull)w.by_gpu, (ull)w.by_cpu,
             (ull)w.initial, (ull)w.after_release, (ull)w.unsatisfied, (ull)w.later_gpu, (ull)w.later_cpu,
             (ull)w.pending, (ull)w.by_gpu_same, (ull)w.by_gpu_other);
    emit(ctx, "waitclasses", w.by_gpu + w.by_cpu + w.initial + w.after_release + w.unsatisfied, why);
    // WaitEvent per call site, the same classes: one row per site, then the bucket of the sites past the
    // table's 64, so the rows sum to the row above. The site is the guest function the wait was recorded from.
    {
        static NvnWaitSite sites[64];
        NvnWaitSite other;
        int ns = gfx_null_wait_sites(sites, 64, &other);
        for (int i = 0; i <= ns; i++) {
            const NvnWaitSite *s = i < ns ? &sites[i] : &other;
            if (i == ns && !s->c.by_gpu && !s->c.by_cpu && !s->c.initial && !s->c.after_release && !s->c.unsatisfied)
                break;
            char key[160], at[120];
            if (i < ns) snprintf(key, sizeof key, "waitsite %s", rt_guest_name(s->caller, at, sizeof at));
            else snprintf(key, sizeof key, "waitsite other sites");
            snprintf(why, sizeof why, "by a GPU signal in order %llu, by a CPU signal before %llu, already satisfied %llu, "
                     "reached after a release %llu, unsatisfied %llu (later by GPU %llu, later by CPU %llu, not satisfied "
                     "by the end %llu)", (ull)s->c.by_gpu, (ull)s->c.by_cpu, (ull)s->c.initial, (ull)s->c.after_release,
                     (ull)s->c.unsatisfied, (ull)s->c.later_gpu, (ull)s->c.later_cpu, (ull)s->c.pending);
            emit(ctx, key, s->c.by_gpu + s->c.by_cpu + s->c.initial + s->c.after_release + s->c.unsatisfied, why);
        }
        // The same waits by the queue pair: a GPU signal's queue against the waiting queue's, and a CPU signal's release.
        snprintf(why, sizeof why, "GPU signal in order: of the waiting queue itself %llu, of another queue %llu; GPU signal of "
                 "another queue after the wait parked its queue %llu; CPU signal: before %llu, after the wait parked its "
                 "queue %llu", (ull)w.by_gpu_same, (ull)w.by_gpu_other, (ull)w.later_gpu, (ull)w.by_cpu, (ull)w.later_cpu);
        emit(ctx, "waitqueues", w.by_gpu_same + w.by_gpu_other + w.later_gpu, why);
        // How long a held queue waited: the bound (a stop) is calibrated on this.
        snprintf(why, sizeof why, "queues held at a wait %llu; longest hold %llu submissions made meanwhile (a stop at %u, or "
                 "%llu s for a wait with no timeout); held now %llu", (ull)w.unsatisfied, (ull)gfx_null_longest_hold(),
                 (unsigned)WO_BOUND_SUBMITS, (ull)(WO_STALL_NS / 1000000000ull), (ull)w.pending);
        emit(ctx, "waitholds", w.unsatisfied, why);
    }
    NvnPoolStats ps;
    nvn_pool_stats(&ps);
    snprintf(why, sizeof why, "memory pools made %llu, live %llu (most at once %llu), %llu bytes live (most %llu); made with "
             "GPU_NO_ACCESS %llu; GPU address window in use %llu bytes (most %llu of %llu)", (ull)ps.all.created,
             (ull)ps.all.live, (ull)ps.all.peak, (ull)ps.all.bytes, (ull)ps.all.peak_bytes, (ull)ps.gpu_no_access,
             (ull)ps.window_bytes, (ull)ps.window_peak, (ull)GFX_GPU_WINDOW_SIZE);
    emit(ctx, "pools", ps.all.created, why);
    static const char *const names[NVN_POOL_CLASSES] = {"pools-cpu", "pools-cpu-no-access", "pools-physical",
                                                        "pools-virtual"};
    for (int i = 0; i < NVN_POOL_CLASSES; i++) {
        const NvnPoolClass *k = &ps.cls[i];
        snprintf(why, sizeof why, "made %llu, live %llu (most at once %llu), %llu bytes live (most %llu)", (ull)k->created,
                 (ull)k->live, (ull)k->peak, (ull)k->bytes, (ull)k->peak_bytes);
        emit(ctx, names[i], k->created, why);
    }
    if (nvn_probe_on()) {   // census mode with census_probe: the D11 probe (nvn_cmd.c)
        NvnProbeStats p;
        nvn_probe_stats(&p);
        char hits[96];
        if (p.hits) snprintf(hits, sizeof hits, "%llu hits", (ull)p.hits);
        else if (p.unchecked_bytes) snprintf(hits, sizeof hits, "not seen over %llu recordings in the ranges checked (%llu bytes "
                                             "were not checked)", (ull)p.recordings, (ull)p.unchecked_bytes);
        else snprintf(hits, sizeof hits, "not seen over %llu recordings", (ull)p.recordings);
        snprintf(why, sizeof why, "D11 probe: %llu command buffers, %llu ranges added (%llu CPU_NO_ACCESS, %llu unreadable, %llu "
                 "over the per-buffer cap), %llu bytes hashed at add, %llu bytes checked, %llu bytes of committed pages read, "
                 "%llu bytes of uncommitted pages not read, %llu bytes unchecked; %s",
                 (ull)p.buffers, (ull)p.ranges, (ull)p.no_access, (ull)p.unreadable, (ull)p.over_cap, (ull)p.bytes_hashed,
                 (ull)p.bytes_checked, (ull)p.bytes_read, (ull)p.bytes_zero, (ull)p.unchecked_bytes, hits);
        emit(ctx, "probe", p.hits, why);
    }
    if (ledger_collected()) {   // a collection run (settings.h collect_variants): only then, so a default census is as it was
        snprintf(why, sizeof why, "NOT A PROOF RUN: %llu draws were skipped, not drawn, for collection (collect_variants), "
                 "over %llu distinct stage and answer pairs; their ledger rows are written", (ull)ledger_collected(),
                 (ull)ledger_collected_kinds());
        emit(ctx, "collected-draws", ledger_collected(), why);
    }
    if (rt_setting(RT_SET_test_movie_blit)) {   // a test switch that changes the picture (settings.h): the row is made whenever it is on
        uint64_t skipped, blits;
        nvn_census_movie_blit(&skipped, &blits);
        snprintf(why, sizeof why, "NOT A PROOF RUN: test_movie_blit is on; the movie chain's dispatches not run %llu, replaced by "
                 "a resample %llu", (ull)skipped, (ull)blits);
        emit(ctx, "test-movie-blit", skipped + blits, why);
    }
    int nsources = __atomic_load_n(&g_nsources, __ATOMIC_ACQUIRE);   // a back end's rows
    for (int i = 0; i < nsources; i++) g_sources[i](emit, ctx);
    emit(ctx, "presents", nvn_present_count(), "nvnQueuePresentTexture calls that reached the back end");
}

void nvn_census_add_source(NvnStatSource fn) {
    static PlatMutex lock = PLAT_MUTEX_INIT;
    plat_mutex_lock(&lock);
    int n = g_nsources, known = 0;
    for (int i = 0; i < n; i++) known |= g_sources[i] == fn;
    if (!known && n < NVN_CENSUS_SOURCES) {
        g_sources[n] = fn;
        __atomic_store_n(&g_nsources, n + 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&lock);
}

// ---- the movie chain's stand-in (settings.h test_movie_blit): what the Vulkan replay did not run or replaced
static uint64_t g_movie_skipped, g_movie_blits;   // atomic

void nvn_census_note_movie_blit(uint64_t skipped, uint64_t blits) {
    if (skipped) __atomic_fetch_add(&g_movie_skipped, skipped, __ATOMIC_RELAXED);
    if (blits) __atomic_fetch_add(&g_movie_blits, blits, __ATOMIC_RELAXED);
}

void nvn_census_movie_blit(uint64_t *skipped, uint64_t *blits) {
    *skipped = __atomic_load_n(&g_movie_skipped, __ATOMIC_RELAXED);
    *blits = __atomic_load_n(&g_movie_blits, __ATOMIC_RELAXED);
}

static void rows(FILE *f, const char *phase, uint64_t frames, const uint64_t *now, const uint64_t *before, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t d = now[i] - (before ? before[i] : 0);
        if (!d) continue;
        const char *name = rt_hostfn_name(RT_HOSTFN_BASE + 16ull * (uint64_t)i);
        if (!name || strncmp(name, "nvn", 3)) continue;
        if (frames) fprintf(f, "%s\t%llu\t\t%s\t%.3f\t%llu\n", phase, (unsigned long long)frames, name,
                            (double)d / (double)frames, (unsigned long long)d);
        else fprintf(f, "%s\t0\t\t%s\t%llu\t%llu\n", phase, name, (unsigned long long)d, (unsigned long long)d);
    }
}

// The records the null back end replayed, by op, as rows "replay:<OP>" of the same table: the second draw count.
static void op_rows(FILE *f, const char *phase, uint64_t frames, const uint64_t *now, const uint64_t *before) {
    for (int i = 0; i < OP_COUNT_; i++) {
        uint64_t d = now[i] - before[i];
        if (!d || !frames) continue;
        fprintf(f, "%s\t%llu\t\treplay:%s\t%.3f\t%llu\n", phase, (unsigned long long)frames, nvn_op_name((uint32_t)i),
                (double)d / (double)frames, (unsigned long long)d);
    }
}

// The census-mode file: window comment lines first (#window name presents a-b seconds), then the table.
void nvn_census_write(FILE *f) {
    int n = rt_hostfn_count();
    uint64_t *cur = malloc((size_t)n * sizeof *cur);
    if (!cur) return;
    rt_hostfn_snapshot(cur, n);
    uint64_t seq = nvn_present_count(), now_ns = plat_now_ns();
    plat_mutex_lock(&g_lock);
    fprintf(f, "#census-mode OOT2_CENSUS=1: not a default run\n#presents %llu\n", (unsigned long long)seq);
    for (int k = 1; k < g_nsnap; k++)
        fprintf(f, "#window w%03d presents %llu-%llu seconds %.3f\n", k, (unsigned long long)g_snap[k - 1].present,
                (unsigned long long)g_snap[k].present, (double)(g_snap[k].ns - g_snap[k - 1].ns) / 1e9);
    if (g_nsnap && seq > g_snap[g_nsnap - 1].present)
        fprintf(f, "#window tail presents %llu-%llu seconds %.3f\n", (unsigned long long)g_snap[g_nsnap - 1].present,
                (unsigned long long)seq, (double)(now_ns - g_snap[g_nsnap - 1].ns) / 1e9);
    // the pools and the process's private memory at each snapshot (point 3), and where the title line fell
    for (int k = 0; k < g_nsnap; k++) {
        const Snap *s = &g_snap[k];
        fprintf(f, "#pools %s%03d presents %llu live %llu most %llu bytes %llu most_bytes %llu window_bytes %llu window_most %llu "
                   "private_bytes %llu (0: not measured at this snapshot)\n", k ? "w" : "boot", k, (unsigned long long)s->present, (unsigned long long)s->pools.all.live,
                (unsigned long long)s->pools.all.peak, (unsigned long long)s->pools.all.bytes,
                (unsigned long long)s->pools.all.peak_bytes, (unsigned long long)s->pools.window_bytes,
                (unsigned long long)s->pools.window_peak, (unsigned long long)s->priv);
    }
    if (g_title_present) fprintf(f, "#title the title line was logged at present %llu\n", (unsigned long long)g_title_present);
    fprintf(f, "phase\tframes\tslot\tfunction\tcalls_per_frame\tcalls\n");
    if (!g_nsnap) {
        rows(f, "boot", 0, cur, NULL, n);
    } else {
        rows(f, "boot", 0, g_snap[0].calls, NULL, g_nslots < n ? g_nslots : n);
        char name[16];
        for (int k = 1; k < g_nsnap; k++) {
            snprintf(name, sizeof name, "w%03d", k);
            rows(f, name, g_snap[k].present - g_snap[k - 1].present, g_snap[k].calls, g_snap[k - 1].calls, g_nslots);
            op_rows(f, name, g_snap[k].present - g_snap[k - 1].present, g_snap[k].ops, g_snap[k - 1].ops);
        }
        const Snap *last = &g_snap[g_nsnap - 1];
        if (seq > last->present) rows(f, "tail", seq - last->present, cur, last->calls, g_nslots);
        if (seq > 1) rows(f, "all", seq - 1, cur, g_snap[0].calls, g_nslots);
        if (seq > 1) {
            uint64_t now_ops[OP_COUNT_];
            gfx_null_op_counts(now_ops);
            op_rows(f, "all", seq - 1, now_ops, g_snap[0].ops);
        }
    }
    plat_mutex_unlock(&g_lock);
    free(cur);
}
