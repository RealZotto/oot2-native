// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The null graphics back end, and the choice of back end (gfx.h). No GPU: pools always initialise, presents are only
// counted and texel reads return zeros. It is the back end of the front end's CPU-side tests, and of a run that needs
// no picture.
//
// A submission is replayed when it is made and completed after, in submission order whichever queue made it: at once,
// or, while completion is held (gfx_null_hold), when gfx_null_complete or the end of the hold completes it. Replay
// walks every record of every recording, checks that each is whole (OP_BEGIN first, sizes that stay 8-byte aligned,
// known ops, no call or jump, OP_END last), decides conditional rendering, shows the record to the op log a test
// installs, and copies each record with a CPU-visible result, since the recording is not the back end's once submit
// returns. Completion writes those results in op-stream order (event signals, counter reports, buffer copies and
// clears, uniform updates), checks each WaitEvent, then publishes the fence value: a fence value is done only once
// everything up to it is written.
//
// Counter reports are {counter, timestamp}: the timestamp is the guest's system tick (31.25 MHz, rt_ticks) read from
// the host clock when the report completes; the counter is 0 for every type, samples passed included, since nothing is
// measured here.
//
// Conditional rendering is decided at replay, as a back end that replays ahead of the GPU must: the counter words at
// addr and addr + 16 are read from memory, except where the last report to that address is not complete (in this
// submission or a held one). Such a report counts as 0 when it is a timestamp or any other counter, and as unknown when
// it is samples passed: an unknown condition renders, as does one whose words the CPU cannot read (a draw costs at most
// the work the condition would have saved). RenderEnable and the condition hold across submissions, as on one GPU
// channel.
//
// WaitEvent is not waited for: completion runs in submission order, so a wait that the work before it satisfies costs
// nothing. Completion checks each one against memory at its place in that order. The first wait met at each call site
// is logged with its answer, and so is a site's first unsatisfied wait: on the console that wait would have held its
// queue until another queue or the CPU signalled, which work in submission order cannot honour (architecture section 5).
// Every queue's work is merged into that one order and a wait or a signal does not say which queue it came from here, so
// a wait that a later submission satisfies may be one queue waiting for another, which runs concurrently on the console.
//
// Adapted from TotK Recomp runtime/gfx_null.c (MIT, Copyright (c) 2026 Valiantsin Dzerakh; licenses/totk-MIT.txt), as
// DKB Recomp carries it at 9d8fa02 and oot-pc-recomp at 8e09aaa (GPL-3.0-or-later): the back-end table, the fence
// counter, the zeroed texel reads, and the CPU-visible results of op streams (oot-pc-recomp's null_replay: event WRITE
// and ADD, counters {0, timestamp}, buffer copies, fills and uniform updates written in stream order). The conditional
// rendering rule is DKB Recomp's runtime/vk_exec_xfer.c at 9d8fa02 (counter_word, cond_enabled: the newest pending report
// to an address decides, a timestamp counts as 0, samples passed as unknown, unreadable words render). Changed here:
// submissions are serialised, so that a fence value is never done before an earlier one even when two queues submit at
// once; replay and completion are two steps and completion can be held; results are copied at replay; a result written
// to an address in no memory pool stops by name (the console's GPU would fault) and one written to the GPU address
// window is dropped (no host memory holds it); WaitEvent is checked and logged; the statistics and the environment
// switch are gone, and gfx_set_backend installs a back end instead.
#include <stdlib.h>
#include <string.h>

#include "nvn_api.h"
#include "oot2/ledger.h"
#include "plat.h"

typedef unsigned long long ull;

static PlatMutex g_lock = PLAT_MUTEX_INIT;   // replay and completion, one submission at a time
static PlatCond g_done_cond = PLAT_COND_INIT;
static uint64_t g_submitted;                 // the newest fence value handed out (g_lock)
static uint64_t g_done;                      // the newest fence value done (written under g_lock, read anywhere)
static uint64_t g_presents, g_records;
static uint64_t g_op_counts[OP_COUNT_];      // records replayed, by op (the second draw count)

static int null_pool_init(pool_t *p) { (void)p; return 1; }
static void null_pool_fini(pool_t *p) { (void)p; }
static int null_pool_map_virtual(pool_t *v, int n, const void *requests) { (void)v, (void)n, (void)requests; return 1; }
static void null_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    (void)pool, (void)id, (void)t, (void)v;
}
static void null_sampler_register(const dpool_ref *pool, int id, const samp_par *s) { (void)pool, (void)id, (void)s; }
static void null_texture_release(const tex_t *t) { (void)t; }
static void null_program_release(const prog_t *p) { (void)p; }

static GfxNullOpLog g_log;
static void *g_log_ctx;

void gfx_null_log_ops(GfxNullOpLog fn, void *ctx) { g_log = fn, g_log_ctx = ctx; }
uint64_t gfx_null_records(void) { return __atomic_load_n(&g_records, __ATOMIC_RELAXED); }
void gfx_null_op_counts(uint64_t *out) {
    for (int i = 0; i < OP_COUNT_; i++) out[i] = __atomic_load_n(&g_op_counts[i], __ATOMIC_RELAXED);
}

// ---------- what a submission leaves to completion: copies of its records with CPU-visible results, in stream order
typedef struct results {
    struct results *next;
    uint64_t fence;
    uint8_t *buf;
    size_t used, size;
} results;
static results g_now;                              // the submission being replayed; its buffer is kept for the next
static results *g_held, **g_held_tail = &g_held;   // replayed, not complete yet, oldest first
static int g_hold;

static void keep(const op_hdr *r) {
    if (g_now.size - g_now.used < r->size) {
        size_t size = g_now.size ? g_now.size : 4096;
        while (size - g_now.used < r->size) size *= 2;
        uint8_t *b = realloc(g_now.buf, size);
        if (!b) rt_fatal("gfx: null back end: %zu bytes of results", size);
        g_now.buf = b, g_now.size = size;
    }
    memcpy(g_now.buf + g_now.used, r, r->size);
    g_now.used += r->size;
}

// ---------- conditional rendering, at replay
#define PENDING 256   // reports remembered, newest last: the newest to an address decides while it is not complete
static struct { uint64_t addr, fence; int samples; } g_pending[PENDING];
static unsigned g_pending_head;
static int g_render = 1;                       // RenderEnable and the condition, as replay leaves them

int gfx_null_render_enabled(void) { return g_render; }

// 1: *v read; 2: a samples-passed report not complete (unknown); 0: the CPU cannot read the word.
static int counter_word(uint64_t addr, uint64_t *v) {
    uint64_t done = __atomic_load_n(&g_done, __ATOMIC_RELAXED);
    for (unsigned k = 1; k <= PENDING && k <= g_pending_head; k++) {
        unsigned i = (g_pending_head - k) % PENDING;
        if (g_pending[i].addr == addr && g_pending[i].fence > done) return *v = 0, g_pending[i].samples ? 2 : 1;
    }
    if (gfx_gpu_only(addr) || !nvn_pool_find(addr, sizeof *v)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    return 1;
}

static int condition(const p_render_cond *p) {
    uint64_t x, y;
    int rx = counter_word(p->addr, &x), ry = counter_word(p->addr + 16, &y);
    if (!rx || !ry) {
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("nvn: null back end: the conditional rendering counters at %#llx are not readable by the CPU: "
                   "rendering", (ull)p->addr);
        return 1;
    }
    if (rx == 2 || ry == 2) return 1;
    return p->mode == 0 ? x == y : x != y;
}

static void replay(uint64_t fence, const op_hdr *r) {
    const void *a = r + 1;
    switch (r->op) {
    case OP_RENDER_ENABLE: g_render = ((const p_flag *)a)->value != 0; break;
    case OP_RENDER_COND: g_render = condition(a); break;
    case OP_REPORT_COUNTER: {
        const p_report_counter *p = a;
        unsigned i = g_pending_head++ % PENDING;
        g_pending[i].addr = p->addr, g_pending[i].fence = fence;
        g_pending[i].samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED;
        keep(r);
        break;
    }
    case OP_SIGNAL_EVENT: case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_UPDATE_UB: case OP_WAIT_EVENT: keep(r); break;
    default: break;
    }
}

static LedgerTrack g_track = {.depth_fmt = -1, .samples = 1};                  // the bound state across records and submissions (g_lock)

static void walk(uint64_t fence, int index, uint64_t handle) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)handle;
    for (size_t at = 0;;) {
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < sizeof h || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL ||
            (at == 0) != (h.op == OP_BEGIN))
            rt_fatal("gfx: null back end: recording %d of submission %llu is not whole at +%zu (op %u, %u bytes)", index,
                     (ull)fence, at, h.op, h.size);
        __atomic_fetch_add(&g_records, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(&g_op_counts[h.op], 1, __ATOMIC_RELAXED);
        const op_hdr *r = (const op_hdr *)(const void *)(p + at);
        replay(fence, r);
        ledger_track_op(&g_track, r);
        if (texheap_on()) texheap_op(r);
        if (g_log) g_log(fence, index, r, g_log_ctx);
        at += h.size;
        if (h.op == OP_END) return;
    }
}

// ---------- completion, in submission order (g_lock held)
// Host memory behind [addr, addr + size), or NULL when the range is in the GPU address window, which no host memory
// holds. A range in no pool is where the console's GPU would fault.
static void *cpu_addr(uint64_t addr, uint64_t size, const char *what) {
    if (gfx_gpu_only(addr)) return NULL;
    if (!nvn_pool_find(addr, size))
        rt_fatal("gfx: null back end: %s at %#llx (%llu bytes) is in no memory pool the GPU can reach", what, (ull)addr,
                 (ull)size);
    return (void *)(uintptr_t)addr;
}

// WaitEvent call sites met, with what was logged for each and what each one's waits were satisfied by: the
// same classes as the totals, kept per site, and one bucket for the sites past the table, so that the sites' counts always
// sum to the totals. The site of a wait is the guest return address in its record, which CallCommands copies unchanged.
#define WAIT_SITES 64
static struct { uint64_t caller; int unsatisfied_logged; GfxNullWaitClasses c; } g_sites[WAIT_SITES];
static GfxNullWaitClasses g_other_sites;
static int g_nsites;
static uint64_t g_waits, g_waits_unsatisfied;

uint64_t gfx_null_wait_events(uint64_t *unsatisfied) {
    plat_mutex_lock(&g_lock);
    uint64_t n = g_waits;
    if (unsatisfied) *unsatisfied = g_waits_unsatisfied;
    plat_mutex_unlock(&g_lock);
    return n;
}

// Who wrote each event word last, as this back end saw it: a GPU signal at completion (in submission order) or the CPU
// (nvnEventSignal, reported through guest_written). A WaitEvent is judged against the word at completion, which in a run
// that completes at submit cannot tell a harmless order from a hazard; the classes below can: satisfied by the work before
// it, by a CPU signal made before, or by neither (initial value), and the unsatisfied waits kept in a pending list that
// each later completion checks again (satisfied later by a CPU signal is the case a back end replaying at submit cannot
// honour, since the console holds the queue until the CPU signals). The words are kept in a table that grows with the
// words signalled (a game makes hundreds of events), so no word is forgotten; the list of unsatisfied waits holds
// PENDING_WAITS, and a wait that finds it full is counted as dropped. Under g_lock.
enum { WRITER_NONE, WRITER_GPU, WRITER_CPU };
#define PENDING_WAITS 64
typedef struct { const volatile uint32_t *val; int writer; } WordRec;   // val NULL: a free slot
static WordRec *g_words;                                                // open addressing, a power of two long
static size_t g_word_cap, g_word_count;
static p_wait_event g_wait_pending[PENDING_WAITS];
static int g_wait_pending_site[PENDING_WAITS];   // the site of each pending wait: an index in g_sites, or -1 for the bucket
static int g_nwait_pending;
static GfxNullWaitClasses g_classes;

static size_t word_slot(const WordRec *t, size_t cap, const volatile uint32_t *val) {
    size_t i = (size_t)(((uint64_t)(uintptr_t)val >> 2) * 0x9E3779B97F4A7C15ull >> 32) & (cap - 1);
    while (t[i].val && t[i].val != val) i = (i + 1) & (cap - 1);
    return i;
}
static void word_note(const volatile uint32_t *val, int writer) {
    if (!g_word_cap || 2 * (g_word_count + 1) > g_word_cap) {   // at most half full
        size_t cap = g_word_cap ? 2 * g_word_cap : 1024;
        WordRec *t = calloc(cap, sizeof *t);
        if (!t) rt_fatal("gfx: null back end: the table of event words");
        for (size_t i = 0; i < g_word_cap; i++)
            if (g_words[i].val) t[word_slot(t, cap, g_words[i].val)] = g_words[i];
        free(g_words);
        g_words = t, g_word_cap = cap;
    }
    WordRec *r = &g_words[word_slot(g_words, g_word_cap, val)];
    if (!r->val) g_word_count++;
    r->val = val, r->writer = writer;
}
static int word_writer(const volatile uint32_t *val) {
    if (!g_word_cap) return WRITER_NONE;
    const WordRec *r = &g_words[word_slot(g_words, g_word_cap, val)];
    return r->val ? r->writer : WRITER_NONE;
}
static int wait_ok(const p_wait_event *p, uint32_t *w) {
    *w = __atomic_load_n(p->val, __ATOMIC_ACQUIRE);
    return p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? *w == p->value : (int32_t)(*w - p->value) >= 0;
}
void gfx_null_wait_classes(GfxNullWaitClasses *out) {
    plat_mutex_lock(&g_lock);
    *out = g_classes;
    out->pending = (uint64_t)g_nwait_pending;
    plat_mutex_unlock(&g_lock);
}
static GfxNullWaitClasses *site_classes(int s) { return s >= 0 ? &g_sites[s].c : &g_other_sites; }

// The sites in the order first met (at most WAIT_SITES, then the bucket for the rest), with their classes. out holds max
// sites; *other is the bucket of the sites past the table (caller 0). Returns the sites listed.
int gfx_null_wait_sites(NvnWaitSite *out, int max, NvnWaitSite *other) {
    plat_mutex_lock(&g_lock);
    int n = g_nsites < max ? g_nsites : max;
    for (int i = 0; i < n; i++) out[i] = (NvnWaitSite){g_sites[i].caller, g_sites[i].c};
    if (other) *other = (NvnWaitSite){0, g_other_sites};
    for (int i = 0; i < g_nwait_pending; i++) {   // not satisfied yet: counted at the site that waits
        int s = g_wait_pending_site[i];
        if (s >= 0 && s < n) out[s].c.pending++;
        else if (s < 0 && other) other->c.pending++;
    }
    plat_mutex_unlock(&g_lock);
    return n;
}

// Called after each completion: an unsatisfied wait whose word has reached its value since is satisfied later, by the
// writer that wrote the word last.
static void recheck_pending(void) {
    for (int i = 0; i < g_nwait_pending;) {
        uint32_t w;
        if (!wait_ok(&g_wait_pending[i], &w)) {
            i++;
            continue;
        }
        GfxNullWaitClasses *sc = site_classes(g_wait_pending_site[i]);
        if (word_writer(g_wait_pending[i].val) == WRITER_CPU) g_classes.later_cpu++, sc->later_cpu++;
        else g_classes.later_gpu++, sc->later_gpu++;
        g_wait_pending[i] = g_wait_pending[--g_nwait_pending];
        g_wait_pending_site[i] = g_wait_pending_site[g_nwait_pending];
    }
}

static void wait_event(const p_wait_event *p) {
    uint32_t w;
    int ok = wait_ok(p, &w);
    g_waits++;
    if (!ok) g_waits_unsatisfied++;
    int s = 0;
    while (s < g_nsites && g_sites[s].caller != p->caller) s++;
    int first = s == g_nsites;
    if (first && g_nsites < WAIT_SITES) g_sites[g_nsites++].caller = p->caller;
    else if (first) s = -1;   // past the table: such a site is not logged, and its waits go to the bucket
    GfxNullWaitClasses *sc = site_classes(s);
    if (ok) {
        int who = word_writer(p->val);
        if (who == WRITER_GPU) g_classes.by_gpu++, sc->by_gpu++;
        else if (who == WRITER_CPU) g_classes.by_cpu++, sc->by_cpu++;
        else g_classes.initial++, sc->initial++;
    } else {
        g_classes.unsatisfied++, sc->unsatisfied++;
        if (g_nwait_pending < PENDING_WAITS) {
            g_wait_pending_site[g_nwait_pending] = s;
            g_wait_pending[g_nwait_pending++] = *p;
        } else {
            g_classes.dropped++, sc->dropped++;
        }
    }
    char where[96];
    if (s >= 0 && (first || (!ok && !g_sites[s].unsatisfied_logged))) {
        int who = word_writer(p->val);
        rt_log("nvn: WaitEvent at %s on the event word %p (%s %u, the word holds %u) is %s at its place in submission "
               "order (the word was last written by %s)", rt_guest_name(p->caller, where, (int)sizeof where),
               (const void *)p->val, p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? "equal to" : "at least", p->value, w,
               ok ? "satisfied" : "NOT satisfied",
               who == WRITER_GPU ? "a GPU signal" : who == WRITER_CPU ? "a CPU signal" : "nothing yet");
        if (!ok) g_sites[s].unsatisfied_logged = 1;
    }
}

static void complete(const results *k) {
    for (size_t at = 0; at < k->used;) {
        const op_hdr *r = (const op_hdr *)(const void *)(k->buf + at);
        const void *a = r + 1;
        at += r->size;
        switch (r->op) {
        case OP_SIGNAL_EVENT: {
            const p_signal_event *p = a;
            if (p->mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(p->val, p->value, __ATOMIC_RELEASE);
            else if (p->mode == NVN_EVENT_SIGNAL_MODE_WRITE) __atomic_store_n(p->val, p->value, __ATOMIC_RELEASE);
            else rt_fatal("gfx: null back end: SignalEvent mode %u is not one of NVN's", p->mode);
            word_note(p->val, WRITER_GPU);
            break;
        }
        case OP_REPORT_COUNTER: {
            uint8_t *d = cpu_addr(((const p_report_counter *)a)->addr, sizeof(NVNcounterData), "a counter report");
            if (d) {
                uint64_t rec[2] = {0, rt_ticks()};
                memcpy(d, rec, sizeof rec);
            }
            break;
        }
        case OP_COPY_B2B: {
            const p_copy_b2b *p = a;
            void *s = cpu_addr(p->src, p->size, "a buffer copy's source"), *d = cpu_addr(p->dst, p->size, "a buffer copy");
            if (s && d) memmove(d, s, p->size);
            break;
        }
        case OP_CLEAR_BUF: {
            const p_clear_buf *p = a;
            uint8_t *d = cpu_addr(p->addr, p->size, "a buffer clear");
            for (uint64_t i = 0; d && i < p->size / 4; i++) memcpy(d + 4 * i, &p->value, 4);
            break;
        }
        case OP_UPDATE_UB: {
            const p_update_ub *p = a;
            void *d = cpu_addr(p->dst, p->size, "a uniform buffer update");
            if (d && p->size) memcpy(d, p->data, p->size);
            break;
        }
        case OP_WAIT_EVENT: wait_event(a); break;
        default: break;
        }
    }
    recheck_pending();
    if (texheap_on()) texheap_complete(k->fence);
    __atomic_store_n(&g_done, k->fence, __ATOMIC_RELEASE);
    plat_cond_broadcast(&g_done_cond);
}

static void complete_held(uint64_t value) {
    while (g_held && (!value || g_held->fence <= value)) {
        results *k = g_held;
        g_held = k->next;
        if (!g_held) g_held_tail = &g_held;
        complete(k);
        free(k->buf);
        free(k);
    }
}

// ---------- the back-end operations
// The lock spans the whole submission, so the results of submission k are written before fence k is published and
// before submission k + 1 starts; a held submission waits in the list until its turn.
static uint64_t null_submit(const uint64_t *handles, int n) {
    plat_mutex_lock(&g_lock);
    uint64_t fence = ++g_submitted;
    g_now.fence = fence, g_now.used = 0, g_now.next = NULL;
    if (texheap_on()) texheap_submit(fence);
    for (int i = 0; i < n; i++) walk(fence, i, handles[i]);
    if (g_hold || g_held) {
        results *k = malloc(sizeof *k);
        if (!k) rt_fatal("gfx: null back end: a held submission");
        *k = g_now;
        g_now.buf = NULL, g_now.size = 0;
        *g_held_tail = k, g_held_tail = &k->next;
    } else {
        complete(&g_now);
    }
    plat_mutex_unlock(&g_lock);
    return fence;
}

void gfx_null_hold(int on) {
    plat_mutex_lock(&g_lock);
    g_hold = on;
    if (!on) complete_held(0);
    plat_mutex_unlock(&g_lock);
}

uint64_t gfx_null_complete(uint64_t value) {
    plat_mutex_lock(&g_lock);
    complete_held(value);
    uint64_t done = __atomic_load_n(&g_done, __ATOMIC_RELAXED);
    plat_mutex_unlock(&g_lock);
    return done;
}

// A value no submit has handed out is not reached by waiting here: the answer is immediate.
static int null_fence_wait(uint64_t value, uint64_t timeout_ns) {
    if (value <= __atomic_load_n(&g_done, __ATOMIC_ACQUIRE)) return 1;
    plat_mutex_lock(&g_lock);
    uint64_t start = plat_now_ns();
    while (value <= g_submitted && __atomic_load_n(&g_done, __ATOMIC_RELAXED) < value) {
        if (timeout_ns == UINT64_MAX) {
            plat_cond_wait(&g_done_cond, &g_lock);
            continue;
        }
        uint64_t spent = plat_now_ns() - start;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_done_cond, &g_lock, timeout_ns - spent);
    }
    int ok = __atomic_load_n(&g_done, __ATOMIC_RELAXED) >= value;
    plat_mutex_unlock(&g_lock);
    return ok;
}

static uint64_t null_fence_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }

static void null_present(const tex_t *t, const int crop[4]) {
    (void)t, (void)crop;
    __atomic_fetch_add(&g_presents, 1, __ATOMIC_RELAXED);
}

// No present timing: every shown tick is unknown, and the answer is immediate.
static uint64_t null_present_shown(uint64_t seq) {
    (void)seq;
    return 0;
}

static void null_finish(void) {
    plat_mutex_lock(&g_lock);
    uint64_t target = g_submitted;
    while (__atomic_load_n(&g_done, __ATOMIC_RELAXED) < target) plat_cond_wait(&g_done_cond, &g_lock);
    plat_mutex_unlock(&g_lock);
}

static void null_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                             uint64_t img) {
    (void)t, (void)v;
    int h = r->height > 0 ? r->height : 1, d = r->depth > 0 ? r->depth : 1;
    for (int z = 0; z < d; z++) memset(dst + (uint64_t)z * img, 0, row * (uint64_t)h);
}

static void null_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src,
                              uint64_t row, uint64_t img) {
    (void)t, (void)v, (void)r, (void)src, (void)row, (void)img;
}

// nvnEventSignal reports the CPU's signal of an event word (4 bytes): the classifier notes it as the word's last writer
static void null_guest_written(uint64_t addr, uint64_t size) {
    if (size != 4) return;
    plat_mutex_lock(&g_lock);
    word_note((const volatile uint32_t *)(uintptr_t)addr, WRITER_CPU);
    plat_mutex_unlock(&g_lock);
}
static int null_guest_read(uint64_t addr, uint64_t size) { (void)addr, (void)size; return 0; }
static void null_cpu_synced(uint64_t fence) { (void)fence; }

const GfxBackend gfx_null_backend = {
    .name = "null",
    .pool_init = null_pool_init,
    .pool_fini = null_pool_fini,
    .pool_map_virtual = null_pool_map_virtual,
    .texture_register = null_texture_register,
    .sampler_register = null_sampler_register,
    .texture_release = null_texture_release,
    .program_release = null_program_release,
    .submit = null_submit,
    .fence_wait = null_fence_wait,
    .fence_done = null_fence_done,
    .present = null_present,
    .finish = null_finish,
    .read_texels = null_read_texels,
    .write_texels = null_write_texels,
    .guest_written = null_guest_written,
    .guest_read = null_guest_read,
    .cpu_synced = null_cpu_synced,
    .present_shown = null_present_shown,
};

uint64_t gfx_null_presents(void) { return __atomic_load_n(&g_presents, __ATOMIC_RELAXED); }

// ---- the choice of back end
static const GfxBackend *g_backend = &gfx_null_backend;

const GfxBackend *gfx_backend(void) { return g_backend; }

void gfx_set_backend(const GfxBackend *b) {
    if (rt_sealed()) rt_fatal("gfx: gfx_set_backend after rt_dispatch_init");
    if (!b || !b->name || !b->pool_init || !b->pool_fini || !b->pool_map_virtual || !b->texture_register ||
        !b->sampler_register || !b->texture_release || !b->program_release || !b->submit || !b->fence_wait ||
        !b->fence_done || !b->present || !b->finish || !b->read_texels || !b->write_texels || !b->guest_written ||
        !b->guest_read || !b->cpu_synced)
        rt_fatal("gfx: back end %s lacks an operation", b && b->name ? b->name : "(unnamed)");
    g_backend = b;
}
