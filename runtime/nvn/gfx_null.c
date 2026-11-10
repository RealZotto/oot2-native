// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The null graphics back end, and the choice of back end (gfx.h). No GPU: pools always initialise, presents are only
// counted and texel reads return zeros. It is the back end of the front end's CPU-side tests, and of a run that needs
// no picture.
//
// A submission is replayed when it is made, unless a WaitEvent holds its queue (below), and completed after: at once, or,
// while completion is held (gfx_null_hold), when gfx_null_complete or the end of the hold completes it. Queues complete
// independently, each in its own order, and the submissions that are whole at the same time complete in the order replay
// finished them (not in fence order: a held queue's earlier fence value is whole after a later one's, and its writes
// follow the signal that released it, as on the console). Replay checks that each recording is whole (OP_BEGIN first,
// sizes that stay 8-byte aligned, known ops, no call or jump, OP_END last), walks every record in the order the queues'
// waits allow, decides conditional rendering, shows the record to the op log a test installs, and copies each record
// with a CPU-visible result, since the recording is not the back end's once submit returns. Completion writes those
// results in op-stream order (event signals, counter reports, buffer copies and clears, uniform updates), then marks the
// fence value done (a submission that parks at a wait writes what it kept before the wait at once, since the replay
// has already let other queues act on it; the rest is written when it is whole): a fence value is done once its submission and the queue's earlier ones are written, whatever another
// queue is held at (fence_done is the highest value with every submission up to it done).
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
// WaitEvent is answered by reordering (oot2/nvn_order.h, gfx.h): the records of every queue pass through the order
// engine, which shows a queue's records in that queue's order and parks the queue at a wait that the replay so far does
// not satisfy, until a signal replayed on another queue, or the CPU's signal (guest_written of the word), satisfies it.
// Replay order can then differ from fence order. Results are copied per fence value at replay and written once the
// submission is whole (those kept before a wait that parks the queue are written at the park), and so is its fence value
// done; a queue's submissions are whole in the queue's order, and another
// queue's held wait is no condition (a thread that finishes or waits on one queue is never held by another's wait, gfx.h
// rule 3). A fence value behind a parked wait of its own queue is not done; a wait nothing releases is a named stop (after
// WO_BOUND_SUBMITS further submissions, or, for a wait with no timeout on a fence value of the queue or on the queue's
// finish, after WO_STALL_NS from the start of that wait). The first wait met at each call site is
// logged with its answer, and so is a site's first unsatisfied wait: the queue is parked there until another queue or
// the CPU signals. Presents are counted once every submission made before them is whole.
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
#include "oot2/nvn_order.h"
#include "plat.h"

typedef unsigned long long ull;

static PlatMutex g_lock = PLAT_MUTEX_INIT;   // replay and completion, one submission at a time
static _Thread_local int g_locked_here;
static PlatCond g_done_cond = PLAT_COND_INIT;
static uint64_t g_submitted;                 // the newest fence value handed out (g_lock)
static uint64_t g_done;                      // every fence value up to this is done (written under g_lock, read anywhere)
static uint64_t g_presents, g_records;
static uint64_t g_op_counts[OP_COUNT_];      // records replayed, by op (the second draw count)

static void null_lock(void) {
    plat_mutex_lock(&g_lock);
    g_locked_here = 1;
}

static void null_unlock(void) {
    g_locked_here = 0;
    plat_mutex_unlock(&g_lock);
}

// A replay callback or a fatal stop hook can ask for the census while this thread already holds the submission lock.
static int stats_lock(void) {
    if (g_locked_here) return 0;
    null_lock();
    return 1;
}

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
// One per fence value above g_done, oldest first. The records of one fence value can be replayed in pieces, with those of
// other fence values between them (a queue held at a wait), so each keeps its own buffer. `whole`: every record of the
// submission has been replayed. `done`: its results are written (it stays in the list until every value below is done).
typedef struct results {
    struct results *next;
    uint64_t fence, queue;
    uint8_t *buf;
    size_t used, size;
    int whole, done;
    uint64_t seq;   // 1, 2, 3 in the order replay finished the submissions (set when whole): the order their results are written in
} results;
static results *g_res, **g_res_tail = &g_res;
static int g_hold;
static uint64_t g_whole_seq;

static WoOrder g_wo;                         // the order the queues' records are replayed in (g_lock)
static uint64_t g_bound_submits, g_bound_ns; // the bounds of a wait nothing releases; 0: WO_BOUND_SUBMITS, WO_STALL_NS

static results *results_find(uint64_t fence) {
    for (results *k = g_res; k; k = k->next)
        if (k->fence == fence) return k;
    return NULL;
}

static results *results_for(uint64_t fence) {
    results *k = results_find(fence);
    if (!k) rt_fatal("gfx: null back end: a record of submission %llu, which was not made", (ull)fence);
    return k;
}

// Fence value `fence` is done (g_lock held): below the prefix, or its results are written. A value not in the list and
// above the prefix is one no submit has handed out.
static int fence_is_done(uint64_t fence) {
    if (fence <= g_done) return 1;
    const results *k = results_find(fence);
    return k && k->done;
}

static void keep(uint64_t fence, const op_hdr *r) {
    results *k = results_for(fence);
    if (k->size - k->used < r->size) {
        size_t size = k->size ? k->size : 4096;
        while (size - k->used < r->size) size *= 2;
        uint8_t *b = realloc(k->buf, size);
        if (!b) rt_fatal("gfx: null back end: %zu bytes of results", size);
        k->buf = b, k->size = size;
    }
    memcpy(k->buf + k->used, r, r->size);
    k->used += r->size;
}

// ---------- conditional rendering, at replay
#define PENDING 256   // reports remembered, newest last: the newest to an address decides while it is not complete
static struct { uint64_t addr, fence; int samples; } g_pending[PENDING];
static unsigned g_pending_head;
static int g_render = 1;                       // RenderEnable and the condition, as replay leaves them

int gfx_null_render_enabled(void) { return g_render; }

// 1: *v read; 2: a samples-passed report not complete (unknown); 0: the CPU cannot read the word.
static int counter_word(uint64_t addr, uint64_t *v) {
    for (unsigned k = 1; k <= PENDING && k <= g_pending_head; k++) {
        unsigned i = (g_pending_head - k) % PENDING;
        if (g_pending[i].addr == addr && !fence_is_done(g_pending[i].fence)) return *v = 0, g_pending[i].samples ? 2 : 1;
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
        keep(fence, r);
        break;
    }
    case OP_SIGNAL_EVENT: case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_UPDATE_UB: keep(fence, r); break;
    default: break;
    }
}

static LedgerTrack g_track = {.depth_fmt = -1, .samples = 1};                  // the bound state across records and submissions (g_lock)

// A recording is whole: OP_BEGIN first, sizes that stay 8-byte aligned, known ops, no call or jump, OP_END last. Checked
// when the submission is made, before any record of it is replayed.
static void check_whole(uint64_t fence, int index, uint64_t handle) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)handle;
    for (size_t at = 0;;) {
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < sizeof h || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL ||
            (at == 0) != (h.op == OP_BEGIN))
            rt_fatal("gfx: null back end: recording %d of submission %llu is not whole at +%zu (op %u, %u bytes)", index,
                     (ull)fence, at, h.op, h.size);
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

// WaitEvent call sites met, with what was logged for each and what each one's waits were answered by: the
// same classes as the totals, kept per site, and one bucket for the sites past the table, so that the sites' counts always
// sum to the totals. The site of a wait is the guest return address in its record, which CallCommands copies unchanged.
#define WAIT_SITES 64
static struct { uint64_t caller; int unsatisfied_logged; GfxNullWaitClasses c; } g_sites[WAIT_SITES];
static GfxNullWaitClasses g_other_sites;
static int g_nsites;
static uint64_t g_waits, g_waits_unsatisfied;

uint64_t gfx_null_wait_events(uint64_t *unsatisfied) {
    int took = stats_lock();
    uint64_t n = g_waits;
    if (unsatisfied) *unsatisfied = g_waits_unsatisfied;
    if (took) null_unlock();
    return n;
}

// The queues parked now, each at one wait: the queue and the site of its wait (an index in g_sites, or -1 for the bucket).
// A queue holds one wait at a time, so the list is as long as the queues are many. Under g_lock.
static struct { uint64_t queue; int site; } *g_parked;
static int g_nparked, g_cparked;
static GfxNullWaitClasses g_classes;

void gfx_null_wait_classes(GfxNullWaitClasses *out) {
    int took = stats_lock();
    *out = g_classes;
    out->pending = (uint64_t)g_nparked;
    if (took) null_unlock();
}
static GfxNullWaitClasses *site_classes(int s) { return s >= 0 ? &g_sites[s].c : &g_other_sites; }

// The sites in the order first met (at most WAIT_SITES, then the bucket for the rest), with their classes. out holds max
// sites; *other is the bucket of the sites past the table (caller 0). Returns the sites listed.
int gfx_null_wait_sites(NvnWaitSite *out, int max, NvnWaitSite *other) {
    int took = stats_lock();
    int n = g_nsites < max ? g_nsites : max;
    for (int i = 0; i < n; i++) out[i] = (NvnWaitSite){g_sites[i].caller, g_sites[i].c};
    if (other) *other = (NvnWaitSite){0, g_other_sites};
    for (int i = 0; i < g_nparked; i++) {   // not released yet: counted at the site that waits
        int s = g_parked[i].site;
        if (s >= 0 && s < n) out[s].c.pending++;
        else if (s < 0 && other) other->c.pending++;
    }
    if (took) null_unlock();
    return n;
}

void gfx_null_wait_bounds(uint64_t submits, uint64_t stall_ns) {
    null_lock();
    g_bound_submits = submits, g_bound_ns = stall_ns;
    null_unlock();
}

int gfx_null_parked(void) {
    int took = stats_lock();
    int n = wo_parked(&g_wo);
    if (took) null_unlock();
    return n;
}

uint64_t gfx_null_longest_hold(void) {
    int took = stats_lock();
    uint64_t n = g_wo.longest_hold;
    if (took) null_unlock();
    return n;
}

uint64_t gfx_null_whole_through(void) {
    int took = stats_lock();
    uint64_t n = wo_whole_through(&g_wo);
    if (took) null_unlock();
    return n;
}

static int site_of(uint64_t caller, int *first) {
    int s = 0;
    while (s < g_nsites && g_sites[s].caller != caller) s++;
    *first = s == g_nsites;
    if (*first && g_nsites < WAIT_SITES) g_sites[g_nsites++].caller = caller;
    else if (*first) s = -1;   // past the table: such a site is not logged, and its waits go to the bucket
    return s;
}

static void log_wait(const p_wait_event *p, const WoWait *how, int ok) {
    char where[96];
    rt_log("nvn: WaitEvent at %s on the event word %p (%s %u, the word holds %u) is %s at its place in its queue's "
           "replay (the word was last written by %s)", rt_guest_name(p->caller, where, (int)sizeof where),
           (const void *)p->val, p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? "equal to" : "at least", p->value, how->seen,
           ok ? "satisfied" : "NOT satisfied",
           how->writer == WO_WRITER_GPU ? "a GPU signal" : how->writer == WO_WRITER_CPU ? "a CPU signal" : "nothing yet");
}

// A wait that does not hold at its place: counted once, here, and its queue is parked.
static void complete_ready(uint64_t value);
static void write_results(results *k);
static void sink_parked(void *ctx, uint64_t queue, uint64_t fence, int index, const p_wait_event *p, const WoWait *how) {
    (void)ctx, (void)index;
    // What this submission kept before the wait is already visible to the queues that go on: write it now, after the
    // submissions that were whole before (replay order), unless completion is held.
    if (!g_hold) {
        complete_ready(0);
        write_results(results_for(fence));
    }
    int first;
    int s = site_of(p->caller, &first);
    GfxNullWaitClasses *sc = site_classes(s);
    g_waits++, g_waits_unsatisfied++;
    g_classes.unsatisfied++, sc->unsatisfied++;
    if (g_nparked == g_cparked) {
        g_cparked = g_cparked ? 2 * g_cparked : 16;
        g_parked = realloc(g_parked, sizeof *g_parked * (size_t)g_cparked);
        if (!g_parked) rt_fatal("gfx: null back end: the list of parked queues");
    }
    g_parked[g_nparked].queue = queue, g_parked[g_nparked++].site = s;
    if (s >= 0 && (first || !g_sites[s].unsatisfied_logged)) {
        log_wait(p, how, 0);
        g_sites[s].unsatisfied_logged = 1;
    }
}

// A wait shown: it held at its place (counted here), or it was parked and is released now (counted when it parked).
static void sink_wait(uint64_t queue, const p_wait_event *p, const WoWait *how) {
    int first;
    int s = site_of(p->caller, &first);
    GfxNullWaitClasses *sc = site_classes(s);
    if (how->parked) {
        for (int i = 0; i < g_nparked; i++)
            if (g_parked[i].queue == queue) {
                g_parked[i] = g_parked[--g_nparked];
                break;
            }
        if (how->writer == WO_WRITER_GPU) g_classes.later_gpu++, sc->later_gpu++;
        else g_classes.later_cpu++, sc->later_cpu++;
        return;
    }
    g_waits++;
    if (how->released) {   // judged after its queue was released from an earlier wait, not at the place it was recorded
        g_classes.after_release++, sc->after_release++;
        if (s >= 0 && first) log_wait(p, how, 1);
        return;
    }
    if (how->writer == WO_WRITER_GPU) {
        g_classes.by_gpu++, sc->by_gpu++;
        if (how->signaller == queue) g_classes.by_gpu_same++, sc->by_gpu_same++;
        else g_classes.by_gpu_other++, sc->by_gpu_other++;
    } else if (how->writer == WO_WRITER_CPU) {
        g_classes.by_cpu++, sc->by_cpu++;
    } else {
        g_classes.initial++, sc->initial++;
    }
    if (s >= 0 && first) log_wait(p, how, 1);
}

static void sink_record(void *ctx, uint64_t queue, uint64_t fence, int index, const op_hdr *r, const WoWait *how) {
    (void)ctx;
    __atomic_fetch_add(&g_records, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&g_op_counts[r->op], 1, __ATOMIC_RELAXED);
    replay(fence, r);
    ledger_track_op(&g_track, r);
    if (texheap_on()) texheap_op(fence, r);
    if (g_log) g_log(fence, index, r, g_log_ctx);
    if (how) sink_wait(queue, (const p_wait_event *)(const void *)(r + 1), how);
}

static void sink_whole(void *ctx, uint64_t queue, uint64_t fence) {
    (void)ctx, (void)queue;
    results *k = results_for(fence);
    k->whole = 1, k->seq = ++g_whole_seq;
}

// Write the results kept so far to memory, in op-stream order, and forget them.
static void write_results(results *k) {
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
            wo_signal_written(&g_wo, p->val, p->mode, p->value);   // the order engine's copy no longer owes an ADD to memory
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
        default: break;
        }
    }
    k->used = 0;
}

static void complete(results *k) {
    write_results(k);
    if (texheap_on()) texheap_complete(k->fence);
    free(k->buf);
    k->buf = NULL, k->size = 0;
    k->done = 1;
}

// Complete the submissions that are whole, up to fence value `value` (0: all of them), in the order replay finished
// them: the one with the smallest seq first, until none is left. A submission held at a wait, or behind one of its
// queue, is not whole and is skipped, not waited for: the queues that are not held complete (a queue's own submissions
// are whole in its own order, so none completes ahead of an earlier one of its queue). Without a wait, replay order is
// fence order. `value` (a test's hook, gfx_null_complete) names which submissions are completed and is not widened: a
// value that leaves out one replay finished earlier shows the later one's writes without it. The done prefix then moves
// up over the values that are done.
static void complete_ready(uint64_t value) {
    for (;;) {
        results *next = NULL;
        for (results *k = g_res; k && (!value || k->fence <= value); k = k->next)
            if (k->whole && !k->done && (!next || k->seq < next->seq)) next = k;
        if (!next) break;
        complete(next);
    }
    while (g_res && g_res->done) {
        results *k = g_res;
        g_res = k->next;
        if (!g_res) g_res_tail = &g_res;
        __atomic_store_n(&g_done, k->fence, __ATOMIC_RELEASE);
        free(k);
    }
    plat_cond_broadcast(&g_done_cond);
}

// ---------- presents, counted once the work before them is whole
// A present made while a queue is held at a wait would show a picture the held work has not drawn: it waits for
// every submission made before it (gfx.h, rule 4).
static struct { uint64_t after; } *g_present_wait;
static size_t g_npresent_wait, g_cpresent_wait;

static void presents_release(void) {
    uint64_t through = wo_whole_through(&g_wo);
    size_t kept = 0;
    for (size_t i = 0; i < g_npresent_wait; i++) {
        if (g_present_wait[i].after <= through) __atomic_fetch_add(&g_presents, 1, __ATOMIC_RELAXED);
        else g_present_wait[kept++] = g_present_wait[i];
    }
    g_npresent_wait = kept;
}

// After the order engine moved: complete what is whole (unless completion is held) and release the presents.
static void settle(void) {
    if (!g_hold) complete_ready(0);
    presents_release();
}

// ---------- the bounds of a wait nothing releases
static uint64_t now_ns(void) { return plat_now_ns(); }

static void stop_waiting(const WoStall *st, const char *why) {
    char where[96];
    rt_fatal("gfx: null back end: the WaitEvent of queue %llu at %s (submission %llu, recording %d) on the event word %p "
             "(%s %u, the word holds %u) was released by no signal in %llu submissions and %llu ms: %s",
             (ull)st->queue, rt_guest_name(st->wait.caller, where, (int)sizeof where), (ull)st->fence, st->index,
             (const void *)st->wait.val, st->wait.mode == NVN_EVENT_WAIT_MODE_EQUAL ? "equal to" : "at least",
             st->wait.value, st->seen, (ull)st->submits_since, (ull)(st->waited_ns / 1000000u), why);
}

static uint64_t bound_submits(void) { return g_bound_submits ? g_bound_submits : WO_BOUND_SUBMITS; }
static uint64_t bound_ns(void) { return g_bound_ns ? g_bound_ns : WO_STALL_NS; }

// ---------- the back-end operations
// The lock spans the whole submission: its records are replayed (those the queue's waits let through), the signals they
// make release the other queues, and what is whole is completed in fence order before the next submission starts.
static uint64_t null_submit(uint64_t queue, const uint64_t *handles, int n) {
    null_lock();
    uint64_t fence = g_submitted + 1;
    for (int i = 0; i < n; i++) check_whole(fence, i, handles[i]);
    g_submitted = fence;
    results *k = calloc(1, sizeof *k);
    if (!k) rt_fatal("gfx: null back end: a submission");
    k->fence = fence, k->queue = queue;
    *g_res_tail = k, g_res_tail = &k->next;
    if (!g_wo.sink.record) wo_start(&g_wo, &(WoSink){NULL, sink_parked, sink_record, sink_whole, now_ns});
    if (texheap_on()) texheap_submit(fence);
    wo_submit(&g_wo, queue, fence, handles, n);
    settle();
    WoStall st;
    if (wo_stalled(&g_wo, &(WoStallQuery){.max_submits = bound_submits(), .queue = WO_ANY_QUEUE}, &st))
        stop_waiting(&st, "the bound on submissions");
    null_unlock();
    return fence;
}

void gfx_null_hold(int on) {
    null_lock();
    g_hold = on;
    if (!on) complete_ready(0);
    null_unlock();
}

uint64_t gfx_null_complete(uint64_t value) {
    null_lock();
    complete_ready(value);
    uint64_t done = __atomic_load_n(&g_done, __ATOMIC_RELAXED);
    null_unlock();
    return done;
}

// Wait on the done count (g_lock held, as plat_cond_wait wants), up to timeout_ns (UINT64_MAX: none) from start. An
// unlimited wait wakes every 20 ms to look for its queue parked longer than the bound: it is the one wait that nothing
// ends when the wait it sits behind is never released.
static void wait_done(uint64_t start, uint64_t timeout_ns) {
    uint64_t slice = 20000000u;
    if (timeout_ns != UINT64_MAX) {
        uint64_t spent = plat_now_ns() - start;
        if (spent >= timeout_ns) return;
        if (timeout_ns - spent < slice) slice = timeout_ns - spent;
    }
    plat_cond_wait_ns(&g_done_cond, &g_lock, slice);
}

// The wait that began at `start` for fence values up to `upto` of `queue` (WO_ANY_QUEUE, 0: all of them) has sat behind a
// parked queue past the time bound: a stop. The time counts from the later of the park and `start`, so a queue parked
// long before a wait began (a slow CPU release, a debugger pause) does not make that wait stop at once; and a queue
// parked that this wait does not depend on is not looked at.
static void check_stall(uint64_t queue, uint64_t upto, uint64_t start) {
    WoStall st;
    if (wo_stalled(&g_wo, &(WoStallQuery){.max_ns = bound_ns(), .since_ns = start, .queue = queue, .upto_fence = upto}, &st))
        stop_waiting(&st, "a wait with no timeout for work held behind it");
}

// A value no submit has handed out is not reached by waiting here: the answer is immediate. A value is done when its
// own queue's work up to it is (gfx.h, rule 3): a queue held at a wait is waited for by the threads that wait on that
// queue and no others.
static int null_fence_wait(uint64_t value, uint64_t timeout_ns) {
    if (value <= __atomic_load_n(&g_done, __ATOMIC_ACQUIRE)) return 1;
    null_lock();
    uint64_t start = plat_now_ns();
    int ok = 0;
    for (;;) {
        if (value > g_submitted) break;
        const results *k = results_find(value);
        if (!k || k->done) {   // not in the list: below the prefix
            ok = 1;
            break;
        }
        if (timeout_ns != UINT64_MAX && plat_now_ns() - start >= timeout_ns) break;
        wait_done(start, timeout_ns);
        if (timeout_ns == UINT64_MAX && (k = results_find(value)) && !k->done) check_stall(k->queue, value, start);
    }
    null_unlock();
    return ok;
}

static uint64_t null_fence_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }

// Shown when the work before it is whole, which is at once unless a queue is held at a wait.
static void null_present(const tex_t *t, const int crop[4]) {
    (void)t, (void)crop;
    null_lock();
    if (g_submitted <= wo_whole_through(&g_wo)) {
        __atomic_fetch_add(&g_presents, 1, __ATOMIC_RELAXED);
    } else {
        if (g_npresent_wait == g_cpresent_wait) {
            g_cpresent_wait = g_cpresent_wait ? 2 * g_cpresent_wait : 16;
            g_present_wait = realloc(g_present_wait, sizeof *g_present_wait * g_cpresent_wait);
            if (!g_present_wait) rt_fatal("gfx: null back end: the list of presents held for a wait");
        }
        g_present_wait[g_npresent_wait++].after = g_submitted;
    }
    null_unlock();
}

// No present timing: every shown tick is unknown, and the answer is immediate.
static uint64_t null_present_shown(uint64_t seq) {
    (void)seq;
    return 0;
}

// Every submission so far of `queue` (0: of every queue) is done.
static int finish_pending(uint64_t queue, uint64_t target) {
    for (const results *k = g_res; k && k->fence <= target; k = k->next)
        if (!k->done && (!queue || k->queue == queue)) return 1;
    return 0;
}

static void null_finish(uint64_t queue) {
    null_lock();
    uint64_t target = 0, start = plat_now_ns();
    for (const results *k = g_res; k; k = k->next)
        if (!k->done && (!queue || k->queue == queue)) target = k->fence;
    // every matching submission up to the newest at the call, not the newest alone: with two queues held at once, a
    // later fence value can be released first
    while (finish_pending(queue, target)) {
        wait_done(start, UINT64_MAX);
        check_stall(queue ? queue : WO_ANY_QUEUE, target, start);
    }
    null_unlock();
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

// nvnEventSignal reports the CPU's signal of an event word (4 bytes): a queue held at a wait on it may go on. Any other
// 4-byte write the CPU reports is treated the same way, and the engine only remembers a word it has not met as written by
// the CPU.
static void null_guest_written(uint64_t addr, uint64_t size) {
    if (size != 4) return;
    null_lock();
    if (!g_wo.sink.record) wo_start(&g_wo, &(WoSink){NULL, sink_parked, sink_record, sink_whole, now_ns});
    wo_cpu_write(&g_wo, (const volatile uint32_t *)(uintptr_t)addr);
    settle();
    null_unlock();
}
static void null_event_initialized(uint64_t addr) {
    null_lock();
    if (!g_wo.sink.record) wo_start(&g_wo, &(WoSink){NULL, sink_parked, sink_record, sink_whole, now_ns});
    wo_word_reset(&g_wo, (const volatile uint32_t *)(uintptr_t)addr);
    settle();
    null_unlock();
}
static void null_guest_read(uint64_t addr, uint64_t size) { (void)addr, (void)size; }
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
    .event_initialized = null_event_initialized,
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
