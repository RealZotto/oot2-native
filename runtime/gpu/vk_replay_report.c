// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's reports, events, waits and conditional rendering (oot2/vk_replay.h; architecture section 5, D17a): what a
// submission's ReportCounter, ResetCounter, SignalEvent and WaitEvent leave to the completion thread, the occlusion
// queries that count samples passed, and the condition of SetRenderEnableConditional.
//
// Completion. The walk copies each of these records into the frame's list, in op-stream order. One effect (vkrec_effect),
// queued after the frame's end and so after the backing map's write-backs of the frame's GPU writes into mirrors, walks
// the list on the completion thread once the device reached the frame's value and before the value is done: it signals
// events (WRITE, ADD), writes counter reports {counter, timestamp} and checks each WaitEvent. A guest that sees an event
// signalled or a fence done therefore sees every result before it in submission order, the GPU's written-back copies
// included (gfx.h, Synchronisation).
//
// Timestamps are the guest's system tick (31.25 MHz, rt_ticks) read from the host clock when the report completes, as
// the null back end writes them: monotonic, never earlier than the work before them. The counter word is 0 for every
// type but samples passed.
//
// Samples passed. Counting starts with the channel's first ResetCounter or ReportCounter of samples passed and stays on
// (NVN's counter is cumulative): from then on the first draw of each pass begins an occlusion query, which ends with the
// pass or at the next samples-passed report or reset (a query begun in a pass ends in it). A frame's queries come from
// pools of its own, VKR_QUERIES at a time (VkReplayOps.queries, reset in the frame's prologue), up to VKR_QUERY_POOLS. The completion thread keeps the running count:
// each query's result is added in order, a reset sets the count to 0 after the queries before it, a report writes it.
// The back end sees every queue's work in one order and does not use the queue that made it (vk_submit.c), so the channel
// has one count.
// Draws before counting started are not counted.
//
// Conditional rendering is decided at replay, from the counter words at addr and addr + 16: read from memory, except
// where the newest report to that address is not done yet (in this submission or one in flight), which counts as 0 for
// a timestamp or any other counter and as unknown for samples passed. An unknown condition renders, and so does one
// whose words the CPU cannot read (a draw costs at most the work the condition would have saved).
//
// Reports into memory the CPU cannot see (a pool in the GPU address window, a virtual pool included) are dropped,
// logged once, as the null back end drops them: nothing the CPU reads holds them. A report or event the completion thread
// writes marks its page's generation (oot2/pagegen.h), as the CPU's own writes are marked, so a mirror of the page is
// brought current before a later submission reads it. A GPU op later in the same submission that reads the words (a copy
// of the report buffer) sees them as they were before the report: the results land at completion, not in op-stream order
// on the GPU.
//
// The completion order and the condition rule follow DKB Recomp's runtime/vk_exec_xfer.c at 9d8fa02
// (GPL-3.0-or-later): event signals and counter reports written on the completion thread in submit order, as gfx_null.c
// writes them, and counter_word and cond_enabled (the newest pending report to an address decides). Changed here: the
// results wait for the backing map's write-backs of the same frame, a query pool per frame reset in the prologue (DKB
// keeps one ring and resets on the host), counting that stays on once begun, and WaitEvent checked at completion as
// the null back end checks it.
#include <stdlib.h>
#include <string.h>

#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/pagegen.h"
#include "oot2/replay_count.h"
#include "oot2/runtime.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"
#include "vk_replay_int.h"

enum { E_SIGNAL, E_REPORT, E_RESET, E_WAIT };
typedef struct {
    uint32_t kind;
    uint32_t q;          // samples-passed report or reset: the frame's queries closed before it
    uint64_t stamp;      // a report's tick taken at replay (vkr_stamp_at_replay only)
    union {
        p_signal_event sig;
        p_wait_event wait;
        struct { uint64_t type, addr; } rep;   // addr 0: dropped
    } u;
} Entry;

// What a frame leaves to the completion thread.
typedef struct {
    const VkReplayOps *ops;
    VkQueryPool pools[VKR_QUERY_POOLS];   // query q is query q % VKR_QUERIES of pools[q / VKR_QUERIES]
    uint32_t np, nq, n;                   // pools taken, queries used, entries
    void *trace;                          // the trace's note of each query (vkrt_queries_take), NULL with the trace off
    Entry e[];
} Done;

// ---------- the replay's side (the submitting thread, one frame at a time)
static struct {
    Entry *e;
    uint32_t n, cap;
    VkQueryPool pools[VKR_QUERY_POOLS];
    uint32_t np, nq;
    int open;            // the query active in the open pass, -1 for none
} R = {.open = -1};

int vkr_no_pending, vkr_no_queries, vkr_stamp_at_replay, vkr_reverse_results;

static int g_counting;   // samples passed are counted (the channel's, across frames)

// Reports remain in submission order until their timeline value is done.
typedef struct { uint64_t addr, value; int samples; } Pending;
static Pending *g_pending;
static size_t g_pending_n, g_pending_cap;

// ---------- the completion thread's side, and what tests read
static PlatMutex g_lock = PLAT_MUTEX_INIT;   // the statistics and the WaitEvent sites
static uint64_t g_running;                   // the samples-passed count (completion thread only)
static VkrReportStats g_st;

#define WAIT_SITES 64
static struct { uint64_t caller; int unsatisfied_logged; } g_sites[WAIT_SITES];
static int g_nsites;
// The call sites whose WaitEvent was unsatisfied in the frame being completed (g_lock): the counters' "first per site per
// frame" (waits_unsat_first). Counted only with the replay counters on.
static uint64_t g_frame_unsat[WAIT_SITES];
static int g_frame_nunsat;

void vkr_report_reset(void) {
    g_counting = 0, g_pending_n = 0, R.n = 0, R.np = 0, R.nq = 0, R.open = -1;
    plat_mutex_lock(&g_lock);
    g_running = 0, g_nsites = 0;
    memset(&g_st, 0, sizeof g_st);
    plat_mutex_unlock(&g_lock);
}

void vkr_report_stats(VkrReportStats *out) {
    plat_mutex_lock(&g_lock);
    *out = g_st;
    plat_mutex_unlock(&g_lock);
}

static void push(Entry e) {
    if (R.n == R.cap) {
        uint32_t cap = R.cap ? 2 * R.cap : 64;
        Entry *p = realloc(R.e, sizeof *p * cap);
        if (!p) rt_fatal("gfx: vulkan: out of memory for %u reports and events", cap);
        R.e = p, R.cap = cap;
    }
    R.e[R.n++] = e;
}

void vkr_report_frame_begin(void) { R.n = 0, R.np = 0, R.nq = 0, R.open = -1; }

static void pending_add(uint64_t addr, uint64_t value, int samples) {
    if (g_pending_n == g_pending_cap) {
        uint64_t done = gpu_sync_done();
        size_t first = 0;
        while (first < g_pending_n && g_pending[first].value <= done) first++;
        if (first) {
            g_pending_n -= first;
            memmove(g_pending, g_pending + first, g_pending_n * sizeof *g_pending);
        }
        if (g_pending_n == g_pending_cap) {
            if (g_pending_cap > SIZE_MAX / 2 / sizeof *g_pending)
                rt_fatal("gfx: vulkan: too many pending counter reports");
            size_t cap = g_pending_cap ? 2 * g_pending_cap : 1024;
            Pending *p = realloc(g_pending, cap * sizeof *p);
            if (!p) rt_fatal("gfx: vulkan: out of memory for pending counter reports");
            g_pending = p, g_pending_cap = cap;
        }
    }
    g_pending[g_pending_n++] = (Pending){addr, value, samples};
}

// The first draw of a pass while counting begins a query.
void vkr_report_draw(void) {
    if (!g_counting || R.open >= 0 || vkr_no_queries) return;
    const VkReplayOps *ops = vkr_frame_ops();
    if (R.nq == R.np * VKR_QUERIES) {
        if (R.np == VKR_QUERY_POOLS)
            vkr_stop("more than %u occlusion queries in one submission", VKR_QUERY_POOLS * VKR_QUERIES);
        R.pools[R.np] = ops->queries ? ops->queries(vkr_frame_value()) : VK_NULL_HANDLE;
        if (!R.pools[R.np]) vkr_stop("no occlusion query pool for the samples-passed counter");
        R.np++;
    }
    vkrec_BeginQuery(R.pools[R.nq / VKR_QUERIES], R.nq % VKR_QUERIES, 0);
    R.open = (int)R.nq++;
    if (vkrt_on) vkrt_query((uint32_t)R.open);
}

// The pass ends, or a samples-passed report or reset closes the segment: the open query ends.
void vkr_report_pass_end(void) {
    if (R.open < 0) return;
    vkrec_EndQuery(R.pools[(uint32_t)R.open / VKR_QUERIES], (uint32_t)R.open % VKR_QUERIES);
    R.open = -1;
}

// The address a report is written to, or 0 when the CPU cannot see it (dropped). An address in the GPU window is no
// host memory, whatever backs it (a virtual pool over a CPU-visible physical pool resolves to that pool's kind, at another
// address). Any other is resolved in the frame, so that the pool's finalization waits for the frame; the backing map stops
// by name on an address in no pool, where the console's GPU would fault.
static uint64_t report_addr(uint64_t addr) {
    BackingRef ref = {.kind = BACKING_NONE};
    if (!gfx_gpu_only(addr)) vkr_frame_ops()->resolve(addr, sizeof(NVNcounterData), vkr_frame_value(), &ref);
    if (ref.kind == BACKING_IMPORT || ref.kind == BACKING_MIRROR) return addr;
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("gfx: vulkan: a counter report at %#llx is in memory the CPU cannot see: dropped (logged once)", (ull)addr);
    plat_mutex_lock(&g_lock);
    g_st.dropped++;
    plat_mutex_unlock(&g_lock);
    return 0;
}

int vkr_report_op(const op_hdr *r) {
    const void *a = r + 1;
    switch (r->op) {
    case OP_REPORT_COUNTER: {
        const p_report_counter *p = a;
        int samples = p->type == NVN_COUNTER_TYPE_SAMPLES_PASSED;
        Entry e = {.kind = E_REPORT};
        if (samples) g_counting = 1, vkr_report_pass_end(), e.q = R.nq;
        e.u.rep.type = p->type, e.u.rep.addr = report_addr(p->addr);
        if (vkr_stamp_at_replay) e.stamp = rt_ticks();
        push(e);
        pending_add(p->addr, vkr_frame_value(), samples);
        return 1;
    }
    case OP_RESET_COUNTER:
        // Only samples passed has a count to reset: every other counter reports 0 here.
        if (((const p_reset_counter *)a)->type == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
            g_counting = 1, vkr_report_pass_end();
            push((Entry){.kind = E_RESET, .q = R.nq});
        }
        return 1;
    case OP_SIGNAL_EVENT: {
        const p_signal_event *p = a;
        if (p->mode != NVN_EVENT_SIGNAL_MODE_WRITE && p->mode != NVN_EVENT_SIGNAL_MODE_ADD)
            vkr_stop("SignalEvent mode %u is not one of NVN's", p->mode);
        push((Entry){.kind = E_SIGNAL, .u.sig = *p});
        return 1;
    }
    case OP_WAIT_EVENT:
        push((Entry){.kind = E_WAIT, .u.wait = *(const p_wait_event *)a});
        return 1;
    default: return 0;
    }
}

// ---------- the condition, at replay
// 1: *v read; 2: a samples-passed report not done (unknown); 0: the CPU cannot read the word.
static int counter_word(uint64_t addr, uint64_t *v) {
    uint64_t done = gpu_sync_done();
    for (size_t i = g_pending_n; !vkr_no_pending && i-- > 0;) {
        if (g_pending[i].value <= done) break;
        if (g_pending[i].addr == addr) return *v = 0, g_pending[i].samples ? 2 : 1;
    }
    const VkReplayOps *ops = vkr_frame_ops();
    if (gfx_gpu_only(addr) || !ops->cpu_reads || !ops->cpu_reads(addr, sizeof *v)) return 0;
    memcpy(v, (const void *)(uintptr_t)addr, sizeof *v);
    return 1;
}

int vkr_condition(const p_render_cond *p) {
    uint64_t x, y;
    int rx = counter_word(p->addr, &x), ry = counter_word(p->addr + 16, &y);
    int render;
    if (!rx || !ry) {
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("gfx: vulkan: the conditional rendering counters at %#llx are not readable by the CPU: rendering "
                   "(logged once)", (ull)p->addr);
        render = 1;
    } else if (rx == 2 || ry == 2) {
        render = 1;
    } else {
        render = p->mode == 0 ? x == y : x != y;
    }
    plat_mutex_lock(&g_lock);
    g_st.conditions++;
    if (!render) g_st.conditions_false++;
    plat_mutex_unlock(&g_lock);
    return render;
}

// ---------- completion
static int wait_ok(const p_wait_event *p, uint32_t *w) {
    *w = __atomic_load_n(p->val, __ATOMIC_ACQUIRE);
    return p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? *w == p->value : (int32_t)(*w - p->value) >= 0;
}

// A WaitEvent at its place in submission order (g_lock held): counted, and logged at the first wait of its call site
// and at the site's first unsatisfied one, which the console's queue would have held for another queue or the CPU.
static void wait_event(const p_wait_event *p) {
    uint32_t w;
    int ok = wait_ok(p, &w);
    g_st.waits++;
    if (!ok) g_st.waits_unsatisfied++;
    if (!ok && RC_ON()) {
        int seen = 0;
        for (int i = 0; i < g_frame_nunsat && !seen; i++) seen = g_frame_unsat[i] == p->caller;
        if (!seen) {
            if (g_frame_nunsat < WAIT_SITES) g_frame_unsat[g_frame_nunsat++] = p->caller;
            g_st.waits_unsat_first++;
        }
    }
    int s = 0;
    while (s < g_nsites && g_sites[s].caller != p->caller) s++;
    int first = s == g_nsites;
    if (first && g_nsites < WAIT_SITES) g_sites[g_nsites++] = (__typeof__(g_sites[0])){p->caller, 0};
    else if (first) return;   // past the table: such a site is counted, not logged
    if (!first && (ok || g_sites[s].unsatisfied_logged)) return;
    char where[96];
    rt_log("gfx: vulkan: WaitEvent at %s on the event word %p (%s %u, the word holds %u) is %s at its place in "
           "submission order", rt_guest_name(p->caller, where, (int)sizeof where), (const void *)p->val,
           p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? "equal to" : "at least", p->value, w, ok ? "satisfied" : "NOT satisfied");
    if (!ok) g_sites[s].unsatisfied_logged = 1;
}

static void complete(void *arg, uint64_t value) {
    (void)value;
    Done *d = arg;
    uint64_t res[VKR_QUERY_POOLS * VKR_QUERIES];
    for (uint32_t i = 0; i < d->np; i++) {   // and each pool goes back
        uint32_t n = d->nq - i * VKR_QUERIES;
        d->ops->query_results(d->pools[i], n < VKR_QUERIES ? n : VKR_QUERIES, res + i * VKR_QUERIES);
    }
    if (d->trace) vkrt_queries_done(d->trace, d->nq, res);
    uint32_t next = 0;
    plat_mutex_lock(&g_lock);
    g_frame_nunsat = 0;
    for (uint32_t k = 0; k < d->n; k++) {
        const Entry *e = &d->e[vkr_reverse_results ? d->n - 1 - k : k];
        switch (e->kind) {
        case E_SIGNAL:
            if (e->u.sig.mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->u.sig.val, e->u.sig.value, __ATOMIC_RELEASE);
            else __atomic_store_n(e->u.sig.val, e->u.sig.value, __ATOMIC_RELEASE);
            pagegen_mark((uint64_t)(uintptr_t)e->u.sig.val, (uint64_t)(uintptr_t)e->u.sig.val + 4);
            g_st.events++;
            break;
        case E_RESET:
            while (next < e->q) g_running += res[next++];
            g_running = 0;
            g_st.resets++;
            break;
        case E_REPORT: {
            uint64_t rec[2] = {0, 0};
            if (e->u.rep.type == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
                while (next < e->q) g_running += res[next++];
                rec[0] = g_running;
            }
            rec[1] = vkr_stamp_at_replay ? e->stamp : rt_ticks();
            if (e->u.rep.addr) {
                memcpy((void *)(uintptr_t)e->u.rep.addr, rec, sizeof rec);
                pagegen_mark(e->u.rep.addr, e->u.rep.addr + sizeof rec);
            }
            g_st.reports++;
            break;
        }
        case E_WAIT: wait_event(&e->u.wait); break;
        }
    }
    while (next < d->nq) g_running += res[next++];   // counted, for a report in a later frame
    g_st.queries += d->nq;
    plat_mutex_unlock(&g_lock);
    free(d);
}

// After the frame's end (the open query closed with its pass): its list goes to the completion thread.
void vkr_report_frame_end(void) {
    if (!R.n && !R.np) return;
    Done *d = malloc(sizeof *d + sizeof d->e[0] * R.n);
    if (!d) rt_fatal("gfx: vulkan: out of memory for a frame's reports and events");
    d->ops = vkr_frame_ops(), d->np = R.np, d->nq = R.nq, d->n = R.n;
    d->trace = vkrt_on ? vkrt_queries_take(R.nq) : NULL;
    memcpy(d->pools, R.pools, sizeof d->pools);
    if (R.n) memcpy(d->e, R.e, sizeof d->e[0] * R.n);
    vkrec_effect(complete, d);
    R.n = 0, R.np = 0, R.nq = 0;
}
