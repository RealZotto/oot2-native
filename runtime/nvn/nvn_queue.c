// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's queues, syncs, events and timestamps (architecture section 5, D7). Windows and
// QueuePresentTexture are in nvn_window.c.
//
// Queues. Every queue's submissions share the back end's one numbering of fence values (gfx.h): QueueSubmitCommands hands
// the recordings over with the queue's serial (nvn_submit_recordings, nvn_cmd.c), so that the back end knows which queue a
// WaitEvent and a SignalEvent belong to, and keeps the fence value it got as the queue's newest; QueueFlush has nothing
// left to send; QueueFinish waits until everything this queue submitted so far is done (never another queue's work, which
// a WaitEvent may hold), with its virtual core given up. A queue's
// memory is only accounted: the game allocates what QueueBuilderGetQueueMemorySize answers, and nothing is written there.
//
// Syncs. A sync is a fence value. QueueFenceSync gives it the queue's newest one: every command submitted to that queue
// so far. SyncWait answers at once when that value is done (the value's own queue's work, gfx.h rule 3); otherwise it
// gives its virtual core up (D7) and waits, first for a queue to fence the sync when none has yet, then for the value,
// until the timeout. QueueWaitSync is a wait on the GPU: the fence's submission is made before every later one, so it
// holds nothing back on the host. A sync no queue has fenced holds nothing back either, as in DKB Recomp, nxnative and
// oot-pc-recomp; it is logged once.
//
// Events live in a memory pool the CPU sees: the GPU's signals land there when their submission completes, the CPU's at
// once. The CPU's signal also reaches the back end (guest_written of the word's 4 bytes), where it can release a queue
// that a WaitEvent holds (gfx.h, oot2/nvn_order.h).
//
// Timestamps. A counter report's timestamp is the guest's system tick (31.25 MHz, rt_ticks), so
// GetTimestampInNanoseconds converts at that rate and GetCurrentTimestampInNanoseconds reads the same clock. The
// console's driver converts with a ratio it keeps in the device (the words at +0x80 and +0x84, read in its
// nvnDeviceGetTimestampInNanoseconds), which it sets at run time; its value was not read. A game that converts through
// these entry points gets exact times whatever that raw unit is; one that converted a raw timestamp itself would not.
//
// Adapted from oot-pc-recomp runtime/nvn_api_core.c and runtime/nvn_api_nvn2.c at 8e09aaa (GPL-3.0-or-later), DKB
// Recomp's at 9d8fa02 with additions, which port MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at
// b75be8cf (MIT, Copyright (c) 2026 jhaizhou; licenses/dkb-xlate-MIT.txt): the queue builder and its memory size, the
// sync as a fence value, its wait results, the event in pool storage with its WRITE and ADD signals, QueueWaitSync and
// QueueFlush holding nothing back, GetError answering no error. Changed here: a sync takes its queue's newest fence value
// rather than the newest of every queue; SyncWait waits for a sync no queue has fenced yet, up to its timeout, instead of
// answering at once; the core is given up through the kernel's blocking call; objects never initialised stop by name, as
// do an event the CPU cannot read and a sync condition NVN does not have; QueueGetError writes nothing into the
// information it is given; the timestamps are this project's.
#include <stdlib.h>
#include <string.h>

#include "nvn_queue.h"
#include "oot2/capture.h"
#include "oot2/kernel.h"
#include "oot2/vsync.h"
#include "plat.h"

typedef unsigned long long ull;

#define QUEUE_MAGIC MAGIC('Q', 'U', 'E', 'U')
#define SYNC_MAGIC MAGIC('S', 'Y', 'N', 'C')
#define EVENT_MAGIC MAGIC('E', 'V', 'N', 'T')

queue_t *nvn_queue_arg(Cpu *c, int i, const char *fn) {
    queue_t *q = arg_p(c, i);
    if (!q || q->magic != QUEUE_MAGIC)
        rt_unimplemented(c, "nvn", "%s on the queue %p, which QueueInitialize did not initialise", fn, (void *)q);
    return q;
}

sync_t *nvn_sync_arg(Cpu *c, int i, const char *fn) {
    sync_t *s = arg_p(c, i);
    if (!s || s->magic != SYNC_MAGIC)
        rt_unimplemented(c, "nvn", "%s on the sync %p, which SyncInitialize did not initialise", fn, (void *)s);
    return s;
}

static ev_t *event_arg(Cpu *c, const char *fn) {
    ev_t *e = arg_p(c, 0);
    if (!e || e->magic != EVENT_MAGIC)
        rt_unimplemented(c, "nvn", "%s on the event %p, which EventInitialize did not initialise", fn, (void *)e);
    return e;
}

// ---------- queue builder
#define QB(c) ((queue_builder_t *)arg_p(c, 0))
// void nvnQueueBuilderSetDefaults(NVNqueueBuilder *b): the device stays (the game sets it first); the memory sizes the
// three reference projects answer
NVN_API(nvnQueueBuilderSetDefaults) {
    queue_builder_t *b = QB(c);
    NVNdevice *d = b->dev;
    memset(b, 0, sizeof *b);
    b->dev = d, b->cmd_sz = 65536, b->comp_sz = 262144, b->ctl_sz = 16384, b->flush = 4096;
}
// void nvnQueueBuilderSetDevice(NVNqueueBuilder *b, NVNdevice *d), SetFlags(b, int flags), and
// Set{Command,Compute,Control}MemorySize(b, size_t n)
NVN_API(nvnQueueBuilderSetDevice) { QB(c)->dev = arg_p(c, 1); }
NVN_API(nvnQueueBuilderSetFlags) { QB(c)->flags = (uint32_t)arg_x(c, 1); }
NVN_API(nvnQueueBuilderSetCommandMemorySize) { QB(c)->cmd_sz = arg_x(c, 1); }
NVN_API(nvnQueueBuilderSetComputeMemorySize) { QB(c)->comp_sz = arg_x(c, 1); }
NVN_API(nvnQueueBuilderSetControlMemorySize) { QB(c)->ctl_sz = arg_x(c, 1); }
// size_t nvnQueueBuilderGetQueueMemorySize(const NVNqueueBuilder *b): the three sizes, rounded up to 4 KiB
NVN_API(nvnQueueBuilderGetQueueMemorySize) {
    const queue_builder_t *b = arg_p(c, 0);
    ret_x(c, (b->cmd_sz + b->comp_sz + b->ctl_sz + 4095) & ~4095ull);
}
// void nvnQueueBuilderSetQueueMemory(NVNqueueBuilder *b, void *memory, size_t size): kept, never written
NVN_API(nvnQueueBuilderSetQueueMemory) {
    queue_builder_t *b = QB(c);
    b->mem = arg_p(c, 1), b->mem_sz = arg_x(c, 2);
}
// void nvnQueueBuilderSetPrimaryQueue(NVNqueueBuilder *b, NVNqueue *primary) (NVN2): a queue tied to another. Every
// queue already shares one order of fence values here and the back end orders their WaitEvents by queue, so nothing is
// kept; the first call is logged.
NVN_API(nvnQueueBuilderSetPrimaryQueue) {
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("nvn: QueueBuilderSetPrimaryQueue(%p): every queue shares one submission order", arg_p(c, 1));
}

// ---------- queues
// NVNboolean nvnQueueInitialize(NVNqueue *q, const NVNqueueBuilder *b)
NVN_API(nvnQueueInitialize) {
    queue_t *q = arg_p(c, 0);
    static uint64_t serials;   // queues initialised so far: each takes the next number, and a queue initialised again a new one
    *q = (queue_t){QUEUE_MAGIC, ((const queue_builder_t *)arg_p(c, 1))->dev, 0, 0,
                   __atomic_add_fetch(&serials, 1, __ATOMIC_RELAXED)};
    ret_x(c, 1);
}
// void nvnQueueFinalize(NVNqueue *q)
NVN_API(nvnQueueFinalize) { nvn_queue_arg(c, 0, "QueueFinalize")->magic = 0; }
// void nvnQueueSubmitCommands(NVNqueue *q, int n, const NVNcommandHandle *handles): the queue keeps the newest fence
// value it was given, even when two threads submit to it at once
NVN_API(nvnQueueSubmitCommands) {
    queue_t *q = nvn_queue_arg(c, 0, "QueueSubmitCommands");
    uint64_t fence = nvn_submit_recordings(c, q->serial, arg_p(c, 2), (int)arg_x(c, 1));
    __atomic_fetch_add(&q->submits, 1, __ATOMIC_RELAXED);
    uint64_t last = __atomic_load_n(&q->last, __ATOMIC_RELAXED);
    while (fence > last &&
           !__atomic_compare_exchange_n(&q->last, &last, fence, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
    }
    capture_submit();   // the M1 capture names the frame's submitting thread
}
// void nvnQueueFlush(NVNqueue *q): every submission is with the back end already
NVN_API(nvnQueueFlush) { nvn_queue_arg(c, 0, "QueueFlush"); }
// void nvnQueueFinish(NVNqueue *q)
NVN_API(nvnQueueFinish) {
    const queue_t *q = nvn_queue_arg(c, 0, "QueueFinish");
    const GfxBackend *b = gfx_backend();
    int token = rt_kernel_block_begin();
    b->finish(q->serial);   // this queue's work, never another queue's held at a wait (gfx.h, rule 3)
    rt_kernel_block_end(token);
    b->cpu_synced(b->fence_done());
}
// NVNqueueGetErrorResult nvnQueueGetError(NVNqueue *q, NVNqueueErrorInfo *info): no error; info is not written
NVN_API(nvnQueueGetError) {
    nvn_queue_arg(c, 0, "QueueGetError");
    ret_x(c, NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR);
}

// ---------- syncs
static PlatMutex g_fenced_lock = PLAT_MUTEX_INIT;
static PlatCond g_fenced = PLAT_COND_INIT;   // a sync was fenced

// The fence value, the tick and the fenced word move together under g_fenced_lock, so a wait that reads them under it sees
// one acquire's (or one queue's) set, never a mix.
void nvn_sync_acquire(sync_t *s, uint64_t value, uint64_t ready) {
    plat_mutex_lock(&g_fenced_lock);
    __atomic_store_n(&s->fence, value, __ATOMIC_RELAXED);
    __atomic_store_n(&s->ready, ready, __ATOMIC_RELAXED);
    __atomic_store_n(&s->fenced, 2, __ATOMIC_RELEASE);
    plat_cond_broadcast(&g_fenced);
    plat_mutex_unlock(&g_fenced_lock);
}

void nvn_sync_fence(sync_t *s, uint32_t how, uint64_t value) {
    plat_mutex_lock(&g_fenced_lock);
    __atomic_store_n(&s->fence, value, __ATOMIC_RELAXED);
    __atomic_store_n(&s->ready, 0, __ATOMIC_RELAXED);   // a queue's fence has no tick: a stale one of an earlier acquire goes
    __atomic_store_n(&s->fenced, how, __ATOMIC_RELEASE);
    plat_cond_broadcast(&g_fenced);
    plat_mutex_unlock(&g_fenced_lock);
}

// The sync's state as one acquire or queue left it: fenced, fence value, tick.
static uint32_t sync_state(const sync_t *s, uint64_t *fence, uint64_t *ready) {
    plat_mutex_lock(&g_fenced_lock);
    uint32_t fenced = __atomic_load_n(&s->fenced, __ATOMIC_ACQUIRE);
    *fence = __atomic_load_n(&s->fence, __ATOMIC_RELAXED);
    *ready = __atomic_load_n(&s->ready, __ATOMIC_RELAXED);
    plat_mutex_unlock(&g_fenced_lock);
    return fenced;
}

// NVNboolean nvnSyncInitialize(NVNsync *s, NVNdevice *d): not fenced yet
NVN_API(nvnSyncInitialize) {
    *(sync_t *)arg_p(c, 0) = (sync_t){SYNC_MAGIC, 0, 0, 0};
    ret_x(c, 1);
}
// void nvnSyncFinalize(NVNsync *s)
NVN_API(nvnSyncFinalize) { nvn_sync_arg(c, 0, "SyncFinalize")->magic = 0; }
// void nvnQueueFenceSync(NVNqueue *q, NVNsync *s, NVNsyncCondition condition, int flags): every command submitted to the
// queue so far. GRAPHICS_WORLD_SPACE_COMPLETE, which asks only for the vertex work, waits for all of it here.
NVN_API(nvnQueueFenceSync) {
    queue_t *q = nvn_queue_arg(c, 0, "QueueFenceSync");
    sync_t *s = nvn_sync_arg(c, 1, "QueueFenceSync");
    uint32_t condition = (uint32_t)arg_x(c, 2);
    if (condition != NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE &&
        condition != NVN_SYNC_CONDITION_GRAPHICS_WORLD_SPACE_COMPLETE)
        rt_unimplemented(c, "nvn", "QueueFenceSync: condition %u is not one of NVN's", condition);
    nvn_sync_fence(s, 1, __atomic_load_n(&q->last, __ATOMIC_ACQUIRE));
}

// The time left of a SyncWait's timeout. The one clock of the whole wait is the vsync clock's own (the driver's when a
// test drives it, the host's otherwise), the clock the tick is on, so a host stall in a driven test spends nothing.
static uint64_t left(uint64_t timeout, uint64_t start) {
    if (timeout == UINT64_MAX) return UINT64_MAX;
    uint64_t spent = rt_vsync_now_ns() - start;
    return spent < timeout ? timeout - spent : 0;
}

// Wait until a queue fences s, or the timeout: 1 once it is fenced.
static int wait_fenced(const sync_t *s, uint64_t timeout, uint64_t start) {
    plat_mutex_lock(&g_fenced_lock);
    while (!__atomic_load_n(&s->fenced, __ATOMIC_ACQUIRE)) {
        uint64_t rest = left(timeout, start);
        if (!rest) break;
        if (rest == UINT64_MAX) plat_cond_wait(&g_fenced, &g_fenced_lock);
        else plat_cond_wait_ns(&g_fenced, &g_fenced_lock, rest);
    }
    int fenced = __atomic_load_n(&s->fenced, __ATOMIC_ACQUIRE) != 0;
    plat_mutex_unlock(&g_fenced_lock);
    return fenced;
}

// began: the host's time at the call, for the trace's own measure of the call.
static void sync_return(Cpu *c, uint32_t result, int traced, uint64_t began) {
    if (traced)
        nvn_trace("sync-return", 0, "result %u waited_ns %llu", result, (unsigned long long)(plat_now_ns() - began));
    ret_x(c, result);
}

// NVNsyncWaitResult nvnSyncWait(const NVNsync *s, uint64_t timeout_ns): ALREADY_SIGNALED when the sync is done at the
// call, CONDITION_SATISFIED when it is reached while waiting, TIMEOUT_EXPIRED otherwise. Done is the fence value (the
// back end's fence_done) and, for a window's acquire sync, the count `ready` (the virtual clock has reached it). The
// state is read at the call, for the answer at once, and again when the wait starts, which picks up the acquire that
// fenced a sync not fenced yet; an acquire that fences the sync again once the wait is on (a game reuses one sync) does
// not move it. The tick is waited for with the virtual core given up, no lock held; the timeout, from the call to the
// end, runs on the clock's own time (rt_vsync_now_ns), the one the tick is on.
NVN_API(nvnSyncWait) {
    const sync_t *s = nvn_sync_arg(c, 0, "SyncWait");
    uint64_t timeout = arg_x(c, 1), start = rt_vsync_now_ns(), began = plat_now_ns();
    const GfxBackend *b = gfx_backend();
    uint64_t fence, ready;
    uint32_t fenced = sync_state(s, &fence, &ready);
    int traced = nvn_trace_on();
    if (traced)
        nvn_trace("sync-enter", 0, "timeout %llu ready %llu fenced %u", (unsigned long long)timeout,
                  (unsigned long long)rt_sched_slot_of(ready, RT_VSYNC_HZ), fenced);   // the tick it falls in
    if (fenced) {
        if ((b->fence_done() >= fence || b->fence_wait(fence, 0)) && (!ready || rt_vsync_counter() >= ready)) {
            b->cpu_synced(fence);
            sync_return(c, NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED, traced, began);
            return;
        }
    }
    if (!timeout) {
        sync_return(c, NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED, traced, began);
        return;
    }
    int token = rt_kernel_block_begin(), ok = 0;
    if (wait_fenced(s, timeout, start)) {
        sync_state(s, &fence, &ready);   // the pair of the acquire or queue that fenced it
        ok = 1;
        if (ready) ok = rt_vsync_wait_counter_for(ready, left(timeout, start));
        if (ok) ok = b->fence_wait(fence, left(timeout, start));
    }
    rt_kernel_block_end(token);
    if (ok) b->cpu_synced(fence);
    sync_return(c, ok ? NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED : NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED, traced, began);
}
// NVNboolean nvnQueueWaitSync(NVNqueue *q, const NVNsync *s): later work of the queue waits for the sync on the GPU
NVN_API(nvnQueueWaitSync) {
    nvn_queue_arg(c, 0, "QueueWaitSync");
    const sync_t *s = nvn_sync_arg(c, 1, "QueueWaitSync");
    if (!__atomic_load_n(&s->fenced, __ATOMIC_ACQUIRE)) {
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("nvn: QueueWaitSync on the sync %p, which no queue has fenced: nothing is held back",
                   (const void *)s);
    }
    ret_x(c, 1);
}

// ---------- events
// void nvnEventBuilderSetDefaults(NVNeventBuilder *b), SetStorage(b, const NVNmemoryPool *pool, int64_t offset)
NVN_API(nvnEventBuilderSetDefaults) { memset(arg_p(c, 0), 0, sizeof(ev_builder_t)); }
NVN_API(nvnEventBuilderSetStorage) {
    *(ev_builder_t *)arg_p(c, 0) = (ev_builder_t){arg_p(c, 1), (int64_t)arg_x(c, 2)};
}
// NVNboolean nvnEventInitialize(NVNevent *e, const NVNeventBuilder *b): the value is the 32-bit word at the storage
NVN_API(nvnEventInitialize) {
    const ev_builder_t *b = arg_p(c, 1);
    const pool_t *p = b->pool;
    if (!p || p->magic != NVN_POOL_MAGIC)
        rt_unimplemented(c, "nvn", "EventInitialize: the memory pool %p is not initialised", (const void *)p);
    if (b->off < 0 || (b->off & 3) || (uint64_t)b->off + 4 > p->size)
        rt_unimplemented(c, "nvn", "EventInitialize: offset %lld in the memory pool %p of %#llx bytes", (long long)b->off,
                         (const void *)p, (ull)p->size);
    if (!p->mem || gfx_gpu_only((uintptr_t)p->mem))
        rt_unimplemented(c, "nvn", "EventInitialize: the memory pool %p is one the CPU cannot see, where the front end "
                         "has no word to signal or read", (const void *)p);
    *(ev_t *)arg_p(c, 0) = (ev_t){EVENT_MAGIC, (volatile uint32_t *)(p->mem + b->off)};
    if (texheap_on()) texheap_range(TEXHEAP_OBJECT, (uintptr_t)(p->mem + b->off), 4);
    // Nothing is written, but a word the back end knows from an earlier event at this storage (the game reuses pool
    // memory) is read again: the back end's copy of its value must not outlive the event it belonged to.
    if (gfx_backend()->event_initialized) gfx_backend()->event_initialized((uintptr_t)(p->mem + b->off));
    ret_x(c, 1);
}
// void nvnEventFinalize(NVNevent *e)
NVN_API(nvnEventFinalize) { event_arg(c, "EventFinalize")->magic = 0; }
// uint32_t nvnEventGetValue(const NVNevent *e)
NVN_API(nvnEventGetValue) { ret_x(c, __atomic_load_n(event_arg(c, "EventGetValue")->val, __ATOMIC_ACQUIRE)); }
// void nvnEventSignal(NVNevent *e, NVNeventSignalMode mode, uint32_t value): the CPU's signal, at once
NVN_API(nvnEventSignal) {
    ev_t *e = event_arg(c, "EventSignal");
    uint32_t mode = (uint32_t)arg_x(c, 1), value = (uint32_t)arg_x(c, 2);
    if (mode == NVN_EVENT_SIGNAL_MODE_ADD) __atomic_fetch_add(e->val, value, __ATOMIC_RELEASE);
    else if (mode == NVN_EVENT_SIGNAL_MODE_WRITE) __atomic_store_n(e->val, value, __ATOMIC_RELEASE);
    else rt_unimplemented(c, "nvn", "EventSignal: mode %u is not one of NVN's", mode);
    gfx_backend()->guest_written((uintptr_t)e->val, 4);
}

// ---------- timestamps
// uint64_t nvnDeviceGetTimestampInNanoseconds(const NVNdevice *d, const NVNcounterData *report)
NVN_API(nvnDeviceGetTimestampInNanoseconds) {
    const NVNcounterData *r = arg_p(c, 1);
    if (!r) rt_unimplemented(c, "nvn", "DeviceGetTimestampInNanoseconds of a report at address 0");
    ret_x(c, (uint64_t)((__uint128_t)r->timestamp * 1000000000u / rt_tick_hz()));
}
// uint64_t nvnDeviceGetCurrentTimestampInNanoseconds(const NVNdevice *d): the clock reports read
NVN_API(nvnDeviceGetCurrentTimestampInNanoseconds) {
    ret_x(c, (uint64_t)((__uint128_t)rt_ticks() * 1000000000u / rt_tick_hz()));
}
