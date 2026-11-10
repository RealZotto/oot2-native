// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The timeline of the Vulkan back end and its completion thread (oot2/gpu.h; architecture section 5 and section 6.1).
// Nothing here is Vulkan: the device's progress comes from a GpuTimelineSource, so the same code runs over the device's
// timeline semaphore (vk_submit.c) and over a test's counter, in continuous integration, which has no GPU.
//
// A submission takes the submission gate for its whole length: the value it is handed, the effects it queues and the
// queue submit that signals the value, so values reach the device and the list below in the order they are handed out.
// The gate is a flag, not a lock owned by a thread: the recorder thread (runtime/gpu/vk_rec.c) ends a frame that the
// submitting thread began, so gpu_sync_end may come from another thread than gpu_sync_begin.
// The list holds every issued value not done yet, oldest first, each with its effects in the order they were added.
// The completion thread waits on the source for the oldest value, and for each value the device reached, in order, runs
// its effects and then publishes it: a value is never done before its effects and those of every earlier value ran.
//
// Adapted from DKB Recomp runtime/vk_sync.c at 9d8fa02 (GPL-3.0-or-later; there adapted from LAS Recomp
// runtime/nvn/vulkan/exec.c at da35b83, MIT, Copyright (c) 2026 Peiikai, licenses/lasrecomp-MIT.txt, and from
// MacNativeGames hle_nvn.c at b75be8c, MIT, Copyright (c) 2026 jhaizhou, licenses/macnativegames-MIT.txt): one value per
// submission, a completion thread that runs the CPU-side effects in submission order and publishes a value only after
// them, waits that time out, a value never handed out answered at once. Changed here: the effects belong to their
// submission and are queued only while it holds the submission lock, so they are in value order by construction
// (DKB's one queue takes a callback for any value from any thread, and a later value's callback added first would hold
// an earlier one back while the earlier value is published); the device's progress is a source, so the order is tested
// without a GPU; no GPU timing, memory collection or asynchronous replay.
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu.h"
#include "oot2/kernel.h"
#include "oot2/runtime.h"
#include "plat.h"

#define WAIT_SLICE_NS 100000000ull       // the completion thread asks the source again at least this often
#define COMPLETION_STACK (1ull << 20)

typedef struct {
    GpuEffect fn;
    void *arg;
} Effect;

typedef struct Sub {
    struct Sub *next;
    uint64_t value;
    Effect *effects;
    int n, cap;
} Sub;

static Sub *g_open;                               // the open submission, written by whoever holds the gate

static PlatMutex g_lock = PLAT_MUTEX_INIT;        // everything below, shared with the completion thread
static PlatCond g_gate_free = PLAT_COND_INIT;     // the gate opened
static int g_gate_busy;                           // one submission at a time, from gpu_sync_begin to gpu_sync_end
static PlatCond g_work = PLAT_COND_INIT;          // a value issued, or stop
static PlatCond g_changed = PLAT_COND_INIT;       // a value done, or the thread ended
static uint64_t g_handed, g_issued, g_done;       // written under g_lock, and read without it too
static Sub *g_head, **g_tail = &g_head;           // issued and not done, oldest first
static Sub *g_free;                               // done, kept for the next submissions
static GpuTimelineSource g_src;
static PlatThread *g_thread;
static int g_running, g_stop, g_ended;

static void completion(void *arg) {
    (void)arg;
    plat_mutex_lock(&g_lock);
    for (;;) {
        while (!g_head && !g_stop) plat_cond_wait(&g_work, &g_lock);
        if (!g_head) break;   // stopping, and nothing issued is left
        uint64_t want = g_head->value;
        plat_mutex_unlock(&g_lock);
        uint64_t reached = g_src.wait(g_src.ctx, want, WAIT_SLICE_NS);
        plat_mutex_lock(&g_lock);
        while (g_head && g_head->value <= reached) {
            Sub *s = g_head;
            g_head = s->next;
            if (!g_head) g_tail = &g_head;
            plat_mutex_unlock(&g_lock);
            for (int i = 0; i < s->n; i++) s->effects[i].fn(s->effects[i].arg, s->value);
            plat_mutex_lock(&g_lock);
            __atomic_store_n(&g_done, s->value, __ATOMIC_RELEASE);
            plat_cond_broadcast(&g_changed);
            s->next = g_free;
            g_free = s;
        }
    }
    g_ended = 1;
    plat_cond_broadcast(&g_changed);
    plat_mutex_unlock(&g_lock);
}

void gpu_sync_start(const GpuTimelineSource *src, uint64_t after) {
    if (!src || !src->wait) rt_fatal("gpu: the timeline has no source");
    plat_mutex_lock(&g_lock);
    if (g_running) rt_fatal("gpu: the completion thread is started twice");
    g_src = *src;
    g_handed = g_issued = after;
    __atomic_store_n(&g_done, after, __ATOMIC_RELEASE);
    g_stop = g_ended = 0;
    g_gate_busy = 0;
    g_running = 1;
    g_thread = plat_thread_start(completion, NULL, COMPLETION_STACK);
    if (!g_thread) rt_fatal("gpu: cannot start the completion thread");
    plat_mutex_unlock(&g_lock);
}

void gpu_sync_stop(void) {
    plat_mutex_lock(&g_lock);
    if (!g_running) rt_fatal("gpu: the completion thread is stopped but was not started");
    if (g_done != g_handed)
        rt_fatal("gpu: the completion thread is stopped with values %llu to %llu in flight",
                 (unsigned long long)g_done + 1, (unsigned long long)g_handed);
    g_stop = 1;
    plat_cond_broadcast(&g_work);
    while (!g_ended) plat_cond_wait(&g_changed, &g_lock);
    g_running = 0;
    plat_mutex_unlock(&g_lock);
    plat_thread_free(g_thread);
    g_thread = NULL;
}

uint64_t gpu_sync_begin(void) {
    int token = 0;
    plat_mutex_lock(&g_lock);
    if (!g_running) rt_fatal("gpu: a submission while the completion thread is not running");
    if (g_gate_busy) {
        // The submission that holds the gate may be waiting for a virtual core (the recorder's begin gives its own up while
        // it waits for an older frame, and takes one again before it goes on): a guest thread that waited here with its
        // core would keep that core from it for ever. So it waits without one, and takes one again once the gate is its.
        plat_mutex_unlock(&g_lock);
        token = rt_kernel_block_begin();
        plat_mutex_lock(&g_lock);
        while (g_gate_busy) plat_cond_wait(&g_gate_free, &g_lock);
    }
    g_gate_busy = 1;
    Sub *s = g_free;
    if (s) g_free = s->next;
    uint64_t value = g_handed + 1;
    __atomic_store_n(&g_handed, value, __ATOMIC_RELEASE);
    plat_mutex_unlock(&g_lock);
    if (!s && !(s = calloc(1, sizeof *s))) rt_fatal("gpu: out of memory for a submission");
    s->next = NULL;
    s->value = value;
    s->n = 0;
    __atomic_store_n(&g_open, s, __ATOMIC_RELEASE);
    rt_kernel_block_end(token);   // the submission is open before the core is waited for
    return value;
}

void gpu_sync_effect(GpuEffect fn, void *arg) {
    Sub *s = g_open;
    if (!s) rt_fatal("gpu: an effect outside a submission");
    if (!fn) rt_fatal("gpu: an effect with no function, for value %llu", (unsigned long long)s->value);
    if (s->n == s->cap) {
        int cap = s->cap ? 2 * s->cap : 8;
        Effect *e = realloc(s->effects, (size_t)cap * sizeof *e);
        if (!e) rt_fatal("gpu: out of memory for %d effects of value %llu", cap, (unsigned long long)s->value);
        s->effects = e, s->cap = cap;
    }
    s->effects[s->n++] = (Effect){fn, arg};
}

void gpu_sync_end(void) {
    Sub *s = g_open;
    if (!s) rt_fatal("gpu: the end of a submission that was not begun");
    __atomic_store_n(&g_open, NULL, __ATOMIC_RELEASE);
    plat_mutex_lock(&g_lock);
    *g_tail = s;
    g_tail = &s->next;
    __atomic_store_n(&g_issued, s->value, __ATOMIC_RELEASE);
    plat_cond_broadcast(&g_work);
    g_gate_busy = 0;
    plat_cond_signal(&g_gate_free);
    plat_mutex_unlock(&g_lock);
}

int gpu_sync_wait(uint64_t value, uint64_t timeout_ns) {
    if (value <= __atomic_load_n(&g_done, __ATOMIC_ACQUIRE)) return 1;
    plat_mutex_lock(&g_lock);
    uint64_t start = plat_now_ns();
    while (value <= g_handed && g_done < value) {
        if (timeout_ns == UINT64_MAX) {
            plat_cond_wait(&g_changed, &g_lock);
            continue;
        }
        uint64_t spent = plat_now_ns() - start;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_changed, &g_lock, timeout_ns - spent);
    }
    int ok = g_done >= value;
    plat_mutex_unlock(&g_lock);
    return ok;
}

uint64_t gpu_sync_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }

uint64_t gpu_sync_handed(void) { return __atomic_load_n(&g_handed, __ATOMIC_ACQUIRE); }

uint64_t gpu_sync_issued(void) { return __atomic_load_n(&g_issued, __ATOMIC_ACQUIRE); }

// The open submission's value. The record is read once and its value is final from gpu_sync_begin on; a submission that
// ends while it is read leaves the record on the free list, where its value is stale, so the answer is only used
// together with gpu_sync_handed: it names the open submission when it equals the newest value handed out.
uint64_t gpu_sync_open(void) {
    const Sub *s = __atomic_load_n(&g_open, __ATOMIC_ACQUIRE);
    return s ? __atomic_load_n(&s->value, __ATOMIC_RELAXED) : 0;
}

void gpu_sync_finish(void) {
    plat_mutex_lock(&g_lock);
    uint64_t target = g_handed;
    while (g_done < target) plat_cond_wait(&g_changed, &g_lock);
    plat_mutex_unlock(&g_lock);
}
