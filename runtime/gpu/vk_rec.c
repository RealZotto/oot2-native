// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The recorder thread, its frames and the breadcrumbs (oot2/vk_rec.h); the packets are vk_rec_pack.c and the ring is
// gpu_ring.c.
//
// A frame is a submission. vkrec_begin (the replaying thread) waits until the frame `frames` before it is done (with its
// virtual core given up, if it is a guest thread), takes the submission gate and the next value through the submit hooks,
// and puts a FRAME_BEGIN packet; the commands follow as packets; vkrec_end puts FRAME_END and returns. The recorder thread reads packets in order: FRAME_BEGIN gets the frame's
// command buffer (begun, with the checkpoint of its start), every command packet becomes its call, FRAME_END ends the
// command buffer and submits it, which also opens the gate. So the gate is held by the frame from the replay's first
// call to the recorder's submit, and a second thread that submits waits for it, as for any submission.
//
// Breadcrumbs. A draw's number and stage record are written by the replaying thread into a table of the frame (chunks of
// records made as the frame grows, kept for good), the packet carries the number, and the recorder sets the checkpoint
// GPU_VK_DRAW_MARKER(value, n) right before the call. A table is named by its frame's value and is reused when the ring of
// MAX_FRAMES + 1 tables comes round, which is `frames` + 1 frames later at the soonest, when the device has finished with
// it: a checkpoint can only name a frame that is not done, and a marker whose value no table holds any more is reported
// as the draw number alone.
//
// Design as in DKB Recomp's runtime/vk_rec.c at 9d8fa02 (GPL-3.0-or-later; there after Savage Wind Recomp #189 and LAS
// Recomp runtime/nvn/vulkan/record.c, MIT, Copyright (c) 2026 Peiikai): the replay thread packs, a recorder thread makes
// the calls and ends the command buffer, and the replay thread's wait is for the frame two before. Changed here: the
// recorder also submits (architecture 6.1: "packets to Vulkan commands, queue submit"), frames in flight are counted
// against the timeline's done value, the breadcrumb is a numbered record per draw that the loss report decodes (the NVIDIA fault
// study) where DKB writes a kind and a number, and the recorder has no direct fallback.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/kernel.h"
#include "oot2/nvn_program.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/vk_rec.h"
#include "plat.h"

#define CRUMB_CHUNK 4096u
#define CRUMB_CHUNKS ((GPU_VK_DRAW_MAX + CRUMB_CHUNK) / CRUMB_CHUNK)
#define MAX_FRAMES VKREC_MAX_FRAMES
#define DEFAULT_FRAMES 2u   // the default of the setting frames_in_flight (oot2/settings.h)
#define SLOTS (MAX_FRAMES + 1)
#define RECORDER_STACK (8ull << 20)   // reserved, committed as the driver's calls grow it
#define POLL_NS 50000000ull   // the recorder looks at its stop flag this often when idle

int vkrec_crumb_skew;
int vkrec_no_limit;

typedef struct {
    uint64_t value;                       // the frame the table holds, 0 when none
    uint32_t count;                       // draws written
    VkRecCrumb *chunk[CRUMB_CHUNKS];      // made by the replaying thread, read by anyone
} CrumbSlot;

typedef struct {
    uint64_t value, seq;
} FrameBegin;
typedef struct {
    uint32_t n;
    const VkSemaphoreSubmitInfo *waits;
} FrameEnd;

// 1 on the thread that has the open frame: commands, effects and the end are that thread's, whichever thread had the
// frame before, and a second frame begun by it is its own deadlock on the gate, so it is refused by name instead.
static _Thread_local int t_frame;

static struct {
    int open;
    GpuRing *ring;
    VkRecCalls calls;
    VkRecSubmit submit;
    uint32_t frames;
    PlatThread *thread;
    int stop, thread_done;
    // the replaying side, handed from one frame's thread to the next through the submission gate
    int frame_open;
    uint64_t started;                     // frames begun
    uint64_t values[MAX_FRAMES];          // the values of the frames begun last, by seq % MAX_FRAMES
    CrumbSlot *slot;                      // the open frame's table
    uint32_t draws;
    const struct nvn_stage *stages[3];
    uint64_t t_begin, cpu_begin, ring_wait_begin, packets_begin, words_begin;
    CrumbSlot slots[SLOTS];
    VkRecFrame log[VKREC_LOG];
} G;

// ---------- breadcrumbs
void vkrec_stages(const struct nvn_stage *geometry, const struct nvn_stage *pixel, const struct nvn_stage *task) {
    G.stages[0] = geometry, G.stages[1] = pixel, G.stages[2] = task;
}

uint32_t vkrec_crumb_next(uint8_t kind) {
    if (!t_frame) rt_fatal("gfx: vulkan: recorder: a draw outside a frame (vkrec_begin first)");
    if (!G.submit.checkpoints) return 0;
    if (G.draws >= GPU_VK_DRAW_MAX)
        rt_fatal("gfx: vulkan: recorder: a frame of more than %u draws, dispatches and mesh draws", GPU_VK_DRAW_MAX);
    uint32_t n = ++G.draws, index = n - 1, c = index / CRUMB_CHUNK;
    CrumbSlot *s = G.slot;
    VkRecCrumb *chunk = __atomic_load_n(&s->chunk[c], __ATOMIC_ACQUIRE);
    if (!chunk) {
        chunk = calloc(CRUMB_CHUNK, sizeof *chunk);
        if (!chunk) rt_fatal("gfx: vulkan: recorder: out of memory for the breadcrumbs of frame %llu",
                             (unsigned long long)s->value);
        __atomic_store_n(&s->chunk[c], chunk, __ATOMIC_RELEASE);
    }
    chunk[index % CRUMB_CHUNK] = (VkRecCrumb){kind, {G.stages[0], G.stages[1], G.stages[2]}};
    __atomic_store_n(&s->count, n, __ATOMIC_RELEASE);
    return n + (vkrec_crumb_skew ? 1u : 0u);
}

int vkrec_crumb(uint64_t marker, VkRecCrumb *out, uint64_t *value, uint32_t *n) {
    if (((marker >> 1) & 7) != GPU_VK_DRAW) return 0;
    uint64_t v = marker >> 28;
    uint32_t k = (uint32_t)((marker >> 4) & GPU_VK_DRAW_MAX);
    if (value) *value = v;
    if (n) *n = k;
    if (!G.open || !k) return 0;
    for (uint32_t i = 0; i < SLOTS; i++) {
        CrumbSlot *s = &G.slots[i];
        if (__atomic_load_n(&s->value, __ATOMIC_ACQUIRE) != v) continue;
        if (k > __atomic_load_n(&s->count, __ATOMIC_ACQUIRE)) return 0;
        const VkRecCrumb *chunk = __atomic_load_n(&s->chunk[(k - 1) / CRUMB_CHUNK], __ATOMIC_ACQUIRE);
        if (!chunk) return 0;
        *out = chunk[(k - 1) % CRUMB_CHUNK];
        if (__atomic_load_n(&s->value, __ATOMIC_ACQUIRE) != v) return 0;   // reused while it was read
        return 1;
    }
    return 0;
}

static const char *crumb_kind(uint8_t k) {
    switch (k) {
    case 'd': return "draw";
    case 'i': return "indexed draw";
    case 'I': return "indirect draw";
    case 'J': return "indexed indirect draw";
    case 'm': return "mesh draw";
    case 'M': return "indirect mesh draw";
    case 'c': return "dispatch";
    case 'C': return "indirect dispatch";
    default: return "draw of an unknown kind";
    }
}

int vkrec_marker_text(uint64_t marker, char *out, size_t cap) {
    VkRecCrumb c;
    uint64_t value = 0;
    uint32_t n = 0;
    int found = vkrec_crumb(marker, &c, &value, &n);
    if (((marker >> 1) & 7) != GPU_VK_DRAW) {
        snprintf(out, cap, "not a draw's marker");
        return 0;
    }
    if (!found) {
        snprintf(out, cap, "draw %u of submission %llu (its stages are no longer known)", n, (unsigned long long)value);
        return 0;
    }
    size_t at = (size_t)snprintf(out, cap, "draw %u of submission %llu (%s):", n, (unsigned long long)value,
                                 crumb_kind(c.kind));
    int any = 0;
    for (int i = 0; i < 3; i++) {
        if (!c.id[i] || at >= cap) continue;
        const char *kind = nvn_stage_kind_name(c.id[i]->kind);
        at += (size_t)snprintf(out + at, cap - at, "%s %s %s", any ? "," : "", kind ? kind : "stage", c.id[i]->key);
        any = 1;
    }
    if (!any && at < cap) snprintf(out + at, cap - at, " no stage identity was set");
    return 1;
}

// ---------- the recorder thread
static void recorder_main(void *arg) {
    (void)arg;
    VkRecExec x = {.calls = &G.calls, .checkpoints = G.submit.checkpoints};
    uint64_t seq = 0, t0 = 0, cpu0 = 0;
    for (;;) {
        uint32_t op;
        size_t bytes;
        const void *p = gpu_ring_get(G.ring, &op, &bytes, POLL_NS);
        if (!p) {
            if (__atomic_load_n(&G.stop, __ATOMIC_ACQUIRE)) break;
            continue;
        }
        (void)bytes;
        if (op == VKR_FRAME_BEGIN) {
            const FrameBegin *f = p;
            seq = f->seq;
            x.value = f->value;
            t0 = plat_now_ns();
            cpu0 = plat_thread_cpu_ns(NULL);
            x.cb = G.submit.cmd();
        } else if (op == VKR_FRAME_END) {
            const FrameEnd *f = p;
            G.submit.end(f->waits, f->n);
            VkRecFrame *row = &G.log[seq % VKREC_LOG];
            row->record_ns = plat_now_ns() - t0;
            row->record_cpu_ns = plat_thread_cpu_ns(NULL) - cpu0;
            x.cb = VK_NULL_HANDLE;
        } else {
            vkrec_exec(&x, op, p);
        }
        gpu_ring_release(G.ring);
    }
    __atomic_store_n(&G.thread_done, 1, __ATOMIC_RELEASE);
}

// ---------- the submit hooks of the device
static uint64_t device_begin(int kind) { return gpu_vk_begin(kind); }
static VkCommandBuffer device_cmd(void) { return gpu_vk_cmd(); }
static uint64_t device_end(const VkSemaphoreSubmitInfo *waits, uint32_t n) { return gpu_vk_end(waits, n); }

void vkrec_submit_device(VkRecSubmit *out) {
    *out = (VkRecSubmit){device_begin, device_cmd, device_end, gpu_vk.checkpoints};
}

void vkrec_open(const VkRecOpen *o) {
    if (G.open) rt_fatal("gfx: vulkan: the recorder is opened twice");
    uint32_t frames = o && o->frames ? o->frames : (uint32_t)rt_setting(RT_SET_frames_in_flight);
    if (frames > MAX_FRAMES) rt_fatal("gfx: vulkan: the recorder allows %u frames in flight at most, not %u", MAX_FRAMES, frames);
    memset(&G, 0, sizeof G);
    G.frames = frames;
    G.ring = gpu_ring_new(o && o->ring_bytes ? o->ring_bytes : GPU_RING_BYTES);
    if (!G.ring) rt_fatal("gfx: vulkan: cannot make the recorder's ring");
    if (o && o->calls) G.calls = *o->calls;
    else vkrec_calls_device(&G.calls);
    if (o && o->submit) G.submit = *o->submit;
    else vkrec_submit_device(&G.submit);
    G.open = 1;
    if (frames != DEFAULT_FRAMES) rt_log("gfx: vulkan: recorder: %u frames in flight (the default is %u)", frames, DEFAULT_FRAMES);
    G.thread = plat_thread_start(recorder_main, NULL, RECORDER_STACK);
    if (!G.thread) rt_fatal("gfx: vulkan: cannot start the recorder thread");
}

void vkrec_idle(void) {
    if (!G.open) return;
    gpu_ring_drain(G.ring);
}

void vkrec_close(void) {
    if (!G.open) return;
    if (G.frame_open) rt_fatal("gfx: vulkan: the recorder is closed inside a frame");
    vkrec_idle();
    __atomic_store_n(&G.stop, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&G.thread_done, __ATOMIC_ACQUIRE)) plat_yield();
    plat_thread_free(G.thread);
    for (uint32_t i = 0; i < SLOTS; i++)
        for (uint32_t c = 0; c < CRUMB_CHUNKS; c++) free(G.slots[i].chunk[c]);
    gpu_ring_free(G.ring);
    G.ring = NULL;
    G.thread = NULL;
    G.open = 0;
}

int vkrec_is_open(void) { return G.open; }

// ---------- the replay's side
void *vkrec_put(uint32_t op, size_t bytes) {
    if (!t_frame) rt_fatal("gfx: vulkan: recorder: a command outside a frame (vkrec_begin first)");
    return gpu_ring_put(G.ring, op, bytes);
}

void vkrec_publish(void) { gpu_ring_publish(G.ring); }

uint64_t vkrec_begin(int kind) {
    if (!G.open) rt_fatal("gfx: vulkan: a frame before the recorder is open");
    if (t_frame) rt_fatal("gfx: vulkan: recorder: a frame is begun inside a frame");
    // The gate first: everything below it is the replaying side's own state, which a frame hands to the next one only
    // through the gate. The wait for the frame `frames` before is made holding it, and cannot deadlock: the gate is ours
    // only once the recorder has submitted the frame before, so the frame `frames` before is submitted too.
    uint64_t g0 = plat_now_ns();
    uint64_t value = G.submit.begin(kind);   // held until the recorder has submitted this frame
    uint64_t gate_wait = plat_now_ns() - g0, waited = 0;
    if (G.started >= G.frames && !vkrec_no_limit) {
        uint64_t t0 = plat_now_ns(), older = G.values[(G.started - G.frames) % MAX_FRAMES];
        if (older > gpu_sync_done()) {
            // A guest thread gives its virtual core up for the wait, as for every wait on the host (D7): the wait ends when
            // the device and the completion thread are through, which no guest thread has a part in. A thread that is
            // none (a test's, the start-up's) holds no core, and the pair is then nothing.
            int token = rt_kernel_block_begin();
            gpu_sync_wait(older, UINT64_MAX);
            rt_kernel_block_end(token);
        }
        waited = plat_now_ns() - t0;
    }
    uint64_t seq = G.started++;
    G.values[seq % MAX_FRAMES] = value;
    G.slot = &G.slots[seq % SLOTS];
    __atomic_store_n(&G.slot->count, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&G.slot->value, value, __ATOMIC_RELEASE);
    G.draws = 0;
    VkRecFrame *row = &G.log[seq % VKREC_LOG];
    memset(row, 0, sizeof *row);
    row->value = value;
    row->gate_wait_ns = gate_wait;
    row->slot_wait_ns = waited;
    GpuRingStats st;
    gpu_ring_stats(G.ring, &st);
    G.ring_wait_begin = st.full_wait_ns, G.packets_begin = st.packets, G.words_begin = st.words;
    G.frame_open = t_frame = 1;
    G.t_begin = plat_now_ns();
    G.cpu_begin = plat_thread_cpu_ns(NULL);
    FrameBegin *f = gpu_ring_put(G.ring, VKR_FRAME_BEGIN, sizeof *f);
    *f = (FrameBegin){value, seq};
    gpu_ring_publish(G.ring);
    return value;
}

void vkrec_effect(GpuEffect fn, void *arg) {
    if (!t_frame) rt_fatal("gfx: vulkan: recorder: an effect outside a frame");
    gpu_sync_effect(fn, arg);
}

uint64_t vkrec_end(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits) {
    if (!t_frame) rt_fatal("gfx: vulkan: recorder: the end of a frame that this thread did not begin");
    for (uint32_t i = 0; i < nwaits; i++)
        if (waits[i].pNext) rt_fatal("gfx: vulkan: recorder: a wait of a frame with a pNext chain has no packet");
    size_t bytes = sizeof(VkSemaphoreSubmitInfo) * nwaits;
    FrameEnd *f = gpu_ring_put(G.ring, VKR_FRAME_END, sizeof *f + bytes);
    VkSemaphoreSubmitInfo *copy = (VkSemaphoreSubmitInfo *)(f + 1);
    if (nwaits) memcpy(copy, waits, bytes);
    *f = (FrameEnd){nwaits, nwaits ? copy : NULL};
    G.frame_open = t_frame = 0;
    // The recorder opens the gate once it has submitted, and another thread may begin the next frame at once and write
    // everything above: this frame's row is filled before the packet that ends it is published.
    uint64_t cpu = plat_thread_cpu_ns(NULL) - G.cpu_begin, t = plat_now_ns() - G.t_begin;
    VkRecFrame *row = &G.log[(G.started - 1) % VKREC_LOG];
    GpuRingStats st;
    gpu_ring_stats(G.ring, &st);
    uint64_t ring_wait = st.full_wait_ns - G.ring_wait_begin, value = row->value;
    row->ring_wait_ns = ring_wait;
    row->replay_ns = t > ring_wait ? t - ring_wait : 0;
    row->replay_cpu_ns = cpu;
    row->draws = G.draws;
    row->packets = (uint32_t)(st.packets - G.packets_begin);
    row->bytes = (st.words - G.words_begin) * 8;
    gpu_ring_publish(G.ring);
    return value;
}

// ---------- numbers
const VkRecFrame *vkrec_frames(uint32_t *n) {
    static VkRecFrame ordered[VKREC_LOG];
    uint64_t count = G.started < VKREC_LOG ? G.started : VKREC_LOG;
    for (uint64_t i = 0; i < count; i++) ordered[i] = G.log[(G.started - count + i) % VKREC_LOG];
    *n = (uint32_t)count;
    return ordered;
}

void vkrec_ring_stats(GpuRingStats *out) {
    if (G.ring) gpu_ring_stats(G.ring, out);
    else memset(out, 0, sizeof *out);
}
