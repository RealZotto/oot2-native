// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the recorder (oot2/gpu_ring.h, oot2/vk_rec.h; runtime/gpu/gpu_ring.c, vk_rec.c, vk_rec_pack.c), driven by
// tests/test_runtime_vk_rec.py: `vk_rec_test <case> [args] [--mutant NAME]`, one case per process. A passing case prints
// "ok <case>"; a failing check exits 1 with the line, a named fatal exits 70.
//
// Cases that need no device (continuous integration runs them): the ring and the recorder over a test's own calls, a
// counter standing in for the device's timeline.
//   ring_wrap          150,000 packets of 0 to 39 words through a 4 KiB ring: in order, intact, padded, back-pressured
//   ring_oversize      a packet bigger than a quarter of the ring: a named fatal
//   pack_roundtrip     every command of oot2/vk_rec.h as a packet, made into the call with the arguments it was given,
//                      from storage the caller had already overwritten
//   pack_pnext         a structure with a pNext chain: a named fatal
//   frame_order        two threads begin and end 600 frames: values in order, no frame's packets inside another's
//   frame_limit        a third frame waits for the first to be done
//   window WHICH       a window of 2 (default), 7, 8 (the most) or the variable OOT2_FRAMES_IN_FLIGHT (setting) frames
//                      in flight: each begin finds the frame `frames` before done and its results in its buffer, never an
//                      older frame's; the window fills to exactly `frames`; every table of breadcrumbs of a frame in
//                      flight is its own; a guest wait on a frame reads that frame's results
//   frames_over        more frames than the most: a named fatal
//   frames_range       the setting frames_in_flight: 1 to the most, 2 by default
//   frames_setting     the value the store gives (a variable out of the range ends the process by name)
//   core_wait          a guest thread's wait for an older frame gives its virtual core up (two threads, one core)
//   core_gate          so does a guest thread that waits for the gate behind it
//   core_lock          so does one that waits for a host mutex which a thread that waits for the gate holds
//   breadcrumbs        every draw's marker names its own record; a frame's records are gone once it is reused
//   breadcrumbs_off    no marker, no record
//   marker_text        the report's words for a marker
// Cases that need a device with every item of oot2/vk_required.h (the desktop's RTX 3060 Ti): synthetic pipelines on the
// uni layout from GLSL modules in the folder given (tests/test_runtime_vk_rec.py compiles them).
//   dev_usable         how many devices have every item (exit 3 when there is no loader or device)
//   dev_draws FOLDER   3,000 draws through the ring and as direct calls, every cell of the target checked
//   dev_third FOLDER   a third frame waits for the first, which the device holds
//   dev_loss FOLDER DIR an injected loss: the report names the last draw by its stages
//   dev_timing FOLDER  the submitting thread's and the recorder's time per frame, ring and direct, 300 frames after 30
// --mutant NAME: overwrite (a ring that does not wait for room), nolimit (a third frame let through), skew (a marker one
// draw past its record), narrow (the window case opens the recorder with one frame fewer than the device expects: the stall
// detector must fire). The check of the case must then fail, which the driver requires.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "endmark.h"
#include "layout.h"
#include "oot2/gpu.h"
#include "oot2/gpu_ring.h"
#include "oot2/gpu_vk.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/nvn_program.h"
#include "oot2/replay_count.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_rec.h"
#include "kernel_lock.h"
#include "plat.h"

#define CHECK(cond)                                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

static uint64_t next_random(uint64_t *s) {
    *s ^= *s << 13, *s ^= *s >> 7, *s ^= *s << 17;
    return *s;
}

static PlatMutex g_join_lock = PLAT_MUTEX_INIT;
static PlatCond g_join_cond = PLAT_COND_INIT;
static int g_ended;
static void ended(void) {
    plat_mutex_lock(&g_join_lock);
    g_ended++;
    plat_cond_broadcast(&g_join_cond);
    plat_mutex_unlock(&g_join_lock);
}
static void join(int n) {
    plat_mutex_lock(&g_join_lock);
    while (g_ended < n) plat_cond_wait(&g_join_cond, &g_join_lock);
    plat_mutex_unlock(&g_join_lock);
}

// ---------- the ring
#define RING_PACKETS 150000
static GpuRing *g_ring;

static void ring_producer(void *arg) {
    (void)arg;
    uint64_t seed = 99;
    for (uint64_t i = 1; i <= RING_PACKETS; i++) {
        size_t words = next_random(&seed) % 40;
        uint64_t *p = gpu_ring_put(g_ring, 1 + (uint32_t)(i % 7), words * 8);
        for (size_t w = 0; w < words; w++) p[w] = i * 0x9E3779B97F4A7C15ull + w;
        gpu_ring_publish(g_ring);
    }
    ended();
}

static void ring_consumer(void *arg) {
    (void)arg;
    uint64_t seed = 99;
    for (uint64_t i = 1; i <= RING_PACKETS; i++) {
        uint32_t op;
        size_t bytes;
        const uint64_t *p = gpu_ring_get(g_ring, &op, &bytes, UINT64_MAX);
        size_t words = next_random(&seed) % 40;
        CHECK(p && op == 1 + (uint32_t)(i % 7) && bytes == words * 8);
        for (size_t w = 0; w < words; w++) CHECK(p[w] == i * 0x9E3779B97F4A7C15ull + w);
        gpu_ring_release(g_ring);
        if (i % 4096 == 0) plat_sleep_ns(300000);   // the consumer falls behind and the ring fills
        else if (i % 64 == 0) plat_yield();
    }
    uint32_t op;
    size_t bytes;
    CHECK(gpu_ring_get(g_ring, &op, &bytes, 0) == NULL);   // an empty look publishes the tail, which a drain waits for
    ended();
}

static void case_ring_wrap(void) {
    g_ring = gpu_ring_new(4096);
    CHECK(g_ring && gpu_ring_bytes(g_ring) == 4096 && gpu_ring_max_payload(g_ring) == 1016);
    CHECK(plat_thread_start(ring_consumer, NULL, 1 << 20));
    CHECK(plat_thread_start(ring_producer, NULL, 1 << 20));
    join(2);
    gpu_ring_drain(g_ring);
    GpuRingStats st;
    gpu_ring_stats(g_ring, &st);
    CHECK(st.packets == RING_PACKETS && st.pads > 0 && st.full_waits > 0);
    printf("ok ring_wrap packets %llu words %llu pads %llu full_waits %llu sleeps %llu\n",
           (unsigned long long)st.packets, (unsigned long long)st.words, (unsigned long long)st.pads,
           (unsigned long long)st.full_waits, (unsigned long long)st.consumer_sleeps);
    gpu_ring_free(g_ring);
}

static void case_ring_oversize(void) {
    GpuRing *r = gpu_ring_new(4096);
    CHECK(r);
    gpu_ring_put(r, 1, 1016);
    gpu_ring_put(r, 1, 1017);
    printf("a packet bigger than a quarter of the ring was accepted\n");
}

// ---------- a test's timeline and submissions
static uint64_t g_counter;
static PlatMutex g_counter_lock = PLAT_MUTEX_INIT;
static PlatCond g_counter_cond = PLAT_COND_INIT;
static uint64_t counter_now(void) { return __atomic_load_n(&g_counter, __ATOMIC_ACQUIRE); }
static void counter_raise(uint64_t v) {
    plat_mutex_lock(&g_counter_lock);
    if (v > g_counter) g_counter = v;
    plat_cond_broadcast(&g_counter_cond);
    plat_mutex_unlock(&g_counter_lock);
}
static uint64_t counter_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    uint64_t t0 = plat_now_ns();
    plat_mutex_lock(&g_counter_lock);
    while (g_counter < value) {
        uint64_t spent = plat_now_ns() - t0;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_counter_cond, &g_counter_lock, timeout_ns - spent);
    }
    uint64_t r = g_counter;
    plat_mutex_unlock(&g_counter_lock);
    return r;
}
static void start_counter(void) {
    g_counter = 0;
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
}

#define FAKE_CB ((VkCommandBuffer)(uintptr_t)0xC0FFEE00u)
static uint64_t g_cur_value;             // the open frame's, set by the thread that began it
static int g_hold;                       // the recorder waits in cmd() while set
static uint64_t g_frames_ended, g_end_waits;
static void (*g_on_cmd)(uint64_t value);
static void (*g_on_end)(uint64_t value);

static uint64_t fake_begin(int kind) {
    (void)kind;
    g_cur_value = gpu_sync_begin();
    return g_cur_value;
}
static VkCommandBuffer fake_cmd(void) {
    while (__atomic_load_n(&g_hold, __ATOMIC_ACQUIRE)) plat_yield();
    if (g_on_cmd) g_on_cmd(g_cur_value);
    return FAKE_CB;
}
static uint64_t fake_end(const VkSemaphoreSubmitInfo *waits, uint32_t n) {
    uint64_t v = g_cur_value;
    g_end_waits = n;
    for (uint32_t i = 0; i < n; i++) CHECK(waits[i].value == 40 + i && waits[i].semaphore == (VkSemaphore)(uintptr_t)(0x500 + i));
    if (g_on_end) g_on_end(v);
    g_frames_ended++;
    gpu_sync_end();
    return v;
}

typedef struct {
    uint8_t *p;
    size_t n, cap;
} Buf;
static Buf g_got, g_want;
static void bput(Buf *b, const void *p, size_t n) {
    if (!n) return;
    if (b->n + n > b->cap) {
        b->cap = (b->n + n) * 2;
        b->p = realloc(b->p, b->cap);
        CHECK(b->p);
    }
    memcpy(b->p + b->n, p, n);
    b->n += n;
}
#define BPUT(b, x)                                                                                                     \
    ({                                                                                                                 \
        __typeof__(x) t_ = (x);                                                                                        \
        bput((b), &t_, sizeof t_);                                                                                     \
    })
static void ser_op(Buf *b, uint32_t op) { bput(b, &op, sizeof op); }

static void fill_bytes(void *p, size_t n, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull + 12345;
    for (size_t i = 0; i < n; i++) ((uint8_t *)p)[i] = (uint8_t)(next_random(&s) >> 24);
}

// ---------- the fakes of the commands with scalar arguments, and a sender for each, both made from the command list
#define DROP1(first, ...) __VA_ARGS__
#define FAKE_F(name, params, packargs, callargs, ...)                                                                  \
    typedef struct __attribute__((packed)) {                                                                           \
        __VA_ARGS__                                                                                                    \
    } S_##name;                                                                                                        \
    static void VKAPI_CALL fake_##name(VkCommandBuffer cb, VKREC_UNPAREN params) {                                     \
        CHECK(cb == FAKE_CB);                                                                                          \
        S_##name got = {VKREC_UNPAREN packargs};                                                                       \
        ser_op(&g_got, VKR_##name);                                                                                    \
        bput(&g_got, &got, sizeof got);                                                                                \
    }                                                                                                                  \
    static void send_##name(uint64_t seed) {                                                                           \
        S_##name store;                                                                                                \
        fill_bytes(&store, sizeof store, seed);                                                                        \
        const S_##name *a = &store;                                                                                    \
        ser_op(&g_want, VKR_##name);                                                                                   \
        bput(&g_want, &store, sizeof store);                                                                           \
        vkrec_##name(DROP1 callargs);                                                                                  \
    }
#define FAKE_D(name, kind, params, packargs, callargs, ...) FAKE_F(name, params, packargs, callargs, __VA_ARGS__)
VKREC_FIXED(FAKE_F)
VKREC_DRAWS(FAKE_D)

static void (*const g_senders[])(uint64_t) = {
#define LIST_F(name, ...) send_##name,
#define LIST_D(name, kind, ...) send_##name,
    VKREC_FIXED(LIST_F) VKREC_DRAWS(LIST_D)
#undef LIST_F
#undef LIST_D
};

// ---------- the commands with arrays: one serialisation used for the arguments as given and as received
static void ser_BindDescriptorSets(Buf *b, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n,
                                   const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    ser_op(b, VKR_BindDescriptorSets);
    BPUT(b, bp), BPUT(b, l), BPUT(b, first), BPUT(b, n), BPUT(b, nd);
    bput(b, s, sizeof *s * n);
    bput(b, d, sizeof *d * nd);
}
static void VKAPI_CALL fake_BindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                               uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd,
                                               const uint32_t *d) {
    CHECK(cb == FAKE_CB);
    ser_BindDescriptorSets(&g_got, bp, l, first, n, s, nd, d);
}

static const void *write_infos(const VkWriteDescriptorSet *w, size_t *size) {
    switch (w->descriptorType) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        *size = sizeof(VkDescriptorBufferInfo) * w->descriptorCount;
        return w->pBufferInfo;
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        *size = sizeof(VkBufferView) * w->descriptorCount;
        return w->pTexelBufferView;
    default:
        *size = sizeof(VkDescriptorImageInfo) * w->descriptorCount;
        return w->pImageInfo;
    }
}
static void ser_Push(Buf *b, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t set, uint32_t n,
                     const VkWriteDescriptorSet *w) {
    ser_op(b, VKR_PushDescriptorSetKHR);
    BPUT(b, bp), BPUT(b, l), BPUT(b, set), BPUT(b, n);
    for (uint32_t i = 0; i < n; i++) {
        BPUT(b, w[i].dstBinding), BPUT(b, w[i].dstArrayElement), BPUT(b, w[i].descriptorCount),
            BPUT(b, w[i].descriptorType);
        size_t size;
        const void *infos = write_infos(&w[i], &size);
        CHECK(infos && w[i].pNext == NULL);
        bput(b, infos, size);
    }
}
static void VKAPI_CALL fake_PushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                                 uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    CHECK(cb == FAKE_CB);
    ser_Push(&g_got, bp, l, set, n, w);
}

static void ser_PushConstants(Buf *b, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t size, const void *v) {
    ser_op(b, VKR_PushConstants);
    BPUT(b, l), BPUT(b, st), BPUT(b, off), BPUT(b, size);
    bput(b, v, size);
}
static void VKAPI_CALL fake_PushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off,
                                          uint32_t size, const void *v) {
    CHECK(cb == FAKE_CB);
    ser_PushConstants(&g_got, l, st, off, size, v);
}

static void ser_BindVertexBuffers2(Buf *b, uint32_t first, uint32_t n, const VkBuffer *buf, const VkDeviceSize *off,
                                   const VkDeviceSize *sz, const VkDeviceSize *st) {
    ser_op(b, VKR_BindVertexBuffers2);
    BPUT(b, first), BPUT(b, n);
    uint8_t has = (uint8_t)((sz ? 1 : 0) | (st ? 2 : 0));
    BPUT(b, has);
    bput(b, buf, sizeof *buf * n);
    bput(b, off, sizeof *off * n);
    bput(b, sz, sz ? sizeof *sz * n : 0);
    bput(b, st, st ? sizeof *st * n : 0);
}
static void VKAPI_CALL fake_BindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *buf,
                                               const VkDeviceSize *off, const VkDeviceSize *sz, const VkDeviceSize *st) {
    CHECK(cb == FAKE_CB);
    ser_BindVertexBuffers2(&g_got, first, n, buf, off, sz, st);
}

#define FAKE_ARRAY(name, ptype, ...)                                                                                   \
    static void ser_##name(Buf *b, uint32_t first, uint32_t n, const ptype *v) {                                        \
        ser_op(b, VKR_##name);                                                                                         \
        BPUT(b, first), BPUT(b, n);                                                                                    \
        bput(b, v, sizeof *v * n);                                                                                     \
    }                                                                                                                  \
    static void VKAPI_CALL fake_##name(VkCommandBuffer cb, __VA_ARGS__) {                                              \
        CHECK(cb == FAKE_CB);                                                                                          \
        ser_##name(&g_got, first, n, v);                                                                               \
    }
FAKE_ARRAY(SetColorBlendEnableEXT, VkBool32, uint32_t first, uint32_t n, const VkBool32 *v)
FAKE_ARRAY(SetColorBlendEquationEXT, VkColorBlendEquationEXT, uint32_t first, uint32_t n, const VkColorBlendEquationEXT *v)
FAKE_ARRAY(SetColorWriteMaskEXT, VkColorComponentFlags, uint32_t first, uint32_t n, const VkColorComponentFlags *v)

static void ser_Viewports(Buf *b, uint32_t n, const VkViewport *v) {
    ser_op(b, VKR_SetViewportWithCount);
    BPUT(b, n);
    bput(b, v, sizeof *v * n);
}
static void VKAPI_CALL fake_SetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport *v) {
    CHECK(cb == FAKE_CB);
    ser_Viewports(&g_got, n, v);
}
static void ser_Scissors(Buf *b, uint32_t n, const VkRect2D *r) {
    ser_op(b, VKR_SetScissorWithCount);
    BPUT(b, n);
    bput(b, r, sizeof *r * n);
}
static void VKAPI_CALL fake_SetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D *r) {
    CHECK(cb == FAKE_CB);
    ser_Scissors(&g_got, n, r);
}

static void ser_VertexInput(Buf *b, uint32_t nb, const VkVertexInputBindingDescription2EXT *bd, uint32_t na,
                            const VkVertexInputAttributeDescription2EXT *ad) {
    ser_op(b, VKR_SetVertexInputEXT);
    BPUT(b, nb), BPUT(b, na);
    bput(b, bd, sizeof *bd * nb);
    bput(b, ad, sizeof *ad * na);
}
static void VKAPI_CALL fake_SetVertexInputEXT(VkCommandBuffer cb, uint32_t nb, const VkVertexInputBindingDescription2EXT *bd,
                                              uint32_t na, const VkVertexInputAttributeDescription2EXT *ad) {
    CHECK(cb == FAKE_CB);
    ser_VertexInput(&g_got, nb, bd, na, ad);
}

static void ser_BlendConstants(Buf *b, const float *c) {
    ser_op(b, VKR_SetBlendConstants);
    bput(b, c, 4 * sizeof *c);
}
static void VKAPI_CALL fake_SetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    CHECK(cb == FAKE_CB);
    ser_BlendConstants(&g_got, c);
}

static void ser_Barrier(Buf *b, const VkDependencyInfo *d) {
    ser_op(b, VKR_PipelineBarrier2);
    BPUT(b, d->dependencyFlags), BPUT(b, d->memoryBarrierCount), BPUT(b, d->bufferMemoryBarrierCount),
        BPUT(b, d->imageMemoryBarrierCount);
    bput(b, d->pMemoryBarriers, sizeof(VkMemoryBarrier2) * d->memoryBarrierCount);
    bput(b, d->pBufferMemoryBarriers, sizeof(VkBufferMemoryBarrier2) * d->bufferMemoryBarrierCount);
    bput(b, d->pImageMemoryBarriers, sizeof(VkImageMemoryBarrier2) * d->imageMemoryBarrierCount);
}
static void VKAPI_CALL fake_PipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *d) {
    CHECK(cb == FAKE_CB);
    ser_Barrier(&g_got, d);
}

static void ser_Rendering(Buf *b, const VkRenderingInfo *r) {
    ser_op(b, VKR_BeginRendering);
    BPUT(b, r->flags), BPUT(b, r->renderArea), BPUT(b, r->layerCount), BPUT(b, r->viewMask),
        BPUT(b, r->colorAttachmentCount);
    uint8_t has = (uint8_t)((r->pDepthAttachment ? 1 : 0) | (r->pStencilAttachment ? 2 : 0));
    BPUT(b, has);
    bput(b, r->pColorAttachments, sizeof(VkRenderingAttachmentInfo) * r->colorAttachmentCount);
    if (r->pDepthAttachment) bput(b, r->pDepthAttachment, sizeof(VkRenderingAttachmentInfo));
    if (r->pStencilAttachment) bput(b, r->pStencilAttachment, sizeof(VkRenderingAttachmentInfo));
}
static void VKAPI_CALL fake_BeginRendering(VkCommandBuffer cb, const VkRenderingInfo *r) {
    CHECK(cb == FAKE_CB);
    ser_Rendering(&g_got, r);
}
static void VKAPI_CALL fake_EndRendering(VkCommandBuffer cb) {
    CHECK(cb == FAKE_CB);
    ser_op(&g_got, VKR_EndRendering);
}

static void ser_Clear(Buf *b, uint32_t na, const VkClearAttachment *a, uint32_t nr, const VkClearRect *r) {
    ser_op(b, VKR_ClearAttachments);
    BPUT(b, na), BPUT(b, nr);
    bput(b, a, sizeof *a * na);
    bput(b, r, sizeof *r * nr);
}
static void VKAPI_CALL fake_ClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr,
                                             const VkClearRect *r) {
    CHECK(cb == FAKE_CB);
    ser_Clear(&g_got, na, a, nr, r);
}

// The transfers: the handles and one region each, the images in the general layout.
static void ser_CopyBuffer(Buf *b, VkBuffer s, VkBuffer d, const VkBufferCopy *r) {
    ser_op(b, VKR_CopyBuffer);
    BPUT(b, s), BPUT(b, d);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_CopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) {
    CHECK(cb == FAKE_CB && n == 1);
    ser_CopyBuffer(&g_got, s, d, r);
}
static void ser_CopyBufferToImage(Buf *b, VkBuffer s, VkImage d, const VkBufferImageCopy *r) {
    ser_op(b, VKR_CopyBufferToImage);
    BPUT(b, s), BPUT(b, d);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_CopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout l, uint32_t n,
                                              const VkBufferImageCopy *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    ser_CopyBufferToImage(&g_got, s, d, r);
}
static void ser_CopyImageToBuffer(Buf *b, VkImage s, VkBuffer d, const VkBufferImageCopy *r) {
    ser_op(b, VKR_CopyImageToBuffer);
    BPUT(b, s), BPUT(b, d);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_CopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout l, VkBuffer d, uint32_t n,
                                              const VkBufferImageCopy *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    ser_CopyImageToBuffer(&g_got, s, d, r);
}
static void ser_CopyImage(Buf *b, VkImage s, VkImage d, const VkImageCopy *r) {
    ser_op(b, VKR_CopyImage);
    BPUT(b, s), BPUT(b, d);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_CopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                      const VkImageCopy *r) {
    CHECK(cb == FAKE_CB && n == 1 && sl == VK_IMAGE_LAYOUT_GENERAL && dl == VK_IMAGE_LAYOUT_GENERAL);
    ser_CopyImage(&g_got, s, d, r);
}
static void ser_BlitImage(Buf *b, VkImage s, VkImage d, const VkImageBlit *r, VkFilter f) {
    ser_op(b, VKR_BlitImage);
    BPUT(b, s), BPUT(b, d), BPUT(b, f);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_BlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                      const VkImageBlit *r, VkFilter f) {
    CHECK(cb == FAKE_CB && n == 1 && sl == VK_IMAGE_LAYOUT_GENERAL && dl == VK_IMAGE_LAYOUT_GENERAL);
    ser_BlitImage(&g_got, s, d, r, f);
}
static void ser_ClearColorImage(Buf *b, VkImage i, const VkClearColorValue *c, const VkImageSubresourceRange *r) {
    ser_op(b, VKR_ClearColorImage);
    BPUT(b, i);
    bput(b, c, sizeof *c);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_ClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c,
                                            uint32_t n, const VkImageSubresourceRange *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    ser_ClearColorImage(&g_got, i, c, r);
}
static void ser_ClearDepthStencilImage(Buf *b, VkImage i, const VkClearDepthStencilValue *v, const VkImageSubresourceRange *r) {
    ser_op(b, VKR_ClearDepthStencilImage);
    BPUT(b, i);
    bput(b, v, sizeof *v);
    bput(b, r, sizeof *r);
}
static void VKAPI_CALL fake_ClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                   const VkClearDepthStencilValue *v, uint32_t n,
                                                   const VkImageSubresourceRange *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    ser_ClearDepthStencilImage(&g_got, i, v, r);
}

// The markers the recorder set, and the draw each preceded.
#define MAX_MARKERS 4096
static uint64_t g_marker[MAX_MARKERS], g_last_marker;
static uint32_t g_nmarkers, g_ndraws_seen;
static uint64_t g_draw_marker[MAX_MARKERS];
static void VKAPI_CALL fake_SetCheckpointNV(VkCommandBuffer cb, const void *m) {
    CHECK(cb == FAKE_CB && g_nmarkers < MAX_MARKERS);
    g_marker[g_nmarkers++] = g_last_marker = (uint64_t)(uintptr_t)m;
}

static VkRecCalls g_calls;
static VkRecSubmit g_submit;
static void install_fakes(VkRecCalls *c) {
    memset(c, 0, sizeof *c);
#define SET_F(name, ...) c->name = fake_##name;
#define SET_D(name, kind, ...) c->name = fake_##name;
#define SET_C(name) c->name = fake_##name;
    VKREC_FIXED(SET_F)
    VKREC_DRAWS(SET_D)
    VKREC_CUSTOM(SET_C)
#undef SET_F
#undef SET_D
#undef SET_C
    c->SetCheckpointNV = fake_SetCheckpointNV;
}

// The recorder copies the calls at vkrec_open: a case that swaps one makes the table with begin_fake, swaps, then opens.
static void begin_fake(void) {
    start_counter();
    install_fakes(&g_calls);
}
static void open_now(size_t ring_bytes, uint32_t frames, int checkpoints) {
    g_submit = (VkRecSubmit){fake_begin, fake_cmd, fake_end, checkpoints};
    VkRecOpen o = {ring_bytes, frames, &g_calls, &g_submit};
    vkrec_open(&o);
}
static void open_fake(size_t ring_bytes, uint32_t frames, int checkpoints) {
    begin_fake();
    open_now(ring_bytes, frames, checkpoints);
}
static void close_fake(void) {
    vkrec_idle();
    counter_raise(gpu_sync_handed());
    gpu_sync_finish();
    vkrec_close();
    gpu_sync_stop();
}

// ---------- every command, as a packet, made into the call
static void send_customs(uint64_t seed) {
    static VkDescriptorSet sets[3];
    static uint32_t dyn[2];
    for (int i = 0; i < 3; i++) sets[i] = (VkDescriptorSet)(uintptr_t)(0x1000 + seed * 16 + (uint64_t)i);
    dyn[0] = (uint32_t)seed * 4, dyn[1] = 77;
    ser_BindDescriptorSets(&g_want, VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipelineLayout)(uintptr_t)0x2000, 1, 3, sets, 2, dyn);
    vkrec_BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipelineLayout)(uintptr_t)0x2000, 1, 3, sets, 2, dyn);
    memset(sets, 0xEE, sizeof sets), memset(dyn, 0xEE, sizeof dyn);
    vkrec_BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)(uintptr_t)0x2001, 0, 0, NULL, 0, NULL);
    ser_BindDescriptorSets(&g_want, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)(uintptr_t)0x2001, 0, 0, NULL, 0, NULL);

    // push descriptors: a buffer, an image, a texel buffer view and a count of three, each a copy
    static VkDescriptorBufferInfo bi[3];
    static VkDescriptorImageInfo ii[2];
    static VkBufferView tv[1];
    static VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) bi[i] = (VkDescriptorBufferInfo){(VkBuffer)(uintptr_t)(0x3000 + i), 256u * (uint64_t)i, 256};
    for (int i = 0; i < 2; i++)
        ii[i] = (VkDescriptorImageInfo){(VkSampler)(uintptr_t)0x4000, (VkImageView)(uintptr_t)(0x4100 + i), VK_IMAGE_LAYOUT_GENERAL};
    tv[0] = (VkBufferView)(uintptr_t)0x4200;
    w[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 3,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = bi};
    w[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 30, .dstArrayElement = 1,
                                  .descriptorCount = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = ii};
    w[2] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 9, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, .pTexelBufferView = tv};
    ser_Push(&g_want, VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipelineLayout)(uintptr_t)0x2000, 1, 3, w);
    vkrec_PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, (VkPipelineLayout)(uintptr_t)0x2000, 1, 3, w);
    memset(bi, 0xEE, sizeof bi), memset(ii, 0xEE, sizeof ii), memset(tv, 0xEE, sizeof tv), memset(w, 0xEE, sizeof w);

    static uint32_t pc[32];
    fill_bytes(pc, sizeof pc, seed + 5);
    ser_PushConstants(&g_want, (VkPipelineLayout)(uintptr_t)0x2000, 0x1f, 0, 128, pc);
    vkrec_PushConstants((VkPipelineLayout)(uintptr_t)0x2000, 0x1f, 0, 128, pc);
    ser_PushConstants(&g_want, (VkPipelineLayout)(uintptr_t)0x2000, 1, 12, 12, pc + 3);   // 12 bytes: rounded in the packet only
    vkrec_PushConstants((VkPipelineLayout)(uintptr_t)0x2000, 1, 12, 12, pc + 3);
    memset(pc, 0xEE, sizeof pc);

    static VkBuffer vb[3];
    static VkDeviceSize vo[3], vs[3], vst[3];
    for (int i = 0; i < 3; i++) vb[i] = (VkBuffer)(uintptr_t)(0x5000 + i), vo[i] = 64u * (uint64_t)i, vs[i] = 4096, vst[i] = 16;
    ser_BindVertexBuffers2(&g_want, 2, 3, vb, vo, vs, vst);
    vkrec_BindVertexBuffers2(2, 3, vb, vo, vs, vst);
    ser_BindVertexBuffers2(&g_want, 0, 3, vb, vo, NULL, NULL);
    vkrec_BindVertexBuffers2(0, 3, vb, vo, NULL, NULL);
    memset(vb, 0xEE, sizeof vb), memset(vo, 0xEE, sizeof vo), memset(vs, 0xEE, sizeof vs), memset(vst, 0xEE, sizeof vst);

    static VkViewport vp[2];
    static VkRect2D sc[3];
    for (int i = 0; i < 2; i++) vp[i] = (VkViewport){(float)i, 2, 640, 480, 0, 1};
    for (int i = 0; i < 3; i++) sc[i] = (VkRect2D){{i, 2 * i}, {100u + (uint32_t)i, 50}};
    ser_Viewports(&g_want, 2, vp);
    vkrec_SetViewportWithCount(2, vp);
    ser_Scissors(&g_want, 3, sc);
    vkrec_SetScissorWithCount(3, sc);
    memset(vp, 0xEE, sizeof vp), memset(sc, 0xEE, sizeof sc);

    static VkVertexInputBindingDescription2EXT bd[2];
    static VkVertexInputAttributeDescription2EXT ad[3];
    for (uint32_t i = 0; i < 2; i++)
        bd[i] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, i, 16 + i, VK_VERTEX_INPUT_RATE_VERTEX, 1};
    for (uint32_t i = 0; i < 3; i++)
        ad[i] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, i, i % 2, VK_FORMAT_R32G32B32A32_SFLOAT, 4 * i};
    ser_VertexInput(&g_want, 2, bd, 3, ad);
    vkrec_SetVertexInputEXT(2, bd, 3, ad);
    memset(bd, 0xEE, sizeof bd), memset(ad, 0xEE, sizeof ad);

    static VkBool32 en[3];
    static VkColorBlendEquationEXT eq[3];
    static VkColorComponentFlags wm[5];
    static float bc[4];
    fill_bytes(en, sizeof en, seed + 1), fill_bytes(eq, sizeof eq, seed + 2), fill_bytes(wm, sizeof wm, seed + 3);
    fill_bytes(bc, sizeof bc, seed + 4);
    ser_SetColorBlendEnableEXT(&g_want, 1, 3, en);
    vkrec_SetColorBlendEnableEXT(1, 3, en);
    ser_SetColorBlendEquationEXT(&g_want, 0, 3, eq);
    vkrec_SetColorBlendEquationEXT(0, 3, eq);
    ser_SetColorWriteMaskEXT(&g_want, 2, 5, wm);
    vkrec_SetColorWriteMaskEXT(2, 5, wm);
    ser_BlendConstants(&g_want, bc);
    vkrec_SetBlendConstants(bc);
    memset(en, 0xEE, sizeof en), memset(eq, 0xEE, sizeof eq), memset(wm, 0xEE, sizeof wm), memset(bc, 0xEE, sizeof bc);

    static VkMemoryBarrier2 mb[1];
    static VkBufferMemoryBarrier2 bb[2];
    static VkImageMemoryBarrier2 ib[3];
    for (int i = 0; i < 1; i++) mb[i] = (VkMemoryBarrier2){VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, 1, 2, 3, 4};
    for (int i = 0; i < 2; i++)
        bb[i] = (VkBufferMemoryBarrier2){VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, NULL, 5, 6, 7, 8, 0, 0, (VkBuffer)(uintptr_t)(0x6000 + i), 0, 256};
    for (int i = 0; i < 3; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = 9, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                        .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = (VkImage)(uintptr_t)(0x7000 + i)};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = 1, .memoryBarrierCount = 1,
                           .pMemoryBarriers = mb, .bufferMemoryBarrierCount = 2, .pBufferMemoryBarriers = bb,
                           .imageMemoryBarrierCount = 3, .pImageMemoryBarriers = ib};
    ser_Barrier(&g_want, &di);
    vkrec_PipelineBarrier2(&di);
    memset(mb, 0xEE, sizeof mb), memset(bb, 0xEE, sizeof bb), memset(ib, 0xEE, sizeof ib), memset(&di, 0xEE, sizeof di);

    static VkRenderingAttachmentInfo ca[2], da, sa;
    for (int i = 0; i < 2; i++)
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = (VkImageView)(uintptr_t)(0x8000 + i),
                                            .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR};
    da = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = (VkImageView)(uintptr_t)0x8100};
    sa = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = (VkImageView)(uintptr_t)0x8200};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{1, 2}, {640, 480}}, .layerCount = 1,
                          .colorAttachmentCount = 2, .pColorAttachments = ca, .pDepthAttachment = &da, .pStencilAttachment = &sa};
    ser_Rendering(&g_want, &ri);
    vkrec_BeginRendering(&ri);
    VkRenderingInfo none = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {8, 8}}, .layerCount = 1,
                            .colorAttachmentCount = 1, .pColorAttachments = ca};
    ser_Rendering(&g_want, &none);
    vkrec_BeginRendering(&none);
    memset(ca, 0xEE, sizeof ca), memset(&da, 0xEE, sizeof da), memset(&sa, 0xEE, sizeof sa), memset(&ri, 0xEE, sizeof ri);
    memset(&none, 0xEE, sizeof none);
    ser_op(&g_want, VKR_EndRendering);
    vkrec_EndRendering();

    static VkClearAttachment cl[2];
    static VkClearRect cr[3];
    fill_bytes(cl, sizeof cl, seed + 5), fill_bytes(cr, sizeof cr, seed + 6);
    ser_Clear(&g_want, 2, cl, 3, cr);
    vkrec_ClearAttachments(2, cl, 3, cr);
    memset(cl, 0xEE, sizeof cl), memset(cr, 0xEE, sizeof cr);

    // the transfers, each with its region copied into the packet
    static VkBufferCopy bcopy;
    static VkBufferImageCopy bic;
    static VkImageCopy icopy;
    static VkImageBlit blit;
    static VkClearColorValue ccv;
    static VkClearDepthStencilValue dsv;
    static VkImageSubresourceRange srange;
    fill_bytes(&bcopy, sizeof bcopy, seed + 7), fill_bytes(&bic, sizeof bic, seed + 8);
    fill_bytes(&icopy, sizeof icopy, seed + 9), fill_bytes(&blit, sizeof blit, seed + 10);
    fill_bytes(&ccv, sizeof ccv, seed + 11), fill_bytes(&dsv, sizeof dsv, seed + 12), fill_bytes(&srange, sizeof srange, seed + 13);
    ser_CopyBuffer(&g_want, (VkBuffer)(uintptr_t)0x9001, (VkBuffer)(uintptr_t)0x9002, &bcopy);
    vkrec_CopyBuffer((VkBuffer)(uintptr_t)0x9001, (VkBuffer)(uintptr_t)0x9002, &bcopy);
    ser_CopyBufferToImage(&g_want, (VkBuffer)(uintptr_t)0x9003, (VkImage)(uintptr_t)0x9004, &bic);
    vkrec_CopyBufferToImage((VkBuffer)(uintptr_t)0x9003, (VkImage)(uintptr_t)0x9004, &bic);
    ser_CopyImageToBuffer(&g_want, (VkImage)(uintptr_t)0x9005, (VkBuffer)(uintptr_t)0x9006, &bic);
    vkrec_CopyImageToBuffer((VkImage)(uintptr_t)0x9005, (VkBuffer)(uintptr_t)0x9006, &bic);
    ser_CopyImage(&g_want, (VkImage)(uintptr_t)0x9007, (VkImage)(uintptr_t)0x9008, &icopy);
    vkrec_CopyImage((VkImage)(uintptr_t)0x9007, (VkImage)(uintptr_t)0x9008, &icopy);
    ser_BlitImage(&g_want, (VkImage)(uintptr_t)0x9009, (VkImage)(uintptr_t)0x900a, &blit, VK_FILTER_LINEAR);
    vkrec_BlitImage((VkImage)(uintptr_t)0x9009, (VkImage)(uintptr_t)0x900a, &blit, VK_FILTER_LINEAR);
    ser_ClearColorImage(&g_want, (VkImage)(uintptr_t)0x900b, &ccv, &srange);
    vkrec_ClearColorImage((VkImage)(uintptr_t)0x900b, &ccv, &srange);
    ser_ClearDepthStencilImage(&g_want, (VkImage)(uintptr_t)0x900c, &dsv, &srange);
    vkrec_ClearDepthStencilImage((VkImage)(uintptr_t)0x900c, &dsv, &srange);
    memset(&bcopy, 0xEE, sizeof bcopy), memset(&bic, 0xEE, sizeof bic), memset(&icopy, 0xEE, sizeof icopy);
    memset(&blit, 0xEE, sizeof blit), memset(&ccv, 0xEE, sizeof ccv), memset(&dsv, 0xEE, sizeof dsv);
    memset(&srange, 0xEE, sizeof srange);
}

static void case_pack_roundtrip(void) {
    open_fake(1u << 20, 2, 0);
    __atomic_store_n(&g_hold, 1, __ATOMIC_RELEASE);   // the recorder makes no call until every caller's storage is gone
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);
    CHECK(v == 1);
    size_t n = sizeof g_senders / sizeof g_senders[0];
    for (size_t i = 0; i < n; i++) g_senders[i](i + 1);
    send_customs(7);
    static const VkSemaphoreSubmitInfo waits[2] = {
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = (VkSemaphore)(uintptr_t)0x500, .value = 40},
        {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = (VkSemaphore)(uintptr_t)0x501, .value = 41}};
    CHECK(vkrec_end(waits, 2) == 1);
    __atomic_store_n(&g_hold, 0, __ATOMIC_RELEASE);
    vkrec_idle();
    CHECK(g_frames_ended == 1 && g_end_waits == 2);
    if (g_got.n != g_want.n) fprintf(stderr, "the calls made hold %zu bytes, the commands sent %zu\n", g_got.n, g_want.n);
    CHECK(g_want.n > 1000 && g_got.n == g_want.n);
    if (memcmp(g_got.p, g_want.p, g_want.n)) {
        size_t at = 0;
        while (g_got.p[at] == g_want.p[at]) at++;
        fprintf(stderr, "the calls differ from the commands at byte %zu of %zu\n", at, g_want.n);
    }
    CHECK(!memcmp(g_got.p, g_want.p, g_want.n));
    VkRecFrame const *rows;
    uint32_t nrows;
    rows = vkrec_frames(&nrows);
    CHECK(nrows == 1 && rows[0].value == 1 && rows[0].packets > n && rows[0].draws == 0);
    printf("ok pack_roundtrip %zu scalar commands, %zu bytes of calls, %u packets, %llu bytes\n", n, g_want.n, rows[0].packets,
           (unsigned long long)rows[0].bytes);
    close_fake();
}

static void case_pack_pnext(void) {
    open_fake(1u << 20, 2, 0);
    vkrec_begin(GPU_VK_REPLAY);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .pNext = &ca, .layerCount = 1};
    vkrec_BeginRendering(&ri);
    printf("a rendering with a pNext chain was packed\n");
}

// ---------- frames
#define ORDER_THREADS 2
#define ORDER_FRAMES 300
static uint64_t g_last_effect_value, g_effects_run;
static uint64_t g_frame_draws_seen, g_frame_expect;
static int g_order_open;
static void order_effect(void *arg, uint64_t value) {
    CHECK((uint64_t)(uintptr_t)arg == value);
    CHECK(counter_now() >= value && gpu_sync_done() == value - 1);
    CHECK(value == g_last_effect_value + 1);
    g_last_effect_value = value;
    g_effects_run++;
}
static void order_on_cmd(uint64_t value) {
    CHECK(!g_order_open && value == g_frames_ended + 1);
    g_order_open = 1;
    g_frame_draws_seen = 0;
    g_frame_expect = value % 7 + 1;
}
static void order_on_end(uint64_t value) {
    CHECK(g_order_open && g_frame_draws_seen == g_frame_expect);
    CHECK(value == g_frames_ended + 1);
    g_order_open = 0;
}
// A draw of frame v carries v in its vertex count: a draw inside another frame's packets is caught here.
static void VKAPI_CALL order_draw(VkCommandBuffer cb, uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi) {
    CHECK(cb == FAKE_CB && g_order_open);
    CHECK(vc == g_frames_ended + 1 && ic == 1 && fv == 0 && fi == 0);
    g_frame_draws_seen++;
}
static void order_thread(void *arg) {
    (void)arg;
    for (int i = 0; i < ORDER_FRAMES; i++) {
        uint64_t v = vkrec_begin(GPU_VK_REPLAY);
        vkrec_effect(order_effect, (void *)(uintptr_t)v);
        for (uint64_t d = 0; d < v % 7 + 1; d++) vkrec_Draw((uint32_t)v, 1, 0, 0);
        CHECK(vkrec_end(NULL, 0) == v);
    }
    ended();
}
static void order_device(void *arg) {
    (void)arg;
    for (;;) {
        uint64_t issued = gpu_sync_issued(), now = counter_now();
        if (now < issued) counter_raise(issued);
        else if (now == ORDER_THREADS * ORDER_FRAMES) break;
        else plat_yield();
    }
    ended();
}
static void case_frame_order(void) {
    begin_fake();
    g_on_cmd = order_on_cmd, g_on_end = order_on_end;
    g_calls.Draw = order_draw;
    open_now(1u << 16, 2, 0);
    // the ring is small (64 KiB) and the recorder, a thread that waits on the ring, takes each frame in its own time
    for (int i = 0; i < ORDER_THREADS; i++) CHECK(plat_thread_start(order_thread, NULL, 1 << 20));
    CHECK(plat_thread_start(order_device, NULL, 1 << 20));
    join(ORDER_THREADS + 1);
    close_fake();
    CHECK(g_frames_ended == ORDER_THREADS * ORDER_FRAMES && g_effects_run == g_frames_ended);
    printf("ok frame_order frames %llu effects %llu\n", (unsigned long long)g_frames_ended, (unsigned long long)g_effects_run);
}

static int g_third_returned, g_third_draws = 1;   // a device frame has no draw outside a render pass: it is empty
static uint64_t g_third_value, g_done_at_third;
static void third_thread(void *arg) {
    (void)arg;
    g_third_value = vkrec_begin(GPU_VK_REPLAY);
    g_done_at_third = gpu_sync_done();
    __atomic_store_n(&g_third_returned, 1, __ATOMIC_RELEASE);
    if (g_third_draws) vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    ended();
}
static void case_frame_limit(void) {
    open_fake(1u << 20, 2, 0);
    uint64_t v1 = vkrec_begin(GPU_VK_REPLAY);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    uint64_t v2 = vkrec_begin(GPU_VK_REPLAY);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    vkrec_idle();   // both submitted; the device has finished neither
    CHECK(v1 == 1 && v2 == 2 && counter_now() == 0 && gpu_sync_done() == 0);
    CHECK(plat_thread_start(third_thread, NULL, 1 << 20));
    plat_sleep_ns(300000000ull);
    CHECK(!__atomic_load_n(&g_third_returned, __ATOMIC_ACQUIRE));   // the third frame waits for the first
    counter_raise(v1);
    for (int i = 0; i < 500 && !__atomic_load_n(&g_third_returned, __ATOMIC_ACQUIRE); i++) plat_sleep_ns(10000000ull);
    CHECK(__atomic_load_n(&g_third_returned, __ATOMIC_ACQUIRE));
    CHECK(g_third_value == 3 && g_done_at_third == v1);   // released by the first frame, not the second
    join(1);
    close_fake();
    printf("ok frame_limit\n");
}

// ---------- where the frame time goes (OOT2_REPLAY_COUNTERS: vk_replay_time.c, gpu_sync.c, vk_submit.c)
// Two frames; the second frame's effect takes 25 ms on the completion thread; the game waits for the first value through the
// back end's fence wait (the device is late by 60 ms) and finishes the queue (late by 45 ms). The case prints the exit line group
// of the counters itself, so the driver reads exact counts (frames 2, effects 2, one fence wait, one finish) and times (the
// waits at least as long as the lateness, the effect at least 25 ms).
static void counted_effect(void *arg, uint64_t value) {
    (void)value;
    if (arg) plat_sleep_ns(25000000ull);
}
static uint64_t g_late_value, g_late_ns;
static void late_device(void *arg) {
    (void)arg;
    plat_sleep_ns(g_late_ns);
    counter_raise(g_late_value);
    ended();
}
static void case_counted_waits(void) {
    rc_enable(600);
    open_fake(1u << 20, 4, 0);
    uint64_t v1 = vkrec_begin(GPU_VK_REPLAY);
    vkrec_effect(counted_effect, NULL);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    uint64_t v2 = vkrec_begin(GPU_VK_REPLAY);
    vkrec_effect(counted_effect, (void *)1);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    vkrec_idle();
    g_late_value = v1, g_late_ns = 60000000ull;
    CHECK(plat_thread_start(late_device, NULL, 1 << 20));
    CHECK(gfx_vulkan_backend.fence_wait(v1, UINT64_MAX));
    join(1);
    g_late_value = v2, g_late_ns = 45000000ull;
    CHECK(plat_thread_start(late_device, NULL, 1 << 20));
    gfx_vulkan_backend.finish(0);
    join(2);
    close_fake();
    uint64_t sum[RC_N];
    rc_sum(sum);
    printf("ok counted_waits fence %llu %llu finish %llu %llu sync %llu %llu\n", (unsigned long long)sum[RC_T_FENCE_N],
           (unsigned long long)sum[RC_T_FENCE_NS], (unsigned long long)sum[RC_T_FINISH_N],
           (unsigned long long)sum[RC_T_FINISH_NS], (unsigned long long)sum[RC_T_SYNC_N],
           (unsigned long long)sum[RC_T_SYNC_NS]);
    rc_print("exit", 0);
}

// ---------- breadcrumbs
#define CRUMB_DRAWS 50
static nvn_stage g_vs[CRUMB_DRAWS], g_ps[CRUMB_DRAWS], g_ts[CRUMB_DRAWS];
static void make_stages(void) {
    for (int i = 0; i < CRUMB_DRAWS; i++) {
        memset(&g_vs[i], 0, sizeof g_vs[i]), memset(&g_ps[i], 0, sizeof g_ps[i]), memset(&g_ts[i], 0, sizeof g_ts[i]);
        g_vs[i].kind = NVN_STAGE_VS, g_ps[i].kind = NVN_STAGE_PS, g_ts[i].kind = NVN_STAGE_TS;
        snprintf(g_vs[i].key, sizeof g_vs[i].key, "%016x.%08x", 0xA000u + (unsigned)i, 1u);
        snprintf(g_ps[i].key, sizeof g_ps[i].key, "%016x.%08x", 0xB000u + (unsigned)i, 2u);
        snprintf(g_ts[i].key, sizeof g_ts[i].key, "%016x.%08x", 0xC000u + (unsigned)i, 3u);
    }
}
static uint8_t crumb_kind_of(int i) { return (uint8_t[]){'d', 'i', 'c', 'm'}[i % 4]; }
// Every draw's marker is the one set right before its call.
static void note_draw(void) {
    CHECK(g_last_marker != 0 && g_ndraws_seen < MAX_MARKERS);
    g_draw_marker[g_ndraws_seen++] = g_last_marker;
    g_last_marker = 0;
}
static void VKAPI_CALL crumb_draw(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    (void)cb, (void)a, (void)b, (void)c, (void)d;
    note_draw();
}
static void VKAPI_CALL crumb_drawi(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) {
    (void)cb, (void)a, (void)b, (void)c, (void)d, (void)e;
    note_draw();
}
static void VKAPI_CALL crumb_dispatch(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c) {
    (void)cb, (void)a, (void)b, (void)c;
    note_draw();
}
static void send_crumb_frame(uint64_t *value) {
    *value = vkrec_begin(GPU_VK_REPLAY);
    for (int i = 0; i < CRUMB_DRAWS; i++) {
        vkrec_stages(&g_vs[i], &g_ps[i], i % 3 == 0 ? &g_ts[i] : NULL);
        switch (i % 4) {
        case 0: vkrec_Draw(3, 1, 0, 0); break;
        case 1: vkrec_DrawIndexed(3, 1, 0, 0, 0); break;
        case 2: vkrec_Dispatch(1, 1, 1); break;
        default: vkrec_DrawMeshTasksEXT(1, 1, 1); break;
        }
    }
    vkrec_end(NULL, 0);
}
static void case_breadcrumbs(void) {
    make_stages();
    begin_fake();
    g_calls.Draw = crumb_draw, g_calls.DrawIndexed = crumb_drawi, g_calls.Dispatch = crumb_dispatch;
    g_calls.DrawMeshTasksEXT = crumb_dispatch;
    open_now(1u << 20, 2, 1);
    uint64_t v;
    send_crumb_frame(&v);
    vkrec_idle();
    CHECK(g_ndraws_seen == CRUMB_DRAWS && g_nmarkers == CRUMB_DRAWS);
    for (uint32_t k = 1; k <= CRUMB_DRAWS; k++) {
        uint64_t m = GPU_VK_DRAW_MARKER(v, k);
        CHECK(g_draw_marker[k - 1] == m);   // the packet carried draw k's number, and the call came right after it
        VkRecCrumb c;
        uint64_t value;
        uint32_t n;
        CHECK(vkrec_crumb(m, &c, &value, &n) && value == v && n == k);
        int i = (int)k - 1;
        CHECK(c.kind == crumb_kind_of(i));
        CHECK(c.id[0] == &g_vs[i] && c.id[1] == &g_ps[i] && c.id[2] == (i % 3 == 0 ? &g_ts[i] : NULL));
        char text[512];
        CHECK(vkrec_marker_text(m, text, sizeof text));
        CHECK(strstr(text, g_vs[i].key) && strstr(text, g_ps[i].key));
        CHECK((i % 3 == 0) == (strstr(text, g_ts[i].key) != NULL));
    }
    VkRecCrumb c;
    CHECK(!vkrec_crumb(GPU_VK_DRAW_MARKER(v, CRUMB_DRAWS + 1), &c, NULL, NULL));   // past the last draw of the frame
    CHECK(!vkrec_crumb(GPU_VK_DRAW_MARKER(v + 1000, 1), &c, NULL, NULL));          // a frame that never was
    CHECK(!vkrec_crumb(GPU_VK_MARKER(v, GPU_VK_REPLAY, 0), &c, NULL, NULL));       // not a draw's marker
    VkRecFrame const *rows;
    uint32_t nrows;
    rows = vkrec_frames(&nrows);
    CHECK(nrows == 1 && rows[0].draws == CRUMB_DRAWS);
    // The frames go on: frame v's table is reused when the ring of tables comes round, VKREC_MAX_FRAMES + 1 frames later
    // (never sooner than `frames + 1`), and no marker names it then.
    uint64_t later;
    for (uint32_t f = 0; f < VKREC_MAX_FRAMES + 1; f++) {
        counter_raise(gpu_sync_handed());
        send_crumb_frame(&later);
    }
    vkrec_idle();
    char text[512];
    CHECK(!vkrec_crumb(GPU_VK_DRAW_MARKER(v, 5), &c, NULL, NULL));
    CHECK(!vkrec_marker_text(GPU_VK_DRAW_MARKER(v, 5), text, sizeof text) && strstr(text, "no longer known"));
    CHECK(vkrec_crumb(GPU_VK_DRAW_MARKER(later, 5), &c, NULL, NULL) && c.id[0] == &g_vs[4]);
    printf("ok breadcrumbs %d draws named by their own records\n", CRUMB_DRAWS);
    close_fake();
}

static void case_breadcrumbs_off(void) {
    make_stages();
    open_fake(1u << 20, 2, 0);
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);
    vkrec_stages(&g_vs[0], &g_ps[0], NULL);
    for (int i = 0; i < 10; i++) vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    vkrec_idle();
    CHECK(g_nmarkers == 0 && g_ndraws_seen == 0 + 0);
    VkRecCrumb c;
    CHECK(!vkrec_crumb(GPU_VK_DRAW_MARKER(v, 1), &c, NULL, NULL));
    VkRecFrame const *rows;
    uint32_t nrows;
    rows = vkrec_frames(&nrows);
    CHECK(nrows == 1 && rows[0].draws == 0);
    printf("ok breadcrumbs_off\n");
    close_fake();
}

static void case_marker_text(void) {
    make_stages();
    open_fake(1u << 20, 2, 1);
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);
    vkrec_stages(&g_vs[3], NULL, NULL);
    vkrec_Dispatch(1, 1, 1);
    vkrec_stages(NULL, NULL, NULL);
    vkrec_DrawIndirect((VkBuffer)(uintptr_t)1, 0, 1, 16);
    vkrec_end(NULL, 0);
    vkrec_idle();
    char text[512], want[512];
    CHECK(vkrec_marker_text(GPU_VK_DRAW_MARKER(v, 1), text, sizeof text));
    snprintf(want, sizeof want, "draw 1 of submission %llu (dispatch): vs %s", (unsigned long long)v, g_vs[3].key);
    CHECK(!strcmp(text, want));
    CHECK(vkrec_marker_text(GPU_VK_DRAW_MARKER(v, 2), text, sizeof text) && strstr(text, "indirect draw") &&
          strstr(text, "no stage identity"));
    CHECK(!vkrec_marker_text(GPU_VK_MARKER(v, GPU_VK_INTERNAL, 1), text, sizeof text) && strstr(text, "not a draw"));
    char small[24];
    vkrec_marker_text(GPU_VK_DRAW_MARKER(v, 1), small, sizeof small);   // never past the end of a short buffer
    CHECK(strlen(small) < sizeof small);
    printf("ok marker_text %s\n", want);
    close_fake();
}

// ---------- the window of frames in flight
// The guest's view of a frame's results with a window of `frames` frames. The guest keeps frames + 1 result buffers, and
// the effects of frame v, which run on the completion thread before v is done, fill buffer v % (frames + 1) with v: a
// buffer is replaced by the frame frames + 1 later, which the window does not let begin before the first has been read.
// The device completes frame s only once frame s + frames - 1 has begun, so the window is as full as it can be, and it
// hurries when the guest waits for a frame. Checked at every begin: the frame `frames` before is done and its buffer
// holds it, never an older frame's; no more than `frames` frames in flight, and `frames` of them at the deepest; every
// frame in flight, and the one just done, still has a breadcrumb table that is its own; a table is reused at the distance
// of the ring. And after a wait on a frame's value the guest reads that frame's results.
#define WIN_FRAMES 45
#define WIN_DRAWS 2
static uint64_t g_win_f;                      // the frames in flight the case expects
static uint64_t g_win_buf[VKREC_MAX_FRAMES + 1];
static uint64_t g_win_begun, g_win_ask, g_win_end, g_win_stalled;   // atomic
static uint64_t g_win_progress;               // atomic: bumped by the guest each time it moves on (a frame begun, ended, waited for)
static int g_win_narrow;                      // the mutant: the recorder is opened with one frame fewer than the case expects
// The stall detector counts poll rounds in which nothing moved: the device's own counter, the frames begun, the frame asked
// for and the guest's progress counter all stayed as they were. Each round sleeps, so a guest that can run does run between
// rounds; a machine so loaded that it runs the guest slowly runs this thread slowly too, and the rounds it needs take
// longer, where a wall-clock limit would fail it. Only a guest that stays where it is for this many rounds is stalled.
#define WIN_STALL_ROUNDS 3000
static uint64_t g_win_values[WIN_FRAMES];
static nvn_stage g_win_stage[WIN_FRAMES];
static uint64_t g_win_deepest, g_win_waits, g_win_reuses;

static uint64_t ld_acq(const uint64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static void st_rel(uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

static void win_effect(void *arg, uint64_t value) {
    CHECK((uint64_t)(uintptr_t)arg == value && gpu_sync_done() == value - 1);
    st_rel(&g_win_buf[value % (g_win_f + 1)], value);
}
static void VKAPI_CALL win_draw(VkCommandBuffer cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    (void)cb, (void)a, (void)b, (void)c, (void)d;
}
static void win_device(void *arg) {
    (void)arg;
    uint64_t idle = 0, seen[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    for (;;) {
        uint64_t issued = gpu_sync_issued(), now = counter_now(), begun = ld_acq(&g_win_begun);
        uint64_t target = begun >= g_win_f ? begun - g_win_f + 1 : 0, ask = ld_acq(&g_win_ask);
        if (ask > target) target = ask;
        if (ld_acq(&g_win_end)) target = issued;
        if (target > issued) target = issued;
        if (now < target) {
            counter_raise(target);
            idle = 0;
            continue;
        }
        if (ld_acq(&g_win_end) && now == issued && gpu_sync_done() == now) break;
        uint64_t cur[4] = {issued, begun, ask, ld_acq(&g_win_progress)};
        if (memcmp(seen, cur, sizeof cur)) {
            memcpy(seen, cur, sizeof cur);
            idle = 0;
        } else if (++idle >= WIN_STALL_ROUNDS && !ld_acq(&g_win_end)) {
            st_rel(&g_win_stalled, 1);   // a narrower window than the case expects: the guest waits for a frame the
            st_rel(&g_win_end, 1);       // device would only complete later; let it go and say so
        }
        plat_sleep_ns(50000ull);
    }
    ended();
}
// Frame n has begun: the frames in flight and the one just done each have a table that is their own.
static void win_check_tables(uint64_t n) {
    for (uint64_t j = n >= g_win_f ? n - g_win_f : 0; j <= n; j++)
        for (uint32_t k = 1; k <= WIN_DRAWS; k++) {
            VkRecCrumb crumb;
            uint64_t value = 0;
            CHECK(vkrec_crumb(GPU_VK_DRAW_MARKER(g_win_values[j], k), &crumb, &value, NULL));
            CHECK(value == g_win_values[j] && crumb.id[0] == &g_win_stage[j]);
        }
    if (n >= VKREC_MAX_FRAMES + 1) {   // the ring has come round: no marker names the frame that had this frame's table
        VkRecCrumb crumb;
        CHECK(!vkrec_crumb(GPU_VK_DRAW_MARKER(g_win_values[n - (VKREC_MAX_FRAMES + 1)], 1), &crumb, NULL, NULL));
        g_win_reuses++;
    }
}
// which: a number of frames given to vkrec_open, "default" (none given, no setting: 2) or "setting" (none given, the
// variable OOT2_FRAMES_IN_FLIGHT says)
static void case_window(const char *which) {
    uint32_t open_frames = 0;
    if (!strcmp(which, "default")) g_win_f = 2;
    else if (!strcmp(which, "setting")) g_win_f = (uint64_t)atoi(getenv("OOT2_FRAMES_IN_FLIGHT"));
    else g_win_f = open_frames = (uint32_t)atoi(which);
    CHECK(g_win_f >= 1 && g_win_f <= VKREC_MAX_FRAMES);
    if (g_win_narrow) {
        CHECK(open_frames >= 2);
        open_frames--;
    }
    for (int i = 0; i < WIN_FRAMES; i++) {
        g_win_stage[i].kind = NVN_STAGE_VS;
        snprintf(g_win_stage[i].key, sizeof g_win_stage[i].key, "%016x.%08x", 0xD000u + (unsigned)i, 7u);
    }
    begin_fake();
    g_calls.Draw = win_draw;
    open_now(1u << 20, open_frames, 1);
    CHECK(plat_thread_start(win_device, NULL, 1 << 20));
    uint64_t rng = 0x1234567ull + g_win_f;
    for (uint64_t n = 0; n < WIN_FRAMES; n++) {
        uint64_t v = vkrec_begin(GPU_VK_REPLAY);
        CHECK(v == n + 1);
        __atomic_add_fetch(&g_win_progress, 1, __ATOMIC_ACQ_REL);
        g_win_values[n] = v;
        uint64_t done = gpu_sync_done();
        CHECK(v - done <= g_win_f);                              // no more frames in flight than the window
        if (v - done > g_win_deepest) g_win_deepest = v - done;
        if (n >= g_win_f) {
            uint64_t w = g_win_values[n - g_win_f];
            CHECK(done >= w);                                    // the frame `frames` before is done ...
            CHECK(ld_acq(&g_win_buf[w % (g_win_f + 1)]) == w);   // ... and its results are in its buffer, not an older frame's
        }
        vkrec_effect(win_effect, (void *)(uintptr_t)v);
        for (int d = 0; d < WIN_DRAWS; d++) {
            vkrec_stages(&g_win_stage[n], NULL, NULL);
            vkrec_Draw(3, 1, 0, 0);
        }
        win_check_tables(n);
        CHECK(vkrec_end(NULL, 0) == v);
        st_rel(&g_win_begun, n + 1);
        __atomic_add_fetch(&g_win_progress, 1, __ATOMIC_ACQ_REL);
        if (n >= 3 * g_win_f && n % 5 == 0) {   // the guest waits for one of the last `frames` frames and reads its results
            uint64_t w = g_win_values[n - next_random(&rng) % g_win_f];
            st_rel(&g_win_ask, w);
            CHECK(gpu_sync_wait(w, 20000000000ull));
            st_rel(&g_win_ask, 0);
            __atomic_add_fetch(&g_win_progress, 1, __ATOMIC_ACQ_REL);
            CHECK(ld_acq(&g_win_buf[w % (g_win_f + 1)]) == w);
            g_win_waits++;
        }
        CHECK(!ld_acq(&g_win_stalled));
    }
    vkrec_idle();
    st_rel(&g_win_end, 1);
    join(1);
    CHECK(!ld_acq(&g_win_stalled));
    close_fake();
    CHECK(g_win_deepest == g_win_f);   // the window was full at some begin, and never fuller
    CHECK(g_win_reuses > 0 && g_win_waits > 0);
    printf("ok window frames %llu begun %d deepest %llu waits %llu table reuses %llu\n", (unsigned long long)g_win_f,
           WIN_FRAMES, (unsigned long long)g_win_deepest, (unsigned long long)g_win_waits,
           (unsigned long long)g_win_reuses);
}

// ---------- the range of the window
// vkrec_open takes 1 to VKREC_MAX_FRAMES frames and stops by name on more; the setting's row says the same range and
// the default of today.
static void case_frames_over(void) {
    begin_fake();
    open_now(1u << 20, VKREC_MAX_FRAMES + 1, 0);
    printf("a window of %u frames was accepted\n", VKREC_MAX_FRAMES + 1);
}
static void case_frames_range(void) {
    int id = rt_setting_find("frames_in_flight");
    CHECK(id >= 0);
    const RtSettingRow *r = rt_setting_row((RtSettingId)id);
    CHECK(r->type == RT_SETTING_INT && r->def == 2 && r->min == 1 && r->max == VKREC_MAX_FRAMES);
    CHECK(rt_setting(RT_SET_frames_in_flight) == r->def);   // no variable, no file: today's window
    printf("ok frames_range %lld %lld %lld\n", (long long)r->min, (long long)r->def, (long long)r->max);
}
// the variable is read through the store, which refuses a value outside the row's range by name (exit code 2)
static void case_frames_setting(void) {
    printf("ok frames_setting %lld\n", (long long)rt_setting(RT_SET_frames_in_flight));
}

// ---------- guest threads: the virtual core in the waits
// Two guest threads on one virtual core, written by hand as host functions in the lifter's shape. The main thread begins
// a third frame that has to wait for the first to be done; the other thread, of worse priority, can run only when the main
// thread has given the core up, and it is what lets the wait end (core_wait), or it submits behind the frame that holds the
// gate and must not keep the core from the thread that holds it (core_gate). In core_lock the first of two such threads
// takes a host mutex and then begins a submission, so it waits for the gate with the mutex held and without the core; the
// second, on the core the first gave up, takes the same mutex: it must wait for it without the core too, or the main thread
// cannot get the core back to open the gate. A wait that keeps its core hangs the case, which its watchdog ends with exit
// code 3.
#define GMB MOD_MAIN_BASE
#define GFN(i) (GMB + 0x100ull * (i))   // the guest address of guest function i
#define GPC (GMB + 0x4000)              // the guest pc the guest threads' SVCs report
enum { FN_MAIN_WAIT, FN_MAIN_GATE, FN_RAISE, FN_GATE, FN_MAIN_LOCK, FN_HOLDER, FN_TAKER, FN_END };
enum { GSTACK = 0x100000 };
static uint64_t g_core_heap;
static uint64_t g_core_steps[8], g_core_nsteps, g_core_at_gate;   // atomic
static PlatMutex g_core_mu = PLAT_MUTEX_INIT;   // the host mutex of core_lock

static void core_step(uint64_t what) {
    uint64_t i = __atomic_fetch_add(&g_core_nsteps, 1, __ATOMIC_SEQ_CST);
    CHECK(i < 8);
    __atomic_store_n(&g_core_steps[i], what, __ATOMIC_RELEASE);
}
static inline uint64_t gsv(Cpu *t, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    t->x[0] = a0, t->x[1] = a1, t->x[2] = a2, t->x[3] = a3;
    rt_svc(t, imm, GPC);
    return t->x[0];
}
// A thread of worse priority than the main thread (44), on the main thread's core, started and joined by it; the stack
// number gives two threads stacks of their own.
static uint32_t gstart_at(Cpu *t, int fn, int prio, int stack) {
    t->x[1] = GFN(fn), t->x[2] = 0, t->x[3] = g_core_heap + GSTACK + 0x10000ull * (uint64_t)stack, t->x[4] = (uint64_t)prio,
    t->x[5] = 0;
    rt_svc(t, 0x08, GPC);   // CreateThread
    CHECK(t->x[0] == 0);
    uint32_t h = (uint32_t)t->x[1];
    CHECK(gsv(t, 0x09, h, 0, 0, 0) == 0);   // StartThread
    return h;
}
static uint32_t gstart(Cpu *t, int fn) { return gstart_at(t, fn, 50, 1); }
static void gjoin(Cpu *t, uint32_t h) {
    *(volatile uint32_t *)(uintptr_t)g_core_heap = h;
    CHECK(gsv(t, 0x18, 0, g_core_heap, 1, ~0ull) == 0);   // WaitSynchronization: the thread ended
    CHECK(gsv(t, 0x16, h, 0, 0, 0) == 0);                 // CloseHandle
}
static void gexit(Cpu *t) { gsv(t, 0x0A, 0, 0, 0, 0); }

// Two frames in, the device has finished neither.
static void core_two_frames(void) {
    CHECK(vkrec_begin(GPU_VK_REPLAY) == 1);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    CHECK(vkrec_begin(GPU_VK_REPLAY) == 2);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    vkrec_idle();
    CHECK(counter_now() == 0 && gpu_sync_done() == 0);
}
static void g_main_wait(Cpu *t) {
    core_two_frames();
    uint32_t h = gstart(t, FN_RAISE);
    core_step(1);
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);   // waits for frame 1 to be done
    core_step(3);
    CHECK(v == 3 && gpu_sync_done() >= 1);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);
    gjoin(t, h);
    gexit(t);
}
// The device reaches frame 1 once this thread has run, which it can only do on the core the main thread gave up.
static void g_raise(Cpu *t) {
    core_step(2);
    counter_raise(1);
    gexit(t);
}
static void g_main_gate(Cpu *t) {
    core_two_frames();
    uint32_t h = gstart(t, FN_GATE);
    core_step(1);
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);   // holds the gate while it waits for frame 1
    core_step(3);
    CHECK(v == 3 && gpu_sync_done() >= 1);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);                        // the recorder submits it and opens the gate
    gjoin(t, h);
    gexit(t);
}
// A submission of the back end's own (a present, an upload) from another thread: it waits for the gate, which is the
// main thread's until its frame is submitted.
static void g_gate(Cpu *t) {
    core_step(2);
    __atomic_store_n(&g_core_at_gate, 1, __ATOMIC_RELEASE);
    CHECK(gpu_sync_begin() == 4);
    core_step(4);
    gpu_sync_end();
    gexit(t);
}
static void gate_device(void *arg) {   // the device reaches frame 1 once the other thread waits at the gate
    (void)arg;
    while (!__atomic_load_n(&g_core_at_gate, __ATOMIC_ACQUIRE)) plat_sleep_ns(100000ull);
    plat_sleep_ns(100000000ull);
    counter_raise(1);
}
// core_lock: the main thread holds the gate while it waits for frame 1, without the core. The holder (the better of the two
// threads, so the first to run) takes a host mutex and begins a submission, which waits for the gate: it has the mutex and
// no core. The taker runs on the core it left, and takes the mutex. The device reaches frame 1 once the taker is at the
// mutex, the main thread needs the core back to go on, and it can have it only if the taker gave it up for the mutex.
static void g_main_lock(Cpu *t) {
    core_two_frames();
    uint32_t h = gstart_at(t, FN_HOLDER, 50, 1), k = gstart_at(t, FN_TAKER, 52, 2);
    core_step(1);
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);   // holds the gate while it waits for frame 1
    core_step(4);
    CHECK(v == 3 && gpu_sync_done() >= 1);
    vkrec_Draw(3, 1, 0, 0);
    vkrec_end(NULL, 0);                        // the recorder submits it and opens the gate
    gjoin(t, h);
    gjoin(t, k);
    gexit(t);
}
static void g_holder(Cpu *t) {
    rt_kernel_lock(&g_core_mu);
    core_step(2);
    CHECK(gpu_sync_begin() == 4);   // the gate is the main thread's: this waits for it without a core, holding the mutex
    core_step(5);
    gpu_sync_end();
    plat_mutex_unlock(&g_core_mu);
    gexit(t);
}
static void g_taker(Cpu *t) {
    core_step(3);
    __atomic_store_n(&g_core_at_gate, 1, __ATOMIC_RELEASE);   // gate_device raises frame 1 a moment after this
    rt_kernel_lock(&g_core_mu);
    core_step(6);
    plat_mutex_unlock(&g_core_mu);
    gexit(t);
}
static const GuestFn g_core_fns[FN_END + 1] = {g_main_wait, g_main_gate, g_raise, g_gate, g_main_lock,
                                               g_holder,    g_taker,     NULL};
static uint64_t g_core_addrs[FN_END + 1];
static void core_guest_end(Cpu *t) { (void)t; }

static const char *const g_core_name[3] = {"core_wait", "core_gate", "core_lock"};
static void core_watchdog(void *arg) {
    plat_sleep_ns(15000000000ull);
    fprintf(stderr, "hung: %s: a wait kept the virtual core (steps %llu)\n", (const char *)arg,
            (unsigned long long)g_core_nsteps);
    _Exit(3);
}
static void case_core(int which) {
    int gate = which != 0;   // in core_gate and core_lock the device reaches frame 1 once a thread is waiting at something
    CHECK(plat_thread_start(core_watchdog, (void *)(uintptr_t)g_core_name[which], 1 << 20));
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20, .system_resource_size = 1ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x1, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1, .no_async_preempt = 1};
    rt_kernel_init(&k);
    static GuestFn fns[FN_END + 1];
    memcpy(fns, g_core_fns, sizeof fns);
    fns[FN_END] = core_guest_end;
    for (int i = 0; i <= FN_END; i++) g_core_addrs[i] = GFN(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", GMB, MOD_MAIN_SIZE, g_core_addrs, fns, (uint64_t)FN_END + 1, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 64ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    g_core_heap = h.x[1];
    begin_fake();
    open_now(1u << 20, 2, 0);
    if (gate) CHECK(plat_thread_start(gate_device, NULL, 1 << 20));
    static const int mains[3] = {FN_MAIN_WAIT, FN_MAIN_GATE, FN_MAIN_LOCK};
    rt_kernel_run(GFN(mains[which]), 0);
    uint64_t want[6] = {1, 2, 3, 4, 5, 6};
    uint64_t n = which == 2 ? 6 : gate ? 4 : 3;
    CHECK(__atomic_load_n(&g_core_nsteps, __ATOMIC_ACQUIRE) == n);
    for (uint64_t i = 0; i < n; i++) CHECK(g_core_steps[i] == want[i]);   // the other thread ran while the main one waited
    close_fake();
    printf("ok %s steps %llu %llu %llu\n", g_core_name[which], (unsigned long long)g_core_steps[0],
           (unsigned long long)g_core_steps[1], (unsigned long long)g_core_steps[2]);
}

// ---------- the device
#define TARGET 1024
#define DRAWS 3000
#define VARIANTS 8
#define CELLS 55
#define TARGET_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define GUEST_BYTES (64u << 20)
#define POOLS 4096

typedef struct {
    VkpLayouts L;
    VkShaderModule vs, fs[VARIANTS];
    VkPipeline pipe[VARIANTS];
    VkDescriptorPool pool;
    VkDescriptorSet set0;
    VkImage image;
    VkDeviceMemory image_mem;
    VkImageView view;
    VkBuffer buf;
    VkDeviceMemory buf_mem;
    VkBuffer readback;
    VkDeviceMemory readback_mem;
    void *readback_map;
    VkShaderStageFlags push_stages;
    uint32_t set1_types[32];
} Dev;
static Dev D;

static void device_check(VkResult r, const char *what) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "check failed: %s returned %d\n", what, (int)r);
        exit(1);
    }
}

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    CHECK(!"no memory type fits");
    return 0;
}

static VkShaderModule load_module(const char *folder, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", folder, name);
    FILE *f = fopen(path, "rb");
    CHECK(f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)size);
    CHECK(code && fread(code, 1, (size_t)size, f) == (size_t)size);
    fclose(f);
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)size, .pCode = code};
    VkShaderModule m;
    device_check(vkCreateShaderModule(gpu_vk.device, &ci, NULL, &m), "vkCreateShaderModule");
    free(code);
    return m;
}

static VkPipeline make_pipeline(VkShaderModule vs, VkShaderModule fs) {
    VkDynamicState dyn[4 * VKP_MAX_DYNAMIC], part[VKP_MAX_DYNAMIC];
    uint32_t nd = 0;
    const VkGraphicsPipelineLibraryFlagsEXT parts[4] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    for (int p = 0; p < 4; p++) {
        uint32_t n = vkp_dynamic_states(parts[p], &gpu_vk.facts, part);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t k = 0;
            while (k < nd && dyn[k] != part[i]) k++;
            if (k == nd) dyn[nd++] = part[i];
        }
    }
    VkPipelineShaderStageCreateInfo st[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = 1, .pAttachments = &cba};
    VkFormat fmt = TARGET_FORMAT;
    VkPipelineRenderingCreateInfo rc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
                                        .pColorAttachmentFormats = &fmt};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = nd, .pDynamicStates = dyn};
    VkGraphicsPipelineCreateInfo gi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rc, .stageCount = 2,
                                       .pStages = st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
                                       .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
                                       .pDepthStencilState = &ds, .pColorBlendState = &cb, .pDynamicState = &dy,
                                       .layout = D.L.graphics};
    VkPipeline p;
    device_check(vkCreateGraphicsPipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &gi, NULL, &p), "vkCreateGraphicsPipelines");
    return p;
}

static void dev_setup(const char *folder) {
    gpu_vk_open();
    vkp_layouts_create(&D.L);
    D.push_stages = vkp_push_range(VKP_GRAPHICS).stageFlags;
    VkDescriptorSetLayoutBinding b[VKP_SET1_MAX];
    uint32_t nb = vkp_set1_bindings(VKP_GRAPHICS, b);
    for (uint32_t i = 0; i < nb; i++) D.set1_types[b[i].binding] = b[i].descriptorType;
    D.vs = load_module(folder, "vs.spv");
    for (int k = 0; k < VARIANTS; k++) {
        char name[32];
        snprintf(name, sizeof name, "fs%d.spv", k);
        D.fs[k] = load_module(folder, name);
        D.pipe[k] = make_pipeline(D.vs, D.fs[k]);
    }
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKP_GLOBAL_IMAGES}, {VK_DESCRIPTOR_TYPE_SAMPLER, VKP_GLOBAL_SAMPLERS}};
    VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ps};
    device_check(vkCreateDescriptorPool(gpu_vk.device, &pci, NULL, &D.pool), "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = D.pool,
                                         .descriptorSetCount = 1, .pSetLayouts = &D.L.global};
    device_check(vkAllocateDescriptorSets(gpu_vk.device, &alloc, &D.set0), "vkAllocateDescriptorSets");

    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = TARGET_FORMAT,
                            .extent = {TARGET, TARGET, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    device_check(vkCreateImage(gpu_vk.device, &ii, NULL, &D.image), "vkCreateImage");
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(gpu_vk.device, D.image, &req);
    VkMemoryAllocateInfo ma = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    device_check(vkAllocateMemory(gpu_vk.device, &ma, NULL, &D.image_mem), "vkAllocateMemory (image)");
    device_check(vkBindImageMemory(gpu_vk.device, D.image, D.image_mem, 0), "vkBindImageMemory");
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = D.image, .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                .format = TARGET_FORMAT, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    device_check(vkCreateImageView(gpu_vk.device, &vi, NULL, &D.view), "vkCreateImageView");

    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4u << 20,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    device_check(vkCreateBuffer(gpu_vk.device, &bi, NULL, &D.buf), "vkCreateBuffer");
    vkGetBufferMemoryRequirements(gpu_vk.device, D.buf, &req);
    ma = (VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    device_check(vkAllocateMemory(gpu_vk.device, &ma, NULL, &D.buf_mem), "vkAllocateMemory (buffer)");
    device_check(vkBindBufferMemory(gpu_vk.device, D.buf, D.buf_mem, 0), "vkBindBufferMemory");

    bi = (VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)TARGET * TARGET * 4,
                              .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    device_check(vkCreateBuffer(gpu_vk.device, &bi, NULL, &D.readback), "vkCreateBuffer (readback)");
    vkGetBufferMemoryRequirements(gpu_vk.device, D.readback, &req);
    ma = (VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    device_check(vkAllocateMemory(gpu_vk.device, &ma, NULL, &D.readback_mem), "vkAllocateMemory (readback)");
    device_check(vkBindBufferMemory(gpu_vk.device, D.readback, D.readback_mem, 0), "vkBindBufferMemory (readback)");
    device_check(vkMapMemory(gpu_vk.device, D.readback_mem, 0, VK_WHOLE_SIZE, 0, &D.readback_map), "vkMapMemory");
}

// ---------- a replay of the synthetic frame, resolved as the pipeline cost study's was (tools/pipecost/vkpc_draw.c, replay mode):
// the pipeline through a hash table keyed by program, 29 buffers through a binary search of 4,096 sorted ranges, the
// uniform buffers copied from guest memory as the snapshot of D17a, texture handles through a table. Each draw changes
// the pipeline (run length 1, as OoT's 601 program binds for 340 draws).
typedef struct {
    uint64_t base;
    uint32_t size, offset;
} Range;
typedef struct {
    uint32_t program;
    uint64_t addr[40];
    uint32_t handle[32];
} Op;
typedef struct {
    float origin[2], size[2], color[4];
    uint32_t handles[24];                 // the rest of the 128 bytes: texture slots
} PushBlock;
static struct {
    uint8_t *guest;
    Range ranges[POOLS];
    uint32_t keys[8192], vals[8192];
    uint32_t *handles;
    Op *ops;
    uint8_t *snapshot;                    // the frame's ring slice of uniform copies
    int ring;                             // 1: through the recorder, 0: the calls on this thread
    VkCommandBuffer cb;
    uint32_t draws;                       // draws of a frame
    const nvn_stage *vs_ids, *ps_ids;     // per draw, when a case names them itself
} R;

// R.ring: 1 the packets, 0 the calls on this thread, 2 nothing (the replay's own work alone, for the timing).
#define E_CALL(name, ...) (R.ring == 2 ? (void)0 : R.ring ? vkrec_##name(__VA_ARGS__) : vkCmd##name(R.cb, ##__VA_ARGS__))

static void replay_setup(void) {
    R.draws = DRAWS;
    R.guest = malloc(GUEST_BYTES);
    CHECK(R.guest);
    for (size_t i = 0; i < GUEST_BYTES; i += 4) *(uint32_t *)(R.guest + i) = (uint32_t)(i * 2654435761u);
    uint64_t base = 0x80000000ull;
    for (uint32_t i = 0; i < POOLS; i++) {
        uint32_t size = 0x10000u << (i % 5);
        R.ranges[i] = (Range){base, size, (uint32_t)((i * 0x10000u) % (GUEST_BYTES - 0x10000))};
        base += size + 0x10000;
    }
    for (uint32_t j = 0; j < VARIANTS; j++) {
        uint32_t program = 0x1000 + j * 37, h = (program * 2654435761u) >> 19;
        while (R.keys[h]) h = (h + 1) & 8191;
        R.keys[h] = program + 1, R.vals[h] = j;
    }
    R.handles = malloc(33025 * sizeof *R.handles);
    for (uint32_t i = 0; i < 33025; i++) R.handles[i] = i % 64;
    R.ops = malloc(DRAWS * sizeof *R.ops);
    R.snapshot = malloc(1u << 20);
    CHECK(R.handles && R.ops && R.snapshot);
    uint64_t r = 77;
    for (uint32_t i = 0; i < DRAWS; i++) {
        Op *op = &R.ops[i];
        op->program = 0x1000 + (i % VARIANTS) * 37;
        for (uint32_t k = 0; k < 40; k++) {
            r ^= r << 13, r ^= r >> 7, r ^= r << 17;
            const Range *g = &R.ranges[r % POOLS];
            op->addr[k] = g->base + (r >> 20) % (g->size - 4096);
        }
        for (uint32_t k = 0; k < 32; k++) op->handle[k] = (uint32_t)((r >> (k % 40)) % 33025);
    }
}

static uint32_t lookup_pipeline(uint32_t program) {
    uint32_t h = (program * 2654435761u) >> 19;
    for (;;) {
        if (R.keys[h] == program + 1) return R.vals[h];
        CHECK(R.keys[h]);
        h = (h + 1) & 8191;
    }
}
static const Range *lookup_range(uint64_t addr) {
    uint32_t lo = 0, hi = POOLS;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) / 2;
        if (R.ranges[mid].base <= addr) lo = mid;
        else hi = mid;
    }
    return &R.ranges[lo];
}

// The colour draw i is given, and what the target holds at its cell's centre afterwards.
static void draw_color(uint32_t i, float c[4]) {
    c[0] = (float)((i * 37) % 255) / 255.0f, c[1] = (float)((i * 91) % 255) / 255.0f, c[2] = (float)((i * 53) % 255) / 255.0f, c[3] = 1;
}

static void set_state(void) {
    VkViewport vp = {0, 0, (float)TARGET, (float)TARGET, 0, 1};
    VkRect2D sc = {{0, 0}, {TARGET, TARGET}};
    E_CALL(SetViewportWithCount, 1, &vp);
    E_CALL(SetScissorWithCount, 1, &sc);
    E_CALL(SetLineWidth, 1.0f);
    E_CALL(SetDepthBias, 0, 0, 0);
    E_CALL(SetCullMode, VK_CULL_MODE_NONE);
    E_CALL(SetFrontFace, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    E_CALL(SetDepthBiasEnable, VK_FALSE);
    E_CALL(SetRasterizerDiscardEnable, VK_FALSE);
    E_CALL(SetPolygonModeEXT, VK_POLYGON_MODE_FILL);
    E_CALL(SetDepthClampEnableEXT, VK_FALSE);
    E_CALL(SetDepthClipNegativeOneToOneEXT, VK_FALSE);
    E_CALL(SetVertexInputEXT, 0, NULL, 0, NULL);   // the draws read no vertex buffer
    E_CALL(SetPrimitiveTopology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    E_CALL(SetPrimitiveRestartEnable, VK_FALSE);
    E_CALL(SetDepthTestEnable, VK_FALSE);
    E_CALL(SetDepthWriteEnable, VK_FALSE);
    E_CALL(SetDepthCompareOp, VK_COMPARE_OP_ALWAYS);
    E_CALL(SetDepthBoundsTestEnable, VK_FALSE);
    E_CALL(SetStencilTestEnable, VK_FALSE);
    E_CALL(SetStencilOp, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS);
    E_CALL(SetStencilCompareMask, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    E_CALL(SetStencilWriteMask, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    E_CALL(SetStencilReference, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    E_CALL(SetDepthBounds, 0.0f, 1.0f);
    float bc[4] = {0, 0, 0, 0};
    float *bcp = bc;
    if (R.ring == 1) vkrec_SetBlendConstants(bcp);
    else if (R.ring == 0) vkCmdSetBlendConstants(R.cb, bcp);
    E_CALL(SetLogicOpEnableEXT, VK_FALSE);
    VkBool32 en[1] = {0};
    VkColorBlendEquationEXT eq[1] = {{VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD}};
    VkColorComponentFlags wm[1] = {0xF};
    E_CALL(SetColorBlendEnableEXT, 0, 1, en);
    E_CALL(SetColorBlendEquationEXT, 0, 1, eq);
    E_CALL(SetColorWriteMaskEXT, 0, 1, wm);
}

static nvn_stage g_stage_vs, g_stage_ps[VARIANTS];

static void barrier(VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                    VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                               .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = D.image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    E_CALL(PipelineBarrier2, &di);
}

// The frame's commands, as packets or as calls; the caller has begun the frame (and set R.ring and R.cb).
static void replay_frame(uint32_t frame) {
    (void)frame;
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = D.view,
                                    .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {TARGET, TARGET}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    E_CALL(BeginRendering, &ri);
    set_state();
    E_CALL(BindDescriptorSets, VK_PIPELINE_BIND_POINT_GRAPHICS, D.L.graphics, 0, 1, &D.set0, 0, NULL);
    uint32_t snap_at = 0;
    for (uint32_t i = 0; i < R.draws; i++) {
        const Op *op = &R.ops[i];
        uint32_t pi = lookup_pipeline(op->program);
        VkDescriptorBufferInfo bi[29];
        VkWriteDescriptorSet w[29];
        for (uint32_t k = 0; k < 29; k++) {
            const Range *g = lookup_range(op->addr[k]);
            uint64_t off = ((op->addr[k] - g->base) & ~(uint64_t)255) % (3u << 20);
            if (k >= 1 && k <= 8) {   // the vertex stage's uniform buffers are snapshotted
                if (snap_at + 256 > (1u << 20)) snap_at = 0;
                memcpy(R.snapshot + snap_at, R.guest + (op->addr[k] % (GUEST_BYTES - 256)), 256);
                off = snap_at, snap_at += 256;
            }
            bi[k] = (VkDescriptorBufferInfo){D.buf, off, 256};
            w[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = k + 1, .descriptorCount = 1,
                                          .descriptorType = D.set1_types[k + 1], .pBufferInfo = &bi[k]};
        }
        PushBlock pb;
        memset(&pb, 0, sizeof pb);
        pb.origin[0] = (float)(i % CELLS) / CELLS, pb.origin[1] = (float)(i / CELLS) / CELLS;
        pb.size[0] = pb.size[1] = 1.0f / CELLS;
        draw_color(i, pb.color);
        for (uint32_t k = 0; k < 24; k++) pb.handles[k] = R.handles[op->handle[k]];
        if (R.ring == 1) vkrec_stages(R.vs_ids ? &R.vs_ids[i] : &g_stage_vs, R.ps_ids ? &R.ps_ids[i] : &g_stage_ps[pi], NULL);
        E_CALL(BindPipeline, VK_PIPELINE_BIND_POINT_GRAPHICS, D.pipe[pi]);
        E_CALL(PushDescriptorSetKHR, VK_PIPELINE_BIND_POINT_GRAPHICS, D.L.graphics, 1, 29, w);
        E_CALL(PushConstants, D.L.graphics, D.push_stages, 0, sizeof pb, &pb);
        E_CALL(Draw, 3, 1, 0, 0);
    }
    E_CALL(EndRendering);
    barrier(VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
}

static void init_stages(void) {
    memset(&g_stage_vs, 0, sizeof g_stage_vs);
    g_stage_vs.kind = NVN_STAGE_VS;
    snprintf(g_stage_vs.key, sizeof g_stage_vs.key, "%016x.%08x", 0xA0A0u, 1u);
    for (int k = 0; k < VARIANTS; k++) {
        memset(&g_stage_ps[k], 0, sizeof g_stage_ps[k]);
        g_stage_ps[k].kind = NVN_STAGE_PS;
        snprintf(g_stage_ps[k].key, sizeof g_stage_ps[k].key, "%016x.%08x", 0xB0B0u + (unsigned)k, 2u);
    }
}

// One frame through the ring or directly; returns its value.
static uint64_t frame_values[8];
static uint64_t g_gate_ns;   // the last ring frame's wait for the submission gate
static uint64_t run_frame(int ring, uint32_t f, uint64_t *slot_wait_ns, uint64_t *busy_ns, uint64_t *cpu_ns) {
    R.ring = ring;
    uint64_t waited = 0, value;
    if (ring) {
        value = vkrec_begin(GPU_VK_REPLAY);
        replay_frame(f);
        vkrec_end(NULL, 0);
        VkRecFrame const *rows;
        uint32_t n;
        rows = vkrec_frames(&n);
        *busy_ns = rows[n - 1].replay_ns, *cpu_ns = rows[n - 1].replay_cpu_ns, waited = rows[n - 1].slot_wait_ns;
        g_gate_ns = rows[n - 1].gate_wait_ns;
    } else {
        // the same limit of two frames in flight, then the frame made on this thread
        if (f >= 2) {
            uint64_t t = plat_now_ns();
            gpu_sync_wait(frame_values[(f - 2) % 8], UINT64_MAX);
            waited = plat_now_ns() - t;
        }
        value = gpu_vk_begin(GPU_VK_REPLAY);
        frame_values[f % 8] = value;
        uint64_t t0 = plat_now_ns(), c0 = plat_thread_cpu_ns(NULL);
        R.cb = gpu_vk_cmd();
        replay_frame(f);
        gpu_vk_end(NULL, 0);
        *busy_ns = plat_now_ns() - t0, *cpu_ns = plat_thread_cpu_ns(NULL) - c0;
    }
    *slot_wait_ns = waited;
    return value;
}

static void read_back(void) {
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    VkBufferImageCopy bic = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {TARGET, TARGET, 1}};
    vkCmdCopyImageToBuffer(cb, D.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, D.readback, 1, &bic);
    VkBufferMemoryBarrier2 bm = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                 .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                                 .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = D.readback, .size = VK_WHOLE_SIZE};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &bm};
    vkCmdPipelineBarrier2(cb, &di);
    gpu_vk_end(NULL, 0);
    CHECK(gpu_sync_wait(v, 10000000000ull));
}

// Every draw's own cell holds its colour times its variant's factor.
static void check_cells(uint32_t *bad_out) {
    const uint8_t *px = D.readback_map;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < DRAWS; i++) {
        uint32_t x = (uint32_t)(((float)(i % CELLS) + 0.5f) / CELLS * TARGET), y = (uint32_t)(((float)(i / CELLS) + 0.5f) / CELLS * TARGET);
        float c[4];
        draw_color(i, c);
        float f = (float)((i % VARIANTS) + 1) / 8.0f;
        const uint8_t *p = px + ((size_t)y * TARGET + x) * 4;
        for (int k = 0; k < 3; k++) {
            int want = (int)(c[k] * f * 255.0f + 0.5f), got = p[k];
            if (got < want - 2 || got > want + 2) {
                if (!bad) fprintf(stderr, "draw %u: channel %d is %d, wanted %d\n", i, k, got, want);
                bad++;
                break;
            }
        }
    }
    *bad_out = bad;
}

static void case_dev_usable(void) {
    gpu_vk_open();
    printf("usable 1\n");
}

static void print_validation(void) {
    uint64_t e, w;
    gpu_vk_validation_counts(&e, &w);
    printf("validation errors %llu warnings %llu\n", (unsigned long long)e, (unsigned long long)w);
}

static void case_dev_draws(const char *folder) {
    dev_setup(folder);
    init_stages();
    replay_setup();
    vkrec_open(NULL);
    uint64_t sw, busy, cpu, v = 0;
    for (uint32_t f = 0; f < 3; f++) v = run_frame(1, f, &sw, &busy, &cpu);   // through the ring
    vkrec_idle();
    gpu_sync_finish();
    read_back();
    uint32_t bad;
    check_cells(&bad);
    CHECK(bad == 0);
    printf("ring: value %llu, %u of %u cells right\n", (unsigned long long)v, DRAWS - bad, DRAWS);
    for (uint32_t f = 0; f < 3; f++) v = run_frame(0, f, &sw, &busy, &cpu);   // the same frames as direct calls
    gpu_sync_finish();
    memset(D.readback_map, 0, (size_t)TARGET * TARGET * 4);
    read_back();
    check_cells(&bad);
    CHECK(bad == 0);
    printf("direct: value %llu, %u of %u cells right\n", (unsigned long long)v, DRAWS - bad, DRAWS);
    VkRecFrame const *rows;
    uint32_t n;
    rows = vkrec_frames(&n);
    CHECK(n == 3 && rows[0].draws == (gpu_vk.checkpoints ? DRAWS : 0));
    printf("frame: %u packets, %llu bytes, %u draws with breadcrumbs\n", rows[0].packets, (unsigned long long)rows[0].bytes, rows[0].draws);
    print_validation();
    vkrec_close();
    printf("ok dev_draws\n");
}

// A third frame waits for the first, which the device holds on a semaphore the test signals.
static void case_dev_third(const char *folder) {
    dev_setup(folder);
    init_stages();
    replay_setup();
    vkrec_open(NULL);
    g_third_draws = 0;
    VkSemaphoreTypeCreateInfo tci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
    VkSemaphore gate;
    device_check(vkCreateSemaphore(gpu_vk.device, &sci, NULL, &gate), "vkCreateSemaphore");
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    R.ring = 1;
    uint64_t v1 = vkrec_begin(GPU_VK_REPLAY);
    replay_frame(0);
    vkrec_end(&wait, 1);   // the device holds this frame until the test signals the semaphore
    uint64_t v2 = vkrec_begin(GPU_VK_REPLAY);
    replay_frame(1);
    vkrec_end(NULL, 0);
    vkrec_idle();
    plat_sleep_ns(100000000ull);
    CHECK(gpu_sync_done() == 0 && gpu_sync_issued() == v2);
    CHECK(plat_thread_start(third_thread, NULL, 1 << 20));
    plat_sleep_ns(400000000ull);
    CHECK(!__atomic_load_n(&g_third_returned, __ATOMIC_ACQUIRE));   // the third frame waits for the first
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = gate, .value = 1};
    device_check(vkSignalSemaphore(gpu_vk.device, &si), "vkSignalSemaphore");
    for (int i = 0; i < 1000 && !__atomic_load_n(&g_third_returned, __ATOMIC_ACQUIRE); i++) plat_sleep_ns(10000000ull);
    CHECK(__atomic_load_n(&g_third_returned, __ATOMIC_ACQUIRE));
    CHECK(g_third_value == v2 + 1 && g_done_at_third >= v1);
    join(1);
    vkrec_idle();
    gpu_sync_finish();
    print_validation();
    printf("ok dev_third value %llu done_at_third %llu\n", (unsigned long long)g_third_value, (unsigned long long)g_done_at_third);
    vkrec_close();
}

// The markers the recorder sets on the real command buffer are captured on the way, and an injected loss reports two of them.
static uint64_t g_real_marker[8192];
static uint32_t g_nreal;
static PFN_vkCmdSetCheckpointNV g_real_checkpoint;
static void VKAPI_CALL capture_checkpoint(VkCommandBuffer cb, const void *m) {
    if (g_nreal < 8192) g_real_marker[g_nreal++] = (uint64_t)(uintptr_t)m;
    g_real_checkpoint(cb, m);
}
static uint64_t g_inject[2];
static void VKAPI_CALL inject_checkpoints(VkQueue q, uint32_t *n, VkCheckpointData2NV *data) {
    (void)q;
    if (!data) {
        *n = 2;
        return;
    }
    data[0].stage = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, data[0].pCheckpointMarker = (void *)(uintptr_t)g_inject[0];
    data[1].stage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, data[1].pCheckpointMarker = (void *)(uintptr_t)g_inject[1];
}
static VkResult VKAPI_CALL inject_fault(VkDevice d, VkDeviceFaultCountsEXT *counts, VkDeviceFaultInfoEXT *info) {
    (void)d;
    memset(&counts->addressInfoCount, 0, sizeof counts->addressInfoCount);
    counts->vendorInfoCount = 0, counts->vendorBinarySize = 0;
    if (info) snprintf(info->description, sizeof info->description, "injected by the test: no real fault");
    return VK_SUCCESS;
}
static void case_dev_loss(const char *folder, const char *dir) {
    dev_setup(folder);
    init_stages();
    replay_setup();
    static nvn_stage vs[64], ps[64];
    for (int i = 0; i < 64; i++) {
        vs[i].kind = NVN_STAGE_VS, ps[i].kind = NVN_STAGE_PS;
        snprintf(vs[i].key, sizeof vs[i].key, "%016x.%08x", 0xD000u + (unsigned)i, 1u);
        snprintf(ps[i].key, sizeof ps[i].key, "%016x.%08x", 0xE000u + (unsigned)i, 2u);
    }
    VkRecCalls calls;
    vkrec_calls_device(&calls);
    g_real_checkpoint = calls.SetCheckpointNV;
    calls.SetCheckpointNV = capture_checkpoint;
    VkRecSubmit submit;
    vkrec_submit_device(&submit);
    CHECK(submit.checkpoints);
    VkRecOpen o = {0, 0, &calls, &submit};
    vkrec_open(&o);
    gpu_set_report_folder(dir);
    R.ring = 1, R.draws = 40, R.vs_ids = vs, R.ps_ids = ps;   // 40 real draws, each with stages of its own
    uint64_t v = vkrec_begin(GPU_VK_REPLAY);
    replay_frame(0);
    vkrec_end(NULL, 0);
    vkrec_idle();
    gpu_sync_finish();
    uint32_t first = 0;
    while (first < g_nreal && ((g_real_marker[first] >> 1) & 7) != GPU_VK_DRAW) first++;
    CHECK(g_nreal >= 40 && first < g_nreal);
    g_inject[0] = g_real_marker[first + 16];   // the vertex stage reached draw 17, the fragment stage draw 15
    g_inject[1] = g_real_marker[first + 14];
    printf("loss value %llu draw17 %s %s draw15 %s %s\n", (unsigned long long)v, vs[16].key, ps[16].key, vs[14].key, ps[14].key);
    vkGetQueueCheckpointData2NV = inject_checkpoints;
    vkGetDeviceFaultInfoEXT = inject_fault;
    gpu_vk_failed(VK_ERROR_DEVICE_LOST, "the test's injected loss");
}

// ---------- the timing of the pipeline cost study, on this device
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static double pct(uint64_t *v, uint32_t n, int p) {
    qsort(v, n, sizeof *v, cmp_u64);
    return (double)v[(uint64_t)n * (uint64_t)p / 100 >= n ? n - 1 : (uint64_t)n * (uint64_t)p / 100] / 1e6;
}
#define TIMING_FRAMES 330
#define TIMING_WARMUP 30
static void time_run(int ring, const char *label) {
    static uint64_t busy[TIMING_FRAMES], cpu[TIMING_FRAMES], wait[TIMING_FRAMES], rec[TIMING_FRAMES], reccpu[TIMING_FRAMES];
    uint64_t t_warm = 0, tw = 0, bs = 0, cs = 0, gate = 0;
    for (uint32_t f = 0; f < TIMING_FRAMES; f++) {
        if (f == TIMING_WARMUP) t_warm = plat_now_ns(), gate = 0;
        uint64_t v = run_frame(ring, f, &tw, &bs, &cs);
        (void)v;
        busy[f] = bs, cpu[f] = cs, wait[f] = tw;
        if (ring) gate += g_gate_ns;
    }
    if (ring) vkrec_idle();
    gpu_sync_finish();
    uint64_t end = plat_now_ns();
    uint32_t n = TIMING_FRAMES - TIMING_WARMUP;
    VkRecFrame const *rows = NULL;
    uint32_t nrows = 0;
    if (ring) {
        rows = vkrec_frames(&nrows);
        for (uint32_t f = 0; f < TIMING_FRAMES; f++) rec[f] = rows[nrows - TIMING_FRAMES + f].record_ns, reccpu[f] = rows[nrows - TIMING_FRAMES + f].record_cpu_ns;
    }
    double slot_wait = 0;
    for (uint32_t f = TIMING_WARMUP; f < TIMING_FRAMES; f++) slot_wait += (double)wait[f] / 1e6;
    printf("timing %s submitting_ms_p50 %.3f\n", label, pct(busy + TIMING_WARMUP, n, 50));
    printf("timing %s submitting_ms_p99 %.3f\n", label, pct(busy + TIMING_WARMUP, n, 99));
    printf("timing %s submitting_cpu_ms_p50 %.3f\n", label, pct(cpu + TIMING_WARMUP, n, 50));
    printf("timing %s frame_ms_mean %.3f\n", label, (double)(end - t_warm) / 1e6 / n);
    printf("timing %s waited_for_frame_two_before_ms_total %.1f\n", label, slot_wait);
    if (ring) {
        GpuRingStats st;
        vkrec_ring_stats(&st);
        printf("timing %s waited_for_the_gate_ms_total %.1f\n", label, (double)gate / 1e6);
        printf("timing %s recorder_ms_p50 %.3f\n", label, pct(rec + TIMING_WARMUP, n, 50));
        printf("timing %s recorder_cpu_ms_p50 %.3f\n", label, pct(reccpu + TIMING_WARMUP, n, 50));
        printf("timing %s ring_full_waits %llu ring_wait_ms %.3f packets_per_frame %u bytes_per_frame %llu\n", label,
               (unsigned long long)st.full_waits, (double)st.full_wait_ns / 1e6, rows[nrows - 1].packets,
               (unsigned long long)rows[nrows - 1].bytes);
    }
}
static void case_dev_timing(const char *folder) {
    dev_setup(folder);
    init_stages();
    replay_setup();
    vkrec_open(NULL);
    printf("breadcrumbs %s\n", gpu_vk.checkpoints ? "on" : "off");
    for (int rep = 0; rep < 2; rep++) {   // direct and ring alternate, so that drift on the machine meets both
        time_run(0, "direct");
        time_run(1, "ring");
        // the replay's own work with no command made at all: what direct and ring both pay before their commands
        static uint64_t common[TIMING_FRAMES];
        R.ring = 2;
        for (uint32_t f = 0; f < TIMING_FRAMES; f++) {
            uint64_t t0 = plat_now_ns();
            replay_frame(f);
            common[f] = plat_now_ns() - t0;
        }
        printf("timing common_replay_work_only submitting_ms_p50 %.3f\n", pct(common + TIMING_WARMUP, TIMING_FRAMES - TIMING_WARMUP, 50));
    }
    uint64_t e, w;
    gpu_vk_validation_counts(&e, &w);
    printf("ok dev_timing validation errors %llu warnings %llu\n", (unsigned long long)e, (unsigned long long)w);
    vkrec_close();
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_rec_test <case> [args] [--mutant NAME]\n");
        return 2;
    }
    int args = argc;
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--mutant")) continue;
        const char *m = argv[i + 1];
        if (!strcmp(m, "overwrite")) gpu_ring_overwrite = 1;
        else if (!strcmp(m, "nolimit")) vkrec_no_limit = 1;
        else if (!strcmp(m, "skew")) vkrec_crumb_skew = 1;
        else if (!strcmp(m, "narrow")) g_win_narrow = 1;
        else {
            fprintf(stderr, "unknown mutant %s\n", m);
            return 2;
        }
        args = i;
        break;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "ring_wrap")) case_ring_wrap();
    else if (!strcmp(cs, "ring_oversize")) case_ring_oversize();
    else if (!strcmp(cs, "pack_roundtrip")) case_pack_roundtrip();
    else if (!strcmp(cs, "pack_pnext")) case_pack_pnext();
    else if (!strcmp(cs, "frame_order")) case_frame_order();
    else if (!strcmp(cs, "frame_limit")) case_frame_limit();
    else if (!strcmp(cs, "counted_waits")) case_counted_waits();
    else if (!strcmp(cs, "window") && args >= 3) case_window(argv[2]);
    else if (!strcmp(cs, "frames_over")) case_frames_over();
    else if (!strcmp(cs, "frames_range")) case_frames_range();
    else if (!strcmp(cs, "frames_setting")) case_frames_setting();
    else if (!strcmp(cs, "core_wait")) case_core(0);
    else if (!strcmp(cs, "core_gate")) case_core(1);
    else if (!strcmp(cs, "core_lock")) case_core(2);
    else if (!strcmp(cs, "breadcrumbs")) case_breadcrumbs();
    else if (!strcmp(cs, "breadcrumbs_off")) case_breadcrumbs_off();
    else if (!strcmp(cs, "marker_text")) case_marker_text();
    else if (!strcmp(cs, "dev_usable")) case_dev_usable();
    else if (!strcmp(cs, "dev_draws") && args >= 3) case_dev_draws(argv[2]);
    else if (!strcmp(cs, "dev_third") && args >= 3) case_dev_third(argv[2]);
    else if (!strcmp(cs, "dev_loss") && args >= 4) case_dev_loss(argv[2], argv[3]);
    else if (!strcmp(cs, "dev_timing") && args >= 3) case_dev_timing(argv[2]);
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
