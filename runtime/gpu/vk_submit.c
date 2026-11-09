// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Submissions of the Vulkan back end on the timeline (oot2/gpu_vk.h, oot2/gpu.h), and its back-end table.
//
// Every submission, replayed or the back end's own, goes through gpu_vk_begin and gpu_vk_end, which wrap the timeline's
// gpu_sync_begin and gpu_sync_end around one vkQueueSubmit2 on the one queue: the command buffer, if one was asked for,
// after the caller's waits, signalling the submission's value on the timeline semaphore at all commands. The submission
// lock spans the whole, so the queue (which Vulkan requires to be used by one thread at a time) and the command pool
// are used by one thread at a time, and the values reach the semaphore in order. The completion thread waits on that
// semaphore (the source below) and runs each value's effects before it publishes the value (gpu_sync.c).
//
// Command buffers come from a set of slots: a slot is taken again once the value it was submitted with is done, so a
// slot never holds a command buffer the GPU may still read; the set grows to the number of submissions in flight. With
// the report on, each command buffer starts and ends with a diagnostic checkpoint naming its value and kind, so the
// last checkpoints a lost queue reached say which submission was running (per-draw checkpoints come with the recorder).
//
// A replayed submission is the replay's (oot2/vk_replay.h): it walks the recordings on the submitting thread and hands
// the frame to the recorder (oot2/vk_rec.h), whose thread records and submits it through the calls below.
//
// Adapted from DKB Recomp runtime/vk_sync.c at 9d8fa02 (GPL-3.0-or-later; there adapted from LAS Recomp
// runtime/nvn/vulkan/exec.c at da35b83, MIT, Copyright (c) 2026 Peiikai, licenses/lasrecomp-MIT.txt): a primary command
// buffer per submission taken from reusable slots, vkQueueSubmit2 signalling the submission's value at all commands, a
// failed wait treated as a lost device. Changed here: the back end's own work submits on the same timeline through the
// same calls, with effects of its own; checkpoints mark each command buffer; slots are freed by the published value.
#include <stdlib.h>
#include <string.h>

#include "oot2/bindless.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_capture.h"
#include "oot2/vk_image.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_present.h"
#include "oot2/vk_replay.h"
#include "plat.h"

// ---------- the timeline's source: the device's timeline semaphore
static uint64_t semaphore_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                              .pSemaphores = &gpu_vk.timeline, .pValues = &value};
    VkResult r = vkWaitSemaphores(gpu_vk.device, &wi, timeout_ns);
    if (r != VK_SUCCESS && r != VK_TIMEOUT) gpu_vk_failed(r, "vkWaitSemaphores (the completion thread)");
    uint64_t counter = 0;
    GPU_VK_OK(vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter),
              "vkGetSemaphoreCounterValue (the completion thread)");
    return counter;
}

// ---------- command buffers (the submission lock's owner only)
typedef struct {
    VkCommandBuffer cb;
    uint64_t value;   // the submission that used it last; free once that value is done
} Slot;
// Two sets of slots, each from a pool of its own: command buffers of one pool are not recorded by two threads at once,
// and a replayed submission is recorded by two (oot2/vk_rec.h): the recorder thread records the main command buffer, the
// replaying thread the prologue and the epilogue. Whoever holds the submission uses one set at a time.
typedef struct {
    Slot *slots;
    int n;
} SlotSet;
static SlotSet g_main;          // command buffers of gpu_vk.pool_main: gpu_vk_cmd's
static SlotSet g_aux;           // of gpu_vk.pool: the prologue's and the epilogue's
static VkCommandBuffer g_cur;   // the open submission's command buffer, once asked for
static VkCommandBuffer g_pro;   // its prologue, once asked for
static VkCommandBuffer g_epi;   // its epilogue, once asked for
static uint64_t g_value;        // the open submission's value, 0 when none is open
static int g_kind;

void gpu_vk_start_timeline(uint64_t after) {
    GpuTimelineSource src = {.wait = semaphore_wait};
    gpu_sync_start(&src, after);
}

void gpu_vk_stop_timeline(void) {
    gpu_sync_stop();
    free(g_main.slots);   // their command buffers go with the pools
    free(g_aux.slots);
    g_main = g_aux = (SlotSet){0};
}

static void checkpoint(int end) {
    if (gpu_vk.checkpoints)
        vkCmdSetCheckpointNV(g_cur, (const void *)(uintptr_t)GPU_VK_MARKER(g_value, g_kind, end));
}

uint64_t gpu_vk_begin(int kind) {
    if (!gpu_vk.device) rt_fatal("gfx: vulkan: a submission before the device is open");
    uint64_t value = gpu_sync_begin();
    g_value = value;
    g_kind = kind;
    g_cur = g_pro = g_epi = VK_NULL_HANDLE;
    return value;
}

// A command buffer of pool from a free slot of set, begun for the open submission.
static VkCommandBuffer slot_begin(SlotSet *set, VkCommandPool pool) {
    uint64_t done = gpu_sync_done();
    int i = 0;
    while (i < set->n && set->slots[i].value > done) i++;
    if (i == set->n) {
        Slot *s = realloc(set->slots, (size_t)(set->n + 1) * sizeof *s);
        if (!s) rt_fatal("gfx: vulkan: out of memory for %d command buffers", set->n + 1);
        set->slots = s;
        VkCommandBufferAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = 1};
        GPU_VK_OK(vkAllocateCommandBuffers(gpu_vk.device, &alloc, &set->slots[i].cb), "vkAllocateCommandBuffers");
        set->n++;
    }
    set->slots[i].value = g_value;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    GPU_VK_OK(vkBeginCommandBuffer(set->slots[i].cb, &bi), "vkBeginCommandBuffer");
    return set->slots[i].cb;
}

VkCommandBuffer gpu_vk_cmd(void) {
    if (!g_value) rt_fatal("gfx: vulkan: a command buffer outside a submission");
    if (g_cur) return g_cur;
    g_cur = slot_begin(&g_main, gpu_vk.pool_main);
    checkpoint(0);
    return g_cur;
}

// Every write before it made available and visible to every access after it, in submission order (D18's barrier).
static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

VkCommandBuffer gpu_vk_prologue(void) {
    if (!g_value) rt_fatal("gfx: vulkan: a prologue outside a submission");
    if (g_pro) return g_pro;
    g_pro = slot_begin(&g_aux, gpu_vk.pool);
    full_barrier(g_pro);
    return g_pro;
}

VkCommandBuffer gpu_vk_epilogue(void) {
    if (!g_value) rt_fatal("gfx: vulkan: an epilogue outside a submission");
    if (g_epi) return g_epi;
    g_epi = slot_begin(&g_aux, gpu_vk.pool);
    full_barrier(g_epi);
    return g_epi;
}

void gpu_vk_effect(GpuEffect fn, void *arg) {
    if (!g_value) rt_fatal("gfx: vulkan: an effect outside a submission");
    gpu_sync_effect(fn, arg);
}

uint64_t gpu_vk_end(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits) {
    return gpu_vk_end_present(waits, nwaits, NULL, NULL, NULL);
}

uint64_t gpu_vk_end_present(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits, const VkSemaphoreSubmitInfo *extra_signal,
                            const VkPresentInfoKHR *present, VkResult *present_result) {
    if (!g_value) rt_fatal("gfx: vulkan: the end of a submission that was not begun");
    uint64_t value = g_value;
    VkCommandBufferSubmitInfo cbi[3];
    uint32_t ncb = 0;
    if (g_pro) {
        full_barrier(g_pro);
        GPU_VK_OK(vkEndCommandBuffer(g_pro), "vkEndCommandBuffer (the prologue)");
        cbi[ncb++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 .commandBuffer = g_pro};
    }
    if (g_cur) {
        checkpoint(1);
        GPU_VK_OK(vkEndCommandBuffer(g_cur), "vkEndCommandBuffer");
        cbi[ncb++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 .commandBuffer = g_cur};
    }
    if (g_epi) {
        GPU_VK_OK(vkEndCommandBuffer(g_epi), "vkEndCommandBuffer (the epilogue)");
        cbi[ncb++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 .commandBuffer = g_epi};
    }
    VkSemaphoreSubmitInfo signal[2] = {{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gpu_vk.timeline,
                                        .value = value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT}};
    uint32_t nsignal = 1;
    if (extra_signal) signal[nsignal++] = *extra_signal;   // a binary semaphore, for a present
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = nwaits,
                        .pWaitSemaphoreInfos = waits, .commandBufferInfoCount = ncb, .pCommandBufferInfos = cbi,
                        .signalSemaphoreInfoCount = nsignal, .pSignalSemaphoreInfos = signal};
    GPU_VK_OK(vkQueueSubmit2(gpu_vk.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit2");
    // vkQueuePresentKHR needs the queue to itself like a submit does, so it is made here, before the submission's gate opens
    if (present) {
        VkResult r = vkQueuePresentKHR(gpu_vk.queue, present);
        if (present_result) *present_result = r;
        else if (r != VK_SUCCESS) gpu_vk_failed(r, "vkQueuePresentKHR");
    }
    g_cur = g_pro = g_epi = VK_NULL_HANDLE;
    g_value = 0;
    gpu_sync_end();
    return value;
}

// ---------- the back-end table
// Pools are the backing map's (oot2/backing.h): pool_fini returns once no submission in flight used the pool.
static int vk_pool_init(pool_t *p) { return backing_pool_init(p); }
// The entries over the pool's storage show the placeholder and its images can no longer be reached, before the pool's
// extent goes back to its block (the images are destroyed once no submission can read them).
static void vk_pool_fini(pool_t *p) {
    uint64_t lo = (uint64_t)(uintptr_t)p->mem;
    bindless_storage_gone(lo, lo + p->size);
    vkb_image_pool_gone(lo, lo + p->size);
    backing_pool_fini(p);
}
static int vk_pool_map_virtual(pool_t *v, int n, const void *requests) {
    return backing_pool_map_virtual(v, n, requests);
}

// The replay (oot2/vk_replay.h) resolves its bindings in the submission (a mirror's first resolve brings it current in
// the prologue) and declares its GPU writes (backing_gpu_wrote); its end reads back what it wrote into mirrors. The
// recorder thread submits the frame.
// The queue is not used yet: the replay merges every queue into one frame per submission and answers no WaitEvent by
// order (it checks and logs the wait at completion, vk_replay_report.c). Parking a queue at a wait splits that frame; the
// order engine of the null back end (oot2/nvn_order.h) is shaped for it, and the device checks decide the wiring.
static uint64_t vk_submit(uint64_t queue, const uint64_t *handles, int n) {
    (void)queue;
    return vkr_submit(handles, n);
}

static int vk_fence_wait(uint64_t value, uint64_t timeout_ns) { return gpu_sync_wait(value, timeout_ns); }
static uint64_t vk_fence_done(void) { return gpu_sync_done(); }
static void vk_finish(uint64_t queue) {   // one timeline: every queue's work is finished, which covers the queue's
    (void)queue;
    gpu_sync_finish();
}

// The bindless tables (oot2/bindless.h): a registration changes one entry of its pool's shadow table and appends one
// record to the change log; the copies of set 0 take it when a submission does.
static void vk_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    bindless_texture_register(pool, id, t, v);
}
static void vk_sampler_register(const dpool_ref *pool, int id, const samp_par *s) {
    bindless_sampler_register(pool, id, s);
}
// The picture goes to the offscreen capture (oot2/vk_capture.h: what the capture and the tools read, taken before the menu is
// drawn) and, when there is a window (vkw_window, oot2/vk_present.h), to the window; both after the work submitted before the call.
static void vk_present(const tex_t *t, const int crop[4]) {
    vkc_present(t, crop);
    if (vkw_active()) vkw_present(t, crop);
}
static void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                           uint64_t img) {
    vkb_read_texels(t, v, r, dst, row, img);
}
static void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src,
                            uint64_t row, uint64_t img) {
    vkb_write_texels(t, v, r, src, row, img);
}
// The serial's host objects may go: the image when nothing holds it. The entry at its ID stays (oot2/bindless.h).
static void vk_texture_release(const tex_t *t) { bindless_texture_release(t); }
// A finalized program's stage keys lose one reference; a key nothing holds any more has its pipelines destroyed once the
// timeline has passed the last frame that could use them (oot2/vk_miss.h).
static void vk_program_release(const prog_t *p) { vkm_program_release(p->serial); }
// The CPU's view of mirrored pool storage (oot2/backing.h, Mirrors).
static void vk_guest_written(uint64_t addr, uint64_t size) { backing_guest_written(addr, size); }
static void vk_guest_read(uint64_t addr, uint64_t size) { backing_guest_read(addr, size); }
static void vk_cpu_synced(uint64_t fence) { backing_cpu_synced(fence); }

const GfxBackend gfx_vulkan_backend = {
    .name = "vulkan",
    .pool_init = vk_pool_init,
    .pool_fini = vk_pool_fini,
    .pool_map_virtual = vk_pool_map_virtual,
    .texture_register = vk_texture_register,
    .sampler_register = vk_sampler_register,
    .texture_release = vk_texture_release,
    .program_release = vk_program_release,
    .submit = vk_submit,
    .fence_wait = vk_fence_wait,
    .fence_done = vk_fence_done,
    .present = vk_present,
    .finish = vk_finish,
    .read_texels = vk_read_texels,
    .write_texels = vk_write_texels,
    .guest_written = vk_guest_written,
    .guest_read = vk_guest_read,
    .cpu_synced = vk_cpu_synced,
    .present_pace = vkw_pace,
    .present_shown = vkw_shown,
};
