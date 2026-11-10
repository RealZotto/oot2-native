// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Vulkan back end (architecture section 6): its settings, the choice and creation of its device, its one queue,
// the submissions on the timeline of oot2/gpu.h, and the device-loss report (D17b). Every call goes through Volk
// (runtime/gpu/vk_volk.c), so a machine without a Vulkan loader starts and fails only when it asks for this back end.
//
// The back end opens once (gpu_vk_open, from gfx_select): the settings below are read then, the device is chosen and
// created with every item of oot2/vk_required.h enabled, and the completion thread starts over the device's timeline
// semaphore. One queue carries every submission, the replayed ones (GfxBackend.submit, every NVN2 queue in one order,
// as the null back end serialises them) and the back end's own (gpu_vk_begin to gpu_vk_end), in the order of their
// values.
//
// Any Vulkan call of the back end that fails stops the run by name (gpu_vk_failed). VK_ERROR_DEVICE_LOST first writes
// the device-loss report to the game folder's logs folder (gpu.h): the fault description, address and vendor
// information of VK_EXT_device_fault with the vendor's binary saved beside it, the last diagnostic checkpoints the queue
// reached, the timeline's values and the memory heaps; then the run stops with the report's path. A second thread that
// meets the loss waits for the first one's report and never returns.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/vk_required.h"

// ---- settings (read once, by gpu_vk_open; the settings store of the configuration menu will keep these names)
//   OOT2_VK_DEVICE       which device: unset or empty, the first discrete device with every item, else the first
//                        integrated, else the first other; digits below the device count, the device at that index of
//                        vkEnumeratePhysicalDevices; anything else, digits past the count included (3060), the first
//                        device whose name contains it, ignoring case
//   OOT2_VK_VALIDATION   off (the default), on (the Khronos validation layer), sync (with synchronization validation),
//                        gpu (with GPU-assisted validation); the layer must be installed when asked for
//   OOT2_VK_FAULT_REPORT on (the default): the device fault and checkpoint extensions are enabled and every submission
//                        carries checkpoints; off: neither extension is enabled and no checkpoint is recorded, so the
//                        report of a lost device holds the timeline and the heaps only
// A value outside these stops the start by name.
enum { GPU_VALIDATION_OFF, GPU_VALIDATION_ON, GPU_VALIDATION_SYNC, GPU_VALIDATION_GPU };
typedef struct GpuVkSettings {
    char device[256];
    int validation;     // GPU_VALIDATION_*
    int fault_report;   // 1 on, 0 off
} GpuVkSettings;
// The settings for these values of the three variables (NULL or "" for one not set): 1 with *out filled, or 0 with the
// reason in why.
int gpu_vk_settings(const char *device, const char *validation, const char *fault_report, GpuVkSettings *out,
                    char *why, size_t why_size);

// ---- the choice of device
typedef struct GpuVkCandidate {
    const char *name;
    VkPhysicalDeviceType type;
    unsigned char has[OOT2_VK_ITEM_COUNT];   // oot2_vk_judge
    const char *unusable;                    // NULL, or what else rules it out (no graphics and compute queue)
} GpuVkCandidate;
// The index of the device setting (GpuVkSettings.device) picks among c[0..n), or -1 with the reason in why: a device
// that lacks an item, or is unusable, is never picked, and why names every item it lacks.
int gpu_vk_choose(const GpuVkCandidate *c, int n, const char *setting, char *why, size_t why_size);

// ---- the device
typedef struct GpuVk {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
    VkPhysicalDevice phys;
    int index;                  // in vkEnumeratePhysicalDevices
    Oot2VkFacts facts;          // what the device has (oot2_vk_query)
    VkDevice device;
    uint32_t family;            // the queue family of the one queue
    VkQueueFlags family_flags;
    VkQueue queue;
    VkSemaphore timeline;
    VkCommandPool pool;         // the prologues and epilogues of submissions, used by the holder of the submission only
    VkCommandPool pool_main;    // their main command buffers, recorded by the recorder thread (oot2/vk_rec.h)
    GpuVkSettings settings;
    int fault, checkpoints;     // VK_EXT_device_fault (with its vendor binary when fault == 2), the checkpoints
    int image_2d_view_of_3d;    // image2DViewOf3D was enabled (an optional feature): a 3D image can be viewed as 2D for storage
} GpuVk;
extern GpuVk gpu_vk;            // valid once gpu_vk_open returned

void gpu_vk_open(void);         // once; later calls return at once
// The prepared game's shader packs and pipeline cache, for the pipeline builder (oot2/vk_pipeline.h): gfx_open_game's
// Vulkan half, after gpu_vk_open (vk_pipeline.c).
void gpu_vk_open_game(const char *folder);
// Finish every submission, stop the completion thread and destroy the device and the instance (tests; a run ends
// without it). gpu_vk_open may follow.
void gpu_vk_close(void);
// Messages of the validation layer since gpu_vk_open: errors and warnings (performance warnings included).
void gpu_vk_validation_counts(uint64_t *errors, uint64_t *warnings);

// Stop the run by name for result r of the call named what: the device-loss report first when r is
// VK_ERROR_DEVICE_LOST.
__attribute__((noreturn)) void gpu_vk_failed(VkResult r, const char *what);
#define GPU_VK_OK(call, what)                                                                                          \
    do {                                                                                                               \
        VkResult gpu_vk_r_ = (call);                                                                                   \
        if (gpu_vk_r_ != VK_SUCCESS) gpu_vk_failed(gpu_vk_r_, what);                                                   \
    } while (0)
const char *gpu_vk_result_name(VkResult r);   // "VK_ERROR_DEVICE_LOST", ...; "VkResult <n>" for others

// ---- submissions on the timeline (vk_submit.c)
// The back end's own work (tests, uploads, conversions, page-table writes) and the replay both submit this way:
// gpu_vk_begin takes the submission lock and returns the value the submission signals; gpu_vk_cmd gives the command
// buffer it records into (begun at the first call, with a checkpoint when the report is on; none is submitted when it is
// never asked for); gpu_vk_effect queues fn(arg, value) for the completion thread, after the device reached the value
// and before the value is done; gpu_vk_end ends the command buffer, submits it to the one queue after the given waits,
// signalling the value at all commands, and releases the lock. Only the thread between begin and end may call the
// others. kind is GPU_VK_INTERNAL or GPU_VK_REPLAY, written in the checkpoints. A host lock held across gpu_vk_begin is
// taken by every guest thread with rt_kernel_lock (kernel_lock.h), as gpu_sync_begin (oot2/gpu.h) says; a lock that is
// only taken inside a submission is taken after the gate, as vkt_untile_host does.
enum { GPU_VK_REPLAY = 1, GPU_VK_INTERNAL = 2 };
uint64_t gpu_vk_begin(int kind);
VkCommandBuffer gpu_vk_cmd(void);
// The submission's prologue: a second command buffer, submitted before gpu_vk_cmd's in the same queue submit, which
// begins with a full barrier over the work of earlier submissions and ends with one over the work after it, so what it
// records (the backing map's copies into mirrors) runs after every earlier submission and before this one's own work,
// whenever during the submission it was recorded. Begun at the first call; none is submitted when never asked for.
VkCommandBuffer gpu_vk_prologue(void);
// The submission's epilogue: a third command buffer, submitted after gpu_vk_cmd's, which begins with a full barrier over
// the work of the main command buffer, so what it records (the backing map's readbacks of mirrors) runs after the
// submission's own work. It is recorded by the thread that holds the submission, which is the replaying thread when the
// main command buffer is the recorder's (oot2/vk_rec.h); recording a readback into the main command buffer there would
// be recording from two threads. Begun at the first call; none is submitted when never asked for.
VkCommandBuffer gpu_vk_epilogue(void);
void gpu_vk_effect(GpuEffect fn, void *arg);
uint64_t gpu_vk_end(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits);
// gpu_vk_end that also signals extra_signal (a binary semaphore a present waits on; NULL for none) and, when present is not
// NULL, calls vkQueuePresentKHR(present) before it releases the submission: a present needs the queue to itself as a
// submit does. The present's own result goes to *present_result (VK_ERROR_OUT_OF_DATE_KHR and VK_SUBOPTIMAL_KHR are the
// caller's to handle); with present_result NULL anything but VK_SUCCESS stops the run by name.
uint64_t gpu_vk_end_present(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits, const VkSemaphoreSubmitInfo *extra_signal,
                            const VkPresentInfoKHR *present, VkResult *present_result);

// A checkpoint's marker: the value, the kind and 1 at the end of the submission's command buffer (0 at its start).
#define GPU_VK_MARKER(value, kind, end) ((uint64_t)(value) << 4 | (uint64_t)(kind) << 1 | (uint64_t)(end))
// The marker of a per-draw checkpoint (the recorder, oot2/vk_rec.h): kind GPU_VK_DRAW, the draw's number n in its
// submission (1 to GPU_VK_DRAW_MAX, counting draws, mesh draws and dispatches) in bits 4 to 27 and the value above.
enum { GPU_VK_DRAW = 4 };
#define GPU_VK_DRAW_MAX 0xFFFFFFu
#define GPU_VK_DRAW_MARKER(value, n) ((uint64_t)(value) << 28 | (uint64_t)(n) << 4 | (uint64_t)GPU_VK_DRAW << 1)

// The back-end table (vk_submit.c): submit, fence_wait, fence_done and finish on the timeline, the pools through the
// backing map (oot2/backing.h, opened with the device); the operations later tickets bring stop by name.
extern const GfxBackend gfx_vulkan_backend;
// The backing map's memory on this device (vk_backing.c): VRAM blocks, host imports, and the staging of mirrors' copies.
extern const BackingOps gpu_vk_backing_ops;
// The map's mirror hook for this device (gpu_vk_open, after backing_open): mirrors on a discrete device, imports on any
// other. A test that covers imports alone sets the hook to NULL after the device opens.
void gpu_vk_mirror_policy(void);
