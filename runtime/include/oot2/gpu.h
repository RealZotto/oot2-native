// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Vulkan back end's module, runtime/gpu (architecture section 2 and section 6): which graphics back end a start
// installs, the runtime's own shaders, and the parts of the back end that need no Vulkan types: the timeline that
// orders every submission and its completion thread, the device-loss report's file, and the check of a recording the
// back end cannot replay yet. The module is built against the Vulkan headers and Volk on every machine
// (tools/build_game.py), so the back end's tickets only add code. The Vulkan half (the device, the submissions, the
// table) is oot2/gpu_vk.h.
#pragma once
#include <stddef.h>
#include <stdint.h>

// ---- the back-end choice (gpu.c)
// The back end a start installs when neither `oot2 --gfx NAME` nor the OOT2_GFX variable names one. It stays "null"
// until the title runs on the Vulkan back end: the boot to the first presents and every continuous-integration run use
// the null one. The
// commit that changes it also makes the check step of `oot2 prepare` require every Vulkan device item
// (tools/prepare/check.py, REQUIRE_VULKAN_ITEMS); tests/test_prepare_check.py ties the two.
#define GFX_DEFAULT_BACKEND "null"

// 1 when name is a back end this program knows ("null" or "vulkan").
int gfx_backend_known(const char *name);
// Install the back end named name with gfx_set_backend (so before rt_dispatch_init) and log it; source says who asked
// ("--gfx", "OOT2_GFX", "the default"). Vulkan opens its device first (gpu_vk_open, oot2/gpu_vk.h), which stops the
// start by name when there is no loader, no device, or no device with every item of oot2/vk_required.h; an unknown
// name is a named fatal.
void gfx_select(const char *name, const char *source);
// After gfx_select, with the game folder: what the installed back end reads from the prepared game. Vulkan opens the
// shader packs (shaders/shader-base.bin, and shaders/shader-variants.bin when prepare's ledger step wrote it) and the
// pipeline cache (pipelines/pipeline.cache, seeded when it names this device and driver; an empty cache when prepare
// skipped its pipeline step), and stops the start by name when the base pack is missing or refused. Null reads nothing.
void gfx_open_game(const char *folder);

// ---- the runtime's own shaders (runtime/gpu/shaders)
// GLSL compute shaders, compiled by tools/build_game.py with glslang for Vulkan 1.3 and validated at every build of the
// runtime library, then embedded in the program as gpu_shaders (the build folder's spirv/gpu_shaders.c), in name order.
typedef struct GpuShader {
    const char *name;        // the file name without .comp
    const uint32_t *code;    // SPIR-V words
    size_t size;             // bytes
} GpuShader;

extern const GpuShader gpu_shaders[];
extern const int gpu_shader_count;
// The embedded shader named name, or NULL.
const GpuShader *gpu_shader(const char *name);

// ---- the timeline (gpu_sync.c; architecture section 5 and section 6.1, the completion thread)
// One counter orders every submission, the replayed ones and the back end's own: each takes the next value, its GPU
// work signals that value, and a completion thread runs the CPU-visible effects queued for each value, in value order
// and in the order they were added, once the device reached the value, and only then publishes it. A value is done
// (gpu_sync_wait, gpu_sync_done, so the back end's fence_wait and fence_done) only after its own effects and every
// earlier value's ran, so a guest woken by a fence sees them. A source tells the completion thread how far the device
// got: the Vulkan back end's timeline semaphore, or a test's counter.
typedef struct GpuTimelineSource {
    // Wait up to timeout_ns for the device's counter to reach value and return the counter, which is below value after
    // a timeout. A source that finds the device lost stops the run itself and does not return.
    uint64_t (*wait)(void *ctx, uint64_t value, uint64_t timeout_ns);
    void *ctx;
} GpuTimelineSource;

// Start the completion thread over src; values go on from `after` (0 for a first device). A fatal while it runs.
void gpu_sync_start(const GpuTimelineSource *src, uint64_t after);
// Stop it, with nothing in flight (gpu_sync_finish first: a fatal otherwise). gpu_sync_start may follow, from
// gpu_sync_done().
void gpu_sync_stop(void);

typedef void (*GpuEffect)(void *arg, uint64_t value);
// A submission: gpu_sync_begin takes the submission gate, waiting while another submission holds it, and hands out the
// next value; each gpu_sync_effect queues fn(arg, value) for the completion thread; gpu_sync_end, called once the GPU
// work that signals the value has been submitted, issues the value and opens the gate. Values are issued in the order
// they are handed out, whichever thread submits. The gate is not a lock owned by a thread: the thread that ends a
// submission may be another than the one that began it (the recorder ends the frames the submitting thread begins), as
// long as the two hand over with a release and an acquire of their own, and every effect is queued before the end. A
// thread that holds a submission must not wait for its own value, nor begin another submission.
// A guest thread that finds the gate taken waits for it without its virtual core (the holder may be waiting for one) and
// takes one again once the gate is its. So a host lock the caller holds across the call is taken by every guest thread with
// rt_kernel_lock (kernel_lock.h), or a thread on that core that waits for the lock with its core held would keep the core
// from the caller, which then never gets to open the gate.
uint64_t gpu_sync_begin(void);
void gpu_sync_effect(GpuEffect fn, void *arg);
void gpu_sync_end(void);
// 1 once value is done, 0 when timeout_ns passed first (UINT64_MAX: no limit); a value not handed out yet answers 0 at
// once. The caller has given its virtual core up for the wait (D7).
int gpu_sync_wait(uint64_t value, uint64_t timeout_ns);
uint64_t gpu_sync_done(void);     // the newest value done
uint64_t gpu_sync_handed(void);   // the newest value handed out
uint64_t gpu_sync_issued(void);   // the newest value whose GPU work was submitted
// The value of the submission that is open (begun, not ended), or 0: its work is not on the queue yet. Read it with
// gpu_sync_handed: it is the open submission when the two are equal.
uint64_t gpu_sync_open(void);
void gpu_sync_finish(void);       // returns once every value handed out before the call is done

// ---- the device-loss report's file (vk_loss.c)
// The game folder (main.c, once it is resolved; the current folder until then). A report goes to its `logs` folder.
void gpu_set_report_folder(const char *dir);
#define GPU_REPORT_DIR "logs"
// Write text, and size bytes of bin when size > 0, as dir/logs/device-lost-<local date>-<time>.txt and .bin beside it
// (the folder is made when missing; -2, -3 and on are added to a name already taken). 1 written, with the text file's
// path in path; 0 when it could not be, with what failed in path.
int gpu_write_report(const char *dir, const char *text, const void *bin, size_t size, char *path, size_t path_size);
