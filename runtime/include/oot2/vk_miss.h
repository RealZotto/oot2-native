// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The miss path of the Vulkan back end (architecture section 6.1, rule 3 of the build policy; the NVIDIA fault study): what the
// replay does when a draw or dispatch asks for a pipeline that nothing built ahead (oot2/vk_pipeline.h), and what it does
// when a program goes (gfx.h program_release). The replay's ops (oot2/vk_replay.h) point at vkm_graphics and
// vkm_compute, so the replay itself only learns that a draw may be skipped.
//
// A miss is an exception, a defect for the gates of M3 and M4: prepare builds every library and compute pipeline, and a
// run that misses says so. Its cost is defined. The replay thread, which holds the submission gate, stops; the recorder's
// ring is drained (every packet of the frame so far has made its Vulkan call) and then the timeline (every submission
// handed to the queue is done), so the queue is idle; the libraries are built, the vertex and the fragment library of a
// graphics pipeline side by side, and linked; the replay resumes and the draw is made. The identity of the stages and the
// stall are logged and counted, and the capture's counters take the miss (CAPTURE_PIPELINE_MISSES).
//
// The bound. A graphics build is made by a job thread, and the replay waits for it for at most the bound (2 s, as the
// architecture proposes: a raw-access vertex module took 7 to 10 s on one thread before the raw-access lowering (the pipeline
// cost study), and a cold mesh library 6 s at p99 (the mesh pipeline cost study); a library built from the player's own cache takes a millisecond).
// Past the bound the draw is skipped, with a named line and a count (CAPTURE_SKIPPED_DRAWS); the build goes on in its
// job and the draws after it that name the same pipeline are skipped at once until it is done. That is the one place
// where a pipeline is created with work in flight: the job's flag (vkp_graphics_ex) tells the builder that the wait is
// over, and `late_builds` counts them. A skip is allowed only where the draw's colour and depth targets hold the picture
// of an earlier frame (the Hookshot survey: a draw into a target this frame has not touched yet leaves last frame's picture, a
// draw into a target this frame already wrote leaves a hole): the replay says so with may_skip, and a draw that may not
// be skipped waits without a bound, counted apart (`waits_kept`). A dispatch is never skipped, a dropped dispatch is lost
// work (compute and mesh misses wait without a bound, and `compute_misses` counts the dispatch waits); a stage with no
// module in the shader cache is not a miss, it is a named fatal of the builder (there is no run-time translation, so
// "program not ready" is 0 by construction).
//
// The override, for development only and off by default: OOT2_VK_MISS_BOUND=unbounded makes every graphics build
// wait without a bound, so that a run can reach zero skipped draws while the counters still say what the production
// bound would have done (`would_skip`: the waits that passed the bound). OOT2_VK_MISS_BOUND=<milliseconds> sets another
// bound; any other value is a named fatal.
//
// program_release. A stage key's pipelines go once no program holds the key and no submission that can use them is in
// flight. The replay tells vkm_program_bound which program (by serial) holds which stage keys when it first binds it;
// vkm_program_release, from gfx.h's program_release, takes the program's references away; a key left with none is given
// the newest value handed out (the open frame included: its commands may name the pipeline), and its objects are
// destroyed once the timeline has published that value (vkm_retire, between frames and after a drain). A program bound
// again before then takes the key back and the release is cancelled.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/vk_pipeline.h"

struct nvn_stage;

// What the miss path asks of the builder and the timeline: the device's, or a test's stand-ins.
typedef struct VkmOps {
    VkPipeline (*find_graphics)(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats);
    VkPipeline (*find_compute)(const VkpStage *cs);
    VkPipeline (*build_graphics)(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                                 const int *late);
    VkPipeline (*build_compute)(const VkpStage *cs);
    void (*graphics_key)(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                         char *out, size_t cap);
    // Wait until the recorder has made every call it was given (the ring is empty), then until every submission handed
    // to the queue is done; returns how many submissions that was.
    uint64_t (*drain)(void);
    void (*release)(const char *key);   // vkp_release
    uint64_t (*handed)(void);           // gpu_sync_handed
    uint64_t (*done)(void);             // gpu_sync_done
    uint64_t (*now_ns)(void);
} VkmOps;

// Use these ops from now on (NULL: the device's).
void vkm_set_ops(const VkmOps *ops);

typedef struct VkmConfig {
    uint64_t bound_ns;
    int unbounded;
} VkmConfig;
#define VKM_DEFAULT_BOUND_NS 2000000000ull
// Use this configuration (tests); NULL reads OOT2_VK_MISS_BOUND again at the next miss.
void vkm_set_config(const VkmConfig *c);

// The pipeline of a graphics draw: the one that exists, else the miss path's. VK_NULL_HANDLE is a draw to skip (only
// when may_skip is set). Called by the replay with the submission gate held.
VkPipeline vkm_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                        int may_skip);
// The pipeline of a dispatch: the one that exists, else built after a drain; never skipped.
VkPipeline vkm_compute(const VkpStage *cs);

// The replay met program `serial` holding these stage records (first bind of the serial; later calls do nothing).
void vkm_program_bound(uint64_t serial, const struct nvn_stage *const *stages, int n);
// gfx.h program_release: the program's serial is finalized (any thread).
void vkm_program_release(uint64_t serial);
// Destroy the objects of every key whose last program went and whose value the timeline has published. Between frames.
void vkm_retire(void);
// Wait for every build still running, retire what can be, print the summary line. Once the device is idle.
void vkm_close(void);

typedef struct VkmStats {
    uint64_t graphics_misses, compute_misses;       // misses by kind (compute: the dispatches that waited)
    uint64_t drains, drain_ns;                      // one per miss that built
    uint64_t drained_submissions;                   // submissions in flight that the drains waited for
    uint64_t stall_ns[2], stall_max_ns[2];          // the replay thread's wait, per kind (graphics, compute), drain included
    uint64_t skipped_draws;                         // graphics draws skipped past the bound
    uint64_t skipped_pending;                       // of them, skipped at once because the build was still running
    uint64_t would_skip;                            // waits past the bound that the override waited out
    uint64_t waits_kept;                            // waits past the bound where the draw may not be skipped
    uint64_t late_builds;                           // builds that finished after the replay stopped waiting
    uint64_t keys_released, keys_pending;           // stage keys whose pipelines were destroyed; waiting for retirement now
    uint64_t programs_bound, programs_released;
} VkmStats;
void vkm_stats(VkmStats *out);
void vkm_stats_reset(void);   // tests

// Tests only (the counter-runs): while non-zero, the drain is not made ("the build waits for an idle queue"); a skip is
// not counted; a released key is destroyed at once ("destroyed in flight"); a build the replay stopped waiting for is
// not told so (it would then meet the rule against creating with work in flight).
extern int vkm_no_drain, vkm_no_count_skip, vkm_release_at_once, vkm_no_late;
