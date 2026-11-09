// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay of the Vulkan back end (architecture D11, D12, D17a, D18; the Vulkan replay): GfxBackend.submit walks the host
// recordings the front end hands it (nvn_obj.h, command records), on the submitting thread, and turns each record into
// the recorder's packets (oot2/vk_rec.h) for one frame: the frame's value is the fence value submit returns.
//
// What the walk keeps. The bound state of one GPU channel across records and across the handles of a submission, as
// the console's channel keeps it between submissions: render targets, viewports, scissors, depth ranges, state objects,
// vertex attributes, streams and buffers, each stage's uniform and storage buffers and texture, image and sampler
// handles, the program, push constants and the descriptor pools. The program's stages and the attachment formats come
// from the ledger's tracker (oot2/ledger.h), which the walk feeds every record, so the null back end and this replay
// share the one walk of bound types. State persists from one submission to the next, as it does on the console.
//
// What a draw or dispatch resolves, at the record (VkReplayOps below; the device's own in runtime/gpu/vk_replay_dev.c):
//   pipeline     the stages' job keys and the attachment formats, through the pipeline builder's blocking get
//                (oot2/vk_pipeline.h) through the miss path (oot2/vk_miss.h): one not built yet is drained for, built, and
//                a graphics draw past the bound is skipped (nothing of it is written)
//   buffers      vertex, index, uniform, storage and indirect ranges through the backing map's resolver
//                (oot2/backing.h), in the frame's submission
//   banks        per stage the driver bank c[0] (the storage buffer table at the stage's base, the base instance, the
//                largest point size; NVN2 compute: a 16-byte head with the group counts, the root tables, then the
//                uniform and storage tables in slot order, sized by the control section), the texture handle bank c[2],
//                the program's own constants c[1] (its code area), the root tables as bank 24, and the support buffer
//                (viewport inverse and size, render scales 1.0), written into the frame's scratch memory
//   storage      a storage buffer binds from the address and size words of its bank (address rounded down to 16 bytes,
//                range the size plus what was rounded off: the first draw study); one the module may write (no NonWritable,
//                the description's `write`) is declared to the backing map (backing_gpu_wrote) and listed for the copies
//                ticket's write-back (vkr_writes)
//   textures     each push constant slot the module reads takes the handle bound at its stage's c[2] slot: the
//                texture ID mapped to its slot of set 0 for the view type the module declares (the bindless tables),
//                the sampler ID in bits 20 to 31
//   snapshot     a uniform range in host memory the GPU reads live (an import) is copied into the frame's scratch at
//                submit, and the draw binds the copy (D17a); a mirror is already a copy taken at submit (oot2/backing.h),
//                and VRAM is the GPU's. The drop rule: a GPU write of the same submission over a copied range (an update
//                of a uniform buffer, a buffer copy or fill, a texture copied to a buffer, a storage store) drops the
//                copy, and every read of the range after the write binds the live memory, which the GPU wrote in order;
//                the reads before it keep their copy. A range written whole or in part is read live from then on
//                (DKB Recomp's vk_usnap_drop does the same).
//   modes        nvn_depth_mode NEAR_IS_MINUS_W sets depth clip control's -1 to 1 range (dynamic state); ZERO_TO_ONE
//                Vulkan's own. nvn_window_origin_mode LOWER_LEFT flips Y through the viewport (a negative height from the
//                viewport's top), the translator's convention (window origin upper left, Y flipped by the viewport)
//
// Passes (D18). Rendering begins lazily at the first draw or clear after a render target change, with every image in
// the general layout, loading and storing; a render target change or a barrier ends the pass, and a full memory barrier
// separates passes. A barrier inside a pass ends it.
//
// Transfers (vk_replay_xfer.c). Buffer and texture copies, buffer fills, texture clears and uniform updates are one
// transfer command each, after the end of the open pass and a full memory barrier; the next dispatch or transfer opens
// with a barrier too. A copy's two sides are resolved through the backing map in the frame's submission, a write into
// memory the CPU can see is declared (backing_gpu_wrote: a mirror writes it back to guest memory once the frame is done,
// an import is the guest's own memory), and the row and image strides of the buffer side are the channel's
// (OP_COPY_STRIDE and OP_COPY_IMAGE_STRIDE, 0 meaning tightly packed). A texture is its image in the general layout
// (oot2/vk_image.h), so a copy needs one: a texture over memory the CPU can see (the texture arena) or in a format the
// device lacks is a named stop. A dispatch's indirect group counts are copied by the GPU into the driver bank c[0] just
// before the dispatch: they may be the output of earlier work of the frame, or of an earlier frame still in flight.
//
// Reports, events and conditions (vk_replay_report.c, gfx.h's Synchronisation). Event signals, counter reports and
// WaitEvent checks are copied in op-stream order and done by one effect on the completion thread, queued after the
// frame's end (after the backing map's write-backs of the frame), before the frame's value is done. A report's
// timestamp is the guest's tick when it completes; samples passed are counted by occlusion queries, one per pass from
// the channel's first samples-passed reset or report on, summed in order on the completion thread; every other counter
// reports 0. SetRenderEnableConditional is decided at replay from earlier reports (a report not done yet counts as 0, or
// as visible for samples passed). A report into memory the CPU cannot see is dropped.
//
// Named stops. Every op of nvn_obj.h's list is replayed or stops by name (a named fatal, rt_fatal: the replay has no
// guest CPU to stop): vkr_op_ticket names the ticket a stopped op waits for, and the stop's message names the op and
// the ticket. So does any state the replay cannot say yet (a masked clear, a storage image, a separate sampler).
//
// The recorder (oot2/vk_rec.h) opens with the first replayed submission unless a test opened it first; gpu_vk_close
// closes it.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/nvn_obj.h"
#include "oot2/vk_pipeline.h"

struct nvn_stage;

// ---- the op list: replayed, or the ticket a stopped op waits for
#define VKR_REPLAYED NULL
// NULL for an op this replay makes packets of (or ignores by contract, as OP_HINT), else the name of what it waits for
// ("task and mesh draws (the mesh draws)", ...). Past OP_COUNT_: "unknown".
const char *vkr_op_ticket(uint32_t op);

// ---- what the replay needs of a device (the device's, or a test's stand-ins: continuous integration checks the
// op-to-packet resolution without one)
typedef struct VkrTarget {
    VkImageView view;            // a view of one level and the layers the record names, in the attachment's format
    VkFormat format;
    uint32_t width, height, layers;
    VkImageAspectFlags aspect;   // colour, or depth and stencil as the format has them
    uint64_t image;              // the image's identity across frames (the miss path's rule on skipping; 0: none)
} VkrTarget;

// A texture's image for a copy or a clear: the image the texture's storage stands on, in the general layout.
typedef struct VkrImage {
    VkImage image;
    VkFormat format;
    VkImageType type;
    VkImageAspectFlags aspect;       // the one a copy takes (depth for a depth-stencil format, unless the texture reads stencil)
    VkImageAspectFlags all_aspects;  // every aspect of the format (a clear takes them all)
    uint32_t width, height, depth, layers, levels, samples;   // level 0; layers: a cube's six faces and an array's layers
} VkrImage;

typedef struct VkReplayOps {
    // The pipeline of a graphics draw and of a dispatch (the device's: the miss path, oot2/vk_miss.h) and the layouts
    // the replay binds against. graphics answers NULL for a draw to skip, which it does only when may_skip is set (every
    // colour and depth target of the pass holds an earlier frame's picture); compute never does.
    VkPipeline (*graphics)(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                           int may_skip);
    VkPipeline (*compute)(const VkpStage *cs);
    // A bind of a program (by serial, with its stage records), at every bind: the device counts the stage keys live
    // programs hold, and a repeated bind of a serial changes nothing unless the program was finalized since
    // (oot2/vk_miss.h). May be NULL.
    void (*program_bound)(uint64_t serial, const struct nvn_stage *const *stages, int n);
    VkPipelineLayout graphics_layout, compute_layout;
    // The description of a stage (oot2/shader_desc.h), parsed once per stage record and kept: NULL is a named stop.
    const struct ShaderBinds *(*binds)(const VkpStage *stage);
    // The backing map (backing_resolve, backing_gpu_wrote) in the frame's submission.
    void (*resolve)(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out);
    void (*gpu_wrote)(uint64_t addr, uint64_t size, uint64_t value);
    // A render target or storage image: the texture as the record holds it, for the frame `value`; 0 when no image
    // stands on its storage (a named stop follows). The view lives until the frame is done.
    int (*target)(const nvn_tex_rec *t, int storage, uint64_t value, VkrTarget *out);
    // The image a texture's storage stands on, for a copy or a clear in the frame `value`: 0 when none does (a named
    // stop follows).
    int (*image)(const nvn_tex_rec *t, uint64_t value, VkrImage *out);
    // The bindless tables for the frame (vkb_acquire, recorded into the frame's prologue) and the slot of a texture ID
    // seen as a view type (vkb_image_slot); UINT32_MAX for none.
    VkDescriptorSet (*acquire)(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers);
    uint32_t (*image_slot)(uint32_t id, int type);
    // A storage view of the texture registered at texture pool ID `id`, of view type `type` (BV_*), for the frame
    // `value` and freed once it is done (vkb_storage_view); VK_NULL_HANDLE for an ID nothing is registered at. *storage
    // and *bytes: the texture's storage, for the written ranges, when its image is placed in it (0 bytes otherwise: an
    // arena image holds the stores itself). NULL: a storage image stops by name.
    VkImageView (*storage_image)(uint32_t id, int type, uint64_t value, uint64_t *storage, uint64_t *bytes);
    // `bytes` of host-visible memory the device reads, for the frame `value` (freed once it is done): the pointer, and
    // the buffer and offset that bind it. Offsets are aligned to `align`.
    void *(*scratch)(uint64_t value, uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset);
    // The device address of a replay scratch buffer (mesh draws use one word for first_task).
    VkDeviceAddress (*buffer_address)(VkBuffer buffer);
    // Allocate and write a set 2 for a task stage; NULL is a named stop when the stage has descriptors.
    VkDescriptorSet (*task_set)(uint64_t value, uint32_t n, const VkWriteDescriptorSet *writes);
    // The page table for a raw-access stage of the frame `value` (gpu_vk_raw_begin): the push words 0 to 3. NULL: a
    // raw-access stage stops by name.
    void (*raw)(uint64_t value, uint32_t push[4]);
    // Around a frame: before its value is taken (the backing map's sync), and after its last packet, before the frame
    // is handed over (the images made for it brought to the general layout, the backing map's end of submission).
    // Either may be NULL.
    void (*frame_begin)(void);
    void (*frame_end)(uint64_t value);
    // Occlusion queries for the samples-passed counter: a pool of VKR_QUERIES for the frame `value`, reset before the
    // frame's work (the prologue); and, on the completion thread once the frame is done, the results of its first n
    // queries into out, after which the pool is taken back. NULL when the ops count nothing (a frame that counts stops
    // by name).
    VkQueryPool (*queries)(uint64_t value);
    void (*query_results)(VkQueryPool pool, uint32_t n, uint64_t *out);
    // 1 when the CPU may read [addr, addr + size) now (a memory pool it sees holds it): the words of a conditional
    // rendering condition. NULL: never (the condition renders).
    int (*cpu_reads)(uint64_t addr, uint64_t size);
} VkReplayOps;
#define VKR_QUERIES 512u       // occlusion queries in one pool
#define VKR_QUERY_POOLS 8u     // pools a frame may take (one query per pass while counting, one more per report)

// Use these ops from now on (NULL: the device's, vkr_device_ops). A test's ops stay valid while it replays.
void vkr_set_ops(const VkReplayOps *ops);
const VkReplayOps *vkr_device_ops(void);
// Free what the device's ops hold (scratch memory, the parsed descriptions); gpu_vk_close, once every frame is done.
void vkr_device_close(void);

// ---- the replay
// GfxBackend.submit of the Vulkan back end: a frame through the recorder (vkrec_begin, the packets, the backing map's
// end of submission, vkrec_end); returns the frame's value at once.
uint64_t vkr_submit(const uint64_t *handles, int n);
// Forget the bound state (tests; a run never does: a channel keeps its state).
void vkr_reset(void);

// Tests only: while non-zero, uniform ranges are bound live and never copied (the counter-run of the snapshot).
extern int vkr_no_snapshot;
// Tests only: while non-zero, the window origin's Y flip is not applied (the counter-run of the origin mode).
extern int vkr_no_flip;
// Tests only: while non-zero, a GPU write drops no uniform copy (the counter-run of the drop rule).
extern int vkr_no_drop;
// Tests only: while non-zero, a GPU write is declared to no one: not to the backing map (the counter-run of the write-back).
extern int vkr_no_declare;
// Tests only: while non-zero, the dynamic state of the core features (line and point fill, depth bias clamp, depth
// clamp, depth bounds, wide lines, more than one viewport) is left at the default a device without them would say.
extern int vkr_no_features;
// Tests only: while non-zero, mesh draws push no first-task address (the counter-run of the mesh ABI).
extern int vkr_no_first_task;
// Tests only, the counter-runs of the reports (vk_replay_report.c): the condition reads memory even under a report not
// done yet; samples passed are never counted; a timestamp is the tick at replay, not at completion; the completion
// thread walks a frame's events and reports backwards.
extern int vkr_no_pending, vkr_no_queries, vkr_stamp_at_replay, vkr_reverse_results;

// ---- what the last frames wrote: the storage and image ranges a draw or dispatch may write, and the ranges a copy, a
// fill or an update wrote, for the frame `value`; *n how many. Kept for the frames in flight.
typedef struct VkrWrite {
    uint64_t addr, size;
    uint32_t draw;     // the draw or dispatch's number in its frame, from 1; 0 for a copy, a fill or an update
    uint8_t image;     // 1 for a storage image's storage, 0 for a storage buffer
} VkrWrite;
const VkrWrite *vkr_writes(uint64_t value, uint32_t *n);

typedef struct VkrStats {
    uint64_t frames, records, draws, dispatches, clears, passes, barriers;
    uint64_t snapshots, snapshot_bytes;             // uniform ranges copied at submit (D17a)
    uint64_t snapshot_drops;                        // copies a GPU write of the same submission dropped
    uint64_t transfers, transfer_bytes;             // copies, fills, updates and texture clears, and the bytes they move
    uint64_t live_uniforms;                         // uniform ranges bound live (a mirror or VRAM)
    uint64_t storage_writes;                        // storage ranges declared written
    uint64_t storage_images;                        // storage images bound (a view of a texture pool ID each)
    uint64_t unregistered_textures;                 // slots whose texture ID had no entry (the placeholder is read)
    uint64_t scratch_bytes;
    // Recorded against replayed (the Hookshot survey): the draw and dispatch records that reached the replay, those not made
    // because the game had rendering off, and the graphics draws skipped for a pipeline not ready (oot2/vk_miss.h).
    // lost is what none of those accounts for: draw_ops less draws, dispatches, render_off and skipped (in either
    // direction), 0 by contract; read it before judging a picture.
    uint64_t draw_ops, render_off, skipped, lost;
    uint64_t raw_binds;                             // draws and dispatches with a raw-access stage (D15)
} VkrStats;
void vkr_stats(VkrStats *out);

// ---- reports, events and waits as the completion thread did them, and the conditions decided at replay
typedef struct VkrReportStats {
    uint64_t reports, dropped;              // counter reports written, and those into memory the CPU cannot see
    uint64_t resets, events, queries;       // samples-passed resets, event signals, occlusion queries read back
    uint64_t waits, waits_unsatisfied;      // WaitEvents checked, and those the work before them did not satisfy
    uint64_t conditions, conditions_false;  // SetRenderEnableConditional decided, and those that disabled rendering
} VkrReportStats;
void vkr_report_stats(VkrReportStats *out);
