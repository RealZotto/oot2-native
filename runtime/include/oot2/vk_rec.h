// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The recorder of the Vulkan back end (architecture D12, D17b, sections 6.1 and 6.2; the recorder, the pipeline cost study).
// The replay runs on the submitting thread and does not make Vulkan calls: it writes each command of a frame as a packet
// into a 16 MiB ring (oot2/gpu_ring.h), and a recorder thread reads the packets in order, makes the calls into the
// frame's command buffer and submits it on the timeline (oot2/gpu_vk.h). The replay's thread time is then the walk of
// the ops and the packing, which the pipeline cost study measured at a third less than recording itself.
//
// A frame is one submission: vkrec_begin waits until fewer than `frames` (2 unless the setting frames_in_flight says more: section 6.1) frames are between their
// begin and the device's completion, then takes the submission gate and the next timeline value (the fence value the
// replay hands the guest), the vkrec_* calls write the frame's packets, and vkrec_end hands the frame over and returns at
// once; the recorder submits it and opens the gate. So (with 2) a third frame's begin waits for the first to be done, and any
// other submission (gpu_vk_begin from another thread) waits behind the frame being replayed, as before. Between the two
// calls the replaying thread makes no other submission. The frame's main command buffer (gpu_vk_cmd, from a pool of its
// own) is the recorder's alone; the replaying thread records the frame's prologue and epilogue itself (gpu_vk_prologue,
// gpu_vk_epilogue: the backing map's mirror uploads and readbacks), since command buffers of one pool are not recorded
// by two threads at once, and the recorder ends them with the main one when it submits.
//
// Packets. Every command the uni layout's draws need (architecture D13), in the order a replay emits them: pipelines and
// vertex or index buffers, descriptor sets, push descriptors, push constants, the dynamic state the pipeline libraries
// leave open (oot2/vk_pipeline.h vkp_dynamic_states), draws, mesh draws and dispatches, barriers, rendering begin and end.
// Each vkrec_X below has the parameters of vkCmdX less the command buffer and copies what they point to into the packet,
// so the caller's storage is free again at once. A structure with a pNext chain has no packet: a named fatal (nothing the
// replay builds has one). A command the later tickets bring (queries, timestamps, conditional rendering)
// is one more line in VKREC_FIXED or one more hand-written pair in vk_rec_pack.c.
//
// Breadcrumbs (D17b, the NVIDIA fault study). Every draw, mesh draw and dispatch gets a number in its frame (from 1) and a record
// of the stage identities in effect when it was written (vkrec_stages); the recorder sets a diagnostic checkpoint whose
// marker is GPU_VK_DRAW_MARKER(value, n) right before the call, so the device-loss report (vk_loss.c) can name the last draw
// that ran by its stages. Records live in tables of the frames in flight, which is as long as a checkpoint can name them.
// They are off with the checkpoints (OOT2_VK_FAULT_REPORT=off).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/gpu.h"
#include "oot2/gpu_ring.h"

struct nvn_stage;

// ---- the command list. F: a command with scalar arguments; D: a draw, mesh draw or dispatch, which also carries the
// number of its breadcrumb. F(name, (params), (pack args), (call args), fields); D(name, kind, ...) likewise, with the
// breadcrumb's kind letter after the name. The call args name the command buffer `cb` and the packet `a`.
#define VKREC_FIXED(F)                                                                                                 \
    F(BindPipeline, (VkPipelineBindPoint bp, VkPipeline p), (bp, p), (cb, a->bp, a->p),                                \
      VkPipelineBindPoint bp; VkPipeline p;)                                                                           \
    F(BindIndexBuffer, (VkBuffer b, VkDeviceSize o, VkIndexType t), (b, o, t), (cb, a->b, a->o, a->t),                 \
      VkBuffer b; VkDeviceSize o; VkIndexType t;)                                                                      \
    F(SetCullMode, (VkCullModeFlags v), (v), (cb, a->v), VkCullModeFlags v;)                                           \
    F(SetFrontFace, (VkFrontFace v), (v), (cb, a->v), VkFrontFace v;)                                                  \
    F(SetPrimitiveTopology, (VkPrimitiveTopology v), (v), (cb, a->v), VkPrimitiveTopology v;)                          \
    F(SetPrimitiveRestartEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                           \
    F(SetRasterizerDiscardEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                          \
    F(SetDepthBiasEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                                  \
    F(SetDepthBias, (float c, float cl, float s), (c, cl, s), (cb, a->c, a->cl, a->s), float c, cl, s;)                \
    F(SetLineWidth, (float w), (w), (cb, a->w), float w;)                                                              \
    F(SetDepthTestEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                                  \
    F(SetDepthWriteEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                                 \
    F(SetDepthCompareOp, (VkCompareOp v), (v), (cb, a->v), VkCompareOp v;)                                             \
    F(SetDepthBoundsTestEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                            \
    F(SetDepthBounds, (float lo, float hi), (lo, hi), (cb, a->lo, a->hi), float lo, hi;)                               \
    F(SetStencilTestEnable, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                                \
    F(SetStencilOp, (VkStencilFaceFlags f, VkStencilOp fail, VkStencilOp pass, VkStencilOp dfail, VkCompareOp cmp),    \
      (f, fail, pass, dfail, cmp), (cb, a->f, a->fail, a->pass, a->dfail, a->cmp),                                     \
      VkStencilFaceFlags f; VkStencilOp fail, pass, dfail; VkCompareOp cmp;)                                           \
    F(SetStencilCompareMask, (VkStencilFaceFlags f, uint32_t v), (f, v), (cb, a->f, a->v),                             \
      VkStencilFaceFlags f; uint32_t v;)                                                                               \
    F(SetStencilWriteMask, (VkStencilFaceFlags f, uint32_t v), (f, v), (cb, a->f, a->v),                               \
      VkStencilFaceFlags f; uint32_t v;)                                                                               \
    F(SetStencilReference, (VkStencilFaceFlags f, uint32_t v), (f, v), (cb, a->f, a->v),                               \
      VkStencilFaceFlags f; uint32_t v;)                                                                               \
    F(SetPolygonModeEXT, (VkPolygonMode v), (v), (cb, a->v), VkPolygonMode v;)                                         \
    F(SetDepthClampEnableEXT, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                              \
    F(SetLogicOpEnableEXT, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                                 \
    F(SetDepthClipNegativeOneToOneEXT, (VkBool32 v), (v), (cb, a->v), VkBool32 v;)                                     \
    F(FillBuffer, (VkBuffer b, VkDeviceSize o, VkDeviceSize n, uint32_t v), (b, o, n, v),                              \
      (cb, a->b, a->o, a->n, a->v), VkBuffer b; VkDeviceSize o, n; uint32_t v;)                                        \
    F(BeginQuery, (VkQueryPool p, uint32_t q, VkQueryControlFlags f), (p, q, f), (cb, a->p, a->q, a->f),               \
      VkQueryPool p; uint32_t q; VkQueryControlFlags f;)                                                               \
    F(EndQuery, (VkQueryPool p, uint32_t q), (p, q), (cb, a->p, a->q), VkQueryPool p; uint32_t q;)                     \
    F(WriteTimestamp2, (VkPipelineStageFlags2 s, VkQueryPool p, uint32_t q), (s, p, q), (cb, a->s, a->p, a->q),        \
      VkPipelineStageFlags2 s; VkQueryPool p; uint32_t q;)

#define VKREC_DRAWS(D)                                                                                                 \
    D(Draw, 'd', (uint32_t vc, uint32_t ic, uint32_t fv, uint32_t fi), (vc, ic, fv, fi),                               \
      (cb, a->vc, a->ic, a->fv, a->fi), uint32_t vc, ic, fv, fi;)                                                      \
    D(DrawIndexed, 'i', (uint32_t ic, uint32_t inst, uint32_t fi, int32_t vo, uint32_t fin), (ic, inst, fi, vo, fin),  \
      (cb, a->ic, a->inst, a->fi, a->vo, a->fin), uint32_t ic, inst, fi; int32_t vo; uint32_t fin;)                    \
    D(DrawIndirect, 'I', (VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s), (b, o, n, s),                           \
      (cb, a->b, a->o, a->n, a->s), VkBuffer b; VkDeviceSize o; uint32_t n, s;)                                        \
    D(DrawIndexedIndirect, 'J', (VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s), (b, o, n, s),                    \
      (cb, a->b, a->o, a->n, a->s), VkBuffer b; VkDeviceSize o; uint32_t n, s;)                                        \
    D(DrawMeshTasksEXT, 'm', (uint32_t x, uint32_t y, uint32_t z), (x, y, z), (cb, a->x, a->y, a->z),                  \
      uint32_t x, y, z;)                                                                                               \
    D(DrawMeshTasksIndirectEXT, 'M', (VkBuffer b, VkDeviceSize o, uint32_t n, uint32_t s), (b, o, n, s),               \
      (cb, a->b, a->o, a->n, a->s), VkBuffer b; VkDeviceSize o; uint32_t n, s;)                                        \
    D(Dispatch, 'c', (uint32_t x, uint32_t y, uint32_t z), (x, y, z), (cb, a->x, a->y, a->z), uint32_t x, y, z;)       \
    D(DispatchIndirect, 'C', (VkBuffer b, VkDeviceSize o), (b, o), (cb, a->b, a->o), VkBuffer b; VkDeviceSize o;)

// The commands with arrays or nested structures, written by hand in vk_rec_pack.c.
#define VKREC_CUSTOM(C)                                                                                                \
    C(BindDescriptorSets) C(PushDescriptorSetKHR) C(PushConstants) C(BindVertexBuffers2) C(SetViewportWithCount)       \
    C(SetScissorWithCount) C(SetVertexInputEXT) C(SetColorBlendEnableEXT) C(SetColorBlendEquationEXT)                  \
    C(SetColorWriteMaskEXT) C(SetBlendConstants) C(PipelineBarrier2) C(BeginRendering) C(EndRendering)               \
    C(ClearAttachments) C(CopyBuffer) C(CopyBufferToImage) C(CopyImageToBuffer) C(CopyImage) C(BlitImage)              \
    C(ClearColorImage) C(ClearDepthStencilImage)

// ---- the calls the recorder makes: the device's entry points, or a test's
typedef struct VkRecCalls {
#define VKREC_PFN_F(name, ...) PFN_vkCmd##name name;
#define VKREC_PFN_D(name, kind, ...) PFN_vkCmd##name name;
#define VKREC_PFN_C(name) PFN_vkCmd##name name;
    VKREC_FIXED(VKREC_PFN_F)
    VKREC_DRAWS(VKREC_PFN_D)
    VKREC_CUSTOM(VKREC_PFN_C)
#undef VKREC_PFN_F
#undef VKREC_PFN_D
#undef VKREC_PFN_C
    PFN_vkCmdSetCheckpointNV SetCheckpointNV;   // the breadcrumbs
} VkRecCalls;
// The device's entry points (Volk's, as loaded by gpu_vk_open).
void vkrec_calls_device(VkRecCalls *out);

// ---- what a frame's submission is made of: gpu_vk_begin, gpu_vk_cmd and gpu_vk_end for the device, a test's own over
// the timeline alone
typedef struct VkRecSubmit {
    uint64_t (*begin)(int kind);                                          // takes the gate, hands the value
    VkCommandBuffer (*cmd)(void);                                         // the frame's command buffer, begun
    uint64_t (*end)(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits); // ends it, submits, opens the gate
    int checkpoints;                                                      // breadcrumbs on
} VkRecSubmit;
// gpu_vk_begin, gpu_vk_cmd and gpu_vk_end, with the breadcrumbs on when the device has its checkpoints on.
void vkrec_submit_device(VkRecSubmit *out);

// The most frames in flight vkrec_open takes. The tables of breadcrumbs are one more than this, and each keeps the chunks
// its largest frame needed (4,096 records of 32 bytes each) once made, so a deeper window costs memory only as frames grow.
#define VKREC_MAX_FRAMES 8u

typedef struct VkRecOpen {
    size_t ring_bytes;              // GPU_RING_BYTES when 0; a test makes the ring small so that it wraps
    uint32_t frames;                // frames in flight, the setting frames_in_flight when 0 (default 2; 1 to VKREC_MAX_FRAMES)
    const VkRecCalls *calls;        // NULL: vkrec_calls_device
    const VkRecSubmit *submit;      // NULL: vkrec_submit_device
} VkRecOpen;
// Start the recorder thread (at the priority architecture 6.1 gives it). Once; the device must be open when the
// defaults are used. A named fatal when it cannot start.
void vkrec_open(const VkRecOpen *o);
// Wait for every frame handed over to be submitted, stop the thread and free the ring (a frame must not be open). Does
// nothing when the recorder is not open.
void vkrec_close(void);
// 1 between vkrec_open and vkrec_close (the replay opens the recorder with its first frame when no one has).
int vkrec_is_open(void);

// ---- the replay's side
// Wait for a free frame (the frame `frames` before this one is done), then begin a frame of kind GPU_VK_REPLAY or
// GPU_VK_INTERNAL: take the submission gate and the next value, and return it. A guest thread gives its virtual core up
// for the wait (D7), inside the call; any other thread has none to give.
uint64_t vkrec_begin(int kind);
// Queue fn(arg, value) for the completion thread, after the device reached the frame's value (gpu_vk_effect).
void vkrec_effect(GpuEffect fn, void *arg);
// Hand the frame over: the recorder ends its command buffer and submits it after the given waits (copied). Returns the
// frame's value, at once.
uint64_t vkrec_end(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits);
// Wait until the recorder has made every call of every frame handed over and submitted them (not until the device
// finished: gpu_sync_finish does that).
void vkrec_idle(void);

// The stage identities the draws written next carry in their breadcrumbs: the vertex, mesh or compute stage, the
// pixel stage and the task stage; NULL for one the program has none of. Set when a program is bound.
void vkrec_stages(const struct nvn_stage *geometry, const struct nvn_stage *pixel, const struct nvn_stage *task);

// The commands, as packets of the open frame. The state, bind and draw commands with scalar arguments:
#define VKREC_DECL_F(name, params, ...) void vkrec_##name params;
#define VKREC_DECL_D(name, kind, params, ...) void vkrec_##name params;
VKREC_FIXED(VKREC_DECL_F)
VKREC_DRAWS(VKREC_DECL_D)
#undef VKREC_DECL_F
#undef VKREC_DECL_D
// and the commands with arrays and nested structures:
void vkrec_BindDescriptorSets(VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first, uint32_t nsets,
                              const VkDescriptorSet *sets, uint32_t ndynamic, const uint32_t *dynamic);
void vkrec_PushDescriptorSetKHR(VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t nwrites,
                                const VkWriteDescriptorSet *writes);   // dstSet is ignored by the call
void vkrec_PushConstants(VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                         const void *values);
void vkrec_BindVertexBuffers2(uint32_t first, uint32_t n, const VkBuffer *buffers, const VkDeviceSize *offsets,
                              const VkDeviceSize *sizes, const VkDeviceSize *strides);   // sizes, strides: NULL allowed
void vkrec_SetViewportWithCount(uint32_t n, const VkViewport *v);
void vkrec_SetScissorWithCount(uint32_t n, const VkRect2D *r);
void vkrec_SetVertexInputEXT(uint32_t nbindings, const VkVertexInputBindingDescription2EXT *bindings,
                             uint32_t nattributes, const VkVertexInputAttributeDescription2EXT *attributes);
void vkrec_SetColorBlendEnableEXT(uint32_t first, uint32_t n, const VkBool32 *enable);
void vkrec_SetColorBlendEquationEXT(uint32_t first, uint32_t n, const VkColorBlendEquationEXT *equations);
void vkrec_SetColorWriteMaskEXT(uint32_t first, uint32_t n, const VkColorComponentFlags *masks);
void vkrec_SetBlendConstants(const float c[4]);
void vkrec_PipelineBarrier2(const VkDependencyInfo *info);
void vkrec_BeginRendering(const VkRenderingInfo *info);
void vkrec_EndRendering(void);
void vkrec_ClearAttachments(uint32_t nattachments, const VkClearAttachment *attachments, uint32_t nrects,
                            const VkClearRect *rects);
// Transfers: one region each, the images in the general layout (every image the replay touches lives there).
void vkrec_CopyBuffer(VkBuffer src, VkBuffer dst, const VkBufferCopy *region);
void vkrec_CopyBufferToImage(VkBuffer src, VkImage dst, const VkBufferImageCopy *region);
void vkrec_CopyImageToBuffer(VkImage src, VkBuffer dst, const VkBufferImageCopy *region);
void vkrec_CopyImage(VkImage src, VkImage dst, const VkImageCopy *region);
void vkrec_BlitImage(VkImage src, VkImage dst, const VkImageBlit *region, VkFilter filter);
void vkrec_ClearColorImage(VkImage image, const VkClearColorValue *color, const VkImageSubresourceRange *range);
void vkrec_ClearDepthStencilImage(VkImage image, const VkClearDepthStencilValue *value,
                                  const VkImageSubresourceRange *range);

// ---- breadcrumbs
typedef struct VkRecCrumb {
    uint8_t kind;                          // 'd' draw, 'i' indexed, 'I' and 'J' indirect, 'm' and 'M' mesh, 'c' and 'C' compute
    const struct nvn_stage *id[3];         // geometry or compute stage, pixel stage, task stage (vkrec_stages)
} VkRecCrumb;
// The breadcrumb a per-draw checkpoint marker names: 1 with *out set and *value, *n the marker's submission and draw
// number, or 0 when the marker is not a draw's or its frame's table has been reused since (the frame is done).
int vkrec_crumb(uint64_t marker, VkRecCrumb *out, uint64_t *value, uint32_t *n);
// The marker in words, for the device-loss report: "draw 17 of submission 42 (draw): vertex stage <key>, pixel stage
// <key>"; 0 when vkrec_crumb finds nothing, with the draw number and value alone in out.
int vkrec_marker_text(uint64_t marker, char *out, size_t cap);

// ---- numbers
typedef struct VkRecFrame {
    uint64_t value;                  // the frame's timeline value
    uint32_t draws;                  // breadcrumbs given (0 with the breadcrumbs off)
    uint32_t packets;
    uint64_t bytes;                  // of packets, headers included
    uint64_t gate_wait_ns;           // vkrec_begin waiting for the submission gate: the frame before not yet submitted
    uint64_t slot_wait_ns;           // then waiting for the frame `frames` before to be done (and for a core again)
    uint64_t ring_wait_ns;           // the replaying thread waiting for room in the ring
    uint64_t replay_ns;              // vkrec_begin returned to vkrec_end returned, less the ring wait: the replaying thread's
    uint64_t replay_cpu_ns;          // thread time of the same stretch (the replaying thread's own, set by vkrec_end)
    uint64_t record_ns;              // the recorder: its frame-begin packet read to its submit made (wall time)
    uint64_t record_cpu_ns;          // the recorder's thread time over the same stretch
} VkRecFrame;
// Rows of frames handed over and submitted, oldest first, the last VKREC_LOG of them; *n is how many. Valid until the
// next vkrec_open. Call after vkrec_idle.
#define VKREC_LOG 4096u
const VkRecFrame *vkrec_frames(uint32_t *n);
// The replay counters' feeds (OOT2_REPLAY_COUNTERS, vk_replay_time.c), called only with the counters on: a frame's values
// as the replaying thread leaves them at vkrec_end, and the recorder's once it has submitted the frame.
void vkrec_count_replay(const VkRecFrame *row);
void vkrec_count_record(uint64_t ns, uint64_t cpu_ns);
void vkrec_ring_stats(GpuRingStats *out);

// ---- inside the recorder (vk_rec.c, vk_rec_pack.c)
enum {
    VKR_PAD = GPU_RING_OP_PAD,
    VKR_FRAME_BEGIN,
    VKR_FRAME_END,
#define VKREC_OP_F(name, ...) VKR_##name,
#define VKREC_OP_D(name, kind, ...) VKR_##name,
#define VKREC_OP_C(name) VKR_##name,
    VKREC_FIXED(VKREC_OP_F)
    VKREC_DRAWS(VKREC_OP_D)
    VKREC_CUSTOM(VKREC_OP_C)
#undef VKREC_OP_F
#undef VKREC_OP_D
#undef VKREC_OP_C
    VKR_COUNT
};
#define VKREC_UNPAREN(...) __VA_ARGS__
// Room for a packet of the open frame (a named fatal outside one); vkrec_publish after it is filled.
void *vkrec_put(uint32_t op, size_t bytes);
void vkrec_publish(void);
// The number and record of the next draw of the open frame, 0 with the breadcrumbs off.
uint32_t vkrec_crumb_next(uint8_t kind);
// What the recorder makes a packet into: calls into cb, and the checkpoints of the frame `value` when on.
typedef struct VkRecExec {
    const VkRecCalls *calls;
    VkCommandBuffer cb;
    uint64_t value;
    int checkpoints;
} VkRecExec;
void vkrec_exec(const VkRecExec *x, uint32_t op, const void *payload);   // a named fatal for an unknown op

// Tests only: while non-zero, the breadcrumb of a draw is numbered one past its record (the counter-run of the numbering:
// a marker that names the wrong draw).
extern int vkrec_crumb_skew;
// Tests only: while non-zero, vkrec_begin does not wait for the frame `frames` before it (the counter-run of the limit on
// frames in flight: one more frame let through).
extern int vkrec_no_limit;
