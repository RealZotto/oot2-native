// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the replay's walk (vk_replay.c) gives its transfers (vk_replay_xfer.c): the frame being replayed, the named stop,
// the frame's scratch memory, the pass and barrier rules, and the declaration of a GPU write. Private to runtime/gpu.
#pragma once
#include <stdint.h>

#include "oot2/vk_replay.h"

typedef unsigned long long ull;

// The ops and the value of the frame being replayed.
const VkReplayOps *vkr_frame_ops(void);
uint64_t vkr_frame_value(void);

// A named fatal that says which submission and draw it stopped after.
__attribute__((noreturn, format(printf, 1, 2))) void vkr_stop(const char *fmt, ...);

// `bytes` of the frame's scratch memory (a named fatal when there is none): the pointer, and what binds it.
void *vkr_scratch(uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset);

// A transfer's opening: the render pass ends and a full memory barrier separates it from what came before (D18).
void vkr_xfer_begin(void);
// A transfer's close: the next op that reads memory without a barrier of its own (a dispatch) opens with one.
void vkr_xfer_end(void);

// The frame's GPU write over [addr, addr + size): the uniform ranges copied at submit that it touches are dropped (D17a), a
// write the frame declared already is not declared again, and a new one is declared to the backing map (backing_gpu_wrote)
// and listed (vkr_writes). `image` 1 for a storage image's storage; `draw` the number of the draw or dispatch that
// writes it, 0 for a copy, fill or update.
void vkr_wrote(uint64_t addr, uint64_t size, int image, uint32_t draw);
// 1 when the frame has declared a GPU write over any byte of [addr, addr + size).
int vkr_written(uint64_t addr, uint64_t size);

// A transfer of `bytes` bytes, for the statistics.
void vkr_count_transfer(uint64_t bytes);

// One op of the transfers, or 0 when it is not one of them. `r` is the record.
int vkr_xfer_op(const op_hdr *r);
// Forget the copy strides (vkr_reset).
void vkr_xfer_reset(void);

// The reports, events and waits (vk_replay_report.c): one op of them, or 0 when it is not one of them; the condition of
// SetRenderEnableConditional (1: render); a draw about to be recorded in the open pass (it begins a samples-passed query
// when counting); the pass about to end (the open query ends); the frame's begin, and its end after the frame's own end
// op (the list goes to the completion thread); and the forgetting of the channel's counting (vkr_reset).
int vkr_report_op(const op_hdr *r);
int vkr_condition(const p_render_cond *p);
void vkr_report_draw(void);
void vkr_report_pass_end(void);
void vkr_report_frame_begin(void);
void vkr_report_frame_end(void);
void vkr_report_reset(void);

// The workload counters of the replay (vk_replay_count.c, oot2/replay_count.h): each is called only with the counters on.
// A submission's start and end (work: its draws and dispatches); a graphics draw about to be issued with its stages (task
// is NULL for a vertex draw), the formats of its pass and its pipeline; the vertex input of a draw (one block, its first
// state_n bytes the attribute and stream state, the first layout_n those and the stage's binds, all full_n those and the
// buffers' addresses); and whether the dynamic state of a draw equals the
// state the draw before it left.
struct VkpStage;
struct VkpFormats;
void vkrc_submit_begin(void);
void vkrc_submit_end(uint32_t work);
void vkrc_draw(int mesh, const struct VkpStage *task, const struct VkpStage *vs, const struct VkpStage *ps,
               const struct VkpFormats *formats, VkPipeline pipe);
void vkrc_vin(const void *state, size_t state_n, size_t layout_n, size_t full_n);
void vkrc_dyn(int same);
