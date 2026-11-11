// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the replay's walk (vk_replay.c) gives its transfers (vk_replay_xfer.c): the frame being replayed, the named stop,
// the frame's scratch memory, the pass and barrier rules, and the declaration of a GPU write. Private to runtime/gpu.
#pragma once
#include <stdint.h>

#include "oot2/replay_count.h"
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

// The replay's trace (OOT2_REPLAY_TRACE, vk_replay_trace.c): each call is made only when vkrt_on is set. Read and count
// only; nothing here reaches the device or changes a decision of the replay. A submission, its passes, then a row for
// each draw, mesh draw and dispatch: vkrt_kind (an op says what kind of graphics draw follows), vkrt_begin or
// vkrt_begin_kind (cause: 0 the draw is made, 1 off by the game's RenderEnable, 2 off by a conditional rendering, which
// closes the row), vkrt_skip (the pipeline was not ready), vkrt_stages, vkrt_unbound ('c' bank c[1] absent, 's' storage
// buffer unbound, 'u' uniform range unbound: each bound as zeros), vkrt_tex_mark before and vkrt_tex after a texture
// slot's bindless slot is asked for, vkrt_end with what the draw drew.
extern int vkrt_on;
typedef struct VkrtEnd {
    uint32_t count;                  // vertices, indices or mesh tasks
    int32_t first, inst_rec;         // first vertex or index, instances as recorded (0: one) ...
    uint32_t inst_run;               // ... and as replayed
    int32_t prim;                    // -1: unknown
    int have_ind;                    // an indirect draw's record was read by the CPU: ind_count, ind_inst
    uint32_t ind_count, ind_inst;
    uint32_t groups[3];              // a direct dispatch's
} VkrtEnd;
struct nvn_stage;
void vkrt_submit_begin(uint64_t value);
void vkrt_submit_end(void);
void vkrt_pass_begin(uint32_t w, uint32_t h, uint32_t ncolor, const VkFormat *color, VkFormat depth, VkFormat stencil);
void vkrt_pass_end(uint32_t draws);
void vkrt_kind(char kind);
void vkrt_begin(int cause);
void vkrt_begin_kind(char kind, int cause);
void vkrt_clear_off(void);
void vkrt_skip(void);
void vkrt_stages(const char *k0, const char *k1, const char *k2, const struct nvn_stage *const *raw, int nraw);
void vkrt_unbound(int what);
void vkrt_tex_mark(void);
void vkrt_tex(int st, uint32_t c2_slot, uint32_t id, uint32_t sampler, uint32_t slot, const VkReplayOps *ops);
void vkrt_end(const VkrtEnd *e);
// The occlusion query q begun in the open pass; a frame's queries taken for the completion thread (NULL when there are
// none); and their results there (frees what was taken).
void vkrt_query(uint32_t q);
void *vkrt_queries_take(uint32_t nq);
void vkrt_queries_done(void *queries, uint32_t nq, const uint64_t *samples);
// A present of the front end (nvn_census_present, through rc_set_present_hook): the summary line, and the M line for the
// raw-access misses the device reported (vkrt_raw_misses is the printer alone, for a test to feed).
void vkrt_present(uint64_t seq);
struct GpuVkRawStats;
void vkrt_raw_misses(const struct GpuVkRawStats *r, uint64_t seq);

// ---- the trace's further columns and the readback statistics (vk_replay_trace.c). Every call is made only when vkrt_on is
// set (the readback ones when vkrt_rb_on is too); read and count only, except that a wanted readback records a copy of the
// colour target into the frame's scratch memory after the pass, which the game never reads and no draw depends on.
extern int vkrt_rb_on;
// One target of a pass (slot 0 to 7 the colour attachments, 8 the depth-stencil one): the storage the texture stands on,
// its level, first layer and layers, and the extent and Vulkan format the pass uses.
void vkrt_pass_target(int slot, uint64_t storage, uint32_t level, uint32_t layer, uint32_t layers, uint32_t w, uint32_t h,
                      VkFormat format);
// The state the draw being prepared will draw with (the blend, mask and depth state of the replay's bound state).
typedef struct VkrtState {
    uint32_t ncolor;
    uint32_t blend_en;                 // bit i: target i blends
    uint32_t chmask;                   // 4 bits a target, r=1 g=2 b=4 a=8, bit set = written
    struct {
        uint8_t set, src_rgb, dst_rgb, src_a, dst_a, eq_rgb, eq_a;
    } blend[8];                        // the blend states the game set, by target (set 0: never set)
    int has_ds, dt, dw, df, st;
    float point_size, line_width;      // point_size < 0: the game never set one
} VkrtState;
void vkrt_state(const VkrtState *s);
// What the draw or dispatch being prepared binds for writing: 'b' a storage buffer (a its address, size its range), 'i' a
// storage image (a its texture pool ID, b and size the storage range its stores land in; 0 when the image holds them itself).
void vkrt_write(char what, uint64_t a, uint64_t b, uint64_t size);
// The readback of a pass's colour target (OOT2_REPLAY_TRACE rb=...): asked after the pass has ended, per colour attachment.
typedef struct VkrtRbWant {
    uint32_t att, width, height, layers;
    VkFormat format;
    uint32_t z;                        // out: the slice to read (a layer of an array, a depth slice of a volume)
} VkrtRbWant;
int vkrt_rb_wants(VkrtRbWant *w);
// A wanted readback that cannot be made (the reason), and one that was recorded: `bytes` of texels in host memory at `data`
// the completion thread reads when the frame is done (before the scratch memory is taken back).
void vkrt_rb_skipped(const VkrtRbWant *w, const char *why);
void vkrt_rb_queued(const VkrtRbWant *w, uint32_t level, uint32_t layer, const void *data, uint64_t bytes);
// The statistics of `count` texels of format f at data: per channel the minimum, maximum and mean of the finite values, the
// count of non-finite ones and of those above `threshold`. 0 for a format with no decoder here.
typedef struct VkrtChan {
    double min, max, sum;
    uint64_t finite, nonfinite, above;
} VkrtChan;
typedef struct VkrtScan {
    int nch;
    VkrtChan ch[4];
} VkrtScan;
uint32_t vkrt_texel_bytes(VkFormat f);   // bytes of a texel the scan decodes, 0 for none
int vkrt_scan(const void *data, VkFormat f, uint64_t count, double threshold, VkrtScan *out);

// ---- the replay stage timer (vk_replay_count.c; OOT2_REPLAY_COUNTERS, group "stages"). A class is timed from its in-call to
// its out-call by the replaying thread only, inclusive of the classes it contains. The call is counted exactly; the clock is
// read for one call in `every` (the classes that run thousands of times a present), and the line scales the sampled time to
// the calls. Each site tests rc_on first, so the timer off costs one load.
enum {
    VKST_WALK,         // the walk of a submission's recordings
    VKST_DRAW,         // a graphics or mesh draw, from its record
    VKST_DISPATCH,
    VKST_CLEAR,
    VKST_COPY,         // a copy, fill, update or texture clear (the transfers)
    VKST_QUERY,        // a report, event, wait or condition record
    VKST_BARRIER,
    VKST_PASS_BEGIN,
    VKST_PASS_END,
    VKST_PIPELINE,     // the device's pipeline lookup of a draw or dispatch
    VKST_STAGE_RES,    // the descriptor build of a stage (stage_resources), the resolves and snapshots below included
    VKST_VERTEX,       // the vertex input of a draw
    VKST_DYN,          // the dynamic state of a draw
    VKST_RESOLVE,      // a backing resolve (sampled)
    VKST_SNAPSHOT,     // a uniform snapshot, its search included (sampled)
    VKST_UNIFORM,      // a uniform range bound (sampled)
    VKST_LEDGER,       // the ledger's tracker, once a record (sampled)
    VKST_N
};
uint64_t vkst_in(int cls);
void vkst_out(int cls, uint64_t t0);
#define ST_TIME(cls, ...)                                                                                              \
    do {                                                                                                               \
        if (__builtin_expect(rc_on, 0)) {                                                                              \
            uint64_t st_t0_ = vkst_in(cls);                                                                            \
            __VA_ARGS__;                                                                                               \
            vkst_out(cls, st_t0_);                                                                                     \
        } else {                                                                                                       \
            __VA_ARGS__;                                                                                               \
        }                                                                                                              \
    } while (0)
// The class of a record op (VKST_DRAW ... VKST_QUERY, VKST_BARRIER), or -1 for a state record that is not timed.
int vkst_class_of(uint32_t op);
// A submission begun, a recording walked (its address, its records and the draws and dispatches among them) and a draw or
// dispatch record by kind: the counters that say why more draws are replayed than the game's calls count.
void vkst_handle(uint64_t handle, uint32_t records, uint32_t ops);
void vkst_sub(void);
void vkst_op(uint32_t op, int32_t instances);
// Zero the timer and its counters (a test).
void vkst_reset(void);
