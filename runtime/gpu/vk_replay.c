// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay of the Vulkan back end (oot2/vk_replay.h): the walk of a submission's recordings on the submitting thread,
// the bound state of the channel, and what each draw, dispatch, clear and barrier turns into as the recorder's packets.
//
// The pass rules (lazy passes, a full barrier before each, every image in the general layout), the Y flip through a
// negative viewport height, the driver bank c[0] and the uniform snapshot's rules follow DKB Recomp at 9d8fa02
// (GPL-3.0-or-later): runtime/vk_exec.c (passes, the scissor clip of clears), runtime/vk_exec_draw.c:2165-2199 (the flip)
// and :1012-1032 (c[0]), runtime/vk_pipeline.c:3639-3644 (the front face under the flip: unchanged for the upper-left
// origin, mirrored for the lower-left one as LAS Recomp does), runtime/vk_usnap.c:1-21 (the snapshot: imported
// uniform ranges copied at submit, a GPU write over a copy drops it). The code is ours: the replay writes packets and binds
// through the uni layout's push descriptors and push constant texture slots. The transfers are vk_replay_xfer.c's; the
// reports, events, waits and conditions vk_replay_report.c's.
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/capture.h"
#include "oot2/gfx.h"
#include "oot2/gpu_vk.h"
#include "oot2/ledger.h"
#include "oot2/mesh_abi.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/shader_desc.h"
#include "oot2/vk_format.h"
#include "oot2/vk_gpu_time.h"
#include "oot2/vk_movie_blit.h"
#include "oot2/vk_nvn.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"
#include "vk_replay_int.h"

// NVN shader stages (NVNshaderStage; task and mesh as the ledger numbers them)
enum { ST_VS = 0, ST_FS = 1, ST_CS = 5, ST_TS = 6, ST_MS = 7, ST_N = 8 };

#define UBO_SLOTS 21           // BindUniformBuffer index r is constant bank c[3 + r], up to c[23]
#define SSBO_SLOTS 16          // BindStorageBuffer index r is row r of the stage's storage table in c[0]
#define TEX_SLOTS 41           // BindTexture indexes (c[2] 0x20 + 8 r up to the separate textures)
#define IMG_SLOTS 8
#define ROOT_TABLES 8          // Push{Graphics,Compute}Constants: eight root tables of 64 words, bank c[24]
#define ROOT_WORDS 64
#define ROOT_BANK_BYTES (ROOT_TABLES * ROOT_WORDS * 4)
#define UNIFORM_MAX 65536u     // the largest uniform range a binding gives (maxUniformBufferRange)
#define STORAGE_ALIGN 16u      // the first draw study: a storage buffer binds from its address rounded down to 16 bytes
#define UNIFORM_ALIGN 256u
#define SNAPS_MAX 4096
#define WRITES_MAX 8192
#define FRAMES_KEPT 4

int vkr_no_snapshot;
int vkr_no_flip;
int vkr_no_drop;
int vkr_no_declare;
int vkr_no_features;
int vkr_no_first_task;

// ---------- the op list
const char *vkr_op_ticket(uint32_t op) {
    switch (op) {
    case OP_BEGIN: case OP_END: case OP_SET_RT: case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_VIEWPORTS:
    case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR:
    case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP:
    case OP_BIND_IMG: case OP_STENCIL: case OP_POLY_OFFSET: case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
    case OP_DISPATCH: case OP_DISPATCH_IND: case OP_BARRIER: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
    case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RENDER_ENABLE: case OP_RASTER_DISCARD: case OP_PRIM_RESTART:
    case OP_PUSH_CONST: case OP_WAIT_SYNC: case OP_SAMPLE_MASK: case OP_DEPTH_BOUNDS: case OP_CONSERVATIVE:
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND: case OP_GEOMETRY_MODE:
    case OP_SHADING_RATE_IMAGE: case OP_SHADING_RATE_PALETTE: case OP_SHADING_RATE_MODE: case OP_SCRATCH: case OP_HINT:
    case OP_COPY_B2B: case OP_UPDATE_UB: case OP_COPY_B2T: case OP_COPY_STRIDE: case OP_COPY_T2B: case OP_COPY_T2T:
    case OP_CLEAR_TEX: case OP_CLEAR_BUF: case OP_COPY_IMAGE_STRIDE:
    case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_COND: case OP_WAIT_EVENT:
        return VKR_REPLAYED;
    case OP_BIND_ACCEL: return "acceleration structures (no ticket: the replay has no ray tracing)";
    case OP_JUMP: case OP_CALL: return "a called recording (the front end inlines every one)";
    default: return "unknown";
    }
}

// ---------- the bound state of the channel, across records and submissions
static struct {
    LedgerTrack track;                        // the program's stages and the attachment formats (oot2/ledger.h)
    p_set_rt rt;                              // the render targets as the last SET_RT gave them
    float vp[16][4], dr[16][2];
    int32_t sc[16][4];
    int nvp, nsc, has_sc;
    blend_st blend[8];
    color_st color;
    chmask_st chmask;
    poly_st poly;
    ds_st ds;
    int has_poly, has_ds, has_blend[8];
    float line_width;
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    float bias[3];                            // factor, units, clamp
    float blend_color[4];
    int depth_clamp, raster_discard, restart_en, render, bounds_en, conservative;
    uint32_t restart_index, sample_mask;
    float bounds[2];
    uint32_t vattr[NVN_VERTEX_BINDINGS];
    vs_st vstream[NVN_VERTEX_BINDINGS];
    NVNbufferRange vbuf[NVN_VERTEX_BINDINGS];
    VknBuf ubo[ST_N][UBO_SLOTS], ssbo[ST_N][SSBO_SLOTS];
    uint64_t tex[ST_N][TEX_SLOTS], img[ST_N][IMG_SLOTS];
    uint32_t root[2][ROOT_TABLES][ROOT_WORDS];
    dpool_ref tex_pool, samp_pool;
    int has_tex_pool, has_samp_pool;
} S;

// ---------- the frame being replayed
typedef struct { uint64_t lo, hi; } Rng;
typedef struct { VkBuffer buf; VkDeviceSize off; } Bound;

typedef struct {
    uint8_t bytes[VKN_BANK_BYTES];
    Bound at;
    int valid;
} BankCache;

typedef struct {
    // the dynamic state as last written into the frame's command buffer
    VkViewport vp[16];
    VkRect2D sc[16];
    uint32_t nvp;
    VkCullModeFlags cull;
    VkFrontFace front;
    VkPolygonMode mode;
    VkBool32 bias_en, discard, clamp, clip11, dt, dw, bt, st, restart, restart_valid, logic;
    float bias[3], line_width, bounds[2], blend_color[4];
    VkCompareOp df;
    VkStencilOp sop[2][3];
    VkCompareOp sfn[2];
    uint32_t smask[2][3];
    VkPrimitiveTopology topo;
    uint32_t ntargets;
    VkBool32 blend_en[8];
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
} Dyn;

static struct {
    const VkReplayOps *ops;
    uint64_t value;
    uint32_t draws;
    int in_pass, pass_any;
    uint32_t pass_w, pass_h, pass_layers;
    VkrTarget color[8], depth;
    uint32_t ncolor;
    int has_depth;
    VkpFormats formats;
    VkDescriptorSet set0;
    int set0_bound[2];
    VkPipeline pipe[2];
    Dyn dyn;
    int dyn_valid;
    Bound zero;
    int has_zero;
    BankCache c0[ST_N], c2[ST_N], roots[2];
    Rng snaps[SNAPS_MAX];
    Bound snap_at[SNAPS_MAX];
    uint32_t nsnaps;
    Rng wrote[WRITES_MAX];
    uint32_t nwrote;
    int wslot;                                // the frame's entry of W
    int after_xfer;                           // a transfer ran since the last full barrier
    uint32_t pass_draws;                      // the open pass's draws and the stages of its first: the gpu timing's labels
    const char *pass_vs, *pass_fs;
    int noting;                               // a dispatch is gathering the buffers its gpu timing mark names
    const op_hdr *cur;                        // the record being replayed, NULL between records
    int movie_open;                           // the movie chain's stand-in (settings.h test_movie_blit): a chain is open
    uint32_t movie_src;                       // and the texture pool ID of the picture its first program stored
    uint32_t movie_skips;                     // the dispatches skipped since it opened: how many, and the keys of the program
    char movie_keep[26], movie_first[26], movie_last[26];   // that opened it and of the first and the last one skipped
} F;
_Static_assert(sizeof(((nvn_stage *)0)->key) == sizeof F.movie_keep, "a stage key fits the movie chain's notes");

// What the frames in flight wrote (vkr_writes), a ring of FRAMES_KEPT by replayed frame (timeline values are shared with
// the back end's own submissions, so a value modulo the ring would let two frames in flight share an entry).
static struct {
    uint64_t value;
    VkrWrite *w;
    uint32_t n, cap;
} W[FRAMES_KEPT];

static VkrStats g_stats;
static const VkReplayOps *g_ops;

void vkr_set_ops(const VkReplayOps *ops) { g_ops = ops; }

static const VkReplayOps *ops_now(void) { return g_ops ? g_ops : vkr_device_ops(); }

void vkr_stats(VkrStats *out) {
    *out = g_stats;
    // Every draw and dispatch record is replayed, skipped by name or not made because rendering was off: what is left is
    // GPU work lost (the Hookshot survey: Hookshot's dark characters and bare foliage were work lost at start), 0 by contract.
    uint64_t accounted = g_stats.draws + g_stats.dispatches + g_stats.render_off + g_stats.skipped + g_stats.collected +
                         g_stats.movie_skipped + g_stats.movie_blits;
    out->lost = g_stats.draw_ops > accounted ? g_stats.draw_ops - accounted : accounted - g_stats.draw_ops;
}

const VkrWrite *vkr_writes(uint64_t value, uint32_t *n) {
    for (int i = 0; i < FRAMES_KEPT; i++)
        if (W[i].value == value && value) return *n = W[i].n, W[i].w;
    *n = 0;
    return NULL;
}

void vkr_reset(void) {
    memset(&S, 0, sizeof S);
    vkr_xfer_reset();
    vkr_report_reset();
    ledger_track_reset(&S.track);
    S.render = 1, S.line_width = 1.0f, S.sample_mask = ~0u;
    S.chmask.mask = 0xFFFFFFFFu;
    for (int i = 0; i < 16; i++) S.dr[i][0] = 0.0f, S.dr[i][1] = 1.0f;
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xFF;
    S.bounds[1] = 1.0f;
}

__attribute__((constructor)) static void replay_init(void) { vkr_reset(); }

// ---------- stops
#define stop(...) vkr_stop(__VA_ARGS__)
// The ledger tracker sees a record after it was replayed, so a replay that stops on a draw or a dispatch would leave the
// answers it stopped on out of the ledger, and the next prepare could never translate the variant that answers it. The
// stop gives the tracker the record first (it only reads the bound state to make the draw's rows; the stop's ledger
// flush writes them).
static int makes_rows(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: case OP_DRAW_MESH: case OP_DRAW_MESH_IND: case OP_DISPATCH:
    case OP_DISPATCH_IND: return 1;
    default: return 0;
    }
}

void vkr_stop(const char *fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (F.cur && makes_rows(F.cur->op)) {
        const op_hdr *r = F.cur;
        F.cur = NULL;
        ledger_track_op(&S.track, r);
    }
    rt_fatal("gfx: vulkan: replay of submission %llu, after %u draws and dispatches: %s", (ull)F.value, F.draws, line);
}

// ---------- collection (settings.h collect_variants, a development switch that never enters a proof run)
// A stop on a named answer that the ledger can fix (a vertex attribute of another scalar type than the module's input)
// would show one stop a run, and a run of the route is long while the ledger step is short. Under the setting such a
// stop writes the draw's ledger row first, as vkr_stop does, counts the draw, logs the pair of stage and answer once,
// and abandons that draw only: the replay goes on with the next record. The draw is not drawn, so the picture is wrong
// and the run says so in its log, in the census and in the capture (the skipped-draw counter and a header mark, so
// the proof judge refuses it). Without the setting, or outside a draw, it is vkr_stop with the same message, so every
// other stop and the default run are as they were. Only a draw reaches it: the vertex input is a graphics state.
static jmp_buf *g_skip;   // the draw being replayed under collection, else NULL (the replay is one thread)

static int collecting(void) { return rt_setting(RT_SET_collect_variants) != 0; }

__attribute__((noreturn, format(printf, 2, 3))) static void stop_answer(const char *stage, const char *fmt, ...) {
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    if (!g_skip || !F.cur || !makes_rows(F.cur->op)) vkr_stop("%s", line);
    const op_hdr *r = F.cur;
    F.cur = NULL;
    ledger_track_op(&S.track, r);   // the draw's rows, which the record's end would have made had it been replayed
    int first = 0;
    uint64_t n = ledger_collect_note(stage, line, &first);
    g_stats.collected++;
    capture_count(CAPTURE_SKIPPED_DRAWS, 1);   // the proof judge fails on a skipped draw: a collection run cannot pass it
    if (n == 1)
        rt_log("gfx: vulkan: COLLECTING VARIANTS (collect_variants is on): a draw that stops on an answer the ledger can "
               "fix is skipped, not drawn, after its ledger row is written. This run is NOT a proof run and its picture "
               "is wrong");
    if (first)
        rt_log("gfx: vulkan: draw skipped for collection, stage %s: %s (logged once for this stage and answer; the "
               "census counts the draws)", stage, line);
    longjmp(*g_skip, 1);
}

__attribute__((noreturn)) static void stop_op(uint32_t op) {
    stop("%s is not replayed yet: %s", nvn_op_name(op), vkr_op_ticket(op));
}

const VkReplayOps *vkr_frame_ops(void) { return F.ops; }
uint64_t vkr_frame_value(void) { return F.value; }

// ---------- the frame's scratch memory
static Bound scratch(uint32_t bytes, uint32_t align, void **out) {
    Bound b;
    void *p = F.ops->scratch(F.value, bytes, align, &b.buf, &b.off);
    if (!p) stop("no scratch memory for %u bytes", bytes);
    g_stats.scratch_bytes += bytes;
    if (out) *out = p;
    return b;
}

static VkDeviceAddress scratch_address(Bound b) {
    if (!F.ops->buffer_address) stop("a mesh draw needs the device address of replay scratch");
    VkDeviceAddress address = F.ops->buffer_address(b.buf);
    if (!address || address > UINT64_MAX - b.off) stop("a mesh draw's scratch buffer has no usable device address");
    return address + b.off;
}

void *vkr_scratch(uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset) {
    void *p;
    Bound b = scratch(bytes, align, &p);
    *buffer = b.buf, *offset = b.off;
    return p;
}

// A zeroed range the size of the largest uniform binding: what an unbound slot reads.
static Bound zero_block(void) {
    if (!F.has_zero) {
        void *p;
        F.zero = scratch(UNIFORM_MAX, UNIFORM_ALIGN, &p);
        memset(p, 0, UNIFORM_MAX);
        F.has_zero = 1;
    }
    return F.zero;
}

// A driver bank: the bytes of the last bank written for the same use, or a new copy.
static Bound bank(BankCache *c, const uint8_t *bytes, uint32_t size) {
    if (c->valid && !memcmp(c->bytes, bytes, size)) return c->at;
    void *p;
    c->at = scratch(size, UNIFORM_ALIGN, &p);
    memcpy(p, bytes, size);
    memcpy(c->bytes, bytes, size);
    c->valid = 1;
    return c->at;
}

// ---------- the snapshot (D17a) and the frame's GPU writes
static int overlaps(Rng a, Rng b) { return a.lo < b.hi && b.lo < a.hi; }

// 1 when the frame has declared a GPU write over any byte of [addr, addr + size).
int vkr_written(uint64_t addr, uint64_t size) {
    Rng r = {addr, addr + size};
    for (uint32_t i = 0; i < F.nwrote; i++)
        if (overlaps(r, F.wrote[i])) return 1;
    return 0;
}

// The frame's GPU write of [addr, addr + size): the copies of uniform ranges it touches are dropped (the drop rule: a read
// of the range after this write binds the live memory, a read before it keeps its copy), and the write is declared to the
// backing map once, unless a write the frame declared already holds all of it (the write-back reads the bytes at the end
// of the submission, so the later write is in it).
void vkr_wrote(uint64_t addr, uint64_t size, int image, uint32_t draw) {
    Rng r = {addr, addr + size};
    if (!vkr_no_drop)
        for (uint32_t i = 0; i < F.nsnaps;) {
            if (!overlaps(r, F.snaps[i])) {
                i++;
                continue;
            }
            F.nsnaps--;
            F.snaps[i] = F.snaps[F.nsnaps], F.snap_at[i] = F.snap_at[F.nsnaps];
            g_stats.snapshot_drops++;
        }
    int k = F.wslot;
    if (W[k].n == W[k].cap) {
        uint32_t cap = W[k].cap ? 2 * W[k].cap : 256;
        VkrWrite *w = realloc(W[k].w, sizeof *w * cap);
        if (!w) rt_fatal("gfx: vulkan: out of memory for %u written ranges", cap);
        W[k].w = w, W[k].cap = cap;
    }
    W[k].w[W[k].n++] = (VkrWrite){addr, size, draw, (uint8_t)image};
    for (uint32_t i = 0; i < F.nwrote; i++)
        if (F.wrote[i].lo <= r.lo && r.hi <= F.wrote[i].hi) return;
    if (F.nwrote == WRITES_MAX) stop("more than %u GPU writes in one submission", WRITES_MAX);
    F.wrote[F.nwrote++] = r;
    if (!vkr_no_declare) F.ops->gpu_wrote(addr, size, F.value);
    g_stats.storage_writes++;
}

// What the gpu timing (oot2/vk_gpu_time.h) says about a buffer a dispatch binds: the letter of its backing.
static char backing_letter(int kind) {
    return kind == BACKING_EXTENT ? 'E' : kind == BACKING_IMPORT ? 'I' : kind == BACKING_MIRROR ? 'M' : '?';
}
static void note(char kind, uint64_t addr, uint64_t bytes, int write) {
    if (vkgt_enabled && F.noting) vkgt_note(kind, addr, bytes, write);
}

static Bound snapshot(uint64_t addr, uint64_t size) {
    Rng r = {addr, addr + size};
    for (uint32_t i = 0; i < F.nsnaps; i++)
        if (F.snaps[i].lo == r.lo && F.snaps[i].hi == r.hi) return F.snap_at[i];
    void *p;
    Bound b = scratch((uint32_t)size, UNIFORM_ALIGN, &p);
    memcpy(p, (const void *)(uintptr_t)addr, size);
    if (F.nsnaps < SNAPS_MAX) F.snaps[F.nsnaps] = r, F.snap_at[F.nsnaps] = b, F.nsnaps++;
    g_stats.snapshots++, g_stats.snapshot_bytes += size;
    return b;
}

// A uniform range: VRAM and mirrors bind as they are (a mirror is a copy taken at submit); host memory the GPU reads
// live is copied, unless the frame wrote it already (the drop rule: the live memory holds the GPU's write, in order).
static Bound uniform(VknBuf b) {
    if (!b.addr || !b.size) return zero_block();
    uint64_t size = b.size < UNIFORM_MAX ? b.size : UNIFORM_MAX;
    BackingRef ref;
    F.ops->resolve(b.addr, size, F.value, &ref);
    if (ref.kind == BACKING_IMPORT && !vkr_no_snapshot && (vkr_no_drop || !vkr_written(b.addr, size))) {
        note('U', b.addr, size, 0);
        return snapshot(b.addr, size);
    }
    note(backing_letter(ref.kind), b.addr, size, 0);
    g_stats.live_uniforms++;
    return (Bound){(VkBuffer)(uintptr_t)ref.buffer, ref.offset};
}

// ---------- passes (D18)
static void full_barrier(void) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkrec_PipelineBarrier2(&di);
    g_stats.barriers++;
    F.after_xfer = 0;
}

static void pass_end(void) {
    if (!F.in_pass) return;
    vkr_report_pass_end();   // a samples-passed query begun in the pass ends in it
    vkrec_EndRendering();
    F.in_pass = 0;
    if (vkgt_enabled) {
        char label[56];
        snprintf(label, sizeof label, "%s+%s", F.pass_vs ? F.pass_vs : "none", F.pass_fs ? F.pass_fs : "none");
        vkgt_mark(VKGT_PASS, label, F.pass_draws, F.pass_w, F.pass_h);
    }
}

// A transfer is one command outside any pass, after a full barrier; what runs next without a barrier of its own (a
// dispatch) opens with one.
void vkr_xfer_begin(void) {
    pass_end();
    full_barrier();
}
void vkr_xfer_end(void) { F.after_xfer = 1; }
void vkr_count_transfer(uint64_t bytes) { g_stats.transfers++, g_stats.transfer_bytes += bytes; }

static int target_of(const nvn_tex_rec *t, VkrTarget *out) {
    if (t->tex.d.samples > 1) stop("a multisampled render target (%d samples) is not replayed yet", t->tex.d.samples);
    if (!F.ops->target(t, 0, F.value, out))
        stop("no image stands on the render target's storage (NVN format %#x, %ux%u): see oot2/vk_image.h",
             (unsigned)(t->has_view && t->view.set_format ? t->view.format : (unsigned)t->tex.d.format),
             t->tex.d.width, t->tex.d.height);
    return 1;
}

// The size of a pass with no attachment: the far corner of viewport 0, else of the scissor (DKB vk_exec.c:362-373).
static void area_without_targets(uint32_t *w, uint32_t *h) {
    double x1 = S.nvp ? (double)S.vp[0][0] + S.vp[0][2] : 0, y1 = S.nvp ? (double)S.vp[0][1] + S.vp[0][3] : 0;
    if ((x1 < 1 || y1 < 1) && S.has_sc) x1 = (double)S.sc[0][0] + S.sc[0][2], y1 = (double)S.sc[0][1] + S.sc[0][3];
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1;
    *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}

static void pass_begin(void) {
    if (F.in_pass) return;
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX;
    F.ncolor = 0, F.has_depth = 0;
    memset(&F.formats, 0, sizeof F.formats);
    VkRenderingAttachmentInfo ca[8], da, sa;
    int n = S.rt.n < 0 ? 0 : S.rt.n > 8 ? 8 : S.rt.n;
    for (int i = 0; i < n; i++) {
        F.formats.color[i] = VK_FORMAT_UNDEFINED;
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                            .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        if (!S.rt.color[i].present) continue;
        target_of(&S.rt.color[i], &F.color[i]);
        ca[i].imageView = F.color[i].view;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        F.formats.color[i] = F.color[i].format;
        if (F.color[i].width < w) w = F.color[i].width;
        if (F.color[i].height < h) h = F.color[i].height;
        if (F.color[i].layers < layers) layers = F.color[i].layers;
        F.ncolor = (uint32_t)i + 1;
    }
    F.formats.ncolor = F.ncolor;
    F.formats.depth = F.formats.stencil = VK_FORMAT_UNDEFINED;
    if (S.rt.depth.present) {
        target_of(&S.rt.depth, &F.depth);
        F.has_depth = 1;
        if (F.depth.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) F.formats.depth = F.depth.format;
        if (F.depth.aspect & VK_IMAGE_ASPECT_STENCIL_BIT) F.formats.stencil = F.depth.format;
        if (F.depth.width < w) w = F.depth.width;
        if (F.depth.height < h) h = F.depth.height;
        if (F.depth.layers < layers) layers = F.depth.layers;
    }
    if (w == UINT32_MAX) area_without_targets(&w, &h), layers = 1;
    F.pass_w = w, F.pass_h = h, F.pass_layers = layers;
    da = sa = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                          .imageView = F.depth.view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                                          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = layers,
                          .colorAttachmentCount = F.ncolor, .pColorAttachments = ca,
                          .pDepthAttachment = F.formats.depth ? &da : NULL,
                          .pStencilAttachment = F.formats.stencil ? &sa : NULL};
    full_barrier();
    vkrec_BeginRendering(&ri);
    F.in_pass = 1, F.pass_any = 1;
    F.pass_draws = 0, F.pass_vs = F.pass_fs = NULL;
    F.dyn_valid = 0;   // the viewport's flip and the scissor clip depend on the pass's height
    g_stats.passes++;
}

// ---------- dynamic state
static int lower_left(void) { return nvn_window_origin_mode() == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT; }

// A rectangle in NVN's window coordinates, as rows from the top of the render area: the lower-left origin measures y
// from the bottom (Inference from GL's window origin, which NVN follows; DKB has only the upper-left one).
static double rect_top(double y, double h) { return lower_left() ? (double)F.pass_h - y - h : y; }

static void dyn_compute(Dyn *d) {
    memset(d, 0, sizeof *d);
    // As many viewports and scissors as the recording set (the two counts must be equal, so the longer one decides; one
    // when it set neither), the rest the whole render area: the device enables multiViewport (oot2/vk_required.h), and
    // viewport 0 is the one a program that writes no viewport index draws to.
    uint32_t n = (uint32_t)(S.nvp > S.nsc ? S.nvp : S.nsc);
    if (n < 1 || vkr_no_features) n = 1;
    d->nvp = n;
    for (uint32_t i = 0; i < n; i++) {
        double x = 0, y = 0, w = F.pass_w, h = F.pass_h;
        if ((int)i < S.nvp) x = S.vp[i][0], y = S.vp[i][1], w = S.vp[i][2], h = S.vp[i][3];
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        double top = rect_top(y, h);
        // The flip: NDC +y at the rectangle's top row, as the console's memory holds it (D18; translator convention,
        // xlate/host/Accessor.cs: the window origin is upper left and Y is flipped by the viewport).
        d->vp[i] = vkr_no_flip ? (VkViewport){(float)x, (float)top, (float)w, (float)h, S.dr[i][0], S.dr[i][1]}
                               : (VkViewport){(float)x, (float)(top + h), (float)w, (float)-h, S.dr[i][0], S.dr[i][1]};
        int64_t sx = 0, sy = 0, sw = F.pass_w, sh = F.pass_h;
        if (S.has_sc && (int)i < S.nsc) {
            sx = S.sc[i][0], sw = S.sc[i][2], sh = S.sc[i][3];
            sy = (int64_t)rect_top(S.sc[i][1], S.sc[i][3]);
        }
        // Into the render area (a scissor larger than it, as a game's "no scissor" is, must not reach past it).
        int64_t sx1 = sx + sw, sy1 = sy + sh;
        if (sx < 0) sx = 0;
        if (sy < 0) sy = 0;
        if (sx1 > (int64_t)F.pass_w) sx1 = F.pass_w;
        if (sy1 > (int64_t)F.pass_h) sy1 = F.pass_h;
        if (sx1 < sx) sx1 = sx;
        if (sy1 < sy) sy1 = sy;
        d->sc[i] = (VkRect2D){{(int32_t)sx, (int32_t)sy}, {(uint32_t)(sx1 - sx), (uint32_t)(sy1 - sy)}};
    }
    // raster
    VkCullModeFlags cull = VK_CULL_MODE_NONE;
    VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPolygonMode mode = VK_POLYGON_MODE_FILL;
    if (S.has_poly) {
        if (!vkn_cull((int)S.poly.cull, &cull)) stop("cull face %u", S.poly.cull);
        if (!vkn_front_face((int)S.poly.front, &front)) stop("front face %u", S.poly.front);
        if (!vkn_polygon_mode((int)S.poly.mode, &mode)) stop("polygon mode %u", S.poly.mode);
    }
    if (lower_left() && !vkr_no_flip)
        front = front == VK_FRONT_FACE_CLOCKWISE ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    d->cull = cull, d->front = front, d->mode = mode;
    // NVNpolygonOffsetEnable: point 1, line 2, fill 4; the enable that matches the polygon mode
    uint32_t want = mode == VK_POLYGON_MODE_POINT ? 1 : mode == VK_POLYGON_MODE_LINE ? 2 : 4;
    d->bias_en = S.has_poly && (S.poly.offset_en & want);
    d->bias[0] = S.bias[0], d->bias[1] = S.bias[1];
    d->bias[2] = S.bias[2];
    d->clamp = (VkBool32)S.depth_clamp;
    // Line and point fill, the bias clamp, depth clamp, depth bounds and wide lines are core features the device
    // enables (oot2/vk_required.h); the width is clamped by the driver to its range.
    if (!(S.line_width > 0.0f) || S.line_width > 1.0e6f) stop("a line width of %g", S.line_width);
    d->line_width = S.line_width;
    d->discard = (VkBool32)S.raster_discard;
    d->clip11 = nvn_depth_mode() == NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
    // depth and stencil
    d->df = VK_COMPARE_OP_LESS;
    d->sfn[0] = d->sfn[1] = VK_COMPARE_OP_ALWAYS;
    for (int f = 0; f < 2; f++)
        for (int k = 0; k < 3; k++) d->sop[f][k] = VK_STENCIL_OP_KEEP;
    if (S.has_ds) {
        d->dt = S.ds.dt, d->dw = S.ds.dw, d->st = S.ds.st;
        if (!vkn_compare((int)S.ds.df, &d->df)) stop("depth function %u", (unsigned)S.ds.df);
        if (!vkn_compare((int)S.ds.sf_f, &d->sfn[0]) || !vkn_compare((int)S.ds.sf_b, &d->sfn[1]))
            stop("stencil functions %u, %u", (unsigned)S.ds.sf_f, (unsigned)S.ds.sf_b);
        const unsigned ops[2][3] = {{S.ds.fail_f, S.ds.zfail_f, S.ds.zpass_f}, {S.ds.fail_b, S.ds.zfail_b, S.ds.zpass_b}};
        for (int f = 0; f < 2; f++)
            for (int k = 0; k < 3; k++)
                if (!vkn_stencil_op((int)ops[f][k], &d->sop[f][k])) stop("stencil operation %u", ops[f][k]);
    }
    for (int f = 0; f < 2; f++) d->smask[f][0] = S.st_vmask[f], d->smask[f][1] = S.st_wmask[f], d->smask[f][2] = S.st_ref[f];
    d->bt = (VkBool32)S.bounds_en;
    d->bounds[0] = S.bounds[0], d->bounds[1] = S.bounds[1];
    if (vkr_no_features)   // the counter-run: the state a device without these features could say
        d->mode = VK_POLYGON_MODE_FILL, d->bias[2] = 0.0f, d->clamp = VK_FALSE, d->line_width = 1.0f, d->bt = VK_FALSE;
    // colour
    d->ntargets = F.ncolor;
    for (uint32_t i = 0; i < F.ncolor; i++) {
        d->wm[i] = (S.chmask.mask >> (4 * i)) & 0xF;
        d->blend_en[i] = (S.color.blend_en >> i) & 1;
        d->eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                            VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
        if (S.has_blend[i]) {
            const blend_st *b = &S.blend[i];
            if (b->adv) stop("advanced blend mode %u on target %u is not replayed yet", (unsigned)b->adv, i);
            if (!vkn_blend_factor((int)b->src_rgb, &d->eq[i].srcColorBlendFactor) ||
                !vkn_blend_factor((int)b->dst_rgb, &d->eq[i].dstColorBlendFactor) ||
                !vkn_blend_factor((int)b->src_a, &d->eq[i].srcAlphaBlendFactor) ||
                !vkn_blend_factor((int)b->dst_a, &d->eq[i].dstAlphaBlendFactor) ||
                !vkn_blend_op((int)b->eq_rgb, &d->eq[i].colorBlendOp) || !vkn_blend_op((int)b->eq_a, &d->eq[i].alphaBlendOp))
                stop("blend state of target %u (functions %u %u %u %u, equations %u %u)", i, (unsigned)b->src_rgb,
                     (unsigned)b->dst_rgb, (unsigned)b->src_a, (unsigned)b->dst_a, (unsigned)b->eq_rgb, (unsigned)b->eq_a);
        }
    }
    // Per-target blend state and dual-source factors need independentBlend and dualSrcBlend (oot2/vk_required.h), both
    // enabled. An equation blending does not use is the default, so that two draws that differ only there compare alike.
    for (uint32_t i = 0; i < F.ncolor; i++) {
        const VkColorBlendEquationEXT *e = &d->eq[i];
        if (!d->blend_en[i]) {
            d->eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                                VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
            continue;
        }
        // Vulkan guarantees one colour target with a dual-source factor (maxFragmentDualSrcAttachments is at least 1).
        const VkBlendFactor f[4] = {e->srcColorBlendFactor, e->dstColorBlendFactor, e->srcAlphaBlendFactor,
                                    e->dstAlphaBlendFactor};
        for (int k = 0; k < 4; k++)
            if (f[k] >= VK_BLEND_FACTOR_SRC1_COLOR && f[k] <= VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA && F.ncolor > 1)
                stop("dual-source blending on target %u of %u: Vulkan guarantees one colour target with it", i, F.ncolor);
    }
    memcpy(d->blend_color, S.blend_color, sizeof d->blend_color);
    if (S.color.logic_op && S.color.logic_op != NVN_LOGIC_OP_COPY)
        stop("logic operation %u is not replayed yet", (unsigned)S.color.logic_op);
}

#define DIFF(field) (!F.dyn_valid || memcmp(&n.field, &o->field, sizeof n.field))
static void dyn_emit(VkPrimitiveTopology topo, VkBool32 restart, int mesh) {
    Dyn n;
    dyn_compute(&n);
    n.topo = topo, n.restart = restart;
    Dyn *o = &F.dyn;
    if (DIFF(vp) || DIFF(nvp)) vkrec_SetViewportWithCount(n.nvp, n.vp);
    if (DIFF(sc) || DIFF(nvp)) vkrec_SetScissorWithCount(n.nvp, n.sc);
    if (DIFF(cull)) vkrec_SetCullMode(n.cull);
    if (DIFF(front)) vkrec_SetFrontFace(n.front);
    if (DIFF(mode)) vkrec_SetPolygonModeEXT(n.mode);
    if (DIFF(bias_en)) vkrec_SetDepthBiasEnable(n.bias_en);
    if (DIFF(bias)) vkrec_SetDepthBias(n.bias[1], n.bias[2], n.bias[0]);
    if (DIFF(line_width)) vkrec_SetLineWidth(n.line_width);
    if (DIFF(discard)) vkrec_SetRasterizerDiscardEnable(n.discard);
    if (DIFF(clamp)) vkrec_SetDepthClampEnableEXT(n.clamp);
    if (DIFF(clip11)) vkrec_SetDepthClipNegativeOneToOneEXT(n.clip11);
    if (DIFF(dt)) vkrec_SetDepthTestEnable(n.dt);
    if (DIFF(dw)) vkrec_SetDepthWriteEnable(n.dw);
    if (DIFF(df)) vkrec_SetDepthCompareOp(n.df);
    if (DIFF(bt)) vkrec_SetDepthBoundsTestEnable(n.bt);
    if (DIFF(bounds)) vkrec_SetDepthBounds(n.bounds[0], n.bounds[1]);
    if (DIFF(st)) vkrec_SetStencilTestEnable(n.st);
    if (DIFF(sop) || DIFF(sfn)) {
        vkrec_SetStencilOp(VK_STENCIL_FACE_FRONT_BIT, n.sop[0][0], n.sop[0][2], n.sop[0][1], n.sfn[0]);
        vkrec_SetStencilOp(VK_STENCIL_FACE_BACK_BIT, n.sop[1][0], n.sop[1][2], n.sop[1][1], n.sfn[1]);
    }
    if (DIFF(smask)) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkrec_SetStencilCompareMask(face, n.smask[f][0]);
            vkrec_SetStencilWriteMask(face, n.smask[f][1]);
            vkrec_SetStencilReference(face, n.smask[f][2]);
        }
    }
    if (!mesh && DIFF(topo)) vkrec_SetPrimitiveTopology(n.topo);
    if (!mesh && (!o->restart_valid || DIFF(restart))) vkrec_SetPrimitiveRestartEnable(n.restart);
    if (DIFF(blend_color)) vkrec_SetBlendConstants(n.blend_color);
    if (DIFF(logic)) vkrec_SetLogicOpEnableEXT(VK_FALSE);
    if (n.ntargets && (DIFF(ntargets) || DIFF(blend_en) || DIFF(eq) || DIFF(wm))) {
        vkrec_SetColorBlendEnableEXT(0, n.ntargets, n.blend_en);
        vkrec_SetColorBlendEquationEXT(0, n.ntargets, n.eq);
        vkrec_SetColorWriteMaskEXT(0, n.ntargets, n.wm);
    }
    if (mesh) n.topo = VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, n.restart_valid = VK_FALSE;
    else n.restart_valid = VK_TRUE;
    *o = n;
    F.dyn_valid = 1;
}
#undef DIFF

// ---------- set 0 and pipelines
static void bind_set0(int compute) {
    // No pool set yet: NULL, the tables as they stand (vkb_acquire).
    if (!F.set0)
        F.set0 = F.ops->acquire(F.value, S.has_tex_pool ? &S.tex_pool : NULL, S.has_samp_pool ? &S.samp_pool : NULL);
    if (F.set0_bound[compute]) return;
    vkrec_BindDescriptorSets(compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
                             compute ? F.ops->compute_layout : F.ops->graphics_layout, 0, 1, &F.set0, 0, NULL);
    F.set0_bound[compute] = 1;
}

static void bind_pipeline(int compute, VkPipeline p) {
    if (F.pipe[compute] == p) return;
    vkrec_BindPipeline(compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    F.pipe[compute] = p;
}

static VkpStage replay_stage(const nvn_stage *s) {
    uint8_t digest[SHC_DIGEST_BYTES];
    ledger_track_answers_digest(&S.track, s->kind, digest);
    VkpStage stage = {.key = s->key, .shape = s};
    vkp_stage_set_ledger_digest(&stage, digest);
    return stage;
}

static const ShaderBinds *binds_of(const VkpStage *stage, int kind) {
    const ShaderBinds *b = F.ops->binds(stage);
    if (!b) stop("stage %s has no description the replay can read", stage->key);
    if (b->kind != kind) stop("stage %s is a %s program where a %s was bound", stage->key, shd_kind_name(b->kind),
                              shd_kind_name(kind));
    return b;
}

// ---------- one stage's descriptors and texture slots
typedef struct {
    VkWriteDescriptorSet w[VKP_SET1_MAX];
    VkDescriptorBufferInfo bi[VKP_SET1_MAX];
    VkDescriptorImageInfo ii[VKP_SET1_MAX];
    uint32_t n;
    uint32_t push[VKP_PUSH_BYTES / 4];
} Set1;

// A raw-access stage reaches memory through D15's page table: its address and span in push words 0 to 3, which every
// stage of the draw shares (oot2/raw_table.h). The first one in a frame keeps the map's invariant (oot2/vk_raw.h), and
// the draw or dispatch asks for the words before any of its binds resolves: from then on nothing the frame binds becomes
// a mirror (oot2/backing.h, The raw path), and the table is rewritten at the frame's end to map what its binds made.
static void raw_words(Set1 *s, const nvn_stage *stage) {
    if (!F.ops->raw) stop("stage %s reaches memory through raw addresses, and these ops give no page table", stage->key);
    F.ops->raw(F.value, s->push);
    g_stats.raw_binds++;
    note('R', 0, 0, 0);
}

static void put_buffer(Set1 *s, uint32_t binding, VkDescriptorType type, Bound b, VkDeviceSize range) {
    if (s->n == VKP_SET1_MAX) stop("more than %u descriptors at set 1", VKP_SET1_MAX);
    s->bi[s->n] = (VkDescriptorBufferInfo){b.buf, b.off, range};
    s->w[s->n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = binding,
                                        .descriptorCount = 1, .descriptorType = type, .pBufferInfo = &s->bi[s->n]};
    s->n++;
}

static void put_image(Set1 *s, uint32_t binding, VkImageView view) {
    if (s->n == VKP_SET1_MAX) stop("more than %u descriptors at set 1", VKP_SET1_MAX);
    s->ii[s->n] = (VkDescriptorImageInfo){VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    s->w[s->n] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = binding,
                                        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                        .pImageInfo = &s->ii[s->n]};
    s->n++;
}

// The word at byte `off` of the bank `slot` of stage st as the replay binds it: c[0] from what it wrote, a user bank
// from guest memory (a storage buffer whose address the game keeps in its own uniform buffer).
static uint64_t bank_word(int st, const uint8_t *c0, uint32_t slot, uint32_t off, uint32_t bytes) {
    uint64_t v = 0;
    if (slot == 0) {
        if (off + bytes > VKN_BANK_BYTES) stop("a storage buffer's address at c[0] %#x is past the driver bank", off);
        memcpy(&v, c0 + off, bytes);
        return v;
    }
    if (slot < 3 || slot - 3 >= UBO_SLOTS) stop("a storage buffer's address in bank c[%u]", slot);
    VknBuf u = S.ubo[st][slot - 3];
    if (!u.addr || off + bytes > u.size) return 0;
    if (gfx_gpu_only(u.addr)) stop("a storage buffer's address in bank c[%u], which the CPU cannot read", slot);
    memcpy(&v, (const void *)(uintptr_t)(u.addr + off), bytes);
    return v;
}

static void stage_resources(Set1 *s, int st, const nvn_stage *stage, const prog_stage_t *ps, const ShaderBinds *b,
                            const uint8_t *c0, uint32_t descriptor_set) {
    for (uint32_t i = 0; i < b->ncb; i++) {
        const ShdCbuf *cb = &b->cb[i];
        if (cb->set != descriptor_set)
            stop("stage %s reads constant bank c[%u] from set %u, expected set %u", stage->key, cb->slot, cb->set,
                 descriptor_set);
        Bound at;
        VkDeviceSize range = UNIFORM_MAX;
        if (cb->slot == 0) {
            at = bank(&F.c0[st], c0, VKN_BANK_BYTES), range = VKN_BANK_BYTES;
        } else if (cb->slot == 1) {
            uint32_t c1_at, c1_end;
            memcpy(&c1_at, stage->ctl + 0x40, 4);
            memcpy(&c1_end, stage->ctl + 0x44, 4);
            if (c1_end > c1_at && ps && ps->code) {
                at = uniform((VknBuf){ps->code + c1_at, c1_end - c1_at});
                range = c1_end - c1_at < UNIFORM_MAX ? c1_end - c1_at : UNIFORM_MAX;
            } else {
                at = zero_block();
            }
        } else if (cb->slot == 2) {
            static const uint64_t none[128];
            uint8_t c2[VKN_BANK_BYTES];
            vkn_c2_fill(c2, S.tex[st], S.img[st], none, none);
            at = bank(&F.c2[st], c2, VKN_BANK_BYTES), range = VKN_BANK_BYTES;
        } else if (cb->slot == 24) {
            int k = st == ST_CS;
            at = bank(&F.roots[k], (const uint8_t *)S.root[k], ROOT_BANK_BYTES), range = ROOT_BANK_BYTES;
        } else if (cb->slot >= 3 && cb->slot - 3 < UBO_SLOTS) {
            VknBuf u = S.ubo[st][cb->slot - 3];
            at = uniform(u);
            if (u.addr && u.size) range = u.size < UNIFORM_MAX ? u.size : UNIFORM_MAX;
        } else {
            stop("stage %s reads constant bank c[%u]", stage->key, cb->slot);
        }
        if (cb->slot == 0 || cb->slot == 2 || cb->slot == 24) note('B', 0, range, 0);   // a bank: the frame's scratch memory
        put_buffer(s, cb->binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, at, range);
    }
    for (uint32_t i = 0; i < b->nsb; i++) {
        const ShdSbuf *sb = &b->sb[i];
        if (sb->set != descriptor_set)
            stop("stage %s reads storage buffer %u from set %u, expected set %u", stage->key, sb->ordinal, sb->set,
                 descriptor_set);
        uint64_t addr = bank_word(st, c0, sb->cb_slot, sb->address_byte_offset, 8);
        uint64_t size = bank_word(st, c0, sb->cb_slot, sb->address_byte_offset + 8u, 4);
        if (!addr || !size) {   // unbound: zeros; one the module writes gets its own, so the shared zeros stay zero
            Bound at = zero_block();
            if (sb->write) {
                void *p;
                at = scratch(STORAGE_ALIGN, STORAGE_ALIGN, &p);
                memset(p, 0, STORAGE_ALIGN);
            }
            put_buffer(s, sb->binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, at, STORAGE_ALIGN);
            continue;
        }
        uint64_t base = addr & ~(uint64_t)(STORAGE_ALIGN - 1), range = size + (addr - base);
        BackingRef ref;
        F.ops->resolve(base, range, F.value, &ref);
        put_buffer(s, sb->binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (Bound){(VkBuffer)(uintptr_t)ref.buffer, ref.offset},
                   range);
        note(backing_letter(ref.kind), base, range, sb->write);
        if (sb->write) vkr_wrote(base, range, 0, F.draws + 1);   // the draw being prepared
    }
    // Storage images: the handle in c[2]'s image rows names a texture pool ID (nvnDeviceGetImageHandle), whose texture
    // the device views for stores, in the general layout every image keeps (D18)
    for (uint32_t i = 0; i < b->nimg; i++) {
        const ShdImg *m = &b->img[i];
        if (!F.ops->storage_image) stop("stage %s binds a storage image, and these ops make no storage view", stage->key);
        if (m->set != 1) stop("stage %s has a storage image at set %u", stage->key, (unsigned)m->set);
        if (m->view == BV_NONE) stop("stage %s stores to a buffer texture", stage->key);
        if (m->c2_slot < 0 || m->c2_slot >= IMG_SLOTS) stop("stage %s stores to image slot %d", stage->key, m->c2_slot);
        uint32_t id = nvn_handle_texture_id(S.img[st][m->c2_slot]);
        uint64_t storage = 0, bytes = 0;
        VkImageView view = F.ops->storage_image(id, m->view, F.value, &storage, &bytes);
        if (!view) stop("stage %s stores to image ID %u, where nothing is registered", stage->key, id);
        put_image(s, m->binding, view);
        note('G', storage, bytes, m->write);
        g_stats.storage_images++;
        if (m->write && bytes) vkr_wrote(storage, bytes, 1, F.draws + 1);   // the draw or dispatch being prepared
    }
    for (uint32_t i = 0; i < b->ntex; i++) {
        const ShdTex *t = &b->tex[i];
        if (t->via == SHD_VIA_POOL) continue;   // the shader reads the pool by a handle from memory
        if (t->via != SHD_VIA_PUSH_SLOT) stop("stage %s has a texture bound as a descriptor", stage->key);
        if (t->kind != SHD_TEX_COMBINED) stop("stage %s binds a separate texture or sampler", stage->key);
        if (t->view == BV_NONE) stop("stage %s reads a buffer texture", stage->key);
        if (t->array_length > 1) stop("stage %s reads an array of %u textures through push slots", stage->key,
                                      (unsigned)t->array_length);
        if (t->c2_slot < 0 || t->c2_slot >= TEX_SLOTS) stop("stage %s reads texture slot %d", stage->key, t->c2_slot);
        if (t->push_offset < 16 || t->push_offset + 4 > (int)VKP_PUSH_BYTES)
            stop("stage %s's texture slot at push constant byte %d", stage->key, t->push_offset);
        uint64_t h = S.tex[st][t->c2_slot];
        uint32_t id = nvn_handle_texture_id(h), sampler = nvn_handle_sampler_id(h);
        uint32_t slot = F.ops->image_slot(id, t->view);
        if (slot == UINT32_MAX) slot = id, g_stats.unregistered_textures++;
        s->push[t->push_offset / 4] = slot | sampler << 20;
    }
}

static void support(Set1 *s) {
    void *p;
    Bound at = scratch(VKN_SUPPORT_BYTES, UNIFORM_ALIGN, &p);
    float w = F.pass_w, h = F.pass_h;
    if (S.nvp) w = S.vp[0][2], h = S.vp[0][3];
    vkn_support_fill(p, w, h);
    put_buffer(s, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, at, VKN_SUPPORT_BYTES);
}

// ---------- draws
// The frame that last drew into or cleared each image that stood as a render target (the miss path may skip a draw only
// where the targets hold an earlier frame's picture, oot2/vk_miss.h). A small table: a frame has a few dozen targets
// (the NVN2 call census: 127 target changes a Forest frame, over far fewer images); an image that falls out is taken to hold
// nothing, which makes its draws wait, not skip.
#define STAMPS 128
static struct {
    uint64_t image, frame;
} T[STAMPS];
static uint32_t T_next;

static void stamp_image(uint64_t image) {
    if (!image) return;
    for (uint32_t i = 0; i < STAMPS; i++)
        if (T[i].image == image) {
            T[i].frame = F.value;
            return;
        }
    T[T_next % STAMPS].image = image, T[T_next % STAMPS].frame = F.value;
    T_next++;
}

static uint64_t stamp_of(uint64_t image) {
    for (uint32_t i = 0; i < STAMPS; i++)
        if (T[i].image == image) return T[i].frame;
    return 0;
}

static void stamp_targets(void) {
    for (uint32_t i = 0; i < F.ncolor; i++) stamp_image(F.color[i].image);
    if (F.has_depth) stamp_image(F.depth.image);
}

// 1 when every target of the pass was drawn into or cleared by an earlier frame and not yet by this one.
static int targets_hold_earlier_picture(void) {
    int any = 0;
    for (uint32_t i = 0; i < F.ncolor; i++) {
        if (!F.color[i].image) continue;
        uint64_t t = stamp_of(F.color[i].image);
        if (!t || t >= F.value) return 0;
        any = 1;
    }
    if (F.has_depth && F.depth.image) {
        uint64_t t = stamp_of(F.depth.image);
        if (!t || t >= F.value) return 0;
        any = 1;
    }
    return any;
}

// The bound program's stage records and code addresses, kept beside the ledger's tracker (which keeps the records).
static struct {
    const nvn_stage *id[ST_N];
    prog_stage_t ps[ST_N];
} P;

static void bind_program(const p_bind_program *p) {
    for (int s = 0; s < ST_N; s++) {
        int applies = (p->stages >> s) & 1;
        if ((s == ST_TS || s == ST_MS) && (p->stages & 0x1F)) applies = 1;
        if (!applies) continue;
        P.id[s] = NULL;
        memset(&P.ps[s], 0, sizeof P.ps[s]);
        for (int i = 0; i < p->nstage && i < 6; i++) {
            const nvn_stage *id = p->id[i];
            if (!id) continue;
            int st = id->kind == NVN_STAGE_VS ? ST_VS : id->kind == NVN_STAGE_PS ? ST_FS : id->kind == NVN_STAGE_CS ? ST_CS
                     : id->kind == NVN_STAGE_TS ? ST_TS : ST_MS;
            if (st == s) P.id[s] = id, P.ps[s] = (prog_stage_t){p->code[i], id};
        }
    }
    vkrec_stages(P.id[ST_VS] ? P.id[ST_VS] : P.id[ST_MS], P.id[ST_FS], P.id[ST_TS]);
    // The device counts the stage keys a live program holds, to know when a released program's pipelines may go.
    // Every bind, not only a new serial: a program finalized since the last bind must take its keys back.
    if (p->serial && F.ops->program_bound) F.ops->program_bound(p->serial, p->id, p->nstage < 6 ? p->nstage : 6);
}

typedef struct {
    VkVertexInputBindingDescription2EXT b[NVN_VERTEX_BINDINGS];
    VkVertexInputAttributeDescription2EXT a[NVN_VERTEX_BINDINGS];
    uint32_t nb, na;
} VertexInput;

static void vertex_input(const ShaderBinds *vb) {
    VertexInput in;
    memset(&in, 0, sizeof in);
    uint32_t used = 0;   // streams the attributes read
    int need_default = 0;
    for (uint32_t i = 0; i < vb->nvin; i++) {
        uint32_t loc = vb->vin[i].location;
        if (loc >= NVN_VERTEX_BINDINGS) stop("vertex input location %u", loc);
        va_st a;
        memcpy(&a, &S.vattr[loc], sizeof a);
        VkFormat fmt = VK_FORMAT_UNDEFINED;
        uint32_t bytes = 0;
        int type = 0;
        if (!a.fmt || !S.vbuf[a.stream].address) {
            need_default = 1;
            in.a[in.na++] = (VkVertexInputAttributeDescription2EXT){
                .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = loc,
                .binding = UINT32_MAX, .offset = 16u * vb->vin[i].type,   // its type's row of the spare binding
                .format = vb->vin[i].type == SHD_SINT   ? VK_FORMAT_R32G32B32A32_SINT
                          : vb->vin[i].type == SHD_UINT ? VK_FORMAT_R32G32B32A32_UINT
                                                        : VK_FORMAT_R32G32B32A32_SFLOAT};
            continue;
        }
        if (!vkn_vertex_format((int)a.fmt, &fmt, &bytes, &type))
            stop("vertex attribute %u has NVN format %#x, which has no vertex format", loc, (unsigned)a.fmt);
        if ((type == VKN_ATTR_FLOAT) != (vb->vin[i].type == SHD_FLOAT) ||
            (type != VKN_ATTR_FLOAT && (type == VKN_ATTR_SINT) != (vb->vin[i].type == SHD_SINT)))
            stop_answer(P.id[ST_VS]->key, "vertex attribute %u (NVN format %#x) is of another scalar type than the module's "
                        "input", loc, (unsigned)a.fmt);
        in.a[in.na++] = (VkVertexInputAttributeDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = loc, .binding = a.stream,
            .format = fmt, .offset = a.offset};
        used |= 1u << a.stream;
    }
    uint32_t spare = UINT32_MAX;
    for (uint32_t s = 0; s < NVN_VERTEX_BINDINGS; s++) {
        if (!(used >> s & 1)) {
            if (spare == UINT32_MAX) spare = s;
            continue;
        }
        uint32_t div = S.vstream[s].divisor;
        if (div > 1) stop("vertex stream %u steps every %u instances", s, div);
        in.b[in.nb++] = (VkVertexInputBindingDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = s, .stride = S.vstream[s].stride,
            .inputRate = div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
        BackingRef ref;
        uint64_t size = S.vbuf[s].size ? S.vbuf[s].size : 1;
        F.ops->resolve(S.vbuf[s].address, size, F.value, &ref);
        VkBuffer buf = (VkBuffer)(uintptr_t)ref.buffer;
        VkDeviceSize off = ref.offset;
        vkrec_BindVertexBuffers2(s, 1, &buf, &off, NULL, NULL);
    }
    if (need_default) {
        // An attribute with no format or no buffer reads (0, 0, 0, 1), as the console reads an unfed input.
        if (spare == UINT32_MAX) stop("every vertex stream is in use and an attribute has no buffer");
        void *p;
        Bound at = scratch(48, 16, &p);   // (0, 0, 0, 1) as floats, signed and unsigned integers (SHD_FLOAT, SINT, UINT)
        const float one[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        const uint32_t ione[4] = {0, 0, 0, 1};
        memcpy(p, one, sizeof one);
        memcpy((uint8_t *)p + 16, ione, sizeof ione);
        memcpy((uint8_t *)p + 32, ione, sizeof ione);
        in.b[in.nb++] = (VkVertexInputBindingDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = spare, .stride = 0,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
        for (uint32_t i = 0; i < in.na; i++)
            if (in.a[i].binding == UINT32_MAX) in.a[i].binding = spare;
        vkrec_BindVertexBuffers2(spare, 1, &at.buf, &at.off, NULL, NULL);
    }
    vkrec_SetVertexInputEXT(in.nb, in.b, in.na, in.a);
}

// The pipeline, the state and the resources of a graphics draw; 0 when rendering is disabled.
static int draw_prepare(int prim, uint32_t base_instance, VkBool32 *restart_out) {
    g_stats.draw_ops++;
    if (!S.render) {
        g_stats.render_off++;
        return 0;
    }
    if (P.id[ST_TS] || P.id[ST_MS]) stop("a vertex draw with task or mesh stages bound (the mesh draws)");
    const nvn_stage *vs = P.id[ST_VS], *fs = P.id[ST_FS];
    if (!vs) stop("a draw with no vertex stage bound");
    if (!fs) stop("a draw with no pixel stage bound (stage %s)", vs->key);
    VkpStage vst = replay_stage(vs), fst = replay_stage(fs);
    const ShaderBinds *vb = binds_of(&vst, SHD_VS), *fb = binds_of(&fst, SHD_PS);
    if (S.conservative) stop("conservative rasterization is not replayed yet");
    if (S.sample_mask != ~0u && (S.sample_mask & 1) != 1) stop("sample mask %#x is not replayed yet", S.sample_mask);
    pass_begin();
    VkPrimitiveTopology topo;
    if (!vkn_topology(prim, &topo)) stop("primitive %d has no Vulkan topology", prim);
    VkBool32 restart = (VkBool32)S.restart_en;
    *restart_out = restart;
    // A pipeline that is not ready and may be skipped (oot2/vk_miss.h) answers NULL: nothing of the draw is written, so
    // the state the frame keeps (the bound pipeline, the dynamic state, the buffers) is as the draw before left it.
    VkPipeline pipe = F.ops->graphics(&vst, NULL, &fst, &F.formats, targets_hold_earlier_picture());
    if (!pipe) {
        g_stats.skipped++;
        return 0;
    }
    bind_pipeline(0, pipe);
    stamp_targets();
    bind_set0(0);
    dyn_emit(topo, restart, 0);
    Set1 s;
    memset(&s, 0, sizeof s);
    if (vb->raw || fb->raw) raw_words(&s, vb->raw ? vs : fs);
    vertex_input(vb);
    support(&s);
    uint8_t c0[VKN_BANK_BYTES];
    vkn_c0_graphics(c0, NVN_STAGE_VS, S.ssbo[ST_VS], base_instance);
    stage_resources(&s, ST_VS, vs, &P.ps[ST_VS], vb, c0, 1);
    vkn_c0_graphics(c0, NVN_STAGE_PS, S.ssbo[ST_FS], base_instance);
    stage_resources(&s, ST_FS, fs, &P.ps[ST_FS], fb, c0, 1);
    vkrec_PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, F.ops->graphics_layout, 1, s.n, s.w);
    vkrec_PushConstants(F.ops->graphics_layout, vkp_push_range(VKP_GRAPHICS).stageFlags, 0, VKP_PUSH_BYTES, s.push);
    vkr_report_draw();
    F.draws++, g_stats.draws++;
    if (!F.pass_draws++) F.pass_vs = vs->key, F.pass_fs = fs->key;
    return 1;
}

// A record's instance count: 0 is one instance (DrawArrays and DrawElements record 0, nvn_obj.h).
static uint32_t instances(int32_t n) { return n > 0 ? (uint32_t)n : 1u; }

static void op_draw(const p_draw *d) {
    VkBool32 restart;
    if (!draw_prepare(d->prim, (uint32_t)d->base_instance, &restart)) return;
    vkrec_Draw((uint32_t)d->count, instances(d->instances), (uint32_t)d->first, (uint32_t)d->base_instance);
}

static void index_buffer(int itype, uint64_t ibuf, uint64_t count) {
    VkIndexType t;
    uint32_t bytes;
    if (!vkn_index_type(itype, &t, &bytes)) stop("index type %d", itype);
    BackingRef ref;
    F.ops->resolve(ibuf, count ? count * bytes : bytes, F.value, &ref);
    vkrec_BindIndexBuffer((VkBuffer)(uintptr_t)ref.buffer, ref.offset, t);
    if (S.restart_en) {
        uint32_t all = bytes == 1 ? 0xFFu : bytes == 2 ? 0xFFFFu : 0xFFFFFFFFu;
        if (S.restart_index != all) stop("primitive restart at index %#x with %u-byte indices", S.restart_index, bytes);
    }
}

static void op_draw_idx(const p_draw_idx *d) {
    VkBool32 restart;
    if (!draw_prepare(d->prim, (uint32_t)d->base_instance, &restart)) return;
    index_buffer(d->itype, d->ibuf, (uint64_t)(uint32_t)d->count);
    vkrec_DrawIndexed((uint32_t)d->count, instances(d->instances), 0, d->base_vertex, (uint32_t)d->base_instance);
}

static void op_draw_idx_ind(const p_draw_idx_ind *d) {
    VkBool32 restart;
    // The record is {count, instances, first index, base vertex, base instance}. Its base instance goes into c[0] too,
    // read at submit from the CPU's view of the record (the pool's storage behind an import or a mirror, not written by
    // this submission); a record the CPU cannot see, or that an earlier submission still in flight wrote, keeps 0 or the
    // CPU's older word there, while the draw itself takes the GPU's first instance. A GPU copy into the bank, as an
    // indirect dispatch does, would have to run outside the pass: left to the title run if a program reads it.
    BackingRef ref;
    F.ops->resolve(d->ind, 20, F.value, &ref);
    uint32_t base_instance = 0;
    int readable = ref.kind == BACKING_IMPORT || ref.kind == BACKING_MIRROR;
    for (uint32_t i = 0; readable && i < F.nwrote; i++)
        if (overlaps((Rng){d->ind, d->ind + 20}, F.wrote[i])) readable = 0;
    if (readable) memcpy(&base_instance, (const uint8_t *)(uintptr_t)d->ind + 16, 4);
    if (!draw_prepare(d->prim, base_instance, &restart)) return;
    index_buffer(d->itype, d->ibuf, 0);
    vkrec_DrawIndexedIndirect((VkBuffer)(uintptr_t)ref.buffer, ref.offset, 1, 20);
}

static void mesh_first_task_push(Set1 *s, VkDeviceAddress address) {
    if (!address) stop("a mesh draw's first-task word has no device address");
    if (vkr_no_first_task) address = 0;
    s->push[MESH_FIRST_TASK_PUSH_WORD] = (uint32_t)address;
    s->push[MESH_FIRST_TASK_PUSH_WORD + 1] = (uint32_t)(address >> 32);
    s->push[MESH_FIRST_TASK_PUSH_WORD + 2] = 0;
}

static Bound mesh_indirect_record(uint64_t indirect) {
    if (!indirect || indirect > UINT64_MAX - MESH_INDIRECT_COMMAND_BYTES + 4)
        stop("a mesh indirect draw's record address is invalid");
    BackingRef src;
    F.ops->resolve(indirect, MESH_INDIRECT_COMMAND_BYTES - 4, F.value, &src);
    void *mapped;
    Bound dst = scratch(MESH_INDIRECT_RECORD_BYTES, 16, &mapped);
    const uint32_t initial[MESH_INDIRECT_RECORD_BYTES / sizeof(uint32_t)] = {0, 1, 1, 0};
    memcpy(mapped, initial, sizeof initial);

    vkr_xfer_begin();
    VkBufferCopy count = {src.offset, dst.off, sizeof(uint32_t)};
    VkBufferCopy first = {src.offset + sizeof(uint32_t), dst.off + MESH_INDIRECT_FIRST_OFFSET, sizeof(uint32_t)};
    vkrec_CopyBuffer((VkBuffer)(uintptr_t)src.buffer, dst.buf, &count);
    vkrec_CopyBuffer((VkBuffer)(uintptr_t)src.buffer, dst.buf, &first);
    vkr_count_transfer(2 * sizeof(uint32_t));
    full_barrier();
    return dst;
}

static void op_draw_mesh(const p_draw_mesh *d, uint64_t indirect) {
    g_stats.draw_ops++;
    if (!S.render) {
        g_stats.render_off++;
        return;
    }
    if (P.id[ST_VS]) stop("a mesh draw with a vertex stage bound");
    const nvn_stage *ts = P.id[ST_TS], *ms = P.id[ST_MS], *fs = P.id[ST_FS];
    if (!ms) stop("a mesh draw with no mesh stage bound");
    if (!fs) stop("a mesh draw with no pixel stage bound (stage %s)", ms->key);
    if (ts && ts->kind != NVN_STAGE_TS) stop("mesh task stage %s has kind %u", ts->key, ts->kind);
    if (ms->kind != NVN_STAGE_MS) stop("mesh stage %s has kind %u", ms->key, ms->kind);
    if (!indirect && (d->first < 0 || d->count < 0))
        stop("a direct mesh draw with first %d and count %d", d->first, d->count);
    VkpStage task = {0};
    if (ts) task = replay_stage(ts);
    VkpStage mesh = replay_stage(ms), pixel = replay_stage(fs);
    const ShaderBinds *tb = ts ? binds_of(&task, SHD_TS) : NULL;
    const ShaderBinds *mb = binds_of(&mesh, SHD_MS), *fb = binds_of(&pixel, SHD_PS);
    if (S.conservative) stop("conservative rasterization is not replayed yet");
    if (S.sample_mask != ~0u && (S.sample_mask & 1) != 1)
        stop("sample mask %#x is not replayed yet", S.sample_mask);

    pass_begin();
    VkPipeline pipe = F.ops->graphics(&mesh, ts ? &task : NULL, &pixel, &F.formats, targets_hold_earlier_picture());
    if (!pipe) {
        g_stats.skipped++;
        return;
    }
    if (!F.ops->buffer_address) stop("mesh draws need replay scratch with buffer device addresses");

    Set1 s, task_set_data;
    memset(&s, 0, sizeof s);
    memset(&task_set_data, 0, sizeof task_set_data);
    if ((ts && tb->raw) || mb->raw || fb->raw) {
        if (ts && tb->raw) raw_words(&s, ts);
        if (mb->raw) raw_words(&s, ms);
        if (fb->raw) raw_words(&s, fs);
    }

    Bound record = {0};
    VkDeviceAddress first_address;
    if (indirect) {
        record = mesh_indirect_record(indirect);
        first_address = scratch_address(record) + MESH_INDIRECT_FIRST_OFFSET;
        pass_begin();
    } else {
        void *first_word;
        Bound first = scratch(MESH_FIRST_TASK_WORD_BYTES, MESH_FIRST_TASK_WORD_BYTES, &first_word);
        memcpy(first_word, &d->first, sizeof(uint32_t));
        first_address = scratch_address(first);
    }
    bind_pipeline(0, pipe);
    stamp_targets();
    bind_set0(0);
    dyn_emit(VK_PRIMITIVE_TOPOLOGY_MAX_ENUM, VK_FALSE, 1);
    support(&s);

    if (ts) {
        uint8_t c0[VKN_BANK_BYTES];
        vkn_c0_graphics(c0, NVN_STAGE_TS, S.ssbo[ST_TS], 0);
        stage_resources(&task_set_data, ST_TS, ts, &P.ps[ST_TS], tb, c0, 2);
        if (task_set_data.n) {
            if (!F.ops->task_set) stop("task stage %s needs set-2 descriptors, and these ops allocate none", ts->key);
            VkDescriptorSet set = F.ops->task_set(F.value, task_set_data.n, task_set_data.w);
            if (!set) stop("task stage %s has no set-2 descriptor set", ts->key);
            vkrec_BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, F.ops->graphics_layout, 2, 1, &set, 0, NULL);
        }
    }
    uint8_t c0[VKN_BANK_BYTES];
    vkn_c0_graphics(c0, NVN_STAGE_MS, S.ssbo[ST_MS], 0);
    stage_resources(&s, ST_MS, ms, &P.ps[ST_MS], mb, c0, 1);
    vkn_c0_graphics(c0, NVN_STAGE_PS, S.ssbo[ST_FS], 0);
    stage_resources(&s, ST_FS, fs, &P.ps[ST_FS], fb, c0, 1);
    mesh_first_task_push(&s, first_address);
    vkrec_PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_GRAPHICS, F.ops->graphics_layout, 1, s.n, s.w);
    vkrec_PushConstants(F.ops->graphics_layout, vkp_push_range(VKP_GRAPHICS).stageFlags, 0, VKP_PUSH_BYTES, s.push);
    vkr_report_draw();
    F.draws++, g_stats.draws++;
    if (!F.pass_draws++) F.pass_vs = ms->key, F.pass_fs = fs->key;
    if (indirect)
        vkrec_DrawMeshTasksIndirectEXT(record.buf, record.off, 1, MESH_INDIRECT_RECORD_BYTES);
    else
        vkrec_DrawMeshTasksEXT((uint32_t)d->count, 1, 1);
}

// ---------- the movie chain's stand-in (settings.h test_movie_blit, oot2/vk_movie_blit.h)
// A TEST switch that changes the picture of every movie; never set in a proof run. The chain's programs are named by the
// person's own key file. The chain's first dispatch runs as it always does, and the replay remembers the image it stores
// to; the dispatches between it and the chain's last are not run; and the last is replaced by a resample of the remembered
// picture into the image the last one stores to, which is where its output would have been. A skipped dispatch asks for
// nothing (no pipeline, no page table, no buffer of the backing map), so a frame whose chain is skipped is no raw-access
// frame, builds none of the chain's pipelines, and has nothing of the chain to write back.

// The texture pool ID of the one storage image a program of the chain stores to; a named stop when it stores to none or to
// several, so that a key file that names a program of another kind is found at the first chain and not by its picture.
static uint32_t stored_image(const nvn_stage *cs, const ShaderBinds *b, const char *role) {
    uint32_t id = UINT32_MAX;
    for (uint32_t i = 0; i < b->nimg; i++) {
        const ShdImg *m = &b->img[i];
        if (!m->write) continue;
        if (m->c2_slot < 0 || m->c2_slot >= IMG_SLOTS) stop("test_movie_blit: stage %s stores to image slot %d", cs->key, m->c2_slot);
        if (id != UINT32_MAX) stop("test_movie_blit: the chain's %s program %s stores to more than one image", role, cs->key);
        id = nvn_handle_texture_id(S.img[ST_CS][m->c2_slot]);
    }
    if (id == UINT32_MAX) stop("test_movie_blit: the chain's %s program %s stores to no image", role, cs->key);
    return id;
}

static void movie_counted(uint64_t skipped, uint64_t blits) {
    nvn_census_note_movie_blit(skipped, blits);
    capture_count(CAPTURE_SKIPPED_DRAWS, skipped + blits);   // so that the proof judge fails the capture on its own counter
    if (g_stats.movie_skipped + g_stats.movie_blits == 1)
        rt_log("gfx: vulkan: MOVIE BLIT (test_movie_blit is on): the movie's enhancement chain is replaced by a resample of "
               "its first picture. This run is NOT a proof run and the pictures of the movies are not the console's");
}

static void movie_blit(const nvn_stage *cs) {
    VkpStage st = replay_stage(cs);
    uint32_t dst_id = stored_image(cs, binds_of(&st, SHD_CS), "last");
    if (!F.ops->image_by_id) stop("test_movie_blit needs ops that give the image of a texture ID, and these give none");
    VkrImage src, dst;
    uint64_t storage = 0, bytes = 0;
    if (!F.ops->image_by_id(F.movie_src, F.value, &src, NULL, NULL))
        stop("test_movie_blit: no image stands on the chain's first picture (texture ID %u)", F.movie_src);
    if (!F.ops->image_by_id(dst_id, F.value, &dst, &storage, &bytes))
        stop("test_movie_blit: no image stands on the output of stage %s (texture ID %u)", cs->key, dst_id);
    if (src.image == dst.image) stop("test_movie_blit: the chain's first picture and its output are one image");
    for (int i = 0; i < 2; i++) {
        const VkrImage *m = i ? &dst : &src;
        if (m->type != VK_IMAGE_TYPE_2D || m->aspect != VK_IMAGE_ASPECT_COLOR_BIT || m->samples > 1 || m->layers != 1 ||
            m->levels != 1 || !m->width || !m->height)
            stop("test_movie_blit: the chain's %s image is not a single 2D colour image of one level (%ux%u, %u levels, %u "
                 "layers, %u samples)", i ? "output" : "first picture", m->width, m->height, m->levels, m->layers, m->samples);
    }
    vkr_xfer_begin();   // the render pass ends and a full barrier follows the first dispatch's stores
    VkImageBlit rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}};
    rg.srcOffsets[1] = (VkOffset3D){(int32_t)src.width, (int32_t)src.height, 1};
    rg.dstOffsets[1] = (VkOffset3D){(int32_t)dst.width, (int32_t)dst.height, 1};
    vkrec_BlitImage(src.image, dst.image, &rg, VK_FILTER_LINEAR);
    vkr_xfer_end();
    if (bytes) vkr_wrote(storage, bytes, 1, F.draws + 1);   // as the dispatch's own stores would have been declared
    F.draws++, g_stats.movie_blits++;
    if (vkgt_enabled) vkgt_mark(VKGT_XFER, "movie_blit", 0, 0, 0);
    movie_counted(0, 1);
    if (g_stats.movie_blits == 1)
        rt_log("gfx: vulkan: movie blit: the chain's last dispatch (stage %s) is replaced by a linear resample of the first "
               "one's picture, %ux%u (Vulkan format %d, texture ID %u) into %ux%u (format %d, texture ID %u)", cs->key,
               src.width, src.height, (int)src.format, F.movie_src, dst.width, dst.height, (int)dst.format, dst_id);
}

// A chain that skipped dispatches and whose blit program never came in the submission would leave the picture stale or
// garbage with nothing said (another movie that reuses the keep program, a last program the key file does not name), so
// it is a named stop at the end of the submission (or at the next keep program). A keep alone, with nothing skipped,
// leaves nothing stale.
static void movie_unclosed(void) {
    if (!F.movie_open || !F.movie_skips) return;
    stop("test_movie_blit: the chain opened by program %s skipped %u dispatches (first %s, last %s) and no blit program came "
         "in this submission, so the movie's picture would be stale", F.movie_keep, F.movie_skips, F.movie_first, F.movie_last);
}

// 1 when this dispatch is the stand-in's (skipped, or replaced by the resample), 0 when it runs as it always does: the
// setting off, a program the file does not name, a program of the chain outside an open chain.
static int movie_replaces(const nvn_stage *cs) {
    VkmbRole role = vkmb_role(cs->key);
    if (role == VKMB_NONE) return 0;
    if (role == VKMB_KEEP) {   // it runs, and its picture is the one resampled
        movie_unclosed();   // a keep program again, with the chain it opened still waiting for its blit
        VkpStage st = replay_stage(cs);
        F.movie_src = stored_image(cs, binds_of(&st, SHD_CS), "first");
        F.movie_open = 1;
        F.movie_skips = 0;
        snprintf(F.movie_keep, sizeof F.movie_keep, "%s", cs->key);
        return 0;
    }
    if (!F.movie_open) return 0;
    if (role == VKMB_BLIT) {
        movie_blit(cs);
        F.movie_open = 0;
        return 1;
    }
    g_stats.movie_skipped++;
    if (!F.movie_skips++) snprintf(F.movie_first, sizeof F.movie_first, "%s", cs->key);
    snprintf(F.movie_last, sizeof F.movie_last, "%s", cs->key);
    movie_counted(1, 0);
    if (g_stats.movie_skipped == 1)
        rt_log("gfx: vulkan: movie blit: the dispatches of the chain between its first and its last are not run (the first "
               "is stage %s; logged once, the census counts them)", cs->key);
    return 1;
}

// ---------- dispatches
// ind: the GPU address of the group counts of an indirect dispatch (0: direct, groups). The counts go into the driver bank
// c[0] too, and the CPU cannot say what they will be: they may be the output of earlier work of this submission, or of a
// submission still in flight (a mirror holds the CPU's view, which is behind). So the GPU copies them from the indirect
// buffer into the bank, which the dispatch has to itself, right before it runs.
static void dispatch(const uint32_t groups[3], uint64_t ind) {
    g_stats.draw_ops++;
    if (!S.render) {
        g_stats.render_off++;
        return;
    }
    const nvn_stage *cs = P.id[ST_CS];
    if (!cs) stop("a dispatch with no compute stage bound");
    if (movie_replaces(cs)) return;   // the movie chain's stand-in, a test switch: 0 at once when the setting is off
    VkpStage cst = replay_stage(cs);
    const ShaderBinds *b = binds_of(&cst, SHD_CS);
    pass_end();
    F.noting = vkgt_enabled;   // the pass's mark is made; the buffers from here on are this dispatch's
    if (F.after_xfer) full_barrier();
    bind_pipeline(1, F.ops->compute(&cst));
    bind_set0(1);
    Set1 s;
    memset(&s, 0, sizeof s);
    if (b->raw) raw_words(&s, cs);
    uint8_t c0[VKN_BANK_BYTES];
    if (!vkn_c0_compute(c0, cs->ctl, cs->ctl_size, groups, (const uint32_t(*)[64])S.root[1], S.ubo[ST_CS], S.ssbo[ST_CS]))
        stop("stage %s: its control section's driver bank layout does not fit %u bytes", cs->key, VKN_BANK_BYTES);
    BackingRef ref = {0};
    Bound patch = {0};
    if (ind) {
        F.ops->resolve(ind, 12, F.value, &ref);
        F.c0[ST_CS].valid = 0;   // a bank of its own: stage_resources finds these bytes and binds this copy
        patch = bank(&F.c0[ST_CS], c0, VKN_BANK_BYTES);
    }
    support(&s);
    stage_resources(&s, ST_CS, cs, &P.ps[ST_CS], b, c0, 1);
    vkrec_PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, F.ops->compute_layout, 1, s.n, s.w);
    vkrec_PushConstants(F.ops->compute_layout, vkp_push_range(VKP_COMPUTE).stageFlags, 0, VKP_PUSH_BYTES, s.push);
    F.draws++, g_stats.dispatches++;
    if (!ind) {
        vkrec_Dispatch(groups[0], groups[1], groups[2]);
        if (vkgt_enabled) vkgt_mark(VKGT_DISPATCH, cs->key, groups[0], groups[1], groups[2]);
        F.noting = 0;
        return;
    }
    VkBuffer buf = (VkBuffer)(uintptr_t)ref.buffer;
    VkBufferCopy counts = {ref.offset, patch.off + 4, 12};   // the group counts lie at byte 4 of the bank's head
    full_barrier();
    vkrec_CopyBuffer(buf, patch.buf, &counts);
    full_barrier();
    vkrec_DispatchIndirect(buf, ref.offset);
    F.c0[ST_CS].valid = 0;   // the bank holds the GPU's counts now, not the CPU's bytes
    if (vkgt_enabled) vkgt_mark(VKGT_DISPATCH, cs->key, 0, 0, 0);   // the counts are the GPU's: unknown here
    F.noting = 0;
}

static void op_dispatch_ind(const p_dispatch_ind *d) {
    static const uint32_t none[3];
    dispatch(none, d->ind);
}

// ---------- clears
static VkRect2D clear_rect(void) {
    VkRect2D r = {{0, 0}, {F.pass_w, F.pass_h}};
    if (!S.has_sc || S.nsc < 1) return r;
    int64_t x0 = S.sc[0][0], y0 = (int64_t)rect_top(S.sc[0][1], S.sc[0][3]);
    int64_t x1 = x0 + S.sc[0][2], y1 = y0 + S.sc[0][3];
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > F.pass_w) x1 = F.pass_w;
    if (y1 > F.pass_h) y1 = F.pass_h;
    if (x1 < x0) x1 = x0;
    if (y1 < y0) y1 = y0;
    return (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
}

static void op_clear_color(const p_clear_color *c) {
    if (!S.render) return;
    if (c->index < 0 || c->index >= 8) stop("a colour clear of target %d", c->index);
    pass_begin();
    if ((uint32_t)c->index >= F.ncolor || !S.rt.color[c->index].present) return;   // no target there: nothing to clear
    if ((c->mask & 0xF) != 0xF) stop("a colour clear with channel mask %#x (a masked clear needs a draw)", c->mask);
    VkClearAttachment a = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)c->index};
    memcpy(a.clearValue.color.float32, c->rgba, sizeof c->rgba);
    VkClearRect r = {clear_rect(), 0, F.pass_layers};
    if (r.rect.extent.width && r.rect.extent.height) vkrec_ClearAttachments(1, &a, 1, &r);
    stamp_image(F.color[c->index].image);
    g_stats.clears++;
}

static void op_clear_ds(const p_clear_ds *c) {
    if (!S.render) return;
    pass_begin();
    if (!F.has_depth) return;
    VkClearAttachment a = {.clearValue.depthStencil = {c->depth, (uint32_t)c->stencil}};
    if (c->depth_mask && (F.depth.aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) a.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if ((F.depth.aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && (c->stencil_mask & 0xFF)) {
        if ((c->stencil_mask & 0xFF) != 0xFF) stop("a stencil clear with write mask %#x (a masked clear needs a draw)",
                                                   c->stencil_mask);
        a.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    if (!a.aspectMask) return;
    VkClearRect r = {clear_rect(), 0, F.pass_layers};
    if (r.rect.extent.width && r.rect.extent.height) vkrec_ClearAttachments(1, &a, 1, &r);
    stamp_image(F.depth.image);
    g_stats.clears++;
}

// ---------- the records
// dst is a table of ST_N rows of `slots` entries (S.tex, S.img, S.ubo, S.ssbo), as its first entry.
static void bind_handles(uint64_t *dst, int slots, const p_bind_handles *b) {
    if (b->stage < 0 || b->stage >= ST_N) stop("handles bound to stage %d", b->stage);
    if (b->n < 0 || b->n > 32) stop("%d handles bound to stage %d", b->n, b->stage);
    for (int i = 0; i < b->n; i++) {
        int k = b->first + i;
        if (k < 0 || k >= slots) stop("a handle bound at slot %d of stage %d", k, b->stage);
        dst[b->stage * slots + k] = b->h[i];
    }
}

static void bind_buffer(VknBuf *dst, int slots, const p_bind_buf *b, const char *what) {
    if (b->stage < 0 || b->stage >= ST_N || b->index < 0 || b->index >= slots)
        stop("a %s bound at index %d of stage %d", what, b->index, b->stage);
    dst[b->stage * slots + b->index] = (VknBuf){b->addr, b->size};
}

static void bind_state(const p_bind_state *s) {
    switch (s->kind) {
    case 0: {
        blend_st b;
        memcpy(&b, s->bytes, sizeof b);
        S.blend[b.target] = b, S.has_blend[b.target] = 1;   // a 3-bit target: 0 to 7
        break;
    }
    case 1: memcpy(&S.color, s->bytes, sizeof S.color); break;
    case 2: memcpy(&S.chmask, s->bytes, sizeof S.chmask); break;
    case 3: break;   // multisample state: the sample count is the targets', checked at the pass
    case 4: memcpy(&S.poly, s->bytes, sizeof S.poly), S.has_poly = 1; break;
    case 5: memcpy(&S.ds, s->bytes, sizeof S.ds), S.has_ds = 1; break;
    case 6: break;   // point size: written by the programs that draw points
    case 7: break;   // alpha reference: the alpha test is a draw variant of the translator
    case 8: memcpy(&S.line_width, s->bytes, sizeof S.line_width); break;
    default: stop("state object of kind %d", s->kind);
    }
}

// The game's variable rate shading (a shading rate image read through its palettes, and the palette mode) is replayed
// at full rate: every pixel runs its fragment shader, so the picture is as fine as the console's, and finer on the
// draws the console shades coarser, for the GPU cost of those extra fragments. Logged once and counted
// (VkrStats.full_rate) so a capture says where it differs from the console's coarser shading.
static void full_rate(const char *what) {
    static int logged;
    if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
        rt_log("gfx: vulkan: the game set %s: replayed at full rate, one fragment per pixel (logged once)", what);
    g_stats.full_rate++;
}

// The geometry processing mode (NVN2, nvnCommandBufferSetGeometryProcessingMode) is one of the modes 0 to 2. It selects
// how the console's front end runs the geometry stages of the draws that follow (the path the work takes), not what the
// draws produce: a mesh draw declares its primitive mode in its own outputs and a classic draw takes its topology from the
// draw. Vulkan has no state that matches, so nothing is set and each draw is made as it would be without the call; the mode
// is only logged, once per value, so that a capture shows which ones the game used. A value outside 0 to 2 is one whose
// meaning this code does not know, and stops by name.
static void geometry_mode(int32_t mode) {
    static int logged[3];
    if (mode < 0 || mode > 2) stop("GEOMETRY_MODE %d is not replayed: only the modes 0 to 2 are", (int)mode);
    if (!__atomic_exchange_n(&logged[mode], 1, __ATOMIC_RELAXED))
        rt_log("gfx: vulkan: the game set geometry processing mode %d: nothing to set in Vulkan, the draws are made the same "
               "(logged once per mode)", (int)mode);
}

static void record_op(const op_hdr *r);

// One record. Under collection a draw is replayed with a way out (a dispatch passes through the same wrapper, but no
// stop_answer call is made in one): a stop on an answer the ledger can fix (stop_answer) lands here after the draw's
// rows were made, and the next record is replayed as if this one were not.
static void record(const op_hdr *r) {
    if (!makes_rows(r->op) || !collecting()) {
        record_op(r);
        return;
    }
    jmp_buf here;
    g_skip = &here;
    if (!setjmp(here)) record_op(r);
    g_skip = NULL;
}

static void record_op(const op_hdr *r) {
    const void *a = r + 1;
    g_stats.records++;
    F.cur = r;
    switch (r->op) {
    case OP_BEGIN: case OP_END: break;
    case OP_SET_RT: {
        pass_end();
        memset(&S.rt, 0, sizeof S.rt);
        memcpy(&S.rt, a, r->size - sizeof *r < sizeof S.rt ? r->size - sizeof *r : sizeof S.rt);
        break;
    }
    case OP_CLEAR_COLOR: op_clear_color(a); break;
    case OP_CLEAR_DS: op_clear_ds(a); break;
    case OP_VIEWPORTS: {
        const p_viewports *v = a;
        if (v->first < 0 || v->n < 0 || v->first + v->n > 16) stop("viewports %d to %d", v->first, v->first + v->n);
        memcpy(S.vp[v->first], v->v, sizeof v->v[0] * (size_t)v->n);
        if (v->first + v->n > S.nvp) S.nvp = v->first + v->n;
        break;
    }
    case OP_SCISSOR: {
        const p_scissor *v = a;
        if (v->first < 0 || v->n < 0 || v->first + v->n > 16) stop("scissors %d to %d", v->first, v->first + v->n);
        memcpy(S.sc[v->first], v->r, sizeof v->r[0] * (size_t)v->n);
        if (v->first + v->n > S.nsc) S.nsc = v->first + v->n;
        S.has_sc = 1;
        break;
    }
    case OP_DEPTH_RANGE: {
        const p_depth_range *v = a;
        if (v->first < 0 || v->n < 0 || v->first + v->n > 16) stop("depth ranges %d to %d", v->first, v->first + v->n);
        memcpy(S.dr[v->first], v->r, sizeof v->r[0] * (size_t)v->n);
        break;
    }
    case OP_BIND_PROGRAM: bind_program(a); break;
    case OP_BIND_STATE: bind_state(a); break;
    case OP_BIND_VATTR: {
        const p_bind_vattr *v = a;
        for (int i = 0; i < v->n; i++)
            if (v->first + i >= 0 && v->first + i < NVN_VERTEX_BINDINGS) S.vattr[v->first + i] = v->st[i];
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *v = a;
        for (int i = 0; i < v->n; i++)
            if (v->first + i >= 0 && v->first + i < NVN_VERTEX_BINDINGS) S.vstream[v->first + i] = v->st[i];
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *v = a;
        for (int i = 0; i < v->n; i++)
            if (v->first + i >= 0 && v->first + i < NVN_VERTEX_BINDINGS) S.vbuf[v->first + i] = v->r[i];
        break;
    }
    case OP_BIND_UBO: bind_buffer(&S.ubo[0][0], UBO_SLOTS, a, "uniform buffer"); break;
    case OP_BIND_SSBO: bind_buffer(&S.ssbo[0][0], SSBO_SLOTS, a, "storage buffer"); break;
    case OP_BIND_TEX: bind_handles(&S.tex[0][0], TEX_SLOTS, a); break;
    case OP_BIND_IMG: bind_handles(&S.img[0][0], IMG_SLOTS, a); break;
    case OP_BIND_SAMP: {
        const p_bind_handles *b = a;
        if (b->n) stop("separate samplers bound (stage %d)", b->stage);
        break;
    }
    case OP_STENCIL: {
        const p_stencil *s = a;
        for (int f = 0; f < 2; f++) {
            if (!(s->face & (1 << f))) continue;   // NVNface: front 1, back 2, both 3
            if (s->which == 0) S.st_vmask[f] = (uint32_t)s->value_mask;
            else if (s->which == 1) S.st_wmask[f] = (uint32_t)s->mask;
            else S.st_ref[f] = (uint32_t)s->ref;
        }
        break;
    }
    case OP_POLY_OFFSET: {
        const p_poly_offset *p = a;
        S.bias[0] = p->factor, S.bias[1] = p->units, S.bias[2] = p->clamp;
        break;
    }
    case OP_DRAW: op_draw(a); break;
    case OP_DRAW_IDX: op_draw_idx(a); break;
    case OP_DRAW_IDX_IND: op_draw_idx_ind(a); break;
    case OP_DRAW_MESH: op_draw_mesh(a, 0); break;
    case OP_DRAW_MESH_IND: op_draw_mesh(NULL, ((const p_draw_mesh_ind *)a)->ind); break;
    case OP_DISPATCH: {
        const p_dispatch *d = a;
        const uint32_t g[3] = {(uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z};
        dispatch(g, 0);
        break;
    }
    case OP_DISPATCH_IND: op_dispatch_ind(a); break;
    case OP_BARRIER:
        pass_end(), full_barrier();
        if (vkgt_enabled) vkgt_mark(VKGT_BARRIER, "barrier", 0, 0, 0);
        break;
    case OP_SET_TEX_POOL: {
        const p_set_pool *p = a;
        if (F.set0 && S.has_tex_pool && p->pool.addr != S.tex_pool.addr)
            stop("a second texture pool in one submission (the copy of set 0 shows one)");
        S.tex_pool = p->pool, S.has_tex_pool = 1;
        break;
    }
    case OP_SET_SAMP_POOL: {
        const p_set_pool *p = a;
        if (F.set0 && S.has_samp_pool && p->pool.addr != S.samp_pool.addr)
            stop("a second sampler pool in one submission (the copy of set 0 shows one)");
        S.samp_pool = p->pool, S.has_samp_pool = 1;
        break;
    }
    case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, sizeof S.blend_color); break;
    case OP_DEPTH_CLAMP: S.depth_clamp = ((const p_flag *)a)->value != 0; break;
    case OP_RENDER_ENABLE: S.render = ((const p_flag *)a)->value != 0; break;
    case OP_RENDER_COND: S.render = vkr_condition(a); break;
    case OP_RASTER_DISCARD: S.raster_discard = ((const p_flag *)a)->value != 0; break;
    case OP_PRIM_RESTART: {
        const p_prim_restart *p = a;
        S.restart_en = p->enable != 0, S.restart_index = p->index;
        break;
    }
    case OP_PUSH_CONST: {
        const p_push_const *p = a;
        if (p->table < 0 || p->table >= ROOT_TABLES || p->offset < 0 || p->offset + p->bytes > ROOT_WORDS * 4)
            stop("push constants of %d bytes at %d in table %d", p->bytes, p->offset, p->table);
        memcpy((uint8_t *)S.root[p->compute ? 1 : 0][p->table] + p->offset, p->w, (size_t)p->bytes);
        break;
    }
    case OP_WAIT_SYNC: break;   // one queue in submission order: what an earlier submission signalled is done before
    case OP_SAMPLE_MASK: S.sample_mask = (uint32_t)((const p_flag *)a)->value; break;
    case OP_DEPTH_BOUNDS: {
        const p_depth_bounds *p = a;
        S.bounds_en = p->enable != 0, S.bounds[0] = p->min, S.bounds[1] = p->max;
        break;
    }
    case OP_CONSERVATIVE: {
        const p_conservative *p = a;
        if (p->which == 0) S.conservative = p->value != 0;
        break;
    }
    case OP_GEOMETRY_MODE: geometry_mode(((const p_flag *)a)->value); break;
    case OP_SHADING_RATE_IMAGE:
        if (((const p_shading_rate_image *)a)->tex.present) full_rate("a shading rate image");
        break;
    case OP_SHADING_RATE_MODE:
        if (((const p_flag *)a)->value) full_rate("a shading rate mode");
        break;
    case OP_SHADING_RATE_PALETTE: break;   // read only through a shading rate image and mode, at full rate above
    case OP_SCRATCH: break;   // the driver's local memory for spilled registers: the device's own on Vulkan
    case OP_HINT:   // scheduling and cache hints change nothing the GPU computes (nvn_obj.h), but the compute overlap
                    // mode, which acts as a full barrier (architecture: the semantics that must survive)
        if (((const p_hint *)a)->what == NVN_HINT_COMPUTE_OVERLAP) pass_end(), full_barrier();
        break;
    default:
        if (vkr_xfer_op(r)) {
            if (vkgt_enabled && r->op != OP_COPY_STRIDE && r->op != OP_COPY_IMAGE_STRIDE)
                vkgt_mark(VKGT_XFER, nvn_op_name(r->op), 0, 0, 0);
        } else if (!vkr_report_op(r)) {
            stop_op(r->op);
        }
    }
    F.cur = NULL;
    ledger_track_op(&S.track, r);
}

static void walk(int index, uint64_t handle) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)handle;
    if (!p) stop("recording %d is a null handle", index);
    for (size_t at = 0;;) {
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < sizeof h || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL ||
            (at == 0) != (h.op == OP_BEGIN))
            stop("recording %d is not whole at +%zu (op %u, %u bytes)", index, at, h.op, h.size);
        record((const op_hdr *)(const void *)(p + at));
        at += h.size;
        if (h.op == OP_END) return;
    }
}

static PlatMutex g_rec_lock = PLAT_MUTEX_INIT;

uint64_t vkr_submit(const uint64_t *handles, int n) {
    // The recorder opens with the first replayed submission (a test that opened its own keeps it), so that a device
    // opened only for the back end's own work never starts its thread.
    if (!vkrec_is_open()) {
        plat_mutex_lock(&g_rec_lock);
        if (!vkrec_is_open()) vkrec_open(NULL);
        plat_mutex_unlock(&g_rec_lock);
    }
    const VkReplayOps *ops = ops_now();
    if (ops->frame_begin) ops->frame_begin();
    uint64_t value = vkrec_begin(GPU_VK_REPLAY);
    memset(&F, 0, sizeof F);
    F.ops = ops, F.value = value;
    if (vkgt_enabled) vkgt_frame_begin(value);
    static uint64_t frames;
    int k = (int)(frames++ % FRAMES_KEPT);
    W[k].value = value, W[k].n = 0;
    F.wslot = k;
    vkr_report_frame_begin();
    full_barrier();   // the frame's work after every earlier submission's (D18)
    for (int i = 0; i < n; i++) walk(i, handles[i]);
    pass_end();
    movie_unclosed();
    if (ops->frame_end) ops->frame_end(value);
    vkr_report_frame_end();   // after the frame's end: its events and reports follow the backing map's write-backs
    g_stats.frames++;
    if (vkgt_enabled) vkgt_frame_end();   // the last timestamps, and the effect that reads them
    return vkrec_end(NULL, 0);
}
