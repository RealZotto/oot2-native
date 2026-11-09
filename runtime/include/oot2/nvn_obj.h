// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// NVN host-side object layouts and the command record format (architecture D10, D11): what the NVN2 front end
// (runtime/nvn) writes and the back ends behind gfx.h read. Objects stay in place in the NVN object memory the game
// gives them, never larger than the size nvn.h reserves for them (FITS). State objects are small structs the game copies
// by value, packed in place with bit-fields.
//
// Taken from oot-pc-recomp runtime/nvn_obj.h at 8e09aaa (GPL-3.0-or-later), which carries, through DKB Recomp, TotK
// Recomp's port (MIT, Copyright (c) 2026 Valiantsin Dzerakh; licenses/totk-MIT.txt) of MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_obj.h and the two types of nvn_tex.h at b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt). Changed here: this comment; pool_t's host field is vmap,
// a virtual pool's mapping state, and its fields are described; nvn_mapping_t, the layout of NVNmappingRequest, comes
// from oot-pc-recomp runtime/nvn_api.h at the same commit; nvn_texdesc keeps a texture's depth-stencil mode in a bit of
// its padding (nxnative keeps it too); prog_t keeps each stage's code address and identity instead of the game's
// NVNshaderData (architecture D16); blend_st takes the advanced blend fields of nxnative dm/gfx/nvn/nvn_obj.h at
// 865cc35 (MIT, licenses/nxnative-MIT.txt) and the normalized destination, ms_st the coverage truncation of NVN2;
// cmd_t is the NVN2 driver's command buffer, which the game's inline code writes; the command records take textures
// by value, programs by serial and stage records, and 32 vertex bindings, add the payloads oot-pc-recomp keeps in its
// runtime/nvn_api.h and those of the NVN2 calls OoT makes; queue_t keeps its newest fence value, win_t the fence value
// each texture's acquire sync waits for, and a WaitEvent record its call site. The
// other layouts are unchanged; the notes marked TOTK record
// what that port found in TotK, not in this game, and hle_nvn.c and nvn_mtl.m name the MacNativeGames files the
// layouts came from.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "nvn.h"

#define MAGIC(a, b, c, d) ((uint32_t)(a) | (uint32_t)(b) << 8 | (uint32_t)(c) << 16 | (uint32_t)(d) << 24)
#define FITS(T, H) _Static_assert(sizeof(T) <= sizeof(H), #T " exceeds the reserved size of " #H)

// ---------- Texture description and storage layout (nvn_tex.h; the functions over them live in nvn_tex.c) ----------
typedef struct {
    int target, format, flags, samples;   // NVNtextureTarget / NVNformat / NVNtextureFlags / sample count (0 or 1 = single-sampled)
    uint32_t width, height, depth, levels, stride;
    signed int pkg_block_h:8; unsigned int swizzle_r:3,swizzle_g:3,swizzle_b:3,swizzle_a:3,dsmode:1,swizzle_pad:11;             // Byte 0 of the packaged texture layout parameters + 1 (low nibble log2 block height, high nibble log2 block depth); 0 = use the automatic algorithm. dsmode: NVNtextureDepthStencilMode (1 = the stencil plane)
} nvn_texdesc;

typedef struct {
    uint64_t size, layer_size;            // Total storage bytes, bytes per layer
    uint32_t align, layers, bpb;          // Alignment, layer count, bytes per block
    unsigned tileH, tileD;                // Block height / block depth (log2 of the GOB count)
} nvn_texlayout;

typedef struct { uint32_t flags; } dev_builder_t;                       FITS(dev_builder_t, NVNdeviceBuilder);
typedef struct { uint32_t magic, flags, origin, depth_mode; void *dbg_cb, *dbg_data; } dev_t_;  FITS(dev_t_, NVNdevice);

typedef struct { NVNdevice *dev; uint32_t flags; uint64_t cmd_sz, comp_sz, ctl_sz, flush; void *mem; uint64_t mem_sz; } queue_builder_t;
FITS(queue_builder_t, NVNqueueBuilder);
// last: the fence value of the queue's newest submission (0: none yet), what QueueFenceSync waits for; serial: this queue's
// number, from 1 in the order the game initialised queues, what the back end knows the queue by (gfx.h, WaitEvent)
typedef struct { uint32_t magic; NVNdevice *dev; uint64_t submits, last, serial; } queue_t;   FITS(queue_t, NVNqueue);

typedef struct { NVNdevice *dev; void *mem; uint64_t size; uint32_t flags; } pool_builder_t;  FITS(pool_builder_t, NVNmemoryPoolBuilder);
// A memory pool (runtime/nvn/nvn_pool.c, architecture D10, D14). mem is its GPU address: the game's storage when the
// CPU can see the pool, else a range of the GPU address window (gfx.h), which nothing maps on the host. game_mem is the
// storage the game gave (NULL for a virtual pool), span the window bytes the pool holds (0 for none), vmap a virtual
// pool's mapping state in host memory (NULL for every other pool).
struct nvn_vmap;
typedef struct { uint32_t magic, flags; uint8_t *mem; uint64_t size; uint8_t *game_mem; struct nvn_vmap *vmap; uint64_t span; } pool_t;
FITS(pool_t, NVNmemoryPool);
// NVNmappingRequest as the game passes MemoryPoolMapVirtual an array of them, 0x28 bytes each.
typedef struct {
    const NVNmemoryPool *physical_pool;   // NULL unmaps
    int64_t physical_offset, virtual_offset;
    uint64_t size;
    int32_t storage_class;
    uint32_t padding;
} nvn_mapping_t;
_Static_assert(sizeof(nvn_mapping_t) == 0x28, "NVNmappingRequest stride");

typedef struct { const pool_t *pool; int64_t off; int32_t n; } dpool_t;                       // Texture / sampler descriptor pool
FITS(dpool_t, NVNtexturePool); FITS(dpool_t, NVNsamplerPool);

// Descriptor table addressing captured by a command, independent of the descriptor-pool object's later lifetime.
typedef struct { uint64_t addr; uint32_t count, pad; } dpool_ref;
#define NVN_DESCRIPTOR_BYTES 32u   // DeviceGetInteger: texture and sampler descriptor sizes
static inline int nvn_dpool_snapshot(const dpool_t *p, uint32_t limit, dpool_ref *out) {
    memset(out, 0, sizeof *out);
    if (!p) return 1;
    if (!p->pool || p->pool->magic != MAGIC('P', 'O', 'O', 'L') || !p->pool->mem || p->off < 0 ||
        p->n <= 0 || (uint32_t)p->n > limit) return 0;
    uint64_t base = (uintptr_t)p->pool->mem, off = (uint64_t)p->off;
    uint64_t bytes = (uint64_t)(uint32_t)p->n * NVN_DESCRIPTOR_BYTES;
    if (off % NVN_DESCRIPTOR_BYTES || off > p->pool->size || bytes > p->pool->size - off ||
        base > UINT64_MAX - off || base + off > UINT64_MAX - bytes || (base + off) % NVN_DESCRIPTOR_BYTES) return 0;
    *out = (dpool_ref){base + off, (uint32_t)p->n, 0};
    return 1;
}

typedef struct {
    NVNdevice *dev; nvn_texdesc d; pool_t *pool; int64_t off;
    uint8_t has_layout, layout[8];
    const void *pkg_data;                // TOTK: SetPackagedTextureData source, copied into storage at Initialize
} tex_builder_t;                                                                             FITS(tex_builder_t, NVNtextureBuilder);
typedef struct {
    uint32_t magic; nvn_texdesc d; uint8_t has_layout, layout[8];
    nvn_texlayout L; uint8_t *mem; pool_t *pool; int64_t off;
    uint64_t serial;                     // Initialization serial: the back end uses it to find its texture in the host table (the game may copy the object by value, so object pointers cannot be stored)
} tex_t;                                                                                     FITS(tex_t, NVNtexture);
// TOTK: the game reserves 16 bytes for NVNtextureView on its stack (SetDefaults at sp+0x30, its own
// data at sp+0x40), so the view is packed into 16 bytes. swz = NVN view swizzle selectors. dsmode:
// NVNtextureDepthStencilMode (nxnative nvn_obj.h:42 packs it the same way).
typedef struct {
    uint8_t set_levels : 1, set_layers : 1, set_format : 1, set_target : 1, set_swizzle : 1, set_dsmode : 1, dsmode : 2;
    uint8_t base_level, nlevels, target; uint16_t min_layer, nlayers; uint16_t format; uint8_t swz[4]; uint8_t pad1[2];
} view_t;
_Static_assert(sizeof(view_t) == 16, "NVNtextureView is 16 bytes in TOTK's NVN");
FITS(view_t, NVNtextureView);

// Sampler parameters (one copy shared by builder and object)
typedef struct { uint8_t minf, magf, ws, wt, wr, cmp_mode, cmp_func, reduction; float lod_min, lod_max, lod_bias, aniso, border[4], lod_snap; } samp_par;   // TOTK: lod_snap stored for queries (Metal has no equivalent)   // Enum values are all < 256
typedef struct { NVNdevice *dev; samp_par p; } samp_builder_t;                                FITS(samp_builder_t, NVNsamplerBuilder);
typedef struct { uint32_t magic; samp_par p; } samp_t;                                        FITS(samp_t, NVNsampler);
_Static_assert(sizeof(samp_par) == 44, "TOTK sampler parameters");

// A program (runtime/nvn/nvn_program.c, architecture D10, D16): per stage, the GPU address of its code area and its
// identity (nvn_program.h), both taken at SetShaders, since the game may free the control section once that returns.
// serial: what a back end keys a program's pipelines on, new at each SetShaders.
struct nvn_stage;
typedef struct { uint64_t code; const struct nvn_stage *id; } prog_stage_t;
typedef struct {
    uint32_t magic, nstage;
    prog_stage_t st[6];
    uint64_t serial;
} prog_t;                                                                                FITS(prog_t, NVNprogram);
_Static_assert(sizeof(prog_t) <= 0xa0, "NVN2 program allocation stride");

// A sync (runtime/nvn/nvn_queue.c): fenced 0 until a queue (1) or a window's acquire (2) gives it fence value `fence`,
// done once the back end's fence_done reaches it. A window's acquire (default placement, oot2/nvn_pace.h) also gives it
// `ready`, a count of the system counter from the virtual clock's origin (oot2/vsync.h): the sync is done once the clock
// has reached it as well (0: none). It is the due count of a tick of the 60 Hz clock, or of a slot of the fixed rate's
// schedule when the game runs at one (the fixed frame rate gate)
typedef struct { uint32_t magic, fenced; uint64_t fence, ready; } sync_t;                            FITS(sync_t, NVNsync);
typedef struct { const pool_t *pool; int64_t off; } ev_builder_t;                             FITS(ev_builder_t, NVNeventBuilder);
typedef struct { uint32_t magic; volatile uint32_t *val; } ev_t;                              FITS(ev_t, NVNevent);

typedef struct { NVNdevice *dev; void *native; int32_t ntex, interval, active; NVNtexture *tex[4]; } win_builder_t;
FITS(win_builder_t, NVNwindowBuilder);
// pace: the due count of the window's last present, in system-counter counts from the virtual clock's origin (the due
// count of a 60 Hz tick, or of a slot of the fixed rate's schedule; oot2/nvn_pace.h; read once presents is not 0);
// avail[i] is the donors' pacing time and stays unused: the acquire gate counts ticks or slots
// fence[i]: the fence value the queue had reached when texture i was last presented (0: never), what its acquire sync
// waits for (runtime/nvn/nvn_window.c)
// target: the count the newest acquire since the last present made its sync ready at (0: none), where the present's due
// count may not be earlier (oot2/nvn_pace.h); written and read under the pace lock
typedef struct { uint32_t magic; void *native; int32_t ntex, interval, next, active, crop[4]; NVNtexture *tex[4]; uint64_t presents, avail[4], pace, ticket[4], fence[4], target; } win_t;   // ticket[i]: ticket of the most recent present of image i
FITS(win_t, NVNwindow);

// A command buffer (runtime/nvn/nvn_cmd.c, architecture D11). The game writes this object itself: NVN2's
// BeginRecording fast path, inlined at twelve sites of main, reads the command write pointer (+0x00) and the control
// write pointer and its end (+0x28, +0x30), writes the recording's start (+0x48, +0x50, +0x58) and flags (+0x11,
// +0x12), puts a 16-byte header at the control write pointer and moves it past, and calls BeginRecording only when
// fewer than 16 bytes of control memory are left. So the fields up to +0x87 keep the offsets and the meaning of the
// console driver's object (NVN 22.2.8, compiled into the game's sdk module, read in its CommandBuffer entry points),
// and the front end's state hangs off +0x88. No command is ever written into command memory: the command write
// pointer stays ~0 and its end 0, where no inline fast path finds room, and the front end stops by name if either
// moves. Control memory is accounted the way the driver fills it, but the front end writes nothing there.
struct nvn_cmd_host;
typedef struct {
    uint64_t cmd_cur, cmd_end;           // +0x00, +0x08: command write pointer and the end of its fast path (~0, 0)
    uint8_t flags10;
    uint8_t recording;                   // +0x11: a recording is open
    uint8_t ctl_moved;                   // +0x12: the open recording went on in other control memory, or ended empty
    uint8_t flags13;                     // +0x13: bit 0, the last recording ended empty
    uint8_t cmd_cb_on, ctl_cb_on;        // +0x14, +0x15: the command and control memory callbacks are enabled
    uint16_t pad16;
    int32_t rec_mode;                    // +0x18: SetRecordingMode
    uint32_t pad1c;
    uint64_t cmd_limit;                  // +0x20: the command memory's end (0: none for inline writers)
    uint8_t *ctl_cur, *ctl_end;          // +0x28, +0x30: control write pointer, and the end less the 16 bytes a jump takes
    uint64_t cmd_start;                  // +0x38
    uint8_t *ctl_start;                  // +0x40: the control memory added last
    uint64_t rec_cmd;                    // +0x48: command write pointer at the recording's start
    uint8_t *rec_ctl;                    // +0x50: the recording's header in control memory; outside one, where the next goes
    uint64_t last_entry;                 // +0x58: the control entry a new command segment may extend (0: none)
    uint64_t cmd_gpu;                    // +0x60
    uint32_t cmd_words, pad6c;           // +0x68
    NVNdevice *dev;                      // +0x70
    PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC cb; void *cb_data;   // +0x78, +0x80
    struct nvn_cmd_host *host;           // +0x88: the front end's state, NULL once finalized
    uint32_t magic, pad94;               // +0x90
} cmd_t;                                                                                     FITS(cmd_t, NVNcommandBuffer);
_Static_assert(__builtin_offsetof(cmd_t, recording) == 0x11 && __builtin_offsetof(cmd_t, ctl_cur) == 0x28 &&
               __builtin_offsetof(cmd_t, rec_cmd) == 0x48 && __builtin_offsetof(cmd_t, rec_ctl) == 0x50 &&
               __builtin_offsetof(cmd_t, last_entry) == 0x58 && __builtin_offsetof(cmd_t, dev) == 0x70 &&
               __builtin_offsetof(cmd_t, cb) == 0x78 && __builtin_offsetof(cmd_t, host) == 0x88,
               "the NVN2 driver's command buffer fields");

// ---------- State objects (defaults follow the NVN / GL defaults) ----------
typedef struct { uint32_t blend_en : 8, logic_op : 4, alpha_func : 4; } color_st;            FITS(color_st, NVNcolorState);
typedef struct { uint32_t mask; } chmask_st;            /* 4 bits per target: r=1 g=2 b=4 a=8, bit set = write */ FITS(chmask_st, NVNchannelMaskState);
// adv: NVNblendAdvancedMode; adv_npm and adv_nnd: the advanced premultiplied source and normalized destination,
// stored inverted so that SetDefaults' zero is NVN_TRUE; adv_ovl: NVNblendAdvancedOverlap.
typedef struct {
    uint64_t target : 3, src_rgb : 7, dst_rgb : 7, src_a : 7, dst_a : 7, eq_rgb : 3, eq_a : 3, adv : 6, adv_npm : 1,
             adv_ovl : 2, adv_nnd : 1;
} blend_st;
FITS(blend_st, NVNblendState);
typedef struct { uint8_t enable, samples, a2c, a2c_dither, raster_samples, cov_mod, cov2color_en, cov2color_out, sl_en, sl_grid, cov_trunc; } ms_st;
FITS(ms_st, NVNmultisampleState);
typedef struct { uint32_t cull : 2, front : 1, mode : 2, offset_en : 3; } poly_st;           FITS(poly_st, NVNpolygonState);
typedef struct {
    uint64_t dt : 1, dw : 1, df : 4, st : 1, sf_f : 4, sf_b : 4,
             fail_f : 4, zfail_f : 4, zpass_f : 4, fail_b : 4, zfail_b : 4, zpass_b : 4;
} ds_st;                                                                                     FITS(ds_st, NVNdepthStencilState);
typedef struct { uint32_t fmt : 8, offset : 19, stream : 5; } va_st;                         FITS(va_st, NVNvertexAttribState);
typedef struct { uint32_t stride, divisor; } vs_st;                                          FITS(vs_st, NVNvertexStreamState);

// ---------- Command records: {op, size (including header, aligned to 8)} + payload ----------
// A command handle (EndRecording's answer, what QueueSubmitCommands receives) points at a host copy of one recording:
// records from OP_BEGIN to OP_END, with every called or copied recording inlined, so a copy holds no OP_CALL or
// OP_JUMP. What the game may change after a call is captured when it records: state objects and textures by value,
// programs by serial and stage records, uniform data and push constants by value. Deferred binds keep the game's
// array (src) and take its contents at submit (nvn_submit_recordings, gfx.h). A record of an array (RECN in
// runtime/nvn) holds only the n elements the call gave: a reader reads no further.
typedef struct { uint32_t op, size; } op_hdr;
enum {
    OP_END, OP_JUMP, OP_BEGIN, OP_CALL,
    OP_SIGNAL_EVENT, OP_REPORT_COUNTER, OP_COPY_B2B, OP_UPDATE_UB,                 // CPU-visible side effects
    OP_SET_RT, OP_CLEAR_COLOR, OP_CLEAR_DS, OP_VIEWPORTS, OP_SCISSOR, OP_DEPTH_RANGE,
    OP_BIND_PROGRAM, OP_BIND_STATE, OP_BIND_VATTR, OP_BIND_VSTREAM, OP_BIND_VBUF,
    OP_BIND_UBO, OP_BIND_SSBO, OP_BIND_TEX, OP_BIND_SAMP,
    OP_STENCIL, OP_POLY_OFFSET, OP_DRAW, OP_DRAW_IDX, OP_DISPATCH, OP_BARRIER,
    OP_SET_TEX_POOL, OP_SET_SAMP_POOL, OP_COPY_B2T, OP_COPY_STRIDE, OP_BLEND_COLOR, OP_COPY_T2B, OP_BIND_IMG, OP_COPY_T2T, OP_RESET_COUNTER, OP_CLEAR_TEX, OP_DEPTH_CLAMP, OP_RENDER_ENABLE, OP_RASTER_DISCARD, OP_RENDER_COND, OP_PRIM_RESTART, OP_DRAW_IDX_IND, OP_DISPATCH_IND, OP_CLEAR_BUF, OP_PUSH_CONST, OP_WAIT_SYNC, OP_COPY_IMAGE_STRIDE,
    // NVN2 and the rest of what OoT records
    OP_DRAW_MESH, OP_DRAW_MESH_IND, OP_GEOMETRY_MODE, OP_WAIT_EVENT, OP_SAMPLE_MASK, OP_DEPTH_BOUNDS, OP_CONSERVATIVE,
    OP_SHADING_RATE_IMAGE, OP_SHADING_RATE_PALETTE, OP_SHADING_RATE_MODE, OP_SCRATCH, OP_BIND_ACCEL, OP_HINT,
    OP_COUNT_
};
// A texture as a record holds it: the object's bytes when the call was made (the game initializes textures again in
// place, and may finalize one before the submit) and the view the call gave. present 0: no texture.
typedef struct { tex_t tex; view_t view; uint8_t present, has_view, pad[6]; } nvn_tex_rec;
// Payload (immediately follows op_hdr)
typedef struct { int32_t n, pad; nvn_tex_rec depth; nvn_tex_rec color[8]; } p_set_rt;   // color[n] and on not recorded
typedef struct { int32_t index, mask; float rgba[4]; } p_clear_color;
typedef struct { float depth; int32_t depth_mask, stencil, stencil_mask; } p_clear_ds;
typedef struct { int32_t first, n; float v[16][4]; } p_viewports;         // x, y, width, height of viewports first..
typedef struct { int32_t first, n; int32_t r[16][4]; } p_scissor;          // x, y, width, height of scissors first..
typedef struct { int32_t first, n; float r[16][2]; } p_depth_range;        // near, far of viewports first..
typedef struct { int32_t value; } p_flag;   // one value of render state: DEPTH_CLAMP, RENDER_ENABLE, RASTER_DISCARD, GEOMETRY_MODE, SAMPLE_MASK, SHADING_RATE_MODE
typedef struct { int32_t mode; int32_t pad; uint64_t addr; } p_render_cond;   // TOTK: SetRenderEnableConditional: mode 0 = render if the two NVNcounterData.counter values at addr / addr+16 are equal, 1 = if not equal
typedef struct { int32_t enable; uint32_t index; } p_prim_restart;   // SetPrimitiveRestart: indexed strips restart at index
// BindProgram: the program as it was then. serial keys its pipelines (gfx.h program_release); id[i] are the stages'
// immortal records (nvn_program.h), code[i] their code areas' GPU addresses; 0 stages unbinds.
typedef struct { uint64_t serial; int32_t stages, nstage; uint64_t code[6]; const struct nvn_stage *id[6]; } p_bind_program;
typedef struct { int32_t kind; uint8_t bytes[0x18]; } p_bind_state;      // kind: 0 blend 1 color 2 chmask 3 ms 4 poly 5 ds 6 point size (float) 7 alpha ref (float) 8 line width (float)
// Vertex attributes, streams and buffers: 32 each, the vertex attributes and buffer bindings nvnDeviceGetInteger answers
#define NVN_VERTEX_BINDINGS 32
typedef struct { int32_t first, n; uint32_t st[NVN_VERTEX_BINDINGS]; } p_bind_vattr;
typedef struct { int32_t first, n; vs_st st[NVN_VERTEX_BINDINGS]; } p_bind_vstream;
typedef struct { int32_t first, n; NVNbufferRange r[NVN_VERTEX_BINDINGS]; } p_bind_vbuf;
typedef struct { int32_t stage, index; uint64_t addr, size; } p_bind_buf;  // UBO / SSBO
// Texture, image, separate texture and sampler handles (nvn_tex.h unpacks them). src: a *Deferred bind's array in
// game memory, read again at submit into h; 0 for a bind whose handles were copied when it recorded.
typedef struct { int32_t stage, first, n, pad_; uint64_t src; uint64_t h[32]; } p_bind_handles;
typedef struct { int32_t face, value_mask, mask, ref, which; } p_stencil;  // which: 0 value mask 1 write mask 2 reference value
typedef struct { float factor, units, clamp; } p_poly_offset;
typedef struct { int32_t prim, first, count, base_instance, instances; } p_draw;   // TOTK: instances 0 = one instance (DrawArrays)
typedef struct { int32_t prim, itype, count, base_vertex, base_instance, instances; uint64_t ibuf; } p_draw_idx;
typedef struct { int32_t x, y, z; uint64_t caller; } p_dispatch;   // caller: return address of the game call site (to locate the dispatcher)
typedef struct { dpool_ref pool; } p_set_pool;
// Buffer to texture, and texture to buffer (OP_COPY_T2B: addr is the destination)
typedef struct { uint64_t addr; nvn_tex_rec tex; NVNcopyRegion r; int32_t flags; } p_copy_b2t;
typedef struct { nvn_tex_rec src, dst; NVNcopyRegion sr, dr; int32_t flags; } p_copy_t2t;
typedef struct { nvn_tex_rec tex; NVNcopyRegion r; int32_t mask; uint8_t is_uint; union { float f[4]; uint32_t u[4]; } c; } p_clear_tex;
typedef struct { int64_t stride; } p_copy_stride;
typedef struct { float rgba[4]; } p_blend_color;
// The payloads oot-pc-recomp keeps in runtime/nvn_api.h at 8e09aaa, with the event's location and flags kept
typedef struct { volatile uint32_t *val; uint32_t mode, value, location, flags; } p_signal_event;   // mode: NVNeventSignalMode
typedef struct { volatile uint32_t *val; uint32_t mode, value; uint64_t caller; } p_wait_event;   // mode: NVNeventWaitMode; caller: the game's return address
typedef struct { uint64_t type, addr; } p_report_counter;  // NVNcounterType, NVNcounterData address
typedef struct { uint64_t type; } p_reset_counter;
typedef struct { uint64_t src, dst, size; int32_t flags, pad; } p_copy_b2b;   // GPU addresses
typedef struct { uint64_t dst, size; uint8_t data[]; } p_update_ub;   // size bytes written at dst, copied when it recorded
typedef struct { int32_t bits, pad; } p_barrier;           // NVNbarrierBits
typedef struct { const sync_t *sync; } p_wait_sync;
// DrawElementsIndirect: ind = GPU address of {count, instances, first index, base vertex, base instance}
typedef struct { int32_t prim, itype; uint64_t ibuf, ind; } p_draw_idx_ind;
typedef struct { uint64_t ind, caller; } p_dispatch_ind;   // ind = GPU address of the 3 group counts
// size bytes at addr filled with the 32-bit value in GPU order (ClearBuffer; ReportValue is a 4-byte one)
typedef struct { uint64_t addr, size; uint32_t value, pad; } p_clear_buf;
// Push{Graphics,Compute}Constants: bytes at byte offset into root table `table` of the graphics (compute 0) or compute
// (compute 1) pipeline, read as constant bank 24. src: a *Deferred push's data in game memory, read at submit into w.
typedef struct { int32_t compute, table, offset, bytes; uint64_t src; uint32_t w[64]; } p_push_const;
typedef struct { int32_t first, count; } p_draw_mesh;      // DrawMeshTasks: task groups first.. (the mesh shader study)
typedef struct { uint64_t ind; } p_draw_mesh_ind;          // DrawMeshTasksIndirect: GPU address of {count, first}
typedef struct { int32_t enable; float min, max; } p_depth_bounds;
typedef struct { int32_t which, value, extra; } p_conservative;   // which 0: SetConservativeRasterEnable; 1: ...Mode (value mode, extra the 4-bit field)
typedef struct { nvn_tex_rec tex; } p_shading_rate_image;  // present 0 unbinds
// ShadingRateImagePalette: entries first.. of palette `index`; has_rates 0: the driver's default rates
typedef struct { int32_t index, n, has_rates, pad; uint32_t rates[16]; } p_shading_rate_palette;
typedef struct { uint64_t addr, size; } p_scratch;         // SetShaderScratchMemory: GPU range (0, 0: none)
typedef struct { int32_t stage, first, n, pad; uint64_t src; } p_bind_accel;   // BindAccelerationStructuresDeferred
// Calls that change how the GPU schedules or caches, never what it computes: the back end may ignore them
enum { NVN_HINT_COMPUTE_OVERLAP, NVN_HINT_COMPUTE_PRIORITY, NVN_HINT_COLOR_REDUCTION, NVN_HINT_COLOR_REDUCTION_THRESHOLDS,
       NVN_HINT_STENCIL_CULL, NVN_HINT_TILED_CACHE_ACTION, NVN_HINT_TILED_CACHE_TILE_SIZE, NVN_HINT_KINDS };
typedef struct { int32_t what; int32_t v[3]; } p_hint;
const char *nvn_op_name(uint32_t op);   // "DRAW_IDX", ...; "?" past OP_COUNT_ (runtime/nvn/nvn_cmd.c)
