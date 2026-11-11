// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The desktop harness of the Vulkan back end's replay (the Vulkan replay; oot2/vk_replay.h), driven by
// tests/test_runtime_vk_replay_draw.py:
//
//   vk_replay_draw probe
//   vk_replay_draw draw DRAW OUT --pack PACK [--depth-out FILE] [--depth-mode near_is_minus_w|near_is_zero]
//                  [--origin lower_left|upper_left] [--imports] [--no-snapshot] [--no-flip] [--rewrite-uniform] [--cold]
//                  [--unpad]
//
// --cold leaves the program's pipeline unbuilt, so that the first draw is a miss: the replay's miss path (oot2/vk_miss.h)
// drains, builds and resumes, and the image must be the one a prebuilt pipeline gives.
// --unpad sets vkp_unpad: the counter-run of the padding rules, which leaves a vertex or mesh stage as translated, so that a
// pixel input nothing writes reaches the validation layer (VUID-RuntimeSpirv-OpEntryPoint-08743).
//
// `probe` opens the Vulkan device and closes it again: "ok probe", or the named fatal of a machine that has none.
//
// `draw` plays one draw, described by the text file DRAW, through the NVN2 entry points as a game calls them (their
// synthetic addresses, arguments in the AAPCS64 registers, every object in guest memory), on the Vulkan back end, and
// writes the colour target as raw 32-bit floats to OUT (row 0 first, as the back end's read_texels gives it) and, with
// --depth-out, the depth target the same way. Each stage's control section and code area come from the files named in
// DRAW, and its job key names the module in PACK. The key hashes both files, so ProgramSetShaders computes it itself
// and the harness stops unless it equals the one DRAW names. Stages may come from game archives or be synthesized for
// a test; no game data is needed.
//
// The draw file holds one directive a line, every file name relative to the file's own folder:
//   size W H                        the colour target (RGBA32F), W by H
//   clear R G B A                   its clear colour
//   viewport X Y W H, scissor X Y W H   default: the whole target
//   depth NEAR FAR CLEAR            a DEPTH32F target, cleared to CLEAR, depth test LESS and write on, depth range NEAR FAR
//   program vs|ps|ts|ms KEY CTL CODE a stage: its job key, control section file, code area file
//   attribute LOC FORMAT OFFSET STREAM    FORMAT r32f rg32f rgb32f rgba32f or the 32-bit integer forms (r32i, r32ui, ...)
//   stream INDEX STRIDE DIVISOR     vertex stream state
//   vertex_buffer INDEX FILE        bound at stream INDEX
//   index_buffer FILE u8|u16|u32    the indices DrawElements reads
//   uniform vs|ps R FILE            BindUniformBuffer index R (constant bank c[3 + R]); FILE a bank
//   storage vs|ps R LOW FILE        BindStorageBuffer index R (row R of the stage's storage table in c[0]); the buffer
//                                   starts LOW bytes (0 to 15) past a 256-byte boundary, as the contract's address
//                                   rounding sees it; FILE holds the bytes the shader reads from the address on
//   texture vs|ps SLOT ID TYPE W H D FILTER ADDRESS FILE
//                                   BindTexture at c[2] slot SLOT of the texture and sampler registered at ID (both
//                                   pools, one entry each); TYPE 2d, 2darray, 3d or cube; D layers or depth; texels
//                                   RGBA32F, layer (slice, face) major, row 0 first
//   draw arrays PRIM FIRST COUNT, draw elements PRIM COUNT, draw indirect PRIM COUNT BASE_INSTANCE
//                                   the last: DrawElementsIndirect over the index buffer, its record {COUNT, 1, 0, 0,
//                                   BASE_INSTANCE} in a buffer of the CPU pool
//   draw mesh direct FIRST COUNT, draw mesh indirect COUNT FIRST
//                                   the indirect record {COUNT, FIRST} is in a buffer of the CPU pool
//   polygon line|point|fill OFFSETS, offset FACTOR UNITS CLAMP
//                                   the polygon state (mode, offset enables: point 1, line 2, fill 4) and the offset
//   depth_clamp, depth_bounds MIN MAX, line_width W
//   viewports N                     N viewports and scissors: the first is the draw's, the others are quarters of the
//                                   target (a program that writes no viewport index draws to the first)
//   targets 2                       a second colour target (RGBA32F, the size of the first, cleared the same), written
//                                   with --out1
//   chmask T HEX                    target T's channel mask (r 1, g 2, b 4, a 8), the others 0xf
//   blend_dual                      blend on target 0 with dual-source factors (SRC1_COLOR, ONE_MINUS_SRC1_COLOR)
//   bctarget SLICE placed|arena [FORMAT]
//                                   the colour target is slice SLICE of a 32x32x20 BC4 volume (8x8x20 blocks of 8 bytes),
//                                   viewed in FORMAT (rgba16ui, the default; rgba16f, rgba16 for unorm, rgba8, bc4), so a texel
//                                   of the target is a block when the format is the block's size: size must be 8 8, no depth,
//                                   no clear. The volume stands in the pool of images (placed) or in the CPU-visible pool
//                                   (an arena image, the game's case); --bc-out FILE gets its blocks as the image holds them
//                                   (8 blocks a row, 8 rows a slice, 20 slices) and "bc image placed|arena" is printed
//   bcsecond SLICE [store]          after the first draw, the same pass again into slice SLICE (a second submission); with
//                                   store, the CPU stores a byte into the volume's last page between the two
//   bcsample FILE [same]            after the passes, the draw file FILE (size 8 8, one texture of the file name @volume: the
//                                   volume as BC4, 3d) is played into a float target, written to OUT; with same, in the
//                                   submission of the last pass (two passes of one frame), else in a submission of its own
//
// The program stages may be synthesized for a test; no game data is needed. ProgramSetShaders computes each stage's key
// from its control section and code area, so both files must match the key in the draw file.
//
// What this expects of the replay (oot2/vk_replay.h and runtime/gpu/vk_replay.c, as written): every record of the
// recording is replayed; the driver banks c[0], c[1] (from the code area at the control section's +0x40 and +0x44), c[2]
// and the support buffer are the replay's own; a uniform range in host memory that the backing map imports (--imports) is
// copied at submit, so a write to it after QueueSubmitCommands does not reach the frame (--no-snapshot binds it live);
// the Y flip is a negative viewport height whichever the window origin, so clip y = +1 lands on memory row 0; NEAR_IS_MINUS_W
// sets depth clip -1 to 1. The colour and depth targets and the textures stand in CPU_NO_ACCESS pools, as images stand
// only on GPU-only storage; the pipeline of the pair is built before the first submission.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "layout.h"
#include "endmark.h"
#include "oot2/backing.h"
#include "oot2/gfx.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/nvn_program.h"
#include "oot2/runtime.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_image.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "vk_draw.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

typedef unsigned long long ull;
#define MiB (1ull << 20)
#define HEAP_BYTES (96 * MiB)
#define CPU_POOL_BYTES (8 * MiB)       // the CPU-visible pool: commands, programs, buffers, the descriptor tables
#define COMMAND_BYTES (256u * 1024u)   // command memory of the recording, at the start of that pool
#define CONTROL_BYTES 0x10000u
#define POOL_IDS 64                    // entries of the texture and of the sampler pool
#define SYNC_TIMEOUT_NS 30000000000ull
#define CPU_NO_ACCESS 0x1u
#define CPU_UNCACHED 0x2u
#define GPU_CACHED 0x20u
#define SHADER_CODE 0x40u
#define LOW32(v) ((v) & 0xffffffffu)

// ---- the draw file
enum { ST_VS = NVN_STAGE_VS, ST_PS = NVN_STAGE_PS, ST_TS = NVN_STAGE_TS, ST_MS = NVN_STAGE_MS };
enum { NVN_STAGE_TASK = 6, NVN_STAGE_MESH = 7 };
#define NAME 128
#define MAX_ATTR 16
#define MAX_STREAM 8
#define MAX_UBO 24
#define MAX_SBO 16
#define MAX_TEX 40
typedef struct {
    int stage;
    char key[40], ctl[NAME], code[NAME];
} Prog;
typedef struct {
    unsigned loc, offset, stream;
    int format;
} Attr;
typedef struct {
    unsigned stride, divisor;
} Stream;
typedef struct {
    unsigned index;
    char file[NAME];
} VBuf;
typedef struct {
    int stage;
    unsigned r, low;
    char file[NAME];
} Bind;
typedef struct {
    int stage, target, nearest, repeat;
    unsigned slot, id, w, h, d;
    char file[NAME];
} Tex;
typedef struct {
    unsigned w, h;
    float clear[4];
    int vp[4], sc[4];
    int depth;
    float dnear, dfar, dclear;
    Prog prog[3];
    int nprog;
    Attr attr[MAX_ATTR];
    int nattr;
    Stream stream[MAX_STREAM];
    int nstream;
    VBuf vbuf[4];
    int nvbuf;
    char ibuf[NAME];
    int itype;
    Bind ubo[MAX_UBO], sbo[MAX_SBO];
    int nubo, nsbo;
    Tex tex[MAX_TEX];
    int ntex;
    int indexed, prim, first, count, mesh, mesh_indirect;
    int polygon, offsets;      // NVNpolygonMode, or -1 for no polygon state
    float offset[3];
    int has_offset, clamp, bounds;
    float bmin, bmax, line_width;
    int nvp, ntargets;
    unsigned chmask[2];
    int has_chmask, dual, indirect, base_instance;
    int bc_slice, bc_arena;    // bctarget: the colour target is slice bc_slice of a BC4 volume, viewed in bc_fmt (-1: none)
    int bc_fmt;                // the view's NVN format, and the Vulkan format the pipeline draws in
    VkFormat bc_vk;
    int bc_second, bc_store;   // bcsecond: another pass into this slice after the first (-1: none), the CPU's store between
    char bc_sample[NAME];      // bcsample: the draw file of a draw that samples the volume afterwards
    int bc_same;               // ... in the submission of the last pass (a later pass of the one frame), not in one of its own
} Draw;

typedef struct {
    const char *draw, *out, *out1, *pack, *depth_out, *bc_out;
    int depth_mode, origin;   // NVNdepthMode, NVNwindowOriginMode
    int imports, no_snapshot, no_flip, rewrite, cold, no_features, unpad;
} Opts;

static void usage(const char *why) {
    fprintf(stderr, "vk_replay_draw: %s\nusage: vk_replay_draw probe | draw DRAW OUT --pack PACK [--depth-out FILE] "
                    "[--depth-mode near_is_minus_w|near_is_zero] [--origin lower_left|upper_left] [--imports] "
                    "[--no-snapshot] [--no-flip] [--rewrite-uniform] [--unpad]\n", why);
    exit(2);
}

static char g_dir[512];   // the draw file's folder, with its separator

static void *read_file(const char *name, size_t *size) {
    char path[1024];
    CHECK(snprintf(path, sizeof path, "%s%s", g_dir, name) < (int)sizeof path);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(n > 0 ? (size_t)n : 1);
    CHECK(p && fread(p, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    *size = (size_t)n;
    return p;
}

static void write_file(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    CHECK(f && fwrite(data, 1, n, f) == n);
    fclose(f);
}

static int stage_of(const char *s) {
    if (!strcmp(s, "vs")) return ST_VS;
    if (!strcmp(s, "ps")) return ST_PS;
    if (!strcmp(s, "ts")) return NVN_STAGE_TASK;
    if (!strcmp(s, "ms")) return NVN_STAGE_MESH;
    fprintf(stderr, "unknown stage %s\n", s);
    exit(2);
}

static int stage_kind(int stage) {
    switch (stage) {
    case ST_VS: return NVN_STAGE_VS;
    case ST_PS: return NVN_STAGE_PS;
    case NVN_STAGE_TASK: return NVN_STAGE_TS;
    case NVN_STAGE_MESH: return NVN_STAGE_MS;
    default: CHECK(0); return 0;
    }
}

// The NVNformat of a vertex attribute's name.
static int attr_format(const char *s) {
    static const struct { const char *name; int format; } t[] = {
        {"r32f", NVN_FORMAT_R32F}, {"rg32f", NVN_FORMAT_RG32F}, {"rgb32f", NVN_FORMAT_RGB32F},
        {"rgba32f", NVN_FORMAT_RGBA32F}, {"r32i", NVN_FORMAT_R32I}, {"rg32i", NVN_FORMAT_RG32I},
        {"rgb32i", NVN_FORMAT_RGB32I}, {"rgba32i", NVN_FORMAT_RGBA32I}, {"r32ui", NVN_FORMAT_R32UI},
        {"rg32ui", NVN_FORMAT_RG32UI}, {"rgb32ui", NVN_FORMAT_RGB32UI}, {"rgba32ui", NVN_FORMAT_RGBA32UI}};
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++)
        if (!strcmp(t[i].name, s)) return t[i].format;
    fprintf(stderr, "unknown attribute format %s\n", s);
    exit(2);
}

static int texture_target(const char *s) {
    if (!strcmp(s, "2d")) return NVN_TEXTURE_TARGET_2D;
    if (!strcmp(s, "2darray")) return NVN_TEXTURE_TARGET_2D_ARRAY;
    if (!strcmp(s, "3d")) return NVN_TEXTURE_TARGET_3D;
    if (!strcmp(s, "cube")) return NVN_TEXTURE_TARGET_CUBEMAP;
    fprintf(stderr, "unknown texture type %s\n", s);
    exit(2);
}

static void parse_draw(const char *path, Draw *d) {
    const char *slash = strrchr(path, '/'), *bslash = strrchr(path, '\\');
    if (bslash > slash) slash = bslash;
    size_t dirlen = slash ? (size_t)(slash - path) + 1 : 0;
    CHECK(dirlen < sizeof g_dir);
    memcpy(g_dir, path, dirlen);
    g_dir[dirlen] = 0;
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    memset(d, 0, sizeof *d);
    d->prim = NVN_DRAW_PRIMITIVE_TRIANGLES;
    d->vp[2] = d->sc[2] = -1;
    d->polygon = -1, d->nvp = 1, d->ntargets = 1, d->chmask[0] = d->chmask[1] = 0xF, d->bc_slice = -1,
    d->bc_second = -1, d->bc_fmt = NVN_FORMAT_RGBA16UI, d->bc_vk = VK_FORMAT_R16G16B16A16_UINT;
    char line[1200];
    while (fgets(line, sizeof line, f)) {
        char w[32], a[NAME], b[NAME], c2[NAME], e[NAME], g[NAME], h[NAME], file[NAME];
        unsigned u[8];
        float x[4];
        int n[4];
        if (sscanf(line, "%31s", w) != 1) continue;
        const char *rest = line + strlen(w);
        if (!strcmp(w, "size")) {
            CHECK(sscanf(rest, "%u %u", &d->w, &d->h) == 2);
        } else if (!strcmp(w, "bctarget")) {
            char fmt[NAME] = "rgba16ui";
            CHECK(sscanf(rest, "%u %127s %127s", &u[0], a, fmt) >= 2 && u[0] < 20 && (!strcmp(a, "arena") || !strcmp(a, "placed")));
            d->bc_slice = (int)u[0], d->bc_arena = !strcmp(a, "arena");
            if (!strcmp(fmt, "rgba16f")) d->bc_fmt = NVN_FORMAT_RGBA16F, d->bc_vk = VK_FORMAT_R16G16B16A16_SFLOAT;
            else if (!strcmp(fmt, "rgba16")) d->bc_fmt = NVN_FORMAT_RGBA16, d->bc_vk = VK_FORMAT_R16G16B16A16_UNORM;
            else if (!strcmp(fmt, "rgba8")) d->bc_fmt = NVN_FORMAT_RGBA8, d->bc_vk = VK_FORMAT_R8G8B8A8_UNORM;
            else if (!strcmp(fmt, "bc4")) d->bc_fmt = NVN_FORMAT_RGTC1_UNORM;   // the compressed format itself: no pass can draw in it
            else CHECK(!strcmp(fmt, "rgba16ui"));
        } else if (!strcmp(w, "bcsecond")) {
            char how[NAME] = "";
            CHECK(sscanf(rest, "%u %127s", &u[0], how) >= 1 && u[0] < 20);
            d->bc_second = (int)u[0], d->bc_store = !strcmp(how, "store");
        } else if (!strcmp(w, "bcsample")) {
            char how[NAME] = "";
            CHECK(sscanf(rest, "%127s %127s", d->bc_sample, how) >= 1);
            d->bc_same = !strcmp(how, "same");
        } else if (!strcmp(w, "clear")) {
            CHECK(sscanf(rest, "%f %f %f %f", &d->clear[0], &d->clear[1], &d->clear[2], &d->clear[3]) == 4);
        } else if (!strcmp(w, "viewport")) {
            CHECK(sscanf(rest, "%d %d %d %d", &n[0], &n[1], &n[2], &n[3]) == 4);
            memcpy(d->vp, n, sizeof n);
        } else if (!strcmp(w, "scissor")) {
            CHECK(sscanf(rest, "%d %d %d %d", &n[0], &n[1], &n[2], &n[3]) == 4);
            memcpy(d->sc, n, sizeof n);
        } else if (!strcmp(w, "depth")) {
            CHECK(sscanf(rest, "%f %f %f", &x[0], &x[1], &x[2]) == 3);
            d->depth = 1, d->dnear = x[0], d->dfar = x[1], d->dclear = x[2];
        } else if (!strcmp(w, "program")) {
            CHECK(sscanf(rest, "%127s %39s %127s %127s", a, b, c2, e) == 4 && d->nprog < 3);
            Prog *p = &d->prog[d->nprog++];
            p->stage = stage_of(a);
            snprintf(p->key, sizeof p->key, "%s", b);
            snprintf(p->ctl, sizeof p->ctl, "%s", c2);
            snprintf(p->code, sizeof p->code, "%s", e);
        } else if (!strcmp(w, "attribute")) {
            CHECK(sscanf(rest, "%u %127s %u %u", &u[0], a, &u[1], &u[2]) == 4 && d->nattr < MAX_ATTR && u[0] < 32);
            d->attr[d->nattr++] = (Attr){u[0], u[1], u[2], attr_format(a)};
        } else if (!strcmp(w, "stream")) {
            CHECK(sscanf(rest, "%u %u %u", &u[0], &u[1], &u[2]) == 3 && u[0] < MAX_STREAM);
            d->stream[u[0]] = (Stream){u[1], u[2]};
            if ((int)u[0] + 1 > d->nstream) d->nstream = (int)u[0] + 1;
        } else if (!strcmp(w, "vertex_buffer")) {
            CHECK(sscanf(rest, "%u %127s", &u[0], file) == 2 && d->nvbuf < 4);
            d->vbuf[d->nvbuf].index = u[0];
            snprintf(d->vbuf[d->nvbuf++].file, NAME, "%s", file);
        } else if (!strcmp(w, "index_buffer")) {
            CHECK(sscanf(rest, "%127s %127s", d->ibuf, a) == 2);
            d->itype = !strcmp(a, "u8")    ? NVN_INDEX_TYPE_UNSIGNED_BYTE
                       : !strcmp(a, "u16") ? NVN_INDEX_TYPE_UNSIGNED_SHORT
                                           : NVN_INDEX_TYPE_UNSIGNED_INT;
        } else if (!strcmp(w, "uniform")) {
            CHECK(sscanf(rest, "%127s %u %127s", a, &u[0], file) == 3 && d->nubo < MAX_UBO);
            Bind *k = &d->ubo[d->nubo++];
            k->stage = stage_of(a), k->r = u[0];
            snprintf(k->file, NAME, "%s", file);
        } else if (!strcmp(w, "storage")) {
            CHECK(sscanf(rest, "%127s %u %u %127s", a, &u[0], &u[1], file) == 4 && d->nsbo < MAX_SBO && u[1] < 16);
            Bind *k = &d->sbo[d->nsbo++];
            k->stage = stage_of(a), k->r = u[0], k->low = u[1];
            snprintf(k->file, NAME, "%s", file);
        } else if (!strcmp(w, "texture")) {
            CHECK(sscanf(rest, "%127s %u %u %127s %u %u %u %127s %127s %127s", a, &u[0], &u[1], b, &u[2], &u[3], &u[4], g,
                         h, file) == 10 && d->ntex < MAX_TEX);
            Tex *t = &d->tex[d->ntex++];
            t->stage = stage_of(a), t->slot = u[0], t->id = u[1], t->target = texture_target(b);
            t->w = u[2], t->h = u[3], t->d = u[4];
            t->nearest = strcmp(g, "linear") != 0, t->repeat = !strcmp(h, "repeat");
            snprintf(t->file, NAME, "%s", file);
        } else if (!strcmp(w, "polygon")) {
            CHECK(sscanf(rest, "%127s %u", a, &u[0]) == 2);
            d->polygon = !strcmp(a, "line") ? NVN_POLYGON_MODE_LINE : !strcmp(a, "point") ? NVN_POLYGON_MODE_POINT
                                                                                           : NVN_POLYGON_MODE_FILL;
            d->offsets = (int)u[0];
        } else if (!strcmp(w, "offset")) {
            CHECK(sscanf(rest, "%f %f %f", &d->offset[0], &d->offset[1], &d->offset[2]) == 3);
            d->has_offset = 1;
        } else if (!strcmp(w, "depth_clamp")) {
            d->clamp = 1;
        } else if (!strcmp(w, "depth_bounds")) {
            CHECK(sscanf(rest, "%f %f", &d->bmin, &d->bmax) == 2);
            d->bounds = 1;
        } else if (!strcmp(w, "line_width")) {
            CHECK(sscanf(rest, "%f", &d->line_width) == 1);
        } else if (!strcmp(w, "viewports")) {
            CHECK(sscanf(rest, "%d", &d->nvp) == 1 && d->nvp >= 1 && d->nvp <= 16);
        } else if (!strcmp(w, "targets")) {
            CHECK(sscanf(rest, "%d", &d->ntargets) == 1 && d->ntargets >= 1 && d->ntargets <= 2);
        } else if (!strcmp(w, "chmask")) {
            CHECK(sscanf(rest, "%u %x", &u[0], &u[1]) == 2 && u[0] < 2);
            d->chmask[u[0]] = u[1], d->has_chmask = 1;
        } else if (!strcmp(w, "blend_dual")) {
            d->dual = 1;
        } else if (!strcmp(w, "draw")) {
            CHECK(sscanf(rest, "%127s", a) == 1);
            if (!strcmp(a, "mesh")) {
                char form[32];
                CHECK(sscanf(rest, "%*s %31s", form) == 1);
                d->mesh = 1;
                if (!strcmp(form, "direct")) {
                    CHECK(sscanf(rest, "%*s %*s %d %d", &d->first, &d->count) == 2);
                } else {
                    CHECK(!strcmp(form, "indirect") &&
                          sscanf(rest, "%*s %*s %d %d", &d->count, &d->first) == 2);
                    d->mesh_indirect = 1;
                }
            } else if (!strcmp(a, "indirect")) {
                CHECK(sscanf(rest, "%*s %d %d %d", &d->prim, &d->count, &d->base_instance) == 3);
                d->indexed = 1, d->indirect = 1;
            } else if (!strcmp(a, "arrays")) {
                CHECK(sscanf(rest, "%*s %d %d %d", &d->prim, &d->first, &d->count) == 3);
            } else {
                CHECK(!strcmp(a, "elements") && sscanf(rest, "%*s %d %d", &d->prim, &d->count) == 2);
                d->indexed = 1;
            }
        } else {
            fprintf(stderr, "unknown directive %s\n", w);
            exit(2);
        }
    }
    fclose(f);
    CHECK(d->w && d->h && d->count > 0);
    if (d->mesh) {
        CHECK(d->nprog == 3 && d->prog[0].stage == NVN_STAGE_TASK && d->prog[1].stage == NVN_STAGE_MESH &&
              d->prog[2].stage == ST_PS && d->nvbuf == 0 && !d->indexed && !d->nattr && !d->nstream);
        CHECK(d->first >= 0);
    } else {
        CHECK(d->nprog == 2 && d->prog[0].stage == ST_VS && d->prog[1].stage == ST_PS && d->nvbuf > 0);
        CHECK(d->bc_slice < 0 || (d->ntargets == 1 && !d->depth && d->w == 8 && d->h == 8));
        CHECK(d->bc_second < 0 || d->bc_slice >= 0);   // the pass of a 32x32 level's blocks
    }
    if (d->vp[2] < 0) d->vp[0] = d->vp[1] = 0, d->vp[2] = (int)d->w, d->vp[3] = (int)d->h;
    if (d->sc[2] < 0) d->sc[0] = d->sc[1] = 0, d->sc[2] = (int)d->w, d->sc[3] = (int)d->h;
}

// ---- the guest side
static Cpu c;
static uint64_t heap, heap_used, loader;

static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP_BYTES);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}

static uint64_t guest_str(const char *s) {
    size_t n = strlen(s) + 1;
    uint64_t a = galloc(n, 16);
    memcpy((void *)(uintptr_t)a, s, n);
    return a;
}

static uint64_t vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}

static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[160];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) {
        c.x[0] = guest_str("nvnDeviceGetProcAddress"), c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
    }
    CHECK(gpa != 0 && n < 160);
    c.x[0] = 0, c.x[1] = guest_str(name);
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0);
    cache[n].name = name, cache[n].addr = c.x[0];
    return cache[n++].addr;
}
#define NVN(name, ...)                                                                                               \
    vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t))
#define OK(call) CHECK(((call) & 0xff) == 1)

// A float argument, in v<i>.
static void fv(int i, float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    c.v[i] = b;
}
static void wr64(uint64_t a, uint64_t v) { memcpy((void *)(uintptr_t)a, &v, 8); }
static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// ---- the back end, in the order the program starts it (runtime/program/main.c): the device and the table first, then the
// runtime, then the entry points. gfx_select("vulkan", ...) is the pair of calls below.
static void open_backend(const Opts *o, const char *pack) {
    gpu_vk_open();
    gfx_set_backend(&gfx_vulkan_backend);
    if (o->imports) backing_set_mirror_hook(NULL);   // CPU-visible pools are host imports, whichever the device
    VkRecOpen ro = {0};
    vkrec_open(&ro);
    char why[512];
    if (!vkp_store_open(pack, NULL, why, sizeof why)) {
        fprintf(stderr, "the pack did not open: %s\n", why);
        exit(1);
    }
    vkp_open(NULL);
    vkr_no_snapshot = o->no_snapshot;
    vkr_no_flip = o->no_flip;
    vkr_no_features = o->no_features;
}

// The module's one stand-in function and its end marker, defined in this order so that the function lies below the marker
// (oot2/runtime.h: code_end). Nothing calls the stand-in: the NVN entry points are host functions at synthetic addresses.
__attribute__((noinline, used)) void stand_in(Cpu *cpu) { (void)cpu; }
__attribute__((noinline, used)) void test_lifted_end(Cpu *cpu) { __asm__ volatile("int3" : : "r"(cpu) : "memory"); }

static void start_runtime(void) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512 * MiB, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256 * MiB, .system_resource_size = MiB};
    rt_mem_init(&m);
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    static const uint64_t addrs[1] = {MOD_MAIN_BASE};
    static const GuestFn fns[1] = {stand_in};
    static const RtModule mods[1] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, addrs, fns, 1, TEST_END}};
    rt_dispatch_init(mods, 1);
    c.x[0] = 0, c.x[1] = HEAP_BYTES;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    c.excl_addr = ~0ull;
}

// ---- objects, as the game makes them
static uint64_t g_dev, g_queue, g_cpu_pool, g_cpu_mem, g_cpu_used, g_gpu_pool, g_gpu_used;

static uint64_t make_pool(uint64_t size, uint32_t flags, uint64_t *storage) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16), mem = galloc(size, 0x10000);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetDevice, b, g_dev);
    NVN(nvnMemoryPoolBuilderSetStorage, b, mem, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, flags);
    OK(NVN(nvnMemoryPoolInitialize, o, b));
    if (storage) *storage = mem;
    return o;
}

static void make_queue(void) {
    uint64_t b = galloc(0x40, 16);
    g_queue = galloc(0x2000, 16);
    NVN(nvnQueueBuilderSetDevice, b, g_dev);
    NVN(nvnQueueBuilderSetDefaults, b);
    NVN(nvnQueueBuilderSetFlags, b, 0);
    NVN(nvnQueueBuilderSetCommandMemorySize, b, 0x10000);
    NVN(nvnQueueBuilderSetComputeMemorySize, b, 0x40000);
    NVN(nvnQueueBuilderSetControlMemorySize, b, 0x4000);
    uint64_t size = NVN(nvnQueueBuilderGetQueueMemorySize, b);
    NVN(nvnQueueBuilderSetQueueMemory, b, galloc(size, 0x1000), size);
    OK(NVN(nvnQueueInitialize, g_queue, b));
}

// Room in the CPU pool: its offset, `low` bytes past the alignment.
static uint64_t cpu_take(uint64_t bytes, uint64_t align, uint64_t low) {
    uint64_t off = align_up(g_cpu_used, align) + low;
    g_cpu_used = off + bytes;
    CHECK(g_cpu_used <= CPU_POOL_BYTES);
    return off;
}

typedef struct {
    uint64_t obj, addr, size;
} Buf;

// A buffer of the CPU pool holding n bytes of data (bytes long, at least 16: a storage buffer's rounding down never leaves
// the pool).
static Buf make_buffer(const void *data, uint64_t n, uint64_t align, uint64_t low) {
    uint64_t bytes = n < 16 ? 16 : n;
    uint64_t off = cpu_take(bytes, align, low);
    uint64_t b = galloc(0x40, 16), o = galloc(0x40, 16);
    NVN(nvnBufferBuilderSetDevice, b, g_dev);
    NVN(nvnBufferBuilderSetDefaults, b);
    NVN(nvnBufferBuilderSetStorage, b, g_cpu_pool, off, bytes);
    OK(NVN(nvnBufferInitialize, o, b));
    Buf r = {o, NVN(nvnBufferGetAddress, o), bytes};
    CHECK(r.addr == g_cpu_mem + off);
    memcpy((void *)(uintptr_t)r.addr, data, n);
    NVN(nvnBufferFlushMappedRange, o, 0, bytes);
    return r;
}

// A texture, planned first (its storage size and alignment place it in the pool of images) and initialised once the pool
// exists.
typedef struct {
    uint64_t builder, tex, view, off, size;
    int target, format;
    uint32_t w, h, d;
    int cpu;   // the storage is in the CPU-visible pool (an arena image) and not in the pool of images
} GTex;

static GTex plan_texture(int target, int format, uint32_t w, uint32_t h, uint32_t d) {
    GTex t = {0};
    t.builder = galloc(0x100, 16), t.tex = galloc(0x100, 16), t.view = galloc(16, 16);
    t.target = target, t.format = format, t.w = w, t.h = h, t.d = d;
    NVN(nvnTextureBuilderSetDevice, t.builder, g_dev);
    NVN(nvnTextureBuilderSetDefaults, t.builder);
    NVN(nvnTextureBuilderSetTarget, t.builder, (uint64_t)target);
    NVN(nvnTextureBuilderSetFormat, t.builder, (uint64_t)format);
    if (target == NVN_TEXTURE_TARGET_2D) NVN(nvnTextureBuilderSetSize2D, t.builder, w, h);
    else NVN(nvnTextureBuilderSetSize3D, t.builder, w, h, d);
    NVN(nvnTextureViewSetDefaults, t.view);
    t.size = NVN(nvnTextureBuilderGetStorageSize, t.builder);
    uint64_t align = NVN(nvnTextureBuilderGetStorageAlignment, t.builder);
    t.off = g_gpu_used = align_up(g_gpu_used, align ? align : 0x1000);
    g_gpu_used += t.size;
    return t;
}

static void init_texture(GTex *t) {
    NVN(nvnTextureBuilderSetStorage, t->builder, t->cpu ? g_cpu_pool : g_gpu_pool, t->off);
    OK(NVN(nvnTextureInitialize, t->tex, t->builder));
}

// The BC4 volume of 32x32x20 texels (8x8x20 blocks of 8 bytes) a block compressor writes: in the pool of images, or, for an
// arena image, in the CPU-visible pool, and the view the pass targets, RGBA16 UINT on one slice (a texel of it a block).
static GTex plan_bc_volume(int arena, int slice, int view_format) {
    uint64_t keep = g_gpu_used;
    GTex t = plan_texture(NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGTC1_UNORM, 32, 32, 20);
    if (arena) {
        uint64_t align = NVN(nvnTextureBuilderGetStorageAlignment, t.builder);
        g_gpu_used = keep;
        t.off = cpu_take(t.size, align ? align : 0x1000, 0);
        t.cpu = 1;
    }
    NVN(nvnTextureViewSetLevels, t.view, 0, 1);
    NVN(nvnTextureViewSetLayers, t.view, (uint64_t)slice, 1);
    NVN(nvnTextureViewSetFormat, t.view, (uint64_t)view_format);
    return t;
}

#define BC_BYTES (8u * 8u * 20u * 8u)

// The volume's blocks as the image holds them (the arena image's stores are its own: the texel calls of a texture the CPU can
// see read the CPU's bytes), through a copy of the image into a buffer: 8 blocks a row, 8 rows a slice, 20 slices.
static void read_bc_volume(const GTex *t, const char *path) {
    tex_t *tt = (tex_t *)(uintptr_t)t->tex;
    VkbImage *img = vkb_image_demand(tt);
    CHECK(img);
    printf("bc image %s\n", vkb_image_placed(img) ? "placed" : "arena");
    DrawBuf out = draw_buffer(BC_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL, 0);
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    VkBufferImageCopy c = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {32, 32, 20}};
    vkCmdCopyImageToBuffer(cb, vkb_image_handle(img), VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    gpu_vk_end(NULL, 0);
    CHECK(gpu_sync_wait(v, 30ull * 1000 * 1000 * 1000));
    write_file(path, out.map, BC_BYTES);
    draw_buffer_free(&out);
}

// The texels of a draw-file texture into the texture, through the back end's write_texels.
static void write_texels(const GTex *t, const char *file) {
    size_t n;
    void *data = read_file(file, &n);
    uint32_t depth = t->target == NVN_TEXTURE_TARGET_2D ? 1 : t->target == NVN_TEXTURE_TARGET_CUBEMAP ? 6 : t->d;
    CHECK(n == (size_t)t->w * t->h * depth * 16);
    uint64_t src = galloc(n, 16), region = galloc(sizeof(NVNcopyRegion), 8);
    memcpy((void *)(uintptr_t)src, data, n);
    *(NVNcopyRegion *)(uintptr_t)region = (NVNcopyRegion){0, 0, 0, (int)t->w, (int)t->h, (int)depth};
    NVN(nvnTextureWriteTexels, t->tex, t->view, region, src);
    free(data);
}

// The target read back through the back end's read_texels into a file.
static void read_target(const GTex *t, uint64_t texel_bytes, const char *path) {
    uint64_t bytes = (uint64_t)t->w * t->h * texel_bytes, dst = galloc(bytes, 16), region = galloc(sizeof(NVNcopyRegion), 8);
    *(NVNcopyRegion *)(uintptr_t)region = (NVNcopyRegion){0, 0, 0, (int)t->w, (int)t->h, 1};
    NVN(nvnTextureReadTexels, t->tex, t->view, region, dst);
    write_file(path, (const void *)(uintptr_t)dst, (size_t)bytes);
}

// The draw's stages: their code areas go into the CPU pool, SetShaders records them, and their runtime keys must match the draw.
static uint64_t make_program(const Draw *d, const nvn_stage *ids[NVN_STAGE_KINDS]) {
    uint64_t sd = galloc(16 * (uint64_t)d->nprog, 16), p = galloc(0xC0, 16);
    for (int i = 0; i < d->nprog; i++) {
        size_t cn, kn;
        void *ctl = read_file(d->prog[i].ctl, &cn), *code = read_file(d->prog[i].code, &kn);
        uint64_t off = cpu_take(kn, 256, 0), ctl_at = galloc(cn, 16);
        memcpy((void *)(uintptr_t)(g_cpu_mem + off), code, kn);
        memcpy((void *)(uintptr_t)ctl_at, ctl, cn);
        wr64(sd + 16ull * (uint64_t)i, g_cpu_mem + off);
        wr64(sd + 16ull * (uint64_t)i + 8, ctl_at);
        free(ctl), free(code);
    }
    OK(NVN(nvnProgramInitialize, p, g_dev));
    OK(NVN(nvnProgramSetShaders, p, (uint64_t)d->nprog, sd));
    const prog_t *x = (const prog_t *)(uintptr_t)p;
    for (int i = 0; i < d->nprog; i++) {
        const nvn_stage *id = nvn_program_stage(x, i);
        CHECK(id != NULL && id->kind == stage_kind(d->prog[i].stage));
        ids[id->kind] = id;
        int same = !strcmp(id->key, d->prog[i].key);
        printf("key %s %s %s\n", nvn_stage_kind_name(id->kind), id->key, same ? "equals the pack's" : "DIFFERS from the pack's");
        if (!same) {
            fprintf(stderr, "ProgramSetShaders keyed stage %d %s, the draw names %s\n", i, id->key, d->prog[i].key);
            exit(1);
        }
    }
    return p;
}

static void print_stats(void) {
    VkrStats s;
    vkr_stats(&s);
    printf("replay frames %llu records %llu draws %llu dispatches %llu clears %llu passes %llu barriers %llu\n",
           (ull)s.frames, (ull)s.records, (ull)s.draws, (ull)s.dispatches, (ull)s.clears, (ull)s.passes, (ull)s.barriers);
    printf("replay snapshots %llu snapshot_bytes %llu live_uniforms %llu storage_writes %llu unregistered_textures %llu "
           "scratch_bytes %llu\n", (ull)s.snapshots, (ull)s.snapshot_bytes, (ull)s.live_uniforms, (ull)s.storage_writes,
           (ull)s.unregistered_textures, (ull)s.scratch_bytes);
    printf("replay draw_ops %llu render_off %llu skipped %llu lost %llu\n", (ull)s.draw_ops, (ull)s.render_off,
           (ull)s.skipped, (ull)s.lost);
    VkmStats m;
    vkm_stats(&m);
    printf("miss graphics %llu compute %llu drains %llu drained_submissions %llu skipped_draws %llu late_builds %llu\n",
           (ull)m.graphics_misses, (ull)m.compute_misses, (ull)m.drains, (ull)m.drained_submissions,
           (ull)m.skipped_draws, (ull)m.late_builds);
}

// The recording and the submission of one draw, as a game makes them (the whole of it, programs and buffers included, each
// time): a volume pass may be played again with another slice (slice >= 0), and a draw's texture of the file name @volume is
// the volume the earlier pass wrote (`volume`, the texture with a view of the whole volume).
typedef struct {
    GTex rt, rt1, depth, tex[MAX_TEX];
} Scene;

static int is_volume(const Tex *t) { return !strcmp(t->file, "@volume"); }

static uint64_t g_held[4];   // command buffers recorded and not yet submitted (hold): the next draw submits them with its own
static int g_nheld;

static void play_draw(const Opts *o, const Draw *dp, Scene *sc, const GTex *volume, int slice, int hold) {
    GTex rt = sc->rt, rt1 = sc->rt1, depth = sc->depth, *tex = sc->tex;
    // the descriptor tables: one entry of each pool for each texture, the texture's own sampler beside it
    // (one texture pool a submission: the draw that submits a held one registers its textures in the held draw's)
    static uint64_t g_tp, g_sp;
    if (!g_nheld) {
        uint64_t tp_off = cpu_take(POOL_IDS * 32, 256, 0), sp_off = cpu_take(POOL_IDS * 32, 256, 0);
        g_tp = galloc(0x40, 16), g_sp = galloc(0x40, 16);
        OK(NVN(nvnTexturePoolInitialize, g_tp, g_cpu_pool, tp_off, POOL_IDS));
        OK(NVN(nvnSamplerPoolInitialize, g_sp, g_cpu_pool, sp_off, POOL_IDS));
    }
    const uint64_t tp = g_tp, sp = g_sp;
    uint64_t handle[MAX_TEX];
    for (int i = 0; i < dp->ntex; i++) {
        const Tex *t = &dp->tex[i];
        CHECK(t->id > 0 && t->id < POOL_IDS);
        const GTex *src = is_volume(t) ? volume : &tex[i];
        if (!is_volume(t)) write_texels(&tex[i], t->file);
        NVN(nvnTexturePoolRegisterTexture, tp, t->id, src->tex, src->view);
        uint64_t sb = galloc(0x60, 16), s = galloc(0x40, 16);
        NVN(nvnSamplerBuilderSetDevice, sb, g_dev);
        NVN(nvnSamplerBuilderSetDefaults, sb);
        uint64_t filter = t->nearest ? NVN_MIN_FILTER_NEAREST : NVN_MIN_FILTER_LINEAR;
        uint64_t mag = t->nearest ? NVN_MAG_FILTER_NEAREST : NVN_MAG_FILTER_LINEAR;
        uint64_t wrap = t->repeat ? NVN_WRAP_MODE_REPEAT : NVN_WRAP_MODE_CLAMP_TO_EDGE;
        NVN(nvnSamplerBuilderSetMinMagFilter, sb, filter, mag);
        NVN(nvnSamplerBuilderSetWrapMode, sb, wrap, wrap, wrap);
        OK(NVN(nvnSamplerInitialize, s, sb));
        NVN(nvnSamplerPoolRegisterSampler, sp, t->id, s);
        handle[i] = NVN(nvnDeviceGetTextureHandle, g_dev, t->id, t->id);
    }

    // the programs, and their pipeline before the first submission (the pair's attachment formats are the targets')
    const nvn_stage *ids[NVN_STAGE_KINDS] = {0};
    uint64_t prog = make_program(dp, ids);
    VkpFormats formats = {(uint32_t)dp->ntargets,
                          {dp->bc_slice >= 0 ? dp->bc_vk : VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
                          dp->depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkpStage vs = {.key = ids[ST_VS] ? ids[ST_VS]->key : NULL, .shape = ids[ST_VS]};
    VkpStage task = {.key = ids[ST_TS] ? ids[ST_TS]->key : NULL, .shape = ids[ST_TS]};
    VkpStage mesh = {.key = ids[ST_MS] ? ids[ST_MS]->key : NULL, .shape = ids[ST_MS]};
    VkpStage ps = {.key = ids[ST_PS]->key, .shape = ids[ST_PS]};
    if (!o->cold) CHECK(vkp_graphics(dp->mesh ? &mesh : &vs, dp->mesh ? &task : NULL, &ps, &formats) != VK_NULL_HANDLE);
    else CHECK(vkp_find_graphics(dp->mesh ? &mesh : &vs, dp->mesh ? &task : NULL, &ps, &formats) == VK_NULL_HANDLE);

    // the buffers
    Buf vbuf[4], ibuf = {0}, ubo[MAX_UBO], sbo[MAX_SBO];
    for (int i = 0; i < dp->nvbuf; i++) {
        size_t n;
        void *data = read_file(dp->vbuf[i].file, &n);
        vbuf[i] = make_buffer(data, n, 256, 0);
        free(data);
    }
    Buf indirect = {0};
    if (dp->indexed) {
        size_t n;
        void *data = read_file(dp->ibuf, &n);
        ibuf = make_buffer(data, n, 256, 0);
        free(data);
    }
    if (dp->mesh_indirect) {
        const uint32_t record[2] = {(uint32_t)dp->count, (uint32_t)dp->first};
        indirect = make_buffer(record, sizeof record, 256, 0);
    } else if (dp->indirect) {
        const uint32_t record[5] = {(uint32_t)dp->count, 1, 0, 0, (uint32_t)dp->base_instance};
        indirect = make_buffer(record, sizeof record, 256, 0);
    }
    for (int i = 0; i < dp->nubo; i++) {
        size_t n;
        void *data = read_file(dp->ubo[i].file, &n);
        ubo[i] = make_buffer(data, n, 256, 0);
        free(data);
    }
    for (int i = 0; i < dp->nsbo; i++) {
        size_t n;
        void *data = read_file(dp->sbo[i].file, &n);
        sbo[i] = make_buffer(data, n, 256, dp->sbo[i].low);
        free(data);
    }
    NVN(nvnMemoryPoolFlushMappedRange, g_cpu_pool, 0, g_cpu_used);

    // vertex state: an attribute at each location (the state is indexed by location) and the streams
    int nloc = 0;
    for (int i = 0; i < dp->nattr; i++)
        if ((int)dp->attr[i].loc + 1 > nloc) nloc = (int)dp->attr[i].loc + 1;
    uint64_t attrs = galloc(4 * (uint64_t)(nloc ? nloc : 1), 16), streams = galloc(8 * (uint64_t)(dp->nstream ? dp->nstream : 1), 16);
    for (int i = 0; i < nloc; i++) NVN(nvnVertexAttribStateSetDefaults, attrs + 4ull * (uint64_t)i);
    for (int i = 0; i < dp->nattr; i++) {
        uint64_t s = attrs + 4ull * dp->attr[i].loc;
        NVN(nvnVertexAttribStateSetFormat, s, (uint64_t)dp->attr[i].format, dp->attr[i].offset);
        NVN(nvnVertexAttribStateSetStreamIndex, s, dp->attr[i].stream);
    }
    for (int i = 0; i < dp->nstream; i++) {
        uint64_t s = streams + 8ull * (uint64_t)i;
        NVN(nvnVertexStreamStateSetDefaults, s);
        NVN(nvnVertexStreamStateSetStride, s, dp->stream[i].stride);
        NVN(nvnVertexStreamStateSetDivisor, s, dp->stream[i].divisor);
    }
    uint64_t ds_state = galloc(0x10, 16);
    if (dp->depth) {
        NVN(nvnDepthStencilStateSetDefaults, ds_state);
        NVN(nvnDepthStencilStateSetDepthTestEnable, ds_state, 1);
        NVN(nvnDepthStencilStateSetDepthWriteEnable, ds_state, 1);
        NVN(nvnDepthStencilStateSetDepthFunc, ds_state, NVN_DEPTH_FUNC_LESS);
    }

    // the recording: what a game records for one draw, in the order it would
    if (slice >= 0) NVN(nvnTextureViewSetLayers, rt.view, (uint64_t)slice, 1);   // the next pass of a volume: another slice
    uint64_t cb = galloc(0xC0, 16);
    OK(NVN(nvnCommandBufferInitialize, cb, g_dev));
    // one command memory, or a half each for the draws of one submission (a held one takes the first half, the draw that
    // submits it the second)
    NVN(nvnCommandBufferAddCommandMemory, cb, g_cpu_pool, hold || g_nheld ? (uint64_t)g_nheld * COMMAND_BYTES / 2 : 0,
        hold || g_nheld ? COMMAND_BYTES / 2 : COMMAND_BYTES);
    NVN(nvnCommandBufferAddControlMemory, cb, galloc(CONTROL_BYTES, 16), CONTROL_BYTES);
    NVN(nvnCommandBufferBeginRecording, cb);
    uint64_t rts = galloc(16, 8), rgba = galloc(16, 16);
    wr64(rts, rt.tex);
    if (dp->ntargets > 1) wr64(rts + 8, rt1.tex);
    memcpy((void *)(uintptr_t)rgba, dp->clear, sizeof dp->clear);
    uint64_t rt_views = 0;
    if (dp->bc_slice >= 0) wr64(rt_views = galloc(8, 8), rt.view);
    NVN(nvnCommandBufferSetRenderTargets, cb, (uint64_t)dp->ntargets, rts, rt_views, dp->depth ? depth.tex : 0, 0);
    NVN(nvnCommandBufferSetViewport, cb, (uint64_t)(int64_t)dp->vp[0], (uint64_t)(int64_t)dp->vp[1], (uint64_t)(int64_t)dp->vp[2],
        (uint64_t)(int64_t)dp->vp[3]);
    NVN(nvnCommandBufferSetScissor, cb, (uint64_t)(int64_t)dp->sc[0], (uint64_t)(int64_t)dp->sc[1], (uint64_t)(int64_t)dp->sc[2],
        (uint64_t)(int64_t)dp->sc[3]);
    if (dp->nvp > 1) {   // the draw's viewport and scissor first, then quarters of the target, each in its own corner
        uint64_t vps = galloc(16ull * (uint64_t)dp->nvp, 16), scs = galloc(16ull * (uint64_t)dp->nvp, 16);
        for (int i = 0; i < dp->nvp; i++) {
            float *v = (float *)(uintptr_t)(vps + 16ull * (uint64_t)i);
            int32_t *k = (int32_t *)(uintptr_t)(scs + 16ull * (uint64_t)i);
            if (i == 0) {
                v[0] = (float)dp->vp[0], v[1] = (float)dp->vp[1], v[2] = (float)dp->vp[2], v[3] = (float)dp->vp[3];
                k[0] = dp->sc[0], k[1] = dp->sc[1], k[2] = dp->sc[2], k[3] = dp->sc[3];
            } else {
                v[0] = (float)((i & 1) * (int)dp->w / 2), v[1] = (float)((i & 2) * (int)dp->h / 4);
                v[2] = (float)dp->w / 2, v[3] = (float)dp->h / 2;
                k[0] = (int32_t)v[0], k[1] = (int32_t)v[1], k[2] = (int32_t)dp->w / 2, k[3] = (int32_t)dp->h / 2;
            }
        }
        NVN(nvnCommandBufferSetViewports, cb, 0, (uint64_t)dp->nvp, vps);
        NVN(nvnCommandBufferSetScissors, cb, 0, (uint64_t)dp->nvp, scs);
    }
    if (dp->depth) {
        fv(0, dp->dnear), fv(1, dp->dfar);
        NVN(nvnCommandBufferSetDepthRange, cb);
        NVN(nvnCommandBufferBindDepthStencilState, cb, ds_state);
    }
    if (dp->polygon >= 0) {
        uint64_t ps = galloc(0x10, 16);
        NVN(nvnPolygonStateSetDefaults, ps);
        NVN(nvnPolygonStateSetPolygonMode, ps, (uint64_t)dp->polygon);
        NVN(nvnPolygonStateSetPolygonOffsetEnables, ps, (uint64_t)dp->offsets);
        NVN(nvnCommandBufferBindPolygonState, cb, ps);
    }
    if (dp->has_offset) {
        fv(0, dp->offset[0]), fv(1, dp->offset[1]), fv(2, dp->offset[2]);
        NVN(nvnCommandBufferSetPolygonOffsetClamp, cb);
    }
    if (dp->clamp) NVN(nvnCommandBufferSetDepthClamp, cb, 1);
    if (dp->bounds) {
        fv(0, dp->bmin), fv(1, dp->bmax);
        NVN(nvnCommandBufferSetDepthBounds, cb, 1);
    }
    if (dp->line_width > 0.0f) {
        fv(0, dp->line_width);
        NVN(nvnCommandBufferSetLineWidth, cb);
    }
    if (dp->has_chmask) {
        uint64_t ms = galloc(0x10, 16);
        NVN(nvnChannelMaskStateSetDefaults, ms);
        for (int i = 0; i < dp->ntargets; i++)
            NVN(nvnChannelMaskStateSetChannelMask, ms, (uint64_t)i, dp->chmask[i] & 1, (dp->chmask[i] >> 1) & 1,
                (dp->chmask[i] >> 2) & 1, (dp->chmask[i] >> 3) & 1);
        NVN(nvnCommandBufferBindChannelMaskState, cb, ms);
    }
    if (dp->dual) {
        uint64_t bs = galloc(0x20, 16), cs = galloc(0x10, 16);
        NVN(nvnBlendStateSetDefaults, bs);
        NVN(nvnBlendStateSetBlendTarget, bs, 0);
        NVN(nvnBlendStateSetBlendFunc, bs, NVN_BLEND_FUNC_SRC1_COLOR, NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR,
            NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO);
        NVN(nvnBlendStateSetBlendEquation, bs, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD);
        NVN(nvnColorStateSetDefaults, cs);
        NVN(nvnColorStateSetBlendEnable, cs, 0, 1);
        NVN(nvnCommandBufferBindBlendState, cb, bs);
        NVN(nvnCommandBufferBindColorState, cb, cs);
    }
    NVN(nvnCommandBufferSetTexturePool, cb, tp);
    NVN(nvnCommandBufferSetSamplerPool, cb, sp);
    if (dp->bc_slice < 0) NVN(nvnCommandBufferClearColor, cb, 0, rgba, 0xF);   // a float clear of a UINT target means nothing
    if (dp->ntargets > 1) NVN(nvnCommandBufferClearColor, cb, 1, rgba, 0xF);
    if (dp->depth) {
        fv(0, dp->dclear);
        NVN(nvnCommandBufferClearDepthStencil, cb, 1, 0, 0);
    }
    uint64_t stage_mask = dp->mesh ? (NVN_SHADER_STAGE_FRAGMENT_BIT | (1u << NVN_STAGE_TASK) | (1u << NVN_STAGE_MESH))
                                 : (NVN_SHADER_STAGE_VERTEX_BIT | NVN_SHADER_STAGE_FRAGMENT_BIT);
    NVN(nvnCommandBufferBindProgram, cb, prog, stage_mask);
    if (!dp->mesh) {
        NVN(nvnCommandBufferBindVertexAttribState, cb, (uint64_t)nloc, attrs);
        NVN(nvnCommandBufferBindVertexStreamState, cb, (uint64_t)dp->nstream, streams);
        for (int i = 0; i < dp->nvbuf; i++)
            NVN(nvnCommandBufferBindVertexBuffer, cb, dp->vbuf[i].index, vbuf[i].addr, vbuf[i].size);
    }
    for (int i = 0; i < dp->nubo; i++)
        NVN(nvnCommandBufferBindUniformBuffer, cb, (uint64_t)dp->ubo[i].stage, dp->ubo[i].r, ubo[i].addr, ubo[i].size);
    for (int i = 0; i < dp->nsbo; i++)
        NVN(nvnCommandBufferBindStorageBuffer, cb, (uint64_t)dp->sbo[i].stage, dp->sbo[i].r, sbo[i].addr, sbo[i].size);
    for (int i = 0; i < dp->ntex; i++)
        NVN(nvnCommandBufferBindTexture, cb, (uint64_t)dp->tex[i].stage, dp->tex[i].slot, handle[i]);
    if (dp->mesh_indirect) NVN(nvnCommandBufferDrawMeshTasksIndirect, cb, indirect.addr);
    else if (dp->mesh) NVN(nvnCommandBufferDrawMeshTasks, cb, (uint64_t)dp->first, (uint64_t)dp->count);
    else if (dp->indirect) NVN(nvnCommandBufferDrawElementsIndirect, cb, (uint64_t)dp->prim, (uint64_t)dp->itype, ibuf.addr, indirect.addr);
    else if (dp->indexed) NVN(nvnCommandBufferDrawElements, cb, (uint64_t)dp->prim, (uint64_t)dp->itype, (uint64_t)dp->count, ibuf.addr);
    else NVN(nvnCommandBufferDrawArrays, cb, (uint64_t)dp->prim, (uint64_t)dp->first, (uint64_t)dp->count);
    uint64_t recorded = NVN(nvnCommandBufferEndRecording, cb);
    CHECK(recorded != 0);
    if (hold) {
        CHECK(g_nheld == 0);
        g_held[g_nheld++] = recorded;
        return;
    }
    uint64_t handles = galloc(8 * (uint64_t)(g_nheld + 1), 8);
    for (int i = 0; i < g_nheld; i++) wr64(handles + 8ull * (uint64_t)i, g_held[i]);
    wr64(handles + 8ull * (uint64_t)g_nheld, recorded);

    // submit; a game may write to its memory the moment QueueSubmitCommands returns, which is what --rewrite-uniform does
    uint64_t fence = galloc(0x40, 16);
    OK(NVN(nvnSyncInitialize, fence, g_dev));
    NVN(nvnQueueSubmitCommands, g_queue, (uint64_t)g_nheld + 1, handles);
    g_nheld = 0;
    NVN(nvnQueueFenceSync, g_queue, fence, 0, 0);
    if (o->rewrite) {
        for (int i = 0; i < dp->nubo; i++) {
            memset((void *)(uintptr_t)ubo[i].addr, 0, (size_t)ubo[i].size);
            NVN(nvnBufferFlushMappedRange, ubo[i].obj, 0, ubo[i].size);
        }
        printf("rewrote %d uniform buffers after the submission\n", dp->nubo);
    }
    uint32_t waited = (uint32_t)LOW32(NVN(nvnSyncWait, fence, SYNC_TIMEOUT_NS));
    CHECK(waited == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED || waited == NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED);

}

static void run_draw(const Opts *o) {
    static Draw d, d2;
    parse_draw(o->draw, &d);
    open_backend(o, o->pack);
    start_runtime();

    // the device, its modes, the queue and the pools
    uint64_t db = galloc(0x40, 16);
    g_dev = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, db);
    OK(NVN(nvnDeviceInitialize, g_dev, db));
    NVN(nvnDeviceSetDepthMode, g_dev, (uint64_t)o->depth_mode);
    NVN(nvnDeviceSetWindowOriginMode, g_dev, (uint64_t)o->origin);
    make_queue();
    g_cpu_pool = make_pool(CPU_POOL_BYTES, CPU_UNCACHED | GPU_CACHED | SHADER_CODE, &g_cpu_mem);
    g_cpu_used = COMMAND_BYTES;

    // the images: the colour target, the depth target, the draw's textures, in one pool the CPU cannot see; a volume case has
    // a second draw (bcsample) that reads the volume, with a target and textures of its own
    static Scene sc, sc2;
    if (d.bc_slice >= 0) sc.rt = plan_bc_volume(d.bc_arena, d.bc_slice, d.bc_fmt);
    else sc.rt = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA32F, d.w, d.h, 1);
    if (d.ntargets > 1) sc.rt1 = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA32F, d.w, d.h, 1);
    if (d.depth) sc.depth = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH32F, d.w, d.h, 1);
    for (int i = 0; i < d.ntex; i++) {
        uint32_t layers = d.tex[i].target == NVN_TEXTURE_TARGET_CUBEMAP ? 6 : d.tex[i].d;
        sc.tex[i] = plan_texture(d.tex[i].target, NVN_FORMAT_RGBA32F, d.tex[i].w, d.tex[i].h, layers);
    }
    if (d.bc_sample[0]) {
        char path[1024];
        CHECK(d.bc_slice >= 0 && snprintf(path, sizeof path, "%s%s", g_dir, d.bc_sample) < (int)sizeof path);
        parse_draw(path, &d2);
        CHECK(d2.bc_slice < 0 && d2.ntargets == 1 && !d2.depth && d2.ntex == 1 && is_volume(&d2.tex[0]));
        sc2.rt = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA32F, d2.w, d2.h, 1);
    }
    g_gpu_pool = make_pool(align_up(g_gpu_used ? g_gpu_used : 1, 0x10000), CPU_NO_ACCESS | GPU_CACHED, NULL);   // none: an arena target
    init_texture(&sc.rt);
    if (d.ntargets > 1) init_texture(&sc.rt1);
    if (d.depth) init_texture(&sc.depth);
    for (int i = 0; i < d.ntex; i++) init_texture(&sc.tex[i]);
    if (d.bc_sample[0]) init_texture(&sc2.rt);

    play_draw(o, &d, &sc, NULL, -1, d.bc_second < 0 && d.bc_same);
    if (d.bc_second >= 0) {   // the next pass of the volume, with the CPU's store to its pages between the two if the case says
        if (d.bc_store) {
            volatile uint8_t *at = (volatile uint8_t *)(uintptr_t)(g_cpu_mem + sc.rt.off + sc.rt.size - 0x800);
            *at = 0x5A;
            NVN(nvnMemoryPoolFlushMappedRange, g_cpu_pool, sc.rt.off + sc.rt.size - 0x1000, 0x1000);
            printf("cpu store into the volume's last page\n");
        }
        play_draw(o, &d, &sc, NULL, d.bc_second, d.bc_same);
    }
    if (d.bc_sample[0]) {   // a draw that samples the volume through its own format, into a float target
        GTex vol = sc.rt;
        vol.view = galloc(16, 16);
        NVN(nvnTextureViewSetDefaults, vol.view);
        play_draw(o, &d2, &sc2, &vol, -1, 0);
        read_target(&sc2.rt, 16, o->out);
    }
    GTex rt = sc.rt, rt1 = sc.rt1, depth = sc.depth;
    if (d.bc_slice >= 0) {
        CHECK(o->bc_out);
        read_bc_volume(&rt, o->bc_out);
    } else if (!d.bc_sample[0]) read_target(&rt, 16, o->out);
    if (d.ntargets > 1) {
        CHECK(o->out1);
        read_target(&rt1, 16, o->out1);
    }
    if (o->depth_out) {
        CHECK(d.depth);
        read_target(&depth, 4, o->depth_out);
    }
    print_stats();

    // close the way the device tests do: the recorder, the pipelines, the device, then the layer's counts
    gfx_backend()->finish(0);
    vkrec_idle();
    vkrec_close();
    vkp_close();
    vkp_store_close();
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
    printf("ok vk_replay_draw\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) usage("a case is needed");
    if (!strcmp(argv[1], "probe")) {
        gpu_vk_open();
        gpu_vk_close();
        printf("ok probe\n");
        return 0;
    }
    if (strcmp(argv[1], "draw") || argc < 4) usage("unknown case");
    Opts o = {.draw = argv[2], .out = argv[3], .depth_mode = NVN_DEPTH_MODE_NEAR_IS_MINUS_W,
              .origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT};
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) o.pack = argv[++i];
        else if (!strcmp(argv[i], "--depth-out") && i + 1 < argc) o.depth_out = argv[++i];
        else if (!strcmp(argv[i], "--out1") && i + 1 < argc) o.out1 = argv[++i];
        else if (!strcmp(argv[i], "--bc-out") && i + 1 < argc) o.bc_out = argv[++i];
        else if (!strcmp(argv[i], "--no-features")) o.no_features = 1;
        else if (!strcmp(argv[i], "--depth-mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "near_is_minus_w")) o.depth_mode = NVN_DEPTH_MODE_NEAR_IS_MINUS_W;
            else if (!strcmp(m, "near_is_zero")) o.depth_mode = NVN_DEPTH_MODE_NEAR_IS_ZERO;
            else usage("--depth-mode takes near_is_minus_w or near_is_zero");
        } else if (!strcmp(argv[i], "--origin") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "lower_left")) o.origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT;
            else if (!strcmp(m, "upper_left")) o.origin = NVN_WINDOW_ORIGIN_MODE_UPPER_LEFT;
            else usage("--origin takes lower_left or upper_left");
        } else if (!strcmp(argv[i], "--imports")) o.imports = 1;
        else if (!strcmp(argv[i], "--no-snapshot")) o.no_snapshot = 1;
        else if (!strcmp(argv[i], "--no-flip")) o.no_flip = 1;
        else if (!strcmp(argv[i], "--rewrite-uniform")) o.rewrite = 1;
        else if (!strcmp(argv[i], "--cold")) o.cold = 1;
        else if (!strcmp(argv[i], "--unpad")) o.unpad = 1;
        else usage("unknown argument");
    }
    if (!o.pack) usage("--pack is needed");
    vkp_unpad = o.unpad;
    run_draw(&o);
    return 0;
}
