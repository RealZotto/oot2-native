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
//   bctarget SLICE placed|arena     the colour target is slice SLICE of a 32x32x20 BC4 volume (8x8x20 blocks of 8 bytes),
//                                   viewed as RGBA16 UINT, so a texel of the target is a block: size must be 8 8, no depth,
//                                   no clear. The volume stands in the pool of images (placed) or in the CPU-visible pool
//                                   (an arena image, the game's case); --bc-out FILE gets its blocks as the image holds them
//                                   (8 blocks a row, 8 rows a slice, 20 slices) and "bc image placed|arena" is printed
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
    int bc_slice, bc_arena;    // bctarget: the colour target is slice bc_slice of a BC4 volume, viewed as RGBA16 UINT (-1: none)
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
    d->polygon = -1, d->nvp = 1, d->ntargets = 1, d->chmask[0] = d->chmask[1] = 0xF, d->bc_slice = -1;
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
            CHECK(sscanf(rest, "%u %127s", &u[0], a) == 2 && u[0] < 20 && (!strcmp(a, "arena") || !strcmp(a, "placed")));
            d->bc_slice = (int)u[0], d->bc_arena = !strcmp(a, "arena");
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
        CHECK(d->bc_slice < 0 || (d->ntargets == 1 && !d->depth && d->w == 8 && d->h == 8));   // the pass of a 32x32 level's blocks
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
static GTex plan_bc_volume(int arena, int slice) {
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
    NVN(nvnTextureViewSetFormat, t.view, NVN_FORMAT_RGBA16UI);
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

static void run_draw(const Opts *o) {
    static Draw d;
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

    // the images: the colour target, the depth target, the draw's textures, in one pool the CPU cannot see
    static GTex rt, rt1, depth, tex[MAX_TEX];
    if (d.bc_slice >= 0) rt = plan_bc_volume(d.bc_arena, d.bc_slice);
    else rt = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA32F, d.w, d.h, 1);
    if (d.ntargets > 1) rt1 = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA32F, d.w, d.h, 1);
    if (d.depth) depth = plan_texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH32F, d.w, d.h, 1);
    for (int i = 0; i < d.ntex; i++) {
        uint32_t layers = d.tex[i].target == NVN_TEXTURE_TARGET_CUBEMAP ? 6 : d.tex[i].d;
        tex[i] = plan_texture(d.tex[i].target, NVN_FORMAT_RGBA32F, d.tex[i].w, d.tex[i].h, layers);
    }
    g_gpu_pool = make_pool(align_up(g_gpu_used ? g_gpu_used : 1, 0x10000), CPU_NO_ACCESS | GPU_CACHED, NULL);   // none: an arena target
    init_texture(&rt);
    if (d.ntargets > 1) init_texture(&rt1);
    if (d.depth) init_texture(&depth);
    for (int i = 0; i < d.ntex; i++) init_texture(&tex[i]);

    // the descriptor tables: one entry of each pool for each texture, the texture's own sampler beside it
    uint64_t tp_off = cpu_take(POOL_IDS * 32, 256, 0), sp_off = cpu_take(POOL_IDS * 32, 256, 0);
    uint64_t tp = galloc(0x40, 16), sp = galloc(0x40, 16);
    OK(NVN(nvnTexturePoolInitialize, tp, g_cpu_pool, tp_off, POOL_IDS));
    OK(NVN(nvnSamplerPoolInitialize, sp, g_cpu_pool, sp_off, POOL_IDS));
    uint64_t handle[MAX_TEX];
    for (int i = 0; i < d.ntex; i++) {
        const Tex *t = &d.tex[i];
        CHECK(t->id > 0 && t->id < POOL_IDS);
        write_texels(&tex[i], t->file);
        NVN(nvnTexturePoolRegisterTexture, tp, t->id, tex[i].tex, tex[i].view);
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
    uint64_t prog = make_program(&d, ids);
    VkpFormats formats = {(uint32_t)d.ntargets,
                          {d.bc_slice >= 0 ? VK_FORMAT_R16G16B16A16_UINT : VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
                          d.depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkpStage vs = {.key = ids[ST_VS] ? ids[ST_VS]->key : NULL, .shape = ids[ST_VS]};
    VkpStage task = {.key = ids[ST_TS] ? ids[ST_TS]->key : NULL, .shape = ids[ST_TS]};
    VkpStage mesh = {.key = ids[ST_MS] ? ids[ST_MS]->key : NULL, .shape = ids[ST_MS]};
    VkpStage ps = {.key = ids[ST_PS]->key, .shape = ids[ST_PS]};
    if (!o->cold) CHECK(vkp_graphics(d.mesh ? &mesh : &vs, d.mesh ? &task : NULL, &ps, &formats) != VK_NULL_HANDLE);
    else CHECK(vkp_find_graphics(d.mesh ? &mesh : &vs, d.mesh ? &task : NULL, &ps, &formats) == VK_NULL_HANDLE);

    // the buffers
    Buf vbuf[4], ibuf = {0}, ubo[MAX_UBO], sbo[MAX_SBO];
    for (int i = 0; i < d.nvbuf; i++) {
        size_t n;
        void *data = read_file(d.vbuf[i].file, &n);
        vbuf[i] = make_buffer(data, n, 256, 0);
        free(data);
    }
    Buf indirect = {0};
    if (d.indexed) {
        size_t n;
        void *data = read_file(d.ibuf, &n);
        ibuf = make_buffer(data, n, 256, 0);
        free(data);
    }
    if (d.mesh_indirect) {
        const uint32_t record[2] = {(uint32_t)d.count, (uint32_t)d.first};
        indirect = make_buffer(record, sizeof record, 256, 0);
    } else if (d.indirect) {
        const uint32_t record[5] = {(uint32_t)d.count, 1, 0, 0, (uint32_t)d.base_instance};
        indirect = make_buffer(record, sizeof record, 256, 0);
    }
    for (int i = 0; i < d.nubo; i++) {
        size_t n;
        void *data = read_file(d.ubo[i].file, &n);
        ubo[i] = make_buffer(data, n, 256, 0);
        free(data);
    }
    for (int i = 0; i < d.nsbo; i++) {
        size_t n;
        void *data = read_file(d.sbo[i].file, &n);
        sbo[i] = make_buffer(data, n, 256, d.sbo[i].low);
        free(data);
    }
    NVN(nvnMemoryPoolFlushMappedRange, g_cpu_pool, 0, g_cpu_used);

    // vertex state: an attribute at each location (the state is indexed by location) and the streams
    int nloc = 0;
    for (int i = 0; i < d.nattr; i++)
        if ((int)d.attr[i].loc + 1 > nloc) nloc = (int)d.attr[i].loc + 1;
    uint64_t attrs = galloc(4 * (uint64_t)(nloc ? nloc : 1), 16), streams = galloc(8 * (uint64_t)(d.nstream ? d.nstream : 1), 16);
    for (int i = 0; i < nloc; i++) NVN(nvnVertexAttribStateSetDefaults, attrs + 4ull * (uint64_t)i);
    for (int i = 0; i < d.nattr; i++) {
        uint64_t s = attrs + 4ull * d.attr[i].loc;
        NVN(nvnVertexAttribStateSetFormat, s, (uint64_t)d.attr[i].format, d.attr[i].offset);
        NVN(nvnVertexAttribStateSetStreamIndex, s, d.attr[i].stream);
    }
    for (int i = 0; i < d.nstream; i++) {
        uint64_t s = streams + 8ull * (uint64_t)i;
        NVN(nvnVertexStreamStateSetDefaults, s);
        NVN(nvnVertexStreamStateSetStride, s, d.stream[i].stride);
        NVN(nvnVertexStreamStateSetDivisor, s, d.stream[i].divisor);
    }
    uint64_t ds_state = galloc(0x10, 16);
    if (d.depth) {
        NVN(nvnDepthStencilStateSetDefaults, ds_state);
        NVN(nvnDepthStencilStateSetDepthTestEnable, ds_state, 1);
        NVN(nvnDepthStencilStateSetDepthWriteEnable, ds_state, 1);
        NVN(nvnDepthStencilStateSetDepthFunc, ds_state, NVN_DEPTH_FUNC_LESS);
    }

    // the recording: what a game records for one draw, in the order it would
    uint64_t cb = galloc(0xC0, 16);
    OK(NVN(nvnCommandBufferInitialize, cb, g_dev));
    NVN(nvnCommandBufferAddCommandMemory, cb, g_cpu_pool, 0, COMMAND_BYTES);
    NVN(nvnCommandBufferAddControlMemory, cb, galloc(CONTROL_BYTES, 16), CONTROL_BYTES);
    NVN(nvnCommandBufferBeginRecording, cb);
    uint64_t rts = galloc(16, 8), rgba = galloc(16, 16);
    wr64(rts, rt.tex);
    if (d.ntargets > 1) wr64(rts + 8, rt1.tex);
    memcpy((void *)(uintptr_t)rgba, d.clear, sizeof d.clear);
    uint64_t rt_views = 0;
    if (d.bc_slice >= 0) wr64(rt_views = galloc(8, 8), rt.view);
    NVN(nvnCommandBufferSetRenderTargets, cb, (uint64_t)d.ntargets, rts, rt_views, d.depth ? depth.tex : 0, 0);
    NVN(nvnCommandBufferSetViewport, cb, (uint64_t)(int64_t)d.vp[0], (uint64_t)(int64_t)d.vp[1], (uint64_t)(int64_t)d.vp[2],
        (uint64_t)(int64_t)d.vp[3]);
    NVN(nvnCommandBufferSetScissor, cb, (uint64_t)(int64_t)d.sc[0], (uint64_t)(int64_t)d.sc[1], (uint64_t)(int64_t)d.sc[2],
        (uint64_t)(int64_t)d.sc[3]);
    if (d.nvp > 1) {   // the draw's viewport and scissor first, then quarters of the target, each in its own corner
        uint64_t vps = galloc(16ull * (uint64_t)d.nvp, 16), scs = galloc(16ull * (uint64_t)d.nvp, 16);
        for (int i = 0; i < d.nvp; i++) {
            float *v = (float *)(uintptr_t)(vps + 16ull * (uint64_t)i);
            int32_t *k = (int32_t *)(uintptr_t)(scs + 16ull * (uint64_t)i);
            if (i == 0) {
                v[0] = (float)d.vp[0], v[1] = (float)d.vp[1], v[2] = (float)d.vp[2], v[3] = (float)d.vp[3];
                k[0] = d.sc[0], k[1] = d.sc[1], k[2] = d.sc[2], k[3] = d.sc[3];
            } else {
                v[0] = (float)((i & 1) * (int)d.w / 2), v[1] = (float)((i & 2) * (int)d.h / 4);
                v[2] = (float)d.w / 2, v[3] = (float)d.h / 2;
                k[0] = (int32_t)v[0], k[1] = (int32_t)v[1], k[2] = (int32_t)d.w / 2, k[3] = (int32_t)d.h / 2;
            }
        }
        NVN(nvnCommandBufferSetViewports, cb, 0, (uint64_t)d.nvp, vps);
        NVN(nvnCommandBufferSetScissors, cb, 0, (uint64_t)d.nvp, scs);
    }
    if (d.depth) {
        fv(0, d.dnear), fv(1, d.dfar);
        NVN(nvnCommandBufferSetDepthRange, cb);
        NVN(nvnCommandBufferBindDepthStencilState, cb, ds_state);
    }
    if (d.polygon >= 0) {
        uint64_t ps = galloc(0x10, 16);
        NVN(nvnPolygonStateSetDefaults, ps);
        NVN(nvnPolygonStateSetPolygonMode, ps, (uint64_t)d.polygon);
        NVN(nvnPolygonStateSetPolygonOffsetEnables, ps, (uint64_t)d.offsets);
        NVN(nvnCommandBufferBindPolygonState, cb, ps);
    }
    if (d.has_offset) {
        fv(0, d.offset[0]), fv(1, d.offset[1]), fv(2, d.offset[2]);
        NVN(nvnCommandBufferSetPolygonOffsetClamp, cb);
    }
    if (d.clamp) NVN(nvnCommandBufferSetDepthClamp, cb, 1);
    if (d.bounds) {
        fv(0, d.bmin), fv(1, d.bmax);
        NVN(nvnCommandBufferSetDepthBounds, cb, 1);
    }
    if (d.line_width > 0.0f) {
        fv(0, d.line_width);
        NVN(nvnCommandBufferSetLineWidth, cb);
    }
    if (d.has_chmask) {
        uint64_t ms = galloc(0x10, 16);
        NVN(nvnChannelMaskStateSetDefaults, ms);
        for (int i = 0; i < d.ntargets; i++)
            NVN(nvnChannelMaskStateSetChannelMask, ms, (uint64_t)i, d.chmask[i] & 1, (d.chmask[i] >> 1) & 1,
                (d.chmask[i] >> 2) & 1, (d.chmask[i] >> 3) & 1);
        NVN(nvnCommandBufferBindChannelMaskState, cb, ms);
    }
    if (d.dual) {
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
    if (d.bc_slice < 0) NVN(nvnCommandBufferClearColor, cb, 0, rgba, 0xF);   // a float clear of a UINT target means nothing
    if (d.ntargets > 1) NVN(nvnCommandBufferClearColor, cb, 1, rgba, 0xF);
    if (d.depth) {
        fv(0, d.dclear);
        NVN(nvnCommandBufferClearDepthStencil, cb, 1, 0, 0);
    }
    uint64_t stage_mask = d.mesh ? (NVN_SHADER_STAGE_FRAGMENT_BIT | (1u << NVN_STAGE_TASK) | (1u << NVN_STAGE_MESH))
                                 : (NVN_SHADER_STAGE_VERTEX_BIT | NVN_SHADER_STAGE_FRAGMENT_BIT);
    NVN(nvnCommandBufferBindProgram, cb, prog, stage_mask);
    if (!d.mesh) {
        NVN(nvnCommandBufferBindVertexAttribState, cb, (uint64_t)nloc, attrs);
        NVN(nvnCommandBufferBindVertexStreamState, cb, (uint64_t)d.nstream, streams);
        for (int i = 0; i < d.nvbuf; i++)
            NVN(nvnCommandBufferBindVertexBuffer, cb, d.vbuf[i].index, vbuf[i].addr, vbuf[i].size);
    }
    for (int i = 0; i < d.nubo; i++)
        NVN(nvnCommandBufferBindUniformBuffer, cb, (uint64_t)d.ubo[i].stage, d.ubo[i].r, ubo[i].addr, ubo[i].size);
    for (int i = 0; i < d.nsbo; i++)
        NVN(nvnCommandBufferBindStorageBuffer, cb, (uint64_t)d.sbo[i].stage, d.sbo[i].r, sbo[i].addr, sbo[i].size);
    for (int i = 0; i < d.ntex; i++)
        NVN(nvnCommandBufferBindTexture, cb, (uint64_t)d.tex[i].stage, d.tex[i].slot, handle[i]);
    if (d.mesh_indirect) NVN(nvnCommandBufferDrawMeshTasksIndirect, cb, indirect.addr);
    else if (d.mesh) NVN(nvnCommandBufferDrawMeshTasks, cb, (uint64_t)d.first, (uint64_t)d.count);
    else if (d.indirect) NVN(nvnCommandBufferDrawElementsIndirect, cb, (uint64_t)d.prim, (uint64_t)d.itype, ibuf.addr, indirect.addr);
    else if (d.indexed) NVN(nvnCommandBufferDrawElements, cb, (uint64_t)d.prim, (uint64_t)d.itype, (uint64_t)d.count, ibuf.addr);
    else NVN(nvnCommandBufferDrawArrays, cb, (uint64_t)d.prim, (uint64_t)d.first, (uint64_t)d.count);
    uint64_t handles = galloc(8, 8);
    wr64(handles, NVN(nvnCommandBufferEndRecording, cb));
    CHECK(((const uint64_t *)(uintptr_t)handles)[0] != 0);

    // submit; a game may write to its memory the moment QueueSubmitCommands returns, which is what --rewrite-uniform does
    uint64_t fence = galloc(0x40, 16);
    OK(NVN(nvnSyncInitialize, fence, g_dev));
    NVN(nvnQueueSubmitCommands, g_queue, 1, handles);
    NVN(nvnQueueFenceSync, g_queue, fence, 0, 0);
    if (o->rewrite) {
        for (int i = 0; i < d.nubo; i++) {
            memset((void *)(uintptr_t)ubo[i].addr, 0, (size_t)ubo[i].size);
            NVN(nvnBufferFlushMappedRange, ubo[i].obj, 0, ubo[i].size);
        }
        printf("rewrote %d uniform buffers after the submission\n", d.nubo);
    }
    uint32_t waited = (uint32_t)LOW32(NVN(nvnSyncWait, fence, SYNC_TIMEOUT_NS));
    CHECK(waited == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED || waited == NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED);

    if (d.bc_slice >= 0) {
        CHECK(o->bc_out);
        read_bc_volume(&rt, o->bc_out);
    } else read_target(&rt, 16, o->out);
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
