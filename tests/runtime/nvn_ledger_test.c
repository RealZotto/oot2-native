// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the ledger's writer on the NVN2 front end and the null back end (oot2/ledger.h, architecture D21),
// driven by tests/test_runtime_nvn_ledger.py: `nvn_ledger_test <case> [folder]`. Each case runs in its own process. A
// passing case prints "ok <case>" after the lines the driver checks: "key <kind> <stage key>" for each stage of a
// program and "row <json>" for each ledger row in the order it was added.
//
// Entry points are called the way generated code calls them (their synthetic address, arguments in the AAPCS64
// registers, objects in guest memory). Programs are synthetic control sections and code areas written here, textures
// are bytes in the front end's layout, nothing is read from a dump. Recordings go through the front end to the null
// back end, which runs the ledger's tracker on every record it replays.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/gfx.h"
#include "oot2/ledger.h"
#include "oot2/memory.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

typedef unsigned long long ull;
#define MiB (1ull << 20)
#define HEAP_BYTES (64 * MiB)
#define POOL_SIZE (4 * MiB)
#define CPU_UNCACHED 0x2u
#define GPU_CACHED 0x20u
#define MB MOD_MAIN_BASE
#define LOW32(v) ((v) & 0xffffffffu)

static Cpu c;
static uint64_t heap, heap_used, loader;
static uint64_t g_dev, g_pool, g_pool_mem, g_cmd_off, g_code_n, g_tp, g_queue;

static void stand_in(Cpu *cpu) { (void)cpu; }
static const GuestFn g_fns[1] = {stand_in};
static uint64_t g_addrs[1];

static void setup(void) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512 * MiB, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256 * MiB, .system_resource_size = MiB};
    rt_mem_init(&m);
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    g_addrs[0] = MB;
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, 1, TEST_END};
    rt_dispatch_init(mods, 1);
    c.x[0] = 0, c.x[1] = HEAP_BYTES;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    c.excl_addr = ~0ull;
}

static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP_BYTES);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}
static void wr64(uint64_t a, uint64_t v) { memcpy((void *)(uintptr_t)a, &v, 8); }

static uint64_t vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}
static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[96];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) {
        size_t len = strlen("nvnDeviceGetProcAddress") + 1;
        uint64_t s = galloc(len, 16);
        memcpy((void *)(uintptr_t)s, "nvnDeviceGetProcAddress", len);
        c.x[0] = s, c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
    }
    CHECK(gpa != 0 && n < 96);
    size_t len = strlen(name) + 1;
    uint64_t s = galloc(len, 16);
    memcpy((void *)(uintptr_t)s, name, len);
    c.x[0] = 0, c.x[1] = s;
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0);
    cache[n].name = name, cache[n].addr = c.x[0];
    return cache[n++].addr;
}
#define NVN(name, ...) vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t))

// ---------- objects, as the game sets them up
static uint64_t memory_pool(uint64_t size, uint64_t *storage) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16), mem = galloc(size, 0x10000);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetDevice, b, g_dev);
    NVN(nvnMemoryPoolBuilderSetStorage, b, mem, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, CPU_UNCACHED | GPU_CACHED);
    CHECK((NVN(nvnMemoryPoolInitialize, o, b) & 0xff) == 1);
    *storage = mem;
    return o;
}

static uint64_t queue(void) {
    uint64_t b = galloc(0x40, 16), q = galloc(0x2000, 16);
    NVN(nvnQueueBuilderSetDevice, b, g_dev);
    NVN(nvnQueueBuilderSetDefaults, b);
    NVN(nvnQueueBuilderSetCommandMemorySize, b, 0x10000);
    NVN(nvnQueueBuilderSetComputeMemorySize, b, 0x40000);
    NVN(nvnQueueBuilderSetControlMemorySize, b, 0x4000);
    uint64_t size = NVN(nvnQueueBuilderGetQueueMemorySize, b);
    NVN(nvnQueueBuilderSetQueueMemory, b, galloc(size, 0x1000), size);
    CHECK((NVN(nvnQueueInitialize, q, b) & 0xff) == 1);
    return q;
}

static uint64_t command_buffer(void) {
    uint64_t o = galloc(0xC0, 16);
    CHECK((NVN(nvnCommandBufferInitialize, o, g_dev) & 0xff) == 1);
    NVN(nvnCommandBufferAddCommandMemory, o, g_pool, g_cmd_off, 0x10000);
    g_cmd_off += 0x10000;
    NVN(nvnCommandBufferAddControlMemory, o, galloc(0x10000, 16), 0x10000);
    return o;
}

static uint64_t g_serial = 100;
// A texture, as the front end's TextureInitialize leaves it: the bytes are what the records and the ledger read.
static uint64_t texture(int target, int format) {
    uint64_t t = galloc(0x100, 16);
    tex_t *x = (tex_t *)(uintptr_t)t;
    x->magic = NVN_TEX_MAGIC, x->serial = ++g_serial;
    x->d.target = target, x->d.format = format, x->d.width = 16, x->d.height = 16, x->d.depth = 1, x->d.levels = 1;
    return t;
}
static uint64_t view_target(int target) {
    uint64_t v = galloc(16, 16);
    NVN(nvnTextureViewSetDefaults, v);
    NVN(nvnTextureViewSetTarget, v, (uint64_t)target);
    return v;
}
static void register_tex(int id, uint64_t tex, uint64_t view) { NVN(nvnTexturePoolRegisterTexture, g_tp, (uint64_t)id, tex, view); }
static uint64_t handle(int id) { return NVN(nvnDeviceGetTextureHandle, g_dev, (uint64_t)id, 0); }
static uint64_t image_handle(int id) { return NVN(nvnDeviceGetImageHandle, g_dev, (uint64_t)id); }

// A program of the stages named by kind ('v' vertex, 'p' pixel, 'c' compute, 't' task, 'm' mesh), each with its own code so its key is its own.
static uint64_t program(const char *kinds, int tag) {
    uint64_t sd = galloc(16 * 6, 16), p = galloc(0xC0, 16);
    int n = 0;
    for (; kinds[n]; n++) {
        uint8_t ctl[0xE0] = {0}, code[0x200] = {0};
        uint32_t words[][2] = {{0, 0x98761234u}, {4, 0xC9}, {0x38, 0x90}, {0x40, 0x100}, {0x44, sizeof code}};
        for (size_t i = 0; i < sizeof words / sizeof *words; i++) memcpy(ctl + words[i][0], &words[i][1], 4);
        uint32_t sph = kinds[n] == 'p' ? (5u << 10 | 1) : (1u << 10 | 1);
        memcpy(code, &sph, 4);
        if (kinds[n] == 'c' || kinds[n] == 't' || kinds[n] == 'm') {
            ctl[0x56] = kinds[n] == 'c' ? 5 : kinds[n] == 't' ? 6 : 7;
            uint32_t local[3] = {8, 8, 1};
            memcpy(ctl + 0x58, local, sizeof local);
        }
        uint32_t t = (uint32_t)tag * 16u + (uint32_t)n;
        memcpy(code + 0x100, &t, 4);
        uint64_t at = g_pool_mem + 0x100000 + 0x400 * g_code_n++;
        memcpy((void *)(uintptr_t)at, code, sizeof code);
        uint64_t ctl_at = galloc(sizeof ctl, 16);
        memcpy((void *)(uintptr_t)ctl_at, ctl, sizeof ctl);
        wr64(sd + 16ull * (uint64_t)n, at);
        wr64(sd + 16ull * (uint64_t)n + 8, ctl_at);
    }
    CHECK((NVN(nvnProgramInitialize, p, g_dev) & 0xff) == 1);
    CHECK((NVN(nvnProgramSetShaders, p, (uint64_t)n, sd) & 0xff) == 1);
    const prog_t *x = (const prog_t *)(uintptr_t)p;
    for (int i = 0; i < n; i++) printf("key %s %s\n", nvn_stage_kind_name(x->st[i].id->kind), x->st[i].id->key);
    return p;
}

// Vertex attribute states: (format, offset) pairs, bound at index 0.
static uint64_t attribs(const int *formats, int n) {
    uint64_t a = galloc(4 * (uint64_t)n, 16);
    for (int i = 0; i < n; i++) {
        uint64_t s = a + 4ull * (uint64_t)i;
        NVN(nvnVertexAttribStateSetDefaults, s);
        NVN(nvnVertexAttribStateSetFormat, s, (uint64_t)formats[i], 4ull * (uint64_t)i);
    }
    return a;
}

static uint64_t submit(uint64_t handle_) {
    uint64_t a = galloc(8, 8);
    wr64(a, handle_);
    NVN(nvnQueueSubmitCommands, g_queue, 1, a);
    return ((const queue_t *)(uintptr_t)g_queue)->last;
}
static uint64_t record_begin(uint64_t cb) { NVN(nvnCommandBufferBeginRecording, cb); return cb; }
static uint64_t record_end(uint64_t cb) { return NVN(nvnCommandBufferEndRecording, cb); }

static void rows(void) {
    for (uint64_t i = 0; ledger_row_text(i); i++) printf("row %s\n", ledger_row_text(i));
}

// ---------- the scene every case draws: textures registered, the render targets, a graphics and a compute program
static struct {
    uint64_t prog, comp, rt_color, rt_depth, attr, tex_array, tex_3d, tex_2d, tex_cube_view, tex_img, tex_img_c, tex_cubea;
    uint64_t colors, depth;
} S;
enum { ID_ARRAY = 1, ID_3D, ID_2D, ID_CUBE, ID_IMG, ID_IMG_C, ID_CUBEA };

static void scene(void) {
    setup();
    uint64_t b = galloc(0x40, 16);
    g_dev = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, b);
    CHECK((NVN(nvnDeviceInitialize, g_dev, b) & 0xff) == 1);
    g_pool = memory_pool(POOL_SIZE, &g_pool_mem);
    g_tp = galloc(0x40, 16);
    CHECK((NVN(nvnTexturePoolInitialize, g_tp, g_pool, 0x8000, 64) & 0xff) == 1);
    g_queue = queue();
    // Descriptors: a 2D array, a 3D, a 2D, a 2D seen as a cube by its view, a 2D R32F image, a 1D array RGBA32UI
    // image, a cube array
    S.tex_array = texture(NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8);
    S.tex_3d = texture(NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGBA8);
    S.tex_2d = texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8);
    S.tex_img = texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_R32F);
    S.tex_img_c = texture(NVN_TEXTURE_TARGET_1D_ARRAY, NVN_FORMAT_RGBA32UI);
    S.tex_cubea = texture(NVN_TEXTURE_TARGET_CUBEMAP_ARRAY, NVN_FORMAT_RGBA8);
    register_tex(ID_ARRAY, S.tex_array, 0);
    register_tex(ID_3D, S.tex_3d, 0);
    register_tex(ID_2D, S.tex_2d, 0);
    register_tex(ID_CUBE, S.tex_2d, view_target(NVN_TEXTURE_TARGET_CUBEMAP));
    register_tex(ID_IMG, S.tex_img, 0);
    register_tex(ID_IMG_C, S.tex_img_c, 0);
    register_tex(ID_CUBEA, S.tex_cubea, 0);
    // Render targets: a uint colour target and a float one, and a depth-stencil target
    S.rt_color = texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_R32UI);
    uint64_t rt1 = texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8);
    S.rt_depth = texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH24_STENCIL8);
    S.colors = galloc(16, 8);
    wr64(S.colors, S.rt_color), wr64(S.colors + 8, rt1);
    static const int formats[4] = {NVN_FORMAT_RGB32F, NVN_FORMAT_RG16I, NVN_FORMAT_RGBA8, NVN_FORMAT_RGBA32UI};
    S.attr = attribs(formats, 4);
    S.prog = program("vp", 1);
    S.comp = program("c", 2);
}

// Everything a draw but its textures needs, bound in the recording of cb.
static void bind_scene(uint64_t cb) {
    NVN(nvnCommandBufferSetTexturePool, cb, g_tp);
    NVN(nvnCommandBufferBindProgram, cb, S.prog, 3);
    NVN(nvnCommandBufferBindVertexAttribState, cb, 4, S.attr);
    NVN(nvnCommandBufferSetRenderTargets, cb, 2, S.colors, 0, S.rt_depth, 0);
}

static void bind_textures(uint64_t cb) {
    NVN(nvnCommandBufferBindTexture, cb, 0, 0, handle(ID_3D));      // the vertex stage's index 0 is a 3D texture
    NVN(nvnCommandBufferBindTexture, cb, 0, 3, handle(ID_2D));
    NVN(nvnCommandBufferBindTexture, cb, 0, 10, handle(ID_ARRAY));   // "10" sorts between "0" and "3" in the answers
    NVN(nvnCommandBufferBindTexture, cb, 1, 0, handle(ID_ARRAY));   // the fragment stage's index 0 is a 2D array
    NVN(nvnCommandBufferBindTexture, cb, 1, 5, handle(ID_CUBE));
    NVN(nvnCommandBufferBindImage, cb, 1, 2, image_handle(ID_IMG));
}

static uint64_t draw_recording(uint64_t cb) {
    record_begin(cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    return record_end(cb);
}

// ---------- case state: a draw, a draw again, a draw after a change, a dispatch, state held across recordings
static void case_state(void) {
    scene();
    uint64_t cb = command_buffer();
    record_begin(cb);
    bind_scene(cb);
    bind_textures(cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    uint64_t h1 = record_end(cb);
    submit(h1);
    CHECK(ledger_rows() == 2);
    submit(h1);   // the same draw again: no new row
    CHECK(ledger_rows() == 2);
    // A dispatch: the compute stage's own tex and image answers
    record_begin(cb);
    NVN(nvnCommandBufferBindProgram, cb, S.comp, 0x20);
    NVN(nvnCommandBufferBindTexture, cb, 5, 1, handle(ID_CUBEA));
    NVN(nvnCommandBufferBindImage, cb, 5, 0, image_handle(ID_IMG_C));
    NVN(nvnCommandBufferDispatchCompute, cb, 1, 1, 1);
    submit(record_end(cb));
    CHECK(ledger_rows() == 3);
    // The graphics state is still bound (one channel): a draw after a texture changed, in another recording, makes a new
    // pixel row, and the vertex row it shares is the one already there
    record_begin(cb);
    NVN(nvnCommandBufferBindTexture, cb, 1, 0, handle(ID_2D));
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(record_end(cb));
    CHECK(ledger_rows() == 4);
    rows();
    printf("ok state\n");
}

// ---------- case call: a recording called from another gives the caller's state, and the caller sees the callee's
static void case_call(void) {
    scene();
    uint64_t cb = command_buffer(), ca = command_buffer(), cc = command_buffer();
    // A draws and binds nothing. Alone it has no program: no row
    uint64_t ha = draw_recording(ca);
    submit(ha);
    CHECK(ledger_rows() == 0);
    // B binds the scene, then calls A
    record_begin(cb);
    bind_scene(cb);
    bind_textures(cb);
    uint64_t arr = galloc(8, 8);
    wr64(arr, ha);
    NVN(nvnCommandBufferCallCommands, cb, 1, arr);
    submit(record_end(cb));
    CHECK(ledger_rows() == 2);
    // D binds a texture at the fragment stage's index 7 and is called from C, which then draws: the callee's state is
    // the caller's too
    uint64_t cd = command_buffer();
    record_begin(cd);
    NVN(nvnCommandBufferBindTexture, cd, 1, 7, handle(ID_CUBEA));
    uint64_t hd = record_end(cd);
    record_begin(cc);
    wr64(arr, hd);
    NVN(nvnCommandBufferCallCommands, cc, 1, arr);
    NVN(nvnCommandBufferDrawArrays, cc, 4, 0, 3);
    submit(record_end(cc));
    CHECK(ledger_rows() == 3);
    rows();
    printf("ok call\n");
}

// ---------- case file: rows are written at a safe point, never on the frame path, and a new run adds to the file
static int g_refused = -2, g_in_submit;
static void flush_in_submit(uint64_t fence, int index, const op_hdr *r, void *ctx) {
    (void)fence, (void)index, (void)ctx;
    if (r->op == OP_DRAW && g_in_submit) g_refused = ledger_flush();
}
static int exists(const char *folder, const char *name) {
    char path[4200];
    snprintf(path, sizeof path, "%s/%s", folder, name);
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}
static void case_file(const char *folder) {
    scene();
    ledger_set_folder(folder);
    CHECK(!exists(folder, LEDGER_FILE));
    uint64_t cb = command_buffer();
    record_begin(cb);
    bind_scene(cb);
    bind_textures(cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    uint64_t h = record_end(cb);
    g_in_submit = 1;
    gfx_null_log_ops(flush_in_submit, NULL);
    for (int frame = 0; frame < 3; frame++) submit(h);
    gfx_null_log_ops(NULL, NULL);
    g_in_submit = 0;
    // Nothing was written by the frames, and a flush asked for inside a submission was refused
    CHECK(ledger_rows() == 2 && ledger_writes() == 0 && !exists(folder, LEDGER_FILE));
    CHECK(g_refused == -1);
    CHECK(ledger_flush() == 1 && ledger_writes() == 1 && exists(folder, LEDGER_FILE) && !exists(folder, LEDGER_FILE ".tmp"));
    CHECK(ledger_flush() == 0 && ledger_writes() == 1);   // nothing new
    printf("first flush wrote %llu rows\n", (ull)ledger_rows());
    // A new run: the file's rows are loaded, a draw it already holds adds nothing, a new one is added to the file
    ledger_reset();
    CHECK(ledger_rows() == 0);
    ledger_set_folder(folder);
    CHECK(ledger_rows() == 2);
    submit(h);
    CHECK(ledger_rows() == 2 && ledger_flush() == 0);
    record_begin(cb);
    NVN(nvnCommandBufferBindTexture, cb, 1, 0, handle(ID_2D));
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(record_end(cb));
    CHECK(ledger_rows() == 3 && ledger_flush() == 1 && ledger_writes() == 1);
    printf("ok file\n");
}

// ---------- case more: indexed and indirect draws, a texture bound in the image table's range, an attribute with no
// format, an image whose format the translator has no storage format for
static void case_more(void) {
    scene();
    uint64_t srgb = texture(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8_SRGB);
    register_tex(8, srgb, 0);
    uint64_t cb = command_buffer();
    record_begin(cb);
    bind_scene(cb);
    bind_textures(cb);
    NVN(nvnCommandBufferBindTexture, cb, 1, 33, handle(ID_2D));   // the image table's range: never a "tex" answer
    NVN(nvnCommandBufferDrawElements, cb, 4, 1, 3, 0x1000);
    submit(record_end(cb));
    CHECK(ledger_rows() == 2);
    record_begin(cb);
    NVN(nvnCommandBufferBindTexture, cb, 1, 0, handle(ID_2D));
    NVN(nvnCommandBufferDrawElementsIndirect, cb, 4, 1, 0x2000);
    submit(record_end(cb));
    CHECK(ledger_rows() == 3);
    static const int formats[4] = {NVN_FORMAT_RGB32F, NVN_FORMAT_NONE, NVN_FORMAT_RG16I, NVN_FORMAT_NONE};
    record_begin(cb);
    NVN(nvnCommandBufferBindVertexAttribState, cb, 4, attribs(formats, 4));
    NVN(nvnCommandBufferBindImage, cb, 1, 1, image_handle(8));
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(record_end(cb));
    CHECK(ledger_rows() == 5);
    rows();
    printf("ok more\n");
}

// ---------- case mesh: a task, mesh and pixel program drawn directly and indirectly
static void case_mesh(void) {
    scene();
    uint64_t mp = program("tmp", 3);
    uint64_t cb = command_buffer();
    record_begin(cb);
    NVN(nvnCommandBufferSetTexturePool, cb, g_tp);
    NVN(nvnCommandBufferBindProgram, cb, mp, 3);
    NVN(nvnCommandBufferSetRenderTargets, cb, 2, S.colors, 0, S.rt_depth, 0);
    NVN(nvnCommandBufferBindTexture, cb, 6, 1, handle(ID_3D));
    NVN(nvnCommandBufferBindTexture, cb, 7, 2, handle(ID_CUBE));
    NVN(nvnCommandBufferBindTexture, cb, 1, 0, handle(ID_ARRAY));
    NVN(nvnCommandBufferDrawMeshTasks, cb, 0, 1);
    submit(record_end(cb));
    CHECK(ledger_rows() == 3);
    record_begin(cb);
    NVN(nvnCommandBufferBindTexture, cb, 1, 0, handle(ID_2D));
    NVN(nvnCommandBufferDrawMeshTasksIndirect, cb, 0x3000);
    submit(record_end(cb));
    CHECK(ledger_rows() == 4);
    rows();
    printf("ok mesh\n");
}

// ---------- case stop: a run that ends through the stop path (ExitProcess, a fatal) writes the file
static void case_stop(const char *folder) {
    scene();
    ledger_set_folder(folder);
    ledger_flush_on_stop();
    uint64_t cb = command_buffer();
    record_begin(cb);
    bind_scene(cb);
    bind_textures(cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(record_end(cb));
    CHECK(ledger_rows() == 2 && ledger_writes() == 0 && !exists(folder, LEDGER_FILE));
    rt_run_stop_hook(0, "");
    CHECK(ledger_writes() == 1 && exists(folder, LEDGER_FILE));
    printf("ok stop\n");
}

static void case_digest(void) {
    LedgerTrack track = {0};
    track.tex[1][10] = 8;          // 3d
    track.img_type[1][2] = 4;      // 2d
    track.img_fmt[1][2] = 0;       // translator format "Unknown"
    track.fout[0] = 3;             // uint
    track.fout[1] = 1;             // float
    uint8_t digest[32];
    ledger_track_answers_digest(&track, NVN_STAGE_PS, digest);
    printf("digest ");
    for (size_t i = 0; i < sizeof digest; i++) printf("%02x", digest[i]);
    printf("\nok digest\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: nvn_ledger_test <case> [folder]\n");
        return 2;
    }
    if (!strcmp(argv[1], "state")) case_state();
    else if (!strcmp(argv[1], "call")) case_call();
    else if (!strcmp(argv[1], "more")) case_more();
    else if (!strcmp(argv[1], "mesh")) case_mesh();
    else if (!strcmp(argv[1], "stop") && argc == 3) case_stop(argv[2]);
    else if (!strcmp(argv[1], "file") && argc == 3) case_file(argv[2]);
    else if (!strcmp(argv[1], "digest")) case_digest();
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    return 0;
}
