// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the NVN2 front end's command buffers (architecture D11), driven by tests/test_runtime_nvn_cmd.py:
// `nvn_cmd_test <case> [arguments]`. Each case runs in its own process. A passing case prints "ok <case>"; a failing
// case must end in one named fatal, which the driver checks.
//
// Entry points are called the way generated code calls them: rt_call at their synthetic address, arguments in the
// AAPCS64 registers, the objects in guest memory. The game's inline BeginRecording is played by inline_begin, from the
// console driver's offsets alone, not from cmd_t. The game's memory callback is a guest function of the harness that
// logs its arguments and adds memory. Recordings are submitted to the null back end, whose op log prints each record.
// Textures, programs and events are bytes the harness writes in the front end's layouts; nothing comes from a dump.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "nvn_cmd.h"
#include "oot2/gfx.h"
#include "oot2/memory.h"
#include "oot2/nvn_pace.h"
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
#define HEAP (64 * MiB)
#define POOL_SIZE (4 * MiB)
#define CPU_UNCACHED 0x2u
#define GPU_CACHED 0x20u
#define CALLBACK_FN (MOD_MAIN_BASE + 0x200)
#define CALLBACK_DATA 0xDA7Aull

static Cpu c;
static uint64_t heap, heap_used, loader;

// ---- the guest side: a stand-in, and the game's memory callback
static void fake_fn(Cpu *cpu) { (void)cpu; }
static uint64_t g_pool, g_pool_mem;                 // a CPU-visible memory pool for command memory
static uint64_t g_cmd_off;                          // command memory handed out so far
static int cb_adds = 1;                             // the callback adds what it is asked for
static uint64_t galloc(uint64_t size, uint64_t align);
static uint64_t nvn_vcall(uint64_t fn, const uint64_t *a, size_t n);
static uint64_t proc(const char *name);
#define NVN(name, ...)                                                                                               \
    nvn_vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t))

static void memory_callback(Cpu *cpu) {
    uint64_t cmd = cpu->x[0], event = cpu->x[1], size = cpu->x[2], data = cpu->x[3];
    printf("cb %llu %#llx\n", (ull)event, (ull)size);
    CHECK(data == CALLBACK_DATA);
    if (!cb_adds) return;
    if (event == 1) {
        NVN(nvnCommandBufferAddControlMemory, cmd, galloc(256, 16), 256);
    } else {
        NVN(nvnCommandBufferAddCommandMemory, cmd, g_pool, g_cmd_off, 4096);
        g_cmd_off += 4096;
    }
}

static void setup(void) {
    RtMemConfig cfg = {.pool_size = 256 * MiB, .heap_region_size = 128 * MiB, .alias_region_size = 1024 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    c.x[0] = 0, c.x[1] = HEAP;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    static const uint64_t addrs[] = {MOD_MAIN_BASE + 0x100, CALLBACK_FN};
    static const GuestFn fns[] = {fake_fn, memory_callback};
    static const RtModule mods[] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, addrs, fns, 2, TEST_END}};
    rt_dispatch_init(mods, 1);
    c.excl_addr = ~0ull;
}

// Guest memory, zeroed, as the game's allocator would give it.
static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP);
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

static uint64_t nvn_vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}

static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[200];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) {
        c.x[0] = guest_str("nvnDeviceGetProcAddress"), c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
    }
    CHECK(gpa != 0 && n < 200);
    c.x[0] = 0, c.x[1] = guest_str(name);
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0);
    cache[n].name = name, cache[n].addr = c.x[0];
    return cache[n++].addr;
}

// A call the driver checks the record of: its name is printed first.
#define CALLP(name, ...) (printf("call %s\n", #name), NVN(name, __VA_ARGS__))
static void fv(int i, float f) {   // a float argument in v<i>
    uint32_t b;
    memcpy(&b, &f, 4);
    c.v[i] = b;
}

static uint64_t device(void) {
    uint64_t b = galloc(0x40, 16), d = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, b);
    CHECK((NVN(nvnDeviceInitialize, d, b) & 0xff) == 1);
    return d;
}

static uint64_t memory_pool(uint64_t size) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16), storage = galloc(size, 0x10000);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetStorage, b, storage, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, CPU_UNCACHED | GPU_CACHED);
    CHECK((NVN(nvnMemoryPoolInitialize, o, b) & 0xff) == 1);
    g_pool_mem = storage;
    return o;
}

// ---- command buffers, as the game sets them up
static uint64_t g_dev;
static uint64_t command_buffer(uint64_t ctl_bytes) {
    uint64_t o = galloc(0xC0, 16);
    CHECK((CALLP(nvnCommandBufferInitialize, o, g_dev) & 0xff) == 1);
    CALLP(nvnCommandBufferSetMemoryCallback, o, CALLBACK_FN);
    CALLP(nvnCommandBufferSetMemoryCallbackData, o, CALLBACK_DATA);
    CALLP(nvnCommandBufferAddCommandMemory, o, g_pool, g_cmd_off, 65536);
    g_cmd_off += 65536;
    if (ctl_bytes) CALLP(nvnCommandBufferAddControlMemory, o, galloc(ctl_bytes, 16), ctl_bytes);
    return o;
}

// A handle array of one, in guest memory.
static uint64_t one_handle(uint64_t h) {
    uint64_t a = galloc(8, 8);
    memcpy((void *)(uintptr_t)a, &h, 8);
    return a;
}

// The game's BeginRecording, inlined at twelve sites of main, played from the console driver's offsets: with 16 bytes
// of control memory left it writes the recording's start, its flags and a 16-byte header itself; otherwise it calls
// nvnCommandBufferBeginRecording. Returns 1 when it took the inline path.
static uint64_t rd64(uint64_t o, int off) { uint64_t v; memcpy(&v, (const void *)(uintptr_t)(o + off), 8); return v; }
static void wr64(uint64_t o, int off, uint64_t v) { memcpy((void *)(uintptr_t)(o + off), &v, 8); }
static int inline_begin(uint64_t o) {
    uint64_t ctl = rd64(o, 0x28), end = rd64(o, 0x30);
    if (ctl + 0x10 > end) {
        NVN(nvnCommandBufferBeginRecording, o);
        return 0;
    }
    wr64(o, 0x58, 0);
    wr64(o, 0x48, rd64(o, 0x00));
    wr64(o, 0x50, ctl);
    uint16_t on = 1;
    memcpy((void *)(uintptr_t)(o + 0x11), &on, 2);
    wr64(ctl, 0, (uint64_t)-0xff);
    wr64(ctl, 8, 0);
    wr64(o, 0x28, ctl + 0x10);
    return 1;
}

// ---- the op log
static int g_print_ops = 1;
static void (*g_check)(const op_hdr *r, int index);
static void op_log(uint64_t fence, int index, const op_hdr *r, void *ctx) {
    (void)ctx;
    if (g_print_ops) printf("rec %llu %d %s\n", (ull)fence, index, nvn_op_name(r->op));
    if (g_check) g_check(r, index);
}
static uint64_t submit(const uint64_t *handles, int n) { return nvn_submit_recordings(&c, 1, handles, n); }

// ---- objects in the front end's layouts
static uint64_t texture(uint32_t width, uint64_t serial) {
    uint64_t t = galloc(0x80, 16);
    tex_t *x = (tex_t *)(uintptr_t)t;
    x->magic = NVN_TEX_MAGIC, x->serial = serial, x->d.width = width, x->d.height = 16, x->d.depth = 1;
    return t;
}
static nvn_stage g_stage[4];
static uint64_t program(void) {
    uint64_t p = galloc(0xC0, 16);
    prog_t *x = (prog_t *)(uintptr_t)p;
    x->magic = NVN_PROG_MAGIC, x->nstage = 2, x->serial = 77;
    x->st[0] = (prog_stage_t){0x1000, &g_stage[0]};
    x->st[1] = (prog_stage_t){0x2000, &g_stage[1]};
    return p;
}

// ---- case ops: one recording with every command the game names, one called from it and one copied
static struct {
    uint64_t rt0, rt1, depth, prog, vattr, vstream, viewports, scissors, ranges, push, push_big, ub, deferred, event,
        word, rates, dpool, results;   // results: pool memory the reports, clears and uniform update write at submit
} O;
static int g_checked;
static int seen[OP_COUNT_];
#define CHECKED(cond)                                                                                                \
    do {                                                                                                             \
        CHECK(cond);                                                                                                 \
        g_checked++;                                                                                                 \
    } while (0)
#define SAW(r, T) ((const T *)(const void *)((r) + 1))
static void check_ops(const op_hdr *r, int index) {
    if (index != 0) return;
    int k = seen[r->op]++;
    switch (r->op) {
    case OP_SET_RT: {
        const p_set_rt *p = SAW(r, p_set_rt);
        CHECK(p->n == 2 && p->color[0].tex.d.width == 640 && p->color[1].tex.d.width == 320);
        CHECK(p->color[0].present && p->color[0].has_view && !p->color[1].has_view);
        CHECK(p->depth.present && p->depth.tex.serial == 3 && r->size == 8 + offsetof(p_set_rt, color) + 2 * sizeof(nvn_tex_rec));
        g_checked++;
        break;
    }
    case OP_BIND_PROGRAM: {
        const p_bind_program *p = SAW(r, p_bind_program);
        CHECK(p->serial == 77 && p->nstage == 2 && p->stages == 0x1F);
        CHECK(p->id[0] == &g_stage[0] && p->id[1] == &g_stage[1] && p->code[0] == 0x1000 && p->code[1] == 0x2000);
        g_checked++;
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *p = SAW(r, p_bind_vattr);
        CHECK(p->n == 32 && p->st[0] == 1 && p->st[31] == 94);
        g_checked++;
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *p = SAW(r, p_bind_vstream);
        CHECK(p->n == 32 && p->st[31].stride == 35);
        g_checked++;
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *p = SAW(r, p_bind_vbuf);
        CHECK(p->first == 31 && p->n == 1 && p->r[0].address == 0xABC000 && p->r[0].size == 0x100);
        g_checked++;
        break;
    }
    case OP_VIEWPORTS: {
        const p_viewports *p = SAW(r, p_viewports);
        if (k == 0) CHECK(p->first == 0 && p->n == 1 && p->v[0][2] == 1280.0f);
        else CHECK(p->first == 2 && p->n == 3 && p->v[2][3] == 23.0f);
        g_checked++;
        break;
    }
    case OP_SCISSOR: {
        const p_scissor *p = SAW(r, p_scissor);
        if (k == 1) CHECKED(p->first == 0 && p->n == 2 && p->r[1][2] == 300);
        break;
    }
    case OP_DEPTH_RANGE: {
        const p_depth_range *p = SAW(r, p_depth_range);
        if (k == 0) CHECK(p->n == 1 && p->r[0][0] == 0.25f && p->r[0][1] == 0.75f);
        else CHECK(p->first == 1 && p->n == 2 && p->r[1][1] == 0.5f);
        g_checked++;
        break;
    }
    case OP_PUSH_CONST: {
        const p_push_const *p = SAW(r, p_push_const);
        if (k == 0) CHECK(!p->compute && p->table == 1 && p->offset == 8 && p->bytes == 12 && p->w[2] == 3);
        else CHECK(p->compute && p->bytes == 0x100 && p->w[63] == 63);
        g_checked++;
        break;
    }
    case OP_UPDATE_UB: {
        const p_update_ub *p = SAW(r, p_update_ub);
        CHECK(p->dst == O.results + 0x10 && p->size == 32 && p->data[31] == 31);
        g_checked++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *p = SAW(r, p_bind_handles);
        if (k == 0) CHECK(nvn_handle_texture_id(p->h[0]) == 5 && nvn_handle_sampler_id(p->h[0]) == 7 && !p->src);
        if (k == 2) {   // deferred: what the array holds at submit
            CHECK(p->src == O.deferred && p->n == 2);
            CHECK(nvn_handle_texture_id(p->h[1]) == 12 && nvn_handle_sampler_id(p->h[1]) == 13);
        }
        if (k == 3) CHECK(nvn_handle_texture_id(p->h[0]) == 9);   // the called recording's bind
        g_checked++;
        break;
    }
    case OP_SIGNAL_EVENT: {
        const p_signal_event *p = SAW(r, p_signal_event);
        CHECK((uintptr_t)p->val == O.word && p->mode == 1 && p->value == 42 && p->location == 2 && p->flags == 3);
        g_checked++;
        break;
    }
    case OP_SAMPLE_MASK: CHECKED(SAW(r, p_flag)->value == 0x2345); break;
    case OP_CONSERVATIVE:
        if (k == 1) CHECKED(SAW(r, p_conservative)->value == 2 && SAW(r, p_conservative)->extra == 7);
        break;
    case OP_SHADING_RATE_PALETTE: {
        const p_shading_rate_palette *p = SAW(r, p_shading_rate_palette);
        CHECK(p->index == 1 && p->n == 3 && p->has_rates && p->rates[2] == 3);
        g_checked++;
        break;
    }
    case OP_SHADING_RATE_IMAGE: CHECKED(SAW(r, p_shading_rate_image)->tex.tex.d.width == 640); break;
    case OP_SET_TEX_POOL: CHECKED(SAW(r, p_set_pool)->pool.addr == g_pool_mem && SAW(r, p_set_pool)->pool.count == 16); break;
    case OP_SCRATCH: CHECKED(SAW(r, p_scratch)->addr == g_pool_mem + 0x100 && SAW(r, p_scratch)->size == 0x2000); break;
    case OP_COPY_T2T: CHECKED(SAW(r, p_copy_t2t)->src.tex.d.width == 320 && SAW(r, p_copy_t2t)->dr.width == 8); break;
    case OP_BIND_STATE:
        if (k == 6) {   // SetPointSize
            float s;
            memcpy(&s, SAW(r, p_bind_state)->bytes, 4);
            CHECKED(SAW(r, p_bind_state)->kind == 6 && s == 2.5f);
        }
        break;
    default: break;
    }
}

static void case_ops(void) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    gfx_null_log_ops(op_log, NULL);
    g_check = check_ops;
    uint64_t a = command_buffer(4096), b = command_buffer(4096), cc = command_buffer(4096);

    // A, begun through nvnCommandBufferBeginRecording: one bind and one draw
    printf("recording A\n");
    CALLP(nvnCommandBufferSetCommandMemoryCallbackEnabled, a, 1);
    CALLP(nvnCommandBufferSetRecordingMode, a, 0);
    CALLP(nvnCommandBufferBeginRecording, a);
    CALLP(nvnCommandBufferBindTexture, a, 4, 0, nvn_handle_pack(9, 1));
    CALLP(nvnCommandBufferDrawArrays, a, 4, 0, 3);
    uint64_t ha = CALLP(nvnCommandBufferEndRecording, a);
    printf("control used A %#llx\n", (ull)NVN(nvnCommandBufferGetControlMemoryUsed, a));
    CHECK(CALLP(nvnCommandBufferIsRecording, a) == 0);
    CHECK(CALLP(nvnCommandBufferGetMemoryCallback, a) == CALLBACK_FN);
    CHECK(CALLP(nvnCommandBufferGetMemoryCallbackData, a) == CALLBACK_DATA);
    CALLP(nvnCommandBufferSetControlMemoryCallbackEnabled, a, 1);
    CALLP(nvnCommandBufferSetDebugLabel, a, guest_str("A"));
    uint64_t used = CALLP(nvnCommandBufferGetCommandMemoryUsed, a);
    CHECK(used > 0 && CALLP(nvnCommandBufferGetCommandMemorySize, a) == 65536);
    CHECK(CALLP(nvnCommandBufferGetCommandMemoryFree, a) == 65536 - used);
    CHECK(CALLP(nvnCommandBufferGetControlMemorySize, a) == 4096);
    CHECK(CALLP(nvnCommandBufferGetControlMemoryFree, a) == 4096 - 0x30);

    O.rt0 = texture(640, 1), O.rt1 = texture(320, 2), O.depth = texture(640, 3), O.prog = program();
    uint64_t rts = galloc(16, 8), views = galloc(16, 8), view = galloc(16, 16), region = galloc(sizeof(NVNcopyRegion), 8);
    wr64(rts, 0, O.rt0), wr64(rts, 8, O.rt1), wr64(views, 0, view), wr64(views, 8, 0);
    ((NVNcopyRegion *)(uintptr_t)region)->width = 8;
    O.vattr = galloc(4 * 32, 8), O.vstream = galloc(8 * 32, 8);
    for (int i = 0; i < 32; i++) {
        ((uint32_t *)(uintptr_t)O.vattr)[i] = 3u * (uint32_t)i + 1;
        ((vs_st *)(uintptr_t)O.vstream)[i] = (vs_st){(uint32_t)i + 4, 0};
    }
    O.viewports = galloc(16 * 4 * 3, 8), O.scissors = galloc(16 * 2, 8), O.ranges = galloc(8 * 2, 8);
    for (int i = 0; i < 12; i++) ((float *)(uintptr_t)O.viewports)[i] = 12.0f + (float)i;
    int32_t sc[8] = {0, 0, 100, 100, 10, 10, 300, 200};
    memcpy((void *)(uintptr_t)O.scissors, sc, sizeof sc);
    float rg[4] = {0.0f, 1.0f, 0.25f, 0.5f};
    memcpy((void *)(uintptr_t)O.ranges, rg, sizeof rg);
    O.push = galloc(16, 8), O.push_big = galloc(0x200, 8), O.ub = galloc(32, 8);
    for (uint32_t i = 0; i < 3; i++) ((uint32_t *)(uintptr_t)O.push)[i] = i + 1;
    for (uint32_t i = 0; i < 128; i++) ((uint32_t *)(uintptr_t)O.push_big)[i] = i;
    for (int i = 0; i < 32; i++) ((uint8_t *)(uintptr_t)O.ub)[i] = (uint8_t)i;
    O.deferred = galloc(16, 8);
    wr64(O.deferred, 0, nvn_handle_pack(1, 2)), wr64(O.deferred, 8, nvn_handle_pack(3, 4));
    O.word = galloc(8, 8), O.event = galloc(0x40, 16);
    wr64(O.event, 8, O.word);   // ev_t: {magic, val}
    O.rates = galloc(16, 8);
    for (uint32_t i = 0; i < 3; i++) ((uint32_t *)(uintptr_t)O.rates)[i] = i + 1;
    O.dpool = galloc(0x20, 16);
    wr64(O.dpool, 0, g_pool), wr64(O.dpool, 8, 0), wr64(O.dpool, 16, 16);   // dpool_t: {pool, offset, n}
    O.results = g_pool_mem + 0x300000;   // past the command memory the case hands out
    uint64_t handles2 = galloc(16, 8), ranges2 = galloc(32, 8), rgba = galloc(16, 8), counters = O.results;
    wr64(handles2, 0, nvn_handle_pack(20, 0)), wr64(handles2, 8, nvn_handle_pack(21, 0));
    uint64_t state = galloc(0x40, 16);   // packed state bytes as a state object holds them

    // B, begun inline: every command the game names, once (the driver pins the set it expects)
    printf("recording B\n");
    CHECK(inline_begin(b) == 1);
    CHECK(NVN(nvnCommandBufferIsRecording, b) == 1);
    CALLP(nvnCommandBufferSetRenderTargets, b, 2, rts, views, O.depth, 0);
    CALLP(nvnCommandBufferClearColor, b, 0, rgba, 0xF);
    fv(0, 1.0f), CALLP(nvnCommandBufferClearDepthStencil, b, 1, 0, 0xFF);
    CALLP(nvnCommandBufferClearTexture, b, O.rt1, 0, region, rgba, 0xF);
    CALLP(nvnCommandBufferSetViewport, b, 0, 0, 1280, 720);
    CALLP(nvnCommandBufferSetViewports, b, 2, 3, O.viewports);
    CALLP(nvnCommandBufferSetScissor, b, 0, 0, 1280, 720);
    CALLP(nvnCommandBufferSetScissors, b, 0, 2, O.scissors);
    fv(0, 0.25f), fv(1, 0.75f), CALLP(nvnCommandBufferSetDepthRange, b);
    CALLP(nvnCommandBufferSetDepthRanges, b, 1, 2, O.ranges);
    fv(0, 0.0f), fv(1, 1.0f), CALLP(nvnCommandBufferSetDepthBounds, b, 1);
    CALLP(nvnCommandBufferSetDepthClamp, b, 1);
    CALLP(nvnCommandBufferSetRenderEnable, b, 1);
    CALLP(nvnCommandBufferSetRasterizerDiscard, b, 0);
    CALLP(nvnCommandBufferSetGeometryProcessingMode, b, 1);
    CALLP(nvnCommandBufferSetSampleMask, b, 0x12345);
    CALLP(nvnCommandBufferSetPrimitiveRestart, b, 1, 0xFFFF);
    CALLP(nvnCommandBufferSetRenderEnableConditional, b, 1, counters);
    CALLP(nvnCommandBufferSetConservativeRasterEnable, b, 1);
    CALLP(nvnCommandBufferSetConservativeRasterMode, b, 2, 0x37);
    CALLP(nvnCommandBufferBindProgram, b, O.prog, 0x1F);
    CALLP(nvnCommandBufferBindBlendState, b, state);
    CALLP(nvnCommandBufferBindColorState, b, state);
    CALLP(nvnCommandBufferBindChannelMaskState, b, state);
    CALLP(nvnCommandBufferBindMultisampleState, b, state);
    CALLP(nvnCommandBufferBindPolygonState, b, state);
    CALLP(nvnCommandBufferBindDepthStencilState, b, state);
    fv(0, 2.5f), CALLP(nvnCommandBufferSetPointSize, b);
    fv(0, 1.0f), CALLP(nvnCommandBufferSetLineWidth, b);
    CALLP(nvnCommandBufferBindVertexAttribState, b, 32, O.vattr);
    CALLP(nvnCommandBufferBindVertexStreamState, b, 32, O.vstream);
    CALLP(nvnCommandBufferBindVertexBuffer, b, 31, 0xABC000, 0x100);
    CALLP(nvnCommandBufferBindUniformBuffer, b, 1, 0, 0x5000, 0x100);
    CALLP(nvnCommandBufferBindStorageBuffer, b, 1, 0, 0x6000, 0x100);
    CALLP(nvnCommandBufferBindStorageBuffers, b, 1, 2, 2, ranges2);
    CALLP(nvnCommandBufferBindTexture, b, 1, 0, nvn_handle_pack(5, 7));
    CALLP(nvnCommandBufferBindTextures, b, 1, 1, 2, handles2);
    CALLP(nvnCommandBufferBindTexturesDeferred, b, 1, 3, 2, O.deferred);
    CALLP(nvnCommandBufferBindImage, b, 4, 0, 6);
    CALLP(nvnCommandBufferBindImages, b, 4, 1, 2, handles2);
    CALLP(nvnCommandBufferBindAccelerationStructuresDeferred, b, 4, 0, 1, handles2);
    CALLP(nvnCommandBufferSetTexturePool, b, O.dpool);
    CALLP(nvnCommandBufferSetSamplerPool, b, O.dpool);
    CALLP(nvnCommandBufferSetShaderScratchMemory, b, g_pool, 0x100, 0x2000);
    CALLP(nvnCommandBufferPushGraphicsConstants, b, 9, 8, 12, O.push);
    CALLP(nvnCommandBufferPushComputeConstants, b, 0, 0, 0x200, O.push_big);
    CALLP(nvnCommandBufferSetStencilValueMask, b, 0, 0xFF);
    CALLP(nvnCommandBufferSetStencilMask, b, 0, 0x0F);
    CALLP(nvnCommandBufferSetStencilRef, b, 0, 1);
    CALLP(nvnCommandBufferSetStencilCullCriteria, b, 7, 1, 0xFF);
    fv(0, 1.0f), fv(1, 2.0f), fv(2, 0.0f), CALLP(nvnCommandBufferSetPolygonOffsetClamp, b);
    CALLP(nvnCommandBufferSetBlendColor, b, rgba);
    CALLP(nvnCommandBufferDrawArrays, b, 4, 0, 3);
    CALLP(nvnCommandBufferDrawArraysInstanced, b, 4, 0, 3, 0, 0);
    CALLP(nvnCommandBufferDrawElements, b, 4, 2, 6, 0x8000);
    CALLP(nvnCommandBufferDrawElementsBaseVertex, b, 4, 2, 6, 0x8000, 10);
    CALLP(nvnCommandBufferDrawElementsInstanced, b, 4, 2, 6, 0x8000, 10, 1, 4);
    CALLP(nvnCommandBufferDrawElementsIndirect, b, 4, 2, 0x8000, 0x9000);
    CALLP(nvnCommandBufferDrawMeshTasks, b, 0, 32);
    CALLP(nvnCommandBufferDrawMeshTasksIndirect, b, 0x9100);
    CALLP(nvnCommandBufferDispatchCompute, b, 8, 4, 1);
    CALLP(nvnCommandBufferDispatchComputeIndirect, b, 0x9200);
    CALLP(nvnCommandBufferBarrier, b, 0x10);
    CALLP(nvnCommandBufferCallCommands, b, 1, one_handle(ha));
    CALLP(nvnCommandBufferSignalEvent, b, O.event, 1, 2, 3, 42);
    CALLP(nvnCommandBufferWaitEvent, b, O.event, 0, 42);
    CALLP(nvnCommandBufferReportCounter, b, 2, O.results + 0x300);
    CALLP(nvnCommandBufferResetCounter, b, 2);
    CALLP(nvnCommandBufferReportValue, b, 5, O.results + 0x400);
    CALLP(nvnCommandBufferClearBuffer, b, O.results + 0x500, 0x40, 0);
    CALLP(nvnCommandBufferUpdateUniformBuffer, b, O.results, 0x100, 0x10, 32, O.ub);
    CALLP(nvnCommandBufferCopyBufferToTexture, b, 0xA000, O.rt0, 0, region, 0);
    CALLP(nvnCommandBufferCopyTextureToBuffer, b, O.rt0, 0, region, 0xB000, 0);
    CALLP(nvnCommandBufferCopyTextureToTexture, b, O.rt1, 0, region, O.rt0, 0, region, 0);
    CALLP(nvnCommandBufferSetCopyRowStride, b, 256);
    CALLP(nvnCommandBufferSetCopyImageStride, b, 4096);
    CALLP(nvnCommandBufferBindShadingRateImage, b, O.rt0, 0);
    CALLP(nvnCommandBufferShadingRateImagePalette, b, 1, 3, O.rates);
    CALLP(nvnCommandBufferSetShadingRatePaletteMode, b, 1);
    CALLP(nvnCommandBufferSetComputeOverlapMode, b, 1);
    CALLP(nvnCommandBufferSetComputeWorkloadPriority, b, 2);
    CALLP(nvnCommandBufferSetColorReductionEnable, b, 1);
    CALLP(nvnCommandBufferSetColorReductionThresholds, b, 3, 10, 20);
    CALLP(nvnCommandBufferSetTiledCacheAction, b, 1);
    CALLP(nvnCommandBufferSetTiledCacheTileSize, b, 16, 16);
    printf("command used B %#llx\n", (ull)CALLP(nvnCommandBufferGetCommandMemoryUsed, b));
    uint64_t hb = CALLP(nvnCommandBufferEndRecording, b);
    CHECK(NVN(nvnCommandBufferIsRecording, b) == 0);
    printf("control used B %#llx\n", (ull)CALLP(nvnCommandBufferGetControlMemoryUsed, b));

    // C, begun inline: A copied in
    printf("recording C\n");
    CHECK(inline_begin(cc) == 1);
    CALLP(nvnCommandBufferCopyCommands, cc, 1, one_handle(ha));
    uint64_t hc = CALLP(nvnCommandBufferEndRecording, cc);
    printf("control used C %#llx\n", (ull)NVN(nvnCommandBufferGetControlMemoryUsed, cc));

    // What the game changes once the calls returned: textures and the program initialized again, arrays rewritten.
    // The records keep what the calls gave, but the deferred bind, which reads its array at submit.
    ((tex_t *)(uintptr_t)O.rt0)->d.width = 1, ((tex_t *)(uintptr_t)O.rt0)->serial = 900;
    ((tex_t *)(uintptr_t)O.rt1)->d.width = 1, ((tex_t *)(uintptr_t)O.rt1)->serial = 901;
    prog_t *pg = (prog_t *)(uintptr_t)O.prog;
    pg->serial = 78, pg->st[0].id = &g_stage[2], pg->st[1].id = &g_stage[3];
    memset((void *)(uintptr_t)O.vattr, 0xFF, 4 * 32);
    memset((void *)(uintptr_t)O.vstream, 0xFF, 8 * 32);
    memset((void *)(uintptr_t)O.viewports, 0, 16 * 4 * 3);
    memset((void *)(uintptr_t)O.scissors, 0, 16 * 2);
    memset((void *)(uintptr_t)O.ranges, 0, 8 * 2);
    memset((void *)(uintptr_t)O.push, 0, 16);
    memset((void *)(uintptr_t)O.push_big, 0, 0x200);
    memset((void *)(uintptr_t)O.ub, 0, 32);
    memset((void *)(uintptr_t)O.rates, 0, 16);
    wr64(O.deferred, 0, nvn_handle_pack(10, 11)), wr64(O.deferred, 8, nvn_handle_pack(12, 13));

    printf("recorded\n");
    CALLP(nvnCommandBufferFinalize, a);
    CALLP(nvnCommandBufferFinalize, b);
    CALLP(nvnCommandBufferFinalize, cc);
    CALLP(nvnDeviceFinalizeCommandHandle, g_dev, ha);
    uint64_t hs[2] = {hb, hc};
    CHECK(submit(hs, 2) == 1);
    // The null back end wrote the results: the event, the uniform data the call gave, the 4-byte report value.
    CHECK(*(const uint32_t *)(uintptr_t)O.word == 42 && ((const uint8_t *)(uintptr_t)O.results)[0x10 + 31] == 31);
    CHECK(*(const uint32_t *)(uintptr_t)(O.results + 0x400) == 5);
    printf("checked %d\n", g_checked);
    printf("ok ops\n");
}

// ---- case begin: the two ways a recording starts, and what the front end leaves in control memory
static void case_begin(void) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    gfx_null_log_ops(op_log, NULL);
    uint64_t o = command_buffer(0), ctl = galloc(0x40, 16);
    NVN(nvnCommandBufferAddControlMemory, o, ctl, 0x40);   // the driver keeps 16 bytes back: room for 0x30
    for (int k = 0; k < 2; k++) {
        int fast = inline_begin(o);
        printf("begin %s\n", fast ? "inline" : "call");
        CHECK(NVN(nvnCommandBufferIsRecording, o) == 1);
        NVN(nvnCommandBufferDrawArrays, o, 4, k, 3);
        uint64_t h = NVN(nvnCommandBufferEndRecording, o);
        printf("control used %#llx\n", (ull)NVN(nvnCommandBufferGetControlMemoryUsed, o));
        CHECK(submit(&h, 1) == (uint64_t)k + 1);
    }
    // The front end wrote nothing into control memory: only the header the inline path wrote is there.
    for (int i = 0x10; i < 0x40; i++) CHECK(((const uint8_t *)(uintptr_t)ctl)[i] == 0);
    CHECK(rd64(ctl, 0) == (uint64_t)-0xff);
    printf("control memory holds the game's header only\n");
    // A recording begun again before it ended keeps what came after the second start, as on the console.
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 100, 3);
    inline_begin(o);
    NVN(nvnCommandBufferDispatchCompute, o, 1, 1, 1);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    CHECK(submit(&h, 1) == 3);
    CHECK(NVN(nvnCommandBufferEndRecording, o) == 0);   // no recording open: the driver answers 0
    NvnCmdCounts k;
    nvn_cmd_counts(&k);
    printf("counts inline %llu slow %llu handle %llu empty %llu left_out %llu\n", (ull)k.begin_inline, (ull)k.begin_slow,
           (ull)k.end_handle, (ull)k.end_empty, (ull)k.left_out);
    printf("ok begin\n");
}

// The front end's counters as census rows (nvn_census.c), printed for the driver: "stat| key count why".
static void print_stat(void *ctx, const char *key, uint64_t count, const char *why) {
    (void)ctx;
    printf("stat| %s %llu %s\n", key, (ull)count, why);
}

// ---- case callbacks: where the memory callback is called, and with what
static void case_callbacks(void) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    gfx_null_log_ops(op_log, NULL);
    g_print_ops = 0;

    printf("step no control memory\n");   // the inline path finds no room and calls BeginRecording
    uint64_t o = galloc(0xC0, 16);
    NVN(nvnCommandBufferInitialize, o, g_dev);
    NVN(nvnCommandBufferSetMemoryCallback, o, CALLBACK_FN);
    NVN(nvnCommandBufferSetMemoryCallbackData, o, CALLBACK_DATA);
    CHECK(inline_begin(o) == 0);
    printf("step no command memory\n");   // the first command, and only it
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    printf("command size %#llx\n", (ull)NVN(nvnCommandBufferGetCommandMemorySize, o));
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    CHECK(submit(&h, 1) == 1);

    printf("step end short of control memory\n");   // the segment entry needs 0x20 bytes, 0x10 are left
    uint64_t p = command_buffer(0x30);
    CHECK(inline_begin(p) == 1);
    NVN(nvnCommandBufferDrawArrays, p, 4, 0, 3);
    uint64_t hp = NVN(nvnCommandBufferEndRecording, p);
    printf("control used %#llx\n", (ull)NVN(nvnCommandBufferGetControlMemoryUsed, p));
    CHECK(submit(&hp, 1) == 2);

    printf("step call short of control memory\n");   // the called recording's segment entry does not fit
    uint64_t q = command_buffer(0x30);
    CHECK(inline_begin(q) == 1);
    NVN(nvnCommandBufferCallCommands, q, 1, one_handle(hp));
    uint64_t hq = NVN(nvnCommandBufferEndRecording, q);
    CHECK(submit(&hq, 1) == 3);

    printf("step control memory exactly full\n");   // header, segment and end take every byte: nothing is called
    uint64_t full = command_buffer(0x40);
    CHECK(inline_begin(full) == 1);
    NVN(nvnCommandBufferDrawArrays, full, 4, 0, 3);
    uint64_t hfull = NVN(nvnCommandBufferEndRecording, full);
    printf("control used %#llx\n", (ull)NVN(nvnCommandBufferGetControlMemoryUsed, full));
    CHECK(submit(&hfull, 1) == 4);

    printf("step callbacks off\n");   // no command memory and the callback off: nothing is called
    uint64_t r = galloc(0xC0, 16);
    NVN(nvnCommandBufferInitialize, r, g_dev);
    NVN(nvnCommandBufferSetMemoryCallback, r, CALLBACK_FN);
    NVN(nvnCommandBufferSetMemoryCallbackData, r, CALLBACK_DATA);
    NVN(nvnCommandBufferSetCommandMemoryCallbackEnabled, r, 0);
    NVN(nvnCommandBufferAddControlMemory, r, galloc(256, 16), 256);
    CHECK(inline_begin(r) == 1);
    NVN(nvnCommandBufferDrawArrays, r, 4, 0, 3);
    uint64_t hr = NVN(nvnCommandBufferEndRecording, r);
    CHECK(submit(&hr, 1) == 5);

    printf("step ample memory\n");   // a recording with room for everything calls nothing
    uint64_t s = command_buffer(4096);
    CHECK(inline_begin(s) == 1);
    for (int i = 0; i < 200; i++) NVN(nvnCommandBufferDrawElements, s, 4, 2, 6, 0x8000);
    NVN(nvnCommandBufferAddCommandMemory, s, g_pool, g_cmd_off, 65536);
    for (int i = 0; i < 200; i++) NVN(nvnCommandBufferBindUniformBuffer, s, 1, 0, 0x5000, 0x100);
    uint64_t hs2 = NVN(nvnCommandBufferEndRecording, s);
    printf("control used %#llx\n", (ull)NVN(nvnCommandBufferGetControlMemoryUsed, s));
    CHECK(submit(&hs2, 1) == 6);
    printf("step end\n");
    nvn_census_stats(print_stat, NULL);   // the callbacks row: the counts of the calls above
    printf("ok callbacks\n");
}

// ---- case loop: 131 command buffers created, recorded, submitted and finalized a frame, as the Forest does
static void case_loop(int frames) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    g_print_ops = 0;
    gfx_null_log_ops(op_log, NULL);
    enum { N = 131 };
    uint64_t objs[N], ctls[N], hs[N];
    for (int k = 0; k < N; k++) objs[k] = galloc(0xC0, 16), ctls[k] = galloc(512, 16);
    PlatProcMem m0 = {0}, m1 = {0};
    for (int f = 0; f < frames; f++) {
        for (int k = 0; k < N; k++) {
            uint64_t o = objs[k];
            NVN(nvnCommandBufferInitialize, o, g_dev);
            NVN(nvnCommandBufferSetMemoryCallback, o, CALLBACK_FN);
            NVN(nvnCommandBufferSetMemoryCallbackData, o, CALLBACK_DATA);
            NVN(nvnCommandBufferAddCommandMemory, o, g_pool, 4096 * (uint64_t)k, 4096);
            NVN(nvnCommandBufferAddControlMemory, o, ctls[k], 512);
            inline_begin(o);
            for (int i = 0; i < 1 + k % 9; i++) {
                NVN(nvnCommandBufferBindUniformBuffer, o, 1, i, 0x5000, 0x100);
                NVN(nvnCommandBufferBindTexture, o, 1, i, nvn_handle_pack((uint32_t)i, 1));
                NVN(nvnCommandBufferDrawElementsInstanced, o, 4, 2, 6, 0x8000, 0, 0, 1);
            }
            NVN(nvnCommandBufferGetCommandMemoryUsed, o);
            NVN(nvnCommandBufferGetControlMemoryUsed, o);
            hs[k] = NVN(nvnCommandBufferEndRecording, o);
            NVN(nvnCommandBufferFinalize, o);
        }
        submit(hs, N);
        if (f == 20) plat_process_memory(&m0);
        if (f == frames - 1) plat_process_memory(&m1);
        if (f % 50 == 0 || f == frames - 1)
            printf("frame %d host %llu copies %llu\n", f, (ull)nvn_cmd_host_bytes(), (ull)nvn_cmd_copies());
    }
    printf("private %llu %llu\n", (ull)m0.private_bytes, (ull)m1.private_bytes);
    printf("records %llu\n", (ull)gfx_null_records());
    printf("ok loop\n");
}

// ---- the D11 probe (census mode with census_probe). The driver sets the environment. Each case records through
// the entry points only (the front end writes nothing into command memory) and the harness plays the game's stores: a word
// written into command memory, or through a handle, after the range was added.
#define PROBE_RANGE 0x10000ull
static uint64_t probe_range_at(uint64_t pool_mem, uint64_t off) { return pool_mem + off; }
static uint64_t probe_cb(uint64_t pool, uint64_t off) {   // a command buffer that adds [pool, off, PROBE_RANGE)
    uint64_t o = galloc(0xC0, 16);
    CHECK((NVN(nvnCommandBufferInitialize, o, g_dev) & 0xff) == 1);
    NVN(nvnCommandBufferSetMemoryCallback, o, CALLBACK_FN);
    NVN(nvnCommandBufferSetMemoryCallbackData, o, CALLBACK_DATA);
    NVN(nvnCommandBufferAddCommandMemory, o, pool, off, PROBE_RANGE);
    NVN(nvnCommandBufferAddControlMemory, o, galloc(4096, 16), 4096);
    return o;
}
static void store32(uint64_t addr, uint32_t v) { memcpy((void *)(uintptr_t)addr, &v, 4); }
static void probe_setup(void) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    g_print_ops = 0;
    gfx_null_log_ops(op_log, NULL);
}
static void probe_report(const char *name) {
    nvn_census_stats(print_stat, NULL);
    printf("ok %s\n", name);
}
static void probe_files(const char *dir) {
    if (dir) nvn_census_extra(dir);
}
// A store at base + Used + 0x10 after GetCommandMemoryUsed answered, before EndRecording: the Luigi's Mansion 3 shape.
static void case_probe_used(const char *dir) {
    probe_setup();
    uint64_t base = probe_range_at(g_pool_mem, 0);
    uint64_t o = probe_cb(g_pool, 0);
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    uint64_t used = NVN(nvnCommandBufferGetCommandMemoryUsed, o);
    store32(base + used + 0x10, 0xC0DEC0DE);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    CHECK(submit(&h, 1) == 1);
    probe_files(dir);
    probe_report("probe_used");
}
// A store at the last word added: a probe that stops one word short does not see it.
static void case_probe_last_word(const char *dir) {
    probe_setup();
    uint64_t base = probe_range_at(g_pool_mem, 0);
    uint64_t o = probe_cb(g_pool, 0);
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    store32(base + PROBE_RANGE - 8, 1);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    CHECK(submit(&h, 1) == 1);
    probe_files(dir);
    probe_report("probe_last_word");
}
// A store through a handle's copy after EndRecording, before submit.
static void case_probe_handle(const char *dir) {
    probe_setup();
    uint64_t o = probe_cb(g_pool, 0);
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    store32(h + 0x14, 0x77);   // inside the DRAW record's payload, not a header: the back end still walks it whole
    CHECK(submit(&h, 1) == 1);
    probe_files(dir);
    probe_report("probe_handle");
}
// Two command buffers, two recordings: a store on the first one's range after both ended, found when both handles are
// submitted, is a hit that names recording 1 (the one whose copy holds the range), not 2 (the newest in the process).
static void case_probe_which_recording(const char *dir) {
    probe_setup();
    uint64_t base = probe_range_at(g_pool_mem, 0);
    uint64_t o1 = probe_cb(g_pool, 0), o2 = probe_cb(g_pool, PROBE_RANGE);
    CHECK(inline_begin(o1) == 1);
    NVN(nvnCommandBufferDrawArrays, o1, 4, 0, 3);
    uint64_t h[2];
    h[0] = NVN(nvnCommandBufferEndRecording, o1);
    CHECK(inline_begin(o2) == 1);
    NVN(nvnCommandBufferDrawArrays, o2, 4, 1, 3);
    h[1] = NVN(nvnCommandBufferEndRecording, o2);
    store32(base + 0x100, 5);
    CHECK(submit(h, 2) == 1);
    probe_files(dir);
    probe_report("probe_which_recording");
}
// A store on the range after EndRecording (a hit at submit), and another after submit before Finalize (a hit at Finalize).
static void case_probe_after_end(const char *dir) {
    probe_setup();
    uint64_t base = probe_range_at(g_pool_mem, 0);
    uint64_t o = probe_cb(g_pool, 0);
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    store32(base + 0x100, 5);
    CHECK(submit(&h, 1) == 1);
    store32(base + 0x200, 6);
    NVN(nvnCommandBufferFinalize, o);
    probe_files(dir);
    probe_report("probe_after_end");
}
// Recordings the front end wrote alone, over several command buffers and recordings, a range added again, a copy called
// from another recording: no hit, and the report says "not seen over N recordings".
static void case_probe_quiet(const char *dir) {
    probe_setup();
    for (int k = 0; k < 4; k++) {
        uint64_t o = probe_cb(g_pool, (uint64_t)k * PROBE_RANGE);
        uint64_t hs[3];
        for (int r = 0; r < 3; r++) {
            CHECK(inline_begin(o) == 1);
            NVN(nvnCommandBufferBindUniformBuffer, o, 1, 0, 0x5000, 0x100);
            NVN(nvnCommandBufferDrawElementsInstanced, o, 4, 2, 6, 0x8000, 0, 0, 1);
            NVN(nvnCommandBufferGetCommandMemoryUsed, o);
            hs[r] = NVN(nvnCommandBufferEndRecording, o);
            if (r == 1) NVN(nvnCommandBufferAddCommandMemory, o, g_pool, (uint64_t)k * PROBE_RANGE, PROBE_RANGE);   // added again
        }
        submit(hs, 3);
        NVN(nvnCommandBufferFinalize, o);
    }
    probe_files(dir);
    probe_report("probe_quiet");
}
// Command memory that is not committed: the probe never reads it, so the commit counts do not move; a page that holds a
// word by the next check is a hit (the store here commits it).
static void case_probe_uncommitted(const char *dir) {
    probe_setup();
    uint64_t raw = (heap + HEAP - 8 * MiB) & ~0xFFFFull;   // far from everything galloc touched
    uint64_t b = galloc(0x40, 16), pool = galloc(0x100, 16);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetStorage, b, raw, 4 * MiB);
    NVN(nvnMemoryPoolBuilderSetFlags, b, CPU_UNCACHED | GPU_CACHED);
    CHECK((NVN(nvnMemoryPoolInitialize, pool, b) & 0xff) == 1);
    int committed = 1;
    CHECK(rt_mem_commit_run(raw, PROBE_RANGE, &committed) >= PROBE_RANGE && !committed);
    RtMemStats m0, m1;
    rt_mem_measure(&m0);
    uint64_t o = probe_cb(pool, 0);
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    CHECK(submit(&h, 1) == 1);
    NVN(nvnCommandBufferFinalize, o);
    rt_mem_measure(&m1);
    CHECK(rt_mem_commit_run(raw, PROBE_RANGE, &committed) >= PROBE_RANGE && !committed);
    printf("touch_commits %llu %llu committed %llu %llu\n", (ull)m0.r[RT_MEM_HEAP].touch_commits,
           (ull)m1.r[RT_MEM_HEAP].touch_commits, (ull)m0.r[RT_MEM_HEAP].committed, (ull)m1.r[RT_MEM_HEAP].committed);
    CHECK(m0.r[RT_MEM_HEAP].touch_commits == m1.r[RT_MEM_HEAP].touch_commits);
    CHECK(m0.r[RT_MEM_HEAP].committed == m1.r[RT_MEM_HEAP].committed);
    nvn_census_stats(print_stat, NULL);   // no hit so far
    uint64_t p = probe_cb(pool, 0x20000);   // a second range: the game stores into it, which commits the page
    CHECK(inline_begin(p) == 1);
    NVN(nvnCommandBufferDrawArrays, p, 4, 0, 3);
    store32(raw + 0x20000 + 0x40, 9);
    uint64_t hp = NVN(nvnCommandBufferEndRecording, p);
    CHECK(submit(&hp, 1) == 2);
    probe_files(dir);
    probe_report("probe_uncommitted");
}
// The same stores with no probe (a default run, or census mode without the key): no state, no row, no file.
static void case_probe_default(const char *dir) {
    probe_setup();
    CHECK(!nvn_probe_on());
    uint64_t base = probe_range_at(g_pool_mem, 0);
    uint64_t o = probe_cb(g_pool, 0);
    CHECK(inline_begin(o) == 1);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    uint64_t used = NVN(nvnCommandBufferGetCommandMemoryUsed, o);
    store32(base + used + 0x10, 0xC0DEC0DE);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    store32(h + 0x14, 0x77);
    CHECK(submit(&h, 1) == 1);
    NVN(nvnCommandBufferFinalize, o);
    NvnProbeStats s;
    nvn_probe_stats(&s);
    CHECK(!s.buffers && !s.ranges && !s.hits && !s.bytes_hashed);
    probe_files(dir);
    probe_report("probe_default");
}

// ---- the census timer's rows (nvn_census.c, with the timer on): calls of NVN2 entry points are timed on this thread, and the
// present writes one row for the thread. Without the flag nothing is written.
static void case_time_rows(const char *dir) {
    probe_setup();
    uint64_t o = probe_cb(g_pool, 0);
    for (uint64_t seq = 1; seq <= 3; seq++) {
        CHECK(inline_begin(o) == 1);
        for (int i = 0; i < 5; i++) NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
        uint64_t h = NVN(nvnCommandBufferEndRecording, o);
        CHECK(submit(&h, 1) == seq);
        nvn_census_present(seq);
    }
    printf("timer %d\n", rt_timer_on());
    probe_files(dir);
    printf("ok time_rows\n");
}

// ---- failing cases: each ends in one named fatal
static uint64_t begun(void) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    uint64_t o = command_buffer(4096);
    CHECK(inline_begin(o) == 1);
    return o;
}
// Game code that writes GPU methods itself moves the command write pointer.
static void case_inline_command_write(void) {
    uint64_t o = begun();
    wr64(o, 0x00, rd64(o, 0x08) + 12);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
}
// Game code that writes control memory itself moves the control write pointer.
static void case_control_moved(void) {
    uint64_t o = begun();
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    wr64(o, 0x28, rd64(o, 0x28) + 0x18);
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
}
// A recording ended by game code, not by EndRecording.
static void case_ended_outside(void) {
    uint64_t o = begun();
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    ((uint8_t *)(uintptr_t)o)[0x11] = 0;
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
}
// Its control memory added again, a handle is gone.
static void case_retired_handle(void) {
    uint64_t o = begun();
    NVN(nvnCommandBufferDrawArrays, o, 4, 0, 3);
    uint64_t h = NVN(nvnCommandBufferEndRecording, o);
    NVN(nvnCommandBufferAddControlMemory, o, rd64(o, 0x40), 4096);
    submit(&h, 1);
}
// A memory callback that adds no control memory where the driver needs some.
static void case_callback_adds_nothing(void) {
    setup();
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE);
    uint64_t o = command_buffer(0);
    cb_adds = 0;
    inline_begin(o);
}
// More vertex attributes than the console has.
static void case_vertex_attributes(void) {
    uint64_t o = begun();
    NVN(nvnCommandBufferBindVertexAttribState, o, 33, galloc(4 * 33, 8));
}
// A render target that is not an initialised texture.
static void case_not_a_texture(void) {
    uint64_t o = begun();
    uint64_t rts = galloc(8, 8);
    wr64(rts, 0, galloc(0x80, 16));
    NVN(nvnCommandBufferSetRenderTargets, o, 1, rts, 0, 0, 0);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: nvn_cmd_test <case> [arguments]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "ops")) case_ops();
    else if (!strcmp(cs, "begin")) case_begin();
    else if (!strcmp(cs, "callbacks")) case_callbacks();
    else if (!strcmp(cs, "loop") && argc > 2) case_loop(atoi(argv[2]));
    else if (!strcmp(cs, "probe_used")) case_probe_used(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_last_word")) case_probe_last_word(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_handle")) case_probe_handle(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_after_end")) case_probe_after_end(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_which_recording")) case_probe_which_recording(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_quiet")) case_probe_quiet(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_uncommitted")) case_probe_uncommitted(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "probe_default")) case_probe_default(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "time_rows")) case_time_rows(argc > 2 ? argv[2] : NULL);
    else if (!strcmp(cs, "inline_command_write")) case_inline_command_write();
    else if (!strcmp(cs, "control_moved")) case_control_moved();
    else if (!strcmp(cs, "ended_outside")) case_ended_outside();
    else if (!strcmp(cs, "retired_handle")) case_retired_handle();
    else if (!strcmp(cs, "callback_adds_nothing")) case_callback_adds_nothing();
    else if (!strcmp(cs, "vertex_attributes")) case_vertex_attributes();
    else if (!strcmp(cs, "not_a_texture")) case_not_a_texture();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
