// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's transfers on a real device (runtime/gpu/vk_replay_xfer.c, oot2/vk_replay.h), driven by
// tests/test_runtime_vk_xfer.py: `vk_xfer_test <case> [imports] [gated] [--pack PACK] [--mutant NAME]`, one case per process.
// They need a Vulkan device and skip without one, unless OOT2_REQUIRE_GPU is set; continuous integration has none.
//
// The recordings are made by hand (records of oot2/nvn_obj.h in host memory, as the front end writes them) and go through
// the back end's own submit, so the whole path runs: the replay, the recorder thread, the backing map, the images and the
// write-back. The CPU-visible pool is mirrored on this device (VRAM kept current from guest memory) unless `imports` makes
// the map import the guest's pages instead; `gated` holds the first submission behind a semaphore, so that the test can
// look at guest memory while the GPU has done nothing.
//
//   buffers        UpdateUniformBuffer, buffer copies in and out of VRAM and fills, into the CPU-visible pool: the
//                  destinations hold the new bytes once the fence passed and cpu_synced ran, and not before
//   textures       a buffer to a texture, texture to texture (copy), a clear of a whole level and of a part of one, and
//                  texture to buffer with the tight stride and with a row stride: each texel as expected
//   uniform_draw   a draw reads a uniform, UpdateUniformBuffer rewrites it, the next draw reads the new value (the
//                  snapshot of host memory is on); the picture is read back through a texture to buffer copy
//   storage        a compute module stores into the CPU-visible pool (the stores reach the CPU), reads a uniform an update
//                  rewrote between two dispatches, and an indirect dispatch takes the group counts an update wrote
//   storage_image  a compute module stores to textures registered at texture pool IDs, one standing in a pool the CPU
//                  cannot see (an image placed in its extent) and one in CPU-visible storage (an arena image): a copy
//                  reads each back and every texel is the stored value; then a hundred more submissions make and drop
//                  storage views while the completion thread destroys them, and the images still read back whole; and a
//                  store into a corner of a texture the CPU wrote leaves the rest as the CPU wrote it, read back by a
//                  later submission
//   virtual_texture a texture in a virtual pool mapped over CPU-visible memory (the title movie's planes): its texels go in
//                  by a buffer copy and come out; its image is not placed, and no backing is made over the pages behind
//                  it; the CPU then writes those pages and a copy takes them as a buffer, which brings their mirror
//                  current, and the texels read back as they were (a placed image would share its bytes with the mirror)
//
// --mutant nodrop (the drop rule off) and nodeclare (writes declared to no one) are the counter-runs: the case must fail.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "layout.h"
#include "oot2/backing.h"
#include "oot2/bindless.h"
#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_image.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3
#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define WIN GFX_GPU_WINDOW_BASE
typedef unsigned long long ull;

#define VS_KEY "5100000000000001.00000000"
#define PS_KEY "5100000000000002.00000000"
#define CS_KEY "5100000000000003.00000000"
#define HEAD_KEY "5100000000000005.00000000"   // reads the group counts the replay puts in its driver bank
#define IMG_KEY "5100000000000006.00000000"    // stores (x, y, 0, 1) to the texel (x, y) of the image at binding 25

static Cpu g_cpu;
static uint64_t g_heap;
static pool_t g_cpool, g_gpool;      // the CPU-visible pool and the GPU's own, 4 MiB each
#define P (g_heap + 4 * MiB)         // the CPU-visible pool's first byte
#define G WIN                        // the GPU pool's

static void guest_heap(uint64_t size) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = size;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1];
}

static pool_t make_pool(uint64_t mem, uint64_t size, uint32_t flags) {
    pool_t p = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = flags, .mem = (uint8_t *)(uintptr_t)mem, .size = size,
                .game_mem = (flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) ? NULL : (uint8_t *)(uintptr_t)mem};
    CHECK(gfx_vulkan_backend.pool_init(&p));
    return p;
}

// ---------- the gate: the first frame waits for a semaphore the test signals
static VkRecSubmit g_sub;
static uint64_t (*g_dev_end)(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits);
static VkSemaphore g_gate;
static int g_gated_done;

static uint64_t gated_end(const VkSemaphoreSubmitInfo *waits, uint32_t n) {
    if (g_gate && !g_gated_done) {
        g_gated_done = 1;
        CHECK(!n);
        VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_gate, .value = 1,
                                      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        return g_dev_end(&wait, 1);
    }
    return g_dev_end(waits, n);
}

static void open_gate(void) {
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = g_gate, .value = 1};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
}

static int g_imports, g_pack_open;

static void setup(int imports, int gated) {
    guest_heap(64 * MiB);
    gpu_vk_open();
    if (imports) backing_set_mirror_hook(NULL);
    g_imports = imports;
    g_cpool = make_pool(P, 4 * MiB, 0);
    g_gpool = make_pool(G, 4 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    for (uint64_t i = 0; i < 4 * MiB / 4; i++) ((volatile uint32_t *)(uintptr_t)P)[i] = 0xAAAAAAAAu;
    if (gated) {
        VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
        VkSemaphoreCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
        CHECK(vkCreateSemaphore(gpu_vk.device, &ci, NULL, &g_gate) == VK_SUCCESS);
    }
    vkrec_submit_device(&g_sub);
    g_dev_end = g_sub.end;
    g_sub.end = gated_end;
    VkRecOpen o = {.submit = &g_sub};
    vkrec_open(&o);
}

static void finish_and_report(const char *name) {
    gfx_vulkan_backend.finish();
    if (g_pack_open) vkp_close(), vkp_store_close();
    gfx_vulkan_backend.pool_fini(&g_gpool);
    gfx_vulkan_backend.pool_fini(&g_cpool);
    if (g_gate) vkDestroySemaphore(gpu_vk.device, g_gate, NULL);
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
    printf("ok %s\n", name);
}

// ---------- recordings
typedef struct {
    _Alignas(8) uint8_t b[1 << 16];
    size_t n;
} Rec;
static Rec g_rec;

static void put(Rec *r, uint32_t op, const void *payload, size_t bytes) {
    size_t size = (sizeof(op_hdr) + bytes + 7) & ~(size_t)7;
    CHECK(r->n + size <= sizeof r->b);
    op_hdr h = {op, (uint32_t)size};
    memset(r->b + r->n, 0, size);
    memcpy(r->b + r->n, &h, sizeof h);
    if (bytes) memcpy(r->b + r->n + sizeof h, payload, bytes);
    r->n += size;
}
#define PUT(r, op, ...) ({ __typeof__(__VA_ARGS__) p_ = __VA_ARGS__; put((r), (op), &p_, sizeof p_); })
static Rec *rec_begin(void) {
    g_rec.n = 0;
    put(&g_rec, OP_BEGIN, NULL, 0);
    return &g_rec;
}

static void update_ub(Rec *r, uint64_t dst, const void *data, uint32_t n) {
    uint8_t buf[sizeof(p_update_ub) + 256];
    CHECK(n <= 256);
    p_update_ub h = {.dst = dst, .size = n};
    memset(buf, 0, sizeof buf);
    memcpy(buf, &h, offsetof(p_update_ub, data));
    memcpy(buf + offsetof(p_update_ub, data), data, n);
    put(r, OP_UPDATE_UB, buf, offsetof(p_update_ub, data) + n);
}

// Submit the recording; returns the fence value.
static uint64_t submit(Rec *r) {
    put(r, OP_END, NULL, 0);
    uint64_t h = (uint64_t)(uintptr_t)r->b;
    uint64_t v = gfx_vulkan_backend.submit(&h, 1);
    CHECK(v);
    return v;
}

// Wait for the fence and run what a CPU that waited runs.
static void wait_done(uint64_t v) {
    CHECK(gfx_vulkan_backend.fence_wait(v, 20000000000ull) == 1);
    gfx_vulkan_backend.cpu_synced(v);
}

static tex_t tex_at(uint64_t storage, int format, uint32_t w, uint32_t h) {
    tex_t t;
    memset(&t, 0, sizeof t);
    t.magic = NVN_TEX_MAGIC;
    t.d.target = NVN_TEXTURE_TARGET_2D, t.d.format = format, t.d.width = w, t.d.height = h, t.d.depth = 1, t.d.levels = 1;
    t.d.swizzle_r = NVN_TEXTURE_SWIZZLE_R, t.d.swizzle_g = NVN_TEXTURE_SWIZZLE_G, t.d.swizzle_b = NVN_TEXTURE_SWIZZLE_B;
    t.d.swizzle_a = NVN_TEXTURE_SWIZZLE_A;
    t.mem = (uint8_t *)(uintptr_t)storage;
    static uint64_t serial;
    t.serial = ++serial;
    nvn_tex_layout(&t.d, &t.L);
    return t;
}
static nvn_tex_rec trec(uint64_t storage, int format, uint32_t w, uint32_t h) {
    nvn_tex_rec r;
    memset(&r, 0, sizeof r);
    r.tex = tex_at(storage, format, w, h);
    r.present = 1;
    return r;
}
static NVNcopyRegion region(int x, int y, int w, int h) { return (NVNcopyRegion){x, y, 0, w, h, 1}; }

static const volatile uint8_t *guest(uint64_t a) { return (const volatile uint8_t *)(uintptr_t)a; }
static int words_are(uint64_t a, uint64_t bytes, uint32_t word) {
    for (uint64_t i = 0; i < bytes / 4; i++)
        if (((const volatile uint32_t *)(uintptr_t)a)[i] != word) return 0;
    return 1;
}
static int bytes_are(uint64_t a, const uint8_t *want, uint64_t n) {
    for (uint64_t i = 0; i < n; i++)
        if (guest(a)[i] != want[i]) return 0;
    return 1;
}

// ---------- buffers
static void case_buffers(int imports, int gated) {
    setup(imports, gated);
    uint8_t pattern[256], update[16];
    for (int i = 0; i < 256; i++) pattern[i] = (uint8_t)(i ^ 0x5A);
    for (int i = 0; i < 16; i++) update[i] = (uint8_t)(0xC0 + i);
    memcpy((void *)(uintptr_t)P, pattern, sizeof pattern);
    Rec *r = rec_begin();
    update_ub(r, P + 0x400, update, sizeof update);
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = P, .dst = G, .size = 256});
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = G, .dst = P + 0x1000, .size = 256});
    PUT(r, OP_CLEAR_BUF, (p_clear_buf){.addr = P + 0x2000, .size = 0x100, .value = 0x11223344u});
    PUT(r, OP_CLEAR_BUF, (p_clear_buf){.addr = G + 0x100, .size = 0x40, .value = 0xCAFEBABEu});
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = G + 0x100, .dst = P + 0x3000, .size = 0x40});
    uint64_t v = submit(r);
    if (gated) {
        plat_sleep_ns(150 * 1000 * 1000);
        CHECK(gpu_sync_done() < v);
        // the GPU has done nothing: guest memory holds what the CPU left, in a mirror and in an import alike
        CHECK(words_are(P + 0x400, 16, 0xAAAAAAAAu) && words_are(P + 0x1000, 256, 0xAAAAAAAAu));
        CHECK(words_are(P + 0x2000, 0x100, 0xAAAAAAAAu) && words_are(P + 0x3000, 0x40, 0xAAAAAAAAu));
        gfx_vulkan_backend.cpu_synced(gpu_sync_done());
        printf("held by a semaphore: the destinations kept the CPU's bytes\n");
        open_gate();
    }
    wait_done(v);
    CHECK(bytes_are(P + 0x400, update, sizeof update));
    CHECK(bytes_are(P + 0x1000, pattern, sizeof pattern));
    CHECK(words_are(P + 0x2000, 0x100, 0x11223344u));
    CHECK(words_are(P + 0x3000, 0x40, 0xCAFEBABEu));
    // nothing beside the destinations changed
    CHECK(words_are(P + 0x410, 0x3F0, 0xAAAAAAAAu) && words_are(P + 0x1100, 0xF00, 0xAAAAAAAAu));
    CHECK(words_are(P + 0x2100, 0xF00, 0xAAAAAAAAu) && words_are(P + 0x3040, 0xFC0, 0xAAAAAAAAu));
    VkrStats st;
    vkr_stats(&st);
    BackingStats bs;
    backing_stats(&bs);
    printf("transfers %llu bytes %llu; %s: mirrors %llu write-backs %llu (%llu bytes)\n", (ull)st.transfers,
           (ull)st.transfer_bytes, imports ? "imports" : "mirror", (ull)bs.mirrors, (ull)bs.writebacks,
           (ull)bs.writeback_bytes);
    CHECK(st.transfers == 6);
    if (!imports) CHECK(bs.writebacks > 0);
    finish_and_report("buffers");
}

// ---------- textures
static void case_textures(int imports, int gated) {
    setup(imports, gated);
    uint8_t src[16 * 16 * 4];
    for (unsigned i = 0; i < sizeof src; i++) src[i] = (uint8_t)(i * 7 + 3);
    memcpy((void *)(uintptr_t)P, src, sizeof src);
    const uint64_t A = G + 0x10000, B = G + 0x20000;
    Rec *r = rec_begin();
    PUT(r, OP_COPY_B2T, (p_copy_b2t){.addr = P, .tex = trec(A, NVN_FORMAT_RGBA8, 16, 16), .r = region(0, 0, 16, 16)});
    PUT(r, OP_CLEAR_TEX, (p_clear_tex){.tex = trec(B, NVN_FORMAT_RGBA8, 16, 16), .r = region(0, 0, 0, 0), .mask = 0xF,
                                       .c = {.f = {0, 0, 1, 1}}});
    PUT(r, OP_COPY_T2T, (p_copy_t2t){.src = trec(A, NVN_FORMAT_RGBA8, 16, 16), .dst = trec(B, NVN_FORMAT_RGBA8, 16, 16),
                                     .sr = region(0, 0, 8, 8), .dr = region(4, 4, 8, 8)});
    PUT(r, OP_CLEAR_TEX, (p_clear_tex){.tex = trec(B, NVN_FORMAT_RGBA8, 16, 16), .r = region(0, 12, 16, 4), .mask = 0xF,
                                       .c = {.f = {1, 0, 0, 1}}});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = P + 0x1000, .tex = trec(B, NVN_FORMAT_RGBA8, 16, 16),
                                     .r = region(0, 0, 16, 16)});
    PUT(r, OP_COPY_STRIDE, (p_copy_stride){64});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = P + 0x2000, .tex = trec(A, NVN_FORMAT_RGBA8, 16, 16),
                                     .r = region(2, 2, 4, 4)});
    uint64_t v = submit(r);
    if (gated) {
        plat_sleep_ns(100 * 1000 * 1000);
        CHECK(words_are(P + 0x1000, 16 * 16 * 4, 0xAAAAAAAAu));
        open_gate();
    }
    wait_done(v);
    // B: blue, A's 8x8 at (4, 4), red in the last four rows
    uint8_t want[16 * 16 * 4];
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
            uint8_t *px = want + 4 * (y * 16 + x);
            if (y >= 12) memcpy(px, (uint8_t[]){0xFF, 0, 0, 0xFF}, 4);
            else if (y >= 4 && x >= 4 && x < 12 && y < 12) memcpy(px, src + 4 * ((y - 4) * 16 + (x - 4)), 4);
            else memcpy(px, (uint8_t[]){0, 0, 0xFF, 0xFF}, 4);
        }
    for (int y = 0; y < 16; y++)
        if (!bytes_are(P + 0x1000 + 64 * (uint64_t)y, want + 64 * y, 64)) {
            fprintf(stderr, "row %d of the copied-back texture B differs\n", y);
            exit(1);
        }
    printf("texture B read back: blue, A's block copied, a part cleared red\n");
    // A's 4x4 block at (2, 2), at a row stride of 64 bytes
    for (int y = 0; y < 4; y++) {
        CHECK(bytes_are(P + 0x2000 + 64 * (uint64_t)y, src + 4 * ((2 + y) * 16 + 2), 16));
        CHECK(words_are(P + 0x2000 + 64 * (uint64_t)y + 16, 48, 0xAAAAAAAAu));
    }
    printf("texture A's block read back at a row stride of 64 bytes\n");
    finish_and_report("textures");
}

// ---------- the stages of the synthetic pack and what they bind
static nvn_stage g_vs, g_ps, g_cs, g_head, g_img;
static ShaderBinds g_vs_binds, g_ps_binds, g_cs_binds, g_head_binds, g_img_binds;
static VkReplayOps g_ops;

static void stage(nvn_stage *s, const char *key, int kind) {
    memset(s, 0, sizeof *s);
    snprintf(s->key, sizeof s->key, "%s", key);
    s->kind = (uint8_t)kind;
    s->ctl_size = NVN_STAGE_CTL_MAX;
}
static void cbuf(ShaderBinds *b, uint8_t slot, uint16_t binding) { b->cb[b->ncb++] = (ShdCbuf){slot, 1, binding}; }

static const ShaderBinds *my_binds(const nvn_stage *s) {
    return s == &g_vs ? &g_vs_binds : s == &g_ps ? &g_ps_binds : s == &g_cs ? &g_cs_binds : s == &g_head ? &g_head_binds
           : s == &g_img ? &g_img_binds : vkr_device_ops()->binds(s);
}

// After setup: the pack opened, the stages and their bindings made (the pack's modules, tests/test_runtime_vk_pipeline.py:
// vertex bank at binding 1 and storage at 9, pixel bank at 14 and storage at 25; compute storage at 9 and bank at 1).
static void open_pack(const char *pack) {
    char why[512];
    if (!vkp_store_open(pack, NULL, why, sizeof why)) {
        fprintf(stderr, "the pack did not open: %s\n", why);
        exit(1);
    }
    vkp_open(NULL);
    g_pack_open = 1;
    stage(&g_vs, VS_KEY, NVN_STAGE_VS);
    stage(&g_ps, PS_KEY, NVN_STAGE_PS);
    stage(&g_cs, CS_KEY, NVN_STAGE_CS);
    stage(&g_head, HEAD_KEY, NVN_STAGE_CS);
    stage(&g_img, IMG_KEY, NVN_STAGE_CS);
    ShaderBinds *v = &g_vs_binds, *p = &g_ps_binds, *c = &g_cs_binds;
    memset(v, 0, sizeof *v), memset(p, 0, sizeof *p), memset(c, 0, sizeof *c);
    v->kind = SHD_VS, v->fits = 1;
    cbuf(v, 3, 1);
    v->sb[v->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 0, .binding = 9, .address_byte_offset = 0x110};
    p->kind = SHD_PS, p->fits = 1;
    cbuf(p, 4, 14);
    p->sb[p->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 0, .binding = 25, .address_byte_offset = 0x510};
    c->kind = SHD_CS, c->fits = 1, c->local_size[0] = 32, c->local_size[1] = c->local_size[2] = 1;
    cbuf(c, 3, 1);
    c->sb[c->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 1, .binding = 9, .address_byte_offset = 0x310};
    ShaderBinds *hd = &g_head_binds;
    memset(hd, 0, sizeof *hd);
    hd->kind = SHD_CS, hd->fits = 1, hd->local_size[0] = hd->local_size[1] = hd->local_size[2] = 1;
    cbuf(hd, 0, 2);   // the driver bank c[0]
    hd->sb[hd->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 1, .binding = 9, .address_byte_offset = 0x310};
    ShaderBinds *im = &g_img_binds;   // compute storage images are bindings 25 to 31 (oot2/vk_pipeline.h)
    memset(im, 0, sizeof *im);
    im->kind = SHD_CS, im->fits = 1, im->local_size[0] = im->local_size[1] = 8, im->local_size[2] = 1;
    im->img[im->nimg++] = (ShdImg){.set = 1, .write = 1, .view = BV_2D, .binding = 25, .c2_slot = 3};
    g_ops = *vkr_device_ops();
    g_ops.binds = my_binds;
    vkr_set_ops(&g_ops);
}

static p_bind_program program_of(nvn_stage *cs) {
    p_bind_program p;
    memset(&p, 0, sizeof p);
    p.stages = 1 << 5, p.nstage = 1, p.id[0] = cs;
    return p;
}

static p_bind_program program(int compute) {
    p_bind_program p;
    memset(&p, 0, sizeof p);
    if (compute) {
        p.stages = 1 << 5, p.nstage = 1, p.id[0] = &g_cs;
    } else {
        p.stages = 1 << 0 | 1 << 1, p.nstage = 2, p.id[0] = &g_vs, p.id[1] = &g_ps;
    }
    return p;
}

static int near4(const float *got, const float *want, float tol) {
    for (int i = 0; i < 4; i++) {
        float d = got[i] - want[i];
        if (d < -tol || d > tol) return 0;
    }
    return 1;
}

// ---------- uniform_draw
static void case_uniform_draw(int imports, int gated, const char *pack) {
    setup(imports, gated);
    open_pack(pack);
    static const VkpFormats F32 = {1, {VK_FORMAT_R32G32B32A32_SFLOAT}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY};
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32));   // no pipeline is made with a submission in flight
    const float colour[4] = {0.25f, 0.5f, 0.75f, 0.0f}, mul[4] = {2.0f, 2.0f, 2.0f, 0.5f};
    const float u1[4] = {0.5f, 0.25f, 0.0f, 0.0f}, u2[4] = {0.0f, 0.5f, 1.0f, 0.5f};
    memcpy((void *)(uintptr_t)(P + 0x2000), colour, 16);
    memcpy((void *)(uintptr_t)(P + 0x2100), mul, 16);
    memcpy((void *)(uintptr_t)(P + 0x3000), u1, 16);
    Rec *r = rec_begin();
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1;
    rt.color[0] = trec(G + 0x30000, NVN_FORMAT_RGBA32F, 32, 32);
    put(r, OP_SET_RT, &rt, sizeof rt);
    PUT(r, OP_CLEAR_COLOR, (p_clear_color){.index = 0, .mask = 0xF, .rgba = {-7, -7, -7, -7}});
    PUT(r, OP_BIND_PROGRAM, program(0));
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 0, .index = 0, .addr = P + 0x2000, .size = 16});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 1, .index = 0, .addr = P + 0x2100, .size = 16});
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 1, .index = 1, .addr = P + 0x3000, .size = 16});
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3});
    update_ub(r, P + 0x3000, u2, 16);
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = P + 0x4000, .tex = rt.color[0], .r = region(0, 0, 32, 32)});
    uint64_t v = submit(r);
    if (gated) {
        plat_sleep_ns(100 * 1000 * 1000);
        open_gate();
    }
    wait_done(v);
    // (storage colour + the padded input (0, 0, 0, 1) + the uniform) * the pixel storage
    float want[4], want_old[4], px[4];
    for (int i = 0; i < 4; i++) {
        float b = i == 3 ? 1.0f : 0.0f;
        want[i] = (colour[i] + b + u2[i]) * mul[i];
        want_old[i] = (colour[i] + b + u1[i]) * mul[i];
    }
    int bad = 0, stale = 0;
    for (int k = 0; k < 32 * 32; k++) {
        memcpy(px, (const void *)(uintptr_t)(P + 0x4000 + 16 * (uint64_t)k), 16);
        bad += !near4(px, want, 1e-5f);
        stale += near4(px, want_old, 1e-5f);
    }
    printf("pixels: %d of 1024 differ from the new uniform's value, %d show the old one; expected (%g %g %g %g)\n", bad, stale,
           want[0], want[1], want[2], want[3]);
    VkrStats st;
    vkr_stats(&st);
    printf("snapshots %llu drops %llu live uniforms %llu\n", (ull)st.snapshots, (ull)st.snapshot_drops,
           (ull)st.live_uniforms);
    CHECK(bad == 0);
    finish_and_report("uniform_draw");
}

// ---------- storage
static void case_storage(int imports, int gated, const char *pack) {
    setup(imports, gated);
    open_pack(pack);
    VkpStage cs = {.key = CS_KEY};
    CHECK(vkp_compute(&cs));
    const uint32_t k1[4] = {3, 0, 0, 0}, k2[4] = {5, 0, 0, 0}, counts[4] = {2, 1, 1, 0};
    memcpy((void *)(uintptr_t)(P + 0x3000), k1, 16);
    memcpy((void *)(uintptr_t)(P + 0x3100), counts, 16);
    Rec *r = rec_begin();
    PUT(r, OP_BIND_PROGRAM, program(1));
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x3000, .size = 16});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x5000, .size = 128 * 4});
    PUT(r, OP_DISPATCH, (p_dispatch){4, 1, 1, 0});
    update_ub(r, P + 0x3000, k2, 16);
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x6000, .size = 128 * 4});
    PUT(r, OP_DISPATCH, (p_dispatch){4, 1, 1, 0});
    // the group counts an update writes in this submission: the CPU cannot know them when it records
    update_ub(r, P + 0x3100, counts, 16);
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x7000, .size = 128 * 4});
    PUT(r, OP_DISPATCH_IND, (p_dispatch_ind){.ind = P + 0x3100});
    uint64_t v = submit(r);
    if (gated) {
        plat_sleep_ns(100 * 1000 * 1000);
        CHECK(words_are(P + 0x5000, 512, 0xAAAAAAAAu));
        open_gate();
    }
    wait_done(v);
    int bad = 0;
    for (uint32_t i = 0; i < 128; i++) {
        uint32_t a, b, c;
        memcpy(&a, (const void *)(uintptr_t)(P + 0x5000 + 4 * i), 4);
        memcpy(&b, (const void *)(uintptr_t)(P + 0x6000 + 4 * i), 4);
        memcpy(&c, (const void *)(uintptr_t)(P + 0x7000 + 4 * i), 4);
        bad += a != i * 3 || b != i * 5 || c != (i < 64 ? i * 5 : 0xAAAAAAAAu);
    }
    printf("128 stored values per dispatch: i*3 (first), i*5 (after the update), i*5 for the first 64 (indirect); %d wrong\n",
           bad);
    CHECK(bad == 0);
    finish_and_report("storage");
}

// ---------- storage_image: compute stores to textures by their texture pool ID
#define IMG_W 256
#define IMG_H 256
static const dpool_ref k_pool = {0x900000, 33025, 0};

// How many texels of the tight RGBA8 readback at `at` are not the (x, y, 0, 255) the module stores at texel (x, y).
static int texels_wrong(uint64_t at) {
    int bad = 0;
    for (uint32_t y = 0; y < IMG_H; y++)
        for (uint32_t x = 0; x < IMG_W; x++) {
            const uint8_t want[4] = {(uint8_t)x, (uint8_t)y, 0, 0xFF};
            bad += memcmp((const void *)(uintptr_t)(at + 4 * ((uint64_t)y * IMG_W + x)), want, 4) != 0;
        }
    return bad;
}

// One submission: the module stores to the first gx by gy groups of 8 by 8 texels of the texture at pool ID `id`, and a copy
// of `copy_of` (when given) reads that texture back, whole, into guest memory at `out` behind it.
static uint64_t store_part(uint32_t id, uint32_t gx, uint32_t gy, const nvn_tex_rec *copy_of, uint64_t out) {
    Rec *r = rec_begin();
    PUT(r, OP_SET_TEX_POOL, (p_set_pool){.pool = k_pool});
    PUT(r, OP_BIND_PROGRAM, program_of(&g_img));
    PUT(r, OP_BIND_IMG, (p_bind_handles){.stage = 5, .first = 3, .n = 1, .h = {id}});
    PUT(r, OP_DISPATCH, (p_dispatch){gx, gy, 1, 0});
    if (copy_of)
        PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = out, .tex = *copy_of,
                                         .r = region(0, 0, (int)copy_of->tex.d.width, (int)copy_of->tex.d.height)});
    return submit(r);
}
static uint64_t store_to(uint32_t id, const nvn_tex_rec *copy_of, uint64_t out) {
    return store_part(id, IMG_W / 8, IMG_H / 8, copy_of, out);
}
// A submission of its own that reads a texture back whole.
static uint64_t copy_back(const nvn_tex_rec *t, uint64_t out) {
    Rec *r = rec_begin();
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = out, .tex = *t, .r = region(0, 0, (int)t->tex.d.width, (int)t->tex.d.height)});
    return submit(r);
}

// A texture the CPU wrote, stored to in part: the stored texels are the module's, every other one is what the CPU wrote.
// The copy that reads it back is a submission of its own, after the one that stored, so the CPU's bytes cannot be
// uploaded again over the stores (they were uploaded before them, in the submission that bound the texture).
#define PART_W 128
#define PART_STORED 32
static void cpu_texel(uint32_t x, uint32_t y, uint8_t out[4]) {
    out[0] = (uint8_t)(0xC0 | (x & 0x3F)), out[1] = (uint8_t)(0x80 | (y & 0x3F)), out[2] = 0x55, out[3] = 0x80;
}
static int part_wrong(uint64_t at) {
    int bad = 0;
    for (uint32_t y = 0; y < PART_W; y++)
        for (uint32_t x = 0; x < PART_W; x++) {
            uint8_t want[4];
            if (x < PART_STORED && y < PART_STORED) memcpy(want, (uint8_t[]){(uint8_t)x, (uint8_t)y, 0, 0xFF}, 4);
            else cpu_texel(x, y, want);
            bad += memcmp((const void *)(uintptr_t)(at + 4 * ((uint64_t)y * PART_W + x)), want, 4) != 0;
        }
    return bad;
}

static void case_storage_image(int imports, const char *pack) {
    setup(imports, 0);
    open_pack(pack);
    VkpStage cs = {.key = IMG_KEY};
    CHECK(vkp_compute(&cs));
    const uint64_t placed_at = G + 0x40000, arena_at = P + 0x200000, out_placed = P + 0x100000, out_arena = P + 0x300000;
    const tex_t placed = tex_at(placed_at, NVN_FORMAT_RGBA8, IMG_W, IMG_H);
    const tex_t arena = tex_at(arena_at, NVN_FORMAT_RGBA8, IMG_W, IMG_H);
    dpool_ref pool = k_pool;
    bindless_texture_register(&pool, 5, &placed, NULL);
    bindless_texture_register(&pool, 6, &arena, NULL);
    // the texture in the GPU's pool has its image when registered, standing in the pool's extent; the one in CPU-visible
    // storage gets its arena image at its first bind
    VkbImage *img = vkb_image_get(&placed);
    CHECK(img && vkb_image_placed(img));
    CHECK(!vkb_image_get(&arena));
    const nvn_tex_rec cp = trec(placed_at, NVN_FORMAT_RGBA8, IMG_W, IMG_H), ca = trec(arena_at, NVN_FORMAT_RGBA8, IMG_W, IMG_H);
    store_to(5, &cp, out_placed);
    uint64_t v = store_to(6, &ca, out_arena);
    wait_done(v);
    int bad_placed = texels_wrong(out_placed), bad_arena = texels_wrong(out_arena);
    printf("placed image: %d of %d texels wrong\n", bad_placed, IMG_W * IMG_H);
    printf("arena image: %d of %d texels wrong\n", bad_arena, IMG_W * IMG_H);
    CHECK(bad_placed == 0 && bad_arena == 0);
    // a hundred submissions more, each with storage views of its own that the completion thread destroys once it is done
    // with them: nothing leaks (the validation layer counts the views left at the device's end), and both images still read
    // back whole
    uint64_t last = 0;
    for (int i = 0; i < 100; i++) last = store_to(i & 1 ? 6 : 5, NULL, 0);
    wait_done(last);
    memset((void *)(uintptr_t)out_placed, 0, 4ull * IMG_W * IMG_H);
    memset((void *)(uintptr_t)out_arena, 0, 4ull * IMG_W * IMG_H);
    store_to(5, &cp, out_placed);
    v = store_to(6, &ca, out_arena);
    wait_done(v);
    bad_placed = texels_wrong(out_placed), bad_arena = texels_wrong(out_arena);
    printf("after 100 more submissions: placed %d, arena %d texels wrong\n", bad_placed, bad_arena);
    CHECK(bad_placed == 0 && bad_arena == 0);
    // the module stores to a corner of a texture the CPU wrote, in one submission, and a later one reads the texture back
    const uint64_t part_at = P + 0x340000, out_part = P + 0x360000;
    const tex_t part = tex_at(part_at, NVN_FORMAT_RGBA8, PART_W, PART_W);
    static uint8_t cpu[PART_W * PART_W * 4];
    for (uint32_t y = 0; y < PART_W; y++)
        for (uint32_t x = 0; x < PART_W; x++) cpu_texel(x, y, cpu + 4 * ((size_t)y * PART_W + x));
    NVNcopyRegion whole = region(0, 0, PART_W, PART_W);
    vkb_write_texels(&part, NULL, &whole, cpu, 4 * PART_W, 4ull * PART_W * PART_W);
    bindless_texture_register(&pool, 7, &part, NULL);
    const nvn_tex_rec cq = trec(part_at, NVN_FORMAT_RGBA8, PART_W, PART_W);
    store_part(7, PART_STORED / 8, PART_STORED / 8, NULL, 0);
    v = copy_back(&cq, out_part);
    wait_done(v);
    int bad_part = part_wrong(out_part);
    printf("partial store into a texture the CPU wrote: %d of %d texels wrong\n", bad_part, PART_W * PART_W);
    CHECK(bad_part == 0);
    VkrStats st;
    vkr_stats(&st);
    printf("storage images bound %llu, written ranges declared %llu\n", (ull)st.storage_images, (ull)st.storage_writes);
    CHECK(st.storage_images == 105);
    finish_and_report("storage_image");
}

// ---------- a texture over memory the CPU can see
// A game maps a virtual pool in the GPU window over CPU-visible memory (the title movie's planes are such textures). The
// backing map keeps that memory as a mirror or an import, never as an extent, so the texture's image gets memory of its own
// and the image's creation makes no backing over the pages. Bound into the mirror's span instead, the image would share its
// bytes with the copy of the CPU's pages, and the refresh a CPU write calls for would write them over the texels.
static void case_virtual_texture(int imports) {
    setup(imports, 0);
    pool_t pc = make_pool(g_heap + 16 * MiB, 4 * MiB, NVN_MEMORY_POOL_FLAGS_PHYSICAL);   // CPU-visible memory to map over
    pool_t v = make_pool(WIN + 32 * MiB, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL);
    nvn_mapping_t m = {.physical_pool = (const NVNmemoryPool *)(const void *)&pc, .physical_offset = 2 * MiB,
                       .virtual_offset = 0, .size = 1 * MiB};
    CHECK(gfx_vulkan_backend.pool_map_virtual(&v, 1, &m));
    // the texture's storage, and the CPU-visible pages behind it
    const uint64_t T = WIN + 32 * MiB, behind = g_heap + 16 * MiB + 2 * MiB;
    const tex_t tex = tex_at(T, NVN_FORMAT_RGBA8, 16, 16);
    VkbImage *img = vkb_image_get(&tex);
    CHECK(img && !vkb_image_placed(img));
    BackingPage bp;
    backing_page(behind, &bp);
    BackingStats bs;
    backing_stats(&bs);
    printf("the texture over CPU-visible pages: its image is not placed; pages behind it backed %d, mirrors %llu, imports %llu\n",
           bp.kind != BACKING_NONE, (ull)bs.mirrors, (ull)bs.imports);
    CHECK(bp.kind == BACKING_NONE && bs.mirrors == 0 && bs.imports == 0);
    uint8_t src[16 * 16 * 4];
    for (unsigned i = 0; i < sizeof src; i++) src[i] = (uint8_t)(i * 7 + 3);
    memcpy((void *)(uintptr_t)P, src, sizeof src);
    Rec *r = rec_begin();
    PUT(r, OP_COPY_B2T, (p_copy_b2t){.addr = P, .tex = trec(T, NVN_FORMAT_RGBA8, 16, 16), .r = region(0, 0, 16, 16)});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = P + 0x1000, .tex = trec(T, NVN_FORMAT_RGBA8, 16, 16), .r = region(0, 0, 16, 16)});
    wait_done(submit(r));
    CHECK(bytes_are(P + 0x1000, src, sizeof src));
    printf("texels in by a buffer copy and out by another: as they went in\n");
    // the CPU writes the pages behind the texture, and a copy takes them as a buffer (their mirror, when there is one, is
    // brought current from them before the submission's own work); the texels, read after it, are as they were
    for (uint64_t i = 0; i < 64 * KiB / 4; i++) ((volatile uint32_t *)(uintptr_t)behind)[i] = 0xC0FFEE11u;
    r = rec_begin();
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = behind, .dst = P + 0x20000, .size = 0x1000});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = P + 0x30000, .tex = trec(T, NVN_FORMAT_RGBA8, 16, 16), .r = region(0, 0, 16, 16)});
    wait_done(submit(r));
    CHECK(words_are(P + 0x20000, 0x1000, 0xC0FFEE11u));
    int overwritten = !bytes_are(P + 0x30000, src, sizeof src);
    printf("%s: the CPU's writes behind the texture %s its texels\n", imports ? "imports" : "mirror",
           overwritten ? "overwrote" : "left");
    CHECK(!overwritten);
    backing_page(behind, &bp);
    CHECK(bp.kind == (imports ? BACKING_IMPORT : BACKING_MIRROR));   // the buffer binds backed the pages, apart from the image
    gfx_vulkan_backend.pool_fini(&v);
    gfx_vulkan_backend.pool_fini(&pc);
    finish_and_report("virtual_texture");
}

// ---------- a render target in CPU-visible storage
// The game renders into an RG8 texture whose storage the CPU can see (after the intro: a 260x260 one). Such storage has no
// image at registration; the pass takes its arena image, made current as a copy's is, so the CPU's bytes are uploaded first
// and the pass draws over them. A later submission's copy reads the pass's texels, not the CPU's again.
#define TGT_W 32
static void case_cpu_target(int imports) {
    setup(imports, 0);
    const uint64_t at = P + 0x200000, out = P + 0x220000;
    const tex_t t = tex_at(at, NVN_FORMAT_RG8, TGT_W, TGT_W);
    CHECK(!vkb_image_get(&t));
    static uint8_t cpu[TGT_W * TGT_W * 2];
    for (unsigned i = 0; i < sizeof cpu; i++) cpu[i] = (uint8_t)(0xA0 | (i & 0x1F));
    NVNcopyRegion whole = region(0, 0, TGT_W, TGT_W);
    vkb_write_texels(&t, NULL, &whole, cpu, 2 * TGT_W, sizeof cpu);
    Rec *r = rec_begin();
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1;
    rt.color[0] = trec(at, NVN_FORMAT_RG8, TGT_W, TGT_W);
    put(r, OP_SET_RT, &rt, sizeof rt);
    PUT(r, OP_SCISSOR, (p_scissor){.first = 0, .n = 1, .r = {{0, 0, TGT_W / 2, TGT_W}}});
    PUT(r, OP_CLEAR_COLOR, (p_clear_color){.index = 0, .mask = 0xF, .rgba = {51 / 255.0f, 153 / 255.0f, 0, 0}});
    wait_done(submit(r));
    wait_done(copy_back(&rt.color[0], out));
    int bad = 0;
    for (uint32_t y = 0; y < TGT_W; y++)
        for (uint32_t x = 0; x < TGT_W; x++) {
            const uint8_t *px = (const uint8_t *)(uintptr_t)(out + 2 * ((uint64_t)y * TGT_W + x));
            const uint8_t *was = cpu + 2 * ((size_t)y * TGT_W + x);
            const uint8_t want[2] = {x < TGT_W / 2 ? 51 : was[0], x < TGT_W / 2 ? 153 : was[1]};
            bad += memcmp(px, want, 2) != 0;
        }
    printf("render target in CPU-visible storage: %d of %d texels wrong\n", bad, TGT_W * TGT_W);
    CHECK(bad == 0);
    finish_and_report("cpu_target");
}

// ---------- head: the group counts in the driver bank c[0], which the module reads
static void case_head(int imports, int gated, const char *pack) {
    setup(imports, gated);
    open_pack(pack);
    VkpStage hd = {.key = HEAD_KEY};
    CHECK(vkp_compute(&hd));
    const uint32_t counts[4] = {2, 1, 1, 0};
    memcpy((void *)(uintptr_t)(P + 0x3100), counts, 16);
    Rec *r = rec_begin();
    PUT(r, OP_BIND_PROGRAM, program_of(&g_head));
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x8000, .size = 64});
    PUT(r, OP_DISPATCH, (p_dispatch){5, 6, 7, 0});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x8100, .size = 64});
    update_ub(r, P + 0x3100, counts, 16);
    PUT(r, OP_DISPATCH_IND, (p_dispatch_ind){.ind = P + 0x3100});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 0, .addr = P + 0x8200, .size = 64});
    PUT(r, OP_DISPATCH, (p_dispatch){3, 1, 1, 0});   // after an indirect one: a bank of its own again
    uint64_t v = submit(r);
    if (gated) open_gate();
    wait_done(v);
    uint32_t a[3], b[3], c[3];
    memcpy(a, (const void *)(uintptr_t)(P + 0x8000), 12);
    memcpy(b, (const void *)(uintptr_t)(P + 0x8100), 12);
    memcpy(c, (const void *)(uintptr_t)(P + 0x8200), 12);
    printf("group counts the module read: direct (%u %u %u), indirect after an update (%u %u %u), direct again (%u %u %u)\n",
           a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    CHECK(a[0] == 5 && a[1] == 6 && a[2] == 7 && b[0] == 2 && b[1] == 1 && b[2] == 1 && c[0] == 3 && c[1] == 1 && c[2] == 1);
    finish_and_report("head");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_xfer_test <case> [imports] [gated] [--pack PACK] [--mutant NAME]\n");
        return 2;
    }
    int imports = 0, gated = 0;
    const char *pack = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "imports")) imports = 1;
        else if (!strcmp(argv[i], "gated")) gated = 1;
        else if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!strcmp(argv[i], "--mutant") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "nodrop")) vkr_no_drop = 1;
            else if (!strcmp(m, "nodeclare")) vkr_no_declare = 1;
            else {
                fprintf(stderr, "unknown mutant %s\n", m);
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        return EXIT_NO_DEVICE;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "buffers")) case_buffers(imports, gated);
    else if (!strcmp(cs, "textures")) case_textures(imports, gated);
    else if (!strcmp(cs, "uniform_draw") && pack) case_uniform_draw(imports, gated, pack);
    else if (!strcmp(cs, "head") && pack) case_head(imports, gated, pack);
    else if (!strcmp(cs, "storage") && pack) case_storage(imports, gated, pack);
    else if (!strcmp(cs, "storage_image") && pack) case_storage_image(imports, pack);
    else if (!strcmp(cs, "virtual_texture")) case_virtual_texture(imports);
    else if (!strcmp(cs, "cpu_target")) case_cpu_target(imports);
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
