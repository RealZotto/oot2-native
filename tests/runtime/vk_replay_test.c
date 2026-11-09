// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the Vulkan back end's replay (oot2/vk_replay.h, runtime/gpu/vk_replay.c) that need no device:
// synthetic recordings (the records of oot2/nvn_obj.h, built here) replayed through stand-in ops (VkReplayOps: the
// pipeline builder, the stage descriptions, the backing map, render targets, the bindless tables and scratch memory)
// into the real recorder (oot2/vk_rec.h), whose calls are stand-ins that write each packet as one line of text. Driven
// by tests/test_runtime_vk_replay.py: `vk_replay_test <case> [args] [--mutant NAME]`. Each case runs in its own process
// and prints its two logs: "op ..." lines for what the replay asked of its ops (the submitting thread), then "pkt ..."
// lines for the calls the recorder made (its thread). A stop must end in one named fatal, which the driver checks.
//
// The fake address space: HOST is real memory of this program that the backing map would import (the replay reads it:
// uniform snapshots and readable indirect records), MIRROR and VRAM are GPU-only addresses. g_vram supplies bytes only
// to the stand-in transfer command, so the mesh-indirect test can model a GPU-written {count, first} record.
// Addresses in the logs are printed from their region's start, so the logs do not depend on where this program loads.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/bindless.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/ledger.h"
#include "oot2/mesh_abi.h"
#include "oot2/nvn.h"
#include "oot2/nvn_obj.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/shader_desc.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef unsigned long long ull;

// ---------- the fake address space
#define HOST_BYTES (1u << 20)
static _Alignas(4096) uint8_t g_host[HOST_BYTES];
static uint8_t g_vram[HOST_BYTES];
#define HOST ((uint64_t)(uintptr_t)g_host)
#define MIRROR 0x30000000ull
#define VRAM 0x40000000ull
#define REGION_BYTES 0x10000000ull
#define BUF_HOST 0xB1u
#define BUF_MIRROR 0xB2u
#define BUF_VRAM 0xB3u
#define BUF_SCRATCH 0x5Cu

static uint8_t g_arena[8u << 20];
static uint32_t g_arena_used;
static int g_mesh_pushes;

// An address as the logs print it: H+, M+ or V+ and the offset from the region's start.
static void where(char *out, size_t cap, uint64_t a) {
    if (a - HOST < HOST_BYTES) snprintf(out, cap, "H+0x%llx", (ull)(a - HOST));
    else if (a - MIRROR < REGION_BYTES) snprintf(out, cap, "M+0x%llx", (ull)(a - MIRROR));
    else if (a - VRAM < REGION_BYTES) snprintf(out, cap, "V+0x%llx", (ull)(a - VRAM));
    else snprintf(out, cap, "0x%llx", (ull)a);
}
// ---------- the two logs
typedef struct {
    char *p;
    size_t n, cap;
} Log;
static Log g_ops, g_pkt;
static void lput(Log *l, const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    CHECK(k >= 0);
    size_t n = (size_t)k < sizeof line ? (size_t)k : sizeof line - 1;
    if (l->n + n + 1 > l->cap) {
        l->cap = (l->n + n + 1) * 2;
        l->p = realloc(l->p, l->cap);
        CHECK(l->p);
    }
    memcpy(l->p + l->n, line, n);
    l->n += n;
    l->p[l->n] = 0;
}

// ---------- the stages and their bindings
// g_vs_raw and g_cs_raw: stages that reach memory through the page table; g_cs_img: a compute stage that stores to an image
static nvn_stage g_vs, g_ps, g_cs, g_ts, g_ms, g_mps, g_vs_raw, g_cs_raw, g_cs_img;
static ShaderBinds g_vs_binds, g_ps_binds, g_cs_binds, g_ts_binds, g_ms_binds, g_mps_binds, g_vs_raw_binds,
    g_cs_raw_binds, g_cs_img_binds;

static void stage(nvn_stage *s, const char *key, int kind, uint32_t c1_at, uint32_t c1_end) {
    memset(s, 0, sizeof *s);
    snprintf(s->key, sizeof s->key, "%s", key);
    s->kind = (uint8_t)kind;
    s->ctl_size = NVN_STAGE_CTL_MAX;
    memcpy(s->ctl + 0x40, &c1_at, 4);
    memcpy(s->ctl + 0x44, &c1_end, 4);
}
static void cbuf(ShaderBinds *b, uint8_t slot, uint16_t binding) { b->cb[b->ncb++] = (ShdCbuf){slot, 1, binding}; }

static void make_stages(void) {
    stage(&g_vs, "00000000000000a1.00000001", NVN_STAGE_VS, 0x100, 0x140);
    stage(&g_ps, "00000000000000b2.00000001", NVN_STAGE_PS, 0, 0);
    stage(&g_cs, "00000000000000c3.00000001", NVN_STAGE_CS, 0, 0);
    stage(&g_ts, "00000000000000d1.00000001", NVN_STAGE_TS, 0, 0);
    stage(&g_ms, "00000000000000d2.00000001", NVN_STAGE_MS, 0, 0);
    stage(&g_mps, "00000000000000d3.00000001", NVN_STAGE_PS, 0, 0);
    // the compute stage's c[0] layout (vk_nvn.h, VknC0Layout): two root words, three uniform rows, two storage rows
    g_cs.ctl[0xA9] = 2, g_cs.ctl[0xB1] = 3, g_cs.ctl[0xB2] = 2;

    ShaderBinds *v = &g_vs_binds;
    memset(v, 0, sizeof *v);
    v->kind = SHD_VS, v->fits = 1;
    cbuf(v, 0, 1), cbuf(v, 3, 2), cbuf(v, 1, 3), cbuf(v, 2, 4), cbuf(v, 24, 5);
    v->vin[v->nvin++] = (ShdVin){0, 0, 4, SHD_FLOAT};
    v->vin[v->nvin++] = (ShdVin){1, 0, 4, SHD_SINT};   // no attribute bound there: it reads (0, 0, 0, 1), as integers

    ShaderBinds *p = &g_ps_binds;
    memset(p, 0, sizeof *p);
    p->kind = SHD_PS, p->fits = 1;
    cbuf(p, 0, 6), cbuf(p, 4, 7), cbuf(p, 2, 8);
    p->sb[p->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 1, .binding = 9, .address_byte_offset = 0x510};
    p->tex[p->ntex++] = (ShdTex){.via = SHD_VIA_PUSH_SLOT, .kind = SHD_TEX_COMBINED, .view = BV_2D, .binding = 0,
                                 .array_length = 1, .push_slot = 2, .push_offset = 24, .c2_slot = 5};
    p->tex[p->ntex++] = (ShdTex){.via = SHD_VIA_POOL, .kind = SHD_TEX_COMBINED, .view = BV_2D, .push_slot = -1,
                                 .push_offset = -1, .c2_slot = -1};

    ShaderBinds *c = &g_cs_binds;
    memset(c, 0, sizeof *c);
    c->kind = SHD_CS, c->fits = 1, c->local_size[0] = 64, c->local_size[1] = c->local_size[2] = 1;
    cbuf(c, 0, 1), cbuf(c, 2, 2), cbuf(c, 24, 3);
    c->sb[c->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 0, .binding = 4, .address_byte_offset = 0x320};

    g_ts_binds.kind = SHD_TS, g_ts_binds.fits = 1;
    g_ts_binds.cb[g_ts_binds.ncb++] = (ShdCbuf){0, 2, 0};
    g_ms_binds.kind = SHD_MS, g_ms_binds.fits = 1;
    cbuf(&g_ms_binds, 0, 1);
    g_ms_binds.ntex = 1;
    g_ms_binds.tex[0] = (ShdTex){.via = SHD_VIA_PUSH_SLOT, .kind = SHD_TEX_COMBINED, .view = BV_2D, .set = 1,
                                 .binding = 0, .array_length = 1, .push_slot = 0, .push_offset = 16, .c2_slot = 0};
    g_mps_binds.kind = SHD_PS, g_mps_binds.fits = 1;

    stage(&g_cs_raw, "00000000000000c4.00000001", NVN_STAGE_CS, 0, 0);   // the same layout as g_cs, and the page table
    g_cs_raw.ctl[0xA9] = 2, g_cs_raw.ctl[0xB1] = 3, g_cs_raw.ctl[0xB2] = 2;
    g_cs_raw_binds = g_cs_binds;
    g_cs_raw_binds.raw = 1;
    stage(&g_vs_raw, "00000000000000a2.00000001", NVN_STAGE_VS, 0x100, 0x140);   // and g_vs, with the page table
    g_vs_raw_binds = g_vs_binds;
    g_vs_raw_binds.raw = 1;
    // the same layout as g_cs, and a module that stores to the image its c[2] image handle 2 names, at binding 25 (compute: storage images 25 to 31)
    stage(&g_cs_img, "00000000000000c5.00000001", NVN_STAGE_CS, 0, 0);
    g_cs_img.ctl[0xA9] = 2, g_cs_img.ctl[0xB1] = 3, g_cs_img.ctl[0xB2] = 2;
    g_cs_img_binds = g_cs_binds;
    g_cs_img_binds.img[g_cs_img_binds.nimg++] = (ShdImg){.set = 1, .write = 1, .view = BV_2D, .binding = 25, .c2_slot = 2};
}

// ---------- the stand-in ops
static int g_skip_next;   // the next graphics get answers NULL, a draw to skip (the miss path's answer)
static VkPipeline op_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *f,
                              int may_skip) {
    (void)may_skip;
    if (task)
        lput(&g_ops, "op graphics task=%s mesh=%s pixel=%s colors=%u c0=%d depth=%d stencil=%d\n", task->key, vs->key,
             ps->key, f->ncolor, f->ncolor ? (int)f->color[0] : 0, (int)f->depth, (int)f->stencil);
    else
        lput(&g_ops, "op graphics %s %s colors=%u c0=%d depth=%d stencil=%d\n", vs->key, ps->key, f->ncolor,
         f->ncolor ? (int)f->color[0] : 0, (int)f->depth, (int)f->stencil);
    if (g_skip_next) {
        g_skip_next = 0;
        return VK_NULL_HANDLE;
    }
    return (VkPipeline)(uintptr_t)0x9A;
}
static VkPipeline op_compute(const VkpStage *cs) {
    lput(&g_ops, "op compute %s\n", cs->key);
    return (VkPipeline)(uintptr_t)0x9C;
}
static const ShaderBinds *op_binds(const VkpStage *stage) {
    const nvn_stage *s = stage->shape;
    return s == &g_vs ? &g_vs_binds : s == &g_ps ? &g_ps_binds : s == &g_cs ? &g_cs_binds
           : s == &g_ts ? &g_ts_binds : s == &g_ms ? &g_ms_binds : s == &g_mps ? &g_mps_binds
           : s == &g_vs_raw ? &g_vs_raw_binds : s == &g_cs_raw ? &g_cs_raw_binds : s == &g_cs_img ? &g_cs_img_binds : NULL;
}
static int g_log_resolves;   // a case that checks the order of the resolves against the raw path's begin sets it
static void op_resolve(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    (void)value;
    if (g_log_resolves) lput(&g_ops, "op resolve 0x%llx+0x%llx\n", (ull)addr, (ull)size);
    memset(out, 0, sizeof *out);
    out->address = addr;
    if (addr - HOST < HOST_BYTES) {
        CHECK(addr - HOST + size <= HOST_BYTES);
        out->buffer = BUF_HOST, out->offset = addr - HOST, out->kind = BACKING_IMPORT;
    } else if (addr - MIRROR < REGION_BYTES) {
        out->buffer = BUF_MIRROR, out->offset = addr - MIRROR, out->kind = BACKING_MIRROR;
    } else if (addr - VRAM < REGION_BYTES) {
        out->buffer = BUF_VRAM, out->offset = addr - VRAM, out->kind = BACKING_EXTENT;
    } else {
        fprintf(stderr, "resolve of 0x%llx, outside the fake regions\n", (ull)addr);
        exit(1);
    }
}
static void op_gpu_wrote(uint64_t addr, uint64_t size, uint64_t value) {
    char w[64];
    where(w, sizeof w, addr);
    lput(&g_ops, "op wrote %s 0x%llx frame %llu\n", w, (ull)size, (ull)value);
}
static int op_target(const nvn_tex_rec *t, int storage, uint64_t value, VkrTarget *out) {
    (void)value;
    if (storage) return 0;
    if (t->tex.d.format == NVN_FORMAT_DEPTH24_STENCIL8)
        *out = (VkrTarget){(VkImageView)(uintptr_t)0x7D00, VK_FORMAT_D24_UNORM_S8_UINT, t->tex.d.width,
                           t->tex.d.height, 1, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0x7D00};
    else
        *out = (VkrTarget){(VkImageView)(uintptr_t)(0x7100 + (uint32_t)t->tex.d.format), VK_FORMAT_R8G8B8A8_UNORM,
                           t->tex.d.width, t->tex.d.height, 1, VK_IMAGE_ASPECT_COLOR_BIT,
                           0x7100 + (uint32_t)t->tex.d.format};
    return 1;
}
// A texture's image: the handle names the texture's serial; storage in host memory has none (the texture arena's).
static int op_image(const nvn_tex_rec *t, uint64_t value, VkrImage *out) {
    (void)value;
    memset(out, 0, sizeof *out);
    if ((uint64_t)(uintptr_t)t->tex.mem - HOST < HOST_BYTES) return 0;
    const nvn_texdesc *d = &t->tex.d;
    out->image = (VkImage)(uintptr_t)(0x1000 + t->tex.serial);
    out->type = VK_IMAGE_TYPE_2D, out->aspect = out->all_aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    out->format = VK_FORMAT_R8G8B8A8_UNORM;
    out->width = d->width, out->height = d->height, out->depth = 1, out->layers = 1, out->samples = 1;
    out->levels = d->levels ? (uint32_t)d->levels : 1;
    if (d->target == NVN_TEXTURE_TARGET_2D_ARRAY) out->layers = d->depth;
    if (d->target == NVN_TEXTURE_TARGET_3D) out->type = VK_IMAGE_TYPE_3D, out->depth = d->depth;
    if (d->samples > 1) out->samples = (uint32_t)d->samples;
    if (d->format == NVN_FORMAT_DEPTH24_STENCIL8) {
        out->format = VK_FORMAT_D24_UNORM_S8_UINT, out->aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        out->all_aspects = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    } else if (d->format == NVN_FORMAT_DEPTH32F) {
        out->format = VK_FORMAT_D32_SFLOAT, out->aspect = out->all_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    return 1;
}
static VkDescriptorSet op_acquire(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers) {
    lput(&g_ops, "op acquire frame %llu textures %d samplers %d\n", (ull)value, textures != NULL, samplers != NULL);
    return (VkDescriptorSet)(uintptr_t)0x5E7;
}
static uint32_t op_image_slot(uint32_t id, int type) { return 1000 + id * 8 + (uint32_t)type; }
// A storage view by texture ID: 9 is an image placed in its storage, a range of VRAM that the stores write; 10 is an arena
// image, which holds the stores itself (no range); nothing is registered at any other ID. The view names the ID.
#define STORE_AT (VRAM + 0x50000)
#define STORE_BYTES 0x4000
static VkImageView op_storage_image(uint32_t id, int type, uint64_t value, uint64_t *storage, uint64_t *bytes) {
    CHECK(type == BV_2D);
    lput(&g_ops, "op storage_image id %u frame %llu\n", id, (ull)value);
    *storage = *bytes = 0;
    if (id != 9 && id != 10) return VK_NULL_HANDLE;
    if (id == 9) *storage = STORE_AT, *bytes = STORE_BYTES;
    return (VkImageView)(uintptr_t)(0x5D00 + id);
}
static void *op_scratch(uint64_t value, uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset) {
    (void)value;
    uint32_t at = (g_arena_used + align - 1) & ~(align - 1);
    CHECK(at + bytes <= sizeof g_arena);
    g_arena_used = at + bytes;
    *buffer = (VkBuffer)(uintptr_t)BUF_SCRATCH, *offset = at;
    return g_arena + at;
}
static VkDeviceAddress op_buffer_address(VkBuffer buffer) {
    CHECK((uint64_t)(uintptr_t)buffer == BUF_SCRATCH);
    return 0x50000000;
}
static VkDescriptorSet op_task_set(uint64_t value, uint32_t n, const VkWriteDescriptorSet *writes) {
    lput(&g_ops, "op task_set frame %llu n=%u\n", (ull)value, n);
    for (uint32_t i = 0; i < n; i++)
        lput(&g_ops, "op task_write b=%u type=%u\n", writes[i].dstBinding, writes[i].descriptorType);
    return (VkDescriptorSet)(uintptr_t)0x5E8;
}
// The page table's words of a frame: the same four for every stage (the device's own begin does the work once).
static void op_raw(uint64_t value, uint32_t push[4]) {
    lput(&g_ops, "op raw frame %llu\n", (ull)value);
    push[0] = 0x7AB1E000u, push[1] = 0x7Eu, push[2] = 0x80u, push[3] = 0x400u;
}
static void op_frame_begin(void) { lput(&g_ops, "op frame_begin\n"); }
static void op_frame_end(uint64_t value) { lput(&g_ops, "op frame_end %llu\n", (ull)value); }
// Query pools named by their frame; query i of any pool counted 10 * (i + 1) samples.
static VkQueryPool op_queries(uint64_t value) {
    lput(&g_ops, "op queries frame %llu\n", (ull)value);
    return (VkQueryPool)(uintptr_t)(0x9900 + value);
}
static int op_cpu_reads(uint64_t addr, uint64_t size) { return addr - HOST < HOST_BYTES && addr - HOST + size <= HOST_BYTES; }
static void op_query_results(VkQueryPool pool, uint32_t n, uint64_t *out) {
    for (uint32_t i = 0; i < n; i++) out[i] = 10ull * (i + 1);
    printf("query results pool 0x%llx n %u\n", (ull)(uintptr_t)pool, n);
}

static const VkReplayOps g_fake_ops = {
    .graphics = op_graphics, .compute = op_compute,
    .graphics_layout = (VkPipelineLayout)(uintptr_t)0x6100, .compute_layout = (VkPipelineLayout)(uintptr_t)0x6200,
    .binds = op_binds, .resolve = op_resolve, .gpu_wrote = op_gpu_wrote, .target = op_target, .image = op_image,
    .acquire = op_acquire,
    .image_slot = op_image_slot, .storage_image = op_storage_image, .scratch = op_scratch,
    .buffer_address = op_buffer_address, .task_set = op_task_set, .raw = op_raw,
    .frame_begin = op_frame_begin,
    .frame_end = op_frame_end, .queries = op_queries, .query_results = op_query_results, .cpu_reads = op_cpu_reads};

// ---------- the stand-in recorder calls: one line per call
#define FAKE_CB ((VkCommandBuffer)(uintptr_t)0xC0FFEE00u)
static void pr_f(double v) { lput(&g_pkt, " %g", v); }
static void pr_p(const void *v) { lput(&g_pkt, " 0x%llx", (ull)(uintptr_t)v); }
static void pr_i(long long v) { lput(&g_pkt, " %lld", v); }
#define PRA(x) _Generic((x), float: pr_f, double: pr_f, VkPipeline: pr_p, VkBuffer: pr_p, VkQueryPool: pr_p, default: pr_i)(x);
#define NARG_(_1, _2, _3, _4, _5, _6, N, ...) N
#define NARG(...) NARG_(__VA_ARGS__, 6, 5, 4, 3, 2, 1)
#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define EACH_1(m, x) m(x)
#define EACH_2(m, x, ...) m(x) EACH_1(m, __VA_ARGS__)
#define EACH_3(m, x, ...) m(x) EACH_2(m, __VA_ARGS__)
#define EACH_4(m, x, ...) m(x) EACH_3(m, __VA_ARGS__)
#define EACH_5(m, x, ...) m(x) EACH_4(m, __VA_ARGS__)
#define EACH_6(m, x, ...) m(x) EACH_5(m, __VA_ARGS__)
#define EACH(m, ...) CAT(EACH_, NARG(__VA_ARGS__))(m, __VA_ARGS__)

#define FAKE_F(name, params, packargs, ...)                                                                            \
    static void VKAPI_CALL fake_##name(VkCommandBuffer cb, VKREC_UNPAREN params) {                                     \
        CHECK(cb == FAKE_CB);                                                                                          \
        lput(&g_pkt, "pkt %s", #name);                                                                                 \
        EACH(PRA, VKREC_UNPAREN packargs)                                                                              \
        lput(&g_pkt, "\n");                                                                                            \
    }
#define FAKE_D(name, kind, params, packargs, ...) FAKE_F(name, params, packargs, __VA_ARGS__)
VKREC_FIXED(FAKE_F)
VKREC_DRAWS(FAKE_D)

static void VKAPI_CALL fake_BindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                               uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd,
                                               const uint32_t *d) {
    CHECK(cb == FAKE_CB && n == 1 && nd == 0);
    (void)d;
    lput(&g_pkt, "pkt BindDescriptorSets %d 0x%llx %u 0x%llx\n", (int)bp, (ull)(uintptr_t)l, first, (ull)(uintptr_t)s[0]);
}

// A buffer's bytes as the replay left them in scratch memory: the non-zero 32-bit words, by byte offset.
static void words(const VkDescriptorBufferInfo *b) {
    if ((uintptr_t)b->buffer != BUF_SCRATCH) return;
    const uint8_t *p = g_arena + b->offset;
    int shown = 0, all_zero = 1;
    for (VkDeviceSize o = 0; o + 4 <= b->range; o += 4) {
        uint32_t w;
        memcpy(&w, p + o, 4);
        if (!w) continue;
        all_zero = 0;
        if (shown++ < 24) lput(&g_pkt, " 0x%llx=%#x", (ull)o, w);
    }
    if (all_zero) lput(&g_pkt, " zero");
}

static void VKAPI_CALL fake_PushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l,
                                                 uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt PushDescriptorSetKHR %d 0x%llx set %u writes %u\n", (int)bp, (ull)(uintptr_t)l, set, n);
    for (uint32_t i = 0; i < n; i++) {
        CHECK(w[i].descriptorCount == 1);
        if (w[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {   // a storage image: its view, in the general layout
            CHECK(w[i].pImageInfo && !w[i].pBufferInfo && !w[i].pImageInfo->sampler);
            lput(&g_pkt, "pkt  w b=%u SI view=0x%llx layout=%d\n", w[i].dstBinding,
                 (ull)(uintptr_t)w[i].pImageInfo->imageView, (int)w[i].pImageInfo->imageLayout);
            continue;
        }
        CHECK(w[i].pBufferInfo);
        const VkDescriptorBufferInfo *b = w[i].pBufferInfo;
        lput(&g_pkt, "pkt  w b=%u %s buf=0x%llx off=0x%llx range=0x%llx", w[i].dstBinding,
             w[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? "UB" : "SB", (ull)(uintptr_t)b->buffer,
             (ull)b->offset, (ull)b->range);
        words(b);
        lput(&g_pkt, "\n");
    }
}
static void VKAPI_CALL fake_PushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off,
                                          uint32_t size, const void *v) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt PushConstants 0x%llx %#x %u %u", (ull)(uintptr_t)l, st, off, size);
    for (uint32_t i = 0; i + 4 <= size; i += 4) {
        uint32_t x;
        memcpy(&x, (const uint8_t *)v + i, 4);
        if (x) lput(&g_pkt, " %u=%#x", i, x);
    }
    if (g_mesh_pushes && off == 0 && size >= MESH_FIRST_TASK_PUSH_OFFSET + sizeof(uint64_t)) {
        VkDeviceAddress address;
        memcpy(&address, (const uint8_t *)v + MESH_FIRST_TASK_PUSH_OFFSET, sizeof address);
        if (address >= 0x50000000 && address - 0x50000000 + sizeof(uint32_t) <= sizeof g_arena) {
            uint32_t first;
            memcpy(&first, g_arena + (address - 0x50000000), sizeof first);
            lput(&g_pkt, " first_task=%u", first);
        } else {
            lput(&g_pkt, " first_task=unavailable");
        }
    }
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_BindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *buf,
                                               const VkDeviceSize *off, const VkDeviceSize *sz, const VkDeviceSize *st) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt BindVertexBuffers2 %u %u", first, n);
    for (uint32_t i = 0; i < n; i++) lput(&g_pkt, " 0x%llx+0x%llx", (ull)(uintptr_t)buf[i], (ull)off[i]);
    lput(&g_pkt, "%s\n", sz || st ? " sizes" : "");
}
static void VKAPI_CALL fake_SetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport *v) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetViewportWithCount %u", n);
    for (uint32_t i = 0; i < n; i++)
        lput(&g_pkt, " [%g %g %g %g %g %g]", v[i].x, v[i].y, v[i].width, v[i].height, v[i].minDepth, v[i].maxDepth);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_SetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D *r) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetScissorWithCount %u", n);
    for (uint32_t i = 0; i < n; i++)
        lput(&g_pkt, " [%d %d %u %u]", r[i].offset.x, r[i].offset.y, r[i].extent.width, r[i].extent.height);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_SetVertexInputEXT(VkCommandBuffer cb, uint32_t nb, const VkVertexInputBindingDescription2EXT *b,
                                              uint32_t na, const VkVertexInputAttributeDescription2EXT *a) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetVertexInputEXT");
    for (uint32_t i = 0; i < nb; i++)
        lput(&g_pkt, " [b%u stride=%u rate=%d div=%u]", b[i].binding, b[i].stride, (int)b[i].inputRate, b[i].divisor);
    for (uint32_t i = 0; i < na; i++)
        lput(&g_pkt, " [l%u b%u fmt=%d off=%u]", a[i].location, a[i].binding, (int)a[i].format, a[i].offset);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_SetColorBlendEnableEXT(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBool32 *v) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetColorBlendEnableEXT %u %u", first, n);
    for (uint32_t i = 0; i < n; i++) lput(&g_pkt, " %u", v[i]);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_SetColorBlendEquationEXT(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                     const VkColorBlendEquationEXT *e) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetColorBlendEquationEXT %u %u", first, n);
    for (uint32_t i = 0; i < n; i++)
        lput(&g_pkt, " [%d %d %d %d %d %d]", (int)e[i].srcColorBlendFactor, (int)e[i].dstColorBlendFactor,
             (int)e[i].colorBlendOp, (int)e[i].srcAlphaBlendFactor, (int)e[i].dstAlphaBlendFactor, (int)e[i].alphaBlendOp);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_SetColorWriteMaskEXT(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                 const VkColorComponentFlags *m) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetColorWriteMaskEXT %u %u", first, n);
    for (uint32_t i = 0; i < n; i++) lput(&g_pkt, " %#x", m[i]);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_SetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt SetBlendConstants %g %g %g %g\n", c[0], c[1], c[2], c[3]);
}
static void VKAPI_CALL fake_PipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *d) {
    CHECK(cb == FAKE_CB && d->memoryBarrierCount == 1 && !d->bufferMemoryBarrierCount && !d->imageMemoryBarrierCount);
    const VkMemoryBarrier2 *m = d->pMemoryBarriers;
    lput(&g_pkt, "pkt PipelineBarrier2 full=%d\n",
         m->srcStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT && m->dstStageMask == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT &&
             m->srcAccessMask == VK_ACCESS_2_MEMORY_WRITE_BIT &&
             m->dstAccessMask == (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
}
static void att(const VkRenderingAttachmentInfo *a) {
    lput(&g_pkt, " [0x%llx layout=%d load=%d store=%d]", (ull)(uintptr_t)a->imageView, (int)a->imageLayout,
         (int)a->loadOp, (int)a->storeOp);
}
static void VKAPI_CALL fake_BeginRendering(VkCommandBuffer cb, const VkRenderingInfo *r) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt BeginRendering %d,%d %ux%u layers=%u colors=%u", r->renderArea.offset.x, r->renderArea.offset.y,
         r->renderArea.extent.width, r->renderArea.extent.height, r->layerCount, r->colorAttachmentCount);
    for (uint32_t i = 0; i < r->colorAttachmentCount; i++) att(&r->pColorAttachments[i]);
    if (r->pDepthAttachment) lput(&g_pkt, " depth"), att(r->pDepthAttachment);
    if (r->pStencilAttachment) lput(&g_pkt, " stencil"), att(r->pStencilAttachment);
    lput(&g_pkt, "\n");
}
static void VKAPI_CALL fake_EndRendering(VkCommandBuffer cb) {
    CHECK(cb == FAKE_CB);
    lput(&g_pkt, "pkt EndRendering\n");
}
static void VKAPI_CALL fake_ClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr,
                                             const VkClearRect *r) {
    CHECK(cb == FAKE_CB && na == 1 && nr == 1);
    lput(&g_pkt, "pkt ClearAttachments aspect=%#x att=%u", a->aspectMask, a->colorAttachment);
    if (a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
        lput(&g_pkt, " color=%g,%g,%g,%g", a->clearValue.color.float32[0], a->clearValue.color.float32[1],
             a->clearValue.color.float32[2], a->clearValue.color.float32[3]);
    else
        lput(&g_pkt, " depth=%g stencil=%u", a->clearValue.depthStencil.depth, a->clearValue.depthStencil.stencil);
    lput(&g_pkt, " rect=%d,%d %ux%u layers=%u+%u\n", r->rect.offset.x, r->rect.offset.y, r->rect.extent.width,
         r->rect.extent.height, r->baseArrayLayer, r->layerCount);
}
static void layers_text(const VkImageSubresourceLayers *l) {
    lput(&g_pkt, "aspect=%#x level=%u layers=%u+%u", l->aspectMask, l->mipLevel, l->baseArrayLayer, l->layerCount);
}
static void region_text(const VkBufferImageCopy *r) {
    lput(&g_pkt, " off=%llu row=%u imgh=%u ", (ull)r->bufferOffset, r->bufferRowLength, r->bufferImageHeight);
    layers_text(&r->imageSubresource);
    lput(&g_pkt, " at=%d,%d,%d ext=%ux%ux%u\n", r->imageOffset.x, r->imageOffset.y, r->imageOffset.z, r->imageExtent.width,
         r->imageExtent.height, r->imageExtent.depth);
}
static void VKAPI_CALL fake_CopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) {
    CHECK(cb == FAKE_CB && n == 1);
    if ((uintptr_t)s == BUF_VRAM && (uintptr_t)d == BUF_SCRATCH && r->srcOffset + r->size <= sizeof g_vram &&
        r->dstOffset + r->size <= sizeof g_arena)
        memcpy(g_arena + r->dstOffset, g_vram + r->srcOffset, (size_t)r->size);
    lput(&g_pkt, "pkt CopyBuffer 0x%llx->0x%llx %llu->%llu size=%llu\n", (ull)(uintptr_t)s, (ull)(uintptr_t)d,
         (ull)r->srcOffset, (ull)r->dstOffset, (ull)r->size);
}
static void VKAPI_CALL fake_CopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout l, uint32_t n,
                                              const VkBufferImageCopy *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    lput(&g_pkt, "pkt CopyBufferToImage buf=0x%llx img=0x%llx", (ull)(uintptr_t)s, (ull)(uintptr_t)d);
    region_text(r);
}
static void VKAPI_CALL fake_CopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout l, VkBuffer d, uint32_t n,
                                              const VkBufferImageCopy *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    lput(&g_pkt, "pkt CopyImageToBuffer img=0x%llx buf=0x%llx", (ull)(uintptr_t)s, (ull)(uintptr_t)d);
    region_text(r);
}
static void VKAPI_CALL fake_CopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                      const VkImageCopy *r) {
    CHECK(cb == FAKE_CB && n == 1 && sl == VK_IMAGE_LAYOUT_GENERAL && dl == VK_IMAGE_LAYOUT_GENERAL);
    lput(&g_pkt, "pkt CopyImage 0x%llx->0x%llx src ", (ull)(uintptr_t)s, (ull)(uintptr_t)d);
    layers_text(&r->srcSubresource);
    lput(&g_pkt, " at=%d,%d,%d dst ", r->srcOffset.x, r->srcOffset.y, r->srcOffset.z);
    layers_text(&r->dstSubresource);
    lput(&g_pkt, " at=%d,%d,%d ext=%ux%ux%u\n", r->dstOffset.x, r->dstOffset.y, r->dstOffset.z, r->extent.width,
         r->extent.height, r->extent.depth);
}
static void VKAPI_CALL fake_BlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                      const VkImageBlit *r, VkFilter f) {
    CHECK(cb == FAKE_CB && n == 1 && sl == VK_IMAGE_LAYOUT_GENERAL && dl == VK_IMAGE_LAYOUT_GENERAL);
    lput(&g_pkt, "pkt BlitImage 0x%llx->0x%llx filter=%d src ", (ull)(uintptr_t)s, (ull)(uintptr_t)d, (int)f);
    layers_text(&r->srcSubresource);
    lput(&g_pkt, " %d,%d,%d..%d,%d,%d dst ", r->srcOffsets[0].x, r->srcOffsets[0].y, r->srcOffsets[0].z, r->srcOffsets[1].x,
         r->srcOffsets[1].y, r->srcOffsets[1].z);
    layers_text(&r->dstSubresource);
    lput(&g_pkt, " %d,%d,%d..%d,%d,%d\n", r->dstOffsets[0].x, r->dstOffsets[0].y, r->dstOffsets[0].z, r->dstOffsets[1].x,
         r->dstOffsets[1].y, r->dstOffsets[1].z);
}
static void range_text(const VkImageSubresourceRange *r) {
    lput(&g_pkt, " aspect=%#x level=%u+%u layers=%u+%u\n", r->aspectMask, r->baseMipLevel, r->levelCount, r->baseArrayLayer,
         r->layerCount);
}
static void VKAPI_CALL fake_ClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c,
                                            uint32_t n, const VkImageSubresourceRange *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    lput(&g_pkt, "pkt ClearColorImage 0x%llx color=%g,%g,%g,%g", (ull)(uintptr_t)i, c->float32[0], c->float32[1],
         c->float32[2], c->float32[3]);
    range_text(r);
}
static void VKAPI_CALL fake_ClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                   const VkClearDepthStencilValue *v, uint32_t n,
                                                   const VkImageSubresourceRange *r) {
    CHECK(cb == FAKE_CB && n == 1 && l == VK_IMAGE_LAYOUT_GENERAL);
    lput(&g_pkt, "pkt ClearDepthStencilImage 0x%llx depth=%g stencil=%u", (ull)(uintptr_t)i, v->depth, v->stencil);
    range_text(r);
}
static void VKAPI_CALL fake_SetCheckpointNV(VkCommandBuffer cb, const void *m) {
    (void)cb, (void)m;
}

// ---------- the stand-in submissions, over a test's timeline
static uint64_t g_counter;
static PlatMutex g_counter_lock = PLAT_MUTEX_INIT;
static PlatCond g_counter_cond = PLAT_COND_INIT;
static void counter_raise(uint64_t v) {
    plat_mutex_lock(&g_counter_lock);
    if (v > g_counter) g_counter = v;
    plat_cond_broadcast(&g_counter_cond);
    plat_mutex_unlock(&g_counter_lock);
}
static uint64_t counter_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    uint64_t t0 = plat_now_ns();
    plat_mutex_lock(&g_counter_lock);
    while (g_counter < value) {
        uint64_t spent = plat_now_ns() - t0;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_counter_cond, &g_counter_lock, timeout_ns - spent);
    }
    uint64_t r = g_counter;
    plat_mutex_unlock(&g_counter_lock);
    return r;
}
static uint64_t g_cur_value;
static int g_hold_timeline;
static uint64_t fake_begin(int kind) {
    CHECK(kind == GPU_VK_REPLAY);
    g_cur_value = gpu_sync_begin();
    return g_cur_value;
}
static VkCommandBuffer fake_cmd(void) { return FAKE_CB; }
static uint64_t fake_end(const VkSemaphoreSubmitInfo *waits, uint32_t n) {
    CHECK(n == 0 && !waits);
    uint64_t v = g_cur_value;
    lput(&g_pkt, "pkt submit %llu\n", (ull)v);
    gpu_sync_end();
    if (!g_hold_timeline) counter_raise(v);   // the stand-in device is normally done at once
    return v;
}

static VkRecCalls g_calls;
static VkRecSubmit g_submit;
static void open_fakes(void) {
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
    memset(&g_calls, 0, sizeof g_calls);
#define SET_F(name, ...) g_calls.name = fake_##name;
#define SET_D(name, kind, ...) g_calls.name = fake_##name;
#define SET_C(name) g_calls.name = fake_##name;
    VKREC_FIXED(SET_F)
    VKREC_DRAWS(SET_D)
    VKREC_CUSTOM(SET_C)
#undef SET_F
#undef SET_D
#undef SET_C
    g_calls.SetCheckpointNV = fake_SetCheckpointNV;
    g_submit = (VkRecSubmit){fake_begin, fake_cmd, fake_end, 0};
    VkRecOpen o = {1u << 20, 2, &g_calls, &g_submit};
    vkrec_open(&o);
    vkr_set_ops(&g_fake_ops);
    make_stages();
}
static void close_fakes(void) {
    vkrec_idle();
    uint32_t nf;
    const VkRecFrame *fr = vkrec_frames(&nf);
    for (uint32_t i = 0; i < nf; i++)   // the packets' size, for the bytes a draw costs (the recorder study: 2.8 KB a draw)
        printf("frame %llu packets %u bytes %llu\n", (ull)fr[i].value, fr[i].packets, (ull)fr[i].bytes);
    gpu_sync_finish();
    vkrec_close();
    gpu_sync_stop();
    fputs(g_ops.p ? g_ops.p : "", stdout);
    fputs(g_pkt.p ? g_pkt.p : "", stdout);
}

// ---------- recordings
typedef struct {
    _Alignas(8) uint8_t b[1 << 16];
    size_t n;
} Rec;
static Rec g_rec[2];

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
static Rec *rec_begin(int i) {
    Rec *r = &g_rec[i];
    r->n = 0;
    put(r, OP_BEGIN, NULL, 0);
    return r;
}
static uint64_t rec_end(Rec *r) {
    put(r, OP_END, NULL, 0);
    return (uint64_t)(uintptr_t)r->b;
}

static p_set_rt one_target(int with_depth) {
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1;
    rt.color[0].present = 1;
    rt.color[0].tex.d = (nvn_texdesc){.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8, .width = 64, .height = 32,
                                      .depth = 1, .levels = 1};
    if (with_depth) {
        rt.depth.present = 1;
        rt.depth.tex.d = rt.color[0].tex.d;
        rt.depth.tex.d.format = NVN_FORMAT_DEPTH24_STENCIL8;
    }
    return rt;
}

static p_bind_program program(int compute) {
    p_bind_program p;
    memset(&p, 0, sizeof p);
    if (compute) {
        p.stages = 1 << 5, p.nstage = 1, p.code[0] = MIRROR + 0x6000, p.id[0] = &g_cs;
    } else {
        p.stages = 1 << 0 | 1 << 1, p.nstage = 2;
        p.code[0] = MIRROR + 0x1000, p.id[0] = &g_vs;
        p.code[1] = MIRROR + 0x2000, p.id[1] = &g_ps;
    }
    return p;
}
static p_bind_program program_mesh(void) {
    p_bind_program p;
    memset(&p, 0, sizeof p);
    p.stages = (1u << NVN_STAGE_TS) | (1u << NVN_STAGE_MS) | (1u << NVN_STAGE_PS);
    p.nstage = 3;
    p.code[0] = MIRROR + 0x3000, p.id[0] = &g_ts;
    p.code[1] = MIRROR + 0x4000, p.id[1] = &g_ms;
    p.code[2] = MIRROR + 0x5000, p.id[2] = &g_mps;
    return p;
}
static p_bind_program program_mesh_only(void) {
    p_bind_program p = program_mesh();
    p.stages &= ~(1u << NVN_STAGE_TS);
    p.nstage = 2;
    p.code[0] = p.code[1], p.id[0] = p.id[1];
    p.code[1] = p.code[2], p.id[1] = p.id[2];
    return p;
}

// The same programs with the stage that reaches memory through the page table: the compute stage, or the vertex stage.
static p_bind_program program_raw(int compute) {
    p_bind_program p = program(compute);
    p.id[0] = compute ? &g_cs_raw : &g_vs_raw;
    return p;
}

// The graphics state of the draws: one target, a viewport and a scissor, a vertex stream, a uniform buffer of the
// vertex stage in host memory (an import: copied at submit), one of the pixel stage in a mirror (bound live), a
// storage buffer the pixel stage writes, a texture.
static void graphics_state(Rec *r) {
    p_set_rt rt = one_target(0);
    put(r, OP_SET_RT, &rt, sizeof rt);
    PUT(r, OP_VIEWPORTS, (p_viewports){.first = 0, .n = 1, .v = {{0, 0, 64, 32}}});
    PUT(r, OP_SCISSOR, (p_scissor){.first = 0, .n = 1, .r = {{8, 4, 16, 8}}});
    PUT(r, OP_BIND_PROGRAM, program(0));
    va_st a = {.fmt = NVN_FORMAT_RGBA32F, .offset = 0, .stream = 0};
    p_bind_vattr va = {.first = 0, .n = 1};
    memcpy(&va.st[0], &a, sizeof a);
    put(r, OP_BIND_VATTR, &va, sizeof va);
    PUT(r, OP_BIND_VSTREAM, (p_bind_vstream){.first = 0, .n = 1, .st = {{16, 0}}});
    PUT(r, OP_BIND_VBUF, (p_bind_vbuf){.first = 0, .n = 1, .r = {{MIRROR + 0x3000, 64}}});
    for (uint32_t i = 0; i < 16; i++) memcpy(g_host + 0x100 + 4 * i, &(uint32_t){0xA0 + i}, 4);
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 0, .index = 0, .addr = HOST + 0x100, .size = 64});
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 1, .index = 1, .addr = MIRROR + 0x4000, .size = 256});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 1, .index = 0, .addr = VRAM + 0x104, .size = 0x40});
    PUT(r, OP_BIND_TEX, (p_bind_handles){.stage = 1, .first = 5, .n = 1, .h = {7u | 3u << NVN_HANDLE_TEXTURE_BITS}});
}

static void case_graphics(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3, .base_instance = 0, .instances = 0});
    PUT(r, OP_DRAW_IDX, (p_draw_idx){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .itype = NVN_INDEX_TYPE_UNSIGNED_SHORT, .count = 6,
                                     .base_vertex = 2, .base_instance = 1, .instances = 2, .ibuf = MIRROR + 0x5000});
    PUT(r, OP_DRAW_IDX_IND, (p_draw_idx_ind){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .itype = NVN_INDEX_TYPE_UNSIGNED_INT,
                                             .ibuf = MIRROR + 0x5000, .ind = VRAM + 0x200});
    PUT(r, OP_CLEAR_COLOR, (p_clear_color){.index = 0, .mask = 0xF, .rgba = {1, 0, 0, 1}});
    PUT(r, OP_BARRIER, (p_barrier){0, 0});
    uint64_t h = rec_end(r);
    uint64_t v = vkr_submit(&h, 1);
    // the bound state lasts: a second submission draws again with nothing bound anew, and its uniform copy is its own
    for (uint32_t i = 0; i < 16; i++) memcpy(g_host + 0x100 + 4 * i, &(uint32_t){0xB0 + i}, 4);
    r = rec_begin(1);
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 3, .count = 3, .base_instance = 0, .instances = 1});
    uint64_t h2 = rec_end(r);
    uint64_t v2 = vkr_submit(&h2, 1);
    CHECK(v2 == v + 1);
    uint32_t n;
    const VkrWrite *w = vkr_writes(v, &n);
    CHECK(w && n == 3);
    for (uint32_t i = 0; i < n; i++) CHECK(w[i].addr == VRAM + 0x100 && w[i].size == 0x44 && w[i].draw == i + 1 && !w[i].image);
    VkrStats st;
    vkr_stats(&st);
    printf("stats frames %llu draws %llu clears %llu passes %llu snapshots %llu live %llu writes %llu\n", (ull)st.frames,
           (ull)st.draws, (ull)st.clears, (ull)st.passes, (ull)st.snapshots, (ull)st.live_uniforms,
           (ull)st.storage_writes);
    close_fakes();
    printf("ok graphics\n");
}

// A draw whose pipeline the miss path skipped (the op answers NULL) writes nothing at all: no bind, no state, no draw.
// The draw after it is complete (it carries every bind a first draw needs), and the recorded-against-replayed count
// accounts for both.
static void case_skip(int silent) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    g_skip_next = 1;
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3, .instances = 1});
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 3, .count = 3, .instances = 1});
    if (silent) PUT(r, OP_RENDER_ENABLE, (p_flag){.value = 0});   // a draw the game turned rendering off for
    if (silent) PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 6, .count = 3, .instances = 1});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    VkrStats st;
    vkr_stats(&st);
    printf("stats draw_ops %llu draws %llu skipped %llu render_off %llu lost %llu\n", (ull)st.draw_ops, (ull)st.draws,
           (ull)st.skipped, (ull)st.render_off, (ull)st.lost);
    close_fakes();
    printf("ok %s\n", silent ? "skip_render_off" : "skip");
}

static void case_depth(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    p_set_rt rt = one_target(1);
    put(r, OP_SET_RT, &rt, sizeof rt);
    PUT(r, OP_CLEAR_DS, (p_clear_ds){.depth = 1.0f, .depth_mask = 1, .stencil = 5, .stencil_mask = 0xFF});
    PUT(r, OP_SCISSOR, (p_scissor){.first = 0, .n = 1, .r = {{0, 0, 32, 8}}});
    PUT(r, OP_CLEAR_COLOR, (p_clear_color){.index = 0, .mask = 0xF, .rgba = {0, 0.5f, 0, 1}});
    PUT(r, OP_SET_RT, one_target(0));
    PUT(r, OP_CLEAR_COLOR, (p_clear_color){.index = 0, .mask = 0xF, .rgba = {0, 0, 1, 1}});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok depth\n");
}

static void case_compute(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    PUT(r, OP_BIND_PROGRAM, program(1));
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 5, .index = 2, .addr = MIRROR + 0x7000, .size = 0x80});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 1, .addr = VRAM + 0x300, .size = 0x20});
    PUT(r, OP_PUSH_CONST, (p_push_const){.compute = 1, .table = 2, .offset = 8, .bytes = 8, .w = {0x11, 0x22}});
    PUT(r, OP_DISPATCH, (p_dispatch){4, 2, 1, 0});
    const uint32_t groups[3] = {5, 6, 7};
    memcpy(g_host + 0x200, groups, sizeof groups);
    PUT(r, OP_DISPATCH_IND, (p_dispatch_ind){.ind = HOST + 0x200});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok compute\n");
}

// A raw-access dispatch or draw asks for its page table's words before any of its binds resolves, so that nothing the
// frame binds from then on becomes a mirror (oot2/backing.h, The raw path); the words reach its push constants, and every
// raw-access stage of the frame asks again and gets the same. Frame 1 is two dispatches, frame 2 a draw.
static void case_raw_first(void) {
    open_fakes();
    g_log_resolves = 1;
    Rec *r = rec_begin(0);
    PUT(r, OP_BIND_PROGRAM, program_raw(1));
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 5, .index = 2, .addr = MIRROR + 0x7000, .size = 0x80});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 5, .index = 1, .addr = VRAM + 0x300, .size = 0x20});
    PUT(r, OP_DISPATCH, (p_dispatch){4, 2, 1, 0});
    PUT(r, OP_DISPATCH, (p_dispatch){1, 1, 1, 0});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    r = rec_begin(1);
    graphics_state(r);
    PUT(r, OP_BIND_PROGRAM, program_raw(0));
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3, .base_instance = 0, .instances = 1});
    uint64_t h2 = rec_end(r);
    vkr_submit(&h2, 1);
    close_fakes();
    printf("ok raw_first\n");
}

// ---------- storage images
static p_bind_program program_img(void) {
    p_bind_program p = program(1);
    p.id[0] = &g_cs_img;
    return p;
}
static p_bind_handles image_handle(uint64_t h) { return (p_bind_handles){.stage = 5, .first = 2, .n = 1, .h = {h}}; }

// A compute module that stores to the image its c[2] image handle 2 names: the replay takes the view of that texture ID
// from the ops and binds it at the module's binding, in the general layout. Frame 1: ID 9, an image placed in its
// storage, whose storage the replay declares written (as an image write) for the dispatch; ID 10, an arena image, which
// holds the stores itself: bound, and nothing declared. Frame 2: the module only reads ID 9: bound, nothing declared.
static void case_storage_image(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    PUT(r, OP_BIND_PROGRAM, program_img());
    PUT(r, OP_BIND_IMG, image_handle(9));
    PUT(r, OP_DISPATCH, (p_dispatch){1, 1, 1, 0});
    PUT(r, OP_BIND_IMG, image_handle(10));
    PUT(r, OP_DISPATCH, (p_dispatch){1, 1, 1, 0});
    uint64_t h = rec_end(r);
    uint64_t v = vkr_submit(&h, 1);
    g_cs_img_binds.img[0].write = 0;
    r = rec_begin(1);
    PUT(r, OP_BIND_IMG, image_handle(9));
    PUT(r, OP_DISPATCH, (p_dispatch){1, 1, 1, 0});
    uint64_t h2 = rec_end(r);
    uint64_t v2 = vkr_submit(&h2, 1);
    CHECK(v2 == v + 1);
    uint32_t n;
    const VkrWrite *w = vkr_writes(v, &n);
    CHECK(w && n == 1 && w[0].addr == STORE_AT && w[0].size == STORE_BYTES && w[0].image && w[0].draw == 1);
    vkr_writes(v2, &n);
    CHECK(n == 0);
    VkrStats st;
    vkr_stats(&st);
    printf("stats storage_images %llu writes %llu\n", (ull)st.storage_images, (ull)st.storage_writes);
    close_fakes();
    printf("ok storage_image\n");
}

// A storage image the replay cannot bind stops by name: an ID nothing is registered at, a buffer texture, an image in a
// set other than 1, a handle slot the translator could not tell (-1), and ops that make no storage view.
static VkReplayOps g_ops_no_storage;
static void case_storage_stop(const char *what) {
    open_fakes();
    ShdImg *m = &g_cs_img_binds.img[0];
    uint64_t id = 9;
    if (!strcmp(what, "unregistered")) id = 11;
    else if (!strcmp(what, "buffer")) m->view = BV_NONE;
    else if (!strcmp(what, "set")) m->set = 2;
    else if (!strcmp(what, "slot")) m->c2_slot = -1;
    else if (!strcmp(what, "no_ops")) {
        g_ops_no_storage = g_fake_ops;
        g_ops_no_storage.storage_image = NULL;
        vkr_set_ops(&g_ops_no_storage);
    } else {
        fprintf(stderr, "unknown stop %s\n", what);
        exit(2);
    }
    Rec *r = rec_begin(0);
    PUT(r, OP_BIND_PROGRAM, program_img());
    PUT(r, OP_BIND_IMG, image_handle(id));
    PUT(r, OP_DISPATCH, (p_dispatch){1, 1, 1, 0});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

// A uniform range copied at submit and a GPU write over it in the same submission, in either order: the drop rule. The
// vertex stage's uniform buffer and the pixel stage's storage buffer (which its module writes) are one range of host
// memory. Read first: the first draw reads the copy and its pixel stage then writes the range, which drops the copy, so
// the second draw reads the live range. Write first: the first draw writes it with the uniform unbound, and the draw
// that binds it afterwards reads it live too.
static void case_conflict(int write_first) {
    open_fakes();
    g_ps_binds.sb[0].address_byte_offset = 0x510;
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 1, .index = 0, .addr = HOST + 0x100, .size = 0x40});
    if (write_first) {
        PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 0, .index = 0, .addr = 0, .size = 0});
        PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .count = 3});
        PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 0, .index = 0, .addr = HOST + 0x100, .size = 64});
    } else {
        PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .count = 3});
    }
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .count = 3});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    VkrStats st;
    vkr_stats(&st);
    printf("stats snapshots %llu drops %llu live %llu writes %llu\n", (ull)st.snapshots, (ull)st.snapshot_drops,
           (ull)st.live_uniforms, (ull)st.storage_writes);
    close_fakes();
    printf("ok conflict_%s\n", write_first ? "write_first" : "read_first");
}

// ---------- transfers
#define TEX_A (VRAM + 0x10000)
#define TEX_B (VRAM + 0x20000)
#define TEX_ARRAY (VRAM + 0x30000)
#define TEX_DEPTH (VRAM + 0x40000)

static nvn_tex_rec texture(uint64_t storage, uint64_t serial, int format, int target, uint32_t w, uint32_t h, uint32_t d) {
    nvn_tex_rec t;
    memset(&t, 0, sizeof t);
    t.present = 1;
    t.tex.d = (nvn_texdesc){.target = target, .format = format, .width = w, .height = h, .depth = d, .levels = 1};
    t.tex.mem = (uint8_t *)(uintptr_t)storage;
    t.tex.serial = serial;
    return t;
}
static nvn_tex_rec tex_a(void) { return texture(TEX_A, 1, NVN_FORMAT_RGBA8, NVN_TEXTURE_TARGET_2D, 64, 32, 1); }
static NVNcopyRegion region(int x, int y, int z, int w, int h, int d) { return (NVNcopyRegion){x, y, z, w, h, d}; }

// The transfers in one submission: updates, buffer copies and fills (writes into a mirror, an import and VRAM), texture
// copies with the channel's strides, blits, clears of a whole level and of a part of one, and a dispatch after them.
static void case_transfers(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    uint8_t up[offsetof(p_update_ub, data) + 8] = {0};
    p_update_ub head = {.dst = MIRROR + 0x900, .size = 8};
    memcpy(up, &head, offsetof(p_update_ub, data));
    for (int i = 0; i < 8; i++) up[offsetof(p_update_ub, data) + i] = (uint8_t)(i + 1);
    put(r, OP_UPDATE_UB, up, sizeof up);
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = HOST + 0x400, .dst = MIRROR + 0xA00, .size = 0x100});
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = HOST + 0x400, .dst = MIRROR + 0xA40, .size = 0x40});   // inside the write above
    PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = VRAM + 0x1000, .dst = HOST + 0x800, .size = 0x40});
    PUT(r, OP_CLEAR_BUF, (p_clear_buf){.addr = VRAM + 0x40, .size = 0x20, .value = 0xDEAD});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = MIRROR + 0x1000, .tex = tex_a(), .r = region(8, 4, 0, 16, 4, 1)});
    PUT(r, OP_COPY_STRIDE, (p_copy_stride){0x100});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = MIRROR + 0x2000, .tex = tex_a(), .r = region(8, 4, 0, 16, 4, 1)});
    PUT(r, OP_COPY_STRIDE, (p_copy_stride){0});
    PUT(r, OP_COPY_IMAGE_STRIDE, (p_copy_stride){0x400});
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = MIRROR + 0x4000,
                                     .tex = texture(TEX_ARRAY, 2, NVN_FORMAT_RGBA8, NVN_TEXTURE_TARGET_2D_ARRAY, 8, 8, 4),
                                     .r = region(0, 0, 1, 8, 8, 2)});
    PUT(r, OP_COPY_IMAGE_STRIDE, (p_copy_stride){0});
    PUT(r, OP_COPY_B2T, (p_copy_b2t){.addr = HOST + 0x2000, .tex = tex_a(), .r = region(0, 0, 0, 8, 8, 1)});
    nvn_tex_rec b = texture(TEX_B, 3, NVN_FORMAT_RGBA8, NVN_TEXTURE_TARGET_2D, 64, 32, 1);
    PUT(r, OP_COPY_T2T, (p_copy_t2t){.src = tex_a(), .dst = b, .sr = region(0, 0, 0, 32, 16, 1), .dr = region(4, 4, 0, 32, 16, 1)});
    PUT(r, OP_COPY_T2T, ((p_copy_t2t){.src = tex_a(), .dst = b, .sr = region(0, 0, 0, 16, 8, 1),
                                      .dr = region(0, 0, 0, 32, 16, 1), .flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT}));
    PUT(r, OP_CLEAR_TEX, (p_clear_tex){.tex = tex_a(), .r = region(0, 0, 0, 0, 0, 0), .mask = 0xF, .c = {.f = {1, 0, 0, 1}}});
    PUT(r, OP_CLEAR_TEX, (p_clear_tex){.tex = tex_a(), .r = region(8, 4, 0, 16, 8, 1), .mask = 0xF, .c = {.f = {0, 1, 0, 1}}});
    p_clear_tex ds = {.tex = texture(TEX_DEPTH, 4, NVN_FORMAT_DEPTH24_STENCIL8, NVN_TEXTURE_TARGET_2D, 64, 32, 1),
                      .r = region(0, 0, 0, 64, 32, 1), .mask = 0xF};
    ds.c.f[0] = 1.0f, ds.c.u[1] = 7;
    put(r, OP_CLEAR_TEX, &ds, sizeof ds);
    PUT(r, OP_BIND_PROGRAM, program(1));
    PUT(r, OP_DISPATCH, (p_dispatch){1, 1, 1, 0});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    VkrStats st;
    vkr_stats(&st);
    printf("stats transfers %llu bytes %llu drops %llu writes %llu\n", (ull)st.transfers, (ull)st.transfer_bytes,
           (ull)st.snapshot_drops, (ull)st.storage_writes);
    close_fakes();
    printf("ok transfers\n");
}

// A transfer that stops by name.
static void case_xstop(const char *what) {
    open_fakes();
    Rec *r = rec_begin(0);
    if (!strcmp(what, "cpu_texture")) {
        nvn_tex_rec t = texture(HOST + 0x8000, 9, NVN_FORMAT_RGBA8, NVN_TEXTURE_TARGET_2D, 64, 32, 1);
        PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = MIRROR + 0x1000, .tex = t, .r = region(0, 0, 0, 8, 8, 1)});
    } else if (!strcmp(what, "overlap")) {
        PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = HOST + 0x100, .dst = HOST + 0x140, .size = 0x100});
    } else if (!strcmp(what, "unaligned_fill")) {
        PUT(r, OP_CLEAR_BUF, (p_clear_buf){.addr = VRAM + 0x41, .size = 0x20, .value = 1});
    } else if (!strcmp(what, "bad_stride")) {
        PUT(r, OP_COPY_STRIDE, (p_copy_stride){6});
        PUT(r, OP_COPY_B2T, (p_copy_b2t){.addr = HOST + 0x2000, .tex = tex_a(), .r = region(0, 0, 0, 8, 8, 1)});
    } else if (!strcmp(what, "masked_clear")) {
        PUT(r, OP_CLEAR_TEX, (p_clear_tex){.tex = tex_a(), .r = region(0, 0, 0, 0, 0, 0), .mask = 0x3});
    } else if (!strcmp(what, "outside")) {
        PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = MIRROR + 0x1000, .tex = tex_a(), .r = region(60, 0, 0, 8, 8, 1)});
    } else if (!strcmp(what, "mirrored")) {
        PUT(r, OP_COPY_T2T, (p_copy_t2t){.src = tex_a(), .dst = tex_a(), .sr = region(0, 0, 0, 8, 8, 1),
                                         .dr = region(8, 8, 0, 8, 8, 1), .flags = NVN_COPY_FLAGS_MIRROR_X_BIT});
    } else if (!strcmp(what, "decoded_b2t") || !strcmp(what, "decoded_t2b")) {
        // The stand-in device makes every colour image RGBA8, so an ASTC texture's image holds decoded texels.
        nvn_tex_rec t = texture(TEX_B, 2, NVN_FORMAT_RGBA_ASTC_4x4, NVN_TEXTURE_TARGET_2D, 64, 32, 1);
        PUT(r, !strcmp(what, "decoded_b2t") ? OP_COPY_B2T : OP_COPY_T2B,
            (p_copy_b2t){.addr = MIRROR + 0x1000, .tex = t, .r = region(0, 0, 0, 8, 8, 1)});
    } else if (!strcmp(what, "decoded_pair")) {
        // The counterpart that goes through: texels to texels, between two images of decoded ASTC.
        nvn_tex_rec a = texture(TEX_B, 2, NVN_FORMAT_RGBA_ASTC_4x4, NVN_TEXTURE_TARGET_2D, 64, 32, 1);
        nvn_tex_rec b = texture(TEX_ARRAY, 3, NVN_FORMAT_RGBA_ASTC_4x4, NVN_TEXTURE_TARGET_2D, 64, 32, 1);
        PUT(r, OP_COPY_T2T, (p_copy_t2t){.src = a, .dst = b, .sr = region(0, 0, 0, 8, 8, 1), .dr = region(8, 8, 0, 8, 8, 1)});
    } else {
        fprintf(stderr, "unknown stop %s\n", what);
        exit(2);
    }
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

// Indirect dispatches: the group counts are the GPU's (a mirror, VRAM, or host memory the frame writes), never read by the CPU.
static void case_indirect(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    PUT(r, OP_BIND_PROGRAM, program(1));
    PUT(r, OP_DISPATCH_IND, (p_dispatch_ind){.ind = VRAM + 0x500});
    PUT(r, OP_DISPATCH_IND, (p_dispatch_ind){.ind = MIRROR + 0x600});
    PUT(r, OP_DISPATCH, (p_dispatch){2, 1, 1, 0});
    PUT(r, OP_DISPATCH_IND, (p_dispatch_ind){.ind = VRAM + 0x500});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok indirect\n");
}

static p_bind_state state_of(int kind, const void *bytes, size_t n) {
    p_bind_state st;
    memset(&st, 0, sizeof st);
    st.kind = kind;
    CHECK(n <= sizeof st.bytes);
    memcpy(st.bytes, bytes, n);
    return st;
}

static void draw3(Rec *r, int prim) {
    PUT(r, OP_DRAW, (p_draw){.prim = prim, .count = 3});
}

// The states that needed a core feature the device did not enable draw now, each as its own packets: line and point
// fill, the depth bias clamp, depth clamp, the depth bounds test, a wide line, several viewports and scissors, a blend
// state and write mask for each of two targets, and dual-source factors on one.
static void case_features(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    poly_st poly = {.cull = 0, .front = 0, .mode = NVN_POLYGON_MODE_LINE, .offset_en = 2};
    PUT(r, OP_BIND_STATE, state_of(4, &poly, sizeof poly));
    PUT(r, OP_POLY_OFFSET, (p_poly_offset){.factor = 1.5f, .units = 2.0f, .clamp = 0.25f});
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);   // draw 1: line fill, bias 2, clamp 0.25, slope 1.5
    poly.mode = NVN_POLYGON_MODE_POINT, poly.offset_en = 1;
    PUT(r, OP_BIND_STATE, state_of(4, &poly, sizeof poly));
    draw3(r, NVN_DRAW_PRIMITIVE_POINTS);      // draw 2: point fill
    poly.mode = NVN_POLYGON_MODE_FILL, poly.offset_en = 0;
    PUT(r, OP_BIND_STATE, state_of(4, &poly, sizeof poly));
    PUT(r, OP_DEPTH_CLAMP, (p_flag){1});
    PUT(r, OP_DEPTH_BOUNDS, (p_depth_bounds){.enable = 1, .min = 0.25f, .max = 0.75f});
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);   // draw 3: depth clamp, depth bounds 0.25 to 0.75
    PUT(r, OP_DEPTH_CLAMP, (p_flag){0});
    PUT(r, OP_DEPTH_BOUNDS, (p_depth_bounds){.enable = 0});
    float width = 3.5f;
    PUT(r, OP_BIND_STATE, state_of(8, &width, sizeof width));
    draw3(r, NVN_DRAW_PRIMITIVE_LINES);       // draw 4: a line 3.5 wide
    PUT(r, OP_VIEWPORTS, (p_viewports){.first = 0, .n = 3, .v = {{0, 0, 64, 32}, {0, 0, 32, 16}, {32, 16, 32, 16}}});
    PUT(r, OP_SCISSOR, (p_scissor){.first = 0, .n = 2, .r = {{8, 4, 16, 8}, {0, 0, 10, 10}}});
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);   // draw 5: 3 viewports, the scissors (2 set) the whole area past them
    // two targets: target 0 blends and writes red and green, target 1 does not blend and writes alpha only
    p_set_rt rt = one_target(0);
    rt.n = 2;
    rt.color[1] = rt.color[0];
    put(r, OP_SET_RT, &rt, sizeof rt);
    blend_st b = {.target = 0, .src_rgb = NVN_BLEND_FUNC_SRC_ALPHA, .dst_rgb = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA,
                  .src_a = NVN_BLEND_FUNC_ONE, .dst_a = NVN_BLEND_FUNC_ZERO, .eq_rgb = NVN_BLEND_EQUATION_ADD,
                  .eq_a = NVN_BLEND_EQUATION_ADD};
    PUT(r, OP_BIND_STATE, state_of(0, &b, sizeof b));
    color_st color = {.blend_en = 1};
    PUT(r, OP_BIND_STATE, state_of(1, &color, sizeof color));
    chmask_st mask = {.mask = 0x3 | 0x8 << 4};
    PUT(r, OP_BIND_STATE, state_of(2, &mask, sizeof mask));
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);   // draw 6
    // dual-source factors on a single target
    PUT(r, OP_SET_RT, one_target(0));
    b.src_rgb = NVN_BLEND_FUNC_SRC1_COLOR, b.dst_rgb = NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR;
    PUT(r, OP_BIND_STATE, state_of(0, &b, sizeof b));
    mask.mask = 0xF;
    PUT(r, OP_BIND_STATE, state_of(2, &mask, sizeof mask));
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);   // draw 7
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok features\n");
}

// Dual-source factors with two colour targets stop by name: Vulkan guarantees one.
static void case_dual_two_targets(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    p_set_rt rt = one_target(0);
    rt.n = 2;
    rt.color[1] = rt.color[0];
    put(r, OP_SET_RT, &rt, sizeof rt);
    blend_st b = {.target = 0, .src_rgb = NVN_BLEND_FUNC_SRC1_ALPHA, .dst_rgb = NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA,
                  .src_a = NVN_BLEND_FUNC_ONE, .dst_a = NVN_BLEND_FUNC_ZERO, .eq_rgb = NVN_BLEND_EQUATION_ADD,
                  .eq_a = NVN_BLEND_EQUATION_ADD};
    PUT(r, OP_BIND_STATE, state_of(0, &b, sizeof b));
    color_st color = {.blend_en = 1};
    PUT(r, OP_BIND_STATE, state_of(1, &color, sizeof color));
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

// An indexed indirect draw whose record the CPU can read puts the record's base instance into c[0]; one it cannot read
// (GPU memory) keeps 0 there.
static void case_draw_indirect(void) {
    open_fakes();
    const uint32_t record[5] = {6, 2, 0, 3, 7};
    memcpy(g_host + 0x240, record, sizeof record);
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_DRAW_IDX_IND, (p_draw_idx_ind){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .itype = NVN_INDEX_TYPE_UNSIGNED_SHORT,
                                             .ibuf = MIRROR + 0x5000, .ind = HOST + 0x240});
    PUT(r, OP_DRAW_IDX_IND, (p_draw_idx_ind){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .itype = NVN_INDEX_TYPE_UNSIGNED_SHORT,
                                             .ibuf = MIRROR + 0x5000, .ind = VRAM + 0x200});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok draw_indirect\n");
}

static void case_mesh(void) {
    open_fakes();
    g_mesh_pushes = 1;
    const uint32_t records[2][2] = {{3, 41}, {0, 7}};
    memcpy(g_vram + 0x100, records, sizeof records);
    Rec *r = rec_begin(0);
    p_set_rt rt = one_target(0);
    put(r, OP_SET_RT, &rt, sizeof rt);
    PUT(r, OP_VIEWPORTS, (p_viewports){.first = 0, .n = 1, .v = {{0, 0, 64, 32}}});
    PUT(r, OP_BIND_PROGRAM, program_mesh_only());
    PUT(r, OP_DRAW_MESH, (p_draw_mesh){.first = 21, .count = 1});
    PUT(r, OP_BIND_PROGRAM, program_mesh());
    PUT(r, OP_DRAW_MESH, (p_draw_mesh){.first = 37, .count = 2});
    PUT(r, OP_DRAW_MESH, (p_draw_mesh){.first = 9, .count = 0});
    PUT(r, OP_DRAW_MESH_IND, (p_draw_mesh_ind){.ind = VRAM + 0x100});
    PUT(r, OP_DRAW_MESH_IND, (p_draw_mesh_ind){.ind = VRAM + 0x108});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok mesh\n");
}

static void case_mesh_restart(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    p_bind_program mesh = program_mesh_only();
    mesh.stages |= 1u << NVN_STAGE_VS;
    PUT(r, OP_BIND_PROGRAM, mesh);
    PUT(r, OP_DRAW_MESH, (p_draw_mesh){.first = 0, .count = 1});
    PUT(r, OP_BIND_PROGRAM, program(0));
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);
    PUT(r, OP_PRIM_RESTART, (p_prim_restart){.enable = 1, .index = 0xFFFF});
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);
    PUT(r, OP_BIND_PROGRAM, mesh);
    PUT(r, OP_DRAW_MESH, (p_draw_mesh){.first = 1, .count = 1});
    PUT(r, OP_BIND_PROGRAM, program(0));
    PUT(r, OP_PRIM_RESTART, (p_prim_restart){.enable = 0, .index = 0xFFFF});
    draw3(r, NVN_DRAW_PRIMITIVE_TRIANGLES);
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    close_fakes();
    printf("ok mesh_restart\n");
}

static void case_oplist(void) {
    for (uint32_t op = 0; op < OP_COUNT_; op++) {
        const char *t = vkr_op_ticket(op);
        printf("op %u %s %s\n", op, nvn_op_name(op), t ? t : "replayed");
    }
    printf("past %s\n", vkr_op_ticket(OP_COUNT_));
    printf("ok oplist\n");
}

// A recording holding op (its record zero past the header, as long as the largest record the test knows).
static void case_stop(uint32_t op) {
    open_fakes();
    Rec *r = rec_begin(0);
    static const uint8_t zero[512];
    put(r, op, zero, sizeof zero);
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

static void case_broken(void) {
    open_fakes();
    Rec *r = &g_rec[0];
    r->n = 0;
    PUT(r, OP_DRAW, (p_draw){0});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

// ---------- reports, events, waits and conditions: written by the completion thread in op-stream order
static uint64_t rd64(uint32_t off) { uint64_t v; memcpy(&v, g_host + off, 8); return v; }
static void wr64(uint32_t off, uint64_t v) { memcpy(g_host + off, &v, 8); }
#define EV ((volatile uint32_t *)(void *)(g_host + 0x900))
static void draw_k(Rec *r, int k) { PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = k, .count = 3}); }

static void case_reports(void) {
    open_fakes();
    wr64(0xA00, 7), wr64(0xA10, 0);   // a timestamp report goes to 0xA00: not done, it counts as 0, not 7
    wr64(0xB00, 5), wr64(0xB10, 5);   // done reports: memory decides
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_RESET_COUNTER, (p_reset_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED});
    draw_k(r, 1);                                                          // query 0 begins with the pass
    draw_k(r, 2);
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x800});   // query 0: 10
    draw_k(r, 3);                                                          // query 1
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP, HOST + 0x810});
    PUT(r, OP_SIGNAL_EVENT, (p_signal_event){EV, NVN_EVENT_SIGNAL_MODE_WRITE, 5, 0, 0});
    PUT(r, OP_SIGNAL_EVENT, (p_signal_event){EV, NVN_EVENT_SIGNAL_MODE_ADD, 3, 0, 0});
    PUT(r, OP_WAIT_EVENT, (p_wait_event){EV, NVN_EVENT_WAIT_MODE_EQUAL, 8, 0x7100});
    PUT(r, OP_WAIT_EVENT, (p_wait_event){EV, NVN_EVENT_WAIT_MODE_EQUAL, 100, 0x7200});
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, VRAM + 0x300});    // dropped
    PUT(r, OP_RENDER_COND, (p_render_cond){0, 0, HOST + 0x800});           // samples passed not done: renders
    draw_k(r, 4);
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP, HOST + 0xA00});
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, HOST + 0xA00});           // 0 != 0: skipped
    draw_k(r, 5);
    PUT(r, OP_RENDER_COND, (p_render_cond){0, 0, HOST + 0xB00});           // 5 == 5: renders
    draw_k(r, 6);
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, HOST + 0xB00});           // skipped
    draw_k(r, 7);
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    uint64_t h = rec_end(r);
    uint64_t v = vkr_submit(&h, 1);
    CHECK(gpu_sync_wait(v, 10000000000ull));
    r = rec_begin(1);
    draw_k(r, 8);                                                          // query 0 of the second pool
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x820});
    PUT(r, OP_RESET_COUNTER, (p_reset_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED});
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x830});
    h = rec_end(r);
    CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    printf("event %u\n", *EV);
    printf("samples %llu %llu %llu\n", (ull)rd64(0x800), (ull)rd64(0x820), (ull)rd64(0x830));
    printf("timestamp counter %llu, ticks in order %d, nonzero %d\n", (ull)rd64(0x810),
           rd64(0x818) >= rd64(0x808) && rd64(0x828) >= rd64(0x818), rd64(0x808) != 0);
    VkrReportStats rs;
    vkr_report_stats(&rs);
    printf("reports %llu dropped %llu resets %llu events %llu queries %llu waits %llu unsatisfied %llu conditions %llu "
           "false %llu\n", (ull)rs.reports, (ull)rs.dropped, (ull)rs.resets, (ull)rs.events, (ull)rs.queries, (ull)rs.waits,
           (ull)rs.waits_unsatisfied, (ull)rs.conditions, (ull)rs.conditions_false);
    VkrStats st;
    vkr_stats(&st);
    printf("draws %llu\n", (ull)st.draws);
    close_fakes();
    printf("ok reports\n");
}

// 600 passes while counting: the frame takes a second query pool after 512 queries, and the sum covers both.
static void case_many_queries(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_RESET_COUNTER, (p_reset_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED});
    for (int i = 0; i < 600; i++) {
        draw_k(r, i);
        PUT(r, OP_BARRIER, (p_barrier){0, 0});   // ends the pass: the next draw begins another query
    }
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x800});
    uint64_t h = rec_end(r);
    CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    printf("samples %llu\n", (ull)rd64(0x800));
    close_fakes();
    printf("ok many_queries\n");
}

static void case_many_pending_reports(void) {
    open_fakes();
    wr64(0x800, 7);
    wr64(0x810, 0);
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x800});
    for (int i = 0; i < 1024; i++)
        PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP, HOST + 0x1000 + (uint64_t)i * 16});
    PUT(r, OP_RENDER_COND, (p_render_cond){0, 0, HOST + 0x800});
    draw_k(r, 1);
    uint64_t h = rec_end(r);
    CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    VkrStats st;
    vkr_stats(&st);
    printf("first draws %llu\n", (ull)st.draws);

    wr64(0x800, 7);
    r = rec_begin(1);
    for (int i = 0; i < 1024; i++)
        PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP, HOST + 0x1000 + (uint64_t)i * 16});
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x800});
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP, HOST + 0x800});
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, HOST + 0x800});
    draw_k(r, 2);
    h = rec_end(r);
    CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    vkr_stats(&st);
    printf("second draws %llu\n", (ull)st.draws);

    wr64(0x800, 7);
    r = rec_begin(0);
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, HOST + 0x800});
    draw_k(r, 3);
    h = rec_end(r);
    CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    vkr_stats(&st);
    printf("third draws %llu\n", (ull)st.draws);
    close_fakes();
    printf("ok many_pending_reports\n");
}

static void case_pending_across_frames(void) {
    open_fakes();
    g_hold_timeline = 1;
    wr64(0x800, 7);
    wr64(0x810, 0);
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED, HOST + 0x800});
    uint64_t h = rec_end(r);
    uint64_t first = vkr_submit(&h, 1);
    r = rec_begin(1);
    for (int i = 0; i < 1024; i++)
        PUT(r, OP_REPORT_COUNTER, (p_report_counter){NVN_COUNTER_TYPE_TIMESTAMP, HOST + 0x1000 + (uint64_t)i * 16});
    PUT(r, OP_RENDER_COND, (p_render_cond){0, 0, HOST + 0x800});
    draw_k(r, 1);
    h = rec_end(r);
    uint64_t last = vkr_submit(&h, 1);
    CHECK(last > first && gpu_sync_done() < first);
    VkrStats st;
    vkr_stats(&st);
    printf("draws %llu\n", (ull)st.draws);
    counter_raise(last);
    CHECK(gpu_sync_wait(last, 10000000000ull));
    close_fakes();
    printf("ok pending_across_frames\n");
}

// The game's variable rate shading is replayed at full rate: the draw after a bound shading rate image and a nonzero
// rate mode is made, and both are counted.
static void case_full_rate(void) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    p_shading_rate_image img = {0};
    img.tex.present = 1;
    PUT(r, OP_SHADING_RATE_IMAGE, img);
    PUT(r, OP_SHADING_RATE_MODE, (p_flag){.value = 1});
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3, .instances = 1});
    PUT(r, OP_SHADING_RATE_MODE, (p_flag){.value = 0});
    PUT(r, OP_SHADING_RATE_IMAGE, (p_shading_rate_image){0});
    uint64_t h = rec_end(r);
    CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    VkrStats st;
    vkr_stats(&st);
    printf("stats draws %llu full_rate %llu lost %llu\n", (ull)st.draws, (ull)st.full_rate, (ull)st.lost);
    close_fakes();
    printf("ok full_rate\n");
}

// The geometry processing mode (NVN2) is accepted for the modes 0 to 2 and logged once per mode, however many times it is
// set and across submissions; the draws around it are made as if it had not been set. Any other mode stops by name.
static void case_geometry(const char *mode_text) {
    int32_t mode = (int32_t)strtol(mode_text, NULL, 0);
    open_fakes();
    for (int pass = 0; pass < 2; pass++) {
        Rec *r = rec_begin(pass);
        graphics_state(r);
        PUT(r, OP_GEOMETRY_MODE, (p_flag){.value = mode});
        PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3, .instances = 1});
        PUT(r, OP_GEOMETRY_MODE, (p_flag){.value = mode});
        uint64_t h = rec_end(r);
        CHECK(gpu_sync_wait(vkr_submit(&h, 1), 10000000000ull));
    }
    VkrStats st;
    vkr_stats(&st);
    printf("stats draws %llu lost %llu\n", (ull)st.draws, (ull)st.lost);
    close_fakes();
    printf("ok geometry\n");
}

static void case_geometry_stop(const char *mode_text) {
    open_fakes();
    Rec *r = rec_begin(0);
    graphics_state(r);
    PUT(r, OP_GEOMETRY_MODE, (p_flag){.value = (int32_t)strtol(mode_text, NULL, 0)});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

// A draw whose vertex attribute is another scalar type than the module's input stops by name, and the stop path writes
// the ledger with the rows of that draw (the tracker sees a record after its replay, so without this the answers the
// draw stopped on would never reach the next prepare).
static void case_attr_stop(const char *folder) {
    open_fakes();
    ledger_set_folder(folder);
    ledger_flush_on_stop();
    Rec *r = rec_begin(0);
    graphics_state(r);
    va_st a = {.fmt = NVN_FORMAT_R32UI, .offset = 0, .stream = 0};   // the module's input at 0 is a float
    p_bind_vattr va = {.first = 0, .n = 1};
    memcpy(&va.st[0], &a, sizeof a);
    put(r, OP_BIND_VATTR, &va, sizeof va);
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3, .instances = 1});
    uint64_t h = rec_end(r);
    vkr_submit(&h, 1);
    printf("submit returned\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_replay_test <case> [args] [--mutant NAME]\n");
        return 2;
    }
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--mutant")) continue;
        const char *m = argv[i + 1];
        if (!strcmp(m, "noflip")) vkr_no_flip = 1;
        else if (!strcmp(m, "nosnapshot")) vkr_no_snapshot = 1;
        else if (!strcmp(m, "nodrop")) vkr_no_drop = 1;
        else if (!strcmp(m, "nodeclare")) vkr_no_declare = 1;
        else if (!strcmp(m, "nofeatures")) vkr_no_features = 1;
        else if (!strcmp(m, "nofirsttask")) vkr_no_first_task = 1;
        else if (!strcmp(m, "nopending")) vkr_no_pending = 1;
        else if (!strcmp(m, "reverse")) vkr_reverse_results = 1;
        else {
            fprintf(stderr, "unknown mutant %s\n", m);
            return 2;
        }
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "graphics")) case_graphics();
    else if (!strcmp(cs, "skip")) case_skip(0);
    else if (!strcmp(cs, "full_rate")) case_full_rate();
    else if (!strcmp(cs, "geometry") && argc >= 3) case_geometry(argv[2]);
    else if (!strcmp(cs, "geometry_stop") && argc >= 3) case_geometry_stop(argv[2]);
    else if (!strcmp(cs, "attr_stop") && argc >= 3) case_attr_stop(argv[2]);
    else if (!strcmp(cs, "skip_render_off")) case_skip(1);
    else if (!strcmp(cs, "depth")) case_depth();
    else if (!strcmp(cs, "compute")) case_compute();
    else if (!strcmp(cs, "raw_first")) case_raw_first();
    else if (!strcmp(cs, "storage_image")) case_storage_image();
    else if (!strcmp(cs, "storage_stop") && argc >= 3) case_storage_stop(argv[2]);
    else if (!strcmp(cs, "conflict_read_first")) case_conflict(0);
    else if (!strcmp(cs, "conflict_write_first")) case_conflict(1);
    else if (!strcmp(cs, "transfers")) case_transfers();
    else if (!strcmp(cs, "xstop") && argc >= 3) case_xstop(argv[2]);
    else if (!strcmp(cs, "indirect")) case_indirect();
    else if (!strcmp(cs, "oplist")) case_oplist();
    else if (!strcmp(cs, "features")) case_features();
    else if (!strcmp(cs, "dual_two_targets")) case_dual_two_targets();
    else if (!strcmp(cs, "draw_indirect")) case_draw_indirect();
    else if (!strcmp(cs, "mesh")) case_mesh();
    else if (!strcmp(cs, "mesh_restart")) case_mesh_restart();
    else if (!strcmp(cs, "stop") && argc >= 3) case_stop((uint32_t)strtoul(argv[2], NULL, 10));
    else if (!strcmp(cs, "broken")) case_broken();
    else if (!strcmp(cs, "reports")) case_reports();
    else if (!strcmp(cs, "many_queries")) case_many_queries();
    else if (!strcmp(cs, "many_pending_reports")) case_many_pending_reports();
    else if (!strcmp(cs, "pending_across_frames")) case_pending_across_frames();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
