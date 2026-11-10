// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Device tests of the uploads of CPU-written textures (runtime/gpu/vk_texture.c, oot2/vk_texture.h, the arena images of
// vk_image.c; architecture D19, the texture upload). Driven by tests/test_runtime_vk_texture.py: `vk_texture_test <case>`, one
// case per process. Needs a device with every item of oot2/vk_required.h; exit 3 (no loader, no device) is a skip.
//
//   untile     the compute pass on the device equals the CPU untiler byte for byte over the table of
//              tests/runtime/untile_table.h (formats, sizes, levels, layers, block heights, both GOB orders), and the table
//              can tell the orders apart (the Switch 2's pass on storage in the X1's order differs from the texels)
//   anchor     with OOT2_T65_DIR set (a folder of real textures of a player's dump and their untiling by Lichii's shaders,
//              which never enter the repository): the compute pass and the CPU untiler equal Lichii's answer for each, in
//              the Switch 2's order, and the X1's order does not; a skip line without the folder
//   upload     textures written into guest memory by the walk of the table (2D, mips, block-compressed, arrays, 3D, cube,
//              packaged block heights, linear) are made images of the arena by the first submission that binds them and
//              copy back, level by level and layer by layer, as the texels that were written
//   texels     the texel calls on textures the CPU can see write and read their guest storage in its own layout: the
//              arena's upload reads back what they wrote, level by level and layer by layer
//   copy_into  a copy into a texture the CPU writes that nothing bound yet lands on the storage's content uploaded first
//   scratch    one texture whose untiled rows are more than the scratch buffer holds (five layers of 2048 x 2048 RGBA8, 80 MiB
//              against 64 MiB): the scratch is reused inside one prologue and every layer still reads back as written
//   rewrite    a texture the CPU rewrites is uploaded again: a bind with no write uploads nothing, a write into one page
//              uploads exactly the regions over that page and the next submission reads the new bytes, the others stay
//   caps       64 MiB textures uploaded one submission after another through the 384 MiB ring: the ring never holds more
//              than its cap and the arena never more than its own
//   ring       six submissions, each binding five 64 MiB textures, with none waited for: the ring (325 MiB of chunks for one
//              submission, a cap of 384) never grows past its cap, a later submission waits for an earlier one when it must, and all
//              thirty textures arrive
//   ring_full  (fatal) one submission binding six of them: the ring is full inside one submission, with nothing earlier to wait for
//   caps_full  (fatal) the same until the arena is full, which stops by name
//   churn      ninety textures of 64 MiB registered one after another at one ID: the arena's extent of each goes back when the
//              next replaces it, so the arena never holds more than a few (it would stop at its cap after fifty-odd otherwise)
//   astc       ASTC textures the CPU wrote (six block sizes, LDR and sRGB, mips, layers, a size that is not a whole number of
//              blocks) make no image when registered, are decoded on the device at the first submission that binds them into
//              RGBA8 images of the arena that read back as the CPU reference's texels, and a block the CPU rewrites is decoded
//              again, only in the regions over its page
//   rawview    the images of BC textures (BC1, BC4, BC5, BC7) can be viewed as the 64 or 128-bit integer format of their
//              block, with storage usage on the view, with the validation layer clean
//   views      the render target and storage views of volumes and layered textures, built as the replay builds them
//              (oot2/vk_view_plan.h) with the validation layer on: a 2D view of one slice of a 3D RGBA8 texture (the first
//              and the last slice) as colour attachment and, where the device has image2DViewOf3D, as storage; a 2D array
//              view of slices; the last layer of a layered texture and a 2D one; a layer past the image is refused by the
//              plan, so no call carries it; the validation layer reports nothing
//   depth_stop (fatal) a depth texture the CPU wrote stops by name at the first submission that binds it
// Every device case ends with the validation layer's counts (the driver sets synchronization validation): 0 and 0.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/bindless.h"
#include "oot2/cpu.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
#include "oot2/texmem.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_format.h"
#include "oot2/vk_image.h"
#include "oot2/vk_texture.h"
#include "oot2/vk_view_plan.h"
#include "plat.h"
#include "astc_blocks.h"
#include "untile_table.h"
#include "vk_draw.h"

#include "astc_ref.c"   // the CPU reference, in this translation unit (the build links one source)

typedef unsigned long long ull;
#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define GiB (1ull << 30)
#define PG 0x1000ull
#define EXIT_NO_DEVICE 3

static Cpu g_cpu;
static uint64_t g_heap, g_heap_size;

static void usable_or_exit(void) {
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        exit(EXIT_NO_DEVICE);
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) {
        printf("no Vulkan 1.3 instance\n");
        exit(EXIT_NO_DEVICE);
    }
    volkLoadInstanceOnly(inst);
    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(inst, &n, NULL) != VK_SUCCESS || !n) {
        printf("no Vulkan device\n");
        exit(EXIT_NO_DEVICE);
    }
    vkDestroyInstance(inst, NULL);
}

// Guest memory (the runtime's own, with the real write watch), then the device.
static void setup(uint64_t heap, uint64_t region) {
    usable_or_exit();
    RtMemConfig cfg = {.pool_size = heap + 512 * MiB > 2048 * MiB ? heap + 512 * MiB : 2048 * MiB, .heap_region_size = region, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = heap;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1], g_heap_size = heap;
    pagegen_reset();
    gpu_vk_open();
    CHECK(vkt_is_open() && texmem_is_open());
}

static void finish(const char *name) {
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation %llu errors, %llu warnings\n", (ull)errors, (ull)warnings);
    CHECK(errors == 0 && warnings == 0);
    printf("ok %s\n", name);
}

// ---------- the untile pass against the CPU untiler

static void case_untile(void) {
    setup(64 * MiB, 1024 * MiB);
    Table t = {0};
    build_table(&t);
    uint64_t regions = 0, differ = 0, bytes = 0;
    for (int i = 0; i < t.n; i += 4) {
        nvn_texdesc d = desc_of(&t.s[i]);
        nvn_texlayout L;
        nvn_tex_layout(&d, &L);
        if (L.size > (3ull << 20)) continue;
        uint8_t *storage = malloc((size_t)L.size);
        CHECK(storage);
        for (unsigned layer = 0; layer < L.layers; layer++)
            for (unsigned level = 0; level < (d.levels ? d.levels : 1); level++) {
                UntileGeom g;
                const char *why;
                CHECK(untile_geom(&d, level, layer, &g, &why));
                size_t pitch = g.row_bytes + 16, n = (size_t)g.rows * g.slices;
                uint8_t *cpu = malloc(n * pitch), *gpu = malloc(n * pitch), *x1 = malloc(n * pitch);
                CHECK(cpu && gpu && x1);
                for (int order = 0; order < 2; order++) {
                    memset(storage, 0xCD, (size_t)L.size);
                    walk(&g, level, layer, order == UNTILE_ORDER_S2 ? k_s2 : k_x1, storage + g.base);
                    memset(cpu, 0xAB, n * pitch), memset(gpu, 0xAB, n * pitch);
                    untile_region(&g, order, storage + g.base, cpu, pitch);
                    vkt_untile_host(&g, order, storage + g.base, gpu, pitch);
                    CHECK(!memcmp(cpu, gpu, n * pitch));
                    for (uint32_t z = 0; z < g.slices; z++)
                        for (uint32_t y = 0; y < g.rows; y++)
                            for (uint32_t x = 0; x < g.row_bytes; x++)
                                CHECK(gpu[((size_t)z * g.rows + y) * pitch + x] == value(level, layer, x, y, z));
                    if (order == UNTILE_ORDER_X1) {   // the Switch 2's pass on the X1's storage is not the texels
                        memset(x1, 0xAB, n * pitch);
                        vkt_untile_host(&g, UNTILE_ORDER_S2, storage + g.base, x1, pitch);
                        differ += memcmp(x1, gpu, n * pitch) != 0;
                    }
                }
                free(cpu), free(gpu), free(x1);
                regions++, bytes += (uint64_t)g.span;
            }
        free(storage);
    }
    VktStats vs;
    vkt_stats(&vs);
    printf("%llu regions, %llu bytes, equal to the CPU untiler in both orders; the orders differ on %llu of them; %llu dispatches\n",
           (ull)regions, (ull)bytes, (ull)differ, (ull)vs.dispatches);
    CHECK(regions > 200 && differ > regions / 4);
    free(t.s);
    finish("untile");
}

// ---------- images of the arena

static tex_t tex_of(uint64_t storage, const Spec *s) {
    tex_t t;
    memset(&t, 0, sizeof t);
    t.magic = NVN_TEX_MAGIC;
    t.d = desc_of(s);
    t.d.swizzle_r = NVN_TEXTURE_SWIZZLE_R, t.d.swizzle_g = NVN_TEXTURE_SWIZZLE_G, t.d.swizzle_b = NVN_TEXTURE_SWIZZLE_B;
    t.d.swizzle_a = NVN_TEXTURE_SWIZZLE_A;
    t.mem = (uint8_t *)(uintptr_t)storage;
    static uint64_t serial;
    t.serial = ++serial;
    nvn_tex_layout(&t.d, &t.L);
    return t;
}

static const dpool_ref k_pool = {0x900000, 33025, 0}, k_spool = {0xA00000, 4096, 0};

// A submission that binds texture ID `id` (the first one made the image and uploads; later ones only what changed).
static uint64_t bind_once(uint32_t id) {
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    dpool_ref pool = k_pool, spool = k_spool;
    vkb_acquire(cb, v, &pool, &spool);
    CHECK(vkb_image_slot(cb, id, BV_NONE) == id);
    gpu_vk_end(NULL, 0);
    CHECK(gpu_sync_wait(v, 30ull * 1000 * 1000 * 1000));
    return v;
}

// Level `level`, layer `layer` of the arena image of t, read back as tight rows of blocks (3D: every slice).
static uint8_t *readback(const tex_t *t, unsigned level, unsigned layer, const UntileGeom *g) {
    VkbImage *img = vkb_image_demand(t);
    CHECK(img);
    size_t bytes = (size_t)g->row_bytes * g->rows * g->slices;
    DrawBuf out = draw_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL, 0);
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    VkBufferImageCopy c = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1},
                           .imageExtent = {g->width, g->height, g->depth}};
    vkCmdCopyImageToBuffer(cb, vkb_image_handle(img), VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    gpu_vk_end(NULL, 0);
    CHECK(gpu_sync_wait(v, 30ull * 1000 * 1000 * 1000));
    uint8_t *copy = malloc(bytes);
    CHECK(copy);
    memcpy(copy, out.map, bytes);
    vkDestroyBuffer(gpu_vk.device, out.buf, NULL);
    vkFreeMemory(gpu_vk.device, out.mem, NULL);
    return copy;
}

// Write texture t's storage in guest memory by the walk of the table (or by rows when linear), the CPU's way.
static void write_storage(const tex_t *t, uint64_t storage) {
    if (nvn_tex_is_linear(&t->d)) {
        int bw, bh;
        unsigned bpb = (unsigned)nvn_fmt_bpb(t->d.format, &bw, &bh);
        uint64_t stride = ((uint64_t)bpb * t->d.width + 31) & ~31ull;
        for (uint32_t y = 0; y < t->d.height; y++)
            for (uint32_t x = 0; x < bpb * t->d.width; x++) ((uint8_t *)(uintptr_t)storage)[y * stride + x] = value(0, 0, x, y, 0);
        return;
    }
    for (unsigned layer = 0; layer < t->L.layers; layer++)
        for (unsigned level = 0; level < (t->d.levels ? t->d.levels : 1); level++) {
            UntileGeom g;
            const char *why;
            CHECK(untile_geom(&t->d, level, layer, &g, &why));
            walk(&g, level, layer, k_s2, (uint8_t *)(uintptr_t)(storage + g.base));
        }
}

static void check_image(const tex_t *t, uint64_t storage, const char *what) {
    (void)storage;
    if (nvn_tex_is_linear(&t->d)) {
        int bw, bh;
        unsigned bpb = (unsigned)nvn_fmt_bpb(t->d.format, &bw, &bh);
        UntileGeom g = {.width = t->d.width, .height = t->d.height, .depth = 1, .row_bytes = bpb * t->d.width,
                        .rows = t->d.height, .slices = 1};
        uint8_t *got = readback(t, 0, 0, &g);
        for (uint32_t y = 0; y < g.rows; y++)
            for (uint32_t x = 0; x < g.row_bytes; x++) CHECK(got[(size_t)y * g.row_bytes + x] == value(0, 0, x, y, 0));
        free(got);
        return;
    }
    for (unsigned layer = 0; layer < t->L.layers; layer++)
        for (unsigned level = 0; level < (t->d.levels ? t->d.levels : 1); level++) {
            UntileGeom g;
            const char *why;
            CHECK(untile_geom(&t->d, level, layer, &g, &why));
            uint8_t *got = readback(t, level, layer, &g);
            for (uint32_t z = 0; z < g.slices; z++)
                for (uint32_t y = 0; y < g.rows; y++)
                    for (uint32_t x = 0; x < g.row_bytes; x++)
                        if (got[((size_t)z * g.rows + y) * g.row_bytes + x] != value(level, layer, x, y, z)) {
                            fprintf(stderr, "%s: level %u layer %u slice %u row %u byte %u: %02x, written %02x\n", what, level, layer, z, y, x,
                                    got[((size_t)z * g.rows + y) * g.row_bytes + x], value(level, layer, x, y, z));
                            exit(1);
                        }
            free(got);
        }
}

static void case_upload(void) {
    setup(64 * MiB, 1024 * MiB);
    static const Spec specs[] = {
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 100, 37, 1, 7, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 0, 130, 66, 1, 8, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_BPTC_UNORM, 0, 260, 131, 1, 9, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGTC2_UNORM, 0, 256, 256, 1, 9, 5},
        {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 0, 40, 24, 3, 6, 0},
        {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGTC1_UNORM, 0, 64, 64, 4, 7, 4},
        {NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGBA8, 0, 17, 9, 5, 5, 0},
        {NVN_TEXTURE_TARGET_CUBEMAP, NVN_FORMAT_RGBA8, 0, 32, 32, 6, 6, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_R16, 0, 63, 63, 1, 6, 3},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RG8, NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT, 200, 100, 1, 8, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA16F, 0, 300, 200, 1, 1, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, NVN_TEXTURE_FLAGS_LINEAR, 100, 50, 1, 1, 0},
    };
    uint64_t at = g_heap + 4 * MiB;
    for (unsigned k = 0; k < sizeof specs / sizeof specs[0]; k++) {
        uint64_t storage = (at + 0xFFFF) & ~0xFFFFull;
        tex_t t = tex_of(storage, &specs[k]);
        at = storage + t.L.size + 0x10000;
        write_storage(&t, storage);
        dpool_ref pool = k_pool;
        bindless_texture_register(&pool, (int)k + 1, &t, NULL);
        bind_once(k + 1);
        char what[64];
        snprintf(what, sizeof what, "spec %u (NVN format %#x)", k, specs[k].format);
        check_image(&t, storage, what);
        printf("spec %u: target %d format %#x %ux%u levels %u layers %u: every level and layer reads back as written\n", k,
               specs[k].target, specs[k].format, specs[k].w, specs[k].h, specs[k].levels, (unsigned)t.L.layers);
    }
    VkbImageStats is;
    vkb_image_stats(&is);
    VktStats vs;
    vkt_stats(&vs);
    TexMemStats ms;
    texmem_stats(&ms);
    printf("arena images %llu, uploads %llu, regions %llu, dispatches %llu, copies %llu, ring peak %llu bytes, arena used %llu bytes\n",
           (ull)is.arena, (ull)vs.uploads, (ull)vs.regions, (ull)vs.dispatches, (ull)vs.copies, (ull)ms.ring_peak, (ull)ms.arena_used);
    CHECK(is.arena == sizeof specs / sizeof specs[0] && vs.uploads == is.arena);
    finish("upload");
}

// The texel calls (nvnTextureWriteTexels, ReadTexels) on textures the CPU can see go through their guest storage: every
// level and layer written as tight rows by vkb_write_texels is what the arena's upload reads back at the first bind, and
// vkb_read_texels returns the rows that were written.
static void case_texels(void) {
    setup(64 * MiB, 1024 * MiB);
    static const Spec specs[] = {
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 100, 37, 1, 7, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_DXT1, 0, 130, 66, 1, 8, 0},
        {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 0, 40, 24, 3, 6, 0},
        {NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGBA8, 0, 17, 9, 5, 5, 0},
        {NVN_TEXTURE_TARGET_1D, NVN_FORMAT_RGBA8, 0, 64, 1, 1, 1, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, NVN_TEXTURE_FLAGS_LINEAR, 100, 50, 1, 1, 0},
    };
    uint64_t at = g_heap + 4 * MiB;
    for (unsigned k = 0; k < sizeof specs / sizeof specs[0]; k++) {
        uint64_t storage = (at + 0xFFFF) & ~0xFFFFull;
        tex_t t = tex_of(storage, &specs[k]);
        at = storage + t.L.size + 0x10000;
        int linear = nvn_tex_is_linear(&t.d);
        if (linear) t.d.stride = (uint32_t)((4ull * t.d.width + 31) & ~31ull);   // what write_storage and check_image read
        unsigned layers = linear ? 1 : t.L.layers, levels = t.d.levels ? t.d.levels : 1;
        int layered = specs[k].target == NVN_TEXTURE_TARGET_2D_ARRAY;
        for (unsigned layer = 0; layer < layers; layer++)
            for (unsigned level = 0; level < levels; level++) {
                UntileGeom g;
                const char *why;
                if (linear) g = (UntileGeom){.width = t.d.width, .height = t.d.height, .depth = 1, .row_bytes = 4 * t.d.width,
                                             .rows = t.d.height, .slices = 1};
                else CHECK(untile_geom(&t.d, level, layer, &g, &why));
                size_t bytes = (size_t)g.row_bytes * g.rows * g.slices;
                uint8_t *rows = malloc(bytes), *back = malloc(bytes);
                CHECK(rows && back);
                for (uint32_t z = 0; z < g.slices; z++)
                    for (uint32_t y = 0; y < g.rows; y++)
                        for (uint32_t x = 0; x < g.row_bytes; x++)
                            rows[((size_t)z * g.rows + y) * g.row_bytes + x] = value(level, layer, x, y, z);
                view_t v;
                memset(&v, 0, sizeof v);
                v.set_levels = 1, v.base_level = (uint8_t)level, v.nlevels = 1;
                NVNcopyRegion r = {.zoffset = layered ? (int)layer : 0, .width = (int)g.width, .height = (int)g.height,
                                   .depth = specs[k].target == NVN_TEXTURE_TARGET_3D ? (int)g.depth : 1};
                uint64_t img = (uint64_t)g.row_bytes * g.rows;
                vkb_write_texels(&t, &v, &r, rows, g.row_bytes, img);
                memset(back, 0xEE, bytes);
                vkb_read_texels(&t, &v, &r, back, g.row_bytes, img);
                CHECK(!memcmp(rows, back, bytes));
                free(rows), free(back);
            }
        dpool_ref pool = k_pool;
        bindless_texture_register(&pool, (int)k + 1, &t, NULL);
        bind_once(k + 1);
        char what[64];
        snprintf(what, sizeof what, "texels spec %u (NVN format %#x)", k, specs[k].format);
        check_image(&t, storage, what);
        printf("texels spec %u: target %d format %#x: written through the texel calls, read back by them and by the arena "
               "image\n", k, specs[k].target, specs[k].format);
    }
    VkbImageStats is;
    vkb_image_stats(&is);
    printf("texel writes %llu, reads %llu, arena images %llu\n", (ull)is.texel_writes, (ull)is.texel_reads, (ull)is.arena);
    CHECK(is.arena == sizeof specs / sizeof specs[0]);
    finish("texels");
}

// A copy into a texture the CPU writes that no submission bound yet (the title's first submission does this): the copy's
// own submission demands the arena image and makes it current (vkb_image_demand_current), so its prologue uploads the
// storage first and the copy lands on top: inside the copied rectangle the copy's bytes, outside it the storage's.
static void case_copy_into(void) {
    setup(64 * MiB, 1024 * MiB);
    Spec s = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 64, 48, 1, 1, 0};
    uint64_t storage = (g_heap + 4 * MiB + 0xFFFF) & ~0xFFFFull;
    tex_t t = tex_of(storage, &s);
    write_storage(&t, storage);
    const uint32_t x0 = 8, y0 = 4, w = 16, h = 8;
    DrawBuf src = draw_buffer(4ull * w * h, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, NULL, 0);
    memset(src.map, 0xC3, 4ull * w * h);
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkbImage *img = vkb_image_demand_current(&t, v);
    CHECK(img);
    VkCommandBuffer cb = gpu_vk_cmd();
    VkMemoryBarrier2 before = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                               .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &before};
    vkCmdPipelineBarrier2(cb, &di);
    VkBufferImageCopy c = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageOffset = {(int32_t)x0, (int32_t)y0, 0},
                           .imageExtent = {w, h, 1}};
    vkCmdCopyBufferToImage(cb, src.buf, vkb_image_handle(img), VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    gpu_vk_end(NULL, 0);
    CHECK(gpu_sync_wait(v, 30ull * 1000 * 1000 * 1000));
    UntileGeom g;
    const char *why;
    CHECK(untile_geom(&t.d, 0, 0, &g, &why));
    uint8_t *got = readback(&t, 0, 0, &g);
    uint64_t inside = 0, outside = 0;
    for (uint32_t y = 0; y < g.rows; y++)
        for (uint32_t x = 0; x < g.row_bytes; x++) {
            int in = y >= y0 && y < y0 + h && x / 4 >= x0 && x / 4 < x0 + w;
            uint8_t want = in ? 0xC3 : value(0, 0, x, y, 0);
            if (got[(size_t)y * g.row_bytes + x] != want) {
                fprintf(stderr, "copy_into: row %u byte %u: %02x, want %02x\n", y, x, got[(size_t)y * g.row_bytes + x], want);
                exit(1);
            }
            inside += in, outside += !in;
        }
    printf("copy into an arena image no submission bound: %llu bytes copied, %llu bytes from the storage\n", (ull)inside,
           (ull)outside);
    free(got);
    vkDestroyBuffer(gpu_vk.device, src.buf, NULL);
    vkFreeMemory(gpu_vk.device, src.mem, NULL);
    finish("copy_into");
}

// The regions of t whose bytes lie over any page of [lo, hi): what a write there makes stale.
static unsigned regions_over(const tex_t *t, uint64_t storage, uint64_t lo, uint64_t hi) {
    unsigned n = 0;
    for (unsigned layer = 0; layer < t->L.layers; layer++)
        for (unsigned level = 0; level < (t->d.levels ? t->d.levels : 1); level++) {
            UntileGeom g;
            const char *why;
            CHECK(untile_geom(&t->d, level, layer, &g, &why));
            uint64_t a = (storage + g.base) & ~(PG - 1), b = (storage + g.base + g.span + PG - 1) & ~(PG - 1);
            n += a < hi && lo < b;
        }
    return n;
}

static void case_scratch(void) {
    setup(512 * MiB, 1024 * MiB);
    Spec s = {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 0, 2048, 2048, 5, 1, 0};
    uint64_t storage = (g_heap + 4 * MiB + 0xFFFF) & ~0xFFFFull;
    tex_t t = tex_of(storage, &s);
    write_storage(&t, storage);
    dpool_ref pool = k_pool;
    bindless_texture_register(&pool, 1, &t, NULL);
    bind_once(1);
    VktStats vs;
    vkt_stats(&vs);
    printf("%llu regions in one prologue, scratch reused %llu times, grown %llu times\n", (ull)vs.regions, (ull)vs.scratch_resets, (ull)vs.scratch_grown);
    CHECK(vs.regions == 5 && vs.scratch_resets >= 1 && vs.scratch_grown == 0);
    check_image(&t, storage, "scratch");
    finish("scratch");
}

static void case_rewrite(void) {
    setup(64 * MiB, 1024 * MiB);
    Spec s = {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 0, 256, 256, 2, 9, 0};
    uint64_t storage = (g_heap + 4 * MiB + 0xFFFF) & ~0xFFFFull;
    tex_t t = tex_of(storage, &s);
    write_storage(&t, storage);
    dpool_ref pool = k_pool;
    bindless_texture_register(&pool, 5, &t, NULL);
    VktStats a, b;
    vkt_stats(&a);
    bind_once(5);
    vkt_stats(&b);
    unsigned all = (unsigned)(t.L.layers * (t.d.levels ? t.d.levels : 1));
    CHECK(b.uploads == a.uploads + 1 && b.regions == a.regions + all);
    check_image(&t, storage, "first upload");
    // a bind with nothing written uploads nothing
    vkt_stats(&a);
    bind_once(5);
    vkt_stats(&b);
    CHECK(b.regions == a.regions && b.uploads == a.uploads);
    printf("a bind with no write uploaded nothing\n");
    // the CPU rewrites 16 bytes inside level 3 of layer 1 (the pages the write touches are the only ones to move)
    UntileGeom g;
    const char *why;
    CHECK(untile_geom(&t.d, 3, 1, &g, &why));
    uint64_t at = storage + g.base + 64;
    uint8_t before[16], after[16];
    memcpy(before, (void *)(uintptr_t)at, 16);
    for (int i = 0; i < 16; i++) after[i] = (uint8_t)(before[i] ^ 0x5A), ((uint8_t *)(uintptr_t)at)[i] = after[i];
    unsigned stale = regions_over(&t, storage, at & ~(PG - 1), (at & ~(PG - 1)) + PG);
    vkt_stats(&a);
    bind_once(5);
    vkt_stats(&b);
    printf("one write: %llu regions uploaded again, %u lie over the page\n", (ull)(b.regions - a.regions), stale);
    CHECK(b.regions - a.regions == stale && stale >= 1 && stale < all);
    // the image shows the new bytes in that region (compare the whole region's bytes with the guest memory untiled)
    uint8_t *got = readback(&t, 3, 1, &g);
    uint8_t *cpu = malloc((size_t)g.row_bytes * g.rows * g.slices);
    untile_region(&g, UNTILE_ORDER_S2, (const uint8_t *)(uintptr_t)(storage + g.base), cpu, g.row_bytes);
    CHECK(!memcmp(got, cpu, (size_t)g.row_bytes * g.rows * g.slices));
    free(got), free(cpu);
    // and another level of that layer is as it was written
    UntileGeom g0;
    CHECK(untile_geom(&t.d, 0, 0, &g0, &why));
    uint8_t *l0 = readback(&t, 0, 0, &g0);
    for (uint32_t y = 0; y < g0.rows; y++)
        for (uint32_t x = 0; x < g0.row_bytes; x++) CHECK(l0[(size_t)y * g0.row_bytes + x] == value(0, 0, x, y, 0));
    free(l0);
    // a second rewrite, at the very start of the texture
    ((uint8_t *)(uintptr_t)storage)[3] ^= 0xFF;
    vkt_stats(&a);
    bind_once(5);
    vkt_stats(&b);
    stale = regions_over(&t, storage, storage & ~(PG - 1), (storage & ~(PG - 1)) + PG);
    CHECK(b.regions - a.regions == stale && stale >= 1);
    UntileGeom g00;
    CHECK(untile_geom(&t.d, 0, 0, &g00, &why));
    uint8_t *got0 = readback(&t, 0, 0, &g00);
    uint8_t *cpu0 = malloc((size_t)g00.row_bytes * g00.rows);
    untile_region(&g00, UNTILE_ORDER_S2, (const uint8_t *)(uintptr_t)storage, cpu0, g00.row_bytes);
    CHECK(!memcmp(got0, cpu0, (size_t)g00.row_bytes * g00.rows));
    free(got0), free(cpu0);
    printf("a second write at the start of the texture: %llu regions again, the image follows\n", (ull)(b.regions - a.regions));
    finish("rewrite");
}

// ---------- the independent anchor: Lichii's answers on real textures
typedef struct Row {
    char name[256], kind[64];
    int target, format, pkg, flags, level, layer;
    unsigned w, h, depth, levels, rows, row_bytes;
} Row;

static uint8_t *slurp(const char *dir, const char *name, const char *ext, long *size) {
    char path[1200];
    snprintf(path, sizeof path, "%s/%s%s", dir, name, ext);
    FILE *f = fopen(path, "rb");
    CHECK(f);
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)*size + 1);
    CHECK(p && fread(p, 1, (size_t)*size, f) == (size_t)*size);
    fclose(f);
    return p;
}

static nvn_texdesc row_desc(const Row *r) {
    nvn_texdesc d;
    memset(&d, 0, sizeof d);
    d.target = r->target, d.format = r->format, d.flags = r->flags, d.samples = 1, d.width = r->w, d.height = r->h, d.depth = r->depth;
    d.levels = r->levels, d.pkg_block_h = r->pkg;
    return d;
}

static void case_anchor(void) {
    const char *dir = getenv("OOT2_T65_DIR");
    if (!dir || !*dir) {
        printf("skip anchor: OOT2_T65_DIR names no folder of real textures\n");
        return;
    }
    char path[1200];
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    FILE *m = fopen(path, "r");
    if (!m) {
        printf("skip anchor: no %s\n", path);
        return;
    }
    setup(1200 * MiB, 2048 * MiB);
    static Row rows[1024];
    int n = 0;
    while (n < 1024 && fscanf(m, "%255s %63s %d %d %u %u %u %u %d %d %d %d %u %u", rows[n].name, rows[n].kind, &rows[n].target, &rows[n].format, &rows[n].w,
                              &rows[n].h, &rows[n].depth, &rows[n].levels, &rows[n].pkg, &rows[n].flags, &rows[n].level, &rows[n].layer, &rows[n].rows,
                              &rows[n].row_bytes) == 14)
        n++;
    fclose(m);
    CHECK(n > 0);
    uint64_t compared = 0, bytes = 0, x1_differs = 0, uploaded = 0, at = g_heap + 4 * MiB;
    for (int i = 0; i < n;) {
        // the rows of one texture
        int j = i;
        while (j < n && !strcmp(rows[j].name, rows[i].name)) j++;
        nvn_texdesc d = row_desc(&rows[i]);
        long size;
        uint8_t *storage = slurp(dir, rows[i].name, ".bin", &size);
        uint8_t *want[64];
        CHECK(j - i <= 64);
        for (int k = i; k < j; k++) {
            UntileGeom g;
            const char *why;
            CHECK(untile_geom(&d, (unsigned)rows[k].level, (unsigned)rows[k].layer, &g, &why));
            CHECK(g.rows == rows[k].rows && g.row_bytes == rows[k].row_bytes);
            CHECK(g.base + g.span <= (uint64_t)size);
            size_t nb = (size_t)g.rows * g.row_bytes;
            char ext[96];
            snprintf(ext, sizeof ext, ".L%d.%d.lichii", rows[k].level, rows[k].layer);
            long wn;
            want[k - i] = slurp(dir, rows[k].name, ext, &wn);
            CHECK((size_t)wn == nb);
            uint8_t *cpu = malloc(nb), *gpu = malloc(nb), *wrong = malloc(nb);
            CHECK(cpu && gpu && wrong);
            untile_region(&g, UNTILE_ORDER_S2, storage + g.base, cpu, g.row_bytes);
            vkt_untile_host(&g, UNTILE_ORDER_S2, storage + g.base, gpu, g.row_bytes);
            vkt_untile_host(&g, UNTILE_ORDER_X1, storage + g.base, wrong, g.row_bytes);
            CHECK(!memcmp(want[k - i], cpu, nb));
            CHECK(!memcmp(want[k - i], gpu, nb));
            x1_differs += memcmp(want[k - i], wrong, nb) != 0;
            compared++, bytes += nb;
            free(cpu), free(gpu), free(wrong);
        }
        // and through the whole path the game's textures take: the storage written into guest memory, the texture bound by a
        // submission, the arena image copied back (block-compressed and ASTC images included: the bytes are blocks)
        const VkfRow *fr = vkf_row(d.format);
        if (!(fr->flags & VKF_ASTC)) {
            uint64_t base = (at + 0xFFFF) & ~0xFFFFull;
            at = base + (uint64_t)size + 0x10000;
            if (at < g_heap + g_heap_size) {
                memcpy((void *)(uintptr_t)base, storage, (size_t)size);
                Spec s = {rows[i].target, rows[i].format, 0, rows[i].w, rows[i].h, rows[i].depth, rows[i].levels, rows[i].pkg};
                tex_t t = tex_of(base, &s);
                dpool_ref pool = k_pool;
                bindless_texture_register(&pool, (int)(uploaded % 1000) + 1, &t, NULL);
                bind_once((uint32_t)(uploaded % 1000) + 1);
                for (int k = i; k < j; k++) {
                    UntileGeom g;
                    const char *why;
                    CHECK(untile_geom(&d, (unsigned)rows[k].level, (unsigned)rows[k].layer, &g, &why));
                    uint8_t *got = readback(&t, (unsigned)rows[k].level, (unsigned)rows[k].layer, &g);
                    if (memcmp(got, want[k - i], (size_t)g.rows * g.row_bytes)) {
                        size_t nb = (size_t)g.rows * g.row_bytes, first = 0, bad = 0;
                        while (first < nb && got[first] == want[k - i][first]) first++;
                        for (size_t b = 0; b < nb; b++) bad += got[b] != want[k - i][b];
                        fprintf(stderr, "%s (%s, %ux%u, %u levels, target %d) level %d layer %d: the image differs from Lichii's rows in %zu of %zu bytes, "
                                "the first at %zu\n", rows[i].name, rows[i].kind, rows[i].w, rows[i].h, rows[i].levels, rows[i].target, rows[k].level,
                                rows[k].layer, bad, nb, first);
                    }
                    CHECK(!memcmp(got, want[k - i], (size_t)g.rows * g.row_bytes));
                    free(got);
                }
                uploaded++;
            }
        }
        for (int k = i; k < j; k++) free(want[k - i]);
        free(storage);
        i = j;
    }
    printf("%llu real regions (%llu bytes): the compute pass and the CPU untiler equal Lichii's answer; the X1's order differs on %llu; %llu textures "
           "went through guest memory, a submission and the arena image and read back as Lichii's rows\n",
           (ull)compared, (ull)bytes, (ull)x1_differs, (ull)uploaded);
    // the orders only differ where a region has more than one piece across or more than two rows, and flat data is the same in both
    CHECK(compared > 0 && x1_differs * 4 >= compared * 3 && uploaded > 0);
    finish("anchor");
}

// ---------- the caps on a device: 64 MiB textures through the ring until the arena is full
static void caps(int to_the_end) {
    setup(4800 * MiB, 8 * GiB);
    // Storage nobody wrote reads as zeros that no guest page is read for: each texture needs only its own place.
    Spec s = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 4096, 4096, 1, 1, 0};
    uint64_t base = (g_heap + 0xFFFF) & ~0xFFFFull;
    int n = to_the_end ? 70 : 24;
    uint64_t used = 0, budget = 0;
    if (gpu_vk_backing_ops.vram(&used, &budget))
        printf("device heap: %llu MiB used of a budget of %llu MiB\n", (ull)(used >> 20), (ull)(budget >> 20));
    fflush(stdout);
    // the whole heap holds this many 64 MiB places
    (void)g_heap_size;
    for (int i = 0; i < n; i++) {
        uint64_t storage = base + (uint64_t)i * 64 * MiB;
        tex_t t = tex_of(storage, &s);
        CHECK(t.L.size == 64 * MiB);
        dpool_ref pool = k_pool;
        bindless_texture_register(&pool, i + 1, &t, NULL);
        bind_once((uint32_t)i + 1);
        TexMemStats ms;
        texmem_stats(&ms);
        CHECK(ms.ring_peak <= TEXRING_CAP && ms.arena_bytes_peak <= TEXARENA_CAP);
    }
    TexMemStats ms;
    texmem_stats(&ms);
    VktStats vs;
    vkt_stats(&vs);
    printf("%d textures of 64 MiB uploaded (%llu MiB through the ring): ring peak %llu MiB of %llu, ring waits %llu, arena %llu MiB in %llu "
           "blocks of a cap of %llu MiB\n", n, (ull)(vs.raw_bytes >> 20), (ull)(ms.ring_peak >> 20), (ull)(TEXRING_CAP >> 20),
           (ull)ms.ring_waits, (ull)(ms.arena_bytes_peak >> 20), (ull)ms.arena_blocks, (ull)(TEXARENA_CAP >> 20));
    CHECK(vs.raw_bytes >= (uint64_t)n * 64 * MiB && vs.raw_bytes > TEXRING_CAP);
    finish(to_the_end ? "caps_full" : "caps");
}

// Submissions that each bind `per` textures of 64 MiB (zeros: nothing wrote their storage), none waited for.
static void ring_run(int per, int subs) {
    setup(4800 * MiB, 8 * GiB);
    Spec s = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 4096, 4096, 1, 1, 0};
    uint64_t base = (g_heap + 0xFFFF) & ~0xFFFFull;
    uint64_t last = 0;
    for (int sub = 0; sub < subs; sub++) {
        for (int k = 0; k < per; k++) {
            int id = sub * per + k + 1;
            tex_t t = tex_of(base + (uint64_t)id * 64 * MiB, &s);
            dpool_ref pool = k_pool;
            bindless_texture_register(&pool, id, &t, NULL);
        }
        uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
        VkCommandBuffer cb = gpu_vk_cmd();
        dpool_ref pool = k_pool, spool = k_spool;
        vkb_acquire(cb, v, &pool, &spool);
        for (int k = 0; k < per; k++) CHECK(vkb_image_slot(cb, (uint32_t)(sub * per + k + 1), BV_NONE) == (uint32_t)(sub * per + k + 1));
        gpu_vk_end(NULL, 0);
        last = v;
        TexMemStats ms;
        texmem_stats(&ms);
        CHECK(ms.ring_peak <= TEXRING_CAP);
    }
    CHECK(gpu_sync_wait(last, 120ull * 1000 * 1000 * 1000));
    TexMemStats ms;
    texmem_stats(&ms);
    VktStats vs;
    vkt_stats(&vs);
    printf("%d submissions of %d textures of 64 MiB: ring peak %llu MiB of %llu, ring waits %llu, %llu MiB read from storage\n", subs, per,
           (ull)(ms.ring_peak >> 20), (ull)(TEXRING_CAP >> 20), (ull)ms.ring_waits, (ull)(vs.raw_bytes >> 20));
    // a fast device finishes a submission before the next one asks for its chunks, and the wait is not needed; the cap holds either way
    CHECK(ms.ring_peak <= TEXRING_CAP && vs.raw_bytes >= (uint64_t)per * subs * 64 * MiB && vs.raw_bytes > 4 * TEXRING_CAP);
    finish("ring");
}

static void case_churn(void) {
    setup(4800 * MiB, 8 * GiB);
    Spec s = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 4096, 4096, 1, 1, 0};
    uint64_t base = (g_heap + 0xFFFF) & ~0xFFFFull;
    int n = 90;
    for (int i = 0; i < n; i++) {
        tex_t t = tex_of(base + (uint64_t)(i % 60) * 64 * MiB, &s);
        dpool_ref pool = k_pool;
        bindless_texture_register(&pool, 1, &t, NULL);
        bind_once(1);
    }
    for (int i = 0; i < 4; i++) bind_once(1);   // let the last views and images go
    TexMemStats ms;
    texmem_stats(&ms);
    VkbImageStats is;
    vkb_image_stats(&is);
    printf("%d textures registered one after another at one ID: %llu images made, %llu destroyed, arena peak %llu MiB, %llu MiB in use now\n", n,
           (ull)is.arena, (ull)is.destroyed, (ull)(ms.arena_bytes_peak >> 20), (ull)(ms.arena_used >> 20));
    CHECK(is.arena == (uint64_t)n && is.destroyed >= (uint64_t)n - 6 && ms.arena_used <= 6 * 66 * MiB);
    finish("churn");
}

// ---------- ASTC textures decoded to RGBA8 at the first bind

typedef struct AstcRegion {
    unsigned level, layer, w, h;
    uint8_t *want;   // the reference's RGBA8 texels of the level, tight rows
} AstcRegion;

// The reference's texels of a level whose blocks are `blocks` (rows of g->row_bytes / 16 blocks), clipped to the level.
static void astc_expect(const UntileGeom *g, const uint8_t *blocks, bool srgb, uint8_t *want) {
    uint32_t bx = g->row_bytes / 16, bw = g->bw, bh = g->bh;
    uint8_t tmp[12 * 12 * 4];
    for (uint32_t j = 0; j < g->rows; j++)
        for (uint32_t i = 0; i < bx; i++) {
            CHECK(astc_ref_block(blocks + 16 * ((size_t)j * bx + i), bw, bh, srgb, 1, tmp) == ASTC_REF_OK);
            for (uint32_t y = 0; y < bh && j * bh + y < g->height; y++)
                for (uint32_t x = 0; x < bw && i * bw + x < g->width; x++)
                    memcpy(want + (((size_t)(j * bh + y)) * g->width + i * bw + x) * 4, tmp + (y * bw + x) * 4, 4);
        }
}

// Put one block into the texture's storage in guest memory, the CPU's way (a store through the guest address).
static void astc_store_block(const UntileGeom *g, uint64_t storage, uint32_t i, uint32_t j, const uint8_t blk[16]) {
    for (uint32_t k = 0; k < 16; k++)
        ((uint8_t *)(uintptr_t)(storage + g->base))[untile_offset(g, UNTILE_ORDER_S2, i * 16 + k, j, 0)] = blk[k];
}

static void case_astc(void) {
    setup(64 * MiB, 1024 * MiB);
    static const Spec specs[] = {
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_4x4, 0, 100, 37, 1, 7, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_8x8_SRGB, 0, 130, 66, 1, 5, 0},
        {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA_ASTC_6x5, 0, 40, 24, 3, 4, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_12x12, 0, 200, 150, 1, 4, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_10x5_SRGB, 0, 64, 64, 1, 3, 0},
        {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA_ASTC_5x5, 0, 77, 51, 1, 1, 0},
    };
    enum { N = sizeof specs / sizeof specs[0] };
    tex_t texs[N];
    uint64_t storages[N];
    AstcRegion regs[N][3 * 8];
    unsigned nreg[N];
    uint64_t at = g_heap + 4 * MiB;
    astc_seed(0x0125);
    for (unsigned k = 0; k < N; k++) {
        storages[k] = (at + 0xFFFF) & ~0xFFFFull;
        texs[k] = tex_of(storages[k], &specs[k]);
        at = storages[k] + texs[k].L.size + 0x10000;
        const bool srgb = (vkf_row(specs[k].format)->flags & VKF_SRGB) != 0;
        nreg[k] = 0;
        for (unsigned layer = 0; layer < texs[k].L.layers; layer++)
            for (unsigned level = 0; level < specs[k].levels; level++) {
                UntileGeom g;
                const char *why;
                CHECK(untile_geom(&texs[k].d, level, layer, &g, &why));
                uint32_t bx = g.row_bytes / 16;
                uint8_t *blocks = malloc((size_t)bx * g.rows * 16);
                AstcRegion *r = &regs[k][nreg[k]++];
                r->level = level, r->layer = layer, r->w = g.width, r->h = g.height;
                r->want = malloc((size_t)g.width * g.height * 4);
                CHECK(blocks && r->want);
                for (uint32_t j = 0; j < g.rows; j++)
                    for (uint32_t i = 0; i < bx; i++) {
                        uint8_t *b = blocks + 16 * ((size_t)j * bx + i);
                        if (astc_rnd() % 11 == 0) astc_void_block(b);
                        else CHECK(astc_draw_block(g.bw, g.bh, b, NULL, NULL));
                        astc_store_block(&g, storages[k], i, j, b);
                    }
                astc_expect(&g, blocks, srgb, r->want);
                free(blocks);
            }
        dpool_ref pool = k_pool;
        bindless_texture_register(&pool, (int)k + 1, &texs[k], NULL);
    }
    VktStats vs;
    VkbImageStats is;
    vkt_stats(&vs);
    vkb_image_stats(&is);
    printf("%u ASTC textures registered: %llu images made, %llu regions decoded\n", (unsigned)N, (ull)is.arena, (ull)vs.astc_regions);
    CHECK(is.arena == 0 && vs.astc_regions == 0);   // registering decodes nothing and makes no image
    unsigned total = 0;
    for (unsigned k = 0; k < N; k++) {
        VktStats a, b;
        vkt_stats(&a);
        bind_once(k + 1);
        vkt_stats(&b);
        CHECK(b.astc_regions - a.astc_regions == nreg[k] && b.uploads == a.uploads + 1);
        VkbImage *img = vkb_image_demand(&texs[k]);
        CHECK(img && vkb_image_format(img) == VK_FORMAT_R8G8B8A8_UNORM);
        for (unsigned q = 0; q < nreg[k]; q++) {
            const AstcRegion *r = &regs[k][q];
            UntileGeom g = {.width = r->w, .height = r->h, .depth = 1, .row_bytes = r->w * 4, .rows = r->h, .slices = 1};
            uint8_t *got = readback(&texs[k], r->level, r->layer, &g);
            for (size_t o = 0; o < (size_t)r->w * r->h * 4; o++)
                if (got[o] != r->want[o]) {
                    fprintf(stderr, "spec %u level %u layer %u: texel byte %zu is %02x, the reference's %02x\n", k, r->level, r->layer, o, got[o],
                            r->want[o]);
                    exit(1);
                }
            free(got);
        }
        total += nreg[k];
        printf("spec %u: NVN format %#x %ux%u, %u levels, %u layers: %u regions decoded at the first bind, equal to the reference\n", k,
               specs[k].format, specs[k].w, specs[k].h, specs[k].levels, (unsigned)texs[k].L.layers, nreg[k]);
    }
    vkt_stats(&vs);
    printf("decoded %llu regions, %llu blocks, %llu RGBA8 bytes\n", (ull)vs.astc_regions, (ull)vs.astc_blocks, (ull)vs.astc_bytes);
    CHECK(vs.astc_regions == total);
    // a block the CPU rewrites: only the regions over its page are decoded again, and the image shows the new block
    UntileGeom g0;
    const char *why;
    CHECK(untile_geom(&texs[0].d, 0, 0, &g0, &why));
    uint8_t nb[16];
    CHECK(astc_draw_block(g0.bw, g0.bh, nb, NULL, NULL));
    astc_store_block(&g0, storages[0], 3, 2, nb);
    uint64_t wr = storages[0] + g0.base + untile_offset(&g0, UNTILE_ORDER_S2, 3 * 16, 2, 0);
    unsigned stale = regions_over(&texs[0], storages[0], wr & ~(PG - 1), (wr & ~(PG - 1)) + PG);
    uint8_t *blocks = malloc((size_t)(g0.row_bytes / 16) * g0.rows * 16);
    CHECK(blocks);
    for (uint32_t j = 0; j < g0.rows; j++)
        for (uint32_t i = 0; i < g0.row_bytes / 16; i++)
            for (uint32_t kb = 0; kb < 16; kb++)
                blocks[((size_t)j * (g0.row_bytes / 16) + i) * 16 + kb] =
                    ((const uint8_t *)(uintptr_t)(storages[0] + g0.base))[untile_offset(&g0, UNTILE_ORDER_S2, i * 16 + kb, j, 0)];
    astc_expect(&g0, blocks, false, regs[0][0].want);
    free(blocks);
    VktStats a, b;
    vkt_stats(&a);
    bind_once(1);
    vkt_stats(&b);
    printf("one block rewritten: %llu regions decoded again, %u lie over its page\n", (ull)(b.astc_regions - a.astc_regions), stale);
    CHECK(b.astc_regions - a.astc_regions == stale && stale >= 1);
    UntileGeom gr = {.width = regs[0][0].w, .height = regs[0][0].h, .depth = 1, .row_bytes = regs[0][0].w * 4, .rows = regs[0][0].h, .slices = 1};
    uint8_t *got = readback(&texs[0], 0, 0, &gr);
    CHECK(!memcmp(got, regs[0][0].want, (size_t)gr.row_bytes * gr.rows));
    free(got);
    for (unsigned k = 0; k < N; k++)
        for (unsigned q = 0; q < nreg[k]; q++) free(regs[k][q].want);
    finish("astc");
}

// ---------- raw views of the images of BC textures

static void case_rawview(void) {
    setup(64 * MiB, 1024 * MiB);
    static const struct {
        int format;
        VkFormat view;
        const char *name;
    } k_cases[] = {
        {NVN_FORMAT_RGBA_DXT1, VK_FORMAT_R32G32_UINT, "BC1"},
        {NVN_FORMAT_RGTC1_UNORM, VK_FORMAT_R32G32_UINT, "BC4"},
        {NVN_FORMAT_RGTC2_UNORM, VK_FORMAT_R32G32B32A32_UINT, "BC5"},
        {NVN_FORMAT_BPTC_UNORM, VK_FORMAT_R32G32B32A32_UINT, "BC7"},
    };
    uint64_t at = g_heap + 4 * MiB;
    for (unsigned k = 0; k < sizeof k_cases / sizeof k_cases[0]; k++) {
        Spec s = {NVN_TEXTURE_TARGET_2D, k_cases[k].format, 0, 64, 64, 1, 4, 0};
        uint64_t storage = (at + 0xFFFF) & ~0xFFFFull;
        tex_t t = tex_of(storage, &s);
        at = storage + t.L.size + 0x10000;
        write_storage(&t, storage);
        dpool_ref pool = k_pool;
        bindless_texture_register(&pool, (int)k + 1, &t, NULL);
        bind_once(k + 1);
        VkbImage *img = vkb_image_demand(&t);
        CHECK(img);
        VkImageViewUsageCreateInfo ui = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                         .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &ui, .image = vkb_image_handle(img),
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = k_cases[k].view,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1}};
        VkImageView view;
        VkResult r = vkCreateImageView(gpu_vk.device, &vi, NULL, &view);
        CHECK(r == VK_SUCCESS);
        vkDestroyImageView(gpu_vk.device, view, NULL);
        printf("%s image: a raw view as format %d of mip level 1 with sampled and storage usage was made\n", k_cases[k].name, (int)k_cases[k].view);
    }
    finish("rawview");
}

// A view of image `img` as the plan says, for `usage`, created and destroyed: the call the validation layer judges.
static void make_view(VkbImage *img, int type, uint32_t level, uint32_t base, uint32_t n, VkImageUsageFlags usage,
                      const char *what) {
    VkbShape sh;
    vkb_image_shape(img, &sh);
    VkvImage vim = {sh.type, sh.depth, sh.layers, sh.levels, sh.flags, sh.usage, gpu_vk.image_2d_view_of_3d};
    VkvWant w = {type, level, base, n, usage};
    VkvPlan plan;
    const char *why;
    if (!vkv_plan(&vim, &w, &plan, &why)) {
        printf("%s: refused by the plan (%s)\n", what, why);
        CHECK(0);
    }
    VkImageViewUsageCreateInfo ui = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = usage};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &ui, .image = vkb_image_handle(img),
                                .viewType = plan.type, .format = sh.format,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, plan.base_layer, plan.layers}};
    VkImageView view;
    CHECK(vkCreateImageView(gpu_vk.device, &vi, NULL, &view) == VK_SUCCESS);
    vkDestroyImageView(gpu_vk.device, view, NULL);
    printf("%s: view type %d, layers %u..%u made\n", what, (int)plan.type, plan.base_layer, plan.base_layer + plan.layers);
}

static VkbImage *image_for(uint64_t *at, const Spec *s, uint32_t id) {
    uint64_t storage = (*at + 0xFFFF) & ~0xFFFFull;
    tex_t *t = malloc(sizeof *t);   // the texture outlives the case: the registry keeps looking at it
    CHECK(t);
    *t = tex_of(storage, s);
    *at = storage + t->L.size + 0x10000;
    write_storage(t, storage);
    dpool_ref pool = k_pool;
    bindless_texture_register(&pool, (int)id, t, NULL);
    bind_once(id);
    VkbImage *img = vkb_image_demand(t);
    CHECK(img);
    return img;
}

static void case_views(void) {
    setup(64 * MiB, 1024 * MiB);
    uint64_t at = g_heap + 4 * MiB;
    const VkImageUsageFlags color = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, store = VK_IMAGE_USAGE_STORAGE_BIT;
    // a volume of 8 slices
    Spec vol = {NVN_TEXTURE_TARGET_3D, NVN_FORMAT_RGBA8, 0, 32, 32, 8, 1, 0};
    VkbImage *v = image_for(&at, &vol, 1);
    VkbShape sh;
    vkb_image_shape(v, &sh);
    CHECK(sh.type == VK_IMAGE_TYPE_3D && sh.depth == 8 && sh.layers == 1);
    CHECK(sh.flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT);
    CHECK(!!(sh.flags & VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT) == !!gpu_vk.image_2d_view_of_3d);
    CHECK(sh.usage & color);
    make_view(v, VKV_PICK, 0, 0, 1, color, "volume, first slice as a render target");
    make_view(v, VKV_PICK, 0, 7, 1, color, "volume, last slice as a render target");
    make_view(v, VKV_PICK, 0, 2, 4, color, "volume, slices 2..5 as a layered render target");
    if (gpu_vk.image_2d_view_of_3d) make_view(v, VKV_PICK, 0, 7, 1, store, "volume, last slice as a storage image");
    else printf("volume, a slice as a storage image: skipped, the device lacks image2DViewOf3D\n");
    make_view(v, BV_3D, 0, 0, 1, store, "volume as a 3D storage image");
    // a layer past the volume is the plan's refusal, so no call carries it
    VkvImage vim = {sh.type, sh.depth, sh.layers, sh.levels, sh.flags, sh.usage, gpu_vk.image_2d_view_of_3d};
    VkvWant past = {VKV_PICK, 0, 8, 1, color};
    VkvPlan plan;
    const char *why;
    CHECK(!vkv_plan(&vim, &past, &plan, &why));
    printf("volume, slice 8 of 8: refused (%s)\n", why);
    // a texture of four layers, and a plain one
    Spec arr = {NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 0, 32, 32, 4, 1, 0};
    VkbImage *a = image_for(&at, &arr, 2);
    make_view(a, VKV_PICK, 0, 3, 1, color, "four layers, last layer as a render target");
    make_view(a, VKV_PICK, 0, 0, 4, color, "four layers as a layered render target");
    Spec flat = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 32, 32, 1, 1, 0};
    VkbImage *f = image_for(&at, &flat, 3);
    make_view(f, VKV_PICK, 0, 0, 1, color, "plain 2D texture as a render target");
    vkb_image_shape(f, &sh);
    VkvImage fim = {sh.type, sh.depth, sh.layers, sh.levels, sh.flags, sh.usage, gpu_vk.image_2d_view_of_3d};
    VkvWant one_past = {VKV_PICK, 0, 1, 1, color};
    CHECK(!vkv_plan(&fim, &one_past, &plan, &why));
    printf("plain 2D texture, layer 1 of 1: refused (%s)\n", why);
    finish("views");
}

static void case_stop(int depth) {
    setup(64 * MiB, 1024 * MiB);
    Spec s = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_DEPTH24_STENCIL8, 0, 64, 64, 1, 1, 0};
    (void)depth;
    uint64_t storage = (g_heap + 4 * MiB + 0xFFFF) & ~0xFFFFull;
    tex_t t = tex_of(storage, &s);
    memset((void *)(uintptr_t)storage, 0x5A, (size_t)t.L.size);
    dpool_ref pool = k_pool;
    bindless_texture_register(&pool, 1, &t, NULL);   // registering takes no memory and stops nothing
    VkbImageStats is;
    vkb_image_stats(&is);
    CHECK(is.arena == 0);
    bind_once(1);
    printf("returned\n");
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: vk_texture_test <untile|anchor|upload|rewrite|caps|caps_full>\n");
        return 2;
    }
    if (!strcmp(argv[1], "untile")) case_untile();
    else if (!strcmp(argv[1], "anchor")) case_anchor();
    else if (!strcmp(argv[1], "upload")) case_upload();
    else if (!strcmp(argv[1], "texels")) case_texels();
    else if (!strcmp(argv[1], "copy_into")) case_copy_into();
    else if (!strcmp(argv[1], "scratch")) case_scratch();
    else if (!strcmp(argv[1], "rewrite")) case_rewrite();
    else if (!strcmp(argv[1], "caps")) caps(0);
    else if (!strcmp(argv[1], "caps_full")) caps(1);
    else if (!strcmp(argv[1], "churn")) case_churn();
    else if (!strcmp(argv[1], "ring")) ring_run(5, 6);
    else if (!strcmp(argv[1], "ring_full")) ring_run(6, 1);
    else if (!strcmp(argv[1], "astc")) case_astc();
    else if (!strcmp(argv[1], "rawview")) case_rawview();
    else if (!strcmp(argv[1], "views")) case_views();
    else if (!strcmp(argv[1], "depth_stop")) case_stop(1);
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    return 0;
}
