// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Uploads of CPU-written textures (oot2/vk_texture.h). The shape of it follows DKB Recomp's runtime/vk_texture.c
// (GPL-3.0-or-later; ported there from MacNativeGames d2r_gfx and LAS Recomp, MIT): a texture's CPU-written bytes go
// through a staging buffer into its image, uploaded again when the CPU rewrote them. Changed here, the staging is the
// bounded ring of oot2/texmem.h, the image is bound into the arena (vk_image.c), the block-linear storage is untiled by a
// compute pass (the Switch 2's GOB order; DKB untiles on the CPU) into a scratch buffer from which the copy into the
// image reads, and the work is recorded in the submission's prologue so that it runs between the earlier submissions
// and this one.
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/texmem.h"
#include "oot2/vk_format.h"
#include "oot2/vk_texture.h"
#include "plat.h"

typedef unsigned long long ull;

#define SCRATCH_MIN (64ull << 20)
#define RETIRE_MAX 16

// The compute pass's push constants (untile.comp).
typedef struct Push {
    uint64_t src, dst;
    uint32_t row_bytes, rows, slices, gobs_x, tile_h, tile_d, pitch, order;
} Push;
_Static_assert(sizeof(Push) == 48, "the push constants of untile.comp are 48 bytes");

// The ASTC decode's push constants (astc_decode.comp).
typedef struct AstcPush {
    uint64_t src, dst;
    uint32_t bw, bh, blocks_x, blocks_y, srgb;
} AstcPush;
_Static_assert(sizeof(AstcPush) == 40, "the push constants of astc_decode.comp are 40 bytes");

typedef struct Retire {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint64_t value;
} Retire;

static struct {
    int open;
    VkPipelineLayout layout;
    VkPipeline pipe;
    VkPipelineLayout alayout;
    VkPipeline apipe;
    VkBuffer scratch;
    VkDeviceMemory scratch_mem;
    uint64_t scratch_addr, scratch_size, scratch_off, scratch_value;
    Retire retire[RETIRE_MAX];
    int nretire;
    VktStats st;
} g;
static PlatMutex g_mu = PLAT_MUTEX_INIT;

static uint64_t up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// ---------- the ring's and the arena's memory

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((bits >> i & 1) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}

// A buffer over its own memory with a device address: 1, or 0 when the device has no room for it.
static int buffer_new(uint64_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid,
                      VkBuffer *buf, VkDeviceMemory *mem, uint64_t *addr) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, buf), "vkCreateBuffer (texture upload)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, *buf, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, want, avoid);
    if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, want, 0);
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no memory type with flags %#x for a texture upload buffer", (unsigned)want);
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi, .allocationSize = mr.size,
                                  .memoryTypeIndex = type};
    VkResult r = vkAllocateMemory(gpu_vk.device, &alloc, NULL, mem);
    if (r == VK_ERROR_DEVICE_LOST) gpu_vk_failed(r, "vkAllocateMemory (texture upload)");
    if (r != VK_SUCCESS) {
        rt_log("gfx: vulkan: vkAllocateMemory of %llu KiB for a texture upload buffer: %s", (ull)(size >> 10), gpu_vk_result_name(r));
        vkDestroyBuffer(gpu_vk.device, *buf, NULL);
        return 0;
    }
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, *buf, *mem, 0), "vkBindBufferMemory (texture upload)");
    VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = *buf};
    *addr = vkGetBufferDeviceAddress(gpu_vk.device, &di);
    return 1;
}

static int op_chunk_new(uint64_t size, TexChunk *out) {
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer buf;
    VkDeviceMemory mem;
    uint64_t addr;
    // Write-combined system memory (not cached, not the small host-visible part of VRAM): the CPU only writes it.
    if (!buffer_new(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv,
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &buf, &mem, &addr))
        return 0;
    void *map;
    GPU_VK_OK(vkMapMemory(gpu_vk.device, mem, 0, VK_WHOLE_SIZE, 0, &map), "vkMapMemory (the staging ring)");
    *out = (TexChunk){.map = map, .address = addr, .size = size, .handle = {(uint64_t)(uintptr_t)buf, (uint64_t)(uintptr_t)mem}};
    return 1;
}

static void op_chunk_free(const TexChunk *c) {
    vkDestroyBuffer(gpu_vk.device, (VkBuffer)(uintptr_t)c->handle[0], NULL);
    vkFreeMemory(gpu_vk.device, (VkDeviceMemory)(uintptr_t)c->handle[1], NULL);
}

// The device-local type an image of these bits takes, not the small host-visible part of VRAM when there is a choice.
static uint32_t op_type_pick(uint32_t bits) {
    uint32_t t = memory_type(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (t == UINT32_MAX) t = memory_type(bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    return t;
}

static int op_block_new(uint32_t type, uint64_t size, uint64_t *handle) {
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size, .memoryTypeIndex = type};
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(gpu_vk.device, &alloc, NULL, &mem);
    if (r == VK_ERROR_DEVICE_LOST) gpu_vk_failed(r, "vkAllocateMemory (a texture arena block)");
    if (r != VK_SUCCESS) {
        rt_log("gfx: vulkan: vkAllocateMemory of %llu MiB for a texture arena block: %s", (ull)(size >> 20), gpu_vk_result_name(r));
        return 0;
    }
    *handle = (uint64_t)(uintptr_t)mem;
    return 1;
}

static void op_block_free(uint64_t handle) { vkFreeMemory(gpu_vk.device, (VkDeviceMemory)(uintptr_t)handle, NULL); }
static int op_budget(uint64_t *used, uint64_t *budget) { return gpu_vk_backing_ops.vram(used, budget); }
static uint64_t op_done(void) { return gpu_sync_done(); }
static void op_wait(uint64_t value) { gpu_sync_wait(value, UINT64_MAX); }

static const TexMemOps k_ops = {op_chunk_new, op_chunk_free, op_type_pick, op_block_new, op_block_free, op_budget, op_done, op_wait};

void vkt_open(void) {
    plat_mutex_lock(&g_mu);
    if (!g.open) {
        memset(&g, 0, sizeof g);
        g.open = 1;
        texmem_open(&k_ops);
    }
    plat_mutex_unlock(&g_mu);
}

int vkt_is_open(void) { return g.open; }

static void retire_reap(void) {
    uint64_t done = gpu_sync_done();
    for (int i = 0; i < g.nretire;) {
        if (g.retire[i].value <= done) {
            vkDestroyBuffer(gpu_vk.device, g.retire[i].buf, NULL);
            vkFreeMemory(gpu_vk.device, g.retire[i].mem, NULL);
            g.retire[i] = g.retire[--g.nretire];
        } else {
            i++;
        }
    }
}

void vkt_close(void) {
    plat_mutex_lock(&g_mu);
    if (g.open) {
        if (g.pipe) vkDestroyPipeline(gpu_vk.device, g.pipe, NULL);
        if (g.layout) vkDestroyPipelineLayout(gpu_vk.device, g.layout, NULL);
        if (g.apipe) vkDestroyPipeline(gpu_vk.device, g.apipe, NULL);
        if (g.alayout) vkDestroyPipelineLayout(gpu_vk.device, g.alayout, NULL);
        if (g.scratch) {
            vkDestroyBuffer(gpu_vk.device, g.scratch, NULL);
            vkFreeMemory(gpu_vk.device, g.scratch_mem, NULL);
        }
        for (int i = 0; i < g.nretire; i++) {
            vkDestroyBuffer(gpu_vk.device, g.retire[i].buf, NULL);
            vkFreeMemory(gpu_vk.device, g.retire[i].mem, NULL);
        }
        texmem_close();
        memset(&g, 0, sizeof g);
    }
    plat_mutex_unlock(&g_mu);
}

// ---------- the compute pass

static void pipeline_ensure(void) {
    if (g.pipe) return;
    const GpuShader *s = gpu_shader("untile");
    if (!s) rt_fatal("gfx: vulkan: the runtime was built without the untile shader");
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->size, .pCode = s->code};
    VkShaderModule m;
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &m), "vkCreateShaderModule (untile)");
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pr};
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &li, NULL, &g.layout), "vkCreatePipelineLayout (untile)");
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = g.layout};
    GPU_VK_OK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &g.pipe), "vkCreateComputePipelines (untile)");
    vkDestroyShaderModule(gpu_vk.device, m, NULL);
}

// The ASTC decode pass, made at the first region that needs it.
static void astc_pipeline_ensure(void) {
    if (g.apipe) return;
    const GpuShader *s = gpu_shader("astc_decode");
    if (!s) rt_fatal("gfx: vulkan: the runtime was built without the astc_decode shader");
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->size, .pCode = s->code};
    VkShaderModule m;
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &m), "vkCreateShaderModule (astc_decode)");
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(AstcPush)};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pr};
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &li, NULL, &g.alayout), "vkCreatePipelineLayout (astc_decode)");
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = g.alayout};
    GPU_VK_OK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &g.apipe), "vkCreateComputePipelines (astc_decode)");
    vkDestroyShaderModule(gpu_vk.device, m, NULL);
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                    VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src_stage, .srcAccessMask = src_access,
                           .dstStageMask = dst_stage, .dstAccessMask = dst_access};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void dispatch(VkCommandBuffer cb, const Push *p) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
    vkCmdPushConstants(cb, g.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof *p, p);
    vkCmdDispatch(cb, (((p->row_bytes + 15) >> 4) + 15) / 16, (p->rows + 3) / 4, p->slices);
    g.st.dispatches++;
}

static void astc_dispatch(VkCommandBuffer cb, const AstcPush *p) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.apipe);
    vkCmdPushConstants(cb, g.alayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof *p, p);
    vkCmdDispatch(cb, (p->blocks_x + 7) / 8, (p->blocks_y + 7) / 8, 1);
    g.st.dispatches++;
}

// ---------- the scratch buffer: the untiled rows, on their way from the compute pass to the copy into the image

// Does `bytes` more scratch fit in the submission's part of it as it is: the buffer's size, from the offset this
// submission has reached? A request that does not fit makes scratch_take wrap or grow, which must not happen under copies
// that have not been recorded yet (vkt_upload records them first).
static int scratch_fits(uint64_t value, uint64_t bytes) {
    uint64_t off = g.scratch_value == value ? g.scratch_off : 0;
    return g.scratch && off + bytes <= g.scratch_size;
}

// `bytes` of scratch for the submission being recorded, aligned to 256: its buffer, the offset in it and the device address
// of that offset. Everything an earlier submission did with the scratch is behind the prologue's first barrier, so each
// submission starts again at 0; one that fills it waits for its own copies before reusing it; one that needs more than it
// holds gets a larger buffer (the old one goes when the submission is done).
static uint64_t scratch_take(VkCommandBuffer cb, uint64_t value, uint64_t bytes, VkBuffer *buf, uint64_t *addr) {
    retire_reap();
    if (g.scratch_value != value) g.scratch_value = value, g.scratch_off = 0;
    if (bytes > g.scratch_size) {
        if (g.scratch) {
            if (g.nretire == RETIRE_MAX) gpu_sync_wait(g.retire[0].value, UINT64_MAX), retire_reap();
            if (g.nretire == RETIRE_MAX) rt_fatal("gfx: vulkan: too many texture scratch buffers wait to be destroyed");
            g.retire[g.nretire++] = (Retire){g.scratch, g.scratch_mem, value};
            g.st.scratch_grown++;
        }
        uint64_t size = bytes > SCRATCH_MIN ? up(bytes, 1ull << 20) : SCRATCH_MIN;
        if (!buffer_new(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &g.scratch, &g.scratch_mem,
                        &g.scratch_addr))
            rt_fatal("gfx: vulkan: no room in VRAM for %llu MiB of texture scratch", (ull)(size >> 20));
        g.scratch_size = size, g.scratch_off = 0;
    } else if (g.scratch_off + bytes > g.scratch_size) {
        barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        g.scratch_off = 0;
        g.st.scratch_resets++;
    }
    uint64_t at = g.scratch_off;
    g.scratch_off = up(at + bytes, 256);
    *buf = g.scratch;
    *addr = g.scratch_addr + at;
    return at;
}

// ---------- uploads

typedef struct Copy {
    VkBuffer buf;
    VkBufferImageCopy c;
} Copy;

// Record the copies of regions [from, to) into the image: the compute pass wrote their rows into the scratch, which the
// copies read, so the compute writes are made visible to the transfer first.
static void record_copies(VkCommandBuffer cb, VkImage image, const Copy *copies, int from, int to) {
    for (int i = from; i < to; i++) {
        vkCmdCopyBufferToImage(cb, copies[i].buf, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copies[i].c);
        g.st.copies++;
    }
}

static uint64_t linear_stride(const nvn_texdesc *d, unsigned bpb, int bw) {
    uint64_t align = (d->flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) ? 128 : 32;
    uint32_t blocks = (d->width + (uint32_t)bw - 1) / (uint32_t)bw;
    return d->stride ? d->stride : ((uint64_t)bpb * blocks + align - 1) & ~(align - 1);
}

uint64_t vkt_linear_stride(const nvn_texdesc *d) {
    int bw, bh;
    unsigned bpb = (unsigned)nvn_fmt_bpb(d->format, &bw, &bh);
    return linear_stride(d, bpb, bw);
}

void vkt_upload(const nvn_texdesc *d, uint64_t storage, VkImage image, VkImageAspectFlags aspect, VktState *st,
                uint64_t value, int astc_decode) {
    if (!g.open) rt_fatal("gfx: vulkan: a texture upload before the texture memory is open");
    plat_mutex_lock(&g_mu);
    int first = !st->sync.defined;
    unsigned cap = texsync_max_regions(d);
    TexRegion *regs = malloc((size_t)cap * sizeof *regs);
    Copy *copies = malloc((size_t)cap * sizeof *copies);
    if (!regs || !copies) rt_fatal("gfx: vulkan: out of memory for the regions of a texture upload");
    const char *why = NULL;
    int n = texsync_regions(d, storage, &st->sync, regs, (int)cap, &why);
    if (n < 0)
        rt_fatal("gfx: vulkan: the texture at %#llx (target %d, NVN format %#x, %ux%u) in storage the CPU wrote cannot be "
                 "uploaded: %s", (ull)storage, d->target, d->format, d->width, d->height, why);
    if (n > (int)cap) rt_fatal("gfx: vulkan: a texture of %u regions listed %d", cap, n);
    if (n == 0) {
        free(regs), free(copies);
        plat_mutex_unlock(&g_mu);
        return;
    }
    pipeline_ensure();
    if (astc_decode) astc_pipeline_ensure();
    const VkfRow *frow = vkf_row(d->format);
    VkCommandBuffer cb = gpu_vk_prologue();
    if (first) {   // no layout yet: every level and layer to GENERAL before the first copy
        VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                   .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                   .newLayout = VK_IMAGE_LAYOUT_GENERAL, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
        vkCmdPipelineBarrier2(cb, &di);
    }
    int bw, bh;
    unsigned bpb = (unsigned)nvn_fmt_bpb(d->format, &bw, &bh);
    int linear = nvn_tex_is_linear(d);
    int dispatched = 0, recorded = 0;   // dispatched: the compute writes since the last barrier; recorded: copies[0, recorded) are in cb
    for (int i = 0; i < n; i++) {
        const TexRegion *r = &regs[i];
        if (linear) {
            uint64_t stride = linear_stride(d, bpb, bw);
            nvn_texlayout L;
            nvn_tex_layout(d, &L);
            if (bw > 1 || bh > 1 || stride % bpb || d->target != NVN_TEXTURE_TARGET_2D)
                rt_fatal("gfx: vulkan: the linear texture at %#llx (target %d, NVN format %#x, stride %llu) is not uploaded: "
                         "only a plain 2D texture of an uncompressed format with a row pitch of whole texels is", (ull)storage,
                         d->target, d->format, (ull)stride);
            TexSlot raw;
            texring_take(value, L.size, 256, &raw);
            if (r->zero) memset(raw.map, 0, (size_t)L.size);
            else memcpy(raw.map, (const void *)(uintptr_t)storage, (size_t)L.size);
            copies[i] = (Copy){.buf = (VkBuffer)(uintptr_t)raw.chunk->handle[0],
                               .c = {.bufferOffset = raw.offset, .bufferRowLength = (uint32_t)(stride / bpb),
                                     .imageSubresource = {aspect, 0, 0, 1},
                                     .imageExtent = {d->width, d->height ? d->height : 1, 1}}};
            g.st.raw_bytes += L.size, g.st.linear_bytes += L.size;
        } else {
            UntileGeom ug;
            const char *w2 = NULL;
            untile_geom(d, r->level, r->layer, &ug, &w2);
            TexSlot raw;
            texring_take(value, ug.span, TEXRING_ALIGN, &raw);
            if (r->zero) memset(raw.map, 0, (size_t)ug.span);
            else memcpy(raw.map, (const void *)(uintptr_t)(storage + ug.base), (size_t)ug.span);
            uint64_t pitch = up(ug.row_bytes, 16), lin = pitch * ug.rows * ug.slices;
            // ASTC to RGBA8: the rows of blocks the untile pass writes are read by the decode, which writes whole blocks of
            // texels after them in the same scratch allocation (one take, so that neither can wrap under the other).
            uint64_t blocks_x = ug.row_bytes / 16, dec_at = 0, dec = 0;
            if (astc_decode) {
                if (!frow || !(frow->flags & VKF_ASTC) || ug.slices != 1 || ug.bpb != 16)
                    rt_fatal("gfx: vulkan: the texture at %#llx (target %d, NVN format %#x) is not decoded from ASTC: only a 2D or "
                             "layered ASTC texture is", (ull)storage, d->target, d->format);
                dec_at = up(lin, 256);
                dec = blocks_x * ug.bw * 4 * ug.rows * ug.bh;
            }
            if (dispatched && !scratch_fits(value, astc_decode ? dec_at + dec : lin)) {
                // The scratch is about to be reused: the regions untiled so far are copied out of it first.
                barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                        VK_ACCESS_2_TRANSFER_READ_BIT);
                record_copies(cb, image, copies, recorded, i);
                recorded = i;
                dispatched = 0;
            }
            VkBuffer sbuf;
            uint64_t saddr, soff = scratch_take(cb, value, astc_decode ? dec_at + dec : lin, &sbuf, &saddr);
            Push p = {.src = raw.address, .dst = saddr, .row_bytes = ug.row_bytes, .rows = ug.rows, .slices = ug.slices,
                      .gobs_x = ug.gobs_x, .tile_h = ug.tile_h, .tile_d = ug.tile_d, .pitch = (uint32_t)pitch,
                      .order = UNTILE_GOB_ORDER};
            dispatch(cb, &p);
            dispatched = 1;
            if (astc_decode) {
                barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_2_SHADER_READ_BIT);
                AstcPush ap = {.src = saddr, .dst = saddr + dec_at, .bw = ug.bw, .bh = ug.bh, .blocks_x = (uint32_t)blocks_x,
                               .blocks_y = ug.rows, .srgb = (frow->flags & VKF_SRGB) != 0};
                astc_dispatch(cb, &ap);
                copies[i] = (Copy){.buf = sbuf,
                                   .c = {.bufferOffset = soff + dec_at, .bufferRowLength = (uint32_t)(blocks_x * ug.bw),
                                         .bufferImageHeight = ug.rows * ug.bh,
                                         .imageSubresource = {aspect, r->level, r->layer, 1},
                                         .imageExtent = {ug.width, ug.height, 1}}};
                g.st.astc_regions++, g.st.astc_blocks += blocks_x * ug.rows, g.st.astc_bytes += dec;
            } else {
                copies[i] = (Copy){.buf = sbuf,
                                   .c = {.bufferOffset = soff, .bufferRowLength = (uint32_t)(pitch / ug.bpb * ug.bw),
                                         .bufferImageHeight = ug.rows * ug.bh,
                                         .imageSubresource = {aspect, r->level, r->layer, 1},
                                         .imageExtent = {ug.width, ug.height, ug.depth}}};
            }
            g.st.raw_bytes += ug.span, g.st.linear_bytes += astc_decode ? dec : lin;
        }
        g.st.zero_regions += (uint64_t)r->zero;
        g.st.regions++;
    }
    if (dispatched)
        barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                VK_ACCESS_2_TRANSFER_READ_BIT);
    record_copies(cb, image, copies, recorded, n);
    g.st.uploads++;
    free(regs), free(copies);
    plat_mutex_unlock(&g_mu);
}

void vkt_untile_host(const UntileGeom *ug, int order, const uint8_t *src, uint8_t *dst, size_t pitch) {
    if (!g.open) rt_fatal("gfx: vulkan: the untile pass before the texture memory is open");
    uint64_t pitch16 = up(ug->row_bytes, 16), lin = pitch16 * ug->rows * ug->slices;
    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);   // the submission's gate first, then this file's lock, the order vkt_upload's callers hold them in
    plat_mutex_lock(&g_mu);
    pipeline_ensure();
    VkCommandBuffer cb = gpu_vk_cmd();
    TexSlot in, out;
    texring_take(value, ug->span, TEXRING_ALIGN, &in);
    memcpy(in.map, src, (size_t)ug->span);
    texring_take(value, lin, 16, &out);
    Push p = {.src = in.address, .dst = out.address, .row_bytes = ug->row_bytes, .rows = ug->rows, .slices = ug->slices,
              .gobs_x = ug->gobs_x, .tile_h = ug->tile_h, .tile_d = ug->tile_d, .pitch = (uint32_t)pitch16,
              .order = (uint32_t)order};
    dispatch(cb, &p);
    barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT);
    plat_mutex_unlock(&g_mu);
    gpu_vk_end(NULL, 0);
    gpu_sync_wait(value, UINT64_MAX);
    for (uint32_t z = 0; z < ug->slices; z++)
        for (uint32_t y = 0; y < ug->rows; y++)
            memcpy(dst + ((size_t)z * ug->rows + y) * pitch, out.map + ((uint64_t)z * ug->rows + y) * pitch16, ug->row_bytes);
}

void vkt_stats(VktStats *out) {
    plat_mutex_lock(&g_mu);
    *out = g.st;
    plat_mutex_unlock(&g_mu);
}

// For the counters' lines (they print at a stop, from the thread that stopped, which may hold this lock inside vkt_upload,
// and from the present thread, which must not wait behind a readback): 0 and *out untouched when the lock is busy.
int vkt_stats_try(VktStats *out) {
    if (!plat_mutex_trylock(&g_mu)) return 0;
    *out = g.st;
    plat_mutex_unlock(&g_mu);
    return 1;
}
