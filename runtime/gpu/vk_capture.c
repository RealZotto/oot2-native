// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The offscreen capture of the Vulkan back end (oot2/vk_capture.h): GfxBackend.present blits the presented crop into one
// RGBA8 image; tests and the dumps read that image back.
//
// One lock spans a present's or a read's whole submission, so the capture image and the read buffer are used by one
// of them at a time; the submissions themselves are ordered by the timeline (gpu_vk_begin). A capture image of another
// size is made at the present that needs it, and the old one is destroyed by an effect of that present, once every
// submission that used it is done. A dump copies the capture into a buffer of its own in the present's submission, and
// the completion thread writes the file and frees the buffer.
//
// That lock is held across gpu_vk_begin, where its holder gives its virtual core up while it waits for the gate, so every
// take of it is rt_kernel_lock (kernel_lock.h): a guest thread on that core that waited for it with its core held would
// keep the core from the holder, which needs it back to go on.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_capture.h"
#include "oot2/vk_image.h"
#include "kernel_lock.h"
#include "plat.h"

typedef unsigned long long ull;

static PlatMutex g_mu = PLAT_MUTEX_INIT;
static struct {
    VkImage image;
    VkDeviceMemory mem;
    uint32_t w, h;
    uint64_t seq;            // the number of the present the image holds
    VkBuffer rbuf;           // the read buffer, host-visible
    VkDeviceMemory rmem;
    void *rmap;
    VkDeviceSize rsize;
    int settings;
    char dir[512];           // OOT2_VK_CAPTURE, "" for none
    uint64_t every;
} C;
static uint64_t g_presents;
int vkc_no_crop;

uint64_t vkc_presents(void) { return __atomic_load_n(&g_presents, __ATOMIC_RELAXED); }

static void settings(void) {
    if (C.settings) return;
    C.settings = 1;
    const char *dir = getenv("OOT2_VK_CAPTURE"), *every = getenv("OOT2_VK_CAPTURE_EVERY");
    if (dir && strlen(dir) >= sizeof C.dir) rt_fatal("gfx: vulkan: OOT2_VK_CAPTURE is longer than %zu bytes", sizeof C.dir - 1);
    snprintf(C.dir, sizeof C.dir, "%s", dir ? dir : "");
    C.every = 60;
    if (every && *every) {
        char *end;
        unsigned long long n = strtoull(every, &end, 10);
        if (*end || n < 1 || every[0] == '-' || every[0] == '+')
            rt_fatal("gfx: vulkan: OOT2_VK_CAPTURE_EVERY=%s is not a count from 1", every);
        C.every = n;
    }
}

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

static void host_buffer(VkDeviceSize size, VkBuffer *buf, VkDeviceMemory *mem, void **map) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, buf), "vkCreateBuffer (the capture's read buffer)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, *buf, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no host-visible coherent memory for the capture's read buffer");
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, mem), "vkAllocateMemory (the capture's read buffer)");
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, *buf, *mem, 0), "vkBindBufferMemory (the capture's read buffer)");
    GPU_VK_OK(vkMapMemory(gpu_vk.device, *mem, 0, VK_WHOLE_SIZE, 0, map), "vkMapMemory (the capture's read buffer)");
}

// ---------- the capture image
typedef struct { VkImage image; VkDeviceMemory mem; } OldImage;
static void old_image_gone(void *arg, uint64_t value) {
    (void)value;
    OldImage *o = arg;
    vkDestroyImage(gpu_vk.device, o->image, NULL);
    vkFreeMemory(gpu_vk.device, o->mem, NULL);
    free(o);
}

// An image of w x h for the open submission (g_mu and the submission held); 1 when it is new, in the undefined layout.
static int capture_image(uint32_t w, uint32_t h) {
    if (C.image && C.w == w && C.h == h) return 0;
    if (C.image) {
        OldImage *o = malloc(sizeof *o);
        if (!o) rt_fatal("gfx: vulkan: out of memory for the old capture image");
        *o = (OldImage){C.image, C.mem};
        gpu_vk_effect(old_image_gone, o);
    }
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    GPU_VK_OK(vkCreateImage(gpu_vk.device, &ci, NULL, &C.image), "vkCreateImage (the capture)");
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(gpu_vk.device, C.image, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, 0);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &C.mem), "vkAllocateMemory (the capture)");
    GPU_VK_OK(vkBindImageMemory(gpu_vk.device, C.image, C.mem, 0), "vkBindImageMemory (the capture)");
    C.w = w, C.h = h;
    return 1;
}

static void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 dst_access, int fresh_image) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = src_access,
                           .dstStageMask = dst, .dstAccessMask = dst_access};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = src,
                                .dstStageMask = dst, .dstAccessMask = dst_access,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = C.image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = fresh_image ? 1u : 0u, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cmd, &di);
}

// Copy the capture image into buf (the open submission's command buffer; the blit or present before it is covered by
// the barrier), then make the bytes visible to the host.
static void copy_out(VkCommandBuffer cmd, VkBuffer buf) {
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT, 0);
    VkBufferImageCopy r = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {C.w, C.h, 1}};
    vkCmdCopyImageToBuffer(cmd, C.image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &r);
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT,
            VK_ACCESS_2_HOST_READ_BIT, 0);
}

// ---------- files
static int write_ppm(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fprintf(f, "P6\n%u %u\n255\n", w, h) > 0;
    uint8_t *row = malloc((size_t)w * 3);
    for (uint32_t y = 0; ok && row && y < h; y++) {
        for (uint32_t x = 0; x < w; x++) memcpy(row + 3 * (size_t)x, rgba + 4 * ((size_t)y * w + x), 3);
        ok = fwrite(row, 3, w, f) == w;
    }
    free(row);
    return fclose(f) == 0 && ok && row;
}

typedef struct { VkBuffer buf; VkDeviceMemory mem; void *map; uint32_t w, h; uint64_t seq; } Dump;
static void dump_written(void *arg, uint64_t value) {
    (void)value;
    Dump *d = arg;
    char path[600];
    snprintf(path, sizeof path, "%s/present-%06llu.ppm", C.dir, (ull)d->seq);
    if (!write_ppm(path, d->map, d->w, d->h)) {
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("gfx: vulkan: the capture could not be written to %s (logged once)", path);
    }
    vkDestroyBuffer(gpu_vk.device, d->buf, NULL);
    vkFreeMemory(gpu_vk.device, d->mem, NULL);
    free(d);
}

// ---------- present
static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        return 1;
    default: return 0;
    }
}

void vkc_present(const tex_t *t, const int crop[4]) {
    VkbImage *img = vkb_image_get(t);
    if (!img)
        rt_fatal("gfx: vulkan: present of the texture at %p (NVN format %#x, %ux%u): no image stands on its storage "
                 "(storage the CPU writes is the texture arena's)", (const void *)t->mem, t->d.format, t->d.width, t->d.height);
    VkbShape sh;
    vkb_image_shape(img, &sh);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(gpu_vk.phys, sh.format, &fp);
    if (sh.all_aspects != VK_IMAGE_ASPECT_COLOR_BIT || int_format(sh.format) ||
        !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT))
        rt_fatal("gfx: vulkan: present of the texture at %p (NVN format %#x, Vulkan format %d): the capture cannot blit "
                 "from it", (const void *)t->mem, t->d.format, (int)sh.format);
    // the crop, clipped to level 0; an empty one is the whole texture
    int64_t x0 = crop[0], y0 = crop[1], x1 = x0 + crop[2], y1 = y0 + crop[3];
    if (crop[2] <= 0 || crop[3] <= 0 || vkc_no_crop) x0 = 0, y0 = 0, x1 = sh.width, y1 = sh.height;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int64_t)sh.width) x1 = sh.width;
    if (y1 > (int64_t)sh.height) y1 = sh.height;
    if (x1 <= x0 || y1 <= y0) {   // the null back end takes any crop: the whole texture, said once
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("gfx: vulkan: a present's crop (%d, %d, %d, %d) is outside its %ux%u texture: the whole texture is "
                   "captured (logged once)", crop[0], crop[1], crop[2], crop[3], sh.width, sh.height);
        x0 = 0, y0 = 0, x1 = sh.width, y1 = sh.height;
    }
    uint32_t w = (uint32_t)(x1 - x0), h = (uint32_t)(y1 - y0);

    rt_kernel_lock(&g_mu);
    settings();
    uint64_t seq = __atomic_add_fetch(&g_presents, 1, __ATOMIC_RELAXED);
    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cmd = gpu_vk_cmd();
    vkb_image_flush(cmd);
    int fresh = capture_image(w, h);
    // after every submission before this one (the first scope of a barrier holds the earlier commands of the queue)
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT, fresh);
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{(int32_t)x0, (int32_t)y0, 0}, {(int32_t)x1, (int32_t)y1, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cmd, vkb_image_handle(img), VK_IMAGE_LAYOUT_GENERAL, C.image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                   VK_FILTER_NEAREST);
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, 0);
    if (C.dir[0] && (seq - 1) % C.every == 0) {
        Dump *d = malloc(sizeof *d);
        if (!d) rt_fatal("gfx: vulkan: out of memory for a capture dump");
        d->w = w, d->h = h, d->seq = seq;
        host_buffer((VkDeviceSize)w * h * 4, &d->buf, &d->mem, &d->map);
        copy_out(cmd, d->buf);
        gpu_vk_effect(dump_written, d);
    }
    vkb_image_use(img, value);
    C.seq = seq;
    gpu_vk_end(NULL, 0);
    plat_mutex_unlock(&g_mu);
}

// ---------- reading
uint64_t vkc_capture_read(uint8_t *dst, size_t cap, uint32_t *w, uint32_t *h) {
    rt_kernel_lock(&g_mu);
    *w = C.w, *h = C.h;
    VkDeviceSize size = (VkDeviceSize)C.w * C.h * 4;
    if (!C.image || cap < size) {
        plat_mutex_unlock(&g_mu);
        return 0;
    }
    if (C.rsize < size) {
        if (C.rbuf) vkDestroyBuffer(gpu_vk.device, C.rbuf, NULL), vkFreeMemory(gpu_vk.device, C.rmem, NULL);
        host_buffer(size, &C.rbuf, &C.rmem, &C.rmap);
        C.rsize = size;
    }
    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);
    copy_out(gpu_vk_cmd(), C.rbuf);
    gpu_vk_end(NULL, 0);
    gpu_sync_wait(value, UINT64_MAX);
    memcpy(dst, C.rmap, (size_t)size);
    uint64_t seq = C.seq;
    plat_mutex_unlock(&g_mu);
    return seq;
}

int vkc_capture_save(const char *path) {
    uint32_t w, h;
    vkc_capture_read(NULL, 0, &w, &h);
    if (!w || !h) return 0;
    size_t size = (size_t)w * h * 4;
    uint8_t *p = malloc(size);
    if (!p) return 0;
    int ok = vkc_capture_read(p, size, &w, &h) && write_ppm(path, p, w, h);
    free(p);
    return ok;
}

void vkc_close(void) {
    rt_kernel_lock(&g_mu);
    if (C.image) vkDestroyImage(gpu_vk.device, C.image, NULL), vkFreeMemory(gpu_vk.device, C.mem, NULL);
    if (C.rbuf) vkDestroyBuffer(gpu_vk.device, C.rbuf, NULL), vkFreeMemory(gpu_vk.device, C.rmem, NULL);
    memset(&C, 0, sizeof C);
    plat_mutex_unlock(&g_mu);
}
