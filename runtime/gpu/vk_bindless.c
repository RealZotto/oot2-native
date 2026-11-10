// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The device half of the bindless tables (oot2/vk_bindless.h): BindlessOps over Vulkan. The sampler state mapping
// follows DKB Recomp's runtime/vk_texture.c (GPL-3.0-or-later; ported there from LAS Recomp runtime/nvn/vulkan/
// textures.c, MIT); changed here, the host samplers are shared by their whole state (oot2/bindless.h), the border colour
// keeps Vulkan's three colours and counts the rest, and the reduction mode is carried.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_image.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_texture.h"
#include "plat.h"

static struct {
    int open;
    VkDescriptorSetLayout layout;
    VkDescriptorPool pool;
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkSampler sampler;
    int cleared;   // the placeholder's layout change and clear are recorded
    VkbStats st;
} g;
static PlatMutex g_mu = PLAT_MUTEX_INIT;

// ---------- samplers

static VkSamplerAddressMode wrap_of(int w) {
    switch (w) {
    case NVN_WRAP_MODE_CLAMP: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // GL_CLAMP blends the border in; the edge is the usual stand-in
    case NVN_WRAP_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case NVN_WRAP_MODE_MIRROR_CLAMP:
    case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER:
        __atomic_fetch_add(&g.st.wrap_approx, 1, __ATOMIC_RELAXED);
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case NVN_WRAP_MODE_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case NVN_WRAP_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // CLAMP_TO_EDGE
    }
}

static VkBorderColor border_of(const float b[4]) {
    uint32_t w[4];
    memcpy(w, b, sizeof w);
    const uint32_t one = 0x3F800000u;
    if (!w[0] && !w[1] && !w[2] && !w[3]) return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    if (!w[0] && !w[1] && !w[2] && w[3] == one) return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    if (w[0] == one && w[1] == one && w[2] == one && w[3] == one) return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    if (!w[0] && !w[1] && !w[2] && w[3] == 1) return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    if (w[0] == 1 && w[1] == 1 && w[2] == 1 && w[3] == 1) return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    __atomic_fetch_add(&g.st.border_approx, 1, __ATOMIC_RELAXED);
    return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
}

void vkb_sampler_info(const samp_par *s, VkSamplerCreateInfo *ci, VkSamplerReductionModeCreateInfo *red) {
    memset(ci, 0, sizeof *ci);
    ci->sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    int nomip = 0;
    switch (s->minf) {
    case NVN_MIN_FILTER_NEAREST: ci->minFilter = VK_FILTER_NEAREST, ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, nomip = 1; break;
    case NVN_MIN_FILTER_LINEAR: ci->minFilter = VK_FILTER_LINEAR, ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST, nomip = 1; break;
    case NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST: ci->minFilter = VK_FILTER_NEAREST, ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; break;
    case NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST: ci->minFilter = VK_FILTER_LINEAR, ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; break;
    case NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR: ci->minFilter = VK_FILTER_NEAREST, ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; break;
    default: ci->minFilter = VK_FILTER_LINEAR, ci->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; break;
    }
    ci->magFilter = s->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    ci->addressModeU = wrap_of(s->ws), ci->addressModeV = wrap_of(s->wt), ci->addressModeW = wrap_of(s->wr);
    ci->mipLodBias = s->lod_bias;
    if (s->aniso > 1.0f) {
        ci->anisotropyEnable = VK_TRUE;
        float max = gpu_vk.facts.props.limits.maxSamplerAnisotropy;
        ci->maxAnisotropy = s->aniso < max ? s->aniso : max;
    } else {
        ci->maxAnisotropy = 1.0f;
    }
    if (s->cmp_mode != NVN_COMPARE_MODE_NONE) {
        ci->compareEnable = VK_TRUE;
        int f = s->cmp_func >= NVN_COMPARE_FUNC_NEVER && s->cmp_func <= NVN_COMPARE_FUNC_ALWAYS ? s->cmp_func : NVN_COMPARE_FUNC_NEVER;
        ci->compareOp = (VkCompareOp)(f - NVN_COMPARE_FUNC_NEVER);   // NEVER..ALWAYS in the same order as Vulkan's
    }
    ci->minLod = s->lod_min;
    ci->maxLod = nomip ? fmaxf(s->lod_min, fminf(s->lod_max, 0.25f)) : s->lod_max;   // 0.25: GL's "no mipmaps"
    ci->borderColor = border_of(s->border);
    if (s->reduction != NVN_SAMPLER_REDUCTION_AVERAGE) {
        memset(red, 0, sizeof *red);
        red->sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
        red->reductionMode = s->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN : VK_SAMPLER_REDUCTION_MODE_MAX;
        ci->pNext = red;
    }
}

static void *sampler_make(const samp_par *s) {
    VkSamplerCreateInfo ci;
    VkSamplerReductionModeCreateInfo red;
    vkb_sampler_info(s, &ci, &red);
    VkSampler sampler;
    GPU_VK_OK(vkCreateSampler(gpu_vk.device, &ci, NULL, &sampler), "vkCreateSampler");
    return (void *)(uintptr_t)sampler;
}

static void sampler_free(void *sampler) { vkDestroySampler(gpu_vk.device, (VkSampler)(uintptr_t)sampler, NULL); }

void vkb_stats(VkbStats *out) { *out = g.st; }

// ---------- copies of set 0

static void *set_make(void) {
    VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = g.pool,
                                      .descriptorSetCount = 1, .pSetLayouts = &g.layout};
    VkDescriptorSet set;
    GPU_VK_OK(vkAllocateDescriptorSets(gpu_vk.device, &alloc, &set), "vkAllocateDescriptorSets (a copy of set 0)");
    uint32_t n = VKP_GLOBAL_IMAGES > VKP_GLOBAL_SAMPLERS ? VKP_GLOBAL_IMAGES : VKP_GLOBAL_SAMPLERS;
    VkDescriptorImageInfo *info = malloc((size_t)n * sizeof *info);
    if (!info) rt_fatal("gfx: vulkan: out of memory for a copy of set 0");
    for (uint32_t i = 0; i < n; i++) info[i] = (VkDescriptorImageInfo){g.sampler, g.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
         .descriptorCount = VKP_GLOBAL_IMAGES, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = info},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
         .descriptorCount = VKP_GLOBAL_SAMPLERS, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = info}};
    vkUpdateDescriptorSets(gpu_vk.device, 2, w, 0, NULL);
    free(info);
    return (void *)(uintptr_t)set;
}

static void set_free(void *set) {
    VkDescriptorSet s = (VkDescriptorSet)(uintptr_t)set;
    vkFreeDescriptorSets(gpu_vk.device, g.pool, 1, &s);
}

#define WRITE_MAX 256
static void write(void *set, const BindlessWrite *w, int n) {
    VkWriteDescriptorSet wr[WRITE_MAX];
    VkDescriptorImageInfo info[WRITE_MAX];
    VkDescriptorSet s = (VkDescriptorSet)(uintptr_t)set;
    for (int i = 0; i < n; i += WRITE_MAX) {
        int m = n - i < WRITE_MAX ? n - i : WRITE_MAX;
        for (int k = 0; k < m; k++) {
            const BindlessWrite *b = &w[i + k];
            if (b->kind == BW_IMAGE) {
                info[k] = (VkDescriptorImageInfo){VK_NULL_HANDLE, b->object ? vkb_view_handle(b->object) : g.view,
                                                  VK_IMAGE_LAYOUT_GENERAL};
            } else {
                info[k] = (VkDescriptorImageInfo){b->object ? (VkSampler)(uintptr_t)b->object : g.sampler, VK_NULL_HANDLE,
                                                  VK_IMAGE_LAYOUT_UNDEFINED};
            }
            wr[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = s,
                                           .dstBinding = b->kind == BW_IMAGE ? 0u : 1u, .dstArrayElement = b->index,
                                           .descriptorCount = 1,
                                           .descriptorType = b->kind == BW_IMAGE ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                                                                                 : VK_DESCRIPTOR_TYPE_SAMPLER,
                                           .pImageInfo = &info[k]};
        }
        vkUpdateDescriptorSets(gpu_vk.device, (uint32_t)m, wr, 0, NULL);
    }
}

static const BindlessOps k_ops = {.view_make = vkb_view_make, .view_free = vkb_view_free, .sampler_make = sampler_make,
                                  .sampler_free = sampler_free, .set_make = set_make, .set_free = set_free,
                                  .write = write, .texture_release = vkb_image_release, .view_demand = vkb_view_demand,
                                  .view_use = vkb_view_use};

static void wait_value(uint64_t value) { gpu_sync_wait(value, UINT64_MAX); }
static const BindlessClock k_clock = {.done = gpu_sync_done, .wait = wait_value};

// ---------- open and close

static uint32_t device_local(uint32_t bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return i;
    rt_fatal("gfx: vulkan: no device-local memory type for the placeholder image (types %#x)", bits);
}

// The placeholder's first use: its layout change and its clear are recorded into the first submission that takes a copy,
// so that opening the device hands out no timeline value (the tests of the device count the values from 1).
static void record_placeholder(VkCommandBuffer cmd) {
    VkImageMemoryBarrier2 to_clear = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = g.image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                            .pImageMemoryBarriers = &to_clear};
    vkCmdPipelineBarrier2(cmd, &dep);
    VkClearColorValue zero = {{0, 0, 0, 0}};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, g.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    VkMemoryBarrier2 after = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                              .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT};
    dep = (VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &after};
    vkCmdPipelineBarrier2(cmd, &dep);
}

void vkb_open(void) {
    plat_mutex_lock(&g_mu);
    if (g.open) {
        plat_mutex_unlock(&g_mu);
        return;
    }
    memset(&g, 0, sizeof g);
    VkDescriptorSetLayoutBinding b[2];
    VkDescriptorBindingFlags flags[2];
    uint32_t n = vkp_set0_bindings(b, flags);
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                      .bindingCount = n, .pBindingFlags = flags};
    VkDescriptorSetLayoutCreateInfo lc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &bf,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                          .bindingCount = n, .pBindings = b};
    GPU_VK_OK(vkCreateDescriptorSetLayout(gpu_vk.device, &lc, NULL, &g.layout), "vkCreateDescriptorSetLayout (the copies of set 0)");
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKP_GLOBAL_IMAGES * VKB_MAX_COPIES},
                                     {VK_DESCRIPTOR_TYPE_SAMPLER, VKP_GLOBAL_SAMPLERS * VKB_MAX_COPIES}};
    VkDescriptorPoolCreateInfo pc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |
                                              VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                     .maxSets = VKB_MAX_COPIES, .poolSizeCount = 2, .pPoolSizes = sizes};
    GPU_VK_OK(vkCreateDescriptorPool(gpu_vk.device, &pc, NULL, &g.pool), "vkCreateDescriptorPool (the copies of set 0)");

    // The placeholder: one transparent texel, which every entry nothing was registered at shows, and a default sampler.
    VkImageCreateInfo ic = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    GPU_VK_OK(vkCreateImage(gpu_vk.device, &ic, NULL, &g.image), "vkCreateImage (the placeholder)");
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(gpu_vk.device, g.image, &mr);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = device_local(mr.memoryTypeBits)};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &g.memory), "vkAllocateMemory (the placeholder)");
    GPU_VK_OK(vkBindImageMemory(gpu_vk.device, g.image, g.memory, 0), "vkBindImageMemory (the placeholder)");
    VkImageViewCreateInfo vc = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g.image,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    GPU_VK_OK(vkCreateImageView(gpu_vk.device, &vc, NULL, &g.view), "vkCreateImageView (the placeholder)");
    VkSamplerCreateInfo sc = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxAnisotropy = 1.0f, .maxLod = 0.25f};
    GPU_VK_OK(vkCreateSampler(gpu_vk.device, &sc, NULL, &g.sampler), "vkCreateSampler (the placeholder)");
    uint32_t limit = gpu_vk.facts.props.limits.maxSamplerAllocationCount;
    // images - second is the game's pool: VKP_POOL_IMAGES IDs at their own indices, the second views after them
    BindlessConfig cfg = {.images = VKP_GLOBAL_IMAGES, .samplers = VKP_GLOBAL_SAMPLERS, .second = VKP_SECOND_IMAGES,
                          .max_copies = VKB_MAX_COPIES, .max_samplers = limit > 160 ? limit - 128 : 32};
    g.open = 1;
    plat_mutex_unlock(&g_mu);
    vkt_open();
    bindless_open(&k_ops, &cfg, &k_clock);
}

void vkb_close(void) {
    plat_mutex_lock(&g_mu);
    if (!g.open) {
        plat_mutex_unlock(&g_mu);
        return;
    }
    plat_mutex_unlock(&g_mu);
    bindless_close();
    vkb_image_close();
    vkt_close();
    plat_mutex_lock(&g_mu);
    vkDestroySampler(gpu_vk.device, g.sampler, NULL);
    vkDestroyImageView(gpu_vk.device, g.view, NULL);
    vkDestroyImage(gpu_vk.device, g.image, NULL);
    vkFreeMemory(gpu_vk.device, g.memory, NULL);
    vkDestroyDescriptorPool(gpu_vk.device, g.pool, NULL);
    vkDestroyDescriptorSetLayout(gpu_vk.device, g.layout, NULL);
    memset(&g, 0, sizeof g);
    plat_mutex_unlock(&g_mu);
}

int vkb_is_open(void) { return g.open; }
VkDescriptorSetLayout vkb_set_layout(void) { return g.layout; }

VkDescriptorSet vkb_acquire(VkCommandBuffer cmd, uint64_t value, const dpool_ref *textures, const dpool_ref *samplers) {
    plat_mutex_lock(&g_mu);
    int first = !g.cleared;
    g.cleared = 1;
    plat_mutex_unlock(&g_mu);
    if (first) record_placeholder(cmd);
    VkDescriptorSet s = (VkDescriptorSet)(uintptr_t)bindless_acquire(value, textures, samplers);
    vkb_image_flush(cmd);
    vkb_image_stamp_pools(value);
    vkb_image_reap();
    return s;
}

uint32_t vkb_image_slot(VkCommandBuffer cmd, uint32_t id, int type) {
    uint32_t slot = bindless_image_slot(id, type);
    vkb_image_flush(cmd);
    return slot;
}
