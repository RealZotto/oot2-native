// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The `uni` pipeline layouts (oot2/vk_pipeline.h) as tables, and their creation. The tables take no device, so the
// tests that tie them to the translator's numbering (xlate/host/UniLayout.cs) and to the specification's limits run
// without one; the creation functions use the back end's device (gpu_vk).
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/vk_pipeline.h"

#define GFX_STAGES (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TASK_BIT_EXT | \
                    VK_SHADER_STAGE_MESH_BIT_EXT)
#define PRE_RASTER_STAGES (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_MESH_BIT_EXT)
#define ALL_STAGES (GFX_STAGES | VK_SHADER_STAGE_COMPUTE_BIT)

// A device is required to give a set 1,048,576 update-after-bind sampled images (oot2/vk_required.h, the item
// update_after_bind_images), which is all that is checked of binding 0's size: it must stay inside that number.
_Static_assert(VKP_GLOBAL_IMAGES <= (1u << 20), "binding 0 is larger than the update_after_bind_images item of oot2/vk_required.h asks of a device");

uint32_t vkp_set0_bindings(VkDescriptorSetLayoutBinding out[2], VkDescriptorBindingFlags flags[2]) {
    out[0] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKP_GLOBAL_IMAGES, ALL_STAGES, NULL};
    out[1] = (VkDescriptorSetLayoutBinding){1, VK_DESCRIPTOR_TYPE_SAMPLER, VKP_GLOBAL_SAMPLERS, ALL_STAGES, NULL};
    // Update-unused-while-pending: the bindless tables write the slot of a second view into a copy that earlier
    // submissions still read, which none of their handles names (oot2/vk_bindless.h).
    flags[0] = flags[1] = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    return 2;
}

// A run of count bindings of one type from first, visible to stages.
static uint32_t run(VkDescriptorSetLayoutBinding *out, uint32_t at, uint32_t first, uint32_t count, VkDescriptorType type,
                    VkShaderStageFlags stages) {
    for (uint32_t i = 0; i < count; i++)
        out[at + i] = (VkDescriptorSetLayoutBinding){first + i, type, 1, stages, NULL};
    return at + count;
}

uint32_t vkp_set1_bindings(int layout, VkDescriptorSetLayoutBinding out[VKP_SET1_MAX]) {
    const VkDescriptorType ub = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sb = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                           si = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    uint32_t n = 0;
    if (layout == VKP_COMPUTE) {
        n = run(out, n, 0, 9, ub, VK_SHADER_STAGE_COMPUTE_BIT);     // the support buffer, then uniform 1 to 8
        n = run(out, n, 9, 16, sb, VK_SHADER_STAGE_COMPUTE_BIT);    // storage 9 to 24
        n = run(out, n, 25, 7, si, VK_SHADER_STAGE_COMPUTE_BIT);    // storage images 25 to 31
    } else {
        n = run(out, n, 0, 1, ub, GFX_STAGES);                      // the support buffer, shared
        n = run(out, n, 1, 8, ub, PRE_RASTER_STAGES);               // vertex or mesh uniform 1 to 8
        n = run(out, n, 9, 5, sb, PRE_RASTER_STAGES);               // vertex or mesh storage 9 to 13
        n = run(out, n, 14, 11, ub, VK_SHADER_STAGE_FRAGMENT_BIT);  // pixel uniform 14 to 24
        n = run(out, n, 25, 5, sb, VK_SHADER_STAGE_FRAGMENT_BIT);   // pixel storage 25 to 29
        n = run(out, n, 30, 2, si, VK_SHADER_STAGE_FRAGMENT_BIT);   // pixel storage images 30 and 31
    }
    return n;
}

uint32_t vkp_set2_bindings(VkDescriptorSetLayoutBinding out[VKP_SET2_MAX]) {
    uint32_t n = run(out, 0, 0, 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_TASK_BIT_EXT);
    return run(out, n, 8, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_TASK_BIT_EXT);
}

VkPushConstantRange vkp_push_range(int layout) {
    return (VkPushConstantRange){layout == VKP_COMPUTE ? VK_SHADER_STAGE_COMPUTE_BIT : GFX_STAGES, 0, VKP_PUSH_BYTES};
}

// ---- dynamic state

static uint32_t add(VkDynamicState *out, uint32_t n, const VkDynamicState *list, uint32_t count) {
    memcpy(out + n, list, count * sizeof *list);
    return n + count;
}

uint32_t vkp_dynamic_states(VkGraphicsPipelineLibraryFlagsEXT part, const Oot2VkFacts *f, VkDynamicState out[VKP_MAX_DYNAMIC]) {
    uint32_t n = 0;
    const VkPhysicalDeviceExtendedDynamicState3FeaturesEXT *e3 = &f->eds3;
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT) {
        static const VkDynamicState s[] = {VK_DYNAMIC_STATE_VERTEX_INPUT_EXT, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                                           VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
        n = add(out, n, s, sizeof s / sizeof s[0]);
    }
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT) {
        static const VkDynamicState s[] = {
            VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_LINE_WIDTH,
            VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
            VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE};
        n = add(out, n, s, sizeof s / sizeof s[0]);
        if (e3->extendedDynamicState3PolygonMode) out[n++] = VK_DYNAMIC_STATE_POLYGON_MODE_EXT;
        if (e3->extendedDynamicState3DepthClampEnable) out[n++] = VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT;
        // NVN's depth mode (nvn_depth_mode): -1 to 1 or 0 to 1, chosen per draw by the replay (oot2/vk_replay.h)
        if (e3->extendedDynamicState3DepthClipNegativeOneToOne)
            out[n++] = VK_DYNAMIC_STATE_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT;
    }
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) {
        static const VkDynamicState s[] = {
            VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
            VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
            VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
            VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_DEPTH_BOUNDS};
        n = add(out, n, s, sizeof s / sizeof s[0]);
    }
    if (part & VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT) {
        out[n++] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
        if (e3->extendedDynamicState3LogicOpEnable) out[n++] = VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT;
        if (e3->extendedDynamicState3ColorBlendEnable) out[n++] = VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT;
        if (e3->extendedDynamicState3ColorBlendEquation) out[n++] = VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT;
        if (e3->extendedDynamicState3ColorWriteMask) out[n++] = VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT;
    }
    return n;
}

// ---- creation

static VkDescriptorSetLayout make_set(const VkDescriptorSetLayoutBinding *b, uint32_t n, VkDescriptorSetLayoutCreateFlags flags,
                                      const VkDescriptorBindingFlags *binding_flags, const char *what) {
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, .bindingCount = n,
        .pBindingFlags = binding_flags};
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .pNext = binding_flags ? &bf : NULL, .flags = flags, .bindingCount = n,
                                          .pBindings = b};
    VkDescriptorSetLayout l;
    GPU_VK_OK(vkCreateDescriptorSetLayout(gpu_vk.device, &ci, NULL, &l), what);
    return l;
}

void vkp_layouts_create(VkpLayouts *l) {
    memset(l, 0, sizeof *l);
    VkDescriptorSetLayoutBinding b[VKP_SET1_MAX];
    VkDescriptorBindingFlags flags[2];
    uint32_t n = vkp_set0_bindings(b, flags);
    l->global = make_set(b, n, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, flags,
                         "vkCreateDescriptorSetLayout (set 0)");
    n = vkp_set1_bindings(VKP_GRAPHICS, b);
    l->graphics_push = make_set(b, n, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, NULL,
                                "vkCreateDescriptorSetLayout (set 1, graphics)");
    n = vkp_set1_bindings(VKP_COMPUTE, b);
    l->compute_push = make_set(b, n, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, NULL,
                               "vkCreateDescriptorSetLayout (set 1, compute)");
    n = vkp_set2_bindings(b);
    l->task = make_set(b, n, 0, NULL, "vkCreateDescriptorSetLayout (set 2)");

    VkDescriptorSetLayout gfx[3] = {l->global, l->graphics_push, l->task};
    VkPushConstantRange pr = vkp_push_range(VKP_GRAPHICS);
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 3,
                                     .pSetLayouts = gfx, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &ci, NULL, &l->graphics), "vkCreatePipelineLayout (graphics)");
    VkDescriptorSetLayout comp[2] = {l->global, l->compute_push};
    pr = vkp_push_range(VKP_COMPUTE);
    ci.setLayoutCount = 2, ci.pSetLayouts = comp, ci.pPushConstantRanges = &pr;
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &ci, NULL, &l->compute), "vkCreatePipelineLayout (compute)");
}

void vkp_layouts_destroy(VkpLayouts *l) {
    if (l->graphics) vkDestroyPipelineLayout(gpu_vk.device, l->graphics, NULL);
    if (l->compute) vkDestroyPipelineLayout(gpu_vk.device, l->compute, NULL);
    VkDescriptorSetLayout sets[4] = {l->global, l->graphics_push, l->compute_push, l->task};
    for (int i = 0; i < 4; i++)
        if (sets[i]) vkDestroyDescriptorSetLayout(gpu_vk.device, sets[i], NULL);
    memset(l, 0, sizeof *l);
}
