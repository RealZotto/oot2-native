// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// HMMA on VK_KHR_cooperative_matrix: the start-up gate and probe (oot2/vk_hmma.h). The probe shader is
// runtime/gpu/shaders/hmma_probe.comp; it runs once, as one internal submission on the timeline, before any module of the
// game is built. Hookshot's start-up check of its own matrix path (gpu/vk/vk_coopmat.cpp, GPL) was read for the idea of
// loading matrices whose components are their positions; nothing of it is copied.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_hmma.h"

enum {
    W_A = 0, W_B = 128, W_C = 192,   // the position matrices, row-major halves
    W_IN = 256,                      // lane t's console fragments: 8 words at W_IN + 8t
    W_ORDER = 512,                   // lane t's A, B and C components: 8 words at W_ORDER + 8t
    W_NATIVE = 768, W_HELPER = 832,  // lane t's D: 2 words at W_NATIVE + 2t and W_HELPER + 2t
    W_LENGTHS = 896,                 // the matrices' per-lane lengths: A, B, C
    W_TOTAL = 900
};

typedef struct {
    uint32_t words[2];   // the buffer's device address, low word first
    uint32_t inv_a[8], inv_b[4], inv_c[4], ord_c[4];
} Push;

static struct {
    int native;
    char reason[192];
} H;

int vkh_native(void) { return H.native; }
const char *vkh_reason(void) { return H.native ? "" : H.reason; }
int vkh_fault;

static void refuse(const char *fmt, const char *detail) {
    H.native = 0;
    snprintf(H.reason, sizeof H.reason, fmt, detail ? detail : "");
}

// ---------- the console's fragments (PTX mma.m16n8k16): lane t, g = t / 4, q = t % 4, element e of a lane's registers
static void console_a(int t, int e, int *row, int *k) {
    int g = t / 4, q = t % 4;
    *row = g + 8 * ((e >> 1) & 1);
    *k = 2 * q + (e & 1) + 8 * (e >> 2);
}
static void console_b(int t, int e, int *k, int *n) {
    int g = t / 4, q = t % 4;
    *k = 2 * q + (e & 1) + 8 * (e >> 1);
    *n = g;
}
static void console_c(int t, int e, int *row, int *n) {
    int g = t / 4, q = t % 4;
    *row = g + 8 * (e >> 1);
    *n = 2 * q + (e & 1);
}

// Halves of small integers (exact), and back.
static uint16_t to_half(int v) {
    if (!v) return 0;
    uint16_t sign = v < 0 ? 0x8000 : 0;
    unsigned m = (unsigned)(v < 0 ? -v : v);
    int e = 0;
    while ((m >> e) > 1) e++;
    return (uint16_t)(sign | (unsigned)(e + 15) << 10 | ((m << 10 >> e) & 0x3ff));
}
static int from_half(uint16_t h, int *ok) {
    int e = (h >> 10) & 0x1f;
    unsigned m = h & 0x3ff;
    if (!e && !m) return 0;
    if (!e || e == 31 || e < 15 || e > 25 || (m & ((1u << (25 - e)) - 1) & 0x3ff)) {
        *ok = 0;   // not a small integer
        return 0;
    }
    int v = (int)((0x400 | m) >> (25 - e));
    return h & 0x8000 ? -v : v;
}
static uint16_t half_of(const uint32_t *w, int index) { return (uint16_t)(w[index / 2] >> (16 * (index & 1))); }

// The values of the multiply: small integers, so every sum is exact in f16 whatever the order or width of the additions.
static int val_a(int r, int k) { return (r * 5 + k * 3 + 1) % 7 - 3; }
static int val_b(int k, int n) { return (k * 7 + n * 5 + 2) % 5 - 2; }
static int val_c(int r, int n) { return (r * 3 + n * 11) % 9 - 4; }

// The component of each console element in every lane's loaded matrix, or -1 with why: the console element e of lane t is
// at position (r, c) of a matrix with cols columns, whose component j holds the value r * cols + c.
static int measure(const uint32_t *w, int at, int count, int cols, void (*console)(int, int, int *, int *), int *order,
                   char *why, size_t cap) {
    for (int e = 0; e < count; e++) {
        order[e] = -1;
        for (int t = 0; t < 32; t++) {
            int r, c, j = 0, ok = 1;
            console(t, e, &r, &c);
            while (j < count && from_half(half_of(w + at + 8 * t, j), &ok) != r * cols + c) j++;
            if (j == count || !ok || (order[e] >= 0 && order[e] != j)) {
                snprintf(why, cap, "lane %d does not hold its element %d (row %d, column %d) where lane 0 does", t, e, r, c);
                return -1;
            }
            order[e] = j;
        }
    }
    return 0;
}

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// The gate: the device's matrix list, its 16-bit floats and the stage. NULL when the device has all, else the reason.
static const char *gate(void) {
    Oot2VkFacts en;
    oot2_vk_enable(&gpu_vk.facts, &en);
    if (!en.coop.cooperativeMatrix || !vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        return "the device was made without cooperative matrices";
    if (!en.f12.shaderFloat16) return "the device has no 16-bit floats in shaders";
    if (!en.f12.vulkanMemoryModel) return "the device was made without the Vulkan memory model";
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR stages = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &stages};
    vkGetPhysicalDeviceProperties2(gpu_vk.phys, &p2);
    if (!(stages.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
        return "the device's cooperative matrices are not for compute";
    uint32_t n = 0;
    if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(gpu_vk.phys, &n, NULL) != VK_SUCCESS || !n)
        return "the device lists no cooperative matrix";
    VkCooperativeMatrixPropertiesKHR *p = calloc(n, sizeof *p);
    if (!p) rt_fatal("gfx: vulkan: out of memory for the cooperative matrix list");
    for (uint32_t i = 0; i < n; i++) p[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    int shape = 0;
    if (vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(gpu_vk.phys, &n, p) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++)
            shape |= p[i].MSize == 16 && p[i].NSize == 8 && p[i].KSize == 16 && p[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                     p[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR && p[i].CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                     p[i].ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR && !p[i].saturatingAccumulation &&
                     p[i].scope == VK_SCOPE_SUBGROUP_KHR;
    free(p);
    return shape ? NULL : "the device lists no 16x8x16 subgroup matrix with f16 A, B, C and result";
}

// Tests only (vkh_fault): the read-back with one fault in it. Lane 3's first half of D is a small integer in both forms, so
// adding one to its word changes that half alone and leaves it no longer the exact sum.
static void inject_fault(uint32_t *words) {
    if (vkh_fault == VKH_FAULT_LENGTHS) words[W_LENGTHS] = 7;
    if (vkh_fault == VKH_FAULT_HELPER || vkh_fault == VKH_FAULT_SUMS) words[W_HELPER + 2 * 3] += 1;
    if (vkh_fault == VKH_FAULT_SUMS) words[W_NATIVE + 2 * 3] += 1;
}

// One dispatch of the probe over words (W_TOTAL), read back into words. 0 with why when the shader is missing.
static int run_probe(uint32_t *words, const Push *push_in, char *why, size_t cap) {
    const GpuShader *s = gpu_shader("hmma_probe");
    if (!s) {
        snprintf(why, cap, "the runtime was built without the hmma_probe shader");
        return 0;
    }
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = W_TOTAL * 4,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &buf), "vkCreateBuffer (the HMMA probe)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, buf, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no host-visible coherent memory for the HMMA probe");
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi,
                                  .allocationSize = mr.size, .memoryTypeIndex = type};
    VkDeviceMemory mem;
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &mem), "vkAllocateMemory (the HMMA probe)");
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, buf, mem, 0), "vkBindBufferMemory (the HMMA probe)");
    void *map;
    GPU_VK_OK(vkMapMemory(gpu_vk.device, mem, 0, VK_WHOLE_SIZE, 0, &map), "vkMapMemory (the HMMA probe)");
    memcpy(map, words, W_TOTAL * 4);
    VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf};
    VkDeviceAddress address = vkGetBufferDeviceAddress(gpu_vk.device, &di);
    Push push = *push_in;
    push.words[0] = (uint32_t)address, push.words[1] = (uint32_t)(address >> 32);

    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->size, .pCode = s->code};
    VkShaderModule m;
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &m), "vkCreateShaderModule (the HMMA probe)");
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pr};
    VkPipelineLayout layout;
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &li, NULL, &layout), "vkCreatePipelineLayout (the HMMA probe)");
    // the probe's 32 lanes are one subgroup (the subgroup_32 item fixes the size at 32), created as the game's programs are
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = layout};
    VkPipeline pipe;
    GPU_VK_OK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &pipe),
              "vkCreateComputePipelines (the HMMA probe)");
    vkDestroyShaderModule(gpu_vk.device, m, NULL);

    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &dep);
    gpu_vk_end(NULL, 0);
    gpu_sync_wait(value, UINT64_MAX);
    memcpy(words, map, W_TOTAL * 4);
    inject_fault(words);

    vkDestroyPipeline(gpu_vk.device, pipe, NULL);
    vkDestroyPipelineLayout(gpu_vk.device, layout, NULL);
    vkDestroyBuffer(gpu_vk.device, buf, NULL);
    vkFreeMemory(gpu_vk.device, mem, NULL);
    return 1;
}

static void text_order(char *out, size_t cap, const int *order, int n) {
    size_t at = 0;
    out[0] = 0;
    for (int i = 0; i < n && at < cap; i++) at += (size_t)snprintf(out + at, cap - at, i ? " %d" : "%d", order[i]);
}

void vkh_probe(void) {
    H.native = 1;
    H.reason[0] = 0;
    const char *setting = getenv("OOT2_VK_HMMA");
    if (setting && *setting && strcmp(setting, "auto") && strcmp(setting, "helper"))
        rt_fatal("gfx: vulkan: OOT2_VK_HMMA is \"%s\"; the values are auto and helper", setting);
    const int want_a[8] = VKH_ORDER_A, want_b[4] = VKH_ORDER_B, want_c[4] = VKH_ORDER_C;
    int got_a[8], got_b[4], got_c[4], measured = 0;
    const char *closed = gate();
    if (setting && !strcmp(setting, "helper")) refuse("OOT2_VK_HMMA=helper%s", NULL);
    else if (closed) refuse("%s", closed);
    else {
        uint32_t *w = calloc(W_TOTAL, 4);
        if (!w) rt_fatal("gfx: vulkan: out of memory for the HMMA probe");
        for (int i = 0; i < 256; i++) {   // positions, row-major: A (16 x 16) r * 16 + k, B (16 x 8) k * 8 + n, C r * 8 + n
            w[W_A + i / 2] |= (uint32_t)to_half(i) << (16 * (i & 1));
            if (i < 128) {
                w[W_B + i / 2] |= (uint32_t)to_half(i) << (16 * (i & 1));
                w[W_C + i / 2] |= (uint32_t)to_half(i) << (16 * (i & 1));
            }
        }
        for (int t = 0; t < 32; t++)   // the console's fragments of the multiply
            for (int e = 0; e < 8; e++) {
                int r, c, at = W_IN + 8 * t;
                console_a(t, e, &r, &c);
                w[at + e / 2] |= (uint32_t)to_half(val_a(r, c)) << (16 * (e & 1));
                if (e < 4) {
                    console_b(t, e, &r, &c);
                    w[at + 4 + e / 2] |= (uint32_t)to_half(val_b(r, c)) << (16 * (e & 1));
                    console_c(t, e, &r, &c);
                    w[at + 6 + e / 2] |= (uint32_t)to_half(val_c(r, c)) << (16 * (e & 1));
                }
            }
        Push push;
        memset(&push, 0, sizeof push);
        for (int e = 0; e < 8; e++) push.inv_a[want_a[e]] = (uint32_t)e;
        for (int e = 0; e < 4; e++) push.inv_b[want_b[e]] = (uint32_t)e, push.inv_c[want_c[e]] = (uint32_t)e;
        for (int e = 0; e < 4; e++) push.ord_c[e] = (uint32_t)want_c[e];
        char why[160];
        if (!run_probe(w, &push, why, sizeof why)) refuse("%s", why);
        else if (w[W_LENGTHS] != 8 || w[W_LENGTHS + 1] != 4 || w[W_LENGTHS + 2] != 4) {
            snprintf(why, sizeof why, "a lane holds %u, %u and %u components of A, B and C, not 8, 4 and 4", w[W_LENGTHS],
                     w[W_LENGTHS + 1], w[W_LENGTHS + 2]);
            refuse("%s", why);
        } else if (measure(w, W_ORDER, 8, 16, console_a, got_a, why, sizeof why) ||
                   measure(w, W_ORDER + 4, 4, 8, console_b, got_b, why, sizeof why) ||
                   measure(w, W_ORDER + 6, 4, 8, console_c, got_c, why, sizeof why)) {
            refuse("the device's components are not the console's fragments: %s", why);
        } else {
            measured = 1;
            if (memcmp(got_a, want_a, sizeof got_a) || memcmp(got_b, want_b, sizeof got_b) || memcmp(got_c, want_c, sizeof got_c))
                refuse("the device's element order is not the translator's%s", NULL);
            else {
                int differ = 0, inexact = 0;
                for (int t = 0; t < 32; t++)
                    for (int e = 0; e < 4; e++) {
                        int r, n, ok = 1, sum;
                        console_c(t, e, &r, &n);
                        sum = val_c(r, n);
                        for (int k = 0; k < 16; k++) sum += val_a(r, k) * val_b(k, n);
                        uint16_t native = half_of(w + W_NATIVE + 2 * t, e), helper = half_of(w + W_HELPER + 2 * t, e);
                        differ += native != helper;
                        inexact += from_half(native, &ok) != sum || !ok;
                    }
                if (differ || inexact) {
                    snprintf(why, sizeof why, "a multiply of small integers differs between the forms in %d of 128 elements "
                             "and the native one from the exact sums in %d", differ, inexact);
                    refuse("%s", why);
                }
            }
        }
        free(w);
    }
    char sa[32], sb[16], sc[16], order[160];
    text_order(sa, sizeof sa, measured ? got_a : want_a, 8);
    text_order(sb, sizeof sb, measured ? got_b : want_b, 4);
    text_order(sc, sizeof sc, measured ? got_c : want_c, 4);
    if (measured) snprintf(order, sizeof order, "the device holds the console's elements at its components A %s, B %s, C %s", sa, sb, sc);
    else snprintf(order, sizeof order, "the device's order not measured");
    if (H.native)
        rt_log("gfx: vulkan: HMMA on cooperative matrices (16x8x16 f16, subgroup): %s, the translator's order, and a "
               "multiply of small integers gives the exact sums in both forms (wiring and order, not rounding): the modules "
               "run as translated", order);
    else
        rt_log("gfx: vulkan: HMMA in the helper: %s; every module's native HMMA becomes a call of the helper, by name (%s)",
               H.reason, order);
}
