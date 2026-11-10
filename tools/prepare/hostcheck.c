// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The host probe of the check step of `oot2 prepare` (tools/prepare/check.py). The check compiles it with the C
// compiler the compile step will use and runs it once. It prints one JSON object on stdout: the CPU features behind
// -march=x86-64-v3, the list of Vulkan device items the renderer requires ("vulkan_items", [id, what] pairs from
// runtime/include/oot2/vk_required.h, which the runtime includes too), and each Vulkan device with its facts and, under
// "items", whether it has each item of that list, judged by the header's own tests. check.py reads the facts and the
// verdicts and decides; it keeps no list of its own.
//
// The Vulkan loader is opened at run time, so no import library is linked and a machine without a loader prints
// "loader": false instead of failing to start. Built without the Vulkan headers, or with OOT2_PROBE_NO_VULKAN, the
// probe prints "vulkan": null and only the CPU and the list are reported. It is built with runtime/include on the
// include path.
//
// Adapted from TotK Recomp installer/hostcheck.c at 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh,
// licenses/totk-MIT.txt): the CPUID reading of the x86-64-v3 features and the run-time loading of the Vulkan loader.
// Changed: each CPU feature is printed on its own rather than as a level, the device report covers the features,
// properties, extensions and memory heaps that architecture section 1 names, and the device items come from
// runtime/include/oot2/vk_required.h.
#include <cpuid.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_include) && !defined(OOT2_PROBE_NO_VULKAN)
#if __has_include(<vulkan/vulkan_core.h>)
#define PROBE_VULKAN 1
#endif
#endif

#ifdef PROBE_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>
#ifdef _WIN32
#include <windows.h>
#define open_lib() ((void *)LoadLibraryA("vulkan-1.dll"))
#define lib_sym(l, n) ((void *)GetProcAddress((HMODULE)(l), (n)))
#else
#include <dlfcn.h>
#define open_lib() dlopen("libvulkan.so.1", RTLD_NOW)
#define lib_sym(l, n) dlsym((l), (n))
#endif
#endif

// After the Vulkan headers, so that with them the header also defines the facts, the query and the judgement.
#include "oot2/vk_required.h"

// A small JSON writer: one "first item" flag per open object or array.
static int first_at[16];
static int depth;

static void sep(void) {
    if (!first_at[depth]) printf(", ");
    first_at[depth] = 0;
}

static void open_(char c) {
    putchar(c);
    first_at[++depth] = 1;
}

static void close_(char c) {
    putchar(c);
    depth--;
}

static void json_string(const char *s) {
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') printf("\\%c", *s);
        else if ((unsigned char)*s < 0x20) printf("\\u%04x", (unsigned)(unsigned char)*s);
        else putchar(*s);
    }
    putchar('"');
}

static void key(const char *k) {
    sep();
    json_string(k);
    printf(": ");
}

static void kbool(const char *k, int v) {
    key(k);
    printf(v ? "true" : "false");
}

// The CPUID bits of the x86-64-v2 and v3 levels, each on its own, so the check can name what is missing.
static void probe_cpu(void) {
    unsigned a, b, c, d;
    unsigned c1 = 0, b7 = 0, cx = 0, max_ext = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) c1 = c;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) b7 = b;
    if (__get_cpuid(0x80000000u, &a, &b, &c, &d)) max_ext = a;
    if (max_ext >= 0x80000001u && __get_cpuid(0x80000001u, &a, &b, &c, &d)) cx = c;
    int ymm = 0;
    if (c1 & (1u << 27)) {   // OSXSAVE: xgetbv may be used
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        (void)hi;
        ymm = (lo & 6u) == 6u;   // the OS saves the XMM and YMM state
    }
    char brand[49] = {0};
    if (max_ext >= 0x80000004u) {
        unsigned r[12];
        for (unsigned i = 0; i < 3; i++) __get_cpuid(0x80000002u + i, &r[4 * i], &r[4 * i + 1], &r[4 * i + 2], &r[4 * i + 3]);
        memcpy(brand, r, 48);
    }
    key("cpu");
    open_('{');
    key("brand");
    json_string(brand);
    kbool("sse3", c1 & (1u << 0));
    kbool("ssse3", c1 & (1u << 9));
    kbool("cx16", c1 & (1u << 13));
    kbool("sse4_1", c1 & (1u << 19));
    kbool("sse4_2", c1 & (1u << 20));
    kbool("popcnt", c1 & (1u << 23));
    kbool("lahf", cx & (1u << 0));
    kbool("avx", c1 & (1u << 28));
    kbool("fma", c1 & (1u << 12));
    kbool("f16c", c1 & (1u << 29));
    kbool("movbe", c1 & (1u << 22));
    kbool("os_ymm", ymm);
    kbool("avx2", b7 & (1u << 5));
    kbool("bmi1", b7 & (1u << 3));
    kbool("bmi2", b7 & (1u << 8));
    kbool("lzcnt", cx & (1u << 5));
    close_('}');
}

#ifdef PROBE_VULKAN
static void ku64(const char *k, uint64_t v) {
    key(k);
    printf("%" PRIu64, v);
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static void version(const char *k, uint32_t v) {
    key(k);
    printf("[%u, %u, %u]", VK_API_VERSION_MAJOR(v), VK_API_VERSION_MINOR(v), VK_API_VERSION_PATCH(v));
}

#define CHAIN(cond, s)              \
    if (cond) {                     \
        *tail = &(s);               \
        tail = &(s).pNext;          \
    }

typedef struct {
    PFN_vkGetPhysicalDeviceProperties props;
    PFN_vkGetPhysicalDeviceProperties2 props2;
    PFN_vkGetPhysicalDeviceFeatures2 features2;
    PFN_vkGetPhysicalDeviceMemoryProperties2 memory2;
    PFN_vkEnumerateDeviceExtensionProperties extensions;
#ifdef VK_KHR_cooperative_matrix
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR coopmat;   // NULL when the loader does not give it
#endif
} Fns;

#ifdef VK_KHR_cooperative_matrix
// Whether the device lists the cooperative matrix the translated tensor programs' native HMMA uses (oot2/vk_hmma.h):
// 16x8x16 on a subgroup, f16 A, B, C and result, without saturation. The runtime's start-up gate asks the same.
static int coop_hmma_shape(const Fns *fn, VkPhysicalDevice dev) {
    uint32_t n = 0;
    if (!fn->coopmat || fn->coopmat(dev, &n, NULL) != VK_SUCCESS || !n) return 0;
    VkCooperativeMatrixPropertiesKHR *p = calloc(n, sizeof *p);
    if (!p) return 0;
    for (uint32_t i = 0; i < n; i++) p[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    int shape = 0;
    if (fn->coopmat(dev, &n, p) == VK_SUCCESS)
        for (uint32_t i = 0; i < n; i++)
            shape |= p[i].MSize == 16 && p[i].NSize == 8 && p[i].KSize == 16 && p[i].AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                     p[i].BType == VK_COMPONENT_TYPE_FLOAT16_KHR && p[i].CType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                     p[i].ResultType == VK_COMPONENT_TYPE_FLOAT16_KHR && !p[i].saturatingAccumulation &&
                     p[i].scope == VK_SCOPE_SUBGROUP_KHR;
    free(p);
    return shape;
}
#endif

static void probe_properties(const Fns *fn, VkPhysicalDevice dev, const VkPhysicalDeviceProperties *p,
                             const VkExtensionProperties *ext, uint32_t ne) {
    int v12 = p->apiVersion >= VK_API_VERSION_1_2, v13 = p->apiVersion >= VK_API_VERSION_1_3;
    VkPhysicalDeviceVulkan11Properties p11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties p12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceProperties2 pp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    void **tail = &pp.pNext;
    CHAIN(v12, p11);
    CHAIN(v12, p12);
    CHAIN(v13, p13);
#ifdef VK_KHR_push_descriptor
    int push = has_ext(ext, ne, "VK_KHR_push_descriptor");
    VkPhysicalDevicePushDescriptorPropertiesKHR ppush = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
    CHAIN(push, ppush);
#endif
#ifdef VK_EXT_graphics_pipeline_library
    int gpl = has_ext(ext, ne, "VK_EXT_graphics_pipeline_library");
    VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT pgpl = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
    CHAIN(gpl, pgpl);
#endif
#ifdef VK_EXT_external_memory_host
    int host = has_ext(ext, ne, "VK_EXT_external_memory_host");
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT phost = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    CHAIN(host, phost);
#endif
#ifdef VK_EXT_mesh_shader
    int mesh = has_ext(ext, ne, "VK_EXT_mesh_shader");
    VkPhysicalDeviceMeshShaderPropertiesEXT pmesh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    CHAIN(mesh, pmesh);
#endif
    (void)tail;
    fn->props2(dev, &pp);
    key("properties");
    open_('{');
    ku64("max_memory_allocation_count", p->limits.maxMemoryAllocationCount);
    ku64("max_sampler_allocation_count", p->limits.maxSamplerAllocationCount);
    if (v12) {
        key("driver_name");
        json_string(p12.driverName);
        key("driver_info");
        json_string(p12.driverInfo);
        ku64("subgroup_size", p11.subgroupSize);
        ku64("max_update_after_bind_sampled_images", p12.maxDescriptorSetUpdateAfterBindSampledImages);
    }
    if (v13) {
        ku64("min_subgroup_size", p13.minSubgroupSize);
        ku64("max_subgroup_size", p13.maxSubgroupSize);
    }
#ifdef VK_KHR_push_descriptor
    if (push) ku64("max_push_descriptors", ppush.maxPushDescriptors);
#endif
#ifdef VK_EXT_graphics_pipeline_library
    if (gpl) kbool("graphics_pipeline_library_fast_linking", pgpl.graphicsPipelineLibraryFastLinking);
#endif
#ifdef VK_EXT_external_memory_host
    if (host) ku64("min_imported_host_pointer_alignment", phost.minImportedHostPointerAlignment);
#endif
#ifdef VK_EXT_mesh_shader
    if (mesh) {
        ku64("max_mesh_output_vertices", pmesh.maxMeshOutputVertices);
        ku64("max_mesh_output_primitives", pmesh.maxMeshOutputPrimitives);
        ku64("max_mesh_output_memory_size", pmesh.maxMeshOutputMemorySize);
    }
#endif
    close_('}');
}

static void probe_features(const Fns *fn, VkPhysicalDevice dev, const VkPhysicalDeviceProperties *p,
                           const VkExtensionProperties *ext, uint32_t ne) {
    int v12 = p->apiVersion >= VK_API_VERSION_1_2, v13 = p->apiVersion >= VK_API_VERSION_1_3;
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 ff = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void **tail = &ff.pNext;
    CHAIN(v12, f12);
    CHAIN(v13, f13);
#ifdef VK_EXT_mesh_shader
    int mesh = has_ext(ext, ne, "VK_EXT_mesh_shader");
    VkPhysicalDeviceMeshShaderFeaturesEXT fmesh = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    CHAIN(mesh, fmesh);
#endif
#ifdef VK_EXT_graphics_pipeline_library
    int gpl = has_ext(ext, ne, "VK_EXT_graphics_pipeline_library");
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    CHAIN(gpl, fgpl);
#endif
#ifdef VK_EXT_shader_object
    int so = has_ext(ext, ne, "VK_EXT_shader_object");
    VkPhysicalDeviceShaderObjectFeaturesEXT fso = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
    CHAIN(so, fso);
#endif
#ifdef VK_KHR_pipeline_binary
    int pb = has_ext(ext, ne, "VK_KHR_pipeline_binary");
    VkPhysicalDevicePipelineBinaryFeaturesKHR fpb = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR};
    CHAIN(pb, fpb);
#endif
#ifdef VK_EXT_vertex_input_dynamic_state
    int vi = has_ext(ext, ne, "VK_EXT_vertex_input_dynamic_state");
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT fvi = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT};
    CHAIN(vi, fvi);
#endif
#ifdef VK_KHR_cooperative_matrix
    int cm = has_ext(ext, ne, "VK_KHR_cooperative_matrix");
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    CHAIN(cm, fcm);
#endif
#ifdef VK_EXT_device_fault
    int fault = has_ext(ext, ne, "VK_EXT_device_fault");
    VkPhysicalDeviceFaultFeaturesEXT ffault = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    CHAIN(fault, ffault);
#endif
    (void)tail;
    fn->features2(dev, &ff);
    key("features");
    open_('{');
    kbool("texture_compression_bc", ff.features.textureCompressionBC);
    kbool("texture_compression_astc_ldr", ff.features.textureCompressionASTC_LDR);
    kbool("sparse_binding", ff.features.sparseBinding);
    kbool("sparse_residency_buffer", ff.features.sparseResidencyBuffer);
    kbool("shader_int64", ff.features.shaderInt64);
    if (v12) {
        kbool("buffer_device_address", f12.bufferDeviceAddress);
        kbool("descriptor_indexing", f12.descriptorIndexing);
        kbool("sampled_image_update_after_bind", f12.descriptorBindingSampledImageUpdateAfterBind);
        kbool("runtime_descriptor_array", f12.runtimeDescriptorArray);
        kbool("timeline_semaphore", f12.timelineSemaphore);
        kbool("vulkan_memory_model", f12.vulkanMemoryModel);
        kbool("vulkan_memory_model_device_scope", f12.vulkanMemoryModelDeviceScope);
        kbool("shader_float16", f12.shaderFloat16);   // the native HMMA's 16-bit floats (oot2/vk_hmma.h)
    }
    if (v13) {
        kbool("dynamic_rendering", f13.dynamicRendering);
        kbool("synchronization2", f13.synchronization2);
        kbool("maintenance4", f13.maintenance4);
        kbool("subgroup_size_control", f13.subgroupSizeControl);
        kbool("compute_full_subgroups", f13.computeFullSubgroups);
    }
#ifdef VK_EXT_mesh_shader
    if (mesh) {
        kbool("task_shader", fmesh.taskShader);
        kbool("mesh_shader", fmesh.meshShader);
    }
#endif
#ifdef VK_EXT_graphics_pipeline_library
    if (gpl) kbool("graphics_pipeline_library", fgpl.graphicsPipelineLibrary);
#endif
#ifdef VK_EXT_shader_object
    if (so) kbool("shader_object", fso.shaderObject);
#endif
#ifdef VK_KHR_pipeline_binary
    if (pb) kbool("pipeline_binaries", fpb.pipelineBinaries);
#endif
#ifdef VK_EXT_vertex_input_dynamic_state
    if (vi) kbool("vertex_input_dynamic_state", fvi.vertexInputDynamicState);
#endif
#ifdef VK_KHR_cooperative_matrix
    if (cm) {
        kbool("cooperative_matrix", fcm.cooperativeMatrix);
        kbool("cooperative_matrix_16x8x16_f16", coop_hmma_shape(fn, dev));
    }
#endif
#ifdef VK_EXT_device_fault
    if (fault) kbool("device_fault", ffault.deviceFault);
#endif
    close_('}');
}

static void probe_memory(const Fns *fn, VkPhysicalDevice dev, const VkExtensionProperties *ext, uint32_t ne) {
    VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    int budget = 0;
    void **tail = &mp.pNext;
#ifdef VK_EXT_memory_budget
    budget = has_ext(ext, ne, "VK_EXT_memory_budget");
    VkPhysicalDeviceMemoryBudgetPropertiesEXT mb = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    CHAIN(budget, mb);
#endif
    (void)tail;
    fn->memory2(dev, &mp);
    key("heaps");
    open_('[');
    for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; i++) {
        const VkMemoryHeap *h = &mp.memoryProperties.memoryHeaps[i];
        sep();
        open_('{');
        ku64("size", h->size);
        kbool("device_local", (h->flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0);
#ifdef VK_EXT_memory_budget
        if (budget) ku64("budget", mb.heapBudget[i]);
#endif
        close_('}');
    }
    close_(']');
    (void)ext;
    (void)ne;
}

// Whether the device has each item of vk_required.h, by the header's query and tests: every item false when the
// instance lacks the Vulkan 1.1 query functions.
static void probe_items(const Fns *fn, VkPhysicalDevice dev, const VkExtensionProperties *ext, uint32_t ne) {
    unsigned char has[OOT2_VK_ITEM_COUNT] = {0};
    if (fn->props2 && fn->features2) {
        Oot2VkFacts f;
        oot2_vk_query(fn->props2, fn->features2, dev, ext, ne, &f);
        oot2_vk_judge(&f, has);
    }
    key("items");
    open_('{');
    for (int i = 0; i < OOT2_VK_ITEM_COUNT; i++) kbool(oot2_vk_item_id(i), has[i]);
    close_('}');
}

static void probe_device(const Fns *fn, VkPhysicalDevice dev) {
    VkPhysicalDeviceProperties p;
    fn->props(dev, &p);
    uint32_t ne = 0;
    VkExtensionProperties *ext = NULL;
    if (fn->extensions(dev, NULL, &ne, NULL) == VK_SUCCESS && ne) {
        ext = calloc(ne, sizeof *ext);
        if (!ext || fn->extensions(dev, NULL, &ne, ext) < 0) ne = 0;
    }
    open_('{');
    key("name");
    json_string(p.deviceName);
    version("api", p.apiVersion);
    ku64("driver_version", p.driverVersion);
    ku64("vendor_id", p.vendorID);
    ku64("device_id", p.deviceID);
    {   // the UUID a pipeline cache is valid for, with the vendor, device and driver (prepare's pipeline step keys on it)
        char uuid[2 * VK_UUID_SIZE + 1];
        for (unsigned i = 0; i < VK_UUID_SIZE; i++) sprintf(uuid + 2 * i, "%02x", p.pipelineCacheUUID[i]);
        key("pipeline_cache_uuid");
        json_string(uuid);
    }
    ku64("type", (uint64_t)p.deviceType);
    if (fn->props2 && fn->features2 && fn->memory2) {
        probe_properties(fn, dev, &p, ext, ne);
        probe_features(fn, dev, &p, ext, ne);
        probe_memory(fn, dev, ext, ne);
    }
    probe_items(fn, dev, ext, ne);
    key("extensions");
    open_('[');
    for (uint32_t i = 0; i < ne; i++) {
        sep();
        json_string(ext[i].extensionName);
    }
    close_(']');
    close_('}');
    free(ext);
}

static void probe_vulkan(void) {
    PFN_vkGetInstanceProcAddr gipa = NULL;
    void *lib = open_lib();
    if (lib) gipa = (PFN_vkGetInstanceProcAddr)lib_sym(lib, "vkGetInstanceProcAddr");
    open_('{');
    kbool("loader", gipa != NULL);
    if (!gipa) {
        close_('}');
        return;
    }
    PFN_vkEnumerateInstanceVersion eiv = (PFN_vkEnumerateInstanceVersion)gipa(NULL, "vkEnumerateInstanceVersion");
    uint32_t iv = VK_API_VERSION_1_0;
    if (eiv) eiv(&iv);
    version("instance_api", iv);
    PFN_vkCreateInstance create = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    // A 1.0 loader refuses any other apiVersion; later ones accept 1.3 whatever the devices support.
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "oot2-check",
                             .apiVersion = iv >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_3 : VK_API_VERSION_1_0};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = create ? create(&ci, NULL, &inst) : VK_ERROR_INITIALIZATION_FAILED;
    key("instance");
    printf("%d", (int)r);
    if (r != VK_SUCCESS) {
        close_('}');
        return;
    }
    PFN_vkEnumeratePhysicalDevices enumerate =
        (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkDestroyInstance destroy = (PFN_vkDestroyInstance)gipa(inst, "vkDestroyInstance");
    Fns fn = {
        .props = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties"),
        .props2 = (PFN_vkGetPhysicalDeviceProperties2)gipa(inst, "vkGetPhysicalDeviceProperties2"),
        .features2 = (PFN_vkGetPhysicalDeviceFeatures2)gipa(inst, "vkGetPhysicalDeviceFeatures2"),
        .memory2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)gipa(inst, "vkGetPhysicalDeviceMemoryProperties2"),
        .extensions = (PFN_vkEnumerateDeviceExtensionProperties)gipa(inst, "vkEnumerateDeviceExtensionProperties"),
#ifdef VK_KHR_cooperative_matrix
        .coopmat = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)gipa(
            inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"),
#endif
    };
    VkPhysicalDevice devs[16];
    uint32_t n = 16;
    if (!enumerate || enumerate(inst, &n, devs) < 0) n = 0;
    key("devices");
    open_('[');
    for (uint32_t i = 0; i < n; i++) {
        sep();
        probe_device(&fn, devs[i]);
    }
    close_(']');
    if (destroy) destroy(inst, NULL);
    close_('}');
}
#endif

// The list of vk_required.h, which needs no Vulkan header: [[id, what], ...].
static void print_items(void) {
    key("vulkan_items");
    open_('[');
    for (int i = 0; i < OOT2_VK_ITEM_COUNT; i++) {
        sep();
        open_('[');
        sep();
        json_string(oot2_vk_item_id(i));
        sep();
        json_string(oot2_vk_item_what(i));
        close_(']');
    }
    close_(']');
}

int main(void) {
    open_('{');
    probe_cpu();
    print_items();
    key("vulkan");
#ifdef PROBE_VULKAN
    probe_vulkan();
#else
    printf("null");
#endif
    close_('}');
    putchar('\n');
    return 0;
}
