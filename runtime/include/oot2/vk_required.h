// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Vulkan device items the renderer requires (architecture section 1), as one list. The check step of
// `oot2 prepare` (tools/prepare/hostcheck.c, read by tools/prepare/check.py) and the runtime (runtime/gpu, and
// `oot2 --vulkan-items`) both include this header, so the check and the device the Vulkan back end creates cannot
// disagree about what a device must have. The list starts with the 19 items of section 1; the back end's later
// work extends it here, and nowhere else.
//
// OOT2_VK_ITEMS(X) expands X(id, what, test, enable) once per item, in this order:
//   id      a C identifier: the item's name in the probe's JSON and in `oot2 --vulkan-items`
//   what    the phrase the check prints for a device that lacks the item
//   test    a C expression over `const Oot2VkFacts *f` (below), true when the device has the item
//   enable  a C expression over `const Oot2VkFacts *f` and `Oot2VkFacts *e`: it sets in e the features the device must
//           be created with for the item ((void)0 when the item is a limit or an extension alone); the device the
//           runtime creates enables these and the optional features of OOT2_VK_OPTIONAL (below) and nothing more,
//           every extension of OOT2_VK_EXTENSIONS the device lists aside
// A use that only names the items (an X that ignores test, as oot2_vk_item_id does) needs no Vulkan header: the
// preprocessor never expands an argument that the replacement does not use. The facts, the query that fills them and
// the judgement are defined only after <vulkan/vulkan_core.h> (or Volk's volk.h) has been included.
//
// A test reads only structures the query chained, and the query chains an extension's structure only when the
// device lists the extension, so an item whose extension is missing reads zero and fails. A test that compares a
// limit with "at most" checks the extension itself (host_memory_import), since zero would pass it.
//
// Items 20 to 27 came with the Vulkan back end's device: what the device itself, the uni layout, the mesh path and the
// translated programs use (architecture D12, D16, D18; the mesh shader, first draw, pipeline cost and translator defects studies).
//
// Items 28 to 30 came with the pipeline builder's capability check (oot2/vk_caps.h): the translated modules
// declare 64-bit integers (the raw-access helper), storage images without a format and cube array images, and the
// check over the whole corpus found no item that enables them. All three are features every desktop GPU that has the
// items above offers. The check found a fourth, non-uniform indexing of storage image arrays, which only the
// translator's own declaration asked for (no module indexes an array of storage images); the translator declares
// it only where one is indexed, so no item asks for it. It is the one optional feature, enabled when the GPU reports
// it: shader packs prepared before that change still declare the capability in about 9,500 modules, and a device that
// lacks the feature refuses such a module by name.
//
// Items 33 to 41 are the core features the replay's draw state needs (the core features requirement): one viewport and scissor for each of
// the 16 NVN has, line and point fill, depth clamp, the depth bias clamp, the depth bounds test, wide lines, a blend
// state and write mask for each colour target, dual-source blend factors and indirect draws with a first instance.
// Every desktop GPU that has the items above has them. robustBufferAccess is not among them and stays off: a raw
// device address is not covered by it, and a read the game guards past a declared range is data, not zero.
//
// Item 42 is 8-bit indices (VK_KHR_index_type_uint8): the title draws with NVN's unsigned byte index type, which Vulkan
// 1.3 takes only through this extension; primitive restart at 0xFF comes with it.
//
// Item 43 is the Vulkan memory model with device scope: SPV_KHR_cooperative_matrix requires it of a module that declares a
// cooperative matrix, which the translated tensor programs' native HMMA does (oot2/vk_hmma.h). Vulkan 1.3 makes both
// features required of every device (the specification's appendix on versions), so the first item already implies it.
// The native HMMA also needs 16-bit floats, which no item asks for: shader_float16 is an optional feature, and a device
// without it runs the helper the modules keep (vk_hmma.c's gate).
#pragma once
#include <stdint.h>
#include <string.h>

#define OOT2_VK_ITEMS(X)                                                                                               \
    X(vulkan_1_3, "Vulkan 1.3", f->props.apiVersion >= VK_API_VERSION_1_3, (void)0)                                   \
    X(bc_textures, "BC texture compression", f->features.textureCompressionBC,                                        \
      (e->features.textureCompressionBC = VK_TRUE))                                                                   \
    X(mesh_shaders, "mesh and task shaders", f->mesh.meshShader && f->mesh.taskShader,                                \
      (e->mesh.meshShader = e->mesh.taskShader = VK_TRUE))                                                            \
    X(mesh_output, "mesh output of 160 vertices, 128 primitives and 28,160 bytes",                                    \
      f->mesh_props.maxMeshOutputVertices >= 160 && f->mesh_props.maxMeshOutputPrimitives >= 128 &&                  \
          f->mesh_props.maxMeshOutputMemorySize >= 28160,                                                             \
      (void)0)                                                                                                        \
    X(pipeline_libraries, "pipeline libraries with fast linking",                                                    \
      f->gpl.graphicsPipelineLibrary && f->gpl_props.graphicsPipelineLibraryFastLinking,                              \
      (e->gpl.graphicsPipelineLibrary = VK_TRUE))                                                                     \
    X(shader_objects, "shader objects", f->shader_object.shaderObject, (e->shader_object.shaderObject = VK_TRUE))     \
    X(pipeline_binaries, "pipeline binaries", f->pipeline_binary.pipelineBinaries,                                    \
      (e->pipeline_binary.pipelineBinaries = VK_TRUE))                                                                \
    X(dynamic_state_3, "extended dynamic state 3", f->ext[OOT2_VK_EXT_extended_dynamic_state3], (e->eds3 = f->eds3)) \
    X(vertex_input_dynamic_state, "vertex input dynamic state", f->vertex_input.vertexInputDynamicState,              \
      (e->vertex_input.vertexInputDynamicState = VK_TRUE))                                                            \
    X(buffer_device_address, "buffer device address", f->f12.bufferDeviceAddress,                                     \
      (e->f12.bufferDeviceAddress = VK_TRUE))                                                                         \
    X(host_memory_import, "host memory import at 4 KiB alignment",                                                   \
      f->ext[OOT2_VK_EXT_external_memory_host] && f->host.minImportedHostPointerAlignment <= 4096, (void)0)           \
    X(sparse_buffers, "sparse buffers", f->features.sparseBinding && f->features.sparseResidencyBuffer,               \
      (e->features.sparseBinding = e->features.sparseResidencyBuffer = VK_TRUE))                                      \
    X(update_after_bind_images, "1,048,576 update-after-bind sampled images",                                         \
      f->f12.descriptorBindingSampledImageUpdateAfterBind &&                                                          \
          f->p12.maxDescriptorSetUpdateAfterBindSampledImages >= (1u << 20),                                          \
      (e->f12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE))                                                \
    X(push_descriptors, "32 push descriptors", f->push.maxPushDescriptors >= 32, (void)0)                             \
    X(subgroup_32, "subgroup size fixed at 32",                                                                       \
      f->p11.subgroupSize == 32 && f->p13.minSubgroupSize == 32 && f->p13.maxSubgroupSize == 32, (void)0)             \
    X(device_fault, "device fault reports", f->fault.deviceFault,                                                     \
      (e->fault.deviceFault = VK_TRUE, e->fault.deviceFaultVendorBinary = f->fault.deviceFaultVendorBinary))          \
    X(diagnostic_checkpoints, "diagnostic checkpoints", f->ext[OOT2_VK_EXT_diagnostic_checkpoints], (void)0)          \
    X(cooperative_matrix, "cooperative matrix", f->coop.cooperativeMatrix, (e->coop.cooperativeMatrix = VK_TRUE))     \
    X(memory_budget, "memory budget", f->ext[OOT2_VK_EXT_memory_budget], (void)0)                                     \
    X(timeline_semaphores, "timeline semaphores", f->f12.timelineSemaphore, (e->f12.timelineSemaphore = VK_TRUE))    \
    X(dynamic_rendering_sync2, "dynamic rendering and synchronization2",                                              \
      f->f13.dynamicRendering && f->f13.synchronization2,                                                             \
      (e->f13.dynamicRendering = e->f13.synchronization2 = VK_TRUE))                                                  \
    X(bindless_arrays, "runtime sampled-image arrays, partially bound, updated while pending, indexed non-uniformly", \
      f->f12.runtimeDescriptorArray && f->f12.descriptorBindingPartiallyBound &&                                      \
          f->f12.descriptorBindingUpdateUnusedWhilePending && f->f12.shaderSampledImageArrayNonUniformIndexing,       \
      (e->f12.runtimeDescriptorArray = e->f12.descriptorBindingPartiallyBound =                                       \
           e->f12.descriptorBindingUpdateUnusedWhilePending = e->f12.shaderSampledImageArrayNonUniformIndexing =      \
               VK_TRUE))                                                                                              \
    X(mesh_subgroup_size, "subgroup size 32 required of task and mesh stages, with full subgroups",                  \
      f->f13.subgroupSizeControl && f->f13.computeFullSubgroups &&                                                    \
          (f->p13.requiredSubgroupSizeStages & OOT2_VK_TASK_MESH) == OOT2_VK_TASK_MESH,                               \
      (e->f13.subgroupSizeControl = e->f13.computeFullSubgroups = VK_TRUE))                                           \
    X(float32_atomic_add, "32-bit float atomic add on storage buffers",                                               \
      f->atomic_float.shaderBufferFloat32AtomicAdd, (e->atomic_float.shaderBufferFloat32AtomicAdd = VK_TRUE))        \
    X(shader_features, "float64, clip and cull distances, extended gather and draw parameters in shaders",            \
      f->features.shaderFloat64 && f->features.shaderClipDistance && f->features.shaderCullDistance &&                \
          f->features.shaderImageGatherExtended && f->f11.shaderDrawParameters,                                       \
      (e->features.shaderFloat64 = e->features.shaderClipDistance = e->features.shaderCullDistance =                  \
           e->features.shaderImageGatherExtended = e->f11.shaderDrawParameters = VK_TRUE))                            \
    X(subgroup_operations, "subgroup basic, vote, ballot and shuffle in graphics, compute, task and mesh stages",      \
      (f->p11.subgroupSupportedOperations & OOT2_VK_SUBGROUP_OPS) == OOT2_VK_SUBGROUP_OPS &&                         \
          (f->p11.subgroupSupportedStages & OOT2_VK_SUBGROUP_STAGES) == OOT2_VK_SUBGROUP_STAGES,                      \
      (void)0)                                                                                                        \
    X(stores_and_atomics, "stores and atomics in vertex and fragment stages",                                         \
      f->features.vertexPipelineStoresAndAtomics && f->features.fragmentStoresAndAtomics,                             \
      (e->features.vertexPipelineStoresAndAtomics = e->features.fragmentStoresAndAtomics = VK_TRUE))                  \
    X(shader_int64, "64-bit integers in shaders", f->features.shaderInt64, (e->features.shaderInt64 = VK_TRUE))       \
    X(storage_image_formats, "storage images read and written without a format",                                      \
      f->features.shaderStorageImageReadWithoutFormat && f->features.shaderStorageImageWriteWithoutFormat,           \
      (e->features.shaderStorageImageReadWithoutFormat = e->features.shaderStorageImageWriteWithoutFormat = VK_TRUE)) \
    X(cube_array_images, "cube array images", f->features.imageCubeArray, (e->features.imageCubeArray = VK_TRUE))     \
    X(sampler_features, "anisotropic filtering, mirror clamp to edge and min and max reduction in samplers",          \
      f->features.samplerAnisotropy && f->f12.samplerMirrorClampToEdge && f->f12.samplerFilterMinmax,                \
      (e->features.samplerAnisotropy = VK_TRUE, e->f12.samplerMirrorClampToEdge = VK_TRUE,                            \
       e->f12.samplerFilterMinmax = VK_TRUE))                                                                         \
    X(depth_clip_control, "depth clip control, set per draw (NVN's -1 to 1 depth)",                                  \
      f->depth_clip.depthClipControl && f->eds3.extendedDynamicState3DepthClipNegativeOneToOne,                       \
      (e->depth_clip.depthClipControl = VK_TRUE))                                                                      \
    X(multi_viewport, "16 viewports and scissors",                                                                     \
      f->features.multiViewport && f->props.limits.maxViewports >= 16,                                                 \
      (e->features.multiViewport = VK_TRUE))                                                                           \
    X(fill_mode_non_solid, "line and point polygon modes, set per draw",                                               \
      f->features.fillModeNonSolid && f->eds3.extendedDynamicState3PolygonMode,                                        \
      (e->features.fillModeNonSolid = VK_TRUE))                                                                        \
    X(depth_clamp, "depth clamp, set per draw",                                                                        \
      f->features.depthClamp && f->eds3.extendedDynamicState3DepthClampEnable,                                         \
      (e->features.depthClamp = VK_TRUE))                                                                              \
    X(depth_bias_clamp, "depth bias clamp",                                                                            \
      f->features.depthBiasClamp,                                                                                      \
      (e->features.depthBiasClamp = VK_TRUE))                                                                          \
    X(depth_bounds, "the depth bounds test",                                                                           \
      f->features.depthBounds,                                                                                         \
      (e->features.depthBounds = VK_TRUE))                                                                             \
    X(wide_lines, "wide lines",                                                                                        \
      f->features.wideLines,                                                                                           \
      (e->features.wideLines = VK_TRUE))                                                                               \
    X(independent_blend, "blend state and write masks set per colour target",                                          \
      f->features.independentBlend && f->eds3.extendedDynamicState3ColorBlendEnable &&                                 \
          f->eds3.extendedDynamicState3ColorBlendEquation && f->eds3.extendedDynamicState3ColorWriteMask,              \
      (e->features.independentBlend = VK_TRUE))                                                                        \
    X(dual_source_blend, "dual-source blending",                                                                       \
      f->features.dualSrcBlend,                                                                                        \
      (e->features.dualSrcBlend = VK_TRUE))                                                                            \
    X(draw_indirect_first_instance, "indirect draws with a first instance",                                            \
      f->features.drawIndirectFirstInstance,                                                                           \
      (e->features.drawIndirectFirstInstance = VK_TRUE))                                                                   X(index_type_uint8, "8-bit indices", f->index_u8.indexTypeUint8, (e->index_u8.indexTypeUint8 = VK_TRUE)) \
    X(vulkan_memory_model, "the Vulkan memory model with device scope",                                               \
      f->f12.vulkanMemoryModel && f->f12.vulkanMemoryModelDeviceScope,                                                \
      (e->f12.vulkanMemoryModel = e->f12.vulkanMemoryModelDeviceScope = VK_TRUE))

// The features the device enables when the GPU reports them and does not require: OOT2_VK_OPTIONAL(X) expands
// X(id, enable) once per feature. A device without one is still chosen, and none is an item (the check,
// `oot2 --vulkan-items` and the probe list the items only). enable is an item's enable: a C expression over f and e,
// and it copies the feature from f, so the feature stays off where the GPU lacks it. The capability check of the
// pipeline builder (oot2/vk_caps.h) reads the structure oot2_vk_enable fills, so a capability behind an optional
// feature is allowed exactly when the feature was enabled. zero_initialize_workgroup_memory is a Vulkan 1.3 core
// feature, so every device the items choose has it; it stays optional because no capability depends on it: the pipeline
// builder zero-initializes a compute module's workgroup memory (spv_zero_workgroup) when it was enabled and leaves the
// module as translated when it was not. shader_float16 is the 16-bit floats of the translated modules' native HMMA
// (oot2/vk_hmma.h): with it off, the start-up gate keeps the helper and the pipeline builder takes the native function and
// its Float16 declaration out of every module (spv_hmma_fallback). image_2d_view_of_3d (VK_EXT_image_2d_view_of_3d) lets a
// storage image be a 2D view of one slice of a 3D image (oot2/vk_view_plan.h): the 3D images are made 2D-view compatible
// where it is on, and a storage view of a slice on a device without it is a named stop at the view (the render targets
// over a 3D image need no extension, they use the core 2D-array compatibility).
#define OOT2_VK_OPTIONAL(X)                                                                                            \
    X(storage_image_array_indexing,                                                                                    \
      (e->f12.shaderStorageImageArrayNonUniformIndexing = f->f12.shaderStorageImageArrayNonUniformIndexing))          \
    X(zero_initialize_workgroup_memory,                                                                                \
      (e->f13.shaderZeroInitializeWorkgroupMemory = f->f13.shaderZeroInitializeWorkgroupMemory))                      \
    X(shader_float16, (e->f12.shaderFloat16 = f->f12.shaderFloat16))                                                   \
    X(image_2d_view_of_3d, (e->view2d.image2DViewOf3D = f->view2d.image2DViewOf3D))

// The device extensions the items read, E(id, name), and the ones they depend on below Vulkan 1.4 (pipeline_library
// for graphics_pipeline_library, maintenance5 for pipeline_binary). The device the back end creates enables every one
// the device lists.
#define OOT2_VK_EXTENSIONS(E)                                                                                          \
    E(push_descriptor, "VK_KHR_push_descriptor")                                                                      \
    E(external_memory_host, "VK_EXT_external_memory_host")                                                            \
    E(graphics_pipeline_library, "VK_EXT_graphics_pipeline_library")                                                  \
    E(mesh_shader, "VK_EXT_mesh_shader")                                                                              \
    E(shader_object, "VK_EXT_shader_object")                                                                          \
    E(pipeline_binary, "VK_KHR_pipeline_binary")                                                                      \
    E(extended_dynamic_state3, "VK_EXT_extended_dynamic_state3")                                                      \
    E(vertex_input_dynamic_state, "VK_EXT_vertex_input_dynamic_state")                                                \
    E(cooperative_matrix, "VK_KHR_cooperative_matrix")                                                                \
    E(device_fault, "VK_EXT_device_fault")                                                                            \
    E(diagnostic_checkpoints, "VK_NV_device_diagnostic_checkpoints")                                                  \
    E(memory_budget, "VK_EXT_memory_budget")                                                                          \
    E(shader_atomic_float, "VK_EXT_shader_atomic_float")                                                              \
    E(depth_clip_control, "VK_EXT_depth_clip_control")                                                                \
    E(pipeline_library, "VK_KHR_pipeline_library")                                                                    \
    E(maintenance5, "VK_KHR_maintenance5")                                                                             \
    E(image_2d_view_of_3d, "VK_EXT_image_2d_view_of_3d")                                                               \
    E(index_type_uint8, "VK_KHR_index_type_uint8")

enum {
#define OOT2_VK_ITEM_ENUM(id, what, test, enable) OOT2_VK_ITEM_##id,
    OOT2_VK_ITEMS(OOT2_VK_ITEM_ENUM)
#undef OOT2_VK_ITEM_ENUM
    OOT2_VK_ITEM_COUNT
};

enum {
#define OOT2_VK_OPTIONAL_ENUM(id, enable) OOT2_VK_OPTIONAL_##id,
    OOT2_VK_OPTIONAL(OOT2_VK_OPTIONAL_ENUM)
#undef OOT2_VK_OPTIONAL_ENUM
    OOT2_VK_OPTIONAL_COUNT
};

enum {
#define OOT2_VK_EXT_ENUM(id, name) OOT2_VK_EXT_##id,
    OOT2_VK_EXTENSIONS(OOT2_VK_EXT_ENUM)
#undef OOT2_VK_EXT_ENUM
    OOT2_VK_EXT_COUNT
};

// Item i's id and phrase, in list order; NULL past the end.
static inline const char *oot2_vk_item_id(int i) {
    static const char *const ids[] = {
#define OOT2_VK_ITEM_ID(id, what, test, enable) #id,
        OOT2_VK_ITEMS(OOT2_VK_ITEM_ID)
#undef OOT2_VK_ITEM_ID
    };
    return i >= 0 && i < OOT2_VK_ITEM_COUNT ? ids[i] : NULL;
}

static inline const char *oot2_vk_item_what(int i) {
    static const char *const whats[] = {
#define OOT2_VK_ITEM_WHAT(id, what, test, enable) what,
        OOT2_VK_ITEMS(OOT2_VK_ITEM_WHAT)
#undef OOT2_VK_ITEM_WHAT
    };
    return i >= 0 && i < OOT2_VK_ITEM_COUNT ? whats[i] : NULL;
}

// Optional feature i's id, in list order; NULL past the end.
static inline const char *oot2_vk_optional_id(int i) {
    static const char *const ids[] = {
#define OOT2_VK_OPTIONAL_ID(id, enable) #id,
        OOT2_VK_OPTIONAL(OOT2_VK_OPTIONAL_ID)
#undef OOT2_VK_OPTIONAL_ID
    };
    return i >= 0 && i < OOT2_VK_OPTIONAL_COUNT ? ids[i] : NULL;
}

// Extension e's name (OOT2_VK_EXT_*), as the device lists it; NULL past the end.
static inline const char *oot2_vk_extension_name(int e) {
    static const char *const names[] = {
#define OOT2_VK_EXT_NAME(id, name) name,
        OOT2_VK_EXTENSIONS(OOT2_VK_EXT_NAME)
#undef OOT2_VK_EXT_NAME
    };
    return e >= 0 && e < OOT2_VK_EXT_COUNT ? names[e] : NULL;
}

#ifdef VK_HEADER_VERSION
#if !defined(VK_EXT_mesh_shader) || !defined(VK_EXT_shader_object) || !defined(VK_KHR_pipeline_binary) ||            \
    !defined(VK_KHR_cooperative_matrix) || !defined(VK_EXT_device_fault) || !defined(VK_EXT_graphics_pipeline_library)
#error "these Vulkan headers predate an extension the renderer requires: use the Vulkan SDK 1.4.357.0 or later"
#endif

// The stages and operations the subgroup items name.
#define OOT2_VK_TASK_MESH (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)
#define OOT2_VK_SUBGROUP_OPS                                                                                           \
    (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |                   \
     VK_SUBGROUP_FEATURE_SHUFFLE_BIT)
#define OOT2_VK_SUBGROUP_STAGES (VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT | OOT2_VK_TASK_MESH)

// What the items' tests read: one device's properties, features and extensions. The same structure, filled by
// oot2_vk_enable, holds the features a device is created with.
typedef struct Oot2VkFacts {
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceVulkan11Properties p11;
    VkPhysicalDeviceVulkan12Properties p12;
    VkPhysicalDeviceVulkan13Properties p13;
    VkPhysicalDeviceVulkan11Features f11;
    VkPhysicalDeviceVulkan12Features f12;
    VkPhysicalDeviceVulkan13Features f13;
    VkPhysicalDevicePushDescriptorPropertiesKHR push;
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host;
    VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gpl_props;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceMeshShaderPropertiesEXT mesh_props;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
    VkPhysicalDeviceShaderObjectFeaturesEXT shader_object;
    VkPhysicalDevicePipelineBinaryFeaturesKHR pipeline_binary;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vertex_input;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop;
    VkPhysicalDeviceFaultFeaturesEXT fault;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float;
    VkPhysicalDeviceDepthClipControlFeaturesEXT depth_clip;
    VkPhysicalDeviceIndexTypeUint8FeaturesKHR index_u8;
    VkPhysicalDeviceImage2DViewOf3DFeaturesEXT view2d;
    unsigned char ext[OOT2_VK_EXT_COUNT];   // 1 for each extension of OOT2_VK_EXTENSIONS the device lists
} Oot2VkFacts;

#define OOT2_VK_LINK(cond, s, type)                                                                                    \
    if (cond) {                                                                                                        \
        (s).sType = (type);                                                                                            \
        (s).pNext = NULL;                                                                                              \
        *tail = &(s);                                                                                                  \
        tail = &(s).pNext;                                                                                             \
    }

// Chain f's feature structures behind ff, as f's API version and extensions allow: the 1.1 and 1.2 structures for a 1.2
// device and the 1.3 one for a 1.3 device, as the specification allows, an extension's structure only when f->ext
// lists the extension. The query reads features through this chain and the device is created with it.
static inline void oot2_vk_chain_features(Oot2VkFacts *f, VkPhysicalDeviceFeatures2 *ff) {
    int v12 = f->props.apiVersion >= VK_API_VERSION_1_2, v13 = f->props.apiVersion >= VK_API_VERSION_1_3;
    ff->sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    ff->pNext = NULL;
    void **tail = &ff->pNext;
    OOT2_VK_LINK(v12, f->f11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
    OOT2_VK_LINK(v12, f->f12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
    OOT2_VK_LINK(v13, f->f13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_graphics_pipeline_library], f->gpl,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_mesh_shader], f->mesh, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_shader_object], f->shader_object,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_pipeline_binary], f->pipeline_binary,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_extended_dynamic_state3], f->eds3,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_vertex_input_dynamic_state], f->vertex_input,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_cooperative_matrix], f->coop,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_device_fault], f->fault, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_shader_atomic_float], f->atomic_float,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_depth_clip_control], f->depth_clip,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_index_type_uint8], f->index_u8,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_image_2d_view_of_3d], f->view2d,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT)
    (void)tail;
}

// Fill f for device dev, whose extensions (vkEnumerateDeviceExtensionProperties) are ext[0..n). props2 and features2
// are the instance's vkGetPhysicalDeviceProperties2 and vkGetPhysicalDeviceFeatures2 (Vulkan 1.1 instance functions).
// The 1.1 and 1.2 structures are chained only for a 1.2 device and the 1.3 ones for a 1.3 device, as the
// specification allows; an extension's structure only when the device lists the extension.
static inline void oot2_vk_query(PFN_vkGetPhysicalDeviceProperties2 props2, PFN_vkGetPhysicalDeviceFeatures2 features2,
                                 VkPhysicalDevice dev, const VkExtensionProperties *ext, uint32_t n, Oot2VkFacts *f) {
    memset(f, 0, sizeof *f);
    for (uint32_t i = 0; i < n; i++)
        for (int e = 0; e < OOT2_VK_EXT_COUNT; e++)
            if (!strcmp(ext[i].extensionName, oot2_vk_extension_name(e))) f->ext[e] = 1;

    VkPhysicalDeviceProperties2 pp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2(dev, &pp);
    int v12 = pp.properties.apiVersion >= VK_API_VERSION_1_2, v13 = pp.properties.apiVersion >= VK_API_VERSION_1_3;
    void **tail = &pp.pNext;
    OOT2_VK_LINK(v12, f->p11, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES)
    OOT2_VK_LINK(v12, f->p12, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES)
    OOT2_VK_LINK(v13, f->p13, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_push_descriptor], f->push,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_external_memory_host], f->host,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_graphics_pipeline_library], f->gpl_props,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT)
    OOT2_VK_LINK(f->ext[OOT2_VK_EXT_mesh_shader], f->mesh_props,
                 VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT)
    props2(dev, &pp);
    f->props = pp.properties;

    VkPhysicalDeviceFeatures2 ff;
    oot2_vk_chain_features(f, &ff);
    features2(dev, &ff);
    f->features = ff.features;
}

// has[i] = 1 when the device of f has item i, else 0. Returns how many items it has.
static inline int oot2_vk_judge(const Oot2VkFacts *f, unsigned char has[OOT2_VK_ITEM_COUNT]) {
    int i = 0, n = 0;
#define OOT2_VK_ITEM_JUDGE(id, what, test, enable)                                                                     \
    has[i] = (test) ? 1 : 0;                                                                                           \
    n += has[i++];
    OOT2_VK_ITEMS(OOT2_VK_ITEM_JUDGE)
#undef OOT2_VK_ITEM_JUDGE
    return n;
}

// e = the features a device with facts f is created with: every item's enable and every optional feature's, and nothing
// else. e keeps f's API
// version and extension list, so oot2_vk_chain_features(e, ...) chains the same structures the query read.
static inline void oot2_vk_enable(const Oot2VkFacts *f, Oot2VkFacts *e) {
    memset(e, 0, sizeof *e);
    e->props.apiVersion = f->props.apiVersion;
    memcpy(e->ext, f->ext, sizeof e->ext);
#define OOT2_VK_ITEM_ENABLE(id, what, test, enable) enable;
    OOT2_VK_ITEMS(OOT2_VK_ITEM_ENABLE)
#undef OOT2_VK_ITEM_ENABLE
#define OOT2_VK_OPTIONAL_ENABLE(id, enable) enable;
    OOT2_VK_OPTIONAL(OOT2_VK_OPTIONAL_ENABLE)
#undef OOT2_VK_OPTIONAL_ENABLE
}
#endif
