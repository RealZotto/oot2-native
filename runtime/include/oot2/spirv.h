// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The things the pipeline builder does to a translated SPIR-V module besides handing it to the driver
// (runtime/gpu/spirv.c, architecture D12 and D13): read what it declares (its capabilities and extensions, which the
// device's required list has to give, the pipeline cost study and the pipeline builder), and add the vertex or mesh
// outputs a pixel module reads and the stage before it does not write, set to what the console's next stage reads from an
// unwritten output, (0, 0, 0, 1) (DKB Recomp's spirv_add_outputs does the same for a vertex module; this is a new
// implementation of the rule, see THIRD_PARTY_NOTICES.md). Nothing here needs Vulkan.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define SPV_MAGIC 0x07230203u
#define SPV_MAX_CAPS 48
#define SPV_MAX_EXTS 16
#define SPV_EXT_LEN 64

typedef struct SpvDecl {
    uint32_t version;                             // header word 1 (0x00010400 for SPIR-V 1.4)
    uint32_t ncaps;                               // OpCapability values, in order of appearance, each once
    uint32_t caps[SPV_MAX_CAPS];
    uint32_t nexts;                               // OpExtension names, in order of appearance, each once
    char exts[SPV_MAX_EXTS][SPV_EXT_LEN];
} SpvDecl;

// What the module declares before its memory model. 1 with *out filled, or 0 with one line in why (a header that is
// not SPIR-V, an instruction that runs past the end, more declarations than the arrays hold).
int spv_declarations(const uint32_t *words, size_t nwords, SpvDecl *out, char *why, size_t cap);

// The SPIR-V name of capability value cap ("Float64", "PhysicalStorageBufferAddresses", ...); NULL when this build
// does not name it.
const char *spv_capability_name(uint32_t cap);

// 1 when a compute pipeline made of the module has to be created with the full-subgroups requirement
// (VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT): the module declares CooperativeMatrixKHR. The Vulkan
// valid usage for any pipeline with a cooperative matrix (VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770) reads "must be
// created with the VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT flag or the shader module must be version 1.6
// or greater", and the translator writes SPIR-V 1.3. The flag needs the computeFullSubgroups feature (VUID-VkPipeline
// ShaderStageCreateInfo-flags-02785) and, with no varying subgroup size or required size in the create info, a local size
// X that is a multiple of the subgroup size (flags-02759 and module-08987): the device's subgroup size is fixed at 32 (an
// item of oot2/vk_required.h) and the translator keeps the helper where X is not a multiple of 32. A module the helper
// replaced (spv_hmma_fallback) declares no cooperative matrix and needs nothing.
int spv_needs_full_subgroups(const SpvDecl *d);

// One variable to add: Output storage class, `components` consecutive components starting at `component` of
// `location`, of a 32-bit scalar type (0 float, 1 signed int, 2 unsigned int; the numbering of oot2/shader_desc.h).
enum { SPV_PAD_FLOAT, SPV_PAD_SINT, SPV_PAD_UINT };

typedef struct SpvOutput {
    uint32_t location, component, components, type;
} SpvOutput;

// The module with one Output variable per entry of out[0..n) added: declared in the entry point's interface, decorated
// with Location (and Component when it is not 0), initialised so that component c of the variable reads 1.0f's bits
// when location component (component + c) is 3 and 0 otherwise. Existing scalar, vector and Output pointer types are
// reused; a type or constant that is missing is declared. Returns a malloc'd module of *out_words words, or NULL with
// the reason in why (not SPIR-V, no entry point or more than one, no annotations and no types, out of memory).
uint32_t *spv_add_outputs(const uint32_t *words, size_t nwords, const SpvOutput *add, uint32_t nadd, size_t *out_words,
                          char *why, size_t cap);

// One output to add to a mesh module: as SpvOutput, and per_primitive 1 when the pixel module reads the words once for
// each primitive (its input is decorated PerPrimitiveEXT), 0 when it reads them for each vertex.
typedef struct SpvMeshOutput {
    uint32_t location, component, components, type;
    uint32_t per_primitive;
} SpvMeshOutput;

// The mesh module with the outputs add[0..nadd) added and given the value the console's next stage reads from an
// unwritten word, (0, 0, 0, 1) component by component (1.0f's bits for a location's fourth component, zero for the rest,
// whatever the type), as spv_add_outputs does for a vertex module. A mesh output is an array over the stage's vertices
// (OutputVertices) or primitives (OutputPrimitivesEXT) of a scalar or vector at its one location, and it may not be
// written before OpSetMeshOutputsEXT, so an initializer would not do: the variables are declared, decorated (Location,
// Component when it is not 0, PerPrimitiveEXT) and listed in the entry point's interface, and right after the module's one
// OpSetMeshOutputsEXT every invocation stores the value to its share of the elements (LocalInvocationIndex, a built-in
// the module is given when it has none): the vertices up to the declared count, the primitives below the primitive count
// the module passes to OpSetMeshOutputsEXT, each store of a partial share behind its own bounds test. A module that has a
// block after OpSetMeshOutputsEXT keeps it, now behind the new blocks. When max_bytes or max_components is not 0 the
// padded module's mesh output use, by Vulkan's formula (every output location is four scalars of four bytes for the
// vertex or primitive count rounded up to 32, the position built-in one such location), is held to it, as the translator
// holds a program to the device's limits: past one the module is refused. max_payload_and_output (the device's
// maxMeshPayloadAndOutputMemorySize, 0 for none) holds the output memory and payload_bytes, the task payload the stage
// reads (its description's payload_bytes, 0 for a stage without a task), to their sum. Returns a malloc'd module of
// *out_words words, or NULL with the reason in why (not SPIR-V 1.4 or later, not exactly one MeshEXT entry point, no
// OpSetMeshOutputsEXT or more than one, a mode (LocalSize, OutputVertices, OutputPrimitivesEXT) missing, an unpaddable
// shape, a limit passed, too few threads for the counts (a chunk offset for each share of the elements, more than the
// rewrite holds: named by the threads and the counts), out of memory).
uint32_t *spv_add_mesh_outputs(const uint32_t *words, size_t nwords, const SpvMeshOutput *add, uint32_t nadd,
                               uint32_t max_bytes, uint32_t max_components, uint32_t max_payload_and_output,
                               uint32_t payload_bytes, size_t *out_words, char *why, size_t cap);

// The module with every Workgroup variable that has no initializer given the null value of its type. A translated
// compute module can read workgroup words it never wrote (the title movie's last pass does: with those words zero its
// picture is right, with NaN bits every tile is black), and a device leaves workgroup memory as the dispatch before left
// it, so what the module reads depends on what ran before; the console's program is right when the words are finite
// (Inference). The initializer is the null value, the zero-initialization of workgroup memory a Vulkan 1.3 device offers
// (VkPhysicalDeviceVulkan13Features::shaderZeroInitializeWorkgroupMemory, which the device must have enabled), done by the
// driver before the first instruction. Returns how many variables were given one: 0 with *out NULL when the module needs
// nothing (it has no such variable), else *out is a malloc'd module of *out_words words; -1 with the reason in why (not
// SPIR-V, a variable whose type this cannot name, out of memory).
int spv_zero_workgroup(const uint32_t *words, size_t nwords, uint32_t **out, size_t *out_words, char *why, size_t cap);

// The module with its native HMMA turned into the helper (oot2/vk_hmma.h), for a device whose start-up probe refused the
// native form: every OpFunctionCall of the function named dmsh_hmma16816_coop calls the function named
// dmsh_hmma16816_f16 instead (the two take the same arguments), and the native function, the cooperative matrix types,
// the 16-bit float types and constants only it used, their names and decorations, the capabilities CooperativeMatrixKHR
// and Float16 and the extension SPV_KHR_cooperative_matrix are taken out. The Vulkan memory model stays (an item of the
// required list). Returns 0 with *out NULL when the module declares no cooperative matrix (nothing to do), 1 with *out a
// malloc'd module of *out_words words, or -1 with the reason in why (not SPIR-V, a malformed instruction, a module with
// cooperative matrices but not both functions, a 16-bit float used outside the native function, out of memory).
int spv_hmma_fallback(const uint32_t *words, size_t nwords, uint32_t **out, size_t *out_words, char *why, size_t cap);
