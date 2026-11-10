// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The pipeline builder of the Vulkan back end (architecture D12, D13 and the build policy of section 6.1; the pipeline cost and
// stage bindings studies): the one pipeline layout of the `uni` scheme, pipeline libraries per module, fast links per
// (pair, attachment formats), compute pipelines per module, and the per-player pipeline cache. The Vulkan back end
// calls it at run time and prepare's pipeline step calls it through a program linked with the runtime library, so a
// cache hit at launch sees the create info prepare built (the code that makes every create info is here, once).
//
// What it builds from: the shader cache pack (oot2/shader_cache.h), by the job key of the stage (the programs and state study, decision
// 1: "<key16>.<variant>", the key an nvn_stage carries); modules and descriptions are read from the mapped pack and
// never copied. A key the pack lacks, a job that did not translate, a description of another schema or stage, a module
// that does not fit its stage's slots, one that declares a capability the device's required list does not give
// (oot2/vk_caps.h): each is a named fatal that prints the stage's shape, because no prepared module could ever answer
// it (the programs and state study: the bytes the game passes at run time have not been compared with the archives).
//
// Every get is blocking: it creates what is missing before it returns, on the calling thread, and any number of threads
// may ask at once (two asking for the same pipeline build it once). No pipeline is created or released while a
// submission is in flight (section 6.1; the NVIDIA fault study): a get that would create one with the queue busy is a named
// fatal, the miss path (oot2/vk_miss.h) drains first. The queue counts as idle while the only submission not done is the
// one being replayed (it has put nothing on the queue yet). A pipeline that exists already is returned with no such check.
//
// Layout of the one pipeline layout (the pipeline cost and stage bindings studies; xlate/host/UniLayout.cs is the source
// the tables here are tested against):
//   set 0  the global arrays, update-after-bind and partially bound: binding 0 the 33,025 sampled images, binding 1 the
//          4,096 samplers; the same set layout for graphics and compute (the bindless ticket fills it)
//   set 1  push descriptors, 32 at most: binding 0 the support buffer; graphics: vertex and mesh uniform 1 to 8 and
//          storage 9 to 13, pixel uniform 14 to 24, storage 25 to 29, storage images 30 and 31; compute: uniform 1 to 8,
//          storage 9 to 24, storage images 25 to 31
//   set 2  the task stage's buffers, written per draw (graphics layout only): uniform 0 to 7, storage 8 to 15
//   push constants, 128 bytes for every stage (graphics layout: vertex, fragment, task and mesh)
// The graphics layout holds the three sets for every graphics pipeline, vertex or mesh, so that a fragment library is
// built once per pixel module whatever it is linked to (libraries link only under identical layouts).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/nvn_program.h"
#include "oot2/shader_cache.h"
#include "oot2/shader_desc.h"
#include "oot2/vk_required.h"

// ---- the layout's tables (no device needed)
#define VKP_GLOBAL_IMAGES 33025u
#define VKP_GLOBAL_SAMPLERS 4096u
#define VKP_PUSH_BYTES 128u
#define VKP_SET1_MAX 32u
#define VKP_SET2_MAX 16u
#define VKP_MAX_DYNAMIC 48u
#define VKP_MAX_COLOR 8u

enum { VKP_GRAPHICS, VKP_COMPUTE };

// Set 0's two bindings and their binding flags (update-after-bind, partially bound); returns 2.
uint32_t vkp_set0_bindings(VkDescriptorSetLayoutBinding out[2], VkDescriptorBindingFlags flags[2]);
// Set 1's bindings for the graphics or compute layout, in binding order; returns how many (32 for both).
uint32_t vkp_set1_bindings(int layout, VkDescriptorSetLayoutBinding out[VKP_SET1_MAX]);
// Set 2's bindings (the task stage's buffers); returns 16.
uint32_t vkp_set2_bindings(VkDescriptorSetLayoutBinding out[VKP_SET2_MAX]);
// The push constant range of the graphics or compute layout.
VkPushConstantRange vkp_push_range(int layout);
// The states a library of `part` (VkGraphicsPipelineLibraryFlagBitsEXT, one bit) makes dynamic on a device with these
// facts (D12: vertex input, topology, viewport, culling, depth, stencil, blend, logic op wherever the device allows);
// returns how many.
uint32_t vkp_dynamic_states(VkGraphicsPipelineLibraryFlagsEXT part, const Oot2VkFacts *facts,
                            VkDynamicState out[VKP_MAX_DYNAMIC]);

// ---- the store: modules and descriptions by stage key (no device needed)
typedef struct VkpStage {
    const char *key;            // the job key, "<key16>.<variant>"
    const uint8_t *answers;     // the answers digest (32 bytes), NULL for the default answers
    const nvn_stage *shape;     // what to print when the stage cannot be served; may be NULL
    uint8_t ledger_digest[SHC_DIGEST_BYTES];
    int has_ledger_digest;
} VkpStage;

typedef struct VkpModule {
    const uint32_t *code;       // SPIR-V words, inside the mapped pack
    uint32_t bytes;
    int from;                   // SHC_FROM_BASE or SHC_FROM_VARIANTS
    ShaderDesc desc;
    const char *description;    // the description's text, inside the mapped pack (not NUL-terminated)
    uint32_t description_bytes;
} VkpModule;

// Open the packs (variants may be NULL). 1, or 0 with the reason in why.
int vkp_store_open(const char *base, const char *variants, char *why, size_t cap);
int vkp_store_open_map(const char *base, const char *variants, const char *map, char *why, size_t cap);
void vkp_store_close(void);
// Copy a full ledger-row digest into a stage for map-aware module, binding, and pipeline-key resolution.
void vkp_stage_set_ledger_digest(VkpStage *stage, const uint8_t digest[SHC_DIGEST_BYTES]);
// The selected pack entry's answers digest, or zero for the base entry or a missing module.
void vkp_stage_content_digest(const VkpStage *stage, uint8_t digest[SHC_DIGEST_BYTES]);
// The module of stage s, whose description must name stage kind `expect` (SHD_*) and say it fits its layout. A named
// fatal otherwise (no row, a failure record, a description that cannot be read or is of another stage).
void vkp_module(const VkpStage *s, int expect, VkpModule *out);

// ---- the device: layouts and the pipeline cache
typedef struct VkpLayouts {
    VkDescriptorSetLayout global;           // set 0, shared by both layouts
    VkDescriptorSetLayout graphics_push;    // set 1, graphics
    VkDescriptorSetLayout compute_push;     // set 1, compute
    VkDescriptorSetLayout task;             // set 2, graphics
    VkPipelineLayout graphics, compute;
} VkpLayouts;

// The layouts alone (vk_layout.c), on the back end's device: the tests make them without a store. vkp_open makes them
// itself.
void vkp_layouts_create(VkpLayouts *out);
void vkp_layouts_destroy(VkpLayouts *l);

typedef enum VkpCacheState {
    VKP_CACHE_NONE,            // no file: a first start, an empty cache
    VKP_CACHE_LOADED,          // the file's header names this device and driver: its data seeds the cache
    VKP_CACHE_OTHER_DRIVER,    // the header names another device or driver: dropped, the cache starts empty
    VKP_CACHE_UNREADABLE,      // the file is not a pipeline cache of header version one (or is cut short): dropped
} VkpCacheState;

// Judge cache file data against a device's properties: VKP_CACHE_LOADED when the header is version one and names
// this vendor, device and pipelineCacheUUID, else why it is refused. Nothing is created.
VkpCacheState vkp_cache_judge(const void *data, size_t size, const VkPhysicalDeviceProperties *props);

// After gpu_vk_open and vkp_store_open: the layouts, the vertex input library and the pipeline cache, seeded from the
// file at cache_path when it is valid for this device and driver (NULL: no file). Once.
void vkp_open(const char *cache_path);
// Destroy everything made (the queue must be idle). The cache is not saved: vkp_cache_save does that.
void vkp_close(void);
const VkpLayouts *vkp_layouts(void);
VkpCacheState vkp_cache_state(void);
// 1 when compute modules are built with their workgroup memory zero-initialized (spv_zero_workgroup: the device was
// made with shaderZeroInitializeWorkgroupMemory), else 0. After vkp_open. The pipeline builder says it in its ready line.
int vkp_zero_workgroup(void);
// Write the cache to the path vkp_open was given, through a temporary file and a rename, so that an interrupted write
// leaves the old file whole. 1, or 0 with the reason in why.
int vkp_cache_save(char *why, size_t cap);
// Write the cache's data to path (a temporary file and a rename), then replace the cache with an empty one, so that
// each flush holds only what was built since the last: prepare's pipeline step keeps one such file per batch and merges
// them at the end (a cache that grew by every flush would be written again and again). *bytes, when not NULL, is the
// file's size. 1, or 0 with the reason in why. Nothing made from the cache may still be created (vkp_warm destroys
// what it makes before it returns).
int vkp_cache_flush(const char *path, size_t *bytes, char *why, size_t cap);
// What the cache holds now, in bytes (the size of the data a flush or a save would write).
uint64_t vkp_cache_size(void);

// ---- pipelines
typedef struct VkpFormats {     // the attachments a graphics pipeline draws into
    uint32_t ncolor;            // 0 to VKP_MAX_COLOR
    VkFormat color[VKP_MAX_COLOR];
    VkFormat depth, stencil;    // VK_FORMAT_UNDEFINED for none
} VkpFormats;

// The graphics pipeline of a vertex and a pixel stage (or of a mesh stage, with its task stage if it has one, and a pixel
// stage) drawing into these formats: the fast link of the vertex or mesh stage's pre-rasterization library (padded with
// the outputs the pixel stage reads and it does not write, keyed by the module and the pad set), the pixel stage's
// fragment library, the output library of the formats and the shared vertex input library. Created once, then kept until a
// vkp_release of either stage or vkp_close.
VkPipeline vkp_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats);
// The compute pipeline of a compute stage.
VkPipeline vkp_compute(const VkpStage *cs);
// What the miss path (oot2/vk_miss.h) needs of the builder. The same key text vkp_graphics keeps its object under.
void vkp_graphics_key(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                      char *out, size_t cap);
// The pipeline when it exists already, else VK_NULL_HANDLE (nothing is created, also not while another thread makes it).
VkPipeline vkp_find_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats);
VkPipeline vkp_find_compute(const VkpStage *cs);
// vkp_graphics for the miss path: when neither the vertex nor the fragment library exists they are built side by side
// (the second on a helper thread), and `late`, when not NULL, points at a flag the caller sets once it stopped waiting
// for this build: from then on the build creates its objects with work in flight (the one place the rule of "no pipeline
// while a submission is in flight" yields, counted by the miss path). vkp_graphics is the call with late NULL.
VkPipeline vkp_graphics_ex(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                           const int *late);
// For the tests that run without a device (tests/runtime/vk_pipeline_test.c): vkp_graphics without the check that the
// builder is open, to reach the rule a pair is linked by in build_link. Only a pair that build_link refuses is safe to
// ask for, since one that links goes on to create Vulkan objects and needs the device.
VkPipeline vkp_graphics_unopened(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats);
// vkp_compute for the miss path. A `late` that is not NULL (here and in vkp_graphics_ex) marks a build made by the miss
// path with the submission gate held: the submission being replayed is then not counted as work in flight. Any other
// caller keeps the strict rule.
VkPipeline vkp_compute_ex(const VkpStage *cs, const int *late);
// What prepare builds ahead of the first frame, one stage at a time (architecture section 6.1, step 1): the
// pre-rasterization library of a vertex stage as translated, with no padding (the padded ones are known only from play,
// and come from the ledger), the fragment library of a pixel stage, the compute pipeline of a compute stage. The same
// create info as vkp_graphics and vkp_compute make, so the driver's cache serves the later get; the object is destroyed
// before the call returns. A key the store lacks, a job that did not translate or a module the device cannot take is a
// named fatal, as it is in the gets.
enum { VKP_WARM_VERTEX, VKP_WARM_PIXEL, VKP_WARM_COMPUTE };
typedef struct VkpWarm {
    uint64_t ns;               // wall time in the driver
    int hit;                   // the application cache served it (stage libraries and compute pipelines only)
    uint32_t module_bytes;
} VkpWarm;
int vkp_warm(int what, const VkpStage *s, VkpWarm *out);
// Tests only: while non-zero, a vertex or mesh module is linked as translated, with no padding (the counter-run of the
// padding rules: the validation layer then reports VUID-RuntimeSpirv-OpEntryPoint-08743 for a pixel input nothing writes).
extern int vkp_unpad;
// Destroy every library and pipeline that was made from the stage with this key (for program_release, when no program
// holds it). The caller guarantees that no submission in flight uses them.
void vkp_release(const char *key);

typedef struct VkpStats {
    // Libraries made (the shared vertex input library, the pre-rasterization and fragment libraries, the output
    // libraries) and how many the cache served; stage_* count only the libraries that hold a shader. The driver caches
    // what it compiled: a library with no shader and a fast link are never reported as hits.
    uint64_t libraries, library_hits;
    uint64_t stage_libraries, stage_library_hits;
    uint64_t links, link_hits;            // fast links
    uint64_t computes, compute_hits;
    uint64_t padded;                      // vertex libraries made from a padded module
    uint64_t resident;                    // objects held now
    uint64_t build_ns[3];                 // wall time in the driver: libraries, links, computes
    uint64_t map_missing, map_unknown_rows;  // full ledger digests without an available map or map entry
} VkpStats;
void vkp_stats(VkpStats *out);
