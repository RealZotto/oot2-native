// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The pipeline builder (oot2/vk_pipeline.h): modules by stage key out of the shader cache pack, pipeline libraries per
// module, fast links per (pair, attachment formats), compute pipelines, and the per-player pipeline cache. The create
// infos follow spike S3's harness (tools/pipecost/vkpc_build.c), the measurement that decided D12: vertex input,
// topology, viewport, culling, depth, stencil and blend state dynamic wherever the device allows, so a pipeline is a
// function of its stages and attachment formats.
//
// Everything made is kept in one table under one lock, by a key text; a get that finds its object returns it, one that
// finds it being made waits for the thread making it, and one that finds nothing makes it with the lock released, so
// builds run in parallel. Objects are not evicted: what stays resident is what callers ask for (the pipeline cost study:
// 20,000 resident pipelines held 7.6 GB), and vkp_release lets a stage's objects go.
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <direct.h>
#include <io.h>
#include <windows.h>
// 64-bit file offsets and a flush to the disk: a pipeline cache of a whole game passes 2 GiB, which a 32-bit `long`
// (Windows' ftell) cannot size, and a cache file renamed into place must not be half written after a power cut.
#define OFF_SEEK _fseeki64
#define OFF_TELL _ftelli64
#define SYNC_FILE(f) _commit(_fileno(f))
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#define OFF_SEEK fseeko
#define OFF_TELL ftello
#define SYNC_FILE(f) fsync(fileno(f))
#endif

#include "oot2/bench_timers.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/replay_count.h"
#include "oot2/runtime.h"
#include "oot2/spirv.h"
#include "oot2/vk_caps.h"
#include "oot2/vk_hmma.h"
#include "oot2/vk_pipeline.h"
#include "plat.h"

#define KEY_MAX 512
#define BUCKETS 8192
#define MAX_PADS 16
// Libraries are made able to be linked with link-time optimisation, as S3 measured them (D12 keeps the fast link; an
// optimised link would need the flag, and its absence would change the figures the pipeline cost study gives for warm builds).
#define LIB_FLAGS (VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT)

#define PART_VERTEX_INPUT VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT
#define PART_PRE_RASTER VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT
#define PART_FRAGMENT VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT
#define PART_OUTPUT VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT

int vkp_unpad;   // tests only: leave the vertex and mesh modules as translated (the counter-run of the padding rules)

enum { K_LIBRARY, K_LINK, K_COMPUTE };

typedef struct Obj {
    struct Obj *next;
    char key[KEY_MAX];
    char ref[3][32];       // the stage keys this object was made from, for vkp_release
    VkPipeline pipe;
    int ready;
} Obj;

static struct {
    int store_open, open;
    ShcSet set;
    VkpLayouts lay;
    VkPipelineCache cache;
    VkPipeline vi;                 // the vertex input interface library, the same for every pipeline
    VkpCacheState cache_state;
    char cache_path[512];
    int flushed;                   // vkp_cache_flush swapped the cache for an empty one: a save would write a part of it
    int unreadable;                // the cache file is there and was not read (the VKP_READ_* cause; 0: read, or no file): a save
                                   // would replace it with this run's part
    int unreadable_err;            // the errno of the open that failed (VKP_READ_OPEN)
    VkcList caps;
    int zero_workgroup;            // the device was made with zero-initialized workgroup memory (an optional feature)
    PlatMutex lock;
    PlatCond cond;
    Obj *tab[BUCKETS];
    VkpStats stats;
    uint64_t map_missing, map_unknown_rows;
} P = {.lock = PLAT_MUTEX_INIT, .cond = PLAT_COND_INIT};

// ---------- the store

int vkp_store_open(const char *base, const char *variants, char *why, size_t cap) {
    ShcError e = shc_set_open(&P.set, base, variants, SHC_DESC_SCHEMA, why, cap);
    P.store_open = e == SHC_OK;
    return P.store_open;
}

int vkp_store_open_map(const char *base, const char *variants, const char *map, char *why, size_t cap) {
    ShcError e = shc_set_open_map(&P.set, base, variants, map, SHC_DESC_SCHEMA, why, cap);
    P.store_open = e == SHC_OK;
    return P.store_open;
}

void vkp_stage_set_ledger_digest(VkpStage *stage, const uint8_t digest[SHC_DIGEST_BYTES]) {
    memcpy(stage->ledger_digest, digest, SHC_DIGEST_BYTES);
    stage->has_ledger_digest = 1;
}

void vkp_stage_content_digest(const VkpStage *stage, uint8_t digest[SHC_DIGEST_BYTES]) {
    memset(digest, 0, SHC_DIGEST_BYTES);
    if (!P.store_open || !stage || !stage->key) return;
    ShcEntry entry;
    int from = stage->has_ledger_digest
                   ? shc_set_find_ledger(&P.set, stage->key, stage->ledger_digest, &entry, NULL)
                   : shc_set_find(&P.set, stage->key, stage->answers, &entry);
    if (from != SHC_MISS) memcpy(digest, entry.answers, SHC_DIGEST_BYTES);
    if (RC_ON()) {   // a lookup of a stage by a draw: whether the store answers with a variant of the base module
        rc_add_(RC_CONTENT_N, 1);
        if (from == SHC_MISS) rc_add_(RC_CONTENT_MISS, 1);
        else if (from == SHC_FROM_VARIANTS) rc_add_(RC_CONTENT_VARIANT, 1);
    }
}

void vkp_store_close(void) {
    if (P.store_open) shc_set_close(&P.set);
    P.store_open = 0;
}

static void shape_text(const VkpStage *s, char *out, size_t cap) {
    if (!s->shape) {
        snprintf(out, cap, "key %s, no shape recorded", s->key);
        return;
    }
    const char *kind = nvn_stage_kind_name(s->shape->kind);
    snprintf(out, cap, "a %s stage, key %s, code area %u bytes, instructions %u bytes, control section %u bytes",
             kind ? kind : "?", s->key, s->shape->code_size, s->shape->inst_size, s->shape->ctl_size);
}

static void digest_text(const VkpStage *s, char *out, size_t cap) {
    out[0] = 0;
    const uint8_t *answers = s ? (s->has_ledger_digest ? s->ledger_digest : s->answers) : NULL;
    if (!answers) return;
    static const uint8_t zero[SHC_DIGEST_BYTES];
    if (!memcmp(answers, zero, sizeof zero)) return;
    snprintf(out, cap, "@%02x%02x%02x%02x%02x%02x%02x%02x", answers[0], answers[1], answers[2], answers[3],
             answers[4], answers[5], answers[6], answers[7]);
}

static void cache_digest_text(const VkpStage *s, char *out, size_t cap) {
    uint8_t digest[SHC_DIGEST_BYTES] = {0};
    out[0] = 0;
    if (!s) return;
    if (s->has_ledger_digest) {
        if (P.store_open)
            shc_set_resolve_ledger(&P.set, s->key, s->ledger_digest, digest);
    } else if (s->answers) {
        memcpy(digest, s->answers, sizeof digest);
    }
    static const uint8_t zero[SHC_DIGEST_BYTES];
    if (!memcmp(digest, zero, sizeof zero)) return;
    size_t at = (size_t)snprintf(out, cap, "@");
    for (size_t i = 0; i < sizeof digest && at < cap; i++)
        at += (size_t)snprintf(out + at, cap - at, "%02x", digest[i]);
}

void vkp_module(const VkpStage *s, int expect, VkpModule *out) {
    char shape[256], why[256], dg[24];
    shape_text(s, shape, sizeof shape);
    digest_text(s, dg, sizeof dg);
    memset(out, 0, sizeof *out);
    ShcEntry e;
    int from = SHC_MISS;
    if (P.store_open && s->key) {
        if (s->has_ledger_digest) {
            int mapped = 0;
            from = shc_set_find_ledger(&P.set, s->key, s->ledger_digest, &e, &mapped);
            if (!P.set.has_variant_map)
                __atomic_fetch_add(&P.map_missing, 1, __ATOMIC_RELAXED);
            else if (!mapped)
                __atomic_fetch_add(&P.map_unknown_rows, 1, __ATOMIC_RELAXED);
        } else {
            from = shc_set_find(&P.set, s->key, s->answers, &e);
        }
    }
    if (from == SHC_MISS)
        rt_fatal("gfx: vulkan: the shader cache has no module for stage %s%s (%s): prepare translates the programs of the "
                 "game's archives, and this program is not among them", s->key ? s->key : "(none)", dg, shape);
    if (e.status != SHC_TRANSLATED)
        rt_fatal("gfx: vulkan: stage %s%s did not translate: %.*s (%s)", s->key, dg, (int)e.description_bytes, e.description,
                 shape);
    if (!shd_parse(e.description, e.description_bytes, SHC_DESC_SCHEMA, &out->desc, why, sizeof why))
        rt_fatal("gfx: vulkan: stage %s%s: %s (%s)", s->key, dg, why, shape);
    if (out->desc.kind != expect)
        rt_fatal("gfx: vulkan: stage %s%s is a %s program where a %s was asked for (%s)", s->key, dg,
                 shd_kind_name(out->desc.kind), shd_kind_name(expect), shape);
    if (!out->desc.fits)
        rt_fatal("gfx: vulkan: stage %s%s does not fit its slots of the uni layout (its description says fits: false; %s)",
                 s->key, dg, shape);
    out->code = e.module;
    out->bytes = e.module_bytes;
    out->from = from;
    out->description = e.description;
    out->description_bytes = e.description_bytes;
}

// ---------- the table

static uint32_t hash_key(const char *k) {
    uint32_t h = 2166136261u;
    for (; *k; k++) h = (h ^ (uint8_t)*k) * 16777619u;
    return h % BUCKETS;
}

static Obj *find_locked(uint32_t h, const char *key) {
    for (Obj *o = P.tab[h]; o; o = o->next)
        if (!strcmp(o->key, key)) return o;
    return NULL;
}

typedef VkPipeline (*BuildFn)(void *ctx);

// The object under key, made by build(ctx) the first time it is asked for, once however many threads ask.
static VkPipeline obj_get(const char *key, const char *ref0, const char *ref1, const char *ref2, BuildFn build, void *ctx) {
    uint32_t h = hash_key(key);
    plat_mutex_lock(&P.lock);
    for (;;) {
        Obj *o = find_locked(h, key);
        if (!o) break;
        if (o->ready) {
            VkPipeline p = o->pipe;
            plat_mutex_unlock(&P.lock);
            return p;
        }
        plat_cond_wait(&P.cond, &P.lock);   // another thread is making it
    }
    Obj *o = calloc(1, sizeof *o);
    if (!o) rt_fatal("gfx: vulkan: out of memory for a pipeline record");
    snprintf(o->key, sizeof o->key, "%s", key);
    if (ref0) snprintf(o->ref[0], sizeof o->ref[0], "%s", ref0);
    if (ref1) snprintf(o->ref[1], sizeof o->ref[1], "%s", ref1);
    if (ref2) snprintf(o->ref[2], sizeof o->ref[2], "%s", ref2);
    o->next = P.tab[h];
    P.tab[h] = o;
    plat_mutex_unlock(&P.lock);
    VkPipeline p = build(ctx);
    plat_mutex_lock(&P.lock);
    o->pipe = p;
    o->ready = 1;
    P.stats.resident++;
    plat_cond_broadcast(&P.cond);
    plat_mutex_unlock(&P.lock);
    return p;
}

// ---------- creation

// The queue is idle when every submission handed out before the one being replayed is done: the open submission holds
// the gate and has put nothing on the queue yet (its commands are in the recorder's ring or in a command buffer that is
// not submitted), so the miss path (vk_miss.c) can drain and then build in the middle of a frame. `late` is NULL for
// an ordinary get; the miss path passes a flag it sets when it stopped waiting for this build, and from then on the
// build runs beside work in flight, the one place the rule yields (oot2/vk_miss.h).
// The open submission is left out only for a miss-path build (`late` not NULL): the miss path holds the gate, so nothing
// submits while it builds. Any other caller keeps the strict rule: between vkQueueSubmit2 and the end of the submission
// the open record still names a submission that is on the queue.
static void require_idle(const char *what, const char *key, const int *late) {
    if (late && __atomic_load_n(late, __ATOMIC_ACQUIRE)) return;
    uint64_t handed = gpu_sync_handed(), open = gpu_sync_open(), done = gpu_sync_done();
    uint64_t queued = late && open && open == handed ? handed - 1 : handed;
    if (queued != done) {
        // the replay may have stopped waiting for this build, and put the frame on the queue, since the flag was read
        if (late && __atomic_load_n(late, __ATOMIC_ACQUIRE)) return;
        rt_fatal("gfx: vulkan: %s for stage %s with work in flight (submissions up to %llu handed out, %llu done): no "
                 "pipeline is created while a submission is in flight (architecture section 6.1)", what, key,
                 (unsigned long long)queued, (unsigned long long)done);
    }
}

// Returns 1 when a compute pipeline of the module has to be created with the full-subgroups requirement
// (spv_needs_full_subgroups); the graphics and mesh callers have no use for it.
static int check_capabilities(const VkpStage *s, const VkpModule *m) {
    SpvDecl d;
    char why[256], shape[256];
    if (!spv_declarations(m->code, m->bytes / 4, &d, why, sizeof why)) {
        shape_text(s, shape, sizeof shape);
        rt_fatal("gfx: vulkan: stage %s: %s (%s)", s->key, why, shape);
    }
    if (vkc_check(&P.caps, &d, why, sizeof why)) {
        shape_text(s, shape, sizeof shape);
        rt_fatal("gfx: vulkan: stage %s declares something the device's required list does not give: %s (%s)", s->key, why,
                 shape);
    }
    return spv_needs_full_subgroups(&d);
}

// What a creation tells note: bit 0 the module was padded, bit 1 the library holds a shader stage, bit 2 the compute pipeline
// was created with the full-subgroups requirement.
enum { LIB_PADDED = 1, LIB_SHADER = 2, CS_FULL_SUBGROUPS = 4 };

static void note(int kind, const VkPipelineCreationFeedback *fb, uint64_t ns, int flags) {
    int hit = (fb->flags & VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT) &&
              (fb->flags & VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT);
    plat_mutex_lock(&P.lock);
    P.stats.build_ns[kind] += ns;
    if (kind == K_LIBRARY) {
        P.stats.libraries++, P.stats.library_hits += hit;
        if (flags & LIB_SHADER) P.stats.stage_libraries++, P.stats.stage_library_hits += hit;
    } else if (kind == K_LINK) {
        P.stats.links++, P.stats.link_hits += hit;
    } else {
        P.stats.computes++, P.stats.compute_hits += hit;
    }
    P.stats.padded += (flags & LIB_PADDED) != 0;
    P.stats.full_subgroups += (flags & CS_FULL_SUBGROUPS) != 0;
    plat_mutex_unlock(&P.lock);
}

static VkShaderModule make_module(const uint32_t *code, size_t words) {
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = words * 4, .pCode = code};
    VkShaderModule m;
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &m), "vkCreateShaderModule");
    return m;
}

// The create info of one library or of a whole graphics pipeline, with every static field at its default.
typedef struct PipeInfo {
    VkPipelineVertexInputStateCreateInfo vis;
    VkPipelineInputAssemblyStateCreateInfo ias;
    VkPipelineViewportStateCreateInfo vps;
    VkPipelineRasterizationStateCreateInfo rs;
    VkPipelineMultisampleStateCreateInfo ms;
    VkPipelineDepthStencilStateCreateInfo ds;
    VkPipelineColorBlendAttachmentState cba[VKP_MAX_COLOR];
    VkPipelineColorBlendStateCreateInfo cbs;
    VkFormat colour[VKP_MAX_COLOR];
    VkPipelineRenderingCreateInfo rendering;
    VkDynamicState dyn[VKP_MAX_DYNAMIC];
    VkPipelineDynamicStateCreateInfo dynamic;
    VkPipelineShaderStageCreateInfo stage[2];
    VkGraphicsPipelineLibraryCreateInfoEXT lib;
    VkGraphicsPipelineCreateInfo gp;
} PipeInfo;

// part: one library bit. module: the stage's module (pre-rasterization or fragment shader part), else VK_NULL_HANDLE.
static void pipe_info(PipeInfo *p, VkGraphicsPipelineLibraryFlagsEXT part, VkShaderModule module, const VkpFormats *f) {
    memset(p, 0, sizeof *p);
    p->vis.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    p->ias = (VkPipelineInputAssemblyStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    p->vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    p->rs = (VkPipelineRasterizationStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f};
    p->ms = (VkPipelineMultisampleStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    p->ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    uint32_t ncolor = (part & PART_OUTPUT) && f ? f->ncolor : 0;
    for (uint32_t i = 0; i < VKP_MAX_COLOR; i++) {
        p->cba[i].colorWriteMask = 0xF;
        p->colour[i] = i < ncolor ? f->color[i] : VK_FORMAT_UNDEFINED;
    }
    p->cbs = (VkPipelineColorBlendStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                   .attachmentCount = ncolor, .pAttachments = p->cba};
    // The fragment shader library does not know the attachments: its depth and stencil formats stay undefined so that
    // it links with an output library of any formats (the specification asks only that the view masks agree).
    p->rendering = (VkPipelineRenderingCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = ncolor,
        .pColorAttachmentFormats = p->colour, .depthAttachmentFormat = (part & PART_OUTPUT) && f ? f->depth : VK_FORMAT_UNDEFINED,
        .stencilAttachmentFormat = (part & PART_OUTPUT) && f ? f->stencil : VK_FORMAT_UNDEFINED};
    uint32_t nd = vkp_dynamic_states(part, &gpu_vk.facts, p->dyn);
    p->dynamic = (VkPipelineDynamicStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                    .dynamicStateCount = nd, .pDynamicStates = p->dyn};
    uint32_t ns = 0;
    if (module) {
        p->stage[0] = (VkPipelineShaderStageCreateInfo){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = (part & PART_PRE_RASTER) ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT, .module = module,
            .pName = "main"};
        ns = 1;
    }
    p->lib = (VkGraphicsPipelineLibraryCreateInfoEXT){.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                      .pNext = &p->rendering, .flags = part};
    int pre = (part & PART_PRE_RASTER) != 0, frag = (part & PART_FRAGMENT) != 0, out = (part & PART_OUTPUT) != 0,
        vin = (part & PART_VERTEX_INPUT) != 0;
    p->gp = (VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &p->lib, .flags = LIB_FLAGS, .stageCount = ns,
        .pStages = ns ? p->stage : NULL, .pVertexInputState = vin ? &p->vis : NULL,
        .pInputAssemblyState = vin ? &p->ias : NULL, .pViewportState = pre ? &p->vps : NULL,
        .pRasterizationState = pre ? &p->rs : NULL, .pMultisampleState = (frag || out) ? &p->ms : NULL,
        .pDepthStencilState = frag ? &p->ds : NULL, .pColorBlendState = out ? &p->cbs : NULL, .pDynamicState = &p->dynamic,
        .layout = (pre || frag) ? P.lay.graphics : VK_NULL_HANDLE};
}

typedef struct MeshCtx {
    const VkpStage *task, *mesh;
    const VkpModule *task_module, *mesh_module;
    ShdPad pad[MAX_PADS];   // what the pixel stage reads and the mesh stage does not write (shd_mesh_link)
    int npad;
    const int *late;
} MeshCtx;

static VkPipeline create_graphics(VkGraphicsPipelineCreateInfo ci, int kind, int flags, const char *what);

// The key a mesh stage's pre-rasterization library is kept under: the task stage and the mesh stage with their answers,
// and the pad set (shd_pad_text: each run's location, component, count, type and, read per primitive, its rate), so that
// two pixel stages that need different padding of one mesh stage get two libraries and two that need the same share one.
void vkp_mesh_library_key(const VkpStage *task, const VkpStage *mesh, const ShdPad *pad, int npad, char *out, size_t cap) {
    char md[66], td[66], pads[MAX_PADS * 16 + 8];
    shd_pad_text(pad, npad, pads, sizeof pads);
    cache_digest_text(mesh, md, sizeof md);
    if (task) {
        cache_digest_text(task, td, sizeof td);
        snprintf(out, cap, "m|t:%s%s|m:%s%s|%s", task->key, td, mesh->key, md, pads);
    } else snprintf(out, cap, "m|m:%s%s|%s", mesh->key, md, pads);
}

// The runs of a pad set as the rewrite takes them, each with the rate of the pixel input that reads it.
uint32_t vkp_mesh_pad_outputs(const ShdPad *pad, int npad, SpvMeshOutput *out) {
    for (int i = 0; i < npad; i++)
        out[i] = (SpvMeshOutput){pad[i].location, pad[i].component, pad[i].components, pad[i].type, pad[i].per_primitive};
    return (uint32_t)npad;
}

static VkPipeline build_mesh_library(void *c) {
    MeshCtx *x = c;
    require_idle("a mesh pre-rasterization library would be created", x->mesh->key, x->late);
    check_capabilities(x->mesh, x->mesh_module);
    VkShaderModule task = VK_NULL_HANDLE;
    if (x->task) {
        check_capabilities(x->task, x->task_module);
        task = make_module(x->task_module->code, x->task_module->bytes / 4);
    }
    const uint32_t *code = x->mesh_module->code;
    size_t words = x->mesh_module->bytes / 4;
    uint32_t *padded = NULL;
    if (x->npad) {
        // The mesh stage writes less than its pixel stage reads: it gets the missing words as outputs of the value the
        // console's next stage reads from an unwritten one, (0, 0, 0, 1), as a vertex stage does. The device's mesh
        // output limits hold the padded stage to what the translator held the program to: the output memory, the output
        // components, and the task payload and the output memory together (the stage's payload as its description says).
        SpvMeshOutput add[MAX_PADS];
        uint32_t nadd = vkp_mesh_pad_outputs(x->pad, x->npad, add);
        char why[256], shape[256], ranges[MAX_PADS * 16 + 8];
        padded = spv_add_mesh_outputs(code, words, add, nadd, gpu_vk.facts.mesh_props.maxMeshOutputMemorySize,
                                      gpu_vk.facts.mesh_props.maxMeshOutputComponents,
                                      gpu_vk.facts.mesh_props.maxMeshPayloadAndOutputMemorySize,
                                      x->mesh_module->desc.mesh_payload_bytes, &words, why, sizeof why);
        if (!padded) {
            shape_text(x->mesh, shape, sizeof shape);
            shd_pad_text(x->pad, x->npad, ranges, sizeof ranges);
            rt_fatal("gfx: vulkan: mesh stage %s cannot be padded with the outputs its pixel stage reads "
                     "(location:component:components:type %s): %s (%s)", x->mesh->key, ranges, why, shape);
        }
        code = padded;
    }
    VkShaderModule mesh = make_module(code, words);
    PipeInfo pi;
    pipe_info(&pi, PART_PRE_RASTER, NULL, NULL);
    uint32_t n = 0;
    if (task)
        pi.stage[n++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                          .stage = VK_SHADER_STAGE_TASK_BIT_EXT, .module = task,
                                                          .pName = "main"};
    pi.stage[n++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      .stage = VK_SHADER_STAGE_MESH_BIT_EXT, .module = mesh,
                                                      .pName = "main"};
    pi.gp.stageCount = n, pi.gp.pStages = pi.stage;
    VkPipeline p = create_graphics(pi.gp, K_LIBRARY, LIB_SHADER | (padded ? LIB_PADDED : 0),
                                   "vkCreateGraphicsPipelines (mesh pre-rasterization library)");
    if (task) vkDestroyShaderModule(gpu_vk.device, task, NULL);
    vkDestroyShaderModule(gpu_vk.device, mesh, NULL);
    free(padded);
    return p;
}

// vkCreateGraphicsPipelines through the cache, with creation feedback; a failed call is gpu_vk_failed's.
static VkPipeline create_graphics(VkGraphicsPipelineCreateInfo ci, int kind, int flags, const char *what) {
    VkPipelineCreationFeedback fb = {0}, stage_fb[2] = {{0}, {0}};
    VkPipelineCreationFeedbackCreateInfo fbi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                                .pNext = ci.pNext, .pPipelineCreationFeedback = &fb,
                                                .pipelineStageCreationFeedbackCount = ci.stageCount,
                                                .pPipelineStageCreationFeedbacks = stage_fb};
    ci.pNext = &fbi;
    uint64_t t0 = plat_now_ns();
    VkPipeline p;
    VkResult r = vkCreateGraphicsPipelines(gpu_vk.device, P.cache, 1, &ci, NULL, &p);
    uint64_t ns = plat_now_ns() - t0;
    if (r != VK_SUCCESS) gpu_vk_failed(r, what);
    note(kind, &fb, ns, flags);
    return p;
}

// ---------- the libraries

typedef struct VertexCtx {
    const VkpStage *stage;
    const VkpModule *module;
    ShdPad pad[MAX_PADS];
    int npad;
    const int *late;
} VertexCtx;

static VkPipeline build_vertex_library(void *c) {
    VertexCtx *x = c;
    require_idle("a pre-rasterization library would be created", x->stage->key, x->late);
    check_capabilities(x->stage, x->module);
    const uint32_t *code = x->module->code;
    size_t words = x->module->bytes / 4;
    uint32_t *padded = NULL;
    if (x->npad) {
        SpvOutput add[MAX_PADS];
        for (int i = 0; i < x->npad; i++)
            add[i] = (SpvOutput){x->pad[i].location, x->pad[i].component, x->pad[i].components, x->pad[i].type};
        char why[256], shape[256];
        padded = spv_add_outputs(code, words, add, (uint32_t)x->npad, &words, why, sizeof why);
        if (!padded) {
            shape_text(x->stage, shape, sizeof shape);
            rt_fatal("gfx: vulkan: stage %s cannot be padded with the outputs its pixel stage reads: %s (%s)",
                     x->stage->key, why, shape);
        }
        code = padded;
    }
    VkShaderModule m = make_module(code, words);
    PipeInfo pi;
    pipe_info(&pi, PART_PRE_RASTER, m, NULL);
    VkPipeline p = create_graphics(pi.gp, K_LIBRARY, LIB_SHADER | (padded ? LIB_PADDED : 0),
                                   "vkCreateGraphicsPipelines (pre-rasterization library)");
    vkDestroyShaderModule(gpu_vk.device, m, NULL);
    free(padded);
    return p;
}

typedef struct FragmentCtx {
    const VkpStage *stage;
    const VkpModule *module;
    const int *late;
} FragmentCtx;

static VkPipeline build_fragment_library(void *c) {
    FragmentCtx *x = c;
    require_idle("a fragment library would be created", x->stage->key, x->late);
    check_capabilities(x->stage, x->module);
    VkShaderModule m = make_module(x->module->code, x->module->bytes / 4);
    PipeInfo pi;
    pipe_info(&pi, PART_FRAGMENT, m, NULL);
    VkPipeline p = create_graphics(pi.gp, K_LIBRARY, LIB_SHADER, "vkCreateGraphicsPipelines (fragment library)");
    vkDestroyShaderModule(gpu_vk.device, m, NULL);
    return p;
}

typedef struct OutputCtx {
    const VkpFormats *formats;
    const int *late;
} OutputCtx;

static VkPipeline build_output_library(void *c) {
    const OutputCtx *x = c;
    require_idle("an output library would be created", "(attachment formats)", x->late);
    PipeInfo pi;
    pipe_info(&pi, PART_OUTPUT, VK_NULL_HANDLE, x->formats);
    return create_graphics(pi.gp, K_LIBRARY, 0, "vkCreateGraphicsPipelines (fragment output library)");
}

static void formats_text(const VkpFormats *f, char *out, size_t cap) {
    size_t at = (size_t)snprintf(out, cap, "%u", f->ncolor);
    for (uint32_t i = 0; i < f->ncolor && i < VKP_MAX_COLOR && at < cap; i++) at += (size_t)snprintf(out + at, cap - at, ",%d", (int)f->color[i]);
    if (at < cap) snprintf(out + at, cap - at, "/%d/%d", (int)f->depth, (int)f->stencil);
}

// ---------- graphics pipelines

typedef struct LinkCtx {
    const VkpStage *vs, *task, *ps;
    const VkpFormats *formats;
    const int *late;
    char vkey[KEY_MAX], pkey[KEY_MAX], okey[KEY_MAX];
} LinkCtx;

static int obj_ready(const char *key) {
    uint32_t h = hash_key(key);
    plat_mutex_lock(&P.lock);
    Obj *o = find_locked(h, key);
    int ready = o && o->ready;
    plat_mutex_unlock(&P.lock);
    return ready;
}

// One library made on a helper thread beside the caller's own (the miss path's workers: the vertex and the fragment
// library of a pipeline do not depend on each other, and the driver compiles on the calling thread).
typedef struct LibJob {
    const char *key, *ref;
    BuildFn fn;
    void *ctx;
    int done;   // under P.lock
} LibJob;

#define HELPER_STACK (8ull << 20)

static void lib_job_main(void *arg) {
    LibJob *j = arg;
    obj_get(j->key, j->ref, NULL, NULL, j->fn, j->ctx);
    plat_mutex_lock(&P.lock);
    j->done = 1;   // j is not touched after the unlock
    plat_cond_broadcast(&P.cond);
    plat_mutex_unlock(&P.lock);
}

static VkPipeline build_link(void *c) {
    LinkCtx *x = c;
    if (x->vs->shape && x->vs->shape->kind == NVN_STAGE_MS) {
        VkpModule tm = {0}, mm, pm;
        if (x->task) vkp_module(x->task, SHD_TS, &tm);
        vkp_module(x->vs, SHD_MS, &mm);
        vkp_module(x->ps, SHD_PS, &pm);
        char why[256], pd[66], ft[96];
        MeshCtx mx = {.task = x->task, .mesh = x->vs, .task_module = x->task ? &tm : NULL, .mesh_module = &mm,
                      .late = x->late};
        mx.npad = shd_mesh_link(&mm.desc, &pm.desc, mx.pad, MAX_PADS, why, sizeof why);
        if (mx.npad < 0)
            rt_fatal("gfx: vulkan: mesh stage %s cannot be linked to pixel stage %s: %s", x->vs->key, x->ps->key, why);
        // What the pixel stage reads and the mesh stage does not write is added to the mesh stage as outputs, as a vertex
        // stage is padded, and the library is keyed by the pad set. vkp_unpad is the tests' counter-run of the rule.
        if (vkp_unpad) mx.npad = 0;
        vkp_mesh_library_key(x->task, x->vs, mx.pad, mx.npad, x->vkey, sizeof x->vkey);
        cache_digest_text(x->ps, pd, sizeof pd);
        snprintf(x->pkey, sizeof x->pkey, "f|%s%s", x->ps->key, pd);
        formats_text(x->formats, ft, sizeof ft);
        snprintf(x->okey, sizeof x->okey, "o|%s", ft);
        OutputCtx ox = {.formats = x->formats, .late = x->late};
        VkPipeline pre = obj_get(x->vkey, x->task ? x->task->key : NULL, x->vs->key, NULL, build_mesh_library, &mx);
        VkPipeline out = obj_get(x->okey, NULL, NULL, NULL, build_output_library, &ox);
        FragmentCtx fx = {.stage = x->ps, .module = &pm, .late = x->late};
        VkPipeline frag = obj_get(x->pkey, x->ps->key, NULL, NULL, build_fragment_library, &fx);
        VkPipeline libs[3] = {pre, frag, out};
        VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
                                             .libraryCount = 3, .pLibraries = libs};
        VkGraphicsPipelineCreateInfo gp = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                           .layout = P.lay.graphics};
        require_idle("a mesh pipeline would be linked", x->vs->key, x->late);
        return create_graphics(gp, K_LINK, 0, "vkCreateGraphicsPipelines (mesh fast link)");
    }
    if (x->task) rt_fatal("gfx: vulkan: a task stage is bound without a mesh stage");
    VkpModule vm, pm;
    vkp_module(x->vs, SHD_VS, &vm);
    vkp_module(x->ps, SHD_PS, &pm);
    VertexCtx vx = {.stage = x->vs, .module = &vm, .late = x->late};
    char pads[MAX_PADS * 16 + 8], why[256], dg[66], shape[256];
    if (!vkp_unpad) {
        vx.npad = shd_pad_set(&vm.desc, &pm.desc, vx.pad, MAX_PADS, why, sizeof why);
        if (vx.npad < 0) {
            shape_text(x->vs, shape, sizeof shape);
            rt_fatal("gfx: vulkan: stage %s cannot be linked to stage %s: %s (%s)", x->vs->key, x->ps->key, why, shape);
        }
    }
    shd_pad_text(vx.pad, vx.npad, pads, sizeof pads);
    cache_digest_text(x->vs, dg, sizeof dg);
    snprintf(x->vkey, sizeof x->vkey, "v|%s%s|%s", x->vs->key, dg, pads);
    cache_digest_text(x->ps, dg, sizeof dg);
    snprintf(x->pkey, sizeof x->pkey, "f|%s%s", x->ps->key, dg);
    FragmentCtx fx = {.stage = x->ps, .module = &pm, .late = x->late};
    char ft[96];
    formats_text(x->formats, ft, sizeof ft);
    snprintf(x->okey, sizeof x->okey, "o|%s", ft);
    OutputCtx ox = {.formats = x->formats, .late = x->late};
    // The vertex and the fragment library build side by side when neither exists: the fragment one on a helper thread.
    LibJob fj = {x->pkey, x->ps->key, build_fragment_library, &fx, 0};
    PlatThread *helper = NULL;
    if (!obj_ready(x->vkey) && !obj_ready(x->pkey)) helper = plat_thread_start(lib_job_main, &fj, HELPER_STACK);
    VkPipeline pre = obj_get(x->vkey, x->vs->key, NULL, NULL, build_vertex_library, &vx);
    VkPipeline out = obj_get(x->okey, NULL, NULL, NULL, build_output_library, &ox);
    if (helper) {
        plat_mutex_lock(&P.lock);
        while (!fj.done) plat_cond_wait(&P.cond, &P.lock);
        plat_mutex_unlock(&P.lock);
        plat_thread_free(helper);
    }
    VkPipeline frag = obj_get(x->pkey, x->ps->key, NULL, NULL, build_fragment_library, &fx);

    require_idle("a pipeline would be linked", x->vs->key, x->late);
    VkPipeline libs[4] = {P.vi, pre, frag, out};
    VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4,
                                         .pLibraries = libs};
    VkGraphicsPipelineCreateInfo gp = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                       .layout = P.lay.graphics};   // flags 0: a fast link
    return create_graphics(gp, K_LINK, 0, "vkCreateGraphicsPipelines (fast link)");
}

void vkp_graphics_key(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                      char *out, size_t cap) {
    char vd[66], td[66], pd[66], ft[96];
    BT_TIME(BT_S3_KEY_DIGEST, cache_digest_text(vs, vd, sizeof vd); cache_digest_text(task, td, sizeof td);
            cache_digest_text(ps, pd, sizeof pd));
    BT_TIME(BT_S3_KEY_FORMATS, formats_text(formats, ft, sizeof ft));
    BT_TIME(BT_S3_KEY_PRINT,
    if (vs->shape && vs->shape->kind == NVN_STAGE_MS) {
        if (task) snprintf(out, cap, "g|t:%s%s|m:%s%s|p:%s%s|%s", task->key, td, vs->key, vd, ps->key, pd, ft);
        else snprintf(out, cap, "g|m:%s%s|p:%s%s|%s", vs->key, vd, ps->key, pd, ft);
    } else {
        if (task) rt_fatal("gfx: vulkan: a task stage is bound without a mesh stage");
        snprintf(out, cap, "g|%s%s|%s%s|%s", vs->key, vd, ps->key, pd, ft);
    });
}

static VkPipeline graphics_get(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                               const int *late) {
    if (formats->ncolor > VKP_MAX_COLOR) rt_fatal("gfx: vulkan: a pipeline with %u colour attachments", formats->ncolor);
    LinkCtx x = {.vs = vs, .task = task, .ps = ps, .formats = formats, .late = late};
    char key[KEY_MAX];
    vkp_graphics_key(vs, task, ps, formats, key, sizeof key);
    return obj_get(key, task ? task->key : vs->key, vs->key, ps->key, build_link, &x);
}

VkPipeline vkp_graphics_ex(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                           const int *late) {
    if (!P.open) rt_fatal("gfx: vulkan: vkp_graphics before vkp_open");
    return graphics_get(vs, task, ps, formats, late);
}

VkPipeline vkp_graphics_unopened(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats) {
    return graphics_get(vs, task, ps, formats, NULL);
}

VkPipeline vkp_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats) {
    return vkp_graphics_ex(vs, task, ps, formats, NULL);
}

#ifdef OOT2_BENCH_HOOKS
// BENCHMARK ONLY (tests/runtime/vk_replay_cost.c): the builder is taken for open without a device, and every pair the replay
// asks for that the table does not hold is entered as a ready pipeline with a made-up handle, at its first ask. The
// lookup below it (the key, the hash, the lock, the chain) is the one a run makes. `filler` unrelated ready objects
// stand for the population of a long session.
int vkp_bench_prime;
uint64_t vkp_bench_primed;
void vkp_bench_open(uint32_t filler) {
    P.open = 1;
    vkp_bench_prime = 1;
    for (uint32_t i = 0; i < filler; i++) {
        Obj *o = calloc(1, sizeof *o);
        if (!o) rt_fatal("bench: out of memory");
        snprintf(o->key, sizeof o->key, "x|filler-%08x|a-key-the-replay-never-asks-for", i * 2654435761u);
        o->pipe = (VkPipeline)(uintptr_t)(0x100000u + i);
        o->ready = 1;
        uint32_t h = hash_key(o->key);
        o->next = P.tab[h], P.tab[h] = o;
    }
}
#endif

#if defined(OOT2_BENCH_HOOKS) || defined(OOT2_BENCH_TIMERS)
// The table's side of vkp_find_graphics: the hash, the lock and the chain (split out so a timer can wrap it).
static VkPipeline find_ready(const char *key) {
    uint32_t h = hash_key(key);
    plat_mutex_lock(&P.lock);
    Obj *o = find_locked(h, key);
#ifdef OOT2_BENCH_HOOKS
    if (!o && vkp_bench_prime) {
        o = calloc(1, sizeof *o);
        if (!o) rt_fatal("bench: out of memory");
        snprintf(o->key, sizeof o->key, "%s", key);
        o->pipe = (VkPipeline)(uintptr_t)(0x1000u + ++vkp_bench_primed);
        o->ready = 1;
        o->next = P.tab[h], P.tab[h] = o;
    }
#endif
    VkPipeline p = o && o->ready ? o->pipe : VK_NULL_HANDLE;
    plat_mutex_unlock(&P.lock);
    return p;
}

VkPipeline vkp_find_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats) {
    if (!P.open) return VK_NULL_HANDLE;
    char key[KEY_MAX];
    BT_TIME(BT_S3_KEY, vkp_graphics_key(vs, task, ps, formats, key, sizeof key));
    VkPipeline p;
    BT_TIME(BT_S3_TABLE, p = find_ready(key));
    return p;
}
#else
VkPipeline vkp_find_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats) {
    if (!P.open) return VK_NULL_HANDLE;
    char key[KEY_MAX];
    vkp_graphics_key(vs, task, ps, formats, key, sizeof key);
    uint32_t h = hash_key(key);
    plat_mutex_lock(&P.lock);
    Obj *o = find_locked(h, key);
    VkPipeline p = o && o->ready ? o->pipe : VK_NULL_HANDLE;
    plat_mutex_unlock(&P.lock);
    return p;
}
#endif

// ---------- compute pipelines

typedef struct ComputeCtx {
    const VkpStage *stage;
    const int *late;
} ComputeCtx;

static VkPipeline build_compute(void *c) {
    const VkpStage *s = ((ComputeCtx *)c)->stage;
    VkpModule m;
    vkp_module(s, SHD_CS, &m);
    require_idle("a compute pipeline would be created", s->key, ((ComputeCtx *)c)->late);
    // A module whose HMMA runs on cooperative matrices calls the helper instead where the start-up probe refused the
    // native form (oot2/vk_hmma.h): the capabilities are checked on the module that is created.
    uint32_t *helper = NULL;
    size_t hwords = 0;
    if (!vkh_native()) {
        char why[256], shape[256];
        if (spv_hmma_fallback(m.code, m.bytes / 4, &helper, &hwords, why, sizeof why) < 0) {
            shape_text(s, shape, sizeof shape);
            rt_fatal("gfx: vulkan: stage %s cannot have its native HMMA turned into the helper: %s (%s)", s->key, why, shape);
        }
        if (helper) m.code = helper, m.bytes = (uint32_t)(hwords * 4);
    }
    // A module that still has its cooperative matrices (the native form) is created with the full-subgroups requirement,
    // which Vulkan asks of a pipeline with a SPIR-V 1.3 module that declares them (spv_needs_full_subgroups); the helper form
    // declares none and is created as every compute pipeline was.
    int full = check_capabilities(s, &m);
    // A compute module's workgroup memory starts as the null value of its type when the device was made with the
    // zero-initialization feature (spv_zero_workgroup says why): the module made from the pack's differs from it only in
    // that, and the pipeline cache keys on what is created.
    const uint32_t *code = m.code;
    size_t words = m.bytes / 4, zwords = 0;
    uint32_t *zeroed = NULL;
    if (P.zero_workgroup) {
        char why[256], shape[256];
        if (spv_zero_workgroup(code, words, &zeroed, &zwords, why, sizeof why) < 0) {
            shape_text(s, shape, sizeof shape);
            rt_fatal("gfx: vulkan: stage %s cannot have its workgroup memory zero-initialized: %s (%s)", s->key, why, shape);
        }
        if (zeroed) code = zeroed, words = zwords;
    }
    VkShaderModule sm = make_module(code, words);
    free(zeroed);
    free(helper);
    VkPipelineCreationFeedback fb = {0}, stage_fb = {0};
    VkPipelineCreationFeedbackCreateInfo fbi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                                .pPipelineCreationFeedback = &fb, .pipelineStageCreationFeedbackCount = 1,
                                                .pPipelineStageCreationFeedbacks = &stage_fb};
    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = &fbi,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .flags = full ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
        .layout = P.lay.compute};
    uint64_t t0 = plat_now_ns();
    VkPipeline p;
    VkResult r = vkCreateComputePipelines(gpu_vk.device, P.cache, 1, &ci, NULL, &p);
    uint64_t ns = plat_now_ns() - t0;
    if (r != VK_SUCCESS) gpu_vk_failed(r, "vkCreateComputePipelines");
    vkDestroyShaderModule(gpu_vk.device, sm, NULL);
    // counted from the create info that was handed to the driver, so that the count says what the driver was given
    note(K_COMPUTE, &fb, ns, (ci.stage.flags & VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT) ? CS_FULL_SUBGROUPS : 0);
    return p;
}

static void compute_key(const VkpStage *cs, char *out, size_t cap) {
    char dg[66];
    cache_digest_text(cs, dg, sizeof dg);
    snprintf(out, cap, "c|%s%s", cs->key, dg);
}

VkPipeline vkp_compute_ex(const VkpStage *cs, const int *late) {
    if (!P.open) rt_fatal("gfx: vulkan: vkp_compute before vkp_open");
    ComputeCtx x = {.stage = cs, .late = late};
    char key[KEY_MAX];
    compute_key(cs, key, sizeof key);
    return obj_get(key, cs->key, NULL, NULL, build_compute, &x);
}

VkPipeline vkp_compute(const VkpStage *cs) { return vkp_compute_ex(cs, NULL); }

VkPipeline vkp_find_compute(const VkpStage *cs) {
    if (!P.open) return VK_NULL_HANDLE;
    char key[KEY_MAX];
    compute_key(cs, key, sizeof key);
    uint32_t h = hash_key(key);
    plat_mutex_lock(&P.lock);
    Obj *o = find_locked(h, key);
    VkPipeline p = o && o->ready ? o->pipe : VK_NULL_HANDLE;
    plat_mutex_unlock(&P.lock);
    return p;
}

// ---------- warming (prepare's pipeline step, tools/pipebuild)

// The objects a launch asks for, built through the cache with the very functions the get calls use and let go at once:
// what the driver compiled stays in the cache, and the program holds no pipeline until it draws.
int vkp_warm(int what, const VkpStage *s, VkpWarm *out) {
    if (!P.open) rt_fatal("gfx: vulkan: vkp_warm before vkp_open");
    memset(out, 0, sizeof *out);
    VkpStats before, after;
    vkp_stats(&before);
    VkpModule m;
    VkPipeline p;
    int kind;
    if (what == VKP_WARM_VERTEX) {
        vkp_module(s, SHD_VS, &m);
        VertexCtx x = {.stage = s, .module = &m};
        p = build_vertex_library(&x);
        kind = K_LIBRARY;
    } else if (what == VKP_WARM_PIXEL) {
        vkp_module(s, SHD_PS, &m);
        FragmentCtx x = {.stage = s, .module = &m};
        p = build_fragment_library(&x);
        kind = K_LIBRARY;
    } else if (what == VKP_WARM_COMPUTE) {
        vkp_module(s, SHD_CS, &m);
        ComputeCtx x = {.stage = s};
        p = build_compute(&x);
        kind = K_COMPUTE;
    } else {
        rt_fatal("gfx: vulkan: vkp_warm of kind %d", what);
    }
    vkDestroyPipeline(gpu_vk.device, p, NULL);
    vkp_stats(&after);
    out->module_bytes = m.bytes;
    out->ns = after.build_ns[kind] - before.build_ns[kind];
    out->hit = kind == K_COMPUTE ? after.compute_hits > before.compute_hits
                                 : after.stage_library_hits > before.stage_library_hits;
    return 1;
}

uint64_t vkp_cache_size(void) {
    if (!P.open) return 0;
    size_t size = 0;
    GPU_VK_OK(vkGetPipelineCacheData(gpu_vk.device, P.cache, &size, NULL), "vkGetPipelineCacheData");
    return size;
}

// ---------- release and stats

void vkp_release(const char *key) {
    plat_mutex_lock(&P.lock);
    for (uint32_t b = 0; b < BUCKETS; b++) {
        Obj **at = &P.tab[b];
        while (*at) {
            Obj *o = *at;
            if (o->ready && (!strcmp(o->ref[0], key) || !strcmp(o->ref[1], key) || !strcmp(o->ref[2], key))) {
                *at = o->next;
                vkDestroyPipeline(gpu_vk.device, o->pipe, NULL);
                P.stats.resident--;
                free(o);
            } else {
                at = &o->next;
            }
        }
    }
    plat_mutex_unlock(&P.lock);
}

void vkp_stats(VkpStats *out) {
    plat_mutex_lock(&P.lock);
    *out = P.stats;
    out->map_missing = __atomic_load_n(&P.map_missing, __ATOMIC_RELAXED);
    out->map_unknown_rows = __atomic_load_n(&P.map_unknown_rows, __ATOMIC_RELAXED);
    plat_mutex_unlock(&P.lock);
}

// ---------- the pipeline cache

VkpCacheState vkp_cache_judge(const void *data, size_t size, const VkPhysicalDeviceProperties *props) {
    VkPipelineCacheHeaderVersionOne h;
    if (size < sizeof h) return VKP_CACHE_UNREADABLE;
    memcpy(&h, data, sizeof h);
    if (h.headerSize < sizeof h || h.headerSize > size || h.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
        return VKP_CACHE_UNREADABLE;
    if (h.vendorID != props->vendorID || h.deviceID != props->deviceID ||
        memcmp(h.pipelineCacheUUID, props->pipelineCacheUUID, VK_UUID_SIZE))
        return VKP_CACHE_OTHER_DRIVER;
    return VKP_CACHE_LOADED;
}

int vkp_read_cap;         // tests only: a cache file larger than this many bytes counts as too large to read (0: no cap)
int vkp_save_stall_ms;    // tests only: a save pauses this long after writing half of its file, so that two processes overlap
int vkp_open_errno;       // tests only: opening the cache file to read it fails with this errno (0: it opens as it does)
int vkp_open_transient;   // tests only: with vkp_open_errno, only the first open fails (then the errno is cleared)

#define OPEN_RETRY_NS 200000000ull   // an open that failed for another reason than "not there" is tried once more after this

// The cache file read whole, or NULL with *fail the cause: VKP_READ_ABSENT (not there: the C library maps the system's
// file-not-found and path-not-found to ENOENT), VKP_READ_OPEN (there, and cannot be opened; *err is the errno), VKP_READ_EMPTY,
// VKP_READ_TOO_LARGE (over the test cap, or the memory is not there) or VKP_READ_UNREADABLE (a seek or a read failed).
static void *read_whole(const char *path, size_t *size, int *fail, int *err) {
    FILE *f = NULL;
    *fail = VKP_READ_OK, *err = 0;
    errno = 0;
    if (vkp_open_errno) {
        errno = vkp_open_errno;
        if (vkp_open_transient) vkp_open_errno = 0;
    } else {
        f = fopen(path, "rb");
    }
    if (!f) {
        *err = errno;
        *fail = *err == ENOENT ? VKP_READ_ABSENT : VKP_READ_OPEN;
        return NULL;
    }
    void *data = NULL;
    *fail = VKP_READ_UNREADABLE;
    if (OFF_SEEK(f, 0, SEEK_END) == 0) {
        long long n = (long long)OFF_TELL(f);
        if (n == 0) {
            *fail = VKP_READ_EMPTY;
        } else if (n > 0 && vkp_read_cap && n > vkp_read_cap) {
            *fail = VKP_READ_TOO_LARGE;
        } else if (n > 0 && OFF_SEEK(f, 0, SEEK_SET) == 0) {
            if ((data = malloc((size_t)n)) == NULL) {
                *fail = VKP_READ_TOO_LARGE;
            } else if (fread(data, 1, (size_t)n, f) == (size_t)n) {
                *size = (size_t)n;
                *fail = VKP_READ_OK;
            } else {
                free(data), data = NULL;
            }
        }
    }
    fclose(f);
    return data;
}

void *vkp_cache_read(const char *path, size_t *size, int *fail, int *err) {
    void *data = read_whole(path, size, fail, err);
    // A file that is there and could not be opened is often a reader or a scanner holding it for a moment: once more, after
    // a short wait, before the whole run is marked as unable to save.
    if (!data && *fail == VKP_READ_OPEN) {
        plat_sleep_ns(OPEN_RETRY_NS);
        data = read_whole(path, size, fail, err);
    }
    return data;
}

const char *vkp_read_fail_text(int fail, int err, char *buf, size_t cap) {
    switch (fail) {
    case VKP_READ_OPEN: snprintf(buf, cap, "could not be opened at launch (%s)", strerror(err)); break;
    case VKP_READ_TOO_LARGE: snprintf(buf, cap, "could not be read at launch (too large for memory)"); break;
    case VKP_READ_UNREADABLE: snprintf(buf, cap, "could not be read at launch (a seek or a read failed)"); break;
    case VKP_READ_EMPTY: snprintf(buf, cap, "is empty"); break;
    case VKP_READ_ABSENT: snprintf(buf, cap, "is not there"); break;
    default: snprintf(buf, cap, "was read"); break;
    }
    return buf;
}

static void cache_open(const char *path) {
    void *data = NULL;
    size_t size = 0;
    uint64_t t0 = plat_now_ns();
    P.cache_state = VKP_CACHE_NONE;
    P.unreadable = 0;
    P.unreadable_err = 0;
    if (path) {
        snprintf(P.cache_path, sizeof P.cache_path, "%s", path);
        int fail = VKP_READ_OK, open_err = 0;
        char said[160];
        data = vkp_cache_read(path, &size, &fail, &open_err);
        if (fail == VKP_READ_EMPTY) {
            rt_log("gfx: vulkan: the pipeline cache %s is empty: starting empty", path);
        } else if (fail == VKP_READ_OPEN || fail == VKP_READ_TOO_LARGE || fail == VKP_READ_UNREADABLE) {
            // Only a file that is not there is absent. One that is there and was not read (the open failed twice, it is too
            // large, a read failed) is not held, and a save at the stop would replace what is in it with this run's part.
            P.unreadable = fail, P.unreadable_err = open_err;
            rt_log("gfx: vulkan: the pipeline cache %s %s: starting empty, and no save will replace it", path,
                   vkp_read_fail_text(fail, open_err, said, sizeof said));
        }
        if (data) {
            P.cache_state = vkp_cache_judge(data, size, &gpu_vk.facts.props);
            if (P.cache_state != VKP_CACHE_LOADED) {
                rt_log("gfx: vulkan: the pipeline cache %s is %s: starting empty", path,
                       P.cache_state == VKP_CACHE_OTHER_DRIVER ? "from another device or driver" : "not a pipeline cache");
                free(data), data = NULL, size = 0;
            }
        }
    }
    VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = data ? size : 0,
                                    .pInitialData = data};
    GPU_VK_OK(vkCreatePipelineCache(gpu_vk.device, &ci, NULL, &P.cache), "vkCreatePipelineCache");
    free(data);
    // The load is the file read whole into memory and handed to the driver: what a launch pays for a big cache.
    if (P.cache_state == VKP_CACHE_LOADED)
        rt_log("gfx: vulkan: the pipeline cache %s seeds %zu bytes (loaded in %.0f ms)", path, size,
               (double)(plat_now_ns() - t0) / 1e6);
}

// The folder a file is in, made when it is not there (one level: a junction's target is the caller's to have made).
static void make_parent(const char *path) {
    char dir[sizeof P.cache_path + 8];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = NULL;
    for (char *c = dir; *c; c++)
        if (*c == '/' || *c == '\\') slash = c;
    if (!slash || slash == dir) return;
    *slash = 0;
#ifdef _WIN32
    (void)_mkdir(dir);
#else
    (void)mkdir(dir, 0777);
#endif
}

// The cache's data into a buffer of its own. Another thread may be creating pipelines with the cache while this reads
// it (the save at a stop, or the timer's), so the size asked for first can be outgrown by the second call, which then
// says VK_INCOMPLETE and has written a part: the buffer is made larger than the size asked for, and a part is read again.
// *size is the bytes read; NULL, with the reason in why, when the driver or the memory fails.
static void *cache_data(VkPipelineCache cache, size_t *size, int fatal, char *why, size_t cap) {
    for (int tries = 0; tries < 4; tries++) {
        size_t asked = 0;
        VkResult r = vkGetPipelineCacheData(gpu_vk.device, cache, &asked, NULL);
        if (r != VK_SUCCESS) {
            if (fatal) gpu_vk_failed(r, "vkGetPipelineCacheData");
            if (why && cap) snprintf(why, cap, "vkGetPipelineCacheData failed: %s", gpu_vk_result_name(r));
            return NULL;
        }
        size_t room = asked + asked / 16 + (1u << 20);
        void *data = malloc(room);
        if (!data) {
            if (why && cap) snprintf(why, cap, "out of memory for %zu bytes of cache", room);
            return NULL;
        }
        size_t got = room;
        r = vkGetPipelineCacheData(gpu_vk.device, cache, &got, data);
        if (r == VK_SUCCESS) {
            *size = got;
            return data;
        }
        free(data);
        if (r != VK_INCOMPLETE) {
            if (fatal) gpu_vk_failed(r, "vkGetPipelineCacheData");
            if (why && cap) snprintf(why, cap, "vkGetPipelineCacheData failed: %s", gpu_vk_result_name(r));
            return NULL;
        }
    }
    if (why && cap) snprintf(why, cap, "the cache kept growing while it was read");
    return NULL;
}

#define MOVE_TRIES 5                    // the move of a save over the cache file: tries, with the wait between two of them
#define MOVE_WAIT_NS 250000000ull       // (4 waits: about a second)

// The temporary file a write goes through. The process id is in its name: two processes that save into one folder (a game
// left running on its timer and a new one, in a folder shared through a junction) never write the same file, so each
// rename puts a whole file in place and the later one wins.
static void temp_name(const char *path, char *out, size_t cap) { snprintf(out, cap, "%s.%u.tmp", path, plat_pid()); }

// One file of the cache's folder named `name`: removed when it is a temporary file of a save ("<cache file>.<id>.tmp") whose
// process is gone. A process that is alive may still be writing its own, and a number that cannot be read is left alone.
static void remove_if_dead_temp(const char *dir, const char *name, const char *base) {
    size_t blen = strlen(base);
    if (strncmp(name, base, blen) || name[blen] != '.') return;
    const char *digits = name + blen + 1;
    size_t n = strspn(digits, "0123456789");
    if (n == 0 || n > 10 || strcmp(digits + n, ".tmp")) return;
    unsigned long long id = strtoull(digits, NULL, 10);
    if (id == 0 || id > 0xffffffffull || id == plat_pid() || plat_pid_alive((unsigned)id)) return;
    char path[sizeof P.cache_path + 300];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (remove(path) == 0)
        rt_log("gfx: vulkan: removed %s: the temporary file of a save by process %llu, which is gone", path, id);
}

// A run that was killed in the middle of a save leaves its temporary file for good (each process writes its own name, so
// no later save overwrites it): the ones of processes that are gone are removed when the saving starts.
static void remove_dead_temps(void) {
    char dir[sizeof P.cache_path + 8];
    snprintf(dir, sizeof dir, "%s", P.cache_path);
    char *slash = NULL;
    for (char *c = dir; *c; c++)
        if (*c == '/' || *c == '\\') slash = c;
    const char *base = slash ? slash + 1 : dir;
    char folder[sizeof dir];
    if (slash) {
        *slash = 0;
        snprintf(folder, sizeof folder, "%s", slash == dir ? "/" : dir);
    } else {
        snprintf(folder, sizeof folder, ".");
    }
    char name[sizeof dir];
    snprintf(name, sizeof name, "%s", base);   // the base, kept apart from the folder text it shared a buffer with
#ifdef _WIN32
    char pattern[sizeof dir + 16];
    snprintf(pattern, sizeof pattern, "%s/%s.*.tmp", folder, name);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) remove_if_dead_temp(folder, fd.cFileName, name);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(folder);
    if (!d) return;
    for (struct dirent *e; (e = readdir(d)) != NULL;) remove_if_dead_temp(folder, e->d_name, name);
    closedir(d);
#endif
}

int vkp_move_tries;   // tests only: the tries the last vkp_move_over made
static int g_denied_lasting;   // a move was refused with "access denied" through the whole wait: later ones try once

// The temporary file moved over the cache file. On Windows another process that reads the cache (vkp_cache_read opens it
// without delete sharing) makes the replace fail with a sharing violation until it lets go, which is a read of
// milliseconds: the move is tried a few times over about a second before it is given up, since a refused move loses all of
// this run's growth. A refusal of "access denied" that lasts through that whole wait is not a reader (the file is read-only,
// or ours to no avail): it is named once, and later moves try once without waiting, instead of paying the second at every
// stop and every round of the timer.
int vkp_move_over(const char *tmp, const char *path) {
    int ok;
    vkp_move_tries = 0;
#ifdef _WIN32
    int lasting = __atomic_load_n(&g_denied_lasting, __ATOMIC_ACQUIRE);
    DWORD e = 0;
    for (int tries = 1;; tries++) {
        vkp_move_tries = tries;
        ok = MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING) != 0;
        e = ok ? 0 : GetLastError();
        int held = e == ERROR_SHARING_VIOLATION || e == ERROR_ACCESS_DENIED || e == ERROR_LOCK_VIOLATION;
        if (ok || !held || tries == MOVE_TRIES || (lasting && e == ERROR_ACCESS_DENIED)) break;
        plat_sleep_ns(MOVE_WAIT_NS);
    }
    if (ok) {
        __atomic_store_n(&g_denied_lasting, 0, __ATOMIC_RELEASE);
    } else if (e == ERROR_ACCESS_DENIED && !lasting) {
        __atomic_store_n(&g_denied_lasting, 1, __ATOMIC_RELEASE);
        rt_log("gfx: vulkan: the pipeline cache %s refused the replace with \"access denied\" through %d tries over %.1f s: it "
               "is not held by a reader (read-only, or not ours to replace); later saves try once, without waiting", path,
               MOVE_TRIES, (double)(MOVE_TRIES - 1) * (double)MOVE_WAIT_NS / 1e9);
    }
#else
    vkp_move_tries = 1;
    (void)g_denied_lasting;
    ok = rename(tmp, path) == 0;
#endif
    return ok;
}

// The cache's data written to path through a temporary file and a rename, so that an interrupted write leaves the old
// file whole; *bytes (when not NULL) is its size. A Vulkan failure is a named fatal when `fatal` (the builder's flush),
// else it is the reason in why (a save, which also runs at a stop, where a second fatal would end the run before the
// stop hook).
static int cache_write(VkPipelineCache cache, const char *path, size_t *bytes, int fatal, char *why, size_t cap) {
    size_t size = 0;
    void *data = cache_data(cache, &size, fatal, why, cap);
    if (!data) return 0;
    char tmp[sizeof P.cache_path + 24];
    temp_name(path, tmp, sizeof tmp);
    FILE *f = fopen(tmp, "wb");
    size_t first = vkp_save_stall_ms > 0 ? size / 2 : size;
    int ok = f && fwrite(data, 1, first, f) == first;
    if (ok && first < size) {   // tests only: the pause in which a second process saves into the same folder
        ok = fflush(f) == 0;
        plat_sleep_ns((uint64_t)vkp_save_stall_ms * 1000000ull);
        ok = ok && fwrite((const char *)data + first, 1, size - first, f) == size - first;
    }
    if (f && ok) ok = fflush(f) == 0 && SYNC_FILE(f) == 0;
    if (f && fclose(f) != 0) ok = 0;
    free(data);
    if (!ok) {
        if (why && cap) snprintf(why, cap, "cannot write %s", tmp);
        remove(tmp);
        return 0;
    }
    ok = vkp_move_over(tmp, path);
    if (!ok) {
        if (why && cap) snprintf(why, cap, "cannot move %s over %s", tmp, path);
        remove(tmp);
        return 0;
    }
    if (bytes) *bytes = size;
    return 1;
}

// What the cache held when it was loaded, or when it was last saved: a save is made when its size is not this one.
static size_t g_saved_bytes;

// Why a save must not replace the cache file, or NULL: the cache holds a part of what the file holds.
static const char *save_refusal(char *why, size_t cap) {
    if (__atomic_load_n(&P.flushed, __ATOMIC_ACQUIRE)) {
        snprintf(why, cap, "the cache was flushed to a segment (vkp_cache_flush), so it holds only what was built since "
                           "and a save would replace %s with that part", P.cache_path);
        return why;
    }
    if (P.unreadable) {
        char said[160];
        snprintf(why, cap, "the cache file %s %s, so this run started empty and a save would replace it with what this run "
                           "built", P.cache_path, vkp_read_fail_text(P.unreadable, P.unreadable_err, said, sizeof said));
        return why;
    }
    return NULL;
}

int vkp_cache_save(char *why, size_t cap) {
    if (!P.open || !P.cache_path[0]) {
        if (why && cap) snprintf(why, cap, "no pipeline cache file was named");
        return 0;
    }
    char refusal[512];
    if (save_refusal(refusal, sizeof refusal)) {
        if (why && cap) snprintf(why, cap, "%s", refusal);
        return 0;
    }
    make_parent(P.cache_path);
    size_t bytes = 0;
    if (!cache_write(P.cache, P.cache_path, &bytes, 0, why, cap)) return 0;
    g_saved_bytes = bytes;
    return 1;
}

int vkp_cache_flush(const char *path, size_t *bytes, char *why, size_t cap) {
    if (!P.open) {
        if (why && cap) snprintf(why, cap, "the pipeline builder is not open");
        return 0;
    }
    if (strlen(path) + 8 > sizeof P.cache_path) {
        if (why && cap) snprintf(why, cap, "the path %s is too long", path);
        return 0;
    }
    if (!cache_write(P.cache, path, bytes, 1, why, cap)) return 0;
    // From here on the cache holds what is built after this call and nothing else: the next flush writes a delta, and a
    // save (vkp_cache_save) is refused, since it would write that delta over the whole file.
    __atomic_store_n(&P.flushed, 1, __ATOMIC_RELEASE);
    VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache fresh;
    GPU_VK_OK(vkCreatePipelineCache(gpu_vk.device, &ci, NULL, &fresh), "vkCreatePipelineCache");
    vkDestroyPipelineCache(gpu_vk.device, P.cache, NULL);
    P.cache = fresh;
    return 1;
}

// ---------- the cache saves itself (oot2/vk_pipeline.h, "The game saves the cache it built by itself")

#define SAVE_SECONDS_DEFAULT 120u
#define SAVE_SECONDS_MAX 86400u
#define STOP_WAIT_NS 20000000000ull   // a stop waits this long for the timer's save before it gives up on its own
#define TIMER_STACK (1ull << 20)

static struct {
    int started, writer_added, stops;
    uint64_t every_ns;           // the timer's period; 0: no timer
    int saving;                  // one save at a time (atomic)
    int refusals_logged;         // one bit per reason (REFUSE_*) the timer has named (atomic); a stop always names its reason
    PlatMutex lock;              // the timer thread's wait
    PlatCond cond;
    int thread_stop, thread_running;
} S = {.lock = PLAT_MUTEX_INIT, .cond = PLAT_COND_INIT};

int vkp_save_settings(const char *save, const char *seconds, VkpSaveSettings *out, char *why, size_t cap) {
    out->enabled = 1;
    out->seconds = SAVE_SECONDS_DEFAULT;
    if (save && *save) {
        if (!strcmp(save, "off")) out->enabled = 0;
        else if (strcmp(save, "on")) {
            snprintf(why, cap, "OOT2_VK_CACHE_SAVE is \"%s\"; the values are on and off", save);
            return 0;
        }
    }
    if (seconds && *seconds) {
        char *end;
        errno = 0;
        unsigned long v = strtoul(seconds, &end, 10);
        if (*seconds < '0' || *seconds > '9' || *end || errno || v > SAVE_SECONDS_MAX) {
            snprintf(why, cap, "OOT2_VK_CACHE_SAVE_SECONDS is \"%s\"; the values are whole seconds from 0 to %u (0: no timer)",
                     seconds, SAVE_SECONDS_MAX);
            return 0;
        }
        out->seconds = (uint32_t)v;
    }
    return 1;
}

// The reasons a save is refused or fails. The timer names each of them once, not every round; a stop names its own.
enum { REFUSE_FLUSHED, REFUSE_UNREADABLE, REFUSE_NO_SIZE, REFUSE_FAILED, REFUSE_OPEN, REFUSE_TOO_LARGE };

// The reason a refused save is named under: one for each way the file was not read at launch.
static int refusal_reason(void) {
    switch (P.unreadable) {
    case 0: return REFUSE_FAILED;
    case VKP_READ_OPEN: return REFUSE_OPEN;
    case VKP_READ_TOO_LARGE: return REFUSE_TOO_LARGE;
    default: return REFUSE_UNREADABLE;
    }
}

static int name_it(int stopping, int reason) {
    int bit = 1 << reason;
    return !(__atomic_fetch_or(&S.refusals_logged, bit, __ATOMIC_ACQ_REL) & bit) || stopping;
}

// One save for `when` ("the stop (exit 70)", "the timer"). A stop is `stopping`: it waits for a save in progress, the timer
// skips its round. Nothing here stops the run: every outcome is a log line, and only a save that wrote something or was
// refused or failed says so (a round of the timer with nothing new is silent, and it names each reason of a refusal once).
static void autosave(const char *when, int stopping) {
    if (!__atomic_load_n(&S.started, __ATOMIC_ACQUIRE) || !P.open || !P.cache_path[0]) return;
    for (uint64_t waited = 0; __atomic_exchange_n(&S.saving, 1, __ATOMIC_ACQ_REL); waited += 10000000ull) {
        if (!stopping) return;
        if (waited >= STOP_WAIT_NS) {
            rt_log("gfx: vulkan: the pipeline cache was not saved at %s: a save was still running after %.0f s", when,
                   (double)STOP_WAIT_NS / 1e9);
            return;
        }
        plat_sleep_ns(10000000ull);
    }
    uint64_t t0 = plat_now_ns();
    char why[512] = "";
    size_t now = 0, before = g_saved_bytes;
    VkResult r = vkGetPipelineCacheData(gpu_vk.device, P.cache, &now, NULL);
    if (__atomic_load_n(&P.flushed, __ATOMIC_ACQUIRE)) {
        vkp_cache_save(why, sizeof why);   // refused by name
        if (name_it(stopping, REFUSE_FLUSHED)) rt_log("gfx: vulkan: the pipeline cache was NOT saved at %s: %s", when, why);
    } else if (r != VK_SUCCESS) {
        if (name_it(stopping, REFUSE_NO_SIZE))
            rt_log("gfx: vulkan: the pipeline cache was NOT saved at %s: vkGetPipelineCacheData failed: %s", when,
                   gpu_vk_result_name(r));
    } else if (now == g_saved_bytes) {
        if (stopping)
            rt_log("gfx: vulkan: the pipeline cache is not saved at %s: it holds %zu bytes, as when it was loaded or last saved",
                   when, now);
    } else if (vkp_cache_save(why, sizeof why)) {
        rt_log("gfx: vulkan: the pipeline cache was saved to %s at %s: %zu bytes (%zu when loaded or last saved) in %.0f ms",
               P.cache_path, when, g_saved_bytes, before, (double)(plat_now_ns() - t0) / 1e6);
    } else if (name_it(stopping, refusal_reason())) {
        rt_log("gfx: vulkan: the pipeline cache was NOT saved at %s: %s", when, why);
    }
    __atomic_store_n(&S.saving, 0, __ATOMIC_RELEASE);
}

static void stop_writer(int code, const char *line) {
    (void)line;
    if (!S.stops) return;
    char when[64];
    snprintf(when, sizeof when, "the stop (exit %d)", code);
    autosave(when, 1);
}

static void timer_main(void *arg) {
    (void)arg;
    plat_mutex_lock(&S.lock);
    uint64_t next = plat_now_ns() + S.every_ns;
    while (!S.thread_stop) {
        uint64_t now = plat_now_ns();
        if (now < next) {
            plat_cond_wait_ns(&S.cond, &S.lock, next - now);
            continue;
        }
        plat_mutex_unlock(&S.lock);
        autosave("the timer", 0);
        plat_mutex_lock(&S.lock);
        next = plat_now_ns() + S.every_ns;
    }
    S.thread_running = 0;
    plat_cond_broadcast(&S.cond);
    plat_mutex_unlock(&S.lock);
}

void vkp_cache_autosave_start(const VkpSaveSettings *settings) {
    if (S.started || !P.open || !P.cache_path[0]) return;
    if (!settings->enabled) {
        rt_log("gfx: vulkan: OOT2_VK_CACHE_SAVE off: the pipeline cache is not saved");
        return;
    }
    size_t base = 0;
    if (vkGetPipelineCacheData(gpu_vk.device, P.cache, &base, NULL) == VK_SUCCESS) g_saved_bytes = base;
    // A temporary file of this very name is left by an earlier process that had this id and ended in the middle of a save.
    char stale[sizeof P.cache_path + 24];
    temp_name(P.cache_path, stale, sizeof stale);
    remove(stale);
    remove_dead_temps();
    S.refusals_logged = 0;
    S.stops = 1;
    S.every_ns = (uint64_t)settings->seconds * 1000000000ull;
    if (!S.writer_added) {
        rt_add_stop_writer(stop_writer);
        S.writer_added = 1;
    }
    __atomic_store_n(&S.started, 1, __ATOMIC_RELEASE);
    int timer = 0;
    if (S.every_ns) {
        S.thread_stop = 0;
        S.thread_running = 1;
        timer = plat_thread_start(timer_main, NULL, TIMER_STACK) != NULL;
        if (!timer) {
            S.thread_running = 0;
            rt_log("gfx: vulkan: the pipeline cache's timer could not start a thread: it is saved at the stop only");
        }
    }
    if (timer) rt_log("gfx: vulkan: the pipeline cache is saved to %s at every stop and every %u s", P.cache_path, settings->seconds);
    else rt_log("gfx: vulkan: the pipeline cache is saved to %s at every stop", P.cache_path);
}

// vkp_close: the timer ends first (a save in progress is waited for), so that nothing reads a cache that is destroyed.
static void autosave_end(void) {
    plat_mutex_lock(&S.lock);
    S.thread_stop = 1;
    plat_cond_broadcast(&S.cond);
    while (S.thread_running) plat_cond_wait(&S.cond, &S.lock);
    plat_mutex_unlock(&S.lock);
    S.stops = 0;
    S.every_ns = 0;
    __atomic_store_n(&S.started, 0, __ATOMIC_RELEASE);
}

VkpCacheState vkp_cache_state(void) { return P.cache_state; }

int vkp_zero_workgroup(void) { return P.zero_workgroup; }

// ---------- open and close

void vkp_open(const char *cache_path) {
    if (P.open) return;
    if (!gpu_vk.device) rt_fatal("gfx: vulkan: vkp_open before gpu_vk_open");
    P.flushed = 0;
    g_saved_bytes = 0;
    vkp_layouts_create(&P.lay);
    vkc_list(&gpu_vk.facts, &P.caps);
    Oot2VkFacts en;
    oot2_vk_enable(&gpu_vk.facts, &en);
    P.zero_workgroup = en.f13.shaderZeroInitializeWorkgroupMemory == VK_TRUE;
    vkh_probe();   // whether the modules' native HMMA runs on this device (oot2/vk_hmma.h), before any module is built
    cache_open(cache_path);
    PipeInfo pi;
    pipe_info(&pi, PART_VERTEX_INPUT, VK_NULL_HANDLE, NULL);
    P.open = 1;   // create_graphics reads the cache and the layouts
    P.vi = create_graphics(pi.gp, K_LIBRARY, 0, "vkCreateGraphicsPipelines (vertex input library)");
}

void vkp_close(void) {
    if (!P.open) return;
    autosave_end();
    gpu_sync_finish();
    for (uint32_t b = 0; b < BUCKETS; b++) {
        for (Obj *o = P.tab[b], *next; o; o = next) {
            next = o->next;
            if (o->ready) vkDestroyPipeline(gpu_vk.device, o->pipe, NULL);
            free(o);
        }
        P.tab[b] = NULL;
    }
    vkDestroyPipeline(gpu_vk.device, P.vi, NULL);
    vkDestroyPipelineCache(gpu_vk.device, P.cache, NULL);
    vkp_layouts_destroy(&P.lay);
    memset(&P.stats, 0, sizeof P.stats);
    __atomic_store_n(&P.map_missing, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&P.map_unknown_rows, 0, __ATOMIC_RELAXED);
    P.vi = VK_NULL_HANDLE, P.cache = VK_NULL_HANDLE;
    P.cache_path[0] = 0;
    P.cache_state = VKP_CACHE_NONE;
    P.flushed = 0;
    P.unreadable = 0, P.unreadable_err = 0;
    P.open = 0;
}

// The prepared folder's files, under the names tools/prepare writes: the translate step's base pack, the ledger step's
// variants pack and full-ledger-digest map (either may be absent for older prepared folders), and the pipelines step's
// cache (absent when the step was skipped, which leaves every first use a miss, served by oot2/vk_miss.h). The run then
// saves the cache it built into the same file (vkp_cache_autosave_start), so a folder nothing prepared is warm from its
// second run.
void gpu_vk_open_game(const char *folder) {
    char base[600], variants[600], variant_map[600], cache[600], why[512];
    if (snprintf(base, sizeof base, "%s/shaders/shader-base.bin", folder) >= (int)sizeof base ||
        snprintf(variants, sizeof variants, "%s/shaders/shader-variants.bin", folder) >= (int)sizeof variants ||
        snprintf(variant_map, sizeof variant_map, "%s/shaders/shader-variant-map.bin", folder) >= (int)sizeof variant_map ||
        snprintf(cache, sizeof cache, "%s/pipelines/pipeline.cache", folder) >= (int)sizeof cache)
        rt_fatal("gfx: vulkan: the game folder's path is too long: %s", folder);
    VkpSaveSettings save;
    if (!vkp_save_settings(getenv("OOT2_VK_CACHE_SAVE"), getenv("OOT2_VK_CACHE_SAVE_SECONDS"), &save, why, sizeof why))
        rt_fatal("gfx: vulkan: %s", why);
    FILE *f = fopen(variants, "rb");
    FILE *m = fopen(variant_map, "rb");
    if (f) fclose(f);
    if (m) fclose(m);
    if (!vkp_store_open_map(base, f ? variants : NULL, m ? variant_map : NULL, why, sizeof why))
        rt_fatal("gfx: vulkan: the shader packs of the game folder cannot be opened (run oot2 prepare): %s", why);
    vkp_open(cache);
    rt_log("gfx: vulkan: shader packs %s%s%s%s%s; pipeline cache %s", base, f ? " and " : " (no variants pack)",
           f ? variants : "", m ? " with " : " (no variant map)", m ? variant_map : "",
           P.cache_state == VKP_CACHE_LOADED ? "seeded" :
           P.cache_state == VKP_CACHE_NONE ? "empty (no file)" : "empty (refused)");
    vkp_cache_autosave_start(&save);
}

const VkpLayouts *vkp_layouts(void) { return &P.lay; }
