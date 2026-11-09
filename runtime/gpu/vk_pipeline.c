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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>
// 64-bit file offsets and a flush to the disk: a pipeline cache of a whole game passes 2 GiB, which a 32-bit `long`
// (Windows' ftell) cannot size, and a cache file renamed into place must not be half written after a power cut.
#define OFF_SEEK _fseeki64
#define OFF_TELL _ftelli64
#define SYNC_FILE(f) _commit(_fileno(f))
#else
#include <unistd.h>
#define OFF_SEEK fseeko
#define OFF_TELL ftello
#define SYNC_FILE(f) fsync(fileno(f))
#endif

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/spirv.h"
#include "oot2/vk_caps.h"
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

int vkp_unpad;   // tests only: leave the vertex modules as translated (the counter-run of the padding rule)

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

static void check_capabilities(const VkpStage *s, const VkpModule *m) {
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
}

// What a creation tells note: bit 0 the module was padded, bit 1 the library holds a shader stage.
enum { LIB_PADDED = 1, LIB_SHADER = 2 };

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
    const int *late;
} MeshCtx;

static VkPipeline create_graphics(VkGraphicsPipelineCreateInfo ci, int kind, int flags, const char *what);

static VkPipeline build_mesh_library(void *c) {
    MeshCtx *x = c;
    require_idle("a mesh pre-rasterization library would be created", x->mesh->key, x->late);
    check_capabilities(x->mesh, x->mesh_module);
    VkShaderModule task = VK_NULL_HANDLE;
    if (x->task) {
        check_capabilities(x->task, x->task_module);
        task = make_module(x->task_module->code, x->task_module->bytes / 4);
    }
    VkShaderModule mesh = make_module(x->mesh_module->code, x->mesh_module->bytes / 4);
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
    VkPipeline p = create_graphics(pi.gp, K_LIBRARY, LIB_SHADER, "vkCreateGraphicsPipelines (mesh pre-rasterization library)");
    if (task) vkDestroyShaderModule(gpu_vk.device, task, NULL);
    vkDestroyShaderModule(gpu_vk.device, mesh, NULL);
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
        char why[256], md[66], td[66], pd[66], ft[96], ranges[MAX_PADS * 16 + 8];
        ShdPad gaps[MAX_PADS];
        int missing = shd_mesh_link(&mm.desc, &pm.desc, gaps, MAX_PADS, why, sizeof why);
        if (missing < 0)
            rt_fatal("gfx: vulkan: mesh stage %s cannot be linked to pixel stage %s: %s", x->vs->key, x->ps->key, why);
        if (missing) {
            shd_pad_text(gaps, missing, ranges, sizeof ranges);
            rt_fatal("gfx: vulkan: mesh stage %s does not write %d interface ranges read by pixel stage %s "
                     "(location:component:components:type %s)", x->vs->key, missing, x->ps->key, ranges);
        }

        MeshCtx mx = {.task = x->task, .mesh = x->vs, .task_module = x->task ? &tm : NULL, .mesh_module = &mm,
                      .late = x->late};
        cache_digest_text(x->vs, md, sizeof md);
        if (x->task) {
            cache_digest_text(x->task, td, sizeof td);
            snprintf(x->vkey, sizeof x->vkey, "m|t:%s%s|m:%s%s", x->task->key, td, x->vs->key, md);
        } else snprintf(x->vkey, sizeof x->vkey, "m|m:%s%s", x->vs->key, md);
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
    cache_digest_text(vs, vd, sizeof vd);
    cache_digest_text(task, td, sizeof td);
    cache_digest_text(ps, pd, sizeof pd);
    formats_text(formats, ft, sizeof ft);
    if (vs->shape && vs->shape->kind == NVN_STAGE_MS) {
        if (task) snprintf(out, cap, "g|t:%s%s|m:%s%s|p:%s%s|%s", task->key, td, vs->key, vd, ps->key, pd, ft);
        else snprintf(out, cap, "g|m:%s%s|p:%s%s|%s", vs->key, vd, ps->key, pd, ft);
    } else {
        if (task) rt_fatal("gfx: vulkan: a task stage is bound without a mesh stage");
        snprintf(out, cap, "g|%s%s|%s%s|%s", vs->key, vd, ps->key, pd, ft);
    }
}

VkPipeline vkp_graphics_ex(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                           const int *late) {
    if (!P.open) rt_fatal("gfx: vulkan: vkp_graphics before vkp_open");
    if (formats->ncolor > VKP_MAX_COLOR) rt_fatal("gfx: vulkan: a pipeline with %u colour attachments", formats->ncolor);
    LinkCtx x = {.vs = vs, .task = task, .ps = ps, .formats = formats, .late = late};
    char key[KEY_MAX];
    vkp_graphics_key(vs, task, ps, formats, key, sizeof key);
    return obj_get(key, task ? task->key : vs->key, vs->key, ps->key, build_link, &x);
}

VkPipeline vkp_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats) {
    return vkp_graphics_ex(vs, task, ps, formats, NULL);
}

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
    check_capabilities(s, &m);
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
    VkPipelineCreationFeedback fb = {0}, stage_fb = {0};
    VkPipelineCreationFeedbackCreateInfo fbi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO,
                                                .pPipelineCreationFeedback = &fb, .pipelineStageCreationFeedbackCount = 1,
                                                .pPipelineStageCreationFeedbacks = &stage_fb};
    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = &fbi,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = sm, .pName = "main"},
        .layout = P.lay.compute};
    uint64_t t0 = plat_now_ns();
    VkPipeline p;
    VkResult r = vkCreateComputePipelines(gpu_vk.device, P.cache, 1, &ci, NULL, &p);
    uint64_t ns = plat_now_ns() - t0;
    if (r != VK_SUCCESS) gpu_vk_failed(r, "vkCreateComputePipelines");
    vkDestroyShaderModule(gpu_vk.device, sm, NULL);
    note(K_COMPUTE, &fb, ns, 0);
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

static void *read_whole(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    void *data = NULL;
    if (OFF_SEEK(f, 0, SEEK_END) == 0) {
        long long n = (long long)OFF_TELL(f);
        if (n > 0 && OFF_SEEK(f, 0, SEEK_SET) == 0 && (data = malloc((size_t)n)) != NULL) {
            if (fread(data, 1, (size_t)n, f) == (size_t)n) *size = (size_t)n;
            else free(data), data = NULL;
        }
    }
    fclose(f);
    return data;
}

static void cache_open(const char *path) {
    void *data = NULL;
    size_t size = 0;
    P.cache_state = VKP_CACHE_NONE;
    if (path) {
        snprintf(P.cache_path, sizeof P.cache_path, "%s", path);
        data = read_whole(path, &size);
        if (!data) {
            FILE *there = fopen(path, "rb");
            if (there) {   // a file that is there and cannot be read is not the same as no file: say so
                fclose(there);
                rt_log("gfx: vulkan: the pipeline cache %s could not be read (empty, or too large for memory): starting empty", path);
            }
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
    if (P.cache_state == VKP_CACHE_LOADED) rt_log("gfx: vulkan: the pipeline cache %s seeds %zu bytes", path, size);
}

// The cache's data written to path through a temporary file and a rename, so that an interrupted write leaves the old
// file whole; *bytes (when not NULL) is its size.
static int cache_write(VkPipelineCache cache, const char *path, size_t *bytes, char *why, size_t cap) {
    size_t size = 0;
    GPU_VK_OK(vkGetPipelineCacheData(gpu_vk.device, cache, &size, NULL), "vkGetPipelineCacheData");
    void *data = malloc(size ? size : 1);
    if (!data) {
        if (why && cap) snprintf(why, cap, "out of memory for %zu bytes of cache", size);
        return 0;
    }
    VkResult r = vkGetPipelineCacheData(gpu_vk.device, cache, &size, data);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) gpu_vk_failed(r, "vkGetPipelineCacheData");
    char tmp[sizeof P.cache_path + 8];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    int ok = f && fwrite(data, 1, size, f) == size;
    if (f && ok) ok = fflush(f) == 0 && SYNC_FILE(f) == 0;
    if (f && fclose(f) != 0) ok = 0;
    free(data);
    if (!ok) {
        if (why && cap) snprintf(why, cap, "cannot write %s", tmp);
        remove(tmp);
        return 0;
    }
#ifdef _WIN32
    ok = MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING) != 0;
#else
    ok = rename(tmp, path) == 0;
#endif
    if (!ok) {
        if (why && cap) snprintf(why, cap, "cannot move %s over %s", tmp, path);
        remove(tmp);
        return 0;
    }
    if (bytes) *bytes = size;
    return 1;
}

int vkp_cache_save(char *why, size_t cap) {
    if (!P.open || !P.cache_path[0]) {
        if (why && cap) snprintf(why, cap, "no pipeline cache file was named");
        return 0;
    }
    return cache_write(P.cache, P.cache_path, NULL, why, cap);
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
    if (!cache_write(P.cache, path, bytes, why, cap)) return 0;
    // From here on the cache holds what is built after this call and nothing else: the next flush writes a delta.
    VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache fresh;
    GPU_VK_OK(vkCreatePipelineCache(gpu_vk.device, &ci, NULL, &fresh), "vkCreatePipelineCache");
    vkDestroyPipelineCache(gpu_vk.device, P.cache, NULL);
    P.cache = fresh;
    return 1;
}

VkpCacheState vkp_cache_state(void) { return P.cache_state; }

int vkp_zero_workgroup(void) { return P.zero_workgroup; }

// ---------- open and close

void vkp_open(const char *cache_path) {
    if (P.open) return;
    if (!gpu_vk.device) rt_fatal("gfx: vulkan: vkp_open before gpu_vk_open");
    vkp_layouts_create(&P.lay);
    vkc_list(&gpu_vk.facts, &P.caps);
    Oot2VkFacts en;
    oot2_vk_enable(&gpu_vk.facts, &en);
    P.zero_workgroup = en.f13.shaderZeroInitializeWorkgroupMemory == VK_TRUE;
    cache_open(cache_path);
    PipeInfo pi;
    pipe_info(&pi, PART_VERTEX_INPUT, VK_NULL_HANDLE, NULL);
    P.open = 1;   // create_graphics reads the cache and the layouts
    P.vi = create_graphics(pi.gp, K_LIBRARY, 0, "vkCreateGraphicsPipelines (vertex input library)");
}

void vkp_close(void) {
    if (!P.open) return;
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
    P.open = 0;
}

// The prepared folder's files, under the names tools/prepare writes: the translate step's base pack, the ledger step's
// variants pack and full-ledger-digest map (either may be absent for older prepared folders), and the pipelines step's
// cache (absent when the step was skipped, which leaves every first use a miss, served by oot2/vk_miss.h).
void gpu_vk_open_game(const char *folder) {
    char base[600], variants[600], variant_map[600], cache[600], why[512];
    if (snprintf(base, sizeof base, "%s/shaders/shader-base.bin", folder) >= (int)sizeof base ||
        snprintf(variants, sizeof variants, "%s/shaders/shader-variants.bin", folder) >= (int)sizeof variants ||
        snprintf(variant_map, sizeof variant_map, "%s/shaders/shader-variant-map.bin", folder) >= (int)sizeof variant_map ||
        snprintf(cache, sizeof cache, "%s/pipelines/pipeline.cache", folder) >= (int)sizeof cache)
        rt_fatal("gfx: vulkan: the game folder's path is too long: %s", folder);
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
}

const VkpLayouts *vkp_layouts(void) { return &P.lay; }
