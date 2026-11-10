// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the pipeline builder (runtime/gpu/vk_pipeline.c and its parts), driven by tests/test_runtime_vk_pipeline.py:
// `vk_pipeline_test <case> args...`, one case per process.
//
// Cases that need no device (continuous integration runs them):
//   tables                       the layout's tables and the dynamic state lists, printed
//   keys                         vertex, mesh and task-mesh pipelines have distinct cache keys
//   desc FILE                    a description parsed: stage, fit, interface by location, or the refusal
//   pad VS_DESC PS_DESC          the padding a vertex stage needs for a pixel stage
//   meshlink MS_DESC PS_DESC     the rule a mesh stage is linked to a pixel stage by: the ranges it does not write, or
//                                the refusal
//   meshlibkey MS_DESC PS_DESC...   for each pixel stage: the pad set its link to the mesh stage needs, the key of the mesh
//                                library for it (without and with a task stage) and the runs handed to the rewrite
//   meshpayload FILE             the task payload bytes a mesh stage's description declares
//   binds FILE                   a description's bindings parsed (shd_parse_binds), or the refusal
//   spvdecl MODULE               the capabilities and extensions a module declares
//   spvfull MODULE               whether a compute pipeline of the module needs the full-subgroups requirement
//                                (spv_needs_full_subgroups): "full_subgroups 1" when it declares cooperative matrices
//   spvpad IN OUT SPEC           a module with outputs added (SPEC "location:component:components:type,...")
//   spvmeshpad IN OUT SPEC [MAX_BYTES MAX_COMPONENTS [MAX_PAYLOAD_AND_OUTPUT PAYLOAD_BYTES]]   a mesh module with outputs
//                                added and stored (a ":p" after a run: read per primitive), held to the device's mesh output
//                                limits when they are given, the output memory and the task payload together to the last
//   spvzero IN OUT               a module with its workgroup variables given the null initializer: how many, or why not
//   spvhmma IN OUT               a module's native HMMA turned into the helper (spv_hmma_fallback): unchanged, the new size,
//                                or why not
//   capcheck MODULE [storage-image-indexing] [drop CAP]   a module judged against the capabilities of a device that has
//                                every item (with the optional feature when the argument names it; a capability may be dropped)
//   optional                     the required list's ids and count, and for facts without and with the optional features:
//                                the items the device has, the features enabled and whether capability 5309 is allowed
//   cachejudge                   cache file headers judged for a device
//   savesettings                 OOT2_VK_CACHE_SAVE and OOT2_VK_CACHE_SAVE_SECONDS judged: the settings, or the refusal
//   cacheread PATH [readcap=N] [openerr=E] [transient]   the cache file read as the launch reads it: the cause, and the wait
//   cachemove TMP PATH [heal]    a temporary file moved over PATH three times: tries and wait of each (Windows)
//   store BASE VARIANTS KEY KIND SHAPE   a stage looked up in a pack of packs (VARIANTS "-" for none)
//   storeopen BASE VARIANTS      the packs opened: "ok" or the refusal
//   devbinds BASE KEY N          N stage records of one key asked for their bindings through the replay's device ops
//                                (the cache whose bucket index must stay inside its table)
//   linkchoice BASE VKEY PKEY KIND   a pair build_link refuses, asked for from a builder never opened: the fatal names
//                                the rule it linked by (KIND "ms": the mesh rule, "vs": the padding rule)
// Cases that need a device with every item of oot2/vk_required.h, on a synthetic pack of GLSL modules:
//   usable                       how many devices have every item (exit 3 when there is no loader or device)
//   layouts                      the layouts are made, valid under validation, and destroyed
//   synth_draw PACK [unpad]      a pair linked for two formats and drawn; the padded input reads (0, 0, 0, 1)
//   synth_meshpad PACK           one mesh stage and four pixel stages (one reading a location once for each primitive) linked:
//                                the padded mesh library is made once for each pad set, whatever the pixel stage and the format
//   synth_compute PACK           a compute pipeline dispatched
//   synth_zeroinit PACK          a compute module reads workgroup memory it never wrote, after a dispatch that filled the
//                                device's workgroup memory with ones: it reads zeros
//   synth_inflight PACK          a pipeline asked for with a submission in flight: a named fatal
//   synth_cache PACK FILE PHASE  cold (build and save), warm (build from the file) or tamper (a driver change)
//   synth_save FOLDER MODE [OPTION...]   a game folder opened the way the game opens it (gpu_vk_open_game: FOLDER/shaders/shader-base.bin,
//                                FOLDER/pipelines/pipeline.cache), the pipelines made, and the run ended by MODE: fatal (a
//                                named stop, exit 70), exit (the guest's ExitProcess, exit 0), kill (no stop hook, as a
//                                process killed from outside), timer (waits for the cache's own timer to write the file,
//                                then ends without a stop hook) or flushed (the cache flushed to a segment as a builder
//                                does, a save refused, then a named stop). Options: more (one more compute pipeline), stale (a
//                                temporary file of this process's name is planted before the start, and must be gone after
//                                it), stall=MS (a save pauses MS after half of its file), readcap=N (a cache file over N
//                                bytes counts as unreadable), openerr=N (opening the cache file at launch fails with errno N),
//                                flush (mode wait only: the cache is flushed to a segment first), wait=MS (mode wait: the
//                                timer's rounds go by for MS, then the run ends with no stop hook)
//   synth_release PACK           a stage's objects released and made again
//   synth_threads PACK           eight threads ask for one pipeline: it is made once
//   synth_absent PACK            a stage the pack lacks: a named fatal with its shape
//   hmma_probe [FAULT]           the device and the builder opened and closed: the start-up probe's decision on HMMA
//                                (oot2/vk_hmma.h), with a fault in its read-back when named (lengths, helper, sums)
//   hmma_build PACK KEY          a translated compute module with HMMA built in the form the probe chose, and how many
//                                pipelines were created with the full-subgroups requirement (1 for the native form, 0 for
//                                the helper)
// Cases on the player's own pack (the desktop, never in continuous integration):
//   firstdraw DRAW PACK VKEY PKEY OUT   tools/firstdraw's draw, built from the pack through the builder
//   corpus_pairs PACK PAIRS THREADS CACHE [save]   pairs built and linked for two formats
//   corpus_compute PACK KEYS THREADS CACHE         compute pipelines built
//   caps PACK KEYS [drop CAP]                      every module's declarations against the required list
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#endif

#include "volk.h"

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/nvn_program.h"
#include "oot2/runtime.h"
#include "oot2/shader_desc.h"
#include "oot2/spirv.h"
#include "oot2/vk_caps.h"
#include "oot2/vk_hmma.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_replay.h"
#include "plat.h"
#include "vk_draw.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3

#ifdef _WIN32
#define STRTOK strtok_s
#else
#define STRTOK strtok_r
#endif

static void *read_all(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = malloc(n > 0 ? (size_t)n : 1);
    CHECK(p && fread(p, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    *size = (size_t)n;
    return p;
}

// ---------- tables

static void print_dynamic(const char *variant, const Oot2VkFacts *f) {
    static const VkGraphicsPipelineLibraryFlagsEXT parts[4] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT};
    static const char *names[4] = {"vertex_input", "pre_rasterization", "fragment", "output"};
    for (int i = 0; i < 4; i++) {
        VkDynamicState d[VKP_MAX_DYNAMIC];
        uint32_t n = vkp_dynamic_states(parts[i], f, d);
        printf("dyn %s %s %u", variant, names[i], n);
        for (uint32_t k = 0; k < n; k++) printf(" %d", (int)d[k]);
        printf("\n");
    }
}

static void case_tables(void) {
    VkDescriptorSetLayoutBinding b[VKP_SET1_MAX];
    VkDescriptorBindingFlags fl[2];
    uint32_t n = vkp_set0_bindings(b, fl);
    for (uint32_t i = 0; i < n; i++)
        printf("set0 %u %d %u %#x %#x\n", b[i].binding, (int)b[i].descriptorType, b[i].descriptorCount, b[i].stageFlags, fl[i]);
    for (int layout = 0; layout < 2; layout++) {
        n = vkp_set1_bindings(layout, b);
        for (uint32_t i = 0; i < n; i++)
            printf("set1 %s %u %d %#x\n", layout == VKP_GRAPHICS ? "graphics" : "compute", b[i].binding,
                   (int)b[i].descriptorType, b[i].stageFlags);
        VkPushConstantRange r = vkp_push_range(layout);
        printf("push %s %#x %u %u\n", layout == VKP_GRAPHICS ? "graphics" : "compute", r.stageFlags, r.offset, r.size);
    }
    n = vkp_set2_bindings(b);
    for (uint32_t i = 0; i < n; i++) printf("set2 %u %d %#x\n", b[i].binding, (int)b[i].descriptorType, b[i].stageFlags);
    Oot2VkFacts f;
    memset(&f, 0, sizeof f);
    print_dynamic("none", &f);
    f.eds3 = (VkPhysicalDeviceExtendedDynamicState3FeaturesEXT){
        .extendedDynamicState3PolygonMode = 1, .extendedDynamicState3DepthClampEnable = 1,
        .extendedDynamicState3LogicOpEnable = 1, .extendedDynamicState3ColorBlendEnable = 1,
        .extendedDynamicState3ColorBlendEquation = 1, .extendedDynamicState3ColorWriteMask = 1,
        .extendedDynamicState3DepthClipNegativeOneToOne = 1};
    print_dynamic("all", &f);
    f.eds3.extendedDynamicState3PolygonMode = 0;
    print_dynamic("nopoly", &f);
    printf("ok tables\n");
}

static void case_keys(void) {
    nvn_stage vs_shape = {.kind = NVN_STAGE_VS}, task_shape = {.kind = NVN_STAGE_TS}, mesh_shape = {.kind = NVN_STAGE_MS};
    VkpStage vs = {.key = "6100000000000001.00000000", .shape = &vs_shape};
    VkpStage task = {.key = "6100000000000002.00000000", .shape = &task_shape};
    VkpStage mesh = {.key = "6100000000000003.00000000", .shape = &mesh_shape};
    VkpStage ps = {.key = "6100000000000004.00000000"};
    VkpFormats formats = {1, {VK_FORMAT_R8G8B8A8_UNORM}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    char vertex_key[256], mesh_key[256], task_mesh_key[256];
    vkp_graphics_key(&vs, NULL, &ps, &formats, vertex_key, sizeof vertex_key);
    vkp_graphics_key(&mesh, NULL, &ps, &formats, mesh_key, sizeof mesh_key);
    vkp_graphics_key(&mesh, &task, &ps, &formats, task_mesh_key, sizeof task_mesh_key);
    CHECK(strcmp(vertex_key, mesh_key) && strcmp(mesh_key, task_mesh_key) && strcmp(vertex_key, task_mesh_key));
    printf("vertex %s\nmesh %s\ntask-mesh %s\nok keys\n", vertex_key, mesh_key, task_mesh_key);
}

// ---------- descriptions, padding, SPIR-V

static void print_io(const char *tag, const ShdIo *io, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        printf("%s %d %u %u %u %u %d %u\n", tag, io[i].location, io[i].component, io[i].components, io[i].type, io[i].bits,
               io[i].array_length, io[i].used);
}

static int load_desc(const char *path, ShaderDesc *d, char *why, size_t cap) {
    size_t n;
    char *text = read_all(path, &n);
    int ok = shd_parse(text, n, SHC_DESC_SCHEMA, d, why, cap);
    free(text);
    return ok;
}

static void case_desc(const char *path) {
    static ShaderDesc d;
    char why[256];
    if (!load_desc(path, &d, why, sizeof why)) {
        printf("refused: %s\n", why);
        return;
    }
    printf("kind %s fits %d raw %d\n", shd_kind_name(d.kind), d.fits, d.raw_pointers);
    print_io("in", d.inputs, d.ninputs);
    print_io("out", d.outputs, d.noutputs);
}

static void case_binds(const char *path) {
    static ShaderBinds b;
    char why[256];
    size_t n;
    char *text = read_all(path, &n);
    int ok = shd_parse_binds(text, n, SHC_DESC_SCHEMA, &b, why, sizeof why);
    free(text);
    if (!ok) {
        printf("refused: %s\n", why);
        return;
    }
    printf("kind %s fits %d raw %u support %u\n", shd_kind_name(b.kind), b.fits, b.raw, b.support_read);
    for (uint32_t i = 0; i < b.ncb; i++) printf("cb %u %u %u\n", b.cb[i].slot, b.cb[i].set, b.cb[i].binding);
    for (uint32_t i = 0; i < b.nsb; i++)
        printf("sb %u %u %u %u %u %u\n", b.sb[i].set, b.sb[i].ordinal, b.sb[i].cb_slot, b.sb[i].write, b.sb[i].binding,
               b.sb[i].address_byte_offset);
    for (uint32_t i = 0; i < b.ntex; i++)
        printf("tex %d %d %d %u %u %u %u %d %d %d\n", b.tex[i].via, b.tex[i].kind, b.tex[i].view, b.tex[i].set, b.tex[i].shadow,
               b.tex[i].binding, b.tex[i].array_length, b.tex[i].push_slot, b.tex[i].push_offset, b.tex[i].c2_slot);
    for (uint32_t i = 0; i < b.nimg; i++)
        printf("img %u %u %d %u %d\n", b.img[i].set, b.img[i].write, b.img[i].view, b.img[i].binding, b.img[i].c2_slot);
    for (uint32_t i = 0; i < b.nvin; i++)
        printf("vin %u %u %u %u\n", b.vin[i].location, b.vin[i].component, b.vin[i].components, b.vin[i].type);
    printf("local %u %u %u\n", b.local_size[0], b.local_size[1], b.local_size[2]);
}

static void case_pad(const char *vs, const char *ps) {
    static ShaderDesc v, p;
    char why[256];
    CHECK(load_desc(vs, &v, why, sizeof why));
    CHECK(load_desc(ps, &p, why, sizeof why));
    ShdPad pad[16];
    int n = shd_pad_set(&v, &p, pad, 16, why, sizeof why);
    if (n < 0) {
        printf("refused: %s\n", why);
        return;
    }
    char text[256];
    shd_pad_text(pad, n, text, sizeof text);
    printf("pad %d %s\n", n, text);
    for (int i = 0; i < n; i++) printf("run %u %u %u %u\n", pad[i].location, pad[i].component, pad[i].components, pad[i].type);
}

static void case_meshlink(const char *ms, const char *ps) {
    static ShaderDesc m, p;
    char why[256];
    CHECK(load_desc(ms, &m, why, sizeof why));
    CHECK(load_desc(ps, &p, why, sizeof why));
    ShdPad missing[16];
    int n = shd_mesh_link(&m, &p, missing, 16, why, sizeof why);
    if (n < 0) {
        printf("refused: %s\n", why);
        return;
    }
    char text[256];
    shd_pad_text(missing, n, text, sizeof text);
    printf("missing %d %s\n", n, text);
}

static void case_spvdecl(const char *path) {
    size_t n;
    uint32_t *w = read_all(path, &n);
    SpvDecl d;
    char why[256];
    if (!spv_declarations(w, n / 4, &d, why, sizeof why)) {
        printf("refused: %s\n", why);
        return;
    }
    printf("version %08x\n", d.version);
    for (uint32_t i = 0; i < d.ncaps; i++) {
        const char *name = spv_capability_name(d.caps[i]);
        printf("cap %u %s\n", d.caps[i], name ? name : "?");
    }
    for (uint32_t i = 0; i < d.nexts; i++) printf("ext %s\n", d.exts[i]);
    free(w);
}

static void case_spvfull(const char *path) {
    size_t n;
    uint32_t *w = read_all(path, &n);
    SpvDecl d;
    char why[256];
    if (!spv_declarations(w, n / 4, &d, why, sizeof why)) printf("refused: %s\n", why);
    else printf("full_subgroups %d\n", spv_needs_full_subgroups(&d));
    free(w);
}

static void case_spvpad(const char *in, const char *out, const char *spec) {
    size_t n;
    uint32_t *w = read_all(in, &n);
    SpvOutput add[16];
    uint32_t nadd = 0;
    for (const char *p = spec; *p && nadd < 16;) {
        unsigned l, c, k, t;
        int used = 0;
        CHECK(sscanf(p, "%u:%u:%u:%u%n", &l, &c, &k, &t, &used) == 4);
        add[nadd++] = (SpvOutput){l, c, k, t};
        p += used;
        if (*p == ',') p++;
    }
    size_t words = 0;
    char why[256];
    uint32_t *o = spv_add_outputs(w, n / 4, add, nadd, &words, why, sizeof why);
    if (!o) {
        printf("refused: %s\n", why);
        return;
    }
    FILE *f = fopen(out, "wb");
    CHECK(f && fwrite(o, 4, words, f) == words);
    fclose(f);
    printf("padded %zu words to %zu\n", n / 4, words);
    free(o), free(w);
}

// spvmeshpad IN OUT SPEC [MAX_BYTES MAX_COMPONENTS]: SPEC is "location:component:components:type" for each output, with a
// fifth ":p" for one read per primitive; the limits are the device's, 0 for none.
static void case_spvmeshpad(const char *in, const char *out, const char *spec, uint32_t max_bytes, uint32_t max_components,
                            uint32_t max_payload_and_output, uint32_t payload_bytes) {
    size_t n;
    uint32_t *w = read_all(in, &n);
    SpvMeshOutput add[32];
    uint32_t nadd = 0;
    for (const char *p = spec; *p && nadd < 32;) {
        unsigned l, c, k, t;
        int used = 0;
        CHECK(sscanf(p, "%u:%u:%u:%u%n", &l, &c, &k, &t, &used) == 4);
        p += used;
        unsigned per_primitive = 0;
        if (p[0] == ':' && p[1] == 'p') per_primitive = 1, p += 2;
        add[nadd++] = (SpvMeshOutput){l, c, k, t, per_primitive};
        if (*p == ',') p++;
    }
    size_t words = 0;
    char why[512];
    uint32_t *o = spv_add_mesh_outputs(w, n / 4, add, nadd, max_bytes, max_components,
                                       max_payload_and_output, payload_bytes, &words, why, sizeof why);
    if (!o) {
        printf("refused: %s\n", why);
        free(w);
        return;
    }
    FILE *f = fopen(out, "wb");
    CHECK(f && fwrite(o, 4, words, f) == words);
    fclose(f);
    printf("padded %zu words to %zu\n", n / 4, words);
    free(o), free(w);
}

// meshlibkey MS_DESC PS_DESC...: what build_link makes of each pair, up to the library's key and the rewrite's runs.
static void case_meshlibkey(const char *ms, int npix, char **pix) {
    static ShaderDesc m, p;
    char why[256];
    CHECK(load_desc(ms, &m, why, sizeof why));
    nvn_stage ts_shape = {.kind = NVN_STAGE_TS}, ms_shape = {.kind = NVN_STAGE_MS};
    VkpStage task = {.key = "6100000000000002.00000000", .shape = &ts_shape};
    VkpStage mesh = {.key = "6100000000000003.00000000", .shape = &ms_shape};
    for (int i = 0; i < npix; i++) {
        CHECK(load_desc(pix[i], &p, why, sizeof why));
        ShdPad pad[16];
        int n = shd_mesh_link(&m, &p, pad, 16, why, sizeof why);
        if (n < 0) {
            printf("refused: %s\n", why);
            continue;
        }
        char text[256], key[512], task_key[512];
        shd_pad_text(pad, n, text, sizeof text);
        vkp_mesh_library_key(NULL, &mesh, pad, n, key, sizeof key);
        vkp_mesh_library_key(&task, &mesh, pad, n, task_key, sizeof task_key);
        SpvMeshOutput out[16];
        uint32_t nout = vkp_mesh_pad_outputs(pad, n, out);
        printf("pad %s\nkey %s\ntask-key %s\nouts", text, key, task_key);
        for (uint32_t k = 0; k < nout; k++)
            printf(" %u:%u:%u:%u:%u", out[k].location, out[k].component, out[k].components, out[k].type, out[k].per_primitive);
        printf("\n");
    }
}

static void case_meshpayload(const char *path) {
    static ShaderDesc d;
    char why[256];
    CHECK(load_desc(path, &d, why, sizeof why));
    printf("payload %u\n", d.mesh_payload_bytes);
}

// A module's native HMMA turned into the helper (spv_hmma_fallback), written to out when it changed.
static void case_spvhmma(const char *in, const char *out) {
    size_t n;
    uint32_t *w = read_all(in, &n);
    uint32_t *o = NULL;
    size_t words = 0;
    char why[256];
    int r = spv_hmma_fallback(w, n / 4, &o, &words, why, sizeof why);
    if (r < 0) printf("refused: %s\n", why);
    else if (!r) printf("unchanged\n");
    else {
        FILE *f = fopen(out, "wb");
        CHECK(f && fwrite(o, 4, words, f) == words);
        fclose(f);
        printf("helper: %zu words to %zu\n", n / 4, words);
    }
    free(o), free(w);
}

static void case_spvzero(const char *in, const char *out) {
    size_t n;
    uint32_t *w = read_all(in, &n);
    uint32_t *o = NULL;
    size_t words = 0;
    char why[256];
    int changed = spv_zero_workgroup(w, n / 4, &o, &words, why, sizeof why);
    if (changed < 0) {
        printf("refused: %s\n", why);
        free(w);
        return;
    }
    if (!changed) {
        CHECK(!o && !words);
        printf("unchanged\n");
        free(w);
        return;
    }
    FILE *f = fopen(out, "wb");
    CHECK(f && fwrite(o, 4, words, f) == words);
    fclose(f);
    printf("zeroed %d variables, %zu words to %zu\n", changed, n / 4, words);
    free(o), free(w);
}

// The facts of a device that has every item of the required list and none of the optional features: the features
// the items enable (read off oot2_vk_enable of empty facts) with the limits, properties and extensions their tests read.
static void facts_with_every_item(Oot2VkFacts *f) {
    Oot2VkFacts zero, en;
    memset(&zero, 0, sizeof zero);
    oot2_vk_enable(&zero, &en);
    memset(f, 0, sizeof *f);
    f->features = en.features, f->f11 = en.f11, f->f12 = en.f12, f->f13 = en.f13;
    f->gpl = en.gpl, f->mesh = en.mesh, f->shader_object = en.shader_object, f->pipeline_binary = en.pipeline_binary;
    f->vertex_input = en.vertex_input, f->coop = en.coop, f->fault = en.fault, f->atomic_float = en.atomic_float;
    f->depth_clip = en.depth_clip, f->index_u8 = en.index_u8;
    f->eds3.extendedDynamicState3DepthClipNegativeOneToOne = f->eds3.extendedDynamicState3PolygonMode = VK_TRUE;
    f->eds3.extendedDynamicState3DepthClampEnable = f->eds3.extendedDynamicState3ColorBlendEnable = VK_TRUE;
    f->eds3.extendedDynamicState3ColorBlendEquation = f->eds3.extendedDynamicState3ColorWriteMask = VK_TRUE;
    memset(f->ext, 1, sizeof f->ext);
    f->props.apiVersion = VK_API_VERSION_1_3;
    f->props.limits.maxViewports = 16;
    f->host.minImportedHostPointerAlignment = 4096;
    f->push.maxPushDescriptors = 32;
    f->p11.subgroupSize = f->p13.minSubgroupSize = f->p13.maxSubgroupSize = 32;
    f->p11.subgroupSupportedOperations = OOT2_VK_SUBGROUP_OPS;
    f->p11.subgroupSupportedStages = OOT2_VK_SUBGROUP_STAGES;
    f->p12.maxDescriptorSetUpdateAfterBindSampledImages = 1u << 20;
    f->p13.requiredSubgroupSizeStages = OOT2_VK_TASK_MESH;
    f->gpl_props.graphicsPipelineLibraryFastLinking = VK_TRUE;
    f->mesh_props.maxMeshOutputVertices = 160;
    f->mesh_props.maxMeshOutputPrimitives = 128;
    f->mesh_props.maxMeshOutputMemorySize = 28160;
    unsigned char has[OOT2_VK_ITEM_COUNT];
    CHECK(oot2_vk_judge(f, has) == OOT2_VK_ITEM_COUNT);   // an item added to the list needs its facts here
}

static void case_capcheck(const char *path, int argc, char **argv) {
    size_t n;
    uint32_t *w = read_all(path, &n);
    SpvDecl d;
    char why[256];
    CHECK(spv_declarations(w, n / 4, &d, why, sizeof why));
    static Oot2VkFacts f;
    facts_with_every_item(&f);
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], "storage-image-indexing")) f.f12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    VkcList l;
    vkc_list(&f, &l);
    for (int i = 0; i + 1 < argc; i++) {
        if (strcmp(argv[i], "drop")) continue;
        uint32_t drop = (uint32_t)strtoul(argv[i + 1], NULL, 10);
        size_t at = 0;
        for (size_t i = 0; i < l.ncaps; i++)
            if (l.caps[i] != drop) l.caps[at++] = l.caps[i];
        l.ncaps = at;
    }
    if (vkc_check(&l, &d, why, sizeof why)) printf("bad: %s\n", why);
    else printf("ok\n");
    printf("list %zu capabilities, %zu extensions\n", l.ncaps, l.nexts);
    free(w);
}

static void case_optional(void) {
    printf("items %d optional %d\n", OOT2_VK_ITEM_COUNT, OOT2_VK_OPTIONAL_COUNT);
    for (int i = 0; i < OOT2_VK_ITEM_COUNT; i++) printf("item %s\n", oot2_vk_item_id(i));
    for (int i = 0; i < OOT2_VK_OPTIONAL_COUNT; i++) printf("optional %s\n", oot2_vk_optional_id(i));
    for (int present = 0; present < 2; present++) {
        const char *tag = present ? "present" : "absent";
        Oot2VkFacts f, en;
        facts_with_every_item(&f);
        f.f12.shaderStorageImageArrayNonUniformIndexing = present ? VK_TRUE : VK_FALSE;
        f.f13.shaderZeroInitializeWorkgroupMemory = present ? VK_TRUE : VK_FALSE;
        f.f12.shaderFloat16 = present ? VK_TRUE : VK_FALSE;
        unsigned char has[OOT2_VK_ITEM_COUNT];
        printf("%s: has %d of %d\n", tag, oot2_vk_judge(&f, has), OOT2_VK_ITEM_COUNT);
        for (int i = 0; i < OOT2_VK_ITEM_COUNT; i++)
            if (!has[i]) printf("%s: lacks %s\n", tag, oot2_vk_item_id(i));
        oot2_vk_enable(&f, &en);
        printf("%s: enabled %d\n", tag, en.f12.shaderStorageImageArrayNonUniformIndexing ? 1 : 0);
        printf("%s: zero-initialized workgroup memory enabled %d\n", tag, en.f13.shaderZeroInitializeWorkgroupMemory ? 1 : 0);
        printf("%s: 16-bit floats enabled %d\n", tag, en.f12.shaderFloat16 ? 1 : 0);
        VkcList l;
        vkc_list(&f, &l);
        printf("%s: capability 5309 %s\n", tag, vkc_has_cap(&l, 5309) ? "allowed" : "refused");
    }
}

static void case_cachejudge(void) {
    VkPhysicalDeviceProperties props;
    memset(&props, 0, sizeof props);
    props.vendorID = 0x10de, props.deviceID = 0x2486;
    for (int i = 0; i < (int)VK_UUID_SIZE; i++) props.pipelineCacheUUID[i] = (uint8_t)(0xa0 + i);
    uint8_t data[64];
    memset(data, 0x5a, sizeof data);
    VkPipelineCacheHeaderVersionOne h = {.headerSize = sizeof h, .headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE,
                                         .vendorID = props.vendorID, .deviceID = props.deviceID};
    memcpy(h.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE);
    memcpy(data, &h, sizeof h);
    printf("matching %d\n", (int)vkp_cache_judge(data, sizeof data, &props));
    data[16 + 5] ^= 1;   // a byte of the UUID
    printf("uuid %d\n", (int)vkp_cache_judge(data, sizeof data, &props));
    data[16 + 5] ^= 1;
    VkPhysicalDeviceProperties other = props;
    other.deviceID = 0x2487;
    printf("device %d\n", (int)vkp_cache_judge(data, sizeof data, &other));
    other = props, other.vendorID = 0x1002;
    printf("vendor %d\n", (int)vkp_cache_judge(data, sizeof data, &other));
    printf("short %d\n", (int)vkp_cache_judge(data, 20, &props));
    printf("empty %d\n", (int)vkp_cache_judge(data, 0, &props));
    uint32_t version = 2;
    memcpy(data + 4, &version, 4);
    printf("version %d\n", (int)vkp_cache_judge(data, sizeof data, &props));
    memcpy(data + 4, &(uint32_t){VK_PIPELINE_CACHE_HEADER_VERSION_ONE}, 4);
    uint32_t hs = 1000;   // a header size past the data
    memcpy(data, &hs, 4);
    printf("headersize %d\n", (int)vkp_cache_judge(data, sizeof data, &props));
    printf("none=%d loaded=%d other=%d unreadable=%d\n", VKP_CACHE_NONE, VKP_CACHE_LOADED, VKP_CACHE_OTHER_DRIVER,
           VKP_CACHE_UNREADABLE);
}

static int kind_of(const char *name) {
    for (int k = 0; k < SHD_KINDS; k++)
        if (!strcmp(name, shd_kind_name(k))) return k;
    return -1;
}

static void case_store(const char *base, const char *variants, const char *key, const char *kind, const char *shape) {
    char why[512];
    if (!vkp_store_open(base, strcmp(variants, "-") ? variants : NULL, why, sizeof why)) {
        printf("refused: %s\n", why);
        return;
    }
    nvn_stage st = {.kind = NVN_STAGE_PS, .ctl_size = 0x57, .code_size = 0x1230, .inst_size = 0x1000};
    snprintf(st.key, sizeof st.key, "%s", key);
    VkpStage s = {.key = key, .answers = NULL, .shape = strcmp(shape, "0") ? &st : NULL};
    static VkpModule m;
    vkp_module(&s, kind_of(kind), &m);
    printf("module %u bytes from %d, kind %s, fits %d, %u inputs, %u outputs\n", m.bytes, m.from, shd_kind_name(m.desc.kind),
           m.desc.fits, m.desc.ninputs, m.desc.noutputs);
    vkp_store_close();
}

// The replay's cache of stage bindings (dev_binds in runtime/gpu/vk_replay_dev.c), asked through the device ops, which
// need no device for it: N stage records of one pack key, each at its own address, each asked twice. The bucket index
// of a record is a function of its address and the pack entry's digest, so many records reach many buckets.
static void case_devbinds(const char *base, const char *key, int n) {
    char why[512];
    if (!vkp_store_open(base, NULL, why, sizeof why)) {
        printf("refused: %s\n", why);
        return;
    }
    CHECK(n > 0 && n <= 4096);
    nvn_stage *recs = calloc((size_t)n, sizeof *recs);
    const ShaderBinds **first = calloc((size_t)n, sizeof *first);
    CHECK(recs && first);
    const VkReplayOps *ops = vkr_device_ops();
    CHECK(ops->binds);
    int distinct = 0, stable = 0;
    for (int i = 0; i < n; i++) {
        recs[i] = (nvn_stage){.kind = NVN_STAGE_PS, .ctl_size = 0x57, .code_size = 0x1230, .inst_size = 0x1000};
        snprintf(recs[i].key, sizeof recs[i].key, "%s", key);
        VkpStage s = {.key = key, .answers = NULL, .shape = &recs[i]};
        first[i] = ops->binds(&s);
        CHECK(first[i]);
        int seen = 0;
        for (int j = 0; j < i; j++) seen |= first[j] == first[i];
        distinct += !seen;
    }
    for (int i = 0; i < n; i++) {
        VkpStage s = {.key = key, .answers = NULL, .shape = &recs[i]};
        stable += ops->binds(&s) == first[i];
    }
    printf("binds %d distinct %d stable %d kind %s\n", n, distinct, stable, shd_kind_name(first[0]->kind));
    free(first);
    free(recs);
    vkp_store_close();
}

// A graphics pair asked for from a builder that was never opened (vkp_graphics_unopened): build_link judges the pair's
// link before it creates anything, so a pair it refuses ends in the named fatal of the rule it chose, with no device.
// KIND "ms" gives the first stage the shape of a mesh stage, "vs" of a vertex stage.
static void case_linkchoice(const char *base, const char *vkey, const char *pkey, const char *kind) {
    char why[512];
    if (!vkp_store_open(base, NULL, why, sizeof why)) {
        printf("refused: %s\n", why);
        return;
    }
    nvn_stage v = {.kind = !strcmp(kind, "ms") ? NVN_STAGE_MS : NVN_STAGE_VS, .ctl_size = 0x57, .code_size = 0x1230,
                   .inst_size = 0x1000};
    nvn_stage p = {.kind = NVN_STAGE_PS, .ctl_size = 0x57, .code_size = 0x1230, .inst_size = 0x1000};
    snprintf(v.key, sizeof v.key, "%s", vkey);
    snprintf(p.key, sizeof p.key, "%s", pkey);
    VkpStage vs = {.key = vkey, .shape = &v}, ps = {.key = pkey, .shape = &p};
    VkpFormats formats = {0};
    vkp_graphics_unopened(&vs, NULL, &ps, &formats);
    printf("linked\n");   // not reached by a pair the builder refuses
    vkp_store_close();
}

static int parse_sha256(const char *text, uint8_t digest[32]) {
    if (strlen(text) != 64) return 0;
    for (int i = 0; i < 32; i++) {
        unsigned value;
        if (sscanf(text + i * 2, "%2x", &value) != 1) return 0;
        digest[i] = (uint8_t)value;
    }
    return 1;
}

static void case_storemap(const char *base, const char *variants, const char *map, const char *key, const char *kind,
                          const char *shape, const char *digest_text, const char *identity_mode) {
    char why[512];
    uint8_t digest[32];
    if (!vkp_store_open_map(base, strcmp(variants, "-") ? variants : NULL, strcmp(map, "-") ? map : NULL,
                            why, sizeof why)) {
        printf("refused: %s\n", why);
        return;
    }
    if (!parse_sha256(digest_text, digest)) {
        printf("bad digest\n");
        vkp_store_close();
        return;
    }
    nvn_stage st = {.kind = NVN_STAGE_PS, .ctl_size = 0x57, .code_size = 0x1230, .inst_size = 0x1000};
    snprintf(st.key, sizeof st.key, "%s", key);
    VkpStage s = {.key = key, .shape = strcmp(shape, "0") ? &st : NULL};
    vkp_stage_set_ledger_digest(&s, digest);
    VkpFormats formats = {0};
    char pipeline_key[512];
    vkp_graphics_key(&s, NULL, &s, &formats, pipeline_key, sizeof pipeline_key);
    printf("key %s\n", pipeline_key);
    uint8_t content_digest[SHC_DIGEST_BYTES];
    vkp_stage_content_digest(&s, content_digest);
    if (!strcmp(identity_mode, "raw")) memcpy(content_digest, digest, sizeof content_digest);
    printf("content ");
    for (size_t i = 0; i < sizeof content_digest; i++) printf("%02x", content_digest[i]);
    printf("\n");
    static VkpModule m;
    vkp_module(&s, kind_of(kind), &m);
    VkpStats stats;
    vkp_stats(&stats);
    printf("module from %d, map missing %llu, unknown %llu\n", m.from,
           (unsigned long long)stats.map_missing, (unsigned long long)stats.map_unknown_rows);
    vkp_store_close();
}

static void case_storeopen(const char *base, const char *variants) {
    char why[512];
    if (!vkp_store_open(base, strcmp(variants, "-") ? variants : NULL, why, sizeof why)) printf("refused: %s\n", why);
    else printf("ok\n"), vkp_store_close();
}


// ---------- usable: how many devices here have every item (the driver skips the device cases when none does)
static void case_usable(void) {
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        exit(EXIT_NO_DEVICE);
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) {
        printf("no Vulkan 1.3 instance\n");
        exit(EXIT_NO_DEVICE);
    }
    volkLoadInstanceOnly(inst);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL) == VK_SUCCESS);
    if (!n) {
        printf("no Vulkan device\n");
        exit(EXIT_NO_DEVICE);
    }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, devs) >= 0);
    int usable = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ne = 0;
        CHECK(vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, NULL) == VK_SUCCESS);
        VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
        CHECK(ext && vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, ext) == VK_SUCCESS);
        Oot2VkFacts f;
        oot2_vk_query(vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceFeatures2, devs[i], ext, ne, &f);
        free(ext);
        unsigned char has[OOT2_VK_ITEM_COUNT];
        usable += oot2_vk_judge(&f, has) == OOT2_VK_ITEM_COUNT;
    }
    vkDestroyInstance(inst, NULL);
    printf("usable %d\nok usable\n", usable);
}

// ---------- the device

static void close_and_report(void) {
    vkp_close();
    vkp_store_close();
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (unsigned long long)errors, (unsigned long long)warnings);
}

#define VS_KEY "5100000000000001.00000000"
#define PS_KEY "5100000000000002.00000000"
#define CS_KEY "5100000000000003.00000000"
#define CS2_KEY "5100000000000004.00000000"
#define ZP_KEY "5100000000000007.00000000"   // fills each workgroup's shared words with ones and stores nothing
#define ZR_KEY "5100000000000008.00000000"   // stores the shared word of each invocation, which no invocation wrote
#define NO_KEY "5100000000000009.00000000"

static void open_all(const char *pack, const char *cache) {
    gpu_vk_open();
    char why[512];
    if (!vkp_store_open(pack, NULL, why, sizeof why)) {
        fprintf(stderr, "the pack did not open: %s\n", why);
        exit(1);
    }
    vkp_open(cache);
}

static const VkpFormats F32 = {1, {VK_FORMAT_R32G32B32A32_SFLOAT}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
static const VkpFormats F16 = {1, {VK_FORMAT_R16G16B16A16_SFLOAT}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
static const VkpFormats F8 = {1, {VK_FORMAT_R8G8B8A8_UNORM}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};

static void print_stats(const char *tag) {
    VkpStats s;
    vkp_stats(&s);
    printf("stats %s libraries %llu hits %llu stage %llu hits %llu links %llu hits %llu computes %llu hits %llu padded %llu "
           "resident %llu build_ms %.1f %.1f %.1f\n", tag, (unsigned long long)s.libraries,
           (unsigned long long)s.library_hits, (unsigned long long)s.stage_libraries,
           (unsigned long long)s.stage_library_hits, (unsigned long long)s.links, (unsigned long long)s.link_hits,
           (unsigned long long)s.computes,
           (unsigned long long)s.compute_hits, (unsigned long long)s.padded, (unsigned long long)s.resident,
           (double)s.build_ns[0] / 1e6, (double)s.build_ns[1] / 1e6, (double)s.build_ns[2] / 1e6);
}

static void case_layouts(void) {
    gpu_vk_open();
    VkpLayouts l;
    vkp_layouts_create(&l);
    CHECK(l.global && l.graphics_push && l.compute_push && l.task && l.graphics && l.compute);
    vkp_layouts_destroy(&l);
    CHECK(!l.graphics);
    gpu_vk_close();
    uint64_t e, w;
    gpu_vk_validation_counts(&e, &w);
    printf("validation errors %llu warnings %llu\nok layouts\n", (unsigned long long)e, (unsigned long long)w);
}

static float half_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {
            e = 127 - 15 + 1;
            while (!(m & 1024)) m <<= 1, e--;
            bits = s | e << 23 | (m & 1023) << 13;
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | m << 13;
    } else {
        bits = s | (e + 127 - 15) << 23 | m << 13;
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static void pixel_of(const void *img, VkFormat fmt, uint32_t w, uint32_t x, uint32_t y, float out[4]) {
    size_t at = ((size_t)y * w + x);
    if (fmt == VK_FORMAT_R32G32B32A32_SFLOAT) memcpy(out, (const float *)img + at * 4, 16);
    else if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT)
        for (int i = 0; i < 4; i++) out[i] = half_to_float(((const uint16_t *)img)[at * 4 + i]);
    else
        for (int i = 0; i < 4; i++) out[i] = (float)((const uint8_t *)img)[at * 4 + i] / 255.0f;
}

// The synthetic pair's draw: a triangle over the whole target, vertex bank 1, vertex storage 9, pixel bank 14, pixel
// storage 25 and one push word. The pixel stage adds the vertex output at location 1, which nothing writes.
static void synth_spec(DrawSpec *s, VkPipeline pipe, VkFormat fmt, void *out) {
    static const float vs_bank[4] = {0, 0, 0, 0}, vs_data[4] = {0.25f, 0.5f, 0.75f, 0.0f};
    static const float ps_bank[4] = {0.5f, 0.25f, 0.0f, 0.0f}, ps_data[4] = {2.0f, 2.0f, 2.0f, 0.5f};
    memset(s, 0, sizeof *s);
    s->pipeline = pipe, s->format = fmt, s->width = s->height = 32, s->vertex_count = 3, s->out = out;
    s->clear[0] = s->clear[1] = s->clear[2] = s->clear[3] = -7.0f;
    s->binds[s->nbinds++] = (DrawBind){1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vs_bank, 16};
    s->binds[s->nbinds++] = (DrawBind){9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vs_data, 16};
    s->binds[s->nbinds++] = (DrawBind){14, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ps_bank, 16};
    s->binds[s->nbinds++] = (DrawBind){25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ps_data, 16};
    s->push[4] = 1;   // words[1].x
}

static int near4(const float *got, const float *want, float tol) {
    for (int i = 0; i < 4; i++) {
        float d = got[i] - want[i];
        if (d < -tol || d > tol) return 0;
    }
    return 1;
}

static void case_synth_draw(const char *pack, int unpad) {
    open_all(pack, NULL);
    vkp_unpad = unpad;
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY};
    VkPipeline p32 = vkp_graphics(&vs, NULL, &ps, &F32), p16 = vkp_graphics(&vs, NULL, &ps, &F16);
    CHECK(p32 && p16 && p32 != p16);
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32) == p32);   // the same pipeline again, made once
    print_stats("graphics");
    static float img32[32 * 32 * 4];
    static uint16_t img16[32 * 32 * 4];
    DrawSpec s;
    synth_spec(&s, p32, F32.color[0], img32);
    draw_run(&s);
    synth_spec(&s, p16, F16.color[0], img16);
    draw_run(&s);
    float px32[4], px16[4];
    pixel_of(img32, F32.color[0], 32, 16, 16, px32);
    pixel_of(img16, F16.color[0], 32, 16, 16, px16);
    printf("pixel f32 %g %g %g %g\npixel f16 %g %g %g %g\n", px32[0], px32[1], px32[2], px32[3], px16[0], px16[1], px16[2], px16[3]);
    // a + b (b padded to (0, 0, 0, 1)) + bank, times storage, plus the push word
    const float want[4] = {2.5f, 2.5f, 2.5f, 1.5f};
    int ok32 = near4(px32, want, 1e-5f), ok16 = near4(px16, want, 1e-2f);
    printf("padded input reads (0, 0, 0, 1): f32 %s, f16 %s\n", ok32 ? "yes" : "no", ok16 ? "yes" : "no");
    close_and_report();
    if (!unpad) CHECK(ok32 && ok16);
    printf("ok synth_draw\n");
}

// One mesh stage that writes location 0 and four pixel stages (the pack of MESH_GLSL in the test): a and c read locations 0
// and 1, b reads 0, 1 and 2 and d reads 0 and, once for each primitive, 3, so a and c have one pad set (location 1), b
// another (locations 1 and 2) and d a third (location 3, per primitive). The libraries are keyed by the module and the pad
// set: the mesh stage's padded library is made once for each pad set, whatever the pixel stage and the format, and each
// pixel stage has its fragment library.
#define MM_KEY "5200000000000001.00000000"
#define MPA_KEY "5200000000000002.00000000"
#define MPB_KEY "5200000000000003.00000000"
#define MPC_KEY "5200000000000004.00000000"
#define MPD_KEY "5200000000000005.00000000"   // reads location 3 once for each primitive

static void case_synth_meshpad(const char *pack) {
    open_all(pack, NULL);
    nvn_stage shape = {.kind = NVN_STAGE_MS, .ctl_size = 0x57, .code_size = 0x1230, .inst_size = 0x1000};
    snprintf(shape.key, sizeof shape.key, "%s", MM_KEY);
    VkpStage m = {.key = MM_KEY, .shape = &shape}, a = {.key = MPA_KEY}, b = {.key = MPB_KEY}, c = {.key = MPC_KEY};
    VkpStage d = {.key = MPD_KEY};
    VkPipeline pa8 = vkp_graphics(&m, NULL, &a, &F8), pc8 = vkp_graphics(&m, NULL, &c, &F8);
    VkPipeline pb8 = vkp_graphics(&m, NULL, &b, &F8), pa16 = vkp_graphics(&m, NULL, &a, &F16);
    VkPipeline pd8 = vkp_graphics(&m, NULL, &d, &F8);
    CHECK(pa8 && pc8 && pb8 && pa16 && pd8);
    CHECK(pa8 != pc8 && pa8 != pb8 && pa8 != pa16 && pc8 != pb8 && pd8 != pa8);
    CHECK(vkp_graphics(&m, NULL, &a, &F8) == pa8);   // the same pipeline again, made once
    print_stats("meshpad");
    close_and_report();
    printf("ok synth_meshpad\n");
}

static void case_synth_compute(const char *pack) {
    open_all(pack, NULL);
    VkpStage cs = {.key = CS_KEY};
    VkPipeline p = vkp_compute(&cs);
    CHECK(p && vkp_compute(&cs) == p);
    uint32_t bank[4] = {3, 0, 0, 0}, push[VKP_PUSH_BYTES / 4] = {7};
    DrawBind binds[2] = {{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bank, 16}, {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, 128 * 4}};
    DrawBuf bufs[2];
    bufs[0] = draw_buffer(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, bank, 16);
    bufs[1] = draw_buffer(128 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, 0);
    dispatch_run(p, binds, 2, bufs, push, 4);
    const uint32_t *out = bufs[1].map;
    for (uint32_t i = 0; i < 128; i++) CHECK(out[i] == i * 3 + 7);
    printf("128 values written by the dispatch: 3 * i + 7\n");
    draw_buffer_free(&bufs[0]), draw_buffer_free(&bufs[1]);
    print_stats("compute");
    close_and_report();
    printf("ok synth_compute\n");
}

// A module whose workgroup memory is never written by it reads zeros, whatever the dispatch before left in the device's
// workgroup memory (the builder gives such a module the null initializer, spv_zero_workgroup): the program that fills the
// shared words with ones runs first, on as many workgroups as fill the device, so the reader's workgroups land on the
// memory it left.
static void case_synth_zeroinit(const char *pack) {
    open_all(pack, NULL);
    VkpStage zp = {.key = ZP_KEY}, zr = {.key = ZR_KEY};
    VkPipeline pp = vkp_compute(&zp), pr = vkp_compute(&zr);
    CHECK(pp && pr && pp != pr);
    enum { GROUPS = 512, WORDS = GROUPS * 64, ROUNDS = 8 };
    uint32_t push[VKP_PUSH_BYTES / 4] = {0};
    DrawBind pbind = {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, 16}, rbind = {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, (size_t)WORDS * 4};
    DrawBuf dummy = draw_buffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, 0);
    DrawBuf out = draw_buffer((VkDeviceSize)WORDS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, 0);
    unsigned long long nonzero = 0;
    for (int round = 0; round < ROUNDS; round++) {
        dispatch_run(pp, &pbind, 1, &dummy, push, GROUPS);
        memset(out.map, 0xAB, (size_t)WORDS * 4);   // a word the reader did not store would show
        dispatch_run(pr, &rbind, 1, &out, push, GROUPS);
        const uint32_t *o = out.map;
        for (uint32_t i = 0; i < WORDS; i++) nonzero += o[i] != 0;
    }
    printf("%llu of %d words read as something other than zero\n", nonzero, ROUNDS * WORDS);
    CHECK(nonzero == 0);
    draw_buffer_free(&dummy), draw_buffer_free(&out);
    print_stats("zeroinit");
    close_and_report();
    printf("ok synth_zeroinit\n");
}

static void case_synth_inflight(const char *pack) {
    open_all(pack, NULL);
    VkpStage cs = {.key = CS_KEY}, cs2 = {.key = CS2_KEY};
    VkPipeline p = vkp_compute(&cs);
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    VkSemaphore gate;
    CHECK(vkCreateSemaphore(gpu_vk.device, &si, NULL, &gate) == VK_SUCCESS);
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    gpu_vk_cmd();
    CHECK(gpu_vk_end(&wait, 1) == v);
    CHECK(gpu_sync_handed() != gpu_sync_done());
    CHECK(vkp_compute(&cs) == p);   // what exists is returned, in flight or not
    printf("the existing pipeline was returned with a submission in flight\n");
    fflush(stdout);
    vkp_compute(&cs2);              // one that must be made: a named fatal
    printf("returned\n");
}

static void case_synth_cache(const char *pack, const char *file, const char *phase) {
    open_all(pack, file);
    printf("cache_state %d\n", (int)vkp_cache_state());
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY}, cs = {.key = CS_KEY};
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32) && vkp_graphics(&vs, NULL, &ps, &F16) && vkp_compute(&cs));
    print_stats(phase);
    if (!strcmp(phase, "cold")) {
        char why[256];
        CHECK(vkp_cache_save(why, sizeof why));
        printf("cache saved\n");
    }
    close_and_report();
    printf("ok synth_cache\n");
}

// The cache file read as the launch reads it (no device): the cause of a failure, its phrase, and how long the read took.
static void case_cacheread(const char *path, int nopt, char **opt) {
    for (int i = 0; i < nopt; i++) {
        if (!strncmp(opt[i], "readcap=", 8)) vkp_read_cap = atoi(opt[i] + 8);
        else if (!strncmp(opt[i], "openerr=", 8)) vkp_open_errno = atoi(opt[i] + 8);
        else if (!strcmp(opt[i], "transient")) vkp_open_transient = 1;
        else {
            printf("unknown option %s\n", opt[i]);
            exit(2);
        }
    }
    static const char *const names[] = {"ok", "absent", "empty", "open", "too_large", "unreadable"};
    size_t size = 0;
    int fail = -1, err = 0;
    char said[160];
    uint64_t t0 = plat_now_ns();
    void *data = vkp_cache_read(path, &size, &fail, &err);
    double ms = (double)(plat_now_ns() - t0) / 1e6;
    printf("read %s size %zu ms %.0f\n", data ? "data" : "none", data ? size : (size_t)0, ms);
    printf("cause %s\n", names[fail]);
    printf("text %s\n", vkp_read_fail_text(fail, err, said, sizeof said));
    free(data);
    printf("ok cacheread\n");
}

#ifdef _WIN32
// A temporary file moved over PATH three times. With PATH read-only the move is refused with "access denied": the first call
// waits through its tries, the second tries once; with "heal" PATH is made writable before the third, which then moves.
static void case_cachemove(const char *tmp, const char *path, int heal) {
    for (int i = 1; i <= 3; i++) {
        if (i == 3 && heal) _chmod(path, _S_IREAD | _S_IWRITE);
        uint64_t t0 = plat_now_ns();
        int ok = vkp_move_over(tmp, path);
        printf("move %d ok %d tries %d ms %.0f\n", i, ok, vkp_move_tries, (double)(plat_now_ns() - t0) / 1e6);
    }
    printf("ok cachemove\n");
}
#else
static void case_cachemove(const char *tmp, const char *path, int heal) {
    (void)tmp, (void)path, (void)heal;
    printf("skipped: no sharing violation off Windows\n");
    printf("ok cachemove\n");
}
#endif

static void case_savesettings(void) {
    static const struct { const char *save, *seconds; } t[] = {
        {NULL, NULL}, {"", ""}, {"on", "0"}, {"off", "5"}, {"on", "86400"}, {NULL, "1"}, {"maybe", NULL}, {"On", NULL},
        {NULL, "86401"}, {NULL, "-1"}, {NULL, "12x"}, {NULL, "1.5"}, {NULL, " 5"}, {NULL, "99999999999999999999"}};
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        VkpSaveSettings s;
        char why[256];
        const char *a = t[i].save ? t[i].save : "(unset)", *b = t[i].seconds ? t[i].seconds : "(unset)";
        if (vkp_save_settings(t[i].save, t[i].seconds, &s, why, sizeof why))
            printf("save [%s] [%s]: enabled %d seconds %u\n", a, b, s.enabled, (unsigned)s.seconds);
        else
            printf("save [%s] [%s]: refused: %s\n", a, b, why);
    }
    printf("ok savesettings\n");
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

static void case_synth_save(const char *folder, const char *mode, int nopt, char **opt) {
    int more = 0, stale = 0, flush = 0, wait_ms = 0;
    for (int i = 0; i < nopt; i++) {
        if (!strcmp(opt[i], "more")) more = 1;
        else if (!strcmp(opt[i], "stale")) stale = 1;
        else if (!strncmp(opt[i], "stall=", 6)) vkp_save_stall_ms = atoi(opt[i] + 6);
        else if (!strncmp(opt[i], "readcap=", 8)) vkp_read_cap = atoi(opt[i] + 8);
        else if (!strncmp(opt[i], "openerr=", 8)) vkp_open_errno = atoi(opt[i] + 8);
        else if (!strcmp(opt[i], "flush")) flush = 1;
        else if (!strncmp(opt[i], "wait=", 5)) wait_ms = atoi(opt[i] + 5);
        else {
            printf("unknown option %s\n", opt[i]);
            exit(2);
        }
    }
    char stale_path[1024];
    snprintf(stale_path, sizeof stale_path, "%s/pipelines/pipeline.cache.%u.tmp", folder, plat_pid());
    if (stale) {   // what a process of this very id left when it ended in the middle of a save
        FILE *f = fopen(stale_path, "wb");
        CHECK(f && fputs("half a cache", f) >= 0);
        fclose(f);
    }
    gpu_vk_open();
    gpu_vk_open_game(folder);
    if (stale) printf("the stale temporary file %s\n", file_exists(stale_path) ? "is still there" : "was removed");
    printf("cache_state %d\n", (int)vkp_cache_state());
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY}, cs = {.key = CS_KEY}, cs2 = {.key = CS2_KEY};
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32) && vkp_graphics(&vs, NULL, &ps, &F16) && vkp_compute(&cs));
    if (more) CHECK(vkp_compute(&cs2));
    print_stats("save");
    fflush(stdout);
    char path[1024];
    snprintf(path, sizeof path, "%s/pipelines/pipeline.cache", folder);
    if (!strcmp(mode, "fatal")) {
        rt_fatal("synth_save: a named stop");
    } else if (!strcmp(mode, "exit")) {
        rt_run_stop_hook(0, "");
        plat_exit(0);
    } else if (!strcmp(mode, "kill")) {
        _Exit(0);
    } else if (!strcmp(mode, "timer")) {
        uint64_t t0 = plat_now_ns();
        while (!file_exists(path) && plat_now_ns() - t0 < 30000000000ull) plat_sleep_ns(50000000ull);
        printf("the cache file %s after %.1f s\n", file_exists(path) ? "was there" : "never came",
               (double)(plat_now_ns() - t0) / 1e9);
        fflush(stdout);
        plat_exit(0);   // no stop hook: only the timer can have written the file
    } else if (!strcmp(mode, "wait")) {   // the timer's rounds go by, then the run ends with no stop hook
        if (flush) {
            char seg[1024], why[256] = "";
            snprintf(seg, sizeof seg, "%s/segment.bin", folder);
            size_t bytes = 0;
            CHECK(vkp_cache_flush(seg, &bytes, why, sizeof why));
        }
        plat_sleep_ns((uint64_t)wait_ms * 1000000ull);
        plat_exit(0);
    } else if (!strcmp(mode, "flushed")) {
        char seg[1024], why[256] = "";
        snprintf(seg, sizeof seg, "%s/segment.bin", folder);
        size_t bytes = 0;
        CHECK(vkp_cache_flush(seg, &bytes, why, sizeof why));
        printf("flushed %zu bytes\n", bytes);
        CHECK(!vkp_cache_save(why, sizeof why));
        printf("save refused: %s\n", why);
        fflush(stdout);
        rt_fatal("synth_save: a named stop after a flush");
    }
    printf("unknown mode %s\n", mode);
    exit(2);
}

#define LONG_KEY "5100000000000010.00000000"   // a compute stage of thousands of dependent operations: a compile of seconds

static void long_compile_main(void *arg) {
    (void)arg;
    VkpStage cs = {.key = LONG_KEY};
    uint64_t t0 = plat_now_ns();
    CHECK(vkp_compute(&cs));
    printf("the long compile ended after %.0f ms\n", (double)(plat_now_ns() - t0) / 1e6);
    fflush(stdout);
}

// A named stop while another thread is inside a long pipeline compile of the same cache: the stop's save must not wait for
// the compile. MODE compile only times the compile on this thread.
static void case_synth_stopcompile(const char *folder, const char *mode, int sleep_ms) {
    gpu_vk_open();
    gpu_vk_open_game(folder);
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY}, cs = {.key = CS_KEY};
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32) && vkp_compute(&cs));
    if (!strcmp(mode, "compile")) {
        long_compile_main(NULL);
        fflush(stdout);
        plat_exit(0);
    }
    CHECK(plat_thread_start(long_compile_main, NULL, 1u << 20) != NULL);
    plat_sleep_ns((uint64_t)sleep_ms * 1000000ull);   // the compile is under way: a fraction of what it takes, set by the caller
    printf("stopping\n");
    fflush(stdout);
    rt_fatal("synth_stopcompile: a named stop during a long compile");
}

static void case_synth_release(const char *pack) {
    open_all(pack, NULL);
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY};
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32));
    VkpStats a, b, c;
    vkp_stats(&a);
    vkp_release(PS_KEY);
    vkp_stats(&b);
    printf("resident %llu before, %llu after releasing the pixel stage\n", (unsigned long long)a.resident,
           (unsigned long long)b.resident);
    CHECK(b.resident < a.resident);
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32));
    vkp_stats(&c);
    printf("the pipeline and the pixel stage's library were made again: links %llu to %llu, libraries %llu to %llu\n",
           (unsigned long long)b.links, (unsigned long long)c.links, (unsigned long long)b.libraries,
           (unsigned long long)c.libraries);
    CHECK(c.links == b.links + 1 && c.libraries == b.libraries + 1);   // the vertex library was kept
    close_and_report();
    printf("ok synth_release\n");
}

#define THREADS 8
static VkPipeline g_thread_pipe[THREADS];
static int g_started, g_done, g_go;

static void thread_main(void *arg) {
    int i = (int)(intptr_t)arg;
    __atomic_add_fetch(&g_started, 1, __ATOMIC_SEQ_CST);
    while (!__atomic_load_n(&g_go, __ATOMIC_ACQUIRE)) plat_yield();
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY};
    g_thread_pipe[i] = vkp_graphics(&vs, NULL, &ps, &F32);
    __atomic_add_fetch(&g_done, 1, __ATOMIC_SEQ_CST);
}

static void case_synth_threads(const char *pack) {
    open_all(pack, NULL);
    for (intptr_t i = 0; i < THREADS; i++) plat_thread_start(thread_main, (void *)i, 8u << 20);
    while (__atomic_load_n(&g_started, __ATOMIC_SEQ_CST) < THREADS) plat_yield();
    __atomic_store_n(&g_go, 1, __ATOMIC_RELEASE);
    while (__atomic_load_n(&g_done, __ATOMIC_SEQ_CST) < THREADS) plat_sleep_ns(1000000);
    for (int i = 1; i < THREADS; i++) CHECK(g_thread_pipe[i] == g_thread_pipe[0] && g_thread_pipe[0]);
    VkpStats s;
    vkp_stats(&s);
    print_stats("threads");
    CHECK(s.links == 1);
    printf("%d threads asked for one pipeline: made once\n", THREADS);
    close_and_report();
    printf("ok synth_threads\n");
}

static void case_synth_absent(const char *pack) {
    open_all(pack, NULL);
    nvn_stage st = {.kind = NVN_STAGE_VS, .ctl_size = 0x57, .code_size = 0x4560, .inst_size = 0x4000};
    snprintf(st.key, sizeof st.key, "%s", NO_KEY);
    VkpStage vs = {.key = NO_KEY, .shape = &st}, ps = {.key = PS_KEY};
    vkp_graphics(&vs, NULL, &ps, &F32);
    printf("returned\n");
}

// HMMA on cooperative matrices: the device and the pipeline builder opened (the builder's open runs the start-up probe,
// which logs the decision) and closed again; with a fault in the probe's read-back when one is named (vkh_fault).
static void case_hmma_probe(const char *fault) {
    if (fault) {
        vkh_fault = !strcmp(fault, "lengths") ? VKH_FAULT_LENGTHS : !strcmp(fault, "helper") ? VKH_FAULT_HELPER
                  : !strcmp(fault, "sums")    ? VKH_FAULT_SUMS : 0;
        CHECK(vkh_fault);
    }
    gpu_vk_open();
    vkp_open(NULL);
    printf("hmma native %d%s%s\n", vkh_native(), vkh_native() ? "" : ": ", vkh_reason());
    vkp_close();
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (unsigned long long)errors, (unsigned long long)warnings);
}

// A translated compute module with HMMA built through the builder, in the form the probe chose (OOT2_VK_HMMA=helper: the
// helper, the native function taken out by spv_hmma_fallback).
static void case_hmma_build(const char *pack, const char *key) {
    open_all(pack, NULL);
    printf("hmma native %d%s%s\n", vkh_native(), vkh_native() ? "" : ": ", vkh_reason());
    VkpStage cs = {.key = key};
    CHECK(vkp_compute(&cs));
    print_stats("hmma");
    // The native form's pipeline is created with the full-subgroups requirement and the helper form's is not. The count is
    // taken from the create info handed to the driver: the validation layer (1.4.357) says nothing of a native pipeline made
    // without the requirement (measured: 0 errors with it removed), so the layer cannot be the check.
    VkpStats made;
    vkp_stats(&made);
    printf("full_subgroups %llu\n", (unsigned long long)made.full_subgroups);
    close_and_report();
    printf("ok hmma_build\n");
}

// ---------- the player's own pack

typedef struct Job {
    char vs[64], ps[64];
    double ms;
} Job;

static Job *g_jobs;
static uint32_t g_njobs, g_next;
static int g_compute_jobs;
static const VkpFormats *g_formats[2] = {&F8, &F16};

static char *trim(char *s) {
    size_t n = strlen(s);
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) s[--n] = 0;
    return s;
}

static void read_jobs(const char *path, int pairs) {
    size_t n;
    char *text = read_all(path, &n), *line, *save = NULL;
    text = realloc(text, n + 1);
    text[n] = 0;
    uint32_t cap = 1024;
    g_jobs = calloc(cap, sizeof *g_jobs);
    for (line = STRTOK(text, "\n", &save); line; line = STRTOK(NULL, "\n", &save)) {
        trim(line);
        if (!*line) continue;
        if (g_njobs == cap) g_jobs = realloc(g_jobs, (cap *= 2) * sizeof *g_jobs), memset(g_jobs + g_njobs, 0, g_njobs * sizeof *g_jobs);
        char *tab = strchr(line, '\t');
        if (pairs) {
            CHECK(tab);
            *tab = 0;
            snprintf(g_jobs[g_njobs].ps, sizeof g_jobs[0].ps, "%s", tab + 1);
        }
        snprintf(g_jobs[g_njobs].vs, sizeof g_jobs[0].vs, "%s", line);
        g_njobs++;
    }
    free(text);
}

static void job_main(void *arg) {
    (void)arg;
    for (;;) {
        uint32_t i = __atomic_fetch_add(&g_next, 1, __ATOMIC_SEQ_CST);
        if (i >= g_njobs) break;
        Job *j = &g_jobs[i];
        uint64_t t0 = plat_now_ns();
        VkpStage a = {.key = j->vs}, b = {.key = j->ps};
        if (g_compute_jobs) {
            CHECK(vkp_compute(&a));
        } else {
            for (int f = 0; f < 2; f++) CHECK(vkp_graphics(&a, NULL, &b, g_formats[f]));
        }
        j->ms = (double)(plat_now_ns() - t0) / 1e6;
    }
    __atomic_add_fetch(&g_done, 1, __ATOMIC_SEQ_CST);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static void run_jobs(int threads) {
    uint64_t t0 = plat_now_ns();
    g_done = 0, g_next = 0;
    for (int i = 0; i < threads; i++) plat_thread_start(job_main, NULL, 8u << 20);
    while (__atomic_load_n(&g_done, __ATOMIC_SEQ_CST) < threads) plat_sleep_ns(10000000);
    double wall = (double)(plat_now_ns() - t0) / 1e9;
    double *ms = malloc((g_njobs ? g_njobs : 1) * sizeof *ms);
    for (uint32_t i = 0; i < g_njobs; i++) ms[i] = g_jobs[i].ms;
    qsort(ms, g_njobs, sizeof *ms, cmp_double);
    printf("jobs %u on %d threads in %.1f s; per job p50 %.3f ms p99 %.3f ms max %.3f ms\n", g_njobs, threads, wall,
           ms[g_njobs / 2], ms[(size_t)((g_njobs - 1) * 0.99)], ms[g_njobs - 1]);
    free(ms);
}

static void case_corpus(const char *pack, const char *list, int pairs, int threads, const char *cache, int save) {
    open_all(pack, strcmp(cache, "-") ? cache : NULL);
    printf("cache_state %d\n", (int)vkp_cache_state());
    read_jobs(list, pairs);
    g_compute_jobs = !pairs;
    run_jobs(threads);
    print_stats(pairs ? "pairs" : "compute");
    // every pair a second time: the same pipelines, nothing made
    VkpStats a, b;
    vkp_stats(&a);
    for (uint32_t i = 0; i < g_njobs; i++) {
        VkpStage x = {.key = g_jobs[i].vs}, y = {.key = g_jobs[i].ps};
        if (pairs) CHECK(vkp_graphics(&x, NULL, &y, g_formats[0]));
        else CHECK(vkp_compute(&x));
    }
    vkp_stats(&b);
    CHECK(a.libraries == b.libraries && a.links == b.links && a.computes == b.computes);
    if (save) {
        char why[256];
        CHECK(vkp_cache_save(why, sizeof why));
        printf("cache saved\n");
    }
    close_and_report();
    printf("ok corpus\n");
}

// tools/firstdraw's draw file, read for the directives the builder's draw needs.
static float *file_floats(const char *path, size_t *bytes) { return read_all(path, bytes); }

static VkFormat attr_format(const char *s) {
    static const struct { const char *name; VkFormat f; } t[] = {
        {"r32f", VK_FORMAT_R32_SFLOAT}, {"rg32f", VK_FORMAT_R32G32_SFLOAT}, {"rgb32f", VK_FORMAT_R32G32B32_SFLOAT},
        {"rgba32f", VK_FORMAT_R32G32B32A32_SFLOAT}, {"r32i", VK_FORMAT_R32_SINT}, {"rg32i", VK_FORMAT_R32G32_SINT},
        {"rgb32i", VK_FORMAT_R32G32B32_SINT}, {"rgba32i", VK_FORMAT_R32G32B32A32_SINT}, {"r32ui", VK_FORMAT_R32_UINT},
        {"rg32ui", VK_FORMAT_R32G32_UINT}, {"rgb32ui", VK_FORMAT_R32G32B32_UINT}, {"rgba32ui", VK_FORMAT_R32G32B32A32_UINT}};
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++)
        if (!strcmp(t[i].name, s)) return t[i].f;
    fprintf(stderr, "unknown attribute format %s\n", s);
    exit(1);
}

static void case_firstdraw(const char *draw, const char *pack, const char *vkey, const char *pkey, const char *out_path) {
    static DrawSpec s;
    static void *keep[256];
    int nkeep = 0;
    size_t n;
    char *text = read_all(draw, &n);
    text = realloc(text, n + 1);
    text[n] = 0;
    char *save = NULL;
    for (char *line = STRTOK(text, "\n", &save); line; line = STRTOK(NULL, "\n", &save)) {
        trim(line);
        char w[32];
        if (sscanf(line, "%31s", w) != 1) continue;
        char *rest = line + strlen(w);
        while (*rest == ' ') rest++;
        unsigned a, b, c, d, e;
        char fmt[32], path[1024], tp[32], filter[32], addr[32], stages[8];
        if (!strcmp(w, "size")) {
            CHECK(sscanf(rest, "%u %u", &a, &b) == 2);
            s.width = a, s.height = b;
        } else if (!strcmp(w, "clear")) {
            CHECK(sscanf(rest, "%f %f %f %f", &s.clear[0], &s.clear[1], &s.clear[2], &s.clear[3]) == 4);
        } else if (!strcmp(w, "vertices")) {
            CHECK(sscanf(rest, "%u %u %1023[^\n]", &a, &b, path) == 3);
            s.vertex_count = a, s.stride = b;
            if (b) {
                size_t vb;
                s.vertices = keep[nkeep++] = file_floats(path, &vb);
                s.vertices_size = (uint32_t)vb;
            }
        } else if (!strcmp(w, "attribute")) {
            CHECK(sscanf(rest, "%u %31s %u", &a, fmt, &b) == 3);
            s.attrs[s.nattrs++] = (DrawAttr){a, attr_format(fmt), b};
        } else if (!strcmp(w, "uniform") || !strcmp(w, "storage")) {
            CHECK(sscanf(rest, "%u %7s %1023[^\n]", &a, stages, path) == 3);
            size_t bytes;
            void *data = keep[nkeep++] = file_floats(path, &bytes);
            s.binds[s.nbinds++] = (DrawBind){a, w[0] == 'u' ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                             data, bytes};
        } else if (!strcmp(w, "image")) {
            CHECK(sscanf(rest, "%u %31s %u %u %u %1023[^\n]", &a, tp, &b, &c, &d, path) == 6);
            size_t bytes;
            VkImageViewType v = !strcmp(tp, "2d") ? VK_IMAGE_VIEW_TYPE_2D : !strcmp(tp, "2darray") ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                : !strcmp(tp, "3d") ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_CUBE;
            s.tex[s.ntex++] = (DrawTex){a, v, b, c, d, keep[nkeep++] = file_floats(path, &bytes)};
        } else if (!strcmp(w, "sampler")) {
            CHECK(sscanf(rest, "%u %31s %31s", &a, filter, addr) == 3);
            s.samp[s.nsamp++] = (DrawSamp){a, !strcmp(filter, "linear") ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                           !strcmp(addr, "repeat") ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
        } else if (!strcmp(w, "push")) {
            CHECK(sscanf(rest, "%u %u", &a, &e) == 2 && a % 4 == 0 && a < VKP_PUSH_BYTES);
            s.push[a / 4] = e;
        }
        (void)c;
    }
    free(text);
    CHECK(s.width && s.height && nkeep < 256);
    s.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    s.out = malloc((size_t)s.width * s.height * 16);
    open_all(pack, NULL);
    VkpStage vs = {.key = vkey}, ps = {.key = pkey};
    s.pipeline = vkp_graphics(&vs, NULL, &ps, &F32);
    VkPipeline p16 = vkp_graphics(&vs, NULL, &ps, &F16);
    CHECK(s.pipeline && p16);
    draw_run(&s);
    FILE *f = fopen(out_path, "wb");
    CHECK(f && fwrite(s.out, 1, (size_t)s.width * s.height * 16, f) == (size_t)s.width * s.height * 16);
    fclose(f);
    print_stats("firstdraw");
    close_and_report();
    printf("ok firstdraw\n");
}

typedef struct Tally {
    uint32_t cap;
    uint64_t count;
} Tally;

static void case_caps(const char *pack, const char *keys, int argc, char **argv) {
    char why[512];
    static ShcSet set;
    CHECK(shc_set_open(&set, pack, NULL, SHC_DESC_SCHEMA, why, sizeof why) == SHC_OK);
    static Oot2VkFacts f;
    VkcList l;
    vkc_list(&f, &l);
    if (argc >= 2 && !strcmp(argv[0], "drop")) {
        uint32_t drop = (uint32_t)strtoul(argv[1], NULL, 10);
        size_t at = 0;
        for (size_t i = 0; i < l.ncaps; i++)
            if (l.caps[i] != drop) l.caps[at++] = l.caps[i];
        l.ncaps = at;
    }
    static Tally caps[256];
    static struct { char name[SPV_EXT_LEN]; uint64_t count; } exts[64];
    size_t ncaps = 0, nexts = 0;
    size_t n;
    char *text = read_all(keys, &n), *save = NULL;
    text = realloc(text, n + 1);
    text[n] = 0;
    uint64_t modules = 0, bad = 0, failed = 0;
    for (char *line = STRTOK(text, "\n", &save); line; line = STRTOK(NULL, "\n", &save)) {
        trim(line);
        char *tab = strchr(line, '\t');
        if (tab) *tab = 0;
        if (!*line) continue;
        ShcEntry e;
        if (shc_set_find(&set, line, NULL, &e) == SHC_MISS) {
            printf("missing %s\n", line);
            bad++;
            continue;
        }
        if (e.status != SHC_TRANSLATED) {
            failed++;
            continue;
        }
        modules++;
        SpvDecl d;
        CHECK(spv_declarations(e.module, e.module_bytes / 4, &d, why, sizeof why));
        for (uint32_t i = 0; i < d.ncaps; i++) {
            size_t k = 0;
            while (k < ncaps && caps[k].cap != d.caps[i]) k++;
            if (k == ncaps) {
                CHECK(ncaps < 256);
                caps[ncaps++].cap = d.caps[i];
            }
            caps[k].count++;
        }
        for (uint32_t i = 0; i < d.nexts; i++) {
            size_t k = 0;
            while (k < nexts && strcmp(exts[k].name, d.exts[i])) k++;
            if (k == nexts) {
                CHECK(nexts < 64);
                snprintf(exts[nexts++].name, SPV_EXT_LEN, "%s", d.exts[i]);
            }
            exts[k].count++;
        }
        if (vkc_check(&l, &d, why, sizeof why)) {
            bad++;
            if (bad <= 5) printf("first refusals: %s: %s\n", line, why);
        }
    }
    printf("modules %llu, failure records %llu, refused %llu\n", (unsigned long long)modules, (unsigned long long)failed,
           (unsigned long long)bad);
    for (size_t i = 0; i < ncaps; i++) {
        const char *name = spv_capability_name(caps[i].cap);
        printf("cap %u %s %llu %s\n", caps[i].cap, name ? name : "?", (unsigned long long)caps[i].count,
               vkc_has_cap(&l, caps[i].cap) ? "in the list" : "NOT in the list");
    }
    for (size_t i = 0; i < nexts; i++) {
        size_t k = 0;
        while (k < l.nexts && strcmp(l.exts[k], exts[i].name)) k++;
        printf("ext %s %llu %s\n", exts[i].name, (unsigned long long)exts[i].count, k < l.nexts ? "in the list" : "NOT in the list");
    }
    free(text);
    vkp_store_close();
    printf("%s caps\n", bad ? "refused" : "ok");
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    const char *cs = argv[1];
    if (!strcmp(cs, "tables")) case_tables();
    else if (!strcmp(cs, "keys")) case_keys();
    else if (!strcmp(cs, "desc") && argc >= 3) case_desc(argv[2]);
    else if (!strcmp(cs, "pad") && argc >= 4) case_pad(argv[2], argv[3]);
    else if (!strcmp(cs, "meshlink") && argc >= 4) case_meshlink(argv[2], argv[3]);
    else if (!strcmp(cs, "binds") && argc >= 3) case_binds(argv[2]);
    else if (!strcmp(cs, "spvdecl") && argc >= 3) case_spvdecl(argv[2]);
    else if (!strcmp(cs, "spvfull") && argc >= 3) case_spvfull(argv[2]);
    else if (!strcmp(cs, "spvpad") && argc >= 5) case_spvpad(argv[2], argv[3], argv[4]);
    else if (!strcmp(cs, "spvmeshpad") && argc >= 5)
        case_spvmeshpad(argv[2], argv[3], argv[4], argc >= 6 ? (uint32_t)strtoul(argv[5], NULL, 10) : 0,
                        argc >= 7 ? (uint32_t)strtoul(argv[6], NULL, 10) : 0,
                        argc >= 8 ? (uint32_t)strtoul(argv[7], NULL, 10) : 0,
                        argc >= 9 ? (uint32_t)strtoul(argv[8], NULL, 10) : 0);
    else if (!strcmp(cs, "meshlibkey") && argc >= 4) case_meshlibkey(argv[2], argc - 3, argv + 3);
    else if (!strcmp(cs, "meshpayload") && argc >= 3) case_meshpayload(argv[2]);
    else if (!strcmp(cs, "spvzero") && argc >= 4) case_spvzero(argv[2], argv[3]);
    else if (!strcmp(cs, "spvhmma") && argc >= 4) case_spvhmma(argv[2], argv[3]);
    else if (!strcmp(cs, "capcheck") && argc >= 3) case_capcheck(argv[2], argc - 3, argv + 3);
    else if (!strcmp(cs, "optional")) case_optional();
    else if (!strcmp(cs, "cachejudge")) case_cachejudge();
    else if (!strcmp(cs, "savesettings")) case_savesettings();
    else if (!strcmp(cs, "cacheread") && argc >= 3) case_cacheread(argv[2], argc - 3, argv + 3);
    else if (!strcmp(cs, "cachemove") && argc >= 4) case_cachemove(argv[2], argv[3], argc >= 5 && !strcmp(argv[4], "heal"));
    else if (!strcmp(cs, "store") && argc >= 7) case_store(argv[2], argv[3], argv[4], argv[5], argv[6]);
    else if (!strcmp(cs, "storemap") && argc >= 9)
        case_storemap(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argc > 9 ? argv[9] : "");
    else if (!strcmp(cs, "storeopen") && argc >= 4) case_storeopen(argv[2], argv[3]);
    else if (!strcmp(cs, "devbinds") && argc >= 5) case_devbinds(argv[2], argv[3], atoi(argv[4]));
    else if (!strcmp(cs, "linkchoice") && argc >= 6) case_linkchoice(argv[2], argv[3], argv[4], argv[5]);
    else if (!strcmp(cs, "usable")) case_usable();
    else if (!strcmp(cs, "caps") && argc >= 4) case_caps(argv[2], argv[3], argc - 4, argv + 4);
    else {
        // the cases below open the device: a machine without a loader or a device skips
        if (volkInitialize() != VK_SUCCESS) {
            printf("no Vulkan loader\n");
            return EXIT_NO_DEVICE;
        }
        if (!strcmp(cs, "layouts")) case_layouts();
        else if (!strcmp(cs, "synth_draw") && argc >= 3) case_synth_draw(argv[2], argc >= 4 && !strcmp(argv[3], "unpad"));
        else if (!strcmp(cs, "synth_compute") && argc >= 3) case_synth_compute(argv[2]);
        else if (!strcmp(cs, "synth_meshpad") && argc >= 3) case_synth_meshpad(argv[2]);
        else if (!strcmp(cs, "synth_zeroinit") && argc >= 3) case_synth_zeroinit(argv[2]);
        else if (!strcmp(cs, "synth_inflight") && argc >= 3) case_synth_inflight(argv[2]);
        else if (!strcmp(cs, "synth_cache") && argc >= 5) case_synth_cache(argv[2], argv[3], argv[4]);
        else if (!strcmp(cs, "synth_save") && argc >= 4) case_synth_save(argv[2], argv[3], argc - 4, argv + 4);
        else if (!strcmp(cs, "synth_stopcompile") && argc >= 4) case_synth_stopcompile(argv[2], argv[3], argc >= 5 ? atoi(argv[4]) : 300);
        else if (!strcmp(cs, "synth_release") && argc >= 3) case_synth_release(argv[2]);
        else if (!strcmp(cs, "synth_threads") && argc >= 3) case_synth_threads(argv[2]);
        else if (!strcmp(cs, "synth_absent") && argc >= 3) case_synth_absent(argv[2]);
        else if (!strcmp(cs, "hmma_probe")) case_hmma_probe(argc >= 3 ? argv[2] : NULL);
        else if (!strcmp(cs, "hmma_build") && argc >= 4) case_hmma_build(argv[2], argv[3]);
        else if (!strcmp(cs, "firstdraw") && argc >= 7) case_firstdraw(argv[2], argv[3], argv[4], argv[5], argv[6]);
        else if (!strcmp(cs, "corpus_pairs") && argc >= 6) case_corpus(argv[2], argv[3], 1, atoi(argv[4]), argv[5], argc >= 7);
        else if (!strcmp(cs, "corpus_compute") && argc >= 6) case_corpus(argv[2], argv[3], 0, atoi(argv[4]), argv[5], argc >= 7);
        else return 2;
    }
    fflush(stdout);
    return 0;
}
