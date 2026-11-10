// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the replay (oot2/vk_replay.h) needs of the device: the pipeline builder, the descriptions of the stages it
// binds, the backing map, the images render targets stand on and their attachment views, the bindless tables, the
// frame's scratch memory, and the occlusion query pools of the samples-passed counter.
//
// Scratch memory comes in chunks of host-visible, coherent memory, each with one buffer over it; a frame takes chunks
// from a free list and hands them back from the completion thread once its value is done, so a chunk the device may
// still read is never written. Attachment views are made per frame and destroyed the same way (a render target change
// is rare next to a draw: the NVN2 call census counts 127 a Forest frame).
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/bench_timers.h"
#include "oot2/bindless.h"
#include "oot2/gpu_vk.h"
#include "oot2/nvn_program.h"
#include "oot2/runtime.h"
#include "oot2/shader_cache.h"
#include "oot2/shader_desc.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_capture.h"
#include "oot2/vk_format.h"
#include "oot2/vk_image.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_raw.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_view_plan.h"
#include "oot2/vk_replay.h"
#include "plat.h"

// ---------- the descriptions of stages, parsed once per stage record (records are immortal: the programs and state study)
#define BIND_BUCKETS 4096
typedef struct BindEntry {
    const nvn_stage *stage;
    uint8_t content_digest[SHC_DIGEST_BYTES];
    ShaderBinds binds;
    struct BindEntry *next;
} BindEntry;
static BindEntry *g_bind[BIND_BUCKETS];
static PlatMutex g_bind_lock = PLAT_MUTEX_INIT;

static const ShaderBinds *dev_binds(const VkpStage *stage) {
    const nvn_stage *s = stage->shape;
    uint8_t content_digest[SHC_DIGEST_BYTES];
    BT_TIME(BT_S2_CONTENT, vkp_stage_content_digest(stage, content_digest));
    uint32_t h = (uint32_t)(((uintptr_t)s >> 4) * 0x9E3779B1u);
    for (size_t i = 0; i < sizeof content_digest; i++) h = (h ^ content_digest[i]) * 0x01000193u;
    h %= BIND_BUCKETS;   // after the digest is mixed in: the mixing leaves any 32-bit value
    plat_mutex_lock(&g_bind_lock);
    for (BindEntry *e = g_bind[h]; e; e = e->next)
        if (e->stage == s && !memcmp(e->content_digest, content_digest, sizeof content_digest)) {
            plat_mutex_unlock(&g_bind_lock);
            return &e->binds;
        }
    plat_mutex_unlock(&g_bind_lock);
    VkpStage st = *stage;
    VkpModule m;
    vkp_module(&st, s->kind, &m);   // the stage kinds are SHD_*'s order
    BindEntry *e = calloc(1, sizeof *e);
    if (!e) rt_fatal("gfx: vulkan: out of memory for the description of stage %s", s->key);
    char why[256];
    if (!shd_parse_binds(m.description, m.description_bytes, SHC_DESC_SCHEMA, &e->binds, why, sizeof why))
        rt_fatal("gfx: vulkan: the description of stage %s: %s", s->key, why);
    e->stage = s;
    memcpy(e->content_digest, content_digest, sizeof content_digest);
    plat_mutex_lock(&g_bind_lock);
    for (BindEntry *existing = g_bind[h]; existing; existing = existing->next)
        if (existing->stage == s && !memcmp(existing->content_digest, content_digest, sizeof content_digest)) {
            plat_mutex_unlock(&g_bind_lock);
            free(e);
            return &existing->binds;
        }
    e->next = g_bind[h], g_bind[h] = e;
    plat_mutex_unlock(&g_bind_lock);
    return &e->binds;
}

// ---------- render targets
static void view_gone(void *arg, uint64_t value) {
    (void)value;
    vkDestroyImageView(gpu_vk.device, (VkImageView)(uintptr_t)arg, NULL);
}

static int dev_target(const nvn_tex_rec *t, int storage, uint64_t value, VkrTarget *out) {
    memset(out, 0, sizeof *out);
    VkbImage *img = vkb_image_get(&t->tex);
    // a target in storage the CPU can see: its arena image, made current so the pass draws over the CPU's bytes
    if (!img) img = vkb_image_demand_current(&t->tex, value);
    if (!img) return 0;
    vkb_image_use(img, value);
    int fmt = t->has_view && t->view.set_format ? t->view.format : t->tex.d.format;
    const VkfRow *row = vkf_row(fmt);
    if (!row || row->vk == VK_FORMAT_UNDEFINED) return 0;
    uint32_t level = t->has_view && t->view.set_levels ? t->view.base_level : 0;
    uint32_t base_layer = t->has_view && t->view.set_layers ? t->view.min_layer : 0;
    uint32_t layers = t->has_view && t->view.set_layers && t->view.nlayers ? t->view.nlayers : 1;
    VkImageAspectFlags aspect = 0;
    if (row->flags & VKF_DEPTH) aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (row->flags & VKF_STENCIL) aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!aspect) aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageUsageFlags want_usage = storage ? VK_IMAGE_USAGE_STORAGE_BIT
                                   : aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                                                         : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    // what the image gives for this level and these layers (a slice of a volume is a 2D view of it, its depth slices
    // counted as layers); a view it cannot give stops by name, with the texture, what was asked and what the image has
    VkbShape sh;
    vkb_image_shape(img, &sh);
    VkvImage vim = {sh.type, sh.depth, sh.layers, sh.levels, sh.flags, sh.usage, gpu_vk.image_2d_view_of_3d};
    VkvWant vw = {VKV_PICK, level, base_layer, layers, want_usage};
    VkvPlan plan;
    const char *refusal;
    // An image made without the usage (its format gives none: the driver took such views before) is no target: skipped, the
    // way an unusable format is, and named once per kind of usage.
    if (vkv_usage_gap(&vim, want_usage)) {
        static unsigned logged;
        unsigned kind = storage ? 1u : aspect == VK_IMAGE_ASPECT_COLOR_BIT ? 2u : 4u;
        if (!(logged & kind)) {
            logged |= kind;
            vkv_plan(&vim, &vw, &plan, &refusal);
            rt_log("gfx: vulkan: the %s view of the texture at %#llx (NVN format %#x) is skipped: %s (usage %#x)",
                   storage ? "storage" : "render target", (unsigned long long)(uintptr_t)t->tex.mem, fmt, refusal,
                   (unsigned)sh.usage);
        }
        return 0;
    }
    if (!vkv_plan(&vim, &vw, &plan, &refusal))
        rt_fatal("gfx: vulkan: the %s view of the texture at %#llx (NVN format %#x, level %u, layers %u..%u) cannot be made: "
                 "%s (the image has %u levels, %u layers, a depth of %u and usage %#x)",
                 storage ? "storage" : "render target", (unsigned long long)(uintptr_t)t->tex.mem, fmt, level, base_layer,
                 base_layer + layers, refusal, sh.levels, sh.layers, sh.depth, (unsigned)sh.usage);
    base_layer = plan.base_layer, layers = plan.layers;
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = want_usage};
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &use,
                                .image = vkb_image_handle(img),
                                .viewType = plan.type,
                                .format = row->vk, .subresourceRange = {aspect, level, 1, base_layer, layers}};
    VkImageView view;
    GPU_VK_OK(vkCreateImageView(gpu_vk.device, &vi, NULL, &view), "vkCreateImageView (a render target)");
    vkrec_effect(view_gone, (void *)(uintptr_t)view);
    uint32_t w = t->tex.d.width >> level, h = t->tex.d.height >> level;
    *out = (VkrTarget){view, row->vk, w ? w : 1, h ? h : 1, layers, aspect, (uint64_t)(uintptr_t)vkb_image_handle(img)};
    return 1;
}

// ---------- the images copies and clears work on
static int dev_image(const nvn_tex_rec *t, uint64_t value, VkrImage *out) {
    memset(out, 0, sizeof *out);
    VkbImage *img = vkb_image_get(&t->tex);
    if (!img) img = vkb_image_demand_current(&t->tex, value);   // a texture the CPU writes: its arena image
    if (!img) return 0;
    vkb_image_use(img, value);
    VkbShape sh;
    vkb_image_shape(img, &sh);
    *out = (VkrImage){vkb_image_handle(img), sh.format, sh.type, sh.aspect, sh.all_aspects, sh.width, sh.height, sh.depth,
                      sh.layers, sh.levels, sh.samples};
    return 1;
}

// The image a texture pool ID's texture stands on, for the movie blit (settings.h test_movie_blit): the image the storage view
// of that ID is made over, with the storage range its stores are declared over when the image is placed in it.
static int dev_image_by_id(uint32_t id, uint64_t value, VkrImage *out, uint64_t *storage, uint64_t *bytes) {
    tex_t t;
    view_t v;
    if (!bindless_texture_entry(id, &t, &v)) return 0;
    nvn_tex_rec rec;
    memset(&rec, 0, sizeof rec);
    rec.tex = t, rec.view = v, rec.present = 1, rec.has_view = 1;
    if (!dev_image(&rec, value, out)) return 0;
    VkbImage *img = vkb_image_get(&t);
    int placed = img && vkb_image_placed(img);
    if (storage) *storage = placed ? (uint64_t)(uintptr_t)t.mem : 0;
    if (bytes) *bytes = placed ? t.L.size : 0;
    return 1;
}

// ---------- the bindless tables, recorded into the frame's prologue
static VkDescriptorSet dev_acquire(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers) {
    return vkb_acquire(gpu_vk_prologue(), value, textures, samplers);
}
static uint32_t dev_image_slot(uint32_t id, int type) { return vkb_image_slot(gpu_vk_prologue(), id, type); }
// A storage view is destroyed by the completion thread as an attachment view is (view_gone): it holds nothing of the image
// layer's, whose lock a thread may hold while the timeline waits for the completion thread.
static VkImageView dev_storage_image(uint32_t id, int type, uint64_t value, uint64_t *storage, uint64_t *bytes) {
    tex_t t;
    view_t v;
    if (!bindless_texture_entry(id, &t, &v)) return VK_NULL_HANDLE;
    VkImageView view = vkb_storage_view(&t, &v, type, value);
    gpu_vk_effect(view_gone, (void *)(uintptr_t)view);
    // A placed image is its storage's VRAM extent: the stores are a GPU write there. An arena image keeps them in the
    // image, newer than the guest storage the CPU sees (oot2/vk_image.h): nothing is declared to the backing map.
    VkbImage *img = vkb_image_get(&t);
    int placed = img && vkb_image_placed(img);
    *storage = placed ? (uint64_t)(uintptr_t)t.mem : 0, *bytes = placed ? t.L.size : 0;
    return view;
}

// ---------- scratch memory
#define CHUNK_BYTES (4u << 20)
typedef struct Chunk {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceAddress address;
    uint8_t *map;
    uint32_t size, used;
    struct Chunk *next;
} Chunk;
static Chunk *g_free;                  // chunks no frame holds (g_chunk_lock)
static PlatMutex g_chunk_lock = PLAT_MUTEX_INIT;
static struct {
    uint64_t value;                    // the frame the chunks below belong to
    Chunk *held;                       // its chunks, the current one first
    VkDescriptorPool task_pool;
    uint32_t task_sets;
} g_frame;

static void task_pool_gone(void *arg, uint64_t value) {
    (void)value;
    vkDestroyDescriptorPool(gpu_vk.device, (VkDescriptorPool)(uintptr_t)arg, NULL);
}

static void task_pool_new(void) {
    static const uint32_t sets_per_pool = 512;
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sets_per_pool * 8},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets_per_pool * 8}};
    VkDescriptorPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .maxSets = sets_per_pool, .poolSizeCount = 2, .pPoolSizes = sizes};
    GPU_VK_OK(vkCreateDescriptorPool(gpu_vk.device, &ci, NULL, &g_frame.task_pool),
              "vkCreateDescriptorPool (task-stage resources)");
    g_frame.task_sets = 0;
    vkrec_effect(task_pool_gone, (void *)(uintptr_t)g_frame.task_pool);
}

static VkDescriptorSet dev_task_set(uint64_t value, uint32_t n, const VkWriteDescriptorSet *writes) {
    if (g_frame.value != value) rt_fatal("gfx: vulkan: task descriptors requested outside their frame's scratch");
    if (n > VKP_SET2_MAX) rt_fatal("gfx: vulkan: task stage has %u set-2 descriptors, at most %u fit", n, VKP_SET2_MAX);
    if (!g_frame.task_pool || g_frame.task_sets == 512) task_pool_new();
    const VkpLayouts *layouts = vkp_layouts();
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = g_frame.task_pool, .descriptorSetCount = 1,
                                      .pSetLayouts = &layouts->task};
    VkDescriptorSet set;
    VkResult r = vkAllocateDescriptorSets(gpu_vk.device, &ai, &set);
    if (r == VK_ERROR_OUT_OF_POOL_MEMORY || r == VK_ERROR_FRAGMENTED_POOL) {
        task_pool_new();
        ai.descriptorPool = g_frame.task_pool;
        GPU_VK_OK(vkAllocateDescriptorSets(gpu_vk.device, &ai, &set),
                  "vkAllocateDescriptorSets (task-stage resources)");
    } else GPU_VK_OK(r, "vkAllocateDescriptorSets (task-stage resources)");
    g_frame.task_sets++;
    VkWriteDescriptorSet updated[VKP_SET2_MAX];
    memcpy(updated, writes, n * sizeof *updated);
    for (uint32_t i = 0; i < n; i++) updated[i].dstSet = set;
    vkUpdateDescriptorSets(gpu_vk.device, n, updated, 0, NULL);
    return set;
}

static VkDeviceAddress dev_buffer_address(VkBuffer buffer) {
    for (Chunk *c = g_frame.held; c; c = c->next)
        if (c->buf == buffer) return c->address;
    return 0;
}

static Chunk *chunk_new(uint32_t size) {
    Chunk *c = calloc(1, sizeof *c);
    if (!c) rt_fatal("gfx: vulkan: out of memory for a scratch chunk");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &c->buf), "vkCreateBuffer (replay scratch)");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu_vk.device, c->buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
        if ((req.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) type = i;
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no host-visible coherent memory for the replay's scratch");
    VkMemoryAllocateFlagsInfo flags = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                       .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &flags,
                                  .allocationSize = req.size, .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &c->mem), "vkAllocateMemory (replay scratch)");
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, c->buf, c->mem, 0), "vkBindBufferMemory (replay scratch)");
    VkBufferDeviceAddressInfo address = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = c->buf};
    c->address = vkGetBufferDeviceAddress(gpu_vk.device, &address);
    void *p;
    GPU_VK_OK(vkMapMemory(gpu_vk.device, c->mem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory (replay scratch)");
    c->map = p, c->size = size;
    return c;
}

static void chunks_back(void *arg, uint64_t value) {
    (void)value;
    Chunk *c = arg;
    plat_mutex_lock(&g_chunk_lock);
    while (c) {
        Chunk *next = c->next;
        c->used = 0, c->next = g_free, g_free = c;
        c = next;
    }
    plat_mutex_unlock(&g_chunk_lock);
}

static void *dev_scratch(uint64_t value, uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset) {
    if (g_frame.value != value) g_frame.value = value, g_frame.held = NULL, g_frame.task_pool = VK_NULL_HANDLE,
                                 g_frame.task_sets = 0;
    Chunk *c = g_frame.held;
    uint32_t at = c ? (c->used + align - 1) & ~(align - 1) : 0;
    if (!c || at + bytes > c->size) {
        c = NULL;
        if (bytes <= CHUNK_BYTES) {
            plat_mutex_lock(&g_chunk_lock);
            if (g_free) c = g_free, g_free = c->next;
            plat_mutex_unlock(&g_chunk_lock);
        }
        if (!c) c = chunk_new(bytes > CHUNK_BYTES ? bytes : CHUNK_BYTES);
        c->used = 0, c->next = g_frame.held, g_frame.held = c;
        at = 0;
    }
    c->used = at + bytes;
    *buffer = c->buf, *offset = at;
    return c->map + at;
}

// ---------- around a frame
static void dev_frame_begin(void) {
    backing_sync();
    vkm_retire();   // the pipelines of programs the game finalized, once the timeline has passed their last frame
}

static void dev_frame_end(uint64_t value) {
    vkb_image_flush(gpu_vk_prologue());   // the images made for the frame, to the general layout before its work
    gpu_vk_raw_end(value);                // raw stores into mirrors and the miss log, when the frame used the table
    backing_submit_end(value);
    if (g_frame.value == value && g_frame.held) vkrec_effect(chunks_back, g_frame.held);
    g_frame.held = NULL, g_frame.value = 0;
}

// ---------- occlusion queries: pools of VKR_QUERIES from a free list, each reset in the prologue of the frame that takes
// it and back on the list once the completion thread read its results
static VkQueryPool *g_qfree;
static uint32_t g_nqfree, g_qcap, g_qmade;
static PlatMutex g_qlock = PLAT_MUTEX_INIT;

static VkQueryPool dev_queries(uint64_t value) {
    (void)value;
    VkQueryPool p = VK_NULL_HANDLE;
    plat_mutex_lock(&g_qlock);
    if (g_nqfree) p = g_qfree[--g_nqfree];
    plat_mutex_unlock(&g_qlock);
    if (!p) {
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_OCCLUSION,
                                    .queryCount = VKR_QUERIES};
        GPU_VK_OK(vkCreateQueryPool(gpu_vk.device, &qi, NULL, &p), "vkCreateQueryPool (samples passed)");
        plat_mutex_lock(&g_qlock);
        g_qmade++;
        plat_mutex_unlock(&g_qlock);
    }
    vkCmdResetQueryPool(gpu_vk_prologue(), p, 0, VKR_QUERIES);
    return p;
}

// The frame is done, so every query it ended is available: WAIT only guards against a driver that is late saying so.
static void dev_query_results(VkQueryPool p, uint32_t n, uint64_t *out) {
    if (n)
        GPU_VK_OK(vkGetQueryPoolResults(gpu_vk.device, p, 0, n, sizeof *out * n, out, sizeof *out,
                                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                  "vkGetQueryPoolResults (samples passed)");
    plat_mutex_lock(&g_qlock);
    if (g_nqfree == g_qcap) {
        uint32_t cap = g_qcap ? 2 * g_qcap : 8;
        VkQueryPool *f = realloc(g_qfree, sizeof *f * cap);
        if (!f) rt_fatal("gfx: vulkan: out of memory for %u query pools", cap);
        g_qfree = f, g_qcap = cap;
    }
    g_qfree[g_nqfree++] = p;
    plat_mutex_unlock(&g_qlock);
}

// The condition's words: the front end's pools, as the null back end reads them (gfx.h, nvn_pool_find).
static int dev_cpu_reads(uint64_t addr, uint64_t size) {
    const pool_t *p = nvn_pool_find(addr, size);
    return p && !(p->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
}

// The trace (OOT2_REPLAY_TRACE) asks what the bindless tables hold at a texture ID: the entry and its format, nothing made.
static int dev_trace_tex(uint32_t id, uint32_t *format, int *astc) {
    tex_t t;
    view_t v;
    if (!bindless_texture_entry(id, &t, &v)) return 0;
    *format = v.set_format ? v.format : t.d.format;
    const VkfRow *row = vkf_row((int)*format);
    *astc = row && (row->flags & VKF_ASTC) != 0;
    return 1;
}

const VkReplayOps *vkr_device_ops(void) {
    static VkReplayOps ops = {.graphics = vkm_graphics, .compute = vkm_compute, .program_bound = vkm_program_bound,
                              .binds = dev_binds,
                              .resolve = backing_resolve, .gpu_wrote = backing_gpu_wrote, .target = dev_target,
                              .image = dev_image,
                              .acquire = dev_acquire, .image_slot = dev_image_slot,
                              .storage_image = dev_storage_image, .image_by_id = dev_image_by_id, .scratch = dev_scratch,
                              .buffer_address = dev_buffer_address, .task_set = dev_task_set,
                              .raw = gpu_vk_raw_begin, .frame_begin = dev_frame_begin, .frame_end = dev_frame_end,
                              .queries = dev_queries, .query_results = dev_query_results, .cpu_reads = dev_cpu_reads,
                              .trace_tex = dev_trace_tex};
    const VkpLayouts *l = vkp_layouts();   // read at every call: the pipeline builder may open after the first frame
    ops.graphics_layout = l->graphics, ops.compute_layout = l->compute;
    ops.raw_snap = gpu_vk_raw_diag() ? gpu_vk_raw_snap : NULL;   // settings.h raw_diag: off, no snapshot packets
    return &ops;
}

void vkr_device_close(void) {
    // Every frame is done (gpu_vk_close finished the timeline first), so every chunk is back on the free list.
    if (g_frame.held) rt_fatal("gfx: vulkan: the replay's scratch is freed inside a frame");
    plat_mutex_lock(&g_chunk_lock);
    while (g_free) {
        Chunk *c = g_free;
        g_free = c->next;
        vkDestroyBuffer(gpu_vk.device, c->buf, NULL);
        vkFreeMemory(gpu_vk.device, c->mem, NULL);
        free(c);
    }
    plat_mutex_unlock(&g_chunk_lock);
    plat_mutex_lock(&g_bind_lock);   // the parsed descriptions: a device opened again parses them again
    for (uint32_t i = 0; i < BIND_BUCKETS; i++)
        while (g_bind[i]) {
            BindEntry *e = g_bind[i];
            g_bind[i] = e->next;
            free(e);
        }
    plat_mutex_unlock(&g_bind_lock);
    plat_mutex_lock(&g_qlock);   // every frame is done, so every pool is back
    if (g_nqfree != g_qmade) rt_fatal("gfx: vulkan: %u of %u query pools are not back at the device's close", g_qmade - g_nqfree, g_qmade);
    while (g_nqfree) vkDestroyQueryPool(gpu_vk.device, g_qfree[--g_nqfree], NULL);
    free(g_qfree);
    g_qfree = NULL, g_qcap = g_qmade = 0;
    plat_mutex_unlock(&g_qlock);
    vkc_close();   // the offscreen capture (oot2/vk_capture.h)
}
