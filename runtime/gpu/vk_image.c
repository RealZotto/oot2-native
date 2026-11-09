// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Images, sampled views and texel copies (oot2/vk_image.h). The image cache keyed by storage address and shape and the
// rule that an image the GPU wrote is the content follow DKB Recomp's runtime/vk_texture.c (GPL-3.0-or-later; its
// behaviour was ported there from MacNativeGames d2r_gfx and LAS Recomp, MIT): changed here, an image is placed at its
// pool offset in a VRAM extent of the backing map instead of owning memory, images of one format class are one image
// with mutable formats, and the transcode is not here. The images of textures the CPU writes live in the texture arena
// (oot2/texmem.h) and are uploaded by runtime/gpu/vk_texture.c: made when the replay binds the texture for the first time
// (vkb_view_demand), made current for each submission that binds it (vkb_view_use).
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu_vk.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/texmem.h"
#include "oot2/untile.h"
#include "oot2/vk_format.h"
#include "oot2/vk_image.h"
#include "oot2/vk_texture.h"
#include "plat.h"

typedef unsigned long long ull;
#define BUCKETS 1024u

typedef struct ImgKey {
    uint64_t storage;
    uint32_t type, w, h, d, layers, levels, samples, cls, linear;
    uint32_t pkg, layout_flags, stride;   // what else decides where the texels lie in storage the CPU wrote
} ImgKey;

struct VkbImage {
    ImgKey key;
    VkImage image;
    VkDeviceMemory dedicated;     // memory of its own when it could not be placed
    VkFormat format;
    VkImageAspectFlags aspect, all_aspects;   // the one a copy takes; every aspect of the format
    VkImageCreateFlags flags;
    VkImageUsageFlags usage;
    uint64_t size;                // the game's storage bytes
    int placed, doomed, refs, pending;
    int arena;                    // its memory is an extent of the texture arena: the CPU writes its texels (vk_texture.c)
    int decoded;                  // an ASTC texture the device does not sample: this image holds its decoded RGBA8 texels
    TexExtent ext;
    nvn_texdesc desc;             // the texture it was made for (an arena image uploads by it)
    VktState up;
    uint64_t last_use;            // newest submission value that used it directly (atomic)
    struct VkbImage *next, *next_aux, *next_doom;   // the bucket chain; the pending list; the doomed list
};

typedef struct Vw {
    VkImageView view;
    VkbImage *img;
} Vw;

static PlatMutex g_mu = PLAT_MUTEX_INIT;
static VkbImage *g_hash[BUCKETS];
static VkbImage *g_pending, *g_doomed;
static VkbImageStats g_st;
static PlatMutex g_xfer_mu = PLAT_MUTEX_INIT;
static VkBuffer g_stage;
static VkDeviceMemory g_stage_mem;
static void *g_stage_map;
static VkDeviceSize g_stage_size;
static uint32_t g_logged_formats[8];   // bit per NVN format (0x00 to 0xFF) already reported

static uint64_t done(void) { return gpu_sync_done(); }

// ---------- formats

static VkFormatFeatureFlags features(VkFormat f) {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(gpu_vk.phys, f, &p);
    return p.optimalTilingFeatures;
}

static VkImageUsageFlags usage_of(VkFormat f) {
    VkFormatFeatureFlags ft = features(f);
    VkImageUsageFlags u = 0;
    if (ft & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) u |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (ft & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) u |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (ft & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) u |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (ft & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (ft & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (ft & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    return u;
}

static int supported(VkFormat f, int depth_or_stencil, void *ctx) {
    (void)ctx;
    VkFormatFeatureFlags ft = features(f);
    if (!(ft & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) return 0;
    return !depth_or_stencil || (ft & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
}

static void log_skip(int nvn_format, const char *what, const char *why) {
    if (nvn_format >= 0 && nvn_format < 256 && !(g_logged_formats[nvn_format / 32] >> (nvn_format % 32) & 1)) {
        g_logged_formats[nvn_format / 32] |= 1u << (nvn_format % 32);
        rt_log("gfx: vulkan: no image for NVN format %#x (%s): %s", nvn_format, what, why ? why : "no reason given");
    }
}

static VkImageAspectFlags aspect_of(const VkfRow *r, int dsmode) {
    if ((r->flags & (VKF_DEPTH | VKF_STENCIL)) == (VKF_DEPTH | VKF_STENCIL))
        return dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if (r->flags & VKF_DEPTH) return VK_IMAGE_ASPECT_DEPTH_BIT;
    if (r->flags & VKF_STENCIL) return VK_IMAGE_ASPECT_STENCIL_BIT;
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

// ---------- the shape of a texture as an image

typedef struct Shape {
    VkImageType type;
    uint32_t w, h, d, layers, levels, samples;
    VkImageCreateFlags flags;
} Shape;

static void shape_of(const nvn_texdesc *d, Shape *s) {
    memset(s, 0, sizeof *s);
    s->w = d->width ? d->width : 1, s->h = d->height ? d->height : 1, s->d = d->depth ? d->depth : 1;
    s->layers = 1, s->levels = d->levels ? d->levels : 1;
    s->samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    s->type = VK_IMAGE_TYPE_2D;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: s->type = VK_IMAGE_TYPE_1D, s->h = 1, s->d = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: s->type = VK_IMAGE_TYPE_1D, s->layers = s->h, s->h = 1, s->d = 1; break;
    case NVN_TEXTURE_TARGET_3D: s->type = VK_IMAGE_TYPE_3D; break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: s->layers = s->d, s->d = 1; break;
    case NVN_TEXTURE_TARGET_CUBEMAP: s->layers = 6, s->d = 1, s->flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: s->layers = s->d, s->d = 1, s->flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT; break;
    default: s->d = 1; break;
    }
}

static uint32_t bucket(uint64_t storage) { return (uint32_t)(storage >> 12) % BUCKETS; }

static int key_equal(const ImgKey *a, const ImgKey *b) { return !memcmp(a, b, sizeof *a); }

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((bits >> i & 1) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}

// ---------- images

// The usage of an arena image: sampled, copies in (the upload) and out (the checks), and stores where the format takes
// them.
static VkImageUsageFlags arena_usage(VkFormat f) {
    VkFormatFeatureFlags ft = features(f);
    VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (ft & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) u |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if (ft & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) u |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // a program may store to a texture the CPU wrote (the title's compute passes do, into an RGBA8 one)
    if (ft & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    // and a pass may render into one (after the intro, an RG8 target of 260x260)
    if (ft & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    return u;
}

// Block-compressed images (BC, and ASTC where the device samples it) can be viewed as the uncompressed format of their block
// size, a texel of the view being a block (the transcode writes BC blocks through such a view). A view's usage must be a
// subset of the image's, and the extended-usage flag only lets the image carry usage its own format does not support (a
// block-compressed format has no storage), so the image asks for storage usage where the device grants it; a device that
// does not keeps the raw views without it, and one that grants neither keeps the image. Without the flags a raw view has no
// usage at all.
#define RAW_VIEW_FLAGS (VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT)

static VkbImage *image_make(const tex_t *t, const VkfRow *row, VkFormat vk, const ImgKey *key, const Shape *sh, int arena) {
    VkFormat fmt = vkf_linear_twin(vk);
    VkImageUsageFlags usage = arena ? arena_usage(fmt) : usage_of(fmt);
    VkImageCreateFlags flags = sh->flags | (arena ? 0 : VK_IMAGE_CREATE_ALIAS_BIT);
    if (!(row->flags & (VKF_DEPTH | VKF_STENCIL))) flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    const int compressed = vkf_class(fmt) >= 2001 && vkf_class(fmt) < 4000;   // the BC and ASTC families
    const int decoded = vkf_decoded(row, fmt);   // an ASTC texture the device does not sample: its image holds RGBA8 texels
    int raw_storage = 0;
    if (compressed && (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT)) {
        flags |= RAW_VIEW_FLAGS;
        usage |= VK_IMAGE_USAGE_STORAGE_BIT, raw_storage = 1;
    }
    VkImageFormatProperties fp;
    VkResult r;
    for (;;) {
        r = vkGetPhysicalDeviceImageFormatProperties(gpu_vk.phys, fmt, sh->type, VK_IMAGE_TILING_OPTIMAL, usage, flags, &fp);
        if (r == VK_SUCCESS) break;
        if (raw_storage) {   // raw views without storage usage
            usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_STORAGE_BIT, raw_storage = 0;
            continue;
        }
        if (flags & RAW_VIEW_FLAGS) {   // a device that takes no raw views of this image still takes the image
            flags &= ~(VkImageCreateFlags)RAW_VIEW_FLAGS;
            continue;
        }
        if (arena) break;
        if (usage & VK_IMAGE_USAGE_STORAGE_BIT) usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_STORAGE_BIT;
        else if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) usage &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        else break;
    }
    if (r != VK_SUCCESS || fp.maxMipLevels < sh->levels || fp.maxArrayLayers < sh->layers ||
        fp.maxExtent.width < sh->w || fp.maxExtent.height < sh->h || fp.maxExtent.depth < sh->d) {
        g_st.skipped_format++;
        log_skip(t->d.format, "the device", "it takes no image of this shape and format");
        return NULL;
    }
    VkSampleCountFlagBits samples = (VkSampleCountFlagBits)sh->samples;
    if (!(fp.sampleCounts & samples)) {
        g_st.skipped_format++;
        log_skip(t->d.format, "the device", "it takes no image of this sample count");
        return NULL;
    }
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = sh->type,
                            .format = fmt, .extent = {sh->w, sh->h, sh->d}, .mipLevels = sh->levels,
                            .arrayLayers = sh->layers, .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage image;
    GPU_VK_OK(vkCreateImage(gpu_vk.device, &ci, NULL, &image), "vkCreateImage");
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(gpu_vk.device, image, &mr);
    nvn_texlayout L;
    nvn_tex_layout(&t->d, &L);

    // An image of the arena is bound into an extent of it, which the texture arena takes from its blocks (and stops by name
    // when it has none to give). Any other is placed at the pool offset: bound into the extent (the extent is made here
    // when no bind made it), where the game's storage lies, when the offset has the alignment the driver asks, the memory
    // type fits and the image takes no more than the game's storage (it would overwrite the neighbour otherwise). An
    // image stands in an extent only. An image of decoded ASTC (the row is ASTC, the image RGBA8) never stands in one: the
    // game's storage there holds the compressed blocks, which the image would read as RGBA8 texels, and the decode would
    // overwrite them for the textures aliased over the same bytes (vkf_extent_ok). A texture in a virtual pool mapped over
    // memory the CPU can see has no extent: the backing map keeps that memory as a mirror (a VRAM copy of the CPU's pages) or an import, and an image bound into a
    // mirror's span shares its bytes with the copy: the refresh writes the CPU's pages over the texels, the texels over
    // the copy, and the span, once the mirror is replaced by a larger one, goes to the next mirror that takes it. So the
    // image gets memory of its own, and no mirror is made for it.
    BackingRef ref = {0};
    TexExtent ext = {0};
    int placed = 0, extent_pool = 1;
    if (arena) {
        texarena_alloc(mr.size, mr.alignment, mr.memoryTypeBits, &ext);
        GPU_VK_OK(vkBindImageMemory(gpu_vk.device, image, (VkDeviceMemory)(uintptr_t)ext.block, ext.offset),
                  "vkBindImageMemory (an image of the texture arena)");
    } else if (!vkf_extent_ok(row, fmt)) {
        // memory of its own, below
    } else if ((extent_pool = backing_extent_pool(key->storage, L.size ? L.size : 1))) {
        backing_resolve(key->storage, L.size ? L.size : 1, 0, &ref);
        placed = ref.kind == BACKING_EXTENT && ref.memory && ref.offset % mr.alignment == 0 && mr.size <= L.size &&
                 (mr.memoryTypeBits >> ref.type & 1);
    }
    VkDeviceMemory dedicated = VK_NULL_HANDLE;
    if (arena) {
        // nothing more: its memory is the extent
    } else if (placed) {
        GPU_VK_OK(vkBindImageMemory(gpu_vk.device, image, (VkDeviceMemory)(uintptr_t)ref.memory, ref.offset),
                  "vkBindImageMemory (an image in its pool's extent)");
    } else {
        uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, 0, 0);
        VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                   .memoryTypeIndex = type};
        GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &dedicated), "vkAllocateMemory (an unaliased image)");
        GPU_VK_OK(vkBindImageMemory(gpu_vk.device, image, dedicated, 0), "vkBindImageMemory (an unaliased image)");
        g_st.unaliased++;
        if (!vkf_extent_ok(row, fmt))
            rt_log("gfx: vulkan: the image of the ASTC texture at %#llx holds its decoded RGBA8 texels, not the game's blocks, so "
                   "it does not stand in its pool's extent: it has memory of its own and loses the game's aliasing",
                   (ull)key->storage);
        else if (!extent_pool)
            rt_log("gfx: vulkan: the image of the texture at %#llx stands over memory the CPU can see (a virtual pool mapped "
                   "over a CPU-visible one), which has no extent: it has memory of its own and loses the game's aliasing",
                   (ull)key->storage);
        else
            rt_log("gfx: vulkan: the image of the texture at %#llx cannot stand in its pool's extent (offset %#llx, driver "
                   "alignment %llu, driver size %llu against the game's %llu, memory types %#x against the extent's %u): it "
                   "has memory of its own and loses the game's aliasing", (ull)key->storage, (ull)ref.offset,
                   (ull)mr.alignment, (ull)mr.size, (ull)L.size, mr.memoryTypeBits, ref.type);
    }
    VkbImage *i = calloc(1, sizeof *i);
    if (!i) rt_fatal("gfx: vulkan: out of memory for an image");
    i->key = *key, i->image = image, i->dedicated = dedicated, i->format = fmt, i->flags = flags, i->usage = usage;
    i->aspect = (row->flags & VKF_DEPTH) ? VK_IMAGE_ASPECT_DEPTH_BIT : (row->flags & VKF_STENCIL) ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                                                              : VK_IMAGE_ASPECT_COLOR_BIT;
    i->all_aspects = i->aspect | ((row->flags & (VKF_DEPTH | VKF_STENCIL)) == (VKF_DEPTH | VKF_STENCIL) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    i->size = L.size, i->placed = placed;
    i->decoded = decoded;
    if (arena) i->arena = 1, i->ext = ext, i->desc = t->d;
    g_st.made++, g_st.live++;
    if (placed) g_st.placed++;
    if (arena) g_st.arena++;
    return i;
}

// The image standing on texture t's storage. demand: the replay binds the texture now, so a texture the CPU writes gets
// its arena image (the formats this layer cannot upload stop here by name); without it such a texture gets none and its
// entry shows the placeholder until a submission binds it.
static VkbImage *image_get(const tex_t *t, int demand) {
    const nvn_texdesc *d = &t->d;
    const VkfRow *row = vkf_row(d->format);
    uint64_t storage = (uint64_t)(uintptr_t)t->mem;
    int cpu_written = !gfx_gpu_only(storage);
    if (demand && cpu_written && row) {
        const char *what = (row->flags & (VKF_DEPTH | VKF_STENCIL)) ? "a depth or stencil texture"
                           : (row->flags & VKF_EXPAND_RGB) ? "a three-component texture, which an upload would expand to four" : NULL;
        if (what)
            rt_fatal("gfx: vulkan: the texture at %#llx (target %d, NVN format %#x, %ux%u) is written by the CPU and is %s; the "
                     "arena's upload does not serve it yet", (ull)storage, d->target, d->format, d->width, d->height, what);
    }
    uint8_t swz[4];
    const char *why;
    VkFormat vk = vkf_pick(row, supported, NULL, swz, &why);
    if (vk == VK_FORMAT_UNDEFINED || (row->flags & VKF_EXPAND_RGB)) {
        plat_mutex_lock(&g_mu);
        g_st.skipped_format++;
        log_skip(d->format, "the format table",
                 row && (row->flags & VKF_EXPAND_RGB) ? "three components need the upload's expansion" : why);
        plat_mutex_unlock(&g_mu);
        return NULL;
    }
    if (cpu_written && !demand) {
        plat_mutex_lock(&g_mu);
        g_st.skipped_cpu++;
        plat_mutex_unlock(&g_mu);
        return NULL;
    }
    Shape sh;
    shape_of(d, &sh);
    ImgKey key = {.storage = storage, .type = (uint32_t)sh.type, .w = sh.w, .h = sh.h, .d = sh.d, .layers = sh.layers,
                  .levels = sh.levels, .samples = sh.samples, .cls = (uint32_t)vkf_class(vkf_linear_twin(vk)),
                  .linear = nvn_tex_is_linear(d)};
    if (cpu_written) {
        key.pkg = (uint32_t)(d->pkg_block_h & 0xff);
        key.layout_flags = (uint32_t)d->flags & (NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT);
        key.stride = d->stride;
    }
    plat_mutex_lock(&g_mu);
    VkbImage *i = NULL;
    for (VkbImage *c = g_hash[bucket(storage)]; c; c = c->next)
        if (key_equal(&c->key, &key)) {
            i = c;
            break;
        }
    if (!i) {
        i = image_make(t, row, vk, &key, &sh, cpu_written);
        if (i) {
            i->next = g_hash[bucket(storage)];
            g_hash[bucket(storage)] = i;
            if (!i->arena) {   // an arena image is brought to its layout by its first upload
                i->pending = 1;
                i->next_aux = g_pending;
                g_pending = i;
            }
        }
    }
    plat_mutex_unlock(&g_mu);
    return i;
}

VkbImage *vkb_image_get(const tex_t *t) { return image_get(t, 0); }
VkbImage *vkb_image_demand(const tex_t *t) { return image_get(t, 1); }

VkImage vkb_image_handle(const VkbImage *i) { return i->image; }
VkFormat vkb_image_format(const VkbImage *i) { return i->format; }
int vkb_image_placed(const VkbImage *i) { return i->placed; }

void vkb_image_shape(const VkbImage *i, VkbShape *out) {
    *out = (VkbShape){(VkImageType)i->key.type, i->key.w, i->key.h, i->key.d, i->key.layers, i->key.levels,
                      i->key.samples, i->aspect, i->all_aspects, i->format};
}

void vkb_image_use(VkbImage *i, uint64_t value) {
    uint64_t seen = __atomic_load_n(&i->last_use, __ATOMIC_RELAXED);
    while (seen < value && !__atomic_compare_exchange_n(&i->last_use, &seen, value, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    if (i->placed) {
        BackingRef r;
        backing_resolve(i->key.storage, 1, value, &r);
    }
}

static void image_destroy(VkbImage *i) {
    vkDestroyImage(gpu_vk.device, i->image, NULL);
    if (i->dedicated) vkFreeMemory(gpu_vk.device, i->dedicated, NULL);
    if (i->arena) texarena_free(&i->ext);
    g_st.destroyed++, g_st.live--;
    free(i);
}

static void unlink_hash(VkbImage *i) {
    for (VkbImage **p = &g_hash[bucket(i->key.storage)]; *p; p = &(*p)->next)
        if (*p == i) {
            *p = i->next;
            return;
        }
}

static void doom(VkbImage *i) {
    if (i->doomed) return;
    unlink_hash(i);
    i->doomed = 1;
    i->next_doom = g_doomed;
    g_doomed = i;
}

void vkb_image_pool_gone(uint64_t lo, uint64_t hi) {
    plat_mutex_lock(&g_mu);
    for (uint32_t b = 0; b < BUCKETS; b++) {
        VkbImage *c = g_hash[b], *next;
        for (; c; c = next) {
            next = c->next;
            if (c->key.storage >= lo && c->key.storage < hi) doom(c);
        }
    }
    plat_mutex_unlock(&g_mu);
}

void vkb_image_release(const tex_t *t) {
    plat_mutex_lock(&g_mu);
    for (VkbImage *c = g_hash[bucket((uint64_t)(uintptr_t)t->mem)]; c; c = c->next)
        if (c->key.storage == (uint64_t)(uintptr_t)t->mem && !c->refs && !c->placed) {
            doom(c);
            break;
        }
    plat_mutex_unlock(&g_mu);
}

void vkb_image_reap(void) {
    plat_mutex_lock(&g_mu);
    uint64_t d = done();
    VkbImage **p = &g_doomed;
    while (*p) {
        VkbImage *i = *p;
        if (i->refs || __atomic_load_n(&i->last_use, __ATOMIC_RELAXED) > d) {
            p = &i->next_doom;
            continue;
        }
        *p = i->next_doom;
        if (i->pending) {
            for (VkbImage **q = &g_pending; *q; q = &(*q)->next_aux)
                if (*q == i) {
                    *q = i->next_aux;
                    break;
                }
        }
        image_destroy(i);
    }
    plat_mutex_unlock(&g_mu);
}

void vkb_image_close(void) {
    plat_mutex_lock(&g_mu);
    for (uint32_t b = 0; b < BUCKETS; b++) {
        VkbImage *c = g_hash[b], *next;
        for (; c; c = next) {
            next = c->next;
            image_destroy(c);
        }
        g_hash[b] = NULL;
    }
    for (VkbImage *i = g_doomed, *next; i; i = next) {
        next = i->next_doom;
        image_destroy(i);
    }
    g_doomed = g_pending = NULL;
    if (g_stage) {
        vkDestroyBuffer(gpu_vk.device, g_stage, NULL);
        vkFreeMemory(gpu_vk.device, g_stage_mem, NULL);
        g_stage = VK_NULL_HANDLE, g_stage_mem = VK_NULL_HANDLE, g_stage_map = NULL, g_stage_size = 0;
    }
    memset(g_logged_formats, 0, sizeof g_logged_formats);
    memset(&g_st, 0, sizeof g_st);
    plat_mutex_unlock(&g_mu);
}

void vkb_image_flush(VkCommandBuffer cmd) {
    plat_mutex_lock(&g_mu);
    VkImageMemoryBarrier2 b[64];
    uint32_t n = 0;
    for (VkbImage *i = g_pending, *next; i; i = next) {
        next = i->next_aux;
        i->pending = 0;
        b[n++] = (VkImageMemoryBarrier2){
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = 0, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = i->image,
            .subresourceRange = {i->all_aspects, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
        g_st.transitions++;
        if (n == 64) {
            VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n, .pImageMemoryBarriers = b};
            vkCmdPipelineBarrier2(cmd, &dep);
            n = 0;
        }
    }
    g_pending = NULL;
    if (n) {
        VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = n, .pImageMemoryBarriers = b};
        vkCmdPipelineBarrier2(cmd, &dep);
    }
    plat_mutex_unlock(&g_mu);
}

void vkb_image_stamp_pools(uint64_t value) {
    plat_mutex_lock(&g_mu);
    uint64_t seen[64];
    int ns = 0;
    // Only images that can still be reached: a doomed image's pool was finalized, and pool_fini waited for every value
    // stamped on it, so nothing in flight reads that image any more.
    for (uint32_t b = 0; b < BUCKETS; b++) {
        for (VkbImage *i = g_hash[b]; i; i = i->next) {
            if (!i->placed) continue;
            const pool_t *p = nvn_pool_find(i->key.storage, 1);
            uint64_t at = p ? (uint64_t)(uintptr_t)p->mem : i->key.storage;
            int dup = 0;
            for (int k = 0; k < ns; k++) dup |= seen[k] == at;
            if (dup) continue;
            if (ns < 64) seen[ns++] = at;
            BackingRef r;
            backing_resolve(at, 1, value, &r);
        }
    }
    plat_mutex_unlock(&g_mu);
}

// ---------- views

static int view_ok(const VkbImage *i, int type) {
    switch (i->key.type) {
    case VK_IMAGE_TYPE_1D: return type == BV_1D || type == BV_1D_ARRAY;
    case VK_IMAGE_TYPE_2D:
        return type == BV_2D || type == BV_2D_ARRAY ||
               ((type == BV_CUBE || type == BV_CUBE_ARRAY) && (i->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT));
    default: return type == BV_3D;
    }
}

static VkComponentSwizzle selector(int s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

static int comp_index(VkComponentSwizzle c) {
    return c == VK_COMPONENT_SWIZZLE_R ? 0 : c == VK_COMPONENT_SWIZZLE_G ? 1 : c == VK_COMPONENT_SWIZZLE_B ? 2 : 3;
}

// The view's components: the texture's swizzle, the view's on top of it, then the format's own (NVN names a packed
// format's components from the least significant bit; an X format samples alpha as 1). swz is the row's selectors.
static VkComponentMapping components(const tex_t *t, const view_t *v, const VkfRow *row, const uint8_t swz[4]) {
    VkComponentSwizzle tex[4] = {selector(t->d.swizzle_r), selector(t->d.swizzle_g), selector(t->d.swizzle_b),
                                 selector(t->d.swizzle_a)};
    VkComponentSwizzle out[4];
    for (int k = 0; k < 4; k++) {
        VkComponentSwizzle s = tex[k];
        if (v && v->set_swizzle) {
            int sel = v->swz[k];
            if (sel >= NVN_TEXTURE_SWIZZLE_R && sel <= NVN_TEXTURE_SWIZZLE_A) s = tex[sel - NVN_TEXTURE_SWIZZLE_R];
            else s = selector(sel);
        }
        out[k] = s;
    }
    // Each output component is NVN's component s (R..A); with a packed stand-in NVN's component j sits in the format's
    // component swz[j] (the row's selectors: SWIZZLE_R.. or 0 for the identity).
    VkComponentSwizzle res[4];
    for (int k = 0; k < 4; k++) {
        VkComponentSwizzle s = out[k];
        if (s == VK_COMPONENT_SWIZZLE_ZERO || s == VK_COMPONENT_SWIZZLE_ONE) {
            res[k] = s;
            continue;
        }
        int j = comp_index(s);
        if ((row->flags & VKF_ALPHA_ONE) && j == 3) {
            res[k] = VK_COMPONENT_SWIZZLE_ONE;
            continue;
        }
        res[k] = swz[j] ? selector(swz[j]) : s;
    }
    return (VkComponentMapping){res[0], res[1], res[2], res[3]};
}

static void *view_make(const tex_t *t, const view_t *v, int type, int demand) {
    VkbImage *img = image_get(t, demand);
    if (!img || !view_ok(img, type)) return NULL;
    int nvn_format = v && v->set_format ? (int)v->format : t->d.format;
    const VkfRow *row = vkf_row(nvn_format);
    uint8_t swz[4];
    const char *why;
    VkFormat vf = vkf_pick(row, supported, NULL, swz, &why);
    if (vf == VK_FORMAT_UNDEFINED) return NULL;
    if (vkf_class(vf) != vkf_class(img->format) && vf != img->format)
        rt_fatal("gfx: vulkan: a view of the texture at %#llx reads NVN format %#x, which is not in the format class of its "
                 "image (Vulkan format %d): the game aliases formats the table cannot view through one image",
                 (ull)img->key.storage, nvn_format, (int)img->format);
    int dsmode = v && v->set_dsmode ? v->dsmode : t->d.dsmode;
    VkImageAspectFlags aspect = aspect_of(row, dsmode);
    uint32_t base_level = v && v->set_levels ? v->base_level : 0;
    uint32_t levels = v && v->set_levels && v->nlevels ? v->nlevels : img->key.levels - base_level;
    uint32_t base_layer = v && v->set_layers ? v->min_layer : 0;
    uint32_t layers = v && v->set_layers && v->nlayers ? v->nlayers : img->key.layers - base_layer;
    if (type == BV_1D || type == BV_2D || type == BV_3D) layers = 1;
    else if (type == BV_CUBE) layers = 6;
    if (img->key.type == VK_IMAGE_TYPE_3D) base_layer = 0, layers = 1;
    if (base_level + levels > img->key.levels || base_layer + layers > img->key.layers)
        rt_fatal("gfx: vulkan: a view of the texture at %#llx asks levels %u..%u and layers %u..%u of an image of %u levels "
                 "and %u layers", (ull)img->key.storage, base_level, base_level + levels, base_layer, base_layer + layers,
                 img->key.levels, img->key.layers);
    VkImageViewUsageCreateInfo uc = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                     .usage = VK_IMAGE_USAGE_SAMPLED_BIT};
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &uc, .image = img->image,
                                .viewType = (VkImageViewType)type, .format = vf,
                                .components = components(t, v, row, swz),
                                .subresourceRange = {aspect, base_level, levels, base_layer, layers}};
    VkImageView view;
    GPU_VK_OK(vkCreateImageView(gpu_vk.device, &ci, NULL, &view), "vkCreateImageView (a sampled view)");
    Vw *w = calloc(1, sizeof *w);
    if (!w) rt_fatal("gfx: vulkan: out of memory for a view");
    w->view = view, w->img = img;
    plat_mutex_lock(&g_mu);
    img->refs++;
    g_st.views_made++, g_st.views_live++;
    plat_mutex_unlock(&g_mu);
    return w;
}

void *vkb_view_make(const tex_t *t, const view_t *v, int type) { return view_make(t, v, type, 0); }

VkImageView vkb_storage_view(const tex_t *t, const view_t *v, int type, uint64_t value) {
    VkbImage *img = vkb_image_get(t);
    if (!img) img = vkb_image_demand_current(t, value);
    int nvn_format = v && v->set_format ? (int)v->format : t->d.format;
    if (!img)
        rt_fatal("gfx: vulkan: a storage image over the texture at %#llx (NVN format %#x) has no image on this device",
                 (ull)(uintptr_t)t->mem, nvn_format);
    const char *refused = vkf_decoded_stop(vkf_row(t->d.format), img->format, VKF_USE_STORAGE_VIEW);
    if (refused)
        rt_fatal("gfx: vulkan: %s over the texture at %#llx (NVN format %#x) reaches an image of ASTC blocks decoded to RGBA8: "
                 "a program would store texels where it expects the game's blocks, which this layer does not serve yet",
                 refused, (ull)img->key.storage, nvn_format);
    if (!(img->usage & VK_IMAGE_USAGE_STORAGE_BIT) || !view_ok(img, type))
        rt_fatal("gfx: vulkan: a storage image over the texture at %#llx (NVN format %#x, view type %d): its image was made "
                 "without storage usage or cannot be seen as that type", (ull)img->key.storage, nvn_format, type);
    const VkfRow *row = vkf_row(nvn_format);
    uint8_t swz[4];
    const char *why;
    VkFormat vf = vkf_pick(row, supported, NULL, swz, &why);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(gpu_vk.phys, vf, &fp);
    if (vf == VK_FORMAT_UNDEFINED || !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) ||
        (vkf_class(vf) != vkf_class(img->format) && vf != img->format))
        rt_fatal("gfx: vulkan: a storage image over the texture at %#llx reads NVN format %#x, which this device does not "
                 "store to through its image (Vulkan format %d)", (ull)img->key.storage, nvn_format, (int)vf);
    uint32_t level = v && v->set_levels ? v->base_level : 0;
    uint32_t base_layer = v && v->set_layers ? v->min_layer : 0;
    uint32_t layers = v && v->set_layers && v->nlayers ? v->nlayers : img->key.layers - base_layer;
    if (type == BV_1D || type == BV_2D || type == BV_3D) layers = 1;
    else if (type == BV_CUBE) layers = 6;
    if (img->key.type == VK_IMAGE_TYPE_3D) base_layer = 0, layers = 1;
    if (level >= img->key.levels || base_layer + layers > img->key.layers)
        rt_fatal("gfx: vulkan: a storage image over the texture at %#llx asks level %u and layers %u..%u of an image of %u "
                 "levels and %u layers", (ull)img->key.storage, level, base_layer, base_layer + layers, img->key.levels,
                 img->key.layers);
    VkImageViewUsageCreateInfo uc = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                     .usage = VK_IMAGE_USAGE_STORAGE_BIT};
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = &uc, .image = img->image,
                                .viewType = (VkImageViewType)type, .format = vf,
                                .subresourceRange = {img->aspect, level, 1, base_layer, layers}};
    VkImageView view;
    GPU_VK_OK(vkCreateImageView(gpu_vk.device, &ci, NULL, &view), "vkCreateImageView (a storage view)");
    // Not counted (no refs, no g_mu): the use keeps the image until the submission is done, and the view is destroyed by
    // the completion thread at that point, which must not wait for this layer's lock (oot2/vk_image.h).
    vkb_image_use(img, value);
    return view;
}
void *vkb_view_demand(const tex_t *t, const view_t *v, int type) { return view_make(t, v, type, 1); }

// A submission that is being recorded reads this view: the image of a texture the CPU writes is made current for it
// (queried and uploaded again if the CPU rewrote it, in the submission's prologue), once a submission however many slots
// of it bind the image.
static void make_current(VkbImage *img, uint64_t value) {
    if (!img->arena) return;
    vkb_image_use(img, value);
    uint64_t seen = __atomic_load_n(&img->up.current_for, __ATOMIC_ACQUIRE);
    if (seen == value) return;
    __atomic_store_n(&img->up.current_for, value, __ATOMIC_RELEASE);
    vkt_upload(&img->desc, img->key.storage, img->image, img->aspect, &img->up, value, img->decoded);
}

void vkb_view_use(void *view, uint64_t value) { make_current(((Vw *)view)->img, value); }

VkbImage *vkb_image_demand_current(const tex_t *t, uint64_t value) {
    VkbImage *img = vkb_image_demand(t);
    if (img) make_current(img, value);
    return img;
}

VkImageView vkb_view_handle(const void *view) { return ((const Vw *)view)->view; }

void vkb_view_free(void *view) {
    Vw *w = view;
    vkDestroyImageView(gpu_vk.device, w->view, NULL);
    plat_mutex_lock(&g_mu);
    w->img->refs--;
    g_st.views_live--;
    // An arena image no view holds goes (its memory is the arena's, 3.4 GiB at most, and an extent nothing can reach would
    // stay until the pool ends): the texture registered again makes it again, and uploads it again. It is destroyed once the
    // newest submission that read it is done (vkb_image_reap).
    if (w->img->arena && !w->img->refs) doom(w->img);
    plat_mutex_unlock(&g_mu);
    free(w);
}

// ---------- texel copies


static void stage_grow(VkDeviceSize need) {
    if (g_stage_size >= need) return;
    if (g_stage) {
        vkDestroyBuffer(gpu_vk.device, g_stage, NULL);
        vkFreeMemory(gpu_vk.device, g_stage_mem, NULL);
    }
    VkDeviceSize size = need < (1u << 20) ? (1u << 20) : need;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &g_stage), "vkCreateBuffer (texel staging)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, g_stage, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no host-visible coherent memory type for the texel staging buffer");
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &g_stage_mem), "vkAllocateMemory (texel staging)");
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, g_stage, g_stage_mem, 0), "vkBindBufferMemory (texel staging)");
    GPU_VK_OK(vkMapMemory(gpu_vk.device, g_stage_mem, 0, VK_WHOLE_SIZE, 0, &g_stage_map), "vkMapMemory (texel staging)");
    g_stage_size = size;
}

typedef struct Xfer {
    VkbImage *img;
    VkBufferImageCopy copy;
    uint64_t bytes;
} Xfer;

static Xfer xfer_of(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img_pitch) {
    VkbImage *img = vkb_image_get(t);
    if (!img)
        rt_fatal("gfx: vulkan: a texel copy over the texture at %p (NVN format %#x, target %d): this layer makes no image "
                 "for it", (const void *)t->mem, t->d.format, t->d.target);
    const VkfRow *fr = vkf_row(t->d.format);
    const char *refused = vkf_decoded_stop(fr, img->format, VKF_USE_TEXEL_COPY);
    if (refused)
        rt_fatal("gfx: vulkan: %s over the texture at %p (NVN format %#x, target %d) reaches an image of ASTC blocks decoded "
                 "to RGBA8, which holds texels where the copy addresses the game's blocks: this layer does not serve it yet",
                 refused, (const void *)t->mem, t->d.format, t->d.target);
    uint64_t bpb = fr->bpb;
    if (row % bpb)
        rt_fatal("gfx: vulkan: a texel copy with a row pitch of %llu bytes over blocks of %llu", (ull)row, (ull)bpb);
    uint32_t level = v && v->set_levels ? v->base_level : 0, layer = v && v->set_layers ? v->min_layer : 0;
    int layered = img->key.layers > 1 || t->d.target == NVN_TEXTURE_TARGET_1D_ARRAY ||
                  t->d.target == NVN_TEXTURE_TARGET_2D_ARRAY;
    uint32_t depth = r->depth > 0 ? (uint32_t)r->depth : 1, rows = r->height > 0 ? (uint32_t)r->height : 1;
    Xfer x = {.img = img};
    x.copy = (VkBufferImageCopy){
        .bufferOffset = 0, .bufferRowLength = row ? (uint32_t)(row / bpb * fr->bw) : 0,
        .bufferImageHeight = row && img_pitch ? (uint32_t)(img_pitch / row * fr->bh) : 0,
        .imageSubresource = {img->aspect, level, layered ? layer + (uint32_t)r->zoffset : layer, layered ? depth : 1},
        .imageOffset = {r->xoffset, r->yoffset, img->key.type == VK_IMAGE_TYPE_3D ? r->zoffset : 0},
        .imageExtent = {(uint32_t)r->width, rows, img->key.type == VK_IMAGE_TYPE_3D ? depth : 1}};
    uint64_t block_rows = (rows + fr->bh - 1) / fr->bh;
    x.bytes = img_pitch ? img_pitch * (depth - 1) + row * block_rows : row * block_rows * depth;
    return x;
}

static void xfer_run(const Xfer *x, int to_image) {
    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cmd = gpu_vk_cmd();
    vkb_image_flush(cmd);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cmd, &dep);
    if (to_image) vkCmdCopyBufferToImage(cmd, g_stage, x->img->image, VK_IMAGE_LAYOUT_GENERAL, 1, &x->copy);
    else vkCmdCopyImageToBuffer(cmd, x->img->image, VK_IMAGE_LAYOUT_GENERAL, g_stage, 1, &x->copy);
    VkMemoryBarrier2 after = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                              .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                              .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_READ_BIT};
    dep.pMemoryBarriers = &after;
    vkCmdPipelineBarrier2(cmd, &dep);
    vkb_image_use(x->img, value);
    gpu_vk_end(NULL, 0);
    gpu_sync_wait(value, UINT64_MAX);
}

// A texture whose storage the CPU can see keeps its texels in guest memory, which the arena's upload reads at the
// next bind (the page generations show the write): the texel calls read and write that storage in its own layout, block
// by block (a block is at most 16 bytes, so it never straddles a 16-byte piece of a GOB). 1 when the texture is one of
// those; 0 leaves the copy to the image. A shape the layout cannot place stops by name.
static int texels_guest(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *buf, uint64_t row,
                        uint64_t img, int write) {
    uint64_t storage = (uint64_t)(uintptr_t)t->mem;
    if (gfx_gpu_only(storage)) return 0;
    const nvn_texdesc *d = &t->d;
    uint32_t level = v && v->set_levels ? v->base_level : 0, layer = v && v->set_layers ? v->min_layer : 0;
    int layered = d->target == NVN_TEXTURE_TARGET_1D_ARRAY || d->target == NVN_TEXTURE_TARGET_2D_ARRAY ||
                  d->target == NVN_TEXTURE_TARGET_CUBEMAP || d->target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY;
    uint32_t depth = r->depth > 0 ? (uint32_t)r->depth : 1, rows = r->height > 0 ? (uint32_t)r->height : 1;
    int bw, bh;
    uint32_t bpb = (uint32_t)nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb || r->xoffset % bw || r->yoffset % bh)
        rt_fatal("gfx: vulkan: a texel copy over the texture at %#llx (NVN format %#x) at %d,%d, not on its %dx%d blocks",
                 (ull)storage, d->format, r->xoffset, r->yoffset, bw, bh);
    uint32_t bx0 = (uint32_t)r->xoffset / (uint32_t)bw, by0 = (uint32_t)r->yoffset / (uint32_t)bh;
    uint32_t nbx = ((uint32_t)r->width + (uint32_t)bw - 1) / (uint32_t)bw, nby = (rows + (uint32_t)bh - 1) / (uint32_t)bh;
    for (uint32_t k = 0; k < depth; k++) {
        uint8_t *plane = buf + (uint64_t)k * img;
        if (nvn_tex_is_linear(d)) {   // rows as they are, d->stride apart: one level, one layer
            if (level || layer || k || r->zoffset)
                rt_fatal("gfx: vulkan: a texel copy over the linear texture at %#llx past its one level and layer",
                         (ull)storage);
            for (uint32_t y = 0; y < nby; y++) {
                uint8_t *g = (uint8_t *)(uintptr_t)(storage + (uint64_t)(by0 + y) * vkt_linear_stride(d) +
                                                    (uint64_t)bx0 * bpb);
                if (write) memcpy(g, plane + y * row, (size_t)nbx * bpb);
                else memcpy(plane + y * row, g, (size_t)nbx * bpb);
            }
            continue;
        }
        UntileGeom ug;
        const char *why;
        uint32_t lay = layered ? layer + (uint32_t)r->zoffset + k : layer;
        uint32_t z = d->target == NVN_TEXTURE_TARGET_3D ? (uint32_t)r->zoffset + k : 0;
        if (!untile_geom(d, level, lay, &ug, &why))
            rt_fatal("gfx: vulkan: a texel copy over the texture at %#llx (NVN format %#x, target %d, level %u, layer %u): "
                     "%s", (ull)storage, d->format, d->target, level, lay, why);
        if (bx0 + nbx > (ug.row_bytes / ug.bpb) || by0 + nby > ug.rows || z >= ug.slices)
            rt_fatal("gfx: vulkan: a texel copy region past level %u of the texture at %#llx", level, (ull)storage);
        for (uint32_t y = 0; y < nby; y++)
            for (uint32_t x = 0; x < nbx; x++) {
                uint8_t *g = (uint8_t *)(uintptr_t)(storage + ug.base +
                                                    untile_offset(&ug, UNTILE_GOB_ORDER, (bx0 + x) * bpb, by0 + y, z));
                uint8_t *h = plane + y * row + (uint64_t)x * bpb;
                if (write) memcpy(g, h, bpb);
                else memcpy(h, g, bpb);
            }
    }
    plat_mutex_lock(&g_mu);
    if (write) g_st.texel_writes++;
    else g_st.texel_reads++;
    plat_mutex_unlock(&g_mu);
    return 1;
}

void vkb_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img) {
    if (texels_guest(t, v, r, dst, row, img, 0)) return;
    plat_mutex_lock(&g_xfer_mu);
    Xfer x = xfer_of(t, v, r, row, img);
    stage_grow(x.bytes);
    xfer_run(&x, 0);
    memcpy(dst, g_stage_map, (size_t)x.bytes);
    plat_mutex_lock(&g_mu);
    g_st.texel_reads++;
    plat_mutex_unlock(&g_mu);
    plat_mutex_unlock(&g_xfer_mu);
}

void vkb_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row,
                      uint64_t img) {
    if (texels_guest(t, v, r, (uint8_t *)src, row, img, 1)) return;
    plat_mutex_lock(&g_xfer_mu);
    Xfer x = xfer_of(t, v, r, row, img);
    stage_grow(x.bytes);
    memcpy(g_stage_map, src, (size_t)x.bytes);
    xfer_run(&x, 1);
    plat_mutex_lock(&g_mu);
    g_st.texel_writes++;
    plat_mutex_unlock(&g_mu);
    plat_mutex_unlock(&g_xfer_mu);
}

void vkb_image_stats(VkbImageStats *out) {
    plat_mutex_lock(&g_mu);
    *out = g_st;
    plat_mutex_unlock(&g_mu);
}
