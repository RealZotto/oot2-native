// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Images of the Vulkan back end (architecture D19, without the uploads and the transcode): what a texture the GPU writes
// stands on, the sampled views of it that the bindless tables hold, and the texel copies of linear data.
//
// One image per (storage, description). A texture whose storage lies in a pool the CPU cannot see (a render target or a
// GPU-written texture) gets an image bound at its pool offset into the backing map's VRAM extent (oot2/backing.h, the
// extent is made when the first image over the pool asks for it), so that textures the game aliases over one storage see
// each other's writes: images of the same shape are one image whose views may differ in format inside the format's
// compatibility class (a texture rendered as RGBA8 and sampled as RGBA8_SRGB, or written as R32_UINT and read as RGBA8,
// is one image: MUTABLE_FORMAT, the image in the linear twin of an sRGB format), and images of different shapes over one
// range alias each other (VK_IMAGE_CREATE_ALIAS_BIT; what one reads of the other's writes is the specification's
// undefined content unless the parameters match). Every image lives in VK_IMAGE_LAYOUT_GENERAL. An image whose placement
// is impossible (the storage is not at the alignment the driver asks, or the driver wants more bytes than the game's
// storage holds, which would overwrite a neighbour, or the pool is a virtual one mapped over memory the CPU can see, which
// the backing map keeps as a mirror or an import and never as an extent: an image bound into a mirror's span would share
// its bytes with the copy of the CPU's pages, backing_extent_pool) gets memory of its own and is counted as unaliased.
// An image of ASTC decoded to RGBA8 (the device does not sample ASTC) is never placed either: the game's storage holds
// blocks, not texels. It gets memory of its own, and a storage view or a texel copy of it stops by name (vkf_decoded,
// oot2/vk_format.h).
// Storage the CPU can see (textures the CPU writes) gets no image at registration: its image is made in the texture
// arena when the replay binds the texture (vkb_view_demand), and the entry shows the table's placeholder until then. A
// format with no image (the formats of oot2/vk_format.h that need an upload expansion or a transcode, or that the device
// lacks) is counted and logged once per format.
//
// Lifetime. A view holds its image; an image is destroyed when no view holds it and the newest submission that used it
// is done, after its pool was finalized (vkb_image_pool_gone) or its texture was released with no view on it. Submissions
// stamp the pools that hold images (vkb_image_stamp_pools), so that pool_fini waits for what could read them.
#pragma once
#include <stdint.h>

#include "volk.h"

#include "oot2/bindless.h"
#include "oot2/vk_format.h"

typedef struct VkbImage VkbImage;

// The image standing on texture t's storage, made when none is, or NULL when this layer makes none for it (see above).
VkbImage *vkb_image_get(const tex_t *t);
// Likewise for a texture in storage the CPU can see: its arena image, made when none is (the checks read it back).
VkbImage *vkb_image_demand(const tex_t *t);
VkImage vkb_image_handle(const VkbImage *i);
VkFormat vkb_image_format(const VkbImage *i);
int vkb_image_placed(const VkbImage *i);      // 1 when it is bound into its pool's extent
// The image's shape: its type and extent (a 1D array's layers and a cube's six faces are layers), the aspect a copy takes
// and every aspect of the format (a clear takes them all), and the format (the linear twin of an sRGB one).
typedef struct VkbShape {
    VkImageType type;
    uint32_t width, height, depth, layers, levels, samples;
    VkImageAspectFlags aspect, all_aspects;
    VkFormat format;
    VkImageCreateFlags flags;   // what the image was created with
    VkImageUsageFlags usage;
} VkbShape;
void vkb_image_shape(const VkbImage *i, VkbShape *out);
// The submission that will signal `value` uses the image: its pool and the image itself wait for it.
void vkb_image_use(VkbImage *i, uint64_t value);

// The sampled view of texture t seen through v as view type `type` (BindlessOps.view_make): a VkImageView wrapped with
// its image; NULL when there is none. Formats, swizzles, aspect and ranges follow the texture's description composed
// with the view's.
void *vkb_view_make(const tex_t *t, const view_t *v, int type);
// Likewise, for a texture the replay binds now (BindlessOps.view_demand): a texture the CPU writes gets its arena image,
// whose memory is taken now; vkb_view_make makes none for it, so that registering thousands of textures takes no memory.
void *vkb_view_demand(const tex_t *t, const view_t *v, int type);
// A view of t for stores (a storage image of a program), of view type `type` at the view's base level, for the
// submission `value`: its image (an arena image made current as vkb_image_demand_current makes it) used by `value`.
// The caller destroys the view (vkDestroyImageView, as an attachment view is) once that submission is done: the use keeps
// the image until then, no count does, so the completion thread can destroy it without this layer's lock, which a thread
// holds while the backing map waits for the timeline, and the timeline waits for the completion thread. A texture whose
// image cannot be stored to stops by name.
VkImageView vkb_storage_view(const tex_t *t, const view_t *v, int type, uint64_t value);
// The submission being recorded, `value`, reads the view (BindlessOps.view_use): an arena image is made current for it.
void vkb_view_use(void *view, uint64_t value);
// The image of a texture for a copy of a submission being recorded (value): vkb_image_get's, else the arena image of a
// texture the CPU writes, made if need be and current for that submission as vkb_view_use makes it (its prologue
// uploads what the CPU wrote, so the copy lands on the storage's content). NULL when the format has no image. A copy
// into an arena image is newer than the guest storage, which keeps the CPU's bytes: the CPU does not see it.
VkbImage *vkb_image_demand_current(const tex_t *t, uint64_t value);
// A pass writes the blocks of this image through a view in the format of the block's size (its texels are blocks): from
// now on the image keeps what the GPU wrote. The CPU's bytes of an arena image upload again when the CPU writes their
// pages (make_current), which would put the guest's older blocks over the pass's; an image so marked takes no more
// uploads. The image as a whole: a CPU write to another level or layer of it is not uploaded either. No effect on a
// placed image (it shares the game's storage, which the pass's write reaches by the backing map).
void vkb_image_gpu_wrote(VkbImage *img);
void vkb_view_free(void *view);
VkImageView vkb_view_handle(const void *view);
void vkb_image_release(const tex_t *t);       // BindlessOps.texture_release: the serial's image may go when nothing holds it

// Record into cmd the barrier that brings every image made since the last call from UNDEFINED to GENERAL.
void vkb_image_flush(VkCommandBuffer cmd);
// Stamp every pool that holds an image with value (a resolve per pool, or per image outside the front end's pools).
void vkb_image_stamp_pools(uint64_t value);
// The pool [lo, hi) is finalized: the images over it can no longer be reached by a new view.
void vkb_image_pool_gone(uint64_t lo, uint64_t hi);
// Destroy the images that can go.
void vkb_image_reap(void);
// Every image (no submission in flight).
void vkb_image_close(void);

// ---- texel copies of linear data (gfx.h read_texels and write_texels): row is the bytes per row of the host data and
// img the bytes per two-dimensional image of the region; a copy goes through a staging buffer and waits for it.
void vkb_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img);
void vkb_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row,
                      uint64_t img);

typedef struct VkbImageStats {
    uint64_t made, live, placed, unaliased, destroyed, arena;   // arena: images made in the texture arena
    uint64_t unaliased_bytes;                  // memory of their own the live unaliased images hold now
    uint64_t skipped_cpu, skipped_format;      // textures with no image: CPU-visible storage; no format
    uint64_t views_made, views_live;
    uint64_t transitions, texel_reads, texel_writes;
    VkfImgTable cls;                           // the live images in the arena or with memory of their own, by class
} VkbImageStats;
void vkb_image_stats(VkbImageStats *out);
int vkb_image_stats_try(VkbImageStats *out);   // 0 when the lock is busy (the counters' lines never wait for it)
