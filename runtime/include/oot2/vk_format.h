// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The format table of the Vulkan back end (architecture D19): every NVNformat of NVN's enumeration (0x01 to 0x9C, the
// range the texture and sampler study sized) and the Vulkan format that stands for it. Images, render targets, vertex attributes and the
// transcode all read this one table; it takes no device, so the check of the table over every format runs without
// one. What depends on the device (a format the device lacks, the stand-in that replaces it) is decided by the
// functions below from a caller's answer to "does the device take this format for these features".
//
// A row holds:
//   vk        the image format. The texel layout of the game's storage, so that an image placed at a pool offset reads
//             what the game wrote; VK_FORMAT_UNDEFINED for a format with no Vulkan equivalent, with `reason`.
//   vertex    the vertex attribute format with the NVN format's own components (three components stay three), or
//             VK_FORMAT_UNDEFINED where NVN gives the format no vertex use.
//   alt       a mandatory format that stands in when the device lacks vk (16-bit packed colour, 24-bit depth), read
//             through alt_swz; vk and alt have the same bytes per block, so an image of either sits at the same offset.
//   swz       the view swizzle that restores NVN's component order when vk names its components the other way round
//             (NVN names a packed format's components from the least significant bit): NVN_TEXTURE_SWIZZLE_R to _A per
//             component, or 0 for the identity.
//   flags     VKF_* below.
//   bw, bh    the block in texels (1 by 1 for uncompressed formats).
//   bpb       the bytes per block of vk's storage (equal to nvn_fmt_bpb for every row an image can stand on; the
//             expanded 3-component rows hold the four-component size and say VKF_EXPAND_RGB).
#pragma once
#include <stdint.h>

#include "volk.h"

enum {
    VKF_EXPAND_RGB = 1,   // 3-component format: Vulkan has no sampled format for it, each texel is expanded to four
                          // components on upload (the texture upload), so no image can stand on the game's own storage
    VKF_ALPHA_ONE = 2,    // X format: alpha is sampled as 1 (a view swizzle)
    VKF_ASTC = 4,         // block-compressed ASTC: the device may not sample it; the transcode makes a BC image (the ASTC transcode)
    VKF_DEPTH = 8,
    VKF_STENCIL = 16,
    VKF_SRGB = 32,
    VKF_BLOCK = 64,       // any block-compressed format (BC and ASTC)
};

typedef struct VkfRow {
    VkFormat vk, vertex, alt;
    uint8_t swz[4], alt_swz[4];
    uint8_t flags, bw, bh;
    uint16_t bpb;
    const char *reason;   // why vk is VK_FORMAT_UNDEFINED, or a note on a stand-in; NULL otherwise
} VkfRow;

#define VKF_FIRST 0x01
#define VKF_LAST 0x9C

// The row of an NVNformat, or NULL for NVN_FORMAT_NONE and a value outside 0x01 to 0x9C. Without a device.
const VkfRow *vkf_row(int nvn_format);

// 1 when the device takes format `f` for texture use in the way the back end needs it (sampled, and as an attachment
// for a render target or depth format); the back end's answer from vkGetPhysicalDeviceFormatProperties.
typedef int (*VkfSupported)(VkFormat f, int depth_or_stencil, void *ctx);

// The format and view swizzle an image of the row's format is made in on a device: vk when supported, else alt when the
// row has one that is supported, else VK_FORMAT_UNDEFINED (with *why from the row). swz gets the four selectors.
VkFormat vkf_pick(const VkfRow *r, VkfSupported supported, void *ctx, uint8_t swz[4], const char **why);

// Formats whose views of one image may differ (VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT): uncompressed colour of one texel
// size share a class, each block-compressed family and each depth or stencil format is a class of its own. 0 for
// VK_FORMAT_UNDEFINED. Equal classes mean the specification's format compatibility class, which is what a view of the
// same image may use.
int vkf_class(VkFormat f);

// The sRGB format's linear twin, or f itself (the image is made in the linear one, so that storage writes and every
// view format of the class are legal).
VkFormat vkf_linear_twin(VkFormat f);

// ---- images of decoded ASTC. A device that does not sample ASTC takes the row's alt (RGBA8, or sRGB) for an ASTC texture,
// and the upload decodes the game's blocks into it: such an image holds texels, not the game's blocks. Without a device.
// 1 when an image made in format `made` for row r is that image of decoded texels (the row is ASTC and `made` is no BC or
// ASTC format); 0 for every other image, an ASTC one the device samples as blocks included.
int vkf_decoded(const VkfRow *r, VkFormat made);
// 1 when an image made so may stand in its pool's extent, where the game's storage lies in the row's own layout. An image of
// decoded texels never may: it would read the compressed blocks as RGBA8 texels, and its upload would overwrite them. It
// gets memory of its own.
int vkf_extent_ok(const VkfRow *r, VkFormat made);
// What a decoded image cannot serve: a program's stores (a storage view) and a copy of linear texel data in the row's block
// layout (a texel copy), which would address blocks of 16 bytes in an image of 4-byte texels.
typedef enum { VKF_USE_STORAGE_VIEW = 1, VKF_USE_TEXEL_COPY = 2 } VkfUse;
// The words naming `use` (for a stop) when an image made in `made` for row r is a decoded one, else NULL (the use is allowed).
const char *vkf_decoded_stop(const VkfRow *r, VkFormat made, VkfUse use);

// ---- the memory the images hold, by class (the memory line's table of the texture arena). Without a device.
// A class is what an image's bytes would save as BC: an ASTC texture decoded to RGBA8 (4 bytes a texel) is one class for each of
// the 14 ASTC block footprints and each of linear and sRGB; BC7 or BC3 would hold it at 1 byte a texel. An image made in BC is
// VKF_CLS_NATIVE_BC (already compressed), any other (including ASTC the device samples as blocks) VKF_CLS_OTHER.
enum { VKF_ASTC_FOOTPRINTS = 14, VKF_CLS_NATIVE_BC = 2 * VKF_ASTC_FOOTPRINTS, VKF_CLS_OTHER, VKF_CLS_COUNT };
// The class of an image made for row r in a format whose decoded state is `decoded` (vkf_decoded).
int vkf_img_class(const VkfRow *r, int decoded);
// The block footprint of an ASTC class (0 to 13, in the order 4x4 5x4 5x5 6x5 6x6 8x5 8x6 8x8 10x5 10x6 10x8 10x10 12x10 12x12)
// as its width and height, and 1 when the class is sRGB; 0 in *w for a class that is not an ASTC one.
void vkf_img_class_footprint(int cls, unsigned *w, unsigned *h, int *srgb);
// The bytes the image would hold as BC at 1 byte a texel: every mip level's width and height rounded up to the BC block of 4,
// times its depth, times the layers. A level halves from the base (a level's size is at least 1 before the rounding).
uint64_t vkf_bc_bytes(uint32_t w, uint32_t h, uint32_t d, uint32_t levels, uint32_t layers);

// The live images by class: how many, the bytes of memory they hold (the driver's size), the BC bytes of the decoded ones, and
// how many of the memory bytes lie in the texture arena (the rest is memory of their own).
typedef struct VkfImgTable {
    uint64_t images[VKF_CLS_COUNT], mem[VKF_CLS_COUNT], bc[VKF_CLS_COUNT], arena[VKF_CLS_COUNT];
} VkfImgTable;
// An image of class cls made (dir +1) or destroyed (dir -1): mem and bc are what it was added with; arena is 1 when its memory
// is an extent of the arena. The caller serialises.
void vkf_img_table_add(VkfImgTable *t, int cls, uint64_t mem, uint64_t bc, int arena, int dir);
// The memory line's columns from the table: decoded ASTC bytes by footprint width (4, 5, 6, 8, 10, 12), those of them that are
// sRGB and that lie in the arena, the decoded images, their BC bytes, and the bytes already BC and the bytes of any other class.
typedef struct VkfImgColumns {
    uint64_t dec_w[6], dec_srgb, dec_arena, dec_images, bc_equiv, native_bc, other;
} VkfImgColumns;
void vkf_img_columns(const VkfImgTable *t, VkfImgColumns *out);
