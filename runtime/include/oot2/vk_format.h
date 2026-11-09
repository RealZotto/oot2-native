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
