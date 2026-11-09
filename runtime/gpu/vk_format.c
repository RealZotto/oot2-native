// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The table of oot2/vk_format.h. The rows follow DKB Recomp's runtime/vk_formats.c (GPL-3.0-or-later; ported there from
// LAS Recomp runtime/nvn/vulkan/textures.c at da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026
// Peiikai): the same NVN to Vulkan pairs, the 16-bit packed stand-ins with their view swizzles and the formats with no
// equivalent. Changed here: the table has no device state (a device's answer comes in through VkfSupported), depth keeps
// the game's format (24-bit depth and depth-stencil stay 24-bit, with no 32-bit stand-in that would be larger than the
// game's storage), a row says the vertex attribute format with the NVN format's own components, and carries its
// block size, bytes per block and flags; BGRX8 samples alpha as 1 like every other X format.
#include <string.h>

#include "oot2/nvn.h"
#include "oot2/vk_format.h"

#define U VK_FORMAT_UNDEFINED
#define SR NVN_TEXTURE_SWIZZLE_R
#define SG NVN_TEXTURE_SWIZZLE_G
#define SB NVN_TEXTURE_SWIZZLE_B
#define SA NVN_TEXTURE_SWIZZLE_A

// An ordinary colour format; its vertex attribute format is itself.
#define C(nvn, f, bpb) [nvn] = {f, f, U, {0}, {0}, 0, 1, 1, bpb, NULL}
// The same, with flags (sRGB).
#define CF(nvn, f, bpb, fl) [nvn] = {f, f, U, {0}, {0}, fl, 1, 1, bpb, NULL}
// A 3-component format: the image is the four-component format, the vertex attribute keeps three.
#define RGB(nvn, image, vertex_f, bpb)                                                                                \
    [nvn] = {image, vertex_f, U, {0}, {0}, VKF_EXPAND_RGB, 1, 1, bpb,                                                 \
             "3-component format: each texel is expanded to 4 components on upload"}
// An X format: the fourth component is not stored, alpha samples as 1.
#define X(nvn, f, bpb, fl) [nvn] = {f, U, U, {0}, {0}, VKF_ALPHA_ONE | (fl), 1, 1, bpb, NULL}
// A format with no Vulkan equivalent.
#define N(nvn, why) [nvn] = {U, U, U, {0}, {0}, 0, 1, 1, 0, why}
// Block-compressed, 4 by 4 texels.
#define BC(nvn, f, bpb, fl) [nvn] = {f, U, U, {0}, {0}, VKF_BLOCK | (fl), 4, 4, bpb, NULL}
// Depth and stencil.
#define D(nvn, f, bpb, fl) [nvn] = {f, U, U, {0}, {0}, fl, 1, 1, bpb, NULL}
// A 16-bit packed colour format: vk names its components from the most significant bit, NVN from the least, so a view
// swizzle (c) puts them right; alt is the mandatory format that stands in when the device lacks vk, with its own (a).
#define P(nvn, f, fl, c0, c1, c2, c3, altf, a0, a1, a2, a3)                                                          \
    [nvn] = {f, U, altf, {c0, c1, c2, c3}, {a0, a1, a2, a3}, fl, 1, 1, 2,                                             \
             "16-bit packed format, NVN names the components from the least significant bit"}
// ASTC: the Vulkan format is the game's own. A device that does not sample it (desktop NVIDIA and AMD) takes alt, the RGBA8
// (or sRGB) image of the decoded texels, which the upload makes with runtime/gpu/shaders/astc_decode.comp: not the same
// bytes per block as vk, and no storage of the game's lies in such an image.
#define A(nvn, w, h, f, fl)                                                                                          \
    [nvn] = {f, U, (fl) ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, {0}, {0}, VKF_BLOCK | VKF_ASTC | (fl), w, h, 16, NULL}

#define UI2F "integer storage sampled as float (UI2F, I2F): Vulkan has no such format, it needs a shader-side conversion"

static const VkfRow rows[VKF_LAST + 1] = {
    C(0x01, VK_FORMAT_R8_UNORM, 1), C(0x02, VK_FORMAT_R8_SNORM, 1), C(0x03, VK_FORMAT_R8_UINT, 1),
    C(0x04, VK_FORMAT_R8_SINT, 1), C(0x05, VK_FORMAT_R16_SFLOAT, 2), C(0x06, VK_FORMAT_R16_UNORM, 2),
    C(0x07, VK_FORMAT_R16_SNORM, 2), C(0x08, VK_FORMAT_R16_UINT, 2), C(0x09, VK_FORMAT_R16_SINT, 2),
    C(0x0A, VK_FORMAT_R32_SFLOAT, 4), C(0x0B, VK_FORMAT_R32_UINT, 4), C(0x0C, VK_FORMAT_R32_SINT, 4),
    C(0x0D, VK_FORMAT_R8G8_UNORM, 2), C(0x0E, VK_FORMAT_R8G8_SNORM, 2), C(0x0F, VK_FORMAT_R8G8_UINT, 2),
    C(0x10, VK_FORMAT_R8G8_SINT, 2), C(0x11, VK_FORMAT_R16G16_SFLOAT, 4), C(0x12, VK_FORMAT_R16G16_UNORM, 4),
    C(0x13, VK_FORMAT_R16G16_SNORM, 4), C(0x14, VK_FORMAT_R16G16_UINT, 4), C(0x15, VK_FORMAT_R16G16_SINT, 4),
    C(0x16, VK_FORMAT_R32G32_SFLOAT, 8), C(0x17, VK_FORMAT_R32G32_UINT, 8), C(0x18, VK_FORMAT_R32G32_SINT, 8),
    RGB(0x19, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8_UNORM, 4),
    RGB(0x1A, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8_SNORM, 4),
    RGB(0x1B, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8_UINT, 4),
    RGB(0x1C, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8_SINT, 4),
    RGB(0x1D, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16_SFLOAT, 8),
    RGB(0x1E, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16_UNORM, 8),
    RGB(0x1F, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16_SNORM, 8),
    RGB(0x20, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16_UINT, 8),
    RGB(0x21, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R16G16B16_SINT, 8),
    RGB(0x22, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32_SFLOAT, 16),
    RGB(0x23, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32_UINT, 16),
    RGB(0x24, VK_FORMAT_R32G32B32A32_SINT, VK_FORMAT_R32G32B32_SINT, 16),
    C(0x25, VK_FORMAT_R8G8B8A8_UNORM, 4), C(0x26, VK_FORMAT_R8G8B8A8_SNORM, 4), C(0x27, VK_FORMAT_R8G8B8A8_UINT, 4),
    C(0x28, VK_FORMAT_R8G8B8A8_SINT, 4), C(0x29, VK_FORMAT_R16G16B16A16_SFLOAT, 8),
    C(0x2A, VK_FORMAT_R16G16B16A16_UNORM, 8), C(0x2B, VK_FORMAT_R16G16B16A16_SNORM, 8),
    C(0x2C, VK_FORMAT_R16G16B16A16_UINT, 8), C(0x2D, VK_FORMAT_R16G16B16A16_SINT, 8),
    C(0x2E, VK_FORMAT_R32G32B32A32_SFLOAT, 16), C(0x2F, VK_FORMAT_R32G32B32A32_UINT, 16),
    C(0x30, VK_FORMAT_R32G32B32A32_SINT, 16),
    D(0x31, VK_FORMAT_S8_UINT, 1, VKF_STENCIL), D(0x32, VK_FORMAT_D16_UNORM, 2, VKF_DEPTH),
    D(0x33, VK_FORMAT_X8_D24_UNORM_PACK32, 4, VKF_DEPTH), D(0x34, VK_FORMAT_D32_SFLOAT, 4, VKF_DEPTH),
    D(0x35, VK_FORMAT_D24_UNORM_S8_UINT, 4, VKF_DEPTH | VKF_STENCIL),
    D(0x36, VK_FORMAT_D32_SFLOAT_S8_UINT, 8, VKF_DEPTH | VKF_STENCIL),
    X(0x37, VK_FORMAT_R8G8B8A8_SRGB, 4, VKF_SRGB), CF(0x38, VK_FORMAT_R8G8B8A8_SRGB, 4, VKF_SRGB),
    // RGBA4: A4B4G4R4 (core in 1.3, optional), else B4G4R4A4 whose A, R, G, B hold NVN's R, G, B, A.
    P(0x39, VK_FORMAT_A4B4G4R4_UNORM_PACK16, 0, 0, 0, 0, 0, VK_FORMAT_B4G4R4A4_UNORM_PACK16, SA, SR, SG, SB),
    P(0x3A, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VKF_ALPHA_ONE, SB, SG, SR, SA, U, 0, 0, 0, 0),
    P(0x3B, VK_FORMAT_A1R5G5B5_UNORM_PACK16, 0, SB, SG, SR, SA, U, 0, 0, 0, 0),
    P(0x3C, VK_FORMAT_B5G6R5_UNORM_PACK16, 0, 0, 0, 0, 0, VK_FORMAT_R5G6B5_UNORM_PACK16, SB, SG, SR, SA),
    [0x3D] = {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UNORM_PACK32, U, {0}, {0}, 0, 1, 1, 4, NULL},
    [0x3E] = {VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, U, {0}, {0}, 0, 1, 1, 4, NULL},
    [0x3F] = {VK_FORMAT_B10G11R11_UFLOAT_PACK32, U, U, {0}, {0}, 0, 1, 1, 4, NULL},
    [0x40] = {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, U, U, {0}, {0}, 0, 1, 1, 4, NULL},
    BC(0x41, VK_FORMAT_BC1_RGB_UNORM_BLOCK, 8, 0), BC(0x42, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 8, 0),
    BC(0x43, VK_FORMAT_BC2_UNORM_BLOCK, 16, 0), BC(0x44, VK_FORMAT_BC3_UNORM_BLOCK, 16, 0),
    BC(0x45, VK_FORMAT_BC1_RGB_SRGB_BLOCK, 8, VKF_SRGB), BC(0x46, VK_FORMAT_BC1_RGBA_SRGB_BLOCK, 8, VKF_SRGB),
    BC(0x47, VK_FORMAT_BC2_SRGB_BLOCK, 16, VKF_SRGB), BC(0x48, VK_FORMAT_BC3_SRGB_BLOCK, 16, VKF_SRGB),
    BC(0x49, VK_FORMAT_BC4_UNORM_BLOCK, 8, 0), BC(0x4A, VK_FORMAT_BC4_SNORM_BLOCK, 8, 0),
    BC(0x4B, VK_FORMAT_BC5_UNORM_BLOCK, 16, 0), BC(0x4C, VK_FORMAT_BC5_SNORM_BLOCK, 16, 0),
    BC(0x4D, VK_FORMAT_BC7_UNORM_BLOCK, 16, 0), BC(0x4E, VK_FORMAT_BC7_SRGB_BLOCK, 16, VKF_SRGB),
    BC(0x4F, VK_FORMAT_BC6H_SFLOAT_BLOCK, 16, 0), BC(0x50, VK_FORMAT_BC6H_UFLOAT_BLOCK, 16, 0),
    // 0x51 to 0x68 (integer storage sampled as float) and 0x6B, 0x6C: no equivalent.
    N(0x51, UI2F), N(0x52, UI2F), N(0x53, UI2F), N(0x54, UI2F), N(0x55, UI2F), N(0x56, UI2F), N(0x57, UI2F),
    N(0x58, UI2F), N(0x59, UI2F), N(0x5A, UI2F), N(0x5B, UI2F), N(0x5C, UI2F), N(0x5D, UI2F), N(0x5E, UI2F),
    N(0x5F, UI2F), N(0x60, UI2F), N(0x61, UI2F), N(0x62, UI2F), N(0x63, UI2F), N(0x64, UI2F), N(0x65, UI2F),
    N(0x66, UI2F), N(0x67, UI2F), N(0x68, UI2F),
    [0x69] = {VK_FORMAT_A2B10G10R10_SNORM_PACK32, VK_FORMAT_A2B10G10R10_SNORM_PACK32, U, {0}, {0}, 0, 1, 1, 4, NULL},
    [0x6A] = {VK_FORMAT_A2B10G10R10_SINT_PACK32, VK_FORMAT_A2B10G10R10_SINT_PACK32, U, {0}, {0}, 0, 1, 1, 4, NULL},
    N(0x6B, UI2F), N(0x6C, UI2F),
    X(0x6D, VK_FORMAT_R8G8B8A8_UNORM, 4, 0), X(0x6E, VK_FORMAT_R8G8B8A8_SNORM, 4, 0),
    X(0x6F, VK_FORMAT_R8G8B8A8_UINT, 4, 0), X(0x70, VK_FORMAT_R8G8B8A8_SINT, 4, 0),
    X(0x71, VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0), X(0x72, VK_FORMAT_R16G16B16A16_UNORM, 8, 0),
    X(0x73, VK_FORMAT_R16G16B16A16_SNORM, 8, 0), X(0x74, VK_FORMAT_R16G16B16A16_UINT, 8, 0),
    X(0x75, VK_FORMAT_R16G16B16A16_SINT, 8, 0), X(0x76, VK_FORMAT_R32G32B32A32_SFLOAT, 16, 0),
    X(0x77, VK_FORMAT_R32G32B32A32_UINT, 16, 0), X(0x78, VK_FORMAT_R32G32B32A32_SINT, 16, 0),
    A(0x79, 4, 4, VK_FORMAT_ASTC_4x4_UNORM_BLOCK, 0), A(0x7A, 5, 4, VK_FORMAT_ASTC_5x4_UNORM_BLOCK, 0),
    A(0x7B, 5, 5, VK_FORMAT_ASTC_5x5_UNORM_BLOCK, 0), A(0x7C, 6, 5, VK_FORMAT_ASTC_6x5_UNORM_BLOCK, 0),
    A(0x7D, 6, 6, VK_FORMAT_ASTC_6x6_UNORM_BLOCK, 0), A(0x7E, 8, 5, VK_FORMAT_ASTC_8x5_UNORM_BLOCK, 0),
    A(0x7F, 8, 6, VK_FORMAT_ASTC_8x6_UNORM_BLOCK, 0), A(0x80, 8, 8, VK_FORMAT_ASTC_8x8_UNORM_BLOCK, 0),
    A(0x81, 10, 5, VK_FORMAT_ASTC_10x5_UNORM_BLOCK, 0), A(0x82, 10, 6, VK_FORMAT_ASTC_10x6_UNORM_BLOCK, 0),
    A(0x83, 10, 8, VK_FORMAT_ASTC_10x8_UNORM_BLOCK, 0), A(0x84, 10, 10, VK_FORMAT_ASTC_10x10_UNORM_BLOCK, 0),
    A(0x85, 12, 10, VK_FORMAT_ASTC_12x10_UNORM_BLOCK, 0), A(0x86, 12, 12, VK_FORMAT_ASTC_12x12_UNORM_BLOCK, 0),
    A(0x87, 4, 4, VK_FORMAT_ASTC_4x4_SRGB_BLOCK, VKF_SRGB), A(0x88, 5, 4, VK_FORMAT_ASTC_5x4_SRGB_BLOCK, VKF_SRGB),
    A(0x89, 5, 5, VK_FORMAT_ASTC_5x5_SRGB_BLOCK, VKF_SRGB), A(0x8A, 6, 5, VK_FORMAT_ASTC_6x5_SRGB_BLOCK, VKF_SRGB),
    A(0x8B, 6, 6, VK_FORMAT_ASTC_6x6_SRGB_BLOCK, VKF_SRGB), A(0x8C, 8, 5, VK_FORMAT_ASTC_8x5_SRGB_BLOCK, VKF_SRGB),
    A(0x8D, 8, 6, VK_FORMAT_ASTC_8x6_SRGB_BLOCK, VKF_SRGB), A(0x8E, 8, 8, VK_FORMAT_ASTC_8x8_SRGB_BLOCK, VKF_SRGB),
    A(0x8F, 10, 5, VK_FORMAT_ASTC_10x5_SRGB_BLOCK, VKF_SRGB), A(0x90, 10, 6, VK_FORMAT_ASTC_10x6_SRGB_BLOCK, VKF_SRGB),
    A(0x91, 10, 8, VK_FORMAT_ASTC_10x8_SRGB_BLOCK, VKF_SRGB), A(0x92, 10, 10, VK_FORMAT_ASTC_10x10_SRGB_BLOCK, VKF_SRGB),
    A(0x93, 12, 10, VK_FORMAT_ASTC_12x10_SRGB_BLOCK, VKF_SRGB), A(0x94, 12, 12, VK_FORMAT_ASTC_12x12_SRGB_BLOCK, VKF_SRGB),
    P(0x95, VK_FORMAT_R5G6B5_UNORM_PACK16, 0, 0, 0, 0, 0, U, 0, 0, 0, 0),
    P(0x96, VK_FORMAT_A1R5G5B5_UNORM_PACK16, VKF_ALPHA_ONE, 0, 0, 0, 0, U, 0, 0, 0, 0),
    P(0x97, VK_FORMAT_A1R5G5B5_UNORM_PACK16, 0, 0, 0, 0, 0, U, 0, 0, 0, 0),
    P(0x98, VK_FORMAT_R5G5B5A1_UNORM_PACK16, 0, 0, 0, 0, 0, U, 0, 0, 0, 0),
    X(0x99, VK_FORMAT_B8G8R8A8_UNORM, 4, 0), C(0x9A, VK_FORMAT_B8G8R8A8_UNORM, 4),
    X(0x9B, VK_FORMAT_B8G8R8A8_SRGB, 4, VKF_SRGB), CF(0x9C, VK_FORMAT_B8G8R8A8_SRGB, 4, VKF_SRGB),
};

const VkfRow *vkf_row(int f) {
    if (f < VKF_FIRST || f > VKF_LAST) return NULL;
    return &rows[f];
}

VkFormat vkf_pick(const VkfRow *r, VkfSupported supported, void *ctx, uint8_t swz[4], const char **why) {
    memset(swz, 0, 4);
    *why = NULL;
    if (!r) {
        *why = "not an NVN format";
        return VK_FORMAT_UNDEFINED;
    }
    if (r->vk == VK_FORMAT_UNDEFINED) {
        *why = r->reason;
        return VK_FORMAT_UNDEFINED;
    }
    int ds = (r->flags & (VKF_DEPTH | VKF_STENCIL)) != 0;
    if (supported(r->vk, ds, ctx)) {
        memcpy(swz, r->swz, 4);
        return r->vk;
    }
    if (r->alt != VK_FORMAT_UNDEFINED && supported(r->alt, ds, ctx)) {
        memcpy(swz, r->alt_swz, 4);
        return r->alt;
    }
    *why = "the device takes neither the format nor a stand-in for it";
    return VK_FORMAT_UNDEFINED;
}

int vkf_class(VkFormat f) {
    switch (f) {
    case VK_FORMAT_UNDEFINED: return 0;
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8_SRGB: return 8;
    case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8_SRGB: case VK_FORMAT_R5G6B5_UNORM_PACK16:
    case VK_FORMAT_B5G6R5_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
    case VK_FORMAT_A4B4G4R4_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return 16;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_SNORM_PACK32:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2B10G10R10_SINT_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_SFLOAT: return 32;
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: return 64;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: return 128;
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return 2001;
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return 2002;
    case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: return 2003;
    case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK: return 2004;
    case VK_FORMAT_BC4_UNORM_BLOCK: case VK_FORMAT_BC4_SNORM_BLOCK: return 2005;
    case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK: return 2006;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK: return 2007;
    case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK: return 2008;
    default: break;
    }
    // ASTC: UNORM and SRGB of one block size alternate from ASTC_4x4_UNORM_BLOCK.
    if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK)
        return 3000 + (int)(f - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) / 2;
    return 1000 + (int)f;   // depth and stencil formats, and anything else: a class of its own
}

VkFormat vkf_linear_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_SRGB: return VK_FORMAT_R8_UNORM;
    case VK_FORMAT_R8G8_SRGB: return VK_FORMAT_R8G8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case VK_FORMAT_BC2_SRGB_BLOCK: return VK_FORMAT_BC2_UNORM_BLOCK;
    case VK_FORMAT_BC3_SRGB_BLOCK: return VK_FORMAT_BC3_UNORM_BLOCK;
    case VK_FORMAT_BC7_SRGB_BLOCK: return VK_FORMAT_BC7_UNORM_BLOCK;
    default: break;
    }
    if (f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK && ((f - VK_FORMAT_ASTC_4x4_UNORM_BLOCK) & 1))
        return (VkFormat)(f - 1);
    return f;
}

// ---------- images of decoded ASTC

int vkf_decoded(const VkfRow *r, VkFormat made) {
    if (!r || !(r->flags & VKF_ASTC)) return 0;
    int c = vkf_class(made);
    return !(c >= 2001 && c < 4000);   // not in the BC and ASTC families: the device took the stand-in
}

int vkf_extent_ok(const VkfRow *r, VkFormat made) { return !vkf_decoded(r, made); }

const char *vkf_decoded_stop(const VkfRow *r, VkFormat made, VkfUse use) {
    if (!vkf_decoded(r, made)) return NULL;
    switch (use) {
    case VKF_USE_STORAGE_VIEW: return "a storage view";
    case VKF_USE_TEXEL_COPY: return "a texel copy";
    }
    return NULL;
}
