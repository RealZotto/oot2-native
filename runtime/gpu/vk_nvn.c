// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The tables of oot2/vk_nvn.h and the fillers of the driver constant banks and the support buffer.
//
// Adapted from DKB Recomp (GPL-3.0-or-later) at 9d8fa02: the NVN to Vulkan pairs of runtime/vk_pipeline.c (blend_factor,
// blend_op, compare_op, stencil_op, polygon_mode, vk_pipeline_topology, lines 334 to 395; ported there from LAS Recomp
// exec.c, MIT, Copyright (c) 2026 Peiikai), the layout of the graphics driver bank of runtime/vk_exec_draw.c (vk_c0_fill
// and c0_table, lines 997 to 1030), its support buffer (support_fill, line 1733) and the limits of the compute layout
// of runtime/nvn2_c0.h (nvn2_c0_valid). Changed here: every table is an explicit switch and refuses what it does not
// map where DKB falls back to a default (an unknown blend factor was ONE, an unknown stencil operation KEEP, LINE_LOOP
// a line strip that does not close, QUAD_STRIP and POLYGON a strip and a fan whose provoking vertex differs); the
// vertex formats are read from the format table (vk_format.c) and given their bytes and scalar type; the compute bank
// follows the translator's mapping (see vk_nvn.h) where DKB packs NVN2's own layout; a size past 32 bits is clamped; the
// support buffer follows tools/firstdraw/first_draw.py (the viewport's third component is 1, as the GPU run matched).
//
// The bank offsets and the c[2] tables are those of nxnative's dm/gfx/nvn/dmsh_abi.h and of the bank fills of
// dm/gfx/nvn/nvn_mtl.m (lines 8385 to 8420) at 865cc35 (MIT, licenses/nxnative-MIT.txt): the storage tables, the
// largest point size, the base instance, the uniform table and the group counts of an NVN2 compute head, and the rule
// that a bound image keeps the row a combined handle past slot 31 would take. Written afresh over those offsets.
#include <stddef.h>
#include <string.h>

#include "oot2/nvn.h"
#include "oot2/nvn_program.h"
#include "oot2/vk_format.h"
#include "oot2/vk_nvn.h"

// A case of a converter: the NVN value and the Vulkan value it maps to.
#define MAP(from, to)                                                                                                  \
    case from:                                                                                                         \
        *out = (to);                                                                                                   \
        return 1;

// ---------- converters

int vkn_topology(int nvn_prim, VkPrimitiveTopology *out) {
    switch (nvn_prim) {
        MAP(NVN_DRAW_PRIMITIVE_POINTS, VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
        MAP(NVN_DRAW_PRIMITIVE_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        MAP(NVN_DRAW_PRIMITIVE_LINE_STRIP, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP)
        MAP(NVN_DRAW_PRIMITIVE_TRIANGLES, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
        MAP(NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP)
        MAP(NVN_DRAW_PRIMITIVE_TRIANGLE_FAN, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN)
        MAP(NVN_DRAW_PRIMITIVE_LINES_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY)
        MAP(NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY)
        MAP(NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY)
        MAP(NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY)
        MAP(NVN_DRAW_PRIMITIVE_PATCHES, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST)
    default:   // LINE_LOOP, QUADS, QUAD_STRIP and POLYGON among them
        return 0;
    }
}

int vkn_index_type(int nvn_itype, VkIndexType *out, uint32_t *bytes) {
    uint32_t n;
    switch (nvn_itype) {
    case NVN_INDEX_TYPE_UNSIGNED_BYTE: *out = VK_INDEX_TYPE_UINT8_EXT, n = 1; break;
    case NVN_INDEX_TYPE_UNSIGNED_SHORT: *out = VK_INDEX_TYPE_UINT16, n = 2; break;
    case NVN_INDEX_TYPE_UNSIGNED_INT: *out = VK_INDEX_TYPE_UINT32, n = 4; break;
    default: return 0;
    }
    if (bytes) *bytes = n;
    return 1;
}

// The depth, stencil, sampler compare and alpha functions are one list in NVN: NEVER 1 to ALWAYS 8.
#define SAME_FUNC(name)                                                                                                \
    _Static_assert((int)NVN_DEPTH_FUNC_##name == (int)NVN_STENCIL_FUNC_##name &&                                       \
                       (int)NVN_DEPTH_FUNC_##name == (int)NVN_COMPARE_FUNC_##name &&                                   \
                       (int)NVN_DEPTH_FUNC_##name == (int)NVN_ALPHA_FUNC_##name,                                       \
                   "NVN's depth, stencil, compare and alpha functions share their values")
SAME_FUNC(NEVER);
SAME_FUNC(LESS);
SAME_FUNC(EQUAL);
SAME_FUNC(LEQUAL);
SAME_FUNC(GREATER);
SAME_FUNC(NOTEQUAL);
SAME_FUNC(GEQUAL);
SAME_FUNC(ALWAYS);

int vkn_compare(int nvn_func, VkCompareOp *out) {
    switch (nvn_func) {
        MAP(NVN_DEPTH_FUNC_NEVER, VK_COMPARE_OP_NEVER)
        MAP(NVN_DEPTH_FUNC_LESS, VK_COMPARE_OP_LESS)
        MAP(NVN_DEPTH_FUNC_EQUAL, VK_COMPARE_OP_EQUAL)
        MAP(NVN_DEPTH_FUNC_LEQUAL, VK_COMPARE_OP_LESS_OR_EQUAL)
        MAP(NVN_DEPTH_FUNC_GREATER, VK_COMPARE_OP_GREATER)
        MAP(NVN_DEPTH_FUNC_NOTEQUAL, VK_COMPARE_OP_NOT_EQUAL)
        MAP(NVN_DEPTH_FUNC_GEQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL)
        MAP(NVN_DEPTH_FUNC_ALWAYS, VK_COMPARE_OP_ALWAYS)
    default: return 0;
    }
}

int vkn_stencil_op(int nvn_op, VkStencilOp *out) {
    switch (nvn_op) {
        MAP(NVN_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP)
        MAP(NVN_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO)
        MAP(NVN_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE)
        MAP(NVN_STENCIL_OP_INCR, VK_STENCIL_OP_INCREMENT_AND_CLAMP)
        MAP(NVN_STENCIL_OP_DECR, VK_STENCIL_OP_DECREMENT_AND_CLAMP)
        MAP(NVN_STENCIL_OP_INVERT, VK_STENCIL_OP_INVERT)
        MAP(NVN_STENCIL_OP_INCR_WRAP, VK_STENCIL_OP_INCREMENT_AND_WRAP)
        MAP(NVN_STENCIL_OP_DECR_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP)
    default: return 0;
    }
}

int vkn_blend_factor(int nvn_func, VkBlendFactor *out) {
    switch (nvn_func) {
        MAP(NVN_BLEND_FUNC_ZERO, VK_BLEND_FACTOR_ZERO)
        MAP(NVN_BLEND_FUNC_ONE, VK_BLEND_FACTOR_ONE)
        MAP(NVN_BLEND_FUNC_SRC_COLOR, VK_BLEND_FACTOR_SRC_COLOR)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR)
        MAP(NVN_BLEND_FUNC_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        MAP(NVN_BLEND_FUNC_DST_ALPHA, VK_BLEND_FACTOR_DST_ALPHA)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        MAP(NVN_BLEND_FUNC_DST_COLOR, VK_BLEND_FACTOR_DST_COLOR)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR)
        MAP(NVN_BLEND_FUNC_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE)
        MAP(NVN_BLEND_FUNC_SRC1_COLOR, VK_BLEND_FACTOR_SRC1_COLOR)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR)
        MAP(NVN_BLEND_FUNC_SRC1_ALPHA, VK_BLEND_FACTOR_SRC1_ALPHA)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA)
        MAP(NVN_BLEND_FUNC_CONSTANT_COLOR, VK_BLEND_FACTOR_CONSTANT_COLOR)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR)
        MAP(NVN_BLEND_FUNC_CONSTANT_ALPHA, VK_BLEND_FACTOR_CONSTANT_ALPHA)
        MAP(NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
    default: return 0;
    }
}

int vkn_blend_op(int nvn_eq, VkBlendOp *out) {
    switch (nvn_eq) {
        MAP(NVN_BLEND_EQUATION_ADD, VK_BLEND_OP_ADD)
        MAP(NVN_BLEND_EQUATION_SUB, VK_BLEND_OP_SUBTRACT)
        MAP(NVN_BLEND_EQUATION_REVERSE_SUB, VK_BLEND_OP_REVERSE_SUBTRACT)
        MAP(NVN_BLEND_EQUATION_MIN, VK_BLEND_OP_MIN)
        MAP(NVN_BLEND_EQUATION_MAX, VK_BLEND_OP_MAX)
    default: return 0;
    }
}

int vkn_cull(int nvn_face, VkCullModeFlags *out) {
    switch (nvn_face) {
        MAP(NVN_FACE_NONE, VK_CULL_MODE_NONE)
        MAP(NVN_FACE_FRONT, VK_CULL_MODE_FRONT_BIT)
        MAP(NVN_FACE_BACK, VK_CULL_MODE_BACK_BIT)
        MAP(NVN_FACE_FRONT_AND_BACK, VK_CULL_MODE_FRONT_AND_BACK)
    default: return 0;
    }
}

int vkn_front_face(int nvn_front, VkFrontFace *out) {
    switch (nvn_front) {
        MAP(NVN_FRONT_FACE_CW, VK_FRONT_FACE_CLOCKWISE)
        MAP(NVN_FRONT_FACE_CCW, VK_FRONT_FACE_COUNTER_CLOCKWISE)
    default: return 0;
    }
}

int vkn_polygon_mode(int nvn_mode, VkPolygonMode *out) {
    switch (nvn_mode) {
        MAP(NVN_POLYGON_MODE_POINT, VK_POLYGON_MODE_POINT)
        MAP(NVN_POLYGON_MODE_LINE, VK_POLYGON_MODE_LINE)
        MAP(NVN_POLYGON_MODE_FILL, VK_POLYGON_MODE_FILL)
    default: return 0;
    }
}

int vkn_logic_op(int nvn_op, VkLogicOp *out) {
    switch (nvn_op) {
        MAP(NVN_LOGIC_OP_CLEAR, VK_LOGIC_OP_CLEAR)
        MAP(NVN_LOGIC_OP_AND, VK_LOGIC_OP_AND)
        MAP(NVN_LOGIC_OP_AND_REVERSE, VK_LOGIC_OP_AND_REVERSE)
        MAP(NVN_LOGIC_OP_COPY, VK_LOGIC_OP_COPY)
        MAP(NVN_LOGIC_OP_AND_INVERTED, VK_LOGIC_OP_AND_INVERTED)
        MAP(NVN_LOGIC_OP_NOOP, VK_LOGIC_OP_NO_OP)
        MAP(NVN_LOGIC_OP_XOR, VK_LOGIC_OP_XOR)
        MAP(NVN_LOGIC_OP_OR, VK_LOGIC_OP_OR)
        MAP(NVN_LOGIC_OP_NOR, VK_LOGIC_OP_NOR)
        MAP(NVN_LOGIC_OP_EQUIV, VK_LOGIC_OP_EQUIVALENT)
        MAP(NVN_LOGIC_OP_INVERT, VK_LOGIC_OP_INVERT)
        MAP(NVN_LOGIC_OP_OR_REVERSE, VK_LOGIC_OP_OR_REVERSE)
        MAP(NVN_LOGIC_OP_COPY_INVERTED, VK_LOGIC_OP_COPY_INVERTED)
        MAP(NVN_LOGIC_OP_OR_INVERTED, VK_LOGIC_OP_OR_INVERTED)
        MAP(NVN_LOGIC_OP_NAND, VK_LOGIC_OP_NAND)
        MAP(NVN_LOGIC_OP_SET, VK_LOGIC_OP_SET)
    default: return 0;
    }
}

// ---------- vertex formats

// The vertex formats of the table (VkfRow.vertex), with their bytes and the scalar type a shader sees. A format of the
// table that is not here (an sRGB one, which the table keeps for its image use) is not a vertex format.
typedef struct {
    VkFormat vk;
    uint8_t bytes, type;
} VertexTraits;
#define VF(name, bytes) {VK_FORMAT_##name, bytes, VKN_ATTR_FLOAT}   // normalized, scaled and floating point
#define VS(name, bytes) {VK_FORMAT_##name, bytes, VKN_ATTR_SINT}
#define VU(name, bytes) {VK_FORMAT_##name, bytes, VKN_ATTR_UINT}
static const VertexTraits vertex_traits[] = {
    VF(R8_UNORM, 1), VF(R8_SNORM, 1), VU(R8_UINT, 1), VS(R8_SINT, 1),
    VF(R8G8_UNORM, 2), VF(R8G8_SNORM, 2), VU(R8G8_UINT, 2), VS(R8G8_SINT, 2),
    VF(R8G8B8_UNORM, 3), VF(R8G8B8_SNORM, 3), VU(R8G8B8_UINT, 3), VS(R8G8B8_SINT, 3),
    VF(R8G8B8A8_UNORM, 4), VF(R8G8B8A8_SNORM, 4), VU(R8G8B8A8_UINT, 4), VS(R8G8B8A8_SINT, 4),
    VF(B8G8R8A8_UNORM, 4),
    VF(R16_SFLOAT, 2), VF(R16_UNORM, 2), VF(R16_SNORM, 2), VU(R16_UINT, 2), VS(R16_SINT, 2),
    VF(R16G16_SFLOAT, 4), VF(R16G16_UNORM, 4), VF(R16G16_SNORM, 4), VU(R16G16_UINT, 4), VS(R16G16_SINT, 4),
    VF(R16G16B16_SFLOAT, 6), VF(R16G16B16_UNORM, 6), VF(R16G16B16_SNORM, 6), VU(R16G16B16_UINT, 6),
    VS(R16G16B16_SINT, 6),
    VF(R16G16B16A16_SFLOAT, 8), VF(R16G16B16A16_UNORM, 8), VF(R16G16B16A16_SNORM, 8), VU(R16G16B16A16_UINT, 8),
    VS(R16G16B16A16_SINT, 8),
    VF(R32_SFLOAT, 4), VU(R32_UINT, 4), VS(R32_SINT, 4),
    VF(R32G32_SFLOAT, 8), VU(R32G32_UINT, 8), VS(R32G32_SINT, 8),
    VF(R32G32B32_SFLOAT, 12), VU(R32G32B32_UINT, 12), VS(R32G32B32_SINT, 12),
    VF(R32G32B32A32_SFLOAT, 16), VU(R32G32B32A32_UINT, 16), VS(R32G32B32A32_SINT, 16),
    VF(A2B10G10R10_UNORM_PACK32, 4), VF(A2B10G10R10_SNORM_PACK32, 4), VU(A2B10G10R10_UINT_PACK32, 4),
    VS(A2B10G10R10_SINT_PACK32, 4),
};

int vkn_vertex_format(int nvn_format, VkFormat *out, uint32_t *bytes, int *type) {
    const VkfRow *row = vkf_row(nvn_format);
    if (!row || row->vertex == VK_FORMAT_UNDEFINED) return 0;
    for (size_t i = 0; i < sizeof vertex_traits / sizeof *vertex_traits; i++) {
        if (vertex_traits[i].vk != row->vertex) continue;
        *out = row->vertex;
        if (bytes) *bytes = vertex_traits[i].bytes;
        if (type) *type = vertex_traits[i].type;
        return 1;
    }
    return 0;
}

// ---------- the banks

// Each table is 16 rows of 16 bytes. The graphics tables and the compute tables must not overlap what lies after them.
#define C0_ROW 16u
#define C0_SSBO_VS 0x110u
#define C0_SSBO_CS 0x310u
#define C0_SSBO_PS 0x510u
#define C0_POINT_SIZE 0x610u
#define C0_BASE_INSTANCE 0x644u
#define C0_SSBO_TS 0x670u
#define C0_SSBO_MS 0x770u
// The group counts are three words at 4, 8 and 0xc; the head's first word stays zero. The translator's modules settle
// it. Of the 4128 compute programs of the prepared pack, 4110 read nothing under 0x10 of the driver bank (none uses a
// NumWorkgroups builtin), 6 read the word at 4 only and 12 read the words at 4 and 8; none reads 0 or 0xc. Each of
// those 18 scales the word at 4 by its local size's x extent and the word at 8 by the y extent (the 8x8, 16x16 and
// 32x32 programs shift by 3, 4 and 5), so 4 is the x count and 8 the y count. Counts at 0, 4 and 8 would hand these
// programs the y count where they read x. A back end that places x at 0 has another layout of the head.
#define C0_DISPATCH_NVN2 4u
#define C0_CS_UBO 0x210u
#define C0_TABLE_BYTES (16u * C0_ROW)
_Static_assert(C0_DISPATCH_NVN2 + 12 <= 0x10, "the group counts lie in the 16-byte head");
_Static_assert(C0_SSBO_VS + C0_TABLE_BYTES <= C0_CS_UBO, "the vertex table ends before the compute uniform table");
_Static_assert(C0_CS_UBO + C0_TABLE_BYTES == C0_SSBO_CS, "the compute uniform table ends where the storage table starts");
_Static_assert(C0_SSBO_CS + C0_TABLE_BYTES <= C0_SSBO_PS, "the compute storage table ends before the pixel table");
_Static_assert(C0_SSBO_PS + C0_TABLE_BYTES <= C0_POINT_SIZE, "the pixel table ends before the point size");
_Static_assert(C0_BASE_INSTANCE + 4 <= C0_SSBO_TS, "the base instance lies before the task table");
_Static_assert(C0_SSBO_TS + C0_TABLE_BYTES == C0_SSBO_MS, "the mesh table follows the task table");
_Static_assert(C0_SSBO_MS + C0_TABLE_BYTES <= VKN_BANK_BYTES, "the mesh table lies in the bank");

#define C2_TEX 0x20u
#define C2_TEX_SLOTS 41u
#define C2_IMG 0x120u
#define C2_IMG_SLOTS 8u
#define C2_SEPTEX 0x168u
#define C2_SAMP 0x568u
#define C2_SEP_SLOTS 128u
#define C2_ROW 8u
#define C2_FIRST_OVERLAP 32u   // the combined slot that shares its row with image slot 0
_Static_assert(C2_TEX + C2_ROW * C2_TEX_SLOTS == C2_SEPTEX, "the combined handles end where the separate textures start");
_Static_assert(C2_IMG == C2_TEX + C2_ROW * C2_FIRST_OVERLAP, "image slot 0 lies over combined slot 32");
_Static_assert(C2_IMG + C2_ROW * C2_IMG_SLOTS <= C2_SEPTEX, "the image rows end before the separate textures");
_Static_assert(C2_SEPTEX + C2_ROW * C2_SEP_SLOTS == C2_SAMP, "the separate textures end where the samplers start");
_Static_assert(C2_SAMP + C2_ROW * C2_SEP_SLOTS <= VKN_BANK_BYTES, "the samplers lie in the bank");

#define SUPPORT_VIEWPORT_INVERSE 144u
#define SUPPORT_VIEWPORT_SIZE 160u
#define SUPPORT_RENDER_SCALE 192u
#define SUPPORT_RENDER_SCALES 73u
_Static_assert(SUPPORT_RENDER_SCALE + 16 * SUPPORT_RENDER_SCALES <= VKN_SUPPORT_BYTES, "the render scales lie in the buffer");

int vkn_c0_ssbo_base(int stage_kind) {
    switch (stage_kind) {
    case NVN_STAGE_VS: return (int)C0_SSBO_VS;
    case NVN_STAGE_PS: return (int)C0_SSBO_PS;
    case NVN_STAGE_CS: return (int)C0_SSBO_CS;
    case NVN_STAGE_TS: return (int)C0_SSBO_TS;
    case NVN_STAGE_MS: return (int)C0_SSBO_MS;
    default: return -1;
    }
}

// 16 rows of {u64 address, u32 size} at `at`; the 4 bytes after a size stay as they were. NULL rows write nothing.
static void put_table(uint8_t *at, const VknBuf rows[16]) {
    if (!rows) return;
    for (unsigned r = 0; r < 16; r++) {
        uint64_t addr = rows[r].addr;
        uint32_t size = rows[r].size > UINT32_MAX ? UINT32_MAX : (uint32_t)rows[r].size;
        memcpy(at + C0_ROW * r, &addr, 8);
        memcpy(at + C0_ROW * r + 8, &size, 4);
    }
}

void vkn_c0_graphics(uint8_t bank[VKN_BANK_BYTES], int stage_kind, const VknBuf ssbo[16], uint32_t base_instance) {
    memset(bank, 0, VKN_BANK_BYTES);
    if (stage_kind == NVN_STAGE_CS) return;
    int base = vkn_c0_ssbo_base(stage_kind);
    if (base < 0) return;
    put_table(bank + base, ssbo);
    const float max_point_size = 2047.0f;
    memcpy(bank + C0_POINT_SIZE, &max_point_size, 4);
    memcpy(bank + C0_BASE_INSTANCE, &base_instance, 4);
}

// The control section of an NVN2 compute stage (nvn_stage.ctl) at these byte offsets.
#define CTL_ROOTS 0xA9u
#define CTL_UBOS 0xB1u
#define CTL_SSBOS 0xB2u
#define CTL_TEXTURES 0xB3u
#define CTL_SEP_TEXTURES 0xB4u
#define CTL_SEP_SAMPLERS 0xB5u
#define CTL_IMAGES 0xB6u
#define CTL_READ_BYTES (CTL_IMAGES + 1)

int vkn_c0_layout(const uint8_t *ctl, uint32_t ctl_size, VknC0Layout *out) {
    if (!ctl || ctl_size < CTL_READ_BYTES) return 0;
    VknC0Layout l;
    memcpy(l.roots, ctl + CTL_ROOTS, sizeof l.roots);
    l.ubos = ctl[CTL_UBOS], l.ssbos = ctl[CTL_SSBOS], l.textures = ctl[CTL_TEXTURES];
    l.sep_textures = ctl[CTL_SEP_TEXTURES], l.sep_samplers = ctl[CTL_SEP_SAMPLERS], l.images = ctl[CTL_IMAGES];
    if (l.ubos > 16 || l.ssbos > 16 || l.textures > 128 || l.sep_textures > C2_SEP_SLOTS ||
        l.sep_samplers > C2_SEP_SLOTS || l.images > 32)
        return 0;
    for (unsigned i = 0; i < sizeof l.roots; i++)
        if (l.roots[i] > 64) return 0;
    *out = l;
    return 1;
}

int vkn_c0_compute(uint8_t bank[VKN_BANK_BYTES], const uint8_t *ctl, uint32_t ctl_size, const uint32_t groups[3],
                   const uint32_t roots[8][64], const VknBuf ubo[16], const VknBuf ssbo[16]) {
    (void)roots;
    memset(bank, 0, VKN_BANK_BYTES);
    VknC0Layout layout;
    if (!vkn_c0_layout(ctl, ctl_size, &layout)) return 0;
    if (groups) memcpy(bank + C0_DISPATCH_NVN2, groups, 12);
    put_table(bank + C0_CS_UBO, ubo);
    put_table(bank + C0_SSBO_CS, ssbo);
    return 1;
}

// One handle: its low 32 bits at the start of an 8-byte row.
static void put_handle(uint8_t *bank, unsigned offset, uint64_t handle) {
    uint32_t word = (uint32_t)handle;
    memcpy(bank + offset, &word, 4);
}

void vkn_c2_fill(uint8_t bank[VKN_BANK_BYTES], const uint64_t tex[41], const uint64_t img[8], const uint64_t septex[128],
                 const uint64_t samp[128]) {
    memset(bank, 0, VKN_BANK_BYTES);
    if (img)
        for (unsigned r = 0; r < C2_IMG_SLOTS; r++) put_handle(bank, C2_IMG + C2_ROW * r, img[r]);
    if (tex) {
        for (unsigned r = 0; r < C2_TEX_SLOTS; r++) {
            // Slots 32 to 39 share their rows with the image slots: a bound image keeps its row, and an unbound combined
            // slot leaves the row as the image slot wrote it.
            if (r >= C2_FIRST_OVERLAP && (!tex[r] || (img && r - C2_FIRST_OVERLAP < C2_IMG_SLOTS && img[r - C2_FIRST_OVERLAP])))
                continue;
            put_handle(bank, C2_TEX + C2_ROW * r, tex[r]);
        }
    }
    if (septex)
        for (unsigned r = 0; r < C2_SEP_SLOTS; r++) put_handle(bank, C2_SEPTEX + C2_ROW * r, septex[r]);
    if (samp)
        for (unsigned r = 0; r < C2_SEP_SLOTS; r++) put_handle(bank, C2_SAMP + C2_ROW * r, samp[r]);
}

void vkn_support_fill(uint8_t b[VKN_SUPPORT_BYTES], float width, float height) {
    memset(b, 0, VKN_SUPPORT_BYTES);
    const float inverse[4] = {width > 0 ? 2.0f / width : 0, height > 0 ? 2.0f / height : 0, 1.0f, 0.0f};
    const float size[4] = {width, height, 1.0f, 0.0f};
    memcpy(b + SUPPORT_VIEWPORT_INVERSE, inverse, sizeof inverse);
    memcpy(b + SUPPORT_VIEWPORT_SIZE, size, sizeof size);
    const float scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};   // every render scale 1: the host renders at the game's size
    for (unsigned i = 0; i < SUPPORT_RENDER_SCALES; i++) memcpy(b + SUPPORT_RENDER_SCALE + 16 * i, scale, sizeof scale);
}
