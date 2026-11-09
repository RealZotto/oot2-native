// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The transfers of the replay (oot2/vk_replay.h): buffer and texture copies, buffer fills, texture clears and uniform
// updates. Each is one transfer command after the end of the open pass and a full memory barrier (D18), with the buffer
// sides resolved through the backing map in the frame's submission. A write into a buffer is declared (vkr_wrote): the
// uniform copies it touches are dropped (D17a) and, in memory the CPU can see, the backing map brings the bytes to guest
// memory once the frame is done (a mirror) or the GPU wrote the guest's own pages (an import).
//
// The copy semantics (region and layer arithmetic, row and image strides, whole-level clears, the choice between a copy
// and a blit) follow DKB Recomp's runtime/vk_exec_xfer.c at 9d8fa02 (GPL-3.0-or-later; there after MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m and LAS Recomp runtime/nvn/vulkan/exec.c, both MIT). The code is ours.
// Changed here: a texture is the image of oot2/vk_image.h in the general layout (a texture over memory the CPU can see is
// a named stop until the texture arena serves copies), a partial clear is made inside a pass of its own instead of being
// skipped, and every refusal is a named stop instead of a logged skip.
#include <string.h>

#include "oot2/nvn_obj.h"
#include "oot2/runtime.h"
#include "oot2/vk_format.h"
#include "oot2/vk_rec.h"
#include "vk_replay_int.h"

#define stop(...) vkr_stop(__VA_ARGS__)

// The copy strides of buffer and texture copies, the channel's state (0: rows and images tightly packed).
static int64_t g_row, g_img;

void vkr_xfer_reset(void) { g_row = g_img = 0; }

static uint32_t max1(uint32_t v) { return v ? v : 1; }

static BackingRef buffer_ref(uint64_t addr, uint64_t size) {
    BackingRef r;
    vkr_frame_ops()->resolve(addr, size ? size : 1, vkr_frame_value(), &r);
    return r;
}

static VkBuffer buffer_of(const BackingRef *r) { return (VkBuffer)(uintptr_t)r->buffer; }

// The frame wrote [addr, addr + size) of a buffer.
static void wrote(uint64_t addr, uint64_t size) { vkr_wrote(addr, size, 0, 0); }

// ---------- buffers
static void op_copy_b2b(const p_copy_b2b *p) {
    if (!p->size) return;
    BackingRef s = buffer_ref(p->src, p->size), d = buffer_ref(p->dst, p->size);
    if (s.buffer == d.buffer && s.offset < d.offset + p->size && d.offset < s.offset + p->size)
        stop("a buffer copy of %#llx bytes from %#llx to %#llx overlaps itself: it has no single transfer command",
             (ull)p->size, (ull)p->src, (ull)p->dst);
    VkBufferCopy rg = {s.offset, d.offset, p->size};
    vkr_xfer_begin();
    vkrec_CopyBuffer(buffer_of(&s), buffer_of(&d), &rg);
    vkr_xfer_end();
    wrote(p->dst, p->size);
    vkr_count_transfer(p->size);
}

// ClearBuffer and ReportValue (a 4-byte clear): vkCmdFillBuffer, in GPU order after a full barrier, so the value lands
// after the work before it.
static void op_clear_buf(const p_clear_buf *p) {
    if (!p->size) return;
    BackingRef d = buffer_ref(p->addr, p->size);
    if ((d.offset | p->size) & 3)
        stop("a buffer fill of %#llx bytes at %#llx (offset %#llx in its buffer) is not 4-byte aligned", (ull)p->size,
             (ull)p->addr, (ull)d.offset);
    vkr_xfer_begin();
    vkrec_FillBuffer(buffer_of(&d), d.offset, p->size, p->value);
    vkr_xfer_end();
    wrote(p->addr, p->size);
    vkr_count_transfer(p->size);
}

// UpdateUniformBuffer: the bytes were copied when it recorded; they go through the frame's scratch memory and one buffer
// copy, whatever the size and alignment (vkCmdUpdateBuffer takes at most 64 KiB, in multiples of 4).
static void op_update_ub(const p_update_ub *p) {
    if (!p->size) return;
    if (p->size > UINT32_MAX) stop("a uniform buffer update of %#llx bytes", (ull)p->size);
    BackingRef d = buffer_ref(p->dst, p->size);
    VkBuffer sb;
    VkDeviceSize so;
    memcpy(vkr_scratch((uint32_t)p->size, 4, &sb, &so), p->data, p->size);
    VkBufferCopy rg = {so, d.offset, p->size};
    vkr_xfer_begin();
    vkrec_CopyBuffer(sb, buffer_of(&d), &rg);
    vkr_xfer_end();
    wrote(p->dst, p->size);
    vkr_count_transfer(p->size);
}

// ---------- textures
// A texture's level, layers and region as image coordinates.
typedef struct {
    VkrImage img;
    const VkfRow *row;
    uint32_t level, layer0;
    int32_t x, y, z;
    uint32_t w, h, nz;      // the region's extent; nz: its slices (3D) or layers
    int is3d;
} Sub;

static void sub_resolve(const char *what, const nvn_tex_rec *t, const NVNcopyRegion *r, Sub *s) {
    memset(s, 0, sizeof *s);
    if (!t->present) stop("%s over a texture the record does not hold", what);
    s->row = vkf_row(t->tex.d.format);
    if (!s->row || !s->row->bpb) stop("%s over NVN format %#x, which has no block size", what, (unsigned)t->tex.d.format);
    if (!vkr_frame_ops()->image(t, vkr_frame_value(), &s->img))
        stop("%s over a texture with no image (NVN format %#x, %ux%u): its storage is in memory the CPU can see (the "
             "texture arena makes those) or its format has no image on this device, see oot2/vk_image.h", what,
             (unsigned)t->tex.d.format, t->tex.d.width, t->tex.d.height);
    if (s->img.samples > 1) stop("%s over a multisampled texture (%u samples)", what, s->img.samples);
    const view_t *v = t->has_view ? &t->view : NULL;
    s->level = v && v->set_levels ? v->base_level : 0;
    s->layer0 = v && v->set_layers ? v->min_layer : 0;
    s->is3d = s->img.type == VK_IMAGE_TYPE_3D;
    uint32_t lw = s->level < 32 ? max1(s->img.width >> s->level) : 1, lh = s->level < 32 ? max1(s->img.height >> s->level) : 1,
             ld = s->level < 32 ? max1(s->img.depth >> s->level) : 1;
    s->x = r->xoffset, s->y = r->yoffset, s->z = r->zoffset;
    s->nz = r->depth > 0 ? (uint32_t)r->depth : 1;
    // A zero extent is the rest of the level from the offset.
    s->w = r->width > 0 ? (uint32_t)r->width : s->x >= 0 && (uint32_t)s->x < lw ? lw - (uint32_t)s->x : 0;
    s->h = r->height > 0 ? (uint32_t)r->height : s->y >= 0 && (uint32_t)s->y < lh ? lh - (uint32_t)s->y : 0;
    uint64_t layers_end = (uint64_t)s->layer0 + (uint64_t)(s->z > 0 ? s->z : 0) + s->nz;
    if (s->level >= s->img.levels || s->x < 0 || s->y < 0 || s->z < 0 || !s->w || !s->h ||
        (uint64_t)s->x + s->w > lw || (uint64_t)s->y + s->h > lh ||
        (s->is3d ? (uint64_t)s->z + s->nz > ld : layers_end > s->img.layers))
        stop("%s over the region %d,%d,%d %ux%ux%u outside level %u of a %ux%u texture with %u levels and %u layers", what,
             s->x, s->y, s->z, s->w, s->h, s->nz, s->level, s->img.width, s->img.height, s->img.levels, s->img.layers);
}

// The aspects a copy between images takes: the colour, or every aspect of a depth or stencil format.
static VkImageAspectFlags image_aspects(const Sub *s) {
    return s->img.aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : s->img.all_aspects;
}

static VkImageSubresourceLayers sub_layers(const Sub *s, VkImageAspectFlags aspect) {
    return (VkImageSubresourceLayers){aspect, s->level, s->is3d ? 0 : s->layer0 + (uint32_t)s->z, s->is3d ? 1 : s->nz};
}
static VkOffset3D sub_offset(const Sub *s) { return (VkOffset3D){s->x, s->y, s->is3d ? s->z : 0}; }
static VkExtent3D sub_extent(const Sub *s) { return (VkExtent3D){s->w, s->h, s->is3d ? s->nz : 1}; }

// The buffer side of a copy between a buffer and a texture: the bytes it covers and the Vulkan region. A row is the
// channel's copy row stride (tight: the blocks of the region's width), an image the copy image stride (tight: the rows
// of the region's height).
typedef struct {
    VkBufferImageCopy c;
    BackingRef ref;
    uint64_t addr, size;
} BufSide;

static BufSide buffer_side(const char *what, const Sub *s, uint64_t addr) {
    if (s->img.aspect != VK_IMAGE_ASPECT_COLOR_BIT && !(s->img.aspect == VK_IMAGE_ASPECT_DEPTH_BIT && s->img.format == VK_FORMAT_D32_SFLOAT))
        stop("%s over a depth or stencil image (Vulkan format %d): only colour and 32-bit depth copy through a buffer", what,
             (int)s->img.format);
    uint32_t bpb = s->row->bpb, bw = max1(s->row->bw), bh = max1(s->row->bh);
    uint64_t wb = (s->w + bw - 1) / bw, hb = (s->h + bh - 1) / bh, tight = wb * bpb;
    uint64_t row = g_row > 0 ? (uint64_t)g_row : tight;
    if (row % bpb || row < tight)
        stop("%s with a copy row stride of %llu bytes over %llu blocks of %u bytes", what, (ull)row, (ull)wb, bpb);
    uint64_t pitch = g_img > 0 ? (uint64_t)g_img : row * hb;
    if (pitch % row || pitch < row * hb)
        stop("%s with a copy image stride of %llu bytes over %llu rows of %llu bytes", what, (ull)pitch, (ull)hb, (ull)row);
    BufSide b = {.addr = addr, .size = pitch * (s->nz - 1) + row * (hb - 1) + tight};
    if (!addr) stop("%s at no address", what);
    b.ref = buffer_ref(addr, b.size);
    if (b.ref.offset % bpb || b.ref.offset % 4)
        stop("%s at %#llx, offset %#llx in its buffer: not a multiple of the block size %u and of 4", what, (ull)addr,
             (ull)b.ref.offset, bpb);
    b.c = (VkBufferImageCopy){.bufferOffset = b.ref.offset, .bufferRowLength = (uint32_t)(row / bpb * bw),
                              .bufferImageHeight = (uint32_t)(pitch / row * bh),
                              .imageSubresource = sub_layers(s, s->img.aspect), .imageOffset = sub_offset(s),
                              .imageExtent = sub_extent(s)};
    return b;
}

static void op_copy_b2t(const p_copy_b2t *p, int to_texture) {
    const char *what = to_texture ? "CopyBufferToTexture" : "CopyTextureToBuffer";
    if (p->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT | NVN_COPY_FLAGS_MIRROR_Z_BIT))
        stop("%s with copy flags %#x: a mirrored copy is not replayed yet", what, p->flags);
    Sub s;
    sub_resolve(what, &p->tex, &p->r, &s);
    // An ASTC texture on a device that does not sample ASTC is an image of decoded texels (vkf_decoded, oot2/vk_format.h)
    // while the buffer holds the game's blocks: the copy would move blocks as texels. A copy between two such images is
    // texel against texel and stays.
    const char *refused = vkf_decoded_stop(s.row, s.img.format, VKF_USE_TEXEL_COPY);
    if (refused)
        stop("%s over a texture of ASTC blocks decoded to RGBA8 (NVN format %#x): %s between the game's blocks and the "
             "image's texels is not replayed yet", what, (unsigned)p->tex.tex.d.format, refused);
    BufSide b = buffer_side(what, &s, p->addr);
    vkr_xfer_begin();
    if (to_texture) {
        vkrec_CopyBufferToImage(buffer_of(&b.ref), s.img.image, &b.c);
    } else {
        vkrec_CopyImageToBuffer(s.img.image, buffer_of(&b.ref), &b.c);
    }
    vkr_xfer_end();
    if (!to_texture) wrote(b.addr, b.size);
    vkr_count_transfer(b.size);
}

static void op_copy_t2t(const p_copy_t2t *p) {
    if (p->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT | NVN_COPY_FLAGS_MIRROR_Z_BIT))
        stop("CopyTextureToTexture with copy flags %#x: a mirrored copy is not replayed yet", p->flags);
    Sub s, d;
    sub_resolve("CopyTextureToTexture (source)", &p->src, &p->sr, &s);
    sub_resolve("CopyTextureToTexture (destination)", &p->dst, &p->dr, &d);
    d.nz = s.nz;   // the layers (slices) the source gives
    if (s.img.aspect != d.img.aspect || (s.img.aspect != VK_IMAGE_ASPECT_COLOR_BIT && s.img.format != d.img.format))
        stop("CopyTextureToTexture between NVN formats %#x and %#x", (unsigned)p->src.tex.d.format,
             (unsigned)p->dst.tex.d.format);
    int same_size = s.w == d.w && s.h == d.h;
    int same_blocks = s.row->bpb == d.row->bpb && max1(s.row->bw) == max1(d.row->bw) && max1(s.row->bh) == max1(d.row->bh);
    vkr_xfer_begin();
    if (same_size && same_blocks) {
        VkImageCopy rg = {sub_layers(&s, image_aspects(&s)), sub_offset(&s), sub_layers(&d, image_aspects(&s)),
                          sub_offset(&d), sub_extent(&s)};
        vkrec_CopyImage(s.img.image, d.img.image, &rg);
    } else {
        if (s.img.aspect != VK_IMAGE_ASPECT_COLOR_BIT || max1(s.row->bw) > 1 || max1(d.row->bw) > 1 ||
            max1(s.row->bh) > 1 || max1(d.row->bh) > 1)
            stop("CopyTextureToTexture of %ux%u to %ux%u texels between NVN formats %#x and %#x: only colour of uncompressed "
                 "formats scales or converts", s.w, s.h, d.w, d.h, (unsigned)p->src.tex.d.format,
                 (unsigned)p->dst.tex.d.format);
        VkImageBlit rg = {.srcSubresource = sub_layers(&s, VK_IMAGE_ASPECT_COLOR_BIT),
                          .dstSubresource = sub_layers(&d, VK_IMAGE_ASPECT_COLOR_BIT)};
        rg.srcOffsets[0] = sub_offset(&s);
        rg.srcOffsets[1] = (VkOffset3D){s.x + (int32_t)s.w, s.y + (int32_t)s.h, s.is3d ? s.z + (int32_t)s.nz : 1};
        rg.dstOffsets[0] = sub_offset(&d);
        rg.dstOffsets[1] = (VkOffset3D){d.x + (int32_t)d.w, d.y + (int32_t)d.h, d.is3d ? d.z + (int32_t)d.nz : 1};
        vkrec_BlitImage(s.img.image, d.img.image, &rg,
                        (p->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    }
    vkr_xfer_end();
    vkr_count_transfer((uint64_t)s.w * s.h * s.nz * s.row->bpb / (max1(s.row->bw) * max1(s.row->bh)));
}

// ClearTexture. A whole level (every layer or slice of the region) is one vkCmdClear*Image; a part of one is a clear
// of that rectangle inside a pass of its own over the level's layers, which the replay's pass rules end again at once.
// The clear value is the record's 16 bytes as the format reads them (float, or the integer forms): depth is the first
// float and stencil the second word when the format has both (the first when stencil alone), as DKB Recomp has it.
static void clear_partial(const char *what, const nvn_tex_rec *t, const Sub *s, const p_clear_tex *p, int colour) {
    if (s->is3d) stop("%s of part of a 3D texture's level", what);
    nvn_tex_rec tr = *t;
    tr.has_view = 1;
    tr.view.set_levels = tr.view.set_layers = 1;
    tr.view.base_level = (uint8_t)s->level, tr.view.nlevels = 1;
    tr.view.min_layer = (uint16_t)(s->layer0 + (uint32_t)s->z), tr.view.nlayers = (uint16_t)s->nz;
    VkrTarget tg;
    if (!vkr_frame_ops()->target(&tr, 0, vkr_frame_value(), &tg))
        stop("%s of part of a texture that cannot be a render target (NVN format %#x)", what, (unsigned)t->tex.d.format);
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = tg.view,
                                     .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {tg.width, tg.height}},
                          .layerCount = tg.layers};
    VkClearAttachment a = {0};
    if (colour) {
        ri.colorAttachmentCount = 1, ri.pColorAttachments = &att;
        a.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        memcpy(&a.clearValue.color, &p->c, sizeof a.clearValue.color);
    } else {
        if (tg.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ri.pDepthAttachment = &att;
        if (tg.aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ri.pStencilAttachment = &att;
        a.aspectMask = tg.aspect;
        a.clearValue.depthStencil = (VkClearDepthStencilValue){(tg.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? p->c.f[0] : 0.0f,
                                                               (tg.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? p->c.u[1] : p->c.u[0]};
    }
    VkClearRect cr = {{{s->x, s->y}, {s->w, s->h}}, 0, tg.layers};
    vkr_xfer_begin();
    vkrec_BeginRendering(&ri);
    vkrec_ClearAttachments(1, &a, 1, &cr);
    vkrec_EndRendering();
    vkr_xfer_end();
}

static void op_clear_tex(const p_clear_tex *p) {
    const char *what = "ClearTexture";
    if ((p->mask & 0xF) != 0xF) stop("%s with channel mask %#x: a masked clear needs a draw", what, p->mask);
    Sub s;
    sub_resolve(what, &p->tex, &p->r, &s);
    if (p->tex.has_view && p->tex.view.set_format && p->tex.view.format != p->tex.tex.d.format) {
        const VkfRow *vr = vkf_row(p->tex.view.format);
        if (!vr || vr->bpb != s.row->bpb || vr->vk != s.row->vk)
            stop("%s through a view of NVN format %#x over a texture of %#x: the clear value would be read in another format",
                 what, (unsigned)p->tex.view.format, (unsigned)p->tex.tex.d.format);
    }
    int colour = s.img.aspect == VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t lw = max1(s.img.width >> s.level), lh = max1(s.img.height >> s.level), ld = max1(s.img.depth >> s.level);
    int whole = s.x == 0 && s.y == 0 && s.w == lw && s.h == lh && (!s.is3d || (s.z == 0 && s.nz == ld));
    if (!whole) {
        clear_partial(what, &p->tex, &s, p, colour);
    } else {
        VkImageSubresourceRange rg = {image_aspects(&s), s.level, 1, s.is3d ? 0 : s.layer0 + (uint32_t)s.z,
                                      s.is3d ? 1 : s.nz};
        vkr_xfer_begin();
        if (colour) {
            VkClearColorValue cv;
            memcpy(&cv, &p->c, sizeof cv);
            vkrec_ClearColorImage(s.img.image, &cv, &rg);
        } else {
            int both = (s.img.all_aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && (s.img.all_aspects & VK_IMAGE_ASPECT_DEPTH_BIT);
            VkClearDepthStencilValue dv = {(s.img.all_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? p->c.f[0] : 0.0f,
                                           both ? p->c.u[1] : p->c.u[0]};
            vkrec_ClearDepthStencilImage(s.img.image, &dv, &rg);
        }
        vkr_xfer_end();
    }
    vkr_count_transfer((uint64_t)s.w * s.h * s.nz * s.row->bpb / (max1(s.row->bw) * max1(s.row->bh)));
}

int vkr_xfer_op(const op_hdr *r) {
    const void *a = r + 1;
    switch (r->op) {
    case OP_COPY_B2B: op_copy_b2b(a); return 1;
    case OP_CLEAR_BUF: op_clear_buf(a); return 1;
    case OP_UPDATE_UB: op_update_ub(a); return 1;
    case OP_COPY_B2T: op_copy_b2t(a, 1); return 1;
    case OP_COPY_T2B: op_copy_b2t(a, 0); return 1;
    case OP_COPY_T2T: op_copy_t2t(a); return 1;
    case OP_CLEAR_TEX: op_clear_tex(a); return 1;
    case OP_COPY_STRIDE:
    case OP_COPY_IMAGE_STRIDE: {
        int64_t v = ((const p_copy_stride *)a)->stride;
        if (v < 0) stop("%s of %lld", nvn_op_name(r->op), (long long)v);
        if (r->op == OP_COPY_STRIDE) g_row = v;
        else g_img = v;
        return 1;
    }
    default: return 0;
    }
}
