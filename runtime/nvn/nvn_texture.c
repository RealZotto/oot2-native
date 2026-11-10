// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Textures (architecture D10, D19): the texture builder and its storage queries, textures, texture views, and the CPU
// texel calls. A texture is stored in the game's NVNtexture (tex_t) and owns no memory: its storage is a range of a
// memory pool, its GPU address the pool's (nvn_pool.c) plus its offset. In Kokiri Forest the game builds about 105
// textures per frame, asks the storage size of 68 and initializes and finalizes about 58 (a count of its NVN2 calls).
//
// Storage queries answer what the console's driver would (nvn_tex.c): the game lays its pools out from them. A format
// no source sizes stops with its number instead of answering a size that would misplace everything after it.
//
// Every initialization takes a new serial, so that a texture the game copies by value keeps naming the same texture,
// and a texture initialized again at the same address names another one (D10): a back end keys what it keeps on the
// serial, never on the object's address. A texture view is a value of 16 bytes with no room for one; a copy of a view
// is the same view.
//
// Ported from oot-pc-recomp runtime/nvn_api_obj.c and nvn_ext.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp carries
// them at 9d8fa02, ported there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c and totk_nvn_ext.c
// at b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt). Changed here: a
// format no source sizes is a named stop, not a size of 0; Initialize refuses storage outside an initialised pool; a
// texture builder keeps its depth-stencil mode instead of stopping on the stencil plane, and a view's format may
// differ from its texture's when texels are written; the ZCull storage size is nxnative's (dm/gfx/nvn/hle_nvn.c,
// zcull_size, at 865cc35, MIT, licenses/nxnative-MIT.txt) instead of 0; the live-texture table, the traces and the
// blocking markers are gone.
#include <string.h>

#include "nvn_api.h"
#include "oot2/nvn_tex.h"

typedef unsigned long long ull;

static uint64_t g_serial;
static uint64_t new_serial(void) { return __atomic_add_fetch(&g_serial, 1, __ATOMIC_RELAXED); }

// The storage of description d, or the named stop for a format no source sizes.
static void layout(Cpu *c, const char *fn, const nvn_texdesc *d, nvn_texlayout *L) {
    nvn_tex_layout(d, L);
    if (!L->bpb)
        rt_unimplemented(c, "nvn", "%s: format 0x%x, which no source sizes (target %d, %ux%ux%u, %u levels)", fn,
                         d->format, d->target, d->width, d->height, d->depth, d->levels);
}

// ---- the texture builder
#define TB(c) ((tex_builder_t *)arg_p(c, 0))
#define CTB(c) ((const tex_builder_t *)arg_p(c, 0))

// void nvnTextureBuilderSetDefaults(NVNtextureBuilder *b): keeps the device; a 2D 1x1x1 texture of one level, the
// identity swizzle, no storage
NVN_API(nvnTextureBuilderSetDefaults) {
    tex_builder_t *b = TB(c);
    NVNdevice *dev = b->dev;
    memset(b, 0, sizeof *b);
    b->dev = dev;
    b->d.swizzle_r = NVN_TEXTURE_SWIZZLE_R, b->d.swizzle_g = NVN_TEXTURE_SWIZZLE_G;
    b->d.swizzle_b = NVN_TEXTURE_SWIZZLE_B, b->d.swizzle_a = NVN_TEXTURE_SWIZZLE_A;
    b->d.target = NVN_TEXTURE_TARGET_2D;
    b->d.width = b->d.height = b->d.depth = b->d.levels = 1;
}
NVN_API(nvnTextureBuilderSetDevice) { TB(c)->dev = arg_p(c, 1); }
NVN_API(nvnTextureBuilderSetFlags) { TB(c)->d.flags = (int)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetTarget) { TB(c)->d.target = (int)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetWidth) { TB(c)->d.width = (uint32_t)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetHeight) { TB(c)->d.height = (uint32_t)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetDepth) { TB(c)->d.depth = (uint32_t)arg_x(c, 1); }
// void nvnTextureBuilderSetSize1D(b, int w), ...SetSize2D(b, w, h), ...SetSize3D(b, w, h, d)
NVN_API(nvnTextureBuilderSetSize1D) {
    nvn_texdesc *d = &TB(c)->d;
    d->width = (uint32_t)arg_x(c, 1), d->height = d->depth = 1;
}
NVN_API(nvnTextureBuilderSetSize2D) {
    nvn_texdesc *d = &TB(c)->d;
    d->width = (uint32_t)arg_x(c, 1), d->height = (uint32_t)arg_x(c, 2), d->depth = 1;
}
NVN_API(nvnTextureBuilderSetSize3D) {
    nvn_texdesc *d = &TB(c)->d;
    d->width = (uint32_t)arg_x(c, 1), d->height = (uint32_t)arg_x(c, 2), d->depth = (uint32_t)arg_x(c, 3);
}
NVN_API(nvnTextureBuilderSetLevels) { TB(c)->d.levels = (uint32_t)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetFormat) { TB(c)->d.format = (int)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetSamples) { TB(c)->d.samples = (int)arg_x(c, 1); }
NVN_API(nvnTextureBuilderSetStride) { TB(c)->d.stride = (uint32_t)arg_x(c, 1); }
// void nvnTextureBuilderSetSwizzle(NVNtextureBuilder *b, NVNtextureSwizzle r, g, b, a)
NVN_API(nvnTextureBuilderSetSwizzle) {
    uint32_t s[4] = {(uint32_t)arg_x(c, 1), (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 3), (uint32_t)arg_x(c, 4)};
    for (int i = 0; i < 4; i++)
        if (s[i] > NVN_TEXTURE_SWIZZLE_A) rt_fatal("nvn: nvnTextureBuilderSetSwizzle: invalid swizzle %u", s[i]);
    nvn_texdesc *d = &TB(c)->d;
    d->swizzle_r = s[0], d->swizzle_g = s[1], d->swizzle_b = s[2], d->swizzle_a = s[3];
}
// void nvnTextureBuilderSetDepthStencilMode(NVNtextureBuilder *b, NVNtextureDepthStencilMode m): which plane of a
// depth-stencil format the texture samples
NVN_API(nvnTextureBuilderSetDepthStencilMode) {
    uint32_t m = (uint32_t)arg_x(c, 1);
    if (m > NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL)
        rt_fatal("nvn: nvnTextureBuilderSetDepthStencilMode: invalid mode %u", m);
    TB(c)->d.dsmode = m;
}
// void nvnTextureBuilderSetStorage(NVNtextureBuilder *b, NVNmemoryPool *p, ptrdiff_t offset)
NVN_API(nvnTextureBuilderSetStorage) {
    tex_builder_t *b = TB(c);
    b->pool = arg_p(c, 1);
    b->off = (int64_t)arg_x(c, 2);
}
// void nvnTextureBuilderSetPackagedTextureLayout(NVNtextureBuilder *b, const NVNpackagedTextureLayout *l): 8 bytes
// of layout parameters from the game's texture files; byte 0 sets the block dimensions (nvn_tex.c); NULL clears them
NVN_API(nvnTextureBuilderSetPackagedTextureLayout) {
    tex_builder_t *b = TB(c);
    const NVNpackagedTextureLayout *l = arg_p(c, 1);
    b->has_layout = l != NULL;
    memset(b->layout, 0, sizeof b->layout);
    if (l) memcpy(b->layout, l->layout, sizeof b->layout);
    b->d.pkg_block_h = l ? l->layout[0] + 1 : 0;
}
// void nvnTextureBuilderSetPackagedTextureData(NVNtextureBuilder *b, const void *data): texels already in the
// console's layout, copied into the texture's storage at Initialize
NVN_API(nvnTextureBuilderSetPackagedTextureData) { TB(c)->pkg_data = arg_p(c, 1); }

// The storage queries. size_t nvnTextureBuilderGetStorageSize(const NVNtextureBuilder *b), ...GetStorageAlignment
NVN_API(nvnTextureBuilderGetStorageSize) {
    nvn_texlayout L;
    layout(c, "nvnTextureBuilderGetStorageSize", &CTB(c)->d, &L);
    ret_x(c, L.size);
}
NVN_API(nvnTextureBuilderGetStorageAlignment) {
    nvn_texlayout L;
    layout(c, "nvnTextureBuilderGetStorageAlignment", &CTB(c)->d, &L);
    ret_x(c, L.align);
}
// NVNstorageClass: the page kind a virtual pool maps for the texture (NVNmappingRequest.storageClass), which this
// runtime's pools do not distinguish. oot-pc-recomp answers 1, 0 for a buffer texture; nxnative answers 0; neither
// value is the console's (the texture and sampler study).
static uint64_t storage_class(const nvn_texdesc *d) { return d->target == NVN_TEXTURE_TARGET_BUFFER ? 0 : 1; }
NVN_API(nvnTextureBuilderGetStorageClass) { ret_x(c, storage_class(&CTB(c)->d)); }
// The size of a depth texture's ZCull save area, which the game allocates from its own pools. The console's formula
// is not known and NVN never answers 0; nxnative's grows with the texture: 8 bytes per 32x32 tile and layer, rounded
// up to 256 bytes, at least 256.
static uint64_t zcull_size(const nvn_texdesc *d) {
    uint64_t tiles = (uint64_t)((d->width + 31) / 32) * ((d->height + 31) / 32) * (d->depth ? d->depth : 1);
    return tiles ? (tiles * 8 + 0xff) & ~0xffull : 0x100;
}
NVN_API(nvnTextureBuilderGetZCullStorageSize) { ret_x(c, zcull_size(&CTB(c)->d)); }

// The builder's getters
static void put_swizzle(const Cpu *c, const nvn_texdesc *d) {
    *(int32_t *)arg_p(c, 1) = d->swizzle_r, *(int32_t *)arg_p(c, 2) = d->swizzle_g;
    *(int32_t *)arg_p(c, 3) = d->swizzle_b, *(int32_t *)arg_p(c, 4) = d->swizzle_a;
}
NVN_API(nvnTextureBuilderGetDevice) { ret_x(c, (uintptr_t)CTB(c)->dev); }
NVN_API(nvnTextureBuilderGetFlags) { ret_x(c, (uint32_t)CTB(c)->d.flags); }
NVN_API(nvnTextureBuilderGetTarget) { ret_x(c, (uint32_t)CTB(c)->d.target); }
NVN_API(nvnTextureBuilderGetWidth) { ret_x(c, CTB(c)->d.width); }
NVN_API(nvnTextureBuilderGetHeight) { ret_x(c, CTB(c)->d.height); }
NVN_API(nvnTextureBuilderGetDepth) { ret_x(c, CTB(c)->d.depth); }
NVN_API(nvnTextureBuilderGetLevels) { ret_x(c, CTB(c)->d.levels); }
NVN_API(nvnTextureBuilderGetFormat) { ret_x(c, (uint32_t)CTB(c)->d.format); }
NVN_API(nvnTextureBuilderGetSamples) { ret_x(c, (uint32_t)CTB(c)->d.samples); }
NVN_API(nvnTextureBuilderGetSwizzle) { put_swizzle(c, &CTB(c)->d); }
NVN_API(nvnTextureBuilderGetDepthStencilMode) { ret_x(c, CTB(c)->d.dsmode); }
NVN_API(nvnTextureBuilderGetStride) { ret_x(c, CTB(c)->d.stride); }
NVN_API(nvnTextureBuilderGetMemoryPool) { ret_x(c, (uintptr_t)CTB(c)->pool); }
NVN_API(nvnTextureBuilderGetMemoryOffset) { ret_x(c, (uint64_t)CTB(c)->off); }
NVN_API(nvnTextureBuilderGetPackagedTextureData) { ret_x(c, (uintptr_t)CTB(c)->pkg_data); }

// The offset of a view inside storage laid out as d: its base level's, then its first layer's.
static uint64_t view_offset(const nvn_texdesc *d, const nvn_texlayout *L, const view_t *v) {
    if (!v) return 0;
    uint64_t off = v->set_levels ? nvn_tex_level_offset(d, v->base_level) : 0;
    if (v->set_layers) off += (uint64_t)v->min_layer * L->layer_size;
    return off;
}
uint64_t nvn_tex_view_offset(const tex_t *t, const view_t *v) { return view_offset(&t->d, &t->L, v); }
// ptrdiff_t nvnTextureBuilderGetViewOffset(const NVNtextureBuilder *b, const NVNtextureView *v)
NVN_API(nvnTextureBuilderGetViewOffset) {
    nvn_texlayout L;
    layout(c, "nvnTextureBuilderGetViewOffset", &CTB(c)->d, &L);
    ret_x(c, view_offset(&CTB(c)->d, &L, arg_p(c, 1)));
}

// ---- textures
static uint64_t g_textures_live;   // textures with a serial now
uint64_t nvn_textures_live(void) { return __atomic_load_n(&g_textures_live, __ATOMIC_RELAXED); }

static void release(tex_t *o) {
    if (o->magic != NVN_TEX_MAGIC || !o->serial) return;
    if (texheap_on()) texheap_texture_end(o);
    gfx_backend()->texture_release(o);
    __atomic_fetch_sub(&g_textures_live, 1, __ATOMIC_RELAXED);
}

// NVNboolean nvnTextureInitialize(NVNtexture *t, const NVNtextureBuilder *b): false for storage outside an initialised
// pool. Packaged data is copied into the storage, which the CPU must then see.
NVN_API(nvnTextureInitialize) {
    tex_t *o = arg_p(c, 0);
    const tex_builder_t *s = arg_p(c, 1);
    nvn_texlayout L;
    layout(c, "nvnTextureInitialize", &s->d, &L);
    const pool_t *p = s->pool;
    if (!p || p->magic != NVN_POOL_MAGIC || s->off < 0 || (uint64_t)s->off > p->size ||
        L.size > p->size - (uint64_t)s->off) {
        ret_x(c, 0);
        return;
    }
    release(o);   // initialized again without Finalize: the old serial goes
    memset(o, 0, sizeof *o);
    o->magic = NVN_TEX_MAGIC;
    o->d = s->d;
    o->has_layout = s->has_layout;
    memcpy(o->layout, s->layout, sizeof o->layout);
    o->L = L;
    o->pool = s->pool;
    o->off = s->off;
    o->mem = p->mem + s->off;
    if (s->pkg_data) {
        if (gfx_gpu_only((uintptr_t)o->mem))
            rt_unimplemented(c, "nvn",
                             "nvnTextureInitialize: packaged texture data for storage the CPU cannot see (pool %p, "
                             "offset %#llx, %#llx bytes)",
                             (const void *)p, (ull)s->off, (ull)L.size);
        memcpy(o->mem, s->pkg_data, (size_t)L.size);
        gfx_backend()->guest_written((uintptr_t)o->mem, L.size);
    }
    o->serial = new_serial();
    __atomic_fetch_add(&g_textures_live, 1, __ATOMIC_RELAXED);
    if (texheap_on()) texheap_texture_init(o, s->pkg_data != NULL);
    ret_x(c, 1);
}
// void nvnTextureFinalize(NVNtexture *t)
NVN_API(nvnTextureFinalize) {
    tex_t *o = arg_p(c, 0);
    release(o);
    o->magic = 0;
}
// void nvnTextureSetDebugLabel(NVNtexture *t, const char *label): for NVN's debugging tools only
NVN_API(nvnTextureSetDebugLabel) { (void)c; }

#define CT(c) ((const tex_t *)arg_p(c, 0))
NVN_API(nvnTextureGetFlags) { ret_x(c, (uint32_t)CT(c)->d.flags); }
NVN_API(nvnTextureGetTarget) { ret_x(c, (uint32_t)CT(c)->d.target); }
NVN_API(nvnTextureGetWidth) { ret_x(c, CT(c)->d.width); }
NVN_API(nvnTextureGetHeight) { ret_x(c, CT(c)->d.height); }
NVN_API(nvnTextureGetDepth) { ret_x(c, CT(c)->d.depth); }
NVN_API(nvnTextureGetLevels) { ret_x(c, CT(c)->d.levels); }
NVN_API(nvnTextureGetFormat) { ret_x(c, (uint32_t)CT(c)->d.format); }
NVN_API(nvnTextureGetSamples) { ret_x(c, (uint32_t)CT(c)->d.samples); }
NVN_API(nvnTextureGetSwizzle) { put_swizzle(c, &CT(c)->d); }
NVN_API(nvnTextureGetDepthStencilMode) { ret_x(c, CT(c)->d.dsmode); }
NVN_API(nvnTextureGetStride) { ret_x(c, CT(c)->d.stride); }
NVN_API(nvnTextureGetStorageSize) { ret_x(c, CT(c)->L.size); }
NVN_API(nvnTextureGetStorageClass) { ret_x(c, storage_class(&CT(c)->d)); }
NVN_API(nvnTextureGetZCullStorageSize) { ret_x(c, zcull_size(&CT(c)->d)); }
NVN_API(nvnTextureGetMemoryPool) { ret_x(c, (uintptr_t)CT(c)->pool); }
NVN_API(nvnTextureGetMemoryOffset) { ret_x(c, (uint64_t)CT(c)->off); }
// NVNtextureAddress nvnTextureGetTextureAddress(const NVNtexture *t): the GPU address of its storage
NVN_API(nvnTextureGetTextureAddress) { ret_x(c, (uintptr_t)CT(c)->mem); }
// uint64_t nvnTextureGetDebugID(const NVNtexture *t): the serial, unique per initialization
NVN_API(nvnTextureGetDebugID) { ret_x(c, CT(c)->serial); }
// ptrdiff_t nvnTextureGetViewOffset(const NVNtexture *t, const NVNtextureView *v)
NVN_API(nvnTextureGetViewOffset) { ret_x(c, nvn_tex_view_offset(CT(c), arg_p(c, 1))); }
// NVNboolean nvnTextureCompare(const NVNtexture *a, const NVNtexture *b): the same description on the same storage
NVN_API(nvnTextureCompare) {
    const tex_t *x = arg_p(c, 0), *y = arg_p(c, 1);
    ret_x(c, !memcmp(&x->d, &y->d, sizeof x->d) && x->mem == y->mem);
}

// ---- texture views: values the game keeps on its stack, 16 bytes (view_t)
#define VW(c) ((view_t *)arg_p(c, 0))
#define CVW(c) ((const view_t *)arg_p(c, 0))
NVN_API(nvnTextureViewSetDefaults) { memset(arg_p(c, 0), 0, sizeof(view_t)); }
// void nvnTextureViewSetLevels(NVNtextureView *v, int base, int n)
NVN_API(nvnTextureViewSetLevels) {
    uint32_t base = (uint32_t)arg_x(c, 1), n = (uint32_t)arg_x(c, 2);
    if (base > 255 || n > 255) rt_fatal("nvn: nvnTextureViewSetLevels(%d, %d)", (int)base, (int)n);
    view_t *v = VW(c);
    v->set_levels = 1, v->base_level = (uint8_t)base, v->nlevels = (uint8_t)n;
}
// void nvnTextureViewSetLayers(NVNtextureView *v, int first, int n)
NVN_API(nvnTextureViewSetLayers) {
    uint32_t first = (uint32_t)arg_x(c, 1), n = (uint32_t)arg_x(c, 2);
    if (first > 65535 || n > 65535) rt_fatal("nvn: nvnTextureViewSetLayers(%d, %d)", (int)first, (int)n);
    view_t *v = VW(c);
    v->set_layers = 1, v->min_layer = (uint16_t)first, v->nlayers = (uint16_t)n;
}
NVN_API(nvnTextureViewSetFormat) {
    uint32_t f = (uint32_t)arg_x(c, 1);
    if (f > 65535) rt_fatal("nvn: nvnTextureViewSetFormat(%d)", (int)f);
    view_t *v = VW(c);
    v->set_format = 1, v->format = (uint16_t)f;
}
NVN_API(nvnTextureViewSetTarget) {
    uint32_t t = (uint32_t)arg_x(c, 1);
    if (t > 255) rt_fatal("nvn: nvnTextureViewSetTarget(%d)", (int)t);
    view_t *v = VW(c);
    v->set_target = 1, v->target = (uint8_t)t;
}
// void nvnTextureViewSetSwizzle(NVNtextureView *v, NVNtextureSwizzle r, g, b, a)
NVN_API(nvnTextureViewSetSwizzle) {
    view_t *v = VW(c);
    for (int i = 0; i < 4; i++) {
        uint32_t s = (uint32_t)arg_x(c, 1 + i);
        if (s > NVN_TEXTURE_SWIZZLE_A) rt_fatal("nvn: nvnTextureViewSetSwizzle: invalid swizzle %u", s);
        v->swz[i] = (uint8_t)s;
    }
    v->set_swizzle = 1;
}
NVN_API(nvnTextureViewSetDepthStencilMode) {
    uint32_t m = (uint32_t)arg_x(c, 1);
    if (m > NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL)
        rt_fatal("nvn: nvnTextureViewSetDepthStencilMode: invalid mode %u", m);
    view_t *v = VW(c);
    v->set_dsmode = 1, v->dsmode = (uint8_t)m;
}
// NVNboolean nvnTextureViewGet{Levels,Layers}(const NVNtextureView *v, int *a, int *b), ...GetFormat(v, NVNformat *),
// ...GetTarget, ...GetSwizzle(v, r, g, b, a), ...GetDepthStencilMode: true when the view sets it
NVN_API(nvnTextureViewGetLevels) {
    const view_t *v = CVW(c);
    *(int32_t *)arg_p(c, 1) = v->base_level, *(int32_t *)arg_p(c, 2) = v->nlevels;
    ret_x(c, v->set_levels);
}
NVN_API(nvnTextureViewGetLayers) {
    const view_t *v = CVW(c);
    *(int32_t *)arg_p(c, 1) = v->min_layer, *(int32_t *)arg_p(c, 2) = v->nlayers;
    ret_x(c, v->set_layers);
}
NVN_API(nvnTextureViewGetFormat) {
    *(int32_t *)arg_p(c, 1) = CVW(c)->format;
    ret_x(c, CVW(c)->set_format);
}
NVN_API(nvnTextureViewGetTarget) {
    *(int32_t *)arg_p(c, 1) = CVW(c)->target;
    ret_x(c, CVW(c)->set_target);
}
NVN_API(nvnTextureViewGetSwizzle) {
    for (int i = 0; i < 4; i++) *(int32_t *)arg_p(c, 1 + i) = CVW(c)->swz[i];
    ret_x(c, CVW(c)->set_swizzle);
}
NVN_API(nvnTextureViewGetDepthStencilMode) {
    *(int32_t *)arg_p(c, 1) = CVW(c)->dsmode;
    ret_x(c, CVW(c)->set_dsmode);
}
// NVNboolean nvnTextureViewCompare(const NVNtextureView *a, const NVNtextureView *b): the same settings, comparing the
// values of only those a view sets
NVN_API(nvnTextureViewCompare) {
    const view_t *x = arg_p(c, 0), *y = arg_p(c, 1);
    int same = x->set_levels == y->set_levels && x->set_layers == y->set_layers && x->set_format == y->set_format &&
               x->set_target == y->set_target && x->set_swizzle == y->set_swizzle && x->set_dsmode == y->set_dsmode;
    if (same && x->set_levels) same = x->base_level == y->base_level && x->nlevels == y->nlevels;
    if (same && x->set_layers) same = x->min_layer == y->min_layer && x->nlayers == y->nlayers;
    if (same && x->set_format) same = x->format == y->format;
    if (same && x->set_target) same = x->target == y->target;
    if (same && x->set_swizzle) same = !memcmp(x->swz, y->swz, sizeof x->swz);
    if (same && x->set_dsmode) same = x->dsmode == y->dsmode;
    ret_x(c, same);
}

// ---- texels through the CPU: the back end reads or writes the view's level and layers
// The bytes per row and per 2D image of a copy region in host memory; 0 asks for tightly packed.
static void texel_strides(const tex_t *o, const NVNcopyRegion *r, uint64_t *row, uint64_t *img, const char *fn) {
    int bw, bh, bpb = nvn_fmt_bpb(o->d.format, &bw, &bh);
    if (o->magic != NVN_TEX_MAGIC || !bpb) rt_fatal("nvn: %s of a texture that is not initialised", fn);
    if (r->width < 0 || r->height < 0 || r->depth < 0)
        rt_fatal("nvn: %s: region %dx%dx%d", fn, r->width, r->height, r->depth);
    uint64_t wb = (uint64_t)(((uint32_t)r->width + bw - 1) / bw) * bpb, hb = ((uint32_t)r->height + bh - 1) / bh;
    if (!*row) *row = wb;
    if (!*img) *img = *row * hb;
    if (*row < wb || *img < *row * hb) rt_fatal("nvn: %s: a stride smaller than the region", fn);
}
// void nvnTextureReadTexels[Strided](const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, void *dst
// [, ptrdiff_t row, ptrdiff_t img])
static void read_texels(Cpu *c, int strided, const char *fn) {
    const tex_t *o = arg_p(c, 0);
    const NVNcopyRegion *r = arg_p(c, 2);
    uint64_t row = strided ? arg_x(c, 4) : 0, img = strided ? arg_x(c, 5) : 0;
    texel_strides(o, r, &row, &img, fn);
    if (texheap_on()) texheap_read_texels();
    gfx_backend()->read_texels(o, arg_p(c, 1), r, arg_p(c, 3), row, img);
}
// void nvnTextureWriteTexels[Strided](const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const
// void *src[, ptrdiff_t row, ptrdiff_t img])
static void write_texels(Cpu *c, int strided, const char *fn) {
    const tex_t *o = arg_p(c, 0);
    const NVNcopyRegion *r = arg_p(c, 2);
    uint64_t row = strided ? arg_x(c, 4) : 0, img = strided ? arg_x(c, 5) : 0;
    texel_strides(o, r, &row, &img, fn);
    gfx_backend()->write_texels(o, arg_p(c, 1), r, arg_p(c, 3), row, img);
}
NVN_API(nvnTextureReadTexels) { read_texels(c, 0, "nvnTextureReadTexels"); }
NVN_API(nvnTextureReadTexelsStrided) { read_texels(c, 1, "nvnTextureReadTexelsStrided"); }
NVN_API(nvnTextureWriteTexels) { write_texels(c, 0, "nvnTextureWriteTexels"); }
NVN_API(nvnTextureWriteTexelsStrided) { write_texels(c, 1, "nvnTextureWriteTexelsStrided"); }
// void nvnTexture{Flush,Invalidate}Texels(const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r): the
// back end's texel calls are done when they return, so there is nothing left to flush or to invalidate
NVN_API(nvnTextureFlushTexels) { (void)c; }
NVN_API(nvnTextureInvalidateTexels) { (void)c; }
