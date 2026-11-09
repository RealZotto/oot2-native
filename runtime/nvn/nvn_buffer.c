// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Buffers (architecture D10): a range of a memory pool, stored in the game's NVNbuffer and owning no storage. In Kokiri
// Forest the game creates and frees about 1,800 buffers per frame and asks for about 1,765 buffer addresses (a count of
// its NVN2 calls), so builder, initialize, finalize, GetAddress and Map are a few loads and stores each: no allocation,
// no lock, no call into the back end. A buffer's GPU address is its pool's (nvn_pool.c) plus its offset, read when
// asked, so a buffer follows its pool's range.
//
// Ported from oot-pc-recomp runtime/nvn_api_core.c and nvn_ext.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp carries
// them at 9d8fa02, ported there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c and totk_nvn_ext.c
// at b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt). Changed here:
// InvalidateMappedRange tells the back end (it was a no-op), and an invalid flush or invalidation range is a fatal for
// both; the trace and statistics counters are gone.
#include <string.h>

#include "nvn_api.h"

#define BUF_MAGIC MAGIC('B', 'U', 'F', 0)

typedef unsigned long long ull;

// void nvnBufferBuilderSetDevice(NVNbufferBuilder *b, NVNdevice *d)
NVN_API(nvnBufferBuilderSetDevice) { ((buf_builder_t *)arg_p(c, 0))->dev = arg_p(c, 1); }
// void nvnBufferBuilderSetDefaults(NVNbufferBuilder *b): keeps the device
NVN_API(nvnBufferBuilderSetDefaults) {
    buf_builder_t *b = arg_p(c, 0);
    *b = (buf_builder_t){b->dev, NULL, 0, 0};
}
// void nvnBufferBuilderSetStorage(NVNbufferBuilder *b, NVNmemoryPool *p, ptrdiff_t offset, size_t size)
NVN_API(nvnBufferBuilderSetStorage) {
    buf_builder_t *b = arg_p(c, 0);
    b->pool = arg_p(c, 1);
    b->off = (int64_t)arg_x(c, 2);
    b->size = arg_x(c, 3);
}
// NVNdevice *nvnBufferBuilderGetDevice(const NVNbufferBuilder *b), NVNmemoryPool *...GetMemoryPool, ptrdiff_t
// ...GetMemoryOffset, size_t ...GetSize
NVN_API(nvnBufferBuilderGetDevice) { ret_x(c, (uintptr_t)((const buf_builder_t *)arg_p(c, 0))->dev); }
NVN_API(nvnBufferBuilderGetMemoryPool) { ret_x(c, (uintptr_t)((const buf_builder_t *)arg_p(c, 0))->pool); }
NVN_API(nvnBufferBuilderGetMemoryOffset) { ret_x(c, (uint64_t)((const buf_builder_t *)arg_p(c, 0))->off); }
NVN_API(nvnBufferBuilderGetSize) { ret_x(c, ((const buf_builder_t *)arg_p(c, 0))->size); }

// NVNboolean nvnBufferInitialize(NVNbuffer *b, const NVNbufferBuilder *builder): false for a range outside an
// initialised pool
NVN_API(nvnBufferInitialize) {
    buf_t *o = arg_p(c, 0);
    const buf_builder_t *s = arg_p(c, 1);
    const pool_t *p = s->pool;
    if (!p || p->magic != NVN_POOL_MAGIC || s->off < 0 || (uint64_t)s->off > p->size ||
        s->size > p->size - (uint64_t)s->off) {
        ret_x(c, 0);
        return;
    }
    *o = (buf_t){BUF_MAGIC, s->pool, s->off, s->size};
    if (texheap_on() && !gfx_gpu_only((uintptr_t)p->mem))
        texheap_range(TEXHEAP_OBJECT, (uintptr_t)p->mem + (uint64_t)s->off, s->size);
    ret_x(c, 1);
}
// void nvnBufferFinalize(NVNbuffer *b)
NVN_API(nvnBufferFinalize) {
    buf_t *o = arg_p(c, 0);
    if (texheap_on() && o->magic == BUF_MAGIC && o->pool && !gfx_gpu_only((uintptr_t)o->pool->mem))
        texheap_range(TEXHEAP_OBJECT_END, (uintptr_t)o->pool->mem + (uint64_t)o->off, o->size);
    *o = (buf_t){0};
}
// void nvnBufferSetDebugLabel(NVNbuffer *b, const char *label): for NVN's debugging tools only
NVN_API(nvnBufferSetDebugLabel) { (void)c; }

static uint64_t address(const buf_t *o) { return o->pool ? (uintptr_t)o->pool->mem + (uint64_t)o->off : 0; }

// NVNbufferAddress nvnBufferGetAddress(const NVNbuffer *b)
NVN_API(nvnBufferGetAddress) { ret_x(c, address(arg_p(c, 0))); }
// void *nvnBufferMap(const NVNbuffer *b): NULL in a pool the CPU cannot see, whose address is in the GPU window
NVN_API(nvnBufferMap) {
    uint64_t a = address(arg_p(c, 0));
    ret_x(c, gfx_gpu_only(a) ? 0 : a);
}
// NVNmemoryPool *nvnBufferGetMemoryPool(const NVNbuffer *b), ptrdiff_t ...GetMemoryOffset, size_t ...GetSize
NVN_API(nvnBufferGetMemoryPool) { ret_x(c, (uintptr_t)((const buf_t *)arg_p(c, 0))->pool); }
NVN_API(nvnBufferGetMemoryOffset) { ret_x(c, (uint64_t)((const buf_t *)arg_p(c, 0))->off); }
NVN_API(nvnBufferGetSize) { ret_x(c, ((const buf_t *)arg_p(c, 0))->size); }

// The guest range [offset, offset + size) of a mapped buffer, for a flush or an invalidation; 0 in a pool the CPU
// cannot see. A range outside the buffer is a fatal: the game asks NVN for something it defines as an error.
static uint64_t mapped_range(Cpu *c, const char *what) {
    const buf_t *o = arg_p(c, 0);
    int64_t off = (int64_t)arg_x(c, 1);
    uint64_t size = arg_x(c, 2);
    if (off < 0 || (uint64_t)off > o->size || size > o->size - (uint64_t)off)
        rt_fatal("nvn: nvnBuffer%sMappedRange %#llx+%#llx outside the buffer (%#llx bytes)", what, (ull)off, (ull)size,
                 (ull)o->size);
    uint64_t a = address(o);
    return a && size && !gfx_gpu_only(a) ? a + (uint64_t)off : 0;
}

// void nvnBufferFlushMappedRange(const NVNbuffer *b, ptrdiff_t offset, size_t size): the CPU wrote the range
NVN_API(nvnBufferFlushMappedRange) {
    uint64_t a = mapped_range(c, "Flush");
    if (a) gfx_backend()->guest_written(a, arg_x(c, 2));
}
// void nvnBufferInvalidateMappedRange(const NVNbuffer *b, ptrdiff_t offset, size_t size): the CPU is about to read
// what the GPU wrote there
NVN_API(nvnBufferInvalidateMappedRange) {
    uint64_t a = mapped_range(c, "Invalidate");
    if (a) nvn_invalidate(c, a, arg_x(c, 2), "BufferInvalidateMappedRange");
}
