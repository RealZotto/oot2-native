// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Memory pools (architecture D10, D14): the GPU address window, the pool index and the MemoryPool entry points.
//
// A pool's GPU address (pool_t.mem) is its guest address when the CPU can see it. A pool the CPU cannot see
// (CPU_NO_ACCESS, physical, virtual or neither) gets a range of the GPU address window (gfx.h) from an interval
// allocator: a sorted list of free spans, first fit, coalesced on free, with no cap on the number of ranges. A virtual
// pool has no storage: its range is reserved at creation, and MemoryPoolMapVirtual points its pages at pages of
// physical pools. Every initialised pool the GPU can reach is in an index sorted by GPU address, which turns an address
// into its pool (nvn_pool_find); lookups share a reader-writer lock, and a pool's creation, end and mapping take it
// alone, outside every call into the back end.
//
// The descriptor path binds a buffer as one contiguous range of backing memory. A virtual pool whose mapped pages lie
// in one physical run (one physical pool, one offset between virtual and physical pages) is therefore exact, and a
// mapping that would leave them in two runs is the named fatal that prints its shape (D14), until a design for it
// exists. Every MapVirtual call logs its shape (requests, offsets, sizes, physical runs), as D14 asks of the first
// census.
//
// Ported from oot-pc-recomp runtime/nvn_api_core.c and nvn_ext.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp carries
// them at 9d8fa02, ported there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c and totk_nvn_ext.c
// at b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt): the MemoryPool entry
// points, the checks of a pool's flags and of a mapping request, the index (nvn_pool_index.h). Changed here: every pool
// the CPU cannot see takes a window range, not only physical and virtual ones; the window's first-fit table, capped
// at 1,024 ranges, is a free list with no cap, and a full window is a fatal instead of a refused pool; a virtual pool
// keeps its mapping state, and a mapping that splits its pages into two physical runs stops with its shape; flushes
// and invalidations reach the back end directly; the trace and statistics counters are gone.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_api.h"
#include "nvn_pool_index.h"
#include "plat.h"

_Static_assert(GFX_GPU_WINDOW_BASE >= RT_ADDRESS_LIMIT, "the GPU address window lies above the guest address space");
_Static_assert(GFX_GPU_WINDOW_BASE % GFX_GPU_PAGE == 0 && GFX_GPU_WINDOW_SIZE % GFX_GPU_PAGE == 0, "window pages");

#define CPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS
#define GPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS
#define PHYSICAL NVN_MEMORY_POOL_FLAGS_PHYSICAL
#define VIRTUAL NVN_MEMORY_POOL_FLAGS_VIRTUAL

typedef unsigned long long ull;

// One lock for the index, the window and the virtual pools' mapping state.
static PlatLock g_lock = PLAT_LOCK_INIT;
static pidx g_index;
static NvnPoolStats g_stats;   // what the pools and the window held, and the most they held at once (g_lock)

// ---- the GPU address window: free spans, sorted by address, disjoint and never adjacent
typedef struct { uint64_t lo, hi; } Span;
static Span *g_free;
static int g_nfree, g_free_cap, g_window_ready;

static void free_insert(int at, Span s) {
    if (g_nfree == g_free_cap) {
        int cap = g_free_cap ? 2 * g_free_cap : 64;
        Span *f = realloc(g_free, (size_t)cap * sizeof *f);
        if (!f) rt_fatal("nvn: out of memory for the GPU address window's free list");
        g_free = f, g_free_cap = cap;
    }
    memmove(g_free + at + 1, g_free + at, (size_t)(g_nfree - at) * sizeof *g_free);
    g_free[at] = s;
    g_nfree++;
}

static void free_remove(int at) {
    memmove(g_free + at, g_free + at + 1, (size_t)(g_nfree - at - 1) * sizeof *g_free);
    g_nfree--;
}

// The lowest free range of size bytes (a multiple of GFX_GPU_PAGE), or 0 when none is left. Under g_lock.
static uint64_t window_take(uint64_t size) {
    if (!g_window_ready) {
        free_insert(0, (Span){GFX_GPU_WINDOW_BASE, GFX_GPU_WINDOW_BASE + GFX_GPU_WINDOW_SIZE});
        g_window_ready = 1;
    }
    for (int i = 0; i < g_nfree; i++) {
        if (g_free[i].hi - g_free[i].lo < size) continue;
        uint64_t at = g_free[i].lo;
        g_free[i].lo += size;
        if (g_free[i].lo == g_free[i].hi) free_remove(i);
        g_stats.window_bytes += size;
        if (g_stats.window_bytes > g_stats.window_peak) g_stats.window_peak = g_stats.window_bytes;
        return at;
    }
    return 0;
}

// Give [lo, lo + size) back, merged with the free spans it touches. Under g_lock.
static void window_give(uint64_t lo, uint64_t size) {
    uint64_t hi = lo + size;
    g_stats.window_bytes -= size < g_stats.window_bytes ? size : g_stats.window_bytes;
    int a = 0, b = g_nfree;   // at = the first free span that starts above lo
    while (a < b) {
        int m = (a + b) / 2;
        if (g_free[m].lo <= lo) a = m + 1;
        else b = m;
    }
    int at = a;
    if ((at > 0 && g_free[at - 1].hi > lo) || (at < g_nfree && g_free[at].lo < hi))
        rt_fatal("nvn: GPU window range %#llx+%#llx given back while part of it is free", (ull)lo, (ull)size);
    int prev = at > 0 && g_free[at - 1].hi == lo, next = at < g_nfree && g_free[at].lo == hi;
    if (prev && next) {
        g_free[at - 1].hi = g_free[at].hi;
        free_remove(at);
    } else if (prev) {
        g_free[at - 1].hi = hi;
    } else if (next) {
        g_free[at].lo = lo;
    } else {
        free_insert(at, (Span){lo, hi});
    }
}

// ---- a virtual pool's mapping state: which of its pages are mapped, and the one physical run they lie in
struct nvn_vmap {
    const pool_t *phys;   // the physical pool of the run, NULL while no page is mapped
    int64_t delta;        // physical offset minus virtual offset, the same for every mapped page
    uint64_t pages, mapped;
    uint64_t bits[];      // one per page, set while mapped
};

static uint64_t bits_count(const struct nvn_vmap *m, uint64_t first, uint64_t n) {
    uint64_t count = 0;
    for (uint64_t p = first; p < first + n; p++) count += m->bits[p / 64] >> (p % 64) & 1;
    return count;
}

static void bits_set(struct nvn_vmap *m, uint64_t first, uint64_t n, int on) {
    for (uint64_t p = first; p < first + n; p++) {
        if (on) m->bits[p / 64] |= 1ull << (p % 64);
        else m->bits[p / 64] &= ~(1ull << (p % 64));
    }
}

// ---- pools
// Where the pool lives on the GPU, from its flags (MacNativeGames totk_vpool.c checks, through oot-pc-recomp): 0 with a
// log line for a combination the game must not ask for, which fails MemoryPoolInitialize.
static int place(pool_t *o) {
    int phys = (o->flags & PHYSICAL) != 0, virt = (o->flags & VIRTUAL) != 0;
    const char *refused = NULL;
    if (phys && virt) refused = "both PHYSICAL and VIRTUAL";
    else if (virt && !(o->flags & CPU_NO_ACCESS)) refused = "a virtual pool the CPU can see";
    else if (virt && o->game_mem) refused = "a virtual pool given storage";
    else if (!virt && !o->game_mem) refused = "a pool without storage";
    else if (!o->size) refused = "an empty pool";
    else if ((phys || virt) && o->size % GFX_GPU_PAGE) refused = "a physical or virtual pool not made of 64 KiB pages";
    if (refused) {
        rt_log("nvn: MemoryPoolInitialize refused: %s (flags %#x, storage %p, size %#llx)", refused, o->flags,
               (void *)o->game_mem, (ull)o->size);
        return 0;
    }
    if (!(o->flags & CPU_NO_ACCESS)) {
        o->mem = o->game_mem;   // guest pointers are host pointers: the GPU address is the guest address
        return 1;
    }
    uint64_t span = (o->size + GFX_GPU_PAGE - 1) & ~(GFX_GPU_PAGE - 1);
    struct nvn_vmap *m = NULL;
    if (virt) {
        uint64_t pages = o->size / GFX_GPU_PAGE;
        m = calloc(1, sizeof *m + (pages + 63) / 64 * sizeof(uint64_t));
        if (!m) rt_fatal("nvn: out of memory for a virtual pool of %#llx bytes", (ull)o->size);
        m->pages = pages;
    }
    plat_lock(&g_lock);
    uint64_t at = window_take(span);
    plat_unlock(&g_lock);
    if (!at)
        rt_fatal("nvn: the GPU address window is full: a pool of %#llx bytes (flags %#x) has no GPU range",
                 (ull)o->size, o->flags);
    o->mem = (uint8_t *)(uintptr_t)at;
    o->span = span;
    o->vmap = m;
    return 1;
}

// What place took: the window range and a virtual pool's mapping state.
static void unplace(pool_t *o) {
    if (o->span) {
        plat_lock(&g_lock);
        window_give((uintptr_t)o->mem, o->span);
        plat_unlock(&g_lock);
    }
    free(o->vmap);
    o->span = 0, o->vmap = NULL;
}

static int gpu_reaches(const pool_t *o) { return !(o->flags & GPU_NO_ACCESS); }

// ---- the counters of the census (gfx.h, NvnPoolStats): an initialised pool counts in its class from MemoryPoolInitialize
// to the end of the pool
static int pool_class(const pool_t *o) {
    return (o->flags & VIRTUAL) ? NVN_POOL_VIRTUAL
           : (o->flags & PHYSICAL) ? NVN_POOL_PHYSICAL
           : (o->flags & CPU_NO_ACCESS) ? NVN_POOL_CPU_NO_ACCESS
                                        : NVN_POOL_CPU;
}

static void stats_add(const pool_t *o) {
    plat_lock(&g_lock);
    NvnPoolClass *k[2] = {&g_stats.cls[pool_class(o)], &g_stats.all};
    for (int i = 0; i < 2; i++) {
        k[i]->created++, k[i]->live++, k[i]->bytes += o->size;
        if (k[i]->live > k[i]->peak) k[i]->peak = k[i]->live;
        if (k[i]->bytes > k[i]->peak_bytes) k[i]->peak_bytes = k[i]->bytes;
    }
    g_stats.gpu_no_access += (o->flags & GPU_NO_ACCESS) != 0;
    plat_unlock(&g_lock);
}

static void stats_del(const pool_t *o) {
    plat_lock(&g_lock);
    NvnPoolClass *k[2] = {&g_stats.cls[pool_class(o)], &g_stats.all};
    for (int i = 0; i < 2; i++) {
        if (k[i]->live) k[i]->live--;
        k[i]->bytes -= o->size < k[i]->bytes ? o->size : k[i]->bytes;
    }
    plat_unlock(&g_lock);
}

void nvn_pool_stats(NvnPoolStats *out) {
    plat_lock_shared(&g_lock);
    *out = g_stats;
    plat_unlock_shared(&g_lock);
}

static void index_add(const pool_t *o) {
    uintptr_t lo = (uintptr_t)o->mem;
    pidx_ent e = {.lo = lo, .hi = lo + o->size, .base = lo, .gpu = lo, .key = o};
    plat_lock(&g_lock);
    int ok = pidx_insert(&g_index, &e);
    plat_unlock(&g_lock);
    if (!ok) rt_fatal("nvn: out of memory for the pool index");
}

static int index_del(const pool_t *o) {
    plat_lock(&g_lock);
    int had = pidx_remove(&g_index, o, NULL);
    plat_unlock(&g_lock);
    return had;
}

const pool_t *nvn_pool_find(uint64_t addr, uint64_t len) {
    plat_lock_shared(&g_lock);
    const pidx_ent *e = pidx_find(&g_index, (uintptr_t)addr, len ? len : 1);
    const pool_t *p = e ? e->key : NULL;
    plat_unlock_shared(&g_lock);
    return p;
}

// The end of a pool: out of the index, out of the back end, its window range and mapping state given back.
static void pool_end(pool_t *o) {
    stats_del(o);
    if (texheap_on()) texheap_pool_end(o);
    if (gpu_reaches(o)) index_del(o);
    gfx_backend()->pool_fini(o);
    unplace(o);
    o->magic = 0;
}

// void nvnMemoryPoolBuilderSetDefaults(NVNmemoryPoolBuilder *b)
NVN_API(nvnMemoryPoolBuilderSetDefaults) { memset(arg_p(c, 0), 0, sizeof(pool_builder_t)); }
// void nvnMemoryPoolBuilderSetDevice(NVNmemoryPoolBuilder *b, NVNdevice *d)
NVN_API(nvnMemoryPoolBuilderSetDevice) { ((pool_builder_t *)arg_p(c, 0))->dev = arg_p(c, 1); }
// void nvnMemoryPoolBuilderSetStorage(NVNmemoryPoolBuilder *b, void *memory, size_t size)
NVN_API(nvnMemoryPoolBuilderSetStorage) {
    pool_builder_t *b = arg_p(c, 0);
    b->mem = arg_p(c, 1);
    b->size = arg_x(c, 2);
}
// void nvnMemoryPoolBuilderSetFlags(NVNmemoryPoolBuilder *b, int flags)
NVN_API(nvnMemoryPoolBuilderSetFlags) { ((pool_builder_t *)arg_p(c, 0))->flags = (uint32_t)arg_x(c, 1); }
// NVNdevice *nvnMemoryPoolBuilderGetDevice(const NVNmemoryPoolBuilder *b), void *...GetMemory, size_t ...GetSize,
// NVNmemoryPoolFlags ...GetFlags
NVN_API(nvnMemoryPoolBuilderGetDevice) { ret_x(c, (uintptr_t)((const pool_builder_t *)arg_p(c, 0))->dev); }
NVN_API(nvnMemoryPoolBuilderGetMemory) { ret_x(c, (uintptr_t)((const pool_builder_t *)arg_p(c, 0))->mem); }
NVN_API(nvnMemoryPoolBuilderGetSize) { ret_x(c, ((const pool_builder_t *)arg_p(c, 0))->size); }
NVN_API(nvnMemoryPoolBuilderGetFlags) { ret_x(c, ((const pool_builder_t *)arg_p(c, 0))->flags); }

// NVNboolean nvnMemoryPoolInitialize(NVNmemoryPool *p, const NVNmemoryPoolBuilder *b)
NVN_API(nvnMemoryPoolInitialize) {
    pool_t *o = arg_p(c, 0);
    const pool_builder_t *b = arg_p(c, 1);
    // A pool object initialised again without MemoryPoolFinalize: the old pool ends first, if it is the one the index
    // holds (a copy of a pool object by value is not, and must not give back the original's range).
    if (o->magic == NVN_POOL_MAGIC && (gpu_reaches(o) ? index_del(o) : 0)) {
        stats_del(o);
        if (texheap_on()) texheap_pool_end(o);
        gfx_backend()->pool_fini(o);
        unplace(o);
    }
    *o = (pool_t){NVN_POOL_MAGIC, b->flags, NULL, b->size, b->mem, NULL, 0};
    int ok = place(o);
    if (ok && !gfx_backend()->pool_init(o)) {
        rt_log("nvn: MemoryPoolInitialize: the %s back end refused the pool at %p (flags %#x, size %#llx)",
               gfx_backend()->name, (void *)o->mem, o->flags, (ull)o->size);
        unplace(o);
        ok = 0;
    }
    if (ok && gpu_reaches(o)) index_add(o);
    if (ok) stats_add(o);
    if (ok && texheap_on()) texheap_pool_init(o);
    if (!ok) o->magic = 0;
    ret_x(c, ok);
}

// void nvnMemoryPoolFinalize(NVNmemoryPool *p)
NVN_API(nvnMemoryPoolFinalize) {
    pool_t *o = arg_p(c, 0);
    if (o->magic == NVN_POOL_MAGIC) pool_end(o);
}

// void nvnMemoryPoolSetDebugLabel(NVNmemoryPool *p, const char *label): for NVN's debugging tools only
NVN_API(nvnMemoryPoolSetDebugLabel) { (void)c; }

// void *nvnMemoryPoolMap(const NVNmemoryPool *p): NULL for a pool the CPU cannot see
NVN_API(nvnMemoryPoolMap) {
    const pool_t *o = arg_p(c, 0);
    ret_x(c, o->magic == NVN_POOL_MAGIC && !(o->flags & CPU_NO_ACCESS) ? (uintptr_t)o->mem : 0);
}

// The part of [off, off + size) inside a pool the CPU can see, as a guest range; 0 when there is none.
static uint64_t cpu_range(const pool_t *p, int64_t off, uint64_t *size) {
    if (p->magic != NVN_POOL_MAGIC || (p->flags & CPU_NO_ACCESS) || off < 0 || (uint64_t)off >= p->size || !*size)
        return 0;
    if (*size > p->size - (uint64_t)off) *size = p->size - (uint64_t)off;
    return (uintptr_t)p->mem + (uint64_t)off;
}

// void nvnMemoryPoolFlushMappedRange(const NVNmemoryPool *p, ptrdiff_t offset, size_t size): the CPU wrote the range,
// so copies the back end holds of it are stale
NVN_API(nvnMemoryPoolFlushMappedRange) {
    uint64_t size = arg_x(c, 2), a = cpu_range(arg_p(c, 0), (int64_t)arg_x(c, 1), &size);
    if (a) {
        if (texheap_on()) texheap_range(TEXHEAP_FLUSH, a, size);
        gfx_backend()->guest_written(a, size);
    }
}

// void nvnMemoryPoolInvalidateMappedRange(const NVNmemoryPool *p, ptrdiff_t offset, size_t size): the CPU is about to
// read what the GPU wrote there, so the back end writes its copies back first
NVN_API(nvnMemoryPoolInvalidateMappedRange) {
    uint64_t size = arg_x(c, 2), a = cpu_range(arg_p(c, 0), (int64_t)arg_x(c, 1), &size);
    if (a) {
        if (texheap_on()) texheap_range(TEXHEAP_INVALIDATE, a, size);
        nvn_invalidate(c, a, size, "MemoryPoolInvalidateMappedRange");
    }
}

// NVNbufferAddress nvnMemoryPoolGetBufferAddress(const NVNmemoryPool *p), NVNmemoryPoolFlags ...GetFlags, size_t
// ...GetSize
NVN_API(nvnMemoryPoolGetBufferAddress) { ret_x(c, (uintptr_t)((const pool_t *)arg_p(c, 0))->mem); }
NVN_API(nvnMemoryPoolGetFlags) { ret_x(c, ((const pool_t *)arg_p(c, 0))->flags); }
NVN_API(nvnMemoryPoolGetSize) { ret_x(c, ((const pool_t *)arg_p(c, 0))->size); }

// ---- MemoryPoolMapVirtual
// Every request is checked before any mapping changes (MacNativeGames totk_vpool.c, through oot-pc-recomp).
static int check_request(const pool_t *v, const nvn_mapping_t *r, int i) {
    uint64_t end = (uint64_t)r->virtual_offset + r->size;
    const char *bad = NULL;
    if (r->virtual_offset < 0 || r->physical_offset < 0 || !r->size) bad = "a negative or empty range";
    else if (((uint64_t)r->virtual_offset | (uint64_t)r->physical_offset | r->size) % GFX_GPU_PAGE)
        bad = "not in 64 KiB pages";
    else if (end < r->size || end > v->size) bad = "past the end of the virtual pool";
    const pool_t *p = (const pool_t *)(const void *)r->physical_pool;
    if (!bad && p) {
        uint64_t pend = (uint64_t)r->physical_offset + r->size;
        if (p->magic != NVN_POOL_MAGIC || !(p->flags & PHYSICAL))
            bad = "from a pool that is not an initialised physical pool";
        else if (pend < r->size || pend > p->size) bad = "past the end of the physical pool";
    }
    if (bad)
        rt_log("nvn: MemoryPoolMapVirtual refused: request %d is %s (virtual %#llx, physical %#llx, size %#llx)", i,
               bad, (ull)r->virtual_offset, (ull)r->physical_offset, (ull)r->size);
    return !bad;
}

// The shape of a call, for the census and for the fatal: every request (first ones only past a few) and the number of
// physical runs among its mappings.
static void describe(const pool_t *v, int n, const nvn_mapping_t *r, char *out, size_t cap) {
    enum { SHOWN = 6 };
    int len = snprintf(out, cap, "virtual pool %#llx (%#llx bytes), %d request%s:", (ull)(uintptr_t)v->mem,
                       (ull)v->size, n, n == 1 ? "" : "s");
    int runs = 0;
    for (int i = 0; i < n; i++) {
        const pool_t *p = (const pool_t *)(const void *)r[i].physical_pool;
        int64_t delta = r[i].physical_offset - r[i].virtual_offset;
        int fresh = p != NULL;
        for (int j = 0; j < i && fresh; j++)
            fresh = !(r[j].physical_pool == r[i].physical_pool && r[j].physical_offset - r[j].virtual_offset == delta);
        runs += fresh;
        if (i >= SHOWN || len < 0 || (size_t)len >= cap) continue;
        if (p)
            len += snprintf(out + len, cap - (size_t)len, " [v+%#llx %#llx <- %#llx+%#llx]", (ull)r[i].virtual_offset,
                            (ull)r[i].size, (ull)(uintptr_t)p->mem, (ull)r[i].physical_offset);
        else
            len += snprintf(out + len, cap - (size_t)len, " [v+%#llx %#llx unmap]", (ull)r[i].virtual_offset,
                            (ull)r[i].size);
    }
    if (len >= 0 && (size_t)len < cap)
        snprintf(out + len, cap - (size_t)len, "%s; %s", n > SHOWN ? " ..." : "",
                 runs == 0   ? "no mapping"
                 : runs == 1 ? "one physical run"
                 : runs == 2 ? "two physical runs"
                             : "three or more physical runs");
}

// NVNboolean nvnMemoryPoolMapVirtual(NVNmemoryPool *p, int n, const NVNmappingRequest *requests)
NVN_API(nvnMemoryPoolMapVirtual) {
    pool_t *v = arg_p(c, 0);
    int n = (int)arg_x(c, 1);
    const nvn_mapping_t *r = arg_p(c, 2);
    if (v->magic != NVN_POOL_MAGIC || !(v->flags & VIRTUAL) || !v->vmap) {
        rt_log("nvn: MemoryPoolMapVirtual refused: %p is not an initialised virtual pool", (void *)v);
        ret_x(c, 0);
        return;
    }
    if (n < 0 || (n && !r)) {
        rt_log("nvn: MemoryPoolMapVirtual refused: %d requests at %p", n, (const void *)r);
        ret_x(c, 0);
        return;
    }
    for (int i = 0; i < n; i++)
        if (!check_request(v, &r[i], i)) {
            ret_x(c, 0);
            return;
        }
    char shape[640];
    describe(v, n, r, shape, sizeof shape);
    // The requests in order: a mapping must join the run the pool's other mapped pages lie in, or start one when no
    // other page is mapped; an unmapping always applies.
    struct nvn_vmap *m = v->vmap;
    plat_lock(&g_lock);
    for (int i = 0; i < n; i++) {
        uint64_t first = (uint64_t)r[i].virtual_offset / GFX_GPU_PAGE, count = r[i].size / GFX_GPU_PAGE;
        uint64_t others = m->mapped - bits_count(m, first, count);
        const pool_t *p = (const pool_t *)(const void *)r[i].physical_pool;
        int64_t delta = r[i].physical_offset - r[i].virtual_offset;
        if (p && others && (p != m->phys || delta != m->delta)) {
            plat_unlock(&g_lock);
            rt_unimplemented(c, "nvn",
                             "nvnMemoryPoolMapVirtual: request %d would leave the pool's pages in two physical runs, "
                             "which the descriptor path cannot express: %s",
                             i, shape);
        }
        bits_set(m, first, count, p != NULL);
        m->mapped = others + (p ? count : 0);
        if (p) m->phys = p, m->delta = delta;
        else if (!m->mapped) m->phys = NULL, m->delta = 0;
    }
    uint64_t mapped = m->mapped, pages = m->pages;
    plat_unlock(&g_lock);
    if (!gfx_backend()->pool_map_virtual(v, n, r))
        rt_fatal("nvn: the %s back end failed MemoryPoolMapVirtual after the front end took it: %s",
                 gfx_backend()->name, shape);
    rt_log("nvn: MemoryPoolMapVirtual %s; %llu of %llu pages mapped", shape, (ull)mapped, (ull)pages);
    ret_x(c, 1);
}
