// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The backing map (oot2/backing.h, architecture D14): pool records in an index sorted by GPU address, their backings
// (a VRAM extent per pool the CPU cannot see, import cells for a pool it can see, one run for a virtual pool), the VRAM
// blocks extents are cut from and the thread that makes them ahead of use, and the deferred-release queue.
//
// The index is the front end's (runtime/nvn/nvn_pool_index.h), so a nested or overlapping pool is still found. A
// resolve that finds its backing holds the lock shared and stamps the submission's value on its pools and its backing
// with an atomic maximum; one that must make a backing takes the lock exclusively and looks again. A backing leaves the
// map only under the exclusive lock, so a stamp made under the shared lock is seen by whoever removes it. Memory is
// freed outside the lock (freeing a 64 MiB import takes about 1.6 ms, the write-watch study).
//
// A cell holds the imports and mirrors of the separate runs of committed pages that binds reached inside it, sorted and
// disjoint. A bind outside them commits its own pages, then takes the whole run of committed pages around it: every
// backing of the cell inside that run is replaced by one over the run.
//
// Mirrors (backing.h, Mirrors): VRAM parts of the blocks, as extents are, over the run an import would cover. A mirror
// is brought current at its first resolve in each submission (mirror_use) from the page generations (oot2/pagegen.h);
// the GPU writes declared into it are read back at the submission's end and written into guest memory by an effect on
// the completion thread (writeback), which runs once the device reached the value and before the value is done.
//
// The mirrors' copies and write-backs are adapted from DKB Recomp runtime/vk_memory.c at 9d8fa02 (GPL-3.0-or-later):
// the admission rule (used + size + budget / 8 within the budget), the generation a mirror is current up to, the copy of
// the pages stamped after it, a GPU write noted at record time and copied back into guest memory at its fence, and the
// copies into a mirror that leave out the bytes of GPU writes not copied back yet. Changed here: a mirror covers the
// committed run around its bind instead of the pool, needs no import (its copies go through staging), is brought current
// at its first resolve in a submission rather than by 2 MiB snapshots per submission, and a new backing over older ones
// waits for the submissions that used them.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
#include "plat.h"
#include "../nvn/nvn_pool_index.h"

#define WINDOW BACKING_IMPORT_WINDOW
#define HPAGE BACKING_HOST_PAGE
#define CPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS
#define GPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS
#define PHYSICAL NVN_MEMORY_POOL_FLAGS_PHYSICAL
#define VIRTUAL NVN_MEMORY_POOL_FLAGS_VIRTUAL
#define BLOCK_THREAD_STACK (256u << 10)

_Static_assert(BACKING_IMPORT_WINDOW == PLAT_PIECE, "an import window is one piece of the watched reservations");
_Static_assert(BACKING_BLOCK % BACKING_EXTENT_ALIGN == 0, "blocks hold whole extent pages");

typedef unsigned long long ull;

enum { CLS_CPU, CLS_GPU, CLS_VIRTUAL };   // a pool the CPU sees; one in the window with storage; a virtual one

// A VRAM block: one allocation and its buffer, cut into extents by a first-fit list of free spans (offsets).
typedef struct Span { uint64_t lo, hi; } Span;
typedef struct Block {
    BackingMem mem;
    uint64_t used;
    int dedicated;   // the block of one extent larger than BACKING_BLOCK
    Span *free;
    int nfree, cap;
} Block;

// An extent or a mirror (a part of a block), or an import. mem is the whole buffer; offset is where gpu_lo lies in it.
typedef struct Backing {
    int kind;
    uint64_t gpu_lo, gpu_hi;
    uint64_t last_use;   // atomic: the newest submission value that resolved into it
    uint64_t written;    // atomic: the newest value that declared a GPU write into it (backing_gpu_wrote)
    BackingMem mem;
    uint64_t offset;
    Block *block;        // an extent's or a mirror's
    uint64_t part;       // the bytes it takes of its block
    uint64_t part_at;    // where that part starts in the block (offset moves when a mirror is cut at its start)
    // A mirror's, under g_coh: every CPU write stamped up to synced is in it (none while fresh); sync_value is the
    // submission it was brought current for last (atomic, read without g_coh to skip a second sync).
    uint64_t synced, sync_value;
    int fresh;
    // A mirror's: the hull of what submission res_value resolved in it, written by the thread recording that submission
    // (one at a time) and read by it under the exclusive lock: the raw path's binds take pages from a mirror only where
    // no earlier bind of their submission reached it (beside).
    uint64_t res_value, res_lo, res_hi;
    // An import's: a bind larger than a uniform's resolved into it (atomic). The next raw-access submission's begin gives
    // such an import the mirror the rule would have made for the bind (promote_hot).
    int hot;
} Backing;

// A pool the CPU sees cut to the windows [k1, k2] one bind touched, and the imports binds made inside it.
typedef struct Cell {
    uint64_t k1, k2, lo, hi;
    Backing **imp;   // sorted by address, disjoint
    int nimp, cap;
} Cell;

typedef struct Rec {
    uint64_t lo, hi;
    uint32_t flags;
    int cls;
    uint64_t last_use;      // atomic
    Backing *ext;           // CLS_GPU: the extent, made at the first bind
    Cell *cells;            // CLS_CPU: sorted by (k1, k2)
    int ncells, cap;
    struct Rec *run;        // CLS_VIRTUAL: the physical pool of the one run (NULL: none mapped)
    int64_t run_delta;      // physical GPU address minus virtual GPU address
    int run_lost;           // the run's physical pool was finalized while pages were mapped
    uint64_t *bits, pages, mapped;
    int mirror_logged;      // a refused mirror of this pool was logged
} Rec;

static PlatLock g_lock = PLAT_LOCK_INIT;
static int g_open;
static BackingOps g_ops;
static pidx g_index;
static Rec **g_virtual;   // the virtual pools, to cut their runs when a physical pool goes
static int g_nvirtual, g_virtual_cap;
static Block **g_blocks;
static int g_nblocks, g_blocks_cap;
static uint64_t g_seen_epoch;
static BackingStats g_st;     // under g_lock
static uint64_t g_resolves, g_misses;   // atomic
static int (*g_mirror_hook)(const BackingAsk *ask);

// The deferred-release queue (g_lock).
typedef struct Deferred { Backing *b; uint64_t value; } Deferred;
static Deferred *g_def;
static int g_ndef, g_def_cap;

// Mirrors: the live ones (g_lock), and the coherence state (g_coh, after g_lock; the completion thread takes it alone):
// GPU writes declared and not written back yet, and the counters of syncs and write-backs.
static Backing **g_mirrors;
static int g_nmirrors, g_mirrors_cap;
static PlatMutex g_coh = PLAT_MUTEX_INIT;
typedef struct Pending { uint64_t value, lo, hi; Backing *m; } Pending;
static Pending *g_pend;
static int g_npend, g_pend_cap;
static BackingStats g_cst;   // the coherence counters (g_coh)

// The raw path (backing.h, The raw path): the map's version, the newest raw-access submission, and what the last pass of
// the invariant saw (g_lock).
static uint64_t g_version = 1;   // atomic, changed under g_lock exclusive
static uint64_t g_raw_last;      // atomic
static uint64_t g_raw_prev;      // atomic: the raw-access submission that began before the newest one (0: none)
static uint64_t g_pools_version, g_fill_epoch, g_fill_pools;
static int g_filled;

// The block thread.
static PlatMutex g_bt_lock = PLAT_MUTEX_INIT;
static PlatCond g_bt_cond = PLAT_COND_INIT;
static int g_bt_want, g_bt_stop, g_bt_running;
static PlatThread *g_bt;

static uint64_t up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
static uint64_t down(uint64_t v, uint64_t a) { return v & ~(a - 1); }

static void stamp(uint64_t *at, uint64_t value) {
    uint64_t old = __atomic_load_n(at, __ATOMIC_RELAXED);
    while (old < value && !__atomic_compare_exchange_n(at, &old, value, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
    }
}

static uint64_t used_by(const Backing *b) { return __atomic_load_n(&b->last_use, __ATOMIC_ACQUIRE); }

// The newest submission that can reach b: its own last resolve, or a raw-access submission, which reaches every
// backing its page table maps (the backing map study: one global value rather than a stamp on every backing).
static uint64_t raw_last(void) { return __atomic_load_n(&g_raw_last, __ATOMIC_ACQUIRE); }
static uint64_t reach(const Backing *b) {
    uint64_t u = used_by(b), r = raw_last();
    return u > r ? u : r;
}

// What an older submission can have written through the page table before the submission `value` (0: none, every issued
// one counts): the newest raw-access submission that is not value's own. Their raw stores into a mirror reach guest memory
// only when they are done, so a copy of those bytes waits for them.
static uint64_t raw_before(uint64_t value) {
    uint64_t r = raw_last();
    return value && r >= value ? __atomic_load_n(&g_raw_prev, __ATOMIC_ACQUIRE) : r;
}
static uint64_t reach_before(const Backing *b, uint64_t value) {
    uint64_t u = used_by(b), r = raw_before(value);
    return u > r ? u : r;
}

// A backing entered or left the map, or a virtual pool's run moved: the raw path's table is out of date. g_lock
// exclusive.
static void changed(void) { __atomic_fetch_add(&g_version, 1, __ATOMIC_RELEASE); }

static void *grow(void *p, int *cap, int need, size_t size, const char *what) {
    if (need <= *cap) return p;
    int c = *cap ? *cap : 16;
    while (c < need) c *= 2;
    void *q = realloc(p, (size_t)c * size);
    if (!q) rt_fatal("gfx: vulkan: backing map: out of memory for %s", what);
    *cap = c;
    return q;
}

// ---------- the rule
void backing_set_mirror_hook(int (*want_mirror)(const BackingAsk *ask)) { g_mirror_hook = want_mirror; }

int backing_rule(const BackingAsk *ask) {
    if (ask->pool_flags & CPU_NO_ACCESS) return BACKING_EXTENT;
    if (g_mirror_hook && g_mirror_hook(ask)) return BACKING_MIRROR;
    return BACKING_IMPORT;
}

int backing_mirror_fits(uint64_t used, uint64_t budget, uint64_t size) {
    return used <= budget && size <= budget - used && budget / BACKING_MIRROR_RESERVE <= budget - used - size;
}

static int has_mirrors(void) { return g_ops.vram != NULL; }

// ---------- VRAM blocks (g_lock exclusive unless said)
static int spare_count(void) {
    int n = 0;
    for (int i = 0; i < g_nblocks; i++) n += !g_blocks[i]->dedicated && !g_blocks[i]->used;
    return n;
}

static Block *block_add(const BackingMem *mem, int dedicated) {
    Block *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for a block");
    b->mem = *mem;
    b->dedicated = dedicated;
    b->free = grow(NULL, &b->cap, 1, sizeof *b->free, "a block's free list");
    b->free[0] = (Span){0, mem->size};
    b->nfree = 1;
    g_blocks = grow(g_blocks, &g_blocks_cap, g_nblocks + 1, sizeof *g_blocks, "the block list");
    g_blocks[g_nblocks++] = b;
    g_st.blocks++, g_st.block_bytes += mem->size;
    return b;
}

static void block_drop(Block *b) {
    for (int i = 0; i < g_nblocks; i++)
        if (g_blocks[i] == b) {
            g_blocks[i] = g_blocks[--g_nblocks];
            break;
        }
    g_st.blocks--, g_st.block_bytes -= b->mem.size;
    free(b->free);
    free(b);
}

// The lowest free span of b that holds size bytes, taken: its offset, or UINT64_MAX.
static uint64_t block_take(Block *b, uint64_t size) {
    for (int i = 0; i < b->nfree; i++) {
        if (b->free[i].hi - b->free[i].lo < size) continue;
        uint64_t at = b->free[i].lo;
        b->free[i].lo += size;
        if (b->free[i].lo == b->free[i].hi) {
            memmove(b->free + i, b->free + i + 1, (size_t)(b->nfree - i - 1) * sizeof *b->free);
            b->nfree--;
        }
        b->used += size;
        return at;
    }
    return UINT64_MAX;
}

static int block_free_at(const Block *b, uint64_t lo, uint64_t hi) {
    if (lo == hi) return 1;
    for (int i = 0; i < b->nfree; i++)
        if (b->free[i].lo <= lo && b->free[i].hi >= hi) return 1;
    return 0;
}

static void block_take_at(Block *b, uint64_t lo, uint64_t hi) {
    if (lo == hi) return;
    for (int i = 0; i < b->nfree; i++) {
        Span s = b->free[i];
        if (s.lo > lo || s.hi < hi) continue;
        if (s.lo == lo && s.hi == hi) {
            memmove(b->free + i, b->free + i + 1, (size_t)(b->nfree - i - 1) * sizeof *b->free);
            b->nfree--;
        } else if (s.lo == lo) {
            b->free[i].lo = hi;
        } else if (s.hi == hi) {
            b->free[i].hi = lo;
        } else {
            b->free = grow(b->free, &b->cap, b->nfree + 1, sizeof *b->free, "a block's free list");
            memmove(b->free + i + 2, b->free + i + 1, (size_t)(b->nfree - i - 1) * sizeof *b->free);
            b->free[i].hi = lo;
            b->free[i + 1] = (Span){hi, s.hi};
            b->nfree++;
        }
        b->used += hi - lo;
        return;
    }
    rt_fatal("gfx: vulkan: backing map: block part [%#llx, %#llx) was not free", (ull)lo, (ull)hi);
}

static void block_give(Block *b, uint64_t lo, uint64_t size) {
    uint64_t hi = lo + size;
    int at = 0;
    while (at < b->nfree && b->free[at].lo < lo) at++;
    if ((at > 0 && b->free[at - 1].hi > lo) || (at < b->nfree && b->free[at].lo < hi))
        rt_fatal("gfx: vulkan: backing map: block part %#llx+%#llx given back twice", (ull)lo, (ull)size);
    int prev = at > 0 && b->free[at - 1].hi == lo, next = at < b->nfree && b->free[at].lo == hi;
    if (prev && next) {
        b->free[at - 1].hi = b->free[at].hi;
        memmove(b->free + at, b->free + at + 1, (size_t)(b->nfree - at - 1) * sizeof *b->free);
        b->nfree--;
    } else if (prev) {
        b->free[at - 1].hi = hi;
    } else if (next) {
        b->free[at].lo = lo;
    } else {
        b->free = grow(b->free, &b->cap, b->nfree + 1, sizeof *b->free, "a block's free list");
        memmove(b->free + at + 1, b->free + at, (size_t)(b->nfree - at) * sizeof *b->free);
        b->free[at] = (Span){lo, hi};
        b->nfree++;
    }
    b->used -= size;
}

static void bt_kick(void) {
    plat_mutex_lock(&g_bt_lock);
    g_bt_want = 1;
    plat_cond_signal(&g_bt_cond);
    plat_mutex_unlock(&g_bt_lock);
}

// The block thread: keeps BACKING_SPARE_BLOCKS empty blocks, making each without the map's lock.
static void bt_main(void *arg) {
    (void)arg;
    for (;;) {
        plat_mutex_lock(&g_bt_lock);
        while (!g_bt_want && !g_bt_stop) plat_cond_wait(&g_bt_cond, &g_bt_lock);
        int stop = g_bt_stop;
        g_bt_want = 0;
        plat_mutex_unlock(&g_bt_lock);
        if (stop) break;
        for (;;) {
            plat_lock_shared(&g_lock);
            int need = spare_count() < BACKING_SPARE_BLOCKS;
            plat_unlock_shared(&g_lock);
            if (!need) break;
            BackingMem m;
            if (!g_ops.block_new(BACKING_BLOCK, &m)) {
                rt_log("gfx: vulkan: backing map: the device has no room for a spare block of %llu MiB",
                       (ull)(BACKING_BLOCK >> 20));
                break;
            }
            // Retirement may have emptied a block while this one was made: it is kept only if still wanted.
            plat_lock(&g_lock);
            int keep = spare_count() < BACKING_SPARE_BLOCKS;
            if (keep) block_add(&m, 0), g_st.blocks_made_ahead++;
            plat_unlock(&g_lock);
            if (!keep) g_ops.free(&m);
        }
    }
    plat_mutex_lock(&g_bt_lock);
    g_bt_running = 0;
    plat_cond_broadcast(&g_bt_cond);
    plat_mutex_unlock(&g_bt_lock);
}

// A block made on the calling thread, because none was ready or the extent needs one of its own: counted and logged.
static Block *block_inline(uint64_t size, int dedicated, const char *what) {
    BackingMem m;
    uint64_t t0 = plat_now_ns();
    if (!g_ops.block_new(size, &m))
        rt_fatal("gfx: vulkan: backing map: the device has no room for a block of %llu MiB (%s)", (ull)(size >> 20),
                 what);
    Block *b = block_add(&m, dedicated);
    if (dedicated) g_st.blocks_dedicated++;
    else g_st.blocks_made_inline++;
    rt_log("gfx: vulkan: backing map: %s: a block of %llu MiB%s made on the resolving thread in %.1f ms", what,
           (ull)(size >> 20), dedicated ? " of its own" : " (no spare was ready)", (double)(plat_now_ns() - t0) / 1e6);
    return b;
}

// size bytes (a multiple of BACKING_EXTENT_ALIGN) of a VRAM block, for what: the block, and the offset in *out.
// g_lock exclusive.
static Block *vram_take(uint64_t size, const char *what, uint64_t *out) {
    uint64_t at = UINT64_MAX;
    Block *blk = NULL;
    if (size > BACKING_BLOCK) {
        blk = block_inline(size, 1, what);
        at = block_take(blk, size);
    } else {
        for (int i = 0; i < g_nblocks && at == UINT64_MAX; i++) {
            if (g_blocks[i]->dedicated) continue;
            int was_empty = !g_blocks[i]->used;
            at = block_take(g_blocks[i], size);
            if (at == UINT64_MAX) continue;
            blk = g_blocks[i];
            if (was_empty) bt_kick();   // a spare went into use
        }
        if (at == UINT64_MAX) {
            blk = block_inline(BACKING_BLOCK, 0, what);
            at = block_take(blk, size);
            bt_kick();
        }
    }
    *out = at;
    return blk;
}

// A new extent over the whole of rec's range (g_lock exclusive).
static Backing *extent_new(Rec *r) {
    uint64_t size = up(r->hi - r->lo, BACKING_EXTENT_ALIGN), at;
    char what[96];
    snprintf(what, sizeof what, "the extent of pool [%#llx, %#llx)", (ull)r->lo, (ull)r->hi);
    Block *blk = vram_take(size, what, &at);
    Backing *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for an extent");
    *b = (Backing){.kind = BACKING_EXTENT, .gpu_lo = r->lo, .gpu_hi = r->lo + size, .mem = blk->mem, .offset = at,
                   .block = blk, .part = size, .part_at = at};
    g_st.extents++, g_st.extent_bytes += size;
    return b;
}

static uint64_t mirror_part_size(uint64_t lo, uint64_t hi) {
    return up(hi - lo, BACKING_EXTENT_ALIGN) + 2 * BACKING_MIRROR_GUARD;
}

// A new mirror over guest pages [lo, hi), fresh: nothing is copied into it until its first use in a submission.
// The reserved ends let an already-bound mirror grow without moving or displacing another backing. g_lock exclusive.
static Backing *mirror_new(uint64_t lo, uint64_t hi) {
    uint64_t size = mirror_part_size(lo, hi), at;
    char what[96];
    snprintf(what, sizeof what, "the mirror of [%#llx, %#llx)", (ull)lo, (ull)hi);
    Block *blk = vram_take(size, what, &at);
    Backing *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for a mirror");
    *b = (Backing){.kind = BACKING_MIRROR, .gpu_lo = lo, .gpu_hi = hi, .mem = blk->mem,
                   .offset = at + BACKING_MIRROR_GUARD, .block = blk, .part = size, .part_at = at, .fresh = 1};
    g_mirrors = grow(g_mirrors, &g_mirrors_cap, g_nmirrors + 1, sizeof *g_mirrors, "the mirror list");
    g_mirrors[g_nmirrors++] = b;
    g_st.mirrors++, g_st.mirror_bytes += hi - lo, g_st.mirrors_made++;
    return b;
}

// ---------- the deferred-release queue
static void defer(Backing *b) {
    if (!b) return;
    changed();
    g_def = grow(g_def, &g_def_cap, g_ndef + 1, sizeof *g_def, "the deferred-release queue");
    g_def[g_ndef++] = (Deferred){b, reach(b)};
    g_st.deferred = (uint64_t)g_ndef;
    if (g_st.deferred > g_st.deferred_peak) g_st.deferred_peak = g_st.deferred;
    if (b->kind == BACKING_EXTENT) {
        g_st.extents--, g_st.extent_bytes -= b->gpu_hi - b->gpu_lo;
    } else if (b->kind == BACKING_MIRROR) {
        g_st.mirrors--, g_st.mirror_bytes -= b->gpu_hi - b->gpu_lo;
        for (int i = 0; i < g_nmirrors; i++)
            if (g_mirrors[i] == b) {
                g_mirrors[i] = g_mirrors[--g_nmirrors];
                break;
            }
    } else {
        g_st.imports--, g_st.import_bytes -= b->gpu_hi - b->gpu_lo;
    }
}

// What to free once the lock is dropped.
typedef struct Freeing { BackingMem *m; int n, cap; } Freeing;

static void to_free(Freeing *f, const BackingMem *m) {
    f->m = grow(f->m, &f->cap, f->n + 1, sizeof *f->m, "the free list");
    f->m[f->n++] = *m;
}

// Take every backing whose value is done (every one when all) out of the queue: an extent goes back to its block, and
// an empty block past the spares, a dedicated one and an import are freed after the lock. g_lock exclusive.
static void retire(Freeing *f, int all) {
    uint64_t done = gpu_sync_done();
    int k = 0;
    for (int i = 0; i < g_ndef; i++) {
        Deferred d = g_def[i];
        if (!all && d.value > done) {
            g_def[k++] = d;
            continue;
        }
        Backing *b = d.b;
        if (b->kind != BACKING_IMPORT) {
            Block *blk = b->block;
            block_give(blk, b->part_at, b->part);
            if (!blk->used && (all || blk->dedicated || spare_count() > BACKING_SPARE_BLOCKS)) {
                to_free(f, &blk->mem);
                block_drop(blk);
            }
        } else {
            to_free(f, &b->mem);
        }
        g_st.freed++;
        free(b);
    }
    g_ndef = k;
    g_st.deferred = (uint64_t)k;
}

static void free_all(Freeing *f) {
    for (int i = 0; i < f->n; i++) g_ops.free(&f->m[i]);
    free(f->m);
    *f = (Freeing){0};
}

static void drain(void) {
    Freeing f = {0};
    plat_lock(&g_lock);
    retire(&f, 0);
    plat_unlock(&g_lock);
    free_all(&f);
}

// Wait until value is done, giving the guest's virtual core up meanwhile when the caller is a guest thread (D7).
static void wait_value(uint64_t value) {
    if (value <= gpu_sync_done()) return;
    int token = rt_kernel_block_begin();
    gpu_sync_wait(value, UINT64_MAX);
    rt_kernel_block_end(token);
}

// Hold the queue to BACKING_DEFERRED_MAX by waiting for its oldest value already issued: a value handed out and not
// yet submitted may be the calling thread's own submission, which must not wait for itself. No lock held.
static void bound(void) {
    for (;;) {
        plat_lock_shared(&g_lock);
        uint64_t issued = gpu_sync_issued(), oldest = UINT64_MAX;
        int over = g_ndef > BACKING_DEFERRED_MAX;
        for (int i = 0; over && i < g_ndef; i++)
            if (g_def[i].value <= issued && g_def[i].value < oldest) oldest = g_def[i].value;
        plat_unlock_shared(&g_lock);
        if (!over || oldest == UINT64_MAX) return;
        plat_lock(&g_lock);
        g_st.deferred_full_waits++;
        plat_unlock(&g_lock);
        wait_value(oldest);
        drain();
    }
}

// ---------- pool records (g_lock exclusive)
static const char *cls_name(const Rec *r) {
    if (r->cls == CLS_CPU) return (r->flags & PHYSICAL) ? "physical CPU-visible" : "CPU-visible";
    if (r->cls == CLS_VIRTUAL) return "virtual";
    return (r->flags & PHYSICAL) ? "physical CPU_NO_ACCESS" : "CPU_NO_ACCESS";
}

// The record of exactly [lo, hi), or NULL.
static Rec *rec_exact(uint64_t lo, uint64_t hi) {
    for (int i = pidx_upper(&g_index, (uintptr_t)lo) - 1; i >= 0 && g_index.e[i].lo == lo; i--)
        if (g_index.e[i].hi == hi) return (Rec *)g_index.e[i].key;
    return NULL;
}

static Rec *rec_find(uint64_t addr, uint64_t size) {
    const pidx_ent *e = pidx_find(&g_index, (uintptr_t)addr, size);
    return e ? (Rec *)e->key : NULL;
}

int backing_pool_init(const pool_t *p) {
    if (!g_open) rt_fatal("gfx: vulkan: backing map: a pool before the map is open");
    // The GPU cannot reach it, as the front end's index leaves it out; a physical pool keeps its record all the same,
    // since the GPU reaches it through the virtual pools mapped over it (the title's one physical pool is GPU_NO_ACCESS)
    if ((p->flags & GPU_NO_ACCESS) && !(p->flags & PHYSICAL)) return 1;
    Rec *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("gfx: vulkan: backing map: out of memory for a pool");
    r->lo = (uintptr_t)p->mem;
    r->hi = r->lo + p->size;
    r->flags = p->flags;
    r->cls = (p->flags & VIRTUAL) ? CLS_VIRTUAL : (p->flags & CPU_NO_ACCESS) ? CLS_GPU : CLS_CPU;
    if (r->cls == CLS_VIRTUAL) {
        r->pages = p->size / GFX_GPU_PAGE;
        r->bits = calloc((size_t)((r->pages + 63) / 64 + 1), sizeof *r->bits);
        if (!r->bits) rt_fatal("gfx: vulkan: backing map: out of memory for a virtual pool");
    }
    pidx_ent e = {.lo = (uintptr_t)r->lo, .hi = (uintptr_t)r->hi, .base = (uintptr_t)r->lo, .gpu = r->lo, .key = r};
    plat_lock(&g_lock);
    if (!pidx_insert(&g_index, &e)) rt_fatal("gfx: vulkan: backing map: out of memory for the pool index");
    if (r->cls == CLS_VIRTUAL) {
        g_virtual = grow(g_virtual, &g_virtual_cap, g_nvirtual + 1, sizeof *g_virtual, "the virtual pool list");
        g_virtual[g_nvirtual++] = r;
    }
    g_st.pools++;
    g_pools_version++;   // the raw path's invariant takes the new pool in
    plat_unlock(&g_lock);
    return 1;
}

// Out of the map: its backings deferred with their own values; the newest value stamped on it or them is returned.
// g_lock exclusive.
static uint64_t rec_end(Rec *r) {
    pidx_remove(&g_index, r, NULL);
    uint64_t value = __atomic_load_n(&r->last_use, __ATOMIC_ACQUIRE);
    if (r->ext && reach(r->ext) > value) value = reach(r->ext);
    defer(r->ext);
    for (int i = 0; i < r->ncells; i++) {
        for (int j = 0; j < r->cells[i].nimp; j++) {
            Backing *b = r->cells[i].imp[j];
            if (reach(b) > value) value = reach(b);
            defer(b);
        }
        free(r->cells[i].imp);
    }
    changed();   // a virtual pool's pages leave the raw path's table too
    for (int i = 0; i < g_nvirtual; i++) {
        if (g_virtual[i] == r) g_virtual[i--] = g_virtual[--g_nvirtual];
        else if (g_virtual[i]->run == r) g_virtual[i]->run = NULL, g_virtual[i]->run_lost = 1;
    }
    g_st.pools--;
    free(r->cells);
    free(r->bits);
    free(r);
    return value;
}

void backing_pool_fini(const pool_t *p) {
    if ((p->flags & GPU_NO_ACCESS) && !(p->flags & PHYSICAL)) return;   // as backing_pool_init
    uint64_t lo = (uintptr_t)p->mem, hi = lo + p->size;
    plat_lock(&g_lock);
    Rec *r = rec_exact(lo, hi);
    if (!r) rt_fatal("gfx: vulkan: backing map: pool_fini of [%#llx, %#llx), which pool_init never saw", (ull)lo, (ull)hi);
    uint64_t value = rec_end(r);
    int must_wait = value > gpu_sync_done();
    g_st.fini_waits += (uint64_t)must_wait;
    plat_unlock(&g_lock);
    if (must_wait) wait_value(value);
    drain();
    bound();
}

int backing_pool_map_virtual(const pool_t *v, int n, const nvn_mapping_t *req) {
    uint64_t lo = (uintptr_t)v->mem;
    plat_lock(&g_lock);
    Rec *r = rec_exact(lo, lo + v->size);
    int ok = r && r->cls == CLS_VIRTUAL;
    for (int i = 0; ok && i < n; i++) {
        uint64_t first = (uint64_t)req[i].virtual_offset / GFX_GPU_PAGE, count = req[i].size / GFX_GPU_PAGE;
        const pool_t *pp = (const pool_t *)(const void *)req[i].physical_pool;
        Rec *phys = NULL;
        if (pp) {
            phys = rec_exact((uintptr_t)pp->mem, (uintptr_t)pp->mem + pp->size);
            if (!phys || phys->cls == CLS_VIRTUAL || !(phys->flags & PHYSICAL)) {
                ok = 0;
                break;
            }
        }
        for (uint64_t k = first; k < first + count && k < r->pages; k++) {
            uint64_t m = 1ull << (k % 64);
            int was = (r->bits[k / 64] & m) != 0;
            if (pp) r->bits[k / 64] |= m;
            else r->bits[k / 64] &= ~m;
            if (pp && !was) r->mapped++;
            if (!pp && was) r->mapped--;
        }
        if (pp) {
            r->run = phys;
            r->run_delta = (int64_t)((uintptr_t)pp->mem + (uint64_t)req[i].physical_offset) -
                           (int64_t)(lo + (uint64_t)req[i].virtual_offset);
            r->run_lost = 0;
        } else if (!r->mapped) {
            r->run = NULL, r->run_delta = 0, r->run_lost = 0;
        }
    }
    if (r) changed();
    plat_unlock(&g_lock);
    return ok;
}

// ---------- imports
// [*rs, *re): the run of committed pages inside [lo, hi) that holds the committed pages [plo, phi).
static void committed_run(uint64_t lo, uint64_t hi, uint64_t plo, uint64_t phi, uint64_t *rs, uint64_t *re) {
    uint64_t a = lo, start = lo, end = plo;
    while (a < plo) {
        int c;
        uint64_t n = rt_mem_commit_run(a, plo - a, &c);
        if (n == UINT64_MAX || !n)
            rt_fatal("gfx: vulkan: backing map: %#llx, in a pool the CPU can see, is not guest memory", (ull)a);
        if (!c) start = a + n;
        a += n;
    }
    while (end < hi) {
        int c;
        uint64_t n = rt_mem_commit_run(end, hi - end, &c);
        if (n == UINT64_MAX || !n || !c) break;
        end += n;
    }
    if (end < phi)
        rt_fatal("gfx: vulkan: backing map: [%#llx, %#llx) was committed for a bind and is not now", (ull)plo,
                 (ull)phi);
    *rs = start, *re = end;
}

// Import [lo, hi), committed pages. A refusal is retried once after a 4 KiB import of the committed page probe clears
// it (the write-watch study); a second refusal is a named fatal with the shape.
static Backing *import_new(uint64_t lo, uint64_t hi, uint64_t probe, const char *why) {
    BackingMem m;
    if (!g_ops.import(lo, hi - lo, &m)) {
        g_st.imports_refused++;
        BackingMem pm;
        int cleared = g_ops.import(probe, BACKING_IMPORT_PROBE, &pm);
        if (cleared) g_ops.free(&pm);
        g_st.imports_retried++;
        if (!cleared || !g_ops.import(lo, hi - lo, &m))
            rt_fatal("gfx: vulkan: backing map: the device refused to import committed host memory [%#llx, %#llx) "
                     "(%llu KiB) for %s, %s", (ull)lo, (ull)hi, (ull)((hi - lo) >> 10), why,
                     cleared ? "again after a 4 KiB import cleared the first refusal"
                             : "and a 4 KiB import of its first page was refused too");
    }
    Backing *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for an import");
    *b = (Backing){.kind = BACKING_IMPORT, .gpu_lo = lo, .gpu_hi = hi, .mem = m};
    g_st.imports++, g_st.import_bytes += hi - lo, g_st.imports_made++;
    return b;
}

// The cell of r for windows [k1, k2], made when missing (and make is set).
static Cell *cell_get(Rec *r, uint64_t k1, uint64_t k2, int make) {
    int a = 0, b = r->ncells;
    while (a < b) {
        int m = (a + b) / 2;
        const Cell *c = &r->cells[m];
        if (c->k1 < k1 || (c->k1 == k1 && c->k2 < k2)) a = m + 1;
        else b = m;
    }
    if (a < r->ncells && r->cells[a].k1 == k1 && r->cells[a].k2 == k2) return &r->cells[a];
    if (!make) return NULL;
    r->cells = grow(r->cells, &r->cap, r->ncells + 1, sizeof *r->cells, "a pool's cells");
    memmove(r->cells + a + 1, r->cells + a, (size_t)(r->ncells - a) * sizeof *r->cells);
    uint64_t lo = k1 * WINDOW, hi = (k2 + 1) * WINDOW;
    r->cells[a] = (Cell){k1, k2, down(lo > r->lo ? lo : r->lo, HPAGE), up(hi < r->hi ? hi : r->hi, HPAGE), NULL, 0, 0};
    r->ncells++;
    return &r->cells[a];
}

// The import of c that holds addr, or NULL.
static Backing *cell_import(const Cell *c, uint64_t addr) {
    int a = 0, b = c->nimp;   // the first import that starts above addr
    while (a < b) {
        int m = (a + b) / 2;
        if (c->imp[m]->gpu_lo <= addr) a = m + 1;
        else b = m;
    }
    return a && addr < c->imp[a - 1]->gpu_hi ? c->imp[a - 1] : NULL;
}

// Put the import b over [rs, re) in c in place of every import inside that run, which are deferred. g_lock exclusive.
static void cell_put(Cell *c, uint64_t rs, uint64_t re, Backing *b) {
    int k = 0, at = -1, replaced = 0;
    for (int i = 0; i < c->nimp; i++) {
        Backing *o = c->imp[i];
        if (o->gpu_hi > rs && o->gpu_lo < re) {
            defer(o);
            replaced++;
            continue;
        }
        if (at < 0 && o->gpu_lo >= re) at = k;
        c->imp[k++] = o;
    }
    c->nimp = k;
    if (at < 0) at = k;
    c->imp = grow(c->imp, &c->cap, c->nimp + 1, sizeof *c->imp, "a cell's imports");
    memmove(c->imp + at + 1, c->imp + at, (size_t)(c->nimp - at) * sizeof *c->imp);
    c->imp[at] = b;
    c->nimp++;
    g_st.imports_grown += (uint64_t)replaced;
    changed();
}

// ---------- mirrors: copies in (g_coh)
typedef struct Runs { Span *r; int n, cap; } Runs;
typedef struct Ups { const Backing *m; BackingCopy *c; int n, cap; uint64_t bytes; } Ups;

static void runs_push(void *ctx, uint64_t lo, uint64_t hi) {
    Runs *s = ctx;
    s->r = grow(s->r, &s->cap, s->n + 1, sizeof *s->r, "a mirror's runs");
    s->r[s->n++] = (Span){lo, hi};
}

static void up_push(Ups *u, uint64_t lo, uint64_t hi) {
    u->c = grow(u->c, &u->cap, u->n + 1, sizeof *u->c, "a mirror's copies");
    u->c[u->n++] = (BackingCopy){&u->m->mem, u->m->offset + (lo - u->m->gpu_lo), hi - lo, (const void *)(uintptr_t)lo};
    u->bytes += hi - lo;
}

// [lo, hi) into u, less the bytes of GPU writes not written back yet (pending entries from i on) and the pages not
// committed now (a mirror over them is retired at the next sync).
static void up_range(Ups *u, uint64_t lo, uint64_t hi, int i) {
    for (; i < g_npend; i++)
        if (g_pend[i].lo < hi && g_pend[i].hi > lo) {
            if (g_pend[i].lo > lo) up_range(u, lo, g_pend[i].lo, i + 1);
            if (g_pend[i].hi < hi) up_range(u, g_pend[i].hi, hi, i + 1);
            return;
        }
    for (uint64_t a = lo; a < hi;) {
        int c;
        uint64_t n = rt_mem_commit_run(a, hi - a, &c);
        if (n == UINT64_MAX || !n) break;
        if (c) up_push(u, a, a + n);
        a += n;
    }
}

// Bring m current for submission value: query its range, then copy in what the CPU wrote since it was current last
// (everything the first time), after the query, so the copy holds every write the query stamped.
static void mirror_bring(Backing *m, uint64_t value) {
    uint64_t g = pagegen_scan(m->gpu_lo, m->gpu_hi);
    if (g == UINT64_MAX)
        rt_fatal("gfx: vulkan: backing map: the write watch no longer covers the mirror of [%#llx, %#llx)",
                 (ull)m->gpu_lo, (ull)m->gpu_hi);
    Runs runs = {0};
    if (m->fresh) runs_push(&runs, m->gpu_lo, m->gpu_hi);
    else pagegen_runs(m->gpu_lo, m->gpu_hi, m->synced, runs_push, &runs);
    m->synced = g, m->fresh = 0;
    Ups u = {.m = m};
    for (int i = 0; i < runs.n; i++) up_range(&u, runs.r[i].lo, runs.r[i].hi, 0);
    if (u.n) g_ops.upload(value, u.c, u.n);
    g_cst.mirror_syncs++, g_cst.uploads += (uint64_t)u.n, g_cst.upload_bytes += u.bytes;
    free(u.c);
    free(runs.r);
}

static void no_run(void *ctx, uint64_t lo, uint64_t hi) { (void)ctx, (void)lo, (void)hi; }

// Raw stores into a mirror reach guest memory only when their submission's value is done (the raw path's write-back,
// backing.h), so a copy into m taken while a raw-access submission is in flight would put the CPU's older bytes over
// them in the mirror, and the write-back would then put the GPU's page over the CPU's newer bytes in guest memory. When
// the CPU wrote m since it was current, wait for that submission first. g_coh held, dropped while waiting.
static void raw_stores_wait(Backing *m, uint64_t value) {
    uint64_t r = raw_before(value);   // not value's own: a raw-access submission waits for the ones before it
    if (m->fresh || !r || r >= value || r <= gpu_sync_done() || r > gpu_sync_issued()) return;
    if (pagegen_scan(m->gpu_lo, m->gpu_hi) == UINT64_MAX) return;   // mirror_bring stops by name
    if (!pagegen_runs(m->gpu_lo, m->gpu_hi, m->synced, no_run, NULL)) return;
    g_cst.raw_waits++;
    plat_mutex_unlock(&g_coh);
    wait_value(r);
    plat_mutex_lock(&g_coh);
}

// The first resolve of mirror m in submission value brings it current; the later ones of that submission find it so.
// g_lock held (shared at least), on the submission's thread.
static void mirror_use(Backing *m, uint64_t value) {
    if (!value || __atomic_load_n(&m->sync_value, __ATOMIC_ACQUIRE) == value) return;
    plat_mutex_lock(&g_coh);
    if (m->sync_value != value) raw_stores_wait(m, value);
    if (m->sync_value != value) {
        mirror_bring(m, value);
        __atomic_store_n(&m->sync_value, value, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&g_coh);
}

// A mirror over [rs, re) in r when the admission and the write watch allow one, else NULL (the bind gets an import), the
// pool's first refusal logged. g_lock exclusive.
static Backing *mirror_admit(Rec *r, uint64_t rs, uint64_t re) {
    uint64_t size = mirror_part_size(rs, re), used = 0, budget = 0;
    char why[160] = "";
    if (!g_ops.vram(&used, &budget) || !backing_mirror_fits(used, budget, size)) {
        g_st.mirrors_refused_budget++;
        snprintf(why, sizeof why, "it does not fit the VRAM budget (device-local use %llu of %llu MiB, an eighth "
                 "kept free)", (ull)(used >> 20), (ull)(budget >> 20));
    } else if (pagegen_scan(rs, re) == UINT64_MAX) {
        g_st.mirrors_refused_untracked++;
        snprintf(why, sizeof why, "the write watch does not cover it");
    } else {
        return mirror_new(rs, re);
    }
    if (!r->mirror_logged) {
        r->mirror_logged = 1;
        rt_log("gfx: vulkan: backing map: the mirror of [%#llx, %#llx) (%llu KiB) in pool [%#llx, %#llx) is refused: "
               "%s; imported", (ull)rs, (ull)re, (ull)((re - rs) >> 10), (ull)r->lo, (ull)r->hi, why);
    }
    return NULL;
}

__attribute__((noreturn)) static void written_now(uint64_t lo, uint64_t hi, uint64_t value, uint64_t rs, uint64_t re) {
    rt_fatal("gfx: vulkan: backing map: [%#llx, %#llx), which submission %llu wrote on the GPU while it is being "
             "recorded, would be replaced by a backing over [%#llx, %#llx): the write cannot move to it", (ull)lo,
             (ull)hi, (ull)value, (ull)rs, (ull)re);
}

// Before a new backing over [rs, re) replaces the backings of c inside that run, in submission value: when a mirror is
// among them or will be one, or a GPU write into a mirror retired earlier is still to be written back there, wait for the
// older submissions that used them, so their GPU writes are in guest memory before the new backing takes its copy of it
// (imports alone alias the same memory and need no wait). A backing the submission being recorded wrote cannot be handed
// over: a named fatal. Value 0 (no submission) waits for issued submissions only. g_lock exclusive; the wait holds it,
// which is safe because the completion thread never takes it.
static void replace_wait(const Cell *c, uint64_t rs, uint64_t re, uint64_t value, int mirror) {
    uint64_t wait = 0, issued = gpu_sync_issued();
    for (int i = 0; i < c->nimp; i++)
        if (c->imp[i]->gpu_hi > rs && c->imp[i]->gpu_lo < re) mirror |= c->imp[i]->kind == BACKING_MIRROR;
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend; i++) {
        const Pending *p = &g_pend[i];
        if (p->hi <= rs || p->lo >= re) continue;
        mirror = 1;
        if (value && p->value == value) written_now(p->lo, p->hi, value, rs, re);
        if ((p->value < value || (!value && p->value <= issued)) && p->value > wait) wait = p->value;
    }
    plat_mutex_unlock(&g_coh);
    if (!mirror) return;
    for (int i = 0; i < c->nimp; i++) {
        const Backing *o = c->imp[i];
        if (o->gpu_hi <= rs || o->gpu_lo >= re) continue;
        // Raw-access submissions reach every backing their page table maps: their stores into a mirror are written
        // back at their end, and those through an import land in the guest memory the new mirror copies. The submission
        // being recorded is not waited for (it cannot be): its own raw stores go through the table it ends with.
        uint64_t u = reach_before(o, value);
        if (value && __atomic_load_n(&o->written, __ATOMIC_ACQUIRE) == value) written_now(o->gpu_lo, o->gpu_hi, value, rs, re);
        // A bind in a raw-access submission takes pages from a mirror in beside() and leaves none in its run, so this
        // stands only as a check on that: a mirror the table maps is never replaced by a larger backing.
        if (value && o->kind == BACKING_MIRROR && raw_last() == value)
            rt_fatal("gfx: vulkan: backing map: the mirror [%#llx, %#llx), which the page table of raw-access "
                     "submission %llu maps, would be replaced by a backing over [%#llx, %#llx) while that submission is "
                     "recorded: its raw accesses and its binds would reach different copies", (ull)o->gpu_lo,
                     (ull)o->gpu_hi, (ull)value, (ull)rs, (ull)re);
        if ((u < value || (!value && u <= issued)) && u > wait) wait = u;
    }
    if (wait <= gpu_sync_done()) return;
    g_st.replace_waits++;
    wait_value(wait);
}

// ---------- a bind in a raw-access submission
static int overlaps(uint64_t lo, uint64_t hi, uint64_t a, uint64_t b) {
    return lo < b && a < hi;
}

// Keep a mirror an earlier bind reached at the same device address when a larger bind needs it. The additional pages
// must fit beside its block part, and any imports they replace must not have been bound by this submission.
static int grow_taken(Rec *r, Cell *c, Backing *m, uint64_t plo, uint64_t phi, uint64_t value) {
    uint64_t old_lo = m->gpu_lo, old_hi = m->gpu_hi;
    uint64_t lo = plo < old_lo ? plo : old_lo, hi = phi > old_hi ? phi : old_hi;
    int moved;
    do {
        moved = 0;
        for (int i = 0; i < c->nimp; i++) {
            const Backing *b = c->imp[i];
            if (b == m || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
            if (b->kind != BACKING_IMPORT || used_by(b) >= value) return 0;
            if (b->gpu_lo < lo) lo = b->gpu_lo, moved = 1;
            if (b->gpu_hi > hi) hi = b->gpu_hi, moved = 1;
        }
    } while (moved);
    // Another cell or pool could otherwise resolve the added pages to a different copy.
    for (int i = 0; i < g_index.n; i++) {
        const Rec *other = (const Rec *)g_index.e[i].key;
        if (other->cls != CLS_CPU) continue;
        for (int j = 0; j < other->ncells; j++) {
            const Cell *cell = &other->cells[j];
            if (other == r && cell == c) continue;
            for (int k = 0; k < cell->nimp; k++) {
                const Backing *b = cell->imp[k];
                if (overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) return 0;
            }
        }
    }
    uint64_t added = old_lo - lo + hi - old_hi;
    if (g_st.mirror_bytes > BACKING_PROMOTE_MIRROR_MAX ||
        added > BACKING_PROMOTE_MIRROR_MAX - g_st.mirror_bytes || old_lo - lo > m->offset)
        return 0;
    uint64_t offset = m->offset - (old_lo - lo);
    if (hi - lo > m->mem.size - offset) return 0;
    uint64_t part_lo = down(offset, BACKING_EXTENT_ALIGN), part_hi = up(offset + hi - lo, BACKING_EXTENT_ALIGN);
    uint64_t old_part_lo = m->part_at, old_part_hi = m->part_at + m->part;
    if (part_lo > old_part_lo) part_lo = old_part_lo;
    if (part_hi < old_part_hi) part_hi = old_part_hi;
    if (part_hi > m->mem.size || !block_free_at(m->block, part_lo, old_part_lo) ||
        !block_free_at(m->block, old_part_hi, part_hi))
        return 0;
    if (pagegen_scan(lo, hi) == UINT64_MAX) return 0;

    uint64_t wait = raw_before(value);
    for (int i = 0; i < c->nimp; i++) {
        const Backing *b = c->imp[i];
        if (b == m || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        uint64_t u = reach_before(b, value);
        if (u > wait) wait = u;
    }
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend; i++) {
        const Pending *p = &g_pend[i];
        if (p->m == m || !overlaps(lo, hi, p->lo, p->hi)) continue;
        if (p->value >= value) {
            plat_mutex_unlock(&g_coh);
            return 0;
        }
        if (p->value > wait) wait = p->value;
    }
    plat_mutex_unlock(&g_coh);
    if (wait > gpu_sync_done()) {
        g_st.replace_waits++;
        wait_value(wait);
    }

    block_take_at(m->block, part_lo, old_part_lo);
    block_take_at(m->block, old_part_hi, part_hi);
    m->part_at = part_lo, m->part = part_hi - part_lo;
    m->gpu_lo = lo, m->gpu_hi = hi, m->offset = offset;
    int k = 0, replaced = 0;
    for (int i = 0; i < c->nimp; i++) {
        Backing *b = c->imp[i];
        if (b != m && overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) {
            defer(b);
            replaced++;
        } else {
            c->imp[k++] = b;
        }
    }
    c->nimp = k;
    g_st.imports_grown += (uint64_t)replaced;
    g_st.mirror_bytes += added;
    changed();

    plat_mutex_lock(&g_coh);
    Ups u = {.m = m};
    if (lo < old_lo) up_range(&u, lo, old_lo, 0);
    if (old_hi < hi) up_range(&u, old_hi, hi, 0);
    if (u.n) g_ops.upload(value, u.c, u.n);
    g_cst.uploads += (uint64_t)u.n, g_cst.upload_bytes += u.bytes;
    free(u.c);
    plat_mutex_unlock(&g_coh);
    return 1;
}

// The page table of a raw-access submission is read after the submission is recorded, and is rewritten as the map moves
// until then (vk_raw.c), so every raw access of the submission reaches the map as the submission leaves it. A mirror is a
// copy of its pages, though: a bind that gave its pages to a larger backing would leave two copies of the same bytes, one
// for what the submission recorded before and one for what it records after. So a bind in such a submission never
// replaces a mirror by a larger backing. It takes the pages it needs from the mirrors it overlaps (a mirror keeps the part
// outside the bind, or is dropped when the bind covers all of it), and its own backing, an import, covers the run of
// committed pages around the bind up to the mirrors that remain: no page has two backings. The bytes a mirror gives up
// are the CPU's view in the import once every older submission that wrote them is done, which is waited for here. A
// mirror this submission already resolved where the bind takes pages, or wrote there on the GPU, is a named fatal: what
// was recorded before reaches the mirror's copy, and what follows reaches the import's.
// *rs and *re, the committed run around the pages [plo, phi) of the bind, are narrowed to what the new backing covers.
// g_lock exclusive.
static void beside(Cell *c, uint64_t plo, uint64_t phi, uint64_t *rs, uint64_t *re, uint64_t value) {
    int gave = 0;
    for (int i = 0; i < c->nimp; i++) {
        Backing *m = c->imp[i];
        if (m->kind != BACKING_MIRROR || m->gpu_hi <= *rs || m->gpu_lo >= *re) continue;
        if (m->gpu_hi <= plo) {   // below the bind: the new backing starts where the mirror ends
            if (m->gpu_hi > *rs) *rs = m->gpu_hi;
            continue;
        }
        if (m->gpu_lo >= phi) {   // above it
            if (m->gpu_lo < *re) *re = m->gpu_lo;
            continue;
        }
        uint64_t cl = m->gpu_lo > plo ? m->gpu_lo : plo, ch = m->gpu_hi < phi ? m->gpu_hi : phi;   // what the bind takes
        if (m->res_value == value && m->res_lo < ch && cl < m->res_hi)
            rt_fatal("gfx: vulkan: backing map: the mirror [%#llx, %#llx) was bound over [%#llx, %#llx) earlier in "
                     "raw-access submission %llu, and a bind over [%#llx, %#llx) now takes pages of it for an import: "
                     "what was recorded before would reach the mirror's copy of the bytes the rest reaches in the "
                     "import", (ull)m->gpu_lo, (ull)m->gpu_hi, (ull)m->res_lo, (ull)m->res_hi, (ull)value, (ull)cl,
                     (ull)ch);
        gave = 1;
        if (m->gpu_lo < plo) {   // it keeps its start; the new backing starts at the bind's first page
            g_st.mirror_bytes -= m->gpu_hi - plo;
            m->gpu_hi = plo;
            if (*rs < plo) *rs = plo;
        } else if (m->gpu_hi > phi) {   // it keeps its end
            g_st.mirror_bytes -= phi - m->gpu_lo;
            m->offset += phi - m->gpu_lo;
            m->gpu_lo = phi;
            if (*re > phi) *re = phi;
        } else {   // the bind covers it all: it goes now, and is freed once the raw-access submission is done
            defer(m);
            memmove(c->imp + i, c->imp + i + 1, (size_t)(c->nimp - i - 1) * sizeof *c->imp);
            c->nimp--;
            i--;
        }
    }
    if (!gave) return;
    changed();
    uint64_t older = raw_before(value);   // raw stores into a mirror reach guest memory when their submission is done
    if (older > gpu_sync_done()) {
        g_st.replace_waits++;
        wait_value(older);
    }
}

// ---------- the resolver
static void fill(BackingRef *out, const Backing *b, uint64_t addr) {
    out->buffer = b->mem.buffer;
    out->offset = b->offset + (addr - b->gpu_lo);
    out->address = b->mem.address + out->offset;
    out->memory = b->mem.memory;
    out->type = b->mem.type;
    out->kind = b->kind;
}

static int virtual_mapped(const Rec *r, uint64_t addr, uint64_t size) {
    for (uint64_t k = (addr - r->lo) / GFX_GPU_PAGE; k <= (addr + size - 1 - r->lo) / GFX_GPU_PAGE; k++)
        if (!(r->bits[k / 64] >> (k % 64) & 1)) return 0;
    return 1;
}

// The backing that holds [addr, addr + size) in r (not virtual), or NULL. g_lock, shared at least.
static Backing *backing_in(Rec *r, uint64_t addr, uint64_t size) {
    if (r->cls == CLS_GPU) return r->ext;
    const Cell *c = cell_get(r, addr / WINDOW, (addr + size - 1) / WINDOW, 0);
    Backing *b = c ? cell_import(c, addr) : NULL;
    return b && addr + size <= b->gpu_hi ? b : NULL;
}

static void fill_pool(Rec *r);

// The backing that holds [*addr, *addr + size) when the map holds one, *addr moved into the physical pool for a virtual
// one; the pool in *rec and the virtual pool gone through in *virt (else NULL). g_lock, shared at least.
static Backing *find(uint64_t *addr, uint64_t size, Rec **rec, Rec **virt) {
    Rec *r = rec_find(*addr, size), *v = NULL;
    *rec = *virt = NULL;
    if (r && r->cls == CLS_VIRTUAL) {
        if (!r->run || !virtual_mapped(r, *addr, size)) return NULL;
        v = r;
        *addr = (uint64_t)((int64_t)*addr + r->run_delta);
        r = r->run;
        if (*addr < r->lo || *addr + size > r->hi) return NULL;
    }
    *rec = r, *virt = v;
    return r ? backing_in(r, *addr, size) : NULL;
}

// What submission value resolved in mirror m, as a hull (beside). Only the thread recording value gets here with it.
static void note_resolved(Backing *m, uint64_t addr, uint64_t size, uint64_t value) {
    if (m->res_value != value) {
        m->res_value = value, m->res_lo = addr, m->res_hi = addr + size;
        return;
    }
    if (addr < m->res_lo) m->res_lo = addr;
    if (addr + size > m->res_hi) m->res_hi = addr + size;
}

// The backing of [addr, addr + size) when the map holds one, stamped, and a mirror brought current; g_lock, shared at
// least.
static int lookup(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    Rec *r, *v;
    Backing *b = find(&addr, size, &r, &v);
    if (!b) return 0;
    if (v) stamp(&v->last_use, value);
    stamp(&r->last_use, value), stamp(&b->last_use, value);
    if (b->kind == BACKING_MIRROR) {
        if (value) note_resolved(b, addr, size, value);
        mirror_use(b, value);
    } else if (b->kind == BACKING_IMPORT && value && size > BACKING_PROMOTE_MIN) {
        // A bind of a range larger than a uniform's reached an import: the page table's invariant imported every committed
        // page, and a bind in a raw-access submission can only import. The GPU reads an import across the bus, so the next
        // raw-access submission's begin gives it the mirror the rule would have made (promote_hot).
        __atomic_store_n(&b->hot, 1, __ATOMIC_RELAXED);
    }
    fill(out, b, addr);
    return 1;
}

// Make what [addr, addr + size) needs in submission value, or stop by name with the shape. g_lock exclusive.
static void make(uint64_t addr, uint64_t size, uint64_t value) {
    Rec *r = rec_find(addr, size);
    if (!r) {
        Rec *s = rec_find(addr, 1);
        if (s)
            rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) runs past the end of its %s pool [%#llx, "
                     "%#llx) by %#llx bytes: it would need two backings", (ull)addr, (ull)(addr + size), cls_name(s),
                     (ull)s->lo, (ull)s->hi, (ull)(addr + size - s->hi));
        rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) lies in no pool the GPU can reach", (ull)addr,
                 (ull)(addr + size));
    }
    if (r->cls == CLS_VIRTUAL) {
        if (r->run_lost)
            rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) in virtual pool [%#llx, %#llx), whose "
                     "physical pool was finalized while its pages were mapped", (ull)addr, (ull)(addr + size),
                     (ull)r->lo, (ull)r->hi);
        if (!r->run || !virtual_mapped(r, addr, size))
            rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) in virtual pool [%#llx, %#llx) reaches pages "
                     "no mapping request mapped (%llu of %llu pages mapped): it would need two backings", (ull)addr,
                     (ull)(addr + size), (ull)r->lo, (ull)r->hi, (ull)r->mapped, (ull)r->pages);
        addr = (uint64_t)((int64_t)addr + r->run_delta);
        r = r->run;
    }
    BackingAsk ask = {addr, size, r->flags};
    int kind = backing_rule(&ask);
    if (kind == BACKING_EXTENT) {
        if (!r->ext) r->ext = extent_new(r), changed();
        return;
    }
    if (kind == BACKING_MIRROR && !has_mirrors())
        rt_fatal("gfx: vulkan: backing map: the mirror hook chose a mirror for [%#llx, %#llx), and the device side "
                 "gives no mirror operations", (ull)addr, (ull)(addr + size));
    uint64_t plo = down(addr, HPAGE), phi = up(addr + size, HPAGE);
    if (!rt_mem_commit_io(plo, phi - plo, 0))
        rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) in pool [%#llx, %#llx) covers memory the guest "
                 "has not mapped", (ull)addr, (ull)(addr + size), (ull)r->lo, (ull)r->hi);
    Cell *c = cell_get(r, addr / WINDOW, (addr + size - 1) / WINDOW, 1);
    uint64_t rs, re;
    committed_run(c->lo, c->hi, plo, phi, &rs, &re);
    // In the raw-access submission being recorded, the page table is rewritten at its end to map what this bind makes, so
    // its raw accesses reach it however early they were recorded. That is the same copy as the earlier binds' only where
    // the new backing aliases the guest's memory, as an import does: a mirror made here would be filled in the prologue,
    // before the submission's own writes through the imports it replaces. A later submission's bind can still make it.
    if (value && raw_last() == value) {
        for (int i = 0; i < c->nimp; i++) {
            Backing *m = c->imp[i];
            if (m->kind != BACKING_MIRROR || m->res_value != value ||
                !overlaps(plo, phi, m->gpu_lo, m->gpu_hi) ||
                !overlaps(plo, phi, m->res_lo, m->res_hi))
                continue;
            if (grow_taken(r, c, m, plo, phi, value)) {
                fill_pool(r);   // a newly committed host chunk can extend beyond the pages of the bind
                return;
            }
            break;
        }
        if (kind == BACKING_MIRROR) kind = BACKING_IMPORT;
        beside(c, plo, phi, &rs, &re, value);
    }
    replace_wait(c, rs, re, value, kind == BACKING_MIRROR);
    Backing *b = kind == BACKING_MIRROR ? mirror_admit(r, rs, re) : NULL;
    if (!b) {
        char why[96];
        snprintf(why, sizeof why, "a bind of [%#llx, %#llx)", (ull)addr, (ull)(addr + size));
        b = import_new(rs, re, plo, why);
    }
    cell_put(c, rs, re, b);
}

void backing_resolve(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    if (!size) size = 1;
    if (addr + size < addr) rt_fatal("gfx: vulkan: backing map: a bind of %#llx+%#llx wraps", (ull)addr, (ull)size);
    __atomic_fetch_add(&g_resolves, 1, __ATOMIC_RELAXED);
    plat_lock_shared(&g_lock);
    int hit = lookup(addr, size, value, out);
    plat_unlock_shared(&g_lock);
    if (hit) return;
    __atomic_fetch_add(&g_misses, 1, __ATOMIC_RELAXED);
    plat_lock(&g_lock);
    make(addr, size, value);
    if (!lookup(addr, size, value, out))
        rt_fatal("gfx: vulkan: backing map: [%#llx, %#llx) has no backing after one was made", (ull)addr,
                 (ull)(addr + size));
    plat_unlock(&g_lock);
    bound();
}

// ---------- coherence (backing.h, Mirrors)
void backing_gpu_wrote(uint64_t addr, uint64_t size, uint64_t value) {
    if (!size) return;
    if (addr + size < addr || !value)
        rt_fatal("gfx: vulkan: backing map: a GPU write declared over %#llx+%#llx outside a submission (value %llu)",
                 (ull)addr, (ull)size, (ull)value);
    plat_lock_shared(&g_lock);
    Rec *r, *v;
    uint64_t a = addr;
    Backing *b = find(&a, size, &r, &v);
    if (!b || used_by(b) != value)
        rt_fatal("gfx: vulkan: backing map: a GPU write declared over [%#llx, %#llx) in submission %llu, which no "
                 "resolve of that submission backed", (ull)addr, (ull)(addr + size), (ull)value);
    stamp(&b->written, value);
    if (b->kind == BACKING_MIRROR) {
        plat_mutex_lock(&g_coh);
        g_pend = grow(g_pend, &g_pend_cap, g_npend + 1, sizeof *g_pend, "the GPU writes to write back");
        g_pend[g_npend++] = (Pending){value, a, a + size, b};
        plat_mutex_unlock(&g_coh);
    }
    plat_unlock_shared(&g_lock);
}

void backing_sync_mirrors(uint64_t value) {
    if (!g_open || !value) return;
    // Not a resolve: it leaves each mirror's last_use alone, since a bind of the submission that took pages from one in
    // beside() asks whether the submission resolved it. A mirror's retirement waits for the raw-access submission anyway
    // (reach), which is the reason a stamp would have had.
    plat_lock_shared(&g_lock);
    for (int i = 0; i < g_nmirrors; i++) mirror_use(g_mirrors[i], value);
    plat_unlock_shared(&g_lock);
}

// The write-back of one submission: its GPU writes into mirrors, read back at its end, and where they go.
typedef struct WbItem { uint64_t lo, size; const void *host; } WbItem;
typedef struct Wb { int n; WbItem item[]; } Wb;

// The completion thread, once the device reached value and before value is done: the bytes go into guest memory (a
// page the guest no longer maps is left out and counted), are stamped as written, and leave the pending list.
static void writeback(void *arg, uint64_t value) {
    Wb *w = arg;
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < w->n; i++) {
        const WbItem *it = &w->item[i];
        if (!rt_mem_commit_io(it->lo, it->size, 1)) {
            if (!g_cst.writebacks_skipped++)
                rt_log("gfx: vulkan: backing map: a GPU write of [%#llx, %#llx) by submission %llu lands in memory the "
                       "guest no longer maps: left out", (ull)it->lo, (ull)(it->lo + it->size), (ull)value);
            continue;
        }
        memcpy((void *)(uintptr_t)it->lo, it->host, it->size);
        pagegen_mark(it->lo, it->lo + it->size);
        g_cst.writebacks++, g_cst.writeback_bytes += it->size;
    }
    int k = 0;
    for (int i = 0; i < g_npend; i++)
        if (g_pend[i].value != value) g_pend[k++] = g_pend[i];
    g_npend = k;
    plat_mutex_unlock(&g_coh);
    free(w);
}

void backing_submit_end(uint64_t value) {
    if (!g_open || !has_mirrors()) return;
    plat_mutex_lock(&g_coh);
    int n = 0;
    for (int i = 0; i < g_npend; i++) n += g_pend[i].value == value;
    if (!n) {
        plat_mutex_unlock(&g_coh);
        return;
    }
    BackingCopy *c = calloc((size_t)n, sizeof *c);
    Wb *w = malloc(sizeof *w + (size_t)n * sizeof *w->item);
    if (!c || !w) rt_fatal("gfx: vulkan: backing map: out of memory for %d write-backs", n);
    w->n = 0;
    for (int i = 0; i < g_npend; i++) {
        const Pending *p = &g_pend[i];
        if (p->value != value) continue;
        c[w->n] = (BackingCopy){&p->m->mem, p->m->offset + (p->lo - p->m->gpu_lo), p->hi - p->lo, NULL};
        w->item[w->n++] = (WbItem){p->lo, p->hi - p->lo, NULL};
    }
    g_ops.readback(value, c, n);
    for (int i = 0; i < n; i++) w->item[i].host = c[i].host;
    gpu_sync_effect(writeback, w);
    plat_mutex_unlock(&g_coh);
    free(c);
}

void backing_guest_written(uint64_t addr, uint64_t size) {
    if (!size || addr >= RT_ADDRESS_LIMIT) return;   // the GPU window: no CPU wrote it
    pagegen_mark(addr, addr + size);
}

__attribute__((noreturn)) static void not_written_back(const Pending *p, const char *when) {
    rt_fatal("gfx: vulkan: backing map: the GPU write of [%#llx, %#llx) by submission %llu into a mirror was not "
             "written back by the time its value was done (%s)", (ull)p->lo, (ull)p->hi, (ull)p->value, when);
}

void backing_guest_read(uint64_t addr, uint64_t size) {
    if (!size) size = 1;
    uint64_t done = gpu_sync_done();   // read first: a value done now had its write-back run before it was published
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend; i++) {
        if (g_pend[i].hi <= addr || g_pend[i].lo >= addr + size) continue;
        if (g_pend[i].value <= done) not_written_back(&g_pend[i], "a CPU read of mapped memory");
        g_cst.reads_in_flight++;
    }
    plat_mutex_unlock(&g_coh);
}

void backing_cpu_synced(uint64_t fence) {
    uint64_t done = gpu_sync_done();
    if (fence > done) fence = done;
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend; i++)
        if (g_pend[i].value <= fence) not_written_back(&g_pend[i], "a CPU wait for a fence");
    plat_mutex_unlock(&g_coh);
}

// ---------- the commit epoch
// Every backing of a cell again: retired when a page under it is no longer committed; an import is also grown to the run
// of committed pages around it inside its cell, which may join its neighbours, up to a mirror (a mirror is not grown: a
// bind outside it makes a new backing). g_lock exclusive.
static void rescan_cell(Cell *c) {
    for (int i = 0; i < c->nimp;) {
        Backing *o = c->imp[i];
        uint64_t a = o->gpu_lo;
        int whole = 1;
        while (a < o->gpu_hi && whole) {
            int st;
            uint64_t n = rt_mem_commit_run(a, o->gpu_hi - a, &st);
            whole = n != UINT64_MAX && n && st;
            a += whole ? n : 0;
        }
        if (!whole) {
            defer(o);
            memmove(c->imp + i, c->imp + i + 1, (size_t)(c->nimp - i - 1) * sizeof *c->imp);
            c->nimp--;
            if (o->kind == BACKING_MIRROR) g_st.mirrors_stale++;
            else g_st.imports_stale++;
            continue;
        }
        if (o->kind == BACKING_MIRROR) {
            i++;
            continue;
        }
        uint64_t rs, re;
        committed_run(c->lo, c->hi, o->gpu_lo, o->gpu_hi, &rs, &re);
        for (int j = 0; j < c->nimp; j++) {   // up to the mirrors beside it
            const Backing *m = c->imp[j];
            if (m->kind != BACKING_MIRROR) continue;
            if (m->gpu_hi <= o->gpu_lo && m->gpu_hi > rs) rs = m->gpu_hi;
            if (m->gpu_lo >= o->gpu_hi && m->gpu_lo < re) re = m->gpu_lo;
        }
        if (rs == o->gpu_lo && re == o->gpu_hi) {
            i++;
            continue;
        }
        char why[96];
        snprintf(why, sizeof why, "chunks committed beside the import [%#llx, %#llx)", (ull)o->gpu_lo, (ull)o->gpu_hi);
        Backing *b = import_new(rs, re, o->gpu_lo, why);
        cell_put(c, rs, re, b);
        i = 0;   // the list changed; imports already grown are whole runs now and pass at once
    }
}

void backing_sync(void) {
    if (!g_open) return;
    uint64_t e = rt_mem_ready() ? rt_mem_commit_epoch() : 0;
    Freeing f = {0};
    plat_lock(&g_lock);
    g_st.syncs++;
    if (e != g_seen_epoch) {
        g_seen_epoch = e;   // read before the scan: a commit during it moves the epoch again
        g_st.sync_rescans++;
        for (int i = 0; i < g_index.n; i++) {
            Rec *r = (Rec *)g_index.e[i].key;
            for (int j = 0; r->cls == CLS_CPU && j < r->ncells; j++) rescan_cell(&r->cells[j]);
        }
    }
    retire(&f, 0);
    plat_unlock(&g_lock);
    free_all(&f);
    bound();
}

// ---------- the raw path (backing.h, The raw path)
// The backing that holds page a of r (not virtual), a mirror before an import when the page is in both (two cells), or
// NULL; *next is the lowest start of a backing above a, UINT64_MAX for none. g_lock, shared at least.
static const Backing *page_backing(const Rec *r, uint64_t a, uint64_t *next) {
    uint64_t nx = UINT64_MAX;
    const Backing *best = NULL;
    if (r->cls == CLS_GPU) {
        best = r->ext && a >= r->ext->gpu_lo && a < r->ext->gpu_hi ? r->ext : NULL;
    } else {
        for (int i = 0; i < r->ncells; i++)
            for (int j = 0; j < r->cells[i].nimp; j++) {
                const Backing *b = r->cells[i].imp[j];
                if (b->gpu_lo <= a && a < b->gpu_hi) {
                    if (!best || (b->kind == BACKING_MIRROR && best->kind != BACKING_MIRROR)) best = b;
                } else if (b->gpu_lo > a && b->gpu_lo < nx) {
                    nx = b->gpu_lo;
                }
            }
    }
    if (next) *next = nx;
    return best;
}

// Every committed page of r backed: an import over each run of committed pages no backing covers, cut to the import
// windows (one cell per window, as a bind inside one window makes). Imports, not mirrors: the invariant can reach the
// whole heap, and a mirror per run would take VRAM up to the admission's limit, which the extents of pools made later
// need (D14 gives them no fallback). g_lock exclusive.
static void fill_pool(Rec *r) {
    for (uint64_t a = r->lo; a < r->hi;) {
        int c;
        uint64_t n = rt_mem_commit_run(a, r->hi - a, &c);
        if (n == UINT64_MAX || !n)
            rt_fatal("gfx: vulkan: backing map: %#llx, in CPU-visible pool [%#llx, %#llx), is not guest memory", (ull)a,
                     (ull)r->lo, (ull)r->hi);
        uint64_t end = a + n;
        while (c && a < end) {
            uint64_t next;
            const Backing *b = page_backing(r, a, &next);
            if (b) {
                a = b->gpu_hi < end ? b->gpu_hi : end;
                continue;
            }
            uint64_t k = a / WINDOW, e = end;
            if (next < e) e = next;
            Cell *cell = cell_get(r, k, k, 1);
            if (cell->hi < e) e = cell->hi;
            cell_put(cell, a, e, import_new(a, e, a, "the page table's invariant (every committed page mapped)"));
            g_st.raw_fill_imports++, g_st.raw_fill_bytes += e - a;
            a = e;
        }
        a = end;
    }
}

// The imports a bind larger than a uniform's reached (lookup flagged them) become mirrors, as the rule gives a bind in any
// other submission on a discrete device: the invariant above imported every committed page, and an import is read across the
// bus (the raw-address study: 22.9 times slower than VRAM), which a compute program that works on its buffers feels in every
// access. It is done here, at the begin of a raw-access submission, before its raw-access stages and the binds that follow
// them, so that those binds and the page table reach the one copy. A bind of the submission before its first raw-access stage
// that only read the import leaves no hazard: the commands it recorded read the import's bytes, the CPU's view, which the
// mirror's first copy repeats, and the import lives until they are done. One the submission wrote on the GPU stays an import
// until the next submission, as replacing a backing the submission wrote is a named fatal (replace_wait). The older
// submissions that reached the import are waited for there too, so what they wrote through it is in guest memory when the
// mirror takes its copy, and backing_sync_mirrors brings the mirror current right after. A mirror the admission refuses
// leaves the import as it was and ends the pass (one refusal a begin, whatever the number of hot imports). g_lock exclusive.
static void promote_hot(uint64_t value) {
    if (!has_mirrors() || !g_mirror_hook) return;
    for (int i = 0; i < g_index.n; i++) {
        Rec *r = (Rec *)g_index.e[i].key;
        if (r->cls != CLS_CPU) continue;
        for (int j = 0; j < r->ncells; j++) {
            Cell *c = &r->cells[j];
            for (int k = 0; k < c->nimp; k++) {
                Backing *b = c->imp[k];
                if (b->kind != BACKING_IMPORT || !__atomic_load_n(&b->hot, __ATOMIC_RELAXED)) continue;
                if (__atomic_load_n(&b->written, __ATOMIC_ACQUIRE) == value) continue;   // this submission wrote it: next time
                uint64_t lo = b->gpu_lo, hi = b->gpu_hi;
                BackingAsk ask = {lo, hi - lo, r->flags};
                if (backing_rule(&ask) != BACKING_MIRROR) {
                    __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);   // the rule says import: it stays one
                    continue;
                }
                if (g_st.mirror_bytes + up(hi - lo, BACKING_EXTENT_ALIGN) > BACKING_PROMOTE_MIRROR_MAX) {
                    // the raw path's write-back holds the pages the stores of one submission reach only while every mirror fits it
                    __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);
                    if (!r->mirror_logged) {
                        r->mirror_logged = 1;
                        rt_log("gfx: vulkan: backing map: the mirror of [%#llx, %#llx) (%llu KiB) in pool [%#llx, %#llx) is not made: "
                               "the mirrors would pass %llu MiB, which the raw write-back holds; imported", (ull)lo, (ull)hi,
                               (ull)((hi - lo) >> 10), (ull)r->lo, (ull)r->hi, (ull)(BACKING_PROMOTE_MIRROR_MAX >> 20));
                    }
                    continue;
                }
                Backing *m = mirror_admit(r, lo, hi);
                if (!m) {   // no room (or no write watch): it stays an import, and the rest wait for the next begin
                    __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);
                    return;
                }
                replace_wait(c, lo, hi, value, 1);
                cell_put(c, lo, hi, m);   // the import is deferred and the mirror takes its place in the cell
                g_st.raw_promoted++, g_st.raw_promote_bytes += hi - lo;
            }
        }
    }
}

void backing_raw_begin(uint64_t value) {
    if (!g_open || !value)
        rt_fatal("gfx: vulkan: backing map: a raw-access submission (value %llu) %s", (ull)value,
                 g_open ? "outside a submission" : "before the map is open");
    uint64_t e = rt_mem_ready() ? rt_mem_commit_epoch() : 0;
    plat_lock(&g_lock);
    uint64_t before = g_raw_last;
    if (before && before < value) __atomic_store_n(&g_raw_prev, before, __ATOMIC_RELEASE);   // the one before this
    stamp(&g_raw_last, value);
    g_st.raw_begins++;
    if (!g_filled || e != g_fill_epoch || g_pools_version != g_fill_pools) {
        g_filled = 1, g_fill_epoch = e, g_fill_pools = g_pools_version;   // read before the pass, as backing_sync does
        g_st.raw_passes++;
        for (int i = 0; i < g_index.n; i++) {
            Rec *r = (Rec *)g_index.e[i].key;
            if (r->cls == CLS_GPU && !r->ext) r->ext = extent_new(r), changed(), g_st.raw_fill_extents++;
            else if (r->cls == CLS_CPU) fill_pool(r);
        }
    }
    promote_hot(value);
    plat_unlock(&g_lock);
    bound();
    backing_sync_mirrors(value);
}

uint64_t backing_version(void) { return __atomic_load_n(&g_version, __ATOMIC_ACQUIRE); }

int backing_pool_span(uint64_t *lo, uint64_t *hi) {
    int any = 0;
    *lo = *hi = 0;
    plat_lock_shared(&g_lock);
    for (int i = 0; i < g_index.n; i++) {
        const Rec *r = (const Rec *)g_index.e[i].key;
        if (!any || r->lo < *lo) *lo = r->lo;
        if (!any || r->hi > *hi) *hi = r->hi;
        any = 1;
    }
    plat_unlock_shared(&g_lock);
    return any;
}

typedef struct RunList { BackingRun *r; int n, cap; } RunList;

static void run_add(RunList *l, uint64_t lo, uint64_t hi, uint64_t address, uint64_t guest, int kind) {
    BackingRun *p = l->n ? &l->r[l->n - 1] : NULL;
    if (p && p->hi == lo && p->kind == kind && p->address + (p->hi - p->lo) == address &&
        p->guest + (p->hi - p->lo) == guest) {
        p->hi = hi;
        return;
    }
    l->r = grow(l->r, &l->cap, l->n + 1, sizeof *l->r, "the raw path's runs");
    l->r[l->n++] = (BackingRun){lo, hi, address, guest, kind};
}

// [lo, hi) mapped to b's bytes of guest address guest on.
static void run_of(RunList *l, const Backing *b, uint64_t lo, uint64_t hi, uint64_t guest) {
    run_add(l, lo, hi, b->mem.address + b->offset + (guest - b->gpu_lo), guest, b->kind);
}

// The pages of virtual pool v mapped to a physical pool that has backings, page by page of the physical pool.
static void virtual_runs(RunList *l, const Rec *v) {
    const Rec *p = v->run;
    if (!p) return;
    for (uint64_t k = 0; k < v->pages; k++) {
        if (!(v->bits[k / 64] >> (k % 64) & 1)) continue;
        uint64_t va = v->lo + k * GFX_GPU_PAGE, pa = (uint64_t)((int64_t)va + v->run_delta);
        for (uint64_t o = 0; o < GFX_GPU_PAGE; o += HPAGE) {
            if (pa + o < p->lo || pa + o >= p->hi) continue;
            const Backing *b = page_backing(p, pa + o, NULL);
            if (b) run_of(l, b, va + o, va + o + HPAGE, pa + o);
        }
    }
}

int backing_raw_runs(BackingRun **out, uint64_t *version) {
    RunList l = {0};
    plat_lock_shared(&g_lock);
    *version = backing_version();
    for (int pass = 0; pass < 2; pass++)   // a mirror's pages after an import's, so a page in both maps to the mirror
        for (int i = 0; i < g_index.n; i++) {
            const Rec *r = (const Rec *)g_index.e[i].key;
            if (r->cls == CLS_VIRTUAL) {
                if (!pass && !r->run_lost) virtual_runs(&l, r);
                continue;
            }
            if (r->cls == CLS_GPU) {
                uint64_t hi = r->ext && r->ext->gpu_hi < r->hi ? r->ext->gpu_hi : r->hi;
                if (!pass && r->ext) run_of(&l, r->ext, r->lo, hi, r->lo);
                continue;
            }
            for (int j = 0; j < r->ncells; j++)
                for (int m = 0; m < r->cells[j].nimp; m++) {
                    const Backing *b = r->cells[j].imp[m];
                    if ((b->kind == BACKING_MIRROR) == pass) run_of(&l, b, b->gpu_lo, b->gpu_hi, b->gpu_lo);
                }
        }
    plat_unlock_shared(&g_lock);
    *out = l.r;
    return l.n;
}

// ---------- decisions and counters
void backing_page(uint64_t addr, BackingPage *out) {
    *out = (BackingPage){BACKING_NONE, 0, addr};
    plat_lock_shared(&g_lock);
    Rec *r = rec_find(addr, 1);
    if (r && r->cls == CLS_VIRTUAL) {
        if (r->run && virtual_mapped(r, addr, 1)) out->gpu = (uint64_t)((int64_t)addr + r->run_delta);
        r = r->run && virtual_mapped(r, addr, 1) ? r->run : NULL;
    }
    const Backing *b = r && r->cls == CLS_GPU ? r->ext : NULL;
    for (int i = 0; r && r->cls == CLS_CPU && i < r->ncells && !b; i++) b = cell_import(&r->cells[i], out->gpu);
    if (b) {
        out->kind = b->kind;
        out->address = b->mem.address + b->offset + (out->gpu - b->gpu_lo);
    }
    plat_unlock_shared(&g_lock);
}

int backing_extent_pool(uint64_t addr, uint64_t size) {
    if (!size) size = 1;
    plat_lock_shared(&g_lock);
    const Rec *r = rec_find(addr, size);
    if (r && r->cls == CLS_VIRTUAL && r->run && virtual_mapped(r, addr, size)) r = r->run;
    int extent = !(r && r->cls == CLS_CPU);
    plat_unlock_shared(&g_lock);
    return extent;
}

void backing_stats(BackingStats *out) {
    plat_lock_shared(&g_lock);
    *out = g_st;
    out->spare_blocks = (uint64_t)spare_count();
    plat_unlock_shared(&g_lock);
    plat_mutex_lock(&g_coh);
    out->mirror_syncs = g_cst.mirror_syncs, out->uploads = g_cst.uploads, out->upload_bytes = g_cst.upload_bytes;
    out->writebacks = g_cst.writebacks, out->writeback_bytes = g_cst.writeback_bytes;
    out->writebacks_skipped = g_cst.writebacks_skipped, out->reads_in_flight = g_cst.reads_in_flight;
    out->pending = (uint64_t)g_npend, out->raw_waits = g_cst.raw_waits;
    plat_mutex_unlock(&g_coh);
    out->resolves = __atomic_load_n(&g_resolves, __ATOMIC_RELAXED);
    out->resolve_misses = __atomic_load_n(&g_misses, __ATOMIC_RELAXED);
}

// ---------- open and close
void backing_open(const BackingOps *ops) {
    if (g_open) rt_fatal("gfx: vulkan: backing map: opened twice");
    if (!ops || !ops->block_new || !ops->import || !ops->free)
        rt_fatal("gfx: vulkan: backing map: an operation is missing from the table");
    int mirror_ops = !!ops->vram + !!ops->upload + !!ops->readback;
    if (mirror_ops && mirror_ops != 3)
        rt_fatal("gfx: vulkan: backing map: the mirror operations are given in part (%d of 3)", mirror_ops);
    g_ops = *ops;
    memset(&g_st, 0, sizeof g_st);
    plat_mutex_lock(&g_coh);
    memset(&g_cst, 0, sizeof g_cst);
    g_npend = 0;
    plat_mutex_unlock(&g_coh);
    g_resolves = g_misses = 0;
    g_seen_epoch = rt_mem_ready() ? rt_mem_commit_epoch() : 0;
    g_raw_last = g_raw_prev = 0, g_filled = 0, g_pools_version = g_fill_pools = g_fill_epoch = 0;
    g_open = 1;
    g_bt_stop = 0, g_bt_want = 1, g_bt_running = 1;
    g_bt = plat_thread_start(bt_main, NULL, BLOCK_THREAD_STACK);
    if (!g_bt) rt_fatal("gfx: vulkan: backing map: the block thread did not start");
}

int backing_is_open(void) { return g_open; }

void backing_close(void) {
    if (!g_open) return;
    plat_mutex_lock(&g_bt_lock);
    g_bt_stop = 1;
    plat_cond_signal(&g_bt_cond);
    while (g_bt_running) plat_cond_wait(&g_bt_cond, &g_bt_lock);
    plat_mutex_unlock(&g_bt_lock);
    plat_thread_free(g_bt);
    g_bt = NULL;
    Freeing f = {0};
    plat_lock(&g_lock);
    while (g_index.n) rec_end((Rec *)g_index.e[g_index.n - 1].key);
    retire(&f, 1);
    while (g_nblocks) {   // the spares
        to_free(&f, &g_blocks[0]->mem);
        block_drop(g_blocks[0]);
    }
    g_nmirrors = 0;   // every one was retired with its pool
    g_mirror_hook = NULL;
    g_open = 0;
    plat_unlock(&g_lock);
    plat_mutex_lock(&g_coh);
    g_npend = 0;   // every submission is done (gpu_sync_finish came first): their write-backs ran
    plat_mutex_unlock(&g_coh);
    free_all(&f);
    if (g_ops.close) g_ops.close();
}
