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
// Mirrors (backing.h, Mirrors): VRAM parts of the blocks, as extents are, over the run an import would cover, with
// BACKING_MIRROR_GUARD bytes reserved at each end so that one a raw-access submission has bound can grow in place
// (grow_taken). A mirror is brought current at its first resolve in each submission (mirror_use) from the page
// generations (oot2/pagegen.h); the GPU writes declared into it are read back at the submission's end and written into
// guest memory by an effect on the completion thread (writeback), which runs once the device reached the value and
// before the value is done.
//
// The mirrors' copies and write-backs are adapted from DKB Recomp runtime/vk_memory.c at 9d8fa02 (GPL-3.0-or-later):
// the admission rule (used + size + budget / 8 within the budget), the generation a mirror is current up to, the copy of
// the pages stamped after it, a GPU write noted at record time and copied back into guest memory at its fence, and the
// copies into a mirror that leave out the bytes of GPU writes not copied back yet. Changed here: a mirror covers the
// committed run around its bind instead of the pool, needs no import (its copies go through staging), is brought current
// at its first resolve in a submission rather than by 2 MiB snapshots per submission, and a new backing over older ones
// waits for the submissions that used them.
#include <stdarg.h>
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

// snprintf for a line built piece by piece: returns the characters it stored (at most size - 1), never the length it
// wanted, so that `at += sn(buf + at, sizeof buf - at, ...)` cannot pass the end of the buffer and size - at cannot wrap.
static size_t sn(char *buf, size_t size, const char *fmt, ...) {
    if (!size) return 0;
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf, size, fmt, ap);
    va_end(ap);
    if (r < 0) return 0;
    return (size_t)r < size ? (size_t)r : size - 1;
}

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
    // An import's (and a mirror's): the same hull, noted by every resolve and every declared GPU write of the submission
    // (note_resolved), and the count of them. Empty is res_lo == UINT64_MAX with res_hi == 0, which a new backing starts
    // with; the raw path's refusal reads it for mirrors only (the straddle dump prints it for imports).
    uint64_t res_value, res_lo, res_hi, res_n;
    // An import's: some resolve of submission res_value stated no extent (size 0, which the resolver counts as 1), so the
    // hull above is not what that submission reached (import_gives).
    int res_loose;
    // An import's: the range its host import covers. gpu_lo and gpu_hi shrink when a growth trims it (grow_taken); the
    // host pages it gave away stay under its buffer until it is freed, and a decommit of one retires it (rescan_cell).
    uint64_t span_lo, span_hi;
    // Where the backing, or its current range, came from (diag_born): what made it, for which submission, and the range
    // it had before the cut or growth that gave it its present one. Written under the exclusive lock, read by the dump.
    uint64_t born_value, born_lo, born_hi;
    int born_by;
    // The first record born() made for the backing and never overwritten: how it started, whatever cut or grew it since.
    uint64_t born0_value, born0_lo, born0_hi;
    int born0_by;
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

// The submission whose bind or begin is making backings now (0: none): fill_pool and rescan_cell take it for the birth
// record of their imports, and cell_put for what an import takes over from the ones it replaces. g_lock exclusive.
static uint64_t g_born_value;
static int g_grow_cap_logged;   // a growth took the mirrors past BACKING_PROMOTE_MIRROR_MAX and said so (once a run)
static int g_union_on = 1;      // the union pass of backing_raw_begin runs (backing_set_union)
static uint64_t g_adj_version;  // the map's version at the last union pass that left no chain alone for its value (g_lock exclusive)
// The chains the pass left alone, by the place and the reason, so that a chain left at every begin is counted and written
// to the ring once, until it changes (a member comes or goes, its reason changes). g_lock exclusive.
typedef struct { const Rec *r; uint64_t k1, k2, lo, hi; int why, seen; } LeftChain;
enum { LEFT_MAX = 64 };
static LeftChain g_left[LEFT_MAX];
static int g_nleft;
static uint64_t g_begin_now;    // the value whose backing_raw_begin is before its stamp (0: none): its events are after the begin

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

// The mirrors' total once a backing entered the map or a growth settled, against the cap a promotion keeps: the highest it
// reached and the most it stood above the cap. Not called by the dry evaluation of the dump. g_lock exclusive.
static void mirror_note(void) {
    if (g_st.mirror_bytes > g_st.mirror_bytes_peak) g_st.mirror_bytes_peak = g_st.mirror_bytes;
    if (g_st.mirror_bytes > BACKING_PROMOTE_MIRROR_MAX &&
        g_st.mirror_bytes - BACKING_PROMOTE_MIRROR_MAX > g_st.mirror_overshoot_peak)
        g_st.mirror_overshoot_peak = g_st.mirror_bytes - BACKING_PROMOTE_MIRROR_MAX;
}

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
                   .offset = at + BACKING_MIRROR_GUARD, .block = blk, .part = size, .part_at = at, .fresh = 1,
                   .res_lo = UINT64_MAX};
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
    *b = (Backing){.kind = BACKING_IMPORT, .gpu_lo = lo, .gpu_hi = hi, .mem = m, .res_lo = UINT64_MAX, .span_lo = lo,
                   .span_hi = hi};
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

// The import b takes the place of the import o: what the submission being recorded (g_born_value) knows of o goes to b, so
// that a later growth sees one hull and one write record for the bytes where two backings stood. b takes the newest
// declared GPU write of o, and the hull of what that submission resolved in o when o holds the hull of that submission
// and no other (a hull of an older value says nothing about this one). Not taken: the stamp of the last use (b is stamped
// by the resolve that made it), the hot flag and the birth. g_lock exclusive.
static void heir_take(Backing *b, const Backing *o) {
    if (b->kind != BACKING_IMPORT || o->kind != BACKING_IMPORT) return;
    int carried = 0;
    uint64_t w = __atomic_load_n(&o->written, __ATOMIC_ACQUIRE);
    if (w) {
        stamp(&b->written, w);
        g_st.raw_heir_written++;
        carried = 1;
    }
    if (g_born_value && o->res_value == g_born_value && o->res_n && o->res_lo < o->res_hi) {
        if (b->res_value != g_born_value) b->res_value = g_born_value, b->res_lo = UINT64_MAX, b->res_hi = 0, b->res_n = 0, b->res_loose = 0;
        if (o->res_lo < b->res_lo) b->res_lo = o->res_lo;
        if (o->res_hi > b->res_hi) b->res_hi = o->res_hi;
        b->res_n += o->res_n;
        b->res_loose |= o->res_loose;
        carried = 1;
    }
    g_st.raw_heirs += (uint64_t)carried;
}

// Put the import b over [rs, re) in c in place of every import inside that run, which are deferred. g_lock exclusive.
static void cell_put(Cell *c, uint64_t rs, uint64_t re, Backing *b) {
    int k = 0, at = -1, replaced = 0;
    for (int i = 0; i < c->nimp; i++) {
        Backing *o = c->imp[i];
        if (o->gpu_hi > rs && o->gpu_lo < re) {
            heir_take(b, o);
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
    mirror_note();
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
    // A new mirror holds nothing: every page goes in. An older one holds, besides every copied CPU write, the bytes of
    // the GPU writes written back into guest memory (pagegen_mark_gpu): those pages are the same bytes in both places
    // until a later stamp, so only the pages the CPU wrote since come in. A mirror that replaced another is new, so it
    // never takes a page for GPU-current on the word of a write-back made into the one it replaced.
    if (m->fresh) runs_push(&runs, m->gpu_lo, m->gpu_hi);
    else pagegen_runs_cpu(m->gpu_lo, m->gpu_hi, m->synced, runs_push, &runs);
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
    if (!pagegen_runs_cpu(m->gpu_lo, m->gpu_hi, m->synced, no_run, NULL)) return;   // a GPU-current page is not copied
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

// What a growth of a mirror may take of an import that lies on the pages it adds. An import no bind of the submission
// reached is deferred whole (its other pages get an import again). One the submission reached stays: a command recorded
// earlier holds its buffer, so it is trimmed in place instead (same buffer, same stamps, hull and write record; only its
// start moves up), and that is exact only where nothing the submission recorded can reach the pages it gives away:
//   S1 the submission declared no GPU write in it (a write into the pages would have to move to the mirror),
//   S2 the submission's hull in it exists and states its extent (res_loose says a resolve did not),
//   S3 the pages, rounded to host pages, do not overlap that hull,
//   and the pages lie below the hull: a resolve reaches upward past its declared size (a stream of size 0, an indirect
//   index buffer), never downward, except by a negative base vertex, which vk_replay.c turns into a size-1 resolve of the
//   stream's start, i.e. loose. So only a tail growth, whose pages are the import's lowest, can be given.
// Pure: no store, no counter, no log (the dump evaluates it too). glo and ghi are the pages of the import the growth takes.
enum { GI_OK, GI_NEWER, GI_WRITTEN, GI_NOHULL, GI_LOOSE, GI_HULL, GI_UP, GI_N };
_Static_assert(GI_N == BACKING_GROW_RULES, "the counters and the rule's answers are the same list");
static const char *const k_gi[GI_N] = {"ok", "newer", "written", "no_hull", "loose", "hull", "up"};
static const char *const k_gi_why[GI_N] = {
    "", "", "was written on the GPU by this submission", "has no hull of this submission",
    "was reached by a resolve whose extent is not stated",
    "", "lies above the part of it this submission reached, and how far a resolve reaches upward is not bounded"};

static int import_gives(const Backing *b, uint64_t glo, uint64_t ghi, uint64_t value) {
    uint64_t u = used_by(b);
    if (u < value) return GI_OK;   // no bind of this submission reached it
    if (u > value) return GI_NEWER;
    if (__atomic_load_n(&b->written, __ATOMIC_ACQUIRE) >= value) return GI_WRITTEN;   // S1
    if (b->res_value != value || !b->res_n || b->res_lo >= b->res_hi) return GI_NOHULL;   // S2, emptiness first
    if (b->res_loose) return GI_LOOSE;
    uint64_t hlo = down(b->res_lo, HPAGE), hhi = up(b->res_hi, HPAGE);   // rounded after the tests above
    if (overlaps(glo, ghi, hlo, hhi)) return GI_HULL;   // S3
    return ghi <= hlo ? GI_OK : GI_UP;
}

// ---------- what the raw path's stop reports
// The refusal in beside() names one import and says nothing of which binds of the submission reached which backing, nor
// where the backings it names came from. This block keeps that (a birth record per backing, the hull of what each
// resolve reached in an import as in a mirror, a ring of the binds), and prints it before the stop. Nothing here decides
// anything: the fields and the ring are written, and read only by the dump, which runs just before the unchanged fatal.
// The dump's one effect is the write watch's scan in guard G8, which resets the marks it reports: it runs only there.
#define DIAG "gfx: vulkan: backing map: straddle dump: "

enum { BORN_NONE, BORN_BIND, BORN_FILL_POOL, BORN_PROMOTE, BORN_BESIDE_FRONT, BORN_BESIDE_END, BORN_GROW, BORN_RESCAN,
       BORN_TRIM, BORN_UNION };
static const char *const k_born[] = {"none", "bind", "fill_pool", "promote", "beside_front", "beside_end", "grow", "rescan",
                                     "trim", "union"};

// beside_front: a mirror that kept its start (the bind took its end); beside_end: one that kept its end. born_lo/born_hi
// are the range the backing had before that cut or growth.
static void born(Backing *b, int by, uint64_t value, uint64_t lo, uint64_t hi) {
    if (b->born0_by == BORN_NONE) b->born0_by = by, b->born0_value = value, b->born0_lo = lo, b->born0_hi = hi;
    b->born_by = by, b->born_value = value, b->born_lo = lo, b->born_hi = hi;
}

// The binds of the newest submissions, newest last: every resolve of a submission that is not older than the raw-access
// submission before the newest one, written by the recording thread (one at a time) with an atomic index.
#define RING_N 512
enum { RING_HIT, RING_MISS, RING_MADE, RING_GREW };   // a hit; a miss, as make() takes it up; what it made; what a growth answered
enum { PHASE_BEFORE, PHASE_AFTER, PHASE_NEWER };   // before backing_raw_begin of the value; after it; after a newer one began
typedef struct RingEnt {
    uint64_t seq, value, addr, size, b_lo, b_hi;
    int how, kind, phase;
} RingEnt;
static RingEnt g_ring[RING_N];
static uint64_t g_ring_seq;   // atomic: entries written so far
static int g_ring_skip;       // g_lock exclusive: the look-up that follows a make is not a second bind

static const char *const k_how[] = {"hit", "miss", "made", "grew"};
static const char *const k_phase[] = {"before_begin", "after_begin", "after_newer_raw"};

static const char *kind_name(int k) {
    return k == BACKING_MIRROR ? "mirror" : k == BACKING_IMPORT ? "import" : k == BACKING_EXTENT ? "extent" : "none";
}

static void ring_note(uint64_t value, uint64_t addr, uint64_t size, int how, int kind, uint64_t b_lo, uint64_t b_hi) {
    uint64_t last = raw_last(), prev = __atomic_load_n(&g_raw_prev, __ATOMIC_ACQUIRE);
    if (!value || (value < last && value != prev)) return;
    uint64_t n = __atomic_fetch_add(&g_ring_seq, 1, __ATOMIC_RELAXED);
    g_ring[n % RING_N] = (RingEnt){n, value, addr, size, b_lo, b_hi, how, kind,
                                   value < last ? PHASE_NEWER : value == last ? PHASE_AFTER : PHASE_BEFORE};
}

// The bind make() is serving, for the dump that beside() prints (g_lock exclusive).
static struct { const Rec *r; uint64_t addr, size; int kind; } g_bind;

static uint64_t fnv(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull; }

static uint64_t fnv_backing(uint64_t h, const Backing *b) {
    h = fnv(h, (uint64_t)b->kind), h = fnv(h, b->gpu_lo), h = fnv(h, b->gpu_hi), h = fnv(h, b->offset);
    h = fnv(h, b->part_at), h = fnv(h, b->part), h = fnv(h, b->res_value), h = fnv(h, b->res_lo);
    h = fnv(h, b->res_hi), h = fnv(h, b->res_n);
    return fnv(h, (uint64_t)b->born_by);
}

// A digest of what the map is: its version and counters, every backing of every pool, every block's free list and the
// deferred-release queue's length. The dump takes it before and after its dry evaluation of the growth's guards, and prints
// both. It leaves out the list of GPU writes to write back: the completion thread changes it under g_coh alone, while
// the dry evaluation holds g_lock, so a digest that read it could differ with nothing mutated (and read it unsynchronised).
static uint64_t diag_digest(void) {
    uint64_t h = 0xcbf29ce484222325ull;
    h = fnv(h, backing_version());
    const uint64_t *w = (const uint64_t *)(const void *)&g_st;
    for (size_t i = 0; i < sizeof g_st / sizeof *w; i++) h = fnv(h, w[i]);
    for (int i = 0; i < g_index.n; i++) {
        const Rec *r = (const Rec *)g_index.e[i].key;
        h = fnv(h, r->lo);
        if (r->ext) h = fnv_backing(h, r->ext);
        for (int j = 0; j < r->ncells; j++) {
            h = fnv(h, r->cells[j].lo), h = fnv(h, r->cells[j].hi);
            for (int k = 0; k < r->cells[j].nimp; k++) h = fnv_backing(h, r->cells[j].imp[k]);
        }
    }
    for (int i = 0; i < g_nblocks; i++) {
        const Block *b = g_blocks[i];
        h = fnv(h, b->used), h = fnv(h, (uint64_t)b->nfree);
        for (int j = 0; j < b->nfree; j++) h = fnv(h, b->free[j].lo), h = fnv(h, b->free[j].hi);
    }
    return fnv(h, (uint64_t)g_ndef);
}

static void diag_hull(char *out, size_t n, const Backing *b) {
    if (!b->res_n || b->res_lo >= b->res_hi) snprintf(out, n, "empty (lo %#llx, hi %#llx)", (ull)b->res_lo, (ull)b->res_hi);
    else snprintf(out, n, "[%#llx, %#llx)", (ull)b->res_lo, (ull)b->res_hi);
}

// The free space of block b right before and right after the part [at, end).
static void block_room(const Block *b, uint64_t at, uint64_t end, uint64_t *before, uint64_t *after) {
    *before = *after = 0;
    for (int i = 0; i < b->nfree; i++) {
        if (b->free[i].hi == at) *before = at - b->free[i].lo;
        if (b->free[i].lo == end) *after = b->free[i].hi - end;
    }
}

static void diag_backing(const Backing *b, const Backing *m) {
    char hull[80], room[96] = "";
    diag_hull(hull, sizeof hull, b);
    if (b->kind == BACKING_MIRROR) {
        uint64_t before, after;
        block_room(b->block, b->part_at, b->part_at + b->part, &before, &after);
        snprintf(room, sizeof room, " room_before %#llx room_after %#llx buffer %#llx", (ull)before, (ull)after,
                 (ull)b->mem.size);
    }
    rt_log(DIAG "backing %s [%#llx, %#llx) offset %#llx part [%#llx, %#llx) last_use %llu written %llu hot %d res (value "
           "%llu, %s, n %llu) born %s for %llu was [%#llx, %#llx) born0 %s for %llu was [%#llx, %#llx)%s%s", kind_name(b->kind), (ull)b->gpu_lo,
           (ull)b->gpu_hi, (ull)b->offset, (ull)b->part_at, (ull)(b->part_at + b->part),
           (ull)__atomic_load_n(&b->last_use, __ATOMIC_ACQUIRE), (ull)__atomic_load_n(&b->written, __ATOMIC_ACQUIRE),
           __atomic_load_n(&b->hot, __ATOMIC_RELAXED), (ull)b->res_value, hull, (ull)b->res_n, k_born[b->born_by],
           (ull)b->born_value, (ull)b->born_lo, (ull)b->born_hi, k_born[b->born0_by], (ull)b->born0_value,
           (ull)b->born0_lo, (ull)b->born0_hi, room, b == m ? " <- the mirror the bind reaches" : "");
}

static void guard_line(const char *tag, int refuses, const char *detail) {
    rt_log(DIAG "guard %s: %s%s%s", tag, refuses ? "refuses" : "passes", *detail ? ": " : "", detail);
}

// Grow_taken's guards, each evaluated on its own and none acted on: no block span is taken, nothing is deferred, no
// count moves, the map's version stays. G1 the cap on one growth, G2 another mirror on the pages it would add, G3 a
// backing of another cell there, G4 an import of this cell the submission bound, G5 the head reaching before the start of
// the buffer, G6 the grown range past its end, G7a block space after the part, G7b before it, G8 the write watch, G9 a GPU
// write of the submission declared over the pages it would add.
static void diag_guards(const Cell *c, const Backing *m, uint64_t plo, uint64_t phi, uint64_t value) {
    const Rec *rec = g_bind.r;
    uint64_t old_lo = m->gpu_lo, old_hi = m->gpu_hi;
    uint64_t lo = plo < old_lo ? plo : old_lo, hi = phi > old_hi ? phi : old_hi;
    uint64_t added = old_lo - lo + hi - old_hi;
    char d[320];
    rt_log(DIAG "dry guards for the mirror [%#llx, %#llx): grown range [%#llx, %#llx) adds %#llx bytes (head %#llx, tail "
           "%#llx); legend: G1 cap, G2 another mirror, G3 another cell, G4 import this submission bound, G5 head before "
           "the buffer's start, G6 range past the buffer's end, G7a block space after the part, G7b before it, G8 write "
           "watch, G9 GPU write declared over the added pages", (ull)old_lo, (ull)old_hi, (ull)lo, (ull)hi, (ull)added,
           (ull)(old_lo - lo), (ull)(hi - old_hi));
    snprintf(d, sizeof d, "adds %#llx, at most %#llx", (ull)added, (ull)BACKING_MIRROR_GROW_MAX);
    guard_line("G1", added > BACKING_MIRROR_GROW_MAX, d);
    int refuse = 0;
    d[0] = 0;
    for (int i = 0; i < c->nimp && !refuse; i++) {
        const Backing *b = c->imp[i];
        if (b == m || b->kind == BACKING_IMPORT || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        snprintf(d, sizeof d, "the mirror [%#llx, %#llx)", (ull)b->gpu_lo, (ull)b->gpu_hi);
        refuse = 1;
    }
    guard_line("G2", refuse, d);
    refuse = 0;
    d[0] = 0;
    for (int i = 0; i < g_index.n && !refuse; i++) {
        const Rec *other = (const Rec *)g_index.e[i].key;
        if (other->cls != CLS_CPU) continue;
        for (int j = 0; j < other->ncells && !refuse; j++) {
            const Cell *cell = &other->cells[j];
            if (other == rec && cell == c) continue;
            for (int k = 0; k < cell->nimp && !refuse; k++) {
                const Backing *b = cell->imp[k];
                if (!overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
                snprintf(d, sizeof d, "the backing [%#llx, %#llx)", (ull)b->gpu_lo, (ull)b->gpu_hi);
                refuse = 1;
            }
        }
    }
    guard_line("G3", refuse, d);
    refuse = 0;
    for (int i = 0; i < c->nimp; i++) {
        const Backing *b = c->imp[i];
        if (b == m || b->kind != BACKING_IMPORT || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        char hull[80];
        diag_hull(hull, sizeof hull, b);
        int has = b->res_value == value && b->res_n && b->res_lo < b->res_hi;
        int on_added = has && ((lo < old_lo && overlaps(lo, old_lo, b->res_lo, b->res_hi)) ||
                               (old_hi < hi && overlaps(old_hi, hi, b->res_lo, b->res_hi)));
        int on_bind = has && overlaps(plo, phi, b->res_lo, b->res_hi);
        int gi = import_gives(b, b->gpu_lo > lo ? b->gpu_lo : lo, b->gpu_hi < hi ? b->gpu_hi : hi, value);
        refuse |= gi != GI_OK;
        snprintf(d, sizeof d, "the import [%#llx, %#llx) used_by %llu written %llu hull %s%s hull_on_added_pages %s "
                 "hull_on_bind %s", (ull)b->gpu_lo, (ull)b->gpu_hi, (ull)used_by(b),
                 (ull)__atomic_load_n(&b->written, __ATOMIC_ACQUIRE), hull, has ? "" : " (not this submission's)",
                 on_added ? "yes" : "no", on_bind ? "yes" : "no");
        guard_line("G4", gi != GI_OK, d);
        // Which answer of the rule it was, on a line of its own (the line above is pinned byte for byte)
        rt_log(DIAG "guard G4 rule: %s%s%s%s", k_gi[gi], gi != GI_LOOSE && b->res_loose && b->res_value == value ? ", loose" : "",
               *k_gi_why[gi] ? ": it " : "", k_gi_why[gi]);
    }
    if (!refuse) guard_line("G4", 0, "no import of the cell the submission bound lies on the pages");
    if (old_lo - lo > m->offset) {
        snprintf(d, sizeof d, "the %#llx bytes it adds before its start reach before the buffer's (offset %#llx)",
                 (ull)(old_lo - lo), (ull)m->offset);
        guard_line("G5", 1, d);
        guard_line("G6", 0, "not evaluated: the offset below the start is the G5 refusal");
        guard_line("G7a", 0, "not evaluated: as G6");
        guard_line("G7b", 0, "not evaluated: as G6");
    } else {
        snprintf(d, sizeof d, "head %#llx bytes, offset %#llx", (ull)(old_lo - lo), (ull)m->offset);
        guard_line("G5", 0, d);
        uint64_t offset = m->offset - (old_lo - lo);
        snprintf(d, sizeof d, "range %#llx bytes at offset %#llx, buffer %#llx", (ull)(hi - lo), (ull)offset,
                 (ull)m->mem.size);
        guard_line("G6", hi - lo > m->mem.size - offset, d);
        uint64_t part_lo = down(offset, BACKING_EXTENT_ALIGN), part_hi = up(offset + hi - lo, BACKING_EXTENT_ALIGN);
        uint64_t old_part_lo = m->part_at, old_part_hi = m->part_at + m->part;
        if (part_lo > old_part_lo) part_lo = old_part_lo;
        if (part_hi < old_part_hi) part_hi = old_part_hi;
        snprintf(d, sizeof d, "[%#llx, %#llx) after the part", (ull)old_part_hi, (ull)part_hi);
        guard_line("G7a", part_hi > m->mem.size || !block_free_at(m->block, old_part_hi, part_hi), d);
        snprintf(d, sizeof d, "[%#llx, %#llx) before the part", (ull)part_lo, (ull)old_part_lo);
        guard_line("G7b", !block_free_at(m->block, part_lo, old_part_lo), d);
    }
    snprintf(d, sizeof d, "scan of [%#llx, %#llx) (it resets the marks it reports; the stop follows)", (ull)lo, (ull)hi);
    guard_line("G8", pagegen_scan(lo, hi) == UINT64_MAX, d);
    refuse = 0;
    d[0] = 0;
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend && !refuse; i++) {
        const Pending *p = &g_pend[i];
        if (p->m == m || !overlaps(lo, hi, p->lo, p->hi) || p->value < value) continue;
        snprintf(d, sizeof d, "the write [%#llx, %#llx) of submission %llu", (ull)p->lo, (ull)p->hi, (ull)p->value);
        refuse = 1;
    }
    plat_mutex_unlock(&g_coh);
    guard_line("G9", refuse, d);
}

static void ring_line(const RingEnt *e) {
    rt_log(DIAG "ring %llu: submission %llu %s %s %s addr %#llx size %#llx backing [%#llx, %#llx)", (ull)e->seq,
           (ull)e->value, k_phase[e->phase], k_how[e->how], kind_name(e->kind), (ull)e->addr, (ull)e->size,
           (ull)e->b_lo, (ull)e->b_hi);
}

static void diag_ring(uint64_t value) {
    uint64_t n = __atomic_load_n(&g_ring_seq, __ATOMIC_RELAXED), first = n > RING_N ? n - RING_N : 0;
    uint64_t before = raw_before(value);
    rt_log(DIAG "the binds of submission %llu, oldest first (the ring holds the last %d of %llu)", (ull)value, RING_N,
           (ull)n);
    for (uint64_t i = first; i < n; i++)
        if (g_ring[i % RING_N].value == value) ring_line(&g_ring[i % RING_N]);
    uint64_t idx[8];
    int seen = 0;
    for (uint64_t i = n; i > first && seen < 8; i--)
        if (before && g_ring[(i - 1) % RING_N].value == before) idx[seen++] = i - 1;
    rt_log(DIAG "the last %d binds of the raw-access submission before it (%llu)", seen, (ull)before);
    while (seen > 0) ring_line(&g_ring[idx[--seen] % RING_N]);
}

// What the growths of this run did and what they refused an import for, the one list the dump and the exit line print.
static void growth_counts(char *out, size_t n) {
    size_t at = sn(out, n, "grows %llu (%llu bytes), %llu trimmed a used import (%llu imports, %llu bytes), "
                                 "%llu waited, heirs %llu (%llu with a GPU write); import refusals:",
                                 (ull)g_st.raw_grows, (ull)g_st.raw_grow_bytes, (ull)g_st.raw_grow_used,
                                 (ull)g_st.raw_grow_trims, (ull)g_st.raw_grow_trim_bytes, (ull)g_st.raw_grow_waits,
                                 (ull)g_st.raw_heirs, (ull)g_st.raw_heir_written);
    for (int i = GI_NEWER; i < GI_N && at < n; i++)
        at += sn(out + at, n - at, " %s %llu", k_gi[i], (ull)g_st.raw_refused[i]);
}

// The mirrors' total against the cap and what the unions of the begins did, the second list the dump and the exit line print.
// Read-only: the dry evaluation's digest covers the counters, and none moves here.
static void cap_counts(char *out, size_t n) {
    snprintf(out, n, "mirror_bytes %llu (peak %llu, a promotion leaves at most %llu, peak overshoot %llu), growths that left it "
             "over the cap %llu; unions %llu (%llu mirrors, %llu imports, %llu bytes, mirrors moved by %lld bytes, %llu of the "
             "%llu bytes of the cap exemption used), chains left alone: written %llu, other cell %llu, past the cell %llu",
             (ull)g_st.mirror_bytes, (ull)g_st.mirror_bytes_peak, (ull)BACKING_PROMOTE_MIRROR_MAX,
             (ull)g_st.mirror_overshoot_peak, (ull)g_st.raw_grow_over_cap, (ull)g_st.raw_union_chains,
             (ull)g_st.raw_union_mirror, (ull)g_st.raw_union_import, (ull)g_st.raw_union_bytes,
             (long long)g_st.raw_union_net_bytes, (ull)g_st.raw_union_overcap_bytes, (ull)BACKING_UNION_OVERSHOOT_MAX,
             (ull)g_st.raw_union_skipped[0], (ull)g_st.raw_union_skipped[1], (ull)g_st.raw_union_skipped[2]);
}

// The refusal of the growth, as it happens: the reason and the mirror. The stop's dump follows from beside().
static void diag_refusal(const Backing *m, uint64_t plo, uint64_t phi, uint64_t value, const char *why) {
    char hull[80];
    diag_hull(hull, sizeof hull, m);
    rt_log(DIAG "growth refused in submission %llu for the bind [%#llx, %#llx) (pages [%#llx, %#llx)) over the mirror "
           "[%#llx, %#llx), hull %s, born %s: %s", (ull)value, (ull)g_bind.addr, (ull)(g_bind.addr + g_bind.size),
           (ull)plo, (ull)phi, (ull)m->gpu_lo, (ull)m->gpu_hi, hull, k_born[m->born_by], why);
}

// ---------- the event ring and the dry evaluation of the candidate growth
// The ring of binds above takes every hit, so a long submission pushes the misses out of it. This ring takes none: only the
// events that change what the map holds or decide it (a raw-access begin, the fill and promotion passes, a miss with the
// rule's answer, the cut a bind gives a mirror, what a bind made or grew, and the verdict of the dry growth). Every event of
// the bind being served carries the number of its miss, so a verdict is tied to the miss it belongs to; the verdict is
// written right after the miss, with nothing between. Written under g_lock exclusive only (every writer holds it), read by
// the dump and by nothing else; a decision never reads it.
#define EV_N 256
#define EV_IMP 4    // the backings an event records in full; the rest are counted
#define EV_VAL 32   // the submissions whose events are counted, to say how many the ring lost
enum { EV_BEGIN, EV_PROMOTE, EV_FILL, EV_MISS, EV_MADE, EV_GREW, EV_CUT, EV_VERDICT, EV_UNION };
static const char *const k_ev[] = {"begin", "promote", "fill", "miss", "made", "grew", "cut", "verdict", "union"};

// A backing as an event saw it, read before anything freed or replaced it.
typedef struct EvImp {
    uint64_t lo, hi, used, res_value, res_lo, res_hi, res_n, born_value, born0_value, born0_lo, born0_hi;
    int kind, born_by, born0_by;
    int added;   // a verdict's replaced import: 0 not asked, 1 its hull is off the pages the growth would add, 2 on them, 3 unknown
    int gi;      // a verdict's replaced import: 0 not asked, else 1 + the answer of import_gives (GI_*; 1 is GI_OK)
} EvImp;
typedef struct EvSnap { EvImp imp[EV_IMP]; int n, more; } EvSnap;

typedef struct EvEnt {
    uint64_t seq, value, miss;   // miss: the number of the bind being served (0: none)
    uint64_t x[14];              // numbers, as each kind of event says (ev_line)
    int what, kind, phase;
    uint32_t mask;               // a verdict's refusals: bit i is k_dry[i]
    uint32_t g4rules;            // a verdict's: bit i is the answer GI_* i of import_gives that refused an import (with G4)
    EvImp mir;                   // a cut's mirror, a verdict's mirror
    EvSnap s;                    // the backings a made, a grew or a verdict replaces
} EvEnt;

// The names of the verdict's refusals, in the order of the counters (BackingStats raw_dry_g) and of the mask's bits. G8,
// the write watch, is asked by pagegen_covered, which resets no mark and stamps no page. G4_hull_off is not a guard: it
// comes with G4 when every import G4 refuses on has a hull of this submission and none of the hulls lies on the pages the
// growth would add (the byte-level G4 would pass); G4 alone leaves that open (a hull that is not this submission's, or one
// on the added pages).
enum { DRY_KIND, DRY_G1, DRY_G2, DRY_G3, DRY_G4, DRY_G5, DRY_G6, DRY_G7A, DRY_G7B, DRY_G8, DRY_G9, DRY_CAP, DRY_G4_OFF,
       DRY_N };
static const char *const k_dry[DRY_N] = {"rule_kind", "G1", "G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G8", "G9", "cap",
                                         "G4_hull_off"};
_Static_assert(DRY_N == sizeof ((BackingStats *)0)->raw_dry_g / sizeof ((BackingStats *)0)->raw_dry_g[0],
               "the counters and the verdict's refusals are the same list");

static EvEnt g_ev[EV_N];
static EvEnt g_ev_null;       // where an event with no submission goes (value 0): never read
static uint64_t g_ev_seq;     // events written so far
static uint64_t g_ev_misses;  // misses numbered so far
static uint64_t g_ev_cur;     // the miss being served now (0: none)
typedef struct EvCount { uint64_t value, n; } EvCount;
static EvCount g_evc[EV_VAL];   // events written per submission, the lowest value dropped when full
static struct { uint64_t evals, would, refused, mutated, g[DRY_N]; } g_dry;
static int g_dump_dry_done;   // a straddle stop's dump printed the dry evaluation this run

static void ev_imp_of(EvImp *o, const Backing *b) {
    *o = (EvImp){b->gpu_lo, b->gpu_hi, used_by(b), b->res_value, b->res_lo, b->res_hi, b->res_n, b->born_value,
                 b->born0_value, b->born0_lo, b->born0_hi, b->kind, b->born_by, b->born0_by, 0, 0};
}

// Whether the hull of what submission value resolved in b lies on the pages a growth of [old_lo, old_hi) to [lo, hi) would
// add: 1 off them, 2 on them, 3 unknown (b has no hull of this submission: it may have been used, by a later one, anywhere).
static int hull_on_added(const Backing *b, uint64_t value, uint64_t lo, uint64_t hi, uint64_t old_lo, uint64_t old_hi) {
    if (b->res_value != value || !b->res_n || b->res_lo >= b->res_hi) return 3;
    int on = (lo < old_lo && overlaps(lo, old_lo, b->res_lo, b->res_hi)) ||
             (old_hi < hi && overlaps(old_hi, hi, b->res_lo, b->res_hi));
    return on ? 2 : 1;
}

// The backings of c inside [lo, hi), except skip, into s (the first EV_IMP in full, the rest counted). When value is not 0
// each also says whether its hull lies on the pages a growth from [old_lo, old_hi) to [lo, hi) adds.
static void ev_snap(EvSnap *s, const Cell *c, uint64_t lo, uint64_t hi, const Backing *skip, uint64_t value,
                    uint64_t old_lo, uint64_t old_hi) {
    *s = (EvSnap){0};
    for (int i = 0; i < c->nimp; i++) {
        const Backing *b = c->imp[i];
        if (b == skip || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        if (s->n < EV_IMP) {
            ev_imp_of(&s->imp[s->n], b);
            if (value) {
                s->imp[s->n].added = hull_on_added(b, value, lo, hi, old_lo, old_hi);
                if (b->kind == BACKING_IMPORT)
                    s->imp[s->n].gi = 1 + import_gives(b, b->gpu_lo > lo ? b->gpu_lo : lo, b->gpu_hi < hi ? b->gpu_hi : hi, value);
            }
            s->n++;
        } else {
            s->more++;
        }
    }
}

static void ev_count(uint64_t value) {
    EvCount *at = NULL;
    for (int i = 0; i < EV_VAL; i++) {
        if (g_evc[i].value == value) {
            at = &g_evc[i];
            break;
        }
        if (!at || g_evc[i].value < at->value) at = &g_evc[i];
    }
    if (at->value != value) *at = (EvCount){value, 0};
    at->n++;
}

// A new event of submission value, in the ring (g_lock exclusive).
static EvEnt *ev_new(uint64_t value, int what) {
    if (!value) return &g_ev_null;
    EvEnt *e = &g_ev[g_ev_seq % EV_N];
    uint64_t last = raw_last();
    *e = (EvEnt){.seq = g_ev_seq++, .value = value, .miss = g_ev_cur, .what = what,
                 .phase = value < last ? PHASE_NEWER : value == last || value == g_begin_now ? PHASE_AFTER : PHASE_BEFORE};
    ev_count(value);
    return e;
}

// The candidate growth of a mirror m the bind [plo, phi) of a raw-access submission cut instead of growing it, evaluated
// and written as an event, acted on by nothing. Each guard of grow_taken is evaluated on its own (as diag_guards does),
// plus the rule's own kind and the cap on the mirrors' total; G8 is asked of the write watch without a reset
// (pagegen_covered); no wait, no span taken, nothing deferred, no counter of the map moved, the version unchanged. g_lock
// exclusive, in the miss, before beside() and before any import is replaced.
static void dry_growth(const Rec *rec, const Cell *c, const Backing *m, uint64_t plo, uint64_t phi, uint64_t value,
                       int kind, uint64_t others) {
    EvEnt *e = ev_new(value, EV_VERDICT);
    uint64_t digest = diag_digest();   // the map before: the evaluation must leave it so (counted, and printed)
    uint64_t old_lo = m->gpu_lo, old_hi = m->gpu_hi;
    uint64_t lo = plo < old_lo ? plo : old_lo, hi = phi > old_hi ? phi : old_hi;
    uint64_t added = old_lo - lo + hi - old_hi, before, after;
    uint32_t mask = 0;
    ev_snap(&e->s, c, lo, hi, m, value, old_lo, old_hi);
    if (kind != BACKING_MIRROR) mask |= 1u << DRY_KIND;
    if (added > BACKING_MIRROR_GROW_MAX) mask |= 1u << DRY_G1;
    int used_on_added = 0;   // an import G4 refuses on may have its hull on the added pages, or have no hull to say
    for (int i = 0; i < c->nimp; i++) {
        const Backing *b = c->imp[i];
        if (b == m || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        if (b->kind != BACKING_IMPORT) {
            mask |= 1u << DRY_G2;
        } else if (import_gives(b, b->gpu_lo > lo ? b->gpu_lo : lo, b->gpu_hi < hi ? b->gpu_hi : hi, value) != GI_OK) {
            mask |= 1u << DRY_G4;
            e->g4rules |= 1u << import_gives(b, b->gpu_lo > lo ? b->gpu_lo : lo, b->gpu_hi < hi ? b->gpu_hi : hi, value);
            used_on_added |= hull_on_added(b, value, lo, hi, old_lo, old_hi) != 1;
        }
    }
    if ((mask >> DRY_G4 & 1u) && !used_on_added) mask |= 1u << DRY_G4_OFF;
    for (int i = 0; i < g_index.n; i++) {
        const Rec *other = (const Rec *)g_index.e[i].key;
        if (other->cls != CLS_CPU) continue;
        for (int j = 0; j < other->ncells; j++) {
            const Cell *cell = &other->cells[j];
            if (other == rec && cell == c) continue;
            for (int k = 0; k < cell->nimp; k++)
                if (overlaps(lo, hi, cell->imp[k]->gpu_lo, cell->imp[k]->gpu_hi)) mask |= 1u << DRY_G3;
        }
    }
    uint64_t old_part_lo = m->part_at, old_part_hi = m->part_at + m->part;
    block_room(m->block, old_part_lo, old_part_hi, &before, &after);
    if (old_lo - lo > m->offset) {
        mask |= 1u << DRY_G5;
    } else {
        uint64_t offset = m->offset - (old_lo - lo);
        if (hi - lo > m->mem.size - offset) mask |= 1u << DRY_G6;
        uint64_t part_lo = down(offset, BACKING_EXTENT_ALIGN), part_hi = up(offset + hi - lo, BACKING_EXTENT_ALIGN);
        if (part_lo > old_part_lo) part_lo = old_part_lo;
        if (part_hi < old_part_hi) part_hi = old_part_hi;
        if (part_hi > m->mem.size || !block_free_at(m->block, old_part_hi, part_hi)) mask |= 1u << DRY_G7A;
        if (!block_free_at(m->block, part_lo, old_part_lo)) mask |= 1u << DRY_G7B;
    }
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend; i++) {
        const Pending *p = &g_pend[i];
        if (p->m != m && overlaps(lo, hi, p->lo, p->hi) && p->value >= value) mask |= 1u << DRY_G9;
    }
    plat_mutex_unlock(&g_coh);
    if (!pagegen_covered(lo, hi)) mask |= 1u << DRY_G8;   // the real G8 asks the same of the whole grown range, with a reset
    if (g_st.mirror_bytes + added > BACKING_PROMOTE_MIRROR_MAX) mask |= 1u << DRY_CAP;
    ev_imp_of(&e->mir, m);
    e->kind = kind, e->mask = mask;
    e->x[0] = old_lo, e->x[1] = old_hi, e->x[2] = lo, e->x[3] = hi, e->x[4] = added, e->x[5] = g_st.mirror_bytes;
    e->x[6] = old_lo - lo, e->x[7] = m->offset, e->x[8] = before, e->x[9] = after, e->x[10] = old_part_lo;
    e->x[11] = old_part_hi, e->x[12] = others;
    e->x[13] = diag_digest() != digest;
    g_dry.mutated += e->x[13];
    g_dry.evals++;
    if (mask) g_dry.refused++;
    else g_dry.would++;
    for (int i = 0; i < DRY_N; i++) g_dry.g[i] += (mask >> i) & 1u;
}

// What the union pass did with a chain of abutting backings (an event's x[4]).
enum { UK_MIRROR, UK_IMPORT_CAP, UK_IMPORT_RULE, UK_IMPORT_ADMIT, UK_LEFT_WRITTEN, UK_LEFT_CELL, UK_LEFT_EDGE };
static const char *const k_union[] = {
    "made one mirror over them", "made one import over them (the mirrors would pass the cap)",
    "made one import over them (the rule says import)", "made one import over them (the admission refused the mirror)",
    "left alone: a member, or a pending GPU write, was written by this submission",
    "left alone: a backing of another cell lies on the union", "left alone: the union runs past its cell"};

static void ev_miss(uint64_t value, uint64_t addr, uint64_t size, uint64_t plo, uint64_t phi, uint64_t rs, uint64_t re,
                    int kind) {
    EvEnt *e = ev_new(value, EV_MISS);
    e->kind = kind;
    e->x[0] = addr, e->x[1] = size, e->x[2] = plo, e->x[3] = phi, e->x[4] = rs, e->x[5] = re;
}

// why: 0 the import became a mirror, 1 the cap on the mirrors held it back, 2 the admission refused the mirror.
static void ev_promote(uint64_t value, uint64_t lo, uint64_t hi, int why) {
    EvEnt *e = ev_new(value, EV_PROMOTE);
    e->x[0] = lo, e->x[1] = hi, e->x[2] = (uint64_t)why, e->x[3] = g_st.mirror_bytes;
}

static void ev_hull(char *out, size_t n, const EvImp *o) {
    if (!o->res_n || o->res_lo >= o->res_hi) snprintf(out, n, "empty");
    else snprintf(out, n, "[%#llx, %#llx) of submission %llu", (ull)o->res_lo, (ull)o->res_hi, (ull)o->res_value);
}

// What a verdict's replaced import says of its hull and the pages the growth would add (EvImp added).
static const char *const k_added[] = {"", " hull_on_added_pages no", " hull_on_added_pages yes",
                                      " hull_on_added_pages unknown (no hull of this submission)"};

static void ev_imp_line(const EvEnt *e, const char *role, const EvImp *o) {
    char hull[112];
    ev_hull(hull, sizeof hull, o);
    char rule[40] = "";
    if (o->gi > 1) snprintf(rule, sizeof rule, " G4 rule %s", k_gi[o->gi - 1]);   // the answer that refused it
    rt_log(DIAG "event %llu: %s %s [%#llx, %#llx) used_by %llu (%s submission %llu) hull %s res_n %llu born %s for %llu "
           "born0 %s for %llu was [%#llx, %#llx)%s%s", (ull)e->seq, role, kind_name(o->kind), (ull)o->lo, (ull)o->hi,
           (ull)o->used, o->used >= e->value ? "reached by" : "before", (ull)e->value, hull, (ull)o->res_n,
           k_born[o->born_by], (ull)o->born_value, k_born[o->born0_by], (ull)o->born0_value, (ull)o->born0_lo,
           (ull)o->born0_hi, k_added[o->added], rule);
}

static void ev_snap_lines(const EvEnt *e, const char *role) {
    for (int i = 0; i < e->s.n; i++) ev_imp_line(e, role, &e->s.imp[i]);
    if (e->s.more) rt_log(DIAG "event %llu: %d more not recorded in full", (ull)e->seq, e->s.more);
}

static void ev_line(const EvEnt *e) {
    char d[800] = "";
    const uint64_t *x = e->x;
    switch (e->what) {
    case EV_BEGIN:
        snprintf(d, sizeof d, "mirror_bytes %llu mirrors %llu", (ull)x[0], (ull)x[1]);
        break;
    case EV_PROMOTE:
        snprintf(d, sizeof d, "import [%#llx, %#llx) %s; mirror_bytes %llu, a promotion leaves at most %llu", (ull)x[0],
                 (ull)x[1], x[2] == 0 ? "made a mirror" : x[2] == 1 ? "not made: over the cap" : "not made: refused",
                 (ull)x[3], (ull)BACKING_PROMOTE_MIRROR_MAX);
        break;
    case EV_FILL:
        snprintf(d, sizeof d, "import [%#llx, %#llx) made for the page table's invariant", (ull)x[0], (ull)x[1]);
        break;
    case EV_MISS:
        snprintf(d, sizeof d, "addr %#llx size %#llx pages [%#llx, %#llx) committed run [%#llx, %#llx) rule says %s",
                 (ull)x[0], (ull)x[1], (ull)x[2], (ull)x[3], (ull)x[4], (ull)x[5], kind_name(e->kind));
        break;
    case EV_MADE:
        snprintf(d, sizeof d, "%s [%#llx, %#llx), replacing %d recorded and %d more backings", kind_name(e->kind),
                 (ull)x[0], (ull)x[1], e->s.n, e->s.more);
        break;
    case EV_GREW:
        snprintf(d, sizeof d, "mirror [%#llx, %#llx) grew to [%#llx, %#llx), over %d recorded and %d more imports (one "
                 "the submission reached was trimmed at its start, the others deferred)",
                 (ull)x[0], (ull)x[1], (ull)x[2], (ull)x[3], e->s.n, e->s.more);
        break;
    case EV_UNION:
        snprintf(d, sizeof d, "[%#llx, %#llx) of %llu abutting backings (%llu of them mirrors): %s; the mirrors' total moved by "
                 "%lld bytes to %llu", (ull)x[0], (ull)x[1], (ull)x[2], (ull)x[3], k_union[x[4]], (long long)(int64_t)x[5],
                 (ull)x[6]);
        break;
    case EV_CUT:
        snprintf(d, sizeof d, "mirror [%#llx, %#llx) %s: now [%#llx, %#llx)", (ull)x[0], (ull)x[1],
                 x[4] == 0 ? "kept its start (the bind took its end)" : x[4] == 1 ? "kept its end (the bind took its start)"
                                                                                  : "taken whole (deferred)",
                 (ull)x[2], (ull)x[3]);
        break;
    default: {
        char list[128] = "";
        size_t at = 0;
        for (int i = 0; i < DRY_N; i++) {
            if (!(e->mask >> i & 1u)) continue;
            at += sn(list + at, sizeof list - at, "%s%s", at ? "," : "", k_dry[i]);
            if (i != DRY_G4) continue;   // G4 names the answers of import_gives that refused: G4[hull+up]
            at += sn(list + at, sizeof list - at, "[");
            int first = 1;
            for (int g = GI_NEWER; g < GI_N; g++)
                if (e->g4rules >> g & 1u) at += sn(list + at, sizeof list - at, "%s%s", first ? "" : "+", k_gi[g]), first = 0;
            at += sn(list + at, sizeof list - at, "]");
        }
        snprintf(d, sizeof d, "the mirror [%#llx, %#llx) would grow to [%#llx, %#llx), adding %#llx bytes (head %#llx, "
                 "its offset %#llx); rule says %s; mirror_bytes %llu, at most %llu; block room before its part [%#llx, "
                 "%#llx) %#llx after %#llx; %llu other mirrors of the cell on the bind; the evaluation left the map %s (its digest leaves out the list of GPU writes); refuses: %s",
                 (ull)x[0], (ull)x[1], (ull)x[2], (ull)x[3], (ull)x[4], (ull)x[6], (ull)x[7], kind_name(e->kind),
                 (ull)x[5], (ull)BACKING_PROMOTE_MIRROR_MAX, (ull)x[10], (ull)x[11], (ull)x[8], (ull)x[9], (ull)x[12],
                 x[13] ? "CHANGED" : "unchanged", e->mask ? list : "nothing (it would grow)");
    }
    }
    rt_log(DIAG "event %llu: submission %llu %s %s [bind %llu]: %s", (ull)e->seq, (ull)e->value, k_phase[e->phase],
           k_ev[e->what], (ull)e->miss, d);
    if (e->what == EV_CUT) ev_imp_line(e, "the mirror cut", &e->mir);
    if (e->what == EV_MADE || e->what == EV_GREW) ev_snap_lines(e, "replaced");
    if (e->what == EV_UNION) ev_snap_lines(e, "joined");
    if (e->what == EV_VERDICT) {
        ev_imp_line(e, "the mirror it would grow", &e->mir);
        ev_snap_lines(e, "would be replaced");
    }
}

// Events written for submission value, or UINT64_MAX when the count was dropped.
static uint64_t ev_written(uint64_t value) {
    for (int i = 0; i < EV_VAL; i++)
        if (g_evc[i].value == value) return g_evc[i].n;
    return UINT64_MAX;
}

// The events of submission value and of the raw-access submission before it, oldest first, and how many of them the ring
// already lost. g_lock exclusive.
static void diag_events(uint64_t value) {
    uint64_t n = g_ev_seq, first = n > EV_N ? n - EV_N : 0, before = raw_before(value);
    rt_log(DIAG "events of submission %llu and of the raw-access submission before it (%llu), oldest first (the ring holds "
           "the last %d of %llu events, none of them a hit)", (ull)value, (ull)before, EV_N, (ull)n);
    uint64_t held[2] = {0, 0};
    for (uint64_t i = first; i < n; i++) {
        const EvEnt *e = &g_ev[i % EV_N];
        if (e->value != value && !(before && e->value == before)) continue;
        held[e->value == value]++;
        ev_line(e);
    }
    uint64_t want[2] = {before ? ev_written(before) : 0, ev_written(value)};
    char d[240];
    size_t at = 0;
    uint64_t lost = 0;
    int known = 1;
    for (int i = 0; i < 2; i++) {
        uint64_t v = i ? value : before;
        if (!v) continue;
        if (want[i] == UINT64_MAX) {
            known = 0;
            at += sn(d + at, sizeof d - at, "%ssubmission %llu: unknown (its count was dropped)", at ? "; " : "",
                                   (ull)v);
        } else {
            lost += want[i] - held[i];
            at += sn(d + at, sizeof d - at, "%ssubmission %llu: %llu of %llu", at ? "; " : "", (ull)v,
                                   (ull)(want[i] - held[i]), (ull)want[i]);
        }
    }
    if (known && !lost) rt_log(DIAG "event ring overwritten: none of the events of these two submissions (%s)", d);
    else rt_log(DIAG "event ring overwritten: %llu events of these two submissions%s (%s)", (ull)lost,
                known ? "" : " at least", d);
}

// Everything the stop of beside() needs to be read, printed before it: the bind, the submission's state, every backing of
// the cell within 2 MiB of the bind, the growth's guards evaluated on their own, the ring. g_lock exclusive.
static void diag_stop(const Cell *c, const Backing *m, uint64_t plo, uint64_t phi, uint64_t rs, uint64_t re,
                      uint64_t value, const char *why) {
    rt_log(DIAG "bind [%#llx, %#llx) pages [%#llx, %#llx) committed run [%#llx, %#llx) submission %llu kind %s then %s "
           "after the raw-access downgrade", (ull)g_bind.addr, (ull)(g_bind.addr + g_bind.size), (ull)plo, (ull)phi,
           (ull)rs, (ull)re, (ull)value, kind_name(g_bind.kind),
           kind_name(g_bind.kind == BACKING_MIRROR ? BACKING_IMPORT : g_bind.kind));
    rt_log(DIAG "raw_last %llu raw_before %llu sync_done %llu issued %llu version %llu why: %s", (ull)raw_last(),
           (ull)raw_before(value), (ull)gpu_sync_done(), (ull)gpu_sync_issued(), (ull)backing_version(), why);
    rt_log(DIAG "mirror_bytes %llu cell [%#llx, %#llx) with %d backings; one growth adds at most %#llx, a mirror reserves "
           "%#llx at each end", (ull)g_st.mirror_bytes, (ull)c->lo, (ull)c->hi, c->nimp,
           (ull)BACKING_MIRROR_GROW_MAX, (ull)BACKING_MIRROR_GUARD);
    char counts[320];
    growth_counts(counts, sizeof counts);
    rt_log(DIAG "growths so far: %s", counts);
    char caps[480];
    cap_counts(caps, sizeof caps);
    rt_log(DIAG "mirror totals so far: %s", caps);
    uint64_t span = 2ull << 20, from = plo > span ? plo - span : 0, to = phi + span;
    for (int i = 0; i < c->nimp; i++)
        if (overlaps(from, to, c->imp[i]->gpu_lo, c->imp[i]->gpu_hi)) diag_backing(c->imp[i], m);
    uint64_t d0 = diag_digest();
    diag_guards(c, m, plo, phi, value);
    uint64_t d1 = diag_digest();
    rt_log(DIAG "the dry evaluation left the map %s (digest %#llx before, %#llx after)",
           d0 == d1 ? "unchanged" : "CHANGED", (ull)d0, (ull)d1);
    g_st.raw_straddle_stops++;
    __atomic_store_n(&g_dump_dry_done, 1, __ATOMIC_RELEASE);   // the dry evaluation is in the dump: the exit lines leave it out
    diag_ring(value);
    diag_events(value);
    rt_log(DIAG "stop %llu of this run follows", (ull)g_st.raw_straddle_stops);
}

// A bind takes pages of mirror m that the submission being recorded (value) already resolved, so m cannot give them to
// an import: what was recorded before reaches m's copy, and what follows would reach the import's. m grows over them in
// place instead, to the hull of its pages and the bind's only, at the same device addresses, so that every bind and every
// raw access of the submission keeps reaching the one copy; the pages it takes were not in a mirror before, so no byte
// has two copies. It grows when: the growth adds at most BACKING_MIRROR_GROW_MAX bytes (the size of this growth, not the
// mirrors' total, which the bind-made mirrors do not keep under any cap; the raw write-back stops by name at the end of
// a submission whose stores reach more pages than it holds); the imports its new range overlaps in this cell are ones no
// bind of this submission used (they go, whole, and fill_pool imports their pages outside the range again) or ones the
// submission used that import_gives lets give their lowest pages (they are trimmed in place, after the submissions issued
// before are done) and no other mirror of the cell, nor any backing of another cell, overlaps it; no GPU write of this submission is declared
// over it; the range lies in m's buffer and the block's free list gives the part beyond m's own (the reserved ends of a
// mirror's allocation, BACKING_MIRROR_GUARD each, are what normally make room); and the write watch covers it. The older
// submissions that reached the replaced imports are waited for, the new pages are copied in as a sync copies them, and
// the map's version moves. Else 0 with the reason in why (GROW_WHY bytes, for the named stop), nothing changed, and the
// caller stops by name. g_lock exclusive.
#define GROW_WHY 256
static int grow_taken(Rec *r, Cell *c, Backing *m, uint64_t plo, uint64_t phi, uint64_t value, char *why) {
    uint64_t old_lo = m->gpu_lo, old_hi = m->gpu_hi;
    uint64_t lo = plo < old_lo ? plo : old_lo, hi = phi > old_hi ? phi : old_hi;
    uint64_t added = old_lo - lo + hi - old_hi;
    if (added > BACKING_MIRROR_GROW_MAX) {
        snprintf(why, GROW_WHY, "the growth would add %llu KiB, more than the %llu KiB one growth may add",
                 (ull)(added >> 10), (ull)(BACKING_MIRROR_GROW_MAX >> 10));
        return 0;
    }
    for (int i = 0; i < c->nimp; i++) {
        const Backing *b = c->imp[i];
        if (b == m || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        if (b->kind != BACKING_IMPORT) {
            snprintf(why, GROW_WHY, "another mirror, [%#llx, %#llx), lies on the pages it would add", (ull)b->gpu_lo,
                     (ull)b->gpu_hi);
            return 0;
        }
        int gi = import_gives(b, b->gpu_lo > lo ? b->gpu_lo : lo, b->gpu_hi < hi ? b->gpu_hi : hi, value);
        if (gi != GI_OK) {   // the head and the text of the first rule, the answer after it when there is one to name
            snprintf(why, GROW_WHY, "the import [%#llx, %#llx), which this submission bound, lies on the pages it "
                     "would add%s%s", (ull)b->gpu_lo, (ull)b->gpu_hi, *k_gi_why[gi] ? ": it " : "", k_gi_why[gi]);
            g_st.raw_refused[gi]++;
            return 0;
        }
        // A used import is trimmed at its start (below), so it must start on the tail the mirror grows over and run past it;
        // import_gives let nothing else through (its pages are below its hull, so they are its first), so this is a check.
        if (used_by(b) >= value && (b->gpu_lo < old_hi || b->gpu_hi <= hi)) {
            snprintf(why, GROW_WHY, "the import [%#llx, %#llx) this submission bound would be cut at its start by the "
                     "growth", (ull)b->gpu_lo, (ull)b->gpu_hi);
            return 0;
        }
    }
    // Another cell or pool could otherwise resolve the added pages to a different copy.
    for (int i = 0; i < g_index.n; i++) {
        const Rec *other = (const Rec *)g_index.e[i].key;
        if (other->cls != CLS_CPU) continue;
        for (int j = 0; j < other->ncells; j++) {
            const Cell *cell = &other->cells[j];
            if (other == r && cell == c) continue;
            for (int k = 0; k < cell->nimp; k++) {
                const Backing *b = cell->imp[k];
                if (!overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
                snprintf(why, GROW_WHY, "a backing of another cell, [%#llx, %#llx), lies on the pages it would add",
                         (ull)b->gpu_lo, (ull)b->gpu_hi);
                return 0;
            }
        }
    }
    if (old_lo - lo > m->offset) {
        snprintf(why, GROW_WHY, "the %llu KiB it would add before its start reach before the start of its buffer",
                 (ull)((old_lo - lo) >> 10));
        return 0;
    }
    uint64_t offset = m->offset - (old_lo - lo);
    if (hi - lo > m->mem.size - offset) {
        snprintf(why, GROW_WHY, "the grown range (%llu KiB) does not fit the rest of its buffer", (ull)((hi - lo) >> 10));
        return 0;
    }
    uint64_t part_lo = down(offset, BACKING_EXTENT_ALIGN), part_hi = up(offset + hi - lo, BACKING_EXTENT_ALIGN);
    uint64_t old_part_lo = m->part_at, old_part_hi = m->part_at + m->part;
    if (part_lo > old_part_lo) part_lo = old_part_lo;
    if (part_hi < old_part_hi) part_hi = old_part_hi;
    if (part_hi > m->mem.size || !block_free_at(m->block, old_part_hi, part_hi)) {
        snprintf(why, GROW_WHY, "its block has no free space after its part for [%#llx, %#llx)", (ull)old_part_hi,
                 (ull)part_hi);
        return 0;
    }
    if (!block_free_at(m->block, part_lo, old_part_lo)) {
        snprintf(why, GROW_WHY, "its block has no free space before its part for [%#llx, %#llx)", (ull)part_lo,
                 (ull)old_part_lo);
        return 0;
    }
    if (pagegen_scan(lo, hi) == UINT64_MAX) {
        snprintf(why, GROW_WHY, "the write watch does not cover [%#llx, %#llx)", (ull)lo, (ull)hi);
        return 0;
    }

    uint64_t wait = raw_before(value);
    int trims = 0;
    for (int i = 0; i < c->nimp; i++) {
        const Backing *b = c->imp[i];
        if (b == m || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) continue;
        if (used_by(b) >= value) {   // reach_before would name this submission, which cannot be waited for (see below)
            trims++;
            continue;
        }
        uint64_t u = reach_before(b, value);
        if (u > wait) wait = u;
    }
    // An import this submission used is kept (trimmed), and its stamp was overwritten by this submission's, so which older
    // submission wrote through it is not known from the stamps: every one issued is waited for (the gate admits one open
    // submission, so these are all below value; the one being recorded is not issued and is not waited for). Nothing is
    // waited for when they are all done, which is the usual state.
    if (trims) {
        uint64_t issued = gpu_sync_issued();
        if (issued >= value) issued = value - 1;
        if (issued > wait) wait = issued;
    }
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_npend; i++) {
        const Pending *p = &g_pend[i];
        if (p->m == m || !overlaps(lo, hi, p->lo, p->hi)) continue;
        if (p->value >= value) {
            plat_mutex_unlock(&g_coh);
            snprintf(why, GROW_WHY, "this submission declared a GPU write over [%#llx, %#llx) of the pages it would "
                     "add", (ull)p->lo, (ull)p->hi);
            return 0;
        }
        if (p->value > wait) wait = p->value;
    }
    plat_mutex_unlock(&g_coh);
    int waited = wait > gpu_sync_done();
    if (waited) {
        g_st.replace_waits++;
        g_st.raw_grow_waits += (uint64_t)(trims > 0);
        wait_value(wait);
    }

    born(m, BORN_GROW, value, old_lo, old_hi);
    block_take_at(m->block, part_lo, old_part_lo);
    block_take_at(m->block, old_part_hi, part_hi);
    m->part_at = part_lo, m->part = part_hi - part_lo;
    m->gpu_lo = lo, m->gpu_hi = hi, m->offset = offset;
    int k = 0, replaced = 0;
    char did[512] = "";
    size_t at = 0;
    for (int i = 0; i < c->nimp; i++) {
        Backing *b = c->imp[i];
        if (b == m || !overlaps(lo, hi, b->gpu_lo, b->gpu_hi)) {
            c->imp[k++] = b;
            continue;
        }
        char hull[80];
        diag_hull(hull, sizeof hull, b);
        int used = used_by(b) >= value;
        if (at < sizeof did)
            at += sn(did + at, sizeof did - at, "; %s [%#llx, %#llx) used_by %llu written %llu hull %s%s",
                                   used ? "trimmed" : "deferred", (ull)b->gpu_lo, (ull)b->gpu_hi, (ull)used_by(b),
                                   (ull)__atomic_load_n(&b->written, __ATOMIC_ACQUIRE), hull,
                                   b->res_loose && b->res_value == value ? " (loose)" : "");
        if (!used) {   // no bind of this submission reached it: it goes whole, and its pages outside the growth get an import again
            defer(b);
            replaced++;
            continue;
        }
        // One it reached stays, with its buffer, its stamps and its hull; only its start moves up to the end of the growth
        // (import_gives: the pages are its lowest, and below everything this submission resolved in it).
        uint64_t cut = hi - b->gpu_lo;
        born(b, BORN_TRIM, value, b->gpu_lo, b->gpu_hi);
        b->offset += cut;
        b->gpu_lo = hi;
        g_st.import_bytes -= cut;
        g_st.raw_grow_trims++, g_st.raw_grow_trim_bytes += cut;
        c->imp[k++] = b;
    }
    c->nimp = k;
    g_st.imports_grown += (uint64_t)replaced;
    g_st.mirror_bytes += added;
    g_st.raw_grows++, g_st.raw_grow_bytes += added;
    g_st.raw_grow_used += (uint64_t)(trims > 0);
    g_st.raw_grow_over_cap += (uint64_t)(g_st.mirror_bytes > BACKING_PROMOTE_MIRROR_MAX);
    mirror_note();
    changed();
    if (g_st.raw_grows <= 64)
        rt_log("gfx: vulkan: backing map: raw growth %llu in submission %llu: the mirror [%#llx, %#llx) grew to [%#llx, "
               "%#llx), adding %#llx bytes; %d used imports trimmed, %d unused deferred%s; mirror_bytes %llu%s",
               (ull)g_st.raw_grows, (ull)value, (ull)old_lo, (ull)old_hi, (ull)lo, (ull)hi, (ull)added, trims, replaced,
               waited ? ", waited for the submissions issued" : "", (ull)g_st.mirror_bytes, did);
    if (g_st.mirror_bytes > BACKING_PROMOTE_MIRROR_MAX && !g_grow_cap_logged) {
        g_grow_cap_logged = 1;
        rt_log("gfx: vulkan: backing map: a growth of a mirror left the mirrors at %llu bytes, past the %llu a promotion "
               "leaves; the growth is exempt (it adds at most %llu bytes); growths over the cap so far %llu, mirror_bytes "
               "peak %llu, peak overshoot %llu", (ull)g_st.mirror_bytes, (ull)BACKING_PROMOTE_MIRROR_MAX,
               (ull)BACKING_MIRROR_GROW_MAX, (ull)g_st.raw_grow_over_cap, (ull)g_st.mirror_bytes_peak,
               (ull)g_st.mirror_overshoot_peak);
    }

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
// mirror this submission already resolved where the bind takes pages cannot give them up, since what was recorded before
// reaches the mirror's copy and what follows would reach the import's: make() grows that mirror over the bind first
// (grow_taken), and when it cannot, or when the submission wrote there on the GPU, this is a named fatal that ends with
// grow's reason (why, set by grow_taken).
// *rs and *re, the committed run around the pages [plo, phi) of the bind, are narrowed to what the new backing covers.
// g_lock exclusive.
static void beside(Cell *c, uint64_t plo, uint64_t phi, uint64_t *rs, uint64_t *re, uint64_t value, const char *why) {
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
        if (m->res_value == value && m->res_lo < ch && cl < m->res_hi) {
            diag_stop(c, m, plo, phi, *rs, *re, value, why);
            rt_fatal("gfx: vulkan: backing map: the mirror [%#llx, %#llx) was bound over [%#llx, %#llx) earlier in "
                     "raw-access submission %llu, and a bind over [%#llx, %#llx) now takes pages of it for an import: "
                     "what was recorded before would reach the mirror's copy of the bytes the rest reaches in the "
                     "import; the mirror cannot grow over them: %s", (ull)m->gpu_lo, (ull)m->gpu_hi, (ull)m->res_lo,
                     (ull)m->res_hi, (ull)value, (ull)cl, (ull)ch, why);
        }
        gave = 1;
        EvEnt *ev = ev_new(value, EV_CUT);
        ev_imp_of(&ev->mir, m);   // as it was: born() and the cut below change it
        ev->x[0] = m->gpu_lo, ev->x[1] = m->gpu_hi, ev->x[2] = ev->x[3] = 0;
        if (m->gpu_lo < plo) {   // it keeps its start; the new backing starts at the bind's first page
            ev->x[4] = 0;
            born(m, BORN_BESIDE_FRONT, value, m->gpu_lo, m->gpu_hi);
            g_st.mirror_bytes -= m->gpu_hi - plo;
            m->gpu_hi = plo;
            if (*rs < plo) *rs = plo;
            ev->x[2] = m->gpu_lo, ev->x[3] = m->gpu_hi;
        } else if (m->gpu_hi > phi) {   // it keeps its end
            ev->x[4] = 1;
            born(m, BORN_BESIDE_END, value, m->gpu_lo, m->gpu_hi);
            g_st.mirror_bytes -= phi - m->gpu_lo;
            m->offset += phi - m->gpu_lo;
            m->gpu_lo = phi;
            if (*re > phi) *re = phi;
            ev->x[2] = m->gpu_lo, ev->x[3] = m->gpu_hi;
        } else {   // the bind covers it all: it goes now, and is freed once the raw-access submission is done
            ev->x[4] = 2;
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

// What submission value resolved in backing m, as a hull (beside reads a mirror's; the straddle dump prints an import's
// too) and the count of the resolves. Only the thread recording value gets here with it.
// A submission older than the hull's is not part of it (a growth reads the hull of the submission it is recorded in). A
// resolve of size 1 or less states no extent (the resolver counts size 0 as 1): the hull is then marked loose.
static void note_resolved(Backing *m, uint64_t addr, uint64_t size, uint64_t value) {
    if (value < m->res_value) return;
    if (m->res_value != value) {
        m->res_value = value, m->res_lo = addr, m->res_hi = addr + size, m->res_n = 1, m->res_loose = size <= 1;
        return;
    }
    if (addr < m->res_lo) m->res_lo = addr;
    if (addr + size > m->res_hi) m->res_hi = addr + size;
    m->res_n++;
    if (size <= 1) m->res_loose = 1;
}

// The backing of [addr, addr + size) when the map holds one, stamped, and a mirror brought current; g_lock, shared at
// least.
static int lookup(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    Rec *r, *v;
    Backing *b = find(&addr, size, &r, &v);
    if (!b) return 0;
    if (v) stamp(&v->last_use, value);
    stamp(&r->last_use, value), stamp(&b->last_use, value);
    if (!g_ring_skip) ring_note(value, addr, size, RING_HIT, b->kind, b->gpu_lo, b->gpu_hi);
    if (b->kind == BACKING_MIRROR) {
        if (value) note_resolved(b, addr, size, value);
        mirror_use(b, value);
    } else if (b->kind == BACKING_IMPORT && value && size > BACKING_PROMOTE_MIN) {
        // A bind of a range larger than a uniform's reached an import: the page table's invariant imported every committed
        // page, and a bind in a raw-access submission can only import. The GPU reads an import across the bus, so the next
        // raw-access submission's begin gives it the mirror the rule would have made (promote_hot).
        __atomic_store_n(&b->hot, 1, __ATOMIC_RELAXED);
    }
    if (b->kind == BACKING_IMPORT && value) note_resolved(b, addr, size, value);
    fill(out, b, addr);
    return 1;
}

// Make what [addr, addr + size) needs in submission value, or stop by name with the shape. g_lock exclusive.
static void make(uint64_t addr, uint64_t size, uint64_t value) {
    g_born_value = value;
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
    ring_note(value, addr, size, RING_MISS, BACKING_NONE, 0, 0);
    BackingAsk ask = {addr, size, r->flags};
    int kind = backing_rule(&ask);
    g_ev_cur = ++g_ev_misses;
    if (kind == BACKING_EXTENT) {
        if (!r->ext) r->ext = extent_new(r), changed();
        ring_note(value, addr, size, RING_MADE, BACKING_EXTENT, r->ext->gpu_lo, r->ext->gpu_hi);
        ev_miss(value, addr, size, 0, 0, 0, 0, kind);
        EvEnt *xe = ev_new(value, EV_MADE);
        xe->kind = BACKING_EXTENT, xe->x[0] = r->ext->gpu_lo, xe->x[1] = r->ext->gpu_hi;
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
    ev_miss(value, addr, size, plo, phi, rs, re, kind);
    // In the raw-access submission being recorded, the page table is rewritten at its end to map what this bind makes, so
    // its raw accesses reach it however early they were recorded. That is the same copy as the earlier binds' only where
    // the new backing aliases the guest's memory, as an import does: a mirror made here would be filled in the prologue,
    // before the submission's own writes through the imports it replaces. A later submission's bind can still make it.
    // A mirror the submission already resolved where the bind takes pages cannot give them up either (beside), so it
    // grows over the bind in place; the committed pages the growth leaves uncovered get their imports (fill_pool).
    if (value && raw_last() == value) {
        g_bind.r = r, g_bind.addr = addr, g_bind.size = size, g_bind.kind = kind;
        char why[GROW_WHY] = "no growth was tried";
        int refused = 0;
        for (int i = 0; i < c->nimp; i++) {
            Backing *m = c->imp[i];
            if (m->kind != BACKING_MIRROR || m->res_value != value ||
                !overlaps(plo, phi, m->gpu_lo, m->gpu_hi) ||
                !overlaps(plo, phi, m->res_lo, m->res_hi))
                continue;
            EvSnap pre;
            uint64_t old_lo = m->gpu_lo, old_hi = m->gpu_hi;
            ev_snap(&pre, c, plo < old_lo ? plo : old_lo, phi > old_hi ? phi : old_hi, m, 0, 0, 0);
            if (grow_taken(r, c, m, plo, phi, value, why)) {
                EvEnt *ge = ev_new(value, EV_GREW);
                ge->kind = BACKING_MIRROR, ge->s = pre;
                ge->x[0] = old_lo, ge->x[1] = old_hi, ge->x[2] = m->gpu_lo, ge->x[3] = m->gpu_hi;
                // The imports it replaced lose their pages outside the bind and a newly committed host chunk can extend
                // beyond it: the committed pages no backing covers now get their imports.
                fill_pool(r);
                ring_note(value, addr, size, RING_GREW, BACKING_MIRROR, m->gpu_lo, m->gpu_hi);
                return;
            }
            diag_refusal(m, plo, phi, value, why);
            refused = 1;
            break;
        }
        // The candidate fix, evaluated and not acted on: a mirror the bind cuts that the submission did not resolve there.
        const Backing *dm = NULL;
        uint64_t others = 0;
        for (int i = 0; i < c->nimp && !refused; i++) {
            const Backing *um = c->imp[i];
            if (um->kind != BACKING_MIRROR || !overlaps(plo, phi, um->gpu_lo, um->gpu_hi) ||
                (um->res_value == value && overlaps(plo, phi, um->res_lo, um->res_hi)))
                continue;
            if (!dm) dm = um;
            else others++;
        }
        if (dm) dry_growth(r, c, dm, plo, phi, value, kind, others);
        if (kind == BACKING_MIRROR) kind = BACKING_IMPORT;
        beside(c, plo, phi, &rs, &re, value, why);
    }
    replace_wait(c, rs, re, value, kind == BACKING_MIRROR);
    Backing *b = kind == BACKING_MIRROR ? mirror_admit(r, rs, re) : NULL;
    if (!b) {
        char why[96];
        snprintf(why, sizeof why, "a bind of [%#llx, %#llx)", (ull)addr, (ull)(addr + size));
        b = import_new(rs, re, plo, why);
    }
    born(b, BORN_BIND, value, b->gpu_lo, b->gpu_hi);
    ring_note(value, addr, size, RING_MADE, b->kind, b->gpu_lo, b->gpu_hi);
    EvEnt *me = ev_new(value, EV_MADE);   // the backings cell_put replaces, read before it does
    me->kind = b->kind, me->x[0] = b->gpu_lo, me->x[1] = b->gpu_hi;
    ev_snap(&me->s, c, rs, re, NULL, 0, 0, 0);
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
    g_ev_cur = 0;
    g_ring_skip = 1;   // the bind is in the ring already, written by make
    int found = lookup(addr, size, value, out);
    g_ring_skip = 0;
    if (!found)
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
    if (b->kind == BACKING_IMPORT) note_resolved(b, a, size, value);
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

uint64_t backing_synced_min(void) {
    uint64_t min = UINT64_MAX;
    plat_lock_shared(&g_lock);
    plat_mutex_lock(&g_coh);
    for (int i = 0; i < g_nmirrors; i++) {
        uint64_t s = g_mirrors[i]->fresh ? 0 : g_mirrors[i]->synced;
        if (s < min) min = s;
    }
    plat_mutex_unlock(&g_coh);
    plat_unlock_shared(&g_lock);
    return min;
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
        // The mirror holds these bytes already: the pages the write covers whole are recorded as GPU-current, so the
        // next sync does not hand them back (they are most of a large write); the rest are marked for a copy.
        pagegen_mark_gpu(it->lo, it->lo + it->size);
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
        // An import trimmed by a growth still stands over the pages it gave away (its host import is whole): a decommit of
        // one of them retires it too, so its buffer never lies over uncommitted memory.
        uint64_t a = o->kind == BACKING_IMPORT ? o->span_lo : o->gpu_lo, end = o->kind == BACKING_IMPORT ? o->span_hi : o->gpu_hi;
        int whole = 1;
        while (a < end && whole) {
            int st;
            uint64_t n = rt_mem_commit_run(a, end - a, &st);
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
        born(b, BORN_RESCAN, g_born_value, rs, re);
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
    g_born_value = 0;
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
            Backing *nb = import_new(a, e, a, "the page table's invariant (every committed page mapped)");
            born(nb, BORN_FILL_POOL, g_born_value, a, e);
            cell_put(cell, a, e, nb);
            g_st.raw_fill_imports++, g_st.raw_fill_bytes += e - a;
            EvEnt *fe = ev_new(g_born_value, EV_FILL);
            fe->x[0] = a, fe->x[1] = e;
            a = e;
        }
        a = end;
    }
}

// ---------- the union at the begin of a raw-access submission
// A mirror next to another backing of its cell (an import fill_pool or rescan_cell made when pages were committed beside
// it, or one a bind made beside it) is a boundary a bind of a raw-access submission can straddle: the bind is a miss, the
// mirror is cut, and an import is made over the pages it gave, so that a later bind across the new boundary finds a mirror
// the submission already resolved and an import it used, and stops by name (beside, grow_taken). The pass below removes the
// boundary before any raw access of the submission exists: at its first backing_raw_begin, every chain of abutting
// backings of one cell with a mirror among them is replaced by ONE backing over their union, a mirror when the rule, the cap
// and the admission allow it, else an import (the mirrors in the chain are given up: exact, and read across the bus). It
// leaves alone a chain that a bind could not be moved off: one a member of which the submission wrote on the GPU (or has a
// GPU write pending over the union), one with a backing of another cell on the union. Those keep the named stop.
//
// It runs before the begin is stamped (raw_last() is still the previous raw-access submission), so replace_wait's check that
// a mirror the page table maps is never replaced cannot fire, and it waits for the older submissions that reached the
// members and for their GPU writes into them, so the union's first copy holds them. It cannot run once a raw access of the
// submission exists: only the first begin of a value runs it. What the submission recorded before the begin read the members'
// buffers, which live until it is done; what it records after reads the union, equal or newer bytes (the tolerance a
// promotion already relies on).
static int chain_end(const Cell *c, int i) {
    int j = i;
    while (j + 1 < c->nimp && c->imp[j]->gpu_hi == c->imp[j + 1]->gpu_lo) j++;
    return j;
}

static int chain_mirrors(const Cell *c, int i, int j) {
    int n = 0;
    for (int k = i; k <= j; k++) n += c->imp[k]->kind == BACKING_MIRROR;
    return n;
}

// UK_MIRROR when the chain c->imp[i..j] may be replaced, else the reason it is left alone (UK_LEFT_*). Pure.
static int union_check(const Rec *r, const Cell *c, int i, int j, uint64_t value) {
    uint64_t lo = c->imp[i]->gpu_lo, hi = c->imp[j]->gpu_hi;
    if (lo < c->lo || hi > c->hi) return UK_LEFT_EDGE;
    for (int k = i; k <= j; k++)
        if (__atomic_load_n(&c->imp[k]->written, __ATOMIC_ACQUIRE) == value) return UK_LEFT_WRITTEN;
    int pending = 0;
    plat_mutex_lock(&g_coh);
    for (int n = 0; n < g_npend; n++)
        pending |= g_pend[n].value == value && overlaps(lo, hi, g_pend[n].lo, g_pend[n].hi);
    plat_mutex_unlock(&g_coh);
    if (pending) return UK_LEFT_WRITTEN;
    for (int n = 0; n < g_index.n; n++) {
        const Rec *other = (const Rec *)g_index.e[n].key;
        if (other->cls != CLS_CPU) continue;
        for (int m = 0; m < other->ncells; m++) {
            const Cell *cell = &other->cells[m];
            if (other == r && cell == c) continue;
            for (int k = 0; k < cell->nimp; k++)
                if (overlaps(lo, hi, cell->imp[k]->gpu_lo, cell->imp[k]->gpu_hi)) return UK_LEFT_CELL;
        }
    }
    return UK_MIRROR;
}

// What the union of c->imp[i..j] is: UK_MIRROR, or the reason it is an import (UK_IMPORT_RULE, UK_IMPORT_CAP). The questions
// promote_hot asks, with the bytes of the chain's own mirrors given back. A union that would pass the cap on the mirrors'
// total is still a mirror when its net add is at most BACKING_MIRROR_GROW_MAX, as a growth is, and the net adds of such unions
// over a run stay at most BACKING_UNION_OVERSHOOT_MAX; *over is the net add of a mirror that passes the cap only so (else 0).
// Pure.
static int union_kind(const Rec *r, const Cell *c, int i, int j, uint64_t *over) {
    uint64_t lo = c->imp[i]->gpu_lo, hi = c->imp[j]->gpu_hi, mine = 0;
    for (int k = i; k <= j; k++)
        if (c->imp[k]->kind == BACKING_MIRROR) mine += c->imp[k]->gpu_hi - c->imp[k]->gpu_lo;
    *over = 0;
    BackingAsk ask = {lo, hi - lo, r->flags};
    if (!has_mirrors() || !g_mirror_hook || backing_rule(&ask) != BACKING_MIRROR) return UK_IMPORT_RULE;
    if (g_st.mirror_bytes - mine + up(hi - lo, BACKING_EXTENT_ALIGN) <= BACKING_PROMOTE_MIRROR_MAX) return UK_MIRROR;
    uint64_t net = (hi - lo) - mine;
    if (net > BACKING_MIRROR_GROW_MAX || g_st.raw_union_overcap_bytes + net > BACKING_UNION_OVERSHOOT_MAX)
        return UK_IMPORT_CAP;
    *over = net;
    return UK_MIRROR;
}

// Replace c->imp[i..j] by one backing over their union (union_check passed), in submission value: wait for the older
// submissions that reached them, take the new backing, and let cell_put defer every member. uk is union_kind's answer: a
// mirror for UK_MIRROR, else an import, and the event says why (UK_IMPORT_RULE, UK_IMPORT_CAP). When the admission refuses
// a mirror the union is an import (UK_IMPORT_ADMIT), unless promoting (a promotion leaves the chain as it was: 0). g_lock
// exclusive.
static int union_make(Rec *r, Cell *c, int i, int j, uint64_t value, int uk, uint64_t over, int promoting) {
    int kind = uk == UK_MIRROR ? BACKING_MIRROR : BACKING_IMPORT;
    uint64_t lo = c->imp[i]->gpu_lo, hi = c->imp[j]->gpu_hi, mine = 0;
    int n = j - i + 1, nm = chain_mirrors(c, i, j);
    for (int k = i; k <= j; k++)
        if (c->imp[k]->kind == BACKING_MIRROR) mine += c->imp[k]->gpu_hi - c->imp[k]->gpu_lo;
    replace_wait(c, lo, hi, value, kind == BACKING_MIRROR);
    Backing *b = kind == BACKING_MIRROR ? mirror_admit(r, lo, hi) : NULL;
    if (!b && promoting) return 0;
    int how = kind == BACKING_MIRROR ? (b ? UK_MIRROR : UK_IMPORT_ADMIT) : uk;
    if (!b) b = import_new(lo, hi, lo, "the union of abutting backings");
    born(b, BORN_UNION, value, lo, hi);
    EvEnt *ev = ev_new(value, EV_UNION);   // the members as they were, read before cell_put replaces them
    ev->kind = b->kind;
    ev_snap(&ev->s, c, lo, hi, NULL, 0, 0, 0);
    g_st.raw_union_chains++;
    if (b->kind == BACKING_MIRROR) g_st.raw_union_mirror++, g_st.raw_union_overcap_bytes += over;
    else g_st.raw_union_import++;
    g_st.raw_union_bytes += hi - lo;
    int64_t net = (int64_t)(b->kind == BACKING_MIRROR ? hi - lo : 0) - (int64_t)mine;
    g_st.raw_union_net_bytes += net;
    cell_put(c, lo, hi, b);
    ev->x[0] = lo, ev->x[1] = hi, ev->x[2] = (uint64_t)n, ev->x[3] = (uint64_t)nm, ev->x[4] = (uint64_t)how;
    ev->x[5] = (uint64_t)net, ev->x[6] = g_st.mirror_bytes;
    return 1;
}

// Whether the chain [lo, hi) of cell c left alone for why is one the pass has not left alone before (and marks it seen). A
// table that is full counts every chain: the counter may then repeat, never miss.
static int left_is_new(const Rec *r, const Cell *c, uint64_t lo, uint64_t hi, int why) {
    for (int n = 0; n < g_nleft; n++) {
        LeftChain *l = &g_left[n];
        if (l->r == r && l->k1 == c->k1 && l->k2 == c->k2 && l->lo == lo && l->hi == hi && l->why == why) return l->seen = 1, 0;
    }
    if (g_nleft < LEFT_MAX) g_left[g_nleft++] = (LeftChain){r, c->k1, c->k2, lo, hi, why, 1};
    return 1;
}

// The pass, at the first begin of value and before its stamp (backing_raw_begin). A scan finds the map as the last clean scan
// left it when the map's version has not moved since (a backing entering or leaving moves it), so a steady scene pays one
// compare. A chain left for a reason that holds until the map moves (another cell, the cell's edge) does not keep the pass
// looking; one left because of a write of the value does, since the next value may not have it. A chain left alone is
// counted and written to the ring once. g_lock exclusive.
static void union_pass(uint64_t value) {
    if (!g_union_on || backing_version() == g_adj_version) return;
    int left = 0;
    for (int n = 0; n < g_nleft; n++) g_left[n].seen = 0;
    for (int n = 0; n < g_index.n; n++) {
        Rec *r = (Rec *)g_index.e[n].key;
        if (r->cls != CLS_CPU) continue;
        for (int m = 0; m < r->ncells; m++) {
            Cell *c = &r->cells[m];
            for (int i = 0; i < c->nimp;) {
                int j = chain_end(c, i);
                if (j == i || !chain_mirrors(c, i, j)) {
                    i = j + 1;
                    continue;
                }
                int why = union_check(r, c, i, j, value);
                if (why != UK_MIRROR) {
                    uint64_t clo = c->imp[i]->gpu_lo, chi = c->imp[j]->gpu_hi;
                    if (left_is_new(r, c, clo, chi, why)) {
                        g_st.raw_union_skipped[why - UK_LEFT_WRITTEN]++;
                        EvEnt *ev = ev_new(value, EV_UNION);
                        ev->x[0] = clo, ev->x[1] = chi, ev->x[2] = (uint64_t)(j - i + 1);
                        ev->x[3] = (uint64_t)chain_mirrors(c, i, j), ev->x[4] = (uint64_t)why, ev->x[6] = g_st.mirror_bytes;
                    }
                    left += why == UK_LEFT_WRITTEN;   // the other reasons hold until the map moves; this one depends on the value
                    i = j + 1;
                    continue;
                }
                uint64_t over, lo = c->imp[i]->gpu_lo;
                int kind = union_kind(r, c, i, j, &over);
                union_make(r, c, i, j, value, kind, over, 0);
                int at = 0;   // the list changed: on from the place of the union (what is below it is not in a chain with it)
                while (at < c->nimp && c->imp[at]->gpu_lo < lo) at++;
                i = at + 1;
            }
        }
    }
    int k = 0;
    for (int n = 0; n < g_nleft; n++)
        if (g_left[n].seen) g_left[k++] = g_left[n];
    g_nleft = k;
    if (!left) g_adj_version = backing_version();
}

// A hot import that the cap on the mirrors holds back stays an import, the pool's first one said so.
static void promote_capped(Rec *r, Backing *b, uint64_t lo, uint64_t hi, uint64_t value) {
    // the raw path writes back across the bus every page its stores reach and keeps a sink over each mirror: bounded
    __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);
    ev_promote(value, lo, hi, 1);
    if (!r->mirror_logged) {
        r->mirror_logged = 1;
        rt_log("gfx: vulkan: backing map: the mirror of [%#llx, %#llx) (%llu KiB) in pool [%#llx, %#llx) is not made: "
               "the mirrors would pass %llu MiB, the most a promotion makes; imported", (ull)lo, (ull)hi,
               (ull)((hi - lo) >> 10), (ull)r->lo, (ull)r->hi, (ull)(BACKING_PROMOTE_MIRROR_MAX >> 20));
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
                if (g_union_on) {
                    // Promoting b beside a backing it abuts would make the boundary the union pass removes: the chain it
                    // is in is promoted whole, or not at all. A chain with a mirror among its members is one the pass left alone.
                    int ci = k, cj = k;
                    while (ci > 0 && c->imp[ci - 1]->gpu_hi == c->imp[ci]->gpu_lo) ci--;
                    while (cj + 1 < c->nimp && c->imp[cj]->gpu_hi == c->imp[cj + 1]->gpu_lo) cj++;
                    if (cj > ci) {
                        uint64_t clo = c->imp[ci]->gpu_lo, chi = c->imp[cj]->gpu_hi, over;
                        if (chain_mirrors(c, ci, cj) || union_check(r, c, ci, cj, value) != UK_MIRROR) {
                            __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);
                            continue;
                        }
                        int uk = union_kind(r, c, ci, cj, &over);
                        if (uk == UK_IMPORT_RULE) {
                            __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);   // the rule says import: the chain stays imports
                            continue;
                        }
                        if (uk == UK_IMPORT_CAP) {
                            promote_capped(r, b, clo, chi, value);
                            continue;
                        }
                        if (!union_make(r, c, ci, cj, value, UK_MIRROR, over, 1)) {
                            __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);
                            ev_promote(value, clo, chi, 2);
                            return;
                        }
                        ev_promote(value, clo, chi, 0);
                        g_st.raw_promoted++, g_st.raw_promote_bytes += chi - clo;
                        k = -1;   // the list changed: this cell again from its start
                        continue;
                    }
                }
                if (g_st.mirror_bytes + up(hi - lo, BACKING_EXTENT_ALIGN) > BACKING_PROMOTE_MIRROR_MAX) {
                    promote_capped(r, b, lo, hi, value);
                    continue;
                }
                Backing *m = mirror_admit(r, lo, hi);
                if (!m) {   // no room (or no write watch): it stays an import, and the rest wait for the next begin
                    __atomic_store_n(&b->hot, 0, __ATOMIC_RELAXED);
                    ev_promote(value, lo, hi, 2);
                    return;
                }
                replace_wait(c, lo, hi, value, 1);
                born(m, BORN_PROMOTE, value, lo, hi);
                cell_put(c, lo, hi, m);   // the import is deferred and the mirror takes its place in the cell
                ev_promote(value, lo, hi, 0);
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
    int first = before < value;   // the first begin of this value: no raw access of it exists yet
    if (before && before < value) __atomic_store_n(&g_raw_prev, before, __ATOMIC_RELEASE);   // the one before this
    g_born_value = value;
    g_begin_now = value;   // its events are the begin's, whether the stamp is made yet or not
    g_st.raw_begins++;
    EvEnt *be = ev_new(value, EV_BEGIN);
    be->x[0] = g_st.mirror_bytes, be->x[1] = g_st.mirrors;
    if (!g_filled || e != g_fill_epoch || g_pools_version != g_fill_pools) {
        g_filled = 1, g_fill_epoch = e, g_fill_pools = g_pools_version;   // read before the pass, as backing_sync does
        g_st.raw_passes++;
        for (int i = 0; i < g_index.n; i++) {
            Rec *r = (Rec *)g_index.e[i].key;
            if (r->cls == CLS_GPU && !r->ext) r->ext = extent_new(r), changed(), g_st.raw_fill_extents++;
            else if (r->cls == CLS_CPU) fill_pool(r);
        }
    }
    // The imports above abut the mirrors beside them: the pass joins every such chain into one backing, before the stamp
    // below (raw_last() is the previous raw-access submission's until then, so no mirror the table maps is in the way of
    // replace_wait) and only the first time (a second call finds raw accesses of the value recorded).
    if (first) union_pass(value);
    stamp(&g_raw_last, value);
    g_begin_now = 0;
    promote_hot(value);
    plat_unlock(&g_lock);
    bound();
    backing_sync_mirrors(value);
}

void backing_set_union(int on) {
    plat_lock(&g_lock);
    g_union_on = !!on;
    plat_unlock(&g_lock);
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
    out->raw_dry_evals = g_dry.evals, out->raw_dry_would_grow = g_dry.would, out->raw_dry_refused = g_dry.refused, out->raw_dry_mutated = g_dry.mutated;
    memcpy(out->raw_dry_g, g_dry.g, sizeof out->raw_dry_g);
    out->resolves = __atomic_load_n(&g_resolves, __ATOMIC_RELAXED);
    out->resolve_misses = __atomic_load_n(&g_misses, __ATOMIC_RELAXED);
}

static int g_exit_done = 1;   // the once-per-run lines are written (or the map is not open); backing_open arms it
static int g_exit_hooked;     // their stop writer and atexit handler are registered (once per process)
static void exit_writer(int code, const char *line);
static void exit_atexit(void);

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
    memset(g_ring, 0, sizeof g_ring);
    memset(g_ev, 0, sizeof g_ev);
    memset(g_evc, 0, sizeof g_evc);
    memset(&g_dry, 0, sizeof g_dry);
    __atomic_store_n(&g_dump_dry_done, 0, __ATOMIC_RELEASE);
    g_ev_seq = g_ev_misses = g_ev_cur = 0;
    g_ring_seq = 0, g_ring_skip = 0, g_born_value = 0, g_grow_cap_logged = 0;
    const char *union_env = getenv("OOT2_BACKING_UNION");
    g_union_on = !(union_env && union_env[0] == '0');   // the control run of the fix turns the pass off
    g_adj_version = 0, g_begin_now = 0, g_nleft = 0;
    g_raw_last = g_raw_prev = 0, g_filled = 0, g_pools_version = g_fill_pools = g_fill_epoch = 0;
    g_open = 1;
    __atomic_store_n(&g_exit_done, 0, __ATOMIC_RELEASE);
    if (!g_exit_hooked) g_exit_hooked = 1, rt_add_stop_writer(exit_writer), atexit(exit_atexit);
    g_bt_stop = 0, g_bt_want = 1, g_bt_running = 1;
    g_bt = plat_thread_start(bt_main, NULL, BLOCK_THREAD_STACK);
    if (!g_bt) rt_fatal("gfx: vulkan: backing map: the block thread did not start");
}

int backing_is_open(void) { return g_open; }

// The once-per-run lines of the counters (the dry evaluation, the caps and unions, the growths), written by whichever comes
// first: backing_close, a stop of the run (rt_run_stop_hook: a named stop, a guest ExitProcess, the census limit) or the
// process returning from main. The counters are read as they stand, with no lock: the stopping thread may hold g_lock (a
// named stop is raised under it), and a line off by a count in flight is better than none. Once a run (g_exit_done). The
// dry evaluation's line is left out only when a straddle stop's dump (diag_stop) printed it, which it flags in
// g_dump_dry_done; every other end (a clean end, the census limit, any other named stop) writes it.

static void exit_lines(void) {
    if (__atomic_exchange_n(&g_exit_done, 1, __ATOMIC_ACQ_REL)) return;
    if (g_st.raw_begins && !__atomic_load_n(&g_dump_dry_done, __ATOMIC_ACQUIRE)) {   // once per run that used the raw path: what the candidate growth would have done (unless a straddle stop's dump printed it)
        char d[260];
        size_t at = 0;
        for (int i = 0; i < DRY_N; i++)
            at += sn(d + at, sizeof d - at, "%s%s %llu", i ? ", " : "", k_dry[i], (ull)g_dry.g[i]);
        rt_log("gfx: vulkan: backing map: dry growth at exit: %llu raw-access submissions, %llu binds cut a mirror the "
               "submission had not resolved there, %llu would have grown it, %llu refused (refusals: %s), the evaluation changed the map %llu times",
               (ull)g_st.raw_begins, (ull)g_dry.evals, (ull)g_dry.would, (ull)g_dry.refused, d, (ull)g_dry.mutated);
    }
    if (g_st.raw_begins) {
        char caps[480];
        cap_counts(caps, sizeof caps);
        rt_log("gfx: vulkan: backing map: mirror totals at exit: %s", caps);
    }
    if (g_st.raw_grows || g_st.raw_heirs || g_st.raw_refused[GI_NEWER] + g_st.raw_refused[GI_WRITTEN] +
        g_st.raw_refused[GI_NOHULL] + g_st.raw_refused[GI_LOOSE] + g_st.raw_refused[GI_HULL] + g_st.raw_refused[GI_UP]) {
        char counts[320];
        growth_counts(counts, sizeof counts);
        rt_log("gfx: vulkan: backing map: raw growth at exit: %s", counts);
    }
}

static void exit_writer(int code, const char *line) {
    (void)code, (void)line;
    exit_lines();
}

static void exit_atexit(void) { exit_lines(); }

void backing_close(void) {
    if (!g_open) return;
    exit_lines();
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
