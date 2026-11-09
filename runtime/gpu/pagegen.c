// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Page generations (oot2/pagegen.h): the only caller of rt_mem_written.
//
// The table has two levels over the guest's address space: 2,048 pieces of 256 MiB, each made at the first stamp of
// one of its pages, holding two 64-bit generations per 4 KiB page (the page's stamp, and the stamp a GPU write-back gave
// it, if the last stamp is that one) and the newest generation of each of its 2 MiB regions, so a reader skips a region
// with nothing newer than what it asks for. A piece holds 1 MiB of stamps; a heap whose every page a query reported
// needs 32 of them for 8 GiB. The GPU stamp is one slot a page, so its state is bounded by the address space however
// many write-backs a run makes.
//
// Adapted from DKB Recomp runtime/vk_memory.c at 9d8fa02 (GPL-3.0-or-later; there adapted from LAS Recomp
// runtime/nvn/vulkan/exec.c at da35b83, MIT, Copyright (c) 2026 Peiikai, licenses/lasrecomp-MIT.txt, and MacNativeGames
// nvn_dirty.c at b75be8c, MIT, Copyright (c) 2026 jhaizhou, licenses/macnativegames-MIT.txt): every write-watch query
// stamps the pages it saw with a new generation shared by all users, in a two-level table with a per-2 MiB maximum.
// Changed here: generations are 64 bits (no wrap at one per flush), the source is a parameter so the stamping is tested
// without a write watch, the table is the only path to the write watch, and readers get runs of pages.
#include <stdlib.h>
#include <string.h>

#include "oot2/memory.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
#include "plat.h"

#define PIECE_SHIFT 28
#define PAGE_SHIFT 12
#define REGION_SHIFT 21
#define PIECES (RT_ADDRESS_LIMIT >> PIECE_SHIFT)
#define PAGES_PER_PIECE (1u << (PIECE_SHIFT - PAGE_SHIFT))
#define REGIONS_PER_PIECE (1u << (PIECE_SHIFT - REGION_SHIFT))
#define BATCH 4096   // pages one call of the source may report; a full answer is asked again

typedef struct Piece {
    uint64_t page[PAGES_PER_PIECE];
    uint64_t gpu[PAGES_PER_PIECE];        // the stamp a GPU write-back gave the page (pagegen_mark_gpu); 0 for none
    uint64_t region[REGIONS_PER_PIECE];   // the newest stamp of each 2 MiB region
} Piece;

static PlatMutex g_lock = PLAT_MUTEX_INIT;
static Piece *g_pieces[PIECES];
static uint64_t g_now;
static PagegenSource g_src;
static uint64_t g_batch[BATCH];
static PagegenStats g_st;

void pagegen_set_source(PagegenSource src) {
    plat_mutex_lock(&g_lock);
    g_src = src;
    plat_mutex_unlock(&g_lock);
}

static Piece *piece(uint64_t addr, int make) {
    uint64_t i = addr >> PIECE_SHIFT;
    if (i >= PIECES) rt_fatal("page generations: %#llx is not a guest address", (unsigned long long)addr);
    if (!g_pieces[i] && make) {
        g_pieces[i] = calloc(1, sizeof(Piece));
        if (!g_pieces[i]) rt_fatal("page generations: out of memory for a table piece");
        g_st.chunks++;
    }
    return g_pieces[i];
}

// g_lock held.
static void stamp(uint64_t page, uint64_t g) {
    Piece *p = piece(page, 1);
    p->page[(page >> PAGE_SHIFT) & (PAGES_PER_PIECE - 1)] = g;
    p->region[(page >> REGION_SHIFT) & (REGIONS_PER_PIECE - 1)] = g;
}

// The write watch's query: the stand-in a test set, else the operating system's. g_lock held.
static PagegenSource source(void) { return g_src ? g_src : rt_mem_written; }

static int whole_pages(uint64_t lo, uint64_t hi) {
    return lo < hi && !(lo & (PAGEGEN_PAGE - 1)) && !(hi & (PAGEGEN_PAGE - 1)) && hi <= RT_ADDRESS_LIMIT;
}

uint64_t pagegen_scan(uint64_t lo, uint64_t hi) {
    if (!whole_pages(lo, hi))
        rt_fatal("page generations: a query of [%#llx, %#llx), which is not whole guest pages", (unsigned long long)lo,
                 (unsigned long long)hi);
    plat_mutex_lock(&g_lock);
    PagegenSource src = source();
    uint64_t g = ++g_now, t0 = plat_now_ns(), found = 0;
    int tracked = 1;
    for (;;) {
        uint64_t n = src(lo, hi - lo, g_batch, BATCH, 1);
        if (n == UINT64_MAX) {
            tracked = 0;
            break;
        }
        for (uint64_t k = 0; k < n; k++) stamp(g_batch[k] & ~(PAGEGEN_PAGE - 1), g);
        found += n;
        if (n < BATCH) break;   // a full answer reset only what it returned (the write-watch study): ask again
    }
    g_st.scans++, g_st.scan_bytes += hi - lo, g_st.scan_pages += found, g_st.scan_ns += plat_now_ns() - t0;
    g_st.untracked += (uint64_t)!tracked;
    plat_mutex_unlock(&g_lock);
    return tracked ? g : UINT64_MAX;
}

uint64_t pagegen_mark(uint64_t lo, uint64_t hi) {
    lo &= ~(PAGEGEN_PAGE - 1);
    hi = (hi + PAGEGEN_PAGE - 1) & ~(PAGEGEN_PAGE - 1);
    plat_mutex_lock(&g_lock);
    uint64_t g = ++g_now;
    if (lo < hi && hi <= RT_ADDRESS_LIMIT) {   // the GPU window lies above the guest's space: no CPU sees it
        for (uint64_t a = lo; a < hi; a += PAGEGEN_PAGE) stamp(a, g);
        g_st.marks++, g_st.mark_pages += (hi - lo) >> PAGE_SHIFT;
    }
    plat_mutex_unlock(&g_lock);
    return g;
}

uint64_t pagegen_mark_gpu(uint64_t lo, uint64_t hi) {
    uint64_t alo = lo & ~(PAGEGEN_PAGE - 1), ahi = (hi + PAGEGEN_PAGE - 1) & ~(PAGEGEN_PAGE - 1);
    uint64_t glo = (lo + PAGEGEN_PAGE - 1) & ~(PAGEGEN_PAGE - 1), ghi = hi & ~(PAGEGEN_PAGE - 1);   // the whole pages
    plat_mutex_lock(&g_lock);
    uint64_t g = ++g_now;
    if (alo < ahi && ahi <= RT_ADDRESS_LIMIT) {
        for (uint64_t a = alo; a < ahi; a += PAGEGEN_PAGE) stamp(a, g);
        g_st.marks++, g_st.mark_pages += (ahi - alo) >> PAGE_SHIFT;
        if (glo < ghi) {
            for (uint64_t a = glo; a < ghi; a += PAGEGEN_PAGE)
                piece(a, 1)->gpu[(a >> PAGE_SHIFT) & (PAGES_PER_PIECE - 1)] = g;
            // The runtime's own copy dirtied these pages in the write watch: take that mark away (its pages are not
            // stamped again, so they stay GPU-current), or the next query would report the write-back as a CPU write.
            PagegenSource src = source();
            for (;;) {
                uint64_t n = src(glo, ghi - glo, g_batch, BATCH, 1);
                if (n == UINT64_MAX || n < BATCH) break;
            }
            g_st.gpu_marks++, g_st.gpu_mark_pages += (ghi - glo) >> PAGE_SHIFT;
        }
    }
    plat_mutex_unlock(&g_lock);
    return g;
}

uint64_t pagegen_gpu_stamp(uint64_t page) {
    plat_mutex_lock(&g_lock);
    const Piece *p = page < RT_ADDRESS_LIMIT ? piece(page, 0) : NULL;
    uint64_t g = p ? p->gpu[(page >> PAGE_SHIFT) & (PAGES_PER_PIECE - 1)] : 0;
    plat_mutex_unlock(&g_lock);
    return g;
}

uint64_t pagegen_now(void) {
    plat_mutex_lock(&g_lock);
    uint64_t g = g_now;
    plat_mutex_unlock(&g_lock);
    return g;
}

uint64_t pagegen_stamp(uint64_t page) {
    plat_mutex_lock(&g_lock);
    const Piece *p = page < RT_ADDRESS_LIMIT ? piece(page, 0) : NULL;
    uint64_t g = p ? p->page[(page >> PAGE_SHIFT) & (PAGES_PER_PIECE - 1)] : 0;
    plat_mutex_unlock(&g_lock);
    return g;
}

static uint64_t runs_of(uint64_t lo, uint64_t hi, uint64_t since, PagegenRun fn, void *ctx, int cpu_only) {
    lo &= ~(PAGEGEN_PAGE - 1);
    hi = (hi + PAGEGEN_PAGE - 1) & ~(PAGEGEN_PAGE - 1);
    if (hi > RT_ADDRESS_LIMIT) hi = RT_ADDRESS_LIMIT;
    uint64_t pages = 0, run = 0;   // run: the start of the run being collected, 0 for none
    plat_mutex_lock(&g_lock);
    for (uint64_t a = lo; a < hi;) {
        uint64_t region_end = (a | ((1ull << REGION_SHIFT) - 1)) + 1;
        if (region_end > hi) region_end = hi;
        const Piece *p = piece(a, 0);
        if (!p || p->region[(a >> REGION_SHIFT) & (REGIONS_PER_PIECE - 1)] <= since) {
            if (run) fn(ctx, run, a), run = 0;
            a = region_end;
            continue;
        }
        for (; a < region_end; a += PAGEGEN_PAGE) {
            uint64_t k = (a >> PAGE_SHIFT) & (PAGES_PER_PIECE - 1);
            // A page whose newest stamp is a GPU write-back's holds, in a mirror that took its bytes, what guest memory
            // holds: the CPU did not write it since (cpu_only skips it).
            int newer = p->page[k] > since && !(cpu_only && p->page[k] == p->gpu[k]);
            if (newer && !run) run = a;
            else if (!newer && run) fn(ctx, run, a), run = 0;
            pages += (uint64_t)newer;
        }
    }
    if (run) fn(ctx, run, hi);
    plat_mutex_unlock(&g_lock);
    return pages;
}

uint64_t pagegen_runs(uint64_t lo, uint64_t hi, uint64_t since, PagegenRun fn, void *ctx) {
    return runs_of(lo, hi, since, fn, ctx, 0);
}

uint64_t pagegen_runs_cpu(uint64_t lo, uint64_t hi, uint64_t since, PagegenRun fn, void *ctx) {
    return runs_of(lo, hi, since, fn, ctx, 1);
}

void pagegen_stats(PagegenStats *out) {
    plat_mutex_lock(&g_lock);
    *out = g_st;
    out->generation = g_now;
    plat_mutex_unlock(&g_lock);
}

void pagegen_reset(void) {
    plat_mutex_lock(&g_lock);
    for (uint64_t i = 0; i < PIECES; i++) free(g_pieces[i]), g_pieces[i] = NULL;
    memset(&g_st, 0, sizeof g_st);
    g_now = 0;
    g_src = NULL;
    plat_mutex_unlock(&g_lock);
}
