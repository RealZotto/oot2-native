// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Which regions of a CPU-written texture are stale (oot2/texsync.h). The upload again by write generation follows DKB
// Recomp's runtime/vk_texture.c (GPL-3.0-or-later; its behaviour was ported there from MacNativeGames d2r_gfx and LAS
// Recomp, MIT): a texture is current at the generation it was last read at, and a page stamped after that makes the
// regions over it stale. Changed here, the stamps are the page generations of oot2/pagegen.h (one owner of the write
// watch, queried here by range and never reset by this file), the unit is a region (a level of a layer) rather than the
// whole texture, and storage nothing ever wrote is zeros that are never read from guest memory.
#include <string.h>

#include "oot2/memory.h"
#include "oot2/nvn_tex.h"
#include "oot2/pagegen.h"
#include "oot2/texsync.h"
#include "oot2/untile.h"
#include "plat.h"

static PlatMutex g_mu = PLAT_MUTEX_INIT;
static TexSyncStats g_st;
static TexSyncCommit g_probe;

void texsync_set_commit_probe(TexSyncCommit fn) { g_probe = fn; }

// Does any page of [lo, hi) hold committed memory? An address the probe knows no region of counts as committed (it is read).
static int any_committed(uint64_t lo, uint64_t hi) {
    TexSyncCommit probe = g_probe ? g_probe : rt_mem_commit_run;
    for (uint64_t a = lo; a < hi;) {
        int committed = 1;
        uint64_t n = probe(a, hi - a, &committed);
        if (n == UINT64_MAX || committed) return 1;
        if (!n) break;
        a += n;
    }
    return 0;
}

unsigned texsync_max_regions(const nvn_texdesc *d) {
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    if (nvn_tex_is_linear(d)) return 1;
    return (d->levels ? d->levels : 1) * (L.layers ? L.layers : 1);
}

static void flag_run(void *ctx, uint64_t lo, uint64_t hi) {
    (void)lo, (void)hi;
    *(int *)ctx = 1;
}

// Does any page of [lo, hi) carry a stamp newer than since?
static int newer(uint64_t lo, uint64_t hi, uint64_t since) {
    int any = 0;
    pagegen_runs(lo, hi, since, flag_run, &any);
    return any;
}

int texsync_regions(const nvn_texdesc *d, uint64_t storage, TexSync *st, TexRegion *out, int max, const char **why) {
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    int linear = nvn_tex_is_linear(d);
    UntileGeom g;
    if (!linear && !untile_geom(d, 0, 0, &g, why)) return -1;
    if (!L.size) {
        *why = "a texture of no storage";
        return -1;
    }
    uint64_t lo = storage & ~(PAGEGEN_PAGE - 1), hi = (storage + L.size + PAGEGEN_PAGE - 1) & ~(PAGEGEN_PAGE - 1);
    uint64_t gen = pagegen_scan(lo, hi);
    int tracked = gen != UINT64_MAX;
    uint64_t since = st->defined && tracked ? st->gen : 0;
    int n = 0, zeros = 0;
    unsigned levels = linear ? 1 : (d->levels ? d->levels : 1), layers = linear ? 1 : L.layers;
    for (unsigned layer = 0; layer < layers; layer++)
        for (unsigned level = 0; level < levels; level++) {
            uint64_t base = 0, span = L.size;
            if (!linear) {
                untile_geom(d, level, layer, &g, why);
                base = g.base, span = g.span;
            }
            uint64_t rlo = storage + base, rhi = rlo + span;
            int stale = !tracked || !st->defined || newer(rlo, rhi, since);
            if (!stale) continue;
            int zero = tracked && !st->defined && !newer(rlo, rhi, 0) && !any_committed(rlo & ~(PAGEGEN_PAGE - 1), (rhi + PAGEGEN_PAGE - 1) & ~(PAGEGEN_PAGE - 1));
            if (n < max) out[n] = (TexRegion){level, layer, zero};
            n++;
            zeros += zero;
        }
    plat_mutex_lock(&g_mu);
    g_st.queries++;
    g_st.untracked += (uint64_t)!tracked;
    g_st.regions += (uint64_t)n;
    g_st.zero_regions += (uint64_t)zeros;
    g_st.unchanged += (uint64_t)(n == 0);
    plat_mutex_unlock(&g_mu);
    if (n > max) return n;   // the caller asks again with room: the state is not consumed (the stamps stay)
    if (tracked) st->gen = gen;
    st->defined = 1;
    return n;
}

void texsync_stats(TexSyncStats *out) {
    plat_mutex_lock(&g_mu);
    *out = g_st;
    plat_mutex_unlock(&g_mu);
}

int texsync_stats_try(TexSyncStats *out) {   // 0 when the lock is busy (the counters' lines never wait for it)
    if (!plat_mutex_trylock(&g_mu)) return 0;
    *out = g_st;
    plat_mutex_unlock(&g_mu);
    return 1;
}

void texsync_reset_stats(void) {
    plat_mutex_lock(&g_mu);
    memset(&g_st, 0, sizeof g_st);
    plat_mutex_unlock(&g_mu);
}
