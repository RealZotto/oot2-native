// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The blocks of 1 MiB or more that our own code allocated, by call site (oot2/memline.h): a block the page walk finds that
// no site here made belongs to the driver, by exclusion. Kept in runtime/core so that every build links it.
#include <string.h>

#include "oot2/memline.h"
#include "plat.h"

static MemlineSite g_sites[MEMLINE_SITES];
static int g_nsites;
static uint64_t g_over;   // blocks noted with no room left in the table
static PlatMutex g_lock = PLAT_MUTEX_INIT;

static MemlineSite *site_for(const char *name) {   // g_lock held
    for (int i = 0; i < g_nsites; i++)
        if (g_sites[i].site == name || !strcmp(g_sites[i].site, name)) return &g_sites[i];
    if (g_nsites == MEMLINE_SITES) return NULL;
    g_sites[g_nsites].site = name;
    return &g_sites[g_nsites++];
}

void memline_note(const char *site, uint64_t bytes) {
    if (bytes < (1ull << 20)) return;
    plat_mutex_lock(&g_lock);
    MemlineSite *s = site_for(site);
    if (s) s->live_blocks++, s->live_bytes += bytes, s->total_blocks++;
    else g_over++;
    plat_mutex_unlock(&g_lock);
}

void memline_unnote(const char *site, uint64_t bytes) {
    if (bytes < (1ull << 20)) return;
    plat_mutex_lock(&g_lock);
    MemlineSite *s = site_for(site);
    if (s && s->live_blocks) s->live_blocks--, s->live_bytes -= s->live_bytes < bytes ? s->live_bytes : bytes;
    plat_mutex_unlock(&g_lock);
}

int memline_sites(MemlineSite *out, int max) {
    plat_mutex_lock(&g_lock);
    int n = g_nsites < max ? g_nsites : max;
    memcpy(out, g_sites, (size_t)n * sizeof *out);
    plat_mutex_unlock(&g_lock);
    return n;
}
