// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's workload counters (oot2/replay_count.h): the blocks, their sum, and the lines that print them.
#include "oot2/replay_count.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/runtime.h"
#include "plat.h"

#define RC_BLOCKS 64        // threads that count, one block each; a thread past them shares the last block (racy, said in the line)
#define RC_WINDOW 600       // presents between the lines of a run that gave no window
#define RC_LINE 1500        // a line is cut here and goes on in another of the same group
#define RC_HOOK_KEYS 256

int rc_on;
_Thread_local RcBlock *rc_tl;

static RcBlock g_blk[RC_BLOCKS];
static int g_nblk;                  // atomic: blocks claimed
static int g_shared;                // atomic: threads that got the shared last block
static unsigned g_window;
static uint64_t g_prev[RC_N];       // the sum at the last window line
static uint64_t g_last_present;     // the last present rc_present was told of
static uint64_t g_window_start;     // the present of the last window line
static int g_writer_added;

static const char *const g_names[RC_N] = {
#define RC_NAME(id, name) name,
    RC_LIST(RC_NAME)
#undef RC_NAME
};

const char *rc_name(int id) { return id >= 0 && id < RC_N ? g_names[id] : "?"; }

RcBlock *rc_claim(void) {
    int i = __atomic_fetch_add(&g_nblk, 1, __ATOMIC_RELAXED);
    if (i >= RC_BLOCKS) {
        __atomic_fetch_add(&g_shared, 1, __ATOMIC_RELAXED);
        i = RC_BLOCKS - 1;
    }
    rc_tl = &g_blk[i];
    return rc_tl;
}

void rc_sum(uint64_t out[RC_N]) {
    memset(out, 0, sizeof(uint64_t) * RC_N);
    int n = __atomic_load_n(&g_nblk, __ATOMIC_RELAXED);
    if (n > RC_BLOCKS) n = RC_BLOCKS;
    for (int b = 0; b < n; b++)
        for (int i = 0; i < RC_N; i++) out[i] += g_blk[b].c[i];
}

// ---------- lines
struct RcLine {
    char buf[RC_LINE + 128];
    size_t n, hdr;
    const char *kind;
    uint64_t present, presents;
    int window;
    char group[24];
};

static struct {
    char key[48];
    uint64_t prev;
} g_hook_prev[RC_HOOK_KEYS];
static int g_nhook_prev;
static RcHook g_hooks[8];
static int g_nhooks;

static void line_open(RcLine *l) {
    l->n = (size_t)snprintf(l->buf, sizeof l->buf, "rc kind=%s group=%s present=%llu presents=%llu", l->kind, l->group,
                            (unsigned long long)l->present, (unsigned long long)l->presents);
    l->hdr = l->n;
}

static void line_flush(RcLine *l) {
    if (l->n > l->hdr) rt_log("%s", l->buf);
    l->n = l->hdr = 0;
}

void rc_group(RcLine *l, const char *name) {
    if (l->group[0]) line_flush(l);
    snprintf(l->group, sizeof l->group, "%s", name);
    line_open(l);
}

static void put_value(RcLine *l, const char *key, uint64_t v) {
    char kv[96];
    int len = snprintf(kv, sizeof kv, " %s=%llu", key, (unsigned long long)v);
    if (l->n + (size_t)len > RC_LINE) {
        line_flush(l);
        line_open(l);
    }
    memcpy(l->buf + l->n, kv, (size_t)len + 1);
    l->n += (size_t)len;
}

void rc_put(RcLine *l, const char *key, uint64_t cumulative) {
    if (!l->window) {
        put_value(l, key, cumulative);
        return;
    }
    int i = 0;
    while (i < g_nhook_prev && strcmp(g_hook_prev[i].key, key)) i++;
    if (i == g_nhook_prev) {
        if (i == RC_HOOK_KEYS) {   // more keys than the table: printed whole, not as a change
            put_value(l, key, cumulative);
            return;
        }
        snprintf(g_hook_prev[i].key, sizeof g_hook_prev[i].key, "%s", key);
        g_hook_prev[i].prev = 0;
        g_nhook_prev++;
    }
    uint64_t prev = g_hook_prev[i].prev;
    g_hook_prev[i].prev = cumulative;
    put_value(l, key, cumulative >= prev ? cumulative - prev : 0);
}

void rc_put_raw(RcLine *l, const char *key, uint64_t v) { put_value(l, key, v); }

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

void rc_series_add(RcSeries *s, uint64_t v) {
    if (!s->v) {
        s->v = calloc(RC_SERIES_CAP, sizeof *s->v);
        if (!s->v) return;   // out of memory: the series stays empty
    }
    uint64_t n = __atomic_load_n(&s->n, __ATOMIC_RELAXED);
    s->v[n % RC_SERIES_CAP] = v;
    __atomic_store_n(&s->n, n + 1, __ATOMIC_RELEASE);
}

void rc_series_put(RcLine *l, RcSeries *s) {
    uint64_t end = __atomic_load_n(&s->n, __ATOMIC_ACQUIRE), from = s->read;
    if (end - from > RC_SERIES_CAP) {
        s->dropped += end - from - RC_SERIES_CAP;
        from = end - RC_SERIES_CAP;
    }
    uint64_t count = end - from, sum = 0, p50 = 0, p90 = 0, max = 0, dropped = s->dropped;
    uint64_t *copy = count ? malloc(count * sizeof *copy) : NULL;
    if (count && copy) {
        for (uint64_t i = 0; i < count; i++) sum += copy[i] = s->v[(from + i) % RC_SERIES_CAP];
        qsort(copy, count, sizeof *copy, cmp_u64);
        p50 = copy[(count - 1) / 2], p90 = copy[(count * 9 + 9) / 10 - 1], max = copy[count - 1];
    }
    free(copy);
    s->read = end, s->dropped = 0;
    char key[64];
    snprintf(key, sizeof key, "%s_n", s->name), put_value(l, key, count);
    snprintf(key, sizeof key, "%s_sum", s->name), put_value(l, key, sum);
    snprintf(key, sizeof key, "%s_p50", s->name), put_value(l, key, p50);
    snprintf(key, sizeof key, "%s_p90", s->name), put_value(l, key, p90);
    snprintf(key, sizeof key, "%s_max", s->name), put_value(l, key, max);
    if (dropped) snprintf(key, sizeof key, "%s_dropped", s->name), put_value(l, key, dropped);
}

void rc_hook(RcHook h) {
    for (int i = 0; i < g_nhooks; i++)
        if (g_hooks[i] == h) return;
    if (g_nhooks < (int)(sizeof g_hooks / sizeof g_hooks[0])) g_hooks[g_nhooks++] = h;
}

static const char *group_of(int id) {
    if (id < RC_D_FIRST) return "sub";
    if (id < RC_BANK_CALLS_C0) return "draw";
    if (id < RC_UNI_CALLS) return "bank";
    if (id < RC_DG_CALLS_VS) return "uniform";
    if (id >= RC_T_FENCE_N) return "time";
    return "ledger";
}

void rc_print(const char *kind, uint64_t present) {
    static PlatMutex lock = PLAT_MUTEX_INIT;
    plat_mutex_lock(&lock);
    RcLine *l = malloc(sizeof *l);
    if (!l) {
        plat_mutex_unlock(&lock);
        return;
    }
    memset(l, 0, sizeof *l);
    l->kind = kind;
    l->window = strcmp(kind, "exit") != 0;
    l->present = present;
    l->presents = l->window ? present - g_window_start : present;
    uint64_t sum[RC_N];
    rc_sum(sum);
    const char *group = "";
    for (int i = 0; i < RC_N; i++) {
        const char *g = group_of(i);
        if (strcmp(g, group)) rc_group(l, g), group = g;
        uint64_t v = sum[i];
        if (l->window) {
            uint64_t prev = g_prev[i];
            g_prev[i] = v;
            v = v >= prev ? v - prev : 0;
        }
        put_value(l, g_names[i], v);
    }
    rc_group(l, "threads");
    put_value(l, "rc_threads", (uint64_t)(__atomic_load_n(&g_nblk, __ATOMIC_RELAXED) > RC_BLOCKS
                                              ? RC_BLOCKS : __atomic_load_n(&g_nblk, __ATOMIC_RELAXED)));
    put_value(l, "rc_threads_sharing_a_block", (uint64_t)__atomic_load_n(&g_shared, __ATOMIC_RELAXED));
    for (int i = 0; i < g_nhooks; i++) {
        g_hooks[i](l);
    }
    line_flush(l);
    if (l->window) g_window_start = present;
    free(l);
    plat_mutex_unlock(&lock);
}

static void (*g_present_hook)(uint64_t);
void rc_set_present_hook(void (*fn)(uint64_t seq)) { g_present_hook = fn; }

void rc_present(uint64_t seq) {
    if (g_present_hook) g_present_hook(seq);
    if (!rc_on) return;
    g_last_present = seq;
    if (g_window && seq % g_window == 0) rc_print("window", seq);
}

static void stop_writer(int code, const char *line) {
    (void)code, (void)line;
    if (rc_on) rc_print("exit", g_last_present);
}

void rc_enable(unsigned window) {
    if (!window) {
        rc_on = 0;
        return;
    }
    g_window = window;
    if (!g_writer_added) {
        g_writer_added = 1;
        rt_add_stop_writer(stop_writer);
    }
    rc_on = 1;
}

void rc_reset(void) {
    memset(g_blk, 0, sizeof g_blk);
    memset(g_prev, 0, sizeof g_prev);
    memset(g_hook_prev, 0, sizeof g_hook_prev);
    g_nhook_prev = 0;
    g_last_present = g_window_start = 0;
    g_shared = 0;
}

__attribute__((constructor)) static void rc_init(void) {
    const char *e = getenv("OOT2_REPLAY_COUNTERS");
    if (!e || !*e) return;
    char *end;
    unsigned long v = strtoul(e, &end, 10);
    if (end != e && !*end) {
        rc_enable(v == 1 ? RC_WINDOW : (unsigned)v);
    } else if (!strcmp(e, "on")) {
        rc_enable(RC_WINDOW);
    } else {
        fprintf(stderr, "FATAL: OOT2_REPLAY_COUNTERS is \"%s\"; the values are 1 or on (a line every %d presents), a number "
                "of presents from 2 up, or 0 (off)\n", e, RC_WINDOW);
        exit(70);
    }
}
