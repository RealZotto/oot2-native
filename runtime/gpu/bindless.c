// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The bindless tables' core (oot2/bindless.h): shadow tables per pool, the change log, the ring of copies, the sharing of
// views and samplers and their retirement. No Vulkan type: the host's objects come through BindlessOps.
//
// Locking: one mutex guards everything. Registrations come from any guest thread, acquire and image_slot from the thread
// that holds the back end's submission lock. Nothing blocks while it is held except the host's own calls (a view made,
// a set allocated); the wait for a pending copy is taken with it released.
#include <stdlib.h>
#include <string.h>

#include "oot2/bindless.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHUNK 256u          // entries per chunk of a pool's shadow table
#define LOG_CAP 4096u       // change log records (a power of two)
#define HASH_BUCKETS 65536u
#define BATCH 256

enum { R_TEX, R_SMP, R_SECOND };

typedef struct Res {
    void *obj;
    int sampler;
    int refs;
    uint64_t stamp;               // the newest submission value that used a copy holding it
    uint32_t hash;
    union { BindlessKey view; samp_par samp; } key;
    struct Res *next;             // hash chain, or the retire list
} Res;

typedef struct Entry {
    Res *res;
    int type;                     // the view type the texture asks for (BV_*)
    uint32_t second[BV_TYPES];    // second slot + 1 per type, 0 none
    uint8_t used;                 // registered, and not gone
    tex_t t;
    view_t v;
} Entry;

typedef struct TexPool {
    uint64_t addr;
    uint32_t count;
    Entry **chunk;
    uint32_t nchunks;
} TexPool;

typedef struct SmpPool {
    uint64_t addr;
    uint32_t count;
    Res **ent;
} SmpPool;

typedef struct Rec {
    int kind;
    uint32_t id;
    void *pool;
} Rec;

typedef struct Copy {
    void *set;
    uint64_t pos, last_value;
    TexPool *tp;
    SmpPool *sp;
    int fresh;
    Res **img, **smp;             // what the copy holds at each slot (NULL: the placeholder)
} Copy;

typedef struct Second {
    Res *res;
    TexPool *pool;
    uint32_t id;
    int type;
} Second;

typedef struct Batch {
    BindlessWrite w[BATCH];
    int n;
} Batch;

static struct {
    int open;
    BindlessOps ops;
    BindlessConfig cfg;
    BindlessClock clock;
    uint32_t primary;
    TexPool **tp;
    int ntp, captp;
    SmpPool **sp;
    int nsp, capsp;
    Copy **copies;
    int ncopies, capcopies, rr;
    Copy *cur;
    Rec *log;
    uint64_t head;
    Res **hash;
    Res *retire;
    Second *sec;
    uint32_t *sec_free;
    uint32_t nsec_free;
    BindlessStats st;
} g;
static PlatMutex g_mu = PLAT_MUTEX_INIT;

typedef unsigned long long ull;

static uint32_t fnv(const void *p, size_t n, uint32_t seed) {
    const uint8_t *b = p;
    uint32_t h = 2166136261u ^ seed;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 16777619u;
    return h;
}

// ---------- view types and keys

int bindless_view_type(const tex_t *t, const view_t *v) {
    int target = v && v->set_target ? v->target : t->d.target;
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: return BV_1D;
    case NVN_TEXTURE_TARGET_2D: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE: case NVN_TEXTURE_TARGET_RECTANGLE: return BV_2D;
    case NVN_TEXTURE_TARGET_3D: return BV_3D;
    case NVN_TEXTURE_TARGET_1D_ARRAY: return BV_1D_ARRAY;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: return BV_2D_ARRAY;
    case NVN_TEXTURE_TARGET_CUBEMAP: return BV_CUBE;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: return BV_CUBE_ARRAY;
    default: return BV_NONE;
    }
}

void bindless_key(const tex_t *t, const view_t *v, int want, BindlessKey *k) {
    memset(k, 0, sizeof *k);
    const nvn_texdesc *d = &t->d;
    k->storage = (uint64_t)(uintptr_t)t->mem;
    k->target = (uint32_t)d->target, k->format = (uint32_t)d->format, k->samples = (uint32_t)d->samples;
    k->flags = (uint32_t)d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET);
    k->pkg = (uint32_t)(d->pkg_block_h & 0xff), k->stride = d->stride;
    k->layout_flags = (uint32_t)d->flags & (NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT);
    k->width = d->width, k->height = d->height, k->depth = d->depth, k->levels = d->levels;
    k->swizzle = (uint32_t)d->swizzle_r | (uint32_t)d->swizzle_g << 3 | (uint32_t)d->swizzle_b << 6 |
                 (uint32_t)d->swizzle_a << 9 | (uint32_t)d->dsmode << 12;
    if (v) {
        k->v_set = (uint32_t)v->set_levels | (uint32_t)v->set_layers << 1 | (uint32_t)v->set_format << 2 |
                   (uint32_t)v->set_target << 3 | (uint32_t)v->set_swizzle << 4 | (uint32_t)v->set_dsmode << 5 |
                   (uint32_t)v->dsmode << 6;
        k->v_format = v->set_format ? v->format : 0, k->v_target = v->set_target ? v->target : 0;
        k->v_base_level = v->set_levels ? v->base_level : 0, k->v_levels = v->set_levels ? v->nlevels : 0;
        k->v_min_layer = v->set_layers ? v->min_layer : 0, k->v_layers = v->set_layers ? v->nlayers : 0;
        if (v->set_swizzle) memcpy(k->v_swz, v->swz, 4);
    }
    k->want = want;
}

// ---------- shared views and samplers

static Res *hash_find(uint32_t h, int sampler, const void *key, size_t n) {
    for (Res *r = g.hash[h & (HASH_BUCKETS - 1)]; r; r = r->next)
        if (r->hash == h && r->sampler == sampler && !memcmp(&r->key, key, n)) return r;
    return NULL;
}

static void hash_add(Res *r) {
    Res **b = &g.hash[r->hash & (HASH_BUCKETS - 1)];
    r->next = *b;
    *b = r;
}

static void hash_remove(Res *r) {
    for (Res **p = &g.hash[r->hash & (HASH_BUCKETS - 1)]; *p; p = &(*p)->next)
        if (*p == r) {
            *p = r->next;
            return;
        }
}

// Take every shared view over storage [lo, hi) out of the hash, so that no registration or bind finds it again. The views
// stay alive for whoever holds them: a copy that shows one keeps it until the copy is brought up to date, and a pending
// submission may still read it. Only sampled views are shared this way (samplers have no storage).
static void hash_detach_storage(uint64_t lo, uint64_t hi) {
    for (uint32_t b = 0; b < HASH_BUCKETS; b++) {
        Res **p = &g.hash[b];
        while (*p) {
            Res *r = *p;
            if (!r->sampler && r->key.view.storage >= lo && r->key.view.storage < hi) {
                *p = r->next;
                continue;
            }
            p = &r->next;
        }
    }
}

// The shared view of (t, v) as `type`; NULL when the host makes none. No reference is taken. demand: the replay binds the
// texture now, so a host that made no view at registration (view_make) is asked again with view_demand.
static Res *view_get(const tex_t *t, const view_t *v, int type, int demand) {
    BindlessKey k;
    bindless_key(t, v, type, &k);
    uint32_t h = fnv(&k, sizeof k, 0);
    Res *r = hash_find(h, 0, &k, sizeof k);
    if (r) return r;
    void *obj = g.ops.view_make(t, v, type);
    if (!obj && demand && g.ops.view_demand) obj = g.ops.view_demand(t, v, type);
    if (!obj) return NULL;
    r = calloc(1, sizeof *r);
    if (!r) rt_fatal("gfx: vulkan: bindless tables: out of memory for a view");
    r->obj = obj, r->hash = h, r->key.view = k;
    hash_add(r);
    g.st.views_made++, g.st.views_live++;
    return r;
}

static Res *sampler_get(const samp_par *s) {
    uint32_t h = fnv(s, sizeof *s, 1);
    Res *r = hash_find(h, 1, s, sizeof *s);
    if (r) return r;
    if (g.st.samplers_live >= g.cfg.max_samplers)
        rt_fatal("gfx: vulkan: bindless tables: %llu distinct sampler states are live and a %llu-th is registered, over "
                 "the %u host samplers the device allows the back end", (ull)g.st.samplers_live,
                 (ull)g.st.samplers_live + 1, g.cfg.max_samplers);
    void *obj = g.ops.sampler_make(s);
    r = calloc(1, sizeof *r);
    if (!r) rt_fatal("gfx: vulkan: bindless tables: out of memory for a sampler");
    r->obj = obj, r->sampler = 1, r->hash = h, r->key.samp = *s;
    hash_add(r);
    g.st.samplers_made++, g.st.samplers_live++;
    return r;
}

static void ref(Res *r) { r->refs++; }

// Drop a reference. A holder that was a copy passes the newest value that used it, so that the object outlives every
// submission that could read it; the entry's own reference passes 0.
static void unref(Res *r, uint64_t stamp) {
    if (!r) return;
    if (stamp > r->stamp) r->stamp = stamp;
    if (--r->refs > 0) return;
    hash_remove(r);
    if (r->sampler) g.st.samplers_live--;
    else g.st.views_live--;
    r->next = g.retire;
    g.retire = r;
}

static void reap(void) {
    if (!g.retire) return;
    uint64_t done = g.clock.done();
    Res **p = &g.retire;
    while (*p) {
        Res *r = *p;
        if (r->stamp > done) {
            p = &r->next;
            continue;
        }
        *p = r->next;
        if (r->sampler) g.ops.sampler_free(r->obj);
        else g.ops.view_free(r->obj);
        g.st.retired++;
        free(r);
    }
}

// ---------- the change log

static void log_add(int kind, void *pool, uint32_t id) {
    g.log[g.head & (LOG_CAP - 1)] = (Rec){kind, id, pool};
    g.head++;
}

// ---------- pools and entries

static TexPool *tpool(const dpool_ref *r) {
    for (int i = 0; i < g.ntp; i++)
        if (g.tp[i]->addr == r->addr) {
            if (r->count > g.tp[i]->count) g.tp[i]->count = r->count;
            return g.tp[i];
        }
    if (g.ntp == g.captp) {
        g.captp = g.captp ? g.captp * 2 : 4;
        g.tp = realloc(g.tp, (size_t)g.captp * sizeof *g.tp);
        if (!g.tp) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    }
    TexPool *p = calloc(1, sizeof *p);
    if (!p) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    p->addr = r->addr, p->count = r->count;
    g.tp[g.ntp++] = p;
    return p;
}

static SmpPool *spool(const dpool_ref *r) {
    for (int i = 0; i < g.nsp; i++)
        if (g.sp[i]->addr == r->addr) {
            if (r->count > g.sp[i]->count) g.sp[i]->count = r->count;
            return g.sp[i];
        }
    if (g.nsp == g.capsp) {
        g.capsp = g.capsp ? g.capsp * 2 : 4;
        g.sp = realloc(g.sp, (size_t)g.capsp * sizeof *g.sp);
        if (!g.sp) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    }
    SmpPool *p = calloc(1, sizeof *p);
    if (!p) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    p->ent = calloc(g.cfg.samplers, sizeof *p->ent);
    if (!p->ent) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    p->addr = r->addr, p->count = r->count;
    g.sp[g.nsp++] = p;
    return p;
}

static Entry *entry_at(TexPool *p, uint32_t id, int make) {
    uint32_t c = id / CHUNK;
    if (c >= p->nchunks) {
        if (!make) return NULL;
        p->chunk = realloc(p->chunk, (size_t)(c + 1) * sizeof *p->chunk);
        if (!p->chunk) rt_fatal("gfx: vulkan: bindless tables: out of memory");
        memset(p->chunk + p->nchunks, 0, (size_t)(c + 1 - p->nchunks) * sizeof *p->chunk);
        p->nchunks = c + 1;
    }
    if (!p->chunk[c]) {
        if (!make) return NULL;
        p->chunk[c] = calloc(CHUNK, sizeof(Entry));
        if (!p->chunk[c]) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    }
    return &p->chunk[c][id % CHUNK];
}

// ---------- second slots

static void second_free_all(Entry *e) {
    for (int t = 0; t < BV_TYPES; t++) {
        if (!e->second[t]) continue;
        uint32_t s = e->second[t] - 1;
        unref(g.sec[s].res, 0);
        g.sec[s].res = NULL;
        g.sec_free[g.nsec_free++] = s;
        log_add(R_SECOND, NULL, s);
        e->second[t] = 0;
        g.st.second_live--;
    }
}

// ---------- copies

static void flush(Copy *c, Batch *b) {
    if (!b->n) return;
    g.ops.write(c->set, b->w, b->n);
    g.st.writes += (uint64_t)b->n;
    b->n = 0;
}

static void put(Copy *c, Batch *b, int kind, uint32_t slot, Res *want) {
    Res **at = kind == BW_IMAGE ? &c->img[slot] : &c->smp[slot];
    if (*at == want) return;
    if (want) ref(want);
    unref(*at, c->last_value);
    *at = want;
    if (b->n == BATCH) flush(c, b);
    b->w[b->n++] = (BindlessWrite){kind, slot, want ? want->obj : NULL};
    if (kind == BW_IMAGE) g.st.image_writes++;
    else g.st.sampler_writes++;
}

static Res *want_image(const Copy *c, uint32_t slot) {
    if (slot >= g.primary) return g.sec[slot - g.primary].res;
    if (!c->tp || slot >= c->tp->count) return NULL;
    Entry *e = entry_at(c->tp, slot, 0);
    return e && e->used ? e->res : NULL;
}

static Res *want_sampler(const Copy *c, uint32_t slot) {
    return c->sp && slot < c->sp->count ? c->sp->ent[slot] : NULL;
}

static void apply(Copy *c, Batch *b, const Rec *r) {
    switch (r->kind) {
    case R_TEX:
        if (r->pool == c->tp) put(c, b, BW_IMAGE, r->id, want_image(c, r->id));
        break;
    case R_SMP:
        if (r->pool == c->sp) put(c, b, BW_SAMPLER, r->id, want_sampler(c, r->id));
        break;
    default:
        put(c, b, BW_IMAGE, g.primary + r->id, want_image(c, g.primary + r->id));
        break;
    }
}

static void sync_copy(Copy *c, TexPool *tp, SmpPool *sp) {
    Batch b = {.n = 0};
    if (c->fresh || c->tp != tp || c->sp != sp || g.head - c->pos > LOG_CAP) {
        g.st.full_syncs++;
        c->tp = tp, c->sp = sp;
        for (uint32_t i = 0; i < g.cfg.images; i++) put(c, &b, BW_IMAGE, i, want_image(c, i));
        for (uint32_t i = 0; i < g.cfg.samplers; i++) put(c, &b, BW_SAMPLER, i, want_sampler(c, i));
        c->fresh = 0;
    } else {
        for (uint64_t i = c->pos; i < g.head; i++) apply(c, &b, &g.log[i & (LOG_CAP - 1)]);
    }
    c->pos = g.head;
    flush(c, &b);
}

static Copy *copy_new(void) {
    Copy *c = calloc(1, sizeof *c);
    if (!c) rt_fatal("gfx: vulkan: bindless tables: out of memory for a copy");
    c->img = calloc(g.cfg.images, sizeof *c->img);
    c->smp = calloc(g.cfg.samplers, sizeof *c->smp);
    if (!c->img || !c->smp) rt_fatal("gfx: vulkan: bindless tables: out of memory for a copy");
    c->set = g.ops.set_make();
    c->fresh = 1;
    if (g.ncopies == g.capcopies) {
        g.capcopies = g.capcopies ? g.capcopies * 2 : 4;
        g.copies = realloc(g.copies, (size_t)g.capcopies * sizeof *g.copies);
        if (!g.copies) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    }
    g.copies[g.ncopies++] = c;
    g.st.copies++;
    return c;
}

static void copy_drop(Copy *c) {
    for (uint32_t i = 0; i < g.cfg.images; i++) unref(c->img[i], c->last_value);
    for (uint32_t i = 0; i < g.cfg.samplers; i++) unref(c->smp[i], c->last_value);
    g.ops.set_free(c->set);
    free(c->img), free(c->smp), free(c);
}

// ---------- open and close

void bindless_open(const BindlessOps *ops, const BindlessConfig *cfg, const BindlessClock *clock) {
    plat_mutex_lock(&g_mu);
    if (g.open) rt_fatal("gfx: vulkan: bindless tables opened twice");
    if (cfg->second >= cfg->images || !cfg->max_copies || !cfg->max_samplers)
        rt_fatal("gfx: vulkan: bindless tables: a configuration of %u images, %u of them second slots, %u copies",
                 cfg->images, cfg->second, cfg->max_copies);
    memset(&g, 0, sizeof g);
    g.ops = *ops, g.cfg = *cfg, g.clock = *clock;
    g.primary = cfg->images - cfg->second;
    g.log = calloc(LOG_CAP, sizeof *g.log);
    g.hash = calloc(HASH_BUCKETS, sizeof *g.hash);
    g.sec = calloc(cfg->second ? cfg->second : 1, sizeof *g.sec);
    g.sec_free = calloc(cfg->second ? cfg->second : 1, sizeof *g.sec_free);
    if (!g.log || !g.hash || !g.sec || !g.sec_free) rt_fatal("gfx: vulkan: bindless tables: out of memory");
    for (uint32_t i = 0; i < cfg->second; i++) g.sec_free[g.nsec_free++] = cfg->second - 1 - i;
    g.open = 1;
    plat_mutex_unlock(&g_mu);
}

void bindless_close(void) {
    plat_mutex_lock(&g_mu);
    if (!g.open) {
        plat_mutex_unlock(&g_mu);
        return;
    }
    for (int i = 0; i < g.ncopies; i++) copy_drop(g.copies[i]);
    for (int i = 0; i < g.ntp; i++) {
        TexPool *p = g.tp[i];
        for (uint32_t c = 0; c < p->nchunks; c++) {
            if (!p->chunk[c]) continue;
            for (uint32_t k = 0; k < CHUNK; k++) unref(p->chunk[c][k].res, 0);
            free(p->chunk[c]);
        }
        free(p->chunk), free(p);
    }
    for (int i = 0; i < g.nsp; i++) {
        for (uint32_t k = 0; k < g.cfg.samplers; k++) unref(g.sp[i]->ent[k], 0);
        free(g.sp[i]->ent), free(g.sp[i]);
    }
    for (uint32_t i = 0; i < g.cfg.second; i++) unref(g.sec[i].res, 0);
    while (g.retire) {
        Res *r = g.retire;
        g.retire = r->next;
        if (r->sampler) g.ops.sampler_free(r->obj);
        else g.ops.view_free(r->obj);
        free(r);
    }
    free(g.tp), free(g.sp), free(g.copies), free(g.log), free(g.hash), free(g.sec), free(g.sec_free);
    memset(&g, 0, sizeof g);
    plat_mutex_unlock(&g_mu);
}

int bindless_is_open(void) { return g.open; }

// ---------- registrations

void bindless_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    plat_mutex_lock(&g_mu);
    if (id < 0 || (uint32_t)id >= g.primary)
        rt_fatal("gfx: vulkan: bindless tables: a texture registered at ID %d of pool %#llx, and the table holds the "
                 "game's IDs below %u (the last %u slots are for second views)", id, (ull)pool->addr, g.primary,
                 g.cfg.second);
    int type = bindless_view_type(t, v);
    if (type == BV_NONE)
        rt_fatal("gfx: vulkan: bindless tables: a buffer texture is registered at ID %d of pool %#llx (target %d): it "
                 "needs a texel buffer view, which the table does not hold", id, (ull)pool->addr, t->d.target);
    TexPool *p = tpool(pool);
    Entry *e = entry_at(p, (uint32_t)id, 1);
    Res *r = view_get(t, v, type, 0);
    g.st.registrations++;
    int same = e->used && e->res == r && e->type == type;
    e->t = *t;
    if (v) e->v = *v;
    else memset(&e->v, 0, sizeof e->v);
    if (same) {
        g.st.unchanged++;
        plat_mutex_unlock(&g_mu);
        return;
    }
    second_free_all(e);
    if (r) ref(r);
    unref(e->res, 0);
    e->res = r, e->type = type, e->used = 1;
    log_add(R_TEX, p, (uint32_t)id);
    plat_mutex_unlock(&g_mu);
}

void bindless_sampler_register(const dpool_ref *pool, int id, const samp_par *s) {
    plat_mutex_lock(&g_mu);
    if (id < 0 || (uint32_t)id >= g.cfg.samplers)
        rt_fatal("gfx: vulkan: bindless tables: a sampler registered at ID %d of pool %#llx, and the array holds %u",
                 id, (ull)pool->addr, g.cfg.samplers);
    SmpPool *p = spool(pool);
    Res *r = sampler_get(s);
    g.st.registrations++;
    if (p->ent[id] == r) {
        g.st.unchanged++;
        plat_mutex_unlock(&g_mu);
        return;
    }
    ref(r);
    unref(p->ent[id], 0);
    p->ent[id] = r;
    log_add(R_SMP, p, (uint32_t)id);
    plat_mutex_unlock(&g_mu);
}

void bindless_texture_release(const tex_t *t) {
    if (g.ops.texture_release) g.ops.texture_release(t);
}

void bindless_storage_gone(uint64_t lo, uint64_t hi) {
    plat_mutex_lock(&g_mu);
    // A pool remade at the same address holds textures at the same storage with the same shape, whose registrations would
    // find the views of the finalized pool's textures by their key while a copy still shows them, and stand on images whose
    // memory went back to the backing map with the pool. Those views are no longer shared.
    hash_detach_storage(lo, hi);
    for (int i = 0; i < g.ntp; i++) {
        TexPool *p = g.tp[i];
        for (uint32_t c = 0; c < p->nchunks; c++) {
            if (!p->chunk[c]) continue;
            for (uint32_t k = 0; k < CHUNK; k++) {
                Entry *e = &p->chunk[c][k];
                uint64_t at = (uint64_t)(uintptr_t)e->t.mem;
                if (!e->used || at < lo || at >= hi) continue;
                second_free_all(e);
                unref(e->res, 0);
                e->res = NULL, e->used = 0;
                log_add(R_TEX, p, c * CHUNK + k);
            }
        }
    }
    plat_mutex_unlock(&g_mu);
}

// ---------- the replay's side

// A copy nothing pending reads, the one idle the longest first, or a new one; NULL with *wait set to the value to wait
// for when every copy is pending and the ring is full. mu held.
static Copy *free_copy(uint64_t done, uint64_t *wait) {
    Copy *best = NULL;
    for (int i = 0; i < g.ncopies; i++) {
        Copy *c = g.copies[i];
        if (c->last_value <= done && (!best || c->last_value < best->last_value)) best = c;
    }
    if (best) return best;
    if ((uint32_t)g.ncopies < g.cfg.max_copies) return copy_new();
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < g.ncopies; i++)
        if (g.copies[i]->last_value < oldest) oldest = g.copies[i]->last_value;
    *wait = oldest;
    return NULL;
}

void *bindless_acquire(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers) {
    plat_mutex_lock(&g_mu);
    reap();
    TexPool *tp = textures ? tpool(textures) : g.cur ? g.cur->tp : NULL;
    SmpPool *sp = samplers ? spool(samplers) : g.cur ? g.cur->sp : NULL;
    Copy *c = g.cur;
    for (;;) {
        uint64_t done = g.clock.done();
        int behind = !c || c->fresh || c->pos != g.head || c->tp != tp || c->sp != sp;
        if (c && (c->last_value <= done || !behind)) break;
        uint64_t wait = 0;
        Copy *n = free_copy(done, &wait);
        if (n) {
            if (c) g.st.copy_switches++;
            c = n;
            break;
        }
        g.st.copy_waits++;
        plat_mutex_unlock(&g_mu);
        g.clock.wait(wait);
        plat_mutex_lock(&g_mu);
        c = g.cur;
    }
    sync_copy(c, tp, sp);
    c->last_value = value;
    g.cur = c;
    g.st.acquires++;
    // One other idle copy that is behind is brought up to date, so that a copy no submission takes for a while does not
    // keep what the entries have since dropped alive.
    if (g.ncopies > 1) {
        Copy *t = g.copies[(uint32_t)g.rr++ % (uint32_t)g.ncopies];
        if (t != c && t->last_value <= g.clock.done() && !t->fresh && t->tp == tp && t->sp == sp &&
            t->pos != g.head && g.head - t->pos <= LOG_CAP) {
            sync_copy(t, tp, sp);
            g.st.tidy_syncs++;
        }
    }
    void *set = c->set;
    plat_mutex_unlock(&g_mu);
    return set;
}

int bindless_texture_entry(uint32_t id, tex_t *t, view_t *v) {
    plat_mutex_lock(&g_mu);
    Copy *c = g.cur;
    Entry *e = c && c->tp && id < g.primary ? entry_at(c->tp, id, 0) : NULL;
    int ok = e && e->used;
    if (ok) *t = e->t, *v = e->v;
    plat_mutex_unlock(&g_mu);
    return ok;
}

uint32_t bindless_image_slot(uint32_t id, int type) {
    plat_mutex_lock(&g_mu);
    Copy *c = g.cur;
    if (!c) rt_fatal("gfx: vulkan: bindless tables: an image slot is asked for before any submission took a copy");
    if (id >= g.primary)
        rt_fatal("gfx: vulkan: bindless tables: a texture handle of ID %u, and the table holds the game's IDs below %u",
                 id, g.primary);
    Entry *e = c->tp ? entry_at(c->tp, id, 0) : NULL;
    void *use = NULL;   // the view this call hands the replay, for view_use once the lock is released
    uint64_t value = c->last_value;
    if (e && e->used && !e->res && g.ops.view_demand) {
        // the host made no view at registration: the texture is bound now, so it makes one, and every copy gets it
        Res *r = view_get(&e->t, &e->v, e->type, 1);
        if (r) {
            ref(r);
            e->res = r;
            log_add(R_TEX, c->tp, id);
            Batch b = {.n = 0};
            put(c, &b, BW_IMAGE, id, r);
            flush(c, &b);
            use = r->obj;   // read now, even when the second view the caller asks for next cannot be made
        }
    }
    if (!e || !e->used || !e->res || type == BV_NONE || type == e->type) {
        if (e && e->used && e->res) use = e->res->obj;
        plat_mutex_unlock(&g_mu);
        if (use && g.ops.view_use) g.ops.view_use(use, value);
        return id;
    }
    uint32_t slot = UINT32_MAX;
    if (e->second[type]) {
        slot = g.primary + e->second[type] - 1;
    } else {
        Res *r = view_get(&e->t, &e->v, type, 1);
        if (r) {
            // A freed slot is taken again only when no copy shows its old view any more: a pending submission may have
            // resolved the slot to it, and a copy it reads is not written again for another view.
            uint32_t s = UINT32_MAX;
            for (uint32_t i = g.nsec_free; i-- > 0 && s == UINT32_MAX;) {
                uint32_t cand = g.sec_free[i];
                int clear = 1;
                for (int k = 0; k < g.ncopies && clear; k++) clear = g.copies[k]->img[g.primary + cand] == NULL;
                if (clear) {
                    s = cand;
                    g.sec_free[i] = g.sec_free[--g.nsec_free];
                }
            }
            if (s == UINT32_MAX)
                rt_fatal("gfx: vulkan: bindless tables: all %u second-view slots are in use or still shown by a copy a "
                         "pending submission reads, and texture ID %u of pool %#llx is read as view type %d",
                         g.cfg.second, id, (ull)c->tp->addr, type);
            ref(r);
            g.sec[s] = (Second){r, c->tp, id, type};
            e->second[type] = s + 1;
            log_add(R_SECOND, NULL, s);
            g.st.second_slots++, g.st.second_live++;
            slot = g.primary + s;
        }
    }
    if (slot != UINT32_MAX) {
        Batch b = {.n = 0};
        put(c, &b, BW_IMAGE, slot, g.sec[slot - g.primary].res);
        flush(c, &b);
        use = g.sec[slot - g.primary].res->obj;
    }
    plat_mutex_unlock(&g_mu);
    if (use && g.ops.view_use) g.ops.view_use(use, value);
    return slot;
}

uint32_t bindless_sampler_slot(uint32_t id) {
    if (id >= g.cfg.samplers)
        rt_fatal("gfx: vulkan: bindless tables: a sampler handle of ID %u, and the array holds %u", id, g.cfg.samplers);
    return id;
}

void bindless_stats(BindlessStats *out) {
    plat_mutex_lock(&g_mu);
    *out = g.st;
    out->log_head = g.head;
    plat_mutex_unlock(&g_mu);
}
