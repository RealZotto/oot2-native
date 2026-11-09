// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Spike S7's instruments and the owner side of lever L1 (oot2/texheap.h has the contract, the selection rule and the file
// format). One lock, g_lock, guards every table here. The fault path (cb_written, cb_touched) takes it on the faulting
// thread after the memory lock is dropped; this file takes it first and the memory lock after (rt_pages_arm and release,
// rt_mem_written), and every rt_pages_release call is made with g_lock held, so cb_busy, which runs under the memory
// lock inside it, reads the tables without taking g_lock.
//
// Page table. The heap and alias regions are cut into chunks of 64 Ki pages (256 MiB of address space), made when the first
// pool lands in one: per page a flag byte (written by the CPU since its pool began, selected), the number of live textures
// that touch it, the number of buffers and events that do, the texture record that owns it (a page wholly inside the
// storage) and the pool it lies in. A texture record is one storage, (guest address, bytes), however many times the game
// initializes a texture there.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nvn_api.h"
#include "oot2/memory.h"
#include "oot2/nvn_tex.h"
#include "plat.h"

typedef unsigned long long ull;

#define PAGE 0x1000ull
#define CHUNK_PAGES 65536u
#define CHUNK_SHIFT 28
#define MAX_CHUNKS 1024
#define REC_CHUNK 4096u
#define MAX_REC_CHUNKS 1024
#define POOL_CHUNK 1024u
#define MAX_POOL_CHUNKS 1024
#define FILE_FORMAT 1

enum { F_WRITTEN = 1, F_SEL = 2 };
enum { G_RT = 1, G_IMG = 2, G_COPY = 4, G_CLEAR = 8 };   // what the GPU does to a texture's storage

typedef struct Chunk {
    uint8_t fl[CHUNK_PAGES], tex[CHUNK_PAGES], oth[CHUNK_PAGES];
    uint32_t rec[CHUNK_PAGES], pool[CHUNK_PAGES];
} Chunk;

typedef struct Rec {
    uint64_t mem, size, first_serial, serial, mark, bound_at, first_read_seq, first_write_seq;
    uint64_t guest_reads, host_reads, guest_writes, host_writes;
    uint32_t pool, inits, binds, pages, cpu_pages, sel_pages, rewrites, recommits, b2t, t2b;
    uint32_t live;   // textures initialised at this storage now (more than one when two alias it)
    uint8_t pkg, gpu, bound, queued;
} Rec;

typedef struct PoolRec {
    uint64_t mem, size, agg_seq;
    uint64_t gen;   // the page generation up to which this pool's writes were read (oot2/pagegen.h)
    uint32_t flags, cls, live, region;
    uint32_t fr_pages, win_frames, win_pages, win_max;   // the writes of this frame, and of the window since the last sample
    uint64_t recs, rec_bytes, cpu_bytes, gpu_bytes, idle_bytes, tex_pages, written, selected, eligible;   // one sample's
} PoolRec;

typedef struct { uint64_t pool; uint32_t id, rec; } Desc;   // rec 0: a free slot
typedef struct { const pool_t *p; uint32_t idx; } PoolKey;
typedef struct Sub { uint64_t fence; uint32_t *recs; uint32_t n, cap; } Sub;

static PlatMutex g_lock = PLAT_MUTEX_INIT;
static int g_ready;
static FILE *g_f;
static char g_folder[512];
static uint64_t g_t0, g_seq, g_every = 300;
static struct { uint64_t base, size; } g_reg[2];            // heap, alias reservations
static uint64_t g_lo[2], g_hi[2];                           // the span of the pools' pages in each region
static Chunk *g_chunk[2][MAX_CHUNKS];
static Rec *g_recs[MAX_REC_CHUNKS];
static uint32_t g_nrec = 1;                                 // record 0 is none
static PoolRec *g_pools[MAX_POOL_CHUNKS];
static uint32_t g_npool = 1;
static uint32_t *g_rh;                                      // records by (mem, size), open addressing
static size_t g_rh_cap;
static PoolKey *g_pk;                                       // pool indexes by the game's pool object
static size_t g_pk_cap, g_pk_n;
static Desc *g_desc;
static size_t g_desc_cap, g_desc_n;
static Sub **g_subs;                                        // submissions in flight, oldest first
static size_t g_sub_head, g_sub_n, g_sub_cap;
static uint32_t *g_ready_recs;                              // bound since the last present, to select
static size_t g_ready_n, g_ready_cap;
static uint64_t g_tex_pool;                                 // the texture pool the replay has bound
static int g_range[2] = {-1, -1};
static uint64_t g_cur;   // the present being counted (its sequence number, after the previous present's)
static TexheapWatch g_watch;   // the reader of the write watch main.c installs (the page generations)

void texheap_set_watch(const TexheapWatch *w) { g_watch = *w; }

static struct {   // the run so far
    uint64_t b2t_copies, b2t_bytes, t2b_copies, pkg_inits, pkg_bytes, read_texels, eligible_total, partial, no_range,
        stale_touches, arms, releases, bind_ids, bind_miss, bind_dead, b2t_nomatch;
    uint64_t inval_calls, inval_tex_calls, inval_tex_bytes, flush_calls, flush_tex_calls, flush_tex_bytes;
} g_n;
static struct { uint64_t pages, tex_pages, pool_pages, outside, pools; } g_fr;   // the frame being counted
static int g_nowatch;
static uint32_t *g_frpools;                                  // pools written in this frame
static size_t g_frpools_n, g_frpools_cap;

static Rec *rec_at(uint32_t i) { return &g_recs[i / REC_CHUNK][i % REC_CHUNK]; }
static void *xrealloc(void *p, size_t n);

// The texture is bound as far as the instruments are concerned (its upload retired): select it at the next present.
static void ready_add(uint32_t ir) {
    Rec *r = &g_recs[ir / REC_CHUNK][ir % REC_CHUNK];
    r->bound = 1;
    r->bound_at = g_seq + 1;   // the present that closes the frame this happened in
    if (r->queued) return;
    r->queued = 1;
    if (g_ready_n == g_ready_cap) {
        g_ready_cap = g_ready_cap ? 2 * g_ready_cap : 1024;
        g_ready_recs = xrealloc(g_ready_recs, g_ready_cap * sizeof *g_ready_recs);
    }
    g_ready_recs[g_ready_n++] = ir;
}
static PoolRec *pool_at(uint32_t i) { return &g_pools[i / POOL_CHUNK][i % POOL_CHUNK]; }

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) rt_fatal("texheap: out of memory for %zu x %zu bytes", n, sz);
    return p;
}
static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n);
    if (!p) rt_fatal("texheap: out of memory for %zu bytes", n);
    return p;
}

// ---------- the file
static void row(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void row(const char *fmt, ...) {
    if (!g_f) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_f, fmt, ap);
    va_end(ap);
}
static uint64_t ms_now(void) { return (plat_now_ns() - g_t0) / 1000000ull; }

static void stop_writer(int code, const char *line) {   // no lock: the stop may come from a thread that holds g_lock
    FILE *f = g_f;
    if (!f) return;
    fprintf(f, "end\t%d\t%s\t%llu\n", code, line ? line : "", (ull)g_seq);
    fflush(f);
}

void texheap_set_folder(const char *dir) { snprintf(g_folder, sizeof g_folder, "%s", dir ? dir : ""); }

// ---------- the page table
static int where(uint64_t a, int *reg) {
    for (int r = 0; r < 2; r++)
        if (g_reg[r].size && a - g_reg[r].base < g_reg[r].size) return *reg = r, 1;
    return 0;
}

// The chunk holding address a and the page's index in it; NULL when no pool made one (create 0).
static Chunk *ck(uint64_t a, int create, uint32_t *i) {
    int r;
    if (!where(a, &r)) return NULL;
    uint64_t off = a - g_reg[r].base;
    Chunk **c = &g_chunk[r][off >> CHUNK_SHIFT];
    if (!*c && create) *c = xcalloc(1, sizeof **c);
    *i = (uint32_t)((off >> 12) & (CHUNK_PAGES - 1));
    return *c;
}

static uint64_t up(uint64_t a) { return (a + PAGE - 1) & ~(PAGE - 1); }
static uint64_t down(uint64_t a) { return a & ~(PAGE - 1); }

// ---------- records
static size_t key_slot(uint64_t mem, uint64_t size, size_t cap) {
    uint64_t h = (mem >> 4) * 0x9E3779B97F4A7C15ull ^ size * 0xC2B2AE3D27D4EB4Full;
    return (size_t)(h >> 29) & (cap - 1);
}

static uint32_t rec_find(uint64_t mem, uint64_t size) {
    if (!g_rh_cap) return 0;
    for (size_t j = key_slot(mem, size, g_rh_cap); g_rh[j]; j = (j + 1) & (g_rh_cap - 1)) {
        const Rec *r = rec_at(g_rh[j]);
        if (r->mem == mem && r->size == size) return g_rh[j];
    }
    return 0;
}

static void rh_insert(uint32_t idx) {
    if ((g_nrec + 1) * 2 > g_rh_cap) {
        size_t cap = g_rh_cap ? 2 * g_rh_cap : 1 << 14;
        free(g_rh);
        g_rh = xcalloc(cap, sizeof *g_rh);
        g_rh_cap = cap;
        for (uint32_t i = 1; i < idx; i++) {   // every record but the new one
            const Rec *r = rec_at(i);
            size_t j = key_slot(r->mem, r->size, cap);
            while (g_rh[j]) j = (j + 1) & (cap - 1);
            g_rh[j] = i;
        }
    }
    const Rec *r = rec_at(idx);
    size_t j = key_slot(r->mem, r->size, g_rh_cap);
    while (g_rh[j]) j = (j + 1) & (g_rh_cap - 1);
    g_rh[j] = idx;
}

static uint32_t rec_get(uint64_t mem, uint64_t size, int create) {
    uint32_t i = rec_find(mem, size);
    if (i || !create) return i;
    if (g_nrec / REC_CHUNK >= MAX_REC_CHUNKS) rt_fatal("texheap: more than %u texture storages", MAX_REC_CHUNKS * REC_CHUNK);
    if (!g_recs[g_nrec / REC_CHUNK]) g_recs[g_nrec / REC_CHUNK] = xcalloc(REC_CHUNK, sizeof(Rec));
    i = g_nrec++;
    Rec *r = rec_at(i);
    r->mem = mem, r->size = size;
    rh_insert(i);
    return i;
}

static uint32_t pool_new(const pool_t *p) {
    if (g_npool / POOL_CHUNK >= MAX_POOL_CHUNKS) rt_fatal("texheap: more than %u pools", MAX_POOL_CHUNKS * POOL_CHUNK);
    if (!g_pools[g_npool / POOL_CHUNK]) g_pools[g_npool / POOL_CHUNK] = xcalloc(POOL_CHUNK, sizeof(PoolRec));
    uint32_t i = g_npool++;
    PoolRec *q = pool_at(i);
    q->mem = (uintptr_t)p->mem, q->size = p->size, q->flags = p->flags, q->live = 1;
    q->cls = (p->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL)    ? NVN_POOL_VIRTUAL
             : (p->flags & NVN_MEMORY_POOL_FLAGS_PHYSICAL) ? NVN_POOL_PHYSICAL
             : (p->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ? NVN_POOL_CPU_NO_ACCESS
                                                                : NVN_POOL_CPU;
    return i;
}

static size_t pk_slot(const pool_t *p, size_t cap) { return (size_t)(((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull >> 29) & (cap - 1); }
static uint32_t pk_find(const pool_t *p) {
    if (!g_pk_cap) return 0;
    for (size_t j = pk_slot(p, g_pk_cap); g_pk[j].p; j = (j + 1) & (g_pk_cap - 1))
        if (g_pk[j].p == p) return g_pk[j].idx;
    return 0;
}
static void pk_set(const pool_t *p, uint32_t idx) {
    if ((g_pk_n + 1) * 2 > g_pk_cap) {
        size_t cap = g_pk_cap ? 2 * g_pk_cap : 4096;
        PoolKey *n = xcalloc(cap, sizeof *n);
        for (size_t i = 0; i < g_pk_cap; i++) {
            if (!g_pk[i].p) continue;
            size_t j = pk_slot(g_pk[i].p, cap);
            while (n[j].p) j = (j + 1) & (cap - 1);
            n[j] = g_pk[i];
        }
        free(g_pk);
        g_pk = n, g_pk_cap = cap;
    }
    size_t j = pk_slot(p, g_pk_cap);
    while (g_pk[j].p && g_pk[j].p != p) j = (j + 1) & (g_pk_cap - 1);
    if (!g_pk[j].p) g_pk_n++;
    g_pk[j] = (PoolKey){p, idx};
}

// ---------- start-up
static void ensure(void) {
    if (g_ready) return;
    g_ready = 1;
    RtMemStats ms;
    rt_mem_measure(&ms);
    g_reg[0].base = ms.r[RT_MEM_HEAP].base, g_reg[0].size = ms.r[RT_MEM_HEAP].size;
    g_reg[1].base = ms.r[RT_MEM_ALIAS].base, g_reg[1].size = ms.r[RT_MEM_ALIAS].size;
    if ((g_reg[0].size >> CHUNK_SHIFT) >= MAX_CHUNKS || (g_reg[1].size >> CHUNK_SHIFT) >= MAX_CHUNKS)
        rt_fatal("texheap: a region is larger than the page table's %u chunks", MAX_CHUNKS);
    g_every = (uint64_t)rt_setting(RT_SET_texheap_every);
    g_t0 = plat_now_ns();
    time_t wall = time(NULL);
    struct tm tm;
#ifdef _WIN32
    gmtime_s(&tm, &wall);
#else
    gmtime_r(&wall, &tm);
#endif
    char stamp[32], path[640];
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    snprintf(path, sizeof path, "%s%s%s-%u.tsv", g_folder, g_folder[0] ? "/texheap-" : "texheap-", stamp, plat_pid());
    g_f = fopen(path, "wb");
    if (!g_f) {
        rt_log("texheap: cannot open %s: the instruments count and write no file", path);
    } else {
        rt_add_stop_writer(stop_writer);
        fprintf(g_f, "oot2-texheap %d\n", FILE_FORMAT);
        const char *level = rt_pages_level() == RT_PAGES_RELEASING ? "release"
                            : rt_pages_level() == RT_PAGES_COUNTING ? "count" : "off";
        fprintf(g_f, "header\tpage_release\t%s\nheader\tevery\t%llu\nheader\tpid\t%u\n", level, (ull)g_every, plat_pid());
        fprintf(g_f, "columns\tframe\tseq\tpages\ttex_pages\tpool_pages\toutside\tpools_written\n");
        fprintf(g_f, "columns\tsample\tseq\tt_ms\theap_mapped\theap_committed\theap_resident\tprivate\tworking_set\tpools\t"
                     "pool_bytes\trecs\trec_bytes\tcpu_recs\tcpu_bytes\tgpu_recs\tgpu_bytes\tidle_recs\tidle_bytes\t"
                     "tex_pages\twritten_pages\teligible_pages\tselected_pages\tarms\treleases\trecommits\tbusy\t"
                     "guest_reads\thost_reads\tguest_writes\thost_writes\tstale_touches\tb2t_copies\tb2t_bytes\t"
                     "t2b_copies\tpkg_inits\tpkg_bytes\tread_texels\tinval_calls\tinval_tex_calls\tinval_tex_bytes\t"
                     "flush_calls\tflush_tex_calls\tflush_tex_bytes\tpartial\tno_range\twhy_unbound\twhy_buffer\t"
                     "why_shared\twhy_owner\tbind_ids\tbind_miss\tbind_dead\tb2t_nomatch\n");
        fprintf(g_f, "columns\tpool\tseq\tid\tclass\tflags\tsize\tmem\tcommitted\tresident\trecs\trec_bytes\tcpu_bytes\t"
                     "gpu_bytes\tidle_bytes\ttex_pages\twritten_pages\tselected_pages\teligible_pages\twindow_frames\t"
                     "window_pages\twindow_max\n");
        fprintf(g_f, "columns\twpool\tseq\tid\tclass\tsize\twindow_frames\twindow_pages\twindow_max\n");
        fprintf(g_f, "columns\tread\tseq\tserial\tsize\tpool\tpages\tguest\tfn\n");
        fflush(g_f);
    }
    rt_log("texheap: the texture heap instruments are on (page_release %s, a sample every %llu presents)%s",
           rt_pages_level() == RT_PAGES_RELEASING ? "release" : rt_pages_level() == RT_PAGES_COUNTING ? "count" : "off",
           (ull)g_every, g_f ? "" : ", without a file");
}

// ---------- pools
// A query over [lo, hi) through the page generations, whose answer this file drops: the pages it stamps are older than the
// generation it returns, which a pool takes as its start, so writes made before the pool began are not the pool's. 0 when
// there is no write watch.
static uint64_t start_gen(uint64_t lo, uint64_t hi) {
    if (g_nowatch) return 0;
    if (!g_watch.scan) return g_nowatch = 1, 0;
    uint64_t g = g_watch.scan(lo, hi);
    if (g == UINT64_MAX) return g_nowatch = 1, 0;
    return g;
}

void texheap_pool_init(const pool_t *p) {
    plat_mutex_lock(&g_lock);
    ensure();
    uint32_t idx = pool_new(p);
    pk_set(p, idx);
    uint64_t a = (uintptr_t)p->mem;
    int r, r2;
    if (!gfx_gpu_only(a) && where(a, &r) && where(a + p->size - 1, &r2) && r == r2) {
        pool_at(idx)->region = (uint32_t)r + 1;
        if (!g_hi[r] || a < g_lo[r]) g_lo[r] = down(a);
        if (a + p->size > g_hi[r]) g_hi[r] = up(a + p->size);
        pool_at(idx)->gen = start_gen(down(a), up(a + p->size));
        for (uint64_t x = down(a); x < up(a + p->size); x += PAGE) {   // a new tenancy: no texture, buffer or write of the last
            uint32_t i;
            Chunk *k = ck(x, 1, &i);
            k->fl[i] &= F_SEL;   // a page still armed or released in the memory layer stays so
            k->tex[i] = k->oth[i] = 0;
            k->rec[i] = 0;
            k->pool[i] = idx;
        }
    }
    plat_mutex_unlock(&g_lock);
}

void texheap_pool_end(const pool_t *p) {
    plat_mutex_lock(&g_lock);
    ensure();
    uint32_t idx = pk_find(p);
    if (idx && pool_at(idx)->live) {
        PoolRec *q = pool_at(idx);
        q->live = 0;
        if (q->region) {
            for (uint64_t x = down(q->mem); x < up(q->mem + q->size); x += PAGE) {
                uint32_t i;
                Chunk *k = ck(x, 0, &i);
                if (k && k->pool[i] == idx) k->pool[i] = 0, k->tex[i] = k->oth[i] = 0, k->rec[i] = 0;
            }
        }
    }
    plat_mutex_unlock(&g_lock);
}

// ---------- textures
static void cover(const Rec *r, int delta) {   // pages the storage touches
    uint64_t a = (uint64_t)r->mem, end = r->mem + r->size;
    for (uint64_t x = down(a); x < up(end); x += PAGE) {
        uint32_t i;
        Chunk *k = ck(x, 0, &i);
        if (!k) continue;
        if (delta > 0 && k->tex[i] < 255) k->tex[i]++;
        else if (delta < 0 && k->tex[i] && k->tex[i] < 255) k->tex[i]--;
    }
}

void texheap_texture_init(const tex_t *t, int pkg) {
    plat_mutex_lock(&g_lock);
    ensure();
    uint32_t idx = rec_get((uintptr_t)t->mem, t->L.size, 1);
    Rec *r = rec_at(idx);
    r->pool = pk_find(t->pool);
    r->live++;
    if (!gfx_gpu_only((uintptr_t)t->mem)) {
        cover(r, 1);
        uint64_t lo = up(r->mem), hi = down(r->mem + r->size);
        r->pages = hi > lo ? (uint32_t)((hi - lo) / PAGE) : 0;
        for (uint64_t x = lo; x < hi; x += PAGE) {
            uint32_t i;
            Chunk *k = ck(x, 0, &i);
            if (k) k->rec[i] = idx;
        }
    }
    r->inits++;
    r->serial = t->serial;
    if (!r->first_serial) r->first_serial = t->serial;
    if (pkg) {
        r->pkg = 1;
        g_n.pkg_inits++, g_n.pkg_bytes += r->size;
    }
    r->bound = 0;
    plat_mutex_unlock(&g_lock);
}

void texheap_texture_end(const tex_t *t) {
    plat_mutex_lock(&g_lock);
    ensure();
    uint32_t idx = rec_find((uintptr_t)t->mem, t->L.size);
    if (idx) {
        Rec *r = rec_at(idx);
        if (r->live) {
            r->live--;
            if (!gfx_gpu_only((uintptr_t)t->mem)) cover(r, -1);
        }
    }
    plat_mutex_unlock(&g_lock);
}

void texheap_read_texels(void) { __atomic_fetch_add(&g_n.read_texels, 1, __ATOMIC_RELAXED); }

// Buffers and events: a page one touches is never a texture page's alone. Flushes and invalidations: a call that covers
// texture storage is the game saying the CPU wrote or will read there.
void texheap_range(int what, uint64_t addr, uint64_t size) {
    if (!size) return;
    plat_mutex_lock(&g_lock);
    ensure();
    uint64_t first = down(addr), last = up(addr + size);
    if (what == TEXHEAP_OBJECT || what == TEXHEAP_OBJECT_END) {
        for (uint64_t x = first; x < last; x += PAGE) {
            uint32_t i;
            Chunk *k = ck(x, 0, &i);
            if (!k) continue;
            if (what == TEXHEAP_OBJECT) {
                if (k->oth[i] < 255) k->oth[i]++;
            } else if (k->oth[i] && k->oth[i] < 255) {
                k->oth[i]--;
            }
        }
    } else {
        uint64_t pages = 0, cap = 1u << 18;
        for (uint64_t x = first; x < last && x - first < cap * PAGE; x += PAGE) {
            uint32_t i;
            Chunk *k = ck(x, 0, &i);
            pages += k && k->tex[i] != 0;
        }
        if (what == TEXHEAP_INVALIDATE) {
            g_n.inval_calls++;
            g_n.inval_tex_calls += pages != 0;
            g_n.inval_tex_bytes += pages * PAGE;
        } else {
            g_n.flush_calls++;
            g_n.flush_tex_calls += pages != 0;
            g_n.flush_tex_bytes += pages * PAGE;
        }
    }
    plat_mutex_unlock(&g_lock);
}

// ---------- descriptors: (texture pool table, ID) to the record a bind names
static size_t desc_slot(uint64_t pool, uint32_t id, size_t cap) {
    uint64_t h = (pool >> 5) * 0x9E3779B97F4A7C15ull ^ (uint64_t)id * 0xC2B2AE3D27D4EB4Full;
    return (size_t)(h >> 29) & (cap - 1);
}

void texheap_desc(const dpool_ref *pool, int id, const tex_t *t) {
    plat_mutex_lock(&g_lock);
    ensure();
    uint32_t idx = rec_get((uintptr_t)t->mem, t->L.size, 1);
    if ((g_desc_n + 1) * 2 > g_desc_cap) {
        size_t cap = g_desc_cap ? 2 * g_desc_cap : 4096;
        Desc *n = xcalloc(cap, sizeof *n);
        for (size_t i = 0; i < g_desc_cap; i++) {
            if (!g_desc[i].rec) continue;
            size_t j = desc_slot(g_desc[i].pool, g_desc[i].id, cap);
            while (n[j].rec) j = (j + 1) & (cap - 1);
            n[j] = g_desc[i];
        }
        free(g_desc);
        g_desc = n, g_desc_cap = cap;
    }
    size_t j = desc_slot(pool->addr, (uint32_t)id, g_desc_cap);
    while (g_desc[j].rec && !(g_desc[j].pool == pool->addr && g_desc[j].id == (uint32_t)id)) j = (j + 1) & (g_desc_cap - 1);
    if (!g_desc[j].rec) g_desc_n++;
    g_desc[j] = (Desc){pool->addr, (uint32_t)id, idx};
    if (rt_setting(RT_SET_texheap_trigger) == 1) {   // register: as a back end that uploads at registration would
        Rec *r = rec_at(idx);
        r->binds++;
        if (r->live && !r->bound) ready_add(idx);
    }
    plat_mutex_unlock(&g_lock);
}

static uint32_t desc_find(uint64_t pool, uint32_t id) {
    if (!g_desc_cap) return 0;
    for (size_t j = desc_slot(pool, id, g_desc_cap); g_desc[j].rec; j = (j + 1) & (g_desc_cap - 1))
        if (g_desc[j].pool == pool && g_desc[j].id == id) return g_desc[j].rec;
    return 0;
}

// ---------- the write watch
// One page the CPU wrote since the last reset: the frame's counters, the page's mark, and a texture rewritten.
static void note_written(uint64_t page) {
    uint32_t i;
    Chunk *k = ck(page, 0, &i);
    uint32_t ip = k ? k->pool[i] : 0;
    if (!ip) return;
    g_fr.pool_pages++;
    PoolRec *q = pool_at(ip);
    if (!q->fr_pages) {
        if (g_frpools_n == g_frpools_cap) {
            g_frpools_cap = g_frpools_cap ? 2 * g_frpools_cap : 1024;
            g_frpools = xrealloc(g_frpools, g_frpools_cap * sizeof *g_frpools);
        }
        g_frpools[g_frpools_n++] = ip;
        g_fr.pools++;
    }
    q->fr_pages++;
    if (k->tex[i]) g_fr.tex_pages++;
    int was = k->fl[i] & F_WRITTEN;
    k->fl[i] |= F_WRITTEN;
    if (k->rec[i]) {
        Rec *r = rec_at(k->rec[i]);
        if (!was) r->cpu_pages++;
        // A write seen in the frame whose present follows the bind's completion is taken to come before the bind (the
        // upload is written, then used); a later one is a rewrite. At the frame's granularity a write after the bind in
        // the same frame is not told from one before it.
        if (r->bound && r->live && r->bound_at < g_cur) r->bound = 0, r->rewrites++;
    }
}

static void runs_cb(void *ctx, uint64_t lo, uint64_t hi) {   // under the page generations' lock: calls nothing of theirs
    (void)ctx;
    for (uint64_t a = lo; a < hi; a += PAGE) note_written(a);
}

// The frame's reading of the write watch: one query per region over the span of the pools' pages, stamped in the page
// generations (oot2/pagegen.h: the runtime's one caller of the write watch, so another consumer of the same pages sees
// them too), then, for each pool, the pages stamped since the generation up to which the pool was last read.
static void read_writes(void) {
    uint64_t gen[2] = {0, 0}, before = g_watch.scanned ? g_watch.scanned() : 0;
    if (!g_watch.scan || !g_watch.runs) g_nowatch = 1;   // no reader installed: nothing is ever selected
    for (int r = 0; r < 2 && !g_nowatch; r++) {
        if (g_hi[r] <= g_lo[r]) continue;
        gen[r] = g_watch.scan(g_lo[r], g_hi[r]);
        if (gen[r] == UINT64_MAX) g_nowatch = 1;   // no write watch here (not Windows): nothing is ever selected
    }
    if (g_nowatch) return;
    for (uint32_t p = 1; p < g_npool; p++) {
        PoolRec *q = pool_at(p);
        if (!q->live || !q->region) continue;
        if (gen[q->region - 1]) {
            g_watch.runs(down(q->mem), up(q->mem + q->size), q->gen, runs_cb, NULL);
            q->gen = gen[q->region - 1];
        }
    }
    uint64_t scanned = (g_watch.scanned ? g_watch.scanned() : 0) - before;   // approximate: another consumer's query too
    g_fr.pages = scanned > g_fr.pool_pages ? scanned : g_fr.pool_pages;
    g_fr.outside = g_fr.pages - g_fr.pool_pages;
}

// ---------- selection
static int cb_busy(void *u, uint64_t addr, uint64_t size) {   // under the memory lock, inside rt_pages_release
    (void)u;
    for (uint64_t x = addr; x < addr + size; x += PAGE) {
        uint32_t i;
        Chunk *k = ck(x, 0, &i);
        if (!k || k->tex[i] != 1 || k->oth[i]) return 1;
    }
    return 0;
}

static void touch_row(const char *kind, const Rec *r, int guest, uint64_t fn) {
    char name[96] = "";
    if (guest) rt_guest_name(fn, name, sizeof name);
    row("%s\t%llu\t%llu\t%llu\t%u\t%u\t%d\t%s\n", kind, (ull)g_seq, (ull)r->first_serial, (ull)r->size, r->pool, r->pages,
        guest, name);
    if (g_f) fflush(g_f);
}

static void cb_touched(void *u, uint64_t page, int write, int guest, uint64_t fn) {
    (void)u;
    plat_mutex_lock(&g_lock);
    uint32_t i;
    Chunk *k = ck(page, 0, &i);
    if (!k) {
        plat_mutex_unlock(&g_lock);
        return;
    }
    k->fl[i] &= (uint8_t)~F_SEL;
    uint32_t ir = k->rec[i];
    if (!ir || !rec_at(ir)->live) {
        g_n.stale_touches++;   // memory a finalized texture left, or a page whose texture is gone, used by something else
        if (g_n.stale_touches <= 16) {
            char name[96] = "";
            if (guest) rt_guest_name(fn, name, sizeof name);
            rt_log("texheap: %s of the selected page %#llx by %s %s, which no live texture owns", write ? "write" : "read",
                   (ull)page, guest ? "guest code in" : "the runtime", guest ? name : "or the back end");
        }
        plat_mutex_unlock(&g_lock);
        return;
    }
    Rec *r = rec_at(ir);
    if (r->sel_pages) r->sel_pages--;
    uint64_t *c = write ? (guest ? &r->guest_writes : &r->host_writes) : (guest ? &r->guest_reads : &r->host_reads);
    (*c)++;
    uint64_t *first = write ? &r->first_write_seq : &r->first_read_seq;
    if (!*first) {
        *first = g_seq ? g_seq : 1;
        touch_row(write ? "write" : "read", r, guest, fn);
        if (!write) {
            char name[96] = "";
            if (guest) rt_guest_name(fn, name, sizeof name);
            rt_log("texheap: %s read a selected page of the texture of serial %llu (%llu bytes, pool %u) at present %llu%s%s",
                   guest ? "guest code" : "the runtime or the back end", (ull)r->first_serial, (ull)r->size, r->pool,
                   (ull)g_seq, guest ? " in " : "", name);
        }
    }
    plat_mutex_unlock(&g_lock);
}

static void cb_written(void *u, uint64_t page) {   // a write put a released page back
    (void)u;
    plat_mutex_lock(&g_lock);
    uint32_t i;
    Chunk *k = ck(page, 0, &i);
    if (k) {
        k->fl[i] &= (uint8_t)~F_SEL;
        if (k->rec[i]) {
            Rec *r = rec_at(k->rec[i]);
            r->recommits++;
            if (r->bound && r->live) r->bound = 0, r->rewrites++;
        }
    }
    plat_mutex_unlock(&g_lock);
}

static int range_for(int region) {   // (region 0 heap, 1 alias)
    if (g_range[region] >= 0) return g_range[region];
    static const RtPageOwner owner = {"texheap", NULL, cb_busy, cb_written, cb_touched};
    int mode = rt_pages_level() == RT_PAGES_RELEASING ? RT_PAGES_RELEASING : RT_PAGES_COUNTING;
    if (rt_pages_level() == RT_PAGES_OFF) return -1;
    g_range[region] = rt_pages_watch(g_reg[region].base, g_reg[region].size, mode, &owner);
    return g_range[region];
}

static int eligible_page(const Chunk *k, uint32_t i, uint32_t ir) {
    return (k->fl[i] & F_WRITTEN) && !(k->fl[i] & F_SEL) && k->tex[i] == 1 && !k->oth[i] && k->rec[i] == ir;
}

static void select_run(Rec *r, uint32_t ir, int region, uint64_t s, uint64_t e) {
    uint64_t n = (e - s) / PAGE;
    g_n.eligible_total += n;
    int id = range_for(region);
    if (id < 0) {
        if (rt_pages_level() != RT_PAGES_OFF) g_n.no_range++;
        return;
    }
    uint64_t got = rt_pages_level() == RT_PAGES_RELEASING ? rt_pages_release(id, s, e - s) : rt_pages_arm(id, s, e - s);
    if (got != n) {
        g_n.partial++;   // not every page was committed or free of another state: none is marked, the counters say so
        return;
    }
    (rt_pages_level() == RT_PAGES_RELEASING ? &g_n.releases : &g_n.arms)[0] += n;
    r->sel_pages += (uint32_t)n;
    for (uint64_t x = s; x < e; x += PAGE) {
        uint32_t i;
        Chunk *k = ck(x, 0, &i);
        if (k) k->fl[i] |= F_SEL;
    }
    (void)ir;
}

static void select_rec(uint32_t ir) {
    Rec *r = rec_at(ir);
    if (!r->live || !r->bound || gfx_gpu_only((uintptr_t)r->mem) || g_nowatch) return;
    uint64_t lo = up(r->mem), hi = down(r->mem + r->size), start = 0;
    int region = 0;
    for (uint64_t x = lo; x < hi; x += PAGE) {
        uint32_t i;
        Chunk *k = ck(x, 0, &i);
        int ok = k && eligible_page(k, i, ir);
        if (ok && !start) {
            start = x;
            where(x, &region);
        }
        if (!ok && start) {
            select_run(r, ir, region, start, x);
            start = 0;
        }
    }
    if (start) select_run(r, ir, region, start, hi);
}

// ---------- the replay's side
void texheap_submit(uint64_t fence) {
    plat_mutex_lock(&g_lock);
    ensure();
    if (g_sub_head + g_sub_n == g_sub_cap) {
        if (g_sub_head) {
            memmove(g_subs, g_subs + g_sub_head, g_sub_n * sizeof *g_subs);
            g_sub_head = 0;
        } else {
            g_sub_cap = g_sub_cap ? 2 * g_sub_cap : 64;
            g_subs = xrealloc(g_subs, g_sub_cap * sizeof *g_subs);
        }
    }
    Sub *s = xcalloc(1, sizeof *s);
    s->fence = fence;
    g_subs[g_sub_head + g_sub_n++] = s;
    plat_mutex_unlock(&g_lock);
}

static uint64_t region_bytes(const nvn_tex_rec *t, const NVNcopyRegion *r) {
    int bw = 1, bh = 1, bpb = nvn_fmt_bpb(t->tex.d.format, &bw, &bh);
    if (bpb <= 0 || r->width <= 0) return 0;
    uint64_t w = ((uint64_t)r->width + (uint64_t)bw - 1) / (uint64_t)bw;
    uint64_t h = ((uint64_t)(r->height > 0 ? r->height : 1) + (uint64_t)bh - 1) / (uint64_t)bh;
    return w * h * (uint64_t)(r->depth > 0 ? r->depth : 1) * (uint64_t)bpb;
}

static uint32_t rec_of_tex(const nvn_tex_rec *t) { return t->present ? rec_find((uintptr_t)t->tex.mem, t->tex.L.size) : 0; }

void texheap_op(uint64_t fence, const op_hdr *h) {
    switch (h->op) {
    case OP_SET_TEX_POOL: case OP_BIND_TEX: case OP_BIND_IMG: case OP_SET_RT: case OP_COPY_B2T: case OP_COPY_T2B:
    case OP_COPY_T2T: case OP_CLEAR_TEX: break;
    default: return;
    }
    const void *a = h + 1;
    plat_mutex_lock(&g_lock);
    switch (h->op) {
    case OP_SET_TEX_POOL: g_tex_pool = ((const p_set_pool *)a)->pool.addr; break;
    case OP_BIND_TEX: {
        const p_bind_handles *b = a;
        Sub *s = NULL;   // the record's own submission: a held queue's records are replayed after later submissions' (gfx.h)
        for (size_t k = g_sub_n; k-- > 0;)
            if (g_subs[g_sub_head + k]->fence == fence) {
                s = g_subs[g_sub_head + k];
                break;
            }
        for (int i = 0; s && i < b->n && i < 32; i++) {
            uint32_t ir = desc_find(g_tex_pool, nvn_handle_texture_id(b->h[i]));
            g_n.bind_ids++;
            if (!ir) {
                g_n.bind_miss++;   // no texture was registered at that ID of the bound texture pool
                continue;
            }
            Rec *r = rec_at(ir);
            if (!r->live) {
                g_n.bind_dead++;   // registered, but its texture is finalized
                continue;
            }
            if (r->mark == s->fence) continue;
            r->mark = s->fence;
            if (s->n == s->cap) {
                s->cap = s->cap ? 2 * s->cap : 256;
                s->recs = xrealloc(s->recs, s->cap * sizeof *s->recs);
            }
            s->recs[s->n++] = ir;
        }
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = a;
        for (int i = 0; i < b->n && i < 32; i++) {
            uint32_t ir = desc_find(g_tex_pool, nvn_handle_texture_id(b->h[i]));
            if (ir) rec_at(ir)->gpu |= G_IMG;
        }
        break;
    }
    case OP_SET_RT: {
        const p_set_rt *p = a;
        uint32_t ir = rec_of_tex(&p->depth);
        if (ir) rec_at(ir)->gpu |= G_RT;
        for (int i = 0; i < p->n && i < 8; i++)
            if ((ir = rec_of_tex(&p->color[i]))) rec_at(ir)->gpu |= G_RT;
        break;
    }
    case OP_COPY_B2T: {
        const p_copy_b2t *p = a;
        uint32_t ir = rec_of_tex(&p->tex);
        g_n.b2t_copies++, g_n.b2t_bytes += region_bytes(&p->tex, &p->r);
        if (ir) rec_at(ir)->gpu |= G_COPY, rec_at(ir)->b2t++;
        else g_n.b2t_nomatch++;   // a copy whose destination has no record (not present, or initialised before the run)
        break;
    }
    case OP_COPY_T2B: {
        const p_copy_b2t *p = a;
        uint32_t ir = rec_of_tex(&p->tex);
        g_n.t2b_copies++;
        if (ir) rec_at(ir)->t2b++;
        break;
    }
    case OP_COPY_T2T: {
        uint32_t ir = rec_of_tex(&((const p_copy_t2t *)a)->dst);
        if (ir) rec_at(ir)->gpu |= G_COPY;
        break;
    }
    case OP_CLEAR_TEX: {
        uint32_t ir = rec_of_tex(&((const p_clear_tex *)a)->tex);
        if (ir) rec_at(ir)->gpu |= G_CLEAR;
        break;
    }
    }
    plat_mutex_unlock(&g_lock);
}

// The submission with fence value `fence` is complete (and only it: queues complete independently, so a later fence value
// can be complete while an earlier one, held at a wait, is not): each texture it bound is bound, and, if the CPU has not
// written it since its last bind, queued for the next present. The writes before this bind are read first, so that a page
// written before the bind is not taken for a write after it.
void texheap_complete(uint64_t fence) {
    plat_mutex_lock(&g_lock);
    ensure();
    for (size_t at = 0; at < g_sub_n; at++) {
        Sub *s = g_subs[g_sub_head + at];
        if (s->fence != fence) continue;
        for (uint32_t k = 0; k < s->n; k++) {
            Rec *r = rec_at(s->recs[k]);
            r->binds++;
            if (!r->live || r->bound) continue;
            ready_add(s->recs[k]);
        }
        free(s->recs);
        free(s);
        memmove(g_subs + g_sub_head + at, g_subs + g_sub_head + at + 1, (g_sub_n - at - 1) * sizeof *g_subs);
        g_sub_n--;
        break;
    }
    if (!g_sub_n) g_sub_head = 0;
    plat_mutex_unlock(&g_lock);
}

// ---------- the frame and the sample
static int rec_class(const Rec *r) { return r->cpu_pages || r->pkg ? 0 : r->gpu ? 1 : 2; }   // cpu, gpu, idle

static uint64_t committed_resident(uint64_t addr, uint64_t size, uint64_t *resident) {
    uint64_t committed = 0, end = addr + size;
    *resident = 0;
    for (uint64_t a = addr; a < end;) {
        int c = 0;
        uint64_t n = rt_mem_commit_run(a, end - a, &c);
        if (!n || n == UINT64_MAX) break;
        if (c) {
            committed += n;
            *resident += plat_resident(a, n);
        }
        a += n;
    }
    return committed;
}

static void sample(uint64_t seq) {
    RtMemStats ms;
    rt_mem_measure(&ms);
    uint64_t pools = 0, pool_bytes = 0, recs = 0, rec_bytes = 0, cb[3] = {0}, cn[3] = {0}, tex_pages = 0, written = 0,
             elig = 0, sel = 0, why[4] = {0};   // why: written pages kept out by no bind, a buffer, another texture, another owner
    for (uint32_t p = 1; p < g_npool; p++) {
        const PoolRec *q = pool_at(p);
        pools += q->live, pool_bytes += q->live ? q->size : 0;
    }
    uint32_t *touched = NULL;
    size_t nt = 0, ct = 0;
    for (uint32_t ir = 1; ir < g_nrec; ir++) {
        const Rec *r = rec_at(ir);
        if (!r->live) continue;
        PoolRec *q = r->pool ? pool_at(r->pool) : NULL;
        int c = rec_class(r);
        recs++, rec_bytes += r->size, cb[c] += r->size, cn[c]++;
        uint64_t rw = 0, rs = 0, re = 0;
        if (!gfx_gpu_only((uintptr_t)r->mem)) {
            uint64_t lo = up(r->mem), hi = down(r->mem + r->size);
            for (uint64_t x = lo; x < hi; x += PAGE) {
                uint32_t i;
                const Chunk *k = ck(x, 0, &i);
                if (!k) continue;
                if (k->rec[i] != ir) {   // the page's owner is another record: a texture initialised over it since
                    why[3] += (k->fl[i] & F_WRITTEN) != 0;
                    continue;
                }
                rw += (k->fl[i] & F_WRITTEN) != 0, rs += (k->fl[i] & F_SEL) != 0;
                if ((k->fl[i] & (F_WRITTEN | F_SEL)) != F_WRITTEN) continue;   // not written, or selected already
                // the first condition a written page fails: no completed bind, a buffer or event, another texture
                if (!r->binds) why[0]++;
                else if (k->oth[i]) why[1]++;
                else if (k->tex[i] != 1) why[2]++;
                else re++;
            }
        }
        tex_pages += r->pages, written += rw, sel += rs, elig += re;
        if (!q) continue;
        if (q->agg_seq != seq) {
            q->agg_seq = seq;
            q->recs = q->rec_bytes = q->cpu_bytes = q->gpu_bytes = q->idle_bytes = q->tex_pages = q->written = q->selected =
                q->eligible = 0;
            if (nt == ct) touched = xrealloc(touched, (ct = ct ? 2 * ct : 256) * sizeof *touched);
            touched[nt++] = r->pool;
        }
        q->recs++, q->rec_bytes += r->size, q->tex_pages += r->pages, q->written += rw, q->selected += rs, q->eligible += re;
        (c == 0 ? &q->cpu_bytes : c == 1 ? &q->gpu_bytes : &q->idle_bytes)[0] += r->size;
    }
    RtPageStats ps = {0}, one;
    uint64_t arms = 0, releases = 0, recommits = 0, busy = 0;
    for (int r = 0; r < 2; r++) {
        if (g_range[r] < 0 || !rt_pages_stats(g_range[r], &one)) continue;
        arms += one.arms, releases += one.releases, recommits += one.recommits, busy += one.busy_refusals;
        ps.guest_reads += one.guest_reads, ps.host_reads += one.host_reads, ps.guest_writes += one.guest_writes;
        ps.host_writes += one.host_writes;
    }
    const uint64_t v[] = {seq, ms_now(), ms.r[RT_MEM_HEAP].mapped, ms.r[RT_MEM_HEAP].committed, ms.r[RT_MEM_HEAP].resident,
                          ms.process_private, ms.process_working_set, pools, pool_bytes, recs, rec_bytes, cn[0], cb[0],
                          cn[1], cb[1], cn[2], cb[2], tex_pages, written, elig, sel, arms, releases, recommits, busy,
                          ps.guest_reads, ps.host_reads, ps.guest_writes, ps.host_writes, g_n.stale_touches, g_n.b2t_copies,
                          g_n.b2t_bytes, g_n.t2b_copies, g_n.pkg_inits, g_n.pkg_bytes, g_n.read_texels, g_n.inval_calls,
                          g_n.inval_tex_calls, g_n.inval_tex_bytes, g_n.flush_calls, g_n.flush_tex_calls,
                          g_n.flush_tex_bytes, g_n.partial, g_n.no_range, why[0], why[1], why[2], why[3], g_n.bind_ids,
                          g_n.bind_miss, g_n.bind_dead, g_n.b2t_nomatch};
    row("sample");
    for (size_t i = 0; i < sizeof v / sizeof *v; i++) row("\t%llu", (ull)v[i]);
    row("\n");
    for (size_t n = 0; n < nt; n++) {
        PoolRec *q = pool_at(touched[n]);
        uint64_t res = 0, com = 0;
        if (q->region) com = committed_resident(q->mem, q->size, &res);
        row("pool\t%llu\t%u\t%u\t%#x\t%llu\t%#llx\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%u\t%u\t%u\n",
            (ull)seq, touched[n], q->cls, q->flags, (ull)q->size, (ull)q->mem, (ull)com, (ull)res, (ull)q->recs,
            (ull)q->rec_bytes, (ull)q->cpu_bytes, (ull)q->gpu_bytes, (ull)q->idle_bytes, (ull)q->tex_pages, (ull)q->written,
            (ull)q->selected, (ull)q->eligible, q->win_frames, q->win_pages, q->win_max);
        q->win_frames = q->win_pages = q->win_max = 0;
    }
    free(touched);
    for (uint32_t p = 1; p < g_npool; p++) {   // pools the CPU wrote and that hold no texture storage
        PoolRec *q = pool_at(p);
        if (!q->win_frames) continue;
        row("wpool\t%llu\t%u\t%u\t%llu\t%u\t%u\t%u\n", (ull)seq, p, q->cls, (ull)q->size, q->win_frames, q->win_pages,
            q->win_max);
        q->win_frames = q->win_pages = q->win_max = 0;
    }
}

void texheap_present(uint64_t seq) {
    plat_mutex_lock(&g_lock);
    ensure();
    g_seq = g_cur = seq;
    read_writes();
    for (size_t i = 0; i < g_frpools_n; i++) {   // fold the frame into the pools' windows
        PoolRec *q = pool_at(g_frpools[i]);
        q->win_frames++, q->win_pages += q->fr_pages;
        if (q->fr_pages > q->win_max) q->win_max = q->fr_pages;
        q->fr_pages = 0;
    }
    row("frame\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n", (ull)seq, (ull)g_fr.pages, (ull)g_fr.tex_pages, (ull)g_fr.pool_pages,
        (ull)g_fr.outside, (ull)g_fr.pools);
    memset(&g_fr, 0, sizeof g_fr);   // what the completions since the last present read counts in this frame, too
    g_frpools_n = 0;
    for (size_t i = 0; i < g_ready_n; i++) {
        Rec *r = rec_at(g_ready_recs[i]);
        r->queued = 0;
        select_rec(g_ready_recs[i]);
    }
    g_ready_n = 0;
    if (seq == 1 || seq % g_every == 0) sample(seq);
    if (g_f) fflush(g_f);
    plat_mutex_unlock(&g_lock);
}
