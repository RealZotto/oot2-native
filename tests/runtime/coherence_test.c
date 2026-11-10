// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the coherence of mirrors that need no GPU: the page generations (runtime/gpu/pagegen.c,
// oot2/pagegen.h) and the mirrors of the backing map (runtime/gpu/backing.c, oot2/backing.h, Mirrors). Driven by
// tests/test_runtime_coherence.py: `coherence_test <case>`, one case per process. A passing case prints "ok <case>"; a
// failing one must end in one named fatal, which the driver checks.
//
// The guest memory is the runtime's own (rt_mem_init, SetHeapSize, first-touch commit). The device stands in as in
// tests/runtime/backing_test.c (a block is host memory, an import the host range itself) and adds a GPU that runs a
// submission's copies when the test says the device reached it: gpu_run(v, work) does v's prologue (the uploads into
// mirrors), then the test's own GPU work, then v's readbacks, and only then raises the counter the completion thread
// waits on. The write watch stands in too, as the operating system's mark of each heap page, which a query with reset
// clears for the pages it returns, at most max a call, as GetWriteWatch does; so the cases run on Linux, which has no
// write watch. `watch` uses the real one, on Windows.
//
//   generations  a query stamps what it reports with a new generation; a second consumer that queried nothing sees the
//                same pages; a query reports a page once; a full answer is asked again; marks; an untracked range
//   upload       a CPU write before a submission is in the mirror that submission reads, the next write in the next one;
//                only the stamped pages are copied, once a submission; a mark (guest_written) is copied too; the
//                texture stand-in sees the pages the mirror's query stamped
//   admission    the rule at its edge; a mirror past it, or over a range the watch does not cover, is an import, and the
//                figures moving back admit the next one
//   guarded_admission   the reserved growth space counts against the mirror's VRAM budget
//   writeback    a declared GPU write reaches guest memory only once the device reached its submission; a copy into the
//                mirror meanwhile leaves its bytes out; guest_read counts a read before the fence, cpu_synced finds
//                nothing left behind
//   writeback_no_reupload, writeback_partial_page, writeback_replaced, writeback_long_run   the pages a write-back put
//                into guest memory are in the mirror already: the next sync does not hand them back (whole pages only),
//                the guest sees the GPU's bytes, a CPU write into one is copied in, a mirror that replaced another takes
//                every page, and a long run of write-backs leaves the record where it was
//   gpu_mark     pagegen_mark_gpu: whole pages recorded and their echo in the write watch settled, partial ones stamped
//   gpu_mark_direct   pagegen_mark_gpu_direct: pages the GPU wrote into guest memory itself; the ones the CPU wrote since
//                the mirror's sync (the watch says so, a scan or a mark stamped them after it) keep the plain stamp, the
//                others are recorded GPU-current, another submission's GPU stamp is not the CPU's, an unwatched range
//                keeps every page plain; gpu_mark_direct_partial stops by name on a range that is not whole pages
//   replace      a bind that grows a mirror's run waits for the submission in flight that wrote it, so the new mirror
//                holds the GPU's write
//   watch        (Windows; a skip line elsewhere) the real write watch: every hook and sync between a CPU write and the
//                next submission leaves the write to that submission, and a second consumer sees the same page
//   raw_grow     a bind of a raw-access submission that starts in a mirror its page table maps and runs past its end:
//                the mirror keeps the part outside the bind, an import covers the rest, the table's runs and the bind
//                reach the same memory for every page
//   raw_grow_head   a bind below a mirror that ends inside it takes its first pages: the rest keeps its own bytes
//   raw_grow_whole, raw_grow_beside   a bind over a whole mirror takes it; one beside a mirror leaves it whole
//   raw_grow_wait   the mirror gives its pages up only once the older raw-access submission in flight is done
//   raw_grow_taken, raw_grow_neighbour, raw_grow_taken_wait, raw_grow_taken_head, raw_grow_trimmed   an earlier bind
//                keeps the same device bytes when a later bind extends its mirror, including imports and older writes
//   raw_grow_taken_layout   the layout of the stop of the first Forest run: a mirror of 700 KiB, a bind of 3 KiB across
//                a page boundary near its start, then a bind that starts below the mirror and ends past that boundary
//   raw_grow_guarded_tail, raw_grow_guarded_head   the same when a GPU-only extent blocks the adjacent block space
//   raw_grow_over_total   the growth is held to its own size, not to the mirrors' total: with 132 MiB of mirrors made by
//                binds (more than the promotions' cap, which only a promotion keeps), one still grows by two pages
//   raw_grow_head_import, raw_grow_tail_import   an import the growth overlaps goes, but the mirror takes only the bind's
//                pages and copies only those in; the import's other pages (128 KiB at the head, 16 MiB at the tail) get
//                an import again
//   raw_diag_hull, raw_diag_dry, raw_diag_birth   the stop of a bind over a mirror that an import the submission bound
//                abuts prints the straddle dump before it: the hull of what each resolve reached in the import and the
//                mirror, the birth of every backing near the bind, the binds of the submission in order (before and after
//                its raw-access begin), the growth's guards each evaluated on its own, and a digest of the map that the
//                evaluation leaves as it was; the stop's own text is the one raw_grow_bound_import pins
//   raw_ev_miss, raw_ev_survives_hits, raw_ev_lost, raw_ev_origin   the same stop with the event ring in the dump: the miss
//                that cut a mirror, the dry verdict of growing it instead (right after the miss, with the imports it would
//                replace as they were), the cut and what the bind made, kept after a thousand hits, counted when the ring
//                already lost some, and with the origin of a mirror a promotion made
//   raw_dry_grow, raw_dry_g4, raw_dry_g4_on, raw_dry_cap, raw_dry_kind, raw_dry_none, raw_dry_g8, raw_dry_g5, raw_dry_g1,
//                raw_dry_g2   the dry verdict's counters: would grow, refused by an import the submission used (its hull off
//                the pages the growth adds, then on them), by the cap on the mirrors' total, by the rule's own kind, not
//                evaluated, by the write watch not covering the pages, by the head reaching before the buffer's start, by
//                the size of the growth, by another mirror on the pages; the map is the same after each evaluation
//   raw_sync_wait   a mirror the CPU wrote is brought current for a raw-access submission after the older one in flight
//   raw_sync_nowait the same with only a GPU-current page stamped since: no wait, nothing is copied over the older stores
//   raw_promote, raw_promote_small, raw_promote_wait, raw_promote_cap   an import a raw-access submission binds a range larger than a
//                uniform's of becomes a mirror at the next raw-access begin: brought current, in the page table, left alone when
//                the submission bound it before its raw stage or the admission refuses, never for a small bind, after the older
//                raw-access submission in flight, and not past the mirrors' cap (BACKING_PROMOTE_MIRROR_MAX)
//   raw_grow_used_import, raw_grow_used_edge   the growth's pages are the lowest of an import the submission used, below
//                everything it resolved in it: the import is trimmed in place (same buffer, memory and offset of what it still
//                holds, one page less at its start), no page has two backings, the page the mirror took is copied in and
//                a write through the mirror reaches guest memory; the edge case has the hull start on the page after
//   raw_grow_used_wait   the growth waits for the plain submission issued before it that wrote through the import, not for
//                the submission being recorded, whose stamp the import carries
//   raw_grow_used_written, raw_grow_used_hull, raw_grow_used_hull_edge, raw_grow_used_loose, raw_grow_used_up   stop by the
//                rule that refuses such an import: written on the GPU by the submission, its hull on the pages (a byte
//                before the page's end included), an extent not stated, the pages above the hull (a head growth)
//   raw_grow_heir, raw_grow_heir_old   an import that replaces two used ones takes over their hull, so a growth over it
//                sees it; a hull of an older submission is not taken over; raw_grow_heir_hull, raw_grow_heir_written stop
//                by the rule that sees what was taken over (the hull on the pages, a GPU write of the submission)
//   unwritten, replace_written, raw_grow_bound_import, raw_grow_fragmented, raw_grow_guarded_blocked,
//   raw_grow_untracked, raw_grow_cap   stop by name, each with the reason the mirror could not grow
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu.h"
#include "oot2/memory.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define GiB (1ull << 30)
#define PG 0x1000ull
typedef unsigned long long ull;

// ---------- the timeline over a test's counter
static PlatMutex g_src_lock = PLAT_MUTEX_INIT;
static PlatCond g_src_cond = PLAT_COND_INIT;
static uint64_t g_reached;

static uint64_t counter_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    plat_mutex_lock(&g_src_lock);
    uint64_t start = plat_now_ns();
    while (g_reached < value) {
        uint64_t spent = plat_now_ns() - start;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_src_cond, &g_src_lock, timeout_ns - spent);
    }
    uint64_t r = g_reached;
    plat_mutex_unlock(&g_src_lock);
    return r;
}

static void counter_raise(uint64_t to) {
    plat_mutex_lock(&g_src_lock);
    if (to > g_reached) g_reached = to;
    plat_cond_broadcast(&g_src_cond);
    plat_mutex_unlock(&g_src_lock);
}

// ---------- the stand-in write watch: one mark per heap page
static uint64_t g_heap, g_heap_size;
static uint8_t *g_marks;
static int g_untracked;
static uint64_t g_src_calls, g_src_resets;   // queries, and those that take the marks they report

// The echo window: pages in [g_echo_lo, g_echo_hi) also count as written when their bytes differ from a copy taken at the
// last query (or when the window was set), as the operating system's mark follows a write of any content by any code,
// the runtime's own write-backs into guest memory included. 0 for none: the cases that do not model it never see it.
static uint64_t g_echo_lo, g_echo_hi;
static uint8_t *g_echo_shadow;

static uint64_t src_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    g_src_calls++;
    g_src_resets += (uint64_t)!!reset;
    if (g_untracked || addr < g_heap || addr + size > g_heap + g_heap_size) return UINT64_MAX;
    uint64_t n = 0;
    for (uint64_t a = addr & ~(PG - 1); a < addr + size && n < max; a += PG) {
        uint64_t i = (a - g_heap) / PG;
        int echo = a >= g_echo_lo && a < g_echo_hi && memcmp((void *)(uintptr_t)a, g_echo_shadow + (a - g_echo_lo), PG);
        if (!g_marks[i] && !echo) continue;
        pages[n++] = a;
        if (reset) {
            g_marks[i] = 0;
            if (echo) memcpy(g_echo_shadow + (a - g_echo_lo), (void *)(uintptr_t)a, PG);
        }
    }
    return n;
}

// From now on the pages of [lo, hi) are written when their content changes, whoever changed it.
static void echo_on(uint64_t lo, uint64_t hi) {
    g_echo_shadow = malloc((size_t)(hi - lo));
    CHECK(g_echo_shadow);
    memcpy(g_echo_shadow, (void *)(uintptr_t)lo, (size_t)(hi - lo));
    g_echo_lo = lo, g_echo_hi = hi;
}

// The CPU writes word over [a, a + n), as the game's stores would: committed on first touch, marked.
static void cpu_fill(uint64_t a, uint64_t n, uint32_t word) {
    for (uint64_t p = a; p < a + n; p += 4) *(volatile uint32_t *)(uintptr_t)p = word;
    for (uint64_t p = a & ~(PG - 1); p < a + n; p += PG) g_marks[(p - g_heap) / PG] = 1;
}

static int words_are(const void *p, uint64_t n, uint32_t word) {
    for (uint64_t i = 0; i < n / 4; i++)
        if (((const volatile uint32_t *)p)[i] != word) return 0;
    return 1;
}

// ---------- the stand-in device and its GPU
static struct {
    uint64_t used, budget;          // the figures vram answers
    uint64_t uploads, upload_bytes, readbacks;
    uint64_t recording;             // the submission being recorded (its thread is the test's)
} D = {.budget = 8 * GiB};

enum { PRO, TAIL };
typedef struct Op { uint64_t value; int phase; const void *src; void *dst; uint64_t size; } Op;
static Op g_q[4096];
static int g_nq;
static void *g_keep[8192];
static int g_nkeep;

static void *keep(void *p) {
    CHECK(p && g_nkeep < 8192);
    return g_keep[g_nkeep++] = p;
}

static uint8_t *mem_at(const BackingMem *m, uint64_t off) { return (uint8_t *)(uintptr_t)(m->buffer + off); }

static int dev_block_new(uint64_t size, BackingMem *out) {
    void *p = calloc(1, (size_t)size);
    if (!p) return 0;
    *out = (BackingMem){.buffer = (uintptr_t)p, .memory = 1, .address = (uintptr_t)p, .size = size};
    return 1;
}

static int dev_import(uint64_t host, uint64_t size, BackingMem *out) {
    *out = (BackingMem){.buffer = host, .memory = 2, .address = host, .size = size};
    return 1;
}

static void dev_free(const BackingMem *m) {
    if (m->memory == 1) free((void *)(uintptr_t)m->buffer);
}

static int dev_vram(uint64_t *used, uint64_t *budget) {
    *used = D.used, *budget = D.budget;
    return 1;
}

static void dev_upload(uint64_t value, const BackingCopy *c, int n) {
    CHECK(value && value == D.recording);
    for (int i = 0; i < n; i++) {
        void *s = keep(malloc((size_t)c[i].size));
        memcpy(s, c[i].host, (size_t)c[i].size);   // the host bytes are taken now
        CHECK(g_nq < 4096);
        g_q[g_nq++] = (Op){value, PRO, s, mem_at(c[i].mem, c[i].offset), c[i].size};
        D.uploads++, D.upload_bytes += c[i].size;
    }
}

static void dev_readback(uint64_t value, BackingCopy *c, int n) {
    CHECK(value && value == D.recording);
    for (int i = 0; i < n; i++) {
        void *s = keep(calloc(1, (size_t)c[i].size));   // zeros until the GPU copies into it
        CHECK(g_nq < 4096);
        g_q[g_nq++] = (Op){value, TAIL, mem_at(c[i].mem, c[i].offset), s, c[i].size};
        c[i].host = s;
        D.readbacks++;
    }
}

static void dev_close(void) {}

static const BackingOps k_ops = {.block_new = dev_block_new, .import = dev_import, .free = dev_free, .vram = dev_vram,
                                 .upload = dev_upload, .readback = dev_readback, .close = dev_close};

// The device reaches v: its prologue, the test's GPU work, its readbacks; then the counter, and the value done.
static void gpu_run(uint64_t v, void (*work)(void)) {
    for (int i = 0; i < g_nq; i++)
        if (g_q[i].value == v && g_q[i].phase == PRO) memcpy(g_q[i].dst, g_q[i].src, (size_t)g_q[i].size);
    if (work) work();
    for (int i = 0; i < g_nq; i++)
        if (g_q[i].value == v && g_q[i].phase == TAIL) memcpy(g_q[i].dst, g_q[i].src, (size_t)g_q[i].size);
    counter_raise(v);
    CHECK(gpu_sync_wait(v, 5000000000ull));
}

static uint64_t sub_begin(void) { return D.recording = gpu_sync_begin(); }

static void sub_end(uint64_t v) {
    backing_submit_end(v);
    gpu_sync_end();
    D.recording = 0;
}

// ---------- guest memory and pools
static Cpu c;

static void setup(uint64_t heap, int real_watch) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    c.x[0] = 0, c.x[1] = heap;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    g_heap = c.x[1], g_heap_size = heap;
    g_marks = calloc((size_t)(heap / PG), 1);
    CHECK(g_marks);
    pagegen_reset();
    if (!real_watch) pagegen_set_source(src_written);
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
    backing_open(&k_ops);
}

static int want_all(const BackingAsk *a) {
    (void)a;
    return 1;
}

static pool_t pool(uint64_t mem, uint64_t size) {
    pool_t p = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = 0, .mem = (uint8_t *)(uintptr_t)mem, .size = size,
                .game_mem = (uint8_t *)(uintptr_t)mem};
    CHECK(backing_pool_init(&p));
    return p;
}

static BackingRef resolve(uint64_t addr, uint64_t size, uint64_t value) {
    BackingRef r;
    memset(&r, 0xA5, sizeof r);
    backing_resolve(addr, size, value, &r);
    return r;
}

static uint8_t *bytes(const BackingRef *r) { return (uint8_t *)(uintptr_t)(r->buffer + r->offset); }

static BackingStats stats(void) {
    BackingStats s;
    backing_stats(&s);
    return s;
}

// The runs pagegen_runs gives, collected.
typedef struct { uint64_t lo[64], hi[64]; int n; } Runs;
static void collect(void *ctx, uint64_t lo, uint64_t hi) {
    Runs *r = ctx;
    CHECK(r->n < 64);
    r->lo[r->n] = lo, r->hi[r->n] = hi, r->n++;
}
static Runs runs(uint64_t lo, uint64_t hi, uint64_t since, uint64_t *pages) {
    Runs r = {.n = 0};
    *pages = pagegen_runs(lo, hi, since, collect, &r);
    return r;
}

static void done(const char *name) {
    gpu_sync_finish();
    backing_close();
    for (int i = 0; i < g_nkeep; i++) free(g_keep[i]);
    printf("ok %s\n", name);
}

// ---------- cases
static void case_generations(void) {
    setup(64 * MiB, 0);
    uint64_t a = g_heap + 1 * MiB, pages;
    cpu_fill(a, 4, 1);
    cpu_fill(a + 3 * PG, 2 * PG, 2);
    uint64_t g1 = pagegen_scan(a, a + 1 * MiB);
    CHECK(g1 == 1 && pagegen_now() == 1);
    CHECK(pagegen_stamp(a) == g1 && pagegen_stamp(a + PG) == 0 && pagegen_stamp(a + 4 * PG) == g1);
    Runs r = runs(a, a + 1 * MiB, 0, &pages);
    CHECK(pages == 3 && r.n == 2 && r.lo[0] == a && r.hi[0] == a + PG && r.lo[1] == a + 3 * PG && r.hi[1] == a + 5 * PG);
    // A second consumer that queried nothing reads the same pages; a second query reports none of them again.
    uint64_t g2 = pagegen_scan(a, a + 1 * MiB);
    CHECK(g2 == g1 + 1);
    runs(a, a + 1 * MiB, g1, &pages);
    CHECK(pages == 0);
    runs(a, a + 1 * MiB, 0, &pages);
    CHECK(pages == 3);
    cpu_fill(a + 7 * PG, 4, 3);
    uint64_t g3 = pagegen_scan(a, a + 1 * MiB);
    r = runs(a, a + 1 * MiB, g2, &pages);
    CHECK(pages == 1 && r.n == 1 && r.lo[0] == a + 7 * PG && pagegen_stamp(a + 7 * PG) == g3);
    // More pages than one call of the source returns: the full answer is asked again, and every page is stamped.
    uint64_t b = g_heap + 8 * MiB;
    cpu_fill(b, 5000 * PG, 4);
    uint64_t calls = g_src_calls, g4 = pagegen_scan(b, b + 5000 * PG);
    runs(b, b + 5000 * PG, g3, &pages);
    CHECK(pages == 5000 && g_src_calls - calls == 2 && pagegen_stamp(b + 4999 * PG) == g4);
    // Marks: a page the CPU said it wrote; the GPU window is no guest memory.
    uint64_t g5 = pagegen_mark(a + 0x10, a + 0x20);
    r = runs(a, a + 1 * MiB, g4, &pages);
    CHECK(pages == 1 && r.lo[0] == a && pagegen_stamp(a) == g5);
    PagegenStats s;
    pagegen_stats(&s);
    uint64_t marked = s.mark_pages;
    pagegen_mark(GFX_GPU_WINDOW_BASE, GFX_GPU_WINDOW_BASE + PG);
    pagegen_stats(&s);
    CHECK(s.mark_pages == marked && s.scans == 4 && s.scan_pages == 3 + 1 + 5000 && s.untracked == 0);
    // A range the source does not watch: nothing says which of its pages changed.
    g_untracked = 1;
    CHECK(pagegen_scan(a, a + PG) == UINT64_MAX);
    pagegen_stats(&s);
    CHECK(s.untracked == 1);
    done("generations");
}

static void case_upload(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB, pages;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0x11111111u);
    uint64_t v = sub_begin();
    BackingRef r = resolve(p + 0x1000, 0x2000, v);
    CHECK(r.kind == BACKING_MIRROR && stats().mirrors == 1 && stats().mirror_bytes == 1 * MiB);
    uint8_t *m = bytes(&r) - 0x1000;   // the mirror of the pool's first byte
    CHECK(!words_are(m, 1 * MiB, 0x11111111u));   // nothing reaches the mirror before the device runs the prologue
    sub_end(v);
    gpu_run(v, NULL);
    CHECK(words_are(m, 1 * MiB, 0x11111111u) && D.upload_bytes == 1 * MiB);   // all of it, the first time

    // A write before the next submission is read by it, and only its page is copied, once in the submission.
    uint64_t tex_since = pagegen_now(), before = D.upload_bytes;
    cpu_fill(p + 0x5000, 0x10, 0x22222222u);
    v = sub_begin();
    resolve(p + 0x1000, 0x2000, v);
    resolve(p + 0x8000, 0x100, v);
    sub_end(v);
    gpu_run(v, NULL);
    CHECK(words_are(m + 0x5000, 0x10, 0x22222222u) && words_are(m + 0x5010, 0x10, 0x11111111u));
    CHECK(D.upload_bytes - before == PG && stats().mirror_syncs == 2);
    // A second consumer (a texture's stand-in) that queried nothing sees the page the mirror's query stamped.
    Runs t = runs(p, p + 1 * MiB, tex_since, &pages);
    CHECK(pages == 1 && t.lo[0] == p + 0x5000);

    // A write the watch does not see, which the CPU says it made (guest_written: a flush): copied at the next use.
    *(volatile uint32_t *)(uintptr_t)(p + 0x9000) = 0x33333333u;
    backing_guest_written(p + 0x9000, 4);
    v = sub_begin();
    resolve(p + 0x1000, 0x2000, v);
    sub_end(v);
    gpu_run(v, NULL);
    CHECK(*(uint32_t *)(m + 0x9000) == 0x33333333u);
    // A submission that resolves no mirror queries nothing.
    uint64_t calls = g_src_calls;
    v = sub_begin();
    sub_end(v);
    gpu_run(v, NULL);
    CHECK(g_src_calls == calls && stats().mirror_syncs == 3);
    backing_pool_fini(&pp);
    CHECK(stats().mirrors == 0 && stats().mirror_bytes == 0);
    done("upload");
}

static void case_admission(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    CHECK(backing_mirror_fits(0, 8 * GiB, 7 * GiB) && !backing_mirror_fits(0, 8 * GiB, 7 * GiB + 1));
    CHECK(!backing_mirror_fits(1, 8 * GiB, 7 * GiB) && backing_mirror_fits(1 * GiB, 8 * GiB, 6 * GiB));
    CHECK(!backing_mirror_fits(9 * GiB, 8 * GiB, 0) && !backing_mirror_fits(0, 8 * GiB, UINT64_MAX));
    pool_t a = pool(g_heap + 1 * MiB, 1 * MiB), b = pool(g_heap + 3 * MiB, 1 * MiB), d = pool(g_heap + 5 * MiB, 1 * MiB);
    pool_t e = pool(g_heap + 7 * MiB, 1 * MiB);
    cpu_fill(g_heap + 1 * MiB, 8 * MiB, 0);
    D.used = 0;
    CHECK(resolve(g_heap + 1 * MiB, 0x100, 0).kind == BACKING_MIRROR);
    // The figures leave half a MiB: a mirror of 1 MiB is refused, and the bind is an import.
    D.used = 8 * GiB - 8 * GiB / BACKING_MIRROR_RESERVE - 512 * KiB;
    CHECK(resolve(g_heap + 3 * MiB, 0x100, 0).kind == BACKING_IMPORT);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.imports == 1 && s.mirrors_refused_budget == 1);
    // The figures move back: the next mirror is admitted.
    D.used = 8 * GiB - 8 * GiB / BACKING_MIRROR_RESERVE - 1 * MiB - 2 * BACKING_MIRROR_GUARD;
    CHECK(resolve(g_heap + 5 * MiB, 0x100, 0).kind == BACKING_MIRROR);
    // A range the write watch does not cover: an import.
    D.used = 0, g_untracked = 1;
    CHECK(resolve(g_heap + 7 * MiB, 0x100, 0).kind == BACKING_IMPORT);
    s = stats();
    CHECK(s.mirrors == 2 && s.imports == 2 && s.mirrors_refused_untracked == 1 && s.mirrors_made == 2);
    backing_pool_fini(&a), backing_pool_fini(&b), backing_pool_fini(&d), backing_pool_fini(&e);
    done("admission");
}

static void case_guarded_admission(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t a = g_heap + 1 * MiB, b = g_heap + 3 * MiB;
    pool_t pa = pool(a, 64 * KiB), pb = pool(b, 64 * KiB);
    cpu_fill(a, 64 * KiB, 0xAAAAAAAAu);
    cpu_fill(b, 64 * KiB, 0xBBBBBBBBu);
    D.used = D.budget - D.budget / BACKING_MIRROR_RESERVE - (64 * KiB + 2 * BACKING_MIRROR_GUARD - 1);
    CHECK(resolve(a, 4, 0).kind == BACKING_IMPORT);
    D.used = D.budget - D.budget / BACKING_MIRROR_RESERVE - 64 * KiB - 2 * BACKING_MIRROR_GUARD;
    CHECK(resolve(b, 4, 0).kind == BACKING_MIRROR);
    CHECK(stats().mirrors_refused_budget == 1);
    backing_pool_fini(&pa), backing_pool_fini(&pb);
    done("guarded_admission");
}

static uint8_t *g_mirror;   // the GPU work of a case writes into it
static void gpu_writes_cc(void) { memset(g_mirror + 0x100, 0xCC, 0x200); }

static void case_writeback(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    uint64_t v1 = sub_begin();
    BackingRef r = resolve(p, 0x4000, v1);
    CHECK(r.kind == BACKING_MIRROR);
    g_mirror = bytes(&r);
    backing_gpu_wrote(p + 0x100, 0x200, v1);
    sub_end(v1);
    // Before the device reached v1: guest memory holds the old bytes, the write waits, and the hooks find nothing
    // left behind; a read of the bytes is counted as early.
    CHECK(words_are((void *)(uintptr_t)(p + 0x100), 0x200, 0xAAAAAAAAu) && stats().pending == 1);
    backing_cpu_synced(gpu_sync_done());
    backing_guest_read(p + 0x180, 4);
    backing_guest_read(p + 0x400, 4);
    CHECK(stats().reads_in_flight == 1);
    // Meanwhile the CPU writes another word of the page, and the next submission copies the page into the mirror:
    // the bytes of the GPU write are left out of the copy.
    cpu_fill(p + 0x800, 4, 0xBBBBBBBBu);
    uint64_t v2 = sub_begin();
    resolve(p, 0x4000, v2);
    sub_end(v2);
    gpu_run(v1, gpu_writes_cc);
    CHECK(words_are((void *)(uintptr_t)(p + 0x100), 0x200, 0xCCCCCCCCu));   // written back once v1 was reached
    CHECK(words_are((void *)(uintptr_t)(p + 0x300), 0x100, 0xAAAAAAAAu));   // and nothing around it
    BackingStats s = stats();
    CHECK(s.pending == 0 && s.writebacks == 1 && s.writeback_bytes == 0x200 && D.readbacks == 1);
    gpu_run(v2, NULL);
    CHECK(words_are(g_mirror + 0x100, 0x200, 0xCCCCCCCCu) && *(uint32_t *)(g_mirror + 0x800) == 0xBBBBBBBBu);
    backing_cpu_synced(v2);
    // The page written back is stamped: another consumer learns the GPU changed it.
    CHECK(pagegen_stamp(p) > 0 && pagegen_stamp(p) <= pagegen_now());
    backing_pool_fini(&pp);
    done("writeback");
}

// A GPU write the write-back put into guest memory is in the mirror already: the next sync must not hand it back
// (the title movie's scratch buffers were copied out and in again every present), the guest's view of it is the GPU's
// bytes (the write-back stays), and a CPU write into one of those pages is still copied in, with the GPU's bytes around
// it. The write watch is modelled with its echo: the write-back's own copy into guest memory dirties the pages, as it does
// in the operating system, and must not be taken for a CPU write.
static void gpu_writes_cc_pages(void) { memset(g_mirror + 0x1000, 0xCC, 0x2000); }

static void case_writeback_no_reupload(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    echo_on(p, p + 1 * MiB);
    uint64_t v1 = sub_begin();
    BackingRef r = resolve(p, 0x4000, v1);
    CHECK(r.kind == BACKING_MIRROR);
    g_mirror = bytes(&r);
    backing_gpu_wrote(p + 0x1000, 0x2000, v1);   // two whole pages
    sub_end(v1);
    uint64_t since = pagegen_now();   // what the mirror's own queries stamped is behind this
    gpu_run(v1, gpu_writes_cc_pages);
    // The guest sees the GPU's bytes (the view the lazy variants would have to keep), and the neighbours stay.
    CHECK(words_are((void *)(uintptr_t)(p + 0x1000), 0x2000, 0xCCCCCCCCu));
    CHECK(words_are((void *)(uintptr_t)(p + 0xF00), 0x100, 0xAAAAAAAAu) && words_are((void *)(uintptr_t)(p + 0x3000), 0x100, 0xAAAAAAAAu));
    CHECK(pagegen_gpu_stamp(p + 0x1000) == pagegen_stamp(p + 0x1000) && pagegen_stamp(p + 0x2000) > 0);
    uint64_t before = D.upload_bytes;
    uint64_t v2 = sub_begin();
    resolve(p, 0x4000, v2);       // the next sync must not hand the mirror its own bytes back
    sub_end(v2);
    CHECK(D.upload_bytes == before);
    gpu_run(v2, NULL);
    CHECK(words_are(g_mirror + 0x1000, 0x2000, 0xCCCCCCCCu));
    // Another consumer still learns that the GPU changed the pages.
    uint64_t pages;
    runs(p, p + 1 * MiB, since, &pages);
    CHECK(pages == 2);
    // A CPU write into one of those pages is copied in: the page, with the GPU's bytes around the written word.
    cpu_fill(p + 0x1800, 4, 0xBBBBBBBBu);
    uint64_t v3 = sub_begin();
    resolve(p, 0x4000, v3);
    sub_end(v3);
    CHECK(D.upload_bytes == before + PG);
    gpu_run(v3, NULL);
    CHECK(*(uint32_t *)(g_mirror + 0x1800) == 0xBBBBBBBBu && *(uint32_t *)(g_mirror + 0x17FC) == 0xCCCCCCCCu);
    CHECK(*(uint32_t *)(g_mirror + 0x2000) == 0xCCCCCCCCu);
    // The CPU's own write after that page was current again is the guest's view too, and the next sync is quiet.
    uint64_t v4 = sub_begin();
    resolve(p, 0x4000, v4);
    sub_end(v4);
    CHECK(D.upload_bytes == before + PG);
    gpu_run(v4, NULL);
    backing_pool_fini(&pp);
    done("writeback_no_reupload");
}

// A write that covers its pages in part leaves the other bytes of those pages to the CPU, which may have written them
// since the mirror took its copy: such a page is stamped as before and copied in again.
static void case_writeback_partial_page(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    echo_on(p, p + 1 * MiB);
    uint64_t v1 = sub_begin();
    BackingRef r = resolve(p, 0x4000, v1);
    g_mirror = bytes(&r);
    // 0x800 bytes of page 1 to 0x800 of page 3: the middle page whole, the other two in part.
    backing_gpu_wrote(p + 0x1800, 0x2000, v1);
    sub_end(v1);
    gpu_run(v1, gpu_writes_cc_pages);   // writes 0xCC over [0x1000, 0x3000) of the mirror, which covers the declared range
    CHECK(pagegen_gpu_stamp(p + 0x2000) == pagegen_stamp(p + 0x2000));
    CHECK(pagegen_gpu_stamp(p + 0x1000) == 0 && pagegen_gpu_stamp(p + 0x3000) == 0);
    uint64_t before = D.upload_bytes;
    uint64_t v2 = sub_begin();
    resolve(p, 0x4000, v2);
    sub_end(v2);
    CHECK(D.upload_bytes == before + 2 * PG);   // the two partial pages, not the whole one
    gpu_run(v2, NULL);
    backing_pool_fini(&pp);
    done("writeback_partial_page");
}

// The mirror a retired one is replaced by is new, and takes every page from guest memory, the pages an older mirror's
// write-back left GPU-current among them: the record is of the pages, not of a mirror, and a new mirror holds none of them.
static void case_writeback_replaced(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    echo_on(p, p + 1 * MiB);
    uint64_t v1 = sub_begin();
    BackingRef r = resolve(p, 0x4000, v1);
    g_mirror = bytes(&r);
    backing_gpu_wrote(p + 0x1000, 0x2000, v1);
    sub_end(v1);
    gpu_run(v1, gpu_writes_cc_pages);
    uint64_t v2 = sub_begin();
    resolve(p, 0x4000, v2);
    sub_end(v2);
    gpu_run(v2, NULL);
    backing_pool_fini(&pp);   // the mirror is retired; a pool over the same pages makes another
    pool_t pq = pool(p, 1 * MiB);
    uint64_t v3 = sub_begin();
    BackingRef r2 = resolve(p, 0x4000, v3);
    CHECK(r2.kind == BACKING_MIRROR);
    sub_end(v3);
    gpu_run(v3, NULL);
    CHECK(words_are(bytes(&r2) + 0x1000, 0x2000, 0xCCCCCCCCu) && words_are(bytes(&r2), 0x1000, 0xAAAAAAAAu));
    CHECK(words_are(bytes(&r2) + 0x3000, 0x1000, 0xAAAAAAAAu));
    backing_pool_fini(&pq);
    done("writeback_replaced");
}

// What the record holds does not grow with the write-backs: a long run of them over the same pool keeps the page
// generations' table at its one piece, copies in nothing but what the CPU wrote, and every one of them reaches guest memory.
static uint64_t g_wr_off;
static uint32_t g_wr_word;
static void gpu_writes_rotating(void) {
    for (uint64_t i = 0; i < PG; i += 4) *(uint32_t *)(g_mirror + g_wr_off + i) = g_wr_word;
}

static void case_writeback_long_run(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    echo_on(p, p + 1 * MiB);
    PagegenStats before_stats = {0};
    uint64_t first = 0;
    for (int i = 0; i < 1000; i++) {
        uint64_t v = sub_begin();
        BackingRef r = resolve(p, 0x4000, v);
        g_mirror = bytes(&r);
        g_wr_off = (uint64_t)(i % 200) * PG;
        g_wr_word = 0x10000000u + (uint32_t)i;
        backing_gpu_wrote(p + g_wr_off, PG, v);
        sub_end(v);
        gpu_run(v, gpu_writes_rotating);
        CHECK(words_are((void *)(uintptr_t)(p + g_wr_off), PG, g_wr_word));
        if (!i) first = D.upload_bytes, pagegen_stats(&before_stats);
    }
    CHECK(first == 1 * MiB && D.upload_bytes == first);   // the first sync's megabyte and nothing since
    PagegenStats s;
    pagegen_stats(&s);
    CHECK(s.chunks == before_stats.chunks && s.gpu_marks == 1000 && s.gpu_mark_pages == 1000);
    backing_pool_fini(&pp);
    done("writeback_long_run");
}

// The page generations' record of a write-back (pagegen_mark_gpu): whole pages are recorded and have the echo settled,
// pages covered in part are stamped and left to the next query.
static void case_gpu_mark(void) {
    setup(64 * MiB, 0);
    uint64_t a = g_heap + 1 * MiB, pages;
    cpu_fill(a, 8 * PG, 1);
    echo_on(a, a + 8 * PG);
    uint64_t g1 = pagegen_scan(a, a + 8 * PG);
    // The runtime copies bytes over [a + 0x800, a + 3 * PG + 0x800): pages 1 and 2 whole, 0 and 3 in part.
    memset((void *)(uintptr_t)(a + 0x800), 0x5A, 3 * PG);
    uint64_t g = pagegen_mark_gpu(a + 0x800, a + 3 * PG + 0x800);
    CHECK(g == g1 + 1);
    for (int k = 0; k < 4; k++) CHECK(pagegen_stamp(a + (uint64_t)k * PG) == g);
    CHECK(pagegen_stamp(a + 4 * PG) == g1);
    CHECK(pagegen_gpu_stamp(a) == 0 && pagegen_gpu_stamp(a + PG) == g && pagegen_gpu_stamp(a + 2 * PG) == g &&
          pagegen_gpu_stamp(a + 3 * PG) == 0);
    // The query that follows reports the two pages the record left alone and not the two it settled.
    uint64_t g2 = pagegen_scan(a, a + 8 * PG);
    CHECK(pagegen_stamp(a) == g2 && pagegen_stamp(a + 3 * PG) == g2 && pagegen_stamp(a + PG) == g &&
          pagegen_stamp(a + 2 * PG) == g);
    runs(a, a + 8 * PG, g1, &pages);
    CHECK(pages == 4);
    Runs r = {.n = 0};
    pages = pagegen_runs_cpu(a, a + 8 * PG, g1, collect, &r);
    CHECK(pages == 2 && r.n == 2 && r.lo[0] == a && r.lo[1] == a + 3 * PG);
    // A CPU write into a recorded page moves it past the record.
    cpu_fill(a + PG + 8, 4, 7);
    uint64_t g3 = pagegen_scan(a, a + 8 * PG);
    CHECK(pagegen_stamp(a + PG) == g3 && pagegen_gpu_stamp(a + PG) == g);
    Runs r2 = {.n = 0};
    pages = pagegen_runs_cpu(a, a + 8 * PG, g2, collect, &r2);
    CHECK(pages == 1 && r2.lo[0] == a + PG);
    // A plain mark over a recorded page does too.
    pagegen_mark(a + 2 * PG, a + 2 * PG + 4);
    Runs r3 = {.n = 0};
    pages = pagegen_runs_cpu(a, a + 8 * PG, g3, collect, &r3);
    CHECK(pages == 1 && r3.lo[0] == a + 2 * PG);
    PagegenStats s;
    pagegen_stats(&s);
    CHECK(s.gpu_marks == 1 && s.gpu_mark_pages == 2);
    // The GPU window lies above the guest's space: no CPU sees it, and nothing is recorded.
    pagegen_mark_gpu(GFX_GPU_WINDOW_BASE, GFX_GPU_WINDOW_BASE + PG);
    pagegen_stats(&s);
    CHECK(s.gpu_marks == 1);
    done("gpu_mark");
}

// pagegen_mark_gpu_direct, the mark of pages the GPU wrote into guest memory itself (the raw path's sinks): nothing copied,
// so nothing to settle, and what the CPU wrote since `since` is told from the GPU's own writes. Eight pages the GPU wrote
// after the mirror's sync g1: page 2 the CPU wrote and nobody has scanned (the watch reports it), page 4 the CPU wrote and
// another consumer's scan stamped after g1, page 5 a plain mark stamped, page 6 another submission's write-back stamped
// GPU-current after g1 (not the CPU's). Pages 2, 4 and 5 keep the plain stamp; the other five are recorded GPU-current.
static void case_gpu_mark_direct(void) {
    setup(64 * MiB, 0);
    uint64_t a = g_heap + 1 * MiB, pages, cpu;
    cpu_fill(a, 8 * PG, 1);
    uint64_t g1 = pagegen_scan(a, a + 8 * PG);
    cpu_fill(a + 2 * PG + 8, 4, 7);
    cpu_fill(a + 4 * PG + 8, 4, 7);
    pagegen_scan(a + 4 * PG, a + 5 * PG);
    pagegen_mark(a + 5 * PG, a + 5 * PG + 4);
    pagegen_mark_gpu(a + 6 * PG, a + 7 * PG);
    uint64_t g = pagegen_mark_gpu_direct(a, a + 8 * PG, g1, &cpu);
    CHECK(cpu == 3);
    for (int k = 0; k < 8; k++) {
        uint64_t page = a + (uint64_t)k * PG;
        int plain = k == 2 || k == 4 || k == 5;
        CHECK(pagegen_stamp(page) == g);
        CHECK(plain ? pagegen_gpu_stamp(page) != g : pagegen_gpu_stamp(page) == g);
    }
    Runs r = {.n = 0};
    pages = pagegen_runs_cpu(a, a + 8 * PG, g - 1, collect, &r);   // what the mirror copies at its next sync
    CHECK(pages == 3 && r.n == 2 && r.lo[0] == a + 2 * PG && r.hi[0] == a + 3 * PG && r.lo[1] == a + 4 * PG &&
          r.hi[1] == a + 6 * PG);
    // The watch was asked with a reset: the next query finds nothing, and the mark has taken the CPU's page 2 for good.
    uint64_t scans = g_src_calls;
    uint64_t g2 = pagegen_scan(a, a + 8 * PG);
    CHECK(g_src_calls > scans && pagegen_stamp(a + 2 * PG) == g && g2 == g + 1);
    // A second submission that wrote them again and saw no CPU write since the first one's stamp: all eight are the GPU's.
    uint64_t since = pagegen_now();
    g = pagegen_mark_gpu_direct(a, a + 8 * PG, since, &cpu);
    CHECK(cpu == 0);
    for (int k = 0; k < 8; k++) CHECK(pagegen_gpu_stamp(a + (uint64_t)k * PG) == g);
    PagegenStats s;
    pagegen_stats(&s);
    CHECK(s.direct_marks == 2 && s.direct_cpu_pages == 3);
    // Where the watch does not cover the range nothing says which pages the CPU wrote: every page keeps the plain stamp.
    g_untracked = 1;
    g = pagegen_mark_gpu_direct(a, a + 4 * PG, pagegen_now(), &cpu);
    CHECK(cpu == 4);
    for (int k = 0; k < 4; k++) CHECK(pagegen_stamp(a + (uint64_t)k * PG) == g && pagegen_gpu_stamp(a + (uint64_t)k * PG) != g);
    done("gpu_mark_direct");
}

// A direct mark of a range that is not whole pages is a named stop.
static void case_gpu_mark_direct_partial(void) {
    setup(64 * MiB, 0);
    pagegen_mark_gpu_direct(g_heap + 0x800, g_heap + PG + 0x800, 0, NULL);
    done("gpu_mark_direct_partial");
}

static void gpu_writes_dd(void) { memset(g_mirror + 0x100, 0xDD, 0x100); }
static uint64_t g_run_later;
static void run_later(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    gpu_run(g_run_later, gpu_writes_dd);
}

static void case_replace(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 8 * MiB);
    cpu_fill(p, 64 * KiB, 0xAAAAAAAAu);
    cpu_fill(p + 128 * KiB, 64 * KiB, 0xAAAAAAAAu);
    uint64_t v1 = sub_begin();
    BackingRef r = resolve(p + 0x100, 0x100, v1);
    CHECK(r.kind == BACKING_MIRROR && stats().mirror_bytes == 64 * KiB);
    g_mirror = bytes(&r) - 0x100;
    backing_gpu_wrote(p + 0x100, 0x100, v1);
    sub_end(v1);
    // A bind across the gap joins both runs: the mirror is replaced, and the new one waits until v1 is written back.
    g_run_later = v1;
    CHECK(plat_thread_start(run_later, NULL, 256 * KiB));
    uint64_t v2 = sub_begin(), t0 = plat_now_ns();
    BackingRef q = resolve(p + 0x8000, 0x20000, v2);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    sub_end(v2);
    BackingStats s = stats();
    CHECK(q.kind == BACKING_MIRROR && s.mirrors == 1 && s.mirror_bytes == 192 * KiB && s.replace_waits == 1);
    CHECK(waited >= 150.0 && gpu_sync_done() >= v1);
    gpu_run(v2, NULL);
    uint8_t *m = bytes(&q) - 0x8000;
    CHECK(words_are(m + 0x100, 0x100, 0xDDDDDDDDu) && words_are(m + 0x200, 0x100, 0xAAAAAAAAu));
    CHECK(words_are(m + 64 * KiB, 64 * KiB, 0) && words_are(m + 128 * KiB, 64 * KiB, 0xAAAAAAAAu));
    printf("replace: the new mirror waited %.1f ms for the submission in flight\n", waited);
    backing_pool_fini(&pp);
    done("replace");
}

static void case_watch(void) {
#ifndef _WIN32
    printf("skip watch: no write watch on this platform\n");
#else
    setup(64 * MiB, 1);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB, pages;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0x11111111u);
    uint64_t v = sub_begin();
    BackingRef r = resolve(p, 0x100, v);
    sub_end(v);
    gpu_run(v, NULL);
    uint8_t *m = bytes(&r);
    CHECK(r.kind == BACKING_MIRROR && words_are(m, 1 * MiB, 0x11111111u));
    uint64_t tex_since = pagegen_now();
    *(volatile uint32_t *)(uintptr_t)(p + 0x3000) = 0x77777777u;   // a CPU write the real write watch marks
    // Every hook and sync between the write and the next submission: none of them takes the page from the mirror.
    backing_sync();
    backing_guest_read(p + 0x10000, 4);
    backing_cpu_synced(gpu_sync_done());
    backing_guest_written(p + 0x20000, 4);
    BackingPage pg;
    backing_page(p, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    stats();
    v = sub_begin();
    resolve(p, 0x100, v);
    sub_end(v);
    gpu_run(v, NULL);
    CHECK(*(uint32_t *)(m + 0x3000) == 0x77777777u);
    Runs t = runs(p, p + 1 * MiB, tex_since, &pages);
    int seen = 0;
    for (int i = 0; i < t.n; i++) seen |= t.lo[i] <= p + 0x3000 && p + 0x3000 < t.hi[i];
    CHECK(seen);
    backing_pool_fini(&pp);
    done("watch");
#endif
}

// ---------- a bind in a raw-access submission (backing.h, The raw path)
// The run of runs[0..n) that holds a (a later one wins), or NULL; the host address a raw access of guest address a reaches
// through them, 0 when no run maps it.
static const BackingRun *run_at(const BackingRun *runs, int n, uint64_t a) {
    const BackingRun *hit = NULL;
    for (int i = 0; i < n; i++)
        if (runs[i].lo <= a && a < runs[i].hi) hit = &runs[i];
    return hit;
}

static uint64_t raw_address(const BackingRun *runs, int n, uint64_t a) {
    const BackingRun *r = run_at(runs, n, a);
    return r ? r->address + (a - r->lo) : 0;
}

// The pool [p, p + 16 MiB) with 256 KiB the CPU wrote at its start, and a small bind of submission v that made the mirror
// of that run; then v holds a raw-access program, whose page table maps the mirror.
static uint64_t raw_scene(uint64_t *p, pool_t *pool_out, BackingRef *small) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    *p = g_heap + 4 * MiB;
    *pool_out = pool(*p, 16 * MiB);
    cpu_fill(*p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v = sub_begin();
    *small = resolve(*p, 0x1200, v);
    CHECK(small->kind == BACKING_MIRROR && stats().mirrors == 1 && stats().mirror_bytes == 256 * KiB);
    backing_raw_begin(v);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(n == 1 && runs[0].kind == BACKING_MIRROR && runs[0].lo == *p && runs[0].hi == *p + 256 * KiB);
    free(runs);
    return v;
}

// A mirror a raw-access submission's table maps, and a later bind of that submission that starts in the mirror's last pages
// and runs far past its end, over pages the guest never touched. It used to stop by name (the table and the bind would
// have had two copies of the pages the mirror held), and its raw accesses to what the bind committed missed the table.
// Now the mirror keeps the part the bind does not take, the bind's import covers the rest, and the page table and the
// bind reach the same memory for every page.
static void case_raw_grow(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t v = raw_scene(&p, &pp, &small);
    BackingRef big = resolve(p + 0x30320, 1 * MiB, v);
    BackingStats s = stats();
    CHECK(big.kind == BACKING_IMPORT && s.imports == 1 && s.import_bytes == 0x110000);   // up to the chunk the bind committed
    CHECK(s.mirrors == 1 && s.mirror_bytes == 0x30000 && s.replace_waits == 0);
    uint64_t lo, hi;
    CHECK(backing_pool_span(&lo, &hi) && lo == p && hi == p + 16 * MiB);

    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(n == 2 && version == backing_version());
    static const uint64_t at[] = {0x0, 0x2F000, 0x30000, 0x3F000, 0x40000, 0x130000};
    static const int want[] = {BACKING_MIRROR, BACKING_MIRROR, BACKING_IMPORT, BACKING_IMPORT, BACKING_IMPORT,
                               BACKING_IMPORT};
    for (int i = 0; i < 6; i++) {
        BackingPage pg;
        backing_page(p + at[i], &pg);
        BackingRef r = resolve(p + at[i], 4, v);
        CHECK(pg.kind == want[i] && r.kind == want[i]);
        CHECK(raw_address(runs, n, p + at[i]) == r.address);   // a raw access and a bind reach the same memory
    }
    BackingPage pg;
    backing_page(p + 0x140000, &pg);
    CHECK(pg.kind == BACKING_NONE && !raw_address(runs, n, p + 0x140000));
    free(runs);

    sub_end(v);
    gpu_run(v, NULL);
    CHECK(words_are(bytes(&small), 0x100, 0xAAAAAAAAu));   // the mirror's part, brought current in the prologue
    CHECK(words_are(bytes(&big), 0x100, 0xAAAAAAAAu));     // the import is the guest's memory: what the CPU wrote
    CHECK(words_are((const void *)(uintptr_t)(p + 0x50000), 0x1000, 0));   // and the pages the bind committed
    s = stats();
    printf("raw_grow: the mirror keeps %llu KiB, the import takes %llu KiB, no page has two backings\n",
           (ull)(s.mirror_bytes >> 10), (ull)(s.import_bytes >> 10));
    backing_pool_fini(&pp);
    done("raw_grow");
}

// A bind that covers a whole mirror the submission has not resolved takes it all: the mirror goes, to be freed once the
// raw-access submission is done.
static void case_raw_grow_whole(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p + 0x10000, 64 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    resolve(p + 0x10000, 0x100, v0);   // the mirror of those 64 KiB, in a submission that is over
    sub_end(v0);
    gpu_run(v0, NULL);
    CHECK(stats().mirrors == 1);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingRef big = resolve(p + 0xE000, 0x14000, v);   // [p + 0xE000, p + 0x22000) holds it; its chunks make 192 KiB
    BackingStats s = stats();
    CHECK(big.kind == BACKING_IMPORT && s.mirrors == 0 && s.mirror_bytes == 0 && s.imports == 1 && s.import_bytes == 0x30000);
    CHECK(s.deferred == 1 && s.freed == 0);
    sub_end(v);
    gpu_run(v, NULL);
    backing_sync();
    s = stats();
    CHECK(s.deferred == 0 && s.freed == 1 && s.mirrors == 0);
    backing_pool_fini(&pp);
    done("raw_grow_whole");
}

// A bind that starts below a mirror and ends inside it takes its first pages: the mirror keeps the rest, its device
// addresses move with its start, every page it kept still reads its own bytes, and its part of the block goes back whole.
static void case_raw_grow_head(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    for (uint64_t i = 0; i < 48; i++) cpu_fill(p + 0x10000 + i * PG, PG, 0xC0DE0000u + (uint32_t)i);   // a tag a page
    uint64_t v0 = sub_begin();
    resolve(p + 0x10000, 0x100, v0);   // the mirror over [p + 0x10000, p + 0x40000), in a submission that is over
    sub_end(v0);
    gpu_run(v0, NULL);
    CHECK(stats().mirrors == 1 && stats().mirror_bytes == 192 * KiB);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingRef big = resolve(p + 0xE000, 0xA000, v);   // [p + 0xE000, p + 0x18000): the mirror's first 8 pages
    BackingStats s = stats();
    CHECK(big.kind == BACKING_IMPORT && s.mirrors == 1 && s.mirror_bytes == 0x28000 && s.imports == 1 &&
          s.import_bytes == 0x18000);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(n == 2);
    for (uint64_t i = 8; i < 48; i += 13) {
        uint64_t a = p + 0x10000 + i * PG;
        BackingRef r = resolve(a, 4, v);
        CHECK(r.kind == BACKING_MIRROR && raw_address(runs, n, a) == r.address);
        CHECK(*(const uint32_t *)bytes(&r) == 0xC0DE0000u + (uint32_t)i);
    }
    BackingPage pg;
    backing_page(p + 0x17000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(p + 0x18000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    free(runs);
    sub_end(v);
    gpu_run(v, NULL);
    backing_pool_fini(&pp);   // the mirror's part goes back to its block at its own offset
    done("raw_grow_head");
}

// A bind beside a mirror, not over it, leaves the mirror whole: the import covers the committed pages next to it, up to it.
static void case_raw_grow_beside(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p + 0x10000, 64 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    resolve(p + 0x10000, 0x100, v0);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p + 0x20000, 64 * KiB, 0xBBBBBBBBu);   // a chunk after the mirror that nothing backs yet
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    // The pages after that chunk, which the bind commits: the import starts where the mirror ends and takes the chunk
    // between. And pages well below the mirror, with a chunk between them and it that no bind needed: the import ends
    // where the mirror starts.
    BackingRef above = resolve(p + 0x30000, 0x2000, v);
    BackingRef below = resolve(p + 0x2000, 0x2000, v);
    BackingStats s = stats();
    CHECK(above.kind == BACKING_IMPORT && below.kind == BACKING_IMPORT);
    CHECK(s.mirrors == 1 && s.mirror_bytes == 64 * KiB && s.imports == 2 && s.import_bytes == 0x30000);
    BackingPage pg;
    backing_page(p + 0x10000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(p + 0x1F000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(p + 0x20000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(p + 0x3F000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(p + 0xF000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(n == 3);   // the import below, the import above, the mirror: no page twice
    free(runs);
    sub_end(v);
    gpu_run(v, NULL);
    backing_pool_fini(&pp);
    done("raw_grow_beside");
}

// A mirror gives pages up only once the older raw-access submissions are done: their stores into it reach guest memory,
// which the import reads, when they are.
static uint64_t g_older;
static void (*g_older_work)(void);
static void run_older_later(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    gpu_run(g_older, g_older_work);
}

static void case_raw_grow_wait(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t r = raw_scene(&p, &pp, &small);   // raw-access submission r, which may store into the mirror
    sub_end(r);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    g_older = r;
    CHECK(plat_thread_start(run_older_later, NULL, 256 * KiB));
    uint64_t t0 = plat_now_ns();
    BackingRef big = resolve(p + 0x30320, 1 * MiB, v);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(big.kind == BACKING_IMPORT && waited >= 150.0 && gpu_sync_done() >= r && stats().replace_waits == 1);
    sub_end(v);
    gpu_run(v, NULL);
    printf("raw_grow_wait: the import waited %.1f ms for the raw-access submission in flight\n", waited);
    backing_pool_fini(&pp);
    done("raw_grow_wait");
}

// A mirror brought current for a raw-access submission waits for the older one in flight when the CPU wrote the mirror
// since: the older one's raw stores reach guest memory only when it is done, and the copy would put the CPU's older bytes
// over them.
static void case_raw_sync_wait(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t r = raw_scene(&p, &pp, &small);   // raw-access submission r, whose table maps the mirror
    sub_end(r);
    cpu_fill(p + 0x1000, 4, 0xDDDDDDDDu);      // the CPU writes a page of the mirror while r is in flight
    uint64_t v = sub_begin();
    g_older = r;
    CHECK(plat_thread_start(run_older_later, NULL, 256 * KiB));
    uint64_t t0 = plat_now_ns();
    backing_raw_begin(v);   // brings the mirror current, after r
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(waited >= 150.0 && gpu_sync_done() >= r && stats().raw_waits == 1);
    sub_end(v);
    gpu_run(v, NULL);
    CHECK(*(const uint32_t *)(bytes(&small) + 0x1000) == 0xDDDDDDDDu);   // the copy carried the CPU's write
    printf("raw_sync_wait: the mirror was brought current %.1f ms after the raw-access submission in flight was done\n",
           waited);
    backing_pool_fini(&pp);
    done("raw_sync_wait");
}

// The same scene, but the only page stamped since the mirror was brought current is one a write-back left GPU-current: the
// mirror holds its bytes, no copy of guest memory is made over the older submission's stores, so there is nothing to wait
// for (the wait would put the older submission's whole run on the critical path of every present).
static void case_raw_sync_nowait(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t r = raw_scene(&p, &pp, &small);   // raw-access submission r, whose table maps the mirror
    sub_end(r);
    pagegen_mark_gpu(p + 0x1000, p + 0x2000);  // a write-back landed in guest memory: the page is GPU-current
    uint64_t v = sub_begin();
    g_older = r;
    CHECK(plat_thread_start(run_older_later, NULL, 256 * KiB));
    uint64_t t0 = plat_now_ns();
    backing_raw_begin(v);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(waited < 100.0 && stats().raw_waits == 0);
    while (gpu_sync_done() < r) plat_sleep_ns(10 * 1000 * 1000);   // r runs on its own thread, 200 ms in
    sub_end(v);
    gpu_run(v, NULL);
    printf("raw_sync_nowait: the mirror was brought current in %.1f ms, with the raw-access submission still in flight\n",
           waited);
    backing_pool_fini(&pp);
    done("raw_sync_nowait");
}

// ---------- an import a raw-access submission binds a large range of becomes a mirror (backing.h, The raw path)
// The page table's invariant imports every committed page, and a bind in a raw-access submission can only import. A bind of a
// range larger than a uniform's marks the import hot; the next raw-access submission's begin gives it the mirror the rule
// would have made, brought current from guest memory, and the page table and the binds of that submission reach the mirror.
// An import that submission wrote on the GPU before its raw-access stage waits for the one after; one it only read does not.
static void case_raw_promote(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    const uint64_t A = p, B = p + 4 * MiB, C = p + 8 * MiB, E = p + 12 * MiB;   // four committed runs
    cpu_fill(A, 512 * KiB, 0xAAAAAAAAu);
    cpu_fill(B, 128 * KiB, 0xCCCCCCCCu);
    cpu_fill(C, 128 * KiB, 0xDDDDDDDDu);
    cpu_fill(E, 128 * KiB, 0xEEEEEEEEu);
    uint64_t v1 = sub_begin();
    backing_raw_begin(v1);
    BackingRef big = resolve(A + 0x1000, 256 * KiB, v1);   // large binds mark A, C and E hot; B only gets a small one
    resolve(C, 128 * KiB, v1);
    resolve(E, 128 * KiB, v1);
    BackingRef small = resolve(B + 0x1000, 0x100, v1);
    CHECK(big.kind == BACKING_IMPORT && small.kind == BACKING_IMPORT);
    BackingStats s = stats();
    CHECK(s.raw_promoted == 0 && s.mirrors == 0 && s.imports == 4);
    sub_end(v1);
    gpu_run(v1, NULL);

    cpu_fill(A + 0x1000, 0x1000, 0xBBBBBBBBu);   // the CPU writes into a hot import's range before the next submission
    // v2 reads C and writes E before its raw-access stage: C may become a mirror (what v2 recorded read the import's bytes,
    // which the mirror's first copy repeats), E may not (replacing a backing the submission wrote is a named fatal)
    uint64_t v2 = sub_begin();
    resolve(C, 128 * KiB, v2);
    resolve(E, 128 * KiB, v2);
    backing_gpu_wrote(E, 0x100, v2);
    backing_raw_begin(v2);
    s = stats();
    CHECK(s.raw_promoted == 2 && s.raw_promote_bytes == 640 * KiB && s.mirrors == 2 && s.mirror_bytes == 640 * KiB);
    CHECK(s.imports == 2);   // B, which only a small bind reached, and E, which v2 wrote
    BackingPage pg;
    backing_page(A + 0x1000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(C + 0x1000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(B + 0x1000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(E + 0x1000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    BackingRef now = resolve(A + 0x1000, 256 * KiB, v2);
    CHECK(now.kind == BACKING_MIRROR);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(version == backing_version());
    const BackingRun *r = run_at(runs, n, A + 0x1000);
    CHECK(r && r->kind == BACKING_MIRROR && raw_address(runs, n, A + 0x1000) == now.address);   // the table and the bind agree
    free(runs);
    sub_end(v2);
    gpu_run(v2, NULL);
    CHECK(words_are(bytes(&now), 0x1000, 0xBBBBBBBBu));            // brought current: what the CPU wrote since
    CHECK(words_are(bytes(&now) + 0x1000, 0x1000, 0xAAAAAAAAu));   // and the rest of what it had written before

    // the next submission promotes E, which the admission now refuses: it stays an import and the refusal is counted
    D.used = D.budget - 1 * MiB;
    uint64_t v3 = sub_begin();
    backing_raw_begin(v3);
    s = stats();
    CHECK(s.raw_promoted == 2 && s.mirrors == 2 && s.mirrors_refused_budget >= 1);
    backing_page(E + 0x1000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    sub_end(v3);
    gpu_run(v3, NULL);
    // with room again a bind marks it hot once more and the submission after promotes it
    D.used = 0;
    uint64_t v4 = sub_begin();
    backing_raw_begin(v4);
    resolve(E, 128 * KiB, v4);
    sub_end(v4);
    gpu_run(v4, NULL);
    uint64_t v5 = sub_begin();
    backing_raw_begin(v5);
    s = stats();
    CHECK(s.raw_promoted == 3 && s.mirrors == 3);
    backing_page(E + 0x1000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    sub_end(v5);
    gpu_run(v5, NULL);
    printf("raw_promote: %llu imports became mirrors (%llu KiB); the one v2 wrote, and the one the admission refused, waited\n",
           (ull)s.raw_promoted, (ull)(s.raw_promote_bytes >> 10));
    backing_pool_fini(&pp);
    done("raw_promote");
}

// A bind of a uniform's size or less never marks an import: no promotion however often it is bound.
static void case_raw_promote_small(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p, 512 * KiB, 0xAAAAAAAAu);
    for (int i = 0; i < 3; i++) {
        uint64_t v = sub_begin();
        backing_raw_begin(v);
        BackingRef a = resolve(p + 0x1000, BACKING_PROMOTE_MIN, v);   // exactly a uniform's largest range
        CHECK(a.kind == BACKING_IMPORT);
        sub_end(v);
        gpu_run(v, NULL);
    }
    BackingStats s = stats();
    CHECK(s.raw_promoted == 0 && s.mirrors == 0 && s.imports == 1);
    backing_pool_fini(&pp);
    done("raw_promote_small");
}

// The promotion waits for the older raw-access submission in flight: what its raw stores wrote through the import is in
// guest memory when it is done, and the mirror's first copy is taken from there.
static uint64_t g_through;
static void write_through_import(void) { memset((void *)(uintptr_t)g_through, 0x5A, 0x100); }
static void run_older_with_write(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    gpu_run(g_older, write_through_import);
}

static void case_raw_promote_wait(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p, 512 * KiB, 0xAAAAAAAAu);
    uint64_t v1 = sub_begin();
    backing_raw_begin(v1);
    BackingRef big = resolve(p + 0x1000, 256 * KiB, v1);
    CHECK(big.kind == BACKING_IMPORT);
    sub_end(v1);   // handed over, and the device has not reached it
    g_older = v1;
    g_through = p + 0x1000;
    uint64_t v2 = sub_begin();
    CHECK(plat_thread_start(run_older_with_write, NULL, 256 * KiB));
    uint64_t t0 = plat_now_ns();
    backing_raw_begin(v2);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(waited >= 150.0 && gpu_sync_done() >= v1 && stats().replace_waits >= 1 && stats().raw_promoted == 1);
    BackingRef now = resolve(p + 0x1000, 256 * KiB, v2);
    CHECK(now.kind == BACKING_MIRROR);
    sub_end(v2);
    gpu_run(v2, NULL);
    CHECK(words_are(bytes(&now), 0x100, 0x5A5A5A5Au));              // the write v1 made through the import is in the copy
    CHECK(words_are(bytes(&now) + 0x100, 0x100, 0xAAAAAAAAu));
    printf("raw_promote_wait: the promotion waited %.1f ms for the raw-access submission in flight\n", waited);
    backing_pool_fini(&pp);
    done("raw_promote_wait");
}

// A promotion that would take the mirrors past BACKING_PROMOTE_MIRROR_MAX leaves its import as it was: the raw path's
// write-back and the sinks under the mirrors are bounded by it.
static void case_raw_promote_cap(void) {
    setup(512 * MiB, 0);
    backing_set_mirror_hook(want_all);
    const uint64_t win = BACKING_IMPORT_WINDOW, len = 60 * MiB;
    uint64_t base = (g_heap + win - 1) & ~(win - 1);   // each run starts a window of its own: one import each
    const uint64_t run[3] = {base, base + 2 * win, base + 4 * win};
    CHECK(3 * len > BACKING_PROMOTE_MIRROR_MAX && 2 * len <= BACKING_PROMOTE_MIRROR_MAX);
    pool_t pp = pool(base, 5 * win);
    for (int i = 0; i < 3; i++) cpu_fill(run[i], len, 0xA0A0A0A0u + (uint32_t)i);
    uint64_t v1 = sub_begin();
    backing_raw_begin(v1);
    for (int i = 0; i < 3; i++) CHECK(resolve(run[i] + 0x1000, 256 * KiB, v1).kind == BACKING_IMPORT);   // each marked hot
    sub_end(v1);
    gpu_run(v1, NULL);
    uint64_t v2 = sub_begin();
    backing_raw_begin(v2);
    BackingStats s = stats();
    int mirrors = 0, imports = 0;
    for (int i = 0; i < 3; i++) {
        BackingPage pg;
        backing_page(run[i] + 0x1000, &pg);
        mirrors += pg.kind == BACKING_MIRROR, imports += pg.kind == BACKING_IMPORT;
    }
    CHECK(mirrors == 2 && imports == 1);   // two fit under the cap together; the third would pass it
    CHECK(s.raw_promoted == 2 && s.raw_promote_bytes == 2 * len && s.mirrors == 2 && s.mirror_bytes == 2 * len);
    CHECK(s.mirrors_refused_budget == 0);   // the cap refused it, not the device's budget
    sub_end(v2);
    gpu_run(v2, NULL);
    for (int i = 0; i < 2; i++) {   // the refused one is bound again, marked hot again, and refused again at every begin
        uint64_t v = sub_begin();
        backing_raw_begin(v);
        for (int k = 0; k < 3; k++) resolve(run[k] + 0x1000, 256 * KiB, v);
        sub_end(v);
        gpu_run(v, NULL);
    }
    s = stats();
    CHECK(s.raw_promoted == 2 && s.mirrors == 2 && s.mirror_bytes == 2 * len);
    printf("raw_promote_cap: %llu MiB of mirrors, the third %llu MiB run stayed an import\n", (ull)(s.mirror_bytes >> 20),
           (ull)(len >> 20));
    backing_pool_fini(&pp);
    done("raw_promote_cap");
}

static BackingRef g_taken_old, g_taken_big;
static uint64_t g_taken_raw, g_taken_fresh;

static void gpu_grow_taken(void) {
    uint32_t *old = (uint32_t *)bytes(&g_taken_old);
    uint32_t *big = (uint32_t *)(bytes(&g_taken_big) + 0x31000 - 0x30320);
    uint32_t *raw = (uint32_t *)(uintptr_t)g_taken_raw;
    CHECK(*old == 0xAAAAAAAAu && *big == 0xAAAAAAAAu && *raw == 0xAAAAAAAAu);
    CHECK(*(uint32_t *)(bytes(&g_taken_big) + g_taken_fresh - 0x30320) == 0xBBBBBBBBu);
    *old = 0xCCCCCCCCu;
    CHECK(*big == 0xCCCCCCCCu && *raw == 0xCCCCCCCCu);
    *raw = 0xDDDDDDDDu;
    CHECK(*old == 0xDDDDDDDDu && *big == 0xDDDDDDDDu);
}

static void finish_grow_taken(uint64_t p, pool_t *pp, uint64_t v, const char *name) {
    backing_gpu_wrote(p + 0x31000, 4, v);   // a write declared before the mirror grows
    g_taken_big = resolve(p + 0x30320, 1 * MiB, v);
    CHECK(g_taken_old.kind == BACKING_MIRROR && g_taken_big.kind == BACKING_MIRROR);
    CHECK(g_taken_big.address + 0x31000 - 0x30320 == g_taken_old.address);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_taken_raw = raw_address(runs, n, p + 0x31000);
    CHECK(g_taken_raw == g_taken_old.address && version == backing_version());
    for (uint64_t a = p; a < p + 0x140000; a += PG) {
        int committed;
        uint64_t len = rt_mem_commit_run(a, PG, &committed);
        CHECK(len != UINT64_MAX && len);
        if (committed) CHECK(raw_address(runs, n, a));   // the whole committed chunk, not only the bound pages
    }
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_taken);
    CHECK(*(uint32_t *)(uintptr_t)(p + 0x31000) == 0xDDDDDDDDu);
    cpu_fill(p + 0x31000, 4, 0xEEEEEEEEu);
    cpu_fill(p + g_taken_fresh, 4, 0xFFFFFFFFu);
    uint64_t next = sub_begin();
    BackingRef old = resolve(p + 0x31000, 4, next);
    BackingRef fresh = resolve(p + g_taken_fresh, 4, next);
    backing_raw_begin(next);
    sub_end(next);
    gpu_run(next, NULL);
    CHECK(old.kind == BACKING_MIRROR && fresh.kind == BACKING_MIRROR);
    CHECK(*(uint32_t *)bytes(&old) == 0xEEEEEEEEu && *(uint32_t *)bytes(&fresh) == 0xFFFFFFFFu);
    backing_pool_fini(pp);
    done(name);
}

// An earlier bind reached pages a later, larger bind also needs. The earlier bind, the larger bind and raw accesses
// must all keep the same device bytes, including stores made before the submission completes.
static void case_raw_grow_taken(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t v = raw_scene(&p, &pp, &small);
    g_taken_old = resolve(p + 0x31000, 0x100, v);
    g_taken_fresh = 0x50000;
    cpu_fill(p + g_taken_fresh, 4, 0xBBBBBBBBu);
    finish_grow_taken(p, &pp, v, "raw_grow_taken");
}

static void case_raw_grow_neighbour(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    g_taken_fresh = 0x80000;
    cpu_fill(p + g_taken_fresh, 4, 0xBBBBBBBBu);
    uint64_t v0 = sub_begin();
    resolve(p, 0x100, v0);
    sub_end(v0);
    gpu_run(v0, NULL);
    uint64_t v = sub_begin();
    backing_raw_begin(v);   // the committed neighbour becomes an import, but this submission has not bound it
    BackingPage pg;
    backing_page(p + g_taken_fresh, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    g_taken_old = resolve(p + 0x31000, 0x100, v);
    finish_grow_taken(p, &pp, v, "raw_grow_neighbour");
}

static uint64_t g_older_import;
static void gpu_write_older_import(void) {
    *(uint32_t *)(uintptr_t)g_older_import = 0xBBBBBBBBu;
}

static void case_raw_grow_taken_wait(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    g_taken_fresh = 0x80000;
    cpu_fill(p + g_taken_fresh, 4, 0x11111111u);
    uint64_t v0 = sub_begin();
    resolve(p, 0x100, v0);
    sub_end(v0);
    gpu_run(v0, NULL);
    uint64_t r = sub_begin();
    backing_raw_begin(r);
    CHECK(resolve(p + g_taken_fresh, 4, r).kind == BACKING_IMPORT);
    sub_end(r);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    g_taken_old = resolve(p + 0x31000, 0x100, v);
    g_older = r, g_older_import = p + g_taken_fresh, g_older_work = gpu_write_older_import;
    CHECK(plat_thread_start(run_older_later, NULL, 256 * KiB));
    uint64_t t0 = plat_now_ns();
    finish_grow_taken(p, &pp, v, "raw_grow_taken_wait");
    CHECK((double)(plat_now_ns() - t0) / 1e6 >= 150.0);
}

static BackingRef g_taken_head;
static void gpu_grow_taken_head(void) {
    *(uint32_t *)bytes(&g_taken_head) = 0xCCCCCCCCu;
}

static void case_raw_grow_taken_head(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB, q = g_heap + 4 * MiB;
    pool_t pp = pool(p - 1 * MiB, 16 * MiB), spare = pool(q, 1 * MiB);
    cpu_fill(q, 1 * MiB, 0x11111111u);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(q, 4, v0).kind == BACKING_MIRROR);
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    backing_pool_fini(&spare);
    backing_sync();   // the free block space precedes the remaining mirror
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingRef old = resolve(p + 0x31000, 4, v);
    g_taken_head = old;
    backing_gpu_wrote(p + 0x31000, 4, v);
    BackingRef big = resolve(p - 0x8000, 0x3A000, v);
    CHECK(old.kind == BACKING_MIRROR && big.kind == BACKING_MIRROR);
    CHECK(big.address + 0x39000 == old.address);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(raw_address(runs, n, p + 0x31000) == old.address);
    for (uint64_t a = p - 0x10000; a < p; a += PG) {
        int committed;
        CHECK(rt_mem_commit_run(a, PG, &committed) != UINT64_MAX);
        if (committed) CHECK(raw_address(runs, n, a));
    }
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_taken_head);
    CHECK(*(uint32_t *)bytes(&old) == 0xCCCCCCCCu);
    CHECK(*(uint32_t *)(uintptr_t)(p + 0x31000) == 0xCCCCCCCCu);
    CHECK(*(uint32_t *)bytes(&big) == 0);
    backing_pool_fini(&pp);
    done("raw_grow_taken_head");
}

static BackingRef g_trimmed_old, g_trimmed_big;
static uint64_t g_trimmed_raw;
static void gpu_grow_trimmed(void) {
    uint32_t *old = (uint32_t *)bytes(&g_trimmed_old);
    uint32_t *big = (uint32_t *)(bytes(&g_trimmed_big) + 0x1000);
    uint32_t *raw = (uint32_t *)(uintptr_t)g_trimmed_raw;
    CHECK(*old == 0xAAAAAAAAu && *big == 0xAAAAAAAAu && *raw == 0xAAAAAAAAu);
    CHECK(*(uint32_t *)(bytes(&g_trimmed_big) + 0x10000) == 0xBBBBBBBBu);
    *old = 0xCCCCCCCCu;
    CHECK(*big == 0xCCCCCCCCu && *raw == 0xCCCCCCCCu);
    *raw = 0xDDDDDDDDu;
}

static void case_raw_grow_trimmed(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 16 * MiB);
    cpu_fill(p, 64 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    resolve(p, 4, v0);
    backing_raw_begin(v0);
    CHECK(resolve(p + 0xE100, 0x3000, v0).kind == BACKING_IMPORT);
    sub_end(v0);
    gpu_run(v0, NULL);
    CHECK(stats().mirror_bytes == 0xE000);   // a mirror trimmed by the preceding submission
    cpu_fill(p + 0x18000, 4, 0xBBBBBBBBu);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    g_trimmed_old = resolve(p + 0x9000, 0x80, v);
    backing_gpu_wrote(p + 0x9000, 4, v);
    g_trimmed_big = resolve(p + 0x8000, 0x20000, v);
    CHECK(g_trimmed_old.kind == BACKING_MIRROR && g_trimmed_big.kind == BACKING_MIRROR);
    CHECK(g_trimmed_big.address + 0x1000 == g_trimmed_old.address);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_trimmed_raw = raw_address(runs, n, p + 0x9000);
    CHECK(g_trimmed_raw == g_trimmed_old.address);
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_trimmed);
    CHECK(*(uint32_t *)(uintptr_t)(p + 0x9000) == 0xDDDDDDDDu);
    backing_pool_fini(&pp);
    done("raw_grow_trimmed");
}

static BackingRef g_guard_old, g_guard_big, g_guard_extent;
static uint64_t g_guard_raw, g_guard_added, g_guard_old_delta;

static void gpu_grow_guarded(void) {
    uint32_t *old = (uint32_t *)bytes(&g_guard_old);
    uint32_t *big = (uint32_t *)(bytes(&g_guard_big) + g_guard_old_delta);
    uint32_t *raw = (uint32_t *)(uintptr_t)g_guard_raw;
    uint32_t *added = (uint32_t *)(uintptr_t)g_guard_added;
    uint32_t *extent = (uint32_t *)bytes(&g_guard_extent);
    CHECK(*old == 0xAAAAAAAAu && *big == 0xAAAAAAAAu && *raw == 0xAAAAAAAAu);
    CHECK(*added == 0xBBBBBBBBu);
    *extent = 0xEEEEEEEEu;
    *old = 0xCCCCCCCCu;
    CHECK(*big == 0xCCCCCCCCu && *raw == 0xCCCCCCCCu);
    *raw = 0xDDDDDDDDu;
}

static void raw_grow_guarded(int head) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB, q = g_heap + 32 * MiB;
    pool_t pp = pool(p - 1 * MiB, 16 * MiB);
    pool_t extent = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS,
                     .mem = (uint8_t *)(uintptr_t)q, .size = 64 * KiB, .game_mem = (uint8_t *)(uintptr_t)q};
    CHECK(backing_pool_init(&extent));
    cpu_fill(p, 64 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    if (head) CHECK(resolve(q, 4, v0).kind == BACKING_EXTENT);
    CHECK(resolve(p + 0x2000, 4, v0).kind == BACKING_MIRROR);
    if (!head) CHECK(resolve(q, 4, v0).kind == BACKING_EXTENT);
    sub_end(v0);
    gpu_run(v0, NULL);

    uint64_t added = head ? p - 0x8000 : p + 0x18000;
    cpu_fill(added, 4, 0xBBBBBBBBu);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    g_guard_extent = resolve(q, 4, v);
    g_guard_old = resolve(p + 0x2000, 4, v);
    backing_gpu_wrote(p + 0x2000, 4, v);
    uint64_t start = head ? p - 0x5000 : p + 0x1000;
    g_guard_big = resolve(start, head ? 0x8000 : 0x11000, v);
    g_guard_old_delta = p + 0x2000 - start;
    CHECK(g_guard_old.kind == BACKING_MIRROR && g_guard_big.kind == BACKING_MIRROR);
    CHECK(g_guard_big.address + g_guard_old_delta == g_guard_old.address);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_guard_raw = raw_address(runs, n, p + 0x2000);
    g_guard_added = raw_address(runs, n, added);
    CHECK(g_guard_raw == g_guard_old.address && g_guard_added && version == backing_version());
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_guarded);
    CHECK(*(uint32_t *)(uintptr_t)(p + 0x2000) == 0xDDDDDDDDu);
    CHECK(*(uint32_t *)bytes(&g_guard_extent) == 0xEEEEEEEEu);
    backing_pool_fini(&pp), backing_pool_fini(&extent);
    done(head ? "raw_grow_guarded_head" : "raw_grow_guarded_tail");
}

static void case_raw_grow_guarded_head(void) { raw_grow_guarded(1); }
static void case_raw_grow_guarded_tail(void) { raw_grow_guarded(0); }

// The layout of the stop that a run of the Forest scene met in a bind that took pages of a mirror its submission had
// already reached: a mirror of 704 KiB (700 KiB written by the CPU, committed in 64 KiB chunks), a bind of 0xC00 bytes at
// 0x3E00 that crosses the page boundary at 0x4000, then a bind that starts two pages below the mirror and ends at that
// boundary, so what it takes of the mirror ([0, 0x4000)) overlaps what the first bind reached ([0x3E00, 0x4A00)). The
// mirror grows two pages at its start, keeps the first bind's bytes where they are, and the page table and both binds
// reach the same memory.
#define LAYOUT_WRITTEN 0xAF000ull
#define LAYOUT_MIRROR 0xB0000ull
#define LAYOUT_BOUND 0x3E00ull
static BackingRef g_layout_old, g_layout_big;
static uint64_t g_layout_raw, g_layout_delta;

static void gpu_grow_layout(void) {
    uint32_t *old = (uint32_t *)bytes(&g_layout_old);
    uint32_t *big = (uint32_t *)(bytes(&g_layout_big) + g_layout_delta);
    uint32_t *raw = (uint32_t *)(uintptr_t)g_layout_raw;
    CHECK(*old == 0xAAAAAAAAu && *big == 0xAAAAAAAAu && *raw == 0xAAAAAAAAu);
    *old = 0xCCCCCCCCu;   // a store through the earlier bind reaches both of the others
    CHECK(*big == 0xCCCCCCCCu && *raw == 0xCCCCCCCCu);
    *raw = 0xDDDDDDDDu;   // and a raw store reaches the binds
    CHECK(*old == 0xDDDDDDDDu && *big == 0xDDDDDDDDu);
}

static void case_raw_grow_taken_layout(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB;
    pool_t pp = pool(p - 1 * MiB, 16 * MiB);
    cpu_fill(p, LAYOUT_WRITTEN, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    CHECK(stats().mirrors == 1 && stats().mirror_bytes == LAYOUT_MIRROR);

    uint64_t v = sub_begin();
    backing_raw_begin(v);
    g_layout_old = resolve(p + LAYOUT_BOUND, 0xC00, v);   // [p + 0x3E00, p + 0x4A00)
    CHECK(g_layout_old.kind == BACKING_MIRROR);
    backing_gpu_wrote(p + LAYOUT_BOUND, 4, v);   // a write declared before the mirror grows
    uint64_t start = p - 0x1800;
    g_layout_big = resolve(start, 0x5800, v);   // pages [p - 0x2000, p + 0x4000)
    g_layout_delta = p + LAYOUT_BOUND - start;
    CHECK(g_layout_big.kind == BACKING_MIRROR && g_layout_big.address + g_layout_delta == g_layout_old.address);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.mirrors_made == 1 && s.mirror_bytes == LAYOUT_MIRROR + 0x2000);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_layout_raw = raw_address(runs, n, p + LAYOUT_BOUND);
    CHECK(g_layout_raw == g_layout_old.address && version == backing_version());
    for (uint64_t a = p - 0x2000; a < p + LAYOUT_MIRROR; a += PG)   // the pages the mirror took and kept: one copy
        CHECK(raw_address(runs, n, a) == g_layout_big.address + (a - start));
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_layout);
    CHECK(*(uint32_t *)(uintptr_t)(p + LAYOUT_BOUND) == 0xDDDDDDDDu);   // written back to where it was declared

    cpu_fill(p - 0x1000, 4, 0xFFFFFFFFu);   // a CPU write to a page the mirror took, copied in by the next submission
    uint64_t next = sub_begin();
    BackingRef head = resolve(p - 0x1000, 4, next);
    BackingRef tail = resolve(p + LAYOUT_WRITTEN - 4, 4, next);
    sub_end(next);
    gpu_run(next, NULL);
    CHECK(head.kind == BACKING_MIRROR && tail.kind == BACKING_MIRROR);
    CHECK(*(uint32_t *)bytes(&head) == 0xFFFFFFFFu && *(uint32_t *)bytes(&tail) == 0xAAAAAAAAu);
    backing_pool_fini(&pp);
    done("raw_grow_taken_layout");
}

static int runs_cover(const BackingRun *runs, int n, uint64_t a) {
    int k = 0;
    for (int i = 0; i < n; i++) k += runs[i].lo <= a && a < runs[i].hi;
    return k;
}

// A growth is held to its own size (BACKING_MIRROR_GROW_MAX), not to a total of the mirrors: the mirrors a bind makes are
// not held to the promotions' cap (only a promotion is), so a run can hold more than that in mirrors and still need
// one of them to grow by two pages. Three mirrors of 44 MiB, one in each window (132 MiB, over the 128 MiB), then the
// layout of the stop of the Forest run grows the first.
static void case_raw_grow_over_total(void) {
    setup(256 * MiB, 0);
    backing_set_mirror_hook(want_all);
    const uint64_t win = BACKING_IMPORT_WINDOW, len = 44 * MiB;
    CHECK(3 * len > BACKING_PROMOTE_MIRROR_MAX && (g_heap & (win - 1)) == 0);
    const uint64_t run[3] = {g_heap + 1 * MiB, g_heap + win + 1 * MiB, g_heap + 2 * win + 1 * MiB};
    const uint64_t p = run[0];
    pool_t pp = pool(g_heap, 3 * win);
    for (int i = 0; i < 3; i++) cpu_fill(run[i], len, i ? 0xA0A0A0A0u + (uint32_t)i : 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    for (int i = 0; i < 3; i++) CHECK(resolve(run[i], 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    CHECK(stats().mirrors == 3 && stats().mirror_bytes == 3 * len && stats().mirror_bytes > BACKING_PROMOTE_MIRROR_MAX);

    uint64_t v = sub_begin();
    backing_raw_begin(v);
    g_layout_old = resolve(p + LAYOUT_BOUND, 0xC00, v);
    CHECK(g_layout_old.kind == BACKING_MIRROR);
    backing_gpu_wrote(p + LAYOUT_BOUND, 4, v);
    uint64_t start = p - 0x1800;
    g_layout_big = resolve(start, 0x5800, v);   // pages [p - 0x2000, p + 0x4000): two pages below the first mirror
    g_layout_delta = p + LAYOUT_BOUND - start;
    CHECK(g_layout_big.kind == BACKING_MIRROR && g_layout_big.address + g_layout_delta == g_layout_old.address);
    BackingStats s = stats();
    CHECK(s.mirrors == 3 && s.mirrors_made == 3 && s.mirror_bytes == 3 * len + 0x2000);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_layout_raw = raw_address(runs, n, p + LAYOUT_BOUND);
    CHECK(g_layout_raw == g_layout_old.address && version == backing_version());
    for (uint64_t a = p - 0x2000; a < p + 0x10000; a += PG)
        CHECK(raw_address(runs, n, a) == g_layout_big.address + (a - start) && runs_cover(runs, n, a) == 1);
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_layout);
    CHECK(*(uint32_t *)(uintptr_t)(p + LAYOUT_BOUND) == 0xDDDDDDDDu);   // written back to where it was declared
    backing_pool_fini(&pp);
    done("raw_grow_over_total");
}

static void gpu_grow_head_import(void) {
    gpu_grow_layout();
    CHECK(*(uint32_t *)bytes(&g_layout_big) == 0xBBBBBBBBu);   // a page the mirror took was copied in from guest memory
}

// The pages a growth takes from an import are the bind's, not the import's: an import of 128 KiB abuts the start of the
// mirror, more than the 64 KiB its allocation reserves there, and a bind two pages into it needs two. The mirror takes
// those two pages (and copies only those in), the import goes, and the pages of it outside the bind get an import again.
static void case_raw_grow_head_import(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB;
    pool_t pp = pool(p - 1 * MiB, 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p - 0x20000, 0x20000, 0xBBBBBBBBu);   // two chunks below the mirror, committed since: an import at the begin
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingStats s = stats();
    BackingPage pg;
    backing_page(p - 0x20000, &pg);
    CHECK(s.mirrors == 1 && s.imports == 1 && s.import_bytes == 0x20000 && pg.kind == BACKING_IMPORT);

    g_layout_old = resolve(p + LAYOUT_BOUND, 0xC00, v);
    CHECK(g_layout_old.kind == BACKING_MIRROR);
    backing_gpu_wrote(p + LAYOUT_BOUND, 4, v);
    uint64_t start = p - 0x1800, uploaded = D.upload_bytes;
    g_layout_big = resolve(start, 0x5800, v);   // pages [p - 0x2000, p + 0x4000)
    g_layout_delta = p + LAYOUT_BOUND - start;
    CHECK(D.upload_bytes - uploaded == 0x2000);   // the two pages of the bind, not the import's 128 KiB
    CHECK(g_layout_big.kind == BACKING_MIRROR && g_layout_big.address + g_layout_delta == g_layout_old.address);
    s = stats();
    CHECK(s.mirrors == 1 && s.mirror_bytes == 256 * KiB + 0x2000);
    CHECK(s.imports == 1 && s.import_bytes == 0x1E000);   // the import's pages below the bind, imported again
    backing_page(p - 0x20000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(p - 0x3000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(p - 0x2000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_layout_raw = raw_address(runs, n, p + LAYOUT_BOUND);
    CHECK(g_layout_raw == g_layout_old.address && version == backing_version());
    for (uint64_t a = p - 0x20000; a < p + 0x10000; a += PG) CHECK(runs_cover(runs, n, a) == 1);   // no page twice
    CHECK(raw_address(runs, n, p - 0x20000) == p - 0x20000 && raw_address(runs, n, p - 0x2000) == g_layout_big.address - 0x800);
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_head_import);
    CHECK(*(uint32_t *)(uintptr_t)(p + LAYOUT_BOUND) == 0xDDDDDDDDu);
    backing_pool_fini(&pp);
    done("raw_grow_head_import");
}

// The same at the other end: a bind of 20 KiB reaches five pages into an import of 16 MiB that abuts the mirror's end.
// The mirror grows by five pages and copies in five pages, and the import's other pages keep an import.
static void case_raw_grow_tail_import(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 24 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p + 256 * KiB, 16 * MiB, 0xBBBBBBBBu);   // committed since: an import at the begin
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    CHECK(stats().mirrors == 1 && stats().imports == 1 && stats().import_bytes == 16 * MiB);

    g_taken_old = resolve(p + 0x31000, 0x100, v);
    CHECK(g_taken_old.kind == BACKING_MIRROR);
    backing_gpu_wrote(p + 0x31000, 4, v);
    uint64_t uploaded = D.upload_bytes;
    g_taken_big = resolve(p + 0x30320, 0x14CE0, v);   // pages [p + 0x30000, p + 0x45000)
    CHECK(D.upload_bytes - uploaded == 0x5000);   // the five pages of the bind, not the import's 16 MiB
    CHECK(g_taken_big.kind == BACKING_MIRROR && g_taken_big.address + 0x31000 - 0x30320 == g_taken_old.address);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.mirror_bytes == 0x45000);
    CHECK(s.imports == 1 && s.import_bytes == 16 * MiB - 0x5000);   // the import's pages beyond the bind, imported again
    BackingPage pg;
    backing_page(p + 0x44000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(p + 0x45000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    backing_page(p + 256 * KiB + 16 * MiB - PG, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    g_taken_raw = raw_address(runs, n, p + 0x31000);
    g_taken_fresh = 0x40000;
    CHECK(g_taken_raw == g_taken_old.address && version == backing_version());
    for (uint64_t a = p; a < p + 256 * KiB + 16 * MiB; a += 16 * PG) CHECK(runs_cover(runs, n, a) == 1);   // no page twice
    CHECK(raw_address(runs, n, p + 0x45000) == p + 0x45000);   // the rest of the import still aliases guest memory
    free(runs);
    sub_end(v);
    gpu_run(v, gpu_grow_taken);
    CHECK(*(uint32_t *)(uintptr_t)(p + 0x31000) == 0xDDDDDDDDu);
    backing_pool_fini(&pp);
    done("raw_grow_tail_import");
}

// ---------- cases that stop by name
static void case_raw_grow_bound_import(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool(p, 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    cpu_fill(p + 0x80000, 4, 0xBBBBBBBBu);
    uint64_t v0 = sub_begin();
    resolve(p, 4, v0);
    sub_end(v0);
    gpu_run(v0, NULL);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    resolve(p + 0x31000, 4, v);
    CHECK(resolve(p + 0x80000, 4, v).kind == BACKING_IMPORT);
    resolve(p + 0x30320, 1 * MiB, v);
    printf("returned\n");
}

static void case_raw_grow_fragmented(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB, q = g_heap + 24 * MiB;
    pool(p, 16 * MiB);
    pool(q, 1 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    cpu_fill(q, 64 * KiB, 0xBBBBBBBBu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    CHECK(resolve(q, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    resolve(p + 0x31000, 4, v);
    resolve(p + 0x30320, 1 * MiB, v);
    printf("returned\n");
}

static void case_raw_grow_guarded_blocked(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB, q = g_heap + 32 * MiB;
    pool(p, 16 * MiB);
    pool_t extent = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS,
                     .mem = (uint8_t *)(uintptr_t)q, .size = 64 * KiB, .game_mem = (uint8_t *)(uintptr_t)q};
    CHECK(backing_pool_init(&extent));
    cpu_fill(p, 64 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    CHECK(resolve(q, 4, v0).kind == BACKING_EXTENT);
    sub_end(v0);
    gpu_run(v0, NULL);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    resolve(q, 4, v);
    resolve(p + 0x2000, 4, v);
    resolve(p + 0x1000, 0x21000, v);   // grows beyond the reserved part into the used extent
    printf("returned\n");
}

static void case_raw_grow_untracked(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t v = raw_scene(&p, &pp, &small);
    resolve(p + 0x31000, 4, v);
    g_untracked = 1;
    resolve(p + 0x30320, 1 * MiB, v);
    printf("returned\n");
}

// A growth adds at most BACKING_MIRROR_GROW_MAX: a bind that reaches 2 MiB past a mirror is not the few pages it is for.
static void case_raw_grow_cap(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t v = raw_scene(&p, &pp, &small);
    resolve(p + 0x31000, 4, v);
    resolve(p + 0x30320, 2 * MiB, v);
    printf("returned\n");
}

static void case_unwritten(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0);
    uint64_t v = sub_begin();
    resolve(p, 0x100, v);
    backing_gpu_wrote(p, 0x100, v);
    gpu_sync_end();   // without backing_submit_end: nothing reads the write back
    D.recording = 0;
    gpu_run(v, NULL);
    backing_cpu_synced(v);
    printf("returned\n");
}

static void case_replace_written(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool(p, 8 * MiB);
    cpu_fill(p, 64 * KiB, 0);
    cpu_fill(p + 128 * KiB, 64 * KiB, 0);
    uint64_t v = sub_begin();
    resolve(p + 0x100, 0x100, v);
    backing_gpu_wrote(p + 0x100, 0x100, v);
    resolve(p + 0x8000, 0x20000, v);   // the same submission grows the run over the mirror it wrote
    printf("returned\n");
}

// ---------- the dump that precedes the stop of a bind over a mirror an earlier bind of the submission reached
// An import abuts the start of a mirror (as in raw_grow_head_import). The submission binds the mirror (once before its
// raw-access begin, once after) and the import (a small bind far from the pages the growth would take), then a bind across
// both takes pages of each: the mirror cannot grow over the pages of an import the submission bound, so it stops by name
// as before, and the log before the stop says what the stop was made of. Driven by tests/test_runtime_coherence.py, which
// reads the key=value line printed here against the dump.
static void diag_scene(const char *name) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB;
    pool(p - 1 * MiB, 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p - 0x20000, 0x20000, 0xBBBBBBBBu);   // committed below the mirror since: an import at the begin
    uint64_t v = sub_begin();
    resolve(p + 0x3000, 4, v);   // the mirror, before the submission's raw-access begin
    backing_raw_begin(v);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.imports == 1 && s.import_bytes == 0x20000);
    resolve(p + 0x100, 4, v);
    resolve(p - 0x10000, 4, v);   // the import, far from the pages the growth would take
    printf("%s: p=%#llx v0=%llu v=%llu bind_addr=%#llx bind_size=%#llx\n", name, (ull)p, (ull)v0, (ull)v,
           (ull)(p - 0x1800), (ull)0x5800);
    resolve(p - 0x1800, 0x5800, v);   // pages [p - 0x2000, p + 0x4000): across the import and the mirror
    printf("returned\n");
}

static void case_raw_diag_hull(void) { diag_scene("raw_diag_hull"); }
static void case_raw_diag_dry(void) { diag_scene("raw_diag_dry"); }

// The birth of the backings the dump lists: a mirror a promotion made (P), one a bind made and a later bind cut (T), one
// grown in place (M) and the import that was re-made beside it, then the stop.
static void case_raw_diag_birth(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB, T = p + 0x80000, P = p + 0x100000;
    pool(p - 1 * MiB, 16 * MiB);
    CHECK((p - 0x20000) / BACKING_IMPORT_WINDOW == (P + 0x20000) / BACKING_IMPORT_WINDOW);   // one cell
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    cpu_fill(T, 256 * KiB, 0xCCCCCCCCu);
    cpu_fill(P, 128 * KiB, 0xDDDDDDDDu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR && resolve(T, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p - 0x20000, 0x20000, 0xBBBBBBBBu);
    uint64_t v1 = sub_begin();
    backing_raw_begin(v1);
    CHECK(resolve(P, 128 * KiB, v1).kind == BACKING_IMPORT);   // larger than a uniform's: hot
    sub_end(v1);
    gpu_run(v1, NULL);
    uint64_t v = sub_begin();
    backing_raw_begin(v);   // the promotion
    CHECK(stats().raw_promoted == 1);
    resolve(T + 0x3F000, 0x20000, v);   // starts in the last page of T's mirror: it keeps its start
    resolve(p + 0x3E00, 0xC00, v);
    CHECK(resolve(p - 0x1800, 0x5800, v).kind == BACKING_MIRROR);   // the mirror of p grows over two pages
    resolve(p - 0x10000, 4, v);
    printf("raw_diag_birth: p=%#llx T=%#llx P=%#llx v0=%llu v1=%llu v=%llu\n", (ull)p, (ull)T, (ull)P, (ull)v0,
           (ull)v1, (ull)v);
    resolve(p - 0x2800, 0x1000, v);   // across the import's new end and the mirror's new start
    printf("returned\n");
}

// ---------- the events a stop prints, and the dry evaluation of the growth a bind's cut makes unnecessary
// The shape of the stop the events are for, with invented addresses. A mirror M = [p, p + 256 KiB) was made by a bind of an
// earlier submission; the pages below it were committed since, so the raw-access begin gives them an import I0 (fill).
// In the raw-access submission v, a first bind B0 = [p - 0x800, p + 0x800) is no hit (it starts in I0 and ends in M), and
// the submission has not resolved M: the bind cuts M's first page off and an import I takes its place over I0 and the
// page. Then M is resolved (and I, so that it has been used), and a bind across I and M's new start takes pages of M the
// submission resolved: M cannot grow over the import it bound, and the run stops by name. Driven by
// tests/test_runtime_coherence.py, which reads the key=value lines printed here against the dump. churn: binds far from
// all this, each a miss with its own import, which fill the event ring (a stop after more events than it holds says how
// many it lost); hits: resolves of M after B0 (more than the ring of binds holds pushes B0 out of that one).
enum { T_KIND, T_G1, T_G2, T_G3, T_G4, T_G5, T_G6, T_G7A, T_G7B, T_G8, T_G9, T_CAP, T_G4_OFF, T_N };   // BackingStats raw_dry_g

static uint64_t ev_prefix_below(uint64_t *p_out, uint64_t *v0_out, int hit_i0, uint64_t below) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB;
    pool(p - (below > 1 * MiB ? below : 1 * MiB), 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p - below, below, 0xBBBBBBBBu);   // committed below the mirror since: an import at the begin
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.imports == 1 && s.import_bytes == below && s.raw_dry_evals == 0);
    if (hit_i0) CHECK(resolve(p - 0x10000, 4, v).kind == BACKING_IMPORT);   // I0 is used by the submission
    *p_out = p, *v0_out = v0;
    return v;
}

static uint64_t ev_prefix(uint64_t *p_out, uint64_t *v0_out, int hit_i0) {
    return ev_prefix_below(p_out, v0_out, hit_i0, 0x20000);
}

static void ev_scene(const char *name, int hits, int churn) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    for (int i = 0; i < churn; i++) {
        uint64_t q = p + 0x100000 + (uint64_t)i * 0x10000;   // one commit granule apart: each a miss
        cpu_fill(q, 4, 0xCCCCCCCCu);
        resolve(q, 4, v);
    }
    CHECK(resolve(p - 0x800, 0x1000, v).kind == BACKING_IMPORT);   // B0
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.mirror_bytes == 256 * KiB - 0x1000 && s.raw_dry_evals == 1 && s.raw_dry_would_grow == 1);
    for (int i = 0; i < hits; i++) resolve(p + 0x2000, 4, v);
    resolve(p + 0x2000, 4, v);
    resolve(p - 0x10000, 4, v);
    printf("%s: p=%#llx v0=%llu v=%llu b0_addr=%#llx b0_size=0x1000 bind_addr=%#llx bind_size=0x2800\n", name, (ull)p,
           (ull)v0, (ull)v, (ull)(p - 0x800), (ull)(p + 0x800));
    resolve(p + 0x800, 0x2800, v);   // pages [p, p + 0x3000): across I and M's new start
    printf("returned\n");
}

static void case_raw_ev_miss(void) { ev_scene("raw_ev_miss", 0, 0); }
static void case_raw_ev_survives_hits(void) { ev_scene("raw_ev_survives_hits", 1000, 0); }
static void case_raw_ev_lost(void) { ev_scene("raw_ev_lost", 0, 200); }

// The mirror B0 cuts was made by a promotion, not by a bind: the dump keeps that origin next to the cut.
static void case_raw_ev_origin(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB, P = p + 0x100000;
    pool(p - 1 * MiB, 16 * MiB);
    cpu_fill(P, 128 * KiB, 0xDDDDDDDDu);
    uint64_t v1 = sub_begin();
    backing_raw_begin(v1);
    CHECK(resolve(P, 128 * KiB, v1).kind == BACKING_IMPORT);   // larger than a uniform's: hot
    sub_end(v1);
    gpu_run(v1, NULL);
    cpu_fill(P - 0x20000, 0x20000, 0xBBBBBBBBu);   // an import I0 at the next begin
    uint64_t v = sub_begin();
    backing_raw_begin(v);   // the promotion: the mirror M = [P, P + 128 KiB)
    CHECK(stats().raw_promoted == 1 && stats().mirrors == 1);
    CHECK(resolve(P - 0x800, 0x1000, v).kind == BACKING_IMPORT);   // B0 cuts M's first page
    resolve(P + 0x2000, 4, v);
    resolve(P - 0x10000, 4, v);
    printf("raw_ev_origin: P=%#llx v1=%llu v=%llu\n", (ull)P, (ull)v1, (ull)v);
    resolve(P + 0x800, 0x2800, v);
    printf("returned\n");
}

static void ev_finish(uint64_t p, const char *name) {
    uint64_t v = D.recording;
    sub_end(v);
    gpu_run(v, NULL);
    BackingStats s = stats();
    CHECK(s.raw_dry_mutated == 0);
    printf("%s: p=%#llx evals=%llu would=%llu refused=%llu\n", name, (ull)p, (ull)s.raw_dry_evals,
           (ull)s.raw_dry_would_grow, (ull)s.raw_dry_refused);
    done(name);
}

static void dry_expect(const BackingStats *s, int would, int only_guard) {
    CHECK(s->raw_dry_evals == 1 && s->raw_dry_would_grow == (uint64_t)would && s->raw_dry_refused == (uint64_t)!would);
    CHECK(s->raw_dry_mutated == 0);
    for (int i = 0; i < T_N; i++) CHECK(s->raw_dry_g[i] == (uint64_t)(i == only_guard));
}

// The same for a set of refusals: bit i of mask is T_* i. A bind can count in several. No case here makes G3, G6 or G7b
// refuse (they need a second cell, a mirror at the end of its buffer, and an allocation just before a mirror's part). G9
// has none either: in the scene tried (a write declared in the other mirror over the first page the growth adds) the run
// stopped by name right after the verdict ("the write cannot move to it"), so its counter could not be read.
static void dry_expect_mask(const BackingStats *s, uint32_t mask) {
    CHECK(s->raw_dry_evals == 1 && s->raw_dry_would_grow == (uint64_t)!mask && s->raw_dry_refused == (uint64_t)!!mask);
    CHECK(s->raw_dry_mutated == 0);
    for (int i = 0; i < T_N; i++) {
        if (s->raw_dry_g[i] != (uint64_t)(mask >> i & 1u))
            fprintf(stderr, "dry mask: counter %d is %llu, want %u\n", i, (ull)s->raw_dry_g[i], mask >> i & 1u);
        CHECK(s->raw_dry_g[i] == (uint64_t)(mask >> i & 1u));
    }
}

// B0 cuts a mirror the submission did not resolve, over an import it did not use: growing the mirror would pass every guard.
static void case_raw_dry_grow(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    uint64_t version = backing_version();
    uint64_t calls = g_src_calls, resets = g_src_resets;
    CHECK(resolve(p - 0x800, 0x1000, v).kind == BACKING_IMPORT);
    CHECK(g_src_calls == calls + 1 && g_src_resets == resets);   // one question to the write watch (G8), and no mark taken
    BackingStats s = stats();
    CHECK(s.raw_dry_evals == 1 && s.raw_dry_would_grow == 1 && s.raw_dry_refused == 0 && s.raw_dry_mutated == 0);
    for (int i = 0; i < T_N; i++) CHECK(s.raw_dry_g[i] == 0);
    CHECK(backing_version() > version);   // the cut itself ran as it always did: the mirror is smaller, the import larger
    CHECK(s.mirrors == 1 && s.mirror_bytes == 256 * KiB - 0x1000 && s.imports == 1 && s.import_bytes == 0x21000);
    ev_finish(p, "raw_dry_grow");
}

// The import beside the mirror was used by the submission first: growing the mirror over it is the G4 refusal.
static void case_raw_dry_g4(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 1);
    CHECK(resolve(p - 0x800, 0x1000, v).kind == BACKING_IMPORT);
    BackingStats s = stats();
    dry_expect_mask(&s, 1u << T_G4 | 1u << T_G4_OFF);   // its hull is a word 0x10000 bytes below the pages the growth adds
    ev_finish(p, "raw_dry_g4");
}

// The same, but the submission used the import ON the pages the growth would add: G4 refuses, and the hull is not off them.
static void case_raw_dry_g4_on(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    CHECK(resolve(p - 0x400, 4, v).kind == BACKING_IMPORT);   // a word of I0 on the page [p - 0x1000, p) the bind adds
    CHECK(resolve(p - 0x800, 0x1000, v).kind == BACKING_IMPORT);
    BackingStats s = stats();
    dry_expect_mask(&s, 1u << T_G4);
    ev_finish(p, "raw_dry_g4_on");
}

// The write watch does not cover the pages the growth would add (the stand-in answers "not watched" for the one query of
// the bind): G8 alone, and the map is the same.
static void case_raw_dry_g8(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    g_untracked = 1;
    BackingRef b0 = resolve(p - 0x800, 0x1000, v);
    g_untracked = 0;
    CHECK(b0.kind == BACKING_IMPORT);
    BackingStats s = stats();
    dry_expect_mask(&s, 1u << T_G8);
    ev_finish(p, "raw_dry_g8");
}

// The head the growth adds (0x18000 bytes) is more than the mirror has before its start in its buffer (a mirror reserves
// BACKING_MIRROR_GUARD there): G5 alone.
static void case_raw_dry_g5(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    CHECK(resolve(p - 0x18000, 0x18800, v).kind == BACKING_IMPORT);   // pages [p - 0x18000, p + 0x1000)
    BackingStats s = stats();
    dry_expect_mask(&s, 1u << T_G5);
    ev_finish(p, "raw_dry_g5");
}

// A growth of more than BACKING_MIRROR_GROW_MAX (1.125 MiB of head): G1, and G5 with it (the head is also past the guard).
static void case_raw_dry_g1(void) {
    uint64_t p, v0, v = ev_prefix_below(&p, &v0, 0, 0x140000);
    CHECK(resolve(p - 0x120000, 0x120800, v).kind == BACKING_IMPORT);
    BackingStats s = stats();
    dry_expect_mask(&s, 1u << T_G1 | 1u << T_G5);
    ev_finish(p, "raw_dry_g1");
}

// Two mirrors M2 = [p - 0x20000, p - 0x10000) and M = [p, p + 256 KiB), the gap between them committed since (an import at
// the raw-access begin), none resolved by the submission: a bind across M2's last page, the gap and M's first page cuts
// one of them, and the pages its growth would add lie on the other mirror: G2.
static void case_raw_dry_g2(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB;
    pool(p - 1 * MiB, 16 * MiB);
    cpu_fill(p - 0x20000, 0x10000, 0xBBBBBBBBu);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p - 0x20000, 4, v0).kind == BACKING_MIRROR);
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p - 0x10000, 0x10000, 0xCCCCCCCCu);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingStats s0 = stats();
    CHECK(s0.mirrors == 2 && s0.imports == 1 && s0.import_bytes == 0x10000);
    CHECK(resolve(p - 0x10800, 0x11000, v).kind == BACKING_IMPORT);   // pages [p - 0x11000, p + 0x1000)
    BackingStats s = stats();
    // G7a with it: M2 would grow 0x11000 bytes at its end, into the block space the other mirror's allocation holds
    dry_expect_mask(&s, 1u << T_G2 | 1u << T_G7A);
    ev_finish(p, "raw_dry_g2");
}

// The mirrors already hold more than the cap on their total (three of 44 MiB, as in raw_grow_over_total): growing one is
// refused by the cap alone.
static void case_raw_dry_cap(void) {
    setup(256 * MiB, 0);
    backing_set_mirror_hook(want_all);
    const uint64_t win = BACKING_IMPORT_WINDOW, len = 44 * MiB;
    const uint64_t run[3] = {g_heap + 1 * MiB, g_heap + win + 1 * MiB, g_heap + 2 * win + 1 * MiB};
    pool(g_heap, 3 * win);
    for (int i = 0; i < 3; i++) cpu_fill(run[i], len, 0xA0A0A0A0u + (uint32_t)i);
    uint64_t v0 = sub_begin();
    for (int i = 0; i < 3; i++) CHECK(resolve(run[i], 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    CHECK(stats().mirror_bytes == 3 * len && stats().mirror_bytes > BACKING_PROMOTE_MIRROR_MAX);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    CHECK(resolve(run[0] - 0x800, 0x1000, v).kind == BACKING_IMPORT);
    BackingStats s = stats();
    dry_expect(&s, 0, T_CAP);
    ev_finish(run[0], "raw_dry_cap");
}

static int want_none(const BackingAsk *a) {
    (void)a;
    return 0;
}

// The rule says import for the bind (the hook no longer asks for mirrors): the candidate growth would copy a range the hook
// wants live, so it is refused for that alone.
static void case_raw_dry_kind(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    backing_set_mirror_hook(want_none);
    CHECK(resolve(p - 0x800, 0x1000, v).kind == BACKING_IMPORT);
    BackingStats s = stats();
    dry_expect(&s, 0, T_KIND);
    ev_finish(p, "raw_dry_kind");
}

// A miss that touches no mirror is not evaluated.
static void case_raw_dry_none(void) {
    uint64_t p, v0, v = ev_prefix(&p, &v0, 0);
    cpu_fill(p + 0x100000, 4, 1);
    CHECK(resolve(p + 0x100000, 4, v).kind == BACKING_IMPORT);
    BackingStats s = stats();
    CHECK(s.resolve_misses == 2 && s.raw_dry_evals == 0 && s.raw_dry_would_grow == 0 && s.raw_dry_refused == 0);
    ev_finish(p, "raw_dry_none");
}

// ---------- a growth over an import the submission used (backing.c, import_gives)
// A mirror M = [p, E) with E = p + 256 KiB, made by an earlier submission; 256 KiB the CPU wrote at E, committed since, which
// the raw-access begin of v gives an import I = [E, E + 256 KiB). In v the mirror is resolved at E - 0xc00 and the import at
// the places the mode says, then a bind across the boundary, [E - 0x600, E + 0x810), takes the mirror's tail growth over E's
// page: the pages [E, E + 0x1000) are I's lowest. Passing: the mirror grows and I is trimmed in place (its start moves up
// one page, its buffer, stamps and hull stay). Stopping: the rule that refused I is named in the stop.
enum { U_OK, U_EDGE, U_HULL, U_EDGE_HULL, U_WRITTEN, U_LOOSE };
static BackingRef g_u_big, g_u_import;

static void used_resolves(int mode, uint64_t E, uint64_t v) {
    CHECK(resolve(E - 0xc00, 0x1d0, v).kind == BACKING_MIRROR);
    g_u_import = mode == U_HULL        ? resolve(E + 0x100, 8, v)    // the hull starts inside the page the mirror takes
                 : mode == U_EDGE_HULL ? resolve(E + 0xff8, 8, v)    // ... in its last 8 bytes
                 : mode == U_EDGE      ? resolve(E + 0x1000, 8, v)   // ... in the page after it
                 : mode == U_LOOSE     ? resolve(E + 0x1800, 0, v)   // an extent not stated
                                       : resolve(E + 0x1800, 0x2000, v);
    CHECK(g_u_import.kind == BACKING_IMPORT);
    CHECK(resolve(E + 0x9000, 0x1d0, v).kind == BACKING_IMPORT);
    if (mode == U_WRITTEN) backing_gpu_wrote(E + 0x1800, 4, v);
}

static void used_world(uint64_t *p_out, pool_t *pp_out) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB;
    *pp_out = pool(p - 1 * MiB, 16 * MiB);
    CHECK(p / BACKING_IMPORT_WINDOW == (p + 512 * KiB) / BACKING_IMPORT_WINDOW);   // one cell
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(p + 256 * KiB, 256 * KiB, 0xBBBBBBBBu);   // committed since: an import at the next begin
    *p_out = p;
}

static uint64_t used_scene(int mode, uint64_t *p_out, pool_t *pp_out) {
    uint64_t p;
    used_world(&p, pp_out);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.mirror_bytes == 256 * KiB && s.imports == 1 && s.import_bytes == 256 * KiB);
    used_resolves(mode, p + 256 * KiB, v);
    *p_out = p;
    return v;
}

static void gpu_used(void) {
    uint8_t *page = bytes(&g_u_big) + 0x600;   // the page at E, as the mirror holds it
    CHECK(*(uint32_t *)page == 0xBBBBBBBBu && *(uint32_t *)(page + 0xff8) == 0xBBBBBBBBu);   // copied in from guest memory
    *(uint32_t *)(page + 0x100) = 0xCCCCCCCCu;   // a store through the mirror, declared below
    CHECK(words_are(bytes(&g_u_import), 0x100, 0xBBBBBBBBu));
}

// Passing: the growth trims I. The mirror and the import reach the same bytes as before, no page has two backings, the
// device copy of the given page is the guest's, and a write through the mirror to it reaches guest memory.
static void used_pass(const char *name, int mode) {
    uint64_t p;
    pool_t pp;
    uint64_t v = used_scene(mode, &p, &pp);
    const uint64_t E = p + 256 * KiB;
    BackingRef before = g_u_import;
    BackingStats s0 = stats();
    uint64_t ver0 = backing_version(), up0 = D.upload_bytes;
    BackingRun *r0;
    uint64_t rv;
    int n0 = backing_raw_runs(&r0, &rv);
    CHECK(run_at(r0, n0, E) && run_at(r0, n0, E)->kind == BACKING_IMPORT);
    free(r0);

    g_u_big = resolve(E - 0x600, 0xe10, v);
    CHECK(g_u_big.kind == BACKING_MIRROR);
    BackingStats s = stats();
    CHECK(s.mirror_bytes == s0.mirror_bytes + 0x1000 && s.import_bytes == s0.import_bytes - 0x1000 && s.imports == s0.imports);
    CHECK(s.raw_grows == 1 && s.raw_grow_bytes == 0x1000 && s.raw_grow_used == 1 && s.raw_grow_trims == 1 &&
          s.raw_grow_trim_bytes == 0x1000 && s.raw_grow_waits == 0 && s.replace_waits == 0);
    CHECK(s.raw_refused[1] + s.raw_refused[2] + s.raw_refused[3] + s.raw_refused[4] + s.raw_refused[5] + s.raw_refused[6] == 0);
    CHECK(D.upload_bytes - up0 == 0x1000);   // the one page, copied from guest memory
    CHECK(backing_version() > ver0);   // the table is out of date: the page changed hands
    BackingPage pg;
    backing_page(E, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(E + 0x1000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    BackingRun *runs;
    int n = backing_raw_runs(&runs, &rv);
    CHECK(run_at(runs, n, E) && run_at(runs, n, E)->kind == BACKING_MIRROR);
    CHECK(run_at(runs, n, E + 0x1000) && run_at(runs, n, E + 0x1000)->kind == BACKING_IMPORT);
    for (uint64_t a = p; a < E + 256 * KiB; a += PG) CHECK(runs_cover(runs, n, a) == 1);   // no page twice
    CHECK(raw_address(runs, n, E + 0x1000) == E + 0x1000 && raw_address(runs, n, E + 0x9000) == E + 0x9000);
    CHECK(raw_address(runs, n, E) == g_u_big.address + 0x600);
    free(runs);
    // what was recorded before the growth still reaches the same bytes: the import's buffer, memory and offset are as they were
    BackingRef again = resolve(E + 0x1800, 0x2000, v);
    CHECK(again.kind == BACKING_IMPORT && again.buffer == before.buffer && again.memory == before.memory);
    CHECK(mode == U_OK ? again.offset == before.offset : 1);
    BackingRef tail = resolve(E + 0x1800, 4, v);
    CHECK(tail.kind == BACKING_IMPORT && tail.buffer == before.buffer && *(uint32_t *)bytes(&tail) == 0xBBBBBBBBu);
    CHECK(tail.address == E + 0x1800);   // the import's device address of that guest address, unchanged by the trim
    g_u_import = again;
    backing_gpu_wrote(E + 0x100, 4, v);
    sub_end(v);
    gpu_run(v, gpu_used);
    CHECK(*(uint32_t *)(uintptr_t)(E + 0x100) == 0xCCCCCCCCu);   // written back to where it was declared
    CHECK(*(uint32_t *)(uintptr_t)(E + 0x104) == 0xBBBBBBBBu);
    s = stats();
    printf("%s: the mirror grew one page over an import the submission used, which was trimmed in place (raw_grows %llu, "
           "raw_grow_trims %llu)\n", name, (ull)s.raw_grows, (ull)s.raw_grow_trims);
    backing_pool_fini(&pp);
    done(name);
}

static void case_raw_grow_used_import(void) { used_pass("raw_grow_used_import", U_OK); }
static void case_raw_grow_used_edge(void) { used_pass("raw_grow_used_edge", U_EDGE); }

static void used_stop(const char *name, int mode) {
    uint64_t p;
    pool_t pp;
    uint64_t v = used_scene(mode, &p, &pp);
    printf("%s: p=%#llx E=%#llx v=%llu\n", name, (ull)p, (ull)(p + 256 * KiB), (ull)v);
    resolve(p + 256 * KiB - 0x600, 0xe10, v);
    printf("returned\n");
}

static void case_raw_grow_used_written(void) { used_stop("raw_grow_used_written", U_WRITTEN); }
static void case_raw_grow_used_hull(void) { used_stop("raw_grow_used_hull", U_HULL); }
static void case_raw_grow_used_hull_edge(void) { used_stop("raw_grow_used_hull_edge", U_EDGE_HULL); }
static void case_raw_grow_used_loose(void) { used_stop("raw_grow_used_loose", U_LOOSE); }
// A head growth: the pages the bind takes are the import's highest, above everything the submission resolved in it.
static void case_raw_grow_used_up(void) { diag_scene("raw_grow_used_up"); }

// The submission's GPU work is older than the growth: a plain submission r wrote through I and is issued, not reached by the
// device; v, a raw-access submission after a first one that is done, used I. I's stamp is v's now, so which submission wrote
// through it is not in the stamps: the growth waits for every submission issued (r), never for v itself.
static volatile int g_wd_done;
static void used_watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(5000ull * 1000 * 1000);
    if (g_wd_done) return;
    fprintf(stderr, "grew waited for its own submission\n");
    exit(1);
}

static void case_raw_grow_used_wait(void) {
    uint64_t p;
    pool_t pp;
    used_world(&p, &pp);
    const uint64_t E = p + 256 * KiB;
    uint64_t w = sub_begin();
    backing_raw_begin(w);   // the import is made here
    CHECK(stats().imports == 1 && stats().import_bytes == 256 * KiB);
    sub_end(w);
    gpu_run(w, NULL);
    uint64_t r = sub_begin();   // a plain submission
    CHECK(resolve(E + 0x1800, 0x2000, r).kind == BACKING_IMPORT);
    backing_gpu_wrote(E + 0x1800, 4, r);
    sub_end(r);   // issued; its work waits for the test
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    CHECK(gpu_sync_done() == w && gpu_sync_issued() == r);
    used_resolves(U_OK, E, v);
    g_older = r, g_older_work = NULL;
    CHECK(plat_thread_start(run_older_later, NULL, 256 * KiB));
    CHECK(plat_thread_start(used_watchdog, NULL, 256 * KiB));
    uint64_t t0 = plat_now_ns();
    BackingRef big = resolve(E - 0x600, 0xe10, v);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    g_wd_done = 1;
    BackingStats s = stats();
    CHECK(big.kind == BACKING_MIRROR && waited >= 150.0 && gpu_sync_done() >= r);
    CHECK(s.raw_grow_waits == 1 && s.raw_grow_trims == 1 && s.raw_grows == 1);
    sub_end(v);
    gpu_run(v, NULL);
    printf("raw_grow_used_wait: the growth waited %.1f ms for the submission issued before it\n", waited);
    backing_pool_fini(&pp);
    done("raw_grow_used_wait");
}

// Two used imports replaced by one bind (a miss across the uncommitted chunk between them): the import that replaces
// them takes over what the submission knew of both, so a later growth over it sees the hull of the first one and the GPU
// write declared in it. A hull of an older submission says nothing about this one.
enum { H_PASS, H_HULL, H_WRITTEN, H_OLD };
static void heir_scene(const char *name, int mode, int stop) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 8 * MiB, A = p + 256 * KiB, B = A + 128 * KiB;   // M = [p, A), I1 at A, a chunk none wrote, I2 at B
    pool_t pp = pool(p - 1 * MiB, 16 * MiB);
    cpu_fill(p, 256 * KiB, 0xAAAAAAAAu);
    uint64_t v0 = sub_begin();
    CHECK(resolve(p, 4, v0).kind == BACKING_MIRROR);
    sub_end(v0);
    gpu_run(v0, NULL);
    cpu_fill(A, 64 * KiB, 0xBBBBBBBBu);
    cpu_fill(B, 64 * KiB, 0xCCCCCCCCu);
    if (mode == H_OLD) {   // I1 was resolved by an earlier raw-access submission, not by v
        uint64_t w = sub_begin();
        backing_raw_begin(w);
        CHECK(resolve(A + 0x100, 8, w).kind == BACKING_IMPORT);
        sub_end(w);
        gpu_run(w, NULL);
    }
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    BackingStats s = stats();
    CHECK(s.mirrors == 1 && s.imports == 2 && s.import_bytes == 128 * KiB);
    CHECK(resolve(A - 0xc00, 0x1d0, v).kind == BACKING_MIRROR);
    const uint64_t h1 = mode == H_HULL ? 0x100 : 0x4000;
    if (mode != H_OLD) CHECK(resolve(A + h1, 8, v).kind == BACKING_IMPORT);
    if (mode == H_WRITTEN) backing_gpu_wrote(A + h1, 4, v);
    CHECK(resolve(B + 0x100, 8, v).kind == BACKING_IMPORT);
    CHECK(resolve(A + 0x10100, 8, v).kind == BACKING_IMPORT);   // commits the chunk: one import over I1, it and I2
    s = stats();
    CHECK(s.imports == 1 && s.import_bytes == 192 * KiB);
    CHECK(s.raw_heirs == (mode == H_OLD ? 1u : 2u) && s.raw_heir_written == (mode == H_WRITTEN ? 1u : 0u));
    if (stop) {
        printf("%s: p=%#llx A=%#llx v=%llu\n", name, (ull)p, (ull)A, (ull)v);
        resolve(A - 0x600, 0xe10, v);
        printf("returned\n");
        return;
    }
    BackingRef big = resolve(A - 0x600, 0xe10, v);
    s = stats();
    CHECK(big.kind == BACKING_MIRROR && s.raw_grows == 1 && s.raw_grow_trims == 1 && s.raw_grow_trim_bytes == 0x1000);
    CHECK(s.imports == 1 && s.import_bytes == 192 * KiB - 0x1000);
    sub_end(v);
    gpu_run(v, NULL);
    backing_pool_fini(&pp);
    done(name);
}

static void case_raw_grow_heir(void) { heir_scene("raw_grow_heir", H_PASS, 0); }
static void case_raw_grow_heir_old(void) { heir_scene("raw_grow_heir_old", H_OLD, 0); }
static void case_raw_grow_heir_hull(void) { heir_scene("raw_grow_heir_hull", H_HULL, 1); }
static void case_raw_grow_heir_written(void) { heir_scene("raw_grow_heir_written", H_WRITTEN, 1); }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: coherence_test <case>\n");
        return 2;
    }
    static const struct { const char *name; void (*fn)(void); } cases[] = {
        {"generations", case_generations}, {"upload", case_upload},   {"admission", case_admission},
        {"guarded_admission", case_guarded_admission},
        {"writeback", case_writeback},     {"writeback_no_reupload", case_writeback_no_reupload},
        {"writeback_partial_page", case_writeback_partial_page}, {"writeback_replaced", case_writeback_replaced},
        {"writeback_long_run", case_writeback_long_run}, {"gpu_mark", case_gpu_mark},
        {"gpu_mark_direct", case_gpu_mark_direct}, {"gpu_mark_direct_partial", case_gpu_mark_direct_partial},
        {"replace", case_replace},   {"watch", case_watch},
        {"unwritten", case_unwritten},     {"replace_written", case_replace_written},
        {"raw_grow", case_raw_grow},       {"raw_grow_whole", case_raw_grow_whole},
        {"raw_grow_head", case_raw_grow_head},
        {"raw_grow_beside", case_raw_grow_beside}, {"raw_grow_wait", case_raw_grow_wait},
        {"raw_sync_wait", case_raw_sync_wait}, {"raw_sync_nowait", case_raw_sync_nowait},
        {"raw_promote", case_raw_promote},   {"raw_promote_small", case_raw_promote_small},
        {"raw_promote_wait", case_raw_promote_wait}, {"raw_promote_cap", case_raw_promote_cap},
        {"raw_grow_taken", case_raw_grow_taken}, {"raw_grow_neighbour", case_raw_grow_neighbour},
        {"raw_grow_taken_wait", case_raw_grow_taken_wait}, {"raw_grow_taken_head", case_raw_grow_taken_head},
        {"raw_grow_trimmed", case_raw_grow_trimmed}, {"raw_grow_taken_layout", case_raw_grow_taken_layout},
        {"raw_grow_guarded_head", case_raw_grow_guarded_head},
        {"raw_grow_guarded_tail", case_raw_grow_guarded_tail},
        {"raw_grow_bound_import", case_raw_grow_bound_import}, {"raw_grow_fragmented", case_raw_grow_fragmented},
        {"raw_grow_guarded_blocked", case_raw_grow_guarded_blocked},
        {"raw_grow_untracked", case_raw_grow_untracked},
        {"raw_grow_over_total", case_raw_grow_over_total}, {"raw_grow_head_import", case_raw_grow_head_import},
        {"raw_grow_tail_import", case_raw_grow_tail_import}, {"raw_grow_cap", case_raw_grow_cap},
        {"raw_diag_hull", case_raw_diag_hull}, {"raw_diag_dry", case_raw_diag_dry},
        {"raw_diag_birth", case_raw_diag_birth},
        {"raw_ev_miss", case_raw_ev_miss}, {"raw_ev_survives_hits", case_raw_ev_survives_hits},
        {"raw_ev_lost", case_raw_ev_lost}, {"raw_ev_origin", case_raw_ev_origin},
        {"raw_dry_grow", case_raw_dry_grow}, {"raw_dry_g4", case_raw_dry_g4}, {"raw_dry_cap", case_raw_dry_cap},
        {"raw_dry_kind", case_raw_dry_kind}, {"raw_dry_none", case_raw_dry_none}, {"raw_dry_g4_on", case_raw_dry_g4_on},
        {"raw_dry_g8", case_raw_dry_g8}, {"raw_dry_g5", case_raw_dry_g5}, {"raw_dry_g1", case_raw_dry_g1},
        {"raw_dry_g2", case_raw_dry_g2},
        {"raw_grow_used_import", case_raw_grow_used_import}, {"raw_grow_used_edge", case_raw_grow_used_edge},
        {"raw_grow_used_wait", case_raw_grow_used_wait}, {"raw_grow_used_written", case_raw_grow_used_written},
        {"raw_grow_used_hull", case_raw_grow_used_hull}, {"raw_grow_used_hull_edge", case_raw_grow_used_hull_edge},
        {"raw_grow_used_loose", case_raw_grow_used_loose}, {"raw_grow_used_up", case_raw_grow_used_up},
        {"raw_grow_heir", case_raw_grow_heir}, {"raw_grow_heir_old", case_raw_grow_heir_old},
        {"raw_grow_heir_hull", case_raw_grow_heir_hull}, {"raw_grow_heir_written", case_raw_grow_heir_written},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(argv[1], cases[i].name)) {
            cases[i].fn();
            return 0;
        }
    fprintf(stderr, "coherence_test: no case %s\n", argv[1]);
    return 2;
}
