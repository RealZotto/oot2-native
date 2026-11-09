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
//   writeback    a declared GPU write reaches guest memory only once the device reached its submission; a copy into the
//                mirror meanwhile leaves its bytes out; guest_read counts a read before the fence, cpu_synced finds
//                nothing left behind
//   read_wait, read_import, read_effect, read_raw   an invalidation waits for an overlapping mirror or import write,
//                a completion effect, or a possible raw store; a disjoint declared write does not delay a read
//   read_unissued, read_replay, read_latest   the recorder is caught up before testing the ranges, and the newest
//                overlapping submission is done before the read returns
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
//   raw_sync_wait   a mirror the CPU wrote is brought current for a raw-access submission after the older one in flight
//   raw_promote, raw_promote_small, raw_promote_wait, raw_promote_cap   an import a raw-access submission binds a range larger than a
//                uniform's of becomes a mirror at the next raw-access begin: brought current, in the page table, left alone when
//                the submission bound it before its raw stage or the admission refuses, never for a small bind, after the older
//                raw-access submission in flight, and not past the mirrors' cap (the raw write-back's size)
//   unwritten, replace_written, raw_grow_taken   stop by name
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
static uint64_t g_src_calls;

static uint64_t src_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    g_src_calls++;
    if (g_untracked || addr < g_heap || addr + size > g_heap + g_heap_size) return UINT64_MAX;
    uint64_t n = 0;
    for (uint64_t a = addr & ~(PG - 1); a < addr + size && n < max; a += PG) {
        uint64_t i = (a - g_heap) / PG;
        if (!g_marks[i]) continue;
        pages[n++] = a;
        if (reset) g_marks[i] = 0;
    }
    return n;
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
    D.used = 8 * GiB - 8 * GiB / BACKING_MIRROR_RESERVE - 1 * MiB;
    CHECK(resolve(g_heap + 5 * MiB, 0x100, 0).kind == BACKING_MIRROR);
    // A range the write watch does not cover: an import.
    D.used = 0, g_untracked = 1;
    CHECK(resolve(g_heap + 7 * MiB, 0x100, 0).kind == BACKING_IMPORT);
    s = stats();
    CHECK(s.mirrors == 2 && s.imports == 2 && s.mirrors_refused_untracked == 1 && s.mirrors_made == 2);
    backing_pool_fini(&a), backing_pool_fini(&b), backing_pool_fini(&d), backing_pool_fini(&e);
    done("admission");
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
    backing_no_read_wait = 1;
    backing_guest_read(p + 0x180, 4);
    backing_guest_read(p + 0x400, 4);
    backing_no_read_wait = 0;
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

static void gpu_writes_dd(void) { memset(g_mirror + 0x100, 0xDD, 0x100); }
static uint64_t g_run_later;
static void run_later(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    gpu_run(g_run_later, gpu_writes_dd);
}

static void finish_later(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    gpu_run(g_run_later, NULL);
}

static void issue_later(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    sub_end(g_run_later);
    gpu_run(g_run_later, NULL);
}

static void write_effect(void *arg, uint64_t value);

static void issue_write_later(void *arg) {
    uint64_t addr = (uint64_t)(uintptr_t)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    backing_cpu_write_pending(addr, 4, g_run_later);
    gpu_sync_effect(write_effect, (void *)(uintptr_t)addr);
    sub_end(g_run_later);
    gpu_run(g_run_later, NULL);
}

static uint64_t g_run_first;
static void finish_two_later(void *arg) {
    (void)arg;
    plat_sleep_ns(100 * 1000 * 1000);
    gpu_run(g_run_first, NULL);
    plat_sleep_ns(100 * 1000 * 1000);
    gpu_run(g_run_later, NULL);
}

static void case_read_wait(void) {
    setup(64 * MiB, 0);
    backing_set_mirror_hook(want_all);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    uint64_t v = sub_begin();
    BackingRef r = resolve(p, 0x4000, v);
    CHECK(r.kind == BACKING_MIRROR);
    g_mirror = bytes(&r);
    backing_gpu_wrote(p + 0x100, 0x100, v);
    sub_end(v);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0xAAAAAAAAu);
    g_run_later = v;
    CHECK(plat_thread_start(run_later, NULL, 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    uint32_t observed = *(volatile uint32_t *)(uintptr_t)(p + 0x100);
    CHECK(gpu_sync_wait(v, 5000000000ull));
    CHECK(observed == 0xDDDDDDDDu);
    backing_pool_fini(&pp);
    done("read_wait");
}

static void case_read_import(void) {
    setup(64 * MiB, 0);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0xAAAAAAAAu);
    uint64_t v = sub_begin();
    BackingRef r = resolve(p, 0x4000, v);
    CHECK(r.kind == BACKING_IMPORT);
    g_mirror = bytes(&r);
    backing_gpu_wrote(p + 0x100, 0x100, v);
    sub_end(v);
    // An adjacent range must not wait for this submission.
    backing_guest_read(p + 0x200, 4);
    CHECK(gpu_sync_done() < v && stats().reads_in_flight == 0);
    g_run_later = v;
    CHECK(plat_thread_start(run_later, NULL, 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0xDDDDDDDDu);
    CHECK(stats().reads_in_flight == 1);
    backing_pool_fini(&pp);
    done("read_import");
}

static void write_effect(void *arg, uint64_t value) {
    (void)value;
    *(uint32_t *)arg = 0x12345678u;
}

static void case_read_effect(void) {
    setup(64 * MiB, 0);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0);
    uint64_t v = sub_begin();
    backing_cpu_write_pending(p + 0x100, 4, v);
    gpu_sync_effect(write_effect, (void *)(uintptr_t)(p + 0x100));
    sub_end(v);
    g_run_later = v;
    CHECK(plat_thread_start(finish_later, NULL, 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0x12345678u);
    backing_pool_fini(&pp);
    done("read_effect");
}

static void case_read_raw(void) {
    setup(64 * MiB, 0);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0);
    uint64_t v = sub_begin();
    backing_raw_begin(v);
    gpu_sync_effect(write_effect, (void *)(uintptr_t)(p + 0x100));
    sub_end(v);
    g_run_later = v;
    CHECK(plat_thread_start(finish_later, NULL, 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0x12345678u);
    backing_pool_fini(&pp);
    done("read_raw");
}

static void case_read_unissued(void) {
    setup(64 * MiB, 0);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0);
    uint64_t v = sub_begin();
    backing_cpu_write_pending(p + 0x100, 4, v);
    gpu_sync_effect(write_effect, (void *)(uintptr_t)(p + 0x100));
    CHECK(gpu_sync_issued() < v);
    g_run_later = v;
    CHECK(plat_thread_start(issue_later, NULL, 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0x12345678u);
    backing_pool_fini(&pp);
    done("read_unissued");
}

static void case_read_replay(void) {
    setup(64 * MiB, 0);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0);
    uint64_t v = sub_begin();
    CHECK(gpu_sync_issued() < v);
    g_run_later = v;
    CHECK(plat_thread_start(issue_write_later, (void *)(uintptr_t)(p + 0x100), 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0x12345678u);
    backing_pool_fini(&pp);
    done("read_replay");
}

static void case_read_latest(void) {
    setup(64 * MiB, 0);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 1 * MiB);
    cpu_fill(p, 1 * MiB, 0);
    uint64_t first = sub_begin();
    backing_cpu_write_pending(p + 0x100, 4, first);
    sub_end(first);
    uint64_t latest = sub_begin();
    backing_cpu_write_pending(p + 0x100, 4, latest);
    gpu_sync_effect(write_effect, (void *)(uintptr_t)(p + 0x100));
    sub_end(latest);
    g_run_first = first, g_run_later = latest;
    CHECK(plat_thread_start(finish_two_later, NULL, 256 * KiB));
    backing_guest_read(p + 0x100, 4);
    CHECK(*(volatile uint32_t *)(uintptr_t)(p + 0x100) == 0x12345678u);
    CHECK(gpu_sync_done() >= latest);
    backing_pool_fini(&pp);
    done("read_latest");
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
static void run_older_later(void *arg) {
    (void)arg;
    plat_sleep_ns(200 * 1000 * 1000);
    gpu_run(g_older, NULL);
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
// write-back holds that many bytes of pages, and a submission's stores cannot reach more than the mirrors it maps.
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

// ---------- cases that stop by name
// An earlier bind of the submission reached the pages a later one takes from the mirror: the table cannot give both the
// same copy of them.
static void case_raw_grow_taken(void) {
    uint64_t p;
    pool_t pp;
    BackingRef small;
    uint64_t v = raw_scene(&p, &pp, &small);
    resolve(p + 0x31000, 0x100, v);
    resolve(p + 0x30320, 1 * MiB, v);
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

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: coherence_test <case>\n");
        return 2;
    }
    static const struct { const char *name; void (*fn)(void); } cases[] = {
        {"generations", case_generations}, {"upload", case_upload},   {"admission", case_admission},
        {"writeback", case_writeback},     {"read_wait", case_read_wait},
        {"read_import", case_read_import}, {"read_effect", case_read_effect}, {"read_raw", case_read_raw},
        {"read_unissued", case_read_unissued},
        {"read_replay", case_read_replay},
        {"read_latest", case_read_latest},
        {"replace", case_replace}, {"watch", case_watch},
        {"unwritten", case_unwritten},     {"replace_written", case_replace_written},
        {"raw_grow", case_raw_grow},       {"raw_grow_whole", case_raw_grow_whole},
        {"raw_grow_head", case_raw_grow_head},
        {"raw_grow_beside", case_raw_grow_beside}, {"raw_grow_wait", case_raw_grow_wait},
        {"raw_sync_wait", case_raw_sync_wait},
        {"raw_promote", case_raw_promote},   {"raw_promote_small", case_raw_promote_small},
        {"raw_promote_wait", case_raw_promote_wait}, {"raw_promote_cap", case_raw_promote_cap},
        {"raw_grow_taken", case_raw_grow_taken},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(argv[1], cases[i].name)) {
            cases[i].fn();
            return 0;
        }
    fprintf(stderr, "coherence_test: no case %s\n", argv[1]);
    return 2;
}
