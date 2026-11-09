// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the texture heap instruments and the owner side of lever L1 (runtime/nvn/nvn_texheap.c, oot2/texheap.h,
// spike S7), driven by tests/test_runtime_texheap.py: `texheap_test <case> <folder>`. Each case runs in its own process with
// OOT2_TEXHEAP, OOT2_TEXHEAP_EVERY and OOT2_PAGE_RELEASE set as the case needs; the instruments write their file into
// <folder>, which the driver reads. A passing case prints "ok <case>" (or "skip <case>" where the platform has no write
// watch); a failing one must end in one named fatal.
//
// Pools, textures, buffers and registrations are made through the NVN2 entry points, the way generated code calls them
// (rt_call at their synthetic address). A recording is built by hand and handed to the null back end's submit, which
// walks it and completes it at once; texheap_present stands for nvnQueuePresentTexture. guest_read and guest_write stand
// for lifted guest code (they are in the function table); host_read32 and host_write32 (pages_host.c) for the runtime. Every
// texel is a number made here.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/gfx.h"
#include "oot2/memory.h"
#include "oot2/nvn_tex.h"
#include "oot2/pagegen.h"
#include "oot2/texheap.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
#define HEAP (32 * MiB)
#define P4 0x1000ull   // the page of the heap and of the page ranges
#define GPAGE GFX_GPU_PAGE
#define CPU_VISIBLE (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED)
#define MB MOD_MAIN_BASE

typedef unsigned long long ull;

static Cpu c;
static uint64_t heap, heap_used, loader, g_seq;

uint32_t host_read32(uint64_t a);
void host_write32(uint64_t a, uint32_t v);

// ---- lifted guest code, as the function table sees it
__attribute__((noinline)) static void guest_read(Cpu *cpu) { cpu->x[1] = LD32(cpu->x[0]) + 0x11; }
__attribute__((noinline)) static void guest_write(Cpu *cpu) { ST32(cpu->x[0], cpu->x[1]); cpu->x[2] = 0x22; }
__attribute__((noinline)) static void guest_end(Cpu *cpu) { cpu->x[2] = 0xE0; }
enum { FN_READ = MB + 0x200, FN_WRITE = MB + 0x300, FN_END = MB + 0x400 };

static uint32_t gread(uint64_t a) {
    c.x[0] = a;
    rt_call(&c, FN_READ);
    return (uint32_t)c.x[1] - 0x11;
}
static void gwrite(uint64_t a, uint32_t v) {
    c.x[0] = a, c.x[1] = v;
    rt_call(&c, FN_WRITE);
}

static void setup(const char *folder) {
    RtMemConfig cfg = {.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    c.x[0] = 0, c.x[1] = HEAP;
    rt_svc(&c, 0x01, 0);
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    static const uint64_t addrs[] = {FN_READ, FN_WRITE, FN_END};
    static const GuestFn fns[] = {guest_read, guest_write, guest_end};
    static const RtModule mods[] = {{"main", MB, MOD_MAIN_SIZE, addrs, fns, 3, TEST_END}};
    rt_dispatch_init(mods, 1);
    c.excl_addr = ~0ull;
    texheap_set_folder(folder);
    texheap_set_watch(&(TexheapWatch){pagegen_scan, pagegen_runs, NULL});
}

static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}
static void *gp(uint64_t a) { return (void *)(uintptr_t)a; }
static uint64_t guest_str(const char *s) {
    uint64_t a = galloc(strlen(s) + 1, 16);
    memcpy(gp(a), s, strlen(s) + 1);
    return a;
}
static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[64];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) {
        c.x[0] = guest_str("nvnDeviceGetProcAddress"), c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
        CHECK(gpa != 0);
    }
    c.x[0] = 0, c.x[1] = guest_str(name);
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0 && n < 64);
    cache[n].name = name, cache[n].addr = c.x[0];
    return cache[n++].addr;
}
static uint64_t nvn_vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}
#define ARGS(...) (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t)
#define NVN(name, ...) nvn_vcall(proc(#name), ARGS(__VA_ARGS__))

// ---- objects
static uint64_t pool(uint64_t size) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16), storage = galloc(size, GPAGE);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetStorage, b, storage, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, CPU_VISIBLE);
    CHECK((NVN(nvnMemoryPoolInitialize, o, b) & 0xff) == 1);
    return o;
}
static uint64_t pool_addr(uint64_t p) { return NVN(nvnMemoryPoolGetBufferAddress, p); }
static uint64_t texture(uint32_t w, uint32_t h, uint64_t p, uint64_t off) {
    uint64_t b = galloc(0x80, 16), t = galloc(0x80, 16);
    NVN(nvnTextureBuilderSetDevice, b, 0x1234);
    NVN(nvnTextureBuilderSetDefaults, b);
    NVN(nvnTextureBuilderSetSize2D, b, w, h);
    NVN(nvnTextureBuilderSetFormat, b, NVN_FORMAT_RGBA8);
    NVN(nvnTextureBuilderSetStorage, b, p, off);
    CHECK((NVN(nvnTextureInitialize, t, b) & 0xff) == 1);
    return t;
}
static uint64_t buffer(uint64_t p, uint64_t off, uint64_t size) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x40, 16);
    NVN(nvnBufferBuilderSetDefaults, b);
    NVN(nvnBufferBuilderSetStorage, b, p, off, size);
    CHECK((NVN(nvnBufferInitialize, o, b) & 0xff) == 1);
    return o;
}
static uint64_t texture_pool(uint64_t p, uint64_t off, int n) {
    uint64_t tp = galloc(0x20, 16);
    CHECK((NVN(nvnTexturePoolInitialize, tp, p, off, (uint64_t)n) & 0xff) == 1);
    return tp;
}
static uint64_t tex_size(uint64_t t) { return NVN(nvnTextureGetStorageSize, t); }

// One flat recording: bind the texture IDs (and, when rt is not NULL, set it as the colour target), then submit it. The
// null back end walks it and completes it before the call returns.
static void submit_on(uint64_t queue, uint64_t table, const uint32_t *ids, int n, const tex_t *rt,
                      volatile uint32_t *wait_on) {
    static uint8_t buf[4096] __attribute__((aligned(8)));
    size_t at = 0;
#define PUT(op_, payload, bytes)                                                       \
    do {                                                                               \
        op_hdr h_ = {(op_), (uint32_t)(sizeof(op_hdr) + (((bytes) + 7) & ~7ull))};     \
        memcpy(buf + at, &h_, sizeof h_);                                              \
        if (bytes) memcpy(buf + at + sizeof h_, payload, bytes);                       \
        at += h_.size;                                                                 \
    } while (0)
    PUT(OP_BEGIN, NULL, 0);
    if (wait_on) {   // the queue is held here until the word is 1
        p_wait_event we = {wait_on, NVN_EVENT_WAIT_MODE_EQUAL, 1, 0x1000};
        PUT(OP_WAIT_EVENT, &we, sizeof we);
    }
    p_set_pool sp = {{table, 64, 0}};
    PUT(OP_SET_TEX_POOL, &sp, sizeof sp);
    p_bind_handles bh;
    memset(&bh, 0, sizeof bh);
    bh.stage = 1, bh.n = n;
    for (int i = 0; i < n; i++) bh.h[i] = ids[i];
    PUT(OP_BIND_TEX, &bh, sizeof bh);
    if (rt) {
        p_set_rt s;
        memset(&s, 0, sizeof s);
        s.n = 1;
        s.color[0].tex = *rt, s.color[0].present = 1;
        PUT(OP_SET_RT, &s, sizeof s);
    }
    PUT(OP_END, NULL, 0);
    uint64_t handle = (uint64_t)(uintptr_t)buf;
    CHECK(gfx_backend()->submit(queue, &handle, 1) != 0);
}
static void submit_binding(uint64_t table, const uint32_t *ids, int n, const tex_t *rt) {
    submit_on(1, table, ids, n, rt, NULL);
}
static void present(void) {   // as nvnQueuePresentTexture calls it
    ++g_seq;
    if (texheap_on()) texheap_present(g_seq);
}

// The CPU writes every page [a, a + size) touches, as a texture upload would.
static void cpu_write(uint64_t a, uint64_t size) {
    for (uint64_t x = a & ~(P4 - 1); x < a + size; x += P4) host_write32(x, 0xAB);
}
static uint64_t first_page(uint64_t a) { return (a + P4 - 1) & ~(P4 - 1); }
static uint64_t last_page(uint64_t a, uint64_t size) { return (a + size) & ~(P4 - 1); }   // one past the last interior page
static RtPageStats stats(void) {
    RtPageStats s;
    CHECK(rt_pages_stats(0, &s));
    return s;
}
static int no_write_watch(void) { return pagegen_scan(heap, heap + P4) == UINT64_MAX; }

// The world of the selection cases. P: a pool of 1 MiB. T1 a CPU-written texture with edge pages, T2 a small one inside
// T1's pages, B a buffer inside one of T1's pages, T3 a render target, T4 a bound texture nobody writes, T5 another CPU-written
// texture. The texture pool's table lies at the end of the pool.
enum { OFF_T1 = 0x800, OFF_T2 = 0x8000, OFF_B = 0x5100, OFF_T3 = 0x20000, OFF_T4 = 0x30000, OFF_T5 = 0x40000, OFF_TP = 0xF0000 };
static struct { uint64_t p, base, tp, t1, t2, t3, t4, t5, s1; } w;

static void world(void) {
    w.p = pool(MiB);
    w.base = pool_addr(w.p);
    w.tp = texture_pool(w.p, OFF_TP, 64);
    w.t1 = texture(128, 128, w.p, OFF_T1);
    w.s1 = tex_size(w.t1);
    CHECK(w.s1 >= 0xA000 && OFF_T1 + w.s1 < OFF_T3);
    w.t2 = texture(32, 32, w.p, OFF_T2);
    CHECK(tex_size(w.t2) == P4);
    (void)buffer(w.p, OFF_B, 0x100);
    w.t3 = texture(64, 64, w.p, OFF_T3);
    w.t4 = texture(64, 64, w.p, OFF_T4);
    w.t5 = texture(64, 64, w.p, OFF_T5);
    uint64_t t[5] = {w.t1, w.t2, w.t3, w.t4, w.t5};
    for (int i = 0; i < 5; i++) NVN(nvnTexturePoolRegisterTexture, w.tp, (uint64_t)i, t[i], 0);
}
static uint64_t tp_addr(void) { return w.base + OFF_TP; }

// Pages of T1 that are interior, minus the one a buffer touches and the one T2 shares.
static uint64_t t1_expected(void) {
    uint64_t lo = first_page(w.base + OFF_T1), hi = last_page(w.base + OFF_T1, w.s1);
    CHECK(w.base + OFF_B >= lo && w.base + OFF_B < hi && w.base + OFF_T2 >= lo && w.base + OFF_T2 < hi);
    return (hi - lo) / P4 - 2;
}
static void write_cpu_textures(void) {
    cpu_write(w.base + OFF_T1, w.s1);
    cpu_write(w.base + OFF_T5, tex_size(w.t5));
    cpu_write(w.base + OFF_T2, P4);
}
static void bind_all(void) {
    static const uint32_t ids[] = {0, 1, 2, 3, 4};
    tex_t rt;
    memcpy(&rt, gp(w.t3), sizeof rt);
    submit_binding(tp_addr(), ids, 5, &rt);
}

// ---- cases
static void case_off(void) {
    world();
    write_cpu_textures();
    bind_all();
    present();
    RtPageStats s;
    CHECK(!rt_pages_stats(0, &s));   // no range was made
    printf("ok off\n");
}

// What is selected: the pages the CPU wrote, wholly inside the storage of one live bound texture, touched by no buffer and
// by no other texture; nothing before a completed bind; a render target and an unwritten texture never.
static void case_select(void) {
    if (no_write_watch()) return (void)printf("skip select\n");
    world();
    write_cpu_textures();
    present();
    RtPageStats s;
    CHECK(!rt_pages_stats(0, &s));   // not bound yet: no selection, so no range either
    RtMemStats before, after;
    rt_mem_measure(&before);
    gfx_null_hold(1);   // the submission is replayed and bound, not complete: its upload has not retired
    bind_all();
    present();
    CHECK(!rt_pages_stats(0, &s));
    gfx_null_hold(0);   // completed
    present();
    s = stats();
    uint64_t want = t1_expected() + (last_page(w.base + OFF_T5, tex_size(w.t5)) - first_page(w.base + OFF_T5)) / P4;
    printf("selected %llu expected %llu\n", (ull)s.released, (ull)want);
    CHECK(s.released == want && s.releases == want && s.recommits == 0 && s.busy_refusals == 0);
    rt_mem_measure(&after);
    CHECK(before.r[RT_MEM_HEAP].committed - after.r[RT_MEM_HEAP].committed == want * P4);
    present();   // nothing changes while nothing is written
    CHECK(stats().released == want);
    printf("ok select\n");
}

// OOT2_TEXHEAP_TRIGGER=register: the registration retires the upload, so the pages are selected with no bind at all.
static void case_register(void) {
    if (no_write_watch()) return (void)printf("skip register\n");
    world();   // every texture is registered here, before any CPU write
    write_cpu_textures();
    present();
    RtPageStats s = stats();
    uint64_t want = t1_expected() + (last_page(w.base + OFF_T5, tex_size(w.t5)) - first_page(w.base + OFF_T5)) / P4;
    printf("selected %llu expected %llu\n", (ull)s.released, (ull)want);
    CHECK(s.released == want);
    printf("ok register\n");
}

// A write to a released page recommits it and puts its texture back to unbound: the next completed bind selects it again.
static void case_rewrite(void) {
    if (no_write_watch()) return (void)printf("skip rewrite\n");
    world();
    write_cpu_textures();
    bind_all();
    present();
    uint64_t was = stats().released;
    gwrite(w.base + 0x2000, 0x55);   // a page of T1, released
    RtPageStats s = stats();
    CHECK(s.recommits == 1 && s.released == was - 1 && s.guest_writes == 1);
    present();
    CHECK(stats().released == was - 1);   // not bound since the write: not selected again
    bind_all();
    present();
    s = stats();
    printf("released %llu releases %llu\n", (ull)s.released, (ull)s.releases);
    CHECK(s.released == was && s.releases == was + 1);
    printf("ok rewrite\n");
}

// Two queues: the first is held at a wait before its bind, the second binds another texture and is whole. A held queue's
// records are replayed after later submissions' and queues complete independently, so a bind belongs to the submission it
// was made in, not to the newest one, and a completed submission completes itself and no earlier one. Completion is
// held in held_bind (the bind is replayed while both submissions are in flight, and each is completed by its fence value)
// and is not in held_complete (the second completes while the first is held).
static volatile uint32_t g_ev;
static uint64_t t5_pages(void) { return (last_page(w.base + OFF_T5, tex_size(w.t5)) - first_page(w.base + OFF_T5)) / P4; }
static void case_held_bind(void) {
    if (no_write_watch()) return (void)printf("skip held_bind\n");
    world();
    write_cpu_textures();
    static const uint32_t t1[] = {0}, t5[] = {4};
    gfx_null_hold(1);
    submit_on(1, tp_addr(), t1, 1, NULL, &g_ev);   // held at the wait, before the bind
    submit_on(2, tp_addr(), t5, 1, NULL, NULL);    // whole, its bind replayed first
    g_ev = 1;
    gfx_backend()->guest_written((uint64_t)(uintptr_t)&g_ev, 4);   // the first queue goes on: its bind is replayed last
    CHECK(gfx_null_parked() == 0);
    CHECK(gfx_null_complete(1) == 1);   // fence value 1 only
    present();
    RtPageStats s;
    uint64_t first = stats().released;
    printf("after-1 selected %llu expected %llu\n", (ull)first, (ull)t1_expected());
    CHECK(first == t1_expected());      // the bind made in submission 1 is submission 1's
    CHECK(gfx_null_complete(2) == 2);
    present();
    s = stats();
    printf("after-2 selected %llu expected %llu\n", (ull)s.released, (ull)(t1_expected() + t5_pages()));
    CHECK(s.released == t1_expected() + t5_pages());
    printf("ok held_bind\n");
}
static void case_held_complete(void) {
    if (no_write_watch()) return (void)printf("skip held_complete\n");
    world();
    write_cpu_textures();
    static const uint32_t t1[] = {0}, t5[] = {4};
    submit_on(1, tp_addr(), t1, 1, NULL, &g_ev);   // held at the wait
    submit_on(2, tp_addr(), t5, 1, NULL, NULL);    // whole and complete at once, while fence value 1 is held
    present();
    uint64_t first = stats().released;
    printf("while-held selected %llu expected %llu\n", (ull)first, (ull)t5_pages());
    CHECK(first == t5_pages());
    g_ev = 1;
    gfx_backend()->guest_written((uint64_t)(uintptr_t)&g_ev, 4);
    present();
    RtPageStats s = stats();
    printf("released selected %llu expected %llu\n", (ull)s.released, (ull)(t1_expected() + t5_pages()));
    CHECK(s.released == t1_expected() + t5_pages());
    printf("ok held_complete\n");
}

// Counting mode: a read of an armed page is counted as guest code's or the runtime's, the first of each texture is
// written to the file, and the page is restored.
static void case_count(void) {
    if (no_write_watch()) return (void)printf("skip count\n");
    world();
    write_cpu_textures();
    bind_all();
    present();
    RtPageStats s = stats();
    CHECK(s.mode == RT_PAGES_COUNTING && s.armed > 0 && s.released == 0);
    CHECK(gread(w.base + 0x2000) == 0xAB);   // T1 by guest code
    CHECK(gread(w.base + 0x3000) == 0xAB);   // T1 again: no second row
    CHECK(host_read32(w.base + OFF_T5 + 0x1000) == 0xAB);   // T5 by the runtime
    s = stats();
    CHECK(s.guest_reads == 2 && s.host_reads == 1);
    present();
    printf("ok count\n");
}

// Releasing mode: a read of a released page is the named fault, and the instruments say whose texture it was.
static void case_read_released(void) {
    if (no_write_watch()) return (void)printf("skip read_released\n");
    world();
    write_cpu_textures();
    bind_all();
    present();
    (void)gread(w.base + 0x2000);
    printf("ok read_released\n");   // not reached
}

// Writes per frame, and what the rule would select with no page range at all (page_release off).
static void case_frames(void) {
    if (no_write_watch()) return (void)printf("skip frames\n");
    world();
    bind_all();
    present();   // frame 1: nothing written since the pool began
    for (int i = 0; i < 7; i++) host_write32(w.base + 0x60000 + (uint64_t)i * 0x2000, 1);   // 7 pages outside every texture
    cpu_write(w.base + OFF_T1, w.s1);
    bind_all();   // the completion reads T1's marks: they count in the frame that is running
    present();    // frame 2
    present();    // frame 3: quiet
    RtPageStats s;
    CHECK(!rt_pages_stats(0, &s));
    uint64_t t1_first = (w.base + OFF_T1) & ~(P4 - 1), t1_end = (w.base + OFF_T1 + w.s1 + P4 - 1) & ~(P4 - 1);
    printf("t1_pages %llu\nt1_eligible %llu\n", (ull)((t1_end - t1_first) / P4), (ull)t1_expected());
    printf("ok frames\n");
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const char *cs = argv[1];
    setup(argv[2]);
    if (!strcmp(cs, "off")) case_off();
    else if (!strcmp(cs, "select")) case_select();
    else if (!strcmp(cs, "rewrite")) case_rewrite();
    else if (!strcmp(cs, "register")) case_register();
    else if (!strcmp(cs, "count")) case_count();
    else if (!strcmp(cs, "read_released")) case_read_released();
    else if (!strcmp(cs, "frames")) case_frames();
    else if (!strcmp(cs, "held_bind")) case_held_bind();
    else if (!strcmp(cs, "held_complete")) case_held_complete();
    else return 2;
    return 0;
}
