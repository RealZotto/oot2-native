// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the NVN2 front end (architecture D10, D14, section 2): the entry-point table, the bootstrap pair,
// the null back end, the device, memory pools, the GPU address window and buffers, driven by
// tests/test_runtime_nvn.py: `nvn_test <case> [arguments]`. Each case runs in its own process. A passing case prints
// "ok <case>"; a failing case must end in one named fatal, which the driver checks.
//
// Entry points are called the way generated code calls them: rt_call at their synthetic address, arguments in the
// AAPCS64 registers, and the objects and strings they read in guest memory (the heap of rt_mem_init). The guest module
// is one synthetic function, which rt_dispatch_init needs; nothing is read from a dump. The pool cases install a
// recording back end, a copy of the null back end that also counts what reaches it.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/gfx.h"
#include "oot2/memory.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
#define KiB (1ull << 10)
#define HEAP (16 * MiB)
#define PAGE GFX_GPU_PAGE

// NVNmemoryPoolFlags
#define CPU_NO_ACCESS 0x1u
#define CPU_UNCACHED 0x2u
#define GPU_NO_ACCESS 0x8u
#define GPU_CACHED 0x20u
#define PHYSICAL 0x100u
#define VIRTUAL 0x200u

static Cpu c;
static uint64_t heap, heap_used;
static uint64_t loader;   // nvnBootstrapLoader, bound as the module loader binds main's import

static void fake_fn(Cpu *cpu) { (void)cpu; }

// ---- the recording back end: the null back end, plus what the pool and buffer entry points hand it
static struct {
    int inits, finis, maps, written, read, refuse_next;
    uint64_t addr, size;   // the newest guest_written or guest_read range
    int map_n;
    const pool_t *map_pool;
} rec;
static int rec_pool_init(pool_t *p) {
    (void)p;
    rec.inits++;
    if (rec.refuse_next) return rec.refuse_next = 0, 0;
    return 1;
}
static void rec_pool_fini(pool_t *p) { (void)p, rec.finis++; }
static int rec_pool_map_virtual(pool_t *v, int n, const void *requests) {
    (void)requests;
    rec.maps++, rec.map_n = n, rec.map_pool = v;
    return 1;
}
static void rec_guest_written(uint64_t addr, uint64_t size) { rec.written++, rec.addr = addr, rec.size = size; }
static void rec_guest_read(uint64_t addr, uint64_t size) { rec.read++, rec.addr = addr, rec.size = size; }
static GfxBackend recording;

// Guest memory, an empty program and the host-function registry sealed; register_nvn says whether start-up gave the
// entry points their addresses first, use_rec whether the recording back end replaces the null one.
static void setup_with(int register_nvn, int use_rec) {
    RtMemConfig cfg = {.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    c.x[0] = 0, c.x[1] = HEAP;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    if (use_rec) {
        recording = gfx_null_backend;
        recording.name = "recording";
        recording.pool_init = rec_pool_init, recording.pool_fini = rec_pool_fini;
        recording.pool_map_virtual = rec_pool_map_virtual;
        recording.guest_written = rec_guest_written, recording.guest_read = rec_guest_read;
        gfx_set_backend(&recording);
    }
    if (register_nvn) nvn_procs_register();
    static const uint64_t addrs[] = {MOD_MAIN_BASE + 0x100};
    static const GuestFn fns[] = {fake_fn};
    static const RtModule mods[] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, addrs, fns, 1, TEST_END}};
    rt_dispatch_init(mods, 1);
    c.excl_addr = ~0ull;
}
static void setup(int register_nvn) { setup_with(register_nvn, 0); }

// Guest memory for an object, a string or a pool's storage, zeroed, as the game's allocator would give it.
static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}

// A string in guest memory, as the game passes one.
static uint64_t guest_str(const char *s) {
    size_t n = strlen(s) + 1;
    uint64_t a = galloc(n, 16);
    memcpy((void *)(uintptr_t)a, s, n);
    return a;
}

static uint64_t call(uint64_t fn, uint64_t x0, uint64_t x1) {
    c.x[0] = x0, c.x[1] = x1;
    rt_call(&c, fn);
    return c.x[0];
}

// The game's way in: the import nvnBootstrapLoader, asked for nvnDeviceGetProcAddress.
static uint64_t bootstrap(void) { return loader; }

static uint64_t get_proc(const char *name) {
    static uint64_t gpa;
    if (!gpa) gpa = call(bootstrap(), guest_str("nvnDeviceGetProcAddress"), 0);
    CHECK(gpa != 0);
    return call(gpa, 0, name ? guest_str(name) : 0);
}

// An entry point asked for once, as the game keeps the answers in its own table.
static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[96];
    static int n;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    CHECK(n < 96);
    uint64_t a = get_proc(name);
    CHECK(a != 0);
    cache[n].name = name, cache[n].addr = a;
    return cache[n++].addr;
}

// NVN(nvnX, arguments...): call entry point nvnX with up to eight integer or pointer arguments, as x0 to x7.
static uint64_t nvn_vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}
#define NVN(name, ...)                                                                                               \
    nvn_vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t))

// ---- objects in guest memory, made through the entry points
static uint64_t device(void) {
    uint64_t b = galloc(0x40, 16), d = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, b);
    NVN(nvnDeviceBuilderSetFlags, b, 0);
    CHECK((NVN(nvnDeviceInitialize, d, b) & 0xff) == 1);
    return d;
}

// MemoryPoolInitialize's answer for a pool object at o with this storage, size and flags.
static int pool_init_at(uint64_t o, uint64_t storage, uint64_t size, uint32_t flags) {
    static uint64_t dev;
    if (!dev) dev = device();
    uint64_t b = galloc(0x40, 16);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetDevice, b, dev);
    NVN(nvnMemoryPoolBuilderSetStorage, b, storage, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, flags);
    CHECK(NVN(nvnMemoryPoolBuilderGetDevice, b) == dev && NVN(nvnMemoryPoolBuilderGetMemory, b) == storage);
    CHECK(NVN(nvnMemoryPoolBuilderGetSize, b) == size && NVN(nvnMemoryPoolBuilderGetFlags, b) == flags);
    return (int)(NVN(nvnMemoryPoolInitialize, o, b) & 0xff);
}
static uint64_t pool(uint64_t storage, uint64_t size, uint32_t flags) {
    uint64_t o = galloc(0x100, 16);
    CHECK(pool_init_at(o, storage, size, flags) == 1);
    return o;
}
static uint64_t pool_addr(uint64_t p) { return NVN(nvnMemoryPoolGetBufferAddress, p); }

static uint64_t buffer(uint64_t p, uint64_t off, uint64_t size) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x30, 16);
    NVN(nvnBufferBuilderSetDevice, b, 0x1234);
    NVN(nvnBufferBuilderSetDefaults, b);
    NVN(nvnBufferBuilderSetStorage, b, p, off, size);
    CHECK(NVN(nvnBufferBuilderGetDevice, b) == 0x1234 && NVN(nvnBufferBuilderGetMemoryPool, b) == p);
    CHECK(NVN(nvnBufferBuilderGetMemoryOffset, b) == off && NVN(nvnBufferBuilderGetSize, b) == size);
    CHECK((NVN(nvnBufferInitialize, o, b) & 0xff) == 1);
    return o;
}

// NVNmappingRequest, 0x28 bytes
typedef struct { uint64_t pool; int64_t poff, voff; uint64_t size; int32_t storage_class, pad; } MapReq;
_Static_assert(sizeof(MapReq) == 0x28, "NVNmappingRequest stride");
static uint64_t map_virtual(uint64_t v, int n, const MapReq *r) {
    uint64_t a = galloc(sizeof *r * (uint64_t)n, 16);
    memcpy((void *)(uintptr_t)a, r, sizeof *r * (uint64_t)n);
    return NVN(nvnMemoryPoolMapVirtual, v, (uint64_t)n, a) & 0xff;
}

// ---- passing cases of the seam

// Every listed name resolves to its own address, named after it, through GetProcAddress and the bootstrap alike. The
// driver compares the printed names with nvn_procs.inc.
static void case_table(void) {
    setup(1);
    static uint8_t used[RT_HOSTFN_SIZE / 16];
    int n = nvn_proc_count(), bodies = 0;
    CHECK(n > 0 && nvn_proc_name(n) == NULL && nvn_proc_name(-1) == NULL);
    for (int i = 0; i < n; i++) {
        const char *name = nvn_proc_name(i);
        uint64_t a = get_proc(name);
        CHECK(a != 0 && a == nvn_proc_address(name) && a == call(bootstrap(), guest_str(name), 0));
        CHECK(a - RT_HOSTFN_BASE < RT_HOSTFN_SIZE && (a & 15) == 0);
        CHECK(!used[(a - RT_HOSTFN_BASE) / 16]);
        used[(a - RT_HOSTFN_BASE) / 16] = 1;
        const char *named = rt_hostfn_name(a);
        CHECK(named && !strcmp(named, name));
        bodies += nvn_proc_has_body(i);
        printf("name %s %d\n", name, nvn_proc_has_body(i));
    }
    printf("ok table %d %d\n", n, bodies);
}

static void case_unknown(void) {
    setup(1);
    CHECK(get_proc("nvnNotAName") == 0);
    CHECK(get_proc("nvnNotAName") == 0);
    CHECK(get_proc("nvnDeviceGetProcAddresss") == 0);
    CHECK(get_proc(NULL) == 0);
    CHECK(call(bootstrap(), guest_str("nvnNotAName"), 0) == 0);
    printf("ok unknown\n");
}

static void case_unknown_many(void) {
    setup(1);
    char name[32];
    for (int i = 0; i < 40; i++) {
        snprintf(name, sizeof name, "nvnUnknown%02d", i);
        CHECK(get_proc(name) == 0);
        CHECK(get_proc(name) == 0);
    }
    printf("ok unknown_many\n");
}

// The null back end: installed by default, each submission done when it returns, fence values in submission order.
static void case_null_backend(void) {
    setup(1);
    const GfxBackend *b = gfx_backend();
    CHECK(b == &gfx_null_backend && !strcmp(b->name, "null"));
    CHECK(b->fence_done() == 0 && b->fence_wait(0, 0) == 1 && b->fence_wait(1, 0) == 0);
    for (uint64_t k = 1; k <= 3; k++) {
        CHECK(b->submit(1, NULL, 0) == k);
        CHECK(b->fence_done() == k);
        CHECK(b->fence_wait(k, 0) == 1 && b->fence_wait(k + 1, 0) == 0);
    }
    const int crop[4] = {0, 0, 1280, 720};
    b->present(NULL, crop);
    b->present(NULL, crop);
    CHECK(gfx_null_presents() == 2);
    uint8_t texels[2][64];
    memset(texels, 0xAB, sizeof texels);
    const NVNcopyRegion r = {0, 0, 0, 4, 2, 2};
    b->read_texels(NULL, NULL, &r, texels[0], 16, 64);   // two images of two 16-byte rows
    for (int z = 0; z < 2; z++)
        for (int i = 0; i < 64; i++) CHECK(texels[z][i] == (i < 32 ? 0 : 0xAB));
    b->finish(0);
    CHECK(b->pool_init(NULL) == 1);
    printf("ok null_backend\n");
}

// ---- the device

// The builder, initialize and finalize, the modes the back end reads, and the calls the game makes once.
static void case_device(void) {
    setup(1);
    uint64_t b = galloc(0x40, 16), d = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, b);
    CHECK(NVN(nvnDeviceBuilderGetFlags, b) == 0);
    NVN(nvnDeviceBuilderSetFlags, b, 0x100);
    CHECK(NVN(nvnDeviceBuilderGetFlags, b) == 0x100);
    CHECK((NVN(nvnDeviceInitialize, d, b) & 0xff) == 1);
    CHECK(NVN(nvnDeviceGetFlags, d) == 0x100);
    CHECK(nvn_depth_mode() == 0 && nvn_window_origin_mode() == 0);
    CHECK(NVN(nvnDeviceGetDepthMode, d) == 0 && NVN(nvnDeviceGetWindowOriginMode, d) == 0);
    NVN(nvnDeviceSetDepthMode, d, 1);
    NVN(nvnDeviceSetWindowOriginMode, d, 1);
    CHECK(nvn_depth_mode() == 1 && nvn_window_origin_mode() == 1);
    CHECK(NVN(nvnDeviceGetDepthMode, d) == 1 && NVN(nvnDeviceGetWindowOriginMode, d) == 1);
    NVN(nvnDeviceInstallDebugCallback, d, 0x5678, 0x9abc, 1);
    NVN(nvnDeviceSetDebugLabel, d, guest_str("device"));
    CHECK((NVN(nvnDeviceRegisterFastClearColor, d, galloc(16, 16), 0x25) & 0xff) == 0);
    NVN(nvnDeviceFinalize, d);
    CHECK(*(uint32_t *)(uintptr_t)d == 0);
    printf("ok device\n");
}

// nvnDeviceGetInteger for each item given (hexadecimal), each asked twice: "item <n> <value>". The driver holds the
// table of the console's answers and their sources.
static void case_devinfo(int argc, char **argv) {
    setup(1);
    uint64_t d = device(), out = galloc(4, 4);
    for (int i = 0; i < argc; i++) {
        uint64_t w = strtoull(argv[i], NULL, 16);
        int32_t v[2];
        for (int k = 0; k < 2; k++) {
            *(int32_t *)(uintptr_t)out = -12345;
            NVN(nvnDeviceGetInteger, d, w, out);
            v[k] = *(int32_t *)(uintptr_t)out;
        }
        CHECK(v[0] == v[1]);
        printf("item %#llx %d\n", (unsigned long long)w, v[0]);
    }
    printf("ok devinfo\n");
}

// ---- pools, buffers and the GPU address window

// A buffer in a pool the CPU can see: its address is the guest address of the pool's storage plus its offset, Map
// gives the same pointer, and flushes and invalidations reach the back end with that guest range.
static void case_cpu_visible(void) {
    setup_with(1, 1);
    uint64_t size = 256 * KiB, storage = galloc(size, 4096);
    uint64_t p = pool(storage, size, CPU_UNCACHED | GPU_CACHED);
    CHECK(rec.inits == 1);
    CHECK(pool_addr(p) == storage && NVN(nvnMemoryPoolMap, p) == storage);
    CHECK(NVN(nvnMemoryPoolGetFlags, p) == (CPU_UNCACHED | GPU_CACHED) && NVN(nvnMemoryPoolGetSize, p) == size);
    NVN(nvnMemoryPoolSetDebugLabel, p, guest_str("pool"));
    uint64_t b = buffer(p, 0x1000, 0x2000);
    CHECK(NVN(nvnBufferGetAddress, b) == storage + 0x1000 && NVN(nvnBufferMap, b) == storage + 0x1000);
    CHECK(NVN(nvnBufferGetMemoryPool, b) == p && NVN(nvnBufferGetMemoryOffset, b) == 0x1000);
    CHECK(NVN(nvnBufferGetSize, b) == 0x2000);
    NVN(nvnBufferSetDebugLabel, b, guest_str("buffer"));
    NVN(nvnBufferFlushMappedRange, b, 0x10, 0x20);
    CHECK(rec.written == 1 && rec.addr == storage + 0x1010 && rec.size == 0x20);
    NVN(nvnBufferInvalidateMappedRange, b, 0, 0x2000);
    CHECK(rec.read == 1 && rec.addr == storage + 0x1000 && rec.size == 0x2000);
    NVN(nvnMemoryPoolFlushMappedRange, p, 0x100, 0x10);
    CHECK(rec.written == 2 && rec.addr == storage + 0x100 && rec.size == 0x10);
    NVN(nvnMemoryPoolInvalidateMappedRange, p, size - 0x1000, 0x10000);   // clipped to the pool
    CHECK(rec.read == 2 && rec.addr == storage + size - 0x1000 && rec.size == 0x1000);
    // A range outside its pool, or a pool that is not initialised, fails BufferInitialize.
    uint64_t bb = galloc(0x40, 16), o = galloc(0x30, 16), raw = galloc(0x100, 16);
    NVN(nvnBufferBuilderSetDefaults, bb);
    NVN(nvnBufferBuilderSetStorage, bb, p, size - 0x1000, 0x1001);
    CHECK((NVN(nvnBufferInitialize, o, bb) & 0xff) == 0);
    NVN(nvnBufferBuilderSetStorage, bb, raw, 0, 0x10);
    CHECK((NVN(nvnBufferInitialize, o, bb) & 0xff) == 0);
    NVN(nvnBufferFinalize, b);
    CHECK(NVN(nvnBufferGetAddress, b) == 0);
    NVN(nvnMemoryPoolFinalize, p);
    CHECK(rec.finis == 1 && NVN(nvnMemoryPoolMap, p) == 0);
    printf("ok cpu_visible\n");
}

// Pools the CPU cannot see get distinct 64 KiB aligned ranges of the GPU window, past the oracle's 1,024, and a freed
// range goes to the next pool that fits: two adjacent freed ranges merge, and the lowest fitting range is taken.
#define NWIN 1100
static uint64_t win_size(int i) { return i % 3 == 0 ? 64 * KiB : i % 3 == 1 ? 4 * KiB : 100 * KiB; }
static uint64_t win_span(int i) { return (win_size(i) + PAGE - 1) & ~(PAGE - 1); }
static int by_addr(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}
static void case_window(void) {
    setup_with(1, 1);
    uint64_t storage = galloc(128 * KiB, 4096);   // given to every pool, never touched for one the CPU cannot see
    static uint64_t pools[NWIN], addr[NWIN], ranges[NWIN][2];
    uint64_t top = 0;
    for (int i = 0; i < NWIN; i++) {
        pools[i] = pool(storage, win_size(i), CPU_NO_ACCESS | GPU_CACHED);
        addr[i] = pool_addr(pools[i]);
        CHECK(gfx_gpu_only(addr[i]) && addr[i] % PAGE == 0);
        CHECK(NVN(nvnMemoryPoolMap, pools[i]) == 0);
        ranges[i][0] = addr[i], ranges[i][1] = addr[i] + win_span(i);
        if (ranges[i][1] > top) top = ranges[i][1];
    }
    qsort(ranges, NWIN, sizeof ranges[0], by_addr);
    for (int i = 1; i < NWIN; i++) CHECK(ranges[i - 1][1] <= ranges[i][0]);
    uint64_t sum_size = 0, sum_span = 0;
    for (int i = 0; i < NWIN; i++) sum_size += win_size(i), sum_span += win_span(i);
    NvnPoolStats ps;
    nvn_pool_stats(&ps);
    CHECK(ps.all.created == NWIN && ps.all.live == NWIN && ps.all.bytes == sum_size && ps.all.peak_bytes == sum_size);
    CHECK(ps.cls[NVN_POOL_CPU_NO_ACCESS].created == NWIN && ps.cls[NVN_POOL_CPU].created == 0);
    CHECK(ps.window_bytes == sum_span && ps.window_peak == sum_span);
    // A buffer in one of them: a window address, with no CPU view.
    uint64_t b = buffer(pools[1050], 0x100, 0x200);
    CHECK(NVN(nvnBufferGetAddress, b) == addr[1050] + 0x100 && NVN(nvnBufferMap, b) == 0);
    // Pools 9 and 10 are adjacent (64 KiB each in the window), 500 holds 128 KiB.
    CHECK(addr[10] == addr[9] + PAGE && win_span(500) == 2 * PAGE);
    int finis = rec.finis;
    NVN(nvnMemoryPoolFinalize, pools[9]);
    NVN(nvnMemoryPoolFinalize, pools[10]);
    NVN(nvnMemoryPoolFinalize, pools[500]);
    CHECK(rec.finis == finis + 3);
    CHECK(pool_addr(pool(storage, 128 * KiB, CPU_NO_ACCESS | GPU_CACHED)) == addr[9]);
    CHECK(pool_addr(pool(storage, 100 * KiB, CPU_NO_ACCESS | GPU_CACHED)) == addr[500]);
    CHECK(pool_addr(pool(storage, 64 * KiB, CPU_NO_ACCESS | GPU_CACHED)) == top);
    // three pools ended (64 KiB, 4 KiB, 100 KiB; 4 pages of window) and three made (128 KiB, 100 KiB, 64 KiB; 5 pages)
    nvn_pool_stats(&ps);
    CHECK(ps.all.created == NWIN + 3 && ps.all.live == NWIN && ps.all.peak == NWIN);
    CHECK(ps.all.bytes == sum_size - (64 + 4 + 100) * KiB + (128 + 100 + 64) * KiB);
    CHECK(ps.window_bytes == sum_span + PAGE && ps.window_peak == sum_span + PAGE);
    printf("ok window %d\n", NWIN);
}

// A virtual pool: a window range from creation, mapped from one physical base in two requests, unmapped, and mapped
// again from another base once no page is left on the first. Malformed pools and requests are refused.
static void case_virtual(void) {
    setup_with(1, 1);
    uint64_t pstore = galloc(4 * PAGE, PAGE);
    uint64_t phys = pool(pstore, 4 * PAGE, CPU_NO_ACCESS | GPU_CACHED | PHYSICAL);
    uint64_t v = pool(0, 8 * PAGE, CPU_NO_ACCESS | GPU_CACHED | VIRTUAL), va = pool_addr(v);
    CHECK(gfx_gpu_only(va) && va % PAGE == 0 && gfx_gpu_only(pool_addr(phys)));
    // the census counters: one physical pool of 4 pages, one virtual pool of 8, both with a window range
    NvnPoolStats ps;
    nvn_pool_stats(&ps);
    CHECK(ps.all.created == 2 && ps.all.live == 2 && ps.all.peak == 2 && ps.all.bytes == 12 * PAGE &&
          ps.all.peak_bytes == 12 * PAGE);
    CHECK(ps.cls[NVN_POOL_PHYSICAL].created == 1 && ps.cls[NVN_POOL_PHYSICAL].bytes == 4 * PAGE);
    CHECK(ps.cls[NVN_POOL_VIRTUAL].created == 1 && ps.cls[NVN_POOL_VIRTUAL].bytes == 8 * PAGE);
    CHECK(ps.cls[NVN_POOL_CPU].created == 0 && ps.cls[NVN_POOL_CPU_NO_ACCESS].created == 0 && ps.gpu_no_access == 0);
    CHECK(ps.window_bytes == 12 * PAGE && ps.window_peak == 12 * PAGE);
    MapReq two[2] = {{phys, 0, 0, 2 * PAGE, 0, 0}, {phys, 2 * PAGE, 2 * PAGE, 2 * PAGE, 0, 0}};
    CHECK(map_virtual(v, 2, two) == 1);
    CHECK(rec.maps == 1 && rec.map_n == 2 && rec.map_pool == (const pool_t *)(uintptr_t)v);
    uint64_t b = buffer(v, PAGE + 0x80, 0x100);
    CHECK(NVN(nvnBufferGetAddress, b) == va + PAGE + 0x80 && NVN(nvnBufferMap, b) == 0);
    CHECK(nvn_pool_find(va + PAGE + 0x80, 0x100) == (const pool_t *)(uintptr_t)v);
    MapReq unmap = {0, 0, 0, 4 * PAGE, 0, 0}, again = {phys, 0, 4 * PAGE, 2 * PAGE, 0, 0};
    CHECK(map_virtual(v, 1, &unmap) == 1 && map_virtual(v, 1, &again) == 1 && rec.maps == 3);
    // Refused: a request off the 64 KiB grid, past the pool, from a pool that is not physical; a map of a pool that is
    // not virtual; a virtual pool with storage, one the CPU can see, one both physical and virtual, a physical pool
    // that is not made of whole pages.
    MapReq odd = {phys, 0, 0x8000, PAGE, 0, 0}, past = {phys, 0, 8 * PAGE, PAGE, 0, 0}, notphys = {v, 0, 0, PAGE, 0, 0};
    CHECK(map_virtual(v, 1, &odd) == 0 && map_virtual(v, 1, &past) == 0 && map_virtual(v, 1, &notphys) == 0);
    CHECK(map_virtual(phys, 1, &again) == 0 && rec.maps == 3);
    uint64_t o = galloc(0x100, 16);
    CHECK(pool_init_at(o, pstore, 4 * PAGE, CPU_NO_ACCESS | VIRTUAL) == 0);
    CHECK(pool_init_at(o, 0, 4 * PAGE, CPU_UNCACHED | VIRTUAL) == 0);
    CHECK(pool_init_at(o, pstore, 4 * PAGE, CPU_NO_ACCESS | PHYSICAL | VIRTUAL) == 0);
    CHECK(pool_init_at(o, pstore, PAGE + 0x1000, CPU_NO_ACCESS | PHYSICAL) == 0);
    // A pool the back end refuses fails, and its range goes back to the window: the next pool takes it.
    uint64_t probe = pool(pstore, PAGE, CPU_NO_ACCESS | GPU_CACHED), at = pool_addr(probe);
    NVN(nvnMemoryPoolFinalize, probe);
    rec.refuse_next = 1;
    CHECK(pool_init_at(o, pstore, PAGE, CPU_NO_ACCESS | GPU_CACHED) == 0);
    CHECK(pool_addr(pool(pstore, PAGE, CPU_NO_ACCESS | GPU_CACHED)) == at);
    // four pools made (the probe among them, the refused one not), three live, never more than three, 13 pages
    nvn_pool_stats(&ps);
    CHECK(ps.all.created == 4 && ps.all.live == 3 && ps.all.peak == 3 && ps.all.bytes == 13 * PAGE &&
          ps.all.peak_bytes == 13 * PAGE);
    CHECK(ps.cls[NVN_POOL_CPU_NO_ACCESS].created == 2 && ps.cls[NVN_POOL_CPU_NO_ACCESS].live == 1 &&
          ps.cls[NVN_POOL_CPU_NO_ACCESS].bytes == PAGE && ps.cls[NVN_POOL_CPU_NO_ACCESS].peak == 1);
    CHECK(ps.window_bytes == 13 * PAGE && ps.window_peak == 13 * PAGE);
    NVN(nvnMemoryPoolFinalize, v);
    NVN(nvnMemoryPoolFinalize, phys);
    nvn_pool_stats(&ps);
    CHECK(ps.all.created == 4 && ps.all.live == 1 && ps.all.peak == 3 && ps.all.bytes == PAGE &&
          ps.all.peak_bytes == 13 * PAGE);
    CHECK(ps.cls[NVN_POOL_VIRTUAL].live == 0 && ps.cls[NVN_POOL_VIRTUAL].bytes == 0 &&
          ps.cls[NVN_POOL_VIRTUAL].peak_bytes == 8 * PAGE);
    CHECK(ps.window_bytes == PAGE && ps.window_peak == 13 * PAGE);
    printf("ok virtual\n");
}

// The pool index: an address inside a pool the GPU can reach finds that pool, from its first byte to its last; an
// address outside every pool, a range that runs past its pool, a pool the GPU cannot reach and a finalised pool find
// nothing.
static void case_lookup(void) {
    setup_with(1, 1);
    uint64_t storage = galloc(512 * KiB, PAGE);
    uint64_t a = pool(storage, 192 * KiB, CPU_UNCACHED | GPU_CACHED);
    pool(storage + 256 * KiB, 64 * KiB, CPU_UNCACHED | GPU_NO_ACCESS);
    uint64_t w = pool(storage, 96 * KiB, CPU_NO_ACCESS | GPU_CACHED), wa = pool_addr(w);
    uint64_t v = pool(0, 4 * PAGE, CPU_NO_ACCESS | GPU_CACHED | VIRTUAL), va = pool_addr(v);
    const pool_t *A = (const pool_t *)(uintptr_t)a, *W = (const pool_t *)(uintptr_t)w;
    const pool_t *V = (const pool_t *)(uintptr_t)v;
    NvnPoolStats ps;
    nvn_pool_stats(&ps);   // two CPU-visible pools (one of them GPU_NO_ACCESS), one CPU_NO_ACCESS, one virtual
    CHECK(ps.all.created == 4 && ps.all.live == 4 && ps.all.bytes == (192 + 64 + 96) * KiB + 4 * PAGE);
    CHECK(ps.cls[NVN_POOL_CPU].created == 2 && ps.cls[NVN_POOL_CPU].bytes == (192 + 64) * KiB);
    CHECK(ps.cls[NVN_POOL_CPU_NO_ACCESS].created == 1 && ps.cls[NVN_POOL_CPU_NO_ACCESS].bytes == 96 * KiB);
    CHECK(ps.cls[NVN_POOL_VIRTUAL].created == 1 && ps.cls[NVN_POOL_PHYSICAL].created == 0 && ps.gpu_no_access == 1);
    CHECK(ps.window_bytes == 2 * PAGE + 4 * PAGE);   // the 96 KiB pool's range is two pages, the virtual pool's four
    CHECK(nvn_pool_find(storage, 0) == A && nvn_pool_find(storage + 0x1234, 16) == A);
    CHECK(nvn_pool_find(storage + 192 * KiB - 1, 1) == A && nvn_pool_find(storage + 192 * KiB - 16, 16) == A);
    CHECK(nvn_pool_find(storage + 192 * KiB, 1) == NULL && nvn_pool_find(storage + 192 * KiB - 8, 16) == NULL);
    CHECK(nvn_pool_find(storage + 256 * KiB, 1) == NULL);   // the GPU cannot reach that pool
    CHECK(nvn_pool_find(wa, 1) == W && nvn_pool_find(wa + 96 * KiB - 1, 1) == W);
    CHECK(nvn_pool_find(wa + 96 * KiB, 1) == NULL);   // the rest of its 128 KiB range is not the pool's
    CHECK(nvn_pool_find(va + 3 * PAGE, 4) == V && nvn_pool_find(0x1000, 1) == NULL);
    NVN(nvnMemoryPoolFinalize, w);
    CHECK(nvn_pool_find(wa, 1) == NULL && nvn_pool_find(storage, 1) == A);
    nvn_pool_stats(&ps);
    CHECK(ps.all.live == 3 && ps.all.peak == 4 && ps.cls[NVN_POOL_CPU_NO_ACCESS].live == 0 && ps.window_bytes == 4 * PAGE);
    printf("ok lookup\n");
}

// The cost of BufferInitialize, BufferGetAddress and BufferFinalize through the dispatch, as the game calls them:
// "bench init <ns> address <ns> finalize <ns> calls <n>", the mean over n calls each.
static void case_bench(int n) {
    setup(1);
    CHECK(n > 0);
    uint64_t storage = galloc(64 * KiB, 4096), p = pool(storage, 64 * KiB, CPU_UNCACHED | GPU_CACHED);
    uint64_t bb = galloc(0x40, 16), o = galloc(0x30, 16);
    NVN(nvnBufferBuilderSetDefaults, bb);
    NVN(nvnBufferBuilderSetStorage, bb, p, 0x100, 0x100);
    uint64_t init = proc("nvnBufferInitialize"), addr = proc("nvnBufferGetAddress"), fini = proc("nvnBufferFinalize");
    double ns[3];
    uint64_t t = plat_now_ns(), sum = 0;
    for (int i = 0; i < n; i++) sum += call(init, o, bb) & 0xff;
    ns[0] = (double)(plat_now_ns() - t) / n;
    t = plat_now_ns();
    for (int i = 0; i < n; i++) sum += call(addr, o, 0);
    ns[1] = (double)(plat_now_ns() - t) / n;
    t = plat_now_ns();
    for (int i = 0; i < n; i++) call(fini, o, 0);
    ns[2] = (double)(plat_now_ns() - t) / n;
    CHECK(sum == (uint64_t)n + (uint64_t)n * (storage + 0x100));
    printf("bench init %.1f address %.1f finalize %.1f calls %d\n", ns[0], ns[1], ns[2], n);
    printf("ok bench\n");
}

// ---- failing cases

static void case_unbodied(void) {
    setup(1);
    uint64_t a = get_proc("nvnSyncInitializeFromFencedGLSync");   // GL interop: no OoT call, no body
    CHECK(a != 0);
    call(a, 0, 0);
}

static void case_bootstrap_unregistered(void) {
    setup(0);
    call(bootstrap(), guest_str("nvnDeviceGetProcAddress"), 0);
}

static void case_register_late(void) {
    setup(0);
    nvn_procs_register();
}

static GfxBackend partial;
static void case_backend_partial(void) {
    partial = gfx_null_backend;
    partial.name = "partial";
    partial.fence_done = NULL;
    gfx_set_backend(&partial);
}

static void case_backend_late(void) {
    setup(1);
    gfx_set_backend(&gfx_null_backend);
}

// A virtual pool whose pages would lie in two physical runs: its first page from the physical pool's first, its second
// from the physical pool's third.
static void case_two_runs(void) {
    setup_with(1, 1);
    uint64_t phys = pool(galloc(4 * PAGE, PAGE), 4 * PAGE, CPU_NO_ACCESS | GPU_CACHED | PHYSICAL);
    uint64_t v = pool(0, 4 * PAGE, CPU_NO_ACCESS | GPU_CACHED | VIRTUAL);
    MapReq r[2] = {{phys, 0, 0, PAGE, 0, 0}, {phys, 2 * PAGE, PAGE, PAGE, 0, 0}};
    map_virtual(v, 2, r);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: nvn_test <case> [arguments]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "table")) case_table();
    else if (!strcmp(cs, "unknown")) case_unknown();
    else if (!strcmp(cs, "unknown_many")) case_unknown_many();
    else if (!strcmp(cs, "null_backend")) case_null_backend();
    else if (!strcmp(cs, "device")) case_device();
    else if (!strcmp(cs, "devinfo")) case_devinfo(argc - 2, argv + 2);
    else if (!strcmp(cs, "cpu_visible")) case_cpu_visible();
    else if (!strcmp(cs, "window")) case_window();
    else if (!strcmp(cs, "virtual")) case_virtual();
    else if (!strcmp(cs, "lookup")) case_lookup();
    else if (!strcmp(cs, "bench")) case_bench(argc > 2 ? atoi(argv[2]) : 1000);
    else if (!strcmp(cs, "unbodied")) case_unbodied();
    else if (!strcmp(cs, "bootstrap_unregistered")) case_bootstrap_unregistered();
    else if (!strcmp(cs, "register_late")) case_register_late();
    else if (!strcmp(cs, "backend_partial")) case_backend_partial();
    else if (!strcmp(cs, "backend_late")) case_backend_late();
    else if (!strcmp(cs, "two_runs")) case_two_runs();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
