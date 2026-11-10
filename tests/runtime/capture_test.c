// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The M1 capture's writer (runtime/nvn/nvn_capture.c, oot2/capture.h), driven by tests/test_runtime_capture.py:
// `capture_test <case> <folder>`. Each case runs in its own process, writes its capture into folder when the
// environment turns the capture on, and prints "ok <case>" when it ends by itself; the Python side reads the file.
//
// The game is a guest thread written by hand as a host function, as tests/runtime/nvn_pace_test.c writes it: it acquires,
// submits nothing (QueueSubmitCommands with no handle), and presents, through the NVN entry points' synthetic addresses.
// No game code, no dump.
//
// frames    the clock is driven tick by tick. A worker on the second virtual core burns 30 ms of its own CPU between
//           acquire and present of frames 4 and 7 (presents 5 and 8), and sleeps 5 ms at a time otherwise (a sleep of
//           1 ms or less spins in the kernel and would burn CPU): the capture must put the burn in the frame lines of
//           presents 5 and 8, not 6 and 9.
// gap      as frames, but the worker holds its virtual core for 30 ms in frames 4 and 7 without running (a sleep of the host
//           thread, which keeps the core): its core time is 30 ms and its CPU time near 0, so the "gap" line of the
//           worker, the core time less the CPU time summed over the frames, shows the two holds.
// stop70, stop71, stop72, exitprocess, census73
//           three presents on the host's clock, then a fatal, a named fatal, a guest stop, ExitProcess, or the census
//           frame limit: the closing record must still be the file's last line.
// length    the capture ends the run 2.2 s after its title mark (present 2), with exit code 0; a pipeline miss and a GPU
//           pass are counted at present 5; memory lines come once a second.
// off       no OOT2_CAPTURE: no file, and no frame boundary marked (every thread's frame time stays 0).
// shown     a back end that knows each present's shown tick two presents later: the capture writes it as it comes.
// cost      600 presents, 32 sleeping guest threads beside the game: the CPU time of the present call, with the
//           capture on or off as the environment says.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/capture.h"
#include "oot2/gfx.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/nvn_pace.h"
#include "oot2/nvn_tex.h"
#include "oot2/vsync.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

typedef unsigned long long ull;
#define MiB (1ull << 20)
#define HEAP_BYTES (64 * MiB)
#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))   // the guest address of guest function i
#define PC (MB + 0x4000)             // the guest pc the guest threads' calls report
#define LOW32(v) ((v) & 0xffffffffu)
#define MS 1000000ull

// The schedule, written here without vsync.c: 31,250,000 counts a second of 32 ns, tick n due at ceil(n * 31.25e6 / 60).
#define ORIGIN 123456789012ull
static uint64_t ref_due_ns(uint64_t n) { return ORIGIN + ((n * 31250000ull + 59) / 60) * 32ull; }

static Cpu c;   // the host thread's Cpu: entry points called before the run
static uint64_t heap, heap_used, loader, g_area;
static uint64_t g_dev, g_queue, g_win;
static uint64_t f_acquire, f_present, f_submit;
static const char *g_folder;

// ---------- the driven time
static uint64_t g_now;   // atomic
static uint64_t drv_now(void *ctx) {
    (void)ctx;
    return __atomic_load_n(&g_now, __ATOMIC_ACQUIRE);
}
static void drv_wait(void *ctx, uint64_t deadline) {   // the clock thread never starts in these cases
    (void)ctx, (void)deadline;
    fprintf(stderr, "FAIL: the clock thread waited\n");
    exit(1);
}
static void set_now(uint64_t ns) { __atomic_store_n(&g_now, ns, __ATOMIC_RELEASE); }
static void drive_clock(void) {
    g_now = ref_due_ns(0);
    RtVsyncDriver d = {.origin_ns = ORIGIN, .now = drv_now, .wait_until = drv_wait};
    rt_vsync_drive(&d);
}

// ---------- the guest side
enum { IDX = 0x0, HANDLES = 0x100, STACKS = 0x100000, STACK_BYTES = 0x4000, AREA = 0x200000 };
#define W(off) (g_area + (off))
static inline uint64_t sv(Cpu *t, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    t->x[0] = a0, t->x[1] = a1, t->x[2] = a2, t->x[3] = a3;
    rt_svc(t, imm, PC);
    return t->x[0];
}
// A call through a PLT slot, as the lifter writes one.
static inline uint64_t gcall(Cpu *t, uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2) {
    t->x[0] = a0, t->x[1] = a1, t->x[2] = a2, t->x[3] = 0, t->x[30] = PC + 4, t->pc = fn;
    rt_dispatch(t);
    return t->x[0];
}
static inline uint64_t ld_acq(const uint64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void st_rel(uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static void guest_sleep(Cpu *t, uint64_t ns) { sv(t, 0x0B, ns, 0, 0, 0); }   // SleepThread: the core is given up

static uint32_t new_thread(Cpu *t, int fn, int priority, int core, int index) {
    t->x[1] = G(fn), t->x[2] = 0, t->x[3] = W(STACKS) + (uint64_t)STACK_BYTES * (uint64_t)(index + 1);
    t->x[4] = (uint64_t)priority, t->x[5] = (uint64_t)core;
    rt_svc(t, 0x08, PC);   // CreateThread
    CHECK(t->x[0] == 0);
    uint32_t h = (uint32_t)t->x[1];
    CHECK(sv(t, 0x09, h, 0, 0, 0) == 0);   // StartThread
    return h;
}

// One frame: acquire, then wait for the driver's go when there is a driver, submit nothing, present.
static uint64_t A_ACQ, A_GO, A_PRES;   // atomic: acquires returned, presents allowed, presents done
static int g_handshake;
static void frame(Cpu *t, uint64_t k) {
    CHECK(LOW32(gcall(t, f_acquire, g_win, 0, W(IDX))) == 0);
    st_rel(&A_ACQ, k + 1);
    if (g_handshake)
        while (ld_acq(&A_GO) < k + 1) sv(t, 0x1E, 0, 0, 0, 0);   // GetSystemTick: the exit is a safe point
    gcall(t, f_submit, g_queue, 0, 0);
    uint32_t idx = *(volatile uint32_t *)(uintptr_t)W(IDX);
    t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = idx, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
    rt_dispatch(t);
    st_rel(&A_PRES, k + 1);
}

// ---- frames: the worker burns CPU on request, and sleeps otherwise
#define FRAMES 10
static uint64_t W_BURN, W_BURNED, W_DONE;   // atomic: the frame number + 1 to burn in, the last one burned, the end
static int g_hold_core;                     // burn by holding the virtual core without running
static void g_worker(Cpu *t) {
    printf("worker handle %#x\n", rt_kernel_current());
    while (!ld_acq(&W_DONE)) {
        uint64_t want = ld_acq(&W_BURN);
        if (want > ld_acq(&W_BURNED)) {
            uint64_t c0 = plat_thread_cpu_ns(NULL);
            volatile uint64_t x = 1;
            if (g_hold_core) plat_sleep_ns(30 * MS);   // the core stays with this thread, which does not run
            else
                while (plat_thread_cpu_ns(NULL) - c0 < 30 * MS)
                    for (int i = 0; i < 10000; i++) x = x * 6364136223846793005ull + 1;
            st_rel(&W_BURNED, want);
        } else {
            guest_sleep(t, 5 * MS);   // longer than the kernel's spin threshold (1 ms on Windows): a timer sleep, no CPU
        }
    }
    sv(t, 0x0A, 0, 0, 0, 0);   // ExitThread
}

static void g_game_frames(Cpu *t) {
    printf("game handle %#x\n", rt_kernel_current());
    uint32_t w = new_thread(t, 1, 44, 1, 0);
    for (uint64_t k = 0; k < FRAMES; k++) frame(t, k);
    st_rel(&W_DONE, 1);
    uint64_t hs = W(HANDLES);
    *(volatile uint32_t *)(uintptr_t)hs = w;
    CHECK(sv(t, 0x18, 0, hs, 1, ~0ull) == 0);   // WaitSynchronization: the worker ended
    CHECK(sv(t, 0x16, w, 0, 0, 0) == 0);        // CloseHandle
    sv(t, 0x0A, 0, 0, 0, 0);
}

// ---- stop cases and length: the game presents on the host's clock, then stops the way the case names
static int g_stop;   // 70, 71, 72, 7 (ExitProcess), 73 (the census frame limit ends it), 0 (length: never stops itself)
static void g_game_stop(Cpu *t) {
    for (uint64_t k = 0;; k++) {
        if (g_stop && g_stop != 73 && k == 3) {
            if (g_stop == 70) rt_fatal("capture test: a fatal after %llu presents", (ull)k);
            if (g_stop == 71) rt_unimplemented(t, "test", "a named fatal after %llu presents", (ull)k);
            if (g_stop == 72) rt_guest_stop(t, "a guest stop after %llu presents", (ull)k);
            sv(t, 0x07, 0, 0, 0, 0);   // ExitProcess
        }
        if (k == 5 && !g_stop) {   // the length case's counters
            capture_count(CAPTURE_PIPELINE_MISSES, 1);
            capture_gpu_pass(5, 0, 1234);
        }
        frame(t, k);
        CHECK(k < 100000);
    }
}

// ---- off: three presents, then the thread times say whether a frame boundary was marked
static void g_game_off(Cpu *t) {
    for (uint64_t k = 0; k < 3; k++) frame(t, k);
    RtThreadTime tt[8];
    int n = rt_kernel_thread_times(tt, 8);
    CHECK(n >= 1);
    for (int i = 0; i < n && i < 8; i++) CHECK(tt[i].cpu_ns == 0 && tt[i].core_ns == 0);
    printf("off: %d thread(s), no frame time recorded\n", n);
    sv(t, 0x0A, 0, 0, 0, 0);
}

// ---- cost: 32 sleeping threads beside the game, the CPU time of each present call
#define COST_FRAMES 600
#define COST_THREADS 32
static uint64_t g_cost[COST_FRAMES];
static void g_sleeper(Cpu *t) {
    for (;;) guest_sleep(t, 100000 * MS);
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static void g_game_cost(Cpu *t) {
    for (int i = 0; i < COST_THREADS; i++) new_thread(t, 4, 50, 0, i + 1);
    for (uint64_t k = 0; k < COST_FRAMES; k++) {
        CHECK(LOW32(gcall(t, f_acquire, g_win, 0, W(IDX))) == 0);
        uint32_t idx = *(volatile uint32_t *)(uintptr_t)W(IDX);
        uint64_t c0 = plat_thread_cpu_ns(NULL);
        t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = idx, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
        rt_dispatch(t);
        g_cost[k] = plat_thread_cpu_ns(NULL) - c0;
        st_rel(&A_PRES, k + 1);
    }
    int on = capture_on();
    capture_end(0);
    uint64_t sum = 0;
    for (int k = 0; k < COST_FRAMES; k++) sum += g_cost[k];
    qsort(g_cost, COST_FRAMES, sizeof *g_cost, cmp_u64);
    printf("cost: capture %s, %d presents beside %d sleeping guest threads: the present call took %.1f us of CPU on "
           "average, p50 %.1f, p99 %.1f, max %.1f\n", on ? "on" : "off", COST_FRAMES, COST_THREADS,
           (double)sum / COST_FRAMES / 1e3, (double)g_cost[COST_FRAMES / 2] / 1e3,
           (double)g_cost[COST_FRAMES * 99 / 100] / 1e3, (double)g_cost[COST_FRAMES - 1] / 1e3);
    printf("ok cost\n");
    fflush(stdout);
    plat_exit(0);   // the sleepers never end
}

// ---- shown: twelve presents; the back end below knows each one's shown tick two presents later
static void g_game_shown(Cpu *t) {
    for (uint64_t k = 0; k < 12; k++) frame(t, k);
    sv(t, 0x0A, 0, 0, 0, 0);
}
static GfxBackend g_shown_backend;
static uint64_t shown_tick(uint64_t seq) { return seq + 2 <= nvn_present_count() ? seq * 10 : 0; }

static void guest_end(Cpu *t) { (void)t; }
static const GuestFn g_fns[] = {g_game_frames, g_worker,    g_game_stop, g_game_off,
                                g_sleeper,     g_game_cost, g_game_shown, guest_end};
#define NFN ((int)(sizeof g_fns / sizeof *g_fns))
static uint64_t g_addrs[NFN];

// ---------- set-up
static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP_BYTES);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}

static uint64_t vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}
static uint64_t proc(const char *name) {
    static uint64_t gpa;
    if (!gpa) {
        size_t len = strlen("nvnDeviceGetProcAddress") + 1;
        uint64_t s = galloc(len, 16);
        memcpy((void *)(uintptr_t)s, "nvnDeviceGetProcAddress", len);
        c.x[0] = s, c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
    }
    CHECK(gpa != 0);
    size_t len = strlen(name) + 1;
    uint64_t s = galloc(len, 16);
    memcpy((void *)(uintptr_t)s, name, len);
    c.x[0] = 0, c.x[1] = s;
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0);
    return c.x[0];
}
#define NVN(name, ...) vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / 8)

static uint64_t texture(uint64_t serial) {
    uint64_t t = galloc(0x100, 16);
    tex_t *x = (tex_t *)(uintptr_t)t;
    x->magic = NVN_TEX_MAGIC, x->serial = serial, x->d.width = 1280, x->d.height = 720, x->d.depth = 1;
    return t;
}

// Memory, kernel with two virtual cores, entry points, a device, a queue and a window of 3 textures at interval 1; the
// clock driven when asked; the hold off whatever the environment says. The capture starts last, as main.c starts it
// after the start-up.
static void objects(const GfxBackend *b, int driven) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512 * MiB, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256 * MiB, .system_resource_size = MiB};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x3, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1, .no_async_preempt = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++)
        k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    if (b) gfx_set_backend(b);
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    for (int i = 0; i < NFN; i++) g_addrs[i] = G(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, (uint64_t)NFN, TEST_END};
    rt_dispatch_init(mods, 1);
    c.x[0] = 0, c.x[1] = HEAP_BYTES;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    c.excl_addr = ~0ull;
    if (driven) drive_clock();
    nvn_present_hold(0);

    uint64_t db = galloc(0x40, 16);
    g_dev = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, db);
    CHECK((NVN(nvnDeviceInitialize, g_dev, db) & 0xff) == 1);
    uint64_t qb = galloc(0x40, 16);
    g_queue = galloc(0x2000, 16);
    NVN(nvnQueueBuilderSetDevice, qb, g_dev);
    NVN(nvnQueueBuilderSetDefaults, qb);
    NVN(nvnQueueBuilderSetFlags, qb, 0);
    NVN(nvnQueueBuilderSetCommandMemorySize, qb, 0x10000);
    NVN(nvnQueueBuilderSetComputeMemorySize, qb, 0x40000);
    NVN(nvnQueueBuilderSetControlMemorySize, qb, 0x4000);
    uint64_t size = NVN(nvnQueueBuilderGetQueueMemorySize, qb);
    NVN(nvnQueueBuilderSetQueueMemory, qb, galloc(size, 0x1000), size);
    CHECK((NVN(nvnQueueInitialize, g_queue, qb) & 0xff) == 1);
    uint64_t arr = galloc(8 * 4, 8), wb = galloc(0x40, 16);
    g_win = galloc(0x180, 16);
    for (int i = 0; i < 3; i++) *(uint64_t *)(uintptr_t)(arr + 8ull * (uint64_t)i) = texture((uint64_t)i + 1);
    NVN(nvnWindowBuilderSetDevice, wb, g_dev);
    NVN(nvnWindowBuilderSetDefaults, wb);
    NVN(nvnWindowBuilderSetNativeWindow, wb, 0xAB00);
    NVN(nvnWindowBuilderSetTextures, wb, 3, arr);
    CHECK((NVN(nvnWindowInitialize, g_win, wb) & 0xff) == 1);
    g_area = galloc(AREA, 0x10000);
    f_acquire = proc("nvnWindowAcquireTexture");
    f_present = proc("nvnQueuePresentTexture");
    f_submit = proc("nvnQueueSubmitCommands");
    capture_note("test_before", "noted before the start\twith a tab");
    capture_start(g_folder);
    capture_note("test_after", "noted after the start");
}

// ---------- the drivers, host threads
static void await(const uint64_t *p, uint64_t want, const char *what) {
    uint64_t end = plat_now_ns() + 10000000000ull;
    while (ld_acq(p) < want) {
        if (plat_now_ns() > end) {
            fprintf(stderr, "FAIL: %s\n", what);
            exit(1);
        }
        plat_sleep_ns(200000ull);
    }
}

// frames: acquire k, the worker's burn when k asks for one, the go, present k, then tick k + 1.
static void driver_frames(void *arg) {
    (void)arg;
    for (uint64_t k = 0; k < FRAMES; k++) {
        await(&A_ACQ, k + 1, "the acquire did not come back");
        if (k == 4 || k == 7) {
            st_rel(&W_BURN, k + 1);
            await(&W_BURNED, k + 1, "the worker did not burn");
        }
        st_rel(&A_GO, k + 1);
        await(&A_PRES, k + 1, "the present did not come back");
        set_now(ref_due_ns(k + 1));
    }
}

// shown and cost: each tick as soon as the frame before it was presented.
static uint64_t g_auto_frames;
static void driver_auto(void *arg) {
    (void)arg;
    for (uint64_t k = 0; k < g_auto_frames; k++) {
        await(&A_PRES, k + 1, "the present did not come back");
        set_now(ref_due_ns(k + 1));
    }
}

// ---------- cases
#define HANG_NS 90000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    rt_kernel_dump();
    _Exit(3);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 3) {
        fprintf(stderr, "usage: capture_test <case> <folder>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    g_folder = argv[2];
    if (!strcmp(cs, "frames") || !strcmp(cs, "gap")) {
        g_hold_core = !strcmp(cs, "gap");
        g_handshake = 1;
        objects(NULL, 1);
        CHECK(plat_thread_start(driver_frames, NULL, 1ull << 20));
        rt_kernel_run(G(0), 0);
        capture_end(0);
    } else if (!strcmp(cs, "stop70") || !strcmp(cs, "stop71") || !strcmp(cs, "stop72") ||
               !strcmp(cs, "exitprocess") || !strcmp(cs, "census73") || !strcmp(cs, "length")) {
        g_stop = !strcmp(cs, "exitprocess") ? 7 : !strcmp(cs, "length") ? 0 : atoi(cs + strlen(cs) - 2);
        objects(NULL, 0);
        rt_kernel_run(G(2), 0);
        fprintf(stderr, "FAIL: the run did not stop\n");
        return 1;
    } else if (!strcmp(cs, "off")) {
        objects(NULL, 0);
        CHECK(!capture_on());
        rt_kernel_run(G(3), 0);
        capture_end(0);
    } else if (!strcmp(cs, "shown")) {
        g_shown_backend = gfx_null_backend;
        g_shown_backend.name = "test-shown";
        g_shown_backend.present_shown = shown_tick;
        g_auto_frames = 12;
        objects(&g_shown_backend, 1);
        CHECK(plat_thread_start(driver_auto, NULL, 1ull << 20));
        rt_kernel_run(G(6), 0);
        capture_end(0);
    } else if (!strcmp(cs, "cost")) {
        g_auto_frames = COST_FRAMES;
        objects(NULL, 1);
        CHECK(plat_thread_start(driver_auto, NULL, 1ull << 20));
        rt_kernel_run(G(5), 0);
    } else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
