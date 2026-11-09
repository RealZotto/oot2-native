// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the NVN2 front end's queues, syncs, events, reports and windows (architecture section 5), driven by
// tests/test_runtime_nvn_queue.py: `nvn_queue_test <case>`. Each case runs in its own process. A passing case prints
// "ok <case>"; a failing case must end in one named fatal, which the driver checks.
//
// Entry points are called the way generated code calls them: their synthetic address, arguments in the AAPCS64
// registers, objects in guest memory. Recordings go to the null back end, whose completion a case may hold
// (gfx_null_hold) to see what is and is not written before a fence value is done. The syncwait case runs two guest
// threads, written by hand as host functions in the lifter's shape, on one virtual core. Textures are bytes written in
// the front end's layout; nothing is read from a dump.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "nvn_api.h"
#include "oot2/gfx.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/nvn_tex.h"
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
#define POOL_SIZE (4 * MiB)
#define RESULTS 0x300000ull            // pool offset of the words reports, events and counters use
#define CPU_NO_ACCESS 0x1u
#define GPU_NO_ACCESS 0x8u
#define CPU_UNCACHED 0x2u
#define GPU_CACHED 0x20u
#define TIMESTAMP 0u                   // NVNcounterType
#define SAMPLES_PASSED 1u
#define TIMESTAMP_TOP 0x10u
#define WRITE 0u                       // NVNeventSignalMode
#define ADD 1u
#define EQUAL 0u                       // NVNeventWaitMode, NVNconditionalRenderMode
#define GEQUAL_WRAP 1u
#define NOT_EQUAL 1u
#define ALREADY_SIGNALED 0u            // NVNsyncWaitResult
#define CONDITION_SATISFIED 1u
#define TIMEOUT_EXPIRED 2u
#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))     // the guest address of guest function i
#define PC (MB + 0x4000)               // the guest pc the guest threads' calls report

static Cpu c;                          // the host thread's Cpu: entry points called before or without guest threads
static uint64_t heap, heap_used, loader;
static uint64_t g_dev, g_pool, g_pool_mem, g_cmd_off;

// ---------- the guest side of the syncwait case: two threads on one virtual core
static uint64_t f_syncwait, g_sync, g_fence;   // set by the host before the run
static uint64_t g_area;                         // guest memory of the case's words and the second thread's stack
enum { ORDER_N = 0x0, RESULT = 0x8, ORDER = 0x40, HANDLES = 0x100, STACKS = 0x100000, AREA = 0x120000 };
#define W(off) (g_area + (off))
static inline uint64_t sv(Cpu *t, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    t->x[0] = a0, t->x[1] = a1, t->x[2] = a2, t->x[3] = a3;
    rt_svc(t, imm, PC);
    return t->x[0];
}
// A call through a PLT slot, as the lifter writes one: x30, then the dispatch of the bound address.
static inline uint64_t gcall(Cpu *t, uint64_t fn, uint64_t a0, uint64_t a1) {
    t->x[0] = a0, t->x[1] = a1, t->x[30] = PC + 4, t->pc = fn;
    rt_dispatch(t);
    return t->x[0];
}
static inline void record_step(uint32_t v) {
    uint64_t i = __atomic_fetch_add((uint64_t *)(uintptr_t)W(ORDER_N), 1, __ATOMIC_SEQ_CST);
    ST32(W(ORDER) + 4 * i, v);
}
enum { FN_MAIN_WAIT, FN_SIGNALLER, FN_END };
static void guest_end(Cpu *t);

// The waiter: starts a thread of worse priority on its own core, then waits on a sync whose submission is held. The
// other thread can run only once the waiter has given the core up.
static void g_main_wait(Cpu *t) {
    t->x[1] = G(FN_SIGNALLER), t->x[2] = 0, t->x[3] = W(STACKS) + 0x10000, t->x[4] = 50, t->x[5] = 0;
    rt_svc(t, 0x08, PC);   // CreateThread
    CHECK(t->x[0] == 0);
    uint32_t h = (uint32_t)t->x[1];
    CHECK(sv(t, 0x09, h, 0, 0, 0) == 0);   // StartThread
    record_step(1);
    uint64_t r = gcall(t, f_syncwait, g_sync, 5000000000ull) & 0xffffffffu;
    record_step(4);
    ST64(W(RESULT), r);
    ST32(W(HANDLES), h);
    CHECK(sv(t, 0x18, 0, W(HANDLES), 1, ~0ull) == 0);   // WaitSynchronization: the other thread ended
    CHECK(sv(t, 0x16, h, 0, 0, 0) == 0);                // CloseHandle
    sv(t, 0x0A, 0, 0, 0, 0);                            // ExitThread
}
// The signaller: the waiter's fence is still held while it runs; it completes it.
static void g_signaller(Cpu *t) {
    record_step(2);
    CHECK(gfx_backend()->fence_done() < g_fence);
    gfx_null_complete(0);
    record_step(3);
    sv(t, 0x0A, 0, 0, 0, 0);
}
static void guest_end(Cpu *t) { (void)t; }
static const GuestFn g_fns[FN_END + 1] = {g_main_wait, g_signaller, guest_end};
static uint64_t g_addrs[FN_END + 1];

// ---------- set-up
static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static uint64_t g_other;
static void other_fn(Cpu *t) { t->x[0] = 0; }

// Guest memory, the guest threads' kernel when kernel is set (one virtual core), back end b when not NULL, the entry
// points and the host-function registry sealed, then the heap.
static void setup(int kernel, const GfxBackend *b) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512 * MiB, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256 * MiB, .system_resource_size = MiB};
    rt_mem_init(&m);
    if (kernel) {
        RtKernelConfig k = {.core_mask = 0x1, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                            .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                            .program_id = 0x0100000000001000ull, .is_application = 1, .no_async_preempt = 1};
        for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++)
            k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
        rt_kernel_init(&k);
    }
    if (b) gfx_set_backend(b);
    g_other = rt_hostfn_register("census_test_other", other_fn);   // not an NVN2 entry point: census-nvn.tsv leaves it out
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    for (int i = 0; i <= FN_END; i++) g_addrs[i] = G(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, (uint64_t)FN_END + 1, TEST_END};
    rt_dispatch_init(mods, 1);
    c.x[0] = 0, c.x[1] = HEAP_BYTES;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    c.excl_addr = ~0ull;
}

// Guest memory, zeroed, as the game's allocator would give it.
static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP_BYTES);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}

static uint64_t rd64(uint64_t a) { uint64_t v; memcpy(&v, (const void *)(uintptr_t)a, 8); return v; }
static void wr64(uint64_t a, uint64_t v) { memcpy((void *)(uintptr_t)a, &v, 8); }

// An entry point called from guest code at return address `site` (x30), as the game's call sites call them.
static uint64_t vcall_at(uint64_t site, uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    if (site) {
        c.x[30] = site, c.pc = fn;
        rt_dispatch(&c);
    } else {
        rt_call(&c, fn);
    }
    return c.x[0];
}

static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[96];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) {
        size_t len = strlen("nvnDeviceGetProcAddress") + 1;
        uint64_t s = galloc(len, 16);
        memcpy((void *)(uintptr_t)s, "nvnDeviceGetProcAddress", len);
        c.x[0] = s, c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
    }
    CHECK(gpa != 0 && n < 96);
    size_t len = strlen(name) + 1;
    uint64_t s = galloc(len, 16);
    memcpy((void *)(uintptr_t)s, name, len);
    c.x[0] = 0, c.x[1] = s;
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0);
    cache[n].name = name, cache[n].addr = c.x[0];
    return cache[n++].addr;
}
#define ARGS(...) (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t)
#define NVN(name, ...) vcall_at(0, proc(#name), ARGS(__VA_ARGS__))
#define NVN_AT(site, name, ...) vcall_at(site, proc(#name), ARGS(__VA_ARGS__))
#define LOW32(v) ((v) & 0xffffffffu)

// ---------- objects, as the game sets them up
static uint64_t device(void) {
    uint64_t b = galloc(0x40, 16), d = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, b);
    CHECK((NVN(nvnDeviceInitialize, d, b) & 0xff) == 1);
    return d;
}

static uint64_t memory_pool(uint64_t size, uint32_t flags, uint64_t *storage) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16), mem = galloc(size, 0x10000);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetDevice, b, g_dev);
    NVN(nvnMemoryPoolBuilderSetStorage, b, mem, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, flags);
    CHECK((NVN(nvnMemoryPoolInitialize, o, b) & 0xff) == 1);
    if (storage) *storage = mem;
    return o;
}

// A device, and a CPU-visible pool for command memory and results.
static void objects(void) {
    g_dev = device();
    g_pool = memory_pool(POOL_SIZE, CPU_UNCACHED | GPU_CACHED, &g_pool_mem);
}

static uint64_t queue(void) {
    uint64_t b = galloc(0x40, 16), q = galloc(0x2000, 16);
    NVN(nvnQueueBuilderSetDevice, b, g_dev);
    NVN(nvnQueueBuilderSetDefaults, b);
    NVN(nvnQueueBuilderSetFlags, b, 0);
    NVN(nvnQueueBuilderSetCommandMemorySize, b, 0x10000);
    NVN(nvnQueueBuilderSetComputeMemorySize, b, 0x40000);
    NVN(nvnQueueBuilderSetControlMemorySize, b, 0x4000);
    uint64_t size = NVN(nvnQueueBuilderGetQueueMemorySize, b);
    CHECK(size == 0x54000);
    NVN(nvnQueueBuilderSetQueueMemory, b, galloc(size, 0x1000), size);
    CHECK((NVN(nvnQueueInitialize, q, b) & 0xff) == 1);
    return q;
}

static uint64_t sync_object(void) {
    uint64_t s = galloc(0x40, 16);
    CHECK((NVN(nvnSyncInitialize, s, g_dev) & 0xff) == 1);
    return s;
}
static uint64_t sync_fence(uint64_t s) { return ((const sync_t *)(uintptr_t)s)->fence; }

static uint64_t event_object(uint64_t pool, uint64_t offset) {
    uint64_t b = galloc(0x20, 16), e = galloc(0x40, 16);
    NVN(nvnEventBuilderSetDefaults, b);
    NVN(nvnEventBuilderSetStorage, b, pool, offset);
    CHECK((NVN(nvnEventInitialize, e, b) & 0xff) == 1);
    return e;
}

static uint64_t command_buffer(void) {
    uint64_t o = galloc(0xC0, 16);
    CHECK((NVN(nvnCommandBufferInitialize, o, g_dev) & 0xff) == 1);
    NVN(nvnCommandBufferAddCommandMemory, o, g_pool, g_cmd_off, 0x10000);
    g_cmd_off += 0x10000;
    NVN(nvnCommandBufferAddControlMemory, o, galloc(0x10000, 16), 0x10000);
    return o;
}

static uint64_t submit(uint64_t q, uint64_t handle) {
    uint64_t a = galloc(8, 8);
    wr64(a, handle);
    NVN(nvnQueueSubmitCommands, q, 1, a);
    return ((const queue_t *)(uintptr_t)q)->last;
}

static uint32_t sync_wait(uint64_t s, uint64_t timeout) { return (uint32_t)LOW32(NVN(nvnSyncWait, s, timeout)); }

// ---------- case fences: completion in submission order, across two queues
static void case_fences(void) {
    setup(0, NULL);
    objects();
    uint64_t q1 = queue(), q2 = queue(), cb = command_buffer();
    uint64_t ev = event_object(g_pool, RESULTS), rep = g_pool_mem + RESULTS + 0x100;
    uint64_t s[3] = {sync_object(), sync_object(), sync_object()};
    gfx_null_hold(1);
    // Submission 1 on queue 1 writes 1, submission 2 on queue 2 writes 2, submission 3 on queue 1 adds 10: only
    // submission order gives 12.
    const struct { uint64_t q; uint32_t mode, value; } sub[3] = {{q1, WRITE, 1}, {q2, WRITE, 2}, {q1, ADD, 10}};
    for (int k = 0; k < 3; k++) {
        NVN(nvnCommandBufferBeginRecording, cb);
        NVN(nvnCommandBufferSignalEvent, cb, ev, sub[k].mode, 2, 0, sub[k].value);
        NVN(nvnCommandBufferReportCounter, cb, TIMESTAMP, rep + 16ull * (uint64_t)k);
        CHECK(submit(sub[k].q, NVN(nvnCommandBufferEndRecording, cb)) == (uint64_t)k + 1);
        NVN(nvnQueueFenceSync, sub[k].q, s[k], 0, 0);
        CHECK(sync_fence(s[k]) == (uint64_t)k + 1);
    }
    // Nothing is written before its submission completes.
    CHECK(LOW32(NVN(nvnEventGetValue, ev)) == 0);
    for (int k = 0; k < 3; k++) CHECK(rd64(rep + 16ull * (uint64_t)k + 8) == 0 && sync_wait(s[k], 0) == TIMEOUT_EXPIRED);
    // A wait on the GPU for a fence another queue gave holds nothing back on the host.
    CHECK((NVN(nvnQueueWaitSync, q2, s[0]) & 0xff) == 1);
    const uint32_t want[3] = {1, 2, 12};
    for (int k = 0; k < 3; k++) {
        CHECK(gfx_null_complete((uint64_t)k + 1) == (uint64_t)k + 1);
        for (int j = 0; j < 3; j++) CHECK(sync_wait(s[j], 0) == (j <= k ? ALREADY_SIGNALED : TIMEOUT_EXPIRED));
        uint32_t v = (uint32_t)LOW32(NVN(nvnEventGetValue, ev));
        CHECK(v == want[k]);
        for (int j = 0; j < 3; j++) CHECK((rd64(rep + 16ull * (uint64_t)j + 8) != 0) == (j <= k));
        printf("complete %d event %u\n", k + 1, v);
    }
    // Three more, alternating queues, completed at once: the last in submission order is the value left.
    for (int k = 0; k < 3; k++) {
        NVN(nvnCommandBufferBeginRecording, cb);
        NVN(nvnCommandBufferSignalEvent, cb, ev, WRITE, 2, 0, 5 + k);
        submit(k % 2 ? q2 : q1, NVN(nvnCommandBufferEndRecording, cb));
        NVN(nvnQueueFenceSync, k % 2 ? q2 : q1, s[k], 0, 0);
    }
    CHECK(sync_wait(s[2], 0) == TIMEOUT_EXPIRED);
    CHECK(gfx_null_complete(0) == 6);
    printf("all at once event %u\n", (uint32_t)LOW32(NVN(nvnEventGetValue, ev)));
    CHECK(sync_wait(s[0], 0) == ALREADY_SIGNALED && sync_wait(s[2], 0) == ALREADY_SIGNALED);
    // QueueFinish and the CPU's own signal.
    gfx_null_hold(0);
    NVN(nvnQueueFinish, q1);
    NVN(nvnEventSignal, ev, ADD, 100);
    CHECK(LOW32(NVN(nvnEventGetValue, ev)) == 107);
    NVN(nvnQueueFlush, q2);
    CHECK(LOW32(NVN(nvnQueueGetError, q1, 0)) == 0);
    printf("ok fences\n");
}

// ---------- case timestamps: reports in 31.25 MHz ticks of the host clock, read when they complete
static uint64_t report_once(uint64_t q, uint64_t cb, uint32_t type, uint64_t addr) {
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferReportCounter, cb, type, addr);
    return submit(q, NVN(nvnCommandBufferEndRecording, cb));
}
static void case_timestamps(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer(), a = g_pool_mem + RESULTS, b = a + 16;
    uint64_t t0 = rt_ticks(), n0 = plat_now_ns();
    report_once(q, cb, TIMESTAMP, a);
    uint64_t n1 = plat_now_ns(), t1 = rt_ticks();
    plat_sleep_ns(20000000);
    uint64_t n2 = plat_now_ns();
    report_once(q, cb, TIMESTAMP, b);
    uint64_t n3 = plat_now_ns();
    uint64_t ta = rd64(a + 8), tb = rd64(b + 8);
    CHECK(rd64(a) == 0 && rd64(b) == 0);   // the counter word: nothing measured
    CHECK(t0 <= ta && ta <= t1);           // the guest's tick when the report completed
    CHECK(ta < tb);
    uint64_t span = (tb - ta) * 32;        // 32 ns a tick
    CHECK(span + 64 >= n2 - n1 && span <= n3 - n0 + 64);
    printf("two reports %llu ticks apart, %llu ns, the host between %llu and %llu ns\n", (ull)(tb - ta), (ull)span,
           (ull)(n2 - n1), (ull)(n3 - n0));
    CHECK(NVN(nvnDeviceGetTimestampInNanoseconds, g_dev, a) == ta * 32);
    uint64_t now = NVN(nvnDeviceGetCurrentTimestampInNanoseconds, g_dev);
    CHECK(now >= tb * 32 && now <= rt_ticks() * 32);
    // Sixty-four reports in one recording, of three types: every one timestamped, never going back.
    uint64_t many = a + 0x100;
    NVN(nvnCommandBufferBeginRecording, cb);
    for (int i = 0; i < 64; i++) {
        uint32_t type = i % 3 == 0 ? TIMESTAMP : i % 3 == 1 ? TIMESTAMP_TOP : SAMPLES_PASSED;
        NVN(nvnCommandBufferReportCounter, cb, type, many + 16ull * (uint64_t)i);
    }
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    for (int i = 0; i < 64; i++) {
        uint64_t at = many + 16ull * (uint64_t)i;
        CHECK(rd64(at) == 0 && rd64(at + 8) >= tb && (i == 0 || rd64(at + 8) >= rd64(at - 8)));
    }
    // Held: the timestamp is the time of completion, not of the submit.
    uint64_t h = a + 0x800;
    gfx_null_hold(1);
    report_once(q, cb, TIMESTAMP, h);
    uint64_t submitted = rt_ticks();
    plat_sleep_ns(10000000);
    uint64_t c0 = rt_ticks();
    CHECK(rd64(h + 8) == 0);
    gfx_null_complete(0);
    uint64_t c1 = rt_ticks();
    gfx_null_hold(0);
    CHECK(rd64(h + 8) > submitted && c0 <= rd64(h + 8) && rd64(h + 8) <= c1);
    printf("ok timestamps\n");
}

// ---------- case render_cond: conditional rendering decided at replay, on complete and incomplete reports
static void draw_log(uint64_t fence, int index, const op_hdr *r, void *ctx) {
    (void)fence, (void)index, (void)ctx;
    if (r->op == OP_DRAW)
        printf("draw %d %s\n", ((const p_draw *)(const void *)(r + 1))->first,
               gfx_null_render_enabled() ? "renders" : "skipped");
}
static void case_render_cond(void) {
    setup(0, NULL);
    objects();
    gfx_null_log_ops(draw_log, NULL);
    uint64_t q = queue(), cb = command_buffer(), cnt = g_pool_mem + RESULTS;   // two NVNcounterData: cnt, cnt + 16
    // Complete reports: the words in memory decide.
    wr64(cnt, 5), wr64(cnt + 16, 5);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferSetRenderEnableConditional, cb, EQUAL, cnt);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 1, 3);
    NVN(nvnCommandBufferSetRenderEnableConditional, cb, NOT_EQUAL, cnt);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 2, 3);
    NVN(nvnCommandBufferSetRenderEnable, cb, 1);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 3, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    // A timestamp report not complete counts as 0: memory says 7 and 0, the condition sees 0 and 0.
    wr64(cnt, 7), wr64(cnt + 16, 0);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferReportCounter, cb, TIMESTAMP, cnt);
    NVN(nvnCommandBufferSetRenderEnableConditional, cb, EQUAL, cnt);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 4, 3);
    NVN(nvnCommandBufferSetRenderEnable, cb, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    // Once complete, the report's counter word (0) is read from memory.
    CHECK(rd64(cnt) == 0 && rd64(cnt + 8) != 0);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferSetRenderEnableConditional, cb, NOT_EQUAL, cnt);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 5, 3);
    NVN(nvnCommandBufferSetRenderEnable, cb, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    // A samples-passed report not complete reads as visible: memory and 0 would both say equal.
    wr64(cnt, 0), wr64(cnt + 16, 0);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferReportCounter, cb, SAMPLES_PASSED, cnt + 16);
    NVN(nvnCommandBufferSetRenderEnableConditional, cb, NOT_EQUAL, cnt);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 6, 3);
    NVN(nvnCommandBufferSetRenderEnable, cb, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    // A timestamp report in an earlier submission that is held counts as 0 too: memory says 9 and 9.
    wr64(cnt, 9), wr64(cnt + 16, 9);
    gfx_null_hold(1);
    report_once(q, cb, TIMESTAMP, cnt);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferSetRenderEnableConditional, cb, NOT_EQUAL, cnt);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 7, 3);
    NVN(nvnCommandBufferSetRenderEnable, cb, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_hold(0);
    printf("ok render_cond\n");
}

// ---------- case window: three textures acquired and presented in order
static const tex_t *g_shown[64];
static int g_crops[64][4], g_nshown;
static void shown(const tex_t *t, const int crop[4]) {
    CHECK(g_nshown < 64);
    g_shown[g_nshown] = t;
    memcpy(g_crops[g_nshown++], crop, sizeof g_crops[0]);
    gfx_null_backend.present(t, crop);
}
static GfxBackend g_presenting;

static uint64_t texture(uint32_t width, uint32_t height, uint64_t serial) {
    uint64_t t = galloc(0x100, 16);
    tex_t *x = (tex_t *)(uintptr_t)t;
    x->magic = NVN_TEX_MAGIC, x->serial = serial, x->d.width = width, x->d.height = height, x->d.depth = 1;
    return t;
}

static uint64_t window(int n, const uint64_t *tex) {
    uint64_t arr = galloc(8 * 4, 8), b = galloc(0x40, 16), w = galloc(0x180, 16);
    for (int i = 0; i < n; i++) wr64(arr + 8ull * (uint64_t)i, tex[i]);
    NVN(nvnWindowBuilderSetDevice, b, g_dev);
    NVN(nvnWindowBuilderSetDefaults, b);
    NVN(nvnWindowBuilderSetNativeWindow, b, 0xAB00);
    NVN(nvnWindowBuilderSetTextures, b, (uint64_t)n, arr);
    CHECK((NVN(nvnWindowInitialize, w, b) & 0xff) == 1);
    return w;
}

static int acquire(uint64_t w, uint64_t s) {
    uint64_t idx = galloc(4, 4);
    ST32(idx, 0xFFFF);
    CHECK(LOW32(NVN(nvnWindowAcquireTexture, w, s, idx)) == 0);
    return (int)LD32(idx);
}

static void case_window(void) {
    g_presenting = gfx_null_backend;
    g_presenting.name = "presenting", g_presenting.present = shown;
    setup(0, &g_presenting);
    objects();
    uint64_t q = queue(), cb = command_buffer(), s = sync_object(), rect = galloc(16, 8);
    const uint64_t tex[3] = {texture(1920, 1080, 1), texture(1920, 1080, 2), texture(1920, 1080, 3)};
    uint64_t w = window(3, tex);
    CHECK(LOW32(NVN(nvnWindowGetPresentInterval, w)) == 1);   // the builder's default
    // This case is about the fence values of the acquire syncs: interval 0 keeps the clock's tick out of them (the pacing
    // of the acquire sync is tests/runtime/nvn_sync_test.c's).
    NVN(nvnWindowSetPresentInterval, w, 0);
    NVN(nvnWindowGetCrop, w, rect);                           // before SetCrop: the whole first texture
    CHECK(LD32(rect) == 0 && LD32(rect + 4) == 0 && LD32(rect + 8) == 1920 && LD32(rect + 12) == 1080);
    NVN(nvnWindowSetCrop, w, 0, 0, 1280, 720);
    NVN(nvnWindowGetCrop, w, rect);
    CHECK(LD32(rect + 8) == 1280 && LD32(rect + 12) == 720);
    for (int f = 0; f < 9; f++) {
        int i = acquire(w, s);
        CHECK(i == f % 3);
        CHECK(sync_wait(s, 0) == ALREADY_SIGNALED);   // the work before that texture's last present is done
        NVN(nvnCommandBufferBeginRecording, cb);
        NVN(nvnCommandBufferDrawArrays, cb, 4, f, 3);
        submit(q, NVN(nvnCommandBufferEndRecording, cb));
        NVN(nvnQueuePresentTexture, q, w, (uint64_t)i);
        printf("frame %d texture %d shown %llu\n", f, i, (ull)g_shown[g_nshown - 1]->serial);
    }
    CHECK(g_nshown == 9 && gfx_null_presents() == 9);
    for (int k = 0; k < 9; k++) {
        CHECK(g_shown[k] == (const tex_t *)(uintptr_t)tex[k % 3]);
        CHECK(g_crops[k][0] == 0 && g_crops[k][1] == 0 && g_crops[k][2] == 1280 && g_crops[k][3] == 720);
    }
    // The acquire sync waits for the work submitted before that texture's last present.
    gfx_null_hold(1);
    for (int k = 0; k < 3; k++) {
        int i = acquire(w, s);
        CHECK(i == k);
        NVN(nvnCommandBufferBeginRecording, cb);
        NVN(nvnCommandBufferDrawArrays, cb, 4, 9 + k, 3);
        submit(q, NVN(nvnCommandBufferEndRecording, cb));
        NVN(nvnQueuePresentTexture, q, w, (uint64_t)i);
    }
    CHECK(acquire(w, s) == 0 && sync_wait(s, 0) == TIMEOUT_EXPIRED);
    gfx_null_complete(0);
    CHECK(sync_wait(s, 0) == ALREADY_SIGNALED);
    gfx_null_hold(0);
    // The interval as written, with no clamping; two active textures from the next acquire.
    NVN(nvnWindowSetPresentInterval, w, 1);
    NVN(nvnWindowSetPresentInterval, w, 2);
    CHECK(LOW32(NVN(nvnWindowGetPresentInterval, w)) == 2);
    NVN(nvnWindowSetPresentInterval, w, 0);
    CHECK(LOW32(NVN(nvnWindowGetPresentInterval, w)) == 0);
    NVN(nvnWindowSetNumActiveTextures, w, 2);
    int seq[4];
    for (int k = 0; k < 4; k++) seq[k] = acquire(w, 0);
    CHECK(seq[0] == 1 && seq[1] == 0 && seq[2] == 1 && seq[3] == 0);
    NVN(nvnWindowFinalize, w);
    printf("ok window\n");
}

// ---------- case syncwait: a SyncWait gives its virtual core up while it blocks
#define HANG_NS 20000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    rt_kernel_dump();
    _Exit(3);
}
static void case_syncwait(void) {
    setup(1, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer();
    g_sync = sync_object();
    g_area = galloc(AREA, 0x10000);
    f_syncwait = proc("nvnSyncWait");
    gfx_null_hold(1);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    NVN(nvnQueueFenceSync, q, g_sync, 0, 0);
    g_fence = sync_fence(g_sync);
    CHECK(g_fence == 1 && sync_wait(g_sync, 0) == TIMEOUT_EXPIRED);
    rt_kernel_run(G(FN_MAIN_WAIT), 0);
    uint32_t order[4];
    CHECK(LD64(W(ORDER_N)) == 4);
    for (int i = 0; i < 4; i++) order[i] = LD32(W(ORDER) + 4ull * (uint64_t)i);
    printf("order %u %u %u %u result %llu\n", order[0], order[1], order[2], order[3], (ull)LD64(W(RESULT)));
    CHECK(order[0] == 1 && order[1] == 2 && order[2] == 3 && order[3] == 4);
    CHECK(LD64(W(RESULT)) == CONDITION_SATISFIED);
    gfx_null_hold(0);
    printf("ok syncwait\n");
}

// The front end's own counters as census rows (nvn_census.c), printed for the driver: "stat| key count why".
static void print_stat(void *ctx, const char *key, uint64_t count, const char *why) {
    (void)ctx;
    printf("stat| %s %llu %s\n", key, (ull)count, why);
}

static void census_stop(int code, const char *line) {
    printf("stop| %d %s\n", code, line);
    nvn_census_stats(print_stat, NULL);
    printf("stop| census complete\n");
}

static void census_at_draw(uint64_t fence, int index, const op_hdr *r, void *ctx) {
    (void)fence, (void)index;
    if (r->op != OP_DRAW) return;
    if (ctx) rt_fatal("queue test: fatal while replaying a draw");
    nvn_census_stats(print_stat, NULL);
    printf("draw| census complete\n");
}

static struct {
    PlatThread *thread;
    int started, finished;
    uint64_t waits, unsatisfied;
} g_census_thread;

static void census_on_other_thread(void *arg) {
    (void)arg;
    __atomic_store_n(&g_census_thread.started, 1, __ATOMIC_RELEASE);
    g_census_thread.waits = gfx_null_wait_events(&g_census_thread.unsatisfied);
    __atomic_store_n(&g_census_thread.finished, 1, __ATOMIC_RELEASE);
}

static void census_other_at_draw(uint64_t fence, int index, const op_hdr *r, void *ctx) {
    (void)fence, (void)index, (void)ctx;
    if (r->op != OP_DRAW) return;
    g_census_thread.thread = plat_thread_start(census_on_other_thread, NULL, 1ull << 20);
    CHECK(g_census_thread.thread);
    for (int i = 0; i < 200 && !__atomic_load_n(&g_census_thread.started, __ATOMIC_ACQUIRE); i++)
        plat_sleep_ns(10000000);
    CHECK(__atomic_load_n(&g_census_thread.started, __ATOMIC_ACQUIRE));
    plat_sleep_ns(50000000);
    CHECK(!__atomic_load_n(&g_census_thread.finished, __ATOMIC_ACQUIRE));
}

static void census_locked_case(int mode) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer(), ev = event_object(g_pool, RESULTS);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN_AT(MB + 0x6000, nvnCommandBufferWaitEvent, cb, ev, EQUAL, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));

    rt_set_stop_hook(census_stop);
    if (mode <= 1 || mode == 5) {
        gfx_null_log_ops(mode == 5 ? census_other_at_draw : census_at_draw, mode == 1 ? &mode : NULL);
        NVN(nvnCommandBufferBeginRecording, cb);
        NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
        submit(q, NVN(nvnCommandBufferEndRecording, cb));
        gfx_null_log_ops(NULL, NULL);
        if (mode == 5) {
            for (int i = 0; i < 200 && !__atomic_load_n(&g_census_thread.finished, __ATOMIC_ACQUIRE); i++)
                plat_sleep_ns(10000000);
            CHECK(__atomic_load_n(&g_census_thread.finished, __ATOMIC_ACQUIRE));
            CHECK(g_census_thread.waits == 1 && g_census_thread.unsatisfied == 1);
            plat_thread_free(g_census_thread.thread);
            printf("ok census_other_thread\n");
        } else printf("ok census_at_draw\n");
    } else {
        if (mode >= 3) gfx_null_hold(1);
        report_once(q, cb, TIMESTAMP, 0x1000);
        if (mode == 3) gfx_null_complete(0);
        if (mode == 4) gfx_null_hold(0);
        printf("fatal completion returned\n");
    }
}

static void case_census_at_draw(void) { census_locked_case(0); }
static void case_census_fatal_replay(void) { census_locked_case(1); }
static void case_census_fatal_complete(void) { census_locked_case(2); }
static void case_census_fatal_held_complete(void) { census_locked_case(3); }
static void case_census_fatal_held_release(void) { census_locked_case(4); }
static void case_census_other_thread(void) { census_locked_case(5); }

// ---------- case wait_events: each WaitEvent checked at completion, logged once per call site
static void case_wait_events(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer(), ev = event_object(g_pool, RESULTS);
    uint64_t evb = event_object(g_pool, RESULTS + 0x40), evc = event_object(g_pool, RESULTS + 0x80);
    NVN(nvnEventSignal, evb, WRITE, 5);   // the CPU signals before the submission: a wait on it is satisfied by the CPU
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferSignalEvent, cb, ev, WRITE, 2, 0, 3);
    NVN_AT(MB + 0x1000, nvnCommandBufferWaitEvent, cb, ev, EQUAL, 3);         // satisfied, by the GPU in order
    NVN_AT(MB + 0x1000, nvnCommandBufferWaitEvent, cb, ev, GEQUAL_WRAP, 2);   // satisfied, the same site
    NVN_AT(MB + 0x1000, nvnCommandBufferWaitEvent, cb, evb, EQUAL, 5);        // satisfied by the CPU's earlier signal
    NVN_AT(MB + 0x1000, nvnCommandBufferWaitEvent, cb, evc, GEQUAL_WRAP, 0);  // satisfied by the word's own value
    NVN_AT(MB + 0x2000, nvnCommandBufferWaitEvent, cb, ev, EQUAL, 9);         // not satisfied
    NVN_AT(MB + 0x2000, nvnCommandBufferWaitEvent, cb, ev, GEQUAL_WRAP, 4);   // not satisfied, the same site
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    uint64_t unsatisfied = 0;
    CHECK(gfx_null_wait_events(&unsatisfied) == 6 && unsatisfied == 2);
    GfxNullWaitClasses k;
    gfx_null_wait_classes(&k);
    CHECK(k.by_gpu == 2 && k.by_cpu == 1 && k.initial == 1 && k.unsatisfied == 2 && k.later_gpu == 0 && k.later_cpu == 0);
    // The CPU signals the word after the submission and a later completion finds both waits satisfied: by the CPU.
    NVN(nvnEventSignal, ev, WRITE, 9);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_wait_classes(&k);
    CHECK(k.later_cpu == 2 && k.later_gpu == 0);
    // A wait that a GPU signal of a later submission satisfies: unsatisfied now, satisfied later by the GPU.
    uint64_t evd = event_object(g_pool, RESULTS + 0xC0);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN_AT(MB + 0x3000, nvnCommandBufferWaitEvent, cb, evd, EQUAL, 7);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_wait_classes(&k);
    CHECK(k.unsatisfied == 3 && k.later_gpu == 0);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferSignalEvent, cb, evd, WRITE, 2, 0, 7);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_wait_classes(&k);
    CHECK(k.later_gpu == 1 && k.later_cpu == 2);
    CHECK(gfx_null_wait_events(&unsatisfied) == 7 && unsatisfied == 3);
    printf("waits 7 unsatisfied %llu%s", (ull)unsatisfied, "\n");
    printf("classes gpu %llu cpu %llu initial %llu unsatisfied %llu later_gpu %llu later_cpu %llu\n", (ull)k.by_gpu,
           (ull)k.by_cpu, (ull)k.initial, (ull)k.unsatisfied, (ull)k.later_gpu, (ull)k.later_cpu);
    nvn_census_stats(print_stat, NULL);
    // The list of unsatisfied waits is bounded (64): of 70 waits on a word nobody has signalled, a CPU signal and one
    // more completion find 64 of them satisfied later by the CPU.
    uint64_t eve = event_object(g_pool, RESULTS + 0x100);
    NVN(nvnCommandBufferBeginRecording, cb);
    for (int i = 0; i < 70; i++) NVN_AT(MB + 0x4000, nvnCommandBufferWaitEvent, cb, eve, EQUAL, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    NVN(nvnEventSignal, eve, WRITE, 1);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_wait_classes(&k);
    printf("bounded unsatisfied %llu later_cpu %llu dropped %llu pending %llu\n", (ull)k.unsatisfied, (ull)k.later_cpu,
           (ull)k.dropped, (ull)k.pending);
    // A wait on a word nobody signals stays unresolved: counted as pending, and the row for it says so.
    uint64_t evf = event_object(g_pool, RESULTS + 0x140);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN_AT(MB + 0x5000, nvnCommandBufferWaitEvent, cb, evf, EQUAL, 1);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_wait_classes(&k);
    CHECK(k.unsatisfied == k.later_gpu + k.later_cpu + k.pending + k.dropped);
    printf("unresolved unsatisfied %llu pending %llu dropped %llu\n", (ull)k.unsatisfied, (ull)k.pending, (ull)k.dropped);
    // a second pool, for a window range and no GPU access, made and ended: the pool rows tell made from live
    NVN(nvnMemoryPoolFinalize, memory_pool(0x10000, CPU_NO_ACCESS | GPU_NO_ACCESS, NULL));
    nvn_census_stats(print_stat, NULL);
    printf("ok wait_events\n");
}

// ---------- case wait_sites: WaitEvent counted per call site, by what satisfied it. A wait inside a recording that
// is called twice counts at its own site twice (CallCommands copies the record, caller included); 70 more sites than the
// table holds go to one bucket; the sites and the bucket sum to the totals and to the classes.
static const char *site_name(uint64_t caller, char *buf, int size) { return rt_guest_name(caller, buf, size); }
static void case_wait_sites(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer(), cb2 = command_buffer(), ev = event_object(g_pool, RESULTS);
    NVN(nvnCommandBufferBeginRecording, cb2);
    NVN_AT(MB + 0x6000, nvnCommandBufferWaitEvent, cb2, ev, GEQUAL_WRAP, 0);   // satisfied by the word's own value, twice
    NVN_AT(MB + 0x6000, nvnCommandBufferWaitEvent, cb2, ev, GEQUAL_WRAP, 0);   // per call of the recording
    uint64_t called = NVN(nvnCommandBufferEndRecording, cb2), arr = galloc(8, 8);
    wr64(arr, called);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferCallCommands, cb, 1, arr);
    NVN(nvnCommandBufferCallCommands, cb, 1, arr);
    NVN_AT(MB + 0x7000, nvnCommandBufferWaitEvent, cb, ev, EQUAL, 9);          // not satisfied, until the CPU signals
    for (int i = 0; i < 70; i++) NVN_AT(MB + 0x100 + 0x10 * (uint64_t)i, nvnCommandBufferWaitEvent, cb, ev, GEQUAL_WRAP, 0);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    NVN(nvnEventSignal, ev, WRITE, 9);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    uint64_t unsatisfied = 0, waits = gfx_null_wait_events(&unsatisfied);
    GfxNullWaitClasses k;
    gfx_null_wait_classes(&k);
    NvnWaitSite sites[64], other;
    int n = gfx_null_wait_sites(sites, 64, &other);
    GfxNullWaitClasses sum = other.c;
    uint64_t sum_waits = 0;
    char name[96];
    for (int i = 0; i < n; i++) {
        const GfxNullWaitClasses *s = &sites[i].c;
        sum.by_gpu += s->by_gpu, sum.by_cpu += s->by_cpu, sum.initial += s->initial, sum.unsatisfied += s->unsatisfied;
        sum.later_gpu += s->later_gpu, sum.later_cpu += s->later_cpu, sum.dropped += s->dropped, sum.pending += s->pending;
    }
    sum_waits = sum.by_gpu + sum.by_cpu + sum.initial + sum.unsatisfied;
    CHECK(n == 64 && waits == 4 + 1 + 70 && sum_waits == waits);
    CHECK(sum.by_gpu == k.by_gpu && sum.by_cpu == k.by_cpu && sum.initial == k.initial && sum.unsatisfied == k.unsatisfied);
    CHECK(sum.later_gpu == k.later_gpu && sum.later_cpu == k.later_cpu && sum.dropped == k.dropped && sum.pending == k.pending);
    for (int i = 0; i < 2; i++) {
        const GfxNullWaitClasses *s = &sites[i].c;
        printf("site %s gpu %llu cpu %llu initial %llu unsatisfied %llu later_gpu %llu later_cpu %llu\n",
               site_name(sites[i].caller, name, sizeof name), (ull)s->by_gpu, (ull)s->by_cpu, (ull)s->initial,
               (ull)s->unsatisfied, (ull)s->later_gpu, (ull)s->later_cpu);
    }
    printf("other caller %llu initial %llu unsatisfied %llu\n", (ull)other.caller, (ull)other.c.initial,
           (ull)other.c.unsatisfied);
    printf("totals waits %llu unsatisfied %llu initial %llu later_cpu %llu\n", (ull)waits, (ull)unsatisfied,
           (ull)k.initial, (ull)k.later_cpu);
    nvn_census_stats(print_stat, NULL);
    printf("ok wait_sites\n");
}

// Two pending waits at different sites, on words with different writers, satisfied in an order other than the one they were
// submitted in: the first is satisfied by a GPU signal, which takes it off the pending list and moves the second into its
// place; the second is then satisfied by a CPU signal. Each site must keep its own answer.
static void case_wait_sites_order(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer();
    uint64_t evg = event_object(g_pool, RESULTS + 0x200), evc = event_object(g_pool, RESULTS + 0x240);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN_AT(MB + 0x8000, nvnCommandBufferWaitEvent, cb, evg, EQUAL, 3);
    NVN_AT(MB + 0x9000, nvnCommandBufferWaitEvent, cb, evc, EQUAL, 4);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferSignalEvent, cb, evg, WRITE, 2, 0, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    NVN(nvnEventSignal, evc, WRITE, 4);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN(nvnCommandBufferDrawArrays, cb, 4, 0, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    NvnWaitSite sites[64], other;
    int n = gfx_null_wait_sites(sites, 64, &other);
    CHECK(n == 2);
    char name[96];
    for (int i = 0; i < n; i++) {
        const GfxNullWaitClasses *s = &sites[i].c;
        printf("site %s unsatisfied %llu later_gpu %llu later_cpu %llu pending %llu\n",
               site_name(sites[i].caller, name, sizeof name), (ull)s->unsatisfied, (ull)s->later_gpu, (ull)s->later_cpu,
               (ull)s->pending);
    }
    GfxNullWaitClasses k;
    gfx_null_wait_classes(&k);
    printf("totals unsatisfied %llu later_gpu %llu later_cpu %llu pending %llu\n", (ull)k.unsatisfied, (ull)k.later_gpu,
           (ull)k.later_cpu, (ull)k.pending);
    printf("ok wait_sites_order\n");
}

// ---------- case event_words: the writer of every event word is remembered (a game makes hundreds of events), and only a
// 4-byte write by the CPU is an event signal
static void case_event_words(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer();
    enum { N = 300 };
    uint64_t ev[N];
    for (int i = 0; i < N; i++) ev[i] = event_object(g_pool, RESULTS + 0x1000 + 8 * (uint64_t)i);
    NVN(nvnCommandBufferBeginRecording, cb);
    for (int i = 0; i < N; i += 2) NVN(nvnCommandBufferSignalEvent, cb, ev[i], WRITE, 2, 0, 3);   // the GPU signals the even ones
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    for (int i = 1; i < N; i += 2) NVN(nvnEventSignal, ev[i], WRITE, 3);                          // the CPU the odd ones
    NVN(nvnCommandBufferBeginRecording, cb);
    for (int i = 0; i < N; i++) NVN_AT(MB + 0x1000 + 0x100 * (uint64_t)(i & 1), nvnCommandBufferWaitEvent, cb, ev[i], EQUAL, 3);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    GfxNullWaitClasses k;
    gfx_null_wait_classes(&k);
    printf("event_words gpu %llu cpu %llu initial %llu unsatisfied %llu\n", (ull)k.by_gpu, (ull)k.by_cpu, (ull)k.initial,
           (ull)k.unsatisfied);
    // a write of 8 bytes to a word is not a signal; one of 4 bytes is
    uint64_t evg = event_object(g_pool, RESULTS + 0x2000), evh = event_object(g_pool, RESULTS + 0x2008);
    gfx_backend()->guest_written(g_pool_mem + RESULTS + 0x2000, 8);
    gfx_backend()->guest_written(g_pool_mem + RESULTS + 0x2008, 4);
    NVN(nvnCommandBufferBeginRecording, cb);
    NVN_AT(MB + 0x2000, nvnCommandBufferWaitEvent, cb, evg, GEQUAL_WRAP, 0);
    NVN_AT(MB + 0x2000, nvnCommandBufferWaitEvent, cb, evh, GEQUAL_WRAP, 0);
    submit(q, NVN(nvnCommandBufferEndRecording, cb));
    gfx_null_wait_classes(&k);
    printf("event_sizes gpu %llu cpu %llu initial %llu\n", (ull)k.by_gpu, (ull)k.by_cpu, (ull)k.initial);
    printf("ok event_words\n");
}

// ---------- failing cases
static void case_event_cpu_invisible(void) {
    setup(0, NULL);
    objects();
    uint64_t hidden = memory_pool(0x10000, CPU_NO_ACCESS | GPU_CACHED, NULL);
    event_object(hidden, 0);
}
static void case_sync_uninitialised(void) {
    setup(0, NULL);
    objects();
    sync_wait(galloc(0x40, 16), 0);
}
static void case_present_past_textures(void) {
    setup(0, NULL);
    objects();
    const uint64_t tex[2] = {texture(64, 64, 1), texture(64, 64, 2)};
    uint64_t q = queue(), w = window(2, tex);
    NVN(nvnQueuePresentTexture, q, w, 2);
}
static void case_window_five(void) {
    setup(0, NULL);
    objects();
    uint64_t arr = galloc(8 * 5, 8), b = galloc(0x40, 16);
    for (int i = 0; i < 5; i++) wr64(arr + 8ull * (uint64_t)i, texture(64, 64, (uint64_t)i + 1));
    NVN(nvnWindowBuilderSetDefaults, b);
    NVN(nvnWindowBuilderSetTextures, b, 5, arr);
}
static void case_report_no_pool(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer();
    report_once(q, cb, TIMESTAMP, 0x1000);
}

// ---------- cases census_*: the per-present counters and census mode (the null back end boot)
// n presents on a window of two textures at interval 0 (no wait on the clock), one empty recording each.
static int g_presents_interval;   // the window's present interval for presents()
static void presents(int n) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer(), s = sync_object();
    const uint64_t tex[2] = {texture(64, 64, 1), texture(64, 64, 2)};
    uint64_t w = window(2, tex);
    NVN(nvnWindowSetPresentInterval, w, g_presents_interval);
    for (int f = 0; f < n; f++) {
        if (f == 10) plat_sleep_ns(30000000);   // 30 ms between present 1 and present 60: the log line's seconds show it
        int i = acquire(w, s);
        NVN(nvnCommandBufferBeginRecording, cb);
        NVN(nvnCommandBufferDrawArrays, cb, 4, f, 3);
        submit(q, NVN(nvnCommandBufferEndRecording, cb));
        NVN(nvnQueuePresentTexture, q, w, (uint64_t)i);
    }
}
// Default mode: one log line for present 1 and for every 60th; no census-mode line; a gap is the named fatal.
static void case_census_default(void) {
    presents(125);
    CHECK(gfx_null_presents() == 125);
    printf("ok census_default\n");
}
// One window of the clock's own interval, 1: the line is not marked.
static void case_census_interval1(void) {
    g_presents_interval = 1;
    presents(61);
    CHECK(gfx_null_presents() == 61);
    printf("ok census_interval1\n");
}
static void case_census_gap_default(void) {
    setup(0, NULL);
    objects();
    NVN(nvnQueueSetDebugLabel, galloc(0x2000, 16), 0);
}
// Census mode (the driver sets OOT2_CENSUS=1): a gap answers 0 and is logged once per name; the table of calls per frame.
static void case_census_mode(void) {
    presents(130);
    NVN_AT(PC + 0x10, nvnQueueSetDebugLabel, 0, 0);
    CHECK(c.x[0] == 0);
    NVN_AT(PC + 0x20, nvnQueueSetDebugLabel, 0, 0);
    NVN_AT(PC + 0x30, nvnWindowGetNumTextures, 0);
    rt_call(&c, g_other);                // a host function that is no NVN2 entry point: not in census-nvn.tsv
    plat_sleep_ns(30000000);             // 30 ms after present 120: the tail window's seconds show it
    FILE *f = tmpfile();
    CHECK(f != NULL);
    nvn_census_write(f);
    rewind(f);
    char line[512];
    while (fgets(line, sizeof line, f)) printf("census| %s", line);
    nvn_census_stats(print_stat, NULL);
    printf("ok census_mode\n");
}
// ---------- cases title_*: the title line of the movie's frame (the movie player)
// n presents like presents(), but each frame's recording holds the calls shape() asks for: SetRenderTargets, DispatchCompute and
// DrawElementsInstanced (a window's counts a present are the front end's own calls over its presents).
typedef void (*Shape)(int f, int *rt, int *dp, int *dr);
static void shaped(int n, Shape shape) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), cb = command_buffer(), s = sync_object();
    const uint64_t tex[2] = {texture(64, 64, 1), texture(64, 64, 2)};
    uint64_t w = window(2, tex);
    NVN(nvnWindowSetPresentInterval, w, 0);
    for (int f = 0; f < n; f++) {
        int rt = 0, dp = 0, dr = 0;
        shape(f, &rt, &dp, &dr);
        int i = acquire(w, s);
        NVN(nvnCommandBufferBeginRecording, cb);
        for (int k = 0; k < rt; k++) NVN(nvnCommandBufferSetRenderTargets, cb, 0, 0, 0, 0, 0);
        for (int k = 0; k < dp; k++) NVN(nvnCommandBufferDispatchCompute, cb, 8, 4, 1);
        for (int k = 0; k < dr; k++) NVN(nvnCommandBufferDrawElementsInstanced, cb, 4, 2, 6, 0x8000, 0, 0, 1);
        submit(q, NVN(nvnCommandBufferEndRecording, cb));
        NVN(nvnQueuePresentTexture, q, w, (uint64_t)i);
    }
}
// The movie's frame from the start (10 targets, 14 dispatches), and the menu's two draws a present from present 181 on.
static void shape_movie(int f, int *rt, int *dp, int *dr) { *rt = 10; *dp = 14; *dr = f >= 180 ? 2 : 0; }
// One dispatch short of the movie's frame, and one target short of it.
static void shape_short_dispatch(int f, int *rt, int *dp, int *dr) { (void)f; (void)dr; *rt = 10; *dp = 13; }
static void shape_short_target(int f, int *rt, int *dp, int *dr) { (void)f; (void)dr; *rt = 9; *dp = 14; }
// The movie's frame, then 60 presents without it, then the movie's frame again: never two windows in a row.
static void shape_gap(int f, int *rt, int *dp, int *dr) {
    (void)dr;
    int plain = f >= 60 && f < 120;
    *rt = plain ? 0 : 10;
    *dp = plain ? 0 : 14;
}
static void case_title_movie(void) { shaped(300, shape_movie); printf("ok title_movie\n"); }
static void case_title_short_dispatch(void) { shaped(300, shape_short_dispatch); printf("ok title_short_dispatch\n"); }
static void case_title_short_target(void) { shaped(300, shape_short_target); printf("ok title_short_target\n"); }
static void case_title_gap(void) { shaped(180, shape_gap); printf("ok title_gap\n"); }

// A run that ends on a window boundary (present 120, a snapshot taken there) has no tail window.
static void case_census_boundary(void) {
    presents(120);
    FILE *f = tmpfile();
    CHECK(f != NULL);
    nvn_census_write(f);
    rewind(f);
    char line[512];
    while (fgets(line, sizeof line, f)) printf("census| %s", line);
    printf("ok census_boundary\n");
}
// The census is rewritten every 600 presents while a census-mode run goes on.
static int g_flushes;
static void count_flush(void) { g_flushes++; }
static void case_census_flush(void) {
    rt_census_set_flush(count_flush);
    presents(1250);
    CHECK(g_flushes == 2);
    printf("ok census_flush\n");
}
// Windows are kept up to 1024 snapshots; the first one that does not fit says so, once, and the table ends there.
static void case_census_cap(void) {
    setup(0, NULL);
    objects();
    uint64_t q = queue(), s = sync_object();
    const uint64_t tex[2] = {texture(64, 64, 1), texture(64, 64, 2)};
    uint64_t w = window(2, tex);
    NVN(nvnWindowSetPresentInterval, w, 0);
    for (int f = 0; f < 61500; f++) {   // no recording: the control memory of a command buffer would not last
        int i = acquire(w, s);
        NVN(nvnQueuePresentTexture, q, w, (uint64_t)i);
    }
    FILE *f = tmpfile();
    CHECK(f != NULL);
    nvn_census_write(f);
    rewind(f);
    char line[512];
    int windows = 0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "#window w", 9)) windows++;
    printf("windows kept %d\n", windows);
    printf("ok census_cap\n");
}
// With a frame limit the run ends at that present with its own exit code.
static void case_census_limit(void) {
    presents(20);
    printf("not reached\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: nvn_queue_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    static const struct { const char *name; void (*fn)(void); } cases[] = {
        {"fences", case_fences}, {"timestamps", case_timestamps}, {"render_cond", case_render_cond},
        {"window", case_window}, {"syncwait", case_syncwait}, {"wait_events", case_wait_events}, {"wait_sites", case_wait_sites}, {"wait_sites_order", case_wait_sites_order},
        {"event_cpu_invisible", case_event_cpu_invisible}, {"sync_uninitialised", case_sync_uninitialised},
        {"present_past_textures", case_present_past_textures}, {"window_five", case_window_five},
        {"report_no_pool", case_report_no_pool}, {"census_default", case_census_default}, {"census_interval1", case_census_interval1},
        {"census_at_draw", case_census_at_draw}, {"census_fatal_replay", case_census_fatal_replay},
        {"census_fatal_complete", case_census_fatal_complete}, {"census_fatal_held_complete", case_census_fatal_held_complete},
        {"census_fatal_held_release", case_census_fatal_held_release}, {"census_other_thread", case_census_other_thread},
        {"census_gap_default", case_census_gap_default}, {"census_mode", case_census_mode},
        {"census_cap", case_census_cap}, {"census_limit", case_census_limit}, {"census_flush", case_census_flush},
        {"census_boundary", case_census_boundary}, {"event_words", case_event_words},
        {"title_movie", case_title_movie}, {"title_short_dispatch", case_title_short_dispatch},
        {"title_short_target", case_title_short_target}, {"title_gap", case_title_gap},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(cs, cases[i].name)) {
            cases[i].fn();
            return 0;
        }
    fprintf(stderr, "unknown case %s\n", cs);
    return 2;
}
