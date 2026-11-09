// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The synchronisation checks of the NVN2 queue and sync work, on the Vulkan back end, on a real device (runtime/gpu/vk_replay_report.c,
// runtime/gpu/vk_capture.c), driven by tests/test_runtime_vk_sync.py: `vk_sync_test <case> [imports] [--pack PACK]
// [--mutant NAME]`, one case per process. They need a Vulkan device and skip without one, unless OOT2_REQUIRE_GPU is set.
//
// Recordings are made by hand (records of oot2/nvn_obj.h) and go through the back end's own submit, so the replay, the
// recorder thread, the completion thread and the backing map all run. While a case holds, each submission it makes waits
// for its own value of a gate semaphore, so the case can look at memory with the GPU's work not done and release the
// submissions one at a time. Results go to the CPU-visible pool, mirrored on this device unless `imports` is given.
//
//   fences       three submissions signal an event (WRITE 1, WRITE 2, ADD 10) and report a timestamp: nothing is written
//                before its submission is released, each fence is done in submission order, the event ends at 12
//   timestamps   reports in 31.25 MHz ticks of the host clock read at completion: bracketed by the clock around the
//                submit and the wait, two reports 20 ms apart, 64 reports of three types never going back, and a held
//                report stamped when it is released, not when it was submitted
//   samples      samples passed counted by occlusion queries: draws clipped to 5x3, 4x2 and 2x2 pixels, a draw with
//                rendering disabled, reports between them and resets
//   render_cond  seven draws in columns of a target, rendered or skipped by conditions on done and not-done reports
//                (renders, skipped, renders, renders, skipped, renders, renders)
//   events       events signalled in op-stream order after the frame's GPU copies are back in guest memory: a spinning CPU
//                that sees the event sees the copied bytes; WaitEvents checked at completion
//   present      three textures presented in order, each to the capture with the crop applied; a present right after a
//                held submission shows that submission's work
//
// --mutant replaystamp (timestamps at replay), noqueries (no occlusion query), nopending (the condition reads memory
// under a report not done), reverse (completion walks a frame backwards) and nocrop (a present shows the whole texture)
// are the counter-runs: the case must fail.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "layout.h"
#include "oot2/backing.h"
#include "oot2/cpu.h"
#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/vk_capture.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3
#define MiB (1ull << 20)
#define WIN GFX_GPU_WINDOW_BASE
typedef unsigned long long ull;

#define VS_KEY "5100000000000001.00000000"
#define PS_KEY "5100000000000002.00000000"

static Cpu g_cpu;
static uint64_t g_heap;
static pool_t g_cpool, g_gpool;   // the CPU-visible pool and the GPU's own, 4 MiB each
#define P (g_heap + 4 * MiB)      // the CPU-visible pool's first byte
#define G WIN                     // the GPU pool's

static void guest_heap(uint64_t size) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = size;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1];
}

static pool_t make_pool(uint64_t mem, uint64_t size, uint32_t flags) {
    pool_t p = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = flags, .mem = (uint8_t *)(uintptr_t)mem, .size = size,
                .game_mem = (uint8_t *)(uintptr_t)mem};
    CHECK(gfx_vulkan_backend.pool_init(&p));
    return p;
}

// ---------- the gate: while held, each submission waits for its own value of a semaphore the case signals
static VkRecSubmit g_sub;
static uint64_t (*g_dev_end)(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits);
static VkSemaphore g_gate;
static int g_to_hold;     // replayed frames still to hold: the recorder thread ends them, in order, after submit returned
static uint64_t g_held;   // gate values handed to held submissions

static uint64_t gated_end(const VkSemaphoreSubmitInfo *waits, uint32_t n) {
    if (__atomic_load_n(&g_to_hold, __ATOMIC_ACQUIRE) <= 0) return g_dev_end(waits, n);
    __atomic_sub_fetch(&g_to_hold, 1, __ATOMIC_ACQ_REL);
    CHECK(!n);
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_gate, .value = ++g_held,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    return g_dev_end(&wait, 1);
}

static void release(uint64_t k) {   // the first k held submissions may run
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = g_gate, .value = k};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
}

static int g_pack_open;

static void setup(int imports) {
    guest_heap(64 * MiB);
    gpu_vk_open();
    if (imports) backing_set_mirror_hook(NULL);
    g_cpool = make_pool(P, 4 * MiB, 0);
    g_gpool = make_pool(G, 4 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    memset((void *)(uintptr_t)P, 0, 4 * MiB);
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    CHECK(vkCreateSemaphore(gpu_vk.device, &ci, NULL, &g_gate) == VK_SUCCESS);
    vkrec_submit_device(&g_sub);
    g_dev_end = g_sub.end;
    g_sub.end = gated_end;
    VkRecOpen o = {.submit = &g_sub, .frames = 4};   // three frames held at once
    vkrec_open(&o);
}

static void finish_and_report(const char *name) {
    gfx_vulkan_backend.finish(0);
    if (g_pack_open) vkp_close(), vkp_store_close();
    gfx_vulkan_backend.pool_fini(&g_gpool);
    gfx_vulkan_backend.pool_fini(&g_cpool);
    vkDestroySemaphore(gpu_vk.device, g_gate, NULL);
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
    printf("ok %s\n", name);
}

// ---------- recordings
typedef struct {
    _Alignas(8) uint8_t b[1 << 16];
    size_t n;
} Rec;
static Rec g_recs[8];
static int g_next_rec;

static void put(Rec *r, uint32_t op, const void *payload, size_t bytes) {
    size_t size = (sizeof(op_hdr) + bytes + 7) & ~(size_t)7;
    CHECK(r->n + size <= sizeof r->b);
    op_hdr h = {op, (uint32_t)size};
    memset(r->b + r->n, 0, size);
    memcpy(r->b + r->n, &h, sizeof h);
    if (bytes) memcpy(r->b + r->n + sizeof h, payload, bytes);
    r->n += size;
}
#define PUT(r, op, ...) ({ __typeof__(__VA_ARGS__) p_ = __VA_ARGS__; put((r), (op), &p_, sizeof p_); })
// A recording of its own (the replay reads it during submit only, but held cases keep several apart for clarity).
static Rec *rec_begin(void) {
    Rec *r = &g_recs[g_next_rec++ % 8];
    r->n = 0;
    put(r, OP_BEGIN, NULL, 0);
    return r;
}
static uint64_t submit(Rec *r) {
    put(r, OP_END, NULL, 0);
    uint64_t h = (uint64_t)(uintptr_t)r->b;
    uint64_t v = gfx_vulkan_backend.submit(1, &h, 1);
    CHECK(v);
    return v;
}
static void wait_done(uint64_t v) {
    CHECK(gfx_vulkan_backend.fence_wait(v, 20000000000ull) == 1);
    gfx_vulkan_backend.cpu_synced(v);
}

static uint64_t rd64(uint64_t a) { uint64_t v; memcpy(&v, (const void *)(uintptr_t)a, 8); return v; }
static void wr64(uint64_t a, uint64_t v) { memcpy((void *)(uintptr_t)a, &v, 8); }
static uint32_t rd32(uint64_t a) { return __atomic_load_n((const volatile uint32_t *)(uintptr_t)a, __ATOMIC_ACQUIRE); }

static void report(Rec *r, uint64_t type, uint64_t addr) { PUT(r, OP_REPORT_COUNTER, (p_report_counter){type, addr}); }
static void signal_event(Rec *r, uint64_t addr, uint32_t mode, uint32_t value) {
    PUT(r, OP_SIGNAL_EVENT, (p_signal_event){(volatile uint32_t *)(uintptr_t)addr, mode, value, 0, 0});
}

static tex_t tex_at(uint64_t storage, int format, uint32_t w, uint32_t h) {
    tex_t t;
    memset(&t, 0, sizeof t);
    t.magic = NVN_TEX_MAGIC;
    t.d.target = NVN_TEXTURE_TARGET_2D, t.d.format = format, t.d.width = w, t.d.height = h, t.d.depth = 1, t.d.levels = 1;
    t.d.swizzle_r = NVN_TEXTURE_SWIZZLE_R, t.d.swizzle_g = NVN_TEXTURE_SWIZZLE_G, t.d.swizzle_b = NVN_TEXTURE_SWIZZLE_B;
    t.d.swizzle_a = NVN_TEXTURE_SWIZZLE_A;
    t.mem = (uint8_t *)(uintptr_t)storage;
    static uint64_t serial;
    t.serial = ++serial;
    nvn_tex_layout(&t.d, &t.L);
    return t;
}
static nvn_tex_rec trec(const tex_t *t) {
    nvn_tex_rec r;
    memset(&r, 0, sizeof r);
    r.tex = *t;
    r.present = 1;
    return r;
}
static NVNcopyRegion region(int x, int y, int w, int h) { return (NVNcopyRegion){x, y, 0, w, h, 1}; }

// ---------- the synthetic pack's draw (tests/test_runtime_vk_pipeline.py): a triangle over the whole target, its colour
// from storage and a uniform (the uniform_draw case of tests/runtime/vk_xfer_test.c)
static nvn_stage g_vs, g_ps;
static ShaderBinds g_vs_binds, g_ps_binds;
static VkReplayOps g_ops;
static const VkpFormats F32 = {1, {VK_FORMAT_R32G32B32A32_SFLOAT}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};

static void stage(nvn_stage *s, const char *key, int kind) {
    memset(s, 0, sizeof *s);
    snprintf(s->key, sizeof s->key, "%s", key);
    s->kind = (uint8_t)kind;
    s->ctl_size = NVN_STAGE_CTL_MAX;
}
static void cbuf(ShaderBinds *b, uint8_t slot, uint16_t binding) { b->cb[b->ncb++] = (ShdCbuf){slot, 1, binding}; }
static const ShaderBinds *my_binds(const VkpStage *stage) {
    const nvn_stage *s = stage->shape;
    return s == &g_vs ? &g_vs_binds : s == &g_ps ? &g_ps_binds : vkr_device_ops()->binds(stage);
}

// The test's pools are the back end's alone (no front end): the CPU reads whatever is outside the GPU window.
static int my_cpu_reads(uint64_t addr, uint64_t size) { (void)size; return !gfx_gpu_only(addr); }

static void open_pack(const char *pack) {
    char why[512];
    if (!pack || !vkp_store_open(pack, NULL, why, sizeof why)) {
        fprintf(stderr, "the pack did not open: %s\n", pack ? why : "no --pack");
        exit(1);
    }
    vkp_open(NULL);
    g_pack_open = 1;
    stage(&g_vs, VS_KEY, NVN_STAGE_VS);
    stage(&g_ps, PS_KEY, NVN_STAGE_PS);
    ShaderBinds *v = &g_vs_binds, *p = &g_ps_binds;
    memset(v, 0, sizeof *v), memset(p, 0, sizeof *p);
    v->kind = SHD_VS, v->fits = 1;
    cbuf(v, 3, 1);
    v->sb[v->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 0, .binding = 9, .address_byte_offset = 0x110};
    p->kind = SHD_PS, p->fits = 1;
    cbuf(p, 4, 14);
    p->sb[p->nsb++] = (ShdSbuf){.set = 1, .ordinal = 0, .cb_slot = 0, .write = 0, .binding = 25, .address_byte_offset = 0x510};
    g_ops = *vkr_device_ops();
    g_ops.binds = my_binds;
    g_ops.cpu_reads = my_cpu_reads;
    vkr_set_ops(&g_ops);
    VkpStage vs = {.key = VS_KEY}, ps = {.key = PS_KEY};
    CHECK(vkp_graphics(&vs, NULL, &ps, &F32));   // made now: no pipeline is made with a submission held
    const float colour[4] = {0.25f, 0.5f, 0.75f, 0.0f}, mul[4] = {1, 1, 1, 1}, u[4] = {0, 0, 0, 0};
    memcpy((void *)(uintptr_t)(P + 0x2000), colour, 16);
    memcpy((void *)(uintptr_t)(P + 0x2100), mul, 16);
    memcpy((void *)(uintptr_t)(P + 0x3000), u, 16);
}

// The draw's target (RGBA32F, w x h in the GPU pool), cleared to -7, and the program and its buffers.
static tex_t g_target;
static void draw_state(Rec *r, uint32_t w, uint32_t h) {
    g_target = tex_at(G + 0x40000, NVN_FORMAT_RGBA32F, w, h);
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1;
    rt.color[0] = trec(&g_target);
    put(r, OP_SET_RT, &rt, sizeof rt);
    PUT(r, OP_CLEAR_COLOR, (p_clear_color){.index = 0, .mask = 0xF, .rgba = {-7, -7, -7, -7}});
    p_bind_program pg;
    memset(&pg, 0, sizeof pg);
    pg.stages = 1 << 0 | 1 << 1, pg.nstage = 2, pg.id[0] = &g_vs, pg.id[1] = &g_ps;
    PUT(r, OP_BIND_PROGRAM, pg);
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 0, .index = 0, .addr = P + 0x2000, .size = 16});
    PUT(r, OP_BIND_SSBO, (p_bind_buf){.stage = 1, .index = 0, .addr = P + 0x2100, .size = 16});
    PUT(r, OP_BIND_UBO, (p_bind_buf){.stage = 1, .index = 1, .addr = P + 0x3000, .size = 16});
}
// A draw clipped to the scissor (x, y, w, h), in rows measured as the window origin measures them.
static void draw_in(Rec *r, int x, int y, int w, int h) {
    PUT(r, OP_SCISSOR, (p_scissor){.first = 0, .n = 1, .r = {{x, y, w, h}}});
    PUT(r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = 3});
}

// ---------- fences
static void case_fences(int imports) {
    setup(imports);
    const uint64_t ev = P + 0x1000, rep = P + 0x1100;
    g_to_hold = 3;   // the three submissions below
    const struct { uint32_t mode, value; } sub[3] = {{NVN_EVENT_SIGNAL_MODE_WRITE, 1}, {NVN_EVENT_SIGNAL_MODE_WRITE, 2},
                                                     {NVN_EVENT_SIGNAL_MODE_ADD, 10}};
    uint64_t v[3];
    for (int k = 0; k < 3; k++) {
        Rec *r = rec_begin();
        signal_event(r, ev, sub[k].mode, sub[k].value);
        report(r, NVN_COUNTER_TYPE_TIMESTAMP, rep + 16ull * (uint64_t)k);
        v[k] = submit(r);
        CHECK(k == 0 || v[k] > v[k - 1]);
    }
    plat_sleep_ns(100 * 1000 * 1000);
    CHECK(rd32(ev) == 0 && gfx_vulkan_backend.fence_done() < v[0]);
    for (int k = 0; k < 3; k++) CHECK(rd64(rep + 16ull * (uint64_t)k + 8) == 0);
    printf("held: no event, no report, no fence\n");
    const uint32_t want[3] = {1, 2, 12};
    for (int k = 0; k < 3; k++) {
        release((uint64_t)k + 1);
        wait_done(v[k]);
        plat_sleep_ns(50 * 1000 * 1000);   // what is still held stays held
        CHECK(gfx_vulkan_backend.fence_done() == v[k]);
        CHECK(rd32(ev) == want[k]);
        for (int j = 0; j < 3; j++) CHECK((rd64(rep + 16ull * (uint64_t)j + 8) != 0) == (j <= k));
        printf("complete %d event %u\n", k + 1, rd32(ev));
    }
    finish_and_report("fences");
}

// ---------- timestamps
static void case_timestamps(int imports) {
    setup(imports);
    const uint64_t a = P + 0x1000, b = a + 16;
    uint64_t t0 = rt_ticks(), n0 = plat_now_ns();
    Rec *r = rec_begin();
    report(r, NVN_COUNTER_TYPE_TIMESTAMP, a);
    wait_done(submit(r));
    uint64_t n1 = plat_now_ns(), t1 = rt_ticks();
    plat_sleep_ns(20 * 1000 * 1000);
    uint64_t n2 = plat_now_ns();
    r = rec_begin();
    report(r, NVN_COUNTER_TYPE_TIMESTAMP, b);
    wait_done(submit(r));
    uint64_t n3 = plat_now_ns();
    uint64_t ta = rd64(a + 8), tb = rd64(b + 8);
    CHECK(rd64(a) == 0 && rd64(b) == 0);   // the counter word of a timestamp
    CHECK(t0 <= ta && ta <= t1);
    CHECK(ta < tb);
    uint64_t span = (tb - ta) * 32;        // 32 ns a tick
    CHECK(span + 64 >= n2 - n1 && span <= n3 - n0 + 64);
    printf("two reports %llu ticks apart, %llu ns, the host between %llu and %llu ns\n", (ull)(tb - ta), (ull)span,
           (ull)(n2 - n1), (ull)(n3 - n0));
    // sixty-four reports of three types in one recording: every one stamped, never going back
    const uint64_t many = a + 0x100;
    r = rec_begin();
    for (int i = 0; i < 64; i++)
        report(r, i % 3 == 0 ? NVN_COUNTER_TYPE_TIMESTAMP : i % 3 == 1 ? 0x10 : NVN_COUNTER_TYPE_SAMPLES_PASSED,
               many + 16ull * (uint64_t)i);
    wait_done(submit(r));
    for (int i = 0; i < 64; i++) {
        uint64_t at = many + 16ull * (uint64_t)i;
        CHECK(rd64(at) == 0 && rd64(at + 8) >= tb && (i == 0 || rd64(at + 8) >= rd64(at - 8)));
    }
    printf("64 reports stamped in order\n");
    // held: the stamp is the time of completion, not of the submit
    const uint64_t h = a + 0x800;
    __atomic_store_n(&g_to_hold, 1, __ATOMIC_RELEASE);   // the next submission
    r = rec_begin();
    report(r, NVN_COUNTER_TYPE_TIMESTAMP, h);
    uint64_t v = submit(r);
    uint64_t submitted = rt_ticks();
    plat_sleep_ns(10 * 1000 * 1000);
    uint64_t c0 = rt_ticks();
    CHECK(rd64(h + 8) == 0);
    release(1);
    wait_done(v);
    uint64_t c1 = rt_ticks();
    printf("held report: submitted at %llu, released at %llu, stamped %llu, waited at %llu\n", (ull)submitted, (ull)c0,
           (ull)rd64(h + 8), (ull)c1);
    CHECK(rd64(h + 8) > submitted && c0 <= rd64(h + 8) && rd64(h + 8) <= c1);
    finish_and_report("timestamps");
}

// ---------- samples passed
static void case_samples(int imports, const char *pack) {
    setup(imports);
    open_pack(pack);
    const uint64_t c = P + 0x1000;   // reports at c + 16 k
    Rec *r = rec_begin();
    draw_state(r, 32, 32);
    PUT(r, OP_RESET_COUNTER, (p_reset_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED});
    report(r, NVN_COUNTER_TYPE_SAMPLES_PASSED, c);
    draw_in(r, 0, 0, 5, 3);
    report(r, NVN_COUNTER_TYPE_SAMPLES_PASSED, c + 16);
    draw_in(r, 10, 10, 4, 2);
    report(r, NVN_COUNTER_TYPE_SAMPLES_PASSED, c + 32);
    submit(r);
    r = rec_begin();
    PUT(r, OP_RENDER_ENABLE, (p_flag){0});
    draw_in(r, 0, 0, 32, 32);   // skipped: nothing counted
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    draw_in(r, 20, 20, 2, 2);
    report(r, NVN_COUNTER_TYPE_SAMPLES_PASSED, c + 48);
    PUT(r, OP_RESET_COUNTER, (p_reset_counter){NVN_COUNTER_TYPE_SAMPLES_PASSED});
    report(r, NVN_COUNTER_TYPE_SAMPLES_PASSED, c + 64);
    wait_done(submit(r));
    printf("samples passed: %llu %llu %llu %llu %llu\n", (ull)rd64(c), (ull)rd64(c + 16), (ull)rd64(c + 32), (ull)rd64(c + 48),
           (ull)rd64(c + 64));
    CHECK(rd64(c) == 0 && rd64(c + 16) == 15 && rd64(c + 32) == 23 && rd64(c + 48) == 27 && rd64(c + 64) == 0);
    for (int k = 0; k < 5; k++) CHECK(rd64(c + 16ull * (uint64_t)k + 8) != 0);
    VkrReportStats rs;
    vkr_report_stats(&rs);
    printf("queries %llu resets %llu\n", (ull)rs.queries, (ull)rs.resets);
    finish_and_report("samples");
}

// ---------- render_cond
static void case_render_cond(int imports, const char *pack) {
    setup(imports);
    open_pack(pack);
    const uint64_t cnt = P + 0x1000;   // two NVNcounterData: cnt and cnt + 16
    Rec *r = rec_begin();
    draw_state(r, 8, 4);
    // done reports: memory decides
    wr64(cnt, 5), wr64(cnt + 16, 5);
    PUT(r, OP_RENDER_COND, (p_render_cond){0, 0, cnt});
    draw_in(r, 1, 0, 1, 4);   // renders
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, cnt});
    draw_in(r, 2, 0, 1, 4);   // skipped
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    draw_in(r, 3, 0, 1, 4);   // renders
    wait_done(submit(r));
    // a timestamp report not done counts as 0: memory says 7 and 0, the condition sees 0 and 0
    wr64(cnt, 7), wr64(cnt + 16, 0);
    r = rec_begin();
    report(r, NVN_COUNTER_TYPE_TIMESTAMP, cnt);
    PUT(r, OP_RENDER_COND, (p_render_cond){0, 0, cnt});
    draw_in(r, 4, 0, 1, 4);   // renders
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    wait_done(submit(r));
    // once done, the report's counter word (0) is read from memory
    CHECK(rd64(cnt) == 0 && rd64(cnt + 8) != 0);
    r = rec_begin();
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, cnt});
    draw_in(r, 5, 0, 1, 4);   // skipped
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    wait_done(submit(r));
    // a samples-passed report not done reads as visible: memory and 0 would both say equal
    wr64(cnt, 0), wr64(cnt + 16, 0);
    r = rec_begin();
    report(r, NVN_COUNTER_TYPE_SAMPLES_PASSED, cnt + 16);
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, cnt});
    draw_in(r, 6, 0, 1, 4);   // renders
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    wait_done(submit(r));
    // a timestamp report in an earlier submission that is held counts as 0 too: memory says 9 and 9
    wr64(cnt, 9), wr64(cnt + 16, 9);
    __atomic_store_n(&g_to_hold, 1, __ATOMIC_RELEASE);   // the next submission
    r = rec_begin();
    report(r, NVN_COUNTER_TYPE_TIMESTAMP, cnt);
    submit(r);
    r = rec_begin();
    PUT(r, OP_RENDER_COND, (p_render_cond){1, 0, cnt});
    draw_in(r, 7, 0, 1, 4);   // renders
    PUT(r, OP_RENDER_ENABLE, (p_flag){1});
    submit(r);
    release(1);
    r = rec_begin();
    PUT(r, OP_COPY_T2B, (p_copy_b2t){.addr = P + 0x4000, .tex = trec(&g_target), .r = region(0, 0, 8, 4)});
    wait_done(submit(r));
    char row[9] = {0};
    for (int x = 0; x < 8; x++) {
        float px[4];
        memcpy(px, (const void *)(uintptr_t)(P + 0x4000 + 16ull * (uint64_t)x), 16);
        row[x] = px[0] == -7.0f ? '.' : 'R';
    }
    printf("columns 0 to 7: %s\n", row);
    CHECK(!strcmp(row, ".R.RR.RR"));
    VkrReportStats rs;
    vkr_report_stats(&rs);
    printf("conditions %llu false %llu\n", (ull)rs.conditions, (ull)rs.conditions_false);
    finish_and_report("render_cond");
}

// ---------- events
static void case_events(int imports) {
    setup(imports);
    const uint64_t ev = P + 0x1000, sum = P + 0x1004, src = P + 0x8000, dst = P + 0x9000;
    int stale = 0;
    for (uint32_t k = 1; k <= 20; k++) {
        for (int i = 0; i < 64; i++) ((volatile uint32_t *)(uintptr_t)src)[i] = k * 1000 + (uint32_t)i;
        Rec *r = rec_begin();
        PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = src, .dst = G + 0x100, .size = 256});
        PUT(r, OP_COPY_B2B, (p_copy_b2b){.src = G + 0x100, .dst = dst, .size = 256});
        signal_event(r, ev, NVN_EVENT_SIGNAL_MODE_WRITE, k);
        submit(r);
        uint64_t t0 = plat_now_ns();
        while (rd32(ev) != k) CHECK(plat_now_ns() - t0 < 10000000000ull);
        for (int i = 0; i < 64; i++) stale += ((volatile uint32_t *)(uintptr_t)dst)[i] != k * 1000 + (uint32_t)i;
    }
    printf("20 events seen by a spinning CPU, %d stale words of the copies before them\n", stale);
    CHECK(stale == 0);
    Rec *r = rec_begin();
    signal_event(r, sum, NVN_EVENT_SIGNAL_MODE_WRITE, 5);
    signal_event(r, sum, NVN_EVENT_SIGNAL_MODE_ADD, 3);
    PUT(r, OP_WAIT_EVENT, (p_wait_event){(volatile uint32_t *)(uintptr_t)sum, NVN_EVENT_WAIT_MODE_EQUAL, 8, 0x7100});
    PUT(r, OP_WAIT_EVENT, (p_wait_event){(volatile uint32_t *)(uintptr_t)sum, NVN_EVENT_WAIT_MODE_GEQUAL_WRAP, 9, 0x7200});
    wait_done(submit(r));
    VkrReportStats rs;
    vkr_report_stats(&rs);
    printf("event %u; waits %llu unsatisfied %llu\n", rd32(sum), (ull)rs.waits, (ull)rs.waits_unsatisfied);
    CHECK(rd32(sum) == 8 && rs.waits == 2 && rs.waits_unsatisfied == 1);
    finish_and_report("events");
}

// ---------- present
static int capture_is(uint32_t want_w, uint32_t want_h, uint32_t rgba, uint64_t want_seq) {
    static uint8_t px[256 * 256 * 4];
    uint32_t w, h;
    uint64_t seq = vkc_capture_read(px, sizeof px, &w, &h);
    if (seq != want_seq || w != want_w || h != want_h) {
        printf("capture: present %llu, %ux%u\n", (ull)seq, w, h);
        return 0;
    }
    for (uint32_t i = 0; i < w * h; i++)
        if (memcmp(px + 4 * i, &rgba, 4)) {
            printf("capture pixel %u: %02x%02x%02x%02x, not %08x\n", i, px[4 * i], px[4 * i + 1], px[4 * i + 2],
                   px[4 * i + 3], rgba);
            return 0;
        }
    return 1;
}

static void clear_tex(Rec *r, const tex_t *t, NVNcopyRegion rg, const float c[4]) {
    p_clear_tex p = {.tex = trec(t), .r = rg, .mask = 0xF};
    memcpy(p.c.f, c, 16);
    PUT(r, OP_CLEAR_TEX, p);
}

static void case_present(int imports) {
    setup(imports);
    tex_t tex[3];
    // texture k: blue-ish everywhere, its own colour in its first 32 x 16 texels (rows from the first in memory)
    const float own[3][4] = {{1, 0, 0, 1}, {0, 1, 0, 1}, {1, 1, 0, 1}}, rest[4] = {0, 0, 1, 1};
    const uint32_t own_rgba[3] = {0xFF0000FFu, 0xFF00FF00u, 0xFF00FFFFu};   // bytes R, G, B, A read as little-endian
    Rec *r = rec_begin();
    for (int k = 0; k < 3; k++) {
        tex[k] = tex_at(G + 0x100000 * (uint64_t)(k + 1), NVN_FORMAT_RGBA8, 64, 64);
        clear_tex(r, &tex[k], region(0, 0, 0, 0), rest);
        clear_tex(r, &tex[k], region(0, 0, 32, 16), own[k]);
    }
    wait_done(submit(r));
    const int crop[4] = {0, 0, 32, 16};
    for (int f = 0; f < 9; f++) {
        gfx_vulkan_backend.present(&tex[f % 3], crop);
        CHECK(capture_is(32, 16, own_rgba[f % 3], (uint64_t)f + 1));
        printf("present %d: texture %d in the capture\n", f + 1, f % 3);
    }
    const int whole[4] = {0, 0, 0, 0};
    gfx_vulkan_backend.present(&tex[0], whole);
    {
        static uint8_t px[64 * 64 * 4];
        uint32_t w, h;
        CHECK(vkc_capture_read(px, sizeof px, &w, &h) == 10 && w == 64 && h == 64);
        uint32_t corner, far;
        memcpy(&corner, px, 4), memcpy(&far, px + 4 * (63 * 64 + 63), 4);
        CHECK(corner == own_rgba[0] && far == 0xFFFF0000u);
    }
    // a present right after a held submission shows that submission's work
    const float white[4] = {1, 1, 1, 1};
    __atomic_store_n(&g_to_hold, 1, __ATOMIC_RELEASE);   // the next submission
    r = rec_begin();
    clear_tex(r, &tex[1], region(0, 0, 0, 0), white);
    submit(r);
    gfx_vulkan_backend.present(&tex[1], crop);
    plat_sleep_ns(100 * 1000 * 1000);
    release(1);
    int ok = capture_is(32, 16, 0xFFFFFFFFu, 11);
    printf("present after a held clear: %s\n", ok ? "the clear's white" : "NOT the clear's white");
    CHECK(ok);
    if (getenv("OOT2_SYNC_TEST_SAVE")) CHECK(vkc_capture_save(getenv("OOT2_SYNC_TEST_SAVE")));
    finish_and_report("present");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_sync_test <case> [imports] [--pack PACK] [--mutant NAME]\n");
        return 2;
    }
    int imports = 0;
    const char *pack = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "imports")) imports = 1;
        else if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!strcmp(argv[i], "--mutant") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "replaystamp")) vkr_stamp_at_replay = 1;
            else if (!strcmp(m, "noqueries")) vkr_no_queries = 1;
            else if (!strcmp(m, "nopending")) vkr_no_pending = 1;
            else if (!strcmp(m, "reverse")) vkr_reverse_results = 1;
            else if (!strcmp(m, "nocrop")) vkc_no_crop = 1;
            else {
                fprintf(stderr, "unknown mutant %s\n", m);
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        return EXIT_NO_DEVICE;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "fences")) case_fences(imports);
    else if (!strcmp(cs, "timestamps")) case_timestamps(imports);
    else if (!strcmp(cs, "samples")) case_samples(imports, pack);
    else if (!strcmp(cs, "render_cond")) case_render_cond(imports, pack);
    else if (!strcmp(cs, "events")) case_events(imports);
    else if (!strcmp(cs, "present")) case_present(imports);
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
