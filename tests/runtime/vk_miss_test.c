// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the miss path (runtime/gpu/vk_miss.c, oot2/vk_miss.h), driven by tests/test_runtime_vk_miss.py:
// `vk_miss_test <case> [args] [--mutant NAME]`, one case per process. A passing case prints "ok <case>"; a failing check
// exits 1 with the line, a named fatal exits 70.
//
// Cases that need no device (continuous integration runs them): the miss path over stand-in builds, the real recorder
// over stand-in calls and the real timeline over a counter standing in for the device's semaphore.
//   task_stage        a task stage survives the asynchronous graphics miss build
//   drain_order       a miss in the middle of a frame: the ring is empty and the timeline idle when the build starts
//   bound_skip        a graphics build past the bound: the draw is skipped, counted, the next draw of it skipped at once,
//                     the build finishes in its job and serves the draw after it
//   override          the same with the override: the draw waits, is not skipped, and the wait past the bound is counted
//   kept_wait         a draw that may not be skipped waits without a bound and the wait is counted apart
//   compute_wait      a compute miss waits for its build, whatever the bound, and is never skipped
//   release_order     a key's pipelines go once no program holds it and the timeline has passed the release
//   env VALUE         OOT2_VK_MISS_BOUND (the case sets nothing: the driver does) read at the first miss
// Cases that need a device with every item of oot2/vk_required.h, on a synthetic pack of GLSL modules (the driver makes
// it, as tests/test_runtime_vk_pipeline.py does):
//   dev_drain PACK    a real graphics miss in the middle of a frame, with a submission the device holds: the build waits
//                     for it, and the builder's own check (no creation with work in flight) agrees
//   dev_release PACK  a pipeline bound in a command buffer the device holds is released: destroyed only after the value
//   dev_late PACK     a build past a tiny bound goes on beside work in flight and is counted
// --mutant NAME: nodrain (no drain), nocount (a skip not counted), release_now (a released key destroyed at once), nolate
// (the build is not told the replay stopped waiting). The check of the case must then fail, which the driver requires.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/nvn_program.h"
#include "oot2/runtime.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_rec.h"
#include "plat.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3
typedef unsigned long long ull;

static void sleep_ms(uint64_t ms) { plat_sleep_ns(ms * 1000000ull); }

// ---------- a test's timeline: a counter standing in for the device's semaphore
static uint64_t g_counter;
static PlatMutex g_counter_lock = PLAT_MUTEX_INIT;
static PlatCond g_counter_cond = PLAT_COND_INIT;
static void counter_raise(uint64_t v) {
    plat_mutex_lock(&g_counter_lock);
    if (v > g_counter) g_counter = v;
    plat_cond_broadcast(&g_counter_cond);
    plat_mutex_unlock(&g_counter_lock);
}
static uint64_t counter_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    uint64_t t0 = plat_now_ns();
    plat_mutex_lock(&g_counter_lock);
    while (g_counter < value) {
        uint64_t spent = plat_now_ns() - t0;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_counter_cond, &g_counter_lock, timeout_ns - spent);
    }
    uint64_t r = g_counter;
    plat_mutex_unlock(&g_counter_lock);
    return r;
}
static void start_counter(void) {
    g_counter = 0;
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
}

// ---------- stand-in builds
#define PIPE_G ((VkPipeline)(uintptr_t)0x101)
#define PIPE_C ((VkPipeline)(uintptr_t)0x102)
static const nvn_stage g_vs = {.key = "5100000000000001.00000000", .kind = NVN_STAGE_VS};
static const nvn_stage g_ts = {.key = "5100000000000004.00000000", .kind = NVN_STAGE_TS};
static const nvn_stage g_ps = {.key = "5100000000000002.00000000", .kind = NVN_STAGE_PS};
static const nvn_stage g_cs = {.key = "5100000000000003.00000000", .kind = NVN_STAGE_CS};
static const VkpFormats g_fmt = {1, {VK_FORMAT_R8G8B8A8_UNORM}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};

static uint64_t g_build_ms;
static int g_g_built, g_c_built, g_late_at_end, g_builds, g_expect_task;
static uint64_t g_at_executed, g_at_done, g_at_issued;   // what the build saw when it started (the drain case)
static uint64_t g_sent, g_executed;                       // packets put, calls made

static VkPipeline f_find_g(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *f) {
    (void)vs, (void)task, (void)ps, (void)f;
    return __atomic_load_n(&g_g_built, __ATOMIC_ACQUIRE) ? PIPE_G : VK_NULL_HANDLE;
}
static VkPipeline f_find_c(const VkpStage *cs) {
    (void)cs;
    return __atomic_load_n(&g_c_built, __ATOMIC_ACQUIRE) ? PIPE_C : VK_NULL_HANDLE;
}
static VkPipeline f_build_g(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *f,
                            const int *late) {
    (void)vs, (void)ps, (void)f;
    CHECK((task != NULL) == g_expect_task);
    if (task) CHECK(!strcmp(task->key, g_ts.key));
    __atomic_add_fetch(&g_builds, 1, __ATOMIC_RELAXED);
    g_at_executed = __atomic_load_n(&g_executed, __ATOMIC_ACQUIRE);
    g_at_done = gpu_sync_done(), g_at_issued = gpu_sync_issued();
    sleep_ms(g_build_ms);
    g_late_at_end = late ? __atomic_load_n(late, __ATOMIC_ACQUIRE) : 0;
    __atomic_store_n(&g_g_built, 1, __ATOMIC_RELEASE);
    return PIPE_G;
}
static VkPipeline f_build_c(const VkpStage *cs) {
    (void)cs;
    __atomic_add_fetch(&g_builds, 1, __ATOMIC_RELAXED);
    sleep_ms(g_build_ms);
    __atomic_store_n(&g_c_built, 1, __ATOMIC_RELEASE);
    return PIPE_C;
}
static void f_key(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *f, char *out,
                  size_t cap) {
    (void)task;
    (void)f;
    snprintf(out, cap, "g|%s|%s", vs->key, ps->key);
}
static uint64_t f_drain(void) { return 0; }

static char g_released[16][32];
static int g_nreleased;
static uint64_t g_handed, g_done;
static void f_release(const char *key) {
    CHECK(g_nreleased < 16);
    snprintf(g_released[g_nreleased++], sizeof g_released[0], "%s", key);
}
static uint64_t f_handed(void) { return g_handed; }
static uint64_t f_done(void) { return g_done; }

static VkmOps fake_ops(int own_drain, int own_values) {
    VkmOps o = {.find_graphics = f_find_g, .find_compute = f_find_c, .build_graphics = f_build_g,
                .build_compute = f_build_c, .graphics_key = f_key, .release = f_release};
    if (own_drain) o.drain = f_drain;
    if (own_values) o.handed = f_handed, o.done = f_done;
    return o;
}

static VkpStage vstage(void) { return (VkpStage){.key = g_vs.key, .shape = &g_vs}; }
static VkpStage tstage(void) { return (VkpStage){.key = g_ts.key, .shape = &g_ts}; }
static VkpStage pstage(void) { return (VkpStage){.key = g_ps.key, .shape = &g_ps}; }

static void wait_built(void) {
    for (int i = 0; i < 400 && !__atomic_load_n(&g_g_built, __ATOMIC_ACQUIRE); i++) sleep_ms(10);
    CHECK(__atomic_load_n(&g_g_built, __ATOMIC_ACQUIRE));
}

// ---------- task_stage
static void case_task_stage(void) {
    VkmOps o = fake_ops(1, 0);
    vkm_set_ops(&o);
    VkmConfig c = {0, 0};
    vkm_set_config(&c);
    g_build_ms = 1;
    g_expect_task = 1;
    VkpStage vs = vstage(), task = tstage(), ps = pstage();
    CHECK(vkm_graphics(&vs, &task, &ps, &g_fmt, 0) == PIPE_G);
    CHECK(__atomic_load_n(&g_builds, __ATOMIC_RELAXED) == 1);
    vkm_close();
    printf("ok task_stage\n");
}

// ---------- drain_order
#define FAKE_CB ((VkCommandBuffer)(uintptr_t)0xC0FFEE00u)
static uint64_t g_cur_value;
static uint64_t fake_begin(int kind) {
    (void)kind;
    return g_cur_value = gpu_sync_begin();
}
static VkCommandBuffer fake_cmd(void) { return FAKE_CB; }
static uint64_t fake_end(const VkSemaphoreSubmitInfo *w, uint32_t n) {
    (void)w, (void)n;
    uint64_t v = g_cur_value;
    gpu_sync_end();
    return v;
}
static void VKAPI_CALL fake_cull(VkCommandBuffer cb, VkCullModeFlags v) {
    (void)cb, (void)v;
    sleep_ms(2);   // a recorder that is slow, so the packets are still in the ring when the miss comes
    __atomic_add_fetch(&g_executed, 1, __ATOMIC_RELEASE);
}

static uint64_t g_raise_to;
static void raise_later(void *arg) {
    (void)arg;
    sleep_ms(150);
    counter_raise(g_raise_to);
}

static void case_drain_order(void) {
    start_counter();
    VkRecCalls calls;
    memset(&calls, 0, sizeof calls);
    calls.SetCullMode = fake_cull;
    VkRecSubmit sub = {fake_begin, fake_cmd, fake_end, 0};
    VkRecOpen open = {.calls = &calls, .submit = &sub};
    vkrec_open(&open);
    uint64_t v1 = vkrec_begin(GPU_VK_REPLAY);   // a frame the device has not finished
    vkrec_end(NULL, 0);
    while (gpu_sync_issued() < v1) plat_yield();
    CHECK(gpu_sync_done() < v1);
    uint64_t v2 = vkrec_begin(GPU_VK_REPLAY);   // the frame being replayed, with packets in the ring
    enum { PACKETS = 20 };
    for (int i = 0; i < PACKETS; i++) vkrec_SetCullMode(VK_CULL_MODE_BACK_BIT);
    g_sent = PACKETS;
    g_raise_to = v1;
    CHECK(plat_thread_start(raise_later, NULL, 1 << 20));

    VkmOps o = fake_ops(0, 0);   // the device's drain, handed and done: the ring and the timeline
    vkm_set_ops(&o);
    g_build_ms = 5;
    VkpStage vs = vstage(), ps = pstage();
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &g_fmt, 1);
    CHECK(p == PIPE_G);
    printf("at the build: %llu of %llu packets made into calls, submissions issued %llu done %llu\n", (ull)g_at_executed,
           (ull)g_sent, (ull)g_at_issued, (ull)g_at_done);
    CHECK(g_at_executed == g_sent);
    CHECK(g_at_issued == v1 && g_at_done == g_at_issued);
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.graphics_misses == 1 && s.drains == 1 && s.drained_submissions == 1);
    CHECK(s.stall_ns[0] >= 100ull * 1000000ull);   // the replay waited for the device's frame
    vkrec_end(NULL, 0);
    counter_raise(v2);
    vkrec_close();
    gpu_sync_finish();
    vkm_close();
    gpu_sync_stop();
    printf("ok drain_order\n");
}

// ---------- the bound
static void case_bound_skip(void) {
    VkmOps o = fake_ops(1, 0);
    vkm_set_ops(&o);
    VkmConfig c = {100ull * 1000000ull, 0};
    vkm_set_config(&c);
    g_build_ms = 500;
    VkpStage vs = vstage(), ps = pstage();
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &g_fmt, 1);
    CHECK(p == VK_NULL_HANDLE);   // past the bound: a draw to skip
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.graphics_misses == 1 && s.skipped_draws == 1 && s.skipped_pending == 0 && s.would_skip == 0);
    CHECK(!g_g_built);
    CHECK(vkm_graphics(&vs, NULL, &ps, &g_fmt, 1) == VK_NULL_HANDLE);   // the build is still running: skipped at once
    vkm_stats(&s);
    CHECK(s.graphics_misses == 1 && s.skipped_draws == 2 && s.skipped_pending == 1);
    CHECK(__atomic_load_n(&g_builds, __ATOMIC_RELAXED) == 1);     // no second build of the same pipeline
    wait_built();
    CHECK(g_late_at_end == 1);   // the build was told that the replay stopped waiting
    CHECK(vkm_graphics(&vs, NULL, &ps, &g_fmt, 1) == PIPE_G);   // and now it serves the draw
    vkm_close();
    vkm_stats(&s);
    CHECK(s.late_builds == 1 && s.skipped_draws == 2);
    printf("skipped %llu (at once %llu), late builds %llu, stall %.0f ms\n", (ull)s.skipped_draws, (ull)s.skipped_pending,
           (ull)s.late_builds, (double)s.stall_ns[0] / 1e6);
    printf("ok bound_skip\n");
}

static void case_override(void) {
    VkmOps o = fake_ops(1, 0);
    vkm_set_ops(&o);
    VkmConfig c = {100ull * 1000000ull, 1};   // the bound the production run would have, and the override
    vkm_set_config(&c);
    g_build_ms = 300;
    VkpStage vs = vstage(), ps = pstage();
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &g_fmt, 1);
    CHECK(p == PIPE_G);   // waited
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.skipped_draws == 0 && s.would_skip == 1 && s.waits_kept == 0 && s.late_builds == 0);
    CHECK(s.stall_ns[0] >= 250ull * 1000000ull);
    vkm_close();
    printf("waited %.0f ms; the production bound would have skipped it (would_skip %llu)\nok override\n",
           (double)s.stall_ns[0] / 1e6, (ull)s.would_skip);
}

static void case_kept_wait(void) {
    VkmOps o = fake_ops(1, 0);
    vkm_set_ops(&o);
    VkmConfig c = {100ull * 1000000ull, 0};
    vkm_set_config(&c);
    g_build_ms = 300;
    VkpStage vs = vstage(), ps = pstage();
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &g_fmt, 0);   // the draw's target holds no earlier picture: it may not be skipped
    CHECK(p == PIPE_G);
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.skipped_draws == 0 && s.waits_kept == 1 && s.would_skip == 0);
    vkm_close();
    printf("ok kept_wait\n");
}

static void case_compute_wait(void) {
    VkmOps o = fake_ops(1, 0);
    vkm_set_ops(&o);
    VkmConfig c = {50ull * 1000000ull, 0};
    vkm_set_config(&c);
    g_build_ms = 300;
    VkpStage cs = {.key = g_cs.key, .shape = &g_cs};
    VkPipeline p = vkm_compute(&cs);
    CHECK(p == PIPE_C);
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.compute_misses == 1 && s.graphics_misses == 0 && s.skipped_draws == 0);
    CHECK(s.stall_ns[1] >= 250ull * 1000000ull && s.stall_max_ns[1] == s.stall_ns[1]);
    CHECK(vkm_compute(&cs) == PIPE_C && __atomic_load_n(&g_builds, __ATOMIC_RELAXED) == 1);   // then it is found
    vkm_close();
    printf("ok compute_wait\n");
}

// ---------- release
static void release_check(const char *key, int n) {
    CHECK(g_nreleased == n && n > 0);
    CHECK(!strcmp(g_released[n - 1], key));
}

static void case_release_order(void) {
    VkmOps o = fake_ops(1, 1);
    vkm_set_ops(&o);
    static const nvn_stage a = {.key = "aaaaaaaaaaaaaaaa.00000001", .kind = NVN_STAGE_VS};
    static const nvn_stage b = {.key = "bbbbbbbbbbbbbbbb.00000002", .kind = NVN_STAGE_PS};
    static const nvn_stage c = {.key = "cccccccccccccccc.00000003", .kind = NVN_STAGE_PS};
    const nvn_stage *p7[2] = {&a, &b}, *p8[2] = {&b, &c};
    g_handed = 5, g_done = 3;   // frames up to 5 are handed out, 3 are done
    vkm_program_bound(7, p7, 2);
    vkm_program_bound(8, p8, 2);
    vkm_program_bound(7, p7, 2);   // again: nothing
    vkm_program_release(99);       // a program the replay never saw
    vkm_program_release(7);        // a loses its only program; b is still held by 8
    vkm_retire();
    printf("after the release of program 7, timeline at 4 of 5: %d key(s) destroyed\n", g_nreleased);
    g_done = 4;
    vkm_retire();
    CHECK(g_nreleased == 0);   // the last frame that could use a is not done
    g_done = 5;
    vkm_retire();
    release_check(a.key, 1);   // done: a goes, b stays
    CHECK(g_nreleased == 1);

    g_handed = 6;
    vkm_program_release(8);    // b and c lose their last program
    g_done = 5;
    vkm_retire();
    CHECK(g_nreleased == 1);
    g_done = 6;
    vkm_retire();
    CHECK(g_nreleased == 3);
    CHECK((!strcmp(g_released[1], b.key) && !strcmp(g_released[2], c.key)) ||
          (!strcmp(g_released[1], c.key) && !strcmp(g_released[2], b.key)));

    // a program bound again before the release is retired takes the key back
    const nvn_stage *p9[1] = {&a};
    g_handed = 7;
    vkm_program_bound(9, p9, 1);
    vkm_program_release(9);
    vkm_program_bound(10, p9, 1);
    g_done = 100;
    vkm_retire();
    CHECK(g_nreleased == 3);
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.keys_released == 3 && s.keys_pending == 0 && s.programs_bound == 4 && s.programs_released == 3);

    // the same serial bound again after its release (a recording replayed after the game finalized the program) takes
    // the keys back: nothing is destroyed under the frame that binds it
    static const nvn_stage d = {.key = "dddddddddddddddd.00000004", .kind = NVN_STAGE_VS};
    const nvn_stage *p11[1] = {&d};
    g_handed = 8;
    vkm_program_bound(11, p11, 1);
    vkm_program_release(11);
    vkm_program_bound(11, p11, 1);   // that bind names a finalized serial: held for this frame only
    g_done = 8;
    vkm_retire();
    CHECK(g_nreleased == 4 && !strcmp(g_released[3], d.key));
    // a program finalized before its first bind: the bind holds the keys for the frame, and they go after it
    const nvn_stage *p12[1] = {&c};
    g_handed = 9;
    vkm_program_release(12);
    vkm_program_bound(12, p12, 1);
    g_done = 8;
    vkm_retire();
    CHECK(g_nreleased == 4);
    g_done = 9;
    vkm_retire();
    CHECK(g_nreleased == 5 && !strcmp(g_released[4], c.key));
    vkm_stats(&s);
    CHECK(s.keys_pending == 0);
    vkm_close();
    printf("ok release_order\n");
}

// A release that falls due while a graphics build is still running waits for it: the build holds the handles of the
// libraries it made.
static void case_release_during_build(void) {
    VkmOps o = fake_ops(1, 1);
    vkm_set_ops(&o);
    VkmConfig c = {50ull * 1000000ull, 0};
    vkm_set_config(&c);
    static const nvn_stage a = {.key = "aaaaaaaaaaaaaaaa.00000001", .kind = NVN_STAGE_VS};
    const nvn_stage *p1[1] = {&a};
    g_handed = 1, g_done = 1;
    vkm_program_bound(1, p1, 1);
    g_build_ms = 400;
    VkpStage vs = vstage(), ps = pstage();
    CHECK(vkm_graphics(&vs, NULL, &ps, &g_fmt, 1) == VK_NULL_HANDLE);   // skipped: the build goes on
    vkm_program_release(1);
    g_done = 100;
    vkm_retire();
    CHECK(g_nreleased == 0);   // a job is running
    wait_built();
    sleep_ms(50);
    vkm_retire();
    CHECK(g_nreleased == 1);   // once the build is done the key goes
    vkm_close();
    printf("ok release_during_build\n");
}

static void case_env(const char *expect) {
    VkmOps o = fake_ops(1, 0);
    vkm_set_ops(&o);
    vkm_set_config(NULL);   // read OOT2_VK_MISS_BOUND at the first miss
    g_build_ms = 300;
    VkpStage vs = vstage(), ps = pstage();
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &g_fmt, 1);
    printf("%s\n", p ? "waited" : "skipped");
    CHECK(!strcmp(expect, p ? "waited" : "skipped"));
    wait_built();
    vkm_close();
    printf("ok env\n");
}

// ---------- on a device
static void open_all(const char *pack) {
    gpu_vk_open();
    char why[512];
    if (!vkp_store_open(pack, NULL, why, sizeof why)) {
        fprintf(stderr, "the pack did not open: %s\n", why);
        exit(1);
    }
    vkp_open(NULL);
}

static VkSemaphore g_gate;
static void close_and_report(void) {
    vkm_close();
    vkp_close();
    vkp_store_close();
    if (g_gate) {   // the device is idle: every submission was finished by the callers
        GPU_VK_OK(vkDeviceWaitIdle(gpu_vk.device), "vkDeviceWaitIdle");
        vkDestroySemaphore(gpu_vk.device, g_gate, NULL);
        g_gate = VK_NULL_HANDLE;
    }
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
}

static uint64_t g_gate_value;
static void make_gate(void) {
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    CHECK(vkCreateSemaphore(gpu_vk.device, &si, NULL, &g_gate) == VK_SUCCESS);
    g_gate_value = 1;
}
static void open_gate(void) {
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = g_gate, .value = g_gate_value};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
}
static void open_gate_later(void *arg) {
    sleep_ms((uint64_t)(uintptr_t)arg);
    open_gate();
}
// A submission the device holds until the gate opens; `bind` is bound in its command buffer.
static uint64_t held_submission(VkPipeline bind, VkPipelineBindPoint bp) {
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_gate,
                                  .value = g_gate_value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    if (bind) vkCmdBindPipeline(cb, bp, bind);
    CHECK(gpu_vk_end(&wait, 1) == v);
    return v;
}

static void case_dev_drain(const char *pack) {
    open_all(pack);
    make_gate();
    VkmOps o = {0};
    vkm_set_ops(&o);   // the device's everything
    uint64_t held = held_submission(VK_NULL_HANDLE, VK_PIPELINE_BIND_POINT_GRAPHICS);
    CHECK(gpu_sync_issued() == held && gpu_sync_done() < held);
    // the frame being replayed: the gate is held, the frame has put nothing on the queue
    VkRecOpen ro = {0};
    vkrec_open(&ro);
    uint64_t frame = vkrec_begin(GPU_VK_REPLAY);
    CHECK(gpu_sync_open() == frame && gpu_sync_handed() == frame);
    CHECK(plat_thread_start(open_gate_later, (void *)(uintptr_t)200, 1 << 20));
    VkpStage vs = {.key = "5100000000000001.00000000"}, ps = {.key = "5100000000000002.00000000"};
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &(VkpFormats){1, {VK_FORMAT_R8G8B8A8_UNORM}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED}, 1);
    CHECK(p != VK_NULL_HANDLE);
    CHECK(gpu_sync_done() >= held);   // the build came after the device finished the held submission
    VkmStats s;
    vkm_stats(&s);
    printf("miss: drained %llu submission(s) in %.0f ms, stall %.0f ms, graphics misses %llu\n", (ull)s.drained_submissions,
           (double)s.drain_ns / 1e6, (double)s.stall_ns[0] / 1e6, (ull)s.graphics_misses);
    CHECK(s.drained_submissions == 1 && s.graphics_misses == 1 && s.skipped_draws == 0);
    CHECK(s.drain_ns >= 100ull * 1000000ull);
    CHECK(vkm_graphics(&vs, NULL, &ps, &(VkpFormats){1, {VK_FORMAT_R8G8B8A8_UNORM}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED}, 1) == p);
    vkrec_end(NULL, 0);
    vkrec_close();
    gpu_sync_finish();
    close_and_report();
    printf("ok dev_drain\n");
}

static void case_dev_release(const char *pack) {
    open_all(pack);
    make_gate();
    VkmOps o = {0};
    vkm_set_ops(&o);
    VkpStage cs = {.key = "5100000000000003.00000000"};
    VkPipeline p = vkm_compute(&cs);   // a miss: built after a drain
    CHECK(p);
    static const nvn_stage st = {.key = "5100000000000003.00000000", .kind = NVN_STAGE_CS};
    const nvn_stage *ids[1] = {&st};
    vkm_program_bound(1, ids, 1);
    // a command buffer that binds the pipeline, held by the device
    uint64_t held = held_submission(p, VK_PIPELINE_BIND_POINT_COMPUTE);
    vkm_program_release(1);   // the program is finalized while the submission is in flight
    VkpStats a, b;
    vkp_stats(&a);
    vkm_retire();
    vkp_stats(&b);
    printf("released with the submission in flight: resident %llu before, %llu after the retire\n", (ull)a.resident,
           (ull)b.resident);
    CHECK(b.resident == a.resident);   // not destroyed while the device holds the command buffer
    open_gate();
    gpu_sync_wait(held, UINT64_MAX);
    vkm_retire();
    vkp_stats(&b);
    printf("after the device finished it: resident %llu\n", (ull)b.resident);
    CHECK(b.resident == a.resident - 1);
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.keys_released == 1);
    close_and_report();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    CHECK(errors == 0);
    printf("ok dev_release\n");
}

// The real builder after a pause, so that the frame's work has reached the queue by the time the builder checks the queue
// (the check that the flag of a build the replay stopped waiting for lets through).
static VkPipeline slow_build(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *f,
                             const int *late) {
    (void)task;
    sleep_ms(150);
    return vkp_graphics_ex(vs, NULL, ps, f, late);
}

static void case_dev_late(const char *pack) {
    open_all(pack);
    make_gate();
    VkmOps o = {.build_graphics = slow_build};
    vkm_set_ops(&o);
    VkmConfig c = {1000000ull, 0};   // 1 ms: a real build is past it
    vkm_set_config(&c);
    VkRecOpen ro = {0};
    vkrec_open(&ro);
    uint64_t frame = vkrec_begin(GPU_VK_REPLAY);
    (void)frame;
    VkpStage vs = {.key = "5100000000000001.00000000"}, ps = {.key = "5100000000000002.00000000"};
    VkpFormats f = {1, {VK_FORMAT_R8G8B8A8_UNORM}, VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
    VkPipeline p = vkm_graphics(&vs, NULL, &ps, &f, 1);
    CHECK(p == VK_NULL_HANDLE);   // skipped
    // the frame goes on and its work reaches the queue, held by the device, while the build is still creating objects
    vkrec_end(NULL, 0);
    uint64_t held = held_submission(VK_NULL_HANDLE, VK_PIPELINE_BIND_POINT_GRAPHICS);   // the queue is busy from here
    CHECK(gpu_sync_issued() == held && gpu_sync_done() < held);
    CHECK(plat_thread_start(open_gate_later, (void *)(uintptr_t)500, 1 << 20));
    for (int i = 0; i < 1000 && !vkp_find_graphics(&vs, NULL, &ps, &f); i++) sleep_ms(10);
    CHECK(vkp_find_graphics(&vs, NULL, &ps, &f) != VK_NULL_HANDLE);
    VkmStats s;
    vkm_stats(&s);
    CHECK(s.skipped_draws == 1 && s.graphics_misses == 1);
    vkrec_close();
    gpu_sync_finish();
    close_and_report();
    vkm_stats(&s);
    printf("skipped %llu, late builds %llu\n", (ull)s.skipped_draws, (ull)s.late_builds);
    CHECK(s.late_builds == 1);
    printf("ok dev_late\n");
}

int main(int argc, char **argv) {
    const char *cs = argc > 1 ? argv[1] : "";
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--mutant")) continue;
        const char *m = argv[i + 1];
        if (!strcmp(m, "nodrain")) vkm_no_drain = 1;
        else if (!strcmp(m, "nocount")) vkm_no_count_skip = 1;
        else if (!strcmp(m, "release_now")) vkm_release_at_once = 1;
        else if (!strcmp(m, "nolate")) vkm_no_late = 1;
        else {
            fprintf(stderr, "unknown mutant %s\n", m);
            return 2;
        }
    }
    if (!strcmp(cs, "task_stage")) case_task_stage();
    else if (!strcmp(cs, "drain_order")) case_drain_order();
    else if (!strcmp(cs, "bound_skip")) case_bound_skip();
    else if (!strcmp(cs, "override")) case_override();
    else if (!strcmp(cs, "kept_wait")) case_kept_wait();
    else if (!strcmp(cs, "compute_wait")) case_compute_wait();
    else if (!strcmp(cs, "release_order")) case_release_order();
    else if (!strcmp(cs, "release_during_build")) case_release_during_build();
    else if (!strcmp(cs, "env") && argc >= 3) case_env(argv[2]);
    else if (!strcmp(cs, "dev_drain") && argc >= 3) case_dev_drain(argv[2]);
    else if (!strcmp(cs, "dev_release") && argc >= 3) case_dev_release(argv[2]);
    else if (!strcmp(cs, "dev_late") && argc >= 3) case_dev_late(argv[2]);
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
