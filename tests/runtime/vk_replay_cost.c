// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// MEASUREMENT, not a contract test: the cost of the Vulkan replay (runtime/gpu/vk_replay.c) per graphics draw, as a frame of
// N draws grows. Driven by tests/test_runtime_vk_replay_cost.py, which makes the shader pack, the variant map and the
// manifest this program reads and prints the table:
//   vk_replay_cost --base B --variants V --map M --keys K --pattern a|b|c --n 340,1000,2000,3000 --kind import|mirror
//
// What runs for real. vkr_submit and everything it calls on the replay thread: the walk, the ledger tracker and its
// answers digest (nvn_ledger.c), the device ops' stage descriptions (vkr_device_ops()->binds, dev_binds) over a shader
// store opened on synthetic packs with a variant map (vkp_store_open_map, shader_cache.c), the miss path's entry
// (vkm_graphics, vkp_find_graphics, vkp_graphics_key, the pipeline table's hash, lock and chain), the backing map
// (backing_resolve over pools of guest memory, with stand-ins for the device behind it as tests/runtime/backing_test.c has),
// the snapshots, the banks, the dynamic state and the recorder's packing (vkrec_*, vk_rec_pack.c, the ring) with a recorder
// thread whose Vulkan calls do nothing.
// What is faked: the device (no Vulkan call is made, the pipelines are made-up handles entered into the pipeline table at
// their first ask, vkp_bench_open), the scratch memory (ordinary cached RAM in two arenas, not the write-combined memory a
// device gives), the render target and bindless ops of vkr_device_ops, the frame's begin and end hooks, the completion
// (the stand-in device is done at once), and, with --kind mirror, the kind of the backing of a uniform range (a real
// import, relabelled a mirror, so that the replay binds it live as it does a mirror's).
// Timers: with -DOOT2_BENCH_TIMERS the replay's own sections are timed (oot2/bench_timers.h); without it the program times
// nothing but the frame.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <unistd.h>
#endif

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/bench_timers.h"
#include "oot2/cpu.h"
#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/ledger.h"
#include "oot2/loader.h"
#include "oot2/memory.h"
#include "oot2/nvn.h"
#include "oot2/nvn_obj.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/shader_cache.h"
#include "oot2/shader_desc.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"

#ifndef OOT2_BENCH_HOOKS
#error "this program needs -DOOT2_BENCH_HOOKS (vkp_bench_open in runtime/gpu/vk_pipeline.c)"
#endif
extern void vkp_bench_open(uint32_t filler);
extern uint64_t vkp_bench_primed;

typedef unsigned long long ull;
#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define NPAIR 64               // the (vertex stage, pixel stage) pairs a gameplay frame cycles through
#define PAIR_A 32              // the pair pattern a repeats: one whose stages map to the base module (the driver's variants are the first)
#define ST_VS_ 0               // the replay's stage numbers (vk_replay.c ST_VS, ST_FS)
#define ST_FS_ 1
#define DRAW_VERTICES 36
#define ARENA_BYTES (64u << 20)
#define TEX_IDS 4000
#define UBO_BYTES 256          // the range one draw binds
#define UBO_SLOTS_USED 4096    // the per-draw ranges are cut from a ring of this many

// ---------- the clock
static double g_ns_per_tsc;
static uint64_t g_start_ns;
static void calibrate_tsc(void) {
    uint64_t n0 = plat_now_ns(), c0 = __builtin_ia32_rdtsc();
    while (plat_now_ns() - n0 < 100000000ull) {
    }
    uint64_t n1 = plat_now_ns(), c1 = __builtin_ia32_rdtsc();
    g_ns_per_tsc = (double)(n1 - n0) / (double)(c1 - c0);
}

#ifdef OOT2_BENCH_TIMERS
uint64_t bench_acc[BT_COUNT], bench_n[BT_COUNT];
// Each timer's parent (the timer it runs inside), for taking the timers' own cost off the figures.
static const int bt_parent[BT_COUNT] = {
    [BT_WALK] = -1,           [BT_RECORD] = BT_WALK,           [BT_DRAW_PREPARE] = BT_RECORD,   [BT_S1_DIGEST] = BT_DRAW_PREPARE,
    [BT_S1_TEXT] = BT_S1_DIGEST, [BT_S1_SHA] = BT_S1_TEXT,     [BT_S2_BINDS] = BT_DRAW_PREPARE,
    [BT_S2_CONTENT] = BT_S2_BINDS, [BT_S3_GRAPHICS] = BT_DRAW_PREPARE, [BT_S3_KEY] = BT_S3_GRAPHICS,
    [BT_S3_KEY_DIGEST] = BT_S3_KEY, [BT_S3_KEY_FORMATS] = BT_S3_KEY, [BT_S3_KEY_PRINT] = BT_S3_KEY,
    [BT_S3_TABLE] = BT_S3_GRAPHICS, [BT_S4_VERTEX] = BT_DRAW_PREPARE, [BT_S4_RESOLVE] = BT_S4_VERTEX,
    [BT_S4_PACK] = BT_S4_VERTEX, [BT_S5_C0] = BT_DRAW_PREPARE,  [BT_S5_RESOURCES] = BT_DRAW_PREPARE,
    [BT_S5_BANK] = BT_S5_RESOURCES, [BT_S5_C2] = BT_S5_RESOURCES, [BT_S5_RESOLVE] = BT_S5_RESOURCES,
    [BT_S5_SNAPSHOT] = BT_S5_RESOURCES, [BT_S5_SNAP_SEARCH] = BT_S5_SNAPSHOT, [BT_S5_WRITTEN] = BT_S5_RESOURCES,
    [BT_S5_WROTE] = BT_S5_RESOURCES, [BT_S6_SUPPORT] = BT_DRAW_PREPARE, [BT_S6_PUSHDESC] = BT_DRAW_PREPARE,
    [BT_S6_PUSHCONST] = BT_DRAW_PREPARE, [BT_U_DYN] = BT_DRAW_PREPARE, [BT_U_STAMPS] = BT_DRAW_PREPARE,
    [BT_U_PROGRAM] = BT_RECORD, [BT_U_LEDGER_OP] = BT_RECORD,  [BT_U_LEDGER_ROW] = BT_U_LEDGER_OP,
    [BT_DUMMY] = -1,
};
static const char *const bt_name[BT_COUNT] = {
    [BT_WALK] = "walk (record() of every record)", [BT_RECORD] = "record_op (all records)",
    [BT_DRAW_PREPARE] = "draw_prepare",
    [BT_S1_DIGEST] = "S1 ledger_track_answers_digest", [BT_S1_TEXT] = "S1 digest recomputed (answers_text)",
    [BT_S1_SHA] = "S1 rt_sha256", [BT_S2_BINDS] = "S2 binds_of -> dev_binds",
    [BT_S2_CONTENT] = "S2 vkp_stage_content_digest (store lookup)", [BT_S3_GRAPHICS] = "S3 ops->graphics (vkm_graphics)",
    [BT_S3_KEY] = "S3 vkp_graphics_key", [BT_S3_KEY_DIGEST] = "S3 cache_digest_text calls",
    [BT_S3_KEY_FORMATS] = "S3 formats_text", [BT_S3_KEY_PRINT] = "S3 key snprintf",
    [BT_S3_TABLE] = "S3 pipeline table (hash, lock, chain)", [BT_S4_VERTEX] = "S4 vertex_input",
    [BT_S4_RESOLVE] = "S4 resolve per stream", [BT_S4_PACK] = "S4 vkrec_BindVertexBuffers2 + SetVertexInputEXT",
    [BT_S5_C0] = "S5 vkn_c0_graphics (x2)", [BT_S5_RESOURCES] = "S5 stage_resources (x2)",
    [BT_S5_BANK] = "S5 bank() compare and copy", [BT_S5_C2] = "S5 vkn_c2_fill",
    [BT_S5_RESOLVE] = "S5 ops->resolve of a uniform", [BT_S5_SNAPSHOT] = "S5 snapshot()",
    [BT_S5_SNAP_SEARCH] = "S5 snapshot() linear search", [BT_S5_WRITTEN] = "S5 vkr_written",
    [BT_S5_WROTE] = "S5 vkr_wrote", [BT_S6_SUPPORT] = "S6 support() scratch",
    [BT_S6_PUSHDESC] = "S6 vkrec_PushDescriptorSetKHR", [BT_S6_PUSHCONST] = "S6 vkrec_PushConstants",
    [BT_U_DYN] = "U dyn_emit", [BT_U_STAMPS] = "U target stamps", [BT_U_PROGRAM] = "U bind_program",
    [BT_U_LEDGER_OP] = "U ledger_track_op", [BT_U_LEDGER_ROW] = "U ledger rows (add_row)", [BT_DUMMY] = "dummy",
};
#endif

// ---------- the stand-in submissions, over a counter the recorder raises (as vk_replay_test.c has them)
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
#define FAKE_CB ((VkCommandBuffer)(uintptr_t)0xC0FFEE00u)
static uint64_t g_cur_value;
static uint64_t fake_begin(int kind) {
    CHECK(kind == GPU_VK_REPLAY);
    g_cur_value = gpu_sync_begin();
    return g_cur_value;
}
static VkCommandBuffer fake_cmd(void) { return FAKE_CB; }
static uint64_t fake_end(const VkSemaphoreSubmitInfo *waits, uint32_t n) {
    CHECK(n == 0 && !waits);
    uint64_t v = g_cur_value;
    gpu_sync_end();
    counter_raise(v);   // the stand-in device is done at once
    return v;
}

// ---------- the recorder's calls: every one does nothing (the replay thread is what is measured)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#define NOP_F(name, params, ...) static void VKAPI_CALL nop_##name(VkCommandBuffer cb, VKREC_UNPAREN params) {}
#define NOP_D(name, kind, params, ...) NOP_F(name, params)
VKREC_FIXED(NOP_F)
VKREC_DRAWS(NOP_D)
static void VKAPI_CALL nop_BindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first,
                                              uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {}
static void VKAPI_CALL nop_PushDescriptorSetKHR(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t set,
                                                uint32_t n, const VkWriteDescriptorSet *w) {}
static void VKAPI_CALL nop_PushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off,
                                         uint32_t size, const void *v) {}
static void VKAPI_CALL nop_BindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *buf,
                                              const VkDeviceSize *off, const VkDeviceSize *sz, const VkDeviceSize *st) {}
static void VKAPI_CALL nop_SetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport *v) {}
static void VKAPI_CALL nop_SetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D *r) {}
static void VKAPI_CALL nop_SetVertexInputEXT(VkCommandBuffer cb, uint32_t nb, const VkVertexInputBindingDescription2EXT *b,
                                             uint32_t na, const VkVertexInputAttributeDescription2EXT *a) {}
static void VKAPI_CALL nop_SetColorBlendEnableEXT(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBool32 *v) {}
static void VKAPI_CALL nop_SetColorBlendEquationEXT(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                    const VkColorBlendEquationEXT *e) {}
static void VKAPI_CALL nop_SetColorWriteMaskEXT(VkCommandBuffer cb, uint32_t first, uint32_t n,
                                                const VkColorComponentFlags *m) {}
static void VKAPI_CALL nop_SetBlendConstants(VkCommandBuffer cb, const float c[4]) {}
static void VKAPI_CALL nop_PipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *d) {}
static void VKAPI_CALL nop_BeginRendering(VkCommandBuffer cb, const VkRenderingInfo *r) {}
static void VKAPI_CALL nop_EndRendering(VkCommandBuffer cb) {}
static void VKAPI_CALL nop_ClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a, uint32_t nr,
                                            const VkClearRect *r) {}
static void VKAPI_CALL nop_CopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) {}
static void VKAPI_CALL nop_CopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout l, uint32_t n,
                                             const VkBufferImageCopy *r) {}
static void VKAPI_CALL nop_CopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout l, VkBuffer d, uint32_t n,
                                             const VkBufferImageCopy *r) {}
static void VKAPI_CALL nop_CopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                     const VkImageCopy *r) {}
static void VKAPI_CALL nop_BlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n,
                                     const VkImageBlit *r, VkFilter f) {}
static void VKAPI_CALL nop_ClearColorImage(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue *c,
                                           uint32_t n, const VkImageSubresourceRange *r) {}
static void VKAPI_CALL nop_ClearDepthStencilImage(VkCommandBuffer cb, VkImage i, VkImageLayout l,
                                                  const VkClearDepthStencilValue *v, uint32_t n,
                                                  const VkImageSubresourceRange *r) {}
static void VKAPI_CALL nop_SetCheckpointNV(VkCommandBuffer cb, const void *m) {}
#pragma clang diagnostic pop

static VkRecCalls g_calls;
static VkRecSubmit g_submit;
static void open_recorder(void) {
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
    memset(&g_calls, 0, sizeof g_calls);
#define SET_F(name, ...) g_calls.name = nop_##name;
#define SET_D(name, kind, ...) g_calls.name = nop_##name;
#define SET_C(name) g_calls.name = nop_##name;
    VKREC_FIXED(SET_F)
    VKREC_DRAWS(SET_D)
    VKREC_CUSTOM(SET_C)
#undef SET_F
#undef SET_D
#undef SET_C
    g_calls.SetCheckpointNV = nop_SetCheckpointNV;
    // the breadcrumbs on, as a device with VK_NV_device_diagnostic_checkpoints has them (the default of a run)
    g_submit = (VkRecSubmit){fake_begin, fake_cmd, fake_end, 1};
    VkRecOpen o = {0, 2, &g_calls, &g_submit};   // the default ring (16 MiB), two frames in flight
    vkrec_open(&o);
}

// ---------- guest memory and the backing map's stand-in device (tests/runtime/backing_test.c)
enum { TAG_BLOCK = 1, TAG_IMPORT = 2 };
static int range_committed(uint64_t a, uint64_t n) {
#ifdef _WIN32
    while (n) {
        MEMORY_BASIC_INFORMATION m;
        if (VirtualQuery((void *)(uintptr_t)a, &m, sizeof m) != sizeof m || m.State != MEM_COMMIT) return 0;
        uint64_t run = (uint64_t)(uintptr_t)m.BaseAddress + m.RegionSize - a;
        if (run >= n) return 1;
        a += run, n -= run;
    }
    return 1;
#else
    static int fd[2] = {-1, -1};
    if (fd[0] < 0) CHECK(pipe(fd) == 0);
    for (uint64_t p = a; p < a + n; p += BACKING_HOST_PAGE) {
        char c;
        if (write(fd[1], (const void *)(uintptr_t)p, 1) != 1) return 0;
        CHECK(read(fd[0], &c, 1) == 1);
    }
    return 1;
#endif
}
static int dev_block_new(uint64_t size, BackingMem *out) {
    void *p = calloc(1, (size_t)size);
    if (!p) return 0;
    *out = (BackingMem){.buffer = (uintptr_t)p, .memory = TAG_BLOCK, .address = (uintptr_t)p, .size = size};
    return 1;
}
static int dev_import(uint64_t host, uint64_t size, BackingMem *out) {
    if ((host % BACKING_HOST_PAGE) || !size || (size % BACKING_HOST_PAGE) || !range_committed(host, size)) return 0;
    *out = (BackingMem){.buffer = host, .memory = TAG_IMPORT, .address = host, .size = size};
    return 1;
}
static void dev_free(const BackingMem *m) {
    if (m->memory == TAG_BLOCK) free((void *)(uintptr_t)m->buffer);
}
static const BackingOps k_backing_ops = {.block_new = dev_block_new, .import = dev_import, .free = dev_free};

static Cpu g_cpu;
static uint64_t g_heap;
static uint64_t svc(uint32_t imm, uint64_t x0, uint64_t x1, uint64_t x2) {
    g_cpu.x[0] = x0, g_cpu.x[1] = x1, g_cpu.x[2] = x2;
    rt_svc(&g_cpu, imm, 0);
    return g_cpu.x[0];
}
static void touch(uint64_t a, uint64_t n) {   // commit by first touch, as the game's stores would
    for (uint64_t p = a; p < a + n; p += BACKING_HOST_PAGE) *(volatile uint8_t *)(uintptr_t)p = 1;
}

// The pools the draws bind from, all CPU-visible: uniforms, program code areas (a stage's own constants, c[1]), vertex data.
#define POOL_U_AT (1 * MiB)
#define POOL_U_SIZE (8 * MiB)
#define POOL_C_AT (10 * MiB)
#define POOL_C_SIZE (8 * MiB)
#define POOL_V_AT (20 * MiB)
#define POOL_V_SIZE (8 * MiB)
static pool_t g_pool[3];
static uint64_t POOL_U, POOL_C, POOL_V;
static void make_pool(pool_t *p, uint64_t mem, uint64_t size) {
    *p = (pool_t){.magic = MAGIC('P', 'O', 'O', 'L'), .flags = 0, .mem = (uint8_t *)(uintptr_t)mem, .size = size,
                  .game_mem = (uint8_t *)(uintptr_t)mem};
    CHECK(backing_pool_init(p));
}
static void setup_memory(void) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 256 * MiB, 0) == 0);
    g_heap = g_cpu.x[1];
    POOL_U = g_heap + POOL_U_AT, POOL_C = g_heap + POOL_C_AT, POOL_V = g_heap + POOL_V_AT;
}
static void setup_backing(void) {
    backing_open(&k_backing_ops);
    make_pool(&g_pool[0], POOL_U, POOL_U_SIZE);
    make_pool(&g_pool[1], POOL_C, POOL_C_SIZE);
    make_pool(&g_pool[2], POOL_V, POOL_V_SIZE);
    touch(POOL_U, POOL_U_SIZE);
    touch(POOL_V, POOL_V_SIZE);
    touch(POOL_C, POOL_C_SIZE);   // the code areas lie in one committed run, as a game's code pool does
    // some bytes in the uniform pool that are not all alike
    for (uint64_t o = 0; o < POOL_U_SIZE; o += 4) *(volatile uint32_t *)(uintptr_t)(POOL_U + o) = (uint32_t)(o * 2654435761u);
}

// ---------- the stand-in ops (the others are the device's: vkr_device_ops)
static int g_mirror;   // 1: a uniform range's backing is relabelled a mirror (live binds, no snapshot)
static void bench_resolve(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    backing_resolve(addr, size, value, out);
    if (g_mirror && out->kind == BACKING_IMPORT) out->kind = BACKING_MIRROR;
}

#define BUF_SCRATCH 0x5Cu
static uint8_t *g_arena[2];
static uint32_t g_arena_used;
static uint64_t g_arena_value;
static int g_arena_cur;
static void *b_scratch(uint64_t value, uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset) {
    if (value != g_arena_value) g_arena_value = value, g_arena_cur ^= 1, g_arena_used = 0;
    uint32_t at = (g_arena_used + align - 1) & ~(align - 1);
    CHECK((uint64_t)at + bytes <= ARENA_BYTES);
    g_arena_used = at + bytes;
    *buffer = (VkBuffer)(uintptr_t)BUF_SCRATCH, *offset = at;
    return g_arena[g_arena_cur] + at;
}
static int b_target(const nvn_tex_rec *t, int storage, uint64_t value, VkrTarget *out) {
    (void)value;
    if (storage) return 0;
    *out = (VkrTarget){(VkImageView)(uintptr_t)(0x7100 + (uint32_t)t->tex.d.format), VK_FORMAT_R8G8B8A8_UNORM,
                       t->tex.d.width, t->tex.d.height, 1, VK_IMAGE_ASPECT_COLOR_BIT, 0x7100 + (uint32_t)t->tex.d.format};
    return 1;
}
static VkDescriptorSet b_acquire(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers) {
    (void)value, (void)textures, (void)samplers;
    return (VkDescriptorSet)(uintptr_t)0x5E7;
}
static uint32_t b_image_slot(uint32_t id, int type) { return 1000 + id * 8 + (uint32_t)type; }
static void b_frame_begin(void) {
    backing_sync();
    vkm_retire();
}
static void b_frame_end(uint64_t value) { backing_submit_end(value); }

static VkReplayOps g_ops;
static void make_ops(void) {
    g_ops = *vkr_device_ops();   // graphics = vkm_graphics, program_bound, binds = dev_binds, gpu_wrote, cpu_reads: the device's
    g_ops.graphics_layout = (VkPipelineLayout)(uintptr_t)0x6100;
    g_ops.compute_layout = (VkPipelineLayout)(uintptr_t)0x6200;
    g_ops.resolve = bench_resolve;   // backing_resolve, the real one
    g_ops.target = b_target, g_ops.image = NULL, g_ops.acquire = b_acquire, g_ops.image_slot = b_image_slot;
    g_ops.storage_image = NULL, g_ops.image_by_id = NULL, g_ops.scratch = b_scratch, g_ops.buffer_address = NULL;
    g_ops.task_set = NULL, g_ops.raw = NULL, g_ops.queries = NULL, g_ops.query_results = NULL;
    g_ops.frame_begin = b_frame_begin, g_ops.frame_end = b_frame_end;
    vkr_set_ops(&g_ops);
}

// ---------- the stages: 64 vertex and 64 pixel stages, their keys from the manifest the driver wrote with the packs
static char g_keys[2][NPAIR][32];
static nvn_stage g_vs[NPAIR], g_ps[NPAIR];
static p_bind_program g_prog[NPAIR];

static void load_manifest(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open the manifest %s\n", path);
        exit(2);
    }
    char kind[8], key[64];
    int i, n = 0;
    while (fscanf(f, "%7s %d %63s", kind, &i, key) == 3) {
        CHECK(i >= 0 && i < NPAIR && strlen(key) < sizeof g_keys[0][0]);
        snprintf(g_keys[kind[0] == 'p'][i], sizeof g_keys[0][0], "%s", key);
        n++;
    }
    fclose(f);
    CHECK(n == 2 * NPAIR);
}
static void stage_init(nvn_stage *s, const char *key, int kind, uint32_t c1_at, uint32_t c1_end) {
    memset(s, 0, sizeof *s);
    snprintf(s->key, sizeof s->key, "%s", key);
    s->kind = (uint8_t)kind;
    s->ctl_size = NVN_STAGE_CTL_MAX;
    memcpy(s->ctl + 0x40, &c1_at, 4);
    memcpy(s->ctl + 0x44, &c1_end, 4);
}
static void make_stages(void) {
    for (int p = 0; p < NPAIR; p++) {
        stage_init(&g_vs[p], g_keys[0][p], NVN_STAGE_VS, 0x100, 0x1C0);   // each stage's own constants: 192 bytes of its code area
        stage_init(&g_ps[p], g_keys[1][p], NVN_STAGE_PS, 0x40, 0x100);
        memset(&g_prog[p], 0, sizeof g_prog[p]);
        g_prog[p].serial = (uint64_t)p + 1;
        g_prog[p].stages = 1 << 0 | 1 << 1;
        g_prog[p].nstage = 2;
        g_prog[p].code[0] = POOL_C + (uint64_t)(2 * p) * 0x10000;
        g_prog[p].id[0] = &g_vs[p];
        g_prog[p].code[1] = POOL_C + (uint64_t)(2 * p + 1) * 0x10000;
        g_prog[p].id[1] = &g_ps[p];
    }
}
// The texture pool's descriptors, as the game registered them: ledger_desc_register (the ledger answers the type of a bound
// texture from them, nvn_ledger.c bind_tex).
#define TEXPOOL_ADDR 0x1000000ull
static void register_textures(void) {
    dpool_ref pool = {.addr = TEXPOOL_ADDR, .count = 4096};
    tex_t t;
    memset(&t, 0, sizeof t);
    t.d = (nvn_texdesc){.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8, .width = 256, .height = 256, .depth = 1,
                        .levels = 1};
    for (int id = 1; id <= TEX_IDS; id++) ledger_desc_register(&pool, id, &t, NULL);
}

// ---------- recordings
typedef struct {
    uint8_t *b;
    size_t n, cap;
} Rec;
static void put(Rec *r, uint32_t op, const void *payload, size_t bytes) {
    size_t size = (sizeof(op_hdr) + bytes + 7) & ~(size_t)7;
    if (r->n + size > r->cap) {
        r->cap = (r->n + size) * 2;
        r->b = realloc(r->b, r->cap);
        CHECK(r->b);
    }
    op_hdr h = {op, (uint32_t)size};
    memset(r->b + r->n, 0, size);
    memcpy(r->b + r->n, &h, sizeof h);
    if (bytes) memcpy(r->b + r->n + sizeof h, payload, bytes);
    r->n += size;
}
#define PUT(r, op, ...)                                                                                                \
    ({                                                                                                                 \
        __typeof__(__VA_ARGS__) p_ = __VA_ARGS__;                                                                      \
        put((r), (op), &p_, sizeof p_);                                                                                \
    })

static uint64_t ubo_addr(int slot) { return POOL_U + (uint64_t)(slot % UBO_SLOTS_USED) * UBO_BYTES; }
#define VS_FIXED_UBO (POOL_U + 0x700000)
#define PS_FIXED_UBO (POOL_U + 0x701000)
static uint64_t vbuf_addr(int stream, int slot) { return POOL_V + (uint64_t)stream * MiB + (uint64_t)(slot % 256) * 0x1000; }
// pattern d: the range of a storage buffer the pixel stage writes, a different one each draw
static uint64_t ssbo_addr(int slot) { return POOL_V + 4 * MiB + (uint64_t)(slot % 8192) * 512; }

static p_bind_vattr vattrs(void) {   // position, normal, texture coordinates and colour: two streams
    static const struct {
        int fmt;
        uint32_t offset, stream;
    } a[4] = {{NVN_FORMAT_RGB32F, 0, 0}, {NVN_FORMAT_RGB32F, 0, 1}, {NVN_FORMAT_RG32F, 12, 1}, {NVN_FORMAT_RGBA8, 20, 1}};
    p_bind_vattr v;
    memset(&v, 0, sizeof v);
    v.first = 0, v.n = 4;
    for (int i = 0; i < 4; i++) {
        va_st s = {.fmt = (uint32_t)a[i].fmt, .offset = a[i].offset, .stream = a[i].stream};
        memcpy(&v.st[i], &s, sizeof s);
    }
    return v;
}
static p_bind_vstream vstreams(void) {
    p_bind_vstream v;
    memset(&v, 0, sizeof v);
    v.first = 0, v.n = 2;
    v.st[0] = (vs_st){12, 0};
    v.st[1] = (vs_st){24, 0};
    return v;
}
static p_bind_vbuf vbufs(int slot) {
    p_bind_vbuf v;
    memset(&v, 0, sizeof v);
    v.first = 0, v.n = 2;
    v.r[0] = (NVNbufferRange){vbuf_addr(0, slot), 0x1000};
    v.r[1] = (NVNbufferRange){vbuf_addr(1, slot), 0x1000};
    return v;
}
static p_bind_handles texture_handle(int draw) {
    p_bind_handles h;
    memset(&h, 0, sizeof h);
    h.stage = ST_FS_, h.first = 0, h.n = 1;
    h.h[0] = nvn_handle_pack((uint32_t)(1 + (draw * 7) % TEX_IDS), 3);
    return h;
}

// One frame of n draws. a: the state is bound once and the draws repeat. b: before each draw, the program of the next pair
// (cycling through NPAIR), one texture and one uniform range of the vertex stage. c: b, and the vertex attributes, streams and
// buffers (a different mesh each draw). d: b, and a storage buffer the pixel stage writes, a different range each draw (the
// pack's pixel stages declare it: the driver's "ssbo" packs).
static Rec build_frame(char pattern, int n) {
    Rec r = {0};
    put(&r, OP_BEGIN, NULL, 0);
    PUT(&r, OP_SET_TEX_POOL, (p_set_pool){.pool = {.addr = TEXPOOL_ADDR, .count = 4096}});
    p_set_rt rt;
    memset(&rt, 0, sizeof rt);
    rt.n = 1;
    rt.color[0].present = 1;
    rt.color[0].tex.d = (nvn_texdesc){.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8, .width = 1280, .height = 720,
                                      .depth = 1, .levels = 1};
    put(&r, OP_SET_RT, &rt, sizeof rt);
    PUT(&r, OP_VIEWPORTS, (p_viewports){.first = 0, .n = 1, .v = {{0, 0, 1280, 720}}});
    PUT(&r, OP_SCISSOR, (p_scissor){.first = 0, .n = 1, .r = {{0, 0, 1280, 720}}});
    PUT(&r, OP_BIND_PROGRAM, g_prog[PAIR_A]);
    PUT(&r, OP_BIND_VATTR, vattrs());
    PUT(&r, OP_BIND_VSTREAM, vstreams());
    PUT(&r, OP_BIND_VBUF, vbufs(0));
    PUT(&r, OP_BIND_UBO, (p_bind_buf){.stage = ST_VS_, .index = 0, .addr = ubo_addr(0), .size = UBO_BYTES});
    PUT(&r, OP_BIND_UBO, (p_bind_buf){.stage = ST_VS_, .index = 1, .addr = VS_FIXED_UBO, .size = 1024});
    PUT(&r, OP_BIND_UBO, (p_bind_buf){.stage = ST_FS_, .index = 0, .addr = PS_FIXED_UBO, .size = 512});
    PUT(&r, OP_BIND_TEX, texture_handle(0));
    for (int i = 0; i < n; i++) {
        if (pattern != 'a') {
            PUT(&r, OP_BIND_PROGRAM, g_prog[i % NPAIR]);
            PUT(&r, OP_BIND_TEX, texture_handle(i));
            PUT(&r, OP_BIND_UBO, (p_bind_buf){.stage = ST_VS_, .index = 0, .addr = ubo_addr(i), .size = UBO_BYTES});
        }
        if (pattern == 'c') {
            PUT(&r, OP_BIND_VATTR, vattrs());
            PUT(&r, OP_BIND_VSTREAM, vstreams());
            PUT(&r, OP_BIND_VBUF, vbufs(i));
        }
        if (pattern == 'd')
            PUT(&r, OP_BIND_SSBO, (p_bind_buf){.stage = ST_FS_, .index = 0, .addr = ssbo_addr(i), .size = 256});
        PUT(&r, OP_DRAW, (p_draw){.prim = NVN_DRAW_PRIMITIVE_TRIANGLES, .first = 0, .count = DRAW_VERTICES, .base_instance = 0,
                                  .instances = 1});
    }
    put(&r, OP_END, NULL, 0);
    return r;
}

// ---------- measuring
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static double median(const uint64_t *v, int n) {
    uint64_t *c = malloc((size_t)n * sizeof *c);
    CHECK(c);
    memcpy(c, v, (size_t)n * sizeof *c);
    qsort(c, (size_t)n, sizeof *c, cmp_u64);
    double m = n & 1 ? (double)c[n / 2] : ((double)c[n / 2 - 1] + (double)c[n / 2]) / 2.0;
    free(c);
    return m;
}
static uint8_t *g_evict;
static size_t g_evict_bytes;
static void evict_caches(void) {
    for (size_t i = 0; i < g_evict_bytes; i += 64) g_evict[i]++;
}

#ifdef OOT2_BENCH_TIMERS
static double g_self_cyc, g_child_cyc, g_pair_cyc;   // one empty timer's interval; what a timer nested in another adds to it; what a pair costs the program
static void calibrate_timers(void) {
    enum { REPS = 4000000 };
    memset(bench_acc, 0, sizeof bench_acc), memset(bench_n, 0, sizeof bench_n);
    uint64_t t0 = __builtin_ia32_rdtsc();
    for (int i = 0; i < REPS; i++) BT_TIME(BT_DUMMY, (void)0);
    g_pair_cyc = (double)(__builtin_ia32_rdtsc() - t0) / REPS;
    g_self_cyc = (double)bench_acc[BT_DUMMY] / REPS;
    memset(bench_acc, 0, sizeof bench_acc), memset(bench_n, 0, sizeof bench_n);
    for (int i = 0; i < REPS; i++) BT_TIME(BT_DUMMY, BT_TIME(BT_RECORD, (void)0));
    g_child_cyc = (double)bench_acc[BT_DUMMY] / REPS - g_self_cyc;
    if (g_child_cyc < 0) g_child_cyc = 0;
    printf("TIMER self_ns=%.2f child_ns=%.2f pair_ns=%.2f ns_per_tsc=%.5f\n", g_self_cyc * g_ns_per_tsc, g_child_cyc * g_ns_per_tsc,
           g_pair_cyc * g_ns_per_tsc, g_ns_per_tsc);
    memset(bench_acc, 0, sizeof bench_acc), memset(bench_n, 0, sizeof bench_n);
}
static int is_below(int d, int anc) {   // d runs inside anc
    for (int p = bt_parent[d]; p >= 0; p = bt_parent[p])
        if (p == anc) return 1;
    return 0;
}
static void print_split(char pattern, int n, int kind, int frames) {
    double draws = (double)n * frames, pairs = 0;
    for (int i = 0; i < BT_COUNT; i++) pairs += i == BT_DUMMY ? 0.0 : (double)bench_n[i];
    printf("SPLITOVERHEAD pattern=%c n=%d kind=%s timer_pairs_draw=%.2f timer_overhead_ns_draw=%.1f\n", pattern, n,
           kind ? "mirror" : "import", pairs / draws, pairs * g_pair_cyc * g_ns_per_tsc / draws);
    for (int i = 0; i < BT_COUNT; i++) {
        if (i == BT_DUMMY || !bench_n[i]) continue;
        double cyc = (double)bench_acc[i] - g_self_cyc * (double)bench_n[i];
        for (int d = 0; d < BT_COUNT; d++)
            if (d != BT_DUMMY && is_below(d, i)) cyc -= g_child_cyc * (double)bench_n[d];
        printf("SPLIT pattern=%c n=%d kind=%s id=%d name=\"%s\" incl_ns_draw=%.2f calls_draw=%.3f raw_ns_draw=%.2f\n", pattern, n,
               kind ? "mirror" : "import", i, bt_name[i], cyc * g_ns_per_tsc / draws, (double)bench_n[i] / draws,
               (double)bench_acc[i] * g_ns_per_tsc / draws);
    }
}
#endif

typedef struct {
    const char *base, *variants, *map, *keys;
    char pattern;
    int kind_mirror, frames, warmup, evict_mb, ns[32], nns;
    uint32_t filler;
} Cfg;

static void run_config(const Cfg *cfg, int n) {
    Rec rec = build_frame(cfg->pattern, n);
    uint64_t handle = (uint64_t)(uintptr_t)rec.b;
    g_mirror = cfg->kind_mirror;
    vkr_reset();
    vkm_stats_reset();
    int total = cfg->warmup + cfg->frames;
    uint64_t *wall = calloc((size_t)total, sizeof *wall), *cpu = calloc((size_t)total, sizeof *cpu);
    CHECK(wall && cpu);
    VkrStats before = {0};
    uint64_t primed0 = vkp_bench_primed;
    for (int f = 0; f < total; f++) {
        if (cfg->evict_mb) evict_caches();
        if (f == cfg->warmup) {
            vkr_stats(&before);
#ifdef OOT2_BENCH_TIMERS
            memset(bench_acc, 0, sizeof bench_acc), memset(bench_n, 0, sizeof bench_n);
#endif
        }
        uint64_t c0 = plat_thread_cpu_ns(NULL), t0 = plat_now_ns();
        vkr_submit(&handle, 1);
        uint64_t t1 = plat_now_ns(), c1 = plat_thread_cpu_ns(NULL);
        wall[f] = t1 - t0, cpu[f] = c1 - c0;
    }
    vkrec_idle();
    VkrStats after;
    vkr_stats(&after);
    VkmStats ms;
    vkm_stats(&ms);
    VkpStats pst;
    vkp_stats(&pst);
    if (pst.map_missing || pst.map_unknown_rows) {   // the digests the driver computed are not the tracker's: no lookup was the real one
        fprintf(stderr, "the variant map does not name the ledger digests of the frames (%llu without a map, %llu rows unknown)\n",
                (ull)pst.map_missing, (ull)pst.map_unknown_rows);
        exit(3);
    }
    uint32_t nf;
    const VkRecFrame *fr = vkrec_frames(&nf);
    CHECK(nf >= (uint32_t)total);
    fr += nf - (uint32_t)total;
    uint64_t *rwall = calloc((size_t)cfg->frames, sizeof *rwall), *rcpu = calloc((size_t)cfg->frames, sizeof *rcpu);
    uint64_t *ring = calloc((size_t)cfg->frames, sizeof *ring), *slot = calloc((size_t)cfg->frames, sizeof *slot);
    CHECK(rwall && rcpu && ring && slot);
    for (int f = 0; f < cfg->frames; f++) {
        rwall[f] = fr[cfg->warmup + f].replay_ns, rcpu[f] = fr[cfg->warmup + f].replay_cpu_ns;
        ring[f] = fr[cfg->warmup + f].ring_wait_ns, slot[f] = fr[cfg->warmup + f].slot_wait_ns;
    }
    uint64_t d = after.draws - before.draws;
    if (d != (uint64_t)n * cfg->frames || after.skipped != before.skipped || after.lost != 0 || ms.graphics_misses != 0 ||
        ms.skipped_draws != 0) {
        fprintf(stderr, "the replay is not the one measured: draws %llu (want %llu), skipped %llu, lost %llu, pipeline misses "
                "%llu\n", (ull)d, (ull)n * (ull)cfg->frames, (ull)(after.skipped - before.skipped), (ull)after.lost,
                (ull)ms.graphics_misses);
        exit(3);
    }
    const uint64_t *w = wall + cfg->warmup, *c = cpu + cfg->warmup;
    uint64_t lo = w[0], hi = w[0];
    for (int f = 1; f < cfg->frames; f++) lo = w[f] < lo ? w[f] : lo, hi = w[f] > hi ? w[f] : hi;
    double nd = (double)n;
    double fd = (double)cfg->frames;
    printf("BENCH pattern=%c n=%d kind=%s evict_mb=%d frames=%d wall_ns_draw=%.1f wall_min_ns_draw=%.1f wall_max_ns_draw=%.1f "
           "cpu_ns_draw=%.1f replay_ns_draw=%.1f replay_cpu_ns_draw=%.1f frame_us=%.1f ring_wait_us=%.1f slot_wait_us=%.1f "
           "snapshots_draw=%.3f live_uniforms_draw=%.3f scratch_bytes_draw=%.0f snapshot_drops_draw=%.3f primed=%llu draws=%llu skipped=%llu lost=%llu misses=%llu\n",
           cfg->pattern, n, cfg->kind_mirror ? "mirror" : "import", cfg->evict_mb, cfg->frames, median(w, cfg->frames) / nd,
           (double)lo / nd, (double)hi / nd, median(c, cfg->frames) / nd, median(rwall, cfg->frames) / nd,
           median(rcpu, cfg->frames) / nd, median(w, cfg->frames) / 1000.0, median(ring, cfg->frames) / 1000.0,
           median(slot, cfg->frames) / 1000.0, (double)(after.snapshots - before.snapshots) / (nd * fd),
           (double)(after.live_uniforms - before.live_uniforms) / (nd * fd),
           (double)(after.scratch_bytes - before.scratch_bytes) / (nd * fd),
           (double)(after.snapshot_drops - before.snapshot_drops) / (nd * fd), (ull)(vkp_bench_primed - primed0), (ull)d,
           (ull)(after.skipped - before.skipped), (ull)after.lost, (ull)ms.graphics_misses);
#ifdef OOT2_BENCH_TIMERS
    print_split(cfg->pattern, n, cfg->kind_mirror, cfg->frames);
    printf("SPLITTOTAL pattern=%c n=%d kind=%s wall_ns_draw=%.1f\n", cfg->pattern, n, cfg->kind_mirror ? "mirror" : "import",
           (double)(median(w, cfg->frames)) / nd);
#endif
    fflush(stdout);
    free(wall), free(cpu), free(rwall), free(rcpu), free(ring), free(slot);
    free(rec.b);
}

static void usage(void) {
    fprintf(stderr, "usage: vk_replay_cost --base B --variants V --map M --keys K [--pattern a|b|c|d] [--n 340,1000,...] "
            "[--kind import|mirror] [--frames 20] [--warmup 5] [--evict-mb 64] [--filler 6000]\n");
    exit(2);
}

int main(int argc, char **argv) {
    Cfg cfg = {.pattern = 'b', .frames = 20, .warmup = 5, .evict_mb = 64, .filler = 6000};
    cfg.ns[cfg.nns++] = 340, cfg.ns[cfg.nns++] = 1000, cfg.ns[cfg.nns++] = 2000, cfg.ns[cfg.nns++] = 3000;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--base") && v) cfg.base = v, i++;
        else if (!strcmp(a, "--variants") && v) cfg.variants = v, i++;
        else if (!strcmp(a, "--map") && v) cfg.map = v, i++;
        else if (!strcmp(a, "--keys") && v) cfg.keys = v, i++;
        else if (!strcmp(a, "--pattern") && v) cfg.pattern = v[0], i++;
        else if (!strcmp(a, "--kind") && v) cfg.kind_mirror = !strcmp(v, "mirror"), i++;
        else if (!strcmp(a, "--frames") && v) cfg.frames = atoi(v), i++;
        else if (!strcmp(a, "--warmup") && v) cfg.warmup = atoi(v), i++;
        else if (!strcmp(a, "--evict-mb") && v) cfg.evict_mb = atoi(v), i++;
        else if (!strcmp(a, "--filler") && v) cfg.filler = (uint32_t)atoi(v), i++;
        else if (!strcmp(a, "--n") && v) {
            cfg.nns = 0;
            for (char *s = strdup(v), *t = strtok(s, ","); t && cfg.nns < 32; t = strtok(NULL, ",")) cfg.ns[cfg.nns++] = atoi(t);
            i++;
        } else usage();
    }
    if (!cfg.base || !cfg.map || !cfg.keys || cfg.frames < 1 ||
        (cfg.pattern != 'a' && cfg.pattern != 'b' && cfg.pattern != 'c' && cfg.pattern != 'd'))
        usage();
    g_start_ns = plat_now_ns();
    calibrate_tsc();
    for (int i = 0; i < 2; i++) {
        g_arena[i] = calloc(1, ARENA_BYTES);
        CHECK(g_arena[i]);
        for (size_t o = 0; o < ARENA_BYTES; o += 4096) g_arena[i][o] = 1;
    }
    g_evict_bytes = (size_t)cfg.evict_mb << 20;
    if (g_evict_bytes) {
        g_evict = malloc(g_evict_bytes);
        CHECK(g_evict);
        memset(g_evict, 1, g_evict_bytes);
    }
    setup_memory();
    open_recorder();
    setup_backing();
    char why[512];
    if (!vkp_store_open_map(cfg.base, cfg.variants, cfg.map, why, sizeof why)) {
        fprintf(stderr, "the shader packs cannot be opened: %s\n", why);
        return 2;
    }
    vkp_bench_open(cfg.filler);
    load_manifest(cfg.keys);
    make_stages();
    register_textures();
    make_ops();
#ifdef OOT2_BENCH_TIMERS
    calibrate_timers();
#endif
    for (int i = 0; i < cfg.nns; i++) run_config(&cfg, cfg.ns[i]);
    vkrec_idle();
    gpu_sync_finish();
#ifdef _WIN32
    {   // this process's own CPU time, so that the driver can tell what other processes used while it ran
        FILETIME created, exited, kernel, user;
        GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
        uint64_t k = (uint64_t)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime, u = (uint64_t)user.dwHighDateTime << 32 | user.dwLowDateTime;
        printf("PROCESS cpu_s=%.3f wall_s=%.3f\n", (double)(k + u) / 1e7, (double)(plat_now_ns() - g_start_ns) / 1e9);
    }
#endif
    fflush(stdout);
    return 0;
}
