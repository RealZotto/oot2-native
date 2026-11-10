// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the memory line (runtime/nvn/nvn_memline.c, oot2/memline.h), driven by tests/test_runtime_memline.py:
// `memline_test <case> <folder>`. Each case runs in its own process on the null back end (no device term exists there: the
// Vulkan fill is not registered, and the rows say 0 for them). A passing case prints "ok <case>" and the numbers the driver
// reads. Counters of the device are synthetic, put in by the test hook.
//   sum     memline_sum on a hand-made term set: the total, a -1 term counting 0, the residual's sign
//   rows    the helper thread writes minute, title, scene and end rows with synthetic counters and a noted block
//   gather  the real gather on the null back end against rt_mem_measure, and its cost per row
//   mark    what memline_mark and memline_note cost the calling thread while the helper gathers
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "oot2/gfx.h"
#include "oot2/memline.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
typedef unsigned long long ull;

static void setup(void) {
    RtMemConfig cfg = {.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    Cpu c = {0};
    c.x[0] = 0, c.x[1] = 32 * MiB;
    rt_svc(&c, 0x01, 0);
    CHECK(c.x[0] == 0);
}

// A term set that sums to 994 MiB of a private commit of 1000 MiB
static void hand_made(MemTerms *t) {
    memset(t, 0, sizeof *t);
    t->stack_committed = 8 * MiB, t->heap_committed = 300 * MiB, t->alias_committed = 100 * MiB;
    t->modules_committed = 50 * MiB, t->kernel_committed = 2 * MiB;
    t->wc32_bytes = 64 * MiB, t->wcodd_bytes = 20 * MiB, t->host_other_bytes = 190 * MiB;
    t->bk_block_bytes = 120 * MiB, t->tex_arena_bytes = 100 * MiB, t->img_own_bytes = 40 * MiB;
    t->private = 1000 * MiB;
}

static void case_sum(void) {
    MemTerms t;
    hand_made(&t);
    int64_t res;
    uint64_t total = memline_sum(&t, t.private, &res);
    CHECK(total == 994 * MiB);
    CHECK(res == (int64_t)(6 * MiB));
    t.img_own_bytes = UINT64_MAX;   // a term that could not be read counts 0
    CHECK(memline_sum(&t, t.private, &res) == 954 * MiB);
    CHECK(res == (int64_t)(46 * MiB));
    CHECK(memline_sum(&t, 10 * MiB, &res) == 954 * MiB && res < 0);
    CHECK(memline_sum(&t, 1000 * MiB, NULL) == 954 * MiB);
    printf("ok sum\n");
}

static unsigned g_calls;
static void synthetic(MemTerms *t) {
    hand_made(t);   // device counters, and the guest and host ones too: the rows then sum by construction
    unsigned n = __atomic_add_fetch(&g_calls, 1, __ATOMIC_RELAXED);
    t->heap_committed += (uint64_t)n * MiB;
    t->private += (uint64_t)n * MiB;
    t->astc_dec_w4 = 100 * MiB, t->astc_dec_w6 = 1500 * MiB, t->astc_dec_w12 = 40 * MiB, t->astc_dec_srgb_bytes = 900 * MiB;
    t->astc_dec_arena_bytes = 1600 * MiB, t->astc_dec_images = 321, t->bc_equiv_bytes = 410 * MiB;
    t->native_bc_bytes = 55 * MiB, t->other_bytes = 66 * MiB;
    t->img_own = 3, t->img_live = 800, t->vk_heap0_used = 2000 * MiB, t->vk_heap0_budget = 7000 * MiB;
}

static void case_rows(const char *folder) {
    setup();
    memline_set_test_hook(synthetic);
    memline_note("test-site", 4 * MiB);
    memline_note("test-site", 6 * MiB);
    memline_note("small", MiB - 1);   // under 1 MiB: not noted
    memline_start(folder, "test", 1);
    CHECK(memline_on());
    plat_sleep_ns(500000000ull);
    memline_mark(MEMLINE_TITLE, 1800);
    plat_sleep_ns(300000000ull);
    memline_mark(MEMLINE_SCENE, 0);
    plat_sleep_ns(300000000ull);
    memline_unnote("test-site", 4 * MiB);
    memline_end();
    memline_end();   // a second end is nothing
    CHECK(!memline_on());
    printf("ok rows\n");
}

static void case_gather(const char *folder) {
    (void)folder;
    setup();
    MemTerms t;
    RtMemStats m;
    memline_gather(&t, 0);   // the first call opens the system queries
    uint64_t t0 = plat_now_ns(), worst = 0;
    for (int i = 0; i < 20; i++) {
        uint64_t a = plat_now_ns();
        memline_gather(&t, 0);
        uint64_t d = plat_now_ns() - a;
        if (d > worst) worst = d;
    }
    uint64_t avg = (plat_now_ns() - t0) / 20;
    rt_mem_measure(&m);
    CHECK(t.heap_committed == m.r[RT_MEM_HEAP].committed);
    CHECK(t.stack_committed == m.r[RT_MEM_STACK].committed);
    CHECK(t.alias_committed == m.r[RT_MEM_ALIAS].committed);
    CHECK(t.private > 0 && t.working_set > 0 && t.ram_available > 0);
    CHECK(t.nvn_buffers == 0 && t.nvn_textures == 0 && t.nvn_pools == 0);
    CHECK(t.vk_heap0_used == 0 && t.bk_block_bytes == 0 && t.img_own_bytes == 0);   // the null back end: no device
    MemTerms c;
    memline_gather(&c, 1);   // the cheap gather: what needs a runtime lock reads -1
    CHECK(c.heap_committed == UINT64_MAX && c.bk_block_bytes == UINT64_MAX && c.private > 0);
#ifdef _WIN32
    CHECK(t.wc32_bytes != UINT64_MAX && t.host_other_bytes > 0);
    CHECK(t.sys_committed > 0 && t.sys_commit_limit >= t.sys_committed);
#endif
    printf("ok gather\ngather_avg_us %llu\ngather_max_us %llu\nheap_committed %llu\nhost_other_bytes %lld\n",
           (ull)(avg / 1000), (ull)(worst / 1000), (ull)t.heap_committed, (long long)t.host_other_bytes);
}

static void case_mark(const char *folder) {
    setup();
    memline_start(folder, "mark", 2);
    enum { N = 1000000 };
    uint64_t a = plat_now_ns();
    for (int i = 0; i < N; i++) {
        memline_mark(MEMLINE_SCENE, 5);   // no scene present set: returns at once
        memline_note("mark-site", 100);   // under 1 MiB: returns at once
    }
    uint64_t per = (plat_now_ns() - a) / N;
    memline_note("mark-site", 2 * MiB);
    uint64_t b = plat_now_ns();
    memline_mark(MEMLINE_TITLE, 7);   // sets a flag
    uint64_t one = plat_now_ns() - b;
    memline_end();
    printf("ok mark\nmark_note_ns %llu\nmark_title_ns %llu\n", (ull)per, (ull)one);
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const char *cs = argv[1];
    if (!strcmp(cs, "sum")) case_sum();
    else if (!strcmp(cs, "rows")) case_rows(argv[2]);
    else if (!strcmp(cs, "gather")) case_gather(argv[2]);
    else if (!strcmp(cs, "mark")) case_mark(argv[2]);
    else return 2;
    return 0;
}
