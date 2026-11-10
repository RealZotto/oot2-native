// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// BENCHMARK ONLY (tests/runtime/vk_replay_cost.c): the accumulators the replay-cost measurement reads.
// Compiled in by -DOOT2_BENCH_TIMERS and nowhere else; without it every macro here is the bare statement, so the code
// that includes this header is the code of the default build. A timer is two TSC reads and an add; the benchmark measures
// what a timer costs (an empty one, and one nested in another) and takes it off the figures it reports.
#pragma once
#include <stdint.h>

enum BenchTimer {
    BT_WALK,            // record() as the walk calls it, every record: the walk's own work and record_op
    BT_RECORD,          //   record_op as a whole: every record of the walk
    BT_DRAW_PREPARE,    //   draw_prepare as a whole (a graphics draw)
    BT_S1_DIGEST,       //     S1: replay_stage -> ledger_track_answers_digest (the vertex and the pixel stage)
    BT_S1_TEXT,         //       the digest was not cached: answers_text + free
    BT_S1_SHA,          //         rt_sha256 of the answers text
    BT_S2_BINDS,        //     S2: binds_of -> ops->binds (dev_binds)
    BT_S2_CONTENT,      //       vkp_stage_content_digest (the store lookup)
    BT_S3_GRAPHICS,     //     S3: ops->graphics (vkm_graphics -> vkp_find_graphics)
    BT_S3_KEY,          //       vkp_graphics_key
    BT_S3_KEY_DIGEST,   //         the cache_digest_text calls
    BT_S3_KEY_FORMATS,  //         formats_text
    BT_S3_KEY_PRINT,    //         the key's snprintf
    BT_S3_TABLE,        //       hash, mutex and chain of the pipeline table
    BT_S4_VERTEX,       //     S4: vertex_input
    BT_S4_RESOLVE,      //       one resolve per stream
    BT_S4_PACK,         //       vkrec_BindVertexBuffers2 and vkrec_SetVertexInputEXT
    BT_S5_C0,           //     vkn_c0_graphics (two a draw)
    BT_S5_RESOURCES,    //     S5: stage_resources (two a draw)
    BT_S5_BANK,         //       bank(): compare and copy of a driver bank
    BT_S5_C2,           //       vkn_c2_fill
    BT_S5_RESOLVE,      //       ops->resolve of a uniform range
    BT_S5_SNAPSHOT,     //       snapshot()
    BT_S5_SNAP_SEARCH,  //         its linear search of the copies made so far
    BT_S5_WRITTEN,      //       vkr_written
    BT_S5_WROTE,        //       vkr_wrote
    BT_S6_SUPPORT,      //     S6: the support buffer's scratch
    BT_S6_PUSHDESC,     //     S6: vkrec_PushDescriptorSetKHR
    BT_S6_PUSHCONST,    //     S6: vkrec_PushConstants
    BT_U_DYN,           //     not listed: dyn_emit (dyn_compute and the comparison)
    BT_U_STAMPS,        //     not listed: targets_hold_earlier_picture and stamp_targets
    BT_U_PROGRAM,       //   not listed: bind_program (the replay's) with ops->program_bound
    BT_U_LEDGER_OP,     //   not listed: ledger_track_op (the tracker, every record)
    BT_U_LEDGER_ROW,    //     the rows a draw makes (add_row)
    BT_DUMMY,           // the calibration's
    BT_COUNT
};

#ifdef OOT2_BENCH_TIMERS
extern uint64_t bench_acc[BT_COUNT], bench_n[BT_COUNT];
static inline uint64_t bt_now(void) { return __builtin_ia32_rdtsc(); }
#define BT_TIME(id, ...)                                                                                               \
    do {                                                                                                               \
        uint64_t bt0_ = bt_now();                                                                                      \
        __VA_ARGS__;                                                                                                   \
        bench_acc[id] += bt_now() - bt0_;                                                                              \
        bench_n[id]++;                                                                                                 \
    } while (0)
#else
#define BT_TIME(id, ...)                                                                                               \
    do {                                                                                                               \
        __VA_ARGS__;                                                                                                   \
    } while (0)
#endif
