// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's workload counters (runtime/core/replay_count.c): how a real run feeds the Vulkan replay, counted without
// changing what the replay does. Off by default. The environment variable OOT2_REPLAY_COUNTERS turns them on: "1" (or
// "on") prints one line group every 600 presents, a number from 2 up is the window in presents, and 0 or unset is off.
// A run that ends, normally or by a stop, prints the totals once more. Lines go to the log (stderr) as
//   [rt] rc kind=window|exit group=<name> present=<n> presents=<window> key=value key=value ...
// every value an unsigned integer, window lines the change over the window, exit lines the whole run.
//
// A counter is a plain increment in the calling thread's own block (no atomic, no lock on the path); the blocks are summed
// when a line is printed, so a line read while a thread is counting may be a few increments behind. With the counters off
// a site costs one load of rc_on and a branch that is never taken.
#pragma once
#include <stddef.h>
#include <stdint.h>

// id, name in the line. Arrays by kind (c0, c2, roots; vertex stage, pixel stage, any other) are ids in a row.
#define RC_LIST(X)                                                                                                     \
    /* submissions: the replay's units of work, with a histogram of the draws and dispatches in each */                  \
    X(SUB_N, "sub_n") X(SUB_WORK, "sub_work") X(SUB_H0, "sub_h0") X(SUB_H1, "sub_h1_15") X(SUB_H2, "sub_h16_63")         \
    X(SUB_H3, "sub_h64_255") X(SUB_H4, "sub_h256_1023") X(SUB_H5, "sub_h1024_4095") X(SUB_H6, "sub_h4096_up")            \
    X(DRAW_CLASSIC, "draw_classic") X(DRAW_MESH, "draw_mesh") X(DRAW_DISPATCH, "dispatch")                              \
    /* a graphics draw that was issued, against the draw before it in the same submission */                            \
    X(D_FIRST, "d_first") X(D_CMP, "d_cmp") X(D_VS_SAME, "d_vs_same") X(D_PS_SAME, "d_ps_same")                         \
    X(D_PAIR_SAME, "d_pair_same") X(D_VSDIG_SAME, "d_vsdig_same") X(D_PSDIG_SAME, "d_psdig_same")                       \
    X(D_KEY_SAME, "d_key_same") X(D_PIPE_SAME, "d_pipe_same") X(D_FIRST_PAIR_XSUB, "d_first_pair_xsub")                 \
    X(D_TUPLE_NEW, "d_tuple_new") X(D_TUPLE_LRU16, "d_tuple_lru16")                                                     \
    X(D_VIN_N, "d_vin_n") X(D_VIN_STATE_SAME, "d_vin_state_same") X(D_VIN_LAYOUT_SAME, "d_vin_layout_same")             \
    X(D_VIN_FULL_SAME, "d_vin_full_same")                                                                               \
    X(D_DYN_N, "d_dyn_n") X(D_DYN_SAME, "d_dyn_same")                                                                   \
    /* stage lookups of the store: the stage's module is a variant */                                                   \
    X(CONTENT_N, "content_n") X(CONTENT_VARIANT, "content_variant") X(CONTENT_MISS, "content_miss")                     \
    /* the driver banks */                                                                                              \
    X(BANK_CALLS_C0, "bank_c0_calls") X(BANK_CALLS_C2, "bank_c2_calls") X(BANK_CALLS_ROOTS, "bank_roots_calls")         \
    X(BANK_EQUAL_C0, "bank_c0_equal") X(BANK_EQUAL_C2, "bank_c2_equal") X(BANK_EQUAL_ROOTS, "bank_roots_equal")         \
    X(BANK_BYTES_C0, "bank_c0_copied_bytes") X(BANK_BYTES_C2, "bank_c2_copied_bytes")                                   \
    X(BANK_BYTES_ROOTS, "bank_roots_copied_bytes")                                                                      \
    /* uniform ranges, snapshots, the written-range scans, and the backing resolves the replay itself asks for */       \
    X(UNI_CALLS, "uni_calls") X(UNI_ZERO, "uni_zero") X(UNI_SNAPSHOT, "uni_snapshot") X(UNI_LIVE, "uni_live")           \
    X(SNAP_CALLS, "snap_calls") X(SNAP_HIT, "snap_hit") X(SNAP_STEPS, "snap_search_steps")                              \
    X(WRITTEN_CALLS, "written_calls") X(WRITTEN_STEPS, "written_scan_steps") X(WROTE_CALLS, "wrote_calls")              \
    X(WROTE_DROP_STEPS, "wrote_drop_steps") X(WROTE_COVER_STEPS, "wrote_cover_steps")                                   \
    X(IND_SCAN_STEPS, "indirect_scan_steps")                                                                            \
    X(RES_UNI, "resolve_uniform") X(RES_VERTEX, "resolve_vertex") X(RES_STORAGE, "resolve_storage")                     \
    X(RES_INDEX, "resolve_index") X(RES_INDIRECT, "resolve_indirect")                                                   \
    /* the ledger tracker's answers digest, by stage (vertex, pixel, other), and texture binds */                       \
    X(DG_CALLS_VS, "digest_vs_calls") X(DG_CALLS_PS, "digest_ps_calls") X(DG_CALLS_OTHER, "digest_other_calls")         \
    X(DG_HIT_VS, "digest_vs_valid_hit") X(DG_HIT_PS, "digest_ps_valid_hit") X(DG_HIT_OTHER, "digest_other_valid_hit")   \
    X(DG_RECOMP_VS, "digest_vs_recompute") X(DG_RECOMP_PS, "digest_ps_recompute")                                       \
    X(DG_RECOMP_OTHER, "digest_other_recompute")                                                                        \
    X(DG_INSAME_VS, "digest_vs_inputs_same") X(DG_INSAME_PS, "digest_ps_inputs_same")                                   \
    X(DG_INSAME_OTHER, "digest_other_inputs_same")                                                                      \
    X(DG_VALSAME_VS, "digest_vs_value_same") X(DG_VALSAME_PS, "digest_ps_value_same")                                   \
    X(DG_VALSAME_OTHER, "digest_other_value_same")                                                                      \
    X(TEX_OPS, "tex_bind_ops") X(TEX_OPS_SAME_TYPE, "tex_bind_ops_same_type") X(TEX_BINDS, "tex_binds")                 \
    X(TEX_SAME_HANDLE, "tex_same_handle") X(TEX_SAME_TYPE, "tex_same_type")

enum RcId {
#define RC_ENUM(id, name) RC_##id,
    RC_LIST(RC_ENUM)
#undef RC_ENUM
    RC_N
};

typedef struct RcBlock {
    uint64_t c[RC_N];
} RcBlock;

extern int rc_on;
extern _Thread_local RcBlock *rc_tl;
RcBlock *rc_claim(void);
static inline void rc_add_(int id, uint64_t n) {
    RcBlock *b = rc_tl;
    if (!b) b = rc_claim();
    b->c[id] += n;
}
#define RC_ON() __builtin_expect(rc_on, 0)
#define RC_ADD(id, n)                                                                                                  \
    do {                                                                                                               \
        if (__builtin_expect(rc_on, 0)) rc_add_((id), (n));                                                            \
    } while (0)

// On with a window in presents (0: off). A test turns the counters on and off with it; a run reads the environment.
void rc_enable(unsigned window);
// The sum of every block (a test reads exact values with this, after the threads that counted have stopped).
void rc_sum(uint64_t out[RC_N]);
// Zero every block and forget the window's baseline (a test).
void rc_reset(void);
const char *rc_name(int id);

// A line group. A hook adds its own values to a line; each value is a cumulative count, the line prints its change over
// the window or the whole value at the exit.
typedef struct RcLine RcLine;
typedef void (*RcHook)(RcLine *l);
void rc_hook(RcHook h);   // once per function
void rc_group(RcLine *l, const char *name);
void rc_put(RcLine *l, const char *key, uint64_t cumulative);

// A present of the front end (nvn_census_present): the window's line group when the window is full.
void rc_present(uint64_t seq);
// The lines now (a test; the exit does it by itself).
void rc_print(const char *kind, uint64_t present);
