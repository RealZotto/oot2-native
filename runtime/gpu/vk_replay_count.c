// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's draw-to-draw counters (oot2/replay_count.h): how often a graphics draw repeats what the draw before it in
// the same submission used, and the line group that prints the replay's own statistics. Nothing here changes what the
// replay does: every function reads what it is given and counts. The replay thread is the only caller of the draw
// functions (the submission gate serialises it), so the state of the previous draw is plain static data.
#include <stdio.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/replay_count.h"
#include "oot2/texsync.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_image.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_replay.h"
#include "oot2/vk_texture.h"
#include "plat.h"
#include "vk_replay_int.h"

#define SEEN_SLOTS 4096     // distinct tuples a submission remembers (a power of two); past 2,048 a tuple counts as new
#define LRU 16

static struct {
    int have;                                  // a draw of this submission came before
    const nvn_stage *task, *vs, *ps;
    uint8_t td[SHC_DIGEST_BYTES], vd[SHC_DIGEST_BYTES], pd[SHC_DIGEST_BYTES];
    VkpFormats formats;
    VkPipeline pipe;
    int vin_have;
    uint8_t vin[1024];
    size_t vin_n;
} prev;
static const nvn_stage *last_task, *last_vs, *last_ps;   // the last draw of the submissions before, never cleared
static int have_last;
static uint64_t seen[SEEN_SLOTS];
static uint32_t nseen;
static uint64_t lru[LRU];
static int nlru;

static uint64_t mix(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

static int formats_equal(const VkpFormats *a, const VkpFormats *b) {
    if (a->ncolor != b->ncolor || a->depth != b->depth || a->stencil != b->stencil) return 0;
    for (uint32_t i = 0; i < a->ncolor; i++)
        if (a->color[i] != b->color[i]) return 0;
    return 1;
}

static void report(RcLine *l);
static void report_stages(RcLine *l);

void vkrc_submit_begin(void) {
    static int hooked;   // the replay thread's alone
    if (!hooked) hooked = 1, rc_hook(report);
    memset(seen, 0, sizeof seen);
    nseen = 0;
    prev.have = 0;
    prev.vin_have = 0;
    nlru = 0;
}

void vkrc_submit_end(uint32_t work) {
    RC_ADD(RC_SUB_N, 1);
    RC_ADD(RC_SUB_WORK, work);
    int h = work == 0 ? RC_SUB_H0 : work < 16 ? RC_SUB_H1 : work < 64 ? RC_SUB_H2 : work < 256 ? RC_SUB_H3
            : work < 1024 ? RC_SUB_H4 : work < 4096 ? RC_SUB_H5 : RC_SUB_H6;
    RC_ADD(h, 1);
}

void vkrc_draw(int mesh, const VkpStage *task, const VkpStage *vs, const VkpStage *ps, const VkpFormats *formats,
               VkPipeline pipe) {
    static const uint8_t zero[SHC_DIGEST_BYTES];
    const nvn_stage *tsh = task ? task->shape : NULL;
    const uint8_t *td = task ? task->ledger_digest : zero;
    RC_ADD(mesh ? RC_DRAW_MESH : RC_DRAW_CLASSIC, 1);
    if (!prev.have) {
        RC_ADD(RC_D_FIRST, 1);
        if (have_last && last_task == tsh && last_vs == vs->shape && last_ps == ps->shape) RC_ADD(RC_D_FIRST_PAIR_XSUB, 1);
    } else {
        int pair = prev.task == tsh && prev.vs == vs->shape && prev.ps == ps->shape;
        int vdig = !memcmp(prev.vd, vs->ledger_digest, SHC_DIGEST_BYTES);
        int pdig = !memcmp(prev.pd, ps->ledger_digest, SHC_DIGEST_BYTES);
        int tdig = !memcmp(prev.td, td, SHC_DIGEST_BYTES);
        RC_ADD(RC_D_CMP, 1);
        RC_ADD(RC_D_VS_SAME, prev.vs == vs->shape);
        RC_ADD(RC_D_PS_SAME, prev.ps == ps->shape);
        RC_ADD(RC_D_PAIR_SAME, pair);
        RC_ADD(RC_D_VSDIG_SAME, vdig);
        RC_ADD(RC_D_PSDIG_SAME, pdig);
        RC_ADD(RC_D_KEY_SAME, pair && vdig && pdig && tdig && formats_equal(&prev.formats, formats));
        RC_ADD(RC_D_PIPE_SAME, prev.pipe == pipe);
    }
    // the tuple a pipeline is keyed by: the stages, their answers digests and the attachment formats
    uint64_t h = 0xCBF29CE484222325ull;
    h = mix(h, &tsh, sizeof tsh), h = mix(h, &vs->shape, sizeof vs->shape), h = mix(h, &ps->shape, sizeof ps->shape);
    h = mix(h, td, SHC_DIGEST_BYTES), h = mix(h, vs->ledger_digest, SHC_DIGEST_BYTES);
    h = mix(h, ps->ledger_digest, SHC_DIGEST_BYTES);
    h = mix(h, &formats->ncolor, sizeof formats->ncolor);
    h = mix(h, formats->color, sizeof formats->color[0] * formats->ncolor);
    h = mix(h, &formats->depth, sizeof formats->depth), h = mix(h, &formats->stencil, sizeof formats->stencil);
    h |= 1;   // 0 marks an empty slot
    int hit = 0;
    for (int i = 0; i < nlru && !hit; i++)
        if (lru[i] == h) {
            memmove(lru + 1, lru, sizeof lru[0] * (size_t)i);
            hit = 1;
        }
    if (!hit) {
        if (nlru < LRU) nlru++;
        memmove(lru + 1, lru, sizeof lru[0] * (size_t)(nlru - 1));
    }
    lru[0] = h;
    RC_ADD(RC_D_TUPLE_LRU16, hit);
    uint32_t at = (uint32_t)h & (SEEN_SLOTS - 1);
    int found = 0;
    while (seen[at]) {
        if (seen[at] == h) {
            found = 1;
            break;
        }
        at = (at + 1) & (SEEN_SLOTS - 1);
    }
    if (!found) {
        RC_ADD(RC_D_TUPLE_NEW, 1);
        if (nseen < SEEN_SLOTS / 2) seen[at] = h, nseen++;
    }
    prev.have = 1;
    prev.task = tsh, prev.vs = vs->shape, prev.ps = ps->shape;
    memcpy(prev.td, td, SHC_DIGEST_BYTES);
    memcpy(prev.vd, vs->ledger_digest, SHC_DIGEST_BYTES);
    memcpy(prev.pd, ps->ledger_digest, SHC_DIGEST_BYTES);
    prev.formats = *formats;
    prev.pipe = pipe;
    last_task = tsh, last_vs = vs->shape, last_ps = ps->shape;
    have_last = 1;
}

void vkrc_vin(const void *state, size_t state_n, size_t layout_n, size_t full_n) {
    if (full_n > sizeof prev.vin) return;   // never: sized for the state
    RC_ADD(RC_D_VIN_N, 1);
    if (prev.vin_have && prev.vin_n == full_n) {
        RC_ADD(RC_D_VIN_STATE_SAME, !memcmp(prev.vin, state, state_n));
        RC_ADD(RC_D_VIN_LAYOUT_SAME, !memcmp(prev.vin, state, layout_n));
        RC_ADD(RC_D_VIN_FULL_SAME, !memcmp(prev.vin, state, full_n));
    }
    memcpy(prev.vin, state, full_n), prev.vin_n = full_n;
    prev.vin_have = 1;
}

void vkrc_dyn(int same) {
    RC_ADD(RC_D_DYN_N, 1);
    RC_ADD(RC_D_DYN_SAME, same);
}

// The replay's own statistics (VkrStats, which nothing printed before) and the backing map's resolve count.
static void report(RcLine *l) {
    VkrStats s;
    vkr_stats(&s);
    rc_group(l, "vkr");
    rc_put(l, "vkr_frames", s.frames), rc_put(l, "vkr_records", s.records), rc_put(l, "vkr_draws", s.draws);
    rc_put(l, "vkr_dispatches", s.dispatches), rc_put(l, "vkr_clears", s.clears), rc_put(l, "vkr_passes", s.passes);
    rc_put(l, "vkr_barriers", s.barriers), rc_put(l, "vkr_snapshots", s.snapshots);
    rc_put(l, "vkr_snapshot_bytes", s.snapshot_bytes), rc_put(l, "vkr_snapshot_drops", s.snapshot_drops);
    rc_put(l, "vkr_transfers", s.transfers), rc_put(l, "vkr_transfer_bytes", s.transfer_bytes);
    rc_put(l, "vkr_live_uniforms", s.live_uniforms), rc_put(l, "vkr_storage_writes", s.storage_writes);
    rc_put(l, "vkr_storage_images", s.storage_images);
    rc_put(l, "vkr_unregistered_textures", s.unregistered_textures), rc_put(l, "vkr_scratch_bytes", s.scratch_bytes);
    rc_put(l, "vkr_draw_ops", s.draw_ops), rc_put(l, "vkr_render_off", s.render_off), rc_put(l, "vkr_skipped", s.skipped);
    rc_put(l, "vkr_lost", s.lost), rc_put(l, "vkr_collected", s.collected), rc_put(l, "vkr_raw_binds", s.raw_binds);
    rc_put(l, "vkr_full_rate", s.full_rate);
    rc_put(l, "vkr_movie_skipped", s.movie_skipped), rc_put(l, "vkr_movie_blits", s.movie_blits);
    uint64_t resolves, resolve_misses;
    backing_resolve_counts(&resolves, &resolve_misses);   // two atomics, no lock: the stopping thread may hold the map's lock
    rc_group(l, "backing");
    rc_put(l, "backing_resolves", resolves), rc_put(l, "backing_resolve_misses", resolve_misses);
    // The counters of work the replay drops, approximates or reads as nothing without a stop, which nothing printed before
    // (vkr_unregistered_textures, above, was printed already): what rendering off turned off by cause, the conditions that
    // decided against drawing, the waits the work before them did not satisfy (all, and the first of each call site in
    // each frame), the reports into memory the CPU cannot see, the sampler approximations, the texture regions uploaded as
    // zeros, and the textures that got no image.
    // The line prints at a stop, from the thread that stopped, and at every window, from the present thread. A thread
    // that stops inside a locked section (vkt_upload names its fatals under the texture lock) would wait for itself on
    // a lock here, and the present thread would wait behind a texture readback, so these four are read with a try: a
    // busy lock leaves its values out of the line and sets its bit in silent_unavailable (1 reports, 2 textures, 4
    // texture sync, 8 images). The backing map's two printed counters are atomics, read with no lock, so they have no bit. A key left out keeps its baseline, so the next window line shows the change over both.
    VkrReportStats r = {0};
    VkbStats bs;
    vkb_stats(&bs);
    VktStats ts = {0};
    TexSyncStats xs = {0};
    VkbImageStats is = {0};
    int have_r = vkr_report_stats_try(&r), have_t = vkt_stats_try(&ts), have_x = texsync_stats_try(&xs);
    int have_i = vkb_image_stats_try(&is);
    rc_group(l, "silent");
    rc_put_raw(l, "silent_unavailable", (uint64_t)(!have_r | !have_t << 1 | !have_x << 2 | !have_i << 3));
    rc_put(l, "vkr_render_off_enable", s.render_off_enable), rc_put(l, "vkr_render_off_cond", s.render_off_cond);
    rc_put(l, "vkr_clears_off", s.clears_off);
    if (have_r) {
        rc_put(l, "rep_conditions", r.conditions), rc_put(l, "rep_conditions_false", r.conditions_false);
        rc_put(l, "rep_waits", r.waits), rc_put(l, "rep_waits_unsatisfied", r.waits_unsatisfied);
        rc_put(l, "rep_waits_unsat_first_per_site_frame", r.waits_unsat_first), rc_put(l, "rep_reports_dropped", r.dropped);
    }
    rc_put(l, "bind_wrap_approx", bs.wrap_approx), rc_put(l, "bind_border_approx", bs.border_approx);
    if (have_t) rc_put(l, "tex_zero_regions_uploaded", ts.zero_regions);
    if (have_x) rc_put(l, "texsync_zero_regions", xs.zero_regions);
    if (have_i) rc_put(l, "img_skipped_cpu", is.skipped_cpu), rc_put(l, "img_skipped_format", is.skipped_format);
    report_stages(l);
}

// ---------- the replay stage timer (vk_replay_int.h VKST_*; OOT2_REPLAY_COUNTERS, group "stages")
// Where the replay walk's CPU time goes, by class of operation, per window of presents. A class is timed from vkst_in to
// vkst_out by the replaying thread (the only one that calls them), inclusive of the classes it contains: a draw holds its
// pipeline lookup, descriptor build, vertex input and dynamic state; the descriptor build holds the backing resolves, the
// uniform ranges and their snapshot search. Every call is counted. The clock is read for one call in `every`: the classes
// that run thousands of times a present are sampled, and a sample stands for `every` calls (its time is added `every`
// times), so the printed time of a class is an estimate of the sum of all its calls, exact in the count.
//
// The sampling period of a class is a prime (5, 17, 31), never a power of two: the descriptor build runs twice a draw
// (vertex stage, then pixel stage) and the ledger once a record, so a period that divides such a pattern would sample the
// same stage or the same opcode every time, and a sample scaled by the period would then be consistently wrong while the
// counts stayed exact. A call is sampled when the calls since the last sample have reached the period (a down-counter, no
// divide at the site): the first call of a class is sampled, then one in `every`.
//
// The cost, in three parts, none of them a bound on the whole perturbation. st_clock_reads times st_clock_ns_x1000 (the
// mean of 2,000 consecutive reads, times 1000, taken once when the first line is printed) is st_clock_overhead_ns.
// st_site_calls is the number of vkst_in calls, sampled or not, and st_site_ns_x1000 the measured cost of one unsampled
// vkst_in plus vkst_out pair in a hot loop (a floor: the replay's own cache pressure is not in it); their product is
// st_site_overhead_ns. st_hash_ns is the time spent hashing the recordings for the repeat counters, which runs inside the
// walk's time. All three are measured in a hot loop and are floors; the perturbation of a real run is the difference of
// st_walk_ns, or of the frame time, between a run with the counters on and one with them off. With the counters off a
// site tests rc_on and nothing else.
#define STAGE_NAMES(X)                                                                                                 \
    X(WALK, "walk", 1) X(DRAW, "draw", 1) X(DISPATCH, "dispatch", 1) X(CLEAR, "clear", 1) X(COPY, "copy", 1)           \
    X(QUERY, "query", 1) X(BARRIER, "barrier", 1) X(PASS_BEGIN, "pass_begin", 1) X(PASS_END, "pass_end", 1)            \
    X(PIPELINE, "pipeline", 5) X(STAGE_RES, "descriptors", 5) X(VERTEX, "vertex", 5) X(DYN, "dynamic_state", 5)        \
    X(RESOLVE, "resolve", 17) X(SNAPSHOT, "snapshot", 17) X(UNIFORM, "uniform", 17) X(LEDGER, "ledger", 31)

static const char *const st_name[VKST_N] = {
#define ST_NAME(id, name, every) name,
    STAGE_NAMES(ST_NAME)
#undef ST_NAME
};
// One more slot than classes: the calibration of the site's own cost (the printing thread's, never read as a class).
static const uint32_t st_every[VKST_N + 1] = {
#define ST_EVERY(id, name, every) every,
    STAGE_NAMES(ST_EVERY)
#undef ST_EVERY
    UINT32_MAX};
typedef struct {
    uint64_t n, est_ns, samples;
    uint32_t left;   // calls to pass before the next sample: the replaying thread's alone
} StSlot;
static StSlot st[VKST_N + 1];
static uint64_t st_reads, st_hash_ns;

// The timer's clock: plat_now_ns, or a test's (a function that returns a time that is always odd, since a sample's start
// time carries a low bit that marks it).
uint64_t (*vkst_clock)(void);
static uint64_t now(void) { return vkst_clock ? vkst_clock() : plat_now_ns(); }

// A cumulative counter of one writer (the replaying thread), read by the printing one: stored relaxed so the read is whole.
static void st_add(uint64_t *p, uint64_t v) { __atomic_store_n(p, *p + v, __ATOMIC_RELAXED); }

uint64_t vkst_in(int cls) {
    StSlot *s = &st[cls];
    st_add(&s->n, 1);
    if (s->left) {
        s->left--;
        return 0;
    }
    s->left = st_every[cls] - 1;
    st_add(&st_reads, 1);
    return now() | 1u;   // never 0: 0 marks a call that was not sampled
}

void vkst_out(int cls, uint64_t t0) {
    if (!t0) return;
    uint64_t t = now();
    st_add(&st_reads, 1);
    st_add(&st[cls].samples, 1);
    st_add(&st[cls].est_ns, (t > t0 ? t - t0 : 0) * st_every[cls]);
}

int vkst_class_of(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND: case OP_DRAW_MESH: case OP_DRAW_MESH_IND: return VKST_DRAW;
    case OP_DISPATCH: case OP_DISPATCH_IND: return VKST_DISPATCH;
    case OP_CLEAR_COLOR: case OP_CLEAR_DS: return VKST_CLEAR;
    case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_UPDATE_UB: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T:
    case OP_CLEAR_TEX: return VKST_COPY;
    case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_SIGNAL_EVENT: case OP_WAIT_EVENT: case OP_RENDER_COND:
        return VKST_QUERY;
    case OP_BARRIER: return VKST_BARRIER;
    default: return -1;
    }
}

// ---- why more draws are replayed than the game's calls count
// The game's census counts the calls the guest makes while it records; the replay walks a recording every time it is
// submitted. So the replayed draws are the draws of the recordings walked: these counters give the submissions, the
// recordings walked, those walked before (same address, same bytes, same number of records), the draw operations made
// inside those, and the draw operations by kind.
//
// The key of a repeat is the recording's address, the hash of its bytes and its record count. The address alone is not
// enough: the allocator recycles recording memory, so a recording freed and written again at the same address with other
// content is a new recording, which the bytes tell apart. What the key cannot tell apart is a recording written again at
// the same address with the same bytes (a game that rebuilds an identical list every frame): that counts as a repeat, and
// is one in the sense that matters here, the replay walking the same work again. The set is never emptied per submission or
// per present: it spans presents (a recording submitted every frame is seen again in the next), and is emptied only when
// half full, so a repeat across that emptying is not seen.
enum { K_DRAW, K_DRAW_IDX, K_DRAW_IDX_IND, K_DRAW_MESH, K_DRAW_MESH_IND, K_DISPATCH, K_DISPATCH_IND, K_N };
static const char *const k_name[K_N] = {"draw", "draw_idx", "draw_idx_ind", "draw_mesh", "draw_mesh_ind", "dispatch",
                                        "dispatch_ind"};
static uint64_t k_ops[K_N], k_inst, c_subs, c_handles, c_repeat, c_ops, c_ops_repeat, c_records, c_records_repeat;

void vkst_op(uint32_t op, int32_t instances) {
    int k = op == OP_DRAW ? K_DRAW : op == OP_DRAW_IDX ? K_DRAW_IDX : op == OP_DRAW_IDX_IND ? K_DRAW_IDX_IND
          : op == OP_DRAW_MESH ? K_DRAW_MESH : op == OP_DRAW_MESH_IND ? K_DRAW_MESH_IND : op == OP_DISPATCH ? K_DISPATCH
          : op == OP_DISPATCH_IND ? K_DISPATCH_IND : -1;
    if (k < 0) return;
    st_add(&k_ops[k], 1);
    if (k == K_DRAW || k == K_DRAW_IDX) st_add(&k_inst, instances > 0 ? (uint64_t)instances : 1u);
}

#define SEEN_HANDLES 8192   // a power of two; the set is emptied when half full, so a repeat past that is not seen
static uint64_t seen_handle[SEEN_HANDLES];
static uint32_t nseen_handle;

void vkst_sub(void) { st_add(&c_subs, 1); }

// The bytes of a recording, eight at a time (every record is a multiple of 8 bytes).
static uint64_t content_hash(const void *p, size_t n) {
    const uint8_t *b = p;
    uint64_t h = 0xCBF29CE484222325ull;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    for (; i < n; i++) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

void vkst_handle(uint64_t handle, const void *bytes, size_t nbytes, uint32_t records, uint32_t ops) {
    uint64_t h0 = plat_now_ns();   // the hash is the counters' own cost, and runs inside the walk's time: it is measured
    uint64_t key = (handle * 0x9E3779B97F4A7C15ull ^ (uint64_t)records * 0xC2B2AE3D27D4EB4Full ^
                    content_hash(bytes, nbytes) * 0xD6E8FEB86659FD93ull) | 1u;
    uint32_t at = (uint32_t)(key >> 17) & (SEEN_HANDLES - 1);
    int found = 0;
    while (seen_handle[at]) {
        if (seen_handle[at] == key) {
            found = 1;
            break;
        }
        at = (at + 1) & (SEEN_HANDLES - 1);
    }
    if (!found) {
        if (nseen_handle >= SEEN_HANDLES / 2) {
            memset(seen_handle, 0, sizeof seen_handle), nseen_handle = 0;
            at = (uint32_t)(key >> 17) & (SEEN_HANDLES - 1);
        }
        seen_handle[at] = key, nseen_handle++;
    }
    st_add(&c_handles, 1), st_add(&c_records, records), st_add(&c_ops, ops);
    if (found) st_add(&c_repeat, 1), st_add(&c_records_repeat, records), st_add(&c_ops_repeat, ops);
    uint64_t h1 = plat_now_ns();
    st_add(&st_hash_ns, h1 > h0 ? h1 - h0 : 0);
}

// The cost of one clock read in nanoseconds times 1000, measured once (on the real clock, whatever the timer reads).
static uint64_t clock_cost_x1000(void) {
    static uint64_t cost;
    if (cost) return cost;
    uint64_t t0 = plat_now_ns(), last = t0;
    for (int i = 0; i < 2000; i++) last = plat_now_ns();
    cost = (last - t0) * 1000 / 2000;
    if (!cost) cost = 1;
    return cost;
}

// The cost of one site that does not sample, nanoseconds times 1000, measured once: vkst_in then vkst_out on the
// calibration slot, which never samples, so the call, the counter and the branch are all that is timed.
static uint64_t site_cost_x1000(void) {
    static uint64_t cost;
    if (cost) return cost;
    st[VKST_N].left = UINT32_MAX;
    uint64_t t0 = plat_now_ns();
    for (int i = 0; i < 2000; i++) vkst_out(VKST_N, vkst_in(VKST_N));
    cost = (plat_now_ns() - t0) * 1000 / 2000;
    memset(&st[VKST_N], 0, sizeof st[VKST_N]);
    if (!cost) cost = 1;
    return cost;
}

static void report_stages(RcLine *l) {
    rc_group(l, "stages");
    for (int i = 0; i < VKST_N; i++) {
        char k[48];
        snprintf(k, sizeof k, "st_%s_n", st_name[i]);
        rc_put(l, k, __atomic_load_n(&st[i].n, __ATOMIC_RELAXED));
        snprintf(k, sizeof k, "st_%s_ns", st_name[i]);
        rc_put(l, k, __atomic_load_n(&st[i].est_ns, __ATOMIC_RELAXED));
        snprintf(k, sizeof k, "st_%s_samples", st_name[i]);
        rc_put(l, k, __atomic_load_n(&st[i].samples, __ATOMIC_RELAXED));
    }
    uint64_t reads = __atomic_load_n(&st_reads, __ATOMIC_RELAXED), cost = clock_cost_x1000();
    uint64_t sites = 0, site_cost = site_cost_x1000();
    for (int i = 0; i < VKST_N; i++) sites += __atomic_load_n(&st[i].n, __ATOMIC_RELAXED);
    rc_put(l, "st_clock_reads", reads);
    rc_put_raw(l, "st_clock_ns_x1000", cost);
    rc_put(l, "st_clock_overhead_ns", reads * cost / 1000);
    rc_put(l, "st_site_calls", sites);
    rc_put_raw(l, "st_site_ns_x1000", site_cost);
    rc_put(l, "st_site_overhead_ns", sites * site_cost / 1000);
    rc_put(l, "st_hash_ns", __atomic_load_n(&st_hash_ns, __ATOMIC_RELAXED));
    rc_put(l, "st_subs", __atomic_load_n(&c_subs, __ATOMIC_RELAXED));
    rc_put(l, "st_handles", __atomic_load_n(&c_handles, __ATOMIC_RELAXED));
    rc_put(l, "st_handles_repeat", __atomic_load_n(&c_repeat, __ATOMIC_RELAXED));
    rc_put(l, "st_handle_records", __atomic_load_n(&c_records, __ATOMIC_RELAXED));
    rc_put(l, "st_handle_records_repeat", __atomic_load_n(&c_records_repeat, __ATOMIC_RELAXED));
    rc_put(l, "st_handle_ops", __atomic_load_n(&c_ops, __ATOMIC_RELAXED));
    rc_put(l, "st_handle_ops_repeat", __atomic_load_n(&c_ops_repeat, __ATOMIC_RELAXED));
    for (int i = 0; i < K_N; i++) {
        char k[48];
        snprintf(k, sizeof k, "st_ops_%s", k_name[i]);
        rc_put(l, k, __atomic_load_n(&k_ops[i], __ATOMIC_RELAXED));
    }
    rc_put(l, "st_draw_instances", __atomic_load_n(&k_inst, __ATOMIC_RELAXED));
}

// Zero the timer and the counters above (a test).
void vkst_reset(void) {
    memset(st, 0, sizeof st);
    memset(k_ops, 0, sizeof k_ops);
    memset(seen_handle, 0, sizeof seen_handle);
    nseen_handle = 0;
    st_reads = st_hash_ns = k_inst = c_subs = c_handles = c_repeat = c_ops = c_ops_repeat = c_records = c_records_repeat = 0;
}
