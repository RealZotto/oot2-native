// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's draw-to-draw counters (oot2/replay_count.h): how often a graphics draw repeats what the draw before it in
// the same submission used, and the line group that prints the replay's own statistics. Nothing here changes what the
// replay does: every function reads what it is given and counts. The replay thread is the only caller of the draw
// functions (the submission gate serialises it), so the state of the previous draw is plain static data.
#include <string.h>

#include "oot2/backing.h"
#include "oot2/replay_count.h"
#include "oot2/texsync.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_image.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_replay.h"
#include "oot2/vk_texture.h"
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
}
