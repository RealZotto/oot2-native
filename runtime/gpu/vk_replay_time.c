// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Where the frame time goes (OOT2_REPLAY_COUNTERS=1, group "frametime"): the numbers the recorder already fills for every
// frame it hands over (VkRecFrame, oot2/vk_rec.h) and the completion thread's effect time, printed for each window of
// presents as the count, sum, median, 90th percentile and largest of the frame's values (oot2/replay_count.h, RcSeries),
// in nanoseconds. A frame here is one submission of the replay; a present is made of several. Nothing here changes what
// the replay, the recorder or the timeline do: the feeds read a row that is final and add it to a series, behind the
// counters' own test of rc_on.
//
//   tm_gate_wait    waiting for the submission gate (the frame before not yet submitted)
//   tm_slot_wait    then waiting for the frame `frames` before it to be done (and for a virtual core again)
//   tm_ring_wait    the replaying thread waiting for room in the recorder's ring
//   tm_replay       the replaying thread's wall time for the frame, less the ring wait
//   tm_replay_cpu   that thread's CPU time over the same stretch
//   tm_record       the recorder thread, its frame-begin packet read to its submit made (wall time)
//   tm_record_cpu   the recorder thread's CPU time over the same stretch
// The completion thread's effect time (tm_effect, group "completion") is counted in gpu_sync.c, and the time the game's
// threads wait for the published value in vk_submit.c and gpu_sync.c (the "time" group's t_* counters).
// The sums over a window divided by the line's presents are the per-present times; the same sums against the interval
// between presents say which of them is the wait.
#include "oot2/replay_count.h"
#include "oot2/vk_rec.h"
#include "vk_replay_int.h"

RC_SERIES(s_gate, "tm_gate_wait_ns");
RC_SERIES(s_slot, "tm_slot_wait_ns");
RC_SERIES(s_ring, "tm_ring_wait_ns");
RC_SERIES(s_replay, "tm_replay_ns");
RC_SERIES(s_replay_cpu, "tm_replay_cpu_ns");
RC_SERIES(s_record, "tm_record_ns");
RC_SERIES(s_record_cpu, "tm_record_cpu_ns");

static void report(RcLine *l) {
    rc_group(l, "frametime");
    rc_series_put(l, &s_gate), rc_series_put(l, &s_slot), rc_series_put(l, &s_ring);
    rc_series_put(l, &s_replay), rc_series_put(l, &s_replay_cpu);
    rc_series_put(l, &s_record), rc_series_put(l, &s_record_cpu);
}

static void ensure_hook(void) {
    static int added;
    if (!__atomic_exchange_n(&added, 1, __ATOMIC_RELAXED)) rc_hook(report);
}

// The replaying thread, at the end of a frame (vkrec_end): the values that thread set.
void vkrec_count_replay(const VkRecFrame *row) {
    ensure_hook();
    rc_series_add(&s_gate, row->gate_wait_ns);
    rc_series_add(&s_slot, row->slot_wait_ns);
    rc_series_add(&s_ring, row->ring_wait_ns);
    rc_series_add(&s_replay, row->replay_ns);
    rc_series_add(&s_replay_cpu, row->replay_cpu_ns);
}

// The recorder thread, when it has submitted a frame.
void vkrec_count_record(uint64_t ns, uint64_t cpu_ns) {
    ensure_hook();
    rc_series_add(&s_record, ns);
    rc_series_add(&s_record_cpu, cpu_ns);
}
