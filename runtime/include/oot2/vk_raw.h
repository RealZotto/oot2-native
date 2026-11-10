// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The raw path of the Vulkan back end (architecture D15): the page table the translator's raw-access helper walks, built
// from the backing map's runs (oot2/backing.h, The raw path) in the layout of oot2/raw_table.h, and what a raw-access
// submission needs around it.
//
// The table. One leaf entry per 4 KiB page a run maps, (device address - guest address) | 1; every other entry 0, so an
// unmapped page reads zero, drops its write and takes a miss log entry. The span runs over every pool in the map (a
// backing is inside a pool, so the map cannot grow out of it). The table is built again, whole, when the map's version
// moved since the copy in use was built; a copy a submission in flight may read is never written, except in the one case
// below, so the table has a few copies (RAW_COPIES_MAX), each in device-local host-visible memory where the device has it
// (the raw-address study: there the walk ran at a direct pointer's speed), else in host-visible memory. Building again
// rather than editing in place is what keeps both of the Hookshot survey's checks true by construction: a backing that
// left the map leaves no entry, and one whose range a newer backing took leaves the newer entries alone. A build costs one
// write of the table (8 bytes per mapped page): see the raw page table study.
//
// A submission (gpu_vk_raw_begin, at its first raw-access stage; gpu_vk_raw_end at its end). The begin keeps the map's
// invariant (backing_raw_begin), takes a current copy and returns the push words 0 to 3 for every raw-access stage of
// the submission. The stages take the copy's address and span as they are recorded, but the device reads the table only
// after the submission is submitted, and the binds between the first stage and the end can move the map (an import over
// pages a bind commits, a mirror cut back for a bind that takes pages from it): so the end writes the map as it is
// then into the same copy, over the same span, when its version moved since the copy was built (refresh), after the
// earlier submissions that read the copy are done. Every raw access of the submission then reaches the map as the
// submission leaves it, whichever stage recorded it. The end records, in the submission's epilogue (after its own work),
// the runtime shader raw_writeback.comp, which copies the table's header and miss log out, zeroes the miss count, and
// for each mirrored page the table maps whose store bit is set clears the bit, notes the page in a bitmap and copies the
// page into the guest's own page through the page's sink; then queues an effect that, once the device reached the value
// and before it is done, logs the misses by address, kind and reason, and stamps the noted pages as written
// (oot2/pagegen.h). So a raw store into a mirror reaches the CPU when the value is done, as the descriptor path's writes
// do (backing.h, Mirrors), and the next submission's misses are its own. Store bits of pages that are not mirrored are
// left set: the helper reads a bit before it sets it, so a set bit costs it nothing.
//
// The sinks. The GPU writes the pages back itself: no CPU thread reads or copies their bytes (the copy on the completion
// thread was 14 ms of a present's 18 in the opening, and the value waits for it). A sink is a host import over committed
// guest pages (the backing map's import operation, the same device memory type as an import of the descriptor path), one
// for each run of committed pages under a mirror, cut at the import windows; the table's page list names, for each
// mirrored page, the device address of the guest's page through its sink. The sinks are the raw path's own, in vk_raw.c:
// they are made at a begin over the mirrors the map holds and when a table is built over a mirror, dropped when no mirror
// lies over them, and freed once the newest submission that may use them is done, so a page unmapped while a submission
// is in flight is written by the device into memory that stays valid until then. An import holds the pages it was made
// over: a page the guest decommits and commits again is another page, which its sink does not reach, so a sink one of
// whose pages a decommit took since the sink was made (oot2/memory.h, the decommit record) is dropped at the next begin,
// whether or not the page is committed again, and the pages under the mirror get a new sink; a decommit elsewhere, or a
// commit, drops none. A device that refuses a sink's import stops by name (as a refused import of the descriptor path
// does), unless the guest no longer has the pages committed when it is refused (it decommitted them on another thread
// while the sink was made): those pages get no sink, counted (sinks_refused) and logged.
// What the effect stamps is then only the pages that are still committed when it runs (the ones the guest unmapped are left
// out and counted, as before), through pagegen_mark_gpu_direct: a page the CPU wrote since every mirror was last brought
// current (the write watch says so, or its stamp is newer than the generation of that sync) keeps the plain stamp and the
// mirror takes it from guest memory at its next sync; the others are recorded GPU-current, which the mirror skips. The rule
// for a store that did not reach the guest's page: a committed page whose sink a decommit took it from since the sink was
// made (before the device wrote it, while it did, or while the effect runs), and a committed page that had no sink (the
// guest had not committed it when the table was built), do not hold the device's bytes. Their stores are lost: counted
// (writebacks_lost), logged with the submission and the first page, and the page gets the plain stamp, so the mirror takes
// the guest's bytes at its next sync; never the GPU-current stamp, which would leave the mirror holding a store the guest
// never sees. A mirrored page with no sink that the guest has not committed when the effect runs is left out and counted.
// While the table in use names a mirrored page with no sink, every begin looks again, so the page gets one at the first
// begin after the guest committed it.
//
// A CPU write into a page the GPU wrote in the same submission: the GPU's write is the whole page, made at the submission's
// end from the mirror, which holds the page as it was when it was brought current, so the CPU's write is under it when it
// came first. That window is the whole GPU run (the memcpy path's was the same, from the sync to the copy). Nothing here
// merges the two (a snapshot compare, as one of the other recompilations does for emitters writing beside the GPU): the
// contract is the one of the declared write-backs, that the CPU does not write what the GPU is writing. A CPU write after
// the GPU's page, before the stamp, is kept: the page keeps the plain stamp.
//
// One submission is recorded at a time (the submission gate of oot2/gpu.h), so begin and end need no lock of their own, and
// the sinks need none; the effect takes the out buffers' lock only.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/backing.h"
#include "oot2/raw_table.h"

#define RAW_COPIES_MAX 4             // copies of the table; a fifth with every copy in a submission not issued is fatal

// ---- the builder (no device: runtime/gpu/raw_table.c)
typedef struct RawLayout {
    uint32_t span_base, span_count;   // push words 2 and 3
    uint32_t blocks;                  // leaf blocks in use
    uint64_t words;                   // the table's size in 32-bit words
    uint64_t pages, mirror_pages;     // page entries the runs write; those of mirrors
} RawLayout;
// The layout of runs' table: 1, or 0 with the reason in why (more leaf blocks than RAW_LEAF_BLOCK_CAPACITY, a delta that
// is not a multiple of 4 KiB, a run that is not whole pages).
int raw_layout(const BackingRun *runs, int n, RawLayout *l, char *why, size_t why_size);
// The same over a span the caller chose, the chunks span_base to span_base + span_count - 1 of 2 MiB (a span_count of 0
// takes the span from the runs): a table rewritten later keeps the push words stages already took, so it keeps the span,
// and a run that reaches outside it is refused with its name.
int raw_layout_span(const BackingRun *runs, int n, uint32_t span_base, uint32_t span_count, RawLayout *l, char *why,
                    size_t why_size);
// A mirrored page as the write-back shader reads it (two entries of its uvec4 array): its leaf slot (its store bit) and the
// device address of its bytes in the mirror, and the device address of the guest's own page through its sink (0: none;
// the builder leaves it 0 and vk_raw.c names the sinks).
typedef struct RawMirrorPage {
    uint32_t slot, pad;
    uint64_t address;
    uint64_t sink, pad2;
} RawMirrorPage;
_Static_assert(sizeof(RawMirrorPage) == 32, "raw_writeback.comp reads a mirrored page as two uvec4");
// Write the table of runs, laid out as l: every one of its l->words words (unmapped entries and the logs zero), the
// mirrored pages into pages and the guest addresses their bytes stand for into guest (l->mirror_pages of each; both may
// be NULL when there are none). Runs are written in order: a later run's page replaces an earlier one's.
void raw_write(const BackingRun *runs, int n, const RawLayout *l, uint32_t *words, RawMirrorPage *pages, uint64_t *guest);

// ---- the device (runtime/gpu/vk_raw.c)
// Submission value holds a raw-access program: the push words 0 to 3 (RAW_PUSH_*) for it. The first call in a
// submission does the work; later ones return the same words.
void gpu_vk_raw_begin(uint64_t value, uint32_t push[4]);
// The end of submission value, on its thread before backing_submit_end and gpu_vk_end: nothing when it never began.
void gpu_vk_raw_end(uint64_t value);
// gpu_vk_close, once every submission is done: the copies, the sinks, the out buffers and the shader go, and the state
// between submissions starts over.
void gpu_vk_raw_close(void);
// Tests: the operations the sinks are imported and freed with, the device's (gpu_vk_backing_ops) unless set; NULL, and
// gpu_vk_raw_close, restore the device's. Set between submissions, with the device open.
void gpu_vk_raw_set_ops(const BackingOps *ops);

typedef struct RawMiss {
    uint64_t address;
    uint32_t kind, reason;   // RAW_MISS_KIND_*, RAW_MISS_REASON_*
} RawMiss;
typedef struct GpuVkRawStats {
    uint64_t submissions, builds, build_ns, build_ns_max;   // raw-access submissions; table builds and their time
    uint64_t refreshes, refresh_ns, refresh_waits;          // tables rewritten at a submission's end; their time; waits for readers
    uint64_t copies, copy_waits, table_bytes;               // copies made; builds that waited for one; the newest's size
    uint64_t pages, mirror_pages, blocks, span_count;        // the newest build's
    uint64_t writebacks, writeback_bytes, writebacks_skipped;   // raw stores into mirrors written back; pages left out
    uint64_t writebacks_lost;                               // pages whose stores did not reach the guest's page (The sinks)
    uint64_t writeback_cpu_pages;                           // of the written-back pages, those the CPU wrote since the
                                                            // mirrors were brought current (plain stamp: the mirror takes them)
    uint64_t ended_ns, ended_ns_max, mark_ns;               // the write-back effect on the completion thread: its time in all,
                                                            // the longest, and the part in pagegen_mark_gpu_direct
    uint64_t sinks, sink_bytes, sinks_made, sinks_dropped, sink_ns;   // live now (host imports under mirrors); made, dropped
                                                            // since the start, and the time making them took
    uint64_t sinks_dead, sinks_freed, sinks_refused;        // dropped and not freed yet (a submission that may use them
                                                            // is not done); freed; imports refused over pages decommitted
    uint64_t misses, miss_submissions;                      // misses counted by the helper; submissions with any
    uint64_t last_miss_value;                               // the newest submission with misses
    uint32_t last_misses;                                   // its entries in last_miss (at most RAW_MISS_ENTRIES)
    RawMiss last_miss[RAW_MISS_ENTRIES];
} GpuVkRawStats;
void gpu_vk_raw_stats(GpuVkRawStats *out);
