// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Which regions of a CPU-written texture to upload for a use now (architecture D19: rewritten storage is uploaded
// again). The decision reads the page generations of oot2/pagegen.h, the one owner of the write watch, and calls
// rt_mem_written itself never (a test reads the runtime for a second caller): the storage's pages are queried, which
// stamps what the CPU wrote since anyone last asked, and a region is stale when one of its pages carries a stamp newer
// than the one the texture was last current at.
//
// A texture's state is the generation it is current at (TexSync.gen) and whether its image holds defined texels at all.
// A texture never uploaded before gets every region: a region none of whose pages was ever stamped and none of whose
// pages is committed (nothing, the CPU and host I/O included, ever wrote there, which reads as zeros) is zeros, so that the
// pages nobody touched are not committed by reading them; the others are read from guest memory. Committed counts as
// written whatever the write watch saw, since host I/O into guest pages may not show in it (the mirror coherence study). A range the write watch does not cover (pagegen_scan answers UINT64_MAX: no write
// watch on this platform, or storage outside the watched regions) cannot say what changed, so every region is stale at
// every use and the use is counted.
//
// No Vulkan type, no guest memory read: the caller copies the regions this lists after the call returns, which is the
// order the generations ask for (a write that lands during the copy keeps its mark for the next query).
#pragma once
#include <stdint.h>

#include "oot2/nvn_obj.h"

typedef struct TexSync {
    uint64_t gen;    // the generation of the query the texture is current at
    int defined;     // the image holds a defined upload
} TexSync;

typedef struct TexRegion {
    uint32_t level, layer;
    int zero;        // no page of the region was ever written by the CPU: upload zeros
} TexRegion;

// What says which guest pages are committed: rt_mem_commit_run's signature. A test sets a stand-in; NULL restores it.
typedef uint64_t (*TexSyncCommit)(uint64_t addr, uint64_t size, int *committed);
void texsync_set_commit_probe(TexSyncCommit fn);

// The regions to upload for texture d at `storage` (its first byte in guest memory), updating st. A tiled texture gives
// one region per (level, layer) that is stale; a linear one a single region 0, 0 for its whole storage. Fills at most
// `max` entries of out and returns how many regions are stale (more than max: the caller asks again with room), or
// -1 with *why set for a texture this layer does not upload (untile_geom's reasons).
int texsync_regions(const nvn_texdesc *d, uint64_t storage, TexSync *st, TexRegion *out, int max, const char **why);

// The most regions a texture can have: its levels times its layers (a 3D level is one region).
unsigned texsync_max_regions(const nvn_texdesc *d);

typedef struct TexSyncStats {
    uint64_t queries, untracked;       // page queries made; uses of storage the write watch does not cover
    uint64_t regions, zero_regions;    // regions listed, and those that were zeros
    uint64_t unchanged;                // uses that found nothing stale
} TexSyncStats;
void texsync_stats(TexSyncStats *out);
int texsync_stats_try(TexSyncStats *out);   // 0 when the lock is busy (the counters' lines never wait for it)
void texsync_reset_stats(void);
