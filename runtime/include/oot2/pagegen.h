// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Page generations: the one owner of the write watch (architecture D14, D19; the write-watch study).
//
// rt_mem_written resets the operating system's mark of every page it reports (oot2/memory.h), so two callers would take
// each other's pages: what one query reported, the next one never sees. Nothing but runtime/gpu/pagegen.c calls it (a
// test reads the tree for a second caller). A query here stamps each page it reports with a new generation in one table
// that every consumer reads: the backing map's mirrors now; texture uploads, lever L1 and the raw path's write-back
// later. A consumer keeps the generation up to which its copy is current, and a page is stale for it when its stamp is
// newer, so two consumers of one range see the same pages whichever of them asked for the query.
//
// Generations grow by one per query and per mark, from 1 (0: never stamped), in 64 bits, which never wrap. A query that
// returns g has stamped, with g or earlier, every page written before it began; a page written while it ran keeps its
// mark for the next query. So a consumer that copies a range after a query that returned g holds every write stamped up
// to g, and a later stamp of a page it already copied is a harmless repeat.
//
// Cost (the write-watch study, spike S5, on the reference PC): a query costs about 1.3 ms per committed GiB of its range plus 0.1
// microsecond per page it reports, and the game then pays about 0.7 microsecond at its next write of each page the query
// reset; separate queries of separate ranges cost what one query of the same pages costs. So a consumer queries only the
// ranges it keeps a copy of, range by range, and never a whole region.
//
// Pages are 4 KiB (the write watch's granularity on the reference PC, and the page of commit). Ranges lie in the guest's
// address space (below RT_ADDRESS_LIMIT); a mark in the GPU window (gfx.h) is ignored, since no CPU sees it. Locking:
// one mutex over the table and every query; any thread.
#pragma once
#include <stdint.h>

#define PAGEGEN_PAGE 0x1000ull

// What gives the operating system's marks: rt_mem_written's signature. A test sets a stand-in (continuous integration's
// Linux runner has no write watch); NULL restores rt_mem_written.
typedef uint64_t (*PagegenSource)(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset);
void pagegen_set_source(PagegenSource src);

// Query [lo, hi) (whole pages; one region of the source) with reset, and stamp every page it reports with a new
// generation, which is returned. UINT64_MAX when the source does not watch the range (no write watch on this platform,
// or the range is not in the heap or alias region): nothing then says which of its pages changed.
uint64_t pagegen_scan(uint64_t lo, uint64_t hi);
// Stamp every page of [lo, hi) with a new generation without a query, and return it: the CPU said it wrote them (a
// flush of a mapped range, the runtime's own writes, a write-back of GPU results).
uint64_t pagegen_mark(uint64_t lo, uint64_t hi);
// A mark for a write-back of GPU results into guest memory, called right after the runtime copied them in, on the
// thread that copied them: every page [lo, hi) touches is stamped as pagegen_mark does, and every page it covers WHOLE is
// also recorded as GPU-current (pagegen_gpu_stamp equals its stamp) and has the write watch's mark of the copy taken
// away, so the next query does not report the runtime's own write as the CPU's. A mirror that holds the bytes the GPU
// wrote, which the copy put in guest memory, needs those pages no copy of guest memory (pagegen_runs_cpu skips them) until
// a later stamp (a CPU write, a mark) moves them past the record. The pages a bound covers in part keep the plain mark: a
// copy of them into a mirror stays. A CPU write into a page between the copy and this call is taken for the copy's own
// (the caller's contract: the CPU does not write bytes the GPU is writing).
uint64_t pagegen_mark_gpu(uint64_t lo, uint64_t hi);
uint64_t pagegen_gpu_stamp(uint64_t page);   // the stamp pagegen_mark_gpu gave the page (0: none)
uint64_t pagegen_now(void);   // the newest generation handed out (0 before the first)

// The runs of pages of [lo, hi) stamped after since, in address order: fn(ctx, run_lo, run_hi) for each run of
// consecutive such pages. Returns the number of pages.
typedef void (*PagegenRun)(void *ctx, uint64_t lo, uint64_t hi);
uint64_t pagegen_runs(uint64_t lo, uint64_t hi, uint64_t since, PagegenRun fn, void *ctx);
// The same, less the pages whose newest stamp is a pagegen_mark_gpu's: what the CPU wrote (or marked) since.
uint64_t pagegen_runs_cpu(uint64_t lo, uint64_t hi, uint64_t since, PagegenRun fn, void *ctx);
uint64_t pagegen_stamp(uint64_t page);   // the stamp of one page (0: never stamped)

typedef struct PagegenStats {
    uint64_t generation;                    // pagegen_now
    uint64_t scans, scan_bytes, scan_ns;    // queries, the bytes they covered and their time
    uint64_t scan_pages;                    // pages the queries reported
    uint64_t untracked;                     // queries the source did not watch
    uint64_t marks, mark_pages;             // marks and their pages
    uint64_t chunks;                        // 256 MiB pieces of the table made
    uint64_t gpu_marks, gpu_mark_pages;     // write-back marks (pagegen_mark_gpu) and the pages they recorded GPU-current
} PagegenStats;
void pagegen_stats(PagegenStats *out);
// Forget every stamp and counter, and restore rt_mem_written as the source (tests, between cases of one process).
void pagegen_reset(void);
