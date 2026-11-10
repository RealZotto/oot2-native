// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The backing map of the Vulkan back end (architecture D14): what memory the GPU sees behind every GPU address the game
// can bind, decided per page, and the resolver that turns a bound range into a buffer and an offset for the descriptor
// path (vertex, index, uniform, storage and indirect bindings, copies). The raw path (D15) reads the same decisions.
// Nothing here has a Vulkan type: memory comes from a BackingOps table (runtime/gpu/vk_backing.c on a device, a test's
// stand-ins without one), and its lifetime follows the timeline of oot2/gpu.h.
//
// Pools are named by their GPU range, as gfx.h names them. A page of a pool is, from the first bind that reaches it:
//   extent   VRAM. A pool the CPU cannot see (CPU_NO_ACCESS, physical) gets one extent over the whole pool at its first
//            bind, so no bind inside the pool spans two extents; a pool no bind and no image reaches gets none.
//            An image placed over a pool's storage (render targets and GPU-written textures, vk_image.c) resolves
//            the same way, so it makes the extent. Up to BACKING_BLOCK bytes it is a part of a VRAM block; past that,
//            a block of its own.
//   import   host memory imported over committed guest pages: a pool the CPU can see whose mirror the rule refused.
//   mirror   VRAM kept current from the guest's pages, in the place an import would take (Mirrors below).
//   none     before that first bind, and for pages of a pool the CPU can see that no import or mirror covers.
// A page of a virtual pool is the page of the physical pool its one run maps it to (the front end allows one run per
// virtual pool, the device and pool study); a bind over an unmapped page is a named fatal.
//
// The page is 4 KiB in a pool the CPU can see: the page of commit and of host import, and NVN aligns pool storage to
// 4 KiB only, so one 64 KiB page can hold two pools. In the GPU address window (gfx.h) it is 64 KiB.
//
// Imports. A host import covers committed pages only: one over an uncommitted page fails, and after a failed import the
// driver refuses every import of that size or larger until a smaller one succeeds (the raw-address study, the write-watch study).
// Imports are cut to windows of BACKING_IMPORT_WINDOW aligned in the address space, and on Windows the guest's watched
// regions are reserved in pieces of that size (runtime/core/plat.c), so an import walks no further than its window
// (the write-watch study: the driver walks the run of committed pages that follows the pointer, to the reservation's end).
// A cell is a pool cut to the windows one bind touches: one window, or two for a bind across a boundary. The bind's own
// pages are committed first, as host I/O commits its target (rt_mem_commit_io), then the cell's import covers the run of
// committed pages around the bind inside the cell (rt_mem_commit_run). A later bind the import does not cover replaces it
// with the larger run. Imports of one page alias the same host memory, so a page in two cells reads the same bytes
// through both.
//
// Textures. A decision is taken at a page's first bind, never at pool_init or texture_register: a pool that only holds
// textures is never bound as a buffer and is never imported (D14, D19). A texture registered over pages a bind already
// imported leaves them imported, and those pages cannot be released by lever L1 while the import holds them
// (backing_page tells the texture ticket which pages are imported).
//
// Retirement. A resolve stamps its pool and the backing it used with the submission's value. A backing that leaves the
// map (replaced by a larger import, over pages the guest decommitted, or its pool finalized) waits in a deferred-release
// queue until the newest value stamped on it is done; the queue holds BACKING_DEFERRED_MAX backings and a full queue
// waits for its oldest. pool_fini returns once every value stamped on its pool is done (gfx.h): the front end hands the
// range to the next pool at once. A pool no submission used returns at once.
//
// The commit epoch (oot2/memory.h). backing_sync, called before a submission, checks every cell again when the epoch
// moved since its last call: an import grows over chunks committed since inside its cell (D15's invariant: every
// committed page a pointer can reach is mapped before a raw-access submission), and an import over pages decommitted
// since is retired. It also frees what retired.
//
// VRAM blocks are made off the frame path: a thread of this module keeps BACKING_SPARE_BLOCKS empty blocks ready, so an
// extent takes its memory from a block made earlier (vkAllocateMemory took 8 to 19 ms per block in DKB Recomp,
// runtime/vk_headroom.c, whose idea this follows; no code is taken from it). A block made on the resolving thread
// because none was ready, and every block of its own, is counted and logged.
//
// Mirrors (D14, D17a, section 8). A bind in a pool the CPU can see asks the mirror hook (the Vulkan back end's says yes
// on a discrete device, where imported memory is read across the bus) and then the admission rule: the mirror is made
// when the device-local heap's use plus the mirror's allocation plus an eighth of its budget fits the budget, the
// figures of VK_EXT_memory_budget (BackingOps.vram), and the write watch covers its range; otherwise the bind gets an
// import. A mirror is VRAM from the blocks, over the range the import would cover (the run of committed pages around
// the bind inside its cell), and needs no import beside it. Its allocation reserves BACKING_MIRROR_GUARD bytes at
// each end so it can grow by one aligned block part without moving already-recorded bindings; those reserved bytes
// count against the admission budget but do not add pages to the mirror's raw write-back:
//   CPU to mirror   the first resolve of a mirror in a submission brings it current before that submission's own work:
//                   it queries the mirror's range through the page generations (oot2/pagegen.h, the one owner of the
//                   write watch), and copies the pages stamped after the mirror's generation (all of them the first
//                   time) from guest memory into a staging copy taken then, which the GPU copies in the submission's
//                   prologue, after every earlier submission's work and before its own. So a CPU write before a submit
//                   is read by that submission, as on the console, whichever submission of the frame it is.
//   GPU to guest    a GPU write into a mirror is declared by whoever records it (backing_gpu_wrote: the replay's
//                   storage writes, copies and fills); at the submission's end its bytes are copied out to staging, and
//                   once the device reached the submission's value, on the completion thread and before the value is
//                   done, they are written into guest memory. Never before: a CPU that waited for the fence reads them,
//                   one that did not reads the old bytes, as on the console.
//   both            a copy into a mirror leaves out the bytes of GPU writes not yet written back, so a page the CPU and
//                   the GPU both write keeps the GPU's bytes (DKB Recomp's shadow-storage rule).
// A new backing over a run that holds older ones (a bind that grew the run) waits for the older submissions that used
// them, so their GPU writes are in guest memory before the new backing takes its copy; one written by the submission
// being recorded is a named fatal. A mirror over pages the guest decommitted is retired at the next sync, as an import is.
// Lever L1 must not release pages a mirror covers (backing_page), as it must not release imported ones.
//
// The raw path (D15, runtime/gpu/vk_raw.c). A raw-access submission reaches memory through a page table rather than
// through resolves, so the map answers it in three calls: backing_raw_begin keeps D15's invariant (every committed page
// a pointer can reach is mapped before the submission: each pool the CPU cannot see gets its extent, and every run of
// committed pages of a pool it can see that no backing covers gets an import, cut to the import windows; a pass runs
// when the commit epoch moved or a pool was made since the last one, so a chunk committed after its pool was mapped is
// mapped at the next raw-access submission), brings every mirror current (backing_sync_mirrors) and records the
// submission as the newest raw-access one; backing_version and backing_raw_runs give the table's builder the map as runs.
// A raw-access submission can reach every backing, so a backing that leaves the map waits for the newest one as well as
// for its own resolves (the backing map study: one value, not a stamp on every backing), and so does pool_fini. Raw stores into
// a mirror are known only from the table's store bits, after the work: the raw path writes those pages back at the
// submission's end, before its value is done, as the descriptor path's writes are. A copy into a mirror the CPU wrote
// while a raw-access submission is in flight waits for it first, so its write-back does not meet a newer copy.
// The table of a submission is read after the submission is recorded, and vk_raw.c rewrites it in place as the map moves
// until then, so every raw access of the submission reaches the map as the submission leaves it. A mirror is a copy of its
// pages, so a bind in such a submission never replaces one by a larger backing (two copies of the same bytes, one for what
// was recorded before and one for what follows): it makes an import, never a mirror, and takes the pages it needs from the
// mirrors it overlaps, which keep the part outside the bind (or go, when the bind covers all of them), waiting first for
// the older raw-access submissions that may have stored into them; the import covers the committed run around the bind up
// to the mirrors that remain. A mirror already resolved where the bind takes pages can instead grow in place, over its
// own pages and the bind's only (an import it overlaps that the submission did not bind goes, and its pages outside the
// bind get an import again), when the growth adds at most BACKING_MIRROR_GROW_MAX bytes, its enlarged range stays
// write-watched and fits in the block, and no other mirror, import the submission bound, or backing of another cell lies
// on the added pages; otherwise it is a named fatal that says which of these held. The mirrors' total is not checked
// there: the raw path's write-back reaches every mirrored page (vk_raw.h, The sinks), whatever the total.
//
// An import is never what the rule would give a bind on a discrete device, and the invariant above makes one of every
// committed page: the GPU reads an import across the bus, so a compute program that works on its buffers stalls on every
// access (the raw-address study measured 22.9 times VRAM for a lookup; a title run moved 3 GB a second each way over the
// bus at a few percent of the GPU's own memory traffic). A bind of a range larger than a uniform's (BACKING_PROMOTE_MIN)
// that lands in an import marks it hot, and the next raw-access submission's backing_raw_begin gives each hot import the
// mirror the rule would have made, over the same pages, before the submission binds anything of its own: the older
// submissions that reached the import are waited for, the mirror is brought current from guest memory, and its GPU writes
// are written back like any mirror's (the page table's store bits for raw stores, the declared ranges for binds). An
// import the submission wrote stays one until the next, and one the admission refuses stays an import. So does one whose
// mirror would take the mirrors past BACKING_PROMOTE_MIRROR_MAX: the raw path writes back every page its stores reach
// across the bus at the end of the submission and keeps a host import (a sink, vk_raw.h) over each mirror, so what a
// promotion makes is bounded.
//
// Locking: one reader-writer lock. A resolve that finds its backing holds it shared; one that makes a backing holds it
// exclusively, and imports are made under it. The mirrors' coherence state (their generations, the GPU writes not yet
// written back) has a mutex of its own, taken after that lock and never by a thread that waits for a value; the
// completion thread takes only it. Every call may come from any thread.
#pragma once
#include <stdint.h>

#include "oot2/gfx.h"

// ---- the rule (one function, its constants named: spike S5 and the raw-address study move them)
#define BACKING_IMPORT_WINDOW (64ull << 20)   // the write-watch study: a bounded 64 MiB import costs 0.8 ms to make, 1.6 ms to free
#define BACKING_IMPORT_PROBE 0x1000ull        // the write-watch study: the small import that clears a failed one
#define BACKING_BLOCK (256ull << 20)          // a VRAM block; an extent larger than this has a block of its own
#define BACKING_SPARE_BLOCKS 1                // empty blocks kept ready by the block thread
#define BACKING_DEFERRED_MAX 1024             // backings waiting for retirement; a full queue waits for its oldest
#define BACKING_EXTENT_ALIGN 0x10000ull       // extents are 64 KiB aligned in their block, as window pages are
#define BACKING_HOST_PAGE 0x1000ull           // the page of commit and of host import
#define BACKING_MIRROR_GUARD BACKING_EXTENT_ALIGN   // reserved at either end for in-place growth
#define BACKING_MIRROR_GROW_MAX (1ull << 20)  // the most one in-place growth of a bound mirror adds (a few pages, not a new mirror)
#define BACKING_MIRROR_RESERVE 8              // section 8: a mirror leaves an eighth of the VRAM budget free
#define BACKING_PROMOTE_MIN (64ull << 10)     // a bind larger than a uniform's (maxUniformBufferRange) marks its import hot
#define BACKING_PROMOTE_MIRROR_MAX (128ull << 20)   // a promotion leaves the mirrors at or under this many bytes (the raw write-back's traffic and sinks, vk_raw.h)
enum { BACKING_NONE, BACKING_EXTENT, BACKING_IMPORT, BACKING_MIRROR };

// What a bind asks for: its range and its pool's flags (NVNmemoryPoolFlags).
typedef struct BackingAsk {
    uint64_t addr, size;
    uint32_t pool_flags;
} BackingAsk;
// BACKING_EXTENT for a pool the CPU cannot see; for one it can, BACKING_MIRROR when the mirror hook says so, else
// BACKING_IMPORT. A mirror the admission refuses is an import after all (backing_resolve). No hook: imports.
int backing_rule(const BackingAsk *ask);
void backing_set_mirror_hook(int (*want_mirror)(const BackingAsk *ask));
// The admission (section 8): 1 when used + size + budget / BACKING_MIRROR_RESERVE <= budget.
int backing_mirror_fits(uint64_t used, uint64_t budget, uint64_t size);

// ---- the memory behind backings
typedef struct BackingMem {
    uint64_t buffer;    // the buffer over the whole allocation (a VkBuffer handle on a device)
    uint64_t memory;    // the allocation (a VkDeviceMemory handle on a device); the ops' own
    uint64_t address;   // the buffer's device address
    uint64_t size;
    uint32_t type;      // the allocation's memory type index (a block's, for an image placed in it)
} BackingMem;
// A copy between host memory and backing memory: size bytes at offset in mem, and host's bytes (an upload) or the
// place they are read from (a readback).
typedef struct BackingCopy {
    const BackingMem *mem;
    uint64_t offset, size;
    const void *host;
} BackingCopy;
typedef struct BackingOps {
    // A VRAM block of size bytes with one buffer over it: 1, or 0 when the device has no room (a named fatal follows).
    int (*block_new)(uint64_t size, BackingMem *out);
    // Committed host pages [host, host + size), 4 KiB aligned, imported: 1, or 0 when the device refused the import.
    int (*import)(uint64_t host, uint64_t size, BackingMem *out);
    // Free an allocation no submission can use any more.
    void (*free)(const BackingMem *m);
    // ---- mirrors: all four, or none (a map without mirrors; a hook that asks for one then stops by name)
    // The heap blocks come from, as VK_EXT_memory_budget tells it: 1 with its use and its budget in bytes.
    int (*vram)(uint64_t *used, uint64_t *budget);
    // In the submission `value` being recorded, on its thread: copy each host range into its backing memory in the
    // submission's prologue, after every command of earlier submissions and before every command of this one. The host
    // bytes are taken before the call returns.
    void (*upload)(uint64_t value, const BackingCopy *c, int n);
    // Likewise, at the end of the submission, after every command it recorded: copy each backing range out, and set
    // c[i].host to where its bytes can be read once value is done (and until a later submission's value is).
    void (*readback)(uint64_t value, BackingCopy *c, int n);
    // backing_close, once every backing is freed: the memory uploads and readbacks used goes.
    void (*close)(void);
} BackingOps;

// Open the map over ops (the Vulkan back end at gpu_vk_open, after its timeline started; a test with stand-ins), and
// start the block thread. Close frees every backing and block once every submission is done (gpu_sync_finish first),
// and stops the thread; backing_open may follow. Neither may run while another thread uses the map.
void backing_open(const BackingOps *ops);
void backing_close(void);
int backing_is_open(void);

// ---- the back end's pool operations (gfx.h's contract)
int backing_pool_init(const pool_t *p);
void backing_pool_fini(const pool_t *p);
int backing_pool_map_virtual(const pool_t *v, int n, const nvn_mapping_t *requests);

// ---- the resolver
// The backing of [addr, addr + size) (size 0 counts as 1) for the submission with the given value (gpu_sync_begin's;
// 0 for none): its buffer, the offset of addr in it and addr's device address. A range that is not wholly inside one
// pool, or that would need two backings (pages of a virtual pool in no run), is a named fatal that prints its shape.
// A resolve with a value is made in that submission, on its thread (between gpu_sync_begin and gpu_sync_end): the first
// one of a mirror brings it current for the submission (Mirrors above). Value 0 only reads the map.
typedef struct BackingRef {
    uint64_t buffer, offset, address;
    uint64_t memory;   // the allocation behind the buffer: an image placed at `offset` binds to it (vk_image.c)
    uint32_t type;     // its memory type index
    int kind;   // BACKING_EXTENT, BACKING_IMPORT or BACKING_MIRROR
} BackingRef;
void backing_resolve(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out);
// Before a submission: grow and retire imports after the commit epoch moved, retire mirrors over pages decommitted since,
// and free what retired.
void backing_sync(void);

// ---- coherence (Mirrors above), in the submission `value` being recorded, on its thread
// The submission writes [addr, addr + size) on the GPU, through the backing a resolve of it in this submission gave: in
// a mirror the bytes go back to guest memory once the device reached value; in an import they are there already.
// Every GPU write of a replayed submission into a CPU-visible pool is declared, or the CPU never sees it.
void backing_gpu_wrote(uint64_t addr, uint64_t size, uint64_t value);
// Every live mirror current for the submission, as its first resolve would make it: for work that reaches memory
// without a resolve (the raw path, D15).
void backing_sync_mirrors(uint64_t value);
// The submission's end, before gpu_sync_end: the readbacks of what it declared written, and their write-back.
void backing_submit_end(uint64_t value);

// ---- the CPU hooks of gfx.h, as the Vulkan back end answers them
// guest_written: the CPU wrote [addr, addr + size) (a flush, the front end's own writes): the pages are stamped, so the
// mirrors over them are brought current at their next use.
void backing_guest_written(uint64_t addr, uint64_t size);
// guest_read and cpu_synced: what the GPU wrote into mirrors is in guest memory by the time a value is done, so these
// check it: a write-back still waiting for a value that is done is a named fatal. A read of bytes a GPU write still in
// flight will replace is counted (the game reads before its fence; the console gives it the old bytes too).
void backing_guest_read(uint64_t addr, uint64_t size);
void backing_cpu_synced(uint64_t fence);

// ---- the raw path (The raw path above)
// Submission `value` holds a raw-access program: keep the invariant, bring the mirrors current and record it. On the
// submission's thread, before the table is read.
void backing_raw_begin(uint64_t value);
// The page generation up to which every live mirror is current (the lowest of their syncs; 0 for one never brought
// current), UINT64_MAX when there is none: a page stamped at or before it by the CPU is in its mirror, one stamped after it
// may not be (the raw path's write-back, oot2/pagegen.h, pagegen_mark_gpu_direct). Read after backing_raw_begin.
uint64_t backing_synced_min(void);
// The map's version: it changes whenever a backing enters or leaves the map or a virtual pool's pages move.
uint64_t backing_version(void);
// The lowest address and the highest end of the pools in the map: 1, or 0 with none. A backing is inside a pool, so a table
// whose span covers this range can take every backing the map makes later.
int backing_pool_span(uint64_t *lo, uint64_t *hi);
// A run of GPU addresses mapped to consecutive device memory: [lo, hi) at device address `address`, whose bytes stand for
// guest memory from `guest` on (lo, except through a virtual pool: the physical pool's address), of one kind.
typedef struct BackingRun {
    uint64_t lo, hi, address, guest;
    int kind;
} BackingRun;
// Every mapped run, a mirror's after an import's where a page is in both (the table's builder writes them in order); the
// list is malloc'd into *out and its count returned, with the version it was read at.
int backing_raw_runs(BackingRun **out, uint64_t *version);

// ---- the decision of one page, for tests and for the raw path's page table: the kind, and when it has a backing, the
// device address of addr and, for a page of a virtual pool, the physical GPU address it maps to (else addr).
typedef struct BackingPage {
    int kind;
    uint64_t address, gpu;
} BackingPage;
void backing_page(uint64_t addr, BackingPage *out);

// 0 when [addr, addr + size) stands in memory the CPU can see: a pool the CPU sees, or a virtual pool mapped over one. The
// map keeps that memory as a mirror (a VRAM copy of the CPU's pages) or an import, never as an extent, so an image must
// not be bound into it (oot2/vk_image.h). 1 for a pool of extents, a virtual pool mapped over one, and for what the map
// cannot say (a range in no pool, a virtual pool not mapped there): the resolve that follows names those.
int backing_extent_pool(uint64_t addr, uint64_t size);

typedef struct BackingStats {
    uint64_t pools, extents, extent_bytes, imports, import_bytes;     // live now
    uint64_t blocks, block_bytes, spare_blocks;                       // VRAM blocks live now, empty ones ready
    uint64_t imports_made, imports_refused, imports_retried, imports_grown, imports_stale;
    uint64_t blocks_made_ahead, blocks_made_inline, blocks_dedicated;  // by the block thread; on a resolving thread
    uint64_t deferred, deferred_peak, deferred_full_waits, freed;      // the deferred-release queue
    uint64_t fini_waits, resolves, resolve_misses, syncs, sync_rescans;
    uint64_t mirrors, mirror_bytes;                                    // live now
    uint64_t mirrors_made, mirrors_refused_budget, mirrors_refused_untracked, mirrors_stale, replace_waits;
    uint64_t mirror_syncs, uploads, upload_bytes;                      // syncs, and the copies into mirrors they made
    uint64_t writebacks, writeback_bytes, writebacks_skipped;          // GPU writes written back; into unmapped memory
    uint64_t pending, reads_in_flight;                                 // GPU writes not written back yet; early reads
    uint64_t raw_begins, raw_passes;                                   // raw-access submissions; passes of the invariant
    uint64_t raw_fill_imports, raw_fill_bytes, raw_fill_extents;       // what those passes made
    uint64_t raw_waits;                                                // mirror copies that waited for raw stores
    uint64_t raw_promoted, raw_promote_bytes;                          // imports given a mirror at a raw-access begin
    uint64_t raw_straddle_stops;                                       // raw-access stops whose dump was printed (beside)
    // The candidate growth of a mirror a raw-access bind cut, evaluated and never done (nothing changes): binds that cut a
    // mirror the submission had not resolved there, those where growing it would have passed every guard, those where it
    // would not, and the refusals by name in the order rule's kind, G1, G2, G3, G4, G5, G6, G7a, G7b, G8 (never evaluated:
    // its query would reset the write watch's marks), G9, the cap on the mirrors' total (one bind can count in several).
    uint64_t raw_dry_evals, raw_dry_would_grow, raw_dry_refused, raw_dry_g[12];
    uint64_t raw_dry_mutated;   // evaluations after which the map's digest differed: always 0
} BackingStats;
void backing_stats(BackingStats *out);
