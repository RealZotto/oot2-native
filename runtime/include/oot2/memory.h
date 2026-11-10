// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The guest address space (architecture D6). Guest addresses are host addresses, so every region the guest sees is a
// reservation at the address the guest uses, and pages are committed on first touch: private commit then tracks the
// pages the game touches, not the sizes it declares.
//
// Regions, all below RT_ADDRESS_LIMIT:
//   stack, heap, alias,  one block found at start-up from RT_REGION_SEARCH_FROM, below the modules
//   kernel
//   modules              [RT_MODULE_BASE, RT_MODULE_LIMIT), placed by tools/layout.py (loader.h)
//   host functions       RT_HOSTFN_BASE (runtime.h), reserved and never mapped
// Everything else below RT_ADDRESS_LIMIT belongs to the host, and QueryMemory reports it as inaccessible. The kernel
// region holds what the kernel gives each thread in guest memory (its thread-local region, TPIDRRO_EL0) and the user
// exception context; the kernel allocates and commits those pages itself (runtime/core/kernel.c).
//
// A first touch of a page inside a range the guest's SVCs declared mapped (the heap below its current size, alias
// pages MapPhysicalMemory mapped, stacks MapMemory moved, the main thread's stack, module bss) commits the 64 KiB chunk
// around it, clipped to that range, and the access runs again. A touch anywhere else under RT_ADDRESS_LIMIT that is not
// host memory is the named fatal "guest memory: ..." (exit RT_EXIT_FATAL), never a silent commit.
//
// Start-up order: rt_mem_init (registers the memory SVCs), rt_load_modules, then rt_dispatch_init.
#pragma once
#include <stdint.h>

#include "runtime.h"

#define RT_MODULE_BASE 0x7000000000ull         // tools/layout.py FIRST_BASE
#define RT_REGION_SEARCH_FROM 0x1000000000ull  // 64 GiB: above where the host puts its own heaps and stacks
#define RT_COMMIT_CHUNK 0x10000ull             // what a first touch commits
#define RT_KERNEL_REGION_SIZE 0x4000000ull     // 64 MiB: thread-local regions of a few thousand threads

// Region sizes the program runs with. GetInfo reports them as they are, so the game sizes its heap as on the console.
typedef struct RtMemConfig {
    uint64_t pool_size;              // TotalMemorySize (GetInfo 6, 21): the application's memory pool
    uint64_t heap_region_size;       // HeapRegionSize (GetInfo 5): the largest heap SetHeapSize can set
    uint64_t alias_region_size;      // AliasRegionSize (GetInfo 3): where MapPhysicalMemory maps
    uint64_t stack_region_size;      // StackRegionSize (GetInfo 15): where MapMemory moves thread stacks
    uint64_t system_resource_size;   // TotalSystemResourceSize (GetInfo 16), from the program's main.npdm
} RtMemConfig;

// The region sizes of Horizon's 39-bit address space (heap 8 GiB, alias 64 GiB, stack 2 GiB) and an application pool
// of 9 GiB. The pool is a planning figure, not a value read from a console: the build glue sets the program's own.
RtMemConfig rt_mem_default_config(void);

// Reserve every region, install the first-touch handler, register the memory SVCs (SetHeapSize,
// SetMemoryPermission, SetMemoryAttribute, MapMemory, UnmapMemory, QueryMemory, MapPhysicalMemory,
// UnmapPhysicalMemory). Fatal when a fixed region is taken by the host or the configuration is malformed.
void rt_mem_init(const RtMemConfig *cfg);
int rt_mem_ready(void);

// Host I/O into guest memory: a system call that writes into (or reads from) a page that is reserved but not
// committed fails instead of faulting, so commit the range first. 1 when all of [addr, addr+size) is guest memory the
// access may use (write: writable) and is now committed; 0 otherwise, and the caller returns an error to the guest.
int rt_mem_commit_io(uint64_t addr, uint64_t size, int write);

// GetInfo's memory ids (2 to 7, 12 to 17, 21, 22, 28): 1 and the value, or 0 for an id that is not about memory. The
// GetInfo SVC belongs to the kernel module, which asks here.
int rt_mem_info(uint32_t id, uint64_t *out);

// The guest's QueryMemory answer for addr (also what the SVC writes).
typedef struct RtMemInfo {
    uint64_t base, size;
    uint32_t state, attr, perm;
} RtMemInfo;
RtMemInfo rt_mem_query(uint64_t addr);

// Pages of the heap or alias region written since the last call with reset (the GPU back end uploads them again,
// D14): addresses to pages, at most max; the count is returned. UINT64_MAX when [addr, addr+size) is not inside one of
// those two regions or the platform has no write watch. A call with reset takes the pages it reports from every later
// caller, so the runtime has one caller, the page generations (oot2/pagegen.h), which every consumer reads instead.
uint64_t rt_mem_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset);

// The commit state of guest pages, for the GPU back end's host imports (D14): an import over a page that is not
// committed fails, and after a failed import the driver refuses every import of that size or larger for a while
// (the raw-address study, the write-watch study). rt_mem_commit_run gives the length in bytes, from addr, of the run of pages
// in one state that holds addr, up to addr + size and up to the end of addr's region, with *committed set to 1 for
// committed pages and 0 for pages that are not; a caller walks a range run by run. UINT64_MAX when addr lies in no
// region (stack, heap, alias, modules, kernel); 0 for size 0. It reads the page map without the lock, so its answer is
// a snapshot: the commit epoch grows by one after each change of the page map (a first touch, host I/O, SetHeapSize,
// MapMemory and UnmapMemory, UnmapPhysicalMemory, a page released or committed again by lever L1), so a reader that
// read the epoch, then the map, and later reads a larger epoch knows the map may have changed since its reading.
uint64_t rt_mem_commit_run(uint64_t addr, uint64_t size, int *committed);
uint64_t rt_mem_commit_epoch(void);
// The decommit record, for host imports that outlive a look at the page map (the raw path's sinks, oot2/vk_raw.h): an import
// holds the pages it was made over, so a page the guest decommits and commits again after the import is another page, which
// the import does not reach, though rt_mem_commit_run says committed. rt_mem_decommit_epoch counts the decommits of
// committed pages (a heap that shrinks, MapMemory's source and UnmapMemory's stack, UnmapPhysicalMemory, a kernel page
// freed, a page released by lever L1) and moves on nothing else, a commit included. rt_mem_decommitted(since, lo, hi) is 1
// when one of the decommits after epoch `since` took a page of [lo, hi), and also when the record no longer holds all of
// them (it keeps the last 1024, so a reader more than that behind is told 1); 0 means no page of [lo, hi) was decommitted
// since. A reader takes the epoch before it makes its import, and asks with it later. No lock; any thread. A decommit is
// recorded under the memory lock, before any commit of its pages again can take that lock.
uint64_t rt_mem_decommit_epoch(void);
int rt_mem_decommitted(uint64_t since, uint64_t lo, uint64_t hi);

// Measurement, and spike S7's first instrument: per region, what the guest declared mapped, what the runtime
// committed, and how much of that is resident now.
enum { RT_MEM_STACK, RT_MEM_HEAP, RT_MEM_ALIAS, RT_MEM_MODULES, RT_MEM_KERNEL, RT_MEM_REGIONS };
typedef struct RtMemRegion {
    const char *name;
    uint64_t base, size;      // the reservation
    uint64_t mapped;          // declared mapped: heap size, alias pages, moved stacks, module images
    uint64_t committed;       // committed now
    uint64_t resident;        // committed and in the working set now
    uint64_t touch_commits;   // chunks committed by a first touch since start-up
} RtMemRegion;
typedef struct RtMemStats {
    RtMemRegion r[RT_MEM_REGIONS];
    uint64_t process_private, process_working_set, process_peak_working_set;
} RtMemStats;
void rt_mem_measure(RtMemStats *out);

// ---- Page ranges: counting and releasing (lever L1, architecture D19 and D27). Off unless the setting page_release
// (oot2/settings.h; OOT2_PAGE_RELEASE in the environment) is count (counting ranges only) or release (both modes), read
// once by rt_mem_init; a default run keeps first-touch commit as it is, and rt_pages_watch returns -1 for every request.
//
// A range lies in the heap or alias region, in whole 4 KiB pages; the owner (runtime/nvn for texture pages) chooses the
// pages and what they mean, this file knows no textures. Pages are 4 KiB inside the 64 KiB first-touch chunks: an armed
// or released page is never taken for a first touch, and a first touch beside it commits the rest of its chunk only.
//   Counting:  rt_pages_arm takes the committed pages' access away. The first access to an armed page, read or write,
//              by guest code or by the host, is counted, the page gets its access back and the access runs again.
//   Releasing: rt_pages_release decommits the committed pages of a sub-range, unless the owner's busy callback says the
//              range is in use (nothing is released then). A write to a released page commits it again, zero-filled,
//              counts it and calls the owner's written callback; a read is the named fault "guest memory: ... a page of
//              range "<name>" was released and a read of it is a fault; read by guest code in <fn>" (or "by runtime or
//              back-end code"), counted before the process stops. A read-modify-write instruction is a write to the CPU:
//              it recommits and reads zeros. Host I/O (rt_mem_commit_io) is a host access: a write recommits, a read of
//              a released page returns 0.
// Who read: a fault whose host pc lies in a lifted function (the function table of rt_dispatch_init; the extent of a
// function is up to the next function start in host address order, or its module's end marker) is guest code; any other is runtime
// or back-end code. Locking: busy runs under the runtime's exclusive memory lock and must not call rt_pages_* or touch
// guest memory; written runs after the lock is dropped, on the writing thread (possibly in a signal handler on POSIX).
// The write watch (rt_mem_written, Windows): a release clears the page's mark, so query a page through the page
// generations (pagegen_scan) before releasing it; the write that recommits a page puts its mark back. The pages of a range that the guest unmaps
// or shrinks away lose their state with their memory.
#define RT_MAX_PAGE_RANGES 64
enum { RT_PAGES_OFF = 0, RT_PAGES_COUNTING = 1, RT_PAGES_RELEASING = 2 };
typedef struct RtPageOwner {
    const char *name;   // for the fault message and the census
    void *user;
    int (*busy)(void *user, uint64_t addr, uint64_t size);       // may be NULL; nonzero: the owner needs this range now
    void (*written)(void *user, uint64_t page_addr);             // may be NULL; a write recommitted the page
    // May be NULL. Called once for each access this file counts on a page of the range, after the lock is dropped, on the
    // accessing thread (possibly in an exception or signal handler): an access to an armed page (read or write, which
    // restores it), a read of a released page (just before the named fault), a write to a released page (beside written).
    // guest is 1 when guest code made it, with guest_fn the lifted function's address; host reads and writes (the runtime,
    // the back end, host I/O) have guest 0. It must not call rt_pages_* or touch guest memory.
    void (*touched)(void *user, uint64_t page_addr, int write, int guest, uint64_t guest_fn);
} RtPageOwner;
typedef struct RtPageStats {
    int mode;
    uint64_t armed, released;              // pages in that state now
    uint64_t arms, releases;               // pages armed and released since the range was made
    uint64_t guest_reads, host_reads;      // counting: reads that hit an armed page; releasing: reads of released pages
    uint64_t guest_writes, host_writes;    // counting: writes that hit an armed page; releasing: writes that recommitted
    uint64_t recommits, busy_refusals;     // pages committed again by a write; releases the owner refused
} RtPageStats;
int rt_pages_level(void);   // RT_PAGES_OFF, RT_PAGES_COUNTING or RT_PAGES_RELEASING
// A range for the owner; its id, or -1 (off, a mode above the level, outside heap and alias, unaligned, overlapping).
// Fatal before rt_dispatch_init has sealed the function tables: the index that tells guest code from runtime code
// is built here, never in a fault handler.
int rt_pages_watch(uint64_t addr, uint64_t size, int mode, const RtPageOwner *owner);
void rt_pages_unwatch(int id);   // armed pages get their access back, released ones are committed again (zero-filled)
uint64_t rt_pages_arm(int id, uint64_t addr, uint64_t size);       // pages armed; counting ranges
uint64_t rt_pages_release(int id, uint64_t addr, uint64_t size);   // pages released; releasing ranges
int rt_pages_stats(int id, RtPageStats *out);
