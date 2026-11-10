// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What explains a raw-access miss (oot2/vk_raw.h, setting raw_diag): the facts about a missed address, the class they make,
// and the counts of the classes. No device, no lock and no call into the runtime: the Vulkan side (vk_raw.c) gathers the
// facts and calls these, and tests/runtime/raw_diag_test.c checks them on their own.
//
// A raw address the page table does not map reads zero and drops its write (oot2/raw_table.h). It can be missed for these
// reasons, and the log of a miss alone (address, kind, reason) says none of them:
//   A  the address lies in a live pool of the CPU, its page is committed, and the map has no backing for it: a page the
//      table lacks. The one reason to fix the table.
//   L  the same, but the map holds the page now and the map has moved since the table that served the submission was built:
//      the page was committed (or the pool made) after that table, and the next raw-access submission's begin learns it.
//   T  the same, but the map is still at the version the table was built at: the map holds a page the table of that very
//      version lacks. Nothing was learned later, so the table's build (or the write that learns a page) lost it: the other
//      reason to fix the table.
//   C  the address lies in a live pool of the CPU and its page was never committed. Nothing on the CPU wrote it: a load reads
//      the zero a fresh page holds, and a store to it is lost (the kind in the class says which it was).
//   B  the address lies in no pool of the GPU's (never one, or one finalized before the submission, or one made after
//      it): a pointer the console's GPU could not follow either (docs/architecture.md D15: pointers originate from buffer
//      addresses, which lie in pools), so the pointer itself is the question.
// Pools the GPU cannot reach (GPU_NO_ACCESS) are N. A pool of the GPU's window (CPU_NO_ACCESS) has no commit state: A when
// the map has no extent for it, L when it has. A page of a virtual pool is the page of the physical pool it is mapped to:
// B when no mapping request mapped it, else C, A or L by that physical page, as a page of a pool of the CPU's is. E: the pool
// that held the address for the submission was finalized before the address was classified (a submission or two later), so
// what the host and the map say is of a later state and no verdict is given.
//
// The index of pools. Every pool the game makes is kept with its range, the present and the timeline value it was made at,
// and, once finalized, the same at its end: a sorted array that never forgets (a long run makes thousands, 40 bytes each),
// so a lookup is a binary search and says whether the pool was live for the submission that missed. The last
// RAWDIAG_RING finalized pools are also kept in a ring, so a lookup that finds a finalized pool says how many finalized
// since. One thread at a time: the caller holds its lock.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/backing.h"
#include "oot2/raw_table.h"

#define RAWDIAG_RING 512   // finalized pools kept in the ring
#define RAWDIAG_KEY 26     // bytes of a stage key with its terminator (nvn_stage.key)
#define RAWDIAG_NAME 84    // bytes of a stage's name in the counts, with its terminator: a key, or up to three joined by '+'

// What kind of pool holds an address. NONE: no pool the game made. GONE: one finalized before the submission. LATER: one
// made after it. The others are live pools, by their flags.
enum {
    RAWDIAG_POOL_NONE,
    RAWDIAG_POOL_GONE,
    RAWDIAG_POOL_LATER,
    RAWDIAG_POOL_CPU,        // the CPU and the GPU see it
    RAWDIAG_POOL_GPU_ONLY,   // CPU_NO_ACCESS
    RAWDIAG_POOL_PHYSICAL,
    RAWDIAG_POOL_VIRTUAL,
    RAWDIAG_POOL_CPU_ONLY,   // GPU_NO_ACCESS: the GPU cannot reach it
    RAWDIAG_POOL_KINDS
};

typedef struct RawDiagPool {
    uint64_t lo, hi;                       // [lo, hi) in GPU addresses
    uint32_t flags;                        // the pool's NVN flags
    int cls;                               // the RAWDIAG_POOL_* of a live pool with these flags
    int live;                              // 0 once finalized
    uint32_t made_present, gone_present;   // the front end's present count when it was made and when it ended
    uint64_t made_value, gone_value;       // the newest timeline value handed out then
} RawDiagPool;

// A pool made: [lo, hi), its flags, the RAWDIAG_POOL_* its flags make, the present and the timeline value now.
void rawdiag_pool_made(uint64_t lo, uint64_t hi, uint32_t flags, int cls, uint32_t present, uint64_t value);
// The pool [lo, hi) finalized: 1, or 0 when no live pool has exactly that range (counted; the pool was never seen).
int rawdiag_pool_gone(uint64_t lo, uint64_t hi, uint32_t present, uint64_t value);

typedef struct RawDiagHit {
    int kind;                // RAWDIAG_POOL_*
    RawDiagPool pool;        // every kind but NONE
    int ring;                // GONE: 0 for the newest finalized pool, up to RAWDIAG_RING - 1; -1 when older than the ring
    uint64_t below, above;   // NONE: where the nearest pool below the address ends and the nearest above begins (0: none)
} RawDiagHit;
// The pool that holds addr for the submission `value`: a pool live at that value wins, then one finalized before it, then
// one made after it. Returns hit->kind.
int rawdiag_pool_find(uint64_t addr, uint64_t value, RawDiagHit *hit);
typedef struct RawDiagPoolCounts {
    uint64_t made, finalized, live;   // since the start
    uint64_t unknown_ends;            // finalizations of a range no live pool had
    uint64_t not_kept;                // pools left out for lack of memory
} RawDiagPoolCounts;
void rawdiag_pool_counts(RawDiagPoolCounts *out);
void rawdiag_pool_reset(void);

// ---- the facts about one missed address, and the class they make
enum { RAWDIAG_DECL_OTHER, RAWDIAG_DECL_MAPPED, RAWDIAG_DECL_FREE };   // QueryMemory: not heap or alias, mapped, free
enum { RAWDIAG_COMMIT_UNKNOWN, RAWDIAG_COMMIT_NO, RAWDIAG_COMMIT_YES };
typedef struct RawDiagFacts {
    int pool;     // RAWDIAG_POOL_*
    int decl;     // RAWDIAG_DECL_*: whether the game declared the address mapped (the heap below its size, an alias page
                  // MapPhysicalMemory mapped; for a page of a virtual pool, whether a mapping request mapped it)
    int commit;   // RAWDIAG_COMMIT_*: whether the host has the page committed
    int now;      // BACKING_*: what the map holds the page as now
    int reason;   // RAW_MISS_REASON_*: what the table said
    int kind;     // RAW_MISS_KIND_*
    int ended;    // 1 when the pool that held the address for the submission was finalized since: decl, commit and now are of a
                  // later state
    int same_map; // 1 when the map's version is still the one the table of the submission was built at (only asked when the
                  // map holds the page now: set it to 0 otherwise)
} RawDiagFacts;
uint32_t rawdiag_class(const RawDiagFacts *f);
void rawdiag_class_facts(uint32_t cls, RawDiagFacts *f);
// The letter of the class (see the top of this file): 'A', 'L', 'T', 'C', 'B', 'N', 'E' or '?' (committed state unknown).
char rawdiag_verdict(const RawDiagFacts *f);
const char *rawdiag_verdict_text(char verdict);
// "load/no-chunk/pool:cpu/decl:mapped/commit:no/now:none": the class as a key for a row.
size_t rawdiag_class_name(uint32_t cls, char *buf, size_t size);
// The facts as a sentence: where the address lies, what the game declared, what the host committed, what the map holds.
size_t rawdiag_describe(const RawDiagFacts *f, const RawDiagHit *hit, char *buf, size_t size);

// ---- counts
// The share of count misses that the i-th of kept logged entries stands for: count / kept, and the first count % kept
// entries one more, so that the shares of all the entries add up to count.
uint64_t rawdiag_share(uint64_t count, uint32_t kept, uint32_t i);

typedef struct RawDiagTally RawDiagTally;
RawDiagTally *rawdiag_tally_new(void);
void rawdiag_tally_free(RawDiagTally *t);
// An entry of the stage named `key` (a stage key, or keys joined by '+'; stage 'c' for a dispatch, 'd' for the draws of a
// pass, 'e' for the misses no stage owns) in class cls, standing for `reads` misses. 1 when this stage and class were not
// seen before: the caller logs them.
int rawdiag_tally_add(RawDiagTally *t, const char *key, char stage, uint32_t cls, uint64_t reads);
typedef struct RawDiagClassRow {
    uint32_t cls;
    uint64_t entries, reads;
} RawDiagClassRow;
typedef struct RawDiagStageRow {
    char key[RAWDIAG_NAME];
    char stage;
    uint64_t entries, reads;
} RawDiagStageRow;
// Row i in order of first sight; 0 past the end.
int rawdiag_tally_class(const RawDiagTally *t, int i, RawDiagClassRow *out);
int rawdiag_tally_stage(const RawDiagTally *t, int i, RawDiagStageRow *out);
uint64_t rawdiag_tally_entries(const RawDiagTally *t);
uint64_t rawdiag_tally_reads(const RawDiagTally *t);
uint64_t rawdiag_tally_dropped(const RawDiagTally *t);   // entries not counted by stage or class for lack of room

// ---- how the missed addresses of a stage lie
// n addresses in the order they were logged, as a sentence: the first four, how many differ, their range, and whether the
// step from one to the next repeats (addresses computed from a lane's index rather than read from a buffer: every step of
// the first sixteen the same, or most of them). The length the text needs, cut to size.
size_t rawdiag_shape(const uint64_t *a, uint32_t n, char *buf, size_t size);

// ---- the pointer
// Where the 64-bit word `target` lies in n bytes at p, at 4-byte steps: the first equal word (*exact), and the word nearest
// below it within `near` bytes (*below): the base of an array a pointer to a later record points into. Returns a mask, 1 when
// *exact was set and 2 when *below was; offsets are from p.
typedef struct RawDiagWord {
    uint64_t offset, word;
} RawDiagWord;
int rawdiag_scan_word(const uint8_t *p, uint64_t n, uint64_t target, uint64_t near, RawDiagWord *exact, RawDiagWord *below);
