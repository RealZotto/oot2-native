// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The index of pools, the classes and the counts of a raw-access miss (runtime/gpu/raw_diag.c) on their own, for
// tests/test_raw_diag.py. Every case prints "ok <name>" or "FAIL <name>: <what>"; the exit status is 1 after any failure.
// Addresses are round numbers made up for the cases.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/raw_diag.h"

static int g_failed;
static const char *g_case;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            printf("FAIL %s: line %d: %s\n", g_case, __LINE__, #cond);                                                 \
            g_failed = 1;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#define M (1ull << 20)

static RawDiagHit find(uint64_t addr, uint64_t value) {
    RawDiagHit h;
    rawdiag_pool_find(addr, value, &h);
    return h;
}

static void case_find(void) {
    rawdiag_pool_reset();
    rawdiag_pool_made(0x100 * M, 0x110 * M, 1, RAWDIAG_POOL_CPU, 3, 10);
    rawdiag_pool_made(0x200 * M, 0x201 * M, 2, RAWDIAG_POOL_GPU_ONLY, 4, 11);
    RawDiagHit h = find(0x100 * M, 20);
    CHECK(h.kind == RAWDIAG_POOL_CPU && h.pool.lo == 0x100 * M && h.pool.made_present == 3);
    CHECK(find(0x110 * M - 1, 20).kind == RAWDIAG_POOL_CPU);
    h = find(0x110 * M, 20);   // the end is outside
    CHECK(h.kind == RAWDIAG_POOL_NONE && h.below == 0x110 * M && h.above == 0x200 * M);
    h = find(0xff * M, 20);
    CHECK(h.kind == RAWDIAG_POOL_NONE && h.below == 0 && h.above == 0x100 * M);
    h = find(0x300 * M, 20);
    CHECK(h.kind == RAWDIAG_POOL_NONE && h.below == 0x201 * M && h.above == 0);
    CHECK(find(0x200 * M + 5, 20).kind == RAWDIAG_POOL_GPU_ONLY);
}

static void case_nested(void) {
    rawdiag_pool_reset();
    rawdiag_pool_made(0x100 * M, 0x900 * M, 0, RAWDIAG_POOL_CPU, 1, 1);
    rawdiag_pool_made(0x200 * M, 0x300 * M, 0, RAWDIAG_POOL_PHYSICAL, 2, 2);
    CHECK(find(0x250 * M, 5).kind == RAWDIAG_POOL_PHYSICAL);   // the inner pool starts last
    RawDiagHit h = find(0x500 * M, 5);                         // past the inner pool's end: the outer one holds it
    CHECK(h.kind == RAWDIAG_POOL_CPU && h.pool.hi == 0x900 * M);
    CHECK(find(0x900 * M, 5).kind == RAWDIAG_POOL_NONE);
}

static void case_lifetime(void) {
    rawdiag_pool_reset();
    rawdiag_pool_made(0x100 * M, 0x110 * M, 0, RAWDIAG_POOL_CPU, 3, 10);
    CHECK(find(0x100 * M, 9).kind == RAWDIAG_POOL_LATER);   // a submission before the pool existed
    CHECK(find(0x100 * M, 10).kind == RAWDIAG_POOL_CPU);
    CHECK(rawdiag_pool_gone(0x100 * M, 0x110 * M, 8, 20) == 1);
    CHECK(find(0x100 * M, 20).kind == RAWDIAG_POOL_CPU);    // a submission handed out before the end may have used it
    RawDiagHit h = find(0x100 * M, 21);
    CHECK(h.kind == RAWDIAG_POOL_GONE && h.ring == 0 && h.pool.gone_present == 8 && h.pool.made_present == 3);
    // the range is made again: the older submission still finds the old pool gone, a later one the new pool
    rawdiag_pool_made(0x100 * M, 0x110 * M, 0, RAWDIAG_POOL_CPU, 9, 30);
    CHECK(find(0x100 * M, 25).kind == RAWDIAG_POOL_GONE);
    h = find(0x100 * M, 35);
    CHECK(h.kind == RAWDIAG_POOL_CPU && h.pool.made_present == 9);
    CHECK(find(0x100 * M, 5).kind == RAWDIAG_POOL_LATER);
    // a range nobody made is counted
    CHECK(rawdiag_pool_gone(0x500 * M, 0x510 * M, 1, 1) == 0);
    CHECK(rawdiag_pool_gone(0x100 * M, 0x110 * M, 10, 40) == 1);   // the second pool ends too
    CHECK(rawdiag_pool_gone(0x100 * M, 0x110 * M, 11, 41) == 0);   // and not twice
    RawDiagPoolCounts c;
    rawdiag_pool_counts(&c);
    CHECK(c.made == 2 && c.finalized == 2 && c.live == 0 && c.unknown_ends == 2 && c.not_kept == 0);
}

static void case_ring(void) {
    rawdiag_pool_reset();
    enum { N = RAWDIAG_RING + 88 };
    for (int i = 0; i < N; i++) rawdiag_pool_made((uint64_t)(0x100 + 2 * i) * M, (uint64_t)(0x101 + 2 * i) * M, 0,
                                                  RAWDIAG_POOL_CPU, 1, (uint64_t)(100 + i));
    for (int i = 0; i < N; i++) CHECK(rawdiag_pool_gone((uint64_t)(0x100 + 2 * i) * M, (uint64_t)(0x101 + 2 * i) * M, 2,
                                                         (uint64_t)(1000 + i)) == 1);
    RawDiagHit h = find((uint64_t)(0x100 + 2 * (N - 1)) * M, 5000);   // the newest finalized
    CHECK(h.kind == RAWDIAG_POOL_GONE && h.ring == 0);
    h = find((uint64_t)(0x100 + 2 * (N - RAWDIAG_RING)) * M, 5000);   // the oldest the ring keeps
    CHECK(h.kind == RAWDIAG_POOL_GONE && h.ring == RAWDIAG_RING - 1);
    h = find((uint64_t)(0x100 + 2 * (N - RAWDIAG_RING - 1)) * M, 5000);   // one before it: the ring lost it, the index did not
    CHECK(h.kind == RAWDIAG_POOL_GONE && h.ring == -1);
    h = find((uint64_t)0x100 * M, 5000);
    CHECK(h.kind == RAWDIAG_POOL_GONE && h.ring == -1 && h.pool.gone_value == 1000);
}

static uint64_t lcg(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return *s >> 33;
}

static void case_many(void) {
    rawdiag_pool_reset();
    enum { N = 4096 };
    static uint64_t lo[N];
    uint64_t seed = 7;
    for (int i = 0; i < N; i++) lo[i] = (uint64_t)(0x1000 + 3 * i) * M;   // 2 MiB pools with a 1 MiB gap, made in a shuffled order
    for (int i = N - 1; i > 0; i--) {
        int j = (int)(lcg(&seed) % (uint64_t)(i + 1));
        uint64_t t = lo[i];
        lo[i] = lo[j], lo[j] = t;
    }
    for (int i = 0; i < N; i++) rawdiag_pool_made(lo[i], lo[i] + 2 * M, 0, RAWDIAG_POOL_CPU, 1, 1);
    for (int i = 0; i < N; i++) {
        CHECK(find(lo[i] + M, 2).kind == RAWDIAG_POOL_CPU && find(lo[i] + M, 2).pool.lo == lo[i]);
        CHECK(find(lo[i] + 2 * M, 2).kind == RAWDIAG_POOL_NONE);   // the gap above each pool
    }
    for (int i = 0; i < N; i += 2) CHECK(rawdiag_pool_gone(lo[i], lo[i] + 2 * M, 2, 3) == 1);
    for (int i = 0; i < N; i++) {
        RawDiagHit h = find(lo[i] + 1, 9);
        CHECK(h.pool.lo == lo[i] && h.kind == (i % 2 ? RAWDIAG_POOL_CPU : RAWDIAG_POOL_GONE));
    }
    RawDiagPoolCounts c;
    rawdiag_pool_counts(&c);
    CHECK(c.made == N && c.finalized == N / 2 && c.live == N / 2);
}

static void case_classes(void) {
    // every combination of the facts maps to its own class and back
    uint32_t seen = 0;
    for (int pool = 0; pool < RAWDIAG_POOL_KINDS; pool++)
        for (int decl = 0; decl < 3; decl++)
            for (int commit = 0; commit < 3; commit++)
                for (int now = 0; now < 4; now++)
                    for (int reason = 1; reason < 4; reason++)
                        for (int kind = 1; kind < 4; kind++)
                            for (int ended = 0; ended < 2; ended++)
                                for (int same = 0; same < 2; same++) {
                                    RawDiagFacts f = {pool, decl, commit, now, reason, kind, ended, same}, g;
                                    uint32_t c = rawdiag_class(&f);
                                    rawdiag_class_facts(c, &g);
                                    CHECK(!memcmp(&f, &g, sizeof f));
                                    seen++;
                                }
    CHECK(seen == (uint32_t)RAWDIAG_POOL_KINDS * 3 * 3 * 4 * 3 * 3 * 2 * 2);
    RawDiagFacts a = {RAWDIAG_POOL_CPU, RAWDIAG_DECL_MAPPED, RAWDIAG_COMMIT_YES, BACKING_NONE, RAW_MISS_REASON_NO_CHUNK,
                      RAW_MISS_KIND_LOAD, 0, 0};
    RawDiagFacts b = a;
    b.now = BACKING_IMPORT;
    CHECK(rawdiag_class(&a) != rawdiag_class(&b));
    b = a;
    b.ended = 1;   // the same facts of a pool that ended since are another class
    CHECK(rawdiag_class(&a) != rawdiag_class(&b));
    char name[128];
    rawdiag_class_name(rawdiag_class(&a), name, sizeof name);
    CHECK(!strcmp(name, "load/no-chunk/pool:cpu/decl:mapped/commit:yes/now:none"));
    rawdiag_class_name(rawdiag_class(&b), name, sizeof name);
    CHECK(!strcmp(name, "load/no-chunk/pool:cpu/decl:mapped/commit:yes/now:none/ended"));
    b = a;
    b.now = BACKING_IMPORT, b.same_map = 1;   // a page the table of the map's own version lacks is not the page learned later
    rawdiag_class_name(rawdiag_class(&b), name, sizeof name);
    CHECK(!strcmp(name, "load/no-chunk/pool:cpu/decl:mapped/commit:yes/now:import/same-map"));
    RawDiagFacts later = b;
    later.same_map = 0;
    CHECK(rawdiag_class(&b) != rawdiag_class(&later));
}

static void case_verdicts(void) {
    enum { MAPPED = RAWDIAG_DECL_MAPPED, FREE = RAWDIAG_DECL_FREE, OTHER = RAWDIAG_DECL_OTHER };
    enum { YES = RAWDIAG_COMMIT_YES, NO = RAWDIAG_COMMIT_NO, UNKNOWN = RAWDIAG_COMMIT_UNKNOWN };
    struct {
        int pool, decl, commit, now, ended;
        char verdict;
        int same;   // the map is still at the version the table was built at
    } t[] = {
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_NONE, 0, 'A', 0},
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_IMPORT, 0, 'L', 0},
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_MIRROR, 0, 'L', 0},
        {RAWDIAG_POOL_CPU, MAPPED, NO, BACKING_NONE, 0, 'C', 0},
        {RAWDIAG_POOL_CPU, MAPPED, NO, BACKING_IMPORT, 0, 'C', 0},   // an uncommitted page is never a table bug
        {RAWDIAG_POOL_CPU, MAPPED, UNKNOWN, BACKING_NONE, 0, '?', 0},
        {RAWDIAG_POOL_PHYSICAL, MAPPED, YES, BACKING_NONE, 0, 'A', 0},
        // a pool that ended since the submission: the map and the host are read in a later state, whatever they say
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_NONE, 1, 'E', 0},
        {RAWDIAG_POOL_CPU, MAPPED, NO, BACKING_NONE, 1, 'E', 0},
        {RAWDIAG_POOL_GPU_ONLY, OTHER, UNKNOWN, BACKING_NONE, 1, 'E', 0},
        // a pool of the GPU's window: an extent or none
        {RAWDIAG_POOL_GPU_ONLY, OTHER, UNKNOWN, BACKING_NONE, 0, 'A', 0},
        {RAWDIAG_POOL_GPU_ONLY, OTHER, UNKNOWN, BACKING_EXTENT, 0, 'L', 0},
        // a page of a virtual pool is its physical page: unmapped reaches nothing; mapped is judged by the physical page
        {RAWDIAG_POOL_VIRTUAL, FREE, UNKNOWN, BACKING_NONE, 0, 'B', 0},
        {RAWDIAG_POOL_VIRTUAL, MAPPED, NO, BACKING_NONE, 0, 'C', 0},
        {RAWDIAG_POOL_VIRTUAL, MAPPED, YES, BACKING_NONE, 0, 'A', 0},
        {RAWDIAG_POOL_VIRTUAL, MAPPED, YES, BACKING_IMPORT, 0, 'L', 0},
        {RAWDIAG_POOL_VIRTUAL, MAPPED, UNKNOWN, BACKING_NONE, 0, '?', 0},
        {RAWDIAG_POOL_CPU_ONLY, MAPPED, YES, BACKING_NONE, 0, 'N', 0},
        {RAWDIAG_POOL_NONE, MAPPED, YES, BACKING_NONE, 0, 'B', 0},
        {RAWDIAG_POOL_NONE, MAPPED, NO, BACKING_NONE, 0, 'B', 0},
        {RAWDIAG_POOL_NONE, MAPPED, YES, BACKING_IMPORT, 0, 'B', 0},   // memory the map holds in no pool is still no pool
        {RAWDIAG_POOL_GONE, MAPPED, YES, BACKING_NONE, 0, 'B', 0},
        {RAWDIAG_POOL_LATER, MAPPED, YES, BACKING_IMPORT, 0, 'B', 0},
        // the map holds the page and has not moved since the table was built: the table lost it (T), never "learned later"
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_IMPORT, 0, 'T', 1},
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_MIRROR, 0, 'T', 1},
        {RAWDIAG_POOL_PHYSICAL, MAPPED, YES, BACKING_IMPORT, 0, 'T', 1},
        {RAWDIAG_POOL_GPU_ONLY, OTHER, UNKNOWN, BACKING_EXTENT, 0, 'T', 1},
        {RAWDIAG_POOL_VIRTUAL, MAPPED, YES, BACKING_IMPORT, 0, 'T', 1},
        {RAWDIAG_POOL_CPU, MAPPED, NO, BACKING_IMPORT, 0, 'C', 1},   // an uncommitted page is still never a table bug
        {RAWDIAG_POOL_CPU, MAPPED, YES, BACKING_IMPORT, 1, 'E', 1},
        {RAWDIAG_POOL_NONE, MAPPED, YES, BACKING_IMPORT, 0, 'B', 1},
        {RAWDIAG_POOL_VIRTUAL, FREE, UNKNOWN, BACKING_IMPORT, 0, 'B', 1},
    };
    for (size_t i = 0; i < sizeof t / sizeof *t; i++) {
        RawDiagFacts f = {.pool = t[i].pool, .decl = t[i].decl, .commit = t[i].commit, .now = t[i].now, .reason = 2,
                          .kind = 1, .ended = t[i].ended, .same_map = t[i].same};
        if (rawdiag_verdict(&f) != t[i].verdict) {
            printf("FAIL %s: row %zu: pool %d decl %d commit %d now %d ended %d gave %c, wanted %c\n", g_case, i, t[i].pool,
                   t[i].decl, t[i].commit, t[i].now, t[i].ended, rawdiag_verdict(&f), t[i].verdict);
            g_failed = 1;
            return;
        }
    }
    CHECK(strstr(rawdiag_verdict_text('A'), "table lacks it"));
    CHECK(strstr(rawdiag_verdict_text('T'), "same map version lacks it"));
    CHECK(strstr(rawdiag_verdict_text('C'), "never committed"));
    CHECK(strstr(rawdiag_verdict_text('B'), "pointer is the question"));
    CHECK(strstr(rawdiag_verdict_text('E'), "ended after the submission"));
}

static void case_describe(void) {
    rawdiag_pool_reset();
    rawdiag_pool_made(0x100 * M, 0x110 * M, 0x4, RAWDIAG_POOL_CPU, 3, 10);
    rawdiag_pool_made(0x200 * M, 0x210 * M, 0x4, RAWDIAG_POOL_CPU, 4, 12);
    rawdiag_pool_gone(0x200 * M, 0x210 * M, 7, 15);
    char s[512];
    RawDiagFacts f = {RAWDIAG_POOL_CPU, RAWDIAG_DECL_MAPPED, RAWDIAG_COMMIT_NO, BACKING_NONE, RAW_MISS_REASON_NO_CHUNK,
                      RAW_MISS_KIND_LOAD, 0, 0};
    RawDiagHit h = find(0x100 * M + 8, 20);
    rawdiag_describe(&f, &h, s, sizeof s);
    CHECK(strstr(s, "in a live cpu pool [0x10000000, 0x11000000) made at present 3 (flags 0x4)"));
    CHECK(strstr(s, "the game declared it mapped; not committed; the map has no backing for it now"));
    f.pool = RAWDIAG_POOL_NONE, f.decl = RAWDIAG_DECL_FREE, f.commit = RAWDIAG_COMMIT_UNKNOWN;
    h = find(0x150 * M, 20);
    rawdiag_describe(&f, &h, s, sizeof s);
    CHECK(strstr(s, "in no pool (the nearest pool below ends at 0x11000000, the nearest above starts at 0x20000000)"));
    CHECK(strstr(s, "the game did not map it"));
    CHECK(strstr(s, "not guest memory"));
    f.pool = RAWDIAG_POOL_GONE, f.decl = RAWDIAG_DECL_OTHER, f.commit = RAWDIAG_COMMIT_YES, f.now = BACKING_MIRROR;
    h = find(0x200 * M, 20);
    rawdiag_describe(&f, &h, s, sizeof s);
    CHECK(strstr(s, "made at present 4 and finalized at present 7, before this submission (0 pools finalized since)"));
    CHECK(strstr(s, "outside the heap and alias regions; committed; the map holds it now as a mirror"));
    // a pool live for the submission and finalized since (the submission's value 20 is before the end at 50)
    rawdiag_pool_made(0x300 * M, 0x310 * M, 0x4, RAWDIAG_POOL_CPU, 5, 10);
    rawdiag_pool_gone(0x300 * M, 0x310 * M, 9, 50);
    h = find(0x300 * M + 16, 20);
    CHECK(h.kind == RAWDIAG_POOL_CPU && !h.pool.live);
    f = (RawDiagFacts){RAWDIAG_POOL_CPU, RAWDIAG_DECL_MAPPED, RAWDIAG_COMMIT_YES, BACKING_NONE, RAW_MISS_REASON_NO_CHUNK,
                       RAW_MISS_KIND_LOAD, 1, 0};
    rawdiag_describe(&f, &h, s, sizeof s);
    CHECK(strstr(s, "in a cpu pool [0x30000000, 0x31000000) made at present 5 and finalized at present 9, after this "
                    "submission (flags 0x4)"));
    CHECK(!strstr(s, "in a live"));
    CHECK(strstr(s, "(read after the pool ended)"));
    char tiny[16];
    rawdiag_describe(&f, &h, tiny, sizeof tiny);   // a short buffer is cut, never overrun
    CHECK(strlen(tiny) < sizeof tiny);
}

static void case_tally(void) {
    CHECK(rawdiag_share(10, 4, 0) == 3 && rawdiag_share(10, 4, 1) == 3 && rawdiag_share(10, 4, 2) == 2 &&
          rawdiag_share(10, 4, 3) == 2);
    uint64_t sum = 0;
    for (uint32_t i = 0; i < 64; i++) sum += rawdiag_share(12000, 64, i);
    CHECK(sum == 12000);
    sum = 0;
    for (uint32_t i = 0; i < 5; i++) sum += rawdiag_share(5, 5, i);
    CHECK(sum == 5 && rawdiag_share(3, 0, 0) == 0);
    RawDiagTally *t = rawdiag_tally_new();
    CHECK(rawdiag_tally_add(t, "stage.one", 'c', 7, 3) == 1);    // new stage and class
    CHECK(rawdiag_tally_add(t, "stage.one", 'c', 7, 3) == 0);    // seen
    CHECK(rawdiag_tally_add(t, "stage.one", 'c', 8, 2) == 1);    // the same stage, another class
    CHECK(rawdiag_tally_add(t, "stage.two", 'c', 7, 1) == 1);    // another stage, the same class
    CHECK(rawdiag_tally_add(t, "stage.one", 'd', 7, 1) == 1);    // the draws of a pass with that key are not its dispatch
    CHECK(rawdiag_tally_add(t, "", 'e', 7, 4) == 1);
    CHECK(rawdiag_tally_entries(t) == 6 && rawdiag_tally_reads(t) == 14 && rawdiag_tally_dropped(t) == 0);
    RawDiagClassRow c;
    CHECK(rawdiag_tally_class(t, 0, &c) && c.cls == 7 && c.entries == 5 && c.reads == 12);
    CHECK(rawdiag_tally_class(t, 1, &c) && c.cls == 8 && c.entries == 1 && c.reads == 2);
    CHECK(!rawdiag_tally_class(t, 2, &c));
    RawDiagStageRow s;
    CHECK(rawdiag_tally_stage(t, 0, &s) && !strcmp(s.key, "stage.one") && s.stage == 'c' && s.entries == 3 && s.reads == 8);
    CHECK(rawdiag_tally_stage(t, 2, &s) && s.stage == 'd' && s.reads == 1);
    CHECK(rawdiag_tally_stage(t, 3, &s) && s.stage == 'e' && s.key[0] == 0 && s.reads == 4);
    CHECK(!rawdiag_tally_stage(t, 4, &s));
    rawdiag_tally_free(t);
}

static void case_tally_room(void) {
    // more stages and classes than the tally keeps: the totals stay right, the rest is counted as dropped
    RawDiagTally *t = rawdiag_tally_new();
    uint64_t expect = 0;
    for (uint32_t i = 0; i < 6000; i++) {
        char key[RAWDIAG_NAME];
        snprintf(key, sizeof key, "stage.%u", i);
        rawdiag_tally_add(t, key, 'c', i, 2);
        expect += 2;
    }
    CHECK(rawdiag_tally_entries(t) == 6000 && rawdiag_tally_reads(t) == expect);
    CHECK(rawdiag_tally_dropped(t) > 0);
    RawDiagClassRow c;
    int rows = 0;
    while (rawdiag_tally_class(t, rows, &c)) rows++;
    CHECK(rows == 4096);
    rawdiag_tally_free(t);
}

static void case_shape(void) {
    char s[400];
    uint64_t a[20];
    for (int i = 0; i < 20; i++) a[i] = 0x2000000000ull + (uint64_t)i * 0x40;
    rawdiag_shape(a, 20, s, sizeof s);   // a constant step: addresses computed from an index
    CHECK(strstr(s, "first 0x2000000000 0x2000000040 0x2000000080 0x20000000c0; 20 distinct, from 0x2000000000 to 0x20000004c0"));
    CHECK(strstr(s, "every step is 64 bytes: an arithmetic series"));
    const uint64_t big = 0x30000000ull;   // a step of 768 MiB, as a stride of a computed address can be
    char want[80];
    for (int i = 0; i < 20; i++) a[i] = 0x2000000000ull + (uint64_t)i * big;
    rawdiag_shape(a, 20, s, sizeof s);
    snprintf(want, sizeof want, "every step is %lld bytes", (long long)big);
    CHECK(strstr(s, want));
    a[5] += 0x77, a[11] += 0x123;   // four of the first fifteen steps broken: 11 of 15 stay
    rawdiag_shape(a, 20, s, sizeof s);
    snprintf(want, sizeof want, "11 of 15 steps are %lld bytes", (long long)big);
    CHECK(strstr(s, want) && !strstr(s, "arithmetic series"));
    uint64_t seed = 99;
    for (int i = 0; i < 20; i++) a[i] = 0x1000000000ull + (lcg(&seed) % 0x1000000000ull) / 32 * 32;   // scattered
    rawdiag_shape(a, 20, s, sizeof s);
    CHECK(strstr(s, "no step repeats") && !strstr(s, "arithmetic"));
    uint64_t same[8] = {0x500000, 0x500000, 0x500000, 0x500000, 0x500000, 0x500000, 0x500000, 0x500000};   // one address: no step
    rawdiag_shape(same, 8, s, sizeof s);
    CHECK(strstr(s, "1 distinct, from 0x500000 to 0x500000") && strstr(s, "no step repeats"));
    uint64_t few[2] = {0x500000, 0x500040};
    rawdiag_shape(few, 2, s, sizeof s);
    CHECK(strstr(s, "2 distinct") && strstr(s, "too few to tell"));
    rawdiag_shape(few, 0, s, sizeof s);
    CHECK(!strcmp(s, "no address"));
    char tiny[12];
    rawdiag_shape(a, 20, tiny, sizeof tiny);   // cut, never overrun
    CHECK(strlen(tiny) < sizeof tiny);
}

static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

static void case_scan(void) {
    uint8_t buf[256];
    memset(buf, 0, sizeof buf);
    uint64_t target = 0x1500000a0ull;
    RawDiagWord e = {0}, b = {0};
    CHECK(rawdiag_scan_word(buf, sizeof buf, target, 1 << 20, &e, &b) == 0);
    put64(buf + 36, target);                 // equal, at a 4-byte step that is not 8-aligned
    put64(buf + 64, target - 0x400);         // a base a little below
    put64(buf + 96, target - 0x40);          // a nearer one
    put64(buf + 128, target - (2 << 20));    // too far below
    put64(buf + 160, target + 8);            // above: not a base of an array that holds the target
    CHECK(rawdiag_scan_word(buf, sizeof buf, target, 1 << 20, &e, &b) == 3);   // an equal word and one below
    CHECK(e.offset == 36 && e.word == target);
    CHECK(b.offset == 96 && b.word == target - 0x40);
    memset(&e, 0, sizeof e), memset(&b, 0, sizeof b);
    CHECK(rawdiag_scan_word(buf, 100, target, 16, &e, &b) == 1 && e.offset == 36);   // within 16 bytes only the equal word
    CHECK(rawdiag_scan_word(buf, 40, target, 1 << 20, &e, &b) == 0);                   // the word at 36 reaches past 40
    uint8_t edge[8];
    put64(edge, target - 0x40);              // exactly as far below as the window reaches: still a base
    memset(&e, 0, sizeof e), memset(&b, 0, sizeof b);
    CHECK(rawdiag_scan_word(edge, sizeof edge, target, 0x40, &e, &b) == 2 && b.offset == 0);
    CHECK(rawdiag_scan_word(edge, sizeof edge, target, 0x3f, &e, &b) == 0);            // one byte farther: not
}

int main(int argc, char **argv) {
    static const struct {
        const char *name;
        void (*fn)(void);
    } cases[] = {{"find", case_find},         {"nested", case_nested},     {"lifetime", case_lifetime},
                 {"ring", case_ring},         {"many", case_many},         {"classes", case_classes},
                 {"verdicts", case_verdicts}, {"describe", case_describe}, {"tally", case_tally},
                 {"tally_room", case_tally_room}, {"shape", case_shape}, {"scan", case_scan}};
    int ran = 0;
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (argc > 1 && strcmp(argv[1], cases[i].name)) continue;
        g_case = cases[i].name;
        int before = g_failed;
        cases[i].fn();
        if (g_failed == before) printf("ok %s\n", g_case);
        ran++;
    }
    if (!ran) {
        printf("no case named %s\n", argc > 1 ? argv[1] : "");
        return 2;
    }
    return g_failed;
}
