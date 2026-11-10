// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The index of the game's memory pools, the classes of a raw-access miss and their counts (oot2/raw_diag.h). No device, no
// lock and no call into the runtime: out of memory only stops the index and the counts from growing, and is counted.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/raw_diag.h"

typedef unsigned long long ull;

#define POOLS_MAX (4u << 20)   // pools kept; a run makes thousands
#define TALLY_MAX 4096         // stages with classes, classes, and stage-and-class pairs kept

// ---------- the pools
static RawDiagPool *g_pool;   // every pool made, sorted by lo (a finalized pool stays, marked)
static uint64_t *g_maxhi;     // g_maxhi[i]: the largest hi among g_pool[0..i]
static size_t g_npool, g_cap;
static RawDiagPool g_ring[RAWDIAG_RING];   // the last finalized pools, in order of finalization
static uint64_t g_nring;                   // finalized so far; the ring holds the last RAWDIAG_RING
static RawDiagPoolCounts g_counts;

// The number of pools whose range starts at or below addr.
static size_t upper(uint64_t addr) {
    size_t a = 0, b = g_npool;
    while (a < b) {
        size_t m = (a + b) / 2;
        if (g_pool[m].lo <= addr) a = m + 1;
        else b = m;
    }
    return a;
}

static void fix_maxhi(size_t from) {
    for (size_t i = from; i < g_npool; i++) {
        uint64_t prev = i ? g_maxhi[i - 1] : 0;
        g_maxhi[i] = g_pool[i].hi > prev ? g_pool[i].hi : prev;
    }
}

void rawdiag_pool_made(uint64_t lo, uint64_t hi, uint32_t flags, int cls, uint32_t present, uint64_t value) {
    if (g_npool == g_cap) {
        size_t cap = g_cap ? g_cap * 2 : 1024;
        if (g_npool >= POOLS_MAX) {
            g_counts.not_kept++;
            return;
        }
        RawDiagPool *p = realloc(g_pool, cap * sizeof *p);
        if (!p) {
            g_counts.not_kept++;
            return;
        }
        g_pool = p;
        uint64_t *m = realloc(g_maxhi, cap * sizeof *m);
        if (!m) {
            g_counts.not_kept++;
            return;
        }
        g_maxhi = m;
        g_cap = cap;
    }
    size_t at = upper(lo);
    memmove(g_pool + at + 1, g_pool + at, (g_npool - at) * sizeof *g_pool);
    g_pool[at] = (RawDiagPool){.lo = lo, .hi = hi, .flags = flags, .cls = cls, .live = 1, .made_present = present,
                               .made_value = value};
    g_npool++;
    fix_maxhi(at);
    g_counts.made++;
    g_counts.live++;
}

int rawdiag_pool_gone(uint64_t lo, uint64_t hi, uint32_t present, uint64_t value) {
    for (size_t i = upper(lo); i-- > 0 && g_pool[i].lo == lo;) {
        RawDiagPool *p = &g_pool[i];
        if (p->hi != hi || !p->live) continue;
        p->live = 0, p->gone_present = present, p->gone_value = value;
        g_ring[g_nring++ % RAWDIAG_RING] = *p;
        g_counts.finalized++, g_counts.live--;
        return 1;
    }
    g_counts.unknown_ends++;
    return 0;
}

// How many finalized pools ago the pool p (finalized at gone_value) ended: 0 for the newest; -1 when the ring has lost it.
static int ring_rank(const RawDiagPool *p) {
    uint64_t kept = g_nring < RAWDIAG_RING ? g_nring : RAWDIAG_RING;
    for (uint64_t k = 0; k < kept; k++) {
        const RawDiagPool *r = &g_ring[(g_nring - 1 - k) % RAWDIAG_RING];
        if (r->lo == p->lo && r->hi == p->hi && r->gone_value == p->gone_value && r->made_value == p->made_value)
            return (int)k;
    }
    return -1;
}

int rawdiag_pool_find(uint64_t addr, uint64_t value, RawDiagHit *hit) {
    *hit = (RawDiagHit){.kind = RAWDIAG_POOL_NONE, .ring = -1};
    size_t at = upper(addr);
    const RawDiagPool *gone = NULL, *later = NULL;
    for (size_t j = at; j-- > 0 && g_maxhi[j] > addr;) {
        const RawDiagPool *p = &g_pool[j];
        if (p->hi <= addr) continue;
        if (p->made_value > value) {
            if (!later || p->made_value < later->made_value) later = p;
        } else if (p->live || value <= p->gone_value) {
            hit->kind = p->cls, hit->pool = *p;   // live for that submission
            return hit->kind;
        } else if (!gone || p->gone_value > gone->gone_value) {
            gone = p;
        }
    }
    if (gone) {
        hit->kind = RAWDIAG_POOL_GONE, hit->pool = *gone, hit->ring = ring_rank(gone);
    } else if (later) {
        hit->kind = RAWDIAG_POOL_LATER, hit->pool = *later;
    } else {
        hit->below = at ? g_maxhi[at - 1] : 0;   // no pool holds addr, so every end before it is below it
        hit->above = at < g_npool ? g_pool[at].lo : 0;
    }
    return hit->kind;
}

void rawdiag_pool_counts(RawDiagPoolCounts *out) { *out = g_counts; }

void rawdiag_pool_reset(void) {
    free(g_pool);
    free(g_maxhi);
    g_pool = NULL, g_maxhi = NULL, g_npool = g_cap = 0;
    memset(g_ring, 0, sizeof g_ring);
    g_nring = 0;
    memset(&g_counts, 0, sizeof g_counts);
}

// ---------- the classes
// pool, decl, commit, now, reason and kind packed in mixed radix: 8 * 3 * 3 * 4 * 4 * 4 * 2 * 2 classes
enum { N_DECL = 3, N_COMMIT = 3, N_NOW = 4, N_REASON = 4, N_KIND = 4, N_ENDED = 2, N_SAME = 2 };

uint32_t rawdiag_class(const RawDiagFacts *f) {
    uint32_t c = (uint32_t)f->pool;
    c = c * N_DECL + (uint32_t)f->decl;
    c = c * N_COMMIT + (uint32_t)f->commit;
    c = c * N_NOW + (uint32_t)f->now;
    c = c * N_REASON + (uint32_t)f->reason;
    c = c * N_KIND + (uint32_t)f->kind;
    c = c * N_ENDED + (uint32_t)(f->ended != 0);
    return c * N_SAME + (uint32_t)(f->same_map != 0);
}

void rawdiag_class_facts(uint32_t c, RawDiagFacts *f) {
    f->same_map = (int)(c % N_SAME), c /= N_SAME;
    f->ended = (int)(c % N_ENDED), c /= N_ENDED;
    f->kind = (int)(c % N_KIND), c /= N_KIND;
    f->reason = (int)(c % N_REASON), c /= N_REASON;
    f->now = (int)(c % N_NOW), c /= N_NOW;
    f->commit = (int)(c % N_COMMIT), c /= N_COMMIT;
    f->decl = (int)(c % N_DECL), c /= N_DECL;
    f->pool = (int)c;
}

// A page the map holds now though the table lacked it: T when the map has not moved since the table was built, else L.
static char held_verdict(const RawDiagFacts *f) {
    if (f->now == BACKING_NONE) return 'A';
    return f->same_map ? 'T' : 'L';
}

// The verdict for a page of memory the host commits (a pool of the CPU's, or the physical page of a virtual pool).
static char commit_verdict(const RawDiagFacts *f) {
    if (f->commit == RAWDIAG_COMMIT_UNKNOWN) return '?';
    if (f->commit == RAWDIAG_COMMIT_NO) return 'C';
    return held_verdict(f);
}

char rawdiag_verdict(const RawDiagFacts *f) {
    if (f->ended) return 'E';
    switch (f->pool) {
    case RAWDIAG_POOL_CPU:
    case RAWDIAG_POOL_PHYSICAL:
        return commit_verdict(f);
    case RAWDIAG_POOL_VIRTUAL:
        return f->decl == RAWDIAG_DECL_FREE ? 'B' : commit_verdict(f);   // a page no mapping request mapped reaches nothing
    case RAWDIAG_POOL_GPU_ONLY:
        return held_verdict(f);
    case RAWDIAG_POOL_CPU_ONLY:
        return 'N';
    default:   // NONE, GONE, LATER
        return 'B';
    }
}

const char *rawdiag_verdict_text(char v) {
    switch (v) {
    case 'A': return "a page of a live pool that the map has no backing for: the table lacks it";
    case 'L': return "a page the map holds now: the map moved since the table that served the submission was built";
    case 'T':
        return "a page the map holds now although the table of the same map version lacks it: the table's build or the write "
               "that learns a page lost it";
    case 'C':
        return "a page of a live pool that was never committed: nothing on the CPU wrote it, so a load reads the zero a fresh "
               "page holds (a store to it is lost)";
    case 'B': return "an address in no pool the GPU could reach at that submission: the pointer is the question";
    case 'N': return "an address in a pool the GPU cannot reach (GPU_NO_ACCESS)";
    case 'E': return "an address in a pool that ended after the submission: what the host and the map say is of a later state";
    default: return "an address whose commit state is unknown";
    }
}

static const char *const k_pool[RAWDIAG_POOL_KINDS] = {"none", "finalized", "later", "cpu", "gpu-only", "physical",
                                                       "virtual", "cpu-only"};
static const char *const k_decl[N_DECL] = {"other", "mapped", "free"};
static const char *const k_commit[N_COMMIT] = {"unknown", "no", "yes"};
static const char *const k_now[N_NOW] = {"none", "extent", "import", "mirror"};
static const char *const k_reason[N_REASON] = {"?", "outside-span", "no-chunk", "no-page"};
static const char *const k_kind[N_KIND] = {"?", "load", "store", "atomic"};

static const char *pick(const char *const *names, int n, int i) { return i >= 0 && i < n ? names[i] : "?"; }

size_t rawdiag_class_name(uint32_t cls, char *buf, size_t size) {
    RawDiagFacts f;
    rawdiag_class_facts(cls, &f);
    int n = snprintf(buf, size, "%s/%s/pool:%s/decl:%s/commit:%s/now:%s%s%s", pick(k_kind, N_KIND, f.kind),
                     pick(k_reason, N_REASON, f.reason), pick(k_pool, RAWDIAG_POOL_KINDS, f.pool),
                     pick(k_decl, N_DECL, f.decl), pick(k_commit, N_COMMIT, f.commit), pick(k_now, N_NOW, f.now),
                     f.ended ? "/ended" : "", f.same_map ? "/same-map" : "");
    return n < 0 ? 0 : (size_t)n;
}

static size_t put(char *buf, size_t size, size_t at, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static size_t put(char *buf, size_t size, size_t at, const char *fmt, ...) {
    if (at >= size) return at;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + at, size - at, fmt, ap);
    va_end(ap);
    return n < 0 ? at : at + (size_t)n;
}

size_t rawdiag_describe(const RawDiagFacts *f, const RawDiagHit *hit, char *buf, size_t size) {
    size_t at = 0;
    const RawDiagPool *p = &hit->pool;
    switch (hit->kind) {
    case RAWDIAG_POOL_NONE:
        at = put(buf, size, at, "in no pool");
        if (hit->below) at = put(buf, size, at, " (the nearest pool below ends at %#llx", (ull)hit->below);
        else at = put(buf, size, at, " (no pool below");
        if (hit->above) at = put(buf, size, at, ", the nearest above starts at %#llx)", (ull)hit->above);
        else at = put(buf, size, at, ", none above)");
        break;
    case RAWDIAG_POOL_GONE:
        at = put(buf, size, at, "in a pool [%#llx, %#llx) made at present %u and finalized at present %u, before this "
                 "submission", (ull)p->lo, (ull)p->hi, p->made_present, p->gone_present);
        if (hit->ring >= 0) at = put(buf, size, at, " (%d pools finalized since)", hit->ring);
        else at = put(buf, size, at, " (more than %d pools finalized since)", RAWDIAG_RING);
        break;
    case RAWDIAG_POOL_LATER:
        at = put(buf, size, at, "in a pool [%#llx, %#llx) made at present %u, after this submission", (ull)p->lo,
                 (ull)p->hi, p->made_present);
        break;
    default:
        if (p->live)
            at = put(buf, size, at, "in a live %s pool [%#llx, %#llx) made at present %u (flags %#x)",
                     pick(k_pool, RAWDIAG_POOL_KINDS, hit->kind), (ull)p->lo, (ull)p->hi, p->made_present, p->flags);
        else
            at = put(buf, size, at, "in a %s pool [%#llx, %#llx) made at present %u and finalized at present %u, after this "
                     "submission (flags %#x)", pick(k_pool, RAWDIAG_POOL_KINDS, hit->kind), (ull)p->lo, (ull)p->hi,
                     p->made_present, p->gone_present, p->flags);
    }
    at = put(buf, size, at, "; %s; %s; %s",
             f->decl == RAWDIAG_DECL_MAPPED ? "the game declared it mapped"
             : f->decl == RAWDIAG_DECL_FREE ? "the game did not map it (free in QueryMemory)"
                                            : "outside the heap and alias regions",
             f->commit == RAWDIAG_COMMIT_YES  ? "committed"
             : f->commit == RAWDIAG_COMMIT_NO ? "not committed"
                                              : "not guest memory",
             f->now == BACKING_NONE ? "the map has no backing for it now"
             : f->now == BACKING_EXTENT ? "the map holds it now as an extent"
             : f->now == BACKING_IMPORT ? "the map holds it now as an import"
                                        : "the map holds it now as a mirror");
    if (f->ended) at = put(buf, size, at, " (read after the pool ended)");
    return at;
}

// ---------- counts
uint64_t rawdiag_share(uint64_t count, uint32_t kept, uint32_t i) {
    if (!kept) return 0;
    return count / kept + (i < count % kept ? 1 : 0);
}

typedef struct StageSeen {
    char key[RAWDIAG_NAME];
    char stage;
    uint32_t cls;
} StageSeen;

struct RawDiagTally {
    RawDiagClassRow *cls;
    int ncls, ccap;
    RawDiagStageRow *stg;
    int nstg, scap;
    StageSeen *seen;
    int nseen, ucap;
    uint64_t entries, reads, dropped;
};

RawDiagTally *rawdiag_tally_new(void) { return calloc(1, sizeof(RawDiagTally)); }

void rawdiag_tally_free(RawDiagTally *t) {
    if (!t) return;
    free(t->cls);
    free(t->stg);
    free(t->seen);
    free(t);
}

static void *room(void *p, int *cap, int need, size_t size) {
    if (need <= *cap) return p;
    int c = *cap ? *cap * 2 : 16;
    while (c < need) c *= 2;
    void *q = realloc(p, (size_t)c * size);
    if (!q) return NULL;
    *cap = c;
    return q;
}

int rawdiag_tally_add(RawDiagTally *t, const char *key, char stage, uint32_t cls, uint64_t reads) {
    t->entries++, t->reads += reads;
    int i;
    for (i = 0; i < t->ncls && t->cls[i].cls != cls; i++) {
    }
    if (i == t->ncls) {
        void *q = t->ncls < TALLY_MAX ? room(t->cls, &t->ccap, t->ncls + 1, sizeof *t->cls) : NULL;
        if (q) t->cls = q, t->cls[t->ncls++] = (RawDiagClassRow){.cls = cls};
        else t->dropped++;
    }
    if (i < t->ncls) t->cls[i].entries++, t->cls[i].reads += reads;
    int s;
    for (s = 0; s < t->nstg && (t->stg[s].stage != stage || strncmp(t->stg[s].key, key, RAWDIAG_NAME - 1)); s++) {
    }
    if (s == t->nstg) {
        void *q = t->nstg < TALLY_MAX ? room(t->stg, &t->scap, t->nstg + 1, sizeof *t->stg) : NULL;
        if (q) {
            t->stg = q;
            RawDiagStageRow *r = &t->stg[t->nstg++];
            memset(r, 0, sizeof *r);
            snprintf(r->key, sizeof r->key, "%s", key);
            r->stage = stage;
        } else {
            t->dropped++;
        }
    }
    if (s < t->nstg) t->stg[s].entries++, t->stg[s].reads += reads;
    int u;
    for (u = 0; u < t->nseen; u++)
        if (t->seen[u].cls == cls && t->seen[u].stage == stage && !strncmp(t->seen[u].key, key, RAWDIAG_NAME - 1)) return 0;
    void *q = t->nseen < TALLY_MAX ? room(t->seen, &t->ucap, t->nseen + 1, sizeof *t->seen) : NULL;
    if (!q) {
        t->dropped++;
        return 0;
    }
    t->seen = q;
    StageSeen *n = &t->seen[t->nseen++];
    memset(n, 0, sizeof *n);
    snprintf(n->key, sizeof n->key, "%s", key);
    n->stage = stage, n->cls = cls;
    return 1;
}

int rawdiag_tally_class(const RawDiagTally *t, int i, RawDiagClassRow *out) {
    if (i < 0 || i >= t->ncls) return 0;
    *out = t->cls[i];
    return 1;
}

int rawdiag_tally_stage(const RawDiagTally *t, int i, RawDiagStageRow *out) {
    if (i < 0 || i >= t->nstg) return 0;
    *out = t->stg[i];
    return 1;
}

uint64_t rawdiag_tally_entries(const RawDiagTally *t) { return t->entries; }
uint64_t rawdiag_tally_reads(const RawDiagTally *t) { return t->reads; }
uint64_t rawdiag_tally_dropped(const RawDiagTally *t) { return t->dropped; }

// ---------- how the missed addresses of a stage lie
size_t rawdiag_shape(const uint64_t *a, uint32_t n, char *buf, size_t size) {
    size_t at = 0;
    if (!n) return put(buf, size, 0, "no address");
    uint64_t lo = a[0], hi = a[0];
    uint32_t distinct = 0;
    for (uint32_t i = 0; i < n; i++) {
        lo = a[i] < lo ? a[i] : lo, hi = a[i] > hi ? a[i] : hi;
        uint32_t j = 0;
        while (j < i && a[j] != a[i]) j++;
        distinct += j == i;
    }
    at = put(buf, size, at, "first");
    for (uint32_t i = 0; i < n && i < 4; i++) at = put(buf, size, at, " %#llx", (ull)a[i]);
    at = put(buf, size, at, "; %u distinct, from %#llx to %#llx", distinct, (ull)lo, (ull)hi);
    uint32_t pairs = n > 16 ? 15 : n - 1, best = 0;
    int64_t step = 0;
    for (uint32_t i = 0; i < pairs; i++) {
        int64_t d = (int64_t)(a[i + 1] - a[i]);
        uint32_t same = 0;
        for (uint32_t j = 0; j < pairs; j++) same += (int64_t)(a[j + 1] - a[j]) == d;
        if (same > best) best = same, step = d;
    }
    if (pairs < 3) at = put(buf, size, at, "; too few to tell whether the step repeats");
    else if (best == pairs && step) at = put(buf, size, at, "; every step is %lld bytes: an arithmetic series", (long long)step);
    else if (best * 2 > pairs && step) at = put(buf, size, at, "; %u of %u steps are %lld bytes", best, pairs, (long long)step);
    else at = put(buf, size, at, "; no step repeats");
    return at;
}

// ---------- the pointer
int rawdiag_scan_word(const uint8_t *p, uint64_t n, uint64_t target, uint64_t near, RawDiagWord *exact, RawDiagWord *below) {
    int found = 0;
    for (uint64_t off = 0; off + 8 <= n; off += 4) {
        uint64_t w;
        memcpy(&w, p + off, sizeof w);
        if (w == target) {
            if (!(found & 1)) *exact = (RawDiagWord){off, w}, found |= 1;
        } else if (w < target && target - w <= near && (!(found & 2) || w > below->word)) {
            *below = (RawDiagWord){off, w}, found |= 2;
        }
    }
    return found;
}
