// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The order a back end replays the queues' records in, so that a WaitEvent is answered (oot2/nvn_order.h).
//
// Lanes. One lane per queue, made when its first submission arrives. A lane holds the submissions of its queue that
// are not whole yet, oldest first, and, when it is parked, the wait it stopped at. A submission that arrives at an
// empty lane is shown straight from the caller's recordings; when it stops at a wait, the rest of it is copied (from the
// wait record on, with every later recording whole) into memory of the engine's. A submission that arrives at a lane
// that is parked is copied whole and queued.
//
// Release. Each wait is judged against the engine's own copy of the event word. A signal replayed, or a CPU write
// reported, that makes a parked wait hold releases it for good (it is latched), even if the next signal of the same
// stretch of replay overwrites the word: a queue polling for the value sees it when it is written. After every
// submission and every CPU write the engine looks at the parked lanes and runs the ones released, until none moves: a
// lane that runs may signal what another lane waits for. The order inside that loop is the order the lanes were made,
// so a run is deterministic.
//
// Words. The engine's table of event words is open addressing with a 1024-slot start that doubles at half full, as the
// null back end's was. A word is entered when a wait or a signal names it, and only then: a CPU write to a word not in the
// table is not entered (the back end hears of every 4-byte flush, event or not); its address is kept in a fixed table of
// the latest such writes, and the word is read, and labelled as written by the CPU if it is in that table, when a wait or
// signal first names it. Each word also keeps the ADD deltas of GPU signals replayed and not yet written to memory, so that a CPU
// re-read of the word (memory) does not lose them.
#include <stdlib.h>
#include <string.h>

#include "oot2/nvn_order.h"
#include "oot2/runtime.h"

typedef unsigned long long ull;

struct WoWord {
    const volatile uint32_t *val;   // NULL: a free slot
    uint32_t v;                     // the value as replay has left it
    uint32_t pend_add;              // the ADD deltas of GPU signals replayed and not yet written to memory
    int writer;
    uint64_t queue;
};

typedef struct { int index; size_t off, size; } WoSeg;

struct WoSub {
    WoSub *lane_next, *next, *prev;
    uint64_t fence;
    int seg;                  // the recording (of the segment list, or of the caller's handles) being shown
    size_t at;                // the byte offset in it
    int copied;               // the records are in `bytes` (segs[]), not in the caller's recordings
    WoSeg *segs;
    int nseg;
    uint8_t *bytes;
};

struct WoLane {
    WoLane *next;
    uint64_t queue;
    WoSub *head, *tail;
    int parked;               // stopped at the head submission's current record
    int latched;              // the wait held at some moment since it parked (a later signal may have overwritten the word)
    p_wait_event wait;        // that wait
    int wait_index;
    uint64_t park_submits;    // WoOrder.submits when it parked
    uint64_t park_ns;
};

// ---------- the words
static size_t word_slot(const WoWord *t, size_t cap, const volatile uint32_t *val) {
    size_t i = (size_t)(((uint64_t)(uintptr_t)val >> 2) * 0x9E3779B97F4A7C15ull >> 32) & (cap - 1);
    while (t[i].val && t[i].val != val) i = (i + 1) & (cap - 1);
    return i;
}

static WoWord *word_find(WoOrder *o, const volatile uint32_t *val) {
    if (!o->word_cap) return NULL;
    WoWord *w = &o->words[word_slot(o->words, o->word_cap, val)];
    return w->val ? w : NULL;
}

// The word, entered with the value memory holds now when it is new.
static WoWord *word_get(WoOrder *o, const volatile uint32_t *val) {
    WoWord *w = word_find(o, val);
    if (w) return w;
    if (!o->word_cap || 2 * (o->word_count + 1) > o->word_cap) {
        size_t cap = o->word_cap ? 2 * o->word_cap : 1024;
        WoWord *t = calloc(cap, sizeof *t);
        if (!t) rt_fatal("gfx: wait order: the table of event words");
        for (size_t i = 0; i < o->word_cap; i++)
            if (o->words[i].val) t[word_slot(t, cap, o->words[i].val)] = o->words[i];
        free(o->words);
        o->words = t, o->word_cap = cap;
    }
    w = &o->words[word_slot(o->words, o->word_cap, val)];
    w->val = val, w->v = __atomic_load_n(val, __ATOMIC_ACQUIRE), w->pend_add = 0, w->writer = WO_WRITER_NONE, w->queue = 0;
    const volatile uint32_t **recent = &o->cpu_recent[((uintptr_t)val >> 2) & (WO_RECENT - 1)];
    if (*recent == val) w->writer = WO_WRITER_CPU, *recent = NULL;   // written by the CPU before anything named it
    o->word_count++;
    return w;
}

static int holds(const p_wait_event *p, uint32_t v) {
    return p->mode == NVN_EVENT_WAIT_MODE_EQUAL ? v == p->value : (int32_t)(v - p->value) >= 0;
}

// ---------- the submissions
static void sub_free(WoSub *s) {
    free(s->segs);
    free(s->bytes);
    free(s);
}

// The offset after the OP_END of the recording at p, starting at `at`.
static size_t record_end(const uint8_t *p, size_t at, int index, uint64_t fence) {
    for (;;) {
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < sizeof h || (h.size & 7))
            rt_fatal("gfx: wait order: recording %d of submission %llu has a record of %u bytes at +%zu", index,
                     (ull)fence, h.size, at);
        at += h.size;
        if (h.op == OP_END) return at;
    }
}

// Copy what is left of s (from its current record on, then every later recording) out of the caller's recordings.
static void sub_copy(WoSub *s, const uint64_t *handles, int n) {
    if (s->copied) return;
    int first = s->seg;
    size_t total = 0;
    size_t sizes[64];
    int count = n - first;
    WoSeg *segs = calloc((size_t)(count > 0 ? count : 1), sizeof *segs);
    size_t *sz = count > 64 ? malloc(sizeof *sz * (size_t)count) : sizes;
    if (!segs || !sz) rt_fatal("gfx: wait order: out of memory for a parked submission");
    for (int i = 0; i < count; i++) {
        const uint8_t *p = (const uint8_t *)(uintptr_t)handles[first + i];
        size_t from = i == 0 ? s->at : 0;
        sz[i] = record_end(p, from, first + i, s->fence) - from;
        total += sz[i];
    }
    uint8_t *bytes = malloc(total ? total : 1);
    if (!bytes) rt_fatal("gfx: wait order: out of memory for a parked submission");
    size_t off = 0;
    for (int i = 0; i < count; i++) {
        const uint8_t *p = (const uint8_t *)(uintptr_t)handles[first + i];
        size_t from = i == 0 ? s->at : 0;
        memcpy(bytes + off, p + from, sz[i]);
        segs[i] = (WoSeg){first + i, off, sz[i]};
        off += sz[i];
    }
    if (sz != sizes) free(sz);
    s->segs = segs, s->nseg = count, s->bytes = bytes, s->seg = 0, s->at = 0, s->copied = 1;
}

static void sub_copy_all(WoSub *s, const uint64_t *handles, int n) {
    s->seg = 0, s->at = 0;
    sub_copy(s, handles, n);
}

static uint64_t now(const WoOrder *o) { return o->sink.now_ns ? o->sink.now_ns() : 0; }

static WoLane *lane_for(WoOrder *o, uint64_t queue) {
    WoLane **pp = &o->lanes;
    while (*pp && (*pp)->queue != queue) pp = &(*pp)->next;
    if (!*pp) {
        WoLane *l = calloc(1, sizeof *l);
        if (!l) rt_fatal("gfx: wait order: out of memory for a queue");
        l->queue = queue;
        *pp = l;   // appended: lanes run in the order they were made
    }
    return *pp;
}

// A parked wait on this word that holds now is released, whatever the word becomes next: a GPU queue polling for the value
// sees it at the moment it is written, and the signals after it in the same stretch of replay do not take it back. (A
// wait for EQUAL 5, with the signals 5 and then 6 made one after the other, is released.)
static void latch(WoOrder *o, const WoWord *w) {
    for (WoLane *p = o->lanes; p; p = p->next)
        if (p->parked && p->wait.val == w->val && holds(&p->wait, w->v)) p->latched = 1;
}

static void answer(WoOrder *o, WoWord *w, WoWait *how) {
    (void)o;
    how->writer = w->writer, how->signaller = w->queue, how->seen = w->v;
}

// Show the records of s from where it stands. 1: whole. 0: stopped at a wait that does not hold (the lane is parked
// there; for a submission shown from the caller's recordings the rest is copied first).
static int sub_run(WoOrder *o, WoLane *l, WoSub *s, const uint64_t *handles, int n) {
    for (;;) {
        const uint8_t *p;
        int index;
        if (s->copied) {
            if (s->seg >= s->nseg) return 1;
            p = s->bytes + s->segs[s->seg].off, index = s->segs[s->seg].index;
        } else {
            if (s->seg >= n) return 1;
            p = (const uint8_t *)(uintptr_t)handles[s->seg], index = s->seg;
        }
        op_hdr h;
        memcpy(&h, p + s->at, sizeof h);
        if (h.size < sizeof h || (h.size & 7))
            rt_fatal("gfx: wait order: recording %d of submission %llu has a record of %u bytes at +%zu", index,
                     (ull)s->fence, h.size, s->at);
        const op_hdr *r = (const op_hdr *)(const void *)(p + s->at);
        WoWait how = {0};
        if (h.op == OP_WAIT_EVENT) {
            const p_wait_event *w = (const p_wait_event *)(const void *)(r + 1);
            WoWord *wd = word_get(o, w->val);
            answer(o, wd, &how);
            if (!l->latched && !holds(w, wd->v)) {
                if (!l->parked) {
                    how.parked = 1;
                    l->parked = 1, l->wait = *w, l->wait_index = index;
                    l->park_submits = o->submits, l->park_ns = now(o);
                    o->parks++;
                    if (o->sink.parked) o->sink.parked(o->sink.ctx, l->queue, s->fence, index, w, &how);
                }
                if (!s->copied) sub_copy(s, handles, n);
                return 0;
            }
            if (l->parked) {
                how.parked = 1, l->parked = 0, l->latched = 0, o->releases++;
                if (o->submits - l->park_submits > o->longest_hold) o->longest_hold = o->submits - l->park_submits;
            }
        } else if (h.op == OP_SIGNAL_EVENT) {
            const p_signal_event *g = (const p_signal_event *)(const void *)(r + 1);
            WoWord *wd = word_get(o, g->val);
            if (g->mode == NVN_EVENT_SIGNAL_MODE_ADD) wd->v += g->value, wd->pend_add += g->value;
            else wd->v = g->value;
            wd->writer = WO_WRITER_GPU, wd->queue = l->queue;
            latch(o, wd);
        }
        if (o->sink.record)
            o->sink.record(o->sink.ctx, l->queue, s->fence, index, r, h.op == OP_WAIT_EVENT ? &how : NULL);
        s->at += h.size;
        if (h.op == OP_END) s->seg++, s->at = 0;
    }
}

static void sub_whole(WoOrder *o, WoLane *l, WoSub *s) {
    if (o->sink.whole) o->sink.whole(o->sink.ctx, l->queue, s->fence);
    l->head = s->lane_next;
    if (!l->head) l->tail = NULL;
    if (s->prev) s->prev->next = s->next;
    else o->first = s->next;
    if (s->next) s->next->prev = s->prev;
    else o->last = s->prev;
    sub_free(s);
}

// Run a lane from its head while it can.
static void lane_run(WoOrder *o, WoLane *l) {
    while (l->head) {
        WoSub *s = l->head;
        if (!sub_run(o, l, s, NULL, 0)) return;
        sub_whole(o, l, s);
    }
}

// Every parked lane whose wait holds now goes on, until no lane moves.
static void release_all(WoOrder *o) {
    for (int moved = 1; moved;) {
        moved = 0;
        for (WoLane *l = o->lanes; l; l = l->next) {
            if (!l->parked) continue;
            WoWord *wd = word_find(o, l->wait.val);
            if (!l->latched && (!wd || !holds(&l->wait, wd->v))) continue;
            lane_run(o, l);
            moved = 1;
        }
    }
}

// ---------- the calls
void wo_start(WoOrder *o, const WoSink *sink) {
    wo_end(o);
    *o = (WoOrder){.sink = *sink};
}

void wo_end(WoOrder *o) {
    while (o->lanes) {
        WoLane *l = o->lanes;
        o->lanes = l->next;
        while (l->head) {
            WoSub *s = l->head;
            l->head = s->lane_next;
            sub_free(s);
        }
        free(l);
    }
    free(o->words);
    *o = (WoOrder){0};
}

void wo_submit(WoOrder *o, uint64_t queue, uint64_t fence, const uint64_t *handles, int n) {
    WoLane *l = lane_for(o, queue);
    WoSub *s = calloc(1, sizeof *s);
    if (!s) rt_fatal("gfx: wait order: out of memory for a submission");
    s->fence = fence;
    s->prev = o->last;
    if (o->last) o->last->next = s;
    else o->first = s;
    o->last = s;
    o->last_fence = fence;
    o->submits++;
    if (l->head) {   // the lane is parked: this one waits behind
        sub_copy_all(s, handles, n);
        l->tail->lane_next = s, l->tail = s;
    } else {
        l->head = l->tail = s;
        if (sub_run(o, l, s, handles, n)) sub_whole(o, l, s);
    }
    release_all(o);
}

void wo_cpu_write(WoOrder *o, const volatile uint32_t *val) {
    WoWord *w = word_find(o, val);
    if (!w) {   // not met: read at first sight, and the table does not grow with every flush of 4 bytes
        o->cpu_recent[((uintptr_t)val >> 2) & (WO_RECENT - 1)] = val;
        return;
    }
    w->v = __atomic_load_n(val, __ATOMIC_ACQUIRE) + w->pend_add;   // memory, and the GPU ADDs it has not received yet
    w->writer = WO_WRITER_CPU, w->queue = 0;
    latch(o, w);
    release_all(o);
}

void wo_word_reset(WoOrder *o, const volatile uint32_t *val) {
    WoWord *w = word_find(o, val);
    o->cpu_recent[((uintptr_t)val >> 2) & (WO_RECENT - 1)] = NULL;
    if (!w) return;
    w->v = __atomic_load_n(val, __ATOMIC_ACQUIRE), w->pend_add = 0, w->writer = WO_WRITER_NONE, w->queue = 0;
    latch(o, w);
    release_all(o);
}

void wo_signal_written(WoOrder *o, const volatile uint32_t *val, uint32_t mode, uint32_t value) {
    WoWord *w = word_find(o, val);
    if (w && mode == NVN_EVENT_SIGNAL_MODE_ADD) w->pend_add -= value;
}

uint64_t wo_whole_through(const WoOrder *o) { return o->first ? o->first->fence - 1 : o->last_fence; }

int wo_parked(const WoOrder *o) {
    int n = 0;
    for (const WoLane *l = o->lanes; l; l = l->next) n += l->parked != 0;
    return n;
}

int wo_stalled(const WoOrder *o, const WoStallQuery *q, WoStall *out) {
    uint64_t t = now(o);
    for (const WoLane *l = o->lanes; l; l = l->next) {
        if (!l->parked) continue;
        if (q->queue != WO_ANY_QUEUE && l->queue != q->queue) continue;
        if (q->upto_fence && l->head->fence > q->upto_fence) continue;
        uint64_t from = l->park_ns > q->since_ns ? l->park_ns : q->since_ns;   // the wait that asks may be later than the park
        uint64_t since = o->submits - l->park_submits, waited = t > from ? t - from : 0;
        if (!((q->max_submits && since > q->max_submits) || (q->max_ns && waited > q->max_ns))) continue;
        if (out) {
            const WoWord *w = word_find((WoOrder *)(uintptr_t)o, l->wait.val);
            *out = (WoStall){l->queue, l->head->fence, l->wait_index, l->wait, w ? w->v : 0, since, waited};
        }
        return 1;
    }
    return 0;
}
