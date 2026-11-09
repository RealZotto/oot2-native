// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The order a back end replays the queues' records in, so that a WaitEvent is answered (runtime/nvn/nvn_order.c;
// gfx.h, Synchronisation; architecture section 5).
//
// On the console each queue runs by itself and a WaitEvent holds its queue until the event word reaches the value:
// another queue's SignalEvent, or the CPU's EventSignal, releases it. A back end that merges every queue into one
// replay order has to do the same with the order: a queue whose wait the replay so far does not satisfy is parked at
// that record, its later records (and the later submissions of that queue) wait, the other queues go on, and a signal
// that is replayed, or a CPU write that is reported, lets the parked queue go on. Nothing blocks a thread.
//
// The engine owns no records of the back end's and no frames. It is given submissions (a queue, a fence value, the
// recordings of nvn_obj.h's command records) and hands every record of them to the back end's sink in the order it
// decides: a queue's records in that queue's order, a wait only once it holds. A submission that stops at a wait is
// copied from the wait on (the front end's copies are not the back end's after submit returns). The sink also hears
// when a submission is whole, in any order: the back end keeps gfx.h's fence rule itself (a value is done when its
// submission and the queue's earlier ones are whole and written), with wo_whole_through to say how far the whole
// submissions of every queue reach.
//
// What the engine knows of an event word is its own copy of the value as replay has left it: every signal replayed so
// far, WRITE or ADD, and every CPU write reported. The word in memory is read the first time the engine meets it, and
// again at a CPU write of a word it knows. A write to a word it has not met is not entered (the back end hears of every
// 4-byte flush, event or not, so the table would grow with them): its address goes in a fixed table of the latest such
// writes, which only labels the word's writer as the CPU if a wait or signal names it later. A back end writes a GPU signal to memory only when its submission
// completes, later than replay, so memory is not what the next wait should see: the engine keeps, per word, the ADD
// deltas replayed and not yet written (wo_signal_written says when one is), and a CPU re-read adds them back. A WRITE
// replayed and not yet written is not kept: a CPU write to the same word in that window takes the memory's value. All
// calls are made by one thread at a time (the back end's lock), and the sink must not call back.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "nvn_obj.h"

// The bound of a wait nothing releases (decision: gfx.h). A parked queue is stopped for when more than WO_BOUND_SUBMITS
// further submissions, of any queue, have been made since it parked; and a wait with no timeout for a fence value of that
// queue, or for that queue to finish, stops after WO_STALL_NS counted from the start of that wait (or from the park, if
// the park is later).
#define WO_BOUND_SUBMITS 1024u
#define WO_STALL_NS (10ull * 1000000000ull)

enum { WO_WRITER_NONE, WO_WRITER_GPU, WO_WRITER_CPU };   // what last wrote an event word, as replay left it
#define WO_RECENT 8192   // CPU writes to words not met yet that are remembered (direct mapped on the address)

// How a WaitEvent was answered.
typedef struct {
    int writer;           // WO_WRITER_*: who wrote the word last (NONE: it held the value from the start)
    uint64_t signaller;   // writer GPU: the queue of that signal
    int parked;           // the wait did not hold at its place: its queue was held for it
    uint32_t seen;        // the word's value, as replay left it, when the wait was judged
} WoWait;

typedef struct {
    void *ctx;
    // A wait that does not hold at its place: its queue is parked here and the record comes later, through record().
    void (*parked)(void *ctx, uint64_t queue, uint64_t fence, int index, const p_wait_event *w, const WoWait *how);
    // One record of a submission (every record, OP_BEGIN and OP_END included), in replay order. index is the record's
    // recording in the submission. For OP_WAIT_EVENT, how says what answered it (how->parked: after a park); NULL for
    // every other op. A signal is already part of the word the engine keeps when it is shown.
    void (*record)(void *ctx, uint64_t queue, uint64_t fence, int index, const op_hdr *rec, const WoWait *how);
    // Every record of the submission has been shown.
    void (*whole)(void *ctx, uint64_t queue, uint64_t fence);
    uint64_t (*now_ns)(void);   // NULL: time is not kept (the time bound never fires)
} WoSink;

typedef struct WoLane WoLane;
typedef struct WoSub WoSub;
typedef struct WoWord WoWord;
typedef struct {
    WoSink sink;
    WoLane *lanes;
    WoSub *first, *last;         // submissions not whole yet, in fence order
    uint64_t last_fence;
    uint64_t submits;            // submissions made
    WoWord *words;               // the event words met, open addressing, a power of two long
    size_t word_cap, word_count;
    const volatile uint32_t *cpu_recent[WO_RECENT];   // words the CPU wrote before any wait or signal named them
    uint64_t parks, releases;    // waits that parked, and that were released after
    uint64_t longest_hold;       // the most submissions (of any queue) made while one queue was held at one wait, once released
} WoOrder;

// Start (or restart) an engine; frees what an earlier one kept.
void wo_start(WoOrder *o, const WoSink *sink);
void wo_end(WoOrder *o);

// A submission of `queue` (a serial from the front end; 0 is a queue of its own) with fence value `fence`, which
// grows by one per submission over every queue. handles[] are the recordings of nvn_obj.h (host pointers). Shows what can
// be shown, and everything the signals in it release on the other queues.
void wo_submit(WoOrder *o, uint64_t queue, uint64_t fence, const uint64_t *handles, int n);

// The CPU wrote the 4-byte event word `val` (nvnEventSignal, reported through guest_written): if
// the engine knows the word it reads it again (adding the GPU ADDs not yet written) and releases the queues it
// satisfies. A word it has not met is only remembered as written by the CPU (the latest 8192 such addresses).
void wo_cpu_write(WoOrder *o, const volatile uint32_t *val);

// An event was initialised at the word `val` (nvnEventInitialize): what the engine knew of the word is dropped, it is
// read again at the next wait or signal that names it, and a parked wait on it is judged again.
void wo_word_reset(WoOrder *o, const volatile uint32_t *val);

// The back end has written a GPU signal (mode and value as in its record) of `val` to memory: an ADD is no longer pending.
void wo_signal_written(WoOrder *o, const volatile uint32_t *val, uint32_t mode, uint32_t value);

// The highest fence value F such that every submission up to F is whole (the last fence value when all are).
uint64_t wo_whole_through(const WoOrder *o);
int wo_parked(const WoOrder *o);   // queues parked now

// A wait nothing released within a bound: the first parked queue, in the order the queues were made, that meets the query,
// or 0. A bound of 0 is no such bound.
#define WO_ANY_QUEUE UINT64_MAX
typedef struct {
    uint64_t max_submits;   // further submissions, of any queue, since the queue parked
    uint64_t max_ns;        // time parked, counted from the later of the park and since_ns
    uint64_t since_ns;      // the start of the wait that asks (0: the park)
    uint64_t queue;         // only this queue (WO_ANY_QUEUE: any)
    uint64_t upto_fence;    // only a queue whose oldest unfinished submission has a fence value up to this (0: any)
} WoStallQuery;
typedef struct {
    uint64_t queue, fence;       // the parked queue, and the submission its wait is in
    int index;
    p_wait_event wait;
    uint32_t seen;               // the word's value now
    uint64_t submits_since;
    uint64_t waited_ns;
} WoStall;
int wo_stalled(const WoOrder *o, const WoStallQuery *q, WoStall *out);
