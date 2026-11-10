// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the order a back end replays the queues' records in, so that a WaitEvent is answered (oot2/nvn_order.h,
// gfx.h, architecture section 5), driven by tests/test_runtime_nvn_order.py: `nvn_order_test <case>`. Each case runs in its
// own process. A passing case prints lines the driver checks against orders written there by hand, then "ok <case>"; a
// stop must end in one named fatal, which the driver checks too.
//
// The recordings are synthetic: records of nvn_obj.h's op headers (OP_BEGIN, OP_HINT as a marker with a label, OP_SIGNAL_EVENT,
// OP_WAIT_EVENT, OP_END) written into byte arrays, as the front end's copies are laid out, with the event words in this
// file's own memory. Two kinds of case: the order engine alone, with a sink that logs what it is shown, and the null back
// end through its table (submit with a queue, guest_written for the CPU's signal, fence_wait, present), which is the
// reference implementation of gfx.h's rules. No device, no guest code, nothing from a dump.
//
// A hung case does not hang the run: a watchdog ends the process with exit 3 after 20 s.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gfx.h"
#include "oot2/nvn_order.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

typedef unsigned long long ull;
#define EQUAL NVN_EVENT_WAIT_MODE_EQUAL
#define GEQUAL NVN_EVENT_WAIT_MODE_GEQUAL_WRAP
#define WRITE NVN_EVENT_SIGNAL_MODE_WRITE
#define ADD NVN_EVENT_SIGNAL_MODE_ADD

// ---------- recordings: OP_BEGIN, records, OP_END, every record 8-byte aligned
typedef struct { _Alignas(8) uint8_t b[4096]; size_t n; } Rec;
static void put(Rec *r, uint32_t op, const void *payload, size_t bytes) {
    uint32_t size = (uint32_t)((sizeof(op_hdr) + bytes + 7) & ~(size_t)7);
    op_hdr h = {op, size};
    CHECK(r->n + size <= sizeof r->b);
    memset(r->b + r->n, 0, size);
    memcpy(r->b + r->n, &h, sizeof h);
    if (bytes) memcpy(r->b + r->n + sizeof h, payload, bytes);
    r->n += size;
}
static void begin(Rec *r) { memset(r, 0, sizeof *r), put(r, OP_BEGIN, NULL, 0); }
static void end(Rec *r) { put(r, OP_END, NULL, 0); }
static void mark(Rec *r, int label) {   // a record that does nothing, with a label to find it by
    p_hint p = {0, {label, 0, 0}};
    put(r, OP_HINT, &p, sizeof p);
}
static void sig(Rec *r, volatile uint32_t *w, uint32_t mode, uint32_t value) {
    p_signal_event p = {w, mode, value, 2, 0};
    put(r, OP_SIGNAL_EVENT, &p, sizeof p);
}
static void waitev(Rec *r, volatile uint32_t *w, uint32_t mode, uint32_t value, uint64_t caller) {
    p_wait_event p = {w, mode, value, caller};
    put(r, OP_WAIT_EVENT, &p, sizeof p);
}
static uint64_t handle(Rec *r) { return (uint64_t)(uintptr_t)r->b; }

// ---------- what the engine and the null back end show, as text: "q1:m3 q2:S q1:W"
static char g_log[16384];
static void logf_add(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void logf_add(const char *fmt, ...) {
    size_t n = strlen(g_log);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_log + n, sizeof g_log - n, fmt, ap);
    va_end(ap);
}
static void log_clear(void) { g_log[0] = 0; }
// The log without its trailing space, and cleared.
static const char *take(void) {
    static char out[16384];
    size_t n = strlen(g_log);
    if (n && g_log[n - 1] == ' ') n--;
    memcpy(out, g_log, n);
    out[n] = 0;
    log_clear();
    return out;
}
static void tag(uint64_t queue, const op_hdr *r) {
    switch (r->op) {
    case OP_HINT: logf_add("q%llu:m%d ", (ull)queue, ((const p_hint *)(const void *)(r + 1))->v[0]); break;
    case OP_SIGNAL_EVENT: logf_add("q%llu:S ", (ull)queue); break;
    case OP_WAIT_EVENT: logf_add("q%llu:W ", (ull)queue); break;
    default: break;
    }
}

// ---------- the engine alone
static WoOrder g_o;
static int g_parked_calls;
static WoWait g_last_wait;        // the answer of the newest wait shown
static uint64_t g_clock;
static uint64_t clock_ns(void) { return g_clock; }
static void eng_parked(void *ctx, uint64_t queue, uint64_t fence, int index, const p_wait_event *w, const WoWait *how) {
    (void)ctx, (void)queue, (void)fence, (void)index, (void)w, (void)how;
    g_parked_calls++;
}
static void eng_record(void *ctx, uint64_t queue, uint64_t fence, int index, const op_hdr *r, const WoWait *how) {
    (void)ctx, (void)fence, (void)index;
    tag(queue, r);
    if (how) g_last_wait = *how;
}
static char g_whole[256];
static void eng_whole(void *ctx, uint64_t queue, uint64_t fence) {
    (void)ctx, (void)queue;
    size_t n = strlen(g_whole);
    snprintf(g_whole + n, sizeof g_whole - n, "w%llu ", (ull)fence);
}
static const char *whole_take(void) {
    static char out[256];
    size_t n = strlen(g_whole);
    if (n && g_whole[n - 1] == ' ') n--;
    memcpy(out, g_whole, n);
    out[n] = 0;
    g_whole[0] = 0;
    return out;
}
static void eng_start(void) {
    log_clear();
    g_whole[0] = 0;
    g_parked_calls = 0;
    g_clock = 0;
    wo_start(&g_o, &(WoSink){NULL, eng_parked, eng_record, eng_whole, clock_ns});
}
static void put_out(const char *what, const char *s) { printf("%s %s\n", what, s); }
static void sub1(uint64_t queue, uint64_t fence, Rec *r) {
    uint64_t h = handle(r);
    wo_submit(&g_o, queue, fence, &h, 1);
}

// (a) a wait released by a later submission of another queue
static void case_order_a(void) {
    static volatile uint32_t E;
    static Rec r1, r2;
    eng_start();
    begin(&r1), mark(&r1, 1), waitev(&r1, &E, EQUAL, 1, 0x1000), mark(&r1, 2), end(&r1);
    begin(&r2), mark(&r2, 3), sig(&r2, &E, WRITE, 1), mark(&r2, 4), end(&r2);
    sub1(1, 1, &r1);
    put_out("after-wait", take());
    CHECK(g_parked_calls == 1 && wo_parked(&g_o) == 1 && wo_whole_through(&g_o) == 0);
    sub1(2, 2, &r2);
    put_out("after-signal", take());
    put_out("whole", whole_take());
    CHECK(g_parked_calls == 1 && wo_parked(&g_o) == 0 && wo_whole_through(&g_o) == 2);
    CHECK(g_last_wait.parked == 1 && g_last_wait.writer == WO_WRITER_GPU && g_last_wait.signaller == 2 &&
          g_last_wait.seen == 1);
    printf("longest-hold %llu parks %llu releases %llu\n", (ull)g_o.longest_hold, (ull)g_o.parks, (ull)g_o.releases);
    CHECK(E == 0);   // the engine writes no memory: the back end writes a signal when its submission completes
    wo_end(&g_o);
    printf("ok order_a\n");
}

// (b) a wait released by a CPU signal made after the submit; the queue's later submissions wait behind it, other queues don't
static void case_order_b(void) {
    static volatile uint32_t E;
    static Rec r1, r2, r3;
    eng_start();
    begin(&r1), mark(&r1, 1), waitev(&r1, &E, EQUAL, 5, 0x1000), mark(&r1, 2), end(&r1);
    begin(&r2), mark(&r2, 3), end(&r2);
    begin(&r3), mark(&r3, 4), end(&r3);
    sub1(1, 1, &r1);
    sub1(1, 2, &r2);   // behind the held wait
    sub1(2, 3, &r3);   // another queue goes on
    put_out("before", take());
    put_out("whole-before", whole_take());
    CHECK(wo_parked(&g_o) == 1 && wo_whole_through(&g_o) == 0);
    E = 4;
    wo_cpu_write(&g_o, &E);   // not the value
    CHECK(wo_parked(&g_o) == 1 && !take()[0]);
    E = 5;
    wo_cpu_write(&g_o, &E);
    put_out("after-cpu", take());
    put_out("whole-after", whole_take());
    CHECK(wo_parked(&g_o) == 0 && wo_whole_through(&g_o) == 3);
    CHECK(g_last_wait.parked == 1 && g_last_wait.writer == WO_WRITER_CPU && g_last_wait.seen == 5);
    wo_end(&g_o);
    printf("ok order_b\n");
}

// A release that cascades: the queue a signal releases signals what a third queue waits for, and that queue's lane comes
// before it in the engine's order, so the loop has to go round again. And a queue held at two waits in a row, released
// one at a time.
static void case_order_cascade(void) {
    static volatile uint32_t E, G, H, I;
    static Rec r1, r2, r3, r4, r5, r6, r7;
    eng_start();
    begin(&r1), waitev(&r1, &G, EQUAL, 1, 0x2000), mark(&r1, 9), end(&r1);
    begin(&r2), waitev(&r2, &E, EQUAL, 1, 0x1000), mark(&r2, 1), sig(&r2, &G, WRITE, 1), mark(&r2, 2), end(&r2);
    begin(&r3), sig(&r3, &E, WRITE, 1), end(&r3);
    sub1(3, 1, &r1);   // held at G
    sub1(1, 2, &r2);   // held at E
    CHECK(wo_parked(&g_o) == 2 && !take()[0]);
    sub1(2, 3, &r3);   // releases queue 1, whose signal of G releases queue 3
    put_out("cascade", take());
    CHECK(wo_parked(&g_o) == 0 && wo_whole_through(&g_o) == 3);
    begin(&r4), waitev(&r4, &H, EQUAL, 1, 0x3000), mark(&r4, 7), waitev(&r4, &I, EQUAL, 1, 0x3100), mark(&r4, 8), end(&r4);
    begin(&r5), sig(&r5, &H, WRITE, 1), end(&r5);
    begin(&r6), sig(&r6, &I, WRITE, 1), end(&r6);
    sub1(4, 4, &r4);
    sub1(2, 5, &r5);   // the first wait only
    put_out("chain-first", take());
    CHECK(wo_parked(&g_o) == 1 && wo_whole_through(&g_o) == 3);
    sub1(2, 6, &r6);
    put_out("chain-second", take());
    CHECK(wo_parked(&g_o) == 0 && wo_whole_through(&g_o) == 6);
    (void)r7;
    wo_end(&g_o);
    printf("ok order_cascade\n");
}

// A signal that the same stretch of replay overwrites: a wait for EQUAL 5, signals 5 and 6 one after the other, is released;
// a wait for a value that was never written is not.
static void case_order_transient(void) {
    static volatile uint32_t E, F;
    static Rec r1, r2, r3;
    eng_start();
    begin(&r1), waitev(&r1, &E, EQUAL, 5, 0x1000), mark(&r1, 1), end(&r1);
    begin(&r2), waitev(&r2, &F, EQUAL, 7, 0x1100), mark(&r2, 2), end(&r2);
    begin(&r3), sig(&r3, &E, WRITE, 5), sig(&r3, &F, WRITE, 8), sig(&r3, &E, WRITE, 6), end(&r3);
    sub1(1, 1, &r1);
    sub1(2, 2, &r2);
    CHECK(wo_parked(&g_o) == 2);
    sub1(3, 3, &r3);
    put_out("transient", take());
    CHECK(wo_parked(&g_o) == 1 && wo_whole_through(&g_o) == 1);   // fence 1 whole, 2 held, 3 whole
    wo_end(&g_o);
    printf("ok order_transient\n");
}

static int stalled(uint64_t max_submits, uint64_t max_ns, uint64_t since_ns, uint64_t queue, uint64_t upto, WoStall *st) {
    WoStallQuery q = {max_submits, max_ns, since_ns, queue, upto};
    return wo_stalled(&g_o, &q, st);
}

// (c) a wait nothing releases is reported after the stated bound in submissions, and after the stated time
static void case_order_c(void) {
    static volatile uint32_t E;
    static Rec r1, rm;
    eng_start();
    g_clock = 1000;
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    sub1(1, 1, &r1);
    WoStall st;
    CHECK(wo_parked(&g_o) == 1 && !stalled(WO_BOUND_SUBMITS, 0, 0, WO_ANY_QUEUE, 0, &st));
    begin(&rm), mark(&rm, 1), end(&rm);
    uint64_t fence = 1;
    // the other queue keeps submitting: not a stop while the count stays within the bound (1024 further submissions)
    for (int i = 0; i < (int)WO_BOUND_SUBMITS; i++) {
        sub1(2, ++fence, &rm);
        CHECK(!stalled(WO_BOUND_SUBMITS, 0, 0, WO_ANY_QUEUE, 0, &st));
    }
    sub1(2, ++fence, &rm);   // the 1025th
    CHECK(stalled(WO_BOUND_SUBMITS, 0, 0, WO_ANY_QUEUE, 0, &st));
    printf("stalled queue %llu fence %llu index %d mode %u value %u seen %u since %llu\n", (ull)st.queue, (ull)st.fence,
           st.index, st.wait.mode, st.wait.value, st.seen, (ull)st.submits_since);
    // the other queue was never held: every one of its marks was shown
    size_t marks = 0;
    for (const char *p = g_log; (p = strstr(p, "q2:m1")); p++) marks++;
    printf("shown %zu\n", marks);
    // a count of 0 is no bound
    CHECK(!stalled(0, 0, 0, WO_ANY_QUEUE, 0, &st));
    // time: not at the bound, past it
    g_clock = 1000 + WO_STALL_NS;
    CHECK(!stalled(0, WO_STALL_NS, 0, WO_ANY_QUEUE, 0, &st));
    g_clock = 1000 + WO_STALL_NS + 1;
    CHECK(stalled(0, WO_STALL_NS, 0, WO_ANY_QUEUE, 0, &st) && st.waited_ns == WO_STALL_NS + 1);
    // the time counts from the later of the park and the start of the wait that asks: a wait that began 5 ns ago has
    // waited 5 ns, however long the queue has been parked
    CHECK(!stalled(0, WO_STALL_NS, g_clock - 5, WO_ANY_QUEUE, 0, &st));
    g_clock += WO_STALL_NS;
    CHECK(stalled(0, WO_STALL_NS, g_clock - WO_STALL_NS - 5, WO_ANY_QUEUE, 0, &st) && st.waited_ns == WO_STALL_NS + 5);
    // only the queue asked about, and only a submission up to the value asked about
    CHECK(!stalled(0, WO_STALL_NS, 0, 2, 0, &st));
    CHECK(stalled(0, WO_STALL_NS, 0, 1, 0, &st) && st.queue == 1);
    CHECK(stalled(0, WO_STALL_NS, 0, WO_ANY_QUEUE, 1, &st));
    // a signal releases it, and nothing is stalled after
    static Rec rs, r4;
    begin(&rs), sig(&rs, &E, WRITE, 1), end(&rs);
    sub1(3, ++fence, &rs);
    CHECK(wo_parked(&g_o) == 0 && !stalled(WO_BOUND_SUBMITS, WO_STALL_NS, 0, WO_ANY_QUEUE, 0, &st));
    // a queue parked later, at a higher fence value, is not what a wait for a lower value sits behind
    static volatile uint32_t E4;
    begin(&r4), waitev(&r4, &E4, EQUAL, 1, 0x2000), end(&r4);
    sub1(4, ++fence, &r4);
    uint64_t parked_fence = fence;
    g_clock += WO_STALL_NS + 1;
    CHECK(stalled(0, WO_STALL_NS, 0, WO_ANY_QUEUE, parked_fence, &st) && st.fence == parked_fence);
    CHECK(!stalled(0, WO_STALL_NS, 0, WO_ANY_QUEUE, parked_fence - 1, &st));
    wo_end(&g_o);
    printf("ok order_c\n");
}

// (d) the fence rule: a queue's records in the queue's order, results of a lower fence value first, done only when every
// submission up to it is whole
static void case_order_d(void) {
    static volatile uint32_t E;
    static Rec r1, r2, r3, r4;
    eng_start();
    begin(&r1), mark(&r1, 1), waitev(&r1, &E, EQUAL, 1, 0x1000), mark(&r1, 2), end(&r1);
    begin(&r2), mark(&r2, 3), end(&r2);
    begin(&r3), mark(&r3, 4), sig(&r3, &E, WRITE, 1), end(&r3);
    begin(&r4), mark(&r4, 5), end(&r4);
    sub1(1, 1, &r1);
    printf("through %llu\n", (ull)wo_whole_through(&g_o));
    sub1(2, 2, &r2);   // whole, but fence value 1 is not: nothing is done through 2
    printf("through %llu\n", (ull)wo_whole_through(&g_o));
    sub1(3, 3, &r3);   // its signal releases fence value 1, which becomes whole after 2 and 3
    printf("through %llu\n", (ull)wo_whole_through(&g_o));
    sub1(2, 4, &r4);
    printf("through %llu\n", (ull)wo_whole_through(&g_o));
    put_out("order", take());
    put_out("whole", whole_take());
    wo_end(&g_o);
    printf("ok order_d\n");
}


// ADD accumulates, from different queues and from the CPU, and GEQUAL_WRAP compares across the wrap
static void case_order_add(void) {
    static volatile uint32_t E, F, G;
    static Rec r1, r2, r3, r4, r5, r6;
    eng_start();
    // a wait for at least 3: 2 is not enough, 2 and 2 from another queue are
    begin(&r1), waitev(&r1, &E, GEQUAL, 3, 0x1000), mark(&r1, 1), end(&r1);
    begin(&r2), sig(&r2, &E, ADD, 2), end(&r2);
    begin(&r3), sig(&r3, &E, ADD, 2), end(&r3);
    sub1(1, 1, &r1);
    sub1(2, 2, &r2);
    CHECK(wo_parked(&g_o) == 1);
    put_out("add-first", take());
    sub1(3, 3, &r3);
    put_out("add-second", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 4 && g_last_wait.writer == WO_WRITER_GPU && g_last_wait.signaller == 3);
    // the same wait, the second ADD made by the CPU on top of the GPU's
    static volatile uint32_t H;
    begin(&r4), waitev(&r4, &H, GEQUAL, 3, 0x1100), mark(&r4, 2), end(&r4);
    begin(&r5), sig(&r5, &H, ADD, 2), end(&r5);
    sub1(4, 4, &r4);
    sub1(5, 5, &r5);
    CHECK(wo_parked(&g_o) == 1);
    H = 1;   // the GPU's add is replayed and not written; the CPU's add of 1 lands on memory
    wo_cpu_write(&g_o, &H);
    put_out("add-cpu", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 3 && g_last_wait.writer == WO_WRITER_CPU);
    // wrap: the word holds 0xFFFFFFFF, the wait is for 1: held until an ADD of 2 takes the word to 1
    F = 0xFFFFFFFFu;
    begin(&r6), waitev(&r6, &F, GEQUAL, 1, 0x1200), mark(&r6, 3), end(&r6);
    sub1(6, 6, &r6);
    CHECK(wo_parked(&g_o) == 1);
    static Rec r7, r8;
    begin(&r7), sig(&r7, &F, ADD, 2), end(&r7);
    sub1(7, 7, &r7);
    put_out("wrap-add", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 1);
    // and a word at 1 satisfies a wait for 0xFFFFFFFF at its place (2 ahead across the wrap), where an unsigned compare
    // would hold it
    G = 1;
    begin(&r8), waitev(&r8, &G, GEQUAL, 0xFFFFFFFFu, 0x1300), mark(&r8, 4), end(&r8);
    sub1(8, 8, &r8);
    put_out("wrap-place", take());
    CHECK(wo_parked(&g_o) == 0 && g_parked_calls == 3);
    wo_end(&g_o);
    printf("ok order_add\n");
}

// The engine's copy of a word against memory: the ADD of a GPU signal not written yet survives a CPU re-read, is not
// owed any more once written, a CPU write to a word no wait or signal named is not entered, and a word that a wait names
// first is read as it is then, and labelled as the CPU's; an event initialised again at a known word starts over
static void case_order_pending(void) {
    static volatile uint32_t H, Z;
    static Rec r1, r2, r3, r4;
    eng_start();
    begin(&r1), sig(&r1, &H, ADD, 2), end(&r1);
    sub1(1, 1, &r1);   // replayed, not written: memory still 0
    CHECK(H == 0);
    H = 1;             // the CPU's add
    wo_cpu_write(&g_o, &H);
    begin(&r2), waitev(&r2, &H, EQUAL, 3, 0x1000), mark(&r2, 1), end(&r2);
    sub1(2, 2, &r2);
    put_out("pending", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 3 && g_last_wait.parked == 0);
    // the back end writes the GPU's add: memory is 3, nothing is owed, and a CPU write of 5 is read as 5
    H += 2;
    wo_signal_written(&g_o, &H, ADD, 2);
    H = 5;
    wo_cpu_write(&g_o, &H);
    begin(&r3), waitev(&r3, &H, EQUAL, 5, 0x1100), mark(&r3, 2), end(&r3);
    sub1(3, 3, &r3);
    put_out("written", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 5);
    // the event is initialised again over the same storage (memory 0 now): the engine's 5 is dropped
    H = 0;
    wo_word_reset(&g_o, &H);
    static Rec r5;
    begin(&r5), waitev(&r5, &H, EQUAL, 0, 0x1300), mark(&r5, 4), end(&r5);
    sub1(5, 5, &r5);
    put_out("reset", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 0 && g_last_wait.writer == WO_WRITER_NONE);
    // a word nobody has named is not entered by a CPU write (a flush of 4 bytes), and is read when first named
    size_t words = g_o.word_count;
    Z = 9;
    wo_cpu_write(&g_o, &Z);
    CHECK(g_o.word_count == words);
    begin(&r4), waitev(&r4, &Z, EQUAL, 9, 0x1200), mark(&r4, 3), end(&r4);
    sub1(4, 4, &r4);
    put_out("unmet", take());
    CHECK(wo_parked(&g_o) == 0 && g_o.word_count == words + 1 && g_last_wait.seen == 9 && g_last_wait.writer == WO_WRITER_CPU);
    wo_end(&g_o);
    printf("ok order_pending\n");
}


// A queue's pending ADD dropped by a reset: the event is initialised again while a GPU ADD of the old one is replayed and
// not written. The write then lands in memory without the engine owing it (the old code took it off a count that was
// already zero, which wrapped below 0, and a CPU write of 5 was then read as 3).
static void case_order_reset_pending(void) {
    static volatile uint32_t H;
    static Rec r1, r2;
    eng_start();
    begin(&r1), sig(&r1, &H, ADD, 2), end(&r1);
    sub1(1, 1, &r1);
    wo_word_reset(&g_o, &H);
    H += 2;
    wo_signal_written(&g_o, &H, ADD, 2);
    H = 5;
    wo_cpu_write(&g_o, &H);
    begin(&r2), waitev(&r2, &H, EQUAL, 5, 0x1000), mark(&r2, 1), end(&r2);
    sub1(2, 2, &r2);
    put_out("reset-pending", take());
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.parked == 0 && g_last_wait.seen == 5);
    wo_end(&g_o);
    printf("ok order_reset_pending\n");
}

// An event initialised again over a word with a parked wait releases the wait when the word now holds its value.
static void case_order_reset_release(void) {
    static volatile uint32_t E = 9;
    static Rec r1;
    eng_start();
    begin(&r1), mark(&r1, 1), waitev(&r1, &E, EQUAL, 0, 0x1000), mark(&r1, 2), end(&r1);
    sub1(1, 1, &r1);
    CHECK(wo_parked(&g_o) == 1 && g_last_wait.seen == 0);
    put_out("held", take());
    E = 0;   // the storage is initialised again: no CPU write is reported for it
    wo_word_reset(&g_o, &E);
    put_out("released", take());
    CHECK(wo_parked(&g_o) == 0 && wo_whole_through(&g_o) == 1 && g_last_wait.parked == 1 && g_last_wait.seen == 0 &&
          g_last_wait.writer == WO_WRITER_NONE);
    wo_end(&g_o);
    printf("ok order_reset_release\n");
}

// The table of event words grows past 512 words (1024 slots, doubling at half full): a word met before the growth is still
// the same word after it.
#define MANY 700
static volatile uint32_t g_many[MANY];
static void case_order_words(void) {
    static Rec rs[MANY / 50], rq1, rq2;
    eng_start();
    for (int k = 0; k < MANY / 50; k++) {
        begin(&rs[k]);
        for (int i = 0; i < 50; i++) sig(&rs[k], &g_many[50 * k + i], WRITE, (uint32_t)(50 * k + i + 1));
        end(&rs[k]);
        sub1(1, (uint64_t)k + 1, &rs[k]);
    }
    printf("words %zu cap %zu\n", g_o.word_count, g_o.word_cap);
    CHECK(g_o.word_count == MANY && g_o.word_cap == 2048);
    // memory holds 0 everywhere: a word the table lost would be read from memory again and a wait for its value would park
    begin(&rq1), waitev(&rq1, &g_many[3], EQUAL, 4, 0x1000), end(&rq1);
    sub1(2, 100, &rq1);
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == 4 && g_last_wait.writer == WO_WRITER_GPU && g_last_wait.signaller == 1);
    begin(&rq2), waitev(&rq2, &g_many[MANY - 1], EQUAL, MANY, 0x1100), end(&rq2);
    sub1(2, 101, &rq2);
    CHECK(wo_parked(&g_o) == 0 && g_last_wait.seen == MANY && g_parked_calls == 0);
    wo_end(&g_o);
    printf("ok order_words\n");
}

// Submissions of several recordings: a wait in the second parks the queue there; a second submission of two recordings
// queues behind; two waits in the second recording are released one at a time, then the third recording and the queued
// submission follow. The stop query names the recording the wait is in.
static void case_order_recordings(void) {
    static volatile uint32_t E, F;
    static Rec a, b, c, d, e, s1, s2;
    eng_start();
    g_clock = 100;
    begin(&a), mark(&a, 1), end(&a);
    begin(&b), mark(&b, 2), waitev(&b, &E, EQUAL, 1, 0x1000), mark(&b, 3), waitev(&b, &F, EQUAL, 1, 0x1100), mark(&b, 4), end(&b);
    begin(&c), mark(&c, 5), end(&c);
    uint64_t h1[3] = {handle(&a), handle(&b), handle(&c)};
    wo_submit(&g_o, 1, 1, h1, 3);
    put_out("parked-in-second", take());
    WoStall st;
    g_clock = 100 + WO_STALL_NS + 1;
    CHECK(stalled(0, WO_STALL_NS, 0, 1, 0, &st) && st.index == 1 && st.wait.val == &E);
    begin(&d), mark(&d, 6), end(&d);
    begin(&e), mark(&e, 7), end(&e);
    uint64_t h2[2] = {handle(&d), handle(&e)};
    wo_submit(&g_o, 1, 2, h2, 2);
    CHECK(!take()[0] && wo_parked(&g_o) == 1);
    begin(&s1), sig(&s1, &E, WRITE, 1), end(&s1);
    begin(&s2), sig(&s2, &F, WRITE, 1), end(&s2);
    g_clock = 200;
    sub1(2, 3, &s1);
    put_out("first-release", take());
    CHECK(wo_parked(&g_o) == 1 && wo_whole_through(&g_o) == 0);
    g_clock = 200 + WO_STALL_NS + 1;
    CHECK(stalled(0, WO_STALL_NS, 0, 1, 0, &st) && st.index == 1 && st.wait.val == &F);
    sub1(2, 4, &s2);
    put_out("second-release", take());
    put_out("whole", whole_take());
    CHECK(wo_parked(&g_o) == 0 && wo_whole_through(&g_o) == 4);
    wo_end(&g_o);
    printf("ok order_recordings\n");
}

// ---------- the null back end through its table
static const GfxBackend *B(void) { return &gfx_null_backend; }
static void op_log(uint64_t fence, int index, const op_hdr *r, void *ctx) {
    (void)index, (void)ctx;
    switch (r->op) {
    case OP_HINT: logf_add("f%llu:m%d ", (ull)fence, ((const p_hint *)(const void *)(r + 1))->v[0]); break;
    case OP_SIGNAL_EVENT: logf_add("f%llu:S ", (ull)fence); break;
    case OP_WAIT_EVENT: logf_add("f%llu:W ", (ull)fence); break;
    default: break;
    }
}
static uint64_t nsubmit(uint64_t queue, Rec *r) {
    uint64_t h = handle(r);
    return B()->submit(queue, &h, 1);
}

// (a), (d) in the null back end: the replay order, the fence value done only in order, the results written in replay order
static void case_null_hold(void) {
    static volatile uint32_t E, F;
    static Rec r1, r2;
    log_clear();
    gfx_null_log_ops(op_log, NULL);
    begin(&r1), mark(&r1, 1), waitev(&r1, &E, EQUAL, 1, 0x1000), mark(&r1, 2), sig(&r1, &F, WRITE, 7), end(&r1);
    begin(&r2), sig(&r2, &E, WRITE, 1), mark(&r2, 3), end(&r2);
    gfx_null_hold(1);
    CHECK(nsubmit(1, &r1) == 1);
    put_out("first", take());
    CHECK(gfx_null_parked() == 1 && gfx_null_whole_through() == 0 && B()->fence_done() == 0 && E == 0 && F == 0);
    CHECK(nsubmit(2, &r2) == 2);
    put_out("second", take());
    CHECK(gfx_null_parked() == 0 && gfx_null_whole_through() == 2);
    // both are replayed, neither is complete: completion is held
    CHECK(B()->fence_done() == 0 && E == 0 && F == 0);
    // the hook completes what it is told to, up to fence value 1: fence value 1 alone (null_result_order has the writes
    // of a value that covers both, which follow replay order)
    CHECK(gfx_null_complete(1) == 1);
    printf("after-1 E %u F %u done %llu\n", E, F, (ull)B()->fence_done());
    CHECK(gfx_null_complete(2) == 2);
    printf("after-2 E %u F %u done %llu\n", E, F, (ull)B()->fence_done());
    printf("ok null_hold\n");
}

// A fence value behind a held wait is not done and a wait for it with a timeout answers 0; a present waits for the work
// before it; the CPU's signal, reported as guest_written of the word, releases the queue from another thread.
static volatile uint32_t g_cpu_word;
static void cpu_signaller(void *arg) {
    (void)arg;
    plat_sleep_ns(100000000u);
    __atomic_store_n(&g_cpu_word, 5, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)&g_cpu_word, 4);
}
static void case_null_cpu(void) {
    static Rec r1, r2;
    log_clear();
    gfx_null_log_ops(op_log, NULL);
    begin(&r1), mark(&r1, 1), waitev(&r1, &g_cpu_word, EQUAL, 5, 0x1000), mark(&r1, 2), end(&r1);
    begin(&r2), mark(&r2, 3), end(&r2);
    CHECK(nsubmit(1, &r1) == 1);
    CHECK(nsubmit(2, &r2) == 2);   // another queue: whole at once, and done though fence value 1 is not
    CHECK(gfx_null_parked() == 1 && gfx_null_whole_through() == 0 && B()->fence_done() == 0);
    printf("while-held done %llu whole-through %llu\n", (ull)B()->fence_done(), (ull)gfx_null_whole_through());
    CHECK(B()->fence_wait(1, 10000000u) == 0 && B()->fence_wait(2, 0) == 1);
    static Rec r3;
    begin(&r3), mark(&r3, 5), end(&r3);
    CHECK(nsubmit(1, &r3) == 3);   // behind the held wait of its own queue: not whole, not done
    CHECK(B()->fence_wait(3, 0) == 0 && gfx_null_whole_through() == 0);
    B()->present(NULL, (const int[4]){0, 0, 1, 1});   // after the held work: not shown yet
    printf("presents-held %llu\n", (ull)gfx_null_presents());
    CHECK(plat_thread_start(cpu_signaller, NULL, 1ull << 20));
    uint64_t t0 = plat_now_ns();
    CHECK(B()->fence_wait(3, UINT64_MAX) == 1);   // blocks until the CPU's signal from the other thread
    uint64_t waited = plat_now_ns() - t0;
    CHECK(waited >= 80000000u);
    CHECK(gfx_null_parked() == 0 && B()->fence_done() == 3);
    printf("presents-released %llu\n", (ull)gfx_null_presents());
    B()->present(NULL, (const int[4]){0, 0, 1, 1});   // nothing held: at once
    printf("presents-after %llu\n", (ull)gfx_null_presents());
    put_out("order", take());
    CHECK(B()->fence_wait(1, 0) == 1);
    printf("ok null_cpu\n");
}

// The classes by queue pair, and by who released
static volatile uint32_t g_ev[8];
static void case_null_classes(void) {
    static Rec r1, r2, r3, r4, r5, r6, r7;
    gfx_null_log_ops(NULL, NULL);
    volatile uint32_t *E = &g_ev[0], *F = &g_ev[1], *G = &g_ev[2], *H = &g_ev[3];
    begin(&r1), sig(&r1, E, WRITE, 1), waitev(&r1, E, EQUAL, 1, 0xA000), end(&r1);   // the queue's own signal
    begin(&r2), waitev(&r2, E, EQUAL, 1, 0xB000), end(&r2);                          // another queue's signal, before
    begin(&r3), waitev(&r3, F, EQUAL, 1, 0xC000), end(&r3);                          // held until another queue signals
    begin(&r4), sig(&r4, F, WRITE, 1), end(&r4);
    begin(&r5), waitev(&r5, G, EQUAL, 1, 0xD000), end(&r5);                          // held until the CPU signals
    begin(&r6), waitev(&r6, H, GEQUAL, 0, 0xE000), end(&r6);                         // the word's own value
    begin(&r7), sig(&r7, H, ADD, 2), end(&r7);
    volatile uint32_t *J = &g_ev[4];
    static Rec r8, r9, r10;
    begin(&r8), sig(&r8, J, WRITE, 1), end(&r8);                                    // queue 3's signal
    begin(&r9), waitev(&r9, J, EQUAL, 1, 0xF000), end(&r9);                         // queue 3's own signal, in order
    begin(&r10), waitev(&r10, J, EQUAL, 1, 0xF100), end(&r10);                      // queue 3's signal, awaited by queue 2
    nsubmit(1, &r1);
    nsubmit(2, &r2);
    nsubmit(3, &r3);
    nsubmit(1, &r4);
    nsubmit(4, &r5);
    nsubmit(5, &r6);
    nsubmit(6, &r7);
    nsubmit(3, &r8);   // queue 3 is released by r4 above
    nsubmit(3, &r9);
    nsubmit(2, &r10);
    GfxNullWaitClasses k;
    gfx_null_wait_classes(&k);
    printf("before-cpu gpu %llu (same %llu other %llu) cpu %llu initial %llu unsatisfied %llu later_gpu %llu later_cpu %llu "
           "pending %llu\n", (ull)k.by_gpu, (ull)k.by_gpu_same, (ull)k.by_gpu_other, (ull)k.by_cpu, (ull)k.initial,
           (ull)k.unsatisfied, (ull)k.later_gpu, (ull)k.later_cpu, (ull)k.pending);
    __atomic_store_n(G, 1, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)G, 4);
    gfx_null_wait_classes(&k);
    printf("after-cpu gpu %llu (same %llu other %llu) cpu %llu initial %llu unsatisfied %llu later_gpu %llu later_cpu %llu "
           "pending %llu\n", (ull)k.by_gpu, (ull)k.by_gpu_same, (ull)k.by_gpu_other, (ull)k.by_cpu, (ull)k.initial,
           (ull)k.unsatisfied, (ull)k.later_gpu, (ull)k.later_cpu, (ull)k.pending);
    uint64_t unsatisfied = 0, waits = gfx_null_wait_events(&unsatisfied);
    printf("waits %llu unsatisfied %llu\n", (ull)waits, (ull)unsatisfied);
    printf("ok null_classes\n");
}


// A thread that waits on one queue is not held by a wait on another, whose release is a CPU signal that the same thread
// makes next (gfx.h, rule 3): QueueFinish and a wait on a fence value of queue 2 return while queue 1 is parked.
static void case_null_cross_queue(void) {
    static volatile uint32_t E, F;
    static Rec r1, r2, r3;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 100000000u);   // a wait held behind the other queue would be a stop after 100 ms
    begin(&r1), waitev(&r1, &E, EQUAL, 5, 0x1000), end(&r1);
    begin(&r2), sig(&r2, &F, WRITE, 7), end(&r2);
    begin(&r3), mark(&r3, 1), end(&r3);
    CHECK(nsubmit(1, &r1) == 1);
    CHECK(nsubmit(2, &r2) == 2);
    CHECK(gfx_null_parked() == 1 && B()->fence_done() == 0);
    CHECK(B()->fence_wait(2, 0) == 1);
    CHECK(__atomic_load_n(&F, __ATOMIC_ACQUIRE) == 7);   // the CPU sees queue 2's signal while queue 1 is held
    B()->finish(2);
    CHECK(B()->fence_wait(2, UINT64_MAX) == 1);
    CHECK(nsubmit(2, &r3) == 3);
    B()->finish(2);
    printf("queue-2-done\n");
    // the thread makes the signal it was never waiting for
    __atomic_store_n(&E, 5, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)&E, 4);
    CHECK(gfx_null_parked() == 0 && B()->fence_done() == 3);
    B()->finish(0);
    printf("ok null_cross_queue\n");
}

// A wait with no timeout that begins long after the queue parked is not a stop at once: the time counts from the start of
// the wait (the park was 600 ms ago, the bound is 400 ms, the release comes 100 ms into the wait)
static volatile uint32_t g_late_word;
static void late_signaller(void *arg) {
    (void)arg;
    plat_sleep_ns(100000000u);
    __atomic_store_n(&g_late_word, 5, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)&g_late_word, 4);
}
static void case_null_late_wait(void) {
    static Rec r1;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 400000000u);
    begin(&r1), waitev(&r1, &g_late_word, EQUAL, 5, 0x1000), end(&r1);
    CHECK(nsubmit(1, &r1) == 1);
    plat_sleep_ns(600000000u);
    CHECK(plat_thread_start(late_signaller, NULL, 1ull << 20));
    CHECK(B()->fence_wait(1, UINT64_MAX) == 1);
    CHECK(gfx_null_parked() == 0);
    printf("ok null_late_wait\n");
}

// A parked queue that the wait does not depend on is no stop: queue 2's work is whole and waits for completion, which is
// held, for 300 ms, while queue 1 has been parked past the bound of 100 ms
static void late_release(void *arg) {
    (void)arg;
    plat_sleep_ns(300000000u);
    gfx_null_hold(0);
}
static void case_null_unrelated_park(void) {
    static volatile uint32_t E;
    static Rec r1, r2;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 100000000u);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    begin(&r2), mark(&r2, 1), end(&r2);
    nsubmit(1, &r1);
    gfx_null_hold(1);
    CHECK(nsubmit(2, &r2) == 2);
    CHECK(B()->fence_wait(2, 0) == 0);   // whole, not complete
    CHECK(plat_thread_start(late_release, NULL, 1ull << 20));
    CHECK(B()->fence_wait(2, UINT64_MAX) == 1);
    CHECK(gfx_null_parked() == 1);
    B()->finish(2);
    printf("ok null_unrelated_park\n");
}

// The stops, by name: a wait nothing releases
static void case_null_stop_submits(void) {
    static volatile uint32_t E;
    static Rec r1, rm;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(8, 0);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    begin(&rm), mark(&rm, 1), end(&rm);
    nsubmit(1, &r1);
    for (int i = 0; i < 8; i++) nsubmit(2, &rm);   // eight further submissions are within the bound
    printf("within\n");
    nsubmit(2, &rm);   // the ninth is not
    printf("not stopped\n");
}
static void case_null_stop_time(void) {
    static volatile uint32_t E;
    static Rec r1;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 100000000u);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    nsubmit(1, &r1);
    printf("waiting\n");
    B()->fence_wait(1, UINT64_MAX);
    printf("not stopped\n");
}
static void case_null_stop_finish(void) {
    static volatile uint32_t E;
    static Rec r1;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 100000000u);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    nsubmit(1, &r1);
    printf("finishing\n");
    B()->finish(1);
    printf("not stopped\n");
}
// A wait with a timeout answers 0 when it passes, past the time bound too: only a wait with no timeout is a stop
static void case_null_timeout(void) {
    static volatile uint32_t E;
    static Rec r1;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 50000000u);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    nsubmit(1, &r1);
    uint64_t t0 = plat_now_ns();
    CHECK(B()->fence_wait(1, 200000000u) == 0);
    CHECK(plat_now_ns() - t0 >= 190000000u);
    printf("ok null_timeout\n");
}


// Results are written in the order replay finished the submissions, not in fence order: queue 1 (fence value 1) waits for E,
// then writes X = 7; queue 2 (fence value 2) writes X = 3 and then signals E. The console ends at X = 7 (queue 1's write
// follows E, which follows X = 3). Once with completion at once, once with completion held and then asked for, up to
// fence value 4, which covers both: queue 2's results are written first, as replay finished it first.
static volatile uint32_t g_x1, g_e1, g_x2, g_e2;
static void case_null_result_order(void) {
    static Rec r1, r2, r3, r4;
    gfx_null_log_ops(NULL, NULL);
    begin(&r1), waitev(&r1, &g_e1, EQUAL, 1, 0x1000), sig(&r1, &g_x1, WRITE, 7), end(&r1);
    begin(&r2), sig(&r2, &g_x1, WRITE, 3), sig(&r2, &g_e1, WRITE, 1), end(&r2);
    CHECK(nsubmit(1, &r1) == 1);
    CHECK(nsubmit(2, &r2) == 2);
    CHECK(B()->fence_done() == 2);
    printf("released x %u e %u done %llu\n", g_x1, g_e1, (ull)B()->fence_done());
    begin(&r3), waitev(&r3, &g_e2, EQUAL, 1, 0x1100), sig(&r3, &g_x2, WRITE, 7), end(&r3);
    begin(&r4), sig(&r4, &g_x2, WRITE, 3), sig(&r4, &g_e2, WRITE, 1), end(&r4);
    gfx_null_hold(1);
    CHECK(nsubmit(1, &r3) == 3);
    CHECK(nsubmit(2, &r4) == 4);
    CHECK(g_x2 == 0 && g_e2 == 0);
    CHECK(gfx_null_complete(4) == 4);
    printf("held x %u e %u done %llu\n", g_x2, g_e2, (ull)B()->fence_done());
    gfx_null_hold(0);
    printf("ok null_result_order\n");
}

// What a submission kept before a wait that parks its queue is written at the park, not when the whole submission
// completes: queue 1 writes X = 1 and waits for E, which nobody signals; queue 2 waits for X = 1 and writes Y = 1. The CPU
// sees X and Y. With completion held nothing is written at a park.
static volatile uint32_t g_px, g_pe, g_py, g_qx, g_qe;
static void case_null_park_writes(void) {
    static Rec r1, r2, r3;
    gfx_null_log_ops(NULL, NULL);
    begin(&r1), sig(&r1, &g_px, WRITE, 1), waitev(&r1, &g_pe, EQUAL, 1, 0x1000), end(&r1);
    begin(&r2), waitev(&r2, &g_px, EQUAL, 1, 0x1100), sig(&r2, &g_py, WRITE, 1), end(&r2);
    CHECK(nsubmit(1, &r1) == 1);
    CHECK(__atomic_load_n(&g_px, __ATOMIC_ACQUIRE) == 1);   // written at the park
    CHECK(nsubmit(2, &r2) == 2);
    printf("x %u y %u parked %d done %llu\n", g_px, g_py, gfx_null_parked(), (ull)B()->fence_done());
    CHECK(B()->fence_wait(2, 0) == 1 && B()->fence_wait(1, 0) == 0);
    begin(&r3), sig(&r3, &g_qx, WRITE, 1), waitev(&r3, &g_qe, EQUAL, 1, 0x1200), end(&r3);
    gfx_null_hold(1);
    CHECK(nsubmit(3, &r3) == 3);
    printf("held x %u parked %d\n", g_qx, gfx_null_parked());
    CHECK(g_qx == 0);
    gfx_null_hold(0);
    printf("ok null_park_writes\n");
}

// finish(0) is every queue: two queues held at once, released out of order (the later fence value first), and the finish
// returns only when both are done.
static volatile uint32_t g_fa, g_fc;
static void finish_releaser(void *arg) {
    (void)arg;
    plat_sleep_ns(100000000u);
    __atomic_store_n(&g_fc, 1, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)&g_fc, 4);
    plat_sleep_ns(200000000u);
    __atomic_store_n(&g_fa, 1, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)&g_fa, 4);
}
static void case_null_finish_all(void) {
    static Rec r1, r2;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(0, 3000000000u);
    begin(&r1), waitev(&r1, &g_fa, EQUAL, 1, 0x1000), end(&r1);
    begin(&r2), waitev(&r2, &g_fc, EQUAL, 1, 0x1100), end(&r2);
    CHECK(nsubmit(1, &r1) == 1);
    CHECK(nsubmit(2, &r2) == 2);
    CHECK(plat_thread_start(finish_releaser, NULL, 1ull << 20));
    uint64_t t0 = plat_now_ns();
    B()->finish(0);
    uint64_t waited = plat_now_ns() - t0;
    printf("finish(0) parked %d done %llu\n", gfx_null_parked(), (ull)B()->fence_done());
    CHECK(waited >= 250000000u && gfx_null_parked() == 0 && B()->fence_done() == 2);
    printf("ok null_finish_all\n");
}

// A wait behind a parked wait of its own queue is judged after the release and counted as such, not as a signal "before":
// queue 2's wait on a word the CPU wrote is by_cpu; queue 1 parks at E, with behind it a wait on a word that holds its
// value, one on the word the CPU wrote, and, in a second submission queued behind, one more.
static volatile uint32_t g_ar[4];
static void case_null_after_release(void) {
    static Rec r0, r1, r2;
    gfx_null_log_ops(NULL, NULL);
    volatile uint32_t *E = &g_ar[0], *G = &g_ar[1], *H = &g_ar[2];
    __atomic_store_n(H, 5, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)H, 4);
    begin(&r0), waitev(&r0, H, EQUAL, 5, 0xA000), end(&r0);
    begin(&r1), waitev(&r1, E, EQUAL, 1, 0xB000), waitev(&r1, G, GEQUAL, 0, 0xB100), waitev(&r1, H, EQUAL, 5, 0xB200), end(&r1);
    begin(&r2), waitev(&r2, G, GEQUAL, 0, 0xB300), end(&r2);
    nsubmit(2, &r0);
    nsubmit(1, &r1);
    nsubmit(1, &r2);
    GfxNullWaitClasses k;
    gfx_null_wait_classes(&k);
    printf("held cpu %llu initial %llu after_release %llu unsatisfied %llu later_cpu %llu pending %llu\n", (ull)k.by_cpu,
           (ull)k.initial, (ull)k.after_release, (ull)k.unsatisfied, (ull)k.later_cpu, (ull)k.pending);
    __atomic_store_n(E, 1, __ATOMIC_RELEASE);
    B()->guest_written((uint64_t)(uintptr_t)E, 4);
    gfx_null_wait_classes(&k);
    printf("released cpu %llu initial %llu after_release %llu unsatisfied %llu later_cpu %llu pending %llu\n", (ull)k.by_cpu,
           (ull)k.initial, (ull)k.after_release, (ull)k.unsatisfied, (ull)k.later_cpu, (ull)k.pending);
    uint64_t unsatisfied = 0, waits = gfx_null_wait_events(&unsatisfied);
    printf("waits %llu unsatisfied %llu\n", (ull)waits, (ull)unsatisfied);
    printf("ok null_after_release\n");
}

// The default bounds, with no gfx_null_wait_bounds call: 1024 further submissions are within, the 1025th is a stop; and a
// wait with no timeout for the held queue's work stops after WO_STALL_NS (10 s).
static void case_null_stop_default_submits(void) {
    static volatile uint32_t E;
    static Rec r1, rm;
    gfx_null_log_ops(NULL, NULL);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    begin(&rm), mark(&rm, 1), end(&rm);
    nsubmit(1, &r1);
    for (int i = 0; i < (int)WO_BOUND_SUBMITS; i++) nsubmit(2, &rm);
    printf("within\n");
    nsubmit(2, &rm);
    printf("not stopped\n");
}
static void case_null_stop_default_time(void) {
    static volatile uint32_t E;
    static Rec r1;
    gfx_null_log_ops(NULL, NULL);
    begin(&r1), waitev(&r1, &E, EQUAL, 1, 0x1000), end(&r1);
    nsubmit(1, &r1);
    printf("waiting\n");
    B()->fence_wait(1, UINT64_MAX);
    printf("not stopped\n");
}

// A wait in a recording above index 0: the stop names recording 1
static void case_null_stop_recording(void) {
    static volatile uint32_t E;
    static Rec r1, r2, rm;
    gfx_null_log_ops(NULL, NULL);
    gfx_null_wait_bounds(8, 0);
    begin(&r1), mark(&r1, 1), end(&r1);
    begin(&r2), mark(&r2, 2), waitev(&r2, &E, EQUAL, 1, 0x1000), end(&r2);
    begin(&rm), mark(&rm, 3), end(&rm);
    uint64_t h[2] = {handle(&r1), handle(&r2)};
    B()->submit(1, h, 2);
    for (int i = 0; i < 8; i++) nsubmit(2, &rm);
    printf("within\n");
    nsubmit(2, &rm);
    printf("not stopped\n");
}

static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(20000000000ull);
    fprintf(stderr, "hung\n");
    _Exit(3);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: nvn_order_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    static const struct { const char *name; void (*fn)(void); } cases[] = {
        {"order_a", case_order_a}, {"order_b", case_order_b}, {"order_cascade", case_order_cascade}, {"order_transient", case_order_transient},
        {"order_c", case_order_c}, {"order_d", case_order_d}, {"null_hold", case_null_hold}, {"null_cpu", case_null_cpu},
        {"null_classes", case_null_classes}, {"null_stop_submits", case_null_stop_submits},
        {"null_stop_time", case_null_stop_time}, {"null_stop_finish", case_null_stop_finish},
        {"null_timeout", case_null_timeout}, {"order_add", case_order_add}, {"order_pending", case_order_pending},
        {"null_cross_queue", case_null_cross_queue}, {"null_late_wait", case_null_late_wait},
        {"null_unrelated_park", case_null_unrelated_park}, {"order_reset_pending", case_order_reset_pending},
        {"order_reset_release", case_order_reset_release}, {"order_words", case_order_words},
        {"order_recordings", case_order_recordings}, {"null_result_order", case_null_result_order},
        {"null_park_writes", case_null_park_writes}, {"null_finish_all", case_null_finish_all},
        {"null_after_release", case_null_after_release}, {"null_stop_default_submits", case_null_stop_default_submits},
        {"null_stop_default_time", case_null_stop_default_time}, {"null_stop_recording", case_null_stop_recording},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(argv[1], cases[i].name)) {
            cases[i].fn();
            return 0;
        }
    fprintf(stderr, "unknown case %s\n", argv[1]);
    return 2;
}
