// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the audio renderer and its services (runtime/audio, runtime/hos/srv_audio.c): a synthetic guest thread
// written as a host function opens a renderer through aud:u and audren:u over the in-process IPC, builds the update blocks
// the SDK's client would build (tests/runtime/audren_blocks.h, layouts as numbers), and reads the blocks back. The frame
// clock is driven by the test: time moves only when the test moves it, so a wave buffer's release and the frame event's
// cadence are pinned to the instant. No game code is involved.
//
//   audio_test CASE DIR   runs one case with DIR as the game folder; prints "ok CASE" and exits 0, or exits 1 with a FAIL
//                         line. The cases that end in a named fatal exit with the runtime's code
//                         (tests/test_runtime_audio.py).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>   // _exit
#endif

#include "layout.h"
#include "endmark.h"
#include "oot2/audio.h"
#include "oot2/audio_sink.h"
#include "opus.h"
#include "audren_state.h"   // the renderer's lock, for the case that holds it
#include "stall.h"
#include "oot2/boot.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"
#include "av1_fake.h"   // runtime/hos links the AV1 service; the fake stands for the decoder
#include "../../runtime/hos/hos.h"
#include "../../runtime/hos/ipc.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))
#define PC 0x7000000040ull
#define CL_PC PC
#include "cmif_client.h"
#include "audren_blocks.h"

#define R_TIMED_OUT 0xEA01u

static uint64_t HEAP;
static const char *g_case;

// ---- the driven clock: time moves when the test says so, and every waiter wakes to look
static PlatMutex g_tm_lock = PLAT_MUTEX_INIT;
static PlatCond g_tm_cond;
static uint64_t g_tm_now = 1000000000ull;
static unsigned g_tm_wake;
static uint64_t tm_now(void *ctx) {
    (void)ctx;
    plat_mutex_lock(&g_tm_lock);
    uint64_t t = g_tm_now;
    plat_mutex_unlock(&g_tm_lock);
    return t;
}
static void tm_wait_until(void *ctx, uint64_t deadline) {
    (void)ctx;
    plat_mutex_lock(&g_tm_lock);
    unsigned gen = g_tm_wake;
    while (g_tm_now < deadline && g_tm_wake == gen) plat_cond_wait(&g_tm_cond, &g_tm_lock);
    plat_mutex_unlock(&g_tm_lock);
}
static void tm_wake(void *ctx) {
    (void)ctx;
    plat_mutex_lock(&g_tm_lock);
    g_tm_wake++;
    plat_cond_broadcast(&g_tm_cond);
    plat_mutex_unlock(&g_tm_lock);
}
static void tm_set(uint64_t t) {
    plat_mutex_lock(&g_tm_lock);
    g_tm_now = t;
    plat_cond_broadcast(&g_tm_cond);
    plat_mutex_unlock(&g_tm_lock);
}
// tick k of a renderer of count samples at rate Hz, from the origin: k x count x 1e9 / rate rounded up
static uint64_t tick_ns(uint64_t rate, uint64_t count, uint64_t k) { return (k * count * 1000000000ull + rate - 1) / rate; }
// wait (in real time) until the renderer has signalled n frames; false after 5 s
static int await_signals(uint64_t n) {
    for (int i = 0; i < 50000; i++) {
        AudioPosition p;
        audio_position(&p);
        if (p.signals >= n) return 1;
        plat_sleep_ns(100000);
    }
    return 0;
}
static void settle(void) { plat_sleep_ns(40000000ull); }   // long enough for a wrongly early frame to show

// ---- guest helpers
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
static uint32_t open_sm(Cpu *c) {
    memcpy((void *)(uintptr_t)(HEAP + 0x100), "sm:", 4);
    CHECK(sv(c, 0x1F, 0, HEAP + 0x100, 0, 0) == 0);
    uint32_t h = (uint32_t)c->x[1];
    ClReq rq = {.tipc = 1, .cmd = 0, .pid = 1};
    ClRep rp;
    CHECK(cl_send(c, h, &rq, &rp, 0) == 0 && rp.result == 0);
    return h;
}
static uint32_t get_service(Cpu *c, uint32_t sm, const char *name) {
    uint64_t n = 0;
    memcpy(&n, name, strlen(name));
    ClReq rq = {.tipc = 1, .cmd = 1, .data = &n, .size = 8};
    ClRep rp;
    CHECK(cl_send(c, sm, &rq, &rp, 0) == 0 && rp.result == 0 && rp.nmove == 1);
    return rp.move[0];
}
static uint32_t call(Cpu *c, uint32_t h, uint32_t cmd, const void *data, uint32_t size, ClRep *rp, uint32_t expect) {
    ClReq rq = {.cmd = cmd, .data = data, .size = size};
    CHECK(cl_send(c, h, &rq, rp, expect) == 0);
    return rp->result;
}
static void close_handle(Cpu *c, uint32_t h) { CHECK(sv(c, 0x16, h, 0, 0, 0) == 0); }
// the event, seen from the guest: 0 when signalled (and cleared, as the SDK clears an auto-clear event), else the timeout
static uint64_t poll_event(Cpu *c, uint32_t readable) {
    memcpy((void *)(uintptr_t)(HEAP + 0x380), &readable, 4);
    uint64_t r = sv(c, 0x18, 0, HEAP + 0x380, 1, 0);
    if (r == 0) CHECK(sv(c, 0x12, readable, 0, 0, 0) == 0);
    return r;
}

// ---- opening a renderer through the services
typedef struct TestCfg {
    int32_t rate, count, voices, effects, sinks, submix;
    uint32_t tag;
    uint8_t device, mode;
    int32_t splitters, dests;
} TestCfg;
static TestCfg cfg_default(void) { return (TestCfg){48000, 240, 4, 1, 1, 1, BK_TAG(16), 0, 0, 0, 0}; }
static void put_param(uint8_t *p, const TestCfg *t) {
    memset(p, 0, 0x34);
    bk_u32(p + 0, (uint32_t)t->rate);
    bk_u32(p + 4, (uint32_t)t->count);
    bk_u32(p + 8, 8);                       // mix buffers
    bk_u32(p + 12, (uint32_t)t->submix);
    bk_u32(p + 16, (uint32_t)t->voices);
    bk_u32(p + 20, (uint32_t)t->sinks);
    bk_u32(p + 24, (uint32_t)t->effects);
    p[34] = t->device, p[35] = t->mode;
    bk_u32(p + 36, (uint32_t)t->splitters);
    bk_u32(p + 40, (uint32_t)t->dests);
    bk_u32(p + 48, t->tag);
}
typedef struct Rend { uint32_t sm, aud, mgr, session, event; } Rend;
static Rend rend_open(Cpu *c, const TestCfg *t) {
    Rend r = {0};
    r.sm = open_sm(c);
    r.aud = get_service(c, r.sm, "aud:u");
    ClRep rp;
    CHECK(call(c, r.aud, 2, NULL, 0, &rp, 0) == 0 && rp.nmove == 1);   // GetAudioRendererManager
    r.mgr = rp.move[0];
    uint8_t arg[0x48] = {0};
    put_param(arg, t);
    CHECK(call(c, r.mgr, 1, arg, 0x34, &rp, 8) == 0);                  // GetWorkBufferSize
    CHECK(bk_rd64(rp.data) == 0x10000);
    uint64_t work = 0x10000;
    memcpy(arg + 0x38, &work, 8);
    CHECK(sv(c, 0x15, 0, HEAP + 0x300000, 0x10000, 3) == 0);          // the work buffer's transfer memory
    uint32_t tm = (uint32_t)c->x[1];
    ClReq rq = {.cmd = 0, .data = arg, .size = 0x48, .ncopy = 2};
    rq.copy[0] = tm, rq.copy[1] = 0xFFFF8001u;                         // transfer memory, the process
    CHECK(cl_send(c, r.mgr, &rq, &rp, 0) == 0 && rp.result == 0 && rp.nmove == 1);
    r.session = rp.move[0];
    return r;
}
static uint32_t rend_event(Cpu *c, const Rend *r) {
    ClRep rp;
    CHECK(call(c, r->session, 7, NULL, 0, &rp, 0) == 0 && rp.ncopy == 1);   // QuerySystemEvent
    return rp.copy[0];
}
static void rend_close(Cpu *c, Rend *r) {
    close_handle(c, r->session);
    close_handle(c, r->mgr);
    close_handle(c, r->aud);
    close_handle(c, r->sm);
}
// RequestUpdate (command 4, map-alias buffers) of block b: the answer read into a. Returns the service's result.
#define IN_AT (HEAP + 0x200000)
#define OUT_AT (HEAP + 0x240000)
static uint32_t rend_update(Cpu *c, const Rend *r, const BkBlock *b, BkAnswer *a, uint32_t out_size) {
    uint32_t n = bk_build((uint8_t *)(uintptr_t)IN_AT, b);
    memset((void *)(uintptr_t)OUT_AT, 0xee, out_size);
    ClReq rq = {.cmd = 4, .na = 1, .nb = 1};
    rq.a[0] = (ClBuf){IN_AT, n, 1};
    rq.b[0] = (ClBuf){OUT_AT, out_size, 1};
    ClRep rp;
    CHECK(cl_send(c, r->session, &rq, &rp, 0) == 0);
    if (rp.result == 0) bk_read((const uint8_t *)(uintptr_t)OUT_AT, b, a);
    return rp.result;
}
static void rend_start(Cpu *c, const Rend *r) {
    ClRep rp;
    CHECK(call(c, r->session, 5, NULL, 0, &rp, 0) == 0);
}
static void rend_stop(Cpu *c, const Rend *r) {
    ClRep rp;
    CHECK(call(c, r->session, 6, NULL, 0, &rp, 0) == 0);
}

// a block for the renderer's voices, none in use until voice_play fills one
static void block_new(BkBlock *b, const TestCfg *t) {
    memset(b, 0, sizeof *b);
    b->tag = t->tag;
    b->rev = audio_revision_number(t->tag);
    b->voices = t->voices;
    for (int i = 0; i < t->voices; i++) b->voice[i].id = (uint32_t)i;   // a record of a voice not in use
    b->effects = t->effects;
    b->sinks = t->sinks;
    b->mixes = t->submix + 1;
}
static void voice_play(BkVoice *v, uint32_t id, uint32_t rate, const uint32_t *lengths, int nbuffers) {
    memset(v, 0, sizeof *v);
    v->id = id, v->is_new = 1, v->in_use = 1, v->play_state = 0, v->format = 2, v->sample_rate = rate;
    v->pitch = 1.0f, v->volume = 1.0f;
    for (int i = 0; i < 4; i++) {
        if (i >= nbuffers) {   // a slot nobody filled: the client marks it as sent already, so it is never queued
            v->wave[i].sent = 1;
            continue;
        }
        v->wave[i] = (BkWave){.address = 0x10000000ull + 0x10000ull * id + 0x1000ull * (uint32_t)i, .size = 2ull * lengths[i],
                              .start = 0, .end = (int32_t)lengths[i]};
    }
}
static void block_resend(BkBlock *b) {   // the next block: nothing is new, every buffer was sent already
    for (int i = 0; i < b->voices; i++) {
        b->voice[i].is_new = 0;
        for (int k = 0; k < 4; k++) b->voice[i].wave[k].sent = 1;
    }
}

// ---- cases
// A wave buffer is released when its samples have been consumed at the renderer's rate: N samples take N / rate seconds on
// the driven clock, not earlier, not later. Voice 0 plays 2,400 samples at 48,000 Hz (50 ms), voice 1 plays 1,600 samples at
// its own rate of 32,000 Hz (50 ms), voice 2 plays 2,400 samples and then 480 (10 ms more), voice 3 is never started.
static void g_release_clock(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    voice_play(&b.voice[0], 0, 48000, (uint32_t[]){2400}, 1);
    voice_play(&b.voice[1], 1, 32000, (uint32_t[]){1600}, 1);
    voice_play(&b.voice[2], 2, 48000, (uint32_t[]){2400, 480}, 2);
    voice_play(&b.voice[3], 3, 48000, (uint32_t[]){2400}, 1);
    b.voice[3].play_state = 1;   // stopped: it never plays
    BkAnswer a;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);   // before Start
    CHECK(a.consumed[0] == 0 && a.consumed[1] == 0 && a.consumed[2] == 0 && a.consumed[3] == 0 && a.elapsed_frames == 0);
    uint32_t ev = rend_event(c, &r);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    block_resend(&b);
    for (uint64_t k = 1; k < 10; k++) {   // frames 1 to 9, one at a time
        tm_set(origin + tick_ns(48000, 240, k));
        CHECK(await_signals(k));
    }
    tm_set(origin + tick_ns(48000, 240, 10) - 1);   // 1 ns before the tenth frame is due
    settle();
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.elapsed_frames == 9);
    CHECK(a.consumed[0] == 0 && a.consumed[1] == 0 && a.consumed[2] == 0);
    CHECK(a.played[0] == 9 * 240 && a.played[1] == 9 * 160);   // samples of the voice's own rate
    tm_set(origin + tick_ns(48000, 240, 10));
    CHECK(await_signals(10));
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.elapsed_frames == 10);
    CHECK(a.consumed[0] == 1 && a.consumed[1] == 1 && a.consumed[2] == 1 && a.consumed[3] == 0);
    // the second buffer of voice 2 follows at its own length: 480 samples, 10 ms, the twelfth frame
    tm_set(origin + tick_ns(48000, 240, 11));
    CHECK(await_signals(11));
    tm_set(origin + tick_ns(48000, 240, 12) - 1);
    settle();
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.consumed[2] == 1);
    tm_set(origin + tick_ns(48000, 240, 12));
    CHECK(await_signals(12));
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.consumed[0] == 1 && a.consumed[1] == 1 && a.consumed[2] == 2 && a.consumed[3] == 0);
    CHECK(poll_event(c, ev) == 0);   // the frame event was signalled, and the guest cleared it
    close_handle(c, ev);
    rend_stop(c, &r);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The same at a renderer's rate of 32,000 Hz with 160 samples a frame (5 ms): a voice of the renderer's own rate releases a
// 1,600-sample buffer at the tenth frame as well.
static void g_release_clock_32k(Cpu *c) {
    TestCfg t = cfg_default();
    t.rate = 32000, t.count = 160, t.voices = 1;
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    voice_play(&b.voice[0], 0, 32000, (uint32_t[]){1600}, 1);
    BkAnswer a;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    block_resend(&b);
    for (uint64_t k = 1; k < 10; k++) {
        tm_set(origin + tick_ns(32000, 160, k));
        CHECK(await_signals(k));
    }
    tm_set(origin + tick_ns(32000, 160, 10) - 1);
    settle();
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0 && a.consumed[0] == 0);
    tm_set(origin + tick_ns(32000, 160, 10));
    CHECK(await_signals(10));
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0 && a.consumed[0] == 1);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// One signal a frame while the renderer runs; none before Start, none after Stop, none after Close.
static void g_cadence_running_stopped(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    uint32_t ev = rend_event(c, &r);
    AudioPosition p;
    uint64_t t0 = tm_now(NULL);
    tm_set(t0 + 10 * tick_ns(48000, 240, 1));   // ten frames' time before Start: nothing
    settle();
    audio_position(&p);
    CHECK(p.signals == 0 && p.frames == 0 && !p.running);
    CHECK(poll_event(c, ev) == R_TIMED_OUT);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    for (uint64_t k = 1; k <= 20; k++) {
        tm_set(origin + tick_ns(48000, 240, k) - 1);
        if (k == 1 || k == 2 || k == 20) settle();   // a frame that came early would show by now
        audio_position(&p);
        CHECK(p.signals == k - 1);   // not before the due time
        tm_set(origin + tick_ns(48000, 240, k));
        CHECK(await_signals(k));
        audio_position(&p);
        CHECK(p.signals == k && p.frames == k && p.skipped == 0 && p.running);
        CHECK(poll_event(c, ev) == 0);   // one signal: signalled, and cleared by the poll
        CHECK(poll_event(c, ev) == R_TIMED_OUT);
    }
    rend_stop(c, &r);
    tm_set(origin + tick_ns(48000, 240, 40));   // twenty frames' time after Stop: nothing
    settle();
    audio_position(&p);
    CHECK(p.signals == 20 && p.frames == 20 && !p.running);
    CHECK(poll_event(c, ev) == R_TIMED_OUT);
    rend_start(c, &r);   // a restart takes a new origin
    uint64_t origin2 = tm_now(NULL);
    tm_set(origin2 + tick_ns(48000, 240, 1) - 1);
    settle();
    audio_position(&p);
    CHECK(p.signals == 20);
    tm_set(origin2 + tick_ns(48000, 240, 1));
    CHECK(await_signals(21));
    rend_close(c, &r);
    tm_set(origin2 + tick_ns(48000, 240, 60));   // closed: nothing, and the guest's end of the event is still a handle
    settle();
    audio_position(&p);
    CHECK(p.signals == 21 && p.frames == 21);
    CHECK(poll_event(c, ev) == 0);   // the signal that came before the close, not yet cleared
    CHECK(poll_event(c, ev) == R_TIMED_OUT);
    close_handle(c, ev);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The interval between frames follows sample_count: 5 ms for 240 samples at 48,000 Hz, 10 ms for 480, 5 ms for 160 at 32,000.
static void period_check(Cpu *c, int32_t rate, int32_t count, uint64_t expect_ns) {
    TestCfg t = cfg_default();
    t.rate = rate, t.count = count;
    Rend r = rend_open(c, &t);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    AudioPosition p;
    for (uint64_t k = 1; k <= 3; k++) {
        tm_set(origin + k * expect_ns - 1);
        settle();
        audio_position(&p);
        CHECK(p.frames == k - 1);
        tm_set(origin + k * expect_ns);
        CHECK(await_signals(k));
        audio_position(&p);
        CHECK(p.frames == k);
    }
    rend_close(c, &r);
}
static void g_cadence_period(Cpu *c) {
    period_check(c, 48000, 240, 5000000ull);
    period_check(c, 48000, 480, 10000000ull);
    period_check(c, 32000, 160, 5000000ull);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A wake-up a whole frame late or more renders one frame, counts the frames it passed, and the next frame is on the
// original grid: no burst.
static void g_cadence_no_burst(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    AudioPosition p;
    tm_set(origin + tick_ns(48000, 240, 1));
    CHECK(await_signals(1));
    tm_set(origin + tick_ns(48000, 240, 8));   // seven frames in one step
    CHECK(await_signals(2));
    settle();
    audio_position(&p);
    CHECK(p.signals == 2 && p.frames == 2 && p.skipped == 6);
    tm_set(origin + tick_ns(48000, 240, 9) - 1);
    settle();
    audio_position(&p);
    CHECK(p.signals == 2 && p.skipped == 6);   // the next frame is the ninth, on the original grid
    tm_set(origin + tick_ns(48000, 240, 9));
    CHECK(await_signals(3));
    audio_position(&p);
    CHECK(p.frames == 3 && p.skipped == 6);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The sections of a good block, in revisions 16, 15 and 14, both spellings of the tag, the tag echoed as it came, the memory
// pools answered, the effect status, and the same blocks sent through the pointer buffers (command 10).
static void g_parse_ok(Cpu *c) {
    static const uint32_t tags[] = {BK_TAG(16), 0x40764552u /* "REv@" */, BK_TAG(15), BK_TAG(14), BK_TAG(9)};
    for (unsigned i = 0; i < sizeof tags / sizeof *tags; i++) {
        TestCfg t = cfg_default();
        t.tag = tags[i], t.voices = 3;
        Rend r = rend_open(c, &t);
        BkBlock b;
        block_new(&b, &t);
        voice_play(&b.voice[0], 0, 48000, (uint32_t[]){240}, 1);
        b.pool_state[0] = 4;   // request attach
        b.pool_state[1] = 2;   // request detach
        b.pool_state[2] = 5;   // attached already
        BkAnswer a;
        CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
        CHECK(a.tag == tags[i]);
        CHECK(a.pool_state[0] == 5 && a.pool_state[1] == 3 && a.pool_state[2] == 5);   // attached, detached, attached
        CHECK(a.effect_status[0] == 4);                                                  // disabled
        CHECK(a.elapsed_frames == 0);
        uint32_t eo = b.rev >= 9 ? 0x90 : 0x10;
        uint32_t want = 0x40 + (uint32_t)(b.effects + 4 * b.voices) * 0x10 + (uint32_t)b.voices * 0x10 + (uint32_t)b.effects * eo +
                        (uint32_t)b.sinks * 0x20 + 0x10 + 0xb0 + 0x10;
        CHECK(a.total == want);
        // the same through RequestUpdateAuto with pointer buffers: input X, output C
        uint32_t n = bk_build((uint8_t *)(uintptr_t)IN_AT, &b);
        memset((void *)(uintptr_t)OUT_AT, 0xee, 0x2000);
        ClReq rq = {.cmd = 10, .nx = 1, .nc = 1};
        rq.x[0] = (ClBuf){IN_AT, n, 0};
        rq.c[0] = (ClBuf){OUT_AT, 0x2000, 0};
        ClRep rp;
        CHECK(cl_send(c, r.session, &rq, &rp, 0) == 0 && rp.result == 0);
        CHECK(rp.nx == 1 && rp.x[0].addr == OUT_AT && rp.x[0].size == 0x2000);
        bk_read((const uint8_t *)(uintptr_t)OUT_AT, &b, &a);
        CHECK(a.tag == tags[i] && a.total == want);
        rend_close(c, &r);
    }
    // the renderer's plain queries
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    ClRep rp;
    CHECK(call(c, r.session, 0, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 48000);   // GetSampleRate
    CHECK(call(c, r.session, 1, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 240);     // GetSampleCount
    CHECK(call(c, r.session, 2, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 8);       // GetMixBufferCount
    CHECK(call(c, r.session, 3, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 1);       // GetState: stopped
    rend_start(c, &r);
    CHECK(call(c, r.session, 3, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 0);       // started
    rend_stop(c, &r);
    float drop = 0.25f;
    CHECK(call(c, r.session, 12, &drop, 4, &rp, 0) == 0);
    CHECK(call(c, r.session, 13, NULL, 0, &rp, 4) == 0 && !memcmp(rp.data, &drop, 4));
    CHECK(call(c, r.session, 14, &(uint8_t){1}, 1, &rp, 0) == 0);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A stopped voice gives back every queued buffer at the next frame, a flush releases from the client's current buffer, and an
// empty buffer counts as played when it is reached.
static void g_flush_stop(Cpu *c) {
    TestCfg t = cfg_default();
    t.voices = 3;
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    voice_play(&b.voice[0], 0, 48000, (uint32_t[]){24000, 24000, 24000}, 3);   // a second each
    voice_play(&b.voice[1], 1, 48000, (uint32_t[]){24000, 24000}, 2);
    voice_play(&b.voice[2], 2, 48000, (uint32_t[]){0, 480}, 2);                 // an empty buffer first
    BkAnswer a;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    block_resend(&b);
    tm_set(origin + tick_ns(48000, 240, 1));
    CHECK(await_signals(1));
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.consumed[0] == 0 && a.consumed[1] == 0 && a.consumed[2] == 1);   // the empty buffer of voice 2 was played at once
    b.voice[0].play_state = 1;   // stop voice 0: its three queued buffers come back at the next frame
    b.voice[1].flush = 1;        // flush voice 1 from its current buffer
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.consumed[0] == 0 && a.consumed[1] == 0);   // not before a frame has run
    tm_set(origin + tick_ns(48000, 240, 2));
    CHECK(await_signals(2));
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.consumed[0] == 3 && a.consumed[1] == 1);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The output device the manager hands out: one stereo device named MainAudioOut, its volume read back as set, the change
// events (and the notifications of the newer SDKs, commands 15 and 17, with their 16 bytes of arguments) that never signal.
static void g_device_basic(Cpu *c) {
    uint32_t sm = open_sm(c), aud = get_service(c, sm, "aud:u");
    ClRep rp;
    CHECK(call(c, aud, 2, NULL, 0, &rp, 0) == 0 && rp.nmove == 1);
    uint32_t mgr = rp.move[0];
    uint64_t aruid = 0;
    CHECK(call(c, mgr, 2, &aruid, 8, &rp, 0) == 0 && rp.nmove == 1);        // GetAudioDeviceService
    uint32_t dev = rp.move[0];
    uint32_t revargs[4] = {0x40564552u, 0, 0, 0};
    CHECK(call(c, mgr, 4, revargs, 16, &rp, 0) == 0 && rp.nmove == 1);      // ...WithRevisionInfo: a device of its own
    uint32_t dev2 = rp.move[0];
    CHECK(dev2 != dev);
    uint64_t buf = HEAP + 0x400000;
    static const uint32_t name_cmds[] = {0, 6, 14, 3, 10, 13};   // list (three spellings), then the active name (three)
    for (unsigned i = 0; i < sizeof name_cmds / sizeof *name_cmds; i++) {
        for (int pointer = 0; pointer < 2; pointer++) {
            memset((void *)(uintptr_t)buf, 0x55, 0x200);
            ClReq rq = {.cmd = name_cmds[i]};
            if (pointer) rq.nc = 1, rq.c[0] = (ClBuf){buf, 0x100, 0};
            else rq.nb = 1, rq.b[0] = (ClBuf){buf, 0x100, 1};
            CHECK(cl_send(c, dev, &rq, &rp, 4) == 0 && rp.result == 0);
            CHECK(!strcmp((const char *)(uintptr_t)buf, "MainAudioOut"));
            CHECK(((const uint8_t *)(uintptr_t)buf)[0xff] == 0 && ((const uint8_t *)(uintptr_t)buf)[0x100] == 0x55);   // the whole 0x100
            CHECK(rp.nx == (unsigned)pointer && (!pointer || (rp.x[0].addr == buf && rp.x[0].size == 0x100)));
            if (name_cmds[i] < 7 && name_cmds[i] != 3) CHECK(bk_rd32(rp.data) == 1);   // the list answers one name
        }
    }
    {   // a list buffer too small for a name: no name, a count of 0
        memset((void *)(uintptr_t)buf, 0x55, 0x200);
        ClReq rq = {.cmd = 14, .nb = 1};
        rq.b[0] = (ClBuf){buf, 0x80, 1};
        CHECK(cl_send(c, dev, &rq, &rp, 4) == 0 && rp.result == 0 && bk_rd32(rp.data) == 0 && ((const uint8_t *)(uintptr_t)buf)[0] == 0x55);
    }
    float half = 0.5f, got;
    CHECK(call(c, dev, 2, NULL, 0, &rp, 4) == 0);                            // GetAudioDeviceOutputVolume: 1.0 at first
    memcpy(&got, rp.data, 4);
    CHECK(got == 1.0f);
    CHECK(call(c, dev, 1, &half, 4, &rp, 0) == 0);                           // SetAudioDeviceOutputVolume
    CHECK(call(c, dev, 8, NULL, 0, &rp, 4) == 0);                            // the Auto spelling reads the same value
    memcpy(&got, rp.data, 4);
    CHECK(got == 0.5f);
    CHECK(call(c, dev, 5, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 2);   // GetActiveAudioDeviceChannelCount
    uint8_t one = 1;
    CHECK(call(c, dev, 19, &one, 1, &rp, 0) == 0);                           // SetAudioDeviceOutputVolumeAutoTuneEnabled
    CHECK(call(c, dev, 20, NULL, 0, &rp, 1) == 0 && rp.data[0] == 0);        // off
    CHECK(call(c, dev, 21, NULL, 0, &rp, 1) == 0 && rp.data[0] == 0);        // not low latency
    static const uint32_t event_cmds[] = {4, 11, 12, 15, 17};
    for (unsigned i = 0; i < sizeof event_cmds / sizeof *event_cmds; i++) {
        uint32_t args[4] = {0, 0, 0, 0};
        CHECK(call(c, dev, event_cmds[i], args, event_cmds[i] >= 15 ? 16 : 0, &rp, 0) == 0 && rp.ncopy == 1 && rp.nmove == 0);
        CHECK(poll_event(c, rp.copy[0]) == R_TIMED_OUT);                     // never signalled
        close_handle(c, rp.copy[0]);
    }
    CHECK(call(c, dev, 16, NULL, 0, &rp, 0) == 0 && call(c, dev, 18, NULL, 0, &rp, 0) == 0);   // the releases
    close_handle(c, dev2);
    close_handle(c, dev);
    close_handle(c, mgr);
    close_handle(c, aud);
    close_handle(c, sm);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// auddmg:u: command 0 opens the virtual device manager; its list answers by the kind argument (0, the game's start-up call,
// lists the input devices: none; another kind lists the one output device), and the default device, the format check, the
// property block and the state event follow.
static void g_auddmg_basic(Cpu *c) {
    uint32_t sm = open_sm(c), svc = get_service(c, sm, "auddmg:u");
    ClRep rp;
    uint8_t open_args[16] = {0};
    ClReq rq0 = {.cmd = 0, .data = open_args, .size = 16, .pid = 1};
    CHECK(cl_send(c, svc, &rq0, &rp, 0) == 0 && rp.result == 0 && rp.nmove == 1);   // OpenVirtualDeviceManager
    uint32_t mgr = rp.move[0];
    uint64_t buf = HEAP + 0x400000;
    for (int pointer = 0; pointer < 2; pointer++) {
        for (uint32_t kind = 0; kind < 2; kind++) {
            memset((void *)(uintptr_t)buf, 0x55, 0x800);
            ClReq rq = {.cmd = 0, .data = &kind, .size = 4};
            if (pointer) rq.nc = 1, rq.c[0] = (ClBuf){buf, 0x800, 0};
            else rq.nb = 1, rq.b[0] = (ClBuf){buf, 0x800, 1};
            CHECK(cl_send(c, mgr, &rq, &rp, 4) == 0 && rp.result == 0);
            CHECK(bk_rd32(rp.data) == (kind == 0 ? 0u : 1u));                          // the input list is empty
            if (kind == 0) {
                CHECK(((const uint8_t *)(uintptr_t)buf)[0] == 0x55 && rp.nx == 0);   // nothing written, nothing answered
            } else {
                CHECK(!strcmp((const char *)(uintptr_t)buf, "AudioVirtualOutput0") && ((const uint8_t *)(uintptr_t)buf)[0x100] == 0x55);
                CHECK(rp.nx == (unsigned)pointer);
            }
        }
    }
    {   // room for no name: none listed
        uint32_t kind = 1;
        memset((void *)(uintptr_t)buf, 0x55, 0x800);
        ClReq rq = {.cmd = 0, .data = &kind, .size = 4, .nb = 1};
        rq.b[0] = (ClBuf){buf, 0x80, 1};
        CHECK(cl_send(c, mgr, &rq, &rp, 4) == 0 && bk_rd32(rp.data) == 0);
    }
    memset((void *)(uintptr_t)buf, 0x55, 0x800);
    ClReq rq = {.cmd = 100, .nb = 1};   // the default output device
    rq.b[0] = (ClBuf){buf, 0x100, 1};
    CHECK(cl_send(c, mgr, &rq, &rp, 0) == 0 && rp.result == 0 && !strcmp((const char *)(uintptr_t)buf, "AudioVirtualOutput0"));
    uint32_t fmt[4] = {5, 2, 48000, 1};
    CHECK(call(c, mgr, 300, fmt, 16, &rp, 0) == 0);   // the stream format is accepted
    memset((void *)(uintptr_t)buf, 0x55, 0x800);
    rq = (ClReq){.cmd = 401, .nb = 1};                // the 0x228-byte property block
    rq.b[0] = (ClBuf){buf, 0x228, 1};
    CHECK(cl_send(c, mgr, &rq, &rp, 0) == 0 && rp.result == 0);
    const uint8_t *props = (const uint8_t *)(uintptr_t)buf;
    CHECK(!strcmp((const char *)props, "AudioVirtualOutput0") && bk_rd32(props + 0x100) == 3 && bk_rd32(props + 0x104) == 2 &&
          bk_rd32(props + 0x108) == 2 && bk_rd32(props + 0x10c) == 48000 && !strcmp((const char *)props + 0x110, "MainAudioOut") &&
          bk_rd64(props + 0x218) == 10000000ull && bk_rd32(props + 0x220) == 1 && props[0x228] == 0x55);
    rq.b[0].size = 0x100;                             // too small: the size result
    CHECK(cl_send(c, mgr, &rq, &rp, 0) == 0 && rp.result == 0xCA01u);
    uint8_t ev_args[16] = {0};
    CHECK(call(c, mgr, 500, ev_args, 16, &rp, 0) == 0 && rp.ncopy == 1);   // the state event never signals
    CHECK(poll_event(c, rp.copy[0]) == R_TIMED_OUT);
    close_handle(c, rp.copy[0]);
    close_handle(c, mgr);
    close_handle(c, svc);
    close_handle(c, sm);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// ---- the cases that end in a named fatal
static void g_fatal(Cpu *c) {
    TestCfg t = cfg_default();
    BkAnswer a;
    BkBlock b;
    if (!strncmp(g_case, "open_", 5)) {
        if (!strcmp(g_case, "open_manual")) t.mode = 1;
        if (!strcmp(g_case, "open_rate")) t.rate = 44100;
        if (!strcmp(g_case, "open_device")) t.device = 1;
        if (!strcmp(g_case, "open_tag")) t.tag = 0x30564552u;   // "REV0"
        if (!strcmp(g_case, "open_notmem")) {
            Rend r = {0};
            r.sm = open_sm(c);
            r.aud = get_service(c, r.sm, "aud:u");
            ClRep rp;
            call(c, r.aud, 2, NULL, 0, &rp, 0);
            uint8_t arg[0x48] = {0};
            put_param(arg, &t);
            uint64_t work = 0x10000;
            memcpy(arg + 0x38, &work, 8);
            ClReq rq = {.cmd = 0, .data = arg, .size = 0x48, .ncopy = 1};
            rq.copy[0] = 0xFFFF8001u;   // the process only: no transfer memory
            cl_send(c, rp.move[0], &rq, &rp, 0);
        } else {
            rend_open(c, &t);
        }
    } else if (!strcmp(g_case, "event_twice")) {
        Rend r = rend_open(c, &t);
        rend_event(c, &r);
        rend_event(c, &r);
    } else if (!strcmp(g_case, "manual_execution")) {
        Rend r = rend_open(c, &t);
        ClRep rp;
        call(c, r.session, 11, NULL, 0, &rp, 0);
    } else if (!strcmp(g_case, "auddmg_stream") || !strcmp(g_case, "auddmg_open")) {
        uint32_t svc = get_service(c, open_sm(c), "auddmg:u");
        ClRep rp;
        if (!strcmp(g_case, "auddmg_open")) {
            call(c, svc, 1, NULL, 0, &rp, 0);   // only command 0 opens the manager
        } else {
            call(c, svc, 0, NULL, 0, &rp, 0);
            uint32_t args[4] = {1, 1, 0, 0};
            call(c, rp.move[0], 900, args, 16, &rp, 0);   // a command of the device manager that nothing names
        }
    } else if (!strcmp(g_case, "aud_unknown")) {
        uint32_t aud = get_service(c, open_sm(c), "aud:u");
        ClRep rp;
        call(c, aud, 0, NULL, 0, &rp, 0);   // the input manager: no reference, not met
    } else if (!strncmp(g_case, "parse_", 6)) {
        t.voices = 2;
        Rend r = rend_open(c, &t);
        block_new(&b, &t);
        voice_play(&b.voice[0], 0, 48000, (uint32_t[]){240}, 1);
        voice_play(&b.voice[1], 1, 48000, (uint32_t[]){240}, 1);
        uint32_t out = 0x2000;
        if (!strcmp(g_case, "parse_voice_stride")) b.voice_stride = 0x1a0;   // a layout of another size
        if (!strcmp(g_case, "parse_voice_extra")) b.voices_extra = 8;
        if (!strcmp(g_case, "parse_voice_old")) b.voice_stride = 0x170;      // a revision 14 section under a revision 16 tag
        if (!strcmp(g_case, "parse_pools")) b.pools_extra = 0x20;
        if (!strcmp(g_case, "parse_effects")) b.effects_extra = 0xc0;
        if (!strcmp(g_case, "parse_sinks")) b.sinks_extra = 0x20;
        if (!strcmp(g_case, "parse_behavior")) b.behavior_size = 0x20;
        if (!strcmp(g_case, "parse_total")) b.total_extra = 0x100000;
        if (!strcmp(g_case, "parse_leftover")) b.trailing = 16, b.total_extra = 16;   // 16 bytes after the sections, inside the total
        if (!strcmp(g_case, "parse_out_small")) out = 0x100;
        if (!strcmp(g_case, "parse_voice_id")) b.voice[1].id = 9;
        if (!strcmp(g_case, "parse_wave_index")) b.voice[0].wave_buffer_index = 4;
        if (!strcmp(g_case, "parse_revision")) b.tag = 0x30564552u;   // "REV0"
        if (!strcmp(g_case, "parse_mixes")) b.mix_bytes = 0x930 + 8;
        if (!strcmp(g_case, "parse_splitter_magic")) {
            static uint8_t sp[0x30];
            memset(sp, 0, sizeof sp);
            sp[0] = 'X';
            b.splitters = sp, b.splitter_bytes = sizeof sp;
        }
        rend_update(c, &r, &b, &a, out);
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// ---- mixing: a synthetic voice to the samples a capture sink should hold
#define DATA_AT (HEAP + 0x100000)
#define COEF_AT (DATA_AT + 0x8000)
typedef struct MixRun {
    uint8_t format, channels;
    uint32_t rate;
    float pitch, volume;
    int samples, ticks;
    uint32_t bytes;
    int pooled;   // 1: the data lies in an attached memory pool
    int adpcm;
    int submix, splitters, dests;                  // the renderer's counts (0: the defaults of cfg_default)
    void (*setup)(BkBlock *b, BkVoice *v);         // changes the block before it is sent: routes, filters, loops, sinks
} MixRun;
// Plays voice 0 of a fresh renderer with the data at DATA_AT (a mono voice into the final mix's buffers 0 and 1 at gain 1,
// the device sink reading those two) for ticks frames on the driven clock; the stereo frames the capture sink holds go to
// out (up to out_max frames); returns how many there are.
static size_t mix_run(Cpu *c, const MixRun *m, const void *data, float *out, size_t out_max) {
    memcpy((void *)(uintptr_t)DATA_AT, data, m->bytes);
    TestCfg t = cfg_default();
    if (m->submix) t.submix = m->submix;
    t.splitters = m->splitters, t.dests = m->dests;
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    voice_play(&b.voice[0], 0, m->rate, (uint32_t[]){(uint32_t)m->samples}, 1);
    BkVoice *v = &b.voice[0];
    v->format = m->format, v->channels = m->channels, v->pitch = m->pitch, v->volume = m->volume;
    v->mix_id = 0, v->gain[0] = 1.0f, v->gain[1] = 1.0f;
    v->wave[0].address = DATA_AT, v->wave[0].size = m->bytes;
    if (m->adpcm) v->extra_addr = COEF_AT, v->extra_size = 32;
    b.pool_addr[0] = DATA_AT, b.pool_size[0] = 0x10000, b.pool_state[0] = m->pooled ? 4 : 0;
    b.mix[0] = (BkMix){.volume = 1.0f, .buffer_count = 2, .in_use = 1};
    b.sink[0] = (BkSink){.type = 1, .in_use = 1, .input_count = 2, .inputs = {0, 1}};
    if (m->setup) m->setup(&b, v);
    BkAnswer a;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    for (int k = 1; k <= m->ticks; k++) {
        tm_set(origin + tick_ns(48000, 240, (uint64_t)k));
        CHECK(await_signals((uint64_t)k));
    }
    AudioRenderer *rr = audio_renderer_nth(0);
    CHECK(rr != NULL);
    size_t n = 0;
    const float *d = audio_sink_capture_data(audio_renderer_sink(rr), &n);
    CHECK(d != NULL && n == (size_t)m->ticks * 240);
    if (n > out_max) n = out_max;
    memcpy(out, d, n * 2 * sizeof(float));
    rend_stop(c, &r);
    rend_close(c, &r);
    return n;
}
static int near_f(float a, float b) { return fabsf(a - b) < 1e-6f; }

// A PCM16 mono voice at 48 kHz, volume 0.5, gain 1.0 into mix 0, a device sink on inputs 0 and 1, a 480-sample ramp: two
// frames come out, left and right both sample / 32768 x 0.5.
static void g_mix_voice(Cpu *c) {
    int16_t ramp[480];
    for (int i = 0; i < 480; i++) ramp[i] = (int16_t)((i - 240) * 100);
    MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 0.5f, .samples = 480, .ticks = 2,
                .bytes = sizeof ramp, .pooled = 1};
    static float out[480 * 2];
    CHECK(mix_run(c, &m, ramp, out, 480) == 480);
    for (int i = 0; i < 480; i++) {
        float want = (float)ramp[i] / 32768.0f * 0.5f;
        CHECK(near_f(out[2 * i], want) && near_f(out[2 * i + 1], want));
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// Unused voice records can be zeroed. Dropping one must not silence another, and voice state belongs to the record slot.
static void g_mix_voice_slots(Cpu *c) {
    int16_t left[720], right[720];
    for (int i = 0; i < 720; i++) left[i] = 8192, right[i] = 16384;
    memcpy((void *)(uintptr_t)DATA_AT, left, sizeof left);
    memcpy((void *)(uintptr_t)(DATA_AT + 0x1000), right, sizeof right);

    TestCfg t = cfg_default();
    if (!strcmp(g_case, "mix_voice_slots_old")) t.tag = BK_TAG(14);
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    for (int i = 0; i < 2; i++) {
        voice_play(&b.voice[i], (uint32_t)i, 48000, (uint32_t[]){720}, 1);
        b.voice[i].channels = 1;
        b.voice[i].mix_id = 0;
        b.voice[i].gain[i] = 1.0f;
        b.voice[i].wave[0].address = DATA_AT + (uint64_t)i * 0x1000;
        b.voice[i].wave[0].size = sizeof left;
    }
    b.voice[2].id = b.voice[3].id = 0;
    b.pool_addr[0] = DATA_AT, b.pool_size[0] = 0x10000, b.pool_state[0] = 4;
    b.mix[0] = (BkMix){.volume = 1.0f, .buffer_count = 2, .in_use = 1};
    b.sink[0] = (BkSink){.type = 1, .in_use = 1, .input_count = 2, .inputs = {0, 1}};
    BkAnswer a;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    tm_set(origin + tick_ns(48000, 240, 1));
    CHECK(await_signals(1));
    AudioRenderer *rr = audio_renderer_nth(0);
    CHECK(rr != NULL);
    size_t n = 0;
    const float *out = audio_sink_capture_data(audio_renderer_sink(rr), &n);
    CHECK(out != NULL && n == 240);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], 0.25f) && near_f(out[2 * i + 1], 0.5f));

    block_resend(&b);
    memset(&b.voice[1], 0, sizeof b.voice[1]);
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.played[0] == 240 && a.played[1] == 0);
    tm_set(origin + tick_ns(48000, 240, 2));
    CHECK(await_signals(2));
    out = audio_sink_capture_data(audio_renderer_sink(rr), &n);
    CHECK(out != NULL && n == 480);
    for (int i = 240; i < 480; i++) CHECK(near_f(out[2 * i], 0.25f) && near_f(out[2 * i + 1], 0.0f));

    block_resend(&b);
    voice_play(&b.voice[1], 3, 48000, (uint32_t[]){240}, 1);
    b.voice[1].channels = 1, b.voice[1].mix_id = 0;
    b.voice[1].wave[0].address = DATA_AT + 0x1000, b.voice[1].wave[0].size = 480;
    b.voice[3].id = 0xffffffffu;
    b.voice[3].gain[1] = 1.0f;
    b.voice[3].biquad[0].enable = 1, b.voice[3].biquad[0].num[0] = NAN;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    tm_set(origin + tick_ns(48000, 240, 3));
    CHECK(await_signals(3));
    out = audio_sink_capture_data(audio_renderer_sink(rr), &n);
    CHECK(out != NULL && n == 720);
    for (int i = 480; i < 720; i++) CHECK(near_f(out[2 * i], 0.25f) && near_f(out[2 * i + 1], 0.5f));
    block_resend(&b);
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.played[1] == 240 && a.played[3] == 0);
    rend_stop(c, &r);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The sample formats the client may use, each a ramp of 240 samples, one frame, volume 1.
static void g_mix_formats(Cpu *c) {
    static float out[240 * 2];
    int8_t p8[240];
    uint8_t p24[240 * 3];
    int32_t p32[240];
    float pf[240];
    for (int i = 0; i < 240; i++) {
        int v = (i - 120) * 1000;   // PCM24 range
        p8[i] = (int8_t)(i - 120);
        p24[3 * i] = (uint8_t)v, p24[3 * i + 1] = (uint8_t)(v >> 8), p24[3 * i + 2] = (uint8_t)(v >> 16);
        p32[i] = (i - 120) * 10000000;
        pf[i] = (float)(i - 120) / 128.0f;
    }
    MixRun m = {.channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 240, .ticks = 1, .pooled = 1};
    m.format = 1, m.bytes = sizeof p8;
    CHECK(mix_run(c, &m, p8, out, 240) == 240);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], (float)p8[i] / 128.0f) && near_f(out[2 * i + 1], out[2 * i]));
    m.format = 3, m.bytes = sizeof p24;
    CHECK(mix_run(c, &m, p24, out, 240) == 240);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], (float)((i - 120) * 1000) / 8388608.0f));
    m.format = 4, m.bytes = sizeof p32;
    CHECK(mix_run(c, &m, p32, out, 240) == 240);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], (float)((double)p32[i] / 2147483648.0)));
    m.format = 5, m.bytes = sizeof pf;
    CHECK(mix_run(c, &m, pf, out, 240) == 240);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], pf[i]));
    sv(c, 0x0A, 0, 0, 0, 0);
}

// Resampling is linear by rate x pitch: a 24 kHz voice, and a 48 kHz voice at pitch 0.5, both give each input sample
// followed by the midpoint to the next.
static void g_mix_resample(Cpu *c) {
    int16_t ramp[240];
    for (int i = 0; i < 240; i++) ramp[i] = (int16_t)(i * 128);
    static float out[240 * 2];
    for (int k = 0; k < 2; k++) {
        MixRun m = {.format = 2, .channels = 1, .rate = k ? 48000u : 24000u, .pitch = k ? 0.5f : 1.0f, .volume = 1.0f,
                    .samples = 240, .ticks = 1, .bytes = sizeof ramp, .pooled = 1};
        CHECK(mix_run(c, &m, ramp, out, 240) == 240);
        for (int i = 0; i < 120; i++) {
            CHECK(near_f(out[4 * i], (float)ramp[i] / 32768.0f));
            CHECK(near_f(out[4 * i + 2], ((float)ramp[i] + (float)ramp[i + 1]) / 2 / 32768.0f));
        }
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// DSP-ADPCM with a zero coefficient table decodes each nibble to nibble x scale.
static void g_mix_adpcm(Cpu *c) {
    uint8_t data[16];
    int expect[28];
    static const int nib[14] = {1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2};
    for (int f = 0; f < 2; f++) {
        data[8 * f] = (uint8_t)(f == 0 ? 0x02 : 0x01);   // predictor 0, scale 4 then 2
        for (int n = 0; n < 14; n += 2) data[8 * f + 1 + n / 2] = (uint8_t)(((nib[n] & 15) << 4) | (nib[n + 1] & 15));
        for (int n = 0; n < 14; n++) expect[14 * f + n] = nib[n] * (f == 0 ? 4 : 2);
    }
    memset((void *)(uintptr_t)COEF_AT, 0, 32);
    MixRun m = {.format = 6, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 28, .ticks = 1,
                .bytes = sizeof data, .pooled = 1, .adpcm = 1};
    static float out[240 * 2];
    CHECK(mix_run(c, &m, data, out, 240) == 240);
    for (int i = 0; i < 28; i++) CHECK(near_f(out[2 * i], (float)expect[i] / 32768.0f));
    for (int i = 28; i < 240; i++) CHECK(out[2 * i] == 0.0f);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A wave buffer that no attached memory pool holds is never read: the frames are silent and the log names it once.
static void g_mix_pool_refused(Cpu *c) {
    int16_t ramp[240];
    for (int i = 0; i < 240; i++) ramp[i] = (int16_t)(1000 + i);
    MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 240, .ticks = 2,
                .bytes = sizeof ramp, .pooled = 0};
    static float out[480 * 2];
    CHECK(mix_run(c, &m, ramp, out, 480) == 480);
    for (int i = 0; i < 480; i++) CHECK(out[2 * i] == 0.0f && out[2 * i + 1] == 0.0f);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// ---- what the first mixing cases leave out: routes through splitters and sub mixes, filters, loops, channels, sinks
static uint8_t g_split_bytes[0x400];
static float ramp_s(int i) { return (float)((i - 120) * 100) / 32768.0f; }
static void put_ramp(int16_t *r, int n) {
    for (int i = 0; i < n; i++) r[i] = (int16_t)((i - n / 2) * 100);
}

// The voice goes through splitter 0 to a destination feeding sub mix 2 (volumes 0.8 and 0.4 on its two buffers); mix 2 (volume
// 0.5) feeds mix 1 (volume 0.8, a matrix that crosses its first buffer into both outputs), which feeds the final mix (volume
// 0.25). Mix 2 comes first in the dependency order though it has the higher number. The voice's volume is 0.5.
static void setup_route(BkBlock *b, BkVoice *v) {
    BkSplitDest d = {.mix = 2};
    d.vol[0] = 0.8f, d.vol[1] = 0.4f;
    b->splitter_bytes = bk_splitter(g_split_bytes, 1, &d);
    b->splitters = g_split_bytes;
    v->via_splitter = 1, v->splitter_id = 0;
    b->mix[0].volume = 0.25f;
    b->mix[2] = (BkMix){.volume = 0.5f, .buffer_count = 2, .in_use = 1, .dest_mix_id = 1, .matrix = {{1, 0}, {0, 1}}};
    b->mix[1] = (BkMix){.volume = 0.8f, .buffer_count = 2, .in_use = 1, .dest_mix_id = 0, .matrix = {{1.0f, 0.5f}, {0, 0.25f}}};
}
static void g_mix_route(Cpu *c) {
    int16_t ramp[240];
    put_ramp(ramp, 240);
    MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 0.5f, .samples = 240, .ticks = 1,
                .bytes = sizeof ramp, .pooled = 1, .submix = 2, .splitters = 1, .dests = 1, .setup = setup_route};
    static float out[240 * 2];
    CHECK(mix_run(c, &m, ramp, out, 240) == 240);
    for (int i = 0; i < 240; i++) {
        float s = (float)ramp[i] / 32768.0f * 0.5f;
        float m2b0 = s * 0.8f, m2b1 = s * 0.4f;              // the splitter's destination volumes into mix 2
        float m1b0 = m2b0 * 0.5f, m1b1 = m2b1 * 0.5f;        // mix 2's volume, its identity matrix, into mix 1
        float f0 = m1b0 * 1.0f * 0.8f;                       // mix 1's volume and matrix into the final mix
        float f1 = (m1b0 * 0.5f + m1b1 * 0.25f) * 0.8f;
        CHECK(near_f(out[2 * i], f0 * 0.25f) && near_f(out[2 * i + 1], f1 * 0.25f));   // the final mix's volume
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The final mix's volume 0.25 and the device volume 0.5 both scale the output; the device volume is back at 1 afterwards.
static void setup_final_volume(BkBlock *b, BkVoice *v) {
    (void)v;
    b->mix[0].volume = 0.25f;
}
static void g_mix_device_volume(Cpu *c) {
    int16_t ramp[240];
    put_ramp(ramp, 240);
    MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 240, .ticks = 1,
                .bytes = sizeof ramp, .pooled = 1, .setup = setup_final_volume};
    static float out[240 * 2];
    CHECK(audio_device_volume() == 1.0f);
    CHECK(mix_run(c, &m, ramp, out, 240) == 240);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], ramp_s(i) * 0.25f));
    audio_device_volume_set(0.5f);
    CHECK(mix_run(c, &m, ramp, out, 240) == 240);
    audio_device_volume_set(1.0f);
    for (int i = 0; i < 240; i++) CHECK(near_f(out[2 * i], ramp_s(i) * 0.25f * 0.5f) && near_f(out[2 * i + 1], out[2 * i]));
    sv(c, 0x0A, 0, 0, 0, 0);
}

// Both voice filters, in order, with the state carried from one frame to the next: the first has a feedback term, the second
// is a gain of 0.5. y = a0 x + z0, z0 = a1 x + b1 y + z1, z1 = a2 x + b2 y.
static void setup_biquads(BkBlock *b, BkVoice *v) {
    (void)b;
    v->biquad[0].enable = 1;
    v->biquad[0].num[0] = 0.5f, v->biquad[0].num[1] = 0.25f, v->biquad[0].num[2] = 0.125f;
    v->biquad[0].den[0] = 0.5f, v->biquad[0].den[1] = -0.25f;
    v->biquad[1].enable = 1;
    v->biquad[1].num[0] = 0.5f;
}
static void g_mix_biquad(Cpu *c) {
    int16_t ramp[480];
    put_ramp(ramp, 480);
    MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 480, .ticks = 2,
                .bytes = sizeof ramp, .pooled = 1, .setup = setup_biquads};
    static float out[480 * 2];
    CHECK(mix_run(c, &m, ramp, out, 480) == 480);
    double z0 = 0, z1 = 0;
    for (int i = 0; i < 480; i++) {
        double x = (double)ramp[i] / 32768.0, y = 0.5 * x + z0;
        z0 = 0.25 * x + 0.5 * y + z1;
        z1 = 0.125 * x - 0.25 * y;
        CHECK(fabs((double)out[2 * i] - 0.5 * y) < 1e-6 && fabs((double)out[2 * i + 1] - 0.5 * y) < 1e-6);
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// An endless loop over a 100-sample buffer, from its start and from its sample 40: 240 output samples repeat it.
static void setup_loop(BkBlock *b, BkVoice *v) {
    (void)b;
    v->wave[0].loop = 1, v->wave[0].loop_count = -1;
}
static void setup_loop_from_40(BkBlock *b, BkVoice *v) {
    setup_loop(b, v);
    v->wave[0].loop_start = 40;
}
static void g_mix_loop(Cpu *c) {
    int16_t ramp[100];
    put_ramp(ramp, 100);
    static float out[240 * 2];
    for (int k = 0; k < 2; k++) {
        MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 100, .ticks = 1,
                    .bytes = sizeof ramp, .pooled = 1, .setup = k ? setup_loop_from_40 : setup_loop};
        CHECK(mix_run(c, &m, ramp, out, 240) == 240);
        for (int i = 0; i < 240; i++) {
            int at = i < 100 ? i : k ? 40 + (i - 100) % 60 : i % 100;
            CHECK(near_f(out[2 * i], (float)ramp[at] / 32768.0f));
        }
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A stereo voice: interleaved left and right samples, each channel with its own channel resource and gains: the left into
// the first output only, the right into the second only.
static void setup_stereo(BkBlock *b, BkVoice *v) {
    b->second_channel = 1;
    v->gain[0] = 1.0f, v->gain[1] = 0.0f;
    v->gain1[0] = 0.0f, v->gain1[1] = 1.0f;
}
static void g_mix_stereo(Cpu *c) {
    int16_t pcm[2 * 240];
    for (int i = 0; i < 240; i++) pcm[2 * i] = (int16_t)((i - 120) * 100), pcm[2 * i + 1] = (int16_t)(1000 + 7 * i);
    MixRun m = {.format = 2, .channels = 2, .rate = 48000, .pitch = 1.0f, .volume = 0.5f, .samples = 240, .ticks = 1,
                .bytes = sizeof pcm, .pooled = 1, .setup = setup_stereo};
    static float out[240 * 2];
    CHECK(mix_run(c, &m, pcm, out, 240) == 240);
    for (int i = 0; i < 240; i++) {
        CHECK(near_f(out[2 * i], (float)pcm[2 * i] / 32768.0f * 0.5f));
        CHECK(near_f(out[2 * i + 1], (float)pcm[2 * i + 1] / 32768.0f * 0.5f));
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A device sink of six inputs on a final mix of six buffers (front left and right, centre, low frequency, rear left and
// right) comes down to two: with the usual 0.7071 for the centre and the rears, and with the sink's own coefficients; the
// low-frequency input is not heard. The mono voice puts a different gain into each buffer.
static void setup_six(BkBlock *b, BkVoice *v) {
    b->mix[0].buffer_count = 6;
    b->sink[0] = (BkSink){.type = 1, .in_use = 1, .input_count = 6, .inputs = {0, 1, 2, 3, 4, 5}};
    static const float g[6] = {0.5f, 0.25f, 1.0f, 0.8f, 0.125f, 0.0625f};
    for (int i = 0; i < 6; i++) v->gain[i] = g[i];
}
static void setup_six_coefficients(BkBlock *b, BkVoice *v) {
    setup_six(b, v);
    b->sink[0].downmix_enabled = 1;
    b->sink[0].downmix[0] = 0.5f, b->sink[0].downmix[1] = 0.25f, b->sink[0].downmix[2] = 0.125f;
}
static void g_mix_sink6(Cpu *c) {
    int16_t ramp[240];
    put_ramp(ramp, 240);
    static float out[240 * 2];
    for (int k = 0; k < 2; k++) {
        MixRun m = {.format = 2, .channels = 1, .rate = 48000, .pitch = 1.0f, .volume = 1.0f, .samples = 240, .ticks = 1,
                    .bytes = sizeof ramp, .pooled = 1, .setup = k ? setup_six_coefficients : setup_six};
        CHECK(mix_run(c, &m, ramp, out, 240) == 240);
        const float ff = k ? 0.5f : 1.0f, cf = k ? 0.25f : 0.7071f, sf = k ? 0.125f : 0.7071f;
        for (int i = 0; i < 240; i++) {
            float s = (float)ramp[i] / 32768.0f;
            CHECK(near_f(out[2 * i], s * (ff * 0.5f + cf * 1.0f + sf * 0.125f)));
            CHECK(near_f(out[2 * i + 1], s * (ff * 0.25f + cf * 1.0f + sf * 0.0625f)));
        }
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A buffer with no samples that loops for ever starves its voice and nothing else: the frame goes on (its signal comes, the
// position reads), the voice consumes nothing and releases nothing behind it, and a voice whose empty buffer loops twice
// gives it back and plays the buffer behind it.
static void g_starve_empty_loop(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    voice_play(&b.voice[0], 0, 48000, (uint32_t[]){0, 240}, 2);
    b.voice[0].wave[0].loop = 1, b.voice[0].wave[0].loop_count = -1;   // endless, over nothing
    voice_play(&b.voice[1], 1, 48000, (uint32_t[]){0, 240}, 2);
    b.voice[1].wave[0].loop = 1, b.voice[1].wave[0].loop_count = 2;    // twice, over nothing: finite
    BkAnswer a;
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    block_resend(&b);
    for (uint64_t k = 1; k <= 3; k++) {
        tm_set(origin + tick_ns(48000, 240, k));
        CHECK(await_signals(k));
    }
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    CHECK(a.elapsed_frames == 3);
    CHECK(a.consumed[0] == 0 && a.played[0] == 0);   // starved: nothing played, nothing released
    CHECK(a.consumed[1] == 2);                        // the empty loop and the 240 samples behind it
    rend_stop(c, &r);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}


// ---- the sinks, with no guest
static void sink_null_case(void) {
    AudioSink *s = audio_sink_open(AUDIO_SINK_NULL, 48000, NULL);
    CHECK(s && audio_sink_kind(s) == AUDIO_SINK_NULL && !strcmp(audio_sink_kind_name(s), "null"));
    static float lr[2 * 4800];
    for (int i = 0; i < 2 * 4800; i++) lr[i] = 0.25f;
    AudioSinkStats st;
    CHECK(audio_sink_write(s, lr, 480) == 480);       // 10 ms: below the 20 ms prefill, so it does not play yet
    tm_set(tm_now(NULL) + 5000000ull);
    audio_sink_stats(s, &st);
    CHECK(st.written == 480 && st.played == 0 && st.queued == 480 && st.underruns == 0);
    CHECK(audio_sink_write(s, lr, 480) == 480);       // 20 ms queued: it starts to play
    tm_set(tm_now(NULL) + 10000000ull);               // 10 ms: 480 frames taken
    audio_sink_stats(s, &st);
    CHECK(st.written == 960 && st.played == 480 && st.queued == 480 && st.underruns == 0);
    CHECK(audio_sink_write(s, lr, 4800) == 0);        // 100 ms is the most it holds: refused whole
    audio_sink_stats(s, &st);
    CHECK(st.overruns == 1 && st.written == 960 && st.queued == 480);
    CHECK(audio_sink_write(s, lr, 4320) == 4320);     // exactly full
    audio_sink_stats(s, &st);
    CHECK(st.queued == 4800 && st.overruns == 1);
    tm_set(tm_now(NULL) + 1000000000ull);             // a second with no write: it ran dry once
    audio_sink_stats(s, &st);
    CHECK(st.queued == 0 && st.underruns == 1 && st.played == st.written);
    tm_set(tm_now(NULL) + 1000000000ull);
    audio_sink_stats(s, &st);
    CHECK(st.underruns == 1);                         // a sink that is not playing does not underrun again
    CHECK(audio_sink_write(s, lr, 480) == 480);
    audio_sink_pause(s, 1);                           // a stopped renderer: no queue, no underrun, nothing accepted
    CHECK(audio_sink_queued(s) == 0 && audio_sink_write(s, lr, 480) == 0);
    audio_sink_stats(s, &st);
    CHECK(st.overruns == 1 && st.underruns == 1);
    audio_sink_pause(s, 0);
    CHECK(audio_sink_write(s, lr, 480) == 480);
    size_t n;
    CHECK(audio_sink_capture_data(s, &n) == NULL && n == 0);
    audio_sink_close(s);
}

static void sink_capture_case(const char *dir) {
    char path[600];
    snprintf(path, sizeof path, "%s/capture.wav", dir);
    AudioSink *s = audio_sink_open(AUDIO_SINK_CAPTURE, 48000, path);
    CHECK(s && audio_sink_kind(s) == AUDIO_SINK_CAPTURE && !strcmp(audio_sink_device(s), path));
    float lr[2 * 1000];
    for (int i = 0; i < 1000; i++) lr[2 * i] = (float)i / 1000.0f, lr[2 * i + 1] = -(float)i / 1000.0f;
    CHECK(audio_sink_write(s, lr, 1000) == 1000);
    AudioSinkStats st;
    audio_sink_stats(s, &st);
    CHECK(st.written == 1000 && st.played == 1000 && st.queued == 0 && st.underruns == 0 && st.overruns == 0);
    size_t n;
    const float *d = audio_sink_capture_data(s, &n);
    CHECK(d && n == 1000 && !memcmp(d, lr, sizeof lr));
    audio_sink_close(s);
    FILE *f = fopen(path, "rb");   // a 16-bit stereo WAV of the frames
    CHECK(f != NULL);
    uint8_t h[44];
    CHECK(fread(h, 1, 44, f) == 44);
    CHECK(!memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WAVEfmt ", 8) && !memcmp(h + 36, "data", 4));
    CHECK(h[20] == 1 && h[22] == 2 && bk_rd32(h + 24) == 48000 && h[34] == 16);
    CHECK(bk_rd32(h + 40) == 4000 && bk_rd32(h + 4) == 4036);
    int16_t pcm[2 * 1000];
    CHECK(fread(pcm, 4, 1000, f) == 1000);
    fclose(f);
    for (int i = 0; i < 1000; i++) {
        CHECK(pcm[2 * i] == (int16_t)lrintf(lr[2 * i] * 32767.0f) && pcm[2 * i + 1] == (int16_t)lrintf(lr[2 * i + 1] * 32767.0f));
    }
    // auto and sdl3 in a build with no SDL3: auto is the null sink with the reason, sdl3 refuses
    AudioSink *a = audio_sink_open(AUDIO_SINK_AUTO, 32000, NULL);
    CHECK(a && audio_sink_kind(a) == AUDIO_SINK_NULL && strstr(audio_sink_device(a), "no device: this build has no SDL3 audio"));
    audio_sink_close(a);
    CHECK(audio_sink_open(AUDIO_SINK_SDL3, 48000, NULL) == NULL && strstr(audio_sink_why(), "no SDL3"));
}

// ---- the stall detector: the watchdog's own clock, moved by the test, apart from the renderer's
static PlatMutex g_wd_lock = PLAT_MUTEX_INIT;
static PlatCond g_wd_cond;
static uint64_t g_wd_now = 5000000000ull;
static unsigned g_wd_wake;
static uint64_t wd_now(void *ctx) {
    (void)ctx;
    plat_mutex_lock(&g_wd_lock);
    uint64_t t = g_wd_now;
    plat_mutex_unlock(&g_wd_lock);
    return t;
}
static void wd_wait_until(void *ctx, uint64_t deadline) {
    (void)ctx;
    plat_mutex_lock(&g_wd_lock);
    unsigned gen = g_wd_wake;
    while (g_wd_now < deadline && g_wd_wake == gen) plat_cond_wait(&g_wd_cond, &g_wd_lock);
    plat_mutex_unlock(&g_wd_lock);
}
static void wd_wake(void *ctx) {
    (void)ctx;
    plat_mutex_lock(&g_wd_lock);
    g_wd_wake++;
    plat_cond_broadcast(&g_wd_cond);
    plat_mutex_unlock(&g_wd_lock);
}
// The watchdog's time is t, and one poll is made at t: the watchdog's own thread polls at that time as well, and its stale
// polls are ignored.
static void wd_poll(uint64_t t) {
    plat_mutex_lock(&g_wd_lock);
    g_wd_now = t;
    plat_mutex_unlock(&g_wd_lock);
    audio_watchdog_poll_at(t);
}
static void wd_set(uint64_t t) {
    plat_mutex_lock(&g_wd_lock);
    g_wd_now = t;
    plat_mutex_unlock(&g_wd_lock);
}
static uint64_t stalls_now(void) {
    AudioPosition p;
    audio_position(&p);
    return p.stalls;
}

#define WD_START 5000000000ull
#define BOUND AUDIO_STALL_BOUND_NS
#define STOP_AT (HEAP + 0x3c8)
#define WOKEN_AT (HEAP + 0x3cc)
#define EVENT_AT (HEAP + 0x3c0)
#define LIST_AT (HEAP + 0x3d0)
// a guest thread that waits on the renderer's frame event, clears it as the SDK does, and counts its wakes, until told to stop
static void g_stall_waiter(Cpu *c) {
    uint32_t ev = LD32(EVENT_AT);
    ST32(LIST_AT, ev);
    while (!LD32(STOP_AT)) {
        uint64_t r = sv(c, 0x18, 0, LIST_AT, 1, ~0ull);
        if (r == 0) sv(c, 0x12, ev, 0, 0, 0);
        __atomic_add_fetch((uint32_t *)(uintptr_t)WOKEN_AT, 1, __ATOMIC_SEQ_CST);
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}
static uint32_t spawn_waiter(Cpu *c, uint32_t ev) {
    ST32(EVENT_AT, ev);
    ST32(STOP_AT, 0);
    ST32(WOKEN_AT, 0);
    c->x[1] = G(2), c->x[2] = 0, c->x[3] = HEAP + 0x800000, c->x[4] = 40, c->x[5] = 0;
    rt_svc(c, 0x08, PC);
    CHECK(c->x[0] == 0);
    uint32_t h = (uint32_t)c->x[1];
    CHECK(sv(c, 0x09, h, 0, 0, 0) == 0);
    return h;
}
static void stop_waiter(Cpu *c, uint32_t h) {
    ST32(STOP_AT, 1);
    CHECK(sv(c, 0x19, h, 0, 0, 0) == 0);
    ST32(HEAP + 0x3e0, h);
    CHECK(sv(c, 0x18, 0, HEAP + 0x3e0, 1, ~0ull) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == 0);
}
// the guest thread is blocked on the event: the kernel's waiter view shows it
static void await_blocked(Cpu *c, uint32_t ev) {
    RtWaiter w[4];
    for (int i = 0; i < 5000; i++) {
        if (rt_kernel_waiters(ev, w, 4) == 1 && w[0].blocked) return;
        sv(c, 0x0B, 1000000, 0, 0, 0);
    }
    CHECK(!"the waiter never blocked");
}

// A guest thread waits on the event of a renderer that renders nothing (started, its clock never moves): no report at the
// bound less 1 ms, one at the bound, none more until 10 s later, when the line is repeated (not a new episode).
static void g_stall_report(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    uint32_t ev = rend_event(c, &r);
    rend_start(c, &r);   // no update was ever sent, so the missing one is not what this case reports
    uint32_t w = spawn_waiter(c, ev);
    await_blocked(c, ev);
    wd_poll(WD_START);   // first seen
    CHECK(stalls_now() == 0);
    wd_poll(WD_START + BOUND - 1000000);
    CHECK(stalls_now() == 0);
    wd_poll(WD_START + BOUND);
    CHECK(stalls_now() == 1);
    AudioPosition p;
    audio_position(&p);
    CHECK(p.event_wait_max_ns == BOUND);
    wd_poll(WD_START + BOUND + 60000000ull);
    CHECK(stalls_now() == 1);
    wd_poll(WD_START + BOUND + 10000000000ull);   // the same episode: its line again, not a second stall
    CHECK(stalls_now() == 1);
    stop_waiter(c, w);
    close_handle(c, ev);
    rend_stop(c, &r);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The same waiter on a renderer that renders every frame: it is woken at each one, and three bounds of the watchdog's time
// pass with no report.
static void g_stall_healthy(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    uint32_t ev = rend_event(c, &r);
    BkBlock b;
    block_new(&b, &t);
    BkAnswer a;
    uint64_t origin = tm_now(NULL);
    rend_start(c, &r);
    uint32_t w = spawn_waiter(c, ev);
    await_blocked(c, ev);
    wd_poll(WD_START);
    for (uint64_t k = 1; k <= 600; k++) {   // 3 s of frames, the update block sent now and then as a client does
        tm_set(origin + tick_ns(48000, 240, k));
        CHECK(await_signals(k));
        for (int i = 0; i < 5000 && LD32(WOKEN_AT) < k; i++) sv(c, 0x0B, 100000, 0, 0, 0);   // it has been woken for this frame
        CHECK(LD32(WOKEN_AT) >= k);
        await_blocked(c, ev);
        wd_poll(WD_START + k * 5000000ull);
        if (k % 10 == 0) {
            block_resend(&b);
            CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
        }
    }
    CHECK(stalls_now() == 0);
    AudioPosition p;
    audio_position(&p);
    CHECK(p.event_wait_max_ns < BOUND / 100);
    stop_waiter(c, w);
    close_handle(c, ev);
    rend_stop(c, &r);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A wave buffer submitted and not released: the bound is its own duration, the buffers ahead of it, and a second, from the
// later of its submission and the voice's start; the report names the voice, the slot and the guest thread that submitted it.
// The renderer is stopped, so the stall is the wall clock's alone. Voice 0 holds 2,400 samples (50 ms) and then 4,800 (100 ms).
static void g_stall_names(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    voice_play(&b.voice[0], 0, 48000, (uint32_t[]){2400, 4800}, 2);
    BkAnswer a;
    wd_set(WD_START);
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);   // submitted now, by this thread
    wd_poll(WD_START);
    uint64_t first = WD_START + BOUND + 50000000ull, second = first + 100000000ull;
    wd_poll(first - 1000000);
    CHECK(stalls_now() == 0);
    wd_poll(first);
    CHECK(stalls_now() == 1);
    wd_poll(second - 1000000);
    CHECK(stalls_now() == 1);
    wd_poll(second);
    CHECK(stalls_now() == 2);
    wd_poll(second + 60000000ull);
    CHECK(stalls_now() == 2);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// A running renderer that has been updated and then is not: the report comes at the bound after the last update, and an update
// puts it off.
static void g_stall_no_update(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    BkBlock b;
    block_new(&b, &t);
    BkAnswer a;
    rend_start(c, &r);
    wd_set(WD_START);
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);
    wd_poll(WD_START + 500000000ull);
    wd_set(WD_START + 500000000ull);
    CHECK(rend_update(c, &r, &b, &a, 0x2000) == 0);   // an update at 0.5 s
    wd_poll(WD_START + BOUND);
    CHECK(stalls_now() == 0);
    wd_poll(WD_START + 500000000ull + BOUND - 1000000);
    CHECK(stalls_now() == 0);
    wd_poll(WD_START + 500000000ull + BOUND);
    CHECK(stalls_now() == 1);
    rend_stop(c, &r);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The renderer's thread holds the renderer's lock and does not let go: the watchdog does not wait on it. It says so at the bound
// (counted from the first poll that could not take the lock), not before, once, and nothing more when the lock is free again.
static void g_stall_lock(Cpu *c) {
    TestCfg t = cfg_default();
    Rend r = rend_open(c, &t);
    AudioRenderer *rr = audio_renderer_nth(0);
    CHECK(rr != NULL);
    wd_set(WD_START);
    wd_poll(WD_START);   // the lock is free: nothing
    CHECK(stall_count() == 0);
    uint64_t first = WD_START + 1000000ull;   // the first poll that finds it held
    // The clock moves to that poll's instant before the lock is taken: the watchdog's own thread also polls, and a poll of
    // it between the lock and the line below would otherwise start the bound at WD_START, a millisecond early.
    wd_set(first);
    plat_mutex_lock(&rr->lock);
    wd_poll(first);
    CHECK(stall_count() == 0);
    wd_poll(first + BOUND - 1000000);
    CHECK(stall_count() == 0);
    wd_poll(first + BOUND);
    CHECK(stall_count() == 1);
    wd_poll(first + BOUND + 60000000ull);
    CHECK(stall_count() == 1);
    plat_mutex_unlock(&rr->lock);
    wd_poll(first + BOUND + 120000000ull);
    CHECK(stall_count() == 1);
    rend_close(c, &r);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// ---- the sound mixer's streams: a ring the game fills and the service empties on a 5 ms grid into a sink
#define SMX_BUF (HEAP + 0x400000)
#define SMX_SIZE 0x4000
#define SMX_IDS (HEAP + 0x500000)
static uint32_t smx_ld(uint64_t off) { return __atomic_load_n((uint32_t *)(uintptr_t)(SMX_BUF + off), __ATOMIC_ACQUIRE); }
static void smx_st(uint64_t off, uint32_t v) { __atomic_store_n((uint32_t *)(uintptr_t)(SMX_BUF + off), v, __ATOMIC_RELEASE); }
// real time until the service has moved the read index to want
static int smx_await_read(uint32_t want) {
    for (int i = 0; i < 50000; i++) {
        if (smx_ld(0x44) == want) return 1;
        plat_sleep_ns(100000);
    }
    return 0;
}
static void smx_write_frames(uint32_t first, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        int16_t l = (int16_t)(((int)(first + i) - 240) * 50), r = (int16_t)-l;
        uint8_t *p = (uint8_t *)(uintptr_t)(SMX_BUF + 0x1040 + 4ull * ((first + i) % 1025));
        memcpy(p, &l, 2);
        memcpy(p + 2, &r, 2);
    }
    smx_st(0x1004, (first + n) % 1025);
}
static void g_sysmix_stream(Cpu *c) {
    uint32_t sm = open_sm(c), smx = get_service(c, sm, "audsmx:u"), dmg = get_service(c, sm, "auddmg:u");
    ClRep rp;
    ClReq rq = {.cmd = 0, .ncopy = 1};
    rq.copy[0] = 0xFFFF8001u;   // the process
    CHECK(cl_send(c, smx, &rq, &rp, 0) == 0 && rp.result == 0 && rp.nmove == 1);
    uint32_t user = rp.move[0];
    CHECK(call(c, dmg, 0, NULL, 0, &rp, 0) == 0 && rp.nmove == 1);
    uint32_t manager = rp.move[0];
    memset((void *)(uintptr_t)SMX_BUF, 0, SMX_SIZE);
    uint8_t arg[32] = {0};
    bk_u32(arg + 0, 1), bk_u32(arg + 4, 0x24), bk_u64(arg + 16, SMX_BUF), bk_u64(arg + 24, SMX_SIZE);
    uint64_t origin = tm_now(NULL);
    CHECK(call(c, user, 1, arg, 32, &rp, 4) == 0 && rp.nmove == 1);
    uint32_t id = bk_rd32(rp.data), stream = rp.move[0];
    CHECK(id == 1);
    uint32_t open[6] = {2, 2, 44100, 1024, 0, 0};   // a rate the mixer does not take
    CHECK(call(c, stream, 0, open, 24, &rp, 0) == 0x40299u);
    open[2] = 48000;
    CHECK(call(c, stream, 0, open, 24, &rp, 0) == 0);
    CHECK(call(c, stream, 0, open, 24, &rp, 0) == 0xc99u);   // already open
    CHECK(call(c, stream, 2, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 1);
    CHECK(call(c, stream, 5, NULL, 0, &rp, 0) == 0 && rp.ncopy == 1);
    uint32_t state_ev = rp.copy[0];
    uint32_t th[2] = {2, 100};
    CHECK(call(c, stream, 7, th, 8, &rp, 0) == 0 && rp.ncopy == 1);
    uint32_t data_ev = rp.copy[0];
    CHECK(poll_event(c, state_ev) == 0 && poll_event(c, data_ev) == 0);   // opened; 1,024 slots free of the 100 asked for
    AudioSink *sink = sysmix_stream_sink(id);
    CHECK(sink && audio_sink_kind(sink) == AUDIO_SINK_CAPTURE);

    smx_write_frames(0, 480);
    tm_set(origin + tick_ns(48000, 240, 1));
    settle();
    CHECK(smx_ld(0x44) == 0);   // not engaged: nothing is taken
    uint32_t att[4] = {id, 1, 0, 0};
    CHECK(call(c, manager, 700, att, 16, &rp, 0) == 0);   // AttachStream engages it
    CHECK(call(c, stream, 2, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 2);
    tm_set(origin + tick_ns(48000, 240, 2));
    CHECK(smx_await_read(240));
    tm_set(origin + tick_ns(48000, 240, 3));
    CHECK(smx_await_read(480));
    tm_set(origin + tick_ns(48000, 240, 4));
    settle();
    CHECK(smx_ld(0x44) == 480);   // the ring is empty
    size_t n;
    const float *d = audio_sink_capture_data(sink, &n);
    CHECK(d && n == 480);
    for (int i = 0; i < 480; i++) {
        float l = (float)((i - 240) * 50) / 32768.0f;
        CHECK(near_f(d[2 * i], l) && near_f(d[2 * i + 1], -l));
    }
    // the gain applies to what follows; a ramp is not supported
    float g[4] = {0.5f, 0, 0, 0};
    CHECK(call(c, stream, 4, g, 16, &rp, 0) == 0);
    CHECK(call(c, stream, 3, NULL, 0, &rp, 4) == 0 && !memcmp(rp.data, g, 4));
    int64_t ramp = 5;
    memcpy(g + 2, &ramp, 8);
    CHECK(call(c, stream, 4, g, 16, &rp, 0) == 0x40299u);
    smx_write_frames(480, 240);
    tm_set(origin + tick_ns(48000, 240, 5));
    CHECK(smx_await_read(720));
    d = audio_sink_capture_data(sink, &n);
    CHECK(n == 720);
    for (int i = 480; i < 720; i++) CHECK(near_f(d[2 * i], (float)((i - 240) * 50) / 32768.0f * 0.5f));
    // detached: the stream stays open and takes nothing more
    CHECK(call(c, manager, 800, att, 16, &rp, 0) == 0);
    CHECK(call(c, stream, 2, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 1);
    smx_write_frames(720, 240);
    tm_set(origin + tick_ns(48000, 240, 6));
    settle();
    CHECK(smx_ld(0x44) == 720);
    ST32(SMX_IDS, id);   // engaged again through the user session
    ClReq re = {.cmd = 2, .na = 1};
    re.a[0] = (ClBuf){SMX_IDS, 4, 1};
    CHECK(cl_send(c, user, &re, &rp, 0) == 0 && rp.result == 0);
    tm_set(origin + tick_ns(48000, 240, 7));
    CHECK(smx_await_read(960));
    // a write index outside the ring is a fault: logged, the stream stops
    smx_st(0x1004, 5000);
    tm_set(origin + tick_ns(48000, 240, 8));
    settle();
    CHECK(call(c, stream, 2, NULL, 0, &rp, 4) == 0 && bk_rd32(rp.data) == 0);
    close_handle(c, state_ev);
    close_handle(c, data_ev);
    CHECK(call(c, stream, 1, NULL, 0, &rp, 0) == 0);
    close_handle(c, stream);
    close_handle(c, manager);
    close_handle(c, user);
    close_handle(c, dmg);
    close_handle(c, smx);
    close_handle(c, sm);
    sv(c, 0x0A, 0, 0, 0, 0);
}

// ---- the hardware Opus overrides, with a fake decoder behind the seam (runtime/audio/opus.h)
static const char *const OPUS_SYMS[] = {
    "_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii",
    "_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi",
    "_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm",
    "_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE",
};
enum { O_WORK, O_INIT, O_RATE, O_CH, O_DECODE, O_RESET, O_FINALIZE, O_COUNT };
typedef struct FakeDec { int channels, count; } FakeDec;
static int g_fake_made, g_fake_freed;
static void *fake_create(int rate, int channels, char *why, size_t n) {
    (void)rate, (void)why, (void)n;
    FakeDec *d = calloc(1, sizeof *d);
    d->channels = channels;
    g_fake_made++;
    return d;
}
static void fake_destroy(void *d) { free(d); g_fake_freed++; }
static int fake_reset(void *d) { ((FakeDec *)d)->count = 0; return 0; }
// The packet's first byte is the frames it decodes to (0xff: a packet it refuses); frame i, channel ch of the n-th decode is
// n x 1000 + i x 2 + ch; the final range is the sum of the packet's bytes.
static int fake_decode(void *dec, const uint8_t *p, int len, int16_t *pcm, int max, uint32_t *range) {
    FakeDec *d = dec;
    if (len < 1 || p[0] == 0xff) return -4;
    int frames = p[0] < max ? p[0] : max;
    d->count++;
    for (int i = 0; i < frames; i++)
        for (int ch = 0; ch < d->channels; ch++) pcm[i * d->channels + ch] = (int16_t)(d->count * 1000 + i * 2 + ch);
    uint32_t sum = 0;
    for (int k = 0; k < len; k++) sum += p[k];
    *range = sum;
    return frames;
}
static const OpusApi g_fake_opus = {"a fake decoder", fake_create, fake_destroy, fake_reset, fake_decode};

static uint64_t g_opus_addr[O_COUNT];
// as the loader binds an import: before the dispatch table is sealed
static void opus_bind(void) {
    for (int i = 0; i < O_COUNT; i++) CHECK((g_opus_addr[i] = rt_hostfn_bind(OPUS_SYMS[i])) != 0);
}
static uint64_t opus_call(Cpu *a, int i, uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3) {
    a->x[0] = x0, a->x[1] = x1, a->x[2] = x2, a->x[3] = x3;
    a->x[4] = 0x9000, a->x[5] = 0;
    rt_call(a, g_opus_addr[i]);
    return a->x[0];
}
// One decode: the frame at in (size bytes offered) into the buffer at OUT (out_size bytes); the result, and what the call
// wrote to *consumed and *samples (0xdead and 0xbeef when it wrote nothing).
#define OPUS_IN (HEAP + 0x800)
#define OPUS_OUT (HEAP + 0x1000)
#define OPUS_RES (HEAP + 0x600)
static uint64_t opus_decode(Cpu *a, uint64_t dec, uint64_t in, uint64_t size, uint64_t out_size, uint64_t *consumed, int32_t *samples) {
    uint64_t c0 = 0xdead;
    int32_t s0 = 0xbeef;
    memcpy((void *)(uintptr_t)OPUS_RES, &c0, 8);
    memcpy((void *)(uintptr_t)(OPUS_RES + 8), &s0, 4);
    memset((void *)(uintptr_t)OPUS_OUT, 0, 0x400);
    a->x[0] = dec, a->x[1] = OPUS_RES, a->x[2] = OPUS_RES + 8, a->x[3] = OPUS_OUT, a->x[4] = out_size, a->x[5] = in, a->x[6] = size;
    rt_call(a, g_opus_addr[O_DECODE]);
    memcpy(consumed, (const void *)(uintptr_t)OPUS_RES, 8);
    memcpy(samples, (const void *)(uintptr_t)(OPUS_RES + 8), 4);
    return a->x[0];
}
static void put_frame(uint64_t at, uint32_t range, const uint8_t *packet, uint32_t n) {
    uint8_t *p = (uint8_t *)(uintptr_t)at;
    p[0] = (uint8_t)(n >> 24), p[1] = (uint8_t)(n >> 16), p[2] = (uint8_t)(n >> 8), p[3] = (uint8_t)n;
    p[4] = (uint8_t)(range >> 24), p[5] = (uint8_t)(range >> 16), p[6] = (uint8_t)(range >> 8), p[7] = (uint8_t)range;
    memcpy(p + 8, packet, n);
}
static int16_t out16(int i) { int16_t v; memcpy(&v, (const void *)(uintptr_t)(OPUS_OUT + 2u * (unsigned)i), 2); return v; }

static void opus_fake_case(void) {
    audio_opus_install(&g_fake_opus);
    Cpu a;
    memset(&a, 0, sizeof a);
    a.excl_addr = ~0ull;
    const uint64_t A = HEAP + 0x400, B = HEAP + 0x440;
    CHECK(opus_call(&a, O_WORK, 48000, 2, 0, 0) == 0x9000 && opus_call(&a, O_WORK, 12000, 1, 0, 0) == 0x9000);
    CHECK(opus_call(&a, O_WORK, 44100, 2, 0, 0) == 0 && opus_call(&a, O_WORK, 48000, 3, 0, 0) == 0);
    CHECK(opus_call(&a, O_INIT, A, 48000, 2, 0) == 0 && opus_call(&a, O_INIT, B, 24000, 1, 0) == 0);
    a.x[4] = 0x100;   // a work buffer smaller than the one the guest was told to reserve
    a.x[0] = A + 0x80, a.x[1] = 48000, a.x[2] = 2;
    rt_call(&a, g_opus_addr[O_INIT]);
    CHECK(a.x[0] == 0x1fcd1u);
    CHECK(opus_call(&a, O_INIT, A + 0x80, 44100, 2, 0) == 0x1fcd1u);   // a rate the decoder does not take
    CHECK(opus_call(&a, O_RATE, A, 0, 0, 0) == 48000 && opus_call(&a, O_CH, A, 0, 0, 0) == 2);
    CHECK(opus_call(&a, O_RATE, B, 0, 0, 0) == 24000 && opus_call(&a, O_CH, B, 0, 0, 0) == 1);
    CHECK(g_fake_made == 2);

    // three frames in a row in one input: each call consumes its own, the packet's size and 8
    const uint8_t p1[3] = {4, 7, 9}, p2[2] = {2, 1}, p3[2] = {2, 5};
    put_frame(OPUS_IN, 20, p1, 3);              // the final range matches
    put_frame(OPUS_IN + 11, 99, p2, 2);         // it does not: logged once, the samples kept
    put_frame(OPUS_IN + 21, 0, p3, 2);          // a header range of 0 is no recorded value: not checked
    uint64_t consumed;
    int32_t samples;
    CHECK(opus_decode(&a, A, OPUS_IN, 31, 0x400, &consumed, &samples) == 0 && consumed == 11 && samples == 4);
    for (int i = 0; i < 8; i++) CHECK(out16(i) == 1000 + i);   // decode 1, four stereo frames, interleaved
    CHECK(out16(8) == 0);                                       // nothing past what was made
    CHECK(opus_decode(&a, A, OPUS_IN + consumed, 20, 0x400, &consumed, &samples) == 0 && consumed == 10 && samples == 2);
    CHECK(out16(0) == 2000 && out16(3) == 2003);
    CHECK(opus_decode(&a, A, OPUS_IN + 21, 10, 0x400, &consumed, &samples) == 0 && consumed == 10 && samples == 2);
    CHECK(out16(0) == 3000);
    // a decoder's state is its own: B is on its first decode, mono, while A has made three
    CHECK(opus_decode(&a, B, OPUS_IN, 31, 0x400, &consumed, &samples) == 0 && consumed == 11 && samples == 4);
    CHECK(out16(0) == 1000 && out16(3) == 1006 && out16(4) == 0);   // mono: frame i is 1000 + 2i
    CHECK(opus_decode(&a, A, OPUS_IN, 31, 0x400, &consumed, &samples) == 0 && out16(0) == 4000);
    CHECK(opus_call(&a, O_RESET, A, 0, 0, 0) == 0);
    CHECK(opus_decode(&a, A, OPUS_IN, 31, 0x400, &consumed, &samples) == 0 && out16(0) == 1000);   // reset: the count starts again
    // the output buffer's size limits the frames: room for one stereo frame (4 bytes) gives one
    CHECK(opus_decode(&a, A, OPUS_IN, 31, 4, &consumed, &samples) == 0 && consumed == 11 && samples == 1 && out16(2) == 0);
    // refusals leave the results alone
    CHECK(opus_decode(&a, A, OPUS_IN, 7, 0x400, &consumed, &samples) == 0x1fcd1u && consumed == 0xdead && samples == 0xbeef);
    put_frame(OPUS_IN + 0x100, 0, p1, 3);
    CHECK(opus_decode(&a, A, OPUS_IN + 0x100, 10, 0x400, &consumed, &samples) == 0x1fcd1u);   // the header says 3 bytes, 2 are there
    const uint8_t bad[1] = {0xff};
    put_frame(OPUS_IN + 0x200, 0, bad, 1);
    CHECK(opus_decode(&a, A, OPUS_IN + 0x200, 9, 0x400, &consumed, &samples) == 0x1fcd1u && consumed == 0xdead);   // the decoder refuses it
    CHECK(opus_call(&a, O_FINALIZE, A, 0, 0, 0) == 0 && opus_call(&a, O_FINALIZE, B, 0, 0, 0) == 0);
    CHECK(g_fake_freed == 2);
    audio_opus_install(NULL);
}

static void guest_end(Cpu *c) { (void)c; }

// ---- host side
static const uint32_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static GuestFn entry_for(const char *name) {
    if (!strcmp(name, "release_clock")) return g_release_clock;
    if (!strcmp(name, "release_clock_32k")) return g_release_clock_32k;
    if (!strcmp(name, "cadence_running_stopped")) return g_cadence_running_stopped;
    if (!strcmp(name, "cadence_period")) return g_cadence_period;
    if (!strcmp(name, "cadence_no_burst")) return g_cadence_no_burst;
    if (!strcmp(name, "parse_ok")) return g_parse_ok;
    if (!strcmp(name, "flush_stop")) return g_flush_stop;
    if (!strcmp(name, "device_basic")) return g_device_basic;
    if (!strcmp(name, "auddmg_basic")) return g_auddmg_basic;
    if (!strcmp(name, "sysmix_stream")) return g_sysmix_stream;
    if (!strcmp(name, "stall_report")) return g_stall_report;
    if (!strcmp(name, "stall_healthy")) return g_stall_healthy;
    if (!strcmp(name, "stall_names")) return g_stall_names;
    if (!strcmp(name, "stall_no_update")) return g_stall_no_update;
    if (!strcmp(name, "mix_voice")) return g_mix_voice;
    if (!strcmp(name, "mix_voice_slots")) return g_mix_voice_slots;
    if (!strcmp(name, "mix_voice_slots_old")) return g_mix_voice_slots;
    if (!strcmp(name, "mix_formats")) return g_mix_formats;
    if (!strcmp(name, "mix_resample")) return g_mix_resample;
    if (!strcmp(name, "mix_adpcm")) return g_mix_adpcm;
    if (!strcmp(name, "mix_pool_refused")) return g_mix_pool_refused;
    if (!strcmp(name, "mix_route")) return g_mix_route;
    if (!strcmp(name, "mix_device_volume")) return g_mix_device_volume;
    if (!strcmp(name, "mix_biquad")) return g_mix_biquad;
    if (!strcmp(name, "mix_loop")) return g_mix_loop;
    if (!strcmp(name, "mix_stereo")) return g_mix_stereo;
    if (!strcmp(name, "mix_sink6")) return g_mix_sink6;
    if (!strcmp(name, "starve_empty_loop")) return g_starve_empty_loop;
    if (!strcmp(name, "stall_lock")) return g_stall_lock;
    return g_fatal;
}

static void boot(const char *folder) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x3, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    hos_init(folder, NULL, 0);
    static const uint64_t addrs[] = {G(1), G(2), G(3)};
    static GuestFn fns[3];
    fns[0] = entry_for(g_case), fns[1] = g_stall_waiter, fns[2] = guest_end;
    if (!strcmp(g_case, "opus_fake")) opus_bind();
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, addrs, fns, 3, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 16ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

// A case that hangs ends as a failure of its own after a minute of real time, not as the runner's timeout: a renderer that never
// returns from a frame shows as this line.
static void hang_guard(void *arg) {
    (void)arg;
    plat_sleep_ns(60ull * 1000000000ull);
    fprintf(stderr, "FAIL %s: still running after 60 s, a hang\n", g_case);
    fflush(stderr);
    _exit(1);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: audio_test CASE DIR\n");
        return 2;
    }
    g_case = argv[1];
    const char *dir = argv[2];
    plat_thread_start(hang_guard, NULL, 1ull << 18);
    static const AudioClockDriver drv = {.now = tm_now, .wait_until = tm_wait_until, .wake = tm_wake};
    audio_clock_drive(&drv);
    static const AudioClockDriver wd = {.now = wd_now, .wait_until = wd_wait_until, .wake = wd_wake};
    audio_watchdog_drive(&wd);
    if (!strcmp(g_case, "sink_null")) {
        sink_null_case();
        printf("ok %s\n", g_case);
        return 0;
    }
    if (!strcmp(g_case, "sink_capture")) {
        sink_capture_case(dir);
        printf("ok %s\n", g_case);
        return 0;
    }
    if (!strncmp(g_case, "mix_", 4) || !strncmp(g_case, "sysmix_", 7)) putenv("OOT2_AUDIO_SINK=capture");   // the mixing cases read the frames back
    boot(dir);
    if (!strcmp(g_case, "opus_fake")) {   // the overrides called from a host Cpu, with no guest thread
        opus_fake_case();
        printf("ok %s\n", g_case);
        return 0;
    }
    rt_kernel_run(G(1), 0);
    printf("ok %s\n", g_case);
    return 0;
}
