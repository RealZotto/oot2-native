// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the Horizon layer (runtime/hos): the service SVCs and sm: through hand-built messages in a guest
// thread's message buffer, the host-object handles they make, the census file and its rows, the override notes, and
// the boot driver's stop when no module defines nn::init::Start. Synthetic guest programs written by hand in the
// lifter's shape run on the kernel's guest threads; no game code is involved. Every file of runtime/hos is linked in,
// so this build is also the compile check of the overrides.
//
//   hos_test CASE DIR   runs one case with DIR as the game folder (census.tsv goes there); prints "ok CASE" and exits
//                       0, or exits 1 with a FAIL line. The cases that end in a named fatal exit with the runtime's
//                       code (tests/test_runtime_hos.py).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include "layout.h"
#include "endmark.h"
#include "oot2/boot.h"
#include "oot2/gfx.h"
#include "oot2/hid_motion.h"
#include "oot2/kernel.h"
#include "oot2/ledger.h"
#include "oot2/memory.h"
#include "plat.h"
#include "av1_fake.h"   // runtime/hos links the AV1 service; the fake stands for the decoder
#include "../../runtime/hos/hos.h"

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
#define R_INVALID_SIZE 0xCA01u
#define R_INVALID_HANDLE 0xE401u
#define SFCI 0x49434653u
#define SFCO 0x4f434653u

static uint64_t HEAP, g_bound;
// the hardware Opus decoder overrides (runtime/hos/ov_opus.c), bound as the loader binds an import; this build has no decoder
static const char *const OPUS_SYMBOLS[] = {
    "_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii",
    "_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi",
    "_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm",
    "_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE",
};
enum { A_OPUS_WORK, A_OPUS_INIT, A_OPUS_RATE, A_OPUS_CH, A_OPUS_DECODE, A_OPUS_RESET, A_OPUS_FINALIZE, A_COUNT };
static uint64_t g_audio[A_COUNT];
// the AV1 decoder service's first step (runtime/hos/ov_codec.c), which is a named fatal when the build has no decoder
static const char MOVIE_SYMBOL[] =
    "_ZN2nn5codec34InitializeAv1DecoderByRemoteMemoryEPNS0_14Av1DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE";
static uint64_t g_movie;
static char g_nvn_name[96];
static const char *g_case;

// An override with a note, which the test binds as the loader would, and one no module names.
static void host_bound(Cpu *c) { ret_x(c, 7); }
HOS_OVERRIDE("hos_test_bound", host_bound, HOS_OFFLINE, "seven, as a console without the service answers");
static void host_unbound(Cpu *c) { (void)c; }
HOS_OVERRIDE("hos_test_unbound", host_unbound, HOS_STUB, "never bound");

// ---- guest helpers
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
static inline uint32_t *msg(Cpu *c) { return (uint32_t *)(uintptr_t)c->tpidrro; }

static uint32_t open_port(Cpu *c, const char *port) {
    memcpy((void *)(uintptr_t)(HEAP + 0x100), port, strlen(port) + 1);
    CHECK(sv(c, 0x1F, 0, HEAP + 0x100, 0, 0) == 0);
    return (uint32_t)c->x[1];
}

// A TIPC request of command cmd with data bytes (and the client's process id when pid is set): the result.
static uint32_t tipc(Cpu *c, uint32_t h, uint32_t cmd, const void *data, uint32_t size, int pid) {
    uint32_t *m = msg(c), *p = m + 2;
    memset(m, 0, 0x100);
    m[0] = 16 + cmd;
    m[1] = (size + 3) / 4 | (pid ? 1u << 31 : 0);
    if (pid) {
        *p++ = 1;
        p += 2;
    }
    memcpy(p, data, size);
    CHECK(sv(c, 0x21, h, 0, 0, 0) == 0);
    CHECK(!(m[1] >> 31));
    return m[2];
}

// A CMIF control request of command cmd: the result; *moved gets the moved handle, *word the first data word.
static uint32_t control(Cpu *c, uint32_t h, uint32_t cmd, uint32_t *moved, uint32_t *word) {
    uint32_t *m = msg(c);
    memset(m, 0, 0x100);
    m[0] = 5;
    m[1] = 8;   // the 16-byte header 16-byte aligned, with libnx's slack
    m[4] = SFCI, m[6] = cmd;
    CHECK(sv(c, 0x21, h, 0, 0, 0) == 0);
    unsigned at = 2;
    if (m[1] >> 31) {
        CHECK((m[2] >> 5 & 0xf) == 1);
        *moved = m[3];
        at = 4;
    }
    at = (at + 3) & ~3u;   // the payload starts 16-byte aligned
    CHECK(m[at] == SFCO);
    if (word) *word = m[at + 4];
    return m[at + 2];
}

// ---- the Opus stand-ins, called from a host Cpu as rt_call does
static uint64_t acall(Cpu *a, int i, uint64_t x0, uint64_t x1) {
    a->x[0] = x0, a->x[1] = x1;
    rt_call(a, g_audio[i]);
    return a->x[0];
}
static int opus_case(void) { return !strcmp(g_case, "opus") || !strcmp(g_case, "opus_libopus"); }
// An NVN2 entry point called from the host thread by its synthetic address.
static uint64_t nvn(Cpu *a, const char *name, uint64_t x0, uint64_t x1, uint64_t x2) {
    a->x[0] = x0, a->x[1] = x1, a->x[2] = x2;
    rt_call(a, nvn_proc_address(name));
    return a->x[0];
}


// The nn::hid six-axis overrides, the nn::pl and nn::pctl overrides: bound as the loader binds an import.
static const char *const HID_SYMBOLS[] = {
    "_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE",
    "_ZN2nn3hid17StopSixAxisSensorERKNS0_19SixAxisSensorHandleE",
    "_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE",
    "_ZN2nn3hid6detail22GetSixAxisSensorStatesEPiPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE",
    "_ZN2nn3hid21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE",
};
enum { H_START, H_STOP, H_STATES, H_DSTATES, H_STATE, H_COUNT };
static uint64_t g_hid[H_COUNT];
static const char *const PL_SYMBOLS[] = {
    "_ZN2nn2pl22GetSharedFontLoadStateENS0_14SharedFontTypeE",
    "_ZN2nn2pl20GetSharedFontAddressENS0_14SharedFontTypeE",
    "_ZN2nn2pl17GetSharedFontSizeENS0_14SharedFontTypeE",
    "_ZN2nn2pl17RequestLoadNxFontEPvm",
    "_ZN2nn4pctl33CheckAgcPermissionUnderMinimumAgeERKNS_7account3UidE",
};
enum { P_STATE, P_ADDR, P_SIZE, P_REQ, P_PCTL, P_COUNT };
static uint64_t g_pl[P_COUNT];

static int case_is(const char *prefix) { return !strncmp(g_case, prefix, strlen(prefix)); }

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static void make_dir(const char *path) {
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

// A made-up font file: an sfnt magic and then a pattern that depends on the seed, nothing of any real font.
static size_t write_blob(const char *path, size_t size, uint8_t seed, const char *magic4) {
    uint8_t *b = malloc(size);
    CHECK(b != NULL);
    for (size_t i = 0; i < size; i++) b[i] = (uint8_t)(seed + i * 7 + (i >> 8));
    if (magic4) memcpy(b, magic4, 4);
    else memcpy(b, "\0\1\0\0", 4);
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    CHECK(fwrite(b, 1, size, f) == size);
    fclose(f);
    free(b);
    return size;
}
static void expect_blob(uint64_t addr, size_t size, uint8_t seed) {
    const uint8_t *g = (const uint8_t *)(uintptr_t)addr;
    CHECK(g[0] == 0 && g[1] == 1 && g[2] == 0 && g[3] == 0);
    for (size_t i = 4; i < size; i++) CHECK(g[i] == (uint8_t)(seed + i * 7 + (i >> 8)));
}

// ---- the six-axis state as the guest reads it
typedef struct {
    int64_t delta_ns, sampling;
    float acc[3], gyro[3], angle[3], dir[9];
    uint32_t attributes, reserved;
} SixState;
_Static_assert(sizeof(SixState) == 0x60, "six-axis state layout");
#define SIX_OUT (HEAP + 0x1000)
#define SIX_HANDLE (HEAP + 0x100)

static uint64_t hcall(Cpu *a, int i, uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3) {
    a->x[0] = x0, a->x[1] = x1, a->x[2] = x2, a->x[3] = x3;
    rt_call(a, g_hid[i]);
    return a->x[0];
}
static void six_copy(SixState *out, int n) { memcpy(out, (const void *)(uintptr_t)SIX_OUT, (size_t)n * sizeof *out); }
// GetSixAxisSensorStates(out, n, &handle): the count it gives back, the states in out
static int six_read(Cpu *a, int n, SixState *out) {
    memset((void *)(uintptr_t)SIX_OUT, 0xcc, (size_t)n * sizeof *out);
    int got = (int)hcall(a, H_STATES, SIX_OUT, (uint64_t)n, SIX_HANDLE, 0);
    six_copy(out, n);
    return got;
}
static void six_feed(float gz, uint64_t dt) {
    const float flat[3] = {0, 0, -1}, gyro[3] = {0, 0, gz};
    hid_motion_feed(flat, gyro, dt, plat_now_ns());
}
static int is_rest(const SixState *s) {
    return s->attributes == 0 && s->delta_ns == 0 && s->acc[0] == 0 && s->acc[1] == 0 && s->acc[2] == -1 &&
           s->gyro[0] == 0 && s->gyro[1] == 0 && s->gyro[2] == 0 && s->angle[2] == 0 && s->dir[0] == 1 &&
           s->dir[4] == 1 && s->dir[8] == 1 && s->dir[1] == 0;
}
#define NEARF(a, b, eps) (fabs((double)(a) - (double)(b)) <= (eps))

// The case hid_motion: a synthetic sensor stream through the host motion layer, read through the nn::hid overrides.
// With replay set (hid_motion_replay) an input file is being replayed: the pad's motion must not reach the game.
static void hid_motion_case(int replay) {
    Cpu a;
    memset(&a, 0, sizeof a);
    a.excl_addr = ~0ull;
    uint32_t handle = 0x20003;
    memcpy((void *)(uintptr_t)SIX_HANDLE, &handle, sizeof handle);
    SixState s[16];
    // not started: samples fed are not read, every state is a controller at rest and the sampling numbers grow
    for (int i = 0; i < 3; i++) six_feed(0.5f, 4000000);
    CHECK(six_read(&a, 16, s) == 16);
    for (int i = 0; i < 16; i++) CHECK(is_rest(&s[i]));
    for (int i = 1; i < 16; i++) CHECK(s[i].sampling == s[0].sampling - i);
    int64_t last_rest = s[0].sampling;
    CHECK(!hid_motion_wanted());
    CHECK(hcall(&a, H_START, SIX_HANDLE, 0, 0, 0) == 0);
    CHECK(hid_motion_wanted());   // the window thread now turns the pad's sensors on
    // started, nothing fed since the start: still at rest, and the numbers still grow
    CHECK(six_read(&a, 16, s) == 16);
    for (int i = 0; i < 16; i++) CHECK(is_rest(&s[i]));
    CHECK(s[0].sampling > last_rest);
    last_rest = s[0].sampling;
    // five samples of a quarter-turn-a-second-per-ten-ticks ramp about the face's normal, 4 ms apart
    for (int k = 1; k <= 5; k++) six_feed(0.1f * (float)k, 4000000);
    CHECK(six_read(&a, 16, s) == 16);
    if (replay) {
        for (int i = 0; i < 16; i++) CHECK(is_rest(&s[i]));   // the replayed input replaces the pad's motion too
        CHECK(hcall(&a, H_STOP, SIX_HANDLE, 0, 0, 0) == 0);
        CHECK(!hid_motion_wanted());
        return;
    }
    for (int i = 0; i < 5; i++) {
        CHECK(s[i].attributes == 1);
        CHECK(s[i].delta_ns == 4000000);
        CHECK(NEARF(s[i].gyro[2], 0.1 * (5 - i), 1e-6) && s[i].gyro[0] == 0 && s[i].gyro[1] == 0);
        CHECK(NEARF(s[i].acc[2], -1, 1e-6));
    }
    CHECK(NEARF(s[0].angle[2], 0.1 * 15 * 0.004, 1e-6));   // the sum of the ramp over the ticks, in revolutions
    CHECK(s[0].sampling > last_rest);                       // above every number given at rest
    for (int i = 1; i < 16; i++) CHECK(s[i].sampling == s[0].sampling - i);
    // the states the pad has not sent yet repeat the oldest sample's pose and move nothing
    for (int i = 5; i < 16; i++) {
        CHECK(s[i].attributes == 1 && s[i].delta_ns == 0);
        CHECK(s[i].gyro[0] == 0 && s[i].gyro[1] == 0 && s[i].gyro[2] == 0);
        CHECK(NEARF(s[i].angle[2], s[4].angle[2], 1e-9) && NEARF(s[i].dir[0], s[4].dir[0], 1e-9));
    }
    // the same samples read again are the same states with the same numbers
    SixState again[16];
    CHECK(six_read(&a, 16, again) == 16);
    CHECK(!memcmp(s, again, sizeof s));
    // the other forms: the detail count form, and the single state form
    int32_t count = -1;
    memset((void *)(uintptr_t)SIX_OUT, 0xcc, sizeof s);
    CHECK(hcall(&a, H_DSTATES, HEAP + 0x200, SIX_OUT, 16, SIX_HANDLE) == 0);
    memcpy(&count, (const void *)(uintptr_t)(HEAP + 0x200), sizeof count);
    CHECK(count == 16);
    six_copy(again, 16);
    CHECK(!memcmp(s, again, sizeof s));
    memset((void *)(uintptr_t)SIX_OUT, 0xcc, sizeof s);
    CHECK(hcall(&a, H_STATE, SIX_OUT, SIX_HANDLE, 0, 0) == 0);
    six_copy(again, 1);
    CHECK(!memcmp(&s[0], &again[0], sizeof s[0]));
    // one more sample moves the newest number by one
    six_feed(0.6f, 4000000);
    CHECK(six_read(&a, 16, again) == 16);
    CHECK(again[0].sampling == s[0].sampling + 1 && NEARF(again[0].gyro[2], 0.6, 1e-6));
    // the pad stops sending (a sample from long ago): at rest again, numbers still above the live ones
    int64_t last_live = again[0].sampling;
    hid_motion_clear();
    six_feed(0.5f, 4000000);
    {
        const float flat[3] = {0, 0, -1}, gyro[3] = {0, 0, 0.5f};
        hid_motion_feed(flat, gyro, 4000000, plat_now_ns() - 1000000000ull);   // a second old
    }
    CHECK(six_read(&a, 16, s) == 16);
    for (int i = 0; i < 16; i++) CHECK(is_rest(&s[i]));
    CHECK(s[0].sampling > last_live);
    // the menu's swallow
    hid_motion_clear();
    six_feed(0.5f, 4000000);
    hid_motion_suppress(1);
    CHECK(six_read(&a, 16, s) == 16 && is_rest(&s[0]));
    hid_motion_suppress(0);
    CHECK(six_read(&a, 16, s) == 16 && s[0].attributes == 1);
    // stopped: at rest and the pad's sensors may go off
    CHECK(hcall(&a, H_STOP, SIX_HANDLE, 0, 0, 0) == 0);
    CHECK(!hid_motion_wanted());
    int64_t before = s[0].sampling;
    CHECK(six_read(&a, 16, s) == 16);
    for (int i = 0; i < 16; i++) CHECK(is_rest(&s[i]));
    CHECK(s[0].sampling > before);
    for (int i = 0; i < H_COUNT; i++) {
        const HosNote *n = hos_note(HID_SYMBOLS[i]);
        CHECK(n && !strcmp(n->status, HOS_IMPLEMENTED) && n->why[0]);
    }
}

// The cases pl_font and pl_font_env: the shared font calls, with a made-up font file in the game folder (and, for the
// second, another that OOT2_SHARED_FONT names and that wins).
static void pl_font_case(int env) {
    Cpu a;
    memset(&a, 0, sizeof a);
    a.excl_addr = ~0ull;
    for (int t = 0; t < 6; t++) {   // the load state is Loaded for every type the console has
        a.x[0] = (uint64_t)t;
        rt_call(&a, g_pl[P_STATE]);
        CHECK((uint32_t)a.x[0] == 1);
    }
    const size_t want = env ? 2222 : 3001;
    const uint8_t seed = env ? 90 : 17;
    uint64_t first = 0;
    for (int t = 0; t < 6; t++) {
        a.x[0] = (uint64_t)t;
        rt_call(&a, g_pl[P_ADDR]);
        uint64_t addr = a.x[0];
        CHECK(addr != 0);
        if (!t) first = addr;
        CHECK(addr == first);   // one font serves every type
        a.x[0] = (uint64_t)t;
        rt_call(&a, g_pl[P_SIZE]);
        CHECK(a.x[0] == want);
        expect_blob(addr, want, seed);
    }
    // RequestLoadNxFont: a buffer of guest memory is accepted and left as it is
    memset((void *)(uintptr_t)(HEAP + 0x4000), 0x5a, 0x1000);
    a.x[0] = HEAP + 0x4000, a.x[1] = 0x1000;
    rt_call(&a, g_pl[P_REQ]);
    const uint8_t *buf = (const uint8_t *)(uintptr_t)(HEAP + 0x4000);
    for (int i = 0; i < 0x1000; i++) CHECK(buf[i] == 0x5a);
    for (int i = P_STATE; i <= P_REQ; i++) {   // every override of the group is in the census, with its reason
        const HosNote *n = hos_note(PL_SYMBOLS[i]);
        CHECK(n && n->why[0] && strcmp(n->status, HOS_GAP));   // answers: no longer a named fatal
    }
}

// The case pctl: the parental-control age check answers without a stop
static void pctl_case(void) {
    Cpu a;
    memset(&a, 0, sizeof a);
    a.excl_addr = ~0ull;
    memset((void *)(uintptr_t)(HEAP + 0x300), 0x11, 16);   // a Uid
    a.x[0] = HEAP + 0x300;
    a.x[1] = 0xdead;
    rt_call(&a, g_pl[P_PCTL]);
    CHECK((a.x[0] & 0xff) == 0);
    const HosNote *n = hos_note(PL_SYMBOLS[P_PCTL]);
    CHECK(n && n->why[0] && strcmp(n->status, HOS_GAP));
}

// ---- guest programs (each ends its thread)
static void g_sm(Cpu *c) {
    uint32_t h = open_port(c, "sm:");
    CHECK(h != 0);
    CHECK(tipc(c, h, 0, NULL, 0, 1) == 0);   // RegisterClient
    CHECK((msg(c)[1] & 0x3ff) == 1);         // one data word, the result: TIPC puts no header before the data
    uint32_t size = 0, clone = 0;
    CHECK(control(c, h, 3, &clone, &size) == 0 && (size & 0xffff) == 0x8000);   // QueryPointerBufferSize
    CHECK(control(c, h, 2, &clone, NULL) == 0 && clone && clone != h);         // CloneCurrentObject
    CHECK(tipc(c, clone, 0, NULL, 0, 1) == 0);   // the clone is the same service
    CHECK(sv(c, 0x16, clone, 0, 0, 0) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == R_INVALID_HANDLE);   // closed
    CHECK(sv(c, 0x21, h, 0, 0, 0) == R_INVALID_HANDLE);
    // transfer memory: a record with a handle; a size that is no whole page is refused
    CHECK(sv(c, 0x15, 0, HEAP + 0x10000, 0x2000, 0) == 0);
    uint32_t t = (uint32_t)c->x[1];
    CHECK(t != 0);
    CHECK(sv(c, 0x21, t, 0, 0, 0) == R_INVALID_HANDLE);   // no session
    CHECK(sv(c, 0x16, t, 0, 0, 0) == 0);
    CHECK(sv(c, 0x15, 0, HEAP + 0x10000, 0x1234, 0) == R_INVALID_SIZE);
    // a host function called from main+0x120: the census names that call site
    c->x[30] = G(1) + 0x24;
    c->pc = g_bound;
    rt_dispatch(c);
    CHECK(c->x[0] == 7);
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void g_fatal(Cpu *c) {
    if (!strcmp(g_case, "port")) open_port(c, "xyz:");
    uint32_t h = open_port(c, "sm:");
    if (!strcmp(g_case, "service") || !strcmp(g_case, "service_census")) {
        uint64_t name = 0;
        memcpy(&name, "lm", 2);
        uint32_t r = tipc(c, h, 1, &name, 8, 0);   // GetServiceHandle
        if (!strcmp(g_case, "service_census")) {
            CHECK(r == 0x415);   // census mode: sm: refuses the name, the run goes on
            CHECK(tipc(c, h, 1, &name, 8, 0) == 0x415);   // the same name again: refused again, listed once
            // 40 more names: refused every time, but the log keeps 32 names in all
            for (int i = 0; i < 40; i++) {
                char n[9];
                snprintf(n, sizeof n, "svc%05d", i);
                memcpy(&name, n, 8);
                CHECK(tipc(c, h, 1, &name, 8, 0) == 0x415);
            }
        }
    } else if (!strcmp(g_case, "command")) {
        tipc(c, h, 7, NULL, 0, 0);
    } else if (!strcmp(g_case, "nvn") || !strcmp(g_case, "nvn_census")) {
        // the first NVN2 entry point of the list with no body, called from main+0x250 as the game calls its table
        const char *name = NULL;
        for (int i = 0; !name && i < nvn_proc_count(); i++)
            if (!nvn_proc_has_body(i)) name = nvn_proc_name(i);
        CHECK(name != NULL);
        snprintf(g_nvn_name, sizeof g_nvn_name, "%s", name);
        fprintf(stderr, "case nvn: %s\n", name);
        c->x[30] = G(2) + 0x54;
        c->pc = nvn_proc_address(name);
        rt_dispatch(c);
    } else if (!strcmp(g_case, "movie")) {
        c->x[0] = HEAP + 0x500, c->x[1] = 0x1000, c->x[2] = 64, c->x[3] = 64;   // decoder, memory size, width, height
        c->x[30] = G(2) + 0x54;
        c->pc = g_movie;
        rt_dispatch(c);
    } else if (case_is("pl_")) {
        // the shared font calls that stop by name, called from main+0x250 as the game calls its table
        int which = !strcmp(g_case, "pl_font_missing") || !strcmp(g_case, "pl_font_not_sfnt") ? P_ADDR
                    : !strcmp(g_case, "pl_font_type") ? P_STATE : P_REQ;
        c->x[0] = !strcmp(g_case, "pl_font_missing") ? 2 : !strcmp(g_case, "pl_font_type") ? 9 : 0;
        c->x[1] = 0x1000;   // pl_nx_buffer: a null buffer
        if (!strcmp(g_case, "pl_nx_size")) c->x[0] = HEAP + 0x4000, c->x[1] = 0;            // no bytes
        if (!strcmp(g_case, "pl_nx_range")) c->x[0] = HEAP + 0x4000, c->x[1] = 1ull << 32;   // runs out of guest memory
        c->x[30] = G(2) + 0x54;
        c->pc = g_pl[which];
        rt_dispatch(c);
    } else if (!strcmp(g_case, "fault")) {
        // a read of unmapped memory by guest code, with a frame record the fault names: x30 and the x29 chain
        uint64_t frame[2] = {0, G(1) + 0x44};
        memcpy((void *)(uintptr_t)(HEAP + 0x800), frame, sizeof frame);
        c->x[29] = HEAP + 0x800;
        c->x[30] = G(1) + 0x24;
        printf("read %llx\n", (unsigned long long)*(volatile uint64_t *)(uintptr_t)0x7f00000000ull);
    } else if (!strcmp(g_case, "fault_chain")) {
        // the same read, with an x29 that is aligned and below the address limit but not mapped: the report's own read of the
        // frame chain faults, and the first fault stays the reason (one FATAL line)
        c->x[29] = 0x7e00000000ull;
        c->x[30] = G(1) + 0x24;
        printf("read %llx\n", (unsigned long long)*(volatile uint64_t *)(uintptr_t)0x7f00000000ull);
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void guest_end(Cpu *c) { (void)c; }

// ---- host side
static const uint32_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static void boot(const char *folder) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x3, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    g_bound = rt_hostfn_bind("hos_test_bound");   // as the loader binds an import
    CHECK(g_bound != 0);
    if (opus_case())
        for (int i = 0; i < A_COUNT; i++) CHECK((g_audio[i] = rt_hostfn_bind(OPUS_SYMBOLS[i])) != 0);
    if (!strcmp(g_case, "movie")) {
        CHECK((g_movie = rt_hostfn_bind(MOVIE_SYMBOL)) != 0);
        g_fake_refuse_open = 1;   // a build with no dav1d kit
    }
    if (!strcmp(g_case, "nvn") || !strcmp(g_case, "nvn_census") || !strcmp(g_case, "census_files") ||
        !strcmp(g_case, "collect_note") || !strcmp(g_case, "collect_none"))
        nvn_procs_register();   // as main.c does, before the layer
    if (case_is("hid_motion"))
        for (int i = 0; i < H_COUNT; i++) CHECK((g_hid[i] = rt_hostfn_bind(HID_SYMBOLS[i])) != 0);
    if (case_is("pl_") || !strcmp(g_case, "pctl"))
        for (int i = 0; i < P_COUNT; i++) CHECK((g_pl[i] = rt_hostfn_bind(PL_SYMBOLS[i])) != 0);
    set_env("OOT2_SHARED_FONT", NULL);
    if (case_is("pl_font") || case_is("pl_nx")) {
        char path[1200];
        snprintf(path, sizeof path, "%s/fonts", folder);
        make_dir(path);
        snprintf(path, sizeof path, "%s/fonts/shared-font.ttf", folder);
        if (!strcmp(g_case, "pl_font_not_sfnt")) write_blob(path, 500, 3, "BFTT");
        else if (strcmp(g_case, "pl_font_missing") && strcmp(g_case, "pl_font_type")) write_blob(path, 3001, 17, NULL);
        if (!strcmp(g_case, "pl_font_env")) {
            snprintf(path, sizeof path, "%s/elsewhere.ttf", folder);
            write_blob(path, 2222, 90, NULL);
            set_env("OOT2_SHARED_FONT", path);
        }
    }
    if (!strcmp(g_case, "hid_motion_replay")) {   // an input file is replayed: the real input, and the pad's motion, are off
        char path[1200];
        snprintf(path, sizeof path, "%s/replay.txt", folder);
        FILE *f = fopen(path, "wb");
        CHECK(f != NULL);
        fputs("at 0 for 1 a\n", f);
        fclose(f);
        set_env("OOT2_INPUT_FILE", path);
    } else {
        set_env("OOT2_INPUT_FILE", NULL);
    }
    hos_init(folder, NULL, 0);
    static const uint64_t addrs[] = {G(1), G(2), G(3), G(4)};
    static const GuestFn fns[] = {g_sm, g_fatal, guest_end, guest_end};
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, addrs, fns, 4, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 16ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

static char *read_census(const char *folder) {
    char path[1024];
    snprintf(path, sizeof path, "%s/census.tsv", folder);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    return buf;
}

#ifdef OOT2_OPUS
// ---- the Opus overrides with libopus behind them (built with the kit; tests/test_runtime_opus.py)
#include <math.h>
#include <opus/opus.h>

#define OPUS_OUT (HEAP + 0x10000)   // the guest's pcm buffer, OPUS_OUT_SIZE bytes
#define OPUS_IN (HEAP + 0x100000)   // the guest's input: a frame and bytes after it
#define OPUS_OUT_SIZE 0x20000u
#define CODEC_INVALID 0x1fcd1u

static void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24), p[1] = (uint8_t)(v >> 16), p[2] = (uint8_t)(v >> 8), p[3] = (uint8_t)v;
}

// One decode call through the override: x0 the object, x1 and x2 where consumed and samples go, x3 the pcm and x4 its size,
// x5 the input and x6 its size. Returns the codec result.
static uint64_t opus_call(Cpu *a, uint64_t obj, uint64_t out_size, uint64_t in_size) {
    a->x[0] = obj, a->x[1] = HEAP + 0x500, a->x[2] = HEAP + 0x508, a->x[3] = OPUS_OUT, a->x[4] = out_size;
    a->x[5] = OPUS_IN, a->x[6] = in_size;
    rt_call(a, g_audio[A_OPUS_DECODE]);
    return a->x[0];
}

// A stream that libopus's encoder makes from a tone is wrapped in Nintendo's frame (the 8-byte big-endian header of the payload
// size and the final range) and decoded by the override. The samples are what libopus's own decoder gives for the same
// packets, `samples` counts frames of one channel, `consumed` is the frame and not the bytes the call offered, the output past
// the samples is untouched whatever room the guest gives (the override caps its decode at 120 ms), a header range that is not
// the decoder's is logged once and the samples are kept, a range of 0 is not compared, and a payload size the input cannot
// hold, like a pcm buffer too small for the packet, is refused.
static void opus_stream(Cpu *a, uint64_t obj, int rate, int channels, int wrong_ranges) {
    const int frame = rate / 50;
    a->x[0] = obj, a->x[1] = (uint64_t)rate, a->x[2] = (uint64_t)channels, a->x[3] = 0, a->x[4] = 0x9000, a->x[5] = 0;
    rt_call(a, g_audio[A_OPUS_INIT]);
    CHECK(a->x[0] == 0);
    int err = 0;
    OpusEncoder *enc = opus_encoder_create(rate, channels, OPUS_APPLICATION_AUDIO, &err);
    CHECK(enc && err == OPUS_OK);
    CHECK(opus_encoder_ctl(enc, OPUS_SET_BITRATE(64000 * channels)) == OPUS_OK);
    OpusDecoder *ref = opus_decoder_create(rate, channels, &err);
    CHECK(ref && err == OPUS_OK);
    int16_t tone[960 * 2], want[960 * 2];
    uint8_t packet[1500];
    for (int p = 0; p < 12; p++) {
        for (int i = 0; i < frame; i++)
            for (int c = 0; c < channels; c++)
                tone[i * channels + c] =
                    (int16_t)lrint(12000.0 * sin(6.283185307179586 * (c ? 1000.0 : 500.0) * (double)(p * frame + i) / rate));
        int n = opus_encode(enc, tone, frame, packet, sizeof packet);
        CHECK(n > 0);
        opus_uint32 range = 0;
        CHECK(opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&range)) == OPUS_OK);
        CHECK(opus_decode(ref, packet, n, want, frame, 0) == frame);
        uint32_t header_range = range;   // packets 0 to 7 carry the encoder's range; then wrong ones; the last carries 0
        if (p >= 8 && p < 8 + wrong_ranges) header_range = range ^ 0x5a5a5a5au;
        if (p == 11) header_range = 0;
        uint8_t *in = (uint8_t *)(uintptr_t)OPUS_IN;
        put_be32(in, (uint32_t)n), put_be32(in + 4, header_range);
        memcpy(in + 8, packet, (size_t)n);
        memset(in + 8 + n, 0xa5, 17);   // the bytes after the frame
        memset((void *)(uintptr_t)OPUS_OUT, 0x55, OPUS_OUT_SIZE);
        memset((void *)(uintptr_t)(HEAP + 0x500), 0xff, 16);
        CHECK(opus_call(a, obj, OPUS_OUT_SIZE, (uint64_t)n + 8 + 17) == 0);
        uint64_t consumed;
        int32_t samples;
        memcpy(&consumed, (const void *)(uintptr_t)(HEAP + 0x500), 8);
        memcpy(&samples, (const void *)(uintptr_t)(HEAP + 0x508), 4);
        CHECK(consumed == (uint64_t)n + 8);   // the frame, not the 17 bytes offered after it
        CHECK(samples == frame);              // for one channel
        const uint8_t *out = (const uint8_t *)(uintptr_t)OPUS_OUT;
        size_t bytes = (size_t)frame * (size_t)channels * sizeof(int16_t);
        CHECK(!memcmp(out, want, bytes));
        for (size_t i = bytes; i < OPUS_OUT_SIZE; i++) CHECK(out[i] == 0x55);
    }
    uint8_t *in = (uint8_t *)(uintptr_t)OPUS_IN;
    put_be32(in, 1000), put_be32(in + 4, 0);   // a payload size the 40 bytes of input cannot hold
    CHECK(opus_call(a, obj, OPUS_OUT_SIZE, 40) == CODEC_INVALID);
    int n = opus_encode(enc, tone, frame, packet, sizeof packet);
    CHECK(n > 0);
    put_be32(in, (uint32_t)n), put_be32(in + 4, 0);
    memcpy(in + 8, packet, (size_t)n);
    CHECK(opus_call(a, obj, (uint64_t)frame * (uint64_t)channels * 2 - 2, (uint64_t)n + 8) == CODEC_INVALID);   // a sample short
    CHECK(acall(a, A_OPUS_FINALIZE, obj, 0) == 0);
    opus_decoder_destroy(ref);
    opus_encoder_destroy(enc);
}
#endif

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: hos_test CASE DIR\n");
        return 2;
    }
    g_case = argv[1];
    const char *dir = argv[2];
    if (!strcmp(g_case, "sm")) {
        boot(dir);
        rt_kernel_run(G(1), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        static const char header[] = "kind\tkey\tstatus\tcalls\tfirst_at\twhy\n";
        CHECK(!strncmp(t, header, sizeof header - 1));
        CHECK(strstr(t, "\nsvc\t0x1f ConnectToNamedPort\timplemented\t1\t0x7000000040 (main+0x40)\t\n"));
        CHECK(strstr(t, "\nsvc\t0x15 CreateTransferMemory\timplemented\t2\t"));
        CHECK(strstr(t, "\ncommand\tsm: tipc 0\timplemented\t2\t"));
        CHECK(strstr(t, "\ncommand\tsm: control 3\timplemented\t1\t"));
        CHECK(strstr(t, "\ncommand\tsm: control 2\timplemented\t1\t"));
        CHECK(strstr(t, "\nimport\thos_test_bound\toffline\t1\t0x7000000120 (main+0x120)\tseven, as a console without "
                        "the service answers\n"));
        CHECK(strstr(t, "\nd3\tsetjmp\timplemented\t0\t"));
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
    } else if (!strcmp(g_case, "notes")) {
        boot(dir);
        const HosNote *n = hos_note("hos_test_bound");
        CHECK(n && !strcmp(n->status, HOS_OFFLINE));
        CHECK(!hos_note("hos_test_absent"));
        int found = 0;
        for (int i = 0; rt_hostfn_unbound(i); i++) {
            CHECK(strcmp(rt_hostfn_unbound(i), "hos_test_bound"));
            found += !strcmp(rt_hostfn_unbound(i), "hos_test_unbound");
        }
        CHECK(found == 1);
    } else if (!strcmp(g_case, "opus")) {
        boot(dir);
        Cpu a;
        memset(&a, 0, sizeof a);
        a.excl_addr = ~0ull;
        // the hardware Opus decoder of a build with no decoder: its format is kept, a decode consumes its frame (8 bytes of
        // header and the packet the header sizes) and makes no sample
        a.x[0] = 48000, a.x[1] = 2, a.x[2] = 0;
        rt_call(&a, g_audio[A_OPUS_WORK]);
        CHECK(a.x[0] == 0x9000);
        a.x[0] = HEAP + 0x400, a.x[1] = 24000, a.x[2] = 1, a.x[3] = 0, a.x[4] = 0x9000, a.x[5] = 0;
        rt_call(&a, g_audio[A_OPUS_INIT]);
        CHECK(a.x[0] == 0);
        a.x[0] = HEAP + 0x400;
        rt_call(&a, g_audio[A_OPUS_RATE]);
        CHECK(a.x[0] == 24000);
        a.x[0] = HEAP + 0x400;
        rt_call(&a, g_audio[A_OPUS_CH]);
        CHECK(a.x[0] == 1);
        memset((void *)(uintptr_t)(HEAP + 0x500), 0xff, 16);
        memset((void *)(uintptr_t)(HEAP + 0x700), 0, 8);
        ((uint8_t *)(uintptr_t)(HEAP + 0x700))[3] = 115;   // a packet of 115 bytes, big endian, final range 0
        a.x[0] = HEAP + 0x400, a.x[1] = HEAP + 0x500, a.x[2] = HEAP + 0x508, a.x[3] = HEAP + 0x600, a.x[4] = 0x1000;
        a.x[5] = HEAP + 0x700, a.x[6] = 125;
        rt_call(&a, g_audio[A_OPUS_DECODE]);
        CHECK(a.x[0] == 0);
        uint64_t consumed;
        uint32_t samples;
        memcpy(&consumed, (const void *)(uintptr_t)(HEAP + 0x500), 8);
        memcpy(&samples, (const void *)(uintptr_t)(HEAP + 0x508), 4);
        CHECK(consumed == 123 && samples == 0);   // 8 and the 115 bytes, not the 125 the call offered
        CHECK(acall(&a, A_OPUS_RESET, HEAP + 0x400, 0) == 0);
        CHECK(acall(&a, A_OPUS_FINALIZE, HEAP + 0x400, 0) == 0);
        for (int i = 0; i < A_COUNT; i++) {   // every override of the file is in the census, with its reason
            const HosNote *n = hos_note(OPUS_SYMBOLS[i]);
            CHECK(n && !strcmp(n->status, HOS_IMPLEMENTED) && n->why[0]);
        }
    } else if (!strcmp(g_case, "opus_libopus")) {
#ifdef OOT2_OPUS
        boot(dir);
        Cpu a;
        memset(&a, 0, sizeof a);
        a.excl_addr = ~0ull;
        // mono first, whose last packet has a header range of 0 and logs nothing; then stereo with two wrong header ranges,
        // which log once
        opus_stream(&a, HEAP + 0x420, 16000, 1, 0);
        opus_stream(&a, HEAP + 0x400, 48000, 2, 2);
#else
        fprintf(stderr, "opus_libopus needs a libopus kit\n");
        return 2;
#endif
    } else if (!strcmp(g_case, "service_census")) {
        boot(dir);
        rt_kernel_run(G(2), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        CHECK(!strncmp(t, "#census-mode OOT2_CENSUS=1", 26));   // marked first
        CHECK(strstr(t, "\nservice\tlm\tcensus-gap\t2\t"));   // met twice, one row
        CHECK(strstr(t, "\nservice\tsvc00039\tcensus-gap\t1\t"));   // every name has its row, listed in the log or not
        CHECK(strstr(t, "\nkind\tkey\tstatus\tcalls\tfirst_at\twhy\n"));
    } else if (!strcmp(g_case, "nvn_census")) {
        // census mode: an NVN2 entry point with no body is passed and the census row says so
        boot(dir);
        rt_kernel_run(G(2), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        char want[400];
        snprintf(want, sizeof want, "\nnvn\t%s\tcensus-gap\t1\t0x7000000250 (main+0x250)\t[census mode] an NVN2 entry point "
                 "with no body: the run answered 0 and went on\n", g_nvn_name);
        CHECK(strstr(t, want));
        CHECK(!strstr(t, "named-fatal\t1\t0x7000000250"));
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
    } else if (!strcmp(g_case, "census_files")) {
        // census.tsv in every mode carries the front end's own counters; census mode also keeps census-nvn.tsv and
        // rewrites both while the run goes on (rt_census_flush), never after the run's own end has been written
        boot(dir);
        char path[1100];
        snprintf(path, sizeof path, "%s/census-nvn.tsv", dir);
        // one recording through the entry points (device, command buffer with control memory, begin, end with a handle,
        // end with nothing open): the front end's counters the census carries are not all zero
        Cpu a;
        memset(&a, 0, sizeof a);
        a.excl_addr = ~0ull;
        const uint64_t b = HEAP + 0x1000, d = HEAP + 0x2000, o = HEAP + 0x6000;
        nvn(&a, "nvnDeviceBuilderSetDefaults", b, 0, 0);
        CHECK((nvn(&a, "nvnDeviceInitialize", d, b, 0) & 0xff) == 1);
        CHECK((nvn(&a, "nvnCommandBufferInitialize", o, d, 0) & 0xff) == 1);
        nvn(&a, "nvnCommandBufferAddControlMemory", o, HEAP + 0x8000, 0x1000);
        nvn(&a, "nvnCommandBufferBeginRecording", o, 0, 0);
        CHECK(nvn(&a, "nvnCommandBufferEndRecording", o, 0, 0) != 0);
        CHECK(nvn(&a, "nvnCommandBufferEndRecording", o, 0, 0) == 0);
        if (rt_census_on()) {
            rt_census_flush();
            const char *t = read_census(dir);
            CHECK(!strncmp(t, "#census-mode OOT2_CENSUS=1", 26));
            CHECK(strstr(t, "\nstop\trunning\tcensus\t1\t\t[census mode] the run was still going when this was written\n"));
            FILE *f = fopen(path, "rb");
            CHECK(f != NULL);
            char head[256] = {0};
            CHECK(fread(head, 1, sizeof head - 1, f) > 0);
            fclose(f);
            CHECK(!strncmp(head, "#census-mode OOT2_CENSUS=1: not a default run\n#presents 0\n", 56));
        }
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
        CHECK(strstr(t, "\nnvn-stat\trecordings\timplemented\t1\t\tinline 0, slow 1, ended with a handle 1, ended empty 1\n"));
        CHECK(strstr(t, "\nnvn-stat\tcallbacks\timplemented\t0\t\tmemory callbacks: control 0, command 0; recordings begun "
                        "over left-out records 0\n"));
        CHECK(strstr(t, "\nnvn-stat\twaitevents\timplemented\t0\t\tWaitEvents met 0, unsatisfied at their place 0\n"));
        CHECK(strstr(t, "\nnvn-stat\twaitclasses\timplemented\t0\t\tby a GPU signal in order 0, by a CPU signal before 0, "
                        "already satisfied 0, reached after a release 0, unsatisfied 0 (later by GPU 0, later by CPU 0, not "
                        "satisfied by the end 0); GPU signals in order: of its own queue 0, of another 0\n"));
        CHECK(strstr(t, "\nnvn-stat\tpools\timplemented\t0\t\tmemory pools made 0, live 0 (most at once 0), 0 bytes live "
                        "(most 0); made with GPU_NO_ACCESS 0; GPU address window in use 0 bytes (most 0 of 68719476736)\n"));
        CHECK(strstr(t, "\nnvn-stat\tpools-virtual\timplemented\t0\t\tmade 0, live 0 (most at once 0), 0 bytes live "
                        "(most 0)\n"));
        CHECK(strstr(t, "\nnvn-stat\tpresents\timplemented\t0\t\tnvnQueuePresentTexture calls that reached the back end\n"));
        if (rt_census_on()) {
            rt_census_flush();   // the run's end is written: a later flush leaves it alone
            t = read_census(dir);
            CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n") && !strstr(t, "\nstop\trunning"));
        } else {
            CHECK(t[0] != '#');   // no census-mode mark in a default run
            FILE *f = fopen(path, "rb");
            CHECK(f == NULL);     // and no calls-per-frame file
        }
    } else if (!strcmp(g_case, "collect_note") || !strcmp(g_case, "collect_none")) {
        // A collection run (settings.h collect_variants) says so in census.tsv, and only then: three draws noted over two
        // distinct (stage, answer) pairs give the banner and the collected-draws row; none noted gives neither.
        boot(dir);
        const int collected = !strcmp(g_case, "collect_note");
        if (collected) {
            int first = 0;
            CHECK(ledger_collect_note("vs.a", "another scalar type", &first) == 1 && first == 1);
            CHECK(ledger_collect_note("vs.b", "another scalar type", &first) == 2 && first == 1);
            CHECK(ledger_collect_note("vs.a", "another scalar type", &first) == 3 && first == 0);
        }
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        const char *banner = "#collect-variants: NOT A PROOF RUN: 3 draws were skipped, not drawn, to collect shader variants\n";
        if (collected) {
            CHECK(!strncmp(t, banner, strlen(banner)));   // first line of a default run
            CHECK(strstr(t, "\nnvn-stat\tcollected-draws\timplemented\t3\t\tNOT A PROOF RUN: 3 draws were skipped, not "
                            "drawn, for collection (collect_variants), over 2 distinct stage and answer pairs; their ledger "
                            "rows are written\n"));
        } else {
            CHECK(!strstr(t, "collect-variants") && !strstr(t, "collected-draws") && !strstr(t, "NOT A PROOF RUN"));
        }
    } else if (!strcmp(g_case, "census_end")) {
        // a census-mode run ends with its own exit code after the census has its stop row
        boot(dir);
        rt_census_end(RT_EXIT_CENSUS, "[census mode] the case ended at its frame limit");
    } else if (case_is("hid_motion")) {
        boot(dir);
        hid_motion_case(!strcmp(g_case, "hid_motion_replay"));
    } else if (!strcmp(g_case, "pl_font") || !strcmp(g_case, "pl_font_env")) {
        boot(dir);
        pl_font_case(!strcmp(g_case, "pl_font_env"));
    } else if (!strcmp(g_case, "pctl")) {
        boot(dir);
        pctl_case();
    } else if (!strcmp(g_case, "service") || !strcmp(g_case, "port") || !strcmp(g_case, "command") ||

               !strcmp(g_case, "nvn") || !strcmp(g_case, "movie") ||
               !strcmp(g_case, "fault") || !strcmp(g_case, "fault_chain") || case_is("pl_")) {
        boot(dir);
        rt_kernel_run(G(2), 0);
    } else if (!strcmp(g_case, "boot")) {
        boot(dir);
        rt_kernel_run(hos_boot_entry(), 0);
    } else {
        fprintf(stderr, "unknown case %s\n", g_case);
        return 2;
    }
    printf("ok %s\n", g_case);
    return 0;
}
