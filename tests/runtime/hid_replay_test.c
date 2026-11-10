// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of input by tick (oot2/hid_replay.h) through the HID service: the full-key state read of
// runtime/hos/ov_hid.c, called as the guest calls it, on the virtual clock driven by the test. Every file of runtime/hos
// is linked in, as in hos_test.c. Nothing is read from a dump.
//
//   hid_replay_test run DIR   boots the Horizon layer with DIR as the game folder (so OOT2_INPUT_FILE and
//                             OOT2_INPUT_RECORD, if set, are read by hos_init as the program reads them), then plays the
//                             script DIR/script.txt: each line is
//                                 TICK READS BUTTONS LX LY RX RY
//                             and sets the clock to TICK ticks after the first line's tick (the first state read is
//                             tick 0 of the input file), publishes that as the real input (BUTTONS in hex), and reads the
//                             full-key state READS times in that tick. Each read is printed as
//                                 state TICK BUTTONS LX LY RX RY ATTRIBUTES
//                             with BUTTONS in hex. At the end the report is written (census.tsv) and "ok run" printed.
//                             A fatal at start-up (a malformed input file) exits with the runtime's code.
//                             If DIR/main exists it is a plain module the test loads first, as the program does, so that
//                             a store the input file's markers read can live in its data (oot2/store.h). A script line
//                                 P OFFSET HEXBYTES
//                             writes those bytes at OFFSET (hex) in the main module's image before the next line runs: how
//                             the test changes the store while the game runs. Three more script lines build what a
//                             few bytes cannot:
//                                 W OFFSET HEAPOFF   the guest address of the heap plus HEAPOFF, as a 64-bit word at OFFSET
//                                                    of the main module (a pointer to a table that lives in the heap)
//                                 F HEAPOFF COUNT HEX   COUNT times the bytes HEX (up to 64 bytes), from the heap plus
//                                                    HEAPOFF (a table of many cells)
//                                 D OFFSET           the page of the main module at OFFSET (hex, page aligned, in its data)
//                                                    is given no access, as the game's SetMemoryPermission would: the next
//                                                    read of the store from that page faults, and the runtime stops by name
//   hid_replay_test run_then_fatal DIR   the same, ending in a fatal of the case's own instead of the report call
//   hid_replay_test run_then_exit DIR    the same, ending in a stop with exit code 3 and no clean-up (TerminateProcess;
//                                        _Exit elsewhere): a process killed from outside, which closes and flushes
//                                        nothing, so the record holds only what each closed run flushed
//
// The driving script and the files are written by tests/test_runtime_hid_replay.py, which computes what every read must
// give on its own.
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "layout.h"
#include "endmark.h"
#include "oot2/boot.h"
#include "oot2/gfx.h"
#include "oot2/hid_input.h"
#include "oot2/kernel.h"
#include "oot2/loader.h"
#include "oot2/memory.h"
#include "oot2/vsync.h"
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
#define ORIGIN 5000000000ull        // the driven time of tick 0: any value
#define FIRST_TICK 500ull           // the tick the script's first line is at: the file's tick 0 is not the clock's

static uint64_t HEAP, g_init, g_ids, g_styles, g_states;   // the nn::hid overrides, bound as the loader binds imports
static uint64_t g_now = ORIGIN;     // the driven clock's time
static uint64_t drv_now(void *ctx) {
    (void)ctx;
    return __atomic_load_n(&g_now, __ATOMIC_ACQUIRE);
}
static void drv_wait(void *ctx, uint64_t deadline) {   // never called: the clock thread is not started
    (void)ctx, (void)deadline;
    fprintf(stderr, "FAIL: the clock thread waited\n");
    exit(1);
}

static void guest_end(Cpu *c) { (void)c; }

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
    char mpath[4200];
    snprintf(mpath, sizeof mpath, "%s/main", folder);
    FILE *mf = fopen(mpath, "rb");
    if (mf) {
        fclose(mf);
        RtModuleFile mfile = {"main", mpath, MB, MOD_MAIN_SIZE, NULL};
        rt_load_modules(&mfile, 1);
    }
    g_init = rt_hostfn_bind("_ZN2nn3hid14InitializeNpadEv");   // before the layer and the dispatch table, as the loader does
    g_ids = rt_hostfn_bind("_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm");
    g_styles = rt_hostfn_bind("_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE");
    g_states = rt_hostfn_bind("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj");
    CHECK(g_init && g_ids && g_styles && g_states);
    RtVsyncDriver d = {.origin_ns = ORIGIN, .now = drv_now, .wait_until = drv_wait};
    rt_vsync_drive(&d);
    hos_init(folder, NULL, 0);   // reads OOT2_INPUT_FILE and OOT2_INPUT_RECORD
    static const uint64_t addrs[] = {G(1)};
    static const GuestFn fns[] = {guest_end};
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, addrs, fns, 1, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 64ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

static void set_tick(uint64_t n) { __atomic_store_n(&g_now, rt_vsync_due_ns(n), __ATOMIC_RELEASE); }

typedef struct { int64_t sampling; uint64_t buttons; int32_t lx, ly, rx, ry; uint32_t attributes, pad; } NpadState;

static int run(const char *dir, int fatal) {
    boot(dir);
    Cpu a;
    memset(&a, 0, sizeof a);
    a.excl_addr = ~0ull;
    // the game's start-up calls: the controller, the ids it wants, the full-key style
    rt_call(&a, g_init);
    uint32_t list[1] = {0};   // NpadId No1
    memcpy((void *)(uintptr_t)(HEAP + 0x100), list, sizeof list);
    a.x[0] = HEAP + 0x100, a.x[1] = 1;
    rt_call(&a, g_ids);
    a.x[0] = 1;   // full key
    rt_call(&a, g_styles);

    char path[4200];
    snprintf(path, sizeof path, "%s/script.txt", dir);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    uint64_t first = 0;
    int have_first = 0;
    char line[256];
    int64_t last_sampling = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long long tick, reads, buttons;
        int lx, ly, rx, ry;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (line[0] == 'P') {
            unsigned long long off;
            char hexbytes[256];
            CHECK(sscanf(line, "P %llx %255s", &off, hexbytes) == 2 && strlen(hexbytes) % 2 == 0);
            for (size_t i = 0; i < strlen(hexbytes) / 2; i++) {
                unsigned b;
                CHECK(sscanf(hexbytes + 2 * i, "%2x", &b) == 1);
                *(volatile uint8_t *)(uintptr_t)(MB + off + i) = (uint8_t)b;
            }
            continue;
        }
        if (line[0] == 'W') {
            unsigned long long off, at;
            CHECK(sscanf(line, "W %llx %llx", &off, &at) == 2);
            uint64_t v = HEAP + at;
            memcpy((void *)(uintptr_t)(MB + off), &v, sizeof v);
            continue;
        }
        if (line[0] == 'F') {
            unsigned long long at, count;
            char hex[160];
            CHECK(sscanf(line, "F %llx %llu %159s", &at, &count, hex) == 3 && strlen(hex) % 2 == 0 && strlen(hex) <= 128);
            uint8_t cell[64];
            size_t n = strlen(hex) / 2;
            for (size_t i = 0; i < n; i++) {
                unsigned b;
                CHECK(sscanf(hex + 2 * i, "%2x", &b) == 1);
                cell[i] = (uint8_t)b;
            }
            for (unsigned long long i = 0; i < count; i++) memcpy((void *)(uintptr_t)(HEAP + at + n * i), cell, n);
            continue;
        }
        if (line[0] == 'D') {
            unsigned long long off;
            CHECK(sscanf(line, "D %llx", &off) == 1);
            Cpu p;
            memset(&p, 0, sizeof p);
            p.x[0] = MB + off, p.x[1] = 0x1000, p.x[2] = 0;   // SetMemoryPermission(addr, size, None)
            rt_svc(&p, 0x02, 0);
            CHECK(p.x[0] == 0);
            continue;
        }
        CHECK(sscanf(line, "%llu %llu %llx %d %d %d %d", &tick, &reads, &buttons, &lx, &ly, &rx, &ry) == 7);
        if (!have_first) first = tick, have_first = 1;
        set_tick(FIRST_TICK + (tick - first));
        HidHostInput in = {.buttons = buttons, .lx = lx, .ly = ly, .rx = rx, .ry = ry};
        hid_host_input_publish(&in);
        for (unsigned long long r = 0; r < reads; r++) {
            a.x[0] = HEAP + 0x200, a.x[1] = 1, a.x[2] = HEAP + 0x100;
            rt_call(&a, g_states);
            CHECK(a.x[0] == 1);
            NpadState s;
            memcpy(&s, (const void *)(uintptr_t)(HEAP + 0x200), sizeof s);
            CHECK(s.sampling > last_sampling);   // the sampling numbers keep growing, whatever is replayed
            last_sampling = s.sampling;
            printf("state %llu %" PRIx64 " %d %d %d %d %u\n", tick, s.buttons, s.lx, s.ly, s.rx, s.ry, s.attributes);
        }
    }
    fclose(f);
    if (fatal == 2) {   // killed from outside: no atexit, no flush of any stream, no stop path
        fflush(stdout);
#ifdef _WIN32
        TerminateProcess(GetCurrentProcess(), 3);   // the C runtime's _Exit flushes the open streams here
#endif
        _Exit(3);
    }
    if (fatal) rt_fatal("the case stopped the run");   // every stop closes the recorder's last run
    hos_census_write("the case ended");   // the report; this is also where the recorder closes its last run
    printf("ok run\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: hid_replay_test CASE DIR\n");
        return 2;
    }
    if (!strcmp(argv[1], "run")) return run(argv[2], 0);
    if (!strcmp(argv[1], "run_then_fatal")) return run(argv[2], 1);
    if (!strcmp(argv[1], "run_then_exit")) return run(argv[2], 2);
    fprintf(stderr, "unknown case %s\n", argv[1]);
    return 2;
}
