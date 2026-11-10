// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the replay's workload counters (oot2/replay_count.h, runtime/core/replay_count.c), driven by
// tests/test_runtime_replay_count.py: `replay_count_test <case>`, each case in its own process.
//
//   off        no environment variable: a counter site does nothing (no block is even claimed)
//   threads    four threads count into their own blocks; the sum is exact
//   windows    a line group every `window` presents with the change over the window, a hook's values as changes, and the exit
//              line with the totals
//   lines      every counter has its name in the exit line, once, and no line is longer than the log allows
//   present    1,300 presents (the driver sets the environment): the lines the window in the environment asks for
//   stop       a stop (rt_fatal) prints the exit line through the stop path
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "layout.h"
#include "endmark.h"
#include "oot2/replay_count.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                            \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

enum { THREADS = 4, EACH = 100000 };

#ifdef _WIN32
static DWORD WINAPI worker(LPVOID arg) {
#else
static void *worker(void *arg) {
#endif
    (void)arg;
    for (int i = 0; i < EACH; i++) {
        RC_ADD(RC_DRAW_CLASSIC, 1);
        if (i % 1000 == 0) RC_ADD(RC_TEX_BINDS, 3);
    }
    return 0;
}

static void case_off(void) {
    CHECK(rc_on == 0);
    RC_ADD(RC_SUB_N, 5);
    CHECK(rc_tl == NULL);   // the site did not even look for a block
    uint64_t sum[RC_N];
    rc_sum(sum);
    for (int i = 0; i < RC_N; i++) CHECK(sum[i] == 0);
}

static void case_threads(void) {
    rc_enable(600);
    RC_ADD(RC_DRAW_CLASSIC, 7);
#ifdef _WIN32
    HANDLE h[THREADS];
    for (int i = 0; i < THREADS; i++) h[i] = CreateThread(NULL, 0, worker, NULL, 0, NULL);
    for (int i = 0; i < THREADS; i++) WaitForSingleObject(h[i], INFINITE), CloseHandle(h[i]);
#else
    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, worker, NULL);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
#endif
    uint64_t sum[RC_N];
    rc_sum(sum);
    CHECK(sum[RC_DRAW_CLASSIC] == 7 + (uint64_t)THREADS * EACH);
    CHECK(sum[RC_TEX_BINDS] == (uint64_t)THREADS * (EACH / 1000) * 3);
    CHECK(sum[RC_SUB_N] == 0);
    rc_print("exit", 0);
}

static uint64_t g_hook_value;
static void hook(RcLine *l) {
    rc_group(l, "hooked");
    rc_put(l, "hook_value", g_hook_value);
}

static void case_windows(void) {
    rc_enable(10);
    rc_hook(hook);
    rc_hook(hook);   // once only
    RC_ADD(RC_SUB_N, 3);
    g_hook_value = 100;
    for (uint64_t p = 1; p <= 9; p++) rc_present(p);   // no line before the window is full
    rc_present(10);
    RC_ADD(RC_SUB_N, 4);
    g_hook_value = 250;
    for (uint64_t p = 11; p <= 25; p++) rc_present(p);
    rc_print("exit", 25);
}

static void case_lines(void) {
    rc_enable(600);
    for (int i = 0; i < RC_N; i++) rc_add_(i, (uint64_t)i + 1);
    rc_print("exit", 0);
}

static void case_present(void) {
    for (uint64_t p = 1; p <= 1300; p++) {
        RC_ADD(RC_SUB_N, 1);
        rc_present(p);
    }
}

static void case_stop(void) {
    rc_enable(600);
    RC_ADD(RC_SUB_N, 9);
    rt_fatal("stopped on purpose");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: replay_count_test <case>\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "off")) case_off();
    else if (!strcmp(cs, "threads")) case_threads();
    else if (!strcmp(cs, "windows")) case_windows();
    else if (!strcmp(cs, "lines")) case_lines();
    else if (!strcmp(cs, "present")) case_present();
    else if (!strcmp(cs, "stop")) case_stop();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
