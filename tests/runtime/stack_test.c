// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Cases of tests/test_runtime_stack.py: a first touch of guest memory is an access violation that the runtime's
// vectored handler settles (plat_fault_hook), and Windows builds the exception frame for it on the faulting thread's own
// stack, under the stack pointer. The frame holds the processor's extended state, which is larger on some processors
// than on others, and the kernel grows a stack only a few pages past its committed end. So a thread must start with
// stack committed well past its first frames (plat_thread_start; the thread that installed the hook). Each case takes
// the violation with the stack pointer `deep` bytes below where the thread's code runs, on stack nothing has touched
// yet: the frame lands where only a stack that was committed ahead can take it, as it would on a processor whose
// state is that large. Without the headroom the process ends with the violation (exit code 0xC0000005) and the handler
// never runs. Windows only; elsewhere the program prints ok and does nothing.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plat.h"

#ifdef _WIN32
#include <windows.h>

#define DEEP (128ull << 10)   // well past what the kernel grows a stack by, well inside the headroom
#define PAGE 4096

static volatile char *g_page;       // reserved, no access: the first store to it is the violation
static volatile LONG g_settled;     // violations the hook settled
static volatile LONG g_done;

static int hook(uint64_t addr, int write, uint64_t host_pc) {
    (void)write, (void)host_pc;
    if (addr - (uint64_t)(uintptr_t)g_page >= PAGE) return 0;
    if (!plat_commit((uint64_t)(uintptr_t)g_page, PAGE, PLAT_RW)) return 0;
    InterlockedIncrement(&g_settled);
    return 1;
}

// One violation with rsp below the caller's frame by `below` bytes (0: where the code runs).
static __attribute__((noinline)) void fault_below(uint64_t below) {
    uint64_t here = (uint64_t)(uintptr_t)__builtin_frame_address(0), saved;
    uint64_t sp = (here - below - 4096) & ~15ull;
    __asm__ volatile("movq %%rsp, %0\n\t"
                     "movq %2, %%rsp\n\t"
                     "movb $1, (%1)\n\t"
                     "movq %0, %%rsp\n\t"
                     : "=&r"(saved)
                     : "r"(g_page), "r"(sp)
                     : "memory");
}

static void thread_body(void *arg) {
    fault_below(*(uint64_t *)arg);
    InterlockedExchange(&g_done, 1);
}

static PlatFibre *g_main_fibre;

static void fibre_body(void *arg) {
    fault_below(*(uint64_t *)arg);
    InterlockedExchange(&g_done, 1);
    for (;;) plat_fibre_switch(NULL, g_main_fibre);   // a fibre never returns
}

static void arm(void) {
    g_page = VirtualAlloc(NULL, PAGE, MEM_RESERVE, PAGE_NOACCESS);
    if (!g_page) {
        fprintf(stderr, "no page to reserve\n");
        exit(3);
    }
    plat_fault_hook(hook);
}

static int settled_once(const char *what) {
    if (g_settled != 1) {
        fprintf(stderr, "%s: %ld violations settled, one expected\n", what, (long)g_settled);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *c = argv[1];
    uint64_t deep = DEEP, shallow = 0;
    if (!strcmp(c, "thread_shallow") || !strcmp(c, "thread_deep")) {
        arm();
        uint64_t *depth = !strcmp(c, "thread_deep") ? &deep : &shallow;
        PlatThread *t = plat_thread_start(thread_body, depth, 256ull << 20);   // a guest thread's host stack
        if (!t) return 3;
        for (int i = 0; i < 3000 && !g_done; i++) plat_sleep_ns(10000000);
        if (!g_done || !settled_once(c)) return 1;
    } else if (!strcmp(c, "small_thread_deep")) {
        arm();
        PlatThread *t = plat_thread_start(thread_body, &deep, 1ull << 20);   // a helper thread: 1 MiB reserved
        if (!t) return 3;
        for (int i = 0; i < 3000 && !g_done; i++) plat_sleep_ns(10000000);
        if (!g_done || !settled_once(c)) return 1;
    } else if (!strcmp(c, "fibre_deep")) {
        arm();
        g_main_fibre = plat_fibre_self();
        PlatFibre *f = g_main_fibre ? plat_fibre_new(256ull << 20, fibre_body, &deep) : NULL;   // a guest fibre's stack
        if (!f) return 3;
        plat_fibre_switch(g_main_fibre, f);
        plat_fibre_unself(g_main_fibre);
        if (!g_done || !settled_once(c)) return 1;
    } else if (!strcmp(c, "first_thread_deep")) {
        arm();   // the thread that installs the hook is the process's first, with the default 1 MiB reserve
        fault_below(deep);
        if (!settled_once(c)) return 1;
    } else {
        return 2;
    }
    printf("ok %s\n", c);
    return 0;
}
#else
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    printf("ok %s\n", argv[1]);
    return 0;
}
#endif
