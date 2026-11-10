// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Logging and the three ways a run stops (runtime/include/oot2/runtime.h, exit codes). A fatal error prints one
// FATAL line first and flushes it, then any detail, so a detail that faults cannot hide the reason. The stop hook (the
// census, runtime/hos/census.c) runs last, after the detail.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/runtime.h"
#include "plat.h"

#define LINE_MAX_BYTES 2048
#define BACKTRACE_FRAMES 16

static int g_stopping;   // a fatal error is being reported; a second one exits at once
static void (*g_stop_hook)(int code, const char *line);
static void (*g_stop_flush)(void);
#define STOP_WRITERS 6
static void (*g_stop_writers[STOP_WRITERS])(int code, const char *line);
static int g_nwriters;
static int g_hook_ran;
static char g_stop_line[LINE_MAX_BYTES];   // the FATAL line, for the stop hook

static size_t grow(size_t n, int added, size_t cap) {
    if (added > 0) n += (size_t)added;
    return n < cap ? n : cap;
}

// prefix, message and suffix on one line, cut to LINE_MAX_BYTES, written with a single call; kept for the stop hook
// when keep is set.
static void put_line(const char *prefix, const char *fmt, va_list ap, const char *suffix, int keep) {
    // Census mode (runtime.h) marks every line it writes: after the prefix of a log line, at the end of a FATAL line. The
    // first line of a census-mode run is the mode's own notice, which the first call here makes.
    int census = rt_census_on();
    char buf[LINE_MAX_BYTES];
    const size_t cap = sizeof buf - 1;   // one byte stays free for the newline
    size_t n = grow(0, snprintf(buf, cap + 1, "%s", prefix), cap);
    if (census && !keep) n = grow(n, snprintf(buf + n, cap + 1 - n, "[census mode] "), cap);
    n = grow(n, vsnprintf(buf + n, cap + 1 - n, fmt, ap), cap);
    if (suffix) n = grow(n, snprintf(buf + n, cap + 1 - n, "%s", suffix), cap);
    if (census && keep) n = grow(n, snprintf(buf + n, cap + 1 - n, " [census mode]"), cap);
    if (keep) {
        memcpy(g_stop_line, buf, n);
        g_stop_line[n] = 0;
    }
    buf[n++] = '\n';
    plat_write_stderr(buf, n);
}

void rt_vlog(const char *prefix, const char *fmt, va_list ap) { put_line(prefix, fmt, ap, NULL, 0); }

void rt_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    put_line("[rt] ", fmt, ap, NULL, 0);
    va_end(ap);
}

void rt_set_stop_hook(void (*hook)(int code, const char *line)) { g_stop_hook = hook; }

void rt_set_stop_flush(void (*flush)(void)) { g_stop_flush = flush; }

void rt_add_stop_writer(void (*writer)(int code, const char *line)) {
    if (g_nwriters == STOP_WRITERS) rt_fatal("rt_add_stop_writer: %d writers already", STOP_WRITERS);
    g_stop_writers[g_nwriters++] = writer;
}

void rt_run_stop_hook(int code, const char *line) {
    if ((g_stop_hook || g_stop_flush || g_nwriters) && !__atomic_exchange_n(&g_hook_ran, 1, __ATOMIC_ACQ_REL)) {
        if (g_stop_flush) g_stop_flush();
        for (int i = 0; i < g_nwriters; i++) g_stop_writers[i](code, line ? line : "");
        if (g_stop_hook) g_stop_hook(code, line ? line : "");
    }
}

static void begin_stop(int code) {
    if (__atomic_exchange_n(&g_stopping, 1, __ATOMIC_ACQ_REL)) {
        plat_timer_release();
        _Exit(code);
    }
}

// Every exit of the runtime gives the system timer's request back (the timer resolution hold; plat.h): here, plat_exit and a normal
// return from main. plat_timer_release is lock-free and silent, so it stays out of the way of a fatal that hangs or
// faults in a hook, and it is called before the hook for that reason; plat_exit is not used here because it flushes the
// streams, which can deadlock inside a fatal. A new exit site that skips it costs nothing lasting (the system drops the
// request when the process ends) but the tests of the timer resolution hold would not know it.
__attribute__((noreturn)) static void end_stop(int code) {
    fflush(stderr);
    plat_timer_release();
    rt_run_stop_hook(code, g_stop_line);
    fflush(stdout);
    fflush(stderr);
    _Exit(code);
}

void rt_fatal(const char *fmt, ...) {
    begin_stop(RT_EXIT_FATAL);
    va_list ap;
    va_start(ap, fmt);
    put_line("FATAL: ", fmt, ap, NULL, 1);
    va_end(ap);
    end_stop(RT_EXIT_FATAL);
}

static void where_suffix(const Cpu *c, char *out, size_t size) {
    char pc[96], lr[96];
    snprintf(out, size, " (pc %s, lr %s)", rt_guest_name(c->pc, pc, sizeof pc), rt_guest_name(c->x[30], lr, sizeof lr));
}

__attribute__((noreturn)) static void unimplemented(const Cpu *c, const char *kind, int trace, const char *fmt,
                                                    va_list ap) {
    begin_stop(RT_EXIT_UNIMPLEMENTED);
    char prefix[96], where[200];
    snprintf(prefix, sizeof prefix, "FATAL: unimplemented %s: ", kind);
    where_suffix(c, where, sizeof where);
    put_line(prefix, fmt, ap, where, 1);
    if (trace) rt_guest_backtrace(c);
    end_stop(RT_EXIT_UNIMPLEMENTED);
}

void rt_unimplemented(const Cpu *c, const char *kind, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    unimplemented(c, kind, 0, fmt, ap);
}

void rt_unimplemented_trace(const Cpu *c, const char *kind, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    unimplemented(c, kind, 1, fmt, ap);
}

// The guest's frame records: x29 points at {previous x29, x30}. Bounded, and only along aligned, rising addresses.
void rt_guest_backtrace(const Cpu *c) {
    uint64_t fp = c->x[29];
    for (int i = 0; i < BACKTRACE_FRAMES; i++) {
        if (!fp || (fp & 15) || fp >= RT_ADDRESS_LIMIT) return;
        uint64_t next = LD64(fp), lr = LD64(fp + 8);
        char name[96];
        rt_log("  frame %d: %s", i, rt_guest_name(lr, name, sizeof name));
        if (next <= fp) return;
        fp = next;
    }
}

void rt_guest_stop(const Cpu *c, const char *fmt, ...) {
    begin_stop(RT_EXIT_GUEST_STOP);
    char where[200];
    where_suffix(c, where, sizeof where);
    va_list ap;
    va_start(ap, fmt);
    put_line("FATAL: guest stop: ", fmt, ap, where, 1);
    va_end(ap);
    rt_guest_backtrace(c);
    end_stop(RT_EXIT_GUEST_STOP);
}

void rt_brk(Cpu *c, uint32_t imm, uint64_t pc) {
    c->pc = pc;
    rt_guest_stop(c, "brk #%#x", imm);
}

void rt_udf(Cpu *c, uint64_t pc) {
    c->pc = pc;
    rt_guest_stop(c, "undefined instruction %#010x", LD32(pc));
}
