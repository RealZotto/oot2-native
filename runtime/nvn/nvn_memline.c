// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The memory line (oot2/memline.h has the switches, the format and the sum; tools/check_memline.py reads the file).
//
// Threads and locks. One helper thread gathers and writes; no other thread gathers. g_file guards the file and g_closed. The
// helper gathers before it takes g_file, so the end of a run, which may be raised by a thread that holds a runtime lock the
// helper is waiting for, can always take g_file (trylock) and write a cheap row of its own after a second. memline_mark and
// memline_note (runtime/core/memnote.c) take no runtime lock: a flag, and a mutex held for a table of sixteen entries.
#include "oot2/memline.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gfx.h"
#include "oot2/memory.h"
#include "oot2/nvn_pace.h"
#include "oot2/runtime.h"
#include "plat.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif

#define MiB (1ull << 20)
#define NONE UINT64_MAX
#define TICK_NS 100000000ull
#define END_WAIT_NS 1000000000ull

// ---------- the sum
uint64_t memline_sum(const MemTerms *t, uint64_t priv, int64_t *residual) {
    uint64_t total = 0;
#define X(n) total += t->n == NONE ? 0 : t->n;
    MEMLINE_SUMMED(X)
#undef X
    if (residual) *residual = (int64_t)priv - (int64_t)total;
    return total;
}

// ---------- what the machine can tell (Windows; elsewhere -1)
#ifdef _WIN32
typedef struct {
    ULONG_PTR zero, free_, modified, modified_nowrite, bad, by_priority[8], repurposed[8], modified_pagefile;
} MemoryListInfo;
typedef LONG(WINAPI *NtQuerySystemInformationFn)(ULONG cls, PVOID buf, ULONG len, PULONG ret);

static uint64_t g_guest_lo[RT_MEM_REGIONS], g_guest_hi[RT_MEM_REGIONS];   // the guest regions, from the last full gather

static int in_guest(uint64_t a) {
    for (int i = 0; i < RT_MEM_REGIONS; i++)
        if (g_guest_hi[i] && a >= g_guest_lo[i] && a < g_guest_hi[i]) return 1;
    return 0;
}

// The private committed bytes outside the guest regions, by allocation: write-combined ones (PAGE_WRITECOMBINE, 0x400) of
// exactly 32 MiB, other write-combined ones, and the rest.
static void walk_private(MemTerms *t) {
    uint64_t wc32 = 0, wc32n = 0, wco = 0, wcon = 0, other = 0;
    uint64_t cur_base = 0, cur_bytes = 0;
    int cur_wc = 0;
    MEMORY_BASIC_INFORMATION mbi;
    uint64_t a = 0;
    for (;;) {
        int got = a < 0x7FFFFFFF0000ull && VirtualQuery((void *)(uintptr_t)a, &mbi, sizeof mbi) == sizeof mbi;
        uint64_t base = got ? (uint64_t)(uintptr_t)mbi.AllocationBase : 0;
        int mine = got && mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE && !in_guest(base);
        if (cur_bytes && (!mine || base != cur_base)) {
            if (cur_wc) {
                if (cur_bytes == 32 * MiB) wc32 += cur_bytes, wc32n++;
                else wco += cur_bytes, wcon++;
            } else {
                other += cur_bytes;
            }
            cur_bytes = 0, cur_wc = 0;
        }
        if (!got) break;
        if (mine) {
            cur_base = base;
            cur_bytes += mbi.RegionSize;
            if (mbi.Protect & 0x400) cur_wc = 1;
        }
        a = (uint64_t)(uintptr_t)mbi.BaseAddress + mbi.RegionSize;
    }
    t->wc32_bytes = wc32, t->wc32_count = wc32n, t->wcodd_bytes = wco, t->wcodd_count = wcon, t->host_other_bytes = other;
}

static void machine(MemTerms *t) {
    uint64_t total, avail;
    plat_system_memory(&total, &avail);
    t->ram_available = avail;
    PERFORMANCE_INFORMATION pi = {.cb = sizeof pi};
    if (K32GetPerformanceInfo(&pi, sizeof pi))
        t->sys_committed = (uint64_t)pi.CommitTotal * pi.PageSize, t->sys_commit_limit = (uint64_t)pi.CommitLimit * pi.PageSize;
    SIZE_T lo, hi;
    DWORD fl;
    if (GetProcessWorkingSetSizeEx(GetCurrentProcess(), &lo, &hi, &fl)) t->ws_cap_max = hi;
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    NtQuerySystemInformationFn q =
        nt ? (NtQuerySystemInformationFn)(void (*)(void))GetProcAddress(nt, "NtQuerySystemInformation") : NULL;
    MemoryListInfo ml;
    memset(&ml, 0, sizeof ml);
    if (q && q(80, &ml, sizeof ml, NULL) == 0) {   // SystemMemoryListInformation
        uint64_t standby = 0;
        for (int i = 0; i < 8; i++) standby += ml.by_priority[i];
        t->ram_standby = standby * 4096, t->ram_modified = (uint64_t)(ml.modified + ml.modified_nowrite) * 4096;
    }
    // the five largest other processes by private commit
    uint64_t priv[5] = {0}, ws[5] = {0};
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        PROCESSENTRY32W pe = {.dwSize = sizeof pe};
        for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
            if (pe.th32ProcessID == GetCurrentProcessId() || !pe.th32ProcessID) continue;
            HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
            if (!h) continue;
            PROCESS_MEMORY_COUNTERS_EX c = {.cb = sizeof c};
            if (K32GetProcessMemoryInfo(h, (PROCESS_MEMORY_COUNTERS *)&c, sizeof c)) {
                uint64_t p = c.PrivateUsage, w = c.WorkingSetSize;
                for (int i = 0; i < 5; i++)
                    if (p > priv[i]) {
                        for (int j = 4; j > i; j--) priv[j] = priv[j - 1], ws[j] = ws[j - 1];
                        priv[i] = p, ws[i] = w;
                        break;
                    }
            }
            CloseHandle(h);
        }
        CloseHandle(snap);
    }
    t->top1_private = priv[0], t->top1_ws = ws[0], t->top2_private = priv[1], t->top2_ws = ws[1];
    t->top3_private = priv[2], t->top3_ws = ws[2], t->top4_private = priv[3], t->top4_ws = ws[3];
    t->top5_private = priv[4], t->top5_ws = ws[4];
}
#else
static void walk_private(MemTerms *t) {
    t->wc32_bytes = t->wc32_count = t->wcodd_bytes = t->wcodd_count = t->host_other_bytes = NONE;
}
static void machine(MemTerms *t) {
    uint64_t total, avail;
    plat_system_memory(&total, &avail);
    t->ram_available = avail;
    t->ram_standby = t->ram_modified = t->sys_committed = t->sys_commit_limit = t->ws_cap_max = NONE;
    t->top1_private = t->top1_ws = t->top2_private = t->top2_ws = t->top3_private = t->top3_ws = NONE;
    t->top4_private = t->top4_ws = t->top5_private = t->top5_ws = NONE;
}
#endif

// ---------- the gather
static MemlineGpuFill g_gpu, g_hook;
void memline_set_gpu(MemlineGpuFill fn) { __atomic_store_n(&g_gpu, fn, __ATOMIC_RELEASE); }
void memline_set_test_hook(MemlineGpuFill fn) { __atomic_store_n(&g_hook, fn, __ATOMIC_RELEASE); }

void memline_gather(MemTerms *t, int cheap) {
    memset(t, cheap ? 0xFF : 0, sizeof *t);   // -1 for what a cheap row cannot tell
    PlatProcMem p;
    plat_process_memory(&p);
    t->private = p.private_bytes, t->working_set = p.working_set, t->peak_working_set = p.peak_working_set;
    t->seq = nvn_present_count();
    t->nvn_buffers = nvn_buffers_live(), t->nvn_textures = nvn_textures_live();
    MemlineSite sites[MEMLINE_SITES];
    int ns = memline_sites(sites, MEMLINE_SITES);
    t->big_blocks = t->big_bytes = 0;
    for (int i = 0; i < ns; i++) t->big_blocks += sites[i].live_blocks, t->big_bytes += sites[i].live_bytes;
    if (!cheap) {
        if (rt_mem_ready()) {
            RtMemStats m;
            rt_mem_measure(&m);
            t->private = m.process_private, t->working_set = m.process_working_set;
            t->peak_working_set = m.process_peak_working_set;
            const RtMemRegion *r = m.r;
            t->stack_committed = r[RT_MEM_STACK].committed, t->stack_resident = r[RT_MEM_STACK].resident;
            t->heap_committed = r[RT_MEM_HEAP].committed, t->heap_resident = r[RT_MEM_HEAP].resident;
            t->alias_committed = r[RT_MEM_ALIAS].committed, t->alias_resident = r[RT_MEM_ALIAS].resident;
            t->modules_committed = r[RT_MEM_MODULES].committed, t->modules_resident = r[RT_MEM_MODULES].resident;
            t->kernel_committed = r[RT_MEM_KERNEL].committed, t->kernel_resident = r[RT_MEM_KERNEL].resident;
#ifdef _WIN32
            for (int i = 0; i < RT_MEM_REGIONS; i++) g_guest_lo[i] = r[i].base, g_guest_hi[i] = r[i].base + r[i].size;
#endif
        } else {
            t->stack_committed = t->stack_resident = t->heap_committed = t->heap_resident = NONE;
            t->alias_committed = t->alias_resident = t->modules_committed = t->modules_resident = NONE;
            t->kernel_committed = t->kernel_resident = NONE;
        }
        NvnPoolStats ps;
        nvn_pool_stats(&ps);
        t->nvn_pools = ps.all.live, t->nvn_pool_bytes = ps.all.bytes;
        t->cmd_host_bytes = nvn_cmd_host_bytes(), t->cmd_copies = nvn_cmd_copies();
        MemlineGpuFill gpu = __atomic_load_n(&g_gpu, __ATOMIC_ACQUIRE);
        if (gpu) gpu(t);
    }
    walk_private(t);
    machine(t);
    MemlineGpuFill hook = __atomic_load_n(&g_hook, __ATOMIC_ACQUIRE);
    if (hook) hook(t);
}

// ---------- the file and the helper thread
static const char *const k_kind[MEMLINE_KINDS] = {"minute", "title", "scene", "end"};
static FILE *g_f;
static PlatMutex g_file = PLAT_MUTEX_INIT;
static int g_closed;   // g_file
static uint64_t g_rows, g_period_ns, g_origin, g_scene_present;
static int g_started, g_ending, g_exited, g_scene_done;
static int g_pending[MEMLINE_KINDS];
static uint64_t g_pending_seq[MEMLINE_KINDS];

static void put_u(FILE *f, uint64_t v) {
    if (v == NONE) fputs("\t-1", f);
    else fprintf(f, "\t%llu", (unsigned long long)v);
}

// g_file held
static void write_row(int kind, const MemTerms *t) {
    if (g_closed) return;
    fprintf(g_f, "row\t%s", k_kind[kind]);
#define X(n) put_u(g_f, t->n);
    MEMLINE_TERMS(X)
#undef X
    fputc('\n', g_f);
    MemlineSite sites[MEMLINE_SITES];
    int ns = memline_sites(sites, MEMLINE_SITES);
    for (int i = 0; i < ns; i++)
        fprintf(g_f, "alloc\t%llu\t%s\t%llu\t%llu\t%llu\n", (unsigned long long)t->seq, sites[i].site,
                (unsigned long long)sites[i].live_blocks, (unsigned long long)sites[i].live_bytes,
                (unsigned long long)sites[i].total_blocks);
    fflush(g_f);
    g_rows++;
}

static void close_file(void) {   // g_file held
    if (g_closed) return;
    fprintf(g_f, "end\t%llu\n", (unsigned long long)g_rows);
    if (ferror(g_f) | fclose(g_f)) rt_log("memline: the file could not be finished");
    g_f = NULL, g_closed = 1;
}

static void emit(int kind, uint64_t seq, int stamp_seq) {
    MemTerms t;
    memline_gather(&t, 0);
    if (stamp_seq) t.seq = seq;
    t.t_ns = plat_now_ns() - g_origin;
    plat_mutex_lock(&g_file);
    write_row(kind, &t);
    plat_mutex_unlock(&g_file);
}

static void helper_main(void *arg) {
    (void)arg;
    uint64_t next = g_origin + g_period_ns;
    while (!__atomic_load_n(&g_ending, __ATOMIC_ACQUIRE)) {
        for (int k = MEMLINE_TITLE; k <= MEMLINE_SCENE; k++)
            if (__atomic_exchange_n(&g_pending[k], 0, __ATOMIC_ACQ_REL))
                emit(k, __atomic_load_n(&g_pending_seq[k], __ATOMIC_ACQUIRE), 1);
        uint64_t now = plat_now_ns();
        if (now >= next) {
            next += g_period_ns;
            if (next <= now) next = now + g_period_ns;
            emit(MEMLINE_MINUTE, 0, 0);
            continue;
        }
        plat_sleep_ns(next - now < TICK_NS ? next - now : TICK_NS);
    }
    MemTerms t;
    plat_mutex_lock(&g_file);
    int done = g_closed;
    plat_mutex_unlock(&g_file);
    if (!done) {   // the end of the run has not written its own: the full row, then the file is closed
        memline_gather(&t, 0);
        t.t_ns = plat_now_ns() - g_origin;
        plat_mutex_lock(&g_file);
        write_row(MEMLINE_END, &t);
        close_file();
        plat_mutex_unlock(&g_file);
    }
    __atomic_store_n(&g_exited, 1, __ATOMIC_RELEASE);
}

void memline_start(const char *folder, const char *stamp, unsigned pid) {
    if (__atomic_load_n(&g_started, __ATOMIC_ACQUIRE)) rt_fatal("memline_start: called twice");
    const char *e = getenv("OOT2_MEMLINE_SECONDS");
    double secs = 60;
    if (e && *e) {
        char *end;
        secs = strtod(e, &end);
        if (*end || !(secs >= 0.05) || secs > 1e6) rt_fatal("OOT2_MEMLINE_SECONDS=%s: expected seconds, 0.05 to 1000000", e);
    }
    g_period_ns = (uint64_t)(secs * 1e9);
    e = getenv("OOT2_MEMLINE_SCENE");
    if (e && *e) {
        char *end;
        unsigned long long v = strtoull(e, &end, 10);
        if (*end || !v) rt_fatal("OOT2_MEMLINE_SCENE=%s: expected a present count of at least 1", e);
        g_scene_present = v;
    }
    char path[4300];
    snprintf(path, sizeof path, "%s/memory-%s-%u.tsv", folder, stamp, pid);
    g_f = fopen(path, "wb");
    if (!g_f) rt_fatal("memline: cannot write %s", path);
    fputs("oot2-memline 1\ncolumns", g_f);
#define X(n) fputs("\t" #n, g_f);
    MEMLINE_TERMS(X)
#undef X
    fputs("\nsum", g_f);
#define X(n) fputs("\t" #n, g_f);
    MEMLINE_SUMMED(X)
#undef X
    fputc('\n', g_f);
    memline_gather(&(MemTerms){0}, 1);   // opens nothing; warms the first call of the system queries off the timed path
    g_origin = plat_now_ns();
    __atomic_store_n(&g_started, 1, __ATOMIC_RELEASE);
    if (!plat_thread_start(helper_main, NULL, 1ull << 20)) rt_fatal("memline: cannot start the helper thread");
    rt_log("memline: writing %s (a row every %.3f s, and at the title mark, the scene entry and the end)", path, secs);
}

int memline_on(void) { return __atomic_load_n(&g_started, __ATOMIC_ACQUIRE) && !__atomic_load_n(&g_ending, __ATOMIC_ACQUIRE); }

void memline_mark(int kind, uint64_t seq) {
    if (kind < MEMLINE_TITLE || kind > MEMLINE_SCENE || !__atomic_load_n(&g_started, __ATOMIC_ACQUIRE)) return;
    if (kind == MEMLINE_SCENE && seq) {   // a present count from the capture: once, at OOT2_MEMLINE_SCENE; 0 means now
        if (!g_scene_present || seq < g_scene_present || __atomic_exchange_n(&g_scene_done, 1, __ATOMIC_ACQ_REL)) return;
    }
    __atomic_store_n(&g_pending_seq[kind], seq, __ATOMIC_RELEASE);
    __atomic_store_n(&g_pending[kind], 1, __ATOMIC_RELEASE);
}

void memline_end(void) {
    if (!__atomic_load_n(&g_started, __ATOMIC_ACQUIRE) || __atomic_exchange_n(&g_ending, 1, __ATOMIC_ACQ_REL)) return;
    uint64_t until = plat_now_ns() + END_WAIT_NS;
    while (!__atomic_load_n(&g_exited, __ATOMIC_ACQUIRE) && plat_now_ns() < until) plat_sleep_ns(2000000);
    if (__atomic_load_n(&g_exited, __ATOMIC_ACQUIRE)) return;
    // The helper is stuck in a gather (a lock the failing thread holds): the end writes its own, cheap row if it can have the file
    if (!plat_mutex_trylock(&g_file)) return;
    if (!g_closed) {
        MemTerms t;
        memline_gather(&t, 1);
        t.t_ns = plat_now_ns() - g_origin;
        write_row(MEMLINE_END, &t);
        close_file();
    }
    plat_mutex_unlock(&g_file);
}
