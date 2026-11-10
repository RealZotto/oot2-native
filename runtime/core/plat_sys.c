// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the M1 capture reads of the machine (plat.h, "the machine"): physical memory, the page file's writes, the
// processor count and the process id. Apart from plat.c because nothing else in the runtime needs them.
//
// The page file's writes on Windows are PDH's \Memory\Pages Output/sec, named by its English path so that the name holds
// in any display language. Its raw value is a count since the system started (a PERF_COUNTER_COUNTER: PDH makes the
// rate from two raw values), so the raw value is read and a caller takes differences. It counts the pages every process
// of the machine wrote, not this process's. pdh.dll is opened when the counter is first asked for, so a program that
// never asks does not load it, and the build links no extra library.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plat.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void plat_system_memory(uint64_t *total, uint64_t *available) {
    MEMORYSTATUSEX m = {.dwLength = sizeof m};
    *total = *available = 0;
    if (!GlobalMemoryStatusEx(&m)) return;
    *total = m.ullTotalPhys;
    *available = m.ullAvailPhys;
}

// pdh.h's types, written out so that only the four entry points are needed.
typedef struct {
    DWORD status;
    FILETIME stamp;
    LONGLONG first, second;
    DWORD multi;
} RawCounter;
typedef LONG(WINAPI *PdhOpenQueryWFn)(LPCWSTR source, DWORD_PTR user, HANDLE *query);
typedef LONG(WINAPI *PdhAddEnglishCounterWFn)(HANDLE query, LPCWSTR path, DWORD_PTR user, HANDLE *counter);
typedef LONG(WINAPI *PdhCollectQueryDataFn)(HANDLE query);
typedef LONG(WINAPI *PdhGetRawCounterValueFn)(HANDLE counter, LPDWORD type, RawCounter *value);

static PlatMutex g_pdh_lock = PLAT_MUTEX_INIT;
static int g_pdh_state;   // 0 not tried, 1 open, -1 unavailable (g_pdh_lock)
static HANDLE g_query, g_counter;
static PdhCollectQueryDataFn g_collect;
static PdhGetRawCounterValueFn g_raw;

static int pdh_open(void) {
    HMODULE m = LoadLibraryW(L"pdh.dll");
    if (!m) return 0;
    PdhOpenQueryWFn open = (PdhOpenQueryWFn)(void (*)(void))GetProcAddress(m, "PdhOpenQueryW");
    PdhAddEnglishCounterWFn add = (PdhAddEnglishCounterWFn)(void (*)(void))GetProcAddress(m, "PdhAddEnglishCounterW");
    g_collect = (PdhCollectQueryDataFn)(void (*)(void))GetProcAddress(m, "PdhCollectQueryData");
    g_raw = (PdhGetRawCounterValueFn)(void (*)(void))GetProcAddress(m, "PdhGetRawCounterValue");
    if (!open || !add || !g_collect || !g_raw) return 0;
    if (open(NULL, 0, &g_query) != 0) return 0;
    return add(g_query, L"\\Memory\\Pages Output/sec", 0, &g_counter) == 0;
}

uint64_t plat_pages_output(void) {
    uint64_t v = UINT64_MAX;
    plat_mutex_lock(&g_pdh_lock);
    if (!g_pdh_state) g_pdh_state = pdh_open() ? 1 : -1;
    RawCounter r;
    DWORD type;
    if (g_pdh_state == 1 && g_collect(g_query) == 0 && g_raw(g_counter, &type, &r) == 0 && r.first >= 0)
        v = (uint64_t)r.first;
    plat_mutex_unlock(&g_pdh_lock);
    return v;
}

const char *plat_pages_output_name(void) {
    return "\\Memory\\Pages Output/sec, raw count through PDH (pages written to the page file by every process of "
           "the machine)";
}

int plat_cpu_count(void) {
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n ? (int)n : 1;
}

unsigned plat_pid(void) { return (unsigned)GetCurrentProcessId(); }

int plat_pid_alive(unsigned pid) {
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (!h) return GetLastError() != ERROR_INVALID_PARAMETER;   // no such id; any other refusal (access) means it exists
    DWORD code = 0;
    int alive = !GetExitCodeProcess(h, &code) || code == STILL_ACTIVE;   // a process that ended can still be held open by others
    CloseHandle(h);
    return alive;
}

#else
#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

// The number after `key` in a /proc file of "name value" lines, times scale; 0 when it is missing.
static uint64_t proc_value(const char *path, const char *key, uint64_t scale, int *found) {
    FILE *f = fopen(path, "r");
    char line[256];
    size_t n = strlen(key);
    uint64_t v = 0;
    *found = 0;
    if (!f) return 0;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, key, n) && (line[n] == ' ' || line[n] == ':' || line[n] == '\t')) {
            v = strtoull(line + n + 1, NULL, 10) * scale;
            *found = 1;
            break;
        }
    fclose(f);
    return v;
}

void plat_system_memory(uint64_t *total, uint64_t *available) {
    int found;
    *total = proc_value("/proc/meminfo", "MemTotal", 1024, &found);
    *available = proc_value("/proc/meminfo", "MemAvailable", 1024, &found);
    if (!*total) {
        long pages = sysconf(_SC_PHYS_PAGES), avail = sysconf(_SC_AVPHYS_PAGES), size = sysconf(_SC_PAGESIZE);
        *total = pages > 0 && size > 0 ? (uint64_t)pages * (uint64_t)size : 0;
        *available = avail > 0 && size > 0 ? (uint64_t)avail * (uint64_t)size : 0;
    }
}

uint64_t plat_pages_output(void) {
    int found;
    uint64_t v = proc_value("/proc/vmstat", "pswpout", 1, &found);
    return found ? v : UINT64_MAX;
}

const char *plat_pages_output_name(void) {
    return "pswpout of /proc/vmstat (pages swapped out by every process of the machine)";
}

int plat_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

unsigned plat_pid(void) { return (unsigned)getpid(); }

int plat_pid_alive(unsigned pid) { return kill((pid_t)pid, 0) == 0 || errno != ESRCH; }
#endif
