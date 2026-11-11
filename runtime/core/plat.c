// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
#ifdef _WIN32
#define _CRT_RAND_S   // rand_s in stdlib.h
#else
#define _GNU_SOURCE   // REG_ERR and REG_RIP in the fault handler's context
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plat.h"

static int (*g_fault_hook)(uint64_t addr, int write, uint64_t host_pc);

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

_Static_assert(sizeof(PlatLock) == sizeof(SRWLOCK), "PlatLock holds an SRWLOCK");

void *plat_alloc_zeroed(size_t bytes) {
    return VirtualAlloc(NULL, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

int plat_protect_read_only(void *p, size_t bytes) {
    DWORD old;
    return VirtualProtect(p, bytes, PAGE_READONLY, &old) != 0;
}

uint64_t plat_now_ns(void) {
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&now);
    // Split to keep the multiplication inside 64 bits for any uptime.
    uint64_t f = (uint64_t)freq.QuadPart, t = (uint64_t)now.QuadPart;
    return t / f * 1000000000ull + t % f * 1000000000ull / f;
}

void plat_yield(void) { SwitchToThread(); }

void plat_write_stderr(const char *p, size_t n) {
    DWORD done;
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    while (n && WriteFile(h, p, (DWORD)n, &done, NULL) && done) { p += done; n -= done; }
}

static DWORD win_prot(int prot) { return prot == PLAT_RW ? PAGE_READWRITE : prot == PLAT_R ? PAGE_READONLY : PAGE_NOACCESS; }

// The end of the piece that holds a (plat.h, PLAT_PIECE), or end when that comes first.
static uint64_t piece_end(uint64_t a, uint64_t end) {
    uint64_t b = (a | (PLAT_PIECE - 1)) + 1;
    return b > a && b < end ? b : end;
}

int plat_reserve_at(uint64_t addr, uint64_t size, int watch) {
    const uint64_t end = addr + size;
    for (uint64_t a = addr; a < end;) {
        uint64_t b = watch ? piece_end(a, end) : end;
        void *want = (void *)(uintptr_t)a;
        void *p = VirtualAlloc(want, b - a, MEM_RESERVE | (watch ? MEM_WRITE_WATCH : 0), PAGE_NOACCESS);
        if (p != want) {
            if (p) VirtualFree(p, 0, MEM_RELEASE);
            if (a > addr) plat_release(addr, a - addr);
            return 0;
        }
        a = b;
    }
    return 1;
}

// A reservation made in one piece or in several: every allocation that starts in [addr, addr+size) is released.
void plat_release(uint64_t addr, uint64_t size) {
    for (uint64_t a = addr; a < addr + size;) {
        MEMORY_BASIC_INFORMATION m;
        if (VirtualQuery((void *)(uintptr_t)a, &m, sizeof m) != sizeof m || (uint64_t)(uintptr_t)m.AllocationBase != a)
            return;
        uint64_t next = a + m.RegionSize;   // past the allocation's last region
        while (next < addr + size && VirtualQuery((void *)(uintptr_t)next, &m, sizeof m) == sizeof m &&
               (uint64_t)(uintptr_t)m.AllocationBase == a)
            next += m.RegionSize;
        VirtualFree((void *)(uintptr_t)a, 0, MEM_RELEASE);
        a = next;
    }
}

int plat_commit(uint64_t addr, uint64_t size, int prot) {
    for (uint64_t a = addr, b; a < addr + size; a = b) {
        b = piece_end(a, addr + size);
        if (!VirtualAlloc((void *)(uintptr_t)a, b - a, MEM_COMMIT, win_prot(prot))) return 0;
    }
    return 1;
}

int plat_decommit(uint64_t addr, uint64_t size) {
    for (uint64_t a = addr, b; a < addr + size; a = b) {
        b = piece_end(a, addr + size);
        if (!VirtualFree((void *)(uintptr_t)a, b - a, MEM_DECOMMIT)) return 0;
    }
    return 1;
}

int plat_protect(uint64_t addr, uint64_t size, int prot) {
    for (uint64_t a = addr, b; a < addr + size; a = b) {
        DWORD old;
        b = piece_end(a, addr + size);
        if (!VirtualProtect((void *)(uintptr_t)a, b - a, win_prot(prot), &old)) return 0;
    }
    return 1;
}

int plat_is_free(uint64_t addr) {
    MEMORY_BASIC_INFORMATION mbi;
    return VirtualQuery((void *)(uintptr_t)addr, &mbi, sizeof mbi) == sizeof mbi && mbi.State == MEM_FREE;
}

// Piece by piece, in address order, until the buffer is full: a query with reset resets only the pages it returns
// (the write-watch study), so pieces after a full buffer keep their marks for the next call.
uint64_t plat_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    uint64_t total = 0;
    for (uint64_t a = addr, b; a < addr + size && total < max; a = b) {
        b = piece_end(a, addr + size);
        ULONG_PTR count = (ULONG_PTR)(max - total);
        ULONG granularity;
        if (GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, (void *)(uintptr_t)a, b - a, (PVOID *)(pages + total),
                          &count, &granularity))
            return UINT64_MAX;
        total += count;
    }
    return total;
}

uint64_t plat_resident(uint64_t addr, uint64_t size) {
    enum { BATCH = 1024 };
    PSAPI_WORKING_SET_EX_INFORMATION q[BATCH];
    uint64_t pages = size >> 12, done = 0, resident = 0;
    while (done < pages) {
        DWORD n = pages - done < BATCH ? (DWORD)(pages - done) : BATCH;
        for (DWORD i = 0; i < n; i++) q[i].VirtualAddress = (void *)(uintptr_t)(addr + ((done + i) << 12));
        if (!K32QueryWorkingSetEx(GetCurrentProcess(), q, n * (DWORD)sizeof q[0])) return UINT64_MAX;
        for (DWORD i = 0; i < n; i++) resident += q[i].VirtualAttributes.Valid;
        done += n;
    }
    return resident << 12;
}

void plat_process_memory(PlatProcMem *out) {
    PROCESS_MEMORY_COUNTERS_EX c = {.cb = sizeof c};
    *out = (PlatProcMem){0};
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof c)) return;
    *out = (PlatProcMem){c.PrivateUsage, c.WorkingSetSize, c.PeakWorkingSetSize};
}

// A first touch of guest memory is an access violation that the vectored handler below settles. Windows delivers it by
// building the exception frame on the faulting thread's own stack, under its stack pointer: the CONTEXT with the
// processor's extended state, about 1.8 KB on an AVX2 processor and ten times that where the vector state is larger
// (AVX-512, AMX), plus the record. The kernel grows a stack only a few pages past its committed end (measured on
// Windows 10: about 12 KB), and a thread starts with a few KB committed, so on a processor with a large state a fault
// taken early in a thread's life (the guest's first store to its stack or heap) could not be dispatched at all: the
// process ended with the violation itself, exit code 0xC0000005, and the handler never ran. Every thread therefore
// commits stack headroom before it runs anything; the pages stay committed for its life.
#define STACK_HEADROOM (256ull << 10)

static __attribute__((noinline)) void stack_commit(uint64_t bytes) {
    if (bytes < 4096) return;
    volatile char *p = __builtin_alloca((size_t)bytes);
    for (uint64_t off = 0; off < bytes; off += 4096) p[bytes - 1 - off] = 0;   // from the top, a page at a time
    p[0] = 0;
}

// The headroom of a thread that reserved reserve bytes of stack: a quarter of the reserve at most.
static uint64_t stack_headroom(uint64_t reserve) { return reserve / 4 < STACK_HEADROOM ? (reserve / 4) & ~4095ull : STACK_HEADROOM; }

// First in the chain, so a guest touch is settled before any crash reporter sees it.
static LONG CALLBACK fault_filter(EXCEPTION_POINTERS *e) {
    const EXCEPTION_RECORD *r = e->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || !g_fault_hook)
        return EXCEPTION_CONTINUE_SEARCH;
    int write = r->ExceptionInformation[0] == 1;
    return g_fault_hook(r->ExceptionInformation[1], write, (uint64_t)(uintptr_t)r->ExceptionAddress)
               ? EXCEPTION_CONTINUE_EXECUTION
               : EXCEPTION_CONTINUE_SEARCH;
}

void plat_fault_hook(int (*hook)(uint64_t addr, int write, uint64_t host_pc)) {
    if (!g_fault_hook) {
        // the calling thread is the process's first: its stack reserves 1 MiB by default
        void *sp = __builtin_frame_address(0);
        MEMORY_BASIC_INFORMATION m;
        if (VirtualQuery(sp, &m, sizeof m) == sizeof m && (uintptr_t)sp > (uintptr_t)m.AllocationBase)
            stack_commit(stack_headroom((uint64_t)((uintptr_t)sp - (uintptr_t)m.AllocationBase)));
        AddVectoredExceptionHandler(1, fault_filter);
    }
    g_fault_hook = hook;
}

const void *plat_map_file(const char *path, uint64_t *size) {
    wchar_t wpath[1024];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024)) return NULL;
    HANDLE f = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER len;
    const void *p = NULL;
    if (GetFileSizeEx(f, &len) && len.QuadPart > 0) {
        HANDLE m = CreateFileMappingW(f, NULL, PAGE_READONLY, 0, 0, NULL);
        if (m) {
            p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
            CloseHandle(m);   // the view keeps the mapping alive
        }
    }
    CloseHandle(f);
    if (p) *size = (uint64_t)len.QuadPart;
    return p;
}

void plat_unmap_file(const void *p, uint64_t size) {
    (void)size;
    if (p) UnmapViewOfFile(p);
}

void plat_lock(PlatLock *l) { AcquireSRWLockExclusive((PSRWLOCK)l); }
void plat_unlock(PlatLock *l) { ReleaseSRWLockExclusive((PSRWLOCK)l); }
void plat_lock_shared(PlatLock *l) { AcquireSRWLockShared((PSRWLOCK)l); }
void plat_unlock_shared(PlatLock *l) { ReleaseSRWLockShared((PSRWLOCK)l); }
int plat_lock_shared_try(PlatLock *l) { return TryAcquireSRWLockShared((PSRWLOCK)l) != 0; }

_Static_assert(sizeof(PlatMutex) == sizeof(SRWLOCK), "PlatMutex holds an SRWLOCK");
_Static_assert(sizeof(PlatCond) == sizeof(CONDITION_VARIABLE), "PlatCond holds a CONDITION_VARIABLE");

void plat_mutex_lock(PlatMutex *m) { AcquireSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_unlock(PlatMutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)m); }
int plat_mutex_trylock(PlatMutex *m) { return TryAcquireSRWLockExclusive((PSRWLOCK)m) != 0; }
void plat_cond_wait(PlatCond *c, PlatMutex *m) { SleepConditionVariableSRW((PCONDITION_VARIABLE)c, (PSRWLOCK)m, INFINITE, 0); }

// The wait counts milliseconds: rounded up, so it never ends early on its own (the caller checks its deadline).
void plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns) {
    uint64_t ms = (ns + 999999) / 1000000;
    SleepConditionVariableSRW((PCONDITION_VARIABLE)c, (PSRWLOCK)m, ms >= INFINITE ? INFINITE - 1 : (DWORD)ms, 0);
}

void plat_cond_signal(PlatCond *c) { WakeConditionVariable((PCONDITION_VARIABLE)c); }
void plat_cond_broadcast(PlatCond *c) { WakeAllConditionVariable((PCONDITION_VARIABLE)c); }

// ---- threads. CONTEXT comes first: it wants 16-byte alignment, which calloc gives on x64.
struct PlatThread {
    CONTEXT ctx;   // what plat_intr_stop read, written back by plat_intr_resume
    HANDLE h;
    void (*fn)(void *arg);
    void *arg;
    uint64_t headroom;   // stack committed before fn runs
};

static DWORD WINAPI thread_start(void *p) {
    PlatThread *t = p;
    stack_commit(t->headroom);
    t->fn(t->arg);   // t is not touched after fn returns: plat_thread_free may already have run
    return 0;
}

PlatThread *plat_thread_start(void (*fn)(void *arg), void *arg, uint64_t stack_reserve) {
    PlatThread *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->fn = fn;
    t->arg = arg;
    t->headroom = stack_headroom(stack_reserve);
    t->h = CreateThread(NULL, (SIZE_T)stack_reserve, thread_start, t, STACK_SIZE_PARAM_IS_A_RESERVATION, NULL);
    if (!t->h) {
        free(t);
        return NULL;
    }
    return t;
}

void plat_thread_free(PlatThread *t) {
    if (!t) return;
    CloseHandle(t->h);
    free(t);
}

// Thread time counts processor cycles (QueryThreadCycleTime: exact, unlike GetThreadTimes, which advances by whole
// clock interrupts of 15.6 ms). Cycles become nanoseconds at the time-stamp counter's rate, measured against the
// performance counter over at least 20 ms since start-up; on an invariant TSC that is the rate the cycles count at.
static uint64_t g_tsc0, g_tsc_ns0;
static uint64_t g_ns_per_cycle_bits;   // a double, 0 until measured; racing first callers store about the same value

__attribute__((constructor)) static void tsc_start(void) {
    g_tsc_ns0 = plat_now_ns();
    g_tsc0 = __builtin_ia32_rdtsc();
}

static double ns_per_cycle(void) {
    uint64_t bits = __atomic_load_n(&g_ns_per_cycle_bits, __ATOMIC_RELAXED), ns;
    double r;
    if (bits) {
        memcpy(&r, &bits, sizeof r);
        return r;
    }
    while ((ns = plat_now_ns() - g_tsc_ns0) < 20000000) Sleep(1);
    r = (double)ns / (double)(__builtin_ia32_rdtsc() - g_tsc0);
    memcpy(&bits, &r, sizeof bits);
    __atomic_store_n(&g_ns_per_cycle_bits, bits, __ATOMIC_RELAXED);
    return r;
}

uint64_t plat_thread_cpu_ns(PlatThread *t) {
    ULONG64 cycles = 0;
    if (!QueryThreadCycleTime(t ? t->h : GetCurrentThread(), &cycles)) return 0;
    return (uint64_t)((double)cycles * ns_per_cycle());
}

int plat_thread_raise_priority(void) { return SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) != 0; }

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x2
#endif

void plat_sleep_ns(uint64_t ns) {
    static _Thread_local HANDLE timer;
    if (!timer) timer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    uint64_t end = plat_now_ns() + ns, now;
    while ((now = plat_now_ns()) < end) {
        LARGE_INTEGER due = {.QuadPart = -(LONGLONG)((end - now + 99) / 100)};   // relative, in 100 ns
        if (timer && SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE)) WaitForSingleObject(timer, INFINITE);
        else Sleep((DWORD)((end - now + 999999) / 1000000));
    }
}

int plat_random(void *p, size_t n) {
    unsigned char *b = p;
    for (size_t i = 0; i < n; i += 4) {
        unsigned v;
        if (rand_s(&v)) return 0;
        memcpy(b + i, &v, n - i < 4 ? n - i : 4);
    }
    return 1;
}

// ---- the system timer resolution. ntdll's NtSetTimerResolution takes the units of 100 ns that timeBeginPeriod cannot
// ask for (it takes whole milliseconds). It is undocumented, so it is resolved at run time (no import library, as
// plat_sys.c does for pdh.dll), and a missing or failing call is the caller's to log. Measured in the vsync clock study: a request
// of 5000 is granted 5000, and a finer one is clamped to the finest the hardware allows.
typedef LONG(NTAPI *NtSetTimerResolutionFn)(ULONG want, BOOLEAN on, PULONG now);
typedef LONG(NTAPI *NtQueryTimerResolutionFn)(PULONG coarsest, PULONG finest, PULONG now);
static NtSetTimerResolutionFn g_nt_set;
static NtQueryTimerResolutionFn g_nt_query;

static void nt_timer_resolve(void) {   // idempotent: racing callers store the same pointers
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    if (!ntdll) return;
    g_nt_set = (NtSetTimerResolutionFn)(void *)GetProcAddress(ntdll, "NtSetTimerResolution");
    g_nt_query = (NtQueryTimerResolutionFn)(void *)GetProcAddress(ntdll, "NtQueryTimerResolution");
}

static int sys_timer_set(uint64_t want_ns, int on, uint64_t *now_ns) {
    if (!g_nt_set) nt_timer_resolve();
    if (!g_nt_set) return -1;
    ULONG now = 0;
    LONG status = g_nt_set((ULONG)(want_ns / 100), on ? TRUE : FALSE, &now);
    if (now_ns) *now_ns = (uint64_t)now * 100;
    return status < 0 ? (int)status : 0;
}

static uint64_t sys_timer_query(int finest) {
    if (!g_nt_query) nt_timer_resolve();
    ULONG coarsest = 0, fine = 0, now = 0;
    if (!g_nt_query || g_nt_query(&coarsest, &fine, &now) < 0) return 0;
    return (uint64_t)(finest ? fine : now) * 100;
}

static uint64_t sys_timer_now(void) { return sys_timer_query(0); }
static uint64_t sys_timer_finest(void) { return sys_timer_query(1); }

// The console control events that end the process by default (Ctrl+C, Ctrl+Break, close, log-off, shutdown). The default
// handler runs after this one, because it returns FALSE, so the exit code stays the system's. The stop hooks are not run
// from here: this is a thread the system made while guest threads run.
static BOOL WINAPI timer_ctrl_event(DWORD type) {
    (void)type;
    plat_timer_release();
    return FALSE;
}

static void timer_exit_hooks(void) { SetConsoleCtrlHandler(timer_ctrl_event, TRUE); }

void plat_exit(int code) {
    plat_timer_release();
    fflush(NULL);
    _Exit(code);
}

// ---- interruption: SuspendThread, Get/SetThreadContext, ResumeThread. GetThreadContext waits for the suspension to
// take effect. A thread stopped inside a system call reports the user-mode address the call returns to. Adapted from
// DKB Recomp runtime/platform/win32.c at 9d8fa02 (GPL-3.0-or-later), "thread interruption".
int plat_intr_init(void) { return 1; }

int plat_intr_stop(PlatThread *t, uint64_t *rip, uint64_t *rsp) {
    if (SuspendThread(t->h) == (DWORD)-1) return 0;
    memset(&t->ctx, 0, sizeof t->ctx);
    t->ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;   // written back whole, so every GPR is restored as read
    if (!GetThreadContext(t->h, &t->ctx)) {
        ResumeThread(t->h);
        return 0;
    }
    *rip = t->ctx.Rip;
    *rsp = t->ctx.Rsp;
    return 1;
}

int plat_intr_resume(PlatThread *t, uint64_t to) {
    int ok = 1;
    if (to) {
        t->ctx.Rip = to;
        ok = SetThreadContext(t->h, &t->ctx) != 0;
    }
    ResumeThread(t->h);
    return ok;
}

// ---- fibres: Windows fibres with FIBER_FLAG_FLOAT_SWITCH (the MXCSR and the x87 control word switch with them).
// Adapted from TotK Recomp runtime/platform/win32.c at 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), as DKB
// Recomp carries it at 9d8fa02. Changed here: a thread's own context is allocated, not thread-local, and the switch
// names its source (Windows knows it; POSIX below needs it).
struct PlatFibre {
    void *h;
    void (*fn)(void *arg);
    void *arg;
    uint64_t headroom;   // stack committed before fn runs, as for a thread
};

PlatFibre *plat_fibre_self(void) {
    PlatFibre *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    f->h = ConvertThreadToFiberEx(NULL, FIBER_FLAG_FLOAT_SWITCH);
    if (!f->h) {
        free(f);
        return NULL;
    }
    return f;
}

void plat_fibre_unself(PlatFibre *own) {
    ConvertFiberToThread();
    free(own);
}

static void WINAPI fibre_start(void *p) {
    PlatFibre *f = p;
    stack_commit(f->headroom);   // a guest fibre takes its first violations early too (see stack_commit)
    f->fn(f->arg);
}

PlatFibre *plat_fibre_new(uint64_t stack_reserve, void (*fn)(void *arg), void *arg) {
    PlatFibre *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    f->fn = fn;
    f->arg = arg;
    f->headroom = stack_headroom(stack_reserve);
    f->h = CreateFiberEx(0x10000, (SIZE_T)stack_reserve, FIBER_FLAG_FLOAT_SWITCH, fibre_start, f);
    if (!f->h) {
        free(f);
        return NULL;
    }
    return f;
}

void plat_fibre_switch(PlatFibre *from, PlatFibre *to) {
    (void)from;
    SwitchToFiber(to->h);
}

void plat_fibre_free(PlatFibre *f) {
    if (!f) return;
    DeleteFiber(f->h);
    free(f);
}

#else
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

void *plat_alloc_zeroed(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

int plat_protect_read_only(void *p, size_t bytes) { return mprotect(p, bytes, PROT_READ) == 0; }

uint64_t plat_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void plat_yield(void) { sched_yield(); }

void plat_write_stderr(const char *p, size_t n) {
    while (n) {
        ssize_t done = write(2, p, n);
        if (done <= 0) return;
        p += done;
        n -= (size_t)done;
    }
}

static int posix_prot(int prot) { return prot == PLAT_RW ? PROT_READ | PROT_WRITE : prot == PLAT_R ? PROT_READ : PROT_NONE; }

int plat_reserve_at(uint64_t addr, uint64_t size, int watch) {
    (void)watch;   // no write watch here: plat_written says so
    void *want = (void *)(uintptr_t)addr;
    void *p = mmap(want, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (p == MAP_FAILED) return 0;
    if (p != want) munmap(p, size);   // a kernel without MAP_FIXED_NOREPLACE took it as a hint
    return p == want;
}

void plat_release(uint64_t addr, uint64_t size) { munmap((void *)(uintptr_t)addr, size); }

int plat_commit(uint64_t addr, uint64_t size, int prot) {
    return mprotect((void *)(uintptr_t)addr, size, posix_prot(prot)) == 0;
}

int plat_decommit(uint64_t addr, uint64_t size) {
    void *p = (void *)(uintptr_t)addr;
    return madvise(p, size, MADV_DONTNEED) == 0 && mprotect(p, size, PROT_NONE) == 0;
}

int plat_protect(uint64_t addr, uint64_t size, int prot) {
    return mprotect((void *)(uintptr_t)addr, size, posix_prot(prot)) == 0;
}

int plat_is_free(uint64_t addr) {
    unsigned char v;
    return mincore((void *)(uintptr_t)(addr & ~0xFFFull), 0x1000, &v) != 0 && errno == ENOMEM;
}

uint64_t plat_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    (void)addr, (void)size, (void)pages, (void)max, (void)reset;
    return UINT64_MAX;
}

uint64_t plat_resident(uint64_t addr, uint64_t size) {
    enum { BATCH = 4096 };
    unsigned char v[BATCH];
    uint64_t pages = size >> 12, done = 0, resident = 0;
    while (done < pages) {
        uint64_t n = pages - done < BATCH ? pages - done : BATCH;
        if (mincore((void *)(uintptr_t)(addr + (done << 12)), n << 12, v)) return UINT64_MAX;
        for (uint64_t i = 0; i < n; i++) resident += v[i] & 1;
        done += n;
    }
    return resident << 12;
}

static uint64_t status_kib(const char *text, const char *key) {
    const char *p = strstr(text, key);
    return p ? strtoull(p + strlen(key), NULL, 10) << 10 : 0;
}

void plat_process_memory(PlatProcMem *out) {
    char buf[4096];
    *out = (PlatProcMem){0};
    int fd = open("/proc/self/status", O_RDONLY);
    if (fd < 0) return;
    ssize_t n = read(fd, buf, sizeof buf - 1);
    close(fd);
    if (n <= 0) return;
    buf[n] = 0;
    *out = (PlatProcMem){status_kib(buf, "RssAnon:"), status_kib(buf, "VmRSS:"), status_kib(buf, "VmHWM:")};
}

static void fault_signal(int sig, siginfo_t *si, void *ctx) {
    int write = 0;
    uint64_t pc = 0;
#if defined(__linux__)
    const ucontext_t *uc = ctx;
    write = (uc->uc_mcontext.gregs[REG_ERR] & 2) != 0;
    pc = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
#else
    (void)ctx;
#endif
    if (g_fault_hook && g_fault_hook((uint64_t)(uintptr_t)si->si_addr, write, pc)) return;
    signal(sig, SIG_DFL);   // the faulting access runs again and ends the process the default way
}

void plat_fault_hook(int (*hook)(uint64_t addr, int write, uint64_t host_pc)) {
    if (!g_fault_hook) {
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_sigaction = fault_signal;
        sa.sa_flags = SA_SIGINFO | SA_NODEFER;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGSEGV, &sa, NULL);
        sigaction(SIGBUS, &sa, NULL);
    }
    g_fault_hook = hook;
}

const void *plat_map_file(const char *path, uint64_t *size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    void *p = MAP_FAILED;
    if (!fstat(fd, &st) && st.st_size > 0) p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return NULL;
    *size = (uint64_t)st.st_size;
    return p;
}

void plat_unmap_file(const void *p, uint64_t size) {
    if (p) munmap((void *)p, size);
}

void plat_lock(PlatLock *l) { pthread_rwlock_wrlock(l); }
void plat_unlock(PlatLock *l) { pthread_rwlock_unlock(l); }
void plat_lock_shared(PlatLock *l) { pthread_rwlock_rdlock(l); }
void plat_unlock_shared(PlatLock *l) { pthread_rwlock_unlock(l); }
int plat_lock_shared_try(PlatLock *l) { return pthread_rwlock_tryrdlock(l) == 0; }

void plat_mutex_lock(PlatMutex *m) { pthread_mutex_lock(m); }
void plat_mutex_unlock(PlatMutex *m) { pthread_mutex_unlock(m); }
int plat_mutex_trylock(PlatMutex *m) { return pthread_mutex_trylock(m) == 0; }
void plat_cond_wait(PlatCond *c, PlatMutex *m) { pthread_cond_wait(c, m); }

void plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);   // the clock of a condition variable made by PTHREAD_COND_INITIALIZER
    uint64_t at = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec + (ns > (1ull << 62) ? 1ull << 62 : ns);
    ts.tv_sec = (time_t)(at / 1000000000ull);
    ts.tv_nsec = (long)(at % 1000000000ull);
    pthread_cond_timedwait(c, m, &ts);
}

void plat_cond_signal(PlatCond *c) { pthread_cond_signal(c); }
void plat_cond_broadcast(PlatCond *c) { pthread_cond_broadcast(c); }

// ---- threads
enum { INTR_IDLE, INTR_REQ, INTR_STOPPED, INTR_GO };

struct PlatThread {
    pthread_t pt;
    void (*fn)(void *arg);
    void *arg;
    int32_t intr;              // INTR_*: the stop handshake with the thread's signal handler
    uint64_t rip, rsp, to;
};

static _Thread_local PlatThread *self_thread;

static void *thread_start(void *p) {
    PlatThread *t = p;
    self_thread = t;
    t->fn(t->arg);   // t is not touched after fn returns: plat_thread_free may already have run
    self_thread = NULL;
    return NULL;
}

PlatThread *plat_thread_start(void (*fn)(void *arg), void *arg, uint64_t stack_reserve) {
    PlatThread *t = calloc(1, sizeof *t);
    pthread_attr_t a;
    if (!t || pthread_attr_init(&a)) {
        free(t);
        return NULL;
    }
    t->fn = fn;
    t->arg = arg;
    pthread_attr_setstacksize(&a, (size_t)stack_reserve);
    pthread_attr_setdetachstate(&a, PTHREAD_CREATE_DETACHED);
    int r = pthread_create(&t->pt, &a, thread_start, t);
    pthread_attr_destroy(&a);
    if (r) {
        free(t);
        return NULL;
    }
    return t;
}

void plat_thread_free(PlatThread *t) { free(t); }

// Another thread's clock is read only while that thread runs (the kernel keeps to this): the clock of a detached
// thread that has ended is undefined.
uint64_t plat_thread_cpu_ns(PlatThread *t) {
    clockid_t id = CLOCK_THREAD_CPUTIME_ID;
    struct timespec ts;
    if (t && pthread_getcpuclockid(t->pt, &id)) return 0;
    if (clock_gettime(id, &ts)) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// A better scheduling class or a negative nice value needs a privilege (CAP_SYS_NICE) a game does not run with.
int plat_thread_raise_priority(void) { return 0; }

void plat_sleep_ns(uint64_t ns) {
    struct timespec at;
    clock_gettime(CLOCK_MONOTONIC, &at);
    uint64_t t = (uint64_t)at.tv_sec * 1000000000ull + (uint64_t)at.tv_nsec + ns;
    at.tv_sec = (time_t)(t / 1000000000ull);
    at.tv_nsec = (long)(t % 1000000000ull);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &at, NULL) == EINTR) {
    }
}

int plat_random(void *p, size_t n) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return 0;
    size_t done = 0;
    while (done < n) {
        ssize_t r = read(fd, (char *)p + done, n - done);
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
    return done == n;
}

// ---- the system timer resolution: the POSIX scheduler has none to ask for (the test seam still records the calls)
static int sys_timer_set(uint64_t want_ns, int on, uint64_t *now_ns) {
    (void)want_ns, (void)on;
    if (now_ns) *now_ns = 0;
    return 0;
}
static uint64_t sys_timer_now(void) { return 0; }
static uint64_t sys_timer_finest(void) { return 0; }
static void timer_exit_hooks(void) {}

void plat_exit(int code) {
    plat_timer_release();
    fflush(NULL);
    _Exit(code);
}

// ---- interruption: a signal whose handler, on the interrupted thread, publishes its pc and stack pointer and spins
// until the interrupting thread says where to go on. The handler takes no lock. Adapted from DKB Recomp
// runtime/platform/linux.c at 9d8fa02 (GPL-3.0-or-later), "thread interruption": the same four-state handshake.
#define INTR_WAIT_NS 10000000ull
static int g_intr_sig;

static void intr_signal(int sig, siginfo_t *si, void *ctx) {
    (void)sig, (void)si;
    PlatThread *t = self_thread;
    ucontext_t *uc = ctx;
    if (!t) return;
    t->rip = (uint64_t)uc->uc_mcontext.gregs[REG_RIP];
    t->rsp = (uint64_t)uc->uc_mcontext.gregs[REG_RSP];
    int32_t req = INTR_REQ;
    if (!__atomic_compare_exchange_n(&t->intr, &req, INTR_STOPPED, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return;
    while (__atomic_load_n(&t->intr, __ATOMIC_ACQUIRE) != INTR_GO) plat_pause();
    if (t->to) uc->uc_mcontext.gregs[REG_RIP] = (greg_t)t->to;
    __atomic_store_n(&t->intr, INTR_IDLE, __ATOMIC_RELEASE);
}

int plat_intr_init(void) {
    if (g_intr_sig) return 1;
    int sig = SIGRTMIN + 4;   // the C library keeps the first real-time signals for itself; SIGRTMIN skips them
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = intr_signal;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(sig, &sa, NULL)) return 0;
    g_intr_sig = sig;
    return 1;
}

int plat_intr_stop(PlatThread *t, uint64_t *rip, uint64_t *rsp) {
    int32_t idle = INTR_IDLE;
    if (!g_intr_sig) return 0;
    if (!__atomic_compare_exchange_n(&t->intr, &idle, INTR_REQ, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    t->to = 0;
    if (pthread_kill(t->pt, g_intr_sig)) {
        __atomic_store_n(&t->intr, INTR_IDLE, __ATOMIC_RELEASE);
        return 0;
    }
    uint64_t end = plat_now_ns() + INTR_WAIT_NS;
    while (__atomic_load_n(&t->intr, __ATOMIC_ACQUIRE) != INTR_STOPPED) {
        if (plat_now_ns() < end) {
            plat_pause();
            continue;
        }
        int32_t req = INTR_REQ;
        if (__atomic_compare_exchange_n(&t->intr, &req, INTR_IDLE, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) return 0;
    }
    *rip = t->rip;
    *rsp = t->rsp;
    return 1;
}

int plat_intr_resume(PlatThread *t, uint64_t to) {
    t->to = to;
    __atomic_store_n(&t->intr, INTR_GO, __ATOMIC_RELEASE);
    return 1;
}

// ---- fibres: a stack switch of our own (the C library's swapcontext saves and restores the signal mask with a system
// call on every switch). plat_fibre_swap saves the callee-saved registers, the MXCSR and the x87 control word on the
// current stack, stores the stack pointer in *save and loads the other one. A new fibre's stack is laid out as
// plat_fibre_swap leaves one, returning into plat_fibre_entry, which realigns the stack and calls fn(arg) (r12, r13).
// Adapted from TotK Recomp runtime/platform/linux.c at 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh). Changed
// here: the switch names its source instead of reading a thread-local current fibre.
struct PlatFibre {
    void *sp;
    void *stack;   // the mapped stack, NULL for a thread's own context
    size_t size;
};

void plat_fibre_swap(void **save, void *load);
void plat_fibre_entry(void);
__asm__(".text\n"
        ".globl plat_fibre_swap\n"
        ".type plat_fibre_swap, @function\n"
        ".p2align 4\n"
        "plat_fibre_swap:\n"
        "    .cfi_startproc\n"
        "    pushq %rbp\n"
        "    pushq %rbx\n"
        "    pushq %r12\n"
        "    pushq %r13\n"
        "    pushq %r14\n"
        "    pushq %r15\n"
        "    subq $16, %rsp\n"
        "    stmxcsr 8(%rsp)\n"
        "    fnstcw (%rsp)\n"
        "    movq %rsp, (%rdi)\n"
        "    movq %rsi, %rsp\n"
        "    fldcw (%rsp)\n"
        "    ldmxcsr 8(%rsp)\n"
        "    addq $16, %rsp\n"
        "    popq %r15\n"
        "    popq %r14\n"
        "    popq %r13\n"
        "    popq %r12\n"
        "    popq %rbx\n"
        "    popq %rbp\n"
        "    ret\n"
        "    .cfi_endproc\n"
        ".size plat_fibre_swap, .-plat_fibre_swap\n"
        ".globl plat_fibre_entry\n"
        ".type plat_fibre_entry, @function\n"
        ".p2align 4\n"
        "plat_fibre_entry:\n"
        // The swap's ret lands here with rsp = top - 8, as at the entry of a called function; the call must be made
        // with rsp 16-byte aligned so that fn starts at 8 mod 16, as the System V ABI demands.
        "    andq $-16, %rsp\n"
        "    movq %r13, %rdi\n"
        "    callq *%r12\n"
        "    ud2\n"
        ".size plat_fibre_entry, .-plat_fibre_entry\n");

PlatFibre *plat_fibre_self(void) { return calloc(1, sizeof(PlatFibre)); }

void plat_fibre_unself(PlatFibre *own) { free(own); }

PlatFibre *plat_fibre_new(uint64_t stack_reserve, void (*fn)(void *arg), void *arg) {
    PlatFibre *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    // Reserved, not committed; the lowest page stays inaccessible, so an overflow faults instead of corrupting.
    void *m = mmap(NULL, (size_t)stack_reserve, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_STACK, -1, 0);
    if (m == MAP_FAILED) {
        free(f);
        return NULL;
    }
    mprotect(m, 4096, PROT_NONE);
    f->stack = m;
    f->size = (size_t)stack_reserve;
    uint64_t *top = (uint64_t *)(((uintptr_t)m + f->size) & ~(uintptr_t)15);
    uint64_t *sp = top - 10;   // fpu state (2 words), r15, r14, r13, r12, rbx, rbp, return address, padding
    sp[0] = 0x037f;            // x87 control word: the default
    sp[1] = 0x1f80;            // MXCSR: the default
    sp[2] = sp[3] = 0;         // r15, r14
    sp[4] = (uint64_t)(uintptr_t)arg;   // r13
    sp[5] = (uint64_t)(uintptr_t)fn;    // r12
    sp[6] = sp[7] = 0;                  // rbx, rbp
    sp[8] = (uint64_t)(uintptr_t)plat_fibre_entry;
    sp[9] = 0;
    f->sp = sp;
    return f;
}

void plat_fibre_switch(PlatFibre *from, PlatFibre *to) { plat_fibre_swap(&from->sp, to->sp); }

void plat_fibre_free(PlatFibre *f) {
    if (!f) return;
    if (f->stack) munmap(f->stack, f->size);
    free(f);
}
#endif

// ---- the system timer resolution, both platforms: the flag, the seam and the exit hooks (declared in plat.h)
static const PlatTimerOps g_sys_timer_ops = {sys_timer_set, sys_timer_now};
static PlatTimerOps g_timer_ops = {sys_timer_set, sys_timer_now};
static int g_timer_held;          // atomic: the request is in force
static int g_timer_hooked;        // the exit hooks are installed (by the first plat_timer_hold, before any other thread asks)
static uint64_t g_timer_want_ns;  // written before g_timer_held is

void plat_timer_ops(const PlatTimerOps *ops) {
    g_timer_ops = ops ? *ops : g_sys_timer_ops;
    if (!g_timer_ops.set) g_timer_ops.set = g_sys_timer_ops.set;
    if (!g_timer_ops.now) g_timer_ops.now = g_sys_timer_ops.now;
}

// Windows 11 ignores a process's request while its window is minimized or hidden, unless the process opts out with
// SetProcessInformation (ProcessPowerThrottling, PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION). Decided with the
// maintainer on 2026-11-07: no opt-out. A minimized or occluded game needs no sub-millisecond pacing, and the system's
// power saving stays intact. Not tested: the development machine runs Windows 10 (the timer resolution study).
int plat_timer_hold(uint64_t want_ns, uint64_t *granted_ns) {
    if (__atomic_load_n(&g_timer_held, __ATOMIC_ACQUIRE)) {
        if (granted_ns) *granted_ns = plat_timer_resolution_ns();
        return 1;
    }
    if (!g_timer_hooked) {   // before the request, so no exit between the two leaves it held
        g_timer_hooked = 1;
        atexit(plat_timer_release);
        timer_exit_hooks();
    }
    uint64_t now = 0;
    if (g_timer_ops.set(want_ns, 1, &now) != 0) return 0;
    if (granted_ns) *granted_ns = now;
    g_timer_want_ns = want_ns;
    __atomic_store_n(&g_timer_held, 1, __ATOMIC_RELEASE);
    return 1;
}

void plat_timer_release(void) {
    if (!__atomic_exchange_n(&g_timer_held, 0, __ATOMIC_ACQ_REL)) return;
    uint64_t now;
    g_timer_ops.set(g_timer_want_ns, 0, &now);
}

uint64_t plat_timer_resolution_ns(void) { return g_timer_ops.now(); }
uint64_t plat_timer_finest_ns(void) { return sys_timer_finest(); }
