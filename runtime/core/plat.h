// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The few operating-system calls runtime/core needs, behind one header. Windows is the target; the POSIX half exists
// so the boundary tests run in continuous integration.
#pragma once
#include <stddef.h>
#include <stdint.h>

void *plat_alloc_zeroed(size_t bytes);         // zero pages, backed on first write; NULL on failure
int plat_protect_read_only(void *p, size_t bytes);   // 1 on success
uint64_t plat_now_ns(void);                    // monotonic
void plat_yield(void);
void plat_write_stderr(const char *p, size_t n);

// ---- address space (architecture D6). A reservation takes address space only and allows no access; commit backs
// pages that read as zero, and only the pages written or read take memory. Every call returns 1 on success.
enum { PLAT_NONE = 0, PLAT_R = 1, PLAT_RW = 3 };
int plat_reserve_at(uint64_t addr, uint64_t size, int watch);   // exactly [addr, addr+size); watch: write-watched
void plat_release(uint64_t addr, uint64_t size);                // a whole reservation made by plat_reserve_at
int plat_commit(uint64_t addr, uint64_t size, int prot);
int plat_decommit(uint64_t addr, uint64_t size);                // the pages read as zero when committed again
int plat_protect(uint64_t addr, uint64_t size, int prot);
int plat_is_free(uint64_t addr);                                // nothing of the host's lies at addr
// Pages of a write-watched range written since the last reset: their addresses go to pages (at most max), the count
// is returned; UINT64_MAX when the platform cannot tell.
uint64_t plat_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset);
uint64_t plat_resident(uint64_t addr, uint64_t size);           // bytes of committed pages in the working set
// On Windows a write-watched reservation (the heap and alias regions) is made of pieces cut at multiples of PLAT_PIECE
// in the address space, and every call above splits its range at those boundaries, a reservation being the unit of each
// of them. The reason is the GPU back end's host imports: the driver walks the run of committed pages that follows an
// imported pointer to the end of its reservation, 14 ms per GiB (the write-watch study), and a piece bounds that walk
// to one import window (oot2/backing.h). The POSIX half reserves in one piece; nothing there walks.
#define PLAT_PIECE (64ull << 20)
typedef struct PlatProcMem {
    uint64_t private_bytes;      // commit charge of the process (Windows PrivateUsage; POSIX: anonymous resident)
    uint64_t working_set, peak_working_set;
} PlatProcMem;
void plat_process_memory(PlatProcMem *out);
// ---- the machine, for the M1 capture (runtime/core/plat_sys.c). Physical memory installed and available now, in bytes
// (0 when the platform cannot tell). Pages written to the page file since the system started, counted for every process
// of the machine, not this one alone (Windows: the raw count behind PDH's \Memory\Pages Output/sec; Linux: pswpout of
// /proc/vmstat), or UINT64_MAX when it cannot be read; plat_pages_output_name says which counter it is. The first call
// opens the counter and may take a moment, so a caller makes it before the timed work. Logical processors; the process id.
void plat_system_memory(uint64_t *total, uint64_t *available);
uint64_t plat_pages_output(void);
const char *plat_pages_output_name(void);
int plat_cpu_count(void);
unsigned plat_pid(void);
// 0 only when no process has this id; 1 when one has, and when that cannot be told (a caller that deletes what a dead process
// left must never take a live process for a dead one).
int plat_pid_alive(unsigned pid);
// Access faults of any thread go to hook first: it returns 1 when the access can be retried (it committed the page),
// 0 to leave the fault to the next handler. write is 1 for a store; host_pc is the faulting host instruction.
void plat_fault_hook(int (*hook)(uint64_t addr, int write, uint64_t host_pc));

// ---- files: a read-only view of a whole file; NULL on failure
const void *plat_map_file(const char *path, uint64_t *size);
void plat_unmap_file(const void *p, uint64_t size);

// ---- a reader-writer lock that needs no initialisation call
#ifdef _WIN32
typedef struct PlatLock { void *opaque; } PlatLock;   // an SRWLOCK
#define PLAT_LOCK_INIT {0}
#else
#include <pthread.h>
typedef pthread_rwlock_t PlatLock;
#define PLAT_LOCK_INIT PTHREAD_RWLOCK_INITIALIZER
#endif
void plat_lock(PlatLock *l);
void plat_unlock(PlatLock *l);
void plat_lock_shared(PlatLock *l);
int plat_lock_shared_try(PlatLock *l);   // 1 when it took the lock shared, 0 when a writer holds it or waits
void plat_unlock_shared(PlatLock *l);

// ---- threads (architecture D7). A host thread per guest thread, with a large reserved stack: every guest call is a
// host call, so guest recursion is host recursion.
typedef struct PlatThread PlatThread;
// Start fn(arg) on a new thread whose stack reserves stack_reserve bytes (committed as it grows); NULL on failure.
// Nobody joins it: the handle stays valid for plat_thread_cpu_ns and plat_intr_* until plat_thread_free.
PlatThread *plat_thread_start(void (*fn)(void *arg), void *arg, uint64_t stack_reserve);
void plat_thread_free(PlatThread *t);   // after the thread stopped running guest code; it may still be returning
// CPU time the thread has used (user and kernel), in nanoseconds; the calling thread's when t is NULL.
uint64_t plat_thread_cpu_ns(PlatThread *t);
// Raise the calling thread above normal priority, for a clock that must wake on time while guest threads keep every
// processor busy: 1 on success, 0 where the platform asks a privilege for it (POSIX), and the thread stays as it was.
int plat_thread_raise_priority(void);
static inline void plat_pause(void) { __builtin_ia32_pause(); }
// Sleep about ns on the system's high-resolution timer: never early, late by up to PLAT_SLEEP_LATE_NS.
void plat_sleep_ns(uint64_t ns);
#ifdef _WIN32
#define PLAT_SLEEP_LATE_NS 500000ull   // high-resolution waitable timer
#else
#define PLAT_SLEEP_LATE_NS 60000ull    // clock_nanosleep and the scheduler's wake-up
#endif
int plat_random(void *p, size_t n);     // bytes from the system's generator; 1 on success
__attribute__((noreturn)) void plat_exit(int code);   // flushes the C streams, then ends the process

// ---- the system timer resolution (the timer resolution hold). The Windows scheduler wakes a waiting thread at its timer's tick, 15.6
// ms by default and as fine as 0.5 ms when some process asks for it; the request is a flag of the calling process, not a
// count, and the system drops it when the process ends, however it ends. plat_timer_hold asks for want_ns; the policy
// (the setting, the log line) is rt_timer_hold_start in timer.c, so this layer stays free of settings.
// Returns 1 when the request is held (also when it already was: a second call changes nothing), 0 when the system
// refused it; *granted_ns, when not NULL, is the resolution the system reports, which is the finest it allows when
// want_ns is finer than that. The first call also makes the process release the request at a normal exit and when a
// console control event ends it (Ctrl+C, Ctrl+Break, closing the console, log-off, shutdown). Every other exit path of
// the runtime calls plat_timer_release itself: plat_exit, and fatal.c's end of a stop.
int plat_timer_hold(uint64_t want_ns, uint64_t *granted_ns);
// Give the request back. Lock-free, silent and idempotent, so a fatal error and a control event on a thread the system
// made can call it: only the call that finds the request held makes the system call.
void plat_timer_release(void);
// The resolution in force now, any process's request included, in nanoseconds; 0 where the platform has none (POSIX).
uint64_t plat_timer_resolution_ns(void);
// The finest resolution the system allows, in nanoseconds; 0 where the platform has none.
uint64_t plat_timer_finest_ns(void);
// A test's view of the system call, so the request and the release can be recorded on every platform and in a process
// that must not touch the machine-wide setting: set(want_ns, on, &now_ns) asks (on) or gives back (off) and returns 0
// on success, with the resolution in force afterwards in now_ns; now() is the current resolution. A NULL member keeps
// the system's, and a NULL table restores both. Install it before the first plat_timer_hold, while no other thread uses
// the timer; each test case is a process of its own.
typedef struct PlatTimerOps {
    int (*set)(uint64_t want_ns, int on, uint64_t *now_ns);
    uint64_t (*now)(void);
} PlatTimerOps;
void plat_timer_ops(const PlatTimerOps *ops);

// ---- fibres: contexts with host stacks of their own that threads switch between (nn::os fibres, oot2/kernel.h). A
// fibre not running may be resumed by any thread. Every switch names both sides, so no thread-local state is read across
// it: a frame that spans a switch may wake on another host thread, where its compiler-cached thread-local addresses
// would be another thread's.
typedef struct PlatFibre PlatFibre;
PlatFibre *plat_fibre_self(void);   // the calling thread's own context as a fibre, before its first switch; NULL on failure
void plat_fibre_unself(PlatFibre *own);   // on the thread's own context again, after its last switch; frees own
// A fibre that runs fn(arg) on a stack reserving stack_reserve bytes (committed as it grows); fn never returns. NULL on
// failure.
PlatFibre *plat_fibre_new(uint64_t stack_reserve, void (*fn)(void *arg), void *arg);
void plat_fibre_switch(PlatFibre *from, PlatFibre *to);   // from: the fibre the calling thread runs now
void plat_fibre_free(PlatFibre *f);                       // a fibre no thread runs

// ---- a mutex and condition variables that need no initialisation call
#ifdef _WIN32
typedef struct PlatMutex { void *opaque; } PlatMutex;   // an SRWLOCK, always taken exclusively
typedef struct PlatCond { void *opaque; } PlatCond;     // a CONDITION_VARIABLE
#define PLAT_MUTEX_INIT {0}
#define PLAT_COND_INIT {0}
#else
typedef pthread_mutex_t PlatMutex;
typedef pthread_cond_t PlatCond;
#define PLAT_MUTEX_INIT PTHREAD_MUTEX_INITIALIZER
#define PLAT_COND_INIT PTHREAD_COND_INITIALIZER
#endif
void plat_mutex_lock(PlatMutex *m);
void plat_mutex_unlock(PlatMutex *m);
int plat_mutex_trylock(PlatMutex *m);   // 1 when it took the mutex, 0 when another thread holds it
void plat_cond_wait(PlatCond *c, PlatMutex *m);
void plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns);   // returns on a signal, a timeout or spuriously
void plat_cond_signal(PlatCond *c);
void plat_cond_broadcast(PlatCond *c);

// ---- interrupting a running thread (asynchronous preemption, D7). plat_intr_stop stops t wherever it runs and reads
// its instruction and stack pointers; plat_intr_resume lets it go on, at `to` when non-zero. Between the two calls the
// caller takes no lock and allocates nothing, since t may hold any lock of the C library. plat_intr_init installs what
// the platform needs (a signal handler on POSIX) and runs once, before the first thread starts.
int plat_intr_init(void);
int plat_intr_stop(PlatThread *t, uint64_t *rip, uint64_t *rsp);   // 0 when t could not be stopped
int plat_intr_resume(PlatThread *t, uint64_t to);                  // 0 when the new pc could not be set
