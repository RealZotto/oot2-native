// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The guest address space with commit on first touch (architecture D6), and the memory SVCs.
// Adapted from DKB Recomp runtime/memory.c at 9d8fa02 (GPL-3.0-or-later; adapted there from MacNativeGames and LAS
// Recomp, MIT): the Horizon states, permissions and results, the alias page map, thread stacks moved by copy, the
// module permission table and the shape of QueryMemory. Changed here: mapping commits nothing; a first touch commits
// the 64 KiB chunk around it, clipped to what the guest mapped, and a touch anywhere else is a named fatal; the regions
// are found once below the modules instead of re-running the process; a moved stack copies only its committed pages;
// what is not the guest's reads as inaccessible, not as code; the kernel's thread-local pages have a region of their
// own, and the main thread's stack lies at the top of the stack region, committed on touch like the rest.
//
// Locking: map changes hold `lock` exclusively and never touch uncommitted guest memory while they do (they commit
// first), because the first-touch handler takes it shared on the faulting thread. Page bitmaps change with atomics,
// since faulting threads commit concurrently under the shared lock.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "kern.h"
#include "oot2/loader.h"
#include "oot2/settings.h"
#include "plat.h"

#define PAGE 0x1000ull
#define GiB (1ull << 30)
#define SEARCH_STEP (4 * GiB)
#define MAX_MOVES 4096
#define TOUCH_REPEAT_LIMIT 16

// Horizon memory states, permissions, attributes and results (Switchbrew, "SVC").
enum { ST_FREE = 0, ST_CODE = 3, ST_CODE_DATA = 4, ST_NORMAL = 5, ST_STACK = 0xB, ST_THREAD_LOCAL = 0xC,
       ST_INACCESSIBLE = 0x10 };
enum { PERM_NONE = 0, PERM_R = 1, PERM_RW = 3, PERM_RX = 5 };
enum { ATTR_LOCKED = 1, ATTR_UNCACHED = 8, ATTR_PERMISSION_LOCKED = 16 };
#define R_OK 0u
#define R_INVALID_SIZE 0xca01u
#define R_INVALID_ADDRESS 0xcc01u
#define R_OUT_OF_MEMORY 0xd001u
#define R_INVALID_CURRENT_MEMORY 0xd401u
#define R_INVALID_NEW_PERMISSION 0xd801u
#define R_INVALID_MEMORY_REGION 0xdc01u
#define R_INVALID_COMBINATION 0xe801u
#define R_LIMIT_REACHED 0x10801u

#define ASLR_BASE 0x8000000ull   // start of Horizon's ASLR region in a 39-bit address space (GetInfo 12)

typedef struct Region {
    const char *name;
    uint64_t base, size;
    int watch;
    uint64_t *bits;            // one bit per committed page
    uint64_t committed;        // bytes, atomic
    uint64_t touch_commits;    // atomic
} Region;

static Region R[RT_MEM_REGIONS] = {
    [RT_MEM_STACK] = {.name = "stack"},
    [RT_MEM_HEAP] = {.name = "heap", .watch = 1},
    [RT_MEM_ALIAS] = {.name = "alias", .watch = 1},
    [RT_MEM_MODULES] = {.name = "modules"},
    [RT_MEM_KERNEL] = {.name = "kernel"},
};
static RtMemConfig C;
static int g_ready;
static PlatLock lock = PLAT_LOCK_INIT;
static uint64_t heap_size, alias_bytes, kernel_bytes;
static uint64_t *alias_map;    // one bit per alias page MapPhysicalMemory mapped
static uint64_t *kernel_map;   // one bit per kernel-region page the kernel allocated (mem_kernel_alloc)
static uint64_t mstack_base, mstack_size;   // the main thread's stack (mem_main_stack), at the top of the stack region
typedef struct Move { uint64_t dst, src, size; } Move;   // MapMemory: a stack moved from src to dst
static Move moves[MAX_MOVES];
static unsigned nmoves;
typedef struct Mod { const RtLoaded *m; uint8_t *perm; } Mod;   // perm: per page from data_off, perm | attribute
static Mod mods[RT_MAX_LOADED];
static int nmods;
static _Thread_local uint64_t last_touch, touch_repeats;

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
static int inside(uint64_t a, uint64_t n, uint64_t base, uint64_t size) { return a >= base && n <= size && a - base <= size - n; }
static int overlaps(uint64_t a, uint64_t n, uint64_t b, uint64_t m) { return a < b + m && b < a + n; }

static int region_of(uint64_t a) {
    for (int r = 0; r < RT_MEM_REGIONS; r++)
        if (a - R[r].base < R[r].size) return r;
    return -1;
}

// ---- page bitmaps
static int bit(const uint64_t *b, uint64_t i) { return (int)((__atomic_load_n(&b[i >> 6], __ATOMIC_RELAXED) >> (i & 63)) & 1); }

static int bit_change(uint64_t *b, uint64_t i, int on) {   // 1 when the bit changed
    uint64_t m = 1ull << (i & 63);
    if (on) return !(__atomic_fetch_or(&b[i >> 6], m, __ATOMIC_RELAXED) & m);
    return (__atomic_fetch_and(&b[i >> 6], ~m, __ATOMIC_RELAXED) & m) != 0;
}

// End of the run of value v that holds index i (exclusive), and its first index; whole words at a time.
static uint64_t run_end(const uint64_t *b, uint64_t i, uint64_t n, int v) {
    const uint64_t full = v ? ~0ull : 0;
    while (i < n) {
        if (!(i & 63) && i + 64 <= n && __atomic_load_n(&b[i >> 6], __ATOMIC_RELAXED) == full) { i += 64; continue; }
        if (bit(b, i) != v) break;
        i++;
    }
    return i;
}

static uint64_t run_start(const uint64_t *b, uint64_t i, int v) {
    const uint64_t full = v ? ~0ull : 0;
    while (i > 0) {
        if (!(i & 63) && __atomic_load_n(&b[(i >> 6) - 1], __ATOMIC_RELAXED) == full) { i -= 64; continue; }
        if (bit(b, i - 1) != v) break;
        i--;
    }
    return i;
}

static uint64_t *new_bitmap(uint64_t bits) {
    uint64_t *p = plat_alloc_zeroed(align_up(bits, 64) / 8);
    if (!p) rt_fatal("guest memory: cannot allocate a page map of %llu bits", (unsigned long long)bits);
    return p;
}

// ---- commit and decommit, counted per region, and the commit epoch (rt_mem_commit_epoch): every change of a page map
// moves it once, after the bits changed
static uint64_t g_commit_epoch;   // atomic

// The decommit record (rt_mem_decommitted): the range of each of the last DECOMMIT_LOG decommits of committed pages, by
// number. Every decommit is made under the exclusive lock (SetHeapSize, MapMemory and UnmapMemory, UnmapPhysicalMemory, a
// kernel page freed, a page released), so one writer at a time; readers take no lock (the GPU back end's completion thread
// is one), so a slot can be written again under a reader: the writer names the number it is writing first (g_dec_start),
// behind a release fence, and a reader that may have read a slot a newer decommit took answers "decommitted".
#define DECOMMIT_LOG 1024u
static uint64_t g_dec_lo[DECOMMIT_LOG], g_dec_hi[DECOMMIT_LOG];   // atomic: decommit k at k % DECOMMIT_LOG
static uint64_t g_dec_start;                                       // atomic: the newest decommit being written
static uint64_t g_dec_epoch;                                       // atomic: the newest decommit written

static void decommit_note(uint64_t a, uint64_t n) {
    uint64_t k = __atomic_load_n(&g_dec_epoch, __ATOMIC_RELAXED) + 1;
    __atomic_store_n(&g_dec_start, k, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&g_dec_lo[k % DECOMMIT_LOG], a, __ATOMIC_RELAXED);
    __atomic_store_n(&g_dec_hi[k % DECOMMIT_LOG], a + n, __ATOMIC_RELAXED);
    __atomic_store_n(&g_dec_epoch, k, __ATOMIC_RELEASE);
}

static void account(int r, uint64_t a, uint64_t n, int on) {
    uint64_t first = (a - R[r].base) / PAGE, changed = 0;
    for (uint64_t i = 0; i < n / PAGE; i++) changed += (uint64_t)bit_change(R[r].bits, first + i, on);
    if (on) __atomic_fetch_add(&R[r].committed, changed * PAGE, __ATOMIC_RELAXED);
    else __atomic_fetch_sub(&R[r].committed, changed * PAGE, __ATOMIC_RELAXED);
    if (changed && !on) decommit_note(a, n);   // before the lock is dropped, so before any commit of these pages again
    if (changed) __atomic_fetch_add(&g_commit_epoch, 1, __ATOMIC_RELEASE);
}

static uint64_t managed_pages;   // atomic: pages of released or armed state in every page range (zero in a default run)
static uint8_t page_state(int r, uint64_t a);
static void pages_forget(uint64_t a, uint64_t n);

static void commit_run(int r, uint64_t a, uint64_t n, int prot) {
    if (!plat_commit(a, n, prot))
        rt_fatal("guest memory: committing %#llx+%#llx in the %s region failed (the system commit limit?)",
                 (unsigned long long)a, (unsigned long long)n, R[r].name);
    account(r, a, n, 1);
}

// A page of a page range that is armed, released or being changed is not a page to commit: committing an armed page
// would lift its protection, and committing a released one would take it for a first touch (lever L1).
static void commit(int r, uint64_t a, uint64_t n, int prot) {
    if (!n) return;
    if (!__atomic_load_n(&managed_pages, __ATOMIC_ACQUIRE) || (r != RT_MEM_HEAP && r != RT_MEM_ALIAS)) {
        commit_run(r, a, n, prot);
        return;
    }
    for (uint64_t e = a + n; a < e;) {
        uint64_t s = a;
        while (a < e && !page_state(r, a)) a += PAGE;
        if (a > s) commit_run(r, s, a - s, prot);
        while (a < e && page_state(r, a)) a += PAGE;
    }
}

static void decommit(int r, uint64_t a, uint64_t n) {
    if (!n) return;
    if (!plat_decommit(a, n))
        rt_fatal("guest memory: decommitting %#llx+%#llx in the %s region failed", (unsigned long long)a,
                 (unsigned long long)n, R[r].name);
    account(r, a, n, 0);
    if (r == RT_MEM_HEAP || r == RT_MEM_ALIAS) pages_forget(a, n);
}

// ---- modules
static int module_at(uint64_t a) {
    for (int i = 0; i < nmods; i++)
        if (a - mods[i].m->base < mods[i].m->size) return i;
    return -1;
}

static unsigned module_perm(int i, uint64_t off) {   // perm | attribute of the page at off
    const RtLoaded *m = mods[i].m;
    if (off < m->data_off) return off < m->rodata_off ? PERM_RX : PERM_R;
    return mods[i].perm[(off - m->data_off) / PAGE];
}

static uint64_t image_bytes(void) {
    uint64_t n = 0;
    for (int i = 0; i < nmods; i++) n += mods[i].m->size;
    return n;
}

// What the application pool holds now: module images, the main thread's stack, the heap and the alias pages.
static uint64_t pool_used(void) { return image_bytes() + mstack_size + heap_size + alias_bytes; }

// ---- first touch
// "0x7000001234 (main+0x1234)", "0x1040000000 (heap+0x0)", or the bare address.
static const char *describe(uint64_t a, char *buf, size_t size) {
    int i = module_at(a), r = region_of(a);
    if (i >= 0) snprintf(buf, size, "%#llx (%s+%#llx)", (unsigned long long)a, mods[i].m->name,
                         (unsigned long long)(a - mods[i].m->base));
    else if (r >= 0) snprintf(buf, size, "%#llx (%s+%#llx)", (unsigned long long)a, R[r].name,
                              (unsigned long long)(a - R[r].base));
    else snprintf(buf, size, "%#llx", (unsigned long long)a);
    return buf;
}

// The fault this thread is reporting, while it reads the thread's frame chain for the report: that read can fault in its turn
// (x29 aligned and below RT_ADDRESS_LIMIT, but not mapped), and then the first fault stays the reason.
static _Thread_local char g_report[512];

__attribute__((noreturn, cold)) static void touch_fatal(uint64_t a, int write, uint64_t host_pc, const char *why) {
    char at[96], fn[96] = "";
    if (g_report[0])
        rt_fatal("%s; reading the thread's frame chain for this report then faulted: %s of %s: %s", g_report,
                 write ? "write" : "read", describe(a, at, sizeof at), why);
    uint64_t guest_fn;
    if (dispatch_host_pc_guest((uintptr_t)host_pc, &guest_fn)) {   // the lifted function whose code touched it
        char name[80];
        snprintf(fn, sizeof fn, ", in the guest function %s", rt_guest_name(guest_fn, name, sizeof name));
    }
    snprintf(g_report, sizeof g_report, "guest memory: %s of %s: %s (host pc %#llx%s)", write ? "write" : "read",
             describe(a, at, sizeof at), why, (unsigned long long)host_pc, fn);
    const RtThread *t = kcurrent();   // the thread that faulted: its link register and frame chain name the caller
    if (t && t->cur) {   // as last saved: lifted code holds its registers in host registers between stores, so these can be stale
        char lr[96];
        rt_log("guest state as last saved before the fault: x30 %s, x29 chain follows",
               rt_guest_name(t->cur->x[30], lr, sizeof lr));
        rt_guest_backtrace(t->cur);
    }
    rt_fatal("%s", g_report);
}

// ---- page ranges: counting and releasing (lever L1, architecture D19, D27)
// A range of the heap or alias region is watched in one of two modes, at 4 KiB granularity inside the 64 KiB chunks a
// first touch commits. Counting: committed pages are armed (no access), the first access of any kind is counted, the
// page gets its access back and the run goes on. Releasing: the owner asks for committed pages to be decommitted
// (unless it holds the range busy); a write commits the page again, zero-filled, and calls the owner back; a read is
// the named fatal "guest memory: ...", counted first. The generic half knows no textures: the owner is a callback pair.
// A page that is armed or released is never taken for a first touch (commit() leaves it out of a chunk) and a fault on
// it is settled here before the first-touch classification sees it.
// Locking: ranges change under the exclusive lock; the fault path holds it shared and settles each page with a
// compare-and-swap through PG_BUSY, so concurrent faults on one page change it once. The owner's `written` runs after
// the lock is dropped; its `busy` runs under the exclusive lock and must not call back into this API.
enum { PG_NORMAL = 0, PG_ARMED, PG_RELEASED, PG_BUSY };   // PG_BUSY: a fault is changing the page
typedef struct PageRange {
    int used, mode, region;
    uint64_t base, size;
    RtPageOwner owner;
    uint8_t *state;   // one byte per 4 KiB page; atomic
    RtPageStats st;   // atomic counters
} PageRange;
static PageRange PR[RT_MAX_PAGE_RANGES];
static int g_pages_level;

int rt_pages_level(void) { return g_pages_level; }

static PageRange *pages_find(uint64_t a) {
    for (int i = 0; i < RT_MAX_PAGE_RANGES; i++)
        if (PR[i].used && a - PR[i].base < PR[i].size) return &PR[i];
    return NULL;
}

static uint8_t page_state(int r, uint64_t a) {   // under lock
    (void)r;
    PageRange *p = pages_find(a);
    return p ? __atomic_load_n(&p->state[(a - p->base) / PAGE], __ATOMIC_ACQUIRE) : (uint8_t)PG_NORMAL;
}

static void state_normal(PageRange *p, uint64_t i) {   // the page leaves its armed, released or busy state
    __atomic_store_n(&p->state[i], PG_NORMAL, __ATOMIC_RELEASE);
    __atomic_fetch_sub(&managed_pages, 1, __ATOMIC_ACQ_REL);
}

// Exclusive lock. After the pages were decommitted for another reason (heap shrink, unmap): their state is gone.
static void pages_forget(uint64_t a, uint64_t n) {
    if (!__atomic_load_n(&managed_pages, __ATOMIC_ACQUIRE)) return;
    for (int k = 0; k < RT_MAX_PAGE_RANGES; k++) {
        PageRange *p = &PR[k];
        if (!p->used || !overlaps(a, n, p->base, p->size)) continue;
        uint64_t lo = a > p->base ? a : p->base, hi = a + n < p->base + p->size ? a + n : p->base + p->size;
        for (uint64_t i = (lo - p->base) / PAGE; i < (hi - p->base) / PAGE; i++) {
            uint8_t s = p->state[i];
            if (s == PG_NORMAL) continue;
            __atomic_fetch_sub(s == PG_ARMED ? &p->st.armed : &p->st.released, 1, __ATOMIC_RELAXED);
            state_normal(p, i);
        }
    }
}

// Exclusive lock. Every armed page of [a, a+n) gets its access back and every released one is committed again,
// zero-filled, with no callback: used before the runtime copies pages itself, and when a range is dropped.
static void pages_reset(uint64_t a, uint64_t n) {
    if (!__atomic_load_n(&managed_pages, __ATOMIC_ACQUIRE)) return;
    for (int k = 0; k < RT_MAX_PAGE_RANGES; k++) {
        PageRange *p = &PR[k];
        if (!p->used || !overlaps(a, n, p->base, p->size)) continue;
        uint64_t lo = a > p->base ? a : p->base, hi = a + n < p->base + p->size ? a + n : p->base + p->size;
        for (uint64_t i = (lo - p->base) / PAGE; i < (hi - p->base) / PAGE; i++) {
            uint8_t s = p->state[i];
            uint64_t pg = p->base + i * PAGE;
            if (s == PG_NORMAL) continue;
            if (s == PG_ARMED) {
                if (!plat_protect(pg, PAGE, PLAT_RW))
                    rt_fatal("guest memory: lifting the protection of %#llx failed", (unsigned long long)pg);
                __atomic_fetch_sub(&p->st.armed, 1, __ATOMIC_RELAXED);
            } else {
                commit_run(p->region, pg, PAGE, PLAT_RW);
                __atomic_fetch_sub(&p->st.released, 1, __ATOMIC_RELAXED);
            }
            state_normal(p, i);
        }
    }
}

static int pages_overlap(uint64_t a, uint64_t n) {
    int hit = 0;
    plat_lock_shared(&lock);
    for (int k = 0; k < RT_MAX_PAGE_RANGES; k++) hit |= PR[k].used && overlaps(a, n, PR[k].base, PR[k].size);
    plat_unlock_shared(&lock);
    return hit;
}

typedef struct PageHit { const char *name; int guest; uint64_t fn; } PageHit;
typedef enum { PA_NONE, PA_HANDLED, PA_READ_RELEASED } PageAccess;

// An access to the page at a: from a fault (host_io 0, host_pc the faulting instruction) or from host I/O through
// rt_mem_commit_io (host_io 1). PA_NONE: not a page of a range in a state this settles. PA_HANDLED: the page is
// accessible now, retry. PA_READ_RELEASED: a read of a released page, counted; the caller names the fault.
static PageAccess page_access(uint64_t a, int write, int host_io, uint64_t host_pc, PageHit *hit) {
    if (!__atomic_load_n(&managed_pages, __ATOMIC_ACQUIRE)) return PA_NONE;
    const uint64_t pg = a & ~(PAGE - 1);
    uint64_t fn = 0;
    const int guest = !host_io && dispatch_host_pc_guest((uintptr_t)host_pc, &fn);
    void (*written)(void *, uint64_t) = NULL;
    void (*touched)(void *, uint64_t, int, int, uint64_t) = NULL;
    void *user = NULL;
    PageAccess res = PA_HANDLED;
    plat_lock_shared(&lock);
    PageRange *p = pages_find(pg);
    if (!p) {
        plat_unlock_shared(&lock);
        return PA_NONE;
    }
    uint8_t *s = &p->state[(pg - p->base) / PAGE];
    uint8_t v;
    while ((v = __atomic_load_n(s, __ATOMIC_ACQUIRE)) == PG_BUSY) plat_pause();
    uint64_t *count = write ? (guest ? &p->st.guest_writes : &p->st.host_writes)
                            : (guest ? &p->st.guest_reads : &p->st.host_reads);
    if (v == PG_NORMAL) {
        res = PA_NONE;   // not armed or released: a fault that raced with another thread's restore, or a first touch
    } else if (v == PG_ARMED) {
        uint8_t e = PG_ARMED;
        if (__atomic_compare_exchange_n(s, &e, PG_BUSY, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            if (!plat_protect(pg, PAGE, PLAT_RW))
                rt_fatal("guest memory: lifting the protection of %#llx failed", (unsigned long long)pg);
            __atomic_fetch_sub(&p->st.armed, 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(count, 1, __ATOMIC_RELAXED);
            state_normal(p, (pg - p->base) / PAGE);
            touched = p->owner.touched;
            user = p->owner.user;
        }
    } else if (!write) {   // PG_RELEASED
        __atomic_fetch_add(count, 1, __ATOMIC_RELAXED);
        hit->name = p->owner.name ? p->owner.name : "(unnamed)";
        hit->guest = guest;
        hit->fn = fn;
        res = PA_READ_RELEASED;
        touched = p->owner.touched;
        user = p->owner.user;
    } else {
        uint8_t e = PG_RELEASED;
        if (__atomic_compare_exchange_n(s, &e, PG_BUSY, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            commit_run(p->region, pg, PAGE, PLAT_RW);
            __atomic_fetch_sub(&p->st.released, 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&p->st.recommits, 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(count, 1, __ATOMIC_RELAXED);
            state_normal(p, (pg - p->base) / PAGE);
            written = p->owner.written;
            touched = p->owner.touched;
            user = p->owner.user;
        }
    }
    plat_unlock_shared(&lock);
    if (touched) touched(user, pg, write, guest, fn);
    if (written) written(user, pg);
    return res;
}

__attribute__((noreturn, cold)) static void released_fatal(uint64_t a, int write, uint64_t host_pc, const PageHit *h) {
    char why[320], fn[96];
    if (h->guest)
        snprintf(why, sizeof why, "a page of range \"%s\" was released and a read of it is a fault; read by guest code in %s",
                 h->name, rt_guest_name(h->fn, fn, sizeof fn));
    else
        snprintf(why, sizeof why, "a page of range \"%s\" was released and a read of it is a fault; read by runtime or "
                 "back-end code", h->name);
    touch_fatal(a, write, host_pc, why);
}

// The public half.
static int pages_slot(int id) { return id >= 0 && id < RT_MAX_PAGE_RANGES && PR[id].used; }

int rt_pages_watch(uint64_t addr, uint64_t size, int mode, const RtPageOwner *owner) {
    if (!rt_mem_ready()) rt_fatal("rt_pages_watch before rt_mem_init");
    if (g_pages_level < mode || (mode != RT_PAGES_COUNTING && mode != RT_PAGES_RELEASING) || !owner) return -1;
    int r = region_of(addr);
    if ((r != RT_MEM_HEAP && r != RT_MEM_ALIAS) || !size || ((addr | size) & (PAGE - 1)) ||
        !inside(addr, size, R[r].base, R[r].size))
        return -1;
    // The host-address index of the function tables is built here, never inside a fault handler (malloc and a lock
    // are not safe in a signal handler), so the function tables must be sealed first.
    if (!rt_sealed()) rt_fatal("rt_pages_watch before rt_dispatch_init: the function tables are not sealed");
    uint8_t *state = calloc(size / PAGE, 1);
    if (!state) return -1;
    dispatch_index_host();
    int id = -1;
    plat_lock(&lock);
    for (int k = 0; k < RT_MAX_PAGE_RANGES; k++) {
        if (PR[k].used) {
            if (overlaps(addr, size, PR[k].base, PR[k].size)) { id = -2; break; }
        } else if (id == -1) {
            id = k;
        }
    }
    if (id >= 0) {
        PR[id] = (PageRange){.used = 1, .mode = mode, .region = r, .base = addr, .size = size, .owner = *owner,
                             .state = state};
        PR[id].st.mode = mode;
    }
    plat_unlock(&lock);
    if (id < 0) {
        free(state);
        return -1;
    }
    return id;
}

void rt_pages_unwatch(int id) {
    plat_lock(&lock);
    if (pages_slot(id)) {
        PageRange *p = &PR[id];
        pages_reset(p->base, p->size);
        p->used = 0;
        free(p->state);
        p->state = NULL;
    }
    plat_unlock(&lock);
}

// Under the exclusive lock: the committed pages of [addr, addr+size) of range p that are in no special state become
// `to` (armed: no access; released: decommitted). Returns how many.
static uint64_t pages_change(PageRange *p, uint64_t addr, uint64_t size, uint8_t to) {
    uint64_t n = 0;
    const uint64_t first = (addr - p->base) / PAGE, count = size / PAGE, rfirst = (p->base - R[p->region].base) / PAGE;
    for (uint64_t i = first; i < first + count;) {
        if (p->state[i] != PG_NORMAL || !bit(R[p->region].bits, rfirst + i)) { i++; continue; }
        uint64_t j = i;
        while (j < first + count && p->state[j] == PG_NORMAL && bit(R[p->region].bits, rfirst + j)) j++;
        uint64_t at = p->base + i * PAGE, bytes = (j - i) * PAGE;
        if (to == PG_ARMED) {
            if (!plat_protect(at, bytes, PLAT_NONE))
                rt_fatal("guest memory: arming %#llx+%#llx failed", (unsigned long long)at, (unsigned long long)bytes);
        } else {
            if (!plat_decommit(at, bytes))
                rt_fatal("guest memory: releasing %#llx+%#llx failed", (unsigned long long)at, (unsigned long long)bytes);
            account(p->region, at, bytes, 0);
        }
        for (uint64_t k = i; k < j; k++) p->state[k] = to;
        __atomic_fetch_add(&managed_pages, j - i, __ATOMIC_ACQ_REL);
        n += j - i;
        i = j;
    }
    return n;
}

static int pages_arg_ok(const PageRange *p, uint64_t addr, uint64_t size) {
    return size && !((addr | size) & (PAGE - 1)) && inside(addr, size, p->base, p->size);
}

uint64_t rt_pages_arm(int id, uint64_t addr, uint64_t size) {
    uint64_t n = 0;
    plat_lock(&lock);
    if (pages_slot(id) && PR[id].mode == RT_PAGES_COUNTING && pages_arg_ok(&PR[id], addr, size)) {
        n = pages_change(&PR[id], addr, size, PG_ARMED);
        __atomic_fetch_add(&PR[id].st.armed, n, __ATOMIC_RELAXED);
        __atomic_fetch_add(&PR[id].st.arms, n, __ATOMIC_RELAXED);
    }
    plat_unlock(&lock);
    return n;
}

uint64_t rt_pages_release(int id, uint64_t addr, uint64_t size) {
    uint64_t n = 0;
    plat_lock(&lock);
    if (pages_slot(id) && PR[id].mode == RT_PAGES_RELEASING && pages_arg_ok(&PR[id], addr, size)) {
        PageRange *p = &PR[id];
        if (p->owner.busy && p->owner.busy(p->owner.user, addr, size)) {
            __atomic_fetch_add(&p->st.busy_refusals, 1, __ATOMIC_RELAXED);
        } else {
            n = pages_change(p, addr, size, PG_RELEASED);
            __atomic_fetch_add(&p->st.released, n, __ATOMIC_RELAXED);
            __atomic_fetch_add(&p->st.releases, n, __ATOMIC_RELAXED);
        }
    }
    plat_unlock(&lock);
    return n;
}

int rt_pages_stats(int id, RtPageStats *out) {
    int ok = 0;
    plat_lock_shared(&lock);
    if (pages_slot(id)) {
        memcpy(out, &PR[id].st, sizeof *out);
        ok = 1;
    }
    plat_unlock_shared(&lock);
    return ok;
}

// Shrink [*lo, *hi), which holds a, so that it leaves out [b, b+n), which does not.
static void clip_around(uint64_t a, uint64_t *lo, uint64_t *hi, uint64_t b, uint64_t n) {
    if (!overlaps(*lo, *hi - *lo, b, n)) return;
    if (b + n <= a) *lo = b + n;
    else *hi = b;
}

// What a first touch of a may commit: the pages around a, inside its 64 KiB chunk, that the guest declared mapped and
// that an access of this kind may use, and their protection; or why there are none. Under lock, shared at least.
typedef struct Touch { int region, prot; uint64_t lo, hi; const char *why; } Touch;

static Touch classify(uint64_t a, int write) {
    Touch t = {.region = region_of(a), .prot = PLAT_RW};
    uint64_t lo = a & ~(RT_COMMIT_CHUNK - 1), hi = lo + RT_COMMIT_CHUNK, mlo = 0, mhi = 0;
    for (unsigned i = 0; i < nmoves; i++) {
        if (a - moves[i].src < moves[i].size) {
            t.why = "MapMemory moved this memory to the stack region";
            return t;
        }
        if (a - moves[i].dst < moves[i].size) {
            mlo = moves[i].dst;
            mhi = mlo + moves[i].size;
        }
    }
    if (!mhi && a - mstack_base < mstack_size) {
        mlo = mstack_base;
        mhi = mstack_base + mstack_size;
    }
    const Region *g = t.region >= 0 ? &R[t.region] : NULL;
    if (mhi) {
        // a moved stack, or the main thread's
    } else if (t.region == RT_MEM_KERNEL) {
        // The kernel commits its pages when it allocates them, so a fault here is never a first touch.
        t.why = bit(kernel_map, (a - g->base) / PAGE) ? "the page is committed and still faults"
                                                      : "no thread-local page is allocated there";
        return t;
    } else if (t.region == RT_MEM_HEAP) {
        if (a - g->base >= heap_size) {
            t.why = "past the heap size SetHeapSize set";
            return t;
        }
        mlo = g->base;
        mhi = g->base + heap_size;
    } else if (t.region == RT_MEM_ALIAS) {
        uint64_t p = (a - g->base) / PAGE, first = (lo - g->base) / PAGE;
        if (!bit(alias_map, p)) {
            t.why = "an alias page MapPhysicalMemory has not mapped";
            return t;
        }
        mlo = g->base + run_start(alias_map, p, 1) * PAGE;
        mhi = g->base + run_end(alias_map, p, first + RT_COMMIT_CHUNK / PAGE, 1) * PAGE;
    } else if (t.region == RT_MEM_STACK) {
        t.why = "no stack was moved there with MapMemory";
        return t;
    } else if (t.region == RT_MEM_MODULES) {
        int i = module_at(a);
        if (i < 0) {
            t.why = "no module lies there";
            return t;
        }
        const RtLoaded *m = mods[i].m;
        uint64_t off = (a - m->base) & ~(PAGE - 1), lazy = align_up(m->bss_off, PAGE);
        unsigned perm = module_perm(i, off), p = perm & 7;
        if (p == PERM_NONE || (write && p != PERM_RW)) {
            t.why = off < m->data_off ? "module text and rodata are read-only" : "SetMemoryPermission protects it";
            return t;
        }
        if (off < lazy) {
            t.why = "the page is committed and still faults";
            return t;
        }
        uint64_t s = off, e = off + PAGE;
        while (s > lazy && module_perm(i, s - PAGE) == perm) s -= PAGE;
        while (e < m->size && module_perm(i, e) == perm) e += PAGE;
        mlo = m->base + s;
        mhi = m->base + e;
        t.prot = p == PERM_RW ? PLAT_RW : PLAT_R;
    } else {
        t.why = "no guest memory is mapped there";
        return t;
    }
    t.lo = mlo > lo ? mlo : lo;
    t.hi = mhi < hi ? mhi : hi;
    for (unsigned i = 0; i < nmoves; i++) clip_around(a, &t.lo, &t.hi, moves[i].src, moves[i].size);
    return t;
}

static int on_fault(uint64_t a, int write, uint64_t host_pc) {
    if (a >= RT_ADDRESS_LIMIT || !__atomic_load_n(&g_ready, __ATOMIC_ACQUIRE)) return 0;
    if (region_of(a) < 0) {
        if (a - RT_HOSTFN_BASE < RT_HOSTFN_SIZE) touch_fatal(a, write, host_pc, "host functions have no memory");
        if (!plat_is_free(a)) return 0;   // the host's own memory: its fault is not the guest's to name
        touch_fatal(a, write, host_pc, "no guest memory is mapped there");
    }
    PageHit hit;
    PageAccess pa = page_access(a, write, 0, host_pc, &hit);
    if (pa == PA_HANDLED) return 1;
    if (pa == PA_READ_RELEASED) released_fatal(a, write, host_pc, &hit);
    // A touch that faults again after its page was committed would loop forever.
    touch_repeats = last_touch == a ? touch_repeats + 1 : 0;
    last_touch = a;
    if (touch_repeats >= TOUCH_REPEAT_LIMIT) touch_fatal(a, write, host_pc, "it faults again after its commit");
    plat_lock_shared(&lock);
    Touch t = classify(a, write);
    if (!t.why) {
        commit(t.region, t.lo, t.hi - t.lo, t.prot);
        __atomic_fetch_add(&R[t.region].touch_commits, 1, __ATOMIC_RELAXED);
    }
    plat_unlock_shared(&lock);
    if (t.why) touch_fatal(a, write, host_pc, t.why);
    return 1;
}

// ---- 0x01 SetHeapSize
static uint32_t set_heap_size(uint64_t size) {
    if (size % 0x200000) return R_INVALID_SIZE;
    if (size > R[RT_MEM_HEAP].size) return R_OUT_OF_MEMORY;
    uint64_t base = R[RT_MEM_HEAP].base;
    uint32_t r = R_OK;
    plat_lock(&lock);
    if (size > heap_size && pool_used() - heap_size + size > C.pool_size) r = R_LIMIT_REACHED;
    for (unsigned i = 0; !r && size < heap_size && i < nmoves; i++)
        if (overlaps(moves[i].src, moves[i].size, base + size, heap_size - size)) r = R_INVALID_CURRENT_MEMORY;
    if (!r && size < heap_size) decommit(RT_MEM_HEAP, base + size, heap_size - size);
    if (!r) heap_size = size;
    plat_unlock(&lock);
    rt_log("SetHeapSize(%#llx): %#x, heap at %#llx", (unsigned long long)size, r, (unsigned long long)base);
    return r;
}

static void svc_set_heap_size(Cpu *c, uint64_t pc) {
    (void)pc;
    uint32_t r = set_heap_size(c->x[1]);
    c->x[0] = r;
    c->x[1] = r ? 0 : R[RT_MEM_HEAP].base;
}

// ---- 0x02 SetMemoryPermission, 0x03 SetMemoryAttribute: module data (RELRO) and the uncached attribute
// The module whose data or bss holds [a, a+n), or -1. Under lock.
static int module_data(uint64_t a, uint64_t n) {
    int i = module_at(a);
    if (i < 0) return -1;
    const RtLoaded *m = mods[i].m;
    uint64_t off = a - m->base;
    return off >= m->data_off && n <= m->size - off ? i : -1;
}

static uint32_t set_permission(uint64_t a, uint64_t n, uint32_t perm) {
    if (a & (PAGE - 1)) return R_INVALID_ADDRESS;
    if (!n || (n & (PAGE - 1))) return R_INVALID_SIZE;
    if (perm != PERM_NONE && perm != PERM_R && perm != PERM_RW) return R_INVALID_NEW_PERMISSION;
    plat_lock(&lock);
    int i = module_data(a, n);
    uint32_t r = i < 0 ? R_INVALID_CURRENT_MEMORY : R_OK;
    uint8_t *t = i < 0 ? NULL : mods[i].perm + (a - mods[i].m->base - mods[i].m->data_off) / PAGE;
    for (uint64_t p = 0; !r && p < n / PAGE; p++)
        if (t[p] & ATTR_PERMISSION_LOCKED) r = R_INVALID_CURRENT_MEMORY;
    if (!r) {
        // bss pages not touched yet are committed now, so that they can carry the protection
        uint64_t lazy = mods[i].m->base + align_up(mods[i].m->bss_off, PAGE), from = a > lazy ? a : lazy;
        if (from < a + n) commit(RT_MEM_MODULES, from, a + n - from, PLAT_RW);
        if (!plat_protect(a, n, perm == PERM_RW ? PLAT_RW : perm == PERM_R ? PLAT_R : PLAT_NONE))
            rt_fatal("guest memory: protecting %#llx+%#llx failed", (unsigned long long)a, (unsigned long long)n);
        memset(t, (int)perm, n / PAGE);
    }
    plat_unlock(&lock);
    return r;
}

static void svc_set_memory_permission(Cpu *c, uint64_t pc) {
    (void)pc;
    c->x[0] = set_permission(c->x[0], c->x[1], (uint32_t)c->x[2]);
}

static uint32_t set_attribute(uint64_t a, uint64_t n, uint32_t mask, uint32_t attr) {
    const uint32_t settable = ATTR_UNCACHED | ATTR_PERMISSION_LOCKED;
    if (a & (PAGE - 1)) return R_INVALID_ADDRESS;
    if (!n || (n & (PAGE - 1))) return R_INVALID_SIZE;
    if ((mask & ~settable) || (attr & ~mask)) return R_INVALID_COMBINATION;
    uint32_t r = R_OK;
    plat_lock(&lock);
    if (mask & ATTR_PERMISSION_LOCKED) {
        // The SDK locks a module's RELRO pages after making them read-only; the lock never comes off.
        int i = module_data(a, n);
        if (i < 0 || !(attr & ATTR_PERMISSION_LOCKED)) r = R_INVALID_CURRENT_MEMORY;
        uint8_t *t = i < 0 ? NULL : mods[i].perm + (a - mods[i].m->base - mods[i].m->data_off) / PAGE;
        for (uint64_t p = 0; !r && p < n / PAGE; p++) t[p] |= ATTR_PERMISSION_LOCKED;
    }
    if (!r && (mask & ATTR_UNCACHED)) {
        // Cache attributes mean nothing to the host; the range must still be normal memory the guest mapped.
        int g = region_of(a);
        int ok = (g == RT_MEM_HEAP && inside(a, n, R[g].base, heap_size)) || (g == RT_MEM_ALIAS && inside(a, n, R[g].base, R[g].size));
        if (!ok) r = R_INVALID_CURRENT_MEMORY;
    }
    plat_unlock(&lock);
    return r;
}

static void svc_set_memory_attribute(Cpu *c, uint64_t pc) {
    (void)pc;
    c->x[0] = set_attribute(c->x[0], c->x[1], (uint32_t)c->x[2], (uint32_t)c->x[3]);
}

// ---- 0x04 MapMemory, 0x05 UnmapMemory: thread stacks move from the heap or the alias region into the stack region.
// The host cannot alias the pages, so they move by copy: only the committed pages, and the side they left is
// decommitted, so a moved stack costs its touched pages once. A touch of the source while moved is a named fatal.
static int alias_all_mapped(uint64_t a, uint64_t n) {
    uint64_t first = (a - R[RT_MEM_ALIAS].base) / PAGE;
    return run_end(alias_map, first, first + n / PAGE, 1) == first + n / PAGE;
}

static void copy_committed(int from_r, uint64_t from, int to_r, uint64_t to, uint64_t n) {
    uint64_t first = (from - R[from_r].base) / PAGE, pages = n / PAGE;
    for (uint64_t i = 0; i < pages;) {
        if (!bit(R[from_r].bits, first + i)) { i++; continue; }
        uint64_t j = run_end(R[from_r].bits, first + i, first + pages, 1) - first;
        commit(to_r, to + i * PAGE, (j - i) * PAGE, PLAT_RW);
        memcpy((void *)(uintptr_t)(to + i * PAGE), (const void *)(uintptr_t)(from + i * PAGE), (j - i) * PAGE);
        i = j;
    }
}

static uint32_t move_stack(int unmap, uint64_t dst, uint64_t src, uint64_t n) {
    if (!n || ((dst | src | n) & (PAGE - 1))) return R_INVALID_ADDRESS;
    if (!inside(dst, n, R[RT_MEM_STACK].base, R[RT_MEM_STACK].size)) return R_INVALID_MEMORY_REGION;
    int sr = region_of(src);
    if ((sr != RT_MEM_HEAP && sr != RT_MEM_ALIAS) || !inside(src, n, R[sr].base, R[sr].size)) return R_INVALID_MEMORY_REGION;
    uint32_t r = R_OK;
    plat_lock(&lock);
    unsigned found = nmoves;
    for (unsigned i = 0; i < nmoves; i++)
        if (moves[i].dst == dst && moves[i].src == src && moves[i].size == n) found = i;
    if (unmap) {
        if (found == nmoves) {
            r = R_INVALID_CURRENT_MEMORY;
        } else {
            pages_reset(src, n);
            copy_committed(RT_MEM_STACK, dst, sr, src, n);
            decommit(RT_MEM_STACK, dst, n);
            moves[found] = moves[--nmoves];
        }
    } else {
        int mapped = sr == RT_MEM_HEAP ? inside(src, n, R[sr].base, heap_size) : alias_all_mapped(src, n);
        if (!mapped || nmoves == MAX_MOVES || overlaps(dst, n, mstack_base, mstack_size)) r = R_INVALID_CURRENT_MEMORY;
        for (unsigned i = 0; !r && i < nmoves; i++)
            if (overlaps(dst, n, moves[i].dst, moves[i].size) || overlaps(src, n, moves[i].src, moves[i].size))
                r = R_INVALID_CURRENT_MEMORY;
        if (!r) {
            pages_reset(src, n);
            copy_committed(sr, src, RT_MEM_STACK, dst, n);
            decommit(sr, src, n);
            moves[nmoves++] = (Move){dst, src, n};
        }
    }
    plat_unlock(&lock);
    return r;
}

static void svc_map_memory(Cpu *c, uint64_t pc) {
    (void)pc;
    c->x[0] = move_stack(0, c->x[0], c->x[1], c->x[2]);
}

static void svc_unmap_memory(Cpu *c, uint64_t pc) {
    (void)pc;
    c->x[0] = move_stack(1, c->x[0], c->x[1], c->x[2]);
}

// ---- 0x2C MapPhysicalMemory, 0x2D UnmapPhysicalMemory. Horizon maps only the pages of the range not mapped yet and
// leaves mapped pages and their contents alone. Mapping commits nothing; unmapping decommits.
static uint32_t alias_map_range(int unmap, uint64_t a, uint64_t n) {
    const Region *g = &R[RT_MEM_ALIAS];
    if (!n || ((a | n) & (PAGE - 1))) return R_INVALID_ADDRESS;
    if (!inside(a, n, g->base, g->size)) return R_INVALID_MEMORY_REGION;
    uint64_t first = (a - g->base) / PAGE, pages = n / PAGE, changed = 0;
    uint32_t r = R_OK;
    plat_lock(&lock);
    if (unmap) {
        for (unsigned i = 0; i < nmoves; i++)
            if (overlaps(a, n, moves[i].src, moves[i].size)) r = R_INVALID_CURRENT_MEMORY;
        for (uint64_t i = 0; !r && i < pages;) {
            uint64_t j = run_end(alias_map, first + i, first + pages, bit(alias_map, first + i)) - first;
            if (bit(alias_map, first + i)) {
                decommit(RT_MEM_ALIAS, a + i * PAGE, (j - i) * PAGE);
                for (uint64_t p = i; p < j; p++) bit_change(alias_map, first + p, 0);
                changed += j - i;
            }
            i = j;
        }
        alias_bytes -= changed * PAGE;
    } else {
        uint64_t fresh = 0;
        for (uint64_t i = 0; i < pages; i++) fresh += (uint64_t)!bit(alias_map, first + i);
        if (pool_used() + fresh * PAGE > C.pool_size) r = R_LIMIT_REACHED;
        for (uint64_t i = 0; !r && i < pages; i++) bit_change(alias_map, first + i, 1);
        if (!r) alias_bytes += fresh * PAGE;
    }
    plat_unlock(&lock);
    return r;
}

static void svc_map_physical(Cpu *c, uint64_t pc) {
    (void)pc;
    c->x[0] = alias_map_range(0, c->x[0], c->x[1]);
}

static void svc_unmap_physical(Cpu *c, uint64_t pc) {
    (void)pc;
    c->x[0] = alias_map_range(1, c->x[0], c->x[1]);
}

// ---- 0x06 QueryMemory
// [lo, hi) between the known ranges below and above a, which none of them holds.
static void gap_around(uint64_t a, const uint64_t (*known)[2], int n, uint64_t *lo, uint64_t *hi) {
    for (int i = 0; i < n; i++) {
        uint64_t b = known[i][0], e = known[i][0] + known[i][1];
        if (e <= a && e > *lo) *lo = e;
        if (b > a && b < *hi) *hi = b;
    }
}

static RtMemInfo query_locked(uint64_t a) {
    if (a >= RT_ADDRESS_LIMIT) return (RtMemInfo){RT_ADDRESS_LIMIT, 0 - RT_ADDRESS_LIMIT, ST_INACCESSIBLE, 0, 0};
    if (a < ASLR_BASE) return (RtMemInfo){0, ASLR_BASE, ST_INACCESSIBLE, 0, 0};
    for (unsigned i = 0; i < nmoves; i++) {
        if (a - moves[i].dst < moves[i].size) return (RtMemInfo){moves[i].dst, moves[i].size, ST_STACK, 0, PERM_RW};
        if (a - moves[i].src < moves[i].size)
            return (RtMemInfo){moves[i].src, moves[i].size, ST_NORMAL, ATTR_LOCKED, PERM_NONE};
    }
    if (a - mstack_base < mstack_size) return (RtMemInfo){mstack_base, mstack_size, ST_STACK, 0, PERM_RW};
    int r = region_of(a);
    const Region *g = r >= 0 ? &R[r] : NULL;
    RtMemInfo q = {0};
    if (r == RT_MEM_KERNEL) {
        // Allocated pages are thread-local regions; the rest is the kernel's and reads as inaccessible.
        uint64_t p = (a - g->base) / PAGE;
        int v = bit(kernel_map, p);
        uint64_t s = run_start(kernel_map, p, v), e = run_end(kernel_map, p, g->size / PAGE, v);
        return (RtMemInfo){g->base + s * PAGE, (e - s) * PAGE, v ? ST_THREAD_LOCAL : ST_INACCESSIBLE, 0,
                           v ? PERM_RW : PERM_NONE};
    } else if (r == RT_MEM_STACK) {
        q = (RtMemInfo){g->base, g->size, ST_FREE, 0, PERM_NONE};
    } else if (r == RT_MEM_HEAP) {
        q = a - g->base < heap_size ? (RtMemInfo){g->base, heap_size, ST_NORMAL, 0, PERM_RW}
                                    : (RtMemInfo){g->base + heap_size, g->size - heap_size, ST_FREE, 0, PERM_NONE};
    } else if (r == RT_MEM_ALIAS) {
        uint64_t p = (a - g->base) / PAGE;
        int v = bit(alias_map, p);
        uint64_t s = run_start(alias_map, p, v), e = run_end(alias_map, p, g->size / PAGE, v);
        q = (RtMemInfo){g->base + s * PAGE, (e - s) * PAGE, v ? ST_NORMAL : ST_FREE, 0, v ? PERM_RW : PERM_NONE};
    } else if (r == RT_MEM_MODULES && module_at(a) >= 0) {
        int i = module_at(a);
        const RtLoaded *m = mods[i].m;
        uint64_t off = (a - m->base) & ~(PAGE - 1), lo = off, hi = off + PAGE;
        unsigned perm = module_perm(i, off);
        int data = off >= m->data_off;
        uint64_t limit = data ? m->data_off : 0, end = data ? m->size : m->data_off;
        while (lo > limit && module_perm(i, lo - PAGE) == perm) lo -= PAGE;
        while (hi < end && module_perm(i, hi) == perm) hi += PAGE;
        return (RtMemInfo){m->base + lo, hi - lo, data ? ST_CODE_DATA : ST_CODE, perm & ATTR_PERMISSION_LOCKED, perm & 7};
    } else if (r == RT_MEM_MODULES) {
        uint64_t known[RT_MAX_LOADED][2], lo = g->base, hi = g->base + g->size;
        for (int i = 0; i < nmods; i++) known[i][0] = mods[i].m->base, known[i][1] = mods[i].m->size;
        gap_around(a, (const uint64_t(*)[2])known, nmods, &lo, &hi);
        return (RtMemInfo){lo, hi - lo, ST_INACCESSIBLE, 0, PERM_NONE};
    } else {
        // The host's: the host-function window, or what lies between the guest's regions.
        uint64_t known[RT_MEM_REGIONS + 1][2], lo = ASLR_BASE, hi = RT_ADDRESS_LIMIT;
        for (int i = 0; i < RT_MEM_REGIONS; i++) known[i][0] = R[i].base, known[i][1] = R[i].size;
        known[RT_MEM_REGIONS][0] = RT_HOSTFN_BASE, known[RT_MEM_REGIONS][1] = RT_HOSTFN_SIZE;
        if (a - RT_HOSTFN_BASE < RT_HOSTFN_SIZE) return (RtMemInfo){RT_HOSTFN_BASE, RT_HOSTFN_SIZE, ST_INACCESSIBLE, 0, 0};
        gap_around(a, (const uint64_t(*)[2])known, RT_MEM_REGIONS + 1, &lo, &hi);
        return (RtMemInfo){lo, hi - lo, ST_INACCESSIBLE, 0, PERM_NONE};
    }
    // Moved stacks and the main thread's cut the free stack region and the heap and alias runs they lie in.
    uint64_t lo = q.base, hi = q.base + q.size;
    for (unsigned i = 0; i < nmoves; i++) {
        clip_around(a, &lo, &hi, moves[i].dst, moves[i].size);
        clip_around(a, &lo, &hi, moves[i].src, moves[i].size);
    }
    clip_around(a, &lo, &hi, mstack_base, mstack_size);
    q.base = lo;
    q.size = hi - lo;
    return q;
}

RtMemInfo rt_mem_query(uint64_t a) {
    if (!rt_mem_ready()) rt_fatal("rt_mem_query before rt_mem_init");
    plat_lock_shared(&lock);
    RtMemInfo q = query_locked(a);
    plat_unlock_shared(&lock);
    return q;
}

// QueryMemory(out MemoryInfo *x0, address x2): w0 result, w1 page info. Written after the lock is released: the
// output may lie in a page not touched yet.
static void svc_query_memory(Cpu *c, uint64_t pc) {
    (void)pc;
    uint64_t out = c->x[0];
    RtMemInfo q = rt_mem_query(c->x[2]);
    ST64(out, q.base);
    ST64(out + 8, q.size);
    ST32(out + 16, q.state);
    ST32(out + 20, q.attr);
    ST32(out + 24, q.perm);
    ST32(out + 28, 0);   // ipc reference count
    ST32(out + 32, 0);   // device reference count
    ST32(out + 36, 0);
    c->x[0] = R_OK;
    c->x[1] = 0;
}

// ---- start-up
RtMemConfig rt_mem_default_config(void) {
    return (RtMemConfig){.pool_size = 9 * GiB, .heap_region_size = 8 * GiB, .alias_region_size = 64 * GiB,
                         .stack_region_size = 2 * GiB, .system_resource_size = 0};
}

int rt_mem_ready(void) { return __atomic_load_n(&g_ready, __ATOMIC_ACQUIRE); }

// Stack, heap, alias and kernel regions one after the other from at; all or none.
static const int block_order[] = {RT_MEM_STACK, RT_MEM_HEAP, RT_MEM_ALIAS, RT_MEM_KERNEL};
#define BLOCK_REGIONS ((int)(sizeof block_order / sizeof *block_order))

static int reserve_block(uint64_t at) {
    int k = 0;
    for (; k < BLOCK_REGIONS; at += R[block_order[k]].size, k++) {
        const int r = block_order[k];
        if (!plat_reserve_at(at, R[r].size, R[r].watch)) break;
        R[r].base = at;
    }
    if (k == BLOCK_REGIONS) return 1;
    while (k-- > 0) plat_release(R[block_order[k]].base, R[block_order[k]].size);
    return 0;
}

void rt_mem_init(const RtMemConfig *cfg) {
    if (rt_mem_ready()) rt_fatal("rt_mem_init called twice");
    if (rt_sealed()) rt_fatal("rt_mem_init after rt_dispatch_init");
    const uint64_t g = 0x200000;
    if (!cfg->pool_size || !cfg->heap_region_size || !cfg->alias_region_size || !cfg->stack_region_size ||
        (cfg->pool_size | cfg->heap_region_size | cfg->alias_region_size | cfg->stack_region_size) % g)
        rt_fatal("rt_mem_init: pool %#llx, heap %#llx, alias %#llx, stack %#llx: each must be a non-zero multiple of "
                 "2 MiB", (unsigned long long)cfg->pool_size, (unsigned long long)cfg->heap_region_size,
                 (unsigned long long)cfg->alias_region_size, (unsigned long long)cfg->stack_region_size);
    C = *cfg;
    R[RT_MEM_STACK].size = C.stack_region_size;
    R[RT_MEM_HEAP].size = C.heap_region_size;
    R[RT_MEM_ALIAS].size = C.alias_region_size;
    R[RT_MEM_MODULES].base = RT_MODULE_BASE;
    R[RT_MEM_MODULES].size = RT_MODULE_LIMIT - RT_MODULE_BASE;
    R[RT_MEM_KERNEL].size = RT_KERNEL_REGION_SIZE;
    uint64_t block = C.stack_region_size + C.heap_region_size + C.alias_region_size + RT_KERNEL_REGION_SIZE;

    // Under high-entropy ASLR, Windows puts the main thread's stack anywhere below 1 TiB, inside this region in about
    // one start in thirty; linked without it (lld: --disable-high-entropy-va), the host's stacks and heaps stay low.
    if (!plat_reserve_at(R[RT_MEM_MODULES].base, R[RT_MEM_MODULES].size, 0))
        rt_fatal("guest memory: the module region %#llx+%#llx is taken by the host (is the program linked with "
                 "--disable-high-entropy-va?)", RT_MODULE_BASE, (unsigned long long)R[RT_MEM_MODULES].size);
    if (!plat_reserve_at(RT_HOSTFN_BASE, RT_HOSTFN_SIZE, 0))
        rt_fatal("guest memory: the host-function window %#llx is taken by the host", RT_HOSTFN_BASE);
    uint64_t at = RT_REGION_SEARCH_FROM;
    while (block <= RT_MODULE_BASE - at && !reserve_block(at)) at += SEARCH_STEP;
    if (block > RT_MODULE_BASE - at)
        rt_fatal("guest memory: no %#llx free bytes between %#llx and the modules for the stack, heap, alias and kernel "
                 "regions", (unsigned long long)block, RT_REGION_SEARCH_FROM);
    for (int r = 0; r < RT_MEM_REGIONS; r++) R[r].bits = new_bitmap(R[r].size / PAGE);
    alias_map = new_bitmap(R[RT_MEM_ALIAS].size / PAGE);
    kernel_map = new_bitmap(R[RT_MEM_KERNEL].size / PAGE);

    rt_svc_register(0x01, svc_set_heap_size);
    rt_svc_register(0x02, svc_set_memory_permission);
    rt_svc_register(0x03, svc_set_memory_attribute);
    rt_svc_register(0x04, svc_map_memory);
    rt_svc_register(0x05, svc_unmap_memory);
    rt_svc_register(0x06, svc_query_memory);
    rt_svc_register(0x2C, svc_map_physical);
    rt_svc_register(0x2D, svc_unmap_physical);
    // Opt-in, in the manner of the census: the setting page_release (OOT2_PAGE_RELEASE) at count allows counting
    // ranges, at release both modes. Its default, off, leaves page ranges refused and first-touch commit as it was; a
    // value the setting does not take stopped the start (oot2/settings.h). The table's names are in the order of
    // RT_PAGES_OFF, RT_PAGES_COUNTING and RT_PAGES_RELEASING.
    g_pages_level = (int)rt_setting(RT_SET_page_release);
    if (g_pages_level) {
        const char *from;
        rt_setting_source(RT_SET_page_release, &from);
        rt_log("page ranges: %s allowed (%s)", g_pages_level == RT_PAGES_RELEASING ? "release" : "count", from);
    }
    plat_fault_hook(on_fault);
    __atomic_store_n(&g_ready, 1, __ATOMIC_RELEASE);
    rt_log("guest memory: stack %#llx+%#llx, heap %#llx+%#llx, alias %#llx+%#llx, kernel %#llx+%#llx, modules "
           "%#llx+%#llx, pool %#llx",
           (unsigned long long)R[RT_MEM_STACK].base, (unsigned long long)R[RT_MEM_STACK].size,
           (unsigned long long)R[RT_MEM_HEAP].base, (unsigned long long)R[RT_MEM_HEAP].size,
           (unsigned long long)R[RT_MEM_ALIAS].base, (unsigned long long)R[RT_MEM_ALIAS].size,
           (unsigned long long)R[RT_MEM_KERNEL].base, (unsigned long long)R[RT_MEM_KERNEL].size, RT_MODULE_BASE,
           (unsigned long long)R[RT_MEM_MODULES].size, (unsigned long long)C.pool_size);
}

// ---- host I/O, GetInfo, write watch, measurement
int rt_mem_commit_io(uint64_t addr, uint64_t size, int write) {
    if (!rt_mem_ready()) rt_fatal("rt_mem_commit_io before rt_mem_init");
    if (!size) return 1;
    if (addr >= RT_ADDRESS_LIMIT || size > RT_ADDRESS_LIMIT - addr) return 0;
    int ok = 1;
    // Armed and released pages first (lever L1): host I/O is a host access, a write recommits, a read of a released
    // page fails. The owner's callback may run here, so no lock is held yet.
    if (__atomic_load_n(&managed_pages, __ATOMIC_ACQUIRE) && pages_overlap(addr, size)) {
        PageHit hit;
        for (uint64_t a = addr & ~(PAGE - 1); a < addr + size; a += PAGE)
            if (page_access(a, write, 1, 0, &hit) == PA_READ_RELEASED) return 0;
    }
    plat_lock_shared(&lock);
    for (uint64_t a = addr & ~(PAGE - 1); ok && a < addr + size;) {
        int r = region_of(a), mi = module_at(a);
        if (r >= 0 && bit(R[r].bits, (a - R[r].base) / PAGE)) {
            unsigned p = mi >= 0 ? module_perm(mi, a - mods[mi].m->base) & 7 : PERM_RW;
            ok = write ? p == PERM_RW : p != PERM_NONE;
            a += PAGE;
            continue;
        }
        Touch t = classify(a, write);
        if (t.why) ok = 0;
        else commit(t.region, t.lo, t.hi - t.lo, t.prot), a = t.hi;
    }
    plat_unlock_shared(&lock);
    return ok;
}

// The system resource is what the kernel spends on the process's page tables; estimated from the alias pages mapped.
static uint64_t system_resource_used(void) {
    uint64_t v = align_up(alias_bytes / 512, PAGE);
    return v < C.system_resource_size ? v : C.system_resource_size;
}

int rt_mem_info(uint32_t id, uint64_t *out) {
    if (!rt_mem_ready()) rt_fatal("rt_mem_info before rt_mem_init");
    uint64_t v;
    plat_lock_shared(&lock);
    switch (id) {
    case 2: v = R[RT_MEM_ALIAS].base; break;    // AliasRegionAddress
    case 3: v = R[RT_MEM_ALIAS].size; break;    // AliasRegionSize
    case 4: v = R[RT_MEM_HEAP].base; break;     // HeapRegionAddress
    case 5: v = R[RT_MEM_HEAP].size; break;     // HeapRegionSize
    case 6: case 21: v = C.pool_size; break;    // TotalMemorySize, TotalNonSystemMemorySize
    case 7: case 22: v = pool_used(); break;    // UsedMemorySize, UsedNonSystemMemorySize
    case 12: v = ASLR_BASE; break;              // AslrRegionAddress
    case 13: v = RT_ADDRESS_LIMIT - ASLR_BASE; break;
    case 14: v = R[RT_MEM_STACK].base; break;   // StackRegionAddress
    case 15: v = R[RT_MEM_STACK].size; break;
    case 16: v = C.system_resource_size; break; // TotalSystemResourceSize
    case 17: v = system_resource_used(); break;
    case 28: v = 0; break;                      // AliasRegionExtraSize
    default: plat_unlock_shared(&lock); return 0;
    }
    plat_unlock_shared(&lock);
    *out = v;
    return 1;
}

uint64_t rt_mem_written(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    for (int r = RT_MEM_HEAP; r <= RT_MEM_ALIAS; r++)
        if (inside(addr, size, R[r].base, R[r].size)) return plat_written(addr, size, pages, max, reset);
    return UINT64_MAX;
}

uint64_t rt_mem_commit_run(uint64_t addr, uint64_t size, int *committed) {
    if (!rt_mem_ready()) rt_fatal("rt_mem_commit_run before rt_mem_init");
    int r = region_of(addr);
    if (r < 0) return UINT64_MAX;
    if (!size) return 0;
    const Region *g = &R[r];
    uint64_t off = addr - g->base, end = size < g->size - off ? off + size : g->size;
    uint64_t first = off / PAGE, stop;
    int v = bit(g->bits, first);
    stop = run_end(g->bits, first, (end + PAGE - 1) / PAGE, v) * PAGE;
    *committed = v;
    return (stop < end ? stop : end) - off;
}

uint64_t rt_mem_commit_epoch(void) { return __atomic_load_n(&g_commit_epoch, __ATOMIC_ACQUIRE); }

uint64_t rt_mem_decommit_epoch(void) { return __atomic_load_n(&g_dec_epoch, __ATOMIC_ACQUIRE); }

int rt_mem_decommitted(uint64_t since, uint64_t lo, uint64_t hi) {
    uint64_t e = __atomic_load_n(&g_dec_epoch, __ATOMIC_ACQUIRE);
    if (e <= since) return 0;
    if (e - since >= DECOMMIT_LOG) return 1;   // the oldest of them are no longer recorded
    for (uint64_t k = since + 1; k <= e; k++)
        if (__atomic_load_n(&g_dec_lo[k % DECOMMIT_LOG], __ATOMIC_RELAXED) < hi &&
            lo < __atomic_load_n(&g_dec_hi[k % DECOMMIT_LOG], __ATOMIC_RELAXED))
            return 1;
    // A decommit newer than e writes the slot of the one DECOMMIT_LOG before it: a slot read here was its own unless a
    // decommit past since + DECOMMIT_LOG began (the writer's fence pairs with this one)
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return __atomic_load_n(&g_dec_start, __ATOMIC_RELAXED) - since >= DECOMMIT_LOG;
}

static uint64_t resident_bytes(const Region *g) {
    uint64_t pages = g->size / PAGE, total = 0;
    for (uint64_t i = 0; i < pages;) {
        if (!bit(g->bits, i)) { i = run_end(g->bits, i, pages, 0); continue; }
        uint64_t j = run_end(g->bits, i, pages, 1), n = plat_resident(g->base + i * PAGE, (j - i) * PAGE);
        if (n != UINT64_MAX) total += n;
        i = j;
    }
    return total;
}

void rt_mem_measure(RtMemStats *out) {
    if (!rt_mem_ready()) rt_fatal("rt_mem_measure before rt_mem_init");
    memset(out, 0, sizeof *out);
    plat_lock_shared(&lock);
    uint64_t moved = 0;
    for (unsigned i = 0; i < nmoves; i++) moved += moves[i].size;
    const uint64_t mapped[RT_MEM_REGIONS] = {moved + mstack_size, heap_size, alias_bytes, image_bytes(), kernel_bytes};
    for (int r = 0; r < RT_MEM_REGIONS; r++)
        out->r[r] = (RtMemRegion){R[r].name, R[r].base, R[r].size, mapped[r],
                                  __atomic_load_n(&R[r].committed, __ATOMIC_RELAXED), resident_bytes(&R[r]),
                                  __atomic_load_n(&R[r].touch_commits, __ATOMIC_RELAXED)};
    plat_unlock_shared(&lock);
    PlatProcMem p;
    plat_process_memory(&p);
    out->process_private = p.private_bytes;
    out->process_working_set = p.working_set;
    out->process_peak_working_set = p.peak_working_set;
}

// ---- for the loader
void mem_module_commit(uint64_t a, uint64_t n) {
    if (!rt_mem_ready()) rt_fatal("loader: modules loaded before rt_mem_init");
    if (!inside(a, n, R[RT_MEM_MODULES].base, R[RT_MEM_MODULES].size))
        rt_fatal("loader: %#llx+%#llx lies outside the module region", (unsigned long long)a, (unsigned long long)n);
    plat_lock(&lock);
    commit(RT_MEM_MODULES, a, n, PLAT_RW);
    plat_unlock(&lock);
}

void mem_module_seal(uint64_t a, uint64_t n) {
    if (n && !plat_protect(a, n, PLAT_R))
        rt_fatal("loader: sealing %#llx+%#llx read-only failed", (unsigned long long)a, (unsigned long long)n);
}

// ---- for the kernel
uint64_t mem_kernel_alloc(uint64_t size) {
    if (!rt_mem_ready()) rt_fatal("kernel: guest memory is not set up (rt_mem_init comes first)");
    const Region *g = &R[RT_MEM_KERNEL];
    uint64_t pages = align_up(size ? size : 1, PAGE) / PAGE, total = g->size / PAGE, at = 0, s = 0;
    plat_lock(&lock);
    for (;;) {
        s = run_end(kernel_map, at, total, 1);   // the next free page
        if (s + pages > total) {
            plat_unlock(&lock);
            rt_fatal("kernel: the kernel region (%#llx bytes) has no %llu free pages left", (unsigned long long)g->size,
                     (unsigned long long)pages);
        }
        uint64_t e = run_end(kernel_map, s, s + pages, 0);
        if (e == s + pages) break;
        at = e;
    }
    for (uint64_t i = 0; i < pages; i++) bit_change(kernel_map, s + i, 1);
    commit(RT_MEM_KERNEL, g->base + s * PAGE, pages * PAGE, PLAT_RW);
    kernel_bytes += pages * PAGE;
    plat_unlock(&lock);
    return g->base + s * PAGE;
}

void mem_kernel_free(uint64_t addr, uint64_t size) {
    const Region *g = &R[RT_MEM_KERNEL];
    uint64_t pages = align_up(size ? size : 1, PAGE) / PAGE, first = (addr - g->base) / PAGE;
    plat_lock(&lock);
    if ((addr & (PAGE - 1)) || !inside(addr, pages * PAGE, g->base, g->size) ||
        run_end(kernel_map, first, first + pages, 1) != first + pages) {
        plat_unlock(&lock);
        rt_fatal("kernel: freeing %#llx+%#llx, which the kernel did not allocate", (unsigned long long)addr,
                 (unsigned long long)size);
    }
    decommit(RT_MEM_KERNEL, addr, pages * PAGE);
    for (uint64_t i = 0; i < pages; i++) bit_change(kernel_map, first + i, 0);
    kernel_bytes -= pages * PAGE;
    plat_unlock(&lock);
}

uint64_t mem_main_stack(uint64_t size) {
    if (!rt_mem_ready()) rt_fatal("kernel: guest memory is not set up (rt_mem_init comes first)");
    const Region *g = &R[RT_MEM_STACK];
    size = align_up(size, PAGE);
    plat_lock(&lock);
    const char *why = mstack_size ? "it is mapped already"
                    : !size || size > g->size ? "it does not fit the stack region"
                    : pool_used() + size > C.pool_size ? "the pool is full" : NULL;
    for (unsigned i = 0; !why && i < nmoves; i++)
        if (overlaps(g->base + g->size - size, size, moves[i].dst, moves[i].size)) why = "a moved stack lies there";
    if (!why) {
        mstack_base = g->base + g->size - size;
        mstack_size = size;
    }
    plat_unlock(&lock);
    if (why)
        rt_fatal("kernel: the main thread's stack of %#llx bytes cannot be mapped: %s", (unsigned long long)size, why);
    return g->base + g->size;
}

void mem_module_add(const RtLoaded *m) {
    if (nmods == RT_MAX_LOADED) rt_fatal("loader: more than %d modules", RT_MAX_LOADED);
    uint64_t pages = (m->size - m->data_off) / PAGE;
    uint8_t *perm = malloc(pages ? pages : 1);
    if (!perm) rt_fatal("loader: module permission table");
    memset(perm, PERM_RW, pages);
    plat_lock(&lock);
    mods[nmods++] = (Mod){m, perm};
    plat_unlock(&lock);
}
