// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Indirect dispatch, guest address to lifted function, in O(1) (architecture D8).
// Adapted from DKB Recomp runtime/dispatch.c at 9d8fa02 (GPL-3.0-or-later): the flat table, its anchor and its
// checks. Changed here: the safe-point check reads c->pending, the host-function window is an array in hostfn.c
// rather than entries of the hash, and the module limit of the fixed address map is enforced.
//
// Layout of the table: one uint32_t slot per 4-byte guest instruction over [lo, hi), where lo is the lowest function
// start of any module and hi is past the highest. A slot is 0 when no function starts there, else the host function's
// address minus `anchor` (an address just below the lowest lifted function), so 4 bytes suffice while all lifted
// code lies within 4 GiB, which one program image guarantees. The table costs 4 bytes per 4 guest bytes of the range
// in address space and commit charge; pages that no function start reaches are never written, so they take no RAM.
//
// The table is filled once by rt_dispatch_init and then made read-only, so any number of threads read it without
// locks. Host functions at fixed addresses outside the modules live in a small hash that is also fixed at init.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "oot2/memline.h"
#include "plat.h"

#define TABLE_SPAN_LIMIT (1ull << 30)   // guest bytes covered by the table, and its size in bytes
#define HOST_CAP 1024                   // host functions at fixed addresses, power of two
#define MAX_MODULES 16

// Hot data read by every rt_dispatch: one cache line.
static struct __attribute__((aligned(64))) {
    const uint32_t *slot;
    uintptr_t anchor;
    uint64_t lo;
    uint64_t slots;
} D;

static const RtModule *g_mods[MAX_MODULES];   // sorted by base
static int g_nmods;
static int g_ready;
static uint64_t g_table_bytes, g_functions;
// Host range of each module's lifted code: from its lowest function to its end marker (the lifted-code end marker).
typedef struct CodeRange { uintptr_t lo, end; const RtModule *m; } CodeRange;
static CodeRange g_range[MAX_MODULES];
static int g_nranges;
static void (*g_safepoint_hook)(Cpu *c);

static struct { uint64_t addr; GuestFn fn; } g_host[HOST_CAP];
static uint64_t g_nhost;

int rt_sealed(void) { return __atomic_load_n(&g_ready, __ATOMIC_ACQUIRE); }

static const RtModule *module_at(uint64_t pc) {
    for (int i = 0; i < g_nmods; i++)
        if (pc - g_mods[i]->base < g_mods[i]->size) return g_mods[i];
    return NULL;
}

const char *rt_guest_name(uint64_t pc, char *buf, int size) {
    const RtModule *m = module_at(pc);
    const char *host = m ? NULL : rt_hostfn_name(pc);
    if (m) snprintf(buf, size, "%#llx (%s+%#llx)", (unsigned long long)pc, m->name, (unsigned long long)(pc - m->base));
    else if (host) snprintf(buf, size, "%#llx (host %s)", (unsigned long long)pc, host);
    else snprintf(buf, size, "%#llx", (unsigned long long)pc);
    return buf;
}

static uint64_t host_hash(uint64_t addr) { return (addr >> 2) * 0x9E3779B97F4A7C15ull >> 54; }   // 10 bits

static GuestFn host_lookup(uint64_t addr) {
    for (uint64_t h = host_hash(addr), n = 0; n < HOST_CAP; h = (h + 1) & (HOST_CAP - 1), n++) {
        if (g_host[h].addr == addr) return g_host[h].fn;
        if (!g_host[h].addr) return NULL;
    }
    return NULL;
}

uint64_t rt_module_base(const char *name) {
    for (int i = 0; i < g_nmods; i++)
        if (!strcmp(g_mods[i]->name, name)) return g_mods[i]->base;
    return 0;
}

void rt_register_host(uint64_t addr, GuestFn fn) {
    if (rt_sealed()) rt_fatal("rt_register_host(%#llx) after rt_dispatch_init", (unsigned long long)addr);
    if (!addr || (addr & 3) || !fn) rt_fatal("rt_register_host: bad entry %#llx", (unsigned long long)addr);
    if (addr - RT_HOSTFN_BASE < RT_HOSTFN_SIZE)
        rt_fatal("rt_register_host: %#llx lies in the host-function window", (unsigned long long)addr);
    if (g_nhost >= HOST_CAP / 2) rt_fatal("rt_register_host: more than %d host functions", HOST_CAP / 2);
    uint64_t h = host_hash(addr);
    while (g_host[h].addr && g_host[h].addr != addr) h = (h + 1) & (HOST_CAP - 1);
    if (g_host[h].addr) rt_fatal("rt_register_host: %#llx registered twice", (unsigned long long)addr);
    g_host[h].addr = addr;
    g_host[h].fn = fn;
    g_nhost++;
}

void rt_set_safepoint_hook(void (*hook)(Cpu *c)) {
    if (rt_sealed()) rt_fatal("rt_set_safepoint_hook after rt_dispatch_init");
    g_safepoint_hook = hook;
}

void rt_safepoint(Cpu *c) {
    if (g_safepoint_hook) g_safepoint_hook(c);
    else __atomic_store_n(&c->pending, 0, __ATOMIC_RELAXED);
}

void rt_dispatch_init(const RtModule *mods, int count) {
    if (rt_sealed()) rt_fatal("rt_dispatch_init called twice");
    if (count <= 0 || count > MAX_MODULES) rt_fatal("rt_dispatch_init: %d modules", count);

    // Modules by base, inside the module range, without overlap.
    for (int i = 0; i < count; i++) {
        if (!mods[i].name || !mods[i].size || mods[i].base + mods[i].size > RT_MODULE_LIMIT ||
            mods[i].base + mods[i].size < mods[i].base)
            rt_fatal("dispatch: module %s at %#llx, size %#llx, leaves the module range below %#llx",
                     mods[i].name ? mods[i].name : "(unnamed)", (unsigned long long)mods[i].base,
                     (unsigned long long)mods[i].size, RT_MODULE_LIMIT);
        int j = g_nmods++;
        while (j > 0 && g_mods[j - 1]->base > mods[i].base) { g_mods[j] = g_mods[j - 1]; j--; }
        g_mods[j] = &mods[i];
    }
    for (int i = 1; i < g_nmods; i++)
        if (g_mods[i - 1]->base + g_mods[i - 1]->size > g_mods[i]->base)
            rt_fatal("dispatch: modules %s and %s overlap", g_mods[i - 1]->name, g_mods[i]->name);

    // Function tables: sorted, aligned, inside their module. Range of guest starts and of host functions.
    uint64_t lo = UINT64_MAX, hi = 0;
    uintptr_t host_lo = UINTPTR_MAX, host_hi = 0;
    for (int i = 0; i < g_nmods; i++) {
        const RtModule *m = g_mods[i];
        if (!m->code_end)
            rt_fatal("dispatch: module %s has no end marker (%s_lifted_end): the lifter emits it into the module's "
                     "function table and the link must place it after the module's lifted code", m->name, m->name);
        uintptr_t mlo = UINTPTR_MAX;
        for (uint64_t k = 0; k < m->count; k++) {
            uint64_t a = m->addrs[k];
            uintptr_t f = (uintptr_t)m->fns[k];
            if ((a & 3) || a - m->base >= m->size)
                rt_fatal("dispatch: %s_fn_addrs[%llu] = %#llx is misaligned or outside the module", m->name,
                         (unsigned long long)k, (unsigned long long)a);
            if (k && a <= m->addrs[k - 1])
                rt_fatal("dispatch: %s_fn_addrs is not strictly ascending at [%llu] = %#llx", m->name,
                         (unsigned long long)k, (unsigned long long)a);
            if (!f) rt_fatal("dispatch: %s_fn_ptrs[%llu] is NULL", m->name, (unsigned long long)k);
            if (f >= (uintptr_t)m->code_end)
                rt_fatal("dispatch: %s_fn_ptrs[%llu] at %p lies at or past the module's end marker %s_lifted_end at %p: "
                         "the link order moved the function table before lifted code", m->name,
                         (unsigned long long)k, (void *)f, m->name, m->code_end);
            if (f < mlo) mlo = f;
            if (f < host_lo) host_lo = f;
            if (f > host_hi) host_hi = f;
        }
        if (m->count) {
            g_range[g_nranges++] = (CodeRange){mlo, (uintptr_t)m->code_end, m};
            if (m->addrs[0] < lo) lo = m->addrs[0];
            if (m->addrs[m->count - 1] + 4 > hi) hi = m->addrs[m->count - 1] + 4;
        }
    }
    if (hi <= lo) rt_fatal("dispatch: no functions in any module");
    if (hi - lo > TABLE_SPAN_LIMIT)
        rt_fatal("dispatch: modules span %#llx guest bytes, over the %#llx the flat table allows",
                 (unsigned long long)(hi - lo), TABLE_SPAN_LIMIT);
    uintptr_t anchor = (host_lo & ~(uintptr_t)15) - 16;
    if (host_hi - anchor > UINT32_MAX)
        rt_fatal("dispatch: lifted code spans %#llx host bytes; slots hold 32-bit offsets (4 GiB)",
                 (unsigned long long)(host_hi - anchor));

    for (uint64_t h = 0; h < HOST_CAP; h++)
        if (g_host[h].addr && module_at(g_host[h].addr))
            rt_fatal("dispatch: host function at %#llx lies inside module %s", (unsigned long long)g_host[h].addr,
                     module_at(g_host[h].addr)->name);

    uint64_t slots = (hi - lo) >> 2, bytes = (slots * 4 + 0xFFFF) & ~0xFFFFull;
    uint32_t *slot = plat_alloc_zeroed(bytes);
    if (!slot) rt_fatal("dispatch: cannot allocate a %llu-byte table", (unsigned long long)bytes);
    memline_note("dispatch-table", bytes);
    for (int i = 0; i < g_nmods; i++) {
        const RtModule *m = g_mods[i];
        for (uint64_t k = 0; k < m->count; k++)
            slot[(m->addrs[k] - lo) >> 2] = (uint32_t)((uintptr_t)m->fns[k] - anchor);
        g_functions += m->count;
    }
    if (!plat_protect_read_only(slot, bytes)) rt_fatal("dispatch: sealing the table failed");

    hostfn_seal();
    svc_seal();
    g_table_bytes = bytes;
    D.slot = slot;
    D.anchor = anchor;
    D.lo = lo;
    D.slots = slots;
    __atomic_store_n(&g_ready, 1, __ATOMIC_RELEASE);
}

static GuestFn slow_lookup(uint64_t pc) {
    if (pc - RT_HOSTFN_BASE < RT_HOSTFN_SIZE) return hostfn_lookup(pc);
    return host_lookup(pc);
}

GuestFn rt_dispatch_lookup(uint64_t pc) {
    if (!rt_sealed()) return NULL;
    uint64_t i = __builtin_rotateright64(pc - D.lo, 2);   // a misaligned pc lands far above D.slots
    if (i < D.slots && D.slot[i]) return (GuestFn)(D.anchor + D.slot[i]);
    return slow_lookup(pc);
}

__attribute__((noinline, noreturn, cold)) static void dispatch_fail(Cpu *c) {
    char at[96], lr[96];
    uint64_t pc = c->pc;
    if (!rt_sealed()) rt_fatal("rt_dispatch to %#llx before rt_dispatch_init", (unsigned long long)pc);
    const char *why = (pc & 3) ? "misaligned address"
                    : module_at(pc) ? "not a function start"
                    : pc - RT_HOSTFN_BASE < RT_HOSTFN_SIZE ? "no host function registered there"
                    : "outside every module, no host function registered";
    rt_fatal("rt_dispatch: no function at guest pc %s: %s (lr %s)", rt_guest_name(pc, at, sizeof at), why,
             rt_guest_name(c->x[30], lr, sizeof lr));
}

// Host functions, and the fatal error. Out of line, so the fast path below keeps no stack frame.
__attribute__((noinline)) static void dispatch_slow(Cpu *c) {
    GuestFn f = rt_sealed() ? slow_lookup(c->pc) : NULL;
    if (!f) dispatch_fail(c);
    __attribute__((musttail)) return f(c);
}

// A pending safe-point request (the guest state is all in the Cpu here). Out of line, so rt_dispatch keeps no frame.
// The jump then goes on without a second look at c->pending: a thread that gave its core up here and got it back runs
// to its next safe point, even when a better thread asked for the core again while it waited. Checking again would
// give the core straight back, and a better thread that yields to it in a loop would wait forever for its progress.
__attribute__((noinline)) static void dispatch_safepoint(Cpu *c) {
    rt_safepoint(c);
    uint64_t i = __builtin_rotateright64(c->pc - D.lo, 2);
    uint32_t off = i < D.slots ? D.slot[i] : 0;
    if (__builtin_expect(!off, 0)) __attribute__((musttail)) return dispatch_slow(c);
    GuestFn f = (GuestFn)(D.anchor + off);
    __attribute__((musttail)) return f(c);
}

// Jump to the guest address in c->pc. Always a tail call, so chains of indirect jumps keep the host stack flat.
void rt_dispatch(Cpu *c) {
    if (__builtin_expect(__atomic_load_n(&c->pending, __ATOMIC_RELAXED) != 0, 0))
        __attribute__((musttail)) return dispatch_safepoint(c);
    uint64_t i = __builtin_rotateright64(c->pc - D.lo, 2);
    uint32_t off = i < D.slots ? D.slot[i] : 0;
    if (__builtin_expect(!off, 0)) __attribute__((musttail)) return dispatch_slow(c);
    GuestFn f = (GuestFn)(D.anchor + off);
    __attribute__((musttail)) return f(c);
}

int dispatch_code_ranges(uintptr_t *lo, uintptr_t *end, int max) {
    int n = rt_sealed() ? g_nranges : 0;
    for (int i = 0; i < n && i < max; i++) {
        lo[i] = g_range[i].lo;
        end[i] = g_range[i].end;
    }
    return n;
}

// The lookup of host addresses over the lifter's tables: [lowest function, end marker) of each module.
const char *rt_host_pc_module(uintptr_t pc) {
    if (!rt_sealed()) return NULL;
    for (int i = 0; i < g_nranges; i++)
        if (pc >= g_range[i].lo && pc < g_range[i].end) return g_range[i].m->name;
    return NULL;
}

RtDispatchStats rt_dispatch_stats(void) {
    return (RtDispatchStats){D.lo, D.lo + D.slots * 4, D.slots, g_table_bytes, g_functions, g_nhost};
}

// ---- host pc to guest function. Sorted by host address; built once, after the table is sealed.
typedef struct HostIdx { uintptr_t host; uint64_t guest; uintptr_t mod_end; } HostIdx;
static HostIdx *g_hidx;
static uint64_t g_hcount;
static int g_hidx_built;
static PlatMutex g_hidx_lock = PLAT_MUTEX_INIT;

static int hidx_cmp(const void *x, const void *y) {
    const HostIdx *a = x, *b = y;
    if (a->host != b->host) return a->host < b->host ? -1 : 1;
    return a->guest < b->guest ? -1 : a->guest > b->guest;
}

void dispatch_index_host(void) {
    if (!rt_sealed() || __atomic_load_n(&g_hidx_built, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&g_hidx_lock);
    if (!g_hidx_built) {
        uint64_t total = 0, n = 0;
        for (int i = 0; i < g_nmods; i++) total += g_mods[i]->count;
        HostIdx *x = malloc((total ? total : 1) * sizeof *x);
        if (x) {
            for (int i = 0; i < g_nmods; i++)
                for (uint64_t k = 0; k < g_mods[i]->count; k++)
                    x[n++] = (HostIdx){(uintptr_t)g_mods[i]->fns[k], g_mods[i]->addrs[k], (uintptr_t)g_mods[i]->code_end};
            qsort(x, n, sizeof *x, hidx_cmp);
            uint64_t m = 0;   // one entry per host address: the lowest guest address that has it
            for (uint64_t i = 0; i < n; i++)
                if (!m || x[m - 1].host != x[i].host) x[m++] = x[i];
            g_hidx = x;
            g_hcount = m;
        }
        __atomic_store_n(&g_hidx_built, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&g_hidx_lock);
}

int dispatch_host_pc_guest(uintptr_t pc, uint64_t *guest_fn) {
    dispatch_index_host();
    if (!__atomic_load_n(&g_hidx_built, __ATOMIC_ACQUIRE) || !g_hcount || pc < g_hidx[0].host) return 0;
    uint64_t lo = 0, hi = g_hcount;   // the last entry whose host address is at most pc
    while (hi - lo > 1) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (g_hidx[mid].host <= pc) lo = mid;
        else hi = mid;
    }
    // A function runs to the next function start, and never past its module's end marker: the last function of a
    // module is followed by host code (the next object the linker placed), not by more lifted code.
    uintptr_t end = lo + 1 < g_hcount ? g_hidx[lo + 1].host : (uintptr_t)-1;
    if (end > g_hidx[lo].mod_end) end = g_hidx[lo].mod_end;
    if (pc >= end) return 0;
    *guest_fn = g_hidx[lo].guest;
    return 1;
}
