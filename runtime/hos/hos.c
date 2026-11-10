// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What every file of runtime/hos shares (hos.h): calling guest code from an override, guest strings and
// out-parameters, the named fatal of a gap, the SDK's system events bound to kernel events, the game folder and its
// files, the census notes, and the layer's start-up.
//
// System events follow oot-pc-recomp runtime/kernel.c at 8e09aaa (GPL-3.0-or-later; kern_bind_system_event), which
// lets the SDK's own code own the SystemEventType. Changed here: the event is the kernel's host event
// (rt_kernel_event_create) attached with nn::os::AttachReadableHandleToSystemEvent, since OoT's SVC mask has no
// CreateEvent (0x45) for the SDK's nn::os::CreateSystemEvent to call.
#include <stdio.h>
#include <string.h>

#include "hos.h"
#include "plat.h"

#include "oot2/hid_replay.h"

#define MAX_EVENTS 256
#define SYM_ATTACH_READABLE "_ZN2nn2os33AttachReadableHandleToSystemEventEPNS0_15SystemEventTypeEjbNS0_14EventClearModeE"

static char g_folder[4096];
static const RtGameFile *g_files;
static int g_nfiles;

// ---- the census notes: a linker section, walked as hostfn.c walks HOST_OVERRIDE's
#ifdef _WIN32
__attribute__((section(".hosn$a"), used, aligned(32))) static const HosNote notes_start[1];
__attribute__((section(".hosn$z"), used, aligned(32))) static const HosNote notes_stop[1];
#else
extern const HosNote __start_oot2_hosn[] __attribute__((weak)), __stop_oot2_hosn[] __attribute__((weak));
#define notes_start __start_oot2_hosn
#define notes_stop __stop_oot2_hosn
#endif

const HosNote *hos_note(const char *symbol) {
    const HosNote *p = notes_start, *e = notes_stop;
    __asm__("" : "+r"(p), "+r"(e));   // separate objects: keep the compiler from reasoning about the bounds
    for (; p < e; p++)
        if (p->symbol && !strcmp(p->symbol, symbol)) return p;
    return NULL;
}

// ---- guest code
uint64_t hos_call(Cpu *c, uint64_t fn, int n, ...) {
    if (n < 0 || n > 8) rt_fatal("hos_call: %d arguments, at most 8 go in registers", n);
    va_list ap;
    va_start(ap, n);
    for (int i = 0; i < n; i++) c->x[i] = va_arg(ap, uint64_t);
    va_end(ap);
    uint64_t pc = c->pc, lr = c->x[30];
    rt_call(c, fn);
    c->pc = pc;
    c->x[30] = lr;
    return c->x[0];
}

uint64_t hos_sdk(Cpu *c, const char *symbol) {
    uint64_t a = rt_symbol(symbol);
    if (!a) hos_gap(c, "boot", "the SDK function %s is not defined by any module, and the runtime answers through it",
                    symbol);
    return a;
}

// ---- guest memory
const char *hos_str(uint64_t addr, char *buf, size_t n) {
    if (!addr) return "(null)";
    size_t i = 0;
    for (; i + 1 < n && addr + i < RT_ADDRESS_LIMIT; i++) {
        char ch = (char)LD8(addr + i);
        if (!ch) break;
        buf[i] = ch;
    }
    buf[i] = 0;
    return buf;
}

const char *hos_caller(const Cpu *c, char *buf, int n) { return rt_guest_name(c->x[30] - 4, buf, n); }

int hos_out(uint64_t addr, const void *src, uint64_t size) {
    if (!size) return 1;
    if (!rt_mem_commit_io(addr, size, 1)) return 0;
    memcpy((void *)(uintptr_t)addr, src, size);
    return 1;
}

int hos_in(void *dst, uint64_t addr, uint64_t size) {
    if (!size) return 1;
    if (!rt_mem_commit_io(addr, size, 0)) return 0;
    memcpy(dst, (const void *)(uintptr_t)addr, size);
    return 1;
}

void hos_gap(Cpu *c, const char *kind, const char *fmt, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_unimplemented_trace(c, kind, "%s", msg);
}

// ---- system events
static struct { uint64_t ev; uint32_t writable; } g_events[MAX_EVENTS];
static int g_nevents;
static PlatMutex g_events_lock = PLAT_MUTEX_INIT;

static uint32_t event_find(uint64_t ev) {
    for (int i = 0; i < g_nevents; i++)
        if (g_events[i].ev == ev) return g_events[i].writable;
    return 0;
}

// nn::os::AttachReadableHandleToSystemEvent(SystemEventType *, Handle, bool managed, EventClearMode): the SDK closes
// the readable end when the game destroys the event (managed); the writable end stays the host's.
uint32_t hos_bind_system_event(Cpu *c, uint64_t ev, int autoclear) {
    plat_mutex_lock(&g_events_lock);
    uint32_t w = event_find(ev);
    plat_mutex_unlock(&g_events_lock);
    if (w) return w;
    uint64_t attach = hos_sdk(c, SYM_ATTACH_READABLE);
    uint32_t readable;
    w = rt_kernel_event_create(&readable);
    hos_call(c, attach, 4, ev, (uint64_t)readable, (uint64_t)1, (uint64_t)(autoclear ? 1 : 0));
    plat_mutex_lock(&g_events_lock);
    if (g_nevents == MAX_EVENTS) rt_fatal("hos: more than %d system events bound to the host", MAX_EVENTS);
    g_events[g_nevents].ev = ev;
    g_events[g_nevents++].writable = w;
    plat_mutex_unlock(&g_events_lock);
    return w;
}

void hos_signal(uint32_t writable) { rt_kernel_event_signal(writable); }

// ---- the game folder
const char *hos_game_folder(void) { return g_folder; }

const char *hos_game_file(const char *what, char *buf, size_t n) {
    for (int i = 0; i < g_nfiles; i++) {
        if (strcmp(g_files[i].what, what)) continue;
        const char *p = g_files[i].path;
        int absolute = p[0] == '/' || p[0] == '\\' || (p[0] && p[1] == ':');
        int k = absolute ? snprintf(buf, n, "%s", p) : snprintf(buf, n, "%s/%s", g_folder, p);
        if (k < 0 || (size_t)k >= n) rt_fatal("hos: the path of %s is too long", what);
        return buf;
    }
    return NULL;
}

// ---- start-up
void hos_init(const char *game_folder, const RtGameFile *files, int file_count) {
    if (rt_sealed()) rt_fatal("hos_init after rt_dispatch_init");
    if (snprintf(g_folder, sizeof g_folder, "%s", game_folder) >= (int)sizeof g_folder)
        rt_fatal("hos: the game folder's path is too long");
    g_files = files;
    g_nfiles = file_count;
    ipc_register();
    srv_audio_register();
    srv_auddmg_register();
    srv_audsmx_register();
    boot_register();
    census_init();
    hid_replay_start();   // OOT2_INPUT_FILE and OOT2_INPUT_RECORD, off by default
    // Overrides no import or definition named: written for a name this game's SDK does not define (drift by name).
    int unbound = 0;
    for (const char *name; (name = rt_hostfn_unbound(unbound)); unbound++)
        rt_log("hos: the override of %s binds nothing in these modules", name);
    rt_log("hos: %d overrides linked, %d of them bind nothing in these modules", rt_hostfn_override_count(), unbound);
}
