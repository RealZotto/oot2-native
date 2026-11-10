// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The census (architecture D9): what the run met on its way, written to census.tsv in the game folder whenever the run
// stops (every fatal, guest stop and ExitProcess through the stop hook of fatal.c) and when it ends. The file names
// the game's functions and addresses, so it stays in the player's game folder, like every other output of a run.
//
// One row per object met, tab separated, after a header line:
//   kind      svc, service, command, import (a host function the loader bound), nvn (an NVN2 entry point), boot (the
//             boot driver's host functions), d3 (setjmp and longjmp), memory, input (the input file the run replayed:
//             its name, SHA-256 and what the run took from it), input-record (the file the run recorded), stop
//   key       "0x1f ConnectToNamedPort", a service name, "sm: tipc 1", a symbol
//   status    implemented, offline, stub, absent (hos.h), named-fatal (the run stops when it is met)
//   calls     how many times the run met it
//   first_at  where it was first met: the guest call site as rt_guest_name prints it
//   why       the override's note (its answer and source), an SVC's recorded reason, or the stop line
#include <stdio.h>
#include <string.h>

#include "hos.h"
#include "oot2/gfx.h"
#include "oot2/hid_replay.h"
#include "oot2/ledger.h"
#include "oot2/nvn_pace.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "plat.h"

#define MAX_SERVICES 64
#define MAX_COMMANDS 256

typedef struct { char name[16]; const char *status; uint64_t calls, first_lr; } ServiceRow;
typedef struct { char service[16]; const char *framing, *status; uint32_t cmd; uint64_t calls, first_lr; } CommandRow;

static ServiceRow g_services[MAX_SERVICES];
static CommandRow g_commands[MAX_COMMANDS];
static int g_nservices, g_ncommands, g_overflow;
static PlatMutex g_lock = PLAT_MUTEX_INIT;

void hos_census_service(const char *name, const char *status, const Cpu *c) {
    plat_mutex_lock(&g_lock);
    int i = 0;
    while (i < g_nservices && strcmp(g_services[i].name, name)) i++;
    if (i == g_nservices) {
        if (i == MAX_SERVICES) g_overflow = 1;
        else {
            snprintf(g_services[i].name, sizeof g_services[i].name, "%s", name);
            g_services[i].first_lr = c->x[30];
            g_nservices++;
        }
    }
    if (i < g_nservices) {
        g_services[i].status = status;
        g_services[i].calls++;
    }
    plat_mutex_unlock(&g_lock);
}

void hos_census_command(const char *service, const char *framing, uint32_t cmd, const char *status, const Cpu *c) {
    plat_mutex_lock(&g_lock);
    int i = 0;
    while (i < g_ncommands &&
           (strcmp(g_commands[i].service, service) || strcmp(g_commands[i].framing, framing) || g_commands[i].cmd != cmd))
        i++;
    if (i == g_ncommands) {
        if (i == MAX_COMMANDS) g_overflow = 1;
        else {
            snprintf(g_commands[i].service, sizeof g_commands[i].service, "%s", service);
            g_commands[i].framing = framing;
            g_commands[i].cmd = cmd;
            g_commands[i].first_lr = c->x[30];
            g_ncommands++;
        }
    }
    if (i < g_ncommands) {
        g_commands[i].status = status;
        g_commands[i].calls++;
    }
    plat_mutex_unlock(&g_lock);
}

// One row; tabs and newlines inside a field become spaces, so a row stays one line of six fields.
static void row(FILE *f, const char *kind, const char *key, const char *status, uint64_t calls, uint64_t first,
                const char *why) {
    char at[96] = "";
    if (first) rt_guest_name(first, at, sizeof at);
    const char *fields[] = {kind, key, status, NULL, at, why ? why : ""};
    for (int i = 0; i < 6; i++) {
        if (i) fputc('\t', f);
        if (i == 3) {
            fprintf(f, "%llu", (unsigned long long)calls);
            continue;
        }
        for (const char *p = fields[i]; *p; p++) fputc(*p == '\t' || *p == '\n' || *p == '\r' ? ' ' : *p, f);
    }
    fputc('\n', f);
}

static const char *import_kind(const char *name) {
    return name[0] == '<' ? "boot" : !strncmp(name, "nvn", 3) ? "nvn" : "import";
}

// An NVN2 entry point the registry lists without a body: calling it is the named fatal "unimplemented nvn"
// (runtime/nvn/nvn_procs.c), so its row must not read as implemented.
static int nvn_unbodied(const char *name) {
    if (strncmp(name, "nvn", 3)) return 0;
    for (int i = 0; i < nvn_proc_count(); i++)
        if (!strcmp(nvn_proc_name(i), name)) return !nvn_proc_has_body(i);
    return 0;
}

static void stat_row(void *f, const char *key, uint64_t count, const char *why) {
    row(f, "nvn-stat", key, "implemented", count, 0, why);
}

static void input_row(void *f, const char *kind, const char *key, const char *status, uint64_t calls, const char *why) {
    row(f, kind, key, status, calls, 0, why);
}

static void write_rows(FILE *f, int code, const char *line) {
    if (rt_census_on()) fprintf(f, "#census-mode OOT2_CENSUS=1: gaps in NVN2 were answered 0 and passed; not a default run\n");
    if (ledger_collected())
        fprintf(f, "#collect-variants: NOT A PROOF RUN: %llu draws were skipped, not drawn, to collect shader variants\n",
                (unsigned long long)ledger_collected());
    if (rt_setting(RT_SET_test_movie_blit)) {   // a test switch that changes the picture (settings.h), whatever the totals
        uint64_t skipped, blits;
        nvn_census_movie_blit(&skipped, &blits);
        fprintf(f, "#test-movie-blit: NOT A PROOF RUN: the movie chain's dispatches were not run (%llu) or replaced by a "
                   "resample (%llu); the pictures of the movies are not the console's\n", (unsigned long long)skipped,
                (unsigned long long)blits);
    }
    fprintf(f, "kind\tkey\tstatus\tcalls\tfirst_at\twhy\n");
    for (uint32_t imm = 0; imm <= 0xFF; imm++) {
        uint64_t n = rt_svc_calls(imm);
        if (!n) continue;
        const char *why, *name = rt_svc_name(imm);
        int s = rt_svc_status(imm, &why);
        char key[64];
        snprintf(key, sizeof key, "%#04x %s", imm, name ? name : "(no known name)");
        row(f, "svc", key, s == RT_SVC_IMPLEMENTED && rt_svc_allowed(imm) ? "implemented" : "named-fatal", n,
            rt_svc_first(imm), s == RT_SVC_IMPLEMENTED ? NULL : why);
    }
    plat_mutex_lock(&g_lock);
    for (int i = 0; i < g_nservices; i++)
        row(f, "service", g_services[i].name, g_services[i].status, g_services[i].calls, g_services[i].first_lr - 4,
            NULL);
    for (int i = 0; i < g_ncommands; i++) {
        char key[48];
        snprintf(key, sizeof key, "%s %s %u", g_commands[i].service, g_commands[i].framing, g_commands[i].cmd);
        row(f, "command", key, g_commands[i].status, g_commands[i].calls, g_commands[i].first_lr - 4, NULL);
    }
    int overflow = g_overflow;
    plat_mutex_unlock(&g_lock);
    for (int i = 0; i < rt_hostfn_count(); i++) {
        uint64_t calls, first;
        int missing;
        const char *name = rt_hostfn_at(i, &calls, &first, &missing);
        if (!calls) continue;
        const HosNote *note = missing ? NULL : hos_note(name);
        int unbodied = !missing && nvn_unbodied(name);
        int gap = unbodied && rt_census_on();
        row(f, import_kind(name), name, gap ? "census-gap" : missing || unbodied ? "named-fatal" : note ? note->status : "implemented", calls,
            first > 1 ? first - 4 : 0,
            missing    ? "no module defines it and no override answers it"
            : gap      ? "[census mode] an NVN2 entry point with no body: the run answered 0 and went on"
            : unbodied ? "an NVN2 entry point with no body yet: it stops with its name"
            : note     ? note->why
                       : NULL);
    }
    // What the front end counted (numbers only): presents, recordings, WaitEvents (nvn_census.c words the rows). The
    // inline begins are the game's own BeginRecording code, which no call counter sees; inline + slow begins equal the
    // EndRecordings with a handle.
    nvn_census_stats(stat_row, f);
    hid_replay_rows(input_row, f);   // the input file and the recorder, when the run had them
    uint64_t setjmps, longjmps;
    rt_jmp_counts(&setjmps, &longjmps);
    row(f, "d3", "setjmp", "implemented", setjmps, 0, "guest jmp_buf records made (RT_SETJMP)");
    row(f, "d3", "longjmp", "implemented", longjmps, 0, "the first 32 are logged one line each");
    RtMemStats m;
    rt_mem_measure(&m);
    char why[160];
    snprintf(why, sizeof why, "heap %llu bytes set by SetHeapSize, %llu committed; process private %llu bytes",
             (unsigned long long)m.r[RT_MEM_HEAP].mapped, (unsigned long long)m.r[RT_MEM_HEAP].committed,
             (unsigned long long)m.process_private);
    row(f, "memory", "heap", "implemented", rt_svc_calls(0x01), rt_svc_first(0x01), why);
    if (overflow) row(f, "census", "overflow", "implemented", 0, 0, "more services or commands than the census keeps");
    char code_s[16];
    if (code < 0) snprintf(code_s, sizeof code_s, "running");
    else snprintf(code_s, sizeof code_s, "exit %d", code);
    row(f, "stop", code_s, code < 0 || code == RT_EXIT_CENSUS ? "census" : code ? "named-fatal" : "implemented", 1, 0, line);
}

static int g_written;                               // the run's own end has been written (g_files_lock)
static PlatMutex g_files_lock = PLAT_MUTEX_INIT;    // one writer of census.tsv and census-nvn.tsv at a time
static _Thread_local int g_writing;                 // this thread is inside a writer: a stop it meets does not wait for itself

// census.tsv (every mode) and, in census mode, census-nvn.tsv: the calls per frame of nvn_census.c.
static int write_files(int code, const char *line) {
    char path[4200];
    snprintf(path, sizeof path, "%s/census.tsv", hos_game_folder());
    FILE *f = fopen(path, "wb");
    if (!f) {
        rt_log("census: cannot write %s", path);
        return 0;
    }
    write_rows(f, code, line);
    int bad = ferror(f) | fclose(f);
    if (code >= 0) rt_log("census: %s %s", bad ? "could not finish" : "written to", path);
    if (code >= 0 && ledger_collected())
        rt_log("COLLECTION RUN, NOT A PROOF RUN: %llu draws were skipped, not drawn, over %llu distinct stage and answer "
               "pairs (collect_variants); the ledger holds their rows, so oot2 prepare writes the variants",
               (unsigned long long)ledger_collected(), (unsigned long long)ledger_collected_kinds());
    if (code >= 0 && rt_setting(RT_SET_test_movie_blit)) {
        uint64_t skipped, blits;
        nvn_census_movie_blit(&skipped, &blits);
        rt_log("TEST RUN, NOT A PROOF RUN: test_movie_blit was on; %llu dispatches of the movie chain were not run and %llu "
               "were replaced by a resample, so the pictures of the movies are not the console's", (unsigned long long)skipped,
               (unsigned long long)blits);
    }
    if (rt_census_on()) {
        snprintf(path, sizeof path, "%s/census-nvn.tsv", hos_game_folder());
        f = fopen(path, "wb");
        if (f) {
            nvn_census_write(f);
            bad |= ferror(f) | fclose(f);
            if (code >= 0) rt_log("census: calls per frame written to %s", path);
        }
        nvn_census_extra(hos_game_folder());
    }
    return !bad;
}

// The end of the run. A flush in progress on another thread finishes first, so the end row is the last one written.
static void write_census(int code, const char *line) {
    if (g_writing) return;
    plat_mutex_lock(&g_files_lock);
    if (!__atomic_exchange_n(&g_written, 1, __ATOMIC_ACQ_REL)) {
        g_writing = 1;
        write_files(code, line);
        g_writing = 0;
    }
    plat_mutex_unlock(&g_files_lock);
}

// A census-mode run rewrites its files while it goes on, so a run killed by a time limit still leaves them. Once the end
// has been written a flush leaves the files alone: the test and the write are one step under the lock.
static void census_flush(void) {
    plat_mutex_lock(&g_files_lock);
    if (!__atomic_load_n(&g_written, __ATOMIC_ACQUIRE)) {
        g_writing = 1;
        write_files(-1, "[census mode] the run was still going when this was written");
        g_writing = 0;
    }
    plat_mutex_unlock(&g_files_lock);
}

// Both ends of a run close the recorder's last run first, so the report says what it wrote.
void hos_census_write(const char *why) {
    hid_replay_finish();
    write_census(0, why);
}

static void stop_hook(int code, const char *line) {
    hid_replay_finish();
    write_census(code, line);
}

void census_init(void) {
    rt_set_stop_hook(stop_hook);
    rt_census_set_flush(census_flush);
}
