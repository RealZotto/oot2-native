// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The save transaction (runtime/hos/savetxn.c) over real host files, with no guest and no GPU. Driven by
// tests/test_runtime_savetxn.py: `savetxn_test <case> <dir> [die]`. The folder <dir>/save/u1 is the save mount's root.
// Every case is its own process so that the parent can end one at any step and start the next one over what it left:
// with die = N the process ends itself (TerminateProcess, or _exit away from Windows: no handle closed, no buffer flushed)
// at the N-th step the transaction reports (the steps of savetxn.h's st_trace, plus the case's own "written"). Each step
// is printed as "ev <step> <index>" as it happens.
//
// The files of a save: account.sav, option.sav, slot_00/data.sav and slot_01/data.sav (they exist, with the OLD content),
// and slot_02/data.sav (made by the case). Their NEW content has another length and another pattern, so a mix of the two
// in one file shows as BAD. `state <name> OLD|NEW|BAD|NONE` lines say what a file holds.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "fs.h"
#include "savetxn.h"

// What hostfile.c needs from the kernel: no guest thread here, so nothing to give up.
int rt_kernel_block_begin(void) { return 0; }
void rt_kernel_block_end(int token) { (void)token; }
void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc(10, stderr);
    exit(1);
}

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

enum { NFILES = 5 };
static const char *const names[NFILES] = {"account.sav", "option.sav", "slot_00/data.sav", "slot_01/data.sav",
                                          "slot_02/data.sav"};
static char root[1024];
static long die_at = -1;
static long seen;

static void die_now(void) {
    fflush(stdout);
#ifdef _WIN32
    TerminateProcess(GetCurrentProcess(), 77);
#else
    _exit(77);
#endif
}

static void on_event(const char *step, int index) {
    printf("ev %s %d\n", step, index);
    fflush(stdout);
    if (seen++ == die_at) die_now();
}

// Every call of hf_sync that the unit makes is a step too (the link wraps it: -Wl,--wrap=hf_sync), so that a commit which
// skips a sync shows as a missing step, and so that an end can fall right after any one of them.
int __real_hf_sync(HostFile *f);
int __wrap_hf_sync(HostFile *f) {
    int ok = __real_hf_sync(f);
    on_event("hfsync", ok);
    return ok;
}

static uint32_t content_len(int version, int file) { return version ? 5000u + 111u * (uint32_t)file : 3000u + 77u * (uint32_t)file; }
static uint8_t content_at(int version, int file, uint32_t i) {
    return (uint8_t)((i * (version ? 13u : 7u)) + (uint32_t)file * 31u + (version ? 0x5a : 0x11) + (i >> 8));
}

static void real_path(int file, char *out, size_t cap) { snprintf(out, cap, "%s/%s", root, names[file]); }

static int write_whole(HostFile *f, int version, int file) {
    uint32_t n = content_len(version, file);
    uint8_t *buf = malloc(n);
    for (uint32_t i = 0; i < n; i++) buf[i] = content_at(version, file, i);
    int ok = hf_set_size(f, 0) == HF_OK && hf_pwrite(f, buf, n, 0) == (int64_t)n;
    free(buf);
    return ok;
}

// What file holds: 0 NONE, 1 OLD, 2 NEW, 3 BAD.
static int state_of(int file) {
    char p[2048];
    real_path(file, p, sizeof p);
    int err = 0;
    HostFile *f = hf_open(p, 0, &err);
    if (!f) return 0;
    uint64_t size = 0;
    CHECK(hf_size(f, &size));
    int result = 3;
    for (int v = 0; v < 2; v++) {
        if (size != content_len(v, file)) continue;
        uint8_t *buf = malloc((size_t)size);
        CHECK(hf_pread(f, buf, (uint32_t)size, 0) == (int64_t)size);
        int same = 1;
        for (uint32_t i = 0; i < size && same; i++) same = buf[i] == content_at(v, file, i);
        free(buf);
        if (same) result = v + 1;
    }
    hf_close(f);
    return result;
}

static void print_states(void) {
    static const char *const word[] = {"NONE", "OLD", "NEW", "BAD"};
    for (int i = 0; i < NFILES; i++) printf("state %s %s\n", names[i], word[state_of(i)]);
    fflush(stdout);
}

static void setup(void) {
    char p[2048];
    snprintf(p, sizeof p, "%s", root);
    for (char *q = p + 1; *q; q++)
        if (*q == '/' && q[-1] != ':') {
            *q = 0;
            hf_mkdir(p);
            *q = '/';
        }
    CHECK(hf_mkdir(p) == HF_OK || hf_kind(p) == HF_DIR);
    for (int d = 0; d < 3; d++) {
        snprintf(p, sizeof p, "%s/slot_%02d", root, d);
        CHECK(hf_mkdir(p) == HF_OK || hf_kind(p) == HF_DIR);
    }
    for (int i = 0; i < NFILES - 1; i++) {
        real_path(i, p, sizeof p);
        int err = 0;
        HostFile *f = hf_create(p, &err);
        CHECK(f && write_whole(f, 0, i) && hf_sync(f));
        hf_close(f);
    }
    printf("ok setup\n");
}

// What the guest does through ov_fs.c: a save file that changes is opened on its shadow; a new one is made as a shadow.
static void change_file(int file) {
    char p[2048], shadow[2048];
    real_path(file, p, sizeof p);
    uint64_t serial = 0;
    int err = file == NFILES - 1 ? st_begin_create(p, 0, &serial) : st_begin(p, &serial);
    CHECK(err == HF_OK && serial);
    CHECK(st_shadow_path(p, shadow, sizeof shadow));
    HostFile *f = hf_open(shadow, 1, &err);
    CHECK(f);
    CHECK(write_whole(f, 1, file));
    hf_close(f);
}

static void change_all(void) {
    for (int i = 0; i < NFILES; i++) change_file(i);
    on_event("written", 0);
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    const char *what = argv[1];
    snprintf(root, sizeof root, "%s/save/u1", argv[2]);
    for (char *q = root; *q; q++)
        if (*q == '\\') *q = '/';
    if (argc > 3) die_at = atol(argv[3]);
    if (!strcmp(what, "setup")) {
        setup();
        return 0;
    }
    int finished = st_recover(root);
    printf("recovered %d\n", finished);
    st_trace = on_event;
    if (!strcmp(what, "read")) {   // a fresh process over whatever is there: recovery ran above
        print_states();
    } else if (!strcmp(what, "commit")) {   // change every file, then Commit; a read handle on a file is open across it
        char p[2048];
        real_path(0, p, sizeof p);
        int err = 0;
        HostFile *reader = hf_open(p, 0, &err);
        CHECK(reader);
        change_all();
        int r = st_commit(root);
        printf("commit result %d\n", r);
        CHECK(r == HF_OK);
        hf_close(reader);
        print_states();
    } else if (!strcmp(what, "nocommit")) {   // the writes, then the process ends before any Commit
        change_all();
        die_now();
    } else if (!strcmp(what, "discard")) {   // the writes, then the mount is unmounted
        change_all();
        st_discard_prefix(root);
        print_states();
    } else if (!strcmp(what, "view")) {   // the guest's view while a change is pending, and after Commit
        char p[2048], shadow[2048];
        real_path(2, p, sizeof p);
        change_file(2);
        CHECK(st_shadow_path(p, shadow, sizeof shadow));
        printf("pending serial %d\n", st_serial(p) != 0);
        printf("real is %s\n", state_of(2) == 1 ? "OLD" : "OTHER");
        int err = 0;
        HostFile *f = hf_open(shadow, 0, &err);
        CHECK(f);
        uint64_t size = 0;
        CHECK(hf_size(f, &size));
        printf("shadow size %llu want %u\n", (unsigned long long)size, content_len(1, 2));
        hf_close(f);
        CHECK(st_commit(root) == HF_OK);
        printf("after commit serial %d\n", st_serial(p) != 0);
        printf("real is %s\n", state_of(2) == 2 ? "NEW" : "OTHER");
        CHECK(hf_kind(shadow) == HF_MISSING);
        // a second change after the commit starts a new shadow of the committed content, with a new serial
        uint64_t again = 0;
        CHECK(st_begin(p, &again) == HF_OK && again);
        printf("second shadow %d\n", again != 0);
        st_discard_prefix(root);
        printf("discarded serial %d\n", st_serial(p) != 0);
        CHECK(st_begin_create(p, 0, &again) == HF_EXISTS);   // a file that is there cannot be made again
        printf("create over existing refused\n");
    } else {
        return 2;
    }
    return 0;
}
