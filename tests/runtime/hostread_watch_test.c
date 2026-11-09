// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Does the write watch of the runtime's own heap see a host file read? Driven by tests/test_runtime_hostread_watch.py:
// `hostread_watch_test <case> <scratch dir>`. The pages are the runtime's: rt_mem_init reserves the heap with the
// write-watch flag, a first touch or rt_mem_commit_io commits it, and rt_mem_written is the query. A file of synthetic
// bytes is made in the scratch directory; each case reads it into committed heap pages after a reset of the watch and
// requires that the watch then reports exactly the pages the read touched. Windows only: elsewhere the case prints a
// skip line.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/loader.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
#define P 0x1000ull

#ifdef _WIN32
static Cpu c;

static RtMemConfig test_config(void) {
    return (RtMemConfig){.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                         .stack_region_size = 64 * MiB, .system_resource_size = MiB};
}

static uint64_t svc(uint32_t imm, uint64_t x0, uint64_t x1, uint64_t x2) {
    c.x[0] = x0, c.x[1] = x1, c.x[2] = x2;
    rt_svc(&c, imm, 0);
    return c.x[0];
}

static uint64_t info(uint32_t id) {
    uint64_t v = 0;
    CHECK(rt_mem_info(id, &v));
    return v;
}

enum { FILE_BYTES = 256 * 1024 };

static uint8_t pattern(uint64_t i) { return (uint8_t)(i * 7 + (i >> 8) + 1); }

static wchar_t g_path[1024];

static void make_file(const char *dir) {
    wchar_t wdir[512];
    CHECK(MultiByteToWideChar(CP_UTF8, 0, dir, -1, wdir, 512));
    swprintf(g_path, 1024, L"%ls\\hostread.bin", wdir);
    HANDLE h = CreateFileW(g_path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    CHECK(h != INVALID_HANDLE_VALUE);
    uint8_t *buf = malloc(FILE_BYTES);
    CHECK(buf);
    for (uint64_t i = 0; i < FILE_BYTES; i++) buf[i] = pattern(i);
    DWORD put = 0;
    CHECK(WriteFile(h, buf, FILE_BYTES, &put, NULL) && put == FILE_BYTES);
    free(buf);
    CloseHandle(h);
}

typedef enum { READ_SYNC, READ_ASYNC, READ_COPY } Mode;

static int g_pending;   // reads that returned ERROR_IO_PENDING (printed: the note says whether the test saw real waits)

// One host read into guest memory, the way a host service does it (hostfile.c: a handle opened for overlapped I/O, an
// offset in the OVERLAPPED, a wait on the event when the call returns pending), or a plain memcpy as the control.
// buffered: the system cache; otherwise the handle is opened with FILE_FLAG_NO_BUFFERING, which makes the file system
// transfer into the caller's pages directly. Returns the bytes read.
static DWORD host_read(uint64_t to, DWORD n, uint64_t from, Mode mode, int buffered) {
    if (mode == READ_COPY) {
        uint8_t *src = malloc(n);
        CHECK(src);
        for (DWORD i = 0; i < n; i++) src[i] = pattern(from + i);
        memcpy((void *)(uintptr_t)to, src, n);
        free(src);
        return n;
    }
    DWORD flags = FILE_ATTRIBUTE_NORMAL | (mode == READ_ASYNC ? FILE_FLAG_OVERLAPPED : 0) |
                  (buffered ? 0 : FILE_FLAG_NO_BUFFERING);
    HANDLE h = CreateFileW(g_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
    CHECK(h != INVALID_HANDLE_VALUE);
    DWORD got = 0;
    if (mode == READ_SYNC) {
        LARGE_INTEGER at = {.QuadPart = (LONGLONG)from};
        CHECK(SetFilePointerEx(h, at, NULL, FILE_BEGIN));
        CHECK(ReadFile(h, (void *)(uintptr_t)to, n, &got, NULL));
    } else {
        HANDLE ev = CreateEventW(NULL, TRUE, FALSE, NULL);
        CHECK(ev);
        OVERLAPPED ov = {.Offset = (DWORD)from, .OffsetHigh = (DWORD)(from >> 32), .hEvent = ev};
        if (!ReadFile(h, (void *)(uintptr_t)to, n, &got, &ov)) {
            CHECK(GetLastError() == ERROR_IO_PENDING);
            g_pending++;
            CHECK(GetOverlappedResult(h, &ov, &got, TRUE));
        }
        CloseHandle(ev);
    }
    CloseHandle(h);
    return got;
}

typedef struct Seen {
    uint64_t n, pages[64];
} Seen;

static Seen written(uint64_t lo, uint64_t size, int reset) {
    Seen s;
    s.n = rt_mem_written(lo, size, s.pages, 64, reset);
    return s;
}

static int skip_unless_watched(uint64_t heap) {
    uint64_t pages[4];
    return rt_mem_written(heap, 0x1000, pages, 4, 0) == UINT64_MAX;
}

// Read len bytes of the file from offset `from` to `to` inside a committed window of the region at base, after a reset
// of the watch, and require that the watch reports exactly the pages the read touched: all of them, and no other of the
// window. Then a query with reset empties the watch, and a second read marks the pages again.
static void expect_marks(const char *what, uint64_t base, uint64_t window, uint64_t to, DWORD len, uint64_t from,
                         Mode mode, int buffered) {
    const uint64_t first = to & ~(P - 1), last = (to + len - 1) & ~(P - 1), pages = (last - first) / P + 1;
    for (int round = 0; round < 2; round++) {
        memset((void *)(uintptr_t)to, 0, len);   // the zeros are the read's absence; their marks go with the reset
        Seen s = written(base, window, 1);
        CHECK(s.n >= 1);
        CHECK(written(base, window, 0).n == 0);
        CHECK(host_read(to, len, from, mode, buffered) == len);
        for (DWORD i = 0; i < len; i += 509) CHECK(*(uint8_t *)(uintptr_t)(to + i) == pattern(from + i));
        CHECK(*(uint8_t *)(uintptr_t)(to + len - 1) == pattern(from + len - 1));
        s = written(base, window, 0);
        printf("  %s: %llu of %llu page(s) reported\n", what, (unsigned long long)s.n, (unsigned long long)pages);
        CHECK(s.n == pages);
        for (uint64_t k = 0; k < s.n; k++) CHECK(s.pages[k] == first + k * P);
        CHECK(written(base, window, 1).n == pages && written(base, window, 0).n == 0);
    }
}

static uint64_t setup_heap(const char *what) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 8 * MiB, 0) == 0);
    const uint64_t heap = info(4);
    if (skip_unless_watched(heap)) {
        printf("skip %s: write watch not available on this platform\n", what);
        exit(0);
    }
    return heap;
}

enum { WINDOW = 0x40000 };

static void run(const char *what, Mode mode, int buffered, int in_alias) {
    const uint64_t heap = setup_heap(what);
    uint64_t base = heap + 0x100000;
    if (in_alias) {
        base = info(2) + 0x10000;
        CHECK(svc(0x2C, base, WINDOW, 0) == 0);
    }
    CHECK(rt_mem_commit_io(base, WINDOW, 1));
    // aligned to the sector, so that an unbuffered read is allowed: three pages, then a long read
    expect_marks(what, base, WINDOW, base + 0x1000, 0x3000, 0x1000, mode, buffered);
    expect_marks(what, base, WINDOW, base, 0x30000, 0x1000, mode, buffered);
    if (buffered) {   // a read that starts and ends inside a page marks the pages it touches
        expect_marks(what, base, WINDOW, base + 0x2800, 0x2800, 0x100, mode, buffered);
        expect_marks(what, base, WINDOW, base + 0x5fff, 2, 0x7ffe, mode, buffered);   // one byte in each of two pages
    }
    printf("  %s: %d read(s) returned pending\n", what, g_pending);
    printf("ok %s\n", what);
}

#endif

static const char *const CASES[] = {"control", "sync", "overlapped", "sync_unbuffered", "overlapped_unbuffered",
                                    "alias_sync", "alias_overlapped"};

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 3) {
        fprintf(stderr, "usage: hostread_watch_test <case> <scratch dir>\n");
        return 2;
    }
    const char *cs = argv[1];
    int known = 0;
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) known |= !strcmp(cs, CASES[i]);
    if (!known) return 2;
#ifdef _WIN32
    make_file(argv[2]);
    if (!strcmp(cs, "control")) run(cs, READ_COPY, 1, 0);
    else if (!strcmp(cs, "sync")) run(cs, READ_SYNC, 1, 0);
    else if (!strcmp(cs, "overlapped")) run(cs, READ_ASYNC, 1, 0);
    else if (!strcmp(cs, "sync_unbuffered")) run(cs, READ_SYNC, 0, 0);
    else if (!strcmp(cs, "overlapped_unbuffered")) run(cs, READ_ASYNC, 0, 0);
    else if (!strcmp(cs, "alias_sync")) run(cs, READ_SYNC, 1, 1);
    else run(cs, READ_ASYNC, 1, 1);
#else
    printf("skip %s: no write watch on this platform\n", cs);
#endif
    return 0;
}
