// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A read-only view of the game's store of named state values (oot2/store.h): the description's reader and the reader of one
// value through guest memory.
#include "oot2/store.h"

#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/loader.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"

#define MAX_FILE 262144u
#define MAX_MODULES 128
#define MAX_ROLES 256
#define MAX_PROBES 1048576u   // cells one probe may visit: a capacity read from memory the description does not fit is not trusted

typedef struct Module {
    uint32_t table, capacity, count, entries, stride, value, index;
    char cls;
    uint8_t shift;
} Module;

typedef struct Role {
    char name[48];
    StoreKind kind;
    uint32_t hash;
} Role;

static uint64_t g_cell;   // the pointer cell's guest address
static Module g_mod[MAX_MODULES];
static int g_nmod;
static Role g_role[MAX_ROLES];
static int g_nrole;

static int fail(char *why, size_t n, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static int fail(char *why, size_t n, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why, n, fmt, ap);
    va_end(ap);
    return 0;
}

// A decimal number of digits only, at most `max`; 0 when t is not one.
static int number(const char *t, uint64_t max, uint64_t *out) {
    if (!*t) return 0;
    uint64_t v = 0;
    for (; *t; t++) {
        if (*t < '0' || *t > '9') return 0;
        v = v * 10 + (uint64_t)(*t - '0');
        if (v > max) return 0;
    }
    *out = v;
    return 1;
}

static int hex_digits(const char *t, size_t n, uint8_t *out) {   // n digits of lower-case hex to n / 2 bytes
    if (strlen(t) != n) return 0;
    for (size_t i = 0; i < n; i++) {
        int c = (unsigned char)t[i], v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (v < 0) return 0;
        if (i % 2 == 0) out[i / 2] = (uint8_t)(v << 4);
        else out[i / 2] |= (uint8_t)v;
    }
    return 1;
}

static const char *const KIND_NAME[] = {"bool", "int", "uint", "float", "string"};

int store_load(const char *path, char *why, size_t why_size) {
    g_nmod = g_nrole = 0;
    g_cell = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return fail(why, why_size, "the store description %s cannot be opened (%s); `oot2 prepare --names FILE` writes it", path, strerror(errno));
    static char buf[MAX_FILE + 1];
    size_t n = fread(buf, 1, MAX_FILE + 1, f);
    int err = ferror(f);
    fclose(f);
    if (err) return fail(why, why_size, "the store description %s cannot be read", path);
    if (n > MAX_FILE) return fail(why, why_size, "the store description %s is larger than %u bytes", path, MAX_FILE);
    buf[n] = 0;
    if (memchr(buf, 0, n)) return fail(why, why_size, "the store description %s holds a NUL byte; it is text", path);

    uint8_t build[32] = {0};
    int have_header = 0, have_build = 0, have_cell = 0, line = 0;
    uint64_t cell_off = 0;
    for (char *p = buf; *p;) {
        char *end = strchr(p, '\n');
        if (end) *end = 0;
        char *next = end ? end + 1 : p + strlen(p);
        line++;
        char *w[12];
        int nw = 0;
        for (char *t = strtok(p, " \t\r"); t && nw < 12; t = strtok(NULL, " \t\r")) w[nw++] = t;
        p = next;
        if (nw == 0 || w[0][0] == '#') continue;
        if (!have_header) {
            if (nw != 2 || strcmp(w[0], "oot2-store") || strcmp(w[1], "1"))
                return fail(why, why_size, "the store description %s is not a file of this format (its first line is not `oot2-store 1`)", path);
            have_header = 1;
            continue;
        }
        uint64_t v[9];
        if (!strcmp(w[0], "build") && nw == 2 && hex_digits(w[1], 64, build)) {
            have_build = 1;
        } else if (!strcmp(w[0], "cell") && nw == 2 && number(w[1], UINT32_MAX, &cell_off)) {
            have_cell = 1;
        } else if (!strcmp(w[0], "module") && nw == 10 && strlen(w[7]) == 1 && strchr("bhwxsd", w[7][0])) {
            if (g_nmod == MAX_MODULES) return fail(why, why_size, "the store description %s, line %d: more than %d tables", path, line, MAX_MODULES);
            int ok = 1;
            for (int i = 0; i < 6; i++) ok &= number(w[1 + i], UINT32_MAX, &v[i]);
            ok &= number(w[8], 6, &v[7]) && number(w[9], UINT32_MAX, &v[8]);
            if (!ok) return fail(why, why_size, "the store description %s, line %d: a table line with a bad number", path, line);
            g_mod[g_nmod++] = (Module){.table = (uint32_t)v[0], .capacity = (uint32_t)v[1], .count = (uint32_t)v[2],
                                       .entries = (uint32_t)v[3], .stride = (uint32_t)v[4], .value = (uint32_t)v[5],
                                       .cls = w[7][0], .shift = (uint8_t)v[7], .index = (uint32_t)v[8]};
        } else if (!strcmp(w[0], "role") && nw == 4) {
            int kind = -1;
            for (int i = 0; i < 5; i++)
                if (!strcmp(w[2], KIND_NAME[i])) kind = i;
            uint8_t h[4];
            if (kind < 0 || strlen(w[1]) >= sizeof g_role[0].name || !hex_digits(w[3], 8, h))
                return fail(why, why_size, "the store description %s, line %d: a role line that is not one", path, line);
            if (g_nrole == MAX_ROLES) return fail(why, why_size, "the store description %s, line %d: more than %d roles", path, line, MAX_ROLES);
            Role *r = &g_role[g_nrole++];
            for (size_t i = 0; i <= strlen(w[1]); i++) r->name[i] = (char)tolower((unsigned char)w[1][i]);
            r->kind = (StoreKind)kind;
            r->hash = (uint32_t)h[0] << 24 | (uint32_t)h[1] << 16 | (uint32_t)h[2] << 8 | h[3];
        } else {
            return fail(why, why_size, "the store description %s, line %d: not a line of the format", path, line);
        }
    }
    if (!have_header || !have_build || !have_cell)
        return fail(why, why_size, "the store description %s has no build line or no cell line", path);
    const RtLoaded *main = NULL;
    for (int i = 0; i < rt_loaded_count(); i++)
        if (!strcmp(rt_loaded(i)->name, "main")) main = rt_loaded(i);
    if (!main) return fail(why, why_size, "the main module is not loaded, so the store description %s cannot be checked against it", path);
    if (memcmp(main->build_id, build, 32))
        return fail(why, why_size, "the store description %s was made from another build of the game's main module; run `oot2 prepare --names FILE` again", path);
    if (cell_off + 8 > main->size) return fail(why, why_size, "the store description %s: the pointer cell lies outside the main module", path);
    g_cell = main->base + cell_off;
    return 1;
}

int store_role_count(void) { return g_nrole; }
const char *store_role_name(int role) { return g_role[role].name; }
StoreKind store_role_kind(int role) { return g_role[role].kind; }

int store_role(const char *name) {
    for (int i = 0; i < g_nrole; i++) {
        const char *a = g_role[i].name, *b = name;
        for (; *a && *b && tolower((unsigned char)*a) == tolower((unsigned char)*b); a++, b++) {}
        if (!*a && !*b) return i;
    }
    return -1;
}

// ---- guest memory, read-only
enum { G_OK, G_WAIT, G_BAD };

// n bytes at a guest address, only if the runtime has them committed. An address that lies in no guest region is G_BAD, one
// in a region that is not committed (or the null address) G_WAIT.
static int guest(uint64_t addr, void *dst, size_t n) {
    if (addr == 0) return G_WAIT;
    int committed = 0;
    uint64_t run = rt_mem_commit_run(addr, n, &committed);
    if (run == UINT64_MAX) return G_BAD;
    if (!committed || run < n) return G_WAIT;
    memcpy(dst, (const void *)(uintptr_t)addr, n);
    return G_OK;
}

typedef struct Ctx {
    char *why;
    size_t why_size;
} Ctx;

#define GET(ctx, addr, var)                                                                               \
    do {                                                                                                  \
        int g_ = guest((addr), &(var), sizeof(var));                                                      \
        if (g_ == G_WAIT) return STORE_WAIT;                                                              \
        if (g_ == G_BAD) {                                                                                \
            snprintf((ctx)->why, (ctx)->why_size,                                                         \
                     "the store description leads to address 0x%llx, which is outside the guest's memory; " \
                     "run `oot2 prepare --names FILE` again", (unsigned long long)(addr));                \
            return STORE_BAD;                                                                             \
        }                                                                                                 \
    } while (0)

static char class_of(StoreKind k) {
    switch (k) {
    case STORE_BOOL: return 'b';
    case STORE_FLOAT: return 's';
    case STORE_STRING: return 'x';
    default: return 'w';
    }
}

// One table: STORE_OK with *out, STORE_MISSING when the table is built and has no such entry, STORE_WAIT when it is not
// built yet.
static StoreRead lookup(Ctx *c, uint64_t obj, const Module *m, const Role *r, StoreValue *out) {
    uint64_t table, entries, cell;
    uint32_t cap, count, key = 0, idx = 0;
    GET(c, obj + m->table, table);
    GET(c, obj + m->capacity, cap);
    if (!table || !cap) return STORE_WAIT;
    GET(c, obj + m->count, count);
    GET(c, obj + m->entries, entries);
    uint32_t pos = r->hash % cap;
    int found = 0;
    for (uint32_t i = 0; i < cap && i < MAX_PROBES; i++) {
        cell = table + ((uint64_t)pos << m->shift);
        GET(c, cell, key);
        if (key == r->hash) {
            GET(c, cell + m->index, idx);
            found = 1;
            break;
        }
        if (key == 0) break;
        pos = pos + 1 == cap ? 0 : pos + 1;
    }
    if (!found || idx >= count || !entries) return STORE_MISSING;
    uint64_t at = entries + (uint64_t)idx * m->stride + m->value;
    switch (r->kind) {
    case STORE_BOOL: {
        uint8_t b;
        GET(c, at, b);
        out->i = b != 0;
        return STORE_OK;
    }
    case STORE_INT: {
        int32_t v;
        GET(c, at, v);
        out->i = v;
        return STORE_OK;
    }
    case STORE_UINT: {
        uint32_t v;
        GET(c, at, v);
        out->i = v;
        return STORE_OK;
    }
    case STORE_FLOAT: {
        float v;
        GET(c, at, v);
        out->f = v;
        return STORE_OK;
    }
    case STORE_STRING: {
        uint64_t p;
        GET(c, at, p);
        if (!p) return STORE_WAIT;
        size_t n = sizeof out->s - 1;
        int committed = 0;
        uint64_t run = rt_mem_commit_run(p, n, &committed);   // the characters may end a page before 63 do
        if (run == UINT64_MAX) {
            snprintf(c->why, c->why_size, "the string of the store lies at address 0x%llx, outside the guest's memory",
                     (unsigned long long)p);
            return STORE_BAD;
        }
        if (!committed) return STORE_WAIT;
        if (run < n) n = (size_t)run;
        memcpy(out->s, (const void *)(uintptr_t)p, n);
        out->s[n] = 0;
        out->s[strnlen(out->s, n)] = 0;
        out->s[sizeof out->s - 1] = 0;
        return STORE_OK;
    }
    }
    return STORE_MISSING;
}

StoreRead store_read(int role, StoreValue *out, char *why, size_t why_size) {
    Ctx c = {why, why_size};
    const Role *r = &g_role[role];
    uint64_t var, obj;
    GET(&c, g_cell, var);
    GET(&c, var, obj);
    if (!obj) return STORE_WAIT;
    int wait = 0, tried = 0;
    for (int i = 0; i < g_nmod; i++) {
        if (g_mod[i].cls != class_of(r->kind)) continue;
        tried++;
        StoreRead s = lookup(&c, obj, &g_mod[i], r, out);
        if (s == STORE_OK || s == STORE_BAD) return s;
        wait |= s == STORE_WAIT;
    }
    if (!tried) {
        snprintf(why, why_size, "the store description holds no table that is read as a %s", KIND_NAME[r->kind]);
        return STORE_BAD;
    }
    return wait ? STORE_WAIT : STORE_MISSING;
}
