// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Shadow files and the commit of a save mount; the design is in savetxn.h. Written for this project, not adapted.
//
// The order that makes a stop at any point harmless:
//   1. every shadow is on the disk (fsync);
//   2. the marker is written under a temporary name, synced, then renamed to its name, so it is whole or absent, and it
//      carries a count and a checksum so a damaged one is read as absent too;
//   3. each shadow is renamed over its file (one atomic replace each);
//   4. the marker is removed.
// A stop before step 2 ends has no marker: recovery deletes the shadows and every file is as it was. A stop after it has a
// marker: recovery renames the shadows that are still there (the ones already renamed are not) and every file is new.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs.h"
#include "plat.h"
#include "savetxn.h"

#define ST_PATH 2048
#define ST_CHUNK (64u << 10)
#define ST_MAGIC "OOT2SAVETXN1"

void (*st_trace)(const char *step, int index);

typedef struct Pending { char real[ST_PATH]; uint64_t serial; } Pending;

static PlatMutex st_lock = PLAT_MUTEX_INIT;   // the tables below; never held across a host call
static Pending *pend;
static int npend, cappend;
static uint64_t next_serial = 1;
static char recovered[16][ST_PATH];
static int nrecovered;

static void trace(const char *step, int index) {
    if (st_trace) st_trace(step, index);
}

int st_shadow_path(const char *real, char *out, size_t cap) {
    return snprintf(out, cap, "%s" ST_SUFFIX, real) < (int)cap;
}

// real is prefix itself or a path under it.
static int under(const char *real, const char *prefix) {
    size_t n = strlen(prefix);
    while (n && prefix[n - 1] == '/') n--;
    return !strncmp(real, prefix, n) && (real[n] == 0 || real[n] == '/');
}

static int find_locked(const char *real) {
    for (int i = 0; i < npend; i++)
        if (!strcmp(pend[i].real, real)) return i;
    return -1;
}

uint64_t st_serial(const char *real) {
    plat_mutex_lock(&st_lock);
    int i = find_locked(real);
    uint64_t s = i < 0 ? 0 : pend[i].serial;
    plat_mutex_unlock(&st_lock);
    return s;
}

// Add real to the table, or return the serial it already has.
static uint64_t add_pending(const char *real) {
    plat_mutex_lock(&st_lock);
    int i = find_locked(real);
    if (i < 0) {
        if (npend == cappend) {
            cappend = cappend ? cappend * 2 : 16;
            pend = realloc(pend, (size_t)cappend * sizeof *pend);
            if (!pend) rt_fatal("fs: out of memory for the save shadow table");
        }
        i = npend++;
        snprintf(pend[i].real, sizeof pend[i].real, "%s", real);
        pend[i].serial = next_serial++;
    }
    uint64_t s = pend[i].serial;
    plat_mutex_unlock(&st_lock);
    return s;
}

static void remove_pending(const char *real) {
    plat_mutex_lock(&st_lock);
    int i = find_locked(real);
    if (i >= 0) pend[i] = pend[--npend];
    plat_mutex_unlock(&st_lock);
}

// The pending paths at or under prefix (a copy; the caller frees it); n of them.
static Pending *snapshot(const char *prefix, int *n) {
    plat_mutex_lock(&st_lock);
    Pending *out = malloc((size_t)(npend ? npend : 1) * sizeof *out);
    int k = 0;
    for (int i = 0; out && i < npend; i++)
        if (under(pend[i].real, prefix)) out[k++] = pend[i];
    plat_mutex_unlock(&st_lock);
    if (!out) rt_fatal("fs: out of memory for the save shadow table");
    *n = k;
    return out;
}

// ---- shadows
static int copy_file(const char *from, const char *to) {
    int err = HF_OK;
    HostFile *src = hf_open(from, 0, &err);
    if (!src) return err;
    hf_unlink(to);   // a stale shadow of a stopped run
    HostFile *dst = hf_create(to, &err);
    if (!dst) {
        hf_close(src);
        return err;
    }
    char *buf = malloc(ST_CHUNK);
    if (!buf) rt_fatal("fs: out of memory for a save copy");
    uint64_t off = 0;
    for (;;) {
        int64_t got = hf_pread(src, buf, ST_CHUNK, off);
        if (got < 0) err = HF_ERROR;
        if (got <= 0) break;
        for (int64_t put = 0; put < got;) {
            int64_t w = hf_pwrite(dst, buf + put, (uint32_t)(got - put), off + (uint64_t)put);
            if (w <= 0) {
                err = HF_ERROR;
                got = 0;
                break;
            }
            put += w;
        }
        if (!got) break;
        off += (uint64_t)got;
    }
    free(buf);
    hf_close(src);
    hf_close(dst);
    if (err != HF_OK) hf_unlink(to);
    return err;
}

int st_begin(const char *real, uint64_t *serial) {
    uint64_t s = st_serial(real);
    if (s) {
        *serial = s;
        return HF_OK;
    }
    char shadow[ST_PATH];
    if (!st_shadow_path(real, shadow, sizeof shadow)) return HF_ERROR;
    if (hf_kind(real) != HF_FILE) return HF_NOT_FOUND;
    int err = copy_file(real, shadow);
    if (err != HF_OK) return err;
    *serial = add_pending(real);
    return HF_OK;
}

int st_begin_create(const char *real, uint64_t size, uint64_t *serial) {
    char shadow[ST_PATH];
    if (!st_shadow_path(real, shadow, sizeof shadow)) return HF_ERROR;
    if (st_serial(real) || hf_kind(real) != HF_MISSING) return HF_EXISTS;
    hf_unlink(shadow);   // a stale shadow of a stopped run
    int err = HF_OK;
    HostFile *f = hf_create(shadow, &err);
    if (!f) return err;
    err = size ? hf_set_size(f, size) : HF_OK;
    hf_close(f);
    if (err != HF_OK) {
        hf_unlink(shadow);
        return err;
    }
    *serial = add_pending(real);
    return HF_OK;
}

// The shadow's data is on the disk.
static int sync_shadow(const char *shadow, int index) {
    int err = HF_OK;
    HostFile *f = hf_open(shadow, 1, &err);
    if (!f) return err;
    int ok = hf_sync(f);
    hf_close(f);
    if (!ok) return HF_ERROR;
    trace("sync", index);
    return HF_OK;
}

// ---- the marker
static uint32_t fnv(const char *s, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 16777619u;
    return h;
}

static int marker_paths(const char *root, char *marker, char *tmp, size_t cap) {
    size_t n = strlen(root);
    while (n && root[n - 1] == '/') n--;
    return snprintf(marker, cap, "%.*s/" ST_MARKER, (int)n, root) < (int)cap &&
           snprintf(tmp, cap, "%.*s/" ST_MARKER_TMP, (int)n, root) < (int)cap;
}

// Write the marker for the n entries (paths relative to root) so that it exists whole or not at all.
static int write_marker(const char *root, const Pending *e, int n) {
    char marker[ST_PATH], tmp[ST_PATH];
    if (!marker_paths(root, marker, tmp, sizeof marker)) return HF_ERROR;
    size_t rootn = strlen(root);
    while (rootn && root[rootn - 1] == '/') rootn--;
    size_t cap = 64;
    for (int i = 0; i < n; i++) cap += strlen(e[i].real) + 2;
    char *text = malloc(cap);
    if (!text) rt_fatal("fs: out of memory for a save commit marker");
    size_t len = (size_t)snprintf(text, cap, ST_MAGIC "\n%d\n", n);
    for (int i = 0; i < n; i++) len += (size_t)snprintf(text + len, cap - len, "%s\n", e[i].real + rootn + 1);
    len += (size_t)snprintf(text + len, cap - len, "END %08x\n", fnv(text, len));
    int err = HF_OK;
    hf_unlink(tmp);
    HostFile *f = hf_create(tmp, &err);
    if (f) {
        if (hf_pwrite(f, text, (uint32_t)len, 0) != (int64_t)len || !hf_sync(f)) err = HF_ERROR;
        hf_close(f);
    }
    free(text);
    if (!f) return err;
    if (err != HF_OK) return err;
    trace("marker-tmp", 0);
    err = hf_replace(tmp, marker);
    if (err == HF_OK) trace("marker", 0);
    return err;
}

int st_commit(const char *root) {
    int n = 0;
    Pending *e = snapshot(root, &n);
    int err = HF_OK;
    if (!n) goto out;
    char shadow[ST_PATH], marker[ST_PATH], tmp[ST_PATH];
    if (!marker_paths(root, marker, tmp, sizeof marker)) {
        err = HF_ERROR;
        goto out;
    }
    for (int i = 0; i < n && err == HF_OK; i++) {
        if (!st_shadow_path(e[i].real, shadow, sizeof shadow)) err = HF_ERROR;
        else err = sync_shadow(shadow, i);
    }
    if (err == HF_OK) err = write_marker(root, e, n);
    for (int i = 0; i < n && err == HF_OK; i++) {
        st_shadow_path(e[i].real, shadow, sizeof shadow);
        err = hf_replace(shadow, e[i].real);
        if (err == HF_OK) {
            remove_pending(e[i].real);
            trace("publish", i);
        }
    }
    if (err == HF_OK) {
        hf_unlink(marker);
        trace("done", 0);
    }
out:
    free(e);
    return err;
}

int st_publish_prefix(const char *prefix) {
    int n = 0;
    Pending *e = snapshot(prefix, &n);
    int err = HF_OK;
    char shadow[ST_PATH];
    for (int i = 0; i < n && err == HF_OK; i++) {
        if (!st_shadow_path(e[i].real, shadow, sizeof shadow)) err = HF_ERROR;
        else err = sync_shadow(shadow, i);
        if (err == HF_OK) err = hf_replace(shadow, e[i].real);
        if (err == HF_OK) remove_pending(e[i].real);
    }
    free(e);
    return err;
}

void st_discard_prefix(const char *prefix) {
    int n = 0;
    Pending *e = snapshot(prefix, &n);
    char shadow[ST_PATH];
    for (int i = 0; i < n; i++) {
        remove_pending(e[i].real);
        if (st_shadow_path(e[i].real, shadow, sizeof shadow)) hf_unlink(shadow);
    }
    free(e);
}

// ---- recovery
typedef struct Names { char **name; int *kind; int n, cap; } Names;

static int names_add(const char *name, int kind, uint64_t size, void *ctx) {
    (void)size;
    Names *l = ctx;
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->name = realloc(l->name, (size_t)l->cap * sizeof *l->name);
        l->kind = realloc(l->kind, (size_t)l->cap * sizeof *l->kind);
        if (!l->name || !l->kind) rt_fatal("fs: out of memory for a save listing");
    }
    char *copy = strdup(name);
    if (!copy) rt_fatal("fs: out of memory for a save listing");
    l->name[l->n] = copy;
    l->kind[l->n++] = kind;
    return 1;
}

static void names_free(Names *l) {
    for (int i = 0; i < l->n; i++) free(l->name[i]);
    free(l->name);
    free(l->kind);
}

static int ends_with(const char *s, const char *suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && !strcmp(s + a - b, suffix);
}

// Remove the shadows under dir that no pending entry names; returns how many.
static int sweep(const char *dir) {
    Names l = {0};
    int removed = 0;
    if (hf_list(dir, names_add, &l) != HF_OK) {
        names_free(&l);
        return 0;
    }
    for (int i = 0; i < l.n; i++) {
        char path[ST_PATH];
        if (snprintf(path, sizeof path, "%s/%s", dir, l.name[i]) >= (int)sizeof path) continue;
        if (l.kind[i] == HF_DIR) {
            removed += sweep(path);
        } else if (l.kind[i] == HF_FILE && ends_with(l.name[i], ST_SUFFIX)) {
            char real[ST_PATH];
            snprintf(real, sizeof real, "%.*s", (int)(strlen(path) - strlen(ST_SUFFIX)), path);
            if (!st_serial(real) && hf_unlink(path) == HF_OK) {
                removed++;
                trace("orphan", removed);
            }
        }
    }
    names_free(&l);
    return removed;
}

// The relative paths of a whole marker, each a malloc'd string in rel[0..n); 0 when the file is absent, short, damaged,
// or names a path that leaves the folder (nothing is allocated then).
static int read_marker(const char *marker, char ***rel, int *n) {
    int err = HF_OK;
    HostFile *f = hf_open(marker, 0, &err);
    uint64_t size = 0;
    if (!f) return 0;
    if (!hf_size(f, &size) || size < 32 || size > (1u << 20)) {
        hf_close(f);
        return 0;
    }
    char *buf = malloc((size_t)size + 1);
    if (!buf) rt_fatal("fs: out of memory for a save commit marker");
    int64_t got = hf_pread(f, buf, (uint32_t)size, 0);
    hf_close(f);
    buf[size] = 0;
    char **lines = NULL;
    long want = -1;
    int count = 0, ok = got == (int64_t)size;
    const char *p = buf;
    if (ok) {
        ok = !strncmp(p, ST_MAGIC "\n", strlen(ST_MAGIC) + 1);
        p += strlen(ST_MAGIC) + 1;
    }
    if (ok) {
        char *end;
        want = strtol(p, &end, 10);
        ok = end != p && want >= 0 && want < 100000 && *end == '\n';
        p = end + 1;
    }
    if (ok) lines = calloc((size_t)want + 1, sizeof *lines);
    for (; ok && count < want; count++) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : 0;
        char canon[ST_PATH], slashed[ST_PATH];
        ok = nl && len && len < sizeof slashed - 2;
        if (ok) {
            snprintf(slashed, sizeof slashed, "/%.*s", (int)len, p);
            ok = fs_path_norm(slashed, canon, sizeof canon, 1) && strlen(canon) == len && !memcmp(canon, p, len);
        }
        if (!ok) break;
        lines[count] = malloc(len + 1);
        if (!lines[count]) rt_fatal("fs: out of memory for a save commit marker");
        memcpy(lines[count], p, len);
        lines[count][len] = 0;
        p = nl + 1;
    }
    if (ok) {   // p is the END line; the checksum covers every byte before it
        char expect[32];
        snprintf(expect, sizeof expect, "END %08x\n", fnv(buf, (size_t)(p - buf)));
        ok = !strcmp(p, expect);
    }
    free(buf);
    if (!ok) {
        for (int i = 0; lines && i < count; i++) free(lines[i]);
        free(lines);
        return 0;
    }
    *rel = lines;
    *n = count;
    return 1;
}

int st_recover(const char *root) {
    plat_mutex_lock(&st_lock);
    for (int i = 0; i < nrecovered; i++)
        if (!strcmp(recovered[i], root)) {
            plat_mutex_unlock(&st_lock);
            return 0;
        }
    if (nrecovered < (int)(sizeof recovered / sizeof recovered[0]))
        snprintf(recovered[nrecovered++], sizeof recovered[0], "%s", root);
    plat_mutex_unlock(&st_lock);

    char marker[ST_PATH], tmp[ST_PATH];
    if (!marker_paths(root, marker, tmp, sizeof marker)) return -1;
    int finished = 0, bad = 0;
    char **rel = NULL;
    int n = 0;
    if (hf_kind(marker) == HF_FILE) {
        if (read_marker(marker, &rel, &n)) {
            for (int i = 0; i < n; i++) {
                char shadow[ST_PATH], real[ST_PATH];
                snprintf(real, sizeof real, "%s/%s", root, rel[i]);
                st_shadow_path(real, shadow, sizeof shadow);
                if (hf_kind(shadow) != HF_FILE) continue;   // published before the stop
                if (hf_replace(shadow, real) == HF_OK) {
                    finished++;
                    trace("recover", i);
                } else {
                    bad = 1;
                }
            }
            for (int i = 0; i < n; i++) free(rel[i]);
            free(rel);
        }
        // A marker that does not read whole was never complete, so no file was renamed: it is dropped with the shadows.
        if (!bad) hf_unlink(marker);
    }
    hf_unlink(tmp);
    if (!bad) sweep(root);
    return bad ? -1 : finished;
}
