// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The shader cache pack reader (oot2/shader_cache.h; the layout is tools/shaders/cache_pack.py's). The file is mapped
// once and checked once: the header, the index's bounds and SHA-256, then every row (key, digest for the role, status
// and sizes, alignment, the entry inside the file, ascending order). The checks run in the order cache_pack.py's
// open_pack runs them and refuse under the same names. Nothing here reads an entry: a lookup is a binary search over
// the index, and the entry it returns is two pointers into the mapping, so a pack of any size costs the pages of its
// index and of the entries the caller reads (nxnative issue 1347 read a 6.7 GB archive whole).
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "oot2/loader.h"
#include "oot2/shader_cache.h"
#include "plat.h"

static const uint8_t zero_digest[SHC_DIGEST_BYTES];

// The pack is little-endian, as the host is.
static uint32_t get32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t get64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static const uint8_t *row_at(const ShcPack *p, uint64_t i) { return p->map + SHC_HEADER_BYTES + i * SHC_ROW_BYTES; }

const char *shc_error_name(ShcError e) {
    switch (e) {
    case SHC_OK: return "OK";
    case SHC_E_OPEN: return "OPEN";
    case SHC_E_SHORT: return "SHORT";
    case SHC_E_MAGIC: return "MAGIC";
    case SHC_E_VERSION: return "VERSION";
    case SHC_E_HEADER: return "HEADER";
    case SHC_E_SCHEMA: return "SCHEMA";
    case SHC_E_ROLE: return "ROLE";
    case SHC_E_TRANSLATOR: return "TRANSLATOR";
    case SHC_E_MAP: return "MAP";
    case SHC_E_INDEX_TRUNCATED: return "INDEX_TRUNCATED";
    case SHC_E_INDEX_HASH: return "INDEX_HASH";
    case SHC_E_ROW: return "ROW";
    case SHC_E_ORDER: return "ORDER";
    case SHC_E_ENTRY_PAST_END: return "ENTRY_PAST_END";
    }
    return "UNKNOWN";
}

static const char *role_name(uint32_t role) { return role == SHC_ROLE_BASE ? "base" : "variants"; }

__attribute__((format(printf, 4, 5)))
static ShcError refuse(ShcError e, char *why, size_t cap, const char *fmt, ...) {
    if (why && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(why, cap, fmt, ap);
        va_end(ap);
    }
    return e;
}

static int key_char(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
           c == '-';
}

// 1 to 31 key characters, then NUL to the end of the field.
static int key_ok(const uint8_t *k) {
    unsigned n = 0;
    while (n < SHC_KEY_BYTES && k[n]) {
        if (!key_char(k[n])) return 0;
        n++;
    }
    if (n == 0 || n == SHC_KEY_BYTES) return 0;
    for (unsigned i = n; i < SHC_KEY_BYTES; i++)
        if (k[i]) return 0;
    return 1;
}

ShcError shc_open_view(ShcPack *p, const void *view, uint64_t size, const char *name, uint32_t role, uint32_t schema,
                       char *why, size_t cap) {
    memset(p, 0, sizeof *p);
    const uint8_t *map = view;
    if (!map || size < SHC_HEADER_BYTES)
        return refuse(SHC_E_SHORT, why, cap, "shader cache pack %s has %llu bytes, less than a header", name,
                      (unsigned long long)(map ? size : 0));
    if (memcmp(map, SHC_MAGIC, 8))
        return refuse(SHC_E_MAGIC, why, cap, "%s is not a shader cache pack (no " SHC_MAGIC " header)", name);
    uint32_t version = get32(map + 8);
    if (version != SHC_VERSION)
        return refuse(SHC_E_VERSION, why, cap, "shader cache pack %s has format version %u, this build reads %u", name,
                      version, SHC_VERSION);
    if (get32(map + 12) != SHC_HEADER_BYTES || get32(map + 16) != SHC_ROW_BYTES || get64(map + 72) != SHC_HEADER_BYTES)
        return refuse(SHC_E_HEADER, why, cap, "shader cache pack %s: the header gives the sizes or offsets of another "
                      "layout", name);
    uint32_t sch = get32(map + 20), rol = get32(map + 24);
    if (sch != schema)
        return refuse(SHC_E_SCHEMA, why, cap, "shader cache pack %s holds descriptions of schema %u, this build reads "
                      "schema %u (run oot2 prepare again)", name, sch, schema);
    if (rol != SHC_ROLE_BASE && rol != SHC_ROLE_VARIANTS)
        return refuse(SHC_E_HEADER, why, cap, "shader cache pack %s has role %u, neither base (1) nor variants (2)",
                      name, rol);
    if (rol != role)
        return refuse(SHC_E_ROLE, why, cap, "shader cache pack %s is a %s pack, a %s pack was expected", name,
                      role_name(rol), role_name(role));
    uint64_t count = get64(map + 64);
    if (count > (size - SHC_HEADER_BYTES) / SHC_ROW_BYTES)
        return refuse(SHC_E_INDEX_TRUNCATED, why, cap, "shader cache pack %s: the index of %llu rows ends past the end "
                      "of the %llu-byte file", name, (unsigned long long)count, (unsigned long long)size);
    uint64_t data_at = SHC_HEADER_BYTES + count * SHC_ROW_BYTES;
    if (get64(map + 80) != data_at)
        return refuse(SHC_E_HEADER, why, cap, "shader cache pack %s: the data offset %llu is not where the index ends",
                      name, (unsigned long long)get64(map + 80));
    uint8_t sha[32];
    rt_sha256(map + SHC_HEADER_BYTES, count * SHC_ROW_BYTES, sha);
    if (memcmp(sha, map + 96, 32))
        return refuse(SHC_E_INDEX_HASH, why, cap, "shader cache pack %s: the index does not match its SHA-256 (the "
                      "file is damaged)", name);

    for (uint64_t i = 0; i < count; i++) {
        const uint8_t *r = map + SHC_HEADER_BYTES + i * SHC_ROW_BYTES;
        unsigned long long n = (unsigned long long)i;
        if (!key_ok(r))
            return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu's key is not 1 to 31 key characters "
                          "padded with NUL", name, n);
        const char *key = (const char *)r;
        int zero = !memcmp(r + SHC_KEY_BYTES, zero_digest, SHC_DIGEST_BYTES);
        if (rol == SHC_ROLE_BASE && !zero)
            return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu (%s) has an answers digest in a base "
                          "pack", name, n, key);
        if (rol == SHC_ROLE_VARIANTS && zero)
            return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu (%s) has the default answers in a "
                          "variants pack", name, n, key);
        uint64_t off = get64(r + 64);
        uint32_t ms = get32(r + 72), ds = get32(r + 76), st = get32(r + 80);
        if (st == SHC_TRANSLATED) {
            if (!ms || ms % 4 || !ds)
                return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu (%s) is translated with a module of "
                              "%u bytes and a description of %u", name, n, key, ms, ds);
        } else if (st == SHC_FAILED) {
            if (ms || !ds)
                return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu (%s) is a failure record with a "
                              "module or no reason", name, n, key);
        } else {
            return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu (%s) has the unknown status %u", name,
                          n, key, st);
        }
        if (off % SHC_ENTRY_ALIGN || off < data_at)
            return refuse(SHC_E_ROW, why, cap, "shader cache pack %s: row %llu (%s)'s entry offset %llu is not a "
                          "multiple of %u after the index", name, n, key, (unsigned long long)off, SHC_ENTRY_ALIGN);
        if (off > size || (uint64_t)ms + ds > size - off)
            return refuse(SHC_E_ENTRY_PAST_END, why, cap, "shader cache pack %s: row %llu (%s)'s entry ends at byte "
                          "%llu, past the end of the %llu-byte file", name, n, key,
                          (unsigned long long)(off + ms + ds), (unsigned long long)size);
        if (i && memcmp(r - SHC_ROW_BYTES, r, SHC_KEY_BYTES + SHC_DIGEST_BYTES) >= 0)
            return refuse(SHC_E_ORDER, why, cap, "shader cache pack %s: row %llu (%s) does not follow row %llu in "
                          "ascending key order", name, n, key, n - 1);
    }

    p->map = map;
    p->size = size;
    p->count = count;
    p->schema = sch;
    p->role = rol;
    p->translator = map + 32;
    return SHC_OK;
}

ShcError shc_open(ShcPack *p, const char *path, uint32_t role, uint32_t schema, char *why, size_t cap) {
    memset(p, 0, sizeof *p);
    uint64_t size = 0;
    const void *map = plat_map_file(path, &size);
    if (!map)
        return refuse(SHC_E_OPEN, why, cap, "cannot open or map the shader cache pack %s", path);
    ShcError e = shc_open_view(p, map, size, path, role, schema, why, cap);
    if (e != SHC_OK) {
        plat_unmap_file(map, size);
        return e;
    }
    p->owns_map = 1;
    return SHC_OK;
}

void shc_close(ShcPack *p) {
    if (p->map && p->owns_map) plat_unmap_file(p->map, p->size);
    memset(p, 0, sizeof *p);
}

int shc_find(const ShcPack *p, const char *key, const uint8_t *answers, ShcEntry *out) {
    if (!p->map || !key) return 0;
    size_t n = strlen(key);
    if (n == 0 || n >= SHC_KEY_BYTES) return 0;
    uint8_t probe[SHC_KEY_BYTES + SHC_DIGEST_BYTES] = {0};
    memcpy(probe, key, n);
    if (answers) memcpy(probe + SHC_KEY_BYTES, answers, SHC_DIGEST_BYTES);
    uint64_t lo = 0, hi = p->count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        const uint8_t *r = row_at(p, mid);
        int c = memcmp(r, probe, sizeof probe);
        if (c < 0) {
            lo = mid + 1;
        } else if (c > 0) {
            hi = mid;
        } else {
            uint64_t off = get64(r + 64);
            uint32_t ms = get32(r + 72), ds = get32(r + 76);
            out->key = (const char *)r;
            out->answers = r + SHC_KEY_BYTES;
            out->status = get32(r + 80);
            out->module = ms ? (const uint32_t *)(const void *)(p->map + off) : NULL;
            out->module_bytes = ms;
            out->description = (const char *)p->map + off + ms;
            out->description_bytes = ds;
            out->sha256 = r + 88;
            return 1;
        }
    }
    return 0;
}

int shc_entry_intact(const ShcEntry *e) {
    const uint8_t *start = e->module ? (const uint8_t *)e->module : (const uint8_t *)e->description;
    uint8_t sha[32];
    rt_sha256(start, (uint64_t)e->module_bytes + e->description_bytes, sha);
    return !memcmp(sha, e->sha256, 32);
}

static ShcError shc_map_open(ShcSet *s, const char *path, char *why, size_t cap) {
    uint64_t size = 0;
    const uint8_t *map = plat_map_file(path, &size);
    if (!map)
        return refuse(SHC_E_MAP, why, cap, "cannot open or map shader variant map %s", path);
    if (size < SHC_MAP_HEADER_BYTES || memcmp(map, SHC_MAP_MAGIC, 8) || get32(map + 8) != SHC_MAP_VERSION ||
        get32(map + 12) != SHC_MAP_HEADER_BYTES || get32(map + 16) != SHC_MAP_ROW_BYTES || get32(map + 20) != 0) {
        plat_unmap_file(map, size);
        return refuse(SHC_E_MAP, why, cap, "shader variant map %s has an unsupported header", path);
    }
    uint64_t count = get64(map + 24);
    if (count > (size - SHC_MAP_HEADER_BYTES) / SHC_MAP_ROW_BYTES ||
        size != SHC_MAP_HEADER_BYTES + count * SHC_MAP_ROW_BYTES) {
        plat_unmap_file(map, size);
        return refuse(SHC_E_MAP, why, cap, "shader variant map %s has a truncated or oversized row table", path);
    }
    uint8_t sha[32];
    rt_sha256(map + SHC_MAP_HEADER_BYTES, count * SHC_MAP_ROW_BYTES, sha);
    if (memcmp(sha, map + 64, sizeof sha)) {
        plat_unmap_file(map, size);
        return refuse(SHC_E_MAP, why, cap, "shader variant map %s has a damaged row table", path);
    }
    const uint8_t *previous = NULL;
    for (uint64_t i = 0; i < count; i++) {
        const uint8_t *r = map + SHC_MAP_HEADER_BYTES + i * SHC_MAP_ROW_BYTES;
        if (!key_ok(r) || (previous && memcmp(previous, r, SHC_KEY_BYTES + SHC_DIGEST_BYTES) >= 0)) {
            plat_unmap_file(map, size);
            return refuse(SHC_E_MAP, why, cap, "shader variant map %s has a malformed or unordered row %llu", path,
                          (unsigned long long)i);
        }
        previous = r;
    }
    s->variant_map = map;
    s->variant_map_size = size;
    s->variant_map_count = count;
    s->owns_variant_map = 1;
    s->has_variant_map = 1;
    return SHC_OK;
}

static void shc_map_close(ShcSet *s) {
    if (s->variant_map && s->owns_variant_map) plat_unmap_file(s->variant_map, s->variant_map_size);
    s->variant_map = NULL;
    s->variant_map_size = s->variant_map_count = 0;
    s->owns_variant_map = s->has_variant_map = 0;
}

static int shc_map_find(const ShcSet *s, const char *key, const uint8_t *digest, const uint8_t **out) {
    if (!s->has_variant_map || !key || !digest) return 0;
    size_t n = strlen(key);
    if (!n || n >= SHC_KEY_BYTES) return 0;
    uint8_t probe[SHC_KEY_BYTES + SHC_DIGEST_BYTES] = {0};
    memcpy(probe, key, n);
    memcpy(probe + SHC_KEY_BYTES, digest, SHC_DIGEST_BYTES);
    uint64_t lo = 0, hi = s->variant_map_count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        const uint8_t *r = s->variant_map + SHC_MAP_HEADER_BYTES + mid * SHC_MAP_ROW_BYTES;
        int c = memcmp(r, probe, sizeof probe);
        if (c < 0) lo = mid + 1;
        else if (c > 0) hi = mid;
        else {
            *out = r + SHC_KEY_BYTES + SHC_DIGEST_BYTES;
            return 1;
        }
    }
    return 0;
}

ShcError shc_set_open_map(ShcSet *s, const char *base, const char *variants, const char *map, uint32_t schema,
                          char *why, size_t cap) {
    memset(s, 0, sizeof *s);
    ShcError e = shc_open(&s->base, base, SHC_ROLE_BASE, schema, why, cap);
    if (e != SHC_OK) return e;
    if (variants) {
        e = shc_open(&s->variants, variants, SHC_ROLE_VARIANTS, schema, why, cap);
        if (e == SHC_OK && memcmp(s->variants.translator, s->base.translator, 32)) {
            e = refuse(SHC_E_TRANSLATOR, why, cap, "shader cache pack %s was written by another translator than the base "
                       "pack %s (run oot2 prepare again)", variants, base);
            shc_close(&s->variants);
        }
        if (e != SHC_OK) {
            shc_close(&s->base);
            return e;
        }
        s->has_variants = 1;
    }
    if (map) {
        e = shc_map_open(s, map, why, cap);
        if (e != SHC_OK) {
            shc_set_close(s);
            return e;
        }
        for (uint64_t i = 0; i < s->variant_map_count; i++) {
            const uint8_t *r = s->variant_map + SHC_MAP_HEADER_BYTES + i * SHC_MAP_ROW_BYTES;
            ShcEntry entry;
            if (!shc_find(&s->base, (const char *)r, NULL, &entry) ||
                (memcmp(r + SHC_KEY_BYTES + SHC_DIGEST_BYTES, zero_digest, SHC_DIGEST_BYTES) &&
                 (!s->has_variants || !shc_find(&s->variants, (const char *)r,
                                                r + SHC_KEY_BYTES + SHC_DIGEST_BYTES, &entry)))) {
                shc_set_close(s);
                return refuse(SHC_E_MAP, why, cap, "shader variant map %s names a row absent from its shader packs",
                              map);
            }
        }
    }
    return SHC_OK;
}

ShcError shc_set_open(ShcSet *s, const char *base, const char *variants, uint32_t schema, char *why, size_t cap) {
    return shc_set_open_map(s, base, variants, NULL, schema, why, cap);
}

int shc_set_find(const ShcSet *s, const char *key, const uint8_t *answers, ShcEntry *out) {
    if (s->has_variants && answers && memcmp(answers, zero_digest, SHC_DIGEST_BYTES) &&
        shc_find(&s->variants, key, answers, out))
        return SHC_FROM_VARIANTS;
    return shc_find(&s->base, key, NULL, out) ? SHC_FROM_BASE : SHC_MISS;
}

int shc_set_resolve_ledger(const ShcSet *s, const char *key, const uint8_t *ledger_digest, uint8_t target[32]) {
    const uint8_t *mapped_target = NULL;
    memset(target, 0, SHC_DIGEST_BYTES);
    if (!shc_map_find(s, key, ledger_digest, &mapped_target)) return 0;
    memcpy(target, mapped_target, SHC_DIGEST_BYTES);
    return 1;
}

int shc_set_find_ledger(const ShcSet *s, const char *key, const uint8_t *ledger_digest, ShcEntry *out, int *mapped) {
    uint8_t target[SHC_DIGEST_BYTES];
    int found = shc_set_resolve_ledger(s, key, ledger_digest, target);
    if (mapped) *mapped = found;
    if (found && memcmp(target, zero_digest, SHC_DIGEST_BYTES)) {
        if (shc_find(&s->variants, key, target, out)) return SHC_FROM_VARIANTS;
        return SHC_MISS;
    }
    return shc_find(&s->base, key, NULL, out) ? SHC_FROM_BASE : SHC_MISS;
}

void shc_set_close(ShcSet *s) {
    shc_map_close(s);
    shc_close(&s->variants);
    shc_close(&s->base);
    s->has_variants = 0;
}
