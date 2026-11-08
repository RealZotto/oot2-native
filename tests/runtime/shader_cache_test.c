// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the shader cache pack reader (runtime/gpu/shader_cache.c), driven by
// tests/test_shader_cache_pack.py, which writes the packs with tools/shaders/cache_pack.py from synthetic modules.
// `shader_cache_test <case> ...`, one case per process:
//
//   open <pack> <role> <schema>             "ok <rows>" or "refused <NAME>: <reason>"
//   find <pack> <role> <queries>            per query: "hit <key> <digest> <status> <module sha> <text sha> <intact>"
//                                           or "miss <key> <digest>"
//   set <base> <variants|-> <queries>       per query: "<base|variants|miss> <key> <digest> <status> <module sha>
//                                           <text sha>", or "refused <NAME>: <reason>" when the set does not open
//   isolated <pack> <role> <key> <digest> <lo> <hi> ...
//                                           maps the pack, makes every byte range [lo, hi) of the mapping
//                                           inaccessible, then opens and looks up: "hit <key> <module sha> <text
//                                           sha>"; a read of a protected byte prints "touched +0x<offset>", exit 3
//   bench <pack> <role> <queries> <rounds>  open time, then per round the time of each lookup and of reading every
//                                           byte of each entry found
//
// A query line is "<key>\t<digest>", the digest "0" for the default answers or 64 hexadecimal digits (the first two
// columns of `cache_pack.py list`). Hashes are SHA-256 in lowercase hexadecimal; "-" for a failure record's module.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/loader.h"
#include "oot2/shader_cache.h"
#include "plat.h"

typedef struct Query { char key[64]; uint8_t digest[32]; int has_digest; char digest_text[65]; } Query;

static void hex(const uint8_t *p, size_t n, char *out) {
    static const char d[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = d[p[i] >> 4];
        out[2 * i + 1] = d[p[i] & 15];
    }
    out[2 * n] = 0;
}

static void sha_hex(const void *p, uint64_t n, char out[65]) {
    uint8_t sha[32];
    rt_sha256(p, n, sha);
    hex(sha, 32, out);
}

static int nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_digest(const char *text, Query *q) {
    memset(q->digest, 0, 32);
    snprintf(q->digest_text, sizeof q->digest_text, "%s", text);
    if (!strcmp(text, "0")) {
        q->has_digest = 0;
        return 1;
    }
    if (strlen(text) != 64) return 0;
    for (int i = 0; i < 32; i++) {
        int a = nibble(text[2 * i]), b = nibble(text[2 * i + 1]);
        if (a < 0 || b < 0) return 0;
        q->digest[i] = (uint8_t)(a << 4 | b);
    }
    q->has_digest = 1;
    return 1;
}

static Query *read_queries(const char *path, size_t *count) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(2);
    }
    size_t cap = 64, n = 0;
    Query *q = malloc(cap * sizeof *q);
    char line[512];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!*line) continue;
        char *tab = strchr(line, '\t');
        char *rest = tab ? strchr(tab + 1, '\t') : NULL;
        if (rest) *rest = 0;
        if (!tab || (size_t)(tab - line) >= sizeof q->key) {
            fprintf(stderr, "bad query line: %s\n", line);
            exit(2);
        }
        *tab = 0;
        if (n == cap) q = realloc(q, (cap *= 2) * sizeof *q);
        snprintf(q[n].key, sizeof q[n].key, "%s", line);
        if (!parse_digest(tab + 1, &q[n])) {
            fprintf(stderr, "bad digest: %s\n", tab + 1);
            exit(2);
        }
        n++;
    }
    fclose(f);
    *count = n;
    return q;
}

static uint32_t role_of(const char *name) {
    if (!strcmp(name, "base")) return SHC_ROLE_BASE;
    if (!strcmp(name, "variants")) return SHC_ROLE_VARIANTS;
    fprintf(stderr, "role is base or variants, not %s\n", name);
    exit(2);
}

static void print_entry(const ShcEntry *e, char mod[65], char text[65]) {
    if (e->module) sha_hex(e->module, e->module_bytes, mod);
    else snprintf(mod, 65, "-");
    sha_hex(e->description, e->description_bytes, text);
}

static int case_open(const char *path, const char *role, const char *schema) {
    ShcPack p;
    char why[512] = "";
    ShcError e = shc_open(&p, path, role_of(role), (uint32_t)strtoul(schema, NULL, 10), why, sizeof why);
    if (e != SHC_OK) {
        printf("refused %s: %s\n", shc_error_name(e), why);
        return 0;
    }
    printf("ok %llu\n", (unsigned long long)p.count);
    shc_close(&p);
    return 0;
}

static int case_find(const char *path, const char *role, const char *queries) {
    ShcPack p;
    char why[512] = "";
    ShcError e = shc_open(&p, path, role_of(role), SHC_DESC_SCHEMA, why, sizeof why);
    if (e != SHC_OK) {
        printf("refused %s: %s\n", shc_error_name(e), why);
        return 1;
    }
    size_t n;
    Query *q = read_queries(queries, &n);
    for (size_t i = 0; i < n; i++) {
        ShcEntry en;
        if (!shc_find(&p, q[i].key, q[i].has_digest ? q[i].digest : NULL, &en)) {
            printf("miss %s %s\n", q[i].key, q[i].digest_text);
            continue;
        }
        char mod[65], text[65];
        print_entry(&en, mod, text);
        printf("hit %s %s %u %s %s %d\n", en.key, q[i].digest_text, en.status, mod, text, shc_entry_intact(&en));
    }
    free(q);
    shc_close(&p);
    return 0;
}

static int case_set(const char *base, const char *variants, const char *queries) {
    ShcSet s;
    char why[512] = "";
    ShcError e = shc_set_open(&s, base, strcmp(variants, "-") ? variants : NULL, SHC_DESC_SCHEMA, why, sizeof why);
    if (e != SHC_OK) {
        printf("refused %s: %s\n", shc_error_name(e), why);
        return 0;
    }
    size_t n;
    Query *q = read_queries(queries, &n);
    for (size_t i = 0; i < n; i++) {
        ShcEntry en;
        int from = shc_set_find(&s, q[i].key, q[i].has_digest ? q[i].digest : NULL, &en);
        if (from == SHC_MISS) {
            printf("miss %s %s\n", q[i].key, q[i].digest_text);
            continue;
        }
        char mod[65], text[65], got[65];
        print_entry(&en, mod, text);
        hex(en.answers, 32, got);
        printf("%s %s %s %u %s %s\n", from == SHC_FROM_VARIANTS ? "variants" : "base", en.key,
               memcmp(en.answers, (uint8_t[32]){0}, 32) ? got : "0", en.status, mod, text);
    }
    free(q);
    shc_set_close(&s);
    return 0;
}

static int case_map(const char *base, const char *variants, const char *map, const char *queries) {
    ShcSet s;
    char why[512] = "";
    ShcError e = shc_set_open_map(&s, base, strcmp(variants, "-") ? variants : NULL,
                                  strcmp(map, "-") ? map : NULL, SHC_DESC_SCHEMA, why, sizeof why);
    if (e != SHC_OK) {
        printf("refused %s: %s\n", shc_error_name(e), why);
        return 0;
    }
    size_t n;
    Query *q = read_queries(queries, &n);
    for (size_t i = 0; i < n; i++) {
        ShcEntry en;
        int mapped = 0;
        int from = shc_set_find_ledger(&s, q[i].key, q[i].digest, &en, &mapped);
        if (from == SHC_MISS) {
            printf("miss %s %s\n", q[i].key, q[i].digest_text);
            continue;
        }
        char mod[65], text[65];
        print_entry(&en, mod, text);
        printf("%s %s %s %u %s %s\n", !mapped ? "fallback" :
               from == SHC_FROM_VARIANTS ? "variants" : "base", en.key, q[i].digest_text, en.status, mod, text);
    }
    free(q);
    shc_set_close(&s);
    return 0;
}

static const uint8_t *g_map;
static uint64_t g_size;

static int touched(uint64_t addr, int write, uint64_t host_pc) {
    (void)write, (void)host_pc;
    uint64_t base = (uint64_t)(uintptr_t)g_map;
    char line[96];
    int n = addr >= base && addr < base + ((g_size + 0xFFF) & ~0xFFFull)
                ? snprintf(line, sizeof line, "touched +0x%llx\n", (unsigned long long)(addr - base))
                : snprintf(line, sizeof line, "fault at 0x%llx outside the pack\n", (unsigned long long)addr);
    plat_write_stderr(line, (size_t)n);
    _Exit(3);
}

static int case_isolated(int argc, char **argv) {
    const char *path = argv[2];
    uint32_t role = role_of(argv[3]);
    Query q;
    snprintf(q.key, sizeof q.key, "%s", argv[4]);
    if (!parse_digest(argv[5], &q)) return 2;
    g_map = plat_map_file(path, &g_size);
    if (!g_map) {
        fprintf(stderr, "cannot map %s\n", path);
        return 2;
    }
    for (int i = 6; i + 1 < argc; i += 2) {
        uint64_t lo = strtoull(argv[i], NULL, 0), hi = strtoull(argv[i + 1], NULL, 0);
        if (!plat_protect((uint64_t)(uintptr_t)g_map + lo, hi - lo, PLAT_NONE)) {
            fprintf(stderr, "cannot protect [%llu, %llu)\n", (unsigned long long)lo, (unsigned long long)hi);
            return 2;
        }
    }
    plat_fault_hook(touched);
    ShcPack p;
    char why[512] = "";
    ShcError e = shc_open_view(&p, g_map, g_size, path, role, SHC_DESC_SCHEMA, why, sizeof why);
    if (e != SHC_OK) {
        printf("refused %s: %s\n", shc_error_name(e), why);
        return 1;
    }
    ShcEntry en;
    if (!shc_find(&p, q.key, q.has_digest ? q.digest : NULL, &en)) {
        printf("miss %s\n", q.key);
        return 1;
    }
    char mod[65], text[65];
    print_entry(&en, mod, text);
    printf("hit %s %s %s\n", en.key, mod, text);
    return 0;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static int case_bench(const char *path, const char *role, const char *queries, const char *rounds_text) {
    size_t n;
    Query *q = read_queries(queries, &n);
    int rounds = atoi(rounds_text);
    ShcPack p;
    char why[512] = "";
    uint64_t t0 = plat_now_ns();
    ShcError e = shc_open(&p, path, role_of(role), SHC_DESC_SCHEMA, why, sizeof why);
    uint64_t t1 = plat_now_ns();
    if (e != SHC_OK) {
        printf("refused %s: %s\n", shc_error_name(e), why);
        return 1;
    }
    printf("bench open_ms=%.3f rows=%llu index_bytes=%llu pack_bytes=%llu queries=%llu\n", (t1 - t0) / 1e6,
           (unsigned long long)p.count, (unsigned long long)p.count * SHC_ROW_BYTES, (unsigned long long)p.size,
           (unsigned long long)n);
    uint64_t *times = malloc((n ? n : 1) * sizeof *times);
    uint64_t sum = 0;
    for (int r = 0; r < rounds; r++) {
        size_t found = 0;
        uint64_t bytes = 0, w0 = plat_now_ns();
        for (size_t i = 0; i < n; i++) {
            ShcEntry en;
            uint64_t a = plat_now_ns();
            int hit = shc_find(&p, q[i].key, q[i].has_digest ? q[i].digest : NULL, &en);
            times[i] = plat_now_ns() - a;
            if (!hit) continue;
            found++;
            // Read every byte of the entry, as creating a shader module from it would.
            const uint8_t *b = en.module ? (const uint8_t *)en.module : (const uint8_t *)en.description;
            uint64_t len = (uint64_t)en.module_bytes + en.description_bytes;
            for (uint64_t k = 0; k + 8 <= len; k += 8) {
                uint64_t v;
                memcpy(&v, b + k, 8);
                sum += v;
            }
            bytes += len;
        }
        uint64_t w1 = plat_now_ns();
        qsort(times, n, sizeof *times, cmp_u64);
        uint64_t p50 = n ? times[n / 2] : 0, p99 = n ? times[(n * 99) / 100] : 0, mx = n ? times[n - 1] : 0;
        printf("bench round=%d found=%llu find_p50_ns=%llu find_p99_ns=%llu find_max_ns=%llu bytes=%llu wall_ms=%.3f\n",
               r, (unsigned long long)found, (unsigned long long)p50, (unsigned long long)p99, (unsigned long long)mx,
               (unsigned long long)bytes, (w1 - w0) / 1e6);
    }
    printf("bench checksum=%llu\n", (unsigned long long)sum);
    free(times);
    free(q);
    shc_close(&p);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc >= 5 && !strcmp(argv[1], "open")) return case_open(argv[2], argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "find")) return case_find(argv[2], argv[3], argv[4]);
    if (argc >= 5 && !strcmp(argv[1], "set")) return case_set(argv[2], argv[3], argv[4]);
    if (argc >= 6 && !strcmp(argv[1], "map")) return case_map(argv[2], argv[3], argv[4], argv[5]);
    if (argc >= 6 && !strcmp(argv[1], "isolated")) return case_isolated(argc, argv);
    if (argc >= 6 && !strcmp(argv[1], "bench")) return case_bench(argv[2], argv[3], argv[4], argv[5]);
    fprintf(stderr, "usage: shader_cache_test open|find|set|isolated|bench ...\n");
    return 2;
}
