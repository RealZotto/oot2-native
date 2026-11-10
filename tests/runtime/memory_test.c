// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the guest address space and the module loader (architecture D6, D9), driven by
// tests/test_runtime_memory.py: `memory_test <case> <scratch dir>`. Each case runs in its own process. A passing case
// prints "ok <case>"; a failing case must end in one named fatal, which the driver checks. The modules are synthetic:
// two plain NSO0 files assembled here, byte by byte, with the relocations and symbols each case needs.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/loader.h"
#include "plat.h"
#ifdef _WIN32
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
#define MB MOD_MAIN_BASE
#define SB MOD_SDK_BASE

// ---- the synthetic modules: text, rodata and data of one page, one page and two pages, then bss
enum { TEXT = 0x1000, RO = 0x1000, DATA = 0x2000, BSS = 0x3800, IMAGE = 0x8000 };
enum { RO_OFF = 0x1000, DATA_OFF = 0x2000, BSS_OFF = 0x4000, DYNSYM_OFF = 0x400 };
enum { RELA_OFF = 0x200, JMPREL_OFF = 0x400, RELR_OFF = 0x600, SLOTS = 0x800 };
#define SLOT(i) (DATA_OFF + SLOTS + 8 * (i))
_Static_assert(MOD_MAIN_SIZE == IMAGE && MOD_SDK_SIZE == IMAGE, "the driver's layout matches these modules");

enum { DT_STRTAB = 5, DT_SYMTAB = 6, DT_RELA = 7, DT_RELASZ = 8, DT_STRSZ = 10, DT_INIT = 12, DT_PLTRELSZ = 2,
       DT_JMPREL = 23, DT_RELRSZ = 35, DT_RELR = 36 };
enum { R_COPY = 1024, R_ABS64 = 257, R_GLOB_DAT = 1025, R_JUMP_SLOT = 1026, R_RELATIVE = 1027 };
enum { GLOBAL = 1, WEAK = 2, NOTYPE = 0, OBJECT = 1, FUNC = 2 };

typedef struct { const char *name; int defined; uint64_t value; int bind, type; } SymSpec;
typedef struct { uint64_t off; uint32_t type, sym; int64_t addend; } RelSpec;

static const char *const MISSING = "_ZN2nn2os12NotThereYetEv";

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

static void put_rels(uint8_t *at, const RelSpec *r, int n) {
    for (int i = 0; i < n; i++) {
        put64(at + 24 * i, r[i].off);
        put64(at + 24 * i + 8, (uint64_t)r[i].sym << 32 | r[i].type);
        put64(at + 24 * i + 16, (uint64_t)r[i].addend);
    }
}

// Variants that break one thing: the driver's failing cases.
enum { V_GOOD, V_COMPRESSED, V_CORRUPT, V_RELOC_OUTSIDE, V_RELOC_TYPE };

static void write_module(const char *path, const SymSpec *syms, int nsyms, RelSpec *rela, int nrela,
                         const RelSpec *jmp, int njmp, const uint64_t *relr, int nrelr, const uint64_t (*init)[2],
                         int ninit, int variant) {
    static uint8_t img[BSS_OFF];
    memset(img, 0, sizeof img);
    put32(img + 4, 0x100);   // MOD0 offset
    memcpy(img + 0x100, "MOD0", 4);
    put32(img + 0x104, DATA_OFF - 0x100);
    put32(img + 0x108, BSS_OFF - 0x100);
    put32(img + 0x10C, IMAGE - 0x100);
    uint32_t str = 1;
    for (int i = 0; i < nsyms; i++) {
        uint8_t *s = img + RO_OFF + DYNSYM_OFF + 24 * (i + 1);
        size_t n = strlen(syms[i].name) + 1;
        memcpy(img + RO_OFF + str, syms[i].name, n);
        put32(s, str);
        s[4] = (uint8_t)(syms[i].bind << 4 | syms[i].type);
        s[6] = syms[i].defined ? 1 : 0;
        put64(s + 8, syms[i].value);
        put64(s + 16, 8);
        str += (uint32_t)n;
    }
    if (variant == V_RELOC_OUTSIDE) rela[0].off = 0x10;
    if (variant == V_RELOC_TYPE) rela[0].type = R_COPY;
    const uint64_t dyn[][2] = {{DT_SYMTAB, RO_OFF + DYNSYM_OFF}, {DT_STRTAB, RO_OFF}, {DT_STRSZ, str},
                               {DT_RELA, DATA_OFF + RELA_OFF}, {DT_RELASZ, 24ull * (uint64_t)nrela},
                               {DT_JMPREL, DATA_OFF + JMPREL_OFF}, {DT_PLTRELSZ, 24ull * (uint64_t)njmp},
                               {DT_RELR, DATA_OFF + RELR_OFF}, {DT_RELRSZ, 8ull * (uint64_t)nrelr}, {DT_INIT, 0x200},
                               {0, 0}};
    memcpy(img + DATA_OFF, dyn, sizeof dyn);
    put_rels(img + DATA_OFF + RELA_OFF, rela, nrela);
    put_rels(img + DATA_OFF + JMPREL_OFF, jmp, njmp);
    for (int i = 0; i < nrelr; i++) put64(img + DATA_OFF + RELR_OFF + 8 * i, relr[i]);
    for (int i = 0; i < ninit; i++) put64(img + init[i][0], init[i][1]);

    uint8_t hdr[0x100] = {0};
    const uint32_t file_off[3] = {0x100, 0x100 + TEXT, 0x100 + TEXT + RO}, mem[3] = {0, RO_OFF, DATA_OFF},
                   size[3] = {TEXT, RO, DATA};
    memcpy(hdr, "NSO0", 4);
    put32(hdr + 0xC, 0x38u | (variant == V_COMPRESSED ? 1u : 0u));   // segment hashes present
    for (int i = 0; i < 3; i++) {
        put32(hdr + 0x10 + 0x10 * i, file_off[i]);
        put32(hdr + 0x14 + 0x10 * i, mem[i]);
        put32(hdr + 0x18 + 0x10 * i, size[i]);
        put32(hdr + 0x60 + 4 * i, size[i]);
        rt_sha256(img + mem[i], size[i], hdr + 0xA0 + 32 * i);
    }
    put32(hdr + 0x3C, BSS);
    memset(hdr + 0x40, 0xAB, 32);   // build id
    put32(hdr + 0x94, str);
    put32(hdr + 0x98, DYNSYM_OFF);
    put32(hdr + 0x9C, 24u * (uint32_t)(nsyms + 1));
    if (variant == V_CORRUPT) img[DATA_OFF + 0x7F0] ^= 1;   // after hashing
    FILE *f = fopen(path, "wb");
    CHECK(f && fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr && fwrite(img, 1, sizeof img, f) == sizeof img);
    fclose(f);
}

static char main_path[1024], sdk_path[1024];

// main imports sdk_func and sdk_data from sdk, longjmp (a host override in runtime/core/jmp.c), one function nothing
// defines and one weak symbol nothing defines; sdk defines longjmp itself, which the override replaces.
static void make_modules(const char *dir, int variant) {
    static const SymSpec main_syms[] = {
        {"sdk_func", 0, 0, GLOBAL, FUNC},   // 1
        {"longjmp", 0, 0, GLOBAL, FUNC},    // 2
        {NULL, 0, 0, GLOBAL, FUNC},         // 3, MISSING
        {"weak_nowhere", 0, 0, WEAK, NOTYPE},   // 4
        {"main_func", 1, 0x200, GLOBAL, FUNC},  // 5
        {"sdk_data", 0, 0, GLOBAL, OBJECT},     // 6
    };
    SymSpec ms[6];
    memcpy(ms, main_syms, sizeof ms);
    ms[2].name = MISSING;
    RelSpec main_rela[] = {
        {SLOT(0), R_RELATIVE, 0, 0x1234}, {SLOT(1), R_ABS64, 1, 8}, {SLOT(2), R_GLOB_DAT, 2, 0},
        {SLOT(3), R_GLOB_DAT, 4, 0}, {SLOT(4), R_GLOB_DAT, 6, 0}, {SLOT(5), R_GLOB_DAT, 5, 0},
        {BSS_OFF + 0x1000, R_RELATIVE, 0, 0x40},   // into bss: committed by the first touch
    };
    const RelSpec main_jmp[] = {{SLOT(6), R_JUMP_SLOT, 1, 0}, {SLOT(7), R_JUMP_SLOT, 3, 0}};
    // SLOT(8) by offset, then a bitmap from SLOT(9): words 0, 2 and 62 (SLOT 9, 11, 71), then one from SLOT(72): word 0
    const uint64_t main_relr[] = {SLOT(8), 1 | 1ull << 1 | 1ull << 3 | 1ull << 63, 1 | 1ull << 1};
    const uint64_t main_init[][2] = {{SLOT(8), 0x80}, {SLOT(9), 0x90}, {SLOT(11), 0xB0}, {SLOT(71), 0x710}, {SLOT(72), 0x720}};
    static const SymSpec sdk_syms[] = {
        {"sdk_func", 1, 0x300, GLOBAL, FUNC}, {"sdk_data", 1, DATA_OFF + 0x900, GLOBAL, OBJECT},
        {"longjmp", 1, 0x400, GLOBAL, FUNC},
    };
    RelSpec sdk_rela[] = {{SLOT(0), R_GLOB_DAT, 3, 0}};
    const RelSpec sdk_jmp[] = {{SLOT(1), R_JUMP_SLOT, 1, 0}};
    snprintf(main_path, sizeof main_path, "%s/main", dir);
    snprintf(sdk_path, sizeof sdk_path, "%s/sdk", dir);
    write_module(main_path, ms, 6, main_rela, 7, main_jmp, 2, main_relr, 3, main_init, 5, variant);
    write_module(sdk_path, sdk_syms, 3, sdk_rela, 1, sdk_jmp, 1, NULL, 0, NULL, 0, V_GOOD);
}

static RtMemConfig test_config(void) {
    return (RtMemConfig){.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                         .stack_region_size = 64 * MiB, .system_resource_size = MiB};
}

static void load(const char *dir, int variant, uint64_t main_size, const char *main_sha256) {
    make_modules(dir, variant);
    RtModuleFile files[2] = {{"main", main_path, MB, main_size, main_sha256}, {"sdk", sdk_path, SB, MOD_SDK_SIZE, NULL}};
    rt_load_modules(files, 2);
}

static Cpu c;

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

static RtMemRegion region(int r) {
    RtMemStats s;
    rt_mem_measure(&s);
    return s.r[r];
}

static void fake_fn(Cpu *cpu) { (void)cpu; }

// ---- passing cases
static void case_sha256(void) {
    static uint8_t buf[1000];
    for (int i = 0; i < 1000; i++) buf[i] = (uint8_t)(i * 7);
    const struct { const void *p; uint64_t n; } in[] = {{"", 0}, {"abc", 3}, {buf, 55}, {buf, 56}, {buf, 64}, {buf, 1000}};
    for (int k = 0; k < 6; k++) {
        uint8_t d[32];
        rt_sha256(in[k].p, in[k].n, d);
        printf("sha256 %d ", k);
        for (int i = 0; i < 32; i++) printf("%02x", d[i]);
        printf("\n");
    }
}

static void case_heap(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    uint64_t heap = c.x[1];
    CHECK(heap == info(4) && info(5) == 64 * MiB && info(6) == 256 * MiB && info(21) == 256 * MiB);
    CHECK(info(3) == 1024 * MiB && info(15) == 64 * MiB && info(12) == 0x8000000 && info(13) == (1ull << 39) - 0x8000000);
    CHECK(info(7) == 4 * MiB && info(16) == MiB && info(28) == 0);
    CHECK(region(RT_MEM_HEAP).committed == 0);
    ST32(heap + 0x12345, 0xC0FFEE);   // one chunk: [0x10000, 0x20000)
    CHECK(region(RT_MEM_HEAP).committed == 0x10000 && region(RT_MEM_HEAP).touch_commits == 1);
    CHECK(LD32(heap + 0x300000) == 0);   // a read commits too
    RtMemRegion h = region(RT_MEM_HEAP);
    CHECK(h.committed == 0x20000 && h.mapped == 4 * MiB && h.resident >= 0x1000 && h.resident <= h.committed);
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);   // shrinking decommits the chunk past 2 MiB
    CHECK(region(RT_MEM_HEAP).committed == 0x10000);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    CHECK(LD32(heap + 0x300000) == 0 && LD32(heap + 0x12345) == 0xC0FFEE);
    CHECK(svc(0x01, 0, 4 * MiB + 0x1000, 0) == 0xca01);   // not a multiple of 2 MiB
    CHECK(svc(0x01, 0, 128 * MiB, 0) == 0xd001);          // past the heap region
    CHECK(svc(0x02, heap, 0x1000, 1) == 0xd401);          // SetMemoryPermission outside module data
    CHECK(svc(0x03, heap, 0x1000, 8) == 0);               // uncached: accepted on the heap (mask 8, attr 0)
    c.x[3] = 2;
    CHECK(svc(0x03, heap, 0x1000, 2) == 0xe801);          // IpcLocked cannot be set
}

static void case_pool(void) {
    RtMemConfig cfg = test_config();
    cfg.pool_size = 8 * MiB;
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 8 * MiB, 0) == 0);
    CHECK(svc(0x2C, info(2), 0x1000, 0) == 0x10801);   // the pool is spent
    CHECK(svc(0x01, 0, 10 * MiB, 0) == 0x10801);
}

static void case_alias(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    uint64_t alias = info(2);
    CHECK(svc(0x2C, alias + 0x3000, 0x2000, 0) == 0);
    CHECK(info(7) == 0x2000);
    ST8(alias + 0x3000, 1);
    ST8(alias + 0x4FFF, 2);
    RtMemRegion a = region(RT_MEM_ALIAS);
    CHECK(a.committed == 0x2000 && a.mapped == 0x2000 && a.touch_commits == 1);   // the chunk, clipped to the mapped pages
    RtMemInfo q = rt_mem_query(alias + 0x4000);
    CHECK(q.base == alias + 0x3000 && q.size == 0x2000 && q.state == 5 && q.perm == 3);
    q = rt_mem_query(alias + 0x10000);
    CHECK(q.base == alias + 0x5000 && q.state == 0 && q.base + q.size == alias + 1024 * MiB);
    CHECK(svc(0x2C, alias + 0x2000, 0x3000, 0) == 0);   // maps only the page not mapped yet, keeps the contents
    CHECK(info(7) == 0x3000 && LD8(alias + 0x3000) == 1);
    CHECK(svc(0x2D, alias + 0x2000, 0x3000, 0) == 0);
    a = region(RT_MEM_ALIAS);
    CHECK(a.committed == 0 && a.mapped == 0 && info(7) == 0);
    q = rt_mem_query(alias + 0x3000);
    CHECK(q.base == alias && q.size == 1024 * MiB && q.state == 0);
    CHECK(svc(0x2C, alias + 0x10, 0x1000, 0) == 0xcc01 && svc(0x2C, alias - 0x1000, 0x2000, 0) == 0xdc01);
}

static void case_stack(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    uint64_t heap = info(4), stack = info(14), src = heap + 0x10000, dst = stack + 0x100000;
    for (int i = 0; i < 0x2000; i += 8) ST64(src + (uint64_t)i, 0x5000 + (uint64_t)i);   // one committed chunk
    CHECK(svc(0x04, dst, src, 0x10000) == 0);
    CHECK(region(RT_MEM_STACK).committed == 0x10000 && region(RT_MEM_HEAP).committed == 0);
    CHECK(LD64(dst + 0x1ff8) == 0x5000 + 0x1ff8 && LD64(dst + 0x8000) == 0);
    RtMemInfo q = rt_mem_query(dst + 0x10);
    CHECK(q.base == dst && q.size == 0x10000 && q.state == 0xB && q.perm == 3);
    q = rt_mem_query(src);
    CHECK(q.base == src && q.size == 0x10000 && q.state == 5 && q.perm == 0 && q.attr == 1);
    q = rt_mem_query(heap);
    CHECK(q.base == heap && q.size == 0x10000 && q.state == 5);   // the heap run stops at the moved source
    CHECK(svc(0x04, dst, src, 0x10000) == 0xd401);              // already moved
    ST64(dst + 0x8000, 77);
    CHECK(svc(0x05, dst, src, 0x10000) == 0);
    CHECK(LD64(src + 0x1ff8) == 0x5000 + 0x1ff8 && LD64(src + 0x8000) == 77);
    CHECK(region(RT_MEM_STACK).committed == 0);
}

static void case_load(const char *dir) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    load(dir, V_GOOD, MOD_MAIN_SIZE, NULL);
    uint64_t lj = rt_hostfn_bind("longjmp");
    const uint64_t md = MB + 0, sd = SB + 0;
    CHECK(lj && rt_hostfn_name(LD64(md + SLOT(7))) && !strcmp(rt_hostfn_name(LD64(md + SLOT(7))), MISSING));
    CHECK(LD64(md + SLOT(0)) == MB + 0x1234 && LD64(md + SLOT(1)) == SB + 0x308 && LD64(md + SLOT(2)) == lj);
    CHECK(LD64(md + SLOT(3)) == 0 && LD64(md + SLOT(4)) == SB + DATA_OFF + 0x900 && LD64(md + SLOT(5)) == MB + 0x200);
    CHECK(LD64(md + SLOT(6)) == SB + 0x300 && LD64(md + BSS_OFF + 0x1000) == MB + 0x40);
    CHECK(LD64(md + SLOT(8)) == MB + 0x80 && LD64(md + SLOT(9)) == MB + 0x90 && LD64(md + SLOT(10)) == 0);
    CHECK(LD64(md + SLOT(11)) == MB + 0xB0 && LD64(md + SLOT(71)) == MB + 0x710 && LD64(md + SLOT(72)) == MB + 0x720);
    CHECK(LD64(sd + SLOT(0)) == lj && LD64(sd + SLOT(1)) == SB + 0x300);
    const RtLoadStats *m = rt_load_stats(0), *s = rt_load_stats(1);
    CHECK(m->relative == 2 && m->relr == 5 && m->abs64 == 1 && m->glob_dat == 4 && m->jump_slot == 2);
    CHECK(m->imports == 5 && m->imports_to[0] == 0 && m->imports_to[1] == 2 && m->imports_to_host == 1);
    CHECK(m->imports_weak == 1 && m->imports_missing == 1 && m->replaced == 0);
    CHECK(s->glob_dat == 1 && s->jump_slot == 1 && s->imports == 0 && s->replaced == 1);
    CHECK(rt_load_replaced_count() == 1 && !strcmp(rt_load_replaced(0)->symbol, "longjmp") && rt_load_replaced(0)->guest == SB + 0x400);
    CHECK(rt_load_replaced(0)->host == lj && rt_symbol("sdk_func") == SB + 0x300 && rt_symbol("nowhere") == 0);
    const RtLoaded *l = rt_loaded(0);
    CHECK(l && l->size == IMAGE && l->data_off == DATA_OFF && l->bss_off == BSS_OFF && l->init == MB + 0x200);
    CHECK(l->build_id[0] == 0xAB && rt_loaded(2) == NULL);
    // main's bss chunk was committed by the relocation into it; sdk's bss is still untouched
    CHECK(region(RT_MEM_MODULES).committed == 0x4000 + 0x4000 + 0x4000 && region(RT_MEM_MODULES).mapped == 2 * IMAGE);
    CHECK(LD64(SB + BSS_OFF + 0x2000) == 0 && region(RT_MEM_MODULES).committed == 0x10000);
    RtMemInfo q = rt_mem_query(MB + 0x10);
    CHECK(q.base == MB && q.size == RO_OFF && q.state == 3 && q.perm == 5);
    q = rt_mem_query(MB + RO_OFF);
    CHECK(q.base == MB + RO_OFF && q.size == DATA_OFF - RO_OFF && q.state == 3 && q.perm == 1);
    q = rt_mem_query(MB + DATA_OFF + 0x10);
    CHECK(q.base == MB + DATA_OFF && q.size == IMAGE - DATA_OFF && q.state == 4 && q.perm == 3);
    q = rt_mem_query(MB + IMAGE);
    CHECK(q.base == MB + IMAGE && q.base + q.size == SB && q.state == 0x10);
    q = rt_mem_query(RT_HOSTFN_BASE + 0x40);
    CHECK(q.base == RT_HOSTFN_BASE && q.state == 0x10);
    q = rt_mem_query(0x1000);
    CHECK(q.base == 0 && q.size == 0x8000000 && q.state == 0x10);
    q = rt_mem_query(1ull << 40);
    CHECK(q.base == 1ull << 39 && q.state == 0x10);
    // RELRO: read-only, then locked for good
    CHECK(svc(0x02, MB + DATA_OFF, 0x1000, 1) == 0);
    q = rt_mem_query(MB + DATA_OFF);
    CHECK(q.size == 0x1000 && q.perm == 1 && LD64(MB + DATA_OFF) != 0);
    c.x[3] = 16;
    CHECK(svc(0x03, MB + DATA_OFF, 0x1000, 16) == 0);
    CHECK(rt_mem_query(MB + DATA_OFF).attr == 16 && svc(0x02, MB + DATA_OFF, 0x1000, 3) == 0xd401);
    CHECK(svc(0x02, MB + BSS_OFF + 0x3000, 0x1000, 0) == 0 && rt_mem_query(MB + BSS_OFF + 0x3000).perm == 0);
    CHECK(info(7) == 2 * IMAGE);
    // QueryMemory through the SVC, into a heap page not touched yet
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);
    uint64_t out = info(4) + 0x1000;
    CHECK(svc(0x06, out, 0, MB + 0x10) == 0 && LD64(out) == MB && LD64(out + 8) == RO_OFF && LD32(out + 16) == 3);
    CHECK(LD32(out + 24) == 5);
    rt_load_report();
}

static void case_io(const char *dir) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    load(dir, V_GOOD, MOD_MAIN_SIZE, NULL);
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);
    uint64_t heap = info(4);
    CHECK(rt_mem_commit_io(heap + 0x20000, 0x3000, 1) && region(RT_MEM_HEAP).committed == 0x10000);
    CHECK(!rt_mem_commit_io(heap + 2 * MiB - 0x1000, 0x2000, 1));   // its second page is past the heap
    CHECK(!rt_mem_commit_io(MB + 0x10, 8, 1) && rt_mem_commit_io(MB + 0x10, 8, 0));   // text: read only
    CHECK(rt_mem_commit_io(SB + BSS_OFF + 0x10, 8, 1));
    CHECK(!rt_mem_commit_io(info(2), 0x1000, 0));   // an alias page nothing mapped
    uint64_t pages[16];
    ST8(heap + 0x5000, 1);
    uint64_t n = rt_mem_written(heap, 0x10000, pages, 16, 1);
    CHECK(n == UINT64_MAX || (n >= 1 && n <= 16));
    CHECK(rt_mem_written(MB, 0x1000, pages, 16, 0) == UINT64_MAX);   // modules are not watched
}

// The write watch over heap and alias (architecture D14, spike S5): a query reports the pages written in its range since
// the last reset, in address order; without reset it reports them again; it never reports a page outside its range, and
// its reset leaves those marked; a range that is not inside heap or alias is UINT64_MAX. Windows only: elsewhere the
// platform has no watch, and the case says it skipped.
static void case_writewatch(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    const uint64_t heap = info(4), alias = info(2);
    uint64_t pages[64];
    ST8(heap + 0x5000, 1);   // committed by its first touch, and written
    uint64_t n = rt_mem_written(heap, 4 * MiB, pages, 64, 1);
    if (n == UINT64_MAX) {
        printf("skip writewatch: write watch not available on this platform\n");
        exit(0);
    }
    CHECK(n == 1 && pages[0] == heap + 0x5000);
    // a query over no written page reports none, without and with reset
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 0);
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 1) == 0);
    // a query without reset reports the same pages again, in address order
    ST8(heap + 0x23000, 2);
    ST8(heap + 0x5000, 3);
    for (int k = 0; k < 2; k++) {
        n = rt_mem_written(heap, 4 * MiB, pages, 64, 0);
        CHECK(n == 2 && pages[0] == heap + 0x5000 && pages[1] == heap + 0x23000);
    }
    // a query over a range with no written page reports none even with pages written beside it; a query over a
    // sub-range sees its own page, and its reset leaves the page outside it marked
    CHECK(rt_mem_written(heap + 0x10000, 0x10000, pages, 64, 1) == 0);
    n = rt_mem_written(heap + 0x20000, 0x10000, pages, 64, 1);
    CHECK(n == 1 && pages[0] == heap + 0x23000);
    n = rt_mem_written(heap, 4 * MiB, pages, 64, 0);
    CHECK(n == 1 && pages[0] == heap + 0x5000);
    CHECK(rt_mem_written(heap, 4 * MiB, pages, 64, 1) == 1 && rt_mem_written(heap, 4 * MiB, pages, 64, 0) == 0);
    // the alias region is watched too
    CHECK(svc(0x2C, alias, 0x10000, 0) == 0);
    ST8(alias + 0x3000, 4);
    n = rt_mem_written(alias, 0x10000, pages, 64, 1);
    CHECK(n == 1 && pages[0] == alias + 0x3000);
    // outside heap and alias: the stack region, a range running from the heap region into the alias region, the
    // module region, and low host memory
    CHECK(rt_mem_written(info(14), 0x1000, pages, 64, 0) == UINT64_MAX);
    CHECK(rt_mem_written(heap + info(5) - 0x1000, 0x2000, pages, 64, 0) == UINT64_MAX);
    CHECK(rt_mem_written(RT_MODULE_BASE, 0x1000, pages, 64, 0) == UINT64_MAX);
    CHECK(rt_mem_written(0x10000, 0x1000, pages, 64, 0) == UINT64_MAX);
}

// The commit query and the commit epoch (architecture D14, for the GPU back end's host imports): a run of pages in one
// state from addr, clipped to the size asked and to the region; the epoch moves once for each change of the page map
// (a first touch by a write or a read, host I/O, a heap that shrinks) and never for an access to committed pages. The
// decommit record moves on the decommits alone and names the ranges they took, through a commit of the same pages again.
static void case_commitrun(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    const uint64_t heap = info(4), region = info(5);
    int st = -1;
    const uint64_t e0 = rt_mem_commit_epoch(), d0 = rt_mem_decommit_epoch();
    CHECK(rt_mem_commit_run(heap, 4 * MiB, &st) == 4 * MiB && st == 0);
    ST32(heap + 0x10010, 1);   // commits the chunk [0x10000, 0x20000)
    CHECK(rt_mem_commit_epoch() == e0 + 1);
    CHECK(rt_mem_commit_run(heap, 4 * MiB, &st) == 0x10000 && st == 0);
    CHECK(rt_mem_commit_run(heap + 0x10000, 4 * MiB, &st) == 0x10000 && st == 1);
    CHECK(rt_mem_commit_run(heap + 0x18000, 0x1000, &st) == 0x1000 && st == 1);    // up to the size asked
    CHECK(rt_mem_commit_run(heap + 0x18800, 4 * MiB, &st) == 0x7800 && st == 1);   // from addr, not from its page
    CHECK(rt_mem_commit_run(heap + 0x20000, 4 * MiB, &st) == 4 * MiB && st == 0);
    CHECK(rt_mem_commit_run(heap + region - 0x1000, 4 * MiB, &st) == 0x1000 && st == 0);   // up to the region's end
    CHECK(rt_mem_commit_run(heap, 0, &st) == 0);
    CHECK(rt_mem_commit_run(0x7f00000000ull, 0x1000, &st) == UINT64_MAX);   // in no region
    CHECK(LD32(heap + 0x10010) == 1);   // an access to committed pages changes nothing
    CHECK(rt_mem_commit_epoch() == e0 + 1);
    CHECK(LD32(heap + 0x300000) == 0);   // a first touch by a read commits too
    CHECK(rt_mem_commit_epoch() == e0 + 2);
    CHECK(rt_mem_commit_io(heap + 0x100000, 0x1000, 1));   // host I/O
    CHECK(rt_mem_commit_epoch() == e0 + 3);
    CHECK(rt_mem_commit_run(heap + 0x100000, 0x10000, &st) == 0x10000 && st == 1);
    CHECK(rt_mem_decommit_epoch() == d0 && !rt_mem_decommitted(d0, heap, heap + 4 * MiB));   // commits are no decommit
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);   // the heap shrinks over the chunk at 3 MiB
    CHECK(rt_mem_commit_epoch() == e0 + 4);
    CHECK(rt_mem_commit_run(heap + 2 * MiB, 2 * MiB, &st) == 2 * MiB && st == 0);
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);   // nothing past 2 MiB is committed now: no change, no move
    CHECK(rt_mem_commit_epoch() == e0 + 4);
    // The decommit record: the shrink is one decommit, of the range it gave back; pages beside it, and anything after it,
    // were not decommitted
    CHECK(rt_mem_decommit_epoch() == d0 + 1);
    CHECK(rt_mem_decommitted(d0, heap + 0x300000, heap + 0x301000));
    CHECK(rt_mem_decommitted(d0, heap + 2 * MiB - 0x1000, heap + 2 * MiB + 0x1000));
    CHECK(!rt_mem_decommitted(d0, heap + 0x10000, heap + 0x20000));
    CHECK(!rt_mem_decommitted(d0, heap, heap + 2 * MiB));
    CHECK(!rt_mem_decommitted(d0 + 1, heap + 0x300000, heap + 0x301000));
    // committed again (the heap grows back, a write touches the chunk): still decommitted since d0, which an import made
    // before the shrink needs to know
    CHECK(svc(0x01, 0, 4 * MiB, 0) == 0);
    ST32(heap + 0x300000, 5);
    CHECK(rt_mem_commit_run(heap + 0x300000, 0x1000, &st) == 0x1000 && st == 1);
    CHECK(rt_mem_decommit_epoch() == d0 + 1 && rt_mem_decommitted(d0, heap + 0x300000, heap + 0x301000));
    // The record keeps the last 1024: a reader further behind is told decommitted, whatever the range, and one less far
    // behind is told the truth
    for (int i = 0; i < 1024; i++) {
        ST32(heap + 2 * MiB + (uint64_t)(i % 32) * 0x10000, 6);   // another chunk each time: the same one would be a fault loop
        CHECK(svc(0x01, 0, 2 * MiB, 0) == 0 && svc(0x01, 0, 4 * MiB, 0) == 0);
    }
    CHECK(rt_mem_decommit_epoch() == d0 + 1025);
    CHECK(rt_mem_decommitted(d0, heap + 0x10000, heap + 0x20000));
    CHECK(!rt_mem_decommitted(d0 + 2, heap + 0x10000, heap + 0x20000));
    CHECK(rt_mem_decommitted(d0 + 2, heap + 0x300000, heap + 0x301000));
}

// Watched reservations in pieces (Windows; plat.h, PLAT_PIECE): the alias region is made of allocations that start at
// multiples of PLAT_PIECE, and host I/O, the write watch (with a buffer too small for both pages, the write-watch study) and
// UnmapPhysicalMemory work across a piece boundary as across any other page. Elsewhere the case says it skipped.
static void case_pieces(void) {
#ifndef _WIN32
    printf("skip pieces: reservations are made in one piece on this platform\n");
    exit(0);
#else
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    const uint64_t alias = info(2), b = (alias + PLAT_PIECE) & ~(PLAT_PIECE - 1);
    CHECK(b - alias < info(3));
    MEMORY_BASIC_INFORMATION m;
    CHECK(VirtualQuery((void *)(uintptr_t)(b + 0x1000), &m, sizeof m) == sizeof m);
    CHECK((uint64_t)(uintptr_t)m.AllocationBase == b && m.State == MEM_RESERVE);
    CHECK(VirtualQuery((void *)(uintptr_t)(b - 0x1000), &m, sizeof m) == sizeof m);
    CHECK((uint64_t)(uintptr_t)m.AllocationBase != b && (uint64_t)(uintptr_t)m.AllocationBase >= alias);
    CHECK(svc(0x2C, b - 0x10000, 0x20000, 0) == 0);
    CHECK(rt_mem_commit_io(b - 0x8000, 0x10000, 1));   // one call across the boundary
    int st = -1;
    CHECK(rt_mem_commit_run(b - 0x10000, 0x20000, &st) == 0x20000 && st == 1);
    ST32(b - 4, 7);
    ST32(b, 9);
    uint64_t pages[8];
    CHECK(rt_mem_written(b - 0x10000, 0x20000, pages, 8, 1) == 2 && pages[0] == b - 0x1000 && pages[1] == b);
    ST32(b - 4, 8);
    ST32(b, 10);
    CHECK(rt_mem_written(b - 0x10000, 0x20000, pages, 1, 1) == 1 && pages[0] == b - 0x1000);
    CHECK(rt_mem_written(b - 0x10000, 0x20000, pages, 8, 1) == 1 && pages[0] == b);   // left marked for the next call
    CHECK(svc(0x2D, b - 0x10000, 0x20000, 0) == 0);   // decommitted across the boundary
    CHECK(rt_mem_commit_run(b - 0x10000, 0x20000, &st) == 0x20000 && st == 0);
    CHECK(VirtualQuery((void *)(uintptr_t)b, &m, sizeof m) == sizeof m && m.State == MEM_RESERVE);
#endif
}

// ---- failing cases: each must end in one named fatal
static void touch(uint64_t a) { printf("read %#llx\n", (unsigned long long)LD64(a)); }

static void case_touch_heap(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);
    touch(info(4) + 3 * MiB);
}

static void case_touch_alias(int after_unmap) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    uint64_t alias = info(2);
    if (after_unmap) {
        CHECK(svc(0x2C, alias, 0x10000, 0) == 0);
        ST8(alias + 0x5000, 1);
        CHECK(svc(0x2D, alias, 0x10000, 0) == 0);
    }
    ST8(alias + 0x5000, 1);
}

static void case_touch_moved(void) {
    RtMemConfig cfg = test_config();
    rt_mem_init(&cfg);
    CHECK(svc(0x01, 0, 2 * MiB, 0) == 0);
    CHECK(svc(0x04, info(14), info(4), 0x10000) == 0);
    touch(info(4) + 0x100);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 3) {
        fprintf(stderr, "usage: memory_test <case> <scratch dir>\n");
        return 2;
    }
    const char *cs = argv[1], *dir = argv[2];
    RtMemConfig cfg = test_config();
    if (!strcmp(cs, "sha256")) case_sha256();
    else if (!strcmp(cs, "heap")) case_heap();
    else if (!strcmp(cs, "pool")) case_pool();
    else if (!strcmp(cs, "alias")) case_alias();
    else if (!strcmp(cs, "stack")) case_stack();
    else if (!strcmp(cs, "load")) case_load(dir);
    else if (!strcmp(cs, "io")) case_io(dir);
    else if (!strcmp(cs, "writewatch")) case_writewatch();
    else if (!strcmp(cs, "commitrun")) case_commitrun();
    else if (!strcmp(cs, "pieces")) case_pieces();
    else if (!strcmp(cs, "touch_heap")) case_touch_heap();
    else if (!strcmp(cs, "touch_alias")) case_touch_alias(0);
    else if (!strcmp(cs, "touch_unmapped_again")) case_touch_alias(1);
    else if (!strcmp(cs, "touch_stack")) { rt_mem_init(&cfg); touch(info(14) + 0x1000); }
    else if (!strcmp(cs, "touch_moved")) case_touch_moved();
    else if (!strcmp(cs, "touch_free")) { rt_mem_init(&cfg); touch(0x7f00000000ull); }
    else if (!strcmp(cs, "touch_hostfn")) { rt_mem_init(&cfg); touch(RT_HOSTFN_BASE + 0x20); }
    else if (!strcmp(cs, "write_text")) { rt_mem_init(&cfg); load(dir, V_GOOD, MOD_MAIN_SIZE, NULL); ST32(MB + 0x10, 1); }
    else if (!strcmp(cs, "write_relro")) {
        rt_mem_init(&cfg);
        load(dir, V_GOOD, MOD_MAIN_SIZE, NULL);
        CHECK(svc(0x02, MB + DATA_OFF, 0x1000, 1) == 0);
        ST64(MB + DATA_OFF + 8, 1);
    } else if (!strcmp(cs, "missing_import_call")) {
        rt_mem_init(&cfg);
        load(dir, V_GOOD, MOD_MAIN_SIZE, NULL);
        static const uint64_t ma[] = {MB + 0x200}, sa[] = {SB + 0x300};
        static const GuestFn f[] = {fake_fn};
        const RtModule mods[] = {{"main", MB, MOD_MAIN_SIZE, ma, f, 1, TEST_END}, {"sdk", SB, MOD_SDK_SIZE, sa, f, 1, TEST_END}};
        rt_dispatch_init(mods, 2);
        rt_call(&c, LD64(MB + SLOT(7)));
    } else if (!strcmp(cs, "load_compressed")) { rt_mem_init(&cfg); load(dir, V_COMPRESSED, MOD_MAIN_SIZE, NULL); }
    else if (!strcmp(cs, "load_corrupt")) { rt_mem_init(&cfg); load(dir, V_CORRUPT, MOD_MAIN_SIZE, NULL); }
    else if (!strcmp(cs, "load_size")) { rt_mem_init(&cfg); load(dir, V_GOOD, MOD_MAIN_SIZE + 0x1000, NULL); }
    else if (!strcmp(cs, "load_sha256")) { rt_mem_init(&cfg); load(dir, V_GOOD, MOD_MAIN_SIZE, "00"); }
    else if (!strcmp(cs, "load_reloc_outside")) { rt_mem_init(&cfg); load(dir, V_RELOC_OUTSIDE, MOD_MAIN_SIZE, NULL); }
    else if (!strcmp(cs, "load_reloc_type")) { rt_mem_init(&cfg); load(dir, V_RELOC_TYPE, MOD_MAIN_SIZE, NULL); }
    else if (!strcmp(cs, "load_before_init")) load(dir, V_GOOD, MOD_MAIN_SIZE, NULL);
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
