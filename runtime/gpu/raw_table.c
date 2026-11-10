// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The page table's builder (oot2/vk_raw.h): runs of the backing map into the words of oot2/raw_table.h. No device and no
// lock: the Vulkan side (vk_raw.c) calls it into host memory and copies the words into a copy of the table.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/runtime.h"
#include "oot2/vk_raw.h"

#define CHUNK (1ull << RAW_CHUNK_LOG2)
#define PAGE (1ull << RAW_PAGE_LOG2)
#define LEAF_ENTRIES (1u << RAW_LEAF_ENTRIES_LOG2)

typedef unsigned long long ull;

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

// The chunks the runs touch, sorted and distinct; the count in *count. NULL with *count 0 for none.
static uint64_t *chunks_of(const BackingRun *runs, int n, uint64_t *count) {
    uint64_t total = 0;
    for (int i = 0; i < n; i++)
        if (runs[i].hi > runs[i].lo) total += ((runs[i].hi - 1) >> RAW_CHUNK_LOG2) - (runs[i].lo >> RAW_CHUNK_LOG2) + 1;
    *count = 0;
    if (!total) return NULL;
    uint64_t *c = malloc((size_t)total * sizeof *c), k = 0;
    if (!c) rt_fatal("gfx: vulkan: raw path: out of memory for %llu chunks", (ull)total);
    for (int i = 0; i < n; i++)
        for (uint64_t ch = runs[i].lo >> RAW_CHUNK_LOG2; runs[i].hi > runs[i].lo && ch <= (runs[i].hi - 1) >> RAW_CHUNK_LOG2;
             ch++)
            c[k++] = ch;
    qsort(c, (size_t)k, sizeof *c, cmp_u64);
    uint64_t d = 0;
    for (uint64_t i = 0; i < k; i++)
        if (!d || c[d - 1] != c[i]) c[d++] = c[i];
    *count = d;
    return c;
}

// The leaf block (from 1) of chunk ch in the sorted list c[0..n).
static uint32_t block_of(const uint64_t *c, uint64_t n, uint64_t ch) {
    uint64_t a = 0, b = n;
    while (a < b) {
        uint64_t m = (a + b) / 2;
        if (c[m] < ch) a = m + 1;
        else b = m;
    }
    return (uint32_t)(a + 1);
}

int raw_layout(const BackingRun *runs, int n, RawLayout *l, char *why, size_t why_size) {
    return raw_layout_span(runs, n, 0, 0, l, why, why_size);
}

int raw_layout_span(const BackingRun *runs, int n, uint32_t span_base, uint32_t span_count, RawLayout *l, char *why,
                    size_t why_size) {
    memset(l, 0, sizeof *l);
    for (int i = 0; i < n; i++) {
        const BackingRun *r = &runs[i];
        if ((r->lo | r->hi) & (PAGE - 1) || r->hi < r->lo) {
            snprintf(why, why_size, "the run [%#llx, %#llx) is not whole 4 KiB pages", (ull)r->lo, (ull)r->hi);
            return 0;
        }
        if ((r->address - r->lo) & (PAGE - 1)) {
            snprintf(why, why_size, "the run [%#llx, %#llx) lies at device address %#llx: its delta is not a multiple of "
                     "4 KiB, so bit 0 of its entries would not be free", (ull)r->lo, (ull)r->hi, (ull)r->address);
            return 0;
        }
        l->pages += (r->hi - r->lo) / PAGE;
        if (r->kind == BACKING_MIRROR) l->mirror_pages += (r->hi - r->lo) / PAGE;
    }
    uint64_t nc;
    uint64_t *c = chunks_of(runs, n, &nc);
    if (nc > RAW_LEAF_BLOCK_CAPACITY) {
        snprintf(why, why_size, "%llu chunks of 2 MiB are mapped, more than the %u leaf blocks of the table (%llu GiB)",
                 (ull)nc, RAW_LEAF_BLOCK_CAPACITY, (ull)((uint64_t)RAW_LEAF_BLOCK_CAPACITY * CHUNK >> 30));
        free(c);
        return 0;
    }
    uint64_t base = nc ? c[0] : 0, count = nc ? c[nc - 1] - c[0] + 1 : 1;
    if (span_count) {
        if (nc && (c[0] < span_base || c[nc - 1] >= (uint64_t)span_base + span_count)) {
            snprintf(why, why_size, "the chunks %#llx to %#llx of the runs reach outside the span %#llx+%#llx given",
                     (ull)c[0], (ull)c[nc - 1], (ull)span_base, (ull)span_count);
            free(c);
            return 0;
        }
        base = span_base, count = span_count;
    }
    free(c);
    if (base > UINT32_MAX || count > UINT32_MAX) {
        snprintf(why, why_size, "the span of chunks %#llx+%#llx does not fit the push words", (ull)base, (ull)count);
        return 0;
    }
    l->span_base = (uint32_t)base, l->span_count = (uint32_t)count, l->blocks = (uint32_t)nc;
    l->words = RAW_TOP_WORD + count + (uint64_t)nc * RAW_LEAF_BLOCK_WORDS;
    return 1;
}

void raw_write(const BackingRun *runs, int n, const RawLayout *l, uint32_t *words, RawMirrorPage *pages, uint64_t *guest) {
    memset(words, 0, (size_t)l->words * 4);
    words[RAW_HEADER_VERSION] = RAW_VERSION;
    words[RAW_HEADER_SPAN_BASE] = l->span_base;
    words[RAW_HEADER_SPAN_COUNT] = l->span_count;
    words[RAW_HEADER_LEAF_BLOCKS] = l->blocks;
    uint64_t nc;
    uint64_t *c = chunks_of(runs, n, &nc);
    uint32_t *top = words + RAW_TOP_WORD, *leaf = top + l->span_count;
    for (uint64_t i = 0; i < nc; i++) top[c[i] - l->span_base] = (uint32_t)(i + 1);
    uint64_t m = 0;
    for (int i = 0; i < n; i++) {
        const BackingRun *r = &runs[i];
        uint64_t delta = (r->address - r->lo) | 1;   // a multiple of 4 KiB (raw_layout), so bit 0 is free
        uint32_t first = 0;   // the first slot of a's chunk, looked up once a chunk
        for (uint64_t a = r->lo; a < r->hi; a += PAGE) {
            if (a == r->lo || !(a & (CHUNK - 1))) first = (block_of(c, nc, a >> RAW_CHUNK_LOG2) - 1) * LEAF_ENTRIES;
            uint32_t slot = first + (uint32_t)((a >> RAW_PAGE_LOG2) & (LEAF_ENTRIES - 1));
            leaf[(uint64_t)slot * RAW_ENTRY_WORDS] = (uint32_t)delta;
            leaf[(uint64_t)slot * RAW_ENTRY_WORDS + 1] = (uint32_t)(delta >> 32);
            if (r->kind != BACKING_MIRROR) continue;
            pages[m] = (RawMirrorPage){slot, 0, r->address + (a - r->lo), 0, 0};
            guest[m++] = r->guest + (a - r->lo);
        }
    }
    free(c);
}
