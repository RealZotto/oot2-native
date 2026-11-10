// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The raw path on the device (oot2/vk_raw.h, architecture D15): the copies of the page table, the sinks the write-back
// goes through into guest memory, the out buffers of the write-back, the runtime shader raw_writeback.comp, and the effect
// that logs misses and stamps the raw stores the GPU wrote into guest memory.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/capture.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/nvn_pace.h"
#include "oot2/pagegen.h"
#include "oot2/raw_diag.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/vk_raw.h"
#include "plat.h"

#define PAGE 0x1000ull
#define RAW_COPY_ROOM (4ull << 20)   // what a copy has beyond its table for the map to grow in a submission (refresh)
#define OUT_HEAD 16u   // raw_writeback.comp: the table's head is copied from this word of the out buffer
#define HEAD_WORDS (RAW_MISS_LOG_WORD + RAW_MISS_ENTRIES * RAW_MISS_ENTRY_WORDS)

typedef unsigned long long ull;

// Memory with one buffer over it, mapped, and the buffer's device address.
typedef struct Mem {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    uint64_t address, size;
} Mem;

typedef struct Copy {
    Mem m;
    uint64_t version;    // the map's version it was built at; 0 never built
    uint64_t last_use;   // the newest submission that read it
    uint64_t prev_use;   // the submission that read it before last_use did, when last_use took it as it stood (else 0)
    RawLayout l;
    uint64_t pages_at;   // byte offset of the mirrored pages' list
    uint64_t *guest;     // the guest address of each mirrored page (host memory)
    uint64_t *made;      // the decommit epoch each one's sink was made at (Sink.made), NO_SINK for none (host memory)
    uint64_t unsunk;     // how many have none
    uint64_t sink_gen;   // the sinks' generation it was built at (g_sink_gen)
} Copy;

typedef struct Out {
    Mem m;
    uint32_t pages;   // mirrored pages its bitmap holds
    struct Out *next;
} Out;

// A sink: a host import over committed guest pages [lo, hi), through which the write-back's shader writes the guest's own
// pages.
typedef struct Sink {
    BackingMem m;
    uint64_t lo, hi;
    uint64_t made;   // the decommit epoch it holds its pages since: read before its import, moved up by each scrub it passes
    uint64_t gone;   // a dead sink: freed once this submission is done
    uint64_t seen;   // the build that last mapped a page to it
} Sink;

#define NO_SINK UINT64_MAX   // Copy.made of a mirrored page with no sink

// What an end hands its effect.
typedef struct Done {
    Out *out;
    const uint64_t *guest;
    const uint64_t *made;
    uint64_t pages;      // the mirrored pages
    uint64_t since;      // the page generation every mirror was current at (pagegen_mark_gpu_direct)
    uint64_t scrubbed;   // the decommit epoch up to which the begin found every live sink's pages its own (sinks_scrub)
    // The miss diagnostics (setting raw_diag); NULL and zero with it off
    struct Snaps *snaps;   // the snapshots of the submission's stages
    RawLayout layout;      // the table the submission ended with, and the map version it was built at
    uint64_t table_version;
} Done;

static Copy g_copy[RAW_COPIES_MAX];
static int g_ncopies;
static Copy *g_current;              // the copy the newest submission used
static uint64_t g_value;             // the submission that began and has not ended (one at a time: the gate)
static uint32_t g_push[4];
static uint32_t *g_image;            // the build's host image
static uint64_t g_image_bytes;
static RawMirrorPage *g_pages;
static uint64_t *g_guest_scratch, *g_made_scratch;
static uint64_t g_pages_cap, g_guest_cap, g_made_cap;
static VkShaderModule g_module;
static VkPipelineLayout g_layout;
static VkPipeline g_pipe;
static PlatMutex g_lock = PLAT_MUTEX_INIT;   // the out buffers' free list and the counters
static Out *g_free;
static GpuVkRawStats g_st;
static const BackingOps *g_ops = &gpu_vk_backing_ops;   // what the sinks are made and freed with (gpu_vk_raw_set_ops)
static Sink *g_sinks;                // live, sorted by address, disjoint
static int g_nsinks, g_sinks_cap;
static Sink *g_dead;                 // dropped, waiting for the submissions that may use them
static int g_ndead, g_dead_cap;
static uint64_t g_sink_gen = 1;      // moves when a sink is made or dropped: a table built before names sinks that may be gone
static uint64_t g_sink_epoch;        // the decommit epoch the live sinks were last checked at
static uint64_t g_scrubbed;          // the same, read by the begin of the submission being recorded (Done.scrubbed)
static uint64_t g_sinks_version;     // the map's version sinks_prepare last made sinks for (0: never)
static uint64_t g_build_id;          // counts the table images built (Sink.seen)
static uint64_t g_since;             // the page generation every mirror was current at when the submission began

static void diag_table(const RawLayout *l, uint64_t version, uint64_t build);   // the miss diagnostics, below
static void diag_drain(void);

// ---------- memory
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// size bytes of memory of the first type that has every flag of one of the wants, in order, with a buffer the shaders
// reach by address; mapped.
static Mem mem_new(uint64_t size, const VkMemoryPropertyFlags *wants, int nwants, const char *what) {
    Mem m = {.size = size};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &m.buf), "vkCreateBuffer (raw path)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, m.buf, &mr);
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    for (int i = 0; i < nwants && r != VK_SUCCESS; i++) {
        uint32_t type = memory_type(mr.memoryTypeBits, wants[i]);
        if (type == UINT32_MAX) continue;
        VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi,
                                      .allocationSize = mr.size, .memoryTypeIndex = type};
        r = vkAllocateMemory(gpu_vk.device, &alloc, NULL, &m.mem);
        if (r == VK_ERROR_DEVICE_LOST) gpu_vk_failed(r, "vkAllocateMemory (raw path)");
    }
    if (r != VK_SUCCESS)
        rt_fatal("gfx: vulkan: raw path: no host-visible memory for %s of %llu KiB (%s)", what, (ull)(size >> 10),
                 gpu_vk_result_name(r));
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, m.buf, m.mem, 0), "vkBindBufferMemory (raw path)");
    void *p;
    GPU_VK_OK(vkMapMemory(gpu_vk.device, m.mem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory (raw path)");
    m.map = p;
    VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = m.buf};
    m.address = vkGetBufferDeviceAddress(gpu_vk.device, &di);
    return m;
}

static void mem_free(Mem *m) {
    if (!m->buf) return;
    vkDestroyBuffer(gpu_vk.device, m->buf, NULL);
    vkFreeMemory(gpu_vk.device, m->mem, NULL);
    memset(m, 0, sizeof *m);
}

static const VkMemoryPropertyFlags k_table[] = {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
static const VkMemoryPropertyFlags k_out[] = {
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};

// ---------- the table's copies
static void wait_for(uint64_t value) {
    if (value <= gpu_sync_done()) return;
    int token = rt_kernel_block_begin();
    gpu_sync_wait(value, UINT64_MAX);
    rt_kernel_block_end(token);
}

// A copy no submission in flight reads, other than the current one: a done one, else one already issued (waited for),
// else a new one. The current copy is kept, since the submission being recorded may follow one that still reads it.
static Copy *copy_free(void) {
    uint64_t done = gpu_sync_done(), issued = gpu_sync_issued();
    Copy *wait = NULL;
    for (int i = 0; i < g_ncopies; i++) {
        Copy *c = &g_copy[i];
        if (c == g_current) continue;
        if (c->last_use <= done) return c;
        if (c->last_use <= issued && (!wait || c->last_use < wait->last_use)) wait = c;
    }
    if (g_ncopies < RAW_COPIES_MAX) return &g_copy[g_ncopies++];
    if (!wait)
        rt_fatal("gfx: vulkan: raw path: every one of the %d copies of the page table is read by a submission that is "
                 "not issued yet", RAW_COPIES_MAX);
    plat_mutex_lock(&g_lock);
    g_st.copy_waits++;
    plat_mutex_unlock(&g_lock);
    wait_for(wait->last_use);
    return wait;
}

static void *grow_to(void *p, uint64_t *cap, uint64_t need, const char *what) {
    if (need <= *cap) return p;
    uint64_t c = need + need / 4;
    void *q = realloc(p, (size_t)c);
    if (!q) rt_fatal("gfx: vulkan: raw path: out of memory for %s (%llu KiB)", what, (ull)(c >> 10));
    *cap = c;
    return q;
}

// ---------- the sinks (oot2/vk_raw.h, The sinks)
static void *sinks_room(void *p, int *cap, int need, size_t size) {
    if (need <= *cap) return p;
    int c = *cap ? *cap * 2 : 8;
    while (c < need) c *= 2;
    void *q = realloc(p, (size_t)c * size);
    if (!q) rt_fatal("gfx: vulkan: raw path: out of memory for %d sinks", c);
    *cap = c;
    return q;
}

// The index of the first live sink that ends above addr.
static int sink_after(uint64_t addr) {
    int a = 0, b = g_nsinks;
    while (a < b) {
        int m = (a + b) / 2;
        if (g_sinks[m].hi <= addr) a = m + 1;
        else b = m;
    }
    return a;
}

static Sink *sink_at(uint64_t addr) {
    int i = sink_after(addr);
    return i < g_nsinks && g_sinks[i].lo <= addr ? &g_sinks[i] : NULL;
}

static void sinks_count(void) {
    uint64_t bytes = 0;
    for (int i = 0; i < g_nsinks; i++) bytes += g_sinks[i].hi - g_sinks[i].lo;
    plat_mutex_lock(&g_lock);
    g_st.sinks = (uint64_t)g_nsinks, g_st.sink_bytes = bytes, g_st.sinks_dead = (uint64_t)g_ndead;
    plat_mutex_unlock(&g_lock);
}

// Live sink i is out of use: it is freed once every submission handed out so far is done, since a table an earlier one
// holds may still name it, and the pages it imports stay valid until then.
static void sink_drop(int i) {
    g_dead = sinks_room(g_dead, &g_dead_cap, g_ndead + 1, sizeof *g_dead);
    g_dead[g_ndead] = g_sinks[i];
    g_dead[g_ndead++].gone = gpu_sync_handed();
    memmove(g_sinks + i, g_sinks + i + 1, (size_t)(g_nsinks - i - 1) * sizeof *g_sinks);
    g_nsinks--;
    g_sink_gen++;
    plat_mutex_lock(&g_lock);
    g_st.sinks_dropped++, g_st.sinks_dead = (uint64_t)g_ndead;
    plat_mutex_unlock(&g_lock);
}

// Free the dead sinks whose submissions are done (all of them at close).
static void sinks_reap(int all) {
    uint64_t done = gpu_sync_done();
    int k = 0, freed = 0;
    for (int i = 0; i < g_ndead; i++) {
        if (!all && g_dead[i].gone > done) {
            g_dead[k++] = g_dead[i];
            continue;
        }
        g_ops->free(&g_dead[i].m);
        freed++;
    }
    g_ndead = k;
    plat_mutex_lock(&g_lock);
    g_st.sinks_freed += (uint64_t)freed, g_st.sinks_dead = (uint64_t)g_ndead;
    plat_mutex_unlock(&g_lock);
}

// Whether the guest has every page of [lo, hi) committed now.
static int committed_whole(uint64_t lo, uint64_t hi) {
    for (uint64_t a = lo; a < hi;) {
        int c;
        uint64_t n = rt_mem_commit_run(a, hi - a, &c);
        if (n == UINT64_MAX || !n || !c) return 0;
        a += n;
    }
    return 1;
}

static int sink_refused(uint64_t lo, uint64_t hi) {
    plat_mutex_lock(&g_lock);
    g_st.sinks_refused++;
    plat_mutex_unlock(&g_lock);
    rt_log("gfx: vulkan: raw path: the import of [%#llx, %#llx) as the sink of a mirror's write-back was refused, and the "
           "guest no longer has all of it committed (it decommitted pages while the sink was made): no sink there now",
           (ull)lo, (ull)hi);
    return 0;
}

// A sink over the committed pages [lo, hi): 1, or 0 when the device refused the import and the guest no longer has all of
// [lo, hi) committed (another thread decommitted pages between the caller's look at the page map and the import): those
// pages get no sink, the effect leaves their stores out (or counts them lost, when committed again by then), and the next
// begin looks again. A refused import of pages still committed is retried once after a 4 KiB import of the first page
// clears it (the write-watch study), and a second refusal of pages still committed is a named stop.
static int sink_make(uint64_t lo, uint64_t hi) {
    uint64_t t0 = plat_now_ns(), made = rt_mem_decommit_epoch();   // before the import: a decommit after it may take its pages
    BackingMem m;
    if (!g_ops->import(lo, hi - lo, &m)) {
        if (!committed_whole(lo, hi)) return sink_refused(lo, hi);
        BackingMem probe;
        int cleared = g_ops->import(lo, BACKING_IMPORT_PROBE, &probe);
        if (cleared) g_ops->free(&probe);
        made = rt_mem_decommit_epoch();
        if (!cleared || !g_ops->import(lo, hi - lo, &m)) {
            if (!committed_whole(lo, hi)) return sink_refused(lo, hi);
            rt_fatal("gfx: vulkan: raw path: the device refused to import committed host memory [%#llx, %#llx) (%llu KiB) "
                     "as the sink of a mirror's write-back, %s: the raw path cannot write the mirror's pages into guest "
                     "memory without it", (ull)lo, (ull)hi, (ull)((hi - lo) >> 10),
                     cleared ? "again after a 4 KiB import cleared the first refusal"
                             : "and a 4 KiB import of its first page was refused too");
        }
    }
    int at = sink_after(lo);
    g_sinks = sinks_room(g_sinks, &g_sinks_cap, g_nsinks + 1, sizeof *g_sinks);
    memmove(g_sinks + at + 1, g_sinks + at, (size_t)(g_nsinks - at) * sizeof *g_sinks);
    g_sinks[at] = (Sink){.m = m, .lo = lo, .hi = hi, .made = made};
    g_nsinks++;
    g_sink_gen++;
    plat_mutex_lock(&g_lock);
    g_st.sinks_made++, g_st.sink_ns += plat_now_ns() - t0;
    plat_mutex_unlock(&g_lock);
    return 1;
}

// Sinks over every committed page of [lo, hi): each gap between the live sinks gets one, cut at the import windows (an
// import walks no further than its window, backing.h). Pages that are not committed get none.
static void sinks_cover(uint64_t lo, uint64_t hi) {
    for (uint64_t a = lo; a < hi;) {
        int c;
        uint64_t n = rt_mem_commit_run(a, hi - a, &c);
        if (n == UINT64_MAX || !n) return;
        uint64_t end = a + n;
        while (c && a < end) {
            const Sink *s = sink_at(a);
            if (s) {
                a = s->hi < end ? s->hi : end;
                continue;
            }
            int i = sink_after(a);   // the first sink above a
            uint64_t e = end, window = (a / BACKING_IMPORT_WINDOW + 1) * BACKING_IMPORT_WINDOW;
            if (i < g_nsinks && g_sinks[i].lo < e) e = g_sinks[i].lo;
            if (window < e) e = window;
            sink_make(a, e);
            a = e;
        }
        a = end;
    }
}

// The live sinks a guest decommit took a page of since they were made go, whether or not the guest has the page committed
// again: an import holds the pages it was made over, and a page committed again is another one, which the device would not
// write (oot2/memory.h, the decommit record). Looked at when the decommit epoch moved since the last look, so a commit, or
// a decommit elsewhere, drops none. A sink that passes holds its pages as one made now would: its epoch moves up, so the
// record's last decommits always cover it. What lay under the mirrors over the pages that remain is made again by
// sinks_prepare.
static void sinks_scrub(void) {
    uint64_t e = rt_mem_decommit_epoch();
    g_scrubbed = e;
    if (e == g_sink_epoch) return;
    g_sink_epoch = e;
    for (int i = 0; i < g_nsinks;) {
        if (rt_mem_decommitted(g_sinks[i].made, g_sinks[i].lo, g_sinks[i].hi)) {
            sink_drop(i);
            g_sinks_version = 0;
        } else {
            g_sinks[i++].made = e;
        }
    }
    sinks_count();
}

// Sinks under the mirrors the map holds now, at the start of a raw-access submission and before its mirrors are brought
// current. Making an import over guest pages marks them all in the write watch (the pages are locked for writing), and a
// mirror brought current afterwards reads that mark as the CPU's write and copies the pages again, the one time. A device's
// writes through the import do not mark them (Measured: vk_raw_test.c, wb). So the imports come first, and the mirrors'
// own sync query takes the marks together with any write the CPU made, losing none. A mirror made after this (a
// promotion in this very begin, a bind later in the submission) gets its sink when its table is built, and its pages
// are copied once more by the next submission. While the table in use names mirrored pages with no sink (the guest had
// not committed them when it was built, or their import was refused), every begin looks again, so a page committed since
// gets its sink, and the table is built again over it, at the next begin rather than at the map's next change.
static void sinks_prepare(void) {
    uint64_t version = backing_version();
    if (version == g_sinks_version && !(g_current && g_current->unsunk)) return;
    BackingRun *runs;
    uint64_t seen;
    int n = backing_raw_runs(&runs, &seen);
    for (int i = 0; i < n; i++)
        if (runs[i].kind == BACKING_MIRROR) sinks_cover(runs[i].guest, runs[i].guest + (runs[i].hi - runs[i].lo));
    free(runs);
    g_sinks_version = seen;
    sinks_count();
}

// Name the sink of each of the n mirrored pages (guest[i] their guest addresses), and the epoch it was made at into made[i]
// (NO_SINK for none): sinks are made over every committed run under them, and those no page maps to go. A page the guest
// has not committed now gets no sink (made over committed pages only): the effect leaves its store out, or counts it lost.
static void sinks_assign(RawMirrorPage *pages, const uint64_t *guest, uint64_t *made, uint64_t n) {
    g_build_id++;
    for (uint64_t i = 0; i < n;) {
        uint64_t j = i;
        while (j + 1 < n && guest[j + 1] == guest[j] + PAGE) j++;
        sinks_cover(guest[i], guest[j] + PAGE);
        i = j + 1;
    }
    for (uint64_t i = 0; i < n; i++) {
        Sink *s = sink_at(guest[i]);
        pages[i].sink = s ? s->m.address + (guest[i] - s->lo) : 0;
        made[i] = s ? s->made : NO_SINK;
        if (s) s->seen = g_build_id;
    }
    for (int i = 0; i < g_nsinks;) {
        if (g_sinks[i].seen != g_build_id) sink_drop(i);
        else i++;
    }
    sinks_count();
}

// Runs reduced to the chunks span_base to span_base + span_count - 1: a run is cut at the span's ends, and one wholly outside
// it goes (a pool made after the table was first built, which no pointer the submission recorded can reach). The count.
static int clip_runs(BackingRun *runs, int n, uint32_t span_base, uint32_t span_count) {
    uint64_t lo = (uint64_t)span_base << RAW_CHUNK_LOG2, hi = (uint64_t)(span_base + span_count) << RAW_CHUNK_LOG2;
    int k = 0;
    for (int i = 0; i < n; i++) {
        BackingRun r = runs[i];
        if (r.hi <= lo || r.lo >= hi) continue;
        if (r.lo < lo) r.address += lo - r.lo, r.guest += lo - r.lo, r.lo = lo;
        if (r.hi > hi) r.hi = hi;
        runs[k++] = r;
    }
    return k;
}

// The map as it is now, laid out over the span of every pool in it, or over the given one (span_count not 0), and written
// into the host image, the mirrored pages and their guest addresses: the layout, and the version the map was read at.
static void make_image(uint32_t span_base, uint32_t span_count, RawLayout *l, uint64_t *version) {
    BackingRun *runs;
    int n = backing_raw_runs(&runs, version);
    uint64_t plo, phi;
    if (!span_count && backing_pool_span(&plo, &phi) && phi > plo) {
        span_base = (uint32_t)(plo >> RAW_CHUNK_LOG2);
        span_count = (uint32_t)(((phi - 1) >> RAW_CHUNK_LOG2) - span_base + 1);
    }
    if (span_count) n = clip_runs(runs, n, span_base, span_count);
    char why[256];
    if (!raw_layout_span(runs, n, span_base, span_count, l, why, sizeof why))
        rt_fatal("gfx: vulkan: raw path: the page table: %s", why);
    g_image = grow_to(g_image, &g_image_bytes, (l->words * 4 + 15) & ~15ull, "the page table's image");
    g_pages = grow_to(g_pages, &g_pages_cap, (l->mirror_pages + 1) * sizeof *g_pages, "the mirrored pages");
    g_guest_scratch = grow_to(g_guest_scratch, &g_guest_cap, (l->mirror_pages + 1) * sizeof *g_guest_scratch,
                              "the mirrored pages");
    g_made_scratch = grow_to(g_made_scratch, &g_made_cap, (l->mirror_pages + 1) * sizeof *g_made_scratch,
                             "the mirrored pages");
    raw_write(runs, n, l, g_image, g_pages, g_guest_scratch);
    free(runs);
    sinks_assign(g_pages, g_guest_scratch, g_made_scratch, l->mirror_pages);
}

static uint64_t table_bytes_of(const RawLayout *l) { return (l->words * 4 + 15) & ~15ull; }
static uint64_t copy_bytes_of(const RawLayout *l) { return table_bytes_of(l) + l->mirror_pages * sizeof(RawMirrorPage); }

// The host image into copy c, which no submission in flight reads, and what the copy says about it.
static void store(Copy *c, const RawLayout *l, uint64_t version) {
    uint64_t table_bytes = table_bytes_of(l);
    memcpy(c->m.map, g_image, (size_t)l->words * 4);
    memcpy(c->m.map + table_bytes, g_pages, (size_t)(l->mirror_pages * sizeof *g_pages));
    free(c->guest);
    free(c->made);
    c->guest = malloc((size_t)(l->mirror_pages + 1) * sizeof *c->guest);
    c->made = malloc((size_t)(l->mirror_pages + 1) * sizeof *c->made);
    if (!c->guest || !c->made)
        rt_fatal("gfx: vulkan: raw path: out of memory for %llu mirrored pages", (ull)l->mirror_pages);
    memcpy(c->guest, g_guest_scratch, (size_t)l->mirror_pages * sizeof *c->guest);
    memcpy(c->made, g_made_scratch, (size_t)l->mirror_pages * sizeof *c->made);
    c->unsunk = 0;
    for (uint64_t i = 0; i < l->mirror_pages; i++) c->unsunk += c->made[i] == NO_SINK;
    c->l = *l, c->version = version, c->pages_at = table_bytes, c->sink_gen = g_sink_gen;
}

static void count_build(const RawLayout *l, uint64_t ns, int refresh) {
    plat_mutex_lock(&g_lock);
    if (refresh) g_st.refreshes++, g_st.refresh_ns += ns;
    else g_st.builds++, g_st.build_ns += ns;
    if (ns > g_st.build_ns_max) g_st.build_ns_max = ns;
    g_st.table_bytes = l->words * 4, g_st.pages = l->pages, g_st.mirror_pages = l->mirror_pages;
    g_st.blocks = l->blocks, g_st.span_count = l->span_count;
    plat_mutex_unlock(&g_lock);
}

// Build the table of the map as it is now into a free copy, which becomes the current one. The copy has room beyond the
// table for the map to grow while the submission that takes it is recorded (refresh).
static void build(void) {
    uint64_t t0 = plat_now_ns(), version;
    RawLayout l;
    make_image(0, 0, &l, &version);
    uint64_t bytes = copy_bytes_of(&l), room = bytes / 4 > RAW_COPY_ROOM ? bytes / 4 : RAW_COPY_ROOM;
    Copy *c = copy_free();
    if (c->m.size < bytes + room) {
        mem_free(&c->m);
        c->m = mem_new(bytes + room, k_table, 2, "a copy of the page table");
        plat_mutex_lock(&g_lock);
        g_st.copies++;
        plat_mutex_unlock(&g_lock);
    }
    store(c, &l, version);
    c->prev_use = 0;
    g_current = c;
    count_build(&l, plat_now_ns() - t0, 0);
    if (gpu_vk_raw_diag()) diag_table(&l, version, g_st.builds);
}

// The map moved while the submission was recorded (a bind made an import, a pool came or went), and the stages it
// recorded took the copy's address and span as they were: write the map as it is now into the same copy, over the same
// span, before the submission is submitted. The copy may be read by an earlier submission still in flight, which is
// waited for first (the copies are shared while the map holds still, so this is the rare case, not the frame's cost).
static void refresh(Copy *c) {
    uint64_t t0 = plat_now_ns(), version;
    RawLayout l;
    make_image(c->l.span_base, c->l.span_count, &l, &version);
    uint64_t bytes = copy_bytes_of(&l);
    if (bytes > c->m.size)
        rt_fatal("gfx: vulkan: raw path: the page table grew to %llu KiB while submission %llu was recorded, past the "
                 "%llu KiB of its copy: the stages it recorded already hold the copy's address", (ull)(bytes >> 10),
                 (ull)c->last_use, (ull)(c->m.size >> 10));
    if (c->prev_use > gpu_sync_done()) {
        plat_mutex_lock(&g_lock);
        g_st.refresh_waits++;
        plat_mutex_unlock(&g_lock);
        wait_for(c->prev_use);
    }
    store(c, &l, version);
    count_build(&l, plat_now_ns() - t0, 1);
}

void gpu_vk_raw_begin(uint64_t value, uint32_t push[4]) {
    if (g_value != value) {
        if (g_value) rt_fatal("gfx: vulkan: raw path: submission %llu began while %llu had not ended", (ull)value,
                              (ull)g_value);
        sinks_reap(0);
        sinks_scrub();
        sinks_prepare();   // before the mirrors are brought current: they take the marks the new imports leave
        backing_raw_begin(value);
        // The misses of earlier submissions are classified here, in the replay thread (the facts need locks no effect may
        // take) and after the invariant pass: a committed page of a live pool that the map still lacks now is one the pass
        // does not map, not a page committed since the last pass
        diag_drain();
        // What the mirrors hold: a page the CPU stamped after this may not be in its mirror (pagegen_mark_gpu_direct)
        g_since = backing_synced_min();
        if (g_since == UINT64_MAX) g_since = pagegen_now();
        if (!g_current || g_current->version != backing_version() || g_current->sink_gen != g_sink_gen) build();
        else g_current->prev_use = g_current->last_use;
        g_current->last_use = value;
        g_value = value;
        g_push[RAW_PUSH_TABLE_LOW] = (uint32_t)g_current->m.address;
        g_push[RAW_PUSH_TABLE_HIGH] = (uint32_t)(g_current->m.address >> 32);
        g_push[RAW_PUSH_SPAN_BASE] = g_current->l.span_base;
        g_push[RAW_PUSH_SPAN_COUNT] = g_current->l.span_count;
    }
    memcpy(push, g_push, sizeof g_push);
}

// ---------- the write-back
static void pipeline(void) {
    if (g_pipe) return;
    const GpuShader *s = gpu_shader("raw_writeback");
    if (!s) rt_fatal("gfx: vulkan: raw path: the runtime shader raw_writeback is not embedded");
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->size,
                                   .pCode = s->code};
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &g_module), "vkCreateShaderModule (raw write-back)");
    VkPushConstantRange pr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 40};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &li, NULL, &g_layout), "vkCreatePipelineLayout (raw write-back)");
    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = g_module, .pName = "main"},
        .layout = g_layout};
    GPU_VK_OK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &g_pipe),
              "vkCreateComputePipelines (raw write-back)");
}

// The out buffer's words: its head, the table's header and miss log, then the bitmap of the pages written, a bit for each of
// the mirrored pages.
static uint64_t out_words(uint32_t pages) { return OUT_HEAD + HEAD_WORDS + ((uint64_t)pages + 31) / 32; }

static Out *out_take(uint32_t pages) {
    plat_mutex_lock(&g_lock);
    Out **at = &g_free;
    while (*at && (*at)->pages < pages) at = &(*at)->next;
    Out *o = *at;
    if (o) *at = o->next;
    plat_mutex_unlock(&g_lock);
    if (!o) {
        o = calloc(1, sizeof *o);
        if (!o) rt_fatal("gfx: vulkan: raw path: out of memory for an out buffer");
        o->pages = pages;
        o->m = mem_new(out_words(pages) * 4, k_out, 2, "the raw write-back");
    }
    memset(o->m.map, 0, (size_t)(out_words(pages) * 4));
    return o;
}

// A submission's misses in the statistics: total counted by the helper over every stage of the submission, and the entries
// of head (a header and its log, or NULL for none) as the newest ones. The total also goes to the capture's raw-access
// counter, which tools/check_capture.py judges a run by: a run with a raw-access miss is not a proof run.
static void miss_count(uint64_t total, uint64_t value, const uint32_t *head) {
    uint32_t kept = head ? head[RAW_HEADER_MISS_COUNT] : 0;
    if (kept > RAW_MISS_ENTRIES) kept = RAW_MISS_ENTRIES;
    plat_mutex_lock(&g_lock);
    g_st.misses += total, g_st.miss_submissions++, g_st.last_miss_value = value, g_st.last_misses = kept;
    for (uint32_t i = 0; i < kept; i++) {
        const uint32_t *e = head + RAW_MISS_LOG_WORD + i * RAW_MISS_ENTRY_WORDS;
        g_st.last_miss[i] = (RawMiss){(uint64_t)e[RAW_MISS_ADDRESS_HIGH] << 32 | e[RAW_MISS_ADDRESS_LOW],
                                      e[RAW_MISS_KIND], e[RAW_MISS_REASON]};
    }
    plat_mutex_unlock(&g_lock);
    capture_count(CAPTURE_RAW_MISSES, total);
}

static void miss_log(const uint32_t *head, uint64_t value) {
    uint32_t count = head[RAW_HEADER_MISS_COUNT], kept = count < RAW_MISS_ENTRIES ? count : RAW_MISS_ENTRIES;
    static const char *const kinds[] = {"?", "load", "store", "atomic"};
    static const char *const reasons[] = {"?", "outside the span", "no chunk", "no page"};
    miss_count(count, value, head);
    rt_log("gfx: vulkan: raw path: submission %llu missed the page table %u times (a miss reads 0 and drops its write)",
           (ull)value, count);
    for (uint32_t i = 0; i < kept; i++) {
        const uint32_t *e = head + RAW_MISS_LOG_WORD + i * RAW_MISS_ENTRY_WORDS;
        uint32_t k = e[RAW_MISS_KIND], r = e[RAW_MISS_REASON];
        rt_log("gfx: vulkan: raw path:   %s at %#llx: %s", kinds[k < 4 ? k : 0],
               (ull)((uint64_t)e[RAW_MISS_ADDRESS_HIGH] << 32 | e[RAW_MISS_ADDRESS_LOW]), reasons[r < 4 ? r : 0]);
    }
}

// ---------- the miss diagnostics (oot2/vk_raw.h, setting raw_diag; the classes are oot2/raw_diag.h's)
#define SNAP_STAGES_MAX 4096u          // stages of a submission that get a snapshot
#define SNAP_BYTES (HEAD_WORDS * 4u)   // the table's header and miss log, copied after each stage
#define SNAP_KEYS 3                    // stage keys named for the draws of a pass
#define SNAP_RANGES 16                 // bound ranges kept for a stage
#define PENDING_MAX 1024u              // stages' misses waiting for the replay thread; past it they are dropped and counted
#define SCAN_RANGE_MAX (2ull << 20)    // bytes of each bound range searched for a missed pointer
#define SCAN_NEAR (1ull << 20)         // how far below a missed address a word still passes for the base of its array
#define SCAN_PIECE (64ull << 10)       // bytes read between two looks at the commit state: a decommit under a read faults
#define SCAN_PER_CLASS 2               // the missed addresses of a new stage and class searched for
#define SCAN_BUDGET (4ull << 30)       // bytes of bound buffers searched in a run (a second or two); the replay thread reads them
#define HORIZON_FREE 0u                // QueryMemory's states (runtime/core/memory.c)
#define HORIZON_NORMAL 5u

typedef struct SnapStage {
    char kind;   // 'c' a dispatch, 'd' the draws of a pass
    uint8_t nkeys, nranges;
    uint32_t draws;
    char key[SNAP_KEYS][RAWDIAG_KEY];
    RawRange range[SNAP_RANGES];
} SnapStage;

typedef struct Snaps {   // the snapshots of one submission
    Mem m;               // SNAP_BYTES for each stage, written by the device
    SnapStage *st;
    uint32_t n, cap, lost;   // stages with a snapshot, room for stages, stages past SNAP_STAGES_MAX
    struct Snaps *next;
} Snaps;

// One stage's misses, from the completion thread to the replay thread.
typedef struct Sample {
    struct Sample *next;
    uint64_t value;
    char kind;   // 'c', 'd', or 'e' for the misses after the last stage's snapshot
    uint8_t nkeys, nranges;
    uint32_t draws, count, kept;
    char key[SNAP_KEYS][RAWDIAG_KEY];
    RawRange range[SNAP_RANGES];
    RawLayout table;
    uint64_t table_version;
    RawMiss e[RAW_MISS_ENTRIES];
} Sample;

static int g_diag_state;                      // 0 unread, 1 off, 2 on
static Snaps *g_snaps;                        // the open submission's (the replay thread)
static Snaps *g_snaps_free;                   // g_lock
static PlatMutex g_dlock = PLAT_MUTEX_INIT;   // the misses waiting, the counts and the index of pools
static Sample *g_pend, *g_pend_last;
static uint32_t g_npend;
static uint64_t g_pend_dropped, g_snaps_lost, g_classified, g_scan_bytes, g_snap_total;
static RawDiagTally *g_tally;

static void census_rows(NvnStatRow emit, void *ctx);

int gpu_vk_raw_diag(void) {
    int s = __atomic_load_n(&g_diag_state, __ATOMIC_ACQUIRE), zero = 0;
    if (!s) {
        s = rt_setting(RT_SET_raw_diag) ? 2 : 1;
        if (__atomic_compare_exchange_n(&g_diag_state, &zero, s, 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE) && s == 2) {
            nvn_census_add_source(census_rows);
            rt_log("gfx: vulkan: raw path: raw_diag is on: each raw-access dispatch and each pass with raw-access draws is "
                   "followed by a copy and a clear of the page table's miss log (GPU time), every pool the game makes is "
                   "indexed, and each stage and class of misses is logged once and counted in census.tsv; this run is not a "
                   "timed run");
        }
        s = __atomic_load_n(&g_diag_state, __ATOMIC_ACQUIRE);
    }
    return s == 2;
}

static int pool_cls(const pool_t *p) {
    uint32_t f = p->flags;
    return (f & NVN_MEMORY_POOL_FLAGS_VIRTUAL)      ? RAWDIAG_POOL_VIRTUAL
           : (f & NVN_MEMORY_POOL_FLAGS_PHYSICAL)   ? RAWDIAG_POOL_PHYSICAL
           : (f & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ? RAWDIAG_POOL_GPU_ONLY
           : (f & NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS) ? RAWDIAG_POOL_CPU_ONLY
                                                       : RAWDIAG_POOL_CPU;
}

void gpu_vk_raw_pool_made(const pool_t *p) {
    if (!gpu_vk_raw_diag()) return;
    uint32_t present = (uint32_t)nvn_present_count();   // before the lock: it takes the pacing lock
    uint64_t value = gpu_sync_handed(), lo = (uint64_t)(uintptr_t)p->mem;
    plat_mutex_lock(&g_dlock);
    rawdiag_pool_made(lo, lo + p->size, p->flags, pool_cls(p), present, value);
    plat_mutex_unlock(&g_dlock);
}

void gpu_vk_raw_pool_gone(const pool_t *p) {
    if (!gpu_vk_raw_diag()) return;
    uint32_t present = (uint32_t)nvn_present_count();
    uint64_t value = gpu_sync_handed(), lo = (uint64_t)(uintptr_t)p->mem;
    plat_mutex_lock(&g_dlock);
    rawdiag_pool_gone(lo, lo + p->size, present, value);
    plat_mutex_unlock(&g_dlock);
}

// ---- the snapshots (the replay thread, between gpu_vk_raw_begin and gpu_vk_raw_end)
static Snaps *snaps_open(void) {
    plat_mutex_lock(&g_lock);
    Snaps *s = g_snaps_free;
    if (s) g_snaps_free = s->next;
    plat_mutex_unlock(&g_lock);
    if (!s) {
        s = calloc(1, sizeof *s);
        if (!s) rt_fatal("gfx: vulkan: raw path: out of memory for the stage snapshots");
        s->m = mem_new((uint64_t)SNAP_STAGES_MAX * SNAP_BYTES, k_out, 2, "the raw path's stage snapshots");
    }
    s->n = s->lost = 0, s->next = NULL;
    return s;
}

static void snaps_give(Snaps *s) {
    plat_mutex_lock(&g_dlock);
    g_snaps_lost += s->lost;
    plat_mutex_unlock(&g_dlock);
    plat_mutex_lock(&g_lock);
    s->next = g_snaps_free, g_snaps_free = s;
    plat_mutex_unlock(&g_lock);
}

int gpu_vk_raw_snap(uint64_t value, int kind, const char *const *keys, int nkeys, uint32_t draws, const RawRange *ranges,
                    int nranges, RawSnap *out) {
    if (!gpu_vk_raw_diag() || !g_value || g_value != value || !g_current) return 0;
    Snaps *s = g_snaps ? g_snaps : (g_snaps = snaps_open());
    if (s->n >= SNAP_STAGES_MAX) {
        s->lost++;
        return 0;
    }
    if (s->n == s->cap) {
        uint32_t cap = s->cap ? s->cap * 2 : 64;
        if (cap > SNAP_STAGES_MAX) cap = SNAP_STAGES_MAX;
        SnapStage *st = realloc(s->st, (size_t)cap * sizeof *st);
        if (!st) rt_fatal("gfx: vulkan: raw path: out of memory for %u stage snapshots", cap);
        s->st = st, s->cap = cap;
    }
    SnapStage *st = &s->st[s->n];
    memset(st, 0, sizeof *st);
    st->kind = (char)kind, st->draws = draws;
    st->nkeys = (uint8_t)(nkeys > SNAP_KEYS ? SNAP_KEYS : nkeys < 0 ? 0 : nkeys);
    for (int i = 0; i < st->nkeys; i++) snprintf(st->key[i], sizeof st->key[i], "%s", keys[i]);
    st->nranges = (uint8_t)(nranges > SNAP_RANGES ? SNAP_RANGES : nranges < 0 ? 0 : nranges);
    if (st->nranges) memcpy(st->range, ranges, st->nranges * sizeof *ranges);
    *out = (RawSnap){.src = (uint64_t)(uintptr_t)g_current->m.buf, .dst = (uint64_t)(uintptr_t)s->m.buf,
                     .dst_offset = (uint64_t)s->n * SNAP_BYTES, .bytes = SNAP_BYTES,
                     .count_offset = (uint64_t)RAW_HEADER_MISS_COUNT * 4u};
    s->n++;
    plat_mutex_lock(&g_dlock);
    g_snap_total++;
    plat_mutex_unlock(&g_dlock);
    return 1;
}

// ---- the completion thread: each stage's misses go to the replay thread, which has the locks the facts need
static void sample_push(Sample *s) {
    if (!s) return;   // out of memory: the diagnostics lose it, nothing else
    plat_mutex_lock(&g_dlock);
    if (g_npend >= PENDING_MAX) {
        g_pend_dropped++;
        plat_mutex_unlock(&g_dlock);
        free(s);
        return;
    }
    if (g_pend_last) g_pend_last->next = s;
    else g_pend = s;
    g_pend_last = s, g_npend++;
    plat_mutex_unlock(&g_dlock);
}

static Sample *sample_new(const Done *d, uint64_t value, const uint32_t *head, const SnapStage *st) {
    Sample *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->value = value;
    s->count = head[RAW_HEADER_MISS_COUNT];
    s->kept = s->count < RAW_MISS_ENTRIES ? s->count : RAW_MISS_ENTRIES;
    s->table = d->layout, s->table_version = d->table_version;
    if (st) {
        s->kind = st->kind, s->nkeys = st->nkeys, s->nranges = st->nranges, s->draws = st->draws;
        memcpy(s->key, st->key, sizeof s->key);
        memcpy(s->range, st->range, sizeof s->range);
    } else {
        s->kind = 'e';
    }
    for (uint32_t i = 0; i < s->kept; i++) {
        const uint32_t *e = head + RAW_MISS_LOG_WORD + i * RAW_MISS_ENTRY_WORDS;
        s->e[i] = (RawMiss){(uint64_t)e[RAW_MISS_ADDRESS_HIGH] << 32 | e[RAW_MISS_ADDRESS_LOW], e[RAW_MISS_KIND],
                            e[RAW_MISS_REASON]};
    }
    return s;
}

// The misses of every stage of a finished submission and of what followed its last stage (tail, the header the epilogue
// copied out): queued for the replay thread; the total counted by the helper, the stages that missed, and the first header
// with entries (for the statistics' newest misses).
static uint64_t harvest(const Done *d, uint64_t value, const uint32_t *tail, uint32_t *stages, const uint32_t **first) {
    uint64_t total = 0;
    *stages = 0, *first = NULL;
    const Snaps *sn = d->snaps;
    for (uint32_t i = 0; sn && i < sn->n; i++) {
        const uint32_t *head = (const uint32_t *)(const void *)(sn->m.map + (uint64_t)i * SNAP_BYTES);
        if (!head[RAW_HEADER_MISS_COUNT]) continue;
        total += head[RAW_HEADER_MISS_COUNT], ++*stages;
        if (!*first) *first = head;
        sample_push(sample_new(d, value, head, &sn->st[i]));
    }
    if (tail[RAW_HEADER_MISS_COUNT]) {
        total += tail[RAW_HEADER_MISS_COUNT], ++*stages;
        if (!*first) *first = tail;
        sample_push(sample_new(d, value, tail, NULL));
    }
    return total;
}

// ---- the replay thread: the facts about each missed address, its class, and the first sight of each stage and class
typedef struct Facts {
    RawDiagFacts f;
    RawDiagHit hit;
} Facts;

// table_version: the map version the table of the submission was built at.
static void gather(const RawMiss *m, uint64_t value, uint64_t table_version, Facts *out) {
    memset(out, 0, sizeof *out);
    RawDiagFacts *f = &out->f;
    plat_mutex_lock(&g_dlock);
    f->pool = rawdiag_pool_find(m->address, value, &out->hit);
    plat_mutex_unlock(&g_dlock);
    // A pool that was live for the submission and has been finalized since: the host and the map are read in a later state
    f->ended = out->hit.kind >= RAWDIAG_POOL_CPU && !out->hit.pool.live;
    f->kind = m->kind < 4 ? (int)m->kind : 0, f->reason = m->reason < 4 ? (int)m->reason : 0;
    f->commit = RAWDIAG_COMMIT_UNKNOWN, f->decl = RAWDIAG_DECL_OTHER;
    if (rt_mem_ready()) {
        RtMemInfo q = rt_mem_query(m->address);
        f->decl = q.state == HORIZON_NORMAL ? RAWDIAG_DECL_MAPPED : q.state == HORIZON_FREE ? RAWDIAG_DECL_FREE
                                                                                           : RAWDIAG_DECL_OTHER;
        int committed = 0;
        uint64_t n = rt_mem_commit_run(m->address & ~(PAGE - 1), PAGE, &committed);
        if (n != UINT64_MAX && n) f->commit = committed ? RAWDIAG_COMMIT_YES : RAWDIAG_COMMIT_NO;
    }
    BackingPage bp;
    backing_page(m->address, &bp);
    f->now = bp.kind;
    // a page the map holds at the very version the table was built at
    f->same_map = bp.kind != BACKING_NONE && table_version == backing_version();
    if (f->pool == RAWDIAG_POOL_VIRTUAL) {
        // A page of a virtual pool is the page of a physical pool it is mapped to (bp.gpu, a guest address; the address itself
        // when no mapping request mapped it): the game declared it mapped by mapping it, and the host commits the physical page
        int mapped = bp.gpu != m->address;
        f->decl = mapped ? RAWDIAG_DECL_MAPPED : RAWDIAG_DECL_FREE;
        f->commit = RAWDIAG_COMMIT_UNKNOWN;
        int committed = 0;
        uint64_t n = mapped && rt_mem_ready() ? rt_mem_commit_run(bp.gpu & ~(PAGE - 1), PAGE, &committed) : UINT64_MAX;
        if (n != UINT64_MAX && n) f->commit = committed ? RAWDIAG_COMMIT_YES : RAWDIAG_COMMIT_NO;
    }
}

static size_t appendf(char *buf, size_t size, size_t at, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static size_t appendf(char *buf, size_t size, size_t at, const char *fmt, ...) {
    if (at >= size) return at;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + at, size - at, fmt, ap);
    va_end(ap);
    return n < 0 ? at : at + (size_t)n;
}

// The stage's name in the counts: its key, or the keys of a pass's raw-access stages joined by '+'; nothing for the misses no
// stage owns.
static void stage_name(const Sample *s, char *buf, size_t size) {
    size_t at = 0;
    buf[0] = 0;
    for (int i = 0; i < s->nkeys; i++) at = appendf(buf, size, at, "%s%s", i ? "+" : "", s->key[i]);
}

static void stage_text(const Sample *s, char *buf, size_t size) {
    char name[RAWDIAG_NAME];
    stage_name(s, name, sizeof name);
    if (s->kind == 'c') snprintf(buf, size, "dispatch %s", name);
    else if (s->kind == 'd') snprintf(buf, size, "the %u raw-access draw(s) of a pass, stage(s) %s", s->draws, name);
    else snprintf(buf, size, "no stage (what the last stage's snapshot left)");
}

// Where a bound range of the stage holds the missed address as a word, or a word just below it, appended to buf.
static size_t scan_pointer(const Sample *s, uint64_t target, uint64_t gap, char *buf, size_t size, size_t at) {
    RawDiagWord ex = {0}, bl = {0};
    int ex_range = -1, bl_range = -1;
    plat_mutex_lock(&g_dlock);
    uint64_t spent = g_scan_bytes;
    plat_mutex_unlock(&g_dlock);
    if (spent >= SCAN_BUDGET)
        return appendf(buf, size, at, "; not searched: the %llu MiB a run may search for pointers are used",
                       (ull)(SCAN_BUDGET >> 20));
    for (int r = 0; r < s->nranges && rt_mem_ready(); r++) {
        const RawRange *rg = &s->range[r];
        uint64_t n = rg->size < SCAN_RANGE_MAX ? rg->size : SCAN_RANGE_MAX;
        for (uint64_t a = rg->addr; a - rg->addr < n;) {
            int committed = 0;
            uint64_t run = rt_mem_commit_run(a, rg->addr + n - a, &committed);
            if (run == UINT64_MAX || !run) break;
            if (run > SCAN_PIECE) run = SCAN_PIECE;
            if (committed) {
                // The game may unmap what a stage bound a submission ago: the pages are looked at again just before each
                // piece is read, and a piece read while a decommit happened is not believed
                uint64_t dec = rt_mem_decommit_epoch();
                RawDiagWord e, b;
                int got = rawdiag_scan_word((const uint8_t *)(uintptr_t)a, run, target, SCAN_NEAR, &e, &b);
                if (rt_mem_decommit_epoch() != dec) got = 0;
                plat_mutex_lock(&g_dlock);
                g_scan_bytes += run;
                plat_mutex_unlock(&g_dlock);
                if ((got & 1) && ex_range < 0) ex = (RawDiagWord){e.offset + (a - rg->addr), e.word}, ex_range = r;
                if ((got & 2) && (bl_range < 0 || b.word > bl.word))
                    bl = (RawDiagWord){b.offset + (a - rg->addr), b.word}, bl_range = r;
            }
            a += run;
        }
    }
    if (ex_range >= 0)
        at = appendf(buf, size, at, "; the address is a word of bound %s buffer %d [%#llx, +%#llx) at +%#llx",
                     s->range[ex_range].kind == 'u' ? "uniform" : "storage", ex_range, (ull)s->range[ex_range].addr,
                     (ull)s->range[ex_range].size, (ull)ex.offset);
    if (bl_range >= 0)
        at = appendf(buf, size, at,
                     "; the nearest word below it is %#llx (%llu bytes under) in bound %s buffer %d at +%#llx",
                     (ull)bl.word, (ull)(target - bl.word), s->range[bl_range].kind == 'u' ? "uniform" : "storage",
                     bl_range, (ull)bl.offset);
    if (ex_range < 0 && bl_range < 0)
        at = appendf(buf, size, at, "; no word of the stage's %d bound buffer(s) (the first %llu KiB of each) holds it or a "
                     "base within %llu KiB below it (as the buffers read now hold it)", s->nranges,
                     (ull)(SCAN_RANGE_MAX >> 10), (ull)(SCAN_NEAR >> 10));
    // The buffers are read when the miss is classified, not at the submission: a buffer the game rewrites (a ring) may no
    // longer hold the pointer that was read, or may hold an equal word that was not
    at = appendf(buf, size, at, " [buffers read %llu timeline value(s) after the miss's submission]", (ull)gap);
    return at;
}

static void log_class(const Sample *s, uint32_t i, const Facts *x) {
    char stage[160], what[512], ptr[640], name[128];
    const RawMiss *m = &s->e[i];
    stage_text(s, stage, sizeof stage);
    rawdiag_describe(&x->f, &x->hit, what, sizeof what);
    rawdiag_class_name(rawdiag_class(&x->f), name, sizeof name);
    char v = rawdiag_verdict(&x->f);
    uint64_t handed = gpu_sync_handed(), gap = handed > s->value ? handed - s->value : 0;   // submissions since the miss
    rt_log("gfx: vulkan: raw path: miss class %c (%s), first seen in %s, submission %llu, which the stage missed %u times: "
           "%#llx", v, name, stage, (ull)s->value, s->count, (ull)m->address);
    rt_log("gfx: vulkan: raw path:   %s", what);
    uint64_t span_lo = (uint64_t)s->table.span_base << RAW_CHUNK_LOG2;
    size_t at = 0;
    ptr[0] = 0;
    at = appendf(ptr, sizeof ptr, at, "verdict %c, %s; the table of the submission: span of %u chunks of 2 MiB from %#llx, "
                 "%u leaf blocks, %llu pages (%llu mirrored), built at map version %llu, the map is at %llu", v,
                 rawdiag_verdict_text(v), s->table.span_count, (ull)span_lo, s->table.blocks, (ull)s->table.pages,
                 (ull)s->table.mirror_pages, (ull)s->table_version, (ull)backing_version());
    rt_log("gfx: vulkan: raw path:   %s", ptr);
    // The word that held the pointer: the first two distinct missed addresses of the stage from this one
    uint64_t done[SCAN_PER_CLASS];
    int nd = 0;
    for (uint32_t j = i; j < s->kept && nd < SCAN_PER_CLASS; j++) {
        int seen = 0;
        for (int k = 0; k < nd; k++) seen |= done[k] == s->e[j].address;
        if (seen) continue;
        done[nd++] = s->e[j].address;
        size_t p = appendf(ptr, sizeof ptr, 0, "address %#llx", (ull)s->e[j].address);
        scan_pointer(s, s->e[j].address, gap, ptr, sizeof ptr, p);
        rt_log("gfx: vulkan: raw path:   %s", ptr);
    }
}

// How the stage's logged addresses lie (oot2/raw_diag.h, rawdiag_shape): whether the step from one to the next repeats says
// whether they were computed from the lane's index or read from a buffer.
static void log_shape(const Sample *s) {
    uint64_t a[RAW_MISS_ENTRIES];
    for (uint32_t i = 0; i < s->kept; i++) a[i] = s->e[i].address;
    char buf[400];
    rawdiag_shape(a, s->kept, buf, sizeof buf);
    rt_log("gfx: vulkan: raw path:   the %u logged addresses of the stage, in the order logged: %s", s->kept, buf);
}

static void classify(const Sample *s) {
    char name[RAWDIAG_NAME];
    stage_name(s, name, sizeof name);
    int any_fresh = 0;
    for (uint32_t i = 0; i < s->kept; i++) {
        Facts x;
        gather(&s->e[i], s->value, s->table_version, &x);
        int fresh = 0;
        plat_mutex_lock(&g_dlock);
        if (!g_tally) g_tally = rawdiag_tally_new();
        if (g_tally)
            fresh = rawdiag_tally_add(g_tally, name, s->kind, rawdiag_class(&x.f), rawdiag_share(s->count, s->kept, i));
        g_classified++;
        plat_mutex_unlock(&g_dlock);
        if (fresh) log_class(s, i, &x), any_fresh = 1;
    }
    if (any_fresh) log_shape(s);   // once for a sample that brought a new class
}

static void diag_drain(void) {
    if (!gpu_vk_raw_diag()) return;
    plat_mutex_lock(&g_dlock);
    Sample *list = g_pend;
    g_pend = g_pend_last = NULL;
    g_npend = 0;
    plat_mutex_unlock(&g_dlock);
    while (list) {
        Sample *s = list;
        list = s->next;
        classify(s);
        free(s);
    }
}

void gpu_vk_raw_diag_drain(void) { diag_drain(); }

// A table build, for the log: the first, then each build that changed the span, and every hundredth.
static void diag_table(const RawLayout *l, uint64_t version, uint64_t build) {
    static uint32_t base, count;
    if (build > 1 && l->span_base == base && l->span_count == count && build % 100) return;
    base = l->span_base, count = l->span_count;
    rt_log("gfx: vulkan: raw path: table build %llu at map version %llu: span of %u chunks of 2 MiB from %#llx to %#llx, %u "
           "leaf blocks, %llu pages (%llu mirrored), %llu KiB", (ull)build, (ull)version, l->span_count,
           (ull)l->span_base << RAW_CHUNK_LOG2, ((ull)l->span_base + l->span_count) << RAW_CHUNK_LOG2, l->blocks,
           (ull)l->pages, (ull)l->mirror_pages, (ull)(l->words * 4 >> 10));
}

// census.tsv: the misses the helper counted, and with the classes, by class and by stage (oot2/gfx.h, nvn_census_add_source).
static void census_rows(NvnStatRow emit, void *ctx) {
    GpuVkRawStats st;
    gpu_vk_raw_stats(&st);
    RawDiagPoolCounts pc;
    char key[160], why[512];
    plat_mutex_lock(&g_dlock);
    rawdiag_pool_counts(&pc);
    uint64_t pending = g_npend, dropped = g_pend_dropped, classified = g_classified, scanned = g_scan_bytes,
             lost = g_snaps_lost;
    snprintf(why, sizeof why, "raw-access misses the page table's helper counted, loads, stores and atomics, over %llu "
             "submissions (0 is a count: the capture's raw column holds the same); %llu entries classified, %llu stages' "
             "misses still waiting for the replay thread, %llu dropped for room, %llu stages left without a snapshot for "
             "room", (ull)st.miss_submissions, (ull)classified, (ull)pending, (ull)dropped, (ull)lost);
    emit(ctx, "raw-misses", st.misses, why);
    snprintf(why, sizeof why, "memory pools the game made %llu, finalized %llu, live %llu (finalizations of a range no pool "
             "had: %llu, pools not indexed: %llu); %llu KiB of bound buffers searched for missed pointers",
             (ull)pc.made, (ull)pc.finalized, (ull)pc.live, (ull)pc.unknown_ends, (ull)pc.not_kept, (ull)(scanned >> 10));
    emit(ctx, "raw-miss-pools", pc.made, why);
    for (int i = 0; g_tally; i++) {
        RawDiagClassRow r;
        if (!rawdiag_tally_class(g_tally, i, &r)) break;
        RawDiagFacts f;
        rawdiag_class_facts(r.cls, &f);
        char name[128];
        rawdiag_class_name(r.cls, name, sizeof name);
        snprintf(key, sizeof key, "raw-miss %c %s", rawdiag_verdict(&f), name);
        snprintf(why, sizeof why, "%llu entries logged (the first 64 of each stage), standing for %llu misses (a stage that "
                 "missed more than it logged is counted by share); %s", (ull)r.entries, (ull)r.reads,
                 rawdiag_verdict_text(rawdiag_verdict(&f)));
        emit(ctx, key, r.reads, why);
    }
    for (int i = 0; g_tally; i++) {
        RawDiagStageRow r;
        if (!rawdiag_tally_stage(g_tally, i, &r)) break;
        snprintf(key, sizeof key, "raw-miss-stage %s %s", r.stage == 'c' ? "dispatch" : r.stage == 'd' ? "draws" : "none",
                 r.key[0] ? r.key : "-");
        snprintf(why, sizeof why, "%llu entries logged, standing for %llu misses", (ull)r.entries, (ull)r.reads);
        emit(ctx, key, r.reads, why);
    }
    plat_mutex_unlock(&g_dlock);
}

static void summary_row(void *ctx, const char *key, uint64_t count, const char *why) {
    (void)ctx;
    rt_log("gfx: vulkan: raw path: summary: %s: %llu (%s)", key, (ull)count, why);
}

static void diag_close(void) {
    if (!gpu_vk_raw_diag()) return;
    diag_drain();
    census_rows(summary_row, NULL);   // the run's numbers in the log too, in case census.tsv is not written
    plat_mutex_lock(&g_dlock);
    rawdiag_tally_free(g_tally);
    g_tally = NULL;
    rawdiag_pool_reset();
    g_pend_dropped = g_classified = g_scan_bytes = g_snaps_lost = g_snap_total = 0;
    plat_mutex_unlock(&g_dlock);
    plat_mutex_lock(&g_lock);
    while (g_snaps_free) {
        Snaps *s = g_snaps_free;
        g_snaps_free = s->next;
        mem_free(&s->m);
        free(s->st);
        free(s);
    }
    plat_mutex_unlock(&g_lock);
    if (g_snaps) mem_free(&g_snaps->m), free(g_snaps->st), free(g_snaps);
    g_snaps = NULL;
    __atomic_store_n(&g_diag_state, 0, __ATOMIC_RELEASE);
}

static int bit_set(const uint32_t *bits, uint64_t i) { return bits[i >> 5] >> (i & 31) & 1; }

// What the walk of one submission's bitmap found: pages stamped as the GPU's, left out, lost; of the first, those the
// CPU had written; the time in pagegen_mark_gpu_direct; the first page left out and the first lost, for the log.
typedef struct Walk { uint64_t marked, skipped, lost, cpu, mark_ns, first_skipped, first_lost; } Walk;

static void left_out(Walk *w, uint64_t lo, uint64_t pages) {
    w->skipped += pages, w->first_skipped = w->first_skipped ? w->first_skipped : lo;
}

// Pages [lo, hi) whose GPU store the guest's page does not hold: the plain stamp, so the mirror takes the guest's bytes at
// its next sync, never the GPU-current one.
static void lost(Walk *w, uint64_t lo, uint64_t hi) {
    pagegen_mark(lo, hi);
    w->lost += (hi - lo) / PAGE, w->first_lost = w->first_lost ? w->first_lost : lo;
}

// Pages [lo, hi) the write-back noted, all through one sink made at decommit epoch `made` (NO_SINK: none). A page the
// guest has not committed now is left out: the device wrote it into memory its sink keeps valid and the guest no longer
// sees. A committed page is stamped as the GPU's, unless the guest's page is not the one the device wrote: a decommit took
// it from under its sink since the sink was made (before the device wrote it, or while it did, or while this ran: checked
// after the stamp, which the plain one then replaces), or it had no sink (the guest had not committed it when the table
// was built, and has since). Those are lost.
static void stamp_run(Walk *w, uint64_t lo, uint64_t hi, uint64_t made, uint64_t since) {
    for (uint64_t a = lo; a < hi;) {
        int committed;
        uint64_t n = rt_mem_commit_run(a, hi - a, &committed);
        if (n == UINT64_MAX || !n) {
            left_out(w, a, (hi - a) / PAGE);
            return;
        }
        if (!committed) {
            left_out(w, a, n / PAGE);
        } else if (made == NO_SINK) {
            lost(w, a, a + n);
        } else {
            uint64_t t1 = plat_now_ns(), cpu = 0;
            pagegen_mark_gpu_direct(a, a + n, since, &cpu);   // the mirror holds these bytes: the next sync does not copy them back
            w->mark_ns += plat_now_ns() - t1, w->cpu += cpu, w->marked += n / PAGE;
            if (rt_mem_decommitted(made, a, a + n))
                for (uint64_t p = a; p < a + n; p += PAGE)
                    if (rt_mem_decommitted(made, p, p + PAGE)) w->marked--, lost(w, p, p + PAGE);
        }
        a += n;
    }
}

// The completion thread, once the device reached value and before it is done: the pages the GPU wrote into guest memory
// through their sinks are stamped. No byte of them is read or copied here. The pages are the set bits of the bitmap, in
// address order (the mirrored pages' order), so the runs of neighbouring pages are found without a sort.
static void ended(void *arg, uint64_t value) {
    uint64_t t0 = plat_now_ns();
    Done *d = arg;
    const uint32_t *words = (const uint32_t *)d->out->m.map;
    if (gpu_vk_raw_diag()) {
        uint32_t stages;
        const uint32_t *first;
        uint64_t total = harvest(d, value, words + OUT_HEAD, &stages, &first);
        if (total) {
            miss_count(total, value, first);
            rt_log("gfx: vulkan: raw path: submission %llu missed the page table %llu times in %u stage(s) (a miss reads 0 "
                   "and drops its write); the classes follow once for each stage", (ull)value, (ull)total, stages);
        }
        if (d->snaps) snaps_give(d->snaps);
    } else if (words[OUT_HEAD + RAW_HEADER_MISS_COUNT]) {
        miss_log(words + OUT_HEAD, value);
    }
    const uint32_t *bits = words + OUT_HEAD + HEAD_WORDS;
    Walk w = {0};
    for (uint64_t i = 0; i < d->pages;) {
        if (!bit_set(bits, i)) {
            i += !(i & 31) && !bits[i >> 5] ? 32 : 1;   // a word with no bit set goes at once
            continue;
        }
        // The run [i, j]: pages written, next to each other in guest memory, through sinks made at one epoch (or none).
        uint64_t j = i;
        while (j + 1 < d->pages && bit_set(bits, j + 1) && d->made[j + 1] == d->made[i] &&
               d->guest[j + 1] == d->guest[j] + PAGE)
            j++;
        // A sink the begin kept held its pages up to d->scrubbed: only a decommit after that can have taken one
        uint64_t made = d->made[i] == NO_SINK || d->made[i] > d->scrubbed ? d->made[i] : d->scrubbed;
        stamp_run(&w, d->guest[i], d->guest[j] + PAGE, made, d->since);
        i = j + 1;
    }
    if (w.first_skipped)
        rt_log("gfx: vulkan: raw path: a raw store of submission %llu into the mirror of %#llx lands in memory the guest "
               "no longer maps: left out (%llu pages)", (ull)value, (ull)w.first_skipped, (ull)w.skipped);
    if (w.first_lost)
        rt_log("gfx: vulkan: raw path: raw stores of submission %llu into the mirror of %#llx are lost (%llu pages): the "
               "guest decommitted the page under the store's sink and committed it again, or had it committed only after "
               "the table was built, so its page does not hold the device's bytes; stamped as the CPU's, the mirror takes "
               "the guest's", (ull)value, (ull)w.first_lost, (ull)w.lost);
    // The shader counts a page when it sets its bit: the two must agree, or a page was written back that this walk lost.
    if (w.marked + w.skipped + w.lost != words[0])
        rt_fatal("gfx: vulkan: raw path: the write-back of submission %llu took %u pages and the walk of its bitmap found "
                 "%llu", (ull)value, words[0], (ull)(w.marked + w.skipped + w.lost));
    uint64_t ns = plat_now_ns() - t0;
    plat_mutex_lock(&g_lock);
    g_st.writebacks += w.marked, g_st.writeback_bytes += w.marked * PAGE, g_st.writebacks_skipped += w.skipped;
    g_st.writebacks_lost += w.lost, g_st.writeback_cpu_pages += w.cpu;
    g_st.ended_ns += ns, g_st.mark_ns += w.mark_ns;
    if (ns > g_st.ended_ns_max) g_st.ended_ns_max = ns;
    d->out->next = g_free, g_free = d->out;
    plat_mutex_unlock(&g_lock);
    free(d);
}

void gpu_vk_raw_end(uint64_t value) {
    if (!g_value || g_value != value) return;
    g_value = 0;
    Copy *c = g_current;
    if (c->version != backing_version()) refresh(c);   // the stages took the copy's address, so it follows the map here
    uint32_t pages = (uint32_t)c->l.mirror_pages;
    Out *o = out_take(pages);
    Done *d = malloc(sizeof *d);
    if (!d) rt_fatal("gfx: vulkan: raw path: out of memory for a submission's end");
    *d = (Done){.out = o, .guest = c->guest, .made = c->made, .pages = pages, .since = g_since, .scrubbed = g_scrubbed,
                .layout = c->l, .table_version = c->version};
    if (g_snaps) {   // the miss diagnostics' stage snapshots (setting raw_diag): handed to the effect, or back when empty
        if (g_snaps->n) d->snaps = g_snaps;
        else snaps_give(g_snaps);
        g_snaps = NULL;
    }
    pipeline();
    VkCommandBuffer cb = gpu_vk_epilogue();   // after a full barrier over the submission's own work
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    uint64_t table_pages = c->m.address + c->pages_at;
    uint32_t pc[10] = {(uint32_t)c->m.address, (uint32_t)(c->m.address >> 32), (uint32_t)table_pages,
                       (uint32_t)(table_pages >> 32), (uint32_t)o->m.address, (uint32_t)(o->m.address >> 32), pages,
                       RAW_STORE_BITS_WORD, RAW_HEADER_MISS_COUNT, HEAD_WORDS};
    vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    uint64_t groups = pages ? pages : 1;
    vkCmdDispatch(cb, (uint32_t)(groups < 65535 ? groups : 65535), 1, 1);
    // The device's writes into the sinks (the guest's own pages) and the out buffer, visible to the host once the value is reached
    int diag = gpu_vk_raw_diag();   // the stage snapshots are transfer writes into host-visible memory
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                           (diag ? VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT : 0),
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                            (diag ? VK_ACCESS_2_TRANSFER_WRITE_BIT : 0),
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT |
                                            VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    gpu_sync_effect(ended, d);
    plat_mutex_lock(&g_lock);
    g_st.submissions++;
    plat_mutex_unlock(&g_lock);
}

void gpu_vk_raw_stats(GpuVkRawStats *out) {
    plat_mutex_lock(&g_lock);
    *out = g_st;
    plat_mutex_unlock(&g_lock);
    RawDiagPoolCounts pc;
    plat_mutex_lock(&g_dlock);
    rawdiag_pool_counts(&pc);
    out->diag_snapshots = g_snap_total, out->diag_classified = g_classified, out->diag_pools = pc.made;
    plat_mutex_unlock(&g_dlock);
}

void gpu_vk_raw_set_ops(const BackingOps *ops) { g_ops = ops ? ops : &gpu_vk_backing_ops; }

void gpu_vk_raw_close(void) {
    diag_close();   // the misses not classified yet, the counts and the index of pools, the snapshots' buffers
    for (int i = 0; i < g_ncopies; i++) {
        mem_free(&g_copy[i].m);
        free(g_copy[i].guest);
        free(g_copy[i].made);
    }
    memset(g_copy, 0, sizeof g_copy);
    g_ncopies = 0, g_current = NULL, g_value = 0;
    for (int i = g_nsinks - 1; i >= 0; i--) g_ops->free(&g_sinks[i].m);   // every submission is done
    sinks_reap(1);
    free(g_sinks), free(g_dead);
    g_sinks = g_dead = NULL, g_nsinks = g_sinks_cap = g_ndead = g_dead_cap = 0;
    g_sink_epoch = g_scrubbed = g_sinks_version = g_build_id = g_since = 0, g_sink_gen = 1;
    g_ops = &gpu_vk_backing_ops;
    while (g_free) {
        Out *o = g_free;
        g_free = o->next;
        mem_free(&o->m);
        free(o);
    }
    if (g_pipe) vkDestroyPipeline(gpu_vk.device, g_pipe, NULL);
    if (g_layout) vkDestroyPipelineLayout(gpu_vk.device, g_layout, NULL);
    if (g_module) vkDestroyShaderModule(gpu_vk.device, g_module, NULL);
    g_pipe = VK_NULL_HANDLE, g_layout = VK_NULL_HANDLE, g_module = VK_NULL_HANDLE;
    free(g_image), free(g_pages), free(g_guest_scratch), free(g_made_scratch);
    g_image = NULL, g_pages = NULL, g_guest_scratch = g_made_scratch = NULL;
    g_image_bytes = g_pages_cap = g_guest_cap = g_made_cap = 0;
    memset(&g_st, 0, sizeof g_st);
}
