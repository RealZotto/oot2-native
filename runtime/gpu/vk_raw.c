// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The raw path on the device (oot2/vk_raw.h, architecture D15): the copies of the page table, the sinks the write-back
// goes through into guest memory, the out buffers of the write-back, the runtime shader raw_writeback.comp, and the effect
// that logs misses and stamps the raw stores the GPU wrote into guest memory.
#include <stdlib.h>
#include <string.h>

#include "oot2/capture.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
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
    if (words[OUT_HEAD + RAW_HEADER_MISS_COUNT]) miss_log(words + OUT_HEAD, value);
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
    *d = (Done){o, c->guest, c->made, pages, g_since, g_scrubbed};
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
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
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
}

void gpu_vk_raw_set_ops(const BackingOps *ops) { g_ops = ops ? ops : &gpu_vk_backing_ops; }

void gpu_vk_raw_close(void) {
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
