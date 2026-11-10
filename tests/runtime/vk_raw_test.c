// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The raw path on a real device (oot2/vk_raw.h, architecture D15), driven by tests/test_runtime_vk_raw.py:
// `vk_raw_test <case> <raw.spv>`, one case per process. raw.spv is the translator's module of the test's raw program
// (the Python side says what it does): each of its 64 threads reads its 32-byte Entry of the pointer table at TABLE, loads
// a word at the entry's probe, stores (thread | 0xA0000000) at the probe plus 4, adds 1 at the entry's atomic address,
// and writes the loaded word and the atomic's old value back into its entry. Every access goes through the page table.
//
//   usable   whether a device here has every item the renderer requires (no device is opened)
//   kinds    16 threads into each backing kind: an import, a mirror, a VRAM extent and a virtual pool over a physical one
//            in the window; each reads its page's word, its store lands, the 16 atomics of each kind land on one counter;
//            the mirror's stores and counter reach guest memory once the value is done, and nothing beside them
//   keep     a mirror's raw stores written back stay in the mirror: the next submission copies nothing into it, reads the
//            stored words and counts on from the stored counter; a CPU write to a page of it is copied in, that page only
//   miss     8 threads at guest memory no pool covers: their loads read 0, their stores and atomics are dropped (guest
//            memory keeps its bytes), all 24 are in the miss log with the address, the kind and the reason; the next
//            submission's misses are its own
//   commit   a chunk of a pool committed after the pool was mapped: a miss before, its word read by the next submission
//   edges    the lowest and the highest mapped page, the lowest in the guest heap and the highest at the top of the window
//   reuse    a pool finalized and another made over the same range, CPU-visible and in the window: the new pool's words
//   grow     one submission, two raw-access stages and two binds: a mirror made by a small bind and mapped by the first
//            stage's table, then a bind that starts in the mirror's last pages and runs far past its end over pages the
//            guest never touched (it used to stop by name), then a second stage whose pointers lie in those pages, and a
//            bound copy of the same bytes out: both stages used one table (the same push words), rewritten at the end,
//            the second stage's loads, stores and atomics all landed, the bound copy and guest memory hold the same
//            bytes, and the mirror's kept pages were written back
//   grow_taken   a bound mirror grows without moving: bound and raw writes before and after the growth, including newly
//                committed pages outside the bind, reach the same bytes on the device and in guest memory
//   grow_guarded_head, grow_guarded_tail   a bound mirror grows beside a GPU-only extent used by the same submission;
//                the committed chunk beside the bind keeps an import over its other pages, which the raw stage reaches
//   heap MIB what the invariant costs over a guest heap of MIB MiB committed in 256 MiB pools, with a 256 MiB pool in the
//            window: resident bytes and VRAM before and after the first raw-access submission, the invariant's pass and
//            the table's build timed, then a second submission with the map unchanged (measured, not judged; the full
//            5.5 GiB heap takes more than 4 GB of RAM: a HEAVY slot)
//
// Every case closes the device and prints the validation counts; tests/test_runtime_vk_raw.py runs them under the
// synchronization layer and requires no error.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#include "oot2/vk_pipeline.h"
#include "oot2/vk_raw.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3
#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define WIN GFX_GPU_WINDOW_BASE
#define PG 0x1000ull
#define THREADS 64
#define TABLE (WIN + 0x40000000ull)   // test_runtime_vk_raw.py, TABLE
#define PROBE 0x40                    // the offset in a page the threads read
#define MARK 0xA0000000u              // the raw store's value is MARK | thread

typedef unsigned long long ull;

typedef struct Entry {
    uint64_t probe, atomic;
    uint32_t loaded, old, pad[2];
} Entry;
_Static_assert(sizeof(Entry) == 32, "the program's entry is 32 bytes");

// ---------- usable
static void case_usable(void) {
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        exit(EXIT_NO_DEVICE);
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) {
        printf("no Vulkan 1.3 instance\n");
        exit(EXIT_NO_DEVICE);
    }
    volkLoadInstanceOnly(inst);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL) == VK_SUCCESS);
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    if (n) CHECK(vkEnumeratePhysicalDevices(inst, &n, devs) >= 0);
    int usable = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ne = 0;
        CHECK(vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, NULL) == VK_SUCCESS);
        VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
        CHECK(ext && vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, ext) == VK_SUCCESS);
        Oot2VkFacts f;
        oot2_vk_query(vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceFeatures2, devs[i], ext, ne, &f);
        free(ext);
        unsigned char has[OOT2_VK_ITEM_COUNT];
        usable += oot2_vk_judge(&f, has) == OOT2_VK_ITEM_COUNT;
    }
    vkDestroyInstance(inst, NULL);
    if (!usable) {
        printf("no device here has every item the renderer requires\n");
        exit(EXIT_NO_DEVICE);
    }
    printf("usable %d\n", usable);
}

// ---------- guest memory and pools
static Cpu g_cpu;
static uint64_t g_heap;
static uint64_t g_mirror_lo, g_mirror_hi;   // the mirror hook's range: mirrors there, imports elsewhere

static void guest_heap(uint64_t size) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = size;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1];
}

static int mirror_hook(const BackingAsk *ask) { return ask->addr >= g_mirror_lo && ask->addr < g_mirror_hi; }

static pool_t make_pool(uint64_t mem, uint64_t size, uint32_t flags) {
    pool_t p = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = flags, .mem = (uint8_t *)(uintptr_t)mem, .size = size,
                .game_mem = (flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) ? NULL : (uint8_t *)(uintptr_t)mem};
    CHECK(gfx_vulkan_backend.pool_init(&p));
    return p;
}

static void map_run(pool_t *v, pool_t *phys, int64_t poff, uint64_t size) {
    nvn_mapping_t m = {.physical_pool = (const NVNmemoryPool *)(const void *)phys, .physical_offset = poff,
                       .virtual_offset = 0, .size = size};
    CHECK(gfx_vulkan_backend.pool_map_virtual(v, 1, &m));
}

static void fill_words(uint64_t a, uint64_t n, uint32_t word) {
    for (uint64_t i = 0; i < n / 4; i++) ((volatile uint32_t *)(uintptr_t)a)[i] = word;
}

static uint32_t word_at(uint64_t a) { return *(volatile const uint32_t *)(uintptr_t)a; }

// ---------- submissions
static VkBuffer g_buf;
static VkDeviceMemory g_mem;
static volatile uint32_t *g_words;   // the readback buffer, 64 KiB
static VkpLayouts g_lay;
static VkPipeline g_pipe;

static void make_buffer(void) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 64 * KiB,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    CHECK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &g_buf) == VK_SUCCESS);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, g_buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; i++)
        if ((mr.memoryTypeBits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) type = i;
    CHECK(type != UINT32_MAX);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                  .memoryTypeIndex = type};
    CHECK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &g_mem) == VK_SUCCESS);
    CHECK(vkBindBufferMemory(gpu_vk.device, g_buf, g_mem, 0) == VK_SUCCESS);
    void *p;
    CHECK(vkMapMemory(gpu_vk.device, g_mem, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS);
    g_words = p;
}

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = dst_stage,
                           .dstAccessMask = dst_access};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void finish(uint64_t v) {
    backing_submit_end(v);
    CHECK(gpu_vk_end(NULL, 0) == v);
    CHECK(gfx_vulkan_backend.fence_wait(v, 5000000000ull) == 1);
    gfx_vulkan_backend.cpu_synced(v);
}

// Fill each of n pages from addr with word + page index, through the resolver, in one submission.
static void fill_pages(uint64_t addr, int n, uint32_t word) {
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    for (int i = 0; i < n; i++) {
        BackingRef r;
        backing_resolve(addr + (uint64_t)i * PG, PG, v, &r);
        vkCmdFillBuffer(cb, (VkBuffer)(uintptr_t)r.buffer, r.offset, PG, word + (uint32_t)i);
    }
    finish(v);
}

// Bind [addr, addr + size) and copy its first 64 KiB at most into the readback buffer, through the resolver.
static void read_back(uint64_t addr, uint64_t size) {
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    BackingRef r;
    backing_resolve(addr, size, v, &r);
    VkCommandBuffer cb = gpu_vk_cmd();
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy c = {.srcOffset = r.offset, .dstOffset = 0, .size = size < 64 * KiB ? size : 64 * KiB};
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)r.buffer, g_buf, 1, &c);
    barrier(cb, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    finish(v);
}

static uint32_t device_word(uint64_t addr) {
    read_back(addr & ~(PG - 1), PG);
    return g_words[(addr & (PG - 1)) / 4];
}

// One raw-access submission of the program over the entries e: the table written into the pointer table's pool, the
// dispatch, the raw path's end; waited for and cpu_synced. The entries come back with the loaded words and old values.
static uint64_t dispatch(Entry e[THREADS]) {
    backing_sync();   // as the replay's submit does first
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    BackingRef r;
    backing_resolve(TABLE, THREADS * sizeof(Entry), v, &r);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cb, (VkBuffer)(uintptr_t)r.buffer, r.offset, THREADS * sizeof(Entry), e);
    barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    uint32_t push[4];
    gpu_vk_raw_begin(v, push);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    vkCmdPushConstants(cb, g_lay.compute, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, push);
    vkCmdDispatch(cb, 1, 1, 1);
    gpu_vk_raw_end(v);
    finish(v);
    read_back(TABLE, THREADS * sizeof(Entry));
    memcpy(e, (const void *)g_words, THREADS * sizeof(Entry));
    return v;
}

static void open_pipeline(const char *spv) {
    FILE *f = fopen(spv, "rb");
    CHECK(f);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)n);
    CHECK(code && fread(code, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)n,
                                   .pCode = code};
    VkShaderModule m;
    CHECK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &m) == VK_SUCCESS);
    free(code);
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = g_lay.compute};
    CHECK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &g_pipe) == VK_SUCCESS);
    vkDestroyShaderModule(gpu_vk.device, m, NULL);
}

static void open_all(const char *spv) {
    guest_heap(256 * MiB);
    gpu_vk_open();
    backing_set_mirror_hook(mirror_hook);
    make_buffer();
    vkp_layouts_create(&g_lay);
    open_pipeline(spv);
}

static void close_all(pool_t **pools, int n) {
    for (int i = 0; i < n; i++) gfx_vulkan_backend.pool_fini(pools[i]);
    vkDestroyPipeline(gpu_vk.device, g_pipe, NULL);
    vkp_layouts_destroy(&g_lay);
    vkDestroyBuffer(gpu_vk.device, g_buf, NULL);
    vkFreeMemory(gpu_vk.device, g_mem, NULL);
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
}

static void stats_line(void) {
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    BackingStats b;
    backing_stats(&b);
    printf("raw: submissions %llu, builds %llu (%.3f ms max), table %llu KiB, pages %llu (mirrored %llu), blocks %llu, "
           "span %llu; write-backs %llu (%llu bytes); misses %llu; invariant passes %llu, imports %llu (%llu KiB)\n",
           (ull)s.submissions, (ull)s.builds, (double)s.build_ns_max / 1e6, (ull)(s.table_bytes >> 10), (ull)s.pages,
           (ull)s.mirror_pages, (ull)s.blocks, (ull)s.span_count, (ull)s.writebacks, (ull)s.writeback_bytes,
           (ull)s.misses, (ull)b.raw_passes, (ull)b.raw_fill_imports, (ull)(b.raw_fill_bytes >> 10));
}

// The table's pool, and every thread pointed at a scratch word of it until a case points it elsewhere.
static pool_t table_pool(Entry e[THREADS]) {
    pool_t t = make_pool(TABLE, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    for (int i = 0; i < THREADS; i++)
        e[i] = (Entry){TABLE + 1 * MiB + (uint64_t)i * PG + PROBE, TABLE + 1 * MiB + 0x80000, 0, 0, {0}};
    return t;
}

// ---------- kinds
enum { K_IMPORT, K_MIRROR, K_EXTENT, K_VIRTUAL, KINDS };
static const char *const k_names[KINDS] = {"import", "mirror", "VRAM extent", "virtual over a physical pool"};

static void case_kinds(const char *spv) {
    open_all(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    const uint64_t imp = g_heap + 16 * MiB, mir = g_heap + 32 * MiB, ext = WIN + 16 * MiB, phys = WIN + 32 * MiB,
                   virt = WIN + 48 * MiB;
    g_mirror_lo = mir, g_mirror_hi = mir + 1 * MiB;
    pool_t pi = make_pool(imp, 2 * MiB, 0), pm = make_pool(mir, 1 * MiB, 0);
    pool_t pe = make_pool(ext, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    pool_t pp = make_pool(phys, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_PHYSICAL);
    pool_t pv = make_pool(virt, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL);
    map_run(&pv, &pp, 64 * KiB, 1 * MiB);
    // Each kind: 16 pages, page i holding tag + i; the counter, 0, at the base + 512 KiB.
    const uint64_t base[KINDS] = {imp, mir, ext, virt}, counter_off = 512 * KiB;
    const uint32_t tag[KINDS] = {0x11000000u, 0x22000000u, 0x33000000u, 0x44000000u};
    for (int i = 0; i < 16; i++) fill_words(imp + i * PG, PG, tag[K_IMPORT] + (uint32_t)i);
    for (int i = 0; i < 16; i++) fill_words(mir + i * PG, PG, tag[K_MIRROR] + (uint32_t)i);
    fill_words(imp + counter_off, PG, 0), fill_words(mir + counter_off, PG, 0);
    fill_words(mir + 32 * PG, PG, 0x5EED5EEDu);   // a mirrored page no thread writes
    read_back(mir, 1 * MiB);                     // the mirror is made at a bind, and brought current
    fill_pages(ext, 16, tag[K_EXTENT]), fill_pages(ext + counter_off, 1, 0);
    fill_pages(virt, 16, tag[K_VIRTUAL]), fill_pages(virt + counter_off, 1, 0);

    for (int k = 0; k < KINDS; k++) {
        for (int i = 0; i < 16; i++)
            e[k * 16 + i] = (Entry){base[k] + (uint64_t)i * PG + PROBE, base[k] + counter_off, 0, 0, {0}};
    }
    dispatch(e);
    static const int want_kind[KINDS] = {BACKING_IMPORT, BACKING_MIRROR, BACKING_EXTENT, BACKING_EXTENT};
    for (int k = 0; k < KINDS; k++) {
        BackingPage bp;
        backing_page(base[k], &bp);
        CHECK(bp.kind == want_kind[k]);   // the import is the invariant's, made at the submission's raw begin
        uint32_t olds = 0;
        for (int i = 0; i < 16; i++) {
            const Entry *x = &e[k * 16 + i];
            CHECK(x->loaded == tag[k] + (uint32_t)i);
            CHECK(x->old < 16 && !(olds >> x->old & 1));
            olds |= 1u << x->old;
        }
        printf("%s: 16 raw loads read their pages' words, 16 atomics returned 0 to 15 once each\n", k_names[k]);
    }
    // Stores and counters: guest memory for the import and the mirror (the mirror's through the raw write-back), a
    // copy through the resolver for the window.
    for (int k = 0; k < KINDS; k++) {
        int cpu = k == K_IMPORT || k == K_MIRROR;
        for (int i = 0; i < 16; i++) {
            uint64_t a = base[k] + (uint64_t)i * PG + PROBE;
            uint32_t want = MARK | (uint32_t)(k * 16 + i);
            CHECK((cpu ? word_at(a + 4) : device_word(a + 4)) == want);
            CHECK((cpu ? word_at(a + 8) : device_word(a + 8)) == tag[k] + (uint32_t)i);   // the word beside
        }
        CHECK((cpu ? word_at(base[k] + counter_off) : device_word(base[k] + counter_off)) == 16);
        printf("%s: the 16 raw stores and the counter (16) seen %s\n", k_names[k],
               cpu ? "in guest memory" : "through a copy");
    }
    CHECK(device_word(phys + 64 * KiB + PROBE + 4) == (MARK | 48));   // the virtual pool's bytes are the physical one's
    CHECK(word_at(mir + 32 * PG) == 0x5EED5EEDu);
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.submissions == 1 && s.misses == 0 && s.writebacks == 17 && s.writeback_bytes == 17 * PG);
    printf("the mirror: 17 pages written back (16 stored, the counter), its other pages untouched\n");
    stats_line();
    pool_t *all[] = {&pv, &pp, &pe, &pm, &pi, &t};
    close_all(all, 6);
    printf("ok kinds\n");
}

// ---------- keep
// What a raw-access submission stored into a mirror, written back into guest memory, stays in the mirror: the next
// submission copies none of it back (the write-back's own copy into guest memory is not a CPU write: the write watch's mark of
// it is taken away), a raw load of it reads the stored word, the counter goes on counting from the stored value, and a CPU
// write to one of those pages is copied in, a page and no more.
static void case_keep(const char *spv) {
    open_all(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    const uint64_t mir = g_heap + 32 * MiB, counter = mir + 512 * KiB;
    g_mirror_lo = mir, g_mirror_hi = mir + 1 * MiB;
    pool_t pm = make_pool(mir, 1 * MiB, 0);
    for (int i = 0; i < 16; i++) fill_words(mir + (uint64_t)i * PG, PG, 0x22000000u + (uint32_t)i);
    fill_words(counter, PG, 0);
    read_back(mir, 1 * MiB);   // the mirror is made at a bind, and brought current
    for (int i = 0; i < 16; i++) e[i] = (Entry){mir + (uint64_t)i * PG + PROBE, counter, 0, 0, {0}};
    dispatch(e);
    for (int i = 0; i < 16; i++) CHECK(word_at(mir + (uint64_t)i * PG + PROBE + 4) == (MARK | (uint32_t)i));
    CHECK(word_at(counter) == 16);
    BackingStats b0;
    backing_stats(&b0);
    CHECK(b0.upload_bytes > 0);   // the committed pages of the mirror went in once
    // The next submission reads the words the first stored, and counts on from 16, with nothing copied into the mirror.
    for (int i = 0; i < 16; i++) e[i] = (Entry){mir + (uint64_t)i * PG + PROBE + 4, counter, 0, 0, {0}};
    dispatch(e);
    for (int i = 0; i < 16; i++) CHECK(e[i].loaded == (MARK | (uint32_t)i));
    CHECK(word_at(counter) == 32);
    BackingStats b1;
    backing_stats(&b1);
    CHECK(b1.upload_bytes == b0.upload_bytes);
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.submissions == 2 && s.misses == 0 && s.writebacks == 34 && s.writeback_bytes == 34 * PG);
    // The CPU writes the counter's page: the third submission takes it from guest memory, one page, and counts from it.
    fill_words(counter, PG, 100);
    for (int i = 0; i < 16; i++) e[i] = (Entry){mir + (uint64_t)i * PG + PROBE + 4, counter, 0, 0, {0}};
    dispatch(e);
    CHECK(word_at(counter) == 116);
    BackingStats b2;
    backing_stats(&b2);
    CHECK(b2.upload_bytes == b1.upload_bytes + PG);
    printf("a mirror's raw stores, written back: the next submission copied %llu bytes into it (17 pages before the "
           "change), a CPU write to a page of it %llu\n", (ull)(b1.upload_bytes - b0.upload_bytes),
           (ull)(b2.upload_bytes - b1.upload_bytes));
    stats_line();
    pool_t *all[] = {&pm, &t};
    close_all(all, 2);
    printf("ok keep\n");
}

// ---------- miss
static void case_miss(const char *spv) {
    open_all(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    const uint64_t imp = g_heap + 16 * MiB, none = g_heap + 128 * MiB;   // guest memory in no pool
    pool_t pi = make_pool(imp, 1 * MiB, 0);
    fill_words(imp, 1 * MiB, 0x01010101u);
    fill_words(none, 16 * PG, 0x77777777u);
    for (int i = 0; i < 8; i++) e[i] = (Entry){none + (uint64_t)i * PG + PROBE, none + 8 * PG, 0, 0, {0}};
    for (int i = 8; i < THREADS; i++) e[i] = (Entry){imp + (uint64_t)(i % 16) * PG + PROBE, imp + 32 * PG, 0, 0, {0}};
    dispatch(e);
    for (int i = 0; i < 8; i++) CHECK(e[i].loaded == 0 && e[i].old == 0);
    for (int i = 8; i < THREADS; i++) CHECK(e[i].loaded == 0x01010101u);
    for (int i = 0; i < 16; i++) CHECK(word_at(none + i * PG + PROBE + 4) == 0x77777777u);
    CHECK(word_at(none + 8 * PG) == 0x77777777u);
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 24 && s.miss_submissions == 1 && s.last_misses == 24);
    int seen[4] = {0};
    for (uint32_t i = 0; i < s.last_misses; i++) {
        const RawMiss *m = &s.last_miss[i];
        CHECK(m->reason == RAW_MISS_REASON_NO_CHUNK && m->kind >= 1 && m->kind <= 3);
        uint64_t want = m->kind == RAW_MISS_KIND_ATOMIC ? none + 8 * PG : none + (m->address - none) / PG * PG + PROBE +
                                                                               (m->kind == RAW_MISS_KIND_STORE ? 4 : 0);
        CHECK(m->address == want);
        seen[m->kind]++;
    }
    CHECK(seen[RAW_MISS_KIND_LOAD] == 8 && seen[RAW_MISS_KIND_STORE] == 8 && seen[RAW_MISS_KIND_ATOMIC] == 8);
    printf("8 threads at guest memory in no pool: loads read 0, stores and atomics dropped; 24 misses logged (8 loads, "
           "8 stores, 8 atomics), each with its address and the reason no chunk\n");
    for (int i = 0; i < 8; i++) e[i] = e[8 + i];
    dispatch(e);
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 24 && s.miss_submissions == 1);
    printf("the next submission, with every pointer mapped, logged no miss\n");
    stats_line();
    pool_t *all[] = {&pi, &t};
    close_all(all, 2);
    printf("ok miss\n");
}

// ---------- commit
static void case_commit(const char *spv) {
    open_all(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    const uint64_t p = g_heap + 64 * MiB;
    pool_t pp = make_pool(p, 4 * MiB, 0);
    fill_words(p, 1 * MiB, 0x0F0F0F0Fu);   // the first chunk only
    int c;
    uint64_t run = rt_mem_commit_run(p + 3 * MiB, 1 * MiB, &c);
    CHECK(run != UINT64_MAX && !c);       // not committed yet
    e[0] = (Entry){p + 3 * MiB + PROBE, p, 0, 0, {0}};
    dispatch(e);
    CHECK(e[0].loaded == 0);
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 2 && s.last_miss[0].address == p + 3 * MiB + PROBE);
    printf("a pointer into the pool's uncommitted part: a miss (load and store)\n");
    fill_words(p + 3 * MiB, PG, 0xC0C0C0C0u);   // the guest commits the page and writes it
    uint64_t run2 = rt_mem_commit_run(p + 3 * MiB, PG, &c);
    CHECK(run2 != UINT64_MAX && c);
    dispatch(e);
    CHECK(e[0].loaded == 0xC0C0C0C0u && word_at(p + 3 * MiB + PROBE + 4) == MARK);
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 2);
    printf("committed after the pool was mapped: the next submission read the CPU's word and stored beside it\n");
    stats_line();
    pool_t *all[] = {&pp, &t};
    close_all(all, 2);
    printf("ok commit\n");
}

// ---------- edges
static void case_edges(const char *spv) {
    open_all(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    const uint64_t low = g_heap, high = WIN + GFX_GPU_WINDOW_SIZE - 2 * MiB;
    pool_t pl = make_pool(low, 1 * MiB, 0);
    pool_t ph = make_pool(high, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    fill_words(low, 1 * MiB, 0x10101010u);
    fill_pages(high + 2 * MiB - PG, 1, 0x20202020u);
    e[0] = (Entry){low + PROBE, low + 8 * PG, 0, 0, {0}};
    e[1] = (Entry){high + 2 * MiB - PG + 0xF00, high + 2 * MiB - PG + 0xF80, 0, 0, {0}};
    BackingRun *runs;
    uint64_t version;
    dispatch(e);
    int n = backing_raw_runs(&runs, &version);
    uint64_t lo = UINT64_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        if (runs[i].lo < lo) lo = runs[i].lo;
        if (runs[i].hi > hi) hi = runs[i].hi;
    }
    free(runs);
    CHECK(lo == low && hi == high + 2 * MiB);   // these are the lowest and the highest pages the table maps
    CHECK(e[0].loaded == 0x10101010u && e[1].loaded == 0x20202020u);
    CHECK(word_at(low + PROBE + 4) == MARK && device_word(high + 2 * MiB - PG + 0xF04) == (MARK | 1));
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 0);
    printf("the lowest mapped page (%#llx) and the highest (%#llx) resolve; the span is %llu chunks\n", (ull)low,
           (ull)(hi - PG), (ull)s.span_count);
    stats_line();
    pool_t *all[] = {&ph, &pl, &t};
    close_all(all, 3);
    printf("ok edges\n");
}

// ---------- reuse
static void case_reuse(const char *spv) {
    open_all(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    const uint64_t cpu = g_heap + 16 * MiB, win = WIN + 16 * MiB;
    pool_t a = make_pool(cpu, 1 * MiB, 0), w = make_pool(win, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    fill_words(cpu, 1 * MiB, 0xAAAAAAAAu);
    fill_pages(win, 1, 0xBBBBBBBBu);
    e[0] = (Entry){cpu + PROBE, cpu + 8 * PG, 0, 0, {0}};
    e[1] = (Entry){win + PROBE, win + 8 * PG, 0, 0, {0}};
    dispatch(e);
    CHECK(e[0].loaded == 0xAAAAAAAAu && e[1].loaded == 0xBBBBBBBBu);
    gfx_vulkan_backend.pool_fini(&a), gfx_vulkan_backend.pool_fini(&w);
    // New pools over the same ranges, the CPU-visible one a mirror this time, made and written before the old backings'
    // deferred release ran.
    g_mirror_lo = cpu, g_mirror_hi = cpu + 1 * MiB;
    pool_t b = make_pool(cpu, 1 * MiB, 0), x = make_pool(win, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    fill_words(cpu, 1 * MiB, 0xCCCCCCCCu);
    read_back(cpu, 64 * KiB);   // the mirror is made at a bind
    fill_pages(win, 1, 0xDDDDDDDDu);
    BackingPage bp;
    backing_page(cpu, &bp);
    CHECK(bp.kind == BACKING_MIRROR);
    for (int i = 0; i < 2; i++) e[i].loaded = e[i].old = 0;
    dispatch(e);
    CHECK(e[0].loaded == 0xCCCCCCCCu && e[1].loaded == 0xDDDDDDDDu);
    CHECK(word_at(cpu + PROBE + 4) == MARK && device_word(win + PROBE + 4) == (MARK | 1));
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 0 && s.builds >= 2);
    printf("pools finalized and others made over their ranges: the raw path reached the new pools' words\n");
    stats_line();
    pool_t *all[] = {&x, &b, &t};
    close_all(all, 3);
    printf("ok reuse\n");
}

// ---------- grow
// One stage of the program in the submission being recorded: the entries it reads, then its dispatch with the table's push
// words. The entries' pool is bound once by the caller.
static void stage(VkCommandBuffer cb, const Entry e[THREADS], const uint32_t push[4], const BackingRef *entries) {
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdUpdateBuffer(cb, (VkBuffer)(uintptr_t)entries->buffer, entries->offset, THREADS * sizeof(Entry), e);
    barrier(cb, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    vkCmdPushConstants(cb, g_lay.compute, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    vkCmdDispatch(cb, 1, 1, 1);
}

static void case_grow(const char *spv) {
    open_all(spv);
    Entry e[THREADS], first[THREADS];
    pool_t t = table_pool(e);
    const uint64_t p = g_heap + 16 * MiB;
    g_mirror_lo = p, g_mirror_hi = p + 1 * MiB;
    pool_t pp = make_pool(p, 16 * MiB, 0);
    fill_words(p, 256 * KiB, 0x11110000u);   // what the CPU wrote: the run a mirror will cover
    memcpy(first, e, sizeof first);
    first[0] = (Entry){p + 0x2000 + PROBE, p + 0x3000, 0, 0, {0}};   // stage 1: a word of the mirror and a counter in it
    // Stage 2: pages the guest never touched, which the bind commits, and one counter in them (inside the 64 KiB the bound
    // copy reads back)
    const uint64_t fresh = p + 0x80000, counter = fresh + 15 * PG + 0x800;
    for (int i = 0; i < 16; i++) e[i] = (Entry){fresh + (uint64_t)i * PG + PROBE, counter, 0, 0, {0}};

    backing_sync();   // as the replay's submit does first
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    BackingRef entries, small, big;
    backing_resolve(TABLE, THREADS * sizeof(Entry), v, &entries);
    backing_resolve(p, 0x1200, v, &small);   // a small bind: the mirror over the 256 KiB
    CHECK(small.kind == BACKING_MIRROR);
    uint32_t push1[4], push2[4];
    gpu_vk_raw_begin(v, push1);   // the table of v maps the mirror
    stage(cb, first, push1, &entries);
    backing_resolve(p + 0x30320, 1 * MiB, v, &big);   // starts in the mirror's last pages, ends far past it
    CHECK(big.kind == BACKING_IMPORT);
    gpu_vk_raw_begin(v, push2);
    CHECK(!memcmp(push1, push2, sizeof push1));   // the second stage takes the first one's table: the same address and span
    stage(cb, e, push2, &entries);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy c = {.srcOffset = big.offset + (fresh - (p + 0x30320)), .dstOffset = 0, .size = 64 * KiB};
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)big.buffer, g_buf, 1, &c);   // the same bytes through the bind's backing
    barrier(cb, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    gpu_vk_raw_end(v);
    finish(v);
    uint32_t bound[16], bound_counter = g_words[(counter - fresh) / 4];
    for (int i = 0; i < 16; i++) bound[i] = g_words[(uint64_t)i * PG / 4 + (PROBE + 4) / 4];
    read_back(TABLE, THREADS * sizeof(Entry));
    memcpy(e, (const void *)g_words, 16 * sizeof(Entry));

    uint32_t olds = 0;
    for (int i = 0; i < 16; i++) {
        CHECK(e[i].loaded == 0 && e[i].old < 16 && !(olds >> e[i].old & 1));   // 16 atomics, each saw another count
        olds |= 1u << e[i].old;
        CHECK(word_at(fresh + (uint64_t)i * PG + PROBE + 4) == (MARK | (uint32_t)i));   // the stores landed in guest memory
        CHECK(bound[i] == (MARK | (uint32_t)i));   // and the bind's backing, read on the device, holds the same bytes
    }
    CHECK(word_at(counter) == 16 && bound_counter == 16);
    CHECK(word_at(p + 0x2000 + PROBE + 4) == MARK && word_at(p + 0x3000) == 0x11110001u);   // stage 1, through the mirror
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    BackingStats b;
    backing_stats(&b);
    CHECK(s.misses == 0 && s.submissions == 1 && s.builds == 1 && s.refreshes == 1);
    CHECK(s.writebacks == 2 && s.writeback_bytes == 2 * PG);   // the mirror's kept pages that stage 1 stored into
    CHECK(b.mirrors == 1 && b.mirror_bytes == 0x30000 && b.imports == 1 && b.import_bytes == 0x110000);
    BackingPage pg;
    backing_page(p + 0x2F000, &pg);
    CHECK(pg.kind == BACKING_MIRROR);
    backing_page(p + 0x30000, &pg);
    CHECK(pg.kind == BACKING_IMPORT);
    printf("one submission, two stages, two binds: the second stage's 16 loads, stores and atomics reached the pages the "
           "bind committed past the mirror, the bound copy and guest memory hold the same bytes, the mirror kept %llu "
           "KiB and its stored pages were written back\n", (ull)(b.mirror_bytes >> 10));
    stats_line();
    pool_t *all[] = {&pp, &t};
    close_all(all, 2);
    printf("ok grow\n");
}

static void case_grow_taken(const char *spv) {
    open_all(spv);
    Entry e[THREADS], first[THREADS], got_first;
    pool_t t = table_pool(e);
    const uint64_t p = g_heap + 16 * MiB;
    const uint64_t old = p + 0x31000, fresh = p + 0x80000, slack = p + 0x13F000;
    g_mirror_lo = p, g_mirror_hi = p + 1 * MiB;
    pool_t pp = make_pool(p, 16 * MiB, 0);
    fill_words(p, 256 * KiB, 0xAAAAAAAAu);
    memcpy(first, e, sizeof first);
    first[0] = (Entry){old + PROBE, old + PG, 0, 0, {0}};
    e[0] = (Entry){old + PROBE + 4, old + PG, 0, 0, {0}};
    e[1] = (Entry){fresh + PROBE, fresh + 0x800, 0, 0, {0}};
    e[2] = (Entry){slack + PROBE, slack + 0x800, 0, 0, {0}};

    backing_sync();
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    BackingRef entries, small, big;
    backing_resolve(TABLE, THREADS * sizeof(Entry), v, &entries);
    backing_resolve(old, 0x100, v, &small);
    CHECK(small.kind == BACKING_MIRROR);
    uint32_t push1[4], push2[4];
    gpu_vk_raw_begin(v, push1);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cb, (VkBuffer)(uintptr_t)small.buffer, small.offset + PROBE, 4, 0x1234ABCDu);
    backing_gpu_wrote(old + PROBE, 4, v);
    stage(cb, first, push1, &entries);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy first_copy = {.srcOffset = entries.offset, .dstOffset = 0, .size = sizeof(Entry)};
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)entries.buffer, g_buf, 1, &first_copy);

    backing_resolve(p + 0x30320, 1 * MiB, v, &big);
    CHECK(big.kind == BACKING_MIRROR && big.address + 0x31000 - 0x30320 == small.address);
    int committed;
    CHECK(rt_mem_commit_run(slack, PG, &committed) != UINT64_MAX && committed);
    gpu_vk_raw_begin(v, push2);
    CHECK(!memcmp(push1, push2, sizeof push1));
    stage(cb, e, push2, &entries);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy old_copy = {.srcOffset = small.offset + PROBE, .dstOffset = 4096, .size = 16};
    VkBufferCopy fresh_copy = {.srcOffset = big.offset + fresh - (p + 0x30320) + PROBE, .dstOffset = 8192, .size = 8};
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)small.buffer, g_buf, 1, &old_copy);
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)big.buffer, g_buf, 1, &fresh_copy);
    barrier(cb, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    gpu_vk_raw_end(v);
    finish(v);
    memcpy(&got_first, (const void *)g_words, sizeof got_first);
    uint32_t bound_old[3], bound_fresh[2];
    for (int i = 0; i < 3; i++) bound_old[i] = g_words[4096 / 4 + i];
    for (int i = 0; i < 2; i++) bound_fresh[i] = g_words[8192 / 4 + i];
    read_back(TABLE, THREADS * sizeof(Entry));
    memcpy(e, (const void *)g_words, 3 * sizeof(Entry));

    CHECK(got_first.loaded == 0x1234ABCDu && got_first.old == 0xAAAAAAAAu);
    CHECK(e[0].loaded == MARK && e[0].old == 0xAAAAAAABu);
    CHECK(e[1].loaded == 0 && e[1].old == 0 && e[2].loaded == 0 && e[2].old == 0);
    CHECK(bound_old[0] == 0x1234ABCDu && bound_old[1] == MARK && bound_old[2] == MARK);
    CHECK(bound_fresh[0] == 0 && bound_fresh[1] == (MARK | 1));
    CHECK(word_at(old + PROBE) == 0x1234ABCDu && word_at(old + PROBE + 4) == MARK &&
          word_at(old + PROBE + 8) == MARK && word_at(old + PG) == 0xAAAAAAACu);
    CHECK(word_at(fresh + PROBE + 4) == (MARK | 1) && word_at(fresh + 0x800) == 1);
    CHECK(word_at(slack + PROBE + 4) == (MARK | 2) && word_at(slack + 0x800) == 1);
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 0 && s.submissions == 1 && s.refreshes == 1);
    printf("one submission: the old and grown binds, both raw stages and guest write-back share the same bytes; "
           "the committed page beyond the bind is mapped\n");
    stats_line();
    pool_t *all[] = {&pp, &t};
    close_all(all, 2);
    printf("ok grow_taken\n");
}

static void case_grow_guarded(const char *spv, int head) {
    open_all(spv);
    Entry e[THREADS], first[THREADS], got_first;
    pool_t t = table_pool(e);
    const uint64_t p = g_heap + 16 * MiB, q = WIN + 16 * MiB, old = p + 0x2000;
    const uint64_t added = head ? p - 0x8000 : p + 0x18000;
    const uint64_t start = head ? p - 0x5000 : p + 0x1000;
    const uint64_t size = head ? 0x8000 : 0x11000;
    g_mirror_lo = p - 1 * MiB, g_mirror_hi = p + 1 * MiB;
    pool_t pp = make_pool(p - 1 * MiB, 16 * MiB, 0);
    pool_t extent = make_pool(q, 64 * KiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    fill_words(p, 64 * KiB, 0xAAAAAAAAu);

    uint64_t v0 = gpu_vk_begin(GPU_VK_INTERNAL);
    BackingRef small, blocker;
    if (head) backing_resolve(q, 4, v0, &blocker);
    backing_resolve(old, 0x100, v0, &small);
    if (!head) backing_resolve(q, 4, v0, &blocker);
    CHECK(small.kind == BACKING_MIRROR && blocker.kind == BACKING_EXTENT && small.buffer == blocker.buffer);
    if (head) CHECK(blocker.offset + 64 * KiB + BACKING_MIRROR_GUARD + 0x2000 == small.offset);
    else CHECK(small.offset - 0x2000 + 64 * KiB + BACKING_MIRROR_GUARD == blocker.offset);
    VkCommandBuffer cb = gpu_vk_cmd();
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cb, (VkBuffer)(uintptr_t)blocker.buffer, blocker.offset, 4, 0x11111111u);
    finish(v0);

    fill_words(added + PROBE, 4, 0xBBBBBBBBu);
    memcpy(first, e, sizeof first);
    first[0] = (Entry){old + PROBE, old + PG, 0, 0, {0}};
    e[0] = (Entry){old + PROBE + 4, old + PG, 0, 0, {0}};
    e[1] = (Entry){added + PROBE, added + 0x800, 0, 0, {0}};

    backing_sync();
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    cb = gpu_vk_cmd();
    BackingRef entries, big, fresh;
    backing_resolve(TABLE, THREADS * sizeof(Entry), v, &entries);
    backing_resolve(q, 4, v, &blocker);
    backing_resolve(old, 0x100, v, &small);
    uint32_t push1[4], push2[4];
    gpu_vk_raw_begin(v, push1);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cb, (VkBuffer)(uintptr_t)blocker.buffer, blocker.offset, 4, 0xEEEEEEEEu);
    vkCmdFillBuffer(cb, (VkBuffer)(uintptr_t)small.buffer, small.offset + PROBE, 4, 0x1234ABCDu);
    backing_gpu_wrote(old + PROBE, 4, v);
    stage(cb, first, push1, &entries);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy first_copy = {.srcOffset = entries.offset, .dstOffset = 0, .size = sizeof(Entry)};
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)entries.buffer, g_buf, 1, &first_copy);

    backing_resolve(start, size, v, &big);
    backing_resolve(added, PG, v, &fresh);
    CHECK(big.kind == BACKING_MIRROR && fresh.kind == BACKING_IMPORT);   // the growth took the bind's pages, not the chunk's
    CHECK(big.address + old - start == small.address);
    gpu_vk_raw_begin(v, push2);
    CHECK(!memcmp(push1, push2, sizeof push1));
    stage(cb, e, push2, &entries);
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy old_copy = {.srcOffset = small.offset + PROBE, .dstOffset = 4096, .size = 12};
    VkBufferCopy added_copy = {.srcOffset = fresh.offset + PROBE, .dstOffset = 8192, .size = 8};
    VkBufferCopy extent_copy = {.srcOffset = blocker.offset, .dstOffset = 12288, .size = 4};
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)small.buffer, g_buf, 1, &old_copy);
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)fresh.buffer, g_buf, 1, &added_copy);
    vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)blocker.buffer, g_buf, 1, &extent_copy);
    barrier(cb, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    gpu_vk_raw_end(v);
    finish(v);

    memcpy(&got_first, (const void *)g_words, sizeof got_first);
    uint32_t bound_old[3], bound_added[2], bound_extent = g_words[12288 / 4];
    for (int i = 0; i < 3; i++) bound_old[i] = g_words[4096 / 4 + i];
    for (int i = 0; i < 2; i++) bound_added[i] = g_words[8192 / 4 + i];
    read_back(TABLE, THREADS * sizeof(Entry));
    memcpy(e, (const void *)g_words, 2 * sizeof(Entry));
    CHECK(got_first.loaded == 0x1234ABCDu && got_first.old == 0xAAAAAAAAu);
    CHECK(e[0].loaded == MARK && e[0].old == 0xAAAAAAABu);
    CHECK(e[1].loaded == 0xBBBBBBBBu && e[1].old == 0);
    CHECK(bound_old[0] == 0x1234ABCDu && bound_old[1] == MARK && bound_old[2] == MARK);
    CHECK(bound_added[0] == 0xBBBBBBBBu && bound_added[1] == (MARK | 1));
    CHECK(bound_extent == 0xEEEEEEEEu);
    CHECK(word_at(old + PROBE) == 0x1234ABCDu && word_at(old + PROBE + 4) == MARK &&
          word_at(old + PROBE + 8) == MARK && word_at(old + PG) == 0xAAAAAAACu);
    CHECK(word_at(added + PROBE) == 0xBBBBBBBBu && word_at(added + PROBE + 4) == (MARK | 1));
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    CHECK(s.misses == 0 && s.submissions == 1 && s.refreshes == 1);
    stats_line();
    pool_t *all[] = {&extent, &pp, &t};
    close_all(all, 3);
    printf("ok grow_guarded_%s\n", head ? "head" : "tail");
}

// ---------- heap
static void mem_line(const char *when) {
    RtMemStats m;
    rt_mem_measure(&m);
    uint64_t used, budget;
    CHECK(gpu_vk_backing_ops.vram(&used, &budget));
    printf("%s: heap committed %llu MiB, resident %llu MiB; process private %llu MiB, working set %llu MiB; VRAM use "
           "%llu MiB of a budget of %llu MiB\n", when, (ull)(m.r[RT_MEM_HEAP].committed >> 20),
           (ull)(m.r[RT_MEM_HEAP].resident >> 20), (ull)(m.process_private >> 20), (ull)(m.process_working_set >> 20),
           (ull)(used >> 20), (ull)(budget >> 20));
}

static void case_heap(const char *spv, uint64_t mib) {
    const uint64_t pool_size = 256 * MiB, size = (mib * MiB + pool_size - 1) / pool_size * pool_size;
    RtMemConfig cfg = rt_mem_default_config();
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = size;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1];
    gpu_vk_open();
    make_buffer();
    vkp_layouts_create(&g_lay);
    open_pipeline(spv);
    Entry e[THREADS];
    pool_t t = table_pool(e);
    int n = (int)(size / pool_size);
    pool_t *pools = calloc((size_t)n + 1, sizeof *pools);
    CHECK(pools);
    for (int i = 0; i < n; i++) pools[i] = make_pool(g_heap + (uint64_t)i * pool_size, pool_size, 0);
    pools[n] = make_pool(WIN + 2048 * MiB, pool_size, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    uint64_t t0 = plat_now_ns();
    for (uint64_t a = g_heap; a < g_heap + size; a += PG) *(volatile uint32_t *)(uintptr_t)a = (uint32_t)(a >> 12);
    printf("heap of %llu MiB in %d pools of 256 MiB, every page written in %.0f ms\n", (ull)(size >> 20), n,
           (double)(plat_now_ns() - t0) / 1e6);
    mem_line("before");
    const uint64_t last = g_heap + size - PG;
    e[0] = (Entry){g_heap, g_heap + 8, 0, 0, {0}};   // each page's first word holds its page number
    e[1] = (Entry){last, last + 8, 0, 0, {0}};
    t0 = plat_now_ns();
    dispatch(e);
    double first = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(e[0].loaded == (uint32_t)(g_heap >> 12) && e[1].loaded == (uint32_t)(last >> 12));
    mem_line("after the first raw-access submission");
    GpuVkRawStats s;
    gpu_vk_raw_stats(&s);
    BackingStats b;
    backing_stats(&b);
    printf("first raw-access submission %.1f ms (submitted and waited): invariant pass made %llu imports (%llu MiB) "
           "and %llu extent; table built in %.2f ms, %llu KiB, %llu pages, %llu leaf blocks, span %llu chunks\n", first,
           (ull)b.raw_fill_imports, (ull)(b.raw_fill_bytes >> 20), (ull)b.raw_fill_extents, (double)s.build_ns / 1e6,
           (ull)(s.table_bytes >> 10), (ull)s.pages, (ull)s.blocks, (ull)s.span_count);
    t0 = plat_now_ns();
    dispatch(e);
    double second = (double)(plat_now_ns() - t0) / 1e6;
    gpu_vk_raw_stats(&s);
    CHECK(s.builds == 1 && s.misses == 0);
    printf("second raw-access submission, the map unchanged: %.1f ms (submitted and waited), no pass, no build\n",
           second);
    mem_line("after the second");
    pool_t **all = calloc((size_t)n + 2, sizeof *all);
    CHECK(all);
    for (int i = 0; i <= n; i++) all[i] = &pools[i];
    all[n + 1] = &t;
    close_all(all, n + 2);
    printf("ok heap\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc >= 2 && !strcmp(argv[1], "usable")) {
        case_usable();
        return 0;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: vk_raw_test <case> <raw.spv>\n");
        return 2;
    }
    const char *cs = argv[1], *spv = argv[2];
    if (!strcmp(cs, "kinds")) case_kinds(spv);
    else if (!strcmp(cs, "keep")) case_keep(spv);
    else if (!strcmp(cs, "miss")) case_miss(spv);
    else if (!strcmp(cs, "commit")) case_commit(spv);
    else if (!strcmp(cs, "edges")) case_edges(spv);
    else if (!strcmp(cs, "reuse")) case_reuse(spv);
    else if (!strcmp(cs, "grow")) case_grow(spv);
    else if (!strcmp(cs, "grow_taken")) case_grow_taken(spv);
    else if (!strcmp(cs, "grow_guarded_head")) case_grow_guarded(spv, 1);
    else if (!strcmp(cs, "grow_guarded_tail")) case_grow_guarded(spv, 0);
    else if (!strcmp(cs, "heap") && argc >= 4) case_heap(spv, strtoull(argv[3], NULL, 10));
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
