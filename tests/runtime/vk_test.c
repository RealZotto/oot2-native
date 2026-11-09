// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the Vulkan back end on a real device (runtime/gpu, oot2/gpu_vk.h), driven by tests/test_runtime_vk.py:
// `vk_test <case> [folder]`, one case per process. They need a Vulkan device and skip without one, unless
// OOT2_REQUIRE_GPU is set; continuous integration has none.
//
//   devices          every device, judged by the items of oot2/vk_required.h: what each lacks (no device is opened)
//   open             the device opens with every item, and closes
//   submits          1,000 empty replayed submissions from two threads, mixed with 100 of the back end's own that fill a
//                    word with their value on the GPU: values in order, each effect after the GPU's write and before
//                    its value is done
//   wait             a submission held by a semaphore the test signals: a fence wait times out, a later one succeeds,
//                    finish leaves nothing in flight
//   loss_submit      vkQueueSubmit2 answers VK_ERROR_DEVICE_LOST: the report is written to the folder, the run stops
//   loss_completion  the completion thread's vkWaitSemaphores answers it: the same, from that thread
//   refuse           OOT2_VK_DEVICE names no device: the start stops by name
//   backing, backing_fini, backing_flat   the backing map's imports and extents
//   mirror_upload    a CPU write into a mirrored range is read by the GPU in the next submission, three times, and
//                    only the written page is copied
//   mirror_writeback a GPU write into a mirrored range reaches guest memory after its fence and cpu_synced, never
//                    before (the submission is held by a semaphore meanwhile)
//   mirror_budget    VK_EXT_memory_budget's figures move with VRAM taken outside the map, and move the admission
//
// The loss cases replace Volk's entry points for the failing call and for the two queries that are valid only on a lost
// device (the fault and the checkpoints) with answers made here, so the whole path runs without hanging the GPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#include "oot2/vk_rec.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3   // no loader or no device: the driver skips

static void close_and_report(void) {
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (unsigned long long)errors, (unsigned long long)warnings);
}

// ---------- devices: the judgement of every device, without opening one
static void case_devices(void) {
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
    if (!n) {
        printf("no Vulkan device\n");
        exit(EXIT_NO_DEVICE);
    }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, devs) >= 0);
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
        int count = oot2_vk_judge(&f, has);
        printf("device %u\t%s\tVulkan %u.%u.%u\t%d of %d", i, f.props.deviceName, VK_API_VERSION_MAJOR(f.props.apiVersion),
               VK_API_VERSION_MINOR(f.props.apiVersion), VK_API_VERSION_PATCH(f.props.apiVersion), count,
               OOT2_VK_ITEM_COUNT);
        for (int k = 0, first = 1; k < OOT2_VK_ITEM_COUNT; k++)
            if (!has[k]) printf("%s%s", first ? "\tlacks " : ", ", oot2_vk_item_id(k)), first = 0;
        printf("\n");
        printf("facts %u\tsubgroup stages %#x operations %#x\trequired-size stages %#x\tvendor binary %u\n", i,
               f.p11.subgroupSupportedStages, f.p11.subgroupSupportedOperations, f.p13.requiredSubgroupSizeStages,
               f.fault.deviceFaultVendorBinary);
        usable += count == OOT2_VK_ITEM_COUNT;
    }
    vkDestroyInstance(inst, NULL);
    printf("usable %d\n", usable);
    printf("ok devices\n");
}

static void case_open(void) {
    gpu_vk_open();
    CHECK(gpu_vk.device && gpu_vk.queue && gpu_vk.timeline && gpu_vk.pool);
    unsigned char has[OOT2_VK_ITEM_COUNT];
    int count = oot2_vk_judge(&gpu_vk.facts, has);
    printf("opened %s, %d of %d items, queue family %u, fault %d, checkpoints %d\n", gpu_vk.facts.props.deviceName,
           count, OOT2_VK_ITEM_COUNT, gpu_vk.family, gpu_vk.fault, gpu_vk.checkpoints);
    CHECK(count == OOT2_VK_ITEM_COUNT);
    gpu_vk_open();   // a second call returns at once
    close_and_report();
    printf("ok open\n");
}

// ---------- a host-visible buffer the GPU writes words into
static VkBuffer g_buf;
static VkDeviceMemory g_mem;
static volatile uint32_t *g_words;
#define WORDS 4096

static void make_buffer(void) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = WORDS * 4,
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
    memset(p, 0, WORDS * 4);
}

static void free_buffer(void) {
    vkUnmapMemory(gpu_vk.device, g_mem);
    vkDestroyBuffer(gpu_vk.device, g_buf, NULL);
    vkFreeMemory(gpu_vk.device, g_mem, NULL);
}

// Records "write value into its word, then make it visible to the host" into the open submission's command buffer.
static void record_fill(uint64_t value) {
    VkCommandBuffer cb = gpu_vk_cmd();
    vkCmdFillBuffer(cb, g_buf, (value % WORDS) * 4, 4, (uint32_t)value);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1,
                           .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// ---------- submits
#define SUBMITTERS 2
#define REPLAYS_PER_THREAD 500
#define INTERNAL_EVERY 10
static _Alignas(8) uint8_t g_empty[16];   // {OP_BEGIN, OP_END}
static uint64_t g_internal_done, g_last_effect;
static int g_left = SUBMITTERS;
static PlatMutex g_join_lock = PLAT_MUTEX_INIT;
static PlatCond g_join_cond = PLAT_COND_INIT;

// The completion thread: the GPU wrote this value's word, every earlier value is done, its own is not.
static void fill_effect(void *arg, uint64_t value) {
    (void)arg;
    CHECK(gpu_sync_done() == value - 1);
    CHECK(g_words[value % WORDS] == (uint32_t)value);
    CHECK(value > g_last_effect);
    g_last_effect = value;
    g_internal_done++;
}

static void submitter(void *arg) {
    uint64_t last = 0, handle = (uint64_t)(uintptr_t)g_empty;
    for (int i = 0; i < REPLAYS_PER_THREAD; i++) {
        uint64_t v = gfx_vulkan_backend.submit(1, &handle, 1);
        CHECK(v > last);
        last = v;
        if (i % INTERNAL_EVERY == (int)(uintptr_t)arg) {
            uint64_t w = gpu_vk_begin(GPU_VK_INTERNAL);
            record_fill(w);
            gpu_vk_effect(fill_effect, NULL);
            CHECK(gpu_vk_end(NULL, 0) == w && w > last);
            last = w;
            if (i % 50 == 0) {   // a guest that waits: the effect ran before the value was done
                CHECK(gfx_vulkan_backend.fence_wait(w, UINT64_MAX) == 1);
                CHECK(g_last_effect >= w && gfx_vulkan_backend.fence_done() >= w);
            }
        }
    }
    plat_mutex_lock(&g_join_lock);
    g_left--;
    plat_cond_broadcast(&g_join_cond);
    plat_mutex_unlock(&g_join_lock);
}

static void case_submits(void) {
    op_hdr rec[2] = {{OP_BEGIN, 8}, {OP_END, 8}};
    memcpy(g_empty, rec, sizeof rec);
    gpu_vk_open();
    make_buffer();
    uint64_t t0 = plat_now_ns();
    for (int i = 0; i < SUBMITTERS; i++) CHECK(plat_thread_start(submitter, (void *)(uintptr_t)i, 1 << 20));
    plat_mutex_lock(&g_join_lock);
    while (g_left) plat_cond_wait(&g_join_cond, &g_join_lock);
    plat_mutex_unlock(&g_join_lock);
    gfx_vulkan_backend.finish(0);
    double ms = (double)(plat_now_ns() - t0) / 1e6;
    uint64_t handed = gpu_sync_handed(), counter = 0;
    CHECK(vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter) == VK_SUCCESS);
    const uint64_t internal = SUBMITTERS * REPLAYS_PER_THREAD / INTERNAL_EVERY;
    printf("values %llu (replayed %d, internal %llu, effects run %llu), done %llu, device counter %llu, %.1f ms\n",
           (unsigned long long)handed, SUBMITTERS * REPLAYS_PER_THREAD, (unsigned long long)internal,
           (unsigned long long)g_internal_done, (unsigned long long)gfx_vulkan_backend.fence_done(),
           (unsigned long long)counter, ms);
    CHECK(handed == SUBMITTERS * REPLAYS_PER_THREAD + internal);
    CHECK(g_internal_done == internal);
    CHECK(gfx_vulkan_backend.fence_done() == handed && counter == handed);
    free_buffer();
    close_and_report();
    printf("ok submits\n");
}

// ---------- wait: a submission held by a semaphore the host signals
static int g_gated_ran;
static void gated_effect(void *arg, uint64_t value) {
    (void)arg, (void)value;
    __atomic_store_n(&g_gated_ran, 1, __ATOMIC_RELEASE);
}

static VkSemaphore make_gate(void) {
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    VkSemaphore gate;
    CHECK(vkCreateSemaphore(gpu_vk.device, &si, NULL, &gate) == VK_SUCCESS);
    return gate;
}

static void case_wait(void) {
    gpu_vk_open();
    VkSemaphore gate = make_gate();
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    gpu_vk_cmd();
    gpu_vk_effect(gated_effect, NULL);
    CHECK(gpu_vk_end(&wait, 1) == v);
    uint64_t t0 = plat_now_ns();
    CHECK(gfx_vulkan_backend.fence_wait(v, 50000000ull) == 0);
    double waited = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(waited >= 50.0);
    CHECK(gfx_vulkan_backend.fence_done() < v && !__atomic_load_n(&g_gated_ran, __ATOMIC_ACQUIRE));
    CHECK(gfx_vulkan_backend.fence_wait(v + 1, 0) == 0);   // never handed out
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = gate, .value = 1};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
    t0 = plat_now_ns();
    CHECK(gfx_vulkan_backend.fence_wait(v, 5000000000ull) == 1);
    double then = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(__atomic_load_n(&g_gated_ran, __ATOMIC_ACQUIRE));
    uint64_t handle = (uint64_t)(uintptr_t)g_empty;
    op_hdr rec[2] = {{OP_BEGIN, 8}, {OP_END, 8}};
    memcpy(g_empty, rec, sizeof rec);
    for (int i = 0; i < 20; i++) gfx_vulkan_backend.submit(1, &handle, 1);
    gfx_vulkan_backend.finish(0);
    uint64_t counter = 0, handed = gpu_sync_handed();
    CHECK(vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter) == VK_SUCCESS);
    CHECK(gfx_vulkan_backend.fence_done() == handed && counter == handed && gpu_sync_issued() == handed);
    printf("timed out after %.1f ms; done %.2f ms after the gate opened; finish left values %llu, counter %llu\n",
           waited, then, (unsigned long long)handed, (unsigned long long)counter);
    vkDestroySemaphore(gpu_vk.device, gate, NULL);
    close_and_report();
    printf("ok wait\n");
}

// ---------- the backing map on the device (oot2/backing.h): guest memory of the runtime's own, pools placed by hand
#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define GiB (1ull << 30)
#define WIN GFX_GPU_WINDOW_BASE
static Cpu g_cpu;
static uint64_t g_heap;

static void guest_heap(uint64_t size) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = size;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1];
}

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

static void barrier(VkCommandBuffer cb, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = dst_stage,
                           .dstAccessMask = dst_access};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// One internal submission: fill [addr, addr + size) with word, or copy it into the readback buffer (g_buf), through the
// resolver; waited for. Returns the submission's value.
enum { FILL, READ };
static uint64_t through(int what, uint64_t addr, uint64_t size, uint32_t word, BackingRef *ref) {
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    backing_resolve(addr, size, v, ref);
    VkCommandBuffer cb = gpu_vk_cmd();
    barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBuffer buf = (VkBuffer)(uintptr_t)ref->buffer;
    if (what == FILL) {
        vkCmdFillBuffer(cb, buf, ref->offset, size, word);
    } else {
        VkBufferCopy c = {.srcOffset = ref->offset, .dstOffset = 0, .size = size};
        vkCmdCopyBuffer(cb, buf, g_buf, 1, &c);
    }
    barrier(cb, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    CHECK(gpu_vk_end(NULL, 0) == v);
    CHECK(gpu_sync_wait(v, 5000000000ull));
    return v;
}

static int words_are(const volatile void *p, uint64_t size, uint32_t word) {
    for (uint64_t i = 0; i < size / 4; i++)
        if (((const volatile uint32_t *)p)[i] != word) return 0;
    return 1;
}

static void touch_range(uint64_t a, uint64_t n) {
    for (uint64_t p = a; p < a + n; p += 0x1000) *(volatile uint8_t *)(uintptr_t)p = 0;
}

static void backing_stats_line(void) {
    BackingStats s;
    backing_stats(&s);
    printf("map: pools %llu, imports %llu (%llu KiB), extents %llu (%llu KiB), blocks %llu (spare %llu, ahead %llu, "
           "inline %llu, own %llu), imports made %llu refused %llu, deferred peak %llu\n", (unsigned long long)s.pools,
           (unsigned long long)s.imports, (unsigned long long)(s.import_bytes >> 10), (unsigned long long)s.extents,
           (unsigned long long)(s.extent_bytes >> 10), (unsigned long long)s.blocks,
           (unsigned long long)s.spare_blocks, (unsigned long long)s.blocks_made_ahead,
           (unsigned long long)s.blocks_made_inline, (unsigned long long)s.blocks_dedicated,
           (unsigned long long)s.imports_made, (unsigned long long)s.imports_refused,
           (unsigned long long)s.deferred_peak);
}

// Reads and writes through the resolver into each kind of backing, with the CPU's view checked after each fence.
static void case_backing(void) {
    guest_heap(512 * MiB);
    gpu_vk_open();
    backing_set_mirror_hook(NULL);   // imports: the mirrors have cases of their own
    make_buffer();
    const uint64_t boundary = (g_heap + 128 * MiB) & ~(BACKING_IMPORT_WINDOW - 1);
    pool_t p = make_pool(g_heap + 16 * MiB, 4 * MiB, 0);
    pool_t w = make_pool(WIN, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
    pool_t pw = make_pool(WIN + 16 * MiB, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_PHYSICAL);
    pool_t v = make_pool(WIN + 32 * MiB, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL);
    pool_t pc = make_pool(g_heap + 64 * MiB, 2 * MiB, NVN_MEMORY_POOL_FLAGS_PHYSICAL);
    pool_t v2 = make_pool(WIN + 48 * MiB, 2 * MiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL);
    pool_t x = make_pool(boundary - 1 * MiB, 2 * MiB, 0);   // across a 64 MiB import window
    map_run(&v, &pw, 64 * KiB, 1 * MiB);
    map_run(&v2, &pc, 128 * KiB, 1 * MiB);
    touch_range(g_heap + 16 * MiB, 1 * MiB);
    touch_range(g_heap + 64 * MiB, 2 * MiB);
    touch_range(boundary - 1 * MiB, 2 * MiB);

    // The premise of the commit query (the raw-address study, the write-watch study), on this device: an import over a page
    // that is not committed is refused, and a 4 KiB import of a committed page succeeds after it.
    BackingMem m;
    CHECK(!gpu_vk_backing_ops.import(g_heap + 16 * MiB, 2 * MiB, &m));
    CHECK(gpu_vk_backing_ops.import(g_heap + 16 * MiB, 0x1000, &m));
    gpu_vk_backing_ops.free(&m);
    printf("an import over uncommitted pages is refused; a 4 KiB import of a committed page then succeeds\n");

    BackingRef r;
    // Imported: the GPU reads what the CPU wrote, and the CPU reads what the GPU wrote, committed pages or not.
    for (uint64_t i = 0; i < 4096; i++) ((volatile uint32_t *)(uintptr_t)(g_heap + 16 * MiB))[i] = 0x5EED0000u + (uint32_t)i;
    through(READ, g_heap + 16 * MiB, 16 * KiB, 0, &r);
    CHECK(r.kind == BACKING_IMPORT);
    for (uint32_t i = 0; i < 4096; i++) CHECK(g_words[i] == 0x5EED0000u + i);
    through(FILL, g_heap + 16 * MiB + 128 * KiB, 16 * KiB, 0xABABABABu, &r);
    CHECK(words_are((const void *)(uintptr_t)(g_heap + 16 * MiB + 128 * KiB), 16 * KiB, 0xABABABABu));
    through(FILL, g_heap + 18 * MiB, 16 * KiB, 0xCDCDCDCDu, &r);   // never touched: committed for the bind
    CHECK(r.kind == BACKING_IMPORT && words_are((const void *)(uintptr_t)(g_heap + 18 * MiB), 16 * KiB, 0xCDCDCDCDu));
    printf("imported: the GPU read the CPU's words and the CPU read the GPU's, on touched and untouched pages\n");

    // A VRAM extent: written, then read back through a copy.
    through(FILL, WIN + 0x10000, 16 * KiB, 0x11111111u, &r);
    CHECK(r.kind == BACKING_EXTENT);
    through(READ, WIN + 0x10000, 16 * KiB, 0, &r);
    CHECK(words_are(g_words, 16 * KiB, 0x11111111u));
    printf("VRAM extent: written and read back\n");

    // Virtual over a physical pool in the window: written through the virtual pool, read through the physical one.
    through(FILL, WIN + 32 * MiB + 0x20000, 16 * KiB, 0x22222222u, &r);
    through(READ, WIN + 16 * MiB + 64 * KiB + 0x20000, 16 * KiB, 0, &r);
    CHECK(r.kind == BACKING_EXTENT && words_are(g_words, 16 * KiB, 0x22222222u));
    // Virtual over a CPU-visible physical pool (the title's shape): the CPU sees the GPU's words in guest memory.
    through(FILL, WIN + 48 * MiB + 0x30000, 16 * KiB, 0x33333333u, &r);
    CHECK(r.kind == BACKING_IMPORT);
    CHECK(words_are((const void *)(uintptr_t)(g_heap + 64 * MiB + 128 * KiB + 0x30000), 16 * KiB, 0x33333333u));
    printf("virtual over a physical run: through the window's extent and through an import of guest memory\n");

    // An import across two reservation pieces: one import over the bind, which crosses the boundary.
    through(FILL, boundary - 8 * KiB, 16 * KiB, 0x44444444u, &r);
    CHECK(r.kind == BACKING_IMPORT && words_are((const void *)(uintptr_t)(boundary - 8 * KiB), 16 * KiB, 0x44444444u));
    printf("an import across a 64 MiB window boundary (two reservation pieces): written and seen\n");
    backing_stats_line();

    // The guest gives imported pages back (a heap that shrinks): the decommit succeeds under the imports, and the next
    // sync retires every import over those pages.
    BackingStats s;
    backing_stats(&s);
    uint64_t imports = s.imports;
    g_cpu.x[0] = 0, g_cpu.x[1] = 8 * MiB;
    rt_svc(&g_cpu, 0x01, 0);
    CHECK(g_cpu.x[0] == 0);
    backing_sync();
    backing_stats(&s);
    CHECK(imports == 4 && s.imports == 0 && s.imports_stale == imports);   // two runs of p, pc, x
    printf("the heap shrank under %llu imports: decommitted, and the imports retired at the next sync\n",
           (unsigned long long)imports);

    pool_t *all[] = {&x, &v2, &pc, &v, &pw, &w, &p};
    for (int i = 0; i < 7; i++) gfx_vulkan_backend.pool_fini(all[i]);
    free_buffer();
    close_and_report();
    printf("ok backing\n");
}

// pool_fini of a pool that a submission in flight used returns only once that submission is done.
static pool_t g_fini;
static int g_fini_back;
static void fini_thread(void *arg) {
    (void)arg;
    gfx_vulkan_backend.pool_fini(&g_fini);
    __atomic_store_n(&g_fini_back, 1, __ATOMIC_RELEASE);
}

static void case_backing_fini(void) {
    guest_heap(64 * MiB);
    gpu_vk_open();
    backing_set_mirror_hook(NULL);
    g_fini = make_pool(g_heap + 4 * MiB, 4 * MiB, 0);   // an import: its memory is freed when the pool goes
    touch_range(g_heap + 4 * MiB, 4 * MiB);
    VkSemaphore gate = make_gate();
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    BackingRef r;
    backing_resolve(g_heap + 4 * MiB, 1 * MiB, v, &r);
    vkCmdFillBuffer(gpu_vk_cmd(), (VkBuffer)(uintptr_t)r.buffer, r.offset, 1 * MiB, 0x77777777u);
    CHECK(gpu_vk_end(&wait, 1) == v);
    CHECK(plat_thread_start(fini_thread, NULL, 256 * KiB));
    plat_sleep_ns(200 * 1000 * 1000);
    int early = __atomic_load_n(&g_fini_back, __ATOMIC_ACQUIRE);
    uint64_t done_before = gpu_sync_done();
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = gate, .value = 1};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
    uint64_t t0 = plat_now_ns();
    while (!__atomic_load_n(&g_fini_back, __ATOMIC_ACQUIRE)) {
        CHECK(plat_now_ns() - t0 < 5000000000ull);
        plat_sleep_ns(100 * 1000);
    }
    double after = (double)(plat_now_ns() - t0) / 1e6;
    CHECK(!early && done_before < v && gpu_sync_done() >= v);
    CHECK(words_are((const void *)(uintptr_t)(g_heap + 4 * MiB), 1 * MiB, 0x77777777u));
    BackingStats s;
    backing_stats(&s);
    CHECK(s.fini_waits == 1 && s.imports == 0 && s.deferred == 0);
    printf("pool_fini held while the submission was in flight; returned %.2f ms after the gate opened\n", after);
    vkDestroySemaphore(gpu_vk.device, gate, NULL);
    close_and_report();
    printf("ok backing_fini\n");
}

// Device-local heap usage (VK_EXT_memory_budget), summed over the device-local heaps.
static uint64_t vram_usage(void) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
    vkGetPhysicalDeviceMemoryProperties2(gpu_vk.phys, &mp);
    uint64_t used = 0;
    for (uint32_t i = 0; i < mp.memoryProperties.memoryHeapCount; i++)
        if (mp.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) used += b.heapUsage[i];
    return used;
}

// 4,660 pools (half CPU-visible, imported; half CPU_NO_ACCESS, VRAM extents) made, bound and finalized, three times:
// device-local memory and the process's private memory end each round where the first round ended.
static void case_backing_flat(void) {
    enum { POOLS = 4660, HALF = POOLS / 2, ROUNDS = 3 };
    static pool_t p[POOLS];
    guest_heap(512 * MiB);
    gpu_vk_open();
    backing_set_mirror_hook(NULL);   // imports and extents, as the backing map measured them
    touch_range(g_heap, HALF * 64 * KiB);
    uint64_t vram0 = 0, priv0 = 0;
    for (int round = 0; round < ROUNDS; round++) {
        uint64_t t0 = plat_now_ns();
        for (int i = 0; i < HALF; i++) p[i] = make_pool(g_heap + (uint64_t)i * 64 * KiB, 64 * KiB, 0);
        for (int i = 0; i < HALF; i++)
            p[HALF + i] = make_pool(WIN + (uint64_t)i * 64 * KiB, 64 * KiB, NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS);
        BackingRef r;
        for (int i = 0; i < POOLS; i++) backing_resolve((uintptr_t)p[i].mem + 0x40, 0x100, 0, &r);
        uint64_t t1 = plat_now_ns(), vram_in = vram_usage();
        for (int i = 0; i < POOLS; i++) gfx_vulkan_backend.pool_fini(&p[i]);
        uint64_t t2 = plat_now_ns();
        plat_sleep_ns(500 * 1000 * 1000);   // the block thread settles
        uint64_t vram = vram_usage();
        PlatProcMem pm;
        plat_process_memory(&pm);
        BackingStats s;
        backing_stats(&s);
        printf("round %d: made and bound in %.0f ms, finalized in %.0f ms; device-local use %llu MiB with the pools, "
               "%llu MiB after; private %llu MiB; pools %llu, imports %llu, extents %llu, blocks %llu\n", round,
               (double)(t1 - t0) / 1e6, (double)(t2 - t1) / 1e6, (unsigned long long)(vram_in >> 20),
               (unsigned long long)(vram >> 20), (unsigned long long)(pm.private_bytes >> 20),
               (unsigned long long)s.pools, (unsigned long long)s.imports, (unsigned long long)s.extents,
               (unsigned long long)s.blocks);
        CHECK(s.pools == 0 && s.imports == 0 && s.extents == 0 && s.deferred == 0);
        if (!round) {
            vram0 = vram, priv0 = pm.private_bytes;
        } else {
            CHECK(vram <= vram0 + 16 * MiB && pm.private_bytes <= priv0 + 32 * MiB);
        }
    }
    close_and_report();
    printf("ok backing_flat\n");
}

// ---------- mirrors (oot2/backing.h, Mirrors): this device is discrete, so a CPU-visible bind gets one
static void mirror_stats_line(void) {
    BackingStats s;
    backing_stats(&s);
    printf("mirrors: live %llu (%llu KiB), made %llu, refused %llu; syncs %llu, uploads %llu (%llu KiB); write-backs "
           "%llu (%llu bytes), pending %llu, early reads %llu\n", (unsigned long long)s.mirrors,
           (unsigned long long)(s.mirror_bytes >> 10), (unsigned long long)s.mirrors_made,
           (unsigned long long)(s.mirrors_refused_budget + s.mirrors_refused_untracked),
           (unsigned long long)s.mirror_syncs, (unsigned long long)s.uploads, (unsigned long long)(s.upload_bytes >> 10),
           (unsigned long long)s.writebacks, (unsigned long long)s.writeback_bytes, (unsigned long long)s.pending,
           (unsigned long long)s.reads_in_flight);
}

static void fill_words(uint64_t a, uint64_t n, uint32_t word) {
    for (uint64_t i = 0; i < n / 4; i++) ((volatile uint32_t *)(uintptr_t)a)[i] = word;
}

// A CPU write into a mirrored range is read by the GPU in the next submission, and the next one in the one after.
static void case_mirror_upload(void) {
    guest_heap(64 * MiB);
    gpu_vk_open();
    make_buffer();
    pool_t p = make_pool(g_heap + 4 * MiB, 1 * MiB, 0);
    fill_words(g_heap + 4 * MiB, 1 * MiB, 0x5EED5EEDu);
    BackingRef r;
    through(READ, g_heap + 4 * MiB, 16 * KiB, 0, &r);
    CHECK(r.kind == BACKING_MIRROR && words_are(g_words, 16 * KiB, 0x5EED5EEDu));
    printf("a first use: the GPU read the CPU's words through the mirror\n");
    for (uint32_t k = 1; k <= 3; k++) {
        fill_words(g_heap + 4 * MiB + 4 * KiB, 4 * KiB, 0xC0DE0000u + k);   // one page, before the next submission
        backing_sync();   // as the replay's submit does first
        through(READ, g_heap + 4 * MiB, 16 * KiB, 0, &r);
        CHECK(r.kind == BACKING_MIRROR && words_are(g_words + 1024, 4 * KiB, 0xC0DE0000u + k));
        CHECK(words_are(g_words, 4 * KiB, 0x5EED5EEDu) && words_are(g_words + 2048, 8 * KiB, 0x5EED5EEDu));
    }
    BackingStats s;
    backing_stats(&s);
    CHECK(s.mirrors == 1 && s.mirror_syncs == 4 && s.upload_bytes == 1 * MiB + 3 * 4 * KiB);
    printf("three CPU writes, each read by the next submission; only the written page was copied each time\n");
    mirror_stats_line();
    gfx_vulkan_backend.pool_fini(&p);
    free_buffer();
    close_and_report();
    printf("ok mirror_upload\n");
}

// A GPU write into a mirrored range reaches the CPU once its submission is done and cpu_synced is called, never before.
static void case_mirror_writeback(void) {
    guest_heap(64 * MiB);
    gpu_vk_open();
    pool_t p = make_pool(g_heap + 4 * MiB, 1 * MiB, 0);
    fill_words(g_heap + 4 * MiB, 1 * MiB, 0xAAAAAAAAu);
    VkSemaphore gate = make_gate();
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    BackingRef r;
    backing_resolve(g_heap + 4 * MiB + 64 * KiB, 64 * KiB, v, &r);
    CHECK(r.kind == BACKING_MIRROR);
    vkCmdFillBuffer(gpu_vk_cmd(), (VkBuffer)(uintptr_t)r.buffer, r.offset, 64 * KiB, 0x600DF00Du);
    backing_gpu_wrote(g_heap + 4 * MiB + 64 * KiB, 64 * KiB, v);
    backing_submit_end(v);
    CHECK(gpu_vk_end(&wait, 1) == v);
    plat_sleep_ns(100 * 1000 * 1000);
    const void *at = (const void *)(uintptr_t)(g_heap + 4 * MiB + 64 * KiB);
    CHECK(gpu_sync_done() < v && words_are(at, 64 * KiB, 0xAAAAAAAAu));   // never before the fence
    gfx_vulkan_backend.guest_read(g_heap + 4 * MiB + 64 * KiB, 4);
    gfx_vulkan_backend.cpu_synced(gpu_sync_done());
    printf("held by a semaphore: guest memory kept the old bytes, a read before the fence was counted\n");
    VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = gate, .value = 1};
    CHECK(vkSignalSemaphore(gpu_vk.device, &si) == VK_SUCCESS);
    CHECK(gfx_vulkan_backend.fence_wait(v, 5000000000ull) == 1);
    gfx_vulkan_backend.cpu_synced(v);
    CHECK(words_are(at, 64 * KiB, 0x600DF00Du));
    CHECK(words_are((const void *)(uintptr_t)(g_heap + 4 * MiB), 64 * KiB, 0xAAAAAAAAu));
    CHECK(words_are((const void *)(uintptr_t)(g_heap + 4 * MiB + 128 * KiB), 64 * KiB, 0xAAAAAAAAu));
    BackingStats s;
    backing_stats(&s);
    CHECK(s.writebacks == 1 && s.writeback_bytes == 64 * KiB && s.pending == 0 && s.reads_in_flight == 1);
    printf("the GPU's write reached guest memory after its fence and cpu_synced, and nothing beside it\n");
    mirror_stats_line();
    vkDestroySemaphore(gpu_vk.device, gate, NULL);
    gfx_vulkan_backend.pool_fini(&p);
    close_and_report();
    printf("ok mirror_writeback\n");
}

// VK_EXT_memory_budget's figures feed the admission: VRAM taken outside the map moves them, and a mirror size between
// the two thresholds is admitted before and refused after.
static void case_mirror_budget(void) {
    guest_heap(64 * MiB);
    gpu_vk_open();
    uint64_t u0, b0, u1, b1, u2, b2;
    CHECK(gpu_vk_backing_ops.vram(&u0, &b0) && b0 > 1 * GiB && u0 < b0);
    BackingMem m;
    CHECK(gpu_vk_backing_ops.block_new(512 * MiB, &m));
    CHECK(gpu_vk_backing_ops.vram(&u1, &b1));
    gpu_vk_backing_ops.free(&m);
    CHECK(gpu_vk_backing_ops.vram(&u2, &b2));
    printf("device-local heap: use %llu MiB of a budget of %llu MiB; with a 512 MiB block %llu of %llu; freed %llu of "
           "%llu\n", (unsigned long long)(u0 >> 20), (unsigned long long)(b0 >> 20), (unsigned long long)(u1 >> 20),
           (unsigned long long)(b1 >> 20), (unsigned long long)(u2 >> 20), (unsigned long long)(b2 >> 20));
    CHECK(u1 >= u0 + 256 * MiB);
    uint64_t t0 = b0 - u0 - b0 / BACKING_MIRROR_RESERVE, t1 = b1 > u1 + b1 / BACKING_MIRROR_RESERVE ?
                                                               b1 - u1 - b1 / BACKING_MIRROR_RESERVE : 0;
    CHECK(t1 + 256 * MiB <= t0);
    uint64_t size = (t0 + t1) / 2;
    CHECK(backing_mirror_fits(u0, b0, size) && !backing_mirror_fits(u1, b1, size));
    printf("the largest mirror admitted: %llu MiB, then %llu MiB with the block; one of %llu MiB admitted before and "
           "refused after\n", (unsigned long long)(t0 >> 20), (unsigned long long)(t1 >> 20),
           (unsigned long long)(size >> 20));
    // And through the map, with the figures as they are now: a small mirror is admitted.
    pool_t p = make_pool(g_heap + 4 * MiB, 1 * MiB, 0);
    fill_words(g_heap + 4 * MiB, 1 * MiB, 0);
    BackingRef r;
    backing_resolve(g_heap + 4 * MiB, 0x100, 0, &r);
    CHECK(r.kind == BACKING_MIRROR);
    gfx_vulkan_backend.pool_fini(&p);
    close_and_report();
    printf("ok mirror_budget\n");
}

// ---------- the device lost: Volk's entry points replaced
static int g_fault_calls, g_checkpoint_calls;

static VkResult VKAPI_CALL lost_submit(VkQueue q, uint32_t n, const VkSubmitInfo2 *s, VkFence f) {
    (void)q, (void)n, (void)s, (void)f;
    return VK_ERROR_DEVICE_LOST;
}

static VkResult VKAPI_CALL lost_wait(VkDevice d, const VkSemaphoreWaitInfo *w, uint64_t t) {
    (void)d, (void)w, (void)t;
    return VK_ERROR_DEVICE_LOST;
}

#define FAKE_BINARY 200
static VkResult VKAPI_CALL fake_fault(VkDevice d, VkDeviceFaultCountsEXT *counts, VkDeviceFaultInfoEXT *info) {
    (void)d;
    g_fault_calls++;
    if (!info) {
        counts->addressInfoCount = 2, counts->vendorInfoCount = 1, counts->vendorBinarySize = FAKE_BINARY;
        return VK_SUCCESS;
    }
    snprintf(info->description, sizeof info->description, "injected by the test: no real fault");
    if (counts->addressInfoCount >= 2) {
        info->pAddressInfos[0] = (VkDeviceFaultAddressInfoEXT){VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT,
                                                                0x7f0012345678ull, 0x1000};
        info->pAddressInfos[1] = (VkDeviceFaultAddressInfoEXT){
            VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT, 0x1234, 0};
    }
    if (counts->vendorInfoCount >= 1) {
        snprintf(info->pVendorInfos[0].description, sizeof info->pVendorInfos[0].description, "injected vendor record");
        info->pVendorInfos[0].vendorFaultCode = 0xdead, info->pVendorInfos[0].vendorFaultData = 0xbeef;
    }
    if (info->pVendorBinaryData && counts->vendorBinarySize) {
        uint8_t *b = info->pVendorBinaryData;
        for (size_t i = 0; i < counts->vendorBinarySize; i++) b[i] = (uint8_t)(i ^ 0x5a);
        VkDeviceFaultVendorBinaryHeaderVersionOneEXT h = {
            .headerSize = sizeof h, .headerVersion = VK_DEVICE_FAULT_VENDOR_BINARY_HEADER_VERSION_ONE_EXT,
            .vendorID = 0x10de, .deviceID = 0x2489, .driverVersion = 0x9a438000, .apiVersion = VK_API_VERSION_1_3};
        if (counts->vendorBinarySize >= sizeof h) memcpy(b, &h, sizeof h);
    }
    return VK_SUCCESS;
}

static void VKAPI_CALL fake_checkpoints(VkQueue q, uint32_t *n, VkCheckpointData2NV *data) {
    (void)q;
    g_checkpoint_calls++;
    if (!data) {
        *n = 2;
        return;
    }
    if (*n >= 2) {
        data[0].stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        data[0].pCheckpointMarker = (void *)(uintptr_t)GPU_VK_MARKER(41, GPU_VK_REPLAY, 0);
        data[1].stage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        data[1].pCheckpointMarker = (void *)(uintptr_t)GPU_VK_MARKER(40, GPU_VK_INTERNAL, 1);
    }
}

static void fake_queries(void) {
    vkGetDeviceFaultInfoEXT = fake_fault;
    vkGetQueueCheckpointData2NV = fake_checkpoints;
}

static void case_loss_submit(const char *dir) {
    gpu_set_report_folder(dir);
    gpu_vk_open();
    uint64_t handle = (uint64_t)(uintptr_t)g_empty;
    op_hdr rec[2] = {{OP_BEGIN, 8}, {OP_END, 8}};
    memcpy(g_empty, rec, sizeof rec);
    for (int i = 0; i < 3; i++) gfx_vulkan_backend.submit(1, &handle, 1);
    gfx_vulkan_backend.finish(0);
    fake_queries();
    vkQueueSubmit2 = lost_submit;
    gfx_vulkan_backend.submit(1, &handle, 1);   // the recorder thread submits the frame, and stops there
    vkrec_idle();
    printf("submit returned after the loss\n");
}

static void case_loss_completion(const char *dir) {
    gpu_set_report_folder(dir);
    gpu_vk_open();
    fake_queries();
    vkWaitSemaphores = lost_wait;
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    gpu_vk_cmd();
    gpu_vk_end(NULL, 0);
    gfx_vulkan_backend.fence_wait(v, 10000000000ull);
    printf("fence_wait returned after the loss\n");
}

static void case_refuse(void) {
    gpu_vk_open();
    printf("gpu_vk_open returned\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_test <case> [folder]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "devices")) case_devices();
    else if (!strcmp(cs, "open")) case_open();
    else if (!strcmp(cs, "submits")) case_submits();
    else if (!strcmp(cs, "wait")) case_wait();
    else if (!strcmp(cs, "loss_submit") && argc >= 3) case_loss_submit(argv[2]);
    else if (!strcmp(cs, "loss_completion") && argc >= 3) case_loss_completion(argv[2]);
    else if (!strcmp(cs, "refuse")) case_refuse();
    else if (!strcmp(cs, "backing")) case_backing();
    else if (!strcmp(cs, "backing_fini")) case_backing_fini();
    else if (!strcmp(cs, "backing_flat")) case_backing_flat();
    else if (!strcmp(cs, "mirror_upload")) case_mirror_upload();
    else if (!strcmp(cs, "mirror_writeback")) case_mirror_writeback();
    else if (!strcmp(cs, "mirror_budget")) case_mirror_budget();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
