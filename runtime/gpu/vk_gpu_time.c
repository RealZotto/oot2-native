// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// GPU time of the replayed frames (oot2/vk_gpu_time.h): a timestamp query pool per frame in flight, a timestamp after
// each marked command, and one row per frame and per mark written by the completion thread once the device reached the
// frame.
//
// A frame's queries, in order of index: 0 at the top of its prologue (the first thing the GPU does for the frame, after
// the barrier that opens it), 1 at the start of the main command buffer (after the prologue's copies), one for each mark
// in the main command buffer, then the end of the main command buffer and the end of the epilogue. The prologue and the
// epilogue are recorded by the replaying thread, so their timestamps are plain Vulkan calls; the main command buffer is
// the recorder's, so its timestamps are packets (oot2/vk_rec.h, WriteTimestamp2) in the order the replay wrote them.
// Every timestamp is taken at the end of all commands before it, so the interval between two marks is the whole time the
// GPU spent between them, with no overlap between a command and the one before it: a tool that measures, not a mode a
// player runs.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu_vk.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
#include "oot2/vk_gpu_time.h"
#include "oot2/vk_raw.h"
#include "oot2/vk_rec.h"
#include "plat.h"

#define QUERIES 1024u                  // a pool: the frame's first and last four, and a mark each between
#define MAX_MARKS (QUERIES - 4u)
#define LABEL_BYTES 56
#define NOTES_BYTES 240
#define STATS_EVERY 60u   // a frame in this many carries the counters of the backing map

typedef unsigned long long ull;

typedef struct {
    uint32_t x, y, z;
    uint8_t what;
    char label[LABEL_BYTES];
    char notes[NOTES_BYTES];
} Mark;

typedef struct Frame {
    VkQueryPool pool;
    uint64_t value;
    uint32_t marks, dropped;
    uint64_t effects_from_ns;   // the completion thread, at the frame's first effect
    int has_stats;              // the backing map's and the raw path's counters when the frame ended (every STATS_EVERY frames)
    BackingStats bs;
    GpuVkRawStats rs;
    PagegenStats pg;
    Mark mark[MAX_MARKS];
    struct Frame *next;
} Frame;

int vkgt_enabled;

static struct {
    int marks;                      // a timestamp after every marked command
    FILE *file;
    double period_ns;               // the length of a tick
    uint64_t mask;                  // the bits of a tick the queue keeps
    PlatMutex lock;                 // the free list
    Frame *free;
    // the replaying thread's
    Frame *cur;
    uint64_t ended;                 // frames ended
    char notes[NOTES_BYTES];
    size_t note_len;
    int note_cut;
    // the completion thread's
    int have_first;
    uint64_t first_tick, prev_end_tick;
    uint64_t frames, marks_done, marks_dropped;
} G = {.lock = PLAT_MUTEX_INIT};

void vkgt_open(void) {
    const char *path = getenv("OOT2_VK_GPU_TIMING");
    if (!path || !*path) return;
    const char *m = getenv("OOT2_VK_GPU_TIMING_MARKS");
    G.marks = 1;
    if (m && *m && !strcmp(m, "off")) G.marks = 0;
    else if (m && *m && strcmp(m, "on"))
        rt_fatal("gfx: vulkan: OOT2_VK_GPU_TIMING_MARKS is \"%s\"; the values are on and off", m);
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu_vk.phys, &n, NULL);
    VkQueueFamilyProperties *q = calloc(n ? n : 1, sizeof *q);
    if (!q) rt_fatal("gfx: vulkan: out of memory for the queue families");
    vkGetPhysicalDeviceQueueFamilyProperties(gpu_vk.phys, &n, q);
    uint32_t bits = gpu_vk.family < n ? q[gpu_vk.family].timestampValidBits : 0;
    free(q);
    if (!bits)
        rt_fatal("gfx: vulkan: OOT2_VK_GPU_TIMING is set and queue family %u has no timestamps", gpu_vk.family);
    G.mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    G.period_ns = (double)gpu_vk.facts.props.limits.timestampPeriod;
    G.file = fopen(path, "wb");
    if (!G.file) rt_fatal("gfx: vulkan: OOT2_VK_GPU_TIMING: cannot write %s", path);
    fprintf(G.file, "#oot2-vk-gpu-timing\t1\tperiod_ns\t%.6f\tvalid_bits\t%u\tmarks\t%s\n", G.period_ns, bits,
            G.marks ? "on" : "off");
    fflush(G.file);
    G.have_first = 0, G.frames = G.marks_done = G.marks_dropped = 0;
    vkgt_enabled = 1;
    rt_log("gfx: vulkan: gpu timing on: %s, a timestamp %s, a tick is %.3f ns on %u valid bits", path,
           G.marks ? "after every dispatch, pass, transfer and barrier" : "at the frame's ends only", G.period_ns, bits);
}

static void free_frame(Frame *f) {
    if (f->pool) vkDestroyQueryPool(gpu_vk.device, f->pool, NULL);
    free(f);
}

void vkgt_close(void) {
    if (!vkgt_enabled) return;
    vkgt_enabled = 0;
    rt_log("gfx: vulkan: gpu timing: %llu frames written, %llu marks, %llu marks past the pool's room", (ull)G.frames,
           (ull)G.marks_done, (ull)G.marks_dropped);
    plat_mutex_lock(&G.lock);
    while (G.free) {
        Frame *f = G.free;
        G.free = f->next;
        free_frame(f);
    }
    plat_mutex_unlock(&G.lock);
    if (G.cur) free_frame(G.cur);
    G.cur = NULL;
    if (G.file) fclose(G.file);
    G.file = NULL;
}

// ---------- the replaying thread
static Frame *take(void) {
    plat_mutex_lock(&G.lock);
    Frame *f = G.free;
    if (f) G.free = f->next;
    plat_mutex_unlock(&G.lock);
    if (!f) {
        f = calloc(1, sizeof *f);
        if (!f) rt_fatal("gfx: vulkan: out of memory for the gpu timing of a frame");
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                    .queryCount = QUERIES};
        GPU_VK_OK(vkCreateQueryPool(gpu_vk.device, &qi, NULL, &f->pool), "vkCreateQueryPool (gpu timing)");
    }
    f->next = NULL, f->marks = f->dropped = 0, f->effects_from_ns = 0, f->has_stats = 0;
    return f;
}

static void effects_began(void *arg, uint64_t value) {
    (void)value;
    ((Frame *)arg)->effects_from_ns = plat_now_ns();
}

static void forget_notes(void) {
    G.note_len = 0, G.note_cut = 0, G.notes[0] = 0;
}

static void frame_done(void *arg, uint64_t value);

void vkgt_frame_begin(uint64_t value) {
    if (G.cur) rt_fatal("gfx: vulkan: gpu timing: a frame begins inside a frame");
    Frame *f = take();
    f->value = value;
    G.cur = f;
    forget_notes();
    VkCommandBuffer pro = gpu_vk_prologue();
    vkCmdResetQueryPool(pro, f->pool, 0, QUERIES);
    // the reset is made before the first timestamp is written into the pool it reset
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(pro, &di);
    vkCmdWriteTimestamp2(pro, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f->pool, 0);
    vkrec_effect(effects_began, f);   // the first effect of the frame: the effects' clock starts here
    vkrec_WriteTimestamp2(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f->pool, 1);
}

void vkgt_note(char kind, uint64_t addr, uint64_t bytes, int write) {
    if (!G.cur) return;
    size_t room = sizeof G.notes - G.note_len;
    char at[24] = "";
    if (addr) snprintf(at, sizeof at, "@%llx", (ull)addr);
    int n = snprintf(G.notes + G.note_len, room, "%s%c%llu%s%s", G.note_len ? "," : "", kind, (ull)((bytes + 1023) >> 10),
                     write ? "w" : "", at);
    if (n > 0 && (size_t)n < room) {
        G.note_len += (size_t)n;
    } else {
        G.notes[G.note_len] = 0;   // a note that does not fit is left out, and the line says so
        G.note_cut = 1;
    }
}

void vkgt_mark(int what, const char *label, uint32_t x, uint32_t y, uint32_t z) {
    Frame *f = G.cur;
    if (f && G.marks) {
        if (f->marks == MAX_MARKS) {
            f->dropped++;
        } else {
            Mark *m = &f->mark[f->marks];
            m->what = (uint8_t)what, m->x = x, m->y = y, m->z = z;
            snprintf(m->label, sizeof m->label, "%s", label ? label : "");
            snprintf(m->notes, sizeof m->notes, "%s%s", G.notes, G.note_cut ? "+" : "");
            vkrec_WriteTimestamp2(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f->pool, 2 + f->marks);
            f->marks++;
        }
    }
    forget_notes();
}

void vkgt_frame_end(void) {
    Frame *f = G.cur;
    if (!f) rt_fatal("gfx: vulkan: gpu timing: the end of a frame that did not begin");
    G.cur = NULL;
    // The counters are read here, between two frames, and written by the effect: the completion thread must not take the
    // backing map's lock, which the replaying thread may hold while it waits for a value that thread publishes.
    if (G.ended++ % STATS_EVERY == 0) {
        backing_stats(&f->bs);
        gpu_vk_raw_stats(&f->rs);
        pagegen_stats(&f->pg);
        f->has_stats = 1;
    }
    uint32_t at = 2 + f->marks;
    vkrec_WriteTimestamp2(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f->pool, at);
    VkCommandBuffer epi = gpu_vk_epilogue();
    vkCmdWriteTimestamp2(epi, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f->pool, at + 1);
    vkrec_effect(frame_done, f);
}

// ---------- the completion thread
static const char *const k_what[] = {"dispatch", "pass", "xfer", "barrier"};

static uint64_t ticks_ns(uint64_t from, uint64_t to) {
    return (uint64_t)((double)((to - from) & G.mask) * G.period_ns + 0.5);
}

static void frame_done(void *arg, uint64_t value) {
    Frame *f = arg;
    uint32_t n = f->marks + 4;
    uint64_t t[QUERIES];
    GPU_VK_OK(vkGetQueryPoolResults(gpu_vk.device, f->pool, 0, n, sizeof t[0] * n, t, sizeof t[0],
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
              "vkGetQueryPoolResults (gpu timing)");
    for (uint32_t i = 0; i < n; i++) t[i] &= G.mask;
    if (!G.have_first) G.have_first = 1, G.first_tick = t[0], G.prev_end_tick = t[0];
    uint64_t eff = f->effects_from_ns ? plat_now_ns() - f->effects_from_ns : 0;
    fprintf(G.file, "F\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%u\t%u\t%llu\t%llu\n", (ull)value,
            (ull)ticks_ns(G.first_tick, t[0]), (ull)ticks_ns(t[0], t[n - 1]), (ull)ticks_ns(t[0], t[1]),
            (ull)ticks_ns(t[1], t[n - 2]), (ull)ticks_ns(t[n - 2], t[n - 1]), f->marks, f->dropped,
            (ull)ticks_ns(G.prev_end_tick, t[0]), (ull)eff);
    for (uint32_t i = 0; i < f->marks; i++) {
        const Mark *m = &f->mark[i];
        fprintf(G.file, "M\t%llu\t%u\t%s\t%s\t%u\t%u\t%u\t%llu\t%s\n", (ull)value, i, k_what[m->what], m->label, m->x, m->y,
                m->z, (ull)ticks_ns(t[1 + i], t[2 + i]), m->notes);
    }
    if (f->has_stats) {
        const BackingStats *b = &f->bs;
        const GpuVkRawStats *r = &f->rs;
        const PagegenStats *pg = &f->pg;
        fprintf(G.file, "B\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu"
                "\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\n", (ull)value,
                (ull)b->imports, (ull)b->import_bytes, (ull)b->mirrors, (ull)b->mirror_bytes, (ull)b->uploads,
                (ull)b->upload_bytes, (ull)b->writebacks, (ull)b->writeback_bytes, (ull)r->writebacks, (ull)r->writeback_bytes,
                (ull)b->raw_promoted, (ull)b->raw_promote_bytes, (ull)b->raw_begins, (ull)b->resolves, (ull)b->resolve_misses,
                (ull)r->ended_ns, (ull)r->mark_ns, (ull)r->writeback_cpu_pages, (ull)r->sink_bytes, (ull)r->sinks_made,
                (ull)pg->scan_ns, (ull)pg->scans);
    }
    fflush(G.file);
    G.prev_end_tick = t[n - 1];
    G.frames++, G.marks_done += f->marks, G.marks_dropped += f->dropped;
    plat_mutex_lock(&G.lock);
    f->next = G.free;
    G.free = f;
    plat_mutex_unlock(&G.lock);
}
