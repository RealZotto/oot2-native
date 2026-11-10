// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The memory line: one row a minute, and one at the title mark, at the scene entry and at the end, from the release build,
// carrying every term that makes up the process's private commit, so that a Task Manager reading can be reconciled and a
// RAM lever proven (runtime/nvn/nvn_memline.c; tools/check_memline.py reads the file).
//
// It runs only with the capture (OOT2_CAPTURE=1; oot2/capture.h) and writes memory-<UTC date>-<time>-<pid>.tsv in the same
// folder, beside the capture file; the capture's own fixed-width memory row is untouched. OOT2_MEMLINE_SECONDS=S changes the
// period (decimal seconds, default 60). OOT2_MEMLINE_SCENE=N marks the scene entry at present N (a present count, like
// OOT2_CAPTURE_TITLE); the front end can call memline_mark(MEMLINE_SCENE, seq) itself.
//
// Cost. Nothing is gathered on the render or the recorder thread: a helper thread wakes every tenth of a second, and
// memline_mark only sets a flag. The gather (rt_mem_measure walks the page map, tens of milliseconds; a VirtualQuery walk)
// takes the memory lock shared for the time of that accessor and no other lock the frame path takes beyond what each
// accessor takes. The end of the run waits at most a second for the helper; past that it writes a cheap row itself, with
// the terms that need a runtime lock set to -1.
//
// File format. Tab separated, numbers only except the first field of a line, one record per line.
//   oot2-memline 1                       the first line: the format and its version
//   columns  name...                     the names of the numeric fields of a "row" line, in order (MEMLINE_TERMS)
//   sum      name...                     the columns that are terms of the sum below (MEMLINE_SUMMED)
//   row  kind  v...                      kind is minute, title, scene or end; then one value per column. -1 is a value that
//                                        could not be read. Times are nanoseconds of the monotonic clock since the start.
//   alloc seq site live_blocks live_bytes total_blocks
//                                        after a row, one line for each noted allocation site that ever held a block of
//                                        1 MiB or more (memline_note); site is a label of our own code
//   end  rows                            the closing line
// The sum. private is the process's commit charge (PrivateUsage). The named terms are: the committed bytes of the five guest
// regions; the private committed bytes the host holds outside those regions, as VirtualQuery sees them, in three groups
// (write-combined allocations of exactly 32 MiB, other write-combined allocations, and the rest); and the three terms that
// are VRAM and so cannot be seen by VirtualQuery (the backing map's blocks, the texture arena's blocks, and the bytes of
// images that have memory of their own). memline_sum returns their total and the residual private - total; the check
// passes within 3 percent. Terms that are only a breakdown of another (the command buffers' host bytes, the pipeline cache,
// the mirror and import bytes, the Vulkan heap usage) are columns but not in the sum. Whether the texture arena's blocks are
// device-local (so not in the VirtualQuery groups) or host-visible (so in them already, and double counted) is decided by
// the first desktop run: its residual says.
// The astc_dec_*, bc_equiv_bytes, native_bc_bytes and other_bytes columns are a breakdown of the images in the arena and the
// images with memory of their own (not in the sum): astc_dec_wN are the bytes of ASTC textures held decoded as RGBA8, by the
// width of their block (4, 5, 6, 8, 10, 12), astc_dec_srgb_bytes and astc_dec_arena_bytes the parts of them that are sRGB and
// that lie in the arena, astc_dec_images their count, bc_equiv_bytes what the same images would hold as BC7 or BC3 (1 byte a
// texel, mips and layers included, edges rounded up to 4), and native_bc_bytes and other_bytes the images made in a BC format
// and in any other. The saving of BC is the sum of the astc_dec_wN less bc_equiv_bytes (tools/check_memline.py prints it).
#pragma once
#include <stdint.h>

// One X per numeric column, in file order: process, guest regions, host private groups, Vulkan heaps, texture memory,
// the backing map, images, pipelines, command buffers, NVN objects, noted blocks, the machine side.
#define MEMLINE_TERMS(X)                                                                                              \
    X(seq) X(t_ns) X(private) X(working_set) X(peak_working_set)                                                      \
    X(stack_committed) X(stack_resident) X(heap_committed) X(heap_resident) X(alias_committed) X(alias_resident)     \
    X(modules_committed) X(modules_resident) X(kernel_committed) X(kernel_resident)                                  \
    X(wc32_bytes) X(wc32_count) X(wcodd_bytes) X(wcodd_count) X(host_other_bytes)                                    \
    X(vk_heap0_used) X(vk_heap0_budget) X(vk_heap1_used) X(vk_heap1_budget)                                          \
    X(tex_ring_chunks) X(tex_ring_bytes) X(tex_arena_blocks) X(tex_arena_bytes) X(tex_arena_used) X(tex_arena_frees) \
    X(bk_blocks) X(bk_block_bytes) X(bk_extents) X(bk_extent_bytes) X(bk_imports) X(bk_import_bytes)                \
    X(bk_mirrors) X(bk_mirror_bytes) X(bk_deferred)                                                                  \
    X(img_live) X(img_own) X(img_own_bytes) X(img_arena)                                                                 X(astc_dec_w4) X(astc_dec_w5) X(astc_dec_w6) X(astc_dec_w8) X(astc_dec_w10) X(astc_dec_w12)                          X(astc_dec_srgb_bytes) X(astc_dec_arena_bytes) X(astc_dec_images) X(bc_equiv_bytes) X(native_bc_bytes)               X(other_bytes)                                                                                                   \
    X(pipe_resident) X(pipe_cache_bytes) X(pipe_libraries) X(pipe_links)                                             \
    X(cmd_host_bytes) X(cmd_copies)                                                                                  \
    X(nvn_buffers) X(nvn_textures) X(nvn_pools) X(nvn_pool_bytes)                                                    \
    X(big_blocks) X(big_bytes)                                                                                       \
    X(ram_available) X(ram_standby) X(ram_modified) X(sys_committed) X(sys_commit_limit) X(ws_cap_max)              \
    X(top1_private) X(top1_ws) X(top2_private) X(top2_ws) X(top3_private) X(top3_ws) X(top4_private) X(top4_ws)     \
    X(top5_private) X(top5_ws)

// The columns that are terms of the sum.
#define MEMLINE_SUMMED(X)                                                                                             \
    X(stack_committed) X(heap_committed) X(alias_committed) X(modules_committed) X(kernel_committed)                 \
    X(wc32_bytes) X(wcodd_bytes) X(host_other_bytes) X(bk_block_bytes) X(tex_arena_bytes) X(img_own_bytes)

typedef struct MemTerms {
#define X(n) uint64_t n;
    MEMLINE_TERMS(X)
#undef X
} MemTerms;
#define MEMLINE_COLUMNS ((int)(sizeof(MemTerms) / sizeof(uint64_t)))

enum { MEMLINE_MINUTE, MEMLINE_TITLE, MEMLINE_SCENE, MEMLINE_END, MEMLINE_KINDS };

// The total of the named terms (a value of -1, UINT64_MAX, counts as 0) and, in *residual when not NULL, private less the
// total (negative when the terms exceed the commit).
uint64_t memline_sum(const MemTerms *t, uint64_t priv, int64_t *residual);

// Fill everything this process can tell now. cheap: only what needs no runtime lock (the end of a run that failed).
// Called by the helper thread; a test calls it directly.
void memline_gather(MemTerms *out, int cheap);

// The Vulkan back end (or a test) registers the function that fills the device terms (heaps, tex_*, bk_*, img_*, pipe_*).
// It runs on the helper thread. NULL: those terms stay 0, as on the null back end.
typedef void (*MemlineGpuFill)(MemTerms *t);
void memline_set_gpu(MemlineGpuFill fn);
// Tests only: runs after memline_gather's own fill, to put synthetic counters in.
void memline_set_test_hook(MemlineGpuFill fn);

// A block of 1 MiB or more that our code allocated (+1 block, +bytes) or freed (-1, -bytes). site is a string literal naming
// the call site. A block under 1 MiB returns at once.
// (runtime/core/memnote.c, so that every build links it.)
void memline_note(const char *site, uint64_t bytes);
void memline_unnote(const char *site, uint64_t bytes);
enum { MEMLINE_SITES = 16 };
typedef struct MemlineSite {
    const char *site;
    uint64_t live_blocks, live_bytes, total_blocks;   // total_blocks: ever noted
} MemlineSite;
int memline_sites(MemlineSite *out, int max);   // the sites noted so far; the count

// The helper thread: started once by capture_start in folder with the stamp and pid of the capture, ended by memline_end.
void memline_start(const char *folder, const char *stamp, unsigned pid);
// title or scene: a flag for the helper to act on, never a gather. The scene mark with a present count seq is acted on once, at
// OOT2_MEMLINE_SCENE; with seq 0, now.
void memline_mark(int kind, uint64_t seq);
void memline_end(void);                      // the closing row; at most a second's wait; once
int memline_on(void);

// Live NVN objects (nvn_buffer.c, nvn_texture.c).
uint64_t nvn_buffers_live(void);
uint64_t nvn_textures_live(void);
