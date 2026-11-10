// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Spike S7's instruments and the owner side of lever L1 (architecture D19, D27; the page release study built the generic half in
// runtime/core/memory.c, which knows no textures). Off by default: the setting texheap (OOT2_TEXHEAP=1) switches them on,
// texheap_every (OOT2_TEXHEAP_EVERY, default 300) is how many presents lie between two samples. They answer how much of
// the texture heap a run touches, how it fills, and whether the game reads a texel page after its upload retired.
//
// What the front end tells this module (every call is a no-op while the setting is off):
//   pools            texheap_pool_init and _end: the pool's range, class and flags, and the page of the heap it lies on
//   textures         texheap_texture_init and _end: one record per storage (guest address, bytes), however many times the game
//                    initializes a texture there (a transient texture initialized again at the same storage every frame is
//                    one record and counts once); the pages wholly inside the storage are its own
//   other objects    texheap_range(TEXHEAP_OBJECT, ...) for a buffer or an event word: a page one touches is never selected
//   descriptors      texheap_desc: a texture pool entry, (table address, ID) to the record, so that a bind can name it
//   replay           texheap_submit, texheap_op for each record, texheap_complete: the null back end's walk. A record binds
//                    a texture when it names a registered ID; a GPU write is a render target, a bound image, the destination
//                    of a copy from a buffer or another texture, a texture clear
//   presents         texheap_present: once a frame, on the presenting thread (see "The frame")
//
// The selection rule. A page of a texture's storage is selected when: the texture is live and a completed submission has
// bound it since the CPU last wrote it (the null back end's stand-in for "upload retired"; a Vulkan back end's real point is
// the fence of the copy that read the page into the texture's own memory, or never for a page it imported, which stays
// busy); the page lies wholly inside the storage; the CPU wrote it (the write watch, rt_mem_written, which only Windows has:
// elsewhere nothing is ever selected); exactly one live texture touches it and no buffer or event does; and it is committed.
// With the setting texheap_trigger=register the point is instead the registration of the texture's descriptor, as for a
// back end that uploads when a texture is registered; it exists to measure what such a back end would release.
// Selected pages are armed (page_release=count) or released (page_release=release) through rt_pages_*, one range for the
// heap region and one for the alias region, made on the first selection. With page_release=off the module only counts the
// pages it would have selected. A write to a page after its selection puts the texture back to unbound: the CPU rewrote it.
//
// The frame. At each present, before anything else, the write watch over the span of the pools' pages is read through the
// page generations (oot2/pagegen.h, the runtime's one reader of it, so a consumer of the same pages sees them too), once
// per region, and for each pool the pages stamped since the generation up to which that pool was last read. Every such
// page is one CPU-written page of that frame: counted per pool, marked written in this module's page table, and, when a
// texture owns it, the texture's rewrite counted. A pool starts at the generation of a query made when it began, so writes
// before it are not its own. A write seen in the frame whose present follows the completion of a texture's bind is taken
// to come before the bind; a later one unbinds the texture (the frame's granularity: a write after the bind in the same
// frame is not told from one before it). Then the textures whose bind completed since the last present are selected, and
// every texheap_every presents a sample is written. The frame row's "outside" is the pages the query reported inside the
// span and outside every pool, counted from the page generations' statistics (approximate if another consumer queries
// at the same moment).
//
// The file. texheap-<UTC date>-<time>-<pid>.tsv in the game folder, numbers only, one record per line, tab separated, the
// kind first; times are milliseconds since the first call. A reader ignores kinds it does not know.
//   oot2-texheap 1
//   header  key  value
//   frame   seq pages tex_pages pool_pages outside pools_written   pages the CPU wrote in the frame that ended with present
//                            seq: all, inside texture storage, inside a pool, outside every pool; pools that had any
//   sample  seq t_ms  heap_mapped heap_committed heap_resident private working_set  pools pool_bytes
//           recs rec_bytes cpu_recs cpu_bytes gpu_recs gpu_bytes idle_recs idle_bytes tex_pages written_pages
//           eligible_pages selected_pages  arms releases recommits busy guest_reads host_reads guest_writes host_writes
//           stale_touches  b2t_copies b2t_bytes t2b_copies pkg_inits pkg_bytes read_texels
//           inval_calls inval_tex_calls inval_tex_bytes flush_calls flush_tex_calls flush_tex_bytes partial no_range
//                            the run so far. A record is a texture storage; cpu: a page of it was written by the CPU, or
//                            packaged data was copied in; gpu: no CPU write and a render target, image or copy destination;
//                            idle: neither. written_pages: pages of texture storage the CPU has written; eligible_pages:
//                            those the rule would select now; selected_pages: pages armed or released now.
//   pool    seq id class flags size mem  committed resident  recs rec_bytes cpu_bytes gpu_bytes idle_bytes
//           tex_pages written_pages selected_pages  window_frames window_pages window_max
//                            one per pool holding a live texture storage at the sample, with the writes of the window since
//                            the last sample: frames in which the CPU wrote the pool, pages written, most in a frame.
//                            committed and resident are bytes of a pool the CPU can see (0 for another).
//   wpool   seq id class size window_frames window_pages window_max
//                            a pool the CPU wrote in the window and that holds no texture storage
//   read    seq serial size pool pages guest fn   the first access to a selected page of a texture, kind read or write
//   write   (same fields)
//   end     code reason seq
// The first access of each kind to each texture is written when it happens (and logged), so a run that stops on a read of a
// released page has it in the file.
#pragma once
#include <stdint.h>

#include "nvn_obj.h"
#include "settings.h"

static inline int texheap_on(void) { return rt_setting(RT_SET_texheap) != 0; }

enum { TEXHEAP_OBJECT, TEXHEAP_OBJECT_END, TEXHEAP_INVALIDATE, TEXHEAP_FLUSH };

// The folder the file goes in (main.c, with ledger_set_folder).
void texheap_set_folder(const char *dir);

// The reader of the write watch: the page generations' (oot2/pagegen.h) scan, runs and scanned-pages count, which main.c
// installs at start-up so that runtime/nvn links without runtime/gpu. With none installed nothing is ever selected.
typedef void (*TexheapRun)(void *ctx, uint64_t lo, uint64_t hi);
typedef struct TexheapWatch {
    uint64_t (*scan)(uint64_t lo, uint64_t hi);
    uint64_t (*runs)(uint64_t lo, uint64_t hi, uint64_t since, TexheapRun fn, void *ctx);
    uint64_t (*scanned)(void);   // pages the scans have reported so far (may be NULL)
} TexheapWatch;
void texheap_set_watch(const TexheapWatch *w);

void texheap_pool_init(const pool_t *p);
void texheap_pool_end(const pool_t *p);
// A texture initialised at t->mem; pkg: packaged texel data was copied into its storage.
void texheap_texture_init(const tex_t *t, int pkg);
// The texture's serial ends (finalized or initialised again): its storage is no longer its own.
void texheap_texture_end(const tex_t *t);
// TEXHEAP_OBJECT: [addr, addr + size) belongs to a buffer or an event until the process ends or the range is given again
// (sizes of 0 are ignored); the others count a call that covered [addr, addr + size), and the bytes of it inside texture
// storage.
void texheap_range(int what, uint64_t addr, uint64_t size);
void texheap_read_texels(void);   // a nvnTextureReadTexels call

void texheap_desc(const dpool_ref *pool, int id, const tex_t *t);

// The null back end's replay: a submission begins, each of its records, and the submission is complete. Queues complete
// independently and a queue held at a wait replays its records after later submissions', so each record and each
// completion names its submission's fence value.
void texheap_submit(uint64_t fence);
// A record of the submission with fence value `fence`. tex_pool is the table address of the texture pool bound on the
// record's queue, as the record leaves it (a queue's pool is its own: a queue resumed after a wait binds in the pool it set,
// not the one another queue set meanwhile).
void texheap_op(uint64_t fence, const op_hdr *r, uint64_t tex_pool);
void texheap_complete(uint64_t fence);              // that submission only

void texheap_present(uint64_t seq);
