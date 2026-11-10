// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The bindless tables of the Vulkan back end (architecture D10, D13): the game's texture and sampler pools mapped to the
// global arrays of set 0 (oot2/vk_pipeline.h: binding 0 the sampled images, binding 1 the samplers). This file is the
// part that takes no Vulkan type, so that the mapping, its reuse rules and the sharing of samplers are checked in
// continuous integration without a device; runtime/gpu/vk_bindless.c gives it Vulkan objects through BindlessOps.
//
// The game registers one ID at a time (gfx.h texture_register, sampler_register), constantly (110 texture
// registrations and 46 texture initialisations per Forest frame, the NVN2 call census) while up to two frames are in flight, and
// a program may read its handles from memory, so an array index must stay the game's own ID. The design:
//
//   Shadow tables. Each texture pool and each sampler pool (named by dpool_ref.addr) keeps what the game registered at
//   every ID: a registration changes exactly that entry of that table, and nothing is written to a descriptor set.
//   Registering what the entry already holds (the same view, the same sampler state) changes nothing at all.
//
//   Copies of the arrays. The descriptor sets are a ring of copies, each indexed by the game's own IDs: the texture ID
//   is the index into binding 0 and the sampler ID the index into binding 1, so a handle a shader reads from memory
//   and a handle the replay puts in a push constant are both right. A registration appends one record to a change log;
//   a submission takes a copy (bindless_acquire), which brings it up to date by applying the records since it was last
//   used, one descriptor write per changed entry. A copy a pending submission may read is never written: a submission
//   that finds the current copy pending and the log ahead of it takes another copy that is not pending, the oldest
//   first, and a ring of N copies runs out only when N submissions are pending, which waits for the oldest.
//
//   Pools. The copy a submission takes shows the texture pool and the sampler pool that submission sets: the first
//   pool named is resident; a different one is brought in by writing the entries that differ (never a whole table when
//   the log reaches back far enough), between submissions. A copy shows one texture pool at a time.
//
//   Views. An entry holds the view its texture's own target asks for (a 2D texture a 2D view, a cube a cube view...)
//   at its ID. A program declares one variable per image type at binding 0 (the translator defects study) and a descriptor must
//   have the view type of the variable that reads it, so a read as another type (a 2D array read of a 2D texture) takes a
//   second slot, one of the last `second` slots of the array, which holds a view of that type over the same image
//   (bindless_image_slot). The slots are past every ID of the game's pool (the array is the pool's IDs and then the second
//   slots), so a second view never takes an ID the game can register. The slot is allocated at the first such read,
//   lives until the entry changes, and appears in every copy through the log like any other entry. A handle a shader
//   reads from memory has no second view.
//
//   Samplers. A host sampler is shared by its state (the samp_par bytes): the sampler pool's 4,096 IDs against the
//   device's 4,000 sampler allocations (D13). A state no entry and no copy holds any more is freed.
//
//   A view the host makes when the replay binds the texture (BindlessOps.view_demand) is written into the copy being replayed at
//   the game's own ID, which a copy a pending submission still reads may show as the placeholder: a submission that read that ID
//   through a handle held in memory, before anything bound it, can see the new descriptor (the second-view slots avoid this by
//   taking only slots no copy shows; the game's own ID cannot move). Handles read from memory are a stated limit of this table.
//
//   Retirement. A view or a sampler that no entry and no copy holds is freed once the newest submission that used a
//   copy holding it is done, so a released serial or an overwritten entry never clears what a pending submission reads
//   (gfx.h texture_release: the host objects of that serial may go, never the entry at its ID).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/nvn_obj.h"

// ---- view types: VkImageViewType's values
enum { BV_1D, BV_2D, BV_3D, BV_CUBE, BV_1D_ARRAY, BV_2D_ARRAY, BV_CUBE_ARRAY, BV_TYPES, BV_NONE = -1 };

// The view type a texture seen through view v (NULL: the whole texture) asks for, or BV_NONE for a buffer texture,
// which has no image view. The target of the view when it sets one, else of the texture; multisampled targets and
// rectangles take the single-sampled 2D type of Vulkan.
int bindless_view_type(const tex_t *t, const view_t *v);

// What makes two registrations share one host view: the storage, the texture's shape and format, the view's
// parameters and the type asked for. Equal keys compare equal as bytes.
typedef struct BindlessKey {
    uint64_t storage;
    uint32_t target, format, flags, samples, width, height, depth, levels, swizzle;
    uint32_t pkg, layout_flags, stride;   // where the texels lie in storage the CPU wrote: the packaged layout byte, SPARSE and MINIMAL_LAYOUT, the stride
    uint32_t v_set, v_format, v_target, v_base_level, v_levels, v_min_layer, v_layers;
    uint8_t v_swz[4];
    int32_t want;   // BV_* asked for
} BindlessKey;
void bindless_key(const tex_t *t, const view_t *v, int want, BindlessKey *out);

// ---- what the tables need of their host (runtime/gpu/vk_bindless.c on a device, a test's stand-ins without one)
enum { BW_IMAGE, BW_SAMPLER };
typedef struct BindlessWrite {
    int kind;          // BW_IMAGE or BW_SAMPLER
    uint32_t index;    // the slot in its array
    void *object;      // a view or a sampler made by this ops table; NULL: the placeholder
} BindlessWrite;

typedef struct BindlessOps {
    // A host view of texture t seen through v as a view of `type` (BV_*), or NULL when the host cannot make one (the
    // entry then shows the placeholder). Called once per distinct BindlessKey that is live.
    void *(*view_make)(const tex_t *t, const view_t *v, int type);
    void (*view_free)(void *view);                    // no submission can use it any more
    void *(*sampler_make)(const samp_par *s);
    void (*sampler_free)(void *sampler);
    void *(*set_make)(void);                          // one copy of both arrays, every entry the placeholder
    void (*set_free)(void *set);
    void (*write)(void *set, const BindlessWrite *w, int n);
    void (*texture_release)(const tex_t *t);          // optional: the serial's host objects may go
    // optional, together: a host view of a texture that view_make made none of because its texels are the CPU's (the
    // texture arena of the texture upload: registering thousands of textures must take no memory), asked for at the moment the replay
    // binds the texture (bindless_image_slot); and the submission being recorded, `value`, reads that view (any view the
    // replay binds), once per call of bindless_image_slot.
    void *(*view_demand)(const tex_t *t, const view_t *v, int type);
    void (*view_use)(void *view, uint64_t value);
} BindlessOps;

typedef struct BindlessConfig {
    uint32_t images;        // slots of binding 0 (VKP_GLOBAL_IMAGES): the game's pool and then the second views
    uint32_t samplers;      // slots of binding 1 (VKP_GLOBAL_SAMPLERS)
    uint32_t second;        // the last `second` image slots are for second views (VKP_SECOND_IMAGES): IDs below
                            // images - second are the game's, so the pool's own size (VKP_POOL_IMAGES) is images - second
    uint32_t max_copies;    // copies of the arrays at most
    uint32_t max_samplers;  // host samplers live at most (the device's allocation limit less what the rest of the back end holds)
} BindlessConfig;

// The timeline: the newest value done, and a wait for a value (which the completion thread reaches on its own).
typedef struct BindlessClock {
    uint64_t (*done)(void);
    void (*wait)(uint64_t value);
} BindlessClock;

// Open the tables over ops. Once; close frees every copy and object (no submission may be in flight).
void bindless_open(const BindlessOps *ops, const BindlessConfig *cfg, const BindlessClock *clock);
void bindless_close(void);
int bindless_is_open(void);

// ---- the front end's side (gfx.h): one entry of a pool per call, from any guest thread
// An ID past the pool or past the game's part of the array, or a view the table cannot hold, is a named fatal.
void bindless_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v);
void bindless_sampler_register(const dpool_ref *pool, int id, const samp_par *s);
// A texture the game finalized (gfx.h texture_release): the host objects of its serial may go; no entry changes.
void bindless_texture_release(const tex_t *t);
// The storage [lo, hi) is gone (its pool was finalized): every entry whose view is over it shows the placeholder, and no
// view over it is shared with a registration made later: a pool remade at the same address registers textures at the same
// storage with the same description, which must stand on new images, not on the finalized pool's (whose memory is freed
// while a copy may still show the old view).
// The range stays closed until bindless_storage_settled: a registration or a bind over it takes the placeholder, so that
// the host cannot make a view of an image of the pool while the host's own pool_gone has not yet run. Every call is
// followed by a settled call for the same range, once the host's images of the pool can no longer be found.
void bindless_storage_gone(uint64_t lo, uint64_t hi);
void bindless_storage_settled(uint64_t lo, uint64_t hi);

// ---- the replay's side: one submission at a time (the caller holds the back end's submission lock)
// The copy for the submission that will signal `value`, brought up to date for the texture pool and the sampler pool the
// submission sets (NULL: the pool the last submission used, or none). Returns the copy's set: bound once, at set 0.
void *bindless_acquire(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers);
// The index into binding 0 that reads texture ID `id` of the pool acquired as a view of `type` (BV_NONE: the type its
// texture asks for): `id` itself, or a second slot written into the copy being replayed; UINT32_MAX for an ID no
// texture was registered at, or one that cannot be seen as `type`. The caller then pushes slot | sampler << 20. An entry
// whose texture the host made no view of at registration gets it now (view_demand), and the view about to be read is
// told to the host (view_use, with the value of the submission the copy was acquired for).
uint32_t bindless_image_slot(uint32_t id, int type);
// The texture and view registered at `id` in the texture pool of the copy being replayed (an image handle names a
// texture pool ID, as nvnDeviceGetImageHandle makes it): copied into *t and *v; 0 for an ID nothing is registered at.
int bindless_texture_entry(uint32_t id, tex_t *t, view_t *v);
// The index into binding 1 for sampler ID `id`: `id`, after the check against the array.
uint32_t bindless_sampler_slot(uint32_t id);

typedef struct BindlessStats {
    uint64_t registrations, unchanged;           // texture and sampler registrations; those that changed no entry
    uint64_t writes, image_writes, sampler_writes;   // descriptors written into copies
    uint64_t acquires, copy_switches, copy_waits;    // copies taken instead of the current one; waits for a pending copy
    uint64_t full_syncs, tidy_syncs;
    uint64_t copies, second_slots, second_live;
    uint64_t views_made, views_live, samplers_made, samplers_live, retired;
    uint64_t log_head;
} BindlessStats;
void bindless_stats(BindlessStats *out);
