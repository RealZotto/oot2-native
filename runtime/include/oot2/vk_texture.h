// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Uploads of CPU-written textures on the Vulkan back end (architecture D19, section 8): the device half of
// oot2/texmem.h (the staging ring and the texture arena) and of oot2/texsync.h (which regions are stale), and the compute
// pass that untiles block-linear storage (runtime/gpu/shaders/untile.comp, oot2/untile.h).
//
// An image of the arena (vk_image.c makes it: its memory is an extent of the arena, not the game's storage) is made
// current for a submission when the replay binds it (vkb_image_slot): the storage's pages are queried, and each stale
// region is copied from guest memory into the staging ring at once, then, in the submission's prologue (gpu_vk_prologue:
// after every earlier submission's work and before this one's own), untiled by the compute pass into a scratch buffer
// and copied from there into the image. So a CPU write before a submit is read by that submission, as on the console,
// and an earlier submission still reading the image has finished before the prologue rewrites it. Images live in
// VK_IMAGE_LAYOUT_GENERAL. Linear textures need no untiling and are copied from the ring as they are.
//
// This layer never calls rt_mem_written: it reads the page generations (oot2/pagegen.h).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/nvn_obj.h"
#include "oot2/texsync.h"
#include "oot2/untile.h"

// What an image keeps of its uploads.
typedef struct VktState {
    TexSync sync;
    uint64_t current_for;    // the newest submission value the image was made current for (atomic)
} VktState;

void vkt_open(void);       // with the tables (vkb_open); once
void vkt_close(void);      // no submission may be in flight
int vkt_is_open(void);

// Make the image of texture d (storage at `storage` in guest memory) current for the submission `value` that is being
// recorded, on its thread: query, copy the stale regions into the ring, record their untiling and copies in the
// prologue. image is bound to memory and in no layout yet on the first call (it is transitioned here). A texture this
// layer does not upload stops by name. astc_decode: the texture is ASTC and the image is the RGBA8 (or sRGB) image of its
// decoded texels, not ASTC (the device does not sample ASTC): each region is untiled into rows of blocks, decoded by
// runtime/gpu/shaders/astc_decode.comp into RGBA8 texels in the scratch buffer, and copied from there into the image. The
// decode happens here, at the first bind and again for a rewritten region, never at registration.
void vkt_upload(const nvn_texdesc *d, uint64_t storage, VkImage image, VkImageAspectFlags aspect, VktState *st,
                uint64_t value, int astc_decode);

// The bytes between rows of a linear texture's storage: its stride, else the width's blocks rounded as the upload rounds them.
uint64_t vkt_linear_stride(const nvn_texdesc *d);

// The compute pass alone, on a device, for the checks: the region g's span bytes at src untiled in the given order into
// dst (pitch bytes a row, at least row_bytes rounded up to 16), through the ring, in an internal submission that is
// waited for.
void vkt_untile_host(const UntileGeom *g, int order, const uint8_t *src, uint8_t *dst, size_t pitch);

typedef struct VktStats {
    uint64_t uploads;          // images made current with something to copy
    uint64_t regions, raw_bytes, linear_bytes;   // regions untiled and copied, the bytes read from guest memory and written
    uint64_t dispatches, copies, zero_regions;
    uint64_t scratch_resets, scratch_grown;
    uint64_t astc_regions, astc_blocks, astc_bytes;   // regions decoded from ASTC, their blocks, and the RGBA8 bytes written for them
} VktStats;
void vkt_stats(VktStats *out);
int vkt_stats_try(VktStats *out);   // 0 when the lock is busy (the counters' lines never wait for it)
