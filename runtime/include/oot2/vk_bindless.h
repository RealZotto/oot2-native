// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The bindless tables on a device (oot2/bindless.h gives them their logic, oot2/vk_image.h their images): the copies of
// set 0 (oot2/vk_pipeline.h), the host samplers shared by state, the placeholder every unregistered entry shows, and the
// calls the replay makes for a submission. Opened with the device (gpu_vk_open) and closed with it.
//
// A copy is a descriptor set allocated from a pool of its own, from a layout defined as vkp_set0_bindings defines
// set 0 (so that it is compatible with every pipeline layout of oot2/vk_pipeline.h), with the binding flags
// update-after-bind, partially bound and update-unused-while-pending. The back end writes a copy only while no submitted
// command buffer can read the entry written: a copy no pending submission uses, or, for a second view's slot, a slot
// no pending submission's handle names (update-unused-while-pending).
#pragma once
#include <stdint.h>

#include "volk.h"

#include "oot2/bindless.h"

// Binding 0 of a copy is the game's texture pool at its own indices (VKP_POOL_IMAGES of them), then VKP_SECOND_IMAGES slots for
// second views (oot2/vk_pipeline.h, oot2/bindless.h): a game ID is never taken by a second view.
#define VKB_MAX_COPIES 16u      // copies of set 0: submissions with a different table in flight at once

void vkb_open(void);    // after the backing map; gpu_vk_open calls it
void vkb_close(void);   // no submission in flight; gpu_vk_close calls it, before the backing map closes
int vkb_is_open(void);
VkDescriptorSetLayout vkb_set_layout(void);

// Take the copy of set 0 for the submission that will signal `value` (bindless_acquire), record into cmd the layout
// changes of every image made so far and stamp the pools that hold images with value. The set is bound at set 0 of the
// submission's pipeline layouts.
VkDescriptorSet vkb_acquire(VkCommandBuffer cmd, uint64_t value, const dpool_ref *textures, const dpool_ref *samplers);
// bindless_image_slot, with the layout change of an image the slot's view made recorded into cmd.
uint32_t vkb_image_slot(VkCommandBuffer cmd, uint32_t id, int type);

// The Vulkan sampler state samp_par stands for (exposed for the tests that check the mapping of every NVN value).
void vkb_sampler_info(const samp_par *s, VkSamplerCreateInfo *out, VkSamplerReductionModeCreateInfo *reduction);

typedef struct VkbStats {
    uint64_t border_approx;   // samplers whose border colour is not one of Vulkan's three: the nearest is used
    uint64_t wrap_approx;     // wrap modes with no Vulkan equivalent (mirror clamp to border)
} VkbStats;
void vkb_stats(VkbStats *out);
