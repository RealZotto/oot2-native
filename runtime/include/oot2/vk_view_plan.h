// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The decision of which image view an image can give (architecture D19): the view type, the first layer and the number of
// layers a render target or a storage image gets over the layers and levels the game's view names. It takes no device, so
// CI checks every rule; the callers (vkb_storage_view, the replay's render targets) apply the answer to vkCreateImageView
// and stop by name on a refusal, never creating a view the specification forbids.
//
// The rules (each is a validation error of a view created without it):
//   - a 2D or 2D-array view of a 3D image needs the image made with VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT (a render
//     target: colour or depth attachment, a slice of the volume or a run of slices as layers) or, for a storage image,
//     VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT with the device feature image2DViewOf3D enabled; a 2D view of such an
//     image counts its layers in depth slices of the level, not in the image's array layers (which are 1)
//   - a view of any other image takes layers inside the image's array layers
//   - a view's usage is a subset of the image's usage
//   - a 3D view of a 3D image always covers the whole volume (first layer 0, one layer)
#pragma once
#include <stdint.h>

#include "volk.h"

// What the image was made as.
typedef struct VkvImage {
    VkImageType type;
    uint32_t depth, layers, levels;   // extent depth of level 0, array layers, mip levels
    VkImageCreateFlags flags;
    VkImageUsageFlags usage;
    int view2d_of_3d;                 // the device enabled image2DViewOf3D
} VkvImage;

// What the caller wants. type is a VkImageViewType, or VKV_PICK for the type the usage and the layer count call for
// (a render target: 2D for one layer, 2D array for more). nlayers 0 is one.
enum { VKV_PICK = -1 };
typedef struct VkvWant {
    int type;
    uint32_t level, base_layer, nlayers;
    VkImageUsageFlags usage;
} VkvWant;

typedef struct VkvPlan {
    VkImageViewType type;
    uint32_t base_layer, layers;
} VkvPlan;

// 1 and *plan when the image gives the view; 0 and *why (a sentence that completes "the view cannot be made: ") when it
// cannot. A refusal is a stop for the caller, never a clamp.
int vkv_plan(const VkvImage *im, const VkvWant *w, VkvPlan *plan, const char **why);

// The creation flags a 3D image of a format that takes them carries so the views above exist: the attachment views
// (core, always) and, where the device enabled image2DViewOf3D, the storage ones. Other images get none.
VkImageCreateFlags vkv_image_flags(VkImageType type, int view2d_of_3d);
