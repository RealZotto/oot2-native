// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The view decision of oot2/vk_view_plan.h.
#include "oot2/vk_view_plan.h"

#include "oot2/bindless.h"

#define ATTACH (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)

VkImageCreateFlags vkv_image_flags(VkImageType type, int view2d_of_3d) {
    if (type != VK_IMAGE_TYPE_3D) return 0;
    return VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT | (view2d_of_3d ? VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT : 0);
}

VkImageUsageFlags vkv_usage_gap(const VkvImage *im, VkImageUsageFlags usage) { return usage & ~im->usage; }

int vkv_plan(const VkvImage *im, const VkvWant *w, VkvPlan *plan, const char **why) {
    static const char *const none = "";
    *why = none;
    uint32_t n = w->nlayers ? w->nlayers : 1;
    if (w->usage & ~im->usage) {
        *why = (w->usage & ~im->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ? "its image was made without colour attachment usage"
               : (w->usage & ~im->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
                   ? "its image was made without depth-stencil attachment usage"
               : (w->usage & ~im->usage & VK_IMAGE_USAGE_STORAGE_BIT) ? "its image was made without storage usage"
                                                                       : "its image was made without that usage";
        return 0;
    }
    if (w->level >= im->levels) {
        *why = "the level is past the image's levels";
        return 0;
    }
    if (im->type != VK_IMAGE_TYPE_3D) {
        if ((uint64_t)w->base_layer + n > im->layers) {   // 64 bits: a count of 0 - base wraps
            *why = "the layers are past the image's layers";
            return 0;
        }
        int t = w->type == VKV_PICK ? (n > 1 ? BV_2D_ARRAY : BV_2D) : w->type;
        if (im->type == VK_IMAGE_TYPE_1D) {
            if (t != BV_1D && t != BV_1D_ARRAY) {
                *why = "a 1D image gives only 1D views";
                return 0;
            }
        } else if (t == BV_CUBE || t == BV_CUBE_ARRAY) {
            if (!(im->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)) {
                *why = "its image was made without cube compatibility";
                return 0;
            }
        } else if (t != BV_2D && t != BV_2D_ARRAY) {
            *why = "a 2D image gives only 2D and cube views";
            return 0;
        }
        static const VkImageViewType map[BV_TYPES] = {VK_IMAGE_VIEW_TYPE_1D, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_3D,
                                                      VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_1D_ARRAY,
                                                      VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY};
        *plan = (VkvPlan){map[t], w->base_layer, n};
        return 1;
    }
    // a 3D image
    if (w->type == BV_3D) {   // the whole volume
        *plan = (VkvPlan){VK_IMAGE_VIEW_TYPE_3D, 0, 1};
        return 1;
    }
    if (w->type != VKV_PICK && w->type != BV_2D && w->type != BV_2D_ARRAY) {
        *why = "a 3D image gives only 3D, 2D and 2D array views";
        return 0;
    }
    uint32_t slices = im->depth >> w->level;
    if (!slices) slices = 1;
    if ((uint64_t)w->base_layer + n > slices) {
        *why = "the layers are past the depth slices of the 3D image at that level";
        return 0;
    }
    if (w->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        if (!(im->flags & VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT) || !im->view2d_of_3d) {
            *why = "a 2D view of a 3D image for storage needs the image made 2D-view compatible, on a device with "
                   "VK_EXT_image_2d_view_of_3d";
            return 0;
        }
        // a storage view of a slice is a 2D view of one slice; an array view of one slice is that same slice
        if (n != 1) {
            *why = "a storage view of a 3D image takes one slice, as a 2D view";
            return 0;
        }
        *plan = (VkvPlan){VK_IMAGE_VIEW_TYPE_2D, w->base_layer, 1};
        return 1;
    } else if (!(im->flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT)) {
        *why = "a 2D view of a 3D image needs the image made 2D-array compatible";
        return 0;
    }
    if (!(w->usage & ATTACH) && !(w->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
        *why = "a 2D view of a 3D image serves a render target or a storage image only";
        return 0;
    }
    int array = w->type == BV_2D_ARRAY || (w->type == VKV_PICK && n > 1);
    *plan = (VkvPlan){array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, w->base_layer, n};
    return 1;
}
