// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the view decision (oot2/vk_view_plan.h), driven by tests/test_runtime_vk_view_plan.py: `vk_view_plan_test`
// prints one line "<name>: ok" for each check that holds and "<name>: FAIL ..." (exit 1) for one that does not. It takes
// no device, so continuous integration runs it. The checks come in the three kinds of validation errors the first forest
// run counted, each wrong under the decision the render target path had before (a 2D or 2D-array view by the layer count,
// the layers and usage taken as asked) and right under this one:
//   volume_*  a 2D view of a 3D image needs the image made with the compatibility flags
//   layers_*  a first layer or a layer count past the image's layers (a volume's depth slices) is refused
//   usage_*   a view usage the image was made without is refused
#include <stdio.h>
#include <stdlib.h>

#include "oot2/bindless.h"
#include "oot2/vk_view_plan.h"

static int g_fail;

#define CHECK(name, cond)                                                                                              \
    do {                                                                                                               \
        if (cond) printf("%s: ok\n", name);                                                                            \
        else printf("%s: FAIL (line %d)\n", name, __LINE__), g_fail = 1;                                               \
    } while (0)

enum { COLOR = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, DEPTH = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
       STORE = VK_IMAGE_USAGE_STORAGE_BIT, SAMPLE = VK_IMAGE_USAGE_SAMPLED_BIT };
#define ALL_USAGE (COLOR | STORE | SAMPLE | VK_IMAGE_USAGE_TRANSFER_DST_BIT)

// the volume as image_make creates it on a device with (or without) image2DViewOf3D
static VkvImage volume(uint32_t depth, uint32_t levels, int view2d, VkImageUsageFlags usage) {
    return (VkvImage){VK_IMAGE_TYPE_3D, depth, 1, levels, vkv_image_flags(VK_IMAGE_TYPE_3D, view2d), usage, view2d};
}
static VkvImage plain2d(uint32_t layers, uint32_t levels, VkImageUsageFlags usage) {
    return (VkvImage){VK_IMAGE_TYPE_2D, 1, layers, levels, 0, usage, 1};
}
static int plan(const VkvImage *im, int type, uint32_t level, uint32_t base, uint32_t n, VkImageUsageFlags usage,
                VkvPlan *p) {
    VkvWant w = {type, level, base, n, usage};
    const char *why;
    return vkv_plan(im, &w, p, &why);
}
static const char *why_of(const VkvImage *im, int type, uint32_t level, uint32_t base, uint32_t n, VkImageUsageFlags usage) {
    VkvWant w = {type, level, base, n, usage};
    VkvPlan p;
    const char *why;
    return vkv_plan(im, &w, &p, &why) ? "" : why;
}
static int has(const char *s, const char *needle) {
    for (; *s; s++) {
        const char *a = s, *b = needle;
        while (*a && *b && *a == *b) a++, b++;
        if (!*b) return 1;
    }
    return 0;
}

static void volume_checks(void) {
    VkvPlan p;
    VkvImage v = volume(16, 1, 0, ALL_USAGE);
    // the image is made 2D-array compatible, and 2D-view compatible only where the device has the feature
    CHECK("volume_flags_array_compatible",
          (vkv_image_flags(VK_IMAGE_TYPE_3D, 0) & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) != 0);
    CHECK("volume_flags_view_compatible_with_feature",
          (vkv_image_flags(VK_IMAGE_TYPE_3D, 1) & VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT) != 0);
    CHECK("volume_flags_view_compatible_not_without_feature",
          (vkv_image_flags(VK_IMAGE_TYPE_3D, 0) & VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT) == 0);
    CHECK("volume_flags_none_for_2d", vkv_image_flags(VK_IMAGE_TYPE_2D, 1) == 0 && vkv_image_flags(VK_IMAGE_TYPE_1D, 1) == 0);
    // a render target of a volume: a 2D view of one slice
    CHECK("volume_target_is_a_2d_view_of_one_slice",
          plan(&v, VKV_PICK, 0, 3, 1, COLOR, &p) && p.type == VK_IMAGE_VIEW_TYPE_2D && p.base_layer == 3 && p.layers == 1);
    CHECK("volume_target_of_slices_is_a_2d_array_view",
          plan(&v, VKV_PICK, 0, 2, 4, COLOR, &p) && p.type == VK_IMAGE_VIEW_TYPE_2D_ARRAY && p.base_layer == 2 && p.layers == 4);
    // an image made without the flags (what the target path assumed) gives no 2D view
    VkvImage bare = {VK_IMAGE_TYPE_3D, 16, 1, 1, 0, ALL_USAGE, 1};
    CHECK("volume_target_refused_without_the_array_flag",
          !plan(&bare, VKV_PICK, 0, 0, 1, COLOR, &p) && has(why_of(&bare, VKV_PICK, 0, 0, 1, COLOR), "2D-array compatible"));
    // a storage image over a slice needs the 2D-view flag and the feature
    VkvImage w = volume(16, 1, 1, ALL_USAGE);
    CHECK("volume_storage_slice_is_a_2d_view_with_the_feature",
          plan(&w, VKV_PICK, 0, 5, 1, STORE, &p) && p.type == VK_IMAGE_VIEW_TYPE_2D && p.base_layer == 5 && p.layers == 1);
    CHECK("volume_storage_slice_refused_without_the_feature",
          !plan(&v, VKV_PICK, 0, 5, 1, STORE, &p) && has(why_of(&v, VKV_PICK, 0, 5, 1, STORE), "VK_EXT_image_2d_view_of_3d"));
    // an array view of one slice is the 2D view of that slice; a run of slices has no storage view
    CHECK("volume_storage_array_view_of_one_slice_is_a_2d_view",
          plan(&w, BV_2D_ARRAY, 0, 5, 1, STORE, &p) && p.type == VK_IMAGE_VIEW_TYPE_2D && p.base_layer == 5 && p.layers == 1);
    CHECK("volume_storage_run_of_slices_refused", !plan(&w, BV_2D_ARRAY, 0, 5, 2, STORE, &p) && !plan(&w, VKV_PICK, 0, 5, 2, STORE, &p));
    VkvImage nodev = volume(16, 1, 1, ALL_USAGE);
    nodev.view2d_of_3d = 0;   // the flag on an image, the feature off on the device
    CHECK("volume_storage_slice_refused_when_the_device_feature_is_off", !plan(&nodev, VKV_PICK, 0, 5, 1, STORE, &p));
    // the volume as a 3D view is the whole volume
    CHECK("volume_3d_view_covers_the_volume", plan(&v, BV_3D, 0, 7, 4, STORE, &p) && p.type == VK_IMAGE_VIEW_TYPE_3D &&
                                                  p.base_layer == 0 && p.layers == 1);
    // a 2D image keeps the view type the caller asked, a cube needs its flag
    VkvImage a2 = plain2d(6, 3, ALL_USAGE);
    CHECK("volume_2d_image_view_types", plan(&a2, BV_2D_ARRAY, 0, 1, 2, SAMPLE, &p) && p.type == VK_IMAGE_VIEW_TYPE_2D_ARRAY &&
                                            plan(&a2, VKV_PICK, 0, 0, 1, COLOR, &p) && p.type == VK_IMAGE_VIEW_TYPE_2D);
    int cube_without = plan(&a2, BV_CUBE, 0, 0, 6, SAMPLE, &p);
    a2.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    CHECK("volume_cube_needs_its_flag", !cube_without && plan(&a2, BV_CUBE, 0, 0, 6, SAMPLE, &p));
}

static void layers_checks(void) {
    VkvPlan p;
    VkvImage v = volume(16, 3, 1, ALL_USAGE);
    CHECK("layers_volume_last_slice_ok", plan(&v, VKV_PICK, 0, 15, 1, COLOR, &p) && p.base_layer == 15);
    CHECK("layers_volume_slice_past_depth_refused", !plan(&v, VKV_PICK, 0, 16, 1, COLOR, &p) &&
                                                       has(why_of(&v, VKV_PICK, 0, 16, 1, COLOR), "past the depth slices"));
    CHECK("layers_volume_run_past_depth_refused", !plan(&v, VKV_PICK, 0, 14, 3, COLOR, &p));
    CHECK("layers_volume_depth_halves_per_level", plan(&v, VKV_PICK, 2, 3, 1, COLOR, &p) && !plan(&v, VKV_PICK, 2, 4, 1, COLOR, &p));
    CHECK("layers_volume_level_past_levels_refused", !plan(&v, VKV_PICK, 3, 0, 1, COLOR, &p));
    VkvImage a = plain2d(4, 1, ALL_USAGE);
    CHECK("layers_2d_array_last_layer_ok", plan(&a, VKV_PICK, 0, 3, 1, COLOR, &p) && p.base_layer == 3 && p.layers == 1);
    CHECK("layers_2d_array_layer_past_refused", !plan(&a, VKV_PICK, 0, 4, 1, COLOR, &p) &&
                                                   has(why_of(&a, VKV_PICK, 0, 4, 1, COLOR), "past the image's layers"));
    CHECK("layers_2d_array_run_past_refused", !plan(&a, VKV_PICK, 0, 2, 3, COLOR, &p));
    // a texture of one layer asked for layer 1 (the forest run's baseArrayLayer 1 of 1)
    VkvImage one = plain2d(1, 1, ALL_USAGE);
    CHECK("layers_single_layer_image_layer_1_refused", !plan(&one, VKV_PICK, 0, 1, 1, COLOR, &p) && plan(&one, VKV_PICK, 0, 0, 1, COLOR, &p));
    // a count taken as "the rest" from a first layer past the image wraps to a huge count: still refused
    CHECK("layers_wrapped_rest_refused", !plan(&one, VKV_PICK, 0, 3, (uint32_t)(1u - 3u), COLOR, &p));
}

static void usage_checks(void) {
    VkvPlan p;
    VkvImage no_color = plain2d(1, 1, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | SAMPLE | STORE);
    CHECK("usage_color_attachment_refused_on_an_image_without_it",
          !plan(&no_color, VKV_PICK, 0, 0, 1, COLOR, &p) && has(why_of(&no_color, VKV_PICK, 0, 0, 1, COLOR), "colour attachment usage"));
    CHECK("usage_depth_attachment_refused_on_an_image_without_it",
          !plan(&no_color, VKV_PICK, 0, 0, 1, DEPTH, &p) && has(why_of(&no_color, VKV_PICK, 0, 0, 1, DEPTH), "depth-stencil"));
    VkvImage no_store = plain2d(1, 1, COLOR | SAMPLE);
    CHECK("usage_storage_refused_on_an_image_without_it",
          !plan(&no_store, VKV_PICK, 0, 0, 1, STORE, &p) && has(why_of(&no_store, VKV_PICK, 0, 0, 1, STORE), "storage usage"));
    // the gap the replay's render target reads to skip a target by name instead of stopping
    CHECK("usage_gap_names_the_missing_bits", vkv_usage_gap(&no_color, COLOR) == COLOR && vkv_usage_gap(&no_color, STORE | SAMPLE) == 0 &&
                                                  vkv_usage_gap(&no_color, COLOR | STORE) == COLOR);
    CHECK("usage_subset_accepted", plan(&no_store, VKV_PICK, 0, 0, 1, COLOR, &p) && plan(&no_color, VKV_PICK, 0, 0, 1, STORE, &p));
    // the volume of the forest run: usage without colour attachment and a slice view of it
    VkvImage vol_no_color = volume(8, 1, 1, SAMPLE | STORE);
    CHECK("usage_volume_without_color_attachment_refused", !plan(&vol_no_color, VKV_PICK, 0, 1, 1, COLOR, &p));
}

int main(void) {
    volume_checks();
    layers_checks();
    usage_checks();
    return g_fail;
}
