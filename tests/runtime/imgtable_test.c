// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Device-free test of the table of image memory by class (oot2/vk_format.h: vkf_bc_bytes, vkf_img_class, VkfImgTable), driven by
// tests/test_runtime_memline.py: `imgtable_test`. Built with runtime/gpu/vk_format.c alone. Prints "ok imgtable" and exits 0, or
// prints the failed check and exits 1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/vk_format.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static void bc_sizes(void) {
    // 1024x1024 with 11 levels: 1024 down to 4 is nine levels (4^2 * (4^9 - 1) / 3 = 1,398,096 texels), then 2x2 and 1x1 each
    // round to one 4x4 block of 16 texels
    CHECK(vkf_bc_bytes(1024, 1024, 1, 11, 1) == 1398096 + 32);
    CHECK(vkf_bc_bytes(1024, 1024, 1, 1, 1) == 1024ull * 1024);
    // a 3-layer 256x256 array, one level
    CHECK(vkf_bc_bytes(256, 256, 1, 1, 3) == 3ull * 65536);
    // odd sizes round up to blocks of 4: 5x7 is 8x8; its levels are 5x7 (8x8), 2x3 (4x4), 1x1 (4x4)
    CHECK(vkf_bc_bytes(5, 7, 1, 1, 1) == 64);
    CHECK(vkf_bc_bytes(5, 7, 1, 3, 1) == 64 + 16 + 16);
    // a volume 16x16x3 with two levels: 256 * 3, then 8x8 at depth 1
    CHECK(vkf_bc_bytes(16, 16, 3, 2, 1) == 768 + 64);
    // layers multiply the whole chain; zero levels or layers are read as one
    CHECK(vkf_bc_bytes(8, 8, 1, 4, 6) == 6ull * (64 + 16 + 16 + 16));
    CHECK(vkf_bc_bytes(8, 8, 1, 0, 0) == 64);
}

static void classes(void) {
    int seen[VKF_CLS_COUNT] = {0}, astc = 0;
    for (int f = VKF_FIRST; f <= VKF_LAST; f++) {
        const VkfRow *r = vkf_row(f);
        if (!r) continue;
        int dec = vkf_img_class(r, 1), nat = vkf_img_class(r, 0);
        if (r->flags & VKF_ASTC) {
            unsigned w, h;
            int srgb;
            CHECK(dec >= 0 && dec < VKF_CLS_NATIVE_BC);
            vkf_img_class_footprint(dec, &w, &h, &srgb);
            CHECK(w == r->bw && h == r->bh && srgb == ((r->flags & VKF_SRGB) != 0));
            CHECK(!seen[dec]++);
            CHECK(nat == VKF_CLS_OTHER);   // ASTC the device samples as blocks is not a decoded image, and not BC
            astc++;
        } else if (r->flags & VKF_BLOCK) {
            CHECK(dec == VKF_CLS_NATIVE_BC && nat == VKF_CLS_NATIVE_BC);
        } else {
            CHECK(dec == VKF_CLS_OTHER && nat == VKF_CLS_OTHER);
        }
    }
    CHECK(astc == 2 * VKF_ASTC_FOOTPRINTS);   // 14 footprints, linear and sRGB, each its own class
    CHECK(vkf_img_class(NULL, 0) == VKF_CLS_OTHER);
    unsigned w, h;
    int srgb;
    vkf_img_class_footprint(VKF_CLS_OTHER, &w, &h, &srgb);
    CHECK(w == 0);
}

static int class_of(unsigned bw, unsigned bh, int srgb) {
    for (int f = VKF_FIRST; f <= VKF_LAST; f++) {
        const VkfRow *r = vkf_row(f);
        if (r && (r->flags & VKF_ASTC) && r->bw == bw && r->bh == bh && ((r->flags & VKF_SRGB) != 0) == srgb)
            return vkf_img_class(r, 1);
    }
    return -1;
}

static void live_counts(void) {
    VkfImgTable t;
    memset(&t, 0, sizeof t);
    int c66s = class_of(6, 6, 1), c44 = class_of(4, 4, 0), c1212 = class_of(12, 12, 0);
    CHECK(c66s >= 0 && c44 >= 0 && c1212 >= 0);
    uint64_t bc = vkf_bc_bytes(1024, 1024, 1, 11, 1);
    uint64_t mem = 4ull * 1024 * 1024 * 4 / 3;   // a made-up driver size for 1024x1024 RGBA8 with mips
    vkf_img_table_add(&t, c66s, mem, bc, 1, +1);
    vkf_img_table_add(&t, c66s, mem, bc, 1, +1);
    vkf_img_table_add(&t, c44, 1000, 250, 0, +1);
    vkf_img_table_add(&t, c1212, 800, 200, 1, +1);
    vkf_img_table_add(&t, VKF_CLS_NATIVE_BC, 500, 0, 1, +1);
    vkf_img_table_add(&t, VKF_CLS_OTHER, 300, 0, 0, +1);
    VkfImgColumns c;
    vkf_img_columns(&t, &c);
    CHECK(c.dec_w[2] == 2 * mem && c.dec_w[0] == 1000 && c.dec_w[5] == 800 && c.dec_w[1] == 0 && c.dec_w[3] == 0);
    CHECK(c.dec_srgb == 2 * mem);
    CHECK(c.dec_arena == 2 * mem + 800);
    CHECK(c.dec_images == 4);
    CHECK(c.bc_equiv == 2 * bc + 250 + 200);
    CHECK(c.native_bc == 500 && c.other == 300);
    // destroying an image takes exactly what it added, so a texture arena that frees leaves the counters at their live value
    vkf_img_table_add(&t, c66s, mem, bc, 1, -1);
    vkf_img_columns(&t, &c);
    CHECK(c.dec_w[2] == mem && c.dec_srgb == mem && c.dec_arena == mem + 800 && c.dec_images == 3);
    CHECK(c.bc_equiv == bc + 250 + 200);
    vkf_img_table_add(&t, c66s, mem, bc, 1, -1);
    vkf_img_table_add(&t, c44, 1000, 250, 0, -1);
    vkf_img_table_add(&t, c1212, 800, 200, 1, -1);
    vkf_img_table_add(&t, VKF_CLS_NATIVE_BC, 500, 0, 1, -1);
    vkf_img_table_add(&t, VKF_CLS_OTHER, 300, 0, 0, -1);
    VkfImgTable zero;
    memset(&zero, 0, sizeof zero);
    CHECK(!memcmp(&t, &zero, sizeof t));
}

int main(void) {
    bc_sizes();
    classes();
    live_counts();
    printf("ok imgtable\n");
    return 0;
}
