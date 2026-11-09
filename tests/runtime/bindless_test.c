// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the format table (oot2/vk_format.h) and the bindless tables (oot2/bindless.h, oot2/vk_bindless.h,
// oot2/vk_image.h, the bindless tables), driven by tests/test_runtime_bindless.py: `bindless_test <case> [files]`, one case per
// process.
//
// Without a device (continuous integration runs them):
//   formats          one line per NVN format, 0x01 to 0x9C: the row of the table and what the front end sizes the format at
//   tables           the mapping with a test's stand-ins for the host: a registration changes one entry; the same
//                    registration again changes none; an ID registered again while a pending submission reads the old
//                    view takes another copy and leaves the old one as it was; a released serial clears no entry; 4,096
//                    sampler registrations with 300 distinct states hold 300 host samplers; second views take second
//                    slots, are written into the copy of the submission that asked and reach the others through the log;
//                    two pools; a copy behind by more than the log holds is brought up by a full sync; a view is freed
//                    only after the newest submission that used a copy holding it; close frees everything
//   demand           a host that makes no view at registration (the texture arena's CPU-written textures) is asked for it when the
//                    replay binds the texture: registering takes nothing, the first bind asks the host (view_demand) and tells it which
//                    submission reads the view (view_use), every copy gets the view through the log, a second view type is asked for
//                    too, and registering another texture at the ID puts the placeholder back until the next bind
//   id_fatal         a texture registered past the game's part of the array is a named fatal (exit 70)
// On a device with every item of oot2/vk_required.h (the driver skips them without one):
//   device           ...
// Every case prints what it checked; a failed check prints its line and exits 1.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/bindless.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_format.h"
#include "oot2/vk_image.h"
#include "oot2/vk_pipeline.h"
#include "plat.h"
#include "vk_draw.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

typedef unsigned long long ull;
enum { MiB = 1 << 20, KiB = 1 << 10 };

#define EXIT_NO_DEVICE 3   // no loader or no device: the driver skips

static void case_usable(void) {
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        exit(EXIT_NO_DEVICE);
    }
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) {
        printf("no Vulkan 1.3 instance\n");
        exit(EXIT_NO_DEVICE);
    }
    volkLoadInstanceOnly(inst);
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL) == VK_SUCCESS);
    if (!n) {
        printf("no Vulkan device\n");
        exit(EXIT_NO_DEVICE);
    }
    VkPhysicalDevice devs[16];
    if (n > 16) n = 16;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, devs) >= 0);
    int usable = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t ne = 0;
        CHECK(vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, NULL) == VK_SUCCESS);
        VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
        CHECK(ext && vkEnumerateDeviceExtensionProperties(devs[i], NULL, &ne, ext) == VK_SUCCESS);
        Oot2VkFacts f;
        oot2_vk_query(vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceFeatures2, devs[i], ext, ne, &f);
        free(ext);
        unsigned char has[OOT2_VK_ITEM_COUNT];
        usable += oot2_vk_judge(&f, has) == OOT2_VK_ITEM_COUNT;
    }
    vkDestroyInstance(inst, NULL);
    printf("usable %d\n", usable);
}

// ---------- the format table

static void case_formats(void) {
    for (int f = 0; f <= 0x9D; f++) {
        const VkfRow *r = vkf_row(f);
        if (!r) {
            printf("fmt %02x none\n", f);
            continue;
        }
        int bw = 0, bh = 0;
        int nvn_bpb = nvn_fmt_bpb(f, &bw, &bh);
        printf("fmt %02x vk=%d vertex=%d alt=%d flags=%d bw=%d bh=%d bpb=%d swz=%d%d%d%d altswz=%d%d%d%d nvn_bpb=%d nvn_bw=%d "
               "nvn_bh=%d reason=%s\n",
               f, (int)r->vk, (int)r->vertex, (int)r->alt, r->flags, r->bw, r->bh, r->bpb, r->swz[0], r->swz[1], r->swz[2],
               r->swz[3], r->alt_swz[0], r->alt_swz[1], r->alt_swz[2], r->alt_swz[3], nvn_bpb, bw, bh,
               r->reason ? "yes" : "no");
    }
    // The class and the twin the images use.
    printf("class %d %d %d\n", vkf_class(VK_FORMAT_R8G8B8A8_UNORM), vkf_class(VK_FORMAT_R32_UINT),
           vkf_class(VK_FORMAT_R16G16_SFLOAT));
    CHECK(vkf_class(VK_FORMAT_R8G8B8A8_UNORM) == vkf_class(VK_FORMAT_R32_UINT));
    CHECK(vkf_class(VK_FORMAT_R8G8B8A8_UNORM) != vkf_class(VK_FORMAT_R16G16B16A16_UNORM));
    CHECK(vkf_class(VK_FORMAT_BC7_UNORM_BLOCK) == vkf_class(VK_FORMAT_BC7_SRGB_BLOCK));
    CHECK(vkf_class(VK_FORMAT_BC3_UNORM_BLOCK) != vkf_class(VK_FORMAT_BC7_UNORM_BLOCK));
    CHECK(vkf_linear_twin(VK_FORMAT_R8G8B8A8_SRGB) == VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(vkf_linear_twin(VK_FORMAT_R32_UINT) == VK_FORMAT_R32_UINT);
    printf("classes ok\n");
}

// What an image of decoded ASTC may serve (vkf_decoded, vkf_extent_ok, vkf_decoded_stop), over every format with a Vulkan
// format. The image is made in the linear twin of the format the device picked, as vk_image.c makes it: for an ASTC row that
// is the row's alt (RGBA8) on a device that does not sample ASTC, and the row's own format on one that does.
static const char *or_none(const char *s) { return s ? s : "none"; }

static void case_decoded(void) {
    printf("null row decoded=%d ok=%d\n", vkf_decoded(NULL, VK_FORMAT_R8G8B8A8_UNORM),
           vkf_extent_ok(NULL, VK_FORMAT_R8G8B8A8_UNORM));
    for (int f = VKF_FIRST; f <= VKF_LAST; f++) {
        const VkfRow *r = vkf_row(f);
        if (!r || r->vk == VK_FORMAT_UNDEFINED) continue;
        VkFormat own = vkf_linear_twin(r->vk);
        if (!(r->flags & VKF_ASTC)) {
            printf("plain fmt %02x decoded=%d ok=%d storage=[%s] copy=[%s]\n", f, vkf_decoded(r, own), vkf_extent_ok(r, own),
                   or_none(vkf_decoded_stop(r, own, VKF_USE_STORAGE_VIEW)), or_none(vkf_decoded_stop(r, own, VKF_USE_TEXEL_COPY)));
            continue;
        }
        VkFormat alt = vkf_linear_twin(r->alt);
        printf("astc fmt %02x decoded=%d ok=%d storage=[%s] copy=[%s] | sampled decoded=%d ok=%d storage=[%s] copy=[%s]\n", f,
               vkf_decoded(r, alt), vkf_extent_ok(r, alt), or_none(vkf_decoded_stop(r, alt, VKF_USE_STORAGE_VIEW)),
               or_none(vkf_decoded_stop(r, alt, VKF_USE_TEXEL_COPY)), vkf_decoded(r, own), vkf_extent_ok(r, own),
               or_none(vkf_decoded_stop(r, own, VKF_USE_STORAGE_VIEW)), or_none(vkf_decoded_stop(r, own, VKF_USE_TEXEL_COPY)));
    }
}

// ---------- stand-ins for the host

#define FAKE_IMAGES 1024
#define FAKE_SECOND 16
#define FAKE_SAMPLERS 4096
#define FAKE_COPIES 4

typedef struct FakeSet {
    void *img[FAKE_IMAGES], *smp[FAKE_SAMPLERS];
    int live;
} FakeSet;

static struct {
    uint64_t done;
    int views_made, views_freed, samplers_made, samplers_freed, sets_made, sets_freed, writes, releases;
    int last_type;
    FakeSet *sets[16];
    int fail_views;   // view_make answers NULL
    int lazy;         // view_make answers NULL and view_demand makes the view: a texture the CPU writes
    int demands, uses;
    uint64_t last_use_value;
    void *last_use_view;
} fk;

static void *fk_view_make(const tex_t *t, const view_t *v, int type) {
    (void)v;
    if (fk.fail_views || fk.lazy) return NULL;
    fk.views_made++;
    fk.last_type = type;
    uint64_t *o = malloc(sizeof *o * 2);
    o[0] = (uint64_t)(uintptr_t)t->mem;
    o[1] = (uint64_t)type;
    return o;
}
static void *fk_view_demand(const tex_t *t, const view_t *v, int type) {
    if (fk.fail_views) return NULL;   // a host that cannot make the view cannot when asked later either
    fk.demands++;
    uint64_t *o = malloc(sizeof *o * 2);
    (void)v;
    fk.views_made++;
    o[0] = (uint64_t)(uintptr_t)t->mem;
    o[1] = (uint64_t)type;
    return o;
}
static void fk_view_use(void *view, uint64_t value) {
    fk.uses++;
    fk.last_use_value = value;
    fk.last_use_view = view;
}
static void fk_view_free(void *view) {
    fk.views_freed++;
    free(view);
}
static void *fk_sampler_make(const samp_par *s) {
    fk.samplers_made++;
    float *o = malloc(sizeof *o);
    *o = s->lod_bias;
    return o;
}
static void fk_sampler_free(void *s) {
    fk.samplers_freed++;
    free(s);
}
static void *fk_set_make(void) {
    FakeSet *s = calloc(1, sizeof *s);
    s->live = 1;
    fk.sets[fk.sets_made++] = s;
    return s;
}
static void fk_set_free(void *set) {
    ((FakeSet *)set)->live = 0;
    fk.sets_freed++;
}
static void fk_write(void *set, const BindlessWrite *w, int n) {
    FakeSet *s = set;
    for (int i = 0; i < n; i++) {
        CHECK(w[i].index < (w[i].kind == BW_IMAGE ? FAKE_IMAGES : FAKE_SAMPLERS));
        (w[i].kind == BW_IMAGE ? s->img : s->smp)[w[i].index] = w[i].object;
    }
    fk.writes += n;
}
static void fk_release(const tex_t *t) {
    (void)t;
    fk.releases++;
}
static uint64_t fk_done(void) { return fk.done; }
static void fk_wait(uint64_t value) {
    if (fk.done < value) fk.done = value;   // the completion thread reaches it
}

static const BindlessOps k_fake_ops = {.view_make = fk_view_make, .view_free = fk_view_free, .sampler_make = fk_sampler_make,
                                       .sampler_free = fk_sampler_free, .set_make = fk_set_make, .set_free = fk_set_free,
                                       .write = fk_write, .texture_release = fk_release, .view_demand = fk_view_demand,
                                       .view_use = fk_view_use};
static const BindlessClock k_fake_clock = {fk_done, fk_wait};

static void fake_open(void) {
    memset(&fk, 0, sizeof fk);
    BindlessConfig cfg = {.images = FAKE_IMAGES, .samplers = FAKE_SAMPLERS, .second = FAKE_SECOND, .max_copies = FAKE_COPIES,
                          .max_samplers = 400};
    bindless_open(&k_fake_ops, &cfg, &k_fake_clock);
}

static tex_t tex_at(uint64_t storage, int target, int format, uint32_t w, uint32_t h, uint32_t d, uint32_t levels) {
    tex_t t;
    memset(&t, 0, sizeof t);
    t.magic = NVN_TEX_MAGIC;
    t.d.target = target, t.d.format = format, t.d.width = w, t.d.height = h, t.d.depth = d, t.d.levels = levels;
    t.d.swizzle_r = NVN_TEXTURE_SWIZZLE_R, t.d.swizzle_g = NVN_TEXTURE_SWIZZLE_G, t.d.swizzle_b = NVN_TEXTURE_SWIZZLE_B;
    t.d.swizzle_a = NVN_TEXTURE_SWIZZLE_A;
    t.mem = (uint8_t *)(uintptr_t)storage;
    static uint64_t serial;
    t.serial = ++serial;
    nvn_tex_layout(&t.d, &t.L);
    return t;
}

static samp_par samp_with(float bias) {
    samp_par s;
    memset(&s, 0, sizeof s);
    s.lod_max = 1000.0f, s.lod_bias = bias;
    return s;
}

// The object a copy shows at an image slot (NULL: the placeholder), and the storage a fake view stands on.
static uint64_t shows(void *set, uint32_t slot) {
    void *o = ((FakeSet *)set)->img[slot];
    return o ? ((uint64_t *)o)[0] : 0;
}

static uint64_t g_val;
static void *acq(const dpool_ref *t, const dpool_ref *s) { return bindless_acquire(++g_val, t, s); }

static void case_tables(void) {
    const dpool_ref pool = {0x100000, 1000, 0}, spool_ref = {0x200000, FAKE_SAMPLERS, 0};
    BindlessStats a, b;
    fake_open();
    tex_t ta = tex_at(0x1000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    tex_t tb = tex_at(0x2000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    tex_t tc = tex_at(0x3000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    tex_t td = tex_at(0x4000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);

    // One entry: the first acquire shows what was registered; a registration then costs one descriptor write.
    bindless_texture_register(&pool, 5, &ta, NULL);
    void *s1 = acq(&pool, &spool_ref);
    CHECK(shows(s1, 5) == 0x1000000 && shows(s1, 6) == 0);
    fk.done = g_val;
    bindless_stats(&a);
    bindless_texture_register(&pool, 6, &tb, NULL);
    void *s2 = acq(NULL, NULL);
    bindless_stats(&b);
    CHECK(s2 == s1);   // idle: brought up to date in place
    CHECK(b.image_writes - a.image_writes == 1 && b.sampler_writes == a.sampler_writes && b.full_syncs == a.full_syncs);
    CHECK(shows(s2, 6) == 0x2000000 && shows(s2, 5) == 0x1000000);
    printf("a registration changed one entry: one descriptor write, no full sync\n");

    // The same registration again changes nothing.
    bindless_stats(&a);
    bindless_texture_register(&pool, 5, &ta, NULL);
    tex_t ta2 = tex_at(0x1000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);   // a new serial over the same storage
    bindless_texture_register(&pool, 5, &ta2, NULL);
    bindless_stats(&b);
    CHECK(b.unchanged - a.unchanged == 2 && b.log_head == a.log_head && fk.views_made == 2);
    printf("registering the same view again (a new serial, the same storage and description) changed no entry\n");

    // An ID registered again while a pending submission may read the old view: another copy takes it.
    fk.done = g_val;
    void *s3 = acq(NULL, NULL);   // this submission is pending from here on: done stays below its value
    CHECK(s3 == s1);
    bindless_texture_register(&pool, 5, &tc, NULL);
    bindless_stats(&a);
    void *s4 = acq(NULL, NULL);
    bindless_stats(&b);
    CHECK(s4 != s3 && b.copy_switches - a.copy_switches == 1);
    CHECK(shows(s3, 5) == 0x1000000);   // the pending submission's copy is as it was
    CHECK(shows(s4, 5) == 0x3000000 && shows(s4, 6) == 0x2000000);
    printf("an ID registered again under a pending submission took another copy; the pending copy was not written\n");

    // The old view is freed only after the newest submission that used a copy holding it is done.
    CHECK(fk.views_freed == 0);
    uint64_t held_by = g_val - 1;   // the value of s3: the copy that holds the old view was last used by it
    for (int i = 0; i < 8; i++) {
        fk.done = held_by - 1;
        acq(NULL, NULL);
    }
    CHECK(fk.views_freed == 0);     // s3's value is not done: the view stays
    fk.done = g_val;
    for (int i = 0; i < 8; i++) acq(NULL, NULL);
    CHECK(fk.views_freed == 1);
    printf("the view an entry dropped stayed while the submission that could read it was pending, and went after\n");

    // A released serial clears no entry.
    int rel = fk.releases;
    bindless_texture_release(&tc);
    fk.done = g_val;
    void *s5 = acq(NULL, NULL);
    CHECK(fk.releases == rel + 1 && shows(s5, 5) == 0x3000000);
    printf("texture_release left the entry at its ID in place\n");

    // Samplers: 4,096 registrations, 300 distinct states.
    bindless_stats(&a);
    for (int i = 0; i < FAKE_SAMPLERS; i++) {
        samp_par s = samp_with((float)(i % 300));
        bindless_sampler_register(&spool_ref, i, &s);
    }
    bindless_stats(&b);
    CHECK(b.samplers_live == 300 && b.samplers_made - a.samplers_made == 300 && fk.samplers_made == 300);
    fk.done = g_val;
    void *s6 = acq(NULL, NULL);
    CHECK(((FakeSet *)s6)->smp[0] == ((FakeSet *)s6)->smp[300] && ((FakeSet *)s6)->smp[0] != ((FakeSet *)s6)->smp[1]);
    printf("4096 sampler registrations with 300 distinct states hold 300 host samplers\n");

    // Second views: written into the copy of the submission that asked, and reaching the others through the log.
    bindless_texture_register(&pool, 7, &td, NULL);
    fk.done = g_val;
    void *s7 = acq(NULL, NULL);
    uint32_t second = bindless_image_slot(7, BV_2D_ARRAY);
    CHECK(second >= FAKE_IMAGES - FAKE_SECOND && second < FAKE_IMAGES);
    CHECK(((uint64_t *)((FakeSet *)s7)->img[second])[1] == BV_2D_ARRAY && ((uint64_t *)((FakeSet *)s7)->img[7])[1] == BV_2D);
    CHECK(bindless_image_slot(7, BV_2D_ARRAY) == second && bindless_image_slot(7, BV_2D) == 7 &&
          bindless_image_slot(7, BV_NONE) == 7);
    // s7's submission is pending; a registration makes the next one take another copy, which has the second view too.
    bindless_texture_register(&pool, 8, &tb, NULL);
    void *s9 = acq(NULL, NULL);
    CHECK(s9 != s7 && ((FakeSet *)s9)->img[second] != NULL && ((uint64_t *)((FakeSet *)s9)->img[second])[1] == BV_2D_ARRAY);
    printf("an entry read as 2D and as 2D array has its own view of each type: slot %u holds the second, in every copy\n", second);
    bindless_stats(&a);
    CHECK(a.second_live == 1);
    tex_t td2 = tex_at(0x5000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    bindless_texture_register(&pool, 7, &td2, NULL);   // the entry changes: its second slot goes
    bindless_stats(&b);
    CHECK(b.second_live == 0);
    bindless_texture_register(&pool, 9, &tb, NULL);
    uint32_t again = bindless_image_slot(9, BV_2D_ARRAY);   // the copy being replayed still shows the old view at `second`
    CHECK(again != second && again >= FAKE_IMAGES - FAKE_SECOND);
    printf("a freed second slot was not taken again while a copy still showed its old view\n");
    fk.done = g_val;
    void *s10 = acq(NULL, NULL);
    CHECK(((FakeSet *)s10)->img[second] == NULL && shows(s10, 7) == 0x5000000);
    printf("registering the entry again released its second slot\n");

    // Two pools: a copy shows the pool a submission sets.
    const dpool_ref pool2 = {0x300000, 1000, 0};
    tex_t te = tex_at(0x6000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    bindless_texture_register(&pool2, 5, &te, NULL);
    fk.done = g_val;
    void *s11 = acq(&pool2, &spool_ref);
    CHECK(shows(s11, 5) == 0x6000000 && shows(s11, 7) == 0);
    fk.done = g_val;
    void *s12 = acq(&pool, &spool_ref);
    CHECK(shows(s12, 5) == 0x3000000 && shows(s12, 7) == 0x5000000);
    printf("a submission that sets the other pool sees its entries, and not the first pool's\n");

    // A copy far behind the log is brought up by a full sync and is right.
    for (int i = 0; i < 6000; i++) {
        tex_t tx = tex_at(0x7000000 + (uint64_t)i * 0x1000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
        bindless_texture_register(&pool, 100 + i % 500, &tx, NULL);
    }
    fk.done = g_val;
    bindless_stats(&a);
    void *sf = acq(NULL, NULL);
    bindless_stats(&b);
    for (int r = 0; r < 500; r += 77) CHECK(shows(sf, 100 + (uint32_t)r) == 0x7000000 + (5500ull + (uint64_t)r) * 0x1000);
    CHECK(b.full_syncs > a.full_syncs || sf != s12);
    printf("a copy far behind the log is right\n");

    // Close frees every host object.
    bindless_close();
    CHECK(fk.views_made == fk.views_freed && fk.samplers_made == fk.samplers_freed && fk.sets_made == fk.sets_freed);
    printf("close freed %d views, %d samplers and %d copies\n", fk.views_freed, fk.samplers_freed, fk.sets_freed);

    // A view the host cannot make shows the placeholder, and the registration is not an error.
    fake_open();
    g_val = 0;
    fk.fail_views = 1;
    bindless_texture_register(&pool, 1, &ta, NULL);
    void *sp = acq(&pool, &spool_ref);
    CHECK(shows(sp, 1) == 0 && bindless_image_slot(1, BV_NONE) == 1 && bindless_image_slot(1, BV_2D_ARRAY) == 1);
    bindless_close();
    printf("an entry with no host view shows the placeholder\n");

    // The view type of every target, and a key that tells two views of one storage apart.
    CHECK(bindless_view_type(&ta, NULL) == BV_2D);
    tex_t arr = tex_at(0x1000000, NVN_TEXTURE_TARGET_2D_ARRAY, 0x25, 64, 64, 4, 1);
    tex_t cube = tex_at(0x1000000, NVN_TEXTURE_TARGET_CUBEMAP, 0x25, 64, 64, 1, 1);
    tex_t buf = tex_at(0x1000000, NVN_TEXTURE_TARGET_BUFFER, 0x25, 64, 1, 1, 1);
    CHECK(bindless_view_type(&arr, NULL) == BV_2D_ARRAY && bindless_view_type(&cube, NULL) == BV_CUBE &&
          bindless_view_type(&buf, NULL) == BV_NONE);
    view_t v2d;
    memset(&v2d, 0, sizeof v2d);
    v2d.set_target = 1, v2d.target = NVN_TEXTURE_TARGET_2D;
    CHECK(bindless_view_type(&arr, &v2d) == BV_2D);
    BindlessKey k1, k2;
    bindless_key(&arr, NULL, BV_2D_ARRAY, &k1);
    bindless_key(&arr, &v2d, BV_2D, &k2);
    CHECK(memcmp(&k1, &k2, sizeof k1) != 0);
    printf("view types and keys\n");
}

static void case_demand(void) {
    const dpool_ref pool = {0x100000, 1000, 0}, spool_ref = {0x200000, FAKE_SAMPLERS, 0};
    fake_open();
    fk.lazy = 1;
    tex_t ta = tex_at(0x1000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    tex_t tb = tex_at(0x2000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    tex_t tc = tex_at(0x3000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    bindless_texture_register(&pool, 5, &ta, NULL);
    bindless_texture_register(&pool, 6, &tb, NULL);
    CHECK(fk.views_made == 0 && fk.demands == 0 && fk.uses == 0);
    void *s1 = acq(&pool, &spool_ref);
    CHECK(shows(s1, 5) == 0 && shows(s1, 6) == 0);
    printf("registering textures the host makes no view of took nothing: 0 views, 0 demands, placeholders in the first copy\n");

    // the replay binds ID 5
    CHECK(bindless_image_slot(5, BV_NONE) == 5);
    CHECK(fk.demands == 1 && fk.uses == 1 && fk.last_use_value == g_val && shows(s1, 5) == 0x1000000 && shows(s1, 6) == 0);
    CHECK(((uint64_t *)fk.last_use_view)[0] == 0x1000000);
    // the same submission binds it again: no second view, the host is told again
    CHECK(bindless_image_slot(5, BV_NONE) == 5);
    CHECK(fk.demands == 1 && fk.uses == 2 && fk.views_made == 1);
    printf("the first bind of ID 5 asked the host for the view and told it which submission reads it; the copy shows it\n");

    // another copy takes the view through the log (the first submission is pending), and the unbound ID shows the placeholder
    bindless_texture_register(&pool, 7, &tc, NULL);
    void *s2 = acq(NULL, NULL);
    CHECK(s2 != s1 && shows(s2, 5) == 0x1000000 && shows(s2, 6) == 0 && shows(s2, 7) == 0);
    printf("another copy took the view through the log; the IDs nothing bound still show the placeholder\n");

    // a second view type of the demanded entry is demanded too and gets its own slot
    uint32_t slot = bindless_image_slot(5, BV_2D_ARRAY);
    CHECK(slot != UINT32_MAX && slot >= FAKE_IMAGES - FAKE_SECOND && fk.demands == 2);
    CHECK(shows(s2, slot) == 0x1000000 && ((uint64_t *)((FakeSet *)s2)->img[slot])[1] == (uint64_t)BV_2D_ARRAY);
    CHECK(fk.uses == 3 && ((uint64_t *)fk.last_use_view)[1] == (uint64_t)BV_2D_ARRAY);
    printf("a second view type of a demanded entry was demanded too and got slot %u\n", slot);

    // another texture registered at the ID: the placeholder is back until the next bind
    tex_t td = tex_at(0x4000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    bindless_texture_register(&pool, 5, &td, NULL);
    fk.done = g_val;
    void *s3 = acq(NULL, NULL);
    CHECK(shows(s3, 5) == 0);
    int before = fk.demands;
    CHECK(bindless_image_slot(5, BV_NONE) == 5);
    CHECK(fk.demands == before + 1 && shows(s3, 5) == 0x4000000);
    printf("registering another texture at the ID put the placeholder back until the next bind\n");
    bindless_close();
}

static void case_id_fatal(void) {
    fake_open();
    const dpool_ref pool = {0x100000, 2000, 0};
    tex_t t = tex_at(0x1000000, NVN_TEXTURE_TARGET_2D, 0x25, 64, 64, 1, 1);
    bindless_texture_register(&pool, FAKE_IMAGES - FAKE_SECOND, &t, NULL);
    printf("not reached\n");
}

// ---------- on a device

static VkPipeline compute_pipeline(const VkpLayouts *l, const char *path) {
    FILE *f = fopen(path, "rb");
    DRAW_CHECK(f);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc((size_t)n);
    DRAW_CHECK(fread(code, 1, (size_t)n, f) == (size_t)n);
    fclose(f);
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)n, .pCode = code};
    VkShaderModule m;
    DRAW_CHECK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &m) == VK_SUCCESS);
    free(code);
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = l->compute};
    VkPipeline p;
    DRAW_CHECK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &p) == VK_SUCCESS);
    vkDestroyShaderModule(gpu_vk.device, m, NULL);
    return p;
}

typedef struct Reader {
    const VkpLayouts *l;
    VkPipeline pipe_2d, pipe_array, pipe_uint, pipe_sample;
    dpool_ref pool, spool;
    DrawBuf out;
} Reader;

// One submission that reads texel (x, y, z) of texture ID `id` (read as `type`: BV_NONE for its own) with sampler ID `sid`
// through `pipe`, held on `gate` when it is not NULL. The four words it wrote are in r->out after the value is done.
static uint64_t submit_read(Reader *r, VkPipeline pipe, uint32_t id, int type, uint32_t sid, uint32_t x, uint32_t y,
                            uint32_t z, VkSemaphore gate, DrawBuf *out) {
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    VkDescriptorSet set = vkb_acquire(cb, v, &r->pool, &r->spool);
    uint32_t slot = type == BV_NONE ? id : vkb_image_slot(cb, id, type);
    DRAW_CHECK(slot != UINT32_MAX);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->l->compute, 0, 1, &set, 0, NULL);
    VkDescriptorBufferInfo bi = {out->buf, 0, 64};
    VkWriteDescriptorSet pushw = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 9, .descriptorCount = 1,
                                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi};
    vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, r->l->compute, 1, 1, &pushw);
    uint32_t push[VKP_PUSH_BYTES / 4] = {slot | sid << 20, x, y, z};
    vkCmdPushConstants(cb, r->l->compute, vkp_push_range(VKP_COMPUTE).stageFlags, 0, VKP_PUSH_BYTES, push);
    vkCmdDispatch(cb, 1, 1, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gate, .value = 1,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    DRAW_CHECK(gpu_vk_end(gate ? &wait : NULL, gate ? 1 : 0) == v);
    return v;
}

static void wait_done(uint64_t v) { DRAW_CHECK(gpu_sync_wait(v, 20ull * 1000 * 1000 * 1000)); }

static uint32_t word(const DrawBuf *b, int i) { return ((const uint32_t *)b->map)[i]; }
static int near_byte(uint32_t bits, int want) {
    float f;
    memcpy(&f, &bits, sizeof f);
    int got = (int)(f * 255.0f + 0.5f);
    return got == want;
}
static int texel_is(const DrawBuf *b, int r, int g, int bl, int a) {
    return near_byte(word(b, 0), r) && near_byte(word(b, 1), g) && near_byte(word(b, 2), bl) && near_byte(word(b, 3), a);
}

static void fill_rgba(uint8_t *p, uint32_t w, uint32_t h, int dx, int dy, int blue) {
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint8_t *t = p + ((size_t)y * w + x) * 4;
            t[0] = (uint8_t)(x + (uint32_t)dx), t[1] = (uint8_t)(y + (uint32_t)dy), t[2] = (uint8_t)blue, t[3] = 255;
        }
}

static void case_device(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: bindless_test device <2d.spv> <array.spv> <uint.spv> <sample.spv>\n");
        exit(2);
    }
    gpu_vk_open();
    VkpLayouts lay;
    vkp_layouts_create(&lay);
    Reader r = {.l = &lay, .pool = {0x900000, 33025, 0}, .spool = {0xA00000, 4096, 0}};
    r.pipe_2d = compute_pipeline(&lay, argv[2]);
    r.pipe_array = compute_pipeline(&lay, argv[3]);
    r.pipe_uint = compute_pipeline(&lay, argv[4]);
    r.pipe_sample = compute_pipeline(&lay, argv[5]);
    DrawBuf out1 = draw_buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, 0);
    DrawBuf out2 = draw_buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, NULL, 0);

    // A pool the CPU cannot see, in the window: the extent the images stand in is made by the first image.
    const uint64_t WIN = GFX_GPU_WINDOW_BASE;
    pool_t pool = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS,
                   .mem = (uint8_t *)(uintptr_t)WIN, .size = 8 * MiB};
    DRAW_CHECK(backing_pool_init(&pool));
    tex_t ta = tex_at(WIN, NVN_TEXTURE_TARGET_2D, 0x25, 256, 256, 1, 1);
    tex_t tb = tex_at(WIN + 1 * MiB, NVN_TEXTURE_TARGET_2D, 0x25, 256, 256, 1, 1);
    uint8_t *texels = malloc(256 * 256 * 4);
    NVNcopyRegion all = {0, 0, 0, 256, 256, 1};
    fill_rgba(texels, 256, 256, 0, 0, 7);
    vkb_write_texels(&ta, NULL, &all, texels, 256 * 4, 256 * 256 * 4);
    fill_rgba(texels, 256, 256, 1, 1, 9);
    vkb_write_texels(&tb, NULL, &all, texels, 256 * 4, 256 * 256 * 4);
    VkbImageStats is;
    vkb_image_stats(&is);
    printf("images made %llu placed %llu unaliased %llu\n", (ull)is.made, (ull)is.placed, (ull)is.unaliased);
    CHECK(is.made == 2 && is.placed == 2);
    printf("two 256 x 256 textures stand at their pool offsets in the pool's extent\n");

    // Texel copies: what write_texels wrote read_texels returns, over a sub-region too.
    uint8_t *back = malloc(256 * 256 * 4);
    NVNcopyRegion sub = {16, 8, 0, 32, 16, 1};
    uint8_t part[32 * 16 * 4];
    for (int i = 0; i < (int)sizeof part; i++) part[i] = (uint8_t)(i * 7 + 3);
    vkb_write_texels(&ta, NULL, &sub, part, 32 * 4, 32 * 16 * 4);
    uint8_t got[32 * 16 * 4];
    memset(got, 0xEE, sizeof got);
    vkb_read_texels(&ta, NULL, &sub, got, 32 * 4, 32 * 16 * 4);
    CHECK(!memcmp(part, got, sizeof part));
    fill_rgba(texels, 256, 256, 0, 0, 7);
    vkb_write_texels(&ta, NULL, &all, texels, 256 * 4, 256 * 256 * 4);
    vkb_read_texels(&ta, NULL, &all, back, 256 * 4, 256 * 256 * 4);
    CHECK(!memcmp(texels, back, 256 * 256 * 4));
    printf("read_texels returned what write_texels wrote, over a region and over the whole texture\n");

    samp_par nearest;
    memset(&nearest, 0, sizeof nearest);
    nearest.lod_max = 1000.0f, nearest.ws = nearest.wt = nearest.wr = NVN_WRAP_MODE_CLAMP_TO_EDGE;
    bindless_sampler_register(&r.spool, 3, &nearest);

    // A registration, then a read through the table.
    bindless_texture_register(&r.pool, 5, &ta, NULL);
    uint64_t v = submit_read(&r, r.pipe_2d, 5, BV_NONE, 3, 3, 4, 0, NULL, &out1);
    wait_done(v);
    CHECK(texel_is(&out1, 3, 4, 7, 255));
    printf("ID 5 read as 2D: texel (3, 4) is (3, 4, 7, 255)\n");
    v = submit_read(&r, r.pipe_sample, 5, BV_NONE, 3, 10, 20, 0, NULL, &out1);
    wait_done(v);
    CHECK(texel_is(&out1, 10, 20, 7, 255));
    printf("a sample through the sampler array's entry 3 (nearest, clamp) gives the same texel\n");

    // A registration changes one entry.
    BindlessStats s0, s1;
    bindless_stats(&s0);
    bindless_texture_register(&r.pool, 6, &tb, NULL);
    v = submit_read(&r, r.pipe_2d, 6, BV_NONE, 3, 3, 4, 0, NULL, &out1);
    wait_done(v);
    bindless_stats(&s1);
    CHECK(texel_is(&out1, 4, 5, 9, 255));
    printf("a registration: %llu image writes, %llu sampler writes, %llu full syncs\n", (ull)(s1.image_writes - s0.image_writes),
           (ull)(s1.sampler_writes - s0.sampler_writes), (ull)(s1.full_syncs - s0.full_syncs));
    CHECK(s1.image_writes - s0.image_writes == 1 && s1.sampler_writes == s0.sampler_writes && s1.full_syncs == s0.full_syncs);

    // An ID registered again while an earlier submission, held on the GPU, reads the old image.
    VkSemaphoreTypeCreateInfo tci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &tci};
    VkSemaphore gate;
    DRAW_CHECK(vkCreateSemaphore(gpu_vk.device, &sci, NULL, &gate) == VK_SUCCESS);
    uint64_t held = submit_read(&r, r.pipe_2d, 5, BV_NONE, 3, 3, 4, 0, gate, &out1);
    bindless_texture_register(&r.pool, 5, &tb, NULL);   // ID 5 now names the other texture
    bindless_stats(&s0);
    uint64_t later = submit_read(&r, r.pipe_2d, 5, BV_NONE, 3, 3, 4, 0, NULL, &out2);
    bindless_stats(&s1);
    CHECK(s1.copy_switches > s0.copy_switches);
    VkSemaphoreSignalInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = gate, .value = 1};
    CHECK(gpu_sync_wait(held, 200ull * 1000 * 1000) == 0);   // still held
    DRAW_CHECK(vkSignalSemaphore(gpu_vk.device, &sig) == VK_SUCCESS);
    wait_done(later);
    wait_done(held);
    CHECK(texel_is(&out1, 3, 4, 7, 255));   // the held submission read the image ID 5 named when it was submitted
    CHECK(texel_is(&out2, 4, 5, 9, 255));   // the later one reads the new one
    vkDestroySemaphore(gpu_vk.device, gate, NULL);
    printf("ID 5 registered again under a held submission: the held one read the old image (3, 4, 7), the next the new (4, 5, 9)\n");

    // A texture read as 2D and as 2D array, and an array texture read as 2D.
    bindless_texture_register(&r.pool, 5, &ta, NULL);
    v = submit_read(&r, r.pipe_2d, 5, BV_NONE, 3, 30, 40, 0, NULL, &out1);
    uint64_t v2 = submit_read(&r, r.pipe_array, 5, BV_2D_ARRAY, 3, 30, 40, 0, NULL, &out2);
    wait_done(v2);
    CHECK(texel_is(&out1, 30, 40, 7, 255) && texel_is(&out2, 30, 40, 7, 255));
    printf("one entry sampled right as 2D (its own view) and as 2D array (a second slot)\n");
    tex_t tc = tex_at(WIN + 3 * MiB, NVN_TEXTURE_TARGET_2D_ARRAY, 0x25, 256, 256, 2, 1);
    for (uint32_t l = 0; l < 2; l++) {
        fill_rgba(texels, 256, 256, 0, 0, 20 + (int)l);
        NVNcopyRegion layer = {0, 0, (int)l, 256, 256, 1};
        vkb_write_texels(&tc, NULL, &layer, texels, 256 * 4, 256 * 256 * 4);
    }
    bindless_texture_register(&r.pool, 9, &tc, NULL);
    v = submit_read(&r, r.pipe_array, 9, BV_NONE, 3, 5, 6, 1, NULL, &out1);
    v2 = submit_read(&r, r.pipe_2d, 9, BV_2D, 3, 5, 6, 0, NULL, &out2);
    wait_done(v2);
    CHECK(texel_is(&out1, 5, 6, 21, 255) && texel_is(&out2, 5, 6, 20, 255));
    printf("an array texture sampled as array (layer 1 is (5, 6, 21)) and as 2D (layer 0 is (5, 6, 20))\n");

    // Two textures over one storage range see each other's GPU writes: written as RGBA8, read as R32_UINT.
    tex_t td1 = tex_at(WIN + 2 * MiB, NVN_TEXTURE_TARGET_2D, 0x25, 256, 256, 1, 1);
    tex_t td2 = tex_at(WIN + 2 * MiB, NVN_TEXTURE_TARGET_2D, 0x0B, 256, 256, 1, 1);
    bindless_texture_register(&r.pool, 11, &td1, NULL);
    vkb_image_stats(&is);
    uint64_t made = is.made;
    bindless_texture_register(&r.pool, 12, &td2, NULL);
    fill_rgba(texels, 256, 256, 0, 0, 7);
    texels[((size_t)4 * 256 + 3) * 4 + 0] = 1, texels[((size_t)4 * 256 + 3) * 4 + 1] = 2;
    texels[((size_t)4 * 256 + 3) * 4 + 2] = 3, texels[((size_t)4 * 256 + 3) * 4 + 3] = 4;
    vkb_write_texels(&td1, NULL, &all, texels, 256 * 4, 256 * 256 * 4);
    v = submit_read(&r, r.pipe_uint, 12, BV_NONE, 3, 3, 4, 0, NULL, &out1);
    wait_done(v);
    CHECK(word(&out1, 0) == 0x04030201u);
    vkb_image_stats(&is);
    CHECK(is.made == made);
    printf("a texture written as RGBA8 read as R32_UINT over the same storage: %#x, one image\n", word(&out1, 0));

    // The same storage as another shape (a 128 x 128 texture over the first quarter of the 256 x 256 one): another image
    // over the same range (VK_IMAGE_CREATE_ALIAS_BIT), placed at the same offset, no validation error.
    tex_t td3 = tex_at(WIN + 2 * MiB, NVN_TEXTURE_TARGET_2D, 0x25, 128, 128, 1, 1);
    bindless_texture_register(&r.pool, 13, &td3, NULL);
    v = submit_read(&r, r.pipe_2d, 13, BV_NONE, 3, 3, 4, 0, NULL, &out1);
    wait_done(v);
    vkb_image_stats(&is);
    CHECK(is.made == made + 1 && is.placed == made + 1);
    printf("the same storage as a smaller texture is a second image aliasing the first: %llu images\n", (ull)is.made);

    // A mip chain: the driver's layout of two levels takes more bytes than the game's own size for them on this device, so
    // the image cannot stand in the game's storage without overwriting what follows it; it has memory of its own.
    tex_t td4 = tex_at(WIN + 4 * MiB, NVN_TEXTURE_TARGET_2D, 0x25, 256, 256, 1, 2);
    uint64_t unaliased_before = is.unaliased;
    bindless_texture_register(&r.pool, 14, &td4, NULL);
    v = submit_read(&r, r.pipe_2d, 14, BV_NONE, 3, 3, 4, 0, NULL, &out1);
    wait_done(v);
    vkb_image_stats(&is);
    printf("a two-level texture: %s (%llu unaliased images)\n",
           is.unaliased > unaliased_before ? "memory of its own, the game's aliasing lost" : "placed in its pool's extent",
           (ull)is.unaliased);

    // 4,096 sampler registrations with a few hundred distinct states.
    for (int i = 0; i < 4096; i++) {
        samp_par s = nearest;
        s.lod_bias = (float)(i % 300) * 0.01f;
        bindless_sampler_register(&r.spool, i, &s);
    }
    bindless_stats(&s1);
    CHECK(s1.samplers_live == 300);
    v = submit_read(&r, r.pipe_sample, 5, BV_NONE, 100, 7, 8, 0, NULL, &out1);
    wait_done(v);
    CHECK(texel_is(&out1, 7, 8, 7, 255));
    printf("4096 sampler registrations with 300 distinct states: %llu host samplers live, and entry 100 samples\n",
           (ull)s1.samplers_live);

    // The pool is finalized: its entries show the placeholder and its images go.
    gfx_vulkan_backend.pool_fini(&pool);
    v = submit_read(&r, r.pipe_2d, 5, BV_NONE, 100, 7, 8, 0, NULL, &out1);
    wait_done(v);
    CHECK(texel_is(&out1, 0, 0, 0, 0));
    printf("after pool_fini the entries over its storage read the placeholder\n");
    for (int i = 0; i < 3; i++) {
        v = submit_read(&r, r.pipe_2d, 5, BV_NONE, 100, 7, 8, 0, NULL, &out1);
        wait_done(v);
    }
    vkb_image_stats(&is);
    printf("images live %llu destroyed %llu\n", (ull)is.live, (ull)is.destroyed);

    draw_buffer_free(&out1);
    draw_buffer_free(&out2);
    free(texels);
    free(back);
    vkDestroyPipeline(gpu_vk.device, r.pipe_2d, NULL);
    vkDestroyPipeline(gpu_vk.device, r.pipe_array, NULL);
    vkDestroyPipeline(gpu_vk.device, r.pipe_uint, NULL);
    vkDestroyPipeline(gpu_vk.device, r.pipe_sample, NULL);
    vkp_layouts_destroy(&lay);
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: bindless_test <case> [files]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "formats")) case_formats();
    else if (!strcmp(cs, "decoded")) case_decoded();
    else if (!strcmp(cs, "tables")) case_tables();
    else if (!strcmp(cs, "demand")) case_demand(), printf("ok demand\n");
    else if (!strcmp(cs, "id_fatal")) case_id_fatal();
    else if (!strcmp(cs, "usable")) case_usable();
    else if (!strcmp(cs, "device")) {
        if (volkInitialize() != VK_SUCCESS) {
            printf("no Vulkan loader\n");
            return EXIT_NO_DEVICE;
        }
        case_device(argc, argv);
    } else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
