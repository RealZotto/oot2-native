// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Present to a window on the Vulkan back end, on a real device and a real window (runtime/gpu/vk_present.c,
// runtime/gpu/vk_overlay.c, runtime/platform/window.c), driven by tests/test_runtime_vk_window.py:
// `vk_window_test <case> [--seconds N] [--mutant NAME]`, one case per process. It needs a Vulkan device, the SDL3 kit and a
// desktop, and ends with exit code 3 and says so without them (the Python side skips).
//
// A scene stands in for the game: three 2560x1440 textures in a GPU pool, cleared by recordings through the back end's own
// submit (a background and a bar that moves 8 pixels a tick), presented on the 60 Hz virtual clock the way the front end
// does it (present_pace, then present, one per tick, never waiting for the display).
//
//   fit        vkw_fit: the rectangle a picture is scaled into, for the shapes the window takes (no device)
//   pacing     --seconds of ticks (60 by default) on the display: every present has a shown tick, and the screen, sampled once
//              per tick as the present spike did, shows each frame once (the intervals, the repeats, the never shown, and how the
//              first-pixel-out times sit on the display's own vertical blanks, read by a thread of its own)
//   pixels     the picture on the desktop: a 2560x1440 texture, its left half red and its right half blue, in a window of
//              another shape: the bars are black, the picture keeps its aspect, the halves are where they should be
//   resize     the window made larger, smaller and taller than wide while frames go on: the swapchain follows each size, and
//              no frame is lost to a size that was not followed
//   mode       windowed to borderless (the screen's size) and back, while frames go on
//   minimise   the window minimised while frames go on: the game is never held, nothing is presented, and the picture comes
//              back when the window does
//   close      the close request reaches the program once and the teardown that follows is clean
//   notes      the capture's header on this back end: driver, display mode, layers, GPU timestamps, noted by the back end
//   exit       the close request ends the process from the window's thread while frames go on: exit code 0, no hang
//   menu       F1 opens the menu over the picture, a real click on a row changes the store, and a live row's hook changes
//              the pacing of the running game
//
// --mutant nofeedback (the shown time is never fed back), norecreate (the swapchain is not made again when the window's
// size changes) and skip10 (the host drops every tenth tick) are the counter-runs: the cases that check them must fail.
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/capture.h"
#include "oot2/cpu.h"
#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/memory.h"
#include "oot2/nvn_pace.h"
#include "oot2/nvn_tex.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/ui.h"
#include "oot2/vk_present.h"
#include "oot2/vk_rec.h"
#include "oot2/vsync.h"
#include "oot2/window.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define EXIT_NO_DEVICE 3
#define MiB (1ull << 20)
#define WIN GFX_GPU_WINDOW_BASE
#define OUT_W 2560
#define OUT_H 1440
typedef unsigned long long ull;

static Cpu g_cpu;
static uint64_t g_heap;
static pool_t g_gpool;

static void guest_heap(uint64_t size) {
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    g_cpu.x[0] = 0, g_cpu.x[1] = size;
    rt_svc(&g_cpu, 0x01, 0);   // SetHeapSize
    CHECK(g_cpu.x[0] == 0);
    g_heap = g_cpu.x[1];
}

// ---------- recordings
typedef struct {
    _Alignas(8) uint8_t b[1 << 14];
    size_t n;
} Rec;
static Rec g_recs[8];
static int g_next_rec;

static void put(Rec *r, uint32_t op, const void *payload, size_t bytes) {
    size_t size = (sizeof(op_hdr) + bytes + 7) & ~(size_t)7;
    CHECK(r->n + size <= sizeof r->b);
    op_hdr h = {op, (uint32_t)size};
    memset(r->b + r->n, 0, size);
    memcpy(r->b + r->n, &h, sizeof h);
    if (bytes) memcpy(r->b + r->n + sizeof h, payload, bytes);
    r->n += size;
}
#define PUT(r, op, ...) ({ __typeof__(__VA_ARGS__) p_ = __VA_ARGS__; put((r), (op), &p_, sizeof p_); })
static Rec *rec_begin(void) {
    Rec *r = &g_recs[g_next_rec++ % 8];
    r->n = 0;
    put(r, OP_BEGIN, NULL, 0);
    return r;
}
static uint64_t submit(Rec *r) {
    put(r, OP_END, NULL, 0);
    uint64_t h = (uint64_t)(uintptr_t)r->b;
    uint64_t v = gfx_vulkan_backend.submit(&h, 1);
    CHECK(v);
    return v;
}

static tex_t tex_at(uint64_t storage, int format, uint32_t w, uint32_t h) {
    tex_t t;
    memset(&t, 0, sizeof t);
    t.magic = NVN_TEX_MAGIC;
    t.d.target = NVN_TEXTURE_TARGET_2D, t.d.format = format, t.d.width = w, t.d.height = h, t.d.depth = 1, t.d.levels = 1;
    t.d.swizzle_r = NVN_TEXTURE_SWIZZLE_R, t.d.swizzle_g = NVN_TEXTURE_SWIZZLE_G, t.d.swizzle_b = NVN_TEXTURE_SWIZZLE_B;
    t.d.swizzle_a = NVN_TEXTURE_SWIZZLE_A;
    t.mem = (uint8_t *)(uintptr_t)storage;
    static uint64_t serial;
    t.serial = ++serial;
    nvn_tex_layout(&t.d, &t.L);
    return t;
}
static nvn_tex_rec trec(const tex_t *t) {
    nvn_tex_rec r;
    memset(&r, 0, sizeof r);
    r.tex = *t;
    r.present = 1;
    return r;
}
static NVNcopyRegion region(int x, int y, int w, int h) { return (NVNcopyRegion){x, y, 0, w, h, 1}; }
static void clear_tex(Rec *r, const tex_t *t, NVNcopyRegion rg, const float c[4]) {
    p_clear_tex p = {.tex = trec(t), .r = rg, .mask = 0xF};
    memcpy(p.c.f, c, 16);
    PUT(r, OP_CLEAR_TEX, p);
}

// ---------- the scene and the window
static tex_t g_tex[3];
static const VkwUi g_ui = {ui_menu_frame, ui_menu_draw_data, ui_menu_set_texture_ops, ui_menu_prepare};
static _Atomic int g_closed, g_resized;

static int test_button(int button, int down) { return ui_menu_input((UiButton)button, down); }
static void test_pointer(float x, float y, int down) { ui_menu_pointer(x, y, down); }
static void test_resized(int w, int h) { (void)w, (void)h, atomic_fetch_add(&g_resized, 1); }
static void test_closed(void) { atomic_fetch_add(&g_closed, 1); }

static void setup(int w, int h, int with_ui) {
    WinConfig cfg = {.title = "oot2 window test", .width = w, .height = h, .mode = WIN_WINDOWED};
    if (!win_open(&cfg)) {
        printf("no window: %s\n", win_why());
        exit(EXIT_NO_DEVICE);
    }
    win_set_input(&(WinInput){.button = test_button, .menu_open = ui_menu_is_open, .pointer = test_pointer,
                              .resized = test_resized, .closed = test_closed});
    vkw_window(&win_vulkan, with_ui ? &g_ui : NULL);
    guest_heap(64 * MiB);
    gpu_vk_open();
    gfx_set_backend(&gfx_vulkan_backend);   // as gfx_select does: the front end's records and the capture ask this table
    VkRecSubmit sub;
    vkrec_submit_device(&sub);
    static VkRecSubmit g_sub;
    g_sub = sub;
    VkRecOpen o = {.submit = &g_sub, .frames = 4};
    vkrec_open(&o);
    pool_t p = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS,
                .mem = (uint8_t *)(uintptr_t)WIN, .size = 64 * MiB, .game_mem = (uint8_t *)(uintptr_t)WIN};
    CHECK(gfx_vulkan_backend.pool_init(&p));
    g_gpool = p;
    for (int k = 0; k < 3; k++) g_tex[k] = tex_at(WIN + 16 * MiB * (uint64_t)k, NVN_FORMAT_RGBA8, OUT_W, OUT_H);
    CHECK(vkw_active());
}

static void finish_and_report(const char *name) {
    gfx_vulkan_backend.finish();
    gfx_vulkan_backend.pool_fini(&g_gpool);
    gpu_vk_close();
    win_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    printf("validation errors %llu warnings %llu\n", (ull)errors, (ull)warnings);
    CHECK(errors == 0);
    printf("ok %s\n", name);
}

// Frame n of the scene into texture n % 3: a background, and a bar that moves 8 pixels a tick.
static int g_fixed;   // a case that painted its own picture: the scene's frames leave the textures alone
static void draw_frame(uint64_t n) {
    if (g_fixed) return;
    const float back[4] = {0.06f, 0.06f, 0.06f, 1.0f}, bar[4] = {0.8f, 0.8f, 0.8f, 1.0f};
    Rec *r = rec_begin();
    const tex_t *t = &g_tex[n % 3];
    clear_tex(r, t, region(0, 0, 0, 0), back);
    clear_tex(r, t, region((int)((n * 8) % (OUT_W - 96)), 0, 96, OUT_H), bar);
    submit(r);
}

static uint64_t g_seq;   // the present's number, from 1, across the case

// One present the way the front end makes it, on the tick the clock is in now: pace, then present, and no wait.
static uint64_t present_frame(uint64_t tick) {
    uint64_t seq = ++g_seq;
    draw_frame(seq);
    gfx_vulkan_backend.present_pace(seq, tick, rt_vsync_due_counter(tick));
    const int crop[4] = {0, 0, 0, 0};
    gfx_vulkan_backend.present(&g_tex[seq % 3], crop);
    return seq;
}

// count presents, one a tick from the next tick on
static void run_ticks(int count) {
    uint64_t tick = rt_vsync_tick() + 1;
    for (int i = 0; i < count; i++, tick++) {
        rt_vsync_wait_tick(tick);
        present_frame(tick);
    }
}

static uint64_t ns_now(void) { return plat_now_ns(); }

// ---------- the display's vertical blanks, read by a thread of its own (the present spike's second instrument)
#define MAX_VBLANKS 60000
static uint64_t g_vblank[MAX_VBLANKS];
static _Atomic int g_nvb, g_vb_stop, g_vb_error;
#ifdef _WIN32
typedef struct { HDC hdc; UINT adapter; LUID luid; UINT source; } OpenFromHdc;
typedef struct { UINT adapter; UINT device; UINT source; } WaitVblank;
typedef struct { UINT adapter; } CloseAdapter;
typedef LONG(APIENTRY *PFN_Open)(OpenFromHdc *);
typedef LONG(APIENTRY *PFN_Wait)(const WaitVblank *);
typedef LONG(APIENTRY *PFN_Close)(const CloseAdapter *);

static DWORD WINAPI vblank_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    HMODULE gdi = LoadLibraryA("gdi32.dll");
    typedef HDC(WINAPI * PFN_CreateDC)(LPCWSTR, LPCWSTR, LPCWSTR, const DEVMODEW *);
    typedef BOOL(WINAPI * PFN_DeleteDC)(HDC);
    PFN_CreateDC create_dc = gdi ? (PFN_CreateDC)(void (*)(void))GetProcAddress(gdi, "CreateDCW") : NULL;
    PFN_DeleteDC delete_dc = gdi ? (PFN_DeleteDC)(void (*)(void))GetProcAddress(gdi, "DeleteDC") : NULL;
    PFN_Open open_ad = gdi ? (PFN_Open)(void (*)(void))GetProcAddress(gdi, "D3DKMTOpenAdapterFromHdc") : NULL;
    PFN_Wait wait_vb = gdi ? (PFN_Wait)(void (*)(void))GetProcAddress(gdi, "D3DKMTWaitForVerticalBlankEvent") : NULL;
    PFN_Close close_ad = gdi ? (PFN_Close)(void (*)(void))GetProcAddress(gdi, "D3DKMTCloseAdapter") : NULL;
    if (!create_dc || !open_ad || !wait_vb) { atomic_store(&g_vb_error, 1); return 0; }
    HDC hdc = create_dc(L"DISPLAY", NULL, NULL, NULL);   // the primary display
    OpenFromHdc o = {hdc, 0, {0, 0}, 0};
    if (open_ad(&o)) { atomic_store(&g_vb_error, 1); return 0; }
    WaitVblank w = {o.adapter, 0, o.source};
    while (!atomic_load(&g_vb_stop)) {
        if (wait_vb(&w)) { atomic_store(&g_vb_error, 1); break; }
        uint64_t t = ns_now();
        int i = atomic_load(&g_nvb);
        if (i < MAX_VBLANKS) { g_vblank[i] = t; atomic_fetch_add(&g_nvb, 1); }
    }
    CloseAdapter c = {o.adapter};
    if (close_ad) close_ad(&c);
    if (delete_dc) delete_dc(hdc);
    return 0;
}
static HANDLE g_vb_thread;
static void vblank_start(void) { g_vb_thread = CreateThread(NULL, 0, vblank_main, NULL, 0, NULL); }
static void vblank_stop(void) {
    atomic_store(&g_vb_stop, 1);
    if (g_vb_thread) WaitForSingleObject(g_vb_thread, 2000);
}
#else
static void vblank_start(void) { atomic_store(&g_vb_error, 1); }
static void vblank_stop(void) {}
#endif

// ---------- cases
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static double pct(uint64_t *v, int n, double p) {
    if (n <= 0) return 0;
    qsort(v, (size_t)n, sizeof *v, cmp_u64);
    return (double)v[(int)(p * (n - 1))] / 1e6;
}

static void case_fit(void) {
    struct { uint32_t sw, sh, dw, dh; int x, y, w, h; } k[] = {
        {2560, 1440, 1280, 720, 0, 0, 1280, 720},     // an exact half
        {2560, 1440, 1920, 1080, 0, 0, 1920, 1080},   // three quarters
        {2560, 1440, 1000, 1000, 0, 218, 1000, 563},  // narrower than the picture: bars above and below
        {2560, 1440, 1600, 600, 266, 0, 1067, 600},   // wider: bars at the sides
        {2560, 1440, 320, 180, 0, 0, 320, 180},       // the smallest window
        {2560, 1440, 1, 1, 0, 0, 1, 1},               // never empty
        {64, 16, 100, 100, 0, 37, 100, 25},
    };
    for (size_t i = 0; i < sizeof k / sizeof *k; i++) {
        VkwRect r = vkw_fit(k[i].sw, k[i].sh, k[i].dw, k[i].dh);
        printf("fit %ux%u into %ux%u: %d,%d %dx%d\n", k[i].sw, k[i].sh, k[i].dw, k[i].dh, r.x, r.y, r.w, r.h);
        CHECK(r.x == k[i].x && r.y == k[i].y && r.w == k[i].w && r.h == k[i].h);
        CHECK(r.x + r.w <= (int)k[i].dw && r.y + r.h <= (int)k[i].dh);
    }
    printf("ok fit\n");
}

static int g_skip;   // --mutant skip10: every tenth tick is not presented

static void case_pacing(int seconds) {
    setup(1280, 720, 1);
    vblank_start();
    plat_sleep_ns(300 * 1000 * 1000);
    run_ticks(120);   // two seconds to settle: the swapchain, the driver's first frames
    plat_sleep_ns(200 * 1000 * 1000);
    int ticks = seconds * 60;
    uint64_t *due = calloc((size_t)ticks, sizeof *due), *tick_of = calloc((size_t)ticks, sizeof *tick_of);
    uint64_t *seqs = calloc((size_t)ticks, sizeof *seqs), *ret = calloc((size_t)ticks, sizeof *ret);
    CHECK(due && tick_of && seqs && ret);
    // the shown time of each present, on the host clock (first pixel out) and as the tick the back end answers; read a few
    // frames after the present, as the capture reads it (the ring behind present_shown keeps the newest 512 presents)
    uint64_t *fpo = calloc((size_t)ticks, sizeof *fpo), *shown_t = calloc((size_t)ticks, sizeof *shown_t);
    CHECK(fpo && shown_t);
    VkwStats before;
    vkw_stats(&before);
    uint64_t tick = rt_vsync_tick() + 2;
    uint64_t t0 = ns_now();
    for (int i = 0; i < ticks; i++, tick++) {
        rt_vsync_wait_tick(tick);
        due[i] = rt_vsync_due_ns(tick);
        tick_of[i] = tick;
        if (i >= 16 && !fpo[i - 16]) {
            fpo[i - 16] = vkw_shown_ns(seqs[i - 16]);
            shown_t[i - 16] = gfx_vulkan_backend.present_shown(seqs[i - 16]);
        }
        if (g_skip && i % 10 == 9) continue;   // the counter-run: the host drops every tenth tick, as the present spike's did
        seqs[i] = present_frame(tick);
        ret[i] = ns_now();
    }
    uint64_t t1 = ns_now();
    plat_sleep_ns(500 * 1000 * 1000);   // the last results
    vblank_stop();
    VkwStats st;
    vkw_stats(&st);
    printf("submit_and_present_ms (the call that holds the submission gate): mean %.3f max %.3f over %llu\n",
           st.end_count ? (double)st.end_ns_sum / (double)st.end_count / 1e6 : 0.0, (double)st.end_ns_max / 1e6,
           (ull)st.end_count);
    int have = 0, never = 0, tick_ok = 0;
    for (int i = 0; i < ticks; i++) {
        if (!fpo[i]) {
            fpo[i] = vkw_shown_ns(seqs[i]);
            shown_t[i] = gfx_vulkan_backend.present_shown(seqs[i]);
        }
        if (fpo[i]) have++; else never++;
        if (!fpo[i] && seqs[i]) printf("no shown time for the present of tick %d (seq %llu)\n", i, (ull)seqs[i]);
        if (shown_t[i] >= tick_of[i] && shown_t[i] <= tick_of[i] + 2) tick_ok++;
    }
    printf("run %.3f s for %d ticks; presents +%llu shown +%llu dropped_ring +%llu dropped_failed +%llu repeats +%llu "
           "timing %d\n", (double)(t1 - t0) / 1e9, ticks, (ull)(st.presents - before.presents),
           (ull)(st.shown - before.shown), (ull)(st.dropped_ring - before.dropped_ring),
           (ull)(st.dropped_failed - before.dropped_failed), (ull)(st.repeats - before.repeats), st.timing);
    printf("shown_time have %d never %d shown_tick_within_2_ticks %d of %d\n", have, never, tick_ok, ticks);
    // intervals between consecutive pictures on the screen
    int b10 = 0, b20 = 0, other = 0, same = 0, pairs = 0;
    uint64_t *lat = calloc((size_t)ticks, sizeof *lat);
    int nlat = 0;
    int prev_shown = -1;
    for (int i = 0; i < ticks; i++) {   // between consecutive pictures that were shown: a tick the host never presented gives 30 or 40 ms
        if (!fpo[i]) continue;
        if (prev_shown < 0) {
            prev_shown = i;
            continue;
        }
        int64_t d = (int64_t)(fpo[i] - fpo[prev_shown]);
        prev_shown = i;
        pairs++;
        if (d > 8500000 && d < 11500000) b10++;
        else if (d > 18500000 && d < 21500000) b20++;
        else if (d < 5000000) same++;
        else other++;
    }
    for (int i = 0; i < ticks; i++)
        if (fpo[i] > due[i]) lat[nlat++] = fpo[i] - due[i];
    printf("intervals %d: 10ms %d 20ms %d other %d two_on_one_refresh %d\n", pairs, b10, b20, other, same);
    double l1 = pct(lat, nlat, 0.01), l50 = pct(lat, nlat, 0.50), l99 = pct(lat, nlat, 0.99), lmax = pct(lat, nlat, 1.0);
    printf("due_to_screen_ms p1 %.2f p50 %.2f p99 %.2f max %.2f\n", l1, l50, l99, lmax);
    // the clock's view: the screen sampled once per tick at due + Lmin + h, where an on-time frame is on screen
    double h = 12.5;
    int repeats = 0, last = -1;
    char *seen = calloc((size_t)ticks, 1);
    for (int i = 0; i < ticks; i++) {
        uint64_t sample = due[i] + (uint64_t)((l1 + h) * 1e6);
        int m = -1;
        for (int k = i > 8 ? i - 8 : 0; k < ticks && k < i + 8; k++)
            if (fpo[k] && fpo[k] <= sample && (m < 0 || fpo[k] >= fpo[m])) m = k;
        if (m >= 0 && m == last) repeats++;
        if (m >= 0) seen[m] = 1;
        last = m;
    }
    int missing = 0;
    for (int i = 8; i + 8 < ticks; i++) missing += !seen[i];
    printf("clock_view repeats %d never_shown %d of %d\n", repeats, missing, ticks - 16);
    // the first-pixel-out times against the display's own vertical blanks
    int nvb = atomic_load(&g_nvb);
    if (!atomic_load(&g_vb_error) && nvb > 100) {
        uint64_t *dist = calloc((size_t)ticks, sizeof *dist);
        int nd = 0;
        for (int i = 0; i < ticks; i++) {
            if (!fpo[i]) continue;
            uint64_t best = UINT64_MAX;
            for (int k = 0; k < nvb; k++) {
                uint64_t d = fpo[i] > g_vblank[k] ? fpo[i] - g_vblank[k] : g_vblank[k] - fpo[i];
                if (d < best) best = d;
            }
            dist[nd++] = best;
        }
        printf("fpo_to_nearest_vblank_ms p50 %.3f p99 %.3f max %.3f over %d vblanks\n", pct(dist, nd, 0.5), pct(dist, nd, 0.99),
               pct(dist, nd, 1.0), nvb);
        free(dist);
    } else {
        printf("vblank thread unavailable\n");
    }
    CHECK(have == ticks - (g_skip ? ticks / 10 : 0) && tick_ok == have);   // every present has its shown time, in a tick within two of its due tick
    CHECK(st.presents - before.presents == (uint64_t)(ticks - (g_skip ? ticks / 10 : 0)));
    CHECK(!g_skip || repeats > 0);   // the clock's view must see the dropped ticks, or the instrument is blind
    finish_and_report("pacing");
}

#ifdef _WIN32
// The client area as the desktop shows it: pixels BGRA, w x h; 0 when it could not be read.
static uint8_t *screen_of(int x, int y, int w, int h) {
    HDC sc = GetDC(NULL);
    HDC mem = CreateCompatibleDC(sc);
    HBITMAP bmp = CreateCompatibleBitmap(sc, w, h);
    HGDIOBJ old = SelectObject(mem, bmp);
    uint8_t *px = NULL;
    if (BitBlt(mem, 0, 0, w, h, sc, x, y, SRCCOPY | CAPTUREBLT)) {
        BITMAPINFO bi = {.bmiHeader = {sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB, 0, 0, 0, 0, 0}};
        px = malloc((size_t)w * h * 4);
        if (px && !GetDIBits(mem, bmp, 0, (UINT)h, px, &bi, DIB_RGB_COLORS)) free(px), px = NULL;
    }
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteDC(mem);
    ReleaseDC(NULL, sc);
    return px;
}
static void save_ppm(const char *path, const uint8_t *bgra, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) fputc(bgra[4 * i + 2], f), fputc(bgra[4 * i + 1], f), fputc(bgra[4 * i], f);
    fclose(f);
}
#endif

// Present until the swapchain is the size asked for and the newest picture is on the desktop (a few frames); 1 when it is.
static int settle_at(int w, int h) {
    for (int i = 0; i < 90; i++) {
        run_ticks(3);
        VkwStats st;
        vkw_stats(&st);
        if (st.width == (uint32_t)w && st.height == (uint32_t)h && st.shown > 5) {
            plat_sleep_ns(150 * 1000 * 1000);
            run_ticks(6);
            plat_sleep_ns(150 * 1000 * 1000);
            return 1;
        }
    }
    return 0;
}

static void case_pixels(void) {
#ifdef _WIN32
    const int W = 1000, H = 1000;
    setup(W, H, 0);
    win_post_topmost(1);   // nothing over it while the screen is read
    // left half red, right half blue
    const float red[4] = {1, 0, 0, 1}, blue[4] = {0, 0, 1, 1};
    for (int k = 0; k < 3; k++) {
        Rec *r = rec_begin();
        clear_tex(r, &g_tex[k], region(0, 0, OUT_W / 2, OUT_H), red);
        clear_tex(r, &g_tex[k], region(OUT_W / 2, 0, OUT_W / 2, OUT_H), blue);
        submit(r);
    }
    gfx_vulkan_backend.finish();
    g_fixed = 1;   // the frames that follow present these textures as they are
    CHECK(settle_at(W, H));
    int ox, oy;
    CHECK(win_client_origin(&ox, &oy));
    uint8_t *px = screen_of(ox, oy, W, H);
    CHECK(px);
    // the picture's rectangle: 1000 x 563 at y = 218 (vkw_fit)
#define AT(x, y) (px + 4 * ((size_t)(y) * W + (x)))
    struct { int x, y; const char *what; int b, g, r; } at[] = {
        {500, 100, "bar above", 0, 0, 0},     {500, 900, "bar below", 0, 0, 0},
        {250, 500, "left half", 0, 0, 255},   {750, 500, "right half", 255, 0, 0},
        {20, 230, "picture's top left", 0, 0, 255}, {980, 770, "picture's bottom right", 255, 0, 0},
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof at / sizeof *at; i++) {
        const uint8_t *p = AT(at[i].x, at[i].y);
        int ok = abs(p[0] - at[i].b) < 12 && abs(p[1] - at[i].g) < 12 && abs(p[2] - at[i].r) < 12;
        printf("pixel %s (%d,%d): b%d g%d r%d, wanted b%d g%d r%d: %s\n", at[i].what, at[i].x, at[i].y, p[0], p[1], p[2],
               at[i].b, at[i].g, at[i].r, ok ? "ok" : "WRONG");
        bad += !ok;
    }
    // the edge between the picture and the bars is where vkw_fit puts it: black at y 214, picture at y 222
    const uint8_t *above = AT(500, 214), *inside = AT(250, 222);
    int edge = above[0] + above[1] + above[2] < 30 && inside[2] > 200;
    printf("edge between the bar and the picture at y 218: %s\n", edge ? "ok" : "WRONG");
    // the picture is kept only when it is the window's own (nothing of the desktop's other windows)
    const char *save = getenv("OOT2_WINDOW_TEST_SAVE");
    if (save && !bad && edge) save_ppm(save, px, W, H);
    free(px);
    CHECK(!bad && edge);
    finish_and_report("pixels");
#else
    printf("no desktop capture on this platform\n");
    exit(EXIT_NO_DEVICE);
#endif
}

static void case_resize(void) {
    setup(800, 450, 0);
    CHECK(settle_at(800, 450));
    VkwStats st0;
    vkw_stats(&st0);
    printf("start: swapchain %ux%u, %llu made\n", st0.width, st0.height, (ull)st0.swapchains);
    const int sizes[][2] = {{1280, 720}, {640, 360}, {900, 1200}, {1600, 900}};
    uint64_t made = st0.swapchains;
    for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) {
        win_request(WIN_WINDOWED, sizes[i][0], sizes[i][1]);
        int ok = settle_at(sizes[i][0], sizes[i][1]);
        VkwStats st;
        vkw_stats(&st);
        WinState ws;
        win_state(&ws);
        printf("window %dx%d: swapchain %ux%u (window says %dx%d), %llu made, out_of_date %llu, dropped_failed %llu: %s\n",
               sizes[i][0], sizes[i][1], st.width, st.height, ws.width, ws.height, (ull)st.swapchains, (ull)st.out_of_date,
               (ull)st.dropped_failed, ok ? "followed" : "NOT FOLLOWED");
        CHECK(ok);
        CHECK(st.swapchains > made);
        made = st.swapchains;
    }
    VkwStats st;
    vkw_stats(&st);
    printf("presents %llu shown %llu dropped_ring %llu dropped_window %llu dropped_failed %llu out_of_date %llu\n", (ull)st.presents,
           (ull)st.shown, (ull)st.dropped_ring, (ull)st.dropped_window, (ull)st.dropped_failed, (ull)st.out_of_date);
    CHECK(st.shown > 40);
    finish_and_report("resize");
}

// Windowed to borderless (the screen's size, no mode change) and back, while frames go on.
static void case_mode(void) {
    setup(800, 450, 0);
    CHECK(settle_at(800, 450));
    win_request(WIN_BORDERLESS, 800, 450);
    WinState ws = {0};
    for (int i = 0; i < 100 && ws.mode != WIN_BORDERLESS; i++) run_ticks(2), win_state(&ws);
    CHECK(ws.mode == WIN_BORDERLESS && ws.width > 800 && ws.height > 450);
    printf("borderless: the window is %dx%d (the screen's size)\n", ws.width, ws.height);
    CHECK(settle_at(ws.width, ws.height));
    VkwStats st;
    vkw_stats(&st);
    printf("swapchain %ux%u, shown %llu\n", st.width, st.height, (ull)st.shown);
    win_request(WIN_WINDOWED, 800, 450);
    CHECK(settle_at(800, 450));
    win_state(&ws);
    CHECK(ws.mode == WIN_WINDOWED && ws.width == 800 && ws.height == 450);
    printf("back to windowed 800x450\n");
    finish_and_report("mode");
}

static void case_minimise(void) {
    setup(800, 450, 0);
    CHECK(settle_at(800, 450));
    VkwStats a;
    vkw_stats(&a);
    uint64_t t0 = ns_now();
    win_post_minimise(1);
    run_ticks(60);   // one second of ticks with the window minimised
    double took = (double)(ns_now() - t0) / 1e9;
    VkwStats b;
    vkw_stats(&b);
    WinState ws;
    win_state(&ws);
    printf("minimised: %d, 60 ticks took %.3f s; presents +%llu, shown +%llu, dropped_window +%llu\n", ws.minimised, took,
           (ull)(b.presents - a.presents), (ull)(b.shown - a.shown), (ull)(b.dropped_window - a.dropped_window));
    CHECK(ws.minimised && ws.width == 0);
    CHECK(took < 1.35);   // the game is paced by the clock, never held by a window with no area
    CHECK(b.presents - a.presents == 60);
    CHECK(b.dropped_window - a.dropped_window >= 50);
    CHECK(b.shown - a.shown <= 6);
    win_post_minimise(0);
    CHECK(settle_at(800, 450));
    VkwStats c;
    vkw_stats(&c);
    printf("restored: shown +%llu since the restore\n", (ull)(c.shown - b.shown));
    CHECK(c.shown - b.shown >= 6);
    finish_and_report("minimise");
}

static void case_close(void) {
    setup(800, 450, 0);
    CHECK(settle_at(800, 450));
    CHECK(atomic_load(&g_closed) == 0);
    win_post_close();
    for (int i = 0; i < 100 && !atomic_load(&g_closed); i++) run_ticks(1), plat_sleep_ns(10 * 1000 * 1000);
    WinState ws;
    win_state(&ws);
    printf("close requests %d, window says %d\n", atomic_load(&g_closed), ws.close_requested);
    CHECK(atomic_load(&g_closed) == 1 && ws.close_requested);
    run_ticks(10);   // the game goes on until the program ends the run
    finish_and_report("close");
}

// The capture's header on this back end: the driver, the display's mode, the layers and the GPU timestamps are noted by
// the back end (the checker of the capture requires them), with the folder the capture goes to in OOT2_WINDOW_TEST_CAPTURE.
static void case_notes(void) {
    const char *folder = getenv("OOT2_WINDOW_TEST_CAPTURE");
    CHECK(folder && getenv("OOT2_CAPTURE"));
    setup(800, 450, 0);
    capture_start(folder);
    CHECK(settle_at(800, 450));
    run_ticks(20);
    capture_end(0);
    finish_and_report("notes");
}

static void test_closed_exit(void) {
    printf("close: the program ends the run\n");
    plat_exit(0);   // what the program does through rt_census_end: the stop hooks, then the process's end
}

// The window thread ends the process while the game is still presenting: no hang, no fault, exit code 0.
static void case_exit(void) {
    setup(800, 450, 0);
    CHECK(settle_at(800, 450));
    win_set_input(&(WinInput){.button = test_button, .menu_open = ui_menu_is_open, .pointer = test_pointer,
                              .resized = test_resized, .closed = test_closed_exit});
    win_post_close();
    for (;;) run_ticks(1);
}

static int row_index(const char *key) {
    for (int i = 0, n = ui_menu_row_count(); i < n; i++) {
        UiRow row;
        if (ui_menu_row(i, &row) && row.key && !strcmp(row.key, key)) return i;
    }
    return -1;
}

static void case_menu(void) {
    setup(1280, 720, 1);
    ui_menu_set_scripted(0);
    CHECK(settle_at(1280, 720));
    CHECK(!ui_menu_is_open());
    VkwStats a;
    vkw_stats(&a);
    win_post_button(0, 1);   // F1
    win_post_button(0, 0);
    run_ticks(12);
    CHECK(ui_menu_is_open());
    VkwStats b;
    vkw_stats(&b);
    printf("F1: the menu is open, overlay frames +%llu in 12 presents\n", (ull)(b.overlay_frames - a.overlay_frames));
    CHECK(b.overlay_frames - a.overlay_frames >= 8);
    // a real click on a row's checkbox: the overlay row (live, a switch of the menu itself)
    int idx = row_index("overlay");
    CHECK(idx >= 0);
    float rect[4];
    CHECK(ui_menu_row_rect(idx, rect));
    float cx = rect[0] + rect[2] / 2, cy = rect[1] + rect[3] / 2;
    CHECK(rt_setting(RT_SET_overlay) == 0);
    win_post_pointer(cx, cy, 0);
    run_ticks(3);
    win_post_pointer(cx, cy, 1);
    run_ticks(3);
    win_post_pointer(cx, cy, 0);
    run_ticks(3);
    printf("click on the overlay row at %.0f,%.0f: overlay is %lld\n", cx, cy, (long long)rt_setting(RT_SET_overlay));
    CHECK(rt_setting(RT_SET_overlay) == 1);
#ifdef _WIN32
    if (getenv("OOT2_WINDOW_TEST_SAVE_MENU")) {   // the menu over the picture, as the desktop shows it (window in front)
        win_post_topmost(1);
        run_ticks(30);
        plat_sleep_ns(200 * 1000 * 1000);
        int ox, oy;
        uint8_t *px = win_client_origin(&ox, &oy) ? screen_of(ox, oy, 1280, 720) : NULL;
        if (px) save_ppm(getenv("OOT2_WINDOW_TEST_SAVE_MENU"), px, 1280, 720);
        free(px);
    }
#endif
    // a live row of the pacing group, through the row call a click makes: the front end's hold follows at once
    nvn_settings_bind();
    int hold = row_index("hold_ticks");
    CHECK(hold >= 0);
    char why[200];
    CHECK(nvn_present_hold_ticks() == 0);
    CHECK(ui_menu_set(hold, "2", why, sizeof why));
    printf("hold_ticks row set to 2: the front end holds %d ticks\n", nvn_present_hold_ticks());
    CHECK(nvn_present_hold_ticks() == 2);
    win_post_button(0, 1);   // F1 again: closes
    win_post_button(0, 0);
    run_ticks(6);
    CHECK(!ui_menu_is_open());
    finish_and_report("menu");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_window_test <case> [--seconds N] [--mutant NAME]\n");
        return 2;
    }
    int seconds = 60;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mutant") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "nofeedback")) vkw_no_feedback = 1;
            else if (!strcmp(m, "norecreate")) vkw_no_recreate = 1;
            else if (!strcmp(m, "skip10")) g_skip = 1;
            else {
                fprintf(stderr, "unknown mutant %s\n", m);
                return 2;
            }
        } else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "fit")) {
        case_fit();
        return 0;
    }
    if (volkInitialize() != VK_SUCCESS) {
        printf("no Vulkan loader\n");
        return EXIT_NO_DEVICE;
    }
    if (!strcmp(cs, "pacing")) case_pacing(seconds);
    else if (!strcmp(cs, "pixels")) case_pixels();
    else if (!strcmp(cs, "resize")) case_resize();
    else if (!strcmp(cs, "minimise")) case_minimise();
    else if (!strcmp(cs, "mode")) case_mode();
    else if (!strcmp(cs, "close")) case_close();
    else if (!strcmp(cs, "menu")) case_menu();
    else if (!strcmp(cs, "exit")) case_exit();
    else if (!strcmp(cs, "notes")) case_notes();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
