// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// oot2.exe's start-up (architecture D4, D6, D8, D9). Everything it knows of the player's dump comes from the program
// glue the build wrote (oot2/program.h): the module files and their SHA-256, the other files it checks, the layout,
// the lifted function tables and the values of main.npdm.
//
// In order:
//   0. the game folder: --game DIR, else the folder of the executable; the glue's relative paths start there, and the
//      Vulkan back end's device-loss report goes to its logs folder
//   1. the settings (oot2/settings.h): --set KEY=VALUE (and --gfx NAME, the setting gfx), the OOT2_ variables, then the
//      file: --config PATH, else the OOT2_CONFIG variable, else oot2.ini in the game folder. They are read before the
//      first log line, because a log line reads census mode, which is a setting. An unknown key, or a value its key
//      does not take, stops the start with exit code 2. Then the system timer resolution (rt_timer_hold_start, the timer
//      resolution hold): 0.5 ms held until the process ends, unless timer_resolution is 0, before any thread starts. Then the
//      graphics back end the setting gfx names (null by default), installed by gfx_select (runtime/gpu/gpu.c) long
//      before rt_dispatch_init, with the name of the layer that chose it; the Vulkan back end opens its device there, or
//      stops the start by name (its settings:
//      oot2/gpu_vk.h)
//   2. rt_mem_init, with main.npdm's system resource size (the pool is still the planning figure, memory.h)
//   3. every file of the glue checked: its size, and its SHA-256 where the glue has one, or the SHA-256 of fixed ranges
//      for the player's dump when the RomFS is read from it in place; a file that changed since the build stops the
//      start, as a module would (after rt_mem_init, so mapping a large file cannot take a region the guest needs)
//   4. rt_kernel_init with the configuration of main.npdm (rt_npdm_parse): the program's cores, priorities, main
//      thread, handle table and SVC mask
//   5. rt_load_modules: each module hashed again against the glue, relocated and bound
//   6. host registrations: the NVN2 entry points (nvn_procs_register), then the Horizon layer (hos_init: the service
//      SVCs, the boot driver, the census), then the hooks of the live settings (nvn_settings_bind) and the settings
//      file's watcher thread, which applies an edit of the file while the game runs
//   7. rt_dispatch_init over the lifted modules' tables
//   8. rt_kernel_run: the main thread starts in the boot driver, which calls the SDK's nn::init::Start as rtld does on
//      the console (no module's first word is code, and rtld is not loaded); it returns when every guest thread ended
//
// A run that stops writes census.tsv into the game folder (runtime/hos/census.c), as does a run that ends. A run that
// ends also writes ledger.jsonl there (oot2/ledger.h): the programs, answers and attachment formats the game used, which
// the next prepare reads; the end of the run is the safe point, never the frame path.
// With OOT2_CAPTURE=1 a run also writes the M1 capture there (oot2/capture.h, judged by tools/check_capture.py), and
// with OOT2_CAPTURE_SECONDS it ends by itself, exit code 0, that long after the capture's title mark.
//
// Usage: oot2 [--game DIR] [--gfx NAME] [--set KEY=VALUE]... [--config PATH] [--vulkan-items]. --vulkan-items prints
// the Vulkan device items the renderer requires (oot2/vk_required.h, the list the check step of oot2 prepare reports
// from), one "id<TAB>what" line each, and exits. Exit codes: runtime.h (70 fatal, 71 unimplemented, 72 guest stop), 2
// bad command line or settings.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "oot2/boot.h"
#include "oot2/capture.h"
#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/kernel.h"
#include "oot2/ledger.h"
#include "oot2/pagegen.h"
#include "oot2/program.h"
#include "oot2/settings.h"
#include "oot2/texheap.h"
#include "oot2/ui.h"
#include "oot2/vk_present.h"
#include "oot2/vk_required.h"
#include "oot2/window.h"
#include "plat.h"

#define PATH_BYTES 4096
#define MAX_SETS 64   // --set and --gfx options

static char g_game[PATH_BYTES];

static double ms_since(uint64_t t0) { return (double)(plat_now_ns() - t0) / 1e6; }

static int is_absolute(const char *p) { return p[0] == '/' || p[0] == '\\' || (p[0] && p[1] == ':'); }

// The folder of this executable, UTF-8, without the trailing separator. 0 when it cannot be read.
static int exe_folder(char *out, size_t size) {
#ifdef _WIN32
    wchar_t w[PATH_BYTES];
    DWORD n = GetModuleFileNameW(NULL, w, PATH_BYTES);
    if (!n || n >= PATH_BYTES) return 0;
    if (!WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)size, NULL, NULL)) return 0;
#else
    ssize_t n = readlink("/proc/self/exe", out, size - 1);
    if (n <= 0) return 0;
    out[n] = 0;
#endif
    char *cut = strrchr(out, '/'), *back = strrchr(out, '\\');
    if (back > cut) cut = back;
    if (!cut) return 0;
    *cut = 0;
    return 1;
}

// path as the program opens it: in the game folder unless the glue gave an absolute path. Returns a copy in buf.
static const char *game_path(const char *path, char *buf, size_t size) {
    int n = is_absolute(path) ? snprintf(buf, size, "%s", path) : snprintf(buf, size, "%s/%s", g_game, path);
    if (n < 0 || (size_t)n >= size) rt_fatal("the path of %s in %s is too long", path, g_game);
    return buf;
}

static void hex(const uint8_t d[32], char out[65]) {
    for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

// The SHA-256 of the ranges tools/build_game.py sample_ranges picks: SAMPLE_COUNT blocks of SAMPLE_BYTES (the whole
// file when it is smaller), the first at the start, the last ending at the end, the others evenly between. Block i
// starts at span * i / (SAMPLE_COUNT - 1), computed without overflow as q * i + r * i / (SAMPLE_COUNT - 1).
#define SAMPLE_COUNT 16u
#define SAMPLE_BYTES (64u << 10)

static void sample_sha256(const uint8_t *p, uint64_t size, char out[65]) {
    uint64_t n = size < SAMPLE_BYTES ? size : SAMPLE_BYTES, span = size - n;
    uint64_t q = span / (SAMPLE_COUNT - 1), r = span % (SAMPLE_COUNT - 1);
    uint8_t *buf = malloc(n * SAMPLE_COUNT + 1);
    if (!buf) rt_fatal("out of memory for %llu bytes of file samples", (unsigned long long)(n * SAMPLE_COUNT));
    for (uint64_t i = 0; i < SAMPLE_COUNT; i++) memcpy(buf + i * n, p + q * i + r * i / (SAMPLE_COUNT - 1), n);
    uint8_t d[32];
    rt_sha256(buf, n * SAMPLE_COUNT, d);
    free(buf);
    hex(d, out);
}

static void check_file(const RtGameFile *f) {
    char buf[PATH_BYTES];
    const char *path = game_path(f->path, buf, sizeof buf);
    // The player's dump read in place (sample_sha256 set) is the player's own file, which oot2 prepare did not write:
    // say so, and that prepare can take it from its new place.
    const char *dump = f->sample_sha256 ? " (the RomFS is read from this dump in place)" : "";
    const char *again = f->sample_sha256 ? "run oot2 prepare again, with --dump naming where the dump is now"
                                         : "run oot2 prepare again";
    uint64_t size = 0;
    const void *p = plat_map_file(path, &size);
    if (!p) rt_fatal("%s: %s is missing or cannot be read%s: %s", f->what, path, dump, again);
    if (size != f->size) {
        plat_unmap_file(p, size);
        rt_fatal("%s: %s holds %llu bytes, the build expects %llu%s: it changed since oot2 prepare built this program; "
                 "%s", f->what, path, (unsigned long long)size, (unsigned long long)f->size, dump, again);
    }
    if (f->sample_sha256 && !f->sha256) {
        char got[65];
        sample_sha256(p, size, got);
        if (strcmp(got, f->sample_sha256)) {
            plat_unmap_file(p, size);
            rt_fatal("%s: %s changed since oot2 prepare built this program%s: the SHA-256 of its %u sampled ranges is "
                     "%s, the build expects %s; %s", f->what, path, dump, SAMPLE_COUNT, got, f->sample_sha256, again);
        }
    }
    if (f->sha256) {
        uint8_t d[32];
        char got[65];
        rt_sha256(p, size, d);
        hex(d, got);
        if (strcmp(got, f->sha256)) {
            plat_unmap_file(p, size);
            rt_fatal("%s: %s has SHA-256 %s, the build expects %s: it changed since oot2 prepare built this program; "
                     "run oot2 prepare again", f->what, path, got, f->sha256);
        }
    }
    plat_unmap_file(p, size);
}

// The kernel's configuration, from the main.npdm step 3 checked.
static RtKernelConfig kernel_config(void) {
    if (!rt_program.has_npdm)
        rt_fatal("the dump has no main.npdm, and the kernel takes the program's cores, priorities and SVC mask from it");
    for (int i = 0; i < rt_program.file_count; i++) {
        const RtGameFile *f = &rt_program.files[i];
        if (strcmp(f->what, "main.npdm")) continue;
        char buf[PATH_BYTES];
        const char *path = game_path(f->path, buf, sizeof buf);
        uint64_t size = 0;
        const void *p = plat_map_file(path, &size);
        if (!p) rt_fatal("main.npdm: %s cannot be read", path);
        RtKernelConfig k;
        const char *why;
        int ok = rt_npdm_parse(p, size, &k, &why);
        plat_unmap_file(p, size);
        if (!ok) rt_fatal("main.npdm: %s: %s", path, why);
        return k;
    }
    rt_fatal("the program glue has main.npdm's values but does not list the file");
}

static int usage(const char *why) {
    fprintf(stderr, "oot2: %s\nusage: oot2 [--game DIR] [--gfx NAME] [--set KEY=VALUE]... [--config PATH] "
                    "[--vulkan-items]\n  --game DIR      the folder oot2 prepare wrote (default: the folder of this "
                    "program)\n  --gfx NAME      the graphics back end, null or vulkan: the setting gfx\n"
                    "  --set KEY=VALUE a setting for this run, over the OOT2_KEY variable and the settings file\n"
                    "  --config PATH   the settings file (default: the OOT2_CONFIG variable, else oot2.ini in the "
                    "game folder)\n  --vulkan-items  print the Vulkan device items the renderer requires, and exit\n",
            why);
    return 2;
}

// The settings not at their default, in one log line: what a measurement of this run must say it ran with.
static void log_settings(void) {
    RtSettingNow now[RT_SETTING_COUNT];
    int n = rt_settings_nondefault(now, RT_SETTING_COUNT);
    char line[1024];
    size_t k = 0;
    line[0] = 0;
    for (int i = 0; i < n && i < RT_SETTING_COUNT && k < sizeof line; i++) {
        int w = snprintf(line + k, sizeof line - k, "%s%s=%s (%s)", i ? ", " : "", now[i].key, now[i].value,
                         now[i].from);
        if (w < 0) break;
        k += (size_t)w;
    }
    rt_log("oot2: settings file %s; %s%s", rt_settings_path(), n ? "not at their default: " : "every setting at its default",
           line);
}

// The list of oot2/vk_required.h, one "id<TAB>what" line per item.
static int print_vulkan_items(void) {
    for (int i = 0; i < OOT2_VK_ITEM_COUNT; i++) printf("%s\t%s\n", oot2_vk_item_id(i), oot2_vk_item_what(i));
    return fflush(stdout) ? 1 : 0;
}

// ---- the window (oot2/window.h, oot2/vk_present.h). Only the Vulkan back end presents to one, and only in a
// build with SDL3; every other start has none and presents to nothing (null) or to the offscreen capture.
static int menu_button(int button, int down) { return ui_menu_input((UiButton)button, down); }
static int menu_open(void) { return ui_menu_is_open(); }
static void menu_pointer(float x, float y, int down) { ui_menu_pointer(x, y, down); }

// The window's close button ends the run through the stop hooks (census.tsv, the ledger), as the capture's end does.
static void window_closed(void) { rt_census_end(0, "the window was closed"); }

// A size the player gave the window by dragging it is remembered: written to the file with the menu's own calls.
static void window_resized(int width, int height) {
    char why[200], text[RT_SETTING_TEXT];
    snprintf(text, sizeof text, "%d", width);
    int ok = rt_settings_set(RT_SET_window_width, text, why, sizeof why);
    snprintf(text, sizeof text, "%d", height);
    ok = rt_settings_set(RT_SET_window_height, text, why, sizeof why) && ok;
    if (ok && !rt_settings_save(why, sizeof why)) rt_log("oot2: the window's size could not be saved: %s", why);
}

// The live window keys, applied on the thread that changed them (the menu's, or the file watcher's): a request only.
static void window_key(const RtSettingChange *change, void *ctx) {
    (void)change;
    (void)ctx;
    win_request((int)rt_setting(RT_SET_window_mode), (int)rt_setting(RT_SET_window_width), (int)rt_setting(RT_SET_window_height));
}

static const VkwUi k_menu = {ui_menu_frame, ui_menu_draw_data, ui_menu_set_texture_ops, ui_menu_prepare};

static void open_window(const char *gfx) {
    if (strcmp(gfx, "vulkan") || !rt_setting(RT_SET_window)) return;
    WinConfig cfg = {.title = "oot2", .width = (int)rt_setting(RT_SET_window_width),
                     .height = (int)rt_setting(RT_SET_window_height), .mode = (int)rt_setting(RT_SET_window_mode)};
    if (!win_open(&cfg)) {
        rt_log("oot2: no window (%s): the picture goes to the offscreen capture only", win_why());
        return;
    }
    win_set_input(&(WinInput){menu_button, menu_open, menu_pointer, window_resized, window_closed});
    rt_settings_hook(RT_SET_window_mode, window_key, NULL);
    rt_settings_hook(RT_SET_window_width, window_key, NULL);
    rt_settings_hook(RT_SET_window_height, window_key, NULL);
    vkw_window(&win_vulkan, &k_menu);
    rt_log("oot2: window %dx%d, %s", cfg.width, cfg.height, cfg.mode == WIN_BORDERLESS ? "borderless" : "windowed");
}

// The texture heap instruments read the write watch through the page generations (oot2/texheap.h).
static uint64_t pagegen_scanned_pages(void) {
    PagegenStats s;
    pagegen_stats(&s);
    return s.scan_pages;
}

int main(int argc, char **argv) {
    const char *game = NULL, *config = NULL;
    static RtSettingArg sets[MAX_SETS];
    static char gfx_set[MAX_SETS][64];
    int nsets = 0;
    for (int i = 1; i < argc; i++) {
        int set = !strcmp(argv[i], "--set") || !strcmp(argv[i], "--gfx");
        if (set && i + 1 < argc && nsets == MAX_SETS) return usage("more settings on the command line than it takes");
        if (!strcmp(argv[i], "--game") && i + 1 < argc) game = argv[++i];
        else if (!strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) sets[nsets++] = (RtSettingArg){argv[++i], "--set"};
        else if (!strcmp(argv[i], "--gfx") && i + 1 < argc) {
            const char *gfx = argv[++i];
            if (!gfx_backend_known(gfx)) return usage("--gfx: no back end has that name");
            snprintf(gfx_set[nsets], sizeof gfx_set[nsets], "gfx=%s", gfx);   // known, so short
            sets[nsets] = (RtSettingArg){gfx_set[nsets], "--gfx"};
            nsets++;
        } else if (!strcmp(argv[i], "--vulkan-items")) return print_vulkan_items();
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) return usage("start the game");
        else return usage("unknown argument");
    }

    if (game) {
        if (snprintf(g_game, sizeof g_game, "%s", game) >= (int)sizeof g_game) return usage("--game: path too long");
    } else if (!exe_folder(g_game, sizeof g_game)) {
        rt_fatal("cannot find the folder of this program; name the game folder with --game");
    }
    char path[PATH_BYTES], why[600];
    const char *env_config = getenv("OOT2_CONFIG");
    int n = config ? snprintf(path, sizeof path, "%s", config)
          : env_config && *env_config ? snprintf(path, sizeof path, "%s", env_config)
          : snprintf(path, sizeof path, "%s/oot2.ini", g_game);
    if (n < 0 || n >= (int)sizeof path) return usage("the settings file's path is too long");
    if (!rt_settings_load(sets, nsets, path, why, sizeof why)) return usage(why);
    rt_timer_hold_start();   // before the first thread: the vsync clock's first wait runs on the held resolution
    const char *gfx_from;
    char gfx[RT_SETTING_TEXT];
    rt_setting_source(RT_SET_gfx, &gfx_from);
    rt_setting_format(RT_SET_gfx, rt_setting(RT_SET_gfx), gfx);
    open_window(gfx);   // before the Vulkan device, which is made able to present to the window or not made
    gfx_select(gfx, gfx_from);
    gfx_open_game(g_game);

    rt_log("oot2: main() is at %p (a host pc in a log line is read against this and the program's symbols)", (void *)&main);
    log_settings();
    {   // the menu is part of the program: its rows read the store just loaded (runtime/ui)
        const UiBuild ui = ui_standin_build();
        rt_log("oot2: menu: %d rows, %d presets, Dear ImGui %s", ui_menu_row_count(), ui_menu_preset_count(),
               ui.imgui_version ? ui.imgui_version : "not built in");
    }
    rt_log("oot2: game folder %s; %d modules, %d other files to check", g_game, rt_program.module_count,
           rt_program.file_count);
    ledger_set_folder(g_game);
    texheap_set_folder(g_game);
    texheap_set_watch(&(TexheapWatch){pagegen_scan, pagegen_runs, pagegen_scanned_pages});
    ledger_flush_on_stop();
    gpu_set_report_folder(g_game);

    uint64_t t0 = plat_now_ns();
    RtMemConfig cfg = rt_mem_default_config();
    cfg.system_resource_size = rt_program.system_resource_size;
    rt_mem_init(&cfg);

    uint64_t t1 = plat_now_ns();
    for (int i = 0; i < rt_program.file_count; i++) check_file(&rt_program.files[i]);
    double checked = ms_since(t1);

    RtKernelConfig kcfg = kernel_config();
    rt_kernel_init(&kcfg);

    static RtModuleFile files[RT_MAX_LOADED];
    static char paths[RT_MAX_LOADED][PATH_BYTES];
    int count = rt_program.module_count;
    if (count <= 0 || count > RT_MAX_LOADED) rt_fatal("the program glue lists %d modules", count);
    for (int i = 0; i < count; i++) {
        files[i] = rt_program.modules[i];
        files[i].path = game_path(files[i].path, paths[i], sizeof paths[i]);
    }
    uint64_t t2 = plat_now_ns();
    rt_load_modules(files, count);
    double loaded = ms_since(t2);
    rt_load_report();
    nvn_procs_register();
    hos_init(g_game, rt_program.files, rt_program.file_count);
    nvn_settings_bind();
    if (!rt_settings_watch()) rt_log("oot2: the settings file is not watched; an edit applies at the next start");

    RtModule mods[RT_MAX_LOADED];
    int lifted = rt_program_lifted(mods, RT_MAX_LOADED);
    uint64_t t3 = plat_now_ns();
    rt_dispatch_init(mods, lifted);
    double dispatch = ms_since(t3);
    RtDispatchStats d = rt_dispatch_stats();
    uint64_t functions = 0;
    for (int i = 0; i < lifted; i++) functions += mods[i].count;
    rt_log("oot2: files checked in %.0f ms, %d modules loaded in %.0f ms, dispatch table over %d lifted modules "
           "(%llu functions, %.1f MiB) in %.0f ms; %d host overrides linked; start-up %.0f ms", checked, count, loaded,
           lifted, (unsigned long long)functions, (double)d.table_bytes / (1 << 20), dispatch,
           rt_hostfn_override_count(), ms_since(t0));

    capture_start(g_game);   // OOT2_CAPTURE=1 only (oot2/capture.h): after the start-up figures, before any present
    rt_log("oot2: the main thread (priority %d, core %d, stack %#llx) starts the SDK", kcfg.main_priority,
           kcfg.main_core, (unsigned long long)kcfg.main_stack_size);
    rt_kernel_run(hos_boot_entry(), 0);
    rt_log("oot2: every guest thread ended");
    ledger_flush();
    capture_end(0);
    hos_census_write("every guest thread ended");
    return 0;
}
