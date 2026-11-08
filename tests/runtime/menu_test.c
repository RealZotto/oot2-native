// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the configuration menu (runtime/ui, oot2/ui.h), driven by tests/test_runtime_menu.py:
// `menu_test <case> [arguments]`, each case in its own process, over the real settings store and a settings file the
// driver wrote (synthetic text, nothing of the game). Linked with runtime/core, runtime/nvn and the C++ menu, with the
// fetched Dear ImGui when the folder exists: the menu then builds its draw lists with no window, no device and no
// back end. Exit code 1 is a failed check.
//
//   rows P [--set K=V]...          load P, then every menu row, one tab-separated line each
//   frame P                        the closed menu draws nothing; the open one, the overlay and the counters draw
//   set P KEY TEXT                 load P, open the menu, set KEY's row through the menu, close it (which saves)
//   click P KEY                    load P, open the menu, click KEY's checkbox with the pointer, close it
//   reload P                       the store changes under the menu (a reload's own call); the row follows
//   presets P                      scramble every key, apply each preset, report every key left at another value
//   held P --set K=V...            a key the command line holds: greyed, refused, kept by a preset
//   keys                           the open key: F1, Escape, Guide, the Minus and Plus chord, a scripted run
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/settings.h"
#include "oot2/ui.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                       \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define MAX_ARGS 16
static RtSettingArg g_args[MAX_ARGS];
static int g_nargs;

static const char *STATE[] = {"live", "restart", "prepare", "held", "greyed"};

static void load(const char *path) {
    char why[600];
    if (!rt_settings_load(g_args, g_nargs, path, why, sizeof why)) {
        fprintf(stderr, "load: %s\n", why);
        exit(2);
    }
}

// --set K=V words from argv[from..]; the first other word is returned.
static const char *parse(int argc, char **argv, int from) {
    const char *rest = NULL;
    for (int i = from; i < argc; i++) {
        if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            CHECK(g_nargs < MAX_ARGS);
            g_args[g_nargs++] = (RtSettingArg){argv[++i], "--set"};
        } else {
            CHECK(!rest);
            rest = argv[i];
        }
    }
    return rest;
}

static int row_of(const char *key) {
    UiRow r;
    for (int i = 0; ui_menu_row(i, &r); i++)
        if (r.key && !strcmp(r.key, key)) return i;
    fprintf(stderr, "no row has the key %s\n", key);
    exit(1);
}

static void frames(int n, int width, int height, UiDraw *last) {
    for (int i = 0; i < n; i++) {
        UiFrame f = {width, height, 1.0 / 60.0};
        ui_menu_frame(&f, last);
    }
}

static void case_rows(void) {
    UiRow r;
    int n = ui_menu_row_count();
    printf("count\t%d\n", n);
    for (int i = 0; i < n; i++) {
        CHECK(ui_menu_row(i, &r));
        printf("row\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n", i, STATE[r.state], r.group, r.label, r.key ? r.key : "-",
               r.value, r.advanced, r.note);
    }
    CHECK(!ui_menu_row(n, &r) && !ui_menu_row(-1, &r));
    printf("presets\t%d\t%s\t%s\n", ui_menu_preset_count(), ui_menu_preset_name(0), ui_menu_preset_name(1));
}

static void case_frame(void) {
    UiDraw d;
    UiFrame f = {1280, 1000, 1.0 / 60.0};
    CHECK(ui_menu_frame(&f, &d) == 0);   // closed, overlay and counters off: nothing is built
    CHECK(d.windows == 0 && d.vertices == 0 && d.commands == 0);
    ui_menu_open(1);
    int expect = 0;
    UiRow r;
    for (int i = 0; ui_menu_row(i, &r); i++) expect += r.advanced ? 0 : 1;
    CHECK(ui_menu_frame(&f, &d) == 1);
    CHECK(ui_menu_frame(&f, &d) == 1);   // the second: the window has its size, nothing is skipped as hidden
    CHECK(d.windows == 1 && d.rows == expect);
    CHECK(d.vertices > 0 && d.indices > 0 && d.commands > 0);
    printf("menu\t%d\t%d\t%d\t%d\t%d\n", d.windows, d.rows, d.vertices, d.indices, d.commands);
    int menu_vertices = d.vertices;
    ui_menu_open(0);
    CHECK(ui_menu_frame(&f, &d) == 0);   // closed again: nothing
    char why[200];
    CHECK(rt_settings_set(RT_SET_overlay, "1", why, sizeof why));
    CHECK(rt_settings_set(RT_SET_counters, "1", why, sizeof why));
    CHECK(ui_menu_frame(&f, &d) == 1);
    CHECK(ui_menu_frame(&f, &d) == 1);   // a new window draws nothing in its first frame
    CHECK(d.windows == 2 && d.rows == 0 && d.vertices > 0);   // the overlay and the counters list
    printf("overlay\t%d\t%d\n", d.windows, d.vertices);
    ui_menu_open(1);
    CHECK(ui_menu_frame(&f, &d) == 1);
    CHECK(ui_menu_frame(&f, &d) == 1);
    CHECK(d.windows == 3 && d.vertices > menu_vertices);
    printf("all\t%d\n", d.windows);
}

static void case_set(const char *path, const char *key, const char *text) {
    load(path);
    ui_menu_open(1);
    char why[300];
    int ok = ui_menu_set(row_of(key), text, why, sizeof why);
    if (!ok) printf("refused\t%s\n", why);
    ui_menu_open(0);   // the key's close: saves a changed setting
    printf("ok\t%d\n", ok);
}

static void case_click(const char *path, const char *key) {
    load(path);
    ui_menu_open(1);
    const int row = row_of(key);
    int before = (int)rt_setting((RtSettingId)rt_setting_find(key));
    float rect[4];
    frames(3, 1280, 1000, NULL);
    CHECK(ui_menu_row_rect(row, rect));
    const float x = rect[0] + rect[2] / 2, y = rect[1] + rect[3] / 2;
    ui_menu_pointer(x, y, 0);
    frames(2, 1280, 1000, NULL);
    ui_menu_pointer(x, y, 1);
    frames(2, 1280, 1000, NULL);
    ui_menu_pointer(x, y, 0);
    frames(2, 1280, 1000, NULL);
    UiRow r;
    CHECK(ui_menu_row(row, &r));
    printf("clicked\t%d\t%s\n", before, r.value);
    ui_menu_open(0);
}

static void case_reload(void) {
    UiRow r;
    int row = row_of("hold_ticks");
    CHECK(ui_menu_row(row, &r) && !strcmp(r.value, "0"));
    char why[200];
    // what the file watcher's reload does: a value reaches the store with no step of the menu's
    CHECK(rt_settings_set(RT_SET_hold_ticks, "9", why, sizeof why));
    CHECK(ui_menu_row(row, &r));
    printf("after\t%s\t%s\n", r.value, STATE[r.state]);
    ui_menu_open(1);
    UiDraw d;
    frames(2, 1280, 1000, &d);
    // the open menu draws its rows when the build has Dear ImGui; a build without it draws nothing and says so, and the
    // row's value above is all this case can then check (the frame cases skip without the library, this one does not)
    if (ui_standin_build().imgui_version) CHECK(d.rows > 0);
    else CHECK(d.rows == 0);
}

static int64_t have_of(int id) {
    int64_t pending;
    return rt_setting_pending((RtSettingId)id, &pending) ? pending : rt_setting((RtSettingId)id);
}

static int names_count(const char *names) {
    int n = 1;
    for (; names && *names; names++) n += *names == '|';
    return n;
}

// Every key at a value other than its default, through the store like a click.
static void scramble(void) {
    char why[200], text[RT_SETTING_TEXT];
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const RtSettingRow *s = rt_setting_row((RtSettingId)id);
        int64_t v = s->type == RT_SETTING_BOOL ? !s->def : s->type == RT_SETTING_ENUM ? names_count(s->names) - 1 : s->max;
        if (s->type == RT_SETTING_ENUM && v == s->def) v = s->def == 0 ? 1 : 0;
        CHECK(v != s->def);
        CHECK(rt_settings_set((RtSettingId)id, rt_setting_format((RtSettingId)id, v, text), why, sizeof why));
        CHECK(have_of(id) == v);
    }
}

static int left_behind(const char *preset, int overlay_on) {
    int left = 0;
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const RtSettingRow *s = rt_setting_row((RtSettingId)id);
        int64_t want = s->def;
        if (overlay_on && (!strcmp(s->key, "overlay") || !strcmp(s->key, "counters"))) want = 1;
        if (have_of(id) != want) {
            printf("left\t%s\t%s\n", preset, s->key);
            left++;
        }
    }
    return left;
}

static void case_presets(void) {
    char note[200];
    CHECK(ui_menu_preset_count() == 2);
    CHECK(!strcmp(ui_menu_preset_name(0), "Reference") && !strcmp(ui_menu_preset_name(1), "Test"));
    CHECK(ui_menu_apply_preset(7, note, sizeof note) == -1);
    scramble();
    CHECK(ui_menu_apply_preset(0, note, sizeof note) == 0);
    int a = left_behind("Reference", 0);
    scramble();
    CHECK(ui_menu_apply_preset(1, note, sizeof note) == 0);
    int b = left_behind("Test", 1);
    printf("presets\t%d\t%d\n", a, b);
    char why[200];
    CHECK(ui_menu_save(why, sizeof why));
}

static void case_held(void) {
    UiRow r;
    int row = row_of("hold_ticks");
    CHECK(ui_menu_row(row, &r));
    printf("held\t%s\t%s\t%s\n", STATE[r.state], r.value, r.note);
    char why[300], note[200];
    int ok = ui_menu_set(row, "7", why, sizeof why);
    printf("set\t%d\t%s\n", ok, why);
    printf("value\t%lld\n", (long long)rt_setting(RT_SET_hold_ticks));
    int held = ui_menu_apply_preset(0, note, sizeof note);
    printf("preset\t%d\t%s\n", held, note);
    printf("value\t%lld\n", (long long)rt_setting(RT_SET_hold_ticks));
}

static void case_keys(void) {
    CHECK(!ui_menu_is_open());
    CHECK(ui_menu_input(UI_KEY_ESCAPE, 1) == 0);   // closed: the game still sees Escape
    CHECK(!ui_menu_is_open());
    CHECK(ui_menu_input(UI_KEY_F1, 1) == 1 && ui_menu_is_open());
    CHECK(ui_menu_input(UI_KEY_F1, 0) == 1 && ui_menu_is_open());   // the release does not toggle
    CHECK(ui_menu_input(UI_KEY_ESCAPE, 1) == 1 && !ui_menu_is_open());
    CHECK(ui_menu_input(UI_KEY_ESCAPE, 0) == 0);
    CHECK(ui_menu_input(UI_PAD_GUIDE, 1) == 1 && ui_menu_is_open());
    CHECK(ui_menu_input(UI_PAD_GUIDE, 0) == 1);
    CHECK(ui_menu_input(UI_KEY_F1, 1) == 1 && !ui_menu_is_open());
    ui_menu_input(UI_KEY_F1, 0);
    // the chord: Minus alone is the game's, both together open it once
    CHECK(ui_menu_input(UI_PAD_MINUS, 1) == 0 && !ui_menu_is_open());
    CHECK(ui_menu_input(UI_PAD_PLUS, 1) == 1 && ui_menu_is_open());
    CHECK(ui_menu_input(UI_PAD_PLUS, 0) == 1 && ui_menu_is_open());
    CHECK(ui_menu_input(UI_PAD_PLUS, 1) == 1 && ui_menu_is_open());   // pressed again with Minus still down: no new chord
    CHECK(ui_menu_input(UI_PAD_MINUS, 0) == 1);
    CHECK(ui_menu_input(UI_PAD_PLUS, 0) == 1);
    CHECK(ui_menu_is_open());
    CHECK(ui_menu_input(UI_PAD_MINUS, 1) == 1 && ui_menu_input(UI_PAD_PLUS, 1) == 1 && !ui_menu_is_open());   // again
    ui_menu_input(UI_PAD_MINUS, 0);
    ui_menu_input(UI_PAD_PLUS, 0);
    CHECK(!ui_menu_is_open());
    ui_menu_set_scripted(1);   // a scripted run ignores every button
    CHECK(ui_menu_input(UI_KEY_F1, 1) == 0 && !ui_menu_is_open());
    CHECK(ui_menu_input(UI_PAD_GUIDE, 1) == 0 && !ui_menu_is_open());
    ui_menu_input(UI_PAD_MINUS, 1);
    CHECK(ui_menu_input(UI_PAD_PLUS, 1) == 0 && !ui_menu_is_open());
    ui_menu_set_scripted(0);
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    const char *c = argv[1];
    if (!strcmp(c, "keys")) {
        case_keys();
    } else if (!strcmp(c, "set") && argc == 5) {
        case_set(argv[2], argv[3], argv[4]);
    } else if (!strcmp(c, "click") && argc == 4) {
        case_click(argv[2], argv[3]);
    } else {
        const char *path = parse(argc, argv, 2);
        if (!path) path = "";
        load(*path ? path : NULL);
        if (!strcmp(c, "rows")) case_rows();
        else if (!strcmp(c, "frame")) case_frame();
        else if (!strcmp(c, "reload")) case_reload();
        else if (!strcmp(c, "presets")) case_presets();
        else if (!strcmp(c, "held")) case_held();
        else return 1;
    }
    printf("ok %s\n", c);
    return 0;
}
