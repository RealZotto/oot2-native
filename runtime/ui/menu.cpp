// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The configuration menu's rows, presets and open key (oot2/ui.h; the menu survey, the menu answers). No Dear ImGui
// in this file: it is the model, which the frame (menu_draw.cpp) draws and the tests read. Every row reads its value
// from the settings store when it is built and writes through rt_settings_set, so a value the file's hot reload or the
// command line changes shows in the menu with no step of its own, and nothing is kept here that the store does not hold
// (the open flag, the unsaved flag and the status line are the menu's own state, and no setting).
//
// Built without exceptions and without run-time type information (tools/build_game.py, CXX_FLAGS).
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstring>

#include "menu_internal.h"

namespace oot2ui {
namespace {

std::atomic<int> g_open{0}, g_scripted{0}, g_minus{0}, g_plus{0}, g_chord{0}, g_dirty{0};
UiCounterFn g_counters = nullptr;
std::string g_status;
std::mutex g_status_mu;   // set from the window thread (the open key saves), read by the thread that builds the frame

// One designed row. A row whose key the store does not have (yet) is greyed with `grey`, and turns into a live row the
// day the key is registered, so the window's rows need no change here beyond their names if those differ.
struct RowDef {
    RtSettingGroup group;
    const char *label;
    const char *key;     // the store's key, or null for a row that has none and no consumer in sight
    const char *help;    // for a row without a key; a row with one takes the store's text
    const char *grey;    // why it is greyed while the key is absent
    const char *quick;   // "value:label|value|...": buttons beside the control
    bool advanced;
};

constexpr const char *NO_WINDOW = "Greyed: the window's key is not in the store.";

// The first rows (the menu answers): pacing, display, overlay and counters; the rest greyed with their reason.
const RowDef DEFS[] = {
    {RT_GROUP_GRAPHICS, "Graphics back end", "gfx", nullptr, nullptr, nullptr, false},
    {RT_GROUP_GRAPHICS, "Resolution regime", nullptr,
     "The game's own dynamic resolution and the networks that scale it.",
     "Greyed until it is shown that a regime can be forced without changing the game.", nullptr, false},

    {RT_GROUP_PACING, "Frame rate target", "frame_rate_target", nullptr, nullptr, "0:Game|30|40|50|60|100", false},
    {RT_GROUP_PACING, "Pacing hold placement", "hold_placement", nullptr, nullptr, nullptr, false},
    {RT_GROUP_PACING, "Hold every present (ticks)", "hold_ticks", nullptr, nullptr, nullptr, false},
    {RT_GROUP_PACING, "Hold the system timer resolution", "timer_resolution", nullptr, nullptr, nullptr, false},
    {RT_GROUP_PACING, "Hold the game's step during movies", "movie_hold",
     "Keeps the game at its own 60 steps a second while a movie plays.",
     "Greyed until the movie hold exists.", nullptr, false},
    {RT_GROUP_PACING, "Interpolation", nullptr,
     "Extra frames between the game's own, for a display above 60 Hz.",
     "Greyed: planned for after the game runs at its own pace at other rates.", nullptr, false},

    {RT_GROUP_DISPLAY, "Window mode", "window_mode",
     "Windowed, or borderless at the screen's size.", NO_WINDOW, nullptr, false},
    {RT_GROUP_DISPLAY, "Window width", "window_width", "The window's width in pixels; the window remembers it.",
     NO_WINDOW, nullptr, false},
    {RT_GROUP_DISPLAY, "Window height", "window_height", "The window's height in pixels; the window remembers it.",
     NO_WINDOW, nullptr, false},
    {RT_GROUP_DISPLAY, "Open a window", "window", nullptr, "Greyed: the key is not in the store.", nullptr, false},
    {RT_GROUP_DISPLAY, "Output size", nullptr, "The size of the image the game draws: 2560x1440.",
     "Greyed: fixed at 2560x1440 for now; a smaller size will come later as an experimental choice.", nullptr, false},
    {RT_GROUP_DISPLAY, "Exclusive fullscreen at 60 Hz", nullptr,
     "Changes the display's mode to 60 Hz while the game runs.",
     "Greyed until its cost has been measured.", nullptr, false},
    {RT_GROUP_DISPLAY, "Overlay (frame time)", "overlay", nullptr, nullptr, nullptr, false},

    {RT_GROUP_AUDIO, "Audio output", nullptr, "Volume and the output device.",
     "Greyed: the audio output is not built yet.", nullptr, false},
    {RT_GROUP_INPUT, "Stick dead zone", "input_dead_zone", nullptr, nullptr, nullptr, false},
    {RT_GROUP_INPUT, "Controls", nullptr,
     "Keyboard sticks use fixed physical WASD and IJKL when no gamepad is connected. Opposite directions cancel.",
     "Greyed: the mapping is the rows below, input_pad_* (a gamepad's buttons) and input_key_* (the keyboard's keys); "
     "set them here or in oot2.ini's [input]; the menu's own keys (F1, Escape, the pad's Guide) stay fixed.",
     nullptr, false},
    {RT_GROUP_ENHANCEMENTS, "Enhancements", nullptr, "Changes to the game's look and behaviour.",
     "Greyed: none is built, and the game is shown as the console shows it.", nullptr, false},

    {RT_GROUP_DEBUG, "Counters", "counters", nullptr, nullptr, nullptr, false},
    {RT_GROUP_DEBUG, "Page release", "page_release", nullptr, nullptr, nullptr, true},
    {RT_GROUP_DEBUG, "Census mode", "census", nullptr, nullptr, nullptr, true},
    {RT_GROUP_DEBUG, "Census: end the run at present", "census_frames", nullptr, nullptr, nullptr, true},
    {RT_GROUP_DEBUG, "Texture heap instruments", "texheap", nullptr, nullptr, nullptr, true},
    // The texture heap's two parameters have no row of their own: they are listed by key, as any new key is.
};

void put(char *out, size_t size, const char *text) { std::snprintf(out, size, "%s", text ? text : ""); }

// A rate other than 20, 30 and 60 does not divide the game's 60 steps a second into whole steps, or is above 60
// (the frame-rate answers): offered, and marked, never refused.
bool experimental_rate(int64_t n) { return n != 0 && n != 20 && n != 30 && n != 60; }

Row make(RtSettingGroup group, const char *label, const char *key, const char *help, const char *grey,
         const char *quick, bool advanced) {
    Row r;
    r.info = UiRow{};   // no key, no help, no note until said otherwise
    r.group = group;
    r.info.group = group_label(group);
    r.info.label = label;
    r.info.advanced = advanced ? 1 : 0;
    r.quick = quick;
    const int id = key ? rt_setting_find(key) : -1;
    if (id < 0) {   // no key in the store: nothing consumes this row yet
        r.info.state = UI_ROW_GREYED;
        r.info.help = help ? help : "";
        put(r.info.note, sizeof r.info.note, grey ? grey : "Greyed: nothing consumes it yet.");
        return r;
    }
    const RtSettingRow *s = rt_setting_row((RtSettingId)id);
    r.id = id;
    r.info.key = s->key;
    r.info.help = help ? help : s->what;
    r.type = s->type, r.min = s->min, r.max = s->max, r.names = s->names;
    // The value the row shows is the store's: for a restart or prepare key changed since start-up, the value the
    // next start uses.
    int64_t pending = 0;
    const bool has_pending = rt_setting_pending((RtSettingId)id, &pending) != 0;
    const int64_t published = rt_setting((RtSettingId)id);
    r.value = has_pending ? pending : published;
    char text[RT_SETTING_TEXT];
    put(r.info.value, sizeof r.info.value, rt_setting_format((RtSettingId)id, r.value, text));
    const char *from = "";
    const RtSettingSource src = rt_setting_source((RtSettingId)id, &from);
    char note[sizeof r.info.note] = "";
    if (src == RT_SOURCE_ENV || src == RT_SOURCE_COMMAND) {
        r.info.state = UI_ROW_HELD;
        std::snprintf(note, sizeof note, "Held by the %s (%s); change it there.", rt_setting_source_name(src), from);
    } else if (s->apply == RT_APPLY_RESTART || s->apply == RT_APPLY_PREPARE) {
        r.info.state = s->apply == RT_APPLY_RESTART ? UI_ROW_RESTART : UI_ROW_PREPARE;
        const char *when = s->apply == RT_APPLY_RESTART ? "the next start" : "the next oot2 prepare";
        if (has_pending) {
            char now[RT_SETTING_TEXT];
            std::snprintf(note, sizeof note, "Takes effect at %s; this run keeps %s.", when,
                          rt_setting_format((RtSettingId)id, published, now));
        } else {
            std::snprintf(note, sizeof note, "Takes effect at %s.", when);
        }
    } else {
        r.info.state = UI_ROW_LIVE;
        if (std::strcmp(s->key, "frame_rate_target") == 0 && experimental_rate(r.value))
            put(note, sizeof note, "Experimental: 60 / N is not a whole number, or N is above 60.");
    }
    put(r.info.note, sizeof r.info.note, note);
    return r;
}

// The Test preset's difference from Reference (every default): the overlay and the counters on.
struct Override {
    const char *key;
    int64_t value;
};
const Override TEST_OVERRIDES[] = {{"overlay", 1}, {"counters", 1}};
struct Preset {
    const char *name;
    const Override *over;
    int count;
};
const Preset PRESETS[] = {{"Reference", nullptr, 0}, {"Test", TEST_OVERRIDES, 2}};

}   // namespace

const char *group_label(RtSettingGroup g) {
    switch (g) {
        case RT_GROUP_GRAPHICS: return "Graphics";
        case RT_GROUP_PACING: return "Frame rate and pacing";
        case RT_GROUP_DISPLAY: return "Display";
        case RT_GROUP_AUDIO: return "Audio";
        case RT_GROUP_INPUT: return "Input";
        case RT_GROUP_ENHANCEMENTS: return "Enhancements";
        case RT_GROUP_DEBUG: return "Debug";
        default: return "";
    }
}

std::vector<Row> rows() {
    std::vector<Row> out;
    bool covered[RT_SETTING_COUNT] = {};
    for (const RowDef &d : DEFS) {
        out.push_back(make(d.group, d.label, d.key, d.help, d.grey, d.quick, d.advanced));
        if (out.back().id >= 0) covered[out.back().id] = true;
    }
    // A key of the store with no row of its own is still in the menu's reach: listed in its group by its own name
    // (behind the Advanced switch when it is a debug key, an instrument and not a choice).
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        if (covered[id]) continue;
        const RtSettingRow *s = rt_setting_row((RtSettingId)id);
        out.push_back(make(s->group, s->key, s->key, nullptr, nullptr, nullptr, s->group == RT_GROUP_DEBUG));
    }
    std::stable_sort(out.begin(), out.end(), [](const Row &a, const Row &b) { return a.group < b.group; });
    return out;
}

bool commit(const Row &row, const char *text, char *why, size_t why_size) {
    if (row.id < 0 || row.info.state == UI_ROW_GREYED) {
        std::snprintf(why, why_size, "%s: %s", row.info.label, row.info.note);
        return false;
    }
    if (row.info.state == UI_ROW_HELD) {
        std::snprintf(why, why_size, "%s: %s", row.info.label, row.info.note);
        return false;
    }
    if (!rt_settings_set((RtSettingId)row.id, text, why, why_size)) return false;
    g_dirty = 1;
    return true;
}

void set_status(const char *text) {
    std::lock_guard<std::mutex> hold(g_status_mu);
    g_status = text ? text : "";
}
std::string status() {
    std::lock_guard<std::mutex> hold(g_status_mu);
    return g_status;
}
UiCounterFn counters() { return g_counters; }

bool own_counter(int index, char *name, size_t name_size, char *value, size_t value_size) {
    if (index == 0) {
        put(name, name_size, "Settings changed since start");
        std::snprintf(value, value_size, "%llu", (unsigned long long)rt_settings_changes());
        return true;
    }
    if (index == 1) {
        put(name, name_size, "Settings not at their default");
        std::snprintf(value, value_size, "%d", rt_settings_nondefault(nullptr, 0));
        return true;
    }
    return false;
}

}   // namespace oot2ui

using namespace oot2ui;

extern "C" int ui_menu_row_count(void) { return (int)rows().size(); }

extern "C" int ui_menu_row(int index, UiRow *out) {
    const std::vector<Row> all = rows();
    if (index < 0 || index >= (int)all.size() || !out) return 0;
    *out = all[(size_t)index].info;
    return 1;
}

extern "C" int ui_menu_set(int index, const char *text, char *why, size_t why_size) {
    const std::vector<Row> all = rows();
    if (index < 0 || index >= (int)all.size()) {
        std::snprintf(why, why_size, "no menu row has the index %d", index);
        return 0;
    }
    return commit(all[(size_t)index], text, why, why_size) ? 1 : 0;
}

extern "C" int ui_menu_preset_count(void) { return (int)(sizeof PRESETS / sizeof PRESETS[0]); }

extern "C" const char *ui_menu_preset_name(int index) {
    return index >= 0 && index < ui_menu_preset_count() ? PRESETS[index].name : nullptr;
}

extern "C" int ui_menu_apply_preset(int index, char *note, size_t note_size) {
    if (index < 0 || index >= ui_menu_preset_count()) return -1;
    const Preset &p = PRESETS[index];
    int held = 0;
    if (note_size) note[0] = '\0';
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const RtSettingRow *s = rt_setting_row((RtSettingId)id);
        int64_t want = s->def;
        for (int k = 0; k < p.count; k++)
            if (std::strcmp(p.over[k].key, s->key) == 0) want = p.over[k].value;
        int64_t pending = 0;
        auto have = [&]() { return rt_setting_pending((RtSettingId)id, &pending) ? pending : rt_setting((RtSettingId)id); };
        if (have() != want) {
            char text[RT_SETTING_TEXT], why[160];
            rt_settings_set((RtSettingId)id, rt_setting_format((RtSettingId)id, want, text), why, sizeof why);
            g_dirty = 1;
        }
        if (have() != want) {   // the command line or the environment holds it
            const char *from = "";
            rt_setting_source((RtSettingId)id, &from);
            if (!held && note_size) std::snprintf(note, note_size, "%s is held by %s", s->key, from);
            held++;
        }
    }
    return held;
}

extern "C" int ui_menu_save(char *why, size_t why_size) {
    if (!rt_settings_save(why, why_size)) return 0;
    g_dirty = 0;
    return 1;
}

static void flush_save(void) {
    if (!g_dirty) return;
    char why[400];
    if (ui_menu_save(why, sizeof why)) set_status("");
    else set_status(why);
}

static void toggle(void) {
    const int was = g_open.fetch_xor(1);
    if (was) flush_save();
}

extern "C" int ui_menu_input(UiButton button, int down) {
    if (g_scripted) return 0;
    switch (button) {
        case UI_KEY_F1:
        case UI_PAD_GUIDE:
            if (down) toggle();
            return 1;
        case UI_KEY_ESCAPE:
            if (!g_open) return 0;
            if (down) ui_menu_open(0);
            return 1;
        case UI_PAD_MINUS:
        case UI_PAD_PLUS: {
            (button == UI_PAD_MINUS ? g_minus : g_plus) = down ? 1 : 0;
            const bool both = g_minus && g_plus;
            if (both && !g_chord.exchange(1)) {
                toggle();
                return 1;
            }
            const bool in_chord = g_chord != 0;
            if (!both && !g_minus && !g_plus) g_chord = 0;
            return in_chord || g_open ? 1 : 0;
        }
    }
    return 0;
}

extern "C" void ui_menu_set_scripted(int on) { g_scripted = on ? 1 : 0; }
extern "C" int ui_menu_is_open(void) { return g_open ? 1 : 0; }

extern "C" void ui_menu_open(int open) {
    if (open) {
        g_open = 1;
    } else if (g_open.exchange(0)) {
        flush_save();
    }
}

extern "C" void ui_menu_set_counters(UiCounterFn fn) { g_counters = fn; }
