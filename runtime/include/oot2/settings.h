// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The settings store (the menu survey): one table of every setting the runtime reads, four layers that
// give each key its value, a file beside the game that a person edits between runs or while the game runs, and the
// hooks that apply a live key at once. Consumers read typed values with rt_setting and never parse a file or the
// environment; later front ends (the menu, the capture header) edit and list this same table.
//
// The layers, highest first: the command line (`oot2 --set key=value`, and `--gfx NAME` for gfx), the environment
// (OOT2_<KEY>, the key in capitals), the file, the default. The value of the highest layer that sets a key is its
// effective value, and the store says which layer and which name (the option, the variable, the file's path) hold it,
// so a menu can grey a row the command line or the environment holds. The variables the runtime read before the store
// existed (OOT2_GFX, OOT2_HOLD_TICKS, OOT2_PAGE_RELEASE, OOT2_CENSUS, OOT2_CENSUS_FRAMES) are the environment names of
// keys named after them, so each keeps working as it did; a value a key does not take is refused by name, never clamped
// and never read as the default.
//
// The file. oot2.ini in the game folder, beside census.tsv and ledger.jsonl, unless `oot2 --config PATH` or the
// OOT2_CONFIG variable names another (the menu survey's conservative default; oot2 prepare never writes or deletes it).
// Its lines:
//   ; a comment, or # a comment       whole lines only; a value runs to the end of its line
//   [pacing]                          a section: the key's group, for people; a key is found by its name in any section
//   hold_ticks = 3                    key = value, spaces around '=' optional
// Blank lines, comments, sections and keys this table does not have are kept as they are, and logged once when read.
// A line that is none of these, a key named twice, or a value its key does not take refuses the file: at start-up the
// program stops with exit code 2, naming the file, the line and the key; on a reload the file is not applied and the
// store keeps the values it had. A save rewrites only the lines of the keys the store changed (rt_settings_set), adds a
// key the file lacks at the end of its group's section (or in a new section at the end), writes a temporary file beside
// the file and renames it over the file, so a failure leaves the old file whole. Nothing saves on the frame path: a
// save happens when its caller asks. No game data and no program key ever goes into the file.
//
// Applying a change. Each key has a mark: live (applied while the game runs), restart (read at start-up only) or
// prepare (built into the prepared game: `oot2 prepare` must run again). A live key's new value is published as soon as
// it is effective, on the thread that changed it (rt_settings_load at start-up, rt_settings_set, the file watcher), and
// the hooks of that key run on that thread, in order of registration; consumers read the published value with rt_setting
// (an atomic load, cheap enough for the frame path) or compare rt_settings_changes, which rises once per published
// change and not when a value is written again unchanged. A restart or prepare key keeps the value it had at start-up:
// a later change is logged once as needing a restart (or a prepare), the hooks see it as pending, and rt_setting_pending
// tells a menu what the next start will use.
//
// The file watcher (rt_settings_watch) is a host thread of its own, never a guest thread and never the present thread:
// every RT_SETTINGS_POLL_NS it reads the file whole, and applies it when two reads in a row agree and differ from what
// it last applied, so an editor's half-written save is never applied; an edit is seen within one to two polls (0.5 to
// 1 s). Without the watcher (tests), rt_settings_poll does one such step.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {   // the menu (runtime/ui) is the one C++ reader of the store
#endif

typedef enum RtSettingType { RT_SETTING_BOOL, RT_SETTING_INT, RT_SETTING_ENUM } RtSettingType;
typedef enum RtSettingApply { RT_APPLY_LIVE, RT_APPLY_RESTART, RT_APPLY_PREPARE } RtSettingApply;
// The menu survey's groups, in menu order; each is the file's section of that name.
typedef enum RtSettingGroup {
    RT_GROUP_GRAPHICS,       // [graphics]
    RT_GROUP_PACING,         // [pacing]: frame rate and pacing
    RT_GROUP_DISPLAY,        // [display]
    RT_GROUP_AUDIO,          // [audio]
    RT_GROUP_INPUT,          // [input]
    RT_GROUP_ENHANCEMENTS,   // [enhancements]
    RT_GROUP_DEBUG,          // [debug]: debug views and measurement switches
    RT_GROUP_COUNT
} RtSettingGroup;
// The layers, lowest first: a higher layer that sets a key wins.
typedef enum RtSettingSource {
    RT_SOURCE_DEFAULT,
    RT_SOURCE_FILE,
    RT_SOURCE_ENV,
    RT_SOURCE_COMMAND,
    RT_SOURCE_COUNT
} RtSettingSource;

// The input mapping's source names (ticket 151), each in the order oot2/hid_input.h's HidPadSource / HidKeySource enums
// name it and finished by "none" (the console button has no source of this kind); runtime/platform/window.c's SDL tables
// sit at those indexes, and the remap (input_pad_* / input_key_*) picks from them.
#define HID_PAD_SOURCE_NAMES                                                                                             \
    "south|east|west|north|left_shoulder|right_shoulder|back|start|dpad_up|dpad_down|dpad_left|dpad_right|"              \
    "left_stick|right_stick|left_trigger|right_trigger|none"
#define HID_KEY_SOURCE_NAMES                                                                                             \
    "up|down|left|right|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|return|space|tab|backspace|lshift|"        \
    "rshift|lctrl|rctrl|none"

// The table: X(key, group, type, apply, default, min, max, names, what), one row per key.
//   key      lower case, digits and '_'; unique; the environment name is OOT2_ and the key in capitals, so a key never
//            takes the name of a variable the tools read (OOT2_CC, OOT2_NINJA, OOT2_CONFIG, OOT2_REQUIRE_*,
//            OOT2_VSYNC_P99_US)
//   type     BOOL takes 0 or 1; INT takes a decimal integer from min to max; ENUM takes one of names, '|'-separated,
//            and its value is the name's index (min and max unused)
//   default  a value of the key's type (for ENUM, an index into names); an INT default outside min..max is an off value
//            that the key takes as well (frame_rate_target: 0, or 20 to 120)
//   what     one line for the menu and the listing
// A key whose consumer later work builds is added by that work, with its consumer; the settings store study
// lists the keys the work in flight is to register.
#define RT_SETTINGS(X)                                                                                                 \
    X(gfx, GRAPHICS, ENUM, RESTART, 0, 0, 0, "null|vulkan",                                                            \
      "the graphics back end: null draws nothing and runs every test; vulkan draws on the GPU")                      \
    X(hold_ticks, PACING, INT, LIVE, 0, 0, 999999, NULL,                                                               \
      "experiment E2: hold every present for this many ticks of the 60 Hz clock, whatever the game asks for; 0 is off") \
    X(hold_placement, PACING, ENUM, LIVE, 0, 0, 0, "sync|acquire",                                                    \
      "where AcquireTexture's pacing hold is: sync, in the sync the call returns, as on the console; acquire, in the call") \
    X(timer_resolution, PACING, BOOL, RESTART, 1, 0, 1, NULL,                                                          \
      "hold a 0.5 ms system timer resolution while the game runs; 0 leaves the system's")                              \
    X(frame_rate_target, PACING, INT, LIVE, 0, 20, 120, NULL,                                                          \
      "fixed frame rate: pace presents on their own schedule at this many a second, whatever interval the game asks for; 0 is off") \
    X(window, DISPLAY, BOOL, RESTART, 1, 0, 1, NULL,                                                                   \
      "open a window and present the picture to it (the Vulkan back end, in a build with SDL3); 0 presents to the offscreen capture only") \
    X(window_mode, DISPLAY, ENUM, LIVE, 0, 0, 0, "windowed|borderless",                                                \
      "windowed: a decorated, resizable window; borderless: a window with no border at the size of the screen")        \
    X(window_width, DISPLAY, INT, LIVE, 1280, 320, 16384, NULL,                                                        \
      "the width in pixels of the window in windowed mode; resizing the window changes it, and it is remembered")      \
    X(window_height, DISPLAY, INT, LIVE, 720, 180, 16384, NULL,                                                        \
      "the height in pixels of the window in windowed mode; resizing the window changes it, and it is remembered")     \
    X(overlay, DISPLAY, BOOL, LIVE, 0, 0, 1, NULL,                                                                     \
      "the frame-time readout drawn over the picture; the Test preset turns it on")                                    \
    X(audio_sink, AUDIO, ENUM, RESTART, 0, 0, 0, "auto|null|capture|sdl3",                                            \
      "where the mixed audio goes: auto is the audio device when this build has one, else null; null has no sound and " \
      "drains on the renderer's clock; capture writes audio-capture.wav in the game folder; sdl3 needs the device")    \
    X(input_dead_zone, INPUT, INT, LIVE, 3200, 0, 32767, NULL,                                                         \
      "controller stick dead zone: a stick displacement at or below this reads as centered (32767 is the stick's range)") \
    X(input_pad_a, INPUT, ENUM, LIVE, 1, 0, 0, HID_PAD_SOURCE_NAMES,                                                  \
      "the gamepad button that presses the controller's A (none: never pressed)")                                       \
    X(input_pad_b, INPUT, ENUM, LIVE, 0, 0, 0, HID_PAD_SOURCE_NAMES,                                                   \
      "the gamepad button that presses the controller's B (none: never pressed)")                                       \
    X(input_pad_x, INPUT, ENUM, LIVE, 3, 0, 0, HID_PAD_SOURCE_NAMES,                                                   \
      "the gamepad button that presses the controller's X (none: never pressed)")                                       \
    X(input_pad_y, INPUT, ENUM, LIVE, 2, 0, 0, HID_PAD_SOURCE_NAMES,                                                   \
      "the gamepad button that presses the controller's Y (none: never pressed)")                                       \
    X(input_pad_l, INPUT, ENUM, LIVE, 4, 0, 0, HID_PAD_SOURCE_NAMES,                                                   \
      "the gamepad button or trigger that presses the controller's L (none: never pressed)")                            \
    X(input_pad_r, INPUT, ENUM, LIVE, 5, 0, 0, HID_PAD_SOURCE_NAMES,                                                   \
      "the gamepad button or trigger that presses the controller's R (none: never pressed)")                            \
    X(input_pad_zl, INPUT, ENUM, LIVE, 14, 0, 0, HID_PAD_SOURCE_NAMES,                                                 \
      "the gamepad button or trigger that presses the controller's ZL (none: never pressed)")                           \
    X(input_pad_zr, INPUT, ENUM, LIVE, 15, 0, 0, HID_PAD_SOURCE_NAMES,                                                 \
      "the gamepad button or trigger that presses the controller's ZR (none: never pressed)")                           \
    X(input_pad_plus, INPUT, ENUM, LIVE, 7, 0, 0, HID_PAD_SOURCE_NAMES,                                                \
      "the gamepad button that presses the controller's Plus (none: never pressed)")                                   \
    X(input_pad_minus, INPUT, ENUM, LIVE, 6, 0, 0, HID_PAD_SOURCE_NAMES,                                                \
      "the gamepad button that presses the controller's Minus (none: never pressed)")                                  \
    X(input_pad_left, INPUT, ENUM, LIVE, 10, 0, 0, HID_PAD_SOURCE_NAMES,                                               \
      "the gamepad button that presses the controller's D-pad Left (none: never pressed)")                             \
    X(input_pad_up, INPUT, ENUM, LIVE, 8, 0, 0, HID_PAD_SOURCE_NAMES,                                                  \
      "the gamepad button that presses the controller's D-pad Up (none: never pressed)")                               \
    X(input_pad_right, INPUT, ENUM, LIVE, 11, 0, 0, HID_PAD_SOURCE_NAMES,                                              \
      "the gamepad button that presses the controller's D-pad Right (none: never pressed)")                            \
    X(input_pad_down, INPUT, ENUM, LIVE, 9, 0, 0, HID_PAD_SOURCE_NAMES,                                               \
      "the gamepad button that presses the controller's D-pad Down (none: never pressed)")                             \
    X(input_pad_stickl, INPUT, ENUM, LIVE, 12, 0, 0, HID_PAD_SOURCE_NAMES,                                             \
      "the gamepad button that presses the controller's left stick click (none: never pressed)")                        \
    X(input_pad_stickr, INPUT, ENUM, LIVE, 13, 0, 0, HID_PAD_SOURCE_NAMES,                                             \
      "the gamepad button that presses the controller's right stick click (none: never pressed)")                       \
    X(input_key_a, INPUT, ENUM, LIVE, 30, 0, 0, HID_KEY_SOURCE_NAMES,                                                  \
      "the keyboard key that presses the controller's A while no gamepad is connected (none: never pressed)")           \
    X(input_key_b, INPUT, ENUM, LIVE, 27, 0, 0, HID_KEY_SOURCE_NAMES,                                                  \
      "the keyboard key that presses the controller's B while no gamepad is connected (none: never pressed)")           \
    X(input_key_x, INPUT, ENUM, LIVE, 6, 0, 0, HID_KEY_SOURCE_NAMES,                                                   \
      "the keyboard key that presses the controller's X while no gamepad is connected (none: never pressed)")           \
    X(input_key_y, INPUT, ENUM, LIVE, 25, 0, 0, HID_KEY_SOURCE_NAMES,                                                  \
      "the keyboard key that presses the controller's Y while no gamepad is connected (none: never pressed)")           \
    X(input_key_l, INPUT, ENUM, LIVE, 34, 0, 0, HID_KEY_SOURCE_NAMES,                                                  \
      "the keyboard key that presses the controller's L while no gamepad is connected (none: never pressed)")           \
    X(input_key_r, INPUT, ENUM, LIVE, 36, 0, 0, HID_KEY_SOURCE_NAMES,                                                  \
      "the keyboard key that presses the controller's R while no gamepad is connected (none: never pressed)")           \
    X(input_key_zl, INPUT, ENUM, LIVE, 20, 0, 0, HID_KEY_SOURCE_NAMES,                                                 \
      "the keyboard key that presses the controller's ZL while no gamepad is connected (none: never pressed)")           \
    X(input_key_zr, INPUT, ENUM, LIVE, 8, 0, 0, HID_KEY_SOURCE_NAMES,                                                  \
      "the keyboard key that presses the controller's ZR while no gamepad is connected (none: never pressed)")           \
    X(input_key_plus, INPUT, ENUM, LIVE, 31, 0, 0, HID_KEY_SOURCE_NAMES,                                               \
      "the keyboard key that presses the controller's Plus while no gamepad is connected (none: never pressed)")         \
    X(input_key_minus, INPUT, ENUM, LIVE, 32, 0, 0, HID_KEY_SOURCE_NAMES,                                              \
      "the keyboard key that presses the controller's Minus while no gamepad is connected (none: never pressed)")       \
    X(input_key_left, INPUT, ENUM, LIVE, 2, 0, 0, HID_KEY_SOURCE_NAMES,                                                \
      "the keyboard key that presses the controller's D-pad Left while no gamepad is connected (none: never pressed)") \
    X(input_key_up, INPUT, ENUM, LIVE, 0, 0, 0, HID_KEY_SOURCE_NAMES,                                                 \
      "the keyboard key that presses the controller's D-pad Up while no gamepad is connected (none: never pressed)")   \
    X(input_key_right, INPUT, ENUM, LIVE, 3, 0, 0, HID_KEY_SOURCE_NAMES,                                               \
      "the keyboard key that presses the controller's D-pad Right while no gamepad is connected (none: never pressed)") \
    X(input_key_down, INPUT, ENUM, LIVE, 1, 0, 0, HID_KEY_SOURCE_NAMES,                                                \
      "the keyboard key that presses the controller's D-pad Down while no gamepad is connected (none: never pressed)") \
    X(input_key_stickl, INPUT, ENUM, LIVE, 38, 0, 0, HID_KEY_SOURCE_NAMES,                                            \
      "the keyboard key that presses the controller's left stick click while no gamepad is connected (none: never pressed)") \
    X(input_key_stickr, INPUT, ENUM, LIVE, 38, 0, 0, HID_KEY_SOURCE_NAMES,                                            \
      "the keyboard key that presses the controller's right stick click while no gamepad is connected (none: never pressed)") \
    X(counters, DEBUG, BOOL, LIVE, 0, 0, 1, NULL,                                                                      \
      "the counters list drawn over the picture; the Test preset turns it on")                                         \
    X(page_release, DEBUG, ENUM, RESTART, 0, 0, 0, "off|count|release",                                                \
      "page ranges of lever L1: off, counted only, or counted and released")                                          \
    X(census, DEBUG, BOOL, RESTART, 0, 0, 1, NULL,                                                                     \
      "census mode: an NVN2 entry point without a body answers 0 and is listed instead of stopping; never a default run") \
    X(census_frames, DEBUG, INT, RESTART, 0, 0, 1000000000, NULL,                                                      \
      "census mode only: end the run at this present, after the census and the ledger are written; 0 is no limit")    \
    X(texheap, DEBUG, BOOL, RESTART, 0, 0, 1, NULL,                                                                    \
      "spike S7's instruments: texture storage pages by state, CPU writes per frame, and the owner side of lever L1 "   \
      "(texheap-*.tsv in the game folder); with page_release the selected pages are armed or released")                \
    X(texheap_every, DEBUG, INT, RESTART, 300, 1, 100000, NULL,                                                        \
      "texheap only: presents between two samples of the texture heap")                                                \
    X(texheap_trigger, DEBUG, ENUM, RESTART, 0, 0, 0, "bind|register",                                                 \
      "texheap only: when a texture's CPU-written pages are selected: bind, after a completed submission that binds it; "\
      "register, once its descriptor is registered, as a back end that uploads at registration would")                  \
    X(census_timer, DEBUG, BOOL, RESTART, 0, 0, 1, NULL,                                                              \
      "census mode only: time every host function call by its own thread CPU time (census-time.tsv); costs about 2 us a call") \
    X(census_probe, DEBUG, BOOL, RESTART, 0, 0, 1, NULL,                                                               \
      "census mode only: the D11 probe, a hash of each command memory range compared at EndRecording, submit and Finalize")

typedef enum RtSettingId {
#define RT_SETTING_ID(key, group, type, apply, def, min, max, names, what) RT_SET_##key,
    RT_SETTINGS(RT_SETTING_ID)
#undef RT_SETTING_ID
    RT_SETTING_COUNT
} RtSettingId;
#define RT_SETTING_ANY (-1)   // rt_settings_hook: every key

typedef struct RtSettingRow {
    const char *key;
    RtSettingGroup group;
    RtSettingType type;
    RtSettingApply apply;
    int64_t def, min, max;   // for ENUM: 0 and the last index
    const char *names;       // ENUM: "a|b|c"; NULL otherwise
    const char *what;
    const char *env;         // OOT2_<KEY>
} RtSettingRow;

#define RT_SETTINGS_POLL_NS 500000000ull   // the file watcher's period
#define RT_SETTING_TEXT 24                 // bytes of a value as text, with its terminator

// ---- the table
const RtSettingRow *rt_setting_row(RtSettingId id);   // NULL out of range
int rt_setting_find(const char *key);                 // the id of key, or -1
const char *rt_setting_group_name(RtSettingGroup g);  // "graphics", ...; the file's section name
const char *rt_setting_apply_name(RtSettingApply a);  // "live", "restart", "prepare"
const char *rt_setting_source_name(RtSettingSource s);   // "default", "file", "environment", "command line"

// ---- reading
// The published value: for BOOL 0 or 1, for INT the number, for ENUM the index into names. Before rt_settings_load,
// the first read takes the environment and the defaults (a program that never loads the store, such as a test, still
// sees its OOT2_ variables); a value there that its key does not take ends the process with exit code 2, naming it.
int64_t rt_setting(RtSettingId id);
// The value as text ("1", "300", "release") into out (RT_SETTING_TEXT bytes); returns out.
const char *rt_setting_format(RtSettingId id, int64_t value, char out[RT_SETTING_TEXT]);
// The layer that holds the published value, and in *from (when from is not NULL) its name: the option ("--set",
// "--gfx"), the variable ("OOT2_HOLD_TICKS"), the file's path, or "the default". The string lives as long as the store.
RtSettingSource rt_setting_source(RtSettingId id, const char **from);
// 1 when the layers now give a restart or prepare key a value other than the one published at start-up (the next start
// uses *value); 0 otherwise.
int rt_setting_pending(RtSettingId id, int64_t *value);
// Published changes since start-up, all keys together: a consumer that keeps a parsed copy compares it.
uint64_t rt_settings_changes(void);

// Every key whose published value is not its default, with its value and source, in table order: fills at most max
// entries and returns how many such keys there are (the capture header's list, the menu survey).
typedef struct RtSettingNow {
    RtSettingId id;
    const char *key;
    char value[RT_SETTING_TEXT];
    RtSettingSource source;
    const char *from;   // as rt_setting_source gives it
} RtSettingNow;
int rt_settings_nondefault(RtSettingNow *out, int max);

// ---- start-up
// One command-line setting: "key=value" and the option that gave it ("--set", or "--gfx" for main.c's own option).
typedef struct RtSettingArg {
    const char *text;
    const char *from;
} RtSettingArg;
// Read the layers once at start-up, before any consumer reads a key: the command line (args, in order: a later one wins
// over an earlier one for the same key), the environment, and the file at path (NULL: none; a path that does not exist
// is an empty file, which a save creates). Returns 1; or 0 with one line in why naming the option, variable or
// file line, the key and what it takes, and the caller stops with exit code 2. Every value is published, no hook runs
// and rt_settings_changes stays 0.
int rt_settings_load(const RtSettingArg *args, int count, const char *path, char *why, size_t why_size);
const char *rt_settings_path(void);   // the file's path, or "" when there is none

// ---- changing and saving (the menu's calls; never on the frame path)
// Set key id's file-layer value from text, as an edit of the file would: 1, or 0 with why when the key does not take
// it. The effective value changes only when no higher layer holds the key; a live key is applied at once.
int rt_settings_set(RtSettingId id, const char *text, char *why, size_t why_size);
// Write the values rt_settings_set changed into the file (see the header): 1, or 0 with why, the file as it was.
int rt_settings_save(char *why, size_t why_size);
// For tests: when on, a save fails after its temporary file is written and closed, as a refused rename would.
void rt_settings_fail_rename(int on);

// ---- the file watcher
// One step of the watcher: read the file, and apply it when it matches the previous step's read and differs from what
// was applied. Returns 0 when nothing changed, 1 when a change was seen and waits for the next step, 2 when the file
// was applied (or refused: a refusal is logged and the old values stay).
int rt_settings_poll(void);
// Start the watcher thread, which runs rt_settings_poll every RT_SETTINGS_POLL_NS until the process ends: 1, or 0 when
// there is no file path or the thread cannot start.
int rt_settings_watch(void);

// ---- hooks
// A change of key id (RT_SETTING_ANY: of any key). value is the new value; pending is 1 for a restart or prepare key
// whose change waits for the next start (its published value is still old_value). The hook runs on the thread that
// made the change, with the store's lock held: it may read settings, never change, save or poll them.
typedef struct RtSettingChange {
    RtSettingId id;
    int64_t old_value, value;
    RtSettingSource source;
    const char *from;
    int pending;
} RtSettingChange;
typedef void (*RtSettingHook)(const RtSettingChange *change, void *ctx);
void rt_settings_hook(int id, RtSettingHook fn, void *ctx);   // at most 32 hooks; fatal past that

// The modules that own live keys apply them through hooks bound at start-up, after rt_settings_load and before the game
// runs; each applies its keys' values once as it binds. runtime/nvn/nvn_settings.c: hold_ticks, hold_placement and
// frame_rate_target.
void nvn_settings_bind(void);

#ifdef __cplusplus
}
#endif
