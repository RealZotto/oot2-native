// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The host input layer (ticket 151): what the window thread publishes and nn::hid answers from, and the pure mapping
// helpers it uses. The window thread (runtime/platform) writes one coherent snapshot per sample; ov_hid.c reads it for
// the full-key GetNpadStates form. The header names no SDL type, so a test can stand in for the producer.
#pragma once
#include <stdint.h>

// The full-key controller's buttons, as nn::hid lays them out (libnx HidNpadButton bits).
enum {
    HID_BTN_A = 1u << 0, HID_BTN_B = 1u << 1, HID_BTN_X = 1u << 2, HID_BTN_Y = 1u << 3,
    HID_BTN_STICKL = 1u << 4, HID_BTN_STICKR = 1u << 5,
    HID_BTN_L = 1u << 6, HID_BTN_R = 1u << 7,
    HID_BTN_ZL = 1u << 8, HID_BTN_ZR = 1u << 9,
    HID_BTN_PLUS = 1u << 10, HID_BTN_MINUS = 1u << 11,
    HID_BTN_LEFT = 1u << 12, HID_BTN_UP = 1u << 13, HID_BTN_RIGHT = 1u << 14, HID_BTN_DOWN = 1u << 15,
    HID_BTN_STICKL_LEFT = 1u << 16, HID_BTN_STICKL_UP = 1u << 17,
    HID_BTN_STICKL_RIGHT = 1u << 18, HID_BTN_STICKL_DOWN = 1u << 19,
    HID_BTN_STICKR_LEFT = 1u << 20, HID_BTN_STICKR_UP = 1u << 21,
    HID_BTN_STICKR_RIGHT = 1u << 22, HID_BTN_STICKR_DOWN = 1u << 23,
};

// One coherent sample. Sticks are -32768..32767 with the dead zone already applied (the producer applies it). sampling
// is the publish's monotonic number (the layer sets it: 1, 2, ...; 0 before the first publish), so a reader can tell how
// fresh the sample it got is.
typedef struct HidHostInput {
    uint64_t sampling;
    uint64_t buttons;      // HID_BTN_* bits
    int32_t lx, ly, rx, ry;
} HidHostInput;

// The window thread publishes the newest sample; ov_hid.c reads it. read returns the last published sample (zeros until
// the first publish), each call seeing one whole sample, never a mix of two.
void hid_host_input_publish(const HidHostInput *in);
void hid_host_input_read(HidHostInput *out);

// The config menu's swallow: while suppressed, read returns a zero state (no buttons, no sticks) whatever the producer
// last published. The window thread turns it on while the menu is open and off when it closes, so the game sees no input.
void hid_host_input_suppress(int on);

// A center dead zone: |v| <= dead reads as 0, otherwise v is kept. dead is in the same 0..32767 scale as v.
int32_t hid_host_input_deadzone(int32_t v, int32_t dead);

// Convert SDL's negative-up vertical axis to HID's positive-up axis, saturating at the positive endpoint.
int32_t hid_host_input_axis_y(int16_t v, int32_t dead);

// ---- the remap (oot2.ini's [input] keys, set through the settings store)
// The console buttons the remap names, in the order the store's input_pad_* and input_key_* keys number them (a, b, x,
// y, l, r, zl, zr, plus, minus, left, up, right, down, stickl, stickr); index i's nn::hid bit is hid_remap_button_bit[i].
enum { HID_REMAP_BUTTONS = 16 };

// The gamepad's sources, in the order the input_pad_* keys name them: SDL's position-neutral face buttons (SOUTH is the
// lower button: B on a Nintendo-style pad), the shoulders, Back, Start, the D-pad, the stick clicks, and the two
// triggers (which press past half travel). Guide is reserved for the menu. none: the console button has no pad source.
enum HidPadSource {
    HID_PAD_SOUTH = 0, HID_PAD_EAST, HID_PAD_WEST, HID_PAD_NORTH,
    HID_PAD_LEFT_SHOULDER, HID_PAD_RIGHT_SHOULDER, HID_PAD_BACK, HID_PAD_START,
    HID_PAD_DPAD_UP, HID_PAD_DPAD_DOWN, HID_PAD_DPAD_LEFT, HID_PAD_DPAD_RIGHT,
    HID_PAD_LEFT_STICK, HID_PAD_RIGHT_STICK, HID_PAD_LEFT_TRIGGER, HID_PAD_RIGHT_TRIGGER,
    HID_PAD_SOURCES, HID_PAD_NONE = HID_PAD_SOURCES,
};

// The keyboard's sources, in the order the input_key_* keys name them: the arrows, the 26 letters, and the named keys.
// The menu's own keys (F1, Escape) are not offered. none: the console button has no keyboard source.
enum HidKeySource {
    HID_KEY_UP = 0, HID_KEY_DOWN, HID_KEY_LEFT, HID_KEY_RIGHT,
    HID_KEY_A, HID_KEY_B, HID_KEY_C, HID_KEY_D, HID_KEY_E, HID_KEY_F, HID_KEY_G, HID_KEY_H, HID_KEY_I,
    HID_KEY_J, HID_KEY_K, HID_KEY_L, HID_KEY_M, HID_KEY_N, HID_KEY_O, HID_KEY_P, HID_KEY_Q, HID_KEY_R,
    HID_KEY_S, HID_KEY_T, HID_KEY_U, HID_KEY_V, HID_KEY_W, HID_KEY_X, HID_KEY_Y, HID_KEY_Z,
    HID_KEY_RETURN, HID_KEY_SPACE, HID_KEY_TAB, HID_KEY_BACKSPACE,
    HID_KEY_LSHIFT, HID_KEY_RSHIFT, HID_KEY_LCTRL, HID_KEY_RCTRL,
    HID_KEY_SOURCES, HID_KEY_NONE = HID_KEY_SOURCES,
};

// Physical WASD and IJKL control the left and right sticks while no gamepad is connected. Opposite directions cancel;
// these fixed stick keys are independent of button remapping and can also activate remapped buttons.
void hid_keyboard_sticks(uint64_t key_down, HidHostInput *out);

// One remap: for console button i, pad[i] is its gamepad source and key[i] its keyboard source, each a HidPadSource /
// HidKeySource value; a value at or past the NONE sentinel (or any other the store does not give) means no source. The
// window thread reads these out of the store's input_pad_* and input_key_* keys (live: every sample), whose defaults are
// the fixed mapping the layer first shipped with (the Nintendo-style pad, and the arrows with Return, X and the
// modifiers on the keyboard); the keyboard's sources are read only while no gamepad is connected.
typedef struct HidRemap {
    uint8_t pad[HID_REMAP_BUTTONS];
    uint8_t key[HID_REMAP_BUTTONS];
} HidRemap;

// The nn::hid bit of each console button, in the remap's order (HID_BTN_A, HID_BTN_B, ..., HID_BTN_STICKR).
extern const uint64_t hid_remap_button_bit[HID_REMAP_BUTTONS];

// The buttons of one sample: bit i of pad_down / key_down says source i (in HidPadSource / HidKeySource order) is
// pressed, and a console button is on when its pad source or its keyboard source is, whatever its other source says.
// The keyboard mask is 64-bit because it has more than 32 sources. Pure: the window thread builds the two bitmasks from
// SDL and applies the remap here, so a test can stand in for it.
uint64_t hid_remap_buttons(const HidRemap *map, uint32_t pad_down, uint64_t key_down);

// nn::hid's vibration value as the window thread plays it (SDL_RumbleGamepad): low and high are 0..0xFFFF, and
// duration_ms how long the rumble stays on. ov_hid.c publishes one request per SendVibrationValue; the window thread
// takes it with hid_rumble_take and plays it on the open gamepad. One pending request is kept: a newer value replaces an
// unread older one (the game sends one every frame while it rumbles, and only the newest matters).
typedef struct HidRumble {
    uint16_t low, high;
    uint32_t duration_ms;
} HidRumble;
// SDL has one motor per frequency rather than one per guest side; retain the strongest amplitude of either side.
HidRumble hid_rumble_mix(float left_low, float left_high, float right_low, float right_high);
// Optional window-loop notification, called on the publishing thread under the rumble lock: only post a wake event;
// do not call back into the rumble API. Clearing it waits for any notification in flight, so teardown is safe.
void hid_rumble_set_wake(void (*wake)(void));
void hid_rumble_publish(const HidRumble *r);
int hid_rumble_take(HidRumble *out);   // 1 with *out when a request is pending, else 0
