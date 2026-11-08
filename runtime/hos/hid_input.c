// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The host input layer (oot2/hid_input.h): one published snapshot, read atomically by whoever answers nn::hid. The
// window thread is the only publisher; nn::hid's answers are the readers. A reader gets one whole sample, never a mix.
#include <stdatomic.h>

#include "oot2/hid_input.h"

static HidHostInput g_state;
static uint64_t g_sampling;
static int g_suppress;
static atomic_flag g_lock = ATOMIC_FLAG_INIT;

static HidRumble g_rumble;
static int g_rumble_pending;
static void (*g_rumble_wake)(void);
static atomic_flag g_rumble_lock = ATOMIC_FLAG_INIT;

void hid_host_input_publish(const HidHostInput *in) {
    while (atomic_flag_test_and_set_explicit(&g_lock, memory_order_acquire)) {}
    g_sampling++;
    g_state = *in;
    g_state.sampling = g_sampling;
    atomic_flag_clear_explicit(&g_lock, memory_order_release);
}

void hid_host_input_read(HidHostInput *out) {
    while (atomic_flag_test_and_set_explicit(&g_lock, memory_order_acquire)) {}
    *out = g_suppress ? (HidHostInput){0} : g_state;
    atomic_flag_clear_explicit(&g_lock, memory_order_release);
}

void hid_host_input_suppress(int on) {
    while (atomic_flag_test_and_set_explicit(&g_lock, memory_order_acquire)) {}
    g_suppress = on;
    atomic_flag_clear_explicit(&g_lock, memory_order_release);
}

int32_t hid_host_input_deadzone(int32_t v, int32_t dead) {
    return (v > dead || v < -dead) ? v : 0;
}

int32_t hid_host_input_axis_y(int16_t v, int32_t dead) {
    int32_t y = -(int32_t)v;
    if (y > 32767) y = 32767;
    return hid_host_input_deadzone(y, dead);
}

static int32_t key_axis(uint64_t keys, int negative, int positive) {
    int lo = (int)((keys >> negative) & 1u);
    int hi = (int)((keys >> positive) & 1u);
    return (hi - lo) * 32767;
}

void hid_keyboard_sticks(uint64_t key_down, HidHostInput *out) {
    out->lx = key_axis(key_down, HID_KEY_A, HID_KEY_D);
    out->ly = key_axis(key_down, HID_KEY_S, HID_KEY_W);
    out->rx = key_axis(key_down, HID_KEY_J, HID_KEY_L);
    out->ry = key_axis(key_down, HID_KEY_K, HID_KEY_I);
}

// The remap's console-button order and the nn::hid bit of each (HID_BTN_A, HID_BTN_B, ..., HID_BTN_STICKR).
const uint64_t hid_remap_button_bit[HID_REMAP_BUTTONS] = {
    HID_BTN_A, HID_BTN_B, HID_BTN_X, HID_BTN_Y,
    HID_BTN_L, HID_BTN_R, HID_BTN_ZL, HID_BTN_ZR,
    HID_BTN_PLUS, HID_BTN_MINUS,
    HID_BTN_LEFT, HID_BTN_UP, HID_BTN_RIGHT, HID_BTN_DOWN,
    HID_BTN_STICKL, HID_BTN_STICKR,
};

uint64_t hid_remap_buttons(const HidRemap *map, uint32_t pad_down, uint64_t key_down) {
    uint64_t b = 0;
    for (int i = 0; i < HID_REMAP_BUTTONS; i++) {
        if (map->pad[i] < HID_PAD_NONE && (pad_down >> map->pad[i]) & 1u) b |= hid_remap_button_bit[i];
        if (map->key[i] < HID_KEY_NONE && (key_down >> map->key[i]) & 1u) b |= hid_remap_button_bit[i];
    }
    return b;
}

static uint16_t rumble_strength(float amp) {
    if (!(amp > 0.0f)) return 0;
    if (amp >= 1.0f) return 0xFFFFu;
    return (uint16_t)(amp * 65535.0f);
}

HidRumble hid_rumble_mix(float left_low, float left_high, float right_low, float right_high) {
    uint16_t ll = rumble_strength(left_low), lh = rumble_strength(left_high);
    uint16_t rl = rumble_strength(right_low), rh = rumble_strength(right_high);
    return (HidRumble){ll > rl ? ll : rl, lh > rh ? lh : rh, 100};
}

void hid_rumble_set_wake(void (*wake)(void)) {
    while (atomic_flag_test_and_set_explicit(&g_rumble_lock, memory_order_acquire)) {}
    g_rumble_wake = wake;
    atomic_flag_clear_explicit(&g_rumble_lock, memory_order_release);
}

void hid_rumble_publish(const HidRumble *r) {
    while (atomic_flag_test_and_set_explicit(&g_rumble_lock, memory_order_acquire)) {}
    g_rumble = *r;
    g_rumble_pending = 1;
    if (g_rumble_wake) g_rumble_wake();
    atomic_flag_clear_explicit(&g_rumble_lock, memory_order_release);
}

int hid_rumble_take(HidRumble *out) {
    int got = 0;
    while (atomic_flag_test_and_set_explicit(&g_rumble_lock, memory_order_acquire)) {}
    if (g_rumble_pending) {
        *out = g_rumble;
        g_rumble_pending = 0;
        got = 1;
    }
    atomic_flag_clear_explicit(&g_rumble_lock, memory_order_release);
    return got;
}
