// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Host tests of the input layer's pure pieces (no SDL, no GPU): the dead zone, the published snapshot, and the button
// bits. Driven by tests/test_runtime_hid_input.py.
#include <stdio.h>
#include <string.h>

#include "oot2/hid_input.h"

#define CHECK(x)                                        \
    do {                                                \
        if (!(x)) {                                     \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #x);   \
            return 1;                                   \
        }                                               \
    } while (0)

static int case_deadzone(void) {
    CHECK(hid_host_input_deadzone(0, 1000) == 0);
    CHECK(hid_host_input_deadzone(1000, 1000) == 0);
    CHECK(hid_host_input_deadzone(-1000, 1000) == 0);
    CHECK(hid_host_input_deadzone(1001, 1000) == 1001);
    CHECK(hid_host_input_deadzone(-1001, 1000) == -1001);
    CHECK(hid_host_input_deadzone(32767, 0) == 32767);
    CHECK(hid_host_input_deadzone(-32768, 0) == -32768);
    CHECK(hid_host_input_axis_y(-32768, 0) == 32767);
    CHECK(hid_host_input_axis_y(32767, 0) == -32767);
    CHECK(hid_host_input_axis_y(-1000, 1000) == 0);
    CHECK(hid_host_input_axis_y(1001, 1000) == -1001);
    printf("ok deadzone\n");
    return 0;
}

static int case_publish(void) {
    HidHostInput in = {.buttons = HID_BTN_A | HID_BTN_UP | HID_BTN_ZR,
                       .lx = -30000, .ly = 20000, .rx = 15, .ry = -1};
    hid_host_input_publish(&in);
    HidHostInput out = {0};
    hid_host_input_read(&out);   // one whole sample, no mix
    CHECK(out.sampling == 1);
    CHECK(out.buttons == in.buttons);
    CHECK(out.lx == in.lx && out.ly == in.ly && out.rx == in.rx && out.ry == in.ry);
    HidHostInput next = {.buttons = HID_BTN_B, .lx = 1, .ly = 2, .rx = 3, .ry = 4};
    hid_host_input_publish(&next);
    hid_host_input_read(&out);
    CHECK(out.sampling == 2);   // the layer numbers each publish, monotonically
    CHECK(out.buttons == HID_BTN_B && out.lx == 1 && out.ly == 2 && out.rx == 3 && out.ry == 4);
    printf("ok publish\n");
    return 0;
}

static int rumble_wakes;
static void rumble_wake(void) { rumble_wakes++; }

static int case_rumble(void) {
    HidRumble r = {0};
    r = hid_rumble_mix(0, 1, 0, 0);
    CHECK(r.low == 0 && r.high == 65535 && r.duration_ms == 100);
    r = hid_rumble_mix(0, 0, 1, 0);
    CHECK(r.low == 65535 && r.high == 0);
    r = hid_rumble_mix(0.25f, 0.75f, 0.5f, 0.125f);
    CHECK(r.low == 32767 && r.high == 49151);
    r = hid_rumble_mix(-1.0f, 2.0f, 0, 0);
    CHECK(r.low == 0 && r.high == 65535);
    CHECK(hid_rumble_take(&r) == 0);   // nothing pending at first
    hid_rumble_set_wake(rumble_wake);
    HidRumble a = {0x1111, 0x2222, 80};
    hid_rumble_publish(&a);
    HidRumble b = {0x3333, 0x4444, 90};
    hid_rumble_publish(&b);   // the newest request replaces the unread older one
    CHECK(rumble_wakes == 2);
    CHECK(hid_rumble_take(&r) == 1);
    CHECK(r.low == 0x3333 && r.high == 0x4444 && r.duration_ms == 90);
    CHECK(hid_rumble_take(&r) == 0);   // taken, so empty again
    hid_rumble_set_wake(NULL);
    hid_rumble_publish(&a);
    CHECK(rumble_wakes == 2);   // teardown disables notifications without losing the request
    CHECK(hid_rumble_take(&r) == 1);
    CHECK(r.low == a.low && r.high == a.high && r.duration_ms == a.duration_ms);
    printf("ok rumble\n");
    return 0;
}

static int case_swallow(void) {
    HidHostInput in = {.buttons = HID_BTN_A, .lx = -1000, .ly = 1000};
    hid_host_input_publish(&in);
    HidHostInput out = {0};
    hid_host_input_read(&out);
    CHECK(out.buttons == HID_BTN_A && out.lx == -1000 && out.ly == 1000);
    hid_host_input_suppress(1);   // the menu hides the game input
    hid_host_input_read(&out);
    CHECK(out.buttons == 0 && out.lx == 0 && out.ly == 0 && out.rx == 0 && out.ry == 0);
    hid_host_input_suppress(0);
    hid_host_input_read(&out);    // the last published state is back
    CHECK(out.buttons == HID_BTN_A && out.lx == -1000 && out.ly == 1000);
    printf("ok swallow\n");
    return 0;
}

static int case_bits(void) {
    // Every button is its own bit, so a masked press is never read as another button.
    static const uint64_t bits[] = {HID_BTN_A,   HID_BTN_B,   HID_BTN_X,   HID_BTN_Y,   HID_BTN_STICKL,
                                    HID_BTN_STICKR, HID_BTN_L,   HID_BTN_R,   HID_BTN_ZL,  HID_BTN_ZR,
                                    HID_BTN_PLUS, HID_BTN_MINUS, HID_BTN_LEFT, HID_BTN_UP, HID_BTN_RIGHT,
                                    HID_BTN_DOWN, HID_BTN_STICKL_LEFT, HID_BTN_STICKL_UP, HID_BTN_STICKL_RIGHT,
                                    HID_BTN_STICKL_DOWN, HID_BTN_STICKR_LEFT, HID_BTN_STICKR_UP, HID_BTN_STICKR_RIGHT,
                                    HID_BTN_STICKR_DOWN};
    for (size_t i = 0; i < sizeof bits / sizeof *bits; i++)
        for (size_t j = i + 1; j < sizeof bits / sizeof *bits; j++) CHECK((bits[i] & bits[j]) == 0);
    printf("ok bits\n");
    return 0;
}

static int case_remap(void) {
    HidRemap map = {0};
    for (int i = 0; i < HID_REMAP_BUTTONS; i++) {
        map.pad[i] = HID_PAD_NONE;
        map.key[i] = HID_KEY_NONE;
    }
    map.pad[0] = HID_PAD_RIGHT_TRIGGER;
    map.key[1] = HID_KEY_RCTRL;
    uint64_t buttons = hid_remap_buttons(&map, 1u << HID_PAD_RIGHT_TRIGGER, UINT64_C(1) << HID_KEY_RCTRL);
    CHECK(buttons == (HID_BTN_A | HID_BTN_B));
    CHECK(HID_PAD_NONE == HID_PAD_SOURCES);
    CHECK(HID_KEY_NONE == HID_KEY_SOURCES);
    printf("ok remap\n");
    return 0;
}

static int case_keyboard_sticks(void) {
    HidHostInput out = {0};
    hid_keyboard_sticks(0, &out);
    CHECK(out.lx == 0 && out.ly == 0 && out.rx == 0 && out.ry == 0);
    hid_keyboard_sticks((UINT64_C(1) << HID_KEY_A) | (UINT64_C(1) << HID_KEY_W) |
                        (UINT64_C(1) << HID_KEY_L) | (UINT64_C(1) << HID_KEY_K), &out);
    CHECK(out.lx == -32767 && out.ly == 32767 && out.rx == 32767 && out.ry == -32767);
    hid_keyboard_sticks((UINT64_C(1) << HID_KEY_D) | (UINT64_C(1) << HID_KEY_S) |
                        (UINT64_C(1) << HID_KEY_J) | (UINT64_C(1) << HID_KEY_I), &out);
    CHECK(out.lx == 32767 && out.ly == -32767 && out.rx == -32767 && out.ry == 32767);
    hid_keyboard_sticks((UINT64_C(1) << HID_KEY_A) | (UINT64_C(1) << HID_KEY_D) |
                        (UINT64_C(1) << HID_KEY_I) | (UINT64_C(1) << HID_KEY_K), &out);
    CHECK(out.lx == 0 && out.ry == 0);
    printf("ok keyboard_sticks\n");
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: hid_input_test <case>\n");
        return 2;
    }

    static const struct { const char *name; int (*fn)(void); } cases[] = {
        {"deadzone", case_deadzone}, {"publish", case_publish}, {"bits", case_bits}, {"rumble", case_rumble},
        {"swallow", case_swallow}, {"remap", case_remap}, {"keyboard_sticks", case_keyboard_sticks},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(argv[1], cases[i].name)) return cases[i].fn();
    fprintf(stderr, "hid_input_test: no case %s\n", argv[1]);
    return 2;
}