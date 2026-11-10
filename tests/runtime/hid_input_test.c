// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Host tests of the input layer's pure pieces (no SDL, no GPU): the dead zone, the published snapshot, and the button
// bits. Driven by tests/test_runtime_hid_input.py.
#include <stdio.h>
#include <string.h>

#include <math.h>

#include "oot2/hid_input.h"
#include "oot2/hid_motion.h"

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

// ---- the motion layer, driven by synthetic sensor samples (no pad, no SDL)
#define NEAR(a, b, eps) (fabs((double)(a) - (double)(b)) <= (eps))
#define TWO_PI 6.283185307179586
#define G 9.80665f
#define MS 1000000ull

static int case_motion_convert(void) {
    float acc[3], gyro[3];
    const float still[3] = {0, 0, 0};
    // a pad lying face up: SDL's accelerometer reads +1 g along its y; the console reads gravity, -1 g along its z
    const float flat[3] = {0, G, 0};
    hid_motion_from_sdl(flat, still, acc, gyro);
    CHECK(NEAR(acc[0], 0, 1e-6) && NEAR(acc[1], 0, 1e-6) && NEAR(acc[2], -1, 1e-6));
    // SDL x is the console's x with the sign of gravity, SDL z (toward the player) is the console's -y
    const float on_x[3] = {G, 0, 0}, on_z[3] = {0, 0, G};
    hid_motion_from_sdl(on_x, still, acc, gyro);
    CHECK(NEAR(acc[0], -1, 1e-6) && NEAR(acc[1], 0, 1e-6) && NEAR(acc[2], 0, 1e-6));
    hid_motion_from_sdl(on_z, still, acc, gyro);
    CHECK(NEAR(acc[0], 0, 1e-6) && NEAR(acc[1], 1, 1e-6) && NEAR(acc[2], 0, 1e-6));
    // one turn a second about each SDL axis (rad/s in) is one revolution a second about the console's axis (rev/s out)
    const float turn_x[3] = {(float)TWO_PI, 0, 0}, turn_y[3] = {0, (float)TWO_PI, 0}, turn_z[3] = {0, 0, (float)TWO_PI};
    hid_motion_from_sdl(flat, turn_x, acc, gyro);
    CHECK(NEAR(gyro[0], 1, 1e-6) && NEAR(gyro[1], 0, 1e-6) && NEAR(gyro[2], 0, 1e-6));
    hid_motion_from_sdl(flat, turn_y, acc, gyro);   // yaw of the pad flat on a table
    CHECK(NEAR(gyro[0], 0, 1e-6) && NEAR(gyro[1], 0, 1e-6) && NEAR(gyro[2], 1, 1e-6));
    hid_motion_from_sdl(flat, turn_z, acc, gyro);
    CHECK(NEAR(gyro[0], 0, 1e-6) && NEAR(gyro[1], -1, 1e-6) && NEAR(gyro[2], 0, 1e-6));
    printf("ok motion_convert\n");
    return 0;
}

static int case_motion_fusion(void) {
    HidMotionState s[HID_MOTION_RING + 8];
    const float flat[3] = {0, 0, -1}, still[3] = {0, 0, 0}, yaw[3] = {0, 0, 0.25f};
    uint64_t now = 1000 * MS;
    CHECK(hid_motion_read(s, 4, now) == 0);   // nothing fed yet
    hid_motion_feed(flat, still, 0, now);     // dt 0: the nominal 4 ms
    CHECK(hid_motion_read(s, 4, now) == 1);
    CHECK(s[0].sampling == 1 && s[0].delta_ns == 4 * (int64_t)MS);
    CHECK(NEAR(s[0].acc[2], -1, 1e-6) && NEAR(s[0].gyro[2], 0, 1e-6) && NEAR(s[0].angle[2], 0, 1e-6));
    for (int i = 0; i < 9; i++) CHECK(NEAR(s[0].dir[i], (i % 4 == 0) ? 1 : 0, 1e-5));   // identity: flat, facing the frame
    // a quarter revolution a second about the face's normal for one second: angle 0.25 revolutions, the controller's x
    // axis has swung from the reference x to the reference y
    for (int i = 0; i < 250; i++) {
        now += 4 * MS;
        hid_motion_feed(flat, yaw, 4 * MS, now);
    }
    CHECK(hid_motion_read(s, 1, now) == 1);
    CHECK(s[0].sampling == 251 && NEAR(s[0].gyro[2], 0.25, 1e-6));
    CHECK(NEAR(s[0].angle[2], 0.25, 1e-4) && NEAR(s[0].angle[0], 0, 1e-6) && NEAR(s[0].angle[1], 0, 1e-6));
    CHECK(NEAR(s[0].dir[0], 0, 2e-3) && NEAR(s[0].dir[1], 1, 2e-3) && NEAR(s[0].dir[2], 0, 2e-3));
    CHECK(NEAR(s[0].dir[3], -1, 2e-3) && NEAR(s[0].dir[4], 0, 2e-3) && NEAR(s[0].dir[5], 0, 2e-3));
    CHECK(NEAR(s[0].dir[6], 0, 2e-3) && NEAR(s[0].dir[7], 0, 2e-3) && NEAR(s[0].dir[8], 1, 2e-3));
    // the ring: newest first, sampling numbers counting down by one, at most HID_MOTION_RING of them
    int n = hid_motion_read(s, HID_MOTION_RING + 8, now);
    CHECK(n == HID_MOTION_RING);
    for (int i = 0; i < n; i++) CHECK(s[i].sampling == 251 - i);
    // a restart: the states before it are not read, the next sample starts a fresh frame, the numbers go on
    hid_motion_restart();
    CHECK(hid_motion_read(s, 4, now) == 0);
    const float tilted[3] = {0.5f, 0, -0.8660254f};
    now += 4 * MS;
    hid_motion_feed(tilted, still, 4 * MS, now);
    CHECK(hid_motion_read(s, 4, now) == 1 && s[0].sampling == 252);
    CHECK(NEAR(s[0].angle[2], 0, 1e-6));
    // the fresh frame's up is the measured gravity's opposite: the controller's axes' reference z components are -acc
    CHECK(NEAR(s[0].dir[2], -0.5, 1e-4) && NEAR(s[0].dir[5], 0, 1e-4) && NEAR(s[0].dir[8], 0.8660254, 1e-4));
    // gravity pulls a drifted tilt back: flat again for twenty seconds with no rotation, the orientation follows
    for (int i = 0; i < 5000; i++) {
        now += 4 * MS;
        hid_motion_feed(flat, still, 4 * MS, now);
    }
    CHECK(hid_motion_read(s, 1, now) == 1);
    CHECK(NEAR(s[0].dir[2], 0, 2e-3) && NEAR(s[0].dir[5], 0, 2e-3) && NEAR(s[0].dir[8], 1, 2e-3));
    printf("ok motion_fusion\n");
    return 0;
}

static int case_motion_gates(void) {
    HidMotionState s[4];
    const float flat[3] = {0, 0, -1}, still[3] = {0, 0, 0};
    uint64_t now = 5000 * MS;
    hid_motion_feed(flat, still, 4 * MS, now);
    CHECK(hid_motion_read(s, 4, now + 249 * MS) == 1);   // within 250 ms of the last sample
    CHECK(hid_motion_read(s, 4, now + 250 * MS) == 0);   // the pad stopped sending: no motion
    CHECK(hid_motion_read(s, 0, now) == 0);
    CHECK(hid_motion_read(s, -1, now) == 0);
    CHECK(hid_motion_read(s, 4, now - 1) == 1);          // a clock a little behind the producer's is not stale
    hid_motion_suppress(1);                              // the menu's swallow
    CHECK(hid_motion_read(s, 4, now) == 0);
    hid_motion_suppress(0);
    CHECK(hid_motion_read(s, 4, now) == 1);
    hid_motion_clear();                                  // the pad went away
    CHECK(hid_motion_read(s, 4, now) == 0);
    hid_motion_feed(flat, still, 4 * MS, now + MS);      // a new pad: its samples only
    CHECK(hid_motion_read(s, 4, now + MS) == 1 && s[0].sampling == 2);
    CHECK(hid_motion_wanted() == 0);
    hid_motion_set_wanted(1);
    CHECK(hid_motion_wanted() == 1);
    hid_motion_set_wanted(0);
    CHECK(hid_motion_wanted() == 0);
    printf("ok motion_gates\n");
    return 0;
}

// Two turns in a row compose in the controller's own frame: a quarter turn about the controller's x axis, then a quarter
// turn about its (new) z axis, leaves the controller's x axis pointing up. Free fall (no gravity reading) so that nothing
// pulls the tilt back.
static int case_motion_compose(void) {
    HidMotionState s[1];
    const float fall[3] = {0, 0, 0}, pitch[3] = {0.25f, 0, 0}, yaw[3] = {0, 0, 0.25f};
    uint64_t now = 1000 * MS;
    hid_motion_feed(fall, pitch, 4 * MS, now);   // the first sample starts the frame, flat
    for (int i = 0; i < 250; i++) {
        now += 4 * MS;
        hid_motion_feed(fall, pitch, 4 * MS, now);
    }
    for (int i = 0; i < 250; i++) {
        now += 4 * MS;
        hid_motion_feed(fall, yaw, 4 * MS, now);
    }
    CHECK(hid_motion_read(s, 1, now) == 1);
    CHECK(NEAR(s[0].dir[0], 0, 5e-3) && NEAR(s[0].dir[1], 0, 5e-3) && NEAR(s[0].dir[2], 1, 5e-3));    // x points up
    CHECK(NEAR(s[0].dir[3], -1, 5e-3) && NEAR(s[0].dir[4], 0, 5e-3) && NEAR(s[0].dir[5], 0, 5e-3));   // y points to -x
    CHECK(NEAR(s[0].dir[6], 0, 5e-3) && NEAR(s[0].dir[7], -1, 5e-3) && NEAR(s[0].dir[8], 0, 5e-3));   // z points to -y
    printf("ok motion_compose\n");
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
        {"motion_convert", case_motion_convert}, {"motion_fusion", case_motion_fusion},
        {"motion_gates", case_motion_gates}, {"motion_compose", case_motion_compose},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(argv[1], cases[i].name)) return cases[i].fn();
    fprintf(stderr, "hid_input_test: no case %s\n", argv[1]);
    return 2;
}