// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The host motion layer: what the window thread publishes from the pad's gyroscope and accelerometer, and what the
// six-axis reads of nn::hid answer from. The header names no SDL type, so a test can stand in for the producer with a
// synthetic stream of samples.
//
// Units and axes are those of the console's six-axis state (nn::hid::SixAxisSensorState): acceleration in G, as gravity
// reads (a controller lying face up reads z = -1), angular velocity in revolutions per second, angle in revolutions, and
// direction the controller's three axes in a reference frame whose z is up. The controller's axes are x to the right, y
// toward the top edge (away from the player) and z out of the face.
//
// The layer is adapted from DKB Recomp's runtime/motion_fusion.c and the axis map of its runtime/input.c (MIT and
// GPL-3.0-or-later; the origin is MacNativeGames): the header of runtime/hos/hid_motion.c names the commits and the
// changes.
//
// The conversion from SDL's gamepad sensors follows the SDL documentation of the sensor axes (SDL_sensor.h: the
// accelerometer reads m/s^2 including gravity, the gyroscope rad/s, counter-clockwise positive) and the axis map that two
// donor runtimes use for the same console frame (dkb and jaigner-oot, both derived from MacNativeGames; nxnative states
// the same frame for its own sources): SDL's x is the console's x, SDL's y (up out of the face when the pad lies flat) is
// the console's z, and SDL's z (toward the player) is the console's -y. The donors agree on the console frame and on
// revolutions per second. The SDL header's wording for gamepads ("held in front of you") is ambiguous about which way y
// points; the map here is the one the donors use on real pads, and it has not been checked against a pad here.
#pragma once
#include <stdint.h>

// One six-axis sample as nn::hid lays it out in SixAxisSensorState (without the attribute word).
typedef struct HidMotionState {
    int64_t delta_ns, sampling;
    float acc[3], gyro[3], angle[3], dir[9];
} HidMotionState;

enum { HID_MOTION_RING = 64 };

// SDL's accelerometer (m/s^2) and gyroscope (rad/s) of a gamepad into the console's acceleration (G) and angular
// velocity (revolutions per second). Pure.
void hid_motion_from_sdl(const float accel_ms2[3], const float gyro_rads[3], float acc_g[3], float gyro_rps[3]);

// The producer: one sample, dt_ns after the one before it (0 or more than 50 ms: the nominal 4 ms), taken at host time
// now_ns. The orientation is integrated from the gyroscope and, while the controller is not being accelerated (|a| within
// 15 % of 1 G), pulled toward the measured gravity; yaw has no reference and drifts with the gyroscope's bias. The newest
// HID_MOTION_RING samples are kept.
void hid_motion_feed(const float acc_g[3], const float gyro_rps[3], uint64_t dt_ns, uint64_t now_ns);

// The reader: up to max states, newest first, with sampling numbers that count the samples from 1, for the states
// since the last restart. Returns how many, 0 when there are none: no sample since the restart, none for 250 ms before
// now_ns (the pad stopped sending or went away), the layer is suppressed, or the producer cleared it.
int hid_motion_read(HidMotionState *out, int max, uint64_t now_ns);

// StartSixAxisSensor: the next sample starts a fresh frame (angle 0, direction from the measured gravity); the states
// before it are not read again. Sampling numbers go on.
void hid_motion_restart(void);
// The pad is gone or has no sensors: drop every sample.
void hid_motion_clear(void);
// The game's wish: set between StartSixAxisSensor and StopSixAxisSensor. The producer turns the pad's sensors on only
// while it is set.
void hid_motion_set_wanted(int on);
int hid_motion_wanted(void);
// The config menu's swallow, as for the buttons (hid_host_input_suppress): while on, reads give nothing.
void hid_motion_suppress(int on);
