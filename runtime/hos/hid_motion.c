// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The host motion layer (oot2/hid_motion.h): the conversion from SDL's gamepad sensors to the console's six-axis units,
// the orientation the sensor fusion keeps, and the ring of states the window thread writes and nn::hid reads. The window
// thread is the only producer; the readers are guest threads. The fusion is a complementary filter on a rotation
// matrix: the gyroscope rotates it, the measured gravity pulls its tilt back.
//
// Adapted from DKB Recomp runtime/motion_fusion.c at c5d5e7c23d0a2c7f94a32a6ccfadda6b138b82ce, whose header says it is
// adapted from MacNativeGames games/totk/runtime/motion_fusion.c at b75be8cfa704df3701e86130c8ced268e79d6b11 (MIT,
// Copyright (c) 2026 MacNativeGames contributors): the fusion (norm3, cross3, orthonormalize, align_to, integrate), the
// feed, the read and the restart. The axis map of hid_motion_from_sdl is DKB Recomp's runtime/input.c
// (input_motion_axes; no licence header: that repository's GPL-3.0-or-later) at the same commit. Changed here: one
// layer instead of an instance per controller, under a spin lock instead of the donor's SRW lock; the conversion gives
// revolutions per second itself; the reader takes the host time and answers nothing after 250 ms without a sample; a
// clear, the game's wish (hid_motion_wanted) and the config menu's suppress are added; the names follow this runtime.
#include <math.h>
#include <stdatomic.h>
#include <string.h>

#include "oot2/hid_motion.h"

#define STALE_NS 250000000ull     // no sample for this long: no motion
#define NOMINAL_DT_NS 4000000ull  // a pad that reports at 250 Hz
#define MAX_DT_NS 50000000ull
#define TILT_GAIN 0.5             // 1/s: the gravity pull, about two seconds to settle
#define G_BAND 0.15               // |a| within this of 1 G: the controller is not being accelerated
#define STANDARD_GRAVITY 9.80665
#define TWO_PI 6.283185307179586

static atomic_flag g_lock = ATOMIC_FLAG_INIT;
static HidMotionState g_ring[HID_MOTION_RING];
static uint64_t g_produced, g_first, g_last_ns;
static double g_R[3][3];   // controller to reference: column i is the controller's axis i in the reference frame
static double g_angle[3];
static int g_restart = 1, g_have, g_suppress;
static atomic_int g_wanted;

static void lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_lock, memory_order_acquire)) {}
}
static void unlock(void) { atomic_flag_clear_explicit(&g_lock, memory_order_release); }

void hid_motion_from_sdl(const float a[3], const float g[3], float acc[3], float gyro[3]) {
    // SDL (x right, y up out of the face, z toward the player) to the console (x right, y away from the player, z out of
    // the face) is a rotation, so the angular velocity keeps its sense; the accelerometer reads the push against gravity,
    // the console's acceleration reads gravity itself, hence the sign.
    acc[0] = (float)(-a[0] / STANDARD_GRAVITY);
    acc[1] = (float)(a[2] / STANDARD_GRAVITY);
    acc[2] = (float)(-a[1] / STANDARD_GRAVITY);
    gyro[0] = (float)(g[0] / TWO_PI);
    gyro[1] = (float)(-g[2] / TWO_PI);
    gyro[2] = (float)(g[1] / TWO_PI);
}

static double norm3(const double v[3]) { return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

// Keep row 2 (the reference up axis in controller coordinates), make row 0 perpendicular to it, row 1 their cross
// product: R stays a rotation however many steps have rounded it.
static void orthonormalize(double R[3][3]) {
    double n = norm3(R[2]);
    for (int j = 0; j < 3; j++) R[2][j] /= n;
    double d = R[0][0] * R[2][0] + R[0][1] * R[2][1] + R[0][2] * R[2][2];
    for (int j = 0; j < 3; j++) R[0][j] -= d * R[2][j];
    n = norm3(R[0]);
    for (int j = 0; j < 3; j++) R[0][j] /= n;
    cross3(R[2], R[0], R[1]);
}

// A fresh orientation: up against the measured gravity, the reference x as near the controller's x as that allows.
static void align_to(double R[3][3], const double acc[3]) {
    double n = norm3(acc), up[3] = {0, 0, 1}, x[3] = {1, 0, 0};
    if (n > 0.5)
        for (int j = 0; j < 3; j++) up[j] = -acc[j] / n;
    if (fabs(up[0]) > 0.9) x[0] = 0, x[1] = 1;
    memcpy(R[2], up, sizeof up);
    memcpy(R[0], x, sizeof x);
    orthonormalize(R);
}

// One step: R_new = R * exp([w] dt), w the body-frame rate (rad/s) plus the gravity pull. The controller turns by the
// rotation in its own coordinates, so the rotation multiplies from the right.
static void integrate(double R[3][3], const double rate[3], const double acc[3], double dt) {
    double w[3] = {rate[0], rate[1], rate[2]}, n = norm3(acc);
    if (fabs(n - 1) < G_BAND) {
        // Gravity's direction in controller coordinates, measured and as R has it; turning the controller by
        // measured x estimated brings the second toward the first.
        double measured[3] = {acc[0] / n, acc[1] / n, acc[2] / n};
        double estimated[3] = {-R[2][0], -R[2][1], -R[2][2]}, e[3];
        cross3(measured, estimated, e);
        for (int j = 0; j < 3; j++) w[j] += TILT_GAIN * e[j];
    }
    double speed = norm3(w), theta = speed * dt;
    if (theta < 1e-12) return;
    double k[3] = {w[0] / speed, w[1] / speed, w[2] / speed}, s = sin(theta), v = 1 - cos(theta);
    double K[3][3] = {{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}}, E[3][3], out[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kk = 0;
            for (int q = 0; q < 3; q++) kk += K[i][q] * K[q][j];
            E[i][j] = (i == j) + s * K[i][j] + v * kk;
        }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double sum = 0;
            for (int q = 0; q < 3; q++) sum += R[i][q] * E[q][j];
            out[i][j] = sum;
        }
    memcpy(R, out, sizeof out);
    orthonormalize(R);
}

void hid_motion_feed(const float acc_g[3], const float gyro_rps[3], uint64_t dt_ns, uint64_t now_ns) {
    if (dt_ns == 0 || dt_ns > MAX_DT_NS) dt_ns = NOMINAL_DT_NS;
    double dt = (double)dt_ns / 1e9, acc[3], rate[3];
    for (int i = 0; i < 3; i++) {
        acc[i] = acc_g[i];
        rate[i] = (double)gyro_rps[i] * TWO_PI;
    }
    lock();
    if (g_restart) {
        align_to(g_R, acc);
        memset(g_angle, 0, sizeof g_angle);
        g_restart = 0;
    } else {
        integrate(g_R, rate, acc, dt);
    }
    HidMotionState *s = &g_ring[g_produced % HID_MOTION_RING];
    *s = (HidMotionState){.delta_ns = (int64_t)dt_ns, .sampling = (int64_t)g_produced + 1};
    for (int i = 0; i < 3; i++) {
        g_angle[i] += (double)gyro_rps[i] * dt;
        s->acc[i] = acc_g[i];
        s->gyro[i] = gyro_rps[i];
        s->angle[i] = (float)g_angle[i];
        for (int j = 0; j < 3; j++) s->dir[i * 3 + j] = (float)g_R[j][i];   // axis i, reference component j
    }
    g_produced++;
    g_last_ns = now_ns;
    g_have = 1;
    unlock();
}

int hid_motion_read(HidMotionState *out, int max, uint64_t now_ns) {
    if (max <= 0) return 0;
    int n = 0;
    lock();
    if (!g_suppress && g_have && g_produced > g_first && (now_ns < g_last_ns || now_ns - g_last_ns < STALE_NS)) {
        uint64_t have = g_produced - g_first;
        if (have > HID_MOTION_RING) have = HID_MOTION_RING;
        n = (int)(have < (uint64_t)max ? have : (uint64_t)max);
        for (int i = 0; i < n; i++) out[i] = g_ring[(g_produced - 1 - (uint64_t)i) % HID_MOTION_RING];
    }
    unlock();
    return n;
}

void hid_motion_restart(void) {
    lock();
    g_restart = 1;
    g_first = g_produced;
    memset(g_angle, 0, sizeof g_angle);
    unlock();
}

void hid_motion_clear(void) {
    lock();
    g_have = 0;
    g_restart = 1;
    g_first = g_produced;
    unlock();
}

void hid_motion_set_wanted(int on) { atomic_store(&g_wanted, on != 0); }
int hid_motion_wanted(void) { return atomic_load(&g_wanted); }

void hid_motion_suppress(int on) {
    lock();
    g_suppress = on != 0;
    unlock();
}
