// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// nn::hid, answered as a docked console with one full-key controller (player 1) and nothing pressed would answer it.
// This runtime has no host input layer yet: every state read is well-formed (a sampling number that grows with each
// read, buttons and sticks at zero, the controller marked connected, mouse, keyboard and touch screen with no input),
// the controller's two vibration devices and its six-axis sensor exist. The six-axis states are the host pad's gyroscope
// and accelerometer (oot2/hid_motion.h) while the game has started the sensor and the pad has them, and a controller
// lying still otherwise; the touch screen has no touches. Docked only: the handheld controller (id 0x20, style bit 1) is never
// connected here.
//
// Handles are the SDK's (nn::hid utility functions): byte 0 the style index (3 full-key, 4 handheld), byte 1 the
// controller id, byte 2 the device (vibration 0 left or 1 right, six-axis 2). The public functions call
// nn::hid::detail, which this file answers in place of the hid service, including the batches the SDK's
// VibrationTarget queues into detail::SendVibrationValues. State layouts follow libnx HID.
//
// Adapted from oot-pc-recomp runtime/ov_input.c at 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17 (GPL-3.0-or-later as a
// repository; the file's own header says it is adapted from MacNativeGames games/totk/runtime/input.c and
// hid_feedback.c at b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors; its
// Switch 2 and SDK 22 sections have no header of their own: the repository's GPL-3.0-or-later). Changed here: host
// functions follow runtime.h and carry a HOS_OVERRIDE note; guest memory is read with hos_in and written with hos_out,
// and a pointer the guest may not use is a guest stop naming the call; there is no gamepad, keyboard, rumble or motion
// code of the donor's (the host calls and the TOTK_* settings are gone; the full-key state and the six-axis states come
// from the host layers, oot2/hid_input.h and oot2/hid_motion.h, written here), a vibration value is remembered, and a
// six-axis read without host motion is a controller at rest with sampling numbers that grow, newest first; the
// handheld mode is not carried; IsVibrationPermitted answers the PermitVibration state.
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "hos.h"
#include "plat.h"

#include "oot2/hid_input.h"
#include "oot2/hid_motion.h"
#include "oot2/hid_replay.h"

typedef struct {
    int64_t sampling;
    uint64_t buttons;
    int32_t lx, ly, rx, ry;
    uint32_t attributes, pad;
} HidNpadState;   // every Npad state form
typedef struct { int64_t sampling; int32_t x, y, dx, dy, wx, wy; uint32_t buttons, attributes; } HidMouseState;
typedef struct { int64_t sampling; uint32_t modifiers, attributes; uint64_t keys[4]; } HidKeyboardState;
typedef struct { int8_t players; uint8_t pad[3]; uint32_t selected; } HidControllerSupportResult;
typedef struct {
    int64_t delta_ns, sampling;
    float acc[3], gyro[3], angle[3], dir[9];   // G, revolutions per second, revolutions, the controller's axes
    uint32_t attributes, reserved;             // attributes bit 0: connected
} HidSixAxisState;
// {amplitude low, frequency low, amplitude high, frequency high}: 0..1 and Hz, neutral 0/160/0/320
typedef struct { float amp_low, freq_low, amp_high, freq_high; } HidVibrationValue;
// type 1 linear resonant actuator; position 1 left, 2 right
typedef struct { uint32_t type, position; } HidVibrationDeviceInfo;
_Static_assert(sizeof(HidNpadState) == 40 && sizeof(HidMouseState) == 40 && sizeof(HidKeyboardState) == 48,
               "hid state ABI");
_Static_assert(sizeof(HidControllerSupportResult) == 8 && sizeof(HidSixAxisState) == 0x60, "hid result ABI");
_Static_assert(sizeof(HidVibrationValue) == 16 && sizeof(HidVibrationDeviceInfo) == 8, "vibration ABI");

#define PAD_ID 0u   // NpadId No1, the docked console's one controller
enum { STYLE_FULL_KEY = 1, STYLE_HANDHELD = 2, ATTR_CONNECTED = 1 };   // NpadStyleTag bits, Npad attribute bit 0
enum { STYLE_INDEX_FULL_KEY = 3, VIBRATION_SIDES = 2 };

// One log line the first time this site runs, from any thread.
#define ONCE(...)                                                                       \
    do {                                                                                \
        static int seen_;                                                               \
        if (!__atomic_exchange_n(&seen_, 1, __ATOMIC_RELAXED)) rt_log(__VA_ARGS__);     \
    } while (0)

static PlatMutex lock = PLAT_MUTEX_INIT;
static uint32_t styles = STYLE_FULL_KEY;                                        // lock: SetSupportedNpadStyleSet
static int enabled, allowed = 1, mouse_ready, keyboard_ready;                   // lock
static int64_t sample;                                                          // lock: the last sampling number given
static const HidVibrationValue vib_neutral = {0, 160, 0, 320};
static HidVibrationValue vib_last[VIBRATION_SIDES] = {{0, 160, 0, 320}, {0, 160, 0, 320}};   // lock
static int vib_allowed = 1;                                                     // lock: PermitVibration
static int six_started, six_live;                                               // lock: Start..Stop; the last read was live
static int64_t motion_offset;                                                   // lock: live sampling numbers = producer's + this

// Hand out n sampling numbers; returns the newest.
static int64_t take_samples(int64_t n) {
    plat_mutex_lock(&lock);
    sample += n;
    int64_t s = sample;
    plat_mutex_unlock(&lock);
    return s;
}

// The 32-bit value a pointer or reference argument names (a controller id, a device handle).
static uint32_t ref32(Cpu *c, int i, const char *fn) {
    uint32_t v = 0;
    if (!hos_in(&v, arg_x(c, i), sizeof v))
        rt_guest_stop(c, "nn::hid::%s: argument %d is not readable guest memory", fn, i + 1);
    return v;
}

// Write an out parameter through argument i. The console faults on a pointer it cannot write, and so does the run.
static void give(Cpu *c, int i, const void *src, uint64_t size, const char *fn) {
    if (!hos_out(arg_x(c, i), src, size))
        rt_guest_stop(c, "nn::hid::%s: argument %d is not writable guest memory", fn, i + 1);
}

// The SDK's id check (utility functions): players 1-8, other (0x10) and handheld (0x20); the public functions abort on
// another id.
static uint32_t handle_id(Cpu *c, int i, const char *fn) {
    uint32_t id = ref32(c, i, fn);
    if (id >= 8 && id != 0x10 && id != 0x20) rt_guest_stop(c, "nn::hid::%s: invalid controller id %#x", fn, id);
    return id;
}

static void ov_hid_ok(Cpu *c) { ret_x(c, HOS_R_OK); }   // Result forms; void forms ignore x0
static void ov_hid_true(Cpu *c) { ret_x(c, 1); }
static void ov_hid_false(Cpu *c) { ret_x(c, 0); }

// ---- Npad
// void InitializeNpad()
static void ov_initialize_npad(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    enabled = 1;
    plat_mutex_unlock(&lock);
    rt_log("hid: controllers initialized (one connected full-key controller, nothing pressed)");
}

// void SetSupportedNpadIdType(const NpadIdType *ids, size_t count)
static void ov_supported_ids(Cpu *c) {
    uint64_t ids = arg_x(c, 0), n = arg_x(c, 1);
    uint32_t list[10] = {0};
    if (n > 10 || (!ids && n))
        rt_guest_stop(c, "nn::hid::SetSupportedNpadIdType: invalid id list (%llu ids)", (unsigned long long)n);
    if (n && !hos_in(list, ids, 4 * n))
        rt_guest_stop(c, "nn::hid::SetSupportedNpadIdType: the id list is not readable guest memory");
    int ok = 0;
    for (uint64_t i = 0; i < n; i++) ok |= list[i] == PAD_ID;
    plat_mutex_lock(&lock);
    allowed = ok;
    plat_mutex_unlock(&lock);
}

// void SetSupportedNpadStyleSet(NpadStyleSet styles)
static void ov_supported_styles(Cpu *c) {
    plat_mutex_lock(&lock);
    styles = (uint32_t)arg_x(c, 0);
    plat_mutex_unlock(&lock);
}

// NpadStyleSet GetNpadStyleSet(const NpadIdType &id)
static void ov_style_set(Cpu *c) {
    uint32_t id = ref32(c, 0, "GetNpadStyleSet");
    plat_mutex_lock(&lock);
    uint32_t v = enabled && allowed && id == PAD_ID ? styles & STYLE_FULL_KEY : 0;
    plat_mutex_unlock(&lock);
    ret_x(c, v);
}

// int GetNpadStates(<State> *out, int count, const NpadIdType &id): the newest state only. want: the style of the form,
// 0 for the Joy-Con forms; only the full-key form of player 1 is a controller here.
static void npad_states(Cpu *c, uint32_t want) {
    int max = (int)arg_x(c, 1);
    uint32_t id = ref32(c, 2, "GetNpadStates");
    if (max <= 0) {
        ret_x(c, 0);
        return;
    }
    plat_mutex_lock(&lock);
    HidNpadState s = {.sampling = ++sample};
    int live = want == STYLE_FULL_KEY && enabled && allowed && id == PAD_ID && (styles & want);
    if (live) {
        HidHostInput in;
        hid_host_input_read(&in);
        hid_replay_filter(&in);   // the input file and the recorder, when asked for (oot2/hid_replay.h); else nothing
        s.buttons = in.buttons;
        s.lx = in.lx, s.ly = in.ly, s.rx = in.rx, s.ry = in.ry;
        s.attributes = ATTR_CONNECTED;
    }
    plat_mutex_unlock(&lock);
    ONCE("hid: GetNpadStates first asked: style %u, id %#x, count %d", want, id, max);
    give(c, 0, &s, sizeof s, "GetNpadStates");
    ret_x(c, 1);
}
static void ov_states_full_key(Cpu *c) { npad_states(c, STYLE_FULL_KEY); }
static void ov_states_handheld(Cpu *c) { npad_states(c, STYLE_HANDHELD); }
static void ov_states_joycon(Cpu *c) { npad_states(c, 0); }

// void SetNpadJoyHoldType(NpadJoyHoldType type)
static void ov_hold_type(Cpu *c) {
    int32_t t = (int32_t)arg_x(c, 0);
    if (t < 0 || t > 1) rt_guest_stop(c, "nn::hid::SetNpadJoyHoldType: invalid hold type %d", t);
}

// void DisconnectNpad(const NpadIdType &id): only the controller in use
static void ov_disconnect(Cpu *c) {
    uint32_t id = ref32(c, 0, "DisconnectNpad");
    char at[96];
    rt_log("hid: the game disconnected controller id %#x (ours is %#x)%s from %s", id, PAD_ID,
           id == PAD_ID ? ": input is off" : "", hos_caller(c, at, sizeof at));
    if (id != PAD_ID) return;
    plat_mutex_lock(&lock);
    allowed = 0;
    plat_mutex_unlock(&lock);
}

// Result ShowControllerSupport(ControllerSupportResultInfo *out, const ControllerSupportArg &arg): the one controller
// is already connected, so the applet returns at once with one player on it.
static void ov_controller_support(Cpu *c) {
    HidControllerSupportResult r = {.players = 1, .selected = PAD_ID};
    if (arg_x(c, 0)) give(c, 0, &r, sizeof r, "ShowControllerSupport");
    ret_x(c, HOS_R_OK);
}

// Result SetNpadJoyAssignmentModeDual(const NpadIdType &), SetNpadJoyAssignmentModeSingle(const NpadIdType &,
// NpadJoyDeviceType), MergeSingleJoyAsDualJoy(const NpadIdType &, const NpadIdType &): the one pad is always a full
// controller, so the assignment mode changes nothing
static void ov_joy_assignment(Cpu *c) { ret_x(c, HOS_R_OK); }

// ---- mouse, keyboard and touch screen: nothing connected
static void ov_initialize_mouse(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    mouse_ready = 1;
    plat_mutex_unlock(&lock);
}

static void ov_initialize_keyboard(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    keyboard_ready = 1;
    plat_mutex_unlock(&lock);
}

// int GetMouseStates(MouseState *out, int count): a mouse that is not connected
static void ov_mouse_states(Cpu *c) {
    int max = (int)arg_x(c, 1);
    if (max <= 0) {
        ret_x(c, 0);
        return;
    }
    plat_mutex_lock(&lock);
    int ready = mouse_ready;
    HidMouseState s = {.sampling = ++sample};
    plat_mutex_unlock(&lock);
    if (!ready) rt_guest_stop(c, "nn::hid::GetMouseStates before InitializeMouse");
    give(c, 0, &s, sizeof s, "GetMouseStates");
    ret_x(c, 1);
}

// void GetKeyboardState(KeyboardState *out): a keyboard with no key down
static void ov_keyboard_state(Cpu *c) {
    plat_mutex_lock(&lock);
    int ready = keyboard_ready;
    HidKeyboardState s = {.sampling = ++sample};
    plat_mutex_unlock(&lock);
    if (!ready) rt_guest_stop(c, "nn::hid::GetKeyboardState before InitializeKeyboard");
    give(c, 0, &s, sizeof s, "GetKeyboardState");
}

// void InitializeTouchScreen(): there is no touch screen on the host
static void ov_touch_init(Cpu *c) {
    (void)c;
    ONCE("hid: the game initialized the touch screen (none on this host)");
}

// void GetTouchScreenState<1>(TouchScreenState<1> *out): sampling number, then a count of 0 touches
static void ov_touch_state1(Cpu *c) {
    uint8_t out[0x38] = {0};   // int64 sampling number, int32 count, int32 reserved, one 0x28-byte TouchState
    int64_t n = take_samples(1);
    memcpy(out, &n, sizeof n);
    give(c, 0, out, sizeof out, "GetTouchScreenState");
}

// ---- vibration: the controller's two devices exist; a value is remembered and nothing plays
// int GetVibrationDeviceHandles(VibrationDeviceHandle *out, int count, const NpadIdType &id, NpadStyleSet style): the
// SDK's table, the first style bit in its order.
static void ov_vibration_handles(Cpu *c) {
    uint64_t out = arg_x(c, 0);
    int max = (int)arg_x(c, 1);
    uint32_t style = (uint32_t)arg_x(c, 3), id = handle_id(c, 2, "GetVibrationDeviceHandles");
    if (!out || max <= 0 || __builtin_popcount(style) != 1)
        rt_guest_stop(c, "nn::hid::GetVibrationDeviceHandles: invalid arguments (count %d, styles %#x)", max, style);
    static const struct { uint8_t bit, n; uint32_t base; } t[] = {
        {0, 2, 0x3}, {1, 2, 0x4}, {2, 2, 0x5}, {3, 1, 0x6}, {4, 1, 0x10007}, {5, 1, 0x8}, {10, 1, 0xd}, {8, 2, 0x4},
        {30, 2, 0x21}, {29, 2, 0x20}};
    uint32_t h[2] = {0, 0};
    int n = 0;
    for (size_t i = 0; i < sizeof t / sizeof *t && !n; i++) {
        if (!((style >> t[i].bit) & 1)) continue;
        n = t[i].n < max ? t[i].n : max;
        for (int k = 0; k < n; k++) h[k] = (t[i].base | (id << 8)) + ((uint32_t)k << 16);
    }
    if (n) give(c, 0, h, 4u * (uint32_t)n, "GetVibrationDeviceHandles");
    ONCE("hid: the game asked for its vibration devices (first handle %#x)", n ? h[0] : 0u);
    ret_x(c, (uint64_t)n);
}

// The vibration device (0 left, 1 right) a handle names on the connected controller, or -1 (logged once).
static int side_of(uint32_t v) {
    uint32_t dev = (v >> 16) & 0xff;
    if ((v & 0xff) == STYLE_INDEX_FULL_KEY && ((v >> 8) & 0xff) == PAD_ID && dev < 2) return (int)dev;
    ONCE("hid: vibration call for handle %#x, not a device of the connected controller: ignored", v);
    return -1;
}

static int vib_handle(Cpu *c, int i, const char *fn, uint32_t *raw) {
    uint32_t v = ref32(c, i, fn);
    if (raw) *raw = v;
    return side_of(v);
}

static void play(int side, const HidVibrationValue *v) {
    if (side < 0) return;
    HidVibrationValue val = *v;
    if (!isfinite(val.amp_low) || !isfinite(val.amp_high) || !isfinite(val.freq_low) || !isfinite(val.freq_high))
        val = vib_neutral;
    ONCE("hid: first vibration value from the game: %s %.2f@%.0f Hz %.2f@%.0f Hz", side ? "right" : "left",
         (double)val.amp_low, (double)val.freq_low, (double)val.amp_high, (double)val.freq_high);
    if (val.amp_low > 0.01f || val.amp_high > 0.01f)
        ONCE("hid: first non-zero vibration from the game: %s %.2f@%.0f Hz %.2f@%.0f Hz", side ? "right" : "left",
             (double)val.amp_low, (double)val.freq_low, (double)val.amp_high, (double)val.freq_high);
    plat_mutex_lock(&lock);
    vib_last[side] = val;
    HidRumble r = hid_rumble_mix(vib_last[0].amp_low, vib_last[0].amp_high,
                                 vib_last[1].amp_low, vib_last[1].amp_high);
    plat_mutex_unlock(&lock);
    hid_rumble_publish(&r);
}

// Result detail::InitializeVibrationDevice(const VibrationDeviceHandle &), and the public form
static void ov_vib_init(Cpu *c) {
    uint32_t h;
    if (vib_handle(c, 0, "InitializeVibrationDevice", &h) >= 0)
        ONCE("hid: the game initialized its vibration devices (handle %#x)", h);
    ret_x(c, HOS_R_OK);
}

// Result detail::SendVibrationValue(const VibrationDeviceHandle &, const VibrationValue &), and the public form
static void ov_vib_send(Cpu *c) {
    int side = vib_handle(c, 0, "SendVibrationValue", NULL);
    HidVibrationValue v;
    if (!hos_in(&v, arg_x(c, 1), sizeof v))
        rt_guest_stop(c, "nn::hid::SendVibrationValue: the value is not readable guest memory");
    play(side, &v);
    ret_x(c, HOS_R_OK);
}

// Result detail::SendVibrationValues(const VibrationDeviceHandle *, const VibrationValue *, int count)
static void ov_vib_send_many(Cpu *c) {
    uint64_t h = arg_x(c, 0), v = arg_x(c, 1);
    int n = (int)arg_x(c, 2);
    if (n < 0 || (n && (!h || !v)))
        rt_guest_stop(c, "nn::hid::detail::SendVibrationValues: invalid batch (%d values)", n);
    for (int i = 0; i < n; i++) {
        uint32_t handle;
        HidVibrationValue val;
        if (!hos_in(&handle, h + 4u * (uint64_t)i, sizeof handle) ||
            !hos_in(&val, v + sizeof val * (uint64_t)i, sizeof val))
            rt_guest_stop(c, "nn::hid::detail::SendVibrationValues: value %d is not readable guest memory", i);
        play(side_of(handle), &val);
    }
    ret_x(c, HOS_R_OK);
}

// Result detail::GetActualVibrationValue(VibrationValue *out, const VibrationDeviceHandle &): the last value sent
static void ov_vib_actual(Cpu *c) {
    int side = vib_handle(c, 1, "GetActualVibrationValue", NULL);
    HidVibrationValue v = vib_neutral;
    plat_mutex_lock(&lock);
    if (side >= 0) v = vib_last[side];
    plat_mutex_unlock(&lock);
    give(c, 0, &v, sizeof v, "GetActualVibrationValue");
    ret_x(c, HOS_R_OK);
}

// void GetActualVibrationValue(VibrationValue *out, const VibrationDeviceHandle &): neutral, nothing is playing
static void ov_actual_vibration_public(Cpu *c) {
    give(c, 0, &vib_neutral, sizeof vib_neutral, "GetActualVibrationValue");
}

// Result detail::GetVibrationDeviceInfo(VibrationDeviceInfo *out, const VibrationDeviceHandle &), and the public form
static void ov_vib_info(Cpu *c) {
    int side = vib_handle(c, 1, "GetVibrationDeviceInfo", NULL);
    HidVibrationDeviceInfo info = {0, 0};
    if (side >= 0) info = (HidVibrationDeviceInfo){1, (uint32_t)side + 1};
    give(c, 0, &info, sizeof info, "GetVibrationDeviceInfo");
    ret_x(c, HOS_R_OK);
}

// Result detail::IsVibrationDeviceMounted(bool *out, const VibrationDeviceHandle &)
static void ov_vib_mounted(Cpu *c) {
    uint8_t mounted = vib_handle(c, 1, "IsVibrationDeviceMounted", NULL) >= 0;
    give(c, 0, &mounted, sizeof mounted, "IsVibrationDeviceMounted");
    ret_x(c, HOS_R_OK);
}

// Result detail::IsVibrationPermitted(bool *out)
static void ov_vib_permitted(Cpu *c) {
    plat_mutex_lock(&lock);
    uint8_t on = (uint8_t)vib_allowed;
    plat_mutex_unlock(&lock);
    give(c, 0, &on, sizeof on, "IsVibrationPermitted");
    ret_x(c, HOS_R_OK);
}

// bool IsVibrationPermitted()
static void ov_vib_permitted_public(Cpu *c) {
    plat_mutex_lock(&lock);
    int on = vib_allowed;
    plat_mutex_unlock(&lock);
    ret_x(c, (uint64_t)on);
}

// Result detail::PermitVibration(bool)
static void ov_vib_permit(Cpu *c) {
    int on = (arg_x(c, 0) & 0xff) != 0;
    plat_mutex_lock(&lock);
    vib_allowed = on;
    plat_mutex_unlock(&lock);
    ret_x(c, HOS_R_OK);
}

// ---- six-axis sensor: the host pad's motion, or at rest
// int GetSixAxisSensorHandles(SixAxisSensorHandle *out, int count, const NpadIdType &id, NpadStyleSet style): the SDK's
// order (one sensor per style, two for the Joy-Con pair).
static void ov_six_axis_handles(Cpu *c) {
    uint64_t out = arg_x(c, 0);
    int max = (int)arg_x(c, 1);
    uint32_t style = (uint32_t)arg_x(c, 3), id = handle_id(c, 2, "GetSixAxisSensorHandles"), base = 0, n = 1;
    if (!out || max <= 0) rt_guest_stop(c, "nn::hid::GetSixAxisSensorHandles: invalid arguments (count %d)", max);
    if (style & 1) base = 0x20003;
    else if (style & 2) base = 0x20004;
    else if (style & 4) { base = 0x5; n = 2; }
    else if (style & 8) base = 0x6;
    else if (style & 0x10) base = 0x10007;
    else if (style & 0x20) base = 0x20003;
    else if (style & 0xe80) n = 0;
    else if (style & 0x100) base = 0x20004;
    else if ((style & 0x60000040) == 0x40) base = 0x20003;
    else n = 0;
    if (n > (uint32_t)max) n = (uint32_t)max;
    uint32_t h[2] = {0, 0};
    for (uint32_t k = 0; k < n; k++) h[k] = (base | (id << 8)) + (k << 16);
    if (n) give(c, 0, h, 4u * n, "GetSixAxisSensorHandles");
    ONCE("hid: the game asked for its six-axis sensor (first handle %#x)", n ? h[0] : 0u);
    ret_x(c, n);
}

// Write max states newest first. While the game has started the sensor and the host pad has a gyroscope and an
// accelerometer (oot2/hid_motion.h), the newest states are the pad's; a read that has fewer samples than states asked
// for is padded with the oldest sample's pose and no motion (zero angular velocity and zero delta), so a state the pad
// did not send adds no rotation. Otherwise every state is at rest, not marked connected: a controller lying still. The
// game reads every state it asks for and ignores the count, so all max are filled. Sampling numbers only ever grow: a
// live read takes the producer's numbers (stable while the samples are the same ones) shifted above every number given
// before, and a read at rest numbers its states from the last number given.
static void six_fill(Cpu *c, uint64_t out, int max, const char *fn) {
    static const HidSixAxisState rest = {.acc = {0, 0, -1}, .dir = {1, 0, 0, 0, 1, 0, 0, 0, 1}};
    HidMotionState live[HID_MOTION_RING];
    int want = max < HID_MOTION_RING ? max : HID_MOTION_RING, n = 0;
    int64_t newest;
    plat_mutex_lock(&lock);
    if (six_started && !hid_replay_replaying()) n = hid_motion_read(live, want, plat_now_ns());
    if (n > 0) {
        if (!six_live) {
            motion_offset = sample + max - live[0].sampling + 1;
            six_live = 1;
        }
        newest = live[0].sampling + motion_offset;
        if (newest > sample) sample = newest;
    } else {
        six_live = 0;
        newest = sample += max;
    }
    plat_mutex_unlock(&lock);
    for (int i = 0; i < max; i++) {
        HidSixAxisState s = rest;
        if (n > 0) {
            const HidMotionState *m = &live[i < n ? i : n - 1];
            s.delta_ns = i < n ? m->delta_ns : 0;
            memcpy(s.acc, m->acc, sizeof s.acc);
            if (i < n) memcpy(s.gyro, m->gyro, sizeof s.gyro);
            memcpy(s.angle, m->angle, sizeof s.angle);
            memcpy(s.dir, m->dir, sizeof s.dir);
            s.attributes = ATTR_CONNECTED;
        }
        s.sampling = newest - i;
        if (!hos_out(out + (uint64_t)i * sizeof s, &s, sizeof s))
            rt_guest_stop(c, "nn::hid::%s: state %d is not writable guest memory", fn, i);
    }
    if (n > 0) ONCE("hid: six-axis read: live motion from the host pad (%d of %d states)", n, max);
    else ONCE("hid: six-axis read: at rest, no host motion (%u states)", (uint32_t)max);
}

// Result StartSixAxisSensor(const SixAxisSensorHandle &), also detail::: the host pad's sensors are turned on, and the
// angle and direction start over.
static void ov_six_start(Cpu *c) {
    uint32_t h = ref32(c, 0, "StartSixAxisSensor");
    ONCE("hid: the game started the six-axis sensor (handle %#x)", h);
    plat_mutex_lock(&lock);
    six_started = 1;
    plat_mutex_unlock(&lock);
    hid_motion_restart();
    hid_motion_set_wanted(1);
    ret_x(c, HOS_R_OK);
}

// Result StopSixAxisSensor(const SixAxisSensorHandle &), also detail::
static void ov_six_stop(Cpu *c) {
    (void)ref32(c, 0, "StopSixAxisSensor");
    plat_mutex_lock(&lock);
    six_started = 0;
    plat_mutex_unlock(&lock);
    hid_motion_set_wanted(0);
    ret_x(c, HOS_R_OK);
}

// int GetSixAxisSensorStates(SixAxisSensorState *out, int count, const SixAxisSensorHandle &): the count is returned
static void ov_six_states_public(Cpu *c) {
    int max = (int)arg_x(c, 1);
    (void)ref32(c, 2, "GetSixAxisSensorStates");
    if (max <= 0) {
        ret_x(c, 0);
        return;
    }
    six_fill(c, arg_x(c, 0), max, "GetSixAxisSensorStates");
    ret_x(c, (uint64_t)(int64_t)max);
}

// Result detail::GetSixAxisSensorStates(int *count, SixAxisSensorState *out, int max, const SixAxisSensorHandle &)
static void ov_six_states(Cpu *c) {
    int max = (int)arg_x(c, 2);
    (void)ref32(c, 3, "detail::GetSixAxisSensorStates");
    int32_t n = max > 0 ? max : 0;
    if (n) six_fill(c, arg_x(c, 1), n, "detail::GetSixAxisSensorStates");
    give(c, 0, &n, sizeof n, "detail::GetSixAxisSensorStates");
    ret_x(c, HOS_R_OK);
}

// [Result detail::]GetSixAxisSensorState(SixAxisSensorState *out, const SixAxisSensorHandle &)
static void ov_six_state(Cpu *c) {
    (void)ref32(c, 1, "GetSixAxisSensorState");
    six_fill(c, arg_x(c, 0), 1, "GetSixAxisSensorState");
    ret_x(c, HOS_R_OK);
}

// GyroscopeZeroDriftMode GetGyroscopeZeroDriftMode(const SixAxisSensorHandle &): 1 = Standard (Loose 0, Tight 2)
static void ov_gyro_mode(Cpu *c) { ret_x(c, 1); }
// Result detail::GetGyroscopeZeroDriftMode(GyroscopeZeroDriftMode *out, SixAxisSensorHandle)
static void ov_gyro_mode_detail(Cpu *c) {
    uint32_t standard = 1;
    give(c, 0, &standard, sizeof standard, "detail::GetGyroscopeZeroDriftMode");
    ret_x(c, HOS_R_OK);
}
// Result detail::IsSixAxisSensorAtRest(bool *out, const SixAxisSensorHandle &)
static void ov_out_true(Cpu *c) {
    static const uint8_t yes = 1;
    give(c, 0, &yes, sizeof yes, "detail::IsSixAxisSensorAtRest");
    ret_x(c, HOS_R_OK);
}
// Result detail::IsSixAxisSensorFusionEnabled(bool *out, const SixAxisSensorHandle &)
static void ov_out_false(Cpu *c) {
    static const uint8_t no = 0;
    give(c, 0, &no, sizeof no, "detail::IsSixAxisSensorFusionEnabled");
    ret_x(c, HOS_R_OK);
}
// [Result detail::]GetSixAxisSensorFusionParameters(float *revise_power, float *revise_range,
// const SixAxisSensorHandle &)
static void ov_fusion_params(Cpu *c) {
    static const float power = 0.03f, range = 0.4f;
    give(c, 0, &power, sizeof power, "GetSixAxisSensorFusionParameters");
    give(c, 1, &range, sizeof range, "GetSixAxisSensorFusionParameters");
    ret_x(c, HOS_R_OK);
}

HOS_OVERRIDE("_ZN2nn3hid14InitializeNpadEv", ov_initialize_npad, HOS_IMPLEMENTED,
             "controllers are initialised: one connected full-key controller; donor ov_input.c:44-51");
HOS_OVERRIDE("_ZN2nn3hid22SetSupportedNpadIdTypeEPKjm", ov_supported_ids, HOS_IMPLEMENTED,
             "the controller is usable when player 1 is in the list; donor ov_input.c:54-63");
HOS_OVERRIDE("_ZN2nn3hid24SetSupportedNpadStyleSetENS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE",
             ov_supported_styles, HOS_IMPLEMENTED, "records the supported style set; donor ov_input.c:66-70");
HOS_OVERRIDE("_ZN2nn3hid15GetNpadStyleSetERKj", ov_style_set, HOS_IMPLEMENTED,
             "the full-key style for player 1 once initialised and supported, else none; donor ov_input.c:73-79");
HOS_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadFullKeyStateEiRKj", ov_states_full_key, HOS_STUB,
             "one connected full-key state, nothing pressed, growing sampling number: no host input layer yet; donor "
             "ov_input.c:114-137");
HOS_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyDualStateEiRKj", ov_states_joycon, HOS_IMPLEMENTED,
             "a docked console has no Joy-Con pair: zeroed, not connected, with a sampling number; donor "
             "ov_input.c:114-139");
HOS_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_16NpadJoyLeftStateEiRKj", ov_states_joycon, HOS_IMPLEMENTED,
             "a docked console has no single Joy-Con: zeroed, not connected, with a sampling number; donor "
             "ov_input.c:114-139");
HOS_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadJoyRightStateEiRKj", ov_states_joycon, HOS_IMPLEMENTED,
             "a docked console has no single Joy-Con: zeroed, not connected, with a sampling number; donor "
             "ov_input.c:114-139");
HOS_OVERRIDE("_ZN2nn3hid13GetNpadStatesEPNS0_17NpadHandheldStateEiRKj", ov_states_handheld, HOS_IMPLEMENTED,
             "a docked console has no handheld controller: zeroed, not connected, with a sampling number; donor "
             "ov_input.c:114-138");
HOS_OVERRIDE("_ZN2nn3hid18SetNpadJoyHoldTypeENS0_15NpadJoyHoldTypeE", ov_hold_type, HOS_STUB,
             "hold type checked and dropped: the one pad is a full controller; donor ov_input.c:142-145");
HOS_OVERRIDE("_ZN2nn3hid14DisconnectNpadERKj", ov_disconnect, HOS_IMPLEMENTED,
             "disconnecting player 1 turns its input off, any other id is ignored; donor ov_input.c:148-157");
HOS_OVERRIDE("_ZN2nn3hid21ShowControllerSupportEPNS0_27ControllerSupportResultInfoERKNS0_20ControllerSupportArgE",
             ov_controller_support, HOS_STUB,
             "the applet is not shown: the one controller is connected, so it returns at once with one player; donor "
             "ov_input.c:161-165");
HOS_OVERRIDE("_ZN2nn3hid28SetNpadJoyAssignmentModeDualERKj", ov_joy_assignment, HOS_STUB,
             "accepted, the one pad is always a full controller; donor ov_input.c:584");
HOS_OVERRIDE("_ZN2nn3hid30SetNpadJoyAssignmentModeSingleERKjNS0_17NpadJoyDeviceTypeE", ov_joy_assignment, HOS_STUB,
             "accepted, the one pad is always a full controller; donor ov_input.c:584");
HOS_OVERRIDE("_ZN2nn3hid23MergeSingleJoyAsDualJoyERKjS2_", ov_joy_assignment, HOS_STUB,
             "accepted, the one pad is always a full controller; donor ov_input.c:584");
HOS_OVERRIDE("_ZN2nn3hid15InitializeMouseEv", ov_initialize_mouse, HOS_IMPLEMENTED,
             "marks the mouse initialised; donor ov_input.c:167-172");
HOS_OVERRIDE("_ZN2nn3hid18InitializeKeyboardEv", ov_initialize_keyboard, HOS_IMPLEMENTED,
             "marks the keyboard initialised; donor ov_input.c:174-179");
HOS_OVERRIDE("_ZN2nn3hid14GetMouseStatesEPNS0_10MouseStateEi", ov_mouse_states, HOS_STUB,
             "one state of a mouse with no input and a growing sampling number: no host input layer yet; donor "
             "ov_input.c:182-192");
HOS_OVERRIDE("_ZN2nn3hid16GetKeyboardStateEPNS0_13KeyboardStateE", ov_keyboard_state, HOS_STUB,
             "a keyboard with no key down and a growing sampling number: no host input layer yet; donor "
             "ov_input.c:195-202");
HOS_OVERRIDE("_ZN2nn3hid21InitializeTouchScreenEv", ov_touch_init, HOS_STUB,
             "accepted, there is no touch screen on the host; donor ov_input.c:565-569");
HOS_OVERRIDE("_ZN2nn3hid19GetTouchScreenStateILm1EEEvPNS0_16TouchScreenStateIXT_EEE", ov_touch_state1, HOS_STUB,
             "a sampling number and no touches: no host input layer yet; donor ov_input.c:572-579");
HOS_OVERRIDE("_ZN2nn3hid25GetVibrationDeviceHandlesEPNS0_21VibrationDeviceHandleEiRKjNS_4util10BitFlagSetILi32ENS0_"
             "12NpadStyleTagEEE",
             ov_vibration_handles, HOS_IMPLEMENTED,
             "the SDK's handle table, the first style bit in its order: left and right for a full-key pad; donor "
             "ov_input.c:264-285");
HOS_OVERRIDE("_ZN2nn3hid23GetSixAxisSensorHandlesEPNS0_19SixAxisSensorHandleEiRKjNS_4util10BitFlagSetILi32ENS0_"
             "12NpadStyleTagEEE",
             ov_six_axis_handles, HOS_IMPLEMENTED,
             "the SDK's handle order, one sensor per style and two for the Joy-Con pair; donor ov_input.c:290-311");
HOS_OVERRIDE("_ZN2nn3hid25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE", ov_vib_init, HOS_IMPLEMENTED,
             "accepts a device of the connected controller; donor ov_input.c:382-388");
HOS_OVERRIDE("_ZN2nn3hid6detail25InitializeVibrationDeviceERKNS0_21VibrationDeviceHandleE", ov_vib_init,
             HOS_IMPLEMENTED, "accepts a device of the connected controller; donor ov_input.c:382-388");
HOS_OVERRIDE("_ZN2nn3hid18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE", ov_vib_send,
             HOS_STUB, "remembers the value, nothing plays: no host rumble layer yet; donor ov_input.c:391-394");
HOS_OVERRIDE("_ZN2nn3hid6detail18SendVibrationValueERKNS0_21VibrationDeviceHandleERKNS0_14VibrationValueE", ov_vib_send,
             HOS_STUB, "remembers the value, nothing plays: no host rumble layer yet; donor ov_input.c:391-394");
HOS_OVERRIDE("_ZN2nn3hid6detail19SendVibrationValuesEPKNS0_21VibrationDeviceHandleEPKNS0_14VibrationValueEi",
             ov_vib_send_many, HOS_STUB,
             "remembers each value, nothing plays: no host rumble layer yet; donor ov_input.c:397-404");
HOS_OVERRIDE("_ZN2nn3hid6detail23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE",
             ov_vib_actual, HOS_IMPLEMENTED, "the last value sent to that device; donor ov_input.c:407-417");
HOS_OVERRIDE("_ZN2nn3hid23GetActualVibrationValueEPNS0_14VibrationValueERKNS0_21VibrationDeviceHandleE",
             ov_actual_vibration_public, HOS_IMPLEMENTED, "neutral, nothing is playing; donor ov_input.c:559-562");
HOS_OVERRIDE("_ZN2nn3hid22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE",
             ov_vib_info, HOS_IMPLEMENTED,
             "a linear resonant actuator, left or right, for a device of the controller; donor ov_input.c:420-426");
HOS_OVERRIDE("_ZN2nn3hid6detail22GetVibrationDeviceInfoEPNS0_19VibrationDeviceInfoERKNS0_21VibrationDeviceHandleE",
             ov_vib_info, HOS_IMPLEMENTED,
             "a linear resonant actuator, left or right, for a device of the controller; donor ov_input.c:420-426");
HOS_OVERRIDE("_ZN2nn3hid6detail24IsVibrationDeviceMountedEPbRKNS0_21VibrationDeviceHandleE", ov_vib_mounted,
             HOS_IMPLEMENTED, "mounted for a device of the connected controller, else not; donor ov_input.c:429-434");
HOS_OVERRIDE("_ZN2nn3hid24IsVibrationDeviceMountedERKNS0_21VibrationDeviceHandleE", ov_hid_true, HOS_IMPLEMENTED,
             "mounted; donor ov_input.c:651");
HOS_OVERRIDE("_ZN2nn3hid29IsOunceVibrationDeviceMountedERKNS0_21VibrationDeviceHandleE", ov_hid_true, HOS_IMPLEMENTED,
             "mounted, as for the other vibration devices handed out; donor ov_input.c:652");
HOS_OVERRIDE("_ZN2nn3hid6detail20IsVibrationPermittedEPb", ov_vib_permitted, HOS_IMPLEMENTED,
             "the PermitVibration state, permitted at start; donor ov_input.c:437-444");
HOS_OVERRIDE("_ZN2nn3hid20IsVibrationPermittedEv", ov_vib_permitted_public, HOS_IMPLEMENTED,
             "the PermitVibration state, permitted at start; donor ov_input.c:650 answered always true");
HOS_OVERRIDE("_ZN2nn3hid6detail15PermitVibrationEb", ov_vib_permit, HOS_IMPLEMENTED,
             "records whether vibration is permitted; donor ov_input.c:447-454");
HOS_OVERRIDE("_ZN2nn3hid27BeginForcedVibrationSessionEv", ov_hid_ok, HOS_STUB,
             "accepted, no vibration plays; donor ov_input.c:654");
HOS_OVERRIDE("_ZN2nn3hid25EndForcedVibrationSessionEv", ov_hid_ok, HOS_STUB,
             "accepted, no vibration plays; donor ov_input.c:655");
HOS_OVERRIDE("_ZN2nn3hid31EnableVibrationNxCompatibleModeEv", ov_hid_ok, HOS_STUB,
             "accepted, no vibration plays; donor ov_input.c:656");
HOS_OVERRIDE("_ZN2nn3hid32DisableVibrationNxCompatibleModeEv", ov_hid_ok, HOS_STUB,
             "accepted, no vibration plays; donor ov_input.c:657");
HOS_OVERRIDE("_ZN2nn3hid18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE", ov_six_start, HOS_IMPLEMENTED,
             "accepted: the host pad's sensors are turned on and the angle starts over; donor ov_input.c:467-477");
HOS_OVERRIDE("_ZN2nn3hid6detail18StartSixAxisSensorERKNS0_19SixAxisSensorHandleE", ov_six_start, HOS_IMPLEMENTED,
             "accepted: the host pad's sensors are turned on and the angle starts over; donor ov_input.c:467-477");
HOS_OVERRIDE("_ZN2nn3hid17StopSixAxisSensorERKNS0_19SixAxisSensorHandleE", ov_six_stop, HOS_IMPLEMENTED,
             "accepted: the host pad's sensors are turned off; donor ov_input.c:480-483");
HOS_OVERRIDE("_ZN2nn3hid6detail17StopSixAxisSensorERKNS0_19SixAxisSensorHandleE", ov_six_stop, HOS_IMPLEMENTED,
             "accepted: the host pad's sensors are turned off; donor ov_input.c:480-483");
HOS_OVERRIDE("_ZN2nn3hid22GetSixAxisSensorStatesEPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE",
             ov_six_states_public, HOS_IMPLEMENTED,
             "the host pad's gyroscope and accelerometer while the sensor is started, else all states at rest; newest "
             "first, sampling numbers growing; donor ov_input.c:488-502, 522-524");
HOS_OVERRIDE("_ZN2nn3hid6detail22GetSixAxisSensorStatesEPiPNS0_18SixAxisSensorStateEiRKNS0_19SixAxisSensorHandleE",
             ov_six_states, HOS_IMPLEMENTED,
             "the host pad's gyroscope and accelerometer while the sensor is started, else all states at rest; newest "
             "first, sampling numbers growing; donor ov_input.c:488-510");
HOS_OVERRIDE("_ZN2nn3hid21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE", ov_six_state,
             HOS_IMPLEMENTED, "one state from the host pad's motion while the sensor is started, else at rest; donor "
             "ov_input.c:513-516");
HOS_OVERRIDE("_ZN2nn3hid6detail21GetSixAxisSensorStateEPNS0_18SixAxisSensorStateERKNS0_19SixAxisSensorHandleE",
             ov_six_state, HOS_IMPLEMENTED, "one state from the host pad's motion while the sensor is started, else at "
             "rest; donor ov_input.c:513-516");
HOS_OVERRIDE("_ZN2nn3hid25SetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleERKNS0_22GyroscopeZeroDriftModeE",
             ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529, 631");
HOS_OVERRIDE("_ZN2nn3hid25GetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleE", ov_gyro_mode, HOS_IMPLEMENTED,
             "Standard, the default mode; donor ov_input.c:533");
HOS_OVERRIDE("_ZN2nn3hid6detail25SetGyroscopeZeroDriftModeENS0_19SixAxisSensorHandleENS0_22GyroscopeZeroDriftModeE",
             ov_hid_ok, HOS_STUB, "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid6detail25GetGyroscopeZeroDriftModeEPNS0_22GyroscopeZeroDriftModeENS0_19SixAxisSensorHandleE",
             ov_gyro_mode_detail, HOS_IMPLEMENTED, "Standard, the default mode; donor ov_input.c:535-539");
HOS_OVERRIDE("_ZN2nn3hid6detail27ResetGyroscopeZeroDriftModeENS0_19SixAxisSensorHandleE", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid21IsSixAxisSensorAtRestERKNS0_19SixAxisSensorHandleE", ov_hid_true, HOS_IMPLEMENTED,
             "at rest, as the sensor always reads; donor ov_input.c:530, 633");
HOS_OVERRIDE("_ZN2nn3hid6detail21IsSixAxisSensorAtRestEPbRKNS0_19SixAxisSensorHandleE", ov_out_true, HOS_IMPLEMENTED,
             "at rest, as the sensor always reads; donor ov_input.c:541-545");
HOS_OVERRIDE("_ZN2nn3hid25EnableSixAxisSensorFusionERKNS0_19SixAxisSensorHandleEb", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid6detail25EnableSixAxisSensorFusionERKNS0_19SixAxisSensorHandleEb", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid28IsSixAxisSensorFusionEnabledERKNS0_19SixAxisSensorHandleE", ov_hid_false,
             HOS_IMPLEMENTED, "fusion off, as the donor answers; donor ov_input.c:531, 635");
HOS_OVERRIDE("_ZN2nn3hid6detail28IsSixAxisSensorFusionEnabledEPbRKNS0_19SixAxisSensorHandleE", ov_out_false,
             HOS_IMPLEMENTED, "fusion off, as the donor answers; donor ov_input.c:546-550");
HOS_OVERRIDE("_ZN2nn3hid32SetSixAxisSensorFusionParametersERKNS0_19SixAxisSensorHandleEff", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid6detail32SetSixAxisSensorFusionParametersERKNS0_19SixAxisSensorHandleEff", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid32GetSixAxisSensorFusionParametersEPfS1_RKNS0_19SixAxisSensorHandleE", ov_fusion_params,
             HOS_IMPLEMENTED, "revise power 0.03 and revise range 0.4, the defaults; donor ov_input.c:552-557");
HOS_OVERRIDE("_ZN2nn3hid6detail32GetSixAxisSensorFusionParametersEPfS2_RKNS0_19SixAxisSensorHandleE", ov_fusion_params,
             HOS_IMPLEMENTED, "revise power 0.03 and revise range 0.4, the defaults; donor ov_input.c:552-557");
HOS_OVERRIDE("_ZN2nn3hid34ResetSixAxisSensorFusionParametersERKNS0_19SixAxisSensorHandleE", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
HOS_OVERRIDE("_ZN2nn3hid6detail34ResetSixAxisSensorFusionParametersERKNS0_19SixAxisSensorHandleE", ov_hid_ok, HOS_STUB,
             "accepted without effect: the sensor has one fixed behaviour; donor ov_input.c:529");
