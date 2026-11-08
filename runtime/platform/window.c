// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The window (oot2/window.h): SDL3, one thread that makes the window and pumps its events, the Vulkan surface calls, the
// menu's keys and pad buttons, the pointer, and the host input layer's producer: it reads the gamepad (or, with none, the
// keyboard), applies the mapping and dead zone, publishes the snapshot (oot2/hid_input.h), and plays the vibration
// requests it is given. Compiled without SDL3 (OOT2_WITH_SDL3 unset: no kit, tools/fetch_sdl3.py) to refusals, so a
// program built that way starts and presents to the offscreen capture.
//
// Why a thread of our own: SDL wants every video call on the thread that initialised it, the game's main thread waits
// for the guest in rt_kernel_run, and no guest thread may touch a window. The thread is that SDL "main thread".
// Everything another thread asks goes through win_request (an atomic mailbox and a wake-up event) or runs on the thread
// with the caller waiting (the surface calls), never the other way round: the settings store's hooks call win_request
// with its lock held, and the thread's resize callback takes that lock.
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include "oot2/window.h"
#include "oot2/hid_input.h"
#include "oot2/settings.h"

#ifndef OOT2_WITH_SDL3

static const char *const none[1];
static const char *why_not(void) { return "this build has no SDL3 (tools/fetch_sdl3.py, SDL3_DIR)"; }
int win_open(const WinConfig *cfg) { (void)cfg; return 0; }
const char *win_why(void) { return why_not(); }
void win_close(void) {}
int win_is_open(void) { return 0; }
void win_state(WinState *out) { memset(out, 0, sizeof *out); }
void win_set_input(const WinInput *in) { (void)in; }
void win_request(int mode, int width, int height) { (void)mode; (void)width; (void)height; }
void win_post_button(int button, int down) { (void)button; (void)down; }
void win_post_pointer(float x, float y, int down) { (void)x; (void)y; (void)down; }
void win_post_minimise(int on) { (void)on; }
void win_post_close(void) {}
void win_post_topmost(int on) { (void)on; }
int win_client_origin(int *x, int *y) { *x = *y = 0; return 0; }
static const char *const *no_exts(uint32_t *n) { *n = 0; return none; }
static int no_present(void *i, void *p, uint32_t f) { (void)i; (void)p; (void)f; return 0; }
static int no_surface(void *i, uint64_t *s) { (void)i; *s = 0; return 0; }
static void no_destroy(void *i, uint64_t s) { (void)i; (void)s; }
static int no_display(char *out, size_t size) {
    snprintf(out, size, "no window (%s)", why_not());
    return 0;
}
const WinVulkan win_vulkan = {no_exts, no_present, no_surface, no_destroy, win_state, no_display};

#else

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>

#include "plat.h"

#define SETTLE_NS 400000000ull   // a resize is settled when no size event came for this long
#define WAIT_MS 50
#define MAX_PADS 8
// oot2/ui.h's UiButton, by value: F1, Escape, Guide, Minus, Plus.
enum { BTN_F1, BTN_ESCAPE, BTN_GUIDE, BTN_MINUS, BTN_PLUS };

static SDL_Window *g_win;
static PlatThread *g_thread;
static SDL_ThreadID g_thread_id;
static PlatMutex g_mu = PLAT_MUTEX_INIT;
static PlatCond g_cv = PLAT_COND_INIT;
static int g_ready, g_stop, g_ended, g_open;
static char g_why[256] = "no window was opened";
static WinConfig g_cfg;
static char g_title[128];
static WinInput g_input;
static Uint32 g_ev_wake, g_ev_job;

// the facts of win_state, written by the window thread, read by any
static _Atomic int s_w, s_h, s_min, s_focus, s_mode, s_close;
static _Atomic uint64_t s_gen;
// win_request's mailbox
static _Atomic int r_mode = -1, r_w, r_h;

typedef struct Job { void (*fn)(void *); void *arg; int done; } Job;

static void fail(const char *what) {
    snprintf(g_why, sizeof g_why, "%s: %s", what, SDL_GetError());
}

const char *win_why(void) { return g_why; }
int win_is_open(void) { return __atomic_load_n(&g_open, __ATOMIC_ACQUIRE); }

void win_state(WinState *out) {
    out->width = atomic_load(&s_w);
    out->height = atomic_load(&s_h);
    out->minimised = atomic_load(&s_min);
    out->focused = atomic_load(&s_focus);
    out->mode = atomic_load(&s_mode);
    out->close_requested = atomic_load(&s_close);
    out->generation = atomic_load(&s_gen);
}

void win_set_input(const WinInput *in) {
    plat_mutex_lock(&g_mu);
    g_input = in ? *in : (WinInput){0};
    plat_mutex_unlock(&g_mu);
}

static WinInput input_copy(void) {
    plat_mutex_lock(&g_mu);
    WinInput in = g_input;
    plat_mutex_unlock(&g_mu);
    return in;
}

// ---------- the window thread
static void read_state(int bump) {
    int pw = 0, ph = 0;
    SDL_GetWindowSizeInPixels(g_win, &pw, &ph);
    SDL_WindowFlags f = SDL_GetWindowFlags(g_win);
    int minimised = (f & SDL_WINDOW_MINIMIZED) != 0;
    int mode = (f & SDL_WINDOW_FULLSCREEN) ? WIN_BORDERLESS : WIN_WINDOWED;
    int w = minimised ? 0 : pw, h = minimised ? 0 : ph;
    int changed = w != atomic_load(&s_w) || h != atomic_load(&s_h) || minimised != atomic_load(&s_min) ||
                  mode != atomic_load(&s_mode);
    atomic_store(&s_w, w), atomic_store(&s_h, h), atomic_store(&s_min, minimised), atomic_store(&s_mode, mode);
    atomic_store(&s_focus, (f & SDL_WINDOW_INPUT_FOCUS) != 0);
    if (changed || bump) atomic_fetch_add(&s_gen, 1);
}

static uint64_t g_resize_at;   // when the last size event came, 0 when none is waiting to settle
static int g_last_pw, g_last_ph;

static void apply_request(void) {
    int mode = atomic_exchange(&r_mode, -1);
    if (mode < 0) return;
    int w = atomic_load(&r_w), h = atomic_load(&r_h);
    SDL_WindowFlags f = SDL_GetWindowFlags(g_win);
    int full = (f & SDL_WINDOW_FULLSCREEN) != 0;
    if (mode == WIN_BORDERLESS && !full) {
        SDL_SetWindowFullscreenMode(g_win, NULL);   // the desktop's own mode: no mode change
        SDL_SetWindowFullscreen(g_win, true);
    } else if (mode == WIN_WINDOWED && full) {
        SDL_SetWindowFullscreen(g_win, false);
    }
    if (mode == WIN_WINDOWED && w > 0 && h > 0) {
        int pw = 0, ph = 0;
        SDL_GetWindowSizeInPixels(g_win, &pw, &ph);
        if (!(f & SDL_WINDOW_MINIMIZED) && !full && (pw != w || ph != h)) SDL_SetWindowSize(g_win, w, h);
    }
    SDL_SyncWindow(g_win);
    read_state(1);
}

static void open_pad(SDL_JoystickID id, SDL_Gamepad **pads) {
    for (int i = 0; i < MAX_PADS; i++)
        if (pads[i] && SDL_GetGamepadID(pads[i]) == id) return;
    for (int i = 0; i < MAX_PADS; i++)
        if (!pads[i]) {
            pads[i] = SDL_OpenGamepad(id);
            return;
        }
}

static void close_pad(SDL_JoystickID id, SDL_Gamepad **pads) {
    for (int i = 0; i < MAX_PADS; i++)
        if (pads[i] && SDL_GetGamepadID(pads[i]) == id) {
            SDL_CloseGamepad(pads[i]);
            pads[i] = NULL;
        }
}

static void note_close(void) {
    if (atomic_exchange(&s_close, 1)) return;
    WinInput in = input_copy();
    if (in.closed) in.closed();
}

static void handle(const SDL_Event *e, SDL_Gamepad **pads, float *px, float *py, int *down) {
    WinInput in = input_copy();
    if (e->type == g_ev_job) {
        Job *j = e->user.data1;
        j->fn(j->arg);
        plat_mutex_lock(&g_mu);
        j->done = 1;
        plat_cond_broadcast(&g_cv);
        plat_mutex_unlock(&g_mu);
        return;
    }
    if (e->type == g_ev_wake) {
        apply_request();
        return;
    }
    switch (e->type) {
    case SDL_EVENT_QUIT:
    case SDL_EVENT_WINDOW_CLOSE_REQUESTED: note_close(); break;
    case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    case SDL_EVENT_WINDOW_RESIZED:
        read_state(0);
        g_resize_at = plat_now_ns();
        break;
    case SDL_EVENT_WINDOW_MINIMIZED:
    case SDL_EVENT_WINDOW_RESTORED:
    case SDL_EVENT_WINDOW_MAXIMIZED:
    case SDL_EVENT_WINDOW_SHOWN:
    case SDL_EVENT_WINDOW_HIDDEN:
    case SDL_EVENT_WINDOW_EXPOSED:
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST:
    case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
    case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN: read_state(0); break;
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
        if (e->key.repeat) break;
        if (in.button && e->key.key == SDLK_F1) in.button(BTN_F1, e->key.down ? 1 : 0);
        else if (in.button && e->key.key == SDLK_ESCAPE) in.button(BTN_ESCAPE, e->key.down ? 1 : 0);
        break;
    case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        int ww = 0, wh = 0, pw = 0, ph = 0;
        SDL_GetWindowSize(g_win, &ww, &wh);
        SDL_GetWindowSizeInPixels(g_win, &pw, &ph);
        float sx = ww > 0 ? (float)pw / (float)ww : 1.0f, sy = wh > 0 ? (float)ph / (float)wh : 1.0f;
        if (e->type == SDL_EVENT_MOUSE_MOTION) *px = e->motion.x * sx, *py = e->motion.y * sy;
        else {
            *px = e->button.x * sx, *py = e->button.y * sy;
            if (e->button.button == SDL_BUTTON_LEFT) *down = e->button.down ? 1 : 0;
        }
        if (in.pointer) in.pointer(*px, *py, *down);
        break;
    }
    case SDL_EVENT_GAMEPAD_ADDED: open_pad(e->gdevice.which, pads); break;
    case SDL_EVENT_GAMEPAD_REMOVED: close_pad(e->gdevice.which, pads); break;
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        int b = e->gbutton.button == SDL_GAMEPAD_BUTTON_GUIDE ? BTN_GUIDE
              : e->gbutton.button == SDL_GAMEPAD_BUTTON_BACK  ? BTN_MINUS
              : e->gbutton.button == SDL_GAMEPAD_BUTTON_START ? BTN_PLUS : -1;
        if (b >= 0 && in.button) in.button(b, e->gbutton.down ? 1 : 0);
        break;
    }
    default: break;
    }
}

// ---------- the host input layer's producer
// The first gamepad opened is player 1's controller. Without one, the keyboard stands in for its buttons and sticks.
// Which gamepad button or keyboard key presses each of the controller's buttons is the remap the settings
// store holds (oot2.ini's [input] keys, input_pad_* and input_key_*; oot2/hid_input.h), read fresh every sample, so a
// change is live at once. The menu's own keys (F1, Escape, the pad's Guide, Back with Start) stay fixed above all of it.
static SDL_Gamepad *first_pad(SDL_Gamepad **pads) {
    for (int i = 0; i < MAX_PADS; i++)
        if (pads[i]) return pads[i];
    return NULL;
}

static int32_t stick(Sint16 v, int32_t dead) { return hid_host_input_deadzone((int32_t)v, dead); }

// A gamepad source's SDL fact, one of its two fields set and the other -1 (in HidPadSource order, designated so the
// order is written where the store names it). The two triggers are axes: they press past half travel, as before.
typedef struct PadRef { int btn; int axis; } PadRef;
static const PadRef pad_ref[HID_PAD_SOURCES] = {
    [HID_PAD_SOUTH] = {SDL_GAMEPAD_BUTTON_SOUTH, -1},
    [HID_PAD_EAST] = {SDL_GAMEPAD_BUTTON_EAST, -1},
    [HID_PAD_WEST] = {SDL_GAMEPAD_BUTTON_WEST, -1},
    [HID_PAD_NORTH] = {SDL_GAMEPAD_BUTTON_NORTH, -1},
    [HID_PAD_LEFT_SHOULDER] = {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, -1},
    [HID_PAD_RIGHT_SHOULDER] = {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, -1},
    [HID_PAD_BACK] = {SDL_GAMEPAD_BUTTON_BACK, -1},
    [HID_PAD_START] = {SDL_GAMEPAD_BUTTON_START, -1},
    [HID_PAD_DPAD_UP] = {SDL_GAMEPAD_BUTTON_DPAD_UP, -1},
    [HID_PAD_DPAD_DOWN] = {SDL_GAMEPAD_BUTTON_DPAD_DOWN, -1},
    [HID_PAD_DPAD_LEFT] = {SDL_GAMEPAD_BUTTON_DPAD_LEFT, -1},
    [HID_PAD_DPAD_RIGHT] = {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, -1},
    [HID_PAD_LEFT_STICK] = {SDL_GAMEPAD_BUTTON_LEFT_STICK, -1},
    [HID_PAD_RIGHT_STICK] = {SDL_GAMEPAD_BUTTON_RIGHT_STICK, -1},
    [HID_PAD_LEFT_TRIGGER] = {-1, SDL_GAMEPAD_AXIS_LEFT_TRIGGER},
    [HID_PAD_RIGHT_TRIGGER] = {-1, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER},
};

// The keyboard's scancode for each source (HidKeySource order). The menu's own keys (F1, Escape) are not among them.
static const SDL_Scancode key_sc[HID_KEY_SOURCES] = {
    [HID_KEY_UP] = SDL_SCANCODE_UP, [HID_KEY_DOWN] = SDL_SCANCODE_DOWN,
    [HID_KEY_LEFT] = SDL_SCANCODE_LEFT, [HID_KEY_RIGHT] = SDL_SCANCODE_RIGHT,
    [HID_KEY_A] = SDL_SCANCODE_A, [HID_KEY_B] = SDL_SCANCODE_B,
    [HID_KEY_C] = SDL_SCANCODE_C, [HID_KEY_D] = SDL_SCANCODE_D,
    [HID_KEY_E] = SDL_SCANCODE_E, [HID_KEY_F] = SDL_SCANCODE_F,
    [HID_KEY_G] = SDL_SCANCODE_G, [HID_KEY_H] = SDL_SCANCODE_H,
    [HID_KEY_I] = SDL_SCANCODE_I, [HID_KEY_J] = SDL_SCANCODE_J,
    [HID_KEY_K] = SDL_SCANCODE_K, [HID_KEY_L] = SDL_SCANCODE_L,
    [HID_KEY_M] = SDL_SCANCODE_M, [HID_KEY_N] = SDL_SCANCODE_N,
    [HID_KEY_O] = SDL_SCANCODE_O, [HID_KEY_P] = SDL_SCANCODE_P,
    [HID_KEY_Q] = SDL_SCANCODE_Q, [HID_KEY_R] = SDL_SCANCODE_R,
    [HID_KEY_S] = SDL_SCANCODE_S, [HID_KEY_T] = SDL_SCANCODE_T,
    [HID_KEY_U] = SDL_SCANCODE_U, [HID_KEY_V] = SDL_SCANCODE_V,
    [HID_KEY_W] = SDL_SCANCODE_W, [HID_KEY_X] = SDL_SCANCODE_X,
    [HID_KEY_Y] = SDL_SCANCODE_Y, [HID_KEY_Z] = SDL_SCANCODE_Z,
    [HID_KEY_RETURN] = SDL_SCANCODE_RETURN, [HID_KEY_SPACE] = SDL_SCANCODE_SPACE,
    [HID_KEY_TAB] = SDL_SCANCODE_TAB, [HID_KEY_BACKSPACE] = SDL_SCANCODE_BACKSPACE,
    [HID_KEY_LSHIFT] = SDL_SCANCODE_LSHIFT, [HID_KEY_RSHIFT] = SDL_SCANCODE_RSHIFT,
    [HID_KEY_LCTRL] = SDL_SCANCODE_LCTRL, [HID_KEY_RCTRL] = SDL_SCANCODE_RCTRL,
};

// The sources held this instant, one bit each in the enums' order; hid_remap_buttons (oot2/hid_input.h) applies the remap
// to the two bitmasks, so nothing of the mapping itself lives here.
static uint32_t pad_sources(SDL_Gamepad *g) {
    uint32_t m = 0;
    for (int i = 0; i < HID_PAD_SOURCES; i++) {
        int held = pad_ref[i].btn >= 0
                       ? SDL_GetGamepadButton(g, (SDL_GamepadButton)pad_ref[i].btn)
                       : SDL_GetGamepadAxis(g, (SDL_GamepadAxis)pad_ref[i].axis) > 16384;
        if (held) m |= 1u << (uint32_t)i;
    }
    return m;
}

static uint64_t key_sources(const bool *k) {
    uint64_t m = 0;
    for (int i = 0; i < HID_KEY_SOURCES; i++)
        if (k[key_sc[i]]) m |= UINT64_C(1) << (uint32_t)i;
    return m;
}

// The store's remap, read fresh: for console button i, its gamepad source and its keyboard source, in the console-button
// order the store's keys and oot2/hid_input.h's HID_REMAP_BUTTONS share (a, b, x, y, l, r, zl, zr, plus, minus, left,
// up, right, down, stickl, stickr). The store keeps each value inside its enum; the NONE checks are a backstop that turn
// anything else into "no source".
static void read_remap(HidRemap *out) {
    static const RtSettingId pad_ids[HID_REMAP_BUTTONS] = {
        [0] = RT_SET_input_pad_a, [1] = RT_SET_input_pad_b, [2] = RT_SET_input_pad_x, [3] = RT_SET_input_pad_y,
        [4] = RT_SET_input_pad_l, [5] = RT_SET_input_pad_r, [6] = RT_SET_input_pad_zl, [7] = RT_SET_input_pad_zr,
        [8] = RT_SET_input_pad_plus, [9] = RT_SET_input_pad_minus, [10] = RT_SET_input_pad_left,
        [11] = RT_SET_input_pad_up, [12] = RT_SET_input_pad_right, [13] = RT_SET_input_pad_down,
        [14] = RT_SET_input_pad_stickl, [15] = RT_SET_input_pad_stickr,
    };
    static const RtSettingId key_ids[HID_REMAP_BUTTONS] = {
        [0] = RT_SET_input_key_a, [1] = RT_SET_input_key_b, [2] = RT_SET_input_key_x, [3] = RT_SET_input_key_y,
        [4] = RT_SET_input_key_l, [5] = RT_SET_input_key_r, [6] = RT_SET_input_key_zl, [7] = RT_SET_input_key_zr,
        [8] = RT_SET_input_key_plus, [9] = RT_SET_input_key_minus, [10] = RT_SET_input_key_left,
        [11] = RT_SET_input_key_up, [12] = RT_SET_input_key_right, [13] = RT_SET_input_key_down,
        [14] = RT_SET_input_key_stickl, [15] = RT_SET_input_key_stickr,
    };
    for (int i = 0; i < HID_REMAP_BUTTONS; i++) {
        int64_t p = rt_setting(pad_ids[i]), k = rt_setting(key_ids[i]);
        out->pad[i] = p >= 0 && p < HID_PAD_NONE ? (uint8_t)p : (uint8_t)HID_PAD_NONE;
        out->key[i] = k >= 0 && k < HID_KEY_NONE ? (uint8_t)k : (uint8_t)HID_KEY_NONE;
    }
}

static void publish_input(SDL_Gamepad **pads) {
    WinInput wi = input_copy();
    int suppressed = wi.menu_open && wi.menu_open();
    hid_host_input_suppress(suppressed);
    if (suppressed) return;   // the menu is open: read returns a zeroed snapshot, publish nothing new
    HidHostInput in = {0};
    int64_t dead = rt_setting(RT_SET_input_dead_zone);
    int32_t dz = dead >= 0 && dead <= 32767 ? (int32_t)dead : 3200;
    HidRemap map;
    read_remap(&map);
    SDL_Gamepad *g = first_pad(pads);
    if (g) {
        in.buttons = hid_remap_buttons(&map, pad_sources(g), 0);   // with a pad open, the keyboard does not stand in
        in.lx = stick(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTX), dz);
        in.ly = hid_host_input_axis_y(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_LEFTY), dz);
        in.rx = stick(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTX), dz);
        in.ry = hid_host_input_axis_y(SDL_GetGamepadAxis(g, SDL_GAMEPAD_AXIS_RIGHTY), dz);
    } else {
        int n = 0;
        const bool *keys = SDL_GetKeyboardState(&n);
        (void)n;
        if (keys) {
            uint64_t down = key_sources(keys);
            in.buttons = hid_remap_buttons(&map, 0, down);
            hid_keyboard_sticks(down, &in);
        }
    }
    hid_host_input_publish(&in);
}

// The vibration requests nn::hid made (oot2/hid_input.h), played on the open gamepad; with no gamepad the requests are
// dropped so a stale rumble never fires at a pad connected later.
static void apply_rumble(SDL_Gamepad **pads) {
    HidRumble r;
    SDL_Gamepad *g = first_pad(pads);
    if (!g) {
        while (hid_rumble_take(&r)) {}
        return;
    }
    if (hid_rumble_take(&r)) SDL_RumbleGamepad(g, r.low, r.high, r.duration_ms);
}

static void wake_rumble(void) {
    SDL_Event e;
    SDL_zero(e);
    e.type = g_ev_wake;
    SDL_PushEvent(&e);   // the mailbox retains the request even if the event queue is full
}

static void thread_main(void *arg) {
    (void)arg;
    SDL_SetMainReady();
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMEPAD)) {
        fail("SDL_Init");
        goto done;
    }
    g_thread_id = SDL_GetCurrentThreadID();
    g_ev_wake = SDL_RegisterEvents(2);
    if (!g_ev_wake) {
        fail("SDL_RegisterEvents");
        goto quit;
    }
    g_ev_job = g_ev_wake + 1;
    if (!SDL_Vulkan_LoadLibrary(NULL)) {
        fail("SDL_Vulkan_LoadLibrary");
        goto quit;
    }
    int w = g_cfg.width > 0 ? g_cfg.width : 1280, h = g_cfg.height > 0 ? g_cfg.height : 720;
    g_win = SDL_CreateWindow(g_title, w, h, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!g_win) {
        fail("SDL_CreateWindow");
        goto quit;
    }
    SDL_SetWindowMinimumSize(g_win, 320, 180);
    if (g_cfg.mode == WIN_BORDERLESS) {
        SDL_SetWindowFullscreenMode(g_win, NULL);
        SDL_SetWindowFullscreen(g_win, true);
    }
    SDL_ShowWindow(g_win);
    SDL_SyncWindow(g_win);
    read_state(1);
    {
        int pw, ph;
        SDL_GetWindowSizeInPixels(g_win, &pw, &ph);
        g_last_pw = pw, g_last_ph = ph;
    }
    SDL_Gamepad *pads[MAX_PADS] = {0};
    int n = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&n);
    for (int i = 0; ids && i < n; i++) open_pad(ids[i], pads);
    SDL_free(ids);
    hid_rumble_set_wake(wake_rumble);
    plat_mutex_lock(&g_mu);
    g_ready = 1;
    plat_cond_broadcast(&g_cv);
    plat_mutex_unlock(&g_mu);

    float px = 0, py = 0;
    int down = 0;
    for (;;) {
        plat_mutex_lock(&g_mu);
        int stop = g_stop;   // win_close sets it
        plat_mutex_unlock(&g_mu);
        if (stop) break;
        SDL_Event e;
        if (SDL_WaitEventTimeout(&e, WAIT_MS)) {
            do handle(&e, pads, &px, &py, &down);
            while (SDL_PollEvent(&e));
        }
        if (atomic_load(&r_mode) >= 0) apply_request();
        if (g_resize_at && plat_now_ns() - g_resize_at >= SETTLE_NS) {
            g_resize_at = 0;
            WinState st;
            win_state(&st);
            WinInput in = input_copy();
            if (in.resized && st.mode == WIN_WINDOWED && !st.minimised && st.width > 0)
                in.resized(st.width, st.height);
        }
        apply_rumble(pads);
        publish_input(pads);
    }
    {   // a job queued as the loop ended still runs: its caller waits for it
        SDL_Event e;
        while (SDL_PollEvent(&e))
            if (e.type == g_ev_job) handle(&e, pads, &px, &py, &down);
    }
    for (int i = 0; i < MAX_PADS; i++)
        if (pads[i]) SDL_CloseGamepad(pads[i]);
    SDL_DestroyWindow(g_win);
    g_win = NULL;
quit:
    hid_rumble_set_wake(NULL);
    SDL_Vulkan_UnloadLibrary();
    SDL_Quit();
done:
    plat_mutex_lock(&g_mu);
    if (!g_ready) g_ready = -1;
    g_ended = 1;
    plat_cond_broadcast(&g_cv);
    plat_mutex_unlock(&g_mu);
}

int win_open(const WinConfig *cfg) {
    plat_mutex_lock(&g_mu);
    if (g_thread) {
        plat_mutex_unlock(&g_mu);
        return 1;
    }
    g_cfg = *cfg;
    snprintf(g_title, sizeof g_title, "%s", cfg->title && *cfg->title ? cfg->title : "oot2");
    g_ready = g_stop = g_ended = 0;
    atomic_store(&s_close, 0);
    snprintf(g_why, sizeof g_why, "the window thread did not start");
    g_thread = plat_thread_start(thread_main, NULL, 1ull << 20);
    if (!g_thread) {
        plat_mutex_unlock(&g_mu);
        return 0;
    }
    while (!g_ready) plat_cond_wait(&g_cv, &g_mu);
    int ok = g_ready > 0;
    if (ok) __atomic_store_n(&g_open, 1, __ATOMIC_RELEASE);
    else {
        while (!g_ended) plat_cond_wait(&g_cv, &g_mu);
        plat_thread_free(g_thread);
        g_thread = NULL;
    }
    plat_mutex_unlock(&g_mu);
    return ok;
}

void win_close(void) {
    plat_mutex_lock(&g_mu);
    if (!g_thread) {
        plat_mutex_unlock(&g_mu);
        return;
    }
    __atomic_store_n(&g_open, 0, __ATOMIC_RELEASE);   // no job is taken from here on: on_thread returns at once
    g_stop = 1;   // the loop ends at its next wake-up (at most WAIT_MS)
    while (!g_ended) plat_cond_wait_ns(&g_cv, &g_mu, 100000000ull);
    plat_thread_free(g_thread);
    g_thread = NULL;
    plat_mutex_unlock(&g_mu);
}

void win_request(int mode, int width, int height) {
    if (!win_is_open()) return;
    atomic_store(&r_w, width), atomic_store(&r_h, height);
    atomic_store(&r_mode, mode);
    SDL_Event e;
    SDL_zero(e);
    e.type = g_ev_wake;
    SDL_PushEvent(&e);   // safe from any thread; the loop also looks at the mailbox at each wake-up
}

// ---------- events posted as if made by the player (window.h)
void win_post_button(int button, int down) {
    if (!win_is_open() || (button != BTN_F1 && button != BTN_ESCAPE)) return;
    SDL_Event e;
    SDL_zero(e);
    e.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
    e.key.key = button == BTN_F1 ? SDLK_F1 : SDLK_ESCAPE;
    e.key.down = down ? true : false;
    SDL_PushEvent(&e);
}

void win_post_pointer(float x, float y, int down) {
    if (!win_is_open()) return;
    SDL_Event e;
    SDL_zero(e);
    e.type = SDL_EVENT_MOUSE_MOTION;
    e.motion.x = x, e.motion.y = y;
    SDL_PushEvent(&e);
    SDL_zero(e);
    e.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
    e.button.button = SDL_BUTTON_LEFT;
    e.button.down = down ? true : false;
    e.button.x = x, e.button.y = y;
    SDL_PushEvent(&e);
}

static void do_topmost(void *arg) {
    SDL_SetWindowAlwaysOnTop(g_win, *(int *)arg ? true : false);
    if (*(int *)arg) SDL_RaiseWindow(g_win);
    SDL_SyncWindow(g_win);
}
static void on_thread(void (*fn)(void *), void *arg);
void win_post_topmost(int on) {
    if (win_is_open()) on_thread(do_topmost, &on);
}

static void do_minimise(void *arg) {
    if (*(int *)arg) SDL_MinimizeWindow(g_win);
    else SDL_RestoreWindow(g_win);
    SDL_SyncWindow(g_win);
    read_state(1);
}
static void on_thread(void (*fn)(void *), void *arg);
void win_post_minimise(int on) {
    if (win_is_open()) on_thread(do_minimise, &on);
}

void win_post_close(void) {
    if (!win_is_open()) return;
    SDL_Event e;
    SDL_zero(e);
    e.type = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
    e.window.windowID = SDL_GetWindowID(g_win);
    SDL_PushEvent(&e);
}

typedef struct { int x, y, ok; } Origin;
static void read_origin(void *arg) {
    Origin *o = arg;
    o->ok = SDL_GetWindowPosition(g_win, &o->x, &o->y) ? 1 : 0;
}
int win_client_origin(int *x, int *y) {
    WinState st;
    win_state(&st);
    if (!win_is_open() || st.minimised) return 0;
    Origin o = {0, 0, 0};
    on_thread(read_origin, &o);
    *x = o.x, *y = o.y;
    return o.ok;
}

// ---------- the Vulkan surface
// Run fn(arg) on the window's thread and return when it has run. With the window closed (or closing) nothing runs: what
// the jobs act on, the window and SDL, is going or gone. The wait is in slices and ends when the thread has, so a job that
// came in as the loop ended (the loop runs the jobs it still holds before it stops) is never waited for in vain.
static void on_thread(void (*fn)(void *), void *arg) {
    if (!win_is_open()) return;
    if (SDL_GetCurrentThreadID() == g_thread_id) {
        fn(arg);
        return;
    }
    Job j = {fn, arg, 0};
    SDL_Event e;
    SDL_zero(e);
    e.type = g_ev_job;
    e.user.data1 = &j;
    if (!SDL_PushEvent(&e)) return;
    plat_mutex_lock(&g_mu);
    while (!j.done && !g_ended) plat_cond_wait_ns(&g_cv, &g_mu, 50000000ull);
    plat_mutex_unlock(&g_mu);
}

static const char *const *vk_instance_extensions(uint32_t *count) {
    Uint32 n = 0;
    const char *const *e = SDL_Vulkan_GetInstanceExtensions(&n);
    *count = e ? n : 0;
    return e;
}

static int vk_can_present(void *instance, void *phys, uint32_t family) {
    return SDL_Vulkan_GetPresentationSupport((VkInstance)instance, (VkPhysicalDevice)phys, family) ? 1 : 0;
}

typedef struct { void *instance; uint64_t surface; int ok; } SurfaceJob;
static void make_surface(void *arg) {
    SurfaceJob *s = arg;
    VkSurfaceKHR surface = 0;
    s->ok = SDL_Vulkan_CreateSurface(g_win, (VkInstance)s->instance, NULL, &surface) ? 1 : 0;
    s->surface = (uint64_t)(uintptr_t)surface;
    if (!s->ok) fail("SDL_Vulkan_CreateSurface");
}
static void drop_surface(void *arg) {
    SurfaceJob *s = arg;
    SDL_Vulkan_DestroySurface((VkInstance)s->instance, (VkSurfaceKHR)(uintptr_t)s->surface, NULL);
}

static int vk_create_surface(void *instance, uint64_t *surface) {
    SurfaceJob s = {instance, 0, 0};
    on_thread(make_surface, &s);
    *surface = s.surface;
    return s.ok;
}

static void vk_destroy_surface(void *instance, uint64_t surface) {
    SurfaceJob s = {instance, surface, 0};
    on_thread(drop_surface, &s);
}

typedef struct { char *out; size_t size; int ok; } DisplayJob;
static void read_display(void *arg) {
    DisplayJob *j = arg;
    SDL_DisplayID id = SDL_GetDisplayForWindow(g_win);
    const SDL_DisplayMode *m = id ? SDL_GetCurrentDisplayMode(id) : NULL;
    if (!m) {
        snprintf(j->out, j->size, "unknown (%s)", SDL_GetError());
        return;
    }
    snprintf(j->out, j->size, "%dx%d at %.2f Hz, scale %.2f, display %u", m->w, m->h, (double)m->refresh_rate,
             (double)m->pixel_density, (unsigned)id);
    j->ok = 1;
}
static int vk_display(char *out, size_t size) {
    DisplayJob j = {out, size, 0};
    on_thread(read_display, &j);
    if (!win_is_open()) snprintf(out, size, "no window");
    return j.ok;
}

const WinVulkan win_vulkan = {vk_instance_extensions, vk_can_present, vk_create_surface, vk_destroy_surface, win_state,
                              vk_display};

#endif
