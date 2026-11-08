// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The platform layer's window (architecture D24; runtime/platform, C11 over SDL3): one window, the thread that pumps its
// events, and the few facts the Vulkan back end and the menu need from it. The header names no SDL type and no Vulkan
// type, so the back end (runtime/gpu) includes it and a test can stand in for the window.
//
// One host thread, started by win_open, makes and owns the window and every SDL object: SDL's video calls belong to the
// thread that initialised it. It waits for events, hands the open key, the pointer and the pad buttons to the menu
// through the callbacks of win_set_input (they run on that thread: the open key's close saves the settings file there, with
// the store's lock and a file write, which is why they are the window thread's and never the frame's), keeps the facts below current, and runs the requests other threads make. A request never blocks its caller
// except where it says so: the settings store's live hooks call win_request with the store's lock held, and the
// thread's own save of a resized window's size takes that lock.
//
// Game input (hid) starts here: the thread also produces the host input snapshot (oot2/hid_input.h) from the gamepad or
// the keyboard, and applies the vibration requests nn::hid made. The header declares only the menu- and pointer-facing
// calls below; the hid producer is window.c's own, and no SDK type crosses this header.
//
// A build without SDL3 (tools/fetch_sdl3.py, SDL3_DIR) compiles the same file to refusals: win_open returns 0 and
// win_why says so, and the program goes on presenting to the offscreen capture (oot2/vk_capture.h).
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum WinMode { WIN_WINDOWED, WIN_BORDERLESS } WinMode;

typedef struct WinConfig {
    const char *title;
    int width, height;   // the client area in pixels of a windowed window (a borderless one takes the screen's size)
    int mode;            // WinMode
} WinConfig;

// What the window is now, from any thread. width and height are the client area in pixels (what a swapchain is made at;
// 0 while the window is minimised). generation rises at each change of size, minimised or mode.
typedef struct WinState {
    int width, height;
    int minimised;
    int focused;
    int mode;               // WinMode
    int close_requested;    // set once the window's close button, Alt+F4 or a quit event came; never cleared
    uint64_t generation;
} WinState;

// Make the window (shown at once) and start its thread: 1, or 0 with the reason in win_why (no SDL3 in the build, no
// display, SDL refusing). Once only; a second call before win_close returns 1 and changes nothing.
int win_open(const WinConfig *cfg);
const char *win_why(void);
// Destroy the window, stop the thread and shut SDL down. The surface the back end made must be gone first.
void win_close(void);
int win_is_open(void);
void win_state(WinState *out);

// Called on the window's thread. button: the menu's UiButton (oot2/ui.h) as an int and 1 for a press or 0 for a release;
// returns 1 when the menu consumed it. menu_open: returns 1 while the configuration menu is open, so this thread's hid
// producer swallows the game input (hid_host_input_suppress). pointer: the pointer in client pixels, with the left button
// down (1) or up (0). resized: the window's size, in pixels, once it has stopped changing (the program saves it as the
// remembered size); it is also called for a size the program set itself (a request that changed it), and not in borderless
// mode. closed: the window's close request, once; the program ends the run (the window is destroyed by win_close or by
// the end of the process).
typedef struct WinInput {
    int (*button)(int button, int down);
    int (*menu_open)(void);
    void (*pointer)(float x, float y, int down);
    void (*resized)(int width, int height);
    void (*closed)(void);
} WinInput;
void win_set_input(const WinInput *in);

// Ask for a mode and a windowed size (the settings store's live keys); returns at once, the window thread applies it.
void win_request(int mode, int width, int height);

// Events made as if the player had made them, for the tests and for scripted runs: each returns at once and the window's
// own thread takes it from its queue like any other, so the whole path (the callbacks of win_set_input included) runs.
// button is the menu's UiButton: F1 and Escape go in as key events (the pad's buttons are not posted). The pointer is in
// client pixels. Minimising and restoring, and the close request (the close button's own event), complete on the window
// thread, which this call waits for.
void win_post_button(int button, int down);
void win_post_pointer(float x, float y, int down);
void win_post_minimise(int on);
void win_post_close(void);
// Put the window in front of every other and keep it there (a test that looks at the screen needs nothing over it).
void win_post_topmost(int on);
// Where the window's client area starts on the desktop, in pixels (for a test that looks at the screen): 1, or 0 when it is
// minimised or there is no window.
int win_client_origin(int *x, int *y);

// What the Vulkan back end asks of the window, from any thread: the instance extensions the surface needs, whether a
// queue family can present to it (instance and physical device are the Vulkan handles), the surface itself (made and
// destroyed on the window's thread; surface is a VkSurfaceKHR), and the state above. win_vulkan is the platform layer's;
// a test may pass its own to gpu_vk_window (oot2/gpu_vk.h).
typedef struct WinVulkan {
    const char *const *(*instance_extensions)(uint32_t *count);
    int (*can_present)(void *instance, void *phys, uint32_t family);
    int (*create_surface)(void *instance, uint64_t *surface);
    void (*destroy_surface)(void *instance, uint64_t surface);
    void (*state)(WinState *out);
    // The display the window is on, as a line of text for the capture's header (size, refresh rate, scale): 1, or 0 with
    // the reason in out.
    int (*display)(char *out, size_t size);
} WinVulkan;
extern const WinVulkan win_vulkan;

#ifdef __cplusplus
}
#endif
