// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A host mutex that a guest thread may hold while it has no virtual core.
//
// A guest thread that holds a host lock and gives its core up (rt_kernel_block_begin) must take a core again before it goes
// on to release the lock. If another guest thread on that core blocks on the same lock with its own core held, the first
// never gets its core back: the second keeps it and waits for the lock. The submission gate is where this happens. A caller
// of gpu_sync_begin (gpu_vk_begin) gives its core up while it waits for the gate, and it may hold a lock of its own across
// the call: the texel copies' staging lock (vk_image.c) and the capture's lock (vk_capture.c). Every guest thread takes such
// a lock with rt_kernel_lock, so one that has to wait for it waits without a core.
//
// Taking the gate before the lock, the order vk_texture.c keeps, does not suit these two: both read their buffer out after
// the submission has ended, so the lock outlives the gate, and a thread that held the gate while it waited for the lock
// would stop every other submission for the device's whole wait.
#pragma once
#include "oot2/kernel.h"
#include "plat.h"

// plat_mutex_lock, and a guest thread that finds the mutex taken waits for it without its virtual core, takes one again
// once the mutex is its, and goes on holding the mutex. A thread that is no guest thread (the start-up's, a test's) holds
// no core, and the pair is then nothing. The lock is only worth this when the holder can give its core up while it holds the
// lock: one that holds it for a few instructions takes it with plat_mutex_lock.
static inline void rt_kernel_lock(PlatMutex *m) {
    if (plat_mutex_trylock(m)) return;   // free: no wait, and the core stays where it is
    int token = rt_kernel_block_begin();
    plat_mutex_lock(m);
    rt_kernel_block_end(token);
}
