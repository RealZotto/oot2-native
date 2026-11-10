// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The packet ring (oot2/gpu_ring.h): one producer, one consumer, 64-bit words, packets that never wrap, back-pressure
// when full. Three counters of words run for ever and index the ring through a mask: put_pos (the producer's, packets
// reserved), head (published to the consumer) and tail (released by the consumer, which is how the producer sees room).
// The consumer keeps its own position and publishes the tail in batches and whenever it runs out of packets, so that a
// producer waiting for room, or draining, is never left waiting for a tail the consumer holds back.
//
// Design as in DKB Recomp's runtime/vk_rec.c at 9d8fa02 (GPL-3.0-or-later; there after Savage Wind Recomp #189 and LAS
// Recomp runtime/nvn/vulkan/record.c at a4b53a3, MIT, Copyright (c) 2026 Peiikai): a header word of operation and length,
// a pad instead of a wrapping packet, head and tail published with release stores, an idle consumer that spins and then
// sleeps with a timeout so that a missed wake costs at most the timeout. Changed here: the ring is a module of its own
// (no Vulkan, tested by itself), its size is a parameter, the producer's wait is accounted, a packet bigger than a
// quarter of the ring is refused by name, and the consumer's tail is published in batches.
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu_ring.h"
#include "oot2/memline.h"
#include "oot2/runtime.h"
#include "plat.h"

#define SPINS 4096               // an idle consumer looks this many times before it sleeps
#define SLEEP_SLICE_NS 2000000ull // and sleeps at most this long: a missed wake costs this much
#define SPIN_WAITS 2000          // a producer waiting for room spins this long before it yields
#define TAIL_BATCH 64            // packets released between two publications of the tail

struct GpuRing {
    uint8_t *mem;
    uint64_t words, mask;
    _Alignas(64) uint64_t head;      // published words, written by the producer
    _Alignas(64) uint64_t tail;      // released words, written by the consumer
    _Alignas(64) uint64_t put_pos;   // the producer's
    GpuRingStats st;                 // the producer's counters, but consumer_sleeps
    _Alignas(64) uint64_t cons_pos;  // the consumer's: the next packet
    uint64_t cons_len;               // the packet get returned, in words
    unsigned released;               // since the tail was last published
    uint64_t sleeps;
    int sleeping;                    // the consumer is, or is about to be, waiting on wake
    PlatMutex lock;
    PlatCond wake;
};

int gpu_ring_overwrite;

GpuRing *gpu_ring_new(size_t bytes) {
    if (bytes < 4096 || (bytes & (bytes - 1))) return NULL;
    GpuRing *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->mem = malloc(bytes);
    if (!r->mem) {
        free(r);
        return NULL;
    }
    memline_note("gpu-ring", bytes);
    r->words = bytes / 8;
    r->mask = r->words - 1;
    r->lock = (PlatMutex)PLAT_MUTEX_INIT;
    r->wake = (PlatCond)PLAT_COND_INIT;
    return r;
}

void gpu_ring_free(GpuRing *r) {
    if (!r) return;
    memline_unnote("gpu-ring", (uint64_t)r->words * 8);
    free(r->mem);
    free(r);
}

size_t gpu_ring_bytes(const GpuRing *r) { return (size_t)r->words * 8; }
size_t gpu_ring_max_payload(const GpuRing *r) { return (size_t)(r->words / 4 - 1) * 8; }

static inline uint64_t *word(GpuRing *r, uint64_t pos) { return (uint64_t *)r->mem + (pos & r->mask); }

// ---------- the producer
static void wake(GpuRing *r) {
    plat_mutex_lock(&r->lock);
    plat_cond_signal(&r->wake);
    plat_mutex_unlock(&r->lock);
}

// The common publish is a release store and a look at the consumer's flag, with no fence between them, as DKB Recomp's:
// on x86 the flag may be read before the head is visible, and the consumer, which looked at the head before it went to
// sleep, then sleeps through this packet until its timeout (SLEEP_SLICE_NS). The fenced publish is for the places where
// the producer is about to wait itself, which must not depend on that timeout.
void gpu_ring_publish(GpuRing *r) {
    __atomic_store_n(&r->head, r->put_pos, __ATOMIC_RELEASE);
    if (__atomic_load_n(&r->sleeping, __ATOMIC_RELAXED)) wake(r);
}

static void publish_fenced(GpuRing *r) {
    __atomic_store_n(&r->head, r->put_pos, __ATOMIC_RELEASE);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (__atomic_load_n(&r->sleeping, __ATOMIC_RELAXED)) wake(r);
}

static void wait_space(GpuRing *r, uint64_t need) {
    if (gpu_ring_overwrite) return;
    if (r->put_pos + need - __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE) <= r->words) return;
    uint64_t t0 = plat_now_ns();
    r->st.full_waits++;
    publish_fenced(r);   // what is put so far is whole: the consumer can make room from it
    for (unsigned i = 0; r->put_pos + need - __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE) > r->words; i++) {
        if (i < SPIN_WAITS) plat_pause();
        else plat_yield();
    }
    r->st.full_wait_ns += plat_now_ns() - t0;
}

void *gpu_ring_put(GpuRing *r, uint32_t op, size_t bytes) {
    if (op == GPU_RING_OP_PAD) rt_fatal("gpu: ring: a packet with the pad's operation number");
    if (bytes > gpu_ring_max_payload(r))
        rt_fatal("gpu: ring: a packet of %zu bytes does not fit a ring of %zu bytes (at most %zu)", bytes,
                 gpu_ring_bytes(r), gpu_ring_max_payload(r));
    uint64_t n = 1 + ((uint64_t)bytes + 7) / 8, pos = r->put_pos & r->mask, pad = 0;
    if (pos + n > r->words) pad = r->words - pos;
    wait_space(r, pad + n);
    if (pad) {
        *word(r, r->put_pos) = (uint64_t)GPU_RING_OP_PAD | pad << 32;
        r->put_pos += pad;
        r->st.pads += pad;
    }
    uint64_t *h = word(r, r->put_pos);
    *h = (uint64_t)op | n << 32;
    r->put_pos += n;
    r->st.packets++;
    r->st.words += n;
    return h + 1;
}

void gpu_ring_drain(GpuRing *r) {
    publish_fenced(r);
    for (unsigned i = 0; __atomic_load_n(&r->tail, __ATOMIC_ACQUIRE) != r->put_pos; i++) {
        if (i < SPIN_WAITS) plat_pause();
        else plat_yield();
    }
}

// ---------- the consumer
static void publish_tail(GpuRing *r) {
    if (__atomic_load_n(&r->tail, __ATOMIC_RELAXED) != r->cons_pos)
        __atomic_store_n(&r->tail, r->cons_pos, __ATOMIC_RELEASE);
    r->released = 0;
}

const void *gpu_ring_get(GpuRing *r, uint32_t *op, size_t *bytes, uint64_t wait_ns) {
    uint64_t start = wait_ns && wait_ns != UINT64_MAX ? plat_now_ns() : 0;
    unsigned spins = 0;
    for (;;) {
        uint64_t head = __atomic_load_n(&r->head, __ATOMIC_ACQUIRE);
        while (r->cons_pos != head) {
            uint64_t w = *word(r, r->cons_pos);
            uint32_t o = (uint32_t)w;
            uint64_t n = w >> 32;
            if (n == 0 || n > r->words || r->cons_pos + n > head)
                rt_fatal("gpu: ring: a packet header of operation %u and %llu words at word %llu, with %llu published", o,
                         (unsigned long long)n, (unsigned long long)r->cons_pos, (unsigned long long)head);
            if (o == GPU_RING_OP_PAD) {
                r->cons_pos += n;
                continue;
            }
            r->cons_len = n;
            *op = o;
            *bytes = (size_t)(n - 1) * 8;
            return word(r, r->cons_pos) + 1;
        }
        publish_tail(r);   // nothing left: the producer may be waiting for room, or draining
        if (!wait_ns) return NULL;
        uint64_t slice = SLEEP_SLICE_NS;
        if (wait_ns != UINT64_MAX) {
            uint64_t spent = plat_now_ns() - start;
            if (spent >= wait_ns) return NULL;
            if (wait_ns - spent < slice) slice = wait_ns - spent;
        }
        if (spins < SPINS) {
            spins++;
            plat_pause();
            continue;
        }
        plat_mutex_lock(&r->lock);
        __atomic_store_n(&r->sleeping, 1, __ATOMIC_SEQ_CST);
        if (__atomic_load_n(&r->head, __ATOMIC_SEQ_CST) == r->cons_pos) {
            r->sleeps++;
            plat_cond_wait_ns(&r->wake, &r->lock, slice);
        }
        __atomic_store_n(&r->sleeping, 0, __ATOMIC_RELAXED);
        plat_mutex_unlock(&r->lock);
        spins = 0;
    }
}

void gpu_ring_release(GpuRing *r) {
    r->cons_pos += r->cons_len;
    r->cons_len = 0;
    if (++r->released >= TAIL_BATCH || r->cons_pos - __atomic_load_n(&r->tail, __ATOMIC_RELAXED) >= r->words / 8)
        publish_tail(r);
}

void gpu_ring_stats(const GpuRing *r, GpuRingStats *out) {
    *out = r->st;
    out->consumer_sleeps = r->sleeps;
}
