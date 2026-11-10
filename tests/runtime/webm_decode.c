// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Headless decode of a WebM file through the runtime's AV1 decoder seam (runtime/video/av1_dav1d.c), with no window,
// no guest and no service: the file's blocks go to av1_send one by one, as the movie library's units go to the video
// decoder service, and the pictures are counted and hashed. Built by tests/test_runtime_webm.py; it can be pointed at any
// WebM file, or at a window of a larger file (an image holding the movies).
//
//   webm_decode FILE [OFFSET SIZE]
//
// Prints one line for the container, one for each track, one for each decoded stream and a last line "result OK ..." or
// "result FAIL reason ...", and exits 0 or 1. Exit 2 is a usage error. A file passes when
//   - the container reads to its end (a file shorter than the segment it declares is "truncated"),
//   - the first video track is AV1 (any other codec fails by its name),
//   - every video block is accepted by the decoder and makes one picture of the track's size, and
//   - the number of video blocks agrees, to one frame, with the duration the container declares divided by the frame
//     period (the track's default duration, else the mean spacing of the blocks).
// The audio tracks are not decoded; their codec, shape and block count are read from the container.
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../runtime/video/av1.h"
#include "webm_demux.h"

static uint64_t mix(uint64_t h, const uint8_t *p, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

typedef struct Stat {
    uint64_t blocks, bytes;
    int64_t first, last;
    int seen;
} Stat;

static void note(Stat *s, const WebmPacket *p) {
    s->blocks++;
    s->bytes += p->size;
    if (!s->seen) s->first = p->ticks, s->seen = 1;
    s->last = p->ticks;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "usage: webm_decode FILE [OFFSET SIZE]\n");
        return 2;
    }
    uint64_t off = argc == 4 ? strtoull(argv[2], NULL, 10) : 0, size = argc == 4 ? strtoull(argv[3], NULL, 10) : 0;
    char err[256] = "";
    Webm w;
    if (webm_open(&w, argv[1], off, size, err, sizeof err) < 0) {
        printf("result FAIL open: %s\n", err);
        return 1;
    }
    double scale = (double)w.timecode_scale;
    double duration_ms = w.duration_ticks < 0 ? -1 : w.duration_ticks * scale / 1e6;
    printf("container doctype=%s timecode_scale=%" PRIu64 " duration_ms=%.3f segment_bytes=%" PRId64 " tracks=%d\n", w.doctype,
           w.timecode_scale, duration_ms, w.segment_end, w.tracks);
    const WebmTrack *video = NULL;
    for (int i = 0; i < w.tracks; i++) {
        const WebmTrack *t = &w.track[i];
        printf("track number=%u type=%u codec=%s", t->number, t->type, t->codec);
        if (t->type == 1) printf(" size=%ux%u default_ns=%" PRIu64, t->width, t->height, t->default_ns);
        if (t->type == 2) printf(" channels=%u rate=%.0f bits=%u codec_delay_ns=%" PRIu64, t->channels, t->rate, t->bits, t->codec_delay_ns);
        if (t->type == 1 && !strcmp(t->codec, "V_AV1") && t->priv_size >= 4 && t->priv[0] == 0x81) {
            int hbd = t->priv[2] >> 6 & 1, twelve = t->priv[2] >> 5 & 1;
            printf(" av1c_profile=%d av1c_level=%d av1c_bits=%d av1c_mono=%d av1c_subsampling=%d%d", t->priv[1] >> 5, t->priv[1] & 31,
                   hbd ? (twelve ? 12 : 10) : 8, t->priv[2] >> 4 & 1, t->priv[2] >> 3 & 1, t->priv[2] >> 2 & 1);
        }
        printf("\n");
        if (t->type == 1 && !video) video = t;
    }
    if (!video) {
        printf("result FAIL no video track\n");
        webm_close(&w);
        return 1;
    }
    if (strcmp(video->codec, "V_AV1") != 0) {
        printf("result FAIL video codec %s is not AV1: the runtime's decoder reads AV1 only\n", video->codec);
        webm_close(&w);
        return 1;
    }
    Av1Decoder *dec = av1_open();
    if (!dec) {
        printf("result FAIL no AV1 decoder (%s)\n", av1_describe());
        webm_close(&w);
        return 1;
    }
    Stat vs = {0}, as[WEBM_TRACKS_MAX] = {{0}};
    uint64_t pictures = 0, refused = 0, wrong_size = 0, hash = 0xcbf29ce484222325ull, keys = 0;
    int64_t first_refused = -1;
    int first_key = -1;
    char why[320] = "";
    int rc;
    WebmPacket p;
    while ((rc = webm_next(&w, &p, err, sizeof err)) > 0) {
        if (p.track != video->number) {
            for (int i = 0; i < w.tracks; i++)
                if (w.track[i].number == p.track) note(&as[i], &p);
            continue;
        }
        if (first_key < 0) first_key = p.key;
        keys += (uint64_t)p.key;
        note(&vs, &p);
        if (av1_send(dec, p.data, p.size, p.ticks) != AV1_OK) {
            if (!refused) first_refused = (int64_t)vs.blocks - 1;
            refused++;
            continue;
        }
        Av1Picture pic;
        while (av1_receive(dec, &pic) == AV1_PICTURE) {
            pictures++;
            if ((uint32_t)pic.w != video->width || (uint32_t)pic.h != video->height) wrong_size++;
            for (int y = 0; y < pic.h; y++) hash = mix(hash, pic.plane[0] + y * pic.stride[0], (size_t)pic.w);
            for (int y = 0; y < (pic.h + 1) / 2; y++) {
                hash = mix(hash, pic.plane[1] + y * pic.stride[1], (size_t)(pic.w + 1) / 2);
                hash = mix(hash, pic.plane[2] + y * pic.stride[2], (size_t)(pic.w + 1) / 2);
            }
        }
    }
    av1_close(dec);
    double period_ns = video->default_ns ? (double)video->default_ns
                       : vs.blocks > 1 ? (double)(vs.last - vs.first) * scale / (double)(vs.blocks - 1) : 0;
    double expected = duration_ms >= 0 && period_ns > 0 ? duration_ms * 1e6 / period_ns : -1;
    printf("video track=%u blocks=%" PRIu64 " pictures=%" PRIu64 " refused=%" PRIu64 " wrong_size=%" PRIu64 " keys=%" PRIu64
           " first_ms=%.3f last_ms=%.3f period_ns=%.0f expected_by_duration=%.2f hash=%016" PRIx64 " decoder=\"%s\"\n",
           video->number, vs.blocks, pictures, refused, wrong_size, keys, (double)vs.first * scale / 1e6, (double)vs.last * scale / 1e6,
           period_ns, expected, hash, av1_describe());
    for (int i = 0; i < w.tracks; i++) {
        if (w.track[i].number == video->number) continue;
        printf("other track=%u codec=%s blocks=%" PRIu64 " bytes=%" PRIu64 " last_ms=%.3f\n", w.track[i].number, w.track[i].codec,
               as[i].blocks, as[i].bytes, (double)as[i].last * scale / 1e6);
    }
    webm_close(&w);
    if (rc < 0) snprintf(why, sizeof why, "%s, after %" PRIu64 " video blocks and %" PRIu64 " pictures", err, vs.blocks, pictures);
    else if (!vs.blocks) snprintf(why, sizeof why, "the video track has no blocks");
    else if (first_key == 0) snprintf(why, sizeof why, "the first video block is not a key frame");
    else if (refused) snprintf(why, sizeof why, "the decoder refused %" PRIu64 " of %" PRIu64 " blocks (the first is block %" PRId64 ")", refused,
                               vs.blocks, first_refused);
    else if (pictures != vs.blocks)
        snprintf(why, sizeof why, "%" PRIu64 " blocks made %" PRIu64 " pictures", vs.blocks, pictures);
    else if (wrong_size) snprintf(why, sizeof why, "%" PRIu64 " pictures differ from the track's size", wrong_size);
    else if (expected >= 0 && fabs((double)vs.blocks - expected) > 1.0)
        snprintf(why, sizeof why, "short or long: %" PRIu64 " frames, the container's duration implies %.1f", vs.blocks, expected);
    if (*why) {
        printf("result FAIL %s\n", why);
        return 1;
    }
    printf("result OK frames=%" PRIu64 " duration_check=%s\n", pictures, expected >= 0 ? "yes" : "no duration or period");
    return 0;
}
