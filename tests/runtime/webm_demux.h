// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A small streaming reader of WebM / Matroska files for the headless movie decode (webm_decode.c). The game's own
// container code runs in the guest and hands the runtime one temporal unit at a time; this reader does the same job on
// the host so a whole file can be pushed through the decoder seam with no guest and no window. It reads the header
// (EBML header, segment, info, tracks) and then returns the blocks of the clusters one at a time, never holding more
// than one block. A file is read from a window of a larger file (an offset and a size), so a file stored inside an
// image can be read in place.
//
// What it refuses, by name: a file that ends before the size it declares (truncated), a size or id that cannot be
// read, a byte that starts no element id, a block group that is malformed, a block of a video track with lacing, a
// block of a track the track list lacks, a track list after the first cluster. Only the first WEBM_TRACKS_MAX tracks of
// the list are kept; the others are dropped without a word, and a block of one of them is then refused as a block of a
// track the list lacks.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define WEBM_TRACKS_MAX 8

typedef struct WebmTrack {
    uint32_t number;
    uint32_t type;                 // 1 video, 2 audio, 17 subtitle, ...
    char codec[40];                // the CodecID, such as V_AV1 or A_OPUS
    uint32_t width, height;        // video: PixelWidth, PixelHeight
    uint32_t channels, bits;       // audio: Channels, BitDepth (0 when absent)
    double rate;                   // audio: SamplingFrequency (0 when absent)
    uint64_t default_ns;           // DefaultDuration, 0 when absent
    uint64_t codec_delay_ns;       // CodecDelay, 0 when absent
    uint8_t priv[16];              // the first bytes of CodecPrivate
    uint32_t priv_size;            // the full size of CodecPrivate
} WebmTrack;

typedef struct WebmPacket {
    uint32_t track;                // the track number the block belongs to
    int64_t ticks;                 // cluster timecode plus the block's, in TimecodeScale units
    int key;                       // key frame (SimpleBlock flag; for a BlockGroup: no ReferenceBlock)
    int lacing;                    // 0 none, 1 Xiph, 2 fixed-size, 3 EBML
    const uint8_t *data;           // the block's payload (valid until the next webm_next)
    size_t size;
} WebmPacket;

typedef struct Webm {
    FILE *f;
    uint64_t base, limit;          // the window in the file: [base, base + limit)
    uint64_t pos;                  // read position, relative to base
    char doctype[16];
    uint64_t timecode_scale;       // nanoseconds per tick; 1000000 when absent
    double duration_ticks;         // Info/Duration in ticks, -1 when absent
    int64_t segment_end;           // relative to base; -1 when the segment's size is unknown
    WebmTrack track[WEBM_TRACKS_MAX];
    int tracks;
    int in_cluster;
    int64_t cluster_end;           // -1 when the cluster's size is unknown
    int64_t cluster_tc;
    uint8_t *buf;
    size_t cap;
} Webm;

// Opens the file (or the window of size bytes at offset; size 0 means to the end of the file) and reads the header up to
// the first cluster. 0 on success, -1 with err filled.
int webm_open(Webm *w, const char *path, uint64_t offset, uint64_t size, char *err, size_t errsz);
// The next block: 1 with *p filled, 0 at the end of the segment, -1 with err filled (truncated file, bad element).
int webm_next(Webm *w, WebmPacket *p, char *err, size_t errsz);
void webm_close(Webm *w);
