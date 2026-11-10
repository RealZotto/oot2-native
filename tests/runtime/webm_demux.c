// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The streaming WebM / Matroska reader of the headless movie decode; see webm_demux.h.
#include "webm_demux.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define SEEK64(f, off) _fseeki64((f), (__int64)(off), SEEK_SET)
#define TELL64(f) _ftelli64(f)
#else
#define SEEK64(f, off) fseeko((f), (off_t)(off), SEEK_SET)
#define TELL64(f) ftello(f)
#endif

enum {
    ID_EBML = 0x1A45DFA3, ID_DOCTYPE = 0x4282, ID_SEGMENT = 0x18538067, ID_SEEKHEAD = 0x114D9B74, ID_INFO = 0x1549A966,
    ID_TIMESCALE = 0x2AD7B1, ID_DURATION = 0x4489, ID_TRACKS = 0x1654AE6B, ID_ENTRY = 0xAE, ID_NUMBER = 0xD7,
    ID_TYPE = 0x83, ID_CODEC = 0x86, ID_PRIVATE = 0x63A2, ID_DEFDUR = 0x23E383, ID_VIDEO = 0xE0, ID_WIDTH = 0xB0,
    ID_HEIGHT = 0xBA, ID_AUDIO = 0xE1, ID_RATE = 0xB5, ID_CHANNELS = 0x9F, ID_BITS = 0x6264, ID_DELAY = 0x56AA,
    ID_CLUSTER = 0x1F43B675, ID_TC = 0xE7, ID_SIMPLE = 0xA3, ID_GROUP = 0xA0, ID_BLOCK = 0xA1, ID_REF = 0xFB,
    ID_CUES = 0x1C53BB6B, ID_TAGS = 0x1254C367, ID_CHAPTERS = 0x1043A770, ID_ATTACH = 0x1941A469,
};

#define UNKNOWN (-1)

static void fail(char *err, size_t errsz, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errsz, fmt, ap);
    va_end(ap);
}

// ---- elements inside a buffer ----

static int id_len(uint8_t b) { return b >= 0x80 ? 1 : b >= 0x40 ? 2 : b >= 0x20 ? 3 : b >= 0x10 ? 4 : 0; }

// The next element of the buffer p[0, n) at *at: its id and payload. 1, 0 at the end, -1 when it does not fit.
static int el_next(const uint8_t *p, size_t n, size_t *at, uint32_t *id, const uint8_t **v, size_t *vn) {
    if (*at >= n) return 0;
    int il = id_len(p[*at]);
    if (!il || *at + (size_t)il >= n) return -1;
    uint32_t i = 0;
    for (int k = 0; k < il; k++) i = i << 8 | p[*at + (size_t)k];
    size_t a = *at + (size_t)il;
    int sl = 1;
    uint8_t m = 0x80;
    while (sl <= 8 && !(p[a] & m)) sl++, m >>= 1;
    if (sl > 8 || a + (size_t)sl > n) return -1;
    uint64_t sz = p[a] & (m - 1);
    for (int k = 1; k < sl; k++) sz = sz << 8 | p[a + (size_t)k];
    a += (size_t)sl;
    if (sz > n - a) return -1;
    *id = i;
    *v = p + a;
    *vn = (size_t)sz;
    *at = a + (size_t)sz;
    return 1;
}

static uint64_t be_uint(const uint8_t *p, size_t n) {
    uint64_t v = 0;
    for (size_t i = 0; i < n && i < 8; i++) v = v << 8 | p[i];
    return v;
}

static double be_float(const uint8_t *p, size_t n) {
    if (n == 4) {
        uint32_t u = (uint32_t)be_uint(p, 4);
        float f;
        memcpy(&f, &u, 4);
        return f;
    }
    if (n == 8) {
        uint64_t u = be_uint(p, 8);
        double d;
        memcpy(&d, &u, 8);
        return d;
    }
    return 0;
}

static int parse_sub(WebmTrack *t, const uint8_t *p, size_t n) {   // a Video or Audio element
    size_t at = 0;
    uint32_t id;
    const uint8_t *v;
    size_t vn;
    int r;
    while ((r = el_next(p, n, &at, &id, &v, &vn)) > 0) {
        if (id == ID_WIDTH) t->width = (uint32_t)be_uint(v, vn);
        else if (id == ID_HEIGHT) t->height = (uint32_t)be_uint(v, vn);
        else if (id == ID_RATE) t->rate = be_float(v, vn);
        else if (id == ID_CHANNELS) t->channels = (uint32_t)be_uint(v, vn);
        else if (id == ID_BITS) t->bits = (uint32_t)be_uint(v, vn);
    }
    return r;
}

static int parse_entry(Webm *w, const uint8_t *p, size_t n) {
    if (w->tracks >= WEBM_TRACKS_MAX) return 0;   // more tracks than carried: the rest are not listed
    WebmTrack *t = &w->track[w->tracks];
    memset(t, 0, sizeof *t);
    size_t at = 0;
    uint32_t id;
    const uint8_t *v;
    size_t vn;
    int r;
    while ((r = el_next(p, n, &at, &id, &v, &vn)) > 0) {
        switch (id) {
        case ID_NUMBER: t->number = (uint32_t)be_uint(v, vn); break;
        case ID_TYPE: t->type = (uint32_t)be_uint(v, vn); break;
        case ID_CODEC: snprintf(t->codec, sizeof t->codec, "%.*s", (int)vn, (const char *)v); break;
        case ID_DEFDUR: t->default_ns = be_uint(v, vn); break;
        case ID_DELAY: t->codec_delay_ns = be_uint(v, vn); break;
        case ID_PRIVATE:
            t->priv_size = (uint32_t)vn;
            memcpy(t->priv, v, vn < sizeof t->priv ? vn : sizeof t->priv);
            break;
        case ID_VIDEO:
        case ID_AUDIO:
            if (parse_sub(t, v, vn) < 0) return -1;
            break;
        default: break;
        }
    }
    if (r < 0) return -1;
    w->tracks++;
    return 0;
}

// ---- the file ----

// Reads n bytes at the position. 1 on success, 0 at the end of the window before the first byte, -1 when the window ends
// inside the read.
static int rd(Webm *w, void *out, size_t n) {
    if (w->pos >= w->limit) return n ? 0 : 1;
    if (w->limit - w->pos < n) return -1;
    if (fread(out, 1, n, w->f) != n) return -1;
    w->pos += n;
    return 1;
}

static int rd_skip(Webm *w, uint64_t n) {
    if (w->limit - w->pos < n) return -1;
    if (SEEK64(w->f, w->base + w->pos + n) != 0) return -1;
    w->pos += n;
    return 1;
}

static int seek_to(Webm *w, uint64_t pos) {
    if (SEEK64(w->f, w->base + pos) != 0) return -1;
    w->pos = pos;
    return 1;
}

// An element id (marker bits kept). 1, 0 at a clean end, -1 when the window ends inside it or the byte is no id.
static int rd_id(Webm *w, uint32_t *id) {
    uint8_t b;
    int r = rd(w, &b, 1);
    if (r <= 0) return r;
    int len = id_len(b);
    if (!len) return -1;
    uint32_t v = b;
    for (int i = 1; i < len; i++) {
        if (rd(w, &b, 1) != 1) return -1;
        v = v << 8 | b;
    }
    *id = v;
    return 1;
}

// An element size, or UNKNOWN. 1, or -1 when the window ends inside it or the first byte is zero.
static int rd_size(Webm *w, int64_t *size) {
    uint8_t b;
    if (rd(w, &b, 1) != 1) return -1;
    int len = 1;
    uint8_t mask = 0x80;
    while (len <= 8 && !(b & mask)) len++, mask >>= 1;
    if (len > 8) return -1;
    uint64_t v = b & (mask - 1), all = mask - 1;
    for (int i = 1; i < len; i++) {
        if (rd(w, &b, 1) != 1) return -1;
        v = v << 8 | b;
        all = all << 8 | 0xFF;
    }
    *size = v == all ? UNKNOWN : (int64_t)v;
    return 1;
}

static int grow(Webm *w, size_t n) {
    if (n <= w->cap) return 1;
    size_t cap = w->cap ? w->cap : 4096;
    while (cap < n) cap *= 2;
    uint8_t *p = realloc(w->buf, cap);
    if (!p) return 0;
    w->buf = p;
    w->cap = cap;
    return 1;
}

// The payload of the element whose header was just read into w->buf. 1, or -1 with err.
static int load(Webm *w, int64_t size, uint32_t id, char *err, size_t errsz) {
    if (size == UNKNOWN || size > (1 << 30) || !grow(w, (size_t)size + 1)) {
        fail(err, errsz, "element 0x%X at %llu has size %lld, which is not read", (unsigned)id, (unsigned long long)w->pos, (long long)size);
        return -1;
    }
    if (rd(w, w->buf, (size_t)size) != 1) {
        fail(err, errsz, "truncated: element 0x%X at %llu wants %lld bytes and the file ends at %llu", (unsigned)id,
             (unsigned long long)w->pos, (long long)size, (unsigned long long)w->limit);
        return -1;
    }
    return 1;
}

static int skip_el(Webm *w, uint32_t id, uint64_t start, int64_t size, char *err, size_t errsz) {
    if (rd_skip(w, (uint64_t)size) < 0) {
        fail(err, errsz, "truncated: element 0x%X at %llu wants %lld bytes and the file ends at %llu", (unsigned)id,
             (unsigned long long)start, (long long)size, (unsigned long long)w->limit);
        return -1;
    }
    return 0;
}

static int is_top(uint32_t id) {
    return id == ID_CLUSTER || id == ID_CUES || id == ID_TAGS || id == ID_CHAPTERS || id == ID_ATTACH || id == ID_TRACKS ||
           id == ID_INFO || id == ID_SEEKHEAD;
}

int webm_open(Webm *w, const char *path, uint64_t offset, uint64_t size, char *err, size_t errsz) {
    memset(w, 0, sizeof *w);
    w->timecode_scale = 1000000;
    w->duration_ticks = -1;
    w->segment_end = UNKNOWN;
    w->f = fopen(path, "rb");
    if (!w->f) {
        fail(err, errsz, "cannot open %s", path);
        return -1;
    }
    setvbuf(w->f, NULL, _IOFBF, 1 << 20);
    long long end = -1;
    if (fseek(w->f, 0, SEEK_END) == 0) {
        // fseek takes a long (32-bit on Windows): for a larger file the end is found with the 64-bit seek
        end = TELL64(w->f);
    }
    if (end < 0 || (uint64_t)end < offset) {
        fail(err, errsz, "the window starts at %llu, past the end of the file (%lld)", (unsigned long long)offset, end);
        goto bad;
    }
    w->base = offset;
    w->limit = size ? size : (uint64_t)end - offset;
    if (offset + w->limit > (uint64_t)end) {
        fail(err, errsz, "the window ends at %llu, past the end of the file (%lld)", (unsigned long long)(offset + w->limit), end);
        goto bad;
    }
    if (seek_to(w, 0) < 0) goto bad;

    uint32_t id;
    int64_t sz;
    if (rd_id(w, &id) != 1 || id != ID_EBML || rd_size(w, &sz) != 1 || sz == UNKNOWN) {
        fail(err, errsz, "not an EBML file: the first element is not the EBML header");
        goto bad;
    }
    if (load(w, sz, ID_EBML, err, errsz) < 0) goto bad;
    {
        size_t at = 0, vn;
        const uint8_t *v;
        uint32_t cid;
        while (el_next(w->buf, (size_t)sz, &at, &cid, &v, &vn) > 0)
            if (cid == ID_DOCTYPE) snprintf(w->doctype, sizeof w->doctype, "%.*s", (int)vn, (const char *)v);
    }
    if (rd_id(w, &id) != 1 || id != ID_SEGMENT || rd_size(w, &sz) != 1) {
        fail(err, errsz, "the EBML header is not followed by a segment");
        goto bad;
    }
    w->segment_end = sz == UNKNOWN ? UNKNOWN : (int64_t)(w->pos + (uint64_t)sz);
    for (;;) {
        uint64_t start = w->pos;
        int r = rd_id(w, &id);
        if (r == 0) break;
        if (r < 0 || rd_size(w, &sz) != 1) {
            fail(err, errsz, "truncated: a segment child at %llu does not fit in the file", (unsigned long long)start);
            goto bad;
        }
        if (id == ID_CLUSTER) {
            if (seek_to(w, start) < 0) goto bad;
            break;
        }
        if (sz == UNKNOWN) {
            fail(err, errsz, "element 0x%X before the first cluster has an unknown size", (unsigned)id);
            goto bad;
        }
        if (id == ID_INFO || id == ID_TRACKS) {
            if (load(w, sz, id, err, errsz) < 0) goto bad;
            size_t at = 0, vn;
            const uint8_t *v;
            uint32_t cid;
            int q;
            while ((q = el_next(w->buf, (size_t)sz, &at, &cid, &v, &vn)) > 0) {
                if (id == ID_INFO && cid == ID_TIMESCALE) w->timecode_scale = be_uint(v, vn);
                else if (id == ID_INFO && cid == ID_DURATION) w->duration_ticks = be_float(v, vn);
                else if (id == ID_TRACKS && cid == ID_ENTRY && parse_entry(w, v, vn) < 0) q = -1;
                if (q < 0) break;
            }
            if (q < 0) {
                fail(err, errsz, "the %s is malformed", id == ID_INFO ? "segment info" : "track list");
                goto bad;
            }
        } else if (skip_el(w, id, start, sz, err, errsz) < 0) {
            goto bad;
        }
    }
    return 0;
bad:
    webm_close(w);
    return -1;
}

static int finish(Webm *w, char *err, size_t errsz) {
    if (w->segment_end != UNKNOWN && (uint64_t)w->segment_end > w->limit) {
        fail(err, errsz, "truncated: the segment declares its end at %lld and the file ends at %llu (%lld bytes short)",
             (long long)w->segment_end, (unsigned long long)w->limit, (long long)(w->segment_end - (int64_t)w->limit));
        return -1;
    }
    return 0;
}

// The block payload p[0, n): track number, 16-bit timecode, flags. 0 or -1.
static int block_head(Webm *w, const uint8_t *p, size_t n, WebmPacket *out) {
    if (!n) return -1;
    size_t len = 1;
    uint8_t m = 0x80;
    while (len <= 8 && !(p[0] & m)) len++, m >>= 1;
    if (len > 8 || n < len + 3) return -1;
    uint64_t track = p[0] & (m - 1);
    for (size_t i = 1; i < len; i++) track = track << 8 | p[i];
    int16_t rel = (int16_t)(p[len] << 8 | p[len + 1]);
    uint8_t flags = p[len + 2];
    out->track = (uint32_t)track;
    out->ticks = w->cluster_tc + rel;
    out->key = flags >> 7;
    out->lacing = (flags >> 1) & 3;
    out->data = p + len + 3;
    out->size = n - len - 3;
    return 0;
}

static const WebmTrack *track_of(const Webm *w, uint32_t number) {
    for (int i = 0; i < w->tracks; i++)
        if (w->track[i].number == number) return &w->track[i];
    return NULL;
}

int webm_next(Webm *w, WebmPacket *p, char *err, size_t errsz) {
    for (;;) {
        uint64_t start = w->pos;
        uint32_t id;
        int64_t sz;
        if (w->in_cluster && w->cluster_end != UNKNOWN && (int64_t)w->pos >= w->cluster_end) w->in_cluster = 0;
        if (!w->in_cluster && w->segment_end != UNKNOWN && (int64_t)w->pos >= w->segment_end) return finish(w, err, errsz);
        int r = rd_id(w, &id);
        if (r == 0) {
            if (w->in_cluster && w->cluster_end != UNKNOWN) {
                fail(err, errsz, "truncated: the cluster declares its end at %lld and the file ends at %llu", (long long)w->cluster_end,
                     (unsigned long long)w->limit);
                return -1;
            }
            return finish(w, err, errsz);
        }
        if (r < 0 || rd_size(w, &sz) != 1) {
            fail(err, errsz, "truncated: an element header at %llu does not fit in the file (ends at %llu)", (unsigned long long)start,
                 (unsigned long long)w->limit);
            return -1;
        }
        if (w->in_cluster && w->cluster_end == UNKNOWN && is_top(id)) {   // an unknown-size cluster ends where the next element starts
            if (seek_to(w, start) < 0) return -1;
            w->in_cluster = 0;
            continue;
        }
        if (!w->in_cluster) {
            if (id == ID_CLUSTER) {
                w->in_cluster = 1;
                w->cluster_end = sz == UNKNOWN ? UNKNOWN : (int64_t)(w->pos + (uint64_t)sz);
                w->cluster_tc = 0;
                continue;
            }
            if (id == ID_TRACKS && !w->tracks) {
                fail(err, errsz, "the track list comes after the first cluster");
                return -1;
            }
            if (sz == UNKNOWN) {
                fail(err, errsz, "element 0x%X at %llu has an unknown size", (unsigned)id, (unsigned long long)start);
                return -1;
            }
            if (skip_el(w, id, start, sz, err, errsz) < 0) return -1;
            continue;
        }
        if (sz == UNKNOWN) {
            fail(err, errsz, "element 0x%X at %llu in a cluster has an unknown size", (unsigned)id, (unsigned long long)start);
            return -1;
        }
        if (id == ID_TC) {
            if (load(w, sz, id, err, errsz) < 0) return -1;
            w->cluster_tc = (int64_t)be_uint(w->buf, (size_t)sz);
        } else if (id == ID_SIMPLE || id == ID_GROUP) {
            if (load(w, sz, id, err, errsz) < 0) return -1;
            const uint8_t *body = w->buf;
            size_t n = (size_t)sz;
            int has_ref = 0;
            if (id == ID_GROUP) {   // the Block inside, and whether a ReferenceBlock sits beside it
                const uint8_t *blk = NULL, *v;
                size_t blk_n = 0, at = 0, vn;
                uint32_t cid;
                while (el_next(body, n, &at, &cid, &v, &vn) > 0) {
                    if (cid == ID_BLOCK) blk = v, blk_n = vn;
                    else if (cid == ID_REF) has_ref = 1;
                }
                if (!blk) {
                    fail(err, errsz, "a block group at %llu holds no block", (unsigned long long)start);
                    return -1;
                }
                body = blk;
                n = blk_n;
            }
            if (block_head(w, body, n, p) < 0) {
                fail(err, errsz, "a block at %llu has no header", (unsigned long long)start);
                return -1;
            }
            if (id == ID_GROUP) p->key = !has_ref;
            const WebmTrack *t = track_of(w, p->track);
            if (!t) {
                fail(err, errsz, "a block at %llu names track %u, which the track list does not have", (unsigned long long)start,
                     (unsigned)p->track);
                return -1;
            }
            if (t->type == 1 && p->lacing) {
                fail(err, errsz, "a block of video track %u at %llu has lacing %d, which is not read", (unsigned)p->track,
                     (unsigned long long)start, p->lacing);
                return -1;
            }
            return 1;
        } else if (skip_el(w, id, start, sz, err, errsz) < 0) {
            return -1;
        }
    }
}

void webm_close(Webm *w) {
    if (w->f) fclose(w->f);
    free(w->buf);
    w->f = NULL;
    w->buf = NULL;
}
