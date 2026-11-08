// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio renderer (oot2/audio.h): it reads each RequestUpdate block the SDK's own client builds, checks every section
// against the header and the layout of the revision (a section that does not match is a named failure with the numbers,
// never a silent misread), keeps the server-side voice state, advances the voices one frame at a time on the frame
// clock, releases a wave buffer in the frame where its samples run out, and writes the output block the client reads
// back. Each frame the mixer (mixer.c) turns the started voices into stereo frames and the sink (oot2/audio_sink.h) takes
// them; the renderer never waits for the sink.
//
// Adapted from DKB Recomp runtime/audren.c at c5d5e7c (MIT; the file's header: adapted from MacNativeGames
// games/totk/runtime/audio_renderer.c at b75be8c, MIT, Copyright (c) 2026 MacNativeGames contributors; its voice
// lifetime (play state, wave buffer release, flush) is ported from Ryujinx.Audio VoiceInfo and DataSourceHelper, MIT,
// Copyright (c) Ryujinx Team and Contributors). Changed here: the renderer is separate from the IPC session (the
// service is runtime/hos/srv_audio.c); a failure is returned with its numbers, not raised; the frame clock is a grid
// from the origin that a test can drive (frameclock.c), where the donor paced on its output device; a voice consumes
// whole samples through an integer accumulator (a worked frame count is exact: 2,400 samples at 48,000 Hz release in
// the tenth 5 ms frame, not the ninth or the eleventh); the output block echoes the input's revision tag as it came;
// the splitter section is walked strictly; counters for the renderer's position, the lateness of the grid and the
// update cadence; the parameters are checked once at open.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audren_state.h"
#include "mixer.h"
#include "oot2/kernel.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "stall.h"

#define MAX_RENDERERS 8
#define COUNTERS_EVERY 1000   // frames between two counters lines: 5 s at 5 ms

static PlatMutex g_reg_lock = PLAT_MUTEX_INIT;
static AudioRenderer *g_reg[MAX_RENDERERS];
static AudioPosition g_last_closed;
static char g_capture_path[1024];   // under g_reg_lock

void audio_set_capture_path(const char *path) {
    plat_mutex_lock(&g_reg_lock);
    snprintf(g_capture_path, sizeof g_capture_path, "%s", path ? path : "");
    plat_mutex_unlock(&g_reg_lock);
}

int audio_revision_number(uint32_t tag) { return (int)(tag >> 24) - '0'; }

const char *audio_config_read(const void *wire, uint32_t size, AudioConfig *cfg) {
    AudrenParameter p;
    if (size < sizeof p) return "the renderer parameter is shorter than 0x34 bytes";
    memcpy(&p, wire, sizeof p);
    *cfg = (AudioConfig){.sample_rate = p.sample_rate, .sample_count = p.sample_count,
                         .mix_buffer_count = p.mix_buffer_count, .submix_count = p.submix_count,
                         .voice_count = p.voice_count, .sink_count = p.sink_count, .effect_count = p.effect_count,
                         .performance_frame_count = p.performance_frame_count,
                         .splitter_count = p.splitter_count,
                         .splitter_destination_count = p.splitter_destination_count,
                         .rendering_device = p.rendering_device, .execution_mode = p.execution_mode,
                         .revision_tag = p.revision};
    if (p.sample_rate != 48000 && p.sample_rate != 32000) return "the output rate is neither 48000 nor 32000 Hz";
    if (p.sample_count <= 0 || p.sample_count > 4800) return "the sample count is outside 1 to 4800";
    if (p.voice_count <= 0 || p.voice_count > 1024 || p.sink_count <= 0 || p.sink_count > 256 || p.effect_count < 0 ||
        p.effect_count > 4096 || p.submix_count < 0 || p.submix_count >= 256 || p.mix_buffer_count <= 0 ||
        p.mix_buffer_count > 512 || p.splitter_count < 0 || p.splitter_count > 4096 ||
        p.splitter_destination_count < 0 || p.splitter_destination_count > 65536)
        return "a count of the renderer is outside what the renderer takes";
    int rev = audio_revision_number(p.revision);
    if (rev < 1 || rev > 16) return "the revision tag names no revision from 1 to 16";
    return NULL;
}

// ---- failures with their numbers
typedef struct { char *buf; size_t n; } Why;
static int fail(Why *w, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static int fail(Why *w, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    if (w->n) vsnprintf(w->buf, w->n, fmt, ap);
    va_end(ap);
    return 0;
}

// ---- registry (for audio_position)
static void registry_add(AudioRenderer *r) {
    plat_mutex_lock(&g_reg_lock);
    int i = 0;
    while (i < MAX_RENDERERS && g_reg[i]) i++;
    if (i == MAX_RENDERERS) {
        plat_mutex_unlock(&g_reg_lock);
        rt_fatal("audio: more than %d renderers open", MAX_RENDERERS);
    }
    g_reg[i] = r;
    plat_mutex_unlock(&g_reg_lock);
}

static void add_position(AudioPosition *sum, const AudioPosition *p) {
    sum->frames += p->frames;
    sum->samples += p->samples;
    sum->skipped += p->skipped;
    sum->signals += p->signals;
    sum->wave_buffers_released += p->wave_buffers_released;
    sum->updates += p->updates;
    if (p->late_max_ns > sum->late_max_ns) sum->late_max_ns = p->late_max_ns;
    if (p->update_gap_max_ns > sum->update_gap_max_ns) sum->update_gap_max_ns = p->update_gap_max_ns;
    if (p->signal_to_update_max_ns > sum->signal_to_update_max_ns) sum->signal_to_update_max_ns = p->signal_to_update_max_ns;
    sum->running |= p->running;
    sum->dropped_frames += p->dropped_frames;
    sum->sink_written += p->sink_written;
    sum->sink_played += p->sink_played;
    sum->sink_queued += p->sink_queued;
    sum->underruns += p->underruns;
    sum->overruns += p->overruns;
}

void audio_renderer_position(const AudioRenderer *cr, AudioPosition *out) {
    AudioRenderer *r = (AudioRenderer *)cr;   // the lock is taken, nothing else changes
    plat_mutex_lock(&r->lock);
    out->frames = r->elapsed_frames;
    out->samples = r->elapsed_frames * (uint64_t)r->cfg.sample_count;
    out->skipped = r->skipped_frames;
    out->signals = r->signals;
    out->wave_buffers_released = r->released;
    out->updates = r->updates;
    out->late_max_ns = r->late_max;
    out->update_gap_max_ns = r->update_gap_max;
    out->signal_to_update_max_ns = r->signal_to_update_max;
    out->running = r->running;
    out->dropped_frames = r->dropped_frames;
    plat_mutex_unlock(&r->lock);
    AudioSinkStats ss;
    audio_sink_stats(r->sink, &ss);
    out->sink_written = ss.written;
    out->sink_played = ss.played;
    out->sink_queued = ss.queued;
    out->underruns = ss.underruns;
    out->overruns = ss.overruns;
}

void audio_position(AudioPosition *out) {
    memset(out, 0, sizeof *out);
    AudioRenderer *open[MAX_RENDERERS];
    int n = 0;
    plat_mutex_lock(&g_reg_lock);
    for (int i = 0; i < MAX_RENDERERS; i++)
        if (g_reg[i]) open[n++] = g_reg[i];
    if (!n) *out = g_last_closed;
    plat_mutex_unlock(&g_reg_lock);
    out->stalls = stall_count();
    out->event_wait_max_ns = stall_event_wait_max_ns();
    for (int i = 0; i < n; i++) {
        AudioPosition p;
        audio_renderer_position(open[i], &p);
        add_position(out, &p);
    }
}

// ---- voices
static void release_buffer(AudioRenderer *r, ServerVoice *v) {
    v->consumed++;
    r->released++;
}

static void voice_reset(AudioRenderer *r, unsigned id) {
    ServerVoice *v = &r->voices[id];
    mixer_voice_reset(r->mixer, id);
    memset(v->valid, 0, sizeof v->valid);
    memset(v->pending, 0, sizeof v->pending);
    v->state = SV_STOPPED;
    v->flush = v->flush_from = 0;
    v->index = 0;
    v->offset = 0;
    v->acc = 0;
    v->played = 0;
    v->consumed = 0;
    v->loops_done = 0;
}

static void update_voice(AudioRenderer *r, unsigned slot, const AudrenVoiceIn *in) {
    ServerVoice *v = &r->voices[slot];
    if (in->is_new || !in->in_use) voice_reset(r, slot);
    v->in = *in;
    v->use_float_biquads = 0;
    if (!in->in_use) return;
    // The play state as the DSP keeps it (Ryujinx VoiceInfo.UpdatePlayState): Stop on a voice that is not stopped
    // becomes Stopping, which the next frame turns into Stopped after giving back every queued buffer.
    if (in->play_state == PLAY_STARTED) {
        if (v->state != SV_STARTED) v->start_ns = stall_now_ns();
        v->state = SV_STARTED;
    }
    else if (in->play_state == PLAY_PAUSED) v->state = SV_PAUSED;
    else if (v->state != SV_STOPPED) v->state = SV_STOPPING;
    for (unsigned i = 0; i < 4; i++) {   // a new buffer waits until a frame finds the voice playing
        const AudrenWaveBuffer *w = &in->wave_buffers[i];
        // The sent flag decides which buffers are new: a client that sets it on the buffers it submits would show real
        // buffers on the second line below and none on the first. The first few of each kind are named in the log.
        unsigned *seen = !w->sent ? &r->diag_new : &r->diag_sent;
        if (w->address && w->size && *seen < 4) {
            (*seen)++;
            rt_log("audio: wave buffer, sent flag %u (%s): voice %u slot %u, %llu bytes at %#llx, samples %d to %d, loop %u "
                   "(count %d), end of stream %u; voice format %u, rate %u, state %u, pitch %.3f, volume %.3f",
                   w->sent, w->sent ? "kept as it is" : "queued", slot, i, (unsigned long long)w->size,
                   (unsigned long long)w->address, w->start, w->end, w->loop, w->loop_count, w->end_of_stream,
                   in->sample_format, in->sample_rate, in->play_state, in->pitch, in->volume);
        }
        if (!w->sent) {
            v->buffers[i] = *w;
            v->pending[i] = 1;
            v->sub_ns[i] = stall_now_ns();
            v->sub_frame[i] = r->elapsed_frames;
            v->sub_thread[i] = rt_kernel_current();
        }
    }
    v->flush += in->flush_wave_buffer_count;
    v->flush_from = in->wave_buffer_index;
    if (in->clear_voice_drop) v->drops = 0;
}

// No samples (no memory, no size or an empty range): the DSP counts the buffer as played when it reaches it, so a
// stream's empty last buffer does not hold up the buffers queued behind it.
int buffer_empty(const AudrenWaveBuffer *b) { return !b->address || !b->size || b->end <= b->start; }

static void skip_empty(AudioRenderer *r, ServerVoice *v) {
    while (v->valid[v->index] && buffer_empty(&v->buffers[v->index])) {
        const AudrenWaveBuffer *b = &v->buffers[v->index];
        if (b->loop && b->loop_count < 0) break;   // an endless empty loop starves the voice, as on the DSP
        v->valid[v->index] = 0;
        release_buffer(r, v);
        v->offset = 0;
        v->loops_done = 0;
        v->index = (v->index + 1) & 3;
        if (b->end_of_stream) {
            v->played = 0;
            break;
        }
    }
}

// Each frame, before rendering (Ryujinx VoiceInfo.UpdateParametersForCommandGeneration): flush, stop, start.
static void voices_prepare(AudioRenderer *r) {
    for (int k = 0; k < r->cfg.voice_count; k++) {
        ServerVoice *v = &r->voices[k];
        if (!v->in.in_use) continue;
        if (v->flush) {   // FlushWaveBuffers: from the client's current buffer, the queued ones count as played
            unsigned i = v->flush_from & 3;
            for (uint32_t n = v->flush; n; n--, i = (i + 1) & 3) {
                v->pending[i] = 0;
                if (v->valid[i]) {
                    v->valid[i] = 0;
                    release_buffer(r, v);
                    v->index = (v->index + 1) & 3;
                    v->offset = 0;
                    v->loops_done = 0;
                }
            }
            v->flush = 0;
        }
        if (v->state == SV_STOPPING) {   // every queued buffer counts as played; unsent ones are dropped
            for (int i = 0; i < 4; i++) {
                if (v->valid[i]) {
                    release_buffer(r, v);
                    v->index = (v->index + 1) & 3;
                }
                v->valid[i] = 0;
                v->pending[i] = 0;
            }
            v->offset = 0;
            v->played = 0;
            v->acc = 0;
            v->loops_done = 0;
            v->state = SV_STOPPED;
            mixer_voice_reset(r->mixer, (unsigned)k);
            continue;
        }
        if (v->state == SV_STARTED) {
            for (int i = 0; i < 4; i++)
                if (v->pending[i]) {
                    v->pending[i] = 0;
                    v->valid[i] = 1;
                    if ((unsigned)i == v->index) {
                        v->offset = 0;
                        v->loops_done = 0;
                    }
                }
            skip_empty(r, v);
        }
    }
}

// One frame of a started voice: it takes floor(owed) of its own samples, the remainder carries. The owed amount is
// sample_count x voice_rate x pitch / engine_rate, kept in whole 1 / engine_rate units so that a pitch of 1 is exact.
double voice_pitch(const ServerVoice *v) {
    const float p = v->in.pitch;
    return p > 0 && isfinite(p) ? (p < AUDIO_PITCH_MAX ? p : AUDIO_PITCH_MAX) : 1.0;
}

uint64_t voice_owed(const AudioRenderer *r, const ServerVoice *v) {
    return (uint64_t)((double)((uint64_t)r->cfg.sample_count * v->in.sample_rate) * voice_pitch(v) + 0.5);
}

static void advance_voice(AudioRenderer *r, ServerVoice *v) {
    if (!v->in.in_use || v->state != SV_STARTED || !v->in.sample_rate) return;
    const uint64_t engine = (uint64_t)r->cfg.sample_rate;
    v->acc += voice_owed(r, v);
    uint64_t samples = v->acc / engine;
    v->acc %= engine;
    while (samples && v->valid[v->index]) {
        AudrenWaveBuffer *b = &v->buffers[v->index];
        // skip_empty keeps one buffer with no samples at the head: an endless loop over nothing. It starves the voice (no
        // sample is consumed, the buffer is never released), and the frame goes on; a step of 0 would loop here for ever.
        if (buffer_empty(b)) break;
        uint64_t length = (uint64_t)((int64_t)b->end - b->start), left = length - v->offset;
        uint64_t step = samples < left ? samples : left;
        v->offset += step;
        v->played += step;
        samples -= step;
        if (v->offset < length) break;
        if (b->loop && (b->loop_count < 0 || v->loops_done < b->loop_count)) {
            v->loops_done++;
            v->offset = b->loop_start > b->start && b->loop_start < b->end ? (uint64_t)(b->loop_start - b->start) : 0;
            continue;
        }
        v->valid[v->index] = 0;
        release_buffer(r, v);
        v->offset = 0;
        v->loops_done = 0;
        v->index = (v->index + 1) & 3;
        if (b->end_of_stream) {
            v->played = 0;
            break;
        }
        skip_empty(r, v);
    }
}

// ---- the frame
static void render_frame(void *ctx, uint64_t tick, uint64_t skipped) {
    (void)tick;
    AudioRenderer *r = ctx;
    plat_mutex_lock(&r->lock);
    voices_prepare(r);
    float *lr = r->frame_buf;
    mixer_render(r, lr);
    if (!audio_sink_write(r->sink, lr, (uint32_t)r->cfg.sample_count)) r->dropped_frames++;
    for (int i = 0; i < r->cfg.voice_count; i++) advance_voice(r, &r->voices[i]);
    __atomic_store_n(&r->elapsed_frames, r->elapsed_frames + 1, __ATOMIC_RELAXED);
    r->skipped_frames += skipped;
    __atomic_store_n(&r->last_frame_wd, stall_now_ns(), __ATOMIC_RELAXED);
    plat_mutex_unlock(&r->lock);
    rt_kernel_event_signal(r->event);
    plat_mutex_lock(&r->lock);
    r->last_signal_ns = frameclock_now_ns();
    r->signals++;
    r->signal_pending = 1;
    uint64_t frames = r->elapsed_frames, released = r->released, updates = r->updates, gap = r->update_gap_max,
             answer = r->signal_to_update_max, signals = r->signals, skips = r->skipped_frames,
             dropped = r->dropped_frames;
    unsigned in_use = 0, started = 0, holding = 0, paused = 0;
    MixerStats ms = {0};
    if (frames % COUNTERS_EVERY == 0) ms = mixer_stats(r->mixer, 1);
    if (frames % COUNTERS_EVERY == 0)
        for (int i = 0; i < r->cfg.voice_count; i++) {
            const ServerVoice *v = &r->voices[i];
            if (!v->in.in_use) continue;
            in_use++;
            started += v->state == SV_STARTED;
            paused += v->state == SV_PAUSED;
            holding += v->valid[0] | v->valid[1] | v->valid[2] | v->valid[3];
        }
    plat_mutex_unlock(&r->lock);
    if (frames % COUNTERS_EVERY == 0) {   // the counters line: a run that ends by its time limit never reaches the close
        FrameClockStats cs = frameclock_stats(r->clock);
        AudioSinkStats ss;
        audio_sink_stats(r->sink, &ss);
        uint64_t stalls = stall_count(), wait_max = stall_event_wait_max_ns();
        rt_log("audio: counters: %llu frames (%llu skipped), %llu frame events, %llu wave buffers released, %llu updates; "
               "sink: written %llu samples (%.2f s), played %llu, queued %llu, underruns %llu, overruns %llu, dropped frames %llu; "
               "mixed %llu voice frames, largest voice sample %.4f, largest output sample %.4f; "
               "stalls %llu, longest event wait seen %.1f ms; latest frame %.3f ms after its due time, longest gap between updates %.3f ms, longest time from a frame's "
               "signal to the next update %.3f ms; voices: %u in use, %u started, %u paused, %u holding a queued buffer",
               (unsigned long long)frames, (unsigned long long)skips, (unsigned long long)signals,
               (unsigned long long)released, (unsigned long long)updates, (unsigned long long)ss.written,
               (double)ss.written / (double)r->cfg.sample_rate, (unsigned long long)ss.played, (unsigned long long)ss.queued,
               (unsigned long long)ss.underruns, (unsigned long long)ss.overruns, (unsigned long long)dropped,
               (unsigned long long)ms.voice_frames, (double)ms.input_peak, (double)ms.output_peak,
               (unsigned long long)stalls, (double)wait_max / 1e6, (double)cs.late_max_ns / 1e6, (double)gap / 1e6, (double)answer / 1e6, in_use, started, paused, holding);
    }
}

// ---- RequestUpdate
typedef struct { const uint8_t *p, *end; } Reader;

static const uint8_t *take(Reader *in, uint32_t n, const char *what, Why *w) {
    if ((size_t)(in->end - in->p) < n) {
        fail(w, "input section %s (%u bytes) overruns the update, %zu bytes left", what, n, (size_t)(in->end - in->p));
        return NULL;
    }
    const uint8_t *p = in->p;
    in->p += n;
    return p;
}

// The splitter section: a header {magic SNDH, info count, destination count, ...} of 0x20 bytes, the infos {magic SNDI,
// id, ..., destination count} with their destination ids and 12 reserved bytes, then the destinations (magic SNDD) of a
// revision-dependent stride; the section may be padded to 16 bytes. Every byte must be accounted for.
static int splitters_walk(const uint8_t *p, uint32_t n, const AudioConfig *cfg, int rev, Why *w) {
    if (!n) return 1;
    if (n < 0x20) return fail(w, "splitter section of %u bytes is shorter than its 0x20-byte header", n);
    uint32_t hdr[3];
    memcpy(hdr, p, 12);
    if (hdr[0] != 0x48444e53u) return fail(w, "splitter header magic %#x is not SNDH", hdr[0]);
    uint32_t ni = hdr[1], nd = hdr[2], at = 0x20;
    for (uint32_t i = 0; i < ni; i++) {
        uint32_t h[4];
        if (at + 16 > n) return fail(w, "splitter info %u at %#x overruns the %u-byte section", i, at, n);
        memcpy(h, p + at, 16);
        at += 16;
        if (h[0] != 0x49444e53u || h[1] >= (uint32_t)cfg->splitter_count || h[3] > (uint32_t)cfg->splitter_destination_count ||
            at + 4u * h[3] > n)
            return fail(w, "splitter info %u at %#x: words %08x %08x %08x %08x (section %u bytes, infos %u, "
                           "destinations %u, limits %d and %d)", i, at - 16, h[0], h[1], h[2], h[3], n, ni, nd,
                        cfg->splitter_count, cfg->splitter_destination_count);
        at += 4u * h[3] + 12;   // 12 reserved bytes follow the ids
        if (at > n) return fail(w, "splitter info %u: the padding after its ids overruns the %u-byte section", i, n);
    }
    uint32_t stride = rev >= 15 ? 0xa8 : rev >= 12 ? 0x90 : 0x70;
    for (uint32_t i = 0; i < nd; i++) {
        if (at > n || stride > n - at)
            return fail(w, "splitter destination %u of %u at %#x (stride %#x) overruns the %u-byte section", i, nd, at,
                        stride, n);
        uint32_t magic, id;
        memcpy(&magic, p + at, 4);
        memcpy(&id, p + at + 4, 4);
        if (magic != 0x44444e53u || id >= (uint32_t)cfg->splitter_destination_count)
            return fail(w, "splitter destination %u at %#x: magic %#x, id %u (limit %d, stride %#x, REV%d)", i, at, magic,
                        id, cfg->splitter_destination_count, stride, rev);
        at += stride;
    }
    if (at <= n && n - at < 16) at = n;   // the section is padded to 16 bytes, and the padding is not cleared
    if (at != n)
        return fail(w, "splitter section is %u bytes and its %u infos and %u destinations (stride %#x, REV%d) take %u",
                    n, ni, nd, stride, rev, at);
    return 1;
}

// The mix section: every mix in full, or the dirty-only form: 0x20 bytes of {magic 0xcafebabe, count} and the mixes sent.
static int mixes_read(AudioRenderer *r, const uint8_t *p, uint32_t n, Why *w) {
    const int count = r->mix_count;
    if (n == (uint32_t)count * sizeof(AudrenMixIn)) {
        memcpy(r->mixes, p, n);
        return 1;
    }
    if (n >= 0x20 && (n - 0x20) % sizeof(AudrenMixIn) == 0) {
        uint32_t magic, cnt;
        memcpy(&magic, p, 4);
        memcpy(&cnt, p + 4, 4);
        if (magic != 0xcafebabeu || (uint64_t)cnt * sizeof(AudrenMixIn) != n - 0x20)
            return fail(w, "dirty mix header: magic %#x, %u mixes for a %u-byte section", magic, cnt, n);
        for (uint32_t i = 0; i < cnt; i++) {
            AudrenMixIn m;
            memcpy(&m, p + 0x20 + (size_t)i * sizeof m, sizeof m);
            if (m.mix_id < 0 || m.mix_id >= count) return fail(w, "dirty mix %u has the id %d, the renderer has %d mixes", i, m.mix_id, count);
            r->mixes[m.mix_id] = m;
        }
        return 1;
    }
    return fail(w, "mix section is %u bytes for %d mixes (%zu each in full, or 0x20 and a multiple of it)", n, count,
                sizeof(AudrenMixIn));
}

// One voice record of the update, in the layout of the revision, as the version-1 struct (the filters of revision 15
// and later are read as floats into fb and mark enable in the struct): 1, or 0 with the reason.
static int read_voice(const uint8_t *src, int rev, int index, int voice_count, AudrenVoiceIn *v, AudrenBiquadFloat fb[2], Why *w) {
    memset(fb, 0, 2 * sizeof *fb);
    if (rev >= 15) {
        memset(v, 0, sizeof *v);
        memcpy(v, src, AUDREN_VOICE_FILTERS);
        memcpy(fb, src + AUDREN_VOICE_FILTERS, 2 * sizeof *fb);
        memcpy((uint8_t *)v + AUDREN_VOICE_TAIL_V1, src + AUDREN_VOICE_TAIL_V2, audren_voice_in_stride(rev) - AUDREN_VOICE_TAIL_V2);
        for (unsigned f = 0; f < 2; f++) {
            v->biquads[f].enable = fb[f].enable;
            if (!v->in_use || !fb[f].enable) continue;
            for (unsigned c = 0; c < 3; c++)
                if (!isfinite(fb[f].numerator[c])) return fail(w, "voice record %d: filter %u has a numerator that is not finite", index, f);
            for (unsigned c = 0; c < 2; c++)
                if (!isfinite(fb[f].denominator[c])) return fail(w, "voice record %d: filter %u has a denominator that is not finite", index, f);
        }
    } else {
        memcpy(v, src, sizeof *v);
    }
    if (v->in_use && v->id >= (uint32_t)voice_count)
        return fail(w, "voice record %d carries the id %u, the renderer has %d voices", index, v->id, voice_count);
    if (v->in_use && v->wave_buffer_index >= 4)
        return fail(w, "voice %d has the wave buffer index %u (of 4)", index, v->wave_buffer_index);
    return 1;
}

static void log_first_update(AudioRenderer *r, const AudrenHeader *h, int rev, uint64_t in_size, uint64_t out_size) {
    const uint32_t vc = (uint32_t)r->cfg.voice_count, ec = (uint32_t)r->cfg.effect_count,
                   sc = (uint32_t)r->cfg.sink_count, np = (uint32_t)r->cfg.effect_count + 4u * vc;
    char tag[5] = {(char)h->revision, (char)(h->revision >> 8), (char)(h->revision >> 16), (char)(h->revision >> 24), 0};
    rt_log("audio: update 1: REV%d (tag %s), behavior %u pools %u voices %u voice_resources %u effects %u mixes %u sinks %u "
           "performance %u splitters %u render %u total %u, in %llu out %llu; per-section size / count, remainder: "
           "pools %u r%u, resources %u r%u (of 112), voices %u r%u, effects %u r%u, sinks %u r%u",
           rev, tag, h->behavior, h->memory_pools, h->voices, h->voice_resources, h->effects, h->mixes, h->sinks,
           h->performance, h->splitters, h->render_info, h->total, (unsigned long long)in_size,
           (unsigned long long)out_size, np ? h->memory_pools / np : 0, np ? h->memory_pools % np : 0,
           h->voice_resources / 112, h->voice_resources % 112, vc ? h->voices / vc : 0, vc ? h->voices % vc : 0,
           ec ? h->effects / ec : 0, ec ? h->effects % ec : 0, sc ? h->sinks / sc : 0, sc ? h->sinks % sc : 0);
}

int audio_renderer_update(AudioRenderer *r, const uint8_t *input, uint64_t in_size, uint8_t *output, uint64_t out_size,
                          uint32_t *out_used, char *why, size_t why_size) {
    Why w = {why, why_size};
    if (why_size) why[0] = 0;
    if (in_size < sizeof(AudrenHeader)) return fail(&w, "the update is %llu bytes, shorter than its 0x40-byte header", (unsigned long long)in_size);
    AudrenHeader h;
    memcpy(&h, input, sizeof h);
    const int rev = audio_revision_number(h.revision);
    if (rev < 1 || rev > 16) return fail(&w, "the update's revision tag %#x names no revision from 1 to 16", h.revision);
    if (h.total > in_size) return fail(&w, "the update's total %u is more than the %llu bytes sent", h.total, (unsigned long long)in_size);
    if (h.total < sizeof h) return fail(&w, "the update's total %u is less than its 0x40-byte header", h.total);
    const int vc = r->cfg.voice_count, ec = r->cfg.effect_count, sc = r->cfg.sink_count;
    const uint32_t npools = (uint32_t)ec + 4u * (uint32_t)vc;
    plat_mutex_lock(&r->lock);
    int ok = 0;
    do {
        if (!r->logged) {
            r->logged = 1;
            log_first_update(r, &h, rev, in_size, out_size);
            if (h.revision != r->cfg.revision_tag)
                rt_log("audio: the update's revision tag %#x is not the one the renderer was opened with (%#x)",
                       h.revision, r->cfg.revision_tag);
        }
        r->revision = rev;
        Reader in = {input + sizeof h, input + h.total};
        const uint32_t vstride = audren_voice_in_stride(rev);
        if (h.behavior != AUDREN_BEHAVIOR_IN) { fail(&w, "behavior section is %u bytes, REV%d takes %u", h.behavior, rev, AUDREN_BEHAVIOR_IN); break; }
        const uint8_t *behavior = take(&in, h.behavior, "behavior", &w);
        if (!behavior) break;
        memcpy(&r->behavior_flags, behavior + 8, 8);
        if (h.memory_pools != npools * sizeof(AudrenPoolIn)) {
            fail(&w, "memory pool section is %u bytes, %u pools of %zu take %u", h.memory_pools, npools, sizeof(AudrenPoolIn), npools * (uint32_t)sizeof(AudrenPoolIn));
            break;
        }
        const AudrenPoolIn *pools = (const AudrenPoolIn *)take(&in, h.memory_pools, "memory pools", &w);
        if (!pools) break;
        if (h.voice_resources % sizeof(AudrenChannelIn) || h.voice_resources / sizeof(AudrenChannelIn) > (uint32_t)vc * 6) {
            fail(&w, "voice channel section is %u bytes, not a multiple of %zu up to %d entries", h.voice_resources, sizeof(AudrenChannelIn), vc * 6);
            break;
        }
        const AudrenChannelIn *chans = (const AudrenChannelIn *)take(&in, h.voice_resources, "voice channels", &w);
        if (!chans) break;
        if (h.voices != (uint32_t)vc * vstride) {
            fail(&w, "voice section is %u bytes: %d voices of %u (REV%d) take %u; the section holds %u per voice, remainder %u",
                 h.voices, vc, vstride, rev, (uint32_t)vc * vstride, h.voices / (uint32_t)vc, h.voices % (uint32_t)vc);
            break;
        }
        const uint8_t *voices = take(&in, h.voices, "voices", &w);
        if (!voices) break;
        if (h.effects != (uint32_t)ec * sizeof(AudrenEffectIn)) {
            fail(&w, "effect section is %u bytes: %d effects of %zu take %u", h.effects, ec, sizeof(AudrenEffectIn), (uint32_t)ec * (uint32_t)sizeof(AudrenEffectIn));
            break;
        }
        const AudrenEffectIn *effects = (const AudrenEffectIn *)take(&in, h.effects, "effects", &w);
        if (!effects) break;
        const uint8_t *splitters = take(&in, h.splitters, "splitters", &w);
        if (!splitters || !splitters_walk(splitters, h.splitters, &r->cfg, rev, &w)) break;
        const uint8_t *mixes = take(&in, h.mixes, "mixes", &w);
        if (!mixes) break;
        if (h.sinks != (uint32_t)sc * sizeof(AudrenSinkIn)) {
            fail(&w, "sink section is %u bytes: %d sinks of %zu take %u", h.sinks, sc, sizeof(AudrenSinkIn), (uint32_t)sc * (uint32_t)sizeof(AudrenSinkIn));
            break;
        }
        const AudrenSinkIn *sinks = (const AudrenSinkIn *)take(&in, h.sinks, "sinks", &w);
        if (!sinks || !take(&in, h.performance, "performance", &w) || !take(&in, h.render_info, "render info", &w)) break;
        if (in.p != in.end) { fail(&w, "%zu bytes of the update are not accounted for by its sections", (size_t)(in.end - in.p)); break; }

        // the voice and channel records are checked before any is applied; a bad mix section or an output block that is too
        // small is found after, and ends the run with a named stop, so the tables it leaves are never played
        int bad = 0;
        for (int i = 0; i < vc && !bad; i++) {
            AudrenVoiceIn v;
            AudrenBiquadFloat fb[2];
            bad = !read_voice(voices + (size_t)i * vstride, rev, i, vc, &v, fb, &w);
        }
        for (uint32_t i = 0; i < h.voice_resources / sizeof(AudrenChannelIn) && !bad; i++)
            if (chans[i].id >= (uint32_t)vc * 6) bad = !fail(&w, "voice channel resource %u carries the id %u (of %d)", i, chans[i].id, vc * 6);
        if (bad) break;
        // Voice state follows the record's slot; an unused record may have a zero or unspecified id.
        for (int i = 0; i < vc; i++) {
            AudrenVoiceIn v;
            AudrenBiquadFloat fb[2];
            read_voice(voices + (size_t)i * vstride, rev, i, vc, &v, fb, &w);
            update_voice(r, (unsigned)i, &v);
            ServerVoice *sv = &r->voices[i];
            sv->use_float_biquads = rev >= 15;
            if (rev >= 15) memcpy(sv->float_biquads, fb, sizeof fb);
        }
        for (uint32_t i = 0; i < h.voice_resources / sizeof(AudrenChannelIn); i++) r->channels[chans[i].id] = chans[i];
        if (!mixes_read(r, mixes, h.mixes, &w)) break;
        mixer_set_splitters(r->mixer, splitters, h.splitters, rev);
        memcpy(r->effects, effects, h.effects);
        memcpy(r->sinks, sinks, h.sinks);

        // ---- the output block
        const uint32_t eo = audren_effect_out_stride(rev);
        AudrenHeader o;
        memset(&o, 0, sizeof o);
        o.revision = h.revision;   // the tag as the client wrote it
        o.memory_pools = npools * (uint32_t)sizeof(AudrenPoolOut);
        o.voices = (uint32_t)vc * (uint32_t)sizeof(AudrenVoiceOut);
        o.effects = (uint32_t)ec * eo;
        o.sinks = (uint32_t)sc * (uint32_t)sizeof(AudrenSinkOut);
        o.performance = AUDREN_PERFORMANCE_OUT;
        o.behavior = AUDREN_BEHAVIOR_OUT;
        o.render_info = AUDREN_RENDER_INFO_OUT;
        o.total = (uint32_t)sizeof o + o.memory_pools + o.voices + o.effects + o.sinks + o.performance + o.behavior + o.render_info;
        if (o.total > out_size) {
            fail(&w, "the output block needs %u bytes (REV%d, effect status %u bytes each), the client gave %llu", o.total, rev, eo, (unsigned long long)out_size);
            break;
        }
        memset(output, 0, (size_t)out_size);
        uint8_t *p = output;
        memcpy(p, &o, sizeof o);
        p += sizeof o;
        for (uint32_t i = 0; i < npools; i++) {
            uint32_t s = pools[i].state;
            if (s == POOL_REQUEST_ATTACH) s = POOL_ATTACHED;
            else if (s == POOL_REQUEST_DETACH) s = POOL_DETACHED;
            if (r->pools[i].address != pools[i].address || r->pools[i].size != pools[i].size || r->pools[i].state != s) r->pool_gen++;
            r->pools[i] = pools[i];
            r->pools[i].state = s;
            AudrenPoolOut po = {s, {0}};
            memcpy(p, &po, sizeof po);
            p += sizeof po;
        }
        for (int i = 0; i < vc; i++) {
            AudrenVoiceOut vo = {r->voices[i].played, r->voices[i].consumed, r->voices[i].drops};
            memcpy(p, &vo, sizeof vo);
            p += sizeof vo;
        }
        for (int i = 0; i < ec; i++, p += eo) p[0] = effects[i].enabled ? 3 : 4;   // EffectStatus Enabled 3, Disabled 4
        p += (size_t)sc * sizeof(AudrenSinkOut) + AUDREN_PERFORMANCE_OUT + AUDREN_BEHAVIOR_OUT;
        memcpy(p, &r->elapsed_frames, 8);   // render info: frames rendered
        *out_used = o.total;

        r->updates++;
        uint64_t now = frameclock_now_ns();
        if (r->running) {
            if (r->last_update_ns && now - r->last_update_ns > r->update_gap_max) r->update_gap_max = now - r->last_update_ns;
            if (r->signal_pending && now - r->last_signal_ns > r->signal_to_update_max) r->signal_to_update_max = now - r->last_signal_ns;
        }
        r->signal_pending = 0;
        r->last_update_ns = now;
        __atomic_store_n(&r->last_update_wd, stall_now_ns(), __ATOMIC_RELAXED);
        __atomic_store_n(&r->last_update_thread, rt_kernel_current(), __ATOMIC_RELAXED);
        ok = 1;
    } while (0);
    plat_mutex_unlock(&r->lock);
    return ok;
}

// ---- lifetime
AudioRenderer *audio_renderer_open(const AudioConfig *cfg, uint32_t event, uint32_t readable) {
    AudioRenderer *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("audio: renderer allocation");
    r->cfg = *cfg;
    r->event = event;
    r->readable = readable;
    r->mix_count = cfg->submix_count + 1;
    const size_t vc = (size_t)cfg->voice_count;
    r->voices = calloc(vc, sizeof *r->voices);
    r->pools = calloc((size_t)cfg->effect_count + 4u * vc, sizeof *r->pools);
    r->mixes = calloc((size_t)r->mix_count, sizeof *r->mixes);
    r->sinks = calloc((size_t)cfg->sink_count, sizeof *r->sinks);
    r->effects = calloc((size_t)(cfg->effect_count ? cfg->effect_count : 1), sizeof *r->effects);
    r->channels = calloc(vc * 6, sizeof *r->channels);
    if (!r->voices || !r->pools || !r->mixes || !r->sinks || !r->effects || !r->channels) rt_fatal("audio: renderer tables");
    r->frame_buf = calloc((size_t)cfg->sample_count * 2, sizeof(float));
    if (!r->frame_buf) rt_fatal("audio: renderer frame buffer");
    r->ctl = (PlatMutex)PLAT_MUTEX_INIT;
    r->lock = (PlatMutex)PLAT_MUTEX_INIT;
    r->mixer = mixer_open(r);
    int64_t kind = rt_setting(RT_SET_audio_sink);
    plat_mutex_lock(&g_reg_lock);
    char path[1024];
    snprintf(path, sizeof path, "%s", g_capture_path);
    plat_mutex_unlock(&g_reg_lock);
    r->sink = audio_sink_open((AudioSinkKind)kind, cfg->sample_rate, path[0] ? path : NULL);
    if (!r->sink) rt_fatal("audio: the audio_sink setting asks for the platform stream and it cannot open: %s", audio_sink_why());
    rt_log("audio: sink: %s %d Hz stereo float, %s", audio_sink_kind_name(r->sink), cfg->sample_rate,
           audio_sink_device(r->sink));
    registry_add(r);
    stall_register(r);
    return r;
}

static void tick_late(AudioRenderer *r) {
    if (!r->clock) return;
    FrameClockStats s = frameclock_stats(r->clock);
    plat_mutex_lock(&r->lock);
    if (s.late_max_ns > r->late_max) r->late_max = s.late_max_ns;
    plat_mutex_unlock(&r->lock);
}

void audio_renderer_start(AudioRenderer *r) {
    plat_mutex_lock(&r->ctl);
    plat_mutex_lock(&r->lock);
    int already = r->running;
    __atomic_store_n(&r->running, 1, __ATOMIC_RELAXED);
    r->last_update_ns = 0;
    r->signal_pending = 0;
    plat_mutex_unlock(&r->lock);
    if (!already) {
        audio_sink_pause(r->sink, 0);
        r->clock = frameclock_start((uint32_t)r->cfg.sample_count, (uint32_t)r->cfg.sample_rate, render_frame, r);
        if (!r->clock) rt_fatal("audio: cannot start the renderer's frame clock");
    }
    plat_mutex_unlock(&r->ctl);
}

static void stop_locked(AudioRenderer *r) {   // r->ctl held
    plat_mutex_lock(&r->lock);
    int was = r->running;
    __atomic_store_n(&r->running, 0, __ATOMIC_RELAXED);
    plat_mutex_unlock(&r->lock);
    if (!was || !r->clock) return;
    int token = rt_kernel_block_begin();   // a guest thread gives its core up while the clock thread ends
    tick_late(r);
    frameclock_stop(r->clock);
    r->clock = NULL;
    rt_kernel_block_end(token);
    audio_sink_pause(r->sink, 1);
}

void audio_renderer_stop(AudioRenderer *r) {
    plat_mutex_lock(&r->ctl);
    stop_locked(r);
    plat_mutex_unlock(&r->ctl);
}

int audio_renderer_running(const AudioRenderer *cr) {
    AudioRenderer *r = (AudioRenderer *)cr;
    plat_mutex_lock(&r->lock);
    int v = r->running;
    plat_mutex_unlock(&r->lock);
    return v;
}

const AudioConfig *audio_renderer_config(const AudioRenderer *r) { return &r->cfg; }

void audio_renderer_close(AudioRenderer *r) {
    plat_mutex_lock(&r->ctl);
    stop_locked(r);
    plat_mutex_unlock(&r->ctl);
    AudioPosition p;
    audio_renderer_position(r, &p);
    stall_unregister(r);
    plat_mutex_lock(&g_reg_lock);
    for (int i = 0; i < MAX_RENDERERS; i++)
        if (g_reg[i] == r) g_reg[i] = NULL;
    g_last_closed = p;
    plat_mutex_unlock(&g_reg_lock);
    rt_kernel_event_close(r->event);
    rt_log("audio: renderer closed after %llu frames (%llu skipped), %llu wave buffers released, %llu updates; latest frame "
           "%.3f ms after its due time, longest gap between updates %.3f ms, longest time from a frame's signal to the next "
           "update %.3f ms", (unsigned long long)p.frames, (unsigned long long)p.skipped,
           (unsigned long long)p.wave_buffers_released, (unsigned long long)p.updates, (double)p.late_max_ns / 1e6,
           (double)p.update_gap_max_ns / 1e6, (double)p.signal_to_update_max_ns / 1e6);
    audio_sink_close(r->sink);
    mixer_close(r->mixer);
    free(r->frame_buf);
    free(r->voices);
    free(r->pools);
    free(r->mixes);
    free(r->sinks);
    free(r->effects);
    free(r->channels);
    free(r);
}

AudioSink *audio_renderer_sink(const AudioRenderer *r) { return r->sink; }

AudioRenderer *audio_renderer_nth(int n) {
    AudioRenderer *found = NULL;
    plat_mutex_lock(&g_reg_lock);
    for (int i = 0; i < MAX_RENDERERS; i++)
        if (g_reg[i] && n-- == 0) {
            found = g_reg[i];
            break;
        }
    plat_mutex_unlock(&g_reg_lock);
    return found;
}
