// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay's trace (OOT2_REPLAY_TRACE, OOT2_REPLAY_TRACE_FILE): a picture that lacks elements with no stop in the log is
// a draw, a pass or a texture that the replay made nothing of, and this file says which. Off by default. It reads and
// counts and nothing else: no Vulkan call, no state of the replay changed, no query beyond the occlusion queries the replay
// already makes once the game counts samples passed, nothing the game could read back.
//
// The spec (OOT2_REPLAY_TRACE), items separated by commas:
//   on        a summary line per present
//   every=N   a summary line every N presents instead of every one (implies on)
//   A-B       the rows of every draw, dispatch, pass and query of the presents A to B (implies on); a single A is A-A
//   rb=WxH    readback statistics (R rows) of the colour target of each pass of that extent, in the presents of the windows
//   rb=WxH/F  and, after the slash, of that VkFormat number only (up to four rb= items; none: every pass)
//   rbpass=A  only the passes numbered A (or A-B) in their submission (the P rows' numbers)
//   rbatt=I   the colour attachment read (default 0; "all" for every one)
//   rbz=N     the slice read: a layer of an array target, a depth slice of a volume target (default 0)
//   rbn=N     at most N readbacks a present (default 8)
//   rbthr=V   the threshold of the "above" count (default 60000)
//   A readback records a copy of the target into the frame's scratch memory after its pass; the completion thread prints the
//   minimum, maximum and mean of the finite values of each channel, the count of NaN and infinity, and the count above V.
//   It costs the copy and the scan when on, never changes what is drawn or what the game reads, and is a test of one static
//   int when off.
// A submission belongs to the present that follows it (the one the game makes next), so the presents are the numbers the
// capture's shots carry. The file is OOT2_REPLAY_TRACE_FILE (created anew), else the run log, each line after "[rt] trace".
// How to read it is in docs/architecture.md (the replay's trace).
//
// Cost: a call site tests the static int vkrt_on, which is 0 unless the variable is set. The counters of the summary are
// atomic increments (the completion thread counts the queries), the rows are formatted only inside a window.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/nvn_program.h"
#include "oot2/replay_count.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/texsync.h"
#include "oot2/vk_format.h"
#include "oot2/vk_raw.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "oot2/vk_texture.h"
#include "plat.h"
#include "vk_replay_int.h"

// snprintf for a line built piece by piece: returns the characters it stored (at most size - 1), never the length it
// wanted, so that `at += sn(buf + at, sizeof buf - at, ...)` cannot pass the end of the buffer and size - at cannot wrap.
static size_t sn(char *buf, size_t size, const char *fmt, ...) {
    if (!size) return 0;
    va_list ap;
    va_start(ap, fmt);
    int r = vsnprintf(buf, size, fmt, ap);
    va_end(ap);
    if (r < 0) return 0;
    return (size_t)r < size ? (size_t)r : size - 1;
}

#define WINDOWS 8
#define SUB_PASSES 4096
#define RAW_KEYS 8
#define ROW_MAX 1536

int vkrt_on;

static FILE *g_f;
static PlatMutex g_lock = PLAT_MUTEX_INIT;   // the file, and the summary's baseline
static uint64_t g_every = 1;
static struct {
    uint64_t a, b;
} g_win[WINDOWS];
static int g_nwin;
static char g_spec[384];
static uint64_t g_present;       // atomic: the newest present made
static int g_writer_added;

// ---------- the counters of the summary lines
#define COUNTERS(X)                                                                                                    \
    X(SUBS, "subs") X(PASSES, "passes") X(PASSES_EMPTY, "passes_with_no_draw") X(DRAWS, "draws") X(MESH, "mesh_draws")   \
    X(DISPATCH, "dispatches") X(OFF_ENABLE, "off_enable") X(OFF_COND, "off_cond") X(CLEARS_OFF, "clears_off")          \
    X(SKIP_PIPE, "skipped_pipeline") X(ZERO_WORK, "zero_work") X(INDIRECT, "indirect") X(RAW, "raw_access")            \
    X(UNB_C1, "unbound_c1") X(UNB_SB, "unbound_storage") X(UNB_UB, "unbound_uniform") X(TEX, "tex_slots")              \
    X(TEX_NOENTRY, "tex_no_entry") X(TEX_UNREG, "tex_unregistered") X(TEX_ASTC, "tex_astc") X(TEX_ZR, "tex_zero_regions") \
    X(QUERIES, "queries") X(Q_ZERO, "queries_zero_samples") X(Q_ZERO_DRAWS, "queries_zero_samples_with_draws") \
    X(RB, "readbacks") X(RB_SKIP, "readbacks_skipped")
enum {
#define C_ENUM(id, name) C_##id,
    COUNTERS(C_ENUM)
#undef C_ENUM
    C_N
};
static const char *const g_cname[C_N] = {
#define C_NAME(id, name) name,
    COUNTERS(C_NAME)
#undef C_NAME
};
static uint64_t g_c[C_N];        // atomic
static uint64_t g_prev[C_N];     // g_lock: the totals at the last summary
static uint64_t g_prev_present;
static uint64_t g_miss_subs;     // g_lock: raw-access miss submissions seen by the last poll

static void cnt(int id, uint64_t n) {
    if (n) __atomic_fetch_add(&g_c[id], n, __ATOMIC_RELAXED);
}

// ---------- output
static void put_line(const char *line) {
    plat_mutex_lock(&g_lock);
    if (g_f) fputs(line, g_f), fputc('\n', g_f);
    else rt_log("trace %s", line);
    plat_mutex_unlock(&g_lock);
}

__attribute__((format(printf, 1, 2))) static void out(const char *fmt, ...) {
    char buf[ROW_MAX];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof buf) buf[sizeof buf - 4] = '.', buf[sizeof buf - 3] = '.', buf[sizeof buf - 2] = '.', buf[sizeof buf - 1] = 0;
    put_line(buf);
}

static void flush_file(void) {
    plat_mutex_lock(&g_lock);
    if (g_f) fflush(g_f);
    plat_mutex_unlock(&g_lock);
}

static void header(void);
static void totals(void);
// The totals once, at a stop (the stop path) or when the process returns normally (atexit): whichever comes first.
static void end_once(void) {
    static int done;
    if (__atomic_exchange_n(&done, 1, __ATOMIC_ACQ_REL)) return;
    totals();
    flush_file();
}
static void stop_writer(int code, const char *line) {
    (void)code, (void)line;
    end_once();
}

// ---------- the replay's side: the submission, its passes and the row being built
static struct {
    uint64_t value, present;
    int rows;                          // this submission's rows are wanted
    uint32_t n_draw, passes;
    VkrStats at_begin;
    uint32_t pass_draws[SUB_PASSES];   // draws in each pass, filled when it ends
    char raw_key[RAW_KEYS][26];
    uint32_t nraw;
    char kind;                         // the kind the op set before its draw_prepare
} T;

static struct {
    int open;
    char kind;
    const char *status;
    uint32_t n;
    const char *k0, *k1, *k2;
    int raw;
    uint32_t unb_c1, unb_sb, unb_ub;
    char tex[8][72];
    uint32_t ntex;
    uint64_t zr_mark;
    int has_state;
    VkrtState st;
    char writes[8][48];
    uint32_t nwrites;
} Row;

typedef struct PassTarget {
    int set;
    uint64_t storage;
    uint32_t level, layer, layers, w, h;
    VkFormat format;
} PassTarget;
static struct {
    uint32_t w, h, ncolor;
    VkFormat color[8], depth, stencil;
    PassTarget tg[9];
} Pass;

static int in_window(uint64_t present) {
    for (int i = 0; i < g_nwin; i++)
        if (present >= g_win[i].a && present <= g_win[i].b) return 1;
    return 0;
}

void vkrt_submit_begin(uint64_t value) {
    header();
    T.value = value;
    T.present = __atomic_load_n(&g_present, __ATOMIC_RELAXED) + 1;
    T.rows = in_window(T.present);
    T.n_draw = T.passes = T.nraw = 0;
    vkr_stats(&T.at_begin);
    Row.open = 0;
    cnt(C_SUBS, 1);
}

// The raw-access stage keys of the last submissions that had any (a miss report names its submission by value, and the keys
// that submission's draws and dispatches carried are the stages that can have missed).
#define KEYS_KEPT 16
static struct {
    uint64_t value;
    char keys[RAW_KEYS * 28 + 8];
} g_rk[KEYS_KEPT];
static unsigned g_rk_n;   // g_lock

static void keys_note(void) {
    if (!T.nraw) return;
    plat_mutex_lock(&g_lock);
    typeof(g_rk[0]) *k = &g_rk[g_rk_n++ % KEYS_KEPT];
    k->value = T.value, k->keys[0] = 0;
    for (uint32_t i = 0; i < T.nraw; i++) {
        size_t n = strlen(k->keys);
        snprintf(k->keys + n, sizeof k->keys - n, "%s%s", i ? "," : "", T.raw_key[i]);
    }
    plat_mutex_unlock(&g_lock);
}

void vkrt_submit_end(void) {
    keys_note();
    if (!T.rows) return;
    VkrStats s;
    vkr_stats(&s);
    const VkrStats *b = &T.at_begin;
    char raw[RAW_KEYS * 28 + 8] = "";
    for (uint32_t i = 0; i < T.nraw; i++) {
        size_t n = strlen(raw);
        snprintf(raw + n, sizeof raw - n, "%s%s", i ? "," : "", T.raw_key[i]);
    }
    // draw_ops against what was made: the replay's own accounting identity, per submission (lost 0 by contract)
    out("S\t%llu\t%llu\tpasses=%u\tdraw_ops=%llu\tdraws=%llu\tdispatches=%llu\toff_enable=%llu\toff_cond=%llu\t"
        "skipped_pipeline=%llu\tclears_off=%llu\tlost=%llu\traw_stage_keys=%s",
        (unsigned long long)T.present, (unsigned long long)T.value, T.passes,
        (unsigned long long)(s.draw_ops - b->draw_ops), (unsigned long long)(s.draws - b->draws),
        (unsigned long long)(s.dispatches - b->dispatches), (unsigned long long)(s.render_off_enable - b->render_off_enable),
        (unsigned long long)(s.render_off_cond - b->render_off_cond), (unsigned long long)(s.skipped - b->skipped),
        (unsigned long long)(s.clears_off - b->clears_off), (unsigned long long)s.lost, raw);
}

void vkrt_pass_begin(uint32_t w, uint32_t h, uint32_t ncolor, const VkFormat *color, VkFormat depth, VkFormat stencil) {
    T.passes++;
    cnt(C_PASSES, 1);
    Pass.w = w, Pass.h = h, Pass.ncolor = ncolor > 8 ? 8 : ncolor, Pass.depth = depth, Pass.stencil = stencil;
    memset(Pass.tg, 0, sizeof Pass.tg);
    for (uint32_t i = 0; i < Pass.ncolor; i++) Pass.color[i] = color[i];
}

void vkrt_pass_end(uint32_t draws) {
    if (T.passes && T.passes <= SUB_PASSES) T.pass_draws[T.passes - 1] = draws;
    if (!draws) cnt(C_PASSES_EMPTY, 1);
    if (!T.rows) return;
    char fm[96] = "";
    for (uint32_t i = 0; i < Pass.ncolor; i++) {
        size_t n = strlen(fm);
        snprintf(fm + n, sizeof fm - n, "%s%d", i ? "," : "", (int)Pass.color[i]);
    }
    char tg[9 * 100] = "";
    for (int i = 0; i < 9; i++) {
        if (!Pass.tg[i].set) continue;
        const PassTarget *t = &Pass.tg[i];
        size_t n = strlen(tg);
        snprintf(tg + n, sizeof tg - n, "\t%s%d=storage:%#llx:fmt:%d:%ux%u:level:%u:layer:%u:layers:%u", i == 8 ? "d" : "t",
                 i == 8 ? 0 : i, (unsigned long long)t->storage, (int)t->format, t->w, t->h, t->level, t->layer, t->layers);
    }
    out("P\t%llu\t%llu\t%u\t%ux%u\tcolor=%s\tdepth=%d\tstencil=%d\tdraws=%u%s", (unsigned long long)T.present,
        (unsigned long long)T.value, T.passes, Pass.w, Pass.h, fm[0] ? fm : "none", (int)Pass.depth, (int)Pass.stencil, draws,
        tg);
}

void vkrt_pass_target(int slot, uint64_t storage, uint32_t level, uint32_t layer, uint32_t layers, uint32_t w, uint32_t h,
                      VkFormat format) {
    if (slot < 0 || slot > 8) return;
    Pass.tg[slot] = (PassTarget){1, storage, level, layer, layers, w, h, format};
}

// ---------- one row per draw and dispatch
static void row_open(char kind, const char *status, uint32_t n) {
    memset(&Row, 0, sizeof Row);
    Row.open = 1, Row.kind = kind, Row.status = status, Row.n = n;
}

// The columns of the state a draw drew with (VkrtState) and of what a draw or dispatch binds for writing, as key=value fields.
static int state_cols(char *buf, size_t size) {
    int n = 0;
    if (Row.has_state) {
        const VkrtState *s = &Row.st;
        n += sn(buf + n, size - (size_t)n, "\tncolor=%u\tblend_en=%#x\twmask=%#x", s->ncolor, s->blend_en, s->chmask);
        for (int i = 0; i < 8; i++)
            if (s->blend[i].set)
                n += sn(buf + n, size - (size_t)n, "\tblend%d=%u/%u/%u/%u/%u/%u", i, s->blend[i].src_rgb, s->blend[i].dst_rgb,
                              s->blend[i].src_a, s->blend[i].dst_a, s->blend[i].eq_rgb, s->blend[i].eq_a);
        if (s->has_ds) n += sn(buf + n, size - (size_t)n, "\tds=test:%d,write:%d,func:%d,stencil:%d", s->dt, s->dw, s->df, s->st);
        else n += sn(buf + n, size - (size_t)n, "\tds=unset");
        if (s->point_size >= 0) n += sn(buf + n, size - (size_t)n, "\tpoint_size=%g", (double)s->point_size);
        else n += sn(buf + n, size - (size_t)n, "\tpoint_size=unset");
        n += sn(buf + n, size - (size_t)n, "\tline_width=%g", (double)s->line_width);
    }
    if (Row.nwrites) {
        n += sn(buf + n, size - (size_t)n, "\twrites=");
        for (uint32_t i = 0; i < Row.nwrites; i++) n += sn(buf + n, size - (size_t)n, "%s%s", i ? "," : "", Row.writes[i]);
    }
    return n;
}

void vkrt_state(const VkrtState *s) {
    if (!Row.open) return;
    Row.st = *s, Row.has_state = 1;
}

void vkrt_write(char what, uint64_t a, uint64_t b, uint64_t size) {
    if (!Row.open || Row.nwrites >= sizeof Row.writes / sizeof Row.writes[0]) return;
    if (what == 'b') snprintf(Row.writes[Row.nwrites++], sizeof Row.writes[0], "buf:%#llx+%#llx", (unsigned long long)a,
                              (unsigned long long)size);
    else snprintf(Row.writes[Row.nwrites++], sizeof Row.writes[0], "img:id%llu@%#llx+%#llx", (unsigned long long)a,
                  (unsigned long long)b, (unsigned long long)size);
}

static void row_emit(const char *extra) {
    if (T.rows) {
        char buf[ROW_MAX];
        int n = (int)sn(buf, sizeof buf, "D\t%llu\t%llu\t%u\t%c\t%s", (unsigned long long)T.present,
                         (unsigned long long)T.value, Row.n, Row.kind, Row.status);
        if (Row.k0) n += sn(buf + n, sizeof buf - (size_t)n, "\tkey0=%s", Row.k0);
        if (Row.k1) n += sn(buf + n, sizeof buf - (size_t)n, "\tkey1=%s", Row.k1);
        if (Row.k2) n += sn(buf + n, sizeof buf - (size_t)n, "\tkey2=%s", Row.k2);
        if (extra) n += sn(buf + n, sizeof buf - (size_t)n, "\t%s", extra);
        n += sn(buf + n, sizeof buf - (size_t)n, "\traw=%d", Row.raw);
        n += state_cols(buf + n, sizeof buf - (size_t)n);
        if (Row.unb_c1 || Row.unb_sb || Row.unb_ub)
            n += sn(buf + n, sizeof buf - (size_t)n, "\tunbound_as_zeros=c1:%u,storage:%u,uniform:%u", Row.unb_c1,
                          Row.unb_sb, Row.unb_ub);
        for (uint32_t i = 0; i < Row.ntex && n < ROW_MAX - 80; i++) n += sn(buf + n, sizeof buf - (size_t)n, "\ttex=%s", Row.tex[i]);
        put_line(buf);
    }
    Row.open = 0;
}

void vkrt_kind(char kind) { T.kind = kind; }

void vkrt_begin_kind(char kind, int off) {
    uint32_t n = ++T.n_draw;
    if (off) {
        row_open(kind, off == 1 ? "off-enable" : "off-cond", n);
        cnt(off == 1 ? C_OFF_ENABLE : C_OFF_COND, 1);
        row_emit(NULL);
        return;
    }
    row_open(kind, "drawn", n);
}

void vkrt_begin(int off) {
    vkrt_begin_kind(T.kind ? T.kind : 'd', off);
    T.kind = 0;
}

void vkrt_clear_off(void) { cnt(C_CLEARS_OFF, 1); }

void vkrt_skip(void) {
    if (!Row.open) return;
    Row.status = "skipped-pipeline-not-ready";
    cnt(C_SKIP_PIPE, 1);
    row_emit(NULL);
}

static void note_raw(const char *k) {
    for (uint32_t i = 0; i < T.nraw; i++)
        if (!strcmp(T.raw_key[i], k)) return;
    if (T.nraw < RAW_KEYS) snprintf(T.raw_key[T.nraw++], sizeof T.raw_key[0], "%s", k);
}

void vkrt_stages(const char *k0, const char *k1, const char *k2, const struct nvn_stage *const *raw, int nraw) {
    if (!Row.open) return;
    Row.k0 = k0, Row.k1 = k1, Row.k2 = k2;
    Row.raw = nraw > 0;
    if (nraw > 0) cnt(C_RAW, 1);
    for (int i = 0; i < nraw; i++) note_raw(raw[i]->key);
}

void vkrt_unbound(int what) {
    if (what == 'c') cnt(C_UNB_C1, 1), Row.unb_c1++;
    else if (what == 's') cnt(C_UNB_SB, 1), Row.unb_sb++;
    else cnt(C_UNB_UB, 1), Row.unb_ub++;
}

static uint64_t zero_regions_now(void) {
    VktStats v;
    vkt_stats(&v);
    TexSyncStats x;
    texsync_stats(&x);
    return v.zero_regions + x.zero_regions;
}

void vkrt_tex_mark(void) { Row.zr_mark = zero_regions_now(); }

void vkrt_tex(int st, uint32_t c2_slot, uint32_t id, uint32_t sampler, uint32_t slot, const VkReplayOps *ops) {
    static const char *const names[8] = {"vs", "ps", "?", "?", "?", "cs", "ts", "ms"};
    uint32_t format = 0;
    int astc = 0, entry = -1;
    if (ops && ops->trace_tex) entry = ops->trace_tex(id, &format, &astc);
    uint64_t zr = zero_regions_now() - Row.zr_mark;
    Row.zr_mark += zr;
    cnt(C_TEX, 1);
    if (entry == 0) cnt(C_TEX_NOENTRY, 1);
    if (slot == UINT32_MAX) cnt(C_TEX_UNREG, 1);
    if (astc) cnt(C_TEX_ASTC, 1);
    cnt(C_TEX_ZR, zr);
    if (!T.rows || !Row.open || Row.ntex >= sizeof Row.tex / sizeof Row.tex[0]) return;
    char slotname[16];
    if (slot == UINT32_MAX) snprintf(slotname, sizeof slotname, "none");
    else snprintf(slotname, sizeof slotname, "%u", slot);
    snprintf(Row.tex[Row.ntex++], sizeof Row.tex[0], "%s%u:id=%u:smp=%u:entry=%s:fmt=%#x:astc=%d:slot=%s:zero_regions=%llu",
             names[st & 7], c2_slot, id, sampler, entry < 0 ? "?" : entry ? "1" : "0", format, astc, slotname,
             (unsigned long long)zr);
}

void vkrt_end(const VkrtEnd *e) {
    if (!Row.open) return;
    char extra[256];
    int n = 0;
    extra[0] = 0;
    char k = Row.kind;
    int graphics = k == 'd' || k == 'i' || k == 'I', mesh = k == 'm' || k == 'M', disp = k == 'c' || k == 'C';
    int indirect = k == 'I' || k == 'M' || k == 'C';
    cnt(graphics ? C_DRAWS : mesh ? C_MESH : C_DISPATCH, 1);
    if (indirect) cnt(C_INDIRECT, 1);
    int zero = disp ? (!indirect && (!e->groups[0] || !e->groups[1] || !e->groups[2])) : (!indirect && e->count == 0);
    if (zero) cnt(C_ZERO_WORK, 1);
    if (disp) {
        if (indirect) n += sn(extra + n, sizeof extra - (size_t)n, "indirect=gpu");
        else n += sn(extra + n, sizeof extra - (size_t)n, "groups=%u,%u,%u", e->groups[0], e->groups[1], e->groups[2]);
    } else {
        if (indirect && e->have_ind)
            n += sn(extra + n, sizeof extra - (size_t)n, "indirect=read\tcount=%u\tinst=%u", e->ind_count, e->ind_inst);
        else if (indirect)
            n += sn(extra + n, sizeof extra - (size_t)n, "indirect=gpu");
        else if (mesh)
            n += sn(extra + n, sizeof extra - (size_t)n, "count=%u\tfirst=%d", e->count, e->first);
        else
            n += sn(extra + n, sizeof extra - (size_t)n, "count=%u\tfirst=%d\tinst=%d>%u", e->count, e->first,
                          e->inst_rec, e->inst_run);
        if (graphics && e->prim >= 0) n += sn(extra + n, sizeof extra - (size_t)n, "\tprim=%d", e->prim);
    }
    row_emit(extra);
}

// ---------- samples passed, from the completion thread
typedef struct VkrtQ {
    uint64_t value;
    uint32_t pass, draws;
    uint8_t rows, present_known;
    uint64_t present;
} VkrtQ;

static VkrtQ g_q[VKR_QUERY_POOLS * VKR_QUERIES];

void vkrt_query(uint32_t q) {
    if (q >= sizeof g_q / sizeof g_q[0]) return;
    g_q[q] = (VkrtQ){T.value, T.passes, 0, (uint8_t)T.rows, 1, T.present};
}

void *vkrt_queries_take(uint32_t nq) {
    if (!nq) return NULL;
    if (nq > sizeof g_q / sizeof g_q[0]) nq = sizeof g_q / sizeof g_q[0];
    VkrtQ *c = malloc(nq * sizeof *c);
    if (!c) return NULL;
    memcpy(c, g_q, nq * sizeof *c);
    for (uint32_t i = 0; i < nq; i++)
        if (c[i].pass && c[i].pass <= SUB_PASSES) c[i].draws = T.pass_draws[c[i].pass - 1];
    return c;
}

void vkrt_queries_done(void *qs, uint32_t nq, const uint64_t *samples) {
    const VkrtQ *q = qs;
    for (uint32_t i = 0; i < nq && i < sizeof g_q / sizeof g_q[0]; i++) {
        cnt(C_QUERIES, 1);
        if (!samples[i]) {
            cnt(C_Q_ZERO, 1);
            if (q[i].draws) cnt(C_Q_ZERO_DRAWS, 1);
        }
        if (q[i].rows)
            out("Q\t%llu\t%llu\tpass=%u\tquery=%u\tsamples=%llu\tpass_draws=%u", (unsigned long long)q[i].present,
                (unsigned long long)q[i].value, q[i].pass, i, (unsigned long long)samples[i], q[i].draws);
    }
    free(qs);
}

// ---------- readback statistics (OOT2_REPLAY_TRACE rb=...)
// Where a white sky or a missing haze is born along the chain of passes: after a chosen pass the replay records a copy of a
// colour target into the frame's scratch memory (vk_replay.c trace_readback), and the completion thread, once the frame is
// done and before the scratch memory is taken back, prints what the texels hold. The scan is a pure function of the bytes.
int vkrt_rb_on;

#define RB_SIZES 4
static struct {
    struct {
        uint32_t w, h;
        int fmt;           // VkFormat, or -1 for any
    } size[RB_SIZES];
    int nsize;
    uint32_t pass_a, pass_b, z, max;
    int att;               // the colour attachment, -1: every one
    double thr;
    uint64_t present;      // the present the count belongs to
    uint32_t done;         // readbacks made for it
} Rb = {.pass_a = 1, .pass_b = UINT32_MAX, .max = 8, .thr = 60000.0};

int vkrt_rb_wants(VkrtRbWant *w) {
    if (!vkrt_rb_on || !T.rows) return 0;
    if (T.passes < Rb.pass_a || T.passes > Rb.pass_b) return 0;
    if (Rb.att >= 0 && w->att != (uint32_t)Rb.att) return 0;
    if (Rb.nsize) {
        int hit = 0;
        for (int i = 0; i < Rb.nsize && !hit; i++)
            hit = Rb.size[i].w == w->width && Rb.size[i].h == w->height && (Rb.size[i].fmt < 0 || Rb.size[i].fmt == (int)w->format);
        if (!hit) return 0;
    }
    if (Rb.present != T.present) Rb.present = T.present, Rb.done = 0;
    if (Rb.done >= Rb.max) return 0;
    Rb.done++;
    w->z = Rb.z;
    return 1;
}

static void rb_head(char *buf, size_t size, const VkrtRbWant *w, uint32_t level, uint32_t layer) {
    snprintf(buf, size, "R\t%llu\t%llu\tpass=%u\tatt=%u\t%ux%u\tfmt=%d\tz=%u\tlevel=%u\tlayer=%u", (unsigned long long)T.present,
             (unsigned long long)T.value, T.passes, w->att, w->width, w->height, (int)w->format, w->z, level, layer);
}

void vkrt_rb_skipped(const VkrtRbWant *w, const char *why) {
    char buf[ROW_MAX];
    rb_head(buf, sizeof buf, w, 0, 0);
    size_t n = strlen(buf);
    snprintf(buf + n, sizeof buf - n, "\tskipped=%s", why);
    put_line(buf);
    cnt(C_RB_SKIP, 1);
}

typedef struct RbJob {
    char head[256];
    VkFormat fmt;
    uint64_t texels;
    const void *data;
    double thr;
} RbJob;

static void rb_done(void *arg, uint64_t value) {
    (void)value;
    RbJob *j = arg;
    VkrtScan sc;
    char buf[ROW_MAX];
    int n = (int)sn(buf, sizeof buf, "%s\ttexels=%llu", j->head, (unsigned long long)j->texels);
    if (!vkrt_scan(j->data, j->fmt, j->texels, j->thr, &sc)) {
        snprintf(buf + n, sizeof buf - (size_t)n, "\tskipped=no_decoder_for_format");
        cnt(C_RB_SKIP, 1);
    } else {
        for (int c = 0; c < sc.nch; c++) {
            const VkrtChan *k = &sc.ch[c];
            if (k->finite)
                n += sn(buf + n, sizeof buf - (size_t)n, "\tc%d.min=%.9g\tc%d.max=%.9g\tc%d.mean=%.9g", c, k->min, c, k->max, c,
                              k->sum / (double)k->finite);
            else
                n += sn(buf + n, sizeof buf - (size_t)n, "\tc%d.min=na\tc%d.max=na\tc%d.mean=na", c, c, c);
            n += sn(buf + n, sizeof buf - (size_t)n, "\tc%d.nonfinite=%llu\tc%d.above=%llu", c,
                          (unsigned long long)k->nonfinite, c, (unsigned long long)k->above);
        }
        cnt(C_RB, 1);
    }
    put_line(buf);
    free(j);
}

void vkrt_rb_queued(const VkrtRbWant *w, uint32_t level, uint32_t layer, const void *data, uint64_t bytes) {
    RbJob *j = calloc(1, sizeof *j);
    if (!j) return;
    rb_head(j->head, sizeof j->head, w, level, layer);
    j->fmt = w->format, j->data = data, j->thr = Rb.thr;
    uint32_t bpt = vkrt_texel_bytes(w->format);
    j->texels = bpt ? bytes / bpt : 0;
    vkrec_effect(rb_done, j);
}

// ---- the decoders
static double half_to_double(uint16_t h, int *finite) {
    int e = (h >> 10) & 31, m = h & 1023;
    double v;
    *finite = 1;
    if (e == 31) {
        *finite = 0;
        return 0;
    }
    v = e ? ldexp(1.0 + m / 1024.0, e - 15) : ldexp(m / 1024.0, -14);
    return (h & 0x8000) ? -v : v;
}

// An unsigned small float of `mb` mantissa bits and a 5-bit exponent (B10G11R11's channels).
static double ufloat(uint32_t bits, int mb, int *finite) {
    uint32_t m = bits & ((1u << mb) - 1u), e = (bits >> mb) & 31u;
    *finite = 1;
    if (e == 31) {
        *finite = 0;
        return 0;
    }
    return e ? ldexp(1.0 + (double)m / (double)(1u << mb), (int)e - 15) : ldexp((double)m / (double)(1u << mb), -14);
}

typedef enum { K_NONE, K_F32, K_F16, K_UNORM8, K_UNORM16, K_UINT32, K_UINT16, K_B10G11R11, K_E5B9G9R9, K_A2B10G10R10 } Kind;
static Kind kind_of(VkFormat f, int *nch, uint32_t *bytes) {
    switch (f) {
    case VK_FORMAT_R32G32B32A32_SFLOAT: *nch = 4, *bytes = 16; return K_F32;
    case VK_FORMAT_R32G32B32_SFLOAT: *nch = 3, *bytes = 12; return K_F32;
    case VK_FORMAT_R32G32_SFLOAT: *nch = 2, *bytes = 8; return K_F32;
    case VK_FORMAT_R32_SFLOAT: *nch = 1, *bytes = 4; return K_F32;
    case VK_FORMAT_R16G16B16A16_SFLOAT: *nch = 4, *bytes = 8; return K_F16;
    case VK_FORMAT_R16G16_SFLOAT: *nch = 2, *bytes = 4; return K_F16;
    case VK_FORMAT_R16_SFLOAT: *nch = 1, *bytes = 2; return K_F16;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_SRGB:
        *nch = 4, *bytes = 4; return K_UNORM8;
    case VK_FORMAT_R8G8_UNORM: *nch = 2, *bytes = 2; return K_UNORM8;
    case VK_FORMAT_R8_UNORM: *nch = 1, *bytes = 1; return K_UNORM8;
    case VK_FORMAT_R16G16B16A16_UNORM: *nch = 4, *bytes = 8; return K_UNORM16;
    case VK_FORMAT_R16G16_UNORM: *nch = 2, *bytes = 4; return K_UNORM16;
    case VK_FORMAT_R16_UNORM: *nch = 1, *bytes = 2; return K_UNORM16;
    case VK_FORMAT_R32G32B32A32_UINT: *nch = 4, *bytes = 16; return K_UINT32;
    case VK_FORMAT_R32G32_UINT: *nch = 2, *bytes = 8; return K_UINT32;
    case VK_FORMAT_R32_UINT: *nch = 1, *bytes = 4; return K_UINT32;
    case VK_FORMAT_R16G16B16A16_UINT: *nch = 4, *bytes = 8; return K_UINT16;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: *nch = 3, *bytes = 4; return K_B10G11R11;
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: *nch = 3, *bytes = 4; return K_E5B9G9R9;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: *nch = 4, *bytes = 4; return K_A2B10G10R10;
    default: return K_NONE;
    }
}

uint32_t vkrt_texel_bytes(VkFormat f) {
    int nch;
    uint32_t bytes = 0;
    return kind_of(f, &nch, &bytes) == K_NONE ? 0 : bytes;
}

static void chan_add(VkrtChan *c, double v, int finite, double thr) {
    if (!finite || v != v) {
        c->nonfinite++;
        return;
    }
    if (!c->finite || v < c->min) c->min = v;
    if (!c->finite || v > c->max) c->max = v;
    c->finite++;
    c->sum += v;
    if (v > thr) c->above++;
}

int vkrt_scan(const void *data, VkFormat f, uint64_t count, double thr, VkrtScan *out) {
    int nch = 0;
    uint32_t bytes = 0;
    Kind k = kind_of(f, &nch, &bytes);
    memset(out, 0, sizeof *out);
    if (k == K_NONE) return 0;
    out->nch = nch;
    const uint8_t *p = data;
    for (uint64_t i = 0; i < count; i++, p += bytes) {
        for (int c = 0; c < nch; c++) {
            double v = 0;
            int fin = 1;
            switch (k) {
            case K_F32: {
                float x;
                memcpy(&x, p + 4 * c, 4);
                v = (double)x, fin = x == x && x - x == 0;
                break;
            }
            case K_F16: {
                uint16_t h;
                memcpy(&h, p + 2 * c, 2);
                v = half_to_double(h, &fin);
                break;
            }
            case K_UNORM8: v = p[c] / 255.0; break;
            case K_UNORM16: {
                uint16_t h;
                memcpy(&h, p + 2 * c, 2);
                v = h / 65535.0;
                break;
            }
            case K_UINT32: {
                uint32_t h;
                memcpy(&h, p + 4 * c, 4);
                v = (double)h;
                break;
            }
            case K_UINT16: {
                uint16_t h;
                memcpy(&h, p + 2 * c, 2);
                v = (double)h;
                break;
            }
            case K_B10G11R11: {
                uint32_t w;
                memcpy(&w, p, 4);
                v = c == 0 ? ufloat(w & 0x7FFu, 6, &fin) : c == 1 ? ufloat((w >> 11) & 0x7FFu, 6, &fin) : ufloat(w >> 22, 5, &fin);
                break;
            }
            case K_E5B9G9R9: {
                uint32_t w;
                memcpy(&w, p, 4);
                int e = (int)(w >> 27) - 15 - 9;
                uint32_t m = c == 0 ? w & 0x1FFu : c == 1 ? (w >> 9) & 0x1FFu : (w >> 18) & 0x1FFu;
                v = ldexp((double)m, e);
                break;
            }
            case K_A2B10G10R10: {
                uint32_t w;
                memcpy(&w, p, 4);
                v = c == 0 ? (w & 1023u) / 1023.0 : c == 1 ? ((w >> 10) & 1023u) / 1023.0
                    : c == 2 ? ((w >> 20) & 1023u) / 1023.0 : (w >> 30) / 3.0;
                break;
            }
            default: break;
            }
            chan_add(&out->ch[c], v, fin, thr);
        }
    }
    return 1;
}

// ---------- the summary, per present, and the raw-access misses the device reported since the last one
// The raw-access misses the device reported since the last call (r: gpu_vk_raw_stats, or a test's): one M line for the newest
// submission that had any, with the kind and reason of the entries the log kept.
void vkrt_raw_misses(const GpuVkRawStats *rs, uint64_t seq) {
    const GpuVkRawStats r = *rs;
    if (r.miss_submissions == g_miss_subs) return;
    uint64_t skipped = r.miss_submissions - g_miss_subs - 1;
    g_miss_subs = r.miss_submissions;
    uint64_t kinds[4] = {0}, reasons[4] = {0};
    for (uint64_t i = 0; i < r.last_misses && i < RAW_MISS_ENTRIES; i++) {
        kinds[r.last_miss[i].kind & 3]++;
        reasons[r.last_miss[i].reason & 3]++;
    }
    // the log keeps at most RAW_MISS_ENTRIES entries a submission; kinds are of the kept ones, misses_total of every one
    out("M\t%llu\tsubmission=%llu\tmisses_total=%llu\tmiss_submissions=%llu\tearlier_miss_submissions_not_shown=%llu\t"
        "kept=%llu\tload=%llu\tstore=%llu\tatomic=%llu\toutside_span=%llu\tno_chunk=%llu\tno_page=%llu",
        (unsigned long long)seq, (unsigned long long)r.last_miss_value, (unsigned long long)r.misses,
        (unsigned long long)r.miss_submissions, (unsigned long long)skipped, (unsigned long long)r.last_misses,
        (unsigned long long)kinds[RAW_MISS_KIND_LOAD], (unsigned long long)kinds[RAW_MISS_KIND_STORE],
        (unsigned long long)kinds[RAW_MISS_KIND_ATOMIC], (unsigned long long)reasons[RAW_MISS_REASON_OUTSIDE_SPAN],
        (unsigned long long)reasons[RAW_MISS_REASON_NO_CHUNK], (unsigned long long)reasons[RAW_MISS_REASON_NO_PAGE]);
    if (!in_window(seq)) return;
    // inside a window, one m line for each kept miss: its kind and reason, the class of its address (the 16 MiB region and
    // the 4 KiB page it lies in) and the raw-access stage keys of the submission that missed
    static const char *const kn[4] = {"?", "load", "store", "atomic"}, *const rn[4] = {"?", "outside_span", "no_chunk", "no_page"};
    char keys[RAW_KEYS * 28 + 8] = "?";
    plat_mutex_lock(&g_lock);
    for (unsigned i = 0; i < KEYS_KEPT; i++)
        if (g_rk[i].value == r.last_miss_value && g_rk[i].keys[0]) snprintf(keys, sizeof keys, "%s", g_rk[i].keys);
    plat_mutex_unlock(&g_lock);
    for (uint64_t i = 0; i < r.last_misses && i < RAW_MISS_ENTRIES; i++)
        out("m\t%llu\tsubmission=%llu\ti=%llu\taddr=%#llx\tkind=%s\treason=%s\tregion=%#llx\tpage=%#llx\tstage_keys=%s",
            (unsigned long long)seq, (unsigned long long)r.last_miss_value, (unsigned long long)i,
            (unsigned long long)r.last_miss[i].address, kn[r.last_miss[i].kind & 3], rn[r.last_miss[i].reason & 3],
            (unsigned long long)(r.last_miss[i].address >> 24), (unsigned long long)(r.last_miss[i].address >> 12), keys);
}

// The totals of the whole run, when it ends (a stop as well).
static void totals(void) {
    header();
    char buf[ROW_MAX];
    int n = (int)sn(buf, sizeof buf, "TOTAL\t%llu", (unsigned long long)__atomic_load_n(&g_present, __ATOMIC_RELAXED));
    for (int i = 0; i < C_N; i++)
        n += sn(buf + n, sizeof buf - (size_t)n, "\t%s=%llu", g_cname[i],
                      (unsigned long long)__atomic_load_n(&g_c[i], __ATOMIC_RELAXED));
    put_line(buf);
}

static void raw_misses(uint64_t seq) {
    GpuVkRawStats r;
    gpu_vk_raw_stats(&r);
    vkrt_raw_misses(&r, seq);
}

void vkrt_present(uint64_t seq) {
    header();
    __atomic_store_n(&g_present, seq, __ATOMIC_RELAXED);
    if (seq % g_every) {
        if (in_window(seq)) flush_file();
        return;
    }
    char buf[ROW_MAX];
    plat_mutex_lock(&g_lock);
    int n = (int)sn(buf, sizeof buf, "PRESENT\t%llu\tpresents=%llu", (unsigned long long)seq,
                     (unsigned long long)(seq - g_prev_present));
    for (int i = 0; i < C_N; i++) {
        uint64_t v = __atomic_load_n(&g_c[i], __ATOMIC_RELAXED);
        n += sn(buf + n, sizeof buf - (size_t)n, "\t%s=%llu", g_cname[i], (unsigned long long)(v - g_prev[i]));
        g_prev[i] = v;
    }
    g_prev_present = seq;
    plat_mutex_unlock(&g_lock);
    put_line(buf);
    raw_misses(seq);
    flush_file();
}

// ---------- the header, written at the first use (the log is not ready at start-up)
static int g_header;
static void header(void) {
    if (__atomic_exchange_n(&g_header, 1, __ATOMIC_ACQ_REL)) return;
    out("# oot2 replay trace v1\tbuild=%s %s\tspec=%s\tsummary_every=%llu\tcounters=%s\tframes_in_flight=%d", __DATE__, __TIME__,
        g_spec, (unsigned long long)g_every, getenv("OOT2_REPLAY_COUNTERS") ? getenv("OOT2_REPLAY_COUNTERS") : "off",
        (int)rt_setting(RT_SET_frames_in_flight));
    out("# S submission: present value passes draw_ops draws dispatches off_enable off_cond skipped_pipeline clears_off lost "
        "raw_stage_keys");
    out("# P pass: present value index WxH color formats (VkFormat numbers) depth stencil draws");
    out("# D draw or dispatch: present value n kind(d direct, i indexed, I indexed indirect, m mesh, M mesh indirect, c "
        "dispatch, C dispatch indirect) status key0 key1 key2 count/first/inst(recorded>replayed)/prim raw unbound tex...");
    out("# Q occlusion query: present value pass query samples pass_draws (read on completion, so after the rows that follow)");
    out("# M raw-access misses of the newest submission that had any since the last summary");
    out("# m (in a window) one kept miss: present submission i address kind reason region(16 MiB) page(4 KiB) stage_keys");
    out("# R readback statistics: present value pass att extent fmt z level layer texels, per channel min max mean finite "
        "nonfinite above");
}

// ---------- the switches
static void bad(const char *e, const char *why) {
    fprintf(stderr, "FATAL: OOT2_REPLAY_TRACE is \"%s\": %s; the items are on, every=N, A-B (presents), rb=WxH[/F], rbpass=A[-B], "
            "rbatt=I|all, rbz=N, rbn=N, rbthr=V, separated by commas, or 0 (off)\n", e, why);
    exit(70);
}

// An item of the readback (rb=, rbpass=, rbatt=, rbz=, rbn=, rbthr=): 1 when tok is one of them (a malformed one is fatal).
static int rb_item(const char *tok, const char *spec) {
    char *end;
    if (!strncmp(tok, "rb=", 3)) {
        unsigned long long w = strtoull(tok + 3, &end, 10), h = 0;
        long long fmt = -1;
        if (end == tok + 3 || *end != 'x') bad(spec, "rb= needs WxH, then optionally /F (a VkFormat number)");
        char *hs = end + 1;
        h = strtoull(hs, &end, 10);
        if (end == hs || !w || !h) bad(spec, "rb= needs WxH, then optionally /F (a VkFormat number)");
        if (*end == '/') {
            char *fs = end + 1;
            fmt = (long long)strtoull(fs, &end, 10);
            if (end == fs) bad(spec, "rb=WxH/F needs a VkFormat number after the slash");
        }
        if (*end) bad(spec, "rb= needs WxH, then optionally /F (a VkFormat number)");
        if (Rb.nsize == RB_SIZES) bad(spec, "too many rb= items");
        Rb.size[Rb.nsize].w = (uint32_t)w, Rb.size[Rb.nsize].h = (uint32_t)h, Rb.size[Rb.nsize].fmt = (int)fmt;
        Rb.nsize++;
    } else if (!strncmp(tok, "rbpass=", 7)) {
        unsigned long long a = strtoull(tok + 7, &end, 10), b = a;
        if (end == tok + 7 || !a) bad(spec, "rbpass= needs a pass number from 1, or A-B");
        if (*end == '-') {
            char *rest = end + 1;
            b = strtoull(rest, &end, 10);
            if (end == rest) bad(spec, "rbpass=A-B needs two pass numbers");
        }
        if (*end || b < a) bad(spec, "rbpass=A-B needs A not past B");
        Rb.pass_a = (uint32_t)a, Rb.pass_b = (uint32_t)b;
    } else if (!strncmp(tok, "rbatt=", 6)) {
        if (!strcmp(tok + 6, "all")) Rb.att = -1;
        else {
            unsigned long long a = strtoull(tok + 6, &end, 10);
            if (end == tok + 6 || *end || a > 7) bad(spec, "rbatt= needs 0 to 7, or all");
            Rb.att = (int)a;
        }
    } else if (!strncmp(tok, "rbz=", 4)) {
        unsigned long long a = strtoull(tok + 4, &end, 10);
        if (end == tok + 4 || *end) bad(spec, "rbz= needs a slice number");
        Rb.z = (uint32_t)a;
    } else if (!strncmp(tok, "rbn=", 4)) {
        unsigned long long a = strtoull(tok + 4, &end, 10);
        if (end == tok + 4 || *end || !a) bad(spec, "rbn= needs a number from 1");
        Rb.max = (uint32_t)a;
    } else if (!strncmp(tok, "rbthr=", 6)) {
        double v = strtod(tok + 6, &end);
        if (end == tok + 6 || *end) bad(spec, "rbthr= needs a number");
        Rb.thr = v;
    } else {
        return 0;
    }
    vkrt_rb_on = 1;
    return 1;
}

__attribute__((constructor)) static void trace_init(void) {
    const char *e = getenv("OOT2_REPLAY_TRACE");
    if (!e || !*e || !strcmp(e, "0") || !strcmp(e, "off")) return;
    snprintf(g_spec, sizeof g_spec, "%s", e);
    char buf[384];
    snprintf(buf, sizeof buf, "%s", e);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        char *end;
        if (!strcmp(tok, "on")) continue;
        if (!strncmp(tok, "rb", 2) && rb_item(tok, e)) continue;
        if (!strncmp(tok, "every=", 6)) {
            unsigned long long v = strtoull(tok + 6, &end, 10);
            if (end == tok + 6 || *end || !v) bad(e, "every= needs a number of presents, 1 or more");
            g_every = v;
            continue;
        }
        unsigned long long a = strtoull(tok, &end, 10), b = a;
        if (end == tok) bad(e, "an item that is none of those");
        if (*end == '-') {
            char *rest = end + 1;
            b = strtoull(rest, &end, 10);
            if (end == rest) bad(e, "a window needs two presents, A-B");
        }
        if (*end || b < a || a < 1) bad(e, "a window is A-B from present 1 up, A not past B");
        if (g_nwin == WINDOWS) bad(e, "too many windows");
        g_win[g_nwin].a = a, g_win[g_nwin].b = b, g_nwin++;
    }
    const char *file = getenv("OOT2_REPLAY_TRACE_FILE");
    if (file && *file) {
        g_f = fopen(file, "wb");
        if (!g_f) {
            fprintf(stderr, "FATAL: OOT2_REPLAY_TRACE_FILE %s cannot be created\n", file);
            exit(70);
        }
    }
    vkrt_on = 1;
    rc_set_present_hook(vkrt_present);
    if (!g_writer_added) g_writer_added = 1, rt_add_stop_writer(stop_writer), atexit(end_once);
}
