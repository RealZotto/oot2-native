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
// A submission belongs to the present that follows it (the one the game makes next), so the presents are the numbers the
// capture's shots carry. The file is OOT2_REPLAY_TRACE_FILE (created anew), else the run log, each line after "[rt] trace".
// How to read it is in docs/architecture.md (the replay's trace).
//
// Cost: a call site tests the static int vkrt_on, which is 0 unless the variable is set. The counters of the summary are
// atomic increments (the completion thread counts the queries), the rows are formatted only inside a window.
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
static char g_spec[128];
static uint64_t g_present;       // atomic: the newest present made
static int g_writer_added;

// ---------- the counters of the summary lines
#define COUNTERS(X)                                                                                                    \
    X(SUBS, "subs") X(PASSES, "passes") X(PASSES_EMPTY, "passes_with_no_draw") X(DRAWS, "draws") X(MESH, "mesh_draws")   \
    X(DISPATCH, "dispatches") X(OFF_ENABLE, "off_enable") X(OFF_COND, "off_cond") X(CLEARS_OFF, "clears_off")          \
    X(SKIP_PIPE, "skipped_pipeline") X(ZERO_WORK, "zero_work") X(INDIRECT, "indirect") X(RAW, "raw_access")            \
    X(UNB_C1, "unbound_c1") X(UNB_SB, "unbound_storage") X(UNB_UB, "unbound_uniform") X(TEX, "tex_slots")              \
    X(TEX_NOENTRY, "tex_no_entry") X(TEX_UNREG, "tex_unregistered") X(TEX_ASTC, "tex_astc") X(TEX_ZR, "tex_zero_regions") \
    X(QUERIES, "queries") X(Q_ZERO, "queries_zero_samples") X(Q_ZERO_DRAWS, "queries_zero_samples_with_draws")
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
} Row;

static struct {
    uint32_t w, h, ncolor;
    VkFormat color[8], depth, stencil;
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

void vkrt_submit_end(void) {
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
    out("P\t%llu\t%llu\t%u\t%ux%u\tcolor=%s\tdepth=%d\tstencil=%d\tdraws=%u", (unsigned long long)T.present,
        (unsigned long long)T.value, T.passes, Pass.w, Pass.h, fm[0] ? fm : "none", (int)Pass.depth, (int)Pass.stencil, draws);
}

// ---------- one row per draw and dispatch
static void row_open(char kind, const char *status, uint32_t n) {
    memset(&Row, 0, sizeof Row);
    Row.open = 1, Row.kind = kind, Row.status = status, Row.n = n;
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
}

// ---------- the switches
static void bad(const char *e, const char *why) {
    fprintf(stderr, "FATAL: OOT2_REPLAY_TRACE is \"%s\": %s; the items are on, every=N, A-B (presents), separated by commas, "
            "or 0 (off)\n", e, why);
    exit(70);
}

__attribute__((constructor)) static void trace_init(void) {
    const char *e = getenv("OOT2_REPLAY_TRACE");
    if (!e || !*e || !strcmp(e, "0") || !strcmp(e, "off")) return;
    snprintf(g_spec, sizeof g_spec, "%s", e);
    char buf[128];
    snprintf(buf, sizeof buf, "%s", e);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        char *end;
        if (!strcmp(tok, "on")) continue;
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
