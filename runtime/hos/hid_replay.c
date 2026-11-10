// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Input by tick (oot2/hid_replay.h): the reader of the input file, the recorder, and the one filter nn::hid's full-key
// state read passes the host input through. The format, the options and the meaning of a tick are in the header.
//
// One state per tick: the first read in a tick takes the state (replayed, or the real input about to be recorded) and
// latches it, and every other read in that tick gets the latch. That is what makes a recorded run replay to the same
// states: the recorder writes exactly the value the game was given, and the replay gives the game that value at that
// tick.
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hos.h"
#include "oot2/hid_replay.h"
#include "oot2/loader.h"
#include "oot2/runtime.h"
#include "oot2/store.h"
#include "oot2/vsync.h"
#include "plat.h"

#define MAX_FILE_BYTES (64u << 20)
#define MAX_LINES 1048576u
#define NUM_MAX 4294967295ull
#define BUTTON_COUNT 24   // the nn::hid button bits oot2/hid_input.h names

enum { AX_LX, AX_LY, AX_RX, AX_RY, AXES };
static const char *const AXIS_NAME[AXES] = {"lx", "ly", "rx", "ry"};
// The nn::hid button bits in bit order (oot2/hid_input.h)
static const char *const BUTTON_NAME[BUTTON_COUNT] = {
    "a", "b", "x", "y", "stickl", "stickr", "l", "r", "zl", "zr", "plus", "minus", "left", "up", "right", "down",
    "stickl_left", "stickl_up", "stickl_right", "stickl_down", "stickr_left", "stickr_up", "stickr_right",
    "stickr_down"};

// What is held: the buttons and the four stick axes.
typedef struct Held {
    uint64_t buttons;
    int32_t axis[AXES];
} Held;

// One line of the file. A stretch starts at a tick (`at`), or, for a `when` line, at the tick its marker fires plus the
// delay: `at` is then set when it fires, and `marker` is the number of its marker plus one.
typedef struct Span {
    uint64_t at, count;
    Held held;
    uint8_t axes_set;   // bit i: the line gives axis i
    uint32_t marker;    // 0 for a line that starts at a tick
} Span;

// What a `when` line waits for: a value of the store compared with a constant. It fires the first tick the comparison holds.
enum { OP_EQ, OP_NE, OP_LT, OP_LE, OP_GT, OP_GE, OPS };
static const char *const OP_NAME[OPS] = {"==", "!=", "<", "<=", ">", ">="};
#define NEVER UINT64_MAX
#define MISSING_TICKS 120   // ticks a value may stay absent from tables that are all built, before the run stops by name
typedef struct Marker {
    int line, role, op;
    int64_t ival;
    double fval;
    char sval[64];
    uint64_t after;           // ticks from the firing to the start of the stretch
    int fired;
    uint64_t missing_since;   // the tick the value was first found absent from a built store, or NEVER
} Marker;

static PlatMutex g_lock = PLAT_MUTEX_INIT;   // everything below that is written after start-up
// Set while this thread is inside hid_replay_filter, holding g_lock. A stop raised there (a fault while a marker reads the
// store) runs the stop hook on this very thread, and the hook closes the recorder and writes the report, which take g_lock:
// the lock is not recursive, so they skip it when this is set (the same arrangement as gfx_null.c's g_locked_here).
static _Thread_local int g_inside;
static int g_started;
static int g_replay_on, g_rec_on;   // written once by hid_replay_start, before any reader

static char g_replay_path[1024], g_rec_path[1024];
static Span *g_spans;
static size_t g_nspans, g_cap;
static Marker *g_markers;
static size_t g_nmarkers, g_markers_cap;
static int g_store_loaded;
static char g_sha_hex[65];
static size_t g_replay_bytes;   // the size of the replayed file

static int g_anchored;
static uint64_t g_base;   // the virtual tick that is tick 0 of the file
static int g_latched;
static uint64_t g_lat_tick;
static Held g_lat;
static uint64_t g_served;   // ticks that were given a state

static FILE *g_rec_file;
static int g_rec_open_run, g_rec_closed, g_rec_failed;
static Held g_run;
static uint64_t g_run_start, g_run_last, g_rec_lines, g_rec_ticks;

// ---- the reader
// What went wrong on a line, as a fatal that names the file and the line.
static void bad(int line, const char *fmt, ...) __attribute__((noreturn, format(printf, 2, 3)));
static void bad(int line, const char *fmt, ...) {
    char what[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(what, sizeof what, fmt, ap);
    va_end(ap);
    rt_fatal("input file %s, line %d: %s", g_replay_path, line, what);
}

// A word of the file as a log shows it: at most 24 bytes, anything that is not printable ASCII a '?'.
static const char *shown(const char *t, char out[32]) {
    size_t n = 0;
    for (; t[n] && n < 24; n++) out[n] = (unsigned char)t[n] >= 0x20 && (unsigned char)t[n] < 0x7f ? t[n] : '?';
    if (t[n]) {
        out[n++] = '.', out[n++] = '.', out[n++] = '.';
    }
    out[n] = 0;
    return out;
}

static int ieq(const char *a, const char *b) {
    for (; *a && *b; a++, b++)
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
    return !*a && !*b;
}

// The next word of a line, ended in place; NULL at the end of the line.
static char *next_word(char **p) {
    char *s = *p;
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) {
        *p = s;
        return NULL;
    }
    char *e = s;
    while (*e && *e != ' ' && *e != '\t') e++;
    if (*e) *e++ = 0;
    *p = e;
    return s;
}

// A decimal number of digits only, no sign: 0, or -1 when t is not one, -2 when it is above NUM_MAX.
static int number(const char *t, uint64_t *out) {
    if (!*t) return -1;
    uint64_t v = 0;
    for (; *t; t++) {
        if (*t < '0' || *t > '9') return -1;
        v = v * 10 + (uint64_t)(*t - '0');
        if (v > NUM_MAX) return -2;
    }
    *out = v;
    return 0;
}

// A stick value: an optional '-' and digits, in -32768..32767; 0, -1 not a number, -2 out of range.
static int stick_value(const char *t, int32_t *out) {
    int neg = *t == '-';
    uint64_t v = 0;
    if (neg) t++;
    if (!*t) return -1;
    for (; *t; t++) {
        if (*t < '0' || *t > '9') return -1;
        v = v * 10 + (uint64_t)(*t - '0');
        if (v > 40000) v = 40000;   // far enough out of range; no overflow
    }
    if (neg ? v > 32768 : v > 32767) return -2;
    *out = neg ? -(int32_t)v : (int32_t)v;
    return 0;
}

// The held items of a line, after its start (tick and count): the buttons and the stick values.
static void parse_held(int line, char **p, Span *s) {
    char shown_buf[32];
    for (char *w; (w = next_word(p));) {
        char *eq = strchr(w, '=');
        if (eq) {
            *eq++ = 0;
            int axis = -1;
            for (int i = 0; i < AXES; i++)
                if (ieq(w, AXIS_NAME[i])) axis = i;
            if (axis < 0)
                bad(line, "unknown stick \"%s\" (the sticks are lx, ly, rx and ry)", shown(w, shown_buf));
            if (s->axes_set >> axis & 1) bad(line, "%s is given twice on the line", AXIS_NAME[axis]);
            int32_t v = 0;
            int r = stick_value(eq, &v);
            char val_buf[32];
            if (r == -1) bad(line, "the value of %s, \"%s\", is not a number", AXIS_NAME[axis], shown(eq, val_buf));
            if (r == -2) bad(line, "%s=%s is outside -32768..32767", AXIS_NAME[axis], shown(eq, val_buf));
            s->held.axis[axis] = v;
            s->axes_set |= (uint8_t)(1u << axis);
            continue;
        }
        int bit = -1;
        for (int i = 0; i < BUTTON_COUNT; i++)
            if (ieq(w, BUTTON_NAME[i])) bit = i;
        if (bit < 0) bad(line, "unknown button \"%s\"", shown(w, shown_buf));
        if (s->held.buttons >> bit & 1) bad(line, "button %s is given twice on the line", BUTTON_NAME[bit]);
        s->held.buttons |= 1ull << bit;
    }
}

// at <tick> for <count> [<held> ...]. The rest of the line after the start (for, the count, the held items) is the same
// for every kind of line, in parse_for.
static void parse_for(int line, char **p, Span *s, char *w) {   // w: the word after the start, if already read
    char buf[32];
    uint64_t v = 0;
    if (!w) w = next_word(p);
    if (!w || strcmp(w, "for")) bad(line, "expected \"for\" and a count after the %s, not \"%s\"", s->marker ? "value" : "tick", w ? shown(w, buf) : "the end of the line");
    w = next_word(p);
    if (!w) bad(line, "\"for\" needs a count");
    int r = number(w, &v);
    if (r == -1) bad(line, "the count \"%s\" is not a decimal number", shown(w, buf));
    if (r == -2) bad(line, "the count \"%s\" is above %llu", shown(w, buf), (unsigned long long)NUM_MAX);
    if (v == 0) bad(line, "the count is 0; a stretch lasts 1 tick or more");
    s->count = v;
    parse_held(line, p, s);
}

static void parse_at(int line, char **p, Span *s) {
    char buf[32];
    char *w = next_word(p);
    uint64_t v = 0;
    if (!w) bad(line, "\"at\" needs a tick, then \"for\" and a count");
    int r = number(w, &v);
    if (r == -1) bad(line, "the tick \"%s\" is not a decimal number", shown(w, buf));
    if (r == -2) bad(line, "the tick \"%s\" is above %llu", shown(w, buf), (unsigned long long)NUM_MAX);
    s->at = v;
    parse_for(line, p, s, NULL);
}

// when <role> <op> <value> [after <ticks>] for <count> [<held> ...]. The role is one the store description holds (the
// prepare step wrote it from the player's names file), the operator one of == != < <= > >=, the value a constant of the
// role's kind. The line is read here and checked against the description now, so a role it does not know, a value that does
// not fit the role's kind and a missing description all stop the run at start-up, naming the line.
static int add_marker(const Marker *m) {
    if (g_nmarkers == g_markers_cap) {
        size_t cap = g_markers_cap ? g_markers_cap * 2 : 8;
        Marker *n = realloc(g_markers, cap * sizeof *g_markers);
        if (!n) rt_fatal("input file %s: out of memory", g_replay_path);
        g_markers = n;
        g_markers_cap = cap;
    }
    g_markers[g_nmarkers] = *m;
    return (int)++g_nmarkers;
}

// A decimal number as a person writes one: an optional sign, digits with an optional point (at least one digit), and an
// optional exponent. Nothing else (no inf, no nan, no hexadecimal).
static int float_syntax(const char *t) {
    if (*t == '+' || *t == '-') t++;
    int digits = 0;
    while (isdigit((unsigned char)*t)) t++, digits++;
    if (*t == '.') {
        t++;
        while (isdigit((unsigned char)*t)) t++, digits++;
    }
    if (!digits) return 0;
    if (*t == 'e' || *t == 'E') {
        t++;
        if (*t == '+' || *t == '-') t++;
        if (!isdigit((unsigned char)*t)) return 0;
        while (isdigit((unsigned char)*t)) t++;
    }
    return !*t;
}

static void parse_when(int line, char **p, Span *s) {
    char buf[32], why[512];
    if (!g_store_loaded) {
        char path[1100];
        snprintf(path, sizeof path, "%s/%s", hos_game_folder(), STORE_FILE);
        if (!store_load(path, why, sizeof why)) bad(line, "a \"when\" line needs the store description: %s", why);
        g_store_loaded = 1;
    }
    char *w = next_word(p);
    if (!w) bad(line, "\"when\" needs a role, an operator and a value");
    int role = store_role(w);
    if (role < 0) {
        char list[256] = "";
        for (int i = 0; i < store_role_count() && strlen(list) + 60 < sizeof list; i++)
            snprintf(list + strlen(list), sizeof list - strlen(list), "%s%s", i ? ", " : "", store_role_name(i));
        bad(line, "unknown role \"%s\" (the store description has: %s)", shown(w, buf), *list ? list : "no roles");
    }
    StoreKind kind = store_role_kind(role);
    Marker m = {.line = line, .role = role, .missing_since = NEVER};
    w = next_word(p);
    if (!w) bad(line, "\"when %s\" needs an operator and a value", store_role_name(role));
    m.op = -1;
    for (int i = 0; i < OPS; i++)
        if (!strcmp(w, OP_NAME[i])) m.op = i;
    if (m.op < 0) bad(line, "unknown operator \"%s\" (use == != < <= > >=)", shown(w, buf));
    if (kind == STORE_STRING && m.op != OP_EQ && m.op != OP_NE)
        bad(line, "%s is a string; only == and != apply to it", store_role_name(role));
    w = next_word(p);
    if (!w) bad(line, "\"when %s %s\" needs a value", store_role_name(role), OP_NAME[m.op]);
    switch (kind) {
    case STORE_BOOL:
        if (!strcmp(w, "1") || ieq(w, "true")) m.ival = 1;
        else if (!strcmp(w, "0") || ieq(w, "false")) m.ival = 0;
        else bad(line, "the value \"%s\" of the bool %s is not 0, 1, true or false", shown(w, buf), store_role_name(role));
        break;
    case STORE_INT:
    case STORE_UINT: {
        int neg = kind == STORE_INT && *w == '-';
        uint64_t v = 0;
        int r = number(w + neg, &v);
        int64_t lo = kind == STORE_INT ? INT32_MIN : 0, hi = kind == STORE_INT ? INT32_MAX : (int64_t)UINT32_MAX;
        if (r == -1) bad(line, "the value \"%s\" of %s is not a whole number", shown(w, buf), store_role_name(role));
        if (r == -2 || (neg ? -(int64_t)v < lo : v > (uint64_t)hi))
            bad(line, "the value \"%s\" of %s is outside %lld..%lld", shown(w, buf), store_role_name(role), (long long)lo, (long long)hi);
        m.ival = neg ? -(int64_t)v : (int64_t)v;
        break;
    }
    case STORE_FLOAT: {
        char *end = NULL;
        // the store holds a single float, which is widened to compare: the constant is rounded to a single float the same way
        m.fval = float_syntax(w) ? (double)(float)strtod(w, &end) : 0;
        if (!end || *end) bad(line, "the value \"%s\" of %s is not a number", shown(w, buf), store_role_name(role));
        break;
    }
    case STORE_STRING:
        if (strlen(w) >= sizeof m.sval) bad(line, "the value of %s is longer than %zu characters", store_role_name(role), sizeof m.sval - 1);
        snprintf(m.sval, sizeof m.sval, "%s", w);
        break;
    }
    // [after <ticks>]: the delay from the firing to the start of the stretch
    w = next_word(p);
    if (w && !strcmp(w, "after")) {
        w = next_word(p);
        if (!w) bad(line, "\"after\" needs a number of ticks");
        uint64_t v = 0;
        int r = number(w, &v);
        if (r == -1) bad(line, "the delay \"%s\" is not a decimal number", shown(w, buf));
        if (r == -2) bad(line, "the delay \"%s\" is above %llu", shown(w, buf), (unsigned long long)NUM_MAX);
        m.after = v;
        w = NULL;
    }
    s->marker = (uint32_t)add_marker(&m);
    parse_for(line, p, s, w);
}

// Extension point: the line kinds. Each kind is the first word of a line and the function that reads the rest of it
// into a span.
typedef struct LineKind {
    const char *word;
    void (*parse)(int line, char **p, Span *s);
} LineKind;
static const LineKind LINE_KINDS[] = {{"at", parse_at}, {"when", parse_when}};

static void add_span(int line, const Span *s) {
    if (g_nspans == MAX_LINES) bad(line, "more than %u lines in the file", MAX_LINES);
    if (g_nspans == g_cap) {
        size_t cap = g_cap ? g_cap * 2 : 16;
        Span *n = realloc(g_spans, cap * sizeof *g_spans);
        if (!n) rt_fatal("input file %s: out of memory", g_replay_path);
        g_spans = n;
        g_cap = cap;
    }
    g_spans[g_nspans++] = *s;
}

static void parse_file(char *buf, size_t n) {
    if (n >= 3 && !memcmp(buf, "\xEF\xBB\xBF", 3)) buf += 3, n -= 3;   // a UTF-8 byte order mark, as Windows tools write
    char *nul = memchr(buf, 0, n);
    if (nul) {
        int line = 1;
        for (const char *c = buf; c < nul; c++) line += *c == '\n';
        bad(line, "a NUL byte; the input file is text");
    }
    int line = 0;
    for (char *start = buf; start < buf + n;) {
        char *end = memchr(start, '\n', (size_t)(buf + n - start));
        if (!end) end = buf + n;
        char *next = end + 1;
        *end = 0;   // buf holds one more byte than the file, always 0
        line++;
        char *hash = strchr(start, '#');
        if (hash) *hash = 0;
        char *p = start;
        for (char *e = p + strlen(p); e > p && e[-1] == '\r'; ) *--e = 0;
        char *kind = next_word(&p);
        if (kind) {
            const LineKind *k = NULL;
            for (size_t i = 0; i < sizeof LINE_KINDS / sizeof *LINE_KINDS; i++)
                if (!strcmp(kind, LINE_KINDS[i].word)) k = &LINE_KINDS[i];
            char shown_buf[32];
            if (!k) bad(line, "unknown line kind \"%s\" (a line starts with \"at\" or \"when\")", shown(kind, shown_buf));
            Span s = {0};
            k->parse(line, &p, &s);
            add_span(line, &s);
        }
        start = next;
    }
}

static void load_replay(const char *path) {
    snprintf(g_replay_path, sizeof g_replay_path, "%s", path);
    FILE *f = fopen(path, "rb");
    if (!f) rt_fatal("input file %s: cannot be opened (%s)", path, strerror(errno));
    size_t cap = 1 << 16, n = 0;
    char *buf = malloc(cap + 1);
    if (!buf) rt_fatal("input file %s: out of memory", path);
    for (;;) {
        size_t got = fread(buf + n, 1, cap - n, f);
        n += got;
        if (n < cap) {
            if (ferror(f)) rt_fatal("input file %s: cannot be read (%s)", path, strerror(errno));
            break;
        }
        if (cap >= MAX_FILE_BYTES) rt_fatal("input file %s: larger than %u bytes", path, MAX_FILE_BYTES);
        cap *= 2;
        char *grown = realloc(buf, cap + 1);
        if (!grown) rt_fatal("input file %s: out of memory", path);
        buf = grown;
    }
    fclose(f);
    buf[n] = 0;
    uint8_t digest[32];
    rt_sha256(buf, n, digest);
    for (int i = 0; i < 32; i++) snprintf(g_sha_hex + 2 * i, 3, "%02x", digest[i]);
    g_replay_bytes = n;
    parse_file(buf, n);
    free(buf);
}

// Whether path is an existing file with exactly the bytes of the replayed file. The name test cannot tell two spellings
// of one file apart (./, upper case, a hard link or a symbolic link), and opening the record for writing would truncate
// the tape; the content can. A different file with the same bytes is refused too: it holds nothing the record would not
// write again.
static int has_replay_bytes(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char *buf = malloc(g_replay_bytes + 2);
    int same = 0;
    if (buf) {
        size_t got = fread(buf, 1, g_replay_bytes + 1, f);   // one byte more than the tape: a longer file is not it
        if (got == g_replay_bytes && !ferror(f)) {
            uint8_t digest[32];
            char hex[65];
            rt_sha256(buf, got, digest);
            for (int i = 0; i < 32; i++) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
            same = !strcmp(hex, g_sha_hex);
        }
        free(buf);
    }
    fclose(f);
    return same;
}

// ---- the replay: the state of a tick
// Whether a stretch is live at a tick: the tick is inside its run. The run of a `when` line exists once its marker fired.
static int span_live(const Span *s, uint64_t tick) {
    if (s->marker && !g_markers[s->marker - 1].fired) return 0;
    return tick >= s->at && tick - s->at < s->count;
}

// Whether the value of the store satisfies a marker.
static int holds(const Marker *m, const StoreValue *v) {
    int c;
    if (store_role_kind(m->role) == STORE_STRING) {
        c = strcmp(v->s, m->sval);
    } else if (store_role_kind(m->role) == STORE_FLOAT) {
        if (v->f != v->f) return 0;   // not a number: no comparison holds
        c = (v->f > m->fval) - (v->f < m->fval);
    } else {
        c = (v->i > m->ival) - (v->i < m->ival);
    }
    switch (m->op) {
    case OP_EQ: return c == 0;
    case OP_NE: return c != 0;
    case OP_LT: return c < 0;
    case OP_LE: return c <= 0;
    case OP_GT: return c > 0;
    default: return c >= 0;
    }
}

// The first read of a tick: look at each marker that has not fired. A marker fires the first tick its value satisfies it,
// once; its stretch starts `after` ticks later. A store the game has not built yet is a wait; a value absent from a built
// store for MISSING_TICKS ticks in a row, or an address the description leads to that is not guest memory, is a stop by name
// (fatal, with the line): the text of that stop goes to *fatal, for the caller to raise once it has let go of the lock.
static void markers_update(uint64_t tick, char *fatal, size_t fatal_size) {
    for (size_t i = 0; i < g_nmarkers && !fatal[0]; i++) {
        Marker *m = &g_markers[i];
        if (m->fired) continue;
        StoreValue v;
        char why[300];
        StoreRead r = store_read(m->role, &v, why, sizeof why);
        if (r == STORE_BAD) {
            snprintf(fatal, fatal_size, "input file %s, line %d: reading %s from the store: %s", g_replay_path, m->line,
                     store_role_name(m->role), why);
        } else if (r == STORE_MISSING) {
            if (m->missing_since == NEVER) m->missing_since = tick;
            if (tick - m->missing_since >= MISSING_TICKS)
                snprintf(fatal, fatal_size,
                         "input file %s, line %d: %s has no entry in the game's store (every table of its kind is built "
                         "and none has it); the name in the names file is not one of the game's, or %s is from another "
                         "game folder",
                         g_replay_path, m->line, store_role_name(m->role), STORE_FILE);
        } else if (r == STORE_OK) {
            m->missing_since = NEVER;
            if (holds(m, &v)) {
                m->fired = 1;
                for (size_t k = 0; k < g_nspans; k++)
                    if (g_spans[k].marker == i + 1) g_spans[k].at = tick + m->after;
                rt_log("hid: the marker of line %d (%s %s) fired at tick %llu; its stretch starts at tick %llu", m->line,
                       store_role_name(m->role), OP_NAME[m->op], (unsigned long long)tick,
                       (unsigned long long)(tick + m->after));
            }
        } else {
            m->missing_since = NEVER;   // a wait: the game has not built the store, or this table, yet
        }
    }
}

static Held replay_state(uint64_t tick) {
    Held h = {0};
    for (size_t i = 0; i < g_nspans; i++) {
        const Span *s = &g_spans[i];
        if (!span_live(s, tick)) continue;
        h.buttons |= s->held.buttons;
        for (int a = 0; a < AXES; a++)
            if (s->axes_set >> a & 1) h.axis[a] = s->held.axis[a];
    }
    return h;
}

// ---- the recorder
static int held_zero(const Held *h) { return !h->buttons && !h->axis[0] && !h->axis[1] && !h->axis[2] && !h->axis[3]; }
static int held_eq(const Held *a, const Held *b) { return !memcmp(a, b, sizeof *a); }

// One closed run, as a line; a stretch of rest is no line, since a tick no line covers holds nothing.
static void emit(const Held *h, uint64_t start, uint64_t count) {
    if (held_zero(h) || !g_rec_file) return;
    int bad_write = fprintf(g_rec_file, "at %llu for %llu", (unsigned long long)start, (unsigned long long)count) < 0;
    for (int i = 0; i < BUTTON_COUNT; i++)
        if (h->buttons >> i & 1) bad_write |= fprintf(g_rec_file, " %s", BUTTON_NAME[i]) < 0;
    for (int a = 0; a < AXES; a++)
        if (h->axis[a]) bad_write |= fprintf(g_rec_file, " %s=%d", AXIS_NAME[a], (int)h->axis[a]) < 0;
    bad_write |= fputc('\n', g_rec_file) == EOF;
    bad_write |= fflush(g_rec_file) != 0;   // a run killed from outside keeps every closed run
    g_rec_lines++;
    if (bad_write && !g_rec_failed) {
        g_rec_failed = 1;
        rt_log("input record: cannot write %s (%s); the record stops here", g_rec_path, strerror(errno));
    }
}

static void record(uint64_t tick, const Held *h) {
    if (!g_rec_open_run) {
        g_rec_open_run = 1;
        g_run = *h;
        g_run_start = g_run_last = tick;
    } else if (held_eq(&g_run, h)) {
        g_run_last = tick;
    } else {
        emit(&g_run, g_run_start, tick - g_run_start);
        g_run = *h;
        g_run_start = g_run_last = tick;
    }
}

// ---- the filter
static Held held_of(const HidHostInput *in) {
    Held h = {.buttons = in->buttons};
    h.axis[AX_LX] = in->lx, h.axis[AX_LY] = in->ly, h.axis[AX_RX] = in->rx, h.axis[AX_RY] = in->ry;
    return h;
}

static void give(const Held *h, HidHostInput *in) {
    in->buttons = h->buttons;
    in->lx = h->axis[AX_LX], in->ly = h->axis[AX_LY], in->rx = h->axis[AX_RX], in->ry = h->axis[AX_RY];
}

void hid_replay_filter(HidHostInput *in) {
    if (!g_replay_on && !g_rec_on) return;
    uint64_t now = rt_vsync_tick();
    char fatal[512] = "";
    plat_mutex_lock(&g_lock);
    g_inside = 1;
    if (!g_anchored) {
        g_anchored = 1;
        g_base = now;
        rt_log("hid: first read of the pad at virtual tick %llu; tick 0 of the input file", (unsigned long long)now);
    }
    uint64_t tick = now >= g_base ? now - g_base : 0;
    if (!g_latched || tick > g_lat_tick) {   // the first read of a tick takes the state
        if (g_replay_on && g_nmarkers) markers_update(tick, fatal, sizeof fatal);
        Held h = g_replay_on ? replay_state(tick) : held_of(in);
        if (g_rec_on && !g_rec_closed) record(tick, &h);
        g_lat = h;
        g_lat_tick = tick;
        g_latched = 1;
        g_served++;
    }
    give(&g_lat, in);
    g_inside = 0;
    plat_mutex_unlock(&g_lock);
    if (fatal[0]) rt_fatal("%s", fatal);   // after the lock: the stop hook closes the recorder under it
}

void hid_replay_finish(void) {
    int held = g_inside;   // a stop raised under the lock by this thread: it already holds it
    if (!held) plat_mutex_lock(&g_lock);
    if (g_rec_on && !g_rec_closed) {
        g_rec_closed = 1;
        if (g_rec_open_run) emit(&g_run, g_run_start, g_run_last - g_run_start + 1);
        g_rec_ticks = g_rec_open_run ? g_run_last + 1 : 0;
        if (g_rec_file && fclose(g_rec_file) && !g_rec_failed) {
            g_rec_failed = 1;
            rt_log("input record: cannot close %s (%s)", g_rec_path, strerror(errno));
        }
        g_rec_file = NULL;
    }
    if (!held) plat_mutex_unlock(&g_lock);
}

void hid_replay_rows(HidReplayRowFn row, void *ctx) {
    char why[400];
    int held = g_inside;   // see hid_replay_finish
    if (!held) plat_mutex_lock(&g_lock);
    if (g_replay_on) {
        if (g_anchored)
            snprintf(why, sizeof why,
                     "sha256 %s; %zu lines; replaces the real input; tick 0 was virtual tick %llu", g_sha_hex, g_nspans,
                     (unsigned long long)g_base);
        else
            snprintf(why, sizeof why, "sha256 %s; %zu lines; replaces the real input; the game has not read the pad",
                     g_sha_hex, g_nspans);
        if (g_nmarkers) {
            size_t fired = 0;
            for (size_t i = 0; i < g_nmarkers; i++) fired += (size_t)g_markers[i].fired;
            size_t n = strlen(why);
            snprintf(why + n, sizeof why - n, "; %zu of %zu markers fired", fired, g_nmarkers);
        }
        row(ctx, "input", g_replay_path, "implemented", g_served, why);
    }
    if (g_rec_on) {
        uint64_t ticks = g_rec_closed ? g_rec_ticks : g_rec_open_run ? g_run_last + 1 : 0;
        snprintf(why, sizeof why, "run-length by tick from the game's first read of the pad; %llu ticks, %s%s",
                 (unsigned long long)ticks, g_rec_closed ? "closed" : "still open", g_rec_failed ? "; a write failed" : "");
        row(ctx, "input-record", g_rec_path, "implemented", g_rec_lines, why);
    }
    if (!held) plat_mutex_unlock(&g_lock);
}

// ---- start-up
void hid_replay_start(void) {
    if (__atomic_exchange_n(&g_started, 1, __ATOMIC_ACQ_REL)) return;
    const char *in = getenv("OOT2_INPUT_FILE"), *rec = getenv("OOT2_INPUT_RECORD");
    if (in && !*in) in = NULL;   // an empty variable is no option, as for OOT2_CONFIG
    if (rec && !*rec) rec = NULL;
    if (in) {
        load_replay(in);
        rt_log("hid: replaying %s: %zu lines, sha256 %s; the real input is replaced", in, g_nspans, g_sha_hex);
        g_replay_on = 1;
    }
    if (rec) {
        if (in && (!strcmp(in, rec) || has_replay_bytes(rec)))
            rt_fatal("input record %s: it is the input file being replayed", rec);
        snprintf(g_rec_path, sizeof g_rec_path, "%s", rec);
        g_rec_file = fopen(rec, "wb");
        if (!g_rec_file) rt_fatal("input record %s: cannot be created (%s)", rec, strerror(errno));
        fputs("# the input the game was given, by tick (oot2/hid_replay.h): at <tick> for <count> <held>...\n", g_rec_file);
        fflush(g_rec_file);
        rt_log("hid: recording the input to %s", rec);
        g_rec_on = 1;
    }
}
