// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The reader of a module's resource description (oot2/shader_desc.h) and the padding rule of a vertex module.
// A description is the translator's JSON, 5 to 40 KiB, and only a handful of members matter to the builder, so this is a
// scanner that walks the text once, reads the members it knows and skips the rest: no tree, no allocation, and a
// refusal at the first malformed byte, with the offset.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/shader_desc.h"

typedef struct J {
    const char *begin, *p, *end;
    const char *err;
} J;

static int fail(J *j, const char *e) {
    if (!j->err) j->err = e;
    return 0;
}

static void ws(J *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t')) j->p++;
}

static int eat(J *j, char c) {
    ws(j);
    if (j->p < j->end && *j->p == c) {
        j->p++;
        return 1;
    }
    return 0;
}

// A string; at most cap - 1 bytes are kept (out may be NULL). Escapes become themselves; \u becomes '?'.
static int jstring(J *j, char *out, size_t cap) {
    ws(j);
    if (j->p >= j->end || *j->p != '"') return fail(j, "a string was expected");
    j->p++;
    size_t n = 0;
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end) break;
            c = *j->p++;
            if (c == 'u') {
                for (int i = 0; i < 4 && j->p < j->end; i++) j->p++;
                c = '?';
            } else if (c == 'n') {
                c = '\n';
            } else if (c == 't') {
                c = '\t';
            }
        }
        if (out && n + 1 < cap) out[n++] = c;
    }
    if (j->p >= j->end) return fail(j, "a string is not terminated");
    j->p++;
    if (out && cap) out[n] = 0;
    return 1;
}

// A number, kept as the integer part (a fraction or an exponent is read and dropped).
static int jint(J *j, long long *v) {
    ws(j);
    int neg = 0;
    if (j->p < j->end && *j->p == '-') neg = 1, j->p++;
    if (j->p >= j->end || *j->p < '0' || *j->p > '9') return fail(j, "a number was expected");
    long long x = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        if (x < (1LL << 52)) x = x * 10 + (*j->p - '0');
        j->p++;
    }
    while (j->p < j->end && (*j->p == '.' || *j->p == 'e' || *j->p == 'E' || *j->p == '+' || *j->p == '-' ||
                             (*j->p >= '0' && *j->p <= '9')))
        j->p++;
    *v = neg ? -x : x;
    return 1;
}

static int jword(J *j, const char *w) {
    size_t n = strlen(w);
    if ((size_t)(j->end - j->p) < n || memcmp(j->p, w, n)) return 0;
    j->p += n;
    return 1;
}

static int jbool(J *j, int *v) {
    ws(j);
    if (jword(j, "true")) return *v = 1, 1;
    if (jword(j, "false")) return *v = 0, 1;
    return fail(j, "true or false was expected");
}

#define MAX_DEPTH 24

static int jskip(J *j, int depth);

// The members of an object: member(j, key, ctx) reads the value of key and returns 0 on a failure.
typedef int (*MemberFn)(J *j, const char *key, void *ctx);

static int jobject(J *j, MemberFn member, void *ctx, int depth) {
    if (depth > MAX_DEPTH) return fail(j, "nested too deeply");
    if (!eat(j, '{')) return fail(j, "an object was expected");
    if (eat(j, '}')) return 1;
    for (;;) {
        char key[48];
        if (!jstring(j, key, sizeof key)) return 0;
        if (!eat(j, ':')) return fail(j, "a colon was expected");
        if (!member(j, key, ctx)) return 0;
        if (eat(j, ',')) continue;
        if (eat(j, '}')) return 1;
        return fail(j, "a comma or the end of the object was expected");
    }
}

typedef int (*ElementFn)(J *j, void *ctx);

static int jarray(J *j, ElementFn element, void *ctx, int depth) {
    if (depth > MAX_DEPTH) return fail(j, "nested too deeply");
    if (!eat(j, '[')) return fail(j, "an array was expected");
    if (eat(j, ']')) return 1;
    for (;;) {
        if (!element(j, ctx)) return 0;
        if (eat(j, ',')) continue;
        if (eat(j, ']')) return 1;
        return fail(j, "a comma or the end of the array was expected");
    }
}

static int skip_member(J *j, const char *key, void *ctx) {
    (void)key;
    return jskip(j, *(int *)ctx + 1) ;
}

static int skip_element(J *j, void *ctx) { return jskip(j, *(int *)ctx + 1); }

static int jskip(J *j, int depth) {
    ws(j);
    if (j->p >= j->end) return fail(j, "the text ends where a value was expected");
    switch (*j->p) {
    case '{': return jobject(j, skip_member, &depth, depth);
    case '[': return jarray(j, skip_element, &depth, depth);
    case '"': return jstring(j, NULL, 0);
    case 't': case 'f': {
        int v;
        return jbool(j, &v);
    }
    case 'n':
        if (jword(j, "null")) return 1;
        return fail(j, "an unknown word");
    default: {
        long long v;
        return jint(j, &v);
    }
    }
}

// ---- the members that matter

typedef struct Ctx {
    ShaderDesc *d;
    char stage[8];
    int have_stage;
    ShdIo cur;
    ShdIo *list;
    uint32_t *count;
    int overflow;
} Ctx;

static int io_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    long long v;
    if (!strcmp(key, "location")) {
        if (!jint(j, &v)) return 0;
        x->cur.location = (int32_t)v;
    } else if (!strcmp(key, "component")) {
        if (!jint(j, &v)) return 0;
        x->cur.component = (uint8_t)(v < 0 || v > 3 ? 255 : v);
    } else if (!strcmp(key, "components")) {
        if (!jint(j, &v)) return 0;
        x->cur.components = (uint8_t)(v < 1 || v > 4 ? 255 : v);
    } else if (!strcmp(key, "array_length")) {
        if (!jint(j, &v)) return 0;
        x->cur.array_length = (int32_t)(v < 1 ? 1 : v > 1024 ? 1024 : v);
    } else if (!strcmp(key, "used")) {
        int b;
        if (!jbool(j, &b)) return 0;
        x->cur.used = (uint8_t)b;
    } else if (!strcmp(key, "array_depth")) {
        if (!jint(j, &v)) return 0;
        x->cur.array_depth = (int8_t)(v < 0 ? -1 : v > 8 ? 8 : v);
    } else if (!strcmp(key, "per_primitive")) {
        int b;
        if (!jbool(j, &b)) return 0;
        x->cur.per_primitive = (int8_t)b;
    } else if (!strcmp(key, "type")) {
        char t[16];
        if (!jstring(j, t, sizeof t)) return 0;
        int bits = 0;
        if (!strncmp(t, "float", 5)) x->cur.type = SHD_FLOAT, bits = atoi(t + 5);
        else if (!strncmp(t, "sint", 4)) x->cur.type = SHD_SINT, bits = atoi(t + 4);
        else if (!strncmp(t, "uint", 4)) x->cur.type = SHD_UINT, bits = atoi(t + 4);
        else return fail(j, "an interface type that is not floatN, sintN or uintN");
        x->cur.bits = (uint8_t)bits;
    } else {
        return jskip(j, 4);
    }
    return 1;
}

static int io_element(J *j, void *c) {
    Ctx *x = c;
    memset(&x->cur, 0, sizeof x->cur);
    x->cur.array_length = 1;
    x->cur.array_depth = x->cur.per_primitive = -1;
    if (!jobject(j, io_member, x, 3)) return 0;
    if (*x->count >= SHD_MAX_IO) {
        x->overflow = 1;
        return fail(j, "more interface entries than the builder reads");
    }
    x->list[(*x->count)++] = x->cur;
    return 1;
}

static int iface_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    if (!strcmp(key, "inputs")) {
        x->list = x->d->inputs, x->count = &x->d->ninputs;
        return jarray(j, io_element, x, 2);
    }
    if (!strcmp(key, "outputs")) {
        x->list = x->d->outputs, x->count = &x->d->noutputs;
        return jarray(j, io_element, x, 2);
    }
    return jskip(j, 2);
}

static int layout_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    if (!strcmp(key, "fits")) return jbool(j, &x->d->fits);
    return jskip(j, 2);
}

static int raw_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    if (!strcmp(key, "physical_pointers")) return jbool(j, &x->d->raw_pointers);
    return jskip(j, 2);
}

static int top_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    long long v;
    if (!strcmp(key, "schema")) {
        if (!jint(j, &v)) return 0;
        x->d->schema = (uint32_t)v;
        return 1;
    }
    if (!strcmp(key, "stage")) {
        x->have_stage = 1;
        return jstring(j, x->stage, sizeof x->stage);
    }
    if (!strcmp(key, "layout")) return jobject(j, layout_member, x, 1);
    if (!strcmp(key, "interface")) return jobject(j, iface_member, x, 1);
    if (!strcmp(key, "raw_access")) return jobject(j, raw_member, x, 1);
    return jskip(j, 1);
}

const char *shd_kind_name(int kind) {
    static const char *const names[SHD_KINDS] = {"vs", "ps", "cs", "ts", "ms"};
    return kind >= 0 && kind < SHD_KINDS ? names[kind] : "?";
}

int shd_parse(const char *json, size_t n, uint32_t schema, ShaderDesc *out, char *why, size_t cap) {
    memset(out, 0, sizeof *out);
    out->kind = -1;
    Ctx x = {.d = out};
    J j = {.begin = json, .p = json, .end = json + n};
    if (!jobject(&j, top_member, &x, 0)) {
        if (why && cap)
            snprintf(why, cap, "the description is not readable: %s (at byte %lu)", j.err ? j.err : "malformed",
                     (unsigned long)(j.p - j.begin));
        return 0;
    }
    ws(&j);
    if (j.p != j.end) {
        if (why && cap)
            snprintf(why, cap, "the description has text after its object (at byte %lu)", (unsigned long)(j.p - j.begin));
        return 0;
    }
    if (out->schema != schema) {
        if (why && cap) snprintf(why, cap, "the description is schema %u, the runtime reads schema %u", out->schema, schema);
        return 0;
    }
    if (!x.have_stage) {
        if (why && cap) snprintf(why, cap, "the description names no stage");
        return 0;
    }
    for (int k = 0; k < SHD_KINDS; k++)
        if (!strcmp(x.stage, shd_kind_name(k))) out->kind = k;
    if (out->kind < 0) {
        if (why && cap)
            snprintf(why, cap, "the description names the stage \"%s\", which is not one of vs, ps, cs, ts, ms", x.stage);
        return 0;
    }
    return 1;
}

// ---- padding

int shd_pad_set(const ShaderDesc *vs, const ShaderDesc *ps, ShdPad *out, int max, char *why, size_t cap) {
    uint8_t have[SHD_MAX_LOCATIONS] = {0}, need[SHD_MAX_LOCATIONS] = {0};
    uint8_t type[SHD_MAX_LOCATIONS][4];
    memset(type, 0, sizeof type);
#define REFUSE(...)                                                                                                    \
    do {                                                                                                               \
        if (why && cap) snprintf(why, cap, __VA_ARGS__);                                                               \
        return -1;                                                                                                     \
    } while (0)
    for (uint32_t i = 0; i < vs->noutputs; i++) {
        const ShdIo *o = &vs->outputs[i];
        if (o->component > 3 || o->components < 1 || o->component + o->components > 4)
            REFUSE("vertex output %d has components %u..%u", o->location, o->component, o->component + o->components);
        for (int k = 0; k < o->array_length; k++) {
            int l = o->location + k;
            if (l < 0 || l >= SHD_MAX_LOCATIONS) REFUSE("vertex output location %d is past the %d the builder reads", l, SHD_MAX_LOCATIONS);
            have[l] |= (uint8_t)(((1u << o->components) - 1) << o->component);
        }
    }
    for (uint32_t i = 0; i < ps->ninputs; i++) {
        const ShdIo *in = &ps->inputs[i];
        if (in->component > 3 || in->components < 1 || in->component + in->components > 4)
            REFUSE("pixel input %d has components %u..%u", in->location, in->component, in->component + in->components);
        for (int k = 0; k < in->array_length; k++) {
            int l = in->location + k;
            if (l < 0 || l >= SHD_MAX_LOCATIONS) REFUSE("pixel input location %d is past the %d the builder reads", l, SHD_MAX_LOCATIONS);
            uint8_t words = (uint8_t)(((1u << in->components) - 1) << in->component);
            if ((words & ~have[l]) && in->bits != 32)
                REFUSE("pixel input %d is %u bits wide: only 32-bit variables are padded", l, in->bits);
            for (int c = 0; c < 4; c++)
                if (words >> c & 1) type[l][c] = in->type;
            need[l] |= words;
        }
    }
    int n = 0;
    for (int l = 0; l < SHD_MAX_LOCATIONS; l++) {
        uint8_t miss = (uint8_t)(need[l] & ~have[l]);
        for (int c = 0; c < 4;) {
            if (!(miss >> c & 1)) {
                c++;
                continue;
            }
            int e = c + 1;
            while (e < 4 && (miss >> e & 1) && type[l][e] == type[l][c]) e++;
            if (n >= max) REFUSE("the pixel module reads more than %d vertex outputs the vertex module does not write", max);
            out[n++] = (ShdPad){(uint16_t)l, (uint8_t)c, (uint8_t)(e - c), type[l][c], 0};
            c = e;
        }
    }
#undef REFUSE
    return n;
}

// ---- the mesh link
//
// A mesh module's outputs are arrays over its vertices or primitives (xlate/host's mesh emission: one float per vertex
// for each component the job's map names, at its location and component). The outer array is the rate, not a run of
// locations: a float[160] at location 1 takes location 1 alone. A pixel input reads its words per vertex unless it is
// marked per primitive; Vulkan matches the two sides word by word, so the rule checks rate, type and width per word.

static const char *io_type_name(uint8_t type) { return type == SHD_SINT ? "sint" : type == SHD_UINT ? "uint" : "float"; }

static const char *rate_name(int per_primitive) { return per_primitive ? "primitive" : "vertex"; }

static int first_word(uint8_t words) {
    int c = 0;
    while (c < 3 && !(words >> c & 1)) c++;
    return c;
}

int shd_mesh_link(const ShaderDesc *ms, const ShaderDesc *ps, ShdPad *missing, int max, char *why, size_t cap) {
    uint8_t vert[SHD_MAX_LOCATIONS] = {0}, prim[SHD_MAX_LOCATIONS] = {0}, need[SHD_MAX_LOCATIONS] = {0};
    uint8_t otype[SHD_MAX_LOCATIONS][4], obits[SHD_MAX_LOCATIONS][4], itype[SHD_MAX_LOCATIONS][4];
    uint8_t irate[SHD_MAX_LOCATIONS][4];
    memset(otype, 0, sizeof otype);
    memset(obits, 0, sizeof obits);
    memset(itype, 0, sizeof itype);
    memset(irate, 0, sizeof irate);
#define REFUSE(...)                                                                                                    \
    do {                                                                                                               \
        if (why && cap) snprintf(why, cap, __VA_ARGS__);                                                               \
        return -1;                                                                                                     \
    } while (0)
#define OLDER(side, at, what)                                                                                          \
    REFUSE("%s at location %d does not give its %s: its description is older than the mesh link rule, translate it "  \
           "again", side, at, what)
    for (uint32_t i = 0; i < ms->noutputs; i++) {
        const ShdIo *o = &ms->outputs[i];
        if (o->array_depth < 0) OLDER("mesh output", o->location, "array depth");
        if (o->per_primitive < 0) OLDER("mesh output", o->location, "rate");
        if (o->array_depth != 1)
            REFUSE("shape: mesh output at location %d has %d array levels: the builder links one array per %s around a "
                   "scalar or a vector", o->location, o->array_depth, rate_name(o->per_primitive));
        if (o->component > 3 || o->components < 1 || o->component + o->components > 4)
            REFUSE("mesh output %d has components %u..%u", o->location, o->component, o->component + o->components);
        int l = o->location;
        if (l < 0 || l >= SHD_MAX_LOCATIONS) REFUSE("mesh output location %d is past the %d the builder reads", l, SHD_MAX_LOCATIONS);
        uint8_t words = (uint8_t)(((1u << o->components) - 1) << o->component);
        if (words & (vert[l] | prim[l]))
            REFUSE("mesh outputs overlap at location %d component %d", l, first_word((uint8_t)(words & (vert[l] | prim[l]))));
        if (o->per_primitive) prim[l] |= words;
        else vert[l] |= words;
        for (int c = 0; c < 4; c++)
            if (words >> c & 1) otype[l][c] = o->type, obits[l][c] = o->bits;
    }
    for (uint32_t i = 0; i < ps->ninputs; i++) {
        const ShdIo *in = &ps->inputs[i];
        if (in->array_depth < 0) OLDER("pixel input", in->location, "array depth");
        if (in->per_primitive < 0) OLDER("pixel input", in->location, "rate");
        if (in->array_depth > 1)
            REFUSE("shape: pixel input at location %d has %d array levels", in->location, in->array_depth);
        if (in->component > 3 || in->components < 1 || in->component + in->components > 4)
            REFUSE("pixel input %d has components %u..%u", in->location, in->component, in->component + in->components);
        uint8_t words = (uint8_t)(((1u << in->components) - 1) << in->component);
        for (int k = 0; k < in->array_length; k++) {
            int l = in->location + k;
            if (l < 0 || l >= SHD_MAX_LOCATIONS) REFUSE("pixel input location %d is past the %d the builder reads", l, SHD_MAX_LOCATIONS);
            uint8_t same = in->per_primitive ? prim[l] : vert[l], other = in->per_primitive ? vert[l] : prim[l];
            if (words & other)
                REFUSE("rate: pixel input at location %d component %d is read per %s, the mesh stage writes it per %s", l,
                       first_word((uint8_t)(words & other)), rate_name(in->per_primitive), rate_name(!in->per_primitive));
            if (in->array_length > 1 && (words & same))
                REFUSE("shape: pixel input at location %d is an array of %d over locations the mesh stage writes one at a "
                       "time", in->location, in->array_length);
            for (int c = 0; c < 4; c++)
                if ((words & same) >> c & 1 && (otype[l][c] != in->type || obits[l][c] != in->bits))
                    REFUSE("type: pixel input at location %d component %d is %s%u, the mesh stage writes %s%u", l, c,
                           io_type_name(in->type), in->bits, io_type_name(otype[l][c]), obits[l][c]);
            if ((words & ~(vert[l] | prim[l])) && in->bits != 32)
                REFUSE("pixel input %d is %u bits wide: only 32-bit variables are padded", l, in->bits);
            for (int c = 0; c < 4; c++)
                if (words >> c & 1) itype[l][c] = in->type, irate[l][c] = in->per_primitive ? 1 : 0;
            need[l] |= words;
        }
    }
    int n = 0;
    for (int l = 0; l < SHD_MAX_LOCATIONS; l++) {
        uint8_t miss = (uint8_t)(need[l] & ~(vert[l] | prim[l]));
        for (int c = 0; c < 4;) {
            if (!(miss >> c & 1)) {
                c++;
                continue;
            }
            // a run is one variable: it never mixes two types or two rates (a pad is an array over vertices or over
            // primitives, of one scalar type)
            int e = c + 1;
            while (e < 4 && (miss >> e & 1) && itype[l][e] == itype[l][c] && irate[l][e] == irate[l][c]) e++;
            if (n >= max) REFUSE("the pixel module reads more than %d ranges the mesh module does not write", max);
            missing[n++] = (ShdPad){(uint16_t)l, (uint8_t)c, (uint8_t)(e - c), itype[l][c], irate[l][c]};
            c = e;
        }
    }
#undef OLDER
#undef REFUSE
    return n;
}

void shd_pad_text(const ShdPad *pad, int n, char *out, size_t cap) {
    size_t at = 0;
    if (cap) out[0] = 0;
    for (int i = 0; i < n; i++) {
        int w = snprintf(out + at, at < cap ? cap - at : 0, pad[i].per_primitive ? "%u:%u:%u:%u:p;" : "%u:%u:%u:%u;",
                         pad[i].location, pad[i].component, pad[i].components, pad[i].type);
        if (w < 0) break;
        at += (size_t)w;
    }
}

// ---- bindings
//
// The second reader, for the members that say where a module's resources are (oot2/shader_desc.h, ShaderBinds). It uses
// the same scanner, but it is strict where the first is lenient: every member it reads must be there, of its type and
// within the range of the field it fills, because the replay binds from what it reads and a default would bind the
// wrong thing silently. Members it does not know are skipped.

typedef struct BCtx {
    ShaderBinds *b;
    uint32_t want_schema;
    char stage[8];
    uint32_t have;          // HAVE_* bits of the top-level members read
    uint32_t seen;          // the members of the element being read, one bit per name of its list
    char at[48];            // "textures[3]." while an element is read, "" at the top
    char msg[200];
    // the element being read
    ShdCbuf cb;
    ShdSbuf sb;
    ShdTex tex;
    ShdImg img;
    ShdVin vin;
    int sb_module_writes, img_store, img_nonwritable, vin_array, field_read;
    int tex_kind, tex_type_is_sampler;
    int raw_page_table, raw_physical;
    int count;              // the support fields or the local size entries read
} BCtx;

enum {
    HAVE_SCHEMA = 1 << 0, HAVE_STAGE = 1 << 1, HAVE_FITS = 1 << 2, HAVE_RAW_PT = 1 << 3, HAVE_RAW_PP = 1 << 4,
    HAVE_SUPPORT = 1 << 5, HAVE_CB = 1 << 6, HAVE_SB = 1 << 7, HAVE_TEX = 1 << 8, HAVE_IMG = 1 << 9, HAVE_VIN = 1 << 10,
    HAVE_LOCAL = 1 << 11
};

// The members of each kind of element, in the bit order of BCtx.seen. All of them are required.
static const char *const CB_NAMES[] = {"set", "binding", "slot"};
static const char *const SB_NAMES[] = {"set", "binding", "ordinal", "cb_slot", "address_byte_offset", "write", "module_writes"};
static const char *const TEX_NAMES[] = {"set", "binding", "via", "push_slot", "push_offset", "type", "shadow", "kind", "array_length", "c2_slot"};
static const char *const IMG_NAMES[] = {"set", "binding", "type", "c2_slot", "image_store", "non_writable"};
static const char *const VIN_NAMES[] = {"location", "component", "components", "type"};
#define COUNT(a) ((int)(sizeof(a) / sizeof((a)[0])))

static int bfail(J *j, BCtx *x, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(x->msg, sizeof x->msg, fmt, ap);
    va_end(ap);
    j->err = x->msg;
    return 0;
}

static int name_index(const char *const *names, int n, const char *key) {
    for (int i = 0; i < n; i++)
        if (!strcmp(names[i], key)) return i;
    return -1;
}

static int peek(J *j, char c) {
    ws(j);
    return j->p < j->end && *j->p == c;
}

// A number of the member `name` within lo..hi.
static int bnum(J *j, BCtx *x, const char *name, long long lo, long long hi, long long *v) {
    ws(j);
    if (j->p >= j->end || !((*j->p >= '0' && *j->p <= '9') || *j->p == '-'))
        return bfail(j, x, "%s%s is not a number", x->at, name);
    if (!jint(j, v)) return 0;
    if (*v < lo || *v > hi) return bfail(j, x, "%s%s is %lld, outside %lld to %lld", x->at, name, *v, lo, hi);
    return 1;
}

static int bbool(J *j, BCtx *x, const char *name, int *v) {
    ws(j);
    if (j->p >= j->end || (*j->p != 't' && *j->p != 'f')) return bfail(j, x, "%s%s is not true or false", x->at, name);
    return jbool(j, v);
}

static int bstr(J *j, BCtx *x, const char *name, char *out, size_t cap) {
    ws(j);
    if (j->p >= j->end || *j->p != '"') return bfail(j, x, "%s%s is not a string", x->at, name);
    return jstring(j, out, cap);
}

// What starts and ends an element of any list: the cap, a fresh element, and every member present.
static int belem_begin(J *j, BCtx *x, const char *list, uint32_t count, uint32_t cap) {
    if (count >= cap) return bfail(j, x, "%s lists more than the %u entries the replay reads", list, cap);
    snprintf(x->at, sizeof x->at, "%s[%u].", list, count);
    x->seen = 0;
    return 1;
}

static int belem_end(J *j, BCtx *x, const char *const *names, int n) {
    for (int i = 0; i < n; i++)
        if (!(x->seen >> i & 1)) return bfail(j, x, "%s%s is missing", x->at, names[i]);
    return 1;
}

// A texture type of the description as the view the replay makes for it (bindless.h BV_*). Multisampled types are the 2D
// types of Vulkan; a buffer texture has no view; a bare sampler has none either. "unknown" is what the writer prints for
// a type it cannot name, and is refused with the rest.
static int view_of(const char *t, int allow_sampler, int *view) {
    static const struct {
        const char *name;
        int view;
    } map[] = {{"1d", BV_1D},   {"1darray", BV_1D_ARRAY}, {"2d", BV_2D},     {"2darray", BV_2D_ARRAY},
               {"2dms", BV_2D}, {"2dmsarray", BV_2D_ARRAY}, {"3d", BV_3D},   {"cube", BV_CUBE},
               {"cubearray", BV_CUBE_ARRAY}, {"buffer", BV_NONE}};
    for (int i = 0; i < COUNT(map); i++)
        if (!strcmp(map[i].name, t)) return *view = map[i].view, 1;
    if (allow_sampler && !strcmp(t, "sampler")) return *view = BV_NONE, 1;
    return 0;
}

// ---- constant banks

static int cb_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    long long v;
    int i = name_index(CB_NAMES, COUNT(CB_NAMES), key);
    if (i < 0) return jskip(j, 4);
    switch (i) {
    case 0: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->cb.set = (uint8_t)v; break;
    case 1: if (!bnum(j, x, key, 0, 65535, &v)) return 0; x->cb.binding = (uint16_t)v; break;
    default: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->cb.slot = (uint8_t)v; break;
    }
    x->seen |= 1u << i;
    return 1;
}

static int cb_element(J *j, void *c) {
    BCtx *x = c;
    if (!belem_begin(j, x, "cbufs", x->b->ncb, SHD_MAX_CBUFS)) return 0;
    memset(&x->cb, 0, sizeof x->cb);
    if (!jobject(j, cb_member, x, 3) || !belem_end(j, x, CB_NAMES, COUNT(CB_NAMES))) return 0;
    x->b->cb[x->b->ncb++] = x->cb;
    return 1;
}

// ---- storage buffers

static int sb_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    long long v;
    int b;
    int i = name_index(SB_NAMES, COUNT(SB_NAMES), key);
    if (i < 0) return jskip(j, 4);
    switch (i) {
    case 0: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->sb.set = (uint8_t)v; break;
    case 1: if (!bnum(j, x, key, 0, 65535, &v)) return 0; x->sb.binding = (uint16_t)v; break;
    case 2: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->sb.ordinal = (uint8_t)v; break;
    case 3: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->sb.cb_slot = (uint8_t)v; break;
    case 4: if (!bnum(j, x, key, 0, 65535, &v)) return 0; x->sb.address_byte_offset = (uint16_t)v; break;
    case 5: if (!bbool(j, x, key, &b)) return 0; x->sb.write = (uint8_t)b; break;
    default: if (!bbool(j, x, key, &x->sb_module_writes)) return 0; break;
    }
    x->seen |= 1u << i;
    return 1;
}

static int sb_element(J *j, void *c) {
    BCtx *x = c;
    if (!belem_begin(j, x, "sbufs", x->b->nsb, SHD_MAX_SBUFS)) return 0;
    memset(&x->sb, 0, sizeof x->sb);
    x->sb_module_writes = 0;
    if (!jobject(j, sb_member, x, 3) || !belem_end(j, x, SB_NAMES, COUNT(SB_NAMES))) return 0;
    if (x->sb_module_writes) x->sb.write = 1;
    x->b->sb[x->b->nsb++] = x->sb;
    return 1;
}

// ---- textures and storage images

static int tex_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    long long v;
    int b;
    char s[24];
    int i = name_index(TEX_NAMES, COUNT(TEX_NAMES), key);
    if (i < 0) return jskip(j, 4);
    switch (i) {
    case 0: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->tex.set = (uint8_t)v; break;
    case 1: if (!bnum(j, x, key, 0, 65535, &v)) return 0; x->tex.binding = (uint16_t)v; break;
    case 2:
        if (!bstr(j, x, key, s, sizeof s)) return 0;
        if (!strcmp(s, "push_slot")) x->tex.via = SHD_VIA_PUSH_SLOT;
        else if (!strcmp(s, "pool")) x->tex.via = SHD_VIA_POOL;
        else if (!strcmp(s, "descriptor")) x->tex.via = SHD_VIA_DESCRIPTOR;
        else return bfail(j, x, "%svia \"%s\" is not push_slot, pool or descriptor", x->at, s);
        break;
    case 3: if (!bnum(j, x, key, -1, 32767, &v)) return 0; x->tex.push_slot = (int16_t)v; break;
    case 4: if (!bnum(j, x, key, -1, 32767, &v)) return 0; x->tex.push_offset = (int16_t)v; break;
    case 5:
        if (!bstr(j, x, key, s, sizeof s)) return 0;
        if (!view_of(s, 1, &b)) return bfail(j, x, "%stype \"%s\" is not a texture type the replay knows", x->at, s);
        x->tex.view = (int8_t)b;
        x->tex_type_is_sampler = !strcmp(s, "sampler");
        break;
    case 6: if (!bbool(j, x, key, &b)) return 0; x->tex.shadow = (uint8_t)b; break;
    case 7:
        if (!bstr(j, x, key, s, sizeof s)) return 0;
        if (!strcmp(s, "combined")) x->tex_kind = SHD_TEX_COMBINED;
        else if (!strcmp(s, "separate_texture")) x->tex_kind = SHD_TEX_SEPARATE;
        else if (!strcmp(s, "sampler")) x->tex_kind = SHD_TEX_SAMPLER;
        else return bfail(j, x, "%skind \"%s\" is not combined, separate_texture or sampler", x->at, s);
        break;
    case 8: if (!bnum(j, x, key, 0, 65535, &v)) return 0; x->tex.array_length = (uint16_t)v; break;
    default: if (!bnum(j, x, key, -1, 32767, &v)) return 0; x->tex.c2_slot = (int16_t)v; break;
    }
    x->seen |= 1u << i;
    return 1;
}

static int tex_element(J *j, void *c) {
    BCtx *x = c;
    if (!belem_begin(j, x, "textures", x->b->ntex, SHD_MAX_TEXS)) return 0;
    memset(&x->tex, 0, sizeof x->tex);
    x->tex_kind = x->tex_type_is_sampler = 0;
    if (!jobject(j, tex_member, x, 3) || !belem_end(j, x, TEX_NAMES, COUNT(TEX_NAMES))) return 0;
    x->tex.kind = (int8_t)x->tex_kind;
    if ((x->tex_kind == SHD_TEX_SAMPLER) != (x->tex_type_is_sampler != 0))
        return bfail(j, x, "%skind and type disagree on whether the entry is a bare sampler", x->at);
    if (x->tex.via == SHD_VIA_PUSH_SLOT && (x->tex.push_slot < 0 || x->tex.push_offset < 0))
        return bfail(j, x, "%sreads through a push slot but names none", x->at);
    x->b->tex[x->b->ntex++] = x->tex;
    return 1;
}

static int img_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    long long v;
    int b;
    char s[24];
    int i = name_index(IMG_NAMES, COUNT(IMG_NAMES), key);
    if (i < 0) return jskip(j, 4);
    switch (i) {
    case 0: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->img.set = (uint8_t)v; break;
    case 1: if (!bnum(j, x, key, 0, 65535, &v)) return 0; x->img.binding = (uint16_t)v; break;
    case 2:
        if (!bstr(j, x, key, s, sizeof s)) return 0;
        if (!view_of(s, 0, &b)) return bfail(j, x, "%stype \"%s\" is not an image type the replay knows", x->at, s);
        x->img.view = (int8_t)b;
        break;
    case 3: if (!bnum(j, x, key, -1, 32767, &v)) return 0; x->img.c2_slot = (int16_t)v; break;
    case 4: if (!bbool(j, x, key, &x->img_store)) return 0; break;
    default: if (!bbool(j, x, key, &x->img_nonwritable)) return 0; break;
    }
    x->seen |= 1u << i;
    return 1;
}

static int img_element(J *j, void *c) {
    BCtx *x = c;
    if (!belem_begin(j, x, "images", x->b->nimg, SHD_MAX_IMGS)) return 0;
    memset(&x->img, 0, sizeof x->img);
    x->img_store = x->img_nonwritable = 0;
    if (!jobject(j, img_member, x, 3) || !belem_end(j, x, IMG_NAMES, COUNT(IMG_NAMES))) return 0;
    x->img.write = (uint8_t)(x->img_store || !x->img_nonwritable);
    x->b->img[x->b->nimg++] = x->img;
    return 1;
}

// ---- vertex inputs

static int vin_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    long long v;
    char s[24];
    if (!strcmp(key, "array_length")) {
        if (!bnum(j, x, key, 1, SHD_MAX_VINS, &v)) return 0;
        x->vin_array = (int)v;
        return 1;
    }
    int i = name_index(VIN_NAMES, COUNT(VIN_NAMES), key);
    if (i < 0) return jskip(j, 4);
    switch (i) {
    case 0: if (!bnum(j, x, key, 0, 255, &v)) return 0; x->vin.location = (uint8_t)v; break;
    case 1: if (!bnum(j, x, key, 0, 3, &v)) return 0; x->vin.component = (uint8_t)v; break;
    case 2: if (!bnum(j, x, key, 1, 4, &v)) return 0; x->vin.components = (uint8_t)v; break;
    default:
        if (!bstr(j, x, key, s, sizeof s)) return 0;
        if (!strcmp(s, "float32")) x->vin.type = SHD_FLOAT;
        else if (!strcmp(s, "sint32")) x->vin.type = SHD_SINT;
        else if (!strcmp(s, "uint32")) x->vin.type = SHD_UINT;
        else return bfail(j, x, "%stype \"%s\" is not float32, sint32 or uint32", x->at, s);
        break;
    }
    x->seen |= 1u << i;
    return 1;
}

static int vin_element(J *j, void *c) {
    BCtx *x = c;
    if (!belem_begin(j, x, "vertex_inputs", x->b->nvin, SHD_MAX_VINS)) return 0;
    memset(&x->vin, 0, sizeof x->vin);
    x->vin_array = 1;
    if (!jobject(j, vin_member, x, 3) || !belem_end(j, x, VIN_NAMES, COUNT(VIN_NAMES))) return 0;
    if (x->vin.component + x->vin.components > 4)
        return bfail(j, x, "%scomponents %u..%u pass the fourth", x->at, x->vin.component, x->vin.component + x->vin.components);
    // An array takes one location per element, each its own entry.
    for (int k = 0; k < x->vin_array; k++) {
        if (x->b->nvin >= SHD_MAX_VINS)
            return bfail(j, x, "vertex_inputs lists more than the %u entries the replay reads", SHD_MAX_VINS);
        if (x->vin.location + k > 255) return bfail(j, x, "%sarray reaches location %d, past 255", x->at, x->vin.location + k);
        ShdVin e = x->vin;
        e.location = (uint8_t)(x->vin.location + k);
        x->b->vin[x->b->nvin++] = e;
    }
    return 1;
}

// ---- the other members

static int field_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    if (strcmp(key, "read")) return jskip(j, 4);
    if (!bbool(j, x, key, &x->field_read)) return 0;
    x->seen |= 1;
    return 1;
}

static int field_element(J *j, void *c) {
    BCtx *x = c;
    snprintf(x->at, sizeof x->at, "support_buffer.fields[%d].", x->count++);
    x->seen = 0;
    x->field_read = 0;
    if (!jobject(j, field_member, x, 4)) return 0;
    if (!(x->seen & 1)) return bfail(j, x, "%sread is missing", x->at);
    x->b->support_read |= (uint8_t)(x->field_read != 0);
    return 1;
}

// An array member of the top level (null where the writer says "none" and `nullable`), counted as present only when it is
// an array.
static int blist(J *j, BCtx *x, const char *name, ElementFn element, uint32_t have, int nullable) {
    ws(j);
    if (nullable && jword(j, "null")) return 1;
    if (!peek(j, '[')) return bfail(j, x, "%s is not an array", name);
    x->have |= have;
    x->at[0] = 0;
    return jarray(j, element, x, 1);
}

static int support_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    if (strcmp(key, "fields")) return jskip(j, 2);
    x->count = 0;
    return blist(j, x, "support_buffer.fields", field_element, HAVE_SUPPORT, 0);
}

static int blayout_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    if (strcmp(key, "fits")) return jskip(j, 2);
    x->have |= HAVE_FITS;
    x->at[0] = 0;
    return bbool(j, x, "layout.fits", &x->b->fits);
}

static int braw_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    x->at[0] = 0;
    if (!strcmp(key, "page_table")) {
        x->have |= HAVE_RAW_PT;
        return bbool(j, x, "raw_access.page_table", &x->raw_page_table);
    }
    if (!strcmp(key, "physical_pointers")) {
        x->have |= HAVE_RAW_PP;
        return bbool(j, x, "raw_access.physical_pointers", &x->raw_physical);
    }
    return jskip(j, 2);
}

static int local_element(J *j, void *c) {
    BCtx *x = c;
    long long v;
    x->at[0] = 0;
    if (x->count >= 3) return bfail(j, x, "compute.local_size has more than 3 entries");
    if (!bnum(j, x, "compute.local_size entry", 1, 65535, &v)) return 0;
    x->b->local_size[x->count++] = (uint32_t)v;
    return 1;
}

static int compute_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    if (strcmp(key, "local_size")) return jskip(j, 2);
    x->count = 0;
    if (!peek(j, '[')) return bfail(j, x, "compute.local_size is not an array");
    if (!jarray(j, local_element, x, 2)) return 0;
    if (x->count != 3) return bfail(j, x, "compute.local_size has %d entries, not 3", x->count);
    x->have |= HAVE_LOCAL;
    return 1;
}

static int btop_member(J *j, const char *key, void *c) {
    BCtx *x = c;
    long long v;
    x->at[0] = 0;
    if (!strcmp(key, "schema")) {
        if (!bnum(j, x, key, 0, 0xffffffffLL, &v)) return 0;
        x->have |= HAVE_SCHEMA;
        if ((uint32_t)v != x->want_schema)
            return bfail(j, x, "the description is schema %u, the runtime reads schema %u", (uint32_t)v, x->want_schema);
        return 1;
    }
    if (!strcmp(key, "stage")) {
        x->have |= HAVE_STAGE;
        return bstr(j, x, key, x->stage, sizeof x->stage);
    }
    if (!strcmp(key, "layout")) return jobject(j, blayout_member, x, 1);
    if (!strcmp(key, "raw_access")) return jobject(j, braw_member, x, 1);
    if (!strcmp(key, "support_buffer")) return jobject(j, support_member, x, 1);
    if (!strcmp(key, "cbufs")) return blist(j, x, key, cb_element, HAVE_CB, 0);
    if (!strcmp(key, "sbufs")) return blist(j, x, key, sb_element, HAVE_SB, 0);
    if (!strcmp(key, "textures")) return blist(j, x, key, tex_element, HAVE_TEX, 0);
    if (!strcmp(key, "images")) return blist(j, x, key, img_element, HAVE_IMG, 0);
    if (!strcmp(key, "vertex_inputs")) return blist(j, x, key, vin_element, HAVE_VIN, 1);
    if (!strcmp(key, "compute")) {
        ws(j);
        if (jword(j, "null")) return 1;
        return jobject(j, compute_member, x, 1);
    }
    return jskip(j, 1);
}

int shd_parse_binds(const char *json, size_t n, uint32_t schema, ShaderBinds *out, char *why, size_t cap) {
    static const struct {
        uint32_t bit;
        const char *name;
    } required[] = {{HAVE_SCHEMA, "schema"},   {HAVE_STAGE, "stage"},         {HAVE_FITS, "layout.fits"},
                    {HAVE_RAW_PT, "raw_access.page_table"}, {HAVE_RAW_PP, "raw_access.physical_pointers"},
                    {HAVE_SUPPORT, "support_buffer.fields"}, {HAVE_CB, "cbufs"}, {HAVE_SB, "sbufs"},
                    {HAVE_TEX, "textures"},    {HAVE_IMG, "images"}};
    memset(out, 0, sizeof *out);
    out->kind = -1;
    BCtx x;
    memset(&x, 0, sizeof x);
    x.b = out;
    x.want_schema = schema;
    J j = {.begin = json, .p = json, .end = json + n};
#define REFUSE(...)                                                                                                    \
    do {                                                                                                               \
        if (why && cap) snprintf(why, cap, __VA_ARGS__);                                                               \
        return 0;                                                                                                      \
    } while (0)
    if (!jobject(&j, btop_member, &x, 0))
        REFUSE("the bindings of the description are not readable: %s (at byte %lu)", j.err ? j.err : "malformed",
               (unsigned long)(j.p - j.begin));
    ws(&j);
    if (j.p != j.end) REFUSE("the description has text after its object (at byte %lu)", (unsigned long)(j.p - j.begin));
    for (int i = 0; i < COUNT(required); i++)
        if (!(x.have & required[i].bit)) REFUSE("the description lacks %s", required[i].name);
    for (int k = 0; k < SHD_KINDS; k++)
        if (!strcmp(x.stage, shd_kind_name(k))) out->kind = k;
    if (out->kind < 0) REFUSE("the description names the stage \"%s\", which is not one of vs, ps, cs, ts, ms", x.stage);
    if (out->kind == SHD_VS && !(x.have & HAVE_VIN)) REFUSE("the description of a vertex module lacks vertex_inputs");
    if (out->kind == SHD_CS && !(x.have & HAVE_LOCAL)) REFUSE("the description of a compute module lacks compute.local_size");
    out->raw = (uint8_t)(x.raw_page_table || x.raw_physical);
#undef REFUSE
    return 1;
}
