// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The program keys of the movie chain's stand-in (oot2/vk_movie_blit.h): the file OOT2_MOVIE_BLIT_KEYS names is read once,
// every line checked, and the role of a compute program looked up in it. No key of the game's is written here; the file
// is the person's own and stays outside the repository, as the player's names file does.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "oot2/vk_movie_blit.h"
#include "plat.h"

#define MAX_KEYS 64
#define KEY_BYTES 25         // "0123456789abcdef.01234567": a stage key as the replay names it
#define FILE_MAX (64u << 10)
// What a fatal of the start-up says last: the setting can be saved in the settings file, and the game stops before its menu.
#define SWITCH_OFF "; to switch it off, set OOT2_TEST_MOVIE_BLIT=0 or remove the test_movie_blit line from the settings file (oot2.ini)"

static struct {
    char key[KEY_BYTES + 1];
    VkmbRole role;
} g_keys[MAX_KEYS];
static int g_n;
static int g_loaded;   // 1 once the table is complete (atomic: the replay thread reads it, the device's open may write it)
static PlatMutex g_lock = PLAT_MUTEX_INIT;

static const char *const ROLE_WORDS[] = {NULL, "keep", "skip", "blit"};

__attribute__((noreturn, format(printf, 3, 4))) static void bad(const char *path, int line, const char *fmt, ...) {
    char what[300];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(what, sizeof what, fmt, ap);
    va_end(ap);
    // Every mistake in the file also stops the game at start-up (vkmb_open), before its menu: the way out is said last.
    if (line) rt_fatal("gfx: vulkan: test_movie_blit is on and its key file %s, line %d: %s" SWITCH_OFF, path, line, what);
    rt_fatal("gfx: vulkan: test_movie_blit is on and its key file %s: %s" SWITCH_OFF, path, what);
}

static int is_key(const char *s) {
    if (strlen(s) != KEY_BYTES || s[16] != '.') return 0;
    for (int i = 0; i < KEY_BYTES; i++) {
        if (i == 16) continue;
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return 0;
    }
    return 1;
}

static int is_blank(char c) { return c == ' ' || c == '\t' || c == '\r'; }

// One line of the file: nothing, or a role word and a key. Returns the role, or VKMB_NONE for a line with nothing in it.
static VkmbRole parse_line(const char *path, int no, char *line, char **key) {
    char *hash = strchr(line, '#');
    if (hash) *hash = 0;
    char *p = line;
    while (is_blank(*p)) p++;
    if (!*p) return VKMB_NONE;
    char *word = p;
    while (*p && !is_blank(*p)) p++;
    if (*p) *p++ = 0;
    while (is_blank(*p)) p++;
    if (!*p) bad(path, no, "\"%s\" has no key after it", word);
    *key = p;
    while (*p && !is_blank(*p)) p++;
    char *end = p;
    while (is_blank(*p)) p++;
    if (*p) bad(path, no, "more than a role and a key (a comment starts with #)");
    *end = 0;
    for (int r = VKMB_KEEP; r <= VKMB_BLIT; r++)
        if (!strcmp(word, ROLE_WORDS[r])) {
            if (!is_key(*key)) bad(path, no, "\"%s\" is not a stage key (16 hex digits, a dot, 8 hex digits, lower case)", *key);
            return (VkmbRole)r;
        }
    bad(path, no, "\"%s\" is not a role (keep, skip or blit)", word);
}

static void parse(const char *path, char *text) {
    int counts[4] = {0};
    int no = 0;
    for (char *line = text; line;) {
        char *next = strchr(line, '\n');
        if (next) *next++ = 0;
        no++;
        char *key;
        VkmbRole role = parse_line(path, no, line, &key);
        line = next;
        if (role == VKMB_NONE) continue;
        for (int i = 0; i < g_n; i++)
            if (!strcmp(g_keys[i].key, key)) {
                if (g_keys[i].role == role) bad(path, no, "the key %s is named twice", key);
                bad(path, no, "the key %s is in two roles (%s and %s)", key, ROLE_WORDS[g_keys[i].role], ROLE_WORDS[role]);
            }
        if (g_n == MAX_KEYS) bad(path, no, "more than %d keys", MAX_KEYS);
        memcpy(g_keys[g_n].key, key, KEY_BYTES + 1);
        g_keys[g_n++].role = role;
        counts[role]++;
    }
    if (!counts[VKMB_KEEP] || !counts[VKMB_BLIT])
        bad(path, 0, "it needs a keep line and a blit line (it has %d and %d)", counts[VKMB_KEEP], counts[VKMB_BLIT]);
    rt_log("gfx: vulkan: test_movie_blit: %d keep, %d skip and %d blit program keys read from %s", counts[VKMB_KEEP],
           counts[VKMB_SKIP], counts[VKMB_BLIT], path);
}

static void load(void) {
    const char *path = getenv("OOT2_MOVIE_BLIT_KEYS");
    if (!path || !*path)
        rt_fatal("gfx: vulkan: test_movie_blit is on and OOT2_MOVIE_BLIT_KEYS is not set: it names the file of the movie "
                 "chain's program keys, kept outside the repository (oot2/vk_movie_blit.h)" SWITCH_OFF);
    FILE *f = fopen(path, "rb");
    if (!f) rt_fatal("gfx: vulkan: test_movie_blit is on and its key file %s cannot be opened" SWITCH_OFF, path);
    static char buf[FILE_MAX + 1];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    int failed = ferror(f), more = !failed && fgetc(f) != EOF;
    fclose(f);
    if (failed) bad(path, 0, "it cannot be read");
    if (more) bad(path, 0, "it is longer than %u bytes", FILE_MAX);
    buf[n] = 0;
    if (strlen(buf) != n) bad(path, 0, "it holds a NUL byte");
    parse(path, buf);
}

static void ensure_loaded(void) {
    if (__atomic_load_n(&g_loaded, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&g_lock);
    if (!g_loaded) {
        load();
        __atomic_store_n(&g_loaded, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&g_lock);
}

void vkmb_open(void) {
    if (!rt_setting(RT_SET_test_movie_blit)) return;
    ensure_loaded();
    rt_log("gfx: vulkan: TEST SWITCH test_movie_blit IS ON: the movie's enhancement chain is replaced by a plain resample "
           "of its first picture. This run is NOT a proof run and the pictures of the movies are not the console's");
}

VkmbRole vkmb_role(const char *key) {
    if (!rt_setting(RT_SET_test_movie_blit)) return VKMB_NONE;
    ensure_loaded();
    for (int i = 0; i < g_n; i++)
        if (!strcmp(g_keys[i].key, key)) return g_keys[i].role;
    return VKMB_NONE;
}
