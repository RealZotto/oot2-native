// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// nn::fs as the host answers it. A path is "<mount>:/<path>" and a mount name maps to: the RomFS pack (romfs.c,
// read-only); save data in <game folder>/save/<user>/, created on first use; temporary storage in <game folder>/temp/,
// emptied the first time it is mounted in a run; cache storage in <game folder>/cache/<index>/. Files and directories
// of the host mounts are host files (hostfile.c). Temporary and cache storage are written in place; save data is written
// to a shadow of each changed file and Commit publishes the shadows (savetxn.h), so a run that stops between two writes
// leaves the old save, never half of a new one; shadows not committed are dropped at Unmount, as the console drops
// uncommitted data. The SDK's own nn::fs code is bypassed for every function here: its handles would name service objects that do not exist.
//
// Handles are 64-bit values the guest keeps in its FileHandle and DirectoryHandle: RomFS files ROMFS_HANDLE_BASE +
// slot (fs.h), host files FILE_BASE + slot, directories DIR_BASE + slot, cache storage lists LIST_BASE + slot.
//
// Adapted from oot-pc-recomp runtime/fs_host.c and runtime/romfs.c at 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17
// (GPL-3.0-or-later; both files' own headers say they were adapted from MacNativeGames games/totk/runtime/fs_host.c and
// romfs.c at b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors): the mount
// table, the path to host path rule, the handle tables, the result of each call, the directory listing snapshot, the
// override thunks and their signatures, the cache storage calls. Changed here: the host layer is hostfile.c; the RomFS
// is a mount kind of the one table (several names may mount it, and the same name again answers success); save data
// is written to shadows that Commit publishes under a marker (savetxn.c), in place of the donor's working copy, atomic
// Commit and backups; the save folder is keyed by
// the user id the guest presents; DeleteDirectory, GetDirectoryEntryCount, CreateFile with an option, Commit with an
// option and MountSaveData with a user id are added; the forms that name another application are named fatals; a
// handle table that is full or a handle that is not open is a named fatal of kind fs; no trace, repeat-open note,
// probe, environment setting or statistics.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs.h"
#include "plat.h"
#include "savetxn.h"

#define FILE_BASE 0x200000000ull
#define DIR_BASE 0x300000000ull
#define LIST_BASE 0x400000000ull
#define NFILES 1024
#define NDIRS 256
#define NLISTS 16
#define NMOUNTS 16
#define GUEST_PATH 1024         // the longest guest path read, with its terminator (nn::fs stops at 768 bytes)
#define ROOT_PATH 1024          // a mount's root on the host
#define HOST_PATH 2048          // a root and a guest path joined
#define READ_CHUNK (64u << 20)  // the largest single host read or write

enum { M_FREE, M_ROM, M_SAVE, M_TEMP, M_CACHE };

typedef struct Mount { int kind; uint32_t id; char name[32]; char root[ROOT_PATH]; } Mount;
// h NULL: free. A save handle also keeps its file's host path (real) and the serial of the shadow it is open on (bound; 0:
// the file itself), and follows the shadow when the file gains or loses one (file_bound).
typedef struct File { HostFile *h; int mode; uint32_t mount; char *real; uint64_t bound; } File;
typedef struct Listed { char *name; int kind; uint64_t size; } Listed;
typedef struct Listing { Listed *e; int n, cap; } Listing;
typedef struct Dir { Listed *e; int n, next, mode, used; } Dir;           // the entries as they were at open; used 0: free
typedef struct CacheList { int index[64]; int count, next, used; } CacheList;

typedef struct DirEntry { char name[769]; uint8_t pad0[3]; int8_t type; uint8_t pad1[3]; int64_t size; } DirEntry;
_Static_assert(sizeof(DirEntry) == 0x310, "nn::fs::DirectoryEntry");
typedef struct CacheInfo { int32_t index; uint8_t reserved[0x1c]; } CacheInfo;
_Static_assert(sizeof(CacheInfo) == 0x20, "nn::fs::CacheStorageInfo");

static Mount mounts[NMOUNTS];
static File file_tab[NFILES];
static Dir dir_tab[NDIRS];
static CacheList list_tab[NLISTS];
static PlatMutex fs_lock = PLAT_MUTEX_INIT;   // the four tables above; never held across a host call
static uint32_t mount_serial;
static int temp_cleared;

// ---- small helpers
static uint32_t fs_result(int e) {
    switch (e) {
    case HF_OK: return 0;
    case HF_NOT_FOUND: return FS_R_PATH_NOT_FOUND;
    case HF_EXISTS: return FS_R_ALREADY_EXISTS;
    case HF_NOT_EMPTY: return FS_R_NOT_EMPTY;
    default: return FS_R_INVALID_ARGUMENT;
    }
}

// A guest string argument, or NULL for a null pointer.
static const char *str_arg(const Cpu *c, int i, char *buf, size_t n) {
    uint64_t a = arg_x(c, i);
    return a ? hos_str(a, buf, n) : NULL;
}

// A guest path argument into buf: 0, or FS_R_INVALID_PATH for a null or too long path.
static uint32_t guest_path(const Cpu *c, int i, char *buf, size_t cap) {
    uint64_t a = arg_x(c, i);
    if (!a) return FS_R_INVALID_PATH;
    hos_str(a, buf, cap);
    return strlen(buf) + 1 >= cap ? FS_R_INVALID_PATH : 0;
}

// A 16-byte user id that the guest passes by reference.
static int uid_arg(uint64_t addr, uint64_t uid[2]) { return addr && hos_in(uid, addr, 16); }

// <game folder>/<leaf>; a game folder that is missing or does not fit is a named fatal.
static void under_game(Cpu *c, char *out, size_t cap, const char *leaf) {
    const char *g = hos_game_folder();
    size_t n = g ? strlen(g) : 0;
    while (n && (g[n - 1] == '/' || g[n - 1] == '\\')) n--;
    if (!g || snprintf(out, cap, "%.*s/%s", (int)n, g, leaf) >= (int)cap)
        hos_gap(c, "fs", "the game folder is missing or its path is too long for %s", leaf);
}

// ---- host directories
static int make_dir(const char *p) {
    int e = hf_mkdir(p);
    return e == HF_OK || (e == HF_EXISTS && hf_kind(p) == HF_DIR);
}

// p and every missing parent; 1 when p is a directory afterwards.
static int make_dirs(const char *p) {
    char buf[HOST_PATH];
    if (snprintf(buf, sizeof buf, "%s", p) >= (int)sizeof buf) return 0;
    for (char *q = buf + 1; *q; q++)
        if ((*q == '/' || *q == '\\') && q[-1] != ':') {
            char ch = *q;
            *q = 0;
            make_dir(buf);
            *q = ch;
        }
    return make_dir(buf);
}

static int listing_add(const char *name, int kind, uint64_t size, void *ctx) {
    Listing *l = ctx;
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 16;
        l->e = realloc(l->e, (size_t)l->cap * sizeof *l->e);
        if (!l->e) rt_fatal("fs: out of memory for a directory listing");
    }
    char *copy = strdup(name);
    if (!copy) rt_fatal("fs: out of memory for a directory listing");
    l->e[l->n++] = (Listed){copy, kind, size};
    return 1;
}

static void listing_free(Listed *e, int n) {
    for (int i = 0; i < n; i++) free(e[i].name);
    free(e);
}

static int listed_cmp(const void *a, const void *b) { return strcmp(((const Listed *)a)->name, ((const Listed *)b)->name); }

// Remove a directory tree; a link is removed as a link, never followed. 0 on success, and when nothing is there.
static int remove_tree(const char *path) {
    int kind = hf_kind(path);
    if (kind == HF_MISSING) return 0;
    if (kind != HF_DIR) return hf_unlink(path) == HF_OK || (kind == HF_LINK && hf_rmdir(path) == HF_OK) ? 0 : -1;
    Listing l = {0};
    int bad = hf_list(path, listing_add, &l) != HF_OK;
    for (int i = 0; i < l.n; i++) {
        char child[HOST_PATH];
        if (snprintf(child, sizeof child, "%s/%s", path, l.e[i].name) >= (int)sizeof child) bad = 1;
        else bad |= remove_tree(child);
    }
    listing_free(l.e, l.n);
    return bad || hf_rmdir(path) != HF_OK ? -1 : 0;
}

// ---- mounts and paths
static uint32_t mount_add(Cpu *c, const char *name, int kind, const char *root) {
    if (!name || !*name || strlen(name) >= sizeof mounts[0].name || strpbrk(name, ":/\\") || !strcmp(name, ".") ||
        !strcmp(name, ".."))
        return kind == M_ROM ? FS_R_INVALID : FS_R_INVALID_ARGUMENT;
    int slot = -1;
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NMOUNTS; i++) {
        if (mounts[i].kind && !strcmp(mounts[i].name, name)) {
            int same = kind == M_ROM && mounts[i].kind == M_ROM;   // the RomFS again under its own name: as asked already
            plat_mutex_unlock(&fs_lock);
            return same ? 0 : FS_R_ALREADY_EXISTS;
        }
        if (!mounts[i].kind && slot < 0) slot = i;
    }
    if (slot < 0) {
        plat_mutex_unlock(&fs_lock);
        hos_gap(c, "fs", "the mount table is full (%d mounts)", NMOUNTS);
    }
    mounts[slot].kind = kind;
    mounts[slot].id = ++mount_serial;
    snprintf(mounts[slot].name, sizeof mounts[slot].name, "%s", name);
    snprintf(mounts[slot].root, sizeof mounts[slot].root, "%s", root);
    plat_mutex_unlock(&fs_lock);
    rt_log("fs mount %s: -> %s", name, kind == M_ROM ? "RomFS" : root);
    return 0;
}

static void unmount(Cpu *c, const char *name) {
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (mounts[i].kind && !strcmp(mounts[i].name, name)) {
            int save = mounts[i].kind == M_SAVE;
            char root[ROOT_PATH];
            snprintf(root, sizeof root, "%s", mounts[i].root);
            mounts[i].kind = M_FREE;
            plat_mutex_unlock(&fs_lock);
            if (save) st_discard_prefix(root);   // changes never committed
            rt_log("fs unmount %s", name);
            return;
        }
    plat_mutex_unlock(&fs_lock);
    hos_gap(c, "fs", "Unmount of %s, which is not mounted", name);
}

// "name:/rest": the mount a path names (a copy) and what follows the colon.
static uint32_t resolve(const char *path, Mount *m, const char **rest) {
    const char *colon = strchr(path, ':');
    if (!colon || colon[1] != '/') return FS_R_INVALID_PATH;
    size_t n = (size_t)(colon - path);
    uint32_t r = FS_R_PATH_NOT_FOUND;
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (mounts[i].kind && strlen(mounts[i].name) == n && !strncmp(mounts[i].name, path, n)) {
            *m = mounts[i];
            r = 0;
            break;
        }
    plat_mutex_unlock(&fs_lock);
    *rest = colon + 1;
    return r;
}

// The host path of what follows the colon, under the mount's root; is_root: it names the root itself.
static uint32_t host_path(const Mount *m, const char *rest, char *out, size_t cap, int *is_root) {
    char rel[GUEST_PATH];
    if (!fs_path_norm(rest, rel, sizeof rel, 1)) return FS_R_INVALID_PATH;
    if (is_root) *is_root = !*rel;
    int n = *rel ? snprintf(out, cap, "%s/%s", m->root, rel) : snprintf(out, cap, "%s", m->root);
    return n < 0 || (size_t)n >= cap ? FS_R_INVALID_PATH : 0;
}

// The host path of a path in a mount that can change: FS_R_UNSUPPORTED_OPEN on the RomFS.
static uint32_t change_path(const char *path, char *out, size_t cap, int *is_root, Mount *mo) {
    Mount m;
    const char *rest;
    uint32_t r = resolve(path, &m, &rest);
    if (r) return r;
    if (m.kind == M_ROM) return FS_R_UNSUPPORTED_OPEN;
    if (mo) *mo = m;
    return host_path(&m, rest, out, cap, is_root);
}

// ---- mounts of each kind
// nn::Result nn::fs::MountRom(const char *name, void *cache, size_t cache_size): the pack is read in place, so the
// cache is not used, and a second name mounts the same image again.
static uint32_t mount_rom(Cpu *c, const char *name) {
    char why[256];
    if (!romfs_ready(why, sizeof why)) hos_gap(c, "fs", "%s", why);
    return mount_add(c, name, M_ROM, "");
}

// <game folder>/save/<user>, made: the folder of this user's save data. A user id of zero names nobody.
static uint32_t uid_root(Cpu *c, const uint64_t uid[2], char *out, size_t cap) {
    if (!uid[0] && !uid[1]) return FS_R_TARGET_NOT_FOUND;
    char leaf[64];
    snprintf(leaf, sizeof leaf, "save/%016llx%016llx", (unsigned long long)uid[1], (unsigned long long)uid[0]);
    under_game(c, out, cap, leaf);
    if (!make_dirs(out)) rt_fatal("fs: cannot create the save folder %s", out);
    return 0;
}

static uint32_t mount_save(Cpu *c, const char *name, const uint64_t uid[2]) {
    char root[ROOT_PATH];
    uint32_t r = uid_root(c, uid, root, sizeof root);
    if (r) return r;
    int finished = st_recover(root);   // a commit or a write the last run was stopped in
    if (finished != 0) rt_log("fs save recovery of %s: %d file(s) finished", root, finished);
    return mount_add(c, name, M_SAVE, root);
}

static uint32_t mount_temp(Cpu *c, const char *name) {
    char root[ROOT_PATH];
    under_game(c, root, sizeof root, "temp");
    plat_mutex_lock(&fs_lock);
    int first = !temp_cleared;
    temp_cleared = 1;
    plat_mutex_unlock(&fs_lock);
    if (first) remove_tree(root);
    if (!make_dirs(root)) rt_fatal("fs: cannot create the temporary folder %s", root);
    return mount_add(c, name, M_TEMP, root);
}

static void cache_dir(Cpu *c, int index, char *out, size_t cap) {
    char leaf[32];
    snprintf(leaf, sizeof leaf, "cache/%d", index);
    under_game(c, out, cap, leaf);
}

static uint32_t create_cache(Cpu *c, int index, int64_t size, int64_t journal) {
    if (index < 0 || size < 0 || journal < 0) return FS_R_INVALID_ARGUMENT;
    char p[ROOT_PATH];
    under_game(c, p, sizeof p, "cache");
    if (!make_dirs(p)) rt_fatal("fs: cannot create the cache folder %s", p);
    cache_dir(c, index, p, sizeof p);
    return fs_result(hf_mkdir(p));
}

static uint32_t delete_cache(Cpu *c, int index) {
    if (index < 0) return FS_R_INVALID_ARGUMENT;
    char p[ROOT_PATH];
    cache_dir(c, index, p, sizeof p);
    return remove_tree(p) ? FS_R_TARGET_NOT_FOUND : 0;
}

static uint32_t mount_cache(Cpu *c, const char *name, int index) {
    if (index < 0) return FS_R_INVALID_ARGUMENT;
    char p[ROOT_PATH];
    cache_dir(c, index, p, sizeof p);
    if (hf_kind(p) != HF_DIR) return FS_R_TARGET_NOT_FOUND;
    return mount_add(c, name, M_CACHE, p);
}

static uint32_t open_cache_list(Cpu *c, uint64_t out) {
    char p[ROOT_PATH];
    under_game(c, p, sizeof p, "cache");
    Listing ls = {0};
    hf_list(p, listing_add, &ls);   // no folder: no cache storage
    CacheList l = {{0}, 0, 0, 1};
    for (int i = 0; i < ls.n; i++) {
        char *end;
        long v = strtol(ls.e[i].name, &end, 10);
        if (ls.e[i].kind == HF_DIR && *ls.e[i].name && !*end && v >= 0 && v <= 0x7fffffff && l.count < 64)
            l.index[l.count++] = (int)v;
    }
    listing_free(ls.e, ls.n);
    int slot = -1;
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NLISTS && slot < 0; i++)
        if (!list_tab[i].used) {
            list_tab[i] = l;
            slot = i;
        }
    plat_mutex_unlock(&fs_lock);
    if (slot < 0) hos_gap(c, "fs", "the cache storage list handle table is full (%d)", NLISTS);
    uint64_t h = LIST_BASE + (uint64_t)slot;
    if (!hos_out(out, &h, 8)) {
        list_tab[slot].used = 0;
        return FS_R_INVALID_ARGUMENT;
    }
    return 0;
}

static CacheList *list_of(uint64_t h) {
    if (h < LIST_BASE || h >= LIST_BASE + NLISTS) return NULL;
    return list_tab[h - LIST_BASE].used ? &list_tab[h - LIST_BASE] : NULL;
}

static uint32_t read_cache_list(uint64_t count_out, uint64_t out, uint64_t h, int max) {
    CacheList *l = list_of(h);
    if (!l || !count_out || max < 0 || (!out && max)) return FS_R_INVALID_ARGUMENT;
    int32_t n = 0;
    while (n < max && l->next < l->count) {
        CacheInfo info;
        memset(&info, 0, sizeof info);
        info.index = l->index[l->next++];
        if (!hos_out(out + (uint64_t)n * sizeof info, &info, sizeof info)) return FS_R_INVALID_ARGUMENT;
        n++;
    }
    return hos_out(count_out, &n, 4) ? 0 : FS_R_INVALID_ARGUMENT;
}

static void close_cache_list(Cpu *c, uint64_t h) {
    CacheList *l = list_of(h);
    if (!l) hos_gap(c, "fs", "CloseCacheStorageList of a handle that is not open (%#llx)", (unsigned long long)h);
    plat_mutex_lock(&fs_lock);
    l->used = 0;
    plat_mutex_unlock(&fs_lock);
}

// nn::Result nn::fs::Commit(const char *name): the data of the mount's open files is on the disk at return; on save data
// the changed files are published whole, all or none after a stop (savetxn.h). Files closed
// concurrently with a Commit of their own mount are not something the game does.
static uint32_t commit(const char *name) {
    uint32_t id = 0;
    int found = 0, n = 0, save = 0;
    char root[ROOT_PATH];
    HostFile *pending[NFILES];
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NMOUNTS; i++)
        if (mounts[i].kind && !strcmp(mounts[i].name, name)) {
            found = 1, id = mounts[i].id, save = mounts[i].kind == M_SAVE;
            snprintf(root, sizeof root, "%s", mounts[i].root);
        }
    for (int i = 0; found && i < NFILES; i++)
        if (file_tab[i].h && (file_tab[i].mode & FS_OPEN_WRITE) && file_tab[i].mount == id) pending[n++] = file_tab[i].h;
    plat_mutex_unlock(&fs_lock);
    if (!found) return FS_R_PATH_NOT_FOUND;
    if (save) return fs_result(st_commit(root));   // publishes the shadows, each synced first
    for (int i = 0; i < n; i++) hf_sync(pending[i]);
    return 0;
}

// ---- files
static int file_slot(HostFile *f, int mode, uint32_t mount, const char *real, uint64_t bound) {
    char *copy = real ? strdup(real) : NULL;
    if (real && !copy) rt_fatal("fs: out of memory for a save handle");
    int slot = -1;
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NFILES && slot < 0; i++)
        if (!file_tab[i].h) {
            file_tab[i] = (File){f, mode, mount, copy, bound};
            slot = i;
        }
    plat_mutex_unlock(&fs_lock);
    if (slot < 0) free(copy);
    return slot;
}

// 1 when h is an open host file, with its file and open mode.
static int file_get(uint64_t h, HostFile **f, int *mode) {
    if (h < FILE_BASE || h >= FILE_BASE + NFILES) return 0;
    plat_mutex_lock(&fs_lock);
    *f = file_tab[h - FILE_BASE].h;
    if (mode) *mode = file_tab[h - FILE_BASE].mode;
    plat_mutex_unlock(&fs_lock);
    return *f != NULL;
}

// Free the slot of handle h and return its file; NULL when h is not an open host file.
static HostFile *take_file(uint64_t h) {
    if (h < FILE_BASE || h >= FILE_BASE + NFILES) return NULL;
    plat_mutex_lock(&fs_lock);
    HostFile *f = file_tab[h - FILE_BASE].h;
    char *real = file_tab[h - FILE_BASE].real;
    file_tab[h - FILE_BASE] = (File){0};
    plat_mutex_unlock(&fs_lock);
    free(real);
    return f;
}

// The host file to use for handle h (NULL when h is not an open host file). A save handle first follows its file: when the
// file has a shadow the handle is not on (or no longer has the one it was on, after a Commit), it is reopened on the right
// one; for_write on a handle open for writing starts the shadow, so the file itself is never written. *mode: the open mode.
// NULL with *err set (an HF_ code) when the shadow cannot be started (disk full, a name too long for a shadow); *err stays
// HF_OK when h is not an open host file. A file deleted or renamed under the handle has no shadow to start: the handle keeps
// the file it is already on.
static HostFile *file_bound(uint64_t h, int *mode, int for_write, int *err) {
    if (err) *err = HF_OK;
    HostFile *f;
    int md;
    if (!file_get(h, &f, &md)) return NULL;
    if (mode) *mode = md;
    char real[HOST_PATH];
    plat_mutex_lock(&fs_lock);
    File *e = &file_tab[h - FILE_BASE];
    int has = e->real && e->h == f && snprintf(real, sizeof real, "%s", e->real) < (int)sizeof real;
    uint64_t bound = e->bound;
    plat_mutex_unlock(&fs_lock);
    if (!has) return f;
    uint64_t want = st_serial(real);
    if (!want && for_write && (md & FS_OPEN_WRITE)) {
        int rc = st_begin(real, &want);
        if (rc == HF_NOT_FOUND) want = 0;   // the file is gone from its path: write to the handle that is open
        else if (rc != HF_OK) {
            if (err) *err = rc;
            return NULL;
        }
    }
    if (want == bound) return f;
    char target[HOST_PATH];
    if (want) st_shadow_path(real, target, sizeof target);
    else snprintf(target, sizeof target, "%s", real);
    HostFile *nf = hf_open(target, (md & FS_OPEN_WRITE) != 0, NULL);
    if (!nf) return f;   // the file is gone (deleted under the handle): keep what it has
    plat_mutex_lock(&fs_lock);
    e = &file_tab[h - FILE_BASE];
    int same = e->h == f;
    if (same) e->h = nf, e->bound = want;
    plat_mutex_unlock(&fs_lock);
    if (!same) {
        hf_close(nf);
        return f;
    }
    hf_close(f);
    return nf;
}

// nn::Result nn::fs::OpenFile(FileHandle *out, const char *path, int mode)
static uint32_t open_file(Cpu *c, uint64_t out, const char *path, int mode) {
    Mount m;
    const char *rest;
    uint32_t r = resolve(path, &m, &rest);
    if (r) return r;
    if (m.kind == M_ROM) {
        uint64_t h = 0;
        if (mode != FS_OPEN_READ) return FS_R_UNSUPPORTED_OPEN;
        r = romfs_open(&h, rest);
        if (r == FS_R_NO_SLOT) hos_gap(c, "fs", "the RomFS file handle table is full");
        if (r) return r;
        if (!hos_out(out, &h, 8)) {
            romfs_close(h);
            return FS_R_INVALID_ARGUMENT;
        }
        return 0;
    }
    if (!out || !(mode & (FS_OPEN_READ | FS_OPEN_WRITE)) || (mode & ~7)) return FS_R_INVALID_ARGUMENT;
    char p[HOST_PATH];
    r = host_path(&m, rest, p, sizeof p, NULL);
    if (r) return r;
    uint64_t shadow = m.kind == M_SAVE ? st_serial(p) : 0;   // a file with a shadow is opened on the shadow
    char target[HOST_PATH];
    if (shadow) st_shadow_path(p, target, sizeof target);
    else snprintf(target, sizeof target, "%s", p);
    if (hf_kind(target) != HF_FILE) return FS_R_PATH_NOT_FOUND;
    int err = 0;
    HostFile *f = hf_open(target, (mode & FS_OPEN_WRITE) != 0, &err);
    if (!f) return fs_result(err);
    int slot = file_slot(f, mode, m.id, m.kind == M_SAVE ? p : NULL, shadow);
    if (slot < 0) {
        hf_close(f);
        hos_gap(c, "fs", "the host file handle table is full (%d)", NFILES);
    }
    uint64_t h = FILE_BASE + (uint64_t)slot;
    if (!hos_out(out, &h, 8)) {
        hf_close(take_file(h));
        return FS_R_INVALID_ARGUMENT;
    }
    return 0;
}

static int64_t rom_pread(void *src, void *buf, uint32_t n, uint64_t off) {
    return romfs_pread((uint64_t)(uintptr_t)src, buf, n, off);
}
static int64_t host_pread(void *src, void *buf, uint32_t n, uint64_t off) { return hf_pread(src, buf, n, off); }

// Read size bytes at off through rd into the guest buffer at addr, in pieces; each piece's pages are committed first
// (a host read into a page that is reserved only fails). *done: the bytes read. A read that returns nothing before size
// is an end of file where eof_ok, else the pack is damaged. 0, or FS_R_INVALID_ARGUMENT when the buffer is not guest
// memory the host may write.
static uint32_t guest_read(uint64_t addr, uint64_t size, uint64_t off, int64_t (*rd)(void *, void *, uint32_t, uint64_t),
                           void *src, int eof_ok, uint64_t *done, const char *what) {
    *done = 0;
    while (*done < size) {
        uint64_t left = size - *done, at = addr + *done;
        uint32_t chunk = left > READ_CHUNK ? READ_CHUNK : (uint32_t)left;
        if (!rt_mem_commit_io(at, chunk, 1)) return FS_R_INVALID_ARGUMENT;
        int64_t got = rd(src, (void *)(uintptr_t)at, chunk, off + *done);
        if (got < 0 || (!got && !eof_ok))
            rt_fatal("fs: a read of the %s failed at offset %#llx", what, (unsigned long long)(off + *done));
        if (!got) break;
        *done += (uint64_t)got;
    }
    return 0;
}

// The read both ReadFile forms share: up to size bytes at offset, fewer at the end of the file.
static uint32_t read_core(uint64_t h, int64_t offset, uint64_t buf, uint64_t size, uint64_t *got) {
    *got = 0;
    if (romfs_handle(h)) {
        int64_t fsize = 0;
        if (offset < 0 || romfs_size(&fsize, h) || (uint64_t)offset > (uint64_t)fsize || (!buf && size)) return FS_R_INVALID;
        uint64_t left = (uint64_t)fsize - (uint64_t)offset, n = size < left ? size : left;
        return guest_read(buf, n, (uint64_t)offset, rom_pread, (void *)(uintptr_t)h, 0, got, "RomFS image");
    }
    HostFile *f;
    int mode;
    if (offset < 0 || (!buf && size) || !(f = file_bound(h, &mode, 0, NULL))) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_READ)) return FS_R_OPEN_MODE;
    return guest_read(buf, size, (uint64_t)offset, host_pread, f, 1, got, "host file");
}

// nn::Result nn::fs::WriteFile(FileHandle, s64 offset, const void *buffer, size_t size, const WriteOption &)
static uint32_t write_file(uint64_t h, int64_t offset, uint64_t buf, uint64_t size, uint64_t option) {
    if (romfs_handle(h)) return FS_R_OPEN_MODE;   // opened for reading only
    HostFile *f;
    int mode;
    if (offset < 0 || (!buf && size) || !file_get(h, &f, &mode)) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_WRITE)) return FS_R_OPEN_MODE;
    int berr;
    f = file_bound(h, NULL, 1, &berr);   // a save file's changes go to its shadow
    if (!f) return berr ? fs_result(berr) : FS_R_INVALID_ARGUMENT;
    uint32_t flags = 0;
    if (option && !hos_in(&flags, option, 4)) return FS_R_INVALID_ARGUMENT;
    uint64_t cur = 0;
    if (!hf_size(f, &cur)) rt_fatal("fs: cannot read the size of a host file");
    if (size > (uint64_t)INT64_MAX - (uint64_t)offset) return FS_R_INVALID_ARGUMENT;
    if ((uint64_t)offset + size > cur && !(mode & FS_OPEN_APPEND)) return FS_R_APPEND_REQUIRED;
    for (uint64_t done = 0; done < size;) {
        uint64_t left = size - done;
        uint32_t chunk = left > READ_CHUNK ? READ_CHUNK : (uint32_t)left;
        if (!rt_mem_commit_io(buf + done, chunk, 0)) return FS_R_INVALID_ARGUMENT;
        int64_t put = hf_pwrite(f, (const void *)(uintptr_t)(buf + done), chunk, (uint64_t)offset + done);
        if (put <= 0) rt_fatal("fs: a write to a host file failed");
        done += (uint64_t)put;
    }
    if (flags & 1) hf_sync(f);   // WriteOptionFlag_Flush
    return 0;
}

static uint32_t file_size(uint64_t out, uint64_t h) {
    int64_t v = 0;
    if (romfs_handle(h)) {
        uint32_t r = romfs_size(&v, h);
        if (r) return r;
    } else {
        HostFile *f;
        uint64_t size = 0;
        if (!(f = file_bound(h, NULL, 0, NULL)) || !hf_size(f, &size)) return FS_R_INVALID_ARGUMENT;
        v = (int64_t)size;
    }
    return hos_out(out, &v, 8) ? 0 : FS_R_INVALID_ARGUMENT;
}

static uint32_t set_file_size(uint64_t h, int64_t size) {
    if (romfs_handle(h)) return FS_R_OPEN_MODE;
    HostFile *f;
    int mode;
    if (size < 0 || !file_get(h, &f, &mode)) return FS_R_INVALID_ARGUMENT;
    if (!(mode & FS_OPEN_WRITE)) return FS_R_OPEN_MODE;
    int berr;
    if (!(f = file_bound(h, NULL, 1, &berr))) return berr ? fs_result(berr) : FS_R_INVALID_ARGUMENT;
    return fs_result(hf_set_size(f, (uint64_t)size));
}

static uint32_t flush_file(uint64_t h) {
    if (romfs_handle(h)) return 0;   // nothing was written
    HostFile *f;
    int mode;
    if (!(f = file_bound(h, &mode, 0, NULL))) return FS_R_INVALID_ARGUMENT;
    if (mode & FS_OPEN_WRITE) hf_sync(f);
    return 0;
}

static void close_file(Cpu *c, uint64_t h) {
    if (romfs_handle(h) ? romfs_close(h) : 0) return;
    HostFile *f = take_file(h);
    if (!f) hos_gap(c, "fs", "CloseFile of a handle that is not open (%#llx)", (unsigned long long)h);
    hf_close(f);
}

static uint32_t create_file(const char *path, int64_t size) {
    char p[HOST_PATH];
    Mount m;
    uint32_t r = change_path(path, p, sizeof p, NULL, &m);
    if (r) return r;
    if (size < 0) return FS_R_INVALID_ARGUMENT;
    if (m.kind == M_SAVE) {   // the new file appears when it is committed
        uint64_t serial;
        return fs_result(st_begin_create(p, (uint64_t)size, &serial));
    }
    int err = 0;
    HostFile *f = hf_create(p, &err);
    if (!f) return fs_result(err);
    err = size ? hf_set_size(f, (uint64_t)size) : HF_OK;
    hf_close(f);
    return fs_result(err);
}

static uint32_t delete_file(const char *path) {
    char p[HOST_PATH];
    Mount m;
    uint32_t r = change_path(path, p, sizeof p, NULL, &m);
    if (r) return r;
    if (hf_kind(p) == HF_DIR) return FS_R_INVALID_ARGUMENT;   // DeleteFile on a directory
    if (m.kind == M_SAVE && st_serial(p)) {   // the shadow goes with the file; a file only made since the last commit is gone
        st_discard_prefix(p);
        if (hf_kind(p) == HF_MISSING) return 0;
    }
    return fs_result(hf_unlink(p));
}

static uint32_t rename_file(const char *from, const char *to) {
    char a[HOST_PATH], b[HOST_PATH];
    int root = 0;
    Mount m;
    uint32_t r = change_path(from, a, sizeof a, &root, &m);
    if (!r) r = change_path(to, b, sizeof b, NULL, NULL);
    if (r) return r;
    size_t n = strcspn(from, ":");
    if (root || strcspn(to, ":") != n || strncmp(from, to, n)) return FS_R_INVALID_PATH;   // a root, or across mounts
    if (hf_kind(b) != HF_MISSING || (m.kind == M_SAVE && st_serial(b))) return FS_R_ALREADY_EXISTS;
    // What moves is published first, by itself: the move is one rename, and a shadow must not stay behind under the old name.
    if (m.kind == M_SAVE && st_publish_prefix(a) != HF_OK) return FS_R_INVALID_ARGUMENT;
    return fs_result(hf_rename(a, b));
}

static uint32_t entry_type(uint64_t out, const char *path) {
    Mount m;
    const char *rest;
    int type = FS_ENTRY_FILE;
    uint32_t r = resolve(path, &m, &rest);
    if (r) return r;
    if (m.kind == M_ROM) {
        r = romfs_entry_type(&type, rest);
        if (r) return r;
    } else {
        char p[HOST_PATH];
        r = host_path(&m, rest, p, sizeof p, NULL);
        if (r) return r;
        int kind = hf_kind(p);
        if (kind == HF_MISSING && m.kind == M_SAVE && st_serial(p)) kind = HF_FILE;   // made since the last commit
        if (kind == HF_MISSING) return FS_R_PATH_NOT_FOUND;
        type = kind == HF_DIR ? FS_ENTRY_DIRECTORY : FS_ENTRY_FILE;
    }
    int32_t v = type;
    return hos_out(out, &v, 4) ? 0 : FS_R_INVALID_ARGUMENT;
}

// ---- directories
// A save folder's listing as the guest sees it: a shadow shows under its file's name (replacing the file's own row, and
// standing for a file not committed yet), and the shadows nobody owns and the commit marker are not shown.
static void save_listing(Listing *l, const char *dir) {
    int out = 0;
    for (int i = 0; i < l->n; i++) {
        Listed e = l->e[i];
        size_t len = strlen(e.name), sl = strlen(ST_SUFFIX);
        char real[HOST_PATH];
        int drop = !strcmp(e.name, ST_MARKER) || !strcmp(e.name, ST_MARKER_TMP);
        if (!drop && len > sl && !strcmp(e.name + len - sl, ST_SUFFIX)) {
            drop = snprintf(real, sizeof real, "%s/%.*s", dir, (int)(len - sl), e.name) >= (int)sizeof real || !st_serial(real);
            if (!drop) e.name[len - sl] = 0;
        } else if (!drop && e.kind == HF_FILE) {   // a file that has a shadow is listed once, by its shadow
            drop = snprintf(real, sizeof real, "%s/%s", dir, e.name) < (int)sizeof real && st_serial(real);
        }
        if (drop) free(e.name);
        else l->e[out++] = e;
    }
    l->n = out;
}

static uint32_t create_directory(const char *path) {
    char p[HOST_PATH];
    uint32_t r = change_path(path, p, sizeof p, NULL, NULL);
    return r ? r : fs_result(hf_mkdir(p));
}

// The root of a mount is not a directory the game can remove: its mount would be gone.
static uint32_t delete_directory(const char *path, int recursive) {
    char p[HOST_PATH];
    int root = 0;
    Mount m;
    uint32_t r = change_path(path, p, sizeof p, &root, &m);
    if (r) return r;
    if (root) return FS_R_INVALID_PATH;
    if (hf_kind(p) != HF_DIR) return FS_R_PATH_NOT_FOUND;
    if (recursive && m.kind == M_SAVE) st_discard_prefix(p);
    if (!recursive) return fs_result(hf_rmdir(p));
    return remove_tree(p) ? FS_R_INVALID_ARGUMENT : 0;
}

// nn::Result nn::fs::OpenDirectory(DirectoryHandle *out, const char *path, int mode): the entries are taken whole now.
static uint32_t open_directory(Cpu *c, uint64_t out, const char *path, int mode) {
    Mount m;
    const char *rest;
    uint32_t r = resolve(path, &m, &rest);
    if (r) return r;
    if (!out || mode < 1 || mode > 3) return FS_R_INVALID_ARGUMENT;
    Listing l = {0};
    if (m.kind == M_ROM) {
        r = romfs_list(rest, listing_add, &l);
    } else {
        char p[HOST_PATH];
        r = host_path(&m, rest, p, sizeof p, NULL);
        if (!r && hf_kind(p) != HF_DIR) r = FS_R_PATH_NOT_FOUND;
        if (!r) r = fs_result(hf_list(p, listing_add, &l));
        if (!r && m.kind == M_SAVE) save_listing(&l, p);
        if (!r && l.n > 1) qsort(l.e, (size_t)l.n, sizeof *l.e, listed_cmp);
    }
    if (r) {
        listing_free(l.e, l.n);
        return r;
    }
    int slot = -1;
    plat_mutex_lock(&fs_lock);
    for (int i = 0; i < NDIRS && slot < 0; i++)
        if (!dir_tab[i].used) {
            dir_tab[i] = (Dir){l.e, l.n, 0, mode, 1};
            slot = i;
        }
    plat_mutex_unlock(&fs_lock);
    if (slot < 0) hos_gap(c, "fs", "the directory handle table is full (%d)", NDIRS);
    uint64_t h = DIR_BASE + (uint64_t)slot;
    if (!hos_out(out, &h, 8)) {
        plat_mutex_lock(&fs_lock);
        dir_tab[slot] = (Dir){0};
        plat_mutex_unlock(&fs_lock);
        listing_free(l.e, l.n);
        return FS_R_INVALID_ARGUMENT;
    }
    return 0;
}

static Dir *dir_of(uint64_t h) {
    if (h < DIR_BASE || h >= DIR_BASE + NDIRS) return NULL;
    return dir_tab[h - DIR_BASE].used ? &dir_tab[h - DIR_BASE] : NULL;
}

// Whether ReadDirectory delivers e: its kind is among the open mode's, and nn::fs can hold its name (768 bytes).
static int entry_wanted(const Dir *d, const Listed *e) {
    return strlen(e->name) < sizeof(((DirEntry *)0)->name) && (d->mode & (e->kind == HF_DIR ? 1 : 2));
}

static uint32_t read_directory(uint64_t count_out, uint64_t out, uint64_t h, int64_t max) {
    Dir *d = dir_of(h);
    if (!d || !count_out || max < 0 || (!out && max)) return FS_R_INVALID_ARGUMENT;
    int64_t n = 0;
    while (n < max && d->next < d->n) {
        const Listed *e = &d->e[d->next++];
        if (!entry_wanted(d, e)) continue;
        DirEntry ent;
        memset(&ent, 0, sizeof ent);
        memcpy(ent.name, e->name, strlen(e->name) + 1);
        ent.type = e->kind == HF_DIR ? FS_ENTRY_DIRECTORY : FS_ENTRY_FILE;
        ent.size = e->kind == HF_DIR ? 0 : (int64_t)e->size;
        if (!hos_out(out + (uint64_t)n * sizeof ent, &ent, sizeof ent)) return FS_R_INVALID_ARGUMENT;
        n++;
    }
    return hos_out(count_out, &n, 8) ? 0 : FS_R_INVALID_ARGUMENT;
}

// nn::Result nn::fs::GetDirectoryEntryCount(s64 *out, DirectoryHandle): the entries ReadDirectory would deliver from the start.
static uint32_t directory_entry_count(uint64_t out, uint64_t h) {
    Dir *d = dir_of(h);
    if (!d) return FS_R_INVALID_ARGUMENT;
    int64_t n = 0;
    for (int i = 0; i < d->n; i++) n += entry_wanted(d, &d->e[i]) != 0;
    return hos_out(out, &n, 8) ? 0 : FS_R_INVALID_ARGUMENT;
}

static void close_directory(Cpu *c, uint64_t h) {
    Dir d = {0};
    if (dir_of(h)) {
        plat_mutex_lock(&fs_lock);
        d = dir_tab[h - DIR_BASE];
        dir_tab[h - DIR_BASE] = (Dir){0};
        plat_mutex_unlock(&fs_lock);
    }
    if (!d.used) hos_gap(c, "fs", "CloseDirectory of a handle that is not open (%#llx)", (unsigned long long)h);
    listing_free(d.e, d.n);
}

// ---- overrides: AAPCS64 thunks. FileHandle, DirectoryHandle and CacheStorageListHandle are 8-byte structs passed in
// one register; account::Uid and the ReadOption and WriteOption are passed by reference, fs::UserId by value in two.
#define I32(i) ((int32_t)arg_x(c, i))
#define I64(i) ((int64_t)arg_x(c, i))
#define PATH_ARG(i, expr)                                         \
    do {                                                          \
        char path_[GUEST_PATH];                                   \
        uint32_t r_ = guest_path(c, i, path_, sizeof path_);      \
        ret_x(c, r_ ? r_ : (expr));                               \
    } while (0)

static void ov_mount_rom(Cpu *c) {
    char n[64];
    ret_x(c, mount_rom(c, str_arg(c, 0, n, sizeof n)));
}

// nn::Result nn::fs::QueryMountRomCacheSize(size_t *out): none is needed, the pack is read in place.
static void ov_query_cache(Cpu *c) {
    uint64_t none = 0;
    ret_x(c, hos_out(arg_x(c, 0), &none, 8) ? 0 : FS_R_INVALID);
}

static void ov_gap_application(Cpu *c) {
    const char *n = rt_hostfn_name(c->pc);
    hos_gap(c, "fs", "%s names an application: another application's RomFS or save data has no host answer, and the host "
            "layer has no way yet to tell this program's own id from another's", n ? n : "an nn::fs function");
}

static void ov_gap_rom_index(Cpu *c) {
    const char *n = rt_hostfn_name(c->pc);
    hos_gap(c, "fs", "%s takes an int that selects among RomFS images, and what it selects is not known here",
            n ? n : "an nn::fs function");
}

static void ov_initialize(Cpu *c) { (void)c; }
static void ov_set_priority(Cpu *c) { (void)c; }

static void ov_open_file(Cpu *c) { PATH_ARG(1, open_file(c, arg_x(c, 0), path_, I32(2))); }

// ReadFile(size_t *out, FileHandle, s64 offset, void *buffer, size_t size [, const ReadOption &])
static void ov_read_file(Cpu *c) {
    uint64_t got = 0;
    uint32_t r = read_core(arg_x(c, 1), I64(2), arg_x(c, 3), arg_x(c, 4), &got);
    if (!r && !hos_out(arg_x(c, 0), &got, 8)) r = FS_R_INVALID_ARGUMENT;
    ret_x(c, r);
}

// ReadFile(FileHandle, s64 offset, void *buffer, size_t size [, const ReadOption &]): all of the range, or an error
static void ov_read_exact(Cpu *c) {
    uint64_t got = 0;
    uint32_t r = read_core(arg_x(c, 0), I64(1), arg_x(c, 2), arg_x(c, 3), &got);
    ret_x(c, r ? r : got == arg_x(c, 3) ? 0 : FS_R_INVALID_ARGUMENT);
}

static void ov_write_file(Cpu *c) { ret_x(c, write_file(arg_x(c, 0), I64(1), arg_x(c, 2), arg_x(c, 3), arg_x(c, 4))); }
static void ov_file_size(Cpu *c) { ret_x(c, file_size(arg_x(c, 0), arg_x(c, 1))); }
static void ov_set_file_size(Cpu *c) { ret_x(c, set_file_size(arg_x(c, 0), I64(1))); }
static void ov_flush_file(Cpu *c) { ret_x(c, flush_file(arg_x(c, 0))); }
static void ov_close_file(Cpu *c) { close_file(c, arg_x(c, 0)); }
static void ov_create_file(Cpu *c) { PATH_ARG(0, create_file(path_, I64(1))); }
static void ov_delete_file(Cpu *c) { PATH_ARG(0, delete_file(path_)); }
static void ov_entry_type(Cpu *c) { PATH_ARG(1, entry_type(arg_x(c, 0), path_)); }
static void ov_create_directory(Cpu *c) { PATH_ARG(0, create_directory(path_)); }
static void ov_delete_directory(Cpu *c) { PATH_ARG(0, delete_directory(path_, 0)); }
static void ov_delete_recursive(Cpu *c) { PATH_ARG(0, delete_directory(path_, 1)); }
static void ov_open_directory(Cpu *c) { PATH_ARG(1, open_directory(c, arg_x(c, 0), path_, I32(2))); }
static void ov_read_directory(Cpu *c) { ret_x(c, read_directory(arg_x(c, 0), arg_x(c, 1), arg_x(c, 2), I64(3))); }
static void ov_entry_count(Cpu *c) { ret_x(c, directory_entry_count(arg_x(c, 0), arg_x(c, 1))); }
static void ov_close_directory(Cpu *c) { close_directory(c, arg_x(c, 0)); }

static void ov_rename_file(Cpu *c) {
    char from[GUEST_PATH], to[GUEST_PATH];
    uint32_t r = guest_path(c, 0, from, sizeof from);
    if (!r) r = guest_path(c, 1, to, sizeof to);
    ret_x(c, r ? r : rename_file(from, to));
}

// MountSaveData(const char *name, const account::Uid &)
static void ov_mount_save(Cpu *c) {
    char n[64];
    uint64_t uid[2];
    ret_x(c, uid_arg(arg_x(c, 1), uid) ? mount_save(c, str_arg(c, 0, n, sizeof n), uid) : FS_R_INVALID_ARGUMENT);
}

// MountSaveData(const char *name, fs::UserId): the 16 bytes of the id arrive in x1 and x2
static void ov_mount_save_user(Cpu *c) {
    char n[64];
    uint64_t uid[2] = {arg_x(c, 1), arg_x(c, 2)};
    ret_x(c, mount_save(c, str_arg(c, 0, n, sizeof n), uid));
}

// EnsureSaveData(const account::Uid &): the save folder is made now instead of at the first mount
static void ov_ensure_save(Cpu *c) {
    char root[ROOT_PATH];
    uint64_t uid[2];
    ret_x(c, uid_arg(arg_x(c, 0), uid) ? uid_root(c, uid, root, sizeof root) : FS_R_INVALID_ARGUMENT);
}

static void ov_mount_temp(Cpu *c) {
    char n[64];
    ret_x(c, mount_temp(c, str_arg(c, 0, n, sizeof n)));
}

static void ov_unmount(Cpu *c) {
    char n[64];
    unmount(c, hos_str(arg_x(c, 0), n, sizeof n));
}

static void ov_commit(Cpu *c) {
    char n[64];
    ret_x(c, commit(hos_str(arg_x(c, 0), n, sizeof n)));
}

static void ov_create_cache(Cpu *c) { ret_x(c, create_cache(c, I32(0), I64(1), I64(2))); }
static void ov_delete_cache(Cpu *c) { ret_x(c, delete_cache(c, I32(0))); }

static void ov_mount_cache(Cpu *c) {
    char n[64];
    ret_x(c, mount_cache(c, str_arg(c, 0, n, sizeof n), I32(1)));
}

static void ov_open_cache_list(Cpu *c) { ret_x(c, open_cache_list(c, arg_x(c, 0))); }
static void ov_read_cache_list(Cpu *c) { ret_x(c, read_cache_list(arg_x(c, 0), arg_x(c, 1), arg_x(c, 2), I32(3))); }
static void ov_close_cache_list(Cpu *c) { close_cache_list(c, arg_x(c, 0)); }
#undef PATH_ARG
#undef I32
#undef I64

// RomFS
HOS_OVERRIDE("_ZN2nn2fs8MountRomEPKcPvm", ov_mount_rom, HOS_IMPLEMENTED,
             "mounts the RomFS pack under that name, read-only, cache unused; donor romfs.c:607, 798");
HOS_OVERRIDE("_ZN2nn2fs8MountRomEPKc", ov_mount_rom, HOS_IMPLEMENTED,
             "the same without cache arguments; donor romfs.c:812");
HOS_OVERRIDE("_ZN2nn2fs8MountRomEPKciPvm", ov_gap_rom_index, HOS_GAP,
             "gap, named fatal: the int that selects a RomFS image has no known meaning; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs8MountRomEPKcNS_13ApplicationIdEPvm", ov_gap_application, HOS_GAP,
             "gap, named fatal: another application's RomFS; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs22QueryMountRomCacheSizeEPm", ov_query_cache, HOS_IMPLEMENTED,
             "answers 0: the pack is read in place; donor romfs.c:800");
HOS_OVERRIDE("_ZN2nn2fs22QueryMountRomCacheSizeEPmi", ov_gap_rom_index, HOS_GAP,
             "gap, named fatal: the int that selects a RomFS image has no known meaning; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs22QueryMountRomCacheSizeEPmNS_13ApplicationIdE", ov_gap_application, HOS_GAP,
             "gap, named fatal: another application's RomFS; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs24InitializeForApplicationEv", ov_initialize, HOS_STUB,
             "nothing to start: no fs service session, the pack opens at the first MountRom; donor romfs.c:806");
// files
HOS_OVERRIDE("_ZN2nn2fs8OpenFileEPNS0_10FileHandleEPKci", ov_open_file, HOS_IMPLEMENTED,
             "RomFS files from the pack (read only), other mounts from host files; donor fs_host.c:846, romfs.c:633");
HOS_OVERRIDE("_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvm", ov_read_file, HOS_IMPLEMENTED,
             "reads up to the size at the offset into guest memory, fewer at the end of the file; donor fs_host.c:876, romfs.c:699");
HOS_OVERRIDE("_ZN2nn2fs8ReadFileEPmNS0_10FileHandleElPvmRKNS0_10ReadOptionE", ov_read_file, HOS_IMPLEMENTED,
             "the same, the read option is ignored; donor fs_host.c:876");
HOS_OVERRIDE("_ZN2nn2fs8ReadFileENS0_10FileHandleElPvm", ov_read_exact, HOS_IMPLEMENTED,
             "reads exactly the size or fails; donor fs_host.c:894");
HOS_OVERRIDE("_ZN2nn2fs8ReadFileENS0_10FileHandleElPvmRKNS0_10ReadOptionE", ov_read_exact, HOS_IMPLEMENTED,
             "the same, the read option is ignored; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs11GetFileSizeEPlNS0_10FileHandleE", ov_file_size, HOS_IMPLEMENTED,
             "the size of the pack's file or the host file; donor fs_host.c:920, romfs.c:719");
HOS_OVERRIDE("_ZN2nn2fs9CloseFileENS0_10FileHandleE", ov_close_file, HOS_IMPLEMENTED,
             "frees the handle; a handle that is not open is a named fatal; donor fs_host.c:947, romfs.c:727");
HOS_OVERRIDE("_ZN2nn2fs9WriteFileENS0_10FileHandleElPKvmRKNS0_11WriteOptionE", ov_write_file, HOS_IMPLEMENTED,
             "writes a host file in place, past the end only with the append mode, flush option honoured; donor fs_host.c:900");
HOS_OVERRIDE("_ZN2nn2fs11SetFileSizeENS0_10FileHandleEl", ov_set_file_size, HOS_IMPLEMENTED,
             "resizes a host file opened for writing; donor fs_host.c:929");
HOS_OVERRIDE("_ZN2nn2fs9FlushFileENS0_10FileHandleE", ov_flush_file, HOS_IMPLEMENTED,
             "flushes a host file opened for writing to the disk; donor fs_host.c:939");
HOS_OVERRIDE("_ZN2nn2fs10CreateFileEPKcl", ov_create_file, HOS_IMPLEMENTED,
             "makes a host file of that size, zero-filled; donor fs_host.c:965");
HOS_OVERRIDE("_ZN2nn2fs10CreateFileEPKcli", ov_create_file, HOS_IMPLEMENTED,
             "the same, the option is ignored; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs10DeleteFileEPKc", ov_delete_file, HOS_IMPLEMENTED, "removes a host file; donor fs_host.c:978");
HOS_OVERRIDE("_ZN2nn2fs10RenameFileEPKcS2_", ov_rename_file, HOS_IMPLEMENTED,
             "renames within one host mount, never over an existing path; donor fs_host.c:986");
HOS_OVERRIDE("_ZN2nn2fs12GetEntryTypeEPNS0_18DirectoryEntryTypeEPKc", ov_entry_type, HOS_IMPLEMENTED,
             "file or directory, from the pack's sorted paths or the host; donor fs_host.c:997, romfs.c:746");
// directories
HOS_OVERRIDE("_ZN2nn2fs15CreateDirectoryEPKc", ov_create_directory, HOS_IMPLEMENTED,
             "makes a host directory; donor fs_host.c:1013");
HOS_OVERRIDE("_ZN2nn2fs15DeleteDirectoryEPKc", ov_delete_directory, HOS_IMPLEMENTED,
             "removes an empty host directory, never a mount's root; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs26DeleteDirectoryRecursivelyEPKc", ov_delete_recursive, HOS_IMPLEMENTED,
             "removes a host directory tree, never a mount's root; donor fs_host.c:1020");
HOS_OVERRIDE("_ZN2nn2fs13OpenDirectoryEPNS0_15DirectoryHandleEPKci", ov_open_directory, HOS_IMPLEMENTED,
             "takes the entries whole at open, from the pack's sorted paths or the host; donor fs_host.c:1028, romfs.c:763");
HOS_OVERRIDE("_ZN2nn2fs13ReadDirectoryEPlPNS0_14DirectoryEntryENS0_15DirectoryHandleEl", ov_read_directory,
             HOS_IMPLEMENTED, "delivers entries in the SDK's DirectoryEntry layout, files and directories by the open mode; donor fs_host.c:1059");
HOS_OVERRIDE("_ZN2nn2fs22GetDirectoryEntryCountEPlNS0_15DirectoryHandleE", ov_entry_count, HOS_IMPLEMENTED,
             "counts the entries the open mode selects; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs14CloseDirectoryENS0_15DirectoryHandleE", ov_close_directory, HOS_IMPLEMENTED,
             "frees the handle; a handle that is not open is a named fatal; donor fs_host.c:1080");
// save data, temporary and cache storage, and the mounts
HOS_OVERRIDE("_ZN2nn2fs13MountSaveDataEPKcRKNS_7account3UidE", ov_mount_save, HOS_IMPLEMENTED,
             "mounts <game folder>/save/<user>, made on first use; donor fs_host.c:696");
HOS_OVERRIDE("_ZN2nn2fs13MountSaveDataEPKcNS0_6UserIdE", ov_mount_save_user, HOS_IMPLEMENTED,
             "the same, the user id passed by value; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs13MountSaveDataEPKcNS_3ncm13ApplicationIdENS0_6UserIdE", ov_gap_application, HOS_GAP,
             "gap, named fatal: another application's save data; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs13MountSaveDataEPKcNS_13ApplicationIdERKNS_7account3UidE", ov_gap_application, HOS_GAP,
             "gap, named fatal: another application's save data; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs21MountSaveDataReadOnlyEPKcNS_13ApplicationIdERKNS_7account3UidE", ov_gap_application, HOS_GAP,
             "gap, named fatal: the donor answers for its own application id, which is not known here; donor fs_host.c:714");
HOS_OVERRIDE("_ZN2nn2fs18IsSaveDataExistingENS_13ApplicationIdERKNS_7account3UidE", ov_gap_application, HOS_GAP,
             "gap, named fatal: the donor answers for its own application id, which is not known here; donor fs_host.c:708");
HOS_OVERRIDE("_ZN2nn2fs14EnsureSaveDataERKNS_7account3UidE", ov_ensure_save, HOS_IMPLEMENTED,
             "makes the user's save folder and succeeds; donor fs_host.c:1150");
HOS_OVERRIDE("_ZN2nn2fs21MountTemporaryStorageEPKc", ov_mount_temp, HOS_IMPLEMENTED,
             "mounts <game folder>/temp, emptied on its first mount of the run; donor fs_host.c:719");
HOS_OVERRIDE("_ZN2nn2fs7UnmountEPKc", ov_unmount, HOS_IMPLEMENTED,
             "removes the mount; a name that is not mounted is a named fatal; donor fs_host.c:788");
HOS_OVERRIDE("_ZN2nn2fs6CommitEPKc", ov_commit, HOS_IMPLEMENTED,
             "flushes the mount's open files to the disk; donor fs_host.c:810");
HOS_OVERRIDE("_ZN2nn2fs6CommitEPKcRKNS0_12CommitOptionE", ov_commit, HOS_IMPLEMENTED,
             "the same, the option is ignored; not in the donor");
HOS_OVERRIDE("_ZN2nn2fs18CreateCacheStorageEill", ov_create_cache, HOS_IMPLEMENTED,
             "makes <game folder>/cache/<index>; donor fs_host.c:728");
HOS_OVERRIDE("_ZN2nn2fs18DeleteCacheStorageEi", ov_delete_cache, HOS_IMPLEMENTED,
             "removes <game folder>/cache/<index>; donor fs_host.c:736");
HOS_OVERRIDE("_ZN2nn2fs17MountCacheStorageEPKci", ov_mount_cache, HOS_IMPLEMENTED,
             "mounts <game folder>/cache/<index> when it exists; donor fs_host.c:743");
HOS_OVERRIDE("_ZN2nn2fs20OpenCacheStorageListEPNS0_22CacheStorageListHandleE", ov_open_cache_list, HOS_IMPLEMENTED,
             "lists the numbered folders of <game folder>/cache; donor fs_host.c:755");
HOS_OVERRIDE("_ZN2nn2fs20ReadCacheStorageListEPiPNS0_16CacheStorageInfoENS0_22CacheStorageListHandleEi",
             ov_read_cache_list, HOS_IMPLEMENTED, "delivers the cache indexes; donor fs_host.c:774");
HOS_OVERRIDE("_ZN2nn2fs21CloseCacheStorageListENS0_22CacheStorageListHandleE", ov_close_cache_list, HOS_IMPLEMENTED,
             "frees the handle; donor fs_host.c:786");
HOS_OVERRIDE("_ZN2nn2fs26SetPriorityOnCurrentThreadENS0_8PriorityE", ov_set_priority, HOS_STUB,
             "the priority orders I/O inside the fs service, which host I/O has none of; donor fs_host.c:1164");
