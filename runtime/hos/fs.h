// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What ov_fs.c (the nn::fs overrides), romfs.c (the read-only RomFS pack) and hostfile.c (host files and directories)
// share. Included by those three files only.
//
// Adapted from oot-pc-recomp runtime/fs.h and runtime/romfs_path.h at 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17
// (GPL-3.0-or-later; each file's own header says it was adapted from MacNativeGames games/totk/runtime/fs_host.h and
// romfs_path.h at b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors).
// Changed here: the results and the handle range of the RomFS are kept; the RomFS interface is this pack's own (rows of
// a mapped OOT2RFS1 index, a positional read the caller drives into guest memory, no overlays, no read cache); the host
// file calls are declared here instead of in a platform layer; one function normalises the path of every kind of mount.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hos.h"

// ---- nn::fs results, as the donor names them (module 2: value = 2 | description << 9)
#define FS_R_PATH_NOT_FOUND 0x202u
#define FS_R_ALREADY_EXISTS 0x402u
#define FS_R_NOT_EMPTY 0x1002u
#define FS_R_UNSUPPORTED_OPEN 0x6402u    // a change to a read-only mount, or a RomFS file opened in any mode but read
#define FS_R_INVALID 0xe402u             // RomFS: bad handle, offset or argument
#define FS_R_TARGET_NOT_FOUND 0x7d402u
#define FS_R_INVALID_ARGUMENT 0x2ee202u
#define FS_R_INVALID_PATH 0x2ee602u
#define FS_R_OPEN_MODE 0x307402u         // a read of a file opened without read mode, a write without write mode
#define FS_R_APPEND_REQUIRED 0x2f5e02u   // a write past the end of a file opened without the append mode
// Never given to the guest: a handle table is full, which ov_fs.c turns into a named fatal.
#define FS_R_NO_SLOT 0xffffffffu

enum { FS_OPEN_READ = 1, FS_OPEN_WRITE = 2, FS_OPEN_APPEND = 4 };   // nn::fs::OpenMode
enum { FS_ENTRY_DIRECTORY = 0, FS_ENTRY_FILE = 1 };                 // nn::fs::DirectoryEntryType

// ---- paths
// What follows a mount name's colon ("/a//b/") in its canonical form ("a/b"; "" for the mount's root): leading, repeated
// and trailing slashes dropped. 0 when the path is malformed: it does not start with '/', holds a backslash or a "."
// or ".." component, or does not fit in cap bytes. strict (host mounts) also refuses what a host would not store under
// the same name: a control character or one of : * ? " < > |, and a name ending in a dot or a space.
static inline int fs_path_norm(const char *rest, char *out, size_t cap, int strict) {
    if (!rest || rest[0] != '/' || !cap) return 0;
    size_t n = 0;
    for (const char *q = rest; *q;) {
        while (*q == '/') q++;
        const char *e = q;
        while (*e && *e != '/') e++;
        size_t len = (size_t)(e - q);
        if (!len) break;
        if ((len == 1 && q[0] == '.') || (len == 2 && q[0] == '.' && q[1] == '.')) return 0;
        for (size_t i = 0; i < len; i++) {
            unsigned char ch = (unsigned char)q[i];
            if (ch == '\\') return 0;
            if (strict && (ch < 32 || strchr(":*?\"<>|", ch))) return 0;
        }
        if (strict && (q[len - 1] == '.' || q[len - 1] == ' ')) return 0;
        if (n + (n ? 1u : 0u) + len + 1 > cap) return 0;
        if (n) out[n++] = '/';
        memcpy(out + n, q, len);
        n += len;
        q = e;
    }
    out[n] = 0;
    return 1;
}

// ---- hostfile.c: host files and directories. Paths are UTF-8; '/' separates on every host. Every call that reaches the
// operating system runs without the guest thread's virtual core (rt_kernel_block_begin), so other guest threads run.
typedef struct HostFile HostFile;
enum { HF_OK, HF_NOT_FOUND, HF_EXISTS, HF_NOT_EMPTY, HF_BUSY, HF_ERROR };   // what a failed call returns
enum { HF_MISSING = -1, HF_FILE, HF_DIR, HF_LINK };                          // hf_kind; a link is a symbolic link or reparse point

HostFile *hf_open(const char *path, int write, int *err);   // an existing regular file; for reading, or for both
HostFile *hf_create(const char *path, int *err);            // a new file for writing; HF_EXISTS when something is there
void hf_close(HostFile *f);
int64_t hf_pread(HostFile *f, void *buf, uint32_t n, uint64_t off);          // bytes read, 0 at the end, -1 on an error
int64_t hf_pwrite(HostFile *f, const void *buf, uint32_t n, uint64_t off);   // bytes written, -1 on an error
int hf_size(HostFile *f, uint64_t *size);                   // 1 on success
int hf_set_size(HostFile *f, uint64_t size);                // HF_OK or an HF_ error
int hf_sync(HostFile *f);                                   // the file's data is on the disk: 1 on success
int hf_kind(const char *path);
int hf_mkdir(const char *path);                             // HF_EXISTS when something is there
int hf_rmdir(const char *path);
int hf_unlink(const char *path);
int hf_rename(const char *from, const char *to);            // never replaces: HF_EXISTS
int hf_replace(const char *from, const char *to);           // replaces an existing file, atomically and durably (savetxn.c)
// add(name, kind, size, ctx) for every entry of dir but "." and ".."; it returns 0 to stop. HF_OK or an HF_ error.
int hf_list(const char *dir, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx);

// ---- romfs.c: the RomFS pack, read-only. A path is what follows a mount name's colon.
// Handles are ROMFS_HANDLE_BASE + slot; host files and directories use other ranges (ov_fs.c).
#define ROMFS_HANDLE_BASE 0x100000000ull
// Open the pack's index and image once, on the first call; thread-safe. 1 when ready; else 0 with the reason in why.
int romfs_ready(char *why, size_t cap);
int romfs_handle(uint64_t h);                                      // h is a RomFS file handle value
uint32_t romfs_open(uint64_t *out, const char *rest);              // a FS_R_ result; FS_R_NO_SLOT when no handle is free
uint32_t romfs_size(int64_t *out, uint64_t h);
int64_t romfs_pread(uint64_t h, void *buf, uint32_t n, uint64_t off);   // bytes read, 0 at the end of the file, -1 on an error
int romfs_close(uint64_t h);                                       // 0 when h was not open
uint32_t romfs_entry_type(int *out, const char *rest);             // FS_ENTRY_FILE or FS_ENTRY_DIRECTORY
// add(name, kind, size, ctx) (kind: HF_FILE or HF_DIR) for each direct child of the directory, in path order, each
// subdirectory once; it returns 0 to stop. The pack stores files only, so a folder without a file under it does not exist.
uint32_t romfs_list(const char *rest, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx);
