// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Host files and directories under the nn::fs overrides (fs.h): positional reads and writes, the calls that make,
// rename and remove, and directory listings. Each call that reaches the operating system gives the guest thread's
// virtual core up while it runs, as the console's thread would wait on the file system service.
//
// Adapted from oot-pc-recomp runtime/platform/win32.c and runtime/platform/linux.c at
// 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17 (files with no licence header there: that repository's GPL-3.0-or-later):
// plat_file_pread, plat_file_pwrite, plat_file_set_size, plat_file_fsync, plat_fs_open, plat_fs_create, plat_fs_kind,
// plat_fs_mkdir, plat_fs_rmdir, plat_fs_unlink, plat_fs_rename, plat_fs_list and their error classes. Changed here: one
// HostFile type for reads and writes; every call blocks the virtual core; no hard links, durable renames or busy-file
// reports; a listing and a rename report only what nn::fs needs; Windows paths are converted from UTF-8 with '/'
// turned into '\' and a name that is not valid UTF-8 is refused.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>
#else
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs.h"

// Run an operating-system call without the virtual core; the value of the call is the value of the expression.
#define BLOCKING(type, call) ({ int tok_ = rt_kernel_block_begin(); type res_ = (call); rt_kernel_block_end(tok_); res_; })

#ifdef _WIN32
#define WPATH 2048

static int widen(const char *s, wchar_t *w) {
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, WPATH)) return 0;
    for (wchar_t *q = w; *q; q++)
        if (*q == L'/') *q = L'\\';
    return 1;
}

static int os_err(DWORD e) {
    switch (e) {
    case ERROR_FILE_NOT_FOUND: case ERROR_PATH_NOT_FOUND: case ERROR_DIRECTORY: return HF_NOT_FOUND;
    case ERROR_FILE_EXISTS: case ERROR_ALREADY_EXISTS: return HF_EXISTS;
    case ERROR_DIR_NOT_EMPTY: return HF_NOT_EMPTY;
    case ERROR_ACCESS_DENIED: case ERROR_SHARING_VIOLATION: return HF_BUSY;
    default: return HF_ERROR;
    }
}

// Handles are opened for overlapped I/O: a read or a write names its offset and each thread waits on an event of its
// own, so threads that use one handle at once share neither the file pointer nor an event.
static HostFile *os_open_as(const char *path, DWORD access, DWORD share, DWORD how, int *err) {
    wchar_t w[WPATH];
    if (!widen(path, w)) {
        if (err) *err = HF_ERROR;
        return NULL;
    }
    HANDLE h = CreateFileW(w, access, share, NULL, how, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        if (err) *err = os_err(GetLastError());
        return NULL;
    }
    return (HostFile *)h;
}

static HostFile *os_open(const char *path, int write, int *err) {
    return os_open_as(path, GENERIC_READ | (write ? GENERIC_WRITE : 0),
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, OPEN_EXISTING, err);
}

static HostFile *os_create(const char *path, int *err) { return os_open_as(path, GENERIC_WRITE, 0, CREATE_NEW, err); }

static void os_close(HostFile *f) { CloseHandle((HANDLE)f); }

static int64_t os_pread(HostFile *f, void *buf, uint32_t n, uint64_t off) {
    static _Thread_local HANDLE ev;
    if (!ev && !(ev = CreateEventW(NULL, TRUE, FALSE, NULL))) return -1;
    OVERLAPPED ov = {.Offset = (DWORD)off, .OffsetHigh = (DWORD)(off >> 32), .hEvent = ev};
    DWORD got = 0;
    if (ReadFile((HANDLE)f, buf, n, &got, &ov)) return got;
    DWORD e = GetLastError();
    if (e == ERROR_IO_PENDING) {
        if (GetOverlappedResult((HANDLE)f, &ov, &got, TRUE)) return got;
        e = GetLastError();
    }
    return e == ERROR_HANDLE_EOF ? 0 : -1;
}

static int64_t os_pwrite(HostFile *f, const void *buf, uint32_t n, uint64_t off) {
    static _Thread_local HANDLE ev;
    if (!ev && !(ev = CreateEventW(NULL, TRUE, FALSE, NULL))) return -1;
    OVERLAPPED ov = {.Offset = (DWORD)off, .OffsetHigh = (DWORD)(off >> 32), .hEvent = ev};
    DWORD put = 0;
    if (WriteFile((HANDLE)f, buf, n, &put, &ov)) return put;
    if (GetLastError() == ERROR_IO_PENDING && GetOverlappedResult((HANDLE)f, &ov, &put, TRUE)) return put;
    return -1;
}

static int os_size(HostFile *f, uint64_t *size) {
    LARGE_INTEGER len;
    if (!GetFileSizeEx((HANDLE)f, &len)) return 0;
    *size = (uint64_t)len.QuadPart;
    return 1;
}

static int os_set_size(HostFile *f, uint64_t size) {
    FILE_END_OF_FILE_INFO eof = {.EndOfFile.QuadPart = (LONGLONG)size};
    return SetFileInformationByHandle((HANDLE)f, FileEndOfFileInfo, &eof, sizeof eof) ? HF_OK : os_err(GetLastError());
}

static int os_sync(HostFile *f) { return FlushFileBuffers((HANDLE)f) != 0; }

static int os_kind(const char *path) {
    wchar_t w[WPATH];
    if (!widen(path, w)) return HF_MISSING;
    DWORD a = GetFileAttributesW(w);
    if (a == INVALID_FILE_ATTRIBUTES) return HF_MISSING;
    return a & FILE_ATTRIBUTE_REPARSE_POINT ? HF_LINK : a & FILE_ATTRIBUTE_DIRECTORY ? HF_DIR : HF_FILE;
}

static int os_mkdir(const char *path) {
    wchar_t w[WPATH];
    if (!widen(path, w)) return HF_ERROR;
    return CreateDirectoryW(w, NULL) ? HF_OK : os_err(GetLastError());
}

static int os_rmdir(const char *path) {
    wchar_t w[WPATH];
    if (!widen(path, w)) return HF_ERROR;
    return RemoveDirectoryW(w) ? HF_OK : os_err(GetLastError());
}

// A read-only file goes as any other, as it does under POSIX, where deleting needs the directory's write access.
static int os_unlink(const char *path) {
    wchar_t w[WPATH];
    if (!widen(path, w)) return HF_ERROR;
    DWORD a = GetFileAttributesW(w);
    if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_READONLY)) SetFileAttributesW(w, a & ~(DWORD)FILE_ATTRIBUTE_READONLY);
    return DeleteFileW(w) ? HF_OK : os_err(GetLastError());
}

// MoveFileEx without MOVEFILE_REPLACE_EXISTING fails when the target exists.
static int os_rename(const char *from, const char *to) {
    wchar_t a[WPATH], b[WPATH];
    if (!widen(from, a) || !widen(to, b)) return HF_ERROR;
    return MoveFileExW(a, b, 0) ? HF_OK : os_err(GetLastError());
}

// A file moved over another one in one step: a reader sees the old file or the new one, never neither. The rename carries
// POSIX semantics (Windows 10 1709 and later), so it also replaces a file some handle still has open (every handle here is
// opened with FILE_SHARE_DELETE); where the system does not know the call, MoveFileEx replaces a file nobody has open. The
// new file's buffers are flushed before it returns.
static int os_replace(const char *from, const char *to) {
    wchar_t a[WPATH], b[WPATH], full[WPATH];
    if (!widen(from, a) || !widen(to, b)) return HF_ERROR;
    DWORD n = GetFullPathNameW(b, WPATH, full, NULL);
    if (!n || n >= WPATH) return HF_ERROR;
    HANDLE h = CreateFileW(a, DELETE | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return os_err(GetLastError());
    _Alignas(8) char buf[sizeof(FILE_RENAME_INFO) + WPATH * sizeof(wchar_t)];
    FILE_RENAME_INFO *ri = (FILE_RENAME_INFO *)buf;
    memset(ri, 0, sizeof *ri);
    ri->Flags = 0x1 | 0x2;   // FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS
    ri->FileNameLength = n * sizeof(wchar_t);
    memcpy(ri->FileName, full, (n + 1) * sizeof(wchar_t));
    int err = HF_OK;
    if (!SetFileInformationByHandle(h, (FILE_INFO_BY_HANDLE_CLASS)22 /* FileRenameInfoEx */, ri,
                                    (DWORD)(sizeof *ri + n * sizeof(wchar_t)))) {
        DWORD e = GetLastError();
        err = os_err(e);
        if (e == ERROR_INVALID_PARAMETER || e == ERROR_INVALID_FUNCTION || e == ERROR_NOT_SUPPORTED) err = -1;
    }
    if (!err) FlushFileBuffers(h);
    CloseHandle(h);
    if (err != -1) return err;
    return MoveFileExW(a, b, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? HF_OK : os_err(GetLastError());
}

static int os_list(const char *dir, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    wchar_t wd[WPATH], pat[WPATH + 4];
    if (!widen(dir, wd)) return HF_ERROR;
    wcscpy(pat, wd);
    wcscat(pat, L"\\*");
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return os_err(GetLastError());
    do {
        if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
        char name[1024];
        if (!WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, name, sizeof name, NULL, NULL)) continue;
        DWORD a = fd.dwFileAttributes;
        int kind = a & FILE_ATTRIBUTE_REPARSE_POINT ? HF_LINK : a & FILE_ATTRIBUTE_DIRECTORY ? HF_DIR : HF_FILE;
        if (!add(name, kind, (uint64_t)fd.nFileSizeHigh << 32 | fd.nFileSizeLow, ctx)) break;
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return HF_OK;
}

#else   // POSIX: a HostFile is the descriptor + 1, so descriptor 0 is not NULL

static int os_err(int e) {
    switch (e) {
    case ENOENT: case ENOTDIR: return HF_NOT_FOUND;
    case EEXIST: return HF_EXISTS;
    case ENOTEMPTY: return HF_NOT_EMPTY;
    case EBUSY: case ETXTBSY: return HF_BUSY;
    default: return HF_ERROR;
    }
}

static int fd_of(HostFile *f) { return (int)(intptr_t)f - 1; }

static HostFile *os_open(const char *path, int write, int *err) {
    struct stat st;
    int fd = open(path, (write ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd >= 0 && (fstat(fd, &st) || !S_ISREG(st.st_mode))) {   // a directory opens for reading: not a file here
        close(fd);
        fd = -1;
        errno = ENOENT;
    }
    if (fd < 0) {
        if (err) *err = os_err(errno);
        return NULL;
    }
    return (HostFile *)(intptr_t)(fd + 1);
}

static HostFile *os_create(const char *path, int *err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) {
        if (err) *err = os_err(errno);
        return NULL;
    }
    return (HostFile *)(intptr_t)(fd + 1);
}

static void os_close(HostFile *f) { close(fd_of(f)); }

static int64_t os_pread(HostFile *f, void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pread(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k;
}

static int64_t os_pwrite(HostFile *f, const void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pwrite(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k;
}

static int os_size(HostFile *f, uint64_t *size) {
    struct stat st;
    if (fstat(fd_of(f), &st)) return 0;
    *size = (uint64_t)st.st_size;
    return 1;
}

static int os_set_size(HostFile *f, uint64_t size) { return ftruncate(fd_of(f), (off_t)size) ? os_err(errno) : HF_OK; }

static int os_sync(HostFile *f) { return fsync(fd_of(f)) == 0; }

static int os_kind(const char *path) {
    struct stat st;
    if (lstat(path, &st)) return HF_MISSING;
    return S_ISLNK(st.st_mode) ? HF_LINK : S_ISDIR(st.st_mode) ? HF_DIR : HF_FILE;
}

static int os_mkdir(const char *path) { return mkdir(path, 0755) ? os_err(errno) : HF_OK; }

static int os_rmdir(const char *path) {
    struct stat st;
    if (!lstat(path, &st) && S_ISLNK(st.st_mode)) return unlink(path) ? os_err(errno) : HF_OK;
    return rmdir(path) ? os_err(errno) : HF_OK;
}

static int os_unlink(const char *path) { return unlink(path) ? os_err(errno) : HF_OK; }

// rename replaces an existing target, so a target that is there is refused first; another process could race that
// check, but the folders under the game folder are this process's own.
static int os_rename(const char *from, const char *to) {
    if (os_kind(to) != HF_MISSING) return HF_EXISTS;
    return rename(from, to) ? os_err(errno) : HF_OK;
}

// rename(2) replaces the target in one step; the folder is then synced so the new name is on the disk too.
static int os_replace(const char *from, const char *to) {
    if (rename(from, to)) return os_err(errno);
    char dir[4096];
    if (snprintf(dir, sizeof dir, "%s", to) >= (int)sizeof dir) return HF_OK;
    char *slash = strrchr(dir, '/');
    if (!slash) snprintf(dir, sizeof dir, ".");
    else if (slash == dir) dir[1] = 0;
    else *slash = 0;
    int fd = open(dir, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
    return HF_OK;
}

static int os_list(const char *dir, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return os_err(errno);
    for (struct dirent *e; (e = readdir(d));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW)) continue;
        int kind = S_ISLNK(st.st_mode) ? HF_LINK : S_ISDIR(st.st_mode) ? HF_DIR : HF_FILE;
        if (!add(e->d_name, kind, kind == HF_FILE ? (uint64_t)st.st_size : 0, ctx)) break;
    }
    closedir(d);
    return HF_OK;
}
#endif

HostFile *hf_open(const char *path, int write, int *err) { return BLOCKING(HostFile *, os_open(path, write, err)); }
HostFile *hf_create(const char *path, int *err) { return BLOCKING(HostFile *, os_create(path, err)); }

void hf_close(HostFile *f) {
    int tok = rt_kernel_block_begin();
    os_close(f);
    rt_kernel_block_end(tok);
}

int64_t hf_pread(HostFile *f, void *buf, uint32_t n, uint64_t off) { return BLOCKING(int64_t, os_pread(f, buf, n, off)); }
int64_t hf_pwrite(HostFile *f, const void *buf, uint32_t n, uint64_t off) {
    return BLOCKING(int64_t, os_pwrite(f, buf, n, off));
}
int hf_size(HostFile *f, uint64_t *size) { return BLOCKING(int, os_size(f, size)); }
int hf_set_size(HostFile *f, uint64_t size) { return BLOCKING(int, os_set_size(f, size)); }
int hf_sync(HostFile *f) { return BLOCKING(int, os_sync(f)); }
int hf_kind(const char *path) { return BLOCKING(int, os_kind(path)); }
int hf_mkdir(const char *path) { return BLOCKING(int, os_mkdir(path)); }
int hf_rmdir(const char *path) { return BLOCKING(int, os_rmdir(path)); }
int hf_unlink(const char *path) { return BLOCKING(int, os_unlink(path)); }
int hf_rename(const char *from, const char *to) { return BLOCKING(int, os_rename(from, to)); }
int hf_replace(const char *from, const char *to) { return BLOCKING(int, os_replace(from, to)); }

int hf_list(const char *dir, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    return BLOCKING(int, os_list(dir, add, ctx));
}
