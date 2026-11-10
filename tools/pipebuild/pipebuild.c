// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// pipebuild: the program that prepare's pipeline step (tools/prepare/pipelines.py, the pipeline step) runs to build pipelines.
// It is linked with runtime/core, runtime/nvn and runtime/gpu and calls the very functions the Vulkan back end calls at
// launch (oot2/vk_pipeline.h), so a pipeline built here is served from the cache to the game later, whatever the
// compiler flags of the two programs.
//
//   pipebuild serve --pack FILE [--variants FILE] [--cache FILE] --segments DIR
//       one worker. It opens the device the runtime would open (the OOT2_VK_* settings of oot2/gpu_vk.h: device,
//       validation, and OOT2_VK_DRIVER_CACHE=off for a cold measurement), the pack and an empty pipeline cache (seeded
//       from --cache when that file is valid for this device and driver), then answers one command per stdin line, one
//       reply per stdout line, everything else on stderr:
//         build vs|ps|cs KEY   ok KIND KEY NS HIT MODULE_BYTES   the vertex library, the fragment library or the compute
//                                                                pipeline of the job KEY; NS is the driver's wall time
//                                                                and HIT 1 when the cache served it
//         flush NAME           saved NAME BYTES                   DIR/NAME.bin holds what was built since the last flush
//         size                 size BYTES                         what the cache holds now
//         stats                stats ...                          the builder's counters
//         quit                 bye
//       A reply "fail ..." names a command it could not take. A module the builder cannot build is a named fatal, as in
//       the game: the process prints FATAL: and exits with 70, and the caller reads the reason from stderr.
//   pipebuild merge --out FILE [--list FILE] [SEGMENT...]
//       one cache file out of segments (the paths in --list, one per line, then the arguments), written through a
//       temporary file and a rename. It prints "merged SEGMENTS INPUT_BYTES OUTPUT_BYTES".
//
// Exit codes: 0 done, 2 bad command line, 3 a refusal (a pack that does not open, a segment that is not for this
// device), 70 a named fatal of the runtime.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#define OFF_SEEK _fseeki64
#define OFF_TELL _ftelli64
#define SYNC_FILE(f) _commit(_fileno(f))
#else
#include <unistd.h>
#define OFF_SEEK fseeko
#define OFF_TELL ftello
#define SYNC_FILE(f) fsync(fileno(f))
#endif
#include <time.h>

#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_hmma.h"
#include "oot2/vk_pipeline.h"

#define LINE_MAX_BYTES 1024
#define PATH_MAX_BYTES 480

static void reply(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    putchar('\n');
    fflush(stdout);
}

static int usage(const char *why) {
    fprintf(stderr, "pipebuild: %s\n", why);
    fprintf(stderr, "usage: pipebuild serve --pack FILE [--variants FILE] [--cache FILE] --segments DIR\n"
                    "       pipebuild merge --out FILE SEGMENT...\n");
    return 2;
}

static int plain_name(const char *s, size_t most) {
    size_t n = strlen(s);
    if (n == 0 || n > most) return 0;
    for (; *s; s++) {
        char c = *s;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return 0;
    }
    return 1;
}

static void hex(const uint8_t *b, size_t n, char *out) {
    for (size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", b[i]);
}

static const char *cache_state_name(VkpCacheState s) {
    switch (s) {
    case VKP_CACHE_NONE: return "none";
    case VKP_CACHE_LOADED: return "loaded";
    case VKP_CACHE_OTHER_DRIVER: return "other_driver";
    default: return "unreadable";
    }
}

static void print_ready(void) {
    const VkPhysicalDeviceProperties *p = &gpu_vk.facts.props;
    char uuid[2 * VK_UUID_SIZE + 1];
    hex(p->pipelineCacheUUID, VK_UUID_SIZE, uuid);
    // zero_workgroup says what this runtime does to a compute module before the driver sees it: a cache built by a
    // runtime that did not zero them holds other modules than the ones a launch asks for (tools/prepare/pipelines.py).
    // hmma says the form the start-up probe chose for the tensor programs (oot2/vk_hmma.h): the same holds of a cache
    // built in the other form.
    reply("ready vendor=%04x device=%04x driver=%08x api=%u.%u.%u cache_uuid=%s cache=%s seeded=%llu zero_workgroup=%d "
          "hmma=%s name=%s",
          p->vendorID, p->deviceID, p->driverVersion, VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion),
          VK_API_VERSION_PATCH(p->apiVersion), uuid, cache_state_name(vkp_cache_state()),
          (unsigned long long)vkp_cache_size(), vkp_zero_workgroup(), vkh_native() ? "native" : "helper", p->deviceName);
}

// A named fatal of the runtime (a module that does not fit, a capability the device lacks, a failed Vulkan call) stops the
// process, and what the worker built since its last flush would die with it. The stop hook saves that cache as a segment
// of its own first and tells the caller in place of the reply it was waiting for: "died NAME BYTES LINE" (NAME is - when
// nothing was built since the last flush).
static char g_segments[PATH_MAX_BYTES + 1];
static int g_built_since_flush;

static void on_stop(int code, const char *line) {
    (void)code;
    char name[64] = "-";
    size_t bytes = 0;
    if (g_built_since_flush) {
        char path[PATH_MAX_BYTES + 96], why[256];
#ifdef _WIN32
        unsigned long pid = (unsigned long)GetCurrentProcessId();
#else
        unsigned long pid = (unsigned long)getpid();
#endif
        // The process id alone could repeat (Windows reuses them quickly) while the segments of an earlier run stay.
        snprintf(name, sizeof name, "seg-d%lu-%llu", pid, (unsigned long long)time(NULL));
        snprintf(path, sizeof path, "%s/%s.bin", g_segments, name);
        if (!vkp_cache_flush(path, &bytes, why, sizeof why)) snprintf(name, sizeof name, "-");
    }
    reply("died %s %llu %s", name, (unsigned long long)bytes, line ? line : "");
}

static int cmd_serve(int argc, char **argv) {
    const char *pack = NULL, *variants = NULL, *cache = NULL, *segments = NULL;
    for (int i = 0; i < argc; i += 2) {
        if (i + 1 >= argc) return usage("an option without a value");
        if (!strcmp(argv[i], "--pack")) pack = argv[i + 1];
        else if (!strcmp(argv[i], "--variants")) variants = argv[i + 1];
        else if (!strcmp(argv[i], "--cache")) cache = argv[i + 1];
        else if (!strcmp(argv[i], "--segments")) segments = argv[i + 1];
        else return usage("an option serve does not take");
    }
    if (!pack || !segments) return usage("serve needs --pack and --segments");
    if (strlen(segments) > PATH_MAX_BYTES) return usage("--segments is too long");
    gpu_vk_open();
    char why[512];
    if (!vkp_store_open(pack, variants, why, sizeof why)) {
        fprintf(stderr, "pipebuild: the pack did not open: %s\n", why);
        return 3;
    }
    vkp_open(cache);
    snprintf(g_segments, sizeof g_segments, "%s", segments);
    rt_set_stop_hook(on_stop);
    print_ready();
    char line[LINE_MAX_BYTES];
    while (fgets(line, sizeof line, stdin)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        char word[16] = "", arg1[64] = "", arg2[LINE_MAX_BYTES] = "";
        int got = sscanf(line, "%15s %63s %1023s", word, arg1, arg2);
        if (got < 1) continue;
        if (!strcmp(word, "build") && got == 3) {
            int what = !strcmp(arg1, "vs") ? VKP_WARM_VERTEX : !strcmp(arg1, "ps") ? VKP_WARM_PIXEL
                       : !strcmp(arg1, "cs") ? VKP_WARM_COMPUTE : -1;
            if (what < 0 || !plain_name(arg2, 31)) {
                reply("fail %s %s a build takes vs, ps or cs and a job key of 1 to 31 plain characters", arg1, arg2);
                continue;
            }
            VkpStage stage = {.key = arg2};
            VkpWarm w;
            vkp_warm(what, &stage, &w);
            g_built_since_flush = 1;
            reply("ok %s %s %llu %d %u", arg1, arg2, (unsigned long long)w.ns, w.hit, w.module_bytes);
        } else if (!strcmp(word, "flush") && got == 2) {
            char path[PATH_MAX_BYTES + 64];
            if (!plain_name(arg1, 48)) {
                reply("fail flush %s a segment name is 1 to 48 plain characters", arg1);
                continue;
            }
            snprintf(path, sizeof path, "%s/%s.bin", segments, arg1);
            size_t bytes = 0;
            if (vkp_cache_flush(path, &bytes, why, sizeof why)) {
                g_built_since_flush = 0;
                reply("saved %s %llu", arg1, (unsigned long long)bytes);
            } else {
                reply("fail flush %s %s", arg1, why);
            }
        } else if (!strcmp(word, "size")) {
            reply("size %llu", (unsigned long long)vkp_cache_size());
        } else if (!strcmp(word, "stats")) {
            VkpStats s;
            vkp_stats(&s);
            reply("stats libraries %llu library_hits %llu stage_libraries %llu stage_library_hits %llu computes %llu "
                  "compute_hits %llu library_ms %.1f compute_ms %.1f",
                  (unsigned long long)s.libraries, (unsigned long long)s.library_hits, (unsigned long long)s.stage_libraries,
                  (unsigned long long)s.stage_library_hits, (unsigned long long)s.computes,
                  (unsigned long long)s.compute_hits, (double)s.build_ns[0] / 1e6, (double)s.build_ns[2] / 1e6);
        } else if (!strcmp(word, "quit")) {
            break;
        } else {
            reply("fail %s unknown command", word);
        }
    }
    vkp_close();
    vkp_store_close();
    gpu_vk_close();
    uint64_t errors, warnings;
    gpu_vk_validation_counts(&errors, &warnings);
    reply("bye validation_errors %llu validation_warnings %llu", (unsigned long long)errors,
          (unsigned long long)warnings);
    return 0;
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    void *data = NULL;
    if (OFF_SEEK(f, 0, SEEK_END) == 0) {   // 64-bit offsets: a segment or a merged cache can pass 2 GiB
        long long n = (long long)OFF_TELL(f);
        if (n > 0 && OFF_SEEK(f, 0, SEEK_SET) == 0 && (data = malloc((size_t)n)) != NULL) {
            if (fread(data, 1, (size_t)n, f) == (size_t)n) *size = (size_t)n;
            else free(data), data = NULL;
        }
    }
    fclose(f);
    return data;
}

static int cmd_merge(int argc, char **argv) {
    const char *out = NULL, *list = NULL;
    int first = 0;
    for (; first < argc; first += 2) {
        if (!strcmp(argv[first], "--out")) {
            if (first + 1 >= argc) return usage("--out without a file");
            out = argv[first + 1];
        } else if (!strcmp(argv[first], "--list")) {
            if (first + 1 >= argc) return usage("--list without a file");
            list = argv[first + 1];
        } else {
            break;
        }
    }
    if (!out) return usage("merge needs --out FILE");
    if (strlen(out) > PATH_MAX_BYTES) return usage("--out is too long");
    // The segments come from --list (one path per line: a long run leaves more segments than a command line holds)
    // and from the arguments after the options.
    char **paths = argv + first;
    int npaths = argc - first;
    char **listed = NULL;
    if (list) {
        size_t size = 0;
        char *text = read_file(list, &size);
        if (!text) return usage("--list names a file that cannot be read");
        char *copy = realloc(text, size + 1);
        if (!copy) return usage("out of memory for --list");
        text = copy;
        text[size] = 0;
        int cap = npaths + 16, n = 0;
        listed = malloc((size_t)cap * sizeof *listed);
        for (int i = 0; i < npaths; i++) listed[n++] = paths[i];
        for (char *line = strtok(text, "\r\n"); line; line = strtok(NULL, "\r\n")) {
            if (!*line) continue;
            if (n == cap) {
                cap *= 2;
                listed = realloc(listed, (size_t)cap * sizeof *listed);
            }
            listed[n++] = line;
        }
        paths = listed;
        npaths = n;
    }
    gpu_vk_open();
    VkPipelineCacheCreateInfo empty = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache dest;
    GPU_VK_OK(vkCreatePipelineCache(gpu_vk.device, &empty, NULL, &dest), "vkCreatePipelineCache");
    unsigned long long input = 0;
    int merged = 0;
    for (int i = 0; i < npaths; i++) {
        size_t size = 0;
        void *data = read_file(paths[i], &size);
        if (!data) {
            fprintf(stderr, "pipebuild: cannot read the segment %s\n", paths[i]);
            return 3;
        }
        if (vkp_cache_judge(data, size, &gpu_vk.facts.props) != VKP_CACHE_LOADED) {
            fprintf(stderr, "pipebuild: the segment %s is not a cache of this device and driver\n", paths[i]);
            return 3;
        }
        VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = size,
                                        .pInitialData = data};
        VkPipelineCache part;
        GPU_VK_OK(vkCreatePipelineCache(gpu_vk.device, &ci, NULL, &part), "vkCreatePipelineCache (a segment)");
        GPU_VK_OK(vkMergePipelineCaches(gpu_vk.device, dest, 1, &part), "vkMergePipelineCaches");
        vkDestroyPipelineCache(gpu_vk.device, part, NULL);
        free(data);
        input += size;
        merged++;
    }
    // Write dest through the builder's own writer: open the builder on the merged cache by saving to a temporary file.
    size_t bytes = 0;
    GPU_VK_OK(vkGetPipelineCacheData(gpu_vk.device, dest, &bytes, NULL), "vkGetPipelineCacheData");
    void *blob = malloc(bytes ? bytes : 1);
    if (!blob) {
        fprintf(stderr, "pipebuild: out of memory for %zu bytes of cache\n", bytes);
        return 3;
    }
    VkResult r = vkGetPipelineCacheData(gpu_vk.device, dest, &bytes, blob);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) gpu_vk_failed(r, "vkGetPipelineCacheData");
    char tmp[PATH_MAX_BYTES + 8];
    snprintf(tmp, sizeof tmp, "%s.tmp", out);
    FILE *f = fopen(tmp, "wb");
    int ok = f && fwrite(blob, 1, bytes, f) == bytes;
    if (f && ok) ok = fflush(f) == 0 && SYNC_FILE(f) == 0;
    if (f && fclose(f) != 0) ok = 0;
    free(blob);
    if (ok) {
#ifdef _WIN32
        ok = MoveFileExA(tmp, out, MOVEFILE_REPLACE_EXISTING) != 0;
#else
        ok = rename(tmp, out) == 0;
#endif
    }
    if (!ok) {
        remove(tmp);
        fprintf(stderr, "pipebuild: cannot write %s\n", out);
        return 3;
    }
    vkDestroyPipelineCache(gpu_vk.device, dest, NULL);
    gpu_vk_close();
    reply("merged %d %llu %llu", merged, input, (unsigned long long)bytes);
    return 0;
}

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);   // lines end as the caller wrote them
#endif
    if (argc < 2) return usage("a command is needed");
    if (!strcmp(argv[1], "serve")) return cmd_serve(argc - 2, argv + 2);
    if (!strcmp(argv[1], "merge")) return cmd_merge(argc - 2, argv + 2);
    return usage("unknown command");
}
