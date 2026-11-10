// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkhmmacost: the GPU side of the HMMA cost measurement (architecture D20b and
// spike S8): it runs one compute module many times with timestamp queries and
// reads its output buffers back, so the caller (tools/hmmacost/hmma_cost.py)
// can check them against its CPU reference and divide the time by the work.
//
//   vkhmmacost info              one JSON line: device, driver, subgroup size,
//                                timestamp period, and the cooperative matrix
//                                shapes the device lists (VK_KHR_cooperative_matrix)
//   vkhmmacost run JOB           runs the job file, one JSON line on stdout
//
// The job file, one directive per line (paths have no spaces):
//   validation on|off       module PATH      groups X Y Z
//   warmup N                reps N           chunk N (dispatches per submit)
//   full_subgroups on|off   (create the pipeline with the full-subgroups requirement, as the game's builder does for a
//                            module with cooperative matrices; off by default)
//   buffer SET BINDING ubo|ssbo SIZE [in=PATH] [out=PATH]
//   push PATH               (push constant bytes for the compute stage)
// A buffer is device-local; `in` fills it once through a staging copy (a short
// file leaves the rest zero), `out` is read back after the last dispatch. Sets
// are ordinary descriptor sets, one per set number up to the highest named.
// Each dispatch sits between two timestamps written at the bottom of the pipe,
// with a compute-to-compute barrier before it, so dispatches never overlap.
// The JSON line holds `ms` (every measured dispatch, in order), the device and
// driver, the module's pipeline build time and, with validation on, the counts
// of validation errors and warnings.
//
// Every feature the device offers is enabled but robustness and protected
// memory (vkpipecost's rule), with VK_KHR_cooperative_matrix when present.
// Third-party implicit layers are the caller's to disable
// (VK_LOADER_LAYERS_DISABLE=~implicit~). Windows only. No game data: modules
// and buffer contents are written by the caller.
//
// Exit codes: 0 measured, 1 the job file is wrong or validation reported
// errors, 2 Vulkan lacks something the job needs, 3 a Vulkan call failed or
// the device was lost.

#define VK_NO_PROTOTYPES
#include "volk.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define MAX_PATH_LEN 1024
#define MAX_BUFFERS 16
#define MAX_SETS 4
#define MAX_REPS 100000

static VkInstance inst;
static VkPhysicalDevice gpu;
static VkDevice dev;
static VkQueue queue;
static uint32_t family = UINT32_MAX;
static VkPhysicalDeviceProperties gpu_props;
static VkPhysicalDeviceMemoryProperties mem_props;
static uint32_t subgroup_size;
static int has_coop;
static unsigned val_errors, val_warnings;
static VkDebugUtilsMessengerEXT messenger;

__attribute__((noreturn, format(printf, 2, 3))) static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vkhmmacost: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stdout);
    exit(code);
}

#define VK_CHECK(call)                                                                   \
    do {                                                                                 \
        VkResult r_ = (call);                                                            \
        if (r_ != VK_SUCCESS) die(3, "%s failed: %d (%s:%d)", #call, (int)r_, __FILE__, __LINE__); \
    } while (0)

static double now_ms(void) {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart * 1000.0 / (double)f.QuadPart;
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    void *p = malloc(n > 0 ? (size_t)n : 1);
    if (!p || (n > 0 && fread(p, 1, (size_t)n, f) != (size_t)n)) {
        free(p);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)(n > 0 ? n : 0);
    return p;
}

// ---- the job file ---------------------------------------------------------------

typedef struct {
    uint32_t set, binding;
    VkDescriptorType type;
    VkDeviceSize size;
    char in[MAX_PATH_LEN], out[MAX_PATH_LEN];
    VkBuffer buf;
    VkDeviceMemory mem;
} Buffer;

typedef struct {
    int validation, full_subgroups;
    char module[MAX_PATH_LEN], push[MAX_PATH_LEN];
    uint32_t groups[3], warmup, reps, chunk;
    Buffer b[MAX_BUFFERS];
    uint32_t nb;
} Job;

static void job_read(const char *path, Job *job) {
    memset(job, 0, sizeof *job);
    job->groups[0] = job->groups[1] = job->groups[2] = 1;
    job->reps = 100;
    job->warmup = 10;
    job->chunk = 16;
    FILE *f = fopen(path, "rb");
    if (!f) die(1, "cannot open the job file %s", path);
    char line[4096];
    unsigned lineno = 0;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char w[64] = {0}, t[6][MAX_PATH_LEN];
        memset(t, 0, sizeof t);
        int k = sscanf(line, "%63s %1023s %1023s %1023s %1023s %1023s %1023s", w, t[0], t[1], t[2], t[3], t[4], t[5]);
        if (!strcmp(w, "validation") && k == 2) job->validation = !strcmp(t[0], "on");
        else if (!strcmp(w, "full_subgroups") && k == 2) job->full_subgroups = !strcmp(t[0], "on");
        else if (!strcmp(w, "module") && k == 2) snprintf(job->module, sizeof job->module, "%s", t[0]);
        else if (!strcmp(w, "push") && k == 2) snprintf(job->push, sizeof job->push, "%s", t[0]);
        else if (!strcmp(w, "groups") && k == 4) for (int i = 0; i < 3; i++) job->groups[i] = (uint32_t)strtoul(t[i], NULL, 0);
        else if (!strcmp(w, "warmup") && k == 2) job->warmup = (uint32_t)strtoul(t[0], NULL, 0);
        else if (!strcmp(w, "reps") && k == 2) job->reps = (uint32_t)strtoul(t[0], NULL, 0);
        else if (!strcmp(w, "chunk") && k == 2) job->chunk = (uint32_t)strtoul(t[0], NULL, 0);
        else if (!strcmp(w, "buffer") && k >= 5) {
            if (job->nb == MAX_BUFFERS) die(1, "line %u: too many buffers", lineno);
            Buffer *b = &job->b[job->nb++];
            b->set = (uint32_t)strtoul(t[0], NULL, 0);
            b->binding = (uint32_t)strtoul(t[1], NULL, 0);
            if (!strcmp(t[2], "ubo")) b->type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            else if (!strcmp(t[2], "ssbo")) b->type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            else die(1, "line %u: buffer type %s", lineno, t[2]);
            b->size = strtoull(t[3], NULL, 0);
            if (b->set >= MAX_SETS || !b->size) die(1, "line %u: set %u or size 0", lineno, b->set);
            for (int i = 4; i < k - 1; i++) {
                if (!strncmp(t[i], "in=", 3)) snprintf(b->in, sizeof b->in, "%s", t[i] + 3);
                else if (!strncmp(t[i], "out=", 4)) snprintf(b->out, sizeof b->out, "%s", t[i] + 4);
                else die(1, "line %u: %s", lineno, t[i]);
            }
        } else die(1, "line %u not understood: %s", lineno, line);
    }
    fclose(f);
    if (!job->module[0]) die(1, "the job names no module");
    if (!job->reps || job->reps > MAX_REPS || !job->chunk) die(1, "reps must be 1..%d and chunk at least 1", MAX_REPS);
}

// ---- the device ------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)user;
    if (types == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) return VK_FALSE;  // the loader's own notices
    const char *kind = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { val_errors++; kind = "error"; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { val_warnings++; kind = "warning"; }
    if (val_errors + val_warnings <= 50)
        fprintf(stderr, "vkhmmacost: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
    return VK_FALSE;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop;
} Features;

static void features_chain(Features *f, int coop) {
    memset(f, 0, sizeof *f);
    f->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->coop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    f->f2.pNext = &f->v11;
    f->v11.pNext = &f->v12;
    f->v12.pNext = &f->v13;
    if (coop) f->v13.pNext = &f->coop;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static void instance_init(int validation) {
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");
    const char *layers[1] = {"VK_LAYER_KHRONOS_validation"};
    const char *iexts[1] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        .pfnUserCallback = on_message};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkhmmacost",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = validation ? &mci : NULL,
                                .pApplicationInfo = &app, .enabledLayerCount = validation ? 1 : 0,
                                .ppEnabledLayerNames = layers, .enabledExtensionCount = validation ? 1 : 0,
                                .ppEnabledExtensionNames = iexts};
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));
    volkLoadInstance(inst);
    if (validation) VK_CHECK(vkCreateDebugUtilsMessengerEXT(inst, &mci, NULL, &messenger));

    uint32_t ngpu = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, NULL));
    if (!ngpu) die(2, "no Vulkan device");
    VkPhysicalDevice *gpus = calloc(ngpu, sizeof *gpus);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, gpus));
    for (int pass = 0; pass < 2 && !gpu; pass++)
        for (uint32_t i = 0; i < ngpu && !gpu; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(gpus[i], &p);
            if (pass == 0 && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            gpu = gpus[i];
            gpu_props = p;
        }
    free(gpus);
    VkPhysicalDeviceVulkan11Properties p11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p11};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    subgroup_size = p11.subgroupSize;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, NULL);
    VkExtensionProperties *dexts = calloc(next ? next : 1, sizeof *dexts);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, dexts);
    has_coop = has_ext(dexts, next, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    free(dexts);
}

static void device_init(void) {
    Features have, want;
    features_chain(&have, has_coop);
    vkGetPhysicalDeviceFeatures2(gpu, &have.f2);
    features_chain(&want, has_coop);
    want.f2.features = have.f2.features;
    want.f2.features.robustBufferAccess = VK_FALSE;
    memcpy((char *)&want.v11 + offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess),
           (char *)&have.v11 + offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess),
           sizeof have.v11 - offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess));
    want.v11.protectedMemory = VK_FALSE;
    memcpy((char *)&want.v12 + offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge),
           (char *)&have.v12 + offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge),
           sizeof have.v12 - offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge));
    memcpy((char *)&want.v13 + offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess),
           (char *)&have.v13 + offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess),
           sizeof have.v13 - offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess));
    want.v13.robustImageAccess = VK_FALSE;
    want.coop.cooperativeMatrix = have.coop.cooperativeMatrix;

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, NULL);
    VkQueueFamilyProperties *qf = calloc(nqf, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, qf);
    for (uint32_t i = 0; i < nqf && family == UINT32_MAX; i++)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) family = i;
    free(qf);
    if (family == UINT32_MAX) die(2, "no compute queue with timestamps");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *dext[1] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want.f2, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = has_coop ? 1 : 0,
                              .ppEnabledExtensionNames = dext};
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
    vkGetDeviceQueue(dev, family, 0, &queue);
}

static const char *component_name(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_SINT64_KHR: return "s64";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
    default: return "other";
    }
}

static void print_device(void) {
    uint32_t dv = gpu_props.driverVersion;
    printf("\"device\": \"%s\", \"driver\": \"%u.%02u\", \"api\": \"%u.%u.%u\", \"subgroup_size\": %u, "
           "\"timestamp_period_ns\": %.6f, \"max_shared_bytes\": %u",
           gpu_props.deviceName, (dv >> 22) & 0x3FF, (dv >> 14) & 0xFF, VK_API_VERSION_MAJOR(gpu_props.apiVersion),
           VK_API_VERSION_MINOR(gpu_props.apiVersion), VK_API_VERSION_PATCH(gpu_props.apiVersion), subgroup_size,
           gpu_props.limits.timestampPeriod, gpu_props.limits.maxComputeSharedMemorySize);
}

static int cmd_info(void) {
    instance_init(0);
    printf("{");
    print_device();
    printf(", \"cooperative_matrix_extension\": %s, \"shapes\": [", has_coop ? "true" : "false");
    if (has_coop) {
        uint32_t n = 0;
        VK_CHECK(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(gpu, &n, NULL));
        VkCooperativeMatrixPropertiesKHR *p = calloc(n ? n : 1, sizeof *p);
        for (uint32_t i = 0; i < n; i++) p[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        VK_CHECK(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(gpu, &n, p));
        for (uint32_t i = 0; i < n; i++)
            printf("%s{\"m\": %u, \"n\": %u, \"k\": %u, \"a\": \"%s\", \"b\": \"%s\", \"c\": \"%s\", \"result\": \"%s\", "
                   "\"saturating\": %u, \"scope\": %u}",
                   i ? ", " : "", p[i].MSize, p[i].NSize, p[i].KSize, component_name(p[i].AType), component_name(p[i].BType),
                   component_name(p[i].CType), component_name(p[i].ResultType), p[i].saturatingAccumulation, p[i].scope);
        free(p);
    }
    printf("]}\n");
    return 0;
}

// ---- buffers -----------------------------------------------------------------------

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    die(2, "no memory type with flags %#x", want);
}

static void buffer_make(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags, VkBuffer *buf,
                        VkDeviceMemory *mem) {
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VK_CHECK(vkCreateBuffer(dev, &bci, NULL, buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, *buf, &req);
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                .memoryTypeIndex = memory_type(req.memoryTypeBits, flags)};
    VK_CHECK(vkAllocateMemory(dev, &mai, NULL, mem));
    VK_CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
}

static VkCommandPool cmd_pool;

static VkCommandBuffer one_shot_begin(void) {
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

static void one_shot_end(VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, cmd_pool, 1, &cb);
}

// Copies between a device-local buffer and a host-visible staging buffer of the same size.
static void transfer(Buffer *b, const void *src, void *dst) {
    VkBuffer st;
    VkDeviceMemory sm;
    buffer_make(b->size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &st, &sm);
    void *map;
    VK_CHECK(vkMapMemory(dev, sm, 0, b->size, 0, &map));
    VkBufferCopy region = {.size = b->size};
    VkCommandBuffer cb = one_shot_begin();
    if (src) {
        memcpy(map, src, (size_t)b->size);
        vkCmdCopyBuffer(cb, st, b->buf, 1, &region);
    } else {
        vkCmdCopyBuffer(cb, b->buf, st, 1, &region);
    }
    one_shot_end(cb);
    if (dst) memcpy(dst, map, (size_t)b->size);
    vkUnmapMemory(dev, sm);
    vkDestroyBuffer(dev, st, NULL);
    vkFreeMemory(dev, sm, NULL);
}

// ---- run ---------------------------------------------------------------------------

static int cmd_run(const char *path) {
    Job job;
    job_read(path, &job);
    instance_init(job.validation);
    device_init();

    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
    VK_CHECK(vkCreateCommandPool(dev, &cpi, NULL, &cmd_pool));

    // Buffers, filled from their files.
    for (uint32_t i = 0; i < job.nb; i++) {
        Buffer *b = &job.b[i];
        VkBufferUsageFlags use = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                              : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        buffer_make(b->size, use, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &b->buf, &b->mem);
        char *init = calloc(1, (size_t)b->size);
        if (!init) die(3, "out of memory");
        if (b->in[0]) {
            size_t n = 0;
            void *data = read_file(b->in, &n);
            if (!data) die(1, "cannot read %s", b->in);
            if (n > b->size) die(1, "%s is larger than its buffer", b->in);
            memcpy(init, data, n);
            free(data);
        }
        transfer(b, init, NULL);
        free(init);
    }

    // Descriptor sets: one layout per set number up to the highest, empty where nothing binds.
    uint32_t nsets = 0;
    for (uint32_t i = 0; i < job.nb; i++)
        if (job.b[i].set + 1 > nsets) nsets = job.b[i].set + 1;
    VkDescriptorSetLayout layouts[MAX_SETS];
    VkDescriptorSet sets[MAX_SETS];
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_BUFFERS}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_BUFFERS}};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = MAX_SETS,
                                      .poolSizeCount = 2, .pPoolSizes = sizes};
    VkDescriptorPool dpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
    for (uint32_t s = 0; s < nsets; s++) {
        VkDescriptorSetLayoutBinding binds[MAX_BUFFERS];
        uint32_t n = 0;
        for (uint32_t i = 0; i < job.nb; i++)
            if (job.b[i].set == s)
                binds[n++] = (VkDescriptorSetLayoutBinding){.binding = job.b[i].binding, .descriptorType = job.b[i].type,
                                                            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
        VkDescriptorSetLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = n,
                                               .pBindings = binds};
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, NULL, &layouts[s]));
        VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
                                           .descriptorSetCount = 1, .pSetLayouts = &layouts[s]};
        VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &sets[s]));
        for (uint32_t i = 0; i < job.nb; i++) {
            if (job.b[i].set != s) continue;
            VkDescriptorBufferInfo info = {.buffer = job.b[i].buf, .range = VK_WHOLE_SIZE};
            VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[s],
                                      .dstBinding = job.b[i].binding, .descriptorCount = 1,
                                      .descriptorType = job.b[i].type, .pBufferInfo = &info};
            vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
        }
    }

    size_t push_size = 0;
    void *push = NULL;
    if (job.push[0]) {
        push = read_file(job.push, &push_size);
        if (!push || !push_size || push_size > 128 || push_size % 4) die(1, "push constants %s: 4 to 128 bytes", job.push);
    }
    VkPushConstantRange range = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = (uint32_t)push_size};
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = nsets,
                                      .pSetLayouts = layouts, .pushConstantRangeCount = push ? 1 : 0, .pPushConstantRanges = &range};
    VkPipelineLayout pl;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

    size_t code_size = 0;
    uint32_t *code = read_file(job.module, &code_size);
    if (!code || !code_size || code_size % 4) die(1, "cannot read the module %s", job.module);
    VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = code_size, .pCode = code};
    VkShaderModule sm;
    VK_CHECK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .flags = job.full_subgroups ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
        .layout = pl};
    VkPipeline pipe;
    double t0 = now_ms();
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));
    double build_ms = now_ms() - t0;

    uint32_t total = job.warmup + job.reps;
    VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                 .queryCount = 2 * total};
    VkQueryPool qp;
    VK_CHECK(vkCreateQueryPool(dev, &qpi, NULL, &qp));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    double wall0 = now_ms();
    for (uint32_t first = 0; first < total; first += job.chunk) {
        uint32_t last = first + job.chunk < total ? first + job.chunk : total;
        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        vkCmdResetQueryPool(cb, qp, 2 * first, 2 * (last - first));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        if (nsets) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, nsets, sets, 0, NULL);
        if (push) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)push_size, push);
        for (uint32_t i = first; i < last; i++) {
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0,
                                 NULL, 0, NULL);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 2 * i);
            vkCmdDispatch(cb, job.groups[0], job.groups[1], job.groups[2]);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 2 * i + 1);
        }
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
        VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue));
    }
    double wall_ms = now_ms() - wall0;
    uint64_t *ts = calloc(2 * total, sizeof *ts);
    VK_CHECK(vkGetQueryPoolResults(dev, qp, 0, 2 * total, 2 * total * sizeof *ts, ts, sizeof *ts,
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

    for (uint32_t i = 0; i < job.nb; i++) {
        Buffer *b = &job.b[i];
        if (!b->out[0]) continue;
        char *data = malloc((size_t)b->size);
        if (!data) die(3, "out of memory");
        transfer(b, NULL, data);
        FILE *f = fopen(b->out, "wb");
        if (!f || fwrite(data, 1, (size_t)b->size, f) != (size_t)b->size) die(1, "cannot write %s", b->out);
        fclose(f);
        free(data);
    }

    printf("{");
    print_device();
    printf(", \"build_ms\": %.3f, \"wall_ms\": %.3f, \"groups\": [%u, %u, %u], \"warmup\": %u, \"validation\": %d, "
           "\"validation_errors\": %u, \"validation_warnings\": %u, \"ms\": [",
           build_ms, wall_ms, job.groups[0], job.groups[1], job.groups[2], job.warmup, job.validation, val_errors, val_warnings);
    double period = gpu_props.limits.timestampPeriod;
    for (uint32_t i = job.warmup; i < total; i++)
        printf("%s%.6f", i > job.warmup ? ", " : "", (double)(ts[2 * i + 1] - ts[2 * i]) * period / 1e6);
    printf("]}\n");
    fflush(stdout);

    free(ts);
    free(code);
    free(push);
    vkDestroyQueryPool(dev, qp, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    vkDestroyShaderModule(dev, sm, NULL);
    vkDestroyPipelineLayout(dev, pl, NULL);
    for (uint32_t s = 0; s < nsets; s++) vkDestroyDescriptorSetLayout(dev, layouts[s], NULL);
    vkDestroyDescriptorPool(dev, dpool, NULL);
    for (uint32_t i = 0; i < job.nb; i++) {
        vkDestroyBuffer(dev, job.b[i].buf, NULL);
        vkFreeMemory(dev, job.b[i].mem, NULL);
    }
    vkDestroyCommandPool(dev, cmd_pool, NULL);
    vkDestroyDevice(dev, NULL);
    if (messenger) vkDestroyDebugUtilsMessengerEXT(inst, messenger, NULL);
    vkDestroyInstance(inst, NULL);
    return val_errors ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "info")) return cmd_info();
    if (argc == 3 && !strcmp(argv[1], "run")) return cmd_run(argv[2]);
    fprintf(stderr, "usage: vkhmmacost info | vkhmmacost run JOB\n");
    return 1;
}
