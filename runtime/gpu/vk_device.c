// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Vulkan back end's device (oot2/gpu_vk.h): its settings, the instance with the validation layer when asked for, the
// choice of a device by the items of oot2/vk_required.h, the device created with exactly those items' features and
// extensions, one queue, the timeline semaphore and the command pool, and the start of the completion thread over that
// semaphore (vk_submit.c).
//
// Adapted from DKB Recomp runtime/vk_device.c at 9d8fa02 (GPL-3.0-or-later; there ported from LAS Recomp
// runtime/nvn/vulkan/device.c at da35b83, MIT, Copyright (c) 2026 Peiikai, licenses/lasrecomp-MIT.txt): the validation
// layer behind a setting, with a messenger that counts errors; the choice of the first discrete device unless a setting
// names an index or part of a name; one queue family with graphics and compute; the timeline semaphore made with the
// device. Changed here: a device is judged by the shared list of required items, and refused by name with every item it
// lacks (DKB requires a few extensions and features of its own and skips a device silently); the features enabled are
// the items' and nothing more (DKB enables everything the driver reports); asking for validation without the layer
// installed stops the start instead of going on without it; sync and GPU-assisted validation; the instance and the device
// are made able to present only when a window was asked for (vkw_window, oot2/vk_present.h), and then with the swapchain
// extension and present timing where the device has it; no second transfer queue, no memory allocator yet.
#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu_vk.h"
#include "oot2/memline.h"
#include "oot2/texmem.h"
#include "oot2/vk_image.h"
#include "oot2/vk_pipeline.h"
#include "oot2/runtime.h"
#include "oot2/vk_bindless.h"
#include "oot2/vk_gpu_time.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_movie_blit.h"
#include "oot2/vk_present.h"
#include "oot2/vk_raw.h"
#include "oot2/vk_rec.h"
#include "oot2/vk_replay.h"
#include "plat.h"

GpuVk gpu_vk;
static int g_open;
static PlatMutex g_open_lock = PLAT_MUTEX_INIT;
static uint64_t g_errors, g_warnings, g_loader_notes;
#define LOGGED_MESSAGES 200   // validation messages logged (loader notes have their own count); validation ones are all counted

#define VALIDATION_LAYER "VK_LAYER_KHRONOS_validation"

// ---------- settings
static const char *const k_validation[] = {"off", "on", "sync", "gpu"};

int gpu_vk_settings(const char *device, const char *validation, const char *fault_report, GpuVkSettings *out,
                    char *why, size_t why_size) {
    memset(out, 0, sizeof *out);
    out->fault_report = 1;
    if (device && strlen(device) >= sizeof out->device) {
        snprintf(why, why_size, "OOT2_VK_DEVICE is %zu characters long; at most %zu", strlen(device),
                 sizeof out->device - 1);
        return 0;
    }
    if (device) snprintf(out->device, sizeof out->device, "%s", device);
    if (validation && *validation) {
        int found = -1;
        for (int i = 0; i < 4; i++)
            if (!strcmp(validation, k_validation[i])) found = i;
        if (found < 0) {
            snprintf(why, why_size, "OOT2_VK_VALIDATION is \"%s\"; the values are off, on, sync and gpu", validation);
            return 0;
        }
        out->validation = found;
    }
    if (fault_report && *fault_report) {
        if (!strcmp(fault_report, "on")) out->fault_report = 1;
        else if (!strcmp(fault_report, "off")) out->fault_report = 0;
        else {
            snprintf(why, why_size, "OOT2_VK_FAULT_REPORT is \"%s\"; the values are on and off", fault_report);
            return 0;
        }
    }
    return 1;
}

// ---------- the choice of device
static int contains_ignoring_case(const char *s, const char *part) {
    size_t n = strlen(part);
    for (; *s; s++) {
        size_t i = 0;
        while (i < n && s[i] && tolower((unsigned char)s[i]) == tolower((unsigned char)part[i])) i++;
        if (i == n) return 1;
    }
    return n == 0;
}

static int usable(const GpuVkCandidate *c) {
    if (c->unusable) return 0;
    for (int i = 0; i < OOT2_VK_ITEM_COUNT; i++)
        if (!c->has[i]) return 0;
    return 1;
}

static const char *type_name(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
    default: return "other";
    }
}

// Appends to why, keeping it terminated whatever its size.
static void add(char *why, size_t size, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void add(char *why, size_t size, const char *fmt, ...) {
    size_t at = strlen(why);
    if (at + 1 >= size) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(why + at, size - at, fmt, ap);
    va_end(ap);
}

// "device i NAME (type) lacks: a, b" or "... has every item" or "... is unusable: why", appended to why.
static void describe(const GpuVkCandidate *c, int i, char *why, size_t size) {
    add(why, size, "device %d %s (%s)", i, c->name, type_name(c->type));
    if (c->unusable) {
        add(why, size, " has no use: %s", c->unusable);
        return;
    }
    int lacks = 0;
    for (int k = 0; k < OOT2_VK_ITEM_COUNT; k++)
        if (!c->has[k]) add(why, size, "%s%s", lacks++ ? ", " : " lacks: ", oot2_vk_item_what(k));
    if (!lacks) add(why, size, " has every item");
}

int gpu_vk_choose(const GpuVkCandidate *c, int n, const char *setting, char *why, size_t why_size) {
    why[0] = 0;
    if (n <= 0) {
        add(why, why_size, "no Vulkan device");
        return -1;
    }
    if (!setting || !*setting) {
        static const VkPhysicalDeviceType order[] = {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                                                     VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU};
        for (int pass = 0; pass < 3; pass++)
            for (int i = 0; i < n; i++)
                if (usable(&c[i]) && (pass == 2 || c[i].type == order[pass])) return i;
        add(why, why_size, "no Vulkan device has every item the renderer requires: ");
        for (int i = 0; i < n; i++) {
            if (i) add(why, why_size, "; ");
            describe(&c[i], i, why, why_size);
        }
        return -1;
    }
    int digits = 1;
    for (const char *p = setting; *p; p++) digits &= isdigit((unsigned char)*p) != 0;
    int pick = -1;
    // Digits below the device count are an index. Digits past it are looked up in the names like any other text,
    // because 3060 is how a 3060 Ti is written.
    if (digits && strlen(setting) <= 4 && strtol(setting, NULL, 10) < n) {
        pick = (int)strtol(setting, NULL, 10);
    } else {
        for (int i = 0; i < n && pick < 0; i++)   // the first match with every item, else the first match
            if (contains_ignoring_case(c[i].name, setting) && usable(&c[i])) pick = i;
        for (int i = 0; i < n && pick < 0; i++)
            if (contains_ignoring_case(c[i].name, setting)) pick = i;
        if (pick < 0) {
            if (digits)
                add(why, why_size, "OOT2_VK_DEVICE=%s, and there %s %d device%s: ", setting, n == 1 ? "is" : "are", n,
                    n == 1 ? "" : "s");
            else
                add(why, why_size, "OOT2_VK_DEVICE=%s, and no device's name contains it: ", setting);
            for (int i = 0; i < n; i++) add(why, why_size, "%s%d %s", i ? ", " : "", i, c[i].name);
            return -1;
        }
    }
    if (!usable(&c[pick])) {
        add(why, why_size, "OOT2_VK_DEVICE=%s names a device the renderer cannot use: ", setting);
        describe(&c[pick], pick, why, why_size);
        return -1;
    }
    return pick;
}

// ---------- validation messages
static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT type,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)user;
    int error = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0;
    // The loader reports through the same messenger from instance creation on: another vendor's layer manifest that
    // will not open, a layer its filter switched off. Those describe the machine, not the layer's verdict on this
    // code, so they are logged under their own name and are not counted.
    if (data && data->pMessageIdName && !strcmp(data->pMessageIdName, "Loader Message")) {
        if (__atomic_add_fetch(&g_loader_notes, 1, __ATOMIC_RELAXED) <= LOGGED_MESSAGES)
            rt_log("gfx: vulkan loader %s: %s", error ? "error" : "warning", data->pMessage ? data->pMessage : "");
        return VK_FALSE;
    }
    uint64_t n = __atomic_add_fetch(error ? &g_errors : &g_warnings, 1, __ATOMIC_RELAXED);
    if (n <= LOGGED_MESSAGES)
        rt_log("gfx: vulkan validation %s%s: %s", error ? "error" : "warning",
               (type & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) ? " (performance)" : "",
               data && data->pMessage ? data->pMessage : "");
    return VK_FALSE;
}

void gpu_vk_validation_counts(uint64_t *errors, uint64_t *warnings) {
    if (errors) *errors = __atomic_load_n(&g_errors, __ATOMIC_RELAXED);
    if (warnings) *warnings = __atomic_load_n(&g_warnings, __ATOMIC_RELAXED);
}

const char *gpu_vk_result_name(VkResult r) {
    switch (r) {
#define NAME(x)                                                                                                        \
    case x: return #x;
        NAME(VK_SUCCESS) NAME(VK_NOT_READY) NAME(VK_TIMEOUT) NAME(VK_INCOMPLETE) NAME(VK_ERROR_OUT_OF_HOST_MEMORY)
        NAME(VK_ERROR_OUT_OF_DEVICE_MEMORY) NAME(VK_ERROR_INITIALIZATION_FAILED) NAME(VK_ERROR_DEVICE_LOST)
        NAME(VK_ERROR_MEMORY_MAP_FAILED) NAME(VK_ERROR_LAYER_NOT_PRESENT) NAME(VK_ERROR_EXTENSION_NOT_PRESENT)
        NAME(VK_ERROR_FEATURE_NOT_PRESENT) NAME(VK_ERROR_INCOMPATIBLE_DRIVER) NAME(VK_ERROR_TOO_MANY_OBJECTS)
        NAME(VK_ERROR_UNKNOWN)
#undef NAME
    default: {
        static char buf[8][32];
        static unsigned at;
        char *b = buf[__atomic_fetch_add(&at, 1, __ATOMIC_RELAXED) % 8];
        snprintf(b, sizeof buf[0], "VkResult %d", (int)r);
        return b;
    }
    }
}

// ---------- the instance
static void instance(void) {
    VkResult r = volkInitialize();
    if (r != VK_SUCCESS)
        rt_fatal("gfx: vulkan: there is no Vulkan loader on this machine (%s); install the GPU's driver, or use the null "
                 "back end", gpu_vk_result_name(r));
    uint32_t version = volkGetInstanceVersion();
    if (version < VK_API_VERSION_1_3)
        rt_fatal("gfx: vulkan: the Vulkan loader is version %u.%u; the renderer requires 1.3", VK_API_VERSION_MAJOR(version),
                 VK_API_VERSION_MINOR(version));
    const char *layers[1] = {VALIDATION_LAYER};
    int validate = gpu_vk.settings.validation != GPU_VALIDATION_OFF;
    if (validate) {
        uint32_t n = 0;
        GPU_VK_OK(vkEnumerateInstanceLayerProperties(&n, NULL), "vkEnumerateInstanceLayerProperties");
        VkLayerProperties *lp = calloc(n ? n : 1, sizeof *lp);
        if (!lp) rt_fatal("gfx: vulkan: out of memory for %u layers", n);
        GPU_VK_OK(vkEnumerateInstanceLayerProperties(&n, lp), "vkEnumerateInstanceLayerProperties");
        int found = 0;
        for (uint32_t i = 0; i < n; i++) found |= !strcmp(lp[i].layerName, VALIDATION_LAYER);
        free(lp);
        if (!found)
            rt_fatal("gfx: vulkan: OOT2_VK_VALIDATION=%s asks for the layer %s, which is not installed (it comes with "
                     "the Vulkan SDK)", k_validation[gpu_vk.settings.validation], VALIDATION_LAYER);
    }
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = on_message};
    VkValidationFeatureEnableEXT enables[2];
    uint32_t nenables = 0;
    if (gpu_vk.settings.validation == GPU_VALIDATION_SYNC) {
        enables[nenables++] = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    } else if (gpu_vk.settings.validation == GPU_VALIDATION_GPU) {
        enables[nenables++] = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT;
        enables[nenables++] = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT;
    }
    VkValidationFeaturesEXT features = {.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, .pNext = &mci,
                                        .enabledValidationFeatureCount = nenables, .pEnabledValidationFeatures = enables};
    // the debug messenger's extension when validating, and the window's surface extensions when a window was asked for
    const char *extensions[16];
    uint32_t nextensions = 0;
    if (validate) extensions[nextensions++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    if (vkw_wanted()) nextensions += vkw_instance_extensions(extensions + nextensions, 16 - nextensions);
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "oot2",
                             .pEngineName = "oot2-native", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = validate ? &features : NULL,
                               .pApplicationInfo = &app, .enabledLayerCount = validate ? 1 : 0,
                               .ppEnabledLayerNames = layers, .enabledExtensionCount = nextensions,
                               .ppEnabledExtensionNames = extensions};
    r = vkCreateInstance(&ci, NULL, &gpu_vk.instance);
    if (r == VK_ERROR_INCOMPATIBLE_DRIVER)   // a loader with no driver behind it
        rt_fatal("gfx: vulkan: there is no Vulkan device on this machine (vkCreateInstance: %s)", gpu_vk_result_name(r));
    if (r != VK_SUCCESS) rt_fatal("gfx: vulkan: vkCreateInstance failed: %s", gpu_vk_result_name(r));
    volkLoadInstanceOnly(gpu_vk.instance);
    if (validate)
        GPU_VK_OK(vkCreateDebugUtilsMessengerEXT(gpu_vk.instance, &mci, NULL, &gpu_vk.messenger),
                  "vkCreateDebugUtilsMessengerEXT");
}

// ---------- the device
// The first queue family with graphics and compute, one with sparse binding too when there is one (D14's sparse
// buffers bind on the queue that uses them); UINT32_MAX when none.
static uint32_t queue_family(VkPhysicalDevice dev, VkQueueFlags *flags) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, NULL);
    VkQueueFamilyProperties *q = calloc(n ? n : 1, sizeof *q);
    if (!q) rt_fatal("gfx: vulkan: out of memory for %u queue families", n);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, q);
    const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    uint32_t found = UINT32_MAX;
    for (int pass = 0; pass < 2 && found == UINT32_MAX; pass++)
        for (uint32_t i = 0; i < n && found == UINT32_MAX; i++)
            if ((q[i].queueFlags & need) == need && (pass || (q[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))) found = i;
    if (found != UINT32_MAX) *flags = q[found].queueFlags;
    free(q);
    return found;
}

static VkExtensionProperties *device_extensions(VkPhysicalDevice dev, uint32_t *n) {
    *n = 0;
    GPU_VK_OK(vkEnumerateDeviceExtensionProperties(dev, NULL, n, NULL), "vkEnumerateDeviceExtensionProperties");
    VkExtensionProperties *e = calloc(*n ? *n : 1, sizeof *e);
    if (!e) rt_fatal("gfx: vulkan: out of memory for %u extensions", *n);
    GPU_VK_OK(vkEnumerateDeviceExtensionProperties(dev, NULL, n, e), "vkEnumerateDeviceExtensionProperties");
    return e;
}

static void choose(void) {
    uint32_t n = 0;
    VkResult r = vkEnumeratePhysicalDevices(gpu_vk.instance, &n, NULL);
    if (r != VK_SUCCESS || !n)
        rt_fatal("gfx: vulkan: there is no Vulkan device on this machine (vkEnumeratePhysicalDevices: %s, %u devices)",
                 gpu_vk_result_name(r), n);
    VkPhysicalDevice *devs = calloc(n, sizeof *devs);
    GpuVkCandidate *cand = calloc(n, sizeof *cand);
    Oot2VkFacts *facts = calloc(n, sizeof *facts);
    VkPhysicalDeviceProperties *props = calloc(n, sizeof *props);
    if (!devs || !cand || !facts || !props) rt_fatal("gfx: vulkan: out of memory for %u devices", n);
    GPU_VK_OK(vkEnumeratePhysicalDevices(gpu_vk.instance, &n, devs), "vkEnumeratePhysicalDevices");
    for (uint32_t i = 0; i < n; i++) {
        vkGetPhysicalDeviceProperties(devs[i], &props[i]);
        uint32_t ne;
        VkExtensionProperties *ext = device_extensions(devs[i], &ne);
        oot2_vk_query(vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceFeatures2, devs[i], ext, ne, &facts[i]);
        free(ext);
        int has = oot2_vk_judge(&facts[i], cand[i].has);
        cand[i].name = props[i].deviceName;
        cand[i].type = props[i].deviceType;
        VkQueueFlags flags = 0;
        uint32_t family = queue_family(devs[i], &flags);
        if (family == UINT32_MAX) cand[i].unusable = "no queue family with graphics and compute";
        else if (vkw_wanted() && !vkw_can_present(devs[i], family)) cand[i].unusable = "it cannot present to the window";
        rt_log("gfx: vulkan: device %u: %s (%s, Vulkan %u.%u.%u), %d of the %d items the renderer requires", i,
               props[i].deviceName, type_name(props[i].deviceType), VK_API_VERSION_MAJOR(props[i].apiVersion),
               VK_API_VERSION_MINOR(props[i].apiVersion), VK_API_VERSION_PATCH(props[i].apiVersion), has,
               OOT2_VK_ITEM_COUNT);
    }
    char why[4096];
    int pick = gpu_vk_choose(cand, (int)n, gpu_vk.settings.device, why, sizeof why);
    if (pick < 0) rt_fatal("gfx: vulkan: %s", why);
    gpu_vk.phys = devs[pick];
    gpu_vk.index = pick;
    gpu_vk.facts = facts[pick];
    gpu_vk.family = queue_family(gpu_vk.phys, &gpu_vk.family_flags);
    rt_log("gfx: vulkan: device %d chosen (%s)", pick,
           gpu_vk.settings.device[0] ? "OOT2_VK_DEVICE" : "the first discrete device with every item");
    free(devs);
    free(cand);
    free(facts);
    free(props);
}

static void create_device(uint64_t after) {
    Oot2VkFacts en;
    oot2_vk_enable(&gpu_vk.facts, &en);
    gpu_vk.fault = gpu_vk.checkpoints = 0;
    if (gpu_vk.settings.fault_report) {
        gpu_vk.fault = en.fault.deviceFaultVendorBinary ? 2 : 1;
        gpu_vk.checkpoints = 1;
    } else {   // neither extension is enabled, so neither structure is chained
        en.ext[OOT2_VK_EXT_device_fault] = en.ext[OOT2_VK_EXT_diagnostic_checkpoints] = 0;
        memset(&en.fault, 0, sizeof en.fault);
    }
    gpu_vk.image_2d_view_of_3d = en.view2d.image2DViewOf3D ? 1 : 0;
    VkPhysicalDeviceFeatures2 features;
    oot2_vk_chain_features(&en, &features);
    features.features = en.features;
    // OOT2_VK_DRIVER_CACHE=off keeps the driver's own pipeline cache from serving or storing anything, so that a build
    // is cold whatever ran before (spike S3's method, the pipeline cost study; prepare's pipeline step measures with it). The
    // structure rides on the device chain, which VK_KHR_pipeline_binary (a required item) makes valid.
    VkDevicePipelineBinaryInternalCacheControlKHR driver_cache = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL_KHR, .disableInternalCache = VK_TRUE};
    const char *dc = getenv("OOT2_VK_DRIVER_CACHE");
    if (dc && *dc && strcmp(dc, "on") && strcmp(dc, "off"))
        rt_fatal("gfx: vulkan: OOT2_VK_DRIVER_CACHE is \"%s\"; the values are on and off", dc);
    if (dc && !strcmp(dc, "off")) {
        driver_cache.pNext = features.pNext;
        features.pNext = &driver_cache;
        rt_log("gfx: vulkan: OOT2_VK_DRIVER_CACHE off: the driver's own pipeline cache is disabled");
    }
    const char *names[OOT2_VK_EXT_COUNT + 8];   // the items' extensions, and a window's: swapchain and present timing
    uint32_t nnames = 0;
    for (int e = 0; e < OOT2_VK_EXT_COUNT; e++)
        if (en.ext[e]) names[nnames++] = oot2_vk_extension_name(e);
    if (vkw_wanted()) vkw_device_extensions(gpu_vk.phys, names, &nnames, &features);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = gpu_vk.family,
                                  .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &features,
                             .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi, .enabledExtensionCount = nnames,
                             .ppEnabledExtensionNames = names};
    GPU_VK_OK(vkCreateDevice(gpu_vk.phys, &di, NULL, &gpu_vk.device), "vkCreateDevice");
    volkLoadDevice(gpu_vk.device);
    vkGetDeviceQueue(gpu_vk.device, gpu_vk.family, 0, &gpu_vk.queue);
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = after};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    GPU_VK_OK(vkCreateSemaphore(gpu_vk.device, &si, NULL, &gpu_vk.timeline), "vkCreateSemaphore (the timeline)");
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                  .queueFamilyIndex = gpu_vk.family};
    GPU_VK_OK(vkCreateCommandPool(gpu_vk.device, &pi, NULL, &gpu_vk.pool), "vkCreateCommandPool");
    GPU_VK_OK(vkCreateCommandPool(gpu_vk.device, &pi, NULL, &gpu_vk.pool_main), "vkCreateCommandPool (main)");
    const VkPhysicalDeviceProperties *p = &gpu_vk.facts.props;
    rt_log("gfx: vulkan: %s, vendor %#x device %#x, driver %#x, Vulkan %u.%u.%u: device made with the %d items' "
           "features and %u extensions, queue family %u (flags %#x), timeline from %llu; validation %s; fault report "
           "%s", p->deviceName, p->vendorID, p->deviceID, p->driverVersion, VK_API_VERSION_MAJOR(p->apiVersion),
           VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion), OOT2_VK_ITEM_COUNT, nnames,
           gpu_vk.family, (unsigned)gpu_vk.family_flags, (unsigned long long)after,
           k_validation[gpu_vk.settings.validation],
           !gpu_vk.fault ? "off" : gpu_vk.fault == 2 ? "on, with the vendor's binary" : "on, without a vendor binary");
}

void gpu_vk_start_timeline(uint64_t after);   // vk_submit.c

// The device terms of the memory line (oot2/memline.h), on its helper thread. It holds g_open_lock, so a device being closed
// is never read, and it reads nothing a frame holds longer than the accessor takes. Vulkan heaps 0 and 1 are what
// VK_EXT_memory_budget reports for this process.
static void memline_fill(MemTerms *t) {
    plat_mutex_lock(&g_open_lock);
    if (g_open) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
        vkGetPhysicalDeviceMemoryProperties2(gpu_vk.phys, &mp);
        uint32_t n = mp.memoryProperties.memoryHeapCount;
        t->vk_heap0_used = n > 0 ? b.heapUsage[0] : 0, t->vk_heap0_budget = n > 0 ? b.heapBudget[0] : 0;
        t->vk_heap1_used = n > 1 ? b.heapUsage[1] : 0, t->vk_heap1_budget = n > 1 ? b.heapBudget[1] : 0;
        TexMemStats tm;
        texmem_stats(&tm);
        t->tex_ring_chunks = tm.ring_chunks, t->tex_ring_bytes = tm.ring_bytes, t->tex_arena_blocks = tm.arena_blocks;
        t->tex_arena_bytes = tm.arena_bytes, t->tex_arena_used = tm.arena_used, t->tex_arena_frees = tm.arena_frees;
        BackingStats bs;
        backing_stats(&bs);
        t->bk_blocks = bs.blocks, t->bk_block_bytes = bs.block_bytes, t->bk_extents = bs.extents;
        t->bk_extent_bytes = bs.extent_bytes, t->bk_imports = bs.imports, t->bk_import_bytes = bs.import_bytes;
        t->bk_mirrors = bs.mirrors, t->bk_mirror_bytes = bs.mirror_bytes, t->bk_deferred = bs.deferred;
        VkbImageStats is;
        vkb_image_stats(&is);
        t->img_live = is.live, t->img_own = is.unaliased, t->img_own_bytes = is.unaliased_bytes, t->img_arena = is.arena;
        VkfImgColumns ic;
        vkf_img_columns(&is.cls, &ic);
        t->astc_dec_w4 = ic.dec_w[0], t->astc_dec_w5 = ic.dec_w[1], t->astc_dec_w6 = ic.dec_w[2];
        t->astc_dec_w8 = ic.dec_w[3], t->astc_dec_w10 = ic.dec_w[4], t->astc_dec_w12 = ic.dec_w[5];
        t->astc_dec_srgb_bytes = ic.dec_srgb, t->astc_dec_arena_bytes = ic.dec_arena, t->astc_dec_images = ic.dec_images;
        t->bc_equiv_bytes = ic.bc_equiv, t->native_bc_bytes = ic.native_bc, t->other_bytes = ic.other;
        VkpStats ps;
        vkp_stats(&ps);
        t->pipe_resident = ps.resident, t->pipe_libraries = ps.libraries, t->pipe_links = ps.links;
        t->pipe_cache_bytes = vkp_cache_size();
    }
    plat_mutex_unlock(&g_open_lock);
}

void gpu_vk_open(void) {
    plat_mutex_lock(&g_open_lock);
    if (g_open) {
        plat_mutex_unlock(&g_open_lock);
        return;
    }
    char why[512];
    GpuVkSettings s;
    if (!gpu_vk_settings(getenv("OOT2_VK_DEVICE"), getenv("OOT2_VK_VALIDATION"), getenv("OOT2_VK_FAULT_REPORT"), &s,
                         why, sizeof why))
        rt_fatal("gfx: vulkan: %s", why);
    uint64_t after = gpu_sync_done();   // a device opened again goes on from the values the last one reached
    memset(&gpu_vk, 0, sizeof gpu_vk);
    gpu_vk.settings = s;
    rt_log("gfx: vulkan: settings: OOT2_VK_DEVICE %s%s%s, OOT2_VK_VALIDATION %s, OOT2_VK_FAULT_REPORT %s",
           s.device[0] ? "\"" : "", s.device[0] ? s.device : "unset (the first discrete device with every item)",
           s.device[0] ? "\"" : "", k_validation[s.validation], s.fault_report ? "on" : "off");
    instance();
    choose();
    create_device(after);
    gpu_vk_start_timeline(after);
    backing_open(&gpu_vk_backing_ops);
    gpu_vk_mirror_policy();
    vkb_open();
    vkgt_open();   // the GPU timing of the replayed frames, when OOT2_VK_GPU_TIMING names a file (oot2/vk_gpu_time.h)
    vkmb_open();   // the movie chain's stand-in, when the test switch test_movie_blit is on (oot2/vk_movie_blit.h)
    vkw_open();   // the window's surface, the menu's pipeline and the present thread, when a window was asked for
    g_open = 1;
    memline_set_gpu(memline_fill);
    plat_mutex_unlock(&g_open_lock);
}

void gpu_vk_stop_timeline(void);   // vk_submit.c

void gpu_vk_close(void) {
    plat_mutex_lock(&g_open_lock);
    if (g_open) {
        vkrec_close();   // the frames handed over, submitted (nothing when the replay never opened the recorder)
        vkw_close();     // the present thread, the swapchain, the menu and the surface, before the device goes
        gpu_sync_finish();
        vkr_device_close();
        vkgt_close();   // every frame is done: the timing's pools and file
        vkm_close();   // the builds still running, the releases that were waiting, the summary of the misses
        gpu_vk_raw_close();
        vkb_close();
        backing_close();
        gpu_vk_stop_timeline();
        GPU_VK_OK(vkDeviceWaitIdle(gpu_vk.device), "vkDeviceWaitIdle");
        vkDestroyCommandPool(gpu_vk.device, gpu_vk.pool, NULL);
        vkDestroyCommandPool(gpu_vk.device, gpu_vk.pool_main, NULL);
        vkDestroySemaphore(gpu_vk.device, gpu_vk.timeline, NULL);
        vkDestroyDevice(gpu_vk.device, NULL);
        if (gpu_vk.messenger) vkDestroyDebugUtilsMessengerEXT(gpu_vk.instance, gpu_vk.messenger, NULL);
        vkDestroyInstance(gpu_vk.instance, NULL);
        GpuVkSettings s = gpu_vk.settings;
        memset(&gpu_vk, 0, sizeof gpu_vk);
        gpu_vk.settings = s;
        g_open = 0;
    }
    plat_mutex_unlock(&g_open_lock);
}
