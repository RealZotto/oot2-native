// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Host overrides of the system services the game reaches through public nn:: entry points: nn::oe (with the applet and
// am resource-user ids), nn::account, nn::aoc, nn::time, nn::pl, nn::pctl, nn::err, nn::diag, nn::prepo, nn::nfp and
// nn::nfc, the start-up initialisers of nn::friends, nn::bcat and nn::ldn, and the nn::os fibre internals that host
// fibres leave without a caller (named fatals). Nothing here talks to a Horizon service:
// each call is answered by the host, by the rule at the top of hos.h, and each override names its answer for the
// census. The network overrides (socket, nifm, ssl, nsd, eventfd), the file system and graphics/input are other files.
//
// Adapted from oot-pc-recomp runtime/ov_system.c at 8e09aaa (MIT file header, Copyright (c) 2026 MacNativeGames
// contributors, adapted there from MacNativeGames games/totk/runtime services.c, applet_host.c, account.c, time_host.c
// and shared_font.c at b75be8c; inside a GPL-3.0-or-later repository): the oe, applet, am, prepo, nfp, err, aoc, pctl,
// diag, account, time and pl answers, the message queue behind PopNotificationMessage, the local user table, the clock
// snapshot layout and the calendar arithmetic. The answers for the network clock task and the network service accounts
// follow what NXNative (MIT) documents (the services study); no code was taken from it.
//
// Changed here: the settings the donor read from environment variables and an ini file are fixed (language en-US,
// docked with performance boost, always in focus), and the local user has one constant Uid instead of a file, so there
// is no configuration. The six observers and the network overrides are left out (the first need a facility this
// runtime does not have, the second are the network file). Every override is a HOS_OVERRIDE with its status and a one
// line why. Out parameters and by-reference inputs go through hos_out and hos_in: an address that is not guest memory
// stops the guest, as the fault would on a console. A value the runtime has no answer for is a named gap (hos_gap)
// instead of a made-up success: a shared font when no open-licence font file is placed (the console's own fonts are never
// read), an unknown font type or mode. Added to the donor's set: the network clock task, the network service accounts and the
// account AsyncContext, the prepo over-frequency event, the nfp device calls the game imports, and
// StandardNetworkSystemClock::now.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core.h"
#include "hos.h"
#include "plat.h"

#define NFP_R_DEVICE_NOT_FOUND 0x8073u   // nfp: no NFC reader on the host (the donor's result)
#define ACCOUNT_R_INVALID 0x7c67u        // the donor's result for an unknown user handle or Uid
#define TIME_R_INVALID 0x74u
#define TIME_R_SOURCE_MISMATCH 0x18074u

// Room for an nn::os::SystemEventType with margin. The SDK's own os:: code reads and writes these objects as lifted
// code (AttachReadableHandleToSystemEvent fills one in), so they live in guest memory: one block of the kernel region,
// taken at the first use, holding the network clock task slots' events and then the prepo over-frequency event.
#define EVENT_SPACE 0x100
#define NETCLOCK_SLOTS 16

static PlatMutex sys_lock = PLAT_MUTEX_INIT;   // guards every piece of shared state below
static uint64_t event_block;                   // sys_lock: guest address of the events, 0 until the first use

// The guest address of event i (0 to NETCLOCK_SLOTS), zeroed until the SDK writes it.
static uint64_t event_at(int i) {
    plat_mutex_lock(&sys_lock);
    if (!event_block) event_block = mem_kernel_alloc((NETCLOCK_SLOTS + 1) * EVENT_SPACE);
    uint64_t at = event_block + (uint64_t)i * EVENT_SPACE;
    plat_mutex_unlock(&sys_lock);
    return at;
}

// An out parameter: a null pointer is not written (the donor's rule); an address that is not guest memory the guest may
// write stops the guest.
static void out_put(Cpu *c, const char *fn, uint64_t addr, const void *src, uint64_t n) {
    if (addr && !hos_out(addr, src, n))
        rt_guest_stop(c, "%s: %#llx is not guest memory the guest may write", fn, (unsigned long long)addr);
}

// A by-reference input of n bytes: null or not guest memory stops the guest.
static void in_get(Cpu *c, const char *fn, void *dst, uint64_t addr, uint64_t n) {
    if (!addr || !hos_in(dst, addr, n))
        rt_guest_stop(c, "%s: %#llx is not guest memory the guest may read", fn, (unsigned long long)addr);
}

// A Result or bool or enum of 0, which a function returning nothing ignores: the answer of every handler below that
// has no value to give.
static void ov_zero(Cpu *c) { ret_x(c, 0); }

// A function the host has nothing to do for and whose registers it leaves alone: constructors, destructors and void
// initialisers.
static void ov_nothing(Cpu *c) { (void)c; }

// ==== nn::oe

// LanguageCode GetDesiredLanguage(): the 8-byte code in x0
static void ov_desired_language(Cpu *c) {
    static const char code[8] = "en-US";
    uint64_t v;
    memcpy(&v, code, sizeof v);
    ret_x(c, v);
}
HOS_OVERRIDE("_ZN2nn2oe18GetDesiredLanguageEv", ov_desired_language, HOS_IMPLEMENTED,
             "fixed en-US: the host has no language setting; donor ov_system.c:114");

static void ov_oe_initialize(Cpu *c) {
    (void)c;
    rt_log("oe: application environment: a local host process");
}
HOS_OVERRIDE("_ZN2nn2oe10InitializeEv", ov_oe_initialize, HOS_STUB,
             "no applet proxy to open, the program is a host process; donor ov_system.c:121");

// void SetPerformanceConfiguration(PerformanceMode mode, int config): the host OS schedules the CPU
static void ov_performance_config(Cpu *c) {
    int32_t mode = (int32_t)arg_x(c, 0);
    if (mode < 0 || mode > 1)
        rt_guest_stop(c, "nn::oe::SetPerformanceConfiguration: invalid performance mode %d", mode);
}
HOS_OVERRIDE("_ZN2nn2oe27SetPerformanceConfigurationENS0_15PerformanceModeEi", ov_performance_config, HOS_STUB,
             "mode checked, no effect: the host OS schedules the CPU; donor ov_system.c:124");

// void SetCpuBoostMode(CpuBoostMode mode)
static void ov_cpu_boost(Cpu *c) {
    int32_t mode = (int32_t)arg_x(c, 0);
    if (mode < 0 || mode > 2) hos_gap(c, "oe", "nn::oe::SetCpuBoostMode: unknown mode %d", mode);
}
HOS_OVERRIDE("_ZN2nn2oe15SetCpuBoostModeENS0_12CpuBoostModeE", ov_cpu_boost, HOS_STUB,
             "mode checked, no effect: the host does not scale clocks; donor ov_system.c:132");

// OperationMode GetOperationMode() and PerformanceMode GetPerformanceMode(): Console (1) and Boost (1), the docked
// console with the most to give. The donor chose by TOTK_OPERATION_MODE, default docked.
static void ov_docked(Cpu *c) { ret_x(c, 1); }
HOS_OVERRIDE("_ZN2nn2oe16GetOperationModeEv", ov_docked, HOS_IMPLEMENTED,
             "docked (console operation mode): fixed, no handheld on the host; donor ov_system.c:137");
HOS_OVERRIDE("_ZN2nn2oe18GetPerformanceModeEv", ov_docked, HOS_IMPLEMENTED,
             "boost performance mode: fixed, docked; donor ov_system.c:138");

// void SetFocusHandlingMode(FocusHandlingMode mode). The nnMain checkpoint of the boot driver is not reported from
// here: hos.h has no call for it.
static void ov_focus_mode(Cpu *c) {
    int32_t mode = (int32_t)arg_x(c, 0);
    if (mode < 0 || mode > 3) hos_gap(c, "oe", "nn::oe::SetFocusHandlingMode: unknown mode %d", mode);
    char at[96];
    rt_log("oe: focus handling mode %d from %s", mode, hos_caller(c, at, (int)sizeof at));
}
HOS_OVERRIDE("_ZN2nn2oe20SetFocusHandlingModeENS0_17FocusHandlingModeE", ov_focus_mode, HOS_STUB,
             "mode checked and logged, no effect: focus never leaves the window; donor ov_system.c:142");

// FocusState GetCurrentFocusState(): 1 in focus (nn::oe::FocusState). Always in focus: with focus handling mode 2 a
// console suspends the whole process while out of focus, and a host that stopped only the main thread was seen to
// leave loader threads and clocks running into a load that never finished (donor, Linux).
static void ov_focus_state(Cpu *c) { ret_x(c, 1); }
HOS_OVERRIDE("_ZN2nn2oe20GetCurrentFocusStateEv", ov_focus_state, HOS_IMPLEMENTED,
             "always in focus: half-suspending a host process breaks loads; donor ov_system.c:154-168");

// void Set{Resume,OperationModeChanged,PerformanceModeChanged}NotificationEnabled(bool): none of the three
// notifications is ever sent
HOS_OVERRIDE("_ZN2nn2oe28SetResumeNotificationEnabledEb", ov_zero, HOS_STUB,
             "accepted, no effect: no resume message is ever sent; donor ov_system.c:171");
HOS_OVERRIDE("_ZN2nn2oe42SetOperationModeChangedNotificationEnabledEb", ov_zero, HOS_STUB,
             "accepted, no effect: the operation mode never changes; donor ov_system.c:172");
HOS_OVERRIDE("_ZN2nn2oe44SetPerformanceModeChangedNotificationEnabledEb", ov_zero, HOS_STUB,
             "accepted, no effect: the performance mode never changes; donor ov_system.c:173");

// The notification message queue: PopNotificationMessage blocks until another host thread pushes a message.
#define MESSAGES 32
static PlatCond message_cv = PLAT_COND_INIT;
static uint32_t messages[MESSAGES];
static unsigned head, tail;
static float balance[2] = {1.0f, 1.0f};

// Queue a notification message for the game (a window or lifecycle event). A full queue drops the new message.
void hos_oe_push_message(uint32_t message) {
    plat_mutex_lock(&sys_lock);
    if (tail - head < MESSAGES) {
        messages[tail++ % MESSAGES] = message;
        plat_cond_broadcast(&message_cv);
    }
    plat_mutex_unlock(&sys_lock);
}

// bool TryPopNotificationMessage(Message *out)
static void ov_try_pop(Cpu *c) {
    uint32_t m = 0;
    plat_mutex_lock(&sys_lock);
    int ok = head != tail;
    if (ok) m = messages[head++ % MESSAGES];
    plat_mutex_unlock(&sys_lock);
    if (ok) out_put(c, "nn::oe::TryPopNotificationMessage", arg_x(c, 0), &m, sizeof m);
    ret_x(c, (uint64_t)ok);
}
HOS_OVERRIDE("_ZN2nn2oe25TryPopNotificationMessageEPj", ov_try_pop, HOS_IMPLEMENTED,
             "pops the host message queue, empty unless the host pushed one; donor ov_system.c:232");

// Message PopNotificationMessage(): blocks like a Horizon wait, so the thread gives its virtual core up meanwhile
static void ov_pop(Cpu *c) {
    int token = rt_kernel_block_begin();
    plat_mutex_lock(&sys_lock);
    while (head == tail) plat_cond_wait(&message_cv, &sys_lock);
    uint32_t m = messages[head++ % MESSAGES];
    plat_mutex_unlock(&sys_lock);
    rt_kernel_block_end(token);
    ret_x(c, m);
}
HOS_OVERRIDE("_ZN2nn2oe22PopNotificationMessageEv", ov_pop, HOS_IMPLEMENTED,
             "blocks on the host message queue with the core released; donor ov_system.c:245");

// void GetExpectedVolumeBalance(float *left, float *right) and SetExpectedVolumeBalance(float left, float right)
static void ov_get_balance(Cpu *c) {
    plat_mutex_lock(&sys_lock);
    float l = balance[0], r = balance[1];
    plat_mutex_unlock(&sys_lock);
    out_put(c, "nn::oe::GetExpectedVolumeBalance", arg_x(c, 0), &l, sizeof l);
    out_put(c, "nn::oe::GetExpectedVolumeBalance", arg_x(c, 1), &r, sizeof r);
}
HOS_OVERRIDE("_ZN2nn2oe24GetExpectedVolumeBalanceEPfS1_", ov_get_balance, HOS_IMPLEMENTED,
             "returns the balance the game set, 1:1 at start; donor ov_system.c:194");

static void ov_set_balance(Cpu *c) {
    float l = arg_s(c, 0), r = arg_s(c, 1);
    if (!(l >= 0 && l <= 1 && r >= 0 && r <= 1))
        rt_guest_stop(c, "nn::oe::SetExpectedVolumeBalance: balance %g/%g out of range", (double)l, (double)r);
    plat_mutex_lock(&sys_lock);
    balance[0] = l;
    balance[1] = r;
    plat_mutex_unlock(&sys_lock);
}
HOS_OVERRIDE("_ZN2nn2oe24SetExpectedVolumeBalanceEff", ov_set_balance, HOS_IMPLEMENTED,
             "range checked and kept for the getter; donor ov_system.c:203");

// TimeSpan GetProgramTotalActiveTime(): nanoseconds since the process started, which the guest counter counts from
static void ov_active_time(Cpu *c) {
    uint64_t t = rt_ticks(), hz = rt_tick_hz();
    ret_x(c, t / hz * 1000000000ull + t % hz * 1000000000ull / hz);
}
HOS_OVERRIDE("_ZN2nn2oe25GetProgramTotalActiveTimeEv", ov_active_time, HOS_IMPLEMENTED,
             "nanoseconds since the process started; donor ov_system.c:214");

HOS_OVERRIDE("_ZN2nn2oe38SetUserInactivityDetectionTimeExtendedEb", ov_zero, HOS_STUB,
             "accepted, no effect: the host never puts the console to sleep; donor ov_system.c:219");
HOS_OVERRIDE("_ZN2nn2oe44SetUserInactivityDetectionTimeExtendedUnsafeEb", ov_zero, HOS_STUB,
             "accepted, no effect: the host never puts the console to sleep; donor ov_system.c:219");

// void FinishStartupLogo(): no system logo is drawn on the host
static void ov_finish_logo(Cpu *c) {
    (void)c;
    rt_log("oe: the game finished its startup logo");
}
HOS_OVERRIDE("_ZN2nn2oe17FinishStartupLogoEv", ov_finish_logo, HOS_STUB,
             "logged, no effect: no system logo is drawn on the host; donor ov_system.c:222");

// bool TryPopLaunchParameter(size_t *size, void *buffer, size_t capacity): started directly, no parameter
static void ov_launch_parameter(Cpu *c) {
    uint64_t zero = 0;
    out_put(c, "nn::oe::TryPopLaunchParameter", arg_x(c, 0), &zero, sizeof zero);
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn2oe21TryPopLaunchParameterEPmPvm", ov_launch_parameter, HOS_IMPLEMENTED,
             "no launch parameter: the program is started directly; donor ov_system.c:225");

// Queries and setters of SDK 22 the host answers with nothing: a PC window has no such setting, and the honest answer
// of an Is* query is "no". Left out on purpose: ExitApplication, RestartProgram and the RequestTo* family, which must
// end or restart the process.
#define WHY_NOTHING "accepted, no effect: a PC window has no such setting; donor ov_system.c:324"
#define WHY_NO "answers no (0): the host is not a console with that feature; donor ov_system.c:324"
#define WHY_ENUM0 "answers enum value 0, the first value: there is no console to ask; donor ov_system.c:324"
HOS_OVERRIDE("_ZN2nn2oe16InitializeAppletEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe17EnableCrashReportEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe18DisableCrashReportEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe16EnableScreenShotEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe17DisableScreenShotEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe13MuteRecordingEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe15UnmuteRecordingEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe15EnableRecordingEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe16DisableRecordingEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe18ReportUserIsActiveEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe26ReportUserIsActiveInternalEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe20SetAudioOutputPolicyENS0_17AudioOutputPolicyE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe20SetupGpuErrorHandlerEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe22SetCopyrightVisibilityEb", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe21SetExpectedThemeColorENS0_14ThemeColorTypeE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe23BeginBlockingHomeButtonEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe21EndBlockingHomeButtonEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe29BeginAutoSleepDisabledSectionEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe27EndAutoSleepDisabledSectionEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe26SetGraphicsTargetFrameRateEi", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe30SetGraphicsTargetFrameRateAutoEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe23SetWirelessPriorityModeENS0_20WirelessPriorityModeE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe27SetTvPowerStateMatchingModeENS0_24TvPowerStateMatchingModeE", ov_zero, HOS_STUB,
             WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe25BeginMediaPlaybackSectionEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe23EndMediaPlaybackSectionEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe33EnableAlbumImageTakenNotificationEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe34DisableAlbumImageTakenNotificationEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe35EnableStartupLogoDisappearedMessageEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe31SetRtcModeChangedMessageEnabledEb", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe22SetCpuOverclockEnabledEb", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe29SetScreenShotImageOrientationENS_5album16ImageOrientationE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe38SetHidInputMagnificationForApplicationERKNS0_10LayoutRectE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe20NotifyRunningForTestEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe33SuppressDisablingSleepTemporarilyENS_8TimeSpanE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe29SetDelayTimeToAbortOnGpuErrorENS_8TimeSpanE", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe34EnableAllThreadDumpOnCrashForDebugEb", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe19SetGpuQuotaForDebugEj", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe21ClearGpuQuotaForDebugEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe22SetScreenRatioForDebugEjj", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe24ClearScreenRatioForDebugEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe20SwitchLcdBacklightOnEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe21SwitchLcdBacklightOffEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe23EnableGamePlayRecordingEPvm", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe22PauseGamePlayRecordingEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe24RestartGamePlayRecordingEv", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe30InitializeCopyrightFrameBufferEPvm", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe27SetPerformanceConfigurationEi", ov_zero, HOS_STUB, WHY_NOTHING);
HOS_OVERRIDE("_ZN2nn2oe14IsSleepEnabledEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe19IsAutoSleepDisabledEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe21IsCpuOverclockEnabledEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe26IsLanguageSelectionLimitedEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe27IsApplicationLicenseLimitedEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe28IsGamePlayRecordingSupportedEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe26IsDisablingSleepSuppressedEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe24IsInStoreDemoModeEnabledEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe19IsTensorRtSupportedEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe37IsUserInactivityDetectionTimeExtendedEv", ov_zero, HOS_IMPLEMENTED, WHY_NO);
HOS_OVERRIDE("_ZN2nn2oe24GetServerEnvironmentTypeEv", ov_zero, HOS_STUB, WHY_ENUM0);
HOS_OVERRIDE("_ZN2nn2oe26GetHdcpAuthenticationStateEv", ov_zero, HOS_STUB, WHY_ENUM0);

// void GetDisplayVersion(DisplayVersion *out): char[16], the version the title shows, which a console reads from the
// program's control data. This program is only ever built from update 1.1.1 (README; oot2 prepare checks the dump), so
// that is the answer until control data is prepared; the donor answers the same string.
static void ov_oe_display_version(Cpu *c) {
    char v[16] = "1.1.1";
    out_put(c, "nn::oe::GetDisplayVersion", arg_x(c, 0), v, sizeof v);
}
HOS_OVERRIDE("_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE", ov_oe_display_version, HOS_IMPLEMENTED,
             "1.1.1, the one update this program is built from (README), not read from control data; donor ov_system.c:327");

// void GetDeviceNickName(DeviceNickName *out): char[0x80]
static void ov_oe_device_nickname(Cpu *c) {
    char v[0x80] = "Switch 2";
    out_put(c, "nn::oe::GetDeviceNickName", arg_x(c, 0), v, sizeof v);
}
HOS_OVERRIDE("_ZN2nn2oe17GetDeviceNickNameEPNS0_14DeviceNickNameE", ov_oe_device_nickname, HOS_IMPLEMENTED,
             "fixed device nickname: the host has no console name; donor ov_system.c:332");

// void GetPseudoDeviceId(Uuid *out): a fixed id, the host has no console
static void ov_oe_pseudo_device_id(Cpu *c) {
    uint8_t v[16];
    for (int i = 0; i < 16; i++) v[i] = (uint8_t)(0xa0 + i);
    out_put(c, "nn::oe::GetPseudoDeviceId", arg_x(c, 0), v, sizeof v);
}
HOS_OVERRIDE("_ZN2nn2oe17GetPseudoDeviceIdEPNS_4util4UuidE", ov_oe_pseudo_device_id, HOS_IMPLEMENTED,
             "fixed pseudo device id: the host has no console; donor ov_system.c:337");

// RegionCode GetRegionCode(): Usa (1)
static void ov_oe_region_code(Cpu *c) { ret_x(c, 1); }
HOS_OVERRIDE("_ZN2nn2oe13GetRegionCodeEv", ov_oe_region_code, HOS_IMPLEMENTED,
             "region code Usa (1), fixed; donor ov_system.c:341");

// int GetPreviousProgramIndex(): none, the program was started directly and not by an ExecuteProgram chain. Also the
// applet:: form.
static void ov_previous_program(Cpu *c) { ret_x(c, (uint32_t)-1); }
HOS_OVERRIDE("_ZN2nn2oe23GetPreviousProgramIndexEv", ov_previous_program, HOS_IMPLEMENTED,
             "-1, no previous program: started directly; donor ov_system.c:342");
HOS_OVERRIDE("_ZN2nn6applet23GetPreviousProgramIndexEv", ov_previous_program, HOS_IMPLEMENTED,
             "-1, no previous program: started directly; donor ov_system.c:256");

// void GetDefaultDisplayResolution(int *width, int *height): the docked 1080p panel
static void ov_oe_display_resolution(Cpu *c) {
    int32_t w = 1920, h = 1080;
    out_put(c, "nn::oe::GetDefaultDisplayResolution", arg_x(c, 0), &w, sizeof w);
    out_put(c, "nn::oe::GetDefaultDisplayResolution", arg_x(c, 1), &h, sizeof h);
}
HOS_OVERRIDE("_ZN2nn2oe27GetDefaultDisplayResolutionEPiS1_", ov_oe_display_resolution, HOS_IMPLEMENTED,
             "1920x1080, the docked panel; donor ov_system.c:344");

// int GetCurrentGraphicsScale(): an enum of three levels, 0 to 2, of which the highest is answered: the host is a
// docked console with the most to give. Unverified against hardware: what the three levels mean is not known.
static void ov_graphics_scale(Cpu *c) { ret_x(c, 2); }
HOS_OVERRIDE("_ZN2nn2oe23GetCurrentGraphicsScaleEv", ov_graphics_scale, HOS_STUB,
             "highest level (2): docked, most to give, unverified; donor ov_system.c:874");
HOS_OVERRIDE("_ZN2nn2oe42SetGraphicsScaleChangedNotificationEnabledEb", ov_zero, HOS_STUB,
             "accepted, no effect: the scale never changes on the host; donor ov_system.c:876");
HOS_OVERRIDE("_ZN2nn2oe24SetAlbumImageOrientationENS_5album16ImageOrientationE", ov_zero, HOS_STUB,
             "accepted, no effect: there is no capture album; donor ov_system.c:878");
HOS_OVERRIDE("_ZN2nn2oe31EnterExitRequestHandlingSectionEv", ov_zero, HOS_STUB,
             "accepted, no effect: no exit request arrives from a host system applet; donor ov_system.c:880");
HOS_OVERRIDE("_ZN2nn2oe31LeaveExitRequestHandlingSectionEv", ov_zero, HOS_STUB,
             "accepted, no effect: no exit request arrives from a host system applet; donor ov_system.c:880");

// ==== applet and am

// AppletResourceUserId GetAppletResourceUserId(): one application, one stable non-zero id
static void ov_aruid(Cpu *c) { ret_x(c, 1); }
HOS_OVERRIDE("_ZN2nn6applet23GetAppletResourceUserIdEv", ov_aruid, HOS_IMPLEMENTED,
             "one stable non-zero id: one application; donor ov_system.c:258");
HOS_OVERRIDE("_ZN2nn2am23GetAppletResourceUserIdEv", ov_aruid, HOS_IMPLEMENTED,
             "one stable non-zero id: one application; donor ov_system.c:258");

// ==== nn::prepo: play reports stay local, nothing is sent
static uint64_t reports;
static void note_report(void) {
    uint64_t n = __atomic_add_fetch(&reports, 1, __ATOMIC_RELAXED);
    if (n % 100 == 1) rt_log("prepo: play report kept local (not sent), %llu so far", (unsigned long long)n);
}

// Result PlayReport::Save() and Save(const account::Uid &)
static void ov_report_save(Cpu *c) {
    note_report();
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5prepo10PlayReport4SaveEv", ov_report_save, HOS_STUB,
             "success, counted and not kept: reports stay local; donor ov_system.c:262");
HOS_OVERRIDE("_ZN2nn5prepo10PlayReport4SaveERKNS_7account3UidE", ov_report_save, HOS_STUB,
             "success, counted and not kept: reports stay local; donor ov_system.c:262");

// Result RequestImmediateTransmission(), Result AddInternetConnectionStatus(PlayReport *, const char *, ...)
HOS_OVERRIDE("_ZN2nn5prepo28RequestImmediateTransmissionEv", ov_zero, HOS_STUB,
             "success, nothing is transmitted; donor ov_system.c:267");
HOS_OVERRIDE("_ZN2nn5prepo27AddInternetConnectionStatusEPNS0_10PlayReportEPKcS4_S4_", ov_zero, HOS_STUB,
             "success, no connection status to add; donor ov_system.c:267");

// AsyncContext PlayReport::Send(const account::Uid &): the object comes back through x8. It is built by the guest's own
// AsyncContext(Result) constructor: a context made from a Result has no IPC object, and its GetCompletionEvent hands
// out an event already signalled, so the game's wait on it ends at once. The game sends a report after its opening
// and waits on that event; with no prepo service the send would otherwise stop the game.
static void ov_prepo_send(Cpu *c) {
    note_report();
    uint64_t out = c->x[8];
    uint64_t ctor = hos_sdk(c, "_ZN2nn5prepo12AsyncContextC1ENS_6ResultE");
    hos_call(c, ctor, 2, out, (uint64_t)HOS_R_OK);
    ret_x(c, out);
}
HOS_OVERRIDE("_ZN2nn5prepo10PlayReport4SendERKNS_7account3UidE", ov_prepo_send, HOS_STUB,
             "a completed AsyncContext built by the guest's Result constructor, nothing sent; donor ov_system.c:271");

// An event that never fires, the last of the event block; the over-frequency event of prepo is one: reports stay local,
// so the limit of saves per period is never exceeded. The return is a pointer in x0 whether the SDK type is a reference
// or a pointer. Manual clear, which a never-signalled event cannot tell from auto clear.
static void ov_prepo_limit_event(Cpu *c) {
    uint64_t ev = event_at(NETCLOCK_SLOTS);
    hos_bind_system_event(c, ev, 0);
    ret_x(c, ev);
}
HOS_OVERRIDE("_ZN2nn5prepo30GetOverSaveFrequencyLimitEventEv", ov_prepo_limit_event, HOS_OFFLINE,
             "a bound event that never fires: reports stay local so no limit is hit; the services study (no donor)");

// ==== nn::nfp: no NFC reader
static int32_t nfp_state;

// Result Initialize() and Finalize(), State GetState()
static void ov_nfp_initialize(Cpu *c) {
    __atomic_store_n(&nfp_state, 1, __ATOMIC_RELAXED);
    ret_x(c, HOS_R_OK);
}
static void ov_nfp_finalize(Cpu *c) {
    __atomic_store_n(&nfp_state, 0, __ATOMIC_RELAXED);
    ret_x(c, HOS_R_OK);
}
static void ov_nfp_get_state(Cpu *c) { ret_x(c, (uint64_t)__atomic_load_n(&nfp_state, __ATOMIC_RELAXED)); }
HOS_OVERRIDE("_ZN2nn3nfp10InitializeEv", ov_nfp_initialize, HOS_OFFLINE,
             "success, library state initialised: no reader to open; donor ov_system.c:284");
HOS_OVERRIDE("_ZN2nn3nfp8FinalizeEv", ov_nfp_finalize, HOS_OFFLINE,
             "success, library state cleared; donor ov_system.c:285");
HOS_OVERRIDE("_ZN2nn3nfp8GetStateEv", ov_nfp_get_state, HOS_OFFLINE,
             "the library state the game set; donor ov_system.c:286");

// Result ListDevices(DeviceHandle *out, int *count, int max)
static void ov_nfp_list(Cpu *c) {
    int32_t zero = 0;
    out_put(c, "nn::nfp::ListDevices", arg_x(c, 1), &zero, sizeof zero);
    ret_x(c, NFP_R_DEVICE_NOT_FOUND);
}
HOS_OVERRIDE("_ZN2nn3nfp11ListDevicesEPNS0_12DeviceHandleEPii", ov_nfp_list, HOS_OFFLINE,
             "no device: count 0 and device-not-found, as with no reader; donor ov_system.c:288");

// Result AttachAvailabilityChangeEvent(os::SystemEventType *event): the availability of a reader never changes. The
// event is bound and never signalled.
static void ov_nfp_availability(Cpu *c) {
    uint64_t ev = arg_x(c, 0);
    if (!ev) rt_guest_stop(c, "nn::nfp::AttachAvailabilityChangeEvent: null event");
    hos_bind_system_event(c, ev, 0);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn3nfp29AttachAvailabilityChangeEventEPNS_2os15SystemEventTypeE", ov_nfp_availability, HOS_OFFLINE,
             "success with a bound event that never fires: no reader appears; donor ov_system.c:294");

// DeviceState GetDeviceState(const DeviceHandle &): 0, the state before any detection
static void ov_nfp_device_state(Cpu *c) { ret_x(c, 0); }
HOS_OVERRIDE("_ZN2nn3nfp14GetDeviceStateERKNS0_12DeviceHandleE", ov_nfp_device_state, HOS_OFFLINE,
             "state 0: no device ever detected; donor ov_system.c:301");

// Every call that needs a device answers that there is none.
static void ov_nfp_no_device(Cpu *c) { ret_x(c, NFP_R_DEVICE_NOT_FOUND); }
#define WHY_NO_DEVICE "device-not-found: no NFC reader on the host; donor ov_system.c:300"
#define WHY_NO_DEVICE_ADDED "device-not-found: no NFC reader; same answer as the donor's device calls (ov_system.c:300)"
HOS_OVERRIDE("_ZN2nn3nfp5MountERKNS0_12DeviceHandleENS0_9ModelTypeENS0_11MountTargetE", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp9GetNpadIdEPjRKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp19AttachActivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE", ov_nfp_no_device,
             HOS_OFFLINE, WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp21AttachDeactivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE", ov_nfp_no_device,
             HOS_OFFLINE, WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp14StartDetectionERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp13StopDetectionERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp10GetTagInfoEPNS0_7TagInfoERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE);
HOS_OVERRIDE("_ZN2nn3nfp13GetCommonInfoEPNS0_10CommonInfoERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp15GetRegisterInfoEPNS0_12RegisterInfoERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp18GetApplicationAreaEPvPmRKNS0_12DeviceHandleEm", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp18SetApplicationAreaERKNS0_12DeviceHandleEPKvm", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp19OpenApplicationAreaERKNS0_12DeviceHandleEj", ov_nfp_no_device, HOS_OFFLINE,
             WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp21CreateApplicationAreaERKNS0_12DeviceHandleERKNS0_25ApplicationAreaCreateInfoE",
             ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp23RecreateApplicationAreaERKNS0_12DeviceHandleERKNS0_25ApplicationAreaCreateInfoE",
             ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp5FlushERKNS0_12DeviceHandleE", ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp13StartRestorerEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE",
             ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp19StartGameDataEraserEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE",
             ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp29StartNicknameAndOwnerSettingsEPNS0_12DeviceHandleEPbPNS0_12RegisterInfoERKNS0_24"
             "AmiiboSettingsStartParamERKNS0_7TagInfoE",
             ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);
HOS_OVERRIDE("_ZN2nn3nfp29StartNicknameAndOwnerSettingsEPNS0_12DeviceHandleEPbPNS0_12RegisterInfoERKNS0_24"
             "AmiiboSettingsStartParamERKNS0_7TagInfoERKS4_",
             ov_nfp_no_device, HOS_OFFLINE, WHY_NO_DEVICE_ADDED);

// ==== start-up initialisers of modules that need a service or a network: they succeed without opening anything
// (nn::nfc, nn::prepo, nn::friends, nn::bcat, nn::ldn). The calls that follow are not answered here.
#define WHY_INIT "success, no service opened: the host serves none and is offline; donor ov_system.c:938-943"
HOS_OVERRIDE("_ZN2nn3nfc10InitializeEv", ov_zero, HOS_STUB, WHY_INIT);
HOS_OVERRIDE("_ZN2nn3nfc8FinalizeEv", ov_zero, HOS_STUB, WHY_INIT);
HOS_OVERRIDE("_ZN2nn5prepo10InitializeEv", ov_zero, HOS_STUB, WHY_INIT);
HOS_OVERRIDE("_ZN2nn7friends10InitializeEv", ov_zero, HOS_STUB, WHY_INIT);
HOS_OVERRIDE("_ZN2nn4bcat10InitializeEv", ov_zero, HOS_STUB, WHY_INIT);
HOS_OVERRIDE("_ZN2nn3ldn10InitializeEv", ov_zero, HOS_STUB, WHY_INIT);
HOS_OVERRIDE("_ZN2nn3ldn8FinalizeEv", ov_zero, HOS_STUB, WHY_INIT);

// ==== nn::err and nn::diag

// void err::ShowError(Result): the game's error dialog, logged
static void ov_show_error(Cpu *c) {
    uint32_t r = (uint32_t)arg_x(c, 0);
    char at[96];
    rt_log("err: game error dialog %04u-%04u (result %#x) from %s", 2000 + (r & 0x1ff), (r >> 9) & 0x1fff, r,
           hos_caller(c, at, (int)sizeof at));
}
HOS_OVERRIDE("_ZN2nn3err9ShowErrorENS_6ResultE", ov_show_error, HOS_STUB,
             "logged, returns at once: no error applet on the host; donor ov_system.c:304");

static void ov_err_context(Cpu *c) {
    (void)c;
    rt_log("err: error contexts stay local");
}
HOS_OVERRIDE("_ZN2nn3err26InitializeContextWriterApiEv", ov_err_context, HOS_STUB,
             "logged, no context writer service to open; donor ov_system.c:381");

static void ov_diag_startup(Cpu *c) {
    (void)c;
    rt_log("diag: host diagnostics");
}
HOS_OVERRIDE("nndiagStartup", ov_diag_startup, HOS_STUB,
             "logged, nothing to set up: the host reports crashes; donor ov_system.c:312");

static void ov_abort_observer(Cpu *c) {
    (void)c;
    rt_log("diag: the host crash reporter observes aborts");
}
HOS_OVERRIDE("_ZN2nn4diag34InitializeApplicationAbortObserverEv", ov_abort_observer, HOS_STUB,
             "logged, no observer to install: the host handles aborts; donor ov_system.c:383");

// ==== nn::os fibre internals. The four public fibre entry points are host functions (runtime/core/fibre.c), so the
// sdk's own context switch, its fibre trampoline and its alias stacks have no caller left in OoT's modules (each is
// reached only through a PLT slot, from those four or from the trampoline). One reached would switch guest stacks inside
// lifted code, which host fibres exist to prevent, or read a saved context host fibres never write (nn::diag's
// backtrace of a suspended fibre): each is a named fatal.
static void ov_fibre_internal(Cpu *c) {
    char at[96];
    hos_gap(c, "fibre", "%s from %s: the SDK's own fibre machinery, which host fibres replace", rt_hostfn_name(c->pc),
            hos_caller(c, at, (int)sizeof at));
}
HOS_OVERRIDE("_ZN2nn2os6detail20DispatchFiberContextEPPNS1_22FiberContextForHorizonES4_", ov_fibre_internal, HOS_GAP,
             "named gap: the SDK's register and stack swap, replaced by host fibres; the nn::os fibre support");
HOS_OVERRIDE("_ZN2nn2os6detail18FiberImplByHorizon10InitializeEPNS0_9FiberTypeE", ov_fibre_internal, HOS_GAP,
             "named gap: the SDK's first fibre frame, replaced by host fibres; the nn::os fibre support");
HOS_OVERRIDE("_ZN2nn2os6detail18FiberImplByHorizon13MapAliasStackEPPvPKvm", ov_fibre_internal, HOS_GAP,
             "named gap: fibres run on the stack they were given; the nn::os fibre support");
HOS_OVERRIDE("_ZN2nn2os6detail18FiberImplByHorizon15UnmapAliasStackEPKvS4_m", ov_fibre_internal, HOS_GAP,
             "named gap: fibres run on the stack they were given; the nn::os fibre support");
HOS_OVERRIDE("_ZN2nn2os6detail17CallFiberFunctionEPvPFPNS0_9FiberTypeES2_E", ov_fibre_internal, HOS_GAP,
             "named gap: the SDK's fibre trampoline, replaced by host fibres; the nn::os fibre support");
HOS_OVERRIDE("_ZN2nn4diag6detail9Backtrace10InitializeEPKNS_2os9FiberTypeE", ov_fibre_internal, HOS_GAP,
             "named gap: a suspended fibre's saved context is not kept by host fibres; the nn::os fibre support");

// ==== nn::aoc: no add-on content
static void ov_aoc_initialize(Cpu *c) {
    (void)c;
    rt_log("aoc: no add-on content");
}
HOS_OVERRIDE("_ZN2nn3aoc10InitializeEv", ov_aoc_initialize, HOS_ABSENT,
             "logged, no add-on content service to open; donor ov_system.c:313");

// int ListAddOnContent(AddOnContentIndex *out, int offset, int count): no entry
static void ov_aoc_list(Cpu *c) {
    int32_t offset = (int32_t)arg_x(c, 1), count = (int32_t)arg_x(c, 2);
    if (offset < 0 || count < 0) rt_guest_stop(c, "nn::aoc::ListAddOnContent: range %d+%d", offset, count);
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn3aoc16ListAddOnContentEPiii", ov_aoc_list, HOS_ABSENT,
             "no entries listed: there is no add-on content; donor ov_system.c:316");
HOS_OVERRIDE("_ZN2nn3aoc17CountAddOnContentEv", ov_zero, HOS_ABSENT,
             "count 0: there is no add-on content; donor ov_system.c:377");
HOS_OVERRIDE("_ZN2nn3aoc8FinalizeEv", ov_zero, HOS_ABSENT, "accepted: nothing was opened; donor ov_system.c:378");
HOS_OVERRIDE("_ZN2nn3aoc21GetAddOnContentBaseIdEv", ov_zero, HOS_ABSENT,
             "base id 0: there is no add-on content; donor ov_system.c:379");
HOS_OVERRIDE("_ZN2nn3aoc28CheckAddOnContentMountStatusEv", ov_zero, HOS_ABSENT,
             "success: nothing is waiting to be mounted; donor ov_system.c:380");

// ==== nn::pctl: no parental controls
static void ov_pctl_initialize(Cpu *c) {
    (void)c;
    rt_log("pctl: no parental controls");
}
HOS_OVERRIDE("_ZN2nn4pctl10InitializeEv", ov_pctl_initialize, HOS_STUB,
             "logged, no parental control service to open; donor ov_system.c:382");

// u8 CheckAgcPermissionUnderMinimumAge(const account::Uid &): the SDK's own function asks the parental control service
// and returns one byte; the game branches on it as a small enumeration (0 and 1 among others), not as a Result. Without
// parental controls the answer is 0. Two other runtimes answer 0 for this function: the jaigner-oot runtime, which runs
// this same game's SDK function (its comment reads "no parental controls ... permitted"), and nxnative, for other games'
// builds of it ("never"). The two disagree on what to call the value and neither documents what 1, 2 and 3 mean, so the
// value is the donors' and its meaning is not settled here.
static void ov_pctl_agc(Cpu *c) { ret_x(c, 0); }
HOS_OVERRIDE("_ZN2nn4pctl33CheckAgcPermissionUnderMinimumAgeERKNS_7account3UidE", ov_pctl_agc, HOS_ABSENT,
             "0, as a console with no parental controls set and both donors that answer it (jaigner-oot, nxnative); "
             "the meaning of the other values is not recorded");

// ==== nn::account: one offline local user with a constant Uid. A console's own user would differ; the game only needs
// one stable Uid, which the donor kept in a file.
typedef struct { uint64_t lo, hi; } Uid;
typedef struct { Uid uid; uint64_t token; } UserHandle;   // the host's own layout: the game passes it by reference only
static const Uid local_uid = {0x6f6f74326c6f6361ull, 1};  // arbitrary and non-zero
static uint64_t tokens[64], next_token = 1;

static int same_uid(const Uid *u) { return u->lo == local_uid.lo && u->hi == local_uid.hi; }

// Holding sys_lock.
static int valid_handle(const UserHandle *h) {
    if (!same_uid(&h->uid)) return 0;
    for (int i = 0; i < 64; i++)
        if (tokens[i] && tokens[i] == h->token) return 1;
    return 0;
}

HOS_OVERRIDE("_ZN2nn7account10InitializeEv", ov_nothing, HOS_OFFLINE,
             "accepted: one offline local user, no account service; donor ov_system.c:468");

// bool TryOpenPreselectedUser(UserHandle *out)
static void ov_open_user(Cpu *c) {
    uint64_t out = arg_x(c, 0);
    if (!out) rt_guest_stop(c, "nn::account::TryOpenPreselectedUser: null handle");
    UserHandle h;
    h.uid = local_uid;
    h.token = 0;
    int ok = 0;
    plat_mutex_lock(&sys_lock);
    for (int i = 0; i < 64 && !ok; i++)
        if (!tokens[i]) {
            h.token = tokens[i] = next_token++;
            ok = 1;
        }
    plat_mutex_unlock(&sys_lock);
    if (ok) out_put(c, "nn::account::TryOpenPreselectedUser", out, &h, sizeof h);
    ret_x(c, (uint64_t)ok);
}
HOS_OVERRIDE("_ZN2nn7account22TryOpenPreselectedUserEPNS0_10UserHandleE", ov_open_user, HOS_OFFLINE,
             "opens the one local user with a fresh token; donor ov_system.c:475");

// void CloseUser(const UserHandle &)
static void ov_close_user(Cpu *c) {
    UserHandle h;
    in_get(c, "nn::account::CloseUser", &h, arg_x(c, 0), sizeof h);
    plat_mutex_lock(&sys_lock);
    int ok = valid_handle(&h);
    for (int i = 0; ok && i < 64; i++)
        if (tokens[i] == h.token) tokens[i] = 0;
    plat_mutex_unlock(&sys_lock);
    if (!ok) rt_guest_stop(c, "nn::account::CloseUser: not an open user handle");
}
HOS_OVERRIDE("_ZN2nn7account9CloseUserERKNS0_10UserHandleE", ov_close_user, HOS_OFFLINE,
             "releases the token, a handle that is not open stops the guest; donor ov_system.c:492");

// Result GetUserId(Uid *out, const UserHandle &)
static void ov_user_id(Cpu *c) {
    UserHandle h;
    in_get(c, "nn::account::GetUserId", &h, arg_x(c, 1), sizeof h);
    plat_mutex_lock(&sys_lock);
    int ok = valid_handle(&h);
    plat_mutex_unlock(&sys_lock);
    if (ok && arg_x(c, 0)) out_put(c, "nn::account::GetUserId", arg_x(c, 0), &local_uid, sizeof local_uid);
    ret_x(c, ok && arg_x(c, 0) ? HOS_R_OK : ACCOUNT_R_INVALID);
}
HOS_OVERRIDE("_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE", ov_user_id, HOS_OFFLINE,
             "the local Uid for an open handle, else an invalid-handle result; donor ov_system.c:502");

// Result GetLastOpenedUser(Uid *out): the local user, the only one the host ever opens
static void ov_last_opened_user(Cpu *c) {
    if (!arg_x(c, 0)) rt_guest_stop(c, "nn::account::GetLastOpenedUser: null output");
    out_put(c, "nn::account::GetLastOpenedUser", arg_x(c, 0), &local_uid, sizeof local_uid);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn7account17GetLastOpenedUserEPNS0_3UidE", ov_last_opened_user, HOS_OFFLINE,
             "the local user, the only one the host opens; donor ov_system.c:513");

// Result GetUserExistence(bool *out, const Uid &)
static void ov_user_existence(Cpu *c) {
    Uid u;
    in_get(c, "nn::account::GetUserExistence", &u, arg_x(c, 1), sizeof u);
    uint8_t exists = (uint8_t)same_uid(&u);
    if (!arg_x(c, 0)) {
        ret_x(c, ACCOUNT_R_INVALID);
        return;
    }
    out_put(c, "nn::account::GetUserExistence", arg_x(c, 0), &exists, sizeof exists);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn7account16GetUserExistenceEPbRKNS0_3UidE", ov_user_existence, HOS_OFFLINE,
             "true for the local Uid only; donor ov_system.c:523");

// Result GetNickname(Nickname *out, const Uid &): 33 bytes, NUL-terminated UTF-8
static void ov_nickname(Cpu *c) {
    Uid u;
    in_get(c, "nn::account::GetNickname", &u, arg_x(c, 1), sizeof u);
    if (!arg_x(c, 0) || !same_uid(&u)) {
        ret_x(c, ACCOUNT_R_INVALID);
        return;
    }
    char nick[33] = "Local";
    out_put(c, "nn::account::GetNickname", arg_x(c, 0), nick, sizeof nick);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE", ov_nickname, HOS_OFFLINE,
             "a fixed nickname for the local Uid; donor ov_system.c:535");

// Network service accounts are never available offline: every call answers failure. The AsyncContext the game builds
// for them is reached only through its four members below, all overridden, so its layout never matters: construction
// and destruction write nothing, GetSystemEvent hands out an already signalled event and GetResult is the failure.
// See the services study.
#define WHY_NSA "failure at once: the network service account is never available offline"
static void ov_account_failure(Cpu *c) { ret_x(c, HOS_R_UNAVAILABLE); }
HOS_OVERRIDE("_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE", ov_account_failure,
             HOS_OFFLINE, WHY_NSA);
HOS_OVERRIDE("_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE",
             ov_account_failure, HOS_OFFLINE, WHY_NSA);
HOS_OVERRIDE("_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE",
             ov_account_failure, HOS_OFFLINE, WHY_NSA);
HOS_OVERRIDE("_ZN2nn7account12AsyncContext9GetResultEv", ov_account_failure, HOS_OFFLINE,
             "the failure: no request is ever sent");

// Result LoadNetworkServiceAccountIdTokenCache(size_t *size, char *buffer, size_t capacity, const UserHandle &)
static void ov_account_token_cache(Cpu *c) {
    uint64_t zero = 0;
    out_put(c, "nn::account::LoadNetworkServiceAccountIdTokenCache", arg_x(c, 0), &zero, sizeof zero);
    ret_x(c, HOS_R_UNAVAILABLE);
}
HOS_OVERRIDE("_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE", ov_account_token_cache,
             HOS_OFFLINE, "size 0 and failure: no token cache offline");

HOS_OVERRIDE("_ZN2nn7account12AsyncContextC1Ev", ov_nothing, HOS_OFFLINE,
             "writes nothing: no IPC object is created offline");
HOS_OVERRIDE("_ZN2nn7account12AsyncContextD1Ev", ov_nothing, HOS_OFFLINE,
             "writes nothing: no IPC object to close");

// Result AsyncContext::GetSystemEvent(os::SystemEvent *event): an already signalled event bound to the game's. The
// SystemEvent holds its SystemEventType first, which is what hos_bind_system_event takes. Manual clear.
static void ov_account_context_event(Cpu *c) {
    uint64_t ev = arg_x(c, 1);
    if (!ev) rt_guest_stop(c, "nn::account::AsyncContext::GetSystemEvent: null event");
    hos_signal(hos_bind_system_event(c, ev, 0));
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn7account12AsyncContext14GetSystemEventEPNS_2os11SystemEventE", ov_account_context_event,
             HOS_OFFLINE, "an already signalled event: the request finished at once with failure");

// ==== nn::time: the host clock and time zone. User and network clocks follow the host wall clock, and the steady clock
// counts seconds on the host's monotonic timer, with one random source id per process. ClockSnapshot has the public
// 0xD0-byte layout, because game code reads its fields through inline accessors.
typedef struct { int64_t value; uint8_t source[16]; } SteadyClockTimePoint;
typedef struct { int64_t offset; SteadyClockTimePoint point; } SystemClockContext;
typedef struct { int16_t year; int8_t month, day, hour, minute, second, pad; } CalendarTime;
typedef struct {
    int32_t day_of_week, day_of_year;
    char name[8];
    uint8_t dst, pad[3];
    int32_t utc_offset;
} CalendarInfo;
typedef struct {
    SystemClockContext user, network;
    int64_t user_time, network_time;
    CalendarTime user_calendar, network_calendar;
    CalendarInfo user_info, network_info;
    SteadyClockTimePoint steady;
    char location[0x24];
    uint8_t automatic_correction, type, pad[2];
} ClockSnapshot;
_Static_assert(sizeof(SteadyClockTimePoint) == 0x18 && sizeof(CalendarTime) == 8 && sizeof(CalendarInfo) == 0x18,
               "nn::time types");
_Static_assert(sizeof(ClockSnapshot) == 0xd0, "nn::time::ClockSnapshot");

static uint8_t source_id[16];
static int source_ready;
static int32_t time_ready;

// The steady clock's current point: whole seconds of the host's monotonic timer under this process's source id.
static SteadyClockTimePoint steady_now(void) {
    SteadyClockTimePoint p;
    plat_mutex_lock(&sys_lock);
    if (!source_ready) {
        if (!plat_random(source_id, sizeof source_id)) rt_fatal("nn::time: no random bytes from the system");
        source_ready = 1;
    }
    memcpy(p.source, source_id, sizeof p.source);
    plat_mutex_unlock(&sys_lock);
    p.value = (int64_t)(plat_now_ns() / 1000000000ull);
    return p;
}

static int64_t posix_now(void) { return (int64_t)time(NULL); }

// The host's local time of a POSIX time: 0 outside what the C library converts.
static int host_localtime(int64_t t, struct tm *out) {
    time_t v = (time_t)t;
    if ((int64_t)v != t) return 0;
#ifdef _WIN32
    return localtime_s(out, &v) == 0;
#else
    return localtime_r(&v, out) != NULL;
#endif
}

// Days since 1970-01-01 of a proleptic Gregorian date, and back (H. Hinnant's civil calendar algorithms).
static int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    int64_t era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - era * 400);
    unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int64_t)doe - 719468;
}

static void civil_from_days(int64_t z, int64_t *y, unsigned *m, unsigned *d) {
    z += 719468;
    int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned mp = (5 * doy + 2) / 153;
    *d = doy - (153 * mp + 2) / 5 + 1;
    *m = mp < 10 ? mp + 3 : mp - 9;
    *y = (int64_t)yoe + era * 400 + (*m <= 2);
}

static int64_t floor_div(int64_t a, int64_t b) { return a / b - (a % b != 0 && (a < 0) != (b < 0)); }

// UTC calendar fields of a POSIX time. 0 if the year does not fit CalendarTime.
static int utc_calendar(int64_t t, CalendarTime *ct, CalendarInfo *info) {
    int64_t days = floor_div(t, 86400), sec = t - days * 86400, y;
    unsigned m, d;
    civil_from_days(days, &y, &m, &d);
    ct->year = (int16_t)y;
    ct->month = (int8_t)m;
    ct->day = (int8_t)d;
    ct->hour = (int8_t)(sec / 3600);
    ct->minute = (int8_t)(sec / 60 % 60);
    ct->second = (int8_t)(sec % 60);
    ct->pad = 0;
    if (info) {
        memset(info, 0, sizeof *info);
        info->day_of_week = (int32_t)(days + 4 - floor_div(days + 4, 7) * 7);   // 1970-01-01 was a Thursday
        info->day_of_year = (int32_t)(days - days_from_civil(y, 1, 1));
        memcpy(info->name, "UTC", 3);
    }
    return y >= -32768 && y <= 32767;
}

// Local calendar fields of a POSIX time in the host's time zone. The zone name is "UTC" or the UTC offset ("+03",
// "-0330"), as tzdata names zones without an abbreviation; the host has no abbreviations. 0 outside the C library's
// range.
static int local_calendar(int64_t t, CalendarTime *ct, CalendarInfo *info) {
    struct tm tm;
    if (!host_localtime(t, &tm)) return 0;
    int64_t as_utc = days_from_civil(tm.tm_year + 1900, (unsigned)tm.tm_mon + 1, (unsigned)tm.tm_mday) * 86400 +
                     tm.tm_hour * 3600 + tm.tm_min * 60 + tm.tm_sec;
    int32_t offset = (int32_t)(as_utc - t);
    ct->year = (int16_t)(tm.tm_year + 1900);
    ct->month = (int8_t)(tm.tm_mon + 1);
    ct->day = (int8_t)tm.tm_mday;
    ct->hour = (int8_t)tm.tm_hour;
    ct->minute = (int8_t)tm.tm_min;
    ct->second = (int8_t)tm.tm_sec;
    ct->pad = 0;
    if (info) {
        memset(info, 0, sizeof *info);
        info->day_of_week = tm.tm_wday;
        info->day_of_year = tm.tm_yday;
        int a = offset < 0 ? -offset : offset;
        if (!offset) memcpy(info->name, "UTC", 3);
        else if (a % 3600)
            snprintf(info->name, sizeof info->name, "%c%02d%02d", offset < 0 ? '-' : '+', a / 3600, a / 60 % 60);
        else
            snprintf(info->name, sizeof info->name, "%c%02d", offset < 0 ? '-' : '+', a / 3600);
        info->dst = tm.tm_isdst > 0;
        info->utc_offset = offset;
    }
    return 1;
}

// The IANA name of the zone's standard offset (the smaller of the offsets in January and July): Etc/GMT+5 is UTC-5
// (POSIX sign); "UTC" for a zone off whole hours.
static void location_name(int64_t now, char out[0x24]) {
    CalendarTime ct;
    CalendarInfo ci;
    memset(out, 0, 0x24);
    if (!local_calendar(now, &ct, &ci)) {
        snprintf(out, 0x24, "UTC");
        return;
    }
    int32_t base = ci.utc_offset;
    int64_t year = ct.year;
    for (unsigned month = 1; month <= 7; month += 6) {
        int64_t first = days_from_civil(year, month, 1) * 86400;
        if (local_calendar(first, &ct, &ci) && ci.utc_offset < base) base = ci.utc_offset;
    }
    if (!base || base % 3600) snprintf(out, 0x24, "UTC");
    else snprintf(out, 0x24, "Etc/GMT%+d", -base / 3600);
}

static void ov_time_initialize(Cpu *c) {
    __atomic_store_n(&time_ready, 1, __ATOMIC_RELAXED);
    ret_x(c, HOS_R_OK);
}
static void ov_time_finalize(Cpu *c) {
    __atomic_store_n(&time_ready, 0, __ATOMIC_RELAXED);
    ret_x(c, HOS_R_OK);
}
static void ov_time_ready(Cpu *c) { ret_x(c, (uint64_t)__atomic_load_n(&time_ready, __ATOMIC_RELAXED)); }
HOS_OVERRIDE("_ZN2nn4time10InitializeEv", ov_time_initialize, HOS_IMPLEMENTED,
             "success, the host clock needs no session; donor ov_system.c:661");
HOS_OVERRIDE("_ZN2nn4time8FinalizeEv", ov_time_finalize, HOS_IMPLEMENTED,
             "success, no session to close; donor ov_system.c:666");
HOS_OVERRIDE("_ZN2nn4time13IsInitializedEv", ov_time_ready, HOS_IMPLEMENTED,
             "the state the game set; donor ov_system.c:667");

// Result StandardUserSystemClock / StandardNetworkSystemClock::GetCurrentTime(PosixTime *out). The network clock
// follows the host wall clock as the donor answers it, although its availability task fails (below): NXNative answers
// failure here, which is an open question for the author.
static void ov_current_time(Cpu *c) {
    if (!arg_x(c, 0)) {
        ret_x(c, TIME_R_INVALID);
        return;
    }
    int64_t now = posix_now();
    out_put(c, "nn::time::GetCurrentTime", arg_x(c, 0), &now, sizeof now);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE", ov_current_time, HOS_IMPLEMENTED,
             "the host wall clock; donor ov_system.c:670");
HOS_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock14GetCurrentTimeEPNS0_9PosixTimeE", ov_current_time,
             HOS_OFFLINE, "host wall clock though the clock task fails (open question); donor ov_system.c:670");

// time_point StandardNetworkSystemClock::now(): the int64 seconds in x0, like the steady clock's
static void ov_network_now(Cpu *c) { ret_x(c, (uint64_t)posix_now()); }
HOS_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock3nowEv", ov_network_now, HOS_OFFLINE,
             "the host wall clock in seconds, as GetCurrentTime does; the game imports it and no donor hooks it");

// time_point StandardSteadyClock::now(): the int64 seconds in x0 only; the SDK's caller reads x0, and returning the
// 0x18-byte point through x8 would write through a stale pointer
static void ov_steady_now(Cpu *c) { ret_x(c, (uint64_t)steady_now().value); }
HOS_OVERRIDE("_ZN2nn4time19StandardSteadyClock3nowEv", ov_steady_now, HOS_IMPLEMENTED,
             "seconds of the host monotonic timer in x0 only, as the SDK's caller reads it; donor ov_system.c:679");

// int GetStandardSteadyClockCurrentTimespec(timespec *) and GetStandardUserSystemClockCurrentTimespec(timespec *): the
// SDK 22 libc installs them as its clock_gettime for CLOCK_MONOTONIC and CLOCK_REALTIME. Their own definitions ask the
// time service, which the host does not run, so clock_gettime failed and std::chrono::steady_clock::now() threw. 0 is
// success.
static void ov_steady_timespec(Cpu *c) {
    if (!arg_x(c, 0)) {
        ret_x(c, 22);   // EINVAL
        return;
    }
    uint64_t ns = plat_now_ns();
    int64_t ts[2] = {(int64_t)(ns / 1000000000ull), (int64_t)(ns % 1000000000ull)};
    out_put(c, "nn::time::GetStandardSteadyClockCurrentTimespec", arg_x(c, 0), ts, sizeof ts);
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn4time37GetStandardSteadyClockCurrentTimespecEP8timespec", ov_steady_timespec, HOS_IMPLEMENTED,
             "the host monotonic timer: the SDK libc's CLOCK_MONOTONIC; donor ov_system.c:685");

static void ov_user_timespec(Cpu *c) {
    if (!arg_x(c, 0)) {
        ret_x(c, 22);   // EINVAL
        return;
    }
    struct timespec now;
    if (!timespec_get(&now, TIME_UTC)) {
        ret_x(c, 5);   // EIO
        return;
    }
    int64_t ts[2] = {(int64_t)now.tv_sec, (int64_t)now.tv_nsec};
    out_put(c, "nn::time::GetStandardUserSystemClockCurrentTimespec", arg_x(c, 0), ts, sizeof ts);
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn4time41GetStandardUserSystemClockCurrentTimespecEP8timespec", ov_user_timespec, HOS_IMPLEMENTED,
             "the host wall clock: the SDK libc's CLOCK_REALTIME; donor ov_system.c:693");

// Result ToCalendarTime(CalendarTime *out, CalendarAdditionalInfo *info, const PosixTime &t): local time
static void ov_to_calendar(Cpu *c) {
    int64_t t;
    in_get(c, "nn::time::ToCalendarTime", &t, arg_x(c, 2), sizeof t);
    CalendarTime ct;
    CalendarInfo ci;
    if (!arg_x(c, 0) || !local_calendar(t, &ct, &ci)) {
        ret_x(c, TIME_R_INVALID);
        return;
    }
    out_put(c, "nn::time::ToCalendarTime", arg_x(c, 0), &ct, sizeof ct);
    out_put(c, "nn::time::ToCalendarTime", arg_x(c, 1), &ci, sizeof ci);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE",
             ov_to_calendar, HOS_IMPLEMENTED, "local calendar fields in the host's time zone; donor ov_system.c:704");

// CalendarTime ToCalendarTimeInUtc(const PosixTime &t): 8 bytes in x0
static void ov_to_calendar_utc(Cpu *c) {
    int64_t t;
    in_get(c, "nn::time::ToCalendarTimeInUtc", &t, arg_x(c, 0), sizeof t);
    CalendarTime ct;
    utc_calendar(t, &ct, NULL);
    uint64_t v;
    memcpy(&v, &ct, sizeof v);
    ret_x(c, v);
}
HOS_OVERRIDE("_ZN2nn4time19ToCalendarTimeInUtcERKNS0_9PosixTimeE", ov_to_calendar_utc, HOS_IMPLEMENTED,
             "UTC calendar fields, the 8-byte struct in x0; donor ov_system.c:718");

// PosixTime ToPosixTimeFromUtc(const CalendarTime &c): months out of 1..12 carry into the year, like timegm
static void ov_to_posix_utc(Cpu *c) {
    CalendarTime ct;
    in_get(c, "nn::time::ToPosixTimeFromUtc", &ct, arg_x(c, 0), sizeof ct);
    int64_t mon = ct.month - 1, y = ct.year + floor_div(mon, 12);
    mon -= floor_div(mon, 12) * 12;
    int64_t t = (days_from_civil(y, (unsigned)mon + 1, 1) + ct.day - 1) * 86400 + ct.hour * 3600 + ct.minute * 60 +
                ct.second;
    ret_x(c, (uint64_t)t);
}
HOS_OVERRIDE("_ZN2nn4time18ToPosixTimeFromUtcERKNS0_12CalendarTimeE", ov_to_posix_utc, HOS_IMPLEMENTED,
             "POSIX time of a UTC calendar time; donor ov_system.c:728");

// ClockSnapshot::ClockSnapshot()
static void ov_snapshot_init(Cpu *c) {
    ClockSnapshot s;
    memset(&s, 0, sizeof s);
    out_put(c, "nn::time::ClockSnapshot::ClockSnapshot", arg_x(c, 0), &s, sizeof s);
}
HOS_OVERRIDE("_ZN2nn4time13ClockSnapshotC1Ev", ov_snapshot_init, HOS_IMPLEMENTED,
             "an all-zero snapshot; donor ov_system.c:739");

// static Result ClockSnapshot::CreateWithStandardSystemClock(ClockSnapshot *out)
static void ov_snapshot_create(Cpu *c) {
    if (!arg_x(c, 0)) {
        ret_x(c, TIME_R_INVALID);
        return;
    }
    SteadyClockTimePoint steady = steady_now();
    int64_t now = posix_now();
    ClockSnapshot n;
    memset(&n, 0, sizeof n);
    n.user.offset = now - steady.value;
    n.user.point = steady;
    n.network = n.user;
    n.user_time = n.network_time = now;
    if (!local_calendar(now, &n.user_calendar, &n.user_info)) rt_fatal("nn::time: the host clock is out of range");
    n.network_calendar = n.user_calendar;
    n.network_info = n.user_info;
    n.steady = steady;
    location_name(now, n.location);
    n.automatic_correction = 1;
    out_put(c, "nn::time::ClockSnapshot::CreateWithStandardSystemClock", arg_x(c, 0), &n, sizeof n);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn4time13ClockSnapshot29CreateWithStandardSystemClockEPS1_", ov_snapshot_create, HOS_IMPLEMENTED,
             "user and network clocks at the host wall time, steady and zone from the host; donor ov_system.c:745");

// Result CalculateSpanBetween(TimeSpan *out, const ClockSnapshot &a, const ClockSnapshot &b): nanoseconds
static void ov_span_between(Cpu *c) {
    ClockSnapshot a, b;
    in_get(c, "nn::time::CalculateSpanBetween", &a, arg_x(c, 1), sizeof a);
    in_get(c, "nn::time::CalculateSpanBetween", &b, arg_x(c, 2), sizeof b);
    if (!arg_x(c, 0)) {
        ret_x(c, TIME_R_INVALID);
        return;
    }
    if (memcmp(a.steady.source, b.steady.source, 16)) {
        ret_x(c, TIME_R_SOURCE_MISMATCH);
        return;
    }
    int64_t span = (b.steady.value - a.steady.value) * 1000000000ll;
    out_put(c, "nn::time::CalculateSpanBetween", arg_x(c, 0), &span, sizeof span);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn4time20CalculateSpanBetweenEPNS_8TimeSpanERKNS0_13ClockSnapshotES5_", ov_span_between,
             HOS_IMPLEMENTED, "ns of steady time between snapshots, a result for mixed sources; donor ov_system.c:765");

// TimeSpan CalculateStandardUserSystemClockDifferenceByUser(const ClockSnapshot &a, const ClockSnapshot &b): the
// change of the user clock offset between the snapshots (manual adjustments) in nanoseconds
static void ov_user_difference(Cpu *c) {
    ClockSnapshot a, b;
    in_get(c, "nn::time::CalculateStandardUserSystemClockDifferenceByUser", &a, arg_x(c, 0), sizeof a);
    in_get(c, "nn::time::CalculateStandardUserSystemClockDifferenceByUser", &b, arg_x(c, 1), sizeof b);
    if (memcmp(a.user.point.source, b.user.point.source, 16)) {
        ret_x(c, 0);
        return;
    }
    ret_x(c, (uint64_t)((b.user.offset - a.user.offset) * 1000000000ll));
}
HOS_OVERRIDE("_ZN2nn4time48CalculateStandardUserSystemClockDifferenceByUserERKNS0_13ClockSnapshotES3_",
             ov_user_difference, HOS_IMPLEMENTED,
             "the user clock offset change in ns, 0 across sources; donor ov_system.c:776");

// The network clock availability task: no request is sent. The game's task object is only reached through the members
// below, so its layout never matters: construction and destruction write nothing, Initialize hands the task an event
// of the host's, StartTask signals it at once, and GetResult is the failure. Each task has its own slot, kept from
// Initialize to the destructor; the event of slot i is event_at(i) and keeps the clear mode of its first use.
// See the services study.
typedef struct {
    uint64_t task;   // the game's task object, 0 when the slot is free
    int bound, autoclear;
} NetClockSlot;
static NetClockSlot netclock[NETCLOCK_SLOTS];

// The slot of task, NULL when it has none. Holding sys_lock.
static NetClockSlot *netclock_find(uint64_t task) {
    for (int i = 0; i < NETCLOCK_SLOTS; i++)
        if (netclock[i].task == task) return &netclock[i];
    return NULL;
}

HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTaskC1Ev", ov_nothing, HOS_OFFLINE,
             "writes nothing: no session is opened offline");
HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask6CancelEv", ov_zero, HOS_OFFLINE,
             "nothing to cancel: the task finishes at once");

// Result Initialize(os::EventClearMode): ManualClear is 0 and AutoClear 1 in the public SDK
static void ov_netclock_initialize(Cpu *c) {
    uint64_t task = arg_x(c, 0);
    int autoclear = (int32_t)arg_x(c, 1) != 0;
    if (!task) rt_guest_stop(c, "nn::time::AsyncEnsureNetworkClockAvailabilityTask::Initialize: null task");
    int mode = autoclear;
    plat_mutex_lock(&sys_lock);
    NetClockSlot *s = netclock_find(task);
    for (int i = 0; !s && i < NETCLOCK_SLOTS; i++)
        if (!netclock[i].task && (!netclock[i].bound || netclock[i].autoclear == autoclear)) s = &netclock[i];
    if (s) {
        s->task = task;
        if (!s->bound) s->autoclear = autoclear;
        s->bound = 1;
        mode = s->autoclear;
    }
    plat_mutex_unlock(&sys_lock);
    if (!s)
        hos_gap(c, "time", "nn::time::AsyncEnsureNetworkClockAvailabilityTask: more than %d tasks live at once",
                NETCLOCK_SLOTS);
    hos_bind_system_event(c, event_at((int)(s - netclock)), mode);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask10InitializeENS_2os14EventClearModeE",
             ov_netclock_initialize, HOS_OFFLINE, "success, the task gets a host event");

// The slot of the task in x0 with its event address and clear mode, a guest stop when the task was never initialised.
static NetClockSlot *netclock_of(Cpu *c, const char *fn, uint64_t *event, int *autoclear) {
    plat_mutex_lock(&sys_lock);
    NetClockSlot *s = netclock_find(arg_x(c, 0));
    if (s) *autoclear = s->autoclear;
    plat_mutex_unlock(&sys_lock);
    if (!s || !arg_x(c, 0))
        rt_guest_stop(c, "nn::time::AsyncEnsureNetworkClockAvailabilityTask::%s: not initialised", fn);
    *event = event_at((int)(s - netclock));
    return s;
}

// void/Result StartTask(): finishes at once, the finish event is signalled and the result is the failure. x0 is 0,
// which is success whether the SDK's return type is Result or void.
static void ov_netclock_start(Cpu *c) {
    uint64_t event;
    int autoclear;
    netclock_of(c, "StartTask", &event, &autoclear);
    hos_signal(hos_bind_system_event(c, event, autoclear));
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask9StartTaskEv", ov_netclock_start, HOS_OFFLINE,
             "finishes at once: the finish event is signalled, no request is sent");

// os::SystemEvent &GetFinishEvent(): the task's event, already signalled by StartTask
static void ov_netclock_event(Cpu *c) {
    uint64_t event;
    int autoclear;
    netclock_of(c, "GetFinishEvent", &event, &autoclear);
    ret_x(c, event);
}
HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask14GetFinishEventEv", ov_netclock_event, HOS_OFFLINE,
             "the task's host event, signalled by StartTask");

// Result GetResult(): the failure
static void ov_netclock_result(Cpu *c) { ret_x(c, HOS_R_UNAVAILABLE); }
HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask9GetResultEv", ov_netclock_result, HOS_OFFLINE,
             "failure: the network clock is never available offline");

static void ov_netclock_destroy(Cpu *c) {
    plat_mutex_lock(&sys_lock);
    NetClockSlot *s = arg_x(c, 0) ? netclock_find(arg_x(c, 0)) : NULL;
    if (s) s->task = 0;
    plat_mutex_unlock(&sys_lock);
}
HOS_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTaskD1Ev", ov_netclock_destroy, HOS_OFFLINE,
             "frees the task's slot, no session to close");

// ==== nn::pl shared fonts. The console's system fonts are never shipped and never read from a dump. The runtime serves
// one open-licence TrueType or OpenType font that the player's game folder holds (fonts/shared-font.ttf, which oot2
// prepare places there, tools/prepare/font.py) or that OOT2_SHARED_FONT names, for every shared font type, as the
// console serves its standard font when a language's own font is missing. The game hands the address and size it gets
// to a font loader
// that recognises a plain sfnt (TrueType, OpenType or collection) by its first four bytes, so the file is served as it
// is. The load state is Loaded though no file may be there: the game spins until it is, so a missing file stops by name at
// the call that asks for the font, with what to place and where, and not in a wait. The SDK function returns a font's
// address and size only for a loaded font, so a font that is not there is a stop and never a null answer.
enum { FONT_TYPES = 6, FONT_MAX_SIZE = 16 << 20 };   // 0 standard .. 5 the button glyphs; the kernel region is 64 MiB
static struct { int loaded; uint64_t addr, size; } shared_font;   // sys_lock

static int32_t font_type(Cpu *c, const char *fn) {
    int32_t t = (int32_t)arg_x(c, 0);
    if (t < 0 || t >= FONT_TYPES) hos_gap(c, "pl", "nn::pl::%s: shared font type %d is not one this runtime knows", fn, t);
    return t;
}

// The first four bytes of an sfnt font: TrueType 0x00010000, 'OTTO', 'true' and a collection 'ttcf'.
static int font_is_sfnt(const uint8_t *b) {
    return !memcmp(b, "\0\1\0\0", 4) || !memcmp(b, "OTTO", 4) || !memcmp(b, "true", 4) || !memcmp(b, "ttcf", 4);
}

// The font file's path: OOT2_SHARED_FONT when set, else fonts/shared-font.ttf in the game folder.
static const char *font_path(char *buf, size_t n) {
    const char *e = getenv("OOT2_SHARED_FONT");
    if (e && *e) snprintf(buf, n, "%s", e);
    else snprintf(buf, n, "%s/fonts/shared-font.ttf", hos_game_folder());
    return buf;
}

// Read the font into guest memory once; a named stop when it is missing, too large or not an sfnt. Holding sys_lock.
static void font_load(Cpu *c, const char *fn, int32_t t) {
    if (shared_font.loaded) return;
    char path[1100], at[96];
    font_path(path, sizeof path);
    FILE *f = fopen(path, "rb");
    if (!f) {
        const char *named = getenv("OOT2_SHARED_FONT");
        if (named && *named)   // prepare places nothing at a path the variable names: the advice is another one
            hos_gap(c, "pl", "nn::pl::%s(%d) from %s: no shared font file at %s, the path OOT2_SHARED_FONT names. Name "
                    "a TrueType or OpenType font that is there, or unset the variable for fonts/shared-font.ttf in the "
                    "game folder, where oot2 prepare places an open-licence font",
                    fn, t, hos_caller(c, at, (int)sizeof at), path);
        hos_gap(c, "pl", "nn::pl::%s(%d) from %s: no shared font file at %s. The console's fonts are not part of this "
                "program: the font step of oot2 prepare places an open-licence font there, or a font of your own with "
                "--shared-font FILE. Run it again, or name a TrueType or OpenType font with OOT2_SHARED_FONT",
                fn, t, hos_caller(c, at, (int)sizeof at), path);
    }
    uint8_t *data = NULL;
    size_t len = 0;
    if (!fseek(f, 0, SEEK_END)) {
        long end = ftell(f);
        if (end > 0 && end <= FONT_MAX_SIZE && !fseek(f, 0, SEEK_SET)) {
            data = malloc((size_t)end);
            if (data && fread(data, 1, (size_t)end, f) == (size_t)end) len = (size_t)end;
        }
    }
    fclose(f);
    if (!len) {
        free(data);
        hos_gap(c, "pl", "nn::pl::%s(%d) from %s: the shared font file %s is empty, unreadable or larger than %d MiB", fn,
                t, hos_caller(c, at, (int)sizeof at), path, FONT_MAX_SIZE >> 20);
    }
    if (len < 12 || !font_is_sfnt(data)) {
        free(data);
        hos_gap(c, "pl", "nn::pl::%s(%d) from %s: the shared font file %s is not a TrueType or OpenType font", fn, t,
                hos_caller(c, at, (int)sizeof at), path);
    }
    uint64_t addr = mem_kernel_alloc(len);
    if (!hos_out(addr, data, len)) rt_fatal("nn::pl::%s: the kernel region block for the shared font is not writable", fn);
    free(data);
    shared_font.addr = addr, shared_font.size = len, shared_font.loaded = 1;
    rt_log("pl: shared font served to every font type: %zu bytes from %s", len, path);
}

// SharedFontLoadState GetSharedFontLoadState(SharedFontType)
static void ov_font_state(Cpu *c) {
    font_type(c, "GetSharedFontLoadState");
    ret_x(c, 1);
}
HOS_OVERRIDE("_ZN2nn2pl22GetSharedFontLoadStateENS0_14SharedFontTypeE", ov_font_state, HOS_ABSENT,
             "Loaded though a font file may not be placed: a missing font stops by name at Address or Size, not in a "
             "spin; donor ov_system.c:828");

// const void *GetSharedFontAddress(SharedFontType) and size_t GetSharedFontSize(SharedFontType)
static void ov_font_address(Cpu *c) {
    int32_t t = font_type(c, "GetSharedFontAddress");
    plat_mutex_lock(&sys_lock);
    font_load(c, "GetSharedFontAddress", t);
    uint64_t addr = shared_font.addr;
    plat_mutex_unlock(&sys_lock);
    ret_x(c, addr);
}
static void ov_font_size(Cpu *c) {
    int32_t t = font_type(c, "GetSharedFontSize");
    plat_mutex_lock(&sys_lock);
    font_load(c, "GetSharedFontSize", t);
    uint64_t size = shared_font.size;
    plat_mutex_unlock(&sys_lock);
    ret_x(c, size);
}
HOS_OVERRIDE("_ZN2nn2pl20GetSharedFontAddressENS0_14SharedFontTypeE", ov_font_address, HOS_IMPLEMENTED,
             "the open-licence font file placed in the game folder (or OOT2_SHARED_FONT), in guest memory, the same for "
             "every type; a named stop when there is none");
HOS_OVERRIDE("_ZN2nn2pl17GetSharedFontSizeENS0_14SharedFontTypeE", ov_font_size, HOS_IMPLEMENTED,
             "the size of that file; a named stop when there is none");

// RequestLoadNxFont(void *buffer, size_t size): the console's service loads its Switch 1 fonts into the buffer the game
// offers (the SDK builds a transfer memory over it). The fonts the game then reads come from GetSharedFontAddress and
// GetSharedFontSize, which this runtime serves from its own memory, so the request needs no copy: it checks that the
// buffer is guest memory, as the transfer memory creation would, and returns. Nothing else of it is observable: the SDK's
// function ignores the size it is given and returns no value.
static void ov_font_request(Cpu *c) {
    uint64_t buf = arg_x(c, 0), size = arg_x(c, 1);
    uint8_t probe;
    if (!size || size > RT_ADDRESS_LIMIT - buf || !hos_in(&probe, buf, 1) || !hos_in(&probe, buf + size - 1, 1))
        rt_guest_stop(c, "nn::pl::RequestLoadNxFont: the buffer 0x%llx of %llu bytes is not guest memory",
                      (unsigned long long)buf, (unsigned long long)size);
}
HOS_OVERRIDE("_ZN2nn2pl17RequestLoadNxFontEPvm", ov_font_request, HOS_IMPLEMENTED,
             "buffer checked, nothing copied: the fonts are served from the runtime's own memory by Address and Size");
