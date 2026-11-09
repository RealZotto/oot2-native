// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The seam between the NVN2 front end (runtime/nvn) and a graphics back end: the null back end of runtime/nvn, which
// lets the front end run and be tested without a GPU, and the Vulkan back end of architecture section 6. The front end
// calls the operations below on guest threads, with the guest's arguments already checked. A back end never runs
// guest code, never calls back into the front end's entry points, and keeps its own lock where it needs one.
//
// Objects and records cross the seam in the layouts of nvn_obj.h: an object pointer is the game's own memory, which
// the game may copy by value or reuse after finalizing it, so a back end keys what it keeps on the serials there, never
// on the pointer. A command handle points at a recording in host memory (nvn_obj.h, command records), which the back
// end reads during submit and keeps no pointer into afterwards.
//
// Synchronisation (architecture section 5). Every queue's submissions share one order: the fence values submit hands
// out. A back end completes them in that order and writes a submission's CPU-visible results (event signals, counter
// reports, buffer copies and clears, uniform updates, in op-stream order) before its fence value is done, so a fence
// on one queue also covers what every queue submitted before it. Report timestamps are the guest's system tick (31.25
// MHz, rt_ticks) read from the host clock when the report completes; a counter the back end cannot measure reports 0.
// Conditional rendering is decided when the back end replays the condition, from the reports before it: a report not
// complete then counts as 0 when it is a timestamp and as visible when it is samples passed.
//
// Presents and pacing. The front end's window rotates through the window's textures and hands each present to
// present() below, which returns at once; the swap interval stays in the window, where the front end answers it. The
// front end also holds the acquire on the virtual 60 Hz clock (D22; runtime/nvn/nvn_window.c, oot2/nvn_pace.h): by
// default AcquireTexture returns at once and its sync is signalled at the gate tick, behind a setting it blocks until
// the gate tick; the gate, the due tick of every present and the per-present records are the front end's, and no back
// end sleeps a guest thread or ticks the game's vsync events (runtime/core). What crosses this seam is two
// optional members. present_pace tells the back end, right before the present() call it belongs to, the present's
// number and the tick it is due at: the tick in progress when the game presented (or, when it presented before its wait
// on the acquire sync, the gate tick of that acquire, which can be a tick still to come), on the clock of oot2/vsync.h
// (rt_vsync_due_ns turns it into host time), so the back end may show the frame at the first refresh it can from then
// on and not before. With a fixed frame rate (the fixed frame rate gate) the due instant is a slot of that rate's schedule and rarely a
// tick: due_tick is then the tick it falls in, and due_count is the instant itself in system-counter counts from the
// clock's origin (rt_vsync_counter_ns turns it into host time); due_count is the one a back end holds a frame to, and it
// is the due count of due_tick when no rate is set. present_shown reports the
// tick a present was actually shown at, from the back end's present timing; it never blocks and answers 0 for a
// present not shown yet or whose time the back end cannot tell (the null back end always answers 0 at once). The
// swapchain, the present mode and the monitor's rate are the Vulkan back end's and never feed the game's timing.
//
// Adapted from TotK Recomp runtime/gfx.h (MIT, Copyright (c) 2026 Valiantsin Dzerakh; licenses/totk-MIT.txt), as DKB
// Recomp at 9d8fa02 and oot-pc-recomp at 8e09aaa carry it (GPL-3.0-or-later): the back-end table and the front end's
// registration call. Changed here: only the back-end operations are taken (the
// window, the shader service and the statistics come with the tickets that need them); registration is stated per pool
// ID (architecture D10: the bindless tables are updated one entry at a time, never rebuilt); present carries no pacing
// time (D22: the front end's gate, with the due tick as an optional member beside it); submit takes no entry tick; a
// back end is installed by gfx_set_backend.
#pragma once
#include <stdio.h>
#include <stdint.h>

#include "nvn_obj.h"

typedef struct GfxBackend {
    const char *name;
    // Memory pools, named by their GPU range [mem, mem + size). pool_init returns 0 to fail MemoryPoolInitialize.
    // pool_fini returns once no submission in flight can still use the pool: the front end gives its range to the
    // next pool at once. pool_map_virtual applies n mapping requests (nvn_mapping_t), which the front end has
    // checked, to the virtual pool v; 0 is a fatal, since the front end's own mapping state has already moved.
    int (*pool_init)(pool_t *p);
    void (*pool_fini)(pool_t *p);
    int (*pool_map_virtual)(pool_t *v, int n, const void *requests);
    // One entry of a texture or sampler pool, at the pool's ID: the back end updates that entry and nothing else.
    void (*texture_register)(const dpool_ref *pool, int id, const tex_t *t, const view_t *v);
    void (*sampler_register)(const dpool_ref *pool, int id, const samp_par *s);
    // A texture or program the game finalized: what the back end keeps for its serial may go once no submission that
    // can use it is in flight.
    void (*texture_release)(const tex_t *t);
    void (*program_release)(const prog_t *p);
    // Run n recordings in order and return this submission's fence value. Fence values grow by one per submission,
    // across every queue, and a value is done only when every submission up to it is done (section 5).
    uint64_t (*submit)(const uint64_t *handles, int n);
    // 1 once fence value `value` is done, 0 when timeout_ns passed first (UINT64_MAX: no limit). A value no submit has
    // handed out yet answers 0 at once. The caller has given its virtual core up for the wait (D7).
    int (*fence_wait)(uint64_t value, uint64_t timeout_ns);
    uint64_t (*fence_done)(void);   // highest fence value whose side effects are all written
    // Queue texture t for display, cropped to crop (x, y, width, height); returns at once. It is shown after the work
    // submitted before the call.
    void (*present)(const tex_t *t, const int crop[4]);
    void (*finish)(void);           // every submission so far is done when it returns
    // Texel copies between a texture view and host memory (CopyTextureToBuffer and the CPU texel calls): row is the
    // bytes per row, img the bytes per two-dimensional image of the region.
    void (*read_texels)(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                        uint64_t img);
    void (*write_texels)(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row,
                         uint64_t img);
    void (*guest_written)(uint64_t addr, uint64_t size);   // the CPU wrote pool storage: copies of it are stale
    int (*guest_read)(uint64_t addr, uint64_t size);       // 1 when in-flight GPU work could write the range
    void (*cpu_synced)(uint64_t fence);                    // the CPU waited for fence and will read what it wrote
    // Optional (NULL: not wanted); see Presents and pacing above. seq counts presents from 1, across windows.
    // Two windows presenting from two threads may reach these calls out of seq order: seq is the key, not arrival.
    void (*present_pace)(uint64_t seq, uint64_t due_tick, uint64_t due_count);
    uint64_t (*present_shown)(uint64_t seq);
} GfxBackend;

extern const GfxBackend gfx_null_backend;
uint64_t gfx_null_presents(void);   // presents the null back end has counted
// The null back end's op log: each record of each recording it walks at submit, in order, to fn (NULL: none), with
// the submission's fence value and the handle's index in it. Called inside the submission.
typedef void (*GfxNullOpLog)(uint64_t fence, int index, const op_hdr *record, void *ctx);
void gfx_null_log_ops(GfxNullOpLog fn, void *ctx);
uint64_t gfx_null_records(void);    // records the null back end has walked
// Inside the op log: 1 when the draws, dispatches and clears of the record shown would render (RenderEnable and the
// condition of conditional rendering, as the replay decided them).
int gfx_null_render_enabled(void);
// Completion held, for tests and for a run that wants work in flight: while on, a submission is replayed at submit and
// completed later, by gfx_null_complete or when the hold ends, always in submission order. gfx_null_complete completes
// the held submissions up to fence value `value` (0: all of them) and returns the newest fence value done.
void gfx_null_hold(int on);
uint64_t gfx_null_complete(uint64_t value);
// WaitEvents the null back end checked at completion, and how many of them what came before them in submission order
// had not satisfied (*unsatisfied, when not NULL).
uint64_t gfx_null_wait_events(uint64_t *unsatisfied);
// The same waits sorted by what satisfied them (gfx_null.c): by a GPU signal earlier in submission order, by a CPU signal
// made before, by neither (the word held the value already), not satisfied at completion, and of those later satisfied
// by a GPU signal or a CPU signal (the hazard for a back end that replays at submit). unsatisfied = later_gpu + later_cpu +
// pending (not satisfied yet when asked) + dropped (found the list of unsatisfied waits, which holds 64, full).
typedef struct { uint64_t by_gpu, by_cpu, initial, unsatisfied, later_gpu, later_cpu, dropped, pending; } GfxNullWaitClasses;
void gfx_null_wait_classes(GfxNullWaitClasses *out);
// The back end the front end uses: the null back end until gfx_set_backend installs another, which it does before
// rt_dispatch_init (fatal after it, or for a table with an operation missing).
const GfxBackend *gfx_backend(void);
void gfx_set_backend(const GfxBackend *b);

// ---- the front end's entry-point table (runtime/nvn/nvn_procs.c, architecture D10)
// Give every NVN2 entry point of the list its synthetic host address. Start-up calls it after the modules are loaded
// and before rt_dispatch_init, which seals the host-function registry; nvnBootstrapLoader is a fatal without it.
void nvn_procs_register(void);
// The entry point named name, as nvnDeviceGetProcAddress answers: its synthetic address, or 0 for a name not on the
// list (logged once per name, as the console's driver returns NULL).
uint64_t nvn_proc_address(const char *name);
int nvn_proc_count(void);              // names on the list
const char *nvn_proc_name(int i);      // in list order (sorted); NULL past the end
int nvn_proc_has_body(int i);          // 1 when an NVN_API body implements it, 0 when it stops with its name
// Per-present counters (nvn_census.c): nvnQueuePresentTexture calls nvn_census_present after present seq. It logs the first
// present and every 60th; in census mode (runtime.h) it also snapshots the host-function calls, rewrites the census every
// 600 presents and ends the run at OOT2_CENSUS_FRAMES. nvn_census_write writes census mode's calls-per-frame table.
void nvn_census_present(uint64_t seq);
// Recording counters (nvn_cmd.c), for the census: begins the front end took up from the game's inline code, begins through
// the entry point, EndRecording answers with a handle and with 0, memory callbacks for control and for command memory, and
// recordings begun while records of an earlier one were left out.
typedef struct {
    uint64_t begin_inline, begin_slow, end_handle, end_empty, callback_control, callback_command, left_out;
} NvnCmdCounts;
void nvn_cmd_counts(NvnCmdCounts *out);

// Memory pool counters (nvn_pool.c), the shape of the pools a run makes (D14): by class (CPU-visible, CPU_NO_ACCESS, physical,
// virtual) and over all, the pools made, the pools live now and the most live at once, with their bytes, and the bytes of
// the GPU address window in use now and at most. A pool counts from the MemoryPoolInitialize that succeeded to its Finalize.
enum { NVN_POOL_CPU, NVN_POOL_CPU_NO_ACCESS, NVN_POOL_PHYSICAL, NVN_POOL_VIRTUAL, NVN_POOL_CLASSES };
typedef struct { uint64_t created, live, peak, bytes, peak_bytes; } NvnPoolClass;
typedef struct {
    NvnPoolClass cls[NVN_POOL_CLASSES], all;
    uint64_t gpu_no_access;   // pools made with GPU_NO_ACCESS (a class of their own, counted among the four above)
    uint64_t window_bytes, window_peak;
} NvnPoolStats;
void nvn_pool_stats(NvnPoolStats *out);

void nvn_census_write(FILE *f);
// The front end's own counters as census rows: emit(ctx, key, count, why) once for each of recordings, callbacks,
// waitevents, waitclasses, pools, pools-cpu, pools-cpu-no-access, pools-physical, pools-virtual and presents (nvn_census.c).
typedef void (*NvnStatRow)(void *ctx, const char *key, uint64_t count, const char *why);
void nvn_census_stats(NvnStatRow emit, void *ctx);

// ---- command recordings (runtime/nvn/nvn_cmd.c, architecture D11)
// QueueSubmitCommands' work on the front end's side: every handle must be a live recording (a named stop otherwise),
// the deferred binds and pushes take what the game's arrays hold now, the copies stay alive while the back end's
// submit reads them, and that submit's fence value is returned. c is the calling guest thread.
struct Cpu;
uint64_t nvn_submit_recordings(struct Cpu *c, const uint64_t *handles, int n);
uint64_t nvn_cmd_host_bytes(void);   // host memory command buffers, recording copies and their index hold now
uint64_t nvn_cmd_copies(void);       // recording copies alive now

// ---- GPU addresses and the pool index (runtime/nvn/nvn_pool.c, architecture D10, D14)
// A pool the CPU can see takes its guest address as its GPU address (guest pointers are host pointers). A pool the CPU
// cannot see gets a range of this window, above the guest's 39-bit address space: 64 KiB aligned, as many ranges as
// the window holds, a freed range given to the next pool that fits. Window addresses are GPU addresses only, never
// mapped on the host; a back end gives them backing (D14).
#define GFX_GPU_WINDOW_BASE (1ull << 39)
#define GFX_GPU_WINDOW_SIZE (64ull << 30)
#define GFX_GPU_PAGE 0x10000ull   // the window's granularity, and the page of virtual pool mappings
static inline int gfx_gpu_only(uint64_t addr) { return addr - GFX_GPU_WINDOW_BASE < GFX_GPU_WINDOW_SIZE; }
// The initialised pool the GPU can reach whose range holds [addr, addr + len) (len 0 counts as 1), or NULL: any
// thread, under a shared lock. The pointer is the game's object, valid while the game keeps the pool.
const pool_t *nvn_pool_find(uint64_t addr, uint64_t len);
// The device's NVNdepthMode and NVNwindowOriginMode: NEAR_IS_MINUS_W and LOWER_LEFT (0, as NVN's defaults are) until
// the game sets them (runtime/nvn/nvn_device.c).
int nvn_depth_mode(void);
int nvn_window_origin_mode(void);
