// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The miss path and the release of pipelines (oot2/vk_miss.h): the drain, the build on workers, the bound and its
// override, the counters and the log, and the reference counts by stage key that decide when a program's pipelines go.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/capture.h"
#include "oot2/gpu.h"
#include "oot2/nvn_program.h"
#include "oot2/runtime.h"
#include "oot2/vk_miss.h"
#include "oot2/vk_rec.h"
#include "plat.h"

#define KEY_MAX 320
#define STAGE_KEY_MAX 26
#define MAX_STAGES 6
#define BUCKETS 1024
#define JOB_STACK (8ull << 20)
// Lines of misses and of skips: the first of each are all printed, then one in LOG_EVERY, so that a run that misses on
// every frame does not drown its log (the counters are exact).
#define LOG_FIRST 48
#define LOG_EVERY 64

int vkm_no_drain, vkm_no_count_skip, vkm_release_at_once, vkm_no_late;

// ---------- the device's ops
static uint64_t dev_drain(void) {
    vkrec_idle();   // every packet put so far made its call (nothing when the recorder is not open)
    uint64_t issued = gpu_sync_issued(), done = gpu_sync_done();
    uint64_t n = issued > done ? issued - done : 0;
    gpu_sync_wait(issued, UINT64_MAX);
    return n;
}

// The miss path builds with the gate held: a non-NULL late marks the build as the miss path's (vk_pipeline.h).
static VkPipeline dev_build_compute(const VkpStage *cs) {
    static const int miss;
    return vkp_compute_ex(cs, &miss);
}

static const VkmOps g_device = {
    .find_graphics = vkp_find_graphics, .find_compute = vkp_find_compute, .build_graphics = vkp_graphics_ex,
    .build_compute = dev_build_compute, .graphics_key = vkp_graphics_key, .drain = dev_drain, .release = vkp_release,
    .handed = gpu_sync_handed, .done = gpu_sync_done, .now_ns = plat_now_ns};

// ---------- state
typedef struct Job {
    struct Job *next;
    char key[KEY_MAX];
    VkpStage vs, task, ps;  // the stage records are immortal (the programs and state study), so keys and shapes stay valid
    VkpFormats formats;
    const VkmOps *ops;
    int late;               // set (release) by the replay once it stopped waiting for this build
    int done;               // under M.lock
    VkPipeline result;
    PlatThread *thread;
} Job;

typedef struct KeyRef {
    struct KeyRef *next;
    char key[STAGE_KEY_MAX];
    int count;              // live programs holding the key
    int pending;            // no program holds it: its objects go once the timeline reaches retire
    uint64_t retire;
} KeyRef;

typedef struct Prog {
    struct Prog *next;
    uint64_t serial;
    int n;
    char key[MAX_STAGES][STAGE_KEY_MAX];
} Prog;

static struct {
    PlatMutex lock;
    PlatCond cond;
    const VkmOps *ops;
    VkmConfig cfg;
    int cfg_known;
    VkmStats st;
    Job *jobs;
    KeyRef *keys[BUCKETS];
    Prog *progs[BUCKETS];
    int npending;           // keys pending release (read without the lock for the fast exit of vkm_retire)
    uint64_t logged[3];     // lines printed: misses, skips, would-skips
} M = {.lock = PLAT_MUTEX_INIT, .cond = PLAT_COND_INIT};

static const VkmOps *ops(void) { return M.ops ? M.ops : &g_device; }

// A test's ops stand in for the ones it fills; the others stay the device's (the drain and the timeline's values work
// without a device: they read the timeline and the recorder alone).
void vkm_set_ops(const VkmOps *o) {
    static VkmOps merged;
    if (!o) {
        M.ops = NULL;
        return;
    }
    merged = *o;
#define FILL(f) merged.f = merged.f ? merged.f : g_device.f
    FILL(find_graphics), FILL(find_compute), FILL(build_graphics), FILL(build_compute), FILL(graphics_key), FILL(drain),
        FILL(release), FILL(handed), FILL(done), FILL(now_ns);
#undef FILL
    M.ops = &merged;
}

void vkm_set_config(const VkmConfig *c) {
    plat_mutex_lock(&M.lock);
    if (c) M.cfg = *c, M.cfg_known = 1;
    else M.cfg_known = 0;
    plat_mutex_unlock(&M.lock);
}

static VkmConfig config(void) {
    plat_mutex_lock(&M.lock);
    if (!M.cfg_known) {
        M.cfg = (VkmConfig){VKM_DEFAULT_BOUND_NS, 0};
        const char *v = getenv("OOT2_VK_MISS_BOUND");
        if (v && *v) {
            if (!strcmp(v, "unbounded")) {
                M.cfg.unbounded = 1;
            } else {
                char *end;
                unsigned long long ms = strtoull(v, &end, 10);
                if (end == v || *end || ms == 0 || ms > 600000)
                    rt_fatal("gfx: vulkan: OOT2_VK_MISS_BOUND is \"%s\"; the values are unbounded, or a number of "
                             "milliseconds from 1 to 600000 (unset: 2000)", v);
                M.cfg.bound_ns = ms * 1000000ull;
            }
        }
        M.cfg_known = 1;
    }
    VkmConfig c = M.cfg;
    plat_mutex_unlock(&M.lock);
    return c;
}

void vkm_stats(VkmStats *out) {
    plat_mutex_lock(&M.lock);
    *out = M.st;
    out->keys_pending = (uint64_t)M.npending;
    plat_mutex_unlock(&M.lock);
}

void vkm_stats_reset(void) {
    plat_mutex_lock(&M.lock);
    memset(&M.st, 0, sizeof M.st);
    memset(M.logged, 0, sizeof M.logged);
    plat_mutex_unlock(&M.lock);
}

static int log_now(int which) {
    uint64_t n = M.logged[which]++;
    return n < LOG_FIRST || n % LOG_EVERY == 0;
}

static double ms(uint64_t ns) { return (double)ns / 1e6; }

// ---------- jobs
// Jobs whose build is done leave the list here: the thread's handle goes, a build that finished after the replay stopped
// waiting for it is counted.
static void reap_jobs(void) {
    Job *gone = NULL;
    plat_mutex_lock(&M.lock);
    for (Job **at = &M.jobs; *at;) {
        Job *j = *at;
        if (j->done) {
            *at = j->next;
            j->next = gone, gone = j;
            if (__atomic_load_n(&j->late, __ATOMIC_ACQUIRE)) M.st.late_builds++;
        } else {
            at = &j->next;
        }
    }
    plat_mutex_unlock(&M.lock);
    while (gone) {
        Job *j = gone;
        gone = j->next;
        plat_thread_free(j->thread);
        free(j);
    }
}

static void job_main(void *arg) {
    Job *j = arg;
    VkPipeline p = j->ops->build_graphics(&j->vs, j->task.key ? &j->task : NULL, &j->ps, &j->formats, &j->late);
    plat_mutex_lock(&M.lock);
    j->result = p;
    j->done = 1;   // j is not touched after the unlock
    plat_cond_broadcast(&M.cond);
    plat_mutex_unlock(&M.lock);
}

static Job *find_job(const char *key) {
    for (Job *j = M.jobs; j; j = j->next)
        if (!strcmp(j->key, key)) return j;
    return NULL;
}

static uint64_t drain(const VkmOps *o, uint64_t *waited) {
    if (vkm_no_drain) return *waited = 0;
    uint64_t t0 = o->now_ns();
    *waited = o->drain();
    uint64_t ns = o->now_ns() - t0;
    plat_mutex_lock(&M.lock);
    M.st.drains++, M.st.drain_ns += ns, M.st.drained_submissions += *waited;
    plat_mutex_unlock(&M.lock);
    vkm_retire();   // the queue is idle: what no program holds any more goes now
    return ns;
}

static void stall_note(int kind, uint64_t ns) {
    plat_mutex_lock(&M.lock);
    M.st.stall_ns[kind] += ns;
    if (ns > M.st.stall_max_ns[kind]) M.st.stall_max_ns[kind] = ns;
    plat_mutex_unlock(&M.lock);
}

static void skip_note(const VkpStage *vs, const VkpStage *ps, uint64_t waited_ns, const VkmConfig *c, int pending) {
    plat_mutex_lock(&M.lock);
    if (!vkm_no_count_skip) M.st.skipped_draws++;
    if (pending) M.st.skipped_pending++;
    int say = log_now(1);
    uint64_t total = M.st.skipped_draws;
    plat_mutex_unlock(&M.lock);
    if (!vkm_no_count_skip) capture_count(CAPTURE_SKIPPED_DRAWS, 1);
    if (!say) return;
    const char *kind = vs->shape && vs->shape->kind == NVN_STAGE_MS ? "mesh" : "vertex";
    if (pending)
        rt_log("gfx: vulkan: draw skipped: the graphics pipeline of %s stage %s and pixel stage %s is still being built "
               "(skipped draws so far: %llu)", kind, vs->key, ps->key,
               (unsigned long long)total);
    else
        rt_log("gfx: vulkan: draw skipped: the graphics pipeline of %s stage %s and pixel stage %s was not ready after "
               "%.0f ms (bound %.0f ms); its build goes on (skipped draws so far: %llu)", kind, vs->key, ps->key,
               ms(waited_ns), ms(c->bound_ns), (unsigned long long)total);
}

// ---------- graphics
VkPipeline vkm_graphics(const VkpStage *vs, const VkpStage *task, const VkpStage *ps, const VkpFormats *formats,
                        int may_skip) {
    const VkmOps *o = ops();
    VkPipeline p = o->find_graphics(vs, task, ps, formats);
    if (p) return p;
    reap_jobs();
    VkmConfig cfg = config();
    char key[KEY_MAX];
    o->graphics_key(vs, task, ps, formats, key, sizeof key);
    uint64_t t0 = o->now_ns();

    // A build of this very pipeline that an earlier draw gave up waiting for: skip at once, or wait for it.
    plat_mutex_lock(&M.lock);
    Job *running = find_job(key);
    if (running && !running->done && may_skip && !cfg.unbounded) {
        plat_mutex_unlock(&M.lock);
        skip_note(vs, ps, 0, &cfg, 1);
        return VK_NULL_HANDLE;
    }
    if (running) {
        while (!running->done) plat_cond_wait(&M.cond, &M.lock);
        VkPipeline r = running->result;
        M.st.waits_kept += !may_skip;
        plat_mutex_unlock(&M.lock);
        stall_note(0, o->now_ns() - t0);
        return r;
    }
    M.st.graphics_misses++;
    plat_mutex_unlock(&M.lock);
    capture_count(CAPTURE_PIPELINE_MISSES, 1);

    uint64_t waited;
    uint64_t drain_ns = drain(o, &waited);
    Job *j = calloc(1, sizeof *j);
    if (!j) rt_fatal("gfx: vulkan: out of memory for a pipeline build");
    snprintf(j->key, sizeof j->key, "%s", key);
    j->vs = *vs, j->ps = *ps, j->formats = *formats, j->ops = o;
    if (task) j->task = *task;
    uint64_t build0 = o->now_ns();
    plat_mutex_lock(&M.lock);
    j->next = M.jobs, M.jobs = j;
    plat_mutex_unlock(&M.lock);
    j->thread = plat_thread_start(job_main, j, JOB_STACK);
    if (!j->thread) job_main(j);   // no thread to build on: the replay builds and waits, as for any wait without a bound

    int bounded = !cfg.unbounded && may_skip;
    uint64_t deadline = build0 + cfg.bound_ns;
    plat_mutex_lock(&M.lock);
    while (!j->done) {
        if (!bounded) {
            plat_cond_wait(&M.cond, &M.lock);
            continue;
        }
        uint64_t now = o->now_ns();
        if (now >= deadline) break;
        plat_cond_wait_ns(&M.cond, &M.lock, deadline - now);
    }
    int done = j->done;
    p = j->result;
    plat_mutex_unlock(&M.lock);

    uint64_t built = o->now_ns() - build0, stall = o->now_ns() - t0;
    if (!done) {
        // From here the build creates its objects beside work in flight: the replay goes on and submits the frame.
        if (!vkm_no_late) __atomic_store_n(&j->late, 1, __ATOMIC_RELEASE);
        stall_note(0, stall);
        skip_note(vs, ps, built, &cfg, 0);
        return VK_NULL_HANDLE;
    }
    int over = !bounded && built > cfg.bound_ns;   // a bounded wait that just made it is not a wait past the bound
    plat_mutex_lock(&M.lock);
    if (over) M.st.would_skip += cfg.unbounded, M.st.waits_kept += !cfg.unbounded;
    int say = log_now(0) || over;
    plat_mutex_unlock(&M.lock);
    stall_note(0, stall);
    if (say)
        rt_log("gfx: vulkan: pipeline miss (graphics): %s stage %s, pixel stage %s: %llu submission(s) drained in "
               "%.1f ms, built in %.1f ms, the replay waited %.1f ms%s",
               vs->shape && vs->shape->kind == NVN_STAGE_MS ? "mesh" : "vertex", vs->key, ps->key,
               (unsigned long long)waited, ms(drain_ns), ms(built), ms(stall),
               !over ? "" : cfg.unbounded ? " (past the bound: without the override this draw would have been skipped)"
                                           : " (past the bound, but the draw's target holds no earlier picture: waited)");
    return p;
}

// ---------- compute
VkPipeline vkm_compute(const VkpStage *cs) {
    const VkmOps *o = ops();
    VkPipeline p = o->find_compute(cs);
    if (p) return p;
    reap_jobs();
    plat_mutex_lock(&M.lock);
    M.st.compute_misses++;
    plat_mutex_unlock(&M.lock);
    capture_count(CAPTURE_PIPELINE_MISSES, 1);
    uint64_t t0 = o->now_ns(), waited;
    uint64_t drain_ns = drain(o, &waited);
    uint64_t build0 = o->now_ns();
    p = o->build_compute(cs);   // a dispatch is never skipped: the replay thread waits for the build
    uint64_t built = o->now_ns() - build0, stall = o->now_ns() - t0;
    stall_note(1, stall);
    plat_mutex_lock(&M.lock);
    int say = log_now(0);
    plat_mutex_unlock(&M.lock);
    if (say)
        rt_log("gfx: vulkan: pipeline miss (compute): stage %s: %llu submission(s) drained in %.1f ms, built in %.1f ms, "
               "the replay waited %.1f ms", cs->key, (unsigned long long)waited, ms(drain_ns), ms(built), ms(stall));
    return p;
}

// ---------- release
static uint32_t hash_text(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u;
    return h % BUCKETS;
}

static KeyRef *key_find(const char *key, int add) {
    uint32_t h = hash_text(key);
    for (KeyRef *k = M.keys[h]; k; k = k->next)
        if (!strcmp(k->key, key)) return k;
    if (!add) return NULL;
    KeyRef *k = calloc(1, sizeof *k);
    if (!k) rt_fatal("gfx: vulkan: out of memory for the reference count of stage %s", key);
    snprintf(k->key, sizeof k->key, "%s", key);
    k->next = M.keys[h], M.keys[h] = k;
    return k;
}

static Prog **prog_slot(uint64_t serial) {
    Prog **at = &M.progs[(uint32_t)((serial * 0x9E3779B97F4A7C15ull) >> 40) % BUCKETS];
    while (*at && (*at)->serial != serial) at = &(*at)->next;
    return at;
}

// Serials of finalized programs. A recording made before the game finalized (or re-set) a program can be submitted
// after, and its binds name a serial that is gone: each such bind holds the keys for its frame only and drops them at
// once (the newest value handed out is the frame's), so a finalized program never holds a key for ever. An entry stays
// for the run (24 bytes a finalized program).
typedef struct Gone {
    struct Gone *next;
    uint64_t serial;
} Gone;
static Gone *g_gone[BUCKETS];

static Gone **gone_slot(uint64_t serial) {
    Gone **at = &g_gone[(uint32_t)((serial * 0x9E3779B97F4A7C15ull) >> 40) % BUCKETS];
    while (*at && (*at)->serial != serial) at = &(*at)->next;
    return at;
}

// The program's references go (M.lock held): a key left with none is given the newest value handed out, the open
// frame's included, since its commands, still being recorded, may name the pipeline.
static void drop_refs_locked(const Prog *p) {
    for (int i = 0; i < p->n; i++) {
        KeyRef *k = key_find(p->key[i], 0);
        if (!k || --k->count > 0) continue;
        k->pending = 1, k->retire = ops()->handed();
        M.npending++;
    }
    M.st.programs_released++;
}

void vkm_program_bound(uint64_t serial, const struct nvn_stage *const *stages, int n) {
    if (!serial) return;
    plat_mutex_lock(&M.lock);
    Prog **at = prog_slot(serial);
    if (*at) {
        plat_mutex_unlock(&M.lock);
        return;
    }
    Prog *p = calloc(1, sizeof *p);
    if (!p) rt_fatal("gfx: vulkan: out of memory for program %llu", (unsigned long long)serial);
    p->serial = serial;
    for (int i = 0; i < n && i < MAX_STAGES; i++) {
        if (!stages[i]) continue;
        snprintf(p->key[p->n], STAGE_KEY_MAX, "%s", stages[i]->key);
        KeyRef *k = key_find(p->key[p->n], 1);
        k->count++;
        if (k->pending) k->pending = 0, M.npending--;   // held again before its objects went: they stay
        p->n++;
    }
    M.st.programs_bound++;
    Gone **g = gone_slot(serial);
    if (*g) {   // already finalized: the frame being replayed holds the keys, and no later frame can
        drop_refs_locked(p);
        free(p);
    } else {
        *at = p;
    }
    plat_mutex_unlock(&M.lock);
}

void vkm_program_release(uint64_t serial) {
    plat_mutex_lock(&M.lock);
    Prog **at = prog_slot(serial);
    Prog *p = *at;
    Gone **slot = gone_slot(serial);
    if (!*slot) {   // remembered: a bind of this serial from now on holds the keys for its frame only
        Gone *g = calloc(1, sizeof *g);
        if (!g) rt_fatal("gfx: vulkan: out of memory for program %llu", (unsigned long long)serial);
        g->serial = serial;
        g->next = *slot, *slot = g;
    }
    if (p) {
        *at = p->next;
        drop_refs_locked(p);
    }
    plat_mutex_unlock(&M.lock);
    free(p);
    if (p && vkm_release_at_once) vkm_retire();
}

void vkm_retire(void) {
    if (!__atomic_load_n(&M.npending, __ATOMIC_ACQUIRE)) return;
    const VkmOps *o = ops();
    uint64_t done = vkm_release_at_once ? UINT64_MAX : o->done();
    plat_mutex_lock(&M.lock);
    // A build still running holds handles of libraries it made or found (it links them when its last library is done),
    // and a release would destroy those under it: nothing goes while a job runs.
    for (const Job *j = M.jobs; j; j = j->next)
        if (!j->done) {
            plat_mutex_unlock(&M.lock);
            return;
        }
    // Destroyed with the lock held: a program bound meanwhile takes the key's count under the same lock, so it either
    // finds the objects whole or finds none and builds them again.
    for (uint32_t b = 0; b < BUCKETS; b++) {
        for (KeyRef **at = &M.keys[b]; *at;) {
            KeyRef *k = *at;
            if (k->count == 0 && k->pending && done >= k->retire) {
                o->release(k->key);
                *at = k->next;
                M.npending--;
                M.st.keys_released++;
                free(k);
            } else {
                at = &k->next;
            }
        }
    }
    plat_mutex_unlock(&M.lock);
}

void vkm_close(void) {
    plat_mutex_lock(&M.lock);
    for (Job *j = M.jobs; j; j = j->next)
        while (!j->done) plat_cond_wait(&M.cond, &M.lock);
    plat_mutex_unlock(&M.lock);
    reap_jobs();
    vkm_retire();
    VkmStats s;
    vkm_stats(&s);
    if (s.graphics_misses || s.compute_misses || s.skipped_draws)
        rt_log("gfx: vulkan: pipeline misses: %llu graphics, %llu compute; the replay waited %.1f ms (graphics, worst %.1f "
               "ms) and %.1f ms (compute, worst %.1f ms); %llu drains took %.1f ms; skipped draws %llu (%llu of them at "
               "once), waits past the bound that the override waited out %llu, kept %llu, builds that finished late %llu",
               (unsigned long long)s.graphics_misses, (unsigned long long)s.compute_misses, ms(s.stall_ns[0]),
               ms(s.stall_max_ns[0]), ms(s.stall_ns[1]), ms(s.stall_max_ns[1]), (unsigned long long)s.drains,
               ms(s.drain_ns), (unsigned long long)s.skipped_draws, (unsigned long long)s.skipped_pending,
               (unsigned long long)s.would_skip, (unsigned long long)s.waits_kept, (unsigned long long)s.late_builds);
    plat_mutex_lock(&M.lock);
    for (uint32_t b = 0; b < BUCKETS; b++) {
        while (M.keys[b]) {
            KeyRef *k = M.keys[b];
            M.keys[b] = k->next;
            free(k);
        }
        while (M.progs[b]) {
            Prog *p = M.progs[b];
            M.progs[b] = p->next;
            free(p);
        }
        while (g_gone[b]) {
            Gone *g = g_gone[b];
            g_gone[b] = g->next;
            free(g);
        }
    }
    M.npending = 0;
    plat_mutex_unlock(&M.lock);
}
