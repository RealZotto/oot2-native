// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Internal header of the NVN2 front end, shared by the files of runtime/nvn. An entry point is a host function
// (runtime.h): its arguments are the guest's AAPCS64 registers (arg_x, arg_p, arg_s), its result goes back with ret_x,
// and the objects it receives live in the game's memory in the layouts of nvn_obj.h.
//
// Adapted from TotK Recomp runtime/nvn_api.h (MIT, Copyright (c) 2026 Valiantsin Dzerakh; licenses/totk-MIT.txt;
// adapted there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c and nvn_impl.h at b75be8cf, MIT,
// Copyright (c) 2026 jhaizhou; licenses/dkb-xlate-MIT.txt), as DKB Recomp carries it at 9d8fa02 (GPL-3.0-or-later):
// the NVN_API macro and nvn_arg_bool. Changed here: a body is registered through the HOST_OVERRIDE table and must name
// an entry point of the list, and the per-call counting and tracing prologue is gone (the host-function registry counts
// every call already, rt_hostfn_calls). NVN_POOL_MAGIC and the two buffer layouts are oot-pc-recomp's, from its
// runtime/nvn_api.h at 8e09aaa (GPL-3.0-or-later).
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/gfx.h"
#include "oot2/nvn_obj.h"
#include "oot2/runtime.h"
#include "oot2/texheap.h"

// Each name of the list has an id, its line in nvn_procs.inc.
enum {
#define NVN_PROC(n) NVN_ID_##n,
#include "nvn_procs.inc"
#undef NVN_PROC
    NVN_PROC_COUNT
};

// NVN_API(nvnX) { body }: the body of entry point nvnX. A name that is not on the list does not compile (NVN_ID_nvnX
// is undeclared), so the table can never hand out a body under a name the list does not have. The body reaches
// nvn_procs_register through the HOST_OVERRIDE linker table, from any file linked into the program.
#define NVN_API(n)                                                                                                   \
    _Static_assert(NVN_ID_##n < NVN_PROC_COUNT, #n " is on the entry-point list");                                   \
    static void nvn_body_##n(Cpu *c);                                                                                \
    HOST_OVERRIDE(#n, nvn_body_##n);                                                                                 \
    static void nvn_body_##n(Cpu *c)

static inline int nvn_arg_bool(const Cpu *c, int i) { return (int)(arg_x(c, i) & 0xff); }   // NVNboolean: low 8 bits

static inline void nvn_invalidate(Cpu *c, uint64_t addr, uint64_t size, const char *entry) {
    int in_flight = gfx_backend()->guest_read(addr, size);
    const char *trace = in_flight ? getenv("OOT2_VK_READ_TRACE") : NULL;
    if (trace && !strcmp(trace, "1"))
        rt_log("nvn: %s read %#llx+%#llx while GPU work could write it, caller %#llx",
               entry, (unsigned long long)addr, (unsigned long long)size, (unsigned long long)c->x[30]);
}

// WaitEvent per call site (gfx_null.c), declared here and not in gfx.h: the null back end's own bookkeeping. Each
// site's waits are split by the classes of GfxNullWaitClasses; the sites past the 64 the table holds share one bucket, so the
// sums over the sites and the bucket are the totals. caller is the guest return address stored in the wait's record.
typedef struct { uint64_t caller; GfxNullWaitClasses c; } NvnWaitSite;
int gfx_null_wait_sites(NvnWaitSite *out, int max, NvnWaitSite *other);   // returns the sites listed, in the order first met

// Records the null back end replayed, by op (OP_COUNT_ entries into out), since start-up: the census's second draw count.
void gfx_null_op_counts(uint64_t *out);

#define NVN_POOL_MAGIC MAGIC('P', 'O', 'O', 'L')   // pool_t.magic of an initialised pool (nvn_pool.c)

// Buffers reference their pool and own no storage (nvn_buffer.c).
typedef struct { NVNdevice *dev; pool_t *pool; int64_t off; uint64_t size; } buf_builder_t;
FITS(buf_builder_t, NVNbufferBuilder);
typedef struct { uint32_t magic; pool_t *pool; int64_t off; uint64_t size; } buf_t;
FITS(buf_t, NVNbuffer);
