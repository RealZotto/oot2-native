// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The ledger (architecture D21): the per-player file of what the game used, written by oot2.exe and read by the next
// prepare, never shipped. One row per (stage key, answers, attachment formats): the stage is a program stage's job key
// (nvn_stage.key, the programs and state study), the answers are the translator's (xlate/host/Job.cs: tex, img, attr, fout), and the
// attachment formats are what a linked pipeline needs (D12).
//
// Where the state comes from. A draw's record carries no state, and the state a draw uses was bound by earlier records,
// possibly by another recording that called this one. The front end inlines a called recording into its caller when
// it records (nvn_obj.h), so a submitted handle is one flat stream, and LedgerTrack walks that stream: it follows the
// bound stages, the textures and images of each stage, the vertex attributes and the render targets across records and
// across handles, as one GPU channel does, and at each draw or dispatch it makes the rows of the stages the draw runs.
// A back end calls ledger_track_op for each record of its replay, in order, so the null back end and the Vulkan replay
// share the one tracker; the Vulkan replay reads its pipeline state from the same LedgerTrack and keeps none of its own.
//
// The per-stage mapping to the translator's slots (xlate/host/ResourceInfo.cs HandleSlots):
//   BindTexture(stage, i)  -> "tex" answer i of that stage's module (c[2] byte offset 0x20 + 8 i, minus 0x20, over 8)
//   BindImage(stage, i)    -> "img" answer i of that stage's module (c[2] 0x120 + 8 i; the combined range's slot 32 + i
//                             is answered from "img" only), with a type and a format
//   vertex attribute k     -> "attr" answer k of the vertex module (the state array's index)
//   colour target k        -> "fout" answer k of the pixel module
// The type is the descriptor's: the texture pool entry the handle's texture ID names, as the game registered it (the
// view's target and format win over the texture's), read when the bind is replayed.
//
// A row holds what was bound to its stage, not what the stage reads: the writer does not parse code, so a texture bound
// earlier and never sampled by this stage is in the row. The reader (the next prepare) projects the answers on the
// slots the stage's description uses and recomputes the digest, so rows that differ only in unread slots give one job.
// Texture index 32 to 39 is the image table's range and is never a "tex" answer.
//
// Writing. Rows are kept in memory, deduplicated, while the frame runs; ledger_flush writes the file at a safe point
// and refuses on the frame path (inside a submission).
#pragma once
#include <stdint.h>

#include "nvn_obj.h"

struct nvn_stage;

// ---- the texture pool's descriptors, as the game registered them (nvnTexturePoolRegisterTexture calls this)
void ledger_desc_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v);

// ---- the tracker
#define LEDGER_STAGES 8        // NVN stage numbers: 0 vertex, 1 fragment, 5 compute, 6 task, 7 mesh (6 and 7 assumed from the control section stage byte, unconfirmed)
#define LEDGER_TEX_SLOTS 41    // BindTexture indexes 0 to 40 (c[2] 0x20 up to the separate textures at 0x168)
#define LEDGER_IMG_SLOTS 8
#define LEDGER_ATTRS 32
#define LEDGER_TARGETS 8
typedef struct LedgerTrack {
    uint64_t tex_pool;                                   // the texture pool's table address (SetTexturePool)
    const struct nvn_stage *prog[LEDGER_STAGES];         // the bound stage of each NVN stage, NULL for none
    uint8_t tex[LEDGER_STAGES][LEDGER_TEX_SLOTS];        // per stage and BindTexture index: a type (ledger.c), 0 none
    uint8_t img_type[LEDGER_STAGES][LEDGER_IMG_SLOTS];
    uint16_t img_fmt[LEDGER_STAGES][LEDGER_IMG_SLOTS];   // an NVNformat
    uint8_t attr[LEDGER_ATTRS];                          // 0 none, 1 float, 2 sint, 3 uint
    uint8_t fout[LEDGER_TARGETS];
    uint16_t color_fmt[LEDGER_TARGETS];                  // the render targets' NVNformat
    int32_t depth_fmt;                                   // -1: none
    uint8_t ncolor, samples;
    uint8_t answers_digest[5][32];
    uint8_t answers_digest_valid;
} LedgerTrack;
void ledger_track_reset(LedgerTrack *t);
// Follow one record of a replay; at a draw or dispatch, make the rows of the stages it runs.
void ledger_track_op(LedgerTrack *t, const op_hdr *r);
// The SHA-256 of the current stage's complete canonical answers object, or zero when it has none. Matches the digest
// written into ledger.jsonl, before prepare projects answers onto the slots the module reads. Cached until that
// stage's relevant tracker state changes.
void ledger_track_answers_digest(LedgerTrack *t, int kind, uint8_t out[32]);

// ---- the rows
// The folder the file lives in (the player's output folder): rows already there are loaded, so a new run adds to them.
// Without a folder rows are kept in memory only and a flush does nothing.
void ledger_set_folder(const char *dir);
#define LEDGER_FILE "ledger.jsonl"
// Write the file when rows were added since the last write: 1 written, 0 nothing to write or no folder, -1 refused (on
// the frame path) or the file could not be written.
int ledger_flush(void);
// The front end brackets a submission with these: a flush inside is refused.
void ledger_frame_enter(void);
void ledger_frame_leave(void);
// Flush when the run stops (rt_set_stop_flush): a guest's ExitProcess, a fatal and a guest stop end the process
// without returning to main, so the end of main is not the only safe point. Not refused on the frame path, since a
// replay that stops on a draw is inside a submission and the process ends right after (a thread that holds the rows
// at that moment is not waited for, and the file is not written).
void ledger_flush_on_stop(void);
uint64_t ledger_rows(void);                  // rows held, loaded ones included
const char *ledger_row_text(uint64_t i);     // row i in the order it was added, NULL past the end (one JSON object)
uint64_t ledger_writes(void);                // files written
void ledger_reset(void);                     // forget every row and the folder (tests)
