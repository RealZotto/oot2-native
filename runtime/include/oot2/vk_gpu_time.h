// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// GPU time of the replayed frames, command by command (runtime/gpu/vk_gpu_time.c). A diagnostic, off by default: the
// replay's own counters say what the CPU spent, and a frame the GPU cannot finish in time shows only as a wait for the
// frame two before (oot2/vk_rec.h, slot_wait_ns), so this measures where the GPU's time goes.
//
//   OOT2_VK_GPU_TIMING        unset or empty: off. A file name: every replayed frame is timed, and one row per frame and
//                             per marked command is written to that file by the completion thread (flushed a frame at a
//                             time, so a run that is killed keeps what it had)
//   OOT2_VK_GPU_TIMING_MARKS  on (the default): a timestamp after every dispatch, render pass, transfer and explicit
//                             barrier; off: the frame's spans only (the prologue, the main command buffer and the
//                             epilogue), which leave the GPU's overlap of one command with the next as it is
//
// A timestamp is written at the end of all commands before it (VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT), so the time of a
// marked command is the time from the previous mark to its own: its work, the barrier before it and whatever of the
// commands before it was still running. Where a frame has an unmarked stretch (the draws of a pass are one mark) the
// pass takes it. A mark drains the GPU as a barrier does, which can only take away overlap the frame had; the frame total
// with marks against the total without them (the two settings of the second variable) bounds what the marks cost.
//
// The file is tab separated. A line starting with # is a comment. Then, in the order the frames finish:
//   F value start_ns total_ns prologue_ns main_ns epilogue_ns marks dropped gap_ns effects_ns
//       one frame: the device clock of its first timestamp relative to the first frame's, its whole span, its three
//       command buffers' spans, the marks it holds and those past the pool's room, the time since the previous frame
//       ended (the other submissions between them, and idle), and the time the completion thread spent in the frame's
//       effects (the write-backs and the reports: the CPU's part of the wait for its value)
//   B value imports import_bytes mirrors mirror_bytes uploads upload_bytes writebacks writeback_bytes raw_writebacks
//     raw_writeback_bytes promoted promote_bytes raw_begins resolves resolve_misses raw_ended_ns raw_mark_ns
//     raw_cpu_pages sink_bytes sinks_made pagegen_scan_ns pagegen_scans
//       the backing map's and the raw path's counters (cumulative; imports and mirrors as they stand) when this frame ended, in
//       the first frame and every 60th after it; the last seven are the raw write-back's: the time its effect took on the
//       completion thread and the part of it in pagegen_mark_gpu_direct, the pages of it the CPU had written since the
//       submission began (plain stamp), the bytes of host imports under mirrors (the write-back's sinks) as they stand and the
//       sinks made, then the write watch queries of all the page generations' consumers (their time and their number)
//   M value index what label x y z ns notes
//       one marked command: what (dispatch, pass, xfer, barrier), its label (a dispatch's stage key, a pass's first
//       vertex and pixel stage keys, a transfer's operation), the dispatch's group counts (0 for an indirect one, a
//       pass's draw count in x), its time, and the buffers it bound: a letter for the backing (E VRAM extent, I host
//       import, M mirror, U a uniform's copy in scratch memory, G a storage image, B a driver bank in scratch
//       memory), its bytes in KiB, w for a range the stage writes and @ and its guest address in hex where it has
//       one, R when the stage walks the page table
#pragma once
#include <stdint.h>

enum { VKGT_DISPATCH, VKGT_PASS, VKGT_XFER, VKGT_BARRIER };

extern int vkgt_enabled;   // 1 once vkgt_open found the setting (the replay tests it before every call below)

// gpu_vk_open and gpu_vk_close: read the settings and open the file; free the pools. Nothing to do when off.
void vkgt_open(void);
void vkgt_close(void);

// The replaying thread, inside a frame (vkrec_begin to vkrec_end): the frame's first timestamp, in its prologue; and its
// last ones, one at the end of the main command buffer and one at the end of the epilogue, with the effect that reads
// them once the device reached the frame.
void vkgt_frame_begin(uint64_t value);
void vkgt_frame_end(void);

// A buffer the next mark's command uses (see the file format: kind is the letter, addr its guest address or 0, bytes its
// size), collected until a mark takes them. A mark: the command done, its label and its three numbers.
void vkgt_note(char kind, uint64_t addr, uint64_t bytes, int write);
void vkgt_mark(int what, const char *label, uint32_t x, uint32_t y, uint32_t z);
