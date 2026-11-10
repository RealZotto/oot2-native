// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The M1 capture (architecture D26, section 12 and R6): what M1's exit test reads, written while the game runs, and
// judged afterwards by tools/check_capture.py. Present cadence, frame times, each guest thread's CPU and core time per
// frame, memory once a second, and the counters of missing work.
//
// Opt-in, never a default: OOT2_CAPTURE=1, read once by capture_start, apart from census mode (OOT2_CENSUS) and from
// census.tsv, which every run writes. Two more variables, read at the same time:
//   OOT2_CAPTURE_TITLE=N    the title mark is present N, counted from the first present (default 1800). The runtime
//                           does not know when the game shows its title, so the mark is a present count, the unit
//                           D26's pad script is keyed to. The default comes from Lichii's runs of the title pass capture, where the
//                           menu appeared over the title movie 1,459 and 1,531 presents after the first, plus about
//                           five seconds of margin; nothing has measured it on this runtime yet.
//   OOT2_CAPTURE_SECONDS=S  the run ends by itself at the first present S seconds or more after the mark (decimal
//                           seconds), with exit code 0 and the closing record "length"; 0 or unset: no end of its own.
//
// The file is capture-<UTC date>-<time>-<pid>.tsv in the game folder: one file per run, numbers only (no name of the
// game's functions or objects), and it stays in the player's game folder like every other output of a run.
//
// Format, one record per line, tab separated, the kind first. Times are nanoseconds on the host's monotonic clock since
// capture_start; ticks are the virtual 60 Hz clock's (oot2/vsync.h); handles and entry addresses are hexadecimal with
// 0x, every other number decimal, -1 for a value the platform cannot read.
//   oot2-capture 1                     the first line: the format and its version
//   header  key  value                 at the start, and later from capture_note (a value may hold spaces)
//   present seq t due shown requested held frame_ns late
//                                      one per present, in seq order: the host time of the call, the due tick, the
//                                      shown tick (0: unknown when the line is written), the interval the game asked
//                                      for and the E2 hold (oot2/nvn_pace.h), the host time since the previous present,
//                                      and the ticks lost against the previous present's due tick plus
//                                      max(requested, held)
//   frame   seq submitter submits submitters presenter [handle cpu_ns core_ns]...
//                                      the frame that ends with present seq: the guest thread that made the most
//                                      QueueSubmitCommands calls in it (0: none), its calls, how many threads submitted,
//                                      the thread that presented, then every guest thread whose CPU or core time in the
//                                      frame is not zero (rt_kernel_thread_times; a thread missing from the line had
//                                      none). The submitter's core time is the render thread's busy time.
//   thread  handle entry priority      the first time a thread shows in a frame line
//   gap     handle entry frames sum_ns max_ns
//                                      one per thread, just before the closing record: over the frames the thread was
//                                      listed in (frames), the time it held a virtual core without CPU time to run on,
//                                      the core time less the CPU time of each frame (0 for a frame with more CPU than
//                                      core), summed and its largest: a thread that waits holding its core shows
//                                      here, and one that runs or gives its core up is near 0. The frame lines hold the
//                                      same two numbers a frame at a time.
//   shown   seq tick                   a shown tick the back end reported after the present line was written
//   title   seq t skipped shader pipeline raw faults
//                                      the title mark, with the counters below as they stood at it
//   memory  seq t working_set private available pages_output
//                                      once a second, on a host thread: the process's working set and private commit,
//                                      the machine's available physical memory and the raw count of pages written to
//                                      the page file (every process of the machine; plat_pages_output_name), and seq,
//                                      the newest present then
//   count   seq t skipped shader pipeline raw faults
//                                      once a second: skipped draws, shader misses, pipeline misses, raw-access misses
//                                      and device faults so far
//   cost    seq t presents sum_ns max_ns sampler_ns
//                                      once a second: the CPU time the capture took on the presenting thread over the
//                                      presents so far (sum and the largest one) and on its own sampling thread
//   gpu     seq pass ns                a GPU timestamp per pass, from capture_gpu_pass (the null back end has none)
//   end     code reason seq t fatals skipped shader pipeline raw faults working_set private available pages_output
//           presents sum_ns max_ns sampler_ns
//                                      the closing record, always the last line: the exit code and its reason (exit,
//                                      length, fatal, unimplemented, guest-stop, census-limit, other), the newest
//                                      present, the named fatals (1 for exit codes 70 to 72) and the final values.
//                                      A file without it was cut short and fails the check.
// Readers ignore header keys and record kinds they do not know, so later tickets add to the header (the settings in
// force) and to the records without a new version.
// The header key test_movie_blit is "0", or "1 (...)" when the run had the test switch of that name on (settings.h): a
// switch that replaces the movies' enhancement chain by a resample and so changes their pictures. The proof judge fails a
// capture whose value is not 0, and the Vulkan replay counts every dispatch it did not run or replaced into the skipped
// draws below, so a capture of such a run fails on its own counter too.
//
// The writer. capture_present, on the presenting guest thread, marks the frame boundary (rt_kernel_frame: the capture
// is its only caller while it runs), reads every thread's time for the frame that just ended, and appends its lines to
// a buffer in memory; the sampling thread writes the buffer to the file once a second, so no disk write lands on a
// guest thread. The closing record is written by capture_end at the normal end and by the stop path for every other
// end (ExitProcess, exits 70 to 73), through rt_add_stop_writer; a run killed from outside leaves the file without one.
#pragma once
#include <stdint.h>

// The counters of missing work (D25 level 1). The null back end leaves them at 0; the Vulkan back end adds to them.
enum {
    CAPTURE_SKIPPED_DRAWS,
    CAPTURE_SHADER_MISSES,
    CAPTURE_PIPELINE_MISSES,
    CAPTURE_RAW_MISSES,
    CAPTURE_DEVICE_FAULTS,
    CAPTURE_COUNTERS
};

// Read the switches; when OOT2_CAPTURE=1, open the file in folder, write the header and start the sampling thread. Once,
// on the start-up thread, after the back end is chosen and before the game's first present.
void capture_start(const char *folder);
int capture_on(void);   // 1 once capture_start opened a file, until the closing record

// For a back end or a settings store: a header item (driver version, display mode, implicit Vulkan layers; the settings
// in force). Before capture_start it is kept and written with the header; after, it follows as a header line.
void capture_note(const char *key, const char *value);
void capture_count(int counter, uint64_t n);                  // any thread; counted whether or not a capture runs
void capture_gpu_pass(uint64_t seq, int pass, uint64_t ns);   // a GPU timestamp of a pass of present seq's frame

// From the front end: nvnQueuePresentTexture after present seq was made, nvnQueueSubmitCommands after a submission.
void capture_present(uint64_t seq);
void capture_submit(void);

// The closing record (exit code and its reason), once; nothing is written after it. The normal end of the run calls it;
// the stop path calls it for every other end.
void capture_end(int code);
