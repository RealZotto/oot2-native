// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The movie chain's stand-in (runtime/gpu/vk_movie_blit.c, settings.h test_movie_blit). A TEST switch, off by default
// and never set in a proof run, that changes the picture of every movie: the movie library enhances each decoded frame
// with a chain of compute dispatches, the first of which turns the decoded planes into a picture and the last of which
// writes the enlarged picture the game draws. With the switch on, the replay runs the first dispatch, does not run the
// ones between, and resamples the first one's picture into the last one's output (vk_replay.c), so the movie plays at
// the game's pace and the chain's cold pipeline builds never happen. The picture is not the console's.
//
// Which dispatches are the chain's is told by a file of program keys that stays outside the repository (no key of the
// game's is written in the source, the tests or the documents), named by the variable OOT2_MOVIE_BLIT_KEYS. Lines, in
// any order:
//   # a comment (also after a line)    blank lines are ignored
//   keep <key>    the chain's first program: it runs as it always does, and the picture it writes is the one resampled
//   skip <key>    a program between the first and the last: not run while the chain is open
//   blit <key>    the chain's last program: replaced by the resample, into the image it writes
// A key is the stage key as the replay and the GPU timing name it (16 hex digits, a dot, 8 hex digits). At least one of
// each of keep and blit, up to 64 keys, a key in one role only. The GPU timing file of a run (OOT2_VK_GPU_TIMING) names
// the chain's dispatches in order, which is where a person gets the keys.
//
// A chain is open from a keep program's dispatch to a blit program's dispatch of the same submission: a skip program
// outside it runs as usual, and so does a blit program with no chain open. A chain that skipped dispatches and is still
// open at the end of the submission, or meets a second keep program first, is a named stop (the picture would be stale);
// a keep program alone, with nothing skipped, is not. A mistake in the setting or the file is a
// named fatal at start-up (vkmb_open) or at the first dispatch; nothing falls back to the default quietly.
#pragma once

typedef enum VkmbRole { VKMB_NONE, VKMB_KEEP, VKMB_SKIP, VKMB_BLIT } VkmbRole;

// At the device's open: with the setting on, read the file now (a named fatal on a mistake) and log that this run is a
// test run. With it off, nothing.
void vkmb_open(void);

// The role of the compute program `key` for the run: VKMB_NONE when the setting is off or the key is in no line of the
// file. The first call with the setting on reads the file when vkmb_open has not.
VkmbRole vkmb_role(const char *key);
