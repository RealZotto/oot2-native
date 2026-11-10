// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Save data that a stopped run cannot corrupt (ov_fs.c, savetxn.c). The console's save mount keeps writes in a journal
// until Commit and shows the old files if the program ends first; a host folder written in place does not. Here a file of
// a save mount that the guest changes is changed in a SHADOW next to it (<name>.oot2wip) holding a copy of the current
// content; the guest's reads and writes go to the shadow while it is pending, and Commit publishes every shadow of the
// mount: it makes each one durable, writes a COMMIT MARKER naming them all, renames each over its file, and removes the
// marker. A run that ends at any point leaves a state that st_recover (the first mount of the folder in the next run)
// settles: a complete marker finishes the commit forward (every file new), no marker removes the shadows (every file
// old). The folder layout stays <root>/{account.sav, option.sav, slot_NN/...}; between commits it may also hold
// <name>.oot2wip files and, during a commit, .oot2commit (all of them hidden from the guest's listings).
//
// A path here is a host path under a mount's root ("<root>/rel"); the unit does no guest-side work. Nothing in it holds a
// lock across a host call.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define ST_SUFFIX ".oot2wip"           // a shadow is its file's name plus this
#define ST_MARKER ".oot2commit"        // in the mount's root: the files a commit is publishing
#define ST_MARKER_TMP ".oot2commit.tmp"

// Finish or undo a commit an earlier run left, then remove the shadows nobody owns under root. Once per root per
// process (a later call for the same root does nothing, so it never touches this run's own shadows). Returns the number of
// files it finished forward (a counted outcome for the log), or -1 when the folder could not be settled.
int st_recover(const char *root);

// The serial of real's pending shadow, or 0 when it has none. A serial never repeats within a process.
uint64_t st_serial(const char *real);
// The shadow's path (real + ST_SUFFIX); 1 when it fits in cap.
int st_shadow_path(const char *real, char *out, size_t cap);

// Start the shadow of an existing file (a copy of its content), or find the one already pending: HF_OK with its serial, or
// an HF_ error (HF_NOT_FOUND when real is not a file).
int st_begin(const char *real, uint64_t *serial);
// Start the shadow of a file that does not exist yet, size bytes long: HF_EXISTS when real or its shadow is there.
int st_begin_create(const char *real, uint64_t size, uint64_t *serial);

// Publish every pending shadow under root as one commit. HF_OK, or the first HF_ error (the commit is then finished by
// the next call or the next run's recovery).
int st_commit(const char *root);
// Publish the pending shadows at or under prefix, one rename each with no marker (a file about to move); HF_OK or an error.
int st_publish_prefix(const char *prefix);
// Drop the pending shadows at or under prefix: their files keep their old content (an unmount, a delete).
void st_discard_prefix(const char *prefix);

// For the tests: called at each step of a commit and of a recovery with the step's name and its index within the step; a
// test records the order or ends the process there. NULL in the game.
extern void (*st_trace)(const char *step, int index);
