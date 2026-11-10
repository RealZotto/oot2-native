// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Input by tick: a file that holds the controller's buttons and sticks for stretches of the virtual 60 Hz clock
// (oot2/vsync.h), so an unattended run has input and a run someone played can be played again, and a recorder that
// writes what the person held in the same format. Both are off by default; with neither option set nothing here runs
// on the pad path.
//
// The options (environment variables, like the other options that name a file):
//   OOT2_INPUT_FILE=<file>     replay <file>. The replayed input REPLACES the real input: while the file is on, the pad
//                              and the keyboard are not read for the game (an unattended run has none, and a touch on
//                              a real pad must not change a run that is meant to be repeated). The menu's swallow of
//                              the input changes nothing either: the file is the source.
//   OOT2_INPUT_RECORD=<file>   record the input the game was given, in the same format, run-length by tick. Alone it
//                              records the real input (the merged pad and keyboard sample the window publishes); with a
//                              replay it records the replayed input, which writes the same file again in canonical form.
//                              The file is written as the run goes and its last run is closed when the run stops, so a
//                              process killed from outside loses that last run.
//   The two may not name the same file (the replay would be overwritten at start). Both are read once, at start-up
//   (hos_init), and a file that cannot be read or has a malformed line stops the run there, by name.
//
// The tick. Tick 0 is the virtual clock's tick in progress when the game first reads the full-key controller's state
// (nn::hid::GetNpadStates), so a file means the same thing in every run however long start-up took. The state of a tick
// is one value: the first read in a tick takes it and the other reads in that tick give the same, as many reads as the
// game makes (replaying or recording only; a run with both options off reads the live sample every time, as before).
// A tick in which the game does not read gives nothing to anyone, so a recorded run extends over such ticks.
//
// The file. Plain text, one line per held stretch, in any order:
//   # a comment: from '#' to the end of the line; blank lines and CR LF line ends are fine
//   at <tick> for <count> [<held> ...]
//     <tick>   decimal, 0 to 4294967295: the first tick of the stretch, counted as above
//     <count>  decimal, 1 to 4294967295: how many ticks it lasts (the ticks tick .. tick+count-1)
//     <held>   a button name, held for the whole stretch, or a stick value `lx=N`, `ly=N`, `rx=N`, `ry=N` with N from
//              -32768 to 32767 (up is positive on both y axes), held for the whole stretch. Names in either case, each
//              at most once on a line: a b x y stickl stickr l r zl zr plus minus left up right down, and the stick
//              directions stickl_left stickl_up stickl_right stickl_down stickr_left stickr_up stickr_right
//              stickr_down (the nn::hid button bits of oot2/hid_input.h, in bit order).
// A line with nothing held is a stretch of rest. Where stretches overlap the buttons are all held, and for a stick axis
// the later line in the file wins; a tick that no line covers holds nothing (all buttons up, sticks centered). Example:
//   at 0 for 120                    # rest for two seconds
//   at 120 for 6 a                  # press A
//   at 200 for 90 lx=32767 zr       # run right with ZR down
//
// A line the reader cannot take stops the run with `FATAL: input file <file>, line <n>: <what is wrong>`.
//
// Later markers. A stretch starts at a tick today. A marker that starts it when a named state value of the game is met,
// instead of at a tick, plugs in at three places of runtime/hos/hid_replay.c, each marked "extension point": the line
// kinds the reader dispatches on (`at` is the one kind), the start of a stretch (a tick), and the check that a stretch is
// live. Nothing of that is built, and no game value name or address belongs in this layer: a marker's name is bound
// elsewhere.
//
// The run report (census.tsv, oot2/boot.h) carries a row `input` with the file's name, its SHA-256 and what the run took
// from it, and a row `input-record` for the recorder.
#pragma once
#include <stdint.h>

#include "oot2/hid_input.h"

// Read OOT2_INPUT_FILE and OOT2_INPUT_RECORD and open what they name; a fatal on a file that cannot be used. Called once
// by hos_init; a second call changes nothing.
void hid_replay_start(void);

// One read of the full-key controller's state at the current tick: when a file is replayed its stretches replace the
// buttons and sticks of *in, when the recorder is on the state is written, and in both cases the tick's first read
// is latched and given to every read of that tick. *in's sampling number is left alone. No-op with both options off.
void hid_replay_filter(HidHostInput *in);

// The run stops: close the recorder's last run and its file. Safe to call again, and with the recorder off.
void hid_replay_finish(void);

// The report's rows: kind "input" and "input-record" for what is on, in the order of census.tsv's columns.
typedef void (*HidReplayRowFn)(void *ctx, const char *kind, const char *key, const char *status, uint64_t calls,
                               const char *why);
void hid_replay_rows(HidReplayRowFn row, void *ctx);
