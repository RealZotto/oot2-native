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
//   The two may not name the same file (the replay would be overwritten at start): two spellings of one file (./, upper
//   case, a link) are caught by content, since an existing record file with the replay's exact bytes is refused too. Both
//   are read once, at start-up (hos_init), and a file that cannot be read or has a malformed line stops the run there,
//   by name.
//
// The tick. Tick 0 is the virtual clock's tick in progress when the game first reads the full-key controller's state
// (nn::hid::GetNpadStates), so a file means the same thing in every run however long start-up took. The state of a tick
// is one value: the first read in a tick takes it and the other reads in that tick give the same, as many reads as the
// game makes (replaying or recording only; a run with both options off reads the live sample every time, as before).
// A tick in which the game does not read gives nothing to anyone, so a recorded run extends over such ticks.
//
// The file. Plain text, one line per held stretch, in any order:
//   # a comment: from '#' to the end of the line; blank lines, CR LF line ends and a UTF-8 byte order mark at the start
//     of the file are fine
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
// Markers. A line can wait for a named state value of the game instead of a tick:
//   when <role> <op> <value> [after <ticks>] for <count> [<held> ...]
//     <role>   a role of the store description (oot2/store.h; `oot2 prepare --names FILE` writes oot2-store.txt from a
//              names file of the player's own, outside the repository): scene, next_scene, entrance, room, age, health,
//              max_health, boss, paused, loading, menu, or `<kind>.<label>` of the player's choosing with a kind of bool,
//              int, uint, float or string. Either case. No name of the game's is in this program or in this file.
//     <op>     == != < <= > >=; a string takes only == and !=
//     <value>  by the role's kind: a bool is 0, 1, true or false; an int is a whole number from -2147483648 to 2147483647,
//              a uint one from 0 to 4294967295; a float a decimal number (sign, digits, point, exponent), rounded to a
//              single float as the store holds one; a string one word of at most 63 characters, compared exactly with
//              the store's string (cut at 63 characters)
//     after    optional: whole ticks, 0 to 4294967295, from the tick the value was met to the first tick of the stretch
//     for ...  the count and the held items are those of an `at` line
// The value is looked at once per tick, at the tick's first read, from the first read of the file: the first tick at which
// the comparison holds fires the line, once (a value that is true at the first read fires at tick 0, and one that turns
// true again later does not fire it again). The stretch is then the ticks fired+delay .. fired+delay+count-1, and it
// combines with the other lines as any stretch does. A store the game has not built yet, or a table that is not built yet,
// is waited for. Examples:
//   when scene == <word> after 30 for 6 a     # press A half a second after the scene changes to the named one
//   when health <= 3 for 90 zl                # hold ZL for 1.5 seconds once health is down to 3
//
// A `when` line is read at start-up and checked against the description then: a role the description lacks, an operator or
// value that does not fit the role's kind, a description that is missing, malformed or made from another build of the main
// module (its build id is compared with the loaded module's), all stop the run with the same fatal as any bad line, naming
// the line. While it runs, a value that no table of its kind has, once every such table is built, for 120 ticks in a row, and
// an address the description leads to that is outside the guest's memory, stop it by name too. The recorder is unchanged: it
// writes the ticks the lines took effect at, so the recording of a run with markers replays tick for tick without them. A file
// with no `when` line never reads the description.
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
