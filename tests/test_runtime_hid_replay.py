# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of input by tick (runtime/hos/hid_replay.c, oot2/hid_replay.h) through the HID service.

Builds tests/runtime/hid_replay_test.c with runtime/core, runtime/nvn, runtime/hos and runtime/audio against a
synthetic layout, at -O2 with -march=x86-64-v3, and runs it once per scenario in a fresh game folder. The program
boots the Horizon layer on a virtual clock the test drives, publishes a scripted "real" input tick by tick, and reads the
full-key controller's state through nn::hid::GetNpadStates the way the guest does; it prints what each read gave. This
file writes the input files and the scripts and computes what every read must give on its own, from a model of the
format that shares no code with the runtime. Nothing is read from a dump.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set, as
continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "hid_replay_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
MODULE_SIZE = 0x8000
TIMEOUT = 60
EXIT_FATAL = 70
EXIT_KILLED = 3   # the harness's run_then_exit: _Exit(3), a process that stops without any clean-up
FIRST_TICK = 500   # the virtual tick of the script's first line (hid_replay_test.c): the file's tick 0 is not the clock's

# The nn::hid button bits in bit order (the layout of the controller's state, as the format names them)
BUTTONS = ["a", "b", "x", "y", "stickl", "stickr", "l", "r", "zl", "zr", "plus", "minus", "left", "up", "right", "down",
           "stickl_left", "stickl_up", "stickl_right", "stickl_down", "stickr_left", "stickr_up", "stickr_right",
           "stickr_down"]
AXES = ["lx", "ly", "rx", "ry"]
ONE_READ = "0 1 0 0 0 0 0\n"   # a script of one read at tick 0, real input at rest
TAPE_A = b"at 0 for 1 a\n"
REST = (0, 0, 0, 0, 0)   # a real input at rest: what an unattended run has
GARBAGE = (0xFFFFFF, 12345, -23456, 3, -4)   # a real input that every replayed read must ignore: all 24 bits, sticks


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def bits(*names: str) -> int:
    return sum(1 << BUTTONS.index(n) for n in names)


# ---- the model of the format: spans of (first tick, count, items), and what is held at a tick
Span = tuple[int, int, list[str]]


def held_at(spans: list[Span], tick: int) -> tuple[int, int, int, int, int]:
    buttons, axis = 0, {a: 0 for a in AXES}
    for at, count, items in spans:
        if not at <= tick < at + count:
            continue
        for item in items:
            if "=" in item:
                name, value = item.split("=")
                axis[name.lower()] = int(value)
            else:
                buttons |= bits(item.lower())
    return (buttons, axis["lx"], axis["ly"], axis["rx"], axis["ry"])


def render(spans: list[Span], eol: str = "\n") -> bytes:
    return "".join(f"at {at} for {count} {' '.join(items)}".rstrip() + eol for at, count, items in spans).encode()


def encode_runs(states: list[tuple[int, tuple[int, int, int, int, int]]]) -> list[str]:
    """The record of a run of reads: states is (tick, held) per read, ticks rising. A run lasts from its first read to the
    read that shows another state (to the last read for the last run); a run of rest is no line."""
    lines = []
    for i, (tick, held) in enumerate(states):
        if i and held == states[i - 1][1]:
            continue
        end = next((t for t, h in states[i + 1:] if h != held), None)
        last = states[-1][0] + 1 if end is None else end
        if any(held):
            items = [BUTTONS[b] for b in range(24) if held[0] >> b & 1]
            items += [f"{a}={v}" for a, v in zip(AXES, held[1:]) if v]
            lines.append(f"at {tick} for {last - tick} {' '.join(items)}".rstrip())
    return lines


def script_text(reads: list[tuple[int, int, tuple[int, int, int, int, int]]]) -> str:
    """reads: (tick, number of reads, real input) per script line; the ticks are relative to the first line's."""
    return "".join(f"{t} {n} {b:x} {lx} {ly} {rx} {ry}\n" for t, n, (b, lx, ly, rx, ry) in reads)


def parse_states(stdout: str) -> list[tuple[int, int, int, int, int, int, int]]:
    out = []
    for line in stdout.splitlines():
        if line.startswith("state "):
            _, tick, buttons, lx, ly, rx, ry, attr = line.split()
            out.append((int(tick), int(buttons, 16), int(lx), int(ly), int(rx), int(ry), int(attr)))
    return out


def held_of(state: tuple) -> tuple:
    return tuple(state[1:6])


class HidReplayTest(unittest.TestCase):
    exe: Path

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-hid-replay-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos", "audio"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        libs = [] if os.name == "nt" else ["-lm"]
        cls.exe = out / f"hid_replay_test{'.exe' if os.name == 'nt' else ''}"
        cmd = [cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
               "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(out), *sources, "-o", str(cls.exe),
               *libs]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build of hid_replay_test failed:\n{done.stdout}{done.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    # ---- running
    def folder(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="run-", dir=self.tmp.name))

    def file(self, name: str, data: bytes) -> Path:
        path = self.folder() / name
        path.write_bytes(data)
        return path

    def run_case(self, script: str = "", replay: Path | None = None, record: Path | str | None = None,
                 case: str = "run") -> tuple[subprocess.CompletedProcess, Path]:
        folder = self.folder()
        (folder / "script.txt").write_text(script)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OOT2_INPUT", "OOT2_CENSUS"))}
        if replay is not None:
            env["OOT2_INPUT_FILE"] = str(replay)
        if record is not None:
            env["OOT2_INPUT_RECORD"] = str(record)
        try:
            done = subprocess.run([str(self.exe), case, str(folder)], capture_output=True, text=True,
                                  timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"the case ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        return done, folder

    def passes(self, *args, **kwargs) -> tuple[list[tuple], Path]:
        done, folder = self.run_case(*args, **kwargs)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ok run", done.stdout)
        self.assertNotIn("FATAL", done.stderr)
        return parse_states(done.stdout), folder

    def rows(self, folder: Path, kind: str) -> list[list[str]]:
        text = (folder / "census.tsv").read_text()
        return [line.split("\t") for line in text.splitlines() if line.startswith(kind + "\t")]

    # ---- replay
    def test_replay_gives_the_state_of_each_tick(self) -> None:
        spans: list[Span] = [
            (0, 3, ["a"]),
            (5, 4, ["ZR", "Plus", "lx=-32768", "ry=32767"]),   # either case; both ends of the stick range
            (20, 1, ["stickr_down"]),
            (40, 2, ["lx=1"]),
            (60, 3, []),                                        # a stretch of rest
        ]
        ticks = list(range(0, 46)) + [60, 61, 300]              # 300: after a long stretch no line covers
        reads = [(t, 1, (0, 0, 0, 0, 0)) for t in ticks]
        path = self.file("tape.txt", b"# a test tape\n\n" + render(spans, "\r\n"))   # comments, blanks and CR LF are fine
        states, _ = self.passes(script_text(reads), replay=path)
        self.assertEqual([s[0] for s in states], ticks)
        for state in states:
            self.assertEqual(held_of(state), held_at(spans, state[0]), f"tick {state[0]}")
            self.assertEqual(state[6], 1, "the controller stays connected")
        self.assertTrue(any(any(held_of(s)) for s in states))

    def test_replay_overlapping_lines(self) -> None:
        spans: list[Span] = [
            (10, 20, ["a", "lx=100", "ly=200"]),
            (15, 5, ["b", "lx=-5"]),
            (17, 10, ["x", "rx=7"]),
            (16, 4, ["lx=9"]),   # later in the file than the line before it: it wins lx
        ]
        ticks = [0, 12, 18, 22, 28]   # tick 0 is the first read; the others inside the lines, away from where one starts or ends
        path = self.file("tape.txt", render(spans))
        states, _ = self.passes(script_text([(t, 1, (0, 0, 0, 0, 0)) for t in ticks]), replay=path)
        want = [held_at(spans, t) for t in ticks]
        self.assertEqual([held_of(s) for s in states], want)
        self.assertEqual(want[2], (bits("a", "b", "x"), 9, 200, 7, 0))   # the model says what the format promises
        self.assertEqual(want[3], (bits("a", "x"), 100, 200, 7, 0))

    def test_replay_replaces_the_real_input(self) -> None:
        spans: list[Span] = [(2, 3, ["a"])]
        ticks = [0, 3, 6]
        path = self.file("tape.txt", render(spans))
        states, _ = self.passes(script_text([(t, 1, GARBAGE) for t in ticks]), replay=path)
        self.assertEqual([held_of(s) for s in states], [(0, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)])

    def test_replay_counts_from_the_first_read(self) -> None:
        # the script's first line is at the clock's tick 500, so tick 5 of the file is the sixth tick after it
        spans: list[Span] = [(5, 1, ["y"])]
        ticks = [0, 4, 5, 6]
        states, _ = self.passes(script_text([(t + 7000, 1, REST) for t in ticks]),
                                replay=self.file("tape.txt", render(spans)))
        self.assertEqual([held_of(s) for s in states][2], (bits("y"), 0, 0, 0, 0))
        self.assertEqual(sum(1 for s in states if any(held_of(s))), 1)

    # ---- the recorder
    def test_record_writes_runs_by_tick(self) -> None:
        reads = [
            (0, 1, (0, 0, 0, 0, 0)), (1, 1, (0, 0, 0, 0, 0)), (2, 1, (0, 0, 0, 0, 0)),
            (3, 1, (bits("a"), 0, 0, 0, 0)), (4, 1, (bits("a"), 0, 0, 0, 0)), (5, 1, (bits("a"), 0, 0, 0, 0)),
            (6, 1, (bits("a"), -1000, 0, 0, 0)), (7, 1, (bits("a"), -1000, 0, 0, 0)),
            (8, 1, (0, 0, 0, 0, 0)),
            (12, 1, (bits("b", "stickr_right"), 0, 32767, 0, -32768)),   # no reads in 9..11
            (20, 1, (bits("b", "stickr_right"), 0, 32767, 0, -32768)),   # no reads in 13..19: the run goes on across them
            (21, 1, (bits("x", "stickl_up"), 0, 0, 0, 0)),               # the last run is closed when the run stops
            (22, 1, (bits("x", "stickl_up"), 0, 0, 0, 0)),
        ]
        rec = self.folder() / "record.txt"
        states, folder = self.passes(script_text(reads), record=rec)
        want = [r[2] for r in reads]
        self.assertEqual([held_of(s) for s in states], want, "recording passes the real input through")
        out = self.rows(folder, "input-record")   # the report names the record and what it holds
        self.assertEqual(len(out), 1, out)
        self.assertEqual(out[0][1:4], [str(rec), "implemented", "4"])   # four lines written
        self.assertIn("23 ticks, closed", out[0][5])
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, encode_runs([(t, r) for (t, _, r) in reads]))
        self.assertEqual(lines, ["at 3 for 3 a", "at 6 for 2 a lx=-1000", "at 12 for 9 b stickr_right ly=32767 ry=-32768",
                                 "at 21 for 2 x stickl_up"])

    def test_one_state_per_tick_while_recording(self) -> None:
        # the real input changes between two reads of one tick: the first read takes the tick's state, the others get it
        reads = [(0, 1, (bits("a"), 0, 0, 0, 0)), (0, 2, (bits("b"), 5, 0, 0, 0)), (1, 1, (bits("x"), 0, 0, 0, 0)),
                 (1, 1, (0, 0, 0, 0, 0))]
        rec = self.folder() / "record.txt"
        states, _ = self.passes(script_text(reads), record=rec)
        self.assertEqual([held_of(s) for s in states],
                         [(bits("a"), 0, 0, 0, 0)] * 3 + [(bits("x"), 0, 0, 0, 0)] * 2)
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 0 for 1 a", "at 1 for 1 x"])

    def test_record_closes_its_last_run_when_the_run_stops(self) -> None:
        reads = [(0, 1, (0, 0, 0, 0, 0)), (1, 1, (bits("a"), 0, 0, 0, 0)), (2, 1, (bits("a"), 0, 0, 0, 0)),
                 (3, 1, (bits("a"), 0, 0, 0, 0))]
        rec = self.folder() / "record.txt"
        done, folder = self.run_case(script_text(reads), record=rec, case="run_then_fatal")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)   # the run stopped on a fatal
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 1 for 3 a"])   # the run of A, open at the stop, is closed and written
        self.assertIn("4 ticks, closed", self.rows(folder, "input-record")[0][5])

    def test_record_counts_a_held_state_across_unread_ticks(self) -> None:
        # A is first read at tick 3, B at tick 10: A was held for the seven ticks 3..9, whatever the game read in between
        reads = [(0, 1, REST), (3, 1, (bits("a"), 0, 0, 0, 0)), (10, 1, (bits("b"), 0, 0, 0, 0))]
        rec = self.folder() / "record.txt"
        self.passes(script_text(reads), record=rec)
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 3 for 7 a", "at 10 for 1 b"])
        self.assertEqual(lines, encode_runs([(t, r) for (t, _, r) in reads]))

    def test_record_keeps_every_closed_run_when_the_process_is_killed(self) -> None:
        # _Exit: no stop path and no stdio flush. The run of A closed when B was read, so it is in the file; B is open
        reads = [(0, 1, REST), (1, 1, (bits("a"), 0, 0, 0, 0)), (2, 1, (bits("a"), 0, 0, 0, 0)),
                 (3, 1, (bits("b"), 0, 0, 0, 0)), (4, 1, (bits("b"), 0, 0, 0, 0))]
        rec = self.folder() / "record.txt"
        done, _ = self.run_case(script_text(reads), record=rec, case="run_then_exit")
        self.assertEqual(done.returncode, EXIT_KILLED, done.stdout + done.stderr)
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 1 for 2 a"])

    def test_record_then_replay_gives_the_same_states(self) -> None:
        reads = [(t, 1, (bits(*names), lx, ly, rx, ry)) for t, (names, lx, ly, rx, ry) in enumerate([
            ((), 0, 0, 0, 0), ((), 0, 0, 0, 0), (("a",), 0, 0, 0, 0), (("a", "zr"), 0, 0, 0, 0),
            (("a", "zr"), 3000, -3000, 0, 0), (("stickl_left", "up"), 3000, -3000, 12, -12), ((), 0, 0, 0, 0),
            ((), 0, 0, 0, 0), (("minus",), -32768, 32767, -32768, 32767), (("minus",), -32768, 32767, -32768, 32767),
            (("l", "r", "left"), 0, 0, 0, 0)])]
        reads += [(40, 1, (bits("b"), 0, 0, 0, 0)), (41, 1, (0, 0, 0, 0, 0)), (90, 1, (bits("y"), 1, 0, 0, 0))]
        rec = self.folder() / "record.txt"
        recorded, _ = self.passes(script_text(reads), record=rec)
        self.assertEqual([held_of(s) for s in recorded], [r[2] for r in reads])
        # the same run again: the recorded file replays, and the real input is rubbish that must not show
        replayed, _ = self.passes(script_text([(t, n, REST) for t, n, _ in reads]), replay=rec)
        self.assertEqual(replayed, recorded)

    def test_record_while_replaying_writes_the_replayed_input(self) -> None:
        spans: list[Span] = [(1, 6, ["a", "lx=10"]), (3, 6, ["a", "ly=5"])]   # overlapping: not the recorder's form
        ticks = list(range(0, 12))
        rec = self.folder() / "record.txt"
        states, _ = self.passes(script_text([(t, 1, REST) for t in ticks]),
                                replay=self.file("tape.txt", render(spans)), record=rec)
        want = [(t, held_at(spans, t)) for t in ticks]
        self.assertEqual([held_of(s) for s in states], [h for _, h in want])
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, encode_runs(want))
        self.assertEqual(lines, ["at 1 for 2 a lx=10", "at 3 for 4 a lx=10 ly=5", "at 7 for 2 a ly=5"])

    # ---- off, and the report
    def test_off_changes_nothing(self) -> None:
        reads = [(0, 1, (bits("a"), 1, 2, 3, 4)), (0, 2, GARBAGE), (1, 1, (bits("b", "zl"), -32768, 32767, 0, 0)),
                 (2, 3, (0, 0, 0, 0, 0))]
        states, folder = self.passes(script_text(reads))
        want = [r[2] for r in reads for _ in range(r[1])]
        self.assertEqual([held_of(s) for s in states], want)   # every read is the live sample, as before
        self.assertEqual(self.rows(folder, "input"), [])
        self.assertEqual(self.rows(folder, "input-record"), [])
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["census.tsv", "script.txt"])

    def test_report_names_the_file_and_its_hash(self) -> None:
        spans: list[Span] = [(0, 2, ["a"]), (3, 2, ["b", "ly=-7"])]
        data = b"# a tape\n" + render(spans) + b"\n"
        path = self.file("tape.txt", data)
        _, folder = self.passes(script_text([(t, 2, REST) for t in range(5)]), replay=path)
        rows = self.rows(folder, "input")
        self.assertEqual(len(rows), 1, rows)
        kind, key, status, calls, first_at, why = rows[0]
        self.assertEqual((kind, key, status), ("input", str(path), "implemented"))
        self.assertEqual(calls, "5")   # five ticks were given a state, two reads each
        self.assertIn("sha256 " + hashlib.sha256(data).hexdigest() + ";", why)
        self.assertIn("2 lines", why)
        self.assertIn(f"tick 0 was virtual tick {FIRST_TICK}", why)
        self.assertEqual(self.rows(folder, "input-record"), [], "no recorder, no row")

    def test_report_changes_one_byte_changes_the_hash(self) -> None:
        hashes = []
        for data in (b"at 0 for 1 a\n", b"at 0 for 1 b\n"):
            _, folder = self.passes(script_text([(0, 1, REST)]), replay=self.file("tape.txt", data))
            hashes.append(self.rows(folder, "input")[0][5].split("sha256 ")[1].split(";")[0])
        self.assertNotEqual(hashes[0], hashes[1])
        self.assertEqual(hashes[0], hashlib.sha256(b"at 0 for 1 a\n").hexdigest())

    # ---- a file that cannot be taken stops by name
    def stops(self, data: bytes | None, line: int | None, what: str) -> None:
        path = self.file("tape.txt", data) if data is not None else self.folder() / "no-such-tape.txt"
        done, folder = self.run_case(ONE_READ, replay=path)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        where = f"input file {path}, line {line}: " if line else f"input file {path}: "
        self.assertEqual(fatal, [f"FATAL: {where}{what}"], done.stderr)
        self.assertNotIn("ok run", done.stdout)
        self.assertEqual(done.stdout.count("state "), 0, "the run stopped before the game could read")
        stop = [row for row in self.rows(folder, "stop")]
        self.assertEqual(len(stop), 1)
        self.assertEqual(stop[0][1:3], ["exit 70", "named-fatal"])

    def test_malformed_missing_file(self) -> None:
        done, folder = self.run_case(ONE_READ, replay=self.folder() / "no-such-tape.txt")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(len(fatal), 1, done.stderr)
        self.assertRegex(fatal[0], r"^FATAL: input file .*no-such-tape\.txt: cannot be opened \(")

    def test_malformed_record_over_the_input(self) -> None:
        path = self.file("tape.txt", TAPE_A)
        done, _ = self.run_case(ONE_READ, replay=path, record=path)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(fatal, [f"FATAL: input record {path}: it is the input file being replayed"])
        self.assertEqual(path.read_bytes(), TAPE_A, "the input file was left as it was")

    def refuses_record_alias(self, tape: Path, alias: str | Path) -> None:
        """The record names the replayed file by another spelling: refused by name, and the tape is left as it was."""
        data = tape.read_bytes()
        done, _ = self.run_case(ONE_READ, replay=tape, record=alias)   # str: a Path would drop the "./"
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(fatal, [f"FATAL: input record {alias}: it is the input file being replayed"])
        self.assertEqual(tape.read_bytes(), data, "the input file was left as it was")

    def test_malformed_record_over_the_input_by_a_dot_alias(self) -> None:
        tape = self.file("tape.txt", b"at 0 for 3 a\nat 10 for 5 b\n")
        self.refuses_record_alias(tape, f"{tape.parent}{os.sep}.{os.sep}{tape.name}")

    def test_malformed_record_over_the_input_by_an_upper_case_alias(self) -> None:
        tape = self.file("Tape.txt", b"at 0 for 3 a\nat 10 for 5 b\n")
        alias = tape.parent / "TAPE.TXT"
        if not alias.exists():
            self.skipTest("the file system tells upper case from lower case: that is no alias of the tape")
        self.refuses_record_alias(tape, alias)

    def test_record_over_another_existing_file_replaces_it(self) -> None:
        # the counterpart of the refusals: a record file that already exists with other bytes is written over, as ever
        tape = self.file("tape.txt", b"at 0 for 3 a\nat 10 for 5 b\n")
        rec = self.file("record.txt", b"old record\n")
        self.passes(script_text([(t, 1, REST) for t in range(12)]), replay=tape, record=rec)
        lines = [l for l in rec.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 0 for 3 a", "at 10 for 2 b"])
        self.assertEqual(tape.read_bytes(), b"at 0 for 3 a\nat 10 for 5 b\n")

    def test_malformed_record_cannot_be_created(self) -> None:
        target = self.folder() / "no-such-folder" / "record.txt"
        done, _ = self.run_case(ONE_READ, record=target)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(len(fatal), 1, done.stderr)
        self.assertTrue(fatal[0].startswith(f"FATAL: input record {target}: cannot be created ("), fatal[0])

    # ---- an empty file and an empty variable
    def test_empty_file_is_rest(self) -> None:
        states, _ = self.passes(script_text([(0, 1, REST), (1, 1, REST)]), replay=self.file("tape.txt", b""))
        self.assertEqual([held_of(s) for s in states], [(0, 0, 0, 0, 0)] * 2)

    def test_replay_skips_a_byte_order_mark(self) -> None:
        # Windows tools write EF BB BF at the start of a UTF-8 file
        spans: list[Span] = [(1, 2, ["a"]), (4, 1, ["b"])]
        path = self.file("tape.txt", b"\xef\xbb\xbf# a tape\n" + render(spans, "\r\n"))
        ticks = list(range(0, 6))
        states, _ = self.passes(script_text([(t, 1, REST) for t in ticks]), replay=path)
        self.assertEqual([held_of(s) for s in states], [held_at(spans, t) for t in ticks])
        self.assertTrue(any(any(held_of(s)) for s in states))

    def test_empty_variable_is_off(self) -> None:
        folder = self.folder()
        (folder / "script.txt").write_text(script_text([(0, 1, GARBAGE)]))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OOT2_INPUT", "OOT2_CENSUS"))}
        env["OOT2_INPUT_FILE"] = env["OOT2_INPUT_RECORD"] = ""
        done = subprocess.run([str(self.exe), "run", str(folder)], capture_output=True, text=True, timeout=TIMEOUT,
                              env=env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(held_of(parse_states(done.stdout)[0]), GARBAGE)
        self.assertEqual(self.rows(folder, "input"), [])


MALFORMED = {
    "unknown_line_kind": (b"# one\nat 0 for 1 a\njump 5 for 3 a\n", 3, 'unknown line kind "jump" (a line starts with "at")'),
    "missing_tick": (b"\n\nat\n", 3, '"at" needs a tick, then "for" and a count'),
    "tick_not_a_number": (b"at x for 3 a\n", 1, 'the tick "x" is not a decimal number'),
    "tick_negative": (b"at -5 for 3 a\n", 1, 'the tick "-5" is not a decimal number'),
    "tick_too_large": (b"at 4294967296 for 1\n", 1, 'the tick "4294967296" is above 4294967295'),
    "missing_for": (b"at 5 3 a\n", 1, 'expected "for" and a count after the tick, not "3"'),
    "missing_for_at_end": (b"at 5\n", 1, 'expected "for" and a count after the tick, not "the end of the line"'),
    "missing_count": (b"at 5 for\n", 1, '"for" needs a count'),
    "count_not_a_number": (b"at 5 for 3x a\n", 1, 'the count "3x" is not a decimal number'),
    "count_too_large": (b"at 5 for 99999999999\n", 1, 'the count "99999999999" is above 4294967295'),
    "count_zero": (b"at 5 for 0 a\n", 1, "the count is 0; a stretch lasts 1 tick or more"),
    "unknown_button": (b"at 1 for 1 a\r\nat 2 for 1 b turbo\r\n", 2, 'unknown button "turbo"'),
    "button_twice": (b"at 1 for 1 a zr A\n", 1, "button a is given twice on the line"),
    "unknown_stick": (b"at 1 for 1 lz=5\n", 1, 'unknown stick "lz" (the sticks are lx, ly, rx and ry)'),
    "stick_twice": (b"at 1 for 1 lx=5 LX=6\n", 1, "lx is given twice on the line"),
    "stick_not_a_number": (b"at 1 for 1 rx=abc\n", 1, 'the value of rx, "abc", is not a number'),
    "stick_empty": (b"at 1 for 1 ry=\n", 1, 'the value of ry, "", is not a number'),
    "stick_too_high": (b"at 1 for 1 lx=32768\n", 1, "lx=32768 is outside -32768..32767"),
    "stick_too_low": (b"at 1 for 1 ly=-32769\n", 1, "ly=-32769 is outside -32768..32767"),
    "binary_file": (b"at 0 for 1 a\nat 1 for 1 b\x00\n", 2, "a NUL byte; the input file is text"),
    "byte_order_mark_not_at_the_start": (b"at 0 for 1 a\n\xef\xbb\xbfat 1 for 1 b\n", 2,
                                         'unknown line kind "???at" (a line starts with "at")'),
    "garbage_word_is_shown_safely": (b"at 0 for 1 a\n\xff\xfe\x01 5\n", 2, 'unknown line kind "???" (a line starts with "at")'),
}


def _malformed(data: bytes, line: int, what: str):
    def test(self: HidReplayTest) -> None:
        self.stops(data, line, what)
    return test


for _name, (_data, _line, _what) in MALFORMED.items():   # one test method each, so one check going red is one test red
    setattr(HidReplayTest, f"test_malformed_{_name}", _malformed(_data, _line, _what))


if __name__ == "__main__":
    unittest.main()
