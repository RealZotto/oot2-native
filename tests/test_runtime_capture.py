# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the M1 capture's writer (runtime/nvn/nvn_capture.c,
runtime/include/oot2/capture.h): the file the runtime writes, read back with
the checker's own reader (tools/check_capture.py), so both sides keep one
format.

Builds tests/runtime/capture_test.c with runtime/core and runtime/nvn at -O1 and
-O2, as tests/test_runtime_nvn_pace.py builds its harness, and runs each case in
its own process with its own folder; the capture switches come from the
environment, as in a run of the game.

- frames: a worker's 30 ms of CPU between acquire and present lands in the frame
  line of that present and not the next one; the submitting and presenting
  thread are named; the header, the title mark and the closing record.
- gap: a worker that holds its virtual core for 30 ms without running shows in the "gap" line of its thread (core time
  less CPU time over the frames, summed and the largest), which always equals what the frame lines give.
- stop70, stop71, stop72, exitprocess, census73: the closing record is written
  on every stop path, with the exit code and its reason.
- length: the run ends by itself at the capture's length, exit code 0; memory,
  count and cost lines once a second; a pipeline miss and a GPU pass recorded.
- off: no file and no frame boundary without OOT2_CAPTURE.
- shown: shown ticks a back end reports late are written as they come.
- cost: the CPU time of the present call with the capture on and off, printed
  for the record (the capture study keeps the figures).

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import check_capture as cc  # noqa: E402
import test_runtime_nvn as nvn  # noqa: E402

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "capture_test.c"
TIMEOUT = 180
EXIT_HUNG = 3   # capture_test.c's watchdog: the case had not ended after 90 s
MS = 1_000_000


class CaptureTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc_path = nvn.find_compiler()
        if not cc_path:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-capture-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"capture_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc_path), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o",
                   str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, env: dict[str, str] | None = None,
                 capture: bool = True) -> tuple[subprocess.CompletedProcess, Path]:
        folder = Path(tempfile.mkdtemp(prefix=f"{case}{level}-", dir=self.tmp.name))
        e = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_") or k in ("OOT2_CC",)}
        if capture:
            e["OOT2_CAPTURE"] = "1"
        # the test's game passes no sync to AcquireTexture and relies on the acquire to pace its presents on the host's
        # clock: the old placement, the hold inside the call (the acquire sync hold; a game that passes no sync is not held by the default)
        e["OOT2_HOLD_PLACEMENT"] = "acquire"
        e.update(env or {})
        try:
            done = subprocess.run([str(self.exes[level]), case, str(folder)], capture_output=True, text=True,
                                  timeout=TIMEOUT, env=e)
        except subprocess.TimeoutExpired as x:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{x.stdout or ''}{x.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done, folder

    def capture(self, folder: Path) -> tuple[Path, cc.Capture]:
        files = sorted(folder.glob("capture-*.tsv"))
        self.assertEqual(len(files), 1, f"capture files in {folder}: {files}")
        return files[0], cc.parse(files[0])

    def assert_closed(self, path: Path, cap: cc.Capture, code: int, reason: str) -> None:
        self.assertIsNotNone(cap.end, path.read_text()[-2000:])
        self.assertEqual(cap.last_kind, "end")
        self.assertFalse(cap.cut_line or cap.after_end)
        self.assertEqual((cap.end.code, cap.end.reason), (code, reason))
        self.assertTrue(path.read_bytes().endswith(b"\n"))

    def assert_gaps_match_frames(self, path: Path, cap: cc.Capture) -> dict[int, tuple[int, int, int]]:
        """The "gap" lines (handle entry frames sum_ns max_ns), one per thread and before the closing record, against the
        same numbers worked out from the frame lines: per frame, the core time less the CPU time, 0 when negative."""
        lines = [x.split("\t") for x in path.read_text().splitlines()]
        kinds = [x[0] for x in lines]
        gaps = [x for x in lines if x[0] == "gap"]
        self.assertTrue(gaps)
        self.assertLess(max(i for i, k in enumerate(kinds) if k == "gap"), kinds.index("end"))
        got = {(int(x[1], 16), int(x[2], 16)): tuple(int(v) for v in x[3:6]) for x in gaps}
        self.assertEqual(len(got), len(gaps))
        want: dict[int, list[int]] = {}
        for f in cap.frames:
            for handle, (cpu, core) in f.threads.items():
                gap = max(core - cpu, 0)
                w = want.setdefault(handle, [0, 0, 0])
                w[0] += 1
                w[1] += gap
                w[2] = max(w[2], gap)
        self.assertEqual({h for h, _ in got}, set(want))
        self.assertEqual({h for h in cap.threads}, set(want))
        for (handle, entry), numbers in got.items():
            self.assertEqual(entry, cap.threads[handle][0])
            self.assertEqual(list(numbers), want[handle], hex(handle))
        return {h: tuple(v) for h, v in want.items()}

    def test_a_thread_that_holds_its_core_without_running_shows_in_its_gap_line(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "gap", {"OOT2_CAPTURE_TITLE": "3"})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok gap\n", done.stdout)
                worker = int(re.search(r"^worker handle (0x[0-9a-f]+)$", done.stdout, re.M).group(1), 16)
                path, cap = self.capture(folder)
                self.assert_closed(path, cap, 0, "exit")
                gaps = self.assert_gaps_match_frames(path, cap)
                frames, total, largest = gaps[worker]
                self.assertGreaterEqual(frames, 2)
                self.assertGreaterEqual(largest, 20 * MS)    # a hold of 30 ms, in one frame
                self.assertGreaterEqual(total, 2 * 20 * MS)  # two of them
                for burn in (5, 8):   # the worker held its core in frames 4 and 7 (presents 5 and 8) and ran in neither
                    cpu, core = cap.frames[burn - 1].threads[worker]
                    self.assertEqual(cap.frames[burn - 1].seq, burn)
                    self.assertGreaterEqual(core - cpu, 20 * MS)

    def test_a_threads_time_lands_in_the_frame_it_was_spent_in(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "frames", {"OOT2_CAPTURE_TITLE": "3"})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok frames\n", done.stdout)
                game = int(re.search(r"^game handle (0x[0-9a-f]+)$", done.stdout, re.M).group(1), 16)
                worker = int(re.search(r"^worker handle (0x[0-9a-f]+)$", done.stdout, re.M).group(1), 16)
                path, cap = self.capture(folder)
                self.assert_closed(path, cap, 0, "exit")
                self.assertEqual([p.seq for p in cap.presents], list(range(1, 11)))
                self.assertEqual([p.due for p in cap.presents], list(range(10)))
                self.assertTrue(all(p.requested == 1 and p.held == 0 and p.late == 0 for p in cap.presents))
                self.assertTrue(all(p.frame_ns > 0 for p in cap.presents[1:]))
                frames = {f.seq: f for f in cap.frames}
                self.assertEqual(sorted(frames), list(range(1, 11)))
                for f in cap.frames:
                    self.assertEqual((f.submitter, f.submits, f.submitters, f.presenter), (game, 1, 1, game))
                cpu = {s: frames[s].threads.get(worker, (0, 0)) for s in frames}
                for burn in (5, 8):   # the worker burned between acquire and present of frames 4 and 7
                    self.assertGreaterEqual(cpu[burn][0], 20 * MS, f"present {burn}: {cpu}")
                    self.assertGreaterEqual(cpu[burn][1], 20 * MS, f"present {burn}: {cpu}")
                for quiet in (4, 6, 7, 9):
                    self.assertLess(cpu[quiet][0], 10 * MS, f"present {quiet}: {cpu}")
                self.assertIn(worker, cap.threads)
                self.assertIn(game, cap.threads)
                self.assert_gaps_match_frames(path, cap)
                h = cap.header
                self.assertEqual((h["back_end"], h["census_mode"], h["title_mark_present"]), ("null", "0", "3"))
                self.assertEqual(h["driver"], "none (null back end)")
                self.assertGreater(int(h["ram_available"]), 0)
                self.assertEqual(h["test_before"], "noted before the start with a tab")
                self.assertEqual(h["test_after"], "noted after the start")
                self.assertEqual(cap.title.seq, 3)
                v = cc.check(cap)
                ok = {c.name for c in v.conditions if c.ok}
                self.assertTrue({"closing record", "judged run", "header", "named fatals", "title reached"} <= ok,
                                cc.report(path, v))
                self.assertIn("window complete", {c.name for c in v.conditions if not c.ok})

    def test_every_stop_path_writes_the_closing_record(self) -> None:
        cases = {"stop70": (70, "fatal", {}), "stop71": (71, "unimplemented", {}), "stop72": (72, "guest-stop", {}),
                 "exitprocess": (0, "exit", {}),
                 "census73": (73, "census-limit", {"OOT2_CENSUS": "1", "OOT2_CENSUS_FRAMES": "3"})}
        for case, (code, reason, env) in cases.items():
            for level in nvn.LEVELS:
                with self.subTest(case=case, level=level):
                    done, folder = self.run_case(level, case, env)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    path, cap = self.capture(folder)
                    self.assert_closed(path, cap, code, reason)
                    self.assertEqual([p.seq for p in cap.presents], [1, 2, 3])
                    self.assertEqual(cap.end.fatals, 1 if code in (70, 71, 72) else 0)
                    self.assertEqual(cap.header["census_mode"], "1" if env else "0")
                    v = cc.check(cap)
                    named = next(c for c in v.conditions if c.name == "named fatals")
                    self.assertEqual(named.ok, code == 0, named.value)

    def test_the_run_ends_at_the_capture_length_with_memory_once_a_second(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "length", {"OOT2_CAPTURE_TITLE": "2",
                                                               "OOT2_CAPTURE_SECONDS": "2.2"})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("the capture ended at its length, 2.200 s after the title mark", done.stderr)
                path, cap = self.capture(folder)
                self.assert_closed(path, cap, 0, "length")
                self.assertEqual(cap.title.seq, 2)
                last = cap.presents[-1]
                self.assertGreaterEqual(last.t - cap.title.t, 2_200_000_000)
                self.assertLess(cap.presents[-2].t - cap.title.t, 2_200_000_000)
                self.assertGreaterEqual(len(cap.memory), 2)
                self.assertGreaterEqual(len(cap.counts), 2)
                self.assertGreaterEqual(len(cap.costs), 2)
                for m in cap.memory:
                    ws, private, available, pages = m.values
                    self.assertGreater(ws, 0)
                    self.assertGreater(private, 0)
                    self.assertGreater(available, 0)
                    self.assertTrue(pages >= 0 or "pages_output_counter_state" in cap.header, m)
                self.assertEqual(cap.end.counters, (0, 0, 1, 0, 0))
                self.assertEqual(cap.gpu, [(5, 0, 1234)])
                self.assertGreater(cap.end.working_set, 0)
                self.assertEqual(cap.end.cost_presents, len(cap.presents))
                v = cc.check(cap)
                self.assertEqual(next(c for c in v.conditions if c.name == "pipeline misses").value, "1")

    def test_a_run_with_a_test_switch_on_says_so_in_the_header_and_the_judge_refuses_it(self) -> None:
        # test_movie_blit changes the pictures of the movies (settings.h): the header names it even for a run that never
        # reached a movie, so a capture of such a run can never pass as a proof
        for level in nvn.LEVELS:
            for on in (True, False):
                with self.subTest(level=level, on=on):
                    env = {"OOT2_CAPTURE_TITLE": "3"}
                    if on:
                        env["OOT2_TEST_MOVIE_BLIT"] = "1"
                    done, folder = self.run_case(level, "frames", env)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    path, cap = self.capture(folder)
                    self.assert_closed(path, cap, 0, "exit")
                    header = cap.header["test_movie_blit"]
                    self.assertEqual(header.startswith("1 ("), on, header)
                    self.assertEqual(header == "0", not on, header)
                    judged = {c.name: c for c in cc.check(cap).conditions}
                    self.assertEqual(judged["test settings"].ok, not on, judged["test settings"])
                    self.assertTrue(judged["judged run"].ok)   # a different condition: census mode is another thing

    def test_without_the_switch_nothing_is_written_and_no_frame_is_marked(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "off", capture=False)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("no frame time recorded", done.stdout)
                self.assertEqual(list(folder.iterdir()), [])
                self.assertNotIn("capture:", done.stderr)

    def test_shown_ticks_reported_late_are_written_as_they_come(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "shown")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                path, cap = self.capture(folder)
                self.assert_closed(path, cap, 0, "exit")
                self.assertEqual(cap.shown, {s: 10 * s for s in range(1, 11)})
                self.assertTrue(all(p.shown == 0 for p in cap.presents))
                self.assertEqual(cap.header["back_end"], "test-shown")
                self.assertNotIn("driver", cap.header)   # a back end other than null notes it itself

    def test_the_capture_cost_on_and_off(self) -> None:
        figures = {}
        for on in (True, False):
            done, folder = self.run_case(nvn.LEVELS[-1], "cost", capture=on)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            m = re.search(r"^cost: capture (on|off), 600 presents beside 32 sleeping guest threads: the present call "
                          r"took ([\d.]+) us of CPU on average, p50 ([\d.]+), p99 ([\d.]+), max ([\d.]+)$",
                          done.stdout, re.M)
            self.assertIsNotNone(m, done.stdout)
            self.assertEqual(m.group(1), "on" if on else "off")
            figures[m.group(1)] = [float(x) for x in m.groups()[1:]]
            if on:
                _, cap = self.capture(folder)
                self.assertEqual(cap.end.cost_presents, 600)
                figures["own"] = [cap.end.cost_sum / 600 / 1e3, cap.end.cost_max / 1e3]
            else:
                self.assertEqual(list(folder.iterdir()), [])
        print(f"\ncapture cost, us of CPU per present call (mean, p50, p99, max): on {figures['on']}, "
              f"off {figures['off']}; the capture's own accounting, mean and max: {figures['own']}")
        self.assertLess(figures["on"][0], 2000.0)

    def test_unknown_case_is_refused(self) -> None:
        done, _ = self.run_case(nvn.LEVELS[0], "no_such_case", capture=False)
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
