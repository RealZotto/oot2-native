# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the NVN2 front end's queues, syncs, events, reports and
windows, and of the null back end's completion (architecture section 5).

Builds tests/runtime/nvn_queue_test.c with runtime/core and runtime/nvn, as
tests/test_runtime_nvn.py builds its harness, at -O1 and -O2, and reads what
it prints.

- fences: three submissions on two queues, their completion held, then
  completed one at a time: each fence value is done in submission order,
  and what a submission writes (an event, a report) appears only once it is.
- timestamps: a report's timestamp is the guest's system tick (31.25 MHz)
  read from the host clock when the report completes, bracketed by the host
  clock around two submissions 20 ms apart, and never goes back.
- render_cond: conditional rendering decided at replay, from memory for a
  complete report, as 0 for a timestamp report not complete yet (in the same
  submission or in a held one), as visible for samples passed not complete.
- window: three textures acquired and presented in order, each present
  handing its texture and the crop to the back end; the acquire sync waits
  for the work before that texture's last present.
- syncwait: a SyncWait on a held fence gives its virtual core up, so a guest
  thread of worse priority on the same core runs and completes it.
- wait_events: every WaitEvent checked at completion, one log line per call
  site, and one more for a site's first unsatisfied wait.
- failing cases: what the front end cannot honour stops by name.

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

import test_runtime_nvn as nvn  # noqa: E402

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_queue_test.c"
TIMEOUT = 120
EXIT_HUNG = 3   # nvn_queue_test.c's watchdog: the case had not ended after 20 s; it logs the kernel's state first

DRAWS = ["1 renders", "2 skipped", "3 renders", "4 renders", "5 skipped", "6 renders", "7 renders"]

# The failing cases: (case, exit code, text the FATAL line must contain)
FAILING = (
    ("event_cpu_invisible", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: EventInitialize: the memory pool "),
    ("sync_uninitialised", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: SyncWait on the sync "),
    ("present_past_textures", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: QueuePresentTexture: texture 2 of the window "),
    ("window_five", nvn.EXIT_UNIMPLEMENTED,
     "FATAL: unimplemented nvn: WindowBuilderSetTextures: 5 textures at "),
    ("report_no_pool", nvn.EXIT_FATAL,
     "gfx: null back end: a counter report at 0x1000 (16 bytes) is in no memory pool the GPU can reach"),
)
FAILING_MORE = {
    "event_cpu_invisible": "is one the CPU cannot see, where the front end has no word to signal or read",
    "sync_uninitialised": "which SyncInitialize did not initialise",
    "present_past_textures": "which has 2",
    "window_five": "past the 4 a window holds here",
}


class QueueTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvnqueue-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_queue_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *, timeout: int = TIMEOUT, **census: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}
        env.update(census)
        try:
            done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout,
                                  env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {timeout} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def assert_passes(self, case: str, **census: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, **census)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                runs.append(done)
        return runs

    def test_fences_complete_in_submission_order_across_two_queues(self) -> None:
        for done in self.assert_passes("fences"):
            lines = [line for line in done.stdout.splitlines() if line.startswith(("complete ", "all at once "))]
            self.assertEqual(lines, ["complete 1 event 1", "complete 2 event 2", "complete 3 event 12",
                                     "all at once event 7"])

    def test_report_timestamps_are_guest_ticks_of_the_host_clock(self) -> None:
        for done in self.assert_passes("timestamps"):
            m = re.search(r"^two reports (\d+) ticks apart, (\d+) ns, the host between (\d+) and (\d+) ns$",
                          done.stdout, re.M)
            self.assertIsNotNone(m, done.stdout)
            ticks, span, low, high = (int(g) for g in m.groups())
            self.assertEqual(span, ticks * 32)
            self.assertTrue(low - 64 <= span <= high + 64, m[0])
            self.assertGreaterEqual(low, 20_000_000)

    def test_conditional_rendering_on_complete_and_incomplete_reports(self) -> None:
        for done in self.assert_passes("render_cond"):
            draws = re.findall(r"^draw (\d+ \w+)$", done.stdout, re.M)
            self.assertEqual(draws, DRAWS)

    def test_window_of_three_textures_acquired_and_presented_in_order(self) -> None:
        for done in self.assert_passes("window"):
            frames = re.findall(r"^frame (\d+) texture (\d+) shown (\d+)$", done.stdout, re.M)
            self.assertEqual([(int(f), int(t), int(s)) for f, t, s in frames],
                             [(f, f % 3, f % 3 + 1) for f in range(9)])
            self.assertIn("nvn: WindowInitialize: 3 textures, 3 active, interval 1, the first 1920x1080", done.stderr)
            self.assertIn("nvn: the game sets present interval 2 (was 1)", done.stderr)

    def test_syncwait_gives_its_core_up_while_it_blocks(self) -> None:
        for done in self.assert_passes("syncwait"):
            self.assertIn("order 1 2 3 4 result 1\n", done.stdout)

    def test_null_census_can_be_read_during_replay_and_from_fatal_hooks(self) -> None:
        cases = ("census_fatal_replay", "census_at_draw", "census_fatal_complete",
                 "census_fatal_held_complete", "census_fatal_held_release")
        for level in nvn.LEVELS:
            for case in cases:
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case, timeout=5)
                    self.assertEqual(done.returncode, 0 if case == "census_at_draw" else nvn.EXIT_FATAL,
                                     done.stdout + done.stderr)
                    self.assertIn("stat| waitevents 1 WaitEvents met 1, unsatisfied at completion 1\n", done.stdout)
                    self.assertIn("unsatisfied 1 (later by GPU 0, later by CPU 0, not satisfied by the end 1, "
                                  "dropped from the full list 0)", done.stdout)
                    self.assertIn("stat| waitsite ", done.stdout)
                    if case == "census_at_draw":
                        self.assertIn("draw| census complete\nok census_at_draw\n", done.stdout)
                    else:
                        self.assertIn("stop| census complete\n", done.stdout)
                        self.assertNotIn("fatal completion returned", done.stdout)
                        self.assertEqual(len(re.findall(r"^FATAL:", done.stderr, re.M)), 1, done.stderr)

    def test_null_census_on_another_thread_waits_for_the_submission_lock(self) -> None:
        self.assert_passes("census_other_thread")

    def test_wait_events_are_counted_per_call_site_by_what_satisfied_them_and_the_sites_sum_to_the_totals(self) -> None:
        for done in self.assert_passes("wait_sites"):
            out = done.stdout
            # the wait inside a recording called twice counts at its own site twice (its record, caller included, is
            # copied by CallCommands), the unsatisfied one is satisfied later by the CPU's signal
            self.assertRegex(out, r"(?m)^site 0x\w+ \(main\+0x6000\) gpu 0 cpu 0 initial 4 unsatisfied 0 later_gpu 0 later_cpu 0$")
            self.assertRegex(out, r"(?m)^site 0x\w+ \(main\+0x7000\) gpu 0 cpu 0 initial 0 unsatisfied 1 later_gpu 0 later_cpu 1$")
            # 72 sites met, 64 in the table: the other 8 (each wait satisfied by the word's own value) share a bucket
            self.assertIn("other caller 0 initial 8 unsatisfied 0\n", out)
            self.assertIn("totals waits 75 unsatisfied 1 initial 74 later_cpu 1\n", out)
            self.assertRegex(out, r"(?m)^stat\| waitsite 0x\w+ \(main\+0x6000\) 4 by a GPU signal in order 0, by a CPU signal before 0, already satisfied 4,")
            self.assertIn("stat| waitsite other sites 8 ", out)

    def test_a_window_of_interval_1_with_no_hold_is_not_marked(self) -> None:
        for done in self.assert_passes("census_interval1", OOT2_CENSUS="0"):
            self.assertRegex(done.stderr, r"(?m)^\[rt\] nvn: present 60 intervals over 60 presents: requested 1 x60, "
                                          r"held 0 x60$")
            self.assertNotIn("NOT every interval 1", done.stderr)

    def test_a_window_paced_by_a_frame_rate_target_names_the_target(self) -> None:
        # the game's interval is ignored while a target is set, so an unmarked "requested 1" line alone would hide
        # what paced the window
        for done in self.assert_passes("census_interval1", OOT2_CENSUS="0", OOT2_FRAME_RATE_TARGET="120"):
            self.assertRegex(done.stderr, r"(?m)^\[rt\] nvn: present 60 intervals over 60 presents: requested 1 x60, "
                                          r"held 0 x60, frame rate target 120 a second on 60 \(the game's interval is "
                                          r"ignored on those\)$")

    def test_two_pending_waits_keep_their_own_site_when_the_first_is_satisfied_first(self) -> None:
        for done in self.assert_passes("wait_sites_order"):
            out = done.stdout
            # the first site's own answer is the GPU's signal, the second's the CPU's (a site the layout does not name prints bare)
            self.assertRegex(out, r"(?m)^site 0x\w*8000 unsatisfied 1 later_gpu 1 later_cpu 0 pending 0$")
            self.assertRegex(out, r"(?m)^site 0x\w*9000 unsatisfied 1 later_gpu 0 later_cpu 1 pending 0$")
            self.assertIn("totals unsatisfied 2 later_gpu 1 later_cpu 1 pending 0\n", out)

    def test_each_wait_event_site_is_logged_with_its_answer(self) -> None:
        for done in self.assert_passes("wait_events"):
            self.assertIn("waits 7 unsatisfied 3\n", done.stdout)
            # sorted by what satisfied them: the GPU in order, a CPU signal before, the word's own value; the
            # unsatisfied waits are satisfied later by the CPU (two: the case a back end replaying at submit cannot
            # honour) or by the GPU of a later submission (one)
            self.assertIn("classes gpu 2 cpu 1 initial 1 unsatisfied 3 later_gpu 1 later_cpu 2\n", done.stdout)
            # the census rows of the same counters: kind nvn-stat, one per key, the count then the words
            self.assertIn("stat| waitevents 7 WaitEvents met 7, unsatisfied at completion 3\n", done.stdout)
            self.assertIn("stat| waitclasses 7 by a GPU signal in order 2, by a CPU signal before 1, already satisfied 1, "
                          "unsatisfied 3 (later by GPU 1, later by CPU 2, not satisfied by the end 0, dropped from the "
                          "full list 0)\n", done.stdout)
            self.assertIn("stat| recordings 4 inline 0, slow 4, ended with a handle 4, ended empty 0\n", done.stdout)
            # the list of unsatisfied waits holds 64: of 70 more waits, 64 are found satisfied later by the CPU
            self.assertIn("bounded unsatisfied 73 later_cpu 66 dropped 6 pending 0\n", done.stdout)
            # six of the 70 found the list full and are counted as dropped; a wait on a word nobody signals is pending
            self.assertIn("unresolved unsatisfied 74 pending 1 dropped 6\n", done.stdout)
            self.assertIn("stat| waitclasses 78 by a GPU signal in order 2, by a CPU signal before 1, already satisfied 1, "
                          "unsatisfied 74 (later by GPU 1, later by CPU 66, not satisfied by the end 1, dropped from the "
                          "full list 6)\n", done.stdout)
            # the pool rows tell pools made from pools live: the case's 4 MiB pool, and a second one (64 KiB, a window range, no
            # GPU access) that it made and ended
            self.assertIn("stat| pools 2 memory pools made 2, live 1 (most at once 2), 4194304 bytes live (most 4259840); "
                          "made with GPU_NO_ACCESS 1; GPU address window in use 0 bytes (most 65536 of 68719476736)\n",
                          done.stdout)
            self.assertIn("stat| pools-cpu 1 made 1, live 1 (most at once 1), 4194304 bytes live (most 4194304)\n", done.stdout)
            self.assertIn("stat| pools-cpu-no-access 1 made 1, live 0 (most at once 1), 0 bytes live (most 65536)\n",
                          done.stdout)
            lines = [line for line in done.stderr.splitlines() if "nvn: WaitEvent at " in line]
            self.assertEqual(len(lines), 5, done.stderr)
            # each line names the word's last writer: a GPU signal, a CPU signal or nothing yet (its own value)
            self.assertRegex(lines[0], r"\(main\+0x1000\) on the event word \S+ \(equal to 3, the word holds 3\) is "
                                       r"satisfied at its place in submission order \(the word was last written by a GPU "
                                       r"signal\)$")
            self.assertRegex(lines[1], r"\(main\+0x2000\) on the event word \S+ \(equal to 9, the word holds 3\) is NOT "
                                       r"satisfied at its place in submission order \(the word was last written by a GPU "
                                       r"signal\)$")
            self.assertRegex(lines[2], r"\(main\+0x3000\) on the event word \S+ \(equal to 7, the word holds 0\) is NOT "
                                       r"satisfied at its place in submission order \(the word was last written by "
                                       r"nothing yet\)$")
            self.assertRegex(lines[3], r"\(main\+0x4000\) .* is NOT satisfied at its place in submission order \(the "
                                       r"word was last written by nothing yet\)$")
            self.assertRegex(lines[4], r"\(main\+0x5000\) .* is NOT satisfied at its place in submission order")

    def test_every_event_word_is_remembered_and_only_a_four_byte_write_is_a_signal(self) -> None:
        for done in self.assert_passes("event_words"):
            # 300 words (past a table of 256): the even ones signalled by the GPU, the odd ones by the CPU, none lost
            self.assertIn("event_words gpu 150 cpu 150 initial 0 unsatisfied 0\n", done.stdout)
            # a write of 8 bytes is not a signal (the wait is satisfied by the word's own value), one of 4 bytes is
            self.assertIn("event_sizes gpu 150 cpu 151 initial 1\n", done.stdout)
            # the first wait at each site is logged with the word's last writer: the even words' site (a GPU signal), the odd
            # words' (a CPU signal), and the 8-byte write's word (nothing: that write is no signal)
            lines = [line for line in done.stderr.splitlines() if "nvn: WaitEvent at " in line]
            self.assertEqual(len(lines), 3, done.stderr)
            self.assertRegex(lines[0], r"\(main\+0x1000\) .* is satisfied at its place in submission order \(the word was "
                                       r"last written by a GPU signal\)$")
            self.assertRegex(lines[1], r"\(main\+0x1100\) .* is satisfied at its place in submission order \(the word was "
                                       r"last written by a CPU signal\)$")
            self.assertRegex(lines[2], r"\(main\+0x2000\) .* is satisfied at its place in submission order \(the word was "
                                       r"last written by nothing yet\)$")

    def test_what_the_front_end_cannot_honour_stops_by_name(self) -> None:
        for case, code, text in FAILING:
            for level in nvn.LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, code, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertIn(FAILING_MORE.get(case, ""), done.stderr)
                    self.assertEqual(done.stderr.count("FATAL"), 1, done.stderr)

    def test_default_run_logs_its_presents_and_a_gap_stays_fatal(self) -> None:
        for done in self.assert_passes("census_default", OOT2_CENSUS="0"):
            lines = re.findall(r"^\[rt\] nvn: present (\d+), ", done.stderr, re.M)
            self.assertEqual(lines, ["1", "60", "120"], done.stderr)
            self.assertRegex(done.stderr, r"(?m)^\[rt\] nvn: present 1, 0\.00 s after the first$")
            # the seconds since present 1, which the case delayed by 30 ms: at least 0.03 (a unit error is far under),
            # under 5 (a loaded machine may be slow; a unit error the other way is far over)
            secs = r"(?:0\.(?:0[3-9]|[1-9]\d)|[1-4]\.\d\d)"
            self.assertRegex(done.stderr, rf"(?m)^\[rt\] nvn: present 60, {secs} s after the first; a present: 0\.0 SetRenderTargets, 0\.0 DispatchCompute, 0\.0 DrawElementsInstanced$")
            self.assertRegex(done.stderr, rf"(?m)^\[rt\] nvn: present 120, {secs} s after the first; a present: 0\.0 SetRenderTargets, 0\.0 DispatchCompute, 0\.0 DrawElementsInstanced$")
            # the intervals the game asked for over each window: one line a window, marked when not 1 and no hold
            # this case presents at interval 0, which is not 1, so every window is marked
            for seq in ("60", "120"):
                self.assertRegex(done.stderr, rf"(?m)^\[rt\] nvn: present {seq} intervals over 60 presents: requested 0 x60, "
                                              r"held 0 x60 \(NOT every interval 1 with no hold\)$")
            self.assertNotIn("nvn: title:", done.stderr)   # no movie frame in this case
            self.assertNotIn("census mode", done.stderr)
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "census_gap_default")
                self.assertEqual(done.returncode, nvn.EXIT_UNIMPLEMENTED, done.stderr)
                self.assertIn("FATAL: unimplemented nvn: nvnQueueSetDebugLabel", done.stderr)
                self.assertNotIn("census mode", done.stderr)

    def test_the_movie_frame_says_the_title_once_and_each_window_carries_the_draws(self) -> None:
        for done in self.assert_passes("title_movie", OOT2_CENSUS="0"):
            rows = {int(m[1]): m[2] for m in re.finditer(
                r"(?m)^\[rt\] nvn: present (\d+), [\d.]+ s after the first; a present: (.*)$", done.stderr)}
            self.assertEqual(sorted(rows), [60, 120, 180, 240, 300], done.stderr)
            self.assertEqual(rows[60], "10.0 SetRenderTargets, 14.0 DispatchCompute, 0.0 DrawElementsInstanced")
            self.assertEqual(rows[180], "10.0 SetRenderTargets, 14.0 DispatchCompute, 0.0 DrawElementsInstanced")
            # the menu's draws step in over the movie: two a present from present 181 on
            self.assertEqual(rows[240], "10.0 SetRenderTargets, 14.0 DispatchCompute, 2.0 DrawElementsInstanced")
            titles = re.findall(r"(?m)^\[rt\] nvn: title: .*$", done.stderr)
            self.assertEqual(titles, ["[rt] nvn: title: the movie's frame (14 dispatches and 10 SetRenderTargets a present) "
                                      "in two windows in a row, at present 120"], done.stderr)

    def test_a_frame_short_of_the_movie_or_a_lone_movie_window_says_no_title(self) -> None:
        for case in ("title_short_dispatch", "title_short_target", "title_gap"):
            for done in self.assert_passes(case, OOT2_CENSUS="0"):
                self.assertIn("nvn: present 60, ", done.stderr)
                self.assertNotIn("nvn: title:", done.stderr, case)
        for done in self.assert_passes("title_short_dispatch", OOT2_CENSUS="0"):
            self.assertIn("a present: 10.0 SetRenderTargets, 13.0 DispatchCompute, 0.0 DrawElementsInstanced", done.stderr)
        for done in self.assert_passes("title_gap", OOT2_CENSUS="0"):   # the movie, a window without it, the movie again
            rows = re.findall(r"(?m)^\[rt\] nvn: present (\d+), [\d.]+ s after the first; a present: (\S+) SetRenderTargets",
                              done.stderr)
            self.assertEqual(rows, [("60", "10.0"), ("120", "0.0"), ("180", "10.0")], done.stderr)

    def test_census_mode_passes_a_gap_once_and_counts_calls_per_frame(self) -> None:
        for done in self.assert_passes("census_mode", OOT2_CENSUS="1"):
            self.assertIn("[census mode] OOT2_CENSUS=1", done.stderr)
            self.assertRegex(done.stderr, r"(?m)^\[rt\] \[census mode\] OOT2_CENSUS=1: .*; this is not a default run$")
            gaps = re.findall(r"^\[rt\] \[census mode\] unbodied nvn: (\w+), answered 0, at ", done.stderr, re.M)
            self.assertEqual(gaps, ["nvnQueueSetDebugLabel", "nvnWindowGetNumTextures"], done.stderr)
            rows = [line[len("census| "):] for line in done.stdout.splitlines() if line.startswith("census| ")]
            self.assertTrue(rows[0].startswith("#census-mode OOT2_CENSUS=1"), rows[0])
            windows = {m[1]: float(m[2]) for m in re.finditer(r"^#window (\w+) presents \d+-\d+ seconds (\d+\.\d+)$",
                                                              "\n".join(rows), re.M)}
            self.assertEqual(sorted(windows), ["tail", "w001", "w002"], rows)
            self.assertIn("#window w001 presents 1-60 seconds ", "\n".join(rows))
            self.assertIn("#window w002 presents 60-120 seconds ", "\n".join(rows))
            self.assertIn("#window tail presents 120-130 seconds ", "\n".join(rows))
            # seconds, not microseconds or milliseconds: the case waits 30 ms inside w001 (presents 1 to 60) and 30 ms
            # before the file is written (the tail); a loaded machine may be slower, never 5 s slower
            self.assertTrue(0.03 <= windows["w001"] < 5, windows)
            self.assertTrue(0.0 <= windows["w002"] < 5, windows)
            self.assertTrue(0.03 <= windows["tail"] < 5, windows)
            table = {}
            for line in rows:
                f = line.split("\t")
                if len(f) == 6 and f[0] != "phase":
                    table[(f[0], f[3])] = (int(f[1]), f[4], int(f[5]))
            # only NVN2 entry points, and only the ones called in the window: the case also calls a host function
            # that is none of them, and most NVN2 functions are not called in a given window
            self.assertTrue(all(name.startswith(("nvn", "replay:")) for _, name in table), sorted(table)[:5])   # replay rows
            self.assertTrue(all(calls > 0 for _, _, calls in table.values()))
            self.assertNotIn(("tail", "census_test_other"), table)
            self.assertNotIn(("w002", "nvnQueueSetDebugLabel"), table)
            # present 1 counts itself: the boot row holds the one call; a window of 59 or 60 presents has one a frame
            self.assertEqual(table[("boot", "nvnQueuePresentTexture")], (0, "1", 1))
            self.assertEqual(table[("w001", "nvnQueuePresentTexture")], (59, "1.000", 59))
            self.assertEqual(table[("w002", "nvnQueuePresentTexture")], (60, "1.000", 60))
            self.assertEqual(table[("tail", "nvnQueuePresentTexture")], (10, "1.000", 10))
            self.assertEqual(table[("all", "nvnQueuePresentTexture")], (129, "1.000", 129))
            self.assertEqual(table[("all", "nvnWindowAcquireTexture")][2], 129)
            # the front end's own counters, as census rows: 130 recordings begun through the entry point, 130 presents
            stats = [line[len("stat| "):] for line in done.stdout.splitlines() if line.startswith("stat| ")]
            self.assertEqual([x.split(" ", 2)[:2] for x in stats],
                             [["recordings", "130"], ["callbacks", "0"], ["waitevents", "0"], ["waitclasses", "0"], ["waitqueues", "0"],
                              ["pools", "1"], ["pools-cpu", "1"], ["pools-cpu-no-access", "0"], ["pools-physical", "0"],
                              ["pools-virtual", "0"], ["presents", "130"]], stats)
            # the case's one memory pool, 4 MiB the CPU can see; no window range in use
            self.assertEqual(stats[5], "pools 1 memory pools made 1, live 1 (most at once 1), 4194304 bytes live (most "
                                       "4194304); made with GPU_NO_ACCESS 0; GPU address window in use 0 bytes (most 0 of "
                                       "68719476736)")
            self.assertEqual(stats[6], "pools-cpu 1 made 1, live 1 (most at once 1), 4194304 bytes live (most 4194304)")
            self.assertEqual(stats[9], "pools-virtual 0 made 0, live 0 (most at once 0), 0 bytes live (most 0)")
            self.assertIn("inline 0, slow 130, ended with a handle 130, ended empty 0", stats[0])
            # the table is read live: the two gap calls after the last present are in its tail and in the whole
            self.assertEqual(table[("tail", "nvnQueueSetDebugLabel")][2], 2)
            self.assertEqual(table[("all", "nvnQueueSetDebugLabel")][2], 2)

    def test_census_mode_a_run_ending_on_a_window_boundary_has_no_tail_window(self) -> None:
        for done in self.assert_passes("census_boundary", OOT2_CENSUS="1"):
            rows = [line[len("census| "):] for line in done.stdout.splitlines() if line.startswith("census| ")]
            text = "\n".join(rows)
            self.assertIn("#window w002 presents 60-120 seconds ", text)
            self.assertNotIn("#window tail", text)
            self.assertFalse([r for r in rows if r.startswith("tail\t")], text)

    def test_census_mode_keeps_1024_snapshots_and_says_so_once(self) -> None:
        for done in self.assert_passes("census_cap", OOT2_CENSUS="1"):
            self.assertIn("windows kept 1023\n", done.stdout)
            self.assertEqual(done.stderr.count("[census mode] nvn: 1024 snapshots kept; later windows are not recorded"), 1,
                             done.stderr[-2000:])

    def test_census_mode_rewrites_its_census_every_600_presents(self) -> None:
        self.assert_passes("census_flush", OOT2_CENSUS="1")

    def test_census_mode_ends_at_its_frame_limit_with_its_own_exit_code(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "census_limit", OOT2_CENSUS="1", OOT2_CENSUS_FRAMES="7")
                self.assertEqual(done.returncode, 73, done.stdout + done.stderr)
                self.assertIn("[census mode] ended at its frame limit, present 7", done.stderr)
                self.assertIn("this is not a default run; it ends at its frame limit\n", done.stderr)
                self.assertNotIn("not reached", done.stdout)
        # without census mode the limit is not read
        done = self.run_case(nvn.LEVELS[0], "census_default", OOT2_CENSUS_FRAMES="7")
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_unknown_case_is_refused(self) -> None:
        done = self.run_case(nvn.LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
