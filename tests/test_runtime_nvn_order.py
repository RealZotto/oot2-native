# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the order a back end replays the queues' records in, so that
a WaitEvent is answered (oot2/nvn_order.h, gfx.h, architecture section 5).

Builds tests/runtime/nvn_order_test.c with runtime/core and runtime/nvn at -O1
and -O2, as tests/test_runtime_nvn_queue.py builds its harness, and runs each
case in its own process. The recordings are synthetic (OP_BEGIN, markers,
OP_SIGNAL_EVENT, OP_WAIT_EVENT, OP_END from nvn_obj.h, written as bytes), the
event words are the test's own memory, and the orders below are written here by
hand, never computed with the engine's rule. Two kinds of case: the order engine
alone with a sink that logs what it is shown, and the null back end through its
table (submit with a queue, guest_written for the CPU's signal, fence_wait,
present). No device: continuous integration runs all of them.

- a wait released by a later submission of another queue: the queue's records
  before the wait are shown, the other queue's signal is shown, then the wait
  and the rest.
- a wait released by a CPU signal made after the submit: the queue's later
  submissions wait behind the wait, other queues are not held, a write that is
  not the value releases nothing, and the signal from another thread releases
  a thread blocked on the fence value.
- a release that cascades through a queue the loop has passed, and two waits in a
  row released one at a time.
- ADD accumulates across queues and from the CPU, GEQUAL_WRAP compares across the
  wrap, and the engine's copy of a word keeps the GPU ADDs not yet written when
  the CPU re-reads memory.
- a wait nothing releases is a named stop after the stated bound (1024 further
  submissions, or 10 s for a wait with no timeout, counted from the start of that
  wait and only for the queue it depends on), and a hung case is not a pass.
- a fence value is done when its own queue's work up to it is whole and written,
  whatever another queue is held at: a thread that finishes or waits on one queue
  is not held by a wait on another, whose release it makes next. Results are
  written as each queue completes, and a present waits for held work.
- the classes of the waits by queue pair.

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

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_order_test.c"
TIMEOUT = 120
EXIT_HUNG = 3   # nvn_order_test.c's watchdog: the case had not ended after 20 s


class OrderTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvnorder-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_order_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}
        try:
            done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        if done.returncode == EXIT_HUNG:
            self.fail(f"{case} at {level} hung (the case's watchdog):\n{done.stdout}{done.stderr}")
        return done

    def assert_passes(self, case: str) -> list[subprocess.CompletedProcess]:
        runs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                runs.append(done)
        return runs

    def assert_stops(self, case: str, first_line: str, text: str, recording: int = 0) -> list[re.Match[str]]:
        found = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case)
                self.assertEqual(done.returncode, nvn.EXIT_FATAL, done.stdout + done.stderr)
                self.assertEqual(done.stdout.splitlines(), [first_line], done.stdout)   # and never "not stopped"
                m = re.search(r"FATAL: gfx: null back end: the WaitEvent of queue 1 at 0x1000 \(submission 1, "
                              rf"recording {recording}\) on the event word \w+ \(equal to 1, the word holds 0\) "
                              r"was released by no signal in (\d+) submissions and (\d+) ms: " + re.escape(text) + r"$", done.stderr, re.M)
                self.assertIsNotNone(m, done.stderr)
                found.append(m)
        return found

    # (a) ---------------------------------------------------------------------------------------------------------------

    def test_a_wait_is_released_by_a_later_submission_of_another_queue(self) -> None:
        for done in self.assert_passes("order_a"):
            lines = done.stdout.splitlines()
            # queue 1 is shown up to its wait; queue 2's records, its signal among them, come next; then the wait and
            # the rest of queue 1. The submission of queue 2 is whole first, queue 1's after it.
            self.assertEqual(lines[:3], ["after-wait q1:m1", "after-signal q2:m3 q2:S q2:m4 q1:W q1:m2", "whole w2 w1"])
            # the wait was held for one submission (the signalling one) and released once
            self.assertEqual(lines[3], "longest-hold 1 parks 1 releases 1")
        for done in self.assert_passes("null_hold"):
            # the same through the null back end, with the fence value each record belongs to
            self.assertEqual(done.stdout.splitlines()[:2], ["first f1:m1", "second f2:S f2:m3 f1:W f1:m2 f1:S"])

    # (b) ---------------------------------------------------------------------------------------------------------------

    def test_a_wait_is_released_by_a_cpu_signal_made_after_the_submit(self) -> None:
        for done in self.assert_passes("order_b"):
            # queue 1 holds at its wait with its later submission behind it, queue 2 goes on; a write of 4 is not the
            # value; the write of 5 releases the wait and then the submission behind it, in the queue's order
            self.assertEqual(done.stdout.splitlines()[:4], ["before q1:m1 q2:m4", "whole-before w3",
                                                            "after-cpu q1:W q1:m2 q1:m3", "whole-after w1 w2"])
        for done in self.assert_passes("null_cpu"):
            # a thread blocked on fence value 3 (behind the wait, on the held queue) is released by the CPU's signal from
            # another thread
            self.assertIn("order f1:m1 f2:m3 f1:W f1:m2 f3:m5\n", done.stdout)

    def test_a_release_cascades_and_a_chain_of_waits_is_released_one_at_a_time(self) -> None:
        for done in self.assert_passes("order_cascade"):
            self.assertEqual(done.stdout.splitlines()[:3], [
                "cascade q2:S q1:W q1:m1 q1:S q1:m2 q3:W q3:m9",
                "chain-first q2:S q4:W q4:m7",
                "chain-second q2:S q4:W q4:m8"])

    def test_a_wait_is_released_by_a_value_the_same_submission_then_overwrites(self) -> None:
        for done in self.assert_passes("order_transient"):
            # queue 3 writes E=5, F=8, E=6 in one submission: the wait for E equal to 5 saw 5 and goes on, the wait for F
            # equal to 7 saw nothing it wanted and stays held
            self.assertEqual(done.stdout.splitlines()[0], "transient q3:S q3:S q3:S q1:W q1:m1")

    # (c) ---------------------------------------------------------------------------------------------------------------

    def test_a_wait_nothing_releases_is_a_named_stop_after_the_bound_and_never_a_hang(self) -> None:
        # the engine: reported at the 1025th further submission, not at the 1024th, and at more than 10 s, not at 10 s
        for done in self.assert_passes("order_c"):
            self.assertEqual(done.stdout.splitlines()[:2],
                             ["stalled queue 1 fence 1 index 0 mode 0 value 1 seen 0 since 1025", "shown 1025"])
        # the null back end stops by name at the 9th submission past a bound of 8, with the 8th still allowed
        for m in self.assert_stops("null_stop_submits", "within", "the bound on submissions"):
            self.assertEqual(m[1], "9")
        # a wait for a fence value of the held queue, and a finish of it, with no timeout: a stop after the time bound
        # (100 ms), not a hang
        for case, first in (("null_stop_time", "waiting"), ("null_stop_finish", "finishing")):
            for m in self.assert_stops(case, first, "a wait with no timeout for work held behind it"):
                self.assertEqual(m[1], "0")
                self.assertTrue(100 <= int(m[2]) < 2000, m[0])

    def test_the_default_bounds_stop_at_the_1025th_submission_and_after_ten_seconds(self) -> None:
        # no gfx_null_wait_bounds call: WO_BOUND_SUBMITS (1024 further submissions are within, the 1025th is a stop) and
        # WO_STALL_NS (a wait with no timeout for the held queue's work stops at 10 s, not before and not far after)
        for m in self.assert_stops("null_stop_default_submits", "within", "the bound on submissions"):
            self.assertEqual(m[1], "1025")
        for m in self.assert_stops("null_stop_default_time", "waiting", "a wait with no timeout for work held behind it"):
            self.assertEqual(m[1], "0")
            self.assertTrue(10000 <= int(m[2]) < 12000, m[0])

    def test_the_stop_names_the_recording_the_wait_is_in(self) -> None:
        # a submission of two recordings with the wait in the second: "recording 1"
        for m in self.assert_stops("null_stop_recording", "within", "the bound on submissions", recording=1):
            self.assertEqual(m[1], "9")

    def test_a_wait_with_a_timeout_answers_zero_and_is_not_a_stop(self) -> None:
        self.assert_passes("null_timeout")

    # (d) ---------------------------------------------------------------------------------------------------------------

    def test_a_fence_value_is_done_when_its_queues_work_up_to_it_is_whole_and_written(self) -> None:
        for done in self.assert_passes("order_d"):
            lines = done.stdout.splitlines()
            # queue 2's second submission (fence value 4) is whole at once, and the prefix of whole submissions stays at 3
            # until the wait of fence value 1 is released by fence value 3; the records of each queue are in that
            # queue's order
            self.assertEqual(lines[:6], ["through 0", "through 0", "through 3", "through 4",
                                         "order q1:m1 q2:m3 q3:m4 q3:S q1:W q1:m2 q2:m5", "whole w2 w3 w1 w4"])
        for done in self.assert_passes("null_hold"):
            # replay put fence value 2's signal before fence value 1's. The hook that completes held work completes
            # what it is asked for, up to a fence value: up to 1 is fence value 1 alone (F), up to 2 adds E. The order
            # among the submissions it completes together is replay order (test_results_are_written_in_the_order_...)
            self.assertEqual(done.stdout.splitlines()[2:4], ["after-1 E 0 F 7 done 1", "after-2 E 1 F 7 done 2"])

    def test_results_are_written_in_the_order_replay_finished_the_submissions(self) -> None:
        # queue 1 (fence value 1) waits for E and writes X = 7, queue 2 (fence value 2) writes X = 3 then signals E: the
        # console ends at 7, with completion at once and with a completion held and then asked for up to fence value 4
        for done in self.assert_passes("null_result_order"):
            self.assertEqual(done.stdout.splitlines()[:2], ["released x 7 e 1 done 2", "held x 7 e 1 done 4"])

    def test_what_a_submission_kept_before_a_wait_is_written_when_its_queue_parks(self) -> None:
        # queue 1 writes X = 1 then waits for an event nobody signals; queue 2 waits for X and writes Y: both are
        # visible, and with completion held nothing is written at the park
        for done in self.assert_passes("null_park_writes"):
            self.assertEqual(done.stdout.splitlines()[:2], ["x 1 y 1 parked 1 done 0", "held x 0 parked 2"])

    def test_finish_of_every_queue_waits_for_all_of_them_whichever_is_released_first(self) -> None:
        for done in self.assert_passes("null_finish_all"):
            self.assertEqual(done.stdout.splitlines()[0], "finish(0) parked 0 done 2")

    def test_a_wait_reached_after_a_release_is_counted_apart_from_the_signals_before_it(self) -> None:
        # queue 2's wait on a word the CPU wrote is "by a CPU signal before"; the three waits behind queue 1's parked wait
        # (two in its recording, one in the submission queued behind) are judged after the release and are not
        for done in self.assert_passes("null_after_release"):
            self.assertEqual(done.stdout.splitlines()[:3], [
                "held cpu 1 initial 0 after_release 0 unsatisfied 1 later_cpu 0 pending 1",
                "released cpu 1 initial 0 after_release 3 unsatisfied 1 later_cpu 1 pending 0",
                "waits 5 unsatisfied 1"])

    def test_a_fence_value_behind_a_held_wait_of_its_queue_is_not_done_but_another_queues_is(self) -> None:
        for done in self.assert_passes("null_cpu"):
            # fence value 2 (another queue) was replayed whole at once and is done, fence value 1 (held) and 3 (behind it
            # on the same queue) are not; the prefix fence_done() reports stays 0 while fence value 1 is held
            self.assertIn("while-held done 0 whole-through 0\n", done.stdout)

    def test_a_thread_that_waits_on_one_queue_is_not_held_by_a_wait_on_another(self) -> None:
        # queue 1 is parked until a CPU signal that the same thread makes after waiting on queue 2: finish, a wait on a
        # fence value and the CPU's view of queue 2's signal do not depend on queue 1 (a stop after 100 ms otherwise)
        for done in self.assert_passes("null_cross_queue"):
            self.assertIn("queue-2-done\n", done.stdout)

    def test_a_wait_that_begins_after_the_park_is_timed_from_its_own_start(self) -> None:
        self.assert_passes("null_late_wait")

    def test_a_parked_queue_the_wait_does_not_depend_on_is_no_stop(self) -> None:
        self.assert_passes("null_unrelated_park")

    def test_add_accumulates_and_gequal_compares_across_the_wrap(self) -> None:
        for done in self.assert_passes("order_add"):
            self.assertEqual(done.stdout.splitlines()[:5], [
                "add-first q2:S",
                "add-second q3:S q1:W q1:m1",
                "add-cpu q5:S q4:W q4:m2",
                "wrap-add q7:S q6:W q6:m3",
                "wrap-place q8:W q8:m4"])

    def test_the_engines_word_keeps_gpu_adds_not_yet_written_and_ignores_words_not_met(self) -> None:
        for done in self.assert_passes("order_pending"):
            self.assertEqual(done.stdout.splitlines()[:4], ["pending q1:S q2:W q2:m1", "written q3:W q3:m2", "reset q5:W q5:m4",
                                                            "unmet q4:W q4:m3"])

    def test_an_event_initialised_again_drops_pending_adds_and_releases_a_wait_the_word_now_satisfies(self) -> None:
        # a GPU ADD replayed and not written when the event is initialised again: the write later must not take the
        # engine's count below zero (a CPU write of 5 is read as 5); a wait parked on the word is judged again
        for done in self.assert_passes("order_reset_pending"):
            self.assertEqual(done.stdout.splitlines()[0], "reset-pending q1:S q2:W q2:m1")
        for done in self.assert_passes("order_reset_release"):
            self.assertEqual(done.stdout.splitlines()[:2], ["held q1:m1", "released q1:W q1:m2"])

    def test_the_table_of_event_words_keeps_its_words_when_it_grows(self) -> None:
        # 700 words: past 512 the 1024-slot table doubles, and a word met before is still the word met after
        for done in self.assert_passes("order_words"):
            self.assertEqual(done.stdout.splitlines()[0], "words 700 cap 2048")

    def test_a_submission_of_several_recordings_parks_in_the_one_with_the_wait(self) -> None:
        for done in self.assert_passes("order_recordings"):
            # three recordings, the wait in the second: the first and the head of the second are shown, a second submission
            # queues behind, two waits in the second recording are released one at a time, then the rest follows
            self.assertEqual(done.stdout.splitlines()[:4], [
                "parked-in-second q1:m1 q1:m2",
                "first-release q2:S q1:W q1:m3",
                "second-release q2:S q1:W q1:m4 q1:m5 q1:m6 q1:m7",
                "whole w3 w4 w1 w2"])

    def test_a_present_waits_for_the_work_submitted_before_it(self) -> None:
        for done in self.assert_passes("null_cpu"):
            # made while a queue is held: not counted; counted when the work before it is whole; one made with nothing
            # held is counted at once
            self.assertEqual([l for l in done.stdout.splitlines() if l.startswith("presents")],
                             ["presents-held 0", "presents-released 1", "presents-after 2"])

    def test_waits_are_counted_by_queue_pair_and_by_who_released_them(self) -> None:
        for done in self.assert_passes("null_classes"):
            self.assertEqual(done.stdout.splitlines()[:3], [
                "before-cpu gpu 4 (same 2 other 2) cpu 0 initial 1 unsatisfied 2 later_gpu 1 later_cpu 0 pending 1",
                "after-cpu gpu 4 (same 2 other 2) cpu 0 initial 1 unsatisfied 2 later_gpu 1 later_cpu 1 pending 0",
                "waits 7 unsatisfied 2"])


if __name__ == "__main__":
    unittest.main()
