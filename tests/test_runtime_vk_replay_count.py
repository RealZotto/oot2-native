# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The replay's workload counters change nothing the replay does (runtime/include/oot2/replay_count.h).

The cases of tests/runtime/vk_replay_test.c (the replay's own contract tests, with the real recorder writing every packet as a
line, no device) run three times each: counters off, counters off again, and counters on (OOT2_REPLAY_COUNTERS=1). What a case
shows is its standard output (the packets, in order, with the scratch ranges, descriptor writes and pipeline handles the
replay chose), its exit code and its standard error less the counters' own lines. The run with the counters on must equal the
first run with them off exactly, as the second run with them off does: that one proves the output holds nothing that varies
from run to run, so that an unequal pair can only be the counters.

Where the address space is randomized (the Linux runners) the packets hold host addresses that differ between two runs of the
same program, so the two runs with the counters off already differ and there is no repeatable output to compare against. A
case that shows this on a system other than Windows is skipped, named as such; on Windows the output is repeatable and the
same difference is a failure.

What this cannot see. The Vulkan calls themselves are not made (no device: the recorder's calls print), so a change that
only reached a call's arguments through memory the recorder does not print is not seen; the cases are the replay's contract
tests, not a game's stream: no run of the real game is compared; the replay's statistics (VkrStats) are not printed by the
cases, only their effects on the packets; the ledger's rows are not compared (the cases leave the ledger folder unset);
and what a counter costs in time, which nothing here measures.

The counter-run: a counter that also clears a bank cache when it counts changes the packets, and the first test goes red.

Run: python -m unittest discover -s tests -p test_runtime_vk_replay_count.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest

from test_runtime_gpu import case_env  # sets the path of tools/
from test_runtime_vk_replay import ReplayBase, tearDownModule  # noqa: F401  (the module's teardown frees the program)

ENV = "OOT2_REPLAY_COUNTERS"

# Passing cases that draw, dispatch, copy, write storage, report and render in each way the replay knows, and the stopping
# cases that end in a named stop.
PASSING = [("graphics",), ("depth",), ("compute",), ("indirect",), ("draw_indirect",), ("features",), ("mesh",),
           ("mesh_restart",), ("storage_image",), ("transfers",), ("skip",), ("skip_render_off",), ("conflict_read_first",),
           ("conflict_write_first",), ("raw_first",), ("full_rate",), ("reports",),
           ("graphics", "--mutant", "nosnapshot"), ("conflict_read_first", "--mutant", "nodrop")]
STOPPING = [("dual_two_targets",), ("broken",)]


def without_counter_lines(text: str) -> str:
    return "".join(line + "\n" for line in text.splitlines() if not line.startswith("[rt] rc "))


class CountersChangeNothingTest(ReplayBase):
    def shown(self, args: tuple[str, ...], on: bool) -> tuple[int, str, str, str]:
        env = {k: v for k, v in case_env().items() if k != ENV}
        if on:
            env[ENV] = "1"
        done = subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=60, env=env)
        return done.returncode, done.stdout, without_counter_lines(done.stderr), done.stderr

    def compare(self, args: tuple[str, ...]) -> tuple[str, str]:
        off1, off2, on = self.shown(args, False), self.shown(args, False), self.shown(args, True)
        if off1[:3] != off2[:3] and os.name != "nt":
            raise unittest.SkipTest(f"{args}: the output varies from run to run here (randomized host addresses)")
        self.assertEqual(off1[:3], off2[:3], f"{args}: two runs with the counters off differ (the output is not repeatable)")
        self.assertEqual(off1[:3], on[:3], f"{args}: the counters changed what the replay did")
        self.assertEqual(off1[3].count("[rt] rc "), 0, args)
        return off1[1], on[3]

    def test_the_packets_exit_codes_and_messages_are_the_same_with_the_counters_on(self) -> None:
        packets, compared = 0, 0
        for args in PASSING:
            with self.subTest(args=args):
                out, rc = self.compare(args)
                compared += 1
                packets += out.count("pkt ")
                self.assertIn(f"ok {args[0]}", out)
                self.assertIn("[rt] rc kind=exit group=sub", rc)   # the counters were on, and counted something
        if compared == len(PASSING):
            self.assertGreater(packets, 500)   # the comparison was over real packet streams

    def test_the_same_holds_for_a_run_that_ends_in_a_stop_and_the_stop_prints_the_totals(self) -> None:
        for args in STOPPING:
            with self.subTest(args=args):
                _, rc = self.compare(args)
                self.assertIn("[rt] rc kind=exit group=sub", rc)

    def test_the_counters_counted_the_draws_of_a_case_they_were_on_for(self) -> None:
        env = {k: v for k, v in case_env().items() if k != ENV}
        env[ENV] = "1"
        done = subprocess.run([str(self.exe), "graphics"], capture_output=True, text=True, timeout=60, env=env)
        flat: dict[str, str] = {}
        for line in done.stderr.splitlines():
            if line.startswith("[rt] rc kind=exit "):
                flat.update(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        self.assertEqual(int(flat["vkr_frames"]), int(flat["sub_n"]))
        self.assertGreater(int(flat["draw_classic"]), 0)
        self.assertEqual(int(flat["draw_classic"]) + int(flat["draw_mesh"]), int(flat["vkr_draws"]))


class StageTimerTest(ReplayBase):
    """The replay stage timer (group "stages" of OOT2_REPLAY_COUNTERS; runtime/gpu/vk_replay_count.c): the calls of each class
    of operation are counted exactly, the clock is read for a sample of the classes that run thousands of times, and the line
    prints what the clock reads cost and the counters that say why more draws are replayed than the game records. The
    stand-in device has no real time to measure, so these tests check counts, sample counts and the identities between them,
    never a duration (apart from the ordering of an interval and the intervals inside it, which holds on any monotone clock)."""

    def stages(self, case: str, window: bool = False) -> dict[str, int]:
        env = {k: v for k, v in case_env().items() if k != ENV}
        env[ENV] = "1"
        done = subprocess.run([str(self.exe), case], capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        line = [x for x in done.stderr.splitlines() if "kind=exit group=stages" in x]
        self.assertEqual(len(line), 1, done.stderr)
        return {k: int(v) for k, v in re.findall(r" (\w+)=(\d+)", line[0])}

    def test_a_recording_submitted_three_times_makes_three_draws_for_one_recorded(self) -> None:
        s = self.stages("stage_repeat")
        self.assertEqual((s["st_subs"], s["st_handles"], s["st_handles_repeat"]), (4, 4, 2))     # 3 of the first, 1 of the second
        self.assertEqual((s["st_handle_ops"], s["st_handle_ops_repeat"]), (4, 2))
        self.assertEqual((s["st_ops_draw"], s["st_ops_draw_idx"], s["st_draw_instances"]), (3, 1, 3 * 1 + 5))
        self.assertEqual(s["st_draw_n"], 4)                                                    # replayed draws: 3 + 1
        self.assertEqual(s["st_walk_n"], 4)

    def test_every_class_counts_its_calls_exactly(self) -> None:
        s = self.stages("graphics")
        self.assertEqual(s["st_draw_n"], 4)                  # a draw, an indexed one, an indirect one, and one in the second frame
        self.assertEqual(s["st_pipeline_n"], 4)              # one lookup a draw
        self.assertEqual(s["st_descriptors_n"], 8)           # two stages a draw
        self.assertEqual(s["st_vertex_n"], 4)
        self.assertEqual(s["st_dynamic_state_n"], 4)
        self.assertEqual((s["st_clear_n"], s["st_barrier_n"], s["st_dispatch_n"]), (1, 1, 0))
        self.assertGreaterEqual(s["st_pass_begin_n"], 1)
        self.assertEqual(s["st_pass_begin_n"], s["st_pass_end_n"])
        s = self.stages("compute")
        self.assertEqual((s["st_dispatch_n"], s["st_draw_n"], s["st_ops_dispatch"], s["st_ops_dispatch_ind"]), (2, 0, 1, 1))
        s = self.stages("transfers")
        self.assertGreater(s["st_copy_n"], 0)
        s = self.stages("reports")
        self.assertGreater(s["st_query_n"], 0)

    def test_the_clock_is_read_for_a_sample_and_the_line_says_how_many_reads_it_made(self) -> None:
        for case in ("graphics", "stage_repeat", "features", "transfers"):
            with self.subTest(case=case):
                s = self.stages(case)
                names = [k[3:-8] for k in s if k.startswith("st_") and k.endswith("_samples")]
                self.assertGreaterEqual(len(names), 17)
                self.assertEqual(s["st_clock_reads"], 2 * sum(s[f"st_{n}_samples"] for n in names))   # a sample is two reads
                for n, every in (("walk", 1), ("draw", 1), ("resolve", 16), ("snapshot", 16), ("uniform", 16), ("ledger", 32),
                                 ("pipeline", 4), ("descriptors", 4)):
                    calls = s[f"st_{n}_n"]
                    self.assertEqual(s[f"st_{n}_samples"], (calls + every - 1) // every, (case, n))
                self.assertGreater(s["st_clock_ns_x1000"], 0)
                self.assertLessEqual(s["st_overhead_ns"], s["st_clock_reads"] * s["st_clock_ns_x1000"] // 1000 + 1)

    def test_an_interval_holds_the_intervals_inside_it(self) -> None:
        s = self.stages("graphics")
        inner = sum(s[f"st_{n}_ns"] for n in ("draw", "dispatch", "clear", "copy", "query", "barrier"))
        self.assertGreaterEqual(s["st_walk_ns"], inner)

    def test_with_the_counters_off_there_is_no_stages_line(self) -> None:
        env = {k: v for k, v in case_env().items() if k != ENV}
        done = subprocess.run([str(self.exe), "graphics"], capture_output=True, text=True, timeout=60, env=env)
        self.assertNotIn("group=stages", done.stderr)
        self.assertNotIn("st_draw_n", done.stderr)


if __name__ == "__main__":
    unittest.main()
