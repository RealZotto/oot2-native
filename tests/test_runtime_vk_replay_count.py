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


if __name__ == "__main__":
    unittest.main()
