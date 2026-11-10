# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the replay's workload counters (runtime/include/oot2/replay_count.h, runtime/core/replay_count.c) through
tests/runtime/replay_count_test.c, which links runtime/core alone: the counters off do nothing, threads count into blocks of
their own and the sum is exact, a window prints the change over the window and the exit prints the totals, a hook's values
come out as changes, every counter is named once, no line passes the log's length, the environment variable
OOT2_REPLAY_COUNTERS sets the window (and refuses a value it does not know), and a stop prints the exit line.

The same counters read against the replay itself are in tests/test_runtime_vk_replay_cost.py (streams whose repeats are
known) and tests/test_runtime_vk_replay_count.py (the replay's packets with the counters on and off).

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set.

Run: python -m unittest discover -s tests -p test_runtime_replay_count.py -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

CORE = ROOT / "runtime" / "core"
TEST_SOURCE = ROOT / "tests" / "runtime" / "replay_count_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"
HEADER = ROOT / "runtime" / "include" / "oot2" / "replay_count.h"
MODULE_SIZE = 0x8000
ENV = "OOT2_REPLAY_COUNTERS"
EXIT_FATAL = 70
LOG_LINE = 2048   # fatal.c LINE_MAX_BYTES: a longer line is cut


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def rc_lines(stderr: str) -> list[dict]:
    out = []
    for line in stderr.splitlines():
        if line.startswith("[rt] rc "):
            out.append(dict(item.split("=", 1) for item in line[len("[rt] rc "):].split()))
    return out


def counter_names() -> list[str]:
    """The names in the header's RC_LIST, in order."""
    text = HEADER.read_text(encoding="utf-8")
    body = text[text.index("#define RC_LIST(X)"):text.index("enum RcId")]
    return re.findall(r'X\(\w+, "(\w+)"\)', body)


class CountersTest(unittest.TestCase):
    exe: Path

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-rccount-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        cls.exe = out / f"replay_count_test{'.exe' if os.name == 'nt' else ''}"
        cmd = [cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
               "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out), *sources, "-o", str(cls.exe)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build failed:\n{done.stdout}{done.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, case: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        full = {k: v for k, v in os.environ.items() if k != ENV}
        full.update(env or {})
        return subprocess.run([str(self.exe), case], capture_output=True, text=True, timeout=120, env=full)

    def passing(self, case: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        done = self.run_case(case, env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        return done

    def test_with_the_counters_off_a_site_does_nothing(self) -> None:
        done = self.passing("off")
        self.assertEqual(rc_lines(done.stderr), [])

    def test_threads_count_into_blocks_of_their_own_and_the_sum_is_exact(self) -> None:
        done = self.passing("threads")
        flat = {k: v for x in rc_lines(done.stderr) for k, v in x.items()}
        self.assertEqual(int(flat["draw_classic"]), 7 + 4 * 100000)
        self.assertEqual(int(flat["rc_threads"]), 5)   # the main thread and four workers
        self.assertEqual(int(flat["rc_threads_sharing_a_block"]), 0)

    def test_a_window_prints_the_change_and_the_exit_the_totals(self) -> None:
        lines = rc_lines(self.passing("windows").stderr)
        sub = [x for x in lines if x["group"] == "sub"]
        self.assertEqual([(x["kind"], x["present"], x["presents"], x["sub_n"]) for x in sub],
                         [("window", "10", "10", "3"), ("window", "20", "10", "4"), ("exit", "25", "25", "7")])
        hooked = [x for x in lines if x["group"] == "hooked"]
        self.assertEqual([(x["kind"], x["hook_value"]) for x in hooked], [("window", "100"), ("window", "150"), ("exit", "250")])

    def test_every_counter_is_named_once_and_no_line_is_too_long(self) -> None:
        names = counter_names()
        self.assertGreater(len(names), 50)
        self.assertEqual(len(names), len(set(names)))
        done = self.passing("lines")
        logged = [line for line in done.stderr.splitlines() if line.startswith("[rt] rc ")]
        self.assertTrue(all(len(line) < LOG_LINE - 16 for line in logged), max(len(x) for x in logged))
        seen = [k for x in rc_lines(done.stderr) for k in x if k in set(names)]
        self.assertEqual(sorted(seen), sorted(names))
        flat = {k: int(v) for x in rc_lines(done.stderr) for k, v in x.items() if k in set(names)}
        self.assertEqual([flat[n] for n in names], list(range(1, len(names) + 1)))   # each name is its own counter's value

    def test_the_environment_sets_the_window(self) -> None:
        def windows(value: str | None) -> list[str]:
            done = self.passing("present", {ENV: value} if value is not None else None)
            return [x["present"] for x in rc_lines(done.stderr) if x["kind"] == "window" and x["group"] == "sub"]
        self.assertEqual(windows("1"), ["600", "1200"])
        self.assertEqual(windows("on"), ["600", "1200"])
        self.assertEqual(windows("500"), ["500", "1000"])
        self.assertEqual(len(windows("7")), 1300 // 7)
        self.assertEqual(windows("0"), [])
        self.assertEqual(windows(None), [])

    def test_a_value_it_does_not_know_stops_the_run_naming_the_choices(self) -> None:
        done = self.run_case("present", {ENV: "sometimes"})
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("OOT2_REPLAY_COUNTERS", done.stderr)
        self.assertIn("sometimes", done.stderr)

    def test_a_series_prints_its_count_sum_median_p90_and_largest_for_each_window(self) -> None:
        series = [x for x in rc_lines(self.passing("series").stderr) if x["group"] == "series"]
        self.assertEqual([(x["kind"], x["present"]) for x in series], [("window", "10"), ("window", "20"), ("window", "30"),
                                                                       ("exit", "33")])

        def stats(x: dict, tail: str = "") -> tuple:
            return tuple(int(x[f"sample{tail}_{k}"]) for k in ("n", "sum", "p50", "p90", "max"))
        self.assertEqual(stats(series[0]), (10, 55, 5, 9, 10))          # the median of 1..10 is the 5th, the 90th the 9th
        self.assertEqual(stats(series[1]), (3, 600, 200, 300, 300))
        cap = 16384
        newest = range(6, cap + 6)                                       # 5 of the cap + 5 values fell out of the window
        self.assertEqual(stats(series[2]), (cap, sum(newest), 6 + (cap - 1) // 2, 6 + (cap * 9 + 9) // 10 - 1, cap + 5))
        self.assertEqual(series[2]["sample_dropped"], "5")
        self.assertNotIn("sample_dropped", series[0])
        # the exit line: the stretch since the last window line under its own names, the whole run under _total_
        self.assertEqual(stats(series[3], "_since_window"), (1, 4, 4, 4, 4))
        for window in series[:3]:                                        # window lines carry no exit key
            self.assertNotIn("sample_total_n", window)
            self.assertNotIn("sample_since_window_n", window)
        self.assertNotIn("sample_n", series[3])
        everything = 55 + 600 + sum(range(1, cap + 6)) + 4              # a value that fell out of a window is in the total
        self.assertEqual([int(series[3][f"sample_total_{k}"]) for k in ("n", "sum", "max")],
                         [10 + 3 + (cap + 5) + 1, everything, cap + 5])
        for x in series:                                                 # a series with no value prints zeros
            tail = "_since_window" if x["kind"] == "exit" else ""
            self.assertEqual([x[f"none{tail}_{k}"] for k in ("n", "sum", "p50", "p90", "max")], ["0"] * 5)
            self.assertEqual(x["raw_value"], "77")                       # a raw value is printed as it is, never as a change
        self.assertEqual([series[3][f"none_total_{k}"] for k in ("n", "sum", "max")], ["0"] * 3)

    def test_the_present_hook_is_called_at_every_present_with_the_counters_off(self) -> None:
        self.assertIn("hook calls 5", self.passing("hook").stdout)

    def test_hooks_registered_by_several_threads_at_once_are_all_kept(self) -> None:
        for attempt in range(40):   # a lost slot needs two registrations to meet: many short runs
            lines = rc_lines(self.passing("hook_race").stderr)
            exit_groups = {x["group"] for x in lines if x["kind"] == "exit"}
            self.assertEqual({f"racer{i}" for i in range(7)} - exit_groups, set(), f"attempt {attempt}")

    def test_a_stop_prints_the_exit_line(self) -> None:
        done = self.run_case("stop")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        sub = [x for x in rc_lines(done.stderr) if x["group"] == "sub"]
        self.assertEqual([(x["kind"], x["sub_n"]) for x in sub], [("exit", "9")])
        self.assertIn("FATAL: stopped on purpose", done.stderr)


if __name__ == "__main__":
    unittest.main()
