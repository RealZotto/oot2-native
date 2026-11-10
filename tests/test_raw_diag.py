# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The raw path's miss diagnostics that need no device (runtime/gpu/raw_diag.c, oot2/raw_diag.h; setting raw_diag): the
index of the game's memory pools (a lookup that says whether a pool was live for the submission that missed, finalized
before it or made after it, how many pools finalized since, and where the nearest pools lie when there is none), the class
a missed address falls into from the facts about it (pool, declared mapped, committed, mapped now) and the verdict that
class makes (a page the table lacks, a page it learned later, a page never committed, an address in no pool), and the
counts by stage and by class, with a stage and class logged once.

tests/runtime/raw_diag_test.c holds the cases; this file builds it with the warnings as errors at -O1 and -O2, runs each
case in its own process and reads its "ok <name>" line.

Needs clang (OOT2_CC or PATH); skips without it, unless OOT2_REQUIRE_CLANG is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXE, LEVELS, ROOT, RUNTIME, clang_or_skip

SOURCE = ROOT / "tests" / "runtime" / "raw_diag_test.c"
CASES = ("find", "nested", "lifetime", "ring", "many", "classes", "verdicts", "describe", "tally", "tally_room", "shape",
         "scan")


class RawDiagTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        cls._build = tempfile.TemporaryDirectory(prefix="oot2-rawdiag-", ignore_cleanup_errors=True)
        for level in LEVELS:
            exe = Path(cls._build.name) / f"raw_diag_test{level}{EXE}"
            done = subprocess.run([cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror",
                                   "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), str(SOURCE),
                                   str(RUNTIME / "gpu" / "raw_diag.c"), "-o", str(exe)], capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f"build of {SOURCE.name} at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_build"):
            cls._build.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=120)

    def test_every_case_passes_at_each_level(self) -> None:
        for case in CASES:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertEqual(done.stdout, f"ok {case}\n")

    def test_the_list_of_cases_is_the_programs(self) -> None:
        """A case added to the program and left out of CASES would never run."""
        text = SOURCE.read_text(encoding="utf-8")
        listed = set(re.findall(r'\{"(\w+)", case_\w+\}', text))
        self.assertEqual(listed, set(CASES))

    def test_an_unknown_case_is_refused(self) -> None:
        done = self.run_case(LEVELS[0], "no_such_case")
        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
