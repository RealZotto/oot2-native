# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Host tests of the input layer's pure pieces (no SDL, no GPU): compiles tests/runtime/hid_input_test.c with
runtime/hos/hid_input.c (which has no runtime dependency) and runs each case. No Vulkan SDK, no game, no dump.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "hid_input_test.c"


class HidInputTest(unittest.TestCase):
    exe: Path

    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        if not cc:
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-hid-input-")
        exe = Path(cls.tmp.name) / "hid_input_test.exe"
        cmd = [cc, "-std=gnu11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-I", str(RUNTIME / "include"), str(RUNTIME / "hos" / "hid_input.c"),
               str(TEST_SOURCE), "-o", str(exe)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build of hid_input_test failed:\n{done.stdout}{done.stderr}")
        cls.exe = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, case: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exe), case], capture_output=True, text=True)

    def test_cases(self) -> None:
        for case in ("deadzone", "publish", "bits", "rumble", "swallow", "remap", "keyboard_sticks"):
            with self.subTest(case=case):
                done = self.run_case(case)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn(f"ok {case}", done.stdout)


if __name__ == "__main__":
    unittest.main()
