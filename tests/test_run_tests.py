# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the parallel test runner (tools/run_tests.py) on synthetic modules.

Each case writes a small folder of test modules and runs the real runner on
it: every way a module can go wrong (a failing test, an error, a crash before
unittest reports, a hang) must make the whole run fail and be named, and the
passing counterpart, with skips, must pass with the counts a plain unittest run
would print.

Run: python -m unittest discover -s tests -p test_run_tests.py -v
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import run_tests  # noqa: E402

PASS = """
import unittest
class T(unittest.TestCase):
    def test_a(self): pass
    def test_b(self): pass
    @unittest.skip("not here")
    def test_c(self): pass
"""
FAIL = """
import unittest
class T(unittest.TestCase):
    def test_a(self): self.assertEqual(1, 2)
"""
ERROR = """
import unittest
class T(unittest.TestCase):
    def test_a(self): raise RuntimeError("boom")
"""
CRASH = """
import os
os._exit(3)
"""
QUIET_EXIT = """
import os
os._exit(0)
"""
HANG = """
import time, unittest
class T(unittest.TestCase):
    def test_a(self): time.sleep(60)
"""


class RunnerTest(unittest.TestCase):
    def run_on(self, modules: dict[str, str], *extra: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as tmp:
            for name, body in modules.items():
                Path(tmp, name + ".py").write_text(textwrap.dedent(body), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = run_tests.main(["--tests-dir", tmp, "--jobs", "2", *extra])
            return code, out.getvalue()

    def test_passing_modules_pass_with_the_plain_counts(self):
        code, out = self.run_on({"test_one": PASS, "test_two": PASS})
        self.assertEqual(code, 0, out)
        self.assertIn("\nRan 6 tests in ", out)
        self.assertTrue(out.rstrip().endswith("OK (skipped=2)"), out)
        self.assertIn("===== test_one: ok", out)
        self.assertIn("===== test_two: ok", out)

    def test_a_failing_test_fails_the_run_and_is_named(self):
        code, out = self.run_on({"test_one": PASS, "test_bad": FAIL})
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED module test_bad (exit 1)", out)
        self.assertNotIn("FAILED module test_one", out)
        self.assertTrue(out.rstrip().endswith("FAILED (modules=1, skipped=1)"), out)

    def test_an_error_fails_the_run(self):
        code, out = self.run_on({"test_one": PASS, "test_bad": ERROR})
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED module test_bad", out)

    def test_a_crash_before_unittest_reports_fails_the_run(self):
        code, out = self.run_on({"test_one": PASS, "test_bad": CRASH})
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED module test_bad (no Ran line)", out)

    def test_an_exit_0_before_unittest_reports_fails_the_run(self):
        code, out = self.run_on({"test_one": PASS, "test_bad": QUIET_EXIT})
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED module test_bad (no Ran line)", out)

    def test_the_wall_clock_modules_run_alone(self):
        parallel, serial = run_tests.order(["test_runtime_vk_miss", "test_runtime_untile"])
        self.assertEqual(parallel, ["test_runtime_untile"])
        self.assertEqual(serial, ["test_runtime_vk_miss"])

    def test_a_hang_is_cut_at_the_timeout_and_fails_the_run(self):
        code, out = self.run_on({"test_one": PASS, "test_bad": HANG}, "--timeout", "5")
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED module test_bad (timed out)", out)

    def test_named_modules_run_alone(self):
        code, out = self.run_on({"test_one": PASS, "test_bad": FAIL}, "test_one.py")
        self.assertEqual(code, 0, out)
        self.assertNotIn("test_bad", out)

    def test_an_unknown_module_is_refused(self):
        with self.assertRaises(SystemExit):
            self.run_on({"test_one": PASS}, "test_none")

    def test_serial_modules_run_after_the_others(self):
        old = run_tests.SERIAL
        run_tests.SERIAL = ("test_a",)
        try:
            code, out = self.run_on({"test_a": PASS, "test_b": PASS, "test_c": PASS})
        finally:
            run_tests.SERIAL = old
        self.assertEqual(code, 0, out)
        self.assertGreater(out.index("===== test_a:"), out.index("===== test_b:"))
        self.assertGreater(out.index("===== test_a:"), out.index("===== test_c:"))

    def test_the_longest_modules_start_first(self):
        parallel, serial = run_tests.order(["test_z", "test_build_game", "test_a"])
        self.assertEqual(parallel, ["test_build_game", "test_a", "test_z"])
        self.assertEqual(serial, [])


if __name__ == "__main__":
    unittest.main()
