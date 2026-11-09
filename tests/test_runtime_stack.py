# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract test of the stack a runtime thread starts with (runtime/core/plat.c, plat_thread_start and plat_fault_hook).

A first touch of guest memory is an access violation that the runtime's vectored handler settles. Windows builds the
exception frame on the faulting thread's own stack, under the stack pointer; the frame holds the processor's extended
state (about 2 KB with AVX2, several times that with AVX-512 or AMX), and the kernel grows a stack only a few pages
past its committed end. A thread that starts with a few KB committed and takes a fault on such a processor is ended by
the violation itself, with exit code 0xC0000005 and no handler call. That is how a continuous-integration run once
lost dozens of the audio and IPC cases at start-up, all with exit code 3221225477 and no output after SetHeapSize.

tests/runtime/stack_test.c takes the violation with the stack pointer 128 KiB below where a thread's code runs, on
stack nothing has touched, which puts the frame where only stack committed ahead can take it (the same place a bigger
frame reaches from a shallow stack). Nothing is read from a dump. Windows only: the other platforms settle the fault on
a signal stack of their own. Needs clang for x86-64 (OOT2_CC, else clang on PATH); without one the tests skip, unless
OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

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

CORE = ROOT / "runtime" / "core"
TEST_SOURCE = ROOT / "tests" / "runtime" / "stack_test.c"
LEVELS = ("-O1", "-O2")


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


@unittest.skipUnless(os.name == "nt", "the stack headroom is a Windows matter")
class StackTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-stack-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": 0x8000}), out)
        sources = [str(TEST_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        for level in LEVELS:
            exe = out / f"stack_test{level}.exe"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror",
                   "-Wl,--disable-high-entropy-va", "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE),
                   "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def passes(self, case: str) -> None:
        for level in LEVELS:
            with self.subTest(level=level, case=case):
                done = subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=60)
                self.assertEqual(done.returncode, 0, f"exit code {done.returncode & 0xFFFFFFFF}\n{done.stdout}{done.stderr}")
                self.assertIn(f"ok {case}", done.stdout)

    def test_a_violation_on_the_stack_just_below_a_threads_code_reaches_the_handler(self) -> None:
        self.passes("thread_shallow")

    def test_a_guest_threads_stack_takes_a_violation_far_below_anything_it_touched(self) -> None:
        self.passes("thread_deep")

    def test_a_helper_threads_stack_takes_a_violation_far_below_anything_it_touched(self) -> None:
        self.passes("small_thread_deep")

    def test_a_guest_fibres_stack_takes_a_violation_far_below_anything_it_touched(self) -> None:
        self.passes("fibre_deep")

    def test_the_stack_of_the_thread_that_installs_the_handler_takes_a_violation_far_below_anything_it_touched(self) -> None:
        self.passes("first_thread_deep")

    def test_unknown_case_is_refused(self) -> None:
        done = subprocess.run([str(self.exes[LEVELS[0]]), "no_such_case"], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 2)


if __name__ == "__main__":
    unittest.main()
