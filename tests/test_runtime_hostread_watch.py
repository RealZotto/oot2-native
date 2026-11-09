# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Does the write watch of the runtime's own heap see a host file read? The page generations
(runtime/gpu/pagegen.c) take their stamps from Windows' write watch (rt_mem_written), and a mirror of guest memory is
brought current from those stamps. A host service that fills guest pages with a file read (the file system service
does) is a write the CPU's own instructions never make, so the question is whether the operating system's mark follows a
write made by the kernel on the process's behalf.

Builds tests/runtime/hostread_watch_test.c with runtime/core at -O1 and -O2 and runs each case in its own process. The
pages are the runtime's: rt_mem_init reserves the heap and the alias region with the write-watch flag, rt_mem_commit_io
commits the target, and rt_mem_written is the query. The file is made here from synthetic bytes. Each case reads into
committed pages after a reset of the watch and requires that the watch then reports exactly the pages the read
touched (and that the bytes arrived), twice over, for a read that covers whole pages, a long one, and (buffered) ones
that start and end inside a page:

- control: a plain memcpy, which the watch has always seen;
- sync, overlapped: ReadFile through the system cache, on a handle opened without and with FILE_FLAG_OVERLAPPED (the
  runtime's host files are opened overlapped, runtime/hos/hostfile.c); the overlapped read returns pending and is waited
  for, as the runtime does;
- sync_unbuffered, overlapped_unbuffered: the same with FILE_FLAG_NO_BUFFERING, where the file system moves the data into
  the caller's pages directly (a locked-pages transfer), the case most likely to bypass a mark made by the CPU;
- alias_sync, alias_overlapped: the same into the alias region, which is watched too.

The answer is the operating system's, and fixed by it: the test asserts it, so a change shows here. Windows only:
elsewhere the platform has no write watch, the cases print a skip line and are skipped; on Windows a skip is a failure.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set, as
continuous integration sets it.

Run: python -m unittest discover -s tests -v
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
TEST_SOURCE = ROOT / "tests" / "runtime" / "hostread_watch_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
HOSTFILE = ROOT / "runtime" / "hos" / "hostfile.c"
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000

PASSING = ("control", "sync", "overlapped", "sync_unbuffered", "overlapped_unbuffered", "alias_sync",
           "alias_overlapped")


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class HostReadWatchTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-hostread-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE, "sdk": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)] + sorted(str(p) for p in CORE.glob("*.c"))
        # Windows: the program links without high-entropy ASLR, as tests/test_runtime_memory.py explains.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"hostread_watch_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory(prefix="oot2-hostread-files-", ignore_cleanup_errors=True) as scratch:
            return subprocess.run([str(self.exes[level]), case, scratch], capture_output=True, text=True, timeout=120)

    def test_a_host_read_marks_the_pages_it_wrote(self) -> None:
        for level in LEVELS:
            for case in PASSING:
                with self.subTest(level=level, case=case):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    if f"skip {case}" in done.stdout:   # no write watch on this platform
                        self.assertNotEqual(os.name, "nt", done.stdout)
                        self.skipTest(done.stdout.strip())
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)
                    # every read reported all of its pages: no line says fewer than were read
                    for got, want in re.findall(r"(\d+) of (\d+) page\(s\) reported", done.stdout):
                        self.assertEqual(got, want, done.stdout)

    def test_the_overlapped_reads_really_waited(self) -> None:
        # The overlapped cases must have met a pending read at least once, or they only tested the fast path.
        for case in ("overlapped", "overlapped_unbuffered"):
            with self.subTest(case=case):
                done = self.run_case(LEVELS[0], case)
                if f"skip {case}" in done.stdout:
                    self.skipTest(done.stdout.strip())
                m = re.search(r"(\d+) read\(s\) returned pending", done.stdout)
                self.assertIsNotNone(m, done.stdout)
                self.assertGreater(int(m.group(1)), 0, done.stdout)

    def test_the_runtime_reads_files_the_way_the_cases_do(self) -> None:
        # The cases cover a handle opened overlapped through the cache. If the host files change to another way of
        # reading (no cache, write-through, a mapped view, scatter reads), the answer above does not cover them.
        code = "\n".join(line.split("//", 1)[0] for line in HOSTFILE.read_text(encoding="utf-8").splitlines())
        self.assertIn("FILE_FLAG_OVERLAPPED", code)
        for other in ("FILE_FLAG_NO_BUFFERING", "FILE_FLAG_WRITE_THROUGH", "ReadFileScatter", "MapViewOfFile"):
            self.assertNotIn(other, code)

    def test_unknown_case_is_refused(self) -> None:
        self.assertEqual(self.run_case(LEVELS[0], "no_such_case").returncode, 2)


if __name__ == "__main__":
    unittest.main()
