# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the coherence of VRAM mirrors with the CPU's view (runtime/gpu/pagegen.c, the mirrors of
runtime/gpu/backing.c; runtime/include/oot2/pagegen.h, runtime/include/oot2/backing.h) that need no GPU: one owner of
the write watch whose generations two consumers read alike, a CPU write read by the next submission, the admission rule
and its fallback to an import, a GPU write written back only once its submission was reached, a mirror replaced only
after the submission that wrote it, invalidation of pending writes after replay and completion, a bind in a raw-access
submission that grows past a mirror its page table maps
(the mirror keeps the part the bind leaves, an import takes the rest, no page has two copies), an import such a
submission binds a range larger than a uniform's of that becomes a mirror at the next raw-access begin (brought current,
in the page table, left alone when the submission bound it before its raw stage or the admission refuses, and not past
the cap on the mirrors that the raw write-back's size sets), and the binds and waits the map refuses by name. The device and the write watch
stand in (tests/runtime/coherence_test.c says how); the `watch` case uses the real write watch on Windows and prints a
skip line elsewhere. The device's own tests are tests/test_runtime_vk.py.

Also, without building anything: the runtime has one caller of rt_mem_written (a query resets the marks it reports, so a
second caller would take pages from the first).

Builds tests/runtime/coherence_test.c like tests/test_runtime_gpu.py's programs (build_program), at -O1 and -O2, and runs
each case in its own process. Needs clang and a Vulkan SDK with the headers and Volk, as tests/test_runtime_gpu.py does.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, LEVELS, ROOT, build_program, case_env, clang_or_skip, fatal_lines, \
    vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "coherence_test.c"

PASSING = ("generations", "upload", "admission", "writeback", "read_wait", "read_import", "read_effect", "read_raw",
           "read_unissued", "read_replay", "read_latest",
           "replace", "raw_grow", "raw_grow_head",
           "raw_grow_whole", "raw_grow_beside", "raw_grow_wait", "raw_sync_wait", "raw_promote", "raw_promote_small",
           "raw_promote_wait", "raw_promote_cap")

# case: text the one FATAL line must hold
FAILING = {
    "unwritten": "into a mirror was not written back by the time its value was done (a CPU wait for a fence)",
    "replace_written": "which submission 1 wrote on the GPU while it is being recorded, would be replaced by a backing "
                       "over",
    "raw_grow_taken": "earlier in raw-access submission 1, and a bind over",
}

# Where rt_mem_written may appear outside comments: its declaration, its definition, and its one caller.
WRITE_WATCH_FILES = {"runtime/include/oot2/memory.h", "runtime/core/memory.c", "runtime/gpu/pagegen.c"}


def code_mentions(path: Path, name: str) -> list[int]:
    """Lines of path that name `name` outside // comments."""
    found = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("//", 1)[0]
        if re.search(rf"\b{name}\b", code):
            found.append(n)
    return found


class WriteWatchOwnerTest(unittest.TestCase):
    """Nothing in the runtime but the page generations calls rt_mem_written."""

    def test_only_the_page_generations_query_the_write_watch(self) -> None:
        users = {}
        for path in sorted((ROOT / "runtime").rglob("*")):
            if path.suffix in (".c", ".h", ".cpp") and path.is_file():
                lines = code_mentions(path, "rt_mem_written")
                if lines:
                    users[path.relative_to(ROOT).as_posix()] = lines
        self.assertEqual(set(users), WRITE_WATCH_FILES, users)
        self.assertEqual(len(users["runtime/gpu/pagegen.c"]), 1, users)


class CoherenceTest(unittest.TestCase):
    """coherence_test.c with the runtime's core, NVN2 front end and gpu module, without a device."""

    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-coherence-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, include, glslang, TEST_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, timeout: int = 120,
                 env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout,
                              env=case_env(env))

    def test_passing_cases(self) -> None:
        for case in PASSING:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertEqual(fatal_lines(done.stderr), [])

    def test_the_real_write_watch_keeps_its_pages_for_the_owner(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "watch")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                if os.name != "nt":
                    self.assertIn("skip watch", done.stdout)
                    self.skipTest("no write watch on this platform")
                self.assertIn("ok watch", done.stdout)
                self.assertEqual(fatal_lines(done.stderr), [])

    def test_counter_run_without_the_invalidate_wait(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "read_wait", env={"OOT2_VK_READ_WAIT": "0"})
                self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("observed == 0xDDDDDDDDu", done.stderr)

    def test_counter_run_without_the_replay_wait(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "read_replay", env={"OOT2_VK_READ_WAIT": "0"})
                self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("0x12345678u", done.stderr)

    def test_what_the_map_cannot_keep_coherent_stops_by_name(self) -> None:
        for case, text in FAILING.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertNotIn("returned", done.stdout)
                    fatal = fatal_lines(done.stderr)
                    self.assertEqual(len(fatal), 1, done.stderr)
                    self.assertIn(text, fatal[0])


if __name__ == "__main__":
    unittest.main()
