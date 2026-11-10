# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the coherence of VRAM mirrors with the CPU's view (runtime/gpu/pagegen.c, the mirrors of
runtime/gpu/backing.c; runtime/include/oot2/pagegen.h, runtime/include/oot2/backing.h) that need no GPU: one owner of
the write watch whose generations two consumers read alike, a CPU write read by the next submission, the admission rule
and its fallback to an import, a GPU write written back only once its submission was reached, a mirror replaced only
after the submission that wrote it, a bind in a raw-access submission that grows past a mirror its page table maps
(the mirror keeps the part the bind leaves, or grows in place if already bound, without creating two copies: by the
bind's pages only, whatever the imports it overlaps and the mirrors' total are), an import such a submission binds a
range larger than a uniform's of that becomes a mirror at the next raw-access begin (brought current, in the page
table, left alone when the submission bound it before its raw stage or the admission refuses, and not past the cap on
the mirrors that the raw write-back's traffic sets), pages the GPU wrote into guest memory itself (the raw path's sinks)
told from the CPU's writes by the page generations, and the binds and waits the map refuses by name, a mirror's growth
with the reason it could not.
The device and the write watch stand in (tests/runtime/coherence_test.c says how); the `watch` case uses the real write
watch on Windows and prints a skip line elsewhere. The device's own tests are tests/test_runtime_vk.py.

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

PASSING = ("generations", "upload", "admission", "guarded_admission", "writeback", "writeback_no_reupload",
           "writeback_partial_page", "writeback_replaced", "writeback_long_run", "gpu_mark", "gpu_mark_direct", "replace",
           "raw_grow", "raw_grow_head", "raw_grow_whole", "raw_grow_beside", "raw_grow_taken", "raw_grow_neighbour",
           "raw_grow_taken_wait", "raw_grow_taken_head", "raw_grow_taken_layout", "raw_grow_trimmed",
           "raw_grow_guarded_head", "raw_grow_guarded_tail", "raw_grow_over_total", "raw_grow_head_import",
           "raw_grow_tail_import", "raw_grow_wait", "raw_sync_wait", "raw_sync_nowait",
           "raw_promote", "raw_promote_small", "raw_promote_wait", "raw_promote_cap")

# case: text the one FATAL line must hold
FAILING = {
    "unwritten": "into a mirror was not written back by the time its value was done (a CPU wait for a fence)",
    "replace_written": "which submission 1 wrote on the GPU while it is being recorded, would be replaced by a backing "
                       "over",
    "gpu_mark_direct_partial": "a direct write-back mark of [",
    # A mirror that cannot grow stops with the reason, after the account of the bind (block offsets are the block's own).
    "raw_grow_bound_import": "which this submission bound, lies on the pages it would add",
    "raw_grow_fragmented": "the mirror cannot grow over them: its block has no free space after its part for "
                           "[0x60000, 0x150000)",
    "raw_grow_guarded_blocked": "the mirror cannot grow over them: its block has no free space after its part for "
                                "[0x30000, 0x40000)",
    "raw_grow_untracked": "the mirror cannot grow over them: the write watch does not cover [",
    "raw_grow_cap": "KiB, more than the 1024 KiB one growth may add",
    # The same stop with its dump before it (test_the_stop_of_a_straddling_bind_prints_what_it_is_made_of).
    "raw_diag_hull": "which this submission bound, lies on the pages it would add",
    "raw_diag_dry": "which this submission bound, lies on the pages it would add",
    "raw_diag_birth": "which this submission bound, lies on the pages it would add",
}

# The dump's lines, as the log writes them: one prefix, the text after it.
DUMP = "[rt] gfx: vulkan: backing map: straddle dump: "

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


def dump_lines(stderr: str) -> list[str]:
    return [line[len(DUMP):] for line in stderr.splitlines() if line.startswith(DUMP)]


def printed(stdout: str, case: str) -> dict[str, int]:
    """The key=value line the case printed before its stop, numbers read as the case wrote them (0x... or decimal)."""
    line = next(line for line in stdout.splitlines() if line.startswith(case + ":"))
    return {key: int(value, 0) for key, value in re.findall(r"(\w+)=(0x[0-9a-f]+|\d+)", line)}


def hx(n: int) -> str:
    return f"{n:#x}"


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

    def run_case(self, level: str, case: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout,
                              env=case_env())

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

    def stop_with_dump(self, level: str, case: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        done = self.run_case(level, case)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertNotIn("returned", done.stdout)
        fatal = fatal_lines(done.stderr)
        self.assertEqual(len(fatal), 1, done.stderr)   # the dump is log lines: the stop is still the one FATAL
        self.assertIn(FAILING[case], fatal[0])
        lines = dump_lines(done.stderr)
        self.assertTrue(lines, done.stderr)
        last = done.stderr.splitlines().index(DUMP + lines[-1])
        self.assertTrue(done.stderr.splitlines()[last + 1].startswith("FATAL"), done.stderr)   # nothing between
        return done, lines

    def backing_line(self, lines: list[str], kind: str, lo: int, hi: int) -> str:
        found = [line for line in lines if line.startswith(f"backing {kind} [{hx(lo)}, {hx(hi)}) ")]
        self.assertEqual(len(found), 1, (kind, hx(lo), hx(hi), lines))
        return found[0]

    def test_the_stop_of_a_straddling_bind_prints_what_it_is_made_of(self) -> None:
        """The hull of what the submission resolved in the import and in the mirror, the binds in order with their phase."""
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_diag_hull")
                k = printed(done.stdout, "raw_diag_hull")
                p, v, v0 = k["p"], k["v"], k["v0"]
                imp = self.backing_line(lines, "import", p - 0x20000, p)
                self.assertIn(f"res (value {v}, [{hx(p - 0x10000)}, {hx(p - 0x10000 + 4)}), n 1) born fill_pool for {v} ", imp)
                mirror = self.backing_line(lines, "mirror", p, p + 0x40000)
                self.assertIn(f"res (value {v}, [{hx(p + 0x100)}, {hx(p + 0x3004)}), n 2) born bind for {v0} "
                              f"was [{hx(p)}, {hx(p + 0x40000)})", mirror)
                self.assertEqual(sum("<- the mirror the bind reaches" in line for line in lines), 1)
                bind = [line for line in lines if line.startswith("bind [")]
                self.assertEqual(len(bind), 1, lines)
                self.assertIn(f"bind [{hx(k['bind_addr'])}, {hx(k['bind_addr'] + k['bind_size'])}) pages "
                              f"[{hx(p - 0x2000)}, {hx(p + 0x4000)}) ", bind[0])
                self.assertTrue(lines[0].startswith(f"growth refused in submission {v} "), lines[0])
                head = next(i for i, line in enumerate(lines) if line.startswith(f"the binds of submission {v},"))
                ring = []
                for line in lines[head + 1:]:
                    if line.startswith("the last "):
                        break
                    ring.append(line)
                want = [f"submission {v} before_begin hit mirror addr {hx(p + 0x3000)} size 0x4 ",
                        f"submission {v} after_begin hit mirror addr {hx(p + 0x100)} size 0x4 ",
                        f"submission {v} after_begin hit import addr {hx(p - 0x10000)} size 0x4 ",
                        f"submission {v} after_begin miss none addr {hx(k['bind_addr'])} size {hx(k['bind_size'])} "]
                self.assertEqual(len(ring), len(want), ring)
                for line, text in zip(ring, want):
                    self.assertIn(text, line)
                self.assertIn(f"stop 1 of this run follows", lines[-1])

    def test_the_dry_evaluation_of_the_growth_leaves_the_map_as_it_was(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_diag_dry")
                k = printed(done.stdout, "raw_diag_dry")
                p, v = k["p"], k["v"]
                left = [line for line in lines if line.startswith("the dry evaluation left the map ")]
                self.assertEqual(len(left), 1, lines)
                match = re.fullmatch(r"the dry evaluation left the map unchanged \(digest (0x[0-9a-f]+) before, "
                                     r"(0x[0-9a-f]+) after\)", left[0])
                self.assertTrue(match, left[0])
                self.assertEqual(match.group(1), match.group(2))
                guards = {}
                for line in lines:
                    m = re.match(r"guard (G\d\w?): (passes|refuses)(?:: (.*))?$", line)
                    if m:
                        guards.setdefault(m.group(1), []).append((m.group(2), m.group(3) or ""))
                self.assertEqual(sorted(guards), ["G1", "G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G8", "G9"])
                for tag in guards:
                    verdict = [d[0] for d in guards[tag]]
                    self.assertEqual(verdict, ["refuses"] if tag == "G4" else ["passes"], (tag, guards[tag]))
                self.assertEqual(guards["G4"][0][1],
                                 f"the import [{hx(p - 0x20000)}, {hx(p)}) used_by {v} written 0 hull "
                                 f"[{hx(p - 0x10000)}, {hx(p - 0x10000 + 4)}) hull_on_added_pages no hull_on_bind no")

    def test_the_dump_says_where_each_backing_came_from(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_diag_birth")
                k = printed(done.stdout, "raw_diag_birth")
                p, T, P, v0, v1, v = k["p"], k["T"], k["P"], k["v0"], k["v1"], k["v"]
                self.assertIn(f"born grow for {v} was [{hx(p)}, {hx(p + 0x40000)})",
                              self.backing_line(lines, "mirror", p - 0x2000, p + 0x40000))
                self.assertIn(f"born fill_pool for {v} ", self.backing_line(lines, "import", p - 0x20000, p - 0x2000))
                self.assertIn(f"born beside_front for {v} was [{hx(T)}, {hx(T + 0x40000)})",
                              self.backing_line(lines, "mirror", T, T + 0x3F000))
                promoted = self.backing_line(lines, "mirror", P, P + 0x20000)
                self.assertIn(f"born promote for {v} was [{hx(P)}, {hx(P + 0x20000)})", promoted)
                # nothing resolved it: the hull is empty, as a new backing's is (a low end above the high end)
                self.assertIn("res (value 0, empty (lo 0xffffffffffffffff, hi 0), n 0)", promoted)
                # how each bind of the submission was served: the cut one made an import, the grown one grew the mirror
                ring = [line for line in lines if line.startswith("ring ") and f"submission {v} " in line]
                self.assertTrue(any(f"made import addr {hx(T + 0x3F000)} size 0x20000 " in line for line in ring), ring)
                self.assertTrue(any(f"grew mirror addr {hx(p - 0x1800)} size 0x5800 " in line for line in ring), ring)
                head = next(i for i, line in enumerate(lines) if line.startswith("the last "))
                self.assertEqual(lines[head], f"the last 1 binds of the raw-access submission before it ({v1})")
                self.assertIn(f"submission {v1} after_begin hit import addr {hx(P)} size 0x20000 ", lines[head + 1])


if __name__ == "__main__":
    unittest.main()
