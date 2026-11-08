# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the ledger's writer on the NVN2 front end and the null back end (oot2/ledger.h, architecture D21).

Builds tests/runtime/nvn_ledger_test.c with runtime/core and runtime/nvn, as tests/test_runtime_nvn_queue.py builds its
harness, at -O1 and -O2, and reads the rows it prints. The cases drive synthetic op streams through the front end (no
game code): programs of synthetic control sections, textures registered in a texture pool, render targets, vertex
attributes.

- state: a draw with a 2D array texture, a 3D texture, a cube view, a storage image, a sint and a uint vertex attribute
  and a uint colour target makes one row for the vertex stage and one for the pixel stage whose answers are the state
  bound to that stage (the same texture index means different textures in the two stages); the same draw again adds
  nothing; a dispatch makes the compute stage's row, with its texture and image answers and no attachment formats;
  state bound in one recording holds for the draw of the next.
- call: a recording that draws, called from another, gives the rows of the caller's state; a recording called from
  another binds state the caller's draw uses.
- more: indexed and indirect draws make rows; a texture bound at an index in the image table's range (32 to 39) is no
  "tex" answer; a vertex attribute with no format is no answer; an image format the translator has no storage format
  for is "Unknown".
- mesh: a task, mesh and pixel program makes a row for each stage on a mesh draw, direct or indirect.
- stop: the run's stop path (ExitProcess, a fatal) flushes the ledger like the end of main.
- file: nothing is written while frames run, a flush asked for inside a submission is refused, a flush at a safe point
  writes the file, and the next run reads the file and adds to it.

Every row's digest is the one tools/shaders/cache_pack.py answers_digest computes for the answers.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set, as
continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_runtime_nvn as nvn  # noqa: E402

sys.path.insert(0, str(nvn.ROOT / "tools" / "shaders"))

import cache_pack as cp  # noqa: E402

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "nvn_ledger_test.c"
TIMEOUT = 120


def nvn_format(name: str) -> int:
    text = (nvn.INCLUDE / "oot2" / "nvn.h").read_text(encoding="utf-8")
    m = re.search(rf"\bNVN_FORMAT_{name} = (0x[0-9A-Fa-f]+|\d+),", text)
    assert m, name
    return int(m[1], 0)


def digest_of(answers: dict) -> str:
    return cp.digest_text(cp.answers_digest(answers))


def row(key: str, kind: str, answers: dict, attach: dict | None) -> dict:
    out = {"key": key, "kind": kind, "answers": answers, "digest": digest_of(answers)}
    if attach is not None:
        out["attach"] = attach
    return out


class Rows:
    """The stage keys and the rows a case printed."""

    def __init__(self, stdout: str) -> None:
        self.keys: dict[str, list[str]] = {}
        self.lines: list[str] = []
        for line in stdout.splitlines():
            if line.startswith("key "):
                _, kind, key = line.split(" ")
                self.keys.setdefault(kind, []).append(key)
            elif line.startswith("row "):
                self.lines.append(line[4:])
        self.rows = [json.loads(text) for text in self.lines]


ATTACH = {"color": [nvn_format("R32UI"), nvn_format("RGBA8")], "depth": nvn_format("DEPTH24_STENCIL8"), "samples": 1}
VS_ANSWERS = {"attr": {"0": "float", "1": "sint", "2": "float", "3": "uint"}, "tex": {"0": "3d", "3": "2d", "10": "2darray"}}
PS_ANSWERS = {"fout": {"0": "uint", "1": "float"}, "img": {"2": {"fmt": "R32Float", "type": "2d"}},
              "tex": {"0": "2darray", "5": "cube"}}
CS_ANSWERS = {"img": {"0": {"fmt": "R32G32B32A32Uint", "type": "1darray"}}, "tex": {"1": "cubearray"}}


def vs_row(keys: Rows) -> dict:
    return row(keys.keys["vs"][0], "vs", VS_ANSWERS, ATTACH)


def ps_row(keys: Rows, tex: dict) -> dict:
    return row(keys.keys["ps"][0], "ps", {**PS_ANSWERS, "tex": tex}, ATTACH)


class LedgerTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-nvnledger-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"nvn_ledger_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, *args: str) -> subprocess.CompletedProcess:
        try:
            return subprocess.run([str(self.exes[level]), *args], capture_output=True, text=True, timeout=TIMEOUT)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{args} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")

    def passes(self, case: str, *args: str):
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case):
                done = self.run_case(level, case, *args)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                yield done

    def check_text(self, rows: Rows) -> None:
        # The answers are the canonical JSON the translator job takes: sorted keys, no spaces
        for text, parsed in zip(rows.lines, rows.rows):
            self.assertIn("\"answers\":" + json.dumps(parsed["answers"], sort_keys=True, separators=(",", ":")), text)
            self.assertTrue(text.isascii())

    def test_a_draw_and_a_dispatch_make_rows_that_match_the_bound_state(self) -> None:
        for done in self.passes("state"):
            rows = Rows(done.stdout)
            self.check_text(rows)
            ps2 = ps_row(rows, {"0": "2d", "5": "cube"})
            self.assertEqual(rows.rows, [
                vs_row(rows),
                ps_row(rows, PS_ANSWERS["tex"]),
                row(rows.keys["cs"][0], "cs", CS_ANSWERS, None),
                ps2,
            ])
            # the same texture index is a different texture in the two stages
            self.assertNotEqual(rows.rows[0]["answers"]["tex"], rows.rows[1]["answers"]["tex"])
            self.assertEqual(rows.rows[0]["digest"], digest_of(VS_ANSWERS))
            self.assertNotEqual(rows.rows[1]["digest"], ps2["digest"])

    def test_runtime_full_digest_matches_sorted_ledger_answers_with_unknown_format(self) -> None:
        answers = {"fout": {"0": "uint", "1": "float"},
                   "img": {"2": {"fmt": "Unknown", "type": "2d"}},
                   "tex": {"10": "3d"}}
        canonical = json.dumps(answers, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        expected = hashlib.sha256(canonical).hexdigest()
        for done in self.passes("digest"):
            self.assertIn(f"digest {expected}", done.stdout)

    def test_digest_cache_is_reused_and_invalidated_by_vertex_answer_changes(self) -> None:
        for done in self.passes("digest-cache"):
            self.assertIn("ok digest-cache", done.stdout)

    def test_a_recording_called_from_another_gives_the_callers_state(self) -> None:
        for done in self.passes("call"):
            rows = Rows(done.stdout)
            self.check_text(rows)
            self.assertEqual(rows.rows, [
                vs_row(rows),
                ps_row(rows, PS_ANSWERS["tex"]),
                ps_row(rows, {**PS_ANSWERS["tex"], "7": "cubearray"}),
            ])

    def test_indexed_and_indirect_draws_unread_formats_and_the_image_range(self) -> None:
        for done in self.passes("more"):
            rows = Rows(done.stdout)
            self.check_text(rows)
            vs2 = row(rows.keys["vs"][0], "vs", {"attr": {"0": "float", "2": "sint"}, "tex": VS_ANSWERS["tex"]}, ATTACH)
            ps3 = row(rows.keys["ps"][0], "ps",
                      {**PS_ANSWERS, "tex": {"0": "2d", "5": "cube"},
                       "img": {"1": {"fmt": "Unknown", "type": "2d"}, **PS_ANSWERS["img"]}}, ATTACH)
            # DrawElements: both rows, with no "tex" answer for index 33; DrawElementsIndirect: the pixel row of the
            # new texture; an attribute with no format is no answer
            self.assertEqual(rows.rows, [vs_row(rows), ps_row(rows, PS_ANSWERS["tex"]),
                                         ps_row(rows, {"0": "2d", "5": "cube"}), vs2, ps3])

    def test_task_mesh_and_pixel_stages_make_rows_for_a_mesh_draw(self) -> None:
        for done in self.passes("mesh"):
            rows = Rows(done.stdout)
            self.check_text(rows)
            ts, ms, ps = rows.keys["ts"][0], rows.keys["ms"][0], rows.keys["ps"][1]
            fout = {"0": "uint", "1": "float"}
            self.assertEqual(rows.rows, [
                row(ts, "ts", {"tex": {"1": "3d"}}, ATTACH),
                row(ms, "ms", {"tex": {"2": "cube"}}, ATTACH),
                row(ps, "ps", {"fout": fout, "tex": {"0": "2darray"}}, ATTACH),
                row(ps, "ps", {"fout": fout, "tex": {"0": "2d"}}, ATTACH),
            ])

    def test_a_run_that_stops_writes_the_file(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory(prefix="oot2-ledger-") as folder:
                done = self.run_case(level, "stop", folder)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok stop", done.stdout)
                lines = (Path(folder) / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 3)

    def test_rows_are_written_at_a_safe_point_and_a_new_run_adds_to_the_file(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory(prefix="oot2-ledger-") as folder:
                done = self.run_case(level, "file", folder)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok file", done.stdout)
                self.assertIn("first flush wrote 2 rows", done.stdout)
                self.assertEqual(sorted(os.listdir(folder)), ["ledger.jsonl"])
                lines = (Path(folder) / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(json.loads(lines[0]), {"oot2_ledger": 1})
                self.assertEqual(lines[1:], sorted(lines[1:]))
                rows = Rows(done.stdout)
                self.assertEqual({json.dumps(json.loads(t), sort_keys=True) for t in lines[1:]},
                                 {json.dumps(r, sort_keys=True) for r in
                                  (vs_row(rows), ps_row(rows, PS_ANSWERS["tex"]),
                                   ps_row(rows, {"0": "2d", "5": "cube"}))})
                self.assertIn("rows read from", done.stderr)


if __name__ == "__main__":
    unittest.main()
