# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the memory line checker (tools/check_memline.py) on synthetic files.

Every file here is written by synth() below in the format of runtime/include/oot2/memline.h,
with the columns read from that header, so a column added to the runtime is a column here.
No game, no runtime: tests/test_runtime_memline.py checks that the runtime writes the same
format and runs this checker on its rows.

The passing file sums within 3 percent at the title, the plateau and the scene entry. Each
failing case differs from it in one thing: the own-memory image bytes dropped from the
plateau row (the counter-run the ticket asks for: the check must fail naming that term), a
residual no term explains, a column missing from the file, a row that is not there.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_memline as cm  # noqa: E402

HEADER = ROOT / "runtime" / "include" / "oot2" / "memline.h"
MIB = 1 << 20


def names(macro: str) -> list[str]:
    text = HEADER.read_text(encoding="utf-8")
    m = re.search(rf"#define {macro}\(X\)(.*?)\n\n", text + "\n\n", re.S)
    assert m, macro
    return re.findall(r"X\((\w+)\)", m[1])


COLUMNS, SUMMED = names("MEMLINE_TERMS"), names("MEMLINE_SUMMED")


def terms(scale: int = 0) -> dict[str, int]:
    """Terms of 994 MiB (plus scale MiB on the heap) against a private commit 0.6 percent above them."""
    t = {c: 0 for c in COLUMNS}
    t.update(stack_committed=8, heap_committed=300 + scale, alias_committed=100, modules_committed=50, kernel_committed=2,
             wc32_bytes=64, wcodd_bytes=20, host_other_bytes=190, bk_block_bytes=120, tex_arena_bytes=100, img_own_bytes=40)
    t = {k: v * MIB if k in SUMMED else v for k, v in t.items()}
    t["private"] = sum(t[c] for c in SUMMED) * 1006 // 1000
    t["vk_heap0_used"] = 400 * MIB + t["bk_block_bytes"] + t["tex_arena_bytes"] + t["img_own_bytes"]
    return t


def synth(path: Path, mutate=None, drop_column: str | None = None, kinds=("minute", "title", "minute", "minute", "scene", "end")) -> Path:
    cols = [c for c in COLUMNS if c != drop_column]
    summed = [c for c in SUMMED if c != drop_column]
    lines = ["oot2-memline 1", "columns\t" + "\t".join(cols), "sum\t" + "\t".join(summed)]
    for i, kind in enumerate(kinds):
        t = terms(i)
        t["seq"], t["t_ns"] = 100 * i, i * 60_000_000_000
        if mutate:
            mutate(i, kind, t)
        lines.append("row\t" + kind + "\t" + "\t".join(str(t[c]) for c in cols))
    lines.append(f"end\t{len(kinds)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class CheckMemlineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-checkmemline-", ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_check(self, path: Path, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cm.main([str(path), *args])
        return code, out.getvalue(), err.getvalue()

    def test_rows_that_sum_within_three_percent_pass(self) -> None:
        code, out, _ = self.run_check(synth(self.dir / "m.tsv"), "--require", "title,plateau,scene")
        self.assertEqual(code, 0, out)
        self.assertIn("PASS 3 row(s)", out)
        for name in ("title:", "plateau:", "scene:"):
            self.assertIn(name, out)

    def test_dropping_the_own_memory_image_bytes_from_the_plateau_fails_by_that_name(self) -> None:
        def drop(i, kind, t):
            if i == 3:   # the last minute row before the scene entry: the plateau
                t["img_own_bytes"] = 0
        code, out, _ = self.run_check(synth(self.dir / "m.tsv", drop))
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"FAIL plateau: .*residual")
        self.assertRegex(out, r"FAIL plateau: term img_own_bytes is 0 here")
        self.assertNotIn("FAIL title", out)
        self.assertNotIn("FAIL scene", out)
        # and the file without the drop passes: the check is red on this term alone
        self.assertEqual(self.run_check(synth(self.dir / "ok.tsv"))[0], 0)

    def test_a_dropped_term_that_is_minus_one_is_named_too(self) -> None:
        def drop(i, kind, t):
            if kind == "scene":
                t["bk_block_bytes"] = cm.UNREAD
        code, out, _ = self.run_check(synth(self.dir / "m.tsv", drop))
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"FAIL scene: term bk_block_bytes is -1 here")

    def test_a_residual_no_term_explains_is_reported_as_a_residual(self) -> None:
        def grow(i, kind, t):
            if kind == "title":
                t["private"] += 600 * MIB
        code, out, _ = self.run_check(synth(self.dir / "m.tsv", grow))
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL title: the terms sum to", out)
        self.assertIn("memory no named term counts", out)
        self.assertNotIn("is missing from this row", out)

    def test_a_term_over_private_commit_fails_too(self) -> None:
        def shrink(i, kind, t):
            if kind == "scene":
                t["private"] -= 200 * MIB
        code, out, _ = self.run_check(synth(self.dir / "m.tsv", shrink))
        self.assertEqual(code, 1, out)
        self.assertIn("FAIL scene", out)

    def test_a_sum_column_the_file_lacks_is_a_format_error(self) -> None:
        path = synth(self.dir / "m.tsv", drop_column="img_own_bytes")
        # the sum line was written without it too, so only the columns line is short: the file is whole, the sum smaller
        code, out, _ = self.run_check(path)
        self.assertEqual(code, 1, out)   # 4 percent of the commit is gone: the sum fails
        text = path.read_text(encoding="utf-8").replace("sum\t", "sum\timg_own_bytes\t", 1)
        path.write_text(text, encoding="utf-8")
        code, _, err = self.run_check(path)
        self.assertEqual(code, 2)
        self.assertIn("img_own_bytes", err)

    def test_a_required_row_that_is_absent_fails_and_an_optional_one_is_skipped(self) -> None:
        path = synth(self.dir / "m.tsv", kinds=("minute", "minute", "title", "end"))
        code, out, _ = self.run_check(path, "--require", "scene")
        self.assertEqual(code, 1)
        self.assertIn("FAIL scene: the file has no such row", out)
        code, out, _ = self.run_check(path)
        self.assertEqual(code, 0, out)
        self.assertIn("scene: no such row", out)

    def test_the_bc_saving_is_printed_for_the_scene_and_end_rows(self) -> None:
        def decoded(i, kind, t):
            t.update(astc_dec_w4=100 * MIB, astc_dec_w6=1500 * MIB, astc_dec_w12=40 * MIB, bc_equiv_bytes=410 * MIB,
                     astc_dec_srgb_bytes=900 * MIB, astc_dec_arena_bytes=1600 * MIB, astc_dec_images=321,
                     native_bc_bytes=55 * MIB, other_bytes=66 * MIB)
        code, out, _ = self.run_check(synth(self.dir / "m.tsv", decoded))
        self.assertEqual(code, 0, out)
        lines = [ln for ln in out.splitlines() if ln.startswith("bc saving")]
        self.assertEqual([ln.split(":")[0] for ln in lines], ["bc saving (scene)", "bc saving (end)"])
        self.assertIn("decoded 1.60 GiB (4: 0.10 GiB, 6: 1.46 GiB, 12: 0.04 GiB) - bc_equiv 0.40 GiB = 1.20 GiB", lines[0])
        self.assertIn("321 decoded images, srgb 0.88 GiB, in the arena 1.56 GiB, already BC 0.05 GiB, other formats 0.06 GiB", lines[0])

    def test_a_row_that_could_not_read_the_counters_prints_no_saving(self) -> None:
        def unread(i, kind, t):
            t["astc_dec_w6"] = cm.UNREAD
        code, out, _ = self.run_check(synth(self.dir / "m.tsv", unread))
        self.assertEqual(code, 0, out)
        self.assertNotIn("bc saving", out)

    def test_a_file_of_the_older_format_without_the_columns_still_parses_and_prints_no_saving(self) -> None:
        new = [c for c in COLUMNS if c.startswith("astc_dec") or c in ("bc_equiv_bytes", "native_bc_bytes", "other_bytes")]
        self.assertEqual(len(new), 12)
        for c in new:
            self.assertNotIn(c, SUMMED)
        path = self.dir / "old.tsv"
        text = synth(path).read_text(encoding="utf-8").splitlines()
        keep = [i for i, c in enumerate(COLUMNS) if c not in new]
        for n, ln in enumerate(text):
            f = ln.split("\t")
            if f[0] == "columns":
                text[n] = "\t".join(["columns"] + [f[1 + i] for i in keep])
            elif f[0] == "row":
                text[n] = "\t".join(f[:2] + [f[2 + i] for i in keep])
        path.write_text("\n".join(text) + "\n", encoding="utf-8")
        code, out, _ = self.run_check(path, "--require", "title,plateau,scene")
        self.assertEqual(code, 0, out)
        self.assertNotIn("bc saving", out)

    def test_not_a_memory_line_is_exit_two(self) -> None:
        path = self.dir / "x.tsv"
        path.write_text("oot2-capture 1\n", encoding="utf-8")
        self.assertEqual(self.run_check(path)[0], 2)


if __name__ == "__main__":
    unittest.main()
