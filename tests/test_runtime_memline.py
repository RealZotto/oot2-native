# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the memory line (runtime/nvn/nvn_memline.c, oot2/memline.h).

Builds tests/runtime/memline_test.c with runtime/core and runtime/nvn at -O1 and -O2
(the null back end) and runs each case in its own process. What the cases show:

- sum: memline_sum totals the named terms, a term that could not be read (-1) counts 0,
  and the residual is the commit less the total;
- rows: the helper thread writes minute rows, the title and scene rows and the closing
  row, in a file whose columns are the ones the header names, with the noted block of
  1 MiB or more and not the small one; the rows of synthetic counters sum within 3
  percent of the commit;
- gather: the gather reads what rt_mem_measure reads, a cheap gather reads -1 where a
  runtime lock would be taken, and the cost per row is reported;
- mark: marking a row and noting a block cost the calling thread under a microsecond,
  a tiny fraction of the 1 ms bound for the render thread.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless
OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402
from test_runtime_nvn import CORE, LEVELS, MODULE_SIZE, NVN, compile_flags, find_compiler  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "memline_test.c"
HOST_SOURCE = ROOT / "tests" / "runtime" / "pages_host.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"
PAGEGEN_SOURCE = ROOT / "runtime" / "gpu" / "pagegen.c"
HEADER = ROOT / "runtime" / "include" / "oot2" / "memline.h"
MIB = 1 << 20


def header_names(macro: str) -> list[str]:
    """The X(name) entries of a macro of oot2/memline.h, in order."""
    text = HEADER.read_text(encoding="utf-8")
    m = re.search(rf"#define {macro}\(X\)(.*?)\n\n", text + "\n\n", re.S)
    assert m, macro
    return re.findall(r"X\((\w+)\)", m[1])


def read_memline(path: Path) -> dict:
    """The file as {'columns', 'sum', 'rows': [(kind, {col: int})], 'alloc': [fields], 'end': rows}."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "oot2-memline 1", lines[0]
    out: dict = {"rows": [], "alloc": [], "end": None}
    for line in lines[1:]:
        f = line.split("\t")
        if f[0] == "columns":
            out["columns"] = f[1:]
        elif f[0] == "sum":
            out["sum"] = f[1:]
        elif f[0] == "row":
            assert len(f) - 2 == len(out["columns"]), line
            out["rows"].append((f[1], dict(zip(out["columns"], map(int, f[2:])))))
        elif f[0] == "alloc":
            out["alloc"].append(f[1:])
        elif f[0] == "end":
            out["end"] = int(f[1])
    return out


class MemlineTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-memline-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = ([str(HOST_SOURCE), str(TEST_SOURCE), str(END_SOURCE), str(PAGEGEN_SOURCE)] +
                   sorted(str(p) for p in CORE.glob("*.c")) + sorted(str(p) for p in NVN.glob("*.c")))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"memline_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*compile_flags(cc), level, *extra, "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str):
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_MEMLINE")}
        env["OOT2_MEMLINE_SECONDS"] = "0.2"
        folder = Path(tempfile.mkdtemp(prefix=f"{case}-", dir=self.tmp.name))
        done = subprocess.run([str(self.exes[level]), case, str(folder)], capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        self.assertNotIn("FATAL", done.stderr)
        return done, folder

    def test_sum_of_the_named_terms(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                self.run_case(level, "sum")

    def test_the_helper_writes_the_rows_and_they_parse_back(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                _, folder = self.run_case(level, "rows")
                files = list(folder.glob("memory-*.tsv"))
                self.assertEqual(len(files), 1)
                data = read_memline(files[0])
                self.assertEqual(data["columns"], header_names("MEMLINE_TERMS"))
                self.assertEqual(data["sum"], header_names("MEMLINE_SUMMED"))
                kinds = [k for k, _ in data["rows"]]
                self.assertGreaterEqual(kinds.count("minute"), 2, kinds)
                self.assertEqual(kinds.count("title"), 1)
                self.assertEqual(kinds.count("scene"), 1)
                self.assertEqual(kinds[-1], "end")
                self.assertEqual(kinds.count("end"), 1)
                self.assertEqual(data["end"], len(kinds))
                title = next(r for k, r in data["rows"] if k == "title")
                self.assertEqual(title["seq"], 1800)
                times = [r["t_ns"] for _, r in data["rows"]]
                self.assertEqual(times, sorted(times))
                for kind, r in data["rows"]:   # synthetic rows sum to within 1 percent of the commit by construction
                    total = sum(r[c] for c in data["sum"])
                    self.assertLess(abs(r["private"] - total) / r["private"], 0.03, (kind, r["private"], total))
                    self.assertEqual(r["vk_heap0_used"], 2000 * MIB)
                last = [a for a in data["alloc"] if a[1] == "test-site"][-1]
                self.assertEqual(last[2:], ["1", str(6 * MIB), "2"])   # the 4 MiB block was freed, 6 MiB is live
                self.assertFalse([a for a in data["alloc"] if a[1] == "small"])
                # the checker passes the runtime's own rows, and fails, by the term's name, a copy with that term dropped
                check = [sys.executable, "-I", str(ROOT / "tools" / "check_memline.py"), str(files[0])]
                done = subprocess.run(check, capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                lines = files[0].read_text(encoding="utf-8").splitlines()
                col = data["columns"].index("img_own_bytes") + 2
                scene = next(i for i, ln in enumerate(lines) if ln.startswith("row\tscene\t"))
                plateau = max(i for i, ln in enumerate(lines[:scene]) if ln.startswith("row\tminute\t"))
                f = lines[plateau].split("\t")
                f[col] = "0"
                lines[plateau] = "\t".join(f)
                dropped = files[0].with_name("dropped.tsv")
                dropped.write_text("\n".join(lines) + "\n", encoding="utf-8")
                done = subprocess.run(check[:-1] + [str(dropped)], capture_output=True, text=True)
                self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
                self.assertIn("term img_own_bytes is 0 here", done.stdout)

    def test_the_gather_reads_what_the_runtime_reads_and_the_cost_is_reported(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "gather")
                avg = int(re.search(r"^gather_avg_us (\d+)$", done.stdout, re.M)[1])
                worst = int(re.search(r"^gather_max_us (\d+)$", done.stdout, re.M)[1])
                print(f"\n[memline {level}] gather: {avg} us on average, {worst} us at most (helper thread, not the render thread)")
                self.assertLess(avg, 2_000_000)

    def test_marking_and_noting_cost_the_caller_next_to_nothing(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "mark")
                per = int(re.search(r"^mark_note_ns (\d+)$", done.stdout, re.M)[1])
                one = int(re.search(r"^mark_title_ns (\d+)$", done.stdout, re.M)[1])
                print(f"\n[memline {level}] mark+note: {per} ns a pair; a title mark: {one} ns")
                self.assertLess(per, 1000)
                self.assertLess(one, 1_000_000)   # the 1 ms bound on the render thread


if __name__ == "__main__":
    unittest.main()
