# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the texture heap instruments and the owner side of lever L1
(runtime/nvn/nvn_texheap.c, oot2/texheap.h; spike S7).

Builds tests/runtime/texheap_test.c with runtime/core and runtime/nvn at -O1 and
-O2 and runs each case in its own process, with OOT2_TEXHEAP, OOT2_TEXHEAP_EVERY
and OOT2_PAGE_RELEASE set as the case needs; the instruments' file is read back
from the folder the case was given. Pools, textures and buffers are made through
the NVN2 entry points; every texel is a number made by the test. What the cases
show:

- off: with the setting off no range and no file exist;
- select: the CPU-written pages wholly inside one live bound texture that no buffer
  and no other texture touches are released, and only those (nothing before a
  completed bind, no edge page, no render target, no texture nobody wrote);
- rewrite: a write to a released page recommits it and the texture is selected
  again only after its next completed bind;
- count: reads of armed pages are counted, guest code told from the runtime, and
  the first read of each texture is a row of the file;
- read_released: a read of a released page is the named fault, and the instruments
  name the texture;
- frames: the pages the CPU wrote in each frame, and the pages the rule would
  select with no page range at all.

The write watch is Windows only: elsewhere the cases that need it print "skip".

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

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
from test_runtime_nvn import CORE, EXIT_FATAL, LEVELS, MODULE_SIZE, NVN, compile_flags, find_compiler  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "texheap_test.c"
HOST_SOURCE = ROOT / "tests" / "runtime" / "pages_host.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"
PAGEGEN_SOURCE = ROOT / "runtime" / "gpu" / "pagegen.c"   # the one reader of the write watch
SWITCHES = ("OOT2_TEXHEAP", "OOT2_TEXHEAP_EVERY", "OOT2_TEXHEAP_TRIGGER", "OOT2_PAGE_RELEASE")


def read_file(folder: Path) -> dict[str, list[dict[str, str]]]:
    """The instruments' file as rows by kind, each a dict by the columns line of its kind."""
    files = sorted(folder.glob("texheap-*.tsv"))
    assert len(files) == 1, files
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert lines[0] == "oot2-texheap 1"
    columns: dict[str, list[str]] = {}
    rows: dict[str, list[dict[str, str]]] = {}
    for line in lines[1:]:
        fields = line.split("\t")
        if fields[0] == "columns":
            columns[fields[1]] = fields[2:]
        elif fields[0] in columns:
            assert len(fields) - 1 == len(columns[fields[0]]), line
            rows.setdefault(fields[0], []).append(dict(zip(columns[fields[0]], fields[1:])))
        else:
            rows.setdefault(fields[0], []).append({"fields": fields[1:]})   # header, end
    return rows


class TexheapTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-texheap-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = ([str(HOST_SOURCE), str(TEST_SOURCE), str(END_SOURCE), str(PAGEGEN_SOURCE)] +
                   sorted(str(p) for p in CORE.glob("*.c")) + sorted(str(p) for p in NVN.glob("*.c")))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"texheap_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*compile_flags(cc), level, *extra, "-I", str(CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, texheap: bool = True, release: str | None = None, trigger: str | None = None):
        env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
        if texheap:
            env["OOT2_TEXHEAP"], env["OOT2_TEXHEAP_EVERY"] = "1", "1"
        if trigger:
            env["OOT2_TEXHEAP_TRIGGER"] = trigger
        if release:
            env["OOT2_PAGE_RELEASE"] = release
        folder = Path(tempfile.mkdtemp(prefix=f"{case}-", dir=self.tmp.name))
        done = subprocess.run([str(self.exes[level]), case, str(folder)], capture_output=True, text=True, timeout=120,
                              env=env)
        return done, folder

    def passing(self, level: str, case: str, **kw):
        done, folder = self.run_case(level, case, **kw)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        if f"skip {case}" in done.stdout:
            self.skipTest(done.stdout.strip())
        self.assertIn(f"ok {case}", done.stdout)
        self.assertNotIn("FATAL", done.stderr)
        return done, folder

    def test_off_makes_no_range_and_no_file(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                _, folder = self.passing(level, "off", texheap=False, release="release")
                self.assertEqual(list(folder.glob("texheap-*")), [])

    def test_selection_rule(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.passing(level, "select", release="release")
                m = re.search(r"^selected (\d+) expected (\d+)$", done.stdout, re.M)
                self.assertIsNotNone(m)
                self.assertGreater(int(m[2]), 10)
                self.assertEqual(m[1], m[2])
                samples = {int(r["seq"]): r for r in read_file(folder)["sample"]}
                self.assertEqual(samples[1]["selected_pages"], "0")   # nothing before a bind
                self.assertEqual(samples[2]["selected_pages"], "0")   # nothing while the binding submission is in flight
                self.assertEqual(samples[3]["selected_pages"], m[2])
                self.assertEqual(samples[3]["releases"], m[2])
                self.assertEqual(samples[3]["partial"], "0")

    def test_the_register_trigger_selects_without_a_bind(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.passing(level, "register", release="release", trigger="register")
                m = re.search(r"^selected (\d+) expected (\d+)$", done.stdout, re.M)
                self.assertIsNotNone(m)
                self.assertGreater(int(m[2]), 10)
                self.assertEqual(m[1], m[2])

    def test_a_rewritten_texture_is_selected_again_after_its_next_bind(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.passing(level, "rewrite", release="release")
                self.assertRegex(done.stdout, r"released (\d+) releases (\d+)")
                samples = {int(r["seq"]): r for r in read_file(folder)["sample"]}
                self.assertEqual(samples[1]["recommits"], "0")
                self.assertEqual(samples[2]["recommits"], "1")   # the write, seen at the present after it
                self.assertEqual(int(samples[3]["releases"]), int(samples[1]["releases"]) + 1)

    def test_reads_of_armed_pages_are_counted_by_who_made_them(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                _, folder = self.passing(level, "count", release="count")
                rows = read_file(folder)
                reads = rows["read"]
                self.assertEqual(len(reads), 2, reads)   # the first read of each of two textures
                self.assertEqual([r["guest"] for r in reads], ["1", "0"])
                self.assertIn("main+", reads[0]["fn"])
                self.assertEqual(reads[1]["fn"], "")
                last = rows["sample"][-1]
                self.assertEqual((last["guest_reads"], last["host_reads"]), ("2", "1"))
                self.assertEqual(last["releases"], "0")
                self.assertGreater(int(last["arms"]), 0)

    def test_a_read_of_a_released_page_names_the_texture(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.run_case(level, "read_released", release="release")
                if "skip read_released" in done.stdout:
                    self.skipTest(done.stdout.strip())
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertIn('range "texheap" was released and a read of it is a fault; read by guest code in',
                              done.stderr)
                self.assertRegex(done.stderr, r"texheap: guest code read a selected page of the texture of serial \d+ "
                                              r"\(\d+ bytes, pool \d+\) at present \d+ in 0x[0-9a-f]+ \(main\+0x200\)")
                self.assertEqual(len(read_file(folder)["read"]), 1)
                self.assertEqual(read_file(folder)["end"][-1]["fields"][0], str(EXIT_FATAL))

    def test_a_bind_of_a_held_queue_belongs_to_its_own_submission(self) -> None:
        # queue 1 is held at a wait before its bind; queue 2 binds another texture and is whole; the CPU releases queue 1,
        # whose bind is replayed after queue 2's. Completing fence value 1 alone selects queue 1's texture and not queue 2's.
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.passing(level, "held_bind", release="release")
                after1 = re.search(r"^after-1 selected (\d+) expected (\d+)$", done.stdout, re.M)
                after2 = re.search(r"^after-2 selected (\d+) expected (\d+)$", done.stdout, re.M)
                self.assertIsNotNone(after1)
                self.assertIsNotNone(after2)
                self.assertGreater(int(after1[2]), 10)
                self.assertEqual(after1[1], after1[2])
                self.assertEqual(after2[1], after2[2])

    def test_a_held_queue_binds_in_the_texture_pool_it_set_not_the_one_another_queue_set_meanwhile(self) -> None:
        # queue 1 sets pool 1 and is held at a wait before its bind of ID 0 (T1 in pool 1); queue 2 sets pool 2, where ID 0
        # is another texture (T5), and binds it; queue 1 then goes on. Completing fence value 1 alone selects T1.
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.passing(level, "held_pool", release="release")
                after1 = re.search(r"^pool-after-1 selected (\d+) expected (\d+)$", done.stdout, re.M)
                after2 = re.search(r"^pool-after-2 selected (\d+) expected (\d+)$", done.stdout, re.M)
                self.assertIsNotNone(after1, done.stdout)
                self.assertIsNotNone(after2, done.stdout)
                self.assertGreater(int(after1[2]), 10)
                self.assertEqual(after1[1], after1[2])
                self.assertGreater(int(after2[2]), int(after1[2]))
                self.assertEqual(after2[1], after2[2])

    def test_a_submission_completes_by_itself_while_an_earlier_one_of_another_queue_is_held(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.passing(level, "held_complete", release="release")
                for line in ("while-held", "released"):
                    m = re.search(rf"^{line} selected (\d+) expected (\d+)$", done.stdout, re.M)
                    self.assertIsNotNone(m, done.stdout)
                    self.assertEqual(m[1], m[2])

    def test_writes_per_frame_and_what_the_rule_would_select(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, folder = self.passing(level, "frames")
                pages = int(re.search(r"^t1_pages (\d+)$", done.stdout, re.M)[1])
                eligible = re.search(r"^t1_eligible (\d+)$", done.stdout, re.M)[1]
                rows = read_file(folder)
                frames = {int(r["seq"]): r for r in rows["frame"]}
                self.assertEqual({k: frames[1][k] for k in ("pages", "pool_pages", "outside")}, {"pages": "0", "pool_pages": "0", "outside": "0"})
                self.assertEqual((frames[2]["pages"], frames[2]["tex_pages"], frames[2]["pools_written"]),
                                 (str(7 + pages), str(pages), "1"))
                self.assertEqual((frames[2]["pool_pages"], frames[2]["outside"]), (str(7 + pages), "0"))
                self.assertEqual(frames[3]["pages"], "0")
                samples = {int(r["seq"]): r for r in rows["sample"]}
                self.assertEqual((samples[3]["eligible_pages"], samples[3]["selected_pages"]), (eligible, "0"))
                header = {r["fields"][0]: r["fields"][1] for r in rows["header"]}
                self.assertEqual(header["page_release"], "off")
                pools = [r for r in rows["pool"] if r["seq"] == "2"]
                self.assertEqual(len(pools), 1)
                # the pool's writes since the sample before: one frame, all of them
                self.assertEqual((pools[0]["window_frames"], pools[0]["window_pages"], pools[0]["window_max"]),
                                 ("1", str(7 + pages), str(7 + pages)))


if __name__ == "__main__":
    unittest.main()
