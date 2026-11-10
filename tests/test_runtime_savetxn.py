# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Can a run that is stopped corrupt a save? The host file layer under the nn::fs save mount (runtime/hos/savetxn.c,
hostfile.c) writes a changed file to a shadow next to it and publishes the shadows at Commit under a marker; recovery at the
next mount finishes or undoes what a stopped run left. No guest, no GPU: tests/runtime/savetxn_test.c drives the real host
files, one process per step, and this file ends those processes at every step and checks what the next one finds.

The save has five files (account.sav, option.sav, slot_00/data.sav, slot_01/data.sav, and slot_02/data.sav made by the
commit). Each holds an OLD content, or a NEW one of another length and pattern, so a mix inside a file reads as BAD:

- (a) write, Commit, read back in a fresh process: every file is NEW, and the folder holds the files and nothing else;
- (b) write with no Commit, the process ends, a fresh process recovers: every file is OLD byte for byte;
- (c) the process ends at EACH step of a Commit (every fsync, the marker, each rename): the next process finds all files OLD
  when the end came before the marker was whole and all NEW after it, never a mix of the two;
- (d) shadows nobody owns, a torn marker and a stray marker temp are removed, and a marker that names a path leaving the folder
  is not followed;
- (e) a second process over the same folder sees the same consistent set, again and again;
- the order of a commit: every shadow synced before the marker, the marker before the first rename;
- a marker written by hand in the documented form is finished forward, one with a wrong checksum is not.

Builds the unit with hostfile.c at -O1 and -O2. Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests
skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOS = ROOT / "runtime" / "hos"
TEST_SOURCE = ROOT / "tests" / "runtime" / "savetxn_test.c"
UNIT_SOURCES = (HOS / "savetxn.c", HOS / "hostfile.c", ROOT / "runtime" / "core" / "plat.c")
LEVELS = ("-O1", "-O2")
DIED = 77   # the test process ends itself with this code (TerminateProcess or _exit)
FILES = ["account.sav", "option.sav", "slot_00/data.sav", "slot_01/data.sav", "slot_02/data.sav"]
ALL_OLD = {**{f: "OLD" for f in FILES[:4]}, FILES[4]: "NONE"}
ALL_NEW = {f: "NEW" for f in FILES}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


def fnv(data: bytes) -> int:
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def marker_text(paths: list[str], good: bool = True) -> bytes:
    body = ("OOT2SAVETXN1\n%d\n" % len(paths) + "".join(p + "\n" for p in paths)).encode()
    return body + b"END %08x\n" % (fnv(body) if good else fnv(body) ^ 1)


class Run:
    def __init__(self, done: subprocess.CompletedProcess) -> None:
        self.code = done.returncode
        self.out = done.stdout
        self.err = done.stderr
        self.states = dict(re.findall(r"^state (\S+) (\w+)$", done.stdout, re.M))
        self.events = re.findall(r"^ev (\S+) (\d+)$", done.stdout, re.M)
        m = re.search(r"^recovered (-?\d+)$", done.stdout, re.M)
        self.recovered = int(m.group(1)) if m else None


class SaveTxnTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-savetxn-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        extra = [] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"savetxn_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(ROOT / "runtime" / "core"), "-I", str(HOS),
                   str(TEST_SOURCE), *map(str, UNIT_SOURCES), "-Wl,--wrap=hf_sync", "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def folder(self) -> Path:
        d = tempfile.TemporaryDirectory(prefix="oot2-savetxn-files-", ignore_cleanup_errors=True)
        self.addCleanup(d.cleanup)
        self.run_case("setup", Path(d.name))
        return Path(d.name)

    def run_case(self, case: str, folder: Path, die: int | None = None, level: str = LEVELS[0]) -> Run:
        args = [str(self.exes[level]), case, str(folder)] + ([str(die)] if die is not None else [])
        return Run(subprocess.run(args, capture_output=True, text=True, timeout=120))

    @staticmethod
    def save_files(folder: Path) -> list[str]:
        root = folder / "save" / "u1"
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    def assertClean(self, folder: Path) -> None:
        names = self.save_files(folder)
        self.assertEqual([n for n in names if n.endswith(".oot2wip") or ".oot2commit" in n], [], names)

    def test_a_write_and_commit_reads_back_the_new_content(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                folder = self.folder()
                done = self.run_case("commit", folder, level=level)
                self.assertEqual(done.code, 0, done.out + done.err)
                self.assertEqual(done.states, ALL_NEW)
                self.assertEqual(self.save_files(folder), sorted(FILES))   # the layout: the files, nothing else
                again = self.run_case("read", folder, level=level)
                self.assertEqual(again.states, ALL_NEW)

    def test_b_a_write_without_commit_leaves_the_old_content(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                folder = self.folder()
                dead = self.run_case("nocommit", folder, level=level)
                self.assertEqual(dead.code, DIED, dead.out + dead.err)
                self.assertTrue(any(n.endswith(".oot2wip") for n in self.save_files(folder)))   # the shadows were really there
                after = self.run_case("read", folder, level=level)
                self.assertEqual(after.states, ALL_OLD, after.out)
                self.assertEqual(after.recovered, 0)
                self.assertEqual(self.save_files(folder), sorted(FILES[:4]))

    def full_run_events(self) -> list[tuple[str, int]]:
        done = self.run_case("commit", self.folder())
        self.assertEqual(done.code, 0, done.out + done.err)
        return [(name, int(i)) for name, i in done.events]

    def test_c_a_stop_at_any_step_of_a_commit_leaves_all_old_or_all_new(self) -> None:
        events = self.full_run_events()
        marker_at = events.index(("marker", 0))
        self.assertGreater(len(events), 12)
        for level in LEVELS:
            for step in range(len(events)):
                with self.subTest(level=level, step=step, event=events[step]):
                    folder = self.folder()
                    dead = self.run_case("commit", folder, die=step, level=level)
                    self.assertEqual(dead.code, DIED, dead.out + dead.err)
                    for round_ in range(2):   # recovery settles it once and leaves it settled
                        after = self.run_case("read", folder, level=level)
                        want = ALL_NEW if step >= marker_at else ALL_OLD
                        self.assertEqual(after.states, want, f"died after {events[step]}: {after.out}")
                        self.assertClean(folder)
                    if step >= marker_at:
                        self.assertEqual(self.save_files(folder), sorted(FILES))

    def test_the_order_of_a_commit_is_sync_then_marker_then_publish(self) -> None:
        events = [name for name, _ in self.full_run_events()]
        begin = events.index("written")
        marker_tmp, marker = events.index("marker-tmp"), events.index("marker")
        publishes = [i for i, n in enumerate(events) if n == "publish"]
        # every shadow of the mount is flushed, and then the marker itself, all before the marker has its name; the host's
        # own flush calls are counted at the link
        self.assertEqual(events[begin:marker_tmp].count("hfsync"), len(FILES) + 1, events)
        self.assertEqual(events[begin:marker_tmp].count("sync"), len(FILES))
        self.assertEqual(events[marker_tmp - 1], "hfsync")   # the marker's own flush is the last one
        self.assertLess(marker_tmp, marker)
        self.assertEqual(len(publishes), len(FILES))
        self.assertLess(marker, min(publishes))   # the marker is whole before the first rename
        self.assertEqual(events[-1], "done")

    def test_d_shadows_nobody_owns_are_removed(self) -> None:
        folder = self.folder()
        root = folder / "save" / "u1"
        (root / "deep" / "er").mkdir(parents=True)
        strays = [root / "slot_00" / "data.sav.oot2wip", root / "fresh.sav.oot2wip", root / "deep" / "er" / "x.oot2wip",
                  root / ".oot2commit.tmp", root / ".oot2commit"]   # the last: a marker that is not one
        for p in strays:
            p.write_bytes(b"half a write\x00\x01")
        after = self.run_case("read", folder)
        self.assertEqual(after.states, ALL_OLD, after.out)
        self.assertEqual(after.recovered, 0)
        for p in strays:
            self.assertFalse(p.exists(), p)
        self.assertEqual(self.save_files(folder), sorted(FILES[:4]))

    def test_a_marker_that_leaves_the_folder_is_not_followed(self) -> None:
        folder = self.folder()
        root = folder / "save" / "u1"
        (folder / "save" / "evil.oot2wip").write_bytes(b"outside")
        for names in (["../evil"], ["account.sav", "/etc/x"], ["a/../../evil"], ["account.sav", ""]):
            (root / ".oot2commit").write_bytes(marker_text(names))
            (root / "account.sav.oot2wip").write_bytes(b"new account")
            after = self.run_case("read", folder)
            self.assertEqual(after.states, ALL_OLD, (names, after.out))
            self.assertFalse((folder / "save" / "evil").exists(), names)
            self.assertFalse((root / ".oot2commit").exists())
            (folder / "save" / "evil.oot2wip").write_bytes(b"outside")

    def test_a_marker_with_a_wrong_checksum_is_not_applied(self) -> None:
        folder = self.folder()
        root = folder / "save" / "u1"
        (root / "account.sav.oot2wip").write_bytes(b"new account")
        (root / "option.sav.oot2wip").write_bytes(b"new option")
        (root / ".oot2commit").write_bytes(marker_text(["account.sav", "option.sav"], good=False))
        after = self.run_case("read", folder)
        self.assertEqual(after.states, ALL_OLD, after.out)
        self.assertClean(folder)
        # a marker cut short
        (root / "account.sav.oot2wip").write_bytes(b"new account")
        (root / ".oot2commit").write_bytes(marker_text(["account.sav", "option.sav"])[:-12])
        after = self.run_case("read", folder)
        self.assertEqual(after.states, ALL_OLD, after.out)
        self.assertClean(folder)

    def test_a_marker_in_the_documented_form_is_finished_forward(self) -> None:
        folder = self.folder()
        root = folder / "save" / "u1"
        (root / "account.sav.oot2wip").write_bytes(b"new account")
        (root / "slot_00" / "data.sav.oot2wip").write_bytes(b"new slot")
        (root / "fresh.sav.oot2wip").write_bytes(b"made by the commit")
        # option.sav was already renamed when the run ended: its shadow is gone and it holds what it holds
        (root / ".oot2commit").write_bytes(marker_text(["account.sav", "option.sav", "slot_00/data.sav", "fresh.sav"]))
        after = self.run_case("read", folder)
        self.assertEqual(after.recovered, 3, after.out)
        self.assertEqual((root / "account.sav").read_bytes(), b"new account")
        self.assertEqual((root / "slot_00" / "data.sav").read_bytes(), b"new slot")
        self.assertEqual((root / "fresh.sav").read_bytes(), b"made by the commit")
        self.assertEqual(after.states["option.sav"], "OLD")
        self.assertClean(folder)

    def test_e_a_second_process_sees_the_same_set(self) -> None:
        folder = self.folder()
        self.assertEqual(self.run_case("commit", folder, die=len(self.full_run_events()) - 3).code, DIED)
        first = self.run_case("read", folder)
        for _ in range(3):
            self.assertEqual(self.run_case("read", folder).states, first.states)
        self.assertEqual(first.states, ALL_NEW)

    def test_a_discarded_write_is_gone_and_a_view_shows_the_shadow(self) -> None:
        folder = self.folder()
        gone = self.run_case("discard", folder)
        self.assertEqual(gone.code, 0, gone.out + gone.err)
        self.assertEqual(gone.states, ALL_OLD)
        self.assertClean(folder)
        view = self.run_case("view", self.folder())
        self.assertEqual(view.code, 0, view.out + view.err)
        for line in ("pending serial 1", "real is OLD", "after commit serial 0", "real is NEW", "second shadow 1",
                     "discarded serial 0", "create over existing refused"):
            self.assertIn(line, view.out)
        self.assertRegex(view.out, r"shadow size (\d+) want \1\b")

    def test_the_glue_in_the_file_service_calls_the_unit(self) -> None:
        # The guest-side calls are not made here (no guest): the file service must still route a save mount through the unit.
        code = "\n".join(l.split("//", 1)[0] for l in (HOS / "ov_fs.c").read_text(encoding="utf-8").splitlines())
        for call in ("st_recover(", "st_commit(", "st_begin(", "st_begin_create(", "st_serial(", "st_discard_prefix(",
                     "st_publish_prefix("):
            self.assertIn(call, code)

    def test_a_save_write_that_cannot_start_its_shadow_returns_an_error(self) -> None:
        # A full disk, a name too long for a shadow, or a file deleted under an open handle must not end the process:
        # the first two come back as the write's result, the last keeps writing to the handle that is open.
        code = "\n".join(l.split("//", 1)[0] for l in (HOS / "ov_fs.c").read_text(encoding="utf-8").splitlines())
        start = code.index("static HostFile *file_bound(")
        body = code[start:code.index("\n}\n", start)]
        self.assertNotIn("rt_fatal", body)
        self.assertIn("rc == HF_NOT_FOUND", body)
        self.assertIn("*err = rc", body)
        for name in ("write_file(", "set_file_size("):
            at = code.index("static uint32_t " + name)
            fn = code[at:code.index("\n}\n", at)]
            self.assertRegex(fn, r"file_bound\(h, NULL, 1, &berr\)", name)
            self.assertIn("fs_result(berr)", fn, name)


if __name__ == "__main__":
    unittest.main()
