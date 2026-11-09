# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of tools/fetch_opus.py, which fetches libopus's release, builds it into a kit and never commits any of it.

None of these needs a network: a download is held to its pinned SHA-256 and a bad one is deleted; the source lists are
read out of the release's own *.mk files (a variable that is missing, an empty one, a file that is not there and a file
named twice are refused by name); the Ninja file compiles each file once with the defines the decoder needs; a kit the
script laid out is current, and a changed, added or recipe-different one is not; the whole build runs on a synthetic
archive with the real compiler (a release whose archive would unpack outside its folder is refused); the command line
leaves a current kit alone, and exits 1 on a failed build and 2 with no tool; the notice row and the licence text exist, and
both workflows build the kit before the tests, the Windows one requiring the decoder test.

The decoder itself is tested in tests/test_runtime_opus.py, against a kit built for real.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import fetch_opus as fo  # noqa: E402

HEX64 = re.compile(r"[0-9a-f]{64}")

CELT_MK = "CELT_SOURCES = \\\ncelt/a.c \\\ncelt/b.c\n\nCELT_SOURCES_X86_RTCD = \\\ncelt/x86/x.c\n"
SILK_MK = ("SILK_SOURCES = \\\nsilk/c.c \\\nsilk/d.c\n\nSILK_SOURCES_X86_RTCD = \\\nsilk/x86/y.c\n\n"
           "SILK_SOURCES_FLOAT = \\\nsilk/float/e.c\n")
OPUS_MK = "OPUS_SOURCES = \\\nsrc/f.c\n\nOPUS_SOURCES_FLOAT = \\\nsrc/g.c\n"
FILES = ["celt/a.c", "celt/b.c", "silk/c.c", "silk/d.c", "silk/float/e.c", "src/f.c", "src/g.c"]


def fake_release(root: Path, mk: dict[str, str] | None = None, skip: str | None = None) -> Path:
    """A release folder of synthetic files: the three lists, the sources they name, the headers and COPYING."""
    mk = {"celt_sources.mk": CELT_MK, "silk_sources.mk": SILK_MK, "opus_sources.mk": OPUS_MK, **(mk or {})}
    for name, text in mk.items():
        (root / name).write_text(text, encoding="utf-8")
    for rel in FILES + ["celt/x86/x.c", "silk/x86/y.c"]:
        if rel != skip:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("int " + re.sub(r"\W", "_", rel[:-2]) + " = 1;\n", encoding="utf-8")
    (root / "include").mkdir(exist_ok=True)
    for h in fo.HEADERS:
        (root / "include" / h).write_text(f"/* synthetic {h} */\n", encoding="utf-8")
    (root / "COPYING").write_text("synthetic licence\n", encoding="utf-8")
    return root


def fake_kit(dest: Path) -> Path:
    """dest laid out as the script lays out a kit, from synthetic files, and recorded the way it records one."""
    (dest / "include" / "opus").mkdir(parents=True)
    (dest / "lib").mkdir()
    (dest / "licenses").mkdir()
    for h in fo.HEADERS:
        (dest / "include" / "opus" / h).write_text(f"// synthetic {h}\n", encoding="utf-8")
    (dest / "lib" / "libopus.a").write_bytes(b"!<arch>\n")
    (dest / "licenses" / "opus-COPYING.txt").write_text("synthetic\n", encoding="utf-8")
    (dest / fo.RECORD).write_text(json.dumps({"pins": fo.pins(), "files": fo.tree_hashes(dest)}, indent=1) + "\n",
                                  encoding="utf-8")
    return dest.resolve()


class PinShape(unittest.TestCase):
    def test_the_pin_has_a_version_an_address_and_a_digest(self) -> None:
        self.assertRegex(fo.VERSION, r"^1\.5\.\d+$")
        self.assertEqual(fo.OPUS.url, f"https://downloads.xiph.org/releases/opus/opus-{fo.VERSION}.tar.gz")
        self.assertRegex(fo.OPUS.sha256, HEX64.pattern)
        self.assertEqual(fo.DEFAULT_ENV, "OPUS_DIR")
        self.assertEqual(fo.pins()["opus"], [fo.OPUS.url, fo.OPUS.sha256])
        self.assertEqual(fo.pins()["defines"], list(fo.DEFINES))
        self.assertEqual(fo.pins()["recipe"], hashlib.sha256((ROOT / "tools" / "fetch_opus.py").read_bytes()).hexdigest())

    def test_the_build_is_the_float_decoder_with_the_encoder_the_tests_use(self) -> None:
        lists = [name for _, name in fo.SOURCE_LISTS]
        for name in ("CELT_SOURCES", "SILK_SOURCES", "SILK_SOURCES_FLOAT", "OPUS_SOURCES", "OPUS_SOURCES_FLOAT"):
            self.assertIn(name, lists)
        for define in ("OPUS_BUILD", "VAR_ARRAYS", "HAVE_LRINT", "HAVE_LRINTF"):
            self.assertIn(define, fo.DEFINES)
        for forbidden in ("FIXED_POINT", "OPUS_HAVE_RTCD", "ENABLE_DEEP_PLC", "ENABLE_DRED", "ENABLE_OSCE"):
            self.assertNotIn(forbidden, fo.DEFINES)
        self.assertEqual(set(fo.HEADERS), {"opus.h", "opus_defines.h", "opus_types.h", "opus_multistream.h"})


class DownloadTest(unittest.TestCase):
    def test_a_download_is_held_to_its_digest_and_a_bad_one_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-fetch-", ignore_cleanup_errors=True) as tmp:
            src = Path(tmp) / "archive.bin"
            src.write_bytes(b"not an Opus release")
            got = Path(tmp) / "got.bin"
            fo.download(fo.Archive(src.as_uri(), hashlib.sha256(b"not an Opus release").hexdigest()), got)
            self.assertEqual(got.read_bytes(), b"not an Opus release")
            with self.assertRaises(fo.FetchError) as caught:
                fo.download(fo.Archive(src.as_uri(), "0" * 64), Path(tmp) / "bad.bin")
            self.assertIn("SHA-256", str(caught.exception))
            self.assertFalse((Path(tmp) / "bad.bin").exists(), "a download that failed the check was kept")
            with self.assertRaises(fo.FetchError) as caught:
                fo.download(fo.Archive((Path(tmp) / "missing.bin").as_uri(), "0" * 64), Path(tmp) / "none.bin")
            self.assertIn("download failed", str(caught.exception))


class SourceListTest(unittest.TestCase):
    def test_a_variable_names_its_files_and_not_those_of_a_longer_name(self) -> None:
        self.assertEqual(fo.mk_list(CELT_MK, "CELT_SOURCES"), ["celt/a.c", "celt/b.c"])
        self.assertEqual(fo.mk_list(SILK_MK, "SILK_SOURCES"), ["silk/c.c", "silk/d.c"])
        self.assertEqual(fo.mk_list(SILK_MK, "SILK_SOURCES_FLOAT"), ["silk/float/e.c"])
        self.assertEqual(fo.mk_list(OPUS_MK, "OPUS_SOURCES"), ["src/f.c"])

    def test_a_variable_that_is_missing_or_empty_is_refused(self) -> None:
        for text in (CELT_MK, "CELT_SOURCES = \\\n\nOTHER = x.c\n"):
            with self.assertRaises(fo.FetchError) as caught:
                fo.mk_list(text, "SILK_SOURCES" if text is CELT_MK else "CELT_SOURCES")
            self.assertIn("no files", str(caught.exception))

    def test_the_files_of_every_list_come_in_order_and_must_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            self.assertEqual(fo.source_files(fake_release(Path(tmp))), FILES)

    def test_a_listed_file_the_release_lacks_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            with self.assertRaises(fo.FetchError) as caught:
                fo.source_files(fake_release(Path(tmp), skip="silk/d.c"))
            self.assertIn("silk/d.c", str(caught.exception))
            self.assertIn("silk_sources.mk", str(caught.exception))

    def test_a_file_named_twice_and_a_missing_list_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            root = fake_release(Path(tmp), {"opus_sources.mk": OPUS_MK.replace("src/g.c", "celt/a.c")})
            with self.assertRaises(fo.FetchError) as caught:
                fo.source_files(root)
            self.assertIn("celt/a.c is named twice", str(caught.exception))
            (root / "celt_sources.mk").unlink()
            with self.assertRaises(fo.FetchError) as caught:
                fo.source_files(root)
            self.assertIn("no celt_sources.mk", str(caught.exception))


class NinjaTest(unittest.TestCase):
    def test_each_file_is_compiled_once_with_the_defines_and_the_include_folders(self) -> None:
        text = fo.ninja_text("clang", "llvm-ar", Path("/some/opus"), FILES)
        builds = re.findall(r"^build (obj/\S+\.o): cc (\S+)", text, re.M)
        self.assertEqual(len(builds), len(FILES))
        self.assertEqual(len({obj for obj, _ in builds}), len(FILES))   # one object each: two files of one stem do not collide
        self.assertEqual([src.split("/some/opus/")[1] for _, src in builds], FILES)
        for define in fo.DEFINES:
            self.assertIn(f"-D{define}", text)
        self.assertIn(f'-DPACKAGE_VERSION=\\"{fo.VERSION}\\"', text)
        for d in ("include", "celt", "silk", "silk/float", "src"):
            self.assertIn(f"-I/some/opus/{d}", text)
        self.assertIn("build libopus.a: ar " + " ".join(obj for obj, _ in builds), text)
        self.assertIn("-std=c99", text)

    def test_a_path_with_a_space_or_a_colon_is_escaped_in_a_build_line(self) -> None:
        text = fo.ninja_text("clang", "llvm-ar", Path("C:/a b/opus"), ["celt/a.c"])
        self.assertIn("build obj/celt_a.o: cc C$:/a$ b/opus/celt/a.c", text)

    def test_object_names_keep_the_folder(self) -> None:
        self.assertEqual(fo.object_name("celt/bands.c"), "obj/celt_bands.o")
        self.assertEqual(fo.object_name("silk/float/burg_modified_FLP.c"), "obj/silk_float_burg_modified_FLP.o")


class KitTest(unittest.TestCase):
    def test_a_kit_the_script_laid_out_is_current_and_a_changed_or_added_one_is_not(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            kit = fake_kit(Path(tmp) / "kit")
            self.assertTrue(fo.is_current(kit))
            (kit / "lib" / "libopus.a").write_bytes(b"!<arch>\nx")
            self.assertFalse(fo.is_current(kit))   # a file changed
            kit = fake_kit(Path(tmp) / "kit2")
            (kit / "lib" / "extra.a").write_bytes(b"x")
            self.assertFalse(fo.is_current(kit))   # a file added
            kit = fake_kit(Path(tmp) / "kit3")
            (kit / "include" / "opus" / "opus.h").unlink()
            self.assertFalse(fo.is_current(kit))   # a file removed
            self.assertFalse(fo.is_current(Path(tmp) / "no kit here"))

    def test_a_kit_of_another_pin_or_another_recipe_is_not_current(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            kit = fake_kit(Path(tmp) / "kit")
            for key, value in (("opus", ["https://example.invalid/opus.tar.gz", "0" * 64]), ("recipe", "0" * 64),
                               ("defines", ["OPUS_BUILD"])):
                record = json.loads((kit / fo.RECORD).read_text(encoding="utf-8"))
                record["pins"][key] = value
                (kit / fo.RECORD).write_text(json.dumps(record), encoding="utf-8")
                self.assertFalse(fo.is_current(kit), key)
                (kit / fo.RECORD).write_text(json.dumps({"pins": fo.pins(), "files": fo.tree_hashes(kit)}), encoding="utf-8")
                self.assertTrue(fo.is_current(kit))
            (kit / fo.RECORD).write_text("not json", encoding="utf-8")
            self.assertFalse(fo.is_current(kit))


def tool(name: str) -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang")) if name == "cc" else shutil.which(os.environ.get("OOT2_NINJA", "ninja"))


def synthetic_archive(path: Path, escape: bool = False) -> None:
    """An archive laid out as the release's is: opus-VERSION/ with the lists, the sources, the headers and COPYING."""
    with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / f"opus-{fo.VERSION}"
        root.mkdir()
        fake_release(root)
        with tarfile.open(path, "w:gz") as t:
            t.add(root, arcname=f"opus-{fo.VERSION}")
            if escape:
                info = tarfile.TarInfo("../escape.txt")
                data = b"x"
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))


class BuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.archive = self.dir / "release.tar.gz"

    def fake_download(self, archive, path) -> None:
        shutil.copyfile(self.archive, path)

    def toolchain(self) -> tuple[str, str, str]:
        cc, ninja = tool("cc"), tool("ninja")
        if not (cc and ninja):
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang or Ninja was found")
            self.skipTest("no clang or Ninja on PATH")
        ar = Path(cc).parent / ("llvm-ar.exe" if os.name == "nt" else "llvm-ar")
        return cc, str(ar) if ar.is_file() else (shutil.which("llvm-ar") or shutil.which("ar") or "ar"), ninja

    def test_a_synthetic_release_is_built_and_laid_out_as_a_kit(self) -> None:
        cc, ar, ninja = self.toolchain()
        synthetic_archive(self.archive)
        dest = self.dir / "kit"
        with mock.patch.object(fo, "download", self.fake_download):
            lines = fo.build(dest, cc, ar, ninja)
        self.assertTrue(any("7 source files" in line for line in lines), lines)
        self.assertGreater((dest / "lib" / "libopus.a").stat().st_size, 8)
        for h in fo.HEADERS:
            self.assertTrue((dest / "include" / "opus" / h).is_file(), h)
        self.assertEqual((dest / "licenses" / "opus-COPYING.txt").read_text(encoding="utf-8"), "synthetic licence\n")
        self.assertTrue(fo.is_current(dest))
        with mock.patch.object(fo, "download", self.fake_download):   # a folder this script laid out is replaced
            fo.build(dest, cc, ar, ninja)
        self.assertTrue(fo.is_current(dest))

    def test_a_folder_that_is_not_the_scripts_own_is_refused_untouched(self) -> None:
        dest = self.dir / "mine"
        dest.mkdir()
        (dest / "notes.txt").write_text("mine", encoding="utf-8")
        with mock.patch.object(fo, "download", side_effect=AssertionError("no download for a refused folder")):
            with self.assertRaises(fo.FetchError) as caught:
                fo.build(dest, "clang", "ar", "ninja")
        self.assertIn("not a folder this script laid out", str(caught.exception))
        self.assertEqual((dest / "notes.txt").read_text(encoding="utf-8"), "mine")

    def test_an_archive_that_would_unpack_outside_its_folder_is_refused(self) -> None:
        cc, ar, ninja = self.toolchain()
        synthetic_archive(self.archive, escape=True)
        with mock.patch.object(fo, "download", self.fake_download):
            with self.assertRaises(fo.FetchError):
                fo.build(self.dir / "kit", cc, ar, ninja)
        self.assertFalse((self.dir / "escape.txt").exists())
        self.assertFalse((self.dir / "kit").exists() and any((self.dir / "kit").iterdir()))

    def test_a_build_that_fails_says_so_and_leaves_no_kit(self) -> None:
        cc, ar, ninja = self.toolchain()
        synthetic_archive(self.archive)
        with mock.patch.object(fo, "download", self.fake_download), mock.patch.object(fo, "ninja_text",
                                                                                      lambda *a: "rule x\n  command = exit 3\nbuild a: x\ndefault a\n"):
            with self.assertRaises(fo.FetchError) as caught:
                fo.build(self.dir / "kit", cc, ar, ninja)
        self.assertIn("building libopus failed", str(caught.exception))
        self.assertFalse((self.dir / "kit" / fo.RECORD).exists())


class MainTest(unittest.TestCase):
    def run_main(self, *args: str, **patches) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fo.main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_a_current_kit_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            kit = fake_kit(Path(tmp) / "kit")
            with mock.patch.object(fo, "build", side_effect=AssertionError("a current kit is not built again")):
                code, out, _ = self.run_main("--dest", str(kit))
            self.assertEqual(code, 0)
            self.assertIn("already holds the pinned build", out)

    def test_no_tool_is_exit_2_and_a_failed_build_is_exit_1(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-opus-", ignore_cleanup_errors=True) as tmp:
            dest = Path(tmp) / "kit"
            with mock.patch.object(fo.shutil, "which", lambda name: None):
                code, _, err = self.run_main("--dest", str(dest))
            self.assertEqual(code, 2)
            self.assertIn("no C compiler found", err)
            with mock.patch.object(fo.shutil, "which", lambda name: name), mock.patch.object(
                    fo, "build", side_effect=fo.FetchError("the archive did not match")):
                code, _, err = self.run_main("--dest", str(dest), "--ar", "ar")
            self.assertEqual(code, 1)
            self.assertIn("the archive did not match", err)
            with mock.patch.object(fo.shutil, "which", lambda name: name), mock.patch.object(fo, "build", lambda *a: ["one step"]):
                code, out, _ = self.run_main("--dest", str(dest), "--ar", "ar")
            self.assertEqual(code, 0)
            self.assertIn("set OPUS_DIR to it", out)


class NoticesAndWorkflowsTest(unittest.TestCase):
    @staticmethod
    def workflow(name: str) -> str:
        return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")

    def test_libopus_is_in_the_notices_with_its_pin_and_licence_text(self) -> None:
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.startswith("| libopus")]
        self.assertEqual(len(rows), 1, rows)
        row = rows[0]
        self.assertIn(fo.VERSION, row)
        self.assertIn(fo.OPUS.sha256, row)   # the digest the script holds the archive to is the one the notice states
        self.assertIn(fo.OPUS.url, row)
        self.assertIn("BSD-3-Clause", row)
        self.assertIn("tools/fetch_opus.py", row)
        for name in re.findall(r"\]\((licenses/[^)]+)\)", row):
            self.assertTrue((ROOT / name).is_file(), f"{name} is named by the libopus row and is missing")
        licence = (ROOT / "licenses" / "opus-BSD-3-Clause.txt").read_text(encoding="utf-8")
        self.assertIn("Xiph.Org, Skype Limited, Octasic", licence)
        self.assertIn("Redistribution and use in source and binary forms", licence)
        self.assertIn("Neither the name of Internet Society, IETF or IETF Trust", licence)
        script = (ROOT / "tools" / "fetch_opus.py").read_text(encoding="utf-8")
        self.assertIn(fo.OPUS.sha256, script)

    def test_windows_builds_the_kit_before_the_tests_and_requires_the_decoder_test(self) -> None:
        text = self.workflow("windows-build.yml")
        self.assertIn('OOT2_REQUIRE_OPUS: "1"', text)
        self.assertIn("python tools/fetch_opus.py --dest $kit", text)
        self.assertIn('"OPUS_DIR=$kit" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV', text)
        self.assertIn("key: opus-kit-${{ runner.os }}-${{ env.LLVM_MINGW }}-${{ hashFiles('tools/fetch_opus.py') }}", text)
        self.assertLess(text.index("python tools/fetch_opus.py"), text.index("python tools/run_tests.py"))

    def test_linux_builds_the_kit_before_the_tests_and_may_skip_the_decoder_test_with_a_reason(self) -> None:
        text = self.workflow("content-guard.yml")
        self.assertIn('python tools/fetch_opus.py --dest "$RUNNER_TEMP/opus-kit"', text)
        self.assertIn('echo "OPUS_DIR=$RUNNER_TEMP/opus-kit" >> "$GITHUB_ENV"', text)
        self.assertIn("key: opus-kit-${{ runner.os }}-${{ hashFiles('tools/fetch_opus.py') }}", text)
        self.assertLess(text.index("python tools/fetch_opus.py"), text.index("python tools/run_tests.py"))
        self.assertNotIn("OOT2_REQUIRE_OPUS", text)   # the Linux job is not a verdict on decoding


if __name__ == "__main__":
    unittest.main()
