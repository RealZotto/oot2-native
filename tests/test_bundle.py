# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the release bundle's builder (tools/bundle.py) and for how prepare finds a bundle's tools
(tools/bundle_env.py).

Every input here is synthetic: archives and files written by the test, never a download, never a game file. The real
notices table is read, so a component added to the builder without its row fails here.

Run: python -m unittest discover -s tests -p test_bundle.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import bundle  # noqa: E402
import bundle_env  # noqa: E402

NOTICES_TEXT = (ROOT / bundle.NOTICES).read_text(encoding="utf-8")


def row(component: str) -> str:
    return f"| `bundle/{component}/`: something | a | b | c | d | No |"


ALL_ROWS = "| File | Source |\n| --- | --- |\n" + "\n".join(row(c) for c in bundle.COMPONENTS) + "\n"


def tmp() -> Path:
    d = Path(tempfile.mkdtemp(prefix="oot2-bundle-test-"))
    return d


def small_stage(root: Path) -> bundle.Stage:
    """A stage with one file for two components and a licence text for each, the least a bundle can be."""
    st = bundle.Stage(root)
    st.put("ninja", "bin/ninja.exe", b"MZ" + b"\0" * 64)
    st.put("ninja", "licenses/ninja/LICENSE", b"Apache-2.0 text\n")
    st.put("oot2", "tools/oot2.py", b"print('hi')\n")
    return st


class NoticesTable(unittest.TestCase):
    def test_every_component_has_a_row_in_the_real_table(self):
        have = bundle.notice_components(NOTICES_TEXT)
        self.assertEqual(sorted(set(bundle.COMPONENTS) - have), [])

    def test_a_row_is_read_from_the_first_cell_only(self):
        text = "| `x` | mentions `bundle/ninja/` in the second cell |\n"
        self.assertEqual(bundle.notice_components(text), set())
        self.assertEqual(bundle.notice_components(row("ninja") + "\n"), {"ninja"})


class RefusesAFileWithoutARow(unittest.TestCase):
    """The counter-run: a bundle whose file has no notices row is refused, the file named."""

    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_complete_table_passes(self):
        rows = small_stage(self.dir).listing()
        bundle.check_notices(rows, ALL_ROWS)

    def test_one_row_removed_is_refused_and_names_the_file(self):
        rows = small_stage(self.dir).listing()
        without = "\n".join(l for l in ALL_ROWS.splitlines() if "bundle/ninja/" not in l) + "\n"
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_notices(rows, without)
        self.assertIn("bin/ninja.exe", str(cm.exception))
        self.assertIn("bundle/ninja/", str(cm.exception))

    def test_the_real_table_with_one_row_removed_is_refused(self):
        rows = small_stage(self.dir).listing()
        cut = "\n".join(l for l in NOTICES_TEXT.splitlines()
                        if not (l.startswith("|") and "`bundle/ninja/`" in l.split("|")[1])) + "\n"
        self.assertNotIn("| `bundle/ninja/`", cut)
        with self.assertRaises(bundle.BundleError):
            bundle.check_notices(rows, cut)
        bundle.check_notices(rows, NOTICES_TEXT)

    def test_a_component_without_a_licence_text_is_refused(self):
        st = bundle.Stage(self.dir)
        st.put("ninja", "bin/ninja.exe", b"MZ")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_notices(st.listing(), ALL_ROWS)
        self.assertIn("ninja", str(cm.exception))

    def test_a_file_no_component_put_there_is_refused(self):
        st = small_stage(self.dir)
        (self.dir / "stray.bin.txt").write_text("x")
        with self.assertRaises(bundle.BundleError) as cm:
            st.listing()
        self.assertIn("stray.bin.txt", str(cm.exception))

    def test_a_component_nobody_declared_is_refused(self):
        with self.assertRaises(bundle.BundleError):
            bundle.Stage(self.dir).put("made-up", "x.txt", b"x")

    def test_a_path_cannot_leave_the_bundle_or_repeat(self):
        st = bundle.Stage(self.dir)
        st.put("oot2", "a.txt", b"1")
        with self.assertRaises(bundle.BundleError):
            st.put("oot2", "a.txt", b"2")
        with self.assertRaises(bundle.BundleError):
            st.put("oot2", "../out.txt", b"2")


class ContentRules(unittest.TestCase):
    """The rules of check_content that name the game apply to every file of the bundle."""

    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def write(self, rel: str, data: bytes) -> Path:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def test_an_executable_and_a_text_file_pass(self):
        self.assertEqual(bundle.guard_file("bin/tool.exe", self.write("bin/tool.exe", b"MZ\x90\0" + b"\1" * 5000)), [])
        self.assertEqual(bundle.guard_file("a.py", self.write("a.py", b"print(1)\n")), [])

    def test_a_game_container_signature_is_refused(self):
        name = "NSO" + "0"
        found = bundle.guard_file("x.dat", self.write("x.dat", name.encode() + b"\0" * 64))
        self.assertTrue(any("NSO" in f for f in found), found)

    def test_a_module_name_a_key_suffix_and_a_game_folder_are_refused(self):
        self.assertTrue(bundle.guard_file("llvm/main", self.write("llvm/main", b"x")))
        self.assertTrue(bundle.guard_file("x.nca", self.write("x.nca", b"x")))
        self.assertTrue(bundle.guard_file("romfs/a.txt", self.write("romfs/a.txt", b"x")))

    def test_the_generated_code_marker_is_refused_anywhere_in_a_big_file(self):
        marker = bundle.MARKER
        data = b"\0" * (9 << 20) + marker + b"\0" * 10
        found = bundle.guard_file("big.dll", self.write("big.dll", data))
        self.assertTrue(any("generated" in f for f in found), found)

    def test_the_marker_split_across_two_reads_is_found(self):
        marker = bundle.MARKER
        cut = 8 << 20
        data = b"\0" * (cut - 5) + marker + b"\0" * 10
        self.assertTrue(bundle.guard_file("split.dll", self.write("split.dll", data)))

    def test_a_key_sized_value_is_refused_except_in_the_kit_record_named_for_it(self):
        text = ('{"recipe": "' + "ab" * 32 + '"}').encode()
        self.assertTrue(bundle.guard_file("notes.json", self.write("notes.json", text)))
        rel = "runtime/sdl3/fetch_sdl3.json"
        self.assertEqual(bundle.guard_file(rel, self.write(rel, text)), [])
        key = ("-----BEGIN " + "RSA PRIVATE KEY-----" + chr(10) + "abc" + chr(10)).encode()
        self.assertTrue(bundle.guard_file(rel, self.write(rel, key)), "a private key is refused even there")

    def test_a_private_key_in_a_text_file_is_refused(self):
        key = ("-----BEGIN " + "RSA PRIVATE KEY-----\nabc\n").encode()
        self.assertTrue(bundle.guard_file("k.txt", self.write("k.txt", key)))

    def test_an_archive_passes_only_where_the_bundle_carries_one(self):
        zipped = b"PK\x03\x04" + b"\0" * 64
        self.assertEqual(bundle.guard_file("python/python314.zip", self.write("python/python314.zip", zipped)), [])
        self.assertTrue(bundle.guard_file("data/other.zip", self.write("data/other.zip", zipped)))

    def test_guard_names_the_file_when_the_bundle_is_refused(self):
        self.write("a.nca", b"x")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.guard(self.dir, [{"path": "a.nca"}])
        self.assertIn("a.nca", str(cm.exception))


class FinishAndZip(unittest.TestCase):
    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_finish_writes_the_manifest_and_the_contents_list(self):
        st = small_stage(self.dir / "b")
        rows = bundle.finish(st, "v", "abc123", ALL_ROWS, say=lambda s: None)
        manifest = json.loads((self.dir / "b" / "bundle.json").read_text(encoding="utf-8"))
        self.assertEqual([r["path"] for r in manifest["files"]], [r["path"] for r in rows])
        self.assertEqual(manifest["source_commit"], "abc123")
        self.assertEqual(manifest["total_bytes"], sum(r["bytes"] for r in rows))
        self.assertIn("bin/ninja.exe", [r["path"] for r in rows])
        tsv = (self.dir / "b" / "CONTENTS.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(tsv[0], "path\tbytes\tsha256\tcomponent\tnotices_row")
        self.assertIn("bin/ninja.exe", "\n".join(tsv))
        self.assertIn("\tninja\tbundle/ninja/", "\n".join(tsv))
        self.assertTrue(all(len(r["sha256"]) == 64 for r in rows))

    def test_finish_refuses_before_it_writes_anything_when_a_row_is_missing(self):
        st = small_stage(self.dir / "b")
        with self.assertRaises(bundle.BundleError):
            bundle.finish(st, "v", "abc", "| File |\n", say=lambda s: None)
        self.assertFalse((self.dir / "b" / "bundle.json").exists())

    def test_finish_refuses_a_game_file_even_with_every_row(self):
        st = small_stage(self.dir / "b")
        st.put("oot2", "tools/romfs/x.txt", b"x")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.finish(st, "v", "abc", ALL_ROWS, say=lambda s: None)
        self.assertIn("romfs", str(cm.exception))

    def test_the_zip_is_deterministic(self):
        st = small_stage(self.dir / "b")
        bundle.finish(st, "v", "abc", ALL_ROWS, say=lambda s: None)
        a, b = self.dir / "a.zip", self.dir / "b.zip"
        bundle.write_zip(self.dir / "b", a)
        os.utime(self.dir / "b" / "tools" / "oot2.py", (1, 1))
        bundle.write_zip(self.dir / "b", b)
        self.assertEqual(bundle.sha256_file(a), bundle.sha256_file(b))
        with zipfile.ZipFile(a) as zf:
            names = zf.namelist()
            self.assertEqual(names, sorted(names))
            self.assertTrue(all(n.startswith("b/") for n in names))
            self.assertEqual({i.date_time for i in zf.infolist()}, {bundle.ZIP_DATE})

    def test_summary_counts_by_component(self):
        rows = small_stage(self.dir / "b").listing()
        lines = bundle.summary(rows)
        self.assertTrue(any(l.strip().startswith("ninja") for l in lines))
        self.assertIn("total", lines[-1])


class FetchChecksTheHash(unittest.TestCase):
    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_a_file_that_matches_is_used(self):
        data = b"payload"
        pin = bundle.Pin("x.bin", "http://invalid.invalid/x", hashlib.sha256(data).hexdigest())
        (self.dir / "x.bin").write_bytes(data)
        self.assertEqual(bundle.fetch(pin, self.dir, say=lambda s: None), self.dir / "x.bin")

    def test_a_file_that_does_not_match_is_deleted_and_refused(self):
        pin = bundle.Pin("x.bin", "http://invalid.invalid/x", "0" * 64)
        (self.dir / "x.bin").write_bytes(b"payload")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.fetch(pin, self.dir, say=lambda s: None)
        self.assertIn("deleted", str(cm.exception))
        self.assertFalse((self.dir / "x.bin").exists())

    def test_a_download_that_fails_leaves_no_part_file(self):
        pin = bundle.Pin("y.bin", "http://invalid.invalid/y", "0" * 64)
        with mock.patch("urllib.request.urlopen", side_effect=OSError("no network")):
            with self.assertRaises(bundle.BundleError):
                bundle.fetch(pin, self.dir, say=lambda s: None)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_every_pin_is_a_sha256_and_a_https_address(self):
        for pin in bundle.PINS:
            self.assertRegex(pin.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue(pin.url.startswith("https://"), pin.url)
        self.assertRegex(bundle.TCLTK_MSI_SHA256, r"^[0-9a-f]{64}$")


class LlvmSubset(unittest.TestCase):
    def test_the_wanted_files(self):
        wanted = bundle.llvm_wanted
        self.assertTrue(wanted("bin/clang.exe"))
        self.assertTrue(wanted("include/c++/v1/vector"))
        self.assertTrue(wanted("x86_64-w64-mingw32/lib/libc++.a"))
        self.assertTrue(wanted("lib/clang/22/include/stddef.h"))
        self.assertTrue(wanted("lib/clang/22/lib/windows/libclang_rt.builtins-x86_64.a"))
        self.assertFalse(wanted("bin/clangd.exe"))
        self.assertFalse(wanted("bin/liblldb.dll"))
        self.assertFalse(wanted("aarch64-w64-mingw32/lib/libc++.a"))
        self.assertFalse(wanted("include/ddk/ntddk.h"))
        self.assertFalse(wanted("include/mshtml.idl"))
        self.assertFalse(wanted("python/bin/python.exe"))

    def test_an_archive_missing_a_file_of_the_list_is_refused_by_name(self):
        d = tmp()
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        archive = d / "llvm.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("llvm-mingw-x/LICENSE.TXT", "text")
            zf.writestr("llvm-mingw-x/bin/clang.exe", "MZ")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.stage_llvm(bundle.Stage(d / "s"), archive)
        self.assertIn("bin/ld.lld.exe", str(cm.exception))

    def test_the_subset_keeps_what_the_link_needs(self):
        for name in ("bin/clang++.exe", "bin/ld.lld.exe", "bin/llvm-ar.exe", "bin/libLLVM-22.dll", "bin/libc++.dll",
                     "bin/libunwind.dll", "bin/x86_64-w64-windows-gnu.cfg"):
            self.assertIn(name, bundle.LLVM_FILES)


class BundleEnv(unittest.TestCase):
    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        (self.dir / "bundle.json").write_text("{}", encoding="utf-8")
        for rel in (bundle_env.CLANG, bundle_env.NINJA, bundle_env.XLATE, bundle_env.RUNTIME_LIB):
            path = self.dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        patcher = mock.patch.dict(os.environ, {"OOT2_BUNDLE_ROOT": str(self.dir)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_bundle_is_found_by_its_manifest(self):
        root = self.dir.resolve()
        self.assertEqual(bundle_env.root(), root)
        self.assertEqual(Path(bundle_env.compiler()), root / bundle_env.CLANG)
        self.assertEqual(bundle_env.runtime_lib(), root / bundle_env.RUNTIME_LIB)
        self.assertIsNone(bundle_env.zbic_archive())

    def test_without_a_manifest_there_is_no_bundle(self):
        (self.dir / "bundle.json").unlink()
        self.assertIsNone(bundle_env.root())
        self.assertIsNone(bundle_env.compiler())
        self.assertEqual(bundle_env.environment({"PATH": "a"}), {"PATH": "a"})

    def test_the_environment_puts_the_bundle_first_and_drops_other_toolchains(self):
        (self.dir / "python").mkdir()
        env = bundle_env.environment({"PATH": os.pathsep.join(["other"]), "VULKAN_SDK": "x", "OOT2_CC": "y",
                                      "KEEP": "1", "PYTHONPATH": "z"})
        folders = env["PATH"].split(os.pathsep)
        self.assertEqual(folders[0], str(self.dir.resolve() / bundle_env.LLVM_BIN))
        self.assertEqual(folders[-1], "other")
        for gone in ("VULKAN_SDK", "OOT2_CC", "PYTHONPATH"):
            self.assertNotIn(gone, env)
        self.assertEqual(env["KEEP"], "1")

    def test_a_bundled_translator_and_library_are_found(self):
        self.assertEqual(bundle_env.translator(), self.dir.resolve() / bundle_env.XLATE)
        self.assertEqual(bundle_env.ninja(), str(self.dir.resolve() / bundle_env.NINJA))


class OpusInTheBundle(unittest.TestCase):
    """libopus is in the bundle's notices and contents: the kit's library, headers and licence text go where build_game and
    the bundle's tools find them, a file the notices do not list is refused, and a runtime library without its kits is
    refused before anything is downloaded."""

    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.kit = fake_opus_kit(self.dir / "kit")

    def staged(self) -> bundle.Stage:
        st = bundle.Stage(self.dir / "bundle")
        bundle.stage_opus(st, self.kit)
        return st

    def test_the_component_is_declared_and_has_its_row_in_the_real_table(self):
        self.assertIn("opus", bundle.COMPONENTS)
        self.assertIn("opus", bundle.notice_components(NOTICES_TEXT))

    def test_the_kit_is_staged_where_the_bundle_tools_look(self):
        rows = self.staged().listing()
        paths = [r["path"] for r in rows]
        self.assertEqual(paths, sorted([
            "runtime/opus/lib/libopus.a", "runtime/opus/include/opus/opus.h", "runtime/opus/include/opus/opus_defines.h",
            "runtime/opus/include/opus/opus_types.h", "runtime/opus/include/opus/opus_multistream.h",
            "licenses/opus/opus-COPYING.txt"]))
        self.assertTrue(all(r["component"] == "opus" for r in rows))
        bundle.check_notices(rows, NOTICES_TEXT)   # the real table has the row and the licence text is there

    def test_the_staged_kit_is_the_one_build_game_finds_inside_a_bundle(self):
        import build_game

        st = self.staged()
        (st.root / bundle_env.MANIFEST).write_text("{}", encoding="utf-8")
        with mock.patch.dict(os.environ, {"OOT2_BUNDLE_ROOT": str(st.root)}):
            os.environ.pop("OPUS_DIR", None)
            self.assertEqual(bundle_env.opus_kit(), st.root.resolve() / "runtime" / "opus")
            self.assertEqual(build_game.find_opus(None), st.root.resolve() / "runtime" / "opus")
        self.assertEqual(bundle_env.OPUS_KIT, "runtime/opus")

    def test_a_bundle_without_the_libopus_row_is_refused_and_the_file_is_named(self):
        rows = self.staged().listing()
        cut = "\n".join(l for l in NOTICES_TEXT.splitlines()
                        if not (l.startswith("|") and "`bundle/opus/`" in l.split("|")[1])) + "\n"
        self.assertNotIn("`bundle/opus/`", cut.split("| File |")[1])
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_notices(rows, cut)
        self.assertIn("runtime/opus/lib/libopus.a", str(cm.exception))
        self.assertIn("bundle/opus/", str(cm.exception))

    def test_a_kit_without_a_licence_text_is_refused(self):
        shutil.rmtree(self.kit / "licenses")
        (self.kit / "licenses").mkdir()
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_notices(self.staged().listing(), NOTICES_TEXT)
        self.assertIn("opus", str(cm.exception))

    def test_a_kit_missing_a_file_the_link_needs_is_refused_by_name(self):
        (self.kit / "lib" / "libopus.a").unlink()
        with self.assertRaises(bundle.BundleError) as cm:
            self.staged()
        self.assertIn("lib/libopus.a", str(cm.exception))

    def test_a_runtime_library_without_its_kits_is_refused_before_any_download(self):
        lib = self.dir / "liboot2rt.a"
        kits = {"dav1d_dir": self.dir, "opus_dir": self.dir, "sdl3_dir": self.dir, "vulkan_dir": self.dir}
        bundle.check_kits(None, None, None, None, None)   # no library given: build_runtime_lib makes every kit
        bundle.check_kits(lib, **kits)
        for name, flag in (("opus_dir", "--opus-dir"), ("dav1d_dir", "--dav1d-dir")):
            with self.assertRaises(bundle.BundleError) as cm:
                bundle.check_kits(lib, **{**kits, name: None})
            self.assertIn(flag, str(cm.exception))
        with mock.patch.object(bundle, "fetch", side_effect=AssertionError("a download before the kits were checked")):
            with self.assertRaises(bundle.BundleError) as cm:
                bundle.build(ROOT, self.dir / "out", self.dir / "cache", None, lib, self.dir, None, self.dir, self.dir, False)
        self.assertIn("--opus-dir", str(cm.exception))


def fake_dll(path: Path, machine: int = 0x8664, extra: bytes = b"") -> Path:
    """A file with a PE header of the given machine and nothing else: no byte of Microsoft's runtime."""
    head = bytearray(0x80)
    head[:2] = b"MZ"
    head[0x3C:0x40] = (0x40).to_bytes(4, "little")
    head[0x40:0x44] = b"PE\0\0"
    head[0x44:0x46] = machine.to_bytes(2, "little")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(head) + extra)
    return path


def fake_redist(folder: Path, names=bundle.VC_RUNTIME_DLLS, machine: int = 0x8664) -> Path:
    for n in names:
        fake_dll(folder / n, machine, extra=n.encode())
    return folder


def fake_install(root: Path, versions=("14.50.1", "14.51.36231")) -> Path:
    """A Visual Studio folder with VC/Redist/MSVC/<version>/x64/Microsoft.VC145.CRT, as vswhere reports it."""
    for v in versions:
        fake_redist(root / "VC" / "Redist" / "MSVC" / v / "x64" / "Microsoft.VC145.CRT")
    (root / "VC" / "Redist" / "MSVC" / "v145").mkdir(parents=True, exist_ok=True)
    return root


class PipebuildAndVcRuntimeInTheBundle(unittest.TestCase):
    """The bundle ships pipebuild.exe and Microsoft's Visual C++ runtime DLLs: the plan names them, they go where the
    tools look, versions and SHA-256 reach the manifest, and a machine without the redistributable folder gets a named
    error. Every DLL here is a synthetic PE header."""

    def setUp(self):
        self.dir = tmp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.crt = fake_redist(self.dir / "crt")
        self.stage = bundle.Stage(self.dir / "bundle")
        self.windows = str(self.dir / "windows")

    def versions(self, path: Path) -> str:
        return {"vcruntime140.dll": "14.51.1.0", "vcruntime140_1.dll": "14.51.2.0", "msvcp140.dll": "14.51.3.0"}[path.name]

    def stage_runtime(self) -> None:
        bundle.stage_vc_runtime(self.stage, self.crt, self.versions, system_root=self.windows)

    def test_the_plan_lists_the_three_dlls_and_the_builder(self):
        self.assertEqual(bundle.REQUIRED_FILES, ("bin/pipebuild.exe" if os.name == "nt" else "bin/pipebuild",
                                                 "bin/vcruntime140.dll", "bin/vcruntime140_1.dll", "bin/msvcp140.dll"))
        self.assertEqual(bundle.VC_RUNTIME_DLLS, ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"))
        for name in ("oot2-pipebuild", "vc-runtime"):
            self.assertIn(name, bundle.COMPONENTS)

    def test_both_components_have_their_row_in_the_real_table(self):
        have = bundle.notice_components(NOTICES_TEXT)
        self.assertIn("oot2-pipebuild", have)
        self.assertIn("vc-runtime", have)
        row = next(l for l in NOTICES_TEXT.splitlines() if l.startswith("| `bundle/vc-runtime/`"))
        for word in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "https://aka.ms/vs/18/redistribution",
                     "Visual C++ redistributable terms", "System32"):
            self.assertIn(word, row)

    def test_the_dlls_are_staged_unmodified_beside_the_tools_with_a_notice(self):
        self.stage_runtime()
        rows = {r["path"]: r for r in self.stage.listing()}
        self.assertEqual(sorted(rows), ["bin/msvcp140.dll", "bin/vcruntime140.dll", "bin/vcruntime140_1.dll",
                                        "licenses/vc-runtime/NOTICE.txt"])
        for name in bundle.VC_RUNTIME_DLLS:
            self.assertEqual((self.stage.root / "bin" / name).read_bytes(), (self.crt / name).read_bytes())
            self.assertEqual(rows[f"bin/{name}"]["component"], "vc-runtime")
        self.assertIn(bundle.VC_TERMS_URL, (self.stage.root / "licenses/vc-runtime/NOTICE.txt").read_text())
        for name in bundle.REQUIRED_FILES[1:]:
            self.assertIn(name, rows)
        bundle.check_notices(list(rows.values()), NOTICES_TEXT)    # the real table has the row and the notice is there

    def test_versions_and_sha256_reach_the_manifest_and_the_notice(self):
        self.stage_runtime()
        bundle.finish(self.stage, "v", "abc", ALL_ROWS, say=lambda s: None)
        manifest = json.loads((self.stage.root / "bundle.json").read_text(encoding="utf-8"))
        files = manifest["vc_runtime"]["files"]
        self.assertEqual(sorted(files), sorted(bundle.VC_RUNTIME_DLLS))
        self.assertEqual(files["msvcp140.dll"]["version"], "14.51.3.0")
        self.assertEqual(files["vcruntime140.dll"]["sha256"],
                         hashlib.sha256((self.crt / "vcruntime140.dll").read_bytes()).hexdigest())
        self.assertEqual(manifest["vc_runtime"]["terms"], bundle.VC_TERMS_URL)
        notice = (self.stage.root / "licenses/vc-runtime/NOTICE.txt").read_text(encoding="utf-8")
        self.assertIn(files["vcruntime140_1.dll"]["sha256"], notice)
        self.assertIn("14.51.2.0", notice)

    def test_a_missing_dll_is_named(self):
        (self.crt / "msvcp140.dll").unlink()
        with self.assertRaises(bundle.BundleError) as cm:
            self.stage_runtime()
        self.assertIn("msvcp140.dll", str(cm.exception))

    def test_a_folder_that_is_not_there_is_named(self):
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.stage_vc_runtime(self.stage, self.dir / "nowhere", self.versions, system_root=self.windows)
        self.assertIn("nowhere", str(cm.exception))

    def test_an_x86_dll_is_refused(self):
        fake_redist(self.crt, machine=0x14C)
        with self.assertRaises(bundle.BundleError) as cm:
            self.stage_runtime()
        self.assertIn("not an x64 DLL", str(cm.exception))

    def test_a_file_that_is_not_a_pe_file_is_refused(self):
        (self.crt / "vcruntime140.dll").write_bytes(b"not a dll")
        with self.assertRaises(bundle.BundleError):
            self.stage_runtime()

    def test_the_system_folder_is_never_a_source(self):
        system32 = fake_redist(self.dir / "windows" / "System32")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.stage_vc_runtime(self.stage, system32, self.versions, system_root=self.windows)
        self.assertIn("System32", str(cm.exception))
        self.assertEqual(self.stage.owner, {})

    def test_the_newest_installed_redistributable_is_found_by_vswhere(self):
        install = fake_install(self.dir / "vs")
        vswhere = self.dir / "vswhere.exe"
        vswhere.write_bytes(b"x")
        seen = []

        def run(cmd, **kw):
            seen.append(cmd)
            return mock.Mock(returncode=0, stdout=f"{install}\n", stderr="")

        found = bundle.installed_vc_redist(vswhere, run)
        self.assertEqual(found, install / "VC" / "Redist" / "MSVC" / "14.51.36231" / "x64" / "Microsoft.VC145.CRT")
        self.assertIn(bundle.VS_COMPONENT, seen[0])
        self.assertIn("-latest", seen[0])

    def test_a_machine_without_vswhere_or_the_folder_gets_a_named_error(self):
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.installed_vc_redist(self.dir / "missing-vswhere.exe", mock.Mock())
        self.assertIn("vswhere.exe was not found", str(cm.exception))
        self.assertIn("--vc-redist", str(cm.exception))
        vswhere = self.dir / "vswhere.exe"
        vswhere.write_bytes(b"x")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.installed_vc_redist(vswhere, lambda cmd, **kw: mock.Mock(returncode=0, stdout="", stderr=""))
        self.assertIn(bundle.VS_COMPONENT, str(cm.exception))
        bare = self.dir / "bare"
        (bare / "VC" / "Redist" / "MSVC" / "14.51.1" / "x64").mkdir(parents=True)
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.installed_vc_redist(vswhere, lambda cmd, **kw: mock.Mock(returncode=0, stdout=f"{bare}\n", stderr=""))
        self.assertIn("Microsoft.VC14x.CRT", str(cm.exception))

    def test_a_build_on_a_machine_without_the_folder_fails_before_any_download_or_build(self):
        with mock.patch.object(bundle, "fetch", side_effect=AssertionError("a download before the folder was found")), \
                mock.patch.object(bundle, "installed_vc_redist", side_effect=bundle.BundleError("no redistributable")):
            with self.assertRaises(bundle.BundleError) as cm:
                bundle.build(ROOT, self.dir / "out", self.dir / "cache", None, None, None, None, None, None, False)
        self.assertIn("no redistributable", str(cm.exception))

    def test_the_builder_is_staged_with_the_licence_texts_of_what_is_in_it(self):
        exe = fake_dll(self.dir / "pipebuild.exe")
        kit = self.dir / "kit"
        (kit / "Licenses").mkdir(parents=True)
        (kit / "Licenses" / "volk-LICENSE.md").write_text("synthetic MIT\n", encoding="utf-8")
        bundle.stage_pipebuild(self.stage, exe, kit)
        rows = self.stage.listing()
        self.assertEqual([r["path"] for r in rows], sorted([bundle_env.PIPEBUILD, "licenses/oot2-pipebuild/LICENSE",
                                                            "licenses/oot2-pipebuild/volk-LICENSE.md"]))
        self.assertTrue(all(r["component"] == "oot2-pipebuild" for r in rows))
        bundle.check_notices(rows, NOTICES_TEXT)
        self.assertEqual(bundle_env.PIPEBUILD, bundle.REQUIRED_FILES[0])

    def test_a_builder_without_volks_licence_text_is_refused(self):
        exe = fake_dll(self.dir / "pipebuild.exe")
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.stage_pipebuild(self.stage, exe, self.dir)
        self.assertIn("volk-LICENSE.md", str(cm.exception))

    def test_a_bundle_without_the_builder_or_a_dll_is_refused_and_each_is_named(self):
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_required(self.stage)
        for rel in bundle.REQUIRED_FILES:
            self.assertIn(rel, str(cm.exception))
        self.stage_runtime()
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_required(self.stage)
        self.assertIn(bundle_env.PIPEBUILD, str(cm.exception))
        self.assertNotIn("vcruntime140.dll", str(cm.exception))

    def test_the_real_table_with_the_vc_runtime_row_removed_refuses_the_dlls_by_name(self):
        self.stage_runtime()
        cut = "\n".join(l for l in NOTICES_TEXT.splitlines()
                        if not (l.startswith("|") and "`bundle/vc-runtime/`" in l.split("|")[1])) + "\n"
        with self.assertRaises(bundle.BundleError) as cm:
            bundle.check_notices(self.stage.listing(), cut)
        self.assertIn("bin/vcruntime140.dll", str(cm.exception))
        self.assertIn("bundle/vc-runtime/", str(cm.exception))

    def test_the_content_rules_pass_the_staged_runtime_and_its_digests(self):
        self.stage_runtime()
        bundle.guard(self.stage.root, self.stage.listing())

    @unittest.skipUnless(os.name == "nt", "version.dll")
    def test_a_real_windows_binary_has_a_dotted_file_version(self):
        kernel = Path(os.environ["SystemRoot"]) / "System32" / "kernel32.dll"
        parts = bundle.file_version(kernel).split(".")
        self.assertEqual(len(parts), 4)
        self.assertTrue(all(p.isdigit() for p in parts))

    def test_a_builder_for_a_bundle_is_made_by_the_function_a_checkout_uses(self):
        from prepare import pipelines
        with mock.patch.object(pipelines, "compile_builder", return_value=self.dir / "pipebuild.exe") as cb, \
                mock.patch("build_game.vulkan_include", return_value=self.dir), \
                mock.patch("build_game.find_glslang", return_value="glslang"):
            out = bundle.build_pipebuild(self.dir / "stage", self.dir, self.dir / "cache", lambda s: None)
        self.assertEqual(out, self.dir / "pipebuild.exe")
        args, kw = cb.call_args
        self.assertEqual(Path(args[0]), self.dir / "stage" / "llvm" / "bin" / "clang.exe")
        self.assertEqual(args[2], "glslang")
        self.assertEqual(args[3], self.dir / "cache" / "pipebuild")
        self.assertTrue(kw["env"]["PATH"].startswith(str(self.dir / "stage" / "llvm" / "bin")))


def fake_opus_kit(dest: Path) -> Path:
    """A kit laid out as tools/fetch_opus.py lays one out, from synthetic files (no byte of libopus)."""
    (dest / "include" / "opus").mkdir(parents=True)
    (dest / "lib").mkdir()
    (dest / "licenses").mkdir()
    for h in ("opus.h", "opus_defines.h", "opus_types.h", "opus_multistream.h"):
        (dest / "include" / "opus" / h).write_text(f"// synthetic {h}\n", encoding="utf-8")
    (dest / "lib" / "libopus.a").write_bytes(b"!<arch>\n")
    (dest / "licenses" / "opus-COPYING.txt").write_text("synthetic licence\n", encoding="utf-8")
    return dest


if __name__ == "__main__":
    unittest.main()
