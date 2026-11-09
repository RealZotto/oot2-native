# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for what `oot2 prepare` does for a stranger's machine: names a failed step and where to read why, refuses
container names and output paths Windows or a hostile file could turn against the player, keeps the PC awake and
unthrottled, and finds a release bundle's tools and prebuilt runtime library before anything else.

Nothing here reads a game file. Names are strings written by the test; links are made in a temporary folder.

Run: python -m unittest discover -s tests -p test_stranger_machine.py -v
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_game  # noqa: E402
import bundle_env  # noqa: E402
from prepare import check, extract, nup, power  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, Step, run_steps  # noqa: E402


def tmp(test: unittest.TestCase) -> Path:
    d = Path(tempfile.mkdtemp(prefix="oot2-stranger-"))
    test.addCleanup(shutil.rmtree, d, ignore_errors=True)
    return d.resolve()


def make_link(link: Path, target: Path) -> bool:
    """A directory link at link to target: a junction on Windows (no privilege needed), a symlink elsewhere."""
    try:
        if os.name == "nt":
            r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
            return r.returncode == 0
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False


class ContainerNames(unittest.TestCase):
    def test_ordinary_names_pass(self):
        for name in ("main", "main.npdm", "subsdk0", "title.bin", "A b.txt", "x.tar.gz", "AUXILIARY", "comx", "CON1",
                     "LPT", "COM10", "a-b_c"):
            self.assertEqual(nup.safe_name(name), name)

    def test_separators_colons_and_dots_are_refused(self):
        for name in ("", ".", "..", "a/b", "a\\b", "C:x", "a:b", "x\0y"):
            with self.assertRaises(nup.NupError, msg=repr(name)):
                nup.safe_name(name)

    def test_wildcards_and_the_characters_windows_cannot_hold_are_refused(self):
        for name in ("a*", "a?", "a\"b", "a<b", "a>b", "a|b"):
            with self.assertRaises(nup.NupError, msg=repr(name)):
                nup.safe_name(name)

    def test_control_characters_are_refused(self):
        for code in (1, 9, 10, 31):
            with self.assertRaises(nup.NupError, msg=code):
                nup.safe_name("a" + chr(code) + "b")

    def test_a_trailing_space_or_dot_is_refused(self):
        for name in ("a ", "a.", "a. ", "a .", "name..."):
            with self.assertRaises(nup.NupError, msg=repr(name)):
                nup.safe_name(name)

    def test_device_names_are_refused_with_or_without_an_extension_and_in_any_case(self):
        for name in ("CON", "con", "Prn", "AUX", "nul", "COM1", "com9", "LPT1", "lpt9", "CON.txt", "nul.tar.gz",
                     "Aux.", "COM1.bin", "CONIN$", "COM" + chr(0xb9), "LPT" + chr(0xb2)):
            with self.assertRaises(nup.NupError, msg=repr(name)):
                nup.safe_name(name)

    def test_the_container_walk_uses_the_rule(self):
        # Nup.walk calls safe_name on every entry, files and folders; the rule is the one tested above.
        self.assertIs(nup.safe_name.__module__, nup.__name__)
        source = Path(nup.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("safe_name(entry.name)"), 1)
        self.assertEqual(source.count("safe_name(self.dirs[child][0])"), 1)


class LinksOnTheWay(unittest.TestCase):
    def setUp(self):
        self.base = tmp(self)
        self.other = tmp(self)
        (self.base / "real").mkdir()

    def test_real_folders_and_missing_paths_are_clear(self):
        self.assertIsNone(nup.link_in_path(self.base / "real" / "x", self.base))
        self.assertIsNone(nup.link_in_path(self.base / "nothing" / "x", self.base))
        nup.refuse_links(self.base / "real" / "file", self.base)

    def test_a_link_below_the_base_is_found_and_refused(self):
        if not make_link(self.base / "exefs", self.other):
            self.skipTest("this machine cannot make a directory link")
        found = nup.link_in_path(self.base / "exefs" / "main", self.base)
        self.assertEqual(found, self.base / "exefs")
        with self.assertRaises(nup.NupError) as cm:
            nup.refuse_links(self.base / "exefs" / "main", self.base)
        self.assertIn("link", str(cm.exception))

    def test_the_base_itself_may_be_a_link(self):
        link = self.other / "dest"
        if not make_link(link, self.base):
            self.skipTest("this machine cannot make a directory link")
        self.assertIsNone(nup.link_in_path(link / "real" / "x", link))

    def test_write_exefs_refuses_an_exefs_folder_that_is_a_link(self):
        dest = tmp(self)
        if not make_link(dest / "exefs", self.other):
            self.skipTest("this machine cannot make a directory link")
        with self.assertRaises(nup.NupError):
            extract.write_exefs(mock.Mock(), {"main": mock.Mock(size=1)}, dest / "exefs")
        self.assertEqual(list(self.other.iterdir()), [])

    def test_write_exefs_refuses_a_file_that_is_a_link(self):
        dest = tmp(self)
        (dest / "exefs").mkdir()
        if not make_link(dest / "exefs" / "main", self.other):
            self.skipTest("this machine cannot make a directory link")
        with self.assertRaises(nup.NupError):
            extract.write_exefs(mock.Mock(), {"main": mock.Mock(size=1)}, dest / "exefs")


@unittest.skipUnless(sys.platform == "win32", "Windows power management")
class Power(unittest.TestCase):
    def test_the_system_accepts_both_calls(self):
        guard = power.keep_awake()
        self.addCleanup(guard.close)
        self.assertTrue(guard.active)
        self.assertTrue(power.opt_out_of_throttling())

    def test_the_guard_ends_and_can_be_closed_twice(self):
        guard = power.keep_awake()
        guard.close()
        self.assertFalse(guard.active)
        guard.close()

    def test_every_process_started_after_install_is_opted_out(self):
        self.addCleanup(power.uninstall)
        calls = []
        with mock.patch.object(power, "opt_out_of_throttling", side_effect=lambda h=None: calls.append(h) or True):
            power.install()
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            proc.wait()
        self.assertGreaterEqual(len(calls), 2, "this process once, then the child")
        self.assertIsNone(calls[0])
        self.assertIsNotNone(calls[-1])

    def test_uninstall_restores_popen(self):
        original = subprocess.Popen.__init__
        power.install()
        self.assertIsNot(subprocess.Popen.__init__, original)
        power.uninstall()
        self.assertIs(subprocess.Popen.__init__, power._original_init)

    def test_the_switch_that_turns_it_off(self):
        with mock.patch.dict(os.environ, {"OOT2_NO_POWER_CONTROL": "1"}):
            self.assertFalse(power.wanted())
        with mock.patch.dict(os.environ, {"OOT2_NO_POWER_CONTROL": ""}):
            self.assertTrue(power.wanted())


class FailingStep(Step):
    id = "boom"

    def run(self, ctx):
        raise PrepareError("it did not work")


class Crashing(Step):
    id = "crash"

    def run(self, ctx):
        raise RuntimeError("bug")


class NamedFailure(unittest.TestCase):
    def run_one(self, step: Step, logs: bool) -> tuple[str, str, str]:
        dest = tmp(self)
        if logs:
            (dest / "temp" / "logs").mkdir(parents=True)
        out, err = io.StringIO(), io.StringIO()
        ctx = Context(dest, argparse.Namespace(), Reporter(out, err), state=statefile.empty())
        result = run_steps([step], ctx)
        return result, out.getvalue(), err.getvalue()

    def test_a_failed_step_names_itself_and_where_to_read_why(self):
        result, out, err = self.run_one(FailingStep(), logs=True)
        self.assertEqual(result, "failed")
        self.assertIn("oot2 prepare: step boom: it did not work", err)
        self.assertIn('the step that failed is boom', out)
        self.assertIn('start with "boom:"', out)
        self.assertIn("temp", out)
        self.assertIn("resumes at this step", out)

    def test_the_log_folder_is_named_only_when_there_is_one(self):
        _, out, _ = self.run_one(FailingStep(), logs=False)
        self.assertNotIn("the logs in", out)

    def test_an_unexpected_error_is_named_too(self):
        result, out, err = self.run_one(Crashing(), logs=False)
        self.assertEqual(result, "failed")
        self.assertIn("internal error in step crash", err)
        self.assertIn("the step that failed is crash", out)

    def test_the_hint_is_a_plain_line_the_setup_window_reads_as_information(self):
        # tools/installer/logic.py takes the last "oot2 prepare: ..." line on stderr as the failure's text, so the hint
        # must not be one.
        _, out, err = self.run_one(FailingStep(), logs=False)
        self.assertNotIn("the step that failed", err)


class BundleFirst(unittest.TestCase):
    """The bundle's tools come before PATH, and an explicit option or variable comes before the bundle."""

    def setUp(self):
        self.root = tmp(self)
        (self.root / "bundle.json").write_text("{}", encoding="utf-8")
        for rel in (bundle_env.CLANG, bundle_env.XLATE, bundle_env.RUNTIME_LIB):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
            path.chmod(0o755)    # shutil.which wants the executable bit off Windows
        patcher = mock.patch.dict(os.environ, {"OOT2_BUNDLE_ROOT": str(self.root)})
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("OOT2_CC", "OOT2_XLATE", "OOT2_RUNTIME_LIB"):
            os.environ.pop(name, None)
        self.ctx = Context(tmp(self), argparse.Namespace(), Reporter(io.StringIO(), io.StringIO()),
                           state=statefile.empty())

    def test_the_compiler_is_the_bundles(self):
        self.assertEqual(Path(check.compiler(self.ctx)), self.root / bundle_env.CLANG)

    def test_an_option_and_a_variable_come_before_the_bundle(self):
        other = tmp(self) / ("mine.exe" if os.name == "nt" else "mine")
        other.write_bytes(b"x")
        other.chmod(0o755)
        self.ctx.options.cc = str(other)
        self.assertEqual(Path(check.compiler(self.ctx)), other)
        self.ctx.options.cc = None
        with mock.patch.dict(os.environ, {"OOT2_CC": str(other)}):
            self.assertEqual(Path(check.compiler(self.ctx)), other)

    def test_the_translator_is_the_bundles_unless_one_is_named(self):
        self.assertEqual(check.translator(self.ctx), (self.root / bundle_env.XLATE).resolve())
        other = tmp(self) / "x.exe"
        other.write_bytes(b"x")
        self.ctx.options.xlate = other
        self.assertEqual(check.translator(self.ctx), other.resolve())

    def test_the_runtime_library_is_the_bundles_unless_one_is_named(self):
        self.assertEqual(check.runtime_lib(self.ctx), (self.root / bundle_env.RUNTIME_LIB).resolve())
        other = tmp(self) / "lib.a"
        other.write_bytes(b"x")
        self.ctx.options.runtime_lib = other
        self.assertEqual(check.runtime_lib(self.ctx), other.resolve())
        with mock.patch.dict(os.environ, {"OOT2_RUNTIME_LIB": str(other)}):
            self.ctx.options.runtime_lib = None
            self.assertEqual(check.runtime_lib(self.ctx), other.resolve())

    def test_a_named_library_that_is_not_there_is_a_failure_not_a_build_from_source(self):
        self.ctx.options.runtime_lib = self.root / "missing.a"
        with self.assertRaises(PrepareError) as cm:
            check.runtime_lib(self.ctx)
        self.assertIn("missing.a", str(cm.exception))

    def test_without_a_bundle_there_is_no_prebuilt_library(self):
        with mock.patch.dict(os.environ, {"OOT2_BUNDLE_ROOT": str(self.root / "nowhere")}):
            self.assertIsNone(check.runtime_lib(self.ctx))

    def test_the_link_step_does_not_ask_for_dear_imgui_with_a_prebuilt_library(self):
        self.assertIn("prebuilt runtime library", check.imgui_note(self.ctx))

    def test_the_pipelines_step_says_it_cannot_build_in_a_bundle(self):
        from prepare import pipelines
        out = io.StringIO()
        ctx = Context(tmp(self), argparse.Namespace(skip_pipelines=False), Reporter(out, io.StringIO()),
                      state=statefile.empty())
        record = pipelines.Pipelines().run(ctx)
        self.assertEqual(record, {"skipped": "release bundle"})
        self.assertIn("release bundle", out.getvalue())

    def with_builder(self, content: bytes = b"builder one") -> Path:
        path = self.root / bundle_env.PIPEBUILD
        path.write_bytes(content)
        path.chmod(0o755)
        return path

    def pipelines_ctx(self, **options) -> Context:
        return Context(tmp(self), argparse.Namespace(skip_pipelines=False, **options),
                       Reporter(io.StringIO(), io.StringIO()), state=statefile.empty())

    def test_a_bundle_with_its_builder_does_not_skip_the_pipelines_step(self):
        from prepare import pipelines
        builder = self.with_builder()
        ctx = self.pipelines_ctx()
        self.assertEqual(pipelines.bundled_builder(ctx), builder.resolve())
        self.assertEqual(pipelines.ensure_builder(ctx, tmp(self)), [str(builder.resolve())])
        record = pipelines.Pipelines().run(ctx)
        self.assertNotEqual(record, {"skipped": "release bundle"})      # it went on to look for a GPU, which a test has none of
        self.assertIn("skipped", record)

    def test_the_step_key_names_the_bundled_builder_and_is_unchanged_without_one(self):
        from prepare import pipelines
        plain = pipelines.Pipelines().key(self.pipelines_ctx())
        self.with_builder(b"builder one")
        one = pipelines.Pipelines().key(self.pipelines_ctx())
        self.with_builder(b"builder two")
        two = pipelines.Pipelines().key(self.pipelines_ctx())
        self.assertEqual(len({plain, one, two}), 3)
        (self.root / bundle_env.PIPEBUILD).unlink()
        self.assertEqual(pipelines.Pipelines().key(self.pipelines_ctx()), plain)
        # Without a bundled builder no part is added at all, so a checkout's key is what it was before the builder shipped.
        seen = []
        real = pipelines.fingerprint
        with mock.patch.object(pipelines, "fingerprint", lambda *parts: seen.append(parts) or real(*parts)):
            pipelines.Pipelines().key(self.pipelines_ctx())
            self.with_builder(b"builder one")
            pipelines.Pipelines().key(self.pipelines_ctx())
        without, with_one = seen
        self.assertEqual([p for p in without if str(p).startswith("bundled-builder")], [])
        self.assertEqual(len([p for p in with_one if str(p).startswith("bundled-builder")]), 1)
        self.assertEqual(len(with_one), len(without) + 1)

    def test_another_runtime_library_does_not_use_the_bundled_builder(self):
        from prepare import pipelines
        self.with_builder()
        other = tmp(self) / "liboot2rt.a"
        other.write_bytes(b"another runtime")
        ctx = self.pipelines_ctx(runtime_lib=other)
        self.assertIsNone(pipelines.bundled_builder(ctx))
        self.assertEqual(pipelines.Pipelines().run(ctx), {"skipped": "release bundle"})

    def test_without_a_bundle_there_is_no_bundled_builder(self):
        from prepare import pipelines
        self.with_builder()
        with mock.patch.dict(os.environ, {"OOT2_BUNDLE_ROOT": str(self.root / "nowhere")}):
            self.assertIsNone(pipelines.bundled_builder(self.pipelines_ctx()))


class VulkanHeadersOnly(unittest.TestCase):
    def test_volk_is_asked_for_only_when_the_runtime_is_built(self):
        sdk = tmp(self)
        header = sdk / "Include" / "vulkan" / "vulkan_core.h"
        header.parent.mkdir(parents=True)
        header.write_text("#define VK_HEADER_VERSION 1\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"OOT2_BUNDLE_ROOT": str(sdk / "nowhere")}):
            self.assertIsNone(build_game.vulkan_include(sdk))
            self.assertEqual(build_game.vulkan_include(sdk, need_volk=False), (sdk / "Include").resolve())
            volk = sdk / "Include" / "Volk"
            volk.mkdir()
            for name in build_game.VOLK_FILES:
                (volk / name).write_text("x", encoding="utf-8")
            self.assertEqual(build_game.vulkan_include(sdk), (sdk / "Include").resolve())


if __name__ == "__main__":
    unittest.main()
