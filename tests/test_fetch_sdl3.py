# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of tools/fetch_sdl3.py and of the SDL3 kit it lays out (ticket 127).

SDL is fetched at a pinned release, built into a static library and never committed. Without a network and without a
build the tests use a kit of synthetic files, laid out the way the script lays out its own. Checked: the pin has its
shape; a download that fails the pinned SHA-256 is refused and deleted; sdl3.pc's libraries are held to the constant the
link uses (exactly on Windows); a configuration with no window driver or no Vulkan support is refused; the licence
texts are cut from the source files as they are written there; a kit the script laid out passes `problems`, and a
missing, changed, added, unrecorded or foreign one is refused by name; a kit of the same pins and another recipe is still
a kit but not a current one; the command line leaves a current kit alone, and exits 1 on a failed build and 2 with no
tool; the notice row and the licence text exist.

A kit built for real (tools/fetch_sdl3.py, then SDL3_DIR) is also read here: it must pass `problems`, its headers must
compile under the runtime's own warning flags, and a program linked with its library and LINK_LIBS must run. Without a
kit those tests skip, unless OOT2_REQUIRE_SDL3 is set, as continuous integration sets it.

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
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import fetch_sdl3 as fs  # noqa: E402

HEX64 = re.compile(r"[0-9a-f]{64}")
WIN = os.name == "nt"

PC = """\
prefix=/somewhere
Name: sdl3
Version: 3.4.20
Requires.private:
Libs: -L${libdir}  -lSDL3 -mwindows  -lm -lkernel32 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lversion -luuid -ladvapi32 -lsetupapi -lshell32 -ldinput8
Cflags: -I${includedir}
"""
FDLIBM_SOURCE = """\
#include "SDL_internal.h"
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* __ieee754_pow(x,y) return x**y */
"""
STB_SOURCE = """\
int x;
*/


/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2017 Sean Barrett
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
------------------------------------------------------------------------------
*/
"""


def fake_sdl3_kit(dest: Path, lib: bytes = b"!<arch>\n") -> Path:
    """dest laid out as the script lays out a kit, from synthetic files (no byte of SDL), and recorded the way it records
    one; returns dest. tests/test_build_game.py builds its kits with it."""
    (dest / "include" / "SDL3").mkdir(parents=True)
    (dest / "lib").mkdir()
    (dest / "licenses").mkdir()
    for name in ("SDL.h", "SDL_vulkan.h"):
        (dest / "include" / "SDL3" / name).write_text(f"// synthetic {name}\n", encoding="utf-8")
    (dest / "lib" / "libSDL3.a").write_bytes(lib)
    for name in (*fs.LICENCE_FILES, "fdlibm-SunPro.txt", "stb-MIT-or-Unlicense.txt", fs.README_NAME):
        (dest / "licenses" / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    (dest / fs.RECORD).write_text(json.dumps({"pins": fs.pins(), "recipe": fs.recipe_hash(), "target": "synthetic",
                                              "files": fs.tree_hashes(dest)}, indent=1) + "\n", encoding="utf-8")
    return dest.resolve()


def rewrite_record(dest: Path, **changes) -> None:
    record = json.loads((dest / fs.RECORD).read_text(encoding="utf-8"))
    record.update(changes)
    (dest / fs.RECORD).write_text(json.dumps(record), encoding="utf-8")


class PinShape(unittest.TestCase):
    def test_the_pin_has_a_version_an_address_and_a_digest(self) -> None:
        self.assertRegex(fs.VERSION, r"^3\.\d+\.\d+$")
        self.assertEqual(fs.SDL3.url, f"https://github.com/libsdl-org/SDL/releases/download/release-{fs.VERSION}/"
                                      f"SDL3-{fs.VERSION}.tar.gz")
        self.assertRegex(fs.SDL3.sha256, HEX64.pattern)
        self.assertEqual(fs.DEFAULT_ENV, "SDL3_DIR")
        self.assertEqual(fs.pins()["sdl3"], [fs.SDL3.url, fs.SDL3.sha256])
        self.assertEqual(fs.pins()["link_libs"], list(fs.LINK_LIBS))

    def test_the_build_is_static_and_keeps_what_the_game_uses(self) -> None:
        for option in ("SDL_STATIC=ON", "SDL_SHARED=OFF", "SDL_TESTS=OFF", "SDL_EXAMPLES=OFF", "SDL_INSTALL_TESTS=OFF",
                       "SDL_VIDEO=ON", "SDL_VULKAN=ON", "SDL_AUDIO=ON", "SDL_JOYSTICK=ON", "SDL_HIDAPI=ON"):
            self.assertIn(option, fs.OPTIONS)

    def test_the_libraries_are_flags_in_order_and_windows_has_the_system_ones(self) -> None:
        for libs in (fs.WINDOWS_LINK_LIBS, fs.LINUX_LINK_LIBS):
            self.assertTrue(all(re.fullmatch(r"-l[A-Za-z0-9_]+", lib) for lib in libs), libs)
            self.assertEqual(len(set(libs)), len(libs))
        for lib in ("-luser32", "-lgdi32", "-lwinmm", "-lsetupapi", "-limm32", "-lole32"):
            self.assertIn(lib, fs.WINDOWS_LINK_LIBS)
        self.assertEqual(fs.LINK_LIBS, fs.WINDOWS_LINK_LIBS if WIN else fs.LINUX_LINK_LIBS)


class DownloadTest(unittest.TestCase):
    def test_a_download_is_held_to_its_digest_and_a_bad_one_is_deleted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-fetch-", ignore_cleanup_errors=True) as tmp:
            src = Path(tmp) / "archive.bin"
            src.write_bytes(b"not an SDL release")
            good = hashlib.sha256(b"not an SDL release").hexdigest()
            got = Path(tmp) / "got.bin"
            fs.download(fs.Archive(src.as_uri(), good), got)   # the right digest passes
            self.assertEqual(got.read_bytes(), b"not an SDL release")
            with self.assertRaisesRegex(fs.FetchError, r"SHA-256 [0-9a-f]{64}, expected " + "0" * 64):
                fs.download(fs.Archive(src.as_uri(), "0" * 64), Path(tmp) / "bad.bin")
            self.assertFalse((Path(tmp) / "bad.bin").exists(), "a download that failed the check was kept")
            with self.assertRaisesRegex(fs.FetchError, "download failed"):
                fs.download(fs.Archive((Path(tmp) / "none.bin").as_uri(), good), Path(tmp) / "none.out")


class ReadingTheBuildTest(unittest.TestCase):
    def test_the_libraries_come_out_of_the_pc_in_order_without_sdl_and_the_subsystem_flag(self) -> None:
        libs = fs.link_libs_of(PC)
        self.assertEqual(libs, fs.WINDOWS_LINK_LIBS)
        self.assertNotIn("-lSDL3", libs)
        self.assertEqual(fs.link_libs_of("Libs: -lSDL3 -lm\nLibs.private: -ldl -lpthread\n"), ("-lm", "-ldl", "-lpthread"))

    def test_windows_must_match_exactly_and_linux_may_name_less(self) -> None:
        fs.check_link_libs(fs.WINDOWS_LINK_LIBS, True)
        with self.assertRaisesRegex(fs.FetchError, "-lcfgmgr32"):
            fs.check_link_libs((*fs.WINDOWS_LINK_LIBS, "-lcfgmgr32"), True)
        with self.assertRaises(fs.FetchError):
            fs.check_link_libs(fs.WINDOWS_LINK_LIBS[:-1], True)   # one fewer is a different list too
        fs.check_link_libs(("-lm",), False)
        with self.assertRaisesRegex(fs.FetchError, "-lX11"):
            fs.check_link_libs(("-lm", "-lX11"), False)

    def test_a_configuration_with_no_window_driver_or_no_vulkan_is_refused(self) -> None:
        good = "#define SDL_VIDEO_DRIVER_WINDOWS 1\n#define SDL_VIDEO_VULKAN 1\n"
        fs.check_config(good, True)
        with self.assertRaisesRegex(fs.FetchError, "no window driver"):
            fs.check_config("#define SDL_VIDEO_VULKAN 1\n/* #undef SDL_VIDEO_DRIVER_WINDOWS */\n", True)
        with self.assertRaisesRegex(fs.FetchError, "no Vulkan support"):
            fs.check_config("#define SDL_VIDEO_DRIVER_WINDOWS 1\n/* #undef SDL_VIDEO_VULKAN */\n", True)
        with self.assertRaisesRegex(fs.FetchError, "no window driver"):
            fs.check_config(good, False)   # Windows' driver is no driver on Linux
        fs.check_config("#define SDL_VIDEO_DRIVER_X11 1\n#define SDL_VIDEO_VULKAN 1\n", False)

    def test_the_licence_texts_are_cut_from_the_files_as_written(self) -> None:
        notice = fs.fdlibm_notice(FDLIBM_SOURCE)
        self.assertTrue(notice.startswith("Copyright (C) 1993"), notice)
        self.assertTrue(notice.endswith("is preserved.\n"), notice)
        self.assertIn("Copyright (C) 1993 by Sun Microsystems, Inc.", notice)
        self.assertIn("is preserved.", notice)
        self.assertNotIn("__ieee754_pow", notice)
        self.assertNotIn(" * ", notice)
        with self.assertRaisesRegex(fs.FetchError, "no notice"):
            fs.fdlibm_notice("/* nothing here */\n")
        stb = fs.stb_notice(STB_SOURCE)
        self.assertTrue(stb.startswith("-----"), stb)
        self.assertIn("ALTERNATIVE A - MIT License", stb)
        self.assertIn("Sean Barrett", stb)
        self.assertIn("ALTERNATIVE B", stb)
        self.assertNotIn("*/", stb)
        with self.assertRaisesRegex(fs.FetchError, "no licence"):
            fs.stb_notice("int x;\n")


class ProblemsTest(unittest.TestCase):
    """A kit laid out with synthetic files, then damaged one way at a time."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-sdl3-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.kit = fake_sdl3_kit(self.tmp / "kit")

    def test_a_kit_laid_out_by_the_script_has_no_problem_and_is_current(self) -> None:
        self.assertEqual(fs.problems(self.kit), [])
        self.assertTrue(fs.is_current(self.kit))

    def test_a_folder_that_is_not_a_folder_or_has_no_record_is_refused(self) -> None:
        self.assertIn("is not a folder", fs.problems(self.tmp / "nothing")[0])
        (self.tmp / "file").write_text("x", encoding="utf-8")
        self.assertIn("is not a folder", fs.problems(self.tmp / "file")[0])
        (self.kit / fs.RECORD).unlink()
        self.assertIn("fetch with tools/fetch_sdl3.py", fs.problems(self.kit)[0])
        (self.kit / fs.RECORD).write_text("not json", encoding="utf-8")
        self.assertIn("missing or unreadable", fs.problems(self.kit)[0])
        self.assertFalse(fs.is_current(self.kit))

    def test_a_kit_for_other_pins_or_other_link_libraries_is_refused(self) -> None:
        pins = fs.pins()
        rewrite_record(self.kit, pins=dict(pins, sdl3=["https://example.invalid/SDL3-3.0.0.tar.gz", "0" * 64]))
        self.assertRegex(fs.problems(self.kit)[0], r"other pins than this tree's")
        rewrite_record(self.kit, pins=dict(pins, link_libs=["-lm"]))
        self.assertRegex(fs.problems(self.kit)[0], r"other pins than this tree's")
        rewrite_record(self.kit, pins=dict(pins, options=pins["options"][:-1]))
        self.assertRegex(fs.problems(self.kit)[0], r"other pins than this tree's")
        rewrite_record(self.kit, pins=pins)   # the green counterpart
        self.assertEqual(fs.problems(self.kit), [])

    def test_a_missing_file_of_the_layout_is_named(self) -> None:
        for name in ("include/SDL3/SDL.h", "include/SDL3/SDL_vulkan.h", "lib/libSDL3.a", "licenses/SDL3-zlib.txt",
                     "licenses/README-sdl3.txt", "licenses/fdlibm-SunPro.txt"):
            with self.subTest(name=name):
                path = self.kit / name
                data = path.read_bytes()
                path.unlink()
                self.assertRegex(fs.problems(self.kit)[0].replace("\\", "/"), rf"{re.escape(name)} is missing")
                self.assertFalse(fs.is_current(self.kit))
                path.write_bytes(data)
                self.assertEqual(fs.problems(self.kit), [])

    def test_a_changed_an_added_and_a_vanished_file_are_named(self) -> None:
        lib = self.kit / "lib" / "libSDL3.a"
        lib.write_bytes(lib.read_bytes() + b"x")
        self.assertRegex(fs.problems(self.kit)[0], r"lib/libSDL3\.a in .* differs from what was laid out")
        lib.write_bytes(b"!<arch>\n")
        (self.kit / "include" / "SDL3" / "SDL_extra.h").write_text("// added\n", encoding="utf-8")
        self.assertIn("include/SDL3/SDL_extra.h is in", fs.problems(self.kit)[0])
        (self.kit / "include" / "SDL3" / "SDL_extra.h").unlink()
        record = json.loads((self.kit / fs.RECORD).read_text(encoding="utf-8"))
        record["files"]["include/SDL3/SDL_gone.h"] = "0" * 64
        rewrite_record(self.kit, files=record["files"])
        self.assertIn("include/SDL3/SDL_gone.h is recorded and missing", fs.problems(self.kit)[0])
        rewrite_record(self.kit, files=None)
        self.assertIn("lists no files", fs.problems(self.kit)[0])

    def test_another_recipe_with_the_same_pins_is_a_kit_but_not_a_current_one(self) -> None:
        rewrite_record(self.kit, recipe="0" * 64)
        self.assertEqual(fs.problems(self.kit), [])
        self.assertFalse(fs.is_current(self.kit))


class CommandLineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-sdl3-main-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_a_current_kit_costs_no_tool_no_download_and_no_build(self) -> None:
        kit = fake_sdl3_kit(self.tmp / "kit")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(fs, "build", side_effect=AssertionError), \
                mock.patch.object(fs, "find_tools", side_effect=AssertionError):
            self.assertEqual(fs.main(["--dest", str(kit)]), 0)
        self.assertIn("already holds the pinned build", out.getvalue())

    def test_a_failed_build_exits_with_1_and_says_why(self) -> None:
        err = io.StringIO()
        tools = ("cc", "c++", "ar", "ninja", "cmake")
        with contextlib.redirect_stderr(err), mock.patch.object(fs, "find_tools", return_value=tools), \
                mock.patch.object(fs, "build", side_effect=fs.FetchError("SHA-256 abc, expected def")):
            self.assertEqual(fs.main(["--dest", str(self.tmp / "kit")]), 1)
        self.assertIn("SHA-256 abc, expected def", err.getvalue())

    def test_a_missing_tool_exits_with_2_naming_it(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch.object(fs.shutil, "which", return_value=None):
            with self.assertRaises(SystemExit) as raised:
                fs.main(["--dest", str(self.tmp / "kit"), "--cc", "no-such-cc"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("no C compiler found", err.getvalue())

    def test_a_folder_that_is_not_the_scripts_own_is_never_replaced(self) -> None:
        dest = self.tmp / "mine"
        dest.mkdir()
        (dest / "mine.txt").write_text("not yours to delete\n", encoding="utf-8")
        with mock.patch.object(fs.subprocess, "run", side_effect=AssertionError):   # not even the compiler is asked
            with self.assertRaisesRegex(fs.FetchError, "not a folder this script laid out"):
                fs.build(dest, "cc", "c++", "ar", "ninja", "cmake")
        self.assertTrue((dest / "mine.txt").is_file())


class NoticesTest(unittest.TestCase):
    """libSDL3.a is linked into the program, so the notice file lists it with its licence text."""

    def test_sdl3_is_in_the_notices_with_its_licence_text(self) -> None:
        text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.startswith("|") and line.split("|")[1].strip().startswith("SDL3")]
        self.assertTrue(rows, "no row of THIRD_PARTY_NOTICES.md starts with SDL3")
        row = rows[0]
        self.assertIn(fs.VERSION, row)
        self.assertIn(fs.SDL3.sha256, row)
        self.assertIn("zlib", row)
        self.assertIn("tools/fetch_sdl3.py", row)
        for name in re.findall(r"\]\((licenses/[^)]+)\)", row):
            self.assertTrue((ROOT / name).is_file(), f"{name} is named by the SDL3 row and is missing")
        licence = (ROOT / "licenses" / "SDL3-zlib.txt").read_text(encoding="utf-8")
        self.assertIn("Sam Lantinga", licence)
        self.assertIn("provided 'as-is'", licence)
        for part in ("fdlibm", "stb_image", "yuv2rgb", "HIDAPI"):   # every notice the kit carries has its own row
            self.assertTrue(any(line.startswith("|") and part in line.split("|")[1] for line in text.splitlines()), part)
        for name in ("hidapi-BSD-3-Clause", "yuv2rgb-BSD-3-Clause", "stb-MIT-or-Unlicense", "fdlibm-SunPro"):
            self.assertTrue((ROOT / "licenses" / f"{name}.txt").is_file(), name)


def kit_or_skip() -> Path:
    """The kit SDL3_DIR names, as the build finds it, or a skip (a failure when OOT2_REQUIRE_SDL3 is set)."""
    named = os.environ.get(fs.DEFAULT_ENV)
    if not named:
        if os.environ.get("OOT2_REQUIRE_SDL3"):
            raise RuntimeError("OOT2_REQUIRE_SDL3 is set but SDL3_DIR is not (tools/fetch_sdl3.py)")
        raise unittest.SkipTest("no SDL3 kit: run tools/fetch_sdl3.py --dest DIR and set SDL3_DIR to DIR")
    bad = fs.problems(Path(named))
    if bad:
        raise RuntimeError(f"SDL3_DIR names a folder that is not the kit: {bad[0]}")
    return Path(named).resolve()


class WorkflowTest(unittest.TestCase):
    """The Windows job builds the kit before the tests, stops when it cannot, and requires the tests that need it."""

    def test_windows_builds_the_kit_before_the_tests_and_requires_the_tests_that_need_it(self) -> None:
        text = (ROOT / ".github" / "workflows" / "windows-build.yml").read_text(encoding="utf-8")
        step = text[text.index("- name: Build SDL3"):]
        step = step[:step.index("\n      - name:", 1)]
        self.assertNotIn("continue-on-error", step)
        self.assertIn("python tools/fetch_sdl3.py --dest $kit", step)
        self.assertIn('"SDL3_DIR=$kit" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV', step)
        self.assertIn('OOT2_REQUIRE_SDL3: "1"', text)
        self.assertLess(text.index("python tools/fetch_sdl3.py"), text.index("python tools/run_tests.py"))


class RealKitTest(unittest.TestCase):
    """The kit built for real: its headers under the runtime's warning flags, and a program linked and run."""

    PROGRAM = """\
#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <stdio.h>
int main(void) {
    SDL_SetMainReady();
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "dummy");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_JOYSTICK | SDL_INIT_GAMEPAD | SDL_INIT_AUDIO)) {
        printf("init failed: %s\\n", SDL_GetError());
        return 1;
    }
    SDL_Window *w = SDL_CreateWindow("kit", 64, 64, SDL_WINDOW_HIDDEN);
    int pads = 0;
    SDL_free(SDL_GetGamepads(&pads));
    void *vk = (void *)SDL_Vulkan_GetInstanceExtensions;   // referenced so that the Vulkan support is linked
    printf("SDL %d.%d.%d window %d vulkan %d\\n", SDL_MAJOR_VERSION, SDL_MINOR_VERSION, SDL_MICRO_VERSION, w != NULL,
           vk != NULL);
    SDL_DestroyWindow(w);
    SDL_Quit();
    return 0;
}
"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.kit = kit_or_skip()
        cls.cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        if not cls.cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-sdl3-real-", ignore_cleanup_errors=True)
        cls.src = Path(cls.tmp.name) / "kit_test.c"
        cls.src.write_text(cls.PROGRAM, encoding="utf-8", newline="\n")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_the_kit_is_the_layout_the_build_checks_for(self) -> None:
        self.assertEqual(fs.problems(self.kit), [])
        self.assertTrue(fs.is_current(self.kit))
        record = json.loads((self.kit / fs.RECORD).read_text(encoding="utf-8"))
        self.assertEqual(record["pins"]["sdl3"], [fs.SDL3.url, fs.SDL3.sha256])
        self.assertTrue((self.kit / "licenses" / "SDL3-zlib.txt").read_text(encoding="utf-8").startswith("Copyright"))

    def test_the_headers_compile_under_the_runtimes_warning_flags(self) -> None:
        import build_game

        done = subprocess.run([self.cc, *build_game.RT_FLAGS, "-Werror", "-DOOT2_WITH_SDL3=1", "-I",
                               str(self.kit / "include"), "-fsyntax-only", str(self.src)], capture_output=True,
                              text=True, errors="replace")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_a_program_links_with_the_library_and_link_libs_and_runs(self) -> None:
        exe = Path(self.tmp.name) / ("kit_test.exe" if WIN else "kit_test")
        cmd = [self.cc, "-std=gnu11", "-O1", "-I", str(self.kit / "include"), str(self.src),
               str(self.kit / "lib" / "libSDL3.a"), *fs.LINK_LIBS, "-o", str(exe)]
        done = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        ran = subprocess.run([str(exe)], capture_output=True, text=True, errors="replace", timeout=60)
        self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
        self.assertIn(f"SDL {fs.VERSION} window 1 vulkan 1", ran.stdout)


if __name__ == "__main__":
    unittest.main()
