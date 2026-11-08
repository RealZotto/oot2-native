# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Present to a window on the Vulkan back end (runtime/gpu/vk_present.c, vk_overlay.c, runtime/platform/window.c) through
tests/runtime/vk_window_test.c.

Builds the driver with the real runtime sources (core, front end, Vulkan back end, the platform layer over the SDL3 kit,
the C++ menu over the fetched Dear ImGui) and runs each case in a process of its own. A scene stands in for the game: three
2560x1440 textures cleared by recordings through the back end's submit and presented on the 60 Hz virtual clock the way the
front end presents (present_pace, then present, one a tick, no wait). Synthetic pictures only.

- vkw_fit, with no device: the rectangle a picture is scaled into, for the shapes a window takes
- pacing: every present has a shown time, in a tick no more than two after its due tick; the intervals between pictures on the
  screen are the 10 and 20 ms a 60 Hz clock gives a 100 Hz display, none lost and none twice; the screen sampled once per tick
  shows each frame once (the present spike's instrument); the first pixel out sits on the display's own vertical blanks
- pixels: the picture on the desktop in a window of another shape: black bars, the aspect kept, each half where it belongs
- resize: the swapchain follows every size the window takes while frames go on
- mode: windowed to borderless at the screen's size and back
- minimise: the game is never held, nothing is presented, the picture comes back
- notes: the capture's header carries the driver, the display's mode and the rest the back end notes
- close: the close request reaches the program once and the teardown is clean; exit: it ends the process from the window's
  thread while frames go on
- menu: F1 opens the menu over the picture, a real click changes the store, a live row changes the running pacing

Every case runs under synchronization validation (OOT2_VK_VALIDATION=sync), which must report nothing. Counter-runs: the shown
time not fed back (pacing must fail) and the swapchain not made again on a resize (resize must fail).

The cases that open a window run only when OOT2_WINDOW_TESTS=1 is set (they put a window on the desktop and need a GPU,
a display and the SDL3 kit); the build and vkw_fit run wherever the kit and the compilers are. OOT2_WINDOW_SECONDS sets the
length of the pacing case (10 by default; a 60 s record is made by hand with OOT2_WINDOW_SECONDS=60).

Run: python -m unittest discover -s tests -p test_runtime_vk_window.py -v
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_runtime_nvn as nvn  # noqa: E402
import build_game  # noqa: E402  (tools/ is on the path through test_runtime_nvn)
import fetch_sdl3  # noqa: E402
from test_build_game import toolchain_or_skip  # noqa: E402
from test_runtime_gpu import NO_VULKAN, case_env, skip_on_partial_device  # noqa: E402

ROOT = nvn.ROOT
RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_window_test.c"
EXE = ".exe" if os.name == "nt" else ""
EXIT_NO_DEVICE = 3
WINDOWS = os.environ.get("OOT2_WINDOW_TESTS") == "1"
SECONDS = int(os.environ.get("OOT2_WINDOW_SECONDS", "10"))
TIMEOUT = 300


def sdl3_or_skip() -> Path:
    kit = build_game.find_sdl3()
    if kit is None:
        if os.environ.get("OOT2_REQUIRE_SDL3"):
            raise RuntimeError(f"OOT2_REQUIRE_SDL3 is set but there is no SDL3 kit ({fetch_sdl3.DEFAULT_ENV} names none)")
        raise unittest.SkipTest(f"no SDL3 kit (tools/fetch_sdl3.py, then set {fetch_sdl3.DEFAULT_ENV})")
    return kit


class WindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if os.name != "nt":
            raise unittest.SkipTest("the window tests are written and run on Windows (the Linux window path is unverified)")
        kit = sdl3_or_skip()
        tool = toolchain_or_skip()
        if tool.vulkan is None or tool.glslang is None:
            raise unittest.SkipTest("no Vulkan SDK with the headers, Volk and glslang")
        cls.imgui = build_game.find_imgui()
        cls.tool = dataclasses.replace(tool, imgui=cls.imgui, sdl3=kit)
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-window-", ignore_cleanup_errors=True)
        top = Path(cls.tmp.name)
        # the C++ menu, as the program's runtime library builds it, from a folder of the headers and runtime/ui
        rt = top / "runtime"
        shutil.copytree(RUNTIME / "include", rt / "include")
        shutil.copytree(RUNTIME / "ui", rt / "ui", ignore=shutil.ignore_patterns("__pycache__"))
        build = build_game.Build(top / "build", cls.tool, runtime=rt, jobs=2)
        build_game.make_runtime_lib(build)
        # the C: the front end's core and the Vulkan back end with its shaders, the platform layer over the kit
        out = top / "objs"
        out.mkdir()
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), top)
        glue = build_game.write_shader_glue(RUNTIME, top, tool.glslang)
        sources = [TEST_SOURCE, nvn.END_SOURCE, RUNTIME / "hos" / "hid_input.c"]
        for part in ("core", "nvn", "gpu", "platform"):
            sources += sorted((RUNTIME / part).glob("*.c"))
        sources.append(glue)
        flags = [tool.cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", "-DOOT2_WITH_SDL3=1",
                 "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "nvn"),
                 "-I", str(RUNTIME / "gpu"), "-I", str(top), "-I", str(tool.vulkan), "-I", str(tool.vulkan / "Volk"),
                 "-I", str(kit / "include")]
        objects = []
        for src in sources:
            obj = out / f"{src.parent.name}_{src.stem}.o"
            done = subprocess.run([*flags, "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f"compiling {src.name} failed:\n{done.stdout}{done.stderr}")
            objects.append(str(obj))
        cls.exe = top / f"vk_window_test{EXE}"
        done = subprocess.run([tool.cxx, *build_game.LINK_FLAGS, *objects, str(build.lib), str(kit / "lib" / "libSDL3.a"),
                               *fetch_sdl3.LINK_LIBS, *build_game.LINK_LIBS, "-o", str(cls.exe)],
                              capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"linking vk_window_test failed:\n{done.stdout}{done.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, *args: str, validation: str = "sync", **env: str) -> subprocess.CompletedProcess:
        e = case_env({"OOT2_VK_VALIDATION": validation, **env})
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=TIMEOUT, env=e)

    def window_case(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        """A case that opens a window: skipped without OOT2_WINDOW_TESTS, a device, a display."""
        if not WINDOWS:
            self.skipTest("opens a window on the desktop: set OOT2_WINDOW_TESTS=1")
        done = self.run_case(*args, **env)
        skip_on_partial_device(done)
        if done.returncode == EXIT_NO_DEVICE or any(m in done.stdout + done.stderr for m in NO_VULKAN):
            if os.environ.get("OOT2_REQUIRE_GPU"):
                self.fail(f"OOT2_REQUIRE_GPU is set but the case could not run:\n{done.stdout}{done.stderr}")
            self.skipTest((done.stdout + done.stderr).strip().splitlines()[-1] if (done.stdout + done.stderr).strip() else
                          "no device or no display")
        return done

    def passes(self, case: str, *args: str, **env: str) -> subprocess.CompletedProcess:
        done = self.window_case(case, *args, **env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}\n", done.stdout)
        self.assertIn("validation errors 0 warnings 0\n", done.stdout, "synchronization validation reported something")
        return done

    # ---- no device
    def test_fit_keeps_the_aspect_centres_and_is_never_empty(self) -> None:
        done = self.run_case("fit")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ok fit\n", done.stdout)

    # ---- the window
    def test_every_tick_is_shown_once_on_the_display(self) -> None:
        done = self.passes("pacing", "--seconds", str(SECONDS))
        out = done.stdout
        ticks = SECONDS * 60
        m = re.search(r"shown_time have (\d+) never (\d+) shown_tick_within_2_ticks (\d+) of (\d+)", out)
        self.assertIsNotNone(m, out)
        have, never, within, total = map(int, m.groups())
        self.assertEqual((have, never, within, total), (ticks, 0, ticks, ticks), out)   # the shown time is fed back, always
        m = re.search(r"intervals (\d+): 10ms (\d+) 20ms (\d+) other (\d+) two_on_one_refresh (\d+)", out)
        self.assertIsNotNone(m, out)
        pairs, b10, b20, other, same = map(int, m.groups())
        # No tick lost, none twice. The whole of a 60 s record is exact (research note); a run of seconds on a machine other
        # sessions share allows one picture in a hundred to meet a scheduling delay of the host's own threads, which the
        # counter-run (a host that drops every tenth tick) is far over.
        tolerance = ticks // 100
        self.assertEqual(same, 0, out)
        self.assertLessEqual(other, tolerance, out)
        self.assertGreater(pairs, ticks - 10)
        self.assertAlmostEqual(b10 / pairs, 1 / 3, delta=0.02, msg=out)   # two thirds of the frames stay 20 ms, one third 10
        # The clock's view samples the screen at the earliest latency seen plus 12.5 ms, which holds only while the spread of the
        # latencies (tick due to screen, p1 to p99) stays inside that: a spread wider than that on a loaded machine makes the
        # instrument see repeats that the intervals above show are not there, so it is judged only when its own condition holds.
        m = re.search(r"due_to_screen_ms p1 ([\d.]+) p50 ([\d.]+) p99 ([\d.]+)", out)
        spread = float(m.group(3)) - float(m.group(1))
        m = re.search(r"clock_view repeats (\d+) never_shown (\d+)", out)
        repeats, missing = map(int, m.groups())
        if spread < 11.0:
            self.assertLessEqual(max(repeats, missing), tolerance, out)
        m = re.search(r"fpo_to_nearest_vblank_ms p50 ([\d.]+) p99 ([\d.]+)", out)
        if m:   # the first pixel out falls on the display's vertical blank
            self.assertLess(float(m.group(1)), 0.5, out)

    def test_the_picture_is_scaled_into_the_window_with_its_aspect(self) -> None:
        self.passes("pixels")

    def test_the_swapchain_follows_every_size_of_the_window(self) -> None:
        self.passes("resize")

    def test_the_window_goes_borderless_at_the_screens_size_and_back(self) -> None:
        self.passes("mode")

    def test_a_minimised_window_never_holds_the_game(self) -> None:
        self.passes("minimise")

    def test_the_close_request_reaches_the_program_once_and_the_teardown_is_clean(self) -> None:
        self.passes("close")

    def test_the_close_request_can_end_the_process_while_frames_go_on(self) -> None:
        done = self.window_case("exit")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("close: the program ends the run\n", done.stdout)

    def test_the_capture_header_carries_what_the_back_end_notes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-window-cap-") as folder:
            self.passes("notes", OOT2_CAPTURE="1", OOT2_WINDOW_TEST_CAPTURE=folder)
            captures = list(Path(folder).glob("capture-*.tsv"))
            self.assertEqual(len(captures), 1)
            header = {}
            for line in captures[0].read_text(encoding="utf-8").splitlines():
                parts = line.split("\t", 2)
                if parts[0] == "header" and len(parts) == 3:
                    header[parts[1]] = parts[2]
            for key in ("back_end", "driver", "display_mode", "implicit_vulkan_layers", "gpu_timestamps", "present_path"):
                self.assertIn(key, header, key)
            self.assertEqual(header["back_end"], "vulkan")
            self.assertRegex(header["display_mode"], r"\d+x\d+ at [\d.]+ Hz")   # the monitor's mode, from the window
            self.assertIn("Vulkan 1.", header["driver"])

    def test_f1_opens_the_menu_a_click_changes_the_store_and_a_live_row_changes_the_pacing(self) -> None:
        if self.imgui is None:
            if os.environ.get("OOT2_REQUIRE_IMGUI"):
                self.fail("OOT2_REQUIRE_IMGUI is set but no Dear ImGui folder was fetched")
            self.skipTest("no fetched Dear ImGui (tools/fetch_imgui.py lays it out)")
        self.passes("menu")

    # ---- the counter-runs
    def test_counter_run_a_shown_time_never_fed_back_fails_the_pacing_case(self) -> None:
        done = self.window_case("pacing", "--seconds", "3", "--mutant", "nofeedback")
        self.assertNotEqual(done.returncode, 0, done.stdout)
        m = re.search(r"shown_time have (\d+) never (\d+) shown_tick_within_2_ticks (\d+) of (\d+)", done.stdout)
        # the shown ticks the back end answers are all 0: the check of the case above is what this breaks
        self.assertIsNotNone(m, done.stdout)
        self.assertEqual(int(m.group(3)), 0, done.stdout)

    def test_counter_run_a_host_that_drops_every_tenth_tick_is_seen_by_the_pacing_case(self) -> None:
        done = self.window_case("pacing", "--seconds", "10", "--mutant", "skip10")
        m = re.search(r"clock_view repeats (\d+) never_shown (\d+)", done.stdout)
        self.assertIsNotNone(m, done.stdout)
        self.assertGreater(int(m.group(1)), 10 * 60 // 100, done.stdout)   # far over what the case tolerates
        m = re.search(r"intervals (\d+): 10ms (\d+) 20ms (\d+) other (\d+)", done.stdout)
        self.assertGreater(int(m.group(4)), 10 * 60 // 100, done.stdout)   # the 30 and 40 ms of a tick that was never presented

    def test_counter_run_a_swapchain_not_made_again_on_a_resize_fails_the_resize_case(self) -> None:
        done = self.window_case("resize", "--mutant", "norecreate")
        self.assertNotEqual(done.returncode, 0, done.stdout)
        self.assertIn("NOT FOLLOWED", done.stdout)


if __name__ == "__main__":
    unittest.main()
