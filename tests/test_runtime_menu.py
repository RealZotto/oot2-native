# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the configuration menu (runtime/ui, oot2/ui.h; the menu survey and answers; the menu's first rows).

Builds tests/runtime/menu_test.c with runtime/core and runtime/nvn (the real settings store) and the C++ menu, with the
fetched Dear ImGui when the folder exists (tools/fetch_imgui.py), and runs each case in its own process over a settings
file written here: synthetic text, nothing of the game. The menu draws with no window, no device and no back end, which
is the headless frame these tests read.

- the rows: every row the menu answers names, each with its group, its state (live, restart, held, or greyed) and, when
  greyed, its reason; every key of the store has a row; the greyed rows have no key yet
- the frame (needs Dear ImGui): a closed menu builds nothing, an open one draws every row, and the overlay and the
  counters list draw on their own
- a row changed through the menu, and saved when the menu closes: the store's value changes and the file keeps every
  byte but that line's value; a value the key refuses leaves the file as it was
- a click on a real control (the pointer, the frame, the checkbox) reaches the store and the file
- a value that reaches the store by another way (a reload) shows in its row with no step of the menu's
- the presets: every key of the store takes the preset's value, whatever it held
- a key the command line holds: greyed with its source, refused, and left alone by a preset
- the open key: F1, Escape, Guide, the Minus and Plus chord, and a scripted run that ignores them all

Needs clang and its C++ driver, Ninja, an archiver and the Vulkan SDK's headers (tools/build_game.py, find_toolchain);
without them the tests skip, unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it. The frame and click
cases need the fetched Dear ImGui and skip without it, unless OOT2_REQUIRE_IMGUI is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
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
from test_build_game import toolchain_or_skip  # noqa: E402
from test_runtime_settings import clean_env  # noqa: E402

ROOT = nvn.ROOT
TEST_SOURCE = ROOT / "tests" / "runtime" / "menu_test.c"
SETTINGS_H = ROOT / "runtime" / "include" / "oot2" / "settings.h"
EXE = ".exe" if os.name == "nt" else ""
TIMEOUT = 120
GROUPS = ["Graphics", "Frame rate and pacing", "Display", "Audio", "Input", "Enhancements", "Debug"]

# the menu answers' rows: label -> (group, state, key). A greyed row has no key until the ticket that consumes it
# registers one (the movie hold's), and then turns live by itself; the window's keys came with the window.
ROWS = {
    "Graphics back end": ("Graphics", "restart", "gfx"),
    "Resolution regime": ("Graphics", "greyed", "-"),
    "Frame rate target": ("Frame rate and pacing", "live", "frame_rate_target"),
    "Pacing hold placement": ("Frame rate and pacing", "live", "hold_placement"),
    "Hold every present (ticks)": ("Frame rate and pacing", "live", "hold_ticks"),
    "Hold the system timer resolution": ("Frame rate and pacing", "restart", "timer_resolution"),
    "Hold the game's step during movies": ("Frame rate and pacing", "greyed", "-"),
    "Interpolation": ("Frame rate and pacing", "greyed", "-"),
    "Window mode": ("Display", "live", "window_mode"),
    "Window width": ("Display", "live", "window_width"),
    "Window height": ("Display", "live", "window_height"),
    "Open a window": ("Display", "restart", "window"),
    "Output size": ("Display", "greyed", "-"),
    "Exclusive fullscreen at 60 Hz": ("Display", "greyed", "-"),
    "Overlay (frame time)": ("Display", "live", "overlay"),
    "Audio output": ("Audio", "greyed", "-"),
    "Stick dead zone": ("Input", "live", "input_dead_zone"),
    "Controls": ("Input", "greyed", "-"),
    "Enhancements": ("Enhancements", "greyed", "-"),
    "Counters": ("Debug", "live", "counters"),
}

INI = """\
; the player's file
[pacing]
hold_ticks   =   3
hold_placement = sync

# kept as it is
[display]
overlay = 0
"""


def store_keys() -> list[str]:
    return re.findall(r"^\s*X\((\w+), ", SETTINGS_H.read_text(encoding="utf-8"), re.M)


def rows_of(stdout: str) -> list[dict[str, str]]:
    out = []
    for line in stdout.splitlines():
        p = line.split("\t")
        if p[0] == "row" and len(p) == 9:
            out.append({"index": p[1], "state": p[2], "group": p[3], "label": p[4], "key": p[5], "value": p[6],
                        "advanced": p[7], "note": p[8]})
    return out


class MenuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tool = toolchain_or_skip()
        cls.imgui = build_game.find_imgui()
        cls.tool = dataclasses.replace(tool, imgui=cls.imgui)
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-menu-", ignore_cleanup_errors=True)
        top = Path(cls.tmp.name)
        # The C++ menu, built the way the program's runtime library is, from a folder of the headers and runtime/ui.
        runtime = top / "runtime"
        shutil.copytree(ROOT / "runtime" / "include", runtime / "include")
        shutil.copytree(ROOT / "runtime" / "ui", runtime / "ui", ignore=shutil.ignore_patterns("__pycache__"))
        build = build_game.Build(top / "build", cls.tool, runtime=runtime, jobs=2)
        build_game.make_runtime_lib(build)
        # The settings store with the rest of the core and the front end, as tests/test_runtime_settings.py builds it.
        out = top / "objs"
        out.mkdir()
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), top)
        sources = [TEST_SOURCE, nvn.END_SOURCE, *sorted(nvn.CORE.glob("*.c")), *sorted(nvn.NVN.glob("*.c"))]
        flags = [*nvn.compile_flags(tool.cc), "-O1", "-I", str(nvn.CORE), "-I", str(top)]

        # The counter-run kit (OOT2_TEST_OBJ_CACHE) keeps the C objects between its runs: an object is reused when its
        # source, the flags and every header of the runtime are the same bytes, so the core compiles once for all.
        cache = Path(os.environ["OOT2_TEST_OBJ_CACHE"]) if os.environ.get("OOT2_TEST_OBJ_CACHE") else None
        headers = hashlib.sha256()
        for h in sorted([*(ROOT / "runtime").rglob("*.h"), *top.glob("*.h")]):
            headers.update(h.read_bytes())

        def compile_one(src: Path) -> str:
            obj = out / (src.stem + ".o")
            kept = None
            if cache is not None:
                digest = hashlib.sha256(src.read_bytes() + headers.digest() + " ".join(flags).encode()).hexdigest()
                kept = cache / f"{digest}.o"
                if kept.is_file():
                    shutil.copyfile(kept, obj)
                    return str(obj)
            done = subprocess.run([*flags, "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f"compiling {src.name} failed:\n{done.stdout}{done.stderr}")
            if kept is not None:
                shutil.copyfile(obj, kept)
            return str(obj)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            objects = list(pool.map(compile_one, sources))
        cls.exe = top / f"menu_test{EXE}"
        done = subprocess.run([tool.cxx, *build_game.LINK_FLAGS, *objects, str(build.lib), *build_game.LINK_LIBS,
                               "-o", str(cls.exe)], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"linking menu_test failed:\n{done.stdout}{done.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="case-", dir=self.tmp.name))

    def ini(self, text: str = INI) -> Path:
        path = self.dir / "oot2.ini"
        path.write_bytes(text.encode("utf-8"))
        return path

    def run_case(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=TIMEOUT,
                              env=clean_env(**env))

    def passes(self, case: str, *args: str, **env: str) -> subprocess.CompletedProcess:
        done = self.run_case(case, *args, **env)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}\n", done.stdout)
        return done

    def need_imgui(self) -> None:
        if self.imgui is None:
            if os.environ.get("OOT2_REQUIRE_IMGUI"):
                self.fail("OOT2_REQUIRE_IMGUI is set but no Dear ImGui folder was fetched (tools/fetch_imgui.py)")
            self.skipTest("no fetched Dear ImGui (tools/fetch_imgui.py lays it out)")

    # ---- the rows
    def test_every_row_the_menu_answers_name_is_there_with_its_state(self) -> None:
        rows = rows_of(self.passes("rows", str(self.ini())).stdout)
        by_label = {r["label"]: r for r in rows}
        for label, (group, state, key) in ROWS.items():
            with self.subTest(row=label):
                self.assertIn(label, by_label)
                r = by_label[label]
                self.assertEqual((r["group"], r["state"], r["key"]), (group, state, key))
                if state == "greyed":   # greyed with its reason, and no control to change
                    self.assertTrue(r["note"].startswith("Greyed"), r["note"])
        # the groups come in the store's order
        order = [GROUPS.index(r["group"]) for r in rows]
        self.assertEqual(order, sorted(order))

    def test_every_key_of_the_store_has_a_row_and_the_greyed_rows_have_none(self) -> None:
        rows = rows_of(self.passes("rows", str(self.ini())).stdout)
        keys = [r["key"] for r in rows if r["key"] != "-"]
        self.assertEqual(sorted(keys), sorted(store_keys()))
        self.assertEqual(len(keys), len(set(keys)), "a key has two rows")
        for r in rows:
            if r["state"] == "greyed":
                self.assertEqual(r["key"], "-")
                self.assertTrue(r["note"], f"{r['label']} is greyed without a reason")

    def test_a_row_shows_the_stores_value(self) -> None:
        rows = {r["key"]: r for r in rows_of(self.passes("rows", str(self.ini())).stdout)}
        self.assertEqual(rows["hold_ticks"]["value"], "3")
        self.assertEqual(rows["hold_placement"]["value"], "sync")
        self.assertEqual(rows["overlay"]["value"], "0")
        self.assertEqual(rows["frame_rate_target"]["value"], "0")

    def test_default_keyboard_buttons_leave_both_sticks_available(self) -> None:
        rows = {r["key"]: r for r in rows_of(self.passes("rows", str(self.ini())).stdout)}
        expected = {"input_key_a": "return", "input_key_b": "x", "input_key_x": "c", "input_key_y": "v",
                    "input_key_zl": "q", "input_key_zr": "e", "input_key_plus": "space", "input_key_minus": "tab"}
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(rows[key]["value"], value)

    def test_a_rate_that_does_not_divide_the_game_s_step_is_marked_experimental(self) -> None:
        for rate, experimental in (("30", False), ("60", False), ("20", False), ("40", True), ("100", True)):
            with self.subTest(rate=rate):
                path = self.ini(f"[pacing]\nframe_rate_target = {rate}\n")
                rows = {r["key"]: r for r in rows_of(self.passes("rows", str(path)).stdout)}
                self.assertEqual("Experimental" in rows["frame_rate_target"]["note"], experimental)

    # ---- the frame
    def test_a_closed_menu_builds_nothing_and_an_open_one_draws_every_row(self) -> None:
        self.need_imgui()
        out = self.passes("frame", str(self.ini())).stdout
        self.assertRegex(out, r"menu\t1\t\d+\t[1-9]\d*\t[1-9]\d*\t[1-9]\d*\n")
        self.assertIn("overlay\t2\t", out)
        self.assertIn("all\t3\n", out)

    # ---- changing a row
    def test_a_row_changed_through_the_menu_rewrites_only_its_line(self) -> None:
        path = self.ini()
        out = self.passes("set", str(path), "hold_ticks", "7").stdout
        self.assertIn("ok\t1", out)
        self.assertEqual(path.read_text(encoding="utf-8"), INI.replace("=   3", "=   7"))

    def test_a_key_the_file_lacks_is_added_to_its_section(self) -> None:
        path = self.ini()
        self.passes("set", str(path), "frame_rate_target", "40")
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("frame_rate_target = 40\n"), 1)
        self.assertEqual(text.replace("frame_rate_target = 40\n", ""), INI)
        self.assertLess(text.index("frame_rate_target"), text.index("[display]"))   # in [pacing]

    def test_a_restart_row_is_saved_for_the_next_start(self) -> None:
        path = self.ini()
        self.passes("set", str(path), "timer_resolution", "0")
        self.assertIn("timer_resolution = 0\n", path.read_text(encoding="utf-8"))

    def test_a_value_the_key_refuses_leaves_the_file_as_it_was(self) -> None:
        path = self.ini()
        out = self.passes("set", str(path), "frame_rate_target", "7").stdout
        self.assertIn("refused\t", out)
        self.assertIn("ok\t0", out)
        self.assertEqual(path.read_text(encoding="utf-8"), INI)

    def test_a_click_on_a_checkbox_reaches_the_store_and_the_file(self) -> None:
        self.need_imgui()
        for key, line in (("overlay", "overlay = 1"), ("timer_resolution", "timer_resolution = 0")):
            with self.subTest(key=key):
                path = self.ini()
                out = self.passes("click", str(path), key).stdout
                before, after = re.search(r"clicked\t(\d)\t(\d)", out).groups()
                self.assertNotEqual(before, after, out)
                self.assertIn(line, path.read_text(encoding="utf-8"))

    def test_a_value_that_reaches_the_store_another_way_shows_in_its_row(self) -> None:
        out = self.passes("reload", str(self.ini("[pacing]\nhold_ticks = 0\n"))).stdout
        self.assertIn("after\t9\tlive", out)

    # ---- the presets
    def test_each_preset_sets_every_key_of_the_store(self) -> None:
        path = self.ini()
        out = self.passes("presets", str(path)).stdout
        self.assertNotIn("left\t", out)
        self.assertIn("presets\t0\t0", out)
        # the file after the second preset (the Test preset): overlay and counters on, the rest at their defaults
        text = path.read_text(encoding="utf-8")
        self.assertIn("overlay = 1", text)
        self.assertIn("counters = 1", text)
        self.assertIn("hold_ticks   =   0", text)

    # ---- a key the command line holds
    def test_a_key_the_command_line_holds_is_greyed_refused_and_kept_by_a_preset(self) -> None:
        out = self.passes("held", "--set", "hold_ticks=3", str(self.ini("[pacing]\nhold_ticks = 5\n"))).stdout
        lines = {line.split("\t")[0]: line.split("\t")[1:] for line in out.splitlines() if "\t" in line}
        self.assertEqual(lines["held"][:2], ["held", "3"])
        self.assertIn("command line", lines["held"][2])
        self.assertIn("--set", lines["held"][2])
        self.assertEqual(lines["set"][0], "0")
        self.assertEqual(out.count("value\t3"), 2)   # the row's change and the preset both left it at 3
        self.assertEqual(lines["preset"][0], "1")
        self.assertIn("hold_ticks", lines["preset"][1])

    # ---- the open key
    def test_the_open_key_and_the_scripted_run(self) -> None:
        self.passes("keys")


if __name__ == "__main__":
    unittest.main()
