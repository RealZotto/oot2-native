# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the settings store (runtime/core/settings.c, oot2/settings.h; the menu survey).

Builds tests/runtime/settings_test.c with runtime/core and runtime/nvn at -O1 and -O2, as tests/test_runtime_nvn.py
builds its harness, and runs each case in its own process with the environment and the settings files written here:
synthetic text, nothing of the game.

- the table: unique keys, OOT2_<KEY> names that are not the tools' variables, defaults in range, the marks the store
  gives, and the enum orders the consumers rely on (gpu.h's default back end, memory.h's page levels)
- the four layers, command line over environment over file over default, for a key of each type, with the source
- each variable the runtime read before the store as its key's environment name, loaded or not, and the E2 hold
  reaching the present gate through its hook
- refusals with exit code 2, by name: a value out of range on each layer (never clamped), an unknown key on the command
  line, a line that is not a setting, a key named twice
- the save: every unrelated byte kept (comments, order, spacing, unknown keys, CRLF, a byte order mark), a missing key
  added to its group's section or a new one, and a refused rename leaving the old file whole with no temporary file
- the hot reload: a live key applied with its hook, a restart key logged as needing a restart and kept, a key the
  command line holds left alone, and the watcher thread seeing an edit
- the change counter: once per published change, never for the same value written again
- the listing of the keys not at their default, each with its source

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set, as
continuous integration sets it.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_runtime_nvn as nvn  # noqa: E402

TEST_SOURCE = nvn.ROOT / "tests" / "runtime" / "settings_test.c"
GPU_H = nvn.ROOT / "runtime" / "include" / "oot2" / "gpu.h"
MEMORY_H = nvn.ROOT / "runtime" / "include" / "oot2" / "memory.h"
TIMEOUT = 60
EXIT_REFUSED = 2
GROUPS = {"graphics", "pacing", "display", "audio", "input", "enhancements", "debug"}
# Variables the tools and the tests read: a key must never take one of these names.
TOOL_VARIABLES = {"OOT2_CC", "OOT2_NINJA", "OOT2_CONFIG", "OOT2_VSYNC_P99_US"}
# The variables the runtime read before the store, each now a key's environment name (the settings store).
OLD_VARIABLES = {"OOT2_GFX": "gfx", "OOT2_HOLD_TICKS": "hold_ticks", "OOT2_PAGE_RELEASE": "page_release",
                 "OOT2_CENSUS": "census", "OOT2_CENSUS_FRAMES": "census_frames",
                 "OOT2_CENSUS_TIMER": "census_timer", "OOT2_CENSUS_PROBE": "census_probe"}
# Keys whose default is an off value below their range (frame_rate_target: 0, or 20 to 120; the fixed frame rate gate).
OFF_VALUE_KEYS = {"frame_rate_target"}


def clean_env(**extra: str) -> dict[str, str]:
    """This process's environment without a settings variable, plus extra."""
    env = {k: v for k, v in os.environ.items() if k not in OLD_VARIABLES and k != "OOT2_CONFIG"}
    env.update(extra)
    return env


def keys_of(stdout: str) -> dict[str, tuple[str, str, str]]:
    """key -> (value, source, from) from the lines print_keys writes."""
    out = {}
    for line in stdout.splitlines():
        m = re.fullmatch(r"([a-z0-9_]+)=([^\t]*)\t([a-z ]+)\t(.*)", line)
        if m:
            out[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    return out


class SettingsTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = nvn.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-settings-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        nvn.layout.write(nvn.layout.place({"main": nvn.MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(nvn.END_SOURCE)] + sorted(str(p) for p in nvn.CORE.glob("*.c")) + \
            sorted(str(p) for p in nvn.NVN.glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        for level in nvn.LEVELS:
            exe = out / f"settings_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [*nvn.compile_flags(cc), level, *extra, "-I", str(nvn.CORE), "-I", str(out), *sources, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="case-", dir=self.tmp.name))

    def run_case(self, level: str, case: str, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=TIMEOUT,
                              env=clean_env(**env))

    def passes(self, case: str, *args: str, **env: str) -> list[subprocess.CompletedProcess]:
        outs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case, args=args, env=env):
                done = self.run_case(level, case, *args, **env)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}\n", done.stdout)
                outs.append(done)
        return outs

    def refused(self, case: str, *args: str, **env: str) -> list[subprocess.CompletedProcess]:
        outs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case, args=args, env=env):
                done = self.run_case(level, case, *args, **env)
                self.assertEqual(done.returncode, EXIT_REFUSED, done.stdout + done.stderr)
                self.assertEqual(done.stdout, "", "nothing runs on a refused value")
                outs.append(done)
        return outs

    def ini(self, text: str | bytes, name: str = "oot2.ini") -> Path:
        path = self.dir / name
        path.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
        return path

    # ---- the table
    def test_the_table_names_each_key_once_with_its_variable_group_mark_and_range(self) -> None:
        done = self.passes("table")[0]
        rows = [line.split("\t") for line in done.stdout.splitlines() if "\t" in line]
        keys = [r[0] for r in rows]
        self.assertEqual(len(keys), len(set(keys)))
        by_key = {}
        for key, group, kind, apply, default, low, high, names, env, what in rows:
            by_key[key] = (group, kind, apply, int(default), int(low), int(high), names, env)
            self.assertRegex(key, r"^[a-z][a-z0-9_]*$")
            self.assertEqual(env, "OOT2_" + key.upper())
            self.assertNotIn(env, TOOL_VARIABLES)
            self.assertFalse(env.startswith("OOT2_REQUIRE_"), env)
            self.assertIn(group, GROUPS)
            self.assertIn(apply, {"live", "restart", "prepare"})
            self.assertIn(kind, {"bool", "int", "enum"})
            if key in OFF_VALUE_KEYS:   # the default is an off value outside the range (the fixed frame rate gate)
                self.assertGreater(int(low), int(default))
            else:
                self.assertLessEqual(int(low), int(default))
                self.assertLessEqual(int(default), int(high))
            if kind == "enum":
                self.assertEqual(int(high), len(names.split("|")) - 1)
            if kind == "bool":
                self.assertEqual((int(low), int(high)), (0, 1))
            self.assertTrue(what)
        for var, key in OLD_VARIABLES.items():
            self.assertEqual(by_key[key][7], var)
        # The settings store's marks: the E2 hold is live, the back end and the page lever are read at start-up.
        self.assertEqual(by_key["hold_ticks"][2], "live")
        self.assertEqual(by_key["gfx"][2], "restart")
        self.assertEqual(by_key["page_release"][2], "restart")
        # gpu.h's default back end is the table's default; memory.h's levels are page_release's indexes.
        default_backend = re.search(r'#define GFX_DEFAULT_BACKEND "(\w+)"', GPU_H.read_text(encoding="utf-8"))[1]
        self.assertEqual(by_key["gfx"][6].split("|")[by_key["gfx"][3]], default_backend)
        self.assertEqual(by_key["gfx"][6], "null|vulkan")
        levels = re.search(r"enum \{ RT_PAGES_OFF = (\d+), RT_PAGES_COUNTING = (\d+), RT_PAGES_RELEASING = (\d+) \}",
                           MEMORY_H.read_text(encoding="utf-8"))
        names = by_key["page_release"][6].split("|")
        self.assertEqual([names[int(levels[i])] for i in (1, 2, 3)], ["off", "count", "release"])

    def test_collect_variants_is_a_development_switch_that_is_off_by_default(self) -> None:
        done = self.passes("table")[0]
        rows = {line.split("\t")[0]: line.split("\t") for line in done.stdout.splitlines() if "\t" in line}
        key, group, kind, apply, default, low, high, names, env, what = rows["collect_variants"]
        self.assertEqual((group, kind, apply, default, low, high, env),
                         ("debug", "bool", "restart", "0", "0", "1", "OOT2_COLLECT_VARIANTS"))
        self.assertIn("DEVELOPMENT", what)
        self.assertIn("never set in a proof run", what)
        for done in self.passes("show"):
            self.assertEqual(keys_of(done.stdout)["collect_variants"], ("0", "default", "the default"))
        for done in self.passes("show", OOT2_COLLECT_VARIANTS="1"):
            self.assertEqual(keys_of(done.stdout)["collect_variants"], ("1", "environment", "OOT2_COLLECT_VARIANTS"))

    def test_test_movie_blit_is_a_test_switch_that_is_off_by_default_and_says_it_changes_the_picture(self) -> None:
        done = self.passes("table")[0]
        rows = {line.split("\t")[0]: line.split("\t") for line in done.stdout.splitlines() if "\t" in line}
        key, group, kind, apply, default, low, high, names, env, what = rows["test_movie_blit"]
        self.assertEqual((group, kind, apply, default, low, high, env),
                         ("debug", "bool", "restart", "0", "0", "1", "OOT2_TEST_MOVIE_BLIT"))
        self.assertIn("TEST switch", what)
        self.assertIn("CHANGES THE PICTURE", what)
        self.assertIn("never set in a proof run", what)
        self.assertIn("OOT2_MOVIE_BLIT_KEYS", what)   # the chain's programs come from a file of the player's own
        self.assertNotIn("OOT2_MOVIE_BLIT_KEYS", {row[8] for row in rows.values()})   # a key file is not a setting
        for done in self.passes("show"):
            self.assertEqual(keys_of(done.stdout)["test_movie_blit"], ("0", "default", "the default"))
        for done in self.passes("show", OOT2_TEST_MOVIE_BLIT="1"):
            self.assertEqual(keys_of(done.stdout)["test_movie_blit"], ("1", "environment", "OOT2_TEST_MOVIE_BLIT"))
        self.refused("show", OOT2_TEST_MOVIE_BLIT="2")   # a value the key does not take is refused by name

    # ---- the layers
    def test_command_line_then_environment_then_file_then_default_for_a_key_of_each_type(self) -> None:
        # key, its default, then the file's, the environment's and the command line's value
        for key, default, by_file, by_env, by_set in (("census", "0", "1", "0", "1"),
                                                      ("hold_ticks", "0", "11", "22", "33"),
                                                      ("page_release", "off", "count", "release", "count")):
            var = "OOT2_" + key.upper()
            path = self.ini(f"[debug]\n{key} = {by_file}\n", f"{key}.ini")
            steps = (((), {}, (default, "default", "the default")),
                     (("--config", str(path)), {}, (by_file, "file", str(path))),
                     (("--config", str(path)), {var: by_env}, (by_env, "environment", var)),
                     (("--config", str(path), "--set", f"{key}={by_set}"), {var: by_env}, (by_set, "command line", "--set")))
            for args, env, want in steps:
                for done in self.passes("show", *args, **env):
                    self.assertEqual(keys_of(done.stdout)[key], want, (args, env))

    def test_each_variable_read_before_the_store_is_the_environment_name_of_its_key(self) -> None:
        for var, key, value in (("OOT2_GFX", "gfx", "vulkan"), ("OOT2_HOLD_TICKS", "hold_ticks", "7"),
                                ("OOT2_PAGE_RELEASE", "page_release", "count"), ("OOT2_CENSUS", "census", "1"),
                                ("OOT2_CENSUS_FRAMES", "census_frames", "9")):
            for case in ("show", "lazy"):   # loaded at start-up, and in a program that never loads the store
                for done in self.passes(case, **{var: value}):
                    self.assertEqual(keys_of(done.stdout)[key], (value, "environment", var))
            for done in self.passes("show", **{var: ""}):   # an empty variable sets nothing, as before the store
                self.assertEqual(keys_of(done.stdout)[key][1], "default")

    def per_level(self, case: str, text: str, *args: str, **env: str) -> list[tuple[Path, subprocess.CompletedProcess]]:
        """Run case at each level on a settings file of its own holding text (the case rewrites it)."""
        outs = []
        for level in nvn.LEVELS:
            with self.subTest(level=level, case=case, env=env):
                path = self.ini(text, f"{case}{level}.ini")
                done = self.run_case(level, case, *args, str(path), **env)
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn(f"ok {case}\n", done.stdout)
                outs.append((path, done))
        return outs

    def test_the_e2_hold_is_a_live_key_applied_through_its_hook(self) -> None:
        for path, done in self.per_level("hold", "[pacing]\nhold_ticks = 4\n"):
            self.assertIn("bound: hold 4\nset 2: hold 2\nfile 7: hold 7\n", done.stdout)
            self.assertIn(f"nvn: experiment E2: every present is held for 4 ticks, whatever the game asks for ({path})",
                          done.stderr)
            self.assertIn("settings: hold_ticks = 2 (", done.stderr)
        for _, done in self.per_level("hold", "", OOT2_HOLD_TICKS="3"):   # the variable holds it over the file
            self.assertIn("bound: hold 3\nset 2: hold 3\nfile 7: hold 3\n", done.stdout)
            self.assertIn("held for 3 ticks, whatever the game asks for (OOT2_HOLD_TICKS)", done.stderr)

    def test_the_run_header_names_the_placement_the_hold_and_the_frame_rate_target_in_every_run(self) -> None:
        # one line per run, default or not, so a default run names what it ran with. The frame rate target is the
        # step override: it replaces the game's own present interval, so a header that left it out would hide it.
        for _, done in self.per_level("hold", ""):   # no layer holds anything: every value is the default
            self.assertEqual(done.stderr.count("nvn: pacing: "), 1, done.stderr)
            self.assertIn("nvn: pacing: placement sync (the default), E2 hold none (the default), "
                          "frame rate target none (the default)\n", done.stderr)
        for _, done in self.per_level("hold", "", OOT2_HOLD_TICKS="3", OOT2_HOLD_PLACEMENT="acquire",
                                      OOT2_FRAME_RATE_TARGET="30"):
            self.assertEqual(done.stderr.count("nvn: pacing: "), 1, done.stderr)
            self.assertIn("nvn: pacing: placement acquire (OOT2_HOLD_PLACEMENT), E2 hold 3 ticks (OOT2_HOLD_TICKS), "
                          "frame rate target 30 a second (OOT2_FRAME_RATE_TARGET)\n", done.stderr)
        for path, done in self.per_level("hold", "[pacing]\nhold_ticks = 4\nframe_rate_target = 40\n"):
            self.assertIn(f"E2 hold 4 ticks ({path}), frame rate target 40 a second ({path})\n", done.stderr)

    # ---- refusals
    def test_a_value_out_of_range_is_refused_by_name_with_exit_code_2_and_never_clamped(self) -> None:
        path = self.ini("; settings\n[debug]\npage_release = all\n")
        for args, env, line in (
                (("--set", "hold_ticks=1000000"), {},
                 "oot2: --set hold_ticks=1000000: hold_ticks takes an integer from 0 to 999999\n"),
                (("--set", "gfx=metal"), {}, "oot2: --set gfx=metal: gfx takes null or vulkan\n"),
                ((), {"OOT2_HOLD_TICKS": "-1"}, "oot2: OOT2_HOLD_TICKS=-1: hold_ticks takes an integer from 0 to 999999\n"),
                ((), {"OOT2_CENSUS": "2"}, "oot2: OOT2_CENSUS=2: census takes 0 or 1\n"),
                (("--config", str(path)), {},
                 f"oot2: {path} line 3: page_release = all: page_release takes off, count or release\n")):
            for done in self.refused("show", *args, **env):
                self.assertEqual(done.stderr, line)
        for done in self.refused("lazy", OOT2_PAGE_RELEASE="yes"):
            self.assertEqual(done.stderr, "oot2: OOT2_PAGE_RELEASE=yes: page_release takes off, count or release\n")

    def test_the_target_rate_takes_0_or_20_to_120_and_nothing_between(self) -> None:
        # the fixed frame rate gate: the key's default, 0, is its off value and lies outside 20..120, so it is taken as well
        for value in ("0", "20", "40", "100", "120"):
            for done in self.passes("show", "--set", f"frame_rate_target={value}"):
                self.assertEqual(keys_of(done.stdout)["frame_rate_target"], (value, "command line", "--set"))
        for value in ("1", "19", "121", "-1", "60.5", "off"):
            for done in self.refused("show", "--set", f"frame_rate_target={value}"):
                self.assertEqual(done.stderr, f"oot2: --set frame_rate_target={value}: frame_rate_target takes an "
                                              f"integer from 20 to 120, or 0\n")
        for done in self.refused("lazy", OOT2_FRAME_RATE_TARGET="19"):
            self.assertEqual(done.stderr, "oot2: OOT2_FRAME_RATE_TARGET=19: frame_rate_target takes an integer from "
                                          "20 to 120, or 0\n")

    def test_an_unknown_key_or_a_word_without_a_value_on_the_command_line_exits_2(self) -> None:
        for done in self.refused("show", "--set", "nosuch=1"):
            self.assertEqual(done.stderr, "oot2: --set nosuch=1: no setting is named nosuch\n")
        for done in self.refused("show", "--set", "hold_ticks"):
            self.assertEqual(done.stderr, "oot2: --set hold_ticks: not key=value\n")

    def test_a_file_line_that_is_not_a_setting_or_a_key_named_twice_is_refused_by_line(self) -> None:
        bad = self.ini("[pacing]\nhold_ticks 3\n", "bad.ini")
        twice = self.ini("[pacing]\nhold_ticks = 1\n\nhold_ticks = 2\n", "twice.ini")
        for done in self.refused("show", "--config", str(bad)):
            self.assertEqual(done.stderr, f"oot2: {bad} line 2: not a comment, a [section] or key = value\n")
        for done in self.refused("show", "--config", str(twice)):
            self.assertEqual(done.stderr, f"oot2: {twice} lines 2 and 4 both set hold_ticks\n")

    def test_an_unknown_key_in_the_file_is_kept_and_logged(self) -> None:
        path = self.ini("[pacing]\nfuture_key = 12\nhold_ticks = 2\n")
        for done in self.passes("show", "--config", str(path)):
            self.assertEqual(keys_of(done.stdout)["hold_ticks"], ("2", "file", str(path)))
            self.assertIn(f"settings: {path} line 2: no setting is named future_key; the line is kept and ignored",
                          done.stderr)

    # ---- the save
    def test_a_save_rewrites_only_the_lines_of_the_keys_it_changed(self) -> None:
        original = ("; oot2 settings, edited by hand\n"
                    "# a second comment style\n"
                    "\n"
                    "[pacing]\n"
                    "hold_ticks   =   1    \n"
                    "mystery_key = keep me\n"
                    "; a comment at the end of pacing\n"
                    "\n"
                    "[debug]\n"
                    "page_release=off\n"
                    "[custom]\n"
                    "anything = goes\n")
        want = original.replace("hold_ticks   =   1    \n", "hold_ticks   =   4    \n") \
                       .replace("page_release=off\n", "page_release=off\ncensus = 1\n") + "\n[graphics]\ngfx = vulkan\n"
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.ini(original)
                done = self.run_case(level, "save", str(path), "hold_ticks=4", "census=1", "gfx=vulkan")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(path.read_bytes(), want.encode("utf-8"))
                shown = keys_of(self.run_case(level, "show", "--config", str(path)).stdout)
                self.assertEqual({k: shown[k][0] for k in ("hold_ticks", "census", "gfx", "page_release")},
                                 {"hold_ticks": "4", "census": "1", "gfx": "vulkan", "page_release": "off"})
                self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["oot2.ini"])

    def test_a_save_keeps_crlf_a_byte_order_mark_and_a_last_line_without_a_break(self) -> None:
        original = b"\xef\xbb\xbf; settings\r\n[pacing]\r\nhold_ticks = 1"
        want = b"\xef\xbb\xbf; settings\r\n[pacing]\r\nhold_ticks = 2\r\n\r\n[debug]\r\ncensus = 1\r\n"
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.ini(original)
                done = self.run_case(level, "save", str(path), "hold_ticks=2", "census=1")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(path.read_bytes(), want)

    def test_a_save_without_a_file_creates_one_and_a_section_takes_a_missing_key_after_its_last_setting(self) -> None:
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.dir / f"new{level}.ini"
                done = self.run_case(level, "save", str(path), "hold_ticks=3")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(path.read_bytes(), b"[pacing]\nhold_ticks = 3\n")
                path = self.ini("[debug]\ncensus = 0\n; trailing comment\n\n[pacing]\n", f"sec{level}.ini")
                done = self.run_case(level, "save", str(path), "page_release=count", "hold_ticks=5")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(path.read_bytes(), b"[debug]\ncensus = 0\npage_release = count\n; trailing comment\n\n"
                                                    b"[pacing]\nhold_ticks = 5\n")

    def test_a_failed_rename_leaves_the_old_file_whole_and_no_temporary_file(self) -> None:
        original = b"; keep me\n[pacing]\nhold_ticks = 1\n"
        path = self.ini(original)
        for done in self.passes("savefail", str(path), "hold_ticks=4"):
            self.assertIn(f"save refused: {path}.tmp: the rename over {path} failed", done.stdout)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["oot2.ini"])

    # ---- the hot reload
    def test_a_hot_reload_applies_a_live_key_and_logs_a_restart_key(self) -> None:
        first = "[pacing]\nhold_ticks = 1\n[debug]\ncensus = 0\n"
        edit = self.ini("[pacing]\nhold_ticks = 3\n[debug]\ncensus = 1\n", "edit.ini")
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.ini(first)
                done = self.run_case(level, "reload", str(path), str(edit))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("polls 1 2 0\n", done.stdout)   # seen, applied when the next read agrees, then quiet
                shown = keys_of(done.stdout)
                self.assertEqual(shown["hold_ticks"], ("3", "file", str(path)))
                self.assertEqual(shown["census"][0], "0")   # this run keeps its start-up value
                self.assertIn("pending census=1\n", done.stdout)
                self.assertNotIn("pending hold_ticks", done.stdout)
                self.assertIn("hook hold_ticks 1->3 applied file\n", done.stdout)
                self.assertIn("hook census 0->1 pending file\n", done.stdout)
                self.assertIn("changes 1\n", done.stdout)
                self.assertIn(f"settings: census = 1 ({path}) needs a restart; this run keeps 0", done.stderr)
                self.assertIn(f"settings: hold_ticks = 3 ({path}), applied now; it was 1", done.stderr)

    def test_a_hot_reload_leaves_a_key_the_command_line_holds(self) -> None:
        edit = self.ini("[pacing]\nhold_ticks = 3\n", "edit.ini")
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.ini("[pacing]\nhold_ticks = 1\n")
                done = self.run_case(level, "reload", str(path), str(edit), "--set", "hold_ticks=9")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(keys_of(done.stdout)["hold_ticks"], ("9", "command line", "--set"))
                self.assertIn("changes 0\n", done.stdout)
                self.assertNotIn("hook hold_ticks", done.stdout)

    def test_a_refused_file_on_reload_keeps_every_value(self) -> None:
        edit = self.ini("[pacing]\nhold_ticks = many\n", "edit.ini")
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.ini("[pacing]\nhold_ticks = 1\n")
                done = self.run_case(level, "reload", str(path), str(edit))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(keys_of(done.stdout)["hold_ticks"][0], "1")
                self.assertIn("changes 0\n", done.stdout)
                self.assertIn("hold_ticks takes an integer from 0 to 999999; the file is not applied", done.stderr)

    def test_the_watcher_thread_applies_an_edit_within_two_polls(self) -> None:
        edit = self.ini("[pacing]\nhold_ticks = 3\n", "edit.ini")
        for level in nvn.LEVELS:
            with self.subTest(level=level):
                path = self.ini("[pacing]\nhold_ticks = 1\n")
                done = self.run_case(level, "watch", str(path), str(edit))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                m = re.search(r"watch: hold_ticks 1 -> 3 after (\d+) ms", done.stdout)
                self.assertTrue(m, done.stdout)
                self.assertLess(int(m[1]), 5000)   # two polls of 500 ms, and a loaded runner's slack

    # ---- the change counter
    def test_the_change_counter_rises_once_per_published_change_and_never_for_the_same_value(self) -> None:
        want = ("start 0\nset 5: 1\nset 5 again: 1\nrewrite polls 1 2: 1\nset 6: 2\ncensus 1: 2\n"
                "refused: hold_ticks = -1: hold_ticks takes an integer from 0 to 999999\n"
                "after a refusal: 2, hold_ticks 6\n")
        for _, done in self.per_level("counter", ""):
            self.assertIn(want, done.stdout)

    # ---- the listing
    def test_the_non_default_listing_names_each_key_with_its_source(self) -> None:
        path = self.ini("[graphics]\ngfx = null\n[debug]\ncensus = 1\n")
        for done in self.passes("nondefault", "--config", str(path), "--set", "page_release=count",
                                OOT2_HOLD_TICKS="5"):
            self.assertIn(f"hold_ticks=5\tenvironment\tOOT2_HOLD_TICKS\npage_release=count\tcommand line\t--set\n"
                          f"census=1\tfile\t{path}\ncount 3\n", done.stdout)
            self.assertNotIn("gfx=", done.stdout)   # set in the file, but to its default
        for done in self.passes("nondefault"):
            self.assertIn("count 0\n", done.stdout)


if __name__ == "__main__":
    unittest.main()
