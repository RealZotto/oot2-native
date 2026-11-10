# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the markers of the input file (runtime/hos/hid_replay.c, runtime/hos/store.c, oot2/store.h): lines that
wait for a value of the game's store instead of a tick.

The test program is the one of tests/test_runtime_hid_replay.py (tests/runtime/hid_replay_test.c). Here it loads a plain main
module that this file assembles (tests/store_fixture.py), whose data holds a synthetic store: the pointer cell, the
variable, the object, tables of cells, arrays of entries and strings, with made-up names, values and displacements. The
store description (oot2-store.txt) is written here from the format the runtime documents, not by the prepare step. A script
line `P OFFSET HEXBYTES` changes the store between two reads, as the game would. This file computes from its own model of
the format at which tick each marker fires and what is held then. Nothing is read from a game.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip, unless OOT2_REQUIRE_CLANG is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import layout  # noqa: E402
import store_fixture as fx  # noqa: E402
from prepare.store import murmur3_32  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "hid_replay_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"
TIMEOUT = 60
MAX_PROBES = 1048576   # the cells one probe of a table may visit (runtime/hos/store.c)
EXIT_FATAL = 70
BUILD = bytes(range(1, 33))   # the build id of the synthetic main module
BUTTONS = ["a", "b", "x", "y", "stickl", "stickr", "l", "r", "zl", "zr", "plus", "minus", "left", "up", "right", "down"]
AXES = ["lx", "ly", "rx", "ry"]
DESCRIPTION = "oot2-store.txt"


def bits(*names: str) -> int:
    return sum(1 << BUTTONS.index(n) for n in names)


def h(name: str) -> int:
    return murmur3_32(name.encode())


def colliding(prefix: str, capacity: int, residue: int, n: int) -> list[str]:
    """n names that fall in the same cell of a table of `capacity` cells."""
    out, i = [], 0
    while len(out) < n:
        if h(f"{prefix}{i}") % capacity == residue:
            out.append(f"{prefix}{i}")
        i += 1
    return out


# ---- the synthetic store: the shapes of its tables, and the names and values they hold
# Arbitrary layouts: every displacement, entry size and cell shape below was drawn once at random for this file, with the
# alignment the loads need and nothing else in common. The cells differ in size and in where the key and the entry number are.
BOOL = fx.Shape(table=0x170, capacity=0x58, count=0x78, entries=0xc8, stride=0x91, value=0x84, shift=5, key=0x14, index=0x18, load="b")
WORD = fx.Shape(table=0x160, capacity=0x30, count=0xf4, entries=0x150, stride=0x58, value=0x04, shift=5, key=0x08, index=0x04, load="w")
UINT = fx.Shape(table=0x1d8, capacity=0x194, count=0x54, entries=0xa0, stride=0x124, value=0x10, shift=5, key=0x08, index=0x00, load="w")
FLOAT = fx.Shape(table=0x28, capacity=0x34, count=0x11c, entries=0x40, stride=0xd0, value=0x2c, shift=6, key=0x28, index=0x3c, load="s")
STRING = fx.Shape(table=0x70, capacity=0x6c, count=0x1b8, entries=0xb8, stride=0x114, value=0xa0, shift=4, key=0x00, index=0x0c, load="x")
DECOY = fx.Shape(table=0x1a8, capacity=0x88, count=0x1a4, entries=0x168, stride=0xd4, value=0x88, shift=6, key=0x0c, index=0x14, load="x")
T_BOOL, T_INT, T_UINT, T_FLOAT, T_STRING, T_DECOY = range(6)

COLLIDERS = colliding("zz_flag_", 8, 7, 3)   # three names with one cell: the second wraps to the table's start
LONG = "".join(chr(ord("a") + i % 26) for i in range(70))
NAMES = {   # role -> (table, name)
    "bool.fa": (T_BOOL, COLLIDERS[0]), "bool.fb": (T_BOOL, COLLIDERS[1]), "bool.fc": (T_BOOL, COLLIDERS[2]),
    "age": (T_BOOL, "zz_age_flag"), "boss": (T_BOOL, "zz_boss_on"),
    "health": (T_INT, "zz_health"), "int.floor": (T_INT, "zz_floor"),
    "uint.coins": (T_UINT, "zz_coins"), "uint.keys": (T_UINT, "zz_keys"),
    "float.speed": (T_FLOAT, "zz_speed"), "float.scale": (T_FLOAT, "zz_scale"), "float.tenth": (T_FLOAT, "zz_tenth"),
    "scene": (T_STRING, "zz_scene"), "next_scene": (T_STRING, "zz_next"), "string.long": (T_STRING, "zz_long"),
    "bool.ghost": (T_BOOL, "zz_ghost"),   # in no table
}
KINDS = {"bool": "bool", "age": "bool", "boss": "bool", "health": "int", "int": "int", "uint": "uint", "float": "float",
         "scene": "string", "next_scene": "string", "string": "string"}


def kind_of(role: str) -> str:
    return KINDS[role.split(".")[0]]


def make_store(base: int) -> fx.Store:
    tables = [
        fx.Table(BOOL, 8, [(h(COLLIDERS[0]), 0), (h(COLLIDERS[1]), 1), (h(COLLIDERS[2]), 0), (h("zz_age_flag"), 1),
                           (h("zz_boss_on"), 0), (h("zz_health"), 1)]),   # the name of an int is also in a table of another class
        fx.Table(WORD, 5, [(h("zz_rooms"), 0), (h("zz_floor"), -3), (h("zz_health"), 12)]),
        fx.Table(UINT, 3, [(h("zz_coins"), 4294967290), (h("zz_keys"), 2)]),
        fx.Table(FLOAT, 5, [(h("zz_speed"), 1.5), (h("zz_scale"), -0.25), (h("zz_tenth"), 0.1)]),
        fx.Table(STRING, 3, [(h("zz_scene"), "zz_field"), (h("zz_next"), ""), (h("zz_long"), LONG)], string_room=96),
        fx.Table(DECOY, 3, [(h("zz_array_one"), "zz_decoy")]),
    ]
    return fx.Store(base, tables)


def description(store: fx.Store, roles: list[str] | None = None, build: bytes = BUILD, cell: int | None = None) -> str:
    """oot2-store.txt as the format documents it (tools/prepare/store.py), written without that module's writer."""
    lines = ["oot2-store 2", f"build {build.hex()}", f"cell {store.CELL if cell is None else cell}"]
    for t in store.tables:
        s = t.shape
        lines.append(f"module {s.table} {s.capacity} {s.count} {s.entries} {s.stride} {s.value} {s.load} {s.shift} {s.key} "
                     f"{s.index}")
    for role in (NAMES if roles is None else roles):
        lines.append(f"role {role} {kind_of(role)} {h(NAMES[role][1]):08x}")
    return "\n".join(lines) + "\n"


class Script:
    """The test program's script: reads at ticks, and pokes of the store between them."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def poke(self, line: str) -> "Script":
        self.parts.append(line)
        return self

    def read(self, *ticks: int, reads: int = 1) -> "Script":
        self.parts += [f"{t} {reads} 0 0 0 0 0\n" for t in ticks]
        return self

    def text(self) -> str:
        return "".join(self.parts)


def parse_states(stdout: str) -> list[tuple[int, int, int, int, int, int]]:
    out = []
    for line in stdout.splitlines():
        if line.startswith("state "):
            _, tick, buttons, lx, ly, rx, ry, _attr = line.split()
            out.append((int(tick), int(buttons, 16), int(lx), int(ly), int(rx), int(ry)))
    return out


def held(state: tuple) -> tuple:
    return tuple(state[1:])


def expected(spans: list[tuple[int, int, list[str]]], tick: int) -> tuple:
    """What the format promises at a tick, for spans of (first tick, count, items) whose start is already a tick."""
    buttons, axis = 0, dict.fromkeys(AXES, 0)
    for at, count, items in spans:
        if at <= tick < at + count:
            for item in items:
                if "=" in item:
                    name, value = item.split("=")
                    axis[name] = int(value)
                else:
                    buttons |= bits(item)
    return (buttons, axis["lx"], axis["ly"], axis["rx"], axis["ry"])


class MarkerTest(unittest.TestCase):
    exe: Path

    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-hid-markers-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        mods = layout.place({"main": fx.IMAGE})
        layout.write(mods, out)
        cls.base = mods[0].base
        sources = [str(TEST_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos", "audio"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        libs = [] if os.name == "nt" else ["-lm"]
        cls.exe = out / f"hid_replay_test{'.exe' if os.name == 'nt' else ''}"
        cmd = [cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
               "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(out), *sources, "-o", str(cls.exe),
               *libs]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build of hid_replay_test failed:\n{done.stdout}{done.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def setUp(self) -> None:
        self.store = make_store(self.base)

    # ---- running
    def run_case(self, lines: str | bytes, script: Script, *, store: fx.Store | None = None, desc: str | None = None,
                 main: bool = True, build: bytes = BUILD, record: Path | None = None, extra_roles: list[str] | None = None,
                 case: str = "run", timeout: int = TIMEOUT) -> tuple[subprocess.CompletedProcess, Path, Path]:
        s = store or self.store
        folder = Path(tempfile.mkdtemp(prefix="run-", dir=self.tmp.name))
        (folder / "script.txt").write_text(script.text())
        if main:
            (folder / "main").write_bytes(s.module(build))
        (folder / DESCRIPTION).write_text(description(s) if desc is None else desc)
        tape = folder.parent / (folder.name + "-input.txt")
        tape.write_bytes(lines if isinstance(lines, bytes) else lines.encode())
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OOT2_INPUT", "OOT2_CENSUS"))}
        env["OOT2_INPUT_FILE"] = str(tape)
        if record is not None:
            env["OOT2_INPUT_RECORD"] = str(record)
        try:
            done = subprocess.run([str(self.exe), case, str(folder)], capture_output=True, text=True, timeout=timeout,
                                  env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"the case ran past {timeout} s:\n{e.stdout or ''}{e.stderr or ''}")
        return done, folder, tape

    def passes(self, lines: str | bytes, script: Script, **kwargs) -> tuple[list[tuple], Path]:
        done, folder, _ = self.run_case(lines, script, **kwargs)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ok run", done.stdout)
        return parse_states(done.stdout), folder

    def check(self, lines: str, script: Script, spans: list[tuple[int, int, list[str]]], **kwargs) -> list[tuple]:
        """Run, and check every read against the model: spans start where the model says the markers fire."""
        states, folder = self.passes(lines, script, **kwargs)
        for state in states:
            self.assertEqual(held(state), expected(spans, state[0]), f"tick {state[0]}")
        return states

    def rows(self, folder: Path, kind: str) -> list[list[str]]:
        text = (folder / "census.tsv").read_text()
        return [line.split("\t") for line in text.splitlines() if line.startswith(kind + "\t")]

    def stops(self, lines: str | bytes, line: int, what: str, script: Script | None = None, **kwargs) -> None:
        done, folder, tape = self.run_case(lines, script or Script().read(0), **kwargs)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(len(fatal), 1, done.stderr)
        want = f"FATAL: input file {tape}, line {line}: "
        self.assertTrue(fatal[0].startswith(want), fatal[0])
        self.assertIn(what, fatal[0][len(want):])
        self.assertEqual(done.stdout.count("state "), 0, "the run stopped before the game could read")

    def poke(self, role: str, value) -> str:
        ti, name = NAMES[role]
        if isinstance(value, str):
            return self.store.set_string(ti, h(name), value)
        return self.store.set_value(ti, h(name), value)

    # ---- the value changes, the marker fires
    def test_fires_at_the_tick_the_value_changes(self) -> None:
        script = Script().read(0, 1, 2, 3).poke(self.poke("health", 2)).read(4, 5, 6, 7, 8, 9)
        self.check("when health <= 3 for 2 a\n", script, [(4, 2, ["a"])])

    def test_every_operator_on_a_value_that_holds_still(self) -> None:
        # health is 12: each line takes a button of its own, and the buttons held at tick 0 are those of the true lines
        rows = [("==", "12", "a", 1), ("!=", "12", "b", 0), ("<", "13", "x", 1), ("<=", "12", "y", 1), (">", "12", "l", 0),
                (">=", "12", "r", 1), ("<", "12", "zl", 0), (">", "11", "zr", 1), ("<=", "11", "plus", 0),
                (">=", "13", "minus", 0), ("!=", "13", "up", 1), ("==", "13", "down", 0)]
        text = "".join(f"when health {op} {v} for 1 {b}\n" for op, v, b, _ in rows)
        spans = [(0, 1, [b]) for _, _, b, fires in rows if fires]
        self.check(text, Script().read(0, 1, 2), spans)

    def test_negative_and_large_values(self) -> None:
        text = ("when int.floor < 0 for 1 a\nwhen int.floor == -3 for 1 b\nwhen int.floor >= -2 for 1 x\n"
                "when int.floor > -4 for 1 y\nwhen uint.coins > 4294967000 for 1 l\n"
                "when uint.coins == 4294967290 for 1 r\nwhen uint.coins < 4294967290 for 1 zl\n")
        self.check(text, Script().read(0, 1), [(0, 1, ["a", "b", "y", "l", "r"])])

    def test_floats(self) -> None:
        text = ("when float.speed > 1.25 for 1 a\nwhen float.speed == 1.5 for 1 b\nwhen float.scale < -0.2 for 1 x\n"
                "when float.speed <= 1e0 for 1 y\nwhen float.scale >= -.25 for 1 l\n")
        self.check(text, Script().read(0, 1), [(0, 1, ["a", "b", "x", "l"])])

    def test_a_decimal_is_compared_as_the_single_float_the_store_holds(self) -> None:
        # 0.1 is not a single float: the store's value is the nearest one, and a line that writes 0.1 means that value
        text = ("when float.tenth == 0.1 for 1 a\nwhen float.tenth <= 0.1 for 1 b\nwhen float.tenth >= 0.1 for 1 x\n"
                "when float.tenth < 0.1 for 1 y\nwhen float.tenth > 0.1 for 1 l\nwhen float.tenth != 0.1 for 1 r\n"
                "when float.tenth == 0.10000000149011612 for 1 zl\nwhen float.tenth > 0.09999999 for 1 zr\n")
        self.check(text, Script().read(0, 1), [(0, 1, ["a", "b", "x", "zl", "zr"])])

    def test_bools_in_a_table_where_three_names_share_a_cell(self) -> None:
        text = ("when bool.fa == 0 for 1 a\nwhen bool.fb == 1 for 1 b\nwhen bool.fc == 0 for 1 x\n"
                "when bool.fa == 1 for 1 y\nwhen bool.fb == false for 1 l\nwhen bool.fc == TRUE for 1 r\n"
                "when age == 1 for 1 zl\nwhen boss == false for 1 zr\nwhen age == 0 for 1 plus\n")
        self.check(text, Script().read(0, 1), [(0, 1, ["a", "b", "x", "zl", "zr"])])

    def test_strings(self) -> None:
        text = ("when scene == zz_field for 1 a\nwhen scene != zz_field for 1 b\nwhen next_scene == zz_field for 1 x\n"
                f"when string.long == {LONG[:63]} for 1 y\nwhen scene == zz_dungeon for 2 l\nwhen next_scene != zz_x for 1 r\n"
                "when scene != zz_fieldx for 1 plus\nwhen scene == zz_fieldx for 1 minus\n")
        script = Script().read(0, 1, 2).poke(self.poke("scene", "zz_dungeon")).read(3, 4, 5, 6)
        # the long string is read up to 63 characters; the scene turns to zz_dungeon at tick 3
        self.check(text, script, [(0, 1, ["a", "y", "r", "plus"]), (3, 2, ["l"]), (3, 1, ["b"])])

    def test_a_string_the_store_changes_back_and_forth(self) -> None:
        script = (Script().read(0).poke(self.poke("scene", "zz_a")).read(1).poke(self.poke("scene", "zz_field")).read(2)
                  .poke(self.poke("scene", "zz_a")).read(3))
        self.check("when scene == zz_a for 1 a\n", script, [(1, 1, ["a"])])   # a marker fires once

    def test_the_stretch_starts_after_the_delay(self) -> None:
        script = Script().read(0, 1, 2, 3).poke(self.poke("age", 0)).read(4, *range(5, 16))
        self.check("when age == 0 after 5 for 3 a\n", script, [(9, 3, ["a"])])

    def test_a_delay_of_zero_is_the_same_as_none(self) -> None:
        script = Script().read(0).poke(self.poke("age", 0)).read(1, 2, 3)
        self.check("when age == 0 after 0 for 2 a\nwhen age == 0 for 2 b\n", script, [(1, 2, ["a", "b"])])

    def test_a_value_that_is_true_at_the_first_read_fires_at_tick_zero(self) -> None:
        self.check("when age == 1 for 2 a\n", Script().read(0, 1, 2, 3), [(0, 2, ["a"])])

    def test_a_marker_is_looked_at_once_per_tick_at_the_ticks_first_read(self) -> None:
        # the value changes between two reads of tick 3: the second read gets the latched state, and the marker sees the
        # new value at the first read of tick 4
        script = Script().read(0, 1, 2, 3).poke(self.poke("health", 2)).read(3).read(4, 5, 6, 7)
        self.check("when health <= 3 for 2 a\n", script, [(4, 2, ["a"])])

    def test_a_marker_fires_once(self) -> None:
        script = (Script().read(0, 1).poke(self.poke("health", 2)).read(2).poke(self.poke("health", 12)).read(3, 4)
                  .poke(self.poke("health", 2)).read(5, 6, 7))
        self.check("when health <= 3 for 1 a\n", script, [(2, 1, ["a"])])

    def test_markers_and_ticks_mix_and_the_later_stick_line_wins(self) -> None:
        text = "at 0 for 6 lx=10 y\nwhen age == 0 for 3 lx=20 ry=-5 a\nat 4 for 4 lx=30\n"
        script = Script().read(0, 1).poke(self.poke("age", 0)).read(2, 3, 4, 5, 6, 7, 8)
        spans = [(0, 6, ["lx=10", "y"]), (2, 3, ["lx=20", "ry=-5", "a"]), (4, 4, ["lx=30"])]
        self.check(text, script, spans)

    def test_roles_and_words_in_either_case_and_lines_in_any_order(self) -> None:
        text = "when HEALTH >= 1 for 1 A\nwhen Float.Speed > 1 for 1 Zr\n# a comment\n\nwhen SCENE == zz_field for 1 b\r\n"
        # the role is found in either case, a value is compared as it is written (a string exactly)
        self.check(text, Script().read(0), [(0, 1, ["a", "zr", "b"])])

    def test_two_markers_fire_independently(self) -> None:
        script = (Script().read(0, 1).poke(self.poke("health", 1)).read(2, 3).poke(self.poke("boss", 1)).read(4, 5, 6))
        text = "when boss == 1 for 2 b\nwhen health < 2 for 2 a\n"
        self.check(text, script, [(2, 2, ["a"]), (4, 2, ["b"])])

    # ---- the store is not built yet
    def test_waits_until_the_game_has_built_its_store(self) -> None:
        script = Script().poke(self.store.unbuild()).read(0, 1, 2, 3, 4, 5).poke(self.store.build_again()).read(6, 7, 8)
        self.check("when age == 1 for 1 a\n", script, [(6, 1, ["a"])])

    def test_waits_while_the_variable_has_no_address(self) -> None:
        script = (Script().poke(self.store.poke(self.store.CELL, bytes(8))).read(0, 1, 2)
                  .poke(self.store.poke(self.store.CELL, (self.base + self.store.VAR).to_bytes(8, "little"))).read(3, 4))
        self.check("when age == 1 for 1 a\n", script, [(3, 1, ["a"])])

    def test_waits_while_a_table_is_not_built(self) -> None:
        s = self.store
        for poke_off, restore in ((s.table_pointer(T_BOOL, 0), s.restore_table_pointer(T_BOOL)),
                                  (s.capacity_poke(T_BOOL, 0), s.capacity_poke(T_BOOL, 8))):
            with self.subTest(poke_off[:12]):
                script = Script().poke(poke_off).read(0, 1, 2).poke(restore).read(3, 4)
                self.check("when age == 1 for 1 a\n", script, [(3, 1, ["a"])])

    def test_waits_while_a_pointer_leads_to_memory_not_committed(self) -> None:
        # a page of the module's bss nothing has touched: the game has not built what is there. That is a wait, not an
        # absent value, so the run goes past the 120 ticks an absent value stops it at
        script = (Script().poke(self.store.table_pointer(T_BOOL, self.base + fx.IMAGE - 0x1000)).read(*range(0, 130))
                  .poke(self.store.restore_table_pointer(T_BOOL)).read(130, 131))
        self.check("when age == 1 for 1 a\n", script, [(130, 1, ["a"])])

    def test_a_bool_is_any_byte_that_is_not_zero(self) -> None:
        s = self.store
        script = Script().poke(s.poke(s.entry_off[(T_BOOL, h("zz_boss_on"))] + BOOL.value, bytes([2]))).read(0, 1)
        self.check("when boss == 1 for 1 a\nwhen boss == 0 for 1 b\n", script, [(0, 1, ["a"])])

    def test_waits_while_another_table_of_the_kind_is_not_built(self) -> None:
        # health is in the int table, which is not built; the uint table of its class is, and has no such value. That is a
        # wait, not an absent value, so the run goes past the 120 ticks an absent value stops it at
        s = self.store
        script = (Script().poke(s.capacity_poke(T_INT, 0)).read(*range(0, 130)).poke(s.capacity_poke(T_INT, 5)).read(130, 131))
        self.check("when uint.keys == 2 for 1 a\nwhen health == 12 for 1 b\n", script, [(0, 1, ["a"]), (130, 1, ["b"])])

    # ---- a value the store does not have, and an address that is not guest memory
    def test_a_value_absent_from_every_built_table_stops_by_name_after_120_ticks(self) -> None:
        text = "# one\nwhen bool.ghost == 1 for 1 a\n"
        done, _, tape = self.run_case(text, Script().read(*range(0, 120)))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)   # 120 ticks, 0 to 119: the 120th has not come
        done, _, tape = self.run_case(text, Script().read(*range(0, 121)))
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(len(fatal), 1, done.stderr)
        self.assertTrue(fatal[0].startswith(f"FATAL: input file {tape}, line 2: bool.ghost has no entry in the game's store"),
                        fatal[0])
        self.assertEqual(done.stdout.count("state "), 120, "it stopped at the read of tick 120, before giving it a state")

    def test_the_count_of_absent_ticks_starts_again_after_a_wait(self) -> None:
        s = self.store
        script = (Script().read(*range(0, 100)).poke(s.unbuild()).read(100).poke(s.build_again()).read(*range(101, 200)))
        self.passes("when bool.ghost == 1 for 1 a\n", script)

    # A table of MAX_PROBES + 8 cells that holds no empty cell, in the heap (the test program builds it): every cell has a key
    # that is not the one looked up, and the one looked up sits `at` cells past the start of its probe.
    def full_table(self, hash_: int, at: int) -> str:
        cells = MAX_PROBES + 8
        size = 1 << BOOL.shift
        first = 0x10000
        other = bytearray(size)
        other[BOOL.key:BOOL.key + 4] = (1).to_bytes(4, "little")
        mine = bytearray(size)
        mine[BOOL.key:BOOL.key + 4] = hash_.to_bytes(4, "little")   # entry number 0
        s = self.store
        return (s.table_into_heap(T_BOOL, first) + s.capacity_poke(T_BOOL, cells)
                + f"F {first:x} {cells} {other.hex()}\n"
                + f"F {first + ((hash_ % cells + at) % cells) * size:x} 1 {mine.hex()}\n")

    def test_a_probe_visits_at_most_max_probes_cells(self) -> None:
        ghost = h("zz_ghost")
        text = "when bool.ghost == 0 for 1 a\n"   # entry 0 of the bool table holds 0
        # a key that is the MAX_PROBES-th cell of the probe is found; one cell further is not looked at, so the value is absent
        self.check(text, Script().poke(self.full_table(ghost, MAX_PROBES - 1)).read(0, 1), [(0, 1, ["a"])])
        self.check(text, Script().poke(self.full_table(ghost, MAX_PROBES)).read(0, 1, 2, 3), [])

    def test_an_entry_numbered_past_the_count_is_absent(self) -> None:
        script = Script().poke(self.store.count_poke(T_BOOL, 0)).read(*range(0, 121))
        done, _, tape = self.run_case("when age == 1 for 1 a\n", script)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn(f"input file {tape}, line 1: age has no entry in the game's store", done.stderr)

    def test_an_address_outside_the_guest_memory_stops_by_name(self) -> None:
        script = Script().read(0, 1).poke(self.store.table_pointer(T_BOOL, 0x1000)).read(2, 3)
        done, _, tape = self.run_case("# a\n# b\nwhen age == 1 after 9 for 1 a\n", script.read(4), build=BUILD)
        # age is true at tick 0, so the marker fired before the pointer broke: it reads nothing after
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        script = Script().poke(self.store.table_pointer(T_BOOL, 0x1000)).read(0, 1)
        done, _, tape = self.run_case("# a\n# b\nwhen age == 1 for 1 a\n", script)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(len(fatal), 1, done.stderr)
        self.assertRegex(fatal[0], rf"^FATAL: input file {re.escape(str(tape))}, line 3: reading age from the store: the "
                                   r"store description leads to address 0x[0-9a-f]+, which is outside the guest's memory")

    def test_a_float_that_is_not_a_number_satisfies_nothing(self) -> None:
        nan = bytes.fromhex("0000c07f")
        s = self.store
        script = Script().poke(s.poke(s.entry_off[(T_FLOAT, h("zz_speed"))] + FLOAT.value, nan)).read(0, 1)
        text = "when float.speed == 1.5 for 1 a\nwhen float.speed >= 0 for 1 b\nwhen float.speed <= 0 for 1 x\n"
        self.check(text, script, [])

    def test_a_fault_while_a_marker_reads_the_store_stops_by_name_and_closes_the_record(self) -> None:
        # the page the store lives on is given no access between two ticks, as the game's own call could: the read of the
        # store faults while the replay holds its lock, and the stop hook that closes the record and writes the report must
        # not wait for that lock. A hang is the failure: the run is given 20 s
        record = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "record.txt"
        text = "at 0 for 3 a\nwhen bool.ghost == 1 for 1 b\n"
        script = Script().read(0, 1, 2).poke(f"D {fx.DATA_OFF:x}\n").read(3, 4)
        done, folder, tape = self.run_case(text, script, record=record, timeout=20)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        fatal = [l for l in done.stderr.splitlines() if l.startswith("FATAL")]
        self.assertEqual(len(fatal), 1, done.stderr)
        self.assertIn("SetMemoryPermission protects it", fatal[0])
        self.assertEqual(done.stdout.count("state "), 3, "the run stopped at the read of tick 3")
        lines = [l for l in record.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 0 for 3 a"])
        row = self.rows(folder, "input")   # the report was written by the stop hook, with the replay's row
        self.assertEqual(len(row), 1, (folder / "census.tsv").read_text())
        self.assertIn("0 of 1 markers fired", row[0][5])

    def test_a_stop_by_a_marker_closes_the_record(self) -> None:
        record = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "record.txt"
        text = "at 0 for 3 a\nwhen bool.ghost == 1 for 1 b\n"
        done, _, _ = self.run_case(text, Script().read(*range(0, 125)), record=record)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)   # and not a hang on the recorder's lock
        lines = [l for l in record.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 0 for 3 a"])

    # ---- a line that cannot be taken stops at start-up, by line
    BAD = {
        "when_alone": ("when\n", '"when" needs a role, an operator and a value'),
        "no_operator": ("when scene\n", '"when scene" needs an operator and a value'),
        "no_value": ("when scene ==\n", '"when scene ==" needs a value'),
        "unknown_role": ("when nope == 1 for 1 a\n", 'unknown role "nope" (the store description has: bool.fa, '),
        "unknown_operator": ("when health = 3 for 1 a\n", 'unknown operator "=" (use == != < <= > >=)'),
        "order_on_a_string": ("when scene < zz_a for 1 a\n", "scene is a string; only == and != apply to it"),
        "bad_bool": ("when age == 2 for 1 a\n", 'the value "2" of the bool age is not 0, 1, true or false'),
        "bad_int": ("when health == x for 1 a\n", 'the value "x" of health is not a whole number'),
        "int_too_high": ("when health == 2147483648 for 1 a\n", 'the value "2147483648" of health is outside -2147483648..2147483647'),
        "int_too_low": ("when health == -2147483649 for 1 a\n", 'the value "-2147483649" of health is outside'),
        "uint_negative": ("when uint.coins == -1 for 1 a\n", 'the value "-1" of uint.coins is not a whole number'),
        "uint_too_high": ("when uint.coins == 4294967296 for 1 a\n", 'the value "4294967296" of uint.coins is outside 0..4294967295'),
        "bad_float": ("when float.speed == abc for 1 a\n", 'the value "abc" of float.speed is not a number'),
        "float_with_two_points": ("when float.speed == 1.5.2 for 1 a\n", 'the value "1.5.2" of float.speed is not a number'),
        "float_nan": ("when float.speed == nan for 1 a\n", 'the value "nan" of float.speed is not a number'),
        "string_too_long": (f"when scene == {'z' * 64} for 1 a\n", "the value of scene is longer than 63 characters"),
        "delay_missing": ("when age == 1 after\n", '"after" needs a number of ticks'),
        "delay_not_a_number": ("when age == 1 after for 1 a\n", 'the delay "for" is not a decimal number'),
        "delay_too_large": ("when age == 1 after 4294967296 for 1 a\n", 'the delay "4294967296" is above 4294967295'),
        "no_for": ("when age == 1 a\n", 'expected "for" and a count after the value, not "a"'),
        "no_for_at_the_end": ("when age == 1\n", 'expected "for" and a count after the value, not "the end of the line"'),
        "no_count": ("when age == 1 for\n", '"for" needs a count'),
        "count_zero": ("when age == 1 for 0 a\n", "the count is 0; a stretch lasts 1 tick or more"),
        "count_not_a_number": ("when age == 1 for x\n", 'the count "x" is not a decimal number'),
        "unknown_button": ("when age == 1 for 1 turbo\n", 'unknown button "turbo"'),
        "button_twice": ("when age == 1 for 1 a A\n", "button a is given twice on the line"),
        "stick_out_of_range": ("when age == 1 for 1 lx=32768\n", "lx=32768 is outside -32768..32767"),
    }

    def test_unknown_line_kind_names_both(self) -> None:
        self.stops("at 0 for 1 a\nwhile age == 1\n", 2, 'unknown line kind "while" (a line starts with "at" or "when")')

    def test_the_line_number_counts_comments_and_blank_lines(self) -> None:
        self.stops("# a\n\nat 0 for 1 a\n# b\nwhen nope == 1 for 1 a\n", 5, 'unknown role "nope"')

    # ---- the description is missing, wrong or from another game
    def test_no_description_stops_by_name(self) -> None:
        done, folder, tape = self.run_case("at 0 for 1 a\nwhen age == 1 for 1 a\n", Script().read(0))
        (folder / DESCRIPTION).unlink()
        done = subprocess.run([str(self.exe), "run", str(folder)], capture_output=True, text=True, timeout=TIMEOUT,
                              env={**os.environ, "OOT2_INPUT_FILE": str(tape)})
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn(f"FATAL: input file {tape}, line 2: a \"when\" line needs the store description: the store "
                      f"description {folder}", done.stderr)
        self.assertIn("oot2-store.txt cannot be opened", done.stderr)

    def test_a_description_of_another_build_stops_by_name(self) -> None:
        self.stops("when age == 1 for 1 a\n", 1, "was made from another build of the game's main module",
                   desc=description(self.store, build=bytes(32)))

    def test_without_the_main_module_the_description_cannot_be_checked(self) -> None:
        self.stops("when age == 1 for 1 a\n", 1, "the main module is not loaded", main=False)

    def test_a_malformed_description_stops_by_line(self) -> None:
        good = description(self.store)
        for key, text, what in [
            ("header", "oot2-store 3\n" + good.split("\n", 1)[1], "is not a file of this format"),
            ("format_1", "oot2-store 1\n" + good.split("\n", 1)[1], "is of format 1, which took the key to be at the start"),
            ("no_key", good.replace(f" b {BOOL.shift} {BOOL.key} {BOOL.index}", f" b {BOOL.shift} {BOOL.index}"),
             "line 4: not a line of the format"),
            ("key_outside_the_cell", good.replace(f" b {BOOL.shift} {BOOL.key} ", f" b {BOOL.shift} {1 << BOOL.shift} "),
             "line 4: a table whose cell cannot hold its key and entry number"),
            ("key_on_the_entry_number", good.replace(f" b {BOOL.shift} {BOOL.key} {BOOL.index}",
                                                     f" b {BOOL.shift} {BOOL.index} {BOOL.index}"),
             "line 4: a table whose cell cannot hold its key and entry number"),
            ("key_over_the_entry_number", good.replace(f" b {BOOL.shift} {BOOL.key} {BOOL.index}",
                                                       f" b {BOOL.shift} {BOOL.index + 2} {BOOL.index}"),
             "line 4: a table whose cell cannot hold its key and entry number"),
            ("cell_larger_than_64", good.replace(f" b {BOOL.shift} ", " b 7 "), "line 4: a table line with a bad number"),
            ("line", good + "role age bool 12\n", "line %d: a role line that is not one" % (good.count("\n") + 1)),
            ("unknown", good + "table 1 2 3\n", "line %d: not a line of the format" % (good.count("\n") + 1)),
            ("number", good.replace("cell ", "cell x", 1), "line 3: not a line of the format"),
            ("module", good.replace(f"module {BOOL.table} {BOOL.capacity}", f"module {BOOL.table} x", 1),
             "line 4: a table line with a bad number"),
            ("no_cell", good.replace("cell ", "cull ", 1), "line 3: not a line of the format"),
            ("cell_outside", good.replace(f"cell {self.store.CELL}", f"cell {fx.IMAGE}", 1), "outside the main module"),
        ]:
            with self.subTest(key):
                self.stops("when age == 1 for 1 a\n", 1, what, desc=text)

    def test_a_file_of_ticks_never_reads_the_description(self) -> None:
        # with no marker line, a description that is missing, from another build, or no description at all changes nothing
        for kwargs in ({"main": False}, {"desc": "not a description\n"}, {"desc": description(self.store, build=bytes(32))}):
            with self.subTest(list(kwargs)):
                states, _ = self.passes("at 0 for 2 a\n", Script().read(0, 1, 2), **kwargs)
                self.assertEqual([held(s)[0] for s in states], [bits("a"), bits("a"), 0])

    # ---- a recorded run replays to the same states, and the report says what fired
    def test_a_run_with_markers_records_by_tick_and_that_replays_without_the_store(self) -> None:
        text = "when health <= 3 after 2 for 3 a lx=500\nwhen age == 1 for 1 zr\n"
        script = Script().read(0, 1, 2).poke(self.poke("health", 3)).read(3, 4, 5, 6, 7, 8, 9)
        record = Path(tempfile.mkdtemp(dir=self.tmp.name)) / "record.txt"
        recorded, _ = self.passes(text, script, record=record)
        self.assertEqual([held(s) for s in recorded][0], (bits("zr"), 0, 0, 0, 0))
        self.assertEqual(sum(1 for s in recorded if held(s)[0] & bits("a")), 3)
        lines = [l for l in record.read_text().splitlines() if l and not l.startswith("#")]
        self.assertEqual(lines, ["at 0 for 1 zr", "at 5 for 3 a lx=500"])
        # the recorded file is plain ticks: no store, no module, and the states are the same
        replayed, _ = self.passes(record.read_bytes(), Script().read(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), main=False,
                                  desc="not a description\n")
        self.assertEqual(replayed, recorded)

    def test_the_report_counts_the_markers_that_fired(self) -> None:
        script = Script().read(0, 1).poke(self.poke("health", 1)).read(2, 3)
        _, folder = self.passes("at 0 for 1 a\nwhen health < 5 for 1 b\nwhen boss == 1 for 1 x\nwhen age == 1 for 1 y\n",
                                script)
        rows = self.rows(folder, "input")
        self.assertEqual(len(rows), 1)
        self.assertIn("4 lines;", rows[0][5])
        self.assertTrue(rows[0][5].endswith("; 2 of 3 markers fired"), rows[0][5])

    def test_no_marker_no_mention_in_the_report(self) -> None:
        _, folder = self.passes("at 0 for 1 a\n", Script().read(0))
        self.assertNotIn("marker", self.rows(folder, "input")[0][5])


def _bad(text: str, what: str):
    def test(self: MarkerTest) -> None:
        self.stops("# one\n" + text, 2, what)
    return test


for _name, (_text, _what) in MarkerTest.BAD.items():   # one test method each, so one check going red is one test red
    setattr(MarkerTest, f"test_malformed_{_name}", _bad(_text, _what))


if __name__ == "__main__":
    unittest.main()
