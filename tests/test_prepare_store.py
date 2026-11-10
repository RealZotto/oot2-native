# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The store step of `oot2 prepare` (tools/prepare/store.py): the names file, the hash, the structural match over machine
code, the description file, and the step as the runner runs it.

Everything here is synthetic. The machine code is assembled by tests/store_fixture.py in the shape the match reads (a
lookup compiled inline: the object through a pointer cell, a table and a capacity, the probe's division, the entry number,
the entry size, the value), with made-up displacements; the names are made up. Nothing is read from a game.

Needs capstone, as the lift step's spin-wait search does (pinned in continuous integration).

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import store_fixture as fx  # noqa: E402
from prepare import all_steps, store  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, Step, run_steps  # noqa: E402

try:
    import capstone  # noqa: F401
except ImportError:   # pragma: no cover
    capstone = None

TEXT = 0x4000   # room for the functions below
CELL = 0x2100   # the address of the pointer cell in the synthetic code: the match reads it from the instructions
OTHER_CELL = 0x2300
BOOL = fx.Shape(table=0x40, capacity=0x4c, count=0x30, entries=0x38, stride=0x14, value=0x0d, load="b")
WORD = fx.Shape(table=0x90, capacity=0x9c, count=0x80, entries=0x88, stride=0x30, value=0x24, shift=4, index=8, load="w")
FLOAT = fx.Shape(table=0x130, capacity=0x13c, count=0x120, entries=0x128, stride=0x30, value=0x24, load="s")
STRING = fx.Shape(table=0x180, capacity=0x18c, count=0x170, entries=0x178, stride=0x54, value=0x20, load="x")


def functions(shape: fx.Shape, copies: int, cell: int = CELL, flaw: str = "", start: int = fx.CODE_AT) -> tuple[list[int], int]:
    """`copies` functions that look a value up, back to back from `start`; the words and the address after them."""
    words: list[int] = []
    at = start
    for _ in range(copies):
        body = fx.lookup(at, cell, shape, flaw=flaw)
        body += [fx.NOP] * (0x24 - len(body))
        words += body
        at += 4 * len(body)
    return words, at


def blob(*parts: tuple[fx.Shape, int, int, str]) -> bytes:
    """Functions for each (shape, copies, cell, flaw), one after the other."""
    words: list[int] = []
    at = fx.CODE_AT
    for shape, copies, cell, flaw in parts:
        more, at = functions(shape, copies, cell, flaw, at)
        words += more
    return fx.assemble(words)


def find(code: bytes):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "main"
        path.write_bytes(fx.make_nso(code, text=TEXT))
        from lift.nso import Nso
        nso = Nso(path)
        return store.find_store(nso.image, nso.text[1])


class HashTest(unittest.TestCase):
    def test_published_vectors(self) -> None:
        # MurmurHash3_x86_32 test vectors published with the algorithm's implementations
        for data, seed, want in [(b"", 0, 0), (b"", 1, 0x514E28B7), (b"test", 0, 0xBA6BD213),
                                 (b"Hello, world!", 0, 0xC0363E43), (b"aaaa", 0x9747B28C, 0x5A97808A),
                                 (b"The quick brown fox jumps over the lazy dog", 0, 0x2E4FF723), (b"abc", 0, 0xB3DD93FA)]:
            self.assertEqual(store.murmur3_32(data, seed), want, (data, seed))

    def test_every_tail_length(self) -> None:
        # a name of 1, 2, 3 and 4 bytes past a block takes a different path each
        seen = {store.murmur3_32(b"abcdefgh"[:n]) for n in range(9)}
        self.assertEqual(len(seen), 9)
        self.assertEqual(store.murmur3_32(b"a"), 0x3C2569B2)
        self.assertEqual(store.murmur3_32(b"ab"), 0x9BBFD75F)
        self.assertEqual(store.murmur3_32(b"abc"), 0xB3DD93FA)
        self.assertEqual(store.murmur3_32(b"abcd"), 0x43ED676A)


class NamesTest(unittest.TestCase):
    def test_a_names_file(self) -> None:
        raw = b"# the player's names\r\n\r\nSCENE = zz_scene_name  # a comment\nbool.door_open=zz_door\nuint.coins = zz_coins\n"
        got = store.parse_names(raw)
        self.assertEqual(got, [("scene", "string", store.murmur3_32(b"zz_scene_name")),
                               ("bool.door_open", "bool", store.murmur3_32(b"zz_door")),
                               ("uint.coins", "uint", store.murmur3_32(b"zz_coins"))])

    def test_every_fixed_role_has_a_known_kind(self) -> None:
        for role, kind in store.FIXED_ROLES.items():
            self.assertIn(kind, store.KINDS, role)
            self.assertEqual(store.parse_names(f"{role} = zz_x\n".encode())[0][:2], (role, kind))

    BAD = {
        "no_equals": (b"scene zz_name\n", 1, "expected `role = name`"),
        "unknown_role": (b"# one\nnot_a_role = zz_name\n", 2, 'unknown role "not_a_role"'),
        "unknown_kind": (b"vector.x = zz_name\n", 1, 'unknown role "vector.x"'),
        "bad_label": (b"bool.Not-Good = zz_name\n", 1, 'unknown role "bool.not-good"'),
        "twice": (b"age = zz_a\nAGE = zz_b\n", 2, "the role age is given twice"),
        "no_name": (b"age =   \n", 1, "the role age has no name"),
        "space_in_name": (b"age = zz a\n", 1, "holds a space or a control character"),
        "non_ascii": (b"age = zz_a\nhealth = zz_\xc3\xa9\n", 2, "a byte that is not ASCII"),
        "too_long": (b"age = " + b"z" * 256 + b"\n", 1, "longer than 255 characters"),
    }

    def test_a_line_that_cannot_be_taken_is_named(self) -> None:
        for key, (raw, line, what) in self.BAD.items():
            with self.subTest(key):
                with self.assertRaises(store.NamesError) as caught:
                    store.parse_names(raw)
                self.assertEqual(caught.exception.line, line)
                self.assertIn(what, caught.exception.what)


class DescriptionTest(unittest.TestCase):
    def make(self) -> store.Description:
        return store.Description("ab" * 32, 1234, (store.Module(64, 76, 48, 56, 20, 13, "b", 3, 4),
                                                   store.Module(144, 156, 128, 136, 48, 36, "w", 4, 8)),
                                 (store.Role("age", "bool", 0xDEADBEEF), store.Role("uint.coins", "uint", 0x1234)))

    def test_round_trip(self) -> None:
        d = self.make()
        text = store.render_description(d)
        self.assertEqual(store.parse_description(text), d)
        self.assertEqual(text.splitlines()[0], "oot2-store 1")
        self.assertIn("role age bool deadbeef\n", text)
        self.assertIn("module 64 76 48 56 20 13 b 3 4\n", text)

    def test_what_is_not_a_description(self) -> None:
        good = store.render_description(self.make())
        for key, text in {"header": "oot2-store 2\n" + good.split("\n", 1)[1], "no_build": good.replace("build", "built"),
                          "bad_kind": good.replace("bool", "bit"), "short_hash": good.replace("deadbeef", "dead"),
                          "bad_class": good.replace(" b 3 4", " q 3 4"), "empty": ""}.items():
            with self.subTest(key):
                with self.assertRaises(ValueError):
                    store.parse_description(text)


@unittest.skipIf(capstone is None, "capstone is not installed")
class MatchTest(unittest.TestCase):
    def test_finds_the_cell_and_reads_every_number_from_the_instructions(self) -> None:
        cell, modules, chains, runner_up = find(blob((BOOL, 10, CELL, ""), (WORD, 10, CELL, ""), (FLOAT, 9, CELL, ""),
                                                      (STRING, 9, CELL, "")))
        self.assertEqual(cell, CELL)
        self.assertEqual(chains, 38)
        self.assertEqual(runner_up, 0)
        want = [(s.table, s.capacity, s.count, s.entries, s.stride, s.value, s.load, s.shift, s.index)
                for s in (BOOL, WORD, FLOAT, STRING)]
        self.assertEqual([m.fields() for m in modules], want)

    def test_the_layout_most_chains_agree_on_wins(self) -> None:
        other = fx.Shape(BOOL.table, BOOL.capacity, BOOL.count, BOOL.entries, 0x18, 0x12, load="b")
        _, modules, _, _ = find(blob((BOOL, 6, CELL, ""), (other, 2, CELL, ""), (WORD, 10, CELL, ""), (FLOAT, 10, CELL, "")))
        self.assertEqual(modules[0].fields()[4:6], (BOOL.stride, BOOL.value))

    def test_equal_layouts_give_the_smaller_whatever_the_order(self) -> None:
        a = fx.Shape(BOOL.table, BOOL.capacity, BOOL.count, BOOL.entries, 0x18, 0x12, load="b")
        b = fx.Shape(BOOL.table, BOOL.capacity, BOOL.count, BOOL.entries, 0x14, 0x0d, load="b")
        c = fx.Shape(BOOL.table, BOOL.capacity, BOOL.count, BOOL.entries, 0x16, 0x03, load="b")
        for order in ((a, b, c), (c, b, a), (b, a, c), (b, c, a)):
            _, modules, _, _ = find(blob(*[(sh, 4, CELL, "") for sh in order], (WORD, 10, CELL, ""), (FLOAT, 10, CELL, "")))
            self.assertEqual(modules[0].fields()[4:6], (0x14, 0x0d), "the layout that sorts first among equals")

    def test_a_table_read_two_ways_keeps_both_reads(self) -> None:
        pointer = fx.Shape(STRING.table, STRING.capacity, STRING.count, STRING.entries, STRING.stride, 0x18, load="x")
        length = fx.Shape(STRING.table, STRING.capacity, STRING.count, STRING.entries, STRING.stride, 0x20, load="w")
        _, modules, _, _ = find(blob((BOOL, 10, CELL, ""), (WORD, 10, CELL, ""), (pointer, 10, CELL, ""),
                                      (length, 10, CELL, "")))
        both = [m for m in modules if m.table == STRING.table]
        self.assertEqual(sorted((m.cls, m.value) for m in both), [("w", 0x20), ("x", 0x18)])

    def test_a_cell_among_decoys(self) -> None:
        cell, _, chains, runner_up = find(blob((BOOL, 12, CELL, ""), (WORD, 12, CELL, ""), (BOOL, 3, OTHER_CELL, ""),
                                                (WORD, 3, OTHER_CELL, "")))
        self.assertEqual((cell, chains, runner_up), (CELL, 24, 6))

    def test_tables_only_a_handed_over_object_reaches(self) -> None:
        # three tables through the cell, and a fourth that two functions are handed the object for: the same arrangement
        # of fields, one table further on at the spacing of the others
        t0 = fx.Shape(0x40, 0x4c, 0x30, 0x38, 0x14, 0x0d, load="b")
        t1 = fx.Shape(0xa0, 0xac, 0x90, 0x98, 0x30, 0x24, load="w")
        t2 = fx.Shape(0x100, 0x10c, 0xf0, 0xf8, 0x30, 0x24, load="s")
        t3 = fx.Shape(0x160, 0x16c, 0x150, 0x158, 0x50, 0x18, load="x")
        stray = fx.Shape(0x168, 0x174, 0x158, 0x160, 0x50, 0x18, load="x")   # off the spacing: not a table of the object
        lonely = fx.Shape(0x1c0, 0x1cc, 0x1b0, 0x1b8, 0x50, 0x18, load="x")   # met once only
        # each of these has the right spacing and is met three times, and differs in one field's place from the tables found
        wrongs = [fx.Shape(0x220, 0x22c, 0x200, 0x208, 0x50, 0x18, load="x"),                # the count
                  fx.Shape(0x280, 0x290, 0x270, 0x278, 0x50, 0x18, load="x"),                # the capacity
                  fx.Shape(0x2e0, 0x2ec, 0x2d0, 0x2e8, 0x50, 0x18, load="x"),                # the entries
                  fx.Shape(0x340, 0x34c, 0x330, 0x338, 0x50, 0x18, shift=4, load="x"),       # the size of a cell
                  fx.Shape(0x3a0, 0x3ac, 0x390, 0x398, 0x50, 0x18, index=8, load="x")]       # the entry number's place
        words: list[int] = []
        at = fx.CODE_AT
        for shape, copies in ((t0, 10), (t1, 10), (t2, 10)):
            more, at = functions(shape, copies, CELL, "", at)
            words += more
        for shape, copies in [(t3, 2), (stray, 3), (lonely, 1)] + [(w, 3) for w in wrongs]:
            for _ in range(copies):
                body = fx.lookup(at, CELL, shape, flaw="no_deref")   # the object is not got from the cell: it is handed over
                body[3] = fx.NOP
                body += [fx.NOP] * (0x24 - len(body))
                words += body
                at += 4 * len(body)
        _, modules, _, _ = find(fx.assemble(words))
        self.assertEqual(sorted(m.table for m in modules), [0x40, 0xa0, 0x100, 0x160])

    def test_two_tables_are_not_enough_to_place_a_third(self) -> None:
        t0 = fx.Shape(0x40, 0x4c, 0x30, 0x38, 0x14, 0x0d, load="b")
        t1 = fx.Shape(0xa0, 0xac, 0x90, 0x98, 0x30, 0x24, load="w")
        t3 = fx.Shape(0x160, 0x16c, 0x150, 0x158, 0x50, 0x18, load="x")
        words, at = functions(t0, 10, CELL, "")
        more, at = functions(t1, 10, CELL, "", at)
        words += more
        for _ in range(3):
            body = fx.lookup(at, CELL, t3, flaw="no_deref")
            body[3] = fx.NOP
            body += [fx.NOP] * (0x24 - len(body))
            words += body
            at += 4 * len(body)
        _, modules, _, _ = find(fx.assemble(words))
        self.assertEqual(sorted(m.table for m in modules), [0x40, 0xa0])

    # one near miss each: a flaw in every function of an otherwise good set breaks the match
    def refused(self, flaw: str) -> None:
        with self.assertRaises(store.MatchError):
            find(blob((BOOL, 12, CELL, flaw), (WORD, 12, CELL, flaw)))

    def test_near_miss_divisor_from_another_object(self) -> None:
        self.refused("other_object")

    def test_near_miss_no_table_pointer_of_the_object(self) -> None:
        self.refused("no_table")

    def test_near_miss_entry_not_multiplied_by_the_entry_size(self) -> None:
        self.refused("no_stride")

    def test_near_miss_entry_number_not_checked_against_the_count(self) -> None:
        self.refused("no_count")

    def test_near_miss_cell_word_taken_for_the_object(self) -> None:
        self.refused("no_deref")

    def test_near_miss_signed_division(self) -> None:
        self.refused("signed_div")

    def test_near_miss_no_value_read(self) -> None:
        self.refused("no_load")

    def test_the_good_set_is_found_by_the_same_calls(self) -> None:
        find(blob((BOOL, 12, CELL, ""), (WORD, 12, CELL, "")))

    def test_too_few_chains(self) -> None:
        find(blob((BOOL, store.MIN_CHAINS, CELL, "")))
        with self.assertRaises(store.MatchError) as caught:
            find(blob((BOOL, store.MIN_CHAINS - 1, CELL, "")))
        self.assertIn("stands out", str(caught.exception))

    def test_two_cells_that_cannot_be_told_apart(self) -> None:
        with self.assertRaises(store.MatchError) as caught:
            find(blob((BOOL, 12, CELL, ""), (BOOL, 12, OTHER_CELL, "")))
        self.assertIn("stands out", str(caught.exception))
        find(blob((BOOL, 3 * 4, CELL, ""), (BOOL, 4, OTHER_CELL, "")))   # exactly the margin
        with self.assertRaises(store.MatchError):
            find(blob((BOOL, 3 * 4 - 1, CELL, ""), (BOOL, 4, OTHER_CELL, "")))

    def test_no_division_at_all(self) -> None:
        with self.assertRaises(store.MatchError) as caught:
            find(fx.assemble([fx.NOP] * 64))
        self.assertIn("no division", str(caught.exception))


def namespace(names: Path | None) -> argparse.Namespace:
    return argparse.Namespace(names=names)


class Fake(Step):
    id = "modules"


@unittest.skipIf(capstone is None, "capstone is not installed")
class StepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / "game"
        (self.dest / "exefs").mkdir(parents=True)
        self.build = bytes(range(32))
        code = blob((BOOL, 10, CELL, ""), (WORD, 10, CELL, ""), (STRING, 9, CELL, ""))
        (self.dest / "exefs" / "main").write_bytes(fx.make_nso(code, build=self.build, text=TEXT))
        self.names = Path(self.tmp.name) / "names.txt"
        self.names.write_bytes(b"scene = zz_scene_name\nage = zz_age_flag\nhealth = zz_health_value\n")

    def run_step(self, names: Path | None = None) -> tuple[str, Context]:
        out = io.StringIO()
        rep = Reporter(out=out, err=out)
        from prepare import state
        ctx = Context(self.dest, namespace(names or self.names), rep, state=state.load(self.dest))
        result = run_steps([Fake(), store.Store()], ctx)
        return result + "\n" + out.getvalue(), ctx

    def test_the_names_file_round_trips_through_prepare(self) -> None:
        text, ctx = self.run_step()
        self.assertTrue(text.startswith("ok"), text)
        written = (self.dest / store.FILE).read_text("ascii")
        d = store.parse_description(written)
        self.assertEqual(d.build, self.build.hex())
        self.assertEqual(d.cell, CELL)
        self.assertEqual([m.table for m in d.modules], [BOOL.table, WORD.table, STRING.table])
        self.assertEqual([(r.role, r.kind, r.hash) for r in d.roles],
                         [("scene", "string", store.murmur3_32(b"zz_scene_name")),
                          ("age", "bool", store.murmur3_32(b"zz_age_flag")),
                          ("health", "int", store.murmur3_32(b"zz_health_value"))])
        record = ctx.state["steps"]["store"]
        self.assertEqual(record["tables"], 3)
        self.assertEqual(record["role_hashes"]["age"], f"{store.murmur3_32(b'zz_age_flag'):08x}")
        self.assertEqual(record["file_sha256"], hashlib.sha256(written.encode()).hexdigest())

    def test_the_names_are_never_written(self) -> None:
        text, ctx = self.run_step()
        written = (self.dest / store.FILE).read_bytes() + (self.dest / "oot2-prepare.json").read_bytes() + text.encode()
        for name in (b"zz_scene_name", b"zz_age_flag", b"zz_health_value"):
            self.assertNotIn(name, written)
        self.assertEqual([p.name for p in self.dest.iterdir() if p.name.endswith(".tmp")], [])

    def test_a_second_run_keeps_the_step_and_other_names_run_it_again(self) -> None:
        self.run_step()
        first = (self.dest / store.FILE).read_bytes()
        text, _ = self.run_step()
        self.assertIn("store: kept", text)
        self.names.write_bytes(b"scene = zz_other_scene\n")
        text, _ = self.run_step()
        self.assertIn("store: started (its inputs changed)", text)
        self.assertNotEqual((self.dest / store.FILE).read_bytes(), first)

    def test_a_deleted_description_is_written_again(self) -> None:
        self.run_step()
        (self.dest / store.FILE).unlink()
        text, _ = self.run_step()
        self.assertIn("store: started (its output is missing)", text)
        self.assertTrue((self.dest / store.FILE).is_file())

    def test_a_names_file_with_a_bad_line_stops_by_line(self) -> None:
        self.names.write_bytes(b"scene = zz_a\nnot_a_role = zz_b\n")
        text, _ = self.run_step()
        self.assertTrue(text.startswith("failed"), text)
        self.assertIn("the names file", text)
        self.assertIn("line 2: unknown role", text)
        self.assertNotIn("internal error", text)
        self.assertFalse((self.dest / store.FILE).exists())

    def test_a_missing_names_file_stops_by_name(self) -> None:
        text, _ = self.run_step(Path(self.tmp.name) / "no-such-names.txt")
        self.assertTrue(text.startswith("failed"), text)
        self.assertIn("no-such-names.txt cannot be read", text)

    def test_a_game_without_the_store_stops_by_name(self) -> None:
        (self.dest / "exefs" / "main").write_bytes(fx.make_nso(fx.assemble([fx.NOP] * 16), text=TEXT))
        text, _ = self.run_step()
        self.assertTrue(text.startswith("failed"), text)
        self.assertIn("the value store was not found in the main module", text)
        self.assertFalse((self.dest / store.FILE).exists())

    def test_a_role_no_table_can_hold_stops_by_name(self) -> None:
        self.names.write_bytes(b"float.speed = zz_speed\n")   # the synthetic game has no table read as a float
        text, _ = self.run_step()
        self.assertTrue(text.startswith("failed"), text)
        self.assertIn("none is read as float", text)


class OffTest(unittest.TestCase):
    def test_the_steps_are_the_same_without_a_names_file(self) -> None:
        from oot2 import steps_for
        base = [s.id for s in all_steps()]
        self.assertNotIn("store", base)
        self.assertEqual([s.id for s in steps_for(argparse.Namespace(names=None))], base)
        with_names = [s.id for s in steps_for(argparse.Namespace(names=Path("names.txt")))]
        self.assertEqual(with_names, base[:base.index("modules") + 1] + ["store"] + base[base.index("modules") + 1:])

    def test_prepare_stops_on_a_bad_names_file_before_any_step(self) -> None:
        import contextlib
        import oot2
        with tempfile.TemporaryDirectory() as tmp:
            names = Path(tmp) / "names.txt"
            names.write_bytes(b"scene = zz_a\nnot_a_role = zz_b\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = oot2.prepare(argparse.Namespace(dest=Path(tmp) / "game", names=names))
            self.assertEqual(code, oot2.EXIT["failed"])
            self.assertIn("line 2: unknown role", err.getvalue())
            self.assertFalse((Path(tmp) / "game").exists(), "nothing was started")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = oot2.prepare(argparse.Namespace(dest=Path(tmp) / "game", names=Path(tmp) / "no-such.txt"))
            self.assertEqual(code, oot2.EXIT["failed"])
            self.assertIn("no-such.txt cannot be read", err.getvalue())

    def test_the_option_is_documented_and_defaults_off(self) -> None:
        import oot2
        text = oot2.__doc__ or ""
        self.assertIn("--names", text)


if __name__ == "__main__":
    unittest.main()
