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
# Arbitrary layouts: every displacement, entry size, cell shape and the spacing of the tables below was drawn once at random for
# this file, with the alignment the loads need and nothing else in common. Each table has a cell shape of its own, and the
# key is at the start of the cell for one of them (loaded with a register for the offset) and elsewhere for the rest.
BOOL = fx.Shape(table=0x118, capacity=0x2bc, count=0x1e8, entries=0x248, stride=0x10a, value=0x6d, shift=6, key=0xc, index=0x8, load="b")
WORD = fx.Shape(table=0x138, capacity=0x10c, count=0x84, entries=0x2c0, stride=0xe4, value=0x2c, shift=5, key=0x18, index=0x8, load="w")
FLOAT = fx.Shape(table=0x168, capacity=0x274, count=0x294, entries=0x90, stride=0x24, value=0x14, shift=5, key=0x0, index=0xc, load="s")
STRING = fx.Shape(table=0x2e8, capacity=0xc0, count=0x250, entries=0x238, stride=0x5c, value=0x28, shift=4, key=0x4, index=0xc, load="x")


def functions(shape: fx.Shape, copies: int, cell: int = CELL, flaw: str = "", start: int = fx.CODE_AT,
              key_form: str = "") -> tuple[list[int], int]:
    """`copies` functions that look a value up, back to back from `start`; the words and the address after them."""
    words: list[int] = []
    at = start
    for _ in range(copies):
        body = fx.lookup(at, cell, shape, flaw=flaw, key_form=key_form)
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
        return store.Description("ab" * 32, 1234, tuple(self.modules()),
                                 (store.Role("age", "bool", 0xDEADBEEF), store.Role("uint.coins", "uint", 0x1234)))

    def modules(self) -> list[store.Module]:
        return [store.Module(s.table, s.capacity, s.count, s.entries, s.stride, s.value, s.load, s.shift, s.key, s.index)
                for s in (BOOL, WORD)]

    def test_round_trip(self) -> None:
        d = self.make()
        text = store.render_description(d)
        self.assertEqual(store.parse_description(text), d)
        self.assertEqual(text.splitlines()[0], "oot2-store 2")
        self.assertIn("role age bool deadbeef\n", text)
        b = BOOL
        self.assertIn(f"module {b.table} {b.capacity} {b.count} {b.entries} {b.stride} {b.value} b {b.shift} {b.key} {b.index}\n",
                      text)

    def test_what_is_not_a_description(self) -> None:
        good = store.render_description(self.make())
        shape = f" b {BOOL.shift} {BOOL.key} {BOOL.index}"
        self.assertIn(shape, good)
        size = 1 << BOOL.shift
        for key, text in {"header": "oot2-store 3\n" + good.split("\n", 1)[1], "no_build": good.replace("build", "built"),
                          "bad_kind": good.replace("bool", "bit"), "short_hash": good.replace("deadbeef", "dead"),
                          "bad_class": good.replace(shape, f" q {BOOL.shift} {BOOL.key} {BOOL.index}"), "empty": "",
                          "no_key": good.replace(shape, f" b {BOOL.shift} {BOOL.index}"),
                          "key_outside_the_cell": good.replace(shape, f" b {BOOL.shift} {size} {BOOL.index}"),
                          "key_over_the_edge": good.replace(shape, f" b {BOOL.shift} {size - 2} {BOOL.index}"),
                          "index_outside_the_cell": good.replace(shape, f" b {BOOL.shift} {BOOL.key} {size}"),
                          "key_on_the_entry_number": good.replace(shape, f" b {BOOL.shift} {BOOL.key} {BOOL.key}"),
                          "key_overlaps_the_entry_number": good.replace(shape, f" b {BOOL.shift} {BOOL.key} {BOOL.key + 2}"),
                          "cell_larger_than_64": good.replace(shape, f" b 7 {BOOL.key} {BOOL.index}")}.items():
            with self.subTest(key):
                with self.assertRaises(ValueError):
                    store.parse_description(text)

    def test_format_1_is_refused_by_name(self) -> None:
        good = store.render_description(self.make())
        old = "oot2-store 1\n" + good.split("\n", 1)[1]
        with self.assertRaises(ValueError) as caught:
            store.parse_description(old)
        self.assertIn("format 1", str(caught.exception))
        self.assertIn("line 1", str(caught.exception))


@unittest.skipIf(capstone is None, "capstone is not installed")
class MatchTest(unittest.TestCase):
    def test_finds_the_cell_and_reads_every_number_from_the_instructions(self) -> None:
        cell, modules, chains, runner_up = find(blob((BOOL, 10, CELL, ""), (WORD, 10, CELL, ""), (FLOAT, 9, CELL, ""),
                                                      (STRING, 9, CELL, "")))
        self.assertEqual(cell, CELL)
        self.assertEqual(chains, 38)
        self.assertEqual(runner_up, 0)
        want = [(s.table, s.capacity, s.count, s.entries, s.stride, s.value, s.load, s.shift, s.key, s.index)
                for s in (BOOL, WORD, FLOAT, STRING)]
        self.assertEqual([m.fields() for m in modules], want)

    def test_the_key_is_read_from_the_load_that_feeds_the_compare(self) -> None:
        # the same table, the key in three places: with a displacement from the cell, with a register for the offset (the
        # start of the cell), and with a displacement of 0
        nonzero = fx.Shape(table=0x1d8, capacity=0x23c, count=0x1f0, entries=0x208, stride=0x6c, value=0x20, shift=5, key=0x14,
                           index=0x4, load="w")
        zero = fx.Shape(nonzero.table, nonzero.capacity, nonzero.count, nonzero.entries, nonzero.stride, nonzero.value,
                        nonzero.shift, 0, 0x10, "w")
        for shape, form, key in ((nonzero, "disp", 0x14), (zero, "reg", 0), (zero, "disp", 0)):
            with self.subTest(form=form, key=key):
                code = fx.assemble(functions(shape, store.MIN_CHAINS, CELL, "", key_form=form)[0])
                _, modules, _, _ = find(code)
                self.assertEqual([(m.key, m.index) for m in modules], [(key, shape.index)])

    def test_the_key_of_each_table_is_read_for_that_table(self) -> None:
        # two tables of one object whose cells differ in where the key is: neither is taken for the other's
        a = fx.Shape(table=0x58, capacity=0x7c, count=0x60, entries=0x98, stride=0x31, value=0x0b, shift=4, key=0x8, index=0x0,
                     load="b")
        b = fx.Shape(table=0xd0, capacity=0xf4, count=0xe0, entries=0x110, stride=0x44, value=0x1c, shift=4, key=0x0, index=0xc,
                     load="w")
        _, modules, _, _ = find(blob((a, 12, CELL, ""), (b, 12, CELL, "")))
        self.assertEqual([(m.table, m.key, m.index) for m in modules], [(a.table, 0x8, 0x0), (b.table, 0x0, 0xc)])

    def test_the_layout_most_chains_agree_on_wins(self) -> None:
        other = fx.Shape(BOOL.table, BOOL.capacity, BOOL.count, BOOL.entries, 0x71, 0x31, BOOL.shift, BOOL.key, BOOL.index, "b")
        _, modules, _, _ = find(blob((BOOL, 6, CELL, ""), (other, 2, CELL, ""), (WORD, 10, CELL, ""), (FLOAT, 10, CELL, "")))
        self.assertEqual(modules[0].fields()[4:6], (BOOL.stride, BOOL.value))

    def test_equal_layouts_give_the_smaller_whatever_the_order(self) -> None:
        def variant(stride: int, value: int) -> fx.Shape:
            return fx.Shape(BOOL.table, BOOL.capacity, BOOL.count, BOOL.entries, stride, value, BOOL.shift, BOOL.key, BOOL.index,
                            "b")
        a, b, c = variant(0xe1, 0x7f), variant(0x5a, 0x03), variant(0xd8, 0x67)
        for order in ((a, b, c), (c, b, a), (b, a, c), (b, c, a)):
            _, modules, _, _ = find(blob(*[(sh, 4, CELL, "") for sh in order], (WORD, 10, CELL, ""), (FLOAT, 10, CELL, "")))
            self.assertEqual(modules[0].fields()[4:6], (0x5a, 0x03), "the layout that sorts first among equals")

    def test_a_table_read_two_ways_keeps_both_reads(self) -> None:
        def read(value: int, load: str) -> fx.Shape:
            return fx.Shape(STRING.table, STRING.capacity, STRING.count, STRING.entries, STRING.stride, value, STRING.shift,
                            STRING.key, STRING.index, load)
        pointer, length = read(STRING.value, "x"), read(0x4c, "w")
        _, modules, _, _ = find(blob((BOOL, 10, CELL, ""), (WORD, 10, CELL, ""), (pointer, 10, CELL, ""),
                                      (length, 10, CELL, "")))
        both = [m for m in modules if m.table == STRING.table]
        self.assertEqual(sorted((m.cls, m.value) for m in both), [("w", 0x4c), ("x", STRING.value)])

    def test_a_cell_among_decoys(self) -> None:
        cell, _, chains, runner_up = find(blob((BOOL, 12, CELL, ""), (WORD, 12, CELL, ""), (BOOL, 3, OTHER_CELL, ""),
                                                (WORD, 3, OTHER_CELL, "")))
        self.assertEqual((cell, chains, runner_up), (CELL, 24, 6))

    # The tables of one object share the arrangement of their own fields and sit a fixed distance apart (drawn at random)
    SPACING, FIRST = 0xa8, 0x68
    ARRANGEMENT = {"cap": 0x40, "count": 0x24, "entries": 0x30, "shift": 4, "key": 0x8, "index": 0x4}

    def table_at(self, table: int, load: str, stride: int, value: int, **changes: int) -> fx.Shape:
        a = {**self.ARRANGEMENT, **changes}
        return fx.Shape(table=table, capacity=table + a["cap"], count=table + a["count"], entries=table + a["entries"],
                        stride=stride, value=value, shift=a["shift"], key=a["key"], index=a["index"], load=load)

    def handed_over(self, shapes: list[tuple[fx.Shape, int]], through_cell: list[tuple[fx.Shape, int]]) -> list[store.Module]:
        words: list[int] = []
        at = fx.CODE_AT
        for shape, copies in through_cell:
            more, at = functions(shape, copies, CELL, "", at)
            words += more
        for shape, copies in shapes:
            for _ in range(copies):
                body = fx.lookup(at, CELL, shape, flaw="no_deref")   # the object is not got from the cell: it is handed over
                body[3] = fx.NOP
                body += [fx.NOP] * (0x24 - len(body))
                words += body
                at += 4 * len(body)
        return list(find(fx.assemble(words))[1])

    def spaced(self, k: int) -> int:
        return self.FIRST + k * self.SPACING

    def test_tables_only_a_handed_over_object_reaches(self) -> None:
        # three tables through the cell, and a fourth that two functions are handed the object for: the same arrangement
        # of fields, one table further on at the spacing of the others
        t0 = self.table_at(self.spaced(0), "b", 0xe5, 0x51)
        t1 = self.table_at(self.spaced(1), "w", 0x48, 0x14)
        t2 = self.table_at(self.spaced(2), "s", 0x118, 0x4c)
        t3 = self.table_at(self.spaced(3), "x", 0x120, 0x38)
        stray = self.table_at(self.spaced(3) + 0x98, "x", 0x44, 0x08)   # off the spacing: not a table of the object
        lonely = self.table_at(self.spaced(4), "x", 0xe0, 0x90)         # met once only
        # each of these has the right spacing and is met three times, and differs in one field's place from the tables found
        changes = [{"count": 0x2c}, {"cap": 0x48}, {"entries": 0x38}, {"shift": 5}, {"key": 0xc}, {"key": 0x8, "index": 0xc}]
        wrongs = [self.table_at(self.spaced(5 + n), "x", 0xc0, 0x70, **change) for n, change in enumerate(changes)]
        modules = self.handed_over([(t3, 2), (stray, 3), (lonely, 1)] + [(w, 3) for w in wrongs],
                                   [(t0, 10), (t1, 10), (t2, 10)])
        self.assertEqual(sorted(m.table for m in modules), [t0.table, t1.table, t2.table, t3.table])

    def test_the_place_of_the_key_is_part_of_the_arrangement(self) -> None:
        # the same as the last, with only the key's place differing in the handed-over table: it is not the object's
        t0 = self.table_at(self.spaced(0), "b", 0xe5, 0x51)
        t1 = self.table_at(self.spaced(1), "w", 0x48, 0x14)
        t2 = self.table_at(self.spaced(2), "s", 0x118, 0x4c)
        same = self.table_at(self.spaced(3), "x", 0x120, 0x38)
        other_key = self.table_at(self.spaced(3), "x", 0x120, 0x38, key=0xc)
        found = [(t0, 10), (t1, 10), (t2, 10)]
        self.assertEqual(len(self.handed_over([(same, 3)], found)), 4)
        self.assertEqual(len(self.handed_over([(other_key, 3)], found)), 3)

    def test_two_tables_are_not_enough_to_place_a_third(self) -> None:
        t0 = self.table_at(self.spaced(0), "b", 0xe5, 0x51)
        t1 = self.table_at(self.spaced(1), "w", 0x48, 0x14)
        t3 = self.table_at(self.spaced(3), "x", 0x120, 0x38)
        modules = self.handed_over([(t3, 3)], [(t0, 10), (t1, 10)])
        self.assertEqual(sorted(m.table for m in modules), [t0.table, t1.table])

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

    def test_near_miss_the_cells_key_is_never_compared(self) -> None:
        self.refused("no_key")

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
