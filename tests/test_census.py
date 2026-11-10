# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for `oot2 census` (tools/census/): the scene list decoded from a RomFS index and its image.

Every input is synthetic and built inside the test: a small RomFS of made-up scenes, folders and files, a BYML writer of
a few lines, an index in the layout tools/prepare/extract.py documents, and an image that is the files one after the
other. No game file is read and nothing is downloaded. The files are stored plain, so the tests need no zstd; the tests
of a zstd frame skip on a Python without compression.zstd.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import census  # noqa: E402
import oot2  # noqa: E402
from census import byml, rules as rulesfile  # noqa: E402
from census.build import Assigner, build, name_parts  # noqa: E402
from census.source import CensusError, Source  # noqa: E402
from prepare import extract  # noqa: E402


# --- a BYML writer: just enough for the reader's tests ----------------------------------------------------------------

def _table(items: list[str]) -> bytes:
    n = len(items)
    offs, blob, at = [], b"", 4 + 4 * (n + 1)
    for item in items:
        offs.append(at + len(blob))
        blob += item.encode() + b"\0"
    offs.append(at + len(blob))
    data = bytes([0xC2]) + n.to_bytes(3, "little") + struct.pack(f"<{n + 1}I", *offs) + blob
    return data + b"\0" * (-len(data) % 4)


def _walk(value, keys: set, strings: set) -> None:
    if isinstance(value, str):
        strings.add(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            keys.add(k)
            _walk(v, keys, strings)
    elif isinstance(value, list):
        for v in value:
            _walk(v, keys, strings)


def encode_byml(root) -> bytes:
    keys, strings = set(), set()
    _walk(root, keys, strings)
    keys, strings = sorted(keys), sorted(strings)
    out = bytearray(16)
    key_at = len(out)
    out += _table(keys)
    str_at = len(out)
    out += _table(strings)

    def node(v):
        if isinstance(v, bool):
            return 0xD0, int(v)
        if isinstance(v, int):
            return 0xD1, v & 0xFFFFFFFF
        if isinstance(v, float):
            return 0xD2, struct.unpack("<I", struct.pack("<f", v))[0]
        if isinstance(v, str):
            return 0xA0, strings.index(v)
        if v is None:
            return 0xFF, 0
        if isinstance(v, list):
            items = [node(x) for x in v]
            at = len(out)
            out.extend(bytes([0xC0]) + len(v).to_bytes(3, "little") + bytes(t for t, _ in items))
            out.extend(b"\0" * (-len(out) % 4))
            for _, word in items:
                out.extend(struct.pack("<I", word))
            return 0xC0, at
        items = sorted((keys.index(k), node(x)) for k, x in v.items())
        at = len(out)
        out.extend(bytes([0xC1]) + len(v).to_bytes(3, "little"))
        for key, (t, word) in items:
            out.extend(key.to_bytes(3, "little") + bytes([t]) + struct.pack("<I", word))
        return 0xC1, at

    _, root_at = node(root)
    out[0:16] = b"YB" + struct.pack("<HIII", 7, key_at, str_at, root_at)
    return bytes(out)


# --- a made-up game -----------------------------------------------------------------------------------------------

POSE = {"Rot": [0.0, 0.0, 0.0], "Pos": [1.0, 2.0, 3.0]}
FLOW = b"flow"


def game_files() -> dict[str, bytes]:
    f = {}

    def scene(sid, params):
        f[f"Zone/Defs/{sid}.scn.pyml"] = encode_byml(params)

    scene("Alpha-Young", {"Force": True, "Region": 1, "Fun": False,
                          "Res": {"Main": "Pool/Alpha/Alpha-Young.map.json"},
                          "Near": [{"Target": "Beta", "Where": [1.0, 2.0, 3.0]}]})
    scene("Alpha-Old", {"Inherit": "Pool/Defs/Alpha-Young.scn.pyml", "Force": True, "WhenEnter": False,
                        "Res": {"Main": "Pool/Alpha/Alpha-Old.map.json"}})
    scene("Beta", {"WhenEnter": True, "Fun": True, "Ground": "Hills-01",
                   "Res": {"Main": "Pool/Beta/Beta.map.json"}})
    scene("Gamma", {"Inherit": "Pool/Defs/Beta.scn.pyml", "Res": {"Main": "Pool/Beta/Gamma.map.json"}})
    f["Zone/Alpha/Alpha-Young-Fixed.map.byml"] = encode_byml(
        {"List": [{"Go": "Beta"}, {"Ev": "Alpha-0001"}, {"Ev": "Alpha-0002"}]})
    f["Zone/Alpha/Alpha-Young-Live.map.byml"] = encode_byml({"X": "Mini-Game", "Y": "Both-0001"})
    f["Zone/Alpha/Alpha-Old-Fixed.map.byml"] = encode_byml({"Go": "Alpha-Young"})
    f["Zone/Beta/Beta-Fixed.map.byml"] = encode_byml({"Ev": "Beta-0001"})
    f["Zone/Beta/Beta-Live.map.byml"] = encode_byml({"Q": "Beta-ClashGo", "Z": "Both-0002"})
    f["Zone/Beta/Gamma-Fixed.map.byml"] = encode_byml({"Go": "Alpha-Old"})
    f["Zone/Tables/Spots.byml"] = encode_byml({"Alpha-Young": {"": POSE, "Gate": POSE}, "Beta": {"Door": POSE},
                                              "Gamma": {}})
    f["Zone/Tables/Notes.byml"] = encode_byml({"Alpha-Old": {"x": 1}, "Beta": {"y": 2}})
    f["Zone/Tables/readme.txt"] = b"not a table"
    for name in ("Alpha-0001", "Alpha-0002", "Alpha-0003", "Alpha-0004", "Beta-0001",
                 "Beta-ClashGo", "Mini-Game", "Far-0001", "Both-0001", "Both-0002", "Both-0003"):
        f[f"Show/Flow/{name}.flow"] = FLOW
    for name in ("Beta-0001", "Alpha-0004", "Lone-0001"):
        f[f"Show/Clip/{name}.clip"] = b"clip"
    f["Brain/Alpha-Shared-Prefab.brn"] = b"\0\0Alpha-0003\0junk"
    f["Brain/Other-Thing.brn"] = b"\0Far-0001\0"
    for d in ("Alpha-Young", "Alpha-Old", "Beta", "Beta-Aged", "Odd-One"):
        f[f"Cull/{d}/a.bin"] = d.encode()
    f["Land/Hills-01/a.bin"] = b"hills"
    f["Flow/Halt.part.brn"] = b"pause"
    f["Flow/Chart.part.brn"] = b"map"
    return f


RULES = {
    "scene_files": {"dir": "Zone/Defs", "suffix": ".scn.pyml"},
    "age_keys": {"force_switch": "Force", "child_on_change": "WhenEnter"},
    "age_tokens": {"Young": "child", "Old": "adult", "Aged": "adult"},
    "events": {"dir": "Show/Flow"},
    "event_sources": ["Brain"],
    "movies": {"dir": "Show/Clip"},
    "tags": [{"tag": "fun", "key": "Fun"}, {"tag": "boss", "event": "-Clash(Go|Done)$"},
             {"tag": "mini", "event": "^Mini-"}, {"tag": "finale", "id": "Gamma"},
             {"tag": "home", "place": "Alpha"}],
    "menus": [{"id": "pause", "path": "Flow/Halt.part.brn"}, {"id": "map", "path": "Flow/Chart.part.brn"}],
    "naming": {"separator": "-", "family_parts": 1, "min_word": 5},
}


def index_bytes(rows: list[tuple[str, int, int, bytes]], image_size: int) -> bytes:
    names, body = bytearray(), bytearray()
    for path, off, size, sha in sorted(rows, key=lambda r: r[0].encode()):
        raw = path.encode()
        body += extract.INDEX_ROW.pack(off, size, len(names), len(raw), sha)
        names += raw + b"\0"
    names_off = extract.INDEX_HEADER.size + len(body)
    head = extract.INDEX_HEADER.pack(extract.INDEX_MAGIC, 1, extract.INDEX_ROW.size, len(rows),
                                     extract.INDEX_HEADER.size, names_off, len(names), image_size, len(rows))
    return head + bytes(body) + bytes(names)


def make_game(root: Path, files: dict[str, bytes], in_place: bool = False, order=None, trim: int = 0,
              flip_in: str | None = None) -> Path:
    """A prepared folder: romfs/romfs-index.bin, and the image as romfs/romfs.bin or, in place, a .nup beside it.
    The image can be cut short by trim bytes, or have one byte of a file changed after the index was made."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "romfs").mkdir(exist_ok=True)
    image = bytearray(b"\0" * 32)                       # a header, so that no file sits at offset 0
    rows = []
    for path in (order or sorted(files)):
        data = files[path]
        image.extend(b"\0" * (-len(image) % 16))
        rows.append((path, len(image), len(data), hashlib.sha256(data).digest()))
        image.extend(data)
    if flip_in:
        off = next(r[1] for r in rows if r[0] == flip_in)
        image[off] ^= 0xFF
    if trim:
        del image[-trim:]
    (root / "romfs" / "romfs-index.bin").write_bytes(index_bytes(rows, len(image)))
    if in_place:
        nup = root.parent / (root.name + ".nup")
        nup.write_bytes(bytes(image))
        (root / "oot2-prepare.json").write_text(json.dumps({
            "format": 1, "inputs": {},
            "steps": {"extract": {"key": "k", "finished": "t", "seconds": 1.0, "romfs_form": "in-place",
                                  "nup": {"path": str(nup)}}}}), encoding="utf-8")
    else:
        (root / "romfs" / "romfs.bin").write_bytes(bytes(image))
    return root


def make_rules(root: Path, **changes) -> Path:
    data = {**RULES, **changes}
    path = root / "rules.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def census(self, files=None, rules_changes=None, **kw) -> dict:
        game = make_game(self.tmp / "game", files or game_files(), **kw)
        rules = rulesfile.load(make_rules(self.tmp, **(rules_changes or {})))
        with Source(game / "romfs" / "romfs-index.bin", game / "romfs" / "romfs.bin") as src:
            return build(src, rules)

    def scene(self, c: dict, sid: str) -> dict:
        return next(s for s in c["scenes"] if s["id"] == sid)


class BymlTest(unittest.TestCase):
    def test_round_trip(self):
        value = {"a": [1, -2, 0.5, True, None, "s"], "b": {"c": "d", "e": []}, "f": 3}
        self.assertEqual(byml.load(encode_byml(value)), value)

    def test_damaged_files_are_refused_by_error_not_by_crash(self):
        good = encode_byml({"a": [1, 2, {"b": "c"}]})
        for bad in (b"", b"YB", b"XX" + good[2:], good[:20], good[:-8], good[:16] + b"\xff" * (len(good) - 16)):
            with self.assertRaises(byml.BymlError):
                byml.load(bad)

    def test_a_nesting_loop_is_refused(self):
        # the root array's only element is the root array itself
        data = bytearray(encode_byml([[0]]))
        root = struct.unpack_from("<I", data, 12)[0]
        inner = struct.unpack_from("<I", data, root + 8)[0]          # the element's offset (after type bytes, padded)
        data[inner + 4] = 0xC0                                       # ... and it is an array that points back
        struct.pack_into("<I", data, inner + 8, root)
        with self.assertRaises(byml.BymlError):
            byml.load(bytes(data))


    def test_too_many_nodes_are_refused(self):
        # forty small arrays: well under a kilobyte, and over the bound once it is lowered
        data = bytearray(encode_byml([[1, 2, 3]] * 40))
        with mock.patch.object(byml, "MAX_NODES", 100):
            with self.assertRaises(byml.BymlError) as cm:
                byml.load(bytes(data))
        self.assertIn("too many nodes", str(cm.exception))
        self.assertEqual(len(byml.load(bytes(data))), 40)


class AssignerTest(unittest.TestCase):
    def test_a_name_belongs_to_the_longest_id_it_starts_with(self):
        a = Assigner({"A", "A-B", "A-B-C", "X-1", "X-2"}, "-")
        self.assertEqual(a.of("A-B-C-Fixed.map.x"), ("A-B-C", "Fixed"))
        self.assertEqual(a.of("A-B-Fixed.map.x"), ("A-B", "Fixed"))
        self.assertEqual(a.of("A-Zzz"), ("A", "Zzz"))
        self.assertEqual(a.of("A-B"), ("A-B", ""))
        self.assertIsNone(a.of("AB-C"))                    # only at an underscore
        self.assertIsNone(a.of("Q-B"))

    def test_a_shared_leading_part_belongs_to_every_scene_that_has_it(self):
        a = Assigner({"A", "A-B", "X-1", "X-2", "Z-1"}, "-")
        self.assertEqual(a.family("Z-Part"), frozenset())              # one scene alone does not make a family
        self.assertEqual(a.family("X-Joint-Prefab.brn"), frozenset(("X-1", "X-2")))
        self.assertEqual(a.family("X-1-Part"), frozenset(("X-1",)))
        self.assertEqual(a.family("A-B-Part"), frozenset(("A-B",)))
        self.assertEqual(a.family("Y-Joint"), frozenset())
        self.assertEqual(a.family("X"), frozenset(("X-1", "X-2")))


class CensusTest(Tmp):
    def test_scenes_places_and_ages(self):
        c = self.census()
        self.assertEqual([s["id"] for s in c["scenes"]], ["Alpha-Old", "Alpha-Young", "Beta", "Gamma"])
        self.assertEqual({p["place"]: p["scenes"] for p in c["places"]},
                         {"Alpha": ["Alpha-Old", "Alpha-Young"], "Beta": ["Beta", "Gamma"]})
        ages = {s["id"]: s["ages"] for s in c["scenes"]}
        # Alpha_Old: its name, and the flag it sets. Alpha_Young: its name, forced with no age given (child).
        # Beta: the flag, and the folder "Beta-Aged" elsewhere. Gamma inherits Beta's flag.
        self.assertEqual(ages, {"Alpha-Old": ["adult"], "Alpha-Young": ["child"], "Beta": ["adult", "child"],
                                "Gamma": ["child"]})
        self.assertEqual(self.scene(c, "Beta")["age_from"], ["flag", "folders"])
        self.assertEqual(self.scene(c, "Alpha-Young")["age_from"], ["forced, age not given", "name"])
        self.assertEqual(c["summary"]["parent_key"], "Inherit")

    def test_entrances_come_from_the_table_that_holds_poses(self):
        c = self.census()
        self.assertEqual(c["summary"]["entrance_table"], "Zone/Tables/Spots.byml")
        self.assertEqual(self.scene(c, "Alpha-Young")["entrances"], ["", "Gate"])
        self.assertEqual(self.scene(c, "Gamma")["entrances"], [])
        self.assertEqual(self.scene(c, "Alpha-Old")["entrances"], [])
        self.assertEqual(self.scene(c, "Alpha-Old")["tables"], ["Notes.byml"])
        self.assertEqual(c["summary"]["tables"], ["Zone/Tables/Notes.byml", "Zone/Tables/Spots.byml"])

    def test_ways_in_and_out(self):
        c = self.census()
        self.assertEqual(self.scene(c, "Alpha-Young")["links_out"], ["Beta"])
        # Alpha_Old inherits Alpha_Young's neighbours and names Alpha_Young in its own map file
        self.assertEqual(self.scene(c, "Alpha-Old")["links_out"], ["Alpha-Young", "Beta"])
        self.assertEqual(self.scene(c, "Gamma")["links_out"], ["Alpha-Old"])
        self.assertEqual(self.scene(c, "Beta")["entered_from"], ["Alpha-Old", "Alpha-Young"])
        self.assertEqual(self.scene(c, "Alpha-Young")["entered_from"], ["Alpha-Old"])
        self.assertEqual(self.scene(c, "Gamma")["entered_from"], [])

    def test_map_files_dirs_and_events(self):
        c = self.census()
        self.assertEqual([f["path"] for f in self.scene(c, "Alpha-Young")["files"]],
                         ["Zone/Alpha/Alpha-Young-Fixed.map.byml", "Zone/Alpha/Alpha-Young-Live.map.byml"])
        self.assertEqual([f["path"] for f in self.scene(c, "Alpha-Old")["files"]],
                         ["Zone/Alpha/Alpha-Old-Fixed.map.byml"])
        self.assertEqual(self.scene(c, "Beta")["dirs"], ["Cull/Beta", "Cull/Beta-Aged", "Land/Hills-01", "Zone/Beta"])
        self.assertEqual(self.scene(c, "Alpha-Young")["dirs"], ["Cull/Alpha-Young"])
        # a map file names its events; a file of another tree named for the place gives them to both of its scenes
        self.assertEqual(self.scene(c, "Alpha-Young")["events"],
                         ["Alpha-0001", "Alpha-0002", "Alpha-0003", "Both-0001", "Mini-Game"])
        self.assertEqual(self.scene(c, "Alpha-Old")["events"], ["Alpha-0003"])
        self.assertEqual(self.scene(c, "Beta")["events"], ["Beta-0001", "Beta-ClashGo", "Both-0002"])

    def test_families_and_movies(self):
        c = self.census()
        # Alpha_0004 is named by no scene; its family is named in one place only, so its scenes get it
        self.assertEqual(self.scene(c, "Alpha-Young")["events_by_family"], ["Alpha-0004"])
        self.assertEqual(self.scene(c, "Alpha-Old")["events_by_family"],
                         ["Alpha-0001", "Alpha-0002", "Alpha-0004"])
        self.assertEqual(self.scene(c, "Beta")["movies"], ["Beta-0001"])
        self.assertEqual(self.scene(c, "Alpha-Young")["movies_by_family"], ["Alpha-0004"])
        # a family named in two places is not spread, and what nothing names stays unplaced
        self.assertEqual(c["unplaced_events"], ["Both-0003", "Far-0001"])
        self.assertEqual(c["unplaced_movies"], ["Lone-0001"])
        s = c["summary"]
        self.assertEqual((s["events_total"], s["events_placed"], s["events_by_family"]), (11, 8, 1))
        self.assertEqual((s["movies_total"], s["movies_placed"], s["movies_by_family"]), (3, 1, 1))

    def test_tags_and_menus(self):
        c = self.census()
        tags = {s["id"]: s["tags"] for s in c["scenes"]}
        self.assertEqual(tags, {"Alpha-Old": ["home"], "Alpha-Young": ["home", "mini"], "Beta": ["boss", "fun"],
                                "Gamma": ["finale", "fun"]})
        self.assertEqual(c["menus"], [{"bytes": 5, "id": "pause", "path": "Flow/Halt.part.brn"},
                                      {"bytes": 3, "id": "map", "path": "Flow/Chart.part.brn"}])
        self.assertEqual(c["summary"]["tags"], {"boss": 1, "finale": 1, "fun": 2, "home": 2, "mini": 1})
        self.assertEqual(c["summary"]["scenes"], 4)
        self.assertEqual(c["summary"]["places"], 2)

    def test_a_name_shorter_than_the_shortest_word_is_not_searched_for(self):
        # the source file names a ten-character event; with words of eleven characters or more it is not found there
        c = self.census(rules_changes={"naming": {**RULES["naming"], "min_word": 11}})
        self.assertNotIn("Alpha-0003", self.scene(c, "Alpha-Old")["events"])
        c = self.census(rules_changes={"naming": {**RULES["naming"], "min_word": 10}})
        self.assertIn("Alpha-0003", self.scene(c, "Alpha-Old")["events"])

    def test_without_a_family_depth_no_event_is_placed_by_family(self):
        c = self.census(rules_changes={"naming": {"separator": "-", "min_word": 5}})
        self.assertEqual([s["events_by_family"] for s in c["scenes"] if s["events_by_family"]], [])
        self.assertEqual(c["summary"]["events_by_family"], 0)

    def test_the_rules_name_the_entrance_table(self):
        game = make_game(self.tmp / "game", game_files())
        rules = rulesfile.load(make_rules(self.tmp, entrance_table="Zone/Tables/Notes.byml"))
        with Source(game / "romfs" / "romfs-index.bin", game / "romfs" / "romfs.bin") as src:
            c = build(src, rules)
        self.assertEqual(self.scene(c, "Alpha-Old")["entrances"], ["x"])
        self.assertEqual(self.scene(c, "Alpha-Young")["entrances"], [])


class OutputTest(Tmp):
    def run_census(self, game: Path, out: Path, rules: Path) -> str:
        buf = io.StringIO()
        census.run(game, out, rules, say=lambda m: buf.write(m + "\n"))
        return buf.getvalue()

    def test_two_runs_write_the_same_bytes_whatever_the_image_order(self):
        rules = make_rules(self.tmp)
        files = game_files()
        a = make_game(self.tmp / "a", files, order=sorted(files))
        b = make_game(self.tmp / "b", files, order=sorted(files, reverse=True), in_place=True)
        self.run_census(a, self.tmp / "o1", rules)
        self.run_census(a, self.tmp / "o2", rules)
        self.run_census(b, self.tmp / "o3", rules)
        for name in ("census.json", "scenes.tsv"):
            one = (self.tmp / "o1" / name).read_bytes()
            self.assertEqual(one, (self.tmp / "o2" / name).read_bytes(), name)
            self.assertEqual(one, (self.tmp / "o3" / name).read_bytes(), f"{name}: another image order")
        self.assertEqual(sorted(p.name for p in (self.tmp / "o1").iterdir()), ["census.json", "scenes.tsv"])

    def test_the_output_has_no_time_path_or_offset_and_is_sorted(self):
        rules = make_rules(self.tmp)
        game = make_game(self.tmp / "game", game_files())
        self.run_census(game, self.tmp / "out", rules)
        text = (self.tmp / "out" / "census.json").read_text(encoding="ascii")
        self.assertNotIn(str(self.tmp), text)
        data = json.loads(text)
        self.assertEqual(text, json.dumps(data, indent=1, sort_keys=True) + "\n")
        self.assertEqual(data["format"], 1)
        rows = (self.tmp / "out" / "scenes.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(rows[0].split("\t"), list(census.TSV_HEAD))
        self.assertEqual([r.split("\t")[0] for r in rows[1:]], ["Alpha-Old", "Alpha-Young", "Beta", "Gamma"])

    def test_the_default_output_folder_is_inside_the_game_folder(self):
        game = make_game(self.tmp / "game", game_files())
        census.run(game, None, make_rules(self.tmp), say=lambda m: None)
        self.assertTrue((game / "census" / "census.json").is_file())

    def test_the_game_folder_is_only_read_when_the_output_is_elsewhere(self):
        game = make_game(self.tmp / "game", game_files(), in_place=True)
        before = sorted((p.relative_to(game), p.stat().st_size) for p in game.rglob("*") if p.is_file())
        self.run_census(game, self.tmp / "out", make_rules(self.tmp))
        after = sorted((p.relative_to(game), p.stat().st_size) for p in game.rglob("*") if p.is_file())
        self.assertEqual(before, after)

    def test_the_command_line(self):
        game = make_game(self.tmp / "game", game_files(), in_place=True)
        rules = make_rules(self.tmp)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = oot2.main(["census", "--game", str(game), "--rules", str(rules), "--out", str(self.tmp / "o")])
        self.assertEqual(rc, 0, err.getvalue())
        self.assertIn("4 scenes in 2 places", out.getvalue())
        self.assertTrue((self.tmp / "o" / "census.json").is_file())
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                oot2.main(["census", "--game", str(game)])        # no rules file given
        self.assertEqual(cm.exception.code, 2)


    def test_an_output_that_cannot_be_written_is_named(self):
        game = make_game(self.tmp / "game", game_files(), in_place=True)
        rules = make_rules(self.tmp)
        blocker = self.tmp / "taken"
        blocker.write_bytes(b"a file where the folder should go")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = oot2.main(["census", "--game", str(game), "--rules", str(rules), "--out", str(blocker)])
        self.assertEqual(rc, 1, err.getvalue())
        self.assertIn("cannot write the census", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


class UnreadableFileTest(Tmp):
    """A file the index lists that cannot be read stops the census, and the message names it."""

    def refuses(self, name: str, **kw) -> str:
        game = make_game(self.tmp / "game", kw.pop("files", None) or game_files(), **kw)
        rules = make_rules(self.tmp)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = oot2.main(["census", "--game", str(game), "--rules", str(rules), "--out", str(self.tmp / "out")])
        self.assertEqual(rc, 1)
        self.assertIn(name, err.getvalue())
        self.assertFalse((self.tmp / "out" / "census.json").exists(), "a census was written although a file failed")
        return err.getvalue()

    def test_a_scene_file_the_image_does_not_hold_is_named(self):
        files = game_files()
        files["Zone/Defs/Zeta.scn.pyml"] = encode_byml({"WhenEnter": True})
        order = [p for p in sorted(files) if p != "Zone/Defs/Zeta.scn.pyml"] + ["Zone/Defs/Zeta.scn.pyml"]
        msg = self.refuses("Zone/Defs/Zeta.scn.pyml", files=files, order=order, trim=8)
        self.assertIn("the index puts it at", msg)

    def test_a_scene_file_changed_in_the_image_is_named(self):
        msg = self.refuses("Zone/Defs/Beta.scn.pyml", flip_in="Zone/Defs/Beta.scn.pyml")
        self.assertIn("SHA-256", msg)

    def test_a_map_file_the_image_does_not_hold_is_named(self):
        files = game_files()
        files["Zone/Alpha/Alpha-Young-Zzz.map.byml"] = encode_byml({"a": "b"})
        order = sorted(f for f in files if f != "Zone/Alpha/Alpha-Young-Zzz.map.byml") + \
            ["Zone/Alpha/Alpha-Young-Zzz.map.byml"]
        msg = self.refuses("Zone/Alpha/Alpha-Young-Zzz.map.byml", files=files, order=order, trim=4)
        self.assertIn("the index puts it at", msg)

    def test_a_scene_file_that_is_not_a_parameter_file_is_named(self):
        files = game_files()
        files["Zone/Defs/Beta.scn.pyml"] = b"this is not a parameter file at all"
        self.refuses("Zone/Defs/Beta.scn.pyml", files=files)

    def test_a_parent_the_index_does_not_list_is_named(self):
        files = game_files()
        del files["Zone/Defs/Beta.scn.pyml"]
        msg = self.refuses("Gamma", files=files)
        self.assertIn("Beta", msg)

    def test_parents_that_name_each_other_are_named(self):
        files = game_files()
        files["Zone/Defs/Alpha-Young.scn.pyml"] = encode_byml({"Inherit": "Pool/Defs/Alpha-Old.scn.pyml"})
        self.refuses("Alpha-Young -> Alpha-Old", files=files)

    def test_a_table_the_index_lists_but_cannot_be_read_is_named(self):
        files = game_files()
        files["Zone/Tables/Broken.byml"] = b"YB\x07\x00" + b"\xff" * 40
        self.refuses("Zone/Tables/Broken.byml", files=files)

    def test_a_menu_file_the_index_does_not_list_is_named(self):
        files = game_files()
        del files["Flow/Chart.part.brn"]
        msg = self.refuses("Flow/Chart.part.brn", files=files)
        self.assertIn("map", msg)

    def test_no_scene_file_listed(self):
        files = {k: v for k, v in game_files().items() if not k.startswith("Zone/Defs/")}
        self.refuses("Zone/Defs/", files=files)

    def test_an_image_that_is_missing(self):
        game = make_game(self.tmp / "game", game_files())
        (game / "romfs" / "romfs.bin").unlink()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = oot2.main(["census", "--game", str(game), "--rules", str(make_rules(self.tmp))])
        self.assertEqual(rc, 1)
        self.assertIn("neither romfs/romfs.bin", err.getvalue())

    def test_a_game_folder_without_an_index(self):
        err = io.StringIO()
        (self.tmp / "empty").mkdir()
        with contextlib.redirect_stderr(err):
            rc = oot2.main(["census", "--game", str(self.tmp / "empty"), "--rules", str(make_rules(self.tmp))])
        self.assertEqual(rc, 1)
        self.assertIn("romfs-index.bin", err.getvalue())

    def test_a_damaged_index(self):
        game = make_game(self.tmp / "game", game_files())
        index = game / "romfs" / "romfs-index.bin"
        index.write_bytes(index.read_bytes()[:50])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = oot2.main(["census", "--game", str(game), "--rules", str(make_rules(self.tmp))])
        self.assertEqual(rc, 1)
        self.assertIn("index", err.getvalue())

    def test_a_frame_that_needs_zstd_on_a_python_without_it_is_named(self):
        files = game_files()
        files["Zone/Defs/Beta.scn.pyml"] = b"\x28\xb5\x2f\xfd" + b"\0" * 12
        game = make_game(self.tmp / "game", files)
        with mock.patch.dict(sys.modules, {"compression": None, "compression.zstd": None}):
            with Source(game / "romfs" / "romfs-index.bin", game / "romfs" / "romfs.bin") as src:
                with self.assertRaises(CensusError) as cm:
                    src.data("Zone/Defs/Beta.scn.pyml")
        self.assertIn("Zone/Defs/Beta.scn.pyml", str(cm.exception))


def _zstd():
    try:
        from compression import zstd
        return zstd
    except ImportError:
        return None


@unittest.skipIf(_zstd() is None, "no compression.zstd in this Python (3.14 and later)")
class ZstdTest(Tmp):
    def test_scene_files_stored_as_zstd_frames_are_read(self):
        z = _zstd()
        plain = self.census()
        files = {p: (z.compress(d) if p.endswith((".pyml", ".byml")) else d) for p, d in game_files().items()}
        packed = self.census(files)
        self.assertEqual(packed["scenes"], [{**s, "files": [{**f, "bytes": len(files[f["path"]])} for f in s["files"]]}
                                            for s in plain["scenes"]])


class RulesTest(Tmp):
    def bad(self, text: str, *words: str) -> None:
        with self.assertRaises(CensusError) as cm:
            rulesfile.parse(text.encode())
        for w in words:
            self.assertIn(w, str(cm.exception))

    def test_a_good_file_is_read(self):
        r = rulesfile.parse(json.dumps(RULES).encode())
        self.assertEqual((r.scene_dir, r.scene_suffix), ("Zone/Defs/", ".scn.pyml"))
        self.assertEqual(len(r.tags), 5)
        self.assertEqual(r.sha256, hashlib.sha256(json.dumps(RULES).encode()).hexdigest())

    def test_refusals(self):
        self.bad("not json", "JSON")
        self.bad("[]", "object")
        self.bad("{}", "scene_files")
        self.bad(json.dumps({**RULES, "extra": 1}), "extra")
        self.bad(json.dumps({**RULES, "scene_files": {"dir": "a/"}}), "scene_files")
        self.bad(json.dumps({**RULES, "age_keys": {"other": "x"}}), "age_keys")
        self.bad(json.dumps({**RULES, "age_tokens": {"Old": "elder"}}), "age_tokens")
        self.bad(json.dumps({**RULES, "tags": [{"tag": "x"}]}), "tags[0]", "selector")
        self.bad(json.dumps({**RULES, "tags": [{"tag": "x", "id": "a", "place": "b"}]}), "tags[0]")
        self.bad(json.dumps({**RULES, "tags": [{"tag": "x", "id": "a", "equals": 1}]}), "tags[0]")
        self.bad(json.dumps({**RULES, "tags": [{"tag": "x", "id": "("}]}), "regular expression")
        self.bad(json.dumps({**RULES, "menus": [{"id": "a", "path": "b"}, {"id": "a", "path": "c"}]}), "twice")
        self.bad(json.dumps({**RULES, "menus": [{"id": "a"}]}), "menus[0]")
        self.bad(json.dumps({**RULES, "events": {"path": "x"}}), "events")
        self.bad(json.dumps({**RULES, "event_sources": "Brain"}), "event_sources")
        self.bad(json.dumps({**RULES, "naming": {"parts": 2}}), "naming")
        self.bad(json.dumps({**RULES, "naming": []}), "naming")
        for sep in ("", "ab", "x", "7", ".", "/"):
            self.bad(json.dumps({**RULES, "naming": {"separator": sep}}), "naming.separator")
        for n in (0, 65, 1.5, "2", True):
            self.bad(json.dumps({**RULES, "naming": {"family_parts": n}}), "naming.family_parts")
            self.bad(json.dumps({**RULES, "naming": {"min_word": n}}), "naming.min_word")

    def test_without_naming_a_name_is_only_its_own_scene(self):
        r = rulesfile.parse(json.dumps({k: v for k, v in RULES.items() if k != "naming"}).encode())
        self.assertEqual((r.separator, r.family_parts, r.min_word), (None, None, 1))
        self.assertEqual((name_parts("a_b-c", None), name_parts("a_b-c", "-")), (["a_b-c"], ["a_b", "c"]))
        a = Assigner({"A", "A-B"}, None)
        self.assertEqual(a.of("A-B.x"), ("A-B", ""))
        self.assertIsNone(a.of("A-B-C.x"))
        self.assertEqual(a.family("A-C"), frozenset())

    def test_a_rules_file_that_is_missing_is_named(self):
        with self.assertRaises(CensusError) as cm:
            rulesfile.load(self.tmp / "nothing.json")
        self.assertIn("nothing.json", str(cm.exception))

    def test_a_rules_error_names_the_file(self):
        path = self.tmp / "r.json"
        path.write_text("{}", encoding="utf-8")
        with self.assertRaises(CensusError) as cm:
            rulesfile.load(path)
        self.assertIn("r.json", str(cm.exception))

    def test_an_entrance_table_that_is_not_a_table_is_refused(self):
        game = make_game(self.tmp / "game", game_files())
        rules = rulesfile.load(make_rules(self.tmp, entrance_table="Zone/Tables/readme.txt"))
        with Source(game / "romfs" / "romfs-index.bin", game / "romfs" / "romfs.bin") as src:
            with self.assertRaises(CensusError) as cm:
                build(src, rules)
        self.assertIn("Zone/Tables/readme.txt", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
