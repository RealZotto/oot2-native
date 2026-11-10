# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The scene census: which scenes the game has, how each is entered, and what happens in it, decoded from its data.

The unit is the scene table: one parameter file per scene, in one folder of the RomFS (the rules file names the folder
and the file-name ending, rules.py). The scene's id is its file name without that ending. Everything else is found by
following the data from there, with no further name written down here:

- A parameter file may name another scene's file as its parent; the key that does is the one whose string value is
  another scene's id, found by looking, and the scene's resolved parameters are the parent's with its own over them.
- The scene's place is the folder its map files sit in, read from the path-like strings of its parameters that end
  with its own id.
- A file or folder belongs to the scene whose id is the longest leading part of its name that ends at an underscore
  ("<id>_Part.ext", "<id>"). The map files of a scene are the ones of that kind in its place; folders elsewhere in the
  RomFS named that way (a folder per scene in another tree) add the scene's variants.
- Any string of a map file that is another scene's id is a way into that scene (links_out, and entered_from the other
  way). Any string that is the name of an event file (the rules name the events folder) is an event of the scene, and a
  movie file named like an event belongs to that event.
- The per-scene tables are the other data files of the scene table's top folder whose top-level keys are scene ids; the
  entrances are the second-level keys of the one that holds poses (or the one the rules name).

What needs the player's rules is only what cannot be seen from structure: the scene table's folder, the keys that give
an age, the words in names that give an age, the events and movies folders, tags, and the menus' files.

A file the index lists that cannot be read stops the census with its name; nothing is skipped.
"""

from __future__ import annotations

import collections
import re

from . import byml
from .rules import Rules
from .source import CensusError, Source

FORMAT = 1
MAX_TABLE_BYTES = 16 << 20       # a data file larger than this in the scene table's top folder is not a table
MAX_SOURCE_BYTES = 8 << 20       # nor is one this large searched for event names
NAME_CHARS = "A-Za-z0-9_.:/-"   # the characters of a name searched for in a file


def word_pattern(rules: Rules) -> re.Pattern:
    """A run of name characters (with the rules' separator) at least the rules' shortest word long."""
    extra = re.escape(rules.separator) if rules.separator and rules.separator not in "_.:/-" else ""
    return re.compile(f"[{NAME_CHARS}{extra}]{{{rules.min_word},}}".encode("ascii"))


def prefixes(stem: str, sep: str | None):
    """stem, then stem without its last part, and so on; only stem when the rules give no separator."""
    yield stem
    while sep and sep in stem:
        stem = stem.rsplit(sep, 1)[0]
        yield stem


def name_parts(name: str, sep: str | None) -> list[str]:
    """The parts of a name, split at the rules' separator; the whole name when the rules give none."""
    return name.split(sep) if sep else [name]


def stem_of(name: str) -> str:
    return name.partition(".")[0]


def walk_strings(value, out: set) -> None:
    """Every string value of a parsed file, keys not included."""
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, str):
            out.add(v)
        elif isinstance(v, dict):
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)


def load_byml(src: Source, path: str):
    data = src.data(path)
    try:
        return byml.load(data)
    except byml.BymlError as e:
        raise CensusError(f"{path} cannot be read as a parameter file: {e}") from None


def scene_table(src: Source, rules: Rules) -> dict[str, str]:
    """{scene id: index path} of the scene table."""
    found = {}
    for path in src.paths:
        if path.startswith(rules.scene_dir) and path.endswith(rules.scene_suffix) and \
                "/" not in path[len(rules.scene_dir):]:
            sid = path[len(rules.scene_dir):-len(rules.scene_suffix)]
            if sid:
                found[sid] = path
    if not found:
        raise CensusError(f"the index lists no scene file under {rules.scene_dir} ending in {rules.scene_suffix}")
    return found


def find_parent_key(raw: dict, ids: set) -> str | None:
    """The top-level key whose string value names another scene's file (by stem): how a scene file inherits."""
    count = collections.Counter()
    for sid, params in raw.items():
        for key, val in params.items():
            target = stem_of(val.rsplit("/", 1)[-1]) if isinstance(val, str) else None
            if target in ids and target != sid:
                count[key] += 1
    if not count:
        return None
    top = max(count.values())
    return sorted(k for k, n in count.items() if n == top)[0]


def resolve(raw: dict, parent_key: str | None, ids: set) -> dict:
    out: dict[str, dict] = {}

    def one(sid: str, chain: tuple) -> dict:
        if sid in out:
            return out[sid]
        if sid in chain:
            raise CensusError(f"the scene files of {' -> '.join(chain + (sid,))} name each other as parents")
        own = dict(raw[sid])
        merged: dict = {}
        if parent_key and isinstance(own.get(parent_key), str):
            pid = stem_of(own[parent_key].rsplit("/", 1)[-1])
            if pid not in ids:
                raise CensusError(f"the scene file of {sid} names the parent {pid}, which the index does not list")
            merged.update(one(pid, chain + (sid,)))
            own.pop(parent_key)
        merged.update(own)
        out[sid] = merged
        return merged

    for sid in sorted(raw):
        one(sid, ())
    return out


class Assigner:
    """Which scene a file or folder name belongs to: the longest id that is a leading run of parts of its stem.

    A name that starts with a part shared by several ids (the word the child and adult scenes of a place begin with)
    and with no id of its own belongs to all of them: family()."""

    def __init__(self, ids: set, sep: str | None):
        self.ids, self.sep = ids, sep
        shared: dict[str, set] = collections.defaultdict(set)
        for sid in ids:
            for pre in list(prefixes(sid, sep))[1:]:
                shared[pre].add(sid)
        self.shared = {pre: frozenset(v) for pre, v in shared.items() if len(v) >= 2}

    def of(self, name: str) -> tuple[str, str] | None:
        """(scene id, the rest of the stem after it) or None."""
        stem = stem_of(name)
        for pre in prefixes(stem, self.sep):
            if pre in self.ids:
                return pre, stem[len(pre):].lstrip(self.sep or "")
        return None

    def family(self, name: str) -> frozenset:
        """The scenes a name belongs to: its own id, or all those that share its leading part, or none."""
        hit = self.of(name)
        if hit:
            return frozenset((hit[0],))
        for pre in prefixes(stem_of(name), self.sep):
            if pre in self.shared:
                return self.shared[pre]
        return frozenset()


def pose_like(entry) -> bool:
    return isinstance(entry, dict) and any(
        isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v) for v in entry.values())


def build(src: Source, rules: Rules) -> dict:
    def split_name(name: str) -> list[str]:
        return name_parts(name, rules.separator)

    table = scene_table(src, rules)
    ids = set(table)
    raw = {}
    for sid, path in sorted(table.items()):
        params = load_byml(src, path)
        if not isinstance(params, dict):
            raise CensusError(f"{path} does not hold a table of parameters")
        raw[sid] = params
    parent_key = find_parent_key(raw, ids)
    resolved = resolve(raw, parent_key, ids)
    assign = Assigner(ids, rules.separator)
    word = word_pattern(rules)

    # Folders of the RomFS two levels down ("<tree>/<name>/"): the files directly inside, and where each name occurs.
    dir2_files: dict[str, list[str]] = collections.defaultdict(list)       # "tree/name" -> files directly inside
    dir2_trees: dict[str, set] = collections.defaultdict(set)              # name -> trees it is in
    for p in src.paths:
        parts = p.split("/")
        if len(parts) >= 3:
            dir2_trees[parts[1]].add(parts[0])
            if len(parts) == 3:
                dir2_files[f"{parts[0]}/{parts[1]}"].append(p)

    def owned(directory: str, sid: str) -> list[str]:
        files = []
        for p in dir2_files.get(directory, []):
            hit = assign.of(p.rsplit("/", 1)[1])
            if hit and hit[0] == sid:
                files.append(p)
        return sorted(files)

    # Places and map files. A scene's path-like strings that end in its own id name folders; the place is the one of
    # them that holds the most of the scene's files (the map files), the strings that name it breaking a tie.
    place_of: dict[str, str] = {}
    map_files: dict[str, list[str]] = {}
    for sid, params in resolved.items():
        votes: collections.Counter = collections.Counter()
        strings: set = set()
        walk_strings(params, strings)
        for s in strings:
            if "/" in s:
                head, _, base = s.rpartition("/")
                if stem_of(base) == sid:
                    votes[head.rpartition("/")[2]] += 1
        best = None
        for comp in sorted(votes):
            for tree in sorted(dir2_trees.get(comp, ())):
                files = owned(f"{tree}/{comp}", sid)
                key = (-len(files), -votes[comp], comp, tree)
                if files and (best is None or key < best[0]):
                    best = (key, comp, files)
        place_of[sid] = best[1] if best else sid
        map_files[sid] = best[2] if best else []

    # Folders elsewhere that belong to a scene; the part of their name after the id is a variant.
    scene_dirs: dict[str, set] = collections.defaultdict(set)
    variants: dict[str, set] = collections.defaultdict(set)
    for name, trees in dir2_trees.items():
        hit = assign.of(name)
        if hit and name == stem_of(name):
            for tree in trees:
                scene_dirs[hit[0]].add(f"{tree}/{name}")
            if hit[1]:
                variants[hit[0]].add(hit[1])
    for sid, params in resolved.items():
        strings = set()
        walk_strings(params, strings)
        for s in strings:
            if len(s) >= rules.min_word and "/" not in s and s in dir2_trees and s not in ids:
                for tree in dir2_trees[s]:
                    scene_dirs[sid].add(f"{tree}/{s}")

    # Events and movies.
    events: dict[str, list[str]] = collections.defaultdict(list)
    if rules.events_dir:
        for p in src.paths:
            if p.startswith(rules.events_dir):
                events[stem_of(p.rsplit("/", 1)[1])].append(p)
    movies: dict[str, list[str]] = collections.defaultdict(list)
    if rules.movies_dir:
        for p in src.paths:
            if p.startswith(rules.movies_dir):
                movies[stem_of(p.rsplit("/", 1)[1])].append(p)

    # Events named in the files of other trees that belong to a scene by name (the rules say which trees).
    source_events: dict[str, set] = collections.defaultdict(set)
    searched = 0
    if events:
        for p in src.paths:
            if not any(p.startswith(d) for d in rules.event_sources) or src.size(p) > MAX_SOURCE_BYTES:
                continue
            owners = assign.family(p.rsplit("/", 1)[1])
            if not owners:
                continue
            searched += 1
            found = {t.decode("ascii") for t in word.findall(src.data(p))} & events.keys()
            for sid in owners:
                source_events[sid] |= found

    # Map files: events, ways into other scenes.
    scene_events: dict[str, set] = {}
    scene_links: dict[str, set] = {}
    map_strings: dict[str, set] = {}
    for sid in sorted(ids):
        strings: set = set()
        for p in map_files[sid]:
            value = load_byml(src, p)
            walk_strings(value, strings)
        map_strings[sid] = strings
        scene_events[sid] = {s for s in strings if s in events} | source_events[sid]
        scene_links[sid] = {s for s in strings if s in ids and s != sid}
    own_strings: dict[str, set] = {}
    for sid, params in resolved.items():
        strings = set()
        walk_strings(params, strings)
        own_strings[sid] = {s for s in strings if s in ids and s != sid}
    entered: dict[str, set] = collections.defaultdict(set)
    for sid in ids:
        for target in scene_links[sid] | own_strings[sid]:
            entered[target].add(sid)

    # Events no scene file names, by family: the events whose names share their leading parts (the rules say how
    # many). A family whose named members all fall in one place gives its other members to the scenes that name it (an
    # inference from naming, kept apart from the events a scene names itself).
    def family_of(event: str) -> str | None:
        if not rules.separator or not rules.family_parts:
            return None
        parts = event.split(rules.separator)
        n = rules.family_parts
        return rules.separator.join(parts[:n]) if len(parts) > n else None

    named: dict[str, set] = collections.defaultdict(set)         # family -> scenes that name a member
    for sid in ids:
        for e in scene_events[sid]:
            if family_of(e):
                named[family_of(e)].add(sid)
    scene_family: dict[str, set] = {sid: set() for sid in ids}
    for fam, owners in named.items():
        if len({place_of[o] for o in owners}) != 1:
            continue
        members = {e for e in events if family_of(e) == fam}
        for sid in owners:
            scene_family[sid] |= members - scene_events[sid]

    # Per-scene tables.
    tables: dict[str, dict] = {}
    top = rules.scene_dir.split("/")[0] + "/"
    for p in src.paths:
        if not p.startswith(top) or p.startswith(rules.scene_dir) or src.size(p) > MAX_TABLE_BYTES:
            continue
        if assign.of(p.rsplit("/", 1)[1]):
            continue
        data = src.data(p)
        if not byml.is_byml(data):
            continue
        try:
            value = byml.load(data)
        except byml.BymlError as e:
            raise CensusError(f"{p} cannot be read as a parameter file: {e}") from None
        if isinstance(value, dict) and value and 2 * len(set(value) & ids) >= len(value):
            tables[p] = value
    chosen = None
    if rules.entrance_table:
        if rules.entrance_table not in tables:
            raise CensusError(f"the entrance table {rules.entrance_table} is not a per-scene table of the scene "
                              "table's folder")
        chosen = rules.entrance_table
    elif tables:
        def poses(p):
            return sum(1 for v in tables[p].values() if isinstance(v, dict) for e in v.values() if pose_like(e))
        best = sorted(tables, key=lambda p: (-poses(p), p))
        chosen = best[0] if poses(best[0]) else None

    # Menus.
    menus = []
    for mid, path in rules.menus:
        if not src.has(path):
            raise CensusError(f"the menu {mid} is at {path}, which the index does not list")
        menus.append({"bytes": src.size(path), "id": mid, "path": path})

    scenes = []
    movie_used: set = set()
    event_used: set = set()
    family_event_used: set = set()
    family_movie_used: set = set()
    for sid in sorted(ids):
        params = resolved[sid]
        flags = {}
        for name, key in rules.age_keys.items():
            if isinstance(params.get(key), bool):
                flags[name] = params[key]
        words = set(split_name(sid)) | {w for v in variants[sid] for w in split_name(v)}
        ages, why = set(), set()
        for w in words:
            if w in rules.age_tokens:
                ages.add(rules.age_tokens[w])
                why.add("name" if w in split_name(sid) else "folders")
        for name in ("child_on_change", "child_on_init"):
            if name in flags:
                ages.add("child" if flags[name] else "adult")
                why.add("flag")
        if not any(n in flags for n in ("child_on_change", "child_on_init")) and flags.get("force_switch"):
            ages.add("child")
            why.add("forced, age not given")
        evs = sorted(scene_events[sid])
        fevs = sorted(scene_family[sid])
        mvs = sorted(m for m in movies if m in map_strings[sid] or m in scene_events[sid])
        fmvs = sorted(m for m in movies if m in scene_family[sid] and m not in mvs)
        event_used.update(evs)
        movie_used.update(mvs)
        family_event_used.update(fevs)
        family_movie_used.update(fmvs)
        entrances = []
        if chosen and isinstance(tables[chosen].get(sid), dict):
            entrances = sorted(tables[chosen][sid])
        tags = set()
        for t in rules.tags:
            if t.selector == "id":
                ok = t.pattern.fullmatch(sid)
            elif t.selector == "place":
                ok = t.pattern.fullmatch(place_of[sid])
            elif t.selector == "event":
                ok = any(t.pattern.search(e) for e in evs + fevs)
            else:
                ok = t.pattern in params and params[t.pattern] == t.equals
            if ok:
                tags.add(t.tag)
        scenes.append({
            "age_flags": dict(sorted(flags.items())),
            "age_from": sorted(why),
            "ages": sorted(ages),
            "dirs": sorted(scene_dirs[sid]),
            "entered_from": sorted(entered[sid]),
            "entrances": entrances,
            "events": evs,
            "events_by_family": fevs,
            "files": [{"bytes": src.size(p), "path": p} for p in map_files[sid]],
            "id": sid,
            "links_out": sorted(scene_links[sid] | own_strings[sid]),
            "movies": mvs,
            "movies_by_family": fmvs,
            "parameter_file": table[sid],
            "place": place_of[sid],
            "tables": sorted(p.rsplit("/", 1)[1] for p in tables if sid in tables[p]),
            "tags": sorted(tags),
        })

    places: dict[str, list] = collections.defaultdict(list)
    for s in scenes:
        places[s["place"]].append(s["id"])
    tag_count = collections.Counter(t for s in scenes for t in s["tags"])
    age_count = collections.Counter("+".join(s["ages"]) or "unset" for s in scenes)
    census = {
        "format": FORMAT,
        "inputs": {"index_files": src.header["count"], "rules_sha256": rules.sha256,
                   "scene_files": rules.scene_dir + "*" + rules.scene_suffix},
        "menus": menus,
        "places": [{"place": p, "scenes": sorted(v)} for p, v in sorted(places.items())],
        "scenes": scenes,
        "summary": {
            "ages": dict(sorted(age_count.items())),
            "entrance_table": chosen,
            "event_files_searched": searched,
            "entrances": sum(len(s["entrances"]) for s in scenes),
            "events_by_family": len(family_event_used - event_used),
            "events_placed": len(event_used),
            "events_total": len(events),
            "menus": len(menus),
            "movies_by_family": len(family_movie_used - movie_used),
            "movies_placed": len(movie_used),
            "movies_total": len(movies),
            "parent_key": parent_key,
            "places": len(places),
            "scene_without_map_files": sum(1 for s in scenes if not s["files"]),
            "scenes": len(scenes),
            "scenes_with_entrances": sum(1 for s in scenes if s["entrances"]),
            "tables": sorted(tables),
            "tags": dict(sorted(tag_count.items())),
        },
        "unplaced_events": sorted(set(events) - event_used - family_event_used),
        "unplaced_movies": sorted(set(movies) - movie_used - family_movie_used),
    }
    return census
