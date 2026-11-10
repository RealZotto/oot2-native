# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The rules file of a census: the few places where the census has to be told a name.

Nothing from the game is written in this repository, so the names that tell the census where the scene files are, which
keys of a scene file say its age, and which scenes carry which tag are given by the player in a JSON file of their own,
kept outside any repository (the census output is the game's data in the same way). The file is read as it is, with
every field checked; a field the census does not know is refused by name.

    {
     "scene_files":    {"dir": "<index directory>/", "suffix": "<file name after the scene id>"},      required
     "age_keys":       {"force_switch": "<key>", "child_on_change": "<key>", "child_on_init": "<key>"},
     "age_tokens":     {"<word in a scene or folder name>": "child" | "adult"},
     "events":         {"dir": "<index directory>/"},
     "event_sources":  ["<index directory>/", ...],
     "movies":         {"dir": "<index directory>/"},
     "entrance_table": "<index path of the per-scene table that holds the entrances>",
     "tags":           [{"tag": "<tag>", <one selector>}, ...],
     "menus":          [{"id": "<row id>", "path": "<index path>"}, ...],
     "naming":         {"separator": "<one character>", "family_parts": <n>, "min_word": <n>}
    }

A tag selector is one of "id" (a regular expression the whole scene id matches), "place" (the same for its place),
"event" (a regular expression found in the name of one of the scene's events) or "key" (a key of the scene's resolved
parameters, with "equals" its value, true when omitted).

"naming" says how the player's file names are built, which is a fact of their data and not of this code. "separator" is
the character that splits a name into parts: with it, a file or folder belongs to the scene whose id is its longest
leading run of parts, and names that share leading parts form families; without it, a name belongs to a scene only when
its stem is the scene's id, and there are no families. "family_parts" is how many leading parts name an event's family
(an event needs at least one part more to be in one); without it, events are not grouped. "min_word" is the shortest run
of name characters taken as a word when a file is searched for event names, and the shortest folder name matched to a
scene's string; it is 1 when omitted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .source import CensusError

AGES = ("child", "adult")
AGE_KEYS = ("force_switch", "child_on_change", "child_on_init")
SELECTORS = ("id", "place", "event", "key")
TOP = {"scene_files", "age_keys", "age_tokens", "events", "event_sources", "movies", "entrance_table", "tags", "menus",
       "naming"}
NAMING = {"separator", "family_parts", "min_word"}


@dataclass
class Tag:
    tag: str
    selector: str
    pattern: object = None          # compiled regular expression, or the key name
    equals: object = True


@dataclass
class Rules:
    scene_dir: str
    scene_suffix: str
    age_keys: dict = field(default_factory=dict)
    age_tokens: dict = field(default_factory=dict)
    events_dir: str | None = None
    event_sources: list = field(default_factory=list)
    movies_dir: str | None = None
    entrance_table: str | None = None
    tags: list = field(default_factory=list)
    menus: list = field(default_factory=list)
    separator: str | None = None
    family_parts: int | None = None
    min_word: int = 1
    sha256: str = ""


def _fail(why: str):
    raise CensusError(f"the rules file {why}")


def _text(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"needs {what} to be a non-empty string")
    return value


def _dir(value, what: str) -> str:
    text = _text(value, what)
    return text if text.endswith("/") else text + "/"


def _count(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 64:
        _fail(f"needs {what} to be a whole number from 1 to 64")
    return value


def parse(data: bytes) -> Rules:
    try:
        raw = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        _fail(f"is not valid UTF-8 JSON ({e})")
    if not isinstance(raw, dict):
        _fail("must hold one JSON object")
    unknown = sorted(set(raw) - TOP)
    if unknown:
        _fail(f"has a field the census does not know: {', '.join(unknown)}")
    sf = raw.get("scene_files")
    if not isinstance(sf, dict) or set(sf) != {"dir", "suffix"}:
        _fail('needs "scene_files" with exactly "dir" and "suffix"')
    rules = Rules(_dir(sf["dir"], '"scene_files.dir"'), _text(sf["suffix"], '"scene_files.suffix"'))
    rules.sha256 = hashlib.sha256(data).hexdigest()
    keys = raw.get("age_keys", {})
    if not isinstance(keys, dict) or set(keys) - set(AGE_KEYS):
        _fail(f'has "age_keys" that is not an object of {", ".join(AGE_KEYS)}')
    rules.age_keys = {k: _text(v, f'"age_keys.{k}"') for k, v in keys.items()}
    tokens = raw.get("age_tokens", {})
    if not isinstance(tokens, dict) or any(v not in AGES for v in tokens.values()):
        _fail('has "age_tokens" that is not an object of words mapped to "child" or "adult"')
    rules.age_tokens = {_text(k, "an age token"): v for k, v in tokens.items()}
    for name in ("events", "movies"):
        if name in raw:
            sec = raw[name]
            if not isinstance(sec, dict) or set(sec) != {"dir"}:
                _fail(f'needs "{name}" to be an object with "dir"')
            setattr(rules, name + "_dir", _dir(sec["dir"], f'"{name}.dir"'))
    srcs = raw.get("event_sources", [])
    if not isinstance(srcs, list):
        _fail('needs "event_sources" to be a list of index directories')
    rules.event_sources = [_dir(d, '"event_sources" entry') for d in srcs]
    if "entrance_table" in raw:
        rules.entrance_table = _text(raw["entrance_table"], '"entrance_table"')
    for n, item in enumerate(raw.get("tags", [])):
        if not isinstance(item, dict) or "tag" not in item:
            _fail(f'has tags[{n}] without a "tag"')
        chosen = [s for s in SELECTORS if s in item]
        extra = set(item) - {"tag", "equals"} - set(SELECTORS)
        if len(chosen) != 1 or extra or ("equals" in item and chosen != ["key"]):
            _fail(f"has tags[{n}] that does not have exactly one selector of {', '.join(SELECTORS)}")
        sel = chosen[0]
        pat = _text(item[sel], f'"tags[{n}].{sel}"')
        if sel != "key":
            try:
                pat = re.compile(pat)
            except re.error as e:
                _fail(f"has a bad regular expression in tags[{n}]: {e}")
        rules.tags.append(Tag(_text(item["tag"], f'"tags[{n}].tag"'), sel, pat, item.get("equals", True)))
    seen = set()
    for n, item in enumerate(raw.get("menus", [])):
        if not isinstance(item, dict) or set(item) != {"id", "path"}:
            _fail(f'needs menus[{n}] to be an object with "id" and "path"')
        mid = _text(item["id"], f'"menus[{n}].id"')
        if mid in seen:
            _fail(f"names the menu {mid} twice")
        seen.add(mid)
        rules.menus.append((mid, _text(item["path"], f'"menus[{n}].path"')))
    naming = raw.get("naming", {})
    if not isinstance(naming, dict) or set(naming) - NAMING:
        _fail(f'has "naming" that is not an object of {", ".join(sorted(NAMING))}')
    if "separator" in naming:
        sep = _text(naming["separator"], '"naming.separator"')
        if len(sep) != 1 or sep.isalnum() or sep in "./":
            _fail('needs "naming.separator" to be one character that is not a letter, a digit, "." or "/"')
        rules.separator = sep
    if "family_parts" in naming:
        rules.family_parts = _count(naming["family_parts"], '"naming.family_parts"')
    if "min_word" in naming:
        rules.min_word = _count(naming["min_word"], '"naming.min_word"')
    return rules


def load(path: Path) -> Rules:
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise CensusError(f"the rules file {path} cannot be read: {e.strerror or e}") from None
    try:
        return parse(data)
    except CensusError as e:
        raise CensusError(f"{path}: {e}") from None
