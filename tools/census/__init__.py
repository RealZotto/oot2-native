# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""`oot2 census`: the list of the game's scenes, decoded from the player's own RomFS (build.py says how).

run() reads the prepared folder's RomFS index and its image, applies the player's rules file (rules.py), and writes two
files into the output folder, byte for byte the same on every run over the same inputs:

    census.json   the whole census: scenes (id, place, ages, entrances, ways in and out, events, movies, tags, files),
                  places, menus, the events and movies no scene names, and a summary
    scenes.tsv    one line per scene for a spreadsheet

The census is the game's data. It goes into the game folder (the default) or a folder of the player's choosing, and
never into a repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import rules as rulesfile
from .build import build
from .source import CensusError, open_game

TSV_HEAD = ("id", "place", "ages", "entrances", "links_out", "entered_from", "events", "movies", "tags", "files")


def render_json(census: dict) -> bytes:
    return (json.dumps(census, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def render_tsv(census: dict) -> bytes:
    lines = ["\t".join(TSV_HEAD)]
    for s in census["scenes"]:
        lines.append("\t".join((s["id"], s["place"], ",".join(s["ages"]), str(len(s["entrances"])),
                                str(len(s["links_out"])), str(len(s["entered_from"])), str(len(s["events"])),
                                str(len(s["movies"])), ",".join(s["tags"]), str(len(s["files"])))))
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run(game: Path, out: Path | None, rules_path: Path, image: Path | None = None, say=print) -> dict:
    """Make the census of a prepared folder and write it into out (default: <game>/census). Returns the census."""
    game = Path(game)
    rules = rulesfile.load(rules_path)
    out = Path(out) if out is not None else game / "census"
    with open_game(game, image) as src:
        census = build(src, rules)
    try:
        out.mkdir(parents=True, exist_ok=True)
        write_atomic(out / "census.json", render_json(census))
        write_atomic(out / "scenes.tsv", render_tsv(census))
    except OSError as e:
        raise CensusError(f"cannot write the census into {out}: {e.strerror or e}") from None
    summary = census["summary"]
    say(f"oot2 census: {summary['scenes']} scenes in {summary['places']} places, {summary['entrances']} entrances, "
        f"{summary['events_placed']} (+{summary['events_by_family']} by family) of {summary['events_total']} events "
        f"and {summary['movies_placed']} (+{summary['movies_by_family']}) of {summary['movies_total']} movies "
        f"placed, {summary['menus']} menus; written to {out}")
    return census


__all__ = ["CensusError", "run"]
