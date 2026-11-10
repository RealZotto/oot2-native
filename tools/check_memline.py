#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Check the memory line of a run (runtime/nvn/nvn_memline.c; the format is in oot2/memline.h).

Reads memory-<stamp>-<pid>.tsv and, for each chosen row, adds the named terms (the columns
of the file's "sum" line: the committed bytes of the five guest regions, the private host
bytes outside them in three groups, and the VRAM terms VirtualQuery cannot see) and
compares the total with the process's private commit. A row passes within 3 percent.

The rows judged: title (the row written at the title mark), scene (at the scene entry) and
plateau (the last minute row before the scene row, else before the end). A row the file
does not hold is not judged unless --require names it.

On a failure the check names what it can: a term whose column is absent from the file, a
term that is 0 or -1 in the failing row yet positive in another row (the likely dropped
term, with the size it has elsewhere), and the residual itself. It also prints, for
information, the Vulkan heap 0 usage that no VRAM term explains.

Exit status: 0 every judged row passes, 1 a row fails or a required row is missing, 2 the
file is not a memory line.

Usage: check_memline.py memory.tsv [--tolerance 0.03] [--require title,plateau,scene]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GIB = 1 << 30
UNREAD = -1


class FormatError(Exception):
    pass


def parse(path: Path) -> tuple[list[str], list[tuple[str, dict[str, int]]], int | None]:
    """The sum line's column names, the rows as (kind, {column: value}) and the closing line's count."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "oot2-memline 1":
        raise FormatError(f"{path}: not an oot2-memline 1 file")
    columns: list[str] | None = None
    summed: list[str] | None = None
    rows: list[tuple[str, dict[str, int]]] = []
    end: int | None = None
    for n, line in enumerate(lines[1:], 2):
        f = line.split("\t")
        if f[0] == "columns":
            columns = f[1:]
        elif f[0] == "sum":
            summed = f[1:]
        elif f[0] == "row":
            if columns is None:
                raise FormatError(f"line {n}: a row before the columns line")
            if len(f) - 2 != len(columns):
                raise FormatError(f"line {n}: {len(f) - 2} values for {len(columns)} columns")
            try:
                rows.append((f[1], dict(zip(columns, (int(v) for v in f[2:])))))
            except ValueError as e:
                raise FormatError(f"line {n}: {e}") from None
        elif f[0] == "end":
            end = int(f[1])
    if columns is None or summed is None:
        raise FormatError(f"{path}: no columns line or no sum line")
    missing = [c for c in summed if c not in columns]
    if missing:
        raise FormatError(f"the sum line names columns the file does not have: {', '.join(missing)}")
    return summed, rows, end


def chosen(rows: list[tuple[str, dict[str, int]]]) -> dict[str, tuple[int, dict[str, int]] | None]:
    """The rows to judge, by name, with their index in the file."""
    def last(kind: str, before: int) -> tuple[int, dict[str, int]] | None:
        for i in range(min(before, len(rows)) - 1, -1, -1):
            if rows[i][0] == kind:
                return i, rows[i][1]
        return None

    first = {k: next((i for i, (kind, _) in enumerate(rows) if kind == k), None) for k in ("title", "scene", "end")}
    out: dict[str, tuple[int, dict[str, int]] | None] = {}
    out["title"] = None if first["title"] is None else (first["title"], rows[first["title"]][1])
    out["scene"] = None if first["scene"] is None else (first["scene"], rows[first["scene"]][1])
    bound = first["scene"] if first["scene"] is not None else first["end"] if first["end"] is not None else len(rows)
    out["plateau"] = last("minute", bound)
    return out


def total(row: dict[str, int], summed: list[str]) -> int:
    return sum(max(row[c], 0) for c in summed)


def gib(v: int) -> str:
    return f"{v / GIB:.2f} GiB"


def judge(name: str, index: int, row: dict[str, int], rows: list[tuple[str, dict[str, int]]], summed: list[str],
          tolerance: float) -> list[str]:
    """The problems of one row; empty when it passes."""
    priv = row["private"]
    if priv <= 0:
        return [f"{name}: the row has no private commit ({priv})"]
    got = total(row, summed)
    residual = priv - got
    share = abs(residual) / priv
    explained = row.get("bk_block_bytes", 0) + row.get("tex_arena_bytes", 0) + row.get("img_own_bytes", 0)
    heap0 = row.get("vk_heap0_used", 0)
    info = f"{name}: named terms {gib(got)} against private {gib(priv)}, residual {gib(residual)} ({share:.1%})"
    if heap0 > 0:
        info += f"; Vulkan heap 0 uses {gib(heap0)}, {gib(max(heap0 - explained, 0))} of it in no VRAM term"
    print(info)
    if share <= tolerance:
        return []
    problems = [f"{name}: the terms sum to {gib(got)} but private commit is {gib(priv)}: residual {gib(residual)} "
                f"({share:.1%}, over {tolerance:.0%})"]
    for c in summed:
        elsewhere = max((r[c] for i, (_, r) in enumerate(rows) if i != index and r.get(c, 0) > 0), default=0)
        if row[c] <= 0 and elsewhere > 0:
            problems.append(f"{name}: term {c} is {row[c]} here and up to {gib(elsewhere)} in other rows of this file: "
                            f"it is missing from this row")
    if len(problems) == 1:
        problems.append(f"{name}: no term is missing as such; the residual {gib(residual)} is memory no named term counts")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--require", default="", help="comma separated rows that must exist: title, plateau, scene")
    args = ap.parse_args(argv)
    try:
        summed, rows, end = parse(args.file)
    except (FormatError, OSError) as e:
        print(f"check_memline: {e}", file=sys.stderr)
        return 2
    if end is None:
        print("check_memline: the file has no closing line (the run was cut short)", file=sys.stderr)
    picks = chosen(rows)
    required = {r for r in args.require.split(",") if r}
    unknown = required - set(picks)
    if unknown:
        print(f"check_memline: unknown row names {sorted(unknown)}", file=sys.stderr)
        return 2
    problems: list[str] = []
    judged = 0
    for name in ("title", "plateau", "scene"):
        pick = picks[name]
        if pick is None:
            if name in required:
                problems.append(f"{name}: the file has no such row")
            else:
                print(f"{name}: no such row in this file (not judged)")
            continue
        judged += 1
        problems += judge(name, pick[0], pick[1], rows, summed, args.tolerance)
    if not judged and not problems:
        problems.append("no row to judge")
    for p in problems:
        print(f"FAIL {p}")
    if not problems:
        print(f"PASS {judged} row(s) within {args.tolerance:.0%}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
