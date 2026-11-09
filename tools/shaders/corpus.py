#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Run the shader corpus through the translator: translate, validate, time, compare answers.

    py -3 tools/shaders/corpus.py run --split <split dir> --xlate <oot2-xlate> --work <dir>
                                      [--pass NAME ...] [--workers 12] [--stages vs,ps] [--limit N]
    py -3 tools/shaders/corpus.py report --work <dir>

The split folder is what tools/shaders/split.py writes (jobs.jsonl, programs.tsv).
Each pass translates a set of jobs on a pool of translator children
(tools/shaders/pool.py) into its own folder, validates every module with
spirv-val (tools/shaders/validate.py) and writes a summary of counts.

Passes:
  default  every job, default answers, capability post-pass on; the translator's
           default optimizer form, which is the sound form (debug false)
  raw      every job, capability post-pass off: the validator's verdict on the
           translator's own output
  repeat   a fixed sample translated again with default answers; every module
           must come out byte-identical to the default pass
  tex      texture and image answers changed for every slot (3d), all stages
  attr     vertex attribute answers changed for every location (sint), vertex
           programs only
  fout     fragment output answers changed for every target (uint), pixel
           programs only
  nodebug  every job, default answers, the AST optimizer's debug mode off
           (expression propagation on), asked for explicitly: the same form as
           `default` since the sound form became the translator's default, kept
           so that a record made before that can be repeated by name
  debug    every job, default answers, the old optimizer form asked for with
           the job option debug true (nothing propagated or dropped), for A/B
           work: tools/shaders/opcheck.py compares `default` with it

The answer passes measure answer sensitivity (architecture D21): a program
whose module changes when an answer changes needs that answer in its cache key
and a variant when the game's draw differs from the default.

Everything is written under --work, which must lie outside the repository:
results name the game's programs and the modules are derived from the game.
Summaries hold counts and normalised messages only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pool  # noqa: E402
import validate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
STAGES = ("vs", "ps", "cs", "ts", "ms")
PASSES = ("default", "raw", "repeat", "tex", "attr", "fout", "nodebug", "debug")
REPEAT_SAMPLE = 2000

# The answers each sensitivity pass sets, and the stages it is run on.
VARIANTS: dict[str, tuple[dict, tuple[str, ...]]] = {
    "default": ({}, STAGES),
    "raw": ({"options": {"capability_fixup": False}}, STAGES),
    "repeat": ({}, STAGES),
    "tex": ({"tex": {"*": "3d"}, "img": {"*": {"type": "3d", "fmt": "R32Float"}}}, STAGES),
    "attr": ({"attr": {"*": "sint"}}, ("vs",)),
    "fout": ({"fout": {"*": "uint"}}, ("ps",)),
    "nodebug": ({"options": {"debug": False}}, STAGES),
    "debug": ({"options": {"debug": True}}, STAGES),
}

RESULT_FIELDS = ("key", "status", "micros", "wall_ms", "spv_bytes", "message")


# --- inputs -------------------------------------------------------------------

def load_jobs(split: Path) -> dict[str, dict]:
    """Job objects by key, from jobs.jsonl."""
    jobs: dict[str, dict] = {}
    with open(split / "jobs.jsonl", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            job = json.loads(line)
            key = job.get("key")
            if not isinstance(key, str) or key in jobs:
                raise ValueError(f"jobs.jsonl line {n}: missing or repeated key")
            jobs[key] = job
    return jobs


def load_meta(split: Path) -> dict[str, dict[str, str]]:
    """programs.tsv rows by job key (the first column), or {} when absent."""
    path = split / "programs.tsv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader((l for l in fh if not l.startswith("#")), delimiter="\t"))
    if not rows:
        return {}
    first = next(iter(rows[0]))
    return {row[first]: row for row in rows}


def variant_job(job: dict, change: dict) -> dict:
    """A copy of `job` with one pass's answers and options merged in."""
    out = json.loads(json.dumps(job))
    for name, value in change.items():
        if name == "options":
            out["options"] = {**out.get("options", {}), **value}
        else:
            out[name] = value
    return out


def select(jobs: dict[str, dict], name: str, stages: Iterable[str], limit: int | None) -> list[str]:
    """The keys a pass translates, in sorted order (so a --limit run is reproducible)."""
    _, pass_stages = VARIANTS[name]
    wanted = set(stages) & set(pass_stages)
    keys = sorted(k for k, j in jobs.items() if j.get("stage") in wanted)
    if name == "repeat" and len(keys) > REPEAT_SAMPLE:
        step = len(keys) / REPEAT_SAMPLE
        keys = [keys[int(i * step)] for i in range(REPEAT_SAMPLE)]
    return keys[:limit] if limit else keys


# --- message grouping ---------------------------------------------------------

_HEX = re.compile(r"0x[0-9A-Fa-f]+")
# A bare number, or a hexadecimal word of three or more digits that holds a decimal
# digit (handles are printed that way); names such as F32FtzRn or 2D stay.
_NUM = re.compile(r"\b(?:\d+|(?=[0-9A-Fa-f]*\d)[0-9A-Fa-f]{3,})\b")
_ID = re.compile(r"%[\w.]+")
_LINE = re.compile(r"^error: line \d+: ")
# The opcode of "Ampere instruction 243 at 0x... (...) is not implemented" names the
# instruction set's operation, not anything of the game, and it is what a failure is
# grouped by; the address and the instruction words after it are dropped.
_OPCODE = re.compile(r"Ampere instruction ([0-9a-f]{3}) at ")


def normalise(message: str, width: int = 160) -> str:
    """A failure message without the numbers and words that differ per program.

    Translator messages can carry instruction words, addresses and handles taken from
    the game; grouping on the normalised text keeps them out of summaries."""
    text = _LINE.sub("error: ", message.strip())
    opcode = _OPCODE.search(text)
    text = _ID.sub("%id", text)
    text = _HEX.sub("<hex>", text)
    text = _NUM.sub("N", text)
    if opcode:
        text = re.sub(r"Ampere instruction \S+ at ", f"Ampere instruction {opcode.group(1)} at ", text, count=1)
    return text[:width]


def fail_kind(message: str) -> str:
    """The kind a host failure starts with ("unsupported_opcode: ..."), else the whole status."""
    head = message.split(":", 1)[0].strip()
    return head if re.fullmatch(r"[a-z_]+", head) else "other"


# --- one pass -----------------------------------------------------------------

def read_results(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["key"]: row for row in csv.DictReader(fh, delimiter="\t")}


def run_pass(name: str, jobs: dict[str, dict], keys: list[str], xlate: Path, out: Path,
             workers: int, timeout: float, memory_limit: int, resume: bool) -> dict:
    """Translate `keys` with the pass's answers into out/<name>/; return pool facts."""
    folder = out / name
    spv_dir = folder / "spv"
    spv_dir.mkdir(parents=True, exist_ok=True)
    results_path = folder / "results.tsv"
    done = read_results(results_path) if resume else {}
    todo = [k for k in keys if k not in done]
    change, _ = VARIANTS[name]
    mode = "a" if done else "w"
    lock = threading.Lock()  # the pool calls on_result from its worker threads
    with open(results_path, mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        if mode == "w":
            writer.writeheader()

        def on_result(r: pool.Result) -> None:
            row = asdict(r)
            row["message"] = row["message"].replace("\t", " ").replace("\r", " ").replace("\n", " ")
            with lock:
                writer.writerow({f: row[f] for f in RESULT_FIELDS})
                fh.flush()

        t0 = time.perf_counter()
        # A .py translator is a stand-in that speaks the same protocol (the tests use one).
        command = [sys.executable, "-I", str(xlate)] if xlate.suffix == ".py" else [str(xlate)]
        _, stats = pool.run_pool(command, ((variant_job(jobs[k], change), spv_dir) for k in todo),
                                 workers, folder / "logs", timeout=timeout, memory_limit=memory_limit,
                                 on_result=on_result)
        wall = time.perf_counter() - t0
    return {"pass": name, "requested": len(keys), "translated_now": len(todo), "wall_s": round(wall, 1),
            "workers": workers, "pool": asdict(stats)}


def validate_pass(name: str, out: Path, tool: str, workers: int) -> None:
    """spirv-val every module the pass produced; failures are checked again with relaxed layout."""
    folder = out / name
    results = read_results(folder / "results.tsv")
    items = [(k, folder / "spv" / f"{k}.spv") for k, r in sorted(results.items()) if r["status"] == "ok"]
    strict = validate.validate_all(tool, items, workers)
    failed = [(v.key, folder / "spv" / f"{v.key}.spv") for v in strict if not v.ok]
    relaxed = {v.key: v for v in validate.validate_all(tool, failed, workers, validate.RELAXED_ARGS)}
    with open(folder / "validation.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(("key", "strict_ok", "relaxed_ok", "message"))
        for v in strict:
            r = relaxed.get(v.key)
            w.writerow((v.key, int(v.ok), int(v.ok or (r is not None and r.ok)), v.message.replace("\t", " ")))


def spv_digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


# --- summaries ----------------------------------------------------------------

def summarise_pass(name: str, out: Path, jobs: dict[str, dict], meta: dict[str, dict[str, str]],
                   run_facts: dict | None) -> dict:
    folder = out / name
    results = read_results(folder / "results.tsv")
    by_stage: dict[str, list[dict[str, str]]] = defaultdict(list)
    for key, row in results.items():
        by_stage[jobs[key]["stage"]].append(row)
    per_stage = {}
    for stage in STAGES:
        rows = by_stage.get(stage, [])
        if not rows:
            continue
        status = Counter(r["status"] for r in rows)
        micros = [int(r["micros"]) for r in rows if r["status"] == "ok"]
        per_stage[stage] = {"jobs": len(rows), "status": dict(sorted(status.items())),
                            "translator": validate.timing_summary(micros)}
    groups: Counter = Counter()
    stages_of: dict[tuple, Counter] = defaultdict(Counter)
    for key, row in results.items():
        if row["status"] == "ok":
            continue
        msg = row["message"]
        group = (row["status"], fail_kind(msg) if row["status"] == "fail" else row["status"], normalise(msg))
        groups[group] += 1
        stages_of[group][jobs[key]["stage"]] += 1
    summary = {
        "pass": name,
        "jobs": len(results),
        "status": dict(sorted(Counter(r["status"] for r in results.values()).items())),
        "translator_all": validate.timing_summary([int(r["micros"]) for r in results.values() if r["status"] == "ok"]),
        "per_stage": per_stage,
        "failures": [{"status": g[0], "kind": g[1], "message": g[2], "count": n, "stages": dict(sorted(stages_of[g].items()))}
                     for g, n in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0]))],
        "run": run_facts,
    }
    vpath = folder / "validation.tsv"
    if vpath.exists():
        with open(vpath, encoding="utf-8", newline="") as fh:
            vrows = list(csv.DictReader(fh, delimiter="\t"))
        rules: Counter = Counter()
        rule_stages: dict[str, Counter] = defaultdict(Counter)
        for v in vrows:
            if v["strict_ok"] == "1":
                continue
            rule = normalise(v["message"])
            rules[rule] += 1
            rule_stages[rule][jobs[v["key"]]["stage"]] += 1
        summary["validation"] = {
            "checked": len(vrows),
            "strict_ok": sum(v["strict_ok"] == "1" for v in vrows),
            "relaxed_only_ok": sum(v["strict_ok"] == "0" and v["relaxed_ok"] == "1" for v in vrows),
            "fail": sum(v["relaxed_ok"] == "0" for v in vrows),
            "rules": [{"rule": r, "count": n, "stages": dict(sorted(rule_stages[r].items()))}
                      for r, n in sorted(rules.items(), key=lambda kv: (-kv[1], kv[0]))],
        }
    if meta:
        extra = Counter()
        for key, row in results.items():
            m = meta.get(key, {})
            extra[(m.get("group", "?"), m.get("only_product100", "?"), row["status"])] += 1
        summary["by_group"] = [{"group": g, "only_product100": p, "status": s, "count": n}
                               for (g, p, s), n in sorted(extra.items())]
    return summary


def compare_with_default(name: str, out: Path, jobs: dict[str, dict]) -> dict:
    """How many modules differ from the default pass (answer sensitivity, or nondeterminism for repeat)."""
    base = read_results(out / "default" / "results.tsv")
    other = read_results(out / name / "results.tsv")
    changed: Counter = Counter()
    same: Counter = Counter()
    status_changed: Counter = Counter()
    for key, row in other.items():
        stage = jobs[key]["stage"]
        b = base.get(key)
        if b is None:
            continue
        if (b["status"] == "ok") != (row["status"] == "ok"):
            status_changed[stage] += 1
            continue
        if row["status"] != "ok":
            continue
        d0 = spv_digest(out / "default" / "spv" / f"{key}.spv")
        d1 = spv_digest(out / name / "spv" / f"{key}.spv")
        (same if d0 == d1 and d0 is not None else changed)[stage] += 1
    return {"against": "default", "changed": dict(sorted(changed.items())), "same": dict(sorted(same.items())),
            "status_changed": dict(sorted(status_changed.items()))}


# --- command line -------------------------------------------------------------

def inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def cmd_run(args: argparse.Namespace) -> int:
    split, work = args.split.resolve(), args.work.resolve()
    if inside_repo(work):
        print(f"corpus: --work {work} is inside the repository; results name the game's programs.", file=sys.stderr)
        return 2
    if not (split / "jobs.jsonl").exists():
        print(f"corpus: no jobs.jsonl in {split}.", file=sys.stderr)
        return 2
    if not args.xlate.exists():
        print(f"corpus: translator not found: {args.xlate}", file=sys.stderr)
        return 2
    tool = validate.find_spirv_val(args.spirv_val)
    if tool is None and not args.no_validate:
        print("corpus: spirv-val not found (--spirv-val, VULKAN_SDK or PATH).", file=sys.stderr)
        return 2
    stages = tuple(s for s in args.stages.split(",") if s) if args.stages else STAGES
    if any(s not in STAGES for s in stages):
        print(f"corpus: unknown stage in {args.stages!r}.", file=sys.stderr)
        return 2
    jobs, meta = load_jobs(split), load_meta(split)
    work.mkdir(parents=True, exist_ok=True)
    available = pool.available_ram()
    limit = min(args.memory_limit * pool.GIB, int(available * 0.7)) if available else args.memory_limit * pool.GIB
    print(f"corpus: {len(jobs)} jobs, available RAM {available / pool.GIB:.2f} GiB, "
          f"children limit {limit / pool.GIB:.2f} GiB", flush=True)
    for name in args.passes:
        keys = select(jobs, name, stages, args.limit)
        print(f"corpus: pass {name}: {len(keys)} jobs", flush=True)
        facts = run_pass(name, jobs, keys, args.xlate.resolve(), work, args.workers, args.timeout, limit,
                         resume=not args.fresh)
        facts["available_ram_at_start"] = available
        if not args.no_validate:
            t0 = time.perf_counter()
            validate_pass(name, work, tool, args.workers)
            facts["validate_wall_s"] = round(time.perf_counter() - t0, 1)
        summary = summarise_pass(name, work, jobs, meta, facts)
        if name not in ("default", "raw", "nodebug", "debug") and (work / "default" / "results.tsv").exists():
            summary["compare"] = compare_with_default(name, work, jobs)
        (work / f"summary-{name}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"corpus: pass {name}: {summary['status']} in {facts['wall_s']} s", flush=True)
    return 0


def summarise_notes(name: str, out: Path, jobs: dict[str, dict]) -> dict:
    """What the modules' resource descriptions say beyond ok: the translator's own log lines
    (an operation it dropped is logged, not failed), the answers taken by default, the
    capabilities the post-pass added, the recovery passes and raw global access."""
    folder = out / name
    results = read_results(folder / "results.tsv")
    notes: dict[str, Counter] = defaultdict(Counter)
    capabilities_added: dict[str, Counter] = defaultdict(Counter)
    passes: Counter = Counter()
    raw: Counter = Counter()
    fetch1d: Counter = Counter()
    unreadable = 0
    for key, row in sorted(results.items()):
        if row["status"] != "ok":
            continue
        stage = jobs[key]["stage"]
        try:
            res = json.loads((folder / "spv" / f"{key}.res.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable += 1
            continue
        for note in sorted({normalise(n) for n in res.get("notes", [])}):
            notes[note][stage] += 1
        for cap in sorted(set(res.get("capabilities_added", []))):
            capabilities_added[str(cap)][stage] += 1
        passes[(stage, res.get("passes", 1))] += 1
        if (res.get("raw_access") or {}).get("physical_pointers"):
            raw[stage] += 1
        if res.get("fetch1d"):
            fetch1d[stage] += 1
    ordered = sorted(notes.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    return {
        "pass": name,
        "unreadable": unreadable,
        "notes": [{"note": n, "programs": sum(c.values()), "stages": dict(sorted(c.items()))} for n, c in ordered],
        "capabilities_added": {cap: dict(sorted(c.items())) for cap, c in sorted(capabilities_added.items())},
        "passes": [{"stage": s, "passes": p, "programs": n} for (s, p), n in sorted(passes.items())],
        "raw_access": dict(sorted(raw.items())),
        "fetch1d": dict(sorted(fetch1d.items())),
    }


def cmd_notes(args: argparse.Namespace) -> int:
    work = args.work.resolve()
    jobs = load_jobs(args.split.resolve())
    for name in args.passes or ["default"]:
        summary = summarise_notes(name, work, jobs)
        (work / f"notes-{name}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"corpus: notes for {name}: {len(summary['notes'])} distinct notes, "
              f"{summary['unreadable']} unreadable descriptions", flush=True)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    work = args.work.resolve()
    for path in sorted(work.glob("summary-*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        t = s["translator_all"]
        print(f"{s['pass']}: {s['jobs']} jobs {s['status']}; translator p50 {t['p50_ms']:.2f} ms, "
              f"p99 {t['p99_ms']:.2f} ms, sum {t['sum_s']:.1f} s; wall {s['run']['wall_s'] if s.get('run') else '?'} s")
        if "validation" in s:
            v = s["validation"]
            print(f"  spirv-val: {v['strict_ok']}/{v['checked']} strict, {v['relaxed_only_ok']} relaxed only, {v['fail']} fail")
        if "compare" in s:
            print(f"  against default: changed {s['compare']['changed']}, status changed {s['compare']['status_changed']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--split", type=Path, required=True)
    run.add_argument("--xlate", type=Path, required=True)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--pass", dest="passes", action="append", choices=PASSES)
    run.add_argument("--workers", type=int, default=12)
    run.add_argument("--stages", default="")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--memory-limit", type=float, default=4.0, help="GiB, also capped at 70%% of available RAM")
    run.add_argument("--spirv-val", default=None)
    run.add_argument("--no-validate", action="store_true")
    run.add_argument("--fresh", action="store_true", help="translate again even where results exist")
    rep = sub.add_parser("report")
    rep.add_argument("--work", type=Path, required=True)
    notes = sub.add_parser("notes", help="summarise the resource descriptions of a finished pass")
    notes.add_argument("--split", type=Path, required=True)
    notes.add_argument("--work", type=Path, required=True)
    notes.add_argument("--pass", dest="passes", action="append", choices=PASSES)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        args.passes = args.passes or ["default"]
        if args.workers < 1:
            ap.error("--workers must be at least 1")
        return cmd_run(args)
    if args.cmd == "notes":
        return cmd_notes(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
