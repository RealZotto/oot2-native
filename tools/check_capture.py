#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Judge one M1 capture against M1's numeric exit conditions (architecture
section 12, D25 level 1, D26).

    py -3 tools/check_capture.py CAPTURE [--json]

The capture is the file oot2.exe writes with OOT2_CAPTURE=1
(runtime/include/oot2/capture.h has the format). The verdict is PASS or FAIL,
then one line per condition with its value and its limit, then figures that
carry information without a limit. Exit codes: 0 pass, 1 fail, 2 the file is
not a capture (it cannot be read at all).

The conditions:

- the closing record is the file's last line: a capture without it was cut
  short (the run was killed, or crashed past the stop path) and fails, whatever
  the rest says;
- the run is a judged run: not census mode (OOT2_CENSUS), not a collection run
  (collect_variants, which skips draws), and the header names
  the back end, the driver, the display mode, the implicit Vulkan layers and
  the available RAM;
- no test setting is on: a header key such as test_movie_blit (a switch that
  replaces the movies' enhancement chain by a resample, so changes their
  pictures) must be 0, or absent from a capture of a build without it. A run
  with it on also counts every dispatch it did not run into the skipped draws,
  which fail on their own;
- no named fatal: the run ended with exit code 0;
- the title mark was reached, and the capture goes on for the whole window,
  60 s of host time from the mark;
- the frame rate over that window is at least 58: the presents whose host time
  lies in [mark, mark + 60 s), divided by 60;
- no skipped draw, shader miss, pipeline miss, raw-access miss or device fault
  over the whole run, each counter on its own (D25 level 1 counts them over the
  route, and the route starts at boot; the ones before the mark are named);
- at 60 s after the mark (the first memory sample at or after it, else the
  closing record when that comes at or after it): working set at most 5 GiB and
  private commit at most 6 GiB.

On a paced clock the frame rate holds by construction (M4 says so), so the
figures that carry the information come after: ticks lost and frame times in
the window, the render thread's busy time (the core time of the thread that
submitted most in each frame), the requested intervals, the capture's own cost.
The picture conditions of M1 belong to the title scoring, not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

WINDOW_NS = 60 * 1_000_000_000      # "60 s at the title" (M1 row)
MIN_FPS = 58.0                      # "averages at least 58 fps"
MAX_WORKING_SET = 5 << 30           # "working set at most 5 GiB ... at 60 s"
MAX_PRIVATE = 6 << 30               # "private commit at most 6 GiB at 60 s"
FATAL_CODES = {70: "fatal", 71: "named fatal (unimplemented)", 72: "guest stop", 73: "census frame limit"}
COUNTERS = ("skipped draws", "shader misses", "pipeline misses", "raw-access misses", "device faults")
# Header keys of the test switches that change a picture (settings.h): a run with one on is no proof run.
TEST_SETTINGS = ("test_movie_blit",)
REQUIRED_HEADER = ("back_end", "driver", "display_mode", "implicit_vulkan_layers", "ram_available", "census_mode",
                   "title_mark_present")

# Fields after the kind, for the records whose shape is fixed. A record with more fields than these keeps the extra ones
# unread, so a later version can append; fewer is an unreadable file.
SHAPES = {
    "present": 8,   # seq t due shown requested held frame_ns late
    "thread": 3,    # handle entry priority
    "shown": 2,     # seq tick
    "title": 2 + len(COUNTERS),
    "memory": 6,    # seq t working_set private available pages_output
    "count": 2 + len(COUNTERS),
    "cost": 6,      # seq t presents sum max sampler
    "gpu": 3,       # seq pass ns
    "end": 4 + 1 + len(COUNTERS) + 4 + 4,
}


class CaptureError(Exception):
    """The file cannot be read as a capture."""


@dataclass
class Present:
    seq: int
    t: int
    due: int
    shown: int
    requested: int
    held: int
    frame_ns: int
    late: int


@dataclass
class Frame:
    seq: int
    submitter: int
    submits: int
    submitters: int
    presenter: int
    threads: dict[int, tuple[int, int]]   # handle -> (cpu_ns, core_ns)


@dataclass
class Sample:
    seq: int
    t: int
    values: tuple[int, ...]


@dataclass
class End:
    code: int
    reason: str
    seq: int
    t: int
    fatals: int
    counters: tuple[int, ...]
    working_set: int
    private: int
    available: int
    pages_output: int
    cost_presents: int
    cost_sum: int
    cost_max: int
    sampler_ns: int


@dataclass
class Capture:
    header: dict[str, str] = field(default_factory=dict)
    presents: list[Present] = field(default_factory=list)
    frames: list[Frame] = field(default_factory=list)
    threads: dict[int, tuple[int, int]] = field(default_factory=dict)   # handle -> (entry, priority)
    shown: dict[int, int] = field(default_factory=dict)
    title: Sample | None = None
    memory: list[Sample] = field(default_factory=list)
    counts: list[Sample] = field(default_factory=list)
    costs: list[Sample] = field(default_factory=list)
    gpu: list[tuple[int, int, int]] = field(default_factory=list)
    end: End | None = None
    last_kind: str = ""
    cut_line: bool = False        # the last line has no line break: the file was cut inside it
    after_end: int = 0            # records after the closing record


def number(text: str, what: str, line: int) -> int:
    try:
        return int(text, 0)
    except ValueError:
        raise CaptureError(f"line {line}: {what}: {text!r} is not a number") from None


def parse(path: Path) -> Capture:
    """Read a capture. Unknown header keys and record kinds are kept or skipped; a known record with too few fields,
    or a field that is not a number, makes the file unreadable."""
    try:
        data = path.read_bytes()
    except OSError as e:
        raise CaptureError(f"{path}: {e.strerror}") from None
    text = data.decode("utf-8", errors="replace")
    cap = Capture()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:
        cap.cut_line = True
        lines.pop()   # a line cut short is no record
    if not lines or lines[0].split("\t")[:1] != ["oot2-capture"]:
        raise CaptureError(f"{path}: not a capture (the first line is not 'oot2-capture')")
    for no, line in enumerate(lines[1:], start=2):
        f = line.split("\t")
        kind, rest = f[0], f[1:]
        if cap.end is not None:
            cap.after_end += 1
        if kind == "header":
            if len(rest) < 2:
                raise CaptureError(f"line {no}: a header line needs a key and a value")
            cap.header[rest[0]] = "\t".join(rest[1:])
            cap.last_kind = kind
            continue
        if kind == "frame":
            if len(rest) < 5 or (len(rest) - 5) % 3:
                raise CaptureError(f"line {no}: a frame line has {len(rest)} fields")
            v = [number(x, "frame", no) for x in rest]
            threads = {v[i]: (v[i + 1], v[i + 2]) for i in range(5, len(v), 3)}
            cap.frames.append(Frame(v[0], v[1], v[2], v[3], v[4], threads))
            cap.last_kind = kind
            continue
        if kind not in SHAPES:
            cap.last_kind = kind   # a record kind this reader does not know: skipped
            continue
        if len(rest) < SHAPES[kind]:
            raise CaptureError(f"line {no}: a {kind} line has {len(rest)} fields, {SHAPES[kind]} expected")
        if kind == "end":
            v = [number(rest[0], "end", no)] + [number(x, "end", no) for x in rest[2:SHAPES[kind]]]
            k = len(COUNTERS)
            cap.end = End(v[0], rest[1], v[1], v[2], v[3], tuple(v[4:4 + k]), *v[4 + k:4 + k + 8])
        else:
            v = [number(x, kind, no) for x in rest[:SHAPES[kind]]]
            if kind == "present":
                cap.presents.append(Present(*v))
            elif kind == "thread":
                cap.threads[v[0]] = (v[1], v[2])
            elif kind == "shown":
                cap.shown[v[0]] = v[1]
            elif kind == "title":
                cap.title = Sample(v[0], v[1], tuple(v[2:]))
            elif kind == "memory":
                cap.memory.append(Sample(v[0], v[1], tuple(v[2:])))
            elif kind == "count":
                cap.counts.append(Sample(v[0], v[1], tuple(v[2:])))
            elif kind == "cost":
                cap.costs.append(Sample(v[0], v[1], tuple(v[2:])))
            elif kind == "gpu":
                cap.gpu.append((v[0], v[1], v[2]))
        cap.last_kind = kind
    return cap


@dataclass
class Condition:
    name: str
    ok: bool
    value: str
    limit: str


@dataclass
class Verdict:
    conditions: list[Condition]
    figures: dict[str, object]

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.conditions)


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def gib(n: int) -> str:
    return f"{n / (1 << 30):.3f} GiB"


def check(cap: Capture) -> Verdict:
    conds: list[Condition] = []
    figs: dict[str, object] = {}

    def cond(name: str, ok: bool, value: str, limit: str) -> None:
        conds.append(Condition(name, ok, value, limit))

    end = cap.end
    closed = end is not None and cap.last_kind == "end" and not cap.after_end and not cap.cut_line
    if closed:
        cond("closing record", True, "present, last line", "the last line")
    else:
        why = "none" if end is None else "records after it" if cap.after_end else "the file was cut after it"
        if cap.cut_line and end is None:
            why = "none; the file was cut inside a line"
        cond("closing record", False, why, "the last line")

    census = cap.header.get("census_mode", "0") != "0"
    # collect_variants (a development switch): the run skipped draws that stop on an answer, so its picture is wrong.
    # The skipped-draw counter fails it too; the mark names why, and a capture without the key is not a collection run.
    collect = cap.header.get("collect_variants", "0") != "0"
    mode = "census mode" if census else "collect_variants (draws skipped to collect shader variants)" if collect else "default mode"
    cond("judged run", not census and not collect, mode, "not census mode, not collect_variants")
    # A test switch that changes a picture (settings.h): its header key is "0", or "1 (...)"; a capture without the key
    # comes from a build that had no such switch.
    tests_on = [k for k in TEST_SETTINGS if cap.header.get(k, "0").split(" ")[0] != "0"]
    cond("test settings", not tests_on,
         "on: " + ", ".join(f"{k} = {cap.header[k]}" for k in tests_on) if tests_on else "none on",
         "none on (a test switch changes the picture; a capture of such a run is no proof)")
    missing = [k for k in REQUIRED_HEADER if k not in cap.header]
    cond("header", not missing, "complete" if not missing else "lacks " + ", ".join(missing), ", ".join(REQUIRED_HEADER))

    if end is None:
        cond("named fatals", False, "unknown: no closing record", "exit code 0")
    else:
        fatal = end.code != 0 or end.fatals != 0
        what = FATAL_CODES.get(end.code, "exit code") if end.code else "exit"
        cond("named fatals", not fatal, f"{end.fatals} ({what} {end.code}, {end.reason})", "exit code 0")

    title = cap.title
    if title is None:
        cond("title reached", False, f"no title mark (present {cap.header.get('title_mark_present', '?')})",
             "the mark's present")
    else:
        cond("title reached", True, f"present {title.seq} at {title.t / 1e9:.3f} s", "the mark's present")

    # The window and the frame rate.
    window: list[Present] = []
    if title is not None:
        t0, t1 = title.t, title.t + WINDOW_NS
        last_t = max((p.t for p in cap.presents), default=0)
        complete = last_t >= t1
        cond("window complete", complete, f"{(last_t - t0) / 1e9:.3f} s of presents after the mark",
             f"{WINDOW_NS / 1e9:.0f} s")
        window = [p for p in cap.presents if t0 <= p.t < t1]
        fps = len(window) / (WINDOW_NS / 1e9)
        cond("frame rate", complete and fps >= MIN_FPS, f"{fps:.2f} fps ({len(window)} presents)",
             f"at least {MIN_FPS:g} fps over {WINDOW_NS / 1e9:.0f} s")
    else:
        cond("window complete", False, "no mark", f"{WINDOW_NS / 1e9:.0f} s")
        cond("frame rate", False, "no window", f"at least {MIN_FPS:g} fps")

    # The counters of missing work, each on its own, over the whole run.
    final = end.counters if end is not None else cap.counts[-1].values if cap.counts else None
    at_mark = title.values if title is not None else None
    for i, name in enumerate(COUNTERS):
        if final is None:
            cond(name, False, "unknown", "0")
            continue
        before = at_mark[i] if at_mark is not None else None
        note = f" ({before} before the mark)" if before else ""
        cond(name, final[i] == 0, f"{final[i]}{note}", "0")

    # Memory at 60 s after the mark.
    if title is None:
        cond("working set at 60 s", False, "no mark", f"at most {gib(MAX_WORKING_SET)}")
        cond("private commit at 60 s", False, "no mark", f"at most {gib(MAX_PRIVATE)}")
    else:
        at = title.t + WINDOW_NS
        sample = next((m for m in cap.memory if m.t >= at), None)
        if sample is not None:
            ws, priv, where = sample.values[0], sample.values[1], f"sample at {sample.t / 1e9:.3f} s"
        elif end is not None and end.t >= at:
            ws, priv, where = end.working_set, end.private, f"closing record at {end.t / 1e9:.3f} s"
        else:
            ws = priv = None
            where = "no sample at or after 60 s"
        if ws is None:
            cond("working set at 60 s", False, where, f"at most {gib(MAX_WORKING_SET)}")
            cond("private commit at 60 s", False, where, f"at most {gib(MAX_PRIVATE)}")
        else:
            cond("working set at 60 s", ws <= MAX_WORKING_SET, f"{gib(ws)} ({where})", f"at most {gib(MAX_WORKING_SET)}")
            cond("private commit at 60 s", priv <= MAX_PRIVATE, f"{gib(priv)} ({where})", f"at most {gib(MAX_PRIVATE)}")

    # Figures without a limit.
    figs["presents"] = len(cap.presents)
    if window:
        frame_ns = [p.frame_ns for p in window if p.frame_ns]
        figs["late ticks in the window"] = sum(p.late for p in window)
        figs["late presents in the window"] = sum(1 for p in window if p.late)
        figs["frame time p50/p99/max ms"] = [round(percentile(frame_ns, q) / 1e6, 3) for q in (0.5, 0.99, 1.0)]
        hist: dict[int, int] = {}
        for p in window:
            hist[p.requested] = hist.get(p.requested, 0) + 1
        figs["requested intervals"] = {str(k): v for k, v in sorted(hist.items())}
        seqs = {p.seq for p in window}
        busy = [f.threads.get(f.submitter, (0, 0))[1] for f in cap.frames if f.seq in seqs and f.submitter]
        figs["render thread busy p50/p99/max ms"] = [round(percentile(busy, q) / 1e6, 3) for q in (0.5, 0.99, 1.0)]
        cpu: dict[int, int] = {}
        for f in cap.frames:
            if f.seq in seqs:
                for h, (c, _) in f.threads.items():
                    cpu[h] = cpu.get(h, 0) + c
        top = sorted(cpu.items(), key=lambda kv: -kv[1])[:5]
        figs["thread CPU ms per frame, top 5"] = {hex(h): round(c / len(window) / 1e6, 3) for h, c in top}
        avail = [m.values[2] for m in cap.memory if title is not None and title.t <= m.t < title.t + WINDOW_NS]
        if avail:
            figs["available RAM min"] = gib(min(avail))
        pages = [m.values[3] for m in cap.memory if title is not None and title.t <= m.t <= title.t + WINDOW_NS
                 and m.values[3] >= 0]
        if len(pages) >= 2:
            figs["pages written to the page file in the window (every process)"] = pages[-1] - pages[0]
    if end is not None and end.cost_presents:
        figs["capture cost per present, mean/max us"] = [round(end.cost_sum / end.cost_presents / 1e3, 2),
                                                         round(end.cost_max / 1e3, 2)]
        figs["capture sampling thread CPU ms"] = round(end.sampler_ns / 1e6, 2)
    if cap.gpu:
        figs["gpu timestamps"] = len(cap.gpu)
    return Verdict(conds, figs)


def report(path: Path, verdict: Verdict) -> str:
    out = [f"{'PASS' if verdict.passed else 'FAIL'} {path}"]
    for c in verdict.conditions:
        out.append(f"  {'ok  ' if c.ok else 'FAIL'} {c.name}: {c.value} (limit: {c.limit})")
    for k, v in verdict.figures.items():
        out.append(f"  info {k}: {v}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge an M1 capture against M1's numeric exit conditions.")
    ap.add_argument("capture", type=Path)
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON")
    a = ap.parse_args(argv)
    try:
        cap = parse(a.capture)
    except CaptureError as e:
        print(f"check_capture: {e}", file=sys.stderr)
        return 2
    verdict = check(cap)
    if a.json:
        print(json.dumps({"passed": verdict.passed, "conditions": [c.__dict__ for c in verdict.conditions],
                          "figures": verdict.figures}, indent=1))
    else:
        print(report(a.capture, verdict))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    sys.exit(main())
