# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The cost of the Vulkan replay per graphics draw, as a frame of N draws grows (tests/runtime/vk_replay_cost.c says what
runs for real and what is faked). Run by hand it is a measurement; under `unittest discover` it is a short smoke case (a few
frames of patterns a and b, no timing is judged) at the end of the file.

This driver
- makes the synthetic shader packs the program opens: a base pack of 40,000 rows, a variants pack of 20 rows, and a map of
  1,000 rows from (stage key, full ledger digest) to the projected digest or to the base; 64 vertex and 64 pixel stages carry
  full descriptions, the others only what the pack format needs;
- builds the program twice, with and without the replay's timers (oot2/bench_timers.h), objects cached, at below-normal
  priority, a few compiles at a time;
- runs it for the patterns and the sizes asked for, and prints the table.

    py -3 tests/test_runtime_vk_replay_cost.py --work <a folder for packs and programs> --out results.json

The patterns: a, every draw the same program and bindings; b, before each draw one texture, one uniform range of the vertex
stage and the program of the next of 64 pairs; c, b and the vertex attributes, streams and buffers of the draw's mesh; d, b and a storage buffer each pixel stage writes; e, ten draws that repeat with fixed program
pairs and textures, for the counters.

The counters (runtime/include/oot2/replay_count.h, OOT2_REPLAY_COUNTERS) run in their own program runs (--count-patterns), never
in a timed one: the COUNTS line holds what they counted over the measured frames and the rc lines on stderr what the program
printed at its exit. The smoke case below checks both against streams whose repeats are known by construction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
import build_game  # noqa: E402
import cache_pack as cp  # noqa: E402
import layout  # noqa: E402
from prepare import ledger as prepare_ledger  # noqa: E402

RUNTIME = ROOT / "runtime"
BENCH_SOURCE = ROOT / "tests" / "runtime" / "vk_replay_cost.c"
MODULE_SIZE = 0x8000
NPAIR = 64
BASE_ROWS = 40000          # rows of the synthetic base pack
MAP_ROWS = 1000            # rows of the synthetic variant map
VARIANT_ROWS = 20          # rows of the synthetic variants pack
TRANSLATOR = hashlib.sha256(b"synthetic translator of the replay cost benchmark").hexdigest()
STUB_MODULE = b"".join(w.to_bytes(4, "little") for w in (0x07230203, 0x00010300, 0, 1, 0))
# The files that hold the replay's timers or the benchmark's hooks: their objects are made once for each build.
INSTRUMENTED = {"gpu/vk_replay.c", "gpu/vk_pipeline.c", "gpu/vk_replay_dev.c", "nvn/nvn_ledger.c"}
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0
IDLE = 0x00000040 if os.name == "nt" else 0
PRIORITY = BELOW_NORMAL   # main() sets IDLE for --priority idle: a compile that runs only on cores nothing else wants


# ---- the packs -------------------------------------------------------------------------------------------------------

def key_of(prefix: str, i: int) -> str:
    return hashlib.sha256(f"{prefix}-{i}".encode()).hexdigest()[:16] + ".00000001"


def canonical(answers: dict) -> str:
    return json.dumps(answers, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# What the ledger's tracker holds for the stages of the frames the program replays (nvn_ledger.c answers_text): four float
# attributes for the vertex stage; a float target and a 2D texture at slot 0 for the pixel stage.
VS_ANSWERS = {"attr": {"0": "float", "1": "float", "2": "float", "3": "float"}}
PS_ANSWERS = {"fout": {"0": "float"}, "tex": {"0": "2d"}}
VS_LEDGER = hashlib.sha256(canonical(VS_ANSWERS).encode()).digest()
PS_LEDGER = hashlib.sha256(canonical(PS_ANSWERS).encode()).digest()
VS_VARIANT = {"attr": {"0": "sint"}}
PS_VARIANT = {"tex": {"0": "cube"}}


def cbuf(slot: int, binding: int, set_: int = 1) -> dict:
    return {"set": set_, "binding": binding, "slot": slot, "ordinal": 0, "size": 65536, "role": "user"}


def tex2d(slot: int = 0, c2_slot: int = 0) -> dict:
    return {"set": 0, "binding": 0, "via": "push_slot", "push_slot": slot, "push_offset": 16 + 4 * slot, "type_raw": 3,
            "type": "2d", "shadow": False, "kind": "combined", "array_length": 1, "separate": False, "cbuf_slot_raw": -1,
            "c2_slot": c2_slot, "format": "Unknown", "flags_raw": 0, "image_store": False}


def vin(location: int, components: int) -> dict:
    return {"location": location, "component": 0, "components": components, "type": "float32", "used": True,
            "answer": "float"}


def ssbo_entry() -> dict:
    """A storage buffer the pixel stage writes: ordinal 0, its address and size at the pixel table of c[0] (vk_nvn.c C0_SSBO_PS)."""
    return {"set": 1, "binding": 25, "ordinal": 0, "cb_slot": 0, "cb_word_offset": 0x510 // 4, "cb_byte_offset": 0x510,
            "role": "guest", "address_byte_offset": 0x510, "size_byte_offset": 0x518, "write": True, "module_reads": True,
            "module_writes": True, "non_writable": False, "non_readable": False}


def description(key: str, stage: str, answers: dict | None = None, ssbo: bool = False) -> bytes:
    """A description with the members the runtime reads of a stage (shader_desc.c): the vertex stage's six banks (the driver's
    c0, its own constants c1, the handles c2, two user banks, the roots c24) and four attributes; the pixel stage's five
    banks and one combined texture through a push constant slot, and with ssbo a storage buffer it writes."""
    vs = stage == "vs"
    ans = {"tex": {}, "img": {}, "attr": {}, "fout": {}}
    ans.update(answers or {})
    d = {"schema": cp.DESC_SCHEMA, "key": key, "stage": stage, "exec_model": "Vertex" if vs else "Fragment", "entry": "main",
         "translator_id": TRANSLATOR,
         "answers": {**ans, "early_fragment_tests": False, "heap_images": 33025, "heap_samplers": 4096},
         "spv_bytes": len(STUB_MODULE), "spv_sha256": hashlib.sha256(STUB_MODULE).hexdigest(),
         "layout": {"name": "uni", "role": "vertex" if vs else "pixel", "fits": True, "overflow": [], "uniform_slots": []},
         "raw_access": {"physical_pointers": False, "page_table": False, "store_bits": False},
         "support_buffer": {"set": 1, "binding": 0, "size": 1392,
                            "fields": [{"name": "viewport_size", "offset": 160, "bytes": 16, "read": False}]},
         "interface": {"inputs": [], "outputs": [], "builtins_in": [], "builtins_out": []},
         "cbufs": ([cbuf(0, 1), cbuf(1, 2), cbuf(2, 3), cbuf(3, 4), cbuf(4, 5), cbuf(24, 6)] if vs
                   else [cbuf(0, 14), cbuf(1, 15), cbuf(2, 16), cbuf(3, 17), cbuf(24, 18)]),
         "sbufs": [ssbo_entry()] if ssbo and not vs else [], "textures": [] if vs else [tex2d()], "images": [],
         "compute": None,
         "vertex_inputs": [vin(0, 3), vin(1, 3), vin(2, 2), vin(3, 4)] if vs else None, "notes": []}
    return json.dumps(d, indent=1).encode("utf-8")


def filler_description(key: str, answers: dict | None = None) -> bytes:
    d = {"schema": cp.DESC_SCHEMA, "key": key, "translator_id": TRANSLATOR, "spv_bytes": len(STUB_MODULE),
         "spv_sha256": hashlib.sha256(STUB_MODULE).hexdigest(), "answers": answers or {}}
    return json.dumps(d).encode("utf-8")


def make_base(path: Path, vs_keys: list[str], ps_keys: list[str], ssbo: bool = False) -> list[str]:
    """The base pack: the 128 used stages and fillers up to BASE_ROWS. Returns the filler keys."""
    if path.exists():
        return [key_of("filler", i) for i in range(BASE_ROWS - 2 * NPAIR)]
    fillers = [key_of("filler", i) for i in range(BASE_ROWS - 2 * NPAIR)]
    entries = [cp.Entry(k, STUB_MODULE, description(k, "vs")) for k in vs_keys]
    entries += [cp.Entry(k, STUB_MODULE, description(k, "ps", ssbo=ssbo)) for k in ps_keys]
    entries += [cp.Entry(k, STUB_MODULE, filler_description(k)) for k in fillers]
    cp.write_pack(path, entries, translator=TRANSLATOR)
    return fillers


def make_variants(vdir: Path, tag: str, vs_keys: list[str], ps_keys: list[str], fillers: list[str], ps_variants: int,
                  vs_variants: int, ssbo: bool = False) -> tuple[Path, Path]:
    """The variants pack and the map for a share of variant stages: the first ps_variants pixel stages and the first
    vs_variants vertex stages of the program pairs map to a variant (the rest to the base); the pack and the map are padded
    with filler rows to VARIANT_ROWS variant rows and MAP_ROWS map rows."""
    pack, vmap = vdir / f"variants-{tag}.bin", vdir / f"map-{tag}.bin"
    if pack.exists() and vmap.exists():
        return pack, vmap
    rng = random.Random(0x5EED)
    entries: list[cp.Entry] = []
    dest: dict[tuple[str, bytes], bytes] = {}
    zero = bytes(32)
    for i, k in enumerate(ps_keys):
        if i < ps_variants:
            digest = cp.answers_digest(PS_VARIANT)
            entries.append(cp.Entry(k, STUB_MODULE, description(k, "ps", PS_VARIANT, ssbo=ssbo), answers=digest))
            dest[(k, PS_LEDGER)] = digest
        else:
            dest[(k, PS_LEDGER)] = zero
    for i, k in enumerate(vs_keys):
        if i < vs_variants:
            digest = cp.answers_digest(VS_VARIANT)
            entries.append(cp.Entry(k, STUB_MODULE, description(k, "vs", VS_VARIANT), answers=digest))
            dest[(k, VS_LEDGER)] = digest
        else:
            dest[(k, VS_LEDGER)] = zero
    # padding: variant rows on filler keys, and map rows (to the base) until the counts are VARIANT_ROWS and MAP_ROWS
    i = 0
    while len(entries) < VARIANT_ROWS:
        k = fillers[i * 997 % len(fillers)]
        digest = cp.answers_digest({"tex": {"1": "3d", "0": str(i)}})
        entries.append(cp.Entry(k, STUB_MODULE, filler_description(k, {"tex": {"1": "3d"}}), answers=digest))
        for _ in range(3):
            dest[(k, rng.randbytes(32))] = digest
        i += 1
    while len(dest) < MAP_ROWS:
        dest[(fillers[rng.randrange(len(fillers))], rng.randbytes(32))] = zero
    cp.write_pack(pack, entries, translator=TRANSLATOR, role=cp.VARIANTS)
    vmap.write_bytes(prepare_ledger.variant_map_bytes(dest, "none"))
    return pack, vmap


# ---- the build -------------------------------------------------------------------------------------------------------

def newest_header() -> float:
    newest = 0.0
    for part in ("include/oot2", "core", "nvn", "gpu"):
        for p in (RUNTIME / part).glob("*.h"):
            newest = max(newest, p.stat().st_mtime)
    return newest


def link_flags(os_name: str | None = None) -> list[str]:
    """The linker arguments of the program: the PE flag on Windows, the thread and dynamic-loader libraries elsewhere."""
    return ["-Wl,--disable-high-entropy-va"] if (os_name or os.name) == "nt" else ["-pthread", "-ldl"]


def build(out: Path, timers: bool, jobs: int, cc: str, include: Path, glslang: str) -> Path:
    """The program with -DOOT2_BENCH_HOOKS and, when timers, -DOOT2_BENCH_TIMERS, in out/<variant>.exe."""
    gen = out / "gen"
    layout.write(layout.place({"main": MODULE_SIZE}), gen)
    glue = build_game.write_shader_glue(RUNTIME, gen, glslang)
    variant = "timed" if timers else "clean"
    sources: list[tuple[Path, Path]] = []
    for part in ("core", "nvn", "gpu"):
        for src in sorted((RUNTIME / part).glob("*.c")):
            rel = f"{part}/{src.name}"
            folder = variant if rel in INSTRUMENTED else "common"
            sources.append((src, out / "obj" / folder / f"{part}_{src.stem}.o"))
    sources.append((glue, out / "obj" / "common" / "gpu_shaders.o"))
    sources.append((BENCH_SOURCE, out / "obj" / variant / "vk_replay_cost.o"))
    flags = ["-std=gnu11", "-O2", "-march=x86-64-v3", "-fno-strict-aliasing", "-Wall", "-Wextra", "-DOOT2_BENCH_HOOKS"]
    if timers:
        flags.append("-DOOT2_BENCH_TIMERS")
    incs = ["-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "nvn"), "-I", str(gen),
            "-I", str(include), "-I", str(include / "Volk")]
    headers = newest_header()
    todo = []
    for src, obj in sources:
        obj.parent.mkdir(parents=True, exist_ok=True)
        specific = obj.parent.name == variant
        defs = flags if specific else [f for f in flags if not f.startswith("-DOOT2_BENCH")]
        stamp = max(src.stat().st_mtime, headers)
        if obj.exists() and obj.stat().st_mtime >= stamp:
            continue
        todo.append((src, obj, defs))

    def compile_one(job) -> tuple[str, str]:
        src, obj, defs = job
        if GUARD:
            GUARD.check()   # no compile starts once the marker names a timed job or the memory ran low
        tmp = obj.with_name(obj.name + ".tmp")
        done = subprocess.run([cc, "-c", *defs, *incs, str(src), "-o", str(tmp)], capture_output=True, text=True,
                              creationflags=PRIORITY)
        if done.returncode:
            raise RuntimeError(f"compile of {src.name} failed:\n{done.stdout}{done.stderr}")
        os.replace(tmp, obj)   # an object is whole or not there
        return src.name, done.stderr

    t0 = time.time()
    if todo and GUARD:
        GUARD.wait_ready()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for name, warnings in pool.map(compile_one, todo):
            if warnings.strip() and name == BENCH_SOURCE.name:
                print(warnings, file=sys.stderr)
    if todo:
        print(f"built {len(todo)} object(s) of {variant} in {time.time() - t0:.0f} s", file=sys.stderr)
    exe = out / f"vk_replay_cost-{variant}.exe"
    objs = [str(o) for _, o in sources]
    if not exe.exists() or any(Path(o).stat().st_mtime > exe.stat().st_mtime for o in objs):
        done = subprocess.run([cc, *objs, "-o", str(exe), *link_flags()], capture_output=True, text=True,
                              creationflags=PRIORITY)
        if done.returncode:
            raise RuntimeError(f"link failed:\n{done.stdout}{done.stderr}")
    return exe


# ---- the machine -----------------------------------------------------------------------------------------------------

class Aborted(Exception):
    """The machine rules stopped the work: it can be started again, the objects made are kept."""


class Guard:
    """The machine rules of this measurement: work starts only while the BUSY line of the busy-marker file does not name a
    timed or heavy job and at least start_gib GiB are available; once started, no compile and no run begins after the marker
    named one or the memory fell under keep_gib."""

    def __init__(self, marker: Path, start_gib: float = 3.0, keep_gib: float = 1.5, wait_minutes: float = 85.0,
                 require_quiet: bool = False):
        self.marker, self.start_gib, self.keep_gib, self.wait_minutes = marker, start_gib, keep_gib, wait_minutes
        self.require_quiet = require_quiet

    def holder_busy(self) -> bool:
        head = busy_line(self.marker)[len("BUSY:"):]
        return not head.strip().lower().startswith(("none", "free")) and re.search(r"TIMED|HEAVY", head, re.I) is not None

    def check(self) -> None:
        if self.holder_busy():
            raise Aborted(f"the busy marker names a timed or heavy job now: {busy_line(self.marker)[:140]}")
        if available_gib() < self.keep_gib:
            raise Aborted(f"only {available_gib():.2f} GiB available, under the {self.keep_gib} GiB to keep")

    def quiet(self, window: float = 2.0, threshold: float = 12.0) -> tuple[bool, str]:
        """No compiler, build or game of anyone's is running and the processors were busy under `threshold` percent over `window`
        seconds (measured while nothing of this program runs)."""
        names = [n for n in foreign_processes() if n.lower() != "dotnet.exe"]   # an idle MSBuild node is not a load
        idle0, total0 = system_times()
        time.sleep(window)
        idle1, total1 = system_times()
        busy = 100.0 * (1 - (idle1 - idle0) / (total1 - total0)) if total1 > total0 else 0.0
        return (not names and busy < threshold), f"{busy:.0f}% busy, running: {','.join(names) or 'nothing foreign'}"

    def wait_quiet(self) -> None:
        end, shown = time.time() + self.wait_minutes * 60, None
        while True:
            ok, why = self.quiet()
            if ok:
                return
            if why != shown:
                print(time.strftime("%H:%M:%S"), "waiting for a quiet machine:", why, file=sys.stderr, flush=True)
                shown = why
            if time.time() > end:
                raise Aborted(f"no quiet machine in {self.wait_minutes:.0f} minutes: {why}")
            time.sleep(3)

    def wait_ready(self) -> None:
        end, shown = time.time() + self.wait_minutes * 60, None
        while True:
            busy, gib = self.holder_busy(), available_gib()
            if not busy and gib >= self.start_gib:
                return
            why = f"marker busy={busy}, {gib:.2f} GiB available (want {self.start_gib})"
            if why != shown:
                print(time.strftime("%H:%M:%S"), "waiting:", why, file=sys.stderr, flush=True)
                shown = why
            if time.time() > end:
                raise Aborted(f"not ready after {self.wait_minutes:.0f} minutes: {why}")
            time.sleep(20)


GUARD: Guard | None = None


def system_times() -> tuple[int, int]:
    """(idle, total) ticks of all processors; (0, 0) off Windows."""
    if os.name != "nt":
        return 0, 0
    import ctypes
    from ctypes import wintypes

    class FT(ctypes.Structure):
        _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

    i, k, u = FT(), FT(), FT()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))  # type: ignore[attr-defined]
    val = lambda f: f.hi << 32 | f.lo  # noqa: E731
    return val(i), val(k) + val(u)   # kernel time includes idle


def available_gib() -> float:
    if os.name != "nt":
        return 0.0
    import ctypes
    from ctypes import wintypes

    class MS(ctypes.Structure):
        _fields_ = [("len", wintypes.DWORD), ("load", wintypes.DWORD), ("tp", ctypes.c_uint64), ("ap", ctypes.c_uint64),
                    ("tpf", ctypes.c_uint64), ("apf", ctypes.c_uint64), ("tv", ctypes.c_uint64), ("av", ctypes.c_uint64),
                    ("ae", ctypes.c_uint64)]

    m = MS()
    m.len = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))  # type: ignore[attr-defined]
    return m.ap / (1 << 30)


def foreign_processes() -> list[str]:
    """Names of the processes that make a timing not final: a foreign compiler or build, the game."""
    if os.name != "nt":
        return []
    done = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True)
    names = {line.split('","')[0].strip('"') for line in done.stdout.splitlines() if line}
    watch = re.compile(r"^(clang(-\d+)?(\+\+)?|ninja|oot2|dotnet|msbuild|cl|link|ld|lld|cmake)\.exe$", re.I)
    return sorted(n for n in names if watch.match(n))


BUSY_ENV = "OOT2_BENCH_BUSY_FILE"
COUNTERS_ENV = "OOT2_REPLAY_COUNTERS"


def default_busy_file() -> Path | None:
    """The busy-marker file named by the environment, or None: the tracked file names no machine's path."""
    named = os.environ.get(BUSY_ENV)
    return Path(named) if named else None


def busy_line(marker: Path | None) -> str:
    """The first line of the marker file that starts with BUSY: ("none" after it, or a text without TIMED or HEAVY, is idle)."""
    if marker is None:
        return "(no busy-marker file)"
    try:
        for line in marker.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("BUSY:"):
                return line[:300]
    except OSError:
        pass
    return "(busy-marker file unreadable)"


# ---- running ---------------------------------------------------------------------------------------------------------

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def parse(stdout: str) -> dict:
    out: dict = {"bench": [], "split": [], "overhead": [], "total": [], "timer": {}, "counts": {}}
    for line in stdout.splitlines():
        kv = {k: v.strip('"') for k, v in KV.findall(line)}
        if line.startswith("BENCH "):
            out["bench"].append(kv)
        elif line.startswith("SPLIT "):
            out["split"].append(kv)
        elif line.startswith("SPLITOVERHEAD "):
            out["overhead"].append(kv)
        elif line.startswith("SPLITTOTAL "):
            out["total"].append(kv)
        elif line.startswith("TIMER "):
            out["timer"] = kv
        elif line.startswith("PROCESS "):
            out["process"] = kv
        elif line.startswith("COUNTS "):
            out["counts"] = {k: int(v) if v.isdigit() else v for k, v in kv.items()}
    return out


def rc_lines(stderr: str) -> list[dict]:
    """The counters' lines of a log: "[rt] rc kind=exit group=sub present=0 presents=0 key=value ..." as dicts of strings."""
    out = []
    for line in stderr.splitlines():
        if line.startswith("[rt] rc "):
            out.append(dict(item.split("=", 1) for item in line[len("[rt] rc "):].split()))
    return out


def run_exe(exe: Path, args: list[str], timeout: int = 600, env: dict | None = None) -> tuple[dict, dict]:
    if GUARD:
        GUARD.wait_ready()
        if GUARD.require_quiet:
            GUARD.wait_quiet()
    idle0, total0 = system_times()
    procs0 = foreign_processes()
    t0 = time.time()
    full_env = {k: v for k, v in os.environ.items() if k != COUNTERS_ENV}
    full_env.update(env or {})
    done = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=timeout, env=full_env)
    wall = time.time() - t0
    idle1, total1 = system_times()
    procs1 = foreign_processes()
    if done.returncode:
        raise RuntimeError(f"{exe.name} {' '.join(args)} exited {done.returncode}:\n{done.stdout}\n{done.stderr}")
    busy = 100.0 * (1 - (idle1 - idle0) / (total1 - total0)) if total1 > total0 else 0.0
    res = parse(done.stdout)
    res["rc"] = rc_lines(done.stderr)
    # what the other processes used: the processors' busy time less this program's own CPU time, over the wall time of the run
    mine = float(res.get("process", {}).get("cpu_s", 0.0))
    ncpu = os.cpu_count() or 1
    others = max(0.0, busy * ncpu / 100.0 * wall - mine) / (ncpu * wall) * 100.0 if wall > 0 else 0.0
    machine = {"seconds": round(wall, 1), "system_cpu_busy_pct": round(busy, 1), "others_cpu_busy_pct": round(others, 1),
               "available_gib": round(available_gib(), 2), "foreign_before": procs0, "foreign_after": procs1,
               "holder_busy_after": bool(GUARD and GUARD.holder_busy())}   # true: a timed run began beside this one
    return res, machine


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except Aborted as e:
        print(f"stopped by the machine rules: {e}", file=sys.stderr)
        return 75


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work", type=Path, required=True, help="folder for the packs, objects and programs")
    ap.add_argument("--out", type=Path, help="write the results here (JSON)")
    ap.add_argument("--jobs", type=int, default=2, help="compiles at a time")
    ap.add_argument("--priority", choices=("below", "idle"), default="below", help="the compiles' priority class")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-timed", action="store_true", help="skip the build with timers")
    ap.add_argument("--ns", default="340,500,1000,1500,2000,2500,3000,4000")
    ap.add_argument("--split-ns", default="340,1000,2000,3000", help="sizes for the runs with timers")
    ap.add_argument("--patterns", default="a,b")
    ap.add_argument("--count-patterns", default="", help="also: runs with the replay's counters on (they are never in a timed "
                    "run) for these patterns, a to e")
    ap.add_argument("--count-n", default="60", help="draws a frame of those runs (pattern e wants a multiple of 10)")
    ap.add_argument("--kinds", default="import,mirror")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--evict-mb", type=int, default=64)
    ap.add_argument("--variant", default="6.25ps", help="variant share of the headline runs: 0, 6.25ps, 100ps, 100both")
    ap.add_argument("--hot", action="store_true", help="also: the matrix without the cache eviction between frames")
    ap.add_argument("--sensitivity", action="store_true", help="also: variant shares, pattern c (vertex state rebound) and "
                    "pattern d (a storage buffer written by each draw)")
    ap.add_argument("--busy-file", type=Path, default=default_busy_file(),
                    help="a busy-marker file (a line 'BUSY: none' or 'BUSY: <text naming a TIMED or HEAVY job>'); default: the "
                    "file named by the environment variable " + BUSY_ENV + ". Without one of them there is NO guard (nothing "
                    "waits for the machine), so leave it out only on a machine of your own")
    ap.add_argument("--quiet", action="store_true", help="each run waits for a machine with no foreign compiler, build or game "
                    "and the processors under 12%% busy (needs the busy-marker file: it is part of the guard)")
    ap.add_argument("--reps", type=int, default=1, help="repetitions of the headline, hot and split runs, interleaved")
    ap.add_argument("--resume", action="store_true", help="skip the runs already in --out")
    ap.add_argument("--no-guard", action="store_true", help="ignore the busy marker and the memory rules (never on a shared machine)")
    ap.add_argument("--wait-minutes", type=float, default=85.0, help="how long a build or a run waits for the machine")
    a = ap.parse_args(argv)
    global PRIORITY, GUARD
    PRIORITY = IDLE if a.priority == "idle" else BELOW_NORMAL
    if a.quiet and (a.no_guard or not a.busy_file):
        ap.error("--quiet needs the guard: give --busy-file (or set " + BUSY_ENV + ") and leave --no-guard off")
    GUARD = None if a.no_guard or not a.busy_file else Guard(a.busy_file, wait_minutes=a.wait_minutes, require_quiet=a.quiet)

    import shutil
    cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
    include = build_game.vulkan_include()
    glslang = build_game.find_glslang(include) if include else None
    if not cc or not include or not glslang:
        print("needs clang (OOT2_CC or PATH) and a Vulkan SDK with Volk and glslang (VULKAN_SDK)", file=sys.stderr)
        return 2
    work = a.work
    work.mkdir(parents=True, exist_ok=True)
    clean = build(work, False, a.jobs, cc, include, glslang)
    timed = None if a.no_timed else build(work, True, a.jobs, cc, include, glslang)
    if a.build_only:
        return 0

    vs_keys = [key_of("bench-vs", i) for i in range(NPAIR)]
    ps_keys = [key_of("bench-ps", i) for i in range(NPAIR)]
    packs = work / "packs"
    packs.mkdir(exist_ok=True)
    manifest = packs / "keys.txt"
    manifest.write_text("".join(f"vs {i} {k}\n" for i, k in enumerate(vs_keys)) +
                        "".join(f"ps {i} {k}\n" for i, k in enumerate(ps_keys)), encoding="utf-8")
    shares = {"0": (0, 0), "6.25ps": (4, 0), "100ps": (NPAIR, 0), "100both": (NPAIR, NPAIR)}
    stores: dict[tuple[str, str], tuple[Path, Path, Path]] = {}

    def store(flavor: str, tag: str) -> tuple[Path, Path, Path]:
        """(base, variants, map) of a flavor ("plain", or "ssbo": the pixel stages write a storage buffer) and a variant share."""
        if (flavor, tag) not in stores:
            t = time.time()
            base = packs / f"base-{flavor}.bin"
            fillers = make_base(base, vs_keys, ps_keys, ssbo=flavor == "ssbo")
            p, v = shares[tag]
            pack, vmap = make_variants(packs, f"{flavor}-{tag}", vs_keys, ps_keys, fillers, p, v, ssbo=flavor == "ssbo")
            stores[(flavor, tag)] = (base, pack, vmap)
            print(f"packs {flavor}/{tag} ready in {time.time() - t:.0f} s ({base.stat().st_size >> 20} MiB base)",
                  file=sys.stderr)
        return stores[(flavor, tag)]

    results: dict = {"holder": busy_line(a.busy_file), "runs": []}
    if a.resume and a.out and a.out.exists():
        results = json.loads(a.out.read_text(encoding="utf-8"))
    done = {(r["label"], tuple(r["args"]), r.get("rep", 0)) for r in results["runs"]}

    def common(tag: str, extra: list[str], flavor: str = "plain") -> list[str]:
        base, pack, vmap = store(flavor, tag)
        return ["--base", str(base), "--variants", str(pack), "--map", str(vmap), "--keys", str(manifest),
                "--frames", str(a.frames), "--warmup", str(a.warmup), *extra]

    jobs: list[tuple[int, str, Path, list[str], int, dict]] = []   # group, label, program, arguments, repetitions, environment
    group = [0]

    def add(label: str, exe: Path, args: list[str], reps: int = 1, env: dict | None = None) -> None:
        jobs.append((group[0], label, exe, args, reps, env or {}))

    def record(label: str, exe: Path, args: list[str], rep: int, env: dict) -> None:
        res, machine = run_exe(exe, args, env=env)
        results["runs"].append({"label": label, "rep": rep, "exe": exe.name, "args": args, "machine": machine, **res})
        for b in res["bench"]:
            print(f"{label:14} r{rep} {b['pattern']} n={b['n']:>5} {b['kind']:6} {float(b['wall_ns_draw']):8.0f} ns/draw wall "
                  f"{float(b['cpu_ns_draw']):8.0f} cpu  (system {machine['system_cpu_busy_pct']}%, others "
                  f"{machine['others_cpu_busy_pct']}%)", file=sys.stderr, flush=True)
        results["holder_end"] = busy_line(a.busy_file)
        if a.out:
            a.out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    ev = ["--evict-mb", str(a.evict_mb)]
    for pattern in a.patterns.split(","):
        for kind in a.kinds.split(","):
            add("headline", clean, common(a.variant, ["--pattern", pattern, "--kind", kind, "--n", a.ns, *ev]), a.reps)
    if timed:
        for pattern in a.patterns.split(","):
            for kind in a.kinds.split(","):
                add("split", timed, common(a.variant, ["--pattern", pattern, "--kind", kind, "--n", a.split_ns, *ev]), a.reps)
    for pattern in filter(None, a.count_patterns.split(",")):
        for kind in a.kinds.split(","):
            add("counts", clean, common(a.variant, ["--pattern", pattern, "--kind", kind, "--n", a.count_n, *ev]), 1,
                {COUNTERS_ENV: "1"})
    group[0] = 1
    if a.hot:   # the same matrix without the cache eviction between frames: every frame starts where the last one ended
        for pattern in a.patterns.split(","):
            for kind in a.kinds.split(","):
                add("hot", clean, common(a.variant, ["--pattern", pattern, "--kind", kind, "--n", a.ns, "--evict-mb", "0"]),
                    min(a.reps, 3))
    group[0] = 2
    sens = min(a.reps, 3)
    if a.sensitivity:
        for tag in shares:
            add(f"variant-{tag}", clean, common(tag, ["--pattern", "b", "--kind", "mirror", "--n", "340,1000,2000,3000", *ev]), sens)
            if timed:
                add(f"variant-{tag}-s", timed, common(tag, ["--pattern", "b", "--kind", "mirror", "--n", "2000", *ev]), sens)
        for kind in a.kinds.split(","):
            add("pattern-c", clean, common(a.variant, ["--pattern", "c", "--kind", kind, "--n", "340,1000,2000,3000", *ev]), sens)
            if timed:
                add("pattern-c-s", timed, common(a.variant, ["--pattern", "c", "--kind", kind, "--n", "2000", *ev]), sens)
            add("pattern-d", clean, common(a.variant, ["--pattern", "d", "--kind", kind, "--n", "340,1000,2000,3000", *ev],
                                           "ssbo"), sens)
            if timed:
                add("pattern-d-s", timed, common(a.variant, ["--pattern", "d", "--kind", kind, "--n", "2000", *ev], "ssbo"), sens)
    # By group (the headline and the split first, then the hot runs, then the sensitivities), and in each group interleaved:
    # every program and pattern once, then again, so that a noisy stretch of the machine spoils one repetition of each and
    # not all repetitions of one
    for g in sorted({j[0] for j in jobs}):
        mine = [j for j in jobs if j[0] == g]
        for rep in range(max(j[4] for j in mine)):
            for _, label, exe, args, reps, env in mine:
                if rep < reps and (label, tuple(args), rep) not in done:
                    record(label, exe, args, rep, env)
    results["holder_end"] = busy_line(a.busy_file)
    if a.out:
        a.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


# ---- the tests -------------------------------------------------------------------------------------------------------
# No device and no timing is judged. The smoke case builds the program, replays a few frames of patterns a and b and checks
# that every draw it was given was counted and none was missed, skipped or lost. The default build is checked free of the
# instrumentation. The timed passes stay a hand-run (the module docstring).

INSTRUMENTED_FILES = ("gpu/vk_replay.c", "gpu/vk_pipeline.c", "gpu/vk_replay_dev.c", "nvn/nvn_ledger.c")
# Names that exist only when the timers or the hooks are compiled in.
INSTRUMENTATION = ("bench_acc", "bench_n", "bt_now", "rdtsc", "vkp_bench_prime", "vkp_bench_open", "vkp_bench_primed",
                   "find_ready", "bank_body", "add_row_body", "draw_prepare_body")


def instrumentation_in(cc: str, include: Path, gen: Path, rel: str, defines: list[str]) -> list[str]:
    """What of INSTRUMENTATION the preprocessed text and the object of RUNTIME/rel hold when it is compiled with defines."""
    base = [cc, "-std=gnu11", "-O2", "-march=x86-64-v3", "-fno-strict-aliasing", *defines,
            "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "nvn"), "-I", str(gen),
            "-I", str(include), "-I", str(include / "Volk"), str(RUNTIME / rel)]
    found: set[str] = set()
    pre = subprocess.run([*base, "-E", "-P"], capture_output=True, text=True)
    if pre.returncode:
        raise RuntimeError(f"preprocessing {rel} failed:\n{pre.stderr}")
    found.update(w for w in INSTRUMENTATION if re.search(rf"\b{w}\b", pre.stdout))
    with tempfile.TemporaryDirectory(prefix="oot2-benchfree-", ignore_cleanup_errors=True) as tmp:
        obj = Path(tmp) / "x.o"
        done = subprocess.run([*base, "-c", "-o", str(obj)], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"compiling {rel} failed:\n{done.stderr}")
        raw = obj.read_bytes()
        found.update(w for w in INSTRUMENTATION if w.encode() in raw)
    return sorted(found)


class DriverTest(unittest.TestCase):
    """The driver's own behaviour that needs no toolchain."""

    def test_help_prints_under_this_python_and_says_what_no_busy_file_means(self) -> None:
        # a "%" in a help string made argparse refuse to build the parser under Python 3.14
        done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--help"], capture_output=True, text=True,
                              env={k: v for k, v in os.environ.items() if k != BUSY_ENV})
        self.assertEqual(done.returncode, 0, done.stderr)
        text = " ".join(done.stdout.split())
        self.assertIn("12% busy", text)
        self.assertIn("NO guard", text)
        self.assertIn(BUSY_ENV, text)

    def test_the_busy_file_comes_from_the_environment_and_has_no_default(self) -> None:
        saved = os.environ.pop(BUSY_ENV, None)
        try:
            self.assertIsNone(default_busy_file())
            os.environ[BUSY_ENV] = "somewhere/busy.txt"
            self.assertEqual(default_busy_file(), Path("somewhere/busy.txt"))
        finally:
            os.environ.pop(BUSY_ENV, None)
            if saved is not None:
                os.environ[BUSY_ENV] = saved
        self.assertEqual(busy_line(None), "(no busy-marker file)")

    def test_the_link_flags_follow_the_operating_system(self) -> None:
        self.assertEqual(link_flags("nt"), ["-Wl,--disable-high-entropy-va"])
        self.assertEqual(link_flags("posix"), ["-pthread", "-ldl"])
        self.assertEqual(link_flags(), link_flags(os.name))

    def test_the_guard_stops_work_while_the_marker_names_a_timed_or_heavy_job(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-benchbusy-", ignore_cleanup_errors=True) as tmp:
            marker = Path(tmp) / "busy.txt"
            for line, busy in (("BUSY: none", False), ("BUSY: a job, TIMED, until 14:40", True),
                               ("BUSY: a job, HEAVY, 10 min", True), ("BUSY: a job, light", False),
                               ("BUSY: none, last was TIMED", False)):
                marker.write_text(f"intro\n{line}\nrest\n", encoding="utf-8")
                self.assertEqual(Guard(marker).holder_busy(), busy, line)
            marker.write_text("BUSY: a job, TIMED\n", encoding="utf-8")
            with self.assertRaises(Aborted):
                Guard(marker).check()


class DefaultBuildTest(unittest.TestCase):
    """Without OOT2_BENCH_TIMERS and OOT2_BENCH_HOOKS the touched files hold nothing of the instrumentation."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_runtime_gpu import clang_or_skip, vulkan_or_skip
        cls.cc = clang_or_skip()
        cls.include, _ = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-benchgen-", ignore_cleanup_errors=True)
        cls.gen = Path(cls.tmp.name)
        layout.write(layout.place({"main": MODULE_SIZE}), cls.gen)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_the_default_objects_hold_none_of_it(self) -> None:
        for rel in INSTRUMENTED_FILES:
            self.assertEqual(instrumentation_in(self.cc, self.include, self.gen, rel, []), [], rel)

    def test_counter_run_the_check_sees_the_instrumentation_when_it_is_compiled_in(self) -> None:
        for rel in INSTRUMENTED_FILES:
            seen = instrumentation_in(self.cc, self.include, self.gen, rel, ["-DOOT2_BENCH_TIMERS", "-DOOT2_BENCH_HOOKS"])
            self.assertNotEqual(seen, [], f"{rel} built with the timers shows nothing: the check above could not fail")


class SmokeTest(unittest.TestCase):
    """A few frames of patterns a and b through the driver, with and without the timers, and of a, b and e with the counters."""

    FRAMES, WARMUP, N = 4, 2, 64
    COUNT_N = 60   # a frame of the counted runs: pattern e repeats every ten draws

    @classmethod
    def setUpClass(cls) -> None:
        from test_runtime_gpu import clang_or_skip, vulkan_or_skip
        clang_or_skip()
        vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-benchsmoke-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name) / "results.json"
        argv = ["--work", str(Path(cls.tmp.name) / "work"), "--no-guard", "--jobs", str(os.cpu_count() or 2),
                "--ns", str(cls.N), "--split-ns", str(cls.N), "--patterns", "a,b", "--kinds", "import,mirror",
                "--count-patterns", "a,b,e", "--count-n", str(cls.COUNT_N),
                "--frames", str(cls.FRAMES), "--warmup", str(cls.WARMUP), "--out", str(out)]
        code = main(argv)
        if code != 0:
            raise RuntimeError(f"the driver exited {code}")
        cls.results = json.loads(out.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def lines(self, label: str) -> list[dict]:
        return [b for r in self.results["runs"] if r["label"] == label for b in r["bench"]]

    def test_every_draw_given_is_counted_and_none_is_missed_skipped_or_lost(self) -> None:
        for label in ("headline", "split"):
            got = self.lines(label)
            self.assertEqual(sorted((b["pattern"], b["kind"]) for b in got),
                             [("a", "import"), ("a", "mirror"), ("b", "import"), ("b", "mirror")], label)
            for b in got:
                what = f"{label} {b['pattern']} {b['kind']}"
                self.assertEqual(int(b["n"]), self.N, what)
                self.assertEqual(int(b["frames"]), self.FRAMES, what)
                self.assertEqual(int(b["draws"]), self.N * self.FRAMES, what)
                self.assertEqual((int(b["skipped"]), int(b["lost"]), int(b["misses"])), (0, 0, 0), what)
                self.assertGreater(int(b["primed"]), 0, what)   # the pipelines the frames asked for were entered
                self.assertGreater(float(b["wall_ns_draw"]), 0.0, what)

    def test_the_timed_build_reports_its_sections(self) -> None:
        runs = [r for r in self.results["runs"] if r["label"] == "split"]
        self.assertEqual(len(runs), 4)
        for r in runs:
            self.assertTrue(r["timer"], "no calibration line")
            names = {s["name"] for s in r["split"]}
            self.assertTrue(any(n.startswith("S3 ") for n in names), names)
            self.assertTrue(any(n.startswith("S5 ") for n in names), names)

    # ---- the counters (runtime/include/oot2/replay_count.h) against streams whose repeats are known by construction
    BANK_C0 = 0xA00       # VKN_BANK_BYTES: what a driver bank copy moves
    BANK_ROOTS = 2048     # the roots bank: 8 tables of 64 words

    def counted(self) -> dict[tuple[str, str], dict]:
        got = {}
        for r in self.results["runs"]:
            if r["label"] == "counts":
                b = r["bench"][0]
                got[(b["pattern"], b["kind"])] = r
        return got

    def expected(self, pattern: str) -> dict[str, int]:
        """What each counter must read over the measured frames of pattern a, b or e at COUNT_N draws a frame, derived from the
        stream (tests/runtime/vk_replay_cost.c build_frame) and the replay's rules, not from a run."""
        n, f = self.COUNT_N, self.FRAMES
        e: dict[str, int] = {
            "sub_n": f, "sub_work": n * f, "sub_h16_63": f, "draw_classic": n * f, "draw_mesh": 0, "dispatch": 0,
            "d_first": f, "d_cmp": (n - 1) * f, "d_vin_n": n * f, "d_vin_state_same": (n - 1) * f,   # the attributes and streams never change
            "d_dyn_n": n * f, "d_dyn_same": (n - 1) * f,
            "content_n": 2 * n * f, "content_miss": 0,
            # five uniform ranges a draw (the vertex stage's three, the pixel stage's two), every one bound and resolved
            "uni_calls": 5 * n * f, "uni_zero": 0, "resolve_uniform": 5 * n * f, "resolve_vertex": 2 * n * f,
            "resolve_storage": 0, "resolve_index": 0, "resolve_indirect": 0,
            "written_scan_steps": 0, "wrote_calls": 0, "indirect_scan_steps": 0,
            "bank_c0_calls": 2 * n * f, "bank_c2_calls": 2 * n * f, "bank_roots_calls": 2 * n * f,
            "bank_c0_equal": (2 * n - 2) * f, "bank_roots_equal": (2 * n - 1) * f,
            "bank_c0_copied_bytes": 2 * f * self.BANK_C0, "bank_roots_copied_bytes": f * self.BANK_ROOTS,
            "digest_vs_calls": n * f, "digest_ps_calls": n * f, "digest_other_calls": 0, "digest_vs_recompute": f,
            "digest_vs_valid_hit": (n - 1) * f, "digest_vs_inputs_same": f, "digest_vs_value_same": f,
            "digest_other_recompute": 0, "digest_other_valid_hit": 0,
        }
        if pattern == "a":
            e.update(d_vin_layout_same=(n - 1) * f, d_vin_full_same=(n - 1) * f,
                     d_vs_same=(n - 1) * f, d_ps_same=(n - 1) * f, d_pair_same=(n - 1) * f, d_vsdig_same=(n - 1) * f,
                     d_psdig_same=(n - 1) * f, d_key_same=(n - 1) * f, d_pipe_same=(n - 1) * f, d_first_pair_xsub=f,
                     d_tuple_new=f, d_tuple_lru16=(n - 1) * f, content_variant=0,
                     bank_c2_equal=(2 * n - 2) * f, bank_c2_copied_bytes=2 * f * self.BANK_C0,
                     tex_binds=f, tex_same_handle=f, tex_same_type=f, tex_bind_ops=f, tex_bind_ops_same_type=f,
                     digest_ps_recompute=f, digest_ps_valid_hit=(n - 1) * f, digest_ps_inputs_same=f,
                     digest_ps_value_same=f)
        elif pattern == "b":
            # every draw the next of 64 pairs and a new texture and range; 60 draws never come round to a pair again
            e.update(d_vin_layout_same=0, d_vin_full_same=0,   # a different vertex stage's binds each draw
                     d_vs_same=0, d_ps_same=0, d_pair_same=0, d_vsdig_same=(n - 1) * f, d_psdig_same=(n - 1) * f,
                     d_key_same=0, d_pipe_same=0, d_first_pair_xsub=0, d_tuple_new=n * f, d_tuple_lru16=0,
                     content_variant=4 * f,   # pairs 0 to 3 have a pixel stage that maps to a variant
                     bank_c2_equal=(n - 1) * f, bank_c2_copied_bytes=(n + 1) * f * self.BANK_C0,
                     tex_binds=(n + 1) * f, tex_same_handle=f, tex_same_type=(n + 1) * f, tex_bind_ops=(n + 1) * f,
                     tex_bind_ops_same_type=(n + 1) * f,
                     digest_ps_recompute=n * f, digest_ps_valid_hit=0, digest_ps_inputs_same=n * f,
                     digest_ps_value_same=n * f)
        else:
            # ten draws repeat: pairs 0 0 1 0 1 1 2 2 2 3 and textures 5 5 6 5 6 6 7 7 7 8; of the 9 steps from a draw to the
            # next in a period 4 repeat the pair (and the texture), the step from a period's last draw to the next one's first
            # does not, and the first draw of a frame has no draw before it
            same = 24 * f
            e.update(d_vin_layout_same=same, d_vin_full_same=same,
                     d_vs_same=same, d_ps_same=same, d_pair_same=same, d_vsdig_same=(n - 1) * f, d_psdig_same=(n - 1) * f,
                     d_key_same=same, d_pipe_same=same, d_first_pair_xsub=0, d_tuple_new=4 * f, d_tuple_lru16=(n - 4) * f,
                     content_variant=n * f,
                     bank_c2_equal=(n - 1) * f + same, bank_c2_copied_bytes=(n + 1 - 24) * f * self.BANK_C0,
                     tex_binds=(n + 1) * f, tex_same_handle=same, tex_same_type=(n + 1) * f, tex_bind_ops=(n + 1) * f,
                     tex_bind_ops_same_type=(n + 1) * f,
                     digest_ps_recompute=n * f, digest_ps_valid_hit=0, digest_ps_inputs_same=n * f,
                     digest_ps_value_same=n * f)
        return e

    def test_the_counters_read_what_the_known_streams_say(self) -> None:
        got = self.counted()
        self.assertEqual(sorted(got), [(p, k) for p in "abe" for k in ("import", "mirror")])
        for (pattern, kind), run in got.items():
            counts = run["counts"]
            wrong = {key: (counts.get(key), value) for key, value in self.expected(pattern).items()
                     if counts.get(key) != value}
            self.assertEqual(wrong, {}, f"pattern {pattern} {kind}: counter -> (read, derived)")

    def test_the_counters_agree_with_what_the_replay_already_counted(self) -> None:
        for (pattern, kind), run in self.counted().items():
            c, bench = run["counts"], run["bench"][0]
            what = f"{pattern} {kind}"
            self.assertEqual(c["draw_classic"] + c["draw_mesh"], int(bench["draws"]), what)   # VkrStats.draws, by another count
            self.assertEqual(c["dispatch"], c["vkr_dispatches"], what)
            self.assertEqual(c["uni_snapshot"] + c["uni_live"] + c["uni_zero"], c["uni_calls"], what)
            self.assertEqual(c["snap_calls"], c["uni_snapshot"], what)
            self.assertEqual(c["snap_calls"] - c["snap_hit"], c["vkr_snapshots"], what)   # a miss is a copy
            self.assertEqual(c["uni_live"], c["vkr_live_uniforms"], what)
            if kind == "mirror":
                self.assertEqual(c["uni_snapshot"], 0, what)
            else:
                self.assertGreater(c["uni_snapshot"], 0, what)
            for use in ("c0", "c2", "roots"):
                miss = c[f"bank_{use}_calls"] - c[f"bank_{use}_equal"]
                self.assertEqual(c[f"bank_{use}_copied_bytes"], miss * (self.BANK_ROOTS if use == "roots" else self.BANK_C0),
                                 f"{what} {use}")
            for stage in ("vs", "ps"):   # a digest call is a hit or a recompute; a recompute from equal inputs has an equal value
                self.assertEqual(c[f"digest_{stage}_calls"], c[f"digest_{stage}_valid_hit"] + c[f"digest_{stage}_recompute"], what)
                self.assertEqual(c[f"digest_{stage}_inputs_same"], c[f"digest_{stage}_value_same"], what)
            self.assertLessEqual(c["tex_same_type"], c["tex_binds"], what)
            self.assertEqual(c["content_variant"] + c["content_miss"] <= c["content_n"], True, what)

    def test_the_exit_line_has_every_group_and_totals_that_agree_with_the_replay_s_statistics(self) -> None:
        for (pattern, kind), run in self.counted().items():
            lines = [x for x in run["rc"] if x["kind"] == "exit"]
            what = f"{pattern} {kind}"
            self.assertEqual({x["group"] for x in lines},
                             {"sub", "draw", "bank", "uniform", "ledger", "time", "threads", "vkr", "backing", "silent",
                              "frametime", "completion", "stages"}, what)
            flat: dict[str, int] = {}
            for x in lines:
                for k, v in x.items():
                    if k not in ("kind", "group"):
                        self.assertTrue(v.isdigit(), f"{what}: {k}={v}")   # a script can parse every value
                        flat[k] = int(v)
            total = self.FRAMES + self.WARMUP
            self.assertEqual((flat["sub_n"], flat["vkr_frames"]), (total, total), what)
            self.assertEqual(flat["draw_classic"] + flat["draw_mesh"], flat["vkr_draws"], what)
            self.assertEqual(flat["vkr_draws"], self.COUNT_N * total, what)
            self.assertEqual((flat["vkr_lost"], flat["vkr_skipped"]), (0, 0), what)
            self.assertGreater(flat["backing_resolves"], 0, what)
            self.assertEqual(flat["rc_threads"], 1, what)   # the replay thread alone counted

    def test_a_timed_or_plain_run_prints_no_counters(self) -> None:
        for r in self.results["runs"]:
            if r["label"] != "counts":
                self.assertEqual(r["rc"], [], r["label"])
                self.assertEqual(r["counts"], {}, r["label"])


if __name__ == "__main__":
    sys.exit(main())
