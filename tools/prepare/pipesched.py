# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""What the pipeline step builds, in what order, on how many workers, and how an interrupted run resumes (architecture
section 6.1, step 1; the pipeline cost and compute build time studies). Nothing here needs a GPU: the builder is a program that speaks a line
protocol (tools/pipebuild/pipebuild.c), and the tests of this module run it as a stand-in script.

The work list. Every job of the shader cache pack that translated becomes one item: the compute pipeline of a compute
module (`cs`), the pre-rasterization library of a vertex module as translated (`vs`, unpadded: the padded ones are known
only from play and come from the ledger), the fragment library of a pixel module (`ps`). Task and mesh modules (`ts`,
`ms`) wait for the mesh draws work and are counted as not built, so every total is a lower bound until then.

The order. Compute first, then graphics; inside each group the longest first, by the size of the module (the pipeline cost study:
build time ranks with SPIR-V size, 0.98 for vertex and 0.99 for pixel modules), so the last items to run are the
short ones and the pool is never left waiting on one long build. Ties break on the key, so the order is the same in
every process.

The workers. A build holds 0.3 to 2.8 GB of host memory (the compute build time study) and the driver compiles on the calling thread,
so the builder runs as one process per worker, each with its own device and its own empty pipeline cache. How many run
follows the free memory (a share of it, not the CPU count alone); each item reserves what a model of its size says it
holds, the largest ones run alone, and compute builds are held to six at a time (twelve threads were no faster than six
for raw and plain programs). A worker whose process dies or stops answering is replaced, and the item is tried again
alone with a longer time limit; an item the runtime refuses by name (a fatal: a module that does not fit its slots, a
capability the device lacks) is a failure that is recorded and not retried.

The resume. A worker writes the cache it holds to a segment file every flush interval and starts again from an empty
cache, so a segment holds only what was built since the last one and each byte is written once. The parent records the
items of a segment in the progress file only after the worker says the segment is saved, so the record never leads the
cache: a run that stops loses at most one interval per worker. The next run with the same key skips every recorded item
whose segment is still there. When everything is built, the builder merges the segments into one cache file.

A seeded builder (tools/prepare/pipeseed.py) is not flushed on a timer, because a flush hands it an empty cache and the
seed would stop serving. It saves when it is recycled and at the end, and every one of its segments holds the whole seed
beside what it built, so a stopped seeded run loses what its workers built since their last save, which can be all of
what they built since they started. Segments saved before the stop stay, on the same key, and their items are skipped.
"""

from __future__ import annotations

import collections
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

TOOLS = Path(__file__).resolve().parent.parent
if str(TOOLS / "shaders") not in sys.path:
    sys.path.insert(0, str(TOOLS / "shaders"))

import pool  # noqa: E402  (tools/shaders/pool.py: working_set, available_ram)

MiB = 1 << 20
GiB = 1 << 30

KINDS = ("cs", "vs", "ps")             # what is built; the build order is compute first, then the rest
NOT_BUILT = ("ms", "ts")               # task and mesh: the mesh draws ticket builds them
KIND_NAMES = {"cs": "compute", "vs": "vertex", "ps": "pixel", "ms": "mesh", "ts": "task"}

# The memory an item reserves: a floor for the process (device, driver, the mapped pack) plus a share of the module's
# size, capped at the largest build measured (the compute build time study: raw-access builds 0.3 GB, tensor builds 0.3 to 2.6
# GB, a 12 MB plain build 2.8 GB). A model that overestimates costs workers, one that underestimates costs paging; the
# step records each worker's measured peak so the constants can be read against a real run.
# Measured on the 3060 Ti (the pipeline step's 28-program sample, compute programs from 28 KB to 11.6 MB, working set of the one
# worker running): 816 KB 0.35 GB, 1.0 MB 0.72, 1.35 MB 0.69, 2.6 MB 0.99, 11.6 MB 2.91.
BASE_BYTES = 320 * MiB
PER_MODULE_BYTE = 400
CAP_BYTES = 3 * GiB
RECYCLE_BYTES = 1 * GiB                # a builder holding this much after an item is replaced (the driver keeps the
                                       # memory of its largest build for the life of the process)
MONITOR_SECONDS = 2.0                  # between two samples of the workers' working sets and the free memory
ALONE_BYTES = 1 * GiB                  # an item that reserves this much runs with nothing else running
MEM_SHARE = 0.8                        # of the free memory the workers may reserve
FLOOR_BYTES = 1 * GiB                  # free memory under which no new item starts
WORKER_BYTES = 512 * MiB               # for counting workers from the free memory
MAX_WORKERS = 12
COMPUTE_AT_ONCE = 6                    # the compute build time study
FLUSH_SECONDS = 60.0
ITEM_TIMEOUT = 900.0                   # seconds one item may take (a tensor program took 303 s while 12 threads built)
RETRY_TIMEOUT = 3600.0                 # an item that got no answer, tried again alone
START_TIMEOUT = 300.0                  # a builder's start: device, pack, empty cache
FATAL_EXITS = (70, 71)                 # the runtime's exit codes for a named fatal (the runtime boundary study)
LOG_MARK = "=== builder start ==="
# A named fatal that says the machine, not the module, failed: memory ran out, or the device was lost. It is a crash
# (the item is tried again alone), never a refusal recorded for good.
TRANSIENT = ("VK_ERROR_OUT_OF_HOST_MEMORY", "VK_ERROR_OUT_OF_DEVICE_MEMORY", "VK_ERROR_TOO_MANY_OBJECTS",
             "VK_ERROR_DEVICE_LOST", "VK_ERROR_INITIALIZATION_FAILED", "VK_ERROR_MEMORY_MAP_FAILED")


def refused(fatal: str) -> bool:
    """True for a named fatal that is about the module (a refusal), false for one about the machine."""
    return bool(fatal) and not any(t in fatal for t in TRANSIENT)


class SchedError(Exception):
    """The step cannot go on: the builder cannot start, or a segment cannot be written."""


@dataclass(frozen=True)
class Item:
    key: str
    kind: str                          # "cs", "vs" or "ps"
    module_bytes: int


@dataclass
class WorkList:
    items: list[Item]
    not_built: dict[str, int]          # "ms" and "ts" job counts
    untranslated: int = 0              # jobs that are failure records in the pack
    unstaged: int = 0                  # jobs whose stage the split does not know
    missing: list[str] = field(default_factory=list)   # keys asked for that the pack does not hold


def work_list(rows: Iterable[tuple[str, int, bool]], stages: dict[str, str], only: Iterable[str] | None = None) -> WorkList:
    """rows are the pack's (key, module bytes, failed); stages maps a job key to vs, ps, cs, ms or ts (the split's
    programs.tsv). `only` limits the list to those keys."""
    wanted = set(only) if only is not None else None
    items: list[Item] = []
    out = WorkList(items, {k: 0 for k in NOT_BUILT})
    seen: set[str] = set()
    for key, size, failed in rows:
        seen.add(key)
        if wanted is not None and key not in wanted:
            continue
        stage = stages.get(key)
        if failed:
            out.untranslated += 1
        elif stage in KINDS:
            items.append(Item(key, stage, size))
        elif stage in NOT_BUILT:
            out.not_built[stage] += 1
        else:
            out.unstaged += 1
    if wanted is not None:
        out.missing = sorted(wanted - seen)
    return out


def order(items: Iterable[Item]) -> list[Item]:
    """Compute first, then graphics; the largest module first inside each; the key last, so the order never changes
    between processes."""
    return sorted(items, key=lambda i: (0 if i.kind == "cs" else 1, -i.module_bytes, i.key))


def expected_bytes(item: Item) -> int:
    return min(CAP_BYTES, BASE_BYTES + item.module_bytes * PER_MODULE_BYTE)


def runs_alone(item: Item) -> bool:
    return expected_bytes(item) >= ALONE_BYTES


def size_workers(free: int, cpus: int, given: int | None = None, seed_bytes: int = 0) -> tuple[int, int, str]:
    """Workers, the memory the pool may reserve, and why: the CPU count (at most MAX_WORKERS) capped by the free
    memory at WORKER_BYTES a worker (and the seed each one loads, when the cache is seeded), unless the player gave a
    number."""
    budget = max(WORKER_BYTES, int(free * MEM_SHARE)) if free else 4 * GiB
    if given:
        return given, budget, "--pipelines-jobs"
    cpus = min(cpus, MAX_WORKERS)
    if not free:
        return cpus, budget, f"{cpus} CPUs, free memory unknown"
    by_mem = max(1, int(free * MEM_SHARE) // (WORKER_BYTES + seed_bytes))
    return min(cpus, by_mem), budget, f"{cpus} CPUs, {free / GiB:.1f} GiB free memory allows {by_mem}"


class Governor:
    """Decides when an item may start: the reserved memory stays inside the budget, compute is held to a few at a
    time, an item that runs alone waits for the others to finish and holds new ones back meanwhile, and nothing starts
    while the machine's free memory is under the floor. The first item is always admitted, so a budget smaller than one
    item cannot stop the step."""

    def __init__(self, budget: int, available: Callable[[], int] = pool.available_ram, floor: int = FLOOR_BYTES,
                 compute_at_once: int = COMPUTE_AT_ONCE):
        self.budget, self.available, self.floor, self.compute_at_once = budget, available, floor, compute_at_once
        self.cv = threading.Condition()
        self.reserved = 0
        self.running = 0
        self.running_cs = 0
        self.alone_running = False
        self.alone_waiting = 0
        self.stopped = False
        self.held_back_s = 0.0
        self.peak_reserved = 0
        self.peak_running = 0

    def stop(self) -> None:
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def _admit(self, item: Item, need: int, alone: bool) -> bool:
        if self.alone_running:
            return False
        if alone:
            return self.running == 0
        if self.alone_waiting:
            return False
        if self.running == 0:
            return True
        if item.kind == "cs" and self.running_cs >= self.compute_at_once:
            return False
        if self.reserved + need > self.budget:
            return False
        free = self.available()
        return not free or free >= self.floor

    def acquire(self, item: Item) -> int:
        """Block until the item may start; the number of items running, this one included."""
        need, alone = expected_bytes(item), runs_alone(item)
        with self.cv:
            t0 = None
            if alone:
                self.alone_waiting += 1
            try:
                while not self._admit(item, need, alone):
                    if self.stopped:
                        raise SchedError("stopped")
                    t0 = t0 or time.monotonic()
                    self.cv.wait(timeout=1.0)
            finally:
                if alone:
                    self.alone_waiting -= 1
            if t0 is not None:
                self.held_back_s += time.monotonic() - t0
            self.running += 1
            self.running_cs += item.kind == "cs"
            self.reserved += need
            self.alone_running = self.alone_running or alone
            self.peak_reserved = max(self.peak_reserved, self.reserved)
            self.peak_running = max(self.peak_running, self.running)
            return self.running

    def release(self, item: Item) -> None:
        with self.cv:
            self.running -= 1
            self.running_cs -= item.kind == "cs"
            self.reserved -= expected_bytes(item)
            if runs_alone(item):
                self.alone_running = False
            self.cv.notify_all()


@dataclass
class Result:
    key: str
    kind: str
    status: str                        # ok, fail (a named refusal, not retried), crash, timeout
    ns: int = 0                        # the driver's time inside the builder
    hit: int = 0                       # the application cache served it
    module_bytes: int = 0
    wall_ms: float = 0.0               # request to reply, as the parent saw it
    concurrency: int = 1               # items running when it started, this one included
    peak_ws: int = 0                   # the worker's working set at the largest sample while it ran
    started: float = 0.0               # seconds since the pool started
    message: str = ""
    segment: str = ""
    dead: bool = True                  # the builder process is gone (False: it refused one command and goes on)


def parse_reply(line: str) -> tuple[str, list[str]]:
    """The first word of a reply line and its other words (the last field of a fail line keeps its spaces)."""
    words = line.strip().split(" ", 5)
    return (words[0], words[1:]) if words and words[0] else ("", [])


class Child:
    """One builder process: a command line on its stdin, one reply line on its stdout, stderr appended to a log."""

    def __init__(self, command: list[str], log: Path, env: dict | None = None):
        self.log_path = log
        self.log = open(log, "ab")
        # One log per worker slot, appended to by each builder that takes the slot: the marker tells where this one's
        # lines start, so that a crash is never reported with the FATAL line of the builder before it.
        self.log.write(f"\n{LOG_MARK}\n".encode())
        self.log.flush()
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     encoding="utf-8", bufsize=1, env=env)
        self.replies: queue.Queue[str | None] = queue.Queue()
        self.ws = 0
        self.ws_peak = 0
        self.base_ws = 0                # the working set right after ready, for a builder that loaded a seed
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def read(self, timeout: float) -> str | None:
        """The next line, None when the process is gone, "" on a timeout."""
        try:
            return self.replies.get(timeout=timeout)
        except queue.Empty:
            return ""

    def ask(self, request: str, timeout: float) -> str | None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(request + "\n")
            self.proc.stdin.flush()
        except OSError:
            return None
        return self.read(timeout)

    def exit_code(self) -> int | None:
        try:
            return self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return None

    def log_tail(self, n: int = 4096) -> str:
        """The end of this builder's own lines of the log (nothing from an earlier builder of the slot)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - n))
                text = f.read().decode("utf-8", "replace")
        except OSError:
            return ""
        return text.rsplit(LOG_MARK, 1)[-1]

    def close(self, kill: bool = False) -> None:
        try:
            if kill:
                self.proc.kill()
            elif self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        self.log.close()


def fatal_line(text: str) -> str:
    """The last line of a log that names a fatal of the runtime, or ''."""
    for line in reversed(text.splitlines()):
        if "FATAL" in line:
            return line.strip()
    return ""


@dataclass
class PoolStats:
    peak_children_bytes: int = 0
    min_available_bytes: int = 0
    recycled: int = 0                  # builders replaced after a big build, to give its memory back
    restarts: int = 0
    lost: int = 0                      # built items a dead worker took with its unsaved cache, built again
    segments: int = 0
    segment_bytes: int = 0
    peak_running: int = 0
    held_back_s: float = 0.0
    validation_errors: int = 0         # the validation layer's, summed over the workers that quit normally
    validation_warnings: int = 0
    notes: list[str] = field(default_factory=list)


class Pool:
    """Workers over an ordered list of items. on_saved(results, segment, bytes) gets the items of every segment once it
    is on disk, on_failed(result) every item that ended without a pipeline (called with the pool's lock held, from a
    worker thread)."""

    def __init__(self, command: Callable[[int], list[str]], items: list[Item], workers: int, governor: Governor,
                 logs: Path, segments: Path, on_saved: Callable[[list[Result], str, int], None],
                 on_failed: Callable[[Result], None], next_segment: int = 1, flush_seconds: float | None = None,
                 timeout: float | None = None, start_timeout: float | None = None, env: dict | None = None,
                 on_result: Callable[[Result], None] | None = None, final_flush: bool = True,
                 expect: dict[str, str] | None = None, runtime: dict[str, str] | None = None,
                 forms: dict[str, tuple[str, ...]] | None = None):
        self.final_flush = final_flush   # a warm pass has nothing to save: its cache is the file it was seeded from
        self.expect = expect or {}       # fields of the builder's ready line (vendor, device, driver, cache_uuid)
        self.runtime = runtime or {}     # fields of it that name the runtime the builder was linked with (zero_workgroup)
        self.runtime_seen: dict[str, str] = {}
        # Fields of it that name a choice the runtime makes when it opens (hmma: the form of the tensor programs, which the
        # start-up probe chooses): each builder must say one of the values allowed, and every builder the same one.
        self.forms = forms or {}
        self.forms_seen: dict[str, str] = {}
        self.cache_seen: dict[str, int] = {}   # what each builder's ready line said of the cache it was given (cache=)
        self.seeded_seen: list[int] = []       # what the builders that said "loaded" said they hold (seeded=)
        self.flush_lock = threading.Lock()     # one flush at a time: each copies the whole cache beside the driver's
        self.command, self.workers, self.gov = command, workers, governor
        self.logs, self.segments = Path(logs), Path(segments)
        self.on_saved, self.on_failed, self.on_result = on_saved, on_failed, on_result
        # The limits are read when the pool is made, not when this module is, so that a test can shorten them.
        self.flush_seconds = FLUSH_SECONDS if flush_seconds is None else flush_seconds
        self.timeout = ITEM_TIMEOUT if timeout is None else timeout
        self.start_timeout = START_TIMEOUT if start_timeout is None else start_timeout
        self.recycle_bytes = RECYCLE_BYTES
        self.env = env
        self.todo: collections.deque[Item] = collections.deque(items)
        self.retry: list[tuple[Item, str]] = []      # (item, why) for the second pass, alone
        self.lock = threading.Lock()
        self.next_segment = next_segment
        self.stats = PoolStats()
        self.error: Exception | None = None
        self.children: dict[int, Child] = {}
        self.t0 = time.monotonic()
        self.stop_monitor = threading.Event()

    # -- the monitor: working sets and free memory, every two seconds
    def _monitor(self) -> None:
        while not self.stop_monitor.wait(MONITOR_SECONDS):
            with self.lock:
                children = list(self.children.values())
            total = 0
            for c in children:
                c.ws = pool.working_set(c.proc)
                c.ws_peak = max(c.ws_peak, c.ws)
                total += c.ws
            free = pool.available_ram()
            with self.lock:
                self.stats.peak_children_bytes = max(self.stats.peak_children_bytes, total)
                if free:
                    low = self.stats.min_available_bytes
                    self.stats.min_available_bytes = free if not low else min(low, free)

    def _take(self) -> Item | None:
        with self.lock:
            return self.todo.popleft() if self.todo and not self.error else None

    def _requeue(self, items: list[Item]) -> None:
        with self.lock:
            self.todo.extendleft(reversed(items))
            self.stats.lost += len(items)

    def _start(self, wid: int, attempt: int) -> Child:
        child = Child(self.command(wid), self.logs / f"worker-{wid}.log", self.env)
        line = child.read(self.start_timeout)
        word, _ = parse_reply(line or "")
        if word != "ready":
            code = child.exit_code()
            tail = child.log_tail()
            child.close(kill=True)
            why = fatal_line(tail) or tail.strip()[-400:] or "no output"
            raise SchedError(f"the pipeline builder did not start (exit {code}): {why}. Its log is {child.log_path}.")
        got = dict(w.split("=", 1) for w in line.split() if "=" in w)
        wrong = {k: got.get(k, "") for k, v in self.expect.items() if got.get(k, "").lower() != v.lower()}
        if wrong:
            # The device index came from a probe that ran with another loader environment: a builder on another GPU
            # would write a cache for it under this device's key.
            child.close(kill=True)
            raise SchedError("the pipeline builder opened another device than the one prepare chose "
                             f"({', '.join(f'{k} {v} for {self.expect[k]}' for k, v in wrong.items())}). Choose the "
                             "device with OOT2_VK_DEVICE, or disable the implicit Vulkan layers that reorder devices.")
        stale = {k: got.get(k, "") for k, v in self.runtime.items() if got.get(k, "") != v}
        if stale:
            # A cache is valid for the runtime that will read it: modules built any other way are other modules to the driver.
            child.close(kill=True)
            raise SchedError("the pipeline builder was made with another runtime than the one prepare ships "
                             f"({', '.join(f'{k}={v or chr(45)}, expected {self.runtime[k]}' for k, v in stale.items())}): it would "
                             "build the compute modules in a form a launch never asks for, and every program with "
                             "workgroup memory would build again, cold, at the first launch.")
        with self.lock:
            odd = {k: got.get(k, "") for k, allowed in self.forms.items()
                   if got.get(k, "") not in allowed or got.get(k) != self.forms_seen.get(k, got.get(k))}
            if not odd:
                self.forms_seen.update({k: got[k] for k in self.forms})
        if odd:
            # The driver's cache keys on the module, and a builder in another form than the one asked for (or than the other
            # builders) writes other modules than a launch in that form asks for.
            child.close(kill=True)
            wanted = {k: self.forms_seen.get(k) or " or ".join(self.forms[k]) for k in odd}
            raise SchedError("the pipeline builder chose another form than the step asked for ("
                             + ", ".join(f"{k}={v or '-'}, expected {wanted[k]}" for k, v in odd.items())
                             + "): its cache would hold other modules than a launch in that form asks for.")
        state = got.get("cache", "?")
        if state == "loaded":
            # What a builder that holds a seed weighs before it builds anything: the driver may keep far more than the
            # seed's bytes, and a builder is replaced for what it grew by, not for what it started with.
            child.base_ws = pool.working_set(child.proc)
        with self.lock:
            self.runtime_seen = {k: got[k] for k in self.runtime}
            self.cache_seen[state] = self.cache_seen.get(state, 0) + 1
            if state == "loaded" and got.get("seeded", "").isdigit():
                self.seeded_seen.append(int(got["seeded"]))
            self.children[wid] = child
            if attempt:
                self.stats.restarts += 1
        return child

    def _flush(self, child: Child, pending: list[Result]) -> None:
        with self.lock:
            name = f"seg-{self.next_segment:06d}"
            self.next_segment += 1
        # A seeded builder's flush copies the seed and what it built, in memory beside the driver's own copy, and the
        # governor's budget does not hold it: the workers flush one at a time instead of all together at the end.
        with self.flush_lock:
            reply = child.ask(f"flush {name}", self.timeout)
        word, rest = parse_reply(reply or "")
        if word != "saved" or len(rest) < 2:
            raise SchedError(f"the pipeline builder could not save a segment: {reply!r}. Its log is {child.log_path}.")
        size = int(rest[1])
        with self.lock:
            self.stats.segments += 1
            self.stats.segment_bytes += size
            for r in pending:
                r.segment = name
            self.on_saved(pending, name, size)

    def _build(self, child: Child, item: Item, concurrency: int, timeout: float) -> Result:
        child.ws_peak = child.ws
        t0 = time.monotonic()
        reply = child.ask(f"build {item.kind} {item.key}", timeout)
        wall = (time.monotonic() - t0) * 1000.0
        r = Result(item.key, item.kind, "ok", wall_ms=wall, concurrency=concurrency, started=t0 - self.t0,
                   peak_ws=child.ws_peak, module_bytes=item.module_bytes)
        if reply is None:
            code = child.exit_code()
            fatal = fatal_line(child.log_tail())
            r.status = "fail" if refused(fatal) and code in FATAL_EXITS else "crash"
            r.message = fatal or f"the builder exited with code {code}"
        elif reply == "":
            r.status, r.message = "timeout", f"no answer in {timeout:.0f} s"
        else:
            word, rest = parse_reply(reply)
            if word == "ok" and len(rest) == 5:
                r.ns, r.hit, r.module_bytes = int(rest[2]), int(rest[3]), int(rest[4])
            elif word == "died":
                # A named fatal of the runtime: the builder saved what it had built as a segment of its own and says
                # so in place of the reply ("died NAME BYTES LINE", NAME is - when there was nothing to save).
                fields = reply.strip().split(" ", 3)
                r.message = fields[3] if len(fields) > 3 else "the builder stopped by name"
                r.status = "fail" if refused(r.message) else "crash"
                if len(fields) > 2 and fields[1] != "-" and fields[2].isdigit():
                    r.segment, r.ns = fields[1], 0
                child.exit_code()
            else:
                # a command the builder refused (a key or a kind it does not take): it is alive and goes on
                r.status, r.message, r.dead = "fail", reply.strip(), False
        return r

    def _bye(self, reply: str | None) -> None:
        """The validation layer's counts the builder reports when it quits (zero when validation is off)."""
        word, rest = parse_reply(reply or "")
        if word == "bye" and len(rest) >= 4 and rest[1].isdigit() and rest[3].isdigit():
            with self.lock:
                self.stats.validation_errors += int(rest[1])
                self.stats.validation_warnings += int(rest[3])

    def _drive(self, wid: int, retry_pass: bool) -> None:
        child: Child | None = None
        pending: list[Result] = []
        pending_items: dict[str, Item] = {}
        last = time.monotonic()
        attempt = 0
        try:
            while True:
                item = self._take()
                if item is None:
                    break
                if child is None:
                    child = self._start(wid, attempt)
                    attempt += 1
                    last = time.monotonic()
                n = self.gov.acquire(item)
                try:
                    r = self._build(child, item, n, RETRY_TIMEOUT if retry_pass else self.timeout)
                finally:
                    self.gov.release(item)
                if r.status == "ok":
                    pending.append(r)
                    pending_items[r.key] = item
                elif not r.dead:
                    with self.lock:
                        self.on_failed(r)
                else:
                    if r.segment and pending:
                        # the dying builder saved its cache: what it had built is safe and is not built again
                        size = (self.segments / f"{r.segment}.bin").stat().st_size
                        with self.lock:
                            self.stats.segments += 1
                            self.stats.segment_bytes += size
                            for x in pending:
                                x.segment = r.segment
                            self.on_saved(pending, r.segment, size)
                        pending, pending_items = [], {}
                        r.segment = ""
                    lost = [pending_items[x.key] for x in pending]
                    pending, pending_items = [], {}
                    child.close(kill=True)
                    with self.lock:
                        self.children.pop(wid, None)
                    child = None
                    self._requeue(lost)
                    if r.status == "fail" or retry_pass:
                        with self.lock:
                            self.on_failed(r)
                    else:
                        with self.lock:
                            self.retry.append((item, f"{r.status}: {r.message}"))
                if self.on_result:
                    self.on_result(r)
                if child is not None and pending and time.monotonic() - last >= self.flush_seconds:
                    self._flush(child, pending)
                    pending, pending_items, last = [], {}, time.monotonic()
                if child is not None and self.final_flush and child.ws_peak >= self.recycle_bytes + child.base_ws:
                    # The driver keeps the memory of a big build for the life of the process, and the governor counts
                    # only the items running: a worker that holds a gigabyte more than it started with is saved and
                    # replaced by a fresh one.
                    if pending:
                        self._flush(child, pending)
                        pending, pending_items, last = [], {}, time.monotonic()
                    self._bye(child.ask("quit", 60))
                    child.close()
                    with self.lock:
                        self.children.pop(wid, None)
                        self.stats.recycled += 1
                    child = None
            if child is not None and pending and self.final_flush:
                self._flush(child, pending)
                pending = []
            if child is not None:
                self._bye(child.ask("quit", 60))
        except SchedError as e:
            with self.lock:
                self.error = self.error or e
            self.gov.stop()
        except Exception as e:   # a bug here must stop the pool, not hang it
            with self.lock:
                self.error = self.error or e
            self.gov.stop()
        finally:
            if child is not None:
                child.close()
            with self.lock:
                self.children.pop(wid, None)

    def run(self, retry_pass: bool = False) -> None:
        """Build every item; the ones that got no answer wait in self.retry for run(retry_pass=True)."""
        self.stop_monitor.clear()
        mon = threading.Thread(target=self._monitor, daemon=True)
        mon.start()
        threads = [threading.Thread(target=self._drive, args=(i, retry_pass), daemon=True) for i in range(self.workers)]
        try:
            for t in threads:
                t.start()
            # Wait on an event-free poll, never on Thread.join alone: a KeyboardInterrupt inside join on Python 3.12
            # marks the thread stopped (bpo-45274) and a second join would return at once.
            while any(t.is_alive() for t in threads):
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.gov.stop()
            with self.lock:
                self.error = self.error or KeyboardInterrupt()
                children = list(self.children.values())
            for c in children:      # a build in flight is lost with its unsaved cache; nothing holds the player's GPU
                c.close(kill=True)
            raise
        finally:
            self.stop_monitor.set()
            self.stats.peak_running = max(self.stats.peak_running, self.gov.peak_running)
            self.stats.held_back_s = self.gov.held_back_s
        if self.error and not isinstance(self.error, KeyboardInterrupt):
            raise self.error


# -- the progress file -------------------------------------------------------------------------------------------

PROGRESS_FIELDS = ("status", "key", "kind", "ns", "hit", "module_bytes", "wall_ms", "concurrency", "peak_ws", "segment",
                   "message")


def progress_line(r: Result) -> str:
    message = r.message.replace("\t", " ").replace("\n", " ")
    return "\t".join(str(v) for v in (r.status, r.key, r.kind, r.ns, r.hit, r.module_bytes, f"{r.wall_ms:.1f}",
                                      r.concurrency, r.peak_ws, r.segment or "-", message))


def read_progress(path: Path, segments: Path) -> dict[str, Result]:
    """The items an earlier run of the same key finished: the ones saved in a segment that is still there, and the ones
    the runtime refused by name. A line cut short by a stop is dropped; crashes and timeouts are not read, so they are
    built again."""
    out: dict[str, Result] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    have = {p.stem for p in Path(segments).glob("seg-*.bin")}
    for line in text.splitlines():
        f = line.split("\t")
        if len(f) != len(PROGRESS_FIELDS):
            continue
        row = dict(zip(PROGRESS_FIELDS, f))
        try:
            r = Result(row["key"], row["kind"], row["status"], int(row["ns"]), int(row["hit"]), int(row["module_bytes"]),
                       float(row["wall_ms"]), int(row["concurrency"]), int(row["peak_ws"]),
                       segment="" if row["segment"] == "-" else row["segment"], message=row["message"])
        except ValueError:
            continue
        if r.status == "ok" and r.segment in have:
            out[r.key] = r
        elif r.status == "fail":
            out[r.key] = r
    return out


def next_segment_number(segments: Path) -> int:
    n = 0
    for p in Path(segments).glob("seg-*.bin"):
        try:
            n = max(n, int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            pass
    return n + 1
