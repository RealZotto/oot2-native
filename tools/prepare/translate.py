# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The translate step of `oot2 prepare` (architecture D20 and D21): the game's shader programs, read from the RomFS the
extract step indexed, translated to SPIR-V with the default answers and packed into the shader cache's base pack.

Input: the RomFS through romfs/romfs-index.bin and the image the extract step's record names (the player's .nup in
place, romfs/romfs.bin in the copy form). Every file read is checked against the SHA-256 its index row gives, so a
damaged dump or copy stops the step by name instead of feeding the translator wrong bytes.

Output, under the output folder:

    shaders/shader-base.bin   the base pack (tools/shaders/cache_pack.py): every job, translated with the default
                              answers or recorded as a named failure; the runtime maps it
    shaders/split/            what tools/shaders/split.py writes: programs.bin (each program's code once), jobs.jsonl,
                              programs.tsv, pairing.tsv and summary.json, which the ledger and pipeline steps read

Scratch, under temp/translate/: spill.bin (every translated job's module and description, appended as the answers
come), results.tsv (one line per job answered, with where its entry lies in spill.bin), the translator
children's logs, and the key of the run they belong to.

In order:

1. Split. The shader archives of the RomFS (zstd frames with dictionaries, which need Python 3.14's compression.zstd)
   are cut into jobs, one per distinct program, with the BAHS records walked (split.py). A split of the same RomFS by
   the same split.py is kept (its own stamp in shaders/split/), so a new translator does not split again.
2. Translate. Every job, with the default answers and no validation, on a pool of translator children
   (tools/shaders/pool.py). The child count follows the CPU count and the free memory (--translate-jobs sets it), the
   pool's memory guard gets the same share of the free memory, and a progress line comes at least every 10 seconds.
   Each answer's two parts ride in the reply line and are appended to spill.bin by the pool thread that got them.
3. Pack. Every job goes into the base pack, read from spill.bin: its module and description, or a failure record.
   The scratch files are deleted once the pack is in place.

Why one spill file. The translator answers with each job's two parts in its reply line instead of loose files, so
the desktop's real-time scanner never reopens them: each pair is read once, by the thread that got the answer, while
the other children keep translating, and packing reads one large file.

The failure rule. A job the translator refuses (a "fail" reply: an instruction it lacks, a program it cannot read) is
packed as a failure record holding "fail: <its reason>", and the step goes on: the runtime stops with that reason, by
name, if the game ever uses the program. A job that got no answer (its child exited, or did not answer within TIMEOUT
seconds) is tried once more after the others, alone, within RETRY_TIMEOUT seconds; if it still gets none it is packed
the same way ("crash: ...", "timeout: ..."). The step fails only when the translator translated none of the jobs:
then the translator, not the programs, is what is broken, and failing before the lift is why this step comes first.
Every failure is named in a warning, and the step's record counts them by status.

Resume. An interrupted run keeps the jobs already answered: results.tsv is written one line per answer, after that
job's entry is in spill.bin, so the next run translates only the rest (S1 took about 8 minutes for every job). A job
is kept when the translator refused it, or when it was translated and its entry lies inside spill.bin; a job that got
no answer is tried again. Scratch of a run with another key (another translator, another RomFS, other tools) is
deleted first. Packing checks every module against its description, so a damaged entry is refused there.

The key: the translator's identity (the SHA-256 of its executable, tools/build_xlate.py), the optimizer form the
pack holds (check.XLATE_OPTIMIZER: the sound form, the translator's default since the AST optimizer rule became it;
check.shader_tools refuses a translator whose default is another), the code of this step, split.py, pool.py and
cache_pack.py, the description schema, and through Step.needs the extract step's key. A new translator translates
again and nothing upstream runs again: a folder prepared before the sound form was the default holds the old
executable's identity, so its translate step, and every step that needs it, runs again once. The translator is found
as the check step finds it (check.translator): --xlate, else the OOT2_XLATE variable, else oot2-xlate on PATH. The
worker count is not in the key.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from . import check
from .extract import IMAGE, IN_PLACE, INDEX, ROMFS, read_index
from .runner import Context, PrepareError, Step, fingerprint

TOOLS = Path(__file__).resolve().parent.parent
SHADER_TOOLS = TOOLS / "shaders"
for _p in (TOOLS, SHADER_TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_game  # noqa: E402  (tools/build_game.py: available_ram)
import cache_pack  # noqa: E402  (tools/shaders/cache_pack.py)
import pool  # noqa: E402  (tools/shaders/pool.py)
import split  # noqa: E402  (tools/shaders/split.py)

FORMAT = 1                      # changes when the outputs change shape: the step runs again
HERE = Path(__file__).resolve().parent
SHADERS = "shaders"             # the output folder of this step
SPLIT = "split"                 # shaders/split/
SCRATCH = "translate"           # temp/translate/
SPILL = "spill.bin"             # temp/translate/spill.bin
STAMP = "stamp.txt"             # the key the folder holding it belongs to
RESULTS = "results.tsv"
# tools/shaders/corpus.py's columns, then where a translated job's entry lies in spill.bin (its module first)
RESULT_FIELDS = ("key", "status", "micros", "wall_ms", "spv_bytes", "message", "spill_offset", "desc_bytes")
SPLIT_FILES = ("programs.bin", "jobs.jsonl", "programs.tsv", "pairing.tsv", "summary.json")
TIMEOUT = 300.0                 # seconds a child may take on one job; S1's slowest took 15.7 s
RETRY_TIMEOUT = 1200.0          # for a job that got no answer, tried again alone
# S1 (the shader translation study): one child's working set reached 473 MB after the largest programs; twelve peaked at 2.97 GB.
WORKER_MEM = 512 << 20
MEM_SHARE = 0.8                 # of the free memory the children may take, as the compile step's jobs
NAMED = 5                       # failures named one by one in the warning
GiB = 1 << 30


class Translate(Step):
    id = "translate"
    needs = ("extract",)        # the RomFS index and the image it reads from

    def key(self, ctx: Context) -> str:
        return fingerprint(f"translate-{FORMAT}", f"translator={check.translator_identity(ctx)}",
                           f"optimizer={check.XLATE_OPTIMIZER}", f"schema={cache_pack.DESC_SCHEMA}",
                           HERE / "translate.py", SHADER_TOOLS / "split.py",
                           SHADER_TOOLS / "pool.py", SHADER_TOOLS / "cache_pack.py")

    def current(self, ctx: Context, record: dict) -> bool:
        try:
            if (ctx.dest / SHADERS / cache_pack.BASE_FILE).stat().st_size != record["pack_bytes"]:
                return False
            folder = ctx.dest / SHADERS / SPLIT
            return all((folder / name).stat().st_size == size for name, size in record["split_files"].items())
        except (OSError, KeyError, TypeError, AttributeError):
            return False

    def verify(self, ctx: Context, record: dict) -> bool:
        """Every entry of the pack against its row and its description, and the pack's rows against the split's
        jobs: one row for each, from the recorded translator."""
        path = ctx.dest / SHADERS / cache_pack.BASE_FILE
        try:
            if cache_pack.verify_pack(path, role=cache_pack.BASE):
                return False
            pack = cache_pack.open_pack(path, role=cache_pack.BASE)
            return pack.translator == record["translator"]["sha256"] and \
                {row.key for row in pack.rows} == set(load_jobs(ctx.dest / SHADERS / SPLIT))
        except (OSError, KeyError, TypeError, ValueError, cache_pack.PackError):
            return False

    def run(self, ctx: Context) -> dict:
        rep = ctx.reporter
        extracted = ctx.record("extract")
        if not extracted:
            raise PrepareError("the extract step has not run, so there is no RomFS to read the shaders from")
        path = check.translator(ctx)
        identity = check.translator_identity(ctx)
        out = ctx.dest / SHADERS
        out.mkdir(parents=True, exist_ok=True)
        pack = out / cache_pack.BASE_FILE
        for stale in (pack, pack.with_name(pack.name + ".tmp")):
            stale.unlink(missing_ok=True)       # write-once: a step that runs again deletes its own pack first
        scratch = open_scratch(ctx, ctx.keys[self.id])

        folder, summary, split_seconds = ensure_split(ctx, extracted)
        jobs = load_jobs(folder)
        if not jobs:
            raise PrepareError(f"no shader program was found in the RomFS (none under {', '.join(split.TREES)}), so "
                               "this is not the game's RomFS. Check the dump.")
        problems = summary.get("error_count", 0)
        if problems:
            reasons = "; ".join(f"{e['label']}: {e['reason']}" for e in summary.get("errors", [])[:NAMED])
            rep.warning(f"the split met {problems:,} problem(s) in the shader archives, so those programs have no job "
                        f"(the game stops by name if it uses one): {reasons}")

        spill = Spill(scratch / SPILL)
        kept = kept_results(read_results(scratch / RESULTS), jobs, spill.end)
        workers, limit, why = sizing(ctx)
        todo = [k for k in sorted(jobs) if k not in kept]
        rep.say(f"translate: {len(jobs):,} jobs, {len(kept):,} kept from an interrupted run, {len(todo):,} to "
                f"translate on {workers} translator processes ({why}); translator {path.name}, identity "
                f"{identity[:16]}")
        command = check.translator_command(path)
        progress = Progress(rep, len(jobs), len(kept))
        t0 = time.monotonic()
        try:
            answers, stats = translate(command, jobs, todo, scratch, spill, workers, limit, TIMEOUT, progress)
            retry = sorted(k for k, a in answers.items() if a["status"] in ("crash", "timeout"))
            if retry:
                rep.say(f"translate: {len(retry):,} job(s) got no answer "
                        f"({status_counts(answers[k]['status'] for k in retry)}); each is tried again alone, within "
                        f"{RETRY_TIMEOUT:.0f} s")
                again, _ = translate(command, jobs, retry, scratch, spill, 1, limit, RETRY_TIMEOUT, progress)
                answers.update(again)
        finally:
            spill.close()
        translate_seconds = time.monotonic() - t0

        final = {**kept, **answers}
        missing = sorted(set(jobs) - set(final))
        if missing:
            raise PrepareError(f"{len(missing):,} jobs got no result from the translator pool, the first {missing[0]}")
        failed = sorted(k for k in jobs if final[k]["status"] != "ok")
        translated = len(jobs) - len(failed)
        if not translated:
            first = failed[0]
            raise PrepareError(f"the translator {path} translated none of the {len(jobs):,} shader programs, so the "
                               f"translator is what is broken; the first reason: {first}: {failure_text(final[first])}. "
                               f"Its children's logs are in {scratch / 'logs'}.")

        t1 = time.monotonic()
        spill_bytes = spill.end
        try:
            with open(scratch / SPILL, "rb") as source:
                written = cache_pack.write_pack(pack, pack_entries(jobs, final, source), translator=identity)
        except cache_pack.PackError as e:
            if e.code == "INPUT":   # an entry that does not belong: translate afresh on the next run
                shutil.rmtree(scratch, ignore_errors=True)
            raise PrepareError(f"the shader pack could not be written ({e}). Run oot2 prepare again.") from None
        pack_seconds = time.monotonic() - t1
        for name in (SPILL, RESULTS):
            (scratch / name).unlink(missing_ok=True)
        split_files = {name: (folder / name).stat().st_size for name in SPLIT_FILES}
        split_bytes = sum(split_files.values())

        if failed:
            named = "; ".join(f"{k}: {failure_text(final[k])}" for k in failed[:NAMED])
            rep.warning(f"{len(failed):,} of {len(jobs):,} shader programs did not translate "
                        f"({status_counts(final[k]['status'] for k in failed)}); each is packed as a failure record, "
                        f"and the game stops with its reason if it ever uses one: {named}"
                        f"{'; ...' if len(failed) > NAMED else ''}")
        rep.say(f"translate: {translated:,} of {len(jobs):,} jobs translated in {translate_seconds:.0f} s on {workers} "
                f"processes (children's peak {mib(stats.peak_children_bytes)}); pack {mib(written['bytes'])} "
                f"(index {mib(written['index_bytes'])}, modules {mib(written['module_bytes'])}, descriptions "
                f"{mib(written['description_bytes'])}) written in {pack_seconds:.1f} s")
        return {
            "bytes": written["bytes"] + split_bytes,
            "pack_bytes": written["bytes"],
            "index_bytes": written["index_bytes"],
            "module_bytes": written["module_bytes"],
            "description_bytes": written["description_bytes"],
            "split_files": split_files,
            "spill_bytes": spill_bytes,
            "scratch_peak_bytes": split_bytes + spill_bytes + written["bytes"],
            "jobs": len(jobs),
            "translated": translated,
            "failed": dict(sorted(status_tally(final[k]["status"] for k in failed).items())),
            "failed_keys": failed[:100],
            "split_problems": problems,
            "kept_jobs": len(kept),
            "retried": len(retry),
            "workers": workers,
            "memory_limit": limit,
            "split_seconds": round(split_seconds, 1),
            "translate_seconds": round(translate_seconds, 1),
            "pack_seconds": round(pack_seconds, 1),
            "peak_children_bytes": stats.peak_children_bytes,
            "min_available_bytes": stats.min_available_bytes,
            "recycled": stats.recycled,
            "restarts": stats.restarts,
            "peak_process": split.peak_working_set(),
            "translator": {"path": str(path), "sha256": identity},
        }


# --- the RomFS ------------------------------------------------------------------------------------------------------

def romfs_image(dest: Path, record: dict) -> Path:
    """The file the RomFS index reads from, as the extract step's record names it: the player's .nup in place,
    romfs/romfs.bin in the copy form."""
    if record.get("romfs_form") == IN_PLACE:
        return Path(record["nup"]["path"])
    return dest / ROMFS / IMAGE


class RomfsImage:
    """split.Image over the extract step's output: the index and the image it reads from. A file is read with one seek
    and checked against the SHA-256 of its index row."""

    def __init__(self, index: Path, image: Path):
        try:
            header, rows = read_index(index)
            size = image.stat().st_size
        except (OSError, ValueError) as e:
            raise split.InputError(f"the RomFS index {index} or its image {image} cannot be read: {e}") from None
        if size != header["image_bytes"]:
            raise split.InputError(f"{image} has {size:,} bytes and the RomFS index gives {header['image_bytes']:,}: "
                                   "run oot2 prepare again so that the extract step indexes it again")
        self.image = image
        self.rows = {path: (off, length, sha) for path, off, length, sha in rows}
        self._fh = None

    def list_paths(self, prefixes) -> list[str]:
        return sorted(p for p in self.rows if p.startswith(tuple(prefixes)))

    def read(self, path: str) -> bytes:
        try:
            off, length, sha = self.rows[path]
        except KeyError:
            raise split.InputError(f"{path} is not in the RomFS index") from None
        if self._fh is None:
            self._fh = open(self.image, "rb")
        self._fh.seek(off)
        data = self._fh.read(length)
        if len(data) != length or hashlib.sha256(data).digest() != sha:
            raise split.InputError(f"the RomFS file {path}, read from {self.image}, does not match the SHA-256 of the "
                                   "RomFS index (the dump or its copy is damaged; oot2 prepare --verify checks them)")
        return data

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# --- the split ------------------------------------------------------------------------------------------------------

def ensure_split(ctx: Context, extracted: dict) -> tuple[Path, dict, float]:
    """shaders/split/ for this RomFS, kept when an earlier run split the same RomFS with the same split.py. Returns the
    folder, split.py's summary and the seconds the split took (0 when kept)."""
    rep = ctx.reporter
    folder = ctx.dest / SHADERS / SPLIT
    stamp = fingerprint(f"split-{FORMAT}", f"extract={ctx.keys.get('extract')}", SHADER_TOOLS / "split.py")
    try:
        if (folder / STAMP).read_text(encoding="utf-8") == stamp and all((folder / n).is_file() for n in SPLIT_FILES):
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            rep.say(f"translate: split kept: {summary['jobs']:,} jobs from an earlier split of the same RomFS")
            return folder, summary, 0.0
    except (OSError, ValueError, KeyError, TypeError):
        pass
    part = folder.with_name(SPLIT + ".part")
    for old in (part, folder):
        if old.exists():
            shutil.rmtree(old)
    t0 = time.monotonic()
    image = None
    try:
        image = RomfsImage(ctx.dest / ROMFS / INDEX, romfs_image(ctx.dest, extracted))
        summary = split.scan_image(image, part, "walk", progress=lambda text: rep.progress(
            "split", f"translate: split: {text}", every=10.0))
    except split.InputError as e:
        raise PrepareError(f"the shader archives cannot be read: {e}") from None
    finally:
        if image is not None:
            image.close()
    seconds = time.monotonic() - t0
    (part / STAMP).write_text(stamp, encoding="utf-8")
    os.rename(part, folder)
    rep.say(f"translate: split {summary['files_scanned']:,} shader files into {summary['jobs']:,} jobs "
            f"({', '.join(f'{n:,} {s}' for s, n in summary['jobs_by_stage'].items())}) in {seconds:.1f} s; programs "
            f"{mib(summary['programs_bin_bytes'])}")
    return folder, summary, seconds


def load_jobs(folder: Path) -> dict[str, dict]:
    """The job objects of jobs.jsonl by key, their code pointing at programs.bin where it is now."""
    code = str((folder / "programs.bin").resolve())
    jobs: dict[str, dict] = {}
    with open(folder / "jobs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                job = json.loads(line)
                job["code"] = code
                jobs[job["key"]] = job
    return jobs


# --- translation ----------------------------------------------------------------------------------------------------

def open_scratch(ctx: Context, key: str) -> Path:
    """temp/translate/, emptied first when it belongs to a run with another key."""
    scratch = ctx.dest / check.TEMP / SCRATCH
    try:
        same = (scratch / STAMP).read_text(encoding="utf-8") == key
    except OSError:
        same = False
    if not same and scratch.exists():
        shutil.rmtree(scratch)
        ctx.reporter.say("translate: deleted the scratch of an earlier run with other inputs")
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / STAMP).write_text(key, encoding="utf-8")
    return scratch


class Spill:
    """spill.bin: each translated job's module, then its description, appended as the answers come, so memory holds
    only the jobs in flight."""

    def __init__(self, path: Path):
        self.f = open(path, "ab")
        self.end = path.stat().st_size      # an interrupted run's entries stay where results.tsv says they are
        self.lock = threading.Lock()

    def add(self, module: bytes, desc: bytes, key: str) -> tuple[int, int, int]:
        """Append the job's module and description to the spill. Returns the entry's offset, module size and description size."""
        with self.lock:
            offset = self.end
            self.f.write(module)
            self.f.write(desc)
            self.f.flush()
            self.end += len(module) + len(desc)
        return offset, len(module), len(desc)

    def close(self) -> None:
        with self.lock:
            self.f.close()


def read_results(path: Path) -> dict[str, dict]:
    """The last line of results.tsv for each key ({} when there is none)."""
    rows: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("key"):
                    rows[row["key"]] = row
    except FileNotFoundError:
        pass
    return rows


def kept_results(rows: dict[str, dict], jobs: dict[str, dict], spill_end: int) -> dict[str, dict]:
    """The answers an interrupted run of the same key leaves that this run keeps: a job the translator refused, and a
    job translated whose entry lies inside spill.bin. A job that got no answer is not kept."""
    keep: dict[str, dict] = {}
    for key, row in rows.items():
        if key not in jobs:
            continue
        status = row.get("status")
        if status == "fail":
            keep[key] = row
        elif status == "ok":
            try:
                if int(row["spill_offset"]) + int(row["spv_bytes"]) + int(row["desc_bytes"]) <= spill_end:
                    keep[key] = row
            except (KeyError, TypeError, ValueError):
                pass
    return keep


def sizing(ctx: Context) -> tuple[int, int, str]:
    """Translator processes, the pool's memory limit, and why: the CPU count capped by the free memory at WORKER_MEM
    a child, unless --translate-jobs says how many."""
    cpus = os.cpu_count() or 4
    free = build_game.available_ram() or 0
    limit = max(WORKER_MEM, int(free * MEM_SHARE)) if free else 4 * GiB
    given = getattr(ctx.options, "translate_jobs", None)
    if given:
        return given, limit, "--translate-jobs"
    if not free:
        return cpus, limit, f"{cpus} CPUs, free memory unknown"
    by_mem = max(1, int(free * MEM_SHARE) // WORKER_MEM)
    return min(cpus, by_mem), limit, f"{cpus} CPUs, {free / GiB:.1f} GiB free memory allows {by_mem}"


class Progress:
    """Counts the answers and prints a line at most every 10 seconds (the pool calls add from its worker threads)."""

    def __init__(self, rep, total: int, kept: int):
        self.rep = rep
        self.total = total
        self.kept = kept
        self.answered = 0           # in this run
        self.failed = 0
        self.t0 = time.monotonic()
        self.lock = threading.Lock()

    def add(self, r: pool.Result) -> None:
        with self.lock:
            self.answered += 1
            self.failed += r.status != "ok"
            done = self.kept + self.answered
            rate = self.answered / max(time.monotonic() - self.t0, 1e-3)
            self.rep.progress("translate", f"translate: {done:,} of {self.total:,} jobs answered, "
                                           f"{self.failed:,} not translated, {rate:,.0f} a second",
                              every=10.0, final=done >= self.total)


def translate(command: list[str], jobs: dict[str, dict], keys: list[str], scratch: Path, spill: Spill,
              workers: int, limit: int, timeout: float, progress: Progress) -> tuple[dict[str, dict], pool.PoolStats]:
    """Translate the jobs of keys on the pool. Each answer is a results.tsv row; a translated job's module and description
    are appended to the spill first, by the thread that got the answer, and the pool keeps none of their bytes. Returns the rows by key. Ctrl+C stops feeding jobs;
    the jobs in flight finish and are recorded, then KeyboardInterrupt goes on to the runner. A second Ctrl+C leaves at
    once, and the answers that come after it are not recorded (those jobs are translated again).

    The calling thread waits on an event the pool's thread sets when it ends, never on Thread.join: on Python 3.12,
    a KeyboardInterrupt raised inside join marks the thread as stopped while it still runs (CPython's bpo-45274
    handling), so a second join returns at once and the results file would close under the jobs in flight."""
    answers: dict[str, dict] = {}
    if not keys:
        return answers, pool.PoolStats()
    lock = threading.Lock()
    stop = threading.Event()
    finished = threading.Event()
    box: dict = {}
    path = scratch / RESULTS
    header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        if header:
            writer.writeheader()
            fh.flush()

        def on_result(r: pool.Result) -> None:
            if fh.closed:                   # after a second Ctrl+C
                return
            row = {"key": r.key, "status": r.status, "micros": r.micros, "wall_ms": r.wall_ms,
                   "spv_bytes": r.spv_bytes, "message": " ".join(r.message.split()), "spill_offset": "",
                   "desc_bytes": ""}
            if r.status == "ok":
                try:
                    row["spill_offset"], row["spv_bytes"], row["desc_bytes"] = spill.add(r.module, r.description, r.key)
                except ValueError:          # the spill closed: a second Ctrl+C
                    return
                except OSError as e:
                    row.update(status="fail", spv_bytes=0,
                               message=f"the translator answered, but its output cannot be read: {e}")
            with lock:
                if fh.closed:
                    return
                writer.writerow(row)
                fh.flush()
                answers[r.key] = {f: str(v) for f, v in row.items()}
            progress.add(pool.Result(r.key, row["status"]))

        def feed():
            for k in keys:
                if stop.is_set():
                    return
                yield jobs[k], None

        def work() -> None:
            try:
                box["stats"] = pool.run_pool(command, feed(), min(workers, len(keys)), scratch / "logs",
                                             timeout=timeout, memory_limit=limit, on_result=on_result)[1]
            except BaseException as e:      # reported on the calling thread
                box["error"] = e
            finally:
                finished.set()

        threading.Thread(target=work, name="translate-pool", daemon=True).start()
        try:
            while not finished.wait(0.2):   # a timed wait lets Ctrl+C through on Windows too
                pass
        except KeyboardInterrupt:
            stop.set()
            while not finished.wait(0.2):
                pass
            raise
    if "error" in box:
        raise box["error"]
    return answers, box["stats"]


# --- packing --------------------------------------------------------------------------------------------------------

def pack_entries(jobs: dict[str, dict], final: dict[str, dict], source) -> list:
    """The base pack's entries, one per job: a translated job's module and description read from the spill (an open
    file) only while the pack writer writes that entry, a failure record's reason."""
    def piece(key: str, offset: int, size: int):
        def read() -> bytes:
            source.seek(offset)
            data = source.read(size)
            if len(data) != size:
                raise cache_pack.PackError("INPUT", f"{key}: spill.bin ends inside its entry")
            return data
        return read

    entries = []
    for key in sorted(jobs):
        answer = final[key]
        if answer["status"] == "ok":
            offset, module = int(answer["spill_offset"]), int(answer["spv_bytes"])
            entries.append(cache_pack.Entry(key, piece(key, offset, module),
                                            piece(key, offset + module, int(answer["desc_bytes"]))))
        else:
            entries.append(cache_pack.Entry(key, None, failure_text(answer), failed=True))
    return entries


def failure_text(answer: dict) -> str:
    """A failure record's reason: "<status>: <the translator's or the pool's message>"."""
    return f"{answer['status']}: {(answer.get('message') or '').strip() or 'no reason given'}"


def status_tally(statuses) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return out


def status_counts(statuses) -> str:
    return ", ".join(f"{s} {n:,}" for s, n in sorted(status_tally(statuses).items()))


def mib(n: int) -> str:
    return f"{n / (1 << 20):,.1f} MiB"
