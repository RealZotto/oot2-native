# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The pipelines step of `oot2 prepare` (architecture D12, D21 and section 6.1, step 1; the pipeline step): with the GPU the
game will use, build every pipeline library and compute pipeline the shader cache holds, into the application pipeline
cache `pipelines/pipeline.cache`, so that a launch creates them again from that file in milliseconds instead of
compiling them. What it builds, in what order, on how many workers and how a stopped run resumes is pipesched.py.

The builder is `tools/pipebuild/pipebuild.c` linked with runtime/core, runtime/nvn and runtime/gpu and compiled here
with the compile step's compiler (about ten seconds), so the create infos are the ones the Vulkan back end makes at
launch. It runs as several worker processes, each with its own device, and leaves one segment file of cache per flush
interval in temp/pipelines/segments/; when everything is built it merges them into the one cache file.

Which device. The one the runtime would open: the first discrete device that has every item of
runtime/include/oot2/vk_required.h, else the first integrated, else the first other (oot2/gpu_vk.h), from the host
probe the check step ran. Without one the step is skipped with a warning that names the missing items, and recorded as
skipped, because while the null back end is oot2.exe's default a GPU is not required (check.REQUIRE_VULKAN_ITEMS); once
that switch is on the same finding fails the step. The device's identity (vendor, device, driver version, pipeline
cache UUID, API version) is in the step's key, so a driver update builds everything again: a cache is valid only for
the driver that wrote it.

Which runtime. A cache is valid only for the runtime that reads it too: the runtime rewrites each compute module (its
workgroup memory starts zeroed) before the driver sees it, and the driver's cache keys on the module. The builder says in
its ready line what it does (`zero_workgroup`); the key names what the step requires (RUNTIME_SAYS), a builder that says
anything else stops the step with the reason, and the record keeps what the builder said, so a record made by an earlier
runtime is not current and the step builds again.

The seed. When the key changes the previous cache is not thrown away: pipeseed.py checks it (header, size, runtime, device
and driver) and keeps it as the seed every builder starts from, so that the programs whose module did not change are served
by the driver in a millisecond or so and only the changed ones build cold. A cache that fails a check is dropped, deleted,
and the drop is named in the log and in the record (`seed`); the step then builds from an empty cache as it always did.
`--pipelines-cold` takes no seed, because it exists to time a cold build. A seeded builder saves only when it is recycled
and at the end (a flush would hand it an empty cache and the seed would stop serving). A stopped seeded run keeps the seed
and the segments its builders saved before the stop (on the same key those items are not built again; only a change of
the key wipes them), and loses what each builder built since its last save. The merged cache keeps the entries of modules
that no longer exist, so it grows with every key change that rebuilt much; `--pipelines-cold` is the only prune, and the
step warns when the cache is more than half as large again as the seed it started from.

What it does not build. Task and mesh pipelines wait for the mesh draws ticket and are counted as not built, so the
totals are a lower bound. Vertex libraries are built as translated, without padding: a pixel stage that reads an
output its vertex stage does not write needs a padded vertex library, and those pairs are known only from play
(the pipeline cost study), so the ledger's pipelines step builds them.

In the output folder:

  pipelines/pipeline.cache   the application pipeline cache (the runtime reads it at launch)
  pipelines/times.tsv        one row per module built or refused: the driver's time, the cache hit, the module size,
                             the worker's memory, how many ran at once; the tier rule reads it

`--pipelines-jobs N` sets the workers, `--pipelines-list FILE` limits the step to the job keys in a file, and
`--pipelines-cold` turns the driver's own pipeline cache off (OOT2_VK_DRIVER_CACHE), which is what a timing needs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import check, pipesched, pipeseed
from .runner import Context, PrepareError, Step, file_sha256, fingerprint

TOOLS = Path(__file__).resolve().parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(TOOLS / "shaders") not in sys.path:
    sys.path.insert(0, str(TOOLS / "shaders"))

import build_game  # noqa: E402  (tools/build_game.py)
import bundle_env  # noqa: E402  (tools/bundle_env.py)
import cache_pack  # noqa: E402  (tools/shaders/cache_pack.py)
import layout  # noqa: E402  (tools/layout.py)

from . import translate  # noqa: E402

FORMAT = 1                       # changes when the outputs change shape: the step runs again
# The runtime the cache must come from, as the builder's ready line says it: the runtime zero-initializes the workgroup memory
# of every compute module before the driver sees it (spv_zero_workgroup), so a cache written by a runtime that did not holds
# other modules than a launch asks for, and the programs with workgroup memory would build again, cold, once. The step's key
# names it, a builder that does not say it is refused, and the record keeps what the builder said; a record without it (made
# by an earlier runtime) is not current.
RUNTIME_SAYS = {"zero_workgroup": "1"}
HERE = Path(__file__).resolve().parent
RUNTIME = build_game.RUNTIME
SOURCE = TOOLS / "pipebuild" / "pipebuild.c"
OUT = "pipelines"                # the output folder of this step
CACHE = "pipeline.cache"
TIMES = "times.tsv"
SCRATCH = "pipelines"            # temp/pipelines/
SEGMENTS = "segments"
PROGRESS = "progress.tsv"
STAMP = "stamp.txt"
BUILDER = "builder"
MODULE_SIZE = 0x8000             # the placeholder module the builder's layout header describes (it loads no game)
MERGE_TIMEOUT = 3600
# The headers whose structures decide a create info, beside runtime/gpu itself: a change in any of them builds again.
KEY_HEADERS = ("vk_pipeline.h", "vk_required.h", "vk_caps.h", "spirv.h", "shader_desc.h", "shader_cache.h", "gpu_vk.h")
KEY_SOURCES = ("vk_pipeline.c", "vk_layout.c", "vk_caps.c", "spirv.c", "shader_desc.c", "vk_device.c")
SAMPLE = 64                      # items verify builds again from the cache
NAMED = 5                        # failures named one by one in the warning
GiB = 1 << 30


def status_counts(statuses) -> str:
    counts: dict[str, int] = {}
    for s in statuses:
        counts[s] = counts.get(s, 0) + 1
    return ", ".join(f"{n:,} {s}" for s, n in sorted(counts.items()))


@dataclass(frozen=True)
class Device:
    index: int                   # in vkEnumeratePhysicalDevices, which OOT2_VK_DEVICE takes
    name: str
    identity: str

    def expect(self) -> dict[str, str]:
        """What a builder's ready line must say, for a builder that opened this device and no other."""
        vendor, device, driver, uuid, _ = self.identity.split(":", 4)
        return {"vendor": vendor, "device": device, "driver": driver, "cache_uuid": uuid}


def usable_device(host: dict) -> tuple[Device | None, str]:
    """The device the runtime would open, from the host probe, or why there is none."""
    vk = host.get("vulkan")
    if not vk:
        return None, "Vulkan was not probed (the host probe was built without the Vulkan headers)"
    if not vk.get("loader"):
        return None, "no Vulkan loader was found"
    if vk.get("instance", 0) != 0:
        return None, f"the Vulkan loader refused to create an instance (VkResult {vk.get('instance')})"
    devices = vk.get("devices") or []
    if not devices:
        return None, "Vulkan found no GPU"
    items = host.get("vulkan_items") or []
    if not items:
        return None, "the host probe printed no Vulkan device items"
    best, lacking = None, []
    for i, d in enumerate(devices):
        verdicts = d.get("items") or {}
        lack = [what for ident, what in items if not verdicts.get(ident)]
        if lack:
            lacking.append(f"{d.get('name', 'a device')} lacks {', '.join(lack)}")
            continue
        rank = {2: 0, 1: 1}.get(d.get("type"), 2)        # discrete, integrated, other; the first of a rank wins
        if best is None or rank < best[0]:
            best = (rank, i, d)
    if best is None:
        return None, "; ".join(lacking)
    _, i, d = best
    uuid = d.get("pipeline_cache_uuid")
    if not uuid:
        return None, f"{d.get('name')}: the host probe gave no pipeline cache UUID"
    identity = (f"{d.get('vendor_id', 0):04x}:{d.get('device_id', 0):04x}:{d.get('driver_version', 0):08x}:{uuid}:"
                f"{'.'.join(map(str, d.get('api', [])))}")
    return Device(i, d.get("name", "?"), identity), ""


def list_keys(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise PrepareError(f"--pipelines-list {path} cannot be read: {e}") from None
    return [s for s in (line.strip() for line in lines) if s and not s.startswith("#")]


def source_files() -> list[Path]:
    """Everything the builder program is compiled from: its stamp, so that it is built again when any of it changes."""
    parts = [RUNTIME / "core", RUNTIME / "nvn", RUNTIME / "gpu"]
    return [SOURCE, *parts, *[RUNTIME / "include" / "oot2" / h for h in KEY_HEADERS]]


def key_files() -> list[Path]:
    """What the step's key holds of the sources: the files that decide a create info (the layouts, the libraries and
    compute pipelines, the capability list, the SPIR-V rewrite, the description reader, the device's features) and the
    headers they share, not the rest of the runtime. An edit elsewhere then costs no hours of building; a file missed
    here costs only a miss at launch, because the driver's cache is keyed by the create info, never by this list."""
    gpu = [RUNTIME / "gpu" / name for name in KEY_SOURCES]
    return [SOURCE, *gpu, *[RUNTIME / "include" / "oot2" / h for h in KEY_HEADERS]]


def bundled_builder(ctx: Context) -> Path | None:
    """The release bundle's pipebuild.exe (tools/bundle.py makes it with the bundle's compiler from the same sources as the
    bundle's runtime library), or None. It is used only with the bundle's own runtime library: a library named with
    --runtime-lib or OOT2_RUNTIME_LIB is another runtime, and a builder from a different runtime would write pipelines the
    game's create infos do not match."""
    exe, lib = bundle_env.pipebuild(), check.runtime_lib(ctx)
    ours = bundle_env.runtime_lib()
    if exe is None or lib is None or ours is None or Path(lib).resolve() != ours.resolve():
        return None
    return exe


class Pipelines(Step):
    id = "pipelines"
    needs = ("translate",)       # the base pack and the split it keeps

    def key(self, ctx: Context) -> str:
        device, why = usable_device(ctx.state.get("host") or {})
        listed = list_keys(getattr(ctx.options, "pipelines_list", None))
        bundled = bundled_builder(ctx)
        # The runtime's sources are not in a bundle, so its builder's bytes stand for them (a checkout's key is unchanged).
        builder = [f"bundled-builder={file_sha256(bundled)}"] if bundled else []
        # A snapshot of the step's record (the dict, as it stands now). The runner drops the record before run, and the
        # cache that record describes is the seed run may use: the runner computes the key (this call) while the record is
        # still there, so the snapshot is kept in ctx.facts for run.
        ctx.facts["pipelines_previous"] = ctx.record(self.id)
        return fingerprint(f"pipelines-{FORMAT}", f"device={device.identity if device else 'skipped: ' + why}",
                           f"runtime={','.join(f'{k}={v}' for k, v in sorted(RUNTIME_SAYS.items()))}",
                           f"skip={bool(getattr(ctx.options, 'skip_pipelines', False))}",
                           f"list={hashlib.sha256(chr(10).join(sorted(listed)).encode()).hexdigest() if listed else 'all'}",
                           *builder, HERE / "pipelines.py", HERE / "pipesched.py", *key_files())

    def current(self, ctx: Context, record: dict) -> bool:
        if record.get("skipped"):
            return True
        if record.get("runtime") != RUNTIME_SAYS:
            return False                        # built by a runtime that did not say it zeroes workgroup memory
        try:
            return (ctx.dest / OUT / CACHE).stat().st_size == record["cache_bytes"] and (ctx.dest / OUT / TIMES).is_file()
        except (OSError, KeyError, TypeError):
            return False

    def verify(self, ctx: Context, record: dict) -> bool:
        """A sample of the modules built again in a fresh process from the cache: every one must be served by it."""
        if record.get("skipped"):
            return True
        try:
            items = sample_items(ctx, record)
            warm = warm_pass(ctx, items, ctx.dest / OUT / CACHE)
            return warm["items"] == len(items) and warm["hits"] == warm["items"] and not warm["failed"]
        except (OSError, KeyError, PrepareError, pipesched.SchedError):
            return False

    def run(self, ctx: Context) -> dict:
        rep = ctx.reporter
        if getattr(ctx.options, "skip_pipelines", False):
            rep.warning("pipelines: skipped, as --skip-pipelines asked. The game starts without prepared pipelines.")
            return {"skipped": "--skip-pipelines"}
        if check.runtime_lib(ctx) is not None and bundled_builder(ctx) is None:
            rep.warning("pipelines: skipped. A prebuilt runtime library comes without the runtime's sources the pipeline "
                        "builder is linked with, and no pipeline builder of the release bundle goes with this library "
                        "(bin/pipebuild.exe of the same bundle, with the bundle's own library). The game starts without "
                        "prepared pipelines.")
            return {"skipped": "release bundle"}
        device, why = usable_device(ctx.state.get("host") or {})
        if device is None:
            if check.REQUIRE_VULKAN_ITEMS:
                raise PrepareError(f"no GPU can build the pipelines: {why}. The renderer requires a device with every "
                                   "item of architecture section 1.")
            rep.warning(f"pipelines: skipped, {why}. The game starts without prepared pipelines; build them on a "
                        "machine with a GPU that has every item the renderer requires (architecture section 1).")
            return {"skipped": why}
        out, scratch = ctx.dest / OUT, ctx.dest / check.TEMP / SCRATCH
        segments, logs = scratch / SEGMENTS, scratch / "logs"
        scratch = open_scratch(ctx, ctx.keys[self.id], scratch)
        segments.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        cold = bool(getattr(ctx.options, "pipelines_cold", False))
        # The previous cache becomes the seed (or is dropped, by name) before the step deletes its own outputs.
        seed_dir = ctx.dest / check.TEMP / pipeseed.SEED_DIR
        seed = pipeseed.settle(out / CACHE, seed_dir, expect=device.expect(), identity=device.identity,
                               runtime=RUNTIME_SAYS, previous=ctx.facts.get("pipelines_previous"), cold=cold,
                               damaged=ctx.reasons.get(self.id) == "damaged")
        say_seed(rep, seed)
        for stale in (out / CACHE, out / TIMES):
            stale.unlink(missing_ok=True)       # write-once: a step that runs again deletes its own outputs first

        pack = ctx.dest / translate.SHADERS / cache_pack.BASE_FILE
        listed = list_keys(getattr(ctx.options, "pipelines_list", None))
        wl = load_work(ctx, pack, listed)
        if wl.missing:
            raise PrepareError(f"--pipelines-list names {len(wl.missing):,} job(s) the shader cache does not hold, the "
                               f"first {wl.missing[0]}")
        if not wl.items:
            raise PrepareError("the shader cache holds no vertex, pixel or compute program to build a pipeline for")
        builder = ensure_builder(ctx, scratch)
        builder_id = builder_identity(builder)      # before the scratch folder, which holds the program, is removed
        items = pipesched.order(wl.items)
        kept = pipesched.read_progress(scratch / PROGRESS, segments)
        todo = [i for i in items if i.key not in kept]
        free = pipesched.pool.available_ram() or 0
        workers, budget, sizing = pipesched.size_workers(free, os.cpu_count() or 4,
                                                         getattr(ctx.options, "pipelines_jobs", None),
                                                         seed.bytes if seed.used else 0)
        workers = max(1, min(workers, len(todo))) if todo else 1
        by_kind = {k: sum(1 for i in items if i.kind == k) for k in pipesched.KINDS}
        rep.say(f"pipelines: {len(items):,} programs to build ({by_kind['cs']:,} compute, {by_kind['vs']:,}"
                f" vertex, {by_kind['ps']:,} pixel), {len(kept):,} kept from an interrupted run, {len(todo):,} to build on "
                f"{workers} builder processes ({sizing}); {device.name}, identity {device.identity.split(':')[3][:8]}; "
                f"not built (task {wl.not_built['ts']:,}, mesh {wl.not_built['ms']:,}), so every total is a lower bound")
        env = dict(os.environ, VK_LOADER_LAYERS_DISABLE="~implicit~", OOT2_VK_DEVICE=str(device.index))
        if cold:
            env["OOT2_VK_DRIVER_CACHE"] = "off"

        results: dict[str, pipesched.Result] = dict(kept)
        progress_file = open(scratch / PROGRESS, "a", encoding="utf-8")

        def record(rs: list[pipesched.Result]) -> None:
            for r in rs:
                results[r.key] = r
                progress_file.write(pipesched.progress_line(r) + "\n")
            progress_file.flush()
            os.fsync(progress_file.fileno())   # a power cut must not leave a record whose segment was never written

        gov = pipesched.Governor(budget)
        shown = Progress(rep, items, kept)
        serve = ["serve", "--pack", str(pack), "--segments", str(segments)]
        if seed.used:
            serve += ["--cache", str(seed.path)]
        p = pipesched.Pool(lambda wid: builder + serve, todo,
                           workers, gov, logs, segments, lambda rs, seg, size: record(rs), lambda r: record([r]),
                           next_segment=pipesched.next_segment_number(segments), env=env, on_result=shown.add,
                           expect=device.expect(), runtime=RUNTIME_SAYS,
                           # A flush replaces the builder's cache with an empty one (vkp_cache_flush), so a seeded builder
                           # that flushed on its timer would serve the seed only until the first flush and build every
                           # later unchanged program cold. It saves when it is recycled and at the end instead.
                           flush_seconds=float("inf") if seed.used else None)
        t0 = time.monotonic()
        try:
            if todo:
                p.run()
                again = [i for i, _ in p.retry]
                if again:
                    rep.say(f"pipelines: {len(again):,} item(s) got no answer; each is built again alone, within "
                            f"{pipesched.RETRY_TIMEOUT:.0f} s")
                    p.retry = []
                    p.workers = 1
                    p.todo.extend(again)
                    p.run(retry_pass=True)
        except pipesched.SchedError as e:
            raise PrepareError(str(e)) from None
        finally:
            progress_file.close()
        build_seconds = time.monotonic() - t0
        thin = thin_starts(seed, p.seeded_seen)
        seed = builders_judge_seed(rep, seed, p.cache_seen, p.seeded_seen)

        done = {k: r for k, r in results.items() if r.status == "ok"}
        failed = sorted(k for k, r in results.items() if r.status != "ok")
        if not done:
            first = failed[0] if failed else "?"
            raise PrepareError(f"no pipeline could be built from {len(items):,} programs, so the builder or the GPU is "
                               f"what is broken; the first reason: {first}: {results[first].message if failed else ''}. "
                               f"The builders' logs are in {logs}.")
        merged = merge(ctx, builder, env, scratch, segments, out / CACHE)
        warn_cache_growth(rep, seed, merged["bytes"])
        write_times(out / TIMES, results)
        shutil.rmtree(scratch, ignore_errors=True)
        pipeseed.discard(seed_dir)               # the merged cache holds it
        if failed:
            named = "; ".join(f"{k}: {results[k].message}" for k in failed[:NAMED])
            rep.warning(f"{len(failed):,} of {len(items):,} programs got no pipeline "
                        f"({status_counts(results[k].status for k in failed)}); the game stops with the reason if it "
                        f"ever needs one: {named}{'; ...' if len(failed) > NAMED else ''}")
        built = {k: sum(1 for r in done.values() if r.kind == k) for k in pipesched.KINDS}
        driver = {k: round(sum(r.ns for r in done.values() if r.kind == k) / 1e9, 1) for k in pipesched.KINDS}
        span = {}
        for k in pipesched.KINDS:
            rs = [r for r in results.values() if r.kind == k and r.started]
            span[k] = round(max(r.started + r.wall_ms / 1000 for r in rs) - min(r.started for r in rs), 1) if rs else 0.0
        rep.say(f"pipelines: {len(done):,} of {len(items):,} programs built in {build_seconds:.0f} s on {workers} workers "
                f"(compute {built['cs']:,}, vertex {built['vs']:,}, pixel {built['ps']:,}); cache {merged['bytes'] / GiB:.2f} GiB "
                f"from {merged['segments']:,} segments")
        return {
            "bytes": merged["bytes"] + (out / TIMES).stat().st_size,
            "cache_bytes": merged["bytes"],
            "scratch_peak_bytes": merged["input_bytes"] + merged["bytes"],
            "segments": merged["segments"],
            "programs": len(items),
            "built": built,
            "failed": len(failed),
            "failed_keys": failed[:100],
            "not_built": dict(wl.not_built),
            "lower_bound": True,
            "untranslated": wl.untranslated,
            "kept_items": len(kept),
            "workers": workers,
            "memory_budget": budget,
            "build_seconds": round(build_seconds, 1),
            "merge_seconds": round(merged["seconds"], 1),
            "driver_seconds": driver,
            "kind_wall_seconds": span,
            "cache_hits": sum(r.hit for r in done.values()),
            "peak_children_bytes": p.stats.peak_children_bytes,
            "min_available_bytes": p.stats.min_available_bytes,
            "peak_running": p.stats.peak_running,
            "held_back_seconds": round(p.stats.held_back_s, 1),
            "restarts": p.stats.restarts,
            "recycled": p.stats.recycled,
            "lost": p.stats.lost,
            "cold": cold,
            "seed": {**seed.record(), "starts_loaded": p.cache_seen.get("loaded", 0) - thin},   # builder starts, restarts included
            "validation": env.get("OOT2_VK_VALIDATION") or "off",
            "validation_errors": p.stats.validation_errors,
            "validation_warnings": p.stats.validation_warnings,
            "device": {"index": device.index, "name": device.name, "identity": device.identity},
            "builder": builder_id,
            "runtime": dict(p.runtime_seen) or dict(RUNTIME_SAYS),     # no worker ran when every item was kept: the key held it
        }


# --- the seed ---------------------------------------------------------------------------------------------------

def say_seed(rep, seed: pipeseed.Seed) -> None:
    if seed.outcome == pipeseed.KEPT:
        rep.say(f"pipelines: the previous cache ({seed.bytes / GiB:.2f} GiB, {seed.source}) is the seed: the programs it "
                "holds are served from it and only the others build cold")
    elif seed.outcome == pipeseed.DROPPED:
        rep.warning(f"pipelines: the previous cache ({seed.bytes:,} bytes, {seed.source}) is dropped, not used as a seed: "
                    f"{seed.reason}. It is deleted and every program builds from an empty cache.")
    elif seed.outcome == pipeseed.OFF:
        rep.say(f"pipelines: no seed: {seed.reason}")


def thin_starts(seed: pipeseed.Seed, held: list[int]) -> int:
    """How many builders that said they loaded the seed hold under half of it (the ready line's seeded=, the bytes the driver
    holds once it has loaded the cache). The header alone passes the builder's own judgement (cache=loaded); a body that is
    cut or damaged is ignored by the driver, which then serves nothing, and only this number tells."""
    return sum(1 for n in held if n * 2 < seed.bytes) if seed.used else 0


def builders_judge_seed(rep, seed: pipeseed.Seed, seen: dict[str, int], held: list[int] | None = None) -> pipeseed.Seed:
    """The builders judge the seed again when they start (pipebuild serve --cache) and say what they made of it in the
    ready line. One that did not load it, or loaded it and holds under half of its bytes, started from an empty cache or
    nearly, so the seed did not do its work: say so."""
    refused = {state: n for state, n in seen.items() if state != "loaded"}
    thin = thin_starts(seed, held or [])
    if not seed.used or not (refused or thin):
        return seed
    said = [f"{n} said {state}" for state, n in sorted(refused.items())]
    if thin:
        said.append(f"{thin} said loaded and hold {min(n for n in held if n * 2 < seed.bytes):,} bytes or less of the "
                    f"{seed.bytes:,}-byte seed, so the driver ignored its body")
    reason = f"the builder judged it unusable when it started ({', '.join(said)}) and began from an empty cache"
    rep.warning(f"pipelines: the seed was not used by every builder: {reason}. The programs it holds built cold again.")
    return pipeseed.Seed(pipeseed.DROPPED, seed.bytes, seed.source, reason)


GROWTH_WARNING = 1.5


def warn_cache_growth(rep, seed: pipeseed.Seed, merged_bytes: int) -> None:
    """The merge keeps the entries of modules that no longer exist, so each key change that rebuilt much adds a copy of
    what it rebuilt. Name it when the cache is over GROWTH_WARNING times the seed it came from."""
    if seed.used and merged_bytes > GROWTH_WARNING * seed.bytes:
        rep.warning(f"pipelines: the cache grew from {seed.bytes / GiB:.2f} to {merged_bytes / GiB:.2f} GiB: the seed's "
                    "entries of modules that no longer exist are kept, so every key change that rebuilds much adds a "
                    "copy. `--pipelines-cold` builds a cache with none of them.")


# --- the work ---------------------------------------------------------------------------------------------------

def load_work(ctx: Context, pack_path: Path, only: list[str] | None) -> pipesched.WorkList:
    if not pack_path.is_file():
        raise PrepareError(f"the shader cache {pack_path} is missing: the translate step writes it")
    try:
        pack = cache_pack.open_pack(pack_path, role=cache_pack.BASE)
        jobs = translate.load_jobs(ctx.dest / translate.SHADERS / translate.SPLIT)
    except (OSError, cache_pack.PackError, ValueError) as e:
        raise PrepareError(f"the shader cache or its job list cannot be read ({e}). Run oot2 prepare again.") from None
    stages = {k: j.get("stage", "") for k, j in jobs.items()}
    return pipesched.work_list(((r.key, r.module_size, r.failed) for r in pack.rows), stages, only)


def sample_items(ctx: Context, record: dict) -> list[pipesched.Item]:
    """An even spread of SAMPLE modules the step built, in order, for verify."""
    pack = ctx.dest / translate.SHADERS / cache_pack.BASE_FILE
    wl = load_work(ctx, pack, None)
    items = pipesched.order(wl.items)
    step = max(1, len(items) // SAMPLE)
    return items[::step][:SAMPLE]


def open_scratch(ctx: Context, key: str, scratch: Path) -> Path:
    """The step's scratch folder, kept when it belongs to this key and emptied otherwise."""
    stamp = scratch / STAMP
    try:
        same = stamp.read_text(encoding="utf-8").strip() == key
    except OSError:
        same = False
    if not same:
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)
        stamp.write_text(key + "\n", encoding="utf-8")
    return scratch


def write_times(path: Path, results: dict[str, pipesched.Result]) -> None:
    fields = [f for f in pipesched.PROGRESS_FIELDS if f != "segment"]
    lines = ["\t".join(fields)]
    for k in sorted(results):
        r = results[k]
        row = {"status": r.status, "key": r.key, "kind": r.kind, "ns": r.ns, "hit": r.hit, "module_bytes": r.module_bytes,
               "wall_ms": f"{r.wall_ms:.1f}", "concurrency": r.concurrency, "peak_ws": r.peak_ws,
               "message": r.message.replace("\t", " ").replace("\n", " ")}
        lines.append("\t".join(str(row[f]) for f in fields))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Progress:
    """A line every 15 seconds while the pool builds: the step is silent for hours otherwise."""

    def __init__(self, rep, items: list[pipesched.Item], kept: dict):
        self.rep, self.total = rep, len(items)
        self.seen = set(kept)             # an item built again after a lost worker, or tried again alone, counts once
        self.done = len(kept)
        self.by_kind = {k: sum(1 for i in items if i.kind == k) for k in pipesched.KINDS}
        self.built = {k: sum(1 for r in kept.values() if r.kind == k) for k in pipesched.KINDS}
        self.t0 = time.monotonic()

    def add(self, r: pipesched.Result) -> None:
        if r.status not in ("ok", "fail") or r.key in self.seen:
            return
        self.seen.add(r.key)
        self.done += 1
        self.built[r.kind] = self.built.get(r.kind, 0) + 1
        free = pipesched.pool.available_ram()
        text = ", ".join(f"{pipesched.KIND_NAMES[k]} {self.built[k]:,} of {self.by_kind[k]:,}" for k in pipesched.KINDS)
        self.rep.progress("pipelines", f"pipelines: {self.done:,} of {self.total:,} ({text})"
                          + (f", {free / GiB:.1f} GiB free" if free else "") + f", {time.monotonic() - self.t0:.0f} s",
                          every=15.0, final=self.done >= self.total)


# --- the builder ------------------------------------------------------------------------------------------------

def ensure_builder(ctx: Context, scratch: Path) -> list[str]:
    """The command that starts the builder (the program, or the stand-in a test gave through the hidden option
    `pipelines_builder`), compiled again only when its sources or the compiler changed."""
    given = getattr(ctx.options, "pipelines_builder", None)
    if given:
        return list(given)
    bundled = bundled_builder(ctx)
    if bundled is not None:                 # a release bundle: the program tools/bundle.py compiled, not the sources
        return [str(bundled)]
    from .program import toolchain          # the compile step's compiler and the Vulkan SDK
    tool = toolchain(ctx)
    if tool.vulkan is None or not tool.glslang:
        raise PrepareError("the pipeline builder needs the Vulkan headers with Volk and glslang (--vulkan-sdk)")
    out = scratch / BUILDER
    exe = out / ("pipebuild.exe" if os.name == "nt" else "pipebuild")
    stamp = out / "stamp.txt"
    want = fingerprint(build_game.tool_id(tool.cc), *source_files())
    try:
        if exe.is_file() and stamp.read_text(encoding="utf-8").strip() == want:
            return [str(exe)]
    except OSError:
        pass
    compile_builder(tool.cc, tool.vulkan, tool.glslang, out)
    stamp.write_text(want + "\n", encoding="utf-8")
    return [str(exe)]


def compile_builder(cc: str, vulkan: Path, glslang: str, out: Path, env: dict | None = None) -> Path:
    """Compile the builder program into out (pipebuild.exe, or pipebuild off Windows) from the runtime's core, nvn and
    gpu sources, the runtime's shaders embedded by glslang, and the placeholder layout header the builder needs (it loads
    no game). The release bundle's builder (tools/bundle.py) is made by this same function, so a bundle's pipelines are
    the ones a checkout's step would build."""
    out = Path(out)
    vulkan = Path(vulkan)
    exe = out / ("pipebuild.exe" if os.name == "nt" else "pipebuild")
    out.mkdir(parents=True, exist_ok=True)
    exe.unlink(missing_ok=True)
    layout.write(layout.place({"main": MODULE_SIZE}), out)
    try:
        glue = build_game.write_shader_glue(RUNTIME, out, glslang)
    except build_game.BuildError as e:
        raise PrepareError(str(e)) from None
    sources = [str(SOURCE)]
    for part in ("core", "nvn", "gpu"):
        sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
    sources.append(str(glue))
    extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread", "-ldl", "-lm"]
    cmd = [cc, "-std=gnu11", "-O1", "-march=x86-64-v3", "-fno-strict-aliasing", "-w",
           "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "nvn"), "-I", str(out),
           "-I", str(vulkan), "-I", str(vulkan / "Volk"), *sources, *extra, "-o", str(exe)]
    done = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env)
    if done.returncode or not exe.is_file():
        raise PrepareError(f"the pipeline builder could not be compiled with {cc}:\n"
                           f"{(done.stderr or done.stdout).strip()[-800:]}")
    return exe


def builder_identity(command: list[str]) -> dict:
    exe = Path(command[-1])
    try:
        return {"path": str(exe), "sha256": hashlib.sha256(exe.read_bytes()).hexdigest()}
    except OSError:
        return {"path": " ".join(command)}


def merge(ctx: Context, builder: list[str], env: dict, scratch: Path, segments: Path, cache: Path) -> dict:
    """One cache file out of the segments (the builder's merge command). The merge holds the whole cache in memory
    twice over, so a machine short of memory finds out here, after every build: the segments stay, and a rerun only
    merges."""
    files = sorted(segments.glob("seg-*.bin"))
    listing = scratch / "segments.txt"
    listing.write_text("\n".join(str(p) for p in files) + "\n", encoding="utf-8")
    ctx.reporter.say(f"pipelines: merging {len(files):,} segments ({sum(p.stat().st_size for p in files) / GiB:.2f} GiB)")
    t0 = time.monotonic()
    try:
        done = subprocess.run(builder + ["merge", "--out", str(cache), "--list", str(listing)], capture_output=True,
                              text=True, errors="replace", env=env, timeout=MERGE_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PrepareError(f"merging the pipeline cache segments took more than {MERGE_TIMEOUT} s. They are kept in "
                           f"{segments}; run oot2 prepare again to merge them.") from None
    seconds = time.monotonic() - t0
    word, rest = pipesched.parse_reply(done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "")
    if done.returncode or word != "merged" or len(rest) < 3 or not cache.is_file():
        raise PrepareError(f"the pipeline cache segments could not be merged (exit {done.returncode}): "
                           f"{(done.stderr or done.stdout).strip()[-500:]}. They are kept in {segments}; run oot2 "
                           "prepare again to merge them.")
    return {"segments": int(rest[0]), "input_bytes": int(rest[1]), "bytes": int(rest[2]), "seconds": seconds}


def warm_pass(ctx: Context, items: list[pipesched.Item], cache: Path, workers: int = 1) -> dict:
    """The items built again by fresh processes seeded with the cache: how many it served and how long that took (the
    time to create every pipeline again at a launch, which a launch needs of the programs it uses)."""
    device, why = usable_device(ctx.state.get("host") or {})
    if device is None:
        raise PrepareError(why)
    scratch = ctx.dest / check.TEMP / SCRATCH
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "warm-logs").mkdir(exist_ok=True)
    builder = ensure_builder(ctx, scratch)
    pack = ctx.dest / translate.SHADERS / cache_pack.BASE_FILE
    env = dict(os.environ, VK_LOADER_LAYERS_DISABLE="~implicit~", OOT2_VK_DEVICE=str(device.index))
    rows: list[pipesched.Result] = []
    failed: list[pipesched.Result] = []
    gov = pipesched.Governor(8 * GiB)
    p = pipesched.Pool(lambda wid: builder + ["serve", "--pack", str(pack), "--cache", str(cache), "--segments",
                                              str(scratch / "warm-segments")],
                       pipesched.order(items), workers, gov, scratch / "warm-logs", scratch / "warm-segments",
                       lambda rs, seg, size: None, failed.append, env=env, on_result=rows.append, final_flush=False,
                       flush_seconds=float("inf"), expect=device.expect(),
                       runtime=RUNTIME_SAYS)
    (scratch / "warm-segments").mkdir(exist_ok=True)
    t0 = time.monotonic()
    p.run()
    seconds = time.monotonic() - t0
    ok = [r for r in rows if r.status == "ok"]
    return {"items": len(rows), "hits": sum(r.hit for r in ok), "failed": len(rows) - len(ok), "seconds": seconds,
            "ns": sum(r.ns for r in ok), "peak_children_bytes": p.stats.peak_children_bytes,
            "cache_bytes": Path(cache).stat().st_size}
