# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The seed of the pipelines step: the previous pipeline cache, kept when the step's key changes.

A change of the step's key (a translator fix, an edit to one of the runtime files the builder is made from, a new
bundle) used to delete `pipelines/pipeline.cache` and build every library from an empty cache. The driver answers by create
info, so the previous cache serves every program whose module and create info did not change in a lookup, a small
fraction of the time of a build, and only the changed programs build cold. This file decides whether the previous cache
may be used for that, and moves it out of the way of the step's own output:

  temp/pipelines-seed/pipeline.cache   the seed (moved from pipelines/pipeline.cache, which the step then writes anew)
  temp/pipelines-seed/seed.json        what was claimed of it when it was taken: size, device identity, runtime

The seed is beside the step's scratch folder, not inside it, because that folder is emptied whenever the key changes
and a second key change after an interrupted run must still find the seed. It is removed when the step has merged its
cache. It is never trusted: it is read only after these checks, and the first that fails drops it, names the reason and
deletes it, and the step builds from an empty cache as it always did:

  1. its header is a version-one pipeline cache header that names this vendor, device and pipeline cache UUID (the same
     rule as vkp_cache_judge in runtime/gpu/vk_pipeline.c: a cache is valid only for the driver that wrote it),
  2. it is as large as the record of the run that wrote it says (a file cut short or changed since is not that cache),
  3. the run that wrote it used the same runtime (the runtime zeroes workgroup memory before the driver sees a compute
     module, so a cache of a runtime that did not holds other modules than a launch asks for),
  4. that run used this device and driver,
  5. the step's full check did not just fail on it (the runner says "damaged": that cache did not serve the programs it
     was built for, so it is the last thing to feed the driver).

Checks 2 to 4 need the record of the run that wrote the cache. A cache with no record at all (a state file that was lost
while the cache stayed) is KEPT on its header alone: the builder judges the file again, and a cache of another runtime
only costs the programs it holds wrongly, built cold, never a wrong answer. The seed of an interrupted run is held to the
claims kept beside it instead, and is dropped when those are gone.

The builder judges the file once more when it starts (pipebuild serve --cache), and the pool reports what it said.

A seeded builder's first segment holds the whole seed (a Vulkan cache cannot be told to write only what it learned), so
the merge unions as many copies of the seed as the segments written. A flush hands the builder an empty cache, so a seeded
builder does not flush on a timer (it would serve the seed only until its first flush): it saves when it is recycled and
at the end, each time with the seed and what it built since it started. A run that is stopped therefore keeps nothing
built since it started, only the seed, which serves in a lookup what it holds and so costs the changed programs alone. The
step counts the seed in the memory it gives each worker, and the cache it writes keeps the entries of modules that no
longer exist: it grows by what a key change built, never shrinks, and the record says both sizes.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

# VkPipelineCacheHeaderVersionOne: headerSize, headerVersion, vendorID, deviceID, pipelineCacheUUID[16].
HEADER = struct.Struct("<IIII16s")
HEADER_VERSION_ONE = 1
SEED_DIR = "pipelines-seed"
CACHE = "pipeline.cache"
META = "seed.json"

NONE, KEPT, DROPPED, OFF = "none", "kept", "dropped", "off"


@dataclass(frozen=True)
class Seed:
    outcome: str                 # none (no previous cache), kept, dropped (named by reason) or off (--pipelines-cold)
    bytes: int = 0
    source: str = ""             # where the cache came from
    reason: str = ""             # why it was dropped or not used
    path: Path | None = None     # the seed file, when kept

    @property
    def used(self) -> bool:
        return self.outcome == KEPT

    def record(self) -> dict:
        """What the step's record keeps of it."""
        return {k: v for k, v in (("outcome", self.outcome), ("bytes", self.bytes), ("source", self.source),
                                  ("reason", self.reason)) if v or k == "outcome"}


def judge_header(head: bytes, size: int, vendor: int, device: int, uuid: str) -> str:
    """Why a file of `size` bytes that begins with `head` is not a pipeline cache of this device and driver, or ''. The
    rule of vkp_cache_judge (runtime/gpu/vk_pipeline.c), with the reason said."""
    if size < HEADER.size or len(head) < HEADER.size:
        return f"it is {size} bytes, shorter than the {HEADER.size}-byte header of a pipeline cache (cut short or damaged)"
    header_size, version, got_vendor, got_device, got_uuid = HEADER.unpack_from(head)
    if header_size < HEADER.size or header_size > size:
        return (f"its header says {header_size} bytes of header in a file of {size} bytes "
                "(cut short, or not a pipeline cache)")
    if version != HEADER_VERSION_ONE:
        return f"its header version is {version}, not {HEADER_VERSION_ONE} (not a pipeline cache)"
    if (got_vendor, got_device) != (vendor, device) or got_uuid.hex() != uuid.lower():
        return (f"it was written by another device or driver (vendor {got_vendor:04x}, device {got_device:04x}, "
                f"cache UUID {got_uuid.hex()}; this device is vendor {vendor:04x}, device {device:04x}, "
                f"cache UUID {uuid.lower()})")
    return ""


def claims_of(record: dict | None) -> dict | None:
    """What the step's record of the run that wrote the cache claims of it, or None when there is no record to hold it to
    (a first run, or a run that was skipped and wrote nothing)."""
    if not record or record.get("skipped"):
        return None
    return {"bytes": record.get("cache_bytes"), "identity": (record.get("device") or {}).get("identity"),
            "runtime": record.get("runtime")}


def judge_claims(claims: dict, size: int, identity: str, runtime: dict) -> str:
    """Why the cache is not the one this run may be seeded with, from what was claimed of it, or ''."""
    want = claims.get("bytes")
    if isinstance(want, int) and want != size:
        return f"it is {size:,} bytes and the record of the run that wrote it says {want:,} (cut short or changed since)"
    if claims.get("runtime") != runtime:
        return ("the run that wrote it used another runtime than this one "
                f"({claims.get('runtime')!r}, this one {runtime!r}): its compute modules are not the ones a launch asks for")
    then = claims.get("identity")
    if then and then.split(":")[:4] != identity.split(":")[:4]:
        return f"the run that wrote it used another device or driver ({then}; this one is {identity})"
    return ""


def _read_head(path: Path) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read(HEADER.size)
    except OSError:
        return None


def _read_meta(seed_dir: Path) -> dict | None:
    try:
        meta = json.loads((seed_dir / META).read_text(encoding="utf-8"))
        return meta if isinstance(meta, dict) else None
    except (OSError, ValueError):
        return None


def settle(out_cache: Path, seed_dir: Path, *, expect: dict[str, str], identity: str, runtime: dict,
           previous: dict | None, cold: bool = False, damaged: bool = False) -> Seed:
    """Decide what becomes of the cache the step is about to replace. `out_cache` is the previous run's pipeline.cache
    (moved into `seed_dir` when it is kept, deleted when it is dropped), `seed_dir` may already hold the seed of a run that
    was interrupted, `expect` is Device.expect() (vendor, device and cache_uuid of the device the builders will open),
    `identity` is the device's identity string, `runtime` is what the builder must say of its runtime, and `previous` is
    the step's record of the run that wrote out_cache (None for a first run). `damaged` says the runner's full check just
    failed on out_cache: that cache is dropped by name whatever its header says."""
    kept = seed_dir / CACHE
    if cold:
        out_cache.unlink(missing_ok=True)
        shutil.rmtree(seed_dir, ignore_errors=True)
        return Seed(OFF, reason="a cold build (--pipelines-cold) is timed with no seed")
    if out_cache.is_file():
        source, path, claims = "the previous run's pipeline.cache", out_cache, claims_of(previous)
    elif kept.is_file():
        source, path, claims = "the seed of an interrupted run", kept, _read_meta(seed_dir)
    else:
        shutil.rmtree(seed_dir, ignore_errors=True)
        return Seed(NONE, reason="there is no previous cache")
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    head = _read_head(path)
    if head is None:
        why = "it cannot be read"
    elif damaged and path is not kept:
        why = "its full check failed (it did not serve the programs it was built for)"
    elif claims is None and path is kept:
        why = "what was claimed of it when it was kept is gone"
    else:
        why = judge_header(head, size, int(expect["vendor"], 16), int(expect["device"], 16), expect["cache_uuid"])
        if not why and claims is not None:
            why = judge_claims(claims, size, identity, runtime)
    if why:
        path.unlink(missing_ok=True)
        shutil.rmtree(seed_dir, ignore_errors=True)
        return Seed(DROPPED, size, source, why)
    if path is not kept:
        shutil.rmtree(seed_dir, ignore_errors=True)
        seed_dir.mkdir(parents=True, exist_ok=True)
        os.replace(path, kept)
    (seed_dir / META).write_text(json.dumps({"bytes": size, "identity": identity, "runtime": runtime}), encoding="utf-8")
    return Seed(KEPT, size, source, path=kept)


def discard(seed_dir: Path) -> None:
    """The step has merged its cache: the seed is part of it now."""
    shutil.rmtree(seed_dir, ignore_errors=True)
