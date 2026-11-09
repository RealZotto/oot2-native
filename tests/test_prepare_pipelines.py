# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The pipelines step of `oot2 prepare` and its scheduling (tools/prepare/pipelines.py, pipesched.py; the pipeline step), with
no GPU: the builder is a stand-in script that speaks the real builder's line protocol (tools/pipebuild/pipebuild.c),
builds nothing and records what it was asked in a log the tests read.

What is held here: the order of the work list (compute first, longest first, the key last), the worker count from the
free memory, the governor's rules (a budget, six compute builds at once, the largest alone), the step's key (a driver
update runs it again, the worker count does not), the device gate (skipped with a warning, or a failure once the
Vulkan items are required), resume (a segment's items are recorded only once it is saved; a lost segment is built
again), the failure rule (a named refusal is recorded and not retried, a dead worker's item is tried again alone,
nothing built fails the step), and the warm pass. The real builder is tested on a device in test_pipebuild.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import cache_pack as cp  # noqa: E402
from prepare import check, pipelines, pipesched  # noqa: E402
from prepare.pipelines import Pipelines  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, Step, run_steps  # noqa: E402

TRANSLATOR = hashlib.sha256(b"synthetic translator of the pipeline step").hexdigest()

STANDIN = r'''
import json, os, struct, sys, time
cfg = json.load(open(os.environ["STANDIN_CONFIG"])) if os.environ.get("STANDIN_CONFIG") else {}
log = os.environ.get("STANDIN_LOG")

def say(*words):
    print(*words, flush=True)

def note(text):
    if log:
        with open(log, "a") as f:
            f.write(text + "\n")

def head():
    """The 32-byte header of a Vulkan pipeline cache for the device the stand-in says it is (what the real builder's cache
    files begin with), so that the step's checks of a seed have a header to read."""
    return struct.pack("<IIII16s", 32, 1, int(cfg.get("vendor", "10de"), 16), int(cfg.get("device", "2486"), 16),
                       bytes.fromhex(cfg.get("uuid", "ab" * 16)))

def serve(args):
    opts = dict(zip(args[::2], args[1::2]))
    segments = opts["--segments"]
    seeded, state, seeded_bytes = [], "none", 0
    if "--cache" in opts and os.path.exists(opts["--cache"]):
        raw = open(opts["--cache"], "rb").read()
        state, seeded_bytes = cfg.get("cache_state", "loaded"), len(raw)
        if state == "loaded":
            seeded = raw[32:].decode().split()
    note("serve " + ("--cache" if "--cache" in opts else "no-cache") + " " + state)
    if cfg.get("start_fail"):
        sys.stderr.write("FATAL: gfx: vulkan: stand-in has no device\n")
        sys.exit(70)
    say(f"ready vendor={cfg.get('vendor', '10de')} device={cfg.get('device', '2486')} driver={cfg.get('driver', '0009a000')} "
        f"api=1.4.351 cache_uuid={cfg.get('uuid', 'ab' * 16)} cache={state} seeded={seeded_bytes}"
        + ("" if cfg.get("older_runtime") else f" zero_workgroup={cfg.get('zero_workgroup', 1)}")
        + " name=Stand-in")
    built = list(seeded)               # the real builder's first segment holds the seed it was given
    for line in sys.stdin:
        w = line.split()
        if not w:
            continue
        if w[0] == "build":
            kind, key = w[1], w[2]
            note(f"build {kind} {key}")
            if key in cfg.get("reject", []):          # a command it refuses, with the process staying alive
                say("fail", kind, key, "the stand-in does not take this key")
                continue
            if key in cfg.get("crash", []):           # dies with nothing on stderr, every time
                os._exit(9)
            if key in cfg.get("transient_once", []):  # a named fatal about the machine, the first time only
                marker = os.path.join(segments, ".transient-" + key)
                if not os.path.exists(marker):
                    open(marker, "w").close()
                    msg = "FATAL: gfx: vulkan: vkCreateComputePipelines failed: VK_ERROR_OUT_OF_HOST_MEMORY"
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                    say("died - 0", msg)
                    os._exit(70)
            if key in cfg.get("fatal", []):
                msg = "FATAL: gfx: vulkan: stand-in refuses " + key
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
                name, size = "-", 0
                if built and not cfg.get("no_salvage"):     # the real builder's stop hook saves what it built
                    name = "seg-d%d" % os.getpid()
                    data = "".join(k + "\n" for k in built)
                    with open(os.path.join(segments, name + ".bin"), "w") as f:
                        f.write(data)
                    note("flush " + name + " " + " ".join(built))
                    size = len(data)
                say("died", name, size, msg)
                os._exit(70)
            if key in cfg.get("crash_once", []):
                marker = os.path.join(segments, ".crashed-" + key)
                if not os.path.exists(marker):
                    open(marker, "w").close()
                    os._exit(9)
            if key in cfg.get("hang", []):
                time.sleep(60)
            time.sleep(cfg.get("delay", 0))
            if key not in built:
                built.append(key)
            say("ok", kind, key, 1000 + len(key), 1 if key in seeded and key not in cfg.get("cold_keys", []) else 0, 100 * (1 + len(key)))
        elif w[0] == "flush":
            if cfg.get("flush_fail"):
                say("fail flush", w[1], "disk full")
                continue
            path = os.path.join(segments, w[1] + ".bin")
            data = "".join(k + "\n" for k in built)
            with open(path + ".tmp", "w") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            note("flush " + w[1] + " " + " ".join(built))
            built = []
            say("saved", w[1], len(data))
        elif w[0] == "quit":
            break
        else:
            say("fail", w[0], "unknown")
    say("bye validation_errors 0 validation_warnings 0")

def merge(args):
    opts = dict(zip(args[::2], args[1::2]))
    files = open(opts["--list"]).read().split("\n")
    files = [f for f in files if f]
    keys, total = [], 0
    for f in files:
        data = open(f).read()
        total += len(data)
        keys += [k for k in data.split() if k not in keys]
    with open(opts["--out"], "wb") as out:
        out.write(head() + "".join(k + "\n" for k in keys).encode())
    say("merged", len(files), total, os.path.getsize(opts["--out"]))

{"serve": serve, "merge": merge}[sys.argv[1]](sys.argv[2:])
'''


def module(n: int, words: int = 8) -> bytes:
    return struct.pack(f"<{5 + words}I", 0x07230203, 0x00010300, 0, 1, 0, *([n] * words))


def description(key: str, mod: bytes) -> bytes:
    return json.dumps({"schema": cp.DESC_SCHEMA, "key": key, "translator_id": TRANSLATOR,
                       "answers": {"tex": {}, "img": {}, "attr": {}, "fout": {}}, "spv_bytes": len(mod),
                       "spv_sha256": hashlib.sha256(mod).hexdigest()}).encode()


def cache_keys(path: Path) -> list[str]:
    """The keys a stand-in cache file holds: the 32-byte header, then one key a line."""
    return path.read_bytes()[32:].decode().split()


def device(*, ok: bool = True, kind: int = 2, uuid: str = "ab" * 16, driver: int = 0x9a000, name: str = "Test GPU") -> dict:
    return {"name": name, "api": [1, 4, 351], "driver_version": driver, "vendor_id": 0x10de, "device_id": 0x2486,
            "pipeline_cache_uuid": uuid, "type": kind, "items": {"vulkan_1_3": True, "mesh_shaders": ok}}


def host(*devices: dict) -> dict:
    return {"vulkan": {"loader": True, "instance": 0, "devices": list(devices) or [device()]},
            "vulkan_items": [["vulkan_1_3", "Vulkan 1.3"], ["mesh_shaders", "mesh and task shaders"]]}


class FakeTranslate(Step):
    id = "translate"


class Env:
    """A destination folder with a pack and a split of synthetic jobs, a stand-in builder, and a log of its work."""

    def __init__(self, test: unittest.TestCase, jobs: dict[str, tuple[str, int]] | None = None, failed=(), config=None):
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-pipes-"))
        test.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.dest = self.tmp / "dest"
        self.log = self.tmp / "standin.log"
        self.script = self.tmp / "standin.py"
        self.script.write_text(STANDIN, encoding="utf-8")
        self.config = self.tmp / "standin.json"
        self.config.write_text(json.dumps(config or {}), encoding="utf-8")
        jobs = jobs if jobs is not None else DEFAULT_JOBS
        shaders = self.dest / "shaders"
        (shaders / "split").mkdir(parents=True)
        entries = []
        with open(shaders / "split" / "jobs.jsonl", "w", encoding="utf-8") as f:
            for key, (stage, words) in sorted(jobs.items()):
                f.write(json.dumps({"key": key, "stage": stage}) + "\n")
                mod = module(len(entries), words)
                if key in failed:
                    entries.append(cp.Entry(key, None, b"fail: synthetic", failed=True))
                else:
                    entries.append(cp.Entry(key, mod, description(key, mod)))
        cp.write_pack(shaders / cp.BASE_FILE, entries, translator=TRANSLATOR)
        self.patch = mock.patch.dict(os.environ, {"STANDIN_CONFIG": str(self.config), "STANDIN_LOG": str(self.log)})
        self.patch.start()
        test.addCleanup(self.patch.stop)

    def standin(self, **settings) -> None:
        """Change what the stand-in builder does or says (its configuration file is read by each process)."""
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config.update(settings)
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def options(self, **extra) -> argparse.Namespace:
        base = {"pipelines_builder": [sys.executable, "-I", str(self.script)], "pipelines_jobs": 1,
                "pipelines_list": None, "pipelines_cold": False, "skip_pipelines": False, "vulkan_sdk": None}
        base.update(extra)
        return argparse.Namespace(**base)

    def context(self, hosts: dict | None = None, **extra) -> Context:
        ctx = Context(self.dest, self.options(**extra), Reporter(io.StringIO(), io.StringIO()))
        ctx.state["host"] = hosts if hosts is not None else host()
        return ctx

    def run(self, ctx: Context | None = None, steps=None) -> tuple[str, Context]:
        ctx = ctx or self.context()
        return run_steps(steps or [FakeTranslate(), Pipelines()], ctx), ctx

    def builds(self) -> list[tuple[str, str]]:
        if not self.log.exists():
            return []
        return [tuple(line.split()[1:3]) for line in self.log.read_text().splitlines() if line.startswith("build ")]

    def say(self, ctx: Context) -> str:
        return ctx.reporter.out.getvalue() + ctx.reporter.err.getvalue()


# Eight compute, five vertex, five pixel, two task and two mesh programs, with sizes that make the order testable.
DEFAULT_JOBS = {
    "c000000000000001.00000000": ("cs", 4), "c000000000000002.00000000": ("cs", 40), "c000000000000003.00000000": ("cs", 12),
    "c000000000000004.00000000": ("cs", 12), "v000000000000001.00000000": ("vs", 30), "v000000000000002.00000000": ("vs", 6),
    "v000000000000003.00000000": ("vs", 18), "p000000000000001.00000000": ("ps", 24), "p000000000000002.00000000": ("ps", 8),
    "p000000000000003.00000000": ("ps", 50), "m000000000000001.00000000": ("ms", 10), "t000000000000001.00000000": ("ts", 10),
}


class WorkListTest(unittest.TestCase):
    def test_compute_first_then_the_longest_module_first_and_the_key_last(self) -> None:
        items = [pipesched.Item("b", "vs", 100), pipesched.Item("a", "vs", 100), pipesched.Item("z", "cs", 5),
                 pipesched.Item("y", "cs", 50), pipesched.Item("c", "ps", 300)]
        ordered = pipesched.order(items)
        self.assertEqual([i.key for i in ordered], ["y", "z", "c", "a", "b"])
        self.assertEqual(pipesched.order(reversed(items)), ordered)

    def test_task_and_mesh_programs_are_counted_not_built_and_failures_are_left_out(self) -> None:
        rows = [("a", 10, False), ("b", 10, False), ("c", 10, True), ("d", 10, False), ("e", 10, False), ("f", 10, False)]
        stages = {"a": "cs", "b": "ms", "c": "vs", "d": "ts", "e": "xx", "f": "ps"}
        wl = pipesched.work_list(rows, stages)
        self.assertEqual([i.key for i in wl.items], ["a", "f"])
        self.assertEqual(wl.not_built, {"ms": 1, "ts": 1})
        self.assertEqual((wl.untranslated, wl.unstaged), (1, 1))

    def test_a_list_limits_the_work_and_names_what_the_pack_lacks(self) -> None:
        rows = [("a", 1, False), ("b", 1, False)]
        wl = pipesched.work_list(rows, {"a": "cs", "b": "ps"}, only=["b", "zzz"])
        self.assertEqual([i.key for i in wl.items], ["b"])
        self.assertEqual(wl.missing, ["zzz"])


class SizingTest(unittest.TestCase):
    def test_free_memory_caps_the_workers(self) -> None:
        workers, budget, _ = pipesched.size_workers(2 * (1 << 30), 12)
        self.assertEqual(workers, 3)                      # 0.8 * 2 GiB at 512 MiB a worker
        self.assertEqual(budget, int(2 * (1 << 30) * 0.8))
        self.assertEqual(pipesched.size_workers(64 << 30, 12)[0], 12)
        self.assertEqual(pipesched.size_workers(64 << 30, 32)[0], pipesched.MAX_WORKERS)

    def test_the_player_can_choose_and_an_unknown_free_memory_falls_back_to_the_cpus(self) -> None:
        self.assertEqual(pipesched.size_workers(1 << 30, 12, given=5)[0], 5)
        self.assertEqual(pipesched.size_workers(0, 8)[0], 8)

    def test_the_largest_modules_reserve_the_most_and_run_alone(self) -> None:
        small, big = pipesched.Item("s", "cs", 50_000), pipesched.Item("b", "cs", 2_000_000)
        self.assertLess(pipesched.expected_bytes(small), pipesched.expected_bytes(big))
        self.assertEqual(pipesched.expected_bytes(pipesched.Item("x", "cs", 1 << 40)), pipesched.CAP_BYTES)
        self.assertFalse(pipesched.runs_alone(small))
        self.assertTrue(pipesched.runs_alone(big))


class GovernorTest(unittest.TestCase):
    def gov(self, budget: int, free: int = 0, **kw) -> pipesched.Governor:
        return pipesched.Governor(budget, available=lambda: free, **kw)

    def test_the_reserved_memory_stays_inside_the_budget_and_the_first_item_always_starts(self) -> None:
        item = pipesched.Item("a", "vs", 100_000)
        need = pipesched.expected_bytes(item)
        g = self.gov(need * 2 + 1)
        self.assertEqual(g.acquire(item), 1)               # a budget of two items admits two
        self.assertEqual(g.acquire(item), 2)
        self.assertFalse(g._admit(item, need, False))
        g.release(item)
        self.assertTrue(g._admit(item, need, False))
        tiny = self.gov(1)
        self.assertEqual(tiny.acquire(item), 1)            # a budget under one item cannot stop the step

    def test_compute_builds_are_held_to_a_few_at_once(self) -> None:
        cs, vs = pipesched.Item("a", "cs", 1000), pipesched.Item("b", "vs", 1000)
        g = self.gov(1 << 40, compute_at_once=2)
        g.acquire(cs)
        g.acquire(cs)
        self.assertFalse(g._admit(cs, pipesched.expected_bytes(cs), False))
        self.assertTrue(g._admit(vs, pipesched.expected_bytes(vs), False))

    def test_an_item_that_runs_alone_waits_for_the_others_and_holds_new_ones_back(self) -> None:
        big, small = pipesched.Item("big", "cs", 3_000_000), pipesched.Item("s", "cs", 1000)
        g = self.gov(1 << 40)
        g.acquire(small)
        self.assertFalse(g._admit(big, pipesched.expected_bytes(big), True))     # one is running
        g.alone_waiting = 1
        self.assertFalse(g._admit(small, pipesched.expected_bytes(small), False))   # held back for the big one
        g.alone_waiting = 0
        g.release(small)
        self.assertTrue(g._admit(big, pipesched.expected_bytes(big), True))
        g.acquire(big)
        self.assertFalse(g._admit(small, pipesched.expected_bytes(small), False))   # nothing joins an item running alone

    def test_nothing_starts_below_the_free_memory_floor_except_the_first(self) -> None:
        item = pipesched.Item("a", "vs", 1000)
        need = pipesched.expected_bytes(item)
        low = self.gov(1 << 40, free=pipesched.FLOOR_BYTES - 1)
        low.acquire(item)
        self.assertFalse(low._admit(item, need, False))
        enough = self.gov(1 << 40, free=pipesched.FLOOR_BYTES)
        enough.acquire(item)
        self.assertTrue(enough._admit(item, need, False))


class DeviceTest(unittest.TestCase):
    def test_the_first_discrete_device_with_every_item_is_the_one(self) -> None:
        integrated = device(kind=1, name="Integrated")
        lacking = device(ok=False, name="Discrete without mesh shaders")
        discrete = device(name="Discrete")
        dev, why = pipelines.usable_device(host(integrated, lacking, discrete))
        self.assertEqual((dev.index, dev.name, why), (2, "Discrete", ""))
        self.assertEqual(pipelines.usable_device(host(integrated, lacking))[0].index, 0)

    def test_a_device_that_lacks_an_item_is_named_with_the_item(self) -> None:
        dev, why = pipelines.usable_device(host(device(ok=False)))
        self.assertIsNone(dev)
        self.assertIn("mesh and task shaders", why)
        self.assertIn("Test GPU", why)

    def test_no_loader_no_device_and_no_probe_are_reasons(self) -> None:
        self.assertIn("not probed", pipelines.usable_device({})[1])
        self.assertIn("no Vulkan loader", pipelines.usable_device({"vulkan": {"devices": []}, "vulkan_items": [1]})[1])
        self.assertIn("no GPU", pipelines.usable_device({"vulkan": {"loader": True, "devices": []}, "vulkan_items": [1]})[1])

    def test_the_identity_names_the_driver_and_the_pipeline_cache_uuid(self) -> None:
        a = pipelines.usable_device(host(device()))[0].identity
        self.assertNotEqual(a, pipelines.usable_device(host(device(driver=0x9b000)))[0].identity)
        self.assertNotEqual(a, pipelines.usable_device(host(device(uuid="cd" * 16)))[0].identity)
        self.assertIsNone(pipelines.usable_device(host({**device(), "pipeline_cache_uuid": ""}))[0])


class StepKeyTest(unittest.TestCase):
    def key(self, hosts, **options) -> str:
        ctx = Context(Path("unused"), argparse.Namespace(pipelines_list=None, **options), Reporter(io.StringIO(), io.StringIO()))
        ctx.state["host"] = hosts
        return Pipelines().key(ctx)

    def test_a_driver_update_changes_the_key_and_the_worker_count_and_cold_flag_do_not(self) -> None:
        base = self.key(host(device()))
        self.assertNotEqual(base, self.key(host(device(driver=0x9b000))))
        self.assertNotEqual(base, self.key(host(device(uuid="cd" * 16))))
        self.assertEqual(base, self.key(host(device()), pipelines_jobs=3, pipelines_cold=True))

    def test_a_list_of_keys_changes_the_key(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a.txt", Path(t) / "b.txt"
            a.write_text("x\ny\n")
            b.write_text("y\nx\n")
            ctx = Context(Path("unused"), argparse.Namespace(pipelines_list=a), Reporter(io.StringIO(), io.StringIO()))
            ctx.state["host"] = host()
            ka = Pipelines().key(ctx)
            ctx.options.pipelines_list = b
            self.assertEqual(ka, Pipelines().key(ctx))              # the order of a list is not its content
            b.write_text("x\n")
            self.assertNotEqual(ka, Pipelines().key(ctx))
        self.assertNotEqual(ka, self.key(host()))

    def test_the_sources_that_decide_a_create_info_are_in_the_key_and_the_rest_of_the_runtime_is_not(self) -> None:
        names = {p.name for p in pipelines.key_files()}
        self.assertTrue({"pipebuild.c", "vk_pipeline.c", "vk_layout.c", "vk_device.c", "spirv.c", "vk_pipeline.h",
                         "vk_required.h"} <= names)
        self.assertFalse({"gpu", "core", "nvn", "gpu_sync.c", "vk_submit.c", "clock.c"} & names)
        self.assertTrue(all(p.is_file() for p in pipelines.key_files()), "a named source that does not exist")
        everything = {p.name for p in pipelines.source_files()}
        self.assertTrue({"core", "nvn", "gpu"} <= everything)       # the program is built again when any of it changes


class ProgressFileTest(unittest.TestCase):
    def test_a_segment_that_is_gone_a_crash_and_a_torn_line_are_not_kept_and_a_refusal_is(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            segments = t / "segments"
            segments.mkdir()
            (segments / "seg-000001.bin").write_text("x")
            rows = [pipesched.Result("a", "cs", "ok", 1, 0, 8, 1.0, 1, 0, segment="seg-000001"),
                    pipesched.Result("b", "vs", "ok", 1, 0, 8, 1.0, 1, 0, segment="seg-000002"),   # its segment is gone
                    pipesched.Result("c", "ps", "crash", message="died"),
                    pipesched.Result("d", "ps", "fail", message="FATAL: no")]
            text = "".join(pipesched.progress_line(r) + "\n" for r in rows) + "ok\te\tcs\t1"        # cut by a stop
            (t / "progress.tsv").write_text(text, encoding="utf-8")
            kept = pipesched.read_progress(t / "progress.tsv", segments)
            self.assertEqual(sorted(kept), ["a", "d"])
            self.assertEqual(pipesched.next_segment_number(segments), 2)


class StepTest(unittest.TestCase):
    def test_every_compute_pipeline_and_library_is_built_compute_first_and_recorded(self) -> None:
        env = Env(self)
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        order = env.builds()
        self.assertEqual([k for k, _ in order[:4]], ["cs"] * 4)                         # compute first, one worker
        self.assertEqual([key for kind, key in order if kind == "cs"][0], "c000000000000002.00000000")   # the largest
        self.assertEqual(sorted(kind for kind, _ in order), ["cs"] * 4 + ["ps"] * 3 + ["vs"] * 3)
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["built"], {"cs": 4, "vs": 3, "ps": 3})
        self.assertEqual(rec["not_built"], {"ms": 1, "ts": 1})
        self.assertTrue(rec["lower_bound"])
        cache = env.dest / "pipelines" / "pipeline.cache"
        self.assertEqual(cache.stat().st_size, rec["cache_bytes"])
        self.assertEqual(len(cache_keys(cache)), 10)                             # the merge holds every segment
        rows = (env.dest / "pipelines" / "times.tsv").read_text().splitlines()
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[0].split("\t")[:3], ["status", "key", "kind"])
        self.assertFalse((env.dest / "temp" / "pipelines").exists())                     # the scratch is gone
        self.assertEqual(rec["device"]["name"], "Test GPU")

    def test_a_kept_step_is_not_built_again_and_a_driver_update_builds_it_again(self) -> None:
        env = Env(self)
        status, ctx = env.run()
        self.assertEqual(status, "ok")
        done = len(env.builds())
        status, ctx = env.run(ctx)
        self.assertEqual(len(env.builds()), done, env.say(ctx))
        self.assertIn("pipelines: kept", env.say(ctx))
        ctx.state["host"] = host(device(driver=0x9b000))        # a driver update
        env.standin(driver="0009b000")
        status, ctx = env.run(ctx)
        self.assertEqual(len(env.builds()), 2 * done)

    def test_the_record_names_the_runtime_the_builder_said_and_the_key_names_what_the_step_requires(self) -> None:
        env = Env(self)
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        self.assertEqual(ctx.state["steps"]["pipelines"]["runtime"], {"zero_workgroup": "1"})
        base = Pipelines().key(ctx)
        with mock.patch.dict(pipelines.RUNTIME_SAYS, {"zero_workgroup": "0"}):
            self.assertNotEqual(base, Pipelines().key(ctx))              # a runtime that asked for less is another key

    def test_a_record_of_an_earlier_runtime_is_not_current_and_the_step_builds_again(self) -> None:
        env = Env(self)
        _, ctx = env.run()
        rec = dict(ctx.state["steps"]["pipelines"])
        self.assertTrue(Pipelines().current(ctx, rec))
        for said in ({}, {"zero_workgroup": "0"}, None):
            old = dict(rec)
            if said is None:
                del old["runtime"]                                       # made before the builder said anything
            else:
                old["runtime"] = said
            self.assertFalse(Pipelines().current(ctx, old), said)
        done = len(env.builds())
        del ctx.state["steps"]["pipelines"]["runtime"]
        env.run(ctx)
        self.assertEqual(len(env.builds()), 2 * done)                    # the cache of the earlier runtime was built again

    def test_a_builder_whose_runtime_does_not_zero_workgroup_memory_stops_the_step(self) -> None:
        for config in ({"zero_workgroup": 0}, {"older_runtime": True}):
            env = Env(self, config=config)
            status, ctx = env.run()
            self.assertEqual(status, "failed", config)
            said = env.say(ctx)
            self.assertIn("another runtime than the one prepare ships", said)
            self.assertIn("zero_workgroup", said)
            self.assertEqual(env.builds(), [])
            self.assertFalse((env.dest / "pipelines" / "pipeline.cache").exists())

    def test_a_missing_cache_runs_the_step_again(self) -> None:
        env = Env(self)
        _, ctx = env.run()
        (env.dest / "pipelines" / "pipeline.cache").unlink()
        self.assertFalse(Pipelines().current(ctx, ctx.state["steps"]["pipelines"]))

    def test_a_gpu_without_every_item_skips_the_step_with_a_warning(self) -> None:
        env = Env(self)
        status, ctx = env.run(env.context(host(device(ok=False))))
        self.assertEqual(status, "ok")
        self.assertEqual(env.builds(), [])
        self.assertIn("mesh and task shaders", ctx.state["steps"]["pipelines"]["skipped"])
        self.assertIn("warning: pipelines: skipped", env.say(ctx))
        self.assertTrue(Pipelines().current(ctx, ctx.state["steps"]["pipelines"]))

    def test_the_same_finding_fails_the_step_once_the_vulkan_items_are_required(self) -> None:
        env = Env(self)
        with mock.patch.object(check, "REQUIRE_VULKAN_ITEMS", True):
            status, ctx = env.run(env.context(host(device(ok=False))))
        self.assertEqual(status, "failed")
        self.assertIn("mesh and task shaders", env.say(ctx))

    def test_a_list_builds_only_those_keys_and_a_key_the_pack_lacks_fails(self) -> None:
        env = Env(self)
        listing = env.tmp / "list.txt"
        listing.write_text("# a comment\np000000000000003.00000000\nc000000000000001.00000000\n")
        status, ctx = env.run(env.context(pipelines_list=listing))
        self.assertEqual(status, "ok", env.say(ctx))
        self.assertEqual(sorted(key for _, key in env.builds()), ["c000000000000001.00000000", "p000000000000003.00000000"])
        listing.write_text("zzzz\n")
        status, ctx = env.run(env.context(pipelines_list=listing))
        self.assertEqual(status, "failed")
        self.assertIn("zzzz", env.say(ctx))

    def test_a_program_the_runtime_refuses_by_name_is_recorded_and_not_built_again(self) -> None:
        bad = "v000000000000001.00000000"
        env = Env(self, config={"fatal": [bad]})
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["failed"], rec["failed_keys"]), (1, [bad]))
        self.assertEqual(sum(1 for _, key in env.builds() if key == bad), 1)             # asked once, not retried
        row = [line for line in (env.dest / "pipelines" / "times.tsv").read_text().splitlines() if bad in line][0]
        self.assertTrue(row.startswith("fail\t"))
        self.assertIn("stand-in refuses", row)
        self.assertIn("1 fail", env.say(ctx))
        # the dying builder saved what it had built, so nothing is built again and the cache holds the other nine
        self.assertEqual((len(env.builds()), rec["lost"]), (10, 0))
        self.assertEqual(len(cache_keys(env.dest / "pipelines" / "pipeline.cache")), 9)

    def test_a_builder_that_dies_without_saving_loses_its_unsaved_items_which_are_built_again(self) -> None:
        bad = "v000000000000001.00000000"
        env = Env(self, config={"fatal": [bad], "no_salvage": True})
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["lost"], 5)                    # four compute programs and the largest pixel program
        self.assertEqual(len(env.builds()), 15)
        self.assertEqual(len(cache_keys(env.dest / "pipelines" / "pipeline.cache")), 9)

    def test_a_worker_that_dies_is_replaced_and_its_item_is_tried_again_alone(self) -> None:
        victim = "c000000000000003.00000000"
        env = Env(self, config={"crash_once": [victim]})
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["built"]["cs"], rec["failed"]), (4, 0))
        self.assertEqual(sum(1 for _, key in env.builds() if key == victim), 2)
        self.assertIn("got no answer", env.say(ctx))

    def test_an_item_that_never_answers_is_a_timeout_failure_and_the_step_goes_on(self) -> None:
        hang = "p000000000000001.00000000"
        env = Env(self, config={"hang": [hang]})
        with mock.patch.object(pipesched, "ITEM_TIMEOUT", 1.0), mock.patch.object(pipesched, "RETRY_TIMEOUT", 1.0):
            status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["failed_keys"], [hang])
        row = [line for line in (env.dest / "pipelines" / "times.tsv").read_text().splitlines() if hang in line][0]
        self.assertTrue(row.startswith("timeout\t"))

    def test_nothing_built_fails_the_step_and_a_builder_that_cannot_start_says_why(self) -> None:
        env = Env(self, config={"fatal": list(DEFAULT_JOBS)})
        status, ctx = env.run()
        self.assertEqual(status, "failed")
        self.assertIn("no pipeline could be built", env.say(ctx))
        env2 = Env(self, config={"start_fail": True})
        status, ctx = env2.run()
        self.assertEqual(status, "failed")
        self.assertIn("did not start", env2.say(ctx))
        self.assertIn("stand-in has no device", env2.say(ctx))

    def test_several_workers_build_everything_once(self) -> None:
        env = Env(self, config={"delay": 0.02})
        status, ctx = env.run(env.context(pipelines_jobs=3))
        self.assertEqual(status, "ok", env.say(ctx))
        keys = [key for _, key in env.builds()]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(ctx.state["steps"]["pipelines"]["workers"], 3)


class RecycleTest(unittest.TestCase):
    """The driver keeps the memory of its largest build for the life of the process; the governor counts only the items
    running. A builder that holds a gigabyte is saved and replaced (the pipeline step's sample: six workers held 4.8 GB and the
    free memory fell to 0.25 GB)."""

    def run_with_working_set(self, size: int) -> tuple[Env, Context]:
        env = Env(self, config={"delay": 0.25})
        with mock.patch.object(pipesched.pool, "working_set", lambda proc: size), \
                mock.patch.object(pipesched, "MONITOR_SECONDS", 0.05):
            status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        return env, ctx

    def test_a_builder_that_holds_a_gigabyte_is_saved_and_replaced_and_nothing_is_lost(self) -> None:
        env, ctx = self.run_with_working_set(2 << 30)
        rec = ctx.state["steps"]["pipelines"]
        self.assertGreaterEqual(rec["recycled"], 5)
        self.assertEqual((rec["lost"], rec["failed"], len(env.builds())), (0, 0, 10))
        self.assertEqual(len(cache_keys(env.dest / "pipelines" / "pipeline.cache")), 10)
        self.assertGreater(rec["segments"], 5)               # each replaced builder saved its segment first

    def test_a_small_builder_is_kept(self) -> None:
        _, ctx = self.run_with_working_set(100 << 20)
        self.assertEqual(ctx.state["steps"]["pipelines"]["recycled"], 0)

    def test_the_model_covers_every_build_measured_on_the_reference_pc(self) -> None:
        # (module bytes, working set of the worker that built it alone), the pipeline step's validation sample: the builds that
        # take seconds, where the working set is the build's own (a small build shows what its worker kept before it)
        measured = [(816 * 1024, 351 << 20), (1045 * 1024, 715 << 20), (1354 * 1024, 689 << 20),
                    (2648 * 1024, 988 << 20), (11620 * 1024, 2911 << 20), (597 * 1024, 427 << 20)]
        for size, held in measured:
            self.assertGreaterEqual(pipesched.expected_bytes(pipesched.Item("x", "cs", size)), held, size)


class FailureKindsTest(unittest.TestCase):
    """What a failed item is called decides whether it is built again: the machine's failures are retried, the
    module's are recorded for good."""

    def test_a_builder_on_another_device_than_the_one_chosen_stops_the_step(self) -> None:
        env = Env(self, config={"uuid": "cd" * 16})            # the probe chose the device with the "ab" UUID
        status, ctx = env.run()
        self.assertEqual(status, "failed")
        said = env.say(ctx)
        self.assertIn("another device than the one prepare chose", said)
        self.assertIn("cache_uuid", said)
        self.assertEqual(env.builds(), [])
        self.assertFalse((env.dest / "pipelines" / "pipeline.cache").exists())

    def test_the_vendor_the_device_and_the_driver_are_compared_too(self) -> None:
        for field, value in (("vendor", "1002"), ("device", "9999"), ("driver", "0009b000")):
            env = Env(self, config={field: value})
            status, ctx = env.run()
            self.assertEqual(status, "failed", field)
            self.assertIn(field, env.say(ctx))

    def test_a_named_fatal_about_memory_is_tried_again_alone_and_not_recorded_as_a_refusal(self) -> None:
        victim = "c000000000000003.00000000"
        env = Env(self, config={"transient_once": [victim]})
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["built"]["cs"], rec["failed"]), (4, 0))
        self.assertEqual(sum(1 for _, key in env.builds() if key == victim), 2)
        self.assertNotIn("fail\t", (env.dest / "pipelines" / "times.tsv").read_text())

    def test_a_refusal_of_one_command_leaves_the_builder_running_and_its_unsaved_work_alone(self) -> None:
        bad = "v000000000000001.00000000"
        env = Env(self, config={"reject": [bad]})
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["failed"], rec["failed_keys"], rec["lost"], rec["restarts"]), (1, [bad], 0, 0))
        self.assertEqual(len(env.builds()), 10)

    def test_a_crash_is_not_reported_with_the_fatal_line_of_the_builder_before_it(self) -> None:
        refused, crasher = "p000000000000003.00000000", "v000000000000003.00000000"
        env = Env(self, config={"fatal": [refused], "crash": [crasher]})
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        rows = {line.split("\t")[1]: line for line in (env.dest / "pipelines" / "times.tsv").read_text().splitlines()[1:]}
        self.assertTrue(rows[refused].startswith("fail\t") and "stand-in refuses" in rows[refused])
        self.assertTrue(rows[crasher].startswith("crash\t"))
        self.assertNotIn("stand-in refuses", rows[crasher])
        self.assertIn("exited with code 9", rows[crasher])


class SkipTest(unittest.TestCase):
    def test_skip_pipelines_leaves_the_step_out_and_changes_its_key(self) -> None:
        env = Env(self)
        status, ctx = env.run(env.context(skip_pipelines=True))
        self.assertEqual(status, "ok", env.say(ctx))
        self.assertEqual(env.builds(), [])
        self.assertEqual(ctx.state["steps"]["pipelines"]["skipped"], "--skip-pipelines")
        self.assertIn("skipped, as --skip-pipelines asked", env.say(ctx))
        skipping = ctx.keys["pipelines"]
        other = env.context()
        run_steps([FakeTranslate(), Pipelines()], other, only={"translate"})
        self.assertNotEqual(skipping, other.keys["pipelines"])      # a later run without the flag builds


class ProgressCountTest(unittest.TestCase):
    def test_an_item_built_again_or_tried_again_counts_once(self) -> None:
        from prepare.pipelines import Progress
        rep = Reporter(io.StringIO(), io.StringIO())
        items = [pipesched.Item("a", "cs", 1), pipesched.Item("b", "vs", 1)]
        p = Progress(rep, items, {})
        p.add(pipesched.Result("a", "cs", "ok"))
        p.add(pipesched.Result("a", "cs", "ok"))          # built again after a lost worker
        p.add(pipesched.Result("b", "vs", "crash"))       # not an answer yet
        self.assertEqual(p.done, 1)
        p.add(pipesched.Result("b", "vs", "ok"))
        self.assertEqual((p.done, p.total), (2, 2))


class ResumeTest(unittest.TestCase):
    def interrupted_run(self, env: Env, keep: int) -> Context:
        """Ctrl+C after the first `keep` programs were built and saved: the pool runs over a shortened list, then the
        interrupt arrives."""
        ctx = env.context()
        real = pipesched.Pool.run

        def cut(pool, retry_pass=False):
            del pool.todo
            pool.todo = __import__("collections").deque(list(pool.todo_all)[:keep])
            real(pool, retry_pass)
            raise KeyboardInterrupt

        original_init = pipesched.Pool.__init__

        def init(pool, command, items, *a, **k):
            pool.todo_all = list(items)
            original_init(pool, command, items, *a, **k)

        with mock.patch.object(pipesched.Pool, "__init__", init), mock.patch.object(pipesched.Pool, "run", cut):
            status = run_steps([FakeTranslate(), Pipelines()], ctx)
        self.assertEqual(status, "cancelled")
        return ctx

    def test_an_interrupted_run_keeps_what_was_saved_and_the_next_builds_only_the_rest(self) -> None:
        env = Env(self)
        ctx = self.interrupted_run(env, 5)
        first = [key for _, key in env.builds()]
        self.assertEqual(len(first), 5)
        scratch = env.dest / "temp" / "pipelines"
        self.assertEqual(len((scratch / "progress.tsv").read_text().splitlines()), 5)
        self.assertEqual(len(list((scratch / "segments").glob("seg-*.bin"))), 1)
        self.assertNotIn("pipelines", ctx.state["steps"])                  # a step stopped halfway is not finished
        status, ctx = env.run(env.context())
        self.assertEqual(status, "ok", env.say(ctx))
        second = [key for _, key in env.builds()][5:]
        self.assertEqual(len(second), 5)
        self.assertFalse(set(first) & set(second))
        self.assertEqual(ctx.state["steps"]["pipelines"]["kept_items"], 5)
        self.assertEqual(ctx.state["steps"]["pipelines"]["built"], {"cs": 4, "vs": 3, "ps": 3})
        self.assertEqual(len(cache_keys(env.dest / "pipelines" / "pipeline.cache")), 10)
        self.assertEqual(len((env.dest / "pipelines" / "times.tsv").read_text().splitlines()), 11)

    def test_a_lost_segment_is_built_again(self) -> None:
        env = Env(self)
        self.interrupted_run(env, 5)
        for seg in (env.dest / "temp" / "pipelines" / "segments").glob("seg-*.bin"):
            seg.unlink()
        status, ctx = env.run(env.context())
        self.assertEqual(status, "ok", env.say(ctx))
        self.assertEqual(len(env.builds()), 5 + 10)
        self.assertEqual(ctx.state["steps"]["pipelines"]["kept_items"], 0)

    def test_the_progress_never_leads_the_cache(self) -> None:
        env = Env(self, config={"flush_fail": True})
        status, ctx = env.run()
        self.assertEqual(status, "failed")
        self.assertIn("could not save a segment", env.say(ctx))
        progress = env.dest / "temp" / "pipelines" / "progress.tsv"
        self.assertEqual(progress.read_text() if progress.exists() else "", "")

    def test_another_key_starts_from_nothing(self) -> None:
        env = Env(self)
        self.interrupted_run(env, 5)
        new = env.context(host(device(driver=0x9b000)))
        env.standin(driver="0009b000")
        status, ctx = env.run(new)
        self.assertEqual(status, "ok", env.say(ctx))
        self.assertEqual(ctx.state["steps"]["pipelines"]["kept_items"], 0)
        self.assertEqual(len(env.builds()), 5 + 10)


class WarmPassTest(unittest.TestCase):
    def test_a_fresh_process_seeded_with_the_cache_is_served_every_item(self) -> None:
        env = Env(self)
        status, ctx = env.run()
        self.assertEqual(status, "ok", env.say(ctx))
        cache = env.dest / "pipelines" / "pipeline.cache"
        items = pipesched.order(pipelines.load_work(ctx, env.dest / "shaders" / cp.BASE_FILE, None).items)
        warm = pipelines.warm_pass(ctx, items, cache)
        self.assertEqual((warm["items"], warm["hits"], warm["failed"]), (10, 10, 0))
        self.assertTrue(Pipelines().verify(ctx, ctx.state["steps"]["pipelines"]))
        cache.write_text("")                                                   # an empty cache serves nothing
        self.assertFalse(Pipelines().verify(ctx, ctx.state["steps"]["pipelines"]))


if __name__ == "__main__":
    unittest.main()
