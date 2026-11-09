# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The real pipeline builder (tools/pipebuild/pipebuild.c, the pipeline step) on a device: a worker that builds a vertex library,
a fragment library and a compute pipeline of a synthetic pack, saves what it built as a segment, merges segments into
one cache file, and a fresh process seeded with that file is served every one of them. Validation (with
synchronization checks) is on and must stay clean. The cases that need no device are in test_prepare_pipelines.py, with
a stand-in builder; these skip without a Vulkan SDK, clang or a device that has every item the renderer requires, and
fail instead when OOT2_REQUIRE_VULKAN, OOT2_REQUIRE_CLANG or OOT2_REQUIRE_GPU is set.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import ROOT, build_program, case_env, clang_or_skip, vulkan_or_skip
from test_runtime_vk_pipeline import SYNTH_KEYS, DeviceTestCase

SOURCE = ROOT / "tools" / "pipebuild" / "pipebuild.c"
VS, PS, CS = SYNTH_KEYS["vs.vert"], SYNTH_KEYS["ps.frag"], SYNTH_KEYS["cs.comp"]
ABSENT = "5100000000000009.00000000"


class Worker:
    """One `pipebuild serve` process: a line in, a line out."""

    def __init__(self, exe: Path, pack: Path, segments: Path, cache: Path | None = None, **env: str):
        cmd = [str(exe), "serve", "--pack", str(pack), "--segments", str(segments)]
        if cache:
            cmd += ["--cache", str(cache)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     env=case_env({"OOT2_VK_VALIDATION": "sync", **env}))
        self.ready = self.read()

    def read(self) -> str:
        return self.proc.stdout.readline().strip()

    def ask(self, line: str) -> str:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.read()

    def finish(self) -> tuple[str, str]:
        """Quit, and return the last reply and stderr."""
        try:
            last = self.ask("quit") if self.proc.poll() is None else ""
        except OSError:      # the process had stopped already
            last = ""
        err = self.proc.stderr.read()
        self.proc.wait(timeout=60)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()
            except OSError:
                pass
        return last, err


class CompileTest(unittest.TestCase):
    """No device needed: the builder compiles clean (-Wall -Wextra -Werror) with the runtime's core, nvn and gpu code on
    every runner that has clang and the Vulkan headers, so a mistake in it shows up in continuous integration, where
    the device cases below skip."""

    def test_the_builder_compiles_without_a_warning_and_prints_its_usage(self) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        with tempfile.TemporaryDirectory(prefix="oot2-pipebuild-cc-", ignore_cleanup_errors=True) as t:
            exe = build_program(cc, include, glslang, SOURCE, Path(t), levels=("-O1",))["-O1"]
            done = subprocess.run([str(exe)], capture_output=True, text=True, env=case_env())
            self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
            self.assertIn("usage: pipebuild serve", done.stderr)


class PipeBuildTest(DeviceTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.out = Path(tempfile.mkdtemp(prefix="oot2-pipebuild-"))
        cls.addClassCleanup(shutil.rmtree, cls.out, True)
        cls.exe = build_program(cc, include, glslang, SOURCE, cls.out, levels=("-O1",))["-O1"]

    def folder(self) -> tuple[Path, Path]:
        tmp = Path(tempfile.mkdtemp(prefix="oot2-pipebuild-case-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        segments = tmp / "segments"
        segments.mkdir()
        return tmp, segments

    def merge(self, tmp: Path, *segments: Path) -> subprocess.CompletedProcess:
        listing = tmp / "list.txt"
        listing.write_text("".join(f"{s}\n" for s in segments), encoding="utf-8")
        return subprocess.run([str(self.exe), "merge", "--out", str(tmp / "pipeline.cache"), "--list", str(listing)],
                              capture_output=True, text=True, env=case_env({"OOT2_VK_VALIDATION": "sync"}), timeout=300)

    def test_a_worker_builds_saves_a_segment_and_a_fresh_process_gets_every_pipeline_from_the_merged_cache(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments)
        self.assertRegex(w.ready, r"^ready vendor=[0-9a-f]{4} device=[0-9a-f]{4} driver=[0-9a-f]{8} api=1\.\d+\.\d+ "
                                  r"cache_uuid=[0-9a-f]{32} cache=none seeded=\d+ zero_workgroup=1 name=")
        built = {}
        for kind, key in (("vs", VS), ("ps", PS), ("cs", CS)):
            reply = w.ask(f"build {kind} {key}").split()
            self.assertEqual(reply[:3], ["ok", kind, key], reply)
            self.assertEqual(reply[4], "0")                         # a cold build: the cache served nothing
            self.assertGreater(int(reply[3]), 0)                    # the driver's time
            built[kind] = int(reply[5])
        self.assertTrue(all(size > 0 for size in built.values()))
        self.assertGreater(int(w.ask("size").split()[1]), 0)
        saved = w.ask("flush first").split()
        self.assertEqual(saved[:2], ["saved", "first"])
        self.assertEqual((segments / "first.bin").stat().st_size, int(saved[2]))
        self.assertEqual(int(w.ask("size").split()[1]), int(w.ask("size").split()[1]))   # the cache starts over, empty
        last, err = w.finish()
        self.assertIn("validation_errors 0 validation_warnings 0", last, err[-2000:])

        done = self.merge(tmp, segments / "first.bin")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        merged = done.stdout.split()
        self.assertEqual(merged[:2], ["merged", "1"])
        cache = tmp / "pipeline.cache"
        self.assertEqual(cache.stat().st_size, int(merged[3]))

        again = Worker(self.exe, self.pack, segments, cache)
        self.assertRegex(again.ready, r"cache=loaded seeded=[1-9]\d*")
        for kind, key in (("vs", VS), ("ps", PS), ("cs", CS)):
            reply = again.ask(f"build {kind} {key}").split()
            self.assertEqual(reply[:3], ["ok", kind, key], reply)
            self.assertEqual(reply[4], "1", f"the {kind} was not served from the cache")
        last, err = again.finish()
        self.assertIn("validation_errors 0 validation_warnings 0", last, err[-2000:])

    def test_a_named_refusal_saves_what_was_built_and_the_segment_merges(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments)
        self.assertEqual(w.ask(f"build ps {PS}").split()[:3], ["ok", "ps", PS])
        died = w.ask(f"build cs {ABSENT}")
        self.assertRegex(died, r"^died seg-d\d+-\d+ [1-9]\d* .*FATAL.*no module for stage " + ABSENT)
        _, err = w.finish()
        self.assertEqual(w.proc.returncode, 70)
        self.assertIn("FATAL", err)
        saved = sorted(segments.glob("seg-d*.bin"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].stat().st_size, int(died.split()[2]))
        done = self.merge(tmp, *saved)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

    def test_a_refusal_with_nothing_built_saves_nothing(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments)
        died = w.ask(f"build vs {ABSENT}")
        self.assertRegex(died, r"^died - 0 .*FATAL")
        w.finish()
        self.assertEqual(list(segments.iterdir()), [])

    def test_a_job_key_of_another_stage_is_refused_by_name(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments)
        died = w.ask(f"build ps {VS}")
        self.assertRegex(died, r"^died - 0 .*FATAL.* is a (vertex|vs|VS)? ?(program|stage)", died)
        w.finish()

    def test_bad_commands_are_answered_and_do_not_stop_the_worker(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments)
        self.assertTrue(w.ask("build xx abc").startswith("fail xx abc"))
        self.assertTrue(w.ask("build vs bad/key").startswith("fail vs bad/key"))
        self.assertTrue(w.ask("flush ../x").startswith("fail flush"))
        self.assertTrue(w.ask("dance").startswith("fail dance"))
        self.assertEqual(w.ask(f"build ps {PS}").split()[:3], ["ok", "ps", PS])
        last, err = w.finish()
        self.assertIn("validation_errors 0", last, err[-2000:])

    def test_a_cache_of_another_driver_is_dropped_and_a_segment_of_one_is_refused(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments)
        w.ask(f"build cs {CS}")
        w.ask("flush a")
        w.finish()
        data = bytearray((segments / "a.bin").read_bytes())
        data[16] ^= 0xFF                                           # the first byte of the header's pipelineCacheUUID
        foreign = tmp / "foreign.bin"
        foreign.write_bytes(bytes(data))
        again = Worker(self.exe, self.pack, segments, foreign)
        self.assertRegex(again.ready, r"cache=other_driver seeded=\d{1,2} ")   # an empty cache is its header alone
        again.finish()
        done = self.merge(tmp, foreign)
        self.assertEqual(done.returncode, 3, done.stdout)
        self.assertIn("not a cache of this device and driver", done.stderr)
        self.assertFalse((tmp / "pipeline.cache").exists())

    def test_the_drivers_own_cache_can_be_turned_off_and_a_wrong_value_is_refused(self) -> None:
        tmp, segments = self.folder()
        w = Worker(self.exe, self.pack, segments, OOT2_VK_DRIVER_CACHE="off")
        self.assertEqual(w.ask(f"build cs {CS}").split()[:3], ["ok", "cs", CS])
        last, err = w.finish()
        self.assertIn("OOT2_VK_DRIVER_CACHE off", err)
        self.assertIn("validation_errors 0", last)
        bad = Worker(self.exe, self.pack, segments, OOT2_VK_DRIVER_CACHE="maybe")
        self.assertEqual(bad.ready, "")
        _, err = bad.finish()
        self.assertEqual(bad.proc.returncode, 70)
        self.assertIn('OOT2_VK_DRIVER_CACHE is "maybe"', err)

    def test_the_step_compiles_the_builder_and_builds_a_synthetic_pack_with_validation_clean(self) -> None:
        """The pipelines step end to end: its own compile of the builder, the host probe's device, workers on the GPU,
        the merge, the validation counts, then the same programs created again from the cache by fresh processes."""
        import argparse
        import io
        import json
        import os
        from unittest import mock

        import cache_pack as cp
        from prepare import check, pipelines, pipesched
        from prepare.pipelines import Pipelines
        from prepare.runner import Context, Reporter, run_steps
        from test_prepare_pipelines import FakeTranslate

        tmp, _ = self.folder()
        dest = tmp / "dest"
        (dest / "shaders" / "split").mkdir(parents=True)
        shutil.copy(self.pack, dest / "shaders" / cp.BASE_FILE)
        stages = {SYNTH_KEYS["vs.vert"]: "vs", SYNTH_KEYS["ps.frag"]: "ps", SYNTH_KEYS["cs.comp"]: "cs",
                  SYNTH_KEYS["cs2.comp"]: "cs"}
        (dest / "shaders" / "split" / "jobs.jsonl").write_text(
            "".join(json.dumps({"key": k, "stage": s}) + "\n" for k, s in stages.items()), encoding="utf-8")
        options = argparse.Namespace(cc=None, vulkan_sdk=None, pipelines_builder=None, pipelines_jobs=2,
                                     pipelines_list=None, pipelines_cold=False)
        ctx = Context(dest, options, Reporter(io.StringIO(), io.StringIO()))
        ctx.state["host"] = check.run_probe(check.build_probe(check.compiler(ctx), tmp / "probe", check.vulkan_include(ctx)))
        with mock.patch.dict(os.environ, {"OOT2_VK_VALIDATION": "sync"}):
            status = run_steps([FakeTranslate(), Pipelines()], ctx)
        said = ctx.reporter.out.getvalue() + ctx.reporter.err.getvalue()
        self.assertEqual(status, "ok", said)
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["built"], {"cs": 2, "vs": 1, "ps": 1}, said)
        self.assertEqual((rec["failed"], rec["validation"], rec["validation_errors"], rec["validation_warnings"]),
                         (0, "sync", 0, 0))
        cache = dest / "pipelines" / "pipeline.cache"
        self.assertEqual(cache.stat().st_size, rec["cache_bytes"])
        self.assertGreater(rec["cache_bytes"], 100)
        self.assertEqual(len((dest / "pipelines" / "times.tsv").read_text().splitlines()), 5)
        self.assertFalse((dest / "temp" / "pipelines").exists())
        items = pipesched.order(pipelines.load_work(ctx, dest / "shaders" / cp.BASE_FILE, None).items)
        with mock.patch.dict(os.environ, {"OOT2_VK_VALIDATION": "sync"}):
            warm = pipelines.warm_pass(ctx, items, cache)
            self.assertEqual((warm["items"], warm["hits"], warm["failed"]), (4, 4, 0))
            self.assertTrue(Pipelines().verify(ctx, rec))

    def test_a_key_change_with_one_module_changed_serves_the_others_from_the_seed_and_builds_the_changed_one_cold(self) -> None:
        """The seed of the pipelines step on a device: the step builds a synthetic pack, its key changes and one module of
        the pack changes, and the step runs again. The previous cache is the seed: every unchanged module is served by
        the driver from it, the changed one is built cold, and the new cache still holds the seed's entries."""
        import argparse
        import io
        import json
        import os
        from unittest import mock

        import cache_pack as cp
        from prepare import check, pipelines, pipeseed
        from prepare.pipelines import Pipelines
        from prepare.runner import Context, Reporter, run_steps
        from test_prepare_pipelines import FakeTranslate
        from test_runtime_vk_pipeline import GLSL, synthetic_pack

        tmp, _ = self.folder()
        (tmp / "changed").mkdir()
        with mock.patch.dict(GLSL, {"cs2.comp": GLSL["cs2.comp"].replace("= 1u", "= 2u")}):
            changed_pack = synthetic_pack(tmp / "changed")             # the same pack with one compute module changed
        dest = tmp / "dest"
        (dest / "shaders" / "split").mkdir(parents=True)
        shutil.copy(self.pack, dest / "shaders" / cp.BASE_FILE)
        stages = {SYNTH_KEYS["vs.vert"]: "vs", SYNTH_KEYS["ps.frag"]: "ps", SYNTH_KEYS["cs.comp"]: "cs",
                  SYNTH_KEYS["cs2.comp"]: "cs"}
        (dest / "shaders" / "split" / "jobs.jsonl").write_text(
            "".join(json.dumps({"key": k, "stage": s}) + "\n" for k, s in stages.items()), encoding="utf-8")
        options = argparse.Namespace(cc=None, vulkan_sdk=None, pipelines_builder=None, pipelines_jobs=2,
                                     pipelines_list=None, pipelines_cold=False)
        ctx = Context(dest, options, Reporter(io.StringIO(), io.StringIO()))
        ctx.state["host"] = check.run_probe(check.build_probe(check.compiler(ctx), tmp / "probe", check.vulkan_include(ctx)))
        marker = tmp / "marker.txt"
        base = pipelines.key_files()

        def hits() -> dict[str, str]:
            rows = (dest / "pipelines" / "times.tsv").read_text().splitlines()
            head = rows[0].split("\t")
            return {r.split("\t")[1]: r.split("\t")[head.index("hit")] for r in rows[1:]}

        with mock.patch.dict(os.environ, {"OOT2_VK_VALIDATION": "sync"}), \
                mock.patch.object(pipelines, "key_files", lambda: [*base, marker]):
            marker.write_text("one")
            self.assertEqual(run_steps([FakeTranslate(), Pipelines()], ctx), "ok", ctx.reporter.err.getvalue())
            first = ctx.state["steps"]["pipelines"]
            self.assertEqual(first["seed"]["outcome"], pipeseed.NONE)
            self.assertEqual(set(hits()), set(stages))
            cache = dest / "pipelines" / "pipeline.cache"
            first_bytes = cache.stat().st_size

            shutil.copy(changed_pack, dest / "shaders" / cp.BASE_FILE)    # one module changed ...
            marker.write_text("two")                                      # ... and the step's key with it
            status = run_steps([FakeTranslate(), Pipelines()], ctx)
            said = ctx.reporter.out.getvalue() + ctx.reporter.err.getvalue()
            self.assertEqual(status, "ok", said)
            self.assertIn("pipelines: started (its inputs changed)", said)
            second = ctx.state["steps"]["pipelines"]
            self.assertEqual(second["seed"]["outcome"], pipeseed.KEPT, said)
            self.assertEqual(second["seed"]["bytes"], first_bytes)
            self.assertGreaterEqual(second["seed"]["builders_loaded"], 1)
            served = hits()
            self.assertEqual({k for k, v in served.items() if v == "0"}, {SYNTH_KEYS["cs2.comp"]}, served)   # built cold
            self.assertEqual({k for k, v in served.items() if v == "1"},
                             {SYNTH_KEYS["vs.vert"], SYNTH_KEYS["ps.frag"], SYNTH_KEYS["cs.comp"]}, served)   # served warm
            self.assertEqual((second["failed"], second["validation_errors"], second["validation_warnings"]), (0, 0, 0))
            self.assertGreaterEqual(cache.stat().st_size, first_bytes)    # the seed's entries are in the new cache
            self.assertFalse((dest / "temp" / "pipelines-seed").exists())
            items = pipelines.pipesched.order(pipelines.load_work(ctx, dest / "shaders" / cp.BASE_FILE, None).items)
            warm = pipelines.warm_pass(ctx, items, cache)
            self.assertEqual((warm["items"], warm["hits"], warm["failed"]), (4, 4, 0))   # all four are served now

    def test_the_command_line_is_checked(self) -> None:
        self.assertEqual(subprocess.run([str(self.exe)], capture_output=True).returncode, 2)
        self.assertEqual(subprocess.run([str(self.exe), "serve", "--pack", "x"], capture_output=True).returncode, 2)
        self.assertEqual(subprocess.run([str(self.exe), "merge"], capture_output=True).returncode, 2)


if __name__ == "__main__":
    unittest.main()
