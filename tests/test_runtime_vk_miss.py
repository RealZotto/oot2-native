# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the miss path of the Vulkan back end (runtime/include/oot2/vk_miss.h, runtime/gpu/vk_miss.c) through
tests/runtime/vk_miss_test.c.

No device needed (continuous integration runs these): a miss in the middle of a frame drains the recorder's ring and the
timeline before it builds (the real recorder over stand-in calls, the real timeline over a counter); a graphics build past
the bound skips the draw, counts it, skips the next draw of the same pipeline at once and serves the draw after the build;
the override waits instead and counts the wait past the bound; a draw that may not be skipped waits and says so apart; a
compute miss waits; a stage key's pipelines go once no program holds it and the timeline has passed the release; and
OOT2_VK_MISS_BOUND is read as unbounded or a number of milliseconds, anything else a named fatal.

On the desktop's RTX 3060 Ti (skipped without a device that has every item of oot2/vk_required.h, unless OOT2_REQUIRE_GPU
is set), on the synthetic pack of GLSL modules of tests/test_runtime_vk_pipeline.py, under synchronization validation: a
real graphics miss in the middle of a frame with a submission the device holds (the build waits for it, and the builder's
own check that nothing is created with work in flight agrees); a pipeline bound in a command buffer the device holds is
released, and destroyed only after the device finished it; a build past a 1 ms bound goes on beside work in flight.

The no-device miss tests also check that the optional task stage survives the asynchronous graphics build and that a
vertex-only miss does not invent one.

Each check has a counter-run, a test-only switch of vk_miss_test that breaks the mechanism the check is for: a miss that
does not drain, a skip that is not counted, a released key destroyed at once, a build that is not told the replay stopped
waiting. The check must then fail.

Run: python -m unittest discover -s tests -p test_runtime_vk_miss.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import (EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines,  # sets the path of tools/
                              skip_on_partial_device, vulkan_or_skip)

import test_runtime_vk_pipeline as tp  # noqa: E402  (its synthetic pack and its device check)

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_miss_test.c"
LEVELS = ("-O1", "-O2")
EXIT_NO_DEVICE = 3


class MissBase(unittest.TestCase):
    """vk_miss_test built once per class, at the levels it is given."""

    levels: tuple[str, ...] = LEVELS

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        cls.include, cls.glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-vkmiss-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, cls.include, cls.glslang, TEST_SOURCE, Path(cls.tmp.name), levels=cls.levels)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str, env: dict[str, str] | None = None,
                 timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case, *args], capture_output=True, text=True, timeout=timeout,
                              env=case_env(env))

    def passes(self, case: str, *args: str, env: dict[str, str] | None = None) -> list[str]:
        """The case at each level: exit 0, its ok line, no fatal. The stdout lines of the last level."""
        done = None
        for level in self.levels:
            done = self.run_case(level, case, *args, env=env)
            self.assertEqual(done.returncode, 0, f"{case} at {level}:\n{done.stdout}{done.stderr}")
            self.assertIn(f"ok {case}", done.stdout)
            self.assertEqual(fatal_lines(done.stderr), [], done.stderr)
        assert done is not None
        return done.stdout.splitlines()

    def counter_run(self, case: str, mutant: str, *args: str, env: dict[str, str] | None = None,
                    level: str = "-O2") -> subprocess.CompletedProcess:
        """The case with a mechanism broken must fail one of its checks (exit 1) or stop by name, never pass."""
        done = self.run_case(level, case, *args, "--mutant", mutant, env=env)
        self.assertNotEqual(done.returncode, 0, f"{case} passed with the {mutant} mutant:\n{done.stdout}{done.stderr}")
        self.assertNotIn(f"ok {case}", done.stdout)
        return done


class MissPathTest(MissBase):
    def test_a_task_stage_survives_the_asynchronous_graphics_build(self) -> None:
        self.passes("task_stage")

    def test_a_miss_in_the_middle_of_a_frame_drains_the_ring_and_the_timeline_before_it_builds(self) -> None:
        out = self.passes("drain_order")
        line = next(l for l in out if l.startswith("at the build:"))
        self.assertRegex(line, r"20 of 20 packets made into calls, submissions issued 1 done 1")
        print("\n" + line)

    def test_counter_run_a_miss_that_does_not_drain(self) -> None:
        done = self.counter_run("drain_order", "nodrain")
        self.assertIn("check failed", done.stderr)

    def test_a_graphics_build_past_the_bound_skips_the_draw_counts_it_and_serves_the_next_one(self) -> None:
        out = self.passes("bound_skip")
        line = next(l for l in out if l.startswith("skipped "))
        self.assertIn("skipped 2 (at once 1), late builds 1", line)
        print("\n" + line)

    def test_counter_run_a_skip_that_is_not_counted(self) -> None:
        done = self.counter_run("bound_skip", "nocount")
        self.assertIn("check failed", done.stderr)

    def test_counter_run_a_build_that_is_not_told_the_replay_stopped_waiting(self) -> None:
        done = self.counter_run("bound_skip", "nolate")
        self.assertIn("check failed", done.stderr)

    def test_the_override_waits_instead_and_counts_the_wait_past_the_bound(self) -> None:
        out = self.passes("override")
        print("\n" + next(l for l in out if l.startswith("waited")))

    def test_a_draw_that_may_not_be_skipped_waits_and_is_counted_apart(self) -> None:
        self.passes("kept_wait")

    def test_a_compute_miss_waits_for_its_build_whatever_the_bound(self) -> None:
        self.passes("compute_wait")

    def test_a_key_is_released_once_no_program_holds_it_and_the_timeline_passed_the_release(self) -> None:
        out = self.passes("release_order")
        print("\n" + next(l for l in out if l.startswith("after the release")))

    def test_a_release_that_falls_due_while_a_build_runs_waits_for_the_build(self) -> None:
        self.passes("release_during_build")

    def test_counter_run_a_released_key_destroyed_at_once(self) -> None:
        done = self.counter_run("release_order", "release_now")
        self.assertIn("check failed", done.stderr)

    def test_the_bound_is_read_from_the_environment(self) -> None:
        self.passes("env", "waited")                                                  # unset: 2,000 ms, a 300 ms build waits
        self.passes("env", "skipped", env={"OOT2_VK_MISS_BOUND": "50"})               # 50 ms: skipped
        self.passes("env", "waited", env={"OOT2_VK_MISS_BOUND": "unbounded"})         # the override
        self.passes("env", "waited", env={"OOT2_VK_MISS_BOUND": "5000"})

    def test_a_bound_that_is_neither_unbounded_nor_milliseconds_is_a_named_fatal(self) -> None:
        for value in ("soon", "0", "-5", "600001", "12ms"):
            done = self.run_case("-O2", "env", "waited", env={"OOT2_VK_MISS_BOUND": value})
            self.assertEqual(done.returncode, EXIT_FATAL, f"{value!r}:\n{done.stdout}{done.stderr}")
            self.assertTrue(any("OOT2_VK_MISS_BOUND" in l for l in fatal_lines(done.stderr)), done.stderr)


def require_gpu() -> bool:
    return bool(os.environ.get("OOT2_REQUIRE_GPU"))


class MissDeviceTest(MissBase):
    levels = ("-O2",)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        done = tp.run("usable")
        skip_on_partial_device(done)
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        assert done.returncode == 0, done.stdout + done.stderr
        if not int(re.search(r"^usable (\d+)$", done.stdout, re.M).group(1)):
            why = "no device here has every item the renderer requires"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
            raise unittest.SkipTest(why)
        cls.dir = tp.scratch()
        cls.pack = tp.synthetic_pack(cls.dir)

    def gpu(self, case: str, *extra: str, mutant: str | None = None) -> subprocess.CompletedProcess:
        args = [case, str(self.pack), *extra] + (["--mutant", mutant] if mutant else [])
        return subprocess.run([str(self.exes["-O2"]), *args], capture_output=True, text=True, timeout=300,
                              env=case_env({"OOT2_VK_VALIDATION": "sync"}))

    def assert_clean(self, done: subprocess.CompletedProcess, case: str) -> None:
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertIn(f"ok {case}", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-3000:])

    def test_a_real_miss_in_a_frame_waits_for_the_submission_the_device_holds_before_it_builds(self) -> None:
        done = self.gpu("dev_drain")
        self.assert_clean(done, "dev_drain")
        print("\n" + next(l for l in done.stdout.splitlines() if l.startswith("miss:")))

    def test_counter_run_the_real_build_meets_the_builders_check_without_the_drain(self) -> None:
        done = self.gpu("dev_drain", mutant="nodrain")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr[-3000:])
        self.assertTrue(any("with work in flight" in l for l in fatal_lines(done.stderr)), done.stderr[-3000:])

    def test_a_pipeline_the_device_holds_in_a_command_buffer_is_destroyed_only_after_it_finished(self) -> None:
        done = self.gpu("dev_release")
        self.assert_clean(done, "dev_release")
        print("\n" + "\n".join(l for l in done.stdout.splitlines() if "resident" in l))

    def test_counter_run_a_pipeline_destroyed_while_the_device_holds_it(self) -> None:
        done = self.gpu("dev_release", mutant="release_now")
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertNotIn("ok dev_release", done.stdout)

    def test_a_build_past_the_bound_goes_on_beside_work_in_flight_and_is_counted(self) -> None:
        done = self.gpu("dev_late")
        self.assert_clean(done, "dev_late")
        print("\n" + next(l for l in done.stdout.splitlines() if l.startswith("skipped")))

    def test_counter_run_a_late_build_that_is_not_told_meets_the_builders_check(self) -> None:
        done = self.gpu("dev_late", mutant="nolate")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr[-3000:])
        self.assertTrue(any("with work in flight" in l for l in fatal_lines(done.stderr)), done.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
