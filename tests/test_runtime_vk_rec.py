# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the recorder (runtime/include/oot2/gpu_ring.h and vk_rec.h; runtime/gpu/gpu_ring.c, vk_rec.c, vk_rec_pack.c)
through tests/runtime/vk_rec_test.c.

No device needed (continuous integration runs these): the packet ring through a 4 KiB ring that wraps and fills; every
command of the recorder as a packet, made into the call with the arguments it was given from storage the caller had
overwritten; frames from two threads in value order with no frame's packets inside another's; a third frame that waits for
the first; the breadcrumbs, one numbered record per draw, which the marker names; the words the loss report uses for it.

On the desktop's RTX 3060 Ti (skipped without a device that has every item of oot2/vk_required.h, unless OOT2_REQUIRE_GPU
is set): 3,000 draws a frame on the uni layout through the ring and as direct calls, every cell of the target checked, under
synchronization validation; a third frame that waits for the first while the device holds it on a semaphore; an injected
device loss whose report names the last draw by its stages. OOT2_TIMING=1 also runs the timing of the pipeline cost study
(the submitting thread's and the recorder's time per frame against direct calls) and prints it.

Each check has a counter-run, a test-only switch of vk_rec_test that breaks the mechanism the check is for: a ring that
does not wait for room, a third frame let through, a marker one draw past its record. The check must then fail.

A source check that needs no build: the two host locks held across a submission (the texel copies' and the capture's) are
taken with rt_kernel_lock, never plat_mutex_lock.

Run: python -m unittest discover -s tests -v
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

import build_game  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_rec_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code (guest threads)
LEVELS = ("-O1", "-O2")
EXIT_NO_DEVICE = 3

VS = """#version 450
layout(push_constant) uniform PC { vec2 origin; vec2 size; vec4 color; } pc;
void main() {
    vec2 c[3] = vec2[](vec2(0.0, 0.0), vec2(2.0, 0.0), vec2(0.0, 2.0));
    vec2 p = pc.origin + c[gl_VertexIndex] * pc.size;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
FS = """#version 450
layout(push_constant) uniform PC { vec2 origin; vec2 size; vec4 color; } pc;
layout(location = 0) out vec4 o;
void main() { o = vec4(pc.color.rgb * (float(K) + 1.0) / 8.0, 1.0); }
"""
VARIANTS = 8


def require_gpu() -> bool:
    return bool(os.environ.get("OOT2_REQUIRE_GPU"))


class RecorderBase(unittest.TestCase):
    """vk_rec_test built once per class, at the levels it is given."""

    levels: tuple[str, ...] = LEVELS

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        cls.include, cls.glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-vkrec-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, cls.include, cls.glslang, TEST_SOURCE, Path(cls.tmp.name), levels=cls.levels,
                                 extra_sources=(END_SOURCE,))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, *args: str, env: dict[str, str] | None = None,
                 timeout: int = 180) -> subprocess.CompletedProcess:
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

    def counter_run(self, case: str, mutant: str, *args: str, level: str = "-O2") -> None:
        """The case with a mechanism broken must fail one of its checks (exit 1) or stop by name, never pass."""
        try:
            done = self.run_case(level, case, *args, "--mutant", mutant, timeout=90)
        except subprocess.TimeoutExpired:
            return   # a ring that overwrites what nobody read may leave its reader waiting for ever: not a pass
        self.assertNotEqual(done.returncode, 0, f"{case} passed with the {mutant} mutant:\n{done.stdout}{done.stderr}")
        self.assertNotIn(f"ok {case}", done.stdout)


class RingAndPacketTest(RecorderBase):
    """What needs no device."""

    def test_the_ring_wraps_fills_and_keeps_every_packet_whole(self) -> None:
        line = [x for x in self.passes("ring_wrap") if x.startswith("ok ring_wrap")][0]
        print("\n" + line)
        pads = int(re.search(r"pads (\d+)", line).group(1))
        waits = int(re.search(r"full_waits (\d+)", line).group(1))
        self.assertGreater(pads, 0)
        self.assertGreater(waits, 0)

    def test_a_packet_bigger_than_a_quarter_of_the_ring_is_a_named_fatal(self) -> None:
        for level in self.levels:
            done = self.run_case(level, "ring_oversize")
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            self.assertTrue(any("does not fit a ring of 4096 bytes" in x for x in fatal_lines(done.stderr)), done.stderr)

    def test_every_command_is_a_packet_made_into_the_call_it_was_given(self) -> None:
        print("\n" + [x for x in self.passes("pack_roundtrip") if x.startswith("ok pack_roundtrip")][0])

    def test_a_structure_with_a_pnext_chain_is_a_named_fatal(self) -> None:
        for level in self.levels:
            done = self.run_case(level, "pack_pnext")
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            self.assertTrue(any("a colour attachment with a pNext chain has no packet" in x for x in fatal_lines(done.stderr))
                            or any("a rendering with a pNext chain has no packet" in x for x in fatal_lines(done.stderr)),
                            done.stderr)

    def test_frames_from_two_threads_come_out_in_value_order_and_whole(self) -> None:
        self.passes("frame_order")

    def test_a_third_frame_waits_for_the_first(self) -> None:
        self.passes("frame_limit")

    def test_the_guest_reads_its_own_frames_results_with_a_window_of_2_7_and_8_frames(self) -> None:
        for which, frames in (("default", 2), ("2", 2), ("7", 7), ("8", 8)):
            line = [x for x in self.passes("window", which) if x.startswith("ok window")][0]
            print("\n" + line)
            self.assertRegex(line, rf"^ok window frames {frames} begun 45 deepest {frames} waits [1-9]\d* table reuses [1-9]\d*$")

    def test_the_window_follows_the_setting_and_says_so_when_it_is_not_the_default(self) -> None:
        for level in self.levels:
            done = self.run_case(level, "window", "setting", env={"OOT2_FRAMES_IN_FLIGHT": "7"})
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("ok window frames 7 ", done.stdout)
            self.assertIn("recorder: 7 frames in flight (the default is 2)", done.stderr)
            done = self.run_case(level, "window", "default")
            self.assertNotIn("frames in flight (the default", done.stderr)

    def test_a_window_of_more_than_the_most_is_a_named_fatal(self) -> None:
        for level in self.levels:
            done = self.run_case(level, "frames_over")
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            self.assertTrue(any("the recorder allows 8 frames in flight at most, not 9" in x for x in fatal_lines(done.stderr)),
                            done.stderr)

    def test_the_setting_is_1_to_8_and_2_by_default_and_refuses_what_is_outside(self) -> None:
        print("\n" + [x for x in self.passes("frames_range") if x.startswith("ok frames_range")][0])
        for value in ("0", "9", "-1", "two"):
            for level in self.levels:
                done = self.run_case(level, "frames_setting", env={"OOT2_FRAMES_IN_FLIGHT": value})
                self.assertEqual(done.returncode, 2, f"{value}: {done.stdout}{done.stderr}")
                self.assertIn("OOT2_FRAMES_IN_FLIGHT", done.stderr)
                self.assertNotIn("ok frames_setting", done.stdout)
        for value in ("1", "8"):
            done = self.run_case("-O2", "frames_setting", env={"OOT2_FRAMES_IN_FLIGHT": value})
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn(f"ok frames_setting {value}\n", done.stdout)

    def test_a_guest_thread_gives_its_core_up_while_a_frame_waits_for_an_older_one(self) -> None:
        self.passes("core_wait")

    def test_a_guest_thread_waiting_for_the_gate_behind_such_a_frame_gives_its_core_up_too(self) -> None:
        self.passes("core_gate")

    def test_a_guest_thread_waiting_for_a_lock_held_across_a_submission_gives_its_core_up(self) -> None:
        # A thread holds a host mutex and waits for the gate, without its core; a second thread on that core takes the mutex.
        # Taken with its core held, the second keeps the core from the frame that holds the gate and the case hangs until
        # its watchdog ends it (exit 3, a line that names the case).
        self.passes("core_lock")

    def test_breadcrumbs_name_their_own_draw(self) -> None:
        print("\n" + [x for x in self.passes("breadcrumbs") if x.startswith("ok breadcrumbs")][0])

    def test_no_breadcrumb_without_the_checkpoints(self) -> None:
        self.passes("breadcrumbs_off")

    def test_the_marker_in_words(self) -> None:
        print("\n" + [x for x in self.passes("marker_text") if x.startswith("ok marker_text")][0])

    # ---- the counter-runs
    def test_counter_run_a_ring_that_overwrites_unread_packets(self) -> None:
        self.counter_run("ring_wrap", "overwrite")

    def test_counter_run_a_third_frame_let_through(self) -> None:
        self.counter_run("frame_limit", "nolimit")

    def test_counter_run_a_deep_window_let_through_reads_an_older_frame(self) -> None:
        for frames in ("7", "8"):
            self.counter_run("window", "nolimit", frames)

    def test_counter_run_a_window_narrower_than_the_device_expects_is_a_stall_and_is_detected(self) -> None:
        for frames in ("2", "7"):
            done = self.run_case("-O2", "window", frames, "--mutant", "narrow", timeout=90)
            self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertNotIn("ok window", done.stdout)

    def test_counter_run_a_breadcrumb_naming_the_wrong_draw(self) -> None:
        self.counter_run("breadcrumbs", "skew")


class LockSitesTest(unittest.TestCase):
    """The host locks held across a submission are taken with rt_kernel_lock. A source check: no build, no device.

    core_lock proves that rt_kernel_lock and the gate's yield work on a private mutex of the test. It cannot tell that the
    real locks are taken that way, and one take that went back to plat_mutex_lock brings the deadlock back (a guest thread
    holds the lock and has given its core up for the gate, a second one on that core waits for the lock with the core held)
    while every other test stays green. The two locks are the texel copies' staging lock (g_xfer_mu, vk_image.c) and the
    capture's (g_mu, vk_capture.c). The other mutex of vk_image.c, g_mu, guards its counters for a few instructions and is
    not one of them.
    """

    GPU = ROOT / "runtime" / "gpu"
    SITES = (("vk_image.c", "g_xfer_mu"), ("vk_capture.c", "g_mu"))
    CALLS = ("rt_kernel_lock", "plat_mutex_lock", "plat_mutex_trylock")

    @classmethod
    def takes(cls, source: str, mutex: str) -> dict[str, int]:
        """How many times each locking call takes `mutex` in the code of `source` (a comment is not a take)."""
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
        return {call: len(re.findall(rf"\b{call}\(\s*&{mutex}\s*\)", code)) for call in cls.CALLS}

    def test_every_take_of_a_lock_held_across_a_submission_is_rt_kernel_lock(self) -> None:
        for name, mutex in self.SITES:
            with self.subTest(file=name, mutex=mutex):
                n = self.takes((self.GPU / name).read_text(encoding="utf-8"), mutex)
                self.assertGreater(n["rt_kernel_lock"], 0, f"{name} has no rt_kernel_lock(&{mutex}): renamed? change SITES")
                self.assertEqual(n["plat_mutex_lock"] + n["plat_mutex_trylock"], 0,
                                 f"{name} takes {mutex} with plat_mutex_lock or plat_mutex_trylock {n}: use rt_kernel_lock")

    def test_counter_run_a_take_that_went_back_to_plat_mutex_lock_is_seen_and_a_comment_is_not_a_take(self) -> None:
        good = ("static PlatMutex g_xfer_mu, g_mu;\n"
                "/* plat_mutex_lock(&g_xfer_mu) */ // plat_mutex_lock(&g_xfer_mu)\n"
                "void f(void) { rt_kernel_lock(&g_xfer_mu); plat_mutex_lock(&g_mu); plat_mutex_unlock(&g_xfer_mu); }\n")
        bad = good.replace("rt_kernel_lock(&g_xfer_mu);", "plat_mutex_lock( &g_xfer_mu );")
        self.assertEqual(self.takes(good, "g_xfer_mu"), {"rt_kernel_lock": 1, "plat_mutex_lock": 0, "plat_mutex_trylock": 0})
        self.assertEqual(self.takes(bad, "g_xfer_mu"), {"rt_kernel_lock": 0, "plat_mutex_lock": 1, "plat_mutex_trylock": 0})
        self.assertEqual(self.takes(good, "g_mu"), {"rt_kernel_lock": 0, "plat_mutex_lock": 1, "plat_mutex_trylock": 0})


class DeviceTest(RecorderBase):
    """The recorder on the device, over synthetic pipelines on the uni layout."""

    levels = ("-O2",)

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        done = subprocess.run([str(cls.exes["-O2"]), "dev_usable"], capture_output=True, text=True, timeout=120,
                              env=case_env())
        skip_on_partial_device(done)
        cls.usable = done.returncode == 0 and "usable 1" in done.stdout
        if not cls.usable:
            why = f"no Vulkan device here that has every item the renderer requires ({done.stderr.strip()[-200:]})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        folder = Path(cls.tmp.name) / "shaders"
        folder.mkdir()
        (folder / "vs.vert").write_text(VS)
        (folder / "fs.frag").write_text(FS)
        flags = list(build_game.GLSLANG_FLAGS)
        for name, source, extra in [("vs", "vs.vert", [])] + [(f"fs{k}", "fs.frag", [f"-DK={k}"]) for k in range(VARIANTS)]:
            done = subprocess.run([cls.glslang, *flags, *extra, "-o", str(folder / f"{name}.spv"), str(folder / source)],
                                  capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f"glslang failed on {source}:\n{done.stdout}{done.stderr}")
        cls.shaders = str(folder)

    def sync_env(self) -> dict[str, str]:
        return {"OOT2_VK_VALIDATION": "sync"}

    def assert_clean(self, done: subprocess.CompletedProcess, case: str) -> None:
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn("validation errors 0 ", done.stdout)

    def test_3000_draws_through_the_ring_and_as_direct_calls_fill_every_cell(self) -> None:
        done = self.run_case("-O2", "dev_draws", self.shaders, env=self.sync_env())
        self.assert_clean(done, "dev_draws")
        self.assertIn("3000 of 3000 cells right", done.stdout)
        self.assertEqual(done.stdout.count("3000 of 3000 cells right"), 2)   # the ring's frames and the direct ones
        print("\n" + "\n".join(x for x in done.stdout.splitlines() if x.startswith(("ring:", "direct:", "frame:"))))

    def test_a_third_frame_waits_for_the_first_which_the_device_holds(self) -> None:
        done = self.run_case("-O2", "dev_third", self.shaders, env=self.sync_env())
        self.assert_clean(done, "dev_third")

    def loss_report(self, *extra: str) -> tuple[str, re.Match]:
        with tempfile.TemporaryDirectory(prefix="oot2-vkrec-loss-", ignore_cleanup_errors=True) as tmp:
            done = self.run_case("-O2", "dev_loss", self.shaders, tmp, *extra)
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            reports = sorted((Path(tmp) / "logs").glob("device-lost-*.txt"))
            self.assertEqual(len(reports), 1, done.stderr)
            m = re.search(r"^loss value (\d+) draw17 (\S+) (\S+) draw15 (\S+) (\S+)$", done.stdout, re.M)
            self.assertIsNotNone(m, done.stdout)
            return reports[0].read_text(encoding="utf-8"), m

    @staticmethod
    def names_draws(report: str, m: re.Match) -> bool:
        value, vs17, ps17, vs15, ps15 = m.groups()
        last = f"last draw the queue reached: draw 17 of submission {value} (draw): vs {vs17}, ps {ps17}"
        oldest = f"oldest draw a pipeline stage was still in: draw 15 of submission {value} (draw): vs {vs15}, ps {ps15}"
        return last in report and oldest in report

    def test_an_injected_loss_names_the_last_draw_by_its_stages(self) -> None:
        report, m = self.loss_report()
        self.assertTrue(self.names_draws(report, m), report)
        print("\n" + "\n".join(x for x in report.splitlines() if "draw 1" in x))

    def test_counter_run_a_marker_that_names_the_wrong_draw(self) -> None:
        report, m = self.loss_report("--mutant", "skew")
        self.assertFalse(self.names_draws(report, m), report)

    def test_counter_run_a_third_frame_let_through_on_the_device(self) -> None:
        done = self.run_case("-O2", "dev_third", self.shaders, "--mutant", "nolimit", env=self.sync_env())
        self.assertNotEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("ok dev_third", done.stdout)

    @unittest.skipUnless(os.environ.get("OOT2_TIMING"), "a timing run needs the machine alone: set OOT2_TIMING=1")
    def test_timing_of_the_pipeline_cost_study(self) -> None:
        for label, env in (("breadcrumbs on", {}), ("breadcrumbs off", {"OOT2_VK_FAULT_REPORT": "off"})):
            done = self.run_case("-O2", "dev_timing", self.shaders, env=env, timeout=600)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            print(f"\n== {label}\n" + "\n".join(x for x in done.stdout.splitlines() if x.startswith(("timing", "breadcrumbs"))))


if __name__ == "__main__":
    unittest.main()
