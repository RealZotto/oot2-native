# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The pipeline cache that the game saves by itself (runtime/gpu/vk_pipeline.c, "the cache saves itself"), through
tests/runtime/vk_pipeline_test.c: a run that stops leaves the cache it built in the game folder's
pipelines/pipeline.cache, so that the next run of that folder starts warm without a prepare step.

Without a device (continuous integration runs them): the two settings, OOT2_VK_CACHE_SAVE and
OOT2_VK_CACHE_SAVE_SECONDS, are judged and a value outside them is refused by name.

On a device with every item of oot2/vk_required.h, with the synthetic pack of tests/test_runtime_vk_pipeline.py and a game
folder opened the way the game opens it (gpu_vk_open_game): a named stop with exit 70 saves and the next run is served
every pipeline by the file; a clean exit saves; a run that made one more pipeline saves the growth; a run that made
nothing new does not write the file again; the timer saves a run that is killed from outside (no stop hook), and with the
timer off a killed run leaves nothing; OOT2_VK_CACHE_SAVE=off saves nothing; a save after the cache was flushed to a
segment (what a builder does) is refused by name and the file stays as it was; the folder is made when it is not there;
two processes saving into one folder at once each write a whole file; a temporary file of the process's own name left by
an earlier process is removed at start, and so are those of processes that are gone (never those of a live one); a move
that meets another process reading the cache is tried again for about a second; a cache file that was there and could not
be read or opened is not replaced by the run's part (an empty one is, and a file that is not there is made); the timer names
a refusal once, not every round; a stop while another thread is inside a long compile saves in milliseconds, without waiting
for it (measured on one driver, and skipped when the compile is too short to tell).

Run: python -m unittest discover -s tests -p test_runtime_vk_cache_save.py -v
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import time
import unittest
from pathlib import Path

import re

from test_runtime_gpu import EXIT_FATAL, case_env, fatal_lines
from test_runtime_vk_pipeline import (GLSL, SYNTH_KEYS, SYNTH_STAGE, DeviceTestCase, _build, cp, describe, io, program, run,
                                      write_pack)

HEADER = struct.Struct("<IIII16s")   # VkPipelineCacheHeaderVersionOne
CACHE = Path("pipelines") / "pipeline.cache"
LONG_KEY = "5100000000000010.00000000"      # tests/runtime/vk_pipeline_test.c, LONG_KEY
LONG_OPERATIONS = 20000                     # dependent operations in the long compute stage: a compile of seconds on this driver
# (5,000 took 0.7 s on it on 2026-11-10, too short for the test below to tell anything)
LONG_COMPILE_MS = 1000                      # under this the compile is too short for the stop test to mean anything


class SaveSettingsTest(unittest.TestCase):
    def test_the_two_variables_are_judged_and_a_value_outside_them_is_refused_by_name(self) -> None:
        done = run("savesettings")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = [line for line in done.stdout.splitlines() if line.startswith("save [")]
        self.assertEqual(lines, [
            "save [(unset)] [(unset)]: enabled 1 seconds 120",
            "save [] []: enabled 1 seconds 120",
            "save [on] [0]: enabled 1 seconds 0",
            "save [off] [5]: enabled 0 seconds 5",
            "save [on] [86400]: enabled 1 seconds 86400",
            "save [(unset)] [1]: enabled 1 seconds 1",
            'save [maybe] [(unset)]: refused: OOT2_VK_CACHE_SAVE is "maybe"; the values are on and off',
            'save [On] [(unset)]: refused: OOT2_VK_CACHE_SAVE is "On"; the values are on and off',
            *(f'save [(unset)] [{v}]: refused: OOT2_VK_CACHE_SAVE_SECONDS is "{v}"; the values are whole seconds from 0 to '
              f"86400 (0: no timer)" for v in ("86401", "-1", "12x", "1.5", " 5", "99999999999999999999")),
        ])
        self.assertIn("ok savesettings", done.stdout)


class SaveDeviceTest(DeviceTestCase):
    def game(self, name: str) -> Path:
        """A game folder with the synthetic pack as its base pack and no pipelines/ folder, like a folder nothing prepared."""
        folder = self.dir / name
        shutil.rmtree(folder, ignore_errors=True)
        (folder / "shaders").mkdir(parents=True)
        shutil.copy(self.pack, folder / "shaders" / "shader-base.bin")
        return folder

    def save_run(self, folder: Path, mode: str, *extra: str, **env: str):
        return run("synth_save", str(folder), mode, *extra, env={"OOT2_VK_VALIDATION": "off", **env}, timeout=120)

    @staticmethod
    def stats(out: str) -> dict[str, int]:
        return DeviceTestCase.stats(out, "save")

    def assert_header(self, cache: Path) -> None:
        data = cache.read_bytes()
        self.assertGreater(len(data), 32)
        size, version, *_ = HEADER.unpack_from(data)
        self.assertEqual((size, version), (32, 1))

    def assert_all_served(self, done) -> None:
        s = self.stats(done.stdout)
        self.assertEqual((s["stage_library_hits"], s["compute_hits"]), (s["stage_libraries"], s["computes"]), s)
        self.assertGreater(s["computes"], 0)

    def test_a_named_stop_with_exit_70_saves_and_the_next_run_starts_warm(self) -> None:
        folder = self.game("stop70")
        cache = folder / CACHE
        self.assertFalse(cache.parent.exists())
        first = self.save_run(folder, "fatal")
        self.assertEqual(first.returncode, EXIT_FATAL, first.stdout + first.stderr)
        self.assertEqual(fatal_lines(first.stderr), ["FATAL: synth_save: a named stop"])
        self.assertIn("cache_state 0", first.stdout)
        self.assertIn("the pipeline cache was saved to ", first.stderr)
        self.assertIn(" at the stop (exit 70): ", first.stderr)
        self.assert_header(cache)                                       # the folder was made, the file is a cache
        cold = self.stats(first.stdout)
        self.assertEqual((cold["stage_library_hits"], cold["compute_hits"]), (0, 0), cold)
        before = (cache.read_bytes(), cache.stat().st_mtime_ns)
        second = self.save_run(folder, "fatal")                         # the next run of the same folder
        self.assertEqual(second.returncode, EXIT_FATAL, second.stdout + second.stderr)
        self.assertIn("cache_state 1", second.stdout)
        self.assertIn("pipeline cache seeded", second.stderr)
        self.assertRegex(second.stderr, r"pipeline\.cache seeds \d+ bytes \(loaded in \d+ ms\)")   # the launch's load time
        self.assert_all_served(second)                                  # every miss of the first run is a hit now
        self.assertIn("the pipeline cache is not saved at the stop (exit 70): it holds ", second.stderr)
        self.assertEqual((cache.read_bytes(), cache.stat().st_mtime_ns), before)   # nothing new, nothing written
        print("\n" + first.stderr[-900:] + second.stderr[-900:])

    def test_a_clean_exit_through_the_stop_hook_saves_too(self) -> None:
        folder = self.game("exit0")
        done = self.save_run(folder, "exit")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn(" at the stop (exit 0): ", done.stderr)
        self.assert_header(folder / CACHE)

    def test_a_run_that_made_one_more_pipeline_saves_the_growth_and_the_next_run_is_served_it(self) -> None:
        folder = self.game("growth")
        cache = folder / CACHE
        self.save_run(folder, "fatal")
        small = cache.stat().st_size
        grown = self.save_run(folder, "fatal", "more")                  # one more compute pipeline than the file holds
        s = self.stats(grown.stdout)
        self.assertEqual((s["computes"], s["compute_hits"]), (2, 1), s)
        self.assertIn("the pipeline cache was saved to ", grown.stderr)
        self.assertGreater(cache.stat().st_size, small)
        third = self.save_run(folder, "fatal", "more")
        s = self.stats(third.stdout)
        self.assertEqual((s["computes"], s["compute_hits"]), (2, 2), s)

    def test_the_timer_saves_a_run_that_is_killed_and_with_the_timer_off_it_leaves_nothing(self) -> None:
        timed = self.game("timer")
        done = self.save_run(timed, "timer", OOT2_VK_CACHE_SAVE_SECONDS="1")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("the cache file was there after", done.stdout)
        self.assertIn(" at the timer: ", done.stderr)
        self.assertNotIn(" at the stop", done.stderr)                   # it ended with no stop hook
        self.assert_header(timed / CACHE)
        killed = self.game("killed")
        done = self.save_run(killed, "kill", OOT2_VK_CACHE_SAVE_SECONDS="0")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertFalse((killed / CACHE).exists())

    def test_the_setting_off_saves_nothing_and_says_so(self) -> None:
        folder = self.game("off")
        done = self.save_run(folder, "fatal", OOT2_VK_CACHE_SAVE="off")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("OOT2_VK_CACHE_SAVE off: the pipeline cache is not saved", done.stderr)
        self.assertFalse((folder / CACHE).exists())

    def test_a_save_after_the_cache_was_flushed_is_refused_by_name_and_the_file_stays_whole(self) -> None:
        folder = self.game("flushed")
        cache = folder / CACHE
        self.save_run(folder, "fatal")                                  # the seed a builder would start from
        before = cache.read_bytes()
        done = self.save_run(folder, "flushed")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("cache_state 1", done.stdout)
        refused = [line for line in done.stdout.splitlines() if line.startswith("save refused: ")]
        self.assertEqual(len(refused), 1, done.stdout)
        self.assertIn("the cache was flushed to a segment (vkp_cache_flush)", refused[0])
        self.assertIn("the pipeline cache was NOT saved at the stop (exit 70): the cache was flushed to a segment", done.stderr)
        self.assertEqual(cache.read_bytes(), before)                    # the seed is not replaced by the part built since
        self.assertTrue((folder / "segment.bin").is_file())

    @staticmethod
    def half_written(pipelines: Path) -> bool:
        try:
            return any(p.suffix == ".tmp" and p.stat().st_size > 0 for p in pipelines.glob("*"))
        except OSError:
            return False

    def test_two_processes_saving_into_one_folder_each_write_a_whole_file_and_the_later_rename_wins(self) -> None:
        folder = self.game("two")
        pipelines = folder / "pipelines"
        slow = subprocess.Popen([str(program()), "synth_save", str(folder), "exit", "more", "stall=12000"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=case_env({"OOT2_VK_VALIDATION": "off"}))
        try:
            deadline = time.time() + 60
            while time.time() < deadline and slow.poll() is None and not self.half_written(pipelines):
                time.sleep(0.05)                                    # until its temporary file is half written
            if slow.poll() is not None:
                self.fail("the slow writer ended before it paused: " + slow.communicate()[1][-1500:])
            quick = self.save_run(folder, "exit")                    # saves a smaller cache while the slow one is paused
            out, err = slow.communicate(timeout=120)
        finally:
            if slow.poll() is None:
                slow.kill()
        for name, code, log in (("quick", quick.returncode, quick.stderr), ("slow", slow.returncode, err)):
            self.assertEqual(code, 0, name + ": " + log[-1500:])
            self.assertIn("the pipeline cache was saved to ", log, name)
            self.assertNotIn("NOT saved", log, name)
        self.assertEqual([p.name for p in pipelines.glob("*.tmp")], [])      # nobody's temporary file is left
        third = self.save_run(folder, "exit", "more")                        # the file is the slow writer's, whole
        self.assertIn("cache_state 1", third.stdout)
        s = self.stats(third.stdout)
        self.assertEqual((s["computes"], s["compute_hits"]), (2, 2), s)
        self.assert_all_served(third)

    def test_a_temporary_file_of_this_process_name_left_by_an_earlier_one_is_removed_at_start(self) -> None:
        folder = self.game("stale")
        (folder / "pipelines").mkdir()
        done = self.save_run(folder, "exit", "stale")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("the stale temporary file was removed", done.stdout)
        self.assertEqual([p.name for p in (folder / "pipelines").glob("*.tmp")], [])

    def test_a_cache_file_that_could_not_be_read_at_launch_is_not_replaced_by_the_runs_part(self) -> None:
        folder = self.game("unread")
        cache = folder / CACHE
        self.save_run(folder, "fatal")
        before = cache.read_bytes()
        done = self.save_run(folder, "fatal", "more", f"readcap={len(before) - 1}")   # one byte too large to read
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("cache_state 0", done.stdout)                              # it started empty
        self.assertIn("could not be read (too large for memory, or unreadable): starting empty, and no save will replace it",
                      done.stderr)
        self.assertIn("the pipeline cache was NOT saved at the stop (exit 70): the cache file ", done.stderr)
        self.assertIn(" could not be read at launch", done.stderr)
        self.assertEqual(cache.read_bytes(), before)                             # the file is as it was
        read = self.save_run(folder, "fatal", "more", f"readcap={len(before)}")  # the green counterpart: it is read, and grown
        self.assertIn("cache_state 1", read.stdout)
        self.assertIn("the pipeline cache was saved to ", read.stderr)
        self.assertGreater(cache.stat().st_size, len(before))

    def test_a_temporary_file_of_a_process_that_is_gone_is_removed_at_start_and_a_live_ones_is_kept(self) -> None:
        folder = self.game("deadtemp")
        pipelines = folder / "pipelines"
        pipelines.mkdir()
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        dead, alive = f"pipeline.cache.{gone.pid}.tmp", f"pipeline.cache.{os.getpid()}.tmp"   # this test's own process lives
        others = ["pipeline.cache.abc.tmp", f"pipeline.cache.{gone.pid}.tmpx", f"other.cache.{gone.pid}.tmp", "pipeline.cache.tmp"]
        for name in (dead, alive, *others):
            (pipelines / name).write_bytes(b"half a cache")
        done = self.save_run(folder, "exit")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertFalse((pipelines / dead).exists(), "the temporary file of a process that is gone is still there")
        self.assertIn(f"{dead}: the temporary file of a save by process {gone.pid}, which is gone", done.stderr)
        self.assertEqual((pipelines / alive).read_bytes(), b"half a cache")              # a live process may be writing it
        for name in others:
            self.assertEqual((pipelines / name).read_bytes(), b"half a cache", name)     # not a temporary file of a save

    def test_a_cache_file_that_could_not_be_opened_is_not_replaced_and_one_that_is_not_there_is_made(self) -> None:
        folder = self.game("denied")
        cache = folder / CACHE
        self.save_run(folder, "fatal")
        before = cache.read_bytes()
        done = self.save_run(folder, "fatal", "more", "openerr=13")                     # EACCES: it is there, held by someone
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("cache_state 0", done.stdout)
        self.assertRegex(done.stderr, r"the pipeline cache .* could not be opened \(.+\): starting empty, and no save will "
                                      r"replace it")
        self.assertIn("the pipeline cache was NOT saved at the stop (exit 70): the cache file ", done.stderr)
        self.assertEqual(cache.read_bytes(), before)                                   # the file is as it was
        absent = self.save_run(folder, "fatal", "more", "openerr=2")                    # ENOENT: not there, the counterpart
        self.assertEqual(absent.returncode, EXIT_FATAL, absent.stdout + absent.stderr)
        self.assertNotIn("could not be opened", absent.stderr)
        self.assertIn("the pipeline cache was saved to ", absent.stderr)

    def test_the_timer_names_each_reason_of_a_refusal_once_not_every_round(self) -> None:
        for name, option in (("unreadable", "openerr=13"), ("flushed", "flush")):
            with self.subTest(refusal=name):
                folder = self.game("rounds-" + name)
                self.save_run(folder, "fatal")                                          # a cache file the timer must not replace
                before = (folder / CACHE).read_bytes()
                done = self.save_run(folder, "wait", "more", option, "wait=4500", OOT2_VK_CACHE_SAVE_SECONDS="1")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                said = [line for line in done.stderr.splitlines() if "was NOT saved at the timer" in line]
                self.assertEqual(len(said), 1, done.stderr[-1500:])                     # four rounds went by: one line
                self.assertEqual((folder / CACHE).read_bytes(), before)

    @unittest.skipUnless(os.name == "nt", "the sharing violation is Windows'")
    def test_a_move_that_meets_another_process_reading_the_cache_is_tried_again(self) -> None:
        folder = self.game("busy")
        cache = folder / CACHE
        pipelines = folder / "pipelines"
        self.save_run(folder, "fatal")
        small = cache.stat().st_size
        slow = subprocess.Popen([str(program()), "synth_save", str(folder), "exit", "more", "stall=1500"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=case_env({"OOT2_VK_VALIDATION": "off"}))
        try:
            deadline = time.time() + 60
            while time.time() < deadline and slow.poll() is None and not self.half_written(pipelines):
                time.sleep(0.02)
            if slow.poll() is not None:
                self.fail("the writer ended before it paused: " + slow.communicate()[1][-1500:])
            with cache.open("rb") as reader:      # a reader without delete sharing, as the game's own read of the cache is
                reader.read(1)
                time.sleep(2.0)                   # past the end of the pause (1.5 s): the first tries of the move meet it
            out, err = slow.communicate(timeout=120)
        finally:
            if slow.poll() is None:
                slow.kill()
        self.assertEqual(slow.returncode, 0, err[-1500:])
        self.assertIn("the pipeline cache was saved to ", err)
        self.assertNotIn("NOT saved", err)
        self.assertGreater(cache.stat().st_size, small)                                 # the run's growth was not lost
        self.assertEqual([p.name for p in pipelines.glob("*.tmp")], [])

    def test_an_empty_cache_file_holds_nothing_to_lose_and_is_saved_over(self) -> None:
        folder = self.game("empty")
        (folder / "pipelines").mkdir()
        (folder / CACHE).write_bytes(b"")
        done = self.save_run(folder, "fatal")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("is empty: starting empty", done.stderr)
        self.assertIn("the pipeline cache was saved to ", done.stderr)
        self.assert_header(folder / CACHE)

    def long_game(self, name: str, salt: int = 0) -> Path:
        """A game folder whose pack holds the synthetic stages and one compute stage that takes seconds to compile. The salt
        changes the stage's constants: two games with different salts never share an entry of the driver's own disk cache."""
        from build_game import GLSLANG_FLAGS  # tools/build_game.py is on the path through test_runtime_gpu
        program()
        work = self.dir / (name + "-pack")
        work.mkdir()
        body = "\n".join(f"    a = a * 1664525u + b; b = (b ^ (a >> {1 + i % 7})) + {i + salt}u;" for i in range(LONG_OPERATIONS))
        sources = dict(GLSL)
        sources["long.comp"] = ("#version 450\nlayout(local_size_x = 32) in;\n"
                                "layout(set = 1, binding = 9) buffer Out { uint v[]; } o;\n"
                                "layout(set = 1, binding = 1) uniform Bank { uvec4 k; } bank;\n"
                                "void main() {\n    uint a = gl_GlobalInvocationID.x + bank.k.x;\n    uint b = bank.k.y;\n"
                                + body + "\n    o.v[gl_GlobalInvocationID.x] = a ^ b;\n}\n")
        keys, stages = dict(SYNTH_KEYS, **{"long.comp": LONG_KEY}), dict(SYNTH_STAGE, **{"long.comp": "cs"})
        entries = []
        for stem, src in sources.items():
            (work / stem).write_text(src, encoding="utf-8")
            spv = work / (stem + ".spv")
            done = subprocess.run([_build["glslang"], *GLSLANG_FLAGS, "-o", str(spv), str(work / stem)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"glslang on {stem}:\n{done.stdout}{done.stderr}")
            mod = spv.read_bytes()
            kw: dict = {}
            if stem == "vs.vert":
                kw = {"outputs": [io(0)]}
            elif stem == "ps.frag":
                kw = {"inputs": [io(0), io(1)], "outputs": [io(0)]}
            entries.append(cp.Entry(keys[stem], mod, describe(keys[stem], stages[stem], mod, **kw)))
        folder = self.dir / name
        (folder / "shaders").mkdir(parents=True)
        shutil.copy(write_pack(work / "long.bin", entries), folder / "shaders" / "shader-base.bin")
        return folder

    def test_a_stop_during_a_long_compile_on_another_thread_saves_without_waiting_for_it(self) -> None:
        # Measured on this driver only: it hands the cache's data out while a compile into it is under way. Another driver may
        # wait for the compile, and then this test is the one to say so.
        env = {"OOT2_VK_VALIDATION": "off"}
        timed = run("synth_stopcompile", str(self.long_game("compile-alone")), "compile", env=env, timeout=120)
        self.assertEqual(timed.returncode, 0, timed.stdout + timed.stderr)
        compile_ms = int(re.search(r"the long compile ended after (\d+) ms", timed.stdout).group(1))
        if compile_ms < LONG_COMPILE_MS:
            self.skipTest(f"the compile takes {compile_ms} ms here, under {LONG_COMPILE_MS} ms: too short to tell a save that "
                          f"waits for it from one that does not")
        folder = self.long_game("compile-stop", salt=7)      # other constants: the alone run left nothing of this compile in the driver's disk cache
        stop_after = compile_ms // 4                          # a fraction of the compile: the stop comes while it is under way
        done = run("synth_stopcompile", str(folder), "stop", str(stop_after), env=env, timeout=120)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertNotIn("the long compile ended", done.stdout)               # the stop came while it was under way
        saved = re.search(r"the pipeline cache was saved to .* at the stop \(exit 70\): \d+ bytes .* in (\d+) ms", done.stderr)
        self.assertIsNotNone(saved, done.stderr[-1500:])
        # the driver reads a cache while another thread compiles into it: the save takes milliseconds, not the compile
        self.assertLess(int(saved.group(1)), max(200, compile_ms // 2), (saved.group(1), compile_ms))
        self.assert_header(folder / CACHE)

    def test_a_setting_outside_its_values_stops_the_start_by_name(self) -> None:
        folder = self.game("bad")
        done = self.save_run(folder, "fatal", OOT2_VK_CACHE_SAVE="maybe")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr), ['FATAL: gfx: vulkan: OOT2_VK_CACHE_SAVE is "maybe"; the values are on and off'])
        self.assertNotIn("cache_state", done.stdout)


if __name__ == "__main__":
    unittest.main()
