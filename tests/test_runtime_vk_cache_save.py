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
segment (what a builder does) is refused by name and the file stays as it was; the folder is made when it is not there.

Run: python -m unittest discover -s tests -p test_runtime_vk_cache_save.py -v
"""

from __future__ import annotations

import shutil
import struct
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, fatal_lines
from test_runtime_vk_pipeline import DeviceTestCase, run

HEADER = struct.Struct("<IIII16s")   # VkPipelineCacheHeaderVersionOne
CACHE = Path("pipelines") / "pipeline.cache"


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

    def test_a_setting_outside_its_values_stops_the_start_by_name(self) -> None:
        folder = self.game("bad")
        done = self.save_run(folder, "fatal", OOT2_VK_CACHE_SAVE="maybe")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr), ['FATAL: gfx: vulkan: OOT2_VK_CACHE_SAVE is "maybe"; the values are on and off'])
        self.assertNotIn("cache_state", done.stdout)


if __name__ == "__main__":
    unittest.main()
