# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the view decision of the Vulkan back end (runtime/gpu/vk_view_plan.c, oot2/vk_view_plan.h), through
tests/runtime/vk_view_plan_test.c. They need no device (continuous integration runs them); without clang or the Vulkan
headers they skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set.

The three kinds of validation errors the first forest run counted on vkCreateImageView, one class each:

- the 2D view of a 3D image: the volume is made 2D-array compatible (a render target: a slice or a run of slices) and
  2D-view compatible where the device has image2DViewOf3D (a storage image over a slice), and an image made without them
  gives no 2D view;
- a first layer or a layer count past the image: a volume counts depth slices of the level, any other image its layers, and
  a count that wrapped is past too;
- a view usage the image was made without (colour or depth attachment, storage): refused by name.

OOT2_VIEW_PLAN_SOURCE names another vk_view_plan.c to build the test against; the counter-run of each kind builds a copy of
the source with that kind's rule taken out and expects exactly that class red.

Run: python -m unittest discover -s tests -p test_runtime_vk_view_plan.py -v
"""

from __future__ import annotations

import atexit
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXE, ROOT, clang_or_skip, vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_view_plan_test.c"
PLAN_SOURCE = ROOT / "runtime" / "gpu" / "vk_view_plan.c"

_results: dict[str, str] = {}


def results() -> dict[str, str]:
    """Check name to its line's verdict ("ok" or "FAIL ..."), from one run of the test program."""
    if not _results:
        cc = clang_or_skip()
        include, _ = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-viewplan-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        source = Path(os.environ.get("OOT2_VIEW_PLAN_SOURCE", PLAN_SOURCE))
        exe = Path(tmp.name) / ("vk_view_plan_test" + EXE)
        cmd = [cc, "-std=gnu11", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "runtime" / "include"),
               "-I", str(include), "-I", str(include / "Volk"), str(TEST_SOURCE), str(source), "-o", str(exe)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"build of the view plan test failed:\n{done.stdout}{done.stderr}")
        ran = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        for line in ran.stdout.splitlines():
            name, _, verdict = line.partition(": ")
            _results[name] = verdict
        if not _results:
            raise RuntimeError(f"the view plan test printed nothing (exit {ran.returncode}): {ran.stderr}")
    return _results


class ViewPlanKind(unittest.TestCase):
    prefix = ""

    def checks(self) -> dict[str, str]:
        found = {k: v for k, v in results().items() if k.startswith(self.prefix)}
        self.assertGreater(len(found), 3, f"the {self.prefix} checks are missing")
        return found

    def test_every_check_holds(self) -> None:
        if not self.prefix:
            self.skipTest("the base class")
        bad = {k: v for k, v in self.checks().items() if v != "ok"}
        self.assertEqual(bad, {})


class VolumeViewTest(ViewPlanKind):
    """A 2D view of a 3D image exists only on an image made with the compatibility flags."""
    prefix = "volume_"

    def test_a_volume_is_made_2d_array_compatible_and_2d_view_compatible_with_the_feature(self) -> None:
        r = results()
        for name in ("volume_flags_array_compatible", "volume_flags_view_compatible_with_feature",
                     "volume_flags_view_compatible_not_without_feature", "volume_flags_none_for_2d"):
            self.assertEqual(r[name], "ok", name)

    def test_a_render_target_of_a_volume_is_a_2d_view_of_its_slices(self) -> None:
        r = results()
        for name in ("volume_target_is_a_2d_view_of_one_slice", "volume_target_of_slices_is_a_2d_array_view",
                     "volume_target_refused_without_the_array_flag"):
            self.assertEqual(r[name], "ok", name)

    def test_a_storage_image_over_a_slice_needs_the_device_feature(self) -> None:
        r = results()
        for name in ("volume_storage_slice_is_a_2d_view_with_the_feature",
                     "volume_storage_slice_refused_without_the_feature",
                     "volume_storage_slice_refused_when_the_device_feature_is_off",
                     "volume_storage_array_view_of_one_slice_is_a_2d_view", "volume_storage_run_of_slices_refused"):
            self.assertEqual(r[name], "ok", name)

    def test_the_other_view_types(self) -> None:
        r = results()
        for name in ("volume_3d_view_covers_the_volume", "volume_2d_image_view_types", "volume_cube_needs_its_flag"):
            self.assertEqual(r[name], "ok", name)


class LayerBoundTest(ViewPlanKind):
    """A first layer or a count past the image's layers (a volume's depth slices) is a refusal, never a view."""
    prefix = "layers_"

    def test_a_volume_slice_past_its_depth_at_the_level_is_refused(self) -> None:
        r = results()
        for name in ("layers_volume_last_slice_ok", "layers_volume_slice_past_depth_refused",
                     "layers_volume_run_past_depth_refused", "layers_volume_depth_halves_per_level",
                     "layers_volume_level_past_levels_refused"):
            self.assertEqual(r[name], "ok", name)

    def test_a_layer_past_a_layered_image_is_refused(self) -> None:
        r = results()
        for name in ("layers_2d_array_last_layer_ok", "layers_2d_array_layer_past_refused",
                     "layers_2d_array_run_past_refused", "layers_single_layer_image_layer_1_refused",
                     "layers_wrapped_rest_refused"):
            self.assertEqual(r[name], "ok", name)


class UsageTest(ViewPlanKind):
    """A view usage the image was made without is a refusal by name."""
    prefix = "usage_"

    def test_a_usage_outside_the_images_is_refused_and_a_subset_accepted(self) -> None:
        r = results()
        for name in ("usage_color_attachment_refused_on_an_image_without_it",
                     "usage_depth_attachment_refused_on_an_image_without_it",
                     "usage_storage_refused_on_an_image_without_it", "usage_subset_accepted",
                     "usage_volume_without_color_attachment_refused", "usage_gap_names_the_missing_bits"):
            self.assertEqual(r[name], "ok", name)


if __name__ == "__main__":
    unittest.main()
