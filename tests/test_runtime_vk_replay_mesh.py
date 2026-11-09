# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Synthetic mesh draws through NVN and the Vulkan replay, with no game data.

Run: python -m unittest discover -s tests -p test_runtime_vk_replay_mesh.py -v

The device test skips when the translator, Vulkan SDK, clang, or a usable Vulkan device is unavailable.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_runtime_gpu import EXIT_FATAL, NO_VULKAN, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
import cache_pack as cp  # noqa: E402
import mesh_draw as md  # noqa: E402
import mesh_programs as mp  # noqa: E402
import pool  # noqa: E402
import split  # noqa: E402

SOURCE = ROOT / "tests" / "runtime" / "vk_replay_draw.c"
LEVEL = "-O1"
VALIDATIONS = ("on", "sync")


def translator_path() -> Path | None:
    override = os.environ.get("OOT2_XLATE")
    if override:
        found = shutil.which(override)
        path = Path(found or override).resolve()
        result = path if path.is_file() else None
    else:
        name = "oot2-xlate.exe" if os.name == "nt" else "oot2-xlate"
        hits = sorted((ROOT / "build" / "xlate" / "publish").rglob(name))
        result = hits[0] if hits else None
    if result is None and os.environ.get("OOT2_REQUIRE_XLATE"):
        raise RuntimeError("OOT2_REQUIRE_XLATE is set but no oot2-xlate executable was found")
    return result


def stage_artifact(work: Path, name: str, stage: str, code: bytes, max_vertices: int = 3) -> tuple[str, dict]:
    ctl = bytearray(0xD0)
    struct.pack_into("<II", ctl, 0, 0x98761234, 0xC9)
    struct.pack_into("<I", ctl, 0x38, len(code))
    struct.pack_into("<II", ctl, 0x40, len(code), len(code))
    if stage == "ts":
        ctl[0x56] = 6
        struct.pack_into("<HH", ctl, 0x58, 32, 32)
    elif stage == "ms":
        ctl[0x56] = 7
        struct.pack_into("<HH", ctl, 0x58, 32, 32)
        ctl[0x5E] = 1
        struct.pack_into("<HH", ctl, 0x60, max_vertices, 1)
    elif stage != "ps":
        raise AssertionError(stage)
    ctl_path, code_path = work / f"{name}.ctl", work / f"{name}.bin"
    ctl_path.write_bytes(ctl)
    code_path.write_bytes(code)
    spec = split.build_job(stage, ctl, code)
    key = f"{split.ctl_key(ctl, code)}.{spec.variant_hash()}"
    job = {"key": key, "stage": stage, "arch": "sm86", "code": str(code_path),
           "code_offset": spec.start, "code_length": spec.length, "c1": spec.c1}
    for field in ("nvn2_cs", "local", "shared", "stage_memory"):
        value = getattr(spec, field)
        if value is not None:
            job[field] = value
    if spec.mesh is not None:
        job["mesh"] = spec.mesh
    if spec.outputs is not None:
        job["outputs"] = spec.outputs
    return key, job


def synthetic_programs(work: Path, max_vertices: int = 3) -> tuple[dict[str, tuple[str, Path, Path]], Path]:
    work.mkdir(parents=True, exist_ok=True)
    xlate = translator_path()
    if xlate is None:
        raise unittest.SkipTest("no oot2-xlate executable under build/xlate/publish (or OOT2_XLATE)")

    task = mp.task_program()
    task[3] = mp.mov_imm(5, 3)
    mesh = mp.mesh_program()
    mesh[-8:] = [mp.mov_imm(13, 1), mp.isbewr(0x88, 3, 13), mp.mov_imm(14, 1), mp.mov_imm(15, 2),
                 mp.isbewr(0x80, 4, 255), mp.isbewr(0x80, 5, 14), mp.isbewr(0x80, 6, 15), mp.EXIT]
    mesh_header = bytearray(0x80)
    mesh_header[0x35] = 0xF0
    mesh_header[0x36] = 0x0F
    programs = {
        "ts": bytes(0x80) + mp.code_bytes(task),
        "ms": bytes(mesh_header) + mp.code_bytes(mesh),
        "ps": mp.pixel_program(),
    }

    jobs = []
    records: dict[str, tuple[str, Path, Path]] = {}
    for name, code in programs.items():
        stage = name
        key, job = stage_artifact(work, name, stage, code, max_vertices)
        jobs.append((job, None))
        records[name] = (key, work / f"{name}.ctl", work / f"{name}.bin")
    translated: dict[str, tuple[bytes, bytes]] = {}
    results, _ = pool.run_pool(
        [str(xlate)], jobs, 1, work / "logs", timeout=180,
        on_result=lambda result: translated.__setitem__(result.key, (result.module, result.description)),
    )
    by_key = {result.key: result for result in results}
    entries = []
    for name, (key, _, _) in records.items():
        result = by_key.get(key)
        if result is None or result.status != "ok":
            raise RuntimeError(f"{name}: translator result {getattr(result, 'status', None)}: "
                               f"{getattr(result, 'message', 'missing')}")
        module, description = translated[key]
        entries.append(cp.Entry(key, module, description))
    pack_path = work / "shader-base.bin"
    cp.write_pack(pack_path, entries, translator=hashlib.sha256(xlate.read_bytes()).hexdigest())
    return records, pack_path


def write_draw(folder: Path, records: dict[str, tuple[str, Path, Path]], form: str, first: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"size {mp.IMAGE} {mp.IMAGE}", "clear 0 0 0 0"]
    for stage in ("ts", "ms", "ps"):
        key, ctl, code = records[stage]
        shutil.copyfile(ctl, folder / ctl.name)
        shutil.copyfile(code, folder / code.name)
        lines.append(f"program {stage} {key} {ctl.name} {code.name}")
    if form == "direct":
        lines.append(f"draw mesh direct {first} 4")
    else:
        lines.append(f"draw mesh indirect 4 {first}")
    path = folder / "draw.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_replay(exe: Path, draw: Path, out: Path, pack_path: Path,
               validation: str) -> subprocess.CompletedProcess:
    env = case_env({"OOT2_VK_VALIDATION": validation})
    return subprocess.run([str(exe), "draw", str(draw), str(out), "--pack", str(pack_path)],
                          capture_output=True, text=True, timeout=180, env=env)


def replay_expected_image(scenario: md.Scenario):
    kind, rgba = md.expected_image(md.workgroups(scenario))
    rows = [slice(y * mp.IMAGE, (y + 1) * mp.IMAGE) for y in range(mp.IMAGE)]
    return ([pixel for row in reversed(rows) for pixel in kind[row]],
            [pixel for row in reversed(rows) for pixel in rgba[row]])


def require_usable_vulkan(done: subprocess.CompletedProcess) -> None:
    if done.returncode == 0:
        return
    text = "\n".join(fatal_lines(done.stderr))
    if done.returncode == EXIT_FATAL and any(marker in text for marker in NO_VULKAN):
        raise unittest.SkipTest(f"no usable Vulkan device here ({text})")
    raise RuntimeError(f"Vulkan probe failed:\n{done.stdout}{done.stderr}")


class VulkanProbeTest(unittest.TestCase):
    def test_device_missing_required_mesh_capabilities_is_skipped(self) -> None:
        done = subprocess.CompletedProcess(
            ["vk_replay_draw", "probe"], EXIT_FATAL, "",
            "FATAL: gfx: vulkan: no Vulkan device has every item the renderer requires; lacks mesh shader\n",
        )
        with self.assertRaisesRegex(unittest.SkipTest, "no usable Vulkan device"):
            require_usable_vulkan(done)

    def test_unrelated_vulkan_probe_failure_remains_an_error(self) -> None:
        done = subprocess.CompletedProcess(["vk_replay_draw", "probe"], 1, "", "unexpected probe error\n")
        with self.assertRaisesRegex(RuntimeError, "Vulkan probe failed"):
            require_usable_vulkan(done)


class TranslatorRequirementTest(unittest.TestCase):
    def test_missing_translator_fails_when_the_environment_requires_it(self) -> None:
        with mock.patch.dict(os.environ, {"OOT2_REQUIRE_XLATE": "1"}, clear=False):
            os.environ.pop("OOT2_XLATE", None)
            with mock.patch.object(Path, "rglob", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "OOT2_REQUIRE_XLATE is set"):
                    translator_path()

    def test_missing_translator_remains_optional_without_the_requirement(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OOT2_REQUIRE_XLATE", None)
            os.environ.pop("OOT2_XLATE", None)
            with mock.patch.object(Path, "rglob", return_value=[]):
                self.assertIsNone(translator_path())


class MeshReplayTest(unittest.TestCase):
    MAX_VERTICES = 3   # the mesh stage's declared vertex count: the length of each per-vertex output array

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="oot2-vkreplay-mesh-", ignore_cleanup_errors=True)
        atexit.register(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        if translator_path() is None:
            raise unittest.SkipTest("no oot2-xlate executable under build/xlate/publish (or OOT2_XLATE)")
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.exe = build_program(cc, include, glslang, SOURCE, cls.work / "build", levels=(LEVEL,))[LEVEL]
        probe = subprocess.run([str(cls.exe), "probe"], capture_output=True, text=True, timeout=60,
                               env=case_env({"OOT2_VK_VALIDATION": "off"}))
        require_usable_vulkan(probe)
        cls.records, cls.pack_path = synthetic_programs(cls.work / "programs", cls.MAX_VERTICES)

    def test_direct_and_indirect_mesh_draws_replay_with_nonzero_first_task(self) -> None:
        scenario = md.Scenario("replay", "direct", 3, 4)
        expected = replay_expected_image(scenario)
        for form in ("direct", "indirect"):
            draw = write_draw(self.work / form, self.records, form)
            for validation in VALIDATIONS:
                out = self.work / form / f"{validation}.f32"
                done = run_replay(self.exe, draw, out, self.pack_path, validation)
                self.assertEqual(done.returncode, 0, f"{form}/{validation}:\n{done.stdout}\n{done.stderr}")
                self.assertIn("ok vk_replay_draw", done.stdout)
                self.assertRegex(done.stdout, r"replay frames \d+ records \d+ draws 1 ")
                self.assertRegex(done.stdout, r"validation errors 0 warnings 0")
                raw = out.read_bytes()
                got = list(struct.iter_unpack("<4f", raw))
                self.assertEqual(len(got), mp.IMAGE * mp.IMAGE)
                compared = md.compare(got, expected)
                self.assertEqual(compared["mismatches"], 0, f"{form}/{validation}: {compared}")
                self.assertTrue(compared["ran_ok"], f"{form}/{validation}: {compared}")
                self.assertEqual(compared["pixels"][md.EDGE], 0, f"{form}/{validation}: {compared}")

    def test_wrong_first_task_input_does_not_match_the_nonzero_reference(self) -> None:
        scenario = md.Scenario("replay", "direct", 3, 4)
        expected = replay_expected_image(scenario)
        for form in ("direct", "indirect"):
            draw = write_draw(self.work / f"{form}-first-zero", self.records, form, first=0)
            out = draw.parent / "first-zero.f32"
            done = run_replay(self.exe, draw, out, self.pack_path, "off")
            self.assertEqual(done.returncode, 0, f"{form}/first-zero:\n{done.stdout}\n{done.stderr}")
            self.assertIn("ok vk_replay_draw", done.stdout)
            raw = out.read_bytes()
            got = list(struct.iter_unpack("<4f", raw))
            self.assertEqual(len(got), mp.IMAGE * mp.IMAGE)
            compared = md.compare(got, expected)
            self.assertGreater(compared["mismatches"], 0, f"{form}/first-zero unexpectedly matched: {compared}")
            self.assertNotEqual(compared["ran"], compared["expected_ran"], f"{form}/first-zero: {compared}")


class WideMeshReplayTest(MeshReplayTest):
    """The same draws with 96 declared vertices (3 written): the colour is a float[96] per component at location 0, which the
    builder links as one location (shd_mesh_link); counted as 96 locations it would stop past the 64 it reads."""

    MAX_VERTICES = 96


if __name__ == "__main__":
    unittest.main()
