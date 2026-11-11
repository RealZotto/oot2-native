# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""A pass into the blocks of a block-compressed volume, through NVN and the Vulkan replay, with no game data.

A block compressor draws one pixel for each block of a compressed volume's slice: the colour target is a view of the volume
in the integer format of the block's size (a BC4 block is 8 bytes, so RGBA16 UINT), a texel of the view is a block, and the
pass is the size of the slice in blocks (8x8 for a 32x32 level). The harness (tests/runtime/vk_replay_draw.c, directive
bctarget) records that pass as a game does (SetRenderTargets with the view, a viewport and scissor of 8x8, one draw of a
synthetic pair of GLSL stages that write a block from the pixel's position) and plays it through the replay on the device under
the validation layers. The blocks of the volume, read back as the image holds them, are the blocks the pixel stage wrote, in
the slice the view names, and no other slice changed. The volume stands in the pool of images (a placed image) and in the
CPU-visible pool (an arena image, which is where the game's volume is); both are asked for.

Run: python -m unittest discover -s tests -p test_runtime_vk_replay_bc.py -v

Skips without clang, the Vulkan SDK or a usable device (an error when OOT2_REQUIRE_GPU or OOT2_REQUIRE_VULKAN is set).
"""

from __future__ import annotations

import atexit
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, NO_VULKAN, ROOT, build_program, case_env, clang_or_skip, fatal_lines, vulkan_or_skip

sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
import cache_pack as cp  # noqa: E402
import mesh_programs as mp  # noqa: E402
import split  # noqa: E402
import test_runtime_vk_pipeline as tp  # noqa: E402  (its description of a stage's interface)
from build_game import GLSLANG_FLAGS  # noqa: E402

SOURCE = ROOT / "tests" / "runtime" / "vk_replay_draw.c"
LEVEL = "-O1"
SIDE = 8                      # blocks along a slice of a 32x32 level
BLOCK = 8                     # bytes of a BC4 block
SLICE_BYTES = SIDE * SIDE * BLOCK
SLICES = 20
NVN_DRAW_TRIANGLES = 4

VS = """#version 450
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
# One 8-byte block from the pixel's position: the four 16-bit components RGBA16 UINT writes.
PS = """#version 450
layout(location = 0) out uvec4 o;
void main() {
    uvec2 p = uvec2(gl_FragCoord.xy);
    o = uvec4(0x100u | p.x, 0x200u | p.y, 0x300u | (p.x ^ p.y), 0xA500u | ((p.x + p.y) & 0xFFu));
}
"""


def block_of(x: int, y: int) -> bytes:
    """The block the pixel stage writes for the pixel (x, y), as the bytes of the RGBA16 UINT texel."""
    return struct.pack("<4H", 0x100 | x, 0x200 | y, 0x300 | (x ^ y), 0xA500 | ((x + y) & 0xFF))


def expected_volume(slice_index: int | None) -> bytes:
    """The volume after the pass: every block zero, except the 8x8 blocks of the slice the view names."""
    out = bytearray(SLICE_BYTES * SLICES)
    if slice_index is not None:
        for y in range(SIDE):
            for x in range(SIDE):
                at = slice_index * SLICE_BYTES + (y * SIDE + x) * BLOCK
                out[at:at + BLOCK] = block_of(x, y)
    return bytes(out)


def differing_slices(got: bytes, want: bytes) -> list[int]:
    return [s for s in range(SLICES) if got[s * SLICE_BYTES:(s + 1) * SLICE_BYTES] != want[s * SLICE_BYTES:(s + 1) * SLICE_BYTES]]


def stage_files(work: Path, name: str, stage: str, code: bytes) -> tuple[str, Path, Path]:
    """A graphics stage's control section and code area, and the key ProgramSetShaders computes from them."""
    ctl = bytearray(0xD0)
    struct.pack_into("<II", ctl, 0, 0x98761234, 0xC9)
    struct.pack_into("<I", ctl, 0x38, len(code))
    struct.pack_into("<II", ctl, 0x40, len(code), len(code))
    spec = split.build_job(stage, bytes(ctl), code)
    key = f"{split.ctl_key(ctl, code)}.{spec.variant_hash()}"
    ctl_path, code_path = work / f"{name}.ctl", work / f"{name}.bin"
    ctl_path.write_bytes(bytes(ctl))
    code_path.write_bytes(code)
    return key, ctl_path, code_path


def synthetic_pair(work: Path, glslang: str) -> tuple[dict[str, tuple[str, Path, Path]], Path]:
    """The vertex and pixel stages: a pack of the two GLSL modules under the keys of their (invented) machine code."""
    work.mkdir(parents=True, exist_ok=True)
    vs_header = bytearray(0x80)
    struct.pack_into("<I", vs_header, 0, 1)                        # SphType 1, vertex
    ps_header = bytearray(0x80)
    struct.pack_into("<I", ps_header, 0, 2 | (5 << 10))            # SphType 2 (pixel), ShaderType 5 (fragment)
    struct.pack_into("<I", ps_header, 18 * 4, 0xF)                 # omap: target 0, four components
    records = {"vs": stage_files(work, "vs", "vs", bytes(vs_header) + mp.EXIT),
               "ps": stage_files(work, "ps", "ps", bytes(ps_header) + mp.EXIT)}
    entries = []
    for name, source, kw in (("vs", VS, {}), ("ps", PS, {"outputs": [tp.io(0, 4, 0, "uint32")]})):
        suffix = {"vs": ".vert", "ps": ".frag"}[name]
        src = work / f"{name}{suffix}"
        src.write_text(source, encoding="utf-8")
        spv = work / f"{name}.spv"
        done = subprocess.run([glslang, *GLSLANG_FLAGS, "-o", str(spv), str(src)], capture_output=True, text=True)
        assert done.returncode == 0, f"glslang on {name}:\n{done.stdout}{done.stderr}"
        module = spv.read_bytes()
        key = records[name][0]
        res = json.loads(tp.describe(key, name, module, **kw))   # and what the replay binds a stage by: nothing
        res.update({"support_buffer": {"fields": []}, "cbufs": [], "sbufs": [], "textures": [], "images": []})
        if name == "vs":
            res["vertex_inputs"] = []
        entries.append(cp.Entry(key, module, json.dumps(res).encode("utf-8")))
    pack = tp.write_pack(work / "pair.bin", entries)
    return records, pack


def write_draw(folder: Path, records: dict[str, tuple[str, Path, Path]], target: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"size {SIDE} {SIDE}", "clear 0 0 0 0"]
    for stage in ("vs", "ps"):
        key, ctl, code = records[stage]
        (folder / ctl.name).write_bytes(ctl.read_bytes())
        (folder / code.name).write_bytes(code.read_bytes())
        lines.append(f"program {stage} {key} {ctl.name} {code.name}")
    (folder / "vertices.bin").write_bytes(bytes(16))   # no stage reads it: the draw file wants a bound stream
    lines += ["stream 0 16 0", "vertex_buffer 0 vertices.bin", target, f"draw arrays {NVN_DRAW_TRIANGLES} 0 3"]
    path = folder / "draw.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class BcPassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="oot2-vkreplay-bc-", ignore_cleanup_errors=True)
        atexit.register(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.exe = build_program(cc, include, glslang, SOURCE, cls.work / "build", levels=(LEVEL,))[LEVEL]
        probe = subprocess.run([str(cls.exe), "probe"], capture_output=True, text=True, timeout=60,
                               env=case_env({"OOT2_VK_VALIDATION": "off"}))
        if probe.returncode != 0:
            text = "\n".join(fatal_lines(probe.stderr))
            if probe.returncode == EXIT_FATAL and any(marker in text for marker in NO_VULKAN):
                raise unittest.SkipTest(f"no usable Vulkan device here ({text})")
            raise RuntimeError(f"Vulkan probe failed:\n{probe.stdout}{probe.stderr}")
        cls.records, cls.pack = synthetic_pair(cls.work / "programs", glslang)

    def play(self, fixture: str, slice_index: int, validation: str = "sync") -> tuple[bytes, str]:
        folder = self.work / f"{fixture}-{slice_index}-{validation}"
        draw = write_draw(folder, self.records, f"bctarget {slice_index} {fixture}")
        out = folder / "unused.bin"
        blocks = folder / "blocks.bin"
        done = subprocess.run([str(self.exe), "draw", str(draw), str(out), "--pack", str(self.pack), "--bc-out", str(blocks)],
                              capture_output=True, text=True, timeout=300, env=case_env({"OOT2_VK_VALIDATION": validation}))
        self.assertEqual(done.returncode, 0, f"{done.stdout}\n{done.stderr}")
        self.assertIn(f"bc image {fixture}", done.stdout)   # the volume stood where the case says
        self.assertIn("validation errors 0 warnings 0", done.stdout)
        self.assertIn("passes 1 ", done.stdout)             # the one pass, begun by the draw
        return blocks.read_bytes(), done.stdout

    def check(self, fixture: str, slice_index: int, validation: str = "sync") -> None:
        got, _ = self.play(fixture, slice_index, validation)
        want = expected_volume(slice_index)
        self.assertEqual(len(got), len(want))
        self.assertEqual(differing_slices(got, want), [], f"{fixture} slice {slice_index}: slices differing from the expectation")
        self.assertEqual(got, want)

    def test_the_expectation_tells_slices_and_blocks_apart(self) -> None:
        self.assertNotEqual(expected_volume(5), expected_volume(6))
        self.assertNotEqual(expected_volume(5), expected_volume(None))
        self.assertNotEqual(block_of(3, 4), block_of(4, 3))
        self.assertEqual(differing_slices(expected_volume(5), expected_volume(None)), [5])

    def test_an_arena_volume_gets_the_blocks_of_the_slice_the_view_names(self) -> None:
        for slice_index in (0, 5, 19):
            self.check("arena", slice_index)

    def test_a_placed_volume_gets_the_blocks_of_the_slice_the_view_names(self) -> None:
        for slice_index in (0, 5, 19):
            self.check("placed", slice_index)

    def test_the_pass_runs_under_the_other_validation_mode_too(self) -> None:
        self.check("arena", 17, validation="on")
        self.check("placed", 17, validation="on")


if __name__ == "__main__":
    unittest.main()
