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
# The same block through a view of the block's size in another numeric type: half floats (small integers are exact in them),
# and 16-bit unorm (a value over 65535 comes back as the value).
PS_FLOAT = """#version 450
layout(location = 0) out vec4 o;
void main() {
    uvec2 p = uvec2(gl_FragCoord.xy);
    o = vec4(float(256u + p.x), float(512u + p.y), float(1024u + (p.x ^ p.y)), float(1536u + ((p.x + p.y) & 0xFFu)));
}
"""
PS_UNORM = """#version 450
layout(location = 0) out vec4 o;
void main() {
    uvec2 p = uvec2(gl_FragCoord.xy);
    o = vec4(float(0x100u | p.x), float(0x200u | p.y), float(0x300u | (p.x ^ p.y)), float(0xA500u | ((p.x + p.y) & 0xFFu))) / 65535.0;
}
"""
# A block BC4 decodes: red0 = red1 = v and every index 0 (the block is v everywhere), v from the pixel's position. The loop
# makes the pass slow (some milliseconds: a draw that reads its write without waiting for it would see zeros) and changes
# nothing: its result is never 0xDEADBEEF, which the guard tests, and the compiler cannot know that.
PS_BC4 = """#version 450
layout(location = 0) out uvec4 o;
void main() {
    uvec2 p = uvec2(gl_FragCoord.xy);
    uint v = (p.x + 16u * p.y) & 0xFFu;
    uint acc = v;
    for (uint i = 0u; i < 30000000u; i++) acc = (acc * 1664525u + 1013904223u) ^ i;
    if (acc == 0xDEADBEEFu) v ^= 1u;
    o = uvec4(v | (v << 8), 0u, 0u, 0u);
}
"""
# A later draw that samples the volume as BC4 (texelFetch at the first texel of each block, through the bindless tables as a
# stage reads a texture: the handle in push word 1) and writes the decoded red of three slices to a float target.
PS_SAMPLE = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 0, binding = 0) uniform texture3D tex[];
layout(set = 0, binding = 1) uniform sampler smp[];
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
layout(location = 0) out vec4 o;
void main() {
    uint h = pc.words[1].x;
    uint t = h & 0xfffffu, s = h >> 20;
    ivec2 p = ivec2(gl_FragCoord.xy) * 4;
#define V sampler3D(tex[nonuniformEXT(t)], smp[nonuniformEXT(s)])
    o = vec4(texelFetch(V, ivec3(p, SLICE_A), 0).r, texelFetch(V, ivec3(p, SLICE_B), 0).r, texelFetch(V, ivec3(p, SLICE_C), 0).r, 1.0);
}
"""
SAMPLED = (3, 9, 19)          # the slices PS_SAMPLE reads: the two the passes write, and one nothing wrote


def block_of(x: int, y: int) -> bytes:
    """The block PS writes for the pixel (x, y), as the bytes of the RGBA16 UINT texel (PS_UNORM's blocks are the same)."""
    return struct.pack("<4H", 0x100 | x, 0x200 | y, 0x300 | (x ^ y), 0xA500 | ((x + y) & 0xFF))


def block_float(x: int, y: int) -> bytes:
    """The block PS_FLOAT writes: four half floats."""
    return struct.pack("<4e", 256 + x, 512 + y, 1024 + (x ^ y), 1536 + ((x + y) & 0xFF))


def bc4_value(x: int, y: int) -> int:
    return (x + 16 * y) & 0xFF


def block_bc4(x: int, y: int) -> bytes:
    """The block PS_BC4 writes: a BC4 block of one value, red0 = red1 = v and every index 0."""
    v = bc4_value(x, y)
    return struct.pack("<4H", v | (v << 8), 0, 0, 0)


def expected_volume(slices: int | tuple[int, ...] | None, block=block_of) -> bytes:
    """The volume after the passes: every block zero, except the 8x8 blocks of the slices the views name."""
    out = bytearray(SLICE_BYTES * SLICES)
    for slice_index in (() if slices is None else (slices,) if isinstance(slices, int) else slices):
        for y in range(SIDE):
            for x in range(SIDE):
                at = slice_index * SLICE_BYTES + (y * SIDE + x) * BLOCK
                out[at:at + BLOCK] = block(x, y)
    return bytes(out)


def differing_slices(got: bytes, want: bytes) -> list[int]:
    return [s for s in range(SLICES) if got[s * SLICE_BYTES:(s + 1) * SLICE_BYTES] != want[s * SLICE_BYTES:(s + 1) * SLICE_BYTES]]


def stage_files(work: Path, name: str, stage: str, code: bytes, tag: int) -> tuple[str, Path, Path]:
    """A graphics stage's control section and code area, and the key ProgramSetShaders computes from them (tag: what tells
    two stages of one kind apart in the key, in sixteen bytes after the code, a second instruction no program reaches)."""
    code = code + struct.pack("<QQ", tag, 0)
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


# What PS_SAMPLE reads a texture by: BindTexture slot 0 (c[2]), its handle in push constant word 1 (byte 16), the combined
# 3D texture of the replay's bindless tables.
SAMPLED_TEXTURE = {"set": 0, "binding": 0, "via": "push_slot", "push_slot": 0, "push_offset": 16, "type": "3d", "shadow": False,
                   "kind": "combined", "array_length": 1, "c2_slot": 0}


def synthetic_stages(work: Path, glslang: str) -> tuple[dict[str, tuple[str, Path, Path]], Path]:
    """The vertex stage and the pixel stages: a pack of the GLSL modules under the keys of their (invented) machine code."""
    work.mkdir(parents=True, exist_ok=True)
    vs_header = bytearray(0x80)
    struct.pack_into("<I", vs_header, 0, 1)                        # SphType 1, vertex
    ps_header = bytearray(0x80)
    struct.pack_into("<I", ps_header, 0, 2 | (5 << 10))            # SphType 2 (pixel), ShaderType 5 (fragment)
    struct.pack_into("<I", ps_header, 18 * 4, 0xF)                 # omap: target 0, four components
    sample = PS_SAMPLE.replace("SLICE_A", str(SAMPLED[0])).replace("SLICE_B", str(SAMPLED[1])).replace("SLICE_C", str(SAMPLED[2]))
    stages = (("vs", "vs", VS, "uint32"), ("ps", "ps", PS, "uint32"), ("ps_float", "ps", PS_FLOAT, "float32"),
              ("ps_unorm", "ps", PS_UNORM, "float32"), ("ps_bc4", "ps", PS_BC4, "uint32"), ("ps_sample", "ps", sample, "float32"))
    records, entries = {}, []
    for tag, (name, kind, source, out_type) in enumerate(stages):
        records[name] = stage_files(work, name, kind, bytes(vs_header if kind == "vs" else ps_header) + mp.EXIT, tag)
        src = work / f"{name}{'.vert' if kind == 'vs' else '.frag'}"
        src.write_text(source, encoding="utf-8")
        spv = work / f"{name}.spv"
        done = subprocess.run([glslang, *GLSLANG_FLAGS, "-o", str(spv), str(src)], capture_output=True, text=True)
        assert done.returncode == 0, f"glslang on {name}:\n{done.stdout}{done.stderr}"
        module = spv.read_bytes()
        key = records[name][0]
        kw = {} if kind == "vs" else {"outputs": [tp.io(0, 4, 0, out_type)]}
        res = json.loads(tp.describe(key, kind, module, **kw))   # and what the replay binds a stage by: nothing, or one texture
        res.update({"support_buffer": {"fields": []}, "cbufs": [], "sbufs": [], "textures": [], "images": []})
        if name == "ps_sample":
            res["textures"] = [SAMPLED_TEXTURE]
        if kind == "vs":
            res["vertex_inputs"] = []
        entries.append(cp.Entry(key, module, json.dumps(res).encode("utf-8")))
    return records, tp.write_pack(work / "stages.bin", entries)


def write_draw(folder: Path, records: dict[str, tuple[str, Path, Path]], target: str, ps: str = "ps", more: tuple[str, ...] = (),
               sample: str | None = None, same: bool = False) -> Path:
    """A draw file of the pass (stage ps, the lines of target and more), and when sample is a stage name, sample.txt: the draw
    that reads the volume with it."""
    folder.mkdir(parents=True, exist_ok=True)

    def program_lines(pixel: str) -> list[str]:
        lines = []
        for stage, name in (("vs", "vs"), ("ps", pixel)):
            key, ctl, code = records[name]
            (folder / ctl.name).write_bytes(ctl.read_bytes())
            (folder / code.name).write_bytes(code.read_bytes())
            lines.append(f"program {stage} {key} {ctl.name} {code.name}")
        return lines

    (folder / "vertices.bin").write_bytes(bytes(16))   # no stage reads it: the draw file wants a bound stream
    body = ["stream 0 16 0", "vertex_buffer 0 vertices.bin"]
    lines = [f"size {SIDE} {SIDE}", "clear 0 0 0 0", *program_lines(ps), *body, target, *more]
    if sample:
        (folder / "sample.txt").write_text("\n".join([f"size {SIDE} {SIDE}", "clear 0 0 0 0", *program_lines(sample), *body,
                                                      "texture ps 0 1 3d 32 32 20 nearest clamp @volume",
                                                      f"draw arrays {NVN_DRAW_TRIANGLES} 0 3"]) + "\n", encoding="utf-8")
        lines.append("bcsample sample.txt" + (" same" if same else ""))
    lines.append(f"draw arrays {NVN_DRAW_TRIANGLES} 0 3")
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
        cls.records, cls.pack = synthetic_stages(cls.work / "programs", glslang)

    def run_case(self, fixture: str, slice_index: int, *, validation: str = "sync", ps: str = "ps", fmt: str | None = None,
                 second: tuple[int, bool] | None = None, sample: str | None = None,
                 same: bool = False) -> tuple[subprocess.CompletedProcess, Path]:
        """The draw harness on a pass into slice_index of a volume of the kind fixture (arena or placed), in the view format fmt:
        optionally a second pass (its slice, whether the CPU stores into the volume before it) and a later draw sampling it (in
        the submission of the last pass if same)."""
        name = f"{fixture}-{slice_index}-{validation}-{ps}-{fmt}-{second}-{sample}-{same}".replace(" ", "")
        folder = self.work / name
        more = () if second is None else (f"bcsecond {second[0]}" + (" store" if second[1] else ""),)
        draw = write_draw(folder, self.records, f"bctarget {slice_index} {fixture}" + (f" {fmt}" if fmt else ""), ps, more, sample, same)
        done = subprocess.run([str(self.exe), "draw", str(draw), str(folder / "out.bin"), "--pack", str(self.pack), "--bc-out",
                               str(folder / "blocks.bin")],
                              capture_output=True, text=True, timeout=300, env=case_env({"OOT2_VK_VALIDATION": validation}))
        return done, folder

    def play(self, fixture: str, slice_index: int, passes: int = 1, **kw) -> tuple[bytes, str, Path]:
        done, folder = self.run_case(fixture, slice_index, **kw)
        self.assertEqual(done.returncode, 0, f"{done.stdout}\n{done.stderr}")
        self.assertIn(f"bc image {fixture}", done.stdout)   # the volume stood where the case says
        self.assertIn("validation errors 0 warnings 0", done.stdout)
        self.assertRegex(done.stdout, rf"passes {passes} ")   # the passes, one for each draw
        return (folder / "blocks.bin").read_bytes(), done.stdout, folder

    def check(self, fixture: str, slice_index: int, validation: str = "sync", **kw) -> None:
        got, _, _ = self.play(fixture, slice_index, validation=validation, **kw)
        want = expected_volume(slice_index)
        self.assertEqual(len(got), len(want))
        self.assertEqual(differing_slices(got, want), [], f"{fixture} slice {slice_index}: slices differing from the expectation")
        self.assertEqual(got, want)

    def test_the_expectation_tells_slices_and_blocks_apart(self) -> None:
        self.assertNotEqual(expected_volume(5), expected_volume(6))
        self.assertNotEqual(expected_volume(5), expected_volume(None))
        self.assertNotEqual(block_of(3, 4), block_of(4, 3))
        self.assertEqual(differing_slices(expected_volume(5), expected_volume(None)), [5])
        self.assertEqual(differing_slices(expected_volume((5, 9)), expected_volume(9)), [5])
        self.assertEqual(len({block_bc4(x, y) for x in range(SIDE) for y in range(SIDE)}), SIDE * SIDE)   # one value a block

    def test_an_arena_volume_gets_the_blocks_of_the_slice_the_view_names(self) -> None:
        for slice_index in (0, 5, 19):
            self.check("arena", slice_index)

    def test_a_placed_volume_gets_the_blocks_of_the_slice_the_view_names(self) -> None:
        for slice_index in (0, 5, 19):
            self.check("placed", slice_index)

    def test_the_pass_runs_under_the_other_validation_mode_too(self) -> None:
        self.check("arena", 17, validation="on")
        self.check("placed", 17, validation="on")

    def test_two_passes_into_two_slices_keep_both_slices(self) -> None:
        """The second pass is a second submission, which makes the arena image current again: the CPU's store to a page of the
        volume between the passes must not upload the guest's bytes (zeros) over the blocks the first pass wrote."""
        for fixture, store in (("arena", True), ("arena", False), ("placed", False)):
            got, out, _ = self.play(fixture, 3, passes=2, second=(9, store))
            if store:
                self.assertIn("cpu store into the volume", out)
            want = expected_volume((3, 9))
            self.assertEqual(differing_slices(got, want), [], f"{fixture} (store {store}): slices differing from the expectation")

    def test_a_later_draw_samples_the_blocks_the_passes_wrote(self) -> None:
        """What a game does with the volume after the passes: a draw samples it as BC4 (the pass's write is visible to it, through
        the BC4 view the bindless tables make over the image the passes wrote), and gets the decoded value of the block in the
        slices the passes named and zero in a slice nothing wrote. In a submission of its own, and in the pass's own (where the
        ordering of the pass's write before the sample is the replay's barrier alone)."""
        for fixture, store, same in (("arena", True, False), ("placed", False, False), ("arena", False, True), ("placed", False, True)):
            got, _, folder = self.play(fixture, SAMPLED[0], passes=3, ps="ps_bc4", second=(SAMPLED[1], store), sample="ps_sample",
                                       same=same)
            self.assertEqual(differing_slices(got, expected_volume(SAMPLED[:2], block_bc4)), [], f"{fixture} same {same}")
            pixels = struct.unpack("<256f", (folder / "out.bin").read_bytes())
            for y in range(SIDE):
                for x in range(SIDE):
                    r, g, b, a = pixels[4 * (y * SIDE + x):4 * (y * SIDE + x) + 4]
                    v = bc4_value(x, y) / 255.0
                    self.assertAlmostEqual(r, v, delta=1e-5, msg=f"{fixture} same {same} pixel {x},{y} slice {SAMPLED[0]}")
                    self.assertAlmostEqual(g, v, delta=1e-5, msg=f"{fixture} same {same} pixel {x},{y} slice {SAMPLED[1]}")
                    self.assertEqual((b, a), (0.0, 1.0), f"{fixture} pixel {x},{y}: a slice nothing wrote, and the alpha")

    def test_a_view_of_the_blocks_size_in_another_numeric_type_counts_blocks_too(self) -> None:
        """The extent in blocks follows the view's size and the image's blocks, not a class of the view's format."""
        for fixture in ("arena", "placed"):
            for ps, fmt, block in (("ps_float", "rgba16f", block_float), ("ps_unorm", "rgba16", block_of)):
                got, _, _ = self.play(fixture, 7, ps=ps, fmt=fmt)
                self.assertEqual(differing_slices(got, expected_volume(7, block)), [], f"{fixture} {fmt}")
                self.assertEqual(got, expected_volume(7, block))

    def test_a_pass_in_the_compressed_format_itself_stops_by_name_with_the_cause(self) -> None:
        for fixture in ("arena", "placed"):
            done, _ = self.run_case(fixture, 5, fmt="bc4")
            text = "\n".join(fatal_lines(done.stderr))
            self.assertEqual(done.returncode, EXIT_FATAL, f"{done.stdout}\n{done.stderr}")
            self.assertIn("no image stands on the render target's storage (NVN format 0x", text)
            self.assertIn("level 0, layers 5..6", text)
            self.assertIn("the device renders to no image of the view's format", text)

    def test_a_view_of_another_size_than_the_block_stops_by_name(self) -> None:
        done, _ = self.run_case("arena", 5, ps="ps_unorm", fmt="rgba8")
        text = "\n".join(fatal_lines(done.stderr))
        self.assertEqual(done.returncode, EXIT_FATAL, f"{done.stdout}\n{done.stderr}")
        self.assertIn("NVN format 0x25 of 4 bytes a texel, over block-compressed storage of 8 bytes a block", text)


if __name__ == "__main__":
    unittest.main()
