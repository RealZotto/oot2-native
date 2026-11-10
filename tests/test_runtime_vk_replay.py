# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the Vulkan back end's replay (runtime/include/oot2/vk_replay.h, runtime/gpu/vk_replay.c) through
tests/runtime/vk_replay_test.c, with no device (continuous integration runs these).

Synthetic recordings go through the replay with stand-in ops and through the real recorder, whose calls write each packet
as a line. The checks: a pass begins lazily behind a full barrier with its targets loaded and stored in the general
layout, and ends at a render target change, a barrier and the frame's end; the default window origin (lower left) flips Y
through a negative viewport height, measures scissors and clear rectangles from the bottom and mirrors the front face;
the default depth mode sets depth clip control's -1 to 1 range; vertex input comes from the attribute, stream and buffer
state, and an attribute with no buffer reads (0, 0, 0, 1) of its own scalar type from a spare binding; a state that needs a
core feature (line and point fill, the depth bias clamp, depth clamp, the depth bounds test, wide lines, several viewports,
a blend state and write mask for each target, dual-source factors) draws as its own packets and an indirect draw's base
instance reaches c[0]; each stage's driver bank c[0] holds its storage table, the
largest point size and the base instance, NVN2 compute's the group counts and its uniform and storage rows; c[2] holds the
bound handles, bank 24 the pushed root tables, the support buffer the viewport; a uniform range in host memory is copied
at submit (the copy holds the bytes of that moment, and the next submission takes a copy of its own) while a mirror binds
live; a storage buffer binds from its bank's words, rounded down to 16 bytes, and one the module writes is declared and
listed; a storage image binds the view the ops make of the texture its c[2] image handle names, in the general layout, and the
stores into one placed in its storage are declared written (an arena image holds its own stores, and a module that only
reads declares nothing); a texture's push constant slot holds its bindless slot and its sampler; draws, indexed and indirect draws, direct
and indirect dispatches and mesh draws become their packets (a mesh draw's first-task word is passed by a scratch device
address, and an indirect `{count, first}` record is converted by GPU copies); an indirect dispatch's counts are copied by
the GPU into a bank of its own; a GPU write over a copied uniform range drops the copy, so that every later read of the range is live; copies,
fills, updates and texture clears are one transfer command each behind a full barrier, with the channel's strides, their
writes into memory the CPU can see declared once; every op of the list is replayed or stops by name with its ticket; the geometry processing mode is replayed for the modes 0 to 2 (logged once per mode, no state set) and any other mode stops by name; a recording that is not whole stops by name.
Reports, events and waits are done by the completion thread in op-stream order (an event WRITE then ADD leaves their
sum; a samples-passed report writes the running sum of the occlusion queries before it, one per pass from the first
reset, ended at each samples-passed report or reset and at the pass's end; a reset sets the count to 0; a timestamp
report's counter word is 0; a report into memory the CPU cannot see is dropped; a WaitEvent is checked at its place);
conditional rendering reads a report not done yet as 0 (a timestamp) or as visible (samples passed), and memory otherwise.

Each check of the ticket's new behaviour has a counter-run, a test-only switch of the replay that breaks the mechanism
(vkr_no_flip: the origin's flip; vkr_no_snapshot: the uniform copy at submit; vkr_no_drop: the drop rule;
vkr_no_declare: the declaration of a transfer's or a storage image's write; vkr_no_features: the states of the core features;
vkr_no_pending: the reports not done at the condition; vkr_reverse_results: the op-stream order of completion).
The check must then fail.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import re
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import (EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines,  # sets the path of tools/
                              vulkan_or_skip)

import check_capture as cc  # noqa: E402  (tools/ is on the path since test_runtime_gpu was imported)

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_replay_test.c"
LEVEL = "-O2"

VS_KEY = "00000000000000a1.00000001"
PS_KEY = "00000000000000b2.00000001"
CS_KEY = "00000000000000c3.00000001"
TS_KEY = "00000000000000d1.00000001"
MS_KEY = "00000000000000d2.00000001"
REPLAYED = {
    "BEGIN", "END", "SET_RT", "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM",
    "BIND_STATE", "BIND_VATTR", "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "BIND_IMG",
    "STENCIL", "POLY_OFFSET", "DRAW", "DRAW_IDX", "DRAW_IDX_IND", "DISPATCH", "DISPATCH_IND", "BARRIER", "SET_TEX_POOL",
    "SET_SAMP_POOL", "BLEND_COLOR", "DEPTH_CLAMP", "RENDER_ENABLE", "RASTER_DISCARD", "PRIM_RESTART", "PUSH_CONST",
    "WAIT_SYNC", "SAMPLE_MASK", "DEPTH_BOUNDS", "CONSERVATIVE", "SHADING_RATE_IMAGE", "SHADING_RATE_MODE", "SCRATCH",
    "HINT", "COPY_B2B", "UPDATE_UB", "COPY_B2T", "COPY_STRIDE", "COPY_T2B", "COPY_T2T", "CLEAR_TEX", "CLEAR_BUF",
    "COPY_IMAGE_STRIDE", "SIGNAL_EVENT", "REPORT_COUNTER", "RESET_COUNTER", "RENDER_COND", "WAIT_EVENT",
    "SHADING_RATE_PALETTE",
    "DRAW_MESH", "DRAW_MESH_IND", "GEOMETRY_MODE",
}


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def packets(out: str) -> list[str]:
    return [line[len("pkt "):] for line in out.splitlines() if line.startswith("pkt ")]


def frame(out: str, n: int) -> list[str]:
    """The packets of the n-th frame (from 1): those before its "submit" line and after the previous one's."""
    frames, cur = [], []
    for p in packets(out):
        if p.startswith("submit "):
            frames.append(cur)
            cur = []
        else:
            cur.append(p)
    return frames[n - 1]


def writes(lines: list[str], binding: int) -> list[str]:
    """The descriptor writes at `binding`, in order, each as "UB buf=... off=... range=... words"."""
    return [line.split(" ", 3)[3] for line in lines if line.startswith(f" w b={binding} ")]


def word(write: str, offset: int) -> int | None:
    m = re.search(rf" {offset:#x}=(0x[0-9a-f]+)\b", write)
    return int(m.group(1), 16) if m else None


# ---- the checks, each a function of the output so that a counter-run can show it failing
def check_pass(out: str) -> None:
    f = frame(out, 1)
    assert f[0] == "PipelineBarrier2 full=1", f[:2]   # the frame after every earlier one
    assert f[1] == "PipelineBarrier2 full=1", f[:2]   # the pass after everything before it
    assert f[2] == "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0]", f[2]
    end = f.index("EndRendering")
    assert f[end + 1] == "PipelineBarrier2 full=1" and f.count("BeginRendering 0,0 64x32 layers=1 colors=1 "
                                                               "[0x7125 layout=1 load=0 store=0]") == 1, f[end:]


def check_flip(out: str) -> None:
    f = frame(out, 1)
    assert "SetViewportWithCount 1 [0 32 64 -32 0 1]" in f, [p for p in f if p.startswith("SetViewport")]
    assert "SetScissorWithCount 1 [8 20 16 8]" in f, [p for p in f if p.startswith("SetScissor")]
    assert "SetFrontFace 1" in f, [p for p in f if p.startswith("SetFrontFace")]   # CCW mirrored to CW
    assert "ClearAttachments aspect=0x1 att=0 color=1,0,0,1 rect=8,20 16x8 layers=0+1" in f


def check_depth_clip(out: str) -> None:
    assert "SetDepthClipNegativeOneToOneEXT 1" in frame(out, 1)


def check_snapshot(out: str) -> None:
    for n, base in ((1, 0xA0), (2, 0xB0)):
        w = writes(frame(out, n), 2)
        assert w and all(x.startswith("UB buf=0x5c ") and "range=0x40" in x for x in w), w
        assert all(word(x, 4 * i) == base + i for x in w for i in range(16)), (n, w)


def check_live_mirror(out: str) -> None:
    w = writes(frame(out, 1), 7)
    assert w and all(x == "UB buf=0xb2 off=0x4000 range=0x100" for x in w), w
    w = writes(frame(out, 1), 3)   # c[1], the program's constants in its code area
    assert w and all(x == "UB buf=0xb2 off=0x1100 range=0x40" for x in w), w


def check_banks(out: str) -> None:
    f = frame(out, 1)
    vs_c0, ps_c0 = writes(f, 1), writes(f, 6)
    assert len(vs_c0) == 3 and len(ps_c0) == 3, (vs_c0, ps_c0)
    for x in vs_c0 + ps_c0:
        assert x.startswith("UB buf=0x5c ") and "range=0xa00" in x and word(x, 0x610) == f32(2047.0), x
    assert word(vs_c0[0], 0x644) is None and word(vs_c0[1], 0x644) == 1, vs_c0   # the base instance of each draw
    assert word(ps_c0[0], 0x510) == 0x40000104 and word(ps_c0[0], 0x518) == 0x40, ps_c0[0]
    ps_c2 = writes(f, 8)
    assert ps_c2 and word(ps_c2[0], 0x20 + 8 * 5) == (7 | 3 << 20), ps_c2   # combined handle row 5
    assert writes(f, 4)[0].endswith(" zero") and writes(f, 5)[0].endswith(" zero")
    support = writes(f, 0)
    assert len(support) == 3, support
    s = support[0]
    assert "range=0x570" in s, s
    assert [word(s, o) for o in (0x90, 0x94, 0x98, 0xa0, 0xa4, 0xa8, 0xc0)] == \
        [f32(2 / 64), f32(2 / 32), f32(1.0), f32(64.0), f32(32.0), f32(1.0), f32(1.0)], s


def check_storage(out: str) -> None:
    w = writes(frame(out, 1), 9)
    assert len(w) == 3 and all(x == "SB buf=0xb3 off=0x100 range=0x44" for x in w), w
    # declared once a frame: the later writes of the same range lie inside the first
    assert out.count("op wrote V+0x100 0x44 frame 1\n") == 1 and "op wrote V+0x100 0x44 frame 2\n" in out


CS_IMG_KEY = "00000000000000c5.00000001"


def check_storage_image(out: str) -> None:
    """The storage-image case: ID 9 (an image placed in its storage) and ID 10 (an arena image) bound by frame 1's two
    dispatches, ID 9 again, read only, by frame 2's. Each dispatch asks the ops for its view; the stores into the placed
    image's storage are declared written (once, by the first dispatch) and nothing else is."""
    ops = [line for line in out.splitlines() if line.startswith(("op storage_image ", "op wrote "))]
    assert ops == ["op storage_image id 9 frame 1", "op wrote V+0x50000 0x4000 frame 1", "op storage_image id 10 frame 1",
                   "op storage_image id 9 frame 2"], ops
    for n, views in ((1, ["SI view=0x5d09 layout=1", "SI view=0x5d0a layout=1"]), (2, ["SI view=0x5d09 layout=1"])):
        got = writes(frame(out, n), 25)
        assert got == views, (n, got)


def check_push_slot(out: str) -> None:
    pcs = [p for p in frame(out, 1) if p.startswith("PushConstants ")]
    assert len(pcs) == 3 and all(p.endswith(f" 0 128 24={1057 | 3 << 20:#x}") for p in pcs), pcs


def check_vertex_input(out: str) -> None:
    f = frame(out, 1)
    vi = [p for p in f if p.startswith("SetVertexInputEXT")]
    assert vi and all(p == "SetVertexInputEXT [b0 stride=16 rate=0 div=1] [b1 stride=0 rate=0 div=1] "
                           "[l0 b0 fmt=109 off=0] [l1 b1 fmt=108 off=16]" for p in vi), vi
    assert "BindVertexBuffers2 0 1 0xb2+0x3000" in f, f


def check_draws(out: str) -> None:
    f = frame(out, 1)
    draws = [p for p in f if p.startswith(("Draw", "BindIndexBuffer", "ClearAttachments"))]
    assert draws == ["Draw 3 1 0 0", "BindIndexBuffer 0xb2 20480 0", "DrawIndexed 6 2 0 2 1",
                     "BindIndexBuffer 0xb2 20480 1", "DrawIndexedIndirect 0xb3 512 1 20",
                     "ClearAttachments aspect=0x1 att=0 color=1,0,0,1 rect=8,20 16x8 layers=0+1"], draws
    assert "BindPipeline 0 0x9a" in f and "BindDescriptorSets 0 0x6100 0 0x5e7" in f
    assert f"op graphics {VS_KEY} {PS_KEY} colors=1 c0=37 depth=0 stencil=0" in out
    assert [p for p in frame(out, 2) if p.startswith("Draw")] == ["Draw 3 1 3 0"]


def check_drop(out: str, want: list[str]) -> None:
    """The vertex stage's uniform binding (binding 2) of each draw of the frame starts with the text given for it."""
    w = writes(frame(out, 1), 2)
    assert len(w) == len(want) and all(x.startswith(t) for x, t in zip(w, want)), w


def check_transfers(out: str) -> None:
    f = frame(out, 1)
    barrier = "PipelineBarrier2 full=1"
    names = ("CopyBuffer", "FillBuffer", "CopyImage", "CopyBufferToImage", "CopyImageToBuffer", "BlitImage", "ClearColorImage",
             "ClearDepthStencilImage", "BeginRendering")
    at = [i for i, p in enumerate(f) if p.split(" ")[0] in names and i > 1]
    assert len(at) == 14, at
    assert all(f[i - 1] == barrier for i in at), "a transfer is not right behind a full barrier"
    got = [f[i] for i in at]
    mir, host, vram = 0xB2, 0xB1, 0xB3
    area = "aspect=0x1 level=0 layers=0+1"
    want = [
        f"CopyBuffer 0x5c->{mir:#x} 0->{0x900} size=8",                           # UpdateUniformBuffer through scratch
        f"CopyBuffer {host:#x}->{mir:#x} {0x400}->{0xA00} size=256",              # host to mirror
        f"CopyBuffer {host:#x}->{mir:#x} {0x400}->{0xA40} size=64",               # inside the write above
        f"CopyBuffer {vram:#x}->{host:#x} {0x1000}->{0x800} size=64",             # VRAM to host
        f"FillBuffer {vram:#x} 64 32 {0xDEAD}",
        # the region's blocks tightly packed, then at a 256-byte row stride, then two layers at a 1 KiB image stride
        f"CopyImageToBuffer img=0x1001 buf={mir:#x} off={0x1000} row=16 imgh=4 {area} at=8,4,0 ext=16x4x1",
        f"CopyImageToBuffer img=0x1001 buf={mir:#x} off={0x2000} row=64 imgh=4 {area} at=8,4,0 ext=16x4x1",
        f"CopyImageToBuffer img=0x1002 buf={mir:#x} off={0x4000} row=8 imgh=32 aspect=0x1 level=0 layers=1+2 at=0,0,0 ext=8x8x1",
        f"CopyBufferToImage buf={host:#x} img=0x1001 off={0x2000} row=8 imgh=8 {area} at=0,0,0 ext=8x8x1",
        f"CopyImage 0x1001->0x1003 src {area} at=0,0,0 dst {area} at=4,4,0 ext=32x16x1",
        f"BlitImage 0x1001->0x1003 filter=1 src {area} 0,0,0..16,8,1 dst {area} 0,0,0..32,16,1",
        "ClearColorImage 0x1001 color=1,0,0,1 aspect=0x1 level=0+1 layers=0+1",
        "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0]",   # the part of a level, in a pass
        "ClearDepthStencilImage 0x1004 depth=1 stencil=7 aspect=0x6 level=0+1 layers=0+1",
    ]
    assert got == want, [(g, w) for g, w in zip(got, want) if g != w]
    part = f.index(want[12])
    assert f[part + 1] == "ClearAttachments aspect=0x1 att=0 color=0,1,0,1 rect=8,4 16x8 layers=0+1", f[part + 1]
    assert f[part + 2] == "EndRendering"
    # the dispatch after the last transfer opens with a barrier of its own
    assert f[f.index("BindPipeline 1 0x9c") - 1] == barrier
    # the writes into memory the CPU can see are declared once each (the copy inside an earlier write adds none)
    writes_ = [line for line in out.splitlines() if line.startswith("op wrote ")]
    assert writes_ == ["op wrote M+0x900 0x8 frame 1", "op wrote M+0xa00 0x100 frame 1", "op wrote H+0x800 0x40 frame 1",
                       "op wrote V+0x40 0x20 frame 1", "op wrote M+0x1000 0x100 frame 1", "op wrote M+0x2000 0x340 frame 1",
                       "op wrote M+0x4000 0x500 frame 1"], writes_


def draw_groups(f: list[str]) -> list[list[str]]:
    """The packets of each draw: the dynamic state set since the previous draw, then the draw itself."""
    groups, cur = [], []
    for p in f:
        cur.append(p)
        if p.startswith("Draw"):
            groups.append(cur)
            cur = []
    return groups


def check_line_fill_and_bias_clamp(g: list[str]) -> None:
    assert "SetPolygonModeEXT 1" in g and "SetDepthBiasEnable 1" in g, g
    assert "SetDepthBias 2 0.25 1.5" in g, [p for p in g if p.startswith("SetDepthBias")]   # constant, clamp, slope


def check_depth_clamp_and_bounds(g: list[str]) -> None:
    assert "SetDepthClampEnableEXT 1" in g, g
    assert "SetDepthBoundsTestEnable 1" in g and "SetDepthBounds 0.25 0.75" in g, g


def check_wide_line(g: list[str]) -> None:
    assert "SetLineWidth 3.5" in g, g


def check_viewports(g: list[str]) -> None:
    # three viewports (each flipped), three scissors: the two set, then the whole area
    assert ("SetViewportWithCount 3 [0 32 64 -32 0 1] [0 32 32 -16 0 1] [32 16 32 -16 0 1]" in g and
            "SetScissorWithCount 3 [8 20 16 8] [0 22 10 10] [0 0 64 32]" in g), g


def check_independent_blend(g: list[str]) -> None:
    assert "SetColorBlendEnableEXT 0 2 1 0" in g, g
    assert "SetColorBlendEquationEXT 0 2 [6 7 0 1 0 0] [1 0 0 1 0 0]" in g, g   # SRC_ALPHA, ONE_MINUS_SRC_ALPHA; default
    assert "SetColorWriteMaskEXT 0 2 0x3 0x8" in g, g


def check_dual_source(g: list[str]) -> None:
    assert "SetColorBlendEquationEXT 0 1 [15 16 0 1 0 0]" in g, g   # SRC1_COLOR, ONE_MINUS_SRC1_COLOR


_BUILT: dict[str, object] = {}


def tearDownModule() -> None:
    if "tmp" in _BUILT:
        _BUILT.pop("tmp").cleanup()  # type: ignore[attr-defined]


class ReplayBase(unittest.TestCase):
    """vk_replay_test built once for the module (the runtime is a long compile), at one level."""

    @classmethod
    def setUpClass(cls) -> None:
        if "error" in _BUILT:
            raise RuntimeError("vk_replay_test did not build (see the first class's error)")
        if "exe" not in _BUILT:
            cc = clang_or_skip()
            include, glslang = vulkan_or_skip()
            tmp = tempfile.TemporaryDirectory(prefix="oot2-vkreplay-", ignore_cleanup_errors=True)
            _BUILT["tmp"] = tmp
            try:
                _BUILT["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
            except RuntimeError:
                _BUILT["error"] = True
                raise
        cls.exe = _BUILT["exe"]

    def run_case(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=60, env=case_env())

    def passing(self, case: str, *args: str) -> str:
        done = self.run_case(case, *args)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        return done.stdout

    def stopping(self, *args: str) -> str:
        done = self.run_case(*args)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertNotIn("returned", done.stdout)
        fatal = fatal_lines(done.stderr)
        self.assertEqual(len(fatal), 1, done.stderr)
        return fatal[0]


class GraphicsTest(ReplayBase):
    CHECKS = (check_pass, check_flip, check_depth_clip, check_snapshot, check_live_mirror, check_banks, check_storage,
              check_push_slot, check_vertex_input, check_draws)

    def test_draws_become_their_packets(self) -> None:
        out = self.passing("graphics")
        for check in self.CHECKS:
            with self.subTest(check=check.__name__):
                check(out)
        self.assertIn("stats frames 2 draws 4 clears 1 passes 2 snapshots 2 live 8 writes 2", out)

    def test_counter_run_without_the_flip(self) -> None:
        out = self.passing("graphics", "--mutant", "noflip")
        with self.assertRaises(AssertionError):
            check_flip(out)
        check_snapshot(out)

    def test_counter_run_without_the_snapshot(self) -> None:
        out = self.passing("graphics", "--mutant", "nosnapshot")
        with self.assertRaises(AssertionError):
            check_snapshot(out)
        check_flip(out)

    def test_a_draw_the_miss_path_skipped_writes_nothing_and_every_record_is_accounted_for(self) -> None:
        out = self.passing("skip")
        f = frame(out, 1)
        draws = [p for p in f if p.startswith("Draw")]
        self.assertEqual(draws, ["Draw 3 1 3 0"], draws)      # the second draw only
        self.assertEqual(f.count("BindPipeline 0 0x9a"), 1)    # the first one bound nothing
        self.assertIn("stats draw_ops 2 draws 1 skipped 1 render_off 0 lost 0", out)

    def test_the_games_shading_rate_is_replayed_at_full_rate(self) -> None:
        out = self.passing("full_rate")
        self.assertEqual(len([p for p in frame(out, 1) if p.startswith("Draw")]), 1)
        self.assertIn("stats draws 1 full_rate 2 lost 0", out)   # the image and the nonzero mode, not their unbinds

    def test_the_geometry_processing_modes_0_to_2_are_accepted_and_logged_once_each(self) -> None:
        for mode in (0, 1, 2):
            with self.subTest(mode=mode):
                done = self.run_case("geometry", str(mode))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok geometry", done.stdout)
                self.assertIn("stats draws 2 lost 0", done.stdout)   # one draw in each of two submissions, mode set 4 times
                lines = [ln for ln in done.stderr.splitlines() if "geometry processing mode" in ln]
                self.assertEqual(len(lines), 1, done.stderr)
                self.assertIn(f"the game set geometry processing mode {mode}:", lines[0])
                self.assertEqual(fatal_lines(done.stderr), [])

    def test_a_geometry_processing_mode_beyond_2_stops_by_name(self) -> None:
        for mode in ("3", "-1", "0x7fffffff"):
            with self.subTest(mode=mode):
                line = self.stopping("geometry_stop", mode)
                value = int(mode, 0)
                self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
                                       f"GEOMETRY_MODE {value} is not replayed: only the modes 0 to 2 are")

    def test_a_draw_that_stops_on_its_answers_leaves_its_rows_in_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-ledger-") as folder:
            line = self.stopping("attr_stop", folder)
            self.assertRegex(line, r"^FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: vertex "
                                   r"attribute 0 \(NVN format 0xb\) is of another scalar type than the module's input$")
            rows = (Path(folder) / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[1:]
            vertex = [r for r in rows if '"kind":"vs"' in r]
            self.assertEqual(len(vertex), 1, rows)
            self.assertIn('"attr":{"0":"uint"}', vertex[0])

    def collecting(self, case: str, folder: str, on: bool) -> subprocess.CompletedProcess:
        """A case with the development setting collect_variants on (the environment layer) or left alone."""
        env = case_env()
        env.pop("OOT2_COLLECT_VARIANTS", None)
        if on:
            env["OOT2_COLLECT_VARIANTS"] = "1"
        return subprocess.run([str(self.exe), case, folder], capture_output=True, text=True, timeout=60, env=env)

    def test_with_collect_variants_on_draws_that_stop_on_their_answers_are_skipped_and_their_rows_written(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-collect-") as folder:
            done = self.collecting("attr_collect", folder, True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn("submit returned", done.stdout)
            self.assertIn("ok attr_collect", done.stdout)
            # unsigned, signed and unsigned again are skipped and counted (3); the float one is drawn (1); nothing is
            # lost; the ledger holds a vertex row for each of the three answers, two of them from the skipped draws.
            self.assertIn("stats draws 1 collected 3 lost 0 ledger_collected 3 kinds 2 vs_rows 3", done.stdout)
            rows = (Path(folder) / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[1:]
            vertex = [r for r in rows if '"kind":"vs"' in r]
            self.assertEqual(len(vertex), 3, rows)
            self.assertTrue(any('"attr":{"0":"uint"}' in r for r in vertex), vertex)
            self.assertTrue(any('"attr":{"0":"sint"}' in r for r in vertex), vertex)
            self.assertEqual(fatal_lines(done.stderr), [])
            # loud: one line that says the run is not a proof, and one line for each stage and answer (not each draw)
            self.assertEqual(done.stderr.count("COLLECTING VARIANTS"), 1, done.stderr)
            self.assertIn("NOT a proof run", done.stderr)
            skipped = [ln for ln in done.stderr.splitlines() if "draw skipped for collection" in ln]
            self.assertEqual(len(skipped), 2, done.stderr)
            self.assertIn(f"stage {VS_KEY}", skipped[0])
            self.assertIn("is of another scalar type than the module's input", skipped[0])

    def test_a_collection_run_counts_its_skipped_draws_in_the_capture_and_the_proof_judge_refuses_it(self) -> None:
        for on in (True, False):
            with self.subTest(collect_variants=on), tempfile.TemporaryDirectory(prefix="oot2-vkreplay-capture-") as folder:
                env = case_env({"OOT2_CAPTURE": "1", **({"OOT2_COLLECT_VARIANTS": "1"} if on else {})})
                done = subprocess.run([str(self.exe), "attr_collect", folder], capture_output=True, text=True,
                                      timeout=60, env=env)
                files = sorted(Path(folder).glob("capture-*.tsv"))
                self.assertEqual(len(files), 1, done.stdout + done.stderr)
                cap = cc.parse(files[0])
                self.assertEqual(cap.header["collect_variants"], "1" if on else "0")
                if not on:   # the first draw stops by name: the stop path closes the capture, no draw counted
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertEqual(cap.end.counters[cc.COUNTERS.index("skipped draws")], 0)
                    continue
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                # the three skipped draws are the capture's skipped draws, and the judge fails the run twice over
                self.assertEqual(cap.end.counters[cc.COUNTERS.index("skipped draws")], 3)
                failed = {c.name for c in cc.check(cap).conditions if not c.ok}
                self.assertLessEqual({"judged run", "skipped draws"}, failed)

    def test_with_collect_variants_off_the_first_draw_that_stops_on_its_answers_stops_by_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-collect-off-") as folder:
            done = self.collecting("attr_collect", folder, False)
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            self.assertNotIn("returned", done.stdout)
            fatal = fatal_lines(done.stderr)
            self.assertEqual(len(fatal), 1, done.stderr)
            self.assertRegex(fatal[0], r"^FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: vertex "
                                       r"attribute 0 \(NVN format 0xb\) is of another scalar type than the module's input$")
            self.assertNotIn("COLLECTING VARIANTS", done.stderr)
            self.assertNotIn("draw skipped for collection", done.stderr)

    def test_with_collect_variants_on_a_stop_that_is_not_an_answer_still_stops(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-collect-other-") as folder:
            done = self.collecting("collect_other_stop", folder, True)
            self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
            self.assertNotIn("returned", done.stdout)
            fatal = fatal_lines(done.stderr)
            self.assertEqual(len(fatal), 1, done.stderr)
            self.assertRegex(fatal[0], r"after 0 draws and dispatches: conservative rasterization is not replayed yet$")
            self.assertNotIn("draw skipped for collection", done.stderr)

    def test_a_draw_with_rendering_off_is_counted_as_that_and_not_as_lost(self) -> None:
        out = self.passing("skip_render_off")
        self.assertIn("stats draw_ops 3 draws 1 skipped 1 render_off 1 lost 0", out)

    def test_depth_targets_and_clears(self) -> None:
        f = frame(self.passing("depth"), 1)
        self.assertEqual(f, [
            "PipelineBarrier2 full=1",
            "PipelineBarrier2 full=1",
            "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0] "
            "depth [0x7d00 layout=1 load=0 store=0] stencil [0x7d00 layout=1 load=0 store=0]",
            "ClearAttachments aspect=0x6 att=0 depth=1 stencil=5 rect=0,0 64x32 layers=0+1",
            "ClearAttachments aspect=0x1 att=0 color=0,0.5,0,1 rect=0,24 32x8 layers=0+1",
            "EndRendering",
            "PipelineBarrier2 full=1",
            "BeginRendering 0,0 64x32 layers=1 colors=1 [0x7125 layout=1 load=0 store=0]",
            "ClearAttachments aspect=0x1 att=0 color=0,0,1,1 rect=0,24 32x8 layers=0+1",
            "EndRendering",
        ])

    def test_states_that_needed_a_core_feature_draw(self) -> None:
        out = self.passing("features")
        groups = draw_groups(frame(out, 1))
        self.assertEqual(len(groups), 7)
        line_fill, point_fill, clamped, wide, viewports, two_targets, dual = groups
        check_line_fill_and_bias_clamp(line_fill)
        self.assertEqual([p for p in point_fill if p.startswith("SetPolygonModeEXT")], ["SetPolygonModeEXT 2"])
        check_depth_clamp_and_bounds(clamped)
        check_wide_line(wide)
        check_viewports(viewports)
        check_independent_blend(two_targets)
        check_dual_source(dual)

    def test_counter_run_without_the_feature_states(self) -> None:
        out = self.passing("features", "--mutant", "nofeatures")
        groups = draw_groups(frame(out, 1))
        for check, group in ((check_line_fill_and_bias_clamp, groups[0]), (check_depth_clamp_and_bounds, groups[2]),
                             (check_wide_line, groups[3]), (check_viewports, groups[4])):
            with self.assertRaises(AssertionError):
                check(group)

    def test_dual_source_blending_with_two_targets_stops_by_name(self) -> None:
        self.assertEqual(self.stopping("dual_two_targets"),
                         "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: dual-source blending "
                         "on target 0 of 2: Vulkan guarantees one colour target with it")

    def test_an_indirect_draw_carries_its_base_instance_into_c0(self) -> None:
        f = frame(self.passing("draw_indirect"), 1)
        draws = [i for i, p in enumerate(f) if p.startswith("DrawIndexedIndirect")]
        self.assertEqual(len(draws), 2)
        readable, gpu = (writes(f[a:b], 1) for a, b in ((0, draws[0]), (draws[0], draws[1])))
        self.assertEqual([word(x, 0x644) for x in readable], [7])   # the record's base instance, read at submit
        self.assertEqual([word(x, 0x644) for x in gpu], [None])     # a record in GPU memory keeps 0

    def test_mesh_draws_preserve_first_task_for_direct_and_gpu_written_indirect_records(self) -> None:
        out = self.passing("mesh")
        f = frame(out, 1)
        pushes = [p for p in f if p.startswith("PushConstants")]
        self.assertEqual(len(pushes), 5, pushes)
        first_tasks = [re.search(r"first_task=(\d+)", p) for p in pushes]
        self.assertEqual([m.group(1) if m else None for m in first_tasks],
                         ["21", "37", "9", "41", "7"], pushes)
        self.assertEqual(sum(p.startswith("DrawMeshTasksEXT") for p in f), 3)
        self.assertEqual(sum(p.startswith("DrawMeshTasksIndirectEXT") for p in f), 2)
        copies = [p for p in f if p.startswith("CopyBuffer")]
        self.assertEqual(len(copies), 4, copies)
        converted = [re.fullmatch(r"CopyBuffer 0xb3->0x5c (\d+)->(\d+) size=4", p) for p in copies]
        self.assertTrue(all(converted), copies)
        values = [(int(m.group(1)), int(m.group(2))) for m in converted if m]
        self.assertEqual([src for src, _ in values], [0x100, 0x104, 0x108, 0x10C])
        self.assertEqual(values[1][1] - values[0][1], 12)
        self.assertEqual(values[3][1] - values[2][1], 12)
        self.assertIn("op task_set frame 1 n=1", out)
        self.assertIn("BindDescriptorSets 0 0x6100 2 0x5e8", f)

    def test_counter_run_without_first_task_propagation(self) -> None:
        out = self.passing("mesh", "--mutant", "nofirsttask")
        pushes = [p for p in frame(out, 1) if p.startswith("PushConstants")]
        self.assertEqual(len(pushes), 5, pushes)
        self.assertTrue(all("first_task=unavailable" in p for p in pushes), pushes)

    def test_vertex_restart_state_is_emitted_after_mesh_draws(self) -> None:
        f = frame(self.passing("mesh_restart"), 1)
        restart = [p for p in f if p.startswith("SetPrimitiveRestartEnable")]
        self.assertEqual(restart, ["SetPrimitiveRestartEnable 0", "SetPrimitiveRestartEnable 1",
                                   "SetPrimitiveRestartEnable 0"])

    def test_a_gpu_write_drops_the_uniform_copy_and_later_reads_are_live(self) -> None:
        # read first: draw 1 reads the copy (taken at submit), its pixel stage writes the range, draw 2 reads it live
        out = self.passing("conflict_read_first")
        check_drop(out, ["UB buf=0x5c ", "UB buf=0xb1 off=0x100 range=0x40"])
        self.assertIn("stats snapshots 1 drops 1 ", out)
        # write first: draw 1 writes it with the uniform unbound, the draw that binds it afterwards reads it live
        out = self.passing("conflict_write_first")
        check_drop(out, ["UB buf=0x5c off=0x1100 range=0x10000 zero", "UB buf=0xb1 off=0x100 range=0x40"])
        self.assertIn("stats snapshots 0 drops 0 ", out)

    def test_counter_run_without_the_drop_rule(self) -> None:
        # the copy outlives the write (read first), and a read after a write is copied anyway (write first): both stale
        out = self.passing("conflict_read_first", "--mutant", "nodrop")
        with self.assertRaises(AssertionError):
            check_drop(out, ["UB buf=0x5c ", "UB buf=0xb1 off=0x100 range=0x40"])
        out = self.passing("conflict_write_first", "--mutant", "nodrop")
        with self.assertRaises(AssertionError):
            check_drop(out, ["UB buf=0x5c off=0x1100 range=0x10000 zero", "UB buf=0xb1 off=0x100 range=0x40"])

    def test_transfers_become_their_packets(self) -> None:
        out = self.passing("transfers")
        check_transfers(out)
        self.assertIn("stats transfers 14 bytes 22504 drops 0 writes 7", out)

    def test_counter_run_without_declaring_the_writes(self) -> None:
        out = self.passing("transfers", "--mutant", "nodeclare")
        with self.assertRaises(AssertionError):
            check_transfers(out)
        self.assertEqual(out.count("op wrote"), 0)

    def test_a_transfer_that_cannot_be_replayed_stops_by_name(self) -> None:
        prefix = "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
        want = {
            "cpu_texture": "CopyTextureToBuffer over a texture with no image (NVN format 0x25, 64x32)",
            "overlap": "a buffer copy of 0x100 bytes from ",
            "unaligned_fill": "a buffer fill of 0x20 bytes at ",
            "bad_stride": "CopyBufferToTexture with a copy row stride of 6 bytes over 8 blocks of 4 bytes",
            "masked_clear": "ClearTexture with channel mask 0x3: a masked clear needs a draw",
            "outside": "CopyTextureToBuffer over the region 60,0,0 8x8x1 outside level 0 of a 64x32 texture",
            "mirrored": "CopyTextureToTexture with copy flags 0x4: a mirrored copy is not replayed yet",
            "decoded_b2t": "CopyBufferToTexture over a texture of ASTC blocks decoded to RGBA8 (NVN format 0x79): a texel "
                           "copy between the game's blocks and the image's texels is not replayed yet",
            "decoded_t2b": "CopyTextureToBuffer over a texture of ASTC blocks decoded to RGBA8 (NVN format 0x79): a texel "
                           "copy between the game's blocks and the image's texels is not replayed yet",
        }
        for what, text in want.items():
            with self.subTest(stop=what):
                line = self.stopping("xstop", what)
                self.assertTrue(line.startswith(prefix + text), line)

    def test_a_copy_between_two_images_of_decoded_astc_goes_through(self) -> None:
        # The green counterpart of the two decoded stops above: texels against texels is a plain image copy.
        done = self.run_case("xstop", "decoded_pair")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("submit returned", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])


class ComputeTest(ReplayBase):
    def test_a_raw_access_dispatch_asks_for_the_page_table_before_its_binds_resolve(self) -> None:
        # The frame's raw path begins before any bind of a raw-access draw or dispatch resolves, so that nothing the frame
        # binds from then on becomes a mirror (oot2/backing.h, The raw path): the words reach the push constants, and the
        # frame's second stage asks again and gets the same.
        out = self.passing("raw_first")
        frames: list[list[str]] = []   # the ops of each frame: frame 1 two dispatches, frame 2 a draw
        for line in (line for line in out.splitlines() if line.startswith("op ")):
            if line == "op frame_begin":
                frames.append([])
            else:
                frames[-1].append(line)
        self.assertEqual(len(frames), 2)
        for n, ops in enumerate(frames, 1):
            first_raw = next(i for i, line in enumerate(ops) if line.startswith("op raw "))
            first_resolve = next(i for i, line in enumerate(ops) if line.startswith("op resolve "))
            self.assertLess(first_raw, first_resolve, (n, ops))
        self.assertEqual([line for line in frames[0] if line.startswith("op raw ")], ["op raw frame 1"] * 2)
        self.assertEqual([line for line in frames[1] if line.startswith("op raw ")], ["op raw frame 2"])
        for n in (1, 2):
            pcs = [p for p in frame(out, n) if p.startswith("PushConstants ")]
            self.assertEqual(len(pcs), 2 if n == 1 else 1, pcs)
            for p in pcs:
                self.assertIn(" 0=0x7ab1e000 4=0x7e 8=0x80 12=0x400", p)

    def test_the_miss_diagnostics_record_a_copy_of_the_miss_log_and_a_clear_of_its_count_after_each_raw_access_stage(self) -> None:
        # Settings raw_diag (oot2/vk_raw.h): the ops give a snapshot after a raw-access dispatch and at the end of a pass whose
        # draws had raw-access stages. The replay records, between full barriers and outside any pass, the copy of the page
        # table's header and miss log into the snapshot and the clear of the count, so each stage keeps 64 entries of its own.
        out = self.passing("raw_snap")
        barrier = "PipelineBarrier2 full=1"
        f = frame(out, 1)
        dispatches = [i for i, p in enumerate(f) if p.startswith("Dispatch ")]
        self.assertEqual([f[i] for i in dispatches], ["Dispatch 4 2 1", "Dispatch 1 1 1", "Dispatch 2 1 1"])
        self.assertEqual(f[dispatches[0] + 1:dispatches[0] + 6],
                         [barrier, "CopyBuffer 0x7ab1e->0xd1a6 0->0 size=1088", barrier, "FillBuffer 0x7ab1e 16 4 0", barrier])
        between = f[dispatches[1] + 1:dispatches[2]]   # the program with no raw access gets none
        self.assertFalse([p for p in between if p.startswith(("CopyBuffer", "FillBuffer"))], between)
        self.assertEqual(f[dispatches[2] + 1:dispatches[2] + 6],
                         [barrier, "CopyBuffer 0x7ab1e->0xd1a6 0->1088 size=1088", barrier, "FillBuffer 0x7ab1e 16 4 0", barrier])
        g = frame(out, 2)   # the pass of the two raw-access draws: the snapshot follows its end, once
        end = g.index("EndRendering")
        self.assertEqual(g[end + 1:end + 6],
                         [barrier, "CopyBuffer 0x7ab1e->0xd1a6 0->2176 size=1088", barrier, "FillBuffer 0x7ab1e 16 4 0", barrier])
        self.assertEqual(len([p for p in g if p.startswith("Draw")]), 2)
        self.assertEqual(len([p for p in g if p.startswith("FillBuffer")]), 1)
        self.assertEqual(max(i for i, p in enumerate(g) if p.startswith("Draw")) + 1, end)   # no copy inside the pass
        # what the replay told the ops: the stage's key, its kind, the draws, and the storage buffer it bound
        ops = [line for line in out.splitlines() if line.startswith("op raw_snap ")]
        self.assertEqual(len(ops), 3)
        self.assertEqual(ops[0], "op raw_snap frame 1 kind c draws 0 keys 00000000000000c4.00000001 ranges s:0x40000300+0x20")
        self.assertEqual(ops[1], ops[0])
        self.assertTrue(ops[2].startswith("op raw_snap frame 2 kind d draws 2 keys 00000000000000a2.00000001 ranges"), ops[2])

    def test_an_indirect_mesh_draw_is_snapshotted_with_its_own_pass_and_not_with_the_one_its_record_ended(self) -> None:
        # The record of an indirect mesh draw is copied by the GPU between two passes: the pass the draw was in ends and begins
        # again. The snapshot of the draws before it is recorded as that pass ends, before the record's copies, and the draw
        # itself belongs to the pass after: its snapshot follows the draw, never precedes it.
        out = self.passing("raw_snap_mesh")
        f = frame(out, 1)
        direct = next(i for i, p in enumerate(f) if p.startswith("DrawMeshTasksEXT"))
        indirect = next(i for i, p in enumerate(f) if p.startswith("DrawMeshTasksIndirectEXT"))
        dispatch = next(i for i, p in enumerate(f) if p.startswith("Dispatch "))
        snaps = [i for i, p in enumerate(f) if p.startswith("CopyBuffer 0x7ab1e")]
        records = [i for i, p in enumerate(f) if p.startswith("CopyBuffer") and "0x7ab1e" not in p and direct < i < indirect]
        self.assertEqual(len(snaps), 3, f)
        self.assertTrue(records, f)
        self.assertTrue(direct < snaps[0] < records[0], (direct, snaps, records))
        self.assertTrue(indirect < snaps[1] < dispatch, (indirect, snaps, dispatch))
        self.assertGreater(snaps[2], dispatch)
        ops = [line for line in out.splitlines() if line.startswith("op raw_snap ")]
        self.assertEqual([line.split(" keys ")[0] for line in ops],
                         ["op raw_snap frame 1 kind d draws 1", "op raw_snap frame 1 kind d draws 1",
                          "op raw_snap frame 1 kind c draws 0"])
        self.assertIn(" keys 00000000000000d4.00000001 ranges", ops[0])
        self.assertIn(" keys 00000000000000d4.00000001 ranges", ops[1])
        self.assertIn(" keys 00000000000000c4.00000001 ranges", ops[2])

    def test_ops_that_refuse_a_snapshot_or_give_none_record_nothing(self) -> None:
        for case in ("raw_snap_refused", "raw_first"):
            with self.subTest(case=case):
                out = self.passing(case)
                self.assertFalse([p for p in packets(out) if p.startswith("FillBuffer")], case)
                self.assertFalse([p for p in packets(out) if p.startswith("CopyBuffer 0x7ab1e")], case)
        refused = self.passing("raw_snap_refused")
        self.assertEqual(len([line for line in refused.splitlines() if line.startswith("op raw_snap ")]), 3)   # asked, answered 0

    def test_a_negative_base_vertex_resolves_the_streams_first_byte_once_more(self) -> None:
        # The draw reads below the start of its stream, inside the buffer, which the stream's resolve does not state: the
        # backing map takes a resolve of one byte for an extent not stated (backing.c, import_gives). Frames 1 and 3 draw
        # with base vertex 2, frame 2 with -2: the one extra resolve is the negative draw's, after the stream's own.
        out = self.passing("base_vertex")
        frames: list[list[str]] = []
        for line in (line for line in out.splitlines() if line.startswith("op ")):
            if line == "op frame_begin":
                frames.append([])
            else:
                frames[-1].append(line)
        self.assertEqual(len(frames), 3)
        stream = [[line for line in ops if line.startswith("op resolve 0x30003000+")] for ops in frames]
        self.assertEqual(stream, [["op resolve 0x30003000+0x40"],
                                  ["op resolve 0x30003000+0x40", "op resolve 0x30003000+0x1"],
                                  ["op resolve 0x30003000+0x40"]])

    def test_dispatches_become_their_packets(self) -> None:
        out = self.passing("compute")
        f = frame(out, 1)
        self.assertEqual(out.count(f"op compute {CS_KEY}\n"), 2)
        self.assertEqual(f[0], "PipelineBarrier2 full=1")
        self.assertIn("BindPipeline 1 0x9c", f)
        self.assertIn("BindDescriptorSets 1 0x6200 0 0x5e7", f)
        self.assertEqual([p for p in f if p.startswith("Dispatch")], ["Dispatch 4 2 1", "DispatchIndirect 0xb1 512"])
        self.assertNotIn("BeginRendering", " ".join(f))
        c0 = writes(f, 1)
        self.assertEqual(len(c0), 2)
        # a direct dispatch's counts are in its bank; an indirect one's are the GPU's, copied in just before it runs
        for x, groups in zip(c0, ((4, 2, 1), (0, 0, 0))):
            self.assertEqual([word(x, o) for o in (4, 8, 12)], [g or None for g in groups], x)
            self.assertEqual((word(x, 0x230), word(x, 0x238)), (0x30007000, 0x80), x)   # uniform row 2
            self.assertEqual((word(x, 0x320), word(x, 0x328)), (0x40000300, 0x20), x)   # storage row 1
        roots = writes(f, 3)
        self.assertTrue(roots and "range=0x800" in roots[0], roots)
        self.assertEqual((word(roots[0], 0x208), word(roots[0], 0x20c)), (0x11, 0x22))
        self.assertEqual(writes(f, 4), ["SB buf=0xb3 off=0x300 range=0x20"] * 2)
        self.assertNotIn("op wrote", out)   # the module only reads it

    def test_a_storage_image_binds_its_view_and_only_one_placed_in_its_storage_is_declared_written(self) -> None:
        out = self.passing("storage_image")
        check_storage_image(out)
        self.assertIn("stats storage_images 3 writes 1", out)

    def test_counter_run_without_declaring_the_image_writes(self) -> None:
        out = self.passing("storage_image", "--mutant", "nodeclare")
        with self.assertRaises(AssertionError):
            check_storage_image(out)

    def test_a_storage_image_that_cannot_be_bound_stops_by_name(self) -> None:
        prefix = f"FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: stage {CS_IMG_KEY}"
        want = {
            "unregistered": " stores to image ID 11, where nothing is registered",
            "buffer": " stores to a buffer texture",
            "set": " has a storage image at set 2",
            "slot": " stores to image slot -1",
            "no_ops": " binds a storage image, and these ops make no storage view",
        }
        for what, text in want.items():
            with self.subTest(stop=what):
                line = self.stopping("storage_stop", what)
                self.assertTrue(line.startswith(prefix + text), line)

    def test_an_indirect_dispatch_has_its_counts_copied_by_the_gpu_into_a_bank_of_its_own(self) -> None:
        f = frame(self.passing("indirect"), 1)
        barrier = "PipelineBarrier2 full=1"
        copies = [i for i, p in enumerate(f) if p.startswith("CopyBuffer")]
        self.assertEqual(len(copies), 3)
        banks = [x for x in writes(f, 1)]
        self.assertEqual(len(banks), 4)   # three indirect dispatches and one direct
        for i, buf, ind in zip(copies, ("0xb3", "0xb2", "0xb3"), (1280, 1536, 1280)):
            self.assertEqual((f[i - 1], f[i + 1]), (barrier, barrier))
            self.assertRegex(f[i], rf"^CopyBuffer {buf}->0x5c {ind}->\d+ size=12$")
            self.assertEqual(f[i + 2], f"DispatchIndirect {buf} {ind}")
        # each copy lands at byte 4 of the bank its dispatch binds, and no two dispatches share a bank
        dst = [int(re.search(r"->(\d+) ", f[i]).group(1)) - 4 for i in copies]
        bank_offsets = [int(re.search(r" off=(0x[0-9a-f]+) ", x).group(1), 16) for x in banks]
        self.assertEqual([bank_offsets[0], bank_offsets[1], bank_offsets[3]], dst)
        self.assertEqual(len(set(bank_offsets)), 4)


KEEP_KEY, MID_RAW_KEY, MID_IMG_KEY, LAST_KEY = (f"00000000000000e{i}.00000001" for i in (1, 2, 3, 4))
MOVIE_KEYS = (f"# a synthetic chain: the keys are the test's own\nkeep {KEEP_KEY}\nskip {MID_RAW_KEY}   # the page table\n"
              f"skip {MID_IMG_KEY}\n\nblit {LAST_KEY}\n")
# the blit of texture ID 31's image (192x108) into ID 33's (256x144), linear
BLIT_PACKET = ("BlitImage 0x201f->0x2021 filter=1 src aspect=0x1 level=0 layers=0+1 0,0,0..192,108,1 "
               "dst aspect=0x1 level=0 layers=0+1 0,0,0..256,144,1")
MOVIE_SWITCH = "OOT2_TEST_MOVIE_BLIT"
MOVIE_FILE = "OOT2_MOVIE_BLIT_KEYS"


def ops_by_frame(out: str) -> list[list[str]]:
    """The ops lines of each frame, split at "op frame_begin"."""
    frames: list[list[str]] = []
    for line in (line for line in out.splitlines() if line.startswith("op ")):
        if line == "op frame_begin":
            frames.append([])
        else:
            frames[-1].append(line)
    return frames


class MovieBlitTest(ReplayBase):
    """The movie chain's stand-in (settings.h test_movie_blit, oot2/vk_movie_blit.h): a test switch that changes the picture,
    off by default. The programs are synthetic; their keys are in a file of the test's own, as the player's would be."""

    last_stderr = ""

    def movie(self, *args: str, on: bool = True, keys: str | bytes | None = MOVIE_KEYS, folder: Path | None = None,
              env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        full = case_env()
        full.pop(MOVIE_SWITCH, None)
        full.pop(MOVIE_FILE, None)
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-movie-", ignore_cleanup_errors=True) as tmp:
            if on:
                full[MOVIE_SWITCH] = "1"
            if keys is not None:
                path = Path(tmp) / "keys.txt"
                path.write_bytes(keys.encode("utf-8") if isinstance(keys, str) else keys)
                full[MOVIE_FILE] = str(path)
            full.update(env or {})
            more = [str(folder)] if folder else []
            return subprocess.run([str(self.exe), *args, *more], capture_output=True, text=True, timeout=60, env=full)

    def chain(self, **kw) -> str:
        done = self.movie("movie_chain", **kw)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ok movie_chain", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.last_stderr = done.stderr
        return done.stdout

    def test_with_the_setting_on_the_chain_runs_its_first_program_and_resamples_its_picture_into_the_last_ones_output(self) -> None:
        out = self.chain()
        ops = ops_by_frame(out)
        self.assertEqual(len(ops), 4)
        # frame 1: the first program and the plain dispatch run; the two middle ones and the last are not made
        compute = [line for line in ops[0] if line.startswith("op compute ")]
        self.assertEqual(compute, [f"op compute {KEEP_KEY}", f"op compute {CS_KEY}"])
        self.assertFalse([line for line in ops[0] if line.startswith("op raw ")], ops[0])   # no page table: no raw frame
        f = frame(out, 1)
        self.assertEqual([p for p in f if p.startswith("Dispatch")], ["Dispatch 1 1 1", "Dispatch 1 1 1"])
        self.assertEqual([p for p in f if p.startswith("BlitImage")], [BLIT_PACKET])
        barrier = "PipelineBarrier2 full=1"
        at = f.index(BLIT_PACKET)
        # a transfer's barriers: one after the first program's stores, one before whatever reads the output
        self.assertEqual((f[at - 1], f[at + 1]), (barrier, barrier), f[at - 2:at + 3])
        self.assertLess(f.index("Dispatch 1 1 1"), at)   # the first program's dispatch comes before the resample
        # the images it asked for: the picture first, then the output, whose storage is declared written once, as the last
        # program's own stores would have been
        wanted = [line for line in ops[0] if line.startswith(("op image_by_id ", "op wrote "))]
        self.assertEqual(wanted, ["op image_by_id id 31 frame 1", "op image_by_id id 33 frame 1",
                                  "op wrote V+0x50000 0x4000 frame 1"])
        # the later frames have no first program before their middle and last ones (frame 2), or no last one (frame 3), or
        # come after a chain that was never closed (frame 4): they run as they always do
        self.assertEqual([line for line in ops[1] if line.startswith("op compute ")],
                         [f"op compute {k}" for k in (MID_RAW_KEY, MID_IMG_KEY, LAST_KEY)])
        self.assertEqual([line for line in ops[1] if line.startswith("op raw ")], ["op raw frame 2"])
        self.assertEqual([line for line in ops[2] if line.startswith("op compute ")], [f"op compute {KEEP_KEY}"])
        self.assertEqual([line for line in ops[3] if line.startswith("op compute ")], [f"op compute {MID_RAW_KEY}"])
        self.assertEqual([line for line in ops[3] if line.startswith("op raw ")], ["op raw frame 4"])
        self.assertEqual([p for p in frame(out, 2) if p.startswith(("Dispatch", "BlitImage"))], ["Dispatch 1 1 1"] * 3)
        # the accounting: ten dispatch records, seven made, two skipped, one resampled, none lost; the census has the same
        self.assertIn("stats chain draw_ops 10 dispatches 7 movie_skipped 2 movie_blits 1 lost 0 raw_binds 2", out)
        self.assertIn("census chain skipped 2 blits 1", out)

    def test_it_is_loud_the_first_time_and_says_the_run_is_no_proof(self) -> None:
        self.chain()
        err = self.last_stderr
        self.assertEqual(err.count("MOVIE BLIT (test_movie_blit is on)"), 1, err)
        self.assertIn("NOT a proof run", err)
        self.assertIn("test_movie_blit: 1 keep, 2 skip and 1 blit program keys read from ", err)
        skipped = [line for line in err.splitlines() if "are not run (the first is stage" in line]
        self.assertEqual(len(skipped), 1, err)   # once, not for each of the dispatches
        self.assertIn(f"stage {MID_RAW_KEY};", skipped[0])
        blit = [line for line in err.splitlines() if "movie blit: the chain's last dispatch" in line]
        self.assertEqual(len(blit), 1, err)
        self.assertIn(f"(stage {LAST_KEY})", blit[0])
        self.assertIn("192x108 (Vulkan format 37, texture ID 31) into 256x144 (format 37, texture ID 33)", blit[0])

    def test_with_the_setting_off_every_program_runs_and_the_key_file_is_not_read(self) -> None:
        out = self.chain(on=False, keys=None, env={MOVIE_FILE: "no-such-file.txt"})
        ops = ops_by_frame(out)
        self.assertEqual([len([x for x in f if x.startswith("op compute ")]) for f in ops], [5, 3, 1, 1])
        self.assertEqual([line for line in ops[0] if line.startswith("op raw ")], ["op raw frame 1"])
        self.assertNotIn("BlitImage", out)
        self.assertNotIn("image_by_id", out)
        self.assertIn("stats chain draw_ops 10 dispatches 10 movie_skipped 0 movie_blits 0 lost 0 raw_binds 3", out)
        self.assertIn("census chain skipped 0 blits 0", out)
        self.assertNotIn("movie blit", self.last_stderr.lower())
        self.assertNotIn("program keys read", self.last_stderr)

    def test_a_key_file_that_names_none_of_the_programs_leaves_the_run_as_it_is_with_the_setting_off(self) -> None:
        other = "keep 00000000000000f1.00000001\nskip 00000000000000f2.00000001\nblit 00000000000000f3.00000001\n"
        off = self.chain(on=False, keys=None)
        on = self.chain(keys=other)
        self.assertEqual(on, off)   # the ops, the packets and the counters, line for line
        self.assertIn("program keys read from", self.last_stderr)   # only its start-up line is new

    def test_a_chain_that_cannot_be_replaced_stops_by_name_at_the_program_that_shows_it(self) -> None:
        prefix = "FATAL: gfx: vulkan: replay of submission 1, after "
        want = {
            "no_store": ("0 draws and dispatches: test_movie_blit: the chain's first program " + KEEP_KEY +
                         " stores to no image"),
            "no_image": "1 draws and dispatches: test_movie_blit: no image stands on the chain's first picture (texture ID 35)",
            "array": ("1 draws and dispatches: test_movie_blit: the chain's output image is not a single 2D colour image of one "
                      "level (192x108, 1 levels, 2 layers, 1 samples)"),
            "same": "1 draws and dispatches: test_movie_blit: the chain's first picture and its output are one image",
            "two_stores": ("1 draws and dispatches: test_movie_blit: the chain's last program " + LAST_KEY +
                           " stores to more than one image"),
            "no_ops": "1 draws and dispatches: test_movie_blit needs ops that give the image of a texture ID",
            # the chain skipped dispatches and its last program never came: the stop names the first program and the skips
            "unclosed": ("1 draws and dispatches: test_movie_blit: the chain opened by program " + KEEP_KEY +
                         " skipped 2 dispatches (first " + MID_RAW_KEY + ", last " + MID_IMG_KEY +
                         ") and no blit program came before the end of this submission"),
            "rekeep": ("1 draws and dispatches: test_movie_blit: the chain opened by program " + KEEP_KEY +
                       " skipped 1 dispatches (first " + MID_IMG_KEY + ", last " + MID_IMG_KEY +
                       ") and the chain's first program came again before any blit program"),
        }
        for what, text in want.items():
            with self.subTest(stop=what):
                done = self.movie("movie_stop", what)
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertNotIn("returned", done.stdout)
                fatal = fatal_lines(done.stderr)
                self.assertEqual(len(fatal), 1, done.stderr)
                self.assertTrue(fatal[0].startswith(prefix + text), fatal[0])

    def test_a_mistake_in_the_setting_or_the_key_file_is_a_named_fatal_at_start_up(self) -> None:
        good = f"keep {KEEP_KEY}\nblit {LAST_KEY}\n"
        done = self.movie("movie_open", keys=good)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("TEST SWITCH test_movie_blit IS ON", done.stderr)
        self.assertIn("NOT a proof run", done.stderr)
        self.assertIn("1 keep, 0 skip and 1 blit program keys read from", done.stderr)
        done = self.movie("movie_open", on=False, keys="not a key file\n")   # off: nothing is read and nothing said
        self.assertEqual((done.returncode, done.stderr), (0, ""), done.stdout + done.stderr)
        key2 = "00000000000000e5.00000001"
        missing = str(Path(tempfile.gettempdir()) / "oot2-no-such-keys-file.txt")
        # (name, the file's content, the environment, what the fatal says after the setting's own first words)
        cases = [
            ("unset", None, {}, "OOT2_MOVIE_BLIT_KEYS is not set"),
            ("unreadable", None, {MOVIE_FILE: missing}, f"its key file {missing} cannot be opened"),
            ("role", f"keeper {KEEP_KEY}\n", {}, ", line 1: \"keeper\" is not a role (keep, skip or blit)"),
            ("no key", "keep\n", {}, ", line 1: \"keep\" has no key after it"),
            ("short key", "keep 00000000000000e1\n", {}, ", line 1: \"00000000000000e1\" is not a stage key"),
            ("upper case", f"keep {KEEP_KEY.upper()}\n", {}, f", line 1: \"{KEEP_KEY.upper()}\" is not a stage key"),
            ("extra", f"keep {KEEP_KEY} {LAST_KEY}\n", {}, ", line 1: more than a role and a key"),
            ("twice", f"keep {KEEP_KEY}\nskip {key2}\nskip {key2}\nblit {LAST_KEY}\n", {},
             f", line 3: the key {key2} is named twice"),
            ("two roles", f"keep {KEEP_KEY}\nblit {KEEP_KEY}\n", {},
             f", line 2: the key {KEEP_KEY} is in two roles (keep and blit)"),
            ("no blit", f"keep {KEEP_KEY}\nskip {key2}\n", {}, ": it needs a keep line and a blit line (it has 1 and 0)"),
            ("no keep", f"blit {LAST_KEY}\n", {}, ": it needs a keep line and a blit line (it has 0 and 1)"),
            ("too many", "".join(f"skip {i:016x}.00000001\n" for i in range(65)), {}, ", line 65: more than 64 keys"),
            ("nul", (good + "\0").encode("utf-8"), {}, ": it holds a NUL byte"),
            ("long", "# " + "x" * 70000 + "\n", {}, ": it is longer than 65536 bytes"),
        ]
        for what, keys, env, tail in cases:
            with self.subTest(file=what):
                done = self.movie("movie_open", keys=keys, env=env)
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertNotIn("ok movie_open", done.stdout)
                fatal = fatal_lines(done.stderr)
                self.assertEqual(len(fatal), 1, done.stderr)
                self.assertTrue(fatal[0].startswith("FATAL: gfx: vulkan: test_movie_blit is on and "), fatal[0])
                self.assertIn(tail, fatal[0])
                # every start-up stop is met by a saved setting before the menu: each says how to undo it, last
                self.assertTrue(fatal[0].endswith("set OOT2_TEST_MOVIE_BLIT=0 or remove the test_movie_blit line from the "
                                                  "settings file (oot2.ini)"), fatal[0])

    def test_a_capture_of_a_run_with_the_setting_on_is_refused_by_the_judge_on_two_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-movie-cap-", ignore_cleanup_errors=True) as tmp:
            self.chain(folder=Path(tmp), env={"OOT2_CAPTURE": "1"})
            cap = cc.parse(next(Path(tmp).glob("capture-*.tsv")))
        self.assertTrue(cap.header["test_movie_blit"].startswith("1 ("), cap.header["test_movie_blit"])
        self.assertEqual(cap.end.counters[0], 3)   # two skipped dispatches and the resample: the skipped draws
        failing = {c.name: c.value for c in cc.check(cap).conditions if not c.ok}
        self.assertIn("test settings", failing)
        self.assertIn("test_movie_blit = 1 (", failing["test settings"])
        self.assertTrue(failing["skipped draws"].startswith("3"), failing)

    def test_a_capture_of_a_run_with_it_off_has_neither_count_against_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="oot2-vkreplay-movie-cap-off-", ignore_cleanup_errors=True) as tmp:
            self.chain(on=False, keys=None, folder=Path(tmp), env={"OOT2_CAPTURE": "1"})
            cap = cc.parse(next(Path(tmp).glob("capture-*.tsv")))
        self.assertEqual(cap.header["test_movie_blit"], "0")
        self.assertEqual(cap.end.counters[0], 0)
        ok = {c.name for c in cc.check(cap).conditions if c.ok}
        self.assertTrue({"test settings", "skipped draws"} <= ok)


REPORTS = [
    "event 8",
    "samples 10 70 0",
    "timestamp counter 0, ticks in order 1, nonzero 1",
    "reports 6 dropped 1 resets 2 events 2 queries 4 waits 2 unsatisfied 1 conditions 4 false 2",
    "draws 6",
]


class ReportsTest(ReplayBase):
    def test_an_old_unfinished_report_survives_more_than_1024_later_reports(self) -> None:
        """A long submission keeps old pending reports and chooses the newest report per address."""
        out = self.passing("many_pending_reports")
        self.assertIn("first draws 1", out.splitlines())
        self.assertIn("second draws 1", out.splitlines())
        self.assertIn("third draws 2", out.splitlines())

        mutant = self.passing("many_pending_reports", "--mutant", "nopending")
        self.assertIn("first draws 0", mutant.splitlines())
        self.assertIn("second draws 1", mutant.splitlines())

    def test_an_unfinished_report_survives_later_submissions(self) -> None:
        """A report in an unfinished frame remains visible to later replayed frames."""
        out = self.passing("pending_across_frames")
        self.assertIn("draws 1", out.splitlines())
        mutant = self.passing("pending_across_frames", "--mutant", "nopending")
        self.assertIn("draws 0", mutant.splitlines())

    def test_reports_events_and_conditions_complete_in_op_stream_order(self) -> None:
        out = self.passing("reports")
        for line in REPORTS:
            self.assertIn(line, out.splitlines())
        pkt = [x for x in out.splitlines() if x.startswith("pkt ")]
        queries = [x.split()[1] for x in pkt if x.split()[1] in ("BeginQuery", "EndQuery", "Draw", "EndRendering")]
        # frame 1: query 0 over draws 1 and 2, query 1 over draw 3, query 2 over draws 4 and 6, ended by the pass;
        # frame 2: query 0 over draw 8
        self.assertEqual(queries, ["BeginQuery", "Draw", "Draw", "EndQuery", "BeginQuery", "Draw", "EndQuery", "BeginQuery",
                                   "Draw", "Draw", "EndQuery", "EndRendering",
                                   "BeginQuery", "Draw", "EndQuery", "EndRendering"])
        self.assertIn("pkt BeginQuery 0x9901 2 0", pkt)
        self.assertIn("query results pool 0x9901 n 3", out)
        self.assertIn("query results pool 0x9902 n 1", out)

    def test_a_frame_past_one_query_pool_takes_another(self) -> None:
        out = self.passing("many_queries")
        # each query i of a pool counted 10 * (i + 1): 512 in the first pool, 88 in the second
        self.assertIn(f"samples {10 * (512 * 513 // 2 + 88 * 89 // 2)}", out.splitlines())
        self.assertIn("query results pool 0x9901 n 512", out)
        self.assertIn("query results pool 0x9901 n 88", out)

    def test_counter_run_without_the_reports_not_done(self) -> None:
        out = self.passing("reports", "--mutant", "nopending")
        self.assertIn("draws 7", out.splitlines())   # draw 5 read the old 7 in memory and rendered

    def test_counter_run_completed_out_of_order(self) -> None:
        out = self.passing("reports", "--mutant", "reverse")
        self.assertIn("event 5", out.splitlines())   # the ADD before the WRITE


class OpListTest(ReplayBase):
    def test_every_op_is_replayed_or_stops_by_name(self) -> None:
        out = self.passing("oplist")
        rows = [re.fullmatch(r"op (\d+) (\S+) (.+)", line) for line in out.splitlines() if line.startswith("op ")]
        self.assertTrue(rows and all(rows))
        ops = {m.group(2): (int(m.group(1)), m.group(3)) for m in rows}
        self.assertEqual(len(ops), len(rows))
        self.assertEqual({name for name, (_, what) in ops.items() if what == "replayed"}, REPLAYED)
        self.assertIn("past unknown", out)
        for name, (op, what) in sorted(ops.items(), key=lambda kv: kv[1][0]):
            if what == "replayed":
                continue
            with self.subTest(op=name):
                self.assertRegex(what, r"\([^()]+\)$")   # "<what> (<the work that brings it, or why it stays a stop>)"
                line = self.stopping("stop", str(op))
                if name in ("JUMP", "CALL"):
                    self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
                                           f"recording 0 is not whole at +8 (op {op}, 520 bytes)")
                else:
                    self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: "
                                           f"{name} is not replayed yet: {what}")

    def test_a_recording_that_is_not_whole_stops_by_name(self) -> None:
        line = self.stopping("broken")
        self.assertEqual(line, "FATAL: gfx: vulkan: replay of submission 1, after 0 draws and dispatches: recording 0 is "
                               "not whole at +0 (op 25, 32 bytes)")


if __name__ == "__main__":
    unittest.main()
