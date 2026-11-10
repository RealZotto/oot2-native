# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the headless movie decode: tests/runtime/webm_decode.c reads a WebM file with its own streaming reader
(tests/runtime/webm_demux.c) and pushes every video block through the AV1 decoder seam (runtime/video/av1_dav1d.c).

The files are made here, by a small muxer in this test, from the synthetic clips of tests/fixtures/av1 (generated pictures
encoded by rav1e; no game file is read or shaped after). The cases: a clip decodes to the number of blocks the container
holds and agrees with the duration it declares; clusters of unknown size, block groups, a second track and a window of a
larger file read the same; and a file that is cut short, one whose duration declares more frames than it holds, a codec
that is not AV1, a damaged unit and a laced video block each fail by name.

The decoder needs a dav1d kit, as tests/test_runtime_av1.py does: without one the tests skip with the reason said, unless
OOT2_REQUIRE_DAV1D is set.

Run: python -m unittest tests.test_runtime_webm -v
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import build_game  # noqa: E402

RUNTIME_TESTS = ROOT / "tests" / "runtime"
SOURCES = [RUNTIME_TESTS / "webm_decode.c", RUNTIME_TESTS / "webm_demux.c", ROOT / "runtime" / "video" / "av1_dav1d.c"]
FIXTURES = ROOT / "tests" / "fixtures" / "av1"
LEVELS = ("-O1", "-O2")
PERIOD_NS = 33_333_333


def ivf_units(name: str) -> list[bytes]:
    """The temporal units of a fixture clip, without the temporal delimiter a container does not carry."""
    data = (FIXTURES / f"{name}.ivf").read_bytes()
    assert data[:4] == b"DKIF" and data[8:12] == b"AV01"
    units, at = [], struct.unpack_from("<H", data, 6)[0]
    while at < len(data):
        (size,) = struct.unpack_from("<I", data, at)
        unit = data[at + 12:at + 12 + size]
        at += 12 + size
        units.append(unit[2:] if unit[:2] == b"\x12\x00" else unit)
    return units


def size_field(n: int) -> bytes:
    for length in range(1, 9):
        if n < (1 << (7 * length)) - 1:
            return ((1 << (7 * length)) | n).to_bytes(length, "big")
    raise ValueError(n)


def uint(n: int) -> bytes:
    return n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")


def el(ident: int, payload: bytes, unknown: bool = False) -> bytes:
    raw = ident.to_bytes((ident.bit_length() + 7) // 8, "big")
    return raw + (b"\x01\xff\xff\xff\xff\xff\xff\xff" if unknown else size_field(len(payload))) + payload


def make_webm(units: list[bytes], w: int, h: int, *, video_codec: str = "V_AV1", audio: bool = True, per_cluster: int = 5,
              unknown_sizes: bool = False, unknown_segment: bool = False, groups: bool = False, duration_frames: float | None = None, lacing: int = 0,
              replace: dict[int, bytes] | None = None, with_duration: bool = True) -> bytes:
    """A WebM file holding the units as the blocks of track 1 (and, with audio, an A_OPUS track of small blocks)."""
    replace = replace or {}
    header = el(0x1A45DFA3, el(0x4286, b"\x01") + el(0x42F7, b"\x01") + el(0x42F2, b"\x04") + el(0x42F3, b"\x08")
                + el(0x4282, b"webm") + el(0x4287, b"\x02") + el(0x4285, b"\x02"))
    frames = len(units) if duration_frames is None else duration_frames
    info = el(0x2AD7B1, uint(1_000_000)) + el(0x4D80, b"synthetic muxer")
    if with_duration:
        info += el(0x4489, struct.pack(">d", frames * PERIOD_NS / 1e6))
    video = el(0xAE, el(0xD7, b"\x01") + el(0x83, b"\x01") + el(0x86, video_codec.encode()) + el(0x23E383, uint(PERIOD_NS))
               + el(0x63A2, bytes([0x81, 0x00, 0x0C, 0x00])) + el(0xE0, el(0xB0, uint(w)) + el(0xBA, uint(h))))
    tracks = video
    if audio:
        tracks += el(0xAE, el(0xD7, b"\x02") + el(0x83, b"\x02") + el(0x86, b"A_OPUS") + el(0x56AA, uint(6_500_000))
                     + el(0xE1, el(0xB5, struct.pack(">d", 48000.0)) + el(0x9F, b"\x02")))
    clusters = []
    for first in range(0, len(units), per_cluster):
        base_ms = first * 33
        body = el(0xE7, uint(base_ms))
        for i in range(first, min(first + per_cluster, len(units))):
            rel = i * 33 - base_ms
            data = replace.get(i, units[i])
            key = 0x80 if units[i][:1] == b"\x0a" else 0   # a unit that opens with a sequence header is a key frame
            flags = key | (lacing << 1)
            block = b"\x81" + struct.pack(">h", rel) + bytes([flags]) + data
            body += el(0xA0, el(0xA1, block) + (b"" if key else el(0xFB, b"\x7f"))) if groups else el(0xA3, block)
            if audio:
                body += el(0xA3, b"\x82" + struct.pack(">h", rel) + b"\x80" + bytes(20 + i))
        clusters.append(el(0x1F43B675, body, unknown=unknown_sizes))
    segment = el(0x1549A966, info) + el(0x1654AE6B, tracks) + b"".join(clusters)
    return header + el(0x18538067, el(0xEC, bytes(30)) + segment, unknown=unknown_sizes or unknown_segment)


class WebmDecodeTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = shutil.which(os.environ.get("OOT2_CC", "clang"))
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        kit = build_game.find_dav1d()
        if kit is None:
            if os.environ.get("OOT2_REQUIRE_DAV1D"):
                raise RuntimeError("OOT2_REQUIRE_DAV1D is set but DAV1D_DIR holds no dav1d kit (tools/fetch_dav1d.py)")
            raise unittest.SkipTest("no dav1d kit: run tools/fetch_dav1d.py --dest DIR and set DAV1D_DIR to DIR")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-webm-", ignore_cleanup_errors=True)
        cls.out = Path(cls.tmp.name)
        extra = [] if os.name == "nt" else ["-pthread", "-lm"]
        for level in LEVELS:
            exe = cls.out / f"webm_decode{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", "-DOOT2_DAV1D=1",
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(kit / "include"),
                   *map(str, SOURCES), str(kit / "lib" / "libdav1d.a"), *extra, "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe
        cls.clip = {name: ivf_units(name) for name in ("clip_200x126", "clip_201x125", "clip_1920x1080")}

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_file(self, data: bytes, *args: object, level: str = "-O2", name: str = "movie.bin") -> subprocess.CompletedProcess:
        path = self.out / name
        path.write_bytes(data)
        return subprocess.run([str(self.exes[level]), str(path), *map(str, args)], capture_output=True, text=True, timeout=180)

    def result(self, done: subprocess.CompletedProcess) -> str:
        return [line for line in done.stdout.splitlines() if line.startswith("result")][-1]

    def small(self, **kw) -> bytes:
        """Twelve frames of the 200x126 clip (the clip three times: each round starts again from its key frame)."""
        return make_webm(self.clip["clip_200x126"] * 3, 200, 126, **kw)

    def test_a_clip_decodes_to_the_blocks_the_container_holds(self) -> None:
        for level in LEVELS:
            done = self.run_file(self.small(), level=level)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertEqual(self.result(done), "result OK frames=12 duration_check=yes")
            self.assertIn("codec=V_AV1 size=200x126", done.stdout)
            self.assertIn("av1c_profile=0 av1c_level=0 av1c_bits=8", done.stdout)
            self.assertIn("blocks=12 pictures=12 refused=0 wrong_size=0 keys=3", done.stdout)
            self.assertIn("other track=2 codec=A_OPUS blocks=12", done.stdout)
            self.assertIn("track number=2 type=2 codec=A_OPUS channels=2 rate=48000", done.stdout)

    def test_the_other_sizes_and_the_odd_one_decode(self) -> None:
        for name, w, h in (("clip_201x125", 201, 125), ("clip_1920x1080", 1920, 1080)):
            done = self.run_file(make_webm(self.clip[name], w, h))
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertIn(f"size={w}x{h}", done.stdout)
            self.assertEqual(self.result(done), f"result OK frames={len(self.clip[name])} duration_check=yes")

    def test_the_same_pictures_come_out_of_every_layout(self) -> None:
        def pic_hash(data: bytes, *args: object) -> str:
            done = self.run_file(data, *args)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            return next(f for f in done.stdout.split() if f.startswith("hash="))

        plain = pic_hash(self.small(audio=False))
        self.assertEqual(pic_hash(self.small(audio=True, per_cluster=3)), plain)
        self.assertEqual(pic_hash(self.small(unknown_sizes=True)), plain)        # clusters and segment of unknown size
        self.assertEqual(pic_hash(self.small(groups=True)), plain)               # blocks in block groups
        self.assertEqual(pic_hash(self.small(with_duration=False)), plain)       # no declared duration: not checked
        window = b"\xaa" * 777 + self.small() + b"\x55" * 4001                   # a file inside a larger one, read in place
        self.assertEqual(pic_hash(window, 777, len(window) - 777 - 4001), plain)

    def test_a_cut_copy_fails_as_truncated(self) -> None:
        data = self.small()
        for cut in (len(data) // 2, len(data) - 1, len(data) - 40):
            done = self.run_file(data[:cut])
            self.assertEqual(done.returncode, 1, done.stdout)
            self.assertIn("result FAIL truncated", self.result(done))
        # cut at the end of a cluster: every block is whole, the segment still declares more
        end_of_first = data.index(b"\x1f\x43\xb6\x75", data.index(b"\x1f\x43\xb6\x75") + 4)
        done = self.run_file(data[:end_of_first])
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL truncated: the segment declares its end", self.result(done))
        # the whole file is accepted
        self.assertEqual(self.run_file(data).returncode, 0)

    def test_a_cut_between_two_blocks_of_a_cluster_is_named_even_when_the_segment_has_no_size(self) -> None:
        units = self.clip["clip_200x126"] * 3
        data = make_webm(units, 200, 126, audio=False, per_cluster=12, unknown_segment=True)
        last = len(el(0xA3, b"\x81\x00\x00\x00" + units[-1]))
        done = self.run_file(data[:-last])
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL truncated: the cluster declares its end", self.result(done))
        self.assertEqual(self.run_file(data).returncode, 0)

    def test_a_cut_copy_of_a_file_of_unknown_size_is_short_by_its_duration(self) -> None:
        data = self.small(unknown_sizes=True)
        cut = data.index(b"\x1f\x43\xb6\x75", data.index(b"\x1f\x43\xb6\x75") + 4)   # after the first cluster
        done = self.run_file(data[:cut])
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL short or long: 5 frames, the container's duration implies 12.0", self.result(done))

    def test_a_duration_that_declares_more_frames_than_the_file_holds_fails(self) -> None:
        done = self.run_file(self.small(duration_frames=14))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL short or long: 12 frames, the container's duration implies 14.0", self.result(done))
        self.assertEqual(self.run_file(self.small(duration_frames=12.9)).returncode, 0)   # within one frame

    def test_blocks_in_groups_know_their_key_frames(self) -> None:
        done = self.run_file(self.small(groups=True))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("keys=3", done.stdout)

    def test_a_block_that_makes_no_picture_a_wrong_size_and_a_first_block_that_is_not_a_key_frame_are_named(self) -> None:
        units = self.clip["clip_200x126"] * 3
        header_only = units[0][:2 + units[0][1]]   # the sequence header OBU alone: taken, and shows nothing
        self.assertEqual(units[0][0], 0x0A)
        done = self.run_file(make_webm(units, 200, 126, replace={11: header_only}))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL 12 blocks made 11 pictures", self.result(done))
        done = self.run_file(make_webm(units, 200, 127))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL 12 pictures differ from the track's size", self.result(done))
        done = self.run_file(make_webm(units[1:], 200, 126))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL the first video block is not a key frame", self.result(done))

    def test_a_codec_that_is_not_av1_fails_by_its_name(self) -> None:
        done = self.run_file(self.small(video_codec="V_VP9"))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL video codec V_VP9 is not AV1", self.result(done))

    def test_a_damaged_unit_is_named(self) -> None:
        units = self.clip["clip_200x126"] * 3
        done = self.run_file(make_webm(units, 200, 126, replace={5: b"\xff" * 300}))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertRegex(self.result(done), r"^result FAIL the decoder refused \d+ of 12 blocks \(the first is block 5\)$")   # (the unit after it may fail too)

    def test_a_laced_video_block_and_other_damage_fail_by_name(self) -> None:
        done = self.run_file(self.small(lacing=2))
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("has lacing 2, which is not read", self.result(done))
        done = self.run_file(b"not a movie at all")
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("result FAIL open: not an EBML file", self.result(done))
        done = self.run_file(self.small(), 99999999, 10)
        self.assertEqual(done.returncode, 1, done.stdout)
        self.assertIn("past the end of the file", self.result(done))

    def test_the_driver_is_told_how_to_be_called(self) -> None:
        done = subprocess.run([str(self.exes["-O2"])], capture_output=True, text=True)
        self.assertEqual(done.returncode, 2)
        self.assertIn("usage: webm_decode FILE [OFFSET SIZE]", done.stderr)


if __name__ == "__main__":
    unittest.main()
