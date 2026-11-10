# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the Vulkan back end's format table and bindless tables (runtime/gpu/vk_format.c, bindless.c, vk_bindless.c,
vk_image.c; architecture D10, D13 and D19, the bindless tables), through tests/runtime/bindless_test.c, built like
tests/test_runtime_gpu.py's programs.

Two groups, by what they need:

- Without a device (continuous integration runs them): the format table over every NVN format from 0x01 to 0x9C, checked
  against what the names say (the Vulkan format's bytes per block derived from its own name, the component order, the
  sRGB, depth, X and block flags, the vertex format with three components kept) and against the sizes the front end gives
  (nvn_fmt_bpb); and the mapping of (texture pool, ID) to the global arrays with stand-ins for the host: one registration
  changes one entry, the same registration again none, an ID registered again under a pending submission takes another
  copy and leaves the pending one as it was, a released serial clears no entry, 4,096 sampler registrations with 300
  distinct states hold 300 host samplers, second views, two pools, a copy far behind the log, retirement, a pool remade at the
  same address getting views of its own; and what an image of
  ASTC decoded to RGBA8 may do (it never stands in its pool's extent, and a storage view or a texel copy of one is refused
  by name), over every ASTC format.
- On a device with every item of oot2/vk_required.h (the desktop; the device is the RTX 3060 Ti): images stand at their
  pool offsets in the pool's extent and the texel copies return what they wrote; a registration changes one entry; an ID
  registered again while a gated earlier submission reads the old image leaves that submission's result right; an entry
  read as 2D and as 2D array samples right both ways; two textures over one storage range see each other's GPU writes;
  4,096 sampler registrations with 300 distinct states hold 300 host samplers; a pool of 288 MiB finalized and remade at its
  address stands the texture registered again on an image of the new extent; a finalized pool's entries read the
  placeholder; the validation layer (with synchronization validation) reports nothing.

Without clang or the Vulkan SDK the program tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set; without a
usable device the device tests skip, unless OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_bindless.py -v
"""

from __future__ import annotations

import atexit
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import (EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines,
                              skip_on_partial_device, vulkan_or_skip)

sys.path.insert(0, str(ROOT / "tools"))
import build_game  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "bindless_test.c"
NVN_HEADER = ROOT / "runtime" / "include" / "oot2" / "nvn.h"
LEVEL = "-O2"
EXIT_NO_DEVICE = 3

_build: dict[str, object] = {}

# VKF_* flags of oot2/vk_format.h.
EXPAND_RGB, ALPHA_ONE, ASTC, DEPTH, STENCIL, SRGB, BLOCK = 1, 2, 4, 8, 16, 32, 64

SHADER = """#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 1) in;
#if defined(ARRAY)
layout(set = 0, binding = 0) uniform texture2DArray tex[];
#elif defined(UINT)
layout(set = 0, binding = 0) uniform utexture2D tex[];
#else
layout(set = 0, binding = 0) uniform texture2D tex[];
#endif
layout(set = 0, binding = 1) uniform sampler smp[];
layout(set = 1, binding = 9) buffer Out { uint o[]; };
layout(push_constant) uniform PC { uint handle; uint x; uint y; uint z; } pc;
void main() {
    uint t = pc.handle & 0xfffffu, s = pc.handle >> 20;
#if defined(ARRAY)
    vec4 c = texelFetch(sampler2DArray(tex[nonuniformEXT(t)], smp[nonuniformEXT(s)]), ivec3(pc.x, pc.y, pc.z), 0);
    uvec4 u = floatBitsToUint(c);
#elif defined(UINT)
    uvec4 u = texelFetch(usampler2D(tex[nonuniformEXT(t)], smp[nonuniformEXT(s)]), ivec2(pc.x, pc.y), 0);
#elif defined(SAMPLE)
    vec4 c = textureLod(sampler2D(tex[nonuniformEXT(t)], smp[nonuniformEXT(s)]), (vec2(pc.x, pc.y) + 0.5) / 256.0, 0.0);
    uvec4 u = floatBitsToUint(c);
#else
    vec4 c = texelFetch(sampler2D(tex[nonuniformEXT(t)], smp[nonuniformEXT(s)]), ivec2(pc.x, pc.y), 0);
    uvec4 u = floatBitsToUint(c);
#endif
    o[0] = u.x; o[1] = u.y; o[2] = u.z; o[3] = u.w;
}
"""


def program() -> Path:
    """The test program, built once per run."""
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-bindless-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _build["tmp"] = tmp
        _build["include"] = include
        _build["glslang"] = glslang
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def run(case: str, *args: str, env: dict[str, str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case, *args], capture_output=True, text=True, timeout=timeout, env=case_env(env))


def vk_names() -> dict[int, str]:
    """VkFormat number to its name (without VK_FORMAT_), from vulkan_core.h; the first name of a number."""
    program()
    text = (Path(_build["include"]) / "vulkan" / "vulkan_core.h").read_text(encoding="utf-8", errors="replace")  # type: ignore[arg-type]
    names: dict[int, str] = {}
    for m in re.finditer(r"^\s*VK_FORMAT_(\w+) = (\d+),", text, re.M):
        n = int(m.group(2))
        if n not in names and not m.group(1).endswith("_KHR") and not m.group(1).endswith("_EXT"):
            names[n] = m.group(1)
    return names


def nvn_names() -> dict[int, str]:
    out: dict[int, str] = {}
    for m in re.finditer(r"^\s*NVN_FORMAT_(\w+) = (0x[0-9A-Fa-f]+),", NVN_HEADER.read_text(encoding="utf-8"), re.M):
        if m.group(1) not in ("NONE", "LARGE"):
            out[int(m.group(2), 16)] = m.group(1)
    return out


def block_bytes(name: str) -> int:
    """Bytes per block of a Vulkan format, from its name alone."""
    if name == "D32_SFLOAT_S8_UINT":
        return 8
    if name in ("D24_UNORM_S8_UINT", "X8_D24_UNORM_PACK32"):
        return 4
    if name.startswith(("BC1", "BC4")):
        return 8
    if name.startswith(("BC", "ASTC")):
        return 16
    if name.endswith("PACK16"):
        return 2
    if name.endswith("PACK32"):
        return 4
    prefix = name.split("_")[0]
    return sum(int(n) for n in re.findall(r"[A-Z](\d+)", prefix)) // 8


def parse_rows(out: str) -> dict[int, dict[str, int | str]]:
    rows: dict[int, dict[str, int | str]] = {}
    for line in out.splitlines():
        m = re.match(r"fmt ([0-9a-f]{2}) (.*)$", line)
        if not m or m.group(2) == "none":
            continue
        row: dict[str, int | str] = {}
        for k, v in re.findall(r"(\w+)=(\S+)", m.group(2)):
            row[k] = int(v) if re.fullmatch(r"-?\d+", v) else v
        rows[int(m.group(1), 16)] = row
    return rows


class DecodedAstcTest(unittest.TestCase):
    """An ASTC texture on a device that does not sample ASTC is an image of RGBA8 texels the upload decodes (no device is
    needed to decide what such an image may do: vkf_decoded, vkf_extent_ok and vkf_decoded_stop of oot2/vk_format.h)."""

    @classmethod
    def setUpClass(cls) -> None:
        done = run("decoded")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "ok decoded" in done.stdout, done.stdout
        cls.out = done.stdout
        cls.astc = re.findall(r"^astc fmt (\w+) decoded=(\d) ok=(\d) storage=\[([^\]]*)\] copy=\[([^\]]*)\] \| sampled decoded=(\d) "
                              r"ok=(\d) storage=\[([^\]]*)\] copy=\[([^\]]*)\]$", cls.out, re.M)
        cls.plain = re.findall(r"^plain fmt (\w+) decoded=(\d) ok=(\d) storage=\[([^\]]*)\] copy=\[([^\]]*)\]$", cls.out, re.M)

    def test_every_astc_format_has_a_decoded_image_on_a_device_that_does_not_sample_it(self) -> None:
        self.assertEqual(len(self.astc), 28)   # 14 block sizes, linear and sRGB
        for fmt, dec, *_ in self.astc:
            self.assertEqual(dec, "1", f"ASTC format {fmt}")

    def test_a_decoded_image_never_stands_in_the_extent(self) -> None:
        self.assertEqual(len(self.astc), 28)
        for fmt, _, ok, *_ in self.astc:
            self.assertEqual(ok, "0", f"ASTC format {fmt}")

    def test_a_storage_view_and_a_texel_copy_of_a_decoded_image_are_refused_by_name(self) -> None:
        self.assertEqual(len(self.astc), 28)
        for fmt, _, _, storage, copy, *_ in self.astc:
            self.assertEqual((storage, copy), ("a storage view", "a texel copy"), f"ASTC format {fmt}")

    def test_an_astc_image_the_device_samples_as_blocks_stands_in_the_extent_and_serves_both(self) -> None:
        for fmt, *_, sdec, sok, sstorage, scopy in self.astc:
            self.assertEqual((sdec, sok, sstorage, scopy), ("0", "1", "none", "none"), fmt)

    def test_every_other_image_stands_in_the_extent_and_serves_both(self) -> None:
        self.assertGreater(len(self.plain), 100)
        for fmt, dec, ok, storage, copy in self.plain:
            self.assertEqual((dec, ok, storage, copy), ("0", "1", "none", "none"), fmt)
        self.assertIn("null row decoded=0 ok=1", self.out)


class FormatTableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        done = run("formats")
        assert done.returncode == 0, done.stdout + done.stderr
        assert "ok formats" in done.stdout, done.stdout
        cls.out = done.stdout
        cls.rows = parse_rows(done.stdout)
        cls.vk = vk_names()
        cls.nvn = nvn_names()

    def test_every_nvn_format_from_0x01_to_0x9c_has_a_row_and_nothing_else_does(self) -> None:
        self.assertEqual(sorted(self.rows), list(range(0x01, 0x9D)))
        self.assertEqual(sorted(self.nvn), list(range(0x01, 0x9D)))
        self.assertIn("fmt 00 none", self.out)
        self.assertIn("fmt 9d none", self.out)

    def test_a_format_has_a_vulkan_format_or_a_reason_and_the_ones_without_are_the_integer_to_float_formats(self) -> None:
        without = {f for f, r in self.rows.items() if r["vk"] == 0}
        self.assertEqual(without, set(range(0x51, 0x69)) | {0x6B, 0x6C})
        for f in without:
            self.assertEqual(self.rows[f]["reason"], "yes", hex(f))
            self.assertIn("I2F", self.nvn[f])
        for f, r in self.rows.items():
            if r["vk"]:
                self.assertIn(r["vk"], self.vk, f"NVN {self.nvn[f]}: no Vulkan format numbered {r['vk']}")

    def test_bytes_per_block_equal_the_front_ends_sizes_and_the_vulkan_formats_own(self) -> None:
        for f, r in self.rows.items():
            if not r["vk"]:
                continue
            name = f"NVN {self.nvn[f]} (0x{f:02x}) as {self.vk[r['vk']]}"
            self.assertEqual(r["bpb"], block_bytes(self.vk[r["vk"]]), name)
            self.assertEqual((r["bw"], r["bh"]), (r["nvn_bw"], r["nvn_bh"]), name)
            if r["flags"] & EXPAND_RGB:
                self.assertEqual(r["bpb"] * 3, r["nvn_bpb"] * 4, name)   # three components stored, four in the image
            else:
                self.assertEqual(r["bpb"], r["nvn_bpb"], name)

    def test_vertex_formats_keep_the_components_of_the_nvn_format(self) -> None:
        for f, r in self.rows.items():
            if not r["vertex"]:
                continue
            name = f"NVN {self.nvn[f]} (0x{f:02x})"
            self.assertEqual(block_bytes(self.vk[r["vertex"]]), r["nvn_bpb"], name)
            if not r["flags"] & EXPAND_RGB:
                self.assertEqual(r["vertex"], r["vk"], name)
            else:
                self.assertTrue(re.match(r"R(8|16|32)G(8|16|32)B(8|16|32)_", self.vk[r["vertex"]]), name)
        # Every plain R, RG, RGB and RGBA format of 8, 16 and 32 bits is a vertex format.
        for f, n in self.nvn.items():
            if re.fullmatch(r"(R|RG|RGB|RGBA)(8|16|32)(SN|UI|I|F)?", n) and not re.fullmatch(r"(R|RG|RGB|RGBA)32", n):
                self.assertTrue(self.rows[f]["vertex"], n)

    def test_regular_formats_are_the_vulkan_format_their_name_says(self) -> None:
        suffix = {"": "UNORM", "SN": "SNORM", "UI": "UINT", "I": "SINT", "F": "SFLOAT"}
        for f, n in self.nvn.items():
            m = re.fullmatch(r"(R|RG|RGB|RGBA|RGBX)(8|16|32)(SN|UI|I|F)?", n)
            if not m or (m.group(2) == "32" and not m.group(3)):
                continue
            comp, bits, suf = m.group(1), m.group(2), m.group(3) or ""
            channels = {"R": "R", "RG": "RG", "RGB": "RGBA", "RGBA": "RGBA", "RGBX": "RGBA"}[comp]
            want = "".join(f"{c}{bits}" for c in channels) + "_" + suffix[suf]
            self.assertEqual(self.vk[self.rows[f]["vk"]], want, n)
        for f, want in {0x99: "B8G8R8A8_UNORM", 0x9A: "B8G8R8A8_UNORM", 0x9B: "B8G8R8A8_SRGB", 0x9C: "B8G8R8A8_SRGB",
                        0x37: "R8G8B8A8_SRGB", 0x38: "R8G8B8A8_SRGB", 0x32: "D16_UNORM", 0x34: "D32_SFLOAT",
                        0x33: "X8_D24_UNORM_PACK32", 0x35: "D24_UNORM_S8_UINT", 0x36: "D32_SFLOAT_S8_UINT",
                        0x31: "S8_UINT", 0x3D: "A2B10G10R10_UNORM_PACK32", 0x3F: "B10G11R11_UFLOAT_PACK32",
                        0x40: "E5B9G9R9_UFLOAT_PACK32", 0x4D: "BC7_UNORM_BLOCK", 0x4E: "BC7_SRGB_BLOCK",
                        0x44: "BC3_UNORM_BLOCK", 0x49: "BC4_UNORM_BLOCK", 0x4B: "BC5_UNORM_BLOCK",
                        0x4F: "BC6H_SFLOAT_BLOCK", 0x50: "BC6H_UFLOAT_BLOCK"}.items():
            self.assertEqual(self.vk[self.rows[f]["vk"]], want, hex(f))

    def test_flags_follow_the_names(self) -> None:
        for f, n in self.nvn.items():
            r = self.rows[f]
            if not r["vk"]:
                continue
            fl = r["flags"]
            self.assertEqual(bool(fl & DEPTH), "DEPTH" in n, n)
            self.assertEqual(bool(fl & STENCIL), "STENCIL" in n, n)
            self.assertEqual(bool(fl & SRGB), "SRGB" in n, n)
            self.assertEqual(bool(fl & ASTC), "ASTC" in n, n)
            self.assertEqual(bool(fl & BLOCK), any(k in n for k in ("ASTC", "DXT", "RGTC", "BPTC")), n)
            self.assertEqual(bool(fl & ALPHA_ONE), "RGBX" in n or "BGRX" in n or n in ("RGB5", "BGR5"), n)
            self.assertEqual(bool(fl & EXPAND_RGB), bool(re.fullmatch(r"RGB(8|16|32)(SN|UI|I|F)?", n)), n)

    def test_astc_rows_carry_their_block_size_and_sixteen_bytes(self) -> None:
        sizes = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10),
                 (12, 10), (12, 12)]
        for i, (w, h) in enumerate(sizes):
            for base in (0x79, 0x87):
                r = self.rows[base + i]
                self.assertEqual((r["bw"], r["bh"], r["bpb"]), (w, h, 16), hex(base + i))
                self.assertIn(f"ASTC_{w}x{h}_", self.vk[r["vk"]])

    def test_packed_formats_name_their_components_through_a_view_swizzle(self) -> None:
        # NVN names a packed format's components from the least significant bit: RGB5A1 holds R in bits 0 to 4, so the
        # Vulkan format that has A in bit 15 and R in bits 14 to 10 needs R and B exchanged. Selectors: R 2, G 3, B 4, A 5.
        def sw(f: int, key: str) -> str:
            return str(self.rows[f][key]).zfill(4)

        self.assertEqual(sw(0x3B, "swz"), "4325")
        self.assertEqual(sw(0x3A, "swz"), "4325")
        self.assertEqual(sw(0x39, "swz"), "0000")
        self.assertEqual(sw(0x39, "altswz"), "5234")   # B4G4R4A4: A, R, G, B hold NVN's R, G, B, A
        self.assertEqual(self.vk[self.rows[0x39]["alt"]], "B4G4R4A4_UNORM_PACK16")
        self.assertEqual(sw(0x3C, "swz"), "0000")
        self.assertEqual(sw(0x3C, "altswz"), "4325")
        for f in (0x39, 0x3A, 0x3B, 0x3C, 0x95, 0x96, 0x97, 0x98):
            self.assertEqual(self.rows[f]["bpb"], 2, hex(f))

    def test_the_classes_the_images_share_views_in(self) -> None:
        self.assertIn("classes ok", self.out)


class TablesTest(unittest.TestCase):
    def test_the_mapping_reuse_samplers_second_views_and_retirement(self) -> None:
        done = run("tables")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr), [])
        for line in (
            "a registration changed one entry: one descriptor write, no full sync",
            "registering the same view again (a new serial, the same storage and description) changed no entry",
            "an ID registered again under a pending submission took another copy; the pending copy was not written",
            "the view an entry dropped stayed while the submission that could read it was pending, and went after",
            "texture_release left the entry at its ID in place",
            "4096 sampler registrations with 300 distinct states hold 300 host samplers",
            "registering the entry again released its second slot",
            "a freed second slot was not taken again while a copy still showed its old view",
            "a submission that sets the other pool sees its entries, and not the first pool's",
            "a copy far behind the log is right",
            "an entry with no host view shows the placeholder",
            "view types and keys",
            "ok tables",
        ):
            self.assertIn(line, done.stdout)
        self.assertRegex(done.stdout, r"slot \d+ holds the second, in every copy")
        self.assertRegex(done.stdout, r"close freed \d+ views, 300 samplers and \d+ copies")

    def test_a_view_the_host_makes_only_when_the_replay_binds_the_texture(self) -> None:
        done = run("demand")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr), [])
        for line in (
            "registering textures the host makes no view of took nothing: 0 views, 0 demands, placeholders in the first copy",
            "the first bind of ID 5 asked the host for the view and told it which submission reads it; the copy shows it",
            "another copy took the view through the log; the IDs nothing bound still show the placeholder",
            "registering another texture at the ID put the placeholder back until the next bind",
            "ok demand",
        ):
            self.assertIn(line, done.stdout)
        self.assertRegex(done.stdout, r"a second view type of a demanded entry was demanded too and got slot \d+")

    def test_a_pool_remade_at_the_same_address_does_not_get_the_finalized_pools_views(self) -> None:
        done = run("remade")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual(fatal_lines(done.stderr), [])
        for line in (
            "before: ID 5 stands on a view and a second view of the pool that goes; IDs 6 and 7 share a view",
            "ID 5 registered again over the same storage and description took a view of its own; "
            "the views of the pool that stays are still shared",
            "its second view type was made again, in another slot than the finalized pool's",
            "the two views of the finalized pool were freed after the submission that could read them",
            "ok remade",
        ):
            self.assertIn(line, done.stdout)

    def test_a_texture_past_the_games_part_of_the_array_is_a_named_fatal(self) -> None:
        done = run("id_fatal")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertNotIn("not reached", done.stdout)
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr)
        self.assertIn("bindless tables: a texture registered at ID 1008 of pool 0x100000", lines[0])


def require_gpu() -> bool:
    return bool(os.environ.get("OOT2_REQUIRE_GPU"))


class DeviceTest(unittest.TestCase):
    usable = 0

    @classmethod
    def setUpClass(cls) -> None:
        done = run("usable")
        skip_on_partial_device(done)
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        assert done.returncode == 0, done.stdout + done.stderr
        cls.usable = int(re.search(r"^usable (\d+)$", done.stdout, re.M).group(1))
        if not cls.usable:
            why = "no device here has every item the renderer requires"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
            raise unittest.SkipTest(why)
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-bindless-spv-", ignore_cleanup_errors=True)
        atexit.register(cls.tmp.cleanup)
        src = Path(cls.tmp.name) / "read.comp"
        src.write_text(SHADER, encoding="utf-8")
        cls.spv = {}
        for name, define in (("2d", None), ("array", "ARRAY"), ("uint", "UINT"), ("sample", "SAMPLE")):
            out = Path(cls.tmp.name) / f"read_{name}.spv"
            cmd = [_build["glslang"], *build_game.GLSLANG_FLAGS]  # type: ignore[list-item]
            if define:
                cmd += [f"-D{define}"]
            cmd += ["-o", str(out), str(src)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            assert done.returncode == 0, done.stdout + done.stderr
            cls.spv[name] = str(out)

    def test_images_registrations_second_views_aliasing_samplers_and_a_finalized_pool_on_the_device(self) -> None:
        done = run("device", self.spv["2d"], self.spv["array"], self.spv["uint"], self.spv["sample"],
                   env={"OOT2_VK_VALIDATION": "sync"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertEqual(fatal_lines(done.stderr), [])
        for line in (
            "two 256 x 256 textures stand at their pool offsets in the pool's extent",
            "read_texels returned what write_texels wrote, over a region and over the whole texture",
            "ID 5 read as 2D: texel (3, 4) is (3, 4, 7, 255)",
            "a sample through the sampler array's entry 3 (nearest, clamp) gives the same texel",
            "ID 5 registered again under a held submission: the held one read the old image (3, 4, 7), the next the new (4, 5, 9)",
            "one entry sampled right as 2D (its own view) and as 2D array (a second slot)",
            "an array texture sampled as array (layer 1 is (5, 6, 21)) and as 2D (layer 0 is (5, 6, 20))",
            "a 288 MiB pool finalized and remade at its address: the texture registered again stands on an image of the "
            "new extent and reads (3, 4, 41)",
            "after pool_fini the entries over its storage read the placeholder",
            "ok device",
        ):
            self.assertIn(line, done.stdout)
        self.assertRegex(done.stdout, r"a registration: 1 image writes, 0 sampler writes, 0 full syncs")
        self.assertRegex(done.stdout, r"read as R32_UINT over the same storage: 0x4030201, one image")
        self.assertRegex(done.stdout, r"smaller texture is a second image aliasing the first: \d+ images")
        self.assertRegex(done.stdout, r"300 host samplers live, and entry 100 samples")
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
