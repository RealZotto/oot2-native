# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of HMMA on VK_KHR_cooperative_matrix: the translator's native form (CodeGen/Spirv/CoopHmma.cs), the pipeline
builder's turn of it into the helper (spv_hmma_fallback, runtime/gpu/spirv.c) and the start-up probe
(runtime/gpu/vk_hmma.c, oot2/vk_hmma.h).

Three groups, by what they need:

- The translator (py -3 tools/build_xlate.py): one program's native form and helper form, the memory operands the Vulkan
  memory model gives its workgroup accesses, its components against the runtime's copy of the order, the option's values,
  and the helper kept for a workgroup whose width is not a whole number of subgroups.
- The translator and the pipeline builder's test program (clang and the Vulkan SDK, as tests/test_runtime_vk_pipeline.py):
  the native module turned into the helper, validated by spirv-val, and what a module without matrices or without the
  helper gets.
- A GPU with every item and VK_KHR_cooperative_matrix, and the harness of the HMMA cost study (built here by
  tools/build_vkdraw.py --harness vkhmmacost): one chain of HMMA run in both forms and compared, one HMMA against the
  CPU model of that study (tools/hmmacost/hmma_cost.py), the start-up probe's decision and its third check against a
  fault in its read-back, and a native module built through the pipeline builder in both forms. The probe's own gate
  (vk_hmma.c, gate()) says when a device cannot run the native form; the cases that need the native form skip on it.

How the forms compare (the chain tests). On inputs whose every partial sum f16 holds exactly (A in {-1, 0, 1}, B in
{-1, 0, 1} / 256, C k / 256), every form gives the exact sums, so the native and the helper results must be equal bit for
bit, and equal to the CPU model, which is how a wrong component order shows. On general inputs they are not equal. The
helper rounds D to f16 at every call (its pack, as the console's register holds it), as the CPU model does by default;
driver 617.14 folds that pack and the next call's unpack away between two inlined calls, so there the helper's results
keep an f32 accumulator across a chain and equal the model's f32-chain variant word for word (the HMMA cost study): a
property of that driver, not of the helper. The tensor cores round every HMMA's D to f16 after their own internal
accumulation (that study: long f16 chains on them stray from a round-to-nearest model). The test bounds the difference
by 2^-6 of the largest magnitude, the study's own tolerance for the helper's cooperative form, far below what a wrong
order gives (errors of the magnitudes themselves): that bound is the portable check. The f32-chain equality is checked
where the helper's results show the fold, and skips where they equal the per-call model (a driver that does not fold).

The programs are assembled from instruction words (the HMMA cost study's assembler), never read from game files. The
translator tests skip when it has not been built, unless OOT2_REQUIRE_XLATE is set; the program tests without clang or
the SDK unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set; the device tests without a usable GPU unless
OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_xlate_hmma_coop.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "hmmacost"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import hmma_cost as hc  # noqa: E402
import pool  # noqa: E402
import validate  # noqa: E402
from test_runtime_gpu import fatal_lines  # noqa: E402
from test_runtime_vk_pipeline import EXIT_NO_DEVICE, cp, require_gpu, run, scratch  # noqa: E402
from test_xlate_host import EXE, SKIP_NO_EXE, SPIRV_VAL, Harness  # noqa: E402

COOP_HEADER = ROOT / "xlate" / "third_party" / "ryujinx" / "src" / "Ryujinx.Graphics.Shader" / "CodeGen" / "Spirv" / "CoopHmma.cs"
RUNTIME_HEADER = ROOT / "runtime" / "include" / "oot2" / "vk_hmma.h"
RUNTIME_PROBE = ROOT / "runtime" / "gpu" / "vk_hmma.c"

NATIVE, HELPER = "dmsh_hmma16816_coop", "dmsh_hmma16816_f16"
CAP_FLOAT16, CAP_VULKAN_MEMORY_MODEL, CAP_DEVICE_SCOPE, CAP_COOPERATIVE_MATRIX = 9, 5345, 5346, 6022
OP_NAME, OP_MEMORY_MODEL, OP_CAPABILITY, OP_CONSTANT, OP_TYPE_POINTER, OP_FUNCTION, OP_FUNCTION_END = 5, 14, 17, 43, 32, 54, 56
OP_FUNCTION_CALL, OP_VARIABLE, OP_LOAD, OP_STORE, OP_ACCESS_CHAIN, OP_COMPOSITE_INSERT = 57, 59, 61, 62, 65, 82
OP_TYPE_COOP, OP_COOP_MUL_ADD = 4456, 4459
SC_WORKGROUP, MODEL_GLSL450, MODEL_VULKAN = 4, 1, 3
NON_PRIVATE, MAKE_AVAILABLE, MAKE_VISIBLE, ALIGNED = 0x20, 0x8, 0x10, 0x2


# ---- reading modules ------------------------------------------------------------------------------------

def instructions(module: bytes) -> list[tuple[int, list[int]]]:
    words = struct.unpack(f"<{len(module) // 4}I", module)
    out, i = [], 5
    while i < len(words):
        count, op = words[i] >> 16, words[i] & 0xFFFF
        assert count, f"a zero word count at word {i}"
        out.append((op, list(words[i + 1:i + count])))
        i += count
    return out


def text(operands: list[int]) -> str:
    return struct.pack(f"<{len(operands)}I", *operands).split(b"\0")[0].decode()


class Module:
    """What the tests read of one module: capabilities, memory model, names, the calls of each named function, the
    instructions inside each function, and the workgroup accesses with their memory operands."""

    def __init__(self, data: bytes):
        self.data = data
        self.ins = instructions(data)
        self.caps = {o[0] for op, o in self.ins if op == OP_CAPABILITY}
        self.model = next(o[1] for op, o in self.ins if op == OP_MEMORY_MODEL)
        self.names = {o[0]: text(o[1:]) for op, o in self.ins if op == OP_NAME}
        self.ids = {v: k for k, v in self.names.items()}
        self.constants = {o[1]: o[2] for op, o in self.ins if op == OP_CONSTANT and len(o) == 3}
        workgroup_types = {o[0] for op, o in self.ins if op == OP_TYPE_POINTER and o[1] == SC_WORKGROUP}
        self.workgroup = {o[1] for op, o in self.ins if op in (OP_VARIABLE, OP_ACCESS_CHAIN) and o[0] in workgroup_types}
        self.bodies: dict[int, list[tuple[int, list[int]]]] = {}
        current = None
        for op, o in self.ins:
            if op == OP_FUNCTION:
                current = o[1]
                self.bodies[current] = []
            elif op == OP_FUNCTION_END:
                current = None
            elif current is not None:
                self.bodies[current].append((op, o))

    def calls(self, name: str) -> int:
        fid = self.ids.get(name)
        return sum(1 for op, o in self.ins if op == OP_FUNCTION_CALL and o[2] == fid) if fid else 0

    def body(self, name: str) -> list[tuple[int, list[int]]]:
        return self.bodies.get(self.ids.get(name), [])

    def workgroup_accesses(self) -> list[tuple[int, list[int]]]:
        """(opcode, memory operands) of every load and store through a pointer to workgroup memory."""
        out = []
        for op, o in self.ins:
            if op == OP_LOAD and o[2] in self.workgroup:
                out.append((op, o[3:]))
            elif op == OP_STORE and o[0] in self.workgroup:
                out.append((op, o[2:]))
        return out


def orders(source: Path, pattern: str) -> list[list[int]]:
    """The three orders a source writes, by a pattern whose group holds one comma-separated list."""
    found = re.findall(pattern, source.read_text(encoding="utf-8"))
    return [[int(x) for x in f.split(",")] for f in found]


def gate_reasons() -> list[str]:
    """What the probe's gate says of a device that cannot run the native form (the strings of gate() in vk_hmma.c): the
    probe's own verdict on the device, on which a case that needs the native form skips, where a refusal by the order or the
    result checks is a failure."""
    text = RUNTIME_PROBE.read_text(encoding="utf-8")
    body = text[text.index("static const char *gate(void)"):]
    return re.findall(r'"(the device[^"]*)"', body[:body.index("\n}\n")])


def verdict(out: str) -> tuple[int, str]:
    """The probe's decision as the pipeline builder's test program prints it: "hmma native N[: reason]"."""
    m = re.search(r"^hmma native (\d)(?:: (.*))?$", out, re.M)
    if not m:
        raise AssertionError(f"no decision in {out[-500:]!r}")
    return int(m.group(1)), (m.group(2) or "").strip()


def skip_on_the_gate(test: unittest.TestCase, native: int, reason: str) -> None:
    """Skip a case that needs the native form when the probe's gate says the device cannot run it."""
    if native == 0 and reason in gate_reasons():
        why = f"the probe's gate keeps the helper on this device: {reason}"
        if require_gpu():
            raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
        test.skipTest(why)


# ---- the translator --------------------------------------------------------------------------------------

def translate(work: Path, key: str, p: hc.Program, hmma: str | None = None,
              local: tuple[int, int, int] = (hc.LOCAL, 1, 1)) -> pool.Result:
    """One program through the translator: its module and description in work, as key.spv and key.res.json."""
    code = work / f"{key}.bin"
    code.write_bytes(p.binary())
    job = {"key": key, "stage": "cs", "arch": "sm86", "code": str(code), "c1": "", "nvn2_cs": [0, 0, 0, 16, 0, 0, 0],
           "local": list(local), "shared": p.shared}
    if hmma is not None:
        job["options"] = {"hmma": hmma}
    (r,), _ = pool.run_pool([str(EXE)], [(job, work)], 1, work / "logs", timeout=600)
    return r


def need_xlate() -> None:
    if EXE is None:
        if os.environ.get("OOT2_REQUIRE_XLATE"):
            raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
        raise unittest.SkipTest(SKIP_NO_EXE)


class NativeFormTest(Harness):
    COUNT = 8   # two HMMA on each of the four accumulator chains

    def form(self, hmma: str | None) -> Module:
        r = translate(self.dir, f"form{hmma or ''}", hc.hmma_program(self.COUNT, False, True), hmma)
        self.assertEqual((r.status, r.message), ("ok", ""))
        path = self.dir / f"form{hmma or ''}.spv"
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, path.stem, path)
            self.assertTrue(v.ok, v.message)
        return Module(path.read_bytes())

    def test_the_default_runs_each_hmma_on_cooperative_matrices_and_keeps_the_helper(self):
        m = self.form(None)
        self.assertLessEqual({CAP_FLOAT16, CAP_COOPERATIVE_MATRIX, CAP_VULKAN_MEMORY_MODEL, CAP_DEVICE_SCOPE}, m.caps)
        self.assertEqual(m.model, MODEL_VULKAN)
        self.assertEqual(m.calls(NATIVE), self.COUNT)
        self.assertEqual(m.calls(HELPER), 0)
        self.assertIn(HELPER, m.ids)                                         # uncalled, kept for the fallback
        self.assertEqual([op for op, _ in m.body(NATIVE)].count(OP_COOP_MUL_ADD), 1)
        self.assertEqual(sum(1 for op, _ in m.ins if op == OP_TYPE_COOP), 3)  # A, B and the accumulator

    def test_the_helper_option_writes_the_form_every_module_had_before(self):
        m = self.form("helper")
        self.assertNotIn(CAP_COOPERATIVE_MATRIX, m.caps)
        self.assertNotIn(CAP_FLOAT16, m.caps)
        self.assertNotIn(CAP_VULKAN_MEMORY_MODEL, m.caps)
        self.assertEqual(m.model, MODEL_GLSL450)
        self.assertNotIn(NATIVE, m.ids)
        self.assertEqual(m.calls(HELPER), self.COUNT)
        self.assertGreater(len(m.workgroup_accesses()), 0)
        self.assertEqual([ops for _, ops in m.workgroup_accesses() if ops], [])   # no memory operands under GLSL450

    def test_workgroup_accesses_are_non_private_at_workgroup_scope_and_nothing_else_changes(self):
        m = self.form(None)
        accesses = m.workgroup_accesses()
        self.assertGreater(len(accesses), 0, "the helper's scratch must be in the module")
        for op, ops in accesses:
            make = MAKE_VISIBLE if op == OP_LOAD else MAKE_AVAILABLE
            self.assertTrue(ops, f"a workgroup {'load' if op == OP_LOAD else 'store'} without memory operands")
            self.assertEqual(ops[0] & ~ALIGNED, NON_PRIVATE | make, (op, ops))
            self.assertEqual(m.constants.get(ops[-1]), 2, "the scope is Workgroup")
        others = 0   # every other load and store (storage buffers, function memory) stays private, as GLSL450 had it
        for op, o in m.ins:
            if op == OP_LOAD and o[2] not in m.workgroup:
                others += 1
                self.assertFalse(len(o) > 3 and o[3] & NON_PRIVATE, o)
            elif op == OP_STORE and o[0] not in m.workgroup:
                others += 1
                self.assertFalse(len(o) > 2 and o[2] & NON_PRIVATE, o)
        self.assertGreater(others, 0)

    def test_the_components_follow_the_order_the_runtime_keeps(self):
        m = self.form(None)
        inserts = [o for op, o in m.body(NATIVE) if op == OP_COMPOSITE_INSERT]
        self.assertEqual(len(inserts), 8 + 4 + 4)
        written = [o[4] for o in inserts]                       # result type, result, object, composite, index
        translator = orders(COOP_HEADER, r"Order[ABC] = \[([0-9, ]+)\]")
        runtime = orders(RUNTIME_HEADER, r"#define VKH_ORDER_[ABC] \{([0-9, ]+)\}")
        self.assertEqual(len(translator), 3)
        self.assertEqual(translator, runtime)
        self.assertEqual(written, translator[0] + translator[1] + translator[2])

    def test_a_workgroup_whose_width_is_not_a_whole_number_of_subgroups_keeps_the_helper(self):
        """A subgroup matrix needs every lane of its subgroup, and Vulkan requires the width of a module that holds one to be
        a multiple of the subgroup size (VUID-VkPipelineShaderStageCreateInfo-module-08987). 48 x 1 x 1 leaves its second
        subgroup partial; 16 x 2 x 1 and 8 x 4 x 4 fill theirs, but not by their width: all three keep the helper. 32 x 4 x 1
        runs natively. The game's 71 tensor programs are 32, 128 or 256 lanes wide (the compute build time census)."""
        for local, native in (((48, 1, 1), False), ((16, 2, 1), False), ((8, 4, 4), False), ((32, 4, 1), True)):
            with self.subTest(local=local):
                key = "local" + "x".join(map(str, local))
                r = translate(self.dir, key, hc.hmma_program(self.COUNT, False, True), local=local)
                self.assertEqual((r.status, r.message), ("ok", ""))
                path = self.dir / f"{key}.spv"
                if SPIRV_VAL:
                    v = validate.validate_one(SPIRV_VAL, path.stem, path)
                    self.assertTrue(v.ok, v.message)
                m = Module(path.read_bytes())
                self.assertEqual((m.calls(NATIVE), m.calls(HELPER)), (self.COUNT, 0) if native else (0, self.COUNT))
                self.assertEqual(CAP_COOPERATIVE_MATRIX in m.caps, native)
                self.assertEqual(m.model, MODEL_VULKAN if native else MODEL_GLSL450)

    def test_an_unknown_value_of_the_option_is_a_bad_job(self):
        r = translate(self.dir, "bad", hc.hmma_program(4, False, True), "native")
        self.assertEqual(r.status, "fail")
        self.assertIn("options.hmma", r.message)

    def test_the_hello_names_the_default(self):
        (reply,) = self.exchange('{"op":"hello"}')
        status, _, _, rest = pool.parse_reply(reply)
        self.assertEqual(status, "ok")
        self.assertEqual(json.loads(rest)["defaults"]["hmma"], "coop")


# ---- the pipeline builder's turn into the helper ------------------------------------------------------------

class FallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_xlate()
        cls.dir = scratch()
        for key, hmma in (("native", None), ("helper", "helper")):
            r = translate(cls.dir, key, hc.hmma_program(8, False, True), hmma)
            assert r.status == "ok", r.message

    def turn(self, name: str, data: bytes | None = None) -> tuple[list[str], Path]:
        src = self.dir / f"{name}.spv"
        if data is not None:
            src = self.dir / f"{name}-in.spv"
            src.write_bytes(data)
        out = self.dir / f"{name}-out.spv"
        done = run("spvhmma", str(src), str(out))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines(), out

    def test_the_native_module_becomes_the_helper_and_validates(self):
        lines, out = self.turn("native")
        self.assertTrue(lines[0].startswith("helper: "), lines)
        m = Module(out.read_bytes())
        self.assertEqual(m.calls(HELPER), 8)
        self.assertNotIn(NATIVE, m.ids)
        self.assertEqual(sum(1 for op, _ in m.ins if op in (OP_TYPE_COOP, OP_COOP_MUL_ADD)), 0)
        self.assertNotIn(CAP_COOPERATIVE_MATRIX, m.caps)
        self.assertNotIn(CAP_FLOAT16, m.caps)
        self.assertEqual(m.model, MODEL_VULKAN)                 # the memory model stays: an item of the required list
        self.assertNotIn(b"SPV_KHR_cooperative_matrix", out.read_bytes())
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, "fallback", out)
            self.assertTrue(v.ok, v.message)

    def test_a_module_without_matrices_is_left_as_it_is(self):
        self.assertEqual(self.turn("helper")[0], ["unchanged"])

    def test_a_module_with_matrices_but_no_helper_is_refused_by_name(self):
        data = (self.dir / "native.spv").read_bytes().replace(HELPER.encode(), b"dmsh_hmma16816_zzz")   # same length
        lines, _ = self.turn("nohelper", data)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("refused: a module with cooperative matrices but without both functions"),
                        lines)

    def test_sixteen_bit_floats_are_optional(self):
        done = run("optional")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("optional shader_float16", done.stdout.splitlines())
        self.assertIn("present: 16-bit floats enabled 1", done.stdout.splitlines())
        self.assertIn("absent: 16-bit floats enabled 0", done.stdout.splitlines())


# ---- the device -------------------------------------------------------------------------------------------

def exact_input(seed: int) -> bytes:
    """hmma_cost's lane input with values whose partial sums f16 holds exactly: A in {-1, 0, 1}, B in {-1, 0, 1} / 256,
    the f16 accumulators k / 256 with |k| <= 64 (every partial sum of 8 HMMA a chain is a multiple of 1/256 below 1)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(-1, 2, (hc.LOCAL, 32)).astype(np.float16)
    b = (rng.integers(-1, 2, (hc.LOCAL, 16)) / 256.0).astype(np.float16)
    c = (rng.integers(-64, 65, (hc.LOCAL, 16)) / 256.0).astype(np.float16)
    words = np.zeros((hc.LOCAL, hc.IN_WORDS), np.uint32)
    for k, arr in ((0, a), (16, b), (24, c)):
        h = arr.view(np.uint16).reshape(hc.LOCAL, -1, 2).astype(np.uint32)
        words[:, k:k + h.shape[1]] = h[:, :, 0] | (h[:, :, 1] << 16)
    return words.tobytes()


class DeviceTest(unittest.TestCase):
    GROUPS = 2

    @classmethod
    def setUpClass(cls):
        need_xlate()
        cls.dir = scratch()
        built = subprocess.run([sys.executable, str(ROOT / "tools" / "build_vkdraw.py"), "--harness", "vkhmmacost",
                                "--out", str(cls.dir / "harness")], capture_output=True, text=True)
        if built.returncode:
            if os.environ.get("OOT2_REQUIRE_CLANG") or os.environ.get("OOT2_REQUIRE_VULKAN") or require_gpu():
                raise RuntimeError(f"the HMMA cost harness did not build: {built.stdout}{built.stderr}")
            raise unittest.SkipTest(f"the HMMA cost harness did not build (clang or the Vulkan SDK): {built.stderr[-300:]}")
        cls.harness = next((cls.dir / "harness").glob("vkhmmacost*"))
        info = subprocess.run([str(cls.harness), "info"], capture_output=True, text=True, env=hc.vk_env())
        shapes = json.loads(info.stdout).get("shapes", []) if info.returncode == 0 and info.stdout.strip() else []
        if not any(s["m"] == 16 and s["n"] == 8 and s["k"] == 16 and s["a"] == s["b"] == s["c"] == s["result"] == "f16"
                   and s["scope"] == 3 for s in shapes):
            why = f"no device here lists a 16x8x16 f16 subgroup cooperative matrix ({(info.stdout or info.stderr)[-200:]})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
            raise unittest.SkipTest(why)

    def run_form(self, p: hc.Program, hmma: str | None, inp: bytes, tag: str) -> np.ndarray:
        """The program translated in one form and run once on GROUPS workgroups under validation; workgroup 0's words."""
        work = self.dir / f"{tag}-{hmma or 'native'}"
        work.mkdir(exist_ok=True)
        r = translate(work, "k", p, hmma)
        self.assertEqual(r.status, "ok", r.message)
        res = json.loads((work / "k.res.json").read_text(encoding="utf-8"))
        out_size = self.GROUPS * hc.LOCAL * 4 * p.out_words
        (work / "in.bin").write_bytes(inp)
        (work / "c0.bin").write_bytes(hc.c0_bank(len(inp), out_size))
        sb, sbufs = res["support_buffer"], {b["cb_byte_offset"]: b for b in res["sbufs"]}
        (c0,) = [c for c in res["cbufs"] if c["slot"] == 0]
        lines = ["validation on", f"module {work / 'k.spv'}", f"groups {self.GROUPS} 1 1", "warmup 0", "reps 1", "chunk 1",
                 f"buffer {sb['set']} {sb['binding']} ubo {sb['size']}",
                 f"buffer {c0['set']} {c0['binding']} ubo 65536 in={work / 'c0.bin'}",
                 f"buffer {sbufs[hc.SB_IN]['set']} {sbufs[hc.SB_IN]['binding']} ssbo {len(inp)} in={work / 'in.bin'}",
                 f"buffer {sbufs[hc.SB_OUT]['set']} {sbufs[hc.SB_OUT]['binding']} ssbo {out_size} out={work / 'out.bin'}"]
        (work / "job.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        done = subprocess.run([str(self.harness), "run", str(work / "job.txt")], capture_output=True, text=True,
                              env=hc.vk_env(), timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        report = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual((report["validation_errors"], report["validation_warnings"]), (0, 0))
        got = np.frombuffer((work / "out.bin").read_bytes(), np.uint32).reshape(self.GROUPS, -1)
        self.assertTrue(np.array_equal(got[0], got[1]), "both workgroups ran the same inputs")
        return got[0]

    def test_one_chain_in_both_forms_is_equal_bit_for_bit_on_exact_inputs(self):
        p = hc.hmma_program(32, False, False)
        inp = exact_input(7)
        native, helper = self.run_form(p, None, inp, "exact"), self.run_form(p, "helper", inp, "exact")
        console = hc.model(p, inp)
        self.assertEqual(int(np.count_nonzero(native != helper)), 0)
        self.assertEqual(int(np.count_nonzero(native != console)), 0)
        self.assertGreater(float(np.max(np.abs(hc.lane_halves(console)))), 0.1)   # not a chain of zeros

    def test_one_chain_in_both_forms_stays_within_the_stated_tolerance_on_general_inputs(self):
        p = hc.hmma_program(32, False, True)
        inp = hc.lane_input(11)
        native, helper = self.run_form(p, None, inp, "general"), self.run_form(p, "helper", inp, "general")
        n, h = hc.lane_halves(native).astype(np.float64), hc.lane_halves(helper).astype(np.float64)
        self.assertTrue(np.all(np.isfinite(n)))
        scale = float(np.max(np.abs(h)))
        self.assertLessEqual(float(np.max(np.abs(n - h))), 2.0 ** -6 * scale)   # the portable check
        # The helper rounds D to f16 at every call (the per-call model); a driver that folds its pack and the next call's
        # unpack (617.14) carries f32 across the chain instead (the f32-chain model). Which one ran is the driver's doing.
        per_call, folded = hc.model(p, inp), hc.model(p, inp, f32_chains=True)
        with self.subTest("the helper's results equal the f32-chain model where the driver folds"):
            self.assertGreater(int(np.count_nonzero(per_call != folded)), 0, "these inputs cannot tell a fold")
            if np.count_nonzero(helper != folded) and not np.count_nonzero(helper != per_call):
                self.skipTest("this driver keeps the helper's rounding of D at every call (its results are the per-call "
                              "model's): the f32-chain model holds only where a driver folds the pack and unpack")
            self.assertEqual(int(np.count_nonzero(helper != folded)), 0)

    def test_one_hmma_equals_the_cpu_model(self):
        p = hc.hmma_program(4, False, False)   # one HMMA on each chain
        inp = exact_input(3)
        self.assertEqual(int(np.count_nonzero(self.run_form(p, None, inp, "single") != hc.model(p, inp))), 0)


class ProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        need_xlate()
        done = run("usable")
        if done.returncode == EXIT_NO_DEVICE or (done.returncode == 0 and "usable 0" in done.stdout):
            why = f"no Vulkan device here has every item ({done.stdout.strip()})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
            raise unittest.SkipTest(why)
        cls.dir = scratch()
        r = translate(cls.dir, "native", hc.hmma_program(8, False, True))
        assert r.status == "ok", r.message
        entry = cp.Entry("native", cls.dir / "native.spv", cls.dir / "native.res.json")
        cls.pack = cls.dir / "hmma.bin"
        cp.write_pack(cls.pack, [entry], translator=hashlib.sha256(EXE.read_bytes()).hexdigest())

    def probe(self, *args: str, **env: str) -> tuple[str, str]:
        done = run("hmma_probe", *args, env={"OOT2_VK_VALIDATION": "on", **env})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
        self.assertIn("validation errors 0 warnings 0", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        return done.stdout, done.stderr

    def test_the_gate_reasons_are_read_from_the_probe(self):
        reasons = gate_reasons()
        self.assertEqual(len(reasons), 6, reasons)
        self.assertIn("the device lists no 16x8x16 subgroup matrix with f16 A, B, C and result", reasons)

    def test_the_probe_measures_the_translators_order_and_keeps_the_native_form(self):
        out, err = self.probe()
        line = next((l for l in err.splitlines() if "HMMA" in l), "")
        skip_on_the_gate(self, *verdict(out))
        a, b, c = orders(RUNTIME_HEADER, r"#define VKH_ORDER_[ABC] \{([0-9, ]+)\}")
        self.assertIn("hmma native 1", out, line)
        self.assertIn(f"components A {' '.join(map(str, a))}, B {' '.join(map(str, b))}, C {' '.join(map(str, c))}, "
                      "the translator's order", line)

    def test_a_fault_in_the_probes_read_back_keeps_the_helper_by_name(self):
        """The third check, counter-run on the device: a read-back whose lanes hold 7 components of A, whose helper's D
        differs from the native one in one half, or whose two forms agree on a half that is not the exact sum, keeps the
        helper, with the count named. Without the lengths check or either count, a case gives the native form."""
        for fault, why in (("lengths", "a lane holds 7, 4 and 4 components of A, B and C, not 8, 4 and 4"),
                           ("helper", "a multiply of small integers differs between the forms in 1 of 128 elements and the "
                                      "native one from the exact sums in 0"),
                           ("sums", "a multiply of small integers differs between the forms in 0 of 128 elements and the "
                                    "native one from the exact sums in 1")):
            with self.subTest(fault=fault):
                out, err = self.probe(fault)
                skip_on_the_gate(self, *verdict(out))
                self.assertEqual(verdict(out), (0, why))
                self.assertIn(f"HMMA in the helper: {why}", err)

    def test_the_setting_forces_the_helper(self):
        out, err = self.probe(OOT2_VK_HMMA="helper")
        self.assertIn("hmma native 0: OOT2_VK_HMMA=helper", out)
        self.assertIn("HMMA in the helper: OOT2_VK_HMMA=helper", err)

    def test_a_native_module_is_built_in_either_form(self):
        for setting, native in (("auto", 1), ("helper", 0)):
            with self.subTest(setting=setting):
                done = run("hmma_build", str(self.pack), "native", env={"OOT2_VK_VALIDATION": "on", "OOT2_VK_HMMA": setting})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-3000:])
                if setting == "auto":
                    skip_on_the_gate(self, *verdict(done.stdout))      # a device without the native form: the helper, named
                self.assertEqual(verdict(done.stdout)[0], native, done.stdout)
                self.assertIn("ok hmma_build", done.stdout)
                self.assertIn("validation errors 0 warnings 0", done.stdout)


if __name__ == "__main__":
    unittest.main()
