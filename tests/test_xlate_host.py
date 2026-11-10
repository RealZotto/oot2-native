# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the translator host (xlate/host), through its real executable.

The tests skip when the executable has not been built (py -3 tools/build_xlate.py),
unless OOT2_REQUIRE_XLATE is set, as the Windows CI job sets it after building it.
Programs are assembled here from a few instruction words, never read from game
files. The module checks use spirv-val from the Vulkan SDK when it is present.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import pool  # noqa: E402
import validate  # noqa: E402

EXE_NAME = "oot2-xlate.exe" if os.name == "nt" else "oot2-xlate"


def find_exe() -> Path | None:
    hits = sorted((ROOT / "build" / "xlate" / "publish").rglob(EXE_NAME))
    return hits[0] if hits else None


def find_tool(name: str) -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    for base in ([Path(sdk) / "Bin", Path(sdk) / "bin"] if sdk else []):
        for suffix in (".exe", ""):
            cand = base / (name + suffix)
            if cand.exists():
                return str(cand)
    return shutil.which(name)


EXE = find_exe()
SPIRV_VAL = find_tool("spirv-val")
SKIP_NO_EXE = "oot2-xlate is not built (run: py -3 tools/build_xlate.py)"

# --- an Ampere (sm_86) program, assembled by hand ---------------------------------------------

# EXIT: opcode 0x94d in bits 0..11 (AmpereOpcodes.Exit), the guard predicate in bits 12..14
# and its negation in bit 15, the condition predicate in bits 87..89 and its negation in bit 90.
# Both predicates are PT (7), the negations are clear: an unconditional exit.
EXIT_LO = 0x94D | (7 << 12)
EXIT_HI = 7 << (87 - 64)
EXIT = struct.pack("<QQ", EXIT_LO, EXIT_HI)


def compute_job(key: str, code: Path, **extra) -> dict:
    job = {"key": key, "stage": "cs", "arch": "sm86", "code": str(code), "c1": "",
           "nvn2_cs": [0, 0, 0, 16, 0, 0, 0], "local": [8, 8, 1], "shared": 0}
    job.update(extra)
    return job


# --- SPIR-V modules, assembled by hand --------------------------------------------------------

OP = dict(Capability=17, MemoryModel=14, EntryPoint=15, ExecutionMode=16, TypeVoid=19, TypeInt=21,
          TypeFloat=22, TypeVector=23, TypeImage=25, TypeArray=28, TypeStruct=30, TypePointer=32,
          TypeFunction=33, Constant=43, ConstantComposite=44, Function=54, FunctionEnd=56,
          Variable=59, Load=61, Decorate=71, MemberDecorate=72, ImageRead=98, ImageWrite=99,
          Label=248, Return=253, GroupNonUniformShuffleUp=347, GroupNonUniformShuffleDown=348,
          TypeRuntimeArray=29, Store=62, AccessChain=65, ArrayLength=68, CopyObject=83,
          AtomicIAdd=234, Extension=10)


def words(op: int, *operands: int) -> list[int]:
    return [((len(operands) + 1) << 16) | op, *operands]


def string_words(text: str) -> list[int]:
    data = text.encode() + b"\0"
    data += b"\0" * (-len(data) % 4)
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def entry_point(model: int, fid: int, name: str, *interface: int) -> list[int]:
    body = [model, fid, *string_words(name), *interface]
    return [((len(body) + 1) << 16) | OP["EntryPoint"], *body]


def module(body: list[list[int]], bound: int) -> bytes:
    flat = [w for inst in body for w in inst]
    return struct.pack(f"<{5 + len(flat)}I", 0x07230203, 0x00010300, 0, bound, 0, *flat)


class Builder:
    """A tiny valid shader: a void main and the ids 1 (void), 2 (void()), 3 (float)."""

    def __init__(self, model: int, caps: tuple[int, ...] = (), extensions: tuple[str, ...] = ()):
        self.model = model
        self.caps = [1, *caps]  # Shader
        self.extensions = extensions
        self.decos: list[list[int]] = []
        self.types = [words(OP["TypeVoid"], 1), words(OP["TypeFunction"], 2, 1), words(OP["TypeFloat"], 3, 32)]
        self.globals: list[list[int]] = []
        self.interface: list[int] = []
        self.code: list[list[int]] = []
        self.next_id = 100

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def bound(self) -> int:
        return self.next_id + 2

    def build(self) -> bytes:
        main = 4
        head = [words(OP["Capability"], c) for c in self.caps]
        head += [[((len(string_words(e)) + 1) << 16) | OP["Extension"], *string_words(e)] for e in self.extensions]
        head.append(words(OP["MemoryModel"], 0, 1))  # Logical, GLSL450
        head.append(entry_point(self.model, main, "main", *self.interface))
        if self.model == 5:
            head.append(words(OP["ExecutionMode"], main, 17, 1, 1, 1))  # LocalSize 1 1 1
        func = [words(OP["Function"], 1, main, 0, 2), words(OP["Label"], 5), *self.code,
                words(OP["Return"]), words(OP["FunctionEnd"])]
        return module(head + self.decos + self.types + self.globals + func, self.bound())


def image_module(dim: int, arrayed: int, ms: int, sampled: int, fmt: int, access: bool | str = False) -> bytes:
    """A compute shader with one image variable; with access, a read and a write of it (2D, unformatted),
    or only the read ("read") or only the write of a constant ("write")."""
    b = Builder(5)
    img, ptr, var = b.new_id(), b.new_id(), b.new_id()
    b.decos += [words(OP["Decorate"], var, 34, 0), words(OP["Decorate"], var, 33, 0)]  # DescriptorSet, Binding
    b.types += [words(OP["TypeImage"], img, 3, dim, 0, arrayed, ms, sampled, fmt),
                words(OP["TypePointer"], ptr, 0, img)]  # UniformConstant
    b.globals.append(words(OP["Variable"], ptr, var, 0))
    if access:
        sint, vec2, vec4, zero, coord = (b.new_id() for _ in range(5))
        loaded, texel = b.new_id(), b.new_id()
        b.types += [words(OP["TypeInt"], sint, 32, 1), words(OP["TypeVector"], vec2, sint, 2),
                    words(OP["TypeVector"], vec4, 3, 4)]
        b.globals += [words(OP["Constant"], sint, zero, 0), words(OP["ConstantComposite"], vec2, coord, zero, zero)]
        b.code.append(words(OP["Load"], img, loaded, var))
        if access == "write":
            fzero = b.new_id()
            b.globals += [words(OP["Constant"], 3, fzero, 0), words(OP["ConstantComposite"], vec4, texel, fzero, fzero, fzero, fzero)]
        else:
            b.code.append(words(OP["ImageRead"], vec4, texel, loaded, coord))
        if access != "read":
            b.code.append(words(OP["ImageWrite"], loaded, coord, texel))
    return b.build()


CAP_SHADER_NON_UNIFORM, CAP_RUNTIME_DESCRIPTOR_ARRAY, CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM = 5301, 5302, 5309
DECORATION_NON_UNIFORM = 5300


def image_array_module(sampled: int, index: str, runtime: bool = False, declared: bool = False) -> bytes:
    """A compute shader with an array of images (set 0, binding 0; Rgba8 when they are storage images, so that no
    capability but the one under test is involved) indexed once at element 1. How the index is marked:
    "chain" the access chain's result is decorated NonUniform, "index" only the index, "load" only the loaded
    image, "uniform" nothing, "none" the array is never indexed. `declared`: the module declares the capability."""
    caps = [CAP_SHADER_NON_UNIFORM]
    if runtime:
        caps.append(CAP_RUNTIME_DESCRIPTOR_ARRAY)
    if declared:
        caps.append(CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM)
    b = Builder(5, caps=tuple(caps), extensions=("SPV_EXT_descriptor_indexing",))
    uint, four, img, arr, parr, pimg, var, one, index_id, chain, loaded = (b.new_id() for _ in range(11))
    b.decos += [words(OP["Decorate"], var, 34, 0), words(OP["Decorate"], var, 33, 0)]  # DescriptorSet, Binding
    b.globals += [words(OP["TypeInt"], uint, 32, 0), words(OP["Constant"], uint, four, 4),
                  words(OP["TypeImage"], img, 3, 1, 0, 0, 0, sampled, 4 if sampled == 2 else 0),
                  words(OP["TypeRuntimeArray"], arr, img) if runtime else words(OP["TypeArray"], arr, img, four),
                  words(OP["TypePointer"], parr, 0, arr), words(OP["TypePointer"], pimg, 0, img),  # UniformConstant
                  words(OP["Variable"], parr, var, 0), words(OP["Constant"], uint, one, 1)]
    if index != "none":
        b.code += [words(OP["CopyObject"], uint, index_id, one),
                   words(OP["AccessChain"], pimg, chain, var, index_id), words(OP["Load"], img, loaded, chain)]
        marked = {"chain": chain, "index": index_id, "load": loaded}.get(index)
        if marked is not None:
            b.decos.append(words(OP["Decorate"], marked, DECORATION_NON_UNIFORM))
    return b.build()


def buffer_module(*uses: str) -> bytes:
    """A compute shader with one storage buffer (set 0, binding 32): a Uniform variable of a
    BufferBlock structure holding a runtime array of uint, as the vendored generator declares
    them. Each use touches word 0: "load", "store", "atomic" (an add), "copy-store" (a store
    through a copy of the pointer), "length" (OpArrayLength only)."""
    b = Builder(5)
    uint, rarr, block, pblock, puint, var, zero, one = (b.new_id() for _ in range(8))
    b.decos += [words(OP["Decorate"], rarr, 6, 4),                 # ArrayStride 4
                words(OP["MemberDecorate"], block, 0, 35, 0),      # Offset 0
                words(OP["Decorate"], block, 3),                   # BufferBlock
                words(OP["Decorate"], var, 34, 0), words(OP["Decorate"], var, 33, 32)]
    b.types += [words(OP["TypeInt"], uint, 32, 0), words(OP["TypeRuntimeArray"], rarr, uint),
                words(OP["TypeStruct"], block, rarr), words(OP["TypePointer"], pblock, 2, block),
                words(OP["TypePointer"], puint, 2, uint)]
    b.globals += [words(OP["Constant"], uint, zero, 0), words(OP["Constant"], uint, one, 1),
                  words(OP["Variable"], pblock, var, 2)]
    for use in uses:
        chain = b.new_id()
        b.code.append(words(OP["AccessChain"], puint, chain, var, zero, zero))
        if use == "load":
            b.code.append(words(OP["Load"], uint, b.new_id(), chain))
        elif use == "store":
            b.code.append(words(OP["Store"], chain, one))
        elif use == "atomic":
            b.code.append(words(OP["AtomicIAdd"], uint, b.new_id(), chain, one, zero, one))  # Device scope, no semantics
        elif use == "copy-store":
            copy = b.new_id()
            b.code += [words(OP["CopyObject"], puint, copy, chain), words(OP["Store"], copy, one)]
        elif use == "length":
            b.code.append(words(OP["ArrayLength"], uint, b.new_id(), var, 0))
        else:
            raise ValueError(use)
    return b.build()


def ampere(op: int, rd: int = 255, ra: int = 255, rb: int = 255, low_bits: int = 0, rc: int = 255, high_bits: int = 0) -> bytes:
    """One sm_86 instruction: the opcode in bits 0..11, guard PT (7) in 12..14, Rd 16..23,
    Ra 24..31, Rb 32..39, more fields from bit 40 (low_bits), Rc 64..71 and fields from bit 72
    (high_bits, given as bit numbers of the whole instruction)."""
    lo = op | (7 << 12) | (rd << 16) | (ra << 24) | (rb << 32) | low_bits
    hi = rc | (high_bits >> 64)
    return struct.pack("<QQ", lo, hi)


def bit(n: int, value: int = 1) -> int:
    """value placed at bit n of the 128-bit instruction (for ampere's high_bits)."""
    return value << n


def mesh_fields(stage: str, payload: int = 256, max_vertices: int = 64, max_primitives: int = 64, has_task: bool = True,
                vertex: list | None = None, primitive: list | None = None, own_shared: bool = False) -> dict:
    """The fields a task or mesh job carries, as split.py writes them: the thread group, the compute model's shared size and
    stage memory layout (accepted, no longer read), the mesh object that gives the sizes and, for a mesh job, its output map."""
    mesh = {"kind": stage, "threads": 32, "payload": payload, "own_shared": own_shared}
    fields = {"local": [32, 1, 1], "shared": 192, "stage_memory": [0, 16, 64, 128], "mesh": mesh}
    if stage == "ms":
        mesh.update(has_task=has_task, max_vertices=max_vertices, max_primitives=max_primitives)
        fields["outputs"] = {"vertex": vertex or [], "primitive": primitive or []}
    return fields


def builtin_module() -> bytes:
    """A vertex shader whose output block has Position, ClipDistance and CullDistance members."""
    b = Builder(0)
    uint, two, vec4, arr, block, ptr, out = (b.new_id() for _ in range(7))
    b.decos += [words(OP["MemberDecorate"], block, 0, 11, 0), words(OP["MemberDecorate"], block, 1, 11, 3),
                words(OP["MemberDecorate"], block, 2, 11, 4), words(OP["Decorate"], block, 2)]  # BuiltIn, Block
    b.types += [words(OP["TypeInt"], uint, 32, 0), words(OP["TypeVector"], vec4, 3, 4)]
    b.globals += [words(OP["Constant"], uint, two, 2), words(OP["TypeArray"], arr, 3, two),
                  words(OP["TypeStruct"], block, vec4, arr, arr),
                  words(OP["TypePointer"], ptr, 3, block),  # Output
                  words(OP["Variable"], ptr, out, 3)]
    b.interface.append(out)
    return b.build()


def shuffle_module(op: str) -> bytes:
    """A compute shader that shuffles up or down; only GroupNonUniformShuffle is declared."""
    b = Builder(5, caps=(61,))  # GroupNonUniformShuffle
    uint, scope, one, result = (b.new_id() for _ in range(4))
    b.types.append(words(OP["TypeInt"], uint, 32, 0))
    b.globals += [words(OP["Constant"], uint, scope, 3), words(OP["Constant"], uint, one, 1)]  # Subgroup scope
    b.code.append(words(OP[op], uint, result, scope, one, one))
    return b.build()


class Harness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if EXE is None:
            if os.environ.get("OOT2_REQUIRE_XLATE"):
                raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
            raise unittest.SkipTest(SKIP_NO_EXE)

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "out"
        self.exit_bin = self.dir / "exit.bin"
        self.exit_bin.write_bytes(EXIT)

    def tearDown(self):
        self._tmp.cleanup()

    def run_jobs(self, jobs: list[dict], workers: int = 1):
        results, stats = pool.run_pool([EXE], [(j, self.out) for j in jobs], workers, self.dir / "logs", timeout=120)
        return {r.key: r for r in results}

    def files(self) -> list[str]:
        return sorted(p.name for p in self.out.iterdir()) if self.out.exists() else []

    def exchange(self, *lines: str) -> list[str]:
        """Raw protocol: send the request lines (CRLF terminated), return the reply lines."""
        done = subprocess.run([str(EXE), "--serve"], input="".join(l + "\r\n" for l in lines),
                              capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.splitlines()


class ServeProtocolTest(Harness):
    def test_hello_and_version_agree(self):
        (reply,) = self.exchange('{"op":"hello"}')
        status, key, micros, rest = pool.parse_reply(reply)
        self.assertEqual((status, key, micros), ("ok", "hello", 0))
        info = json.loads(rest)
        self.assertEqual(info["proto"], 2)
        self.assertEqual(len(info["translator_id"]), 64)
        self.assertEqual(info["target"], "vulkan1.3")
        self.assertEqual(info["defaults"]["debug"], False)      # the sound form of the AST optimizer is the default
        version = subprocess.run([str(EXE), "--version"], capture_output=True, text=True, timeout=60).stdout
        self.assertIn(info["translator_id"], version)
        self.assertTrue(version.rstrip("\r\n").endswith(" optimizer=sound"), version)   # what prepare reads first

    def test_a_translation_rides_in_the_reply_and_the_host_writes_no_file(self):
        request = json.dumps({"job": compute_job("wire", self.exit_bin), "out": str(self.out)})
        (reply,) = self.exchange(request)
        self.assertEqual(reply.count(" "), 4)                       # ok, key, microseconds, module, description
        status, key, micros, rest = pool.parse_reply(reply)
        self.assertEqual((status, key), ("ok", "wire"))
        module, description = pool.decode_ok(key, rest)
        self.assertEqual(module[:4], bytes.fromhex("03022307"))
        res = json.loads(description)
        self.assertEqual((res["schema"], res["key"]), (2, "wire"))
        self.assertEqual(self.files(), [])                          # the out field is ignored: no loose file

    def test_quit_and_end_of_input_exit_cleanly(self):
        self.assertEqual(self.exchange('{"op":"quit"}', '{"op":"hello"}')[0].split()[:2], ["ok", "quit"])
        self.assertEqual(self.exchange(), [])

    def test_a_bad_line_never_stops_the_loop(self):
        replies = self.exchange("not json", '{"op":"nope"}', '{"job":{"key":"k1"},"out":"x","extra":1}', '{"op":"hello"}')
        self.assertEqual(len(replies), 4)
        for line in replies[:3]:
            status, key, _, rest = pool.parse_reply(line)
            self.assertEqual((status, key), ("fail", "-"))
            self.assertTrue(rest.startswith("bad_job:"), rest)
        self.assertTrue(replies[3].startswith("ok hello"))


class AmpereHandlersTest(Harness):
    """Handlers the vendored decoder lacked at nxnative 865cc35 (the translator defect fixes)."""

    def program(self, name: str, *instructions: bytes) -> Path:
        path = self.dir / f"{name}.bin"
        path.write_bytes(b"".join(instructions) + EXIT)
        return path

    def translate(self, key: str, job: dict):
        got = self.run_jobs([job])[key]
        if got.status == "ok" and SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, key, self.out / f"{key}.spv")
            self.assertTrue(v.ok, v.message)
        return got

    # F2IP (0x243): .U8.F32.NTZ is bit 74, the rounding mode bits 78..79.
    def test_f2ip_round_to_nearest_pair(self):
        code = self.program("f2ip", ampere(0x243, rd=0, ra=1, rb=2, high_bits=bit(74)))
        got = self.translate("f2ip", compute_job("f2ip", code))
        self.assertEqual((got.status, got.message), ("ok", ""))

    def test_f2ip_truncating_with_source_a_rz(self):
        code = self.program("f2ipt", ampere(0x243, rd=0, ra=255, rb=2, high_bits=bit(74) | bit(78, 3)))
        got = self.translate("f2ipt", compute_job("f2ipt", code))
        self.assertEqual((got.status, got.message), ("ok", ""))

    def test_f2ip_other_forms_are_refused(self):
        for key, rc, high in (("f2ipc", 3, bit(74)), ("f2ips", 255, bit(74) | bit(72))):  # C not RZ; signed
            with self.subTest(key=key):
                code = self.program(key, ampere(0x243, rd=0, ra=1, rb=2, rc=rc, high_bits=high))
                got = self.translate(key, compute_job(key, code))
                self.assertEqual(got.status, "fail")
                self.assertIn("F2IP other than", got.message)

    # F2I (0x305): the source's size is bits 84..85; a 64-bit float (3) is not converted, so no rule for it is needed.
    def test_f2i_from_a_64_bit_float_is_refused(self):
        code = self.program("f2id", ampere(0x305, rd=0, rb=2, high_bits=bit(72) | bit(75, 2) | bit(78, 3) | bit(84, 3)))
        got = self.translate("f2id", compute_job("f2id", code))
        self.assertEqual(got.status, "fail")
        self.assertTrue(got.message.startswith("unsupported_opcode:"), got.message)
        self.assertIn("F2I from a float that is neither 16-bit nor 32-bit", got.message)

    # FMNMX (0x209): the predicate that picks the minimum is bits 87..90 and .FTZ is bit 80. The other bits of 74..86 select
    # modes the translator does not have, the minimum and maximum that give a NaN for a NaN operand among them: a word
    # with one of them is refused by name, never translated as the plain minimum or maximum.
    def test_fmnmx_mode_bits_are_refused(self):
        for n in (*range(74, 80), *range(81, 87)):
            with self.subTest(bit=n):
                key = f"fmnmx{n}"
                code = self.program(key, ampere(0x209, rd=0, ra=1, rb=2, high_bits=bit(87, 7) | bit(n)))
                got = self.translate(key, compute_job(key, code))
                self.assertEqual(got.status, "fail")
                self.assertTrue(got.message.startswith("unsupported_opcode:"), got.message)
                self.assertIn("FMNMX mode bits 74..79, 81..86", got.message)

    def test_fmnmx_without_a_mode_bit_translates(self):
        """.FTZ (bit 80) is no mode bit, and bit 91 belongs to the form whose second operand is a uniform register (0xc09)."""
        words = (("fmnmx", 0x209, 0), ("fmnmx_ftz", 0x209, bit(80)), ("fmnmx_ur", 0xC09, bit(91)), ("fmnmx_ur_ftz", 0xC09, bit(80) | bit(91)))
        for key, op, high in words:
            with self.subTest(key=key):
                code = self.program(key, ampere(op, rd=0, ra=1, rb=2, high_bits=bit(87, 7) | high))
                got = self.translate(key, compute_job(key, code))
                self.assertEqual((got.status, got.message), ("ok", ""))

    # ATOMG.CAS (0x3a9): .E (64-bit address pair) is bit 72, the type 73..75, the uniform form bit 91.
    def test_atomg_cas(self):
        code = self.program("cas", ampere(0x3a9, rd=0, ra=2, rb=4, rc=5, high_bits=bit(72)))
        got = self.translate("cas", compute_job("cas", code))
        self.assertEqual((got.status, got.message), ("ok", ""))

    def test_atomg_cas_wider_types_are_refused(self):
        code = self.program("cas64", ampere(0x3a9, rd=0, ra=2, rb=4, rc=6, high_bits=bit(72) | bit(73, 2)))
        got = self.translate("cas64", compute_job("cas64", code))
        self.assertEqual(got.status, "fail")
        self.assertIn("ATOMG.CAS type 2", got.message)

    # ISBERD (0x923) in a task shader: the mode is bits 72..79, the offset 40..63.
    def task_job(self, key: str, mode: int) -> dict:
        code = self.program(key, ampere(0x923, rd=0, ra=255, low_bits=0x10 << 40, high_bits=bit(72, mode)))
        return {"key": key, "stage": "ts", "arch": "sm86", "code": str(code), "c1": "", **mesh_fields("ts")}

    def test_isberd_reads_back_the_task_payload(self):
        got = self.translate("isb", self.task_job("isb", 0xB8))
        self.assertEqual((got.status, got.message), ("ok", ""))

    def test_isberd_other_modes_stay_refused(self):
        got = self.translate("isb99", self.task_job("isb99", 0x99))
        self.assertEqual(got.status, "fail")
        self.assertIn("ISBERD mode 99", got.message)

    # RED (0x98e) .E.ADD.F32.FTZ.RN: .E is bit 72, the type 73..75 (3: F32FtzRn), the operation 87..89 (0: add).
    def test_float_reduction_becomes_an_atomic_float_add(self):
        code = self.program("redf", ampere(0x98E, ra=2, rb=4, high_bits=bit(72) | bit(73, 3)))
        got = self.translate("redf", compute_job("redf", code))
        self.assertEqual((got.status, got.message), ("ok", ""))
        res = json.loads((self.out / "redf.res.json").read_text(encoding="utf-8"))
        self.assertIn("AtomicFloat32AddEXT", res["capabilities"])
        self.assertIn("SPV_EXT_shader_atomic_float_add", res["extensions"])
        self.assertFalse([n for n in res["notes"] if "reduction" in n], res["notes"])
        self.assertIn(struct.pack("<I", (7 << 16) | 6035), (self.out / "redf.spv").read_bytes())  # OpAtomicFAddEXT, 7 words

    def test_an_integer_reduction_is_unchanged(self):
        code = self.program("redi", ampere(0x98E, ra=2, rb=4, high_bits=bit(72)))
        got = self.translate("redi", compute_job("redi", code))
        self.assertEqual(got.status, "ok")
        res = json.loads((self.out / "redi.res.json").read_text(encoding="utf-8"))
        self.assertNotIn("AtomicFloat32AddEXT", res["capabilities"])

    # AST (0x322) a[0x5c]: the address is bits 40..49, the value register Rb.
    def test_attribute_0x5c_is_dropped_on_purpose(self):
        header = bytearray(0x80)
        struct.pack_into("<I", header, 0, (1 << 10) | 3)
        code = self.dir / "vprs.bin"
        code.write_bytes(bytes(header) + ampere(0x322, rb=255, low_bits=0x5C << 40) + EXIT)
        job = {"key": "vprs", "stage": "vs", "arch": "sm86", "code": str(code), "c1": ""}
        got = self.translate("vprs", job)
        self.assertEqual(got.status, "ok")
        notes = json.loads((self.out / "vprs.res.json").read_text(encoding="utf-8"))["notes"]
        self.assertTrue(any("0x5C (the variable pixel rate shading table index) is dropped on purpose" in n for n in notes), notes)
        self.assertFalse(any("0x5C is not valid" in n for n in notes), notes)


class TranslateTest(Harness):
    def test_minimal_compute_program(self):
        got = self.run_jobs([compute_job("exit1", self.exit_bin)])["exit1"]
        self.assertEqual((got.status, got.message), ("ok", ""))
        self.assertGreater(got.micros, 0)
        spv = self.out / "exit1.spv"
        self.assertEqual(got.spv_bytes, spv.stat().st_size)
        self.assertEqual(self.files(), ["exit1.res.json", "exit1.spv"])  # nothing else, no .tmp left
        self.assertEqual(spv.read_bytes()[:4], bytes.fromhex("03022307"))
        res = json.loads((self.out / "exit1.res.json").read_text(encoding="utf-8"))
        self.assertEqual((res["schema"], res["key"], res["stage"], res["exec_model"]), (2, "exit1", "cs", "GLCompute"))
        self.assertEqual(res["compute"], {"local_size": [8, 8, 1], "shared_bytes": 0})
        self.assertEqual((res["passes"], res["timing"]["cold"], res["capabilities_added"]), (1, True, []))
        self.assertIn("Shader", res["capabilities"])
        self.assertEqual(len(res["translator_id"]), 64)
        self.assertEqual(len(res["spv_sha256"]), 64)
        self.assertEqual(res["options"]["debug"], False)       # a job that names no option gets the sound form
        if SPIRV_VAL:
            self.assertTrue(validate.validate_one(SPIRV_VAL, "exit1", spv).ok)

    def test_the_slice_of_a_shared_code_file(self):
        both = self.dir / "both.bin"
        both.write_bytes(bytes(16) + EXIT)  # an unknown instruction first, the exit second
        job = compute_job("sliced", both, code_offset=16, code_length=16)
        self.assertEqual(self.run_jobs([job])["sliced"].status, "ok")

    def test_answers_and_options_are_recorded(self):
        job = compute_job("answers", self.exit_bin, tex={"*": "cube", "3": "2darray"},
                          img={"1": {"type": "3d", "fmt": "R32Uint"}}, attr={"*": "sint", "2": "uint"},
                          fout={"0": "sint"}, early_fragment_tests=True, heap_images=100,
                          options={"variant": "v1", "reduced": True, "debug": False})
        self.assertEqual(self.run_jobs([job])["answers"].status, "ok")
        res = json.loads((self.out / "answers.res.json").read_text(encoding="utf-8"))
        self.assertEqual(res["answers"]["tex"], {"*": "cube", "3": "2darray"})
        self.assertEqual(res["answers"]["img"], {"1": {"type": "3d", "fmt": "R32Uint"}})
        self.assertEqual(res["answers"]["attr"], {"*": "sint", "2": "uint"})
        self.assertEqual((res["answers"]["heap_images"], res["answers"]["heap_samplers"]), (100, 4096))
        self.assertEqual((res["options"]["variant"], res["options"]["reduced"], res["options"]["debug"]), ("v1", True, False))

    def test_an_unimplemented_opcode_is_reported_with_its_number(self):
        bad = self.dir / "bad.bin"
        bad.write_bytes(struct.pack("<QQ", 0x123, 0) + EXIT)
        got = self.run_jobs([compute_job("unk", bad)])["unk"]
        self.assertEqual(got.status, "fail")
        self.assertTrue(got.message.startswith("unsupported_opcode:"), got.message)
        self.assertIn("123", got.message)
        self.assertEqual(self.files(), [])

    def graphics_job(self, key: str, stage: str, header_type: int) -> dict:
        # The 0x80-byte header; its word 0 carries the stage in bits 10..13 and a version below.
        header = bytearray(0x80)
        struct.pack_into("<I", header, 0, (header_type << 10) | 3)
        code = self.dir / f"{key}.bin"
        code.write_bytes(bytes(header) + EXIT)
        return {"key": key, "stage": stage, "arch": "sm86", "code": str(code), "c1": ""}

    def test_a_graphics_program_translates_and_gets_the_vertex_extras(self):
        got = self.run_jobs([self.graphics_job("vtx", "vs", 1)])["vtx"]
        self.assertEqual((got.status, got.message), ("ok", ""))
        res = json.loads((self.out / "vtx.res.json").read_text(encoding="utf-8"))
        self.assertEqual(res["exec_model"], "Vertex")
        self.assertIn("ClipDistance", res["capabilities"])  # declared by the post-pass: every vertex block has it
        if SPIRV_VAL:
            self.assertTrue(validate.validate_one(SPIRV_VAL, "vtx", self.out / "vtx.spv").ok)

    def test_the_post_pass_can_be_switched_off(self):
        job = self.graphics_job("raw", "vs", 1)
        job["options"] = {"capability_fixup": False}
        self.assertEqual(self.run_jobs([job])["raw"].status, "ok")
        res = json.loads((self.out / "raw.res.json").read_text(encoding="utf-8"))
        self.assertEqual(res["capabilities_added"], [])
        self.assertNotIn("ClipDistance", res["capabilities"])

    def test_the_header_decides_the_stage(self):
        got = self.run_jobs([self.graphics_job("mix", "vs", 5)])["mix"]
        self.assertEqual(got.status, "fail")
        self.assertTrue(got.message.startswith("bad_job:"), got.message)
        self.assertIn("Fragment", got.message)
        got = self.run_jobs([self.graphics_job("geo", "vs", 4)])["geo"]
        self.assertTrue(got.message.startswith("unsupported_stage:"), got.message)
        self.assertNotIn("mix.spv", self.files())
        self.assertNotIn("geo.spv", self.files())


class UniLayoutTest(Harness):
    """The layout `uni` (the stage bindings work): fixed slots per stage, textures through push constant slots, schema 2."""

    def vertex_job(self, key: str, *instructions: bytes) -> dict:
        header = bytearray(0x80)
        struct.pack_into("<I", header, 0, (1 << 10) | 3)
        code = self.dir / f"{key}.bin"
        code.write_bytes(bytes(header) + b"".join(instructions) + EXIT)
        return {"key": key, "stage": "vs", "arch": "sm86", "code": str(code), "c1": ""}

    def translate(self, job: dict) -> dict:
        got = self.run_jobs([job])[job["key"]]
        self.assertEqual((got.status, got.message), ("ok", ""))
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, job["key"], self.out / f"{job['key']}.spv")
            self.assertTrue(v.ok, v.message)
        return json.loads((self.out / f"{job['key']}.res.json").read_text(encoding="utf-8"))

    # TEX (0xb60) R<rd>, c[2] handle of a combined slot, 2D (61..63 = 1), level zero (87..89 = 1), four
    # components (72..75), no residency predicate (81..83 = PT); coordinates RZ.
    @staticmethod
    def tex(rd: int, c2_slot: int) -> bytes:
        word = (0x20 + 8 * c2_slot) // 4
        return ampere(0xB60, rd=rd, ra=255, rb=255, low_bits=(word << 40) | (2 << 54) | (1 << 61),
                      high_bits=bit(72, 0xF) | bit(81, 7) | bit(87, 1))

    # AST (0x322) a[addr] from Rb, size bits 74..75 (1: two words, 3: four words).
    @staticmethod
    def ast(addr: int, rb: int, size: int) -> bytes:
        return ampere(0x322, rb=rb, low_bits=addr << 40, high_bits=bit(74, size))

    def test_compute_basics(self):
        res = self.translate(compute_job("ub", self.exit_bin))
        self.assertEqual(res["schema"], 2)
        lay = res["layout"]
        self.assertEqual((lay["name"], lay["role"], lay["global_set"], lay["push_set"], lay["buffer_set"]), ("uni", "compute", 0, 1, 1))
        self.assertEqual((lay["push_constant_bytes"], lay["texture_slots_offset"], lay["texture_slots"]), (128, 16, [0, 28]))
        self.assertEqual((lay["fits"], lay["overflow"]), (True, []))
        sb = res["support_buffer"]
        self.assertEqual((sb["set"], sb["binding"], sb["size"]), (1, 0, 1392))
        self.assertEqual([(f["name"], f["offset"]) for f in sb["fields"]],
                         [("alpha_test", 0), ("is_bgra", 16), ("viewport_inverse", 144), ("viewport_size", 160),
                          ("frag_scale_count", 176), ("render_scale", 192), ("tfe_offset", 1360), ("tfe_vertex_count", 1376)])
        self.assertEqual(res["storage_buffer_contract"]["alignment"], 16)
        self.assertEqual((res["push_descriptors"], res["texture_slots"], res["vertex_inputs"]), (1, 0, None))
        self.assertNotIn("descriptor_count_set0", res)

    def test_a_texture_of_a_vertex_program_is_a_push_constant_slot(self):
        res = self.translate(self.vertex_job("vt", self.tex(0, 3), self.ast(0x70, 0, 3), self.ast(0x80, 0, 3)))
        self.assertEqual(res["layout"]["role"], "pre_rasterization")
        (t,) = res["textures"]
        self.assertEqual((t["set"], t["binding"], t["via"], t["push_slot"], t["push_offset"], t["c2_slot"]),
                         (0, 0, "push_slot", 0, 16, 3))
        self.assertEqual((res["push_descriptors"], res["texture_slots"], res["layout"]["fits"]), (1, 1, True))
        self.assertIn("Position", res["interface"]["builtins_out"])
        self.assertEqual([(o["location"], o["components"], o["type"]) for o in res["interface"]["outputs"]], [(0, 4, "float32")])
        self.assertEqual(res["vertex_inputs"], [])
        # The block of slots and the global arrays are declared (debug names are on by default); the code that reads
        # them is checked by spirv-val over the corpus, where textures are sampled for real.
        spv = (self.out / "vt.spv").read_bytes()
        for name in (b"push_constants\0", b"push_images_2d\0", b"push_samplers\0"):
            self.assertIn(name, spv)

    def test_textures_past_the_stage_slots_still_translate(self):
        # Six textures, each kept alive by its own output; the vertex stage has five slots.
        code = []
        for k in range(6):
            code += [self.tex(2 * k, k), self.ast(0x80 + 16 * k, 2 * k, 1)]
        res = self.translate(self.vertex_job("vt6", *code))
        slots = sorted(t["push_slot"] for t in res["textures"])
        self.assertEqual(slots, [0, 1, 2, 3, 4, 28])
        self.assertFalse(res["layout"]["fits"])
        self.assertEqual(len(res["layout"]["overflow"]), 1)
        self.assertIn("no texture slot left for the pre_rasterization stage (5)", res["layout"]["overflow"][0])
        self.assertTrue(any(n.startswith("layout uni:") for n in res["notes"]), res["notes"])


class RejectedJobTest(Harness):
    def reject(self, job: dict, kind: str, needle: str = "") -> str:
        got = self.run_jobs([job])[job["key"]]
        self.assertEqual(got.status, "fail", got)
        self.assertTrue(got.message.startswith(kind + ":"), got.message)
        self.assertIn(needle, got.message)
        self.assertLessEqual(len(got.message), 400)
        self.assertEqual(self.files(), [])
        return got.message

    def test_unknown_field_is_bad_job(self):
        self.reject(compute_job("typo", self.exit_bin, shaerd=4), "bad_job", "shaerd")

    def test_code_length_must_be_a_multiple_of_16(self):
        self.reject(compute_job("odd", self.exit_bin, code_length=24), "bad_job", "multiple of 16")

    def test_file_length_must_be_a_multiple_of_16(self):
        short = self.dir / "short.bin"
        short.write_bytes(bytes(40))
        self.reject(compute_job("short", short), "bad_job", "multiple of 16")

    def test_shape_errors(self):
        cases = [
            ("c1a", dict(c1="abc"), "odd number"),
            ("c1b", dict(c1="zz"), "not a hex"),
            ("nv1", dict(nvn2_cs=[1, 2, 3]), "nvn2_cs"),
            ("loc", dict(local=[64, 64, 1]), "1024"),
            ("loc0", dict(local=[0, 1, 1]), "local"),
            ("tex1", dict(tex={"2": "4d"}), "texture type"),
            ("img1", dict(img={"2": {"type": "2d", "fmt": "Nope"}}), "format"),
            ("opt1", dict(options={"fast": True}), "option"),
            ("att1", dict(attr={"40": "float"}), "slot number"),
            ("sm1", dict(stage_memory=[0, 0, 0, 0]), "not valid for stage cs"),
            ("arch", dict(arch="sm75"), "sm86"),
        ]
        for key, extra, needle in cases:
            with self.subTest(key=key):
                self.reject(compute_job(key, self.exit_bin, **extra), "bad_job", needle)

    def test_missing_field_and_bad_key(self):
        job = compute_job("nolocal", self.exit_bin)
        del job["local"]
        self.reject(job, "bad_job", "local")
        got = self.run_jobs([compute_job("has space", self.exit_bin)])
        self.assertEqual(next(iter(got.values())).status, "fail")

    def test_missing_code_file_is_io(self):
        self.reject(compute_job("nofile", self.dir / "absent.bin"), "io", "not found")

    def test_a_slice_past_the_end_is_io(self):
        self.reject(compute_job("past", self.exit_bin, code_offset=16, code_length=16), "io", "past the end")

    def test_stages_without_a_translation(self):
        for stage in ("gs", "hs", "ds"):
            with self.subTest(stage=stage):
                job = compute_job("st_" + stage, self.exit_bin)
                job["stage"] = stage
                self.reject(job, "unsupported_stage", stage)

    def test_graphics_programs_need_room_for_the_header(self):
        job = {"key": "tiny", "stage": "vs", "arch": "sm86", "code": str(self.exit_bin), "c1": ""}
        self.reject(job, "bad_job", "header")


class MeshOutputMapTest(Harness):
    """The output map of a task or mesh job (the mesh output maps): validated, kept on the job and echoed into res.json. The mesh
    emission (the mesh emission) declares exactly these components; test_xlate_mesh checks that."""

    def stage_job(self, key: str, stage: str, **extra) -> dict:
        job = {"key": key, "stage": stage, "arch": "sm86", "code": str(self.exit_bin), "c1": "", **mesh_fields(stage)}
        job.update(extra)
        return job

    def result(self, job: dict) -> dict:
        got = self.run_jobs([job])[job["key"]]
        self.assertEqual((got.status, got.message), ("ok", ""))
        return json.loads((self.out / f"{job['key']}.res.json").read_text(encoding="utf-8"))

    def reject(self, job: dict, needle: str) -> None:
        got = self.run_jobs([job])[job["key"]]
        self.assertEqual(got.status, "fail", got)
        self.assertTrue(got.message.startswith("bad_job:"), got.message)
        self.assertIn(needle, got.message)

    def test_a_mesh_job_hands_its_output_map_to_the_result(self):
        outputs = {"vertex": [[-1, 0], [-1, 3], [0, 1], [31, 2]], "primitive": [[0, 0], [5, 3]]}
        self.assertEqual(self.result(self.stage_job("ms1", "ms", outputs=outputs))["outputs"], outputs)

    def test_a_task_job_hands_its_output_map_to_the_result(self):
        outputs = {"vertex": [], "primitive": []}
        self.assertEqual(self.result(self.stage_job("ts1", "ts", outputs=outputs))["outputs"], outputs)

    def test_a_job_without_a_map_has_a_null_one(self):
        task = self.stage_job("ts2", "ts")
        self.assertNotIn("outputs", task)
        self.assertIsNone(self.result(task)["outputs"])
        self.assertIsNone(self.result(compute_job("cs1", self.exit_bin))["outputs"])

    def test_a_mesh_job_needs_its_map(self):
        job = self.stage_job("ms2", "ms")
        del job["outputs"]
        self.reject(job, "needs its 'outputs' map")

    def test_a_task_job_takes_an_empty_map_only(self):
        # The output of a task program is its payload; all 20 task jobs of the dump have an empty map
        self.reject(self.stage_job("ts3", "ts", outputs={"vertex": [[0, 0]], "primitive": []}), "task job's output map must be empty")
        self.reject(self.stage_job("ts4", "ts", outputs={"vertex": [], "primitive": [[0, 0]]}), "task job's output map must be empty")

    def test_the_mesh_object_must_match_the_job(self):
        mesh_ms = {"kind": "ms", "threads": 32, "payload": 8, "has_task": True, "max_vertices": 3, "max_primitives": 1}
        cases = [
            ("o1", "ms", {"mesh": {"kind": "ts", "threads": 32, "payload": 8}}, "is not the job's stage ms"),
            ("o2", "ms", {"local": [64, 1, 1]}, "is not mesh.threads 32 by 1 by 1"),
            ("o3", "ts", {"mesh": {"kind": "ts", "threads": 32}}, "missing field 'mesh.payload'"),
            ("o4", "ts", {"mesh": {"kind": "ts", "payload": 8}}, "missing field 'mesh.threads'"),
            ("o5", "ms", {"mesh": {k: v for k, v in mesh_ms.items() if k != "max_vertices"}}, "missing field 'mesh.max_vertices'"),
            ("o6", "ms", {"mesh": {**mesh_ms, "vertex_components": 2}}, "mesh.vertex_components does not match"),
        ]
        for key, stage, extra, needle in cases:
            with self.subTest(key=key):
                self.reject(self.stage_job(key, stage, **extra), needle)

    def test_the_map_must_be_well_formed(self):
        cases = [
            ("m1", {"vertex": [[0, 4]], "primitive": []}, "component 4"),             # a component that does not exist
            ("m2", {"vertex": [[0, 1], [0, 1]], "primitive": []}, "listed twice"),
            ("m3", {"vertex": [[32, 0]], "primitive": []}, "outside"),
            ("m4", {"vertex": [], "primitive": [[-1, 0]]}, "outside"),                # the position is not per primitive
            ("m5", {"vertex": [[0]], "primitive": []}, "array of 2"),
            ("m6", {"vertex": []}, "missing field 'primitive'"),
            ("m7", {"vertex": [], "primitive": [], "extra": []}, "unknown field"),
            ("m8", [], "not an object"),
            ("m9", {"vertex": [[0, -1]], "primitive": []}, "component -1"),
        ]
        for key, outputs, needle in cases:
            with self.subTest(key=key):
                self.reject(self.stage_job(key, "ms", outputs=outputs), needle)

    def test_only_task_and_mesh_jobs_take_a_map(self):
        self.reject(compute_job("cs2", self.exit_bin, outputs={"vertex": [], "primitive": []}), "not valid for stage cs")


@unittest.skipUnless(SPIRV_VAL, "spirv-val is not installed")
class CapabilityPostPassTest(Harness):
    """The post-pass on hand-assembled modules, through the --fixup-module aid."""

    def fix(self, name: str, spirv: bytes) -> tuple[list[str], Path]:
        src, dst = self.dir / f"{name}.spv", self.dir / f"{name}.fixed.spv"
        src.write_bytes(spirv)
        done = subprocess.run([str(EXE), "--fixup-module", str(src), str(dst)], capture_output=True, text=True, timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return done.stdout.split(), dst

    def check(self, name: str, spirv: bytes, expect: list[str], invalid_before: bool = False):
        added, fixed = self.fix(name, spirv)
        self.assertEqual(sorted(added), sorted(expect))
        if invalid_before:
            raw = validate.validate_one(SPIRV_VAL, name, self.dir / f"{name}.spv")
            self.assertFalse(raw.ok, "the module was expected to fail before the post-pass")
        after = validate.validate_one(SPIRV_VAL, name, fixed)
        self.assertTrue(after.ok, after.message)
        # Idempotent: a second pass finds nothing to add.
        again, _ = self.fix(name + "2", fixed.read_bytes())
        self.assertEqual(again, [])

    def test_unformatted_storage_image_access(self):
        self.check("unfmt", image_module(1, 0, 0, 2, 0, access=True),
                   ["StorageImageReadWithoutFormat", "StorageImageWriteWithoutFormat"], invalid_before=True)

    def test_formatted_storage_image_needs_nothing(self):
        # Rgba8; never read nor written, so marked both ways.
        self.check("fmt", image_module(1, 0, 0, 2, 4), ["NonWritable@0.0", "NonReadable@0.0"])

    def test_sampled_1d(self):
        self.check("s1d", image_module(0, 0, 0, 1, 0), ["Sampled1D"], invalid_before=True)

    def test_storage_1d(self):
        self.check("i1d", image_module(0, 0, 0, 2, 4), ["Image1D", "NonWritable@0.0", "NonReadable@0.0"])

    def test_cube_arrays(self):
        self.check("scube", image_module(3, 1, 0, 1, 0), ["SampledCubeArray"])
        self.check("icube", image_module(3, 1, 0, 2, 4), ["ImageCubeArray", "NonWritable@0.0", "NonReadable@0.0"])

    def test_multisampled_storage_array(self):
        self.check("msa", image_module(1, 1, 1, 2, 4),
                   ["ImageMSArray", "StorageImageMultisample", "NonWritable@0.0", "NonReadable@0.0"], invalid_before=True)

    # The declaration of StorageImageArrayNonUniformIndexing: where a storage image array is indexed
    # with a non-uniform index, and nowhere else. The generator no longer declares it.
    UNREAD = ["NonWritable@0.0", "NonReadable@0.0"]   # the Rgba8 images are never read nor written

    def test_a_storage_image_array_indexed_non_uniformly_declares_the_capability(self):
        for index in ("chain", "index", "load"):
            with self.subTest(index=index):
                self.check("sia_" + index, image_array_module(2, index),
                           ["StorageImageArrayNonUniformIndexing", *self.UNREAD])

    def test_a_runtime_array_of_storage_images_counts_too(self):
        self.check("sia_rt", image_array_module(2, "chain", runtime=True),
                   ["StorageImageArrayNonUniformIndexing", *self.UNREAD])

    def test_a_storage_image_array_not_indexed_non_uniformly_declares_nothing(self):
        for index in ("uniform", "none"):
            with self.subTest(index=index):
                self.check("siu_" + index, image_array_module(2, index), self.UNREAD)

    def test_a_sampled_image_array_indexed_non_uniformly_declares_nothing_here(self):
        # SampledImageArrayNonUniformIndexing is the generator's, not the post-pass's.
        self.check("sa_chain", image_array_module(1, "chain"), [])
        self.check("sa_rt", image_array_module(1, "chain", runtime=True), [])

    def test_a_capability_the_module_declares_is_not_added_again(self):
        self.check("sia_decl", image_array_module(2, "chain", declared=True), self.UNREAD)

    def test_clip_and_cull_distance(self):
        self.check("clip", builtin_module(), ["ClipDistance", "CullDistance"])

    def test_shuffle_up_and_down(self):
        self.check("up", shuffle_module("GroupNonUniformShuffleUp"), ["GroupNonUniformShuffleRelative"], invalid_before=True)
        self.check("down", shuffle_module("GroupNonUniformShuffleDown"), ["GroupNonUniformShuffleRelative"], invalid_before=True)

    # Storage access (the translator defect fixes): NonWritable where nothing writes, NonReadable where nothing reads.
    def test_a_read_only_storage_buffer_is_non_writable(self):
        self.check("sb_load", buffer_module("load"), ["NonWritable@0.32"])

    def test_a_write_only_storage_buffer_is_non_readable(self):
        self.check("sb_store", buffer_module("store"), ["NonReadable@0.32"])
        self.check("sb_copy", buffer_module("copy-store"), ["NonReadable@0.32"])

    def test_read_and_write_or_an_atomic_mark_nothing(self):
        self.check("sb_rw", buffer_module("load", "store"), [])
        self.check("sb_atomic", buffer_module("atomic"), [])

    def test_the_length_alone_is_no_access(self):
        self.check("sb_len", buffer_module("length"), ["NonWritable@0.32", "NonReadable@0.32"])

    def test_storage_images_by_access(self):
        self.check("img_r", image_module(1, 0, 0, 2, 0, access="read"), ["StorageImageReadWithoutFormat", "NonWritable@0.0"])
        self.check("img_w", image_module(1, 0, 0, 2, 0, access="write"), ["StorageImageWriteWithoutFormat", "NonReadable@0.0"])

    def test_sampled_images_are_left_alone(self):
        self.check("smp", image_module(1, 0, 0, 1, 0), [])

    def test_a_malformed_module_is_refused(self):
        src, dst = self.dir / "bad.spv", self.dir / "bad.out.spv"
        src.write_bytes(struct.pack("<5I", 0x07230203, 0x10300, 0, 2, 0) + struct.pack("<I", 0))  # an instruction of 0 words
        done = subprocess.run([str(EXE), "--fixup-module", str(src), str(dst)], capture_output=True, text=True, timeout=60)
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(dst.exists())


if __name__ == "__main__":
    unittest.main()
