# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""NaN operands through the translator: float minimum and maximum, saturation, float to integer conversion.

The console's rules (checked against NVIDIA's PTX documentation for the same operations):

* FMNMX returns the other operand when one is NaN, NaN when both are; -0 is less than +0.
* A saturated result (.SAT) that is NaN is 0.
* F2I of NaN is 0 for a 32-bit integer (or smaller) from a single, 1 << 63 for a 64-bit integer.
* HFMA2 and HMUL2 with .RELU leave a NaN a NaN.

Vulkan leaves the result of FMin, FMax and FClamp with a NaN operand to the driver, and of OpConvertFToS and
OpConvertFToU undefined. The programs here are assembled from instruction words (no game files), read their operands
from constant bank 1 so that nothing is folded, and are translated by the real executable. A small evaluator then
executes the straight-line module on the CPU with the SPIR-V specification's meaning of each instruction, and treats
an instruction the specification leaves open for the operands it is given (FMin with a NaN, a conversion of a NaN) as an
undefined value that must not reach the stored result, so a module that relies on a driver's luck is red here.

Run: python -m unittest discover -s tests -v   (skipped without a built oot2-xlate, see test_xlate_host)
"""

from __future__ import annotations

import math
import struct
import unittest

from test_xlate_host import Harness, ampere, bit, compute_job, EXIT, SPIRV_VAL, validate
from test_xlate_uni import SB_BASE, mov_cbuf, scheduled, stg

NAN = float("nan")
INF = float("inf")

# GLSL.std.450 instruction numbers
GLSL = {1: "Round", 2: "RoundEven", 3: "Trunc", 8: "Floor", 9: "Ceil", 37: "FMin", 40: "FMax", 43: "FClamp",
        50: "Fma", 58: "PackHalf2x16", 62: "UnpackHalf2x16", 79: "NMin", 80: "NMax", 81: "NClamp"}
OPEN_FOR_NAN = {"FMin", "FMax", "FClamp"}   # a NaN operand: which operand is the result is left open


class Undefined(AssertionError):
    """A value the SPIR-V specification leaves open reached a store."""


class Poison:
    """The result of an instruction the specification leaves open for its operands (FMin of a NaN: which operand; a
    conversion of a NaN to an integer: any value). It flows through arithmetic, a select that does not pick it drops it,
    and a store of it fails the test."""

    def __repr__(self):
        return "undefined"


POISON = Poison()


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(INF, x)


def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000


def bits_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def nmax(x: float, y: float) -> float:
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x == y == 0:
        return y if math.copysign(1, x) < 0 else x       # -0 compares less than +0
    return y if x < y else x


def nmin(x: float, y: float) -> float:
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    if x == y == 0:
        return x if math.copysign(1, x) < 0 else y
    return y if y < x else x


def half_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def half_value(b: int) -> float:
    return struct.unpack("<e", struct.pack("<H", b & 0xFFFF))[0]


class Evaluator:
    """Runs the entry point of a straight-line compute module for the operations the programs of this file use.
    Constant bank 1 reads come from `bank1` (word index to 32-bit pattern); every other constant buffer word reads zero.
    The words stored to the storage buffer are collected, in order, in `stores`."""

    def __init__(self, spirv: bytes, bank1: dict[int, int]):
        words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
        self.bank1 = bank1
        self.names: dict[int, str] = {}
        self.types: dict[int, tuple] = {}
        self.pointee: dict[int, int] = {}
        self.vals: dict[int, object] = {}
        self.cells: dict[int, object] = {}
        self.stores: list[int] = []
        self.ext_names: dict[int, str] = {}
        self.ext_used: set[str] = set()
        self.ops_used: set[int] = set()
        entry = None
        body: list[tuple[int, list[int]]] = []
        in_entry = False
        i = 5
        while i < len(words):
            n, op = words[i] >> 16, words[i] & 0xFFFF
            w = list(words[i + 1:i + n])
            if op == 5:                                            # OpName
                raw = struct.pack(f"<{len(w) - 1}I", *w[1:])
                self.names[w[0]] = raw.split(b"\0")[0].decode()
            elif op == 15 and entry is None:                       # OpEntryPoint
                entry = w[1]
            elif op == 11:                                         # OpExtInstImport
                raw = struct.pack(f"<{len(w) - 1}I", *w[1:])
                self.ext_names[w[0]] = raw.split(b"\0")[0].decode()
            elif op == 21:
                self.types[w[0]] = ("i", w[1])
            elif op == 22:
                self.types[w[0]] = ("f", w[1])
            elif op == 20:
                self.types[w[0]] = ("b",)
            elif op == 23:
                self.types[w[0]] = ("v", w[2])
            elif op == 32:                                         # OpTypePointer: result, storage class, pointee
                self.pointee[w[0]] = w[2]
            elif op == 43:                                         # OpConstant
                t = self.types[w[0]]
                self.vals[w[1]] = bits_f32(w[2]) if t[0] == "f" else w[2]
            elif op == 41:
                self.vals[w[1]] = True
            elif op == 42:
                self.vals[w[1]] = False
            elif op == 54:                                         # OpFunction
                in_entry = w[1] == entry
            elif op == 56:
                in_entry = False
            elif in_entry:
                body.append((op, w))
            i += n
        self.body = body

    def run(self) -> list[int]:
        for op, w in self.body:
            self.ops_used.add(op)
            self.step(op, w)
            if op == 253:
                break
        return self.stores

    def kind(self, type_id: int) -> str:
        return self.types[type_id][0]

    def step(self, op: int, w: list[int]):
        v = self.vals
        if op in (248, 253):                                       # OpLabel, OpReturn
            return
        if op in (124, 109, 110, 112, 129, 131, 133, 128, 130, 194, 196, 197, 199, 156, 80, 81)                 and any(v.get(x) is POISON for x in w[2:]):
            v[w[1]] = POISON
            return
        if op == 59:                                               # OpVariable
            v[w[1]] = ("var", w[1])
            self.cells[w[1]] = 0.0 if self.kind(self.pointee[w[0]]) == "f" else 0
        elif op == 65:                                             # OpAccessChain
            base = self.names.get(w[2], "")
            idx = [v[x] for x in w[3:]]
            if base == "cp_c1":
                v[w[1]] = ("c1", idx[1] * 4 + idx[2])
            elif base.startswith("cp_c"):
                v[w[1]] = ("c0",)
            else:
                v[w[1]] = ("store", base)
        elif op == 61:                                             # OpLoad
            ptr = v[w[2]]
            if ptr[0] == "var":
                v[w[1]] = self.cells[ptr[1]]
            elif ptr[0] == "c1":
                b = self.bank1.get(ptr[1], 0)
                v[w[1]] = bits_f32(b) if self.kind(w[0]) == "f" else b
            else:
                v[w[1]] = 0.0 if self.kind(w[0]) == "f" else 0
        elif op == 62:                                             # OpStore
            ptr, val = v[w[0]], v[w[1]]
            if val is POISON and ptr[0] == "store":
                raise Undefined("an undefined value was stored as the result")
            if ptr[0] == "var":
                self.cells[ptr[1]] = val
            elif ptr[0] == "store":
                self.stores.append(f32_bits(val) if isinstance(val, float) else val & 0xFFFFFFFF)
            else:
                raise AssertionError(f"a store to {ptr}")
        elif op == 124:                                            # OpBitcast
            src, k = v[w[2]], self.kind(w[0])
            if k == "f":
                v[w[1]] = src if isinstance(src, float) else bits_f32(src)
            else:
                v[w[1]] = f32_bits(src) if isinstance(src, float) else src & 0xFFFFFFFF
        elif op in (109, 110):                                     # OpConvertFToU, OpConvertFToS
            x = v[w[2]]
            r = None if math.isnan(x) or math.isinf(x) else int(x)
            if r is None or (op == 109 and not 0 <= r < 1 << 32) or (op == 110 and not -(1 << 31) <= r < 1 << 31):
                v[w[1]] = POISON          # not representable in the integer
            else:
                v[w[1]] = r & 0xFFFFFFFF
        elif op == 112:                                            # OpConvertUToF
            v[w[1]] = f32(float(v[w[2]]))
        elif op in (129, 131, 133):                                # OpFAdd, OpFSub, OpFMul
            a, b = v[w[2]], v[w[3]]
            v[w[1]] = f32({129: lambda: a + b, 131: lambda: a - b, 133: lambda: a * b}[op]())
        elif op in (128, 130, 194, 196, 197, 199):                 # IAdd, ISub, shifts, Or, And
            a, b = v[w[2]], v[w[3]]
            r = {128: lambda: a + b, 130: lambda: a - b, 194: lambda: a >> b, 196: lambda: a << b,
                 197: lambda: a | b, 199: lambda: a & b}[op]()
            v[w[1]] = r & 0xFFFFFFFF
        elif op == 156:                                            # OpIsNan
            v[w[1]] = math.isnan(v[w[2]])
        elif op == 169:                                            # OpSelect
            v[w[1]] = v[w[3]] if v[w[2]] else v[w[4]]
        elif op == 80:                                             # OpCompositeConstruct
            v[w[1]] = tuple(v[x] for x in w[2:])
        elif op == 81:                                             # OpCompositeExtract
            v[w[1]] = v[w[2]][w[3]]
        elif op == 12:                                             # OpExtInst
            if self.ext_names.get(w[2]) != "GLSL.std.450":
                raise AssertionError("an extended instruction set other than GLSL.std.450")
            name = GLSL.get(w[3])
            if name is None:
                raise AssertionError(f"GLSL.std.450 instruction {w[3]} is not known to the evaluator")
            self.ext_used.add(name)
            args = [v[x] for x in w[4:]]
            if any(a is POISON for a in args) or (name in OPEN_FOR_NAN and any(isinstance(a, float) and math.isnan(a) for a in args)):
                v[w[1]] = POISON          # which operand is the result is left to the driver
            else:
                v[w[1]] = self.ext(name, args)
        else:
            raise AssertionError(f"the evaluator does not know Op {op}")

    def ext(self, name: str, a: list):
        if name == "Trunc":
            return a[0] if math.isnan(a[0]) or math.isinf(a[0]) else float(math.trunc(a[0]))
        if name in ("FMin", "NMin"):
            return nmin(a[0], a[1])
        if name in ("FMax", "NMax"):
            return nmax(a[0], a[1])
        if name in ("FClamp", "NClamp"):
            return nmin(nmax(a[0], a[1]), a[2])
        if name == "Fma":
            return f32(a[0] * a[1] + a[2])
        if name == "UnpackHalf2x16":
            return (half_value(a[0]), half_value(a[0] >> 16))
        if name == "PackHalf2x16":
            return half_bits(a[0][0]) | half_bits(a[0][1]) << 16
        raise AssertionError(f"the evaluator does not know {name}")


def evaluate(spirv: bytes, *operands: int) -> Evaluator:
    """Runs the module with the operands (32-bit patterns) in words 4, 5, 6 of bank 1 (c[1][0x10], c[1][0x14], c[1][0x18])."""
    ev = Evaluator(spirv, {4 + n: o for n, o in enumerate(operands)})
    ev.run()
    return ev


def ext_names_used(spirv: bytes) -> set[str]:
    return evaluate(spirv, 0, 0, 0).ext_used


def only_numeric_forms(test: unittest.TestCase, spirv: bytes, wanted: str):
    used = ext_names_used(spirv)
    test.assertIn(wanted, used)
    test.assertEqual(used & OPEN_FOR_NAN, set(), "an FMin, FMax or FClamp is left in the module")


# --- the programs ------------------------------------------------------------------------------

OUT = 20          # the result register, and 21 for a second word


def program(*instructions: bytes) -> bytes:
    """The storage buffer address into R2:R3, three operands from c[1] into R10, R11 and R12, the instructions, an exit."""
    head = [mov_cbuf(2, 0, SB_BASE), mov_cbuf(3, 0, SB_BASE + 4),
            mov_cbuf(10, 1, 0x10), mov_cbuf(11, 1, 0x14), mov_cbuf(12, 1, 0x18)]
    return b"".join(head + list(instructions)) + EXIT


def store(reg: int, word: int = 0) -> bytes:
    return stg(2, reg, 4 * word)


def fmnmx(maximum: bool) -> bytes:
    """FMNMX R20, R10, R11, PT (the minimum) or !PT (the maximum): the predicate in bits 87..89, its negation in bit 90."""
    return scheduled(ampere(0x209, rd=OUT, ra=10, rb=11, high_bits=bit(87, 7) | (bit(90) if maximum else 0)))


FADD_SAT = scheduled(ampere(0x221, rd=OUT, ra=10, rb=11, high_bits=bit(77)))
FMUL_SAT = scheduled(ampere(0x220, rd=OUT, ra=10, rb=11, high_bits=bit(77) | bit(84, 4)))      # scale field 4: none
FFMA_SAT = scheduled(ampere(0x223, rd=OUT, ra=10, rb=11, rc=12, high_bits=bit(77)))


def f2i(signed: bool) -> bytes:
    """F2I.S32 / .U32 R20, R10, truncating: signedness bit 72, 32 bits (75..77 = 2), rounding 78..80 = 3, an F32 source (84..86 = 2)."""
    return scheduled(ampere(0x305, rd=OUT, ra=255, rb=10, high_bits=(bit(72) if signed else 0) | bit(75, 2) | bit(78, 3) | bit(84, 2)))


F2I_U64 = scheduled(ampere(0x311, rd=OUT, ra=255, rb=10, high_bits=bit(75, 3) | bit(78, 3) | bit(84, 2)))
F2IP_B = scheduled(ampere(0x243, rd=OUT, ra=255, rb=10, high_bits=bit(74) | bit(78, 3)))        # F2IP.U8.F32.NTZ R20, RZ, R10, RZ, truncating
HMUL2_RELU = scheduled(ampere(0x232, rd=OUT, ra=10, rb=11, high_bits=bit(79)))


def mov_imm(rd: int, value: int) -> bytes:
    """MOV Rd, imm32 (0x002 in operand form 4, the immediate in bits 32..63)."""
    return scheduled(ampere(0x802, rd=rd, ra=0, rb=value & 0xFF, low_bits=(value >> 8) << 40))


def fadd_imm_sat(rd: int, ra: int, value: int) -> bytes:
    """FADD.SAT Rd, Ra, imm32 (0x021 in form 4)."""
    return scheduled(ampere(0x821, rd=rd, ra=ra, rb=value & 0xFF, low_bits=(value >> 8) << 40, high_bits=bit(77)))


def same(got: int, want: float) -> bool:
    """A result pattern against the expected float: any NaN for a NaN, the exact pattern (the sign of zero too) otherwise."""
    if math.isnan(want):
        return got & 0x7F800000 == 0x7F800000 and got & 0x7FFFFF != 0
    return got == f32_bits(want)


class NanRulesTest(Harness):
    def module(self, key: str, *instructions: bytes) -> bytes:
        code = self.dir / f"{key}.bin"
        code.write_bytes(program(*instructions))
        got = self.run_jobs([compute_job(key, code)])[key]
        self.assertEqual((got.status, got.message), ("ok", ""))
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, key, self.out / f"{key}.spv")
            self.assertTrue(v.ok, v.message)
        return (self.out / f"{key}.spv").read_bytes()

    def check_float(self, spirv: bytes, cases: list[tuple[tuple[float, ...], float]]):
        for operands, want in cases:
            (out,) = evaluate(spirv, *[f32_bits(o) for o in operands]).stores
            self.assertTrue(same(out, want), f"{operands}: got {bits_f32(out)!r} ({out:#010x}), want {want!r}")

    # -- FMNMX --

    MIN_CASES = [((NAN, 1.0), 1.0), ((1.0, NAN), 1.0), ((NAN, NAN), NAN), ((2.0, 3.0), 2.0), ((3.0, 2.0), 2.0),
                 ((-INF, NAN), -INF), ((NAN, -5.0), -5.0), ((0.0, -0.0), -0.0), ((-0.0, 0.0), -0.0)]
    MAX_CASES = [((NAN, 1.0), 1.0), ((1.0, NAN), 1.0), ((NAN, NAN), NAN), ((2.0, 3.0), 3.0), ((3.0, 2.0), 3.0),
                 ((INF, NAN), INF), ((NAN, -5.0), -5.0), ((0.0, -0.0), 0.0), ((-0.0, 0.0), 0.0)]

    def test_minimum_gives_the_other_operand_for_a_nan(self):
        spirv = self.module("fmin", fmnmx(False), store(OUT))
        only_numeric_forms(self, spirv, "NMin")
        self.check_float(spirv, self.MIN_CASES)

    def test_maximum_gives_the_other_operand_for_a_nan(self):
        spirv = self.module("fmax", fmnmx(True), store(OUT))
        only_numeric_forms(self, spirv, "NMax")
        self.check_float(spirv, self.MAX_CASES)

    # -- saturation --

    def test_a_saturated_add_of_a_nan_is_zero(self):
        spirv = self.module("fadd_sat", FADD_SAT, store(OUT))
        only_numeric_forms(self, spirv, "NClamp")
        self.check_float(spirv, [((NAN, 1.0), 0.0), ((1.0, NAN), 0.0), ((INF, -INF), 0.0), ((0.5, 0.25), 0.75),
                                 ((2.0, 2.0), 1.0), ((-1.0, -1.0), 0.0), ((INF, 1.0), 1.0)])

    def test_a_saturated_multiply_of_a_nan_is_zero(self):
        spirv = self.module("fmul_sat", FMUL_SAT, store(OUT))
        only_numeric_forms(self, spirv, "NClamp")
        self.check_float(spirv, [((NAN, 1.0), 0.0), ((0.0, INF), 0.0), ((0.5, 0.5), 0.25), ((4.0, 4.0), 1.0),
                                 ((-1.0, 2.0), 0.0)])

    def test_a_saturated_fused_multiply_add_of_a_nan_is_zero(self):
        spirv = self.module("ffma_sat", FFMA_SAT, store(OUT))
        only_numeric_forms(self, spirv, "NClamp")
        self.check_float(spirv, [((NAN, 1.0, 0.0), 0.0), ((1.0, 1.0, NAN), 0.0), ((0.5, 0.5, 0.25), 0.5),
                                 ((3.0, 3.0, 3.0), 1.0), ((-1.0, 3.0, 1.0), 0.0)])

    def test_saturating_constants_that_the_translator_folds_gives_zero_for_a_nan(self):
        # Both operands are immediates, so the optimizer folds the add and the clamp: the stored word is a constant.
        spirv = self.module("folded", mov_imm(10, 0x7FC00000), fadd_imm_sat(OUT, 10, f32_bits(1.0)), store(OUT))
        self.assertEqual(ext_names_used(spirv), set())
        (out,) = evaluate(spirv).stores
        self.assertEqual(out, 0, f"got {out:#010x}")

    # -- float to integer --

    def test_a_float_to_signed_integer_conversion_of_a_nan_is_zero(self):
        spirv = self.module("f2i_s32", f2i(True), store(OUT))
        for x, want in [(NAN, 0), (2.7, 2), (-2.7, -2 & 0xFFFFFFFF), (0.0, 0), (1000.0, 1000)]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [want], x)

    def test_a_float_to_unsigned_integer_conversion_of_a_nan_is_zero(self):
        spirv = self.module("f2i_u32", f2i(False), store(OUT))
        for x, want in [(NAN, 0), (3.9, 3), (-1.0, 0), (4000000000.0, 4000000000)]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [want], x)

    def test_a_float_to_64_bit_integer_conversion_of_a_nan_is_the_top_bit(self):
        spirv = self.module("f2i_u64", F2I_U64, store(OUT), store(OUT + 1, 1))
        for x, want in [(NAN, [0, 0x80000000]), (5.0, [5, 0]), (4294967296.0, [0, 1]), (3.0 * 4294967296.0 + 7.0, [0, 3])]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, want, x)

    def test_a_float_to_byte_conversion_pair_of_a_nan_is_zero(self):
        spirv = self.module("f2ip", F2IP_B, store(OUT))
        for x, want in [(NAN, 0), (300.0, 255), (-4.0, 0), (17.9, 17)]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [want], x)

    # -- half pairs with .RELU --

    def test_relu_of_a_half_pair_keeps_a_nan(self):
        spirv = self.module("hmul2_relu", HMUL2_RELU, store(OUT))
        one = half_bits(1.0)
        a = half_bits(-2.0) | half_bits(NAN) << 16
        b = one | one << 16
        (out,) = evaluate(spirv, a, b).stores
        self.assertEqual(out & 0xFFFF, 0, "the negative low half clamps to zero")
        high = out >> 16
        self.assertTrue(high & 0x7C00 == 0x7C00 and high & 0x3FF != 0, f"the NaN high half became {high:#06x}")
        (out,) = evaluate(spirv, half_bits(3.0) | half_bits(-1.0) << 16, b).stores
        self.assertEqual(out, half_bits(3.0) | 0 << 16)


if __name__ == "__main__":
    unittest.main()
