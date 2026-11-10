# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""NaN operands through the translator: float minimum and maximum, saturation, float to integer conversion.

The console's rules (checked against NVIDIA's PTX documentation for the same operations):

* FMNMX returns the other operand when one is NaN, NaN when both are. (The console also orders -0 below +0; NMin and
  NMax may return either zero unless the module asks for signed zeros, so that rule is not checked here.)
* A saturated result (.SAT) that is NaN is 0.
* F2I of NaN is 0 for a 32-bit integer (or smaller) from a single, 1 << 63 for a 64-bit integer.
* A conversion to an integer saturates to the integer's own range, an infinity included: -2^31 and 2^31 - 1 for a
  signed 32-bit integer, 0 and 2^32 - 1 for an unsigned one, the 16-bit and the 8-bit bounds for those, and 0 and all
  ones for the unsigned 64-bit one.
* HFMA2 and HMUL2 with .RELU leave a NaN a NaN.

Vulkan leaves the result of FMin, FMax and FClamp with a NaN operand to the driver. The programs here are assembled
from instruction words (no game files), read their operands from constant bank 1 so that nothing is folded, and are
translated by the real executable. A small evaluator then executes the module on the CPU with the SPIR-V
specification's meaning of each instruction, and treats an instruction whose result the specification leaves open for
the operands it is given (FMin with a NaN) as an undefined value that must not reach the stored result, so a module that
relies on a driver's luck is red here.

OpConvertFToS and OpConvertFToU are another matter: for a NaN, an infinity and a value outside the integer's range it is
the behaviour that is undefined, not only the result, and a select after the conversion does not take that back. The
evaluator therefore fails at such a conversion itself, whatever the module does with its result: a module that converts
first and selects the right value afterwards stores the same words as one that bounds the float first, and only this
tells them apart. The conversions the translator adds around a texture instruction (the level of detail query's fixed
point, the render scale of a texel coordinate and of a size, the level of a size query) and the 64-bit conversion bound
the float before they convert it, and the cases below give each of them a NaN, both infinities, the floats just past the
range, the bounds themselves and an ordinary value. Those programs are pixel programs: what they read from the pipeline
(the queried level of detail, the size, the render scale) is the evaluator's to give.

F2I does the same for every integer it can give (signed and unsigned, 32, 16 and 8 bits) from a single or a half: the
rounded float is clamped to two floats inside the integer's range before the one conversion, so what is converted is
always a number the integer holds (toward zero, the float is clamped as it is and the conversion does the rounding: the
bounds are whole numbers). The largest float below 2^31 is 2^31 - 128 and the largest below 2^32 is 2^32 - 256,
so a 32-bit result takes its top value from a select on the source being above that float; the bounds of the smaller
integers are floats themselves and need nothing after the conversion. NClamp gives its lower bound for a NaN, which is
the 0 of an unsigned result; a signed one selects its 0.

What a green run says is what the module means. The evaluator gives NClamp, NMin, NMax, OpIsNan and the comparisons the
SPIR-V specification's meaning for a NaN and an infinity; Vulkan lets a driver assume that no operand is either unless
the module declares SignedZeroInfNanPreserve, which these modules do not, and for NClamp, NMin and NMax even then.
What a device does with a NaN is not checked here and has not been measured.

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
GLSL = {1: "Round", 2: "RoundEven", 3: "Trunc", 4: "FAbs", 8: "Floor", 9: "Ceil", 37: "FMin", 38: "UMin", 40: "FMax",
        43: "FClamp", 44: "UClamp", 45: "SClamp", 50: "Fma", 58: "PackHalf2x16", 62: "UnpackHalf2x16", 79: "NMin",
        80: "NMax", 81: "NClamp"}
OPEN_FOR_NAN = {"FMin", "FMax", "FClamp"}   # a NaN operand: which operand is the result is left open


class Undefined(AssertionError):
    """A value the SPIR-V specification leaves open reached a store, or the module ran a conversion whose behaviour the
    specification leaves undefined."""


class Poison:
    """The result of an instruction the specification leaves open for its operands (FMin of a NaN: which operand). It
    flows through arithmetic, a select that does not pick it drops it, and a store of it fails the test. A float to
    integer conversion of a NaN, an infinity or a value out of range is not such a value: the evaluator fails there
    (Evaluator.tolerate_open_conversion)."""

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


def fdiv(x: float, y: float) -> float:
    if y == 0:
        return NAN if x == 0 or math.isnan(x) else math.copysign(INF, x) * math.copysign(1, y)
    return x / y


def fmod(x: float, y: float) -> float:
    """OpFMod: the remainder that has the sign of y."""
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or y == 0:
        return NAN
    r = math.fmod(x, y)
    return r + y if r and (r < 0) != (y < 0) else r


def as_signed(x: int) -> int:
    return x - (1 << 32) if x & 0x80000000 else x


def half_bits(x: float) -> int:
    return struct.unpack("<H", struct.pack("<e", x))[0]


def half_value(b: int) -> float:
    return struct.unpack("<e", struct.pack("<H", b & 0xFFFF))[0]


class Evaluator:
    """Runs the entry point of a module for the operations the programs of this file use: straight-line code, and the
    branches and the calls of the translator's own helper functions.
    Constant bank 1 reads come from `bank1` (word index to 32-bit pattern); every other constant buffer word reads zero.
    The words stored to the storage buffer, or to a pixel program's colour, are collected, in order, in `stores`.

    What a pixel program reads from the pipeline is the test's to give: `lod` is the pair an OpImageQueryLod returns,
    `size` what an OpImageQuerySizeLod returns, `render_scale` the support buffer's scale of every texture and
    `frag_coord` the fragment's position. The texel coordinates of each OpImageFetch are collected in `fetches` and the
    level asked of each OpImageQuerySizeLod in `levels`; an undefined value there makes the instruction's result
    undefined.

    An OpConvertFToS or OpConvertFToU of a NaN, an infinity, a value outside the integer's range or an undefined value
    raises Undefined where it runs: no select after it makes the module defined. `tolerate_open_conversion` gives the
    older reading, an undefined value that a select may drop, which shows what a module that converts first would
    store."""

    tolerate_open_conversion = False
    lod = (0.0, 0.0)
    size = (0, 0)
    render_scale = 1.0
    frag_coord = (0.5, 0.5, 0.0, 1.0)

    def __init__(self, spirv: bytes, bank1: dict[int, int]):
        words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
        self.bank1 = bank1
        self.names: dict[int, str] = {}
        self.types: dict[int, tuple] = {}
        self.pointee: dict[int, int] = {}
        self.vals: dict[int, object] = {}
        self.cells: dict[int, object] = {}
        self.stores: list[int] = []
        self.fetches: list[tuple] = []
        self.levels: list[int] = []
        self.functions: dict[int, list[tuple[int, list[int]]]] = {}
        self.frag_coord_id = None
        self.ext_names: dict[int, str] = {}
        self.ext_used: set[str] = set()
        self.ops_used: set[int] = set()
        entry = None
        body = None                                                # the function being read
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
            elif op in (25, 26, 27):                               # OpTypeImage, OpTypeSampler, OpTypeSampledImage
                self.types[w[0]] = ("o",)
            elif op == 71 and w[1:] == [11, 15]:                   # OpDecorate BuiltIn FragCoord
                self.frag_coord_id = w[0]
            elif op == 44:                                         # OpConstantComposite
                self.vals[w[1]] = tuple(self.vals[x] for x in w[2:])
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
                body = self.functions.setdefault(w[1], [])
            elif op == 56:
                body = None
            elif body is not None:
                body.append((op, w))
            i += n
        self.entry = entry
        self.body = self.functions[entry]

    def run(self) -> list[int]:
        self.call(self.entry, [])
        return self.stores

    def call(self, function: int, args: list):
        """Runs one function: its parameters (pointers to the caller's variables) are bound in order, a branch goes to
        its label, and the value of the OpReturnValue it reaches is the result."""
        body = self.functions[function]
        labels = {w[0]: n for n, (op, w) in enumerate(body) if op == 248}
        args = list(args)
        at = 0
        while at < len(body):
            op, w = body[at]
            at += 1
            self.ops_used.add(op)
            if op == 253:                                          # OpReturn
                return None
            if op == 254:                                          # OpReturnValue
                return self.vals[w[0]]
            if op == 55:                                           # OpFunctionParameter
                self.vals[w[1]] = args.pop(0)
            elif op == 57:                                         # OpFunctionCall
                self.vals[w[1]] = self.call(w[2], [self.vals[x] for x in w[3:]])
            elif op == 249:                                        # OpBranch
                at = labels[w[0]]
            elif op == 250:                                        # OpBranchConditional
                if self.vals[w[0]] is POISON:
                    raise Undefined("a branch on an undefined value")
                at = labels[w[1] if self.vals[w[0]] else w[2]]
            elif op != 247:                                        # OpSelectionMerge says nothing to run
                self.step(op, w)
        return None

    def kind(self, type_id: int) -> str:
        return self.types[type_id][0]

    def step(self, op: int, w: list[int]):
        v = self.vals
        if op == 248:                                              # OpLabel
            return
        if op in (124, 111, 112, 127, 129, 131, 133, 136, 141, 128, 130, 194, 196, 197, 199, 202, 156, 171, 180, 184,
                  79, 80, 81, 87, 95, 100, 103, 105) and any(v.get(x) is POISON for x in w[2:]):
            v[w[1]] = POISON
            return
        if op == 59:                                               # OpVariable
            v[w[1]] = ("var", w[1])
            self.cells[w[1]] = 0.0 if self.kind(self.pointee[w[0]]) == "f" else 0
        elif op == 65:                                             # OpAccessChain
            base = self.names.get(w[2], "")
            idx = [v[x] for x in w[3:]]
            if base in ("cp_c1", "fp_c1"):
                v[w[1]] = ("c1", idx[1] * 4 + idx[2])
            elif base.startswith(("cp_c", "fp_c")):
                v[w[1]] = ("c0",)
            elif base == "support_buffer" and idx[0] == 5:         # the render scales, one a texture
                v[w[1]] = ("scale",)
            elif w[2] == self.frag_coord_id:
                v[w[1]] = ("frag", idx[0])
            else:
                v[w[1]] = ("store", base)
        elif op == 61:                                             # OpLoad
            ptr = v[w[2]]
            if ptr[0] == "var":
                v[w[1]] = self.cells[ptr[1]]
            elif ptr[0] == "c1":
                b = self.bank1.get(ptr[1], 0)
                v[w[1]] = bits_f32(b) if self.kind(w[0]) == "f" else b
            elif ptr[0] == "scale":
                v[w[1]] = self.render_scale
            elif ptr[0] == "frag":
                v[w[1]] = self.frag_coord[ptr[1]]
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
            r = None if x is POISON or math.isnan(x) or math.isinf(x) else int(x)
            if r is None or (op == 109 and not 0 <= r < 1 << 32) or (op == 110 and not -(1 << 31) <= r < 1 << 31):
                if not self.tolerate_open_conversion:             # not representable in the integer: undefined behaviour
                    raise Undefined(f"{'OpConvertFToU' if op == 109 else 'OpConvertFToS'} of {x!r}: a conversion of a value "
                                    "the integer cannot hold is undefined, whatever is selected after it")
                v[w[1]] = POISON
            else:
                v[w[1]] = r & 0xFFFFFFFF
        elif op == 112:                                            # OpConvertUToF
            v[w[1]] = f32(float(v[w[2]]))
        elif op == 111:                                            # OpConvertSToF
            v[w[1]] = f32(float(as_signed(v[w[2]])))
        elif op == 127:                                            # OpFNegate
            v[w[1]] = -v[w[2]]
        elif op in (129, 131, 133, 136, 141):                      # OpFAdd, OpFSub, OpFMul, OpFDiv, OpFMod
            a, b = v[w[2]], v[w[3]]
            v[w[1]] = f32({129: lambda: a + b, 131: lambda: a - b, 133: lambda: a * b, 136: lambda: fdiv(a, b),
                           141: lambda: fmod(a, b)}[op]())
        elif op in (171, 180, 184):                                # OpINotEqual, OpFOrdEqual, OpFOrdLessThan
            a, b = v[w[2]], v[w[3]]
            v[w[1]] = a != b if op == 171 else a == b if op == 180 else a < b
        elif op == 202:                                            # OpBitFieldSExtract: base, offset, count
            field = v[w[2]] >> v[w[3]] & (1 << v[w[4]]) - 1
            v[w[1]] = (field - (1 << v[w[4]]) if field >> v[w[4]] - 1 else field) & 0xFFFFFFFF
        elif op in (128, 130, 194, 196, 197, 199):                 # IAdd, ISub, shifts, Or, And
            a, b = v[w[2]], v[w[3]]
            r = {128: lambda: a + b, 130: lambda: a - b, 194: lambda: a >> b, 196: lambda: a << b,
                 197: lambda: a | b, 199: lambda: a & b}[op]()
            v[w[1]] = r & 0xFFFFFFFF
        elif op == 156:                                            # OpIsNan
            x = v[w[2]]
            v[w[1]] = tuple(math.isnan(e) for e in x) if isinstance(x, tuple) else math.isnan(x)
        elif op == 169:                                            # OpSelect
            c, a, b = v[w[2]], v[w[3]], v[w[4]]
            if c is POISON:
                v[w[1]] = POISON
            elif isinstance(c, tuple):
                v[w[1]] = POISON if a is POISON or b is POISON else tuple(x if k else y for k, x, y in zip(c, a, b))
            else:
                v[w[1]] = a if c else b
        elif op == 79:                                             # OpVectorShuffle
            both = v[w[2]] + v[w[3]]
            v[w[1]] = tuple(both[c] for c in w[4:])
        elif op in (86, 100):                                      # OpSampledImage, OpImage: nothing to compute
            v[w[1]] = None
        elif op == 105:                                            # OpImageQueryLod
            v[w[1]] = self.lod
        elif op == 103:                                            # OpImageQuerySizeLod: image, level
            self.levels.append(v[w[3]])
            v[w[1]] = self.size
        elif op in (87, 95):                                       # OpImageSampleImplicitLod, OpImageFetch: image, coordinates
            if op == 95:
                self.fetches.append(v[w[3]])
            v[w[1]] = (0.0, 0.0, 0.0, 0.0)
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
        if name in ("Trunc", "RoundEven", "Floor", "Ceil"):
            if math.isnan(a[0]) or math.isinf(a[0]):
                return a[0]
            whole = {"Trunc": math.trunc, "RoundEven": round, "Floor": math.floor, "Ceil": math.ceil}[name](a[0])
            return math.copysign(float(whole), a[0])              # a zero result keeps the operand's sign
        if name in ("FMin", "NMin"):
            return nmin(a[0], a[1])
        if name in ("FMax", "NMax"):
            return nmax(a[0], a[1])
        if name in ("FClamp", "NClamp"):
            return nmin(nmax(a[0], a[1]), a[2])
        if name == "FAbs":
            return abs(a[0])
        if name == "UMin":
            return min(a[0], a[1])
        if name == "UClamp":
            return min(max(a[0], a[1]), a[2])
        if name == "SClamp":
            return min(max(as_signed(a[0]), as_signed(a[1])), as_signed(a[2])) & 0xFFFFFFFF
        if name == "Fma":
            return f32(a[0] * a[1] + a[2])
        if name == "UnpackHalf2x16":
            return (half_value(a[0]), half_value(a[0] >> 16))
        if name == "PackHalf2x16":
            return half_bits(a[0][0]) | half_bits(a[0][1]) << 16
        raise AssertionError(f"the evaluator does not know {name}")


def evaluate(spirv: bytes, *operands: int, **pipeline) -> Evaluator:
    """Runs the module with the operands (32-bit patterns) in words 4, 5, 6 of bank 1 (c[1][0x10], c[1][0x14], c[1][0x18])
    and what the pipeline gives a pixel program (the evaluator's lod, size, render_scale, frag_coord). The evaluator's
    tolerate_open_conversion is set the same way."""
    ev = Evaluator(spirv, {4 + n: o for n, o in enumerate(operands)})
    for name, value in pipeline.items():
        setattr(ev, name, value)
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


NEAREST, DOWN, UP, TOWARD_ZERO = range(4)      # F2I's rounding field


def f2i(signed: bool, size: int = 2, rounding: int = TOWARD_ZERO, half: int | None = None, absolute: bool = False,
        negate: bool = False) -> bytes:
    """F2I R20, R10: signedness in bit 72, the integer's size in 75..76 (0, 1, 2: 8, 16, 32 bits), the rounding in 78..79,
    the source's size in 84..85 (1 a half, 2 a single). A half source is the low (`half` 0) or the high (1) half of the
    register, 60..61; bit 62 takes the source's magnitude and bit 63 negates it."""
    low = (half or 0) << 60 | int(absolute) << 62 | int(negate) << 63
    high = (bit(72) if signed else 0) | bit(75, size) | bit(78, rounding) | bit(84, 2 if half is None else 1)
    return scheduled(ampere(0x305, rd=OUT, ra=255, rb=10, low_bits=low, high_bits=high))


# The integers F2I gives: a key, signed, the size field, the bits. And its roundings: the field, a key, what rounds.
F2I_FORMS = [("s32", True, 2, 32), ("u32", False, 2, 32), ("s16", True, 1, 16), ("u16", False, 1, 16),
             ("s8", True, 0, 8), ("u8", False, 0, 8)]
ROUNDINGS = [(NEAREST, "nearest", "RoundEven"), (DOWN, "down", "Floor"), (UP, "up", "Ceil"), (TOWARD_ZERO, "toward_zero", "Trunc")]


def f2i_rule(x: float, signed: bool, bits: int, rounding: int = TOWARD_ZERO) -> int:
    """What the console's conversion leaves in the register, as a 32-bit word: 0 for a NaN, otherwise the single rounded
    as the instruction says and brought to the nearest bound of the integer (a signed result is sign extended)."""
    x = f32(x)
    if math.isnan(x):
        return 0
    low, top = (-(1 << bits - 1), (1 << bits - 1) - 1) if signed else (0, (1 << bits) - 1)
    if math.isinf(x):
        return (top if x > 0 else low) & 0xFFFFFFFF
    whole = (round, math.floor, math.ceil, math.trunc)[rounding](x)        # round: an exact half to the even integer
    return min(max(whole, low), top) & 0xFFFFFFFF


F2I_U64 = scheduled(ampere(0x311, rd=OUT, ra=255, rb=10, high_bits=bit(75, 3) | bit(78, 3) | bit(84, 2)))
F2IP_B = scheduled(ampere(0x243, rd=OUT, ra=255, rb=10, high_bits=bit(74) | bit(78, 3)))        # F2IP.U8.F32.NTZ R20, RZ, R10, RZ, truncating
HMUL2_RELU = scheduled(ampere(0x232, rd=OUT, ra=10, rb=11, high_bits=bit(79)))


# Pixel programs that sample. The texture's handle is the first of the table in c[2] (c[2][0x20]: the word offset in bits
# 40..53, the bank in 54..58), a 2D texture (61..63 = 1), no residency predicate (81..83 = PT); the result mask is 72..75.
HANDLE = (0x20 // 4) << 40 | 2 << 54
TEX_2D = HANDLE | 1 << 61
NO_RESIDENCY = bit(81, 7)
# TMML R0, R10: the level of detail a sample at (R10, R11) would use, two results
TMML = scheduled(ampere(0xB69, rd=0, ra=10, rb=255, low_bits=TEX_2D, high_bits=bit(72, 0x3) | NO_RESIDENCY))
# TLD R0, R10: the texel at the integer coordinates (R10, R11) of level zero (87..89 = 1), four results
TLD = scheduled(ampere(0xB66, rd=0, ra=10, rb=255, low_bits=TEX_2D, high_bits=bit(72, 0xF) | NO_RESIDENCY | bit(87, 1)))
# TXQ R0, RZ: the size of level zero (the query, 62..63, is 0), two results
TXQ = scheduled(ampere(0xB6F, rd=0, ra=255, rb=255, low_bits=HANDLE, high_bits=bit(72, 0x3)))
# TEX R0, R10, R12: a sample at (R10, R11) with the texel offset of R12 (AOFFI, bit 76), four results. The translator makes
# the offset a part of the coordinates: it divides it by the size of the level the sample would use.
TEX_OFFSET = scheduled(ampere(0xB60, rd=0, ra=10, rb=12, low_bits=TEX_2D, high_bits=bit(72, 0xF) | bit(76) | NO_RESIDENCY))


def pixel_program(*instructions: bytes) -> bytes:
    """A pixel program: its header (the two type fields of a pixel program in word 0, one colour target with its four
    components in word 18), three operands from c[1] into R10, R11 and R12, the instructions, an exit. R0 to R3 are the
    colour."""
    header = bytearray(0x80)
    struct.pack_into("<I", header, 0, 2 | 5 << 10)
    struct.pack_into("<I", header, 18 * 4, 0xF)
    head = [mov_cbuf(10, 1, 0x10), mov_cbuf(11, 1, 0x14), mov_cbuf(12, 1, 0x18)]
    return bytes(header) + b"".join(head + list(instructions)) + EXIT


# A float on its way to a signed 32-bit integer. 2^31 is a float and is out of range; the largest float below it is
# 2^31 - 128, and -2^31 is a float and is in range.
TWO_31 = 2147483648.0
BELOW_TWO_31 = 2147483520.0
PAST_MINUS_TWO_31 = -2147483904.0       # the next float below -2^31
INT_MIN = 0x80000000                    # what -2^31 converts to
INT_TOP = 0x7FFFFF80                    # what 2147483520.0 converts to
# And to an unsigned one: 2^32 is out of range, the largest float below it is 2^32 - 256.
TWO_32 = 4294967296.0
BELOW_TWO_32 = 4294967040.0


def mov_imm(rd: int, value: int) -> bytes:
    """MOV Rd, imm32 (0x002 in operand form 4, the immediate in bits 32..63)."""
    return scheduled(ampere(0x802, rd=rd, ra=0, rb=value & 0xFF, low_bits=(value >> 8) << 40))


def fadd_imm_sat(rd: int, ra: int, value: int) -> bytes:
    """FADD.SAT Rd, Ra, imm32 (0x021 in form 4)."""
    return scheduled(ampere(0x821, rd=rd, ra=ra, rb=value & 0xFF, low_bits=(value >> 8) << 40, high_bits=bit(77)))


def without_the_bound(spirv: bytes) -> tuple[bytes, int]:
    """The module with each conversion of a clamped float reading the float the clamp read instead, and how many
    conversions that changed. Nothing else changes: the clamp stays, unused by the conversion, and so does every select
    after the conversion."""
    words = list(struct.unpack(f"<{len(spirv) // 4}I", spirv))
    ext = {}                                                       # the result of a GLSL.std.450 import
    clamped = {}                                                   # the result of an NClamp to its first operand
    changed = 0
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xFFFF
        if op == 11:                                               # OpExtInstImport
            ext[words[i + 1]] = struct.pack(f"<{n - 2}I", *words[i + 2:i + n]).split(b"\0")[0]
        elif op == 12 and ext.get(words[i + 3]) == b"GLSL.std.450" and GLSL.get(words[i + 4]) == "NClamp":
            clamped[words[i + 2]] = words[i + 5]
        elif op in (109, 110) and words[i + 3] in clamped:
            words[i + 3] = clamped[words[i + 3]]
            changed += 1
        i += n
    return struct.pack(f"<{len(words)}I", *words), changed


def same(got: int, want: float) -> bool:
    """A result pattern against the expected float: any NaN for a NaN, the exact pattern (the sign of zero too) otherwise."""
    if math.isnan(want):
        return got & 0x7F800000 == 0x7F800000 and got & 0x7FFFFF != 0
    return got == f32_bits(want)


class NanRulesTest(Harness):
    def module(self, key: str, *instructions: bytes) -> bytes:
        code = self.dir / f"{key}.bin"
        code.write_bytes(program(*instructions))
        return self.translated(key, compute_job(key, code))

    def pixel(self, key: str, *instructions: bytes) -> bytes:
        code = self.dir / f"{key}.bin"
        code.write_bytes(pixel_program(*instructions))
        return self.translated(key, {"key": key, "stage": "ps", "arch": "sm86", "code": str(code), "c1": ""})

    def translated(self, key: str, job: dict) -> bytes:
        got = self.run_jobs([job])[key]
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
                 ((-INF, NAN), -INF), ((NAN, -5.0), -5.0)]
    MAX_CASES = [((NAN, 1.0), 1.0), ((1.0, NAN), 1.0), ((NAN, NAN), NAN), ((2.0, 3.0), 3.0), ((3.0, 2.0), 3.0),
                 ((INF, NAN), INF), ((NAN, -5.0), -5.0)]

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
        only_numeric_forms(self, spirv, "NClamp")   # a NaN lands on the clamp's lower bound; an FClamp would leave it open
        for x, want in [(NAN, 0), (3.9, 3), (-1.0, 0), (4000000000.0, 4000000000)]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [want], x)

    def test_a_float_to_64_bit_integer_conversion_of_a_nan_is_the_top_bit(self):
        spirv = self.module("f2i_u64", F2I_U64, store(OUT), store(OUT + 1, 1))
        for x, want in [(NAN, [0, 0x80000000]), (5.0, [5, 0]), (4294967296.0, [0, 1]), (3.0 * 4294967296.0 + 7.0, [0, 3])]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, want, x)

    def test_a_float_to_byte_conversion_pair_of_a_nan_is_zero(self):
        spirv = self.module("f2ip", F2IP_B, store(OUT))
        only_numeric_forms(self, spirv, "NMax")     # its NaN select predates the change, so the stored value cannot tell
        for x, want in [(NAN, 0), (300.0, 255), (-4.0, 0), (17.9, 17)]:
            self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [want], x)

    # -- F2I: the float is bounded before it is converted, for every integer it can give --

    def check_f2i(self, form: tuple, cases: list[tuple[str, float, int]]):
        """One truncating F2I to the integer `form` names (a row of F2I_FORMS), run for each (name, source, result). The
        expected results are written out; f2i_rule must agree with each, which is what lets the cases of the other
        roundings, the modifiers and the half sources take theirs from it."""
        key, signed, size, bits = form
        spirv = self.module(f"f2i_{key}_range", f2i(signed, size), store(OUT))
        for name, x, want in cases:
            with self.subTest(name, x=x):
                self.assertEqual(f2i_rule(x, signed, bits), want & 0xFFFFFFFF, "the test's own statement of the rule")
                self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [want & 0xFFFFFFFF])

    def test_a_conversion_to_a_signed_32_bit_integer_saturates(self):
        self.check_f2i(F2I_FORMS[0], [
            ("NaN", NAN, 0), ("a NaN with its sign set", bits_f32(0xFFC00001), 0),
            ("+Inf", INF, 0x7FFFFFFF), ("-Inf", -INF, INT_MIN),
            ("2^31, just past the top", TWO_31, 0x7FFFFFFF), ("just past the bottom", PAST_MINUS_TWO_31, INT_MIN),
            ("far past the top", 3e9, 0x7FFFFFFF), ("far past the top", 1e30, 0x7FFFFFFF),
            ("far past the bottom", -3e9, INT_MIN), ("far past the bottom", -1e30, INT_MIN),
            ("the largest float below 2^31", BELOW_TWO_31, INT_TOP), ("-2^31, the bottom bound", -TWO_31, INT_MIN),
            ("the float above -2^31", -BELOW_TWO_31, 0x80000080),
            ("in range", 2.7, 2), ("in range", -2.7, -2), ("in range", 0.0, 0), ("in range", -0.0, 0),
            ("in range", -0.5, 0), ("in range", 16777217.0, 16777216), ("in range", -1000.0, -1000)])

    def test_a_conversion_to_an_unsigned_32_bit_integer_saturates(self):
        self.check_f2i(F2I_FORMS[1], [
            ("NaN", NAN, 0), ("a NaN with its sign set", bits_f32(0xFFC00001), 0),
            ("+Inf", INF, 0xFFFFFFFF), ("-Inf", -INF, 0),
            ("2^32, just past the top", TWO_32, 0xFFFFFFFF), ("far past the top", 1e30, 0xFFFFFFFF),
            ("a negative number", -1.0, 0), ("a negative number", -5.5, 0), ("the smallest negative number", bits_f32(0x80000001), 0),
            ("far below zero", -3e9, 0), ("far below zero", -1e30, 0),
            ("the largest float below 2^32", BELOW_TWO_32, 0xFFFFFF00), ("zero, the bottom bound", 0.0, 0),
            ("in range", -0.0, 0), ("in range", -0.5, 0), ("in range", 3.9, 3), ("in range", TWO_31, 0x80000000),
            ("in range", BELOW_TWO_31, INT_TOP), ("in range", 4000000000.0, 4000000000)])

    def test_a_conversion_to_a_signed_16_bit_integer_saturates(self):
        self.check_f2i(F2I_FORMS[2], [
            ("NaN", NAN, 0), ("+Inf", INF, 32767), ("-Inf", -INF, -32768),
            ("just past the top", 32768.0, 32767), ("just past the bottom", -32769.0, -32768),
            ("far past the top", TWO_31, 32767), ("far past the top", 1e30, 32767),
            ("far past the bottom", PAST_MINUS_TWO_31, -32768), ("far past the bottom", -1e30, -32768),
            ("the top bound", 32767.0, 32767), ("a fraction above the top bound", 32767.5, 32767),
            ("the bottom bound", -32768.0, -32768), ("a fraction below the bottom bound", -32768.5, -32768),
            ("in range", 2.7, 2), ("in range", -2.7, -2), ("in range", 12345.0, 12345), ("in range", -0.0, 0)])

    def test_a_conversion_to_an_unsigned_16_bit_integer_saturates(self):
        self.check_f2i(F2I_FORMS[3], [
            ("NaN", NAN, 0), ("a NaN with its sign set", bits_f32(0xFFC00001), 0), ("+Inf", INF, 65535), ("-Inf", -INF, 0),
            ("just past the top", 65536.0, 65535), ("far past the top", TWO_32, 65535), ("far past the top", 1e30, 65535),
            ("a negative number", -1.0, 0), ("a negative number", -40000.0, 0), ("far below zero", -1e30, 0),
            ("the top bound", 65535.0, 65535), ("a fraction above the top bound", 65535.5, 65535),
            ("zero, the bottom bound", 0.0, 0),
            ("in range", 3.9, 3), ("in range", 40000.0, 40000), ("in range", -0.5, 0)])

    def test_a_conversion_to_a_signed_8_bit_integer_saturates(self):
        self.check_f2i(F2I_FORMS[4], [
            ("NaN", NAN, 0), ("+Inf", INF, 127), ("-Inf", -INF, -128),
            ("just past the top", 128.0, 127), ("just past the bottom", -129.0, -128),
            ("past the 16-bit range", 40000.0, 127), ("past the 16-bit range", -40000.0, -128),
            ("far past the top", TWO_31, 127), ("far past the bottom", -1e30, -128),
            ("the top bound", 127.0, 127), ("a fraction above the top bound", 127.5, 127),
            ("the bottom bound", -128.0, -128), ("a fraction below the bottom bound", -128.5, -128),
            ("in range", 2.7, 2), ("in range", -2.7, -2), ("in range", -100.0, -100)])

    def test_a_conversion_to_an_unsigned_8_bit_integer_saturates(self):
        self.check_f2i(F2I_FORMS[5], [
            ("NaN", NAN, 0), ("+Inf", INF, 255), ("-Inf", -INF, 0),
            ("just past the top", 256.0, 255), ("past the 16-bit range", 70000.0, 255), ("far past the top", 1e30, 255),
            ("a negative number", -1.0, 0), ("far below zero", -1e30, 0),
            ("the top bound", 255.0, 255), ("a fraction above the top bound", 255.5, 255), ("zero, the bottom bound", 0.0, 0),
            ("in range", 3.9, 3), ("in range", 200.0, 200)])

    # what every rounding is given: the classes of the cases above, and the halves that tell the roundings apart
    ROUNDED = [NAN, INF, -INF, TWO_31, BELOW_TWO_31, -TWO_31, PAST_MINUS_TWO_31, TWO_32, BELOW_TWO_32, 1e30, -1e30,
               65535.5, 65534.5, 65535.0, 32767.5, -32768.5, 255.5, 254.5, 127.5, -128.5, -1.0, -0.5, -0.25, 0.0, 0.25,
               0.5, 1.5, 2.5, 3.5, -1.5, -2.5, 2.7, -2.7, 8388607.5, -8388607.5]

    def test_every_rounding_of_a_conversion_saturates(self):
        """The float is rounded as the instruction says (to the nearest, an exact half to the even one; down; up; toward
        zero) and the rounded value is what saturates: 65535.5 to the nearest is 65536, past an unsigned 16-bit integer."""
        for key, signed, size, bits in F2I_FORMS:
            for rounding, name, _ in ROUNDINGS:
                spirv = self.module(f"f2i_{key}_{name}", f2i(signed, size, rounding), store(OUT))
                for x in self.ROUNDED:
                    with self.subTest(key, rounding=name, x=x):
                        self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [f2i_rule(x, signed, bits, rounding)])

    def test_the_rule_the_rounding_cases_use_rounds_as_the_console_does(self):
        self.assertEqual([f2i_rule(x, True, 32, NEAREST) for x in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.7)], [0, 2, 2, 0, 0xFFFFFFFE, 0xFFFFFFFE, 3])
        self.assertEqual([f2i_rule(x, True, 32, DOWN) for x in (2.7, -2.7, -0.25)], [2, 0xFFFFFFFD, 0xFFFFFFFF])
        self.assertEqual([f2i_rule(x, True, 32, UP) for x in (2.2, -2.7, 0.25)], [3, 0xFFFFFFFE, 1])
        self.assertEqual([f2i_rule(x, False, 16, NEAREST) for x in (65535.5, 65534.5, -0.5, INF, NAN)], [65535, 65534, 0, 65535, 0])
        self.assertEqual([f2i_rule(x, True, 8, DOWN) for x in (-128.5, 127.5, -INF)], [0xFFFFFF80, 127, 0xFFFFFF80])

    def test_a_conversion_of_a_magnitude_or_a_negation_saturates(self):
        """The instruction's own |x| and -x are applied before anything else: the magnitude of -Inf is past the top, the
        negation of 2^31 is the bottom bound itself, and neither makes a NaN a number."""
        values = [NAN, INF, -INF, TWO_31, -TWO_31, BELOW_TWO_31, PAST_MINUS_TWO_31, TWO_32, -TWO_32, 2.7, -2.7, 5.5, -5.5, 0.0]
        for key, signed, size, bits in (F2I_FORMS[0], F2I_FORMS[1], F2I_FORMS[3]):
            for absolute, negate in ((True, False), (False, True), (True, True)):
                spirv = self.module(f"f2i_{key}_abs{absolute:d}_neg{negate:d}", f2i(signed, size, absolute=absolute, negate=negate), store(OUT))
                for x in values:
                    y = abs(x) if absolute else x
                    with self.subTest(key, absolute=absolute, negate=negate, x=x):
                        self.assertEqual(evaluate(spirv, f32_bits(x)).stores, [f2i_rule(-y if negate else y, signed, bits)])
        self.assertEqual(f2i_rule(abs(-INF), True, 32), 0x7FFFFFFF)
        self.assertEqual(f2i_rule(-TWO_31, True, 32), INT_MIN)
        self.assertEqual(f2i_rule(-5.5, False, 32), 0)

    def test_a_conversion_of_a_half_saturates(self):
        """A half source is one half of the register, widened. Every finite half is inside a 32-bit integer, so only its
        infinities and its NaN meet a 32-bit bound; the 16-bit and the 8-bit integers saturate from 32768 or 256 up."""
        largest = 65504.0
        values = [NAN, INF, -INF, largest, -largest, 32768.0, -32768.0, 300.0, -300.0, 2.5, -2.5, 0.0]
        for key, signed, size, bits in F2I_FORMS:
            for half in (0, 1):
                spirv = self.module(f"f2i_{key}_half{half}", f2i(signed, size, half=half), store(OUT))
                for x in values:
                    other = 1.0 if math.isnan(x) else NAN          # the half the instruction does not read
                    pair = (half_bits(x), half_bits(other)) if half == 0 else (half_bits(other), half_bits(x))
                    with self.subTest(key, half=half, x=x):
                        self.assertEqual(evaluate(spirv, pair[0] | pair[1] << 16).stores, [f2i_rule(x, signed, bits)])
        self.assertEqual([f2i_rule(x, True, 32) for x in (INF, -INF, NAN, largest)], [0x7FFFFFFF, INT_MIN, 0, 65504])
        self.assertEqual([f2i_rule(x, False, 16) for x in (INF, -INF, NAN, largest, -2.5)], [65535, 0, 0, 65504, 0])
        self.assertEqual([f2i_rule(x, True, 8) for x in (INF, -INF, NAN, largest, -300.0)], [127, 0xFFFFFF80, 0, 127, 0xFFFFFF80])

    def test_the_conversion_is_one_clamp_and_the_selects_the_clamp_cannot_give(self):
        """What a conversion costs, as the instructions of its module that compute something: the rounding, one NClamp,
        one conversion; then, for a 32-bit integer, a comparison of the source with the clamp's upper bound and a select
        of the integer's top value (no float is that value); then, for a signed integer, a NaN test and a select of 0 (a
        NaN lands on the clamp's lower bound, which is 0 only for an unsigned one). No integer clamp follows a 16-bit or
        an 8-bit conversion, and none of them branches. Toward zero has no rounding instruction: the conversion rounds
        that way itself, and the clamp's bounds are whole numbers, so the clamped float truncates to what the truncated
        float clamps to."""
        top, nan = [184, 169], [156, 169]       # OpFOrdLessThan, OpSelect; OpIsNan, OpSelect
        after = {"s32": top + nan, "u32": top, "s16": nan, "u16": [], "s8": nan, "u8": []}
        for key, signed, size, bits in F2I_FORMS:
            for rounding, name, instruction in ROUNDINGS:
                with self.subTest(key, rounding=name):
                    ev = evaluate(self.module(f"f2i_{key}_{name}_shape", f2i(signed, size, rounding), store(OUT)), f32_bits(2.5))
                    shape = [GLSL[w[3]] if op == 12 else op for op, w in ev.body
                             if op in (12, 109, 110, 156, 169, 245, 250, 251) or 164 <= op <= 168 or 180 <= op <= 191]
                    rounds = [] if rounding == TOWARD_ZERO else [instruction]
                    self.assertCountEqual(shape, rounds + ["NClamp", 110 if signed else 109] + after[key])
                    self.assertEqual(ev.stores, [f2i_rule(2.5, signed, bits, rounding)])

    def test_a_conversion_reading_the_unclamped_float_fails_whatever_is_selected_after_it(self):
        """What the cases above rest on: each module with its conversion made to read the rounded float itself, not the
        clamped one, keeps every select and still stores the right word for a number in range, and the evaluator fails
        at the conversion of anything else."""
        outside = {"s32": (NAN, INF, -INF, TWO_31, PAST_MINUS_TWO_31), "u32": (NAN, INF, -INF, TWO_32, -1.0),
                   "s16": (NAN, INF, -INF, 1e30), "u16": (NAN, INF, -INF, -1.0, 1e30),
                   "s8": (NAN, INF, -INF, -1e30), "u8": (NAN, INF, -INF, -1.0, 1e30)}
        for key, signed, size, bits in F2I_FORMS:
            spirv, changed = without_the_bound(self.module(f"f2i_{key}_unbounded", f2i(signed, size), store(OUT)))
            self.assertEqual(changed, 1, key)
            self.assertEqual(evaluate(spirv, f32_bits(2.7)).stores, [2], key)
            for x in outside[key]:
                with self.subTest(key, x=x), self.assertRaisesRegex(Undefined, "OpConvertFTo[SU] of"):
                    evaluate(spirv, f32_bits(x))

    # -- the conversions around F2I: the float is bounded before it is converted --

    def test_a_conversion_of_the_unbounded_float_fails_whatever_is_selected_after_it(self):
        """What the cases below rest on. The level of detail query's module with its two conversions made to read the
        product itself, not the clamped product, still selects 0 for a NaN, so the words it stores for a NaN are the
        right ones; it is the conversion of the NaN that is undefined, and the evaluator fails there. The same goes for
        an infinity and a value past the range, which the stored words would give away as well."""
        spirv, changed = without_the_bound(self.pixel("tmml_unbounded", TMML))
        self.assertEqual(changed, 2)
        self.assertEqual(evaluate(spirv, lod=(1.5, 2.25)).stores, [0x240, 0x180, 0, 0])
        self.assertEqual(evaluate(spirv, lod=(NAN, 2.25), tolerate_open_conversion=True).stores, [0x240, 0, 0, 0])
        for x in (NAN, INF, -INF, 8388608.0, -8388609.0):
            for lod in ((x, 2.25), (2.25, x)):
                with self.subTest(lod=lod), self.assertRaisesRegex(Undefined, "OpConvertFToS of"):
                    evaluate(spirv, lod=lod)

    def test_the_64_bit_conversion_saturates_and_converts_nothing_out_of_range(self):
        spirv = self.module("f2i_u64_range", F2I_U64, store(OUT), store(OUT + 1, 1))
        ones = [0xFFFFFFFF, 0xFFFFFFFF]
        cases = [("NaN", NAN, [0, 0x80000000]), ("+Inf", INF, ones), ("-Inf", -INF, [0, 0]),
                 ("2^64, just past the top", 2.0 ** 64, ones), ("past the top", 3e19, ones),
                 ("-1, just past the bottom", -1.0, [0, 0]), ("past the bottom", -3e19, [0, 0]),
                 ("the largest float below 2^64", bits_f32(0x5F7FFFFF), [0, 0xFFFFFF00]), ("zero, the bottom bound", 0.0, [0, 0]),
                 ("in range", 5.0, [5, 0]), ("in range", 2.0 ** 63, [0, 0x80000000]), ("in range", 2.0 ** 40 + 2.0 ** 20, [1 << 20, 1 << 8])]
        for name, x, want in cases:
            with self.subTest(name, x=x):
                self.assertEqual(evaluate(spirv, f32_bits(x)).stores, want)

    def test_the_level_of_detail_query_bounds_its_fixed_point_results(self):
        """TMML gives the two levels of detail a sample would use, times 256, as signed integers. Either can be anything a
        driver's query returns, so each is bounded on both sides: a NaN gives 0, the rest the nearest bound."""
        spirv = self.pixel("tmml", TMML)
        cases = [("NaN", NAN, 0), ("+Inf", INF, INT_TOP), ("-Inf", -INF, INT_MIN),
                 ("just past the top", 8388608.0, INT_TOP),             # times 256: 2^31
                 ("just past the bottom", -8388609.0, INT_MIN),         # times 256: the next float below -2^31
                 ("the top bound", 8388607.5, INT_TOP),                 # times 256: 2147483520
                 ("the bottom bound", -8388608.0, INT_MIN),
                 ("in range", 1.5, 0x180), ("in range", -0.75, -192 & 0xFFFFFFFF)]
        for name, x, want in cases:
            for component in (0, 1):        # of the query's pair; the program's two results are in the other order
                with self.subTest(name, x=x, component=component):
                    stores = evaluate(spirv, lod=(x, 2.25) if component == 0 else (2.25, x)).stores
                    self.assertEqual(stores, [0x240, want, 0, 0] if component == 0 else [want, 0x240, 0, 0])

    def texel(self, spirv: bytes, coordinate: int, **pipeline) -> tuple:
        """The coordinates the texel fetch of TLD's module is given, for the same integer in both of the program's."""
        (fetched,) = evaluate(spirv, coordinate & 0xFFFFFFFF, coordinate & 0xFFFFFFFF, **pipeline).fetches
        return fetched

    def test_a_scaled_texel_coordinate_is_bounded(self):
        """With a render scale other than 1 the integer coordinates of a texel fetch are multiplied by it and converted
        back, in a helper function of the translator's own."""
        spirv = self.pixel("tld", TLD)
        cases = [("NaN", 1, NAN, 0), ("NaN", 0, INF, 0), ("+Inf", 1, INF, INT_TOP), ("-Inf", -1, INF, INT_MIN),
                 ("just past the top", 1, TWO_31, INT_TOP), ("just past the bottom", -1, -PAST_MINUS_TWO_31, INT_MIN),
                 ("the top bound", 1, BELOW_TWO_31, INT_TOP), ("the bottom bound", -1, TWO_31, INT_MIN),
                 ("in range", 5, 2.0, 10), ("in range", -3, 2.5, -7 & 0xFFFFFFFF), ("no scale", 7, 1.0, 7)]
        for name, coordinate, scale, want in cases:
            with self.subTest(name, coordinate=coordinate, scale=scale):
                self.assertEqual(self.texel(spirv, coordinate, render_scale=scale), (want, want))

    def test_a_scaled_texel_coordinate_of_a_negative_scale_is_bounded(self):
        """A negative render scale takes the helper's other path, in a pixel program: the coordinate times the scale's
        magnitude, plus the fragment's own position within the scaled texel."""
        spirv = self.pixel("tld_negative", TLD)
        cases = [("NaN", 0, -INF, 0), ("+Inf", 1, -INF, INT_TOP), ("-Inf", -1, -INF, INT_MIN),
                 ("just past the top", 1, -TWO_31, INT_TOP), ("just past the bottom", -1, PAST_MINUS_TWO_31, INT_MIN),
                 ("the top bound", 1, -BELOW_TWO_31, INT_TOP), ("the bottom bound", -1, -TWO_31, INT_MIN),
                 ("in range", 5, -2.0, 10), ("in range", -3, -2.5, -7 & 0xFFFFFFFF)]
        for name, coordinate, scale, want in cases:
            with self.subTest(name, coordinate=coordinate, scale=scale):
                self.assertEqual(self.texel(spirv, coordinate, render_scale=scale), (want, want))
        # x of the fragment's position for the first coordinate, y for the second
        self.assertEqual(self.texel(spirv, 5, render_scale=-2.0, frag_coord=(3.0, 6.5, 0.0, 1.0)), (11, 10))

    def test_an_unscaled_texture_size_is_bounded(self):
        """With a render scale other than 1 a queried size is divided by the scale's magnitude and converted back, in a
        helper function. A size is not negative, so the quotient is not either: zero is the lower bound, which is also
        where a NaN (0 / 0) goes, and there is no -Inf to give."""
        spirv = self.pixel("txq", TXQ)
        cases = [("NaN", (0, 0), 0.0, [0, 0]), ("NaN", (16, 8), NAN, [0, 0]), ("+Inf", (16, 8), 0.0, [INT_TOP, INT_TOP]),
                 ("just past the top", (16, 8), 2.0 ** -27, [INT_TOP, 1 << 30]),
                 ("the top bound", (16777215, 1), 2.0 ** -7, [INT_TOP, 128]),
                 ("zero, the bottom bound", (0, 4), 0.5, [0, 8]),
                 ("in range", (16, 8), 2.0, [8, 4]), ("in range", (16, 8), -2.0, [8, 4]), ("in range", (16, 9), INF, [0, 0]),
                 ("no scale", (16, 9), 1.0, [16, 9])]
        for name, size, scale, want in cases:
            with self.subTest(name, size=size, scale=scale):
                self.assertEqual(evaluate(spirv, size=size, render_scale=scale).stores, want + [0, 0])

    def test_the_level_of_a_size_query_is_bounded(self):
        """A sample with a texel offset asks which level the sample would use, converts it to an integer and asks that
        level's size, once a coordinate. A level is not negative, so zero is the lower bound (and a NaN's value)."""
        spirv = self.pixel("tex_offset", TEX_OFFSET)
        cases = [("NaN", NAN, 0), ("+Inf", INF, INT_TOP), ("-Inf", -INF, 0),
                 ("just past the top", TWO_31, INT_TOP), ("past the bottom", PAST_MINUS_TWO_31, 0),
                 ("the top bound", BELOW_TWO_31, INT_TOP), ("zero, the bottom bound", 0.0, 0),
                 ("in range", 2.0, 2), ("in range", 3.75, 3)]
        for name, x, want in cases:
            with self.subTest(name, x=x):
                ev = evaluate(spirv, f32_bits(0.5), f32_bits(0.5), 0x21, lod=(x, 0.0), size=(16, 8))
                self.assertEqual(ev.levels, [want, want])

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
