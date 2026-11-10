#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""What one HMMA costs on this GPU in the translator's forms, against VK_KHR_cooperative_matrix.

    py -3 tools/hmmacost/hmma_cost.py info
    py -3 tools/hmmacost/hmma_cost.py run --out DIR [--count N] [--reps N] [--groups N]
                                          [--coop-groups N] [--only NAME,...] [--validate]
                                          [--mutate shuffle|scratch|coop-layout]

Architecture D20b translates Ampere's warp-wide matrix instructions (HMMA, LDSM, MOVM) per lane
and moves HMMA to VK_KHR_cooperative_matrix if that costs too much (spike S8). This tool measures
the cost per instruction. It assembles small Ampere compute programs from instruction words (no
game code), translates them with the project's translator (build/xlate), runs them on the GPU
through vkhmmacost (tools/build_vkdraw.py --harness vkhmmacost), and checks every output word
against a CPU model of the same instructions written from NVIDIA's documented fragment layouts
(PTX mma.m16n8k16, ldmatrix, movmatrix), so that the form timed is a correct one.

Variants (each at --count instructions per lane and at half of it; the cost per instruction is
the difference of the two medians over the difference of the instruction counts):
  scratch       HMMA.16816 with an f16 accumulator in the translator's scratch-memory helper,
                the form every HMMA of the game's tensor programs took before they ran on
                cooperative matrices (translated with the job option hmma = "helper"; the
                game's modules keep the helper for the runtime's fallback)
  shuffle       HMMA.16816 with an f32 accumulator: the translator gathers the fragments with
                subgroup shuffles, the form D20b describes
  *-plain       the same without the per-HMMA operand perturbation (a uniform XOR of the low
                mantissa bits), which shows what the driver removes when operands repeat
  coop-helper   the scratch helper's contract with a cooperative multiply-add as its body
                (coop_helper.comp), the smallest change D20b's move would make
  coop-f16, coop-f32
                operands held in cooperative matrices throughout (coop_direct.comp), the
                ceiling of a translator that maps the fragment chains onto the extension
  ldsm, movm    LDSM.16.M88.4 (normal and transposed in turn) and MOVM.16.MT88, each with its
                integer bookkeeping

Out of DIR: every program, module, job file and output, `report.json` (device, clocks before and
after each variant from nvidia-smi, machine load, every dispatch time, the checks) and
`summary.tsv`. `--validate` first runs each variant with the validation layer and fails on any
error. `--mutate` is a counter-run: shuffle makes one subgroup shuffle of the shuffle variant read
a fixed lane, scratch swaps two stores of the helper in the scratch variant,
coop-layout loads B with the wrong layout in coop-helper, coop-f16 and coop-f32; each must fail its
check.

Exit codes: 0 measured and every check passed, 1 a check failed (or validation reported an
error), 2 a tool is missing.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import pool  # noqa: E402

M32 = 0xFFFFFFFF
LOCAL = 128              # invocations per workgroup, as the game's tensor dispatches (the played programs census)
WARPS = LOCAL // 32
IN_ADDR, OUT_ADDR = 0x10000000, 0x20000000   # guest addresses of the two storage buffers (16-aligned)
SB_IN, SB_OUT = 0x100, 0x110                 # c[0] byte offsets of their address and size pairs
IN_WORDS = 40            # per invocation: A 0..15, B 16..23, accumulators 24..39
SALT_A, SALT_B = 0x9E3779B1, 0x85EBCA77

# --- Ampere (sm_86) instruction words ------------------------------------------------------------
#
# Field positions are the vendored decoder's (xlate/third_party/ryujinx/.../Instructions/InstEmitAmpere*.cs,
# after Mesa NAK's sm70 encoder): opcode 0..11 (an ALU opcode's form in 9..11), guard predicate 12..14
# (PT = 7), Rd 16..23, Ra 24..31, Rb or a 32-bit immediate 32..63, Rc 64..71. Every instruction carries
# the scheduling field "stall 1, no barriers" (bits 104..115), without which the translator hands a
# result only to readers that wait on barrier 0 (tests/test_xlate_uni.py, NO_BARRIERS).

RZ = 255
NO_BARRIERS = 0xFE2 << 40
SR_LANEID, SR_TID_X, SR_CTAID_X = 0, 33, 37
LS_B128 = 6
LUT_XOR_AB = 0xF0 ^ 0xCC


def word(op: int, rd: int = RZ, ra: int = RZ, rb: int = RZ, rc: int = RZ, low: int = 0, high: int = 0) -> bytes:
    """One instruction: low holds fields of bits 0..63 above Rb, high fields of bits 64..127 as bit numbers."""
    lo = op | (7 << 12) | (rd << 16) | (ra << 24) | (rb << 32) | low
    hi = rc | (high >> 64) | NO_BARRIERS
    return struct.pack("<QQ", lo & ((1 << 64) - 1), hi & ((1 << 64) - 1))


def at(n: int, value: int = 1) -> int:
    return value << n


def i_s2r(rd: int, sr: int) -> bytes:
    return word(0x919, rd=rd, high=at(72, sr))


def i_mov_c0(rd: int, offset: int) -> bytes:
    """MOV Rd, c[0][offset] (MOV 0x002 in operand form 5, bank 54..59, byte offset 38..53)."""
    return word(0xA02, rd=rd, ra=0, rb=0, low=offset << 38)


def i_imad(rd: int, ra: int, imm: int, rc: int) -> bytes:
    """IMAD Rd = Ra * imm + Rc (form 4); the predicate output 81..83 is PT."""
    return word(0x824, rd=rd, ra=ra, rb=0, rc=rc, low=(imm & M32) << 32, high=at(81, 7))


def i_imad_wide(rd: int, ra: int, imm: int, rc: int) -> bytes:
    """IMAD.WIDE.U32 {Rd+1:Rd} = Ra * imm + {Rc+1:Rc} (form 4)."""
    return word(0x825, rd=rd, ra=ra, rb=0, rc=rc, low=(imm & M32) << 32, high=at(81, 7))


def i_xor_imm(rd: int, ra: int, imm: int) -> bytes:
    """LOP3 Rd = Ra ^ imm, RZ (form 4, table 72..79), no predicate output (81..83 = PT)."""
    return word(0x812, rd=rd, ra=ra, rb=0, rc=RZ, low=(imm & M32) << 32, high=at(72, LUT_XOR_AB) | at(81, 7))


def i_xor_reg(rd: int, ra: int, rb: int) -> bytes:
    """LOP3 Rd = Ra ^ Rb, RZ (form 1)."""
    return word(0x212, rd=rd, ra=ra, rb=rb, rc=RZ, high=at(72, LUT_XOR_AB) | at(81, 7))


def i_ldg128(rd: int, ra: int, offset: int) -> bytes:
    """LDG.E.128 Rd, [{Ra+1:Ra} + offset] (.E 72, size 73..76)."""
    return word(0x381, rd=rd, ra=ra, rb=0, low=(offset & 0xFFFFFF) << 40, high=at(72) | at(73, LS_B128))


def i_stg128(ra: int, rb: int, offset: int) -> bytes:
    """STG.E.128 [{Ra+1:Ra} + offset], Rb."""
    return word(0x386, ra=ra, rb=rb, low=(offset & 0xFFFFFF) << 40, high=at(72) | at(73, LS_B128))


def i_sts128(ra: int, rb: int, offset: int) -> bytes:
    """STS.128 [Ra + offset], Rb (no stride, 78..79 = 0)."""
    return word(0x388, ra=ra, rb=rb, low=(offset & 0xFFFFFF) << 40, high=at(73, LS_B128))


def i_bar() -> bytes:
    return word(0xB1D)


def i_hmma(rd: int, ra: int, rb: int, rc: int, f32: bool) -> bytes:
    """HMMA.16816 (size bit 75 set, 78..79 clear), f32 accumulator bit 76, f16 sources (82..84 = 0)."""
    return word(0x23C, rd=rd, ra=ra, rb=rb, rc=rc, high=at(75) | (at(76) if f32 else 0))


def i_ldsm4(rd: int, ra: int, offset: int, trans: bool) -> bytes:
    """LDSM.16.M88.4 (count 72..73 = 2) or .MT88.4 (mode 78..79 = 1), Rd..Rd+3, [Ra + offset]."""
    return word(0x83B, rd=rd, ra=ra, rb=0, low=(offset & 0xFFFFFF) << 40, high=at(72, 2) | (at(78) if trans else 0))


def i_movm(rd: int, ra: int) -> bytes:
    return word(0x23A, rd=rd, ra=ra)


EXIT = struct.pack("<QQ", 0x94D | (7 << 12), (7 << (87 - 64)) | NO_BARRIERS)


def mask(i: int, salt: int) -> int:
    """A per-instruction operand perturbation: 5 low mantissa bits of both halves of a word, never zero."""
    h = ((((i + 1) * salt) & M32) >> 11) & 0x1F
    h = h or 1
    return h | (h << 16)


# --- programs: instruction words and the operations the CPU model runs ------------------------------

@dataclass
class Program:
    name: str
    count: int                       # instructions of the measured kind per lane
    kind: str                        # hmma, ldsm, movm
    out_words: int                   # per invocation
    shared: int = 0
    code: list = field(default_factory=list)
    ops: list = field(default_factory=list)

    def emit(self, words: bytes, op: tuple):
        self.code.append(words)
        self.ops.append(op)

    def binary(self) -> bytes:
        return b"".join(self.code) + EXIT


def prologue(p: Program):
    """Registers: R0 tid, R1 workgroup, R2:R3 input, R4:R5 output, R6:R7 this invocation's input,
    R8:R9 its output, R16..R55 its 40 input words."""
    p.emit(i_s2r(0, SR_TID_X), ("s2r", 0, SR_TID_X))
    p.emit(i_s2r(1, SR_CTAID_X), ("s2r", 1, SR_CTAID_X))
    for rd, off in ((2, SB_IN), (3, SB_IN + 4), (4, SB_OUT), (5, SB_OUT + 4)):
        p.emit(i_mov_c0(rd, off), ("c0", rd, off))
    p.emit(i_imad_wide(6, 0, 4 * IN_WORDS, 2), ("imad_wide", 6, 0, 4 * IN_WORDS, 2))
    p.emit(i_imad(10, 1, LOCAL, 0), ("imad", 10, 1, LOCAL, 0))
    p.emit(i_imad_wide(8, 10, 4 * p.out_words, 4), ("imad_wide", 8, 10, 4 * p.out_words, 4))
    for k in range(IN_WORDS // 4):
        p.emit(i_ldg128(16 + 4 * k, 6, 16 * k), ("ldg128", 16 + 4 * k, 6, 16 * k))


def epilogue(p: Program, first: int):
    for k in range(p.out_words // 4):
        p.emit(i_stg128(8, first + 4 * k, 16 * k), ("stg128", 8, first + 4 * k, 16 * k))


def hmma_program(count: int, f32: bool, perturb: bool) -> Program:
    """count HMMA on four accumulator chains: HMMA i adds A[i % 4] x B[(i / 4) % 4] into chain i % 4.
    A sets at R16, R20, R24, R28; B sets at R32..R39; chains at R40 (2 words each, or 4 with f32)."""
    acc = 4 if f32 else 2
    p = Program(f"{'shuffle' if f32 else 'scratch'}{'' if perturb else '-plain'}", count, "hmma", 4 * acc)
    prologue(p)
    for i in range(count):
        s, y = i % 4, (i // 4) % 4
        ra, rb, rc = 16 + 4 * s, 32 + 2 * y, 40 + acc * s
        if perturb:
            for q in range(4):
                p.emit(i_xor_imm(60 + q, ra + q, mask(i, SALT_A)), ("xor_imm", 60 + q, ra + q, mask(i, SALT_A)))
            for q in range(2):
                p.emit(i_xor_imm(64 + q, rb + q, mask(i, SALT_B)), ("xor_imm", 64 + q, rb + q, mask(i, SALT_B)))
            ra, rb = 60, 64
        p.emit(i_hmma(rc, ra, rb, rc, f32), ("hmma", rc, ra, rb, rc, f32))
    epilogue(p, 40)
    return p


def ldsm_program(count: int) -> Program:
    """The invocation's 16 A words go to shared memory at 64 tid (8 KiB); then count LDSM.x4, each
    with its lanes' row addresses permuted (16 lane XOR i % 32) at offset 512 ((i / 32) % 16), the
    even ones normal and the odd ones transposed, each result XORed into R40..R43."""
    p = Program("ldsm", count, "ldsm", 4, shared=8192)
    prologue(p)
    p.emit(i_imad(11, 0, 64, RZ), ("imad", 11, 0, 64, RZ))
    for k in range(4):
        p.emit(i_sts128(11, 16 + 4 * k, 16 * k), ("sts128", 11, 16 + 4 * k, 16 * k))
    p.emit(i_bar(), ("bar",))
    p.emit(i_s2r(12, SR_LANEID), ("s2r", 12, SR_LANEID))
    p.emit(i_imad(13, 12, 16, RZ), ("imad", 13, 12, 16, RZ))
    for i in range(count):
        perm, offset, trans = 16 * (i % 32), 512 * ((i // 32) % 16), bool(i % 2)
        p.emit(i_xor_imm(14, 13, perm), ("xor_imm", 14, 13, perm))
        p.emit(i_ldsm4(44, 14, offset, trans), ("ldsm4", 44, 14, offset, trans))
        for q in range(4):
            p.emit(i_xor_reg(40 + q, 40 + q, 44 + q), ("xor_reg", 40 + q, 40 + q, 44 + q))
    epilogue(p, 40)
    return p


def movm_program(count: int) -> Program:
    """Four chains R40..R43: MOVM i transposes chain i % 4 after XORing it with a mask."""
    p = Program("movm", count, "movm", 4)
    prologue(p)
    for i in range(count):
        r = 40 + i % 4
        p.emit(i_xor_imm(r, r, mask(i, SALT_A)), ("xor_imm", r, r, mask(i, SALT_A)))
        p.emit(i_movm(r, r), ("movm", r, r))
    epilogue(p, 40)
    return p


# --- the CPU model ---------------------------------------------------------------------------------
#
# One workgroup of LOCAL invocations, registers as uint32 arrays over the invocations. The warp-wide
# instructions follow NVIDIA's fragment layouts for mma.m16n8k16 with f16 sources: lane L, g = L / 4,
# t = L % 4. A register r holds row g + 8 (r & 1) at columns 8 (r >> 1) + 2t and + 1; B register r
# rows (k) 8 r + 2t and + 1 of column g; an f16 C / D register r row g + 8 r at columns 2t, 2t + 1;
# an f32 C / D register r row g + 8 (r >> 1), column 2t + (r & 1). The low half of a word is the
# lower index. An HMMA adds the 16 products of an output in k order to the f32 value of C, each
# product exact in f32, one rounding per addition (the order of both translated forms), then rounds
# to f16 (nearest even) for an f16 result.

def halves(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = (w & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (w >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    return lo, hi


def pack(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    l16 = lo.astype(np.float16).view(np.uint16).astype(np.uint32)
    h16 = hi.astype(np.float16).view(np.uint16).astype(np.uint32)
    return l16 | (h16 << 16)


def warp_hmma(regs: dict, w: slice, rd: int, ra: int, rb: int, rc: int, f32: bool, chain: dict | None = None):
    """chain: when given, an f16 result also keeps its f32 value, and an HMMA whose C words are still
    that result reads the f32 value instead (the driver's fold of a pack and unpack pair, below)."""
    lane = np.arange(32)
    g, t = lane // 4, lane % 4
    A = np.zeros((16, 16), np.float32)
    B = np.zeros((16, 8), np.float32)
    C = np.zeros((16, 8), np.float32)
    for r in range(4):
        lo, hi = halves(regs[ra + r][w])
        row, k = g + 8 * (r & 1), 8 * (r >> 1) + 2 * t
        A[row, k], A[row, k + 1] = lo, hi
    for r in range(2):
        lo, hi = halves(regs[rb + r][w])
        k = 8 * r + 2 * t
        B[k, g], B[k + 1, g] = lo, hi
    if f32:
        for r in range(4):
            C[g + 8 * (r >> 1), 2 * t + (r & 1)] = regs[rc + r][w].view(np.float32)
    else:
        for r in range(2):
            lo, hi = halves(regs[rc + r][w])
            C[g + 8 * r, 2 * t], C[g + 8 * r, 2 * t + 1] = lo, hi
        kept = chain.get((rc, w.start)) if chain is not None else None
        if kept is not None and all(np.array_equal(regs[rc + r][w], kept[1][r]) for r in range(2)):
            C = kept[0].copy()
    D = C.copy()
    for k in range(16):
        D = D + A[:, k:k + 1] * B[k:k + 1, :]   # float32: the product is exact, the sum rounds once
    if f32:
        for r in range(4):
            regs[rd + r][w] = D[g + 8 * (r >> 1), 2 * t + (r & 1)].view(np.uint32)
    else:
        for r in range(2):
            regs[rd + r][w] = pack(D[g + 8 * r, 2 * t], D[g + 8 * r, 2 * t + 1])
        if chain is not None:
            chain[(rd, w.start)] = (D, [regs[rd + r][w].copy() for r in range(2)])


def matrix_from_rows(shared: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """An 8 x 8 matrix of halves whose row r starts at byte address rows[r] of shared memory."""
    h = shared.view(np.uint16)
    return np.stack([h[(int(a) // 2):(int(a) // 2) + 8] for a in rows])


def warp_ldsm4(regs: dict, w: slice, shared: np.ndarray, rd: int, ra: int, offset: int, trans: bool):
    lane = np.arange(32)
    g, t = lane // 4, lane % 4
    addr = regs[ra][w].astype(np.int64) + offset
    for m in range(4):
        mat = matrix_from_rows(shared, addr[8 * m:8 * m + 8]).astype(np.uint32)
        if trans:
            lo, hi = mat[2 * t, g], mat[2 * t + 1, g]
        else:
            lo, hi = mat[g, 2 * t], mat[g, 2 * t + 1]
        regs[rd + m][w] = lo | (hi << 16)


def warp_movm(regs: dict, w: slice, rd: int, ra: int):
    lane = np.arange(32)
    g, t = lane // 4, lane % 4
    src = regs[ra][w]
    mat = np.zeros((8, 8), np.uint32)
    mat[g, 2 * t], mat[g, 2 * t + 1] = src & 0xFFFF, src >> 16
    regs[rd][w] = mat[2 * t, g] | (mat[2 * t + 1, g] << 16)


def model(p: Program, inp: bytes, f32_chains: bool = False) -> np.ndarray:
    """The output words of workgroup 0 (LOCAL x out_words) for an input of LOCAL x IN_WORDS words.

    f32_chains: an f16 accumulator that goes from one HMMA to the next keeps its f32 value. The
    console rounds every HMMA's result to f16 in its registers, and so do the model's default and
    the translated SPIR-V (PackHalf2x16 then UnpackHalf2x16); driver 617.14 on the RTX 3060 Ti folds
    that pair away in the translated scratch form, and its results match this model word for word."""
    regs = {r: np.zeros(LOCAL, np.uint32) for r in range(256)}
    chain = {} if f32_chains else None
    in_words = np.frombuffer(inp, np.uint32)
    out = np.zeros(LOCAL * p.out_words, np.uint32)
    shared = np.zeros(max(p.shared, 4), np.uint8)
    tid = np.arange(LOCAL, dtype=np.uint64)
    c0 = {SB_IN: IN_ADDR, SB_IN + 4: 0, SB_OUT: OUT_ADDR, SB_OUT + 4: 0}

    def r64(r):
        return regs[r].astype(np.uint64) | (regs[r + 1].astype(np.uint64) << np.uint64(32))

    for op in p.ops:
        kind = op[0]
        if kind == "s2r":
            regs[op[1]][:] = {SR_TID_X: tid, SR_CTAID_X: 0 * tid, SR_LANEID: tid % 32}[op[2]]
        elif kind == "c0":
            regs[op[1]][:] = c0[op[2]]
        elif kind in ("imad", "imad_wide"):
            _, rd, ra, imm, rc = op
            v = regs[ra].astype(np.uint64) * np.uint64(imm)
            if kind == "imad":
                regs[rd][:] = (v + (0 if rc == RZ else regs[rc])) & M32
            else:
                v = (v + r64(rc)) & np.uint64((1 << 64) - 1)
                regs[rd][:], regs[rd + 1][:] = v & M32, v >> np.uint64(32)
        elif kind == "ldg128":
            _, rd, ra, off = op
            idx = ((r64(ra) + np.uint64(off) - np.uint64(IN_ADDR)) // np.uint64(4)).astype(np.int64)
            for q in range(4):
                regs[rd + q][:] = in_words[idx + q]
        elif kind == "stg128":
            _, ra, rb, off = op
            idx = ((r64(ra) + np.uint64(off) - np.uint64(OUT_ADDR)) // np.uint64(4)).astype(np.int64)
            for q in range(4):
                out[idx + q] = regs[rb + q]
        elif kind == "sts128":
            _, ra, rb, off = op
            words = shared.view(np.uint32)
            idx = (regs[ra].astype(np.int64) + off) // 4
            for q in range(4):
                words[idx + q] = regs[rb + q]
        elif kind == "bar":
            pass   # every store comes before every load in these programs
        elif kind == "xor_imm":
            regs[op[1]][:] = regs[op[2]] ^ np.uint32(op[3])
        elif kind == "xor_reg":
            regs[op[1]][:] = regs[op[2]] ^ regs[op[3]]
        else:
            for wi in range(WARPS):
                w = slice(32 * wi, 32 * wi + 32)
                if kind == "hmma":
                    warp_hmma(regs, w, *op[1:], chain=chain)
                elif kind == "ldsm4":
                    warp_ldsm4(regs, w, shared, *op[1:])
                elif kind == "movm":
                    warp_movm(regs, w, *op[1:])
                else:
                    raise ValueError(kind)
    return out


# --- inputs ------------------------------------------------------------------------------------------

def random_halves(rng: np.random.Generator, n: int, lo_exp: int, hi_exp: int) -> np.ndarray:
    """n normal f16 values of random sign with magnitude in [2^lo_exp, 2^hi_exp)."""
    mag = 2.0 ** rng.uniform(lo_exp, hi_exp, n)
    return (mag * rng.choice([-1.0, 1.0], n)).astype(np.float16)


def lane_input(seed: int) -> bytes:
    """LOCAL x IN_WORDS words: A halves in [1/4, 1), B in [1/64, 1/16), f16 accumulators in [1/16, 1)
    (words 24..31; 32..39 zero); signs random."""
    rng = np.random.default_rng(seed)
    a = random_halves(rng, LOCAL * 32, -2, 0).view(np.uint16).reshape(LOCAL, 16, 2)
    b = random_halves(rng, LOCAL * 16, -6, -4).view(np.uint16).reshape(LOCAL, 8, 2)
    c16 = random_halves(rng, LOCAL * 16, -4, 0).view(np.uint16).reshape(LOCAL, 8, 2)
    words = np.zeros((LOCAL, IN_WORDS), np.uint32)
    for k, arr in ((0, a), (16, b), (24, c16)):
        n = arr.shape[1]
        words[:, k:k + n] = arr[:, :, 0].astype(np.uint32) | (arr[:, :, 1].astype(np.uint32) << 16)
    return words.tobytes()


def lane_input_f32(seed: int) -> bytes:
    """As lane_input, with words 24..39 holding f32 accumulators (the shuffle variant's)."""
    rng = np.random.default_rng(seed + 1)
    words = np.frombuffer(lane_input(seed), np.uint32).reshape(LOCAL, IN_WORDS).copy()
    c32 = (2.0 ** rng.uniform(-4, 0, LOCAL * 16) * rng.choice([-1.0, 1.0], LOCAL * 16)).astype(np.float32)
    words[:, 24:40] = c32.view(np.uint32).reshape(LOCAL, 16)
    return words.tobytes()


def c0_bank(in_size: int, out_size: int) -> bytes:
    """The driver bank c[0] of the translated programs: the two storage buffers' address and size."""
    data = bytearray(65536)
    struct.pack_into("<QI", data, SB_IN, IN_ADDR, in_size)
    struct.pack_into("<QI", data, SB_OUT, OUT_ADDR, out_size)
    return bytes(data)


# --- the cooperative matrix model --------------------------------------------------------------------

def coop_direct_input(seed: int) -> tuple[bytes, bytes]:
    """Per subgroup: A 4 x 16 x 16, B 4 x 16 x 8, f16 accumulators 4 x 16 x 8 (halves), and f32
    accumulators 4 x 16 x 8. Every value is a small multiple of a power of two (A in {-1, 0, 1}, B in
    {-1, 0, 1} / 256, accumulators k / 256 with |k| <= 64), so every partial sum of these chains is a
    multiple of 1/256 below 8 in magnitude, which f16 holds exactly: the result is the exact sum
    whatever the hardware's rounding and internal precision. (Long chains of f16 accumulation on the
    tensor cores do not round like a round-to-nearest model, so values that round could not be
    checked word for word, and a tolerance loose enough for a thousand roundings would hide a
    layout error.)"""
    rng = np.random.default_rng(seed)
    h = np.zeros((WARPS, 2048), np.float16)
    h[:, :1024] = rng.integers(-1, 2, (WARPS, 1024))
    h[:, 1024:1536] = rng.integers(-1, 2, (WARPS, 512)) / 256.0
    h[:, 1536:] = rng.integers(-64, 65, (WARPS, 512)) / 256.0
    f = (rng.integers(-64, 65, WARPS * 512) / 256.0).astype(np.float32)
    return h.tobytes(), f.tobytes()


def coop_direct_model(count: int, f32: bool, inp_h: bytes, inp_f: bytes) -> tuple[np.ndarray, float]:
    """Per subgroup the four accumulators after count multiply-adds, as exact float64 sums in the
    translated order, and the largest magnitude any partial sum reaches (below 8 keeps f16 exact)."""
    h = np.frombuffer(inp_h, np.float16).reshape(WARPS, 2048).astype(np.float64)
    f = np.frombuffer(inp_f, np.float32).reshape(WARPS, 512).astype(np.float64)
    exact = np.zeros((WARPS, 4, 16, 8), np.float64)
    peak = 0.0
    for sg in range(WARPS):
        A = h[sg, :1024].reshape(4, 16, 16)
        B = h[sg, 1024:1536].reshape(4, 16, 8)
        C = (f[sg] if f32 else h[sg, 1536:]).reshape(4, 16, 8)
        for x in range(4):
            acc = C[x].copy()
            for i in range(x, count, 4):
                for k in range(16):
                    acc = acc + A[x][:, k:k + 1] * B[(i // 4) % 4][k:k + 1, :]
                    peak = max(peak, float(np.max(np.abs(acc))))
            exact[sg, x] = acc
    return exact, peak


# --- checks ------------------------------------------------------------------------------------------

def compare_exact(got: np.ndarray, want: np.ndarray) -> dict:
    bad = int(np.count_nonzero(got != want))
    return {"words": int(want.size), "mismatched_words": bad, "exact": bad == 0}


def compare_tolerant(got: np.ndarray, want: np.ndarray, exact: np.ndarray, rel: float = 2.0 ** -6) -> dict:
    """got and want as float arrays of the same shape; passes when every value is within rel of the
    largest magnitude of its output set (a wrong layout errs by about the magnitudes themselves)."""
    scale = float(np.max(np.abs(exact))) or 1.0
    err = np.abs(got.astype(np.float64) - want.astype(np.float64))
    err64 = np.abs(got.astype(np.float64) - exact)
    finite = bool(np.all(np.isfinite(got)))
    worst = float(np.max(err)) if finite else float("inf")
    return {"values": int(want.size), "max_abs_err_vs_emulation": worst, "max_abs_err_vs_float64": float(np.max(err64)),
            "emulation_max_abs_err_vs_float64": float(np.max(np.abs(want.astype(np.float64) - exact))),
            "scale": scale, "tolerance": rel * scale, "exact_words": int(np.count_nonzero(got == want)),
            "pass": finite and worst <= rel * scale}


def lane_halves(words: np.ndarray) -> np.ndarray:
    return words.view(np.float16).astype(np.float32)


# --- tools and machine state -------------------------------------------------------------------------

def find_xlate() -> Path | None:
    name = "oot2-xlate.exe" if os.name == "nt" else "oot2-xlate"
    hits = sorted((ROOT / "build" / "xlate" / "publish").rglob(name))
    return hits[0] if hits else None


def find_sdk_tool(name: str) -> str | None:
    sdk = os.environ.get("VULKAN_SDK")
    for base in ([Path(sdk) / "Bin", Path(sdk) / "bin"] if sdk else []):
        for suffix in (".exe", ""):
            if (base / (name + suffix)).exists():
                return str(base / (name + suffix))
    return shutil.which(name)


def harness_exe() -> Path:
    return ROOT / "build" / "vkhmmacost" / ("vkhmmacost.exe" if os.name == "nt" else "vkhmmacost")


def vk_env() -> dict:
    env = dict(os.environ)
    env["VK_LOADER_LAYERS_DISABLE"] = "~implicit~"   # third-party implicit layers flood the messenger
    return env


CLOCK_QUERY = "clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,pstate,utilization.gpu"


def gpu_clocks() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"error": "no nvidia-smi"}
    out = subprocess.run([exe, f"--query-gpu={CLOCK_QUERY}", "--format=csv,noheader"], capture_output=True, text=True)
    return {"query": CLOCK_QUERY, "line": out.stdout.strip(), "rc": out.returncode}


class ClockSampler:
    """nvidia-smi every 50 ms while a run lasts; the SM and memory clocks of the samples taken while
    the GPU was busy (utilization above 50 percent)."""

    def __init__(self, path: Path):
        self.path = path
        exe = shutil.which("nvidia-smi")
        self.proc = None
        if exe:
            self.file = path.open("w", encoding="utf-8")
            self.proc = subprocess.Popen([exe, f"--query-gpu={CLOCK_QUERY}", "--format=csv,noheader,nounits", "-lms", "50"],
                                         stdout=self.file, stderr=subprocess.DEVNULL)

    def stop(self) -> dict:
        if not self.proc:
            return {"error": "no nvidia-smi"}
        self.proc.terminate()
        self.proc.wait()
        self.file.close()
        sm, mem, busy_n, n = [], [], 0, 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) != 7 or not f[0].isdigit():
                continue
            n += 1
            if f[6].isdigit() and int(f[6]) > 50:
                busy_n += 1
                sm.append(int(f[0]))
                mem.append(int(f[1]))
        out = {"samples": n, "busy_samples": busy_n}
        if sm:
            out.update(sm_mhz_min=min(sm), sm_mhz_median=statistics.median(sm), sm_mhz_max=max(sm),
                       mem_mhz_median=statistics.median(mem))
        return out


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


HEAVY = ("clang", "ninja", "vkpipecost", "vkcomputecost", "oot2", "eden", "lichii", "yuzu", "dotnet", "msbuild", "cl.exe")


def machine_load() -> dict:
    """Free memory, the processor's busy share over one second, and heavy processes by name."""
    load: dict = {}
    if os.name == "nt":
        ms = MemoryStatus()
        ms.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            load["available_mib"] = ms.ullAvailPhys // 1048576
            load["total_mib"] = ms.ullTotalPhys // 1048576
        load["cpu_busy_percent"] = cpu_busy_percent()
        tl = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, env=dict(os.environ, MSYS_NO_PATHCONV="1"))
        names = sorted({line.split('","')[0].strip('"').lower() for line in tl.stdout.splitlines() if line})
        load["heavy_processes"] = [n for n in names if any(h in n for h in HEAVY)]
    return load


def cpu_busy_percent(seconds: float = 1.0) -> float | None:
    """The whole processor's busy share over `seconds`, from GetSystemTimes (idle against kernel + user)."""
    ft = ctypes.c_ulonglong
    k = ctypes.windll.kernel32

    def sample():
        idle, kern, user = ft(), ft(), ft()
        if not k.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
            return None
        return idle.value, kern.value + user.value

    a = sample()
    time.sleep(seconds)
    b = sample()
    if not a or not b or b[1] == a[1]:
        return None
    return round(100.0 * (1.0 - (b[0] - a[0]) / (b[1] - a[1])), 1)


# --- running -----------------------------------------------------------------------------------------

def translate(xlate: Path, p: Program, work: Path, hmma: str | None = None) -> tuple[Path, dict]:
    """hmma: the job option of that name ("coop", the translator's default, or "helper"); None leaves the default."""
    code = work / f"{p.name}-{p.count}.bin"
    code.write_bytes(p.binary())
    key = f"t50{p.name.replace('-', '')}{p.count}{hmma or ''}"
    job = {"key": key, "stage": "cs", "arch": "sm86", "code": str(code), "c1": "",
           "nvn2_cs": [0, 0, 0, 16, 0, 0, 0], "local": [LOCAL, 1, 1], "shared": p.shared}
    if hmma:
        job["options"] = {"hmma": hmma}
    results, _ = pool.run_pool([str(xlate)], [(job, work)], 1, work / "logs", timeout=600)
    r = results[0]
    if r.status != "ok":
        raise RuntimeError(f"translation of {p.name} failed: {r.message}")
    return work / f"{key}.spv", json.loads((work / f"{key}.res.json").read_text(encoding="utf-8"))


def compile_glsl(src: Path, out: Path, defines: dict) -> Path:
    glslang = find_sdk_tool("glslangValidator")
    if not glslang:
        raise FileNotFoundError("glslangValidator (Vulkan SDK) is missing")
    cmd = [glslang, "-V", "--target-env", "vulkan1.3", "-S", "comp", *[f"-D{k}={v}" for k, v in defines.items()],
           str(src), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"glslangValidator failed on {src.name}: {r.stdout}{r.stderr}")
    return out


def run_job(job_lines: list[str], path: Path) -> dict:
    path.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    r = subprocess.run([str(harness_exe()), "run", str(path)], capture_output=True, text=True, env=vk_env())
    if r.returncode not in (0, 1) or not r.stdout.strip():
        raise RuntimeError(f"vkhmmacost exit {r.returncode}: {r.stderr.strip()[-2000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    res["exit"] = r.returncode
    res["stderr_tail"] = r.stderr.strip()[-2000:]
    return res


# Instructions whose first two operands are a result type and a result id (enough to type a lane operand).
TYPED_OPS = {61, 81, 83, 124, 126, 128, 130, 132, 169, 194, 195, 196, 197, 198, 199, 200}


def swap_shuffle_lanes(spv: bytes) -> bytes:
    """Counter-run: the first OpGroupNonUniformShuffle reads one fixed lane for every lane: its lane
    operand becomes a constant from 1 to 31 of the lane's own type (a constant dominates every use, so
    the module stays valid). In the translator's unoptimised modules two different lane ids often hold the same lane,
    so swapping ids between shuffles can leave the result unchanged; a constant cannot."""
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    types, ones, shuffles, i = {}, {}, [], 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        if op in TYPED_OPS and wc >= 3:
            types[words[i + 2]] = words[i + 1]
        elif op == 43 and wc == 4 and 1 <= words[i + 3] <= 31:  # OpConstant type result value
            ones.setdefault(words[i + 1], words[i + 2])
        elif op == 345:  # OpGroupNonUniformShuffle: type, result, scope, value, lane
            shuffles.append(i + 5)
        i += wc
    for at_lane in shuffles:
        one = ones.get(types.get(words[at_lane]))
        if one is not None:
            words[at_lane] = one
            return struct.pack(f"<{len(words)}I", *words)
    raise RuntimeError("no shuffle whose lane type has a constant from 1 to 31")


def swap_helper_stores(spv: bytes) -> bytes:
    """Counter-run: swap the pointers of the first two stores to workgroup memory inside the helper
    dmsh_hmma16816_f16 (A's first two words land in each other's places)."""
    words = list(struct.unpack(f"<{len(spv) // 4}I", spv))
    names, i = {}, 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        if op == 5:  # OpName
            raw = struct.pack(f"<{wc - 2}I", *words[i + 2:i + wc])
            names[words[i + 1]] = raw.split(b"\0")[0].decode()
        i += wc
    helper = next((k for k, v in names.items() if v == "dmsh_hmma16816_f16"), None)
    if helper is None:
        raise RuntimeError("no helper named dmsh_hmma16816_f16")
    inside, stores, i = False, [], 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        if op == 54:  # OpFunction
            inside = words[i + 2] == helper
        elif op == 56:
            inside = False
        elif op == 62 and inside:  # OpStore pointer value
            stores.append(i + 1)
        i += wc
    if len(stores) < 2:
        raise RuntimeError("the helper has fewer than two stores")
    words[stores[0]], words[stores[1]] = words[stores[1]], words[stores[0]]
    return struct.pack(f"<{len(words)}I", *words)


def summarise(ms: list[float]) -> dict:
    s = sorted(ms)
    pick = lambda q: s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p10": pick(0.10), "p50": statistics.median(s), "p90": pick(0.90),
            "p99": pick(0.99), "max": s[-1], "mean": statistics.fmean(s)}


VARIANTS = ("scratch", "shuffle", "scratch-plain", "shuffle-plain", "coop-helper", "coop-f16", "coop-f32", "ldsm", "movm")


def build_variant(name: str, count: int, xlate: Path, work: Path, seed: int, groups: int, mutate: str | None):
    """The module, the job's buffer lines, and a check function of the read-back outputs."""
    work.mkdir(parents=True, exist_ok=True)
    info: dict = {"name": name, "count": count, "groups": groups}
    if name in ("scratch", "shuffle", "scratch-plain", "shuffle-plain", "ldsm", "movm"):
        f32 = name.startswith("shuffle")
        if name in ("ldsm", "movm"):
            p = ldsm_program(count) if name == "ldsm" else movm_program(count)
        else:
            p = hmma_program(count, f32, not name.endswith("-plain"))
        # the scratch variants measure the scratch-memory helper, which the translator now keeps for the runtime's fallback
        # and writes as the only form with the job option hmma = "helper" (its default runs HMMA on cooperative matrices)
        spv, res = translate(xlate, p, work, "helper" if name.startswith("scratch") else None)
        if mutate == "shuffle" and name == "shuffle":
            spv.write_bytes(swap_shuffle_lanes(spv.read_bytes()))
            info["mutated"] = "a shuffle reads one fixed lane"
        if mutate == "scratch" and name == "scratch":
            spv.write_bytes(swap_helper_stores(spv.read_bytes()))
            info["mutated"] = "helper stores swapped"
        inp = lane_input_f32(seed) if f32 else lane_input(seed)
        out_size = groups * LOCAL * 4 * p.out_words
        (work / "in.bin").write_bytes(inp)
        (work / "c0.bin").write_bytes(c0_bank(len(inp), out_size))
        sb = res["support_buffer"]
        sbufs = {b["cb_byte_offset"]: b for b in res["sbufs"]}
        if set(sbufs) != {SB_IN, SB_OUT} or res["raw_access"]["physical_pointers"]:
            raise RuntimeError(f"{name}: the translator did not bind the two storage buffers: {res['sbufs']}")
        (c0,) = [c for c in res["cbufs"] if c["slot"] == 0]
        lines = [f"buffer {sb['set']} {sb['binding']} ubo {sb['size']}",
                 f"buffer {c0['set']} {c0['binding']} ubo 65536 in={work / 'c0.bin'}",
                 f"buffer {sbufs[SB_IN]['set']} {sbufs[SB_IN]['binding']} ssbo {len(inp)} in={work / 'in.bin'}",
                 f"buffer {sbufs[SB_OUT]['set']} {sbufs[SB_OUT]['binding']} ssbo {out_size} out={work / 'out.bin'}"]
        f16_hmma = p.kind == "hmma" and not f32
        want = model(p, inp, f32_chains=f16_hmma)
        console = model(p, inp) if f16_hmma else want
        info.update(instructions=len(p.ops), spv_bytes=spv.stat().st_size, translator_id=res["translator_id"],
                    capabilities=res["capabilities"])

        def check():
            got = np.frombuffer((work / "out.bin").read_bytes(), np.uint32).reshape(groups, LOCAL * p.out_words)
            per = [compare_exact(got[g], want) for g in range(groups)]
            bad = sum(x["mismatched_words"] for x in per)
            out = {"kind": "exact", "model": "f32 chains" if f16_hmma else "console", "words": int(got.size),
                   "mismatched_words": bad, "pass": bad == 0}
            if f16_hmma:
                g0, cons = lane_halves(got[0]), lane_halves(console)
                diff = np.abs(g0.astype(np.float64) - cons)
                scale = float(np.max(np.abs(cons))) or 1.0
                out["against_console_rounding"] = {"mismatched_halves": int(np.count_nonzero(diff)), "halves": int(diff.size),
                                                   "max_abs_diff": float(diff.max()), "max_abs": scale,
                                                   "max_abs_diff_over_max_abs": float(diff.max()) / scale}
            if bad and p.kind == "hmma":
                g0 = got[0]
                if f32:
                    err = np.abs(g0.view(np.float32) - want.view(np.float32))
                else:
                    err = np.abs(lane_halves(g0) - lane_halves(want))
                out["max_abs_err_group0"] = float(np.nanmax(err))
            return out
        return spv, lines, check, info

    if name == "coop-helper":
        spv = compile_glsl(HERE / "coop_helper.comp", work / f"coop-helper-{count}.spv",
                           {"N": count, "PERTURB": 1, "BAD_B_LAYOUT": int(mutate == "coop-layout")})
        if mutate == "coop-layout":
            info["mutated"] = "B loaded row-major"
        inp = lane_input(seed)
        out_size = groups * LOCAL * 4 * 8
        (work / "in.bin").write_bytes(inp)
        lines = [f"buffer 0 0 ssbo {len(inp)} in={work / 'in.bin'}", f"buffer 0 1 ssbo {out_size} out={work / 'out.bin'}"]
        p = hmma_program(count, False, True)
        want = lane_halves(model(p, inp))
        exact = want.astype(np.float64)
        info.update(spv_bytes=spv.stat().st_size)

        def check():
            got = np.frombuffer((work / "out.bin").read_bytes(), np.uint32).reshape(groups, LOCAL * 8)
            per = [compare_tolerant(lane_halves(got[g]), want, exact) for g in range(groups)]
            worst = max(per, key=lambda x: x["max_abs_err_vs_emulation"])
            return {"kind": "tolerant", **worst, "pass": all(x["pass"] for x in per)}
        return spv, lines, check, info

    f32 = name == "coop-f32"
    spv = compile_glsl(HERE / "coop_direct.comp", work / f"{name}-{count}.spv",
                       {"N": count, "BAD_B_LAYOUT": int(mutate == "coop-layout"), **({"ACC_F32": 1} if f32 else {})})
    if mutate == "coop-layout":
        info["mutated"] = "B loaded column-major"
    inp_h, inp_f = coop_direct_input(seed)
    (work / "in_h.bin").write_bytes(inp_h)
    (work / "in_f.bin").write_bytes(inp_f)
    elem = 4 if f32 else 2
    out_size = groups * WARPS * 512 * elem
    lines = [f"buffer 0 0 ssbo {len(inp_h)} in={work / 'in_h.bin'}", f"buffer 0 1 ssbo {len(inp_f)} in={work / 'in_f.bin'}",
             f"buffer 0 2 ssbo {out_size} out={work / 'out.bin'}"]
    exact, peak = coop_direct_model(count, f32, inp_h, inp_f)
    if peak >= 8.0:
        raise RuntimeError(f"{name}: a partial sum reaches {peak}, past what f16 holds exactly; change the seed")
    want = exact.astype(np.float32 if f32 else np.float16)
    info.update(spv_bytes=spv.stat().st_size, largest_partial_sum=peak)

    def check():
        raw = (work / "out.bin").read_bytes()
        got = np.frombuffer(raw, np.float32 if f32 else np.float16).reshape(groups, WARPS * 512)
        bad = int(np.count_nonzero(got != want.reshape(1, -1)))
        return {"kind": "exact", "model": "exact sums", "values": int(got.size), "mismatched_values": bad, "pass": bad == 0}
    return spv, lines, check, info


def instructions_per_dispatch(groups: int, count: int) -> int:
    """Warp-wide instructions of the measured kind in one dispatch: one per subgroup per count."""
    return groups * WARPS * count


def cmd_run(args) -> int:
    xlate = find_xlate()
    if not xlate:
        print("hmma_cost: the translator is not built (py -3 tools/build_xlate.py)", file=sys.stderr)
        return 2
    if not harness_exe().exists():
        print("hmma_cost: vkhmmacost is not built (py -3 tools/build_vkdraw.py --harness vkhmmacost)", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = args.only.split(",") if args.only else list(VARIANTS)
    for n in names:
        if n not in VARIANTS:
            print(f"hmma_cost: unknown variant {n}", file=sys.stderr)
            return 1
    report = {"tool": "hmma_cost", "count": args.count, "reps": args.reps, "warmup": args.warmup, "seed": args.seed,
              "mutate": args.mutate, "load_before": machine_load(), "clocks_before": gpu_clocks(), "variants": []}
    ok = True
    for name in names:
        direct = name in ("coop-f16", "coop-f32")
        groups = args.coop_groups if direct else args.groups
        full_count = args.coop_count if direct else args.count
        entry: dict = {"name": name, "runs": []}
        for count in (full_count, full_count // 2):
            work = out / name / str(count)
            spv, lines, check, info = build_variant(name, count, xlate, work, args.seed, groups, args.mutate)
            head = [f"module {spv}", f"groups {groups} 1 1", f"chunk {args.chunk}"]
            if args.validate:
                v = run_job(head + ["validation on", "warmup 0", "reps 2"] + lines, work / "job-validate.txt")
                info["validation"] = {"errors": v["validation_errors"], "warnings": v["validation_warnings"],
                                      "stderr_tail": v["stderr_tail"]}
                info["validation_check"] = check()
                if v["validation_errors"]:
                    ok = False
            clocks0 = gpu_clocks()
            sampler = ClockSampler(work / "clocks.csv")
            try:
                r = run_job(head + ["validation off", f"warmup {args.warmup}", f"reps {args.reps}"] + lines, work / "job.txt")
            finally:
                during = sampler.stop()
            info.update(clocks_before=clocks0, clocks_during=during, clocks_after=gpu_clocks(), build_ms=r["build_ms"], wall_ms=r["wall_ms"],
                        device=r["device"], driver=r["driver"], ms=r["ms"], stats=summarise(r["ms"]), check=check(),
                        instructions_per_dispatch=instructions_per_dispatch(groups, count))
            ok &= info["check"]["pass"]
            entry["runs"].append(info)
            print(f"{name:14s} count {count:4d} groups {groups:6d}: p50 {info['stats']['p50']:.4f} ms, "
                  f"p90 {info['stats']['p90']:.4f}, build {r['build_ms']:.0f} ms, check {'pass' if info['check']['pass'] else 'FAIL'}",
                  flush=True)
        full, half = entry["runs"]
        dn = full["instructions_per_dispatch"] - half["instructions_per_dispatch"]
        entry["ns_per_instruction"] = 1e6 * (full["stats"]["p50"] - half["stats"]["p50"]) / dn
        entry["ns_per_instruction_whole"] = 1e6 * full["stats"]["p50"] / full["instructions_per_dispatch"]
        entry["half_over_full"] = half["stats"]["p50"] / full["stats"]["p50"]
        report["variants"].append(entry)
    report["clocks_after"] = gpu_clocks()
    report["load_after"] = machine_load()
    by = {v["name"]: v["ns_per_instruction"] for v in report["variants"]}
    report["ratios"] = {f"{a}/{b}": by[a] / by[b] for a in by for b in by
                        if a != b and b.startswith("coop") and not a.startswith("coop") and by[b] > 0}
    report["pass"] = ok
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    rows = ["variant\tns_per_instruction\tns_whole\thalf_over_full\tp50_full_ms\tp50_half_ms\tcheck"]
    for v in report["variants"]:
        rows.append(f"{v['name']}\t{v['ns_per_instruction']:.5f}\t{v['ns_per_instruction_whole']:.5f}\t{v['half_over_full']:.3f}\t"
                    f"{v['runs'][0]['stats']['p50']:.4f}\t{v['runs'][1]['stats']['p50']:.4f}\t"
                    f"{'pass' if all(r['check']['pass'] for r in v['runs']) else 'FAIL'}")
    (out / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))
    print(f"hmma_cost: {'every check passed' if ok else 'a check FAILED'}; report {out / 'report.json'}")
    return 0 if ok else 1


def cmd_info(_args) -> int:
    if not harness_exe().exists():
        print("hmma_cost: vkhmmacost is not built (py -3 tools/build_vkdraw.py --harness vkhmmacost)", file=sys.stderr)
        return 2
    r = subprocess.run([str(harness_exe()), "info"], capture_output=True, text=True, env=vk_env())
    print(r.stdout.strip())
    return r.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--count", type=int, default=128, help="instructions per lane (a multiple of 32)")
    r.add_argument("--reps", type=int, default=300)
    r.add_argument("--warmup", type=int, default=30)
    r.add_argument("--chunk", type=int, default=16, help="dispatches per submission")
    r.add_argument("--groups", type=int, default=2048)
    r.add_argument("--coop-groups", type=int, default=16384)
    r.add_argument("--coop-count", type=int, default=1024, help="multiply-adds per subgroup of coop-f16 and coop-f32")
    r.add_argument("--seed", type=int, default=50)
    r.add_argument("--only", default="")
    r.add_argument("--validate", action="store_true")
    r.add_argument("--mutate", choices=("shuffle", "scratch", "coop-layout"))
    args = ap.parse_args(argv)
    if args.cmd == "info":
        return cmd_info(args)
    if args.count % 32 or args.coop_count % 32:
        ap.error("--count and --coop-count must be multiples of 32")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
