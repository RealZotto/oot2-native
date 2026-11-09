# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the translator's AST optimizer with its debug mode off.

With debug mode on, nothing is propagated and nothing is dropped. With it off, Ryujinx's optimizer
moves an expression with a single use to that use and removes an assignment nobody reads. That
loses an atomic whose result is discarded, moves a read of shared or storage memory past a later
store, moves a shuffle into a branch, and folds an expression into a call's argument, which the
SPIR-V generator cannot take (the translation fails). The optimizer now pins what writes, carries a
read or a convergent expression only to the next statement that uses it with nothing written in
between, and never folds into a call.

Each program is assembled here from instruction words and translated both ways; tools/shaders/spvstats.py
says what was lost and what moved. A check passes only if the shape is really there: the control
program of each pair has no store (or no branch) and must translate smaller with debug off, which
shows that propagation still happens. The tests skip when the translator has not been built
(py -3 tools/build_xlate.py), unless OOT2_REQUIRE_XLATE is set.

Run: python -m unittest discover -s tests -p test_xlate_optimizer.py -v
"""

from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "shaders"))

import spvstats  # noqa: E402
import validate  # noqa: E402
from test_xlate_host import Harness, SPIRV_VAL, ampere, bit, compute_job, EXIT  # noqa: E402
from test_xlate_phi import guard, isetp_ne_zero, s2r  # noqa: E402
from test_xlate_raw import LDG, TID, TID_X, TID_Y, scheduled  # noqa: E402

B32 = 4


def sts(ra: int, rb: int, offset: int = 0) -> bytes:
    """STS [R<ra> + offset], R<rb>, 32 bits."""
    return scheduled(ampere(0x388, ra=ra, rb=rb, low_bits=(offset & 0xFFFFFF) << 40, high_bits=bit(73, B32)))


def lds(rd: int, ra: int, offset: int = 0) -> bytes:
    """LDS Rd, [R<ra> + offset], 32 bits."""
    return scheduled(ampere(0x984, rd=rd, ra=ra, low_bits=(offset & 0xFFFFFF) << 40, high_bits=bit(73, B32)))


def atoms_add(rd: int, ra: int, rb: int) -> bytes:
    """ATOMS.ADD.U32 Rd, [R<ra>], R<rb>; Rd = RZ discards the old value."""
    return scheduled(ampere(0x38C, rd=rd, ra=ra, rb=rb, high_bits=bit(73, 0) | bit(87, 0)))


def shfl_idx(rd: int, ra: int, lane: int) -> bytes:
    """SHFL.IDX Rd, PT, R<ra>, <lane>, 0x1f: the lane and the clamp are immediates (opcode 0xf89), the predicate result PT."""
    return scheduled(ampere(0xF89, rd=rd, ra=ra, rb=255, low_bits=(0x1F << 40) | (lane << 53), high_bits=bit(81, 7)))


X, Y = 10, 11      # registers that hold the thread's x and y
V = 7              # the register a program reads back


class OptimizerTest(Harness):
    def both(self, key: str, instructions: list[bytes]) -> tuple[dict, dict]:
        """Translate the program with debug on and off and return their stats; both must translate and validate."""
        code = self.dir / f"{key}.bin"
        code.write_bytes(b"".join(instructions) + EXIT)
        jobs = [compute_job(f"{key}.{tag}", code, shared=64, options={"debug": debug}) for tag, debug in (("on", True), ("off", False))]
        got = self.run_jobs(jobs)
        out = []
        for tag in ("on", "off"):
            r = got[f"{key}.{tag}"]
            self.assertEqual((r.status, r.message), ("ok", ""), f"debug {tag}")
            path = self.out / f"{key}.{tag}.spv"
            if SPIRV_VAL:
                v = validate.validate_one(SPIRV_VAL, f"{key}.{tag}", path)
                self.assertTrue(v.ok, v.message)
            out.append(spvstats.analyze(path.read_bytes()))
        return out[0], out[1]

    def assert_sound(self, on: dict, off: dict):
        verdict = spvstats.compare(on, off)
        self.assertEqual(verdict["lost"], {}, "an operation was dropped")
        self.assertFalse(verdict["restructured"], "the structure of the program changed")
        self.assertEqual(verdict["moved"], 0, "a read moved")
        self.assertEqual(verdict["moved_convergent"], 0, "a convergent operation moved")

    def assert_propagates(self, on: dict, off: dict):
        """The optimizer still shortens the module: a rule that kept everything in place would pass the checks above for nothing."""
        self.assertLess(off["instructions"], on["instructions"])

    # -- a read of memory against a later store of the same memory --------------------------------

    def test_a_read_is_not_carried_past_a_store(self):
        on, off = self.both("past", [s2r(X, TID_X), sts(255, X, 0), lds(V, 255, 0), s2r(Y, TID_Y), sts(255, Y, 0), sts(255, V, 4)])
        self.assertEqual(on["loads"], {"Workgroup": 1})
        self.assert_sound(on, off)

    def test_the_same_read_with_no_store_between_is_carried(self):
        on, off = self.both("clear", [s2r(X, TID_X), sts(255, X, 0), lds(V, 255, 0), s2r(Y, TID_Y), sts(255, V, 4)])
        self.assert_sound(on, off)
        self.assert_propagates(on, off)

    # -- an operation nobody reads ------------------------------------------------------------------

    def test_an_atomic_whose_result_nobody_reads_stays(self):
        on, off = self.both("atoms", [s2r(X, TID_X), atoms_add(255, 255, X), sts(255, X, 8)])
        self.assertEqual(on["atomics"], 1)
        self.assertEqual(off["atomics"], 1)
        self.assert_sound(on, off)

    def test_an_atomic_whose_result_is_read_is_still_carried(self):
        on, off = self.both("atoms_used", [s2r(X, TID_X), atoms_add(V, 255, X), sts(255, V, 8)])
        self.assertEqual(off["atomics"], 1)
        self.assert_sound(on, off)
        self.assert_propagates(on, off)

    def test_a_raw_atomic_whose_result_nobody_reads_stays(self):
        atomg = ampere(0x3A8, rd=255, ra=2, rb=3, rc=0, high_bits=bit(72))     # ATOMG.E.ADD.U32 RZ, [R2.64], R3
        on, off = self.both("raw_atomic", TID + [atomg, ampere(0x388, ra=255, rb=3)])
        self.assertGreaterEqual(on["calls_write"], 1)
        self.assertEqual(off["calls_write"], on["calls_write"])
        self.assert_sound(on, off)

    # -- a call takes its arguments in variables ----------------------------------------------------

    def test_a_raw_load_whose_result_is_read_translates_with_debug_off(self):
        # The helper logs a miss, so it counts as a writer; its arguments are variables the optimizer used to fold away.
        on, off = self.both("raw_load", TID + [LDG, ampere(0x388, ra=255, rb=0)])
        self.assertGreaterEqual(on["calls_write"], 1)
        self.assertEqual(off["calls_write"], on["calls_write"])
        self.assert_sound(on, off)

    # -- a shuffle needs the lanes it was written for -----------------------------------------------

    def test_a_shuffle_is_not_carried_into_a_branch(self):
        branch = [s2r(X, TID_X), isetp_ne_zero(X), shfl_idx(V, X, 3), guard(sts(255, V, 0))]
        on, off = self.both("shuffle", branch)
        self.assertEqual(on["subgroup"], 1)
        self.assert_sound(on, off)

    def test_a_shuffle_used_at_once_is_carried(self):
        on, off = self.both("shuffle_now", [s2r(X, TID_X), shfl_idx(V, X, 3), sts(255, V, 0)])
        self.assert_sound(on, off)
        self.assert_propagates(on, off)

    # -- the default ----------------------------------------------------------------------------------

    def test_a_job_that_names_no_debug_option_gets_the_sound_form(self):
        """The sound form is the default: the same program with no option is byte for byte the module of `debug: false`,
        and not the module of `debug: true`, which is longer because nothing is propagated."""
        code = self.dir / "default.bin"
        code.write_bytes(b"".join([s2r(X, TID_X), sts(255, X, 0), lds(V, 255, 0), s2r(Y, TID_Y), sts(255, V, 4)]) + EXIT)
        got = self.run_jobs([compute_job("def", code, shared=64),
                             compute_job("sound", code, shared=64, options={"debug": False}),
                             compute_job("old", code, shared=64, options={"debug": True})])
        for key in ("def", "sound", "old"):
            self.assertEqual((got[key].status, got[key].message), ("ok", ""), key)
        modules = {key: (self.out / f"{key}.spv").read_bytes() for key in got}
        descriptions = {key: json.loads((self.out / f"{key}.res.json").read_text(encoding="utf-8")) for key in got}
        self.assertEqual(modules["def"], modules["sound"])
        self.assertNotEqual(modules["def"], modules["old"])
        self.assertLess(len(modules["def"]), len(modules["old"]))
        self.assertEqual([descriptions[k]["options"]["debug"] for k in ("def", "sound", "old")], [False, False, True])


if __name__ == "__main__":
    unittest.main()
