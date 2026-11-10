# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the places of compute, task and mesh modules in the layout `uni` (the compute, task and mesh layout).

Compute has a layout of its own (uniform 1..8, storage 9..24, storage images 25..31, texture
slots 0..27); a mesh module takes the vertex stage's buffer slots (uniform 1..8, storage 9..13)
and two texture slots; a task module's buffers are in set 2 (uniform 0..7, storage 8..15); and
push constant bytes 24..35, the vertex stage's slots 2..4, are kept for a mesh draw's first task.
The programs are assembled here from instruction words: a storage buffer is a global store whose
address the translator traces to a word pair of c[0], a constant bank a MOV from it.

The tests skip when the translator has not been built (py -3 tools/build_xlate.py),
unless OOT2_REQUIRE_XLATE is set.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import unittest

from test_xlate_host import Harness, ampere, bit, compute_job, mesh_fields, EXIT, SPIRV_VAL, validate

# The scheduling fields of bits 104..115: a stall of one cycle and no write or read barrier (7). With
# these bits clear an instruction names barrier 0, and the translator then gives its result to no later
# reader that does not wait on that barrier (Translator.EmitOpsAmpere): an address loaded by a MOV would
# not reach the store, which would then go through the raw helper instead of a storage buffer.
NO_BARRIERS = 0xFE2 << 40

B32 = 4             # LsSize of a 32-bit access (bits 73..76 of LDG and STG)
SB_BASE = 0x100     # c[0] byte offset of the first storage buffer's address pair; one every 0x10 bytes
VALUE_REG = 100     # registers that hold the constant bank words


def scheduled(ins: bytes) -> bytes:
    lo, hi = struct.unpack("<QQ", ins)
    return struct.pack("<QQ", lo, hi | NO_BARRIERS)


def mov_cbuf(rd: int, bank: int, offset: int) -> bytes:
    """MOV Rd, c[bank][offset]: MOV (0x002) in ALU operand form 5 (bits 9..11), its source the constant
    buffer word with the bank in bits 54..59 and the byte offset in 38..53 (InstEmitAmpere.ACbufOperand)."""
    return scheduled(ampere(0xA02, rd=rd, ra=0, rb=0, low_bits=(offset << 38) | (bank << 54)))


def stg(address: int, value: int, offset: int = 0) -> bytes:
    """STG.E [R<address>.64 + offset], R<value>: a 32-bit global store (0x386), .E (bit 72) the register pair."""
    return scheduled(ampere(0x386, ra=address, rb=value, rc=0, low_bits=offset << 40, high_bits=bit(72) | bit(73, B32)))


def ldg(rd: int, ra: int) -> bytes:
    """LDG.E Rd, [R<ra>.64], 32 bits (0x381)."""
    return scheduled(ampere(0x381, rd=rd, ra=ra, rb=255, rc=0, high_bits=bit(72) | bit(73, B32)))


def tex_register(rd: int, handle: int) -> bytes:
    """TEX Rd with its handle in register `handle` (0x361: the first register of the second source block), 2D at
    coordinates RZ, level zero, four components, no residency predicate."""
    return scheduled(ampere(0x361, rd=rd, ra=255, rb=handle, low_bits=(1 << 59) | (1 << 61),
                            high_bits=bit(72, 0xF) | bit(81, 7) | bit(87, 1)))


def declared_capabilities(spirv: bytes) -> set[int]:
    """The numbers of the OpCapability instructions at the head of a module."""
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    found, i = set(), 5
    while i < len(words) and words[i] & 0xFFFF == 17:
        found.add(words[i + 1])
        i += words[i] >> 16
    return found


CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM = 5307
CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM = 5309


def buffers_program(banks: list[int], storage: int) -> bytes:
    """Stores to `storage` storage buffers (a zero to each), then one word of each constant bank in `banks`
    into the first of them; a module of it reads c[0] for the buffer addresses too."""
    assert storage >= 1
    code = []
    for k in range(storage):
        pair = SB_BASE + 0x10 * k
        code += [mov_cbuf(2 + 2 * k, 0, pair), mov_cbuf(3 + 2 * k, 0, pair + 4)]
    code += [stg(2 + 2 * k, 255) for k in range(storage)]
    for j, bank in enumerate(banks):
        code += [mov_cbuf(VALUE_REG + j, bank, 0x10), stg(2, VALUE_REG + j, 4 * (j + 1))]
    return b"".join(code) + EXIT


class UniPlacesTest(Harness):
    def job(self, key: str, stage: str, banks: list[int], storage: int) -> dict:
        code = self.dir / f"{key}.bin"
        code.write_bytes(buffers_program(banks, storage))
        if stage == "cs":
            return compute_job(key, code)
        return {"key": key, "stage": stage, "arch": "sm86", "code": str(code), "c1": "", **mesh_fields(stage)}

    def graphics_job(self, key: str, stage: str, header_type: int) -> dict:
        header = bytearray(0x80)
        struct.pack_into("<I", header, 0, (header_type << 10) | 3)
        code = self.dir / f"{key}.bin"
        code.write_bytes(bytes(header) + EXIT)
        return {"key": key, "stage": stage, "arch": "sm86", "code": str(code), "c1": ""}

    def translate(self, job: dict) -> dict:
        got = self.run_jobs([job])[job["key"]]
        self.assertEqual((got.status, got.message), ("ok", ""))
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, job["key"], self.out / f"{job['key']}.spv")
            self.assertTrue(v.ok, v.message)
        return json.loads((self.out / f"{job['key']}.res.json").read_text(encoding="utf-8"))

    def bindings(self, items: list[dict], buffer_set: int) -> list[int]:
        """The bindings of the module's buffers of one kind, sorted, each checked to be in the stage's set."""
        for b in items:
            self.assertEqual(b["set"], buffer_set, b)
        return sorted(b["binding"] for b in items)

    def assert_contiguous_banks(self, res: dict, banks: int):
        """The module's constant banks fill its stage's uniform slots from the first, each at its ordinal."""
        lay = res["layout"]
        first, count = lay["uniform_slots"]
        got = self.bindings(res["cbufs"], lay["buffer_set"])
        self.assertGreaterEqual(len(got), banks)   # the user banks, and c[0] when the module reads it
        self.assertLessEqual(len(got), count)
        self.assertEqual(got, list(range(first, first + len(got))))
        for b in res["cbufs"]:
            self.assertEqual(b["ordinal"], b["binding"] - first, b)

    def assert_overflow(self, res: dict, kind: str, count: int, n: int):
        """n resources of one kind past the stage's `count` slots: bindings from 32, named in the description."""
        lay = res["layout"]
        self.assertFalse(lay["fits"])
        lines = [o for o in lay["overflow"] if f"no {kind} slot left for the {lay['role']} stage ({count})" in o]
        self.assertEqual(len(lines), n, lay["overflow"])
        self.assertTrue(any(note.startswith("layout uni:") for note in res["notes"]), res["notes"])

    # ---- the places ----

    def test_compute_has_a_layout_of_its_own(self):
        res = self.translate(self.job("cs3", "cs", [3, 4, 5], 3))
        lay = res["layout"]
        self.assertEqual((lay["role"], lay["buffer_set"]), ("compute", 1))
        self.assertEqual((lay["uniform_slots"], lay["storage_slots"], lay["storage_image_slots"], lay["texture_slots"]),
                         ([1, 8], [9, 16], [25, 7], [0, 28]))
        self.assert_contiguous_banks(res, 3)
        self.assertEqual(self.bindings(res["sbufs"], 1), [9, 10, 11])
        self.assertTrue(lay["fits"])
        self.assertEqual(res["push_descriptors"], 1 + len(res["cbufs"]) + 3)

    def test_mesh_takes_the_vertex_buffer_slots_and_two_texture_slots(self):
        res = self.translate(self.job("ms5", "ms", [3, 4], 5))
        lay = res["layout"]
        self.assertEqual((lay["role"], lay["buffer_set"]), ("mesh", 1))
        self.assertEqual((lay["uniform_slots"], lay["storage_slots"], lay["storage_image_slots"], lay["texture_slots"]),
                         ([1, 8], [9, 5], [0, 0], [0, 2]))
        self.assert_contiguous_banks(res, 2)
        self.assertEqual(self.bindings(res["sbufs"], 1), [9, 10, 11, 12, 13])
        self.assertTrue(lay["fits"])

    def test_task_buffers_are_in_set_2(self):
        res = self.translate(self.job("ts4", "ts", [3, 4, 5], 4))
        lay = res["layout"]
        self.assertEqual((lay["role"], lay["buffer_set"], lay["push_set"]), ("task", 2, 1))
        self.assertEqual((lay["uniform_slots"], lay["storage_slots"], lay["storage_image_slots"], lay["texture_slots"]),
                         ([0, 8], [8, 8], [0, 0], [0, 0]))
        self.assert_contiguous_banks(res, 3)
        self.assertEqual(self.bindings(res["sbufs"], 2), [8, 9, 10, 11])
        self.assertTrue(lay["fits"])

    def test_a_task_draw_pushes_only_the_support_buffer(self):
        res = self.translate(self.job("tsn", "ts", [3, 4], 3))
        self.assertEqual(res["push_descriptors"], 1)
        self.assertEqual(res["task_set_descriptors"], len(res["cbufs"]) + 3)

    def test_the_first_task_bytes_are_free_in_a_mesh_pipeline(self):
        # Kept by the task and mesh stages; inside the vertex stage's slots, which a mesh pipeline lacks; clear of
        # D15's 16 bytes, of the mesh stage's slots and of the pixel stage's.
        jobs = [self.job("ftm", "ms", [3], 1), self.job("ftt", "ts", [3], 1), self.job("ftc", "cs", [3], 1),
                self.graphics_job("ftv", "vs", 1), self.graphics_job("ftp", "ps", 5)]
        lay = {j["stage"]: self.translate(j)["layout"] for j in jobs}

        def slot_bytes(stage):
            first, count = lay[stage]["texture_slots"]
            return set(range(lay[stage]["texture_slots_offset"] + 4 * first, lay[stage]["texture_slots_offset"] + 4 * (first + count)))

        self.assertEqual(lay["ms"]["first_task_bytes"], [24, 12])
        self.assertEqual(lay["ts"]["first_task_bytes"], [24, 12])
        for stage in ("cs", "vs", "ps"):
            self.assertEqual(lay[stage]["first_task_bytes"], [0, 0], stage)
        start, length = lay["ms"]["first_task_bytes"]
        first_task = set(range(start, start + length))
        self.assertGreaterEqual(start, 16)
        self.assertLessEqual(start + length, lay["ms"]["push_constant_bytes"])
        self.assertLessEqual(first_task, slot_bytes("vs"))
        for stage in ("ms", "ts", "ps"):
            self.assertFalse(first_task & slot_bytes(stage), stage)

    # ---- modules whose buffers overflow their stage's range ----

    def test_compute_storage_past_its_range(self):
        res = self.translate(self.job("cs17", "cs", [3], 17))
        self.assertEqual(self.bindings(res["sbufs"], 1), list(range(9, 25)) + [32])
        self.assert_overflow(res, "storage", 16, 1)

    def test_mesh_storage_past_its_range(self):
        res = self.translate(self.job("ms6", "ms", [3], 6))
        self.assertEqual(self.bindings(res["sbufs"], 1), [9, 10, 11, 12, 13, 32])
        self.assert_overflow(res, "storage", 5, 1)

    def test_mesh_banks_past_the_uniform_range(self):
        res = self.translate(self.job("ms9", "ms", list(range(3, 12)), 1))
        n = len(res["cbufs"]) - 8
        self.assertGreaterEqual(n, 1)
        self.assertEqual(self.bindings(res["cbufs"], 1), list(range(1, 9)) + list(range(32, 32 + n)))
        self.assert_overflow(res, "uniform", 8, n)

    def test_the_texture_pool_arrays_are_not_storage_image_arrays(self):
        # A handle loaded from a storage buffer is a texture of the pool: the translator declares the arrays of sampled
        # images non-uniformly indexed (the generator's declaration, in 9,533 modules of the corpus) and does not
        # declare StorageImageArrayNonUniformIndexing, which no module needs.
        code = [mov_cbuf(2, 0, SB_BASE), mov_cbuf(3, 0, SB_BASE + 4), ldg(4, 2), tex_register(0, 4),
                stg(2, 0, 8), stg(2, 1, 12)]
        code_file = self.dir / "pool.bin"
        header = bytearray(0x80)
        struct.pack_into("<I", header, 0, (1 << 10) | 3)
        code_file.write_bytes(bytes(header) + b"".join(code) + EXIT)
        res = self.translate({"key": "pool", "stage": "vs", "arch": "sm86", "code": str(code_file), "c1": ""})
        self.assertEqual([(t["via"], t["array_length"]) for t in res["textures"]], [("pool", 33537), ("pool", 4096)])
        declared = declared_capabilities((self.out / "pool.spv").read_bytes())
        self.assertIn(CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM, declared)
        self.assertNotIn(CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM, declared)
        for field in ("capabilities", "capabilities_added"):   # the description says the same
            self.assertNotIn("StorageImageArrayNonUniformIndexing", [n.removesuffix("EXT") for n in res[field]])
        self.assertIn("SampledImageArrayNonUniformIndexing", [n.removesuffix("EXT") for n in res["capabilities"]])

    def test_task_storage_past_set_2(self):
        res = self.translate(self.job("ts9", "ts", [3], 9))
        self.assertEqual(self.bindings(res["sbufs"], 2), list(range(8, 16)) + [32])
        self.assert_overflow(res, "storage", 8, 1)


if __name__ == "__main__":
    unittest.main()
