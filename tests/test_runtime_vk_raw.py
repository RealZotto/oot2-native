# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The raw path on a real device (runtime/gpu/vk_raw.c, architecture D15): a compute program the translator made from a
hand-assembled sm_86 program reads and writes through raw pointers, walking the page table the runtime builds.

The program (raw_program below), for each of its 64 threads: four LDG from the thread's 32-byte entry of a pointer table
at a fixed address in the window for pools the CPU cannot see (a probe address and an atomic's address), a raw load at
the probe address, a raw store of (thread | 0xA0000000) at the probe address plus 4, an ATOMG add of 1 at the atomic's
address, and two STG of the loaded value and the atomic's old value back into the entry. Every one of those accesses
goes through the translator's raw-access helper, since its address comes from the thread's index.

tests/runtime/vk_raw_test.c says what each case checks: every backing kind (import, mirror, VRAM extent, virtual over a
physical pool) read, written and counted by 32-bit atomics with the expected bytes after the fence, a raw store into a
mirror reaching guest memory when its value is done and staying in the mirror (the next submission copies nothing into
it, and a CPU write to a page of it is copied in), an unmapped address reading zero, dropping its write and named in
the miss log, a chunk committed after its pool was mapped reachable by the next submission, the lowest and the highest
mapped page resolving, a range a finalized pool left and another pool took reached through the new pool, and a bind that
grows past a mirror the table maps inside one submission of two raw-access stages (the table follows the map to the end
of the recording, the stage recorded after the bind reaches the pages it committed, and the bound and the raw views of
them are the same bytes). Each runs under the synchronization validation layer, which must report no error. The heap case measures what the invariant costs
(resident bytes, VRAM, the invariant's pass and the table's build) over a heap of OOT2_RAW_HEAP_MIB MiB, 256 by default.

Needs clang, the Vulkan SDK, the translator (py -3 tools/build_xlate.py) and a Vulkan device with every item the
renderer requires; skips without them, unless OOT2_REQUIRE_CLANG, OOT2_REQUIRE_VULKAN, OOT2_REQUIRE_XLATE or
OOT2_REQUIRE_GPU is set. Continuous integration has no device.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import (ROOT, build_program, case_env, clang_or_skip, fatal_lines, skip_on_partial_device,
                              vulkan_or_skip)
from test_xlate_host import EXE as XLATE, SKIP_NO_EXE, ampere, bit, compute_job, EXIT
from test_xlate_phi import B32, ldg, s2r, scheduled
import pool  # noqa: E402  (tools/shaders, put on the path by test_xlate_host)

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_raw_test.c"
LEVEL = "-O2"
EXIT_NO_DEVICE = 3
TID_X = 33
THREADS = 64
TABLE = (1 << 39) + 0x4000_0000   # the pointer table: vk_raw_test.c, TABLE
ENTRY = 32
CASES = ("kinds", "keep", "miss", "commit", "reuse", "edges", "grow")


def mov_imm(rd: int, imm: int) -> bytes:
    """MOV Rd, imm32 (0x802: ALU operand form 4, the immediate in bits 32..63)."""
    return scheduled(ampere(0x802, rd=rd, ra=255, rb=0, low_bits=(imm & 0xFFFFFFFF) << 32))


def imad_imm(rd: int, ra: int, imm: int, rc: int) -> bytes:
    """IMAD Rd, Ra, imm32, Rc (0x824), its predicate output the true predicate."""
    return scheduled(ampere(0x824, rd=rd, ra=ra, rb=0, low_bits=(imm & 0xFFFFFFFF) << 32, rc=rc, high_bits=7 << 81))


def stg(ra: int, rb: int, offset: int = 0) -> bytes:
    """STG.E [R<ra>.64 + offset], R<rb>, 32 bits."""
    return scheduled(ampere(0x386, ra=ra, rb=rb, rc=0, low_bits=(offset & 0xFFFFFF) << 40,
                            high_bits=bit(72) | bit(73, B32)))


def atomg_add(rd: int, ra: int, rb: int) -> bytes:
    """ATOMG.E.ADD.U32 Rd, [R<ra>.64], R<rb> (0x3a8, operation and type 0)."""
    return scheduled(ampere(0x3a8, rd=rd, ra=ra, rb=rb, rc=0, high_bits=bit(72)))


def raw_program() -> bytes:
    return b"".join([
        s2r(2, TID_X),
        mov_imm(20, TABLE & 0xFFFFFFFF),
        imad_imm(4, 2, ENTRY, 20),            # R4:R5 = the thread's entry
        mov_imm(5, TABLE >> 32),
        ldg(6, 4, 0), ldg(7, 4, 4),           # R6:R7 = the probe address
        ldg(8, 4, 8), ldg(9, 4, 12),          # R8:R9 = the atomic's address
        ldg(10, 6, 0),                        # the raw load
        stg(4, 10, 16),
        mov_imm(21, 0xA0000000),
        imad_imm(11, 2, 1, 21),
        stg(6, 11, 4),                        # the raw store
        mov_imm(12, 1),
        atomg_add(13, 8, 12),                 # the atomic
        stg(4, 13, 20),
    ]) + EXIT


_build: dict[str, object] = {}


def program() -> Path:
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vkraw-", ignore_cleanup_errors=True)
        _build["tmp"] = tmp
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def translate(out: Path) -> Path:
    """The raw program through the translator: its SPIR-V's path."""
    if XLATE is None:
        if os.environ.get("OOT2_REQUIRE_XLATE"):
            raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
        raise unittest.SkipTest(SKIP_NO_EXE)
    code = out / "raw.bin"
    code.write_bytes(raw_program())
    job = compute_job("raw", code, local=[THREADS, 1, 1])
    results, _ = pool.run_pool([XLATE], [(job, out)], 1, out / "logs", timeout=120)
    (got,) = results
    if (got.status, got.message) != ("ok", ""):
        raise AssertionError(f"the translator refused the raw program: {got.status} {got.message}")
    res = json.loads((out / "raw.res.json").read_text(encoding="utf-8"))
    ra = res["raw_access"]
    assert (ra["page_table"], ra["store_bits"], ra["push_offset"], ra["push_bytes"]) == (True, True, 0, 16), ra
    return out / "raw.spv"


def run(case: str, *args: str, env: dict[str, str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case, *args], capture_output=True, text=True, timeout=timeout,
                          env=case_env(env))


class VkRawTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-vkraw-spv-", ignore_cleanup_errors=True)
        cls.spv = translate(Path(cls.tmp.name))
        done = run("usable")
        skip_on_partial_device(done)
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if os.environ.get("OOT2_REQUIRE_GPU"):
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        assert done.returncode == 0, done.stdout + done.stderr

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def test_each_case_on_the_device_with_no_validation_error(self) -> None:
        for case in CASES:
            with self.subTest(case=case):
                done = run(case, str(self.spv), env={"OOT2_VK_VALIDATION": "sync"})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
                self.assertIn(f"ok {case}", done.stdout)
                self.assertEqual(fatal_lines(done.stderr), [])
                m = re.search(r"^validation errors (\d+) warnings (\d+)$", done.stdout, re.M)
                self.assertIsNotNone(m, done.stdout)
                self.assertEqual(m.group(1), "0", done.stdout + done.stderr[-4000:])
                print(f"\n{case}:\n" + "".join(f"  {line}\n" for line in done.stdout.splitlines()), end="")

    def test_what_the_invariant_costs_is_measured(self) -> None:
        # A small heap here; the research note's figures come from OOT2_RAW_HEAP_MIB=5632 (5.5 GiB) in a HEAVY slot
        mib = os.environ.get("OOT2_RAW_HEAP_MIB", "256")
        done = run("heap", str(self.spv), mib, timeout=1800)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
        self.assertIn("ok heap", done.stdout)
        self.assertRegex(done.stdout, r"first raw-access submission [\d.]+ ms .* table built in [\d.]+ ms")
        print(f"\nheap {mib} MiB:\n" + "".join(f"  {line}\n" for line in done.stdout.splitlines()), end="")


if __name__ == "__main__":
    unittest.main()
