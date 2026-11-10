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
them are the same bytes), and a previously bound mirror that grows in place with consistent bound/raw writes and
write-back, including when a GPU-only extent already occupies adjacent block space. The write-back cases (a 1024-thread
variant of the program, raw_program(big=True)) store into 10,101 pages the way the opening's chain does and check that
the device wrote them into guest memory byte for byte (the same hash as the build that copied them on the completion
thread), that a CPU write after the device's page and before the stamp is kept and reaches the mirror, that pages the
guest unmapped in between are left out, that pages it gave back and mapped again under a sink get a new sink or have their
stores counted lost (the mirror then takes the guest's bytes), never stamped as the GPU's, that a sink refused over pages
decommitted on its way stops nothing, and that a dropped sink is freed only once its submissions are done. Each runs under
the synchronization validation layer, which must report no error. The heap case measures what the invariant costs
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
CTAID_X = 37
THREADS = 64
BIG_THREADS = 1024   # raw_program_big: one group of the write-back cases
TABLE = (1 << 39) + 0x4000_0000   # the pointer table: vk_raw_test.c, TABLE
ENTRY = 32
CASES = ("kinds", "keep", "miss", "commit", "reuse", "edges", "grow", "grow_taken", "grow_guarded_head",
         "grow_guarded_tail")
# The write-back cases run the 1024-thread program (groups of 1024 threads, one entry each)
BIG_CASES = ("wb", "wbcpu", "wbunmap", "wbflight", "wbremap", "wbremapflight", "wbnosink", "wbrefused", "wbreap")
# What the runtime logs by name in those cases (stderr): the lost stores, the refused sink
LOGS = {
    "wbremapflight": "are lost (64 pages): the guest decommitted the page under the store's sink",
    "wbnosink": "are lost (64 pages): the guest decommitted the page under the store's sink",
    "wbrefused": "as the sink of a mirror's write-back was refused, and the guest no longer has all of it committed",
}


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


def raw_program(big: bool = False) -> bytes:
    """The program above; big: the thread's index is its group's times 1024 plus its place in the group (CTAID_X), so
    that groups of 1024 threads reach one entry each of a table larger than one group's."""
    index = [s2r(2, TID_X)]
    if big:
        index += [s2r(3, CTAID_X), imad_imm(2, 3, BIG_THREADS, 2)]
    return b"".join(index + [
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


def translate(out: Path, big: bool = False) -> Path:
    """The raw program (the 1024-thread one when big) through the translator: its SPIR-V's path."""
    if XLATE is None:
        if os.environ.get("OOT2_REQUIRE_XLATE"):
            raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
        raise unittest.SkipTest(SKIP_NO_EXE)
    name = "rawbig" if big else "raw"
    code = out / f"{name}.bin"
    code.write_bytes(raw_program(big))
    job = compute_job(name, code, local=[BIG_THREADS if big else THREADS, 1, 1])
    results, _ = pool.run_pool([XLATE], [(job, out)], 1, out / "logs", timeout=120)
    (got,) = results
    if (got.status, got.message) != ("ok", ""):
        raise AssertionError(f"the translator refused the raw program: {got.status} {got.message}")
    res = json.loads((out / f"{name}.res.json").read_text(encoding="utf-8"))
    ra = res["raw_access"]
    assert (ra["page_table"], ra["store_bits"], ra["push_offset"], ra["push_bytes"]) == (True, True, 0, 16), ra
    return out / f"{name}.spv"


def run(case: str, *args: str, env: dict[str, str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case, *args], capture_output=True, text=True, timeout=timeout,
                          env=case_env(env))


class VkRawTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-vkraw-spv-", ignore_cleanup_errors=True)
        cls.spv = translate(Path(cls.tmp.name))
        cls.spv_big = translate(Path(cls.tmp.name), big=True)
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

    def test_the_miss_diagnostics_name_the_stage_the_pool_and_the_memory_of_each_miss(self) -> None:
        # Setting raw_diag (OOT2_RAW_DIAG=1): after the dispatch the miss log is copied into a snapshot of its own and cleared
        # (the case does what the replay records), the completion thread hands the stage's 24 misses to the replay thread, and
        # the classification says what the log alone does not: the stage, that no pool holds the address, that the game
        # declared it mapped and the host committed it, that the map holds nothing there. One line a class, with the table and
        # the bound buffer that holds the missed pointer; the capture's raw-access counter holds the 24 misses.
        done = run("missdiag", str(self.spv), env={"OOT2_VK_VALIDATION": "sync", "OOT2_RAW_DIAG": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
        self.assertIn("ok missdiag", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        m = re.search(r"^validation errors (\d+) warnings (\d+)$", done.stdout, re.M)
        self.assertIsNotNone(m, done.stdout)
        self.assertEqual(m.group(1), "0", done.stdout + done.stderr[-4000:])
        self.assertIn("raw_diag is on", done.stderr)
        self.assertRegex(done.stderr, r"\+0x[0-9a-f]+ \[buffers read \d+ timeline value\(s\) after the miss's submission\]")
        self.assertIn("1 stage snapshot, 24 entries classified", done.stdout)
        for kind in ("load", "store", "atomic"):
            with self.subTest(kind=kind):
                self.assertRegex(done.stderr, rf"miss class B \({kind}/no-chunk/pool:none/decl:mapped/commit:yes/now:none\), "
                                              rf"first seen in dispatch synthetic\.dispatch, submission \d+, which the stage "
                                              rf"missed 24 times: 0x")
        self.assertNotIn("miss class A", done.stderr)   # nothing here is a page the table lacks
        self.assertIn("in no pool (the nearest pool below ends at", done.stderr)
        self.assertIn("the game declared it mapped; committed; the map has no backing for it now", done.stderr)
        self.assertIn("verdict B, an address in no pool the GPU could reach at that submission: the pointer is the question",
                      done.stderr)
        self.assertRegex(done.stderr, r"the table of the submission: span of \d+ chunks of 2 MiB from 0x[0-9a-f]+, \d+ leaf "
                                      r"blocks, \d+ pages \(\d+ mirrored\), built at map version \d+, the map is at \d+")
        self.assertRegex(done.stderr, r"address 0x[0-9a-f]+; the address is a word of bound uniform buffer 0 \[0x[0-9a-f]+, "
                                      r"\+0x1000\) at \+0x[0-9a-f]+")
        self.assertRegex(done.stderr, r"raw path: summary: raw-misses: 24 ")   # the numbers once more at the close
        self.assertRegex(done.stderr, r"raw path: summary: raw-miss B load/no-chunk/pool:none/decl:mapped/commit:yes/now:none: ")
        print("\nmissdiag:\n" + "".join(f"  {line}\n" for line in done.stderr.splitlines() if "raw path" in line), end="")

    def test_each_class_of_miss_is_named_by_the_facts_of_a_real_submission(self) -> None:
        # missclass: one miss a submission, each in another place. A page of a live pool the CPU never committed is C; a page
        # committed after the table was built is L (the next begin's pass learned it, so the map moved on and the table is
        # not the same version: never T); a miss in a pool finalized before the drain is E; a page of a virtual pool no
        # mapping request mapped is B, and one mapped over a physical page nothing committed is C by that page.
        done = run("missclass", str(self.spv), env={"OOT2_VK_VALIDATION": "sync", "OOT2_RAW_DIAG": "1"})
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
        self.assertIn("ok missclass", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        m = re.search(r"^validation errors (\d+) warnings (\d+)$", done.stdout, re.M)
        self.assertIsNotNone(m, done.stdout)
        self.assertEqual(m.group(1), "0", done.stdout + done.stderr[-4000:])
        want = {
            "C in a live pool": r"miss class C \(\w+/[a-z-]+/pool:cpu/decl:mapped/commit:no/now:none\)",
            "L after the table": r"miss class L \(\w+/[a-z-]+/pool:cpu/decl:mapped/commit:yes/now:(import|mirror)\)",
            "E in an ended pool": r"miss class E \(\w+/[a-z-]+/pool:cpu/[^)]*/ended\)",
            "B in a virtual pool": r"miss class B \(\w+/[a-z-]+/pool:virtual/decl:free/commit:unknown/now:\w+\)",
            "C by the physical page": r"miss class C \(\w+/[a-z-]+/pool:virtual/decl:mapped/commit:no/now:\w+\)",
        }
        for what, pattern in want.items():
            with self.subTest(what=what):
                self.assertRegex(done.stderr, pattern)
        self.assertNotIn("miss class T", done.stderr)   # nothing here is a page the table of the map's own version lacks
        self.assertNotIn("miss class A", done.stderr)
        self.assertRegex(done.stderr, r"built at map version (\d+), the map is at (?!\1\b)\d+")   # the L row's two versions differ

    def test_with_the_setting_off_a_miss_is_counted_into_the_capture_and_logged_as_it_was(self) -> None:
        done = run("miss", str(self.spv), env={"OOT2_VK_VALIDATION": "sync"})   # the case checks the capture counter and the stats
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
        self.assertIn("ok miss", done.stdout)
        self.assertNotIn("raw_diag", done.stderr)
        self.assertNotIn("miss class", done.stderr)
        self.assertRegex(done.stderr, r"submission \d+ missed the page table 24 times \(a miss reads 0 and drops its write\)")

    def test_the_write_back_of_a_movie_frame_reaches_guest_memory_whole(self) -> None:
        # wb: 10,101 pages (39.5 MiB, what the opening's chain writes a present) byte for byte and the time the raw path's
        # effect takes on the completion thread; wbcpu: a CPU write between the GPU's write-back and its stamp is kept;
        # wbunmap: pages the guest unmapped mid-flight are left out; wbremap and after: pages the guest gave back and
        # mapped again under a sink, a page with no sink, a refused sink, a dead sink's lifetime (vk_raw_test.c). The times
        # are printed, not judged.
        for case in BIG_CASES:
            with self.subTest(case=case):
                done = run(case, str(self.spv_big), env={"OOT2_VK_VALIDATION": "sync"})
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
                self.assertIn(f"ok {case}", done.stdout)
                self.assertEqual(fatal_lines(done.stderr), [])
                if case in LOGS:
                    self.assertIn(LOGS[case], done.stderr)
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
