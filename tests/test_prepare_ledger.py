# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the ledger step of `oot2 prepare` (tools/prepare/ledger.py): ledger rows turned into translated variants.

Every input is synthetic and built inside the test: a container from the .nup writer of tests/test_prepare_extract.py
whose RomFS holds shader archives assembled with the builders of tests/test_shader_split.py, and a ledger.jsonl written
here in the format runtime/nvn/nvn_ledger.c writes. No game file is read. Most tests run a stand-in translator, a
Python script that speaks the pool's line protocol, writes a module that depends on the job's answers and a
description that lists the slots the program reads (the markers TEXn, IMGn, ATTRn and FOUTn in its code) as the real
translator's res.json does, so they run on every runner. The real translator (tools/build_xlate.py; the Windows CI job
builds it and sets OOT2_REQUIRE_XLATE) translates a vertex program that samples a texture, and without it that test
skips. The runtime reader test needs clang for x86-64 (OOT2_CC, else clang on PATH; OOT2_REQUIRE_CLANG makes its absence
an error, as CI sets it).

What is proved, each with a counter-run kept outside the repository (the commit's message names the runs):

- rows whose answers a module reads and the default translation did not use become variants, whose modules differ from
  the base pack's, several to one program, and the variants pack's rows carry the digest of the answers;
- a default row, and an answer the module does not read, add nothing, and rows that differ only in unread slots give
  one variant;
- a variant goes into the variants file and never into the base pack, which keeps its bytes;
- a new ledger runs this step and nothing else, and a new translator runs the translate step and this one after it;
- the prepare map connects a full ledger digest to its projected variant digest, and the runtime resolves it before the
  base pack's default row;
- a variant the translator refuses, or that gets no answer, is a named failure record, rows that cannot be used are
  counted and skipped, and a damaged ledger stops the step by line.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import cache_pack as cp  # noqa: E402
import test_prepare_extract as nupt  # noqa: E402  (FakeCheck)
import test_prepare_translate as trt  # noqa: E402  (Base, container, bnsh, graphics, compute, job_key)
import test_shader_cache_pack as shct  # noqa: E402  (the C reader's build and its query lines)
import test_xlate_host as xlatet  # noqa: E402  (EXE)
from prepare import check, extract, ledger, translate  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, run_steps  # noqa: E402

EXIT = trt.EXIT


# --- the stand-in translator ----------------------------------------------------------------------------------------

STUB = textwrap.dedent('''\
    """A stand-in translator for the ledger step: the pool's line protocol; a module that depends on the program's code
    and on the job's answers for the slots it reads; and a description that lists those slots and the queries asked,
    from the markers TEXn, SAMPn, IMGn, ATTRn and FOUTn in its code, with the type used at each (the job's answer, else
    the default), as res.json does.
    TEXn is a texture the translator asks the type of (a sampler_type query: the module follows the answer), SAMPn a
    texture the instruction types, so nothing is asked, as the real translator does for a TEX instruction.
    SAMEALL asks and answers its default whatever the job says; BLINDMOD builds a module that ignores the answers but
    a description that follows them. REFUSE in the code fails the job. NOCUBE fails a job whose tex answers hold
    "cube". CRASHVAR ends this process when the job sets any answer."""
    import base64, hashlib, json, os, re, struct, sys
    IDENTITY = os.environ.get("STUB_IDENTITY") or hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    if sys.argv[1:] == ["--version"]:
        print(f"oot2-xlate {IDENTITY} proto=2 spirv=1.3 target=vulkan1.3 optimizer=sound")
        sys.exit(0)

    def pick(table, slot, default):
        return table.get(str(slot), table.get("*", default))

    def slots(marker, code):
        return sorted({int(n) for n in re.findall(marker + rb"(\\d+)", code)})

    for line in sys.stdin:
        request = json.loads(line)
        job = request["job"]
        key = job["key"]
        with open(job["code"], "rb") as f:
            f.seek(job.get("code_offset", 0))
            code = f.read(job.get("code_length", 1 << 20))
        tex, img, attr, fout = (job.get(name) or {} for name in ("tex", "img", "attr", "fout"))
        if b"REFUSE" in code:
            print(f"fail {key} 5 unsupported_opcode: Ampere instruction 123 at 0x0 is not implemented", flush=True)
            continue
        if b"NOCUBE" in code and "cube" in tex.values():
            print(f"fail {key} 5 bad_job: no cube texture in this program", flush=True)
            continue
        if b"CRASHVAR" in code and (tex or img or attr or fout):
            os._exit(3)
        answers = {"tex": tex, "img": img, "attr": attr, "fout": fout}
        reads = {"tex": slots(b"TEX", code), "img": slots(b"IMG", code),
                 "attr": slots(b"ATTR", code) if job["stage"] == "vs" else [],
                 "fout": slots(b"FOUT", code) if job["stage"] == "ps" else []}
        same = b"SAMEALL" in code      # asks for the types and uses its default whatever the job answers
        blind = b"BLINDMOD" in code    # a module that ignores every answer, with a description that follows them
        used = {name: {s: v for s, v in table.items() if s != "*" and int(s) in reads[name]}
                for name, table in answers.items()}
        effective = {} if same or blind else {name: table for name, table in used.items() if table}
        text = json.dumps(effective, sort_keys=True, separators=(",", ":"))
        module = struct.pack("<5I", 0x07230203, 0x00010300, 0, 1, 0) + hashlib.sha256(code + text.encode()).digest()
        tex_in, img_in = ({} if same else tex), ({} if same else img)
        default = {"type": "2d", "fmt": "R8G8B8A8Unorm"}
        def kind(table, n):
            return "job" if str(n) in table else "default"
        def query(name, slot, islot, answer, table, n):
            return {"kind": name, "handle": 0, "cbuf_slot": -1, "slot": slot, "image_slot": islot, "answer": answer,
                    "answered_from": kind(table, n), "count": 1}
        queries = [{"kind": "register_texture", "handle": 0, "cbuf_slot": -1, "slot": n, "image_slot": -1,
                    "answer": "", "answered_from": "", "count": 1} for n in slots(rb"(?:TEX|SAMP)", code)]
        queries += [query("sampler_type", n, -1, pick(tex_in, n, "2d"), tex_in, n) for n in reads["tex"]]
        for n in reads["img"]:
            queries += [query("sampler_type", 32 + n, n, pick(img_in, n, default)["type"], img_in, n),
                        query("texture_format", -1, n, pick(img_in, n, default)["fmt"], img_in, n)]
        res = {"schema": 2, "key": key, "stage": job["stage"], "translator_id": IDENTITY, "spv_bytes": len(module),
               "spv_sha256": hashlib.sha256(module).hexdigest(),
               "answers": {**answers, "early_fragment_tests": False, "heap_images": 33025, "heap_samplers": 4096},
               "textures": [{"kind": "combined", "type": pick(tex_in, n, "2d"), "c2_slot": n} for n in reads["tex"]]
                           + [{"kind": "combined", "type": "2d", "c2_slot": n} for n in slots(b"SAMP", code)],
               "images": [{"type": pick(img_in, n, default)["type"], "format": pick(img_in, n, default)["fmt"],
                           "c2_slot": n} for n in reads["img"]],
               "queries": queries, "queries_dropped": 0,
               "vertex_inputs": [{"location": n, "component": 0, "components": 4, "type": "float32", "used": True,
                                  "answer": pick(attr, n, "float")} for n in reads["attr"]]
                                if job["stage"] == "vs" else None,
               "interface": {"inputs": [], "outputs": [{"location": n, "component": 0, "components": 4,
                                                          "type": "float32", "used": True}
                                                         for n in reads["fout"]]}}
        m64 = base64.b64encode(module).decode()
        d64 = base64.b64encode(json.dumps(res, sort_keys=True).encode()).decode()
        print(f"ok {key} 100 {m64} {d64}", flush=True)
''')

# --- programs, ledgers ----------------------------------------------------------------------------------------------

def code(*marks: str) -> bytes:
    """A program's instructions: the markers the stand-in reads, padded to whole instruction slots, then the exit."""
    text = " ".join(marks).encode()
    return text.ljust(-(-len(text) // 16) * 16, b"\0") + EXIT


VS = trt.graphics(1, 1, code("TEX3", "ATTR0", "ATTR1"))        # reads texture 3, attributes 0 and 1
PS = trt.graphics(5, 2, code("TEX3", "TEX4", "FOUT0", "FOUT1"))
CS = trt.compute(3, code("TEX0", "IMG1"))                        # reads texture 0 and image 1
PLAIN = trt.compute(4, EXIT)                                     # reads nothing
NOCUBE = trt.compute(5, code("TEX2", "NOCUBE"))
CRASHVAR = trt.compute(6, code("TEX2", "CRASHVAR"))
REFUSED = trt.compute(7, code("TEX0", "REFUSE"))
SAMPLED = trt.compute(8, code("SAMP2"))                          # samples texture 2 by its instruction: asks nothing
SAMEALL = trt.compute(9, code("TEX1", "SAMEALL"))                # asks for texture 1 and ignores the answer
BLIND = trt.compute(10, code("TEX1", "BLINDMOD"))                # a module that ignores the answer, a description that follows it
KEY = {"vs": trt.job_key("vs", VS), "ps": trt.job_key("ps", PS), "cs": trt.job_key("cs", CS),
       "plain": trt.job_key("cs", PLAIN), "nocube": trt.job_key("cs", NOCUBE),
       "crashvar": trt.job_key("cs", CRASHVAR), "refused": trt.job_key("cs", REFUSED),
       "sampled": trt.job_key("cs", SAMPLED), "sameall": trt.job_key("cs", SAMEALL),
       "blind": trt.job_key("cs", BLIND)}
IMG_3D = {"fmt": "R32Float", "type": "3d"}
IMG_DEFAULT = {"fmt": "R8G8B8A8Unorm", "type": "2d"}


def row(key: str, kind: str, **answers: dict) -> dict:
    """A ledger row as nvn_ledger.c writes it (the attachment formats are not read by this step)."""
    out = {"key": key, "kind": kind, "answers": answers, "digest": cp.digest_text(cp.answers_digest(answers))}
    if kind in ("vs", "ps"):
        out["attach"] = {"color": [37], "depth": None, "samples": 1}
    return out


def ledger_text(rows: list[dict]) -> str:
    """The file's text: the header, then the rows as compact JSON sorted by their text."""
    lines = sorted(json.dumps(r, separators=(",", ":"), sort_keys=True) for r in rows)
    return "\n".join([json.dumps(ledger.HEADER, separators=(",", ":")), *lines]) + "\n"


def digest_of(**answers: dict) -> bytes:
    return cp.answers_digest(answers)


class LedgerBase(trt.Base):
    """The check (its duty to the later steps only), extract, translate and ledger steps through the runner."""

    def setUp(self):
        super().setUp()
        self.stub.write_text(STUB, encoding="utf-8")
        archives = {"Effect/fx.bfsha": trt.bnsh({0: VS, 4: PS}, {5: CS}, {5: PLAIN}, {5: NOCUBE}, {5: CRASHVAR},
                                                {5: REFUSED}, {5: SAMPLED}, {5: SAMEALL}, {5: BLIND})}
        self.dump = self.write_dump(trt.container(archives))

    def run_steps(self, dump: Path, xlate: Path, dest: Path | None = None, verify: bool = False) -> tuple[str, str]:
        out = io.StringIO()
        dest = dest or self.dest
        opts = types.SimpleNamespace(xlate=str(xlate), translate_jobs=2, cc=None, plain_modules=None,
                                     zbic_archive=None)
        ctx = Context(dest, opts, Reporter(out, out), state=statefile.load(dest), verify=verify)
        steps = [nupt.FakeCheck(dump), extract.Extract(), translate.Translate(), ledger.Ledger()]
        return run_steps(steps, ctx), out.getvalue()

    def go(self, rows: list[dict] | None = None, **kwargs) -> str:
        """Write the ledger (None: leave it as it is), run the steps and return their text; they must succeed."""
        if rows is not None:
            self.ledger_path().write_text(ledger_text(rows), encoding="utf-8")
        result, text = self.run_steps(self.dump, self.stub, **kwargs)
        self.assertEqual(result, "ok", text)
        return text

    def ledger_path(self, dest: Path | None = None) -> Path:
        (dest or self.dest).mkdir(parents=True, exist_ok=True)
        return (dest or self.dest) / ledger.LEDGER_FILE

    def variants_path(self, dest: Path | None = None) -> Path:
        return (dest or self.dest) / translate.SHADERS / cp.VARIANTS_FILE

    def map_path(self, dest: Path | None = None) -> Path:
        return (dest or self.dest) / translate.SHADERS / ledger.VARIANT_MAP_FILE

    def variants(self, dest: Path | None = None) -> cp.Pack:
        return cp.open_pack(self.variants_path(dest), role=cp.VARIANTS)

    def record(self) -> dict:
        return self.steps()["ledger"]

    def base_module(self, key: str) -> bytes:
        pack = self.pack()
        return pack.read(pack.find(key))[0]

    def description(self, pack: cp.Pack, key: str, digest: bytes) -> dict:
        return json.loads(pack.read(pack.find(key, digest))[1])


# --- rows become variants -------------------------------------------------------------------------------------------

class VariantTest(LedgerBase):
    def test_rows_turn_into_variants_whose_modules_differ_from_the_default_ones(self):
        both = {"tex": {"3": "3d"}, "attr": {"0": "sint"}}
        rows = [
            row(KEY["vs"], "vs", tex={"3": "3d"}),
            row(KEY["vs"], "vs", attr={"0": "sint"}),
            row(KEY["vs"], "vs", **both),                                    # a third variant of one program
            row(KEY["ps"], "ps", fout={"0": "uint"}),
            row(KEY["ps"], "ps", tex={"4": "cube"}),
            row(KEY["cs"], "cs", img={"1": IMG_3D}),
            row(KEY["cs"], "cs", tex={"0": "2darray"}, img={"1": IMG_3D}),
        ]
        text = self.go(rows)
        want = {(KEY["vs"], digest_of(tex={"3": "3d"})): {"tex": {"3": "3d"}},
                (KEY["vs"], digest_of(attr={"0": "sint"})): {"attr": {"0": "sint"}},
                (KEY["vs"], digest_of(**both)): both,
                (KEY["ps"], digest_of(fout={"0": "uint"})): {"fout": {"0": "uint"}},
                (KEY["ps"], digest_of(tex={"4": "cube"})): {"tex": {"4": "cube"}},
                (KEY["cs"], digest_of(img={"1": IMG_3D})): {"img": {"1": IMG_3D}},
                (KEY["cs"], digest_of(tex={"0": "2darray"}, img={"1": IMG_3D})):
                    {"tex": {"0": "2darray"}, "img": {"1": IMG_3D}}}
        pack = self.variants()
        self.assertEqual({(r.key, r.answers) for r in pack.rows}, set(want))
        self.assertEqual(cp.verify_pack(self.variants_path(), role=cp.VARIANTS), [])
        self.assertEqual(pack.translator, self.pack().translator)
        self.assertEqual(pack.schema, cp.DESC_SCHEMA)
        for (key, digest), answers in want.items():
            module = pack.read(pack.find(key, digest))[0]
            self.assertNotEqual(module, self.base_module(key), (key, answers))
            desc = self.description(pack, key, digest)
            self.assertEqual({k: v for k, v in desc["answers"].items() if k in cp.ANSWER_FIELDS and v}, answers)
            self.assertEqual(desc["key"], key)
        # one module for each variant: no two share their bytes
        self.assertEqual(len({pack.read(r)[0] for r in pack.rows}), len(want))
        rec = self.record()
        self.assertEqual((rec["ledger_rows"], rec["variants"], rec["translated"], rec["failed"], rec["default_rows"]),
                         (7, 7, 7, {}, 0))
        self.assertEqual(rec["pack_bytes"], pack.size)
        self.assertEqual(rec["ledger_sha256"], hashlib.sha256(self.ledger_path().read_bytes()).hexdigest())
        self.assertIn("ledger: 7 of 7 variants translated", text)
        self.assertFalse((self.dest / check.TEMP / ledger.SCRATCH).exists())
        self.assertEqual(cp.open_pack(self.pack_path(), role=cp.BASE).translator, rec["translator"]["sha256"])

    def test_a_default_row_adds_none(self):
        rows = [row(KEY["vs"], "vs"), row(KEY["ps"], "ps"), row(KEY["cs"], "cs"), row(KEY["plain"], "cs"),
                row(KEY["vs"], "vs", tex={"3": "2d"}),                       # what the default translation used
                row(KEY["vs"], "vs", attr={"0": "float", "1": "float"}),
                row(KEY["ps"], "ps", fout={"0": "float", "1": "float"}, tex={"3": "2d", "4": "2d"}),
                row(KEY["cs"], "cs", img={"1": IMG_DEFAULT}, tex={"0": "2d"})]
        text = self.go(rows)
        self.assertFalse(self.variants_path().exists())
        rec = self.record()
        self.assertEqual((rec["ledger_rows"], rec["default_rows"], rec["variants"], rec["pack_bytes"]),
                         (len(rows), len(rows), 0, 0))
        self.assertIn("ledger: no row asks for a variant, so no variants pack is written", text)

    def test_an_answer_the_module_does_not_read_adds_none(self):
        rows = [row(KEY["plain"], "cs", tex={"3": "3d"}, img={"1": IMG_3D}),     # a program that reads nothing
                row(KEY["vs"], "vs", tex={"7": "3d"}, attr={"5": "uint"}),        # slots it does not read
                row(KEY["ps"], "ps", fout={"4": "uint"}, tex={"9": "cube"}),
                row(KEY["cs"], "cs", img={"5": IMG_3D}, tex={"6": "cube"}),
                row(KEY["sampled"], "cs", tex={"2": "3d"})]                      # sampled, never asked: nothing to answer
        self.go(rows)
        self.assertFalse(self.variants_path().exists())
        self.assertEqual((self.record()["default_rows"], self.record()["candidates"], self.record()["variants"]),
                         (5, 0, 0))

    def test_only_the_answers_the_module_reads_make_the_variant_and_rows_that_differ_elsewhere_give_one(self):
        rows = [row(KEY["vs"], "vs", tex={"3": "3d", "7": "3d"}),
                row(KEY["vs"], "vs", tex={"3": "3d", "8": "cube"}, attr={"5": "uint"}),
                row(KEY["vs"], "vs", tex={"3": "3d"})]
        self.go(rows)
        pack = self.variants()
        self.assertEqual([(r.key, r.answers) for r in pack.rows], [(KEY["vs"], digest_of(tex={"3": "3d"}))])
        self.assertEqual(self.description(pack, KEY["vs"], digest_of(tex={"3": "3d"}))["answers"]["tex"], {"3": "3d"})
        rec = self.record()
        self.assertEqual((rec["variants"], rec["variant_rows"], rec["merged_rows"], rec["default_rows"]), (1, 3, 2, 0))

    def test_only_the_slot_whose_answer_differs_from_the_default_counts(self):
        # The pixel program reads textures 3 and 4 as 2D; "3d" differs on slot 3, "2d" is the default on slot 4.
        self.go([row(KEY["ps"], "ps", tex={"3": "3d", "4": "2d"})])
        self.assertEqual([r.answers for r in self.variants().rows], [digest_of(tex={"3": "3d"})])

    def test_an_image_answer_that_differs_in_its_type_or_only_in_its_format_counts(self):
        in_type = {"fmt": "R8G8B8A8Unorm", "type": "3d"}
        in_format = {"fmt": "R32Float", "type": "2d"}
        self.go([row(KEY["cs"], "cs", img={"1": in_type}), row(KEY["cs"], "cs", img={"1": in_format}),
                 row(KEY["cs"], "cs", img={"1": IMG_DEFAULT})])
        self.assertEqual({r.answers for r in self.variants().rows},
                         {digest_of(img={"1": in_type}), digest_of(img={"1": in_format})})
        self.assertEqual(self.record()["default_rows"], 1)

    def test_unknown_image_format_is_translated_and_maps_the_full_row_digest(self):
        complete = {"img": {"1": {"fmt": "Unknown", "type": "2d"}}, "tex": {"9": "cube"}}
        self.go([row(KEY["cs"], "cs", **complete)])
        projected = digest_of(img={"1": complete["img"]["1"]})
        full = digest_of(**complete)
        variant = self.variants().find(KEY["cs"], projected)
        self.assertFalse(variant.failed)
        self.assertNotEqual(full, projected)
        _, mappings = ledger.read_variant_map(self.map_path())
        self.assertEqual(mappings[(KEY["cs"], full)], projected)

    def test_a_variant_that_is_the_base_entry_again_is_not_packed(self):
        # SAMEALL asks for texture 1 and ignores the answer: its variant is the base module and description again.
        text = self.go([row(KEY["sameall"], "cs", tex={"1": "3d"}), row(KEY["vs"], "vs", tex={"3": "3d"})])
        self.assertEqual([(r.key, r.answers) for r in self.variants().rows], [(KEY["vs"], digest_of(tex={"3": "3d"}))])
        rec = self.record()
        self.assertEqual((rec["candidates"], rec["translated"], rec["identical"], rec["variants"]), (2, 2, 1, 1))
        self.assertIn("1 came out as the base module and description and are not packed", text)
        self.assertEqual(cp.verify_pack(self.variants_path(), role=cp.VARIANTS), [])
        # nothing but such a variant: no pack at all
        text = self.go([row(KEY["sameall"], "cs", tex={"1": "3d"})])
        self.assertFalse(self.variants_path().exists())
        self.assertEqual((self.record()["candidates"], self.record()["identical"], self.record()["variants"]), (1, 1, 0))
        self.assertIn("no variants pack is written", text)
        self.assertIn("ledger: kept", self.go())

    def test_a_module_that_ignores_the_answer_still_gives_a_variant_when_its_description_follows_it(self):
        # BLINDMOD builds the base module whatever the answer, but its description records the type that was answered.
        self.go([row(KEY["blind"], "cs", tex={"1": "3d"})])
        pack = self.variants()
        digest = digest_of(tex={"1": "3d"})
        module, desc = pack.read(pack.find(KEY["blind"], digest))
        self.assertEqual(module, self.base_module(KEY["blind"]))
        self.assertEqual([t["type"] for t in json.loads(desc)["textures"]], ["3d"])
        self.assertEqual((self.record()["candidates"], self.record()["identical"], self.record()["variants"]), (1, 0, 1))


# --- variants never enter the base pack -----------------------------------------------------------------------------

class SeparationTest(LedgerBase):
    def test_a_variant_goes_to_the_variants_file_and_the_base_pack_keeps_its_bytes(self):
        self.go([])
        base_bytes = self.pack_path().read_bytes()
        base_stat = self.pack_path().stat().st_mtime_ns
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"}), row(KEY["cs"], "cs", img={"1": IMG_3D})])
        self.assertEqual(self.pack_path().read_bytes(), base_bytes)
        self.assertEqual(self.pack_path().stat().st_mtime_ns, base_stat)
        base = self.pack()
        self.assertTrue(base.rows)
        self.assertTrue(all(r.answers == cp.ZERO_DIGEST for r in base.rows))
        variants = self.variants()
        self.assertEqual(len(variants.rows), 2)
        self.assertTrue(all(r.answers != cp.ZERO_DIGEST for r in variants.rows))
        # each pack opens as its own role only
        with self.assertRaises(cp.PackError) as e:
            cp.open_pack(self.variants_path(), role=cp.BASE)
        self.assertEqual(e.exception.code, "ROLE")
        with self.assertRaises(cp.PackError) as e:
            cp.open_pack(self.pack_path(), role=cp.VARIANTS)
        self.assertEqual(e.exception.code, "ROLE")
        self.assertEqual(self.variants_path().name, cp.VARIANTS_FILE)
        self.assertNotEqual(self.variants_path(), self.pack_path())

    def test_a_ledger_without_a_variant_leaves_no_variants_file_and_removes_a_stale_one(self):
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])
        self.assertTrue(self.variants_path().exists())
        self.go([row(KEY["vs"], "vs")])
        self.assertFalse(self.variants_path().exists())
        self.assertEqual(self.record()["variants"], 0)


# --- keys and reruns ------------------------------------------------------------------------------------------------

class KeyTest(LedgerBase):
    def test_a_first_prepare_has_no_ledger_and_writes_no_variants(self):
        text = self.go()
        self.assertIn("ledger: started (not done yet)", text)
        self.assertIn(f"ledger: no {ledger.LEDGER_FILE} in the output folder yet", text)
        self.assertEqual((self.record()["ledger_rows"], self.record()["variants"], self.record()["ledger_sha256"]),
                         (0, 0, "none"))
        self.assertFalse(self.variants_path().exists())
        text = self.go()
        self.assertIn("ledger: kept (done, inputs unchanged)", text)

    def test_a_changed_ledger_runs_this_step_and_nothing_else(self):
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])
        first = self.variants()
        base_bytes = self.pack_path().read_bytes()
        text = self.go()
        for step in ("extract", "translate", "ledger"):
            self.assertIn(f"{step}: kept", text)
        self.assertEqual(self.variants_path().read_bytes(), first.path.read_bytes())

        text = self.go([row(KEY["vs"], "vs", tex={"3": "3d"}), row(KEY["cs"], "cs", img={"1": IMG_3D})])
        self.assertIn("ledger: started (its inputs changed)", text)
        self.assertIn("extract: kept", text)
        self.assertIn("translate: kept", text)
        self.assertNotIn("translate: started", text)
        self.assertEqual(len(self.variants().rows), 2)
        self.assertEqual(self.pack_path().read_bytes(), base_bytes)

        # a ledger back to the first text gives the first pack again, and no ledger at all gives none
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])
        self.assertEqual(self.variants_path().read_bytes(), first.path.read_bytes())
        self.ledger_path().unlink()
        text = self.go()
        self.assertIn("ledger: started (its inputs changed)", text)
        self.assertIn("translate: kept", text)
        self.assertFalse(self.variants_path().exists())

    def test_a_ledger_that_changes_only_what_a_module_does_not_read_runs_the_step_and_ends_with_the_same_pack(self):
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])
        first = self.variants_path().read_bytes()
        text = self.go([row(KEY["vs"], "vs", tex={"3": "3d", "7": "3d"})])
        self.assertIn("ledger: started (its inputs changed)", text)
        self.assertEqual(self.variants_path().read_bytes(), first)

    def test_a_new_translator_runs_the_translate_step_and_then_this_one(self):
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])
        old = self.variants().translator
        with open(self.stub, "a", encoding="utf-8") as fh:
            fh.write("# another build of the translator\n")
        text = self.go()
        self.assertIn("extract: kept", text)
        self.assertIn("translate: started (its inputs changed)", text)
        self.assertIn("ledger: started (its inputs changed)", text)
        new = hashlib.sha256(self.stub.read_bytes()).hexdigest()
        self.assertEqual((self.variants().translator, self.pack().translator), (new, new))
        self.assertNotEqual(new, old)
        self.assertEqual(self.record()["translator"]["sha256"], new)
        self.assertEqual(cp.verify_pack(self.variants_path(), role=cp.VARIANTS), [])

    def test_the_key_folds_the_ledger_the_translator_and_the_translate_key(self):
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])

        def own_key() -> str:
            ctx = Context(self.dest, types.SimpleNamespace(xlate=str(self.stub)), Reporter(io.StringIO(), io.StringIO()))
            return ledger.Ledger().key(ctx)

        def chained(translate_key: str) -> str:
            ctx = Context(self.dest, types.SimpleNamespace(xlate=str(self.stub)), Reporter(io.StringIO(), io.StringIO()))
            ctx.keys["translate"] = translate_key
            ctx.decide(ledger.Ledger())
            return ctx.keys["ledger"]

        same = own_key()
        self.assertEqual(own_key(), same)
        with mock.patch.object(check, "XLATE_OPTIMIZER", "debug"):      # the form the variants pack holds is in the key
            self.assertNotEqual(own_key(), same)
        self.ledger_path().write_text(ledger_text([row(KEY["vs"], "vs", tex={"3": "cube"})]), encoding="utf-8")
        other_ledger = own_key()
        self.assertNotEqual(other_ledger, same)
        self.ledger_path().unlink()
        self.assertNotEqual(own_key(), other_ledger)
        self.assertNotEqual(own_key(), same)
        with open(self.stub, "a", encoding="utf-8") as fh:
            fh.write("# another build of the translator\n")
        self.assertNotEqual(own_key(), same)
        self.assertNotEqual(chained("a"), chained("b"))

    def test_the_output_is_checked_and_verified(self):
        self.go([row(KEY["vs"], "vs", tex={"3": "3d"})])
        self.variants_path().unlink()
        text = self.go()
        self.assertIn("ledger: started (its output is missing)", text)
        self.assertEqual(len(self.variants().rows), 1)
        # --verify finds a damaged entry and runs the step again
        data = bytearray(self.variants_path().read_bytes())
        data[-5] ^= 0xFF
        self.variants_path().write_bytes(bytes(data))
        text = self.go(verify=True)
        self.assertIn("the output of step ledger failed its full check; it runs again", text)
        self.assertEqual(cp.verify_pack(self.variants_path(), role=cp.VARIANTS), [])
        self.assertIn("translate: kept", text)
        self.map_path().unlink()
        self.assertIn("ledger: started (its output is missing)", self.go())
        self.assertTrue(self.map_path().exists())
        # a variants pack with no variant recorded must not exist
        self.go([row(KEY["vs"], "vs")])
        shutil.copy(self.pack_path(), self.variants_path())
        self.assertIn("ledger: started (its output is missing)", self.go())
        self.assertFalse(self.variants_path().exists())


# --- failures and rows that cannot be used --------------------------------------------------------------------------

class FailureTest(LedgerBase):
    def test_a_variant_the_translator_refuses_is_packed_as_a_named_failure(self):
        text = self.go([row(KEY["nocube"], "cs", tex={"2": "cube"}), row(KEY["nocube"], "cs", tex={"2": "3d"})])
        pack = self.variants()
        refused = pack.find(KEY["nocube"], digest_of(tex={"2": "cube"}))
        self.assertTrue(refused.failed)
        self.assertTrue(pack.read(refused)[1].decode("utf-8").startswith("fail: bad_job: no cube texture"))
        good = pack.find(KEY["nocube"], digest_of(tex={"2": "3d"}))
        self.assertFalse(good.failed)
        self.assertEqual(cp.verify_pack(self.variants_path(), role=cp.VARIANTS), [])
        rec = self.record()
        self.assertEqual((rec["variants"], rec["translated"], rec["failed"]), (2, 1, {"fail": 1}))
        self.assertEqual(rec["failed_keys"], [f"{KEY['nocube']} {cp.digest_text(digest_of(tex={'2': 'cube'}))}"])
        self.assertIn("warning: ledger: 1 of 2 variants did not translate (fail 1); each is packed as a failure record",
                      text)

    def test_a_variant_that_gets_no_answer_twice_is_packed_as_a_crash_record(self):
        text = self.go([row(KEY["crashvar"], "cs", tex={"2": "3d"})])
        self.assertIn("ledger: 1 variant(s) got no answer; each is tried again alone", text)
        pack = self.variants()
        (r,) = pack.rows
        self.assertTrue(r.failed)
        self.assertTrue(pack.read(r)[1].decode("utf-8").startswith("crash:"))
        self.assertEqual(self.record()["failed"], {"crash": 1})
        # the base translation of the same program succeeded: only the answers made it crash
        self.assertFalse(self.pack().find(KEY["crashvar"]).failed)

    def test_rows_that_cannot_be_used_are_counted_named_and_skipped(self):
        other = "0123456789abcdef.000"
        rows = [row(other, "cs", tex={"0": "3d"}),                          # a program the split does not hold
                row(KEY["vs"], "ps", tex={"3": "3d"}),                      # another stage than its program's
                row(KEY["refused"], "cs", tex={"0": "3d"}),                 # a program the base pack holds as a failure
                row(KEY["vs"], "vs", tex={"3": "3d"})]                      # and one that is used
        text = self.go(rows)
        rec = self.record()
        self.assertEqual((rec["unknown_programs"], rec["other_stage"], rec["base_failed"], rec["variants"]), (1, 1, 1, 1))
        self.assertTrue(self.pack().find(KEY["refused"]).failed)
        self.assertIn(f"warning: ledger: 1 row(s) name a program the split does not hold, so they are skipped; the "
                      f"first: {other}", text)
        self.assertIn(f"1 row(s) name another stage than their program's, so they are skipped; the first: {KEY['vs']}",
                      text)
        self.assertIn(f"1 row(s) name a program the base pack holds as a failure record, so they are skipped; the "
                      f"first: {KEY['refused']}", text)
        self.assertEqual([r.key for r in self.variants().rows], [KEY["vs"]])


class DamagedLedgerTest(LedgerBase):
    def test_a_ledger_that_is_not_what_the_writer_writes_stops_the_step_by_line(self):
        good = json.dumps(row(KEY["vs"], "vs", tex={"3": "3d"}), separators=(",", ":"))
        header = json.dumps(ledger.HEADER, separators=(",", ":"))
        wrong = row(KEY["vs"], "vs", tex={"3": "3d"})
        wrong["digest"] = "0"
        cases = {
            "no header": (f"{good}\n", "line 1: the first line is not the header"),
            "another version": ('{"oot2_ledger":2}\n', "line 1: the first line is not the header"),
            "not JSON": (f"{header}\n{good}\n{{not json\n", "line 3: it is not a JSON object"),
            "not an object": (f"{header}\n[1]\n", "line 2: it is not a JSON object"),
            "bad key": (f"{header}\n" + json.dumps({**row(KEY["vs"], "vs"), "key": "bad key"}) + "\n",
                        "line 2: the key is not"),
            "bad kind": (f"{header}\n" + json.dumps({**row(KEY["vs"], "vs"), "kind": "gs"}) + "\n",
                         "line 2: the stage kind 'gs'"),
            "unknown answer": (f"{header}\n" + json.dumps(row(KEY["vs"], "vs", blend={"0": "on"})) + "\n",
                               "line 2: the answers are not an object of attr, fout, img, tex"),
            "bad texture type": (f"{header}\n" + json.dumps(row(KEY["vs"], "vs", tex={"3": "4d"})) + "\n",
                                 "line 2: the tex answer \"4d\" for slot 3"),
            "bad slot": (f"{header}\n" + json.dumps(row(KEY["vs"], "vs", attr={"x": "sint"})) + "\n",
                         "line 2: the attr slot 'x'"),
            "bad image": (f"{header}\n" + json.dumps(row(KEY["cs"], "cs", img={"1": {"type": "3d"}})) + "\n",
                          "line 2: the img answer"),
            "digest": (f"{header}\n{good}\n" + json.dumps(wrong) + "\n", "line 3: the digest does not match"),
        }
        for name, (text, why) in cases.items():
            with self.subTest(name):
                self.ledger_path().write_text(text, encoding="utf-8")
                result, out = self.run_steps(self.dump, self.stub)
                self.assertEqual(result, "failed", out)
                self.assertIn(f"{ledger.LEDGER_FILE}, line", out)
                self.assertIn(why, out)
                self.assertNotIn("ledger", self.steps())
                self.assertFalse(self.variants_path().exists())
                self.assertIn("translate", self.steps())
        self.ledger_path().write_bytes(b"\xff\xfe\x00")
        result, out = self.run_steps(self.dump, self.stub)
        self.assertEqual(result, "failed", out)
        self.assertIn("is not UTF-8 text", out)

    def test_an_empty_ledger_with_only_its_header_is_a_ledger_with_no_rows(self):
        self.go([])
        self.assertEqual((self.record()["ledger_rows"], self.record()["variants"]), (0, 0))


# --- the runtime reader ---------------------------------------------------------------------------------------------

class RuntimeReaderTest(LedgerBase):
    """The C reader (runtime/gpu/shader_cache.c) over the two packs the step wrote."""
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls):
        cc = shct.find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.build = tempfile.TemporaryDirectory(prefix="oot2-ledger-c-", ignore_cleanup_errors=True)
        extra = [] if os.name == "nt" else ["-pthread"]
        for level in shct.LEVELS:
            exe = Path(cls.build.name) / f"shader_cache_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(ROOT / "runtime" / "core"),
                   *map(str, shct.SOURCES), "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "build"):
            cls.build.cleanup()

    def lines(self, *args) -> list[str]:
        outs = []
        for level in shct.LEVELS:
            done = subprocess.run([str(self.exes[level]), *map(str, args)], capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, f"{level} {args[0]}: {done.stdout}{done.stderr}")
            outs.append(done.stdout.splitlines())
        self.assertEqual(outs[0], outs[1])
        return outs[0]

    @staticmethod
    def expect(source: str, pack: cp.Pack, row: cp.Row) -> str:
        module, desc = pack.read(row)
        return (f"{source} {row.key} {cp.digest_text(row.answers)} {row.status} "
                f"{hashlib.sha256(module).hexdigest() if module else '-'} {hashlib.sha256(desc).hexdigest()}")

    def test_the_runtime_reader_takes_the_variant_before_the_base(self):
        tex, attr = digest_of(tex={"3": "3d"}), digest_of(attr={"0": "sint"})
        full_tex = digest_of(tex={"3": "3d", "7": "cube"})
        self.go([row(KEY["vs"], "vs", tex={"3": "3d", "7": "cube"}), row(KEY["vs"], "vs", attr={"0": "sint"}),
                 row(KEY["cs"], "cs", img={"1": IMG_3D})])
        base, variants = self.pack(), self.variants()
        queries = [(KEY["vs"], tex), (KEY["vs"], attr), (KEY["vs"], cp.ZERO_DIGEST),
                   (KEY["vs"], digest_of(fout={"0": "uint"})),          # an answer set no row asked for
                   (KEY["cs"], digest_of(img={"1": IMG_3D})), (KEY["ps"], tex),
                   ("0123456789abcdef.000", tex)]
        want = [self.expect("variants", variants, variants.find(KEY["vs"], tex)),
                self.expect("variants", variants, variants.find(KEY["vs"], attr)),
                self.expect("base", base, base.find(KEY["vs"])),
                self.expect("base", base, base.find(KEY["vs"])),
                self.expect("variants", variants, variants.find(KEY["cs"], digest_of(img={"1": IMG_3D}))),
                self.expect("base", base, base.find(KEY["ps"])),
                f"miss 0123456789abcdef.000 {cp.digest_text(tex)}"]
        q = shct.write_queries(self.root / "set.q", queries)
        self.assertEqual(self.lines("set", self.pack_path(), self.variants_path(), q), want)
        map_unknown = digest_of(fout={"0": "uint"})
        map_queries = shct.write_queries(self.root / "map.q",
                                         [(KEY["vs"], full_tex), (KEY["vs"], map_unknown)])
        mapped = self.expect("variants", variants, variants.find(KEY["vs"], tex)).split()
        unknown = self.expect("base", base, base.find(KEY["vs"])).split()
        mapped[2] = cp.digest_text(full_tex)
        unknown[0], unknown[2] = "fallback", cp.digest_text(map_unknown)
        self.assertEqual(self.lines("map", self.pack_path(), self.variants_path(), self.map_path(), map_queries),
                         [" ".join(mapped), " ".join(unknown)])
        # the module a variant gives is not the base module of its program
        self.assertNotEqual(want[0].split()[4], want[2].split()[4])
        # without the variants pack every query takes the base pack's row
        self.assertEqual(self.lines("set", self.pack_path(), "-", q)[:2], [want[2]] * 2)


# --- the real translator --------------------------------------------------------------------------------------------

class RealTranslatorTest(LedgerBase):
    """A vertex program that samples a texture, through oot2-xlate as the windows-build runner builds it."""

    @classmethod
    def setUpClass(cls):
        if xlatet.EXE is None:
            if os.environ.get("OOT2_REQUIRE_XLATE"):
                raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
            raise unittest.SkipTest(xlatet.SKIP_NO_EXE)

    def setUp(self):
        super().setUp()
        tex, ast = xlatet.UniLayoutTest.tex, xlatet.UniLayoutTest.ast
        self.sampling = trt.graphics(1, 11, tex(0, 3) + ast(0x70, 0, 3) + ast(0x80, 0, 3) + EXIT)
        archives = {"Effect/fx.bfsha": trt.bnsh({0: self.sampling}, {5: trt.compute(12)})}
        self.dump = self.write_dump(trt.container(archives))
        self.key = trt.job_key("vs", self.sampling)
        self.stub = xlatet.EXE

    def test_a_texture_the_instruction_types_is_never_asked_so_its_answers_add_nothing(self):
        """The real translator takes a TEX instruction's own type: its description lists the texture but records no
        sampler_type query, and a job's answer for the slot leaves the module and the description as they are (the
        reason the step reads the queries, not the declared textures)."""
        rows = [row(self.key, "vs", tex={"3": "3d"}),
                row(self.key, "vs", tex={"3": "3d", "5": "cube"}),
                row(self.key, "vs", tex={"5": "3d"}),
                row(self.key, "vs", tex={"3": "2d"}),
                row(self.key, "vs")]
        self.go(rows)
        pack = self.pack()
        base = json.loads(pack.read(pack.find(self.key))[1])
        self.assertEqual([(t["c2_slot"], t["type"]) for t in base["textures"]], [(3, "2d")])
        self.assertEqual({q["kind"] for q in base["queries"]}, {"register_texture"})
        reads = ledger.reads_of(base)
        self.assertEqual((reads.tex, reads.img_type, reads.img_fmt), ({}, {}, {}))
        rec = self.record()
        self.assertEqual((rec["ledger_rows"], rec["default_rows"], rec["candidates"], rec["variants"]), (5, 5, 0, 0))
        self.assertFalse(self.variants_path().exists())
        # the premise, on the translator itself: the answer changes neither the module nor the description
        jobs = translate.load_jobs(self.dest / translate.SHADERS / translate.SPLIT)
        job = ledger.corpus.variant_job(jobs[self.key], {"tex": {"3": "3d"}})
        folder = self.root / "direct"
        folder.mkdir()
        got = ledger.run_batch(check.translator_command(Path(xlatet.EXE)), [(job, folder)], self.root / "logs", 1,
                               1 << 32, 300.0)
        self.assertEqual(got[self.key].status, "ok", got[self.key])
        self.assertTrue(ledger.same_as_base(pack, self.key, got[self.key].module, got[self.key].description))
        self.assertEqual(got[self.key].module, self.base_module(self.key))


# --- absorb: the ledgers of other runs merged into the prepared folder's ---------------------------------------------

def write_ledger(path: Path, rows: list[dict], reorder: bool = False) -> Path:
    """A ledger file in the writer's shape; reorder gives each row's keys in reverse order (the same row, other text)."""
    lines = [json.dumps(ledger.HEADER, separators=(",", ":"))]
    for r in rows:
        r = dict(reversed(list(r.items()))) if reorder else r
        lines.append(json.dumps(r, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path


class AbsorbTest(unittest.TestCase):
    A = row(KEY["vs"], "vs", tex={"3": "3d"})
    B = row(KEY["cs"], "cs", img={"1": IMG_3D})
    C = row(KEY["vs"], "vs", tex={"4": "cube"})
    D = row(KEY["ps"], "ps", tex={"0": "2darray"})

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-absorb-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.prepared = self.tmp / "game" / ledger.LEDGER_FILE

    def run_file(self, name: str, rows: list[dict], **kw) -> Path:
        return write_ledger(self.tmp / name / ledger.LEDGER_FILE, rows, **kw)

    def lines(self) -> list[dict]:
        return [json.loads(t) for t in self.prepared.read_text(encoding="utf-8").splitlines()[1:]]

    def test_union_keeps_the_prepared_rows_first_then_each_run_in_order(self):
        write_ledger(self.prepared, [self.B])
        r1 = self.run_file("r1", [self.A, self.B])
        r2 = self.run_file("r2", [self.D, self.A, self.C])
        rep = ledger.absorb(self.prepared, [r1, r2])
        self.assertEqual(self.lines(), [self.B, self.A, self.D, self.C])
        self.assertEqual([(n, f) for _, n, f in rep.per_file], [(2, 1), (3, 2)])
        self.assertEqual((rep.new_rows, rep.total_rows, rep.written), (3, 4, True))
        self.assertEqual(rep.new_variants, 3)
        read_ledger_rows, _ = ledger.read_ledger(self.prepared)
        self.assertEqual(len(read_ledger_rows), 4)

    def test_a_same_program_with_other_answers_is_a_row_but_a_new_attachment_is_not_a_new_variant(self):
        other = dict(self.D, attach={"color": [38], "depth": None, "samples": 1})
        write_ledger(self.prepared, [self.D])
        rep = ledger.absorb(self.prepared, [self.run_file("r", [other])])
        self.assertEqual((rep.new_rows, rep.new_variants), (1, 0))

    def test_a_missing_prepared_ledger_is_created(self):
        rep = ledger.absorb(self.prepared, [self.run_file("r", [self.A, self.C])])
        self.assertEqual(self.lines(), [self.A, self.C])
        self.assertEqual((rep.new_rows, rep.backup), (2, None))
        self.assertFalse(self.prepared.with_name(self.prepared.name + ".bak").exists())

    def test_a_row_with_the_keys_in_another_order_counts_once(self):
        write_ledger(self.prepared, [self.A])
        rep = ledger.absorb(self.prepared, [self.run_file("r", [self.A], reorder=True)])
        self.assertEqual((rep.new_rows, rep.written), (0, False))
        self.assertEqual(self.lines(), [self.A])
        # and the same inside one run, and across two runs
        rep = ledger.absorb(self.prepared, [self.run_file("r1", [self.C, self.C], reorder=True),
                                            self.run_file("r2", [self.C])])
        self.assertEqual([f for _, _, f in rep.per_file], [1, 0])

    def test_the_prepared_ledger_among_the_inputs_and_repeated_inputs_are_harmless(self):
        write_ledger(self.prepared, [self.A])
        r = self.run_file("r", [self.B])
        rep = ledger.absorb(self.prepared, [self.prepared, r, r])
        self.assertEqual(self.lines(), [self.A, self.B])
        self.assertEqual([f for _, _, f in rep.per_file], [0, 1, 0])

    def test_a_bad_file_stops_everything_and_writes_nothing(self):
        write_ledger(self.prepared, [self.A])
        before = self.prepared.read_bytes()
        good = self.run_file("good", [self.B])
        broken = self.run_file("bad", [self.C])
        bad_digest = dict(self.D, digest="1" * 64)
        cases = {
            "answer outside the vocabulary": (json.dumps(dict(self.D, answers={"tex": {"0": "4d"}})), "line 3"),
            "digest": (json.dumps(bad_digest), "line 3"),
            "not json": ("{oops", "line 3"),
        }
        for name, (text, where) in cases.items():
            with self.subTest(name):
                broken.write_text(broken.read_text(encoding="utf-8") + text + "\n", encoding="utf-8")
                with self.assertRaises(PrepareError) as cm:
                    ledger.absorb(self.prepared, [good, broken])
                self.assertIn(str(broken), str(cm.exception))
                self.assertIn(where, str(cm.exception))
                self.assertEqual(self.prepared.read_bytes(), before)
                self.assertEqual(sorted(p.name for p in self.prepared.parent.iterdir()), [ledger.LEDGER_FILE])
                write_ledger(broken, [self.C])

    def test_a_crlf_a_blank_line_a_bad_header_and_a_missing_file_are_refused(self):
        write_ledger(self.prepared, [self.A])
        before = self.prepared.read_bytes()
        head = json.dumps(ledger.HEADER, separators=(",", ":"))
        a = json.dumps(self.B, separators=(",", ":"))
        texts = {"crlf": (f"{head}\r\n{a}\r\n", "line 1"), "blank": (f"{head}\n{a}\n\n{a}\n", "line 3"),
                 "header": (f'{{"oot2_ledger":2}}\n{a}\n', "line 1")}
        for name, (text, where) in texts.items():
            with self.subTest(name):
                f = self.tmp / name / "ledger.jsonl"
                f.parent.mkdir()
                f.write_bytes(text.encode("utf-8"))
                with self.assertRaises(PrepareError) as cm:
                    ledger.absorb(self.prepared, [f])
                self.assertIn(where, str(cm.exception))
                self.assertEqual(self.prepared.read_bytes(), before)
        with self.assertRaises(PrepareError):
            ledger.absorb(self.prepared, [self.tmp / "nowhere" / "ledger.jsonl"])
        # a damaged prepared ledger is refused too
        self.prepared.write_text("not a ledger\n", encoding="utf-8")
        with self.assertRaises(PrepareError):
            ledger.absorb(self.prepared, [self.run_file("r", [self.B])])

    def test_a_write_that_fails_leaves_the_prepared_ledger_and_no_temp_file(self):
        write_ledger(self.prepared, [self.A])
        before = self.prepared.read_bytes()
        with mock.patch("os.replace", side_effect=OSError(13, "denied")):
            with self.assertRaises(PrepareError):
                ledger.absorb(self.prepared, [self.run_file("r", [self.B])])
        self.assertEqual(self.prepared.read_bytes(), before)
        self.assertFalse(self.prepared.with_name(self.prepared.name + ".new").exists())

    def test_the_new_file_goes_through_a_temp_name_and_a_replace(self):
        write_ledger(self.prepared, [self.A])
        with mock.patch("os.replace", wraps=os.replace) as rep:
            ledger.absorb(self.prepared, [self.run_file("r", [self.B])])
        rep.assert_called_once_with(self.prepared.with_name(self.prepared.name + ".new"), self.prepared)

    def test_the_backup_is_made_once_and_never_overwritten(self):
        write_ledger(self.prepared, [self.A])
        first = self.prepared.read_bytes()
        bak = self.prepared.with_name(self.prepared.name + ".bak")
        rep = ledger.absorb(self.prepared, [self.run_file("r1", [self.B])])
        self.assertEqual((rep.backup, bak.read_bytes()), (bak, first))
        rep = ledger.absorb(self.prepared, [self.run_file("r2", [self.C])])
        self.assertEqual((rep.backup, bak.read_bytes()), (None, first))
        self.assertEqual(self.lines(), [self.A, self.B, self.C])

    def test_a_dry_run_counts_and_writes_nothing(self):
        write_ledger(self.prepared, [self.A])
        before = self.prepared.read_bytes()
        rep = ledger.absorb(self.prepared, [self.run_file("r", [self.B, self.C])], dry_run=True)
        self.assertEqual((rep.new_rows, rep.total_rows, rep.written), (2, 3, False))
        self.assertEqual(self.prepared.read_bytes(), before)
        self.assertEqual([p.name for p in self.prepared.parent.iterdir()], [ledger.LEDGER_FILE])
        gone = self.tmp / "other" / ledger.LEDGER_FILE
        ledger.absorb(gone, [self.run_file("r", [self.B])], dry_run=True)
        self.assertFalse(gone.parent.exists())


class AbsorbKeyTest(LedgerBase):
    """The ledger step's key is the ledger file's SHA-256, so absorb moves it exactly when it adds a row."""

    def test_the_step_runs_again_after_a_new_row_and_not_after_rows_already_held(self):
        path = self.ledger_path()
        write_ledger(path, [row(KEY["vs"], "vs", tex={"3": "3d"})])     # LF bytes, as the writer makes them
        self.go()
        elsewhere = self.dest.parent / "elsewhere" / ledger.LEDGER_FILE
        write_ledger(elsewhere, [row(KEY["vs"], "vs", tex={"3": "3d"})], reorder=True)
        sha = ledger.ledger_digest(path)
        ledger.absorb(path, [elsewhere])
        self.assertEqual(ledger.ledger_digest(path), sha)
        self.assertIn("ledger: kept (done, inputs unchanged)", self.go())

        write_ledger(elsewhere, [row(KEY["cs"], "cs", img={"1": IMG_3D})])
        rep = ledger.absorb(path, [elsewhere])
        self.assertEqual(rep.new_variants, 1)
        self.assertNotEqual(ledger.ledger_digest(path), sha)
        text = self.go()
        self.assertIn("ledger: started (its inputs changed)", text)
        self.assertIn("translate: kept", text)
        self.assertEqual(len(self.variants().rows), 2)


if __name__ == "__main__":
    unittest.main()
