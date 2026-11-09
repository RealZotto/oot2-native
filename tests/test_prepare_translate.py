# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the translate step of `oot2 prepare` (tools/prepare/translate.py) and the shader tools its check step
requires (tools/prepare/check.py).

Every input is synthetic and built inside the test: a container from the .nup writer of tests/test_prepare_extract.py,
whose RomFS holds shader archives assembled with the builders of tests/test_shader_split.py around programs of a few
Ampere instruction words, as tests/test_xlate_host.py assembles them. No game file is read. The real translator
(tools/build_xlate.py; the Windows CI job builds it and sets OOT2_REQUIRE_XLATE) translates those programs, and without
it those tests skip. The other tests run a stand-in translator written here, a Python script that speaks the pool's
line protocol and writes modules and descriptions the shader pack accepts, so they run on every runner. A zstd archive
and the check's shader tools need Python 3.14 (compression.zstd).

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import _thread
import collections
import csv
import hashlib
import io
import json
import os
import struct
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

import cache_pack  # noqa: E402
import split  # noqa: E402
import test_prepare_extract as nupt  # noqa: E402  (NupWriter, FakeCheck)
import test_shader_split as splitt  # noqa: E402  (make_ctl, make_bnsh, in_container)
import test_xlate_host as xlatet  # noqa: E402  (EXE, EXIT)
from prepare import check, extract, translate  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, run_steps  # noqa: E402

try:
    from compression import zstd  # type: ignore[import-not-found]
except ImportError:
    zstd = None

NEEDS_ZSTD = unittest.skipUnless(zstd is not None, "compression.zstd needs Python 3.14")
EXIT = xlatet.EXIT
UNKNOWN = struct.pack("<QQ", 0x123, 0)        # an opcode the decoder lacks (tests/test_xlate_host.py)
HEADER = 0x80                                  # a graphics program's header
COMPUTE = ((0xBA, "<H", 16), (0x58, "<I", 8), (0x5C, "<I", 8), (0x60, "<I", 1), (0x64, "<I", 0))


# --- programs and containers ----------------------------------------------------------------------------------------
# The seed goes to control bytes 0x10..0x13, inside the 0x48 bytes the key hashes and outside what a job carries, so
# programs with the same code are distinct jobs.

def graphics(kind: int, seed: int, body: bytes = EXIT) -> tuple[bytes, bytes]:
    """A vertex (kind 1) or pixel (kind 5) program: its header, then the instructions."""
    header = bytearray(HEADER)
    struct.pack_into("<I", header, 0, (kind << 10) | 3)
    code = bytes(header) + body
    return splitt.make_ctl(inst=len(code), code_len=len(code), c1_at=len(code), patch=((0x10, "<I", seed),)), code


def compute(seed: int, body: bytes = EXIT) -> tuple[bytes, bytes]:
    """A compute program of 8 by 8 threads."""
    return splitt.make_ctl(stage=5, inst=len(body), code_len=len(body), c1_at=len(body),
                           patch=((0x10, "<I", seed),) + COMPUTE), body


def marked(seed: int, marker: bytes) -> tuple[bytes, bytes]:
    """A compute program whose first instruction slot holds a word the stand-in translator acts on."""
    return compute(seed, marker.ljust(16, b"\0") + EXIT)


def job_key(stage: str, program: tuple[bytes, bytes]) -> str:
    ctl, code = program
    return f"{split.ctl_key(ctl, code)}.{split.build_job(stage, ctl, code).variant_hash()}"


def bnsh(*variations: dict) -> bytes:
    return splitt.in_container(splitt.make_bnsh(list(variations)))


def container(archives: dict[str, bytes]) -> bytes:
    """A game's tree: the shader archives beside files that hold no program."""
    files = {"System/RegionLangMask.txt": b"1\n2\n111\n", "Data/x.bin": bytes(300), **archives}
    return nupt.NupWriter().package(0, 1, exefs={"main": b"m"}, romfs=files).build()


# --- the stand-in translator ----------------------------------------------------------------------------------------

STUB = textwrap.dedent('''\
    """A stand-in translator: the pool's line protocol, a module that is SPIR-V by its first words, and a description
    the shader pack accepts. What a program's code holds decides what happens: REFUSE fails it, CRASHONCE ends this
    process the first time it is asked, CRASH every time, HANG never answers."""
    import base64, hashlib, json, os, struct, sys, time
    IDENTITY = os.environ.get("STUB_IDENTITY") or hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    if sys.argv[1:] == ["--version"]:
        form = os.environ.get("STUB_OPTIMIZER", "sound")     # "none": a build older than the sound form names none
        print(f"oot2-xlate {IDENTITY} proto={os.environ.get('STUB_PROTO', '2')} spirv=1.3 target=vulkan1.3"
              + ("" if form == "none" else f" optimizer={form}"))
        sys.exit(0)
    for line in sys.stdin:
        request = json.loads(line)
        job = request["job"]
        key = job["key"]
        with open(job["code"], "rb") as f:
            f.seek(job.get("code_offset", 0))
            code = f.read(job.get("code_length", 1 << 20))
        once = os.path.join(os.path.dirname(os.path.abspath(__file__)), key + ".crashed")
        if b"CRASHONCE" in code and not os.path.exists(once):
            open(once, "w").close()
            os._exit(3)
        if b"CRASHONCE" not in code and b"CRASH" in code:
            os._exit(3)
        if b"HANG" in code:
            time.sleep(3600)
        if b"REFUSE" in code:
            print(f"fail {key} 5 unsupported_opcode: Ampere instruction 123 at 0x0 is not implemented", flush=True)
            continue
        time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
        if os.environ.get("STUB_LOG"):
            with open(os.path.join(os.environ["STUB_LOG"], f"{os.getpid()}.log"), "a") as fh:
                fh.write(key + "\\n")
        module = struct.pack("<5I", 0x07230203, 0x00010300, 0, 1, 0) + hashlib.sha256(code).digest()
        res = {"schema": 2, "key": key, "stage": job["stage"], "translator_id": IDENTITY, "spv_bytes": len(module),
               "spv_sha256": hashlib.sha256(module).hexdigest(), "answers": {}}
        m64 = base64.b64encode(module).decode()
        d64 = base64.b64encode(json.dumps(res, sort_keys=True).encode()).decode()
        print(f"ok {key} 100 {m64} {d64}", flush=True)
''')


def logged(folder: Path) -> list[str]:
    """The keys the stand-in translated, from every child's log."""
    return [line for p in sorted(folder.glob("*.log")) for line in p.read_text().split()]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-translate-", ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.dest = self.root / "out"
        self.stub = self.root / "xlate" / "stub_xlate.py"
        self.stub.parent.mkdir()
        self.stub.write_text(STUB, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write_dump(self, data: bytes) -> Path:
        path = self.root / "game.nup"
        path.write_bytes(data)
        return path

    def run_steps(self, dump: Path, xlate: Path, dest: Path | None = None, verify: bool = False) -> tuple[str, str]:
        """The check (its duty to the later steps only), extract and translate through the runner."""
        out = io.StringIO()
        dest = dest or self.dest
        opts = types.SimpleNamespace(xlate=str(xlate), translate_jobs=2, cc=None, plain_modules=None,
                                     zbic_archive=None)
        ctx = Context(dest, opts, Reporter(out, out), state=statefile.load(dest), verify=verify)
        result = run_steps([nupt.FakeCheck(dump), extract.Extract(), translate.Translate()], ctx)
        return result, out.getvalue()

    def steps(self, dest: Path | None = None) -> dict:
        return statefile.load(dest or self.dest)["steps"]

    def pack_path(self, dest: Path | None = None) -> Path:
        return (dest or self.dest) / translate.SHADERS / cache_pack.BASE_FILE

    def pack(self, dest: Path | None = None) -> cache_pack.Pack:
        return cache_pack.open_pack(self.pack_path(dest), role=cache_pack.BASE)

    def split_keys(self, dest: Path | None = None) -> set[str]:
        return set(translate.load_jobs((dest or self.dest) / translate.SHADERS / translate.SPLIT))


# --- the real translator --------------------------------------------------------------------------------------------

class RealTranslatorTest(Base):
    """Synthetic sm_86 programs through oot2-xlate, as the windows-build runner builds it."""

    @classmethod
    def setUpClass(cls):
        if xlatet.EXE is None:
            if os.environ.get("OOT2_REQUIRE_XLATE"):
                raise RuntimeError("OOT2_REQUIRE_XLATE is set but oot2-xlate is not built under build/xlate")
            raise unittest.SkipTest(xlatet.SKIP_NO_EXE)
        if zstd is None:
            raise unittest.SkipTest("compression.zstd needs Python 3.14")

    def archives(self) -> dict[str, bytes]:
        """Six jobs: a zstd-compressed BFSHA of the Shader tree and a plain effect archive, vertex, pixel, compute."""
        shader = bnsh({0: graphics(1, 1), 4: graphics(5, 2)}, {5: compute(3)})
        effect = bnsh({5: compute(4)}, {0: graphics(1, 5), 4: graphics(5, 6)})
        return {"Shader/Test.Product.110.product.Nin_Ounce_NVN2.bfsha.zs": zstd.compress(shader),
                "Effect/fx.bfsha": effect}

    def test_both_romfs_forms_read_the_same_programs_and_every_job_is_packed(self):
        dump = self.write_dump(container(self.archives()))
        result, text = self.run_steps(dump, xlatet.EXE)
        self.assertEqual(result, "ok", text)
        self.assertEqual(self.steps()["extract"]["romfs_form"], extract.IN_PLACE)
        rec = self.steps()["translate"]
        keys = self.split_keys()
        self.assertEqual(len(keys), 6)
        self.assertEqual((rec["jobs"], rec["translated"], rec["failed"], rec["kept_jobs"]), (6, 6, {}, 0))
        pack = self.pack()
        self.assertEqual({r.key for r in pack.rows}, keys)
        self.assertFalse([r.key for r in pack.rows if r.failed])
        self.assertEqual(pack.translator, hashlib.sha256(xlatet.EXE.read_bytes()).hexdigest())
        self.assertEqual(cache_pack.verify_pack(self.pack_path(), role=cache_pack.BASE), [])
        stages = collections.Counter(json.loads(pack.read(r)[1])["stage"] for r in pack.rows)
        self.assertEqual(stages, {"vs": 2, "ps": 2, "cs": 2})
        # the jobs name no optimizer option, so every module is the translator's default form: the sound one
        self.assertEqual({json.loads(pack.read(r)[1])["options"]["debug"] for r in pack.rows}, {False})
        self.assertEqual(rec["bytes"], pack.size + sum(rec["split_files"].values()))
        self.assertEqual(rec["pack_bytes"], pack.size)
        self.assertEqual(rec["spill_bytes"], rec["module_bytes"] + rec["description_bytes"])
        scratch = self.dest / check.TEMP / translate.SCRATCH
        for name in (translate.SPILL, translate.RESULTS):
            self.assertFalse((scratch / name).exists(), name)
        modules = {r.key: pack.read(r)[0] for r in pack.rows}
        self.assertIn("translate: kept", self.run_steps(dump, xlatet.EXE)[1])

        copied = self.root / "copied"
        with mock.patch.object(extract, "romfs_form", lambda tree: (extract.COPY, "a test asks for it")):
            result, text = self.run_steps(dump, xlatet.EXE, dest=copied)
        self.assertEqual(result, "ok", text)
        self.assertEqual(self.steps(copied)["extract"]["romfs_form"], extract.COPY)
        other = self.pack(copied)
        self.assertEqual({r.key: other.read(r)[0] for r in other.rows}, modules)

    def test_a_program_the_translator_refuses_is_packed_as_a_named_failure(self):
        refused = compute(7, UNKNOWN + EXIT)
        archives = self.archives()
        archives["Lib/refused.bfsha"] = bnsh({5: refused})
        dump = self.write_dump(container(archives))
        result, text = self.run_steps(dump, xlatet.EXE)
        self.assertEqual(result, "ok", text)
        key = job_key("cs", refused)
        rec = self.steps()["translate"]
        self.assertEqual((rec["jobs"], rec["translated"], rec["failed"], rec["failed_keys"]), (7, 6, {"fail": 1}, [key]))
        pack = self.pack()
        self.assertEqual({r.key for r in pack.rows}, self.split_keys())
        row = pack.find(key)
        self.assertTrue(row.failed)
        reason = pack.read(row)[1].decode("utf-8")
        self.assertTrue(reason.startswith("fail: unsupported_opcode:"), reason)
        self.assertIn("warning: 1 of 7 shader programs did not translate (fail 1); each is packed as a failure record",
                      text)
        self.assertIn(f"{key}: fail: unsupported_opcode:", text)

    def test_a_translator_that_translates_nothing_fails_the_step(self):
        archives = {"Lib/refused.bfsha": bnsh({5: compute(7, UNKNOWN + EXIT)}, {5: compute(8, UNKNOWN + EXIT)})}
        result, text = self.run_steps(self.write_dump(container(archives)), xlatet.EXE)
        self.assertEqual(result, "failed", text)
        self.assertIn("translated none of the 2 shader programs, so the translator is what is broken", text)
        self.assertFalse(self.pack_path().exists())
        self.assertNotIn("translate", self.steps())

    def test_the_check_accepts_the_built_translator(self):
        ctx = Context(self.dest, types.SimpleNamespace(xlate=str(xlatet.EXE)), Reporter(io.StringIO(), io.StringIO()))
        text = check.shader_tools(ctx)
        self.assertIn(f"translator {xlatet.EXE.resolve()} (protocol 2, identity "
                      f"{hashlib.sha256(xlatet.EXE.read_bytes()).hexdigest()[:16]})", text)


# --- the stand-in translator ----------------------------------------------------------------------------------------

class StandInTest(Base):
    def test_a_new_translator_runs_this_step_again_and_nothing_upstream(self):
        dump = self.write_dump(container({"Effect/fx.bfsha": bnsh(*({5: compute(s)} for s in range(1, 5)))}))
        result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        first = self.pack().translator
        self.assertEqual(first, hashlib.sha256(self.stub.read_bytes()).hexdigest())
        result, text = self.run_steps(dump, self.stub)
        self.assertIn("extract: kept", text)
        self.assertIn("translate: kept", text)
        with open(self.stub, "a", encoding="utf-8") as fh:
            fh.write("# another build of the translator\n")
        result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        self.assertIn("extract: kept", text)
        self.assertIn("translate: started (its inputs changed)", text)
        self.assertIn("translate: split kept", text)            # the same RomFS, the same split.py
        self.assertIn("translate: deleted the scratch of an earlier run with other inputs", text)
        second = self.pack().translator
        self.assertEqual(second, hashlib.sha256(self.stub.read_bytes()).hexdigest())
        self.assertNotEqual(second, first)
        self.assertEqual(self.steps()["translate"]["translator"]["sha256"], second)
        self.pack_path().unlink()
        result, text = self.run_steps(dump, self.stub)
        self.assertIn("extract: kept", text)
        self.assertIn("translate: started (its output is missing)", text)
        self.assertEqual(result, "ok", text)

    def test_a_pack_made_with_another_identity_is_not_current_and_the_step_rebuilds_it(self):
        """A prepared folder from before the sound form was the default: its record and its pack hold the old
        executable's identity. The step is not kept, and its own verify refuses a pack that is not the recorded one."""
        dump = self.write_dump(container({"Effect/fx.bfsha": bnsh(*({5: compute(s)} for s in range(1, 4)))}))
        result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        record = self.steps()["translate"]
        ctx = Context(self.dest, types.SimpleNamespace(xlate=str(self.stub)), Reporter(io.StringIO(), io.StringIO()))
        self.assertTrue(translate.Translate().verify(ctx, record))
        old = {**record, "translator": {**record["translator"], "sha256": hashlib.sha256(b"the old executable").hexdigest()}}
        self.assertFalse(translate.Translate().verify(ctx, old))
        # the folder as an older prepare left it: the translate step's recorded key is the old one
        state = statefile.load(self.dest)
        state["steps"]["translate"]["key"] = hashlib.sha256(b"the old translate key").hexdigest()
        statefile.save(self.dest, state)
        result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        self.assertIn("translate: started (its inputs changed)", text)
        self.assertEqual(self.pack().translator, hashlib.sha256(self.stub.read_bytes()).hexdigest())

    def test_an_interrupted_run_keeps_what_it_translated(self):
        programs = [compute(s) for s in range(1, 31)]
        dump = self.write_dump(container({"Effect/fx.bfsha": bnsh(*({5: p} for p in programs))}))
        log = self.root / "log"
        log.mkdir()
        answered = []
        add = translate.Progress.add

        def add_then_interrupt(progress, r):
            add(progress, r)
            answered.append(r.key)
            if len(answered) == 5:
                _thread.interrupt_main()         # what Ctrl+C does to the main thread

        with mock.patch.dict(os.environ, {"STUB_SLEEP": "0.1", "STUB_LOG": str(log)}), \
                mock.patch.object(translate.Progress, "add", add_then_interrupt):
            result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "cancelled", text)
        self.assertNotIn("translate", self.steps())
        first = logged(log)
        self.assertGreaterEqual(len(first), 5)
        self.assertLess(len(first), 30)
        self.assertEqual(sorted(first), sorted(answered))     # every answer was recorded
        scratch = self.dest / check.TEMP / translate.SCRATCH
        self.assertFalse((scratch / "out").exists())     # no loose file: every answer went into the spill
        # The spill cut inside its last entry: that job alone is translated again.
        with open(scratch / translate.RESULTS, encoding="utf-8", newline="") as fh:
            last = max((row for row in csv.DictReader(fh, delimiter="\t") if row["status"] == "ok"),
                       key=lambda row: int(row["spill_offset"]))
        spill = scratch / translate.SPILL
        end = int(last["spill_offset"]) + int(last["spv_bytes"]) + int(last["desc_bytes"])
        self.assertEqual(spill.stat().st_size, end)
        with open(spill, "r+b") as fh:
            fh.truncate(end - 4)
        with mock.patch.dict(os.environ, {"STUB_LOG": str(log)}):
            result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        kept = len(first) - 1
        self.assertIn(f"translate: 30 jobs, {kept} kept from an interrupted run, {30 - kept} to translate", text)
        counts = collections.Counter(logged(log))
        self.assertEqual(set(counts), {job_key("cs", p) for p in programs})
        self.assertEqual(sorted(k for k, n in counts.items() if n > 1), [last["key"]])
        self.assertEqual(counts[last["key"]], 2)
        self.assertEqual(self.steps()["translate"]["kept_jobs"], kept)
        pack = self.pack()
        self.assertEqual(len(pack.rows), 30)
        self.assertFalse([r.key for r in pack.rows if r.failed])

    def test_a_job_without_an_answer_is_tried_again_alone_then_packed_as_a_failure(self):
        once, always, hang, refused = (marked(1, b"CRASHONCE"), marked(2, b"CRASH"), marked(3, b"HANG"),
                                       marked(4, b"REFUSE"))
        programs = [once, always, hang, refused, compute(5)]
        dump = self.write_dump(container({"Effect/fx.bfsha": bnsh(*({5: p} for p in programs))}))
        with mock.patch.object(translate, "TIMEOUT", 3.0), mock.patch.object(translate, "RETRY_TIMEOUT", 3.0):
            result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        self.assertIn("translate: 3 job(s) got no answer (crash 2, timeout 1); each is tried again alone", text)
        rec = self.steps()["translate"]
        self.assertEqual((rec["jobs"], rec["translated"], rec["retried"]), (5, 2, 3))
        self.assertEqual(rec["failed"], {"crash": 1, "fail": 1, "timeout": 1})
        pack = self.pack()
        self.assertFalse(pack.find(job_key("cs", once)).failed)
        reasons = {name: pack.read(pack.find(job_key("cs", p)))[1].decode("utf-8")
                   for name, p in (("always", always), ("hang", hang), ("refused", refused))}
        self.assertTrue(reasons["always"].startswith("crash: child "), reasons)
        self.assertTrue(reasons["hang"].startswith("timeout: child "), reasons)
        self.assertTrue(reasons["refused"].startswith("fail: unsupported_opcode: "), reasons)
        self.assertIn("warning: 3 of 5 shader programs did not translate (crash 1, fail 1, timeout 1)", text)

    def test_a_damaged_shader_file_stops_the_step(self):
        dump = self.write_dump(container({"Effect/fx.bfsha": bnsh({5: compute(1)}, {5: compute(2)})}))
        result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "ok", text)
        _, rows = extract.read_index(self.dest / extract.ROMFS / extract.INDEX)
        offset = next(off for path, off, size, sha in rows if path == "Effect/fx.bfsha")
        data = bytearray(dump.read_bytes())
        data[offset + 10] ^= 1                  # inside the bytes before the BNSH: the split alone would not mind
        dump.write_bytes(bytes(data))
        result, text = self.run_steps(dump, self.stub)
        self.assertEqual(result, "failed", text)
        self.assertIn("the RomFS file Effect/fx.bfsha, read from ", text)
        self.assertIn("does not match the SHA-256 of the RomFS index", text)

    def test_verify_finds_a_damaged_pack_and_translates_again(self):
        dump = self.write_dump(container({"Effect/fx.bfsha": bnsh({5: compute(1)}, {5: compute(2)})}))
        self.assertEqual(self.run_steps(dump, self.stub)[0], "ok")
        path = self.pack_path()
        data = bytearray(path.read_bytes())
        data[-1] ^= 1                           # the last entry's description
        path.write_bytes(bytes(data))
        self.assertIn("translate: kept", self.run_steps(dump, self.stub)[1])     # same size: kept without --verify
        result, text = self.run_steps(dump, self.stub, verify=True)
        self.assertEqual(result, "ok", text)
        self.assertIn("translate: started (its output failed the full check)", text)
        self.assertEqual(cache_pack.verify_pack(path, role=cache_pack.BASE), [])


# --- the check step's shader tools ----------------------------------------------------------------------------------

class CheckShaderToolsTest(Base):
    def ctx(self, **options) -> Context:
        return Context(self.dest, types.SimpleNamespace(**{"xlate": None, **options}),
                       Reporter(io.StringIO(), io.StringIO()))

    def test_a_python_without_zstd_is_refused(self):
        with mock.patch.dict(sys.modules, {"compression": None, "compression.zstd": None}):
            with self.assertRaisesRegex(PrepareError, r"has no compression\.zstd, which the translate step reads the "
                                                      r"game's shader archives with .*Python 3\.14 or later"):
                check.shader_tools(self.ctx(xlate=self.stub))

    def test_a_missing_translator_is_refused(self):
        with mock.patch.dict(os.environ, {"OOT2_XLATE": "no-such-translator-here"}):
            with self.assertRaisesRegex(PrepareError, r"no shader translator named 'no-such-translator-here' was "
                                                      r"found\. Give it with --xlate or the OOT2_XLATE variable"):
                check.translator(self.ctx())
            with self.assertRaisesRegex(PrepareError, "no shader translator named"):
                translate.Translate().key(self.ctx())
        with self.assertRaisesRegex(PrepareError, r"no shader translator named '.*missing\.exe' was found"):
            check.translator(self.ctx(xlate=self.root / "missing.exe"))

    @NEEDS_ZSTD
    def test_the_translator_must_give_its_own_identity_and_the_pools_protocol(self):
        identity = hashlib.sha256(self.stub.read_bytes()).hexdigest()
        self.assertIn(f"translator {self.stub.resolve()} (protocol 2, identity {identity[:16]})",
                      check.shader_tools(self.ctx(xlate=self.stub)))
        with mock.patch.dict(os.environ, {"STUB_IDENTITY": "0" * 64}):
            with self.assertRaisesRegex(PrepareError, f"gives the identity {'0' * 64}, but the SHA-256 of its file is "
                                                      f"{identity}"):
                check.shader_tools(self.ctx(xlate=self.stub))
        with mock.patch.dict(os.environ, {"STUB_PROTO": "1"}):
            with self.assertRaisesRegex(PrepareError, "speaks protocol 1, and this prepare speaks protocol 2"):
                check.shader_tools(self.ctx(xlate=self.stub))
        broken = self.root / "broken.py"
        broken.write_text("import sys\nsys.exit(4)\n", encoding="utf-8")
        with self.assertRaisesRegex(PrepareError, r"did not answer --version \(exit code 4\)"):
            check.shader_tools(self.ctx(xlate=broken))

    @NEEDS_ZSTD
    def test_a_translator_whose_default_is_the_old_optimizer_form_is_refused(self):
        """Prepare packs the sound form only: a translator that says the old one, or says nothing because it was built
        before the sound form was the default, would put modules of the other form beside the new ones."""
        for form, said in (("debug", "the optimizer form debug by default"),
                           ("none", "the optimizer form of a build older than the sound form")):
            with self.subTest(form), mock.patch.dict(os.environ, {"STUB_OPTIMIZER": form}):
                with self.assertRaisesRegex(PrepareError, f"translates with {said}.*packs the sound form only"):
                    check.shader_tools(self.ctx(xlate=self.stub))
        self.assertIn("protocol 2", check.shader_tools(self.ctx(xlate=self.stub)))      # the green counterpart

    def test_the_keys_of_the_shader_steps_name_the_form_and_the_identity(self):
        """A folder prepared before the sound form was the default holds the old translator's identity and no form:
        its translate key, the ledger's and so every key chained to them are not the new ones."""
        ctx = self.ctx(xlate=self.stub)
        same = translate.Translate().key(ctx)
        self.assertEqual(translate.Translate().key(ctx), same)
        with mock.patch.object(check, "XLATE_OPTIMIZER", "debug"):
            self.assertNotEqual(translate.Translate().key(ctx), same)
        before = translate.Translate().key(self.ctx(xlate=self.stub))
        with open(self.stub, "a", encoding="utf-8") as fh:
            fh.write("# the old executable's successor\n")
        self.assertNotEqual(translate.Translate().key(self.ctx(xlate=self.stub)), before)

    def test_every_step_that_reads_the_pack_chains_to_the_identity_and_the_form(self):
        """The ledger (the variants pack) and the pipeline cache are built from the base pack, so their keys, as the
        runner chains them (Step.needs), change with the translator identity and with the form: a folder made with the
        old one rebuilds them, it never mixes them with the new base pack."""
        from prepare import ledger, pipelines
        steps = (translate.Translate(), ledger.Ledger(), pipelines.Pipelines())

        def chain() -> dict[str, str]:
            ctx = self.ctx(xlate=self.stub)
            ctx.keys["extract"] = "the extract step's key, which is not what changes here"
            for step in steps:
                ctx.decide(step)
            return {step.id: ctx.keys[step.id] for step in steps}

        same = chain()
        self.assertEqual(chain(), same)
        self.assertEqual(set(same), {"translate", "ledger", "pipelines"})
        with mock.patch.object(check, "XLATE_OPTIMIZER", "debug"):
            other_form = chain()
        with open(self.stub, "a", encoding="utf-8") as fh:
            fh.write("# another executable, so another identity\n")
        other_identity = chain()
        for other in (other_form, other_identity):
            for step_id in same:
                self.assertNotEqual(other[step_id], same[step_id], step_id)

    def test_the_translator_identity_is_hashed_again_only_when_the_file_changes(self):
        ctx = self.ctx(xlate=self.stub)
        first = check.translator_identity(ctx)
        with mock.patch.object(check, "file_sha256", side_effect=AssertionError("hashed again")):
            self.assertEqual(check.translator_identity(ctx), first)
        with open(self.stub, "a", encoding="utf-8") as fh:
            fh.write("# changed\n")
        self.assertEqual(check.translator_identity(ctx), hashlib.sha256(self.stub.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
