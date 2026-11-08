# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the shader cache pack (architecture D21): tools/shaders/cache_pack.py writes and reads it,
runtime/gpu/shader_cache.c reads it.

Every pack is built here from synthetic modules (the SPIR-V magic and header words, then seeded random words) and
synthetic descriptions shaped like the translator's res.json; nothing comes from a game. The C reader is driven
through tests/runtime/shader_cache_test.c, built with runtime/gpu/shader_cache.c, runtime/core/plat.c and
runtime/core/sha256.c at -O1 and -O2:

- cross-read: Python writes a base and a variants pack and lists them; C finds every listed row by key and digest
  and returns the same status, module and description, and misses what is not there;
- isolation: C opens the pack and finds an entry while every other entry's pages are inaccessible, so neither the
  open nor the lookup reads another entry (a control run shows the protection is live);
- variants first: through the pair of packs, a digest the variants pack holds comes from it, anything else from the
  base pack's default row;
- the full-ledger map: a raw ledger digest maps to a projected variant or base, while an unknown digest falls back;
- refusals: a pack of another description schema, an index cut short, an entry that ends past the end of the file,
  and a damaged index, rows out of order, the wrong role, another translator, a short file and another magic, each
  under the name the Python reader gives the same file.

Needs clang for x86-64 (OOT2_CC, else clang on PATH) for the C half. Without one those tests skip, unless
OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
sys.path.insert(0, str(ROOT / "tools"))

import cache_pack as cp  # noqa: E402
from prepare import ledger as prepare_ledger  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "shader_cache_test.c"
SOURCES = (TEST_SOURCE, ROOT / "runtime" / "gpu" / "shader_cache.c", ROOT / "runtime" / "core" / "plat.c",
           ROOT / "runtime" / "core" / "sha256.c")
HEADER_FILE = ROOT / "runtime" / "include" / "oot2" / "shader_cache.h"
RESULT_WRITER = ROOT / "xlate" / "host" / "ResultWriter.cs"
LEVELS = ("-O1", "-O2")
PAGE = 4096
SEED = 43
TRANSLATOR = hashlib.sha256(b"synthetic translator").hexdigest()
OTHER_TRANSLATOR = hashlib.sha256(b"another synthetic translator").hexdigest()
TEX_3D = {"tex": {"*": "3d"}}
ATTR_SINT = {"attr": {"0": "sint"}}


# --- synthetic entries --------------------------------------------------------

def module(rng: random.Random, words: int) -> bytes:
    """A SPIR-V header (magic, version 1.3, generator 0, a bound, schema 0) and random words: no real program."""
    body = [rng.getrandbits(32) for _ in range(max(0, words - 5))]
    return struct.pack(f"<5I{len(body)}I", cp.SPIRV_MAGIC, 0x00010300, 0, rng.randint(1, 4096), 0, *body)


def description(key: str, mod: bytes, *, translator: str = TRANSLATOR, schema: int = cp.DESC_SCHEMA,
                answers: dict | None = None, pad: int = 0) -> bytes:
    ans = {"tex": {}, "img": {}, "attr": {}, "fout": {}, "early_fragment_tests": False, "heap_images": 64,
           "heap_samplers": 16}
    ans.update(answers or {})
    d = {"schema": schema, "key": key, "stage": "ps", "exec_model": "Fragment", "entry": "main",
         "translator_id": translator, "answers": ans, "spv_bytes": len(mod),
         "spv_sha256": hashlib.sha256(mod).hexdigest(), "notes": ["n" * pad]}
    return json.dumps(d, indent=2).encode("utf-8")


def new_key(rng: random.Random, key16: str | None = None) -> str:
    return f"{key16 or rng.randbytes(8).hex()}.{rng.randbytes(4).hex()}"


def translated(rng: random.Random, key: str, words: int, *, answers: dict | None = None,
               translator: str = TRANSLATOR, schema: int = cp.DESC_SCHEMA) -> cp.Entry:
    mod = module(rng, words)
    return cp.Entry(key, mod, description(key, mod, translator=translator, schema=schema, answers=answers,
                                          pad=rng.randrange(200)),
                    answers=cp.answers_digest(answers or {}))


def base_entries(rng: random.Random, count: int = 40) -> list[cp.Entry]:
    """Translated entries of several sizes, keys that share a key16, keys of other lengths whose NUL padding decides
    their order, and failure records."""
    out = []
    shared = rng.randbytes(8).hex()
    for i in range(count):
        out.append(translated(rng, new_key(rng, shared if i % 5 == 0 else None), rng.choice((5, 6, 7, 64, 300, 2000))))
    for key in ("k1", "k1.0", "k10", "K", "a-b_c"):
        out.append(translated(rng, key, rng.randint(5, 40)))
    out.append(cp.Entry(new_key(rng), None, "fail: unsupported_opcode: synthetic", failed=True))
    out.append(cp.Entry(new_key(rng, shared), None, "timeout: no reply in 60 s", failed=True))
    return out


def variant_entries(rng: random.Random, keys: list[str]) -> list[cp.Entry]:
    """Two variants of each key: one with texture answers, one with attribute answers."""
    out = []
    for key in keys:
        out.append(translated(rng, key, rng.randint(5, 80), answers=TEX_3D))
        out.append(translated(rng, key, rng.randint(5, 80), answers=ATTR_SINT))
    return out


def expected(entry: cp.Entry) -> tuple[int, str, str]:
    """(status, module SHA-256 or "-", description SHA-256) as the C test prints them."""
    mod = b"" if entry.failed else bytes(entry.module)
    desc = entry.description.encode("utf-8") if isinstance(entry.description, str) else bytes(entry.description)
    return (cp.FAILED if entry.failed else cp.TRANSLATED, hashlib.sha256(mod).hexdigest() if mod else "-",
            hashlib.sha256(desc).hexdigest())


def write_queries(path: Path, pairs: list[tuple[str, bytes]]) -> Path:
    path.write_text("".join(f"{k}\t{cp.digest_text(d)}\n" for k, d in pairs), encoding="ascii")
    return path


# --- damaged packs, shared by the Python and C reader tests -------------------

def rehash_index(data: bytearray) -> None:
    count = struct.unpack_from("<Q", data, 64)[0]
    data[96:128] = hashlib.sha256(bytes(data[cp.HEADER_BYTES:cp.HEADER_BYTES + count * cp.ROW_BYTES])).digest()


def damaged(kind: str, good: Path, out: Path) -> Path:
    """A copy of the base pack good, broken one way; "schema" is a whole pack of another description schema."""
    data = bytearray(good.read_bytes())
    pack = cp.open_pack(good)
    if kind == "schema":
        rng = random.Random(SEED + 1)
        cp.write_pack(out, [translated(rng, new_key(rng), 9, schema=cp.DESC_SCHEMA + 1) for _ in range(4)],
                      translator=TRANSLATOR, schema=cp.DESC_SCHEMA + 1)
        return out
    if kind == "truncated_index":
        data = data[:cp.HEADER_BYTES + (len(pack.rows) // 2) * cp.ROW_BYTES + 40]
    elif kind == "past_end":
        last = max(pack.rows, key=lambda r: r.offset)
        data = data[:last.offset + (last.module_size + last.description_size) // 2]
    elif kind == "index_hash":
        at = cp.HEADER_BYTES + 3 * cp.ROW_BYTES      # a key byte of row 3, still a key character
        data[at] = ord("x") if data[at] != ord("x") else ord("y")
    elif kind == "order":
        a, b = cp.HEADER_BYTES, cp.HEADER_BYTES + cp.ROW_BYTES
        data[a:a + cp.ROW_BYTES], data[b:b + cp.ROW_BYTES] = data[b:b + cp.ROW_BYTES], data[a:a + cp.ROW_BYTES]
        rehash_index(data)
    elif kind == "short":
        data = data[:100]
    elif kind == "magic":
        data[0:8] = b"OOT2RFS1"
    else:
        raise ValueError(kind)
    out.write_bytes(bytes(data))
    return out


# The name each damaged pack is refused under, by both readers.
DAMAGE = {"schema": "SCHEMA", "truncated_index": "INDEX_TRUNCATED", "past_end": "ENTRY_PAST_END",
          "index_hash": "INDEX_HASH", "order": "ORDER", "short": "SHORT", "magic": "MAGIC"}


class Packs:
    """A base pack and a variants pack in a temporary folder, with the entries they were written from."""

    def __init__(self, folder: Path, align: int = cp.ENTRY_ALIGN):
        rng = random.Random(SEED)
        self.folder = folder
        self.base_entries = base_entries(rng)
        translated_keys = [e.key for e in self.base_entries if not e.failed]
        self.variant_entries = variant_entries(rng, translated_keys[:6])
        self.base = folder / cp.BASE_FILE
        self.variants = folder / cp.VARIANTS_FILE
        self.base_stats = cp.write_pack(self.base, self.base_entries, translator=TRANSLATOR, align=align)
        cp.write_pack(self.variants, self.variant_entries, translator=TRANSLATOR, role=cp.VARIANTS, align=align)


# --- the Python writer and reader ---------------------------------------------

class PythonPackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-shc-")
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_round_trip_finds_every_entry(self) -> None:
        packs = Packs(self.dir)
        for path, entries, role in ((packs.base, packs.base_entries, cp.BASE),
                                    (packs.variants, packs.variant_entries, cp.VARIANTS)):
            pack = cp.open_pack(path, role=role)
            self.assertEqual(len(pack.rows), len(entries))
            self.assertEqual(pack.translator, TRANSLATOR)
            for e in entries:
                row = pack.find(e.key, e.answers)
                self.assertIsNotNone(row, e.key)
                mod, desc = pack.read(row)
                status, msha, dsha = expected(e)
                self.assertEqual(row.status, status)
                self.assertEqual(hashlib.sha256(mod).hexdigest() if mod else "-", msha)
                self.assertEqual(hashlib.sha256(desc).hexdigest(), dsha)
            self.assertEqual(cp.verify_pack(path, role=role), [])
        self.assertEqual(packs.base_stats["failures"], 2)

    def test_rows_follow_the_padded_key_bytes(self) -> None:
        packs = Packs(self.dir)
        rows = cp.open_pack(packs.base).rows
        self.assertEqual([r.sort_bytes() for r in rows], sorted(r.sort_bytes() for r in rows))
        order = [r.key for r in rows if r.key in ("k1", "k1.0", "k10", "K")]
        self.assertEqual(order, ["K", "k1", "k1.0", "k10"])       # NUL < '.' < '0', and 'K' < 'k'
        offsets = [r.offset for r in rows]
        self.assertEqual(offsets, sorted(offsets))
        self.assertTrue(all(o % cp.ENTRY_ALIGN == 0 for o in offsets))

    def test_a_pack_is_written_once(self) -> None:
        packs = Packs(self.dir)
        before = hashlib.sha256(packs.base.read_bytes()).digest()
        rng = random.Random(SEED + 2)
        with self.assertRaises(cp.PackError) as cm:
            cp.write_pack(packs.base, [translated(rng, new_key(rng), 9)], translator=TRANSLATOR)
        self.assertEqual(cm.exception.code, "EXISTS")
        self.assertEqual(hashlib.sha256(packs.base.read_bytes()).digest(), before)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), sorted((cp.BASE_FILE, cp.VARIANTS_FILE)))

    def test_each_role_keeps_its_answers(self) -> None:
        rng = random.Random(SEED + 3)
        cases = (
            ("a variant in a base pack", cp.BASE, [translated(rng, new_key(rng), 9, answers=TEX_3D)]),
            ("a default entry in a variants pack", cp.VARIANTS, [translated(rng, new_key(rng), 9)]),
        )
        for i, (what, role, entries) in enumerate(cases):
            with self.subTest(what):
                with self.assertRaises(cp.PackError) as cm:
                    cp.write_pack(self.dir / f"p{i}", entries, translator=TRANSLATOR, role=role)
                self.assertEqual(cm.exception.code, "INPUT")
        # A description with answers set, under the zero digest, is a variant all the same.
        key = new_key(rng)
        mod = module(rng, 9)
        with self.assertRaises(cp.PackError) as cm:
            cp.write_pack(self.dir / "p2", [cp.Entry(key, mod, description(key, mod, answers=TEX_3D))],
                          translator=TRANSLATOR)
        self.assertIn("in a base pack", cm.exception.message)
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_a_description_must_belong_to_its_module_and_pack(self) -> None:
        rng = random.Random(SEED + 4)
        key = new_key(rng)
        mod = module(rng, 12)
        other = module(rng, 12)
        bad = {
            "schema": description(key, mod, schema=cp.DESC_SCHEMA + 1),
            "key": description(new_key(rng), mod),
            "translator": description(key, mod, translator=OTHER_TRANSLATOR),
            "module": description(key, other),
            "json": b"{not json",
        }
        for what, desc in bad.items():
            with self.subTest(what):
                with self.assertRaises(cp.PackError) as cm:
                    cp.write_pack(self.dir / what, [cp.Entry(key, mod, desc)], translator=TRANSLATOR)
                self.assertEqual(cm.exception.code, "INPUT")
        for what, entry in {"not SPIR-V": cp.Entry(key, b"\0" * 20, description(key, b"\0" * 20)),
                            "a key too long": cp.Entry("k" * 32, mod, description("k" * 32, mod)),
                            "a key that is not ASCII": cp.Entry("k\u00e9", mod, description(key, mod)),
                            "a digest that is not bytes": cp.Entry(key, mod, description(key, mod), answers="0"),
                            "an empty reason": cp.Entry(key, None, "", failed=True)}.items():
            with self.subTest(what):
                with self.assertRaises(cp.PackError) as cm:
                    cp.write_pack(self.dir / "x", [entry], translator=TRANSLATOR)
                self.assertEqual(cm.exception.code, "INPUT")
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_answers_digest(self) -> None:
        self.assertEqual(cp.answers_digest({}), cp.ZERO_DIGEST)
        self.assertEqual(cp.answers_digest({"key": "x", "tex": {}, "attr": {}, "options": {"debug": False}}),
                         cp.ZERO_DIGEST)
        a = cp.answers_digest({"tex": {"1": "3d", "0": "cube"}, "fout": {"0": "uint"}})
        b = cp.answers_digest({"fout": {"0": "uint"}, "tex": {"0": "cube", "1": "3d"}, "stage": "ps"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, cp.ZERO_DIGEST)
        self.assertNotEqual(a, cp.answers_digest({"tex": {"1": "3d", "0": "cube"}}))
        text = json.dumps({"fout": {"0": "uint"}, "tex": {"0": "cube", "1": "3d"}}, sort_keys=True,
                          separators=(",", ":"))
        self.assertEqual(a, hashlib.sha256(text.encode("ascii")).digest())

    def test_the_command_line_packs_a_folder_with_its_failures(self) -> None:
        rng = random.Random(SEED + 5)
        loose = self.dir / "spv"
        loose.mkdir()
        keys = [new_key(rng) for _ in range(6)]
        for key in keys:
            mod = module(rng, rng.randint(5, 50))
            (loose / f"{key}.spv").write_bytes(mod)
            (loose / f"{key}.res.json").write_bytes(description(key, mod))
        failed = [new_key(rng), new_key(rng)]
        results = self.dir / "results.tsv"
        results.write_text("key\tstatus\tmicros\twall_ms\tspv_bytes\tmessage\n"
                           + "".join(f"{k}\tok\t1\t1.0\t20\t\n" for k in keys)
                           + f"{failed[0]}\tfail\t3\t1.0\t0\tunsupported_opcode: synthetic\n"
                           + f"{failed[1]}\tcrash\t0\t9.0\t0\t\n", encoding="utf-8")
        out = self.dir / "pack.bin"
        with contextlib.redirect_stdout(io.StringIO()) as said:
            self.assertEqual(cp.main(["write", "--spv-dir", str(loose), "--translator", TRANSLATOR, "--out", str(out),
                                      "--results", str(results)]), 0)
        self.assertIn("8 rows (2 failure records)", said.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as listed:
            self.assertEqual(cp.main(["list", str(out)]), 0)
        rows = [line.split("\t") for line in listed.getvalue().splitlines()]
        self.assertEqual(sorted(r[0] for r in rows), sorted(keys + failed))
        self.assertTrue(all(r[1] == "0" for r in rows))
        status = {r[0]: r[2] for r in rows}
        self.assertEqual({k for k, s in status.items() if s == str(cp.FAILED)}, set(failed))
        pack = cp.open_pack(out)
        self.assertEqual(pack.read(pack.find(failed[1]))[1], b"crash: no reason given")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cp.main(["verify", str(out)]), 0)
        # A module without its description is refused.
        (loose / f"{keys[0]}.res.json").unlink()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(cp.main(["write", "--spv-dir", str(loose), "--translator", TRANSLATOR, "--out",
                                      str(self.dir / "again.bin")]), 1)
        self.assertIn("INPUT", err.getvalue())

    def test_verify_names_a_damaged_entry(self) -> None:
        packs = Packs(self.dir)
        pack = cp.open_pack(packs.base)
        row = next(r for r in pack.rows if not r.failed)
        data = bytearray(packs.base.read_bytes())
        data[row.offset + row.module_size - 1] ^= 0xFF
        damaged_pack = self.dir / "damaged.bin"
        damaged_pack.write_bytes(bytes(data))
        bad = cp.verify_pack(damaged_pack)
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0].startswith(row.key))

    def test_python_reader_refusals(self) -> None:
        packs = Packs(self.dir)
        for kind, name in DAMAGE.items():
            with self.subTest(kind):
                path = damaged(kind, packs.base, self.dir / f"{kind}.bin")
                with self.assertRaises(cp.PackError) as cm:
                    cp.open_pack(path)
                self.assertEqual(cm.exception.code, name)
        with self.assertRaises(cp.PackError) as cm:
            cp.open_pack(packs.base, role=cp.VARIANTS)
        self.assertEqual(cm.exception.code, "ROLE")

    def test_constants_agree_with_the_c_header_and_the_translator(self) -> None:
        header = HEADER_FILE.read_text(encoding="utf-8")
        defines = dict(re.findall(r"^#define (SHC_\w+) (\S+)", header, re.M))
        self.assertEqual(defines["SHC_MAGIC"], f'"{cp.MAGIC.decode()}"')
        for name, value in (("SHC_VERSION", cp.VERSION), ("SHC_HEADER_BYTES", cp.HEADER_BYTES),
                            ("SHC_ROW_BYTES", cp.ROW_BYTES), ("SHC_KEY_BYTES", cp.KEY_BYTES),
                            ("SHC_DIGEST_BYTES", cp.DIGEST_BYTES), ("SHC_ENTRY_ALIGN", cp.ENTRY_ALIGN),
                            ("SHC_DESC_SCHEMA", cp.DESC_SCHEMA)):
            self.assertEqual(int(defines[name].rstrip("u")), value, name)
        roles = re.search(r"SHC_ROLE_BASE = (\d+), SHC_ROLE_VARIANTS = (\d+)", header)
        self.assertEqual((int(roles[1]), int(roles[2])), (cp.BASE, cp.VARIANTS))
        status = re.search(r"SHC_TRANSLATED = (\d+), SHC_FAILED = (\d+)", header)
        self.assertEqual((int(status[1]), int(status[2])), (cp.TRANSLATED, cp.FAILED))
        schema = re.search(r"public const int Schema = (\d+);", RESULT_WRITER.read_text(encoding="utf-8"))
        self.assertEqual(int(schema[1]), cp.DESC_SCHEMA)


# --- the C reader -------------------------------------------------------------

def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class CReaderTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-shc-c-")
        out = Path(cls.tmp.name)
        extra = [] if os.name == "nt" else ["-pthread"]
        for level in LEVELS:
            exe = out / f"shader_cache_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(ROOT / "runtime" / "include"), "-I", str(ROOT / "runtime" / "core"),
                   *map(str, SOURCES), "-o", str(exe)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory(prefix="oot2-shc-run-")
        self.dir = Path(self.work.name)

    def tearDown(self) -> None:
        self.work.cleanup()

    def run_c(self, level: str, *args: str | Path) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), *map(str, args)], capture_output=True, text=True, timeout=120)

    def lines(self, *args: str | Path) -> list[str]:
        """The output of one case at every level, which must agree."""
        outs = []
        for level in LEVELS:
            done = self.run_c(level, *args)
            self.assertEqual(done.returncode, 0, f"{level} {args[0]}: {done.stdout}{done.stderr}")
            outs.append(done.stdout.splitlines())
        self.assertEqual(outs[0], outs[1])
        return outs[0]

    def test_c_finds_what_python_listed(self) -> None:
        packs = Packs(self.dir)
        rng = random.Random(SEED + 6)
        for path, entries, role in ((packs.base, packs.base_entries, "base"),
                                    (packs.variants, packs.variant_entries, "variants")):
            with self.subTest(role):
                listed = cp.open_pack(path).rows
                by_pair = {(e.key, e.answers): e for e in entries}
                self.assertEqual({(r.key, r.answers) for r in listed}, set(by_pair))
                misses = [(new_key(rng), cp.ZERO_DIGEST), ("k1.", cp.ZERO_DIGEST), ("k", cp.ZERO_DIGEST),
                          (listed[0].key, cp.answers_digest({"fout": {"0": "sint"}}))]
                pairs = [(r.key, r.answers) for r in listed] + misses
                got = self.lines("find", path, role, write_queries(self.dir / f"{role}.q", pairs))
                want = []
                for key, digest in pairs:
                    e = by_pair.get((key, digest))
                    if e is None:
                        want.append(f"miss {key} {cp.digest_text(digest)}")
                    else:
                        status, msha, dsha = expected(e)
                        want.append(f"hit {key} {cp.digest_text(digest)} {status} {msha} {dsha} 1")
                self.assertEqual(got, want)
                if role == "base":
                    self.assertEqual(sum(line.split()[3] == str(cp.FAILED) for line in got if line.startswith("hit")),
                                     2)

    def test_c_finds_an_entry_without_touching_the_others(self) -> None:
        packs = Packs(self.dir, align=PAGE)
        pack = cp.open_pack(packs.base)

        def pages(r: cp.Row) -> tuple[int, int]:
            return r.offset, -(-(r.offset + r.module_size + r.description_size) // PAGE) * PAGE

        translated_rows = [r for r in pack.rows if not r.failed]
        for target in (translated_rows[0], translated_rows[len(translated_rows) // 2], translated_rows[-1]):
            with self.subTest(target.key):
                others = [str(n) for r in pack.rows if r is not target for n in pages(r)]
                e = next(x for x in packs.base_entries if x.key == target.key)
                _, msha, dsha = expected(e)
                got = self.lines("isolated", packs.base, "base", target.key, "0", *others)
                self.assertEqual(got, [f"hit {target.key} {msha} {dsha}"])
        # Control: with the target's own pages protected as well, the lookup's read is caught.
        target = translated_rows[1]
        for level in LEVELS:
            done = self.run_c(level, "isolated", packs.base, "base", target.key, "0",
                              *(str(n) for r in pack.rows for n in pages(r)))
            self.assertEqual(done.returncode, 3, done.stdout + done.stderr)
            self.assertIn(f"touched +0x{target.offset:x}", done.stderr)

    def test_c_takes_the_variant_before_the_base(self) -> None:
        packs = Packs(self.dir)
        base = {e.key: e for e in packs.base_entries}
        variants = {(e.key, e.answers): e for e in packs.variant_entries}
        key = packs.variant_entries[0].key
        tex, attr = cp.answers_digest(TEX_3D), cp.answers_digest(ATTR_SINT)
        only_base = next(e.key for e in packs.base_entries if not e.failed and (e.key, tex) not in variants)
        other = cp.answers_digest({"img": {"*": {"type": "3d", "fmt": "R32Float"}}})
        queries = [(key, tex), (key, attr), (key, cp.ZERO_DIGEST), (key, other), (only_base, tex),
                   ("k" * 9, tex)]

        def line(source: str, e: cp.Entry) -> str:
            status, msha, dsha = expected(e)
            return f"{source} {e.key} {cp.digest_text(e.answers)} {status} {msha} {dsha}"

        want = [line("variants", variants[(key, tex)]), line("variants", variants[(key, attr)]),
                line("base", base[key]), line("base", base[key]), line("base", base[only_base]),
                f"miss {'k' * 9} {cp.digest_text(tex)}"]
        q = write_queries(self.dir / "set.q", queries)
        self.assertEqual(self.lines("set", packs.base, packs.variants, q), want)
        # Without a variants pack every query takes the base pack's row.
        self.assertEqual(self.lines("set", packs.base, "-", q),
                         [line("base", base[key])] * 4 + [line("base", base[only_base]), want[-1]])

    def test_c_maps_full_ledger_digests_to_projected_variants_and_falls_back(self) -> None:
        packs = Packs(self.dir)
        base = {e.key: e for e in packs.base_entries}
        variant = packs.variant_entries[0]
        source = hashlib.sha256(b"full ledger answers, including unread slots").digest()
        base_source = hashlib.sha256(b"default ledger answers").digest()
        unknown = hashlib.sha256(b"row absent from map").digest()
        base_key = next(e.key for e in packs.base_entries if not e.failed and e.key != variant.key)
        entries = {(variant.key, source): variant.answers, (base_key, base_source): cp.ZERO_DIGEST}
        map_path = self.dir / "variant-map.bin"
        map_path.write_bytes(prepare_ledger.variant_map_bytes(entries, hashlib.sha256(b"ledger").hexdigest()))
        self.assertNotEqual(source, variant.answers)

        def line(source_name: str, entry: cp.Entry, digest: bytes) -> str:
            status, msha, dsha = expected(entry)
            return f"{source_name} {entry.key} {digest.hex()} {status} {msha} {dsha}"

        queries = [(variant.key, source), (base_key, base_source), (variant.key, unknown)]
        q = write_queries(self.dir / "map.q", queries)
        self.assertEqual(self.lines("map", packs.base, packs.variants, map_path, q),
                         [line("variants", variant, source), line("base", base[base_key], base_source),
                          line("fallback", base[variant.key], unknown)])
        self.assertEqual(self.lines("map", packs.base, packs.variants, "-", q)[0],
                         line("fallback", base[variant.key], source))

        damaged_map = bytearray(map_path.read_bytes())
        damaged_map[-1] ^= 0x80
        bad_path = self.dir / "damaged-map.bin"
        bad_path.write_bytes(damaged_map)
        self.assertTrue(self.lines("map", packs.base, packs.variants, bad_path, q)[0].startswith("refused MAP: "))

    def assert_refused(self, path: Path, name: str, role: str = "base") -> None:
        got = self.lines("open", path, role, str(cp.DESC_SCHEMA))
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith(f"refused {name}: "), got[0])

    def test_c_opens_a_good_pack(self) -> None:
        packs = Packs(self.dir)
        self.assertEqual(self.lines("open", packs.base, "base", str(cp.DESC_SCHEMA)),
                         [f"ok {len(packs.base_entries)}"])

    def test_c_refuses_a_wrong_schema(self) -> None:
        packs = Packs(self.dir)
        path = damaged("schema", packs.base, self.dir / "schema.bin")
        self.assert_refused(path, "SCHEMA")
        self.assertEqual(self.lines("open", path, "base", str(cp.DESC_SCHEMA + 1)), ["ok 4"])

    def test_c_refuses_a_truncated_index(self) -> None:
        packs = Packs(self.dir)
        self.assert_refused(damaged("truncated_index", packs.base, self.dir / "cut.bin"), "INDEX_TRUNCATED")

    def test_c_refuses_an_entry_past_the_end_of_the_file(self) -> None:
        packs = Packs(self.dir)
        self.assert_refused(damaged("past_end", packs.base, self.dir / "past.bin"), "ENTRY_PAST_END")

    def test_c_refuses_damage_order_role_and_translator(self) -> None:
        packs = Packs(self.dir)
        for kind in ("index_hash", "order", "short", "magic"):
            with self.subTest(kind):
                self.assert_refused(damaged(kind, packs.base, self.dir / f"{kind}.bin"), DAMAGE[kind])
        self.assert_refused(packs.base, "ROLE", role="variants")
        self.assert_refused(self.dir / "absent.bin", "OPEN")
        rng = random.Random(SEED + 7)
        foreign = self.dir / "foreign.bin"
        cp.write_pack(foreign, [translated(rng, new_key(rng), 9, answers=TEX_3D, translator=OTHER_TRANSLATOR)],
                      translator=OTHER_TRANSLATOR, role=cp.VARIANTS)
        got = self.lines("set", packs.base, foreign, write_queries(self.dir / "q", []))
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("refused TRANSLATOR: "), got[0])

    def test_bench_runs(self) -> None:
        packs = Packs(self.dir)
        rows = cp.open_pack(packs.base).rows
        q = write_queries(self.dir / "bench.q", [(r.key, r.answers) for r in rows])
        for level in LEVELS:
            done = self.run_c(level, "bench", packs.base, "base", q, "2")
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            out = done.stdout.splitlines()
            self.assertTrue(out[0].startswith("bench open_ms="), out[0])
            self.assertIn(f"rows={len(rows)} ", out[0])
            rounds = [line for line in out if line.startswith("bench round=")]
            self.assertEqual(len(rounds), 2)
            for line in rounds:
                self.assertIn(f"found={len(rows)} ", line)
            self.assertTrue(out[-1].startswith("bench checksum="))


if __name__ == "__main__":
    unittest.main()
