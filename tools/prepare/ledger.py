# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The ledger step of `oot2 prepare` (architecture D21): the answers the game used on its last run, translated as
variants and packed into the shader cache's variants pack.

Input: ledger.jsonl in the output folder, which oot2.exe writes when a run ends (runtime/include/oot2/ledger.h), and
what the translate step left: shaders/shader-base.bin, shaders/split/. A first prepare has no ledger yet; the step then
records that and writes nothing, and the run that follows the game's first run translates what it recorded.

Output: shaders/shader-variants.bin (tools/shaders/cache_pack.py, role variants), one row for each (program, answers
digest) the ledger asks for that its module reads and the default translation did not already use; and
shaders/shader-variant-map.bin, a sorted map from the ledger's full answers digest to the projected digest or base.
Without variants the variants pack is deleted, but the map is still written. The base pack is never written here.

What a row asks for. A ledger row holds what was bound to its stage when a draw ran (the ledger study), a superset of what
the stage's module reads, so each row is cut down to the answers the module uses before it counts:

- A texture answer for slot s counts when the translator asked for that slot's type: the description's "queries"
  hold a sampler_type query at texture slot s (the translator asks only when an instruction does not carry the type,
  as a TEX instruction does: on the real translator a program that only samples a texture asks nothing and ignores the
  answer). An image answer counts for the type when a sampler_type query names the image slot and for the format when
  a texture_format query does. A vertex attribute answer counts when a vertex module declares an input at that
  location (the description's vertex_inputs); a colour target answer when a pixel module declares an output at that
  location. Answers on other slots are dropped. A description whose query record overflowed ("queries_dropped") reads
  texture and image slots from its declared textures and images instead, which can only add answers.
- An answer counts only when it differs from what the default translation used for that slot: the answer the query
  records (the default 2D, or the type recovered from the module's own instructions; the default image format), the
  attribute's recorded answer, and "float" for a colour target. A row whose answers are all dropped is a default row
  and adds no variant.
- What is left is the variant's answers. Its digest is the answers digest of the job that carries them
  (cache_pack.answers_digest), so rows that differ only in slots the module does not read give one variant. The map
  connects each full ledger-row digest to that projected digest, or to the base when projection adds nothing.
- A variant whose translation is byte for byte the base module and whose description, apart from its answers and
  timing, is the base description's adds nothing either, and is not packed: the runtime reader finds no variant for
  the digest and takes the base pack's row, which is that same module and description. The step counts these.

Each variant is translated again from the program's job with those answers (corpus.variant_job), by the pool of
translator children the translate step uses, and packed with the translator's module and description. A variant the
translator refuses, or that gets no answer twice (the second time alone), is packed as a failure record, as the base
pack does: the runtime then stops with the reason by name, and never serves the default module for a draw whose answers
it does not match. A variant has the key of its program, so variants of one program are translated in separate rounds,
and each answer's module and description are held in memory until the pack writer takes them (a handful of variants,
so the spill the translate step needs for the whole game is not worth the same care here).

The ledger is checked as it is read, because it is a file on the player's disk: a header other than {"oot2_ledger":1},
a line that is not a row, an answer outside the translator's vocabulary or a digest that does not match the answers
stops the step with the line named. A row for a program the split does not hold, for a stage other than the program's,
or for a program the base pack holds only as a failure record is counted, named in a warning and skipped.

The key: the SHA-256 of the ledger file ("none" without one), the translator's identity, the description schema and
the code of this step and of corpus.py (the variant job), and through Step.needs the translate step's key. A new ledger
runs this step and nothing else; a new translator runs the translate step again and this one after it. The ledger is
not an input of any other step.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import check, translate
from .runner import Context, PrepareError, Step, fingerprint

TOOLS = Path(__file__).resolve().parent.parent
SHADER_TOOLS = TOOLS / "shaders"
for _p in (TOOLS, SHADER_TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cache_pack  # noqa: E402  (tools/shaders/cache_pack.py)
import corpus  # noqa: E402  (tools/shaders/corpus.py: variant_job)
import pool  # noqa: E402  (tools/shaders/pool.py)

FORMAT = 2                      # changes when the outputs change shape: the step runs again
HERE = Path(__file__).resolve().parent
LEDGER_FILE = "ledger.jsonl"    # runtime/include/oot2/ledger.h LEDGER_FILE, in the output folder (the game folder)
VARIANT_MAP_FILE = "shader-variant-map.bin"
MAP_MAGIC = b"OOT2VM1\0"
MAP_VERSION = 1
MAP_HEADER = struct.Struct("<8sIIIIQ32s32s")
MAP_ROW = struct.Struct("<32s32s32s")
HEADER = {"oot2_ledger": 1}
SCRATCH = "ledger"              # temp/ledger/
KINDS = ("vs", "ps", "cs", "ts", "ms")
FIELDS = cache_pack.ANSWER_FIELDS
TEX_TYPES = frozenset(("1d", "1darray", "2d", "2darray", "2dms", "2dmsarray", "3d", "cube", "cubearray", "buffer"))
CLASSES = frozenset(("float", "sint", "uint"))
SLOT = re.compile(r"[0-9]{1,3}")
IMAGE_TABLE = 32                # texture slots 32 to 39 are the image table's: BindTexture there answers from "img"
NAMED = 5                       # problems named one by one in a warning


# --- the ledger -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    line: int
    key: str
    kind: str
    answers: dict               # {"tex": {"3": "2darray"}, "img": {"1": {"fmt": ..., "type": ...}}, ...}
    digest: bytes               # SHA-256 of all answers, not the projected job answers


def canonical(answers: dict) -> str:
    """The canonical JSON text of an answers object, as the ledger's writer and cache_pack.answers_digest build it."""
    return json.dumps(answers, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def ledger_digest(path: Path) -> str:
    """The SHA-256 of the ledger file in hexadecimal, or "none" when there is no ledger yet."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return "none"
    except OSError as e:
        raise PrepareError(f"the ledger {path} cannot be read: {e.strerror or e}") from None


def _ledger_digest_bytes(digest: str) -> bytes:
    return bytes(32) if digest == "none" else bytes.fromhex(digest)


def variant_map_bytes(destinations: dict[tuple[str, bytes], bytes], digest: str) -> bytes:
    """The sorted (stage key, full ledger digest) to projected digest map. A zero target means base."""
    records = []
    for (key, source), target in sorted(destinations.items(), key=lambda item: (item[0][0].encode("ascii"),
                                                                                item[0][1])):
        encoded = key.encode("ascii")
        if not encoded or len(encoded) >= 32 or len(source) != 32 or len(target) != 32:
            raise PrepareError("a shader variant map row is outside its fixed fields")
        records.append(MAP_ROW.pack(encoded.ljust(32, b"\0"), source, target))
    body = b"".join(records)
    header = MAP_HEADER.pack(MAP_MAGIC, MAP_VERSION, MAP_HEADER.size, MAP_ROW.size, 0, len(records),
                             _ledger_digest_bytes(digest), hashlib.sha256(body).digest())
    return header + body


def read_variant_map(path: Path) -> tuple[str, dict[tuple[str, bytes], bytes]] | None:
    """Read and validate the variant map, returning its source ledger digest and entries."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < MAP_HEADER.size:
        return None
    magic, version, header_size, row_size, reserved, count, ledger_sha, rows_sha = MAP_HEADER.unpack_from(data)
    if (magic != MAP_MAGIC or version != MAP_VERSION or header_size != MAP_HEADER.size or row_size != MAP_ROW.size or
            reserved or count > (len(data) - header_size) // row_size or
            len(data) != header_size + count * row_size):
        return None
    body = data[header_size:]
    if hashlib.sha256(body).digest() != rows_sha:
        return None
    destinations: dict[tuple[str, bytes], bytes] = {}
    previous = None
    for offset in range(0, len(body), row_size):
        key_field, source, target = MAP_ROW.unpack_from(body, offset)
        end = key_field.find(b"\0")
        if end <= 0 or any(key_field[end:]):
            return None
        try:
            key = key_field[:end].decode("ascii")
        except UnicodeDecodeError:
            return None
        if not cache_pack.KEY_RE.fullmatch(key):
            return None
        order = (key_field, source)
        if previous is not None and order <= previous:
            return None
        previous = order
        destinations[(key, source)] = target
    return ("none" if ledger_sha == bytes(32) else ledger_sha.hex()), destinations


def read_ledger(path: Path) -> tuple[list[Row], str]:
    """The rows of the ledger file and its SHA-256 ("none" and no rows without a file). PrepareError, naming the
    line, for a file that is not what oot2.exe writes."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [], "none"
    except OSError as e:
        raise PrepareError(f"the ledger {path} cannot be read: {e.strerror or e}") from None
    digest = hashlib.sha256(data).hexdigest()

    def bad(line: int, why: str) -> PrepareError:
        return PrepareError(f"the ledger {path}, line {line}: {why}. oot2.exe writes this file when a run ends; delete "
                            "it to start a new one, or put back a good copy.")

    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        raise PrepareError(f"the ledger {path} is not UTF-8 text. Delete it to start a new one.") from None
    if not lines or not _is_header(lines[0]):
        raise bad(1, f"the first line is not the header {json.dumps(HEADER, separators=(',', ':'))} (another file, "
                     "or a ledger of another format)")
    rows: list[Row] = []
    for n, text in enumerate(lines[1:], 2):
        if not text.strip():
            continue
        try:
            obj = json.loads(text)
        except ValueError:
            raise bad(n, "it is not a JSON object") from None
        try:
            rows.append(check_row(n, obj))
        except ValueError as e:
            raise bad(n, str(e)) from None
    return rows, digest


def _is_header(text: str) -> bool:
    try:
        return json.loads(text) == HEADER
    except ValueError:
        return False


def check_row(line: int, obj) -> Row:
    """A ledger row as a Row; ValueError, in a few words, for one that is not shaped as the writer's rows are."""
    if not isinstance(obj, dict):
        raise ValueError("it is not a JSON object")
    key, kind, answers, digest = obj.get("key"), obj.get("kind"), obj.get("answers"), obj.get("digest")
    if not isinstance(key, str) or not cache_pack.KEY_RE.fullmatch(key):
        raise ValueError("the key is not 1 to 31 letters, digits, '.', '_' or '-'")
    if kind not in KINDS:
        raise ValueError(f"the stage kind {kind!r} is none of {', '.join(KINDS)}")
    if not isinstance(answers, dict) or set(answers) - set(FIELDS):
        raise ValueError(f"the answers are not an object of {', '.join(FIELDS)}")
    for name, slots in answers.items():
        if not isinstance(slots, dict):
            raise ValueError(f"the {name} answers are not an object")
        for slot, value in slots.items():
            if not SLOT.fullmatch(slot):
                raise ValueError(f"the {name} slot {slot!r} is not a whole number")
            if not _answer_ok(name, value):
                raise ValueError(f"the {name} answer {json.dumps(value)} for slot {slot} is not one the translator "
                                 "takes")
    expect = hashlib.sha256(canonical(answers).encode("ascii")).hexdigest() if answers else "0"
    if digest != expect:
        raise ValueError("the digest does not match the answers (the row is damaged)")
    return Row(line, key, kind, answers, cache_pack.ZERO_DIGEST if digest == "0" else bytes.fromhex(digest))


def _answer_ok(name: str, value) -> bool:
    if name == "tex":
        return value in TEX_TYPES
    if name == "img":
        return isinstance(value, dict) and set(value) == {"fmt", "type"} and value["type"] in TEX_TYPES \
            and isinstance(value["fmt"], str) and bool(value["fmt"])
    return value in CLASSES


# --- what a module reads --------------------------------------------------------------------------------------------

@dataclass
class Reads:
    """What a program's default translation says about the answers it takes, from its description (res.json): for each
    slot the module reads, what the translation used there."""
    stage: str
    tex: dict[int, set]         # texture slot -> the types the translator was asked for there and used
    img_type: dict[int, set]    # image slot -> the types it was asked for and used
    img_fmt: dict[int, set]     # image slot -> the formats it was asked for and used
    attr: dict[int, set]        # vertex attribute location -> the answers the description records
    fout: set                   # colour target locations a pixel module declares


def description_of(pack: cache_pack.Pack, row: cache_pack.Row) -> dict:
    """The description of a translated row of the base pack, read without its module."""
    try:
        with open(pack.path, "rb") as f:
            f.seek(row.offset + row.module_size)
            desc = json.loads(f.read(row.description_size))
    except (OSError, ValueError) as e:
        raise PrepareError(f"the description of {row.key} in {pack.path.name} cannot be read ({e}). Run oot2 prepare "
                           "--verify to check the pack.") from None
    if not isinstance(desc, dict):
        raise PrepareError(f"the description of {row.key} in {pack.path.name} is not a JSON object. Run oot2 prepare "
                           "--verify to check the pack.")
    return desc


def reads_of(desc: dict) -> Reads:
    """The reads of one description. Entries that are not shaped as the translator writes them are left out, which
    only drops answers (a slot nothing lists is a slot the module does not read)."""
    stage = desc.get("stage")
    tex: dict[int, set] = {}
    img_type: dict[int, set] = {}
    img_fmt: dict[int, set] = {}
    attr: dict[int, set] = {}
    fout: set = set()
    if not desc.get("queries_dropped"):
        # The translator's own record of the answers it asked for (xlate/host/Accessor.cs): an image handle is answered
        # from "img" only, any other handle with a texture slot from "tex".
        for q in desc.get("queries") or ():
            if not isinstance(q, dict) or type(q.get("slot")) is not int or type(q.get("image_slot")) is not int:
                continue
            slot, islot, answer = q["slot"], q["image_slot"], q.get("answer")
            if q.get("kind") == "sampler_type":
                if islot >= 0:
                    img_type.setdefault(islot, set()).add(answer)
                elif slot >= 0:
                    tex.setdefault(slot, set()).add(answer)
            elif q.get("kind") == "texture_format" and islot >= 0:
                img_fmt.setdefault(islot, set()).add(answer)
    else:
        # The record overflowed: what the module declares stands in for it, so no answer it might read is lost.
        for t in desc.get("textures") or ():
            slot = t.get("c2_slot") if isinstance(t, dict) else None
            if type(slot) is not int or slot < 0 or t.get("kind") == "sampler":
                continue                # a bare sampler has no type to answer
            if IMAGE_TABLE <= slot < IMAGE_TABLE + 8:
                img_type.setdefault(slot - IMAGE_TABLE, set()).add(t.get("type"))
            else:
                tex.setdefault(slot, set()).add(t.get("type"))
        for i in desc.get("images") or ():
            slot = i.get("c2_slot") if isinstance(i, dict) else None
            if type(slot) is int and slot >= 0:
                img_type.setdefault(slot, set()).add(i.get("type"))
                img_fmt.setdefault(slot, set()).add(i.get("format"))
    for v in desc.get("vertex_inputs") or ():           # null outside a vertex module
        loc = v.get("location") if isinstance(v, dict) else None
        if type(loc) is int and loc >= 0:
            attr.setdefault(loc, set()).add(v.get("answer"))
    if stage == "ps":
        for o in (desc.get("interface") or {}).get("outputs") or ():
            loc = o.get("location") if isinstance(o, dict) else None
            if type(loc) is int and loc >= 0:
                fout.add(loc)
    return Reads(stage, tex, img_type, img_fmt, attr, fout)


def project(answers: dict, reads: Reads) -> dict:
    """The answers of a ledger row that this module reads and that differ from what its default translation used. {}
    for a row that adds nothing."""
    out: dict[str, dict] = {}
    for slot, value in answers.get("tex", {}).items():
        if int(slot) in reads.tex and reads.tex[int(slot)] != {value}:
            out.setdefault("tex", {})[slot] = value
    for slot, value in answers.get("img", {}).items():
        n = int(slot)
        if (n in reads.img_type and reads.img_type[n] != {value["type"]}) or                 (n in reads.img_fmt and reads.img_fmt[n] != {value["fmt"]}):
            out.setdefault("img", {})[slot] = value
    for slot, value in answers.get("attr", {}).items():
        if int(slot) in reads.attr and reads.attr[int(slot)] != {value}:
            out.setdefault("attr", {})[slot] = value
    for slot, value in answers.get("fout", {}).items():
        if int(slot) in reads.fout and value != "float":
            out.setdefault("fout", {})[slot] = value
    return out


# --- from rows to variants ------------------------------------------------------------------------------------------

@dataclass
class Variant:
    key: str
    digest: bytes
    job: dict
    answers: dict
    rank: int = 0               # among the variants of one program, in digest order


def plan(rows: list[Row], jobs: dict[str, dict], base: cache_pack.Pack) -> tuple[list[Variant], dict, dict]:
    """The variants a ledger asks for, sorted by key and digest, plus each usable row's projected digest or base."""
    counts = {"rows": len(rows), "programs": len({r.key for r in rows}), "unknown_programs": 0, "other_stage": 0,
              "base_failed": 0, "default_rows": 0, "variant_rows": 0, "merged_rows": 0}
    problems: dict[str, list[str]] = {"unknown_programs": [], "other_stage": [], "base_failed": []}
    reads: dict[str, Reads | None] = {}
    found: dict[tuple[str, bytes], Variant] = {}
    destinations: dict[tuple[str, bytes], bytes] = {}

    def skip(name: str, row: Row) -> None:
        counts[name] += 1
        if row.key not in problems[name]:
            problems[name].append(row.key)

    for row in rows:
        job = jobs.get(row.key)
        if job is None:
            skip("unknown_programs", row)
            continue
        if job.get("stage") != row.kind:
            skip("other_stage", row)
            continue
        if row.key not in reads:
            entry = base.find(row.key)
            reads[row.key] = None if entry is None or entry.failed else reads_of(description_of(base, entry))
        if reads[row.key] is None:
            skip("base_failed", row)
            continue
        kept = project(row.answers, reads[row.key])
        if not kept:
            counts["default_rows"] += 1
            destinations[(row.key, row.digest)] = cache_pack.ZERO_DIGEST
            continue
        variant = corpus.variant_job(job, kept)
        digest = cache_pack.answers_digest(variant)
        destinations[(row.key, row.digest)] = digest
        if (row.key, digest) in found:
            counts["merged_rows"] += 1
        else:
            found[(row.key, digest)] = Variant(row.key, digest, variant, kept)
        counts["variant_rows"] += 1
    variants = [found[k] for k in sorted(found)]
    rank: dict[str, int] = {}
    for v in variants:
        v.rank = rank[v.key] = rank.get(v.key, -1) + 1
    counts["problems"] = {name: keys[:NAMED] for name, keys in problems.items() if keys}
    return variants, counts, destinations


# --- translation ----------------------------------------------------------------------------------------------------

def run_batch(command: list[str], items: list[tuple[dict, Path]], logs: Path, workers: int, limit: int,
              timeout: float) -> dict[str, pool.Result]:
    """Translate (job, folder) pairs, each key once, on the pool and return the pool's result for each key. Ctrl+C
    stops feeding; the jobs in flight finish, then KeyboardInterrupt goes on to the runner (the same wait
    translate.translate keeps: never Thread.join, which Python 3.12 lets a KeyboardInterrupt corrupt)."""
    got: dict[str, pool.Result] = {}
    if not items:
        return got
    lock = threading.Lock()
    stop, finished = threading.Event(), threading.Event()
    box: dict = {}

    def on_result(r: pool.Result) -> None:
        with lock:
            got[r.key] = r

    def feed():
        for item in items:
            if stop.is_set():
                return
            yield item

    def work() -> None:
        try:
            pool.run_pool(command, feed(), min(workers, len(items)), logs, timeout=timeout, memory_limit=limit,
                          on_result=on_result)
        except BaseException as e:      # reported on the calling thread
            box["error"] = e
        finally:
            finished.set()

    threading.Thread(target=work, name="ledger-pool", daemon=True).start()
    try:
        while not finished.wait(0.2):
            pass
    except KeyboardInterrupt:
        stop.set()
        while not finished.wait(0.2):
            pass
        raise
    if "error" in box:
        raise box["error"]
    return got


def translate_variants(command: list[str], variants: list[Variant], scratch: Path, workers: int, limit: int,
                       timeout: float) -> dict[tuple[str, bytes], pool.Result]:
    """Every variant, in rounds: round n takes the n-th variant of each program. Variants of one program go in
    separate rounds because a program's key names its files, and the pool answers with the two parts so no loose
    folder is needed. The pool's result for each (key, digest)."""
    out: dict[tuple[str, bytes], pool.Result] = {}
    for rank in sorted({v.rank for v in variants}):
        batch = [v for v in variants if v.rank == rank]
        results = run_batch(command, [(v.job, None) for v in batch], scratch / "logs", workers, limit, timeout)
        for v in batch:
            out[(v.key, v.digest)] = results.get(v.key) or pool.Result(v.key, "crash", message="no result")
    return out


VOLATILE = ("answers", "timing")    # what a description holds that the answers and the clock decide


def same_as_base(base: cache_pack.Pack, key: str, module: bytes, desc: bytes) -> bool:
    """Whether a translated variant is the base entry again: the same module bytes and the same description apart from
    its answers and its timing (a description that records an answer the module did not use, such as a vertex
    attribute's, differs, and the variant stays)."""
    base_module, base_desc = base.read(base.find(key))
    if module != base_module:
        return False
    a, b = json.loads(base_desc), json.loads(desc)
    for name in VOLATILE:
        a.pop(name, None)
        b.pop(name, None)
    return a == b


def entries_of(variants: list[Variant], outcome: dict[tuple[str, bytes], pool.Result], scratch: Path,
               base: cache_pack.Pack) -> tuple[list, int, set[tuple[str, bytes]]]:
    """The variants pack's entries and how many variants were left out: a translated variant's module and description
    (carried in the translator's reply), a failure record for one that did not translate, nothing for one that is the
    base entry again."""
    entries = []
    identical = 0
    identical_keys = set()
    for v in variants:
        r = outcome[(v.key, v.digest)]
        if r.status == "ok" and not (r.module and r.description):
            r = pool.Result(v.key, "fail", message="the translator answered, but its output is not there")
        if r.status == "ok":
            if same_as_base(base, v.key, r.module, r.description):
                identical += 1
                identical_keys.add((v.key, v.digest))
            else:
                entries.append(cache_pack.Entry(v.key, r.module, r.description, answers=v.digest))
        else:
            entries.append(cache_pack.Entry(v.key, None, translate.failure_text({"status": r.status,
                                                                                 "message": r.message}),
                                            answers=v.digest, failed=True))
    return entries, identical, identical_keys


# --- the step -------------------------------------------------------------------------------------------------------

class Ledger(Step):
    id = "ledger"
    needs = ("translate",)      # the base pack and the split, and through it the translator and the RomFS

    def key(self, ctx: Context) -> str:
        return fingerprint(f"ledger-{FORMAT}", f"translator={check.translator_identity(ctx)}",
                           f"optimizer={check.XLATE_OPTIMIZER}", f"schema={cache_pack.DESC_SCHEMA}",
                           f"ledger={ledger_digest(ctx.dest / LEDGER_FILE)}", HERE / "ledger.py",
                           SHADER_TOOLS / "corpus.py")

    def current(self, ctx: Context, record: dict) -> bool:
        path = ctx.dest / translate.SHADERS / cache_pack.VARIANTS_FILE
        map_path = ctx.dest / translate.SHADERS / VARIANT_MAP_FILE
        try:
            pack_current = path.stat().st_size == record["pack_bytes"] if record["variants"] else not path.exists()
            return pack_current and map_path.stat().st_size == record["map_bytes"]
        except (OSError, KeyError, TypeError):
            return False

    def verify(self, ctx: Context, record: dict) -> bool:
        """The pack's entries and the sorted digest map against their rows and the record."""
        folder = ctx.dest / translate.SHADERS
        path = folder / cache_pack.VARIANTS_FILE
        map_path = folder / VARIANT_MAP_FILE
        try:
            base = cache_pack.open_pack(folder / cache_pack.BASE_FILE, role=cache_pack.BASE)
            if record["variants"]:
                if cache_pack.verify_pack(path, role=cache_pack.VARIANTS):
                    return False
                pack = cache_pack.open_pack(path, role=cache_pack.VARIANTS)
                if pack.translator != base.translator or len(pack.rows) != record["variants"] or \
                        not all(base.find(r.key) is not None for r in pack.rows):
                    return False
            elif path.exists():
                return False
            parsed = read_variant_map(map_path)
            if not parsed or parsed[0] != record["ledger_sha256"] or map_path.stat().st_size != record["map_bytes"]:
                return False
            pack = cache_pack.open_pack(path, role=cache_pack.VARIANTS) if record["variants"] else None
            for (key, _), target in parsed[1].items():
                if base.find(key) is None:
                    return False
                if target != cache_pack.ZERO_DIGEST and (pack is None or pack.find(key, target) is None):
                    return False
            return True
        except (OSError, KeyError, TypeError, ValueError, cache_pack.PackError):
            return False

    def run(self, ctx: Context) -> dict:
        rep = ctx.reporter
        if not ctx.record("translate"):
            raise PrepareError("the translate step has not run, so there is no shader pack to add variants to")
        identity = check.translator_identity(ctx)
        folder = ctx.dest / translate.SHADERS
        pack = folder / cache_pack.VARIANTS_FILE
        map_path = folder / VARIANT_MAP_FILE
        for stale in (pack, pack.with_name(pack.name + ".tmp"), map_path, map_path.with_name(map_path.name + ".tmp")):
            stale.unlink(missing_ok=True)       # write-once: a step that runs again deletes its own pack first
        scratch = ctx.dest / check.TEMP / SCRATCH
        shutil.rmtree(scratch, ignore_errors=True)

        rows, digest = read_ledger(ctx.dest / LEDGER_FILE)
        try:
            base = cache_pack.open_pack(folder / cache_pack.BASE_FILE, role=cache_pack.BASE)
        except (OSError, cache_pack.PackError) as e:
            raise PrepareError(f"the base shader pack cannot be read ({e}). Run oot2 prepare again so that the "
                               "translate step writes it.") from None
        if base.translator != identity:
            raise PrepareError("the base shader pack was written by another translator than the one in use. Run oot2 "
                               "prepare again so that the translate step runs.")
        if digest == "none":
            rep.say(f"ledger: no {LEDGER_FILE} in the output folder yet; the game writes it when a run ends, and the "
                    "next prepare translates what it recorded")
        jobs = translate.load_jobs(folder / translate.SPLIT) if rows else {}
        variants, counts, destinations = plan(rows, jobs, base)
        for name, text in (("unknown_programs", "name a program the split does not hold"),
                           ("other_stage", "name another stage than their program's"),
                           ("base_failed", "name a program the base pack holds as a failure record")):
            if counts[name]:
                rep.warning(f"ledger: {counts[name]:,} row(s) {text}, so they are skipped; the first: "
                            f"{', '.join(counts['problems'][name])}")
        if rows:
            rep.say(f"ledger: {len(rows):,} rows of {counts['programs']:,} programs; {counts['default_rows']:,} are "
                    f"the default answers for everything their module reads, {counts['variant_rows']:,} ask for "
                    f"{len(variants):,} variants ({counts['merged_rows']:,} repeat another row's)")

        seconds = 0.0
        translated = identical = 0
        entries: list = []
        failed: list = []
        written = {"bytes": 0, "index_bytes": 0, "module_bytes": 0, "description_bytes": 0}
        workers = limit = 0
        if variants:
            path = check.translator(ctx)
            workers, limit, why = translate.sizing(ctx)
            command = check.translator_command(path)
            rep.say(f"ledger: translating {len(variants):,} variants on {workers} translator processes ({why})")
            t0 = time.monotonic()
            outcome = translate_variants(command, variants, scratch, workers, limit, translate.TIMEOUT)
            retry = [v for v in variants if outcome[(v.key, v.digest)].status in ("crash", "timeout")]
            if retry:
                rep.say(f"ledger: {len(retry):,} variant(s) got no answer; each is tried again alone, within "
                        f"{translate.RETRY_TIMEOUT:.0f} s")
                outcome.update(translate_variants(command, retry, scratch, 1, limit, translate.RETRY_TIMEOUT))
            seconds = time.monotonic() - t0
            failed = [(v, outcome[(v.key, v.digest)]) for v in variants if outcome[(v.key, v.digest)].status != "ok"]
            translated = len(variants) - len(failed)
            entries, identical, identical_keys = entries_of(variants, outcome, scratch, base)
            destinations = {
                source: cache_pack.ZERO_DIGEST if (source[0], target) in identical_keys else target
                for source, target in destinations.items()
            }
            try:
                if entries:
                    written = cache_pack.write_pack(pack, entries, translator=identity, role=cache_pack.VARIANTS)
            except (OSError, cache_pack.PackError) as e:
                raise PrepareError(f"the variants pack could not be written ({e}). Run oot2 prepare again.") from None
            shutil.rmtree(scratch, ignore_errors=True)
            if failed:
                named = "; ".join(f"{v.key} {cache_pack.digest_text(v.digest)[:12]}: "
                                  f"{translate.failure_text({'status': r.status, 'message': r.message})}"
                                  for v, r in failed[:NAMED])
                rep.warning(f"ledger: {len(failed):,} of {len(variants):,} variants did not translate "
                            f"({translate.status_counts(r.status for _, r in failed)}); each is packed as a failure "
                            f"record, and the game stops with its reason if it ever asks for one: {named}"
                            f"{'; ...' if len(failed) > NAMED else ''}")
            rep.say(f"ledger: {translated:,} of {len(variants):,} variants translated in {seconds:.0f} s; "
                    f"{identical:,} came out as the base module and description and are not packed; "
                    + (f"pack {translate.mib(written['bytes'])} written" if entries else "no variants pack is written"))
        elif rows:
            rep.say("ledger: no row asks for a variant, so no variants pack is written")
        map_data = variant_map_bytes(destinations, digest)
        map_tmp = map_path.with_name(map_path.name + ".tmp")
        try:
            map_tmp.write_bytes(map_data)
            map_tmp.replace(map_path)
        except OSError as e:
            map_tmp.unlink(missing_ok=True)
            raise PrepareError(f"the shader variant map could not be written ({e}). Run oot2 prepare again.") from None
        return {
            "bytes": written["bytes"],
            "pack_bytes": written["bytes"],
            "map_bytes": len(map_data),
            "map_rows": len(destinations),
            "index_bytes": written["index_bytes"],
            "module_bytes": written["module_bytes"],
            "description_bytes": written["description_bytes"],
            "ledger_sha256": digest,
            "ledger_rows": counts["rows"],
            "ledger_programs": counts["programs"],
            "unknown_programs": counts["unknown_programs"],
            "other_stage": counts["other_stage"],
            "base_failed": counts["base_failed"],
            "default_rows": counts["default_rows"],
            "variant_rows": counts["variant_rows"],
            "merged_rows": counts["merged_rows"],
            "candidates": len(variants),
            "identical": identical,
            "variants": len(entries),
            "translated": translated,
            "failed": dict(sorted(translate.status_tally(r.status for _, r in failed).items())),
            "failed_keys": [f"{v.key} {cache_pack.digest_text(v.digest)}" for v, _ in failed[:100]],
            "workers": workers,
            "memory_limit": limit,
            "translate_seconds": round(seconds, 1),
            "translator": {"sha256": identity},
        }
