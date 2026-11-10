# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The store step of `oot2 prepare`: where the game keeps its named state values, so that a run can wait for one.

The game keeps everything its logic needs to remember in one global store of named values (a flag, a counter, a scene
name). A value is found by a 32-bit hash of its name (MurmurHash3, seed 0) in an open-addressing table that belongs to its
type, and the tables hang off one object that a global pointer cell leads to. This step finds that cell and the layout of
the per-type tables by reading the player's own main module, and writes what the runtime needs to read one value from a
running game: oot2-store.txt in the output folder. The runtime only ever reads the store (runtime/hos/store.c); the game's
own setters queue their writes, so nothing here writes one.

Off unless the player gives a names file (`oot2 prepare --names FILE`). The file is the player's own, kept outside the
repository: a line `role = name` for each value the player wants to wait for. The role is ours (FIXED_ROLES, or
`<kind>.<label>` for a value of the player's own choosing); the name is the game's. Only the 32-bit hash of each name is
written to oot2-store.txt and to the state file, never the name. Nothing in the repository holds a game name, offset or
hash.

The structural match. No offset or name is known to this code. A lookup of a value is compiled inline, over a hundred
times over, into the game's functions: it loads the store's object through the pointer cell, loads a table pointer and a
capacity from the object, divides the key by the capacity to place the probe, finds the table cell, loads the entry
number from the cell, checks it against the entry count, multiplies it by the entry size, adds the entry array, and reads
the value from a fixed place in the entry. `find_store` follows that chain on every unsigned 32-bit division in the module
(`udiv`) with a small symbolic executor over the machine code, and keeps the chains that complete. The pointer cell the
most chains start from is the store's; every number of the layout is read from the instructions of those chains. Several
chains of different tables start from the one object, one per type, and each gives one `Module` of the description. The
match asks for at least MIN_CHAINS chains and for MARGIN times the next cell's, and otherwise refuses by name.

oot2-store.txt (read by runtime/hos/store.c and by `parse_description`):

    oot2-store 1
    build <64 hex digits>        the main module's build id, as the module header holds it
    cell <image offset>          the pointer cell, an offset in the main module's image
    module <table> <capacity> <count> <entries> <stride> <value> <class> <shift> <index>
    role <role> <kind> <8 hex digits>

The cell holds the address of a variable, and the variable the address of the object (two reads). A `module` line gives
displacements from the object, in decimal: a pointer to the table, the table's capacity (a 32-bit count of cells), the
entry count (32-bit), a pointer to the entry array, the entry size, and the displacement of the value inside an entry,
then the value's class (b byte, h half word, w word, x double word, s single float, d double float), the cell size as a
shift (a cell is 1 << shift bytes, its key at 0), and the displacement of the entry number inside a cell. A `role` line
gives the role, its kind (bool, int, uint, float, string) and the hash of its name. A line starting with # is a comment.
"""

from __future__ import annotations

import bisect
import collections
import hashlib
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .runner import Context, PrepareError, Step, fingerprint

TOOLS = Path(__file__).resolve().parent.parent

FILE = "oot2-store.txt"
FORMAT = 1
EXEFS = "exefs"

# The value kinds a role can have, and the classes of load that read each.
KINDS = ("bool", "int", "uint", "float", "string")
KIND_CLASSES = {"bool": "b", "int": "w", "uint": "w", "float": "s", "string": "x"}
# The roles this repository defines, with their kinds. The player's names file binds each to a game name.
FIXED_ROLES = {
    "scene": "string",        # the scene now loaded
    "next_scene": "string",   # the scene a change has asked for
    "entrance": "string",     # the entrance a change has asked for
    "room": "int",
    "age": "bool",            # the player's age, as the game keeps it: 1 for the younger
    "health": "int",
    "max_health": "int",
    "boss": "bool",           # a boss fight is on
    "paused": "bool",
    "loading": "bool",
    "menu": "bool",
}
LABEL = re.compile(r"[a-z][a-z0-9_]{0,31}\Z")
MAX_NAME = 255

# The match
MIN_CHAINS = 8        # chains a cell must start for it to be the store's
MARGIN = 3            # ... and this many times the next cell's
PREFIX_WORDS = 48     # instructions read before a division when no function start is near, to find its operands
MIN_ORPHANS = 2      # functions a table must be met in when the pointer cell does not lead to it
MAX_FUNCTION_WORDS = 6000   # the longest run from a function start that is taken for one function
PATH_STEPS = 160      # instructions followed after a division, along each path
PATHS_MAX = 48        # paths followed after one division


class NamesError(Exception):
    """A line of the names file that cannot be taken: the message says which and why."""

    def __init__(self, line: int, what: str):
        self.line, self.what = line, what
        super().__init__(f"line {line}: {what}")


class MatchError(Exception):
    pass


@dataclass(frozen=True)
class Module:
    """One per-type table, as the instructions of a lookup read it. Displacements are from the store's object."""
    table: int
    capacity: int
    count: int
    entries: int
    stride: int
    value: int
    cls: str
    shift: int
    index: int

    def fields(self) -> tuple:
        return (self.table, self.capacity, self.count, self.entries, self.stride, self.value, self.cls, self.shift,
                self.index)


@dataclass(frozen=True)
class Role:
    role: str
    kind: str
    hash: int


@dataclass(frozen=True)
class Description:
    build: str
    cell: int
    modules: tuple[Module, ...]
    roles: tuple[Role, ...]


# ---- the hash
def murmur3_32(data: bytes, seed: int = 0) -> int:
    """MurmurHash3, x86 32-bit variant."""
    c1, c2 = 0xCC9E2D51, 0x1B873593
    h = seed & 0xFFFFFFFF
    whole = len(data) // 4
    for (k,) in struct.iter_unpack("<I", data[:whole * 4]):
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF
    tail, k = data[whole * 4:], 0
    for i in range(len(tail) - 1, -1, -1):
        k = (k << 8) | tail[i]
    if tail:
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
    h ^= len(data)
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


# ---- the names file
def role_kind(role: str) -> str | None:
    """The kind of a role, or None when it is neither a fixed role nor `<kind>.<label>`."""
    if role in FIXED_ROLES:
        return FIXED_ROLES[role]
    kind, dot, label = role.partition(".")
    if dot and kind in KINDS and LABEL.match(label):
        return kind
    return None


def parse_names(raw: bytes) -> list[tuple[str, str, int]]:
    """[(role, kind, hash)] for the lines of a names file. Raises NamesError. The names are hashed here and go no further."""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as e:
        line = raw[:e.start].count(b"\n") + 1
        raise NamesError(line, "a byte that is not ASCII; the names file is plain text") from None
    out: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        role, eq, name = (part.strip() for part in line.partition("="))
        if not eq:
            raise NamesError(number, "expected `role = name`")
        role = role.lower()
        kind = role_kind(role)
        if kind is None:
            raise NamesError(number, f"unknown role \"{role[:40]}\" (the fixed roles are {', '.join(FIXED_ROLES)}; "
                                     f"your own are <kind>.<label> with a kind of {', '.join(KINDS)})")
        if role in seen:
            raise NamesError(number, f"the role {role} is given twice")
        if not name:
            raise NamesError(number, f"the role {role} has no name")
        if len(name) > MAX_NAME or any(c.isspace() or not c.isprintable() for c in name):
            raise NamesError(number, f"the name of {role} is empty, longer than {MAX_NAME} characters or holds a space "
                                     "or a control character")
        h = murmur3_32(name.encode("ascii"))
        if h == 0:
            raise NamesError(number, f"the name of {role} hashes to 0, which the store uses for an empty cell")
        seen.add(role)
        out.append((role, kind, h))
    return out


# ---- the structural match
# A small symbolic executor over AArch64 machine code. A register holds nothing known (it is absent from the map) or one of
#   ("page", address)                         the page address an adrp formed
#   ("cell", address)                         the word a pointer cell holds, loaded through a page and a displacement
#   ("obj", cell)                             the word that word points at: the store's object, if the cell is its
#   ("f4", cell, disp) / ("f8", cell, disp)   a 32-bit or 64-bit field of that object
#   ("imm", n)                                a constant
#   ("cellptr", cell, table, shift, cap)      a table cell: the table plus an index shifted left by `shift`
#   ("idx", cell, table, shift, cap, index)   the entry number a table cell holds, `index` bytes into the cell
#   ("prod", idx, n)                          an entry number times a constant
#   ("entry", chain, off)                     the address of an entry plus `off`; chain = (cell, table, cap, shift, index,
#                                             entries, n), n the entry size
# It follows both ways of a conditional branch, one path at a time, from each unsigned 32-bit division whose divisor is a
# field of an object a pointer cell leads to.
LOADS = ("ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ldur", "ldurb", "ldurh", "ldursb", "ldursh", "ldursw")
BYTE_LOADS, HALF_LOADS = ("ldrb", "ldrsb", "ldurb", "ldursb"), ("ldrh", "ldrsh", "ldurh", "ldursh")
COND_BRANCH = ("cbz", "cbnz", "tbz", "tbnz")
ENDS = ("ret", "retaa", "retab", "br", "braa", "brab", "braaz", "brabz", "eret", "eretaa", "eretab", "brk", "hlt", "udf")
CALLS = ("bl", "blr", "blraa", "blrab", "blraaz", "blrabz")
LSL = 1   # capstone's ARM64_SFT_LSL
ZR = 31
# Instructions that write no general register, whatever capstone's alias for them says
WRITES_NONE = ("cmp", "cmn", "tst", "ccmp", "ccmn", "nop", "b", "cbz", "cbnz", "tbz", "tbnz", "str", "strb", "strh",
               "stur", "sturb", "sturh", "stp", "stnp", "stlr", "stlrb", "stlrh")


class _Ins:
    __slots__ = ("addr", "mn", "ops", "writes", "rc")

    def __init__(self, addr, mn, ops, writes, rc):
        self.addr, self.mn, self.ops, self.writes, self.rc = addr, mn, ops, writes, rc   # rc: the first register's class


def _regnum(insn, reg: int) -> int | None:
    """0 to 30 for x0..x30 and w0..w30, ZR for the zero register and sp, None for any other register."""
    name = insn.reg_name(reg)
    if name in ("xzr", "wzr", "sp", "wsp"):
        return ZR
    if len(name) > 1 and name[0] in "xw" and name[1:].isdigit():
        return int(name[1:])
    return None


def _decode(md, image, addr: int) -> _Ins | None:
    from capstone.arm64 import ARM64_OP_IMM, ARM64_OP_MEM, ARM64_OP_REG
    insn = next(md.disasm(bytes(image[addr:addr + 4]), addr), None)
    if insn is None:
        return None
    ops = []
    for o in insn.operands:
        if o.type == ARM64_OP_REG:
            ops.append(("r", _regnum(insn, o.reg), o.shift.type, o.shift.value))
        elif o.type == ARM64_OP_IMM:
            ops.append(("i", o.imm))
        elif o.type == ARM64_OP_MEM:
            ops.append(("m", _regnum(insn, o.mem.base) if o.mem.base else None,
                        _regnum(insn, o.mem.index) if o.mem.index else None, o.mem.disp))
        else:
            ops.append(("o",))
    _, written = insn.regs_access()
    writes = set()
    if insn.mnemonic not in WRITES_NONE and not insn.mnemonic.startswith("b."):
        writes = {n for n in (_regnum(insn, r) for r in written) if n is not None and n != ZR}
    rc = insn.reg_name(insn.operands[0].reg)[:1] if insn.operands and insn.operands[0].type == ARM64_OP_REG else ""
    return _Ins(addr, insn.mnemonic, ops, writes, rc)


def _load_class(ins: _Ins) -> str:
    """b, h, w, x, s or d: how much a load reads and in what kind of register."""
    if ins.mn in BYTE_LOADS:
        return "b"
    if ins.mn in HALF_LOADS:
        return "h"
    if ins.mn in ("ldrsw", "ldursw"):
        return "w"
    return ins.rc if ins.rc in ("x", "w", "s", "d") else ""


def _exec(ins: _Ins, regs: dict, anchor: tuple) -> tuple | None:
    """Apply one instruction to the register map. Returns the Module a value load completes, when it does.

    anchor = (cell, capacity displacement, table displacements); regs["count"] is the displacement of the field the
    entry number was last compared with.
    """
    mn, ops = ins.mn, ins.ops
    dest = ops[0][1] if ops and ops[0][0] == "r" else None
    val = None
    done = None

    def value(i: int):
        return regs.get(ops[i][1]) if i < len(ops) and ops[i][0] == "r" else None

    if mn == "adrp" and len(ops) == 2 and ops[1][0] == "i":
        val = ("page", ops[1][1])
    elif mn in LOADS and len(ops) == 2 and ops[1][0] == "m":
        base, index, disp = ops[1][1], ops[1][2], ops[1][3]
        src = regs.get(base) if base is not None else None
        cls = _load_class(ins)
        if index is None and src is not None:
            if src[0] == "page" and cls == "x":
                val = ("cell", src[1] + disp)
            elif src[0] == "cell" and cls == "x" and disp == 0:
                val = ("obj", src[1])
            elif src[0] == "obj" and cls in ("x", "w"):
                val = ("f8" if cls == "x" else "f4", src[1], disp)
            elif src[0] == "cellptr" and cls == "w":
                val = ("idx", src[1], src[2], src[3], src[4], disp)
            elif src[0] == "entry" and regs.get("count") is not None:
                cell, table, cap, shift, idx, entries, n = src[1]
                done = (cell, Module(table, cap, regs["count"], entries, n, src[2] + disp, cls, shift, idx))
    elif mn == "mov" and len(ops) == 2:
        val = ("imm", ops[1][1]) if ops[1][0] == "i" else value(1)
    elif mn in ("sxtw", "uxtw", "sxth", "uxth", "sxtb", "uxtb") and len(ops) == 2:
        val = value(1)
    elif mn == "csel" and len(ops) >= 3:
        # the bounded entry number: the entry number, or zero
        a, b = ops[1][1] if ops[1][0] == "r" else None, ops[2][1] if ops[2][0] == "r" else None
        if b == ZR:
            val = regs.get(a)
        elif a == ZR:
            val = regs.get(b)
    elif mn in ("cmp", "subs") and len(ops) >= 2:
        a, b = (value(len(ops) - 2), value(len(ops) - 1))
        for x, y in ((a, b), (b, a)):
            if x is not None and y is not None and x[0] == "idx" and y[0] == "f4" and y[1] == x[1]:
                regs["count"] = y[2]
    elif mn == "add" and len(ops) == 3 and ops[1][0] == "r" and ops[2][0] == "i":
        a = value(1)
        if a is not None and a[0] == "entry":
            val = ("entry", a[1], a[2] + ops[2][1])
    elif mn == "add" and len(ops) == 3 and ops[1][0] == "r" and ops[2][0] == "r":
        a, b = value(1), value(2)
        shift = ops[2][3] if ops[2][2] == LSL else 0
        if a is not None and a[0] == "f8" and a[1] == anchor[0] and a[2] in anchor[2] and (b is None or b[0] != "idx"):
            val = ("cellptr", a[1], a[2], shift, anchor[1])
        elif a is not None and a[0] == "f8" and b is not None and b[0] == "idx":
            val = ("entry", (b[1], b[2], b[4], b[3], b[5], a[2], 1 << shift), 0)
        elif a is not None and a[0] == "f8" and b is not None and b[0] == "prod":
            i = b[1]
            val = ("entry", (i[1], i[2], i[4], i[3], i[5], a[2], b[2]), 0)
        elif b is not None and b[0] == "f8" and a is not None and a[0] == "prod":
            i = a[1]
            val = ("entry", (i[1], i[2], i[4], i[3], i[5], b[2], a[2]), 0)
    elif mn in ("mul", "smull", "umull") and len(ops) == 3:
        a, b = value(1), value(2)
        for x, y in ((a, b), (b, a)):
            if x is not None and y is not None and x[0] == "idx" and y[0] == "imm" and y[1] > 0:
                val = ("prod", x, y[1])
    elif mn in ("madd", "smaddl", "umaddl") and len(ops) == 4:
        a, b, c = value(1), value(2), value(3)
        for x, y in ((a, b), (b, a)):
            if x is not None and y is not None and x[0] == "idx" and y[0] == "imm" and y[1] > 0:
                if c is not None and c[0] == "f8":
                    val = ("entry", (x[1], x[2], x[4], x[3], x[5], c[2], y[1]), 0)
                elif ops[3][1] == ZR:
                    val = ("prod", x, y[1])
    elif mn == "lsl" and len(ops) == 3 and ops[2][0] == "i":
        a = value(1)
        if a is not None and a[0] == "idx" and 0 <= ops[2][1] < 32:
            val = ("prod", a, 1 << ops[2][1])
    elif mn in ("ubfiz", "sbfiz") and len(ops) == 4 and ops[2][0] == "i":
        a = value(1)
        if a is not None and a[0] == "idx" and 0 <= ops[2][1] < 32:
            val = ("prod", a, 1 << ops[2][1])
    elif mn in CALLS:
        for r in range(19):
            regs.pop(r, None)
        regs.pop(30, None)
        return None
    for r in ins.writes:
        regs.pop(r, None)
    if dest is not None and val is not None and dest != ZR:
        regs[dest] = val
    return done


def _follow(at, regs: dict, site: int) -> set:
    """The (cell, Module) pairs the lookups that follow the division at `site` complete, given the registers before it."""
    div = at(site)
    if div is None or len(div.ops) != 3 or div.ops[2][0] != "r":
        return set()
    cap = regs.get(div.ops[2][1])
    if cap is None or cap[0] != "f4":
        return set()
    cell = cap[1]
    tables = frozenset(v[2] for v in regs.values() if isinstance(v, tuple) and v[0] == "f8" and v[1] == cell)
    if not tables:
        return set()
    anchor = (cell, cap[2], tables)
    start = dict(regs)
    start.pop("count", None)
    found: set = set()
    paths = [(site + 4, start, frozenset())]
    started = 0
    while paths and started < PATHS_MAX:
        pc, state, seen = paths.pop()
        started += 1
        for _ in range(PATH_STEPS):
            if pc in seen:
                break
            seen = seen | {pc}
            ins = at(pc)
            if ins is None or ins.mn in ENDS:
                break
            if ins.mn == "b" and ins.ops and ins.ops[0][0] == "i":
                pc = ins.ops[0][1]
                continue
            if (ins.mn.startswith("b.") or ins.mn in COND_BRANCH) and ins.ops and ins.ops[-1][0] == "i":
                paths.append((ins.ops[-1][1], dict(state), seen))
                pc += 4
                continue
            done = _exec(ins, state, anchor)
            if done is not None:
                found.add(done)
            pc += 4
    return found


def _function_starts(words) -> list[int]:
    """The addresses of the instructions that open a function: the pointer-authenticating prologue of the compiler."""
    return [i * 4 for i, w in enumerate(words) if w == 0xD503237F]   # pacibsp


def find_store(image, text_size: int) -> tuple[int, tuple[Module, ...], int, int]:
    """(cell, modules, chains, runner-up) for the main module's image; MatchError when no cell is the store's.

    `chains` counts the divisions whose lookup completes from the cell, `runner-up` the next cell's count. Of the chains of
    one table and one class of read, the layout most of them agree on is kept (the smaller of equals). The registers before
    a division are those of a straight run from the function's first instruction (a function keeps the store's object in a
    register across its lookups), and when no function start lies within MAX_FUNCTION_WORDS before it, those of the
    PREFIX_WORDS before it.
    """
    from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs
    md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    md.detail = True
    words = memoryview(bytes(image[:text_size - text_size % 4])).cast("I")
    sites = [i * 4 for i, w in enumerate(words) if w & 0xFFE0FC00 == 0x1AC00800]   # udiv wd, wn, wm
    starts = _function_starts(words)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for site in sites:
        k = bisect.bisect_right(starts, site) - 1
        first = starts[k] if k >= 0 and site - starts[k] <= 4 * MAX_FUNCTION_WORDS else max(0, site - 4 * PREFIX_WORDS)
        groups[first].append(site)
    per_cell: collections.Counter = collections.Counter()
    layouts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    orphans: collections.Counter = collections.Counter()
    for first, group in groups.items():
        cache: dict = {}

        def at(a: int) -> _Ins | None:
            if a not in cache:
                cache[a] = _decode(md, image, a) if 0 <= a < text_size else None
            return cache[a]

        regs: dict = {n: ("obj", ("arg", n)) for n in range(8)}   # an argument may be the object itself
        snapshots: dict[int, dict] = {}
        want = set(group)
        for a in range(first, max(group) + 4, 4):
            if a in want:
                snapshots[a] = dict(regs)
            ins = at(a)
            if ins is not None:
                _exec(ins, regs, (None, None, ()))
        for site in group:
            pairs = _follow(at, snapshots[site], site)
            for cell in {c for c, _ in pairs if isinstance(c, int)}:
                per_cell[cell] += 1
            for cell, module in pairs:
                if isinstance(cell, int):
                    layouts[cell][module] += 1
            for module in {m for c, m in pairs if not isinstance(c, int)}:
                orphans[module] += 1
    ranked = per_cell.most_common(2)
    if not ranked:
        raise MatchError("no division in the module belongs to a lookup of the shape the store's lookups have")
    best, n = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if n < MIN_CHAINS or n < MARGIN * second:
        raise MatchError(f"no pointer cell stands out as the store's: the best starts {n} complete lookups and the next "
                         f"{second} (at least {MIN_CHAINS}, and {MARGIN} times the next, are needed)")
    found = _dominant(layouts[best], set())
    return best, tuple(sorted(found + _more_tables(found, orphans), key=Module.fields)), n, second


def _dominant(seen: collections.Counter, have: set) -> list[Module]:
    """Of the layouts met for one table and one class of read, the one met most often; the smaller of equals, so that the
    answer does not depend on the order the lookups were visited in."""
    chosen: dict[tuple[int, str], Module] = {}
    for module, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0].fields())):
        key = (module.table, module.cls)
        if key not in have and key not in chosen:
            chosen[key] = module
    return list(chosen.values())


def _more_tables(found: list[Module], orphans: collections.Counter) -> list[Module]:
    """The tables of the store that no function reaches through the pointer cell: the lookups of one type are also compiled
    into functions that are handed the object. The tables sit one after the other in the object, each with the same
    arrangement of its own fields, so a lookup of an object that is handed over counts when its arrangement is that of the
    tables found, and its table lies a whole number of tables further on, and the shape was met in MIN_ORPHANS functions."""
    if len(found) < 3:
        return []
    arrangements = collections.Counter((m.capacity - m.table, m.entries - m.table, m.count - m.table, m.shift, m.index)
                                       for m in found)
    arrangement = max(arrangements.items(), key=lambda kv: (kv[1], kv[0]))[0]
    spacing = 0
    for m in found:
        spacing = math.gcd(spacing, m.table - found[0].table)
    if spacing == 0:
        return []
    ours = [m for m in orphans if m.count - m.table == arrangement[2] and m.capacity - m.table == arrangement[0]
            and m.entries - m.table == arrangement[1] and (m.shift, m.index) == arrangement[3:]
            and (m.table - found[0].table) % spacing == 0]
    return _dominant(collections.Counter({m: orphans[m] for m in ours if orphans[m] >= MIN_ORPHANS}),
                     {(m.table, m.cls) for m in found})


# ---- the description
def render_description(d: Description) -> str:
    lines = [f"oot2-store {FORMAT}", f"build {d.build}", f"cell {d.cell}"]
    for m in d.modules:
        lines.append("module " + " ".join(str(x) for x in m.fields()))
    for r in d.roles:
        lines.append(f"role {r.role} {r.kind} {r.hash:08x}")
    return "\n".join(lines) + "\n"


def parse_description(text: str) -> Description:
    """The inverse of render_description. ValueError, naming the line, for a file that is not one."""
    build, cell, modules, roles = None, None, [], []
    lines = text.splitlines()
    if not lines or lines[0].split() != ["oot2-store", str(FORMAT)]:
        raise ValueError("line 1: not an oot2-store file of this format")
    for number, line in enumerate(lines[1:], 2):
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        try:
            if words[0] == "build" and len(words) == 2 and re.fullmatch(r"[0-9a-f]{64}", words[1]):
                build = words[1]
            elif words[0] == "cell" and len(words) == 2:
                cell = int(words[1])
            elif words[0] == "module" and len(words) == 10 and words[7] in "bhwxsd" and len(words[7]) == 1:
                n = [int(x) for x in words[1:7]] + [words[7]] + [int(x) for x in words[8:10]]
                modules.append(Module(*n))
            elif words[0] == "role" and len(words) == 4 and words[2] in KINDS and re.fullmatch(r"[0-9a-f]{8}", words[3]):
                roles.append(Role(words[1], words[2], int(words[3], 16)))
            else:
                raise ValueError("not a line of the format")
        except ValueError:
            raise ValueError(f"line {number}: not a line of the format") from None
    if build is None or cell is None:
        raise ValueError("no build or no cell line")
    return Description(build, cell, tuple(modules), tuple(roles))


def build_id(raw: bytes) -> str:
    """The build id field of an NSO header, all 32 bytes of it, as 64 hex digits."""
    if len(raw) < 0x60 or raw[:4] != b"NSO0":
        raise PrepareError("the main module is not an NSO0 file")
    return raw[0x40:0x60].hex()


# ---- the step
class Store(Step):
    id = "store"
    needs = ("modules",)   # which writes exefs/

    def names(self, ctx: Context) -> Path:
        return Path(ctx.options.names).resolve()

    def key(self, ctx: Context) -> str:
        return fingerprint("store-1", self.names(ctx), Path(__file__))

    def current(self, ctx: Context, record: dict) -> bool:
        path = ctx.dest / FILE
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record.get("file_sha256")

    def run(self, ctx: Context) -> dict:
        rep = ctx.reporter
        names = self.names(ctx)
        try:
            roles = parse_names(names.read_bytes())
        except OSError as e:
            raise PrepareError(f"the names file {names} cannot be read ({e.strerror or e})") from None
        except NamesError as e:
            raise PrepareError(f"the names file {names}, {e}") from None
        main = ctx.dest / EXEFS / "main"
        try:
            raw = main.read_bytes()
        except OSError as e:
            raise PrepareError(f"{main} cannot be read ({e.strerror or e}); the modules step writes it") from None
        from lift.nso import Nso, NsoError   # tools/lift/nso.py
        try:
            nso = Nso(main)
        except NsoError as e:
            raise PrepareError(f"{main}: {e}") from None
        try:
            cell, modules, chains, runner_up = find_store(nso.image, nso.text[1])
        except MatchError as e:
            raise PrepareError(f"the value store was not found in the main module: {e}. The game may differ from the "
                               "build this match was written against; prepare without --names to skip this step.") from None
        kinds = {r[1] for r in roles}
        missing = sorted(k for k in kinds if not any(m.cls == KIND_CLASSES[k] for m in modules))
        if missing:
            raise PrepareError(f"the store was found ({len(modules)} tables) but none is read as {', '.join(missing)}, "
                               "which a role in the names file needs")
        desc = Description(build_id(raw), cell, modules, tuple(Role(*r) for r in roles))
        text = render_description(desc)
        path = ctx.dest / FILE
        tmp = path.with_name(FILE + ".tmp")
        tmp.write_text(text, encoding="ascii", newline="\n")
        os.replace(tmp, path)
        rep.say(f"store: the pointer cell starts {chains} of the module's lookups (the next cell's, {runner_up}); "
                f"{len(modules)} tables; {len(roles)} roles, written to {path}")
        return {"chains": chains, "runner_up": runner_up, "tables": len(modules), "roles": len(roles),
                "build": desc.build, "cell": cell, "role_hashes": {r.role: f"{r.hash:08x}" for r in desc.roles},
                "file_sha256": hashlib.sha256(text.encode("ascii")).hexdigest()}
