# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Synthetic pieces for the tests of the value store: a plain NSO0 module assembled byte by byte, the few AArch64
instructions the structural match reads, and a model of a store (tables, entries, strings) laid out in a module's data.
Everything here is made up by the tests: no number, name or hash comes from a game.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

PAGE = 0x1000
TEXT, RODATA, DATA, BSS = 0x1000, 0x1000, 0x2000, 0x4000
IMAGE = TEXT + RODATA + DATA + BSS   # 0x8000: the size the test layouts give main
DATA_OFF = TEXT + RODATA
MOD0_AT = 0x100
CODE_AT = 0x200


# ---- the module
def make_nso(code: bytes = b"", data: bytes = b"", build: bytes = bytes(32), text: int = TEXT) -> bytes:
    """A plain NSO0 with a text segment of `text` bytes, a RODATA, a DATA and a BSS of the sizes above; `code` goes at CODE_AT
    and `data` 0x100 bytes into DATA."""
    assert len(build) == 32 and CODE_AT + len(code) <= text and 0x100 + len(data) <= DATA and text % PAGE == 0
    data_off = text + RODATA
    image = data_off + DATA + BSS
    body = bytearray(text)
    body[CODE_AT:CODE_AT + len(code)] = code
    struct.pack_into("<I", body, 4, MOD0_AT)
    body[MOD0_AT:MOD0_AT + 28] = b"MOD0" + struct.pack("<6i", data_off - MOD0_AT, data_off + DATA - MOD0_AT,
                                                         image - MOD0_AT, 0, 0, 0)
    ro = bytearray(RODATA)   # a one-byte string table and a symbol table of its null symbol
    dat = bytearray(DATA)
    dyn = struct.pack("<8Q", 6, text + 0x10, 5, text, 10, 1, 0, 0)   # DT_SYMTAB, DT_STRTAB, DT_STRSZ, DT_NULL
    dat[:len(dyn)] = dyn
    dat[0x100:0x100 + len(data)] = data
    header = bytearray(0x100)
    header[0:4] = b"NSO0"
    pos = 0x100
    for i, (blob, mem) in enumerate(((body, 0), (ro, text), (dat, data_off))):
        struct.pack_into("<III", header, 0x10 + 0x10 * i, pos, mem, len(blob))
        struct.pack_into("<I", header, 0x60 + 4 * i, len(blob))
        pos += len(blob)
    struct.pack_into("<I", header, 0x3C, BSS)
    header[0x40:0x60] = build
    struct.pack_into("<II", header, 0x98, 0x10, 24)   # the dynamic symbol table: 24 bytes at rodata + 0x10
    return bytes(header) + bytes(body) + bytes(ro) + bytes(dat)


# ---- AArch64, the instructions the match reads
def adrp(rd: int, pc: int, target: int) -> int:
    imm = ((target >> 12) - (pc >> 12)) & 0x1FFFFF
    return 0x90000000 | (imm & 3) << 29 | (imm >> 2) << 5 | rd


def ldr_x(rt: int, rn: int, disp: int) -> int:
    assert disp % 8 == 0 and 0 <= disp < 0x8000
    return 0xF9400000 | (disp // 8) << 10 | rn << 5 | rt


def ldr_w(rt: int, rn: int, disp: int) -> int:
    assert disp % 4 == 0 and 0 <= disp < 0x4000
    return 0xB9400000 | (disp // 4) << 10 | rn << 5 | rt


def ldrb(rt: int, rn: int, disp: int) -> int:
    return 0x39400000 | disp << 10 | rn << 5 | rt


def ldr_s(rt: int, rn: int, disp: int) -> int:
    assert disp % 4 == 0 and 0 <= disp < 0x4000
    return 0xBD400000 | (disp // 4) << 10 | rn << 5 | rt


def udiv_w(rd: int, rn: int, rm: int) -> int:
    return 0x1AC00800 | rm << 16 | rn << 5 | rd


def sdiv_w(rd: int, rn: int, rm: int) -> int:
    return 0x1AC00C00 | rm << 16 | rn << 5 | rd


def msub_w(rd: int, rn: int, rm: int, ra: int) -> int:
    return 0x1B008000 | rm << 16 | ra << 10 | rn << 5 | rd


def add_lsl(rd: int, rn: int, rm: int, amount: int) -> int:
    return 0x8B000000 | rm << 16 | amount << 10 | rn << 5 | rd


def cmp_w(rn: int, rm: int) -> int:
    return 0x6B00001F | rm << 16 | rn << 5


def mov_w(rd: int, imm: int) -> int:
    return 0x52800000 | imm << 5 | rd


def csel_lo(rd: int, rn: int, rm: int) -> int:
    return 0x1A800000 | rm << 16 | 3 << 12 | rn << 5 | rd


def smaddl(rd: int, rn: int, rm: int, ra: int) -> int:
    return 0x9B200000 | rm << 16 | ra << 10 | rn << 5 | rd


def beq(here: int, target: int) -> int:
    return 0x54000000 | (((target - here) >> 2) & 0x7FFFF) << 5


PACIBSP, RET, NOP = 0xD503237F, 0xD65F03C0, 0xD503201F
ZR = 31


@dataclass(frozen=True)
class Shape:
    """The displacements one synthetic lookup uses, from the object."""
    table: int
    capacity: int
    count: int
    entries: int
    stride: int
    value: int
    shift: int = 3
    index: int = 4
    load: str = "b"   # b: ldrb, w: ldr w, s: ldr s, x: ldr x


def lookup(fn_at: int, cell_addr: int, sh: Shape, *, flaw: str = "") -> list[int]:
    """The words of one function that looks a value up the way the game's inline lookups do, then returns. `flaw` breaks one
    thing: other_object (the divisor comes from another object), no_table, no_stride, no_count, no_deref, signed_div,
    no_load. The probe's hit is a conditional branch to the block that reads the entry, as in compiled code."""
    pc = fn_at
    words: list[int] = [PACIBSP]
    pc += 4
    page = cell_addr & ~0xFFF
    words += [adrp(9, pc, page), ldr_x(9, 9, cell_addr - page)]
    words.append(ldr_x(0, 9, 0) if flaw != "no_deref" else 0xAA0903E0)   # no_deref: mov x0, x9, the cell's word as the object
    words.append(ldr_x(9, 0, sh.table) if flaw != "no_table" else ldr_x(9, 31, 8))   # sp-relative: not the object
    words.append(ldr_w(10, 0, sh.capacity) if flaw != "other_object" else ldr_w(10, 1, sh.capacity))
    words.append(udiv_w(11, 2, 10) if flaw != "signed_div" else sdiv_w(11, 2, 10))
    words.append(msub_w(11, 11, 10, 2))
    here = fn_at + 4 * len(words)
    block = here + 4 * 3   # the branch, a return, a nop: the entry block follows
    words += [beq(here, block), RET, NOP]
    words.append(add_lsl(9, 9, 12, sh.shift))
    words.append(ldr_w(10, 0, sh.count))
    words.append(ldr_x(11, 0, sh.entries))
    words.append(ldr_w(9, 9, sh.index))
    if flaw != "no_count":
        words.append(cmp_w(9, 10))
    words.append(mov_w(10, sh.stride))
    words.append(csel_lo(9, 9, ZR))
    if flaw != "no_stride":
        words.append(smaddl(9, 9, 10, 11))
    else:
        words.append(add_lsl(9, 9, 11, 0))   # entries + index: no multiplication by the entry size
    if flaw != "no_load":
        load = {"b": ldrb, "w": ldr_w, "s": ldr_s, "x": ldr_x}[sh.load]
        words.append(load(9, 9, sh.value))
    words.append(RET)
    return words


def assemble(words: list[int]) -> bytes:
    return b"".join(struct.pack("<I", w) for w in words)


# ---- a model of the store, laid out in a module's data
@dataclass
class Table:
    """One table of the synthetic store: its fields' displacements in the object, its cells and entries."""
    shape: Shape
    capacity: int
    entries: list[tuple[int, object]] = field(default_factory=list)   # (hash, value) in entry order
    string_room: int = 64


class Store:
    """The cell, the variable, the object, the tables and the entries, as bytes at known offsets of a main module whose base
    is `base`. Offsets are those of the image."""
    CELL, VAR, OBJ, OBJ_SIZE = DATA_OFF + 0x100, DATA_OFF + 0x108, DATA_OFF + 0x200, 0x200

    def __init__(self, base: int, tables: list[Table]):
        self.base, self.tables = base, tables
        self.data = bytearray(DATA - 0x100)   # from DATA_OFF + 0x100
        self.at = DATA_OFF + 0x400            # next free offset for a table, an entry array or a string
        self.entry_off: dict[tuple[int, int], int] = {}   # (table index, hash) -> offset of the entry
        self.string_off: dict[tuple[int, int], int] = {}
        self.table_off: dict[int, int] = {}   # table index -> offset of its cells
        self.put64(self.CELL, base + self.VAR)
        self.put64(self.VAR, base + self.OBJ)
        for ti, t in enumerate(tables):
            self.build(ti, t)

    def alloc(self, n: int) -> int:
        off = (self.at + 7) & ~7
        self.at = off + n
        assert self.at <= DATA_OFF + DATA, "the synthetic store does not fit the module's data"
        return off

    def put(self, off: int, blob: bytes) -> None:
        self.data[off - DATA_OFF - 0x100:off - DATA_OFF - 0x100 + len(blob)] = blob

    def put64(self, off: int, v: int) -> None:
        self.put(off, struct.pack("<Q", v))

    def put32(self, off: int, v: int) -> None:
        self.put(off, struct.pack("<I", v & 0xFFFFFFFF))

    def encode(self, sh: Shape, value) -> bytes:
        return {"b": lambda v: bytes([1 if v else 0]), "w": lambda v: struct.pack("<I", v & 0xFFFFFFFF),
                "s": lambda v: struct.pack("<f", v)}[sh.load](value)

    def build(self, ti: int, t: Table) -> None:
        sh = t.shape
        cell_size = 1 << sh.shift
        table_off = self.alloc(t.capacity * cell_size)
        self.table_off[ti] = table_off
        entries_off = self.alloc(max(1, len(t.entries)) * sh.stride)
        cells = [None] * t.capacity   # linear probing, as the game's table is probed
        for index, (h, value) in enumerate(t.entries):
            pos = h % t.capacity
            while cells[pos] is not None:
                pos = (pos + 1) % t.capacity
            cells[pos] = (h, index)
            entry = entries_off + index * sh.stride
            self.entry_off[(ti, h)] = entry
            if sh.load == "x":
                buf = self.alloc(t.string_room)
                self.string_off[(ti, h)] = buf
                self.put(buf, value.encode() + b"\0")
                self.put64(entry + sh.value, self.base + buf)
            else:
                width = 1 if sh.load == "b" else 4
                other = (not value) if sh.load == "b" else (value ^ 0x5A5A5A5A if sh.load == "w" else -value - 1.5)   # the copies either side differ
                self.put(entry + sh.value - width, self.encode(sh, other))
                self.put(entry + sh.value + width, self.encode(sh, other))
                self.put(entry + sh.value, self.encode(sh, value))
        for pos, c in enumerate(cells):
            if c is not None:
                self.put32(table_off + pos * cell_size, c[0])
                self.put32(table_off + pos * cell_size + sh.index, c[1])
        self.put64(self.OBJ + sh.table, self.base + table_off)
        self.put32(self.OBJ + sh.capacity, t.capacity)
        self.put32(self.OBJ + sh.count, len(t.entries))
        self.put64(self.OBJ + sh.entries, self.base + entries_off)

    # the pokes that change the store while the game runs, as lines of the test program's script
    def poke(self, off: int, blob: bytes) -> str:
        return f"P {off:x} {blob.hex()}\n"

    def set_value(self, ti: int, h: int, value) -> str:
        sh = self.tables[ti].shape
        assert sh.load != "x"
        return self.poke(self.entry_off[(ti, h)] + sh.value, self.encode(sh, value))

    def set_string(self, ti: int, h: int, text: str) -> str:
        return self.poke(self.string_off[(ti, h)], text.encode() + b"\0")

    def unbuild(self) -> str:
        """The script line that makes the variable null again (the game has not built its store)."""
        return self.poke(self.VAR, bytes(8))

    def build_again(self) -> str:
        return self.poke(self.VAR, struct.pack("<Q", self.base + self.OBJ))

    def table_pointer(self, ti: int, value: int) -> str:
        return self.poke(self.OBJ + self.tables[ti].shape.table, struct.pack("<Q", value))

    def count_poke(self, ti: int, value: int) -> str:
        return self.poke(self.OBJ + self.tables[ti].shape.count, struct.pack("<I", value))

    def restore_table_pointer(self, ti: int) -> str:
        return self.table_pointer(ti, self.base + self.table_off[ti])

    def capacity_poke(self, ti: int, value: int) -> str:
        return self.poke(self.OBJ + self.tables[ti].shape.capacity, struct.pack("<I", value))

    def module(self, build: bytes = bytes(32)) -> bytes:
        return make_nso(data=bytes(self.data), build=build)
