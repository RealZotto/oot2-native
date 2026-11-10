# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""A reader for BYML, the binary form of the game's parameter files (little endian, versions 2 to 7).

    header   "YB", version (u16), key table offset (u32), string table offset (u32), root offset (u32)
    nodes    a type byte, then a 24-bit count (containers and string tables) or the value in the next four bytes
    array    type 0xC0, count, one type byte per element (padded to four), then one u32 per element
    dict     type 0xC1, count, then per entry a 24-bit key index, a type byte and a u32 value; sorted by key index
    strings  type 0xC2, count, count + 1 u32 offsets from the node, then the NUL-terminated UTF-8 strings

Containers and the 64-bit values are stored as an offset from the start of the file. The reader returns plain Python
values: dict (insertion order is the file's), list, str, bool, int, float, bytes, None. It reads only; a file that
does not fit its own offsets raises BymlError, never an IndexError or a loop. Nesting is bounded.
"""

from __future__ import annotations

import struct

MAX_DEPTH = 64
MAX_NODES = 1 << 20   # values read in one file, every visit of a shared node counted

T_STRING, T_BINARY, T_BINARY_PARAM = 0xA0, 0xA1, 0xA2
T_ARRAY, T_DICT, T_STRING_TABLE = 0xC0, 0xC1, 0xC2
T_BOOL, T_INT, T_FLOAT, T_UINT, T_INT64, T_UINT64, T_DOUBLE = 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6
T_NULL = 0xFF


class BymlError(ValueError):
    """The bytes are not a BYML file this reader can follow."""


def is_byml(data) -> bool:
    return len(data) >= 16 and bytes(data[:2]) == b"YB" and 2 <= struct.unpack_from("<H", data, 2)[0] <= 7


class _Reader:
    def __init__(self, data):
        if len(data) < 16 or bytes(data[:2]) != b"YB":
            raise BymlError("not a BYML file")
        self.d = data
        self.version = struct.unpack_from("<H", data, 2)[0]
        if not 2 <= self.version <= 7:
            raise BymlError(f"BYML version {self.version} is not read")
        keys_at, strings_at, self.root_at = struct.unpack_from("<III", data, 4)
        self.keys = self.string_table(keys_at) if keys_at else []
        self.strings = self.string_table(strings_at) if strings_at else []
        self.nodes = 0

    def need(self, at: int, n: int) -> None:
        if at < 0 or n < 0 or at + n > len(self.d):
            raise BymlError(f"a node at {at} lies outside the file")

    def u32(self, at: int) -> int:
        self.need(at, 4)
        return struct.unpack_from("<I", self.d, at)[0]

    def string_table(self, at: int) -> list[str]:
        self.need(at, 4)
        if self.d[at] != T_STRING_TABLE:
            raise BymlError("a string table is expected here")
        count = int.from_bytes(self.d[at + 1:at + 4], "little")
        self.need(at + 4, 4 * (count + 1))
        offs = struct.unpack_from(f"<{count + 1}I", self.d, at + 4)
        out = []
        for i in range(count):
            start, stop = at + offs[i], at + offs[i + 1]
            if not at <= start <= stop <= len(self.d) or stop == start:
                raise BymlError("a string lies outside its table")
            raw = bytes(self.d[start:stop])
            out.append(raw.rstrip(b"\0").decode("utf-8", "replace"))
        return out

    def value(self, kind: int, word: int, depth: int):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise BymlError("too many nodes")
        if kind == T_STRING:
            if word >= len(self.strings):
                raise BymlError("a string index is out of range")
            return self.strings[word]
        if kind == T_BOOL:
            return bool(word)
        if kind == T_INT:
            return struct.unpack("<i", struct.pack("<I", word))[0]
        if kind == T_UINT:
            return word
        if kind == T_FLOAT:
            return struct.unpack("<f", struct.pack("<I", word))[0]
        if kind == T_NULL:
            return None
        if kind in (T_INT64, T_UINT64, T_DOUBLE):
            self.need(word, 8)
            fmt = {T_INT64: "<q", T_UINT64: "<Q", T_DOUBLE: "<d"}[kind]
            return struct.unpack_from(fmt, self.d, word)[0]
        if kind in (T_BINARY, T_BINARY_PARAM):
            size = self.u32(word)
            skip = 8 if kind == T_BINARY_PARAM else 4
            self.need(word + skip, size)
            return bytes(self.d[word + skip:word + skip + size])
        if kind in (T_ARRAY, T_DICT):
            return self.container(word, kind, depth + 1)
        raise BymlError(f"node type {kind:#x} is not read")

    def container(self, at: int, expect: int | None, depth: int):
        if depth > MAX_DEPTH:
            raise BymlError("nesting is too deep")
        self.need(at, 4)
        kind = self.d[at]
        if expect is not None and kind != expect:
            raise BymlError("a container of another type than its parent says")
        count = int.from_bytes(self.d[at + 1:at + 4], "little")
        if kind == T_ARRAY:
            types_at = at + 4
            vals_at = types_at + ((count + 3) & ~3)
            self.need(vals_at, 4 * count)
            return [self.value(self.d[types_at + i], self.u32(vals_at + 4 * i), depth) for i in range(count)]
        if kind == T_DICT:
            self.need(at + 4, 8 * count)
            out = {}
            for i in range(count):
                e = at + 4 + 8 * i
                key = int.from_bytes(self.d[e:e + 3], "little")
                if key >= len(self.keys):
                    raise BymlError("a key index is out of range")
                out[self.keys[key]] = self.value(self.d[e + 3], self.u32(e + 4), depth)
            return out
        raise BymlError(f"container type {kind:#x} is not read")


def load(data):
    """The root value of a BYML file."""
    r = _Reader(data)
    if r.root_at == 0:
        return None
    return r.container(r.root_at, None, 0)


def keys_of(data) -> list[str]:
    """The key table of a BYML file, in file order."""
    return _Reader(data).keys
