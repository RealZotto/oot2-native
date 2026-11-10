# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Synthetic task, mesh and pixel programs for the first translated mesh draw (the mesh bindings work), assembled from instruction words.

No game code is involved and nothing here reads global memory, so the translator's raw-access helper is not either. The
programs draw a strip of small triangles, one for each (task workgroup n, mesh workgroup j) pair, and encode n and j twice,
in where the triangle lies and in what colour it has, so that the picture says which workgroups ran:

  task   n = the workgroup's number (ISBERD 0x78, which includes the draw's first task), written to the payload as the bits of
         the float 0.5 + n / 32, and the number of mesh workgroups M, read from constant bank 3 word 4 (c[3][0x10]).
  mesh   x = the payload's float, y = 0.5 + j / 32 for its own workgroup number j (as bits), one triangle of three vertices
         (x, y), (x + d, y), (x, y + d), d = 5 / 256, the colour (x, y, 0.5, 1), and as primitive count c[3][0x14].
  mesh   (no task) the same with n = its own number including the first task, and the row fixed.
  pixel  the interpolated colour, from generic attribute 0.

A float in [0.5, 1) is 0x3F000000 plus a mantissa that counts steps of 2^-24, so 0.5 + n / 32 is 0x3F000000 + (n << 19) and the
triangle's leg d is a mantissa step of 5 << 16: integer multiply-adds make every position and colour, and the only instructions
are MOV, IMAD, ISBERD, ISBEWR, IPA and EXIT. In a 256 by 256 target with the default viewport a step of 1 / 32 is 4 pixels, the leg
2.5 pixels, so each triangle covers exactly three pixels, none of them on an edge.
"""

from __future__ import annotations

import struct

# The scheduling fields of bits 104..115: a stall of one cycle and no write or read barrier, so that each result reaches the next reader
NO_BARRIERS = 0xFE2 << 40

HALF = 0x3F000000            # the float 0.5
ONE = 0x3F800000             # the float 1.0
CELL_SHIFT = 19              # 1 / 32 in mantissa steps of 2^-24 is 1 << 19
LEG = 5 << 16                # 5 / 256, a leg of 2.5 pixels of the 256 pixel target

IMAGE = 256                  # the target's side
CELL_PIXELS = 4
LEG_PIXELS = 2.5
MAX_CELLS = 16               # n and j stay under 16: 0.5 + 15 / 32 + d is under 1.0, the same binade
ROW = 2                      # the row of a mesh stage that has no task: y = 0.5 + 2 / 32

# The uni layout's places for the stages of a mesh pipeline (architecture D13; the descriptions say the same)
FIRST_TASK_OFFSET = 24


def ampere(op: int, rd: int = 255, ra: int = 255, rb: int = 255, low_bits: int = 0, rc: int = 255, high_bits: int = 0) -> bytes:
    """One sm_86 instruction: the opcode in bits 0..11, guard PT (7) in 12..14, Rd 16..23, Ra 24..31, Rb 32..39, more fields from
    bit 40 (low_bits), Rc 64..71 and fields from bit 72 (high_bits, given as bit numbers of the whole instruction)."""
    lo = op | (7 << 12) | (rd << 16) | (ra << 24) | (rb << 32) | low_bits
    hi = rc | (high_bits >> 64)
    assert lo >> 64 == 0 and hi >> 64 == 0, "an instruction field past bit 127"
    return struct.pack("<QQ", lo, hi)


def scheduled(ins: bytes) -> bytes:
    lo, hi = struct.unpack("<QQ", ins)
    return struct.pack("<QQ", lo, hi | NO_BARRIERS)


EXIT = struct.pack("<QQ", 0x94D | (7 << 12), 7 << (87 - 64))


def mov_imm(rd: int, imm: int) -> bytes:
    """MOV Rd, imm32: MOV (0x002) in ALU operand form 4 (bits 9..11), the immediate in bits 32..63."""
    return scheduled(ampere(0x802, rd=rd, ra=255, rb=0, low_bits=(imm & 0xFFFFFFFF) << 32))


def mov_cbuf(rd: int, bank: int, offset: int) -> bytes:
    """MOV Rd, c[bank][offset]: the form 5 operand, the bank in bits 54..59 and the byte offset in bits 38..53."""
    return scheduled(ampere(0xA02, rd=rd, ra=0, rb=0, low_bits=(offset << 38) | (bank << 54)))


def imad_imm(rd: int, ra: int, imm: int, rc: int) -> bytes:
    """IMAD Rd, Ra, imm32, Rc: the low word of Ra * imm + Rc (0x024 in form 4: B the immediate, C the register at bits 64..71),
    its predicate output (bits 81..83) the true predicate."""
    return scheduled(ampere(0x824, rd=rd, ra=ra, rb=0, low_bits=(imm & 0xFFFFFFFF) << 32, rc=rc, high_bits=7 << 81))


def isbewr(mode: int, offset: int, value_reg: int) -> bytes:
    """ISBEWR [offset], Rb: the stage buffer store (0x927), data in Rb, the byte address in bits 40..63, the mode in bits 72..80."""
    return scheduled(ampere(0x927, ra=255, rb=value_reg, low_bits=offset << 40, high_bits=mode << 72))


def isberd(mode: int, rd: int, offset: int) -> bytes:
    """ISBERD Rd, [offset]: the stage buffer load (0x923): 0x78 the workgroup's number, 0x38 a payload word."""
    return scheduled(ampere(0x923, rd=rd, ra=255, low_bits=offset << 40, high_bits=mode << 72))


def ipa(rd: int, attribute_byte: int) -> bytes:
    """IPA Rd, a[attribute_byte]: pass, default sample location, the attribute's word address (bytes / 4) in bits 64..71."""
    return scheduled(ampere(0x326, rd=rd, ra=255, rb=255, rc=attribute_byte >> 2))


# ---- the programs ----------------------------------------------------------------------

def vertex_map(colour: int = 4) -> list[list[int]]:
    """The vertex outputs of the mesh stage, in attribute line order: the position, then the first `colour` components of generic
    location 0 (all four unless a test leaves the last ones for the pixel stage to read as what the console reads from an unwritten
    output)."""
    return [[-1, c] for c in range(4)] + [[0, c] for c in range(colour)]


VERTEX_MAP = vertex_map()
LINE_BYTES = 0x80            # one line of 32 words for each written component

PAYLOAD_X = 0x10             # the task's payload word that carries x (byte 16, after the number of mesh workgroups at byte 4)
COUNT_BANK_TASK_WORD = 0x10  # c[3][0x10]: the mesh workgroups each task launches
COUNT_BANK_MESH_WORD = 0x14  # c[3][0x14]: the primitive count of a mesh workgroup

# Registers: R2 x, R3 j, R4 n, R5 the group count, R6 0.5, R7 y, R8 x + d, R9 d, R10 y + d, R12 1.0, R13 the primitive count, R14 and R15 indices


def triangle(x0: int, x1: int, y0: int, y1: int, z: int, w: int, colour: int = 4) -> list[bytes]:
    """ISBEWR of one triangle (x0, y0), (x1, y0), (x0, y1) into the attribute lines, with the colour (x0, y0, z, w), of which only the
    first `colour` components are stored: every register is a register number. Line L is the L-th component of vertex_map(colour),
    vertex v word v of the line."""
    lines = [(x0, x1, x0), (y0, y0, y1), (z, z, z), (w, w, w)]        # the position x, y, z, w of vertices 0, 1, 2
    lines += [(x0, x0, x0), (y0, y0, y0), (z, z, z), (w, w, w)][:colour]   # the colour r, g, b, a
    return [isbewr(0xF8, line * LINE_BYTES + 4 * v, reg) for line, regs in enumerate(lines) for v, reg in enumerate(regs)]


def finish_mesh() -> list[bytes]:
    """The primitive count, from the constant bank, and the three vertex indices 0, 1, 2 as map bytes 4, 5 and 6."""
    return [mov_cbuf(13, 3, COUNT_BANK_MESH_WORD), isbewr(0x88, 3, 13), mov_imm(14, 1), mov_imm(15, 2),
            isbewr(0x80, 4, 255), isbewr(0x80, 5, 14), isbewr(0x80, 6, 15), EXIT]


def task_program() -> list[bytes]:
    """n = ISBERD 0x78 (the workgroup's number with the draw's first task), the group count M from c[3][0x10] at payload byte 4,
    and x = 0.5 + n / 32 as bits at payload byte 0x10."""
    return [isberd(0x78, 4, 0), mov_imm(6, HALF), imad_imm(2, 4, 1 << CELL_SHIFT, 6), mov_cbuf(5, 3, COUNT_BANK_TASK_WORD),
            isbewr(0xB8, 4, 5), isbewr(0xB8, PAYLOAD_X, 2), EXIT]


def mesh_program(colour: int = 4) -> list[bytes]:
    """x from the payload word, y = 0.5 + j / 32 for the mesh workgroup's own number j; stores the first `colour` colour components."""
    code = [isberd(0x38, 2, PAYLOAD_X), isberd(0x78, 3, 0), mov_imm(6, HALF), imad_imm(7, 3, 1 << CELL_SHIFT, 6), mov_imm(9, LEG),
            imad_imm(8, 2, 1, 9), imad_imm(10, 7, 1, 9), mov_imm(12, ONE)]
    return code + triangle(2, 8, 7, 10, 6, 12, colour) + finish_mesh()


def mesh_only_program(row: int, colour: int = 4) -> list[bytes]:
    """A mesh stage without a task: n is its own workgroup number, with the draw's first task, and the row is fixed."""
    code = [isberd(0x78, 4, 0), mov_imm(6, HALF), imad_imm(2, 4, 1 << CELL_SHIFT, 6), mov_imm(7, HALF + (row << CELL_SHIFT)),
            mov_imm(9, LEG), imad_imm(8, 2, 1, 9), imad_imm(10, 7, 1, 9), mov_imm(12, ONE)]
    return code + triangle(2, 8, 7, 10, 6, 12, colour) + finish_mesh()


READS = [(0, c) for c in range(4)]   # the pixel program's inputs, one (location, component) for each of R0..R3


def pixel_program(reads: list[tuple[int, int]] = READS) -> bytes:
    """The pixel program with its 0x80-byte header: each of the four attributes `reads` names, a (generic location, component) for
    each of R0..R3, interpolated in perspective, one render target of four components, and IPA of each into R0..R3 (the target's
    components, in order). The default reads the four components of location 0."""
    header = bytearray(0x80)
    struct.pack_into("<I", header, 0, 2 | (5 << 10))                  # SphType 2 (pixel), ShaderType 5 (fragment)
    for location, component in reads:                                 # imap vector `location`: perspective in the components read
        header[6 * 4 + location] |= 2 << (2 * component)
    struct.pack_into("<I", header, 18 * 4, 0xF)                       # omap: target 0, four components
    return bytes(header) + b"".join(ipa(r, 0x80 + 16 * location + 4 * component) for r, (location, component) in enumerate(reads)) + EXIT


def code_bytes(instructions: list[bytes]) -> bytes:
    return b"".join(instructions)


def mesh_fields(stage: str, payload: int = 256, max_vertices: int = 3, max_primitives: int = 1, has_task: bool = True,
                colour: int = 4) -> dict:
    """The fields of a task or mesh job as the splitter writes them (xlate/host/Job.cs): the thread group, the mesh object that gives the
    sizes and, for a mesh job, its output map."""
    mesh = {"kind": stage, "threads": 32, "payload": payload, "own_shared": False}
    fields = {"local": [32, 1, 1], "shared": 192, "stage_memory": [0, 16, 64, 128], "mesh": mesh}
    if stage == "ms":
        mesh.update(has_task=has_task, max_vertices=max_vertices, max_primitives=max_primitives)
        fields["outputs"] = {"vertex": vertex_map(colour), "primitive": []}
    return fields


def jobs(work_dir) -> dict[str, dict]:
    """Writes the programs under work_dir and returns their jobs by name: "ts" and "ms" (a pair), "ms_only" (a mesh without a task) and "ps"."""
    from pathlib import Path
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    programs = {"ts": ("ts", code_bytes(task_program())), "ms": ("ms", code_bytes(mesh_program())),
                "ms_only": ("ms", code_bytes(mesh_only_program(ROW))), "ps": ("ps", pixel_program())}
    out = {}
    for name, (stage, data) in programs.items():
        path = work_dir / f"{name}.bin"
        path.write_bytes(data)
        job = {"key": f"syn72-{name.replace('_', '-')}", "stage": stage, "arch": "sm86", "code": str(path), "c1": ""}
        if stage != "ps":
            job.update(mesh_fields(stage, has_task=name != "ms_only"))
        out[name] = job
    return out
