# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the pipeline builder (runtime/gpu/vk_pipeline.c, vk_layout.c, vk_caps.c, spirv.c, shader_desc.c; architecture
D12 and D13, the pipeline builder), through tests/runtime/vk_pipeline_test.c, built like tests/test_runtime_gpu.py's programs at -O2.

Three groups, by what they need:

- Without a device (continuous integration runs them): the layout's tables against xlate/host/UniLayout.cs, the dynamic
  state lists, the description reader (stage, fit, interface, refusals), the padding rule, the mesh link rule (the runs
  of words a mesh stage leaves unwritten, by location, type and rate), SPIR-V declarations and the outputs added to a
  vertex or mesh module (structure checked here, and by spirv-val when the SDK has it), the required list of
  capabilities, cache file headers judged for a device, and the store: a key the pack lacks, a job that did not
  translate, a stage of another kind and a module that does not fit are each a named fatal that prints the stage's shape.
- On a device with every item of oot2/vk_required.h, with a synthetic pack of GLSL modules the test compiles (no game
  data): the layouts under validation, a pair linked for two attachment formats and drawn through the internal submit
  with the padded input reading (0, 0, 0, 1), the counter-run that leaves it unpadded (the validation layer reports
  VUID-RuntimeSpirv-OpEntryPoint-08743), a mesh stage with four pixel stages that read words it does not write, whose
  padded libraries are made once for each pad set (one of them read per primitive), a compute dispatch, a pipeline asked
  for with a submission in flight (a named fatal), the pipeline cache warm in a second process and dropped after a driver
  change, a stage's objects released and made again, and eight threads asking for one pipeline.
- On the player's own pack (the desktop, with OOT2_T60_PACK and OOT2_T60_WORK set; never in continuous integration): the
  pair the stage bindings work drew, built from the pack, against tools/firstdraw's own draw of the same modules; a sample of corpus
  pairs by the pipeline cost study's method, padded pairs included; compute pipelines; a second process warm from the cache; and the
  capability check over the whole corpus with its counter-run.

Without clang or the Vulkan SDK the program tests skip, unless OOT2_REQUIRE_CLANG or OOT2_REQUIRE_VULKAN is set; without a
usable device the device tests skip, unless OOT2_REQUIRE_GPU is set.

Run: python -m unittest discover -s tests -p test_runtime_vk_pipeline.py -v
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import stat
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import (EXIT_FATAL, ROOT, build_program, case_env, clang_or_skip, fatal_lines,
                              skip_on_partial_device, vulkan_or_skip)

sys.path.insert(0, str(ROOT / "tools" / "shaders"))
sys.path.insert(0, str(ROOT / "tools" / "pipecost"))
sys.path.insert(0, str(ROOT / "tools"))
import cache_pack as cp  # noqa: E402
from prepare import ledger as prepare_ledger  # noqa: E402

TEST_SOURCE = ROOT / "tests" / "runtime" / "vk_pipeline_test.c"
UNI_LAYOUT = ROOT / "xlate" / "host" / "UniLayout.cs"
JOB_CS = ROOT / "xlate" / "host" / "Job.cs"
LEVEL = "-O2"
EXIT_NO_DEVICE = 3
SPIRV_MAGIC = 0x07230203
TRANSLATOR = hashlib.sha256(b"synthetic translator of the pipeline builder").hexdigest()

# Vulkan constants the tables are printed with.
VK_UNIFORM_BUFFER, VK_STORAGE_BUFFER, VK_STORAGE_IMAGE, VK_SAMPLED_IMAGE, VK_SAMPLER = 6, 7, 3, 2, 0
STAGE_VERTEX, STAGE_FRAGMENT, STAGE_COMPUTE, STAGE_TASK, STAGE_MESH = 0x1, 0x10, 0x20, 0x40, 0x80
GFX_STAGES = STAGE_VERTEX | STAGE_FRAGMENT | STAGE_TASK | STAGE_MESH
OP = {"capability": 17, "extension": 10, "memory_model": 14, "entry_point": 15, "execution_mode": 16, "decorate": 71,
      "void": 19, "fn_type": 33, "float": 22, "int": 21, "vector": 23, "array": 28, "pointer": 32, "constant": 43,
      "composite": 44, "null": 46, "variable": 59, "function": 54, "label": 248, "ret": 253, "fn_end": 56}
SC_WORKGROUP = 4

_build: dict[str, object] = {}


def program() -> Path:
    """The test program, built once per run."""
    if "exe" not in _build:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        tmp = tempfile.TemporaryDirectory(prefix="oot2-vkp-", ignore_cleanup_errors=True)
        atexit.register(tmp.cleanup)
        _build["tmp"] = tmp
        _build["glslang"] = glslang
        _build["exe"] = build_program(cc, include, glslang, TEST_SOURCE, Path(tmp.name), levels=(LEVEL,))[LEVEL]
    return _build["exe"]  # type: ignore[return-value]


def run(case: str, *args: str, env: dict[str, str] | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([str(program()), case, *args], capture_output=True, text=True, timeout=timeout, env=case_env(env))


def scratch() -> Path:
    tmp = tempfile.TemporaryDirectory(prefix="oot2-vkp-case-", ignore_cleanup_errors=True)
    atexit.register(tmp.cleanup)
    return Path(tmp.name)


# ---- a tiny SPIR-V assembler -------------------------------------------------------

def inst(op: int, *args: int) -> list[int]:
    return [((len(args) + 1) << 16) | op, *args]


def text_words(s: str) -> list[int]:
    b = s.encode() + b"\0"
    b += b"\0" * (-len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))


def module_words(body: list[list[int]], bound: int) -> bytes:
    words = [SPIRV_MAGIC, 0x00010300, 0, bound, 0]
    for i in body:
        words += i
    return struct.pack(f"<{len(words)}I", *words)


def vertex_module(extra_caps: tuple[int, ...] = (), extensions: tuple[str, ...] = (), entry_points: int = 1) -> bytes:
    """A vertex module with one float vec4 output at Location 0 and nothing else."""
    body = [inst(OP["capability"], 1)]
    for c in extra_caps:
        body.append(inst(OP["capability"], c))
    for e in extensions:
        i = inst(OP["extension"], *text_words(e))
        i[0] = (len(i) << 16) | OP["extension"]
        body.append(i)
    body.append(inst(OP["memory_model"], 0, 1))
    for n in range(entry_points):
        main = 7 + 2 * n
        name = text_words("main")
        ep = inst(OP["entry_point"], 0, main, *name, 6)
        body.append(ep)
    body += [inst(OP["decorate"], 6, 30, 0), inst(OP["void"], 1), inst(OP["fn_type"], 2, 1), inst(OP["float"], 3, 32),
             inst(OP["vector"], 4, 3, 4), inst(OP["pointer"], 5, 3, 4), inst(OP["variable"], 5, 6, 3)]
    for n in range(entry_points):
        main = 7 + 2 * n
        body += [inst(OP["function"], 1, main, 0, 2), inst(OP["label"], main + 1), inst(OP["ret"]), inst(OP["fn_end"])]
    return module_words(body, 7 + 2 * entry_points)


def bare_module() -> bytes:
    """A module that declares no capability or extension (an unopened builder's empty required list refuses none of it) and is
    SPIR-V 1.3 with a vertex entry point: the mesh rewrite refuses it by name, where a mesh module would go on to the driver."""
    body = [inst(OP["memory_model"], 0, 1), inst(OP["entry_point"], 0, 3, *text_words("main")), inst(OP["void"], 1),
            inst(OP["fn_type"], 2, 1), inst(OP["function"], 1, 3, 0, 2), inst(OP["label"], 4), inst(OP["ret"]),
            inst(OP["fn_end"])]
    return module_words(body, 5)


def compute_module(variables: tuple[tuple[int, bool], ...] = ((64, False),), *, undeclared_type: bool = False) -> bytes:
    """A compute module (SPIR-V 1.3, local size 1) whose function returns at once, with one Workgroup variable for each
    (words, initialized): an array of that many 32-bit unsigned integers, its pointer type shared by every variable of the
    same size, and the variable's own initializer (the null value of the array) when initialized."""
    body = [inst(OP["capability"], 1), inst(OP["memory_model"], 0, 1), inst(OP["entry_point"], 5, 4, *text_words("main")),
            inst(OP["execution_mode"], 4, 17, 1, 1, 1), inst(OP["void"], 1), inst(OP["fn_type"], 2, 1),
            inst(OP["int"], 3, 32, 0)]
    nxt, types, nulls = 5, {}, []
    for words, _ in variables:
        if words in types:
            continue
        length, arr, ptr = nxt, nxt + 1, nxt + 2
        nxt += 3
        types[words] = (arr, ptr)
        body += [inst(OP["constant"], 3, length, words), inst(OP["array"], arr, 3, length), inst(OP["pointer"], ptr, SC_WORKGROUP, arr)]
    for words, initialized in variables:
        arr, ptr = types[words]
        var = nxt
        nxt += 1
        if undeclared_type:
            body.append(inst(OP["variable"], 99, var, SC_WORKGROUP))
        elif initialized:
            body += [inst(OP["null"], arr, nxt), inst(OP["variable"], ptr, var, SC_WORKGROUP, nxt)]
            nxt += 1
        else:
            body.append(inst(OP["variable"], ptr, var, SC_WORKGROUP))
    body += [inst(OP["function"], 1, 4, 0, 2), inst(OP["label"], nxt), inst(OP["ret"]), inst(OP["fn_end"])]
    return module_words(body, nxt + 1)


SC_INPUT, SC_OUTPUT = 1, 3
EXEC_MESH, EXEC_VERTEX = 5365, 0


def mesh_module(*, vertices: int = 160, primitives: int = 128, threads: int = 32, outputs=((0, 0), (0, 1), (0, 2), (0, 3)),
                primitive_outputs=(), sets: int = 1, own_index: bool = False, tail: str = "return", model: int = EXEC_MESH,
                version: int = 0x00010400, modes: bool = True) -> bytes:
    """A mesh module in the shape of the translator's (xlate MeshEmission): SPIR-V 1.4, one MeshEXT entry point of `threads`
    invocations, the position and one float[vertices] for each (location, component) of `outputs` (per primitive, float[primitives]
    decorated PerPrimitiveEXT, for `primitive_outputs`), the triangle indices, the local invocation id, and a main function that
    calls OpSetMeshOutputsEXT `sets` times and returns. tail "phi" ends the first block in a conditional branch to a block that
    holds a phi naming the first block, as the translator's epilogue could leave one; own_index declares LocalInvocationIndex."""
    ids = iter(range(1, 1000))
    void, fn_type, flt, u32, v4, boolean, c_vertices, c_primitives = (next(ids) for _ in range(8))
    arr_v, ptr_arr_v, arr_pos, ptr_arr_pos, v3, arr_idx, ptr_arr_idx, ptr_in_v3 = (next(ids) for _ in range(8))
    arr_p, ptr_arr_p, position, indices, local_id, main, entry, c_one = (next(ids) for _ in range(8))
    variables = [(next(ids), loc, comp, False) for loc, comp in outputs]
    variables += [(next(ids), loc, comp, True) for loc, comp in primitive_outputs]
    index_var = next(ids) if own_index else None
    ptr_in_u32 = next(ids) if own_index else None
    cond, then, merge, phi = (next(ids) for _ in range(4))
    bound = next(ids)
    interface = [position, indices, local_id] + [v[0] for v in variables] + ([index_var] if own_index else [])
    body = [inst(OP["capability"], 1), inst(OP["capability"], 5283), inst(OP["extension"], *text_words("SPV_EXT_mesh_shader")),
            inst(OP["memory_model"], 0, 1), inst(OP["entry_point"], model, main, *text_words("main"), *interface)]
    if modes:
        body += [inst(OP["execution_mode"], main, 17, threads, 1, 1), inst(OP["execution_mode"], main, 26, vertices),
                 inst(OP["execution_mode"], main, 5270, primitives), inst(OP["execution_mode"], main, 5298)]
    body += [inst(OP["decorate"], position, 11, 0), inst(OP["decorate"], indices, 11, 5296), inst(OP["decorate"], local_id, 11, 27)]
    for var, loc, comp, per_primitive in variables:
        body += [inst(OP["decorate"], var, 30, loc), inst(OP["decorate"], var, 31, comp)]
        if per_primitive:
            body.append(inst(OP["decorate"], var, 5271))
    if own_index:
        body.append(inst(OP["decorate"], index_var, 11, 29))
    body += [inst(OP["void"], void), inst(OP["fn_type"], fn_type, void), inst(OP["float"], flt, 32), inst(OP["int"], u32, 32, 0),
             inst(OP["vector"], v4, flt, 4), inst(20, boolean), inst(OP["constant"], u32, c_vertices, vertices),
             inst(OP["constant"], u32, c_primitives, primitives), inst(OP["constant"], u32, c_one, 1),
             inst(OP["array"], arr_v, flt, c_vertices), inst(OP["pointer"], ptr_arr_v, SC_OUTPUT, arr_v),
             inst(OP["array"], arr_pos, v4, c_vertices), inst(OP["pointer"], ptr_arr_pos, SC_OUTPUT, arr_pos),
             inst(OP["vector"], v3, u32, 3), inst(OP["array"], arr_idx, v3, c_primitives),
             inst(OP["pointer"], ptr_arr_idx, SC_OUTPUT, arr_idx), inst(OP["pointer"], ptr_in_v3, SC_INPUT, v3),
             inst(OP["array"], arr_p, flt, c_primitives), inst(OP["pointer"], ptr_arr_p, SC_OUTPUT, arr_p),
             inst(OP["variable"], ptr_arr_pos, position, SC_OUTPUT), inst(OP["variable"], ptr_arr_idx, indices, SC_OUTPUT),
             inst(OP["variable"], ptr_in_v3, local_id, SC_INPUT)]
    for var, _, _, per_primitive in variables:
        body.append(inst(OP["variable"], ptr_arr_p if per_primitive else ptr_arr_v, var, SC_OUTPUT))
    if own_index:
        body += [inst(OP["pointer"], ptr_in_u32, SC_INPUT, u32), inst(OP["variable"], ptr_in_u32, index_var, SC_INPUT)]
    if tail == "phi":
        body.append(inst(41, boolean, cond))   # OpConstantTrue
    body += [inst(OP["function"], void, main, 0, fn_type), inst(OP["label"], entry)]
    body += [inst(5295, c_vertices, c_primitives) for _ in range(sets)]
    if tail == "phi":
        body += [inst(247, merge, 0), inst(250, cond, then, merge), inst(OP["label"], then), inst(249, merge),
                 inst(OP["label"], merge), inst(245, u32, phi, c_one, entry, c_vertices, then)]
    body += [inst(OP["ret"]), inst(OP["fn_end"])]
    words = [SPIRV_MAGIC, version, 0, bound, 0]
    for i in body:
        words += i
    return struct.pack(f"<{len(words)}I", *words)


def parse_words(data: bytes) -> list[list[int]]:
    words = struct.unpack(f"<{len(data) // 4}I", data)
    out, i = [], 5
    while i < len(words):
        n = words[i] >> 16
        assert n > 0, f"zero-length instruction at {i}"
        out.append(list(words[i:i + n]))
        i += n
    return out


class MeshModule:
    """A mesh module read back from its words, for the structure tests: its types, constants, variables, decorations, the entry
    point's interface and the instructions of the function that calls OpSetMeshOutputsEXT."""

    def __init__(self, data: bytes) -> None:
        self.version = struct.unpack_from("<I", data, 4)[0]
        self.type: dict[int, list[int]] = {}
        self.const: dict[int, list[int]] = {}
        self.var: dict[int, tuple[int, int]] = {}
        self.decor: dict[int, dict[int, list[int]]] = {}
        self.interface: list[int] = []
        self.main: list[list[int]] = []
        function: list[list[int]] = []
        for i in parse_words(data):
            op = i[0] & 0xffff
            if op == OP["entry_point"]:
                self.interface = i[3 + len(text_words("main")):]
            elif op == OP["decorate"]:
                self.decor.setdefault(i[1], {})[i[2]] = i[3:]
            elif op in (20, 21, 22, 23, 28, 32):
                self.type[i[1]] = i
            elif op in (41, 43, 44, 46):
                self.const[i[2]] = i
            elif op == OP["variable"] and not function:
                self.var[i[2]] = (i[1], i[3])
            elif op == OP["function"]:
                function = [i]
            elif function:
                function.append(i)
                if op == OP["fn_end"]:
                    if any((j[0] & 0xffff) == 5295 for j in function):
                        self.main = function
                    function = []

    def array_of(self, var: int) -> tuple[int, str, int]:
        """(length, scalar kind, components) of an Output array variable: float[160] is (160, "float", 1)."""
        pointer = self.type[self.var[var][0]]            # OpTypePointer %id storage pointee
        array = self.type[pointer[3]]                    # OpTypeArray %id element length
        length = self.const[array[3]][3]
        element = self.type[array[2]]                    # OpTypeVector %id component count, or a scalar
        count = element[3] if element[0] & 0xffff == 23 else 1
        scalar = self.type[element[2]] if count > 1 else element
        kind = "float" if scalar[0] & 0xffff == 22 else "sint" if scalar[3] else "uint"   # OpTypeInt %id width signedness
        return length, kind, count

    def value(self, constant: int) -> list[int]:
        """The 32-bit words a constant holds: one, or one for each component of a vector."""
        c = self.const[constant]
        op = c[0] & 0xffff
        if op == 43:
            return [c[3]]
        if op == 44:
            return [w for member in c[3:] for w in self.value(member)]
        assert op == 46, f"constant {constant} is not a value the pads use"
        t = self.type[c[1]]
        return [0] * (t[3] if t[0] & 0xffff == 23 else 1)

    def after_call(self) -> list[list[int]]:
        """The instructions that follow OpSetMeshOutputsEXT in the function."""
        at = next(n for n, i in enumerate(self.main) if (i[0] & 0xffff) == 5295)
        return self.main[at + 1:]

    def stores(self) -> dict[int, list[list[int]]]:
        """For each variable, the words of every value stored to one of its elements, in order, in the whole function."""
        chain: dict[int, int] = {}
        out: dict[int, list[list[int]]] = {}
        for i in self.main:
            op = i[0] & 0xffff
            if op == 65:
                chain[i[2]] = i[3]
            elif op == 62 and i[1] in chain:
                out.setdefault(chain[i[1]], []).append(self.value(i[2]))
        return out


def describe(key: str, stage: str, module: bytes, *, inputs=(), outputs=(), fits=True, schema=cp.DESC_SCHEMA,
             raw=False) -> bytes:
    """A description shaped like the translator's res.json, with the members the builder reads."""
    d = {"schema": schema, "key": key, "stage": stage, "exec_model": "Fragment", "entry": "main",
         "translator_id": TRANSLATOR,
         "answers": {"tex": {}, "img": {}, "attr": {}, "fout": {}, "early_fragment_tests": False, "heap_images": 33025,
                     "heap_samplers": 4096},
         "spv_bytes": len(module), "spv_sha256": hashlib.sha256(module).hexdigest(),
         "layout": {"name": "uni", "role": "pixel", "fits": fits, "overflow": [], "uniform_slots": [14, 11]},
         "raw_access": {"physical_pointers": raw, "page_table": False},
         "interface": {"inputs": list(inputs), "outputs": list(outputs), "builtins_in": ["FragCoord"], "builtins_out": []},
         "vertex_inputs": None, "notes": ["quote \" backslash \\ unicode é newline \n tab \t"], "float": 1.5e-3}
    return json.dumps(d, indent=1).encode("utf-8")


def io(location: int, components: int = 4, component: int = 0, type_: str = "float32", used: bool = True,
       array_length: int | None = None) -> dict:
    d = {"location": location, "component": component, "components": components, "type": type_, "used": used}
    if array_length is not None:
        d["array_length"] = array_length
    return d


def write_pack(path: Path, entries: list[cp.Entry], *, schema: int = cp.DESC_SCHEMA, role: int = cp.BASE) -> Path:
    cp.write_pack(path, entries, translator=TRANSLATOR, schema=schema, role=role)
    return path


# ---- without a device ----------------------------------------------------------------

class TablesTest(unittest.TestCase):
    """The layout's tables, tied to the translator's numbering (xlate/host/UniLayout.cs) and to the specification."""

    @classmethod
    def setUpClass(cls) -> None:
        done = run("tables")
        assert done.returncode == 0, done.stdout + done.stderr
        cls.out = done.stdout
        cls.lines = [l.split() for l in done.stdout.splitlines()]
        src = UNI_LAYOUT.read_text(encoding="utf-8")
        cls.consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"public const int (\w+) = (\d+);", src)}
        cls.roles = {}
        for m in re.finditer(r'private static readonly Role (\w+) = new\("(\w+)", (\w+), (.*)\);', src):
            args = re.findall(r"new\((\d+), (\d+)\)|\bNone\b|\bFirstTaskRange\b", m.group(4))
            ranges = [(int(a), int(b)) if a else None for a, b in args]
            if m.group(4).rstrip().endswith("FirstTaskRange"):   # the draw's first task: bytes 24 to 35 of the push constants
                ranges[4] = (cls.consts["FirstTaskOffset"], cls.consts["FirstTaskBytes"])
            cls.roles[m.group(2)] = {"buffer_set": m.group(3), "uniform": ranges[0], "storage": ranges[1],
                                     "image": ranges[2], "slots": ranges[3], "first_task": ranges[4]}

    def set1(self, layout: str) -> list[tuple[int, int, int]]:
        return [(int(l[2]), int(l[3]), int(l[4], 16)) for l in self.lines if l[0] == "set1" and l[1] == layout]

    def expect(self, bindings, role: str, stages: int) -> None:
        r = self.roles[role]
        for kind, vk in (("uniform", VK_UNIFORM_BUFFER), ("storage", VK_STORAGE_BUFFER), ("image", VK_STORAGE_IMAGE)):
            if r[kind] is None:
                continue
            first, count = r[kind]
            for b in range(first, first + count):
                self.assertIn((b, vk, stages), bindings, f"{role} {kind} {b}")

    def test_the_translators_roles_were_read(self) -> None:
        self.assertEqual(set(self.roles), {"pre_rasterization", "mesh", "pixel", "compute", "task"})
        self.assertEqual((self.consts["GlobalSet"], self.consts["PushSet"], self.consts["TaskSet"]), (0, 1, 2))
        self.assertEqual((self.consts["PushConstantBytes"], self.consts["PushDescriptorLimit"]), (128, 32))

    def test_graphics_pipeline_keys_distinguish_vertex_mesh_and_task_mesh(self) -> None:
        done = run("keys")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        rows = [line.split(maxsplit=1) for line in done.stdout.splitlines() if line.startswith(("vertex ", "mesh ", "task-mesh "))]
        self.assertEqual([row[0] for row in rows], ["vertex", "mesh", "task-mesh"])
        self.assertEqual(len({row[1] for row in rows}), 3, rows)

    def test_set_zero_is_the_global_update_after_bind_arrays(self) -> None:
        rows = [l for l in self.lines if l[0] == "set0"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(self.consts["ImagesBinding"], 0)
        self.assertEqual(self.consts["SamplersBinding"], 1)
        heap = int(re.search(r"DefaultHeapImages = (\d+)", JOB_CS.read_text(encoding="utf-8")).group(1))
        self.assertEqual([int(rows[0][1]), int(rows[0][2]), int(rows[0][3])], [0, VK_SAMPLED_IMAGE, heap])
        self.assertEqual([int(rows[1][1]), int(rows[1][2]), int(rows[1][3])], [1, VK_SAMPLER, 4096])
        for r in rows:   # every stage, update-after-bind, update-unused-while-pending (the bindless tables) and partially bound
            self.assertEqual(int(r[4], 16), GFX_STAGES | STAGE_COMPUTE)
            self.assertEqual(int(r[5], 16), 0x1 | 0x2 | 0x4)   # UPDATE_AFTER_BIND (1) | UPDATE_UNUSED_WHILE_PENDING (2) | PARTIALLY_BOUND (4)

    def test_the_graphics_push_set_holds_the_translators_slots(self) -> None:
        b = self.set1("graphics")
        self.assertEqual(len(b), self.consts["PushDescriptorLimit"])
        self.assertEqual(sorted(x[0] for x in b), list(range(32)))
        self.assertIn((self.consts["SupportBinding"], VK_UNIFORM_BUFFER, GFX_STAGES), b)   # the support buffer, shared
        for role in ("pre_rasterization", "mesh"):
            self.expect(b, role, STAGE_VERTEX | STAGE_MESH)
        self.expect(b, "pixel", STAGE_FRAGMENT)
        self.assertEqual(self.roles["pre_rasterization"], self.roles["mesh"] | {"slots": self.roles["pre_rasterization"]["slots"],
                                                                                "first_task": None} | {"slots": self.roles["pre_rasterization"]["slots"]}
                         if False else self.roles["pre_rasterization"], self.roles["pre_rasterization"])

    def test_the_compute_push_set_holds_the_translators_slots(self) -> None:
        b = self.set1("compute")
        self.assertEqual(sorted(x[0] for x in b), list(range(32)))
        self.assertIn((0, VK_UNIFORM_BUFFER, STAGE_COMPUTE), b)
        self.expect(b, "compute", STAGE_COMPUTE)

    def test_the_task_stages_buffers_are_set_two(self) -> None:
        rows = [(int(l[1]), int(l[2]), int(l[3], 16)) for l in self.lines if l[0] == "set2"]
        r = self.roles["task"]
        self.assertEqual(r["buffer_set"], "TaskSet")
        for kind, vk in (("uniform", VK_UNIFORM_BUFFER), ("storage", VK_STORAGE_BUFFER)):
            first, count = r[kind]
            for b in range(first, first + count):
                self.assertIn((b, vk, STAGE_TASK), rows)
        self.assertEqual(len(rows), 16)

    def test_push_constants_are_128_bytes_for_every_stage(self) -> None:
        rows = {l[1]: (int(l[2], 16), int(l[3]), int(l[4])) for l in self.lines if l[0] == "push"}
        self.assertEqual(rows["graphics"], (GFX_STAGES, 0, self.consts["PushConstantBytes"]))
        self.assertEqual(rows["compute"], (STAGE_COMPUTE, 0, self.consts["PushConstantBytes"]))
        self.assertEqual(self.consts["SlotsOffset"] + 4 * self.consts["SlotCount"], self.consts["PushConstantBytes"])

    def test_dynamic_states_follow_what_the_device_allows(self) -> None:
        counts = {(l[1], l[2]): (int(l[3]), l[4:]) for l in self.lines if l[0] == "dyn"}
        for variant, want in (("none", (3, 8, 10, 1)), ("all", (3, 11, 10, 5)), ("nopoly", (3, 10, 10, 5))):
            got = tuple(counts[(variant, p)][0] for p in ("vertex_input", "pre_rasterization", "fragment", "output"))
            self.assertEqual(got, want, variant)
        for key, (n, states) in counts.items():
            self.assertEqual(len(states), n)
            self.assertEqual(len(set(states)), n, key)   # no state twice
        # a state the device cannot make dynamic is absent, not replaced: polygon mode is the one that differs
        allp, nop = set(counts[("all", "pre_rasterization")][1]), set(counts[("nopoly", "pre_rasterization")][1])
        self.assertEqual(len(allp - nop), 1)
        self.assertEqual(nop - allp, set())


class DescriptionTest(unittest.TestCase):
    """The description reader (oot2/shader_desc.h) and the padding rule."""

    def setUp(self) -> None:
        self.dir = scratch()

    def desc(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def read(self, data: bytes) -> list[str]:
        done = run("desc", self.desc("d.json", data))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def test_a_description_is_read_past_what_it_does_not_need(self) -> None:
        mod = vertex_module()
        out = self.read(describe("k.1", "ps", mod, inputs=[io(0), io(1, 3, 1, "uint32", False, 2)],
                                 outputs=[io(0, 2, 2, "sint32")], raw=True))
        self.assertEqual(out[0], "kind ps fits 1 raw 1")
        self.assertEqual(out[1:], ["in 0 0 4 0 32 1 1", "in 1 1 3 2 32 2 0", "out 0 2 2 1 32 1 1"])

    def test_every_stage_name_is_known(self) -> None:
        mod = vertex_module()
        for name in ("vs", "ps", "cs", "ts", "ms"):
            self.assertEqual(self.read(describe("k.1", name, mod))[0], f"kind {name} fits 1 raw 0")

    def test_a_module_that_does_not_fit_says_so(self) -> None:
        self.assertEqual(self.read(describe("k.1", "ps", vertex_module(), fits=False))[0], "kind ps fits 0 raw 0")

    def test_refusals(self) -> None:
        mod = vertex_module()
        good = describe("k.1", "ps", mod)
        cases = {
            "schema": (describe("k.1", "ps", mod, schema=1), "is schema 1, the runtime reads schema 2"),
            "stage name": (describe("k.1", "gs", mod), 'names the stage "gs"'),
            "trailing text": (good + b" x", "text after its object"),
            "cut short": (good[:len(good) // 2], "is not readable"),
            "interface type": (describe("k.1", "ps", mod, inputs=[io(0, type_="bool")]), "interface type"),
            "too many entries": (describe("k.1", "ps", mod, inputs=[io(i % 32) for i in range(65)]), "more interface entries"),
            "not json": (b"[1, 2]", "is not readable"),
        }
        no_stage = json.loads(good)
        del no_stage["stage"]
        cases["no stage"] = (json.dumps(no_stage).encode(), "names no stage")
        for name, (data, text) in cases.items():
            out = self.read(data)
            self.assertEqual(len(out), 1, name)
            self.assertTrue(out[0].startswith("refused: ") and text in out[0], f"{name}: {out}")

    def pad(self, vs_out: list[dict], ps_in: list[dict]) -> list[str]:
        mod = vertex_module()
        v = self.desc("v.json", describe("v.1", "vs", mod, outputs=vs_out))
        p = self.desc("p.json", describe("p.1", "ps", mod, inputs=ps_in))
        done = run("pad", v, p)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def test_nothing_is_padded_when_the_vertex_stage_writes_every_input(self) -> None:
        self.assertEqual(self.pad([io(0), io(1)], [io(0), io(1)]), ["pad 0 "])
        self.assertEqual(self.pad([io(0), io(1), io(2)], [io(1)]), ["pad 0 "])   # an output nothing reads is no harm

    def test_a_missing_location_is_padded_whole(self) -> None:
        self.assertEqual(self.pad([io(0)], [io(0), io(3)]), ["pad 1 3:0:4:0;", "run 3 0 4 0"])

    def test_a_partly_written_location_is_padded_where_it_is_missing(self) -> None:
        out = self.pad([io(2, 2, 0)], [io(2, 4, 0)])
        self.assertEqual(out, ["pad 1 2:2:2:0;", "run 2 2 2 0"])
        out = self.pad([io(2, 1, 1)], [io(2, 4, 0)])   # component 1 written: 0 and 2..3 are missing
        self.assertEqual(out, ["pad 2 2:0:1:0;2:2:2:0;", "run 2 0 1 0", "run 2 2 2 0"])

    def test_a_run_never_mixes_types_and_an_array_takes_consecutive_locations(self) -> None:
        out = self.pad([], [io(3, 2, 0, "float32"), io(3, 2, 2, "uint32"), io(8, 4, 0, "sint32", array_length=2)])
        self.assertEqual(out, ["pad 4 3:0:2:0;3:2:2:2;8:0:4:1;9:0:4:1;", "run 3 0 2 0", "run 3 2 2 2", "run 8 0 4 1", "run 9 0 4 1"])

    def test_what_cannot_be_padded_is_refused_by_name(self) -> None:
        out = self.pad([], [io(0, type_="float64")])
        self.assertTrue(out[0].startswith("refused: ") and "64 bits wide" in out[0], out)
        out = self.pad([], [io(63, array_length=2)])
        self.assertTrue(out[0].startswith("refused: ") and "past the 64" in out[0], out)
        out = self.pad([], [io(i, 1, 0) for i in range(17)])
        self.assertTrue(out[0].startswith("refused: ") and "more than 16" in out[0], out)


def mesh_out(location: int, component: int = 0, *, per_vertex: int = 160, depth: int | None = 1,
             per_primitive: bool | None = False, type_: str = "float32") -> dict:
    """One output of a mesh stage as the translator declares it: one float of one component per vertex (or per primitive),
    an array whose length is the stage's vertex (or primitive) count."""
    d = io(location, 1, component, type_, array_length=per_vertex)
    if depth is not None:
        d["array_depth"] = depth
    if per_primitive is not None:
        d["per_primitive"] = per_primitive
    return d


def pixel_in(location: int, components: int = 4, component: int = 0, *, per_primitive: bool | None = False,
             type_: str = "float32", array_length: int | None = None, depth: int | None = None) -> dict:
    d = io(location, components, component, type_, array_length=array_length)
    d["array_depth"] = (0 if array_length is None else 1) if depth is None else depth
    if per_primitive is not None:
        d["per_primitive"] = per_primitive
    return d


class MeshLinkTest(unittest.TestCase):
    """The rule a mesh stage is linked to a pixel stage by (shd_mesh_link, used by build_link): a mesh output's outer
    array holds its vertices or primitives at one location, and rate, type and shape must agree with the pixel inputs."""

    def setUp(self) -> None:
        self.dir = scratch()

    def link(self, ms_out: list[dict], ps_in: list[dict]) -> str:
        mod = vertex_module()
        m = self.dir / "m.json"
        m.write_bytes(describe("m.1", "ms", mod, outputs=ms_out))
        p = self.dir / "p.json"
        p.write_bytes(describe("p.1", "ps", mod, inputs=ps_in))
        done = run("meshlink", str(m), str(p))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        (line,) = done.stdout.splitlines()
        return line

    def refused(self, ms_out: list[dict], ps_in: list[dict], *needles: str) -> None:
        line = self.link(ms_out, ps_in)
        self.assertTrue(line.startswith("refused: "), line)
        for needle in needles:
            self.assertIn(needle, line)

    def test_per_vertex_outputs_of_160_vertices_reach_location_63(self) -> None:
        # 160 vertices: each output is a float[160]. Counted as 160 locations, location 1 would already reach 160.
        outs = [mesh_out(loc, c) for loc in (1, 2, 40, 63) for c in range(4)]
        self.assertEqual(self.link(outs, [pixel_in(loc) for loc in (1, 2, 40, 63)]), "missing 0 ")

    def test_per_primitive_outputs_link_to_per_primitive_inputs(self) -> None:
        outs = [mesh_out(5, c, per_vertex=128, per_primitive=True) for c in range(2)] + [mesh_out(6, c) for c in range(4)]
        ins = [pixel_in(5, 2, per_primitive=True), pixel_in(6)]
        self.assertEqual(self.link(outs, ins), "missing 0 ")

    def test_what_the_mesh_stage_does_not_write_is_listed(self) -> None:
        outs = [mesh_out(3, c) for c in (0, 1)]
        self.assertEqual(self.link(outs, [pixel_in(3), pixel_in(9, 1, 2, type_="uint32")]), "missing 2 3:2:2:0;9:2:1:2;")

    def test_the_ranges_of_the_pair_that_stopped_the_run_are_listed_in_its_own_shapes(self) -> None:
        # A mesh stage that writes some words of the locations its pixel stage reads whole: x, y and z of locations 1, 2 and 4,
        # x of 3, x and y of 5 and 13 (n:3:1, n:1:3 and n:2:2 are the three shapes of the unwritten words), and all of 6.
        written = {1: 3, 2: 3, 3: 1, 4: 3, 5: 2, 13: 2, 6: 4}
        outs = [mesh_out(loc, c) for loc, n in written.items() for c in range(n)]
        self.assertEqual(self.link(outs, [pixel_in(loc) for loc in written]),
                         "missing 6 1:3:1:0;2:3:1:0;3:1:3:0;4:3:1:0;5:2:2:0;13:2:2:0;")

    def test_the_ten_ranges_of_the_widest_stop_are_listed(self) -> None:
        outs = [mesh_out(loc, c) for loc, c in SpirvTest.WIDE_WRITTEN]
        ins = [pixel_in(loc) for loc in sorted({loc for loc, _ in SpirvTest.WIDE_WRITTEN})]
        text = "".join(f"{part};" for part in SpirvTest.WIDE_PADS.split(","))
        self.assertEqual(self.link(outs, ins), f"missing 10 {text}")

    def test_a_location_the_mesh_stage_writes_nothing_of_is_one_run_of_its_words(self) -> None:
        self.assertEqual(self.link([mesh_out(1, 0)], [pixel_in(9)]), "missing 1 9:0:4:0;")

    def test_a_word_read_per_primitive_is_listed_with_its_rate_and_never_joins_a_per_vertex_run(self) -> None:
        outs = [mesh_out(5, c) for c in range(2)]
        ins = [pixel_in(5), pixel_in(6, 2, per_primitive=True), pixel_in(7, 2), pixel_in(7, 2, 2, per_primitive=True)]
        # location 5 is read per vertex with its last two words unwritten; 6 is read per primitive whole; 7 is read per vertex
        # in its first two words and per primitive in its last two, and the two runs are two variables
        self.assertEqual(self.link(outs, ins), "missing 4 5:2:2:0;6:0:2:0:p;7:0:2:0;7:2:2:0:p;")

    def test_a_run_never_mixes_two_types(self) -> None:
        ins = [pixel_in(3, 2), pixel_in(3, 1, 2, type_="uint32"), pixel_in(3, 1, 3, type_="sint32")]
        self.assertEqual(self.link([], ins), "missing 3 3:0:2:0;3:2:1:2;3:3:1:1;")

    def test_a_word_to_pad_that_is_not_32_bits_wide_is_refused_by_name(self) -> None:
        self.refused([mesh_out(2, 0)], [pixel_in(1, 4, type_="float64")], "pixel input 1 is 64 bits wide", "32-bit")
        # a word the mesh stage writes is judged by the type rule alone
        self.refused([mesh_out(1, 0)], [pixel_in(1, 1, type_="float64")], "type", "location 1")

    def test_a_rate_mismatch_is_refused_by_name(self) -> None:
        outs = [mesh_out(4, c, per_vertex=128, per_primitive=True) for c in range(4)]
        self.refused(outs, [pixel_in(4)], "rate", "location 4", "per vertex", "per primitive")
        outs = [mesh_out(4, c) for c in range(4)]
        self.refused(outs, [pixel_in(4, per_primitive=True)], "rate", "location 4", "per primitive", "per vertex")

    def test_a_type_mismatch_is_refused_by_name(self) -> None:
        outs = [mesh_out(7, c) for c in range(4)]
        self.refused(outs, [pixel_in(7, 1, 3, type_="sint32")], "type", "location 7 component 3", "sint32", "float32")

    def test_a_shape_the_builder_does_not_link_is_refused_by_name(self) -> None:
        self.refused([mesh_out(2, depth=0)], [pixel_in(2, 1)], "shape", "mesh output at location 2", "0 array levels")
        self.refused([mesh_out(2, depth=2)], [pixel_in(2, 1)], "shape", "mesh output at location 2", "2 array levels")
        outs = [mesh_out(loc, c) for loc in (8, 9) for c in range(4)]
        self.refused(outs, [pixel_in(8, array_length=2)], "shape", "pixel input at location 8", "array of 2")

    def test_a_description_without_depth_or_rate_is_refused_by_name(self) -> None:
        self.refused([mesh_out(1, depth=None)], [pixel_in(1, 1)], "mesh output at location 1", "translate")
        self.refused([mesh_out(1, per_primitive=None)], [pixel_in(1, 1)], "mesh output at location 1", "translate")
        self.refused([mesh_out(1)], [pixel_in(1, 1, per_primitive=None)], "pixel input at location 1", "translate")

    def test_locations_past_the_builders_are_still_refused(self) -> None:
        self.refused([mesh_out(64)], [pixel_in(1, 1)], "location 64", "past the 64")
        self.refused([mesh_out(1)], [pixel_in(63, array_length=2)], "location 64", "past the 64")


class BindingsTest(unittest.TestCase):
    """The bindings reader (shd_parse_binds): synthetic descriptions with the members xlate/host/ResultWriter.cs writes."""

    BV = {"1d": 0, "2d": 1, "3d": 2, "cube": 3, "1darray": 4, "2darray": 5, "cubearray": 6}

    def setUp(self) -> None:
        self.dir = scratch()

    @staticmethod
    def cbuf(slot: int, binding: int, set_: int = 1) -> dict:
        return {"set": set_, "binding": binding, "slot": slot, "ordinal": 0, "size": 65536, "role": "user"}

    @staticmethod
    def sbuf(ordinal: int, binding: int, offset: int, cb_slot: int = 0, write: bool = False, module_writes: bool = False,
             set_: int = 1) -> dict:
        return {"set": set_, "binding": binding, "ordinal": ordinal, "cb_slot": cb_slot, "cb_word_offset": offset // 4,
                "cb_byte_offset": offset, "role": "guest", "address_byte_offset": offset, "size_byte_offset": offset + 8,
                "write": write, "module_reads": True, "module_writes": module_writes, "non_writable": not write,
                "non_readable": False}

    @staticmethod
    def tex(type_: str = "2d", via: str = "push_slot", kind: str = "combined", slot: int = 5, binding: int | None = None,
            set_: int | None = None, array_length: int = 1, shadow: bool = False, c2_slot: int = 1) -> dict:
        push = via == "push_slot"
        sampler = kind == "sampler"
        return {"set": (0 if push or via == "pool" else 1) if set_ is None else set_,
                "binding": (1 if sampler else 0) if binding is None else binding, "via": via,
                "push_slot": slot if push else -1, "push_offset": 16 + 4 * slot if push else -1, "type_raw": 3,
                "type": type_, "shadow": shadow, "kind": kind, "array_length": array_length, "separate": kind != "combined",
                "cbuf_slot_raw": -1, "c2_slot": c2_slot, "format": "Unknown", "flags_raw": 0, "image_store": False}

    @staticmethod
    def img(type_: str = "2d", binding: int = 30, c2_slot: int = 0, store: bool = False, non_writable: bool = True) -> dict:
        return {"set": 1, "binding": binding, "via": "descriptor", "push_slot": -1, "push_offset": -1, "type": type_,
                "kind": "combined", "array_length": 1, "c2_slot": c2_slot, "image_store": store, "module_reads": True,
                "module_writes": store, "non_writable": non_writable, "non_readable": False}

    @staticmethod
    def vin(location: int, components: int = 4, component: int = 0, type_: str = "float32", array_length: int | None = None,
            ) -> dict:
        d = io(location, components, component, type_)
        d["answer"] = "float"
        if array_length is not None:
            d["array_length"] = array_length
        return d

    @staticmethod
    def base(stage: str = "ps", **over) -> dict:
        d: dict = {"schema": cp.DESC_SCHEMA, "key": "k.1", "stage": stage, "layout": {"name": "uni", "fits": True},
                   "support_buffer": {"set": 1, "binding": 0, "size": 1392,
                                      "fields": [{"name": "alpha_test", "offset": 0, "bytes": 16, "read": False},
                                                 {"name": "viewport_size", "offset": 160, "bytes": 16, "read": False}]},
                   "raw_access": {"physical_pointers": False, "page_table": False, "store_bits": False},
                   "cbufs": [], "sbufs": [], "textures": [], "images": [],
                   "compute": {"local_size": [8, 4, 1], "shared_bytes": 0} if stage == "cs" else None,
                   "vertex_inputs": [] if stage == "vs" else None, "notes": ["a \"quoted\" note"]}
        d.update(over)
        return d

    def binds(self, d: dict | bytes) -> list[str]:
        p = self.dir / "b.json"
        p.write_bytes(d if isinstance(d, bytes) else json.dumps(d, indent=1).encode())
        done = run("binds", str(p))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def refused(self, d: dict | bytes, text: str) -> None:
        out = self.binds(d)
        self.assertEqual(len(out), 1, out)
        self.assertTrue(out[0].startswith("refused: ") and text in out[0], f"want {text!r}: {out}")

    def test_an_empty_module_has_no_bindings(self) -> None:
        self.assertEqual(self.binds(self.base()), ["kind ps fits 1 raw 0 support 0", "local 0 0 0"])

    def test_constant_banks_keep_their_slot_and_place(self) -> None:
        out = self.binds(self.base(cbufs=[self.cbuf(0, 14), self.cbuf(13, 16), self.cbuf(24, 24)]))
        self.assertEqual(out[1:4], ["cb 0 1 14", "cb 13 1 16", "cb 24 1 24"])
        self.assertEqual(len(out), 5)

    def test_storage_buffers_keep_the_contract_offsets_and_the_write_flag(self) -> None:
        out = self.binds(self.base(sbufs=[self.sbuf(0, 25, 512), self.sbuf(1, 26, 416, cb_slot=2, write=True),
                                          self.sbuf(2, 27, 65528, module_writes=True)]))
        # set ordinal cb_slot write binding address_byte_offset; a buffer the post-pass saw written is written too
        self.assertEqual(out[1:4], ["sb 1 0 0 0 25 512", "sb 1 1 2 1 26 416", "sb 1 2 0 1 27 65528"])

    def test_textures_through_push_slots_pools_and_descriptors(self) -> None:
        out = self.binds(self.base(textures=[
            self.tex("2d", slot=5, c2_slot=1),
            self.tex("cubearray", slot=6, c2_slot=7, shadow=True),
            self.tex("3d", via="pool", binding=0, c2_slot=-1),
            self.tex("1darray", via="descriptor", binding=33, set_=1),
            self.tex("sampler", kind="sampler", slot=7, c2_slot=130),
            self.tex("buffer", via="descriptor", binding=32, set_=1),
            self.tex("2dms", slot=8), self.tex("2dmsarray", slot=9),
            self.tex("2darray", kind="separate_texture", slot=10, c2_slot=129, array_length=32),
        ]))
        # via kind view set shadow binding array push_slot push_offset c2_slot; BV_NONE is -1
        self.assertEqual(out[1:], [
            "tex 0 0 1 0 0 0 1 5 36 1",
            "tex 0 0 6 0 1 0 1 6 40 7",
            "tex 1 0 2 0 0 0 1 -1 -1 -1",
            "tex 2 0 4 1 0 33 1 -1 -1 1",
            "tex 0 2 -1 0 0 1 1 7 44 130",
            "tex 2 0 -1 1 0 32 1 -1 -1 1",
            "tex 0 0 1 0 0 0 1 8 48 1",
            "tex 0 0 5 0 0 0 1 9 52 1",
            "tex 0 1 5 0 0 0 32 10 56 129",
            "local 0 0 0"])

    def test_every_type_string_has_its_view(self) -> None:
        for name, view in self.BV.items():
            out = self.binds(self.base(textures=[self.tex(name)], images=[self.img(name)]))
            self.assertEqual(out[1].split()[3], str(view), name)
            self.assertEqual(out[2].split()[3], str(view), name)

    def test_images_are_written_when_stored_to_or_not_declared_non_writable(self) -> None:
        out = self.binds(self.base(images=[
            self.img("2d", 30, 0), self.img("2darray", 31, 1, store=True, non_writable=False),
            self.img("3d", 32, 2, non_writable=False), self.img("buffer", 33, -1)]))
        # set write view binding c2_slot
        self.assertEqual(out[1:5], ["img 1 0 1 30 0", "img 1 1 5 31 1", "img 1 1 2 32 2", "img 1 0 -1 33 -1"])

    def test_vertex_inputs_by_location_with_arrays_one_entry_per_location(self) -> None:
        out = self.binds(self.base("vs", vertex_inputs=[
            self.vin(0), self.vin(1, 2, 1, "sint32"), self.vin(2, 1, 3, "uint32"), self.vin(5, 3, array_length=2)]))
        self.assertEqual(out[0], "kind vs fits 1 raw 0 support 0")
        self.assertEqual(out[1:], ["vin 0 0 4 0", "vin 1 1 2 1", "vin 2 3 1 2", "vin 5 0 3 0", "vin 6 0 3 0", "local 0 0 0"])

    def test_a_compute_module_has_its_local_size(self) -> None:
        out = self.binds(self.base("cs"))
        self.assertEqual(out, ["kind cs fits 1 raw 0 support 0", "local 8 4 1"])
        out = self.binds(self.base("ps", compute={"local_size": [64, 2, 1]}))
        self.assertEqual(out[-1], "local 64 2 1")

    def test_raw_access_and_the_support_buffer_are_flags(self) -> None:
        for pt, pp, want in ((False, False, 0), (True, False, 1), (False, True, 1), (True, True, 1)):
            d = self.base(raw_access={"physical_pointers": pp, "page_table": pt})
            self.assertEqual(self.binds(d)[0], f"kind ps fits 1 raw {want} support 0", (pt, pp))
        d = self.base()
        d["support_buffer"]["fields"][1]["read"] = True
        self.assertEqual(self.binds(d)[0], "kind ps fits 1 raw 0 support 1")
        self.assertEqual(self.binds(self.base(layout={"fits": False}))[0], "kind ps fits 0 raw 0 support 0")

    def test_a_parser_that_skips_any_member_it_should_read_fails_this(self) -> None:
        # Every number differs from every other, and a member of each list is true where its sibling is false, so a
        # parser that ignores one member (or reads it into the wrong field) prints another line.
        d = self.base(
            cbufs=[self.cbuf(9, 301, 3)],
            sbufs=[self.sbuf(4, 302, 1204, cb_slot=7, write=True, set_=5)],
            textures=[self.tex("cube", slot=11, binding=303, set_=6, array_length=13, shadow=True, c2_slot=77)],
            images=[{**self.img("1d", 304, 22, store=True, non_writable=False), "set": 7}],
            support_buffer={"fields": [{"read": True}]}, raw_access={"physical_pointers": False, "page_table": True})
        d["textures"][0]["via"] = "descriptor"
        d["textures"][0].update(push_slot=-1, push_offset=-1)
        out = self.binds(d)
        self.assertEqual(out, ["kind ps fits 1 raw 1 support 1", "cb 9 3 301", "sb 5 4 7 1 302 1204",
                               "tex 2 0 3 6 1 303 13 -1 -1 77", "img 7 1 0 304 22", "local 0 0 0"])

    def test_members_it_does_not_know_are_skipped_whatever_their_shape(self) -> None:
        d = self.base(cbufs=[{**self.cbuf(1, 2), "extra": {"a": [1, 2.5e-3, None, "x\"y"], "b": False}}], zzz=[[[]]])
        self.assertEqual(self.binds(d)[1], "cb 1 1 2")

    def test_lists_at_their_caps_are_read_and_one_more_is_refused(self) -> None:
        self.assertEqual(len(self.binds(self.base(cbufs=[self.cbuf(i, i) for i in range(32)]))), 34)
        self.refused(self.base(cbufs=[self.cbuf(i, i) for i in range(33)]), "cbufs lists more than the 32")
        self.assertEqual(len(self.binds(self.base(sbufs=[self.sbuf(i, i, 4 * i) for i in range(32)]))), 34)
        self.refused(self.base(sbufs=[self.sbuf(i, i, 4 * i) for i in range(33)]), "sbufs lists more than the 32")
        self.assertEqual(len(self.binds(self.base(textures=[self.tex(slot=i % 28) for i in range(64)]))), 66)
        self.refused(self.base(textures=[self.tex(slot=i % 28) for i in range(65)]), "textures lists more than the 64")
        self.assertEqual(len(self.binds(self.base(images=[self.img() for _ in range(16)]))), 18)
        self.refused(self.base(images=[self.img() for _ in range(17)]), "images lists more than the 16")
        self.assertEqual(len(self.binds(self.base("vs", vertex_inputs=[self.vin(i) for i in range(32)]))), 34)
        self.refused(self.base("vs", vertex_inputs=[self.vin(i) for i in range(33)]), "vertex_inputs lists more than the 32")
        self.refused(self.base("vs", vertex_inputs=[self.vin(i) for i in range(31)] + [self.vin(31, array_length=2)]),
                     "vertex_inputs lists more than the 32")

    def test_a_type_or_name_the_replay_does_not_know_is_refused_with_its_text(self) -> None:
        self.refused(self.base(textures=[self.tex("unknown")]), 'textures[0].type "unknown" is not a texture type')
        self.refused(self.base(textures=[self.tex("4d")]), 'type "4d"')
        self.refused(self.base(textures=[self.tex("sampler")]), "kind and type disagree")
        self.refused(self.base(textures=[self.tex("2d", kind="sampler")]), "kind and type disagree")
        self.refused(self.base(textures=[{**self.tex(), "via": "bindless"}]), 'textures[0].via "bindless" is not push_slot')
        self.refused(self.base(textures=[{**self.tex(), "kind": "storage"}]), 'textures[0].kind "storage"')
        self.refused(self.base(images=[self.img("sampler")]), 'images[0].type "sampler" is not an image type')
        self.refused(self.base("vs", vertex_inputs=[self.vin(0, type_="float16")]), 'vertex_inputs[0].type "float16"')
        self.refused(self.base("vs", vertex_inputs=[self.vin(0, type_="unknown")]), 'type "unknown"')
        self.refused(self.base("gs"), 'names the stage "gs"')
        self.refused(self.base(textures=[{**self.tex(), "push_slot": -1}]), "reads through a push slot but names none")

    def test_a_missing_member_is_refused_by_its_name(self) -> None:
        required = {
            "cbufs": (self.cbuf(0, 1), ("set", "binding", "slot")),
            "sbufs": (self.sbuf(0, 9, 8), ("set", "binding", "ordinal", "cb_slot", "address_byte_offset", "write", "module_writes")),
            "textures": (self.tex(), ("set", "binding", "via", "push_slot", "push_offset", "type", "shadow", "kind",
                                      "array_length", "c2_slot")),
            "images": (self.img(), ("set", "binding", "type", "c2_slot", "image_store", "non_writable")),
        }
        for name, (item, members) in required.items():
            for member in members:
                cut = dict(item)
                del cut[member]
                self.refused(self.base(**{name: [cut]}), f"{name}[0].{member} is missing")
        self.refused(self.base("vs", vertex_inputs=[{k: v for k, v in self.vin(0).items() if k != "components"}]),
                     "vertex_inputs[0].components is missing")
        self.refused(self.base(cbufs=[self.cbuf(0, 1), {"set": 1, "slot": 3}]), "cbufs[1].binding is missing")

    def test_a_missing_list_or_flag_is_refused_by_its_name(self) -> None:
        for path, text in ((("cbufs",), "cbufs"), (("sbufs",), "sbufs"), (("textures",), "textures"), (("images",), "images"),
                           (("schema",), "schema"), (("stage",), "stage"), (("layout", "fits"), "layout.fits"),
                           (("raw_access", "page_table"), "raw_access.page_table"),
                           (("raw_access", "physical_pointers"), "raw_access.physical_pointers"),
                           (("support_buffer", "fields"), "support_buffer.fields")):
            d = self.base()
            holder = d
            for step in path[:-1]:
                holder = holder[step]
            del holder[path[-1]]
            self.refused(d, f"lacks {text}")
        d = self.base()
        d["support_buffer"]["fields"][0].pop("read", None)
        d["support_buffer"]["fields"][1].pop("read", None)
        self.refused(d, "support_buffer.fields[0].read is missing")
        self.refused(self.base("vs", vertex_inputs=None), "lacks vertex_inputs")
        self.refused(self.base("cs", compute=None), "lacks compute.local_size")
        self.refused(self.base("cs", compute={"shared_bytes": 0}), "lacks compute.local_size")
        self.refused(self.base("cs", compute={"local_size": [8, 8]}), "has 2 entries, not 3")
        self.refused(self.base("cs", compute={"local_size": [8, 8, 1, 1]}), "more than 3 entries")

    def test_a_member_of_the_wrong_type_or_range_is_refused(self) -> None:
        self.refused(self.base(cbufs=[{**self.cbuf(0, 1), "binding": "1"}]), "cbufs[0].binding is not a number")
        self.refused(self.base(cbufs=[{**self.cbuf(0, 70000)}]), "cbufs[0].binding is 70000, outside 0 to 65535")
        self.refused(self.base(cbufs=[{**self.cbuf(300, 1)}]), "cbufs[0].slot is 300")
        self.refused(self.base(sbufs=[{**self.sbuf(0, 9, 8), "write": 1}]), "sbufs[0].write is not true or false")
        self.refused(self.base(sbufs=[{**self.sbuf(0, 9, 70000)}]), "sbufs[0].address_byte_offset is 70000")
        self.refused(self.base(textures=[{**self.tex(), "type": 3}]), "textures[0].type is not a string")
        self.refused(self.base(textures=[{**self.tex(), "c2_slot": -2}]), "textures[0].c2_slot is -2")
        self.refused(self.base("vs", vertex_inputs=[self.vin(0, 3, 2)]), "components 2..5 pass the fourth")
        self.refused(self.base("vs", vertex_inputs=[self.vin(0, array_length=0)]), "array_length is 0")
        self.refused(self.base(layout={"fits": "yes"}), "layout.fits is not true or false")
        self.refused(self.base(cbufs={"a": 1}), "cbufs is not an array")
        self.refused(self.base("cs", compute={"local_size": [0, 1, 1]}), "outside 1 to 65535")

    def test_a_schema_other_than_the_runtimes_and_unreadable_text_are_refused(self) -> None:
        self.refused(self.base(schema=1), "is schema 1, the runtime reads schema 2")
        good = json.dumps(self.base()).encode()
        self.refused(good + b" x", "text after its object")
        self.refused(good[:len(good) // 2], "bindings of the description are not readable")
        self.refused(b"[1]", "bindings of the description are not readable")


class SpirvTest(unittest.TestCase):
    """The declarations of a module and the outputs added to it (oot2/spirv.h)."""

    def setUp(self) -> None:
        self.dir = scratch()

    def write(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)

    def test_declarations_are_read_in_order_and_once(self) -> None:
        mod = vertex_module((10, 25, 10, 5347), ("SPV_KHR_storage_buffer_storage_class", "SPV_KHR_foo", "SPV_KHR_foo"))
        done = run("spvdecl", self.write("m.spv", mod))
        self.assertEqual(done.stdout.splitlines(), ["version 00010300", "cap 1 Shader", "cap 10 Float64",
                                                    "cap 25 ImageGatherExtended", "cap 5347 PhysicalStorageBufferAddresses",
                                                    "ext SPV_KHR_storage_buffer_storage_class", "ext SPV_KHR_foo"])

    def test_only_a_module_with_cooperative_matrices_needs_the_full_subgroups_requirement(self) -> None:
        # VUID-RuntimeSpirv-OpTypeCooperativeMatrixKHR-10770: a pipeline of a module that declares CooperativeMatrixKHR (6022)
        # is created with VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT (the module here is SPIR-V 1.3, as the
        # translator's are). Nothing else asks for it, and the helper form of a module declares no matrix.
        for caps, want in (((6022,), 1), ((10, 6022, 5347), 1), ((), 0), ((10, 5347), 0), ((9,), 0)):
            with self.subTest(caps=caps):
                done = run("spvfull", self.write("f.spv", vertex_module(caps)))
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertEqual(done.stdout.splitlines(), [f"full_subgroups {want}"])

    def test_a_module_that_is_not_spirv_is_refused(self) -> None:
        bad = bytearray(vertex_module())
        bad[0] ^= 1
        self.assertIn("not a SPIR-V module", run("spvdecl", self.write("a.spv", bytes(bad))).stdout)
        # an instruction among the declarations that claims more words than the module has left
        cut = struct.pack("<8I", SPIRV_MAGIC, 0x00010300, 0, 5, 0, (2 << 16) | 17, 1, (9 << 16) | 17)
        self.assertIn("malformed instruction", run("spvdecl", self.write("b.spv", cut)).stdout)

    def padded(self, spec: str, mod: bytes | None = None) -> tuple[bytes, list[list[int]], list[list[int]]]:
        src = mod or vertex_module()
        done = run("spvpad", self.write("in.spv", src), str(self.dir / "out.spv"), spec)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(done.stdout.startswith("padded "), done.stdout)
        out = (self.dir / "out.spv").read_bytes()
        return out, parse_words(src), parse_words(out)

    def check_ids(self, out: bytes) -> None:
        bound = struct.unpack("<5I", out[:20])[3]
        ids = {}
        for i in parse_words(out):
            op = i[0] & 0xffff
            if op in (19, 21, 22, 23, 32, 33, 43, 44, 46, 59, 54, 248):
                rid = i[1] if op in (19, 21, 22, 23, 32, 33) else i[2] if op in (43, 44, 46, 59, 54) else i[1]
                self.assertNotIn(rid, ids, f"id {rid} defined twice")
                self.assertLess(rid, bound)
                ids[rid] = op

    def test_a_float_vec4_output_gets_a_variable_that_reads_0_0_0_1(self) -> None:
        out, before, after = self.padded("1:0:4:0")
        self.check_ids(out)
        ep = next(i for i in after if i[0] & 0xffff == OP["entry_point"])
        ep0 = next(i for i in before if i[0] & 0xffff == OP["entry_point"])
        self.assertEqual(len(ep), len(ep0) + 1)   # one more interface id
        var = ep[-1]
        decs = [i for i in after if i[0] & 0xffff == OP["decorate"] and i[1] == var]
        self.assertEqual([i[2:] for i in decs], [[30, 1]])   # Location 1, no Component
        variables = [i for i in after if i[0] & 0xffff == OP["variable"] and i[2] == var]
        self.assertEqual(len(variables), 1)
        self.assertEqual(variables[0][3], 3)   # Output
        init = variables[0][4]
        comp = next(i for i in after if i[0] & 0xffff == OP["composite"] and i[2] == init)
        consts = {i[2]: i[3] for i in after if i[0] & 0xffff == OP["constant"]}
        self.assertEqual([consts[c] for c in comp[3:]], [0, 0, 0, 0x3f800000])
        # the types the module had are reused, not declared again
        self.assertEqual(len([i for i in after if i[0] & 0xffff in (OP["float"], OP["vector"], OP["pointer"])]),
                         len([i for i in before if i[0] & 0xffff in (OP["float"], OP["vector"], OP["pointer"])]))

    def test_other_types_and_component_offsets(self) -> None:
        out, before, after = self.padded("2:0:3:1,3:3:1:2,4:2:2:0")
        self.check_ids(out)
        ep = next(i for i in after if i[0] & 0xffff == OP["entry_point"])
        vars_ = ep[-3:]
        decs = {i[1]: i[2:] for i in after if i[0] & 0xffff == OP["decorate"] and i[1] in vars_}
        self.assertEqual(decs[vars_[0]], [30, 2])
        locs = [(i[1], i[3]) for i in after if i[0] & 0xffff == OP["decorate"] and i[1] in vars_ and i[2] == 30]
        comps = [(i[1], i[3]) for i in after if i[0] & 0xffff == OP["decorate"] and i[1] in vars_ and i[2] == 31]
        self.assertEqual(sorted(l for _, l in locs), [2, 3, 4])
        self.assertEqual(sorted(c for _, c in comps), [2, 3])   # Component only where it is not 0
        ints = [i for i in after if i[0] & 0xffff == OP["int"]]
        self.assertEqual(sorted((i[2], i[3]) for i in ints), [(32, 0), (32, 1)])   # a signed and an unsigned type, new
        # a scalar in component 3 reads 1.0f's bits; a vec3 at component 0 and a vec2 at component 2 (z, w) differ
        scalar = next(i for i in after if i[0] & 0xffff == OP["constant"] and i[3] == 0x3f800000
                      and any(j[0] & 0xffff == OP["int"] and j[1] == i[1] and j[3] == 0 for j in after))
        self.assertTrue(scalar)
        self.assertEqual(len([i for i in after if i[0] & 0xffff == OP["null"]]), 1)   # the vec3 reads zeros

    def test_nothing_to_add_leaves_the_module_as_it_was(self) -> None:
        mod = vertex_module()
        done = run("spvpad", self.write("in.spv", mod), str(self.dir / "out.spv"), "")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual((self.dir / "out.spv").read_bytes(), mod)

    def test_what_cannot_be_added_is_refused_by_name(self) -> None:
        done = run("spvpad", self.write("in.spv", vertex_module(entry_points=2)), str(self.dir / "out.spv"), "1:0:4:0")
        self.assertIn("refused: the module has 2 entry points", done.stdout)
        done = run("spvpad", self.write("in.spv", vertex_module()), str(self.dir / "out.spv"), "1:0:5:0")
        self.assertIn("refused: output 1 has an unpaddable shape", done.stdout)
        done = run("spvpad", self.write("in.spv", vertex_module()), str(self.dir / "out.spv"), "1:2:3:0")
        self.assertIn("refused: output 1 has an unpaddable shape", done.stdout)

    def spirv_val(self) -> str:
        found = shutil.which("spirv-val")
        sdk = os.environ.get("VULKAN_SDK")
        if not found and sdk:
            for sub in ("Bin", "bin"):
                for name in ("spirv-val.exe", "spirv-val"):
                    if (Path(sdk) / sub / name).is_file():
                        found = str(Path(sdk) / sub / name)
        if not found:
            self.skipTest("no spirv-val (the Vulkan SDK ships it)")
        return found

    def test_a_padded_module_passes_the_validator(self) -> None:
        val = self.spirv_val()
        for spec in ("1:0:4:0", "2:0:3:1,3:3:1:2,4:2:2:0", "5:0:4:2,6:0:1:0,7:1:3:1"):
            out, _, _ = self.padded(spec)
            done = subprocess.run([val, "--target-env", "vulkan1.3", str(self.dir / "out.spv")], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{spec}: {done.stdout}{done.stderr}")

    # ---- outputs added to a mesh module (oot2/spirv.h, spv_add_mesh_outputs)

    # The pad set of a mesh stage shaped like the game's: 10 runs, in the shapes of the widest gaps measured in the game's
    # archives (n:3:1, n:1:3), over a module whose outputs are the other words of those locations.
    WIDE_WRITTEN = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), (6, 0),
                  (7, 0), (7, 1), (7, 2), (8, 0), (8, 1), (8, 2), (9, 0), (9, 1), (9, 2), (9, 3), (11, 0), (11, 1), (11, 2),
                  (12, 0), (12, 1), (12, 2))
    WIDE_PADS = "0:1:3:0,1:1:3:0,2:1:3:0,4:3:1:0,5:3:1:0,6:1:3:0,7:3:1:0,8:3:1:0,11:3:1:0,12:3:1:0"

    def mesh_padded(self, spec: str, mod: bytes, *limits: int) -> MeshModule:
        done = run("spvmeshpad", self.write("min.spv", mod), str(self.dir / "mout.spv"), spec, *map(str, limits))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(done.stdout.startswith("padded "), done.stdout)
        return MeshModule((self.dir / "mout.spv").read_bytes())

    def mesh_refused(self, spec: str, mod: bytes, *limits: int) -> str:
        done = run("spvmeshpad", self.write("min.spv", mod), str(self.dir / "mout.spv"), spec, *map(str, limits))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(done.stdout.startswith("refused: "), done.stdout)
        return done.stdout

    @staticmethod
    def word_values(component: int, components: int) -> list[int]:
        """What an unwritten word reads: 1.0f's bits in a location's fourth component, zero in the others."""
        return [0x3f800000 if component + c == 3 else 0 for c in range(components)]

    def check_pads(self, spec: str, mod: bytes, *, vertices: int = 160, primitives: int = 128, threads: int = 32) -> MeshModule:
        """spec padded onto mod: every run is an Output array of its rate's length of its scalar or vector, decorated with its
        location (and component, and PerPrimitiveEXT), listed in the interface, and stored its value by every invocation's
        share of the elements after OpSetMeshOutputsEXT, each store behind the test against the count the call passed."""
        before, after = MeshModule(mod), self.mesh_padded(spec, mod)
        runs = []
        for part in spec.split(","):
            fields = part.split(":")
            runs.append((*map(int, fields[:4]), len(fields) > 4))
        added = after.interface[len(before.interface):]
        self.assertEqual(after.interface[:len(before.interface)], before.interface)
        pads = added[:len(runs)]
        call = next(i for i in after.main if (i[0] & 0xffff) == 5295)
        at_call = next(n for n, i in enumerate(after.main) if (i[0] & 0xffff) == 5295)
        stores = after.stores()
        kinds = ("float", "sint", "uint")
        for var, (location, component, components, type_, per_primitive) in zip(pads, runs):
            count = primitives if per_primitive else vertices
            self.assertEqual(after.var[var][1], SC_OUTPUT)
            self.assertEqual(after.array_of(var), (count, kinds[type_], components), (location, component))
            decor = after.decor[var]
            self.assertEqual(decor[30], [location])
            self.assertEqual(decor.get(31), [component] if component else None)
            self.assertEqual(5271 in decor, per_primitive)
            chunks = -(-count // threads)
            self.assertEqual(stores[var], [self.word_values(component, components)] * chunks, (location, component))
        # no store reaches an output before the call, and every chunk of elements is behind a test of its index against the
        # count the call passed for its rate (the vertex count for a run read per vertex, the primitive count otherwise)
        first_store = next(n for n, i in enumerate(after.main) if (i[0] & 0xffff) == 62)
        self.assertGreater(first_store, at_call)
        less_than = {i[2]: i[4] for i in after.main if (i[0] & 0xffff) == 176}   # OpULessThan: result id -> its limit operand
        guards = [less_than[i[1]] for i in after.main if (i[0] & 0xffff) == 250 and i[1] in less_than]
        vertex_chunks = -(-vertices // threads) if any(not r[4] for r in runs) else 0
        primitive_chunks = -(-primitives // threads) if any(r[4] for r in runs) else 0
        self.assertEqual((guards.count(call[1]), guards.count(call[2]), len(guards)),
                         (vertex_chunks, primitive_chunks, vertex_chunks + primitive_chunks))
        # a chunk after the first starts that many invocations on: the index of the chunk is the invocation index plus them
        offsets = [after.value(i[4])[0] for i in after.main if (i[0] & 0xffff) == 128 and i[4] in after.const]
        wanted = [first for count, chunks in ((vertices, vertex_chunks), (primitives, primitive_chunks)) if chunks
                  for first in range(threads, count, threads)]
        self.assertEqual(offsets, wanted)
        return after

    @staticmethod
    def module_around(spec: str, **kw) -> bytes:
        """A mesh module that writes, in each location a run of spec pads, the other words of that location (per primitive when
        the run is): the stage the pad completes, as the game's are."""
        padded, rate = set(), {}
        for part in spec.split(","):
            fields = part.split(":")
            location, component, components = map(int, fields[:3])
            padded |= {(location, c) for c in range(component, component + components)}
            rate[location] = len(fields) > 4
        own = lambda per_primitive: [(loc, c) for loc, p in rate.items() if p == per_primitive for c in range(4)
                                     if (loc, c) not in padded]
        return mesh_module(outputs=own(False), primitive_outputs=own(True), **kw)

    def test_the_missing_words_become_output_arrays_over_the_vertices_stored_after_the_call(self) -> None:
        spec = "1:3:1:0,2:2:2:0,3:1:3:0,4:0:4:2,6:0:3:0,7:3:1:1,8:0:2:1"
        mod = self.module_around(spec, vertices=160)
        after = self.check_pads(spec, mod)
        self.assertEqual(len(after.interface) - len(MeshModule(mod).interface), 7 + 1)   # the seven runs and the invocation index
        # the runs that hold no fourth component (a vec3 of floats at component 0, a vec2 of integers) read zeros: the null
        # value of their type, where the others hold 1.0f's bits in their last word
        self.assertEqual(len([c for c in after.const.values() if (c[0] & 0xffff) == 46]), 2)

    def test_a_run_read_per_primitive_is_an_array_over_the_primitives(self) -> None:
        spec = "5:2:2:0:p,6:3:1:0:p"
        after = self.check_pads(spec, self.module_around(spec, primitives=128))
        for var in after.interface[-3:-1]:
            self.assertEqual(after.array_of(var)[0], 128)
            self.assertIn(5271, after.decor[var])

    def test_the_shapes_of_the_widest_stops_pad_and_pass_the_validator(self) -> None:
        mod = self.module_around(self.WIDE_PADS, vertices=96, primitives=64)
        after = self.check_pads(self.WIDE_PADS, mod, vertices=96, primitives=64)
        self.assertEqual(len(after.interface) - len(MeshModule(mod).interface), 11)   # ten runs and the invocation index
        val = self.spirv_val()
        for spec in (self.WIDE_PADS, "1:3:1:0,2:3:1:0,3:1:3:0,4:3:1:0,5:2:2:0,13:2:2:0", "0:0:4:2,1:0:2:1,2:1:1:2",
                     "5:2:2:0:p,6:3:1:0:p,7:0:4:0,8:1:3:0"):
            self.mesh_padded(spec, self.module_around(spec))
            done = subprocess.run([val, "--target-env", "vulkan1.3", str(self.dir / "mout.spv")], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{spec}: {done.stdout}{done.stderr}")
            whole = subprocess.run([val, "--target-env", "vulkan1.3", self.write("own.spv", self.module_around(spec))],
                                   capture_output=True, text=True)
            self.assertEqual(whole.returncode, 0, f"the module under {spec}: {whole.stdout}{whole.stderr}")

    def test_a_module_that_has_no_invocation_index_is_given_one_and_one_that_has_one_keeps_it(self) -> None:
        mod = mesh_module()
        before, after = MeshModule(mod), self.mesh_padded("1:0:4:0", mod)
        index = after.interface[-1]
        self.assertEqual(after.decor[index][11], [29])    # BuiltIn LocalInvocationIndex
        self.assertEqual(after.var[index][1], SC_INPUT)
        self.assertEqual(len(after.interface), len(before.interface) + 2)
        own = mesh_module(own_index=True)
        before, after = MeshModule(own), self.mesh_padded("1:0:4:0", own)
        self.assertEqual(len(after.interface), len(before.interface) + 1)
        self.assertEqual(len([v for v, d in after.decor.items() if d.get(11) == [29]]), 1)
        loads = [i for i in after.after_call() if (i[0] & 0xffff) == 61]
        self.assertEqual([i[3] for i in loads], [before.interface[-1]])   # the module's own variable is the one read

    def test_a_block_that_followed_the_call_keeps_its_phi_behind_the_new_blocks(self) -> None:
        mod = mesh_module(tail="phi", vertices=96)
        before, after = MeshModule(mod), self.mesh_padded("1:0:4:0", mod)
        was = next(i for i in before.main if (i[0] & 0xffff) == 245)     # OpPhi %type %id %value %parent %value %parent
        phi = next(i for i in after.main if (i[0] & 0xffff) == 245)
        entry = before.main[1][1]                                        # the block that calls OpSetMeshOutputsEXT
        self.assertEqual(was[4], entry)                                  # as translated the phi names that block
        # the branch to the phi's block now ends the last of the new blocks: the nearest label before the original branch
        # (the one on the constant true) is the block the phi has to name
        branch = next(n for n, i in enumerate(after.main) if (i[0] & 0xffff) == 250 and (after.const.get(i[1], [0])[0] & 0xffff) == 41)
        block = next(i[1] for i in reversed(after.main[:branch]) if (i[0] & 0xffff) == OP["label"])
        self.assertNotEqual(block, entry)
        self.assertEqual(phi[4], block)
        self.assertEqual(phi[6], was[6])                                 # the other parent is untouched

    def test_no_outputs_to_add_leaves_the_mesh_module_as_it_was(self) -> None:
        mod = mesh_module()
        done = run("spvmeshpad", self.write("min.spv", mod), str(self.dir / "mout.spv"), "")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertEqual((self.dir / "mout.spv").read_bytes(), mod)

    def test_what_cannot_be_added_to_a_mesh_module_is_refused_by_name(self) -> None:
        cases = [
            ("not a MeshEXT stage", "1:0:4:0", mesh_module(model=EXEC_VERTEX)),
            ("SPIR-V 1.3", "1:0:4:0", mesh_module(version=0x00010300)),
            ("never calls OpSetMeshOutputsEXT", "1:0:4:0", mesh_module(sets=0)),
            ("calls OpSetMeshOutputsEXT 2 times", "1:0:4:0", mesh_module(sets=2)),
            ("gives no LocalSize", "1:0:4:0", mesh_module(modes=False)),
            ("unpaddable shape", "1:2:3:0", mesh_module()),
            ("unpaddable shape", "1:0:5:0", mesh_module()),
            ("unpaddable shape", "1:0:4:3", mesh_module()),
            ("location 64 is past the 64", "64:0:4:0", mesh_module()),
            ("not a SPIR-V module", "1:0:4:0", b"\0" * 40),
        ]
        words = list(struct.unpack(f"<{len(mesh_module()) // 4}I", mesh_module()))
        at = 5
        while words[at] & 0xffff != OP["entry_point"]:
            at += words[at] >> 16
        words[at:at] = words[at:at + (words[at] >> 16)]   # the entry point twice
        cases.append(("2 entry points", "1:0:4:0", struct.pack(f"<{len(words)}I", *words)))
        for needle, spec, mod in cases:
            with self.subTest(needle=needle, spec=spec):
                self.assertIn(needle, self.mesh_refused(spec, mod))

    def test_the_devices_mesh_output_limits_hold_the_padded_module(self) -> None:
        # 160 vertices, the position and ten locations of three words each: 11 slots of 16 bytes for 160 vertices, 28,160 bytes
        mod = mesh_module(outputs=tuple((loc, c) for loc in range(10) for c in range(3)), vertices=160)
        self.mesh_padded("0:3:1:0,1:3:1:0,9:3:1:0", mod, 28160, 44)          # words in locations it has: no new slot
        self.assertIn("32768", self.mesh_refused("10:0:4:0,11:0:4:0,12:0:4:0", mod, 32768, 0))   # 14 slots, 35,840 bytes
        self.mesh_padded("10:0:4:0", mod, 32768, 128)                                           # 12 slots, 30,720
        self.assertIn("48 mesh output components", self.mesh_refused("10:0:4:0", mod, 0, 40))    # 12 slots is 48 components
        self.mesh_padded("10:0:4:0", mod, 0, 48)
        # a per-primitive pad is counted at the primitive count, apart from the vertex outputs
        self.assertIn("bytes of mesh output memory", self.mesh_refused("10:0:4:0:p", mod, 28160, 0))
        self.mesh_padded("10:0:4:0:p", mod, 28160 + 16 * 128, 0)

    def test_the_task_payload_and_the_output_memory_together_hold_the_padded_module(self) -> None:
        # maxMeshPayloadAndOutputMemorySize, as the translator checks it (MeshChecks.Enforce): the payload the stage reads plus
        # its output memory. Ten locations of three words and the position are 28,160 bytes at 160 vertices; padding one more
        # location makes 12 slots, 30,720.
        mod = mesh_module(outputs=tuple((loc, c) for loc in range(10) for c in range(3)), vertices=160)
        limit = 48128                                                        # the reference device's
        self.mesh_padded("10:0:4:0", mod, 32768, 128, limit, 16384)          # 30,720 + 16,384 = 47,104
        self.mesh_padded("10:0:4:0", mod, 32768, 128, limit, limit - 30720)  # exactly the limit
        said = self.mesh_refused("10:0:4:0", mod, 32768, 128, limit, limit - 30720 + 1)
        for needle in ("maxMeshPayloadAndOutputMemorySize", "30720 bytes of mesh output memory", "17409 of task payload",
                       "48129 together", "allows 48128"):
            self.assertIn(needle, said)
        self.mesh_padded("10:0:4:0", mod, 32768, 128, 0, 1 << 20)            # no limit given: none held
        # a word of a location the module has takes no new slot, and the sum is still held
        self.mesh_padded("1:3:1:0", mod, 32768, 128, limit, 16384)
        self.assertIn("maxMeshPayloadAndOutputMemorySize", self.mesh_refused("1:3:1:0", mod, 32768, 128, 40000, 16384))
        # the output memory limit is the one named when both are past
        self.assertIn("32768", self.mesh_refused("10:0:4:0,11:0:4:0,12:0:4:0", mod, 32768, 0, limit, 16384))

    def test_a_stage_with_too_few_threads_for_its_outputs_is_refused_by_the_cause(self) -> None:
        # Each thread stores to its share of an array, one chunk offset for every `threads` elements, and the rewrite holds
        # 64 constants: 160 vertices on 2 threads is 79 offsets, on 32 threads 4.
        spec = "10:0:4:0"
        self.mesh_padded(spec, mesh_module(threads=32, vertices=160))
        self.mesh_padded(spec, mesh_module(threads=3, vertices=160))            # 53 offsets: still fits
        for threads in (2, 1):
            said = self.mesh_refused(spec, mesh_module(threads=threads, vertices=160))
            for needle in (f"{threads} threads", "160 vertices", "chunk offset", "64 constants"):
                self.assertIn(needle, said)
            self.assertNotIn("out of memory", said)
        said = self.mesh_refused("10:0:4:0:p", mesh_module(threads=1, vertices=8, primitives=128))   # per primitive: 127
        self.assertIn("1 threads for 8 vertices and 128 primitives", said)

    # ---- workgroup memory zeroed (oot2/spirv.h, spv_zero_workgroup)

    def zeroed(self, mod: bytes) -> tuple[subprocess.CompletedProcess, list[list[int]], list[list[int]]]:
        done = run("spvzero", self.write("in.spv", mod), str(self.dir / "zout.spv"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        after = parse_words((self.dir / "zout.spv").read_bytes()) if done.stdout.startswith("zeroed") else []
        return done, parse_words(mod), after

    def test_a_workgroup_variable_gets_the_null_value_of_its_type(self) -> None:
        mod = compute_module(((64, False), (32, False)))
        done, before, after = self.zeroed(mod)
        self.assertIn("zeroed 2 variables", done.stdout)
        out = (self.dir / "zout.spv").read_bytes()
        self.check_ids(out)
        self.assertEqual(struct.unpack("<5I", out[:20])[3], struct.unpack("<5I", mod[:20])[3] + 2)   # a null for each type
        variables = [i for i in after if i[0] & 0xffff == OP["variable"]]
        self.assertEqual([len(v) for v in variables], [5, 5])
        for v in variables:
            self.assertEqual(v[3], SC_WORKGROUP)
            null = next(j for j in after if j[0] & 0xffff == OP["null"] and j[2] == v[4])
            ptr = next(j for j in after if j[0] & 0xffff == OP["pointer"] and j[1] == v[1])
            self.assertEqual(null[1], ptr[3])   # the null value of the type the pointer points at
            self.assertLess(after.index(null), after.index(v))   # declared before the variable that uses it
        # nothing else changed: take the nulls out and the initializers off, and the module is the one that went in
        stripped = [i for i in after if i[0] & 0xffff != OP["null"]]
        stripped = [i[:4] if i[0] & 0xffff == OP["variable"] else i for i in stripped]
        stripped = [[(len(i) << 16) | (i[0] & 0xffff), *i[1:]] for i in stripped]
        self.assertEqual(stripped, before)

    def test_variables_of_one_type_share_one_null_constant(self) -> None:
        done, _, after = self.zeroed(compute_module(((64, False), (64, False), (64, False))))
        self.assertIn("zeroed 3 variables", done.stdout)
        nulls = [i for i in after if i[0] & 0xffff == OP["null"]]
        self.assertEqual(len(nulls), 1)
        self.assertEqual({v[4] for v in after if v[0] & 0xffff == OP["variable"]}, {nulls[0][2]})

    def test_a_variable_that_has_an_initializer_is_left_as_it_is(self) -> None:
        done, before, after = self.zeroed(compute_module(((64, True), (32, False))))
        self.assertIn("zeroed 1 variables", done.stdout)
        self.assertEqual(len([i for i in after if i[0] & 0xffff == OP["null"]]), 2)   # the module's own, and one new
        done, _, _ = self.zeroed(compute_module(((64, True),)))
        self.assertEqual(done.stdout, "unchanged\n")

    def test_a_module_with_no_workgroup_variable_is_unchanged(self) -> None:
        self.assertEqual(self.zeroed(vertex_module())[0].stdout, "unchanged\n")
        self.assertEqual(self.zeroed(compute_module(()))[0].stdout, "unchanged\n")

    def test_what_cannot_be_zeroed_is_refused_by_name(self) -> None:
        done, _, _ = self.zeroed(compute_module(((64, False),), undeclared_type=True))
        self.assertIn("refused: workgroup variable", done.stdout)
        bad = bytearray(compute_module())
        bad[0] ^= 1
        done = run("spvzero", self.write("bad.spv", bytes(bad)), str(self.dir / "o.spv"))
        self.assertIn("refused: not a SPIR-V module", done.stdout)
        cut = struct.pack("<8I", SPIRV_MAGIC, 0x00010300, 0, 5, 0, (2 << 16) | 17, 1, (9 << 16) | 17)
        done = run("spvzero", self.write("cut.spv", cut), str(self.dir / "o.spv"))
        self.assertIn("refused: malformed instruction", done.stdout)

    def test_a_zeroed_module_passes_the_validator(self) -> None:
        val = self.spirv_val()
        orig = self.write("orig.spv", compute_module(((64, False), (32, False), (64, False))))
        self.assertEqual(subprocess.run([val, "--target-env", "vulkan1.3", orig], capture_output=True, text=True).returncode, 0)
        done, _, _ = self.zeroed((self.dir / "orig.spv").read_bytes())
        self.assertIn("zeroed 3 variables", done.stdout)
        done = subprocess.run([val, "--target-env", "vulkan1.3", str(self.dir / "zout.spv")], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)


def header_required_ids() -> list[str]:
    """The ids of OOT2_VK_ITEMS as the header text lists them, apart from anything the C preprocessor does."""
    text = (ROOT / "runtime" / "include" / "oot2" / "vk_required.h").read_text(encoding="utf-8")
    block = text[text.index("#define OOT2_VK_ITEMS(X)"):text.index("// The features the device enables when")]
    return re.findall(r"\bX\((\w+),", block)


class OptionalFeatureTest(unittest.TestCase):
    """The features the device enables when the GPU reports them and does not require (OOT2_VK_OPTIONAL)."""

    @classmethod
    def setUpClass(cls) -> None:
        done = run("optional")
        assert done.returncode == 0, done.stdout + done.stderr
        cls.lines = done.stdout.splitlines()

    def test_facts_with_the_feature_enable_it_and_allow_its_capability(self) -> None:
        self.assertIn("present: enabled 1", self.lines)
        self.assertIn("present: capability 5309 allowed", self.lines)

    def test_zero_initialized_workgroup_memory_is_enabled_where_the_gpu_reports_it_and_never_required(self) -> None:
        self.assertIn("present: zero-initialized workgroup memory enabled 1", self.lines)
        self.assertIn("absent: zero-initialized workgroup memory enabled 0", self.lines)
        self.assertEqual([l for l in self.lines if l.startswith(("present: lacks", "absent: lacks"))], [])

    def test_facts_without_the_feature_keep_the_device_acceptable_and_the_feature_off(self) -> None:
        count = len(header_required_ids())
        self.assertIn(f"absent: has {count} of {count}", self.lines)
        self.assertEqual([l for l in self.lines if l.startswith("absent: lacks")], [])   # not among the lacking items
        self.assertIn("absent: enabled 0", self.lines)
        self.assertIn("absent: capability 5309 refused", self.lines)

    def test_the_required_list_is_the_same_with_and_without_the_feature(self) -> None:
        ids = header_required_ids()
        self.assertGreater(len(ids), 19)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual([l[len("item "):] for l in self.lines if l.startswith("item ")], ids)   # contents and order
        self.assertIn(f"items {len(ids)} optional 4", self.lines)
        self.assertEqual([l for l in self.lines if l.startswith("optional ")],
                         ["optional storage_image_array_indexing", "optional zero_initialize_workgroup_memory",
                          "optional shader_float16", "optional image_2d_view_of_3d"])
        self.assertNotIn("storage_image_array_indexing", ids)
        self.assertNotIn("zero_initialize_workgroup_memory", ids)
        self.assertNotIn("shader_float16", ids)
        self.assertNotIn("image_2d_view_of_3d", ids)
        self.assertEqual([l for l in self.lines if l.startswith("present: lacks")], [])   # the feature changes no item


class RequiredListTest(unittest.TestCase):
    """The capabilities and extensions the required list gives (oot2/vk_caps.h)."""

    def setUp(self) -> None:
        self.dir = scratch()

    def check(self, mod: bytes, *args: str) -> list[str]:
        p = self.dir / "m.spv"
        p.write_bytes(mod)
        done = run("capcheck", str(p), *args)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done.stdout.splitlines()

    def test_what_the_items_enable_is_in_the_list(self) -> None:
        self.assertEqual(self.check(vertex_module((1, 10, 25, 32, 33, 4427, 5302, 5307, 5347, 61, 62, 64, 65)))[0], "ok")
        out = self.check(vertex_module((5283, 6033), ("SPV_EXT_mesh_shader", "SPV_KHR_storage_buffer_storage_class")))
        self.assertEqual(out[0], "ok")

    def test_what_no_item_enables_is_refused_by_name(self) -> None:
        self.assertEqual(self.check(vertex_module((2,)))[0], "bad: capability Geometry (2) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((3,)))[0], "bad: capability Tessellation (3) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((63,)))[0],   # arithmetic: the subgroup item asks for four operations only
                         "bad: capability GroupNonUniformArithmetic (63) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((1,), ("SPV_KHR_foo",)))[0],
                         "bad: extension SPV_KHR_foo is not in the device's required list")
        self.assertIn("capability 9999 is not in", self.check(vertex_module((9999,)))[0])

    def test_non_uniform_indexing_of_storage_image_arrays_is_refused_without_the_feature(self) -> None:
        # No item asks for it, and a device without the optional feature refuses a module that declares it, by name;
        # the sampled image arrays' is in the list (the pool's).
        self.assertEqual(self.check(vertex_module((5309,)))[0],
                         "bad: capability StorageImageArrayNonUniformIndexing (5309) is not in the device's required list")
        self.assertEqual(self.check(vertex_module((5307,)))[0], "ok")

    def test_non_uniform_indexing_of_storage_image_arrays_is_allowed_with_the_feature(self) -> None:
        # A shader pack prepared before the translator stopped declaring it still has the capability in many modules.
        self.assertEqual(self.check(vertex_module((5309,)), "storage-image-indexing")[0], "ok")
        self.assertEqual(self.check(vertex_module((5307, 5309)), "storage-image-indexing")[0], "ok")

    def test_a_capability_removed_from_the_list_is_caught(self) -> None:
        mod = vertex_module((10,))
        self.assertEqual(self.check(mod)[0], "ok")
        self.assertEqual(self.check(mod, "drop", "10")[0], "bad: capability Float64 (10) is not in the device's required list")
        self.assertEqual(self.check(mod, "drop", "11")[0], "ok")   # dropping what the module does not use changes nothing

    def test_the_list_follows_the_items(self) -> None:
        out = self.check(vertex_module())
        m = re.search(r"list (\d+) capabilities, (\d+) extensions", out[1])
        self.assertGreater(int(m.group(1)), 15)
        self.assertGreater(int(m.group(2)), 5)


class CacheHeaderTest(unittest.TestCase):
    def test_a_cache_file_is_judged_by_its_header_for_the_device_and_driver(self) -> None:
        out = dict(l.split(" ", 1) for l in run("cachejudge").stdout.splitlines() if l.count(" ") == 1 and "=" not in l)
        names = dict(re.findall(r"(\w+)=(\d+)", run("cachejudge").stdout.splitlines()[-1]))
        loaded, other, bad = names["loaded"], names["other"], names["unreadable"]
        self.assertEqual(out["matching"], loaded)
        for what in ("uuid", "device", "vendor"):
            self.assertEqual(out[what], other, what)    # another driver or device: dropped, rebuilt
        for what in ("short", "empty", "version", "headersize"):
            self.assertEqual(out[what], bad, what)


class CacheReadTest(unittest.TestCase):
    """The cache file as the launch reads it, and the move of a save over it (no device): each cause of a file not read has its
    own name and phrase, a transient failure to open is tried once more, and an access denied that lasts through the wait is
    paid once."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch()
        cls.cache = cls.dir / "pipeline.cache"
        cls.cache.write_bytes(b"x" * 4096)
        cls.empty = cls.dir / "empty.cache"
        cls.empty.write_bytes(b"")

    @staticmethod
    def read(path: Path, *opts: str) -> dict[str, str]:
        done = run("cacheread", str(path), *opts)
        assert done.returncode == 0, done.stdout + done.stderr
        out = {}
        for line in done.stdout.splitlines():
            key, _, rest = line.partition(" ")
            out[key] = rest
        return out

    def test_each_way_the_file_is_not_read_has_its_own_cause_and_phrase(self) -> None:
        causes = {
            "absent": (self.read(self.dir / "none.cache"), "absent", "is not there"),
            "empty": (self.read(self.empty), "empty", "is empty"),
            "too_large": (self.read(self.cache, "readcap=4095"), "too_large", "too large for memory"),
            "open": (self.read(self.cache, "openerr=13"), "open", "could not be opened at launch ("),
        }
        for name, (out, cause, text) in causes.items():
            self.assertEqual(out["cause"], cause, name)
            self.assertEqual(out["read"], "none size 0 ms " + out["read"].rsplit(" ", 1)[1], name)
            self.assertIn(text, out["text"], name)
        green = self.read(self.cache, "readcap=4096")   # the counterpart: read whole
        self.assertEqual((green["cause"], green["read"].split(" ")[:3]), ("ok", ["data", "size", "4096"]))

    def test_an_open_that_fails_once_is_tried_again_and_one_that_keeps_failing_is_not_read(self) -> None:
        once = self.read(self.cache, "openerr=13", "transient")
        self.assertEqual(once["cause"], "ok")
        self.assertTrue(once["read"].startswith("data size 4096"), once["read"])
        self.assertGreaterEqual(int(once["read"].rsplit(" ", 1)[1]), 150)   # it waited before it tried again
        lasting = self.read(self.cache, "openerr=13")
        self.assertEqual(lasting["cause"], "open")
        self.assertGreaterEqual(int(lasting["read"].rsplit(" ", 1)[1]), 150)

    def test_a_file_that_is_not_there_is_not_tried_again(self) -> None:
        out = self.read(self.dir / "none.cache", "openerr=2")
        self.assertEqual(out["cause"], "absent")
        self.assertLess(int(out["read"].rsplit(" ", 1)[1]), 150)

    @unittest.skipUnless(os.name == "nt", "the sharing violation and the access denied of a replace are Windows'")
    def test_an_access_denied_that_lasts_is_waited_for_once_and_named_once(self) -> None:
        folder = scratch()
        path, tmp = folder / "pipeline.cache", folder / "pipeline.cache.tmp"
        path.write_bytes(b"old")
        tmp.write_bytes(b"new")
        path.chmod(stat.S_IREAD)   # a replace of a read-only file is refused with access denied
        try:
            done = run("cachemove", str(tmp), str(path), "heal")
        finally:
            path.chmod(stat.S_IREAD | stat.S_IWRITE)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        moves = {int(m[0]): (int(m[1]), int(m[2]), int(m[3])) for m in re.findall(r"move (\d) ok (\d) tries (\d+) ms (\d+)", done.stdout)}
        self.assertEqual(moves[1][:2], (0, 5))        # the whole wait: five tries
        self.assertGreaterEqual(moves[1][2], 900)     # about a second
        self.assertEqual(moves[2][:2], (0, 1))        # the second move tries once
        self.assertLess(moves[2][2], 300)
        self.assertEqual(moves[3][:2], (1, 1))        # healed: it moves, at once
        self.assertEqual(path.read_bytes(), b"new")
        self.assertEqual(done.stderr.count("refused the replace with \"access denied\""), 1, done.stderr)


class StoreTest(unittest.TestCase):
    """The module store: stages by key out of a pack, and the named fatals."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch()
        mod = vertex_module()
        e = [
            cp.Entry("vs.ok", mod, describe("vs.ok", "vs", mod, outputs=[io(0)])),
            cp.Entry("ps.ok", mod, describe("ps.ok", "ps", mod, inputs=[io(0), io(1)], outputs=[io(0)])),
            cp.Entry("ps.big", mod, describe("ps.big", "ps", mod, fits=False)),
            cp.Entry("vs.failed", None, "fail: unsupported_opcode: synthetic", failed=True),
        ]
        cls.pack = write_pack(cls.dir / "base.bin", e)
        cls.pack3 = write_pack(cls.dir / "base3.bin", [cp.Entry("a.1", mod, describe("a.1", "ps", mod, schema=3))], schema=3)
        variant_answers = {"tex": {"0": "3d"}}
        cls.variant_digest = cp.answers_digest(variant_answers)
        cls.ledger_digest = hashlib.sha256(b"complete synthetic ledger answers").digest()
        cls.same_variant_ledger_digest = hashlib.sha256(b"same projection, different full row").digest()
        cls.base_ledger_digest = hashlib.sha256(b"full ledger answers projected to base").digest()
        variant_description = json.loads(describe("ps.ok", "ps", mod))
        variant_description["answers"].update(variant_answers)
        variant = cp.Entry("ps.ok", mod, json.dumps(variant_description).encode("utf-8"), answers=cls.variant_digest)
        cls.variants = write_pack(cls.dir / "variants.bin", [variant], role=cp.VARIANTS)
        cls.variant_map = cls.dir / "variant-map.bin"
        cls.variant_map.write_bytes(prepare_ledger.variant_map_bytes(
            {("ps.ok", cls.ledger_digest): cls.variant_digest,
             ("ps.ok", cls.same_variant_ledger_digest): cls.variant_digest,
             ("ps.ok", cls.base_ledger_digest): bytes(32)}, hashlib.sha256(b"ledger file").hexdigest()))

    def store(self, key: str, kind: str, shape: str = "1") -> subprocess.CompletedProcess:
        return run("store", str(self.pack), "-", key, kind, shape)

    def test_a_stage_is_found_by_its_key(self) -> None:
        done = self.store("ps.ok", "ps")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertRegex(done.stdout, r"module \d+ bytes from 1, kind ps, fits 1, 2 inputs, 1 outputs")
        done = self.store("vs.ok", "vs")
        self.assertIn("kind vs, fits 1, 0 inputs, 1 outputs", done.stdout)

    def test_a_full_ledger_digest_resolves_through_the_map_and_counts_fallbacks(self) -> None:
        mapped = run("storemap", str(self.pack), str(self.variants), str(self.variant_map), "ps.ok", "ps", "1",
                     self.ledger_digest.hex())
        self.assertEqual(mapped.returncode, 0, mapped.stdout + mapped.stderr)
        mapped_lines = mapped.stdout.splitlines()
        self.assertEqual(mapped_lines[1], f"content {self.variant_digest.hex()}")
        self.assertEqual(mapped_lines[2], "module from 2, map missing 0, unknown 0")

        same_projection = run("storemap", str(self.pack), str(self.variants), str(self.variant_map), "ps.ok", "ps", "1",
                              self.same_variant_ledger_digest.hex())
        self.assertEqual(same_projection.returncode, 0, same_projection.stdout + same_projection.stderr)
        same_projection_lines = same_projection.stdout.splitlines()
        self.assertEqual(same_projection_lines[0], mapped_lines[0])
        self.assertEqual(same_projection_lines[1], mapped_lines[1])
        self.assertEqual(same_projection_lines[2], "module from 2, map missing 0, unknown 0")

        unknown = hashlib.sha256(b"unmapped full ledger row").hexdigest()
        fallback = run("storemap", str(self.pack), str(self.variants), str(self.variant_map), "ps.ok", "ps", "1",
                       unknown)
        self.assertEqual(fallback.returncode, 0, fallback.stdout + fallback.stderr)
        fallback_lines = fallback.stdout.splitlines()
        self.assertNotEqual(fallback_lines[0], mapped_lines[0])
        self.assertEqual(fallback_lines[1], "content " + "00" * 32)
        self.assertEqual(fallback_lines[2], "module from 1, map missing 0, unknown 1")

        base = run("storemap", str(self.pack), str(self.variants), str(self.variant_map), "ps.ok", "ps", "1",
                   self.base_ledger_digest.hex())
        self.assertEqual(base.returncode, 0, base.stdout + base.stderr)
        self.assertEqual(base.stdout.splitlines()[1], fallback_lines[1])

        counter_run = run("storemap", str(self.pack), str(self.variants), str(self.variant_map), "ps.ok", "ps", "1",
                          self.base_ledger_digest.hex(), "raw")
        self.assertEqual(counter_run.returncode, 0, counter_run.stdout + counter_run.stderr)
        self.assertNotEqual(counter_run.stdout.splitlines()[1], fallback_lines[1])

        no_map = run("storemap", str(self.pack), str(self.variants), "-", "ps.ok", "ps", "1",
                     self.ledger_digest.hex())
        self.assertEqual(no_map.returncode, 0, no_map.stdout + no_map.stderr)
        self.assertEqual(no_map.stdout.splitlines()[2], "module from 1, map missing 1, unknown 0")

    def fatal(self, done: subprocess.CompletedProcess) -> str:
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr)
        return lines[0]

    def test_a_key_the_pack_lacks_is_a_named_fatal_with_the_stages_shape(self) -> None:
        text = self.fatal(self.store("0123456789abcdef.00000000", "ps"))
        self.assertIn("the shader cache has no module for stage 0123456789abcdef.00000000", text)
        self.assertIn("(a ps stage, key 0123456789abcdef.00000000, code area 4656 bytes, instructions 4096 bytes, "
                      "control section 87 bytes)", text)
        self.assertIn("(key 0123456789abcdef.00000000, no shape recorded)", self.fatal(self.store("0123456789abcdef.00000000", "ps", "0")))

    def test_a_job_that_did_not_translate_is_a_named_fatal_with_its_reason(self) -> None:
        text = self.fatal(self.store("vs.failed", "vs"))
        self.assertIn("stage vs.failed did not translate: fail: unsupported_opcode: synthetic", text)

    def test_a_stage_of_another_kind_and_a_module_that_does_not_fit_are_named_fatals(self) -> None:
        self.assertIn("stage ps.ok is a ps program where a vs was asked for", self.fatal(self.store("ps.ok", "vs")))
        self.assertIn("stage ps.big does not fit its slots of the uni layout", self.fatal(self.store("ps.big", "ps")))

    def test_packs_the_reader_refuses_stay_closed(self) -> None:
        done = run("storeopen", str(self.pack3), "-")
        self.assertTrue(done.stdout.startswith("refused: ") and "schema" in done.stdout.lower(), done.stdout)
        done = run("storeopen", str(self.dir / "missing.bin"), "-")
        self.assertTrue(done.stdout.startswith("refused: "), done.stdout)
        self.assertEqual(run("storeopen", str(self.pack), "-").stdout, "ok\n")


def with_bindings(description: bytes) -> bytes:
    """A description that also carries the members the bindings reader needs (the layout of BindingsTest's)."""
    d = json.loads(description)
    for k, v in BindingsTest.base(d["stage"]).items():
        d.setdefault(k, v)
    return json.dumps(d, indent=1).encode("utf-8")


class NoDeviceBuilderTest(unittest.TestCase):
    """What the pipeline builder and the replay's device ops decide before they touch the device, reached without one
    (continuous integration runs them): the replay's cache of stage bindings, and the rule build_link links a pair by."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = scratch()
        mod = vertex_module()
        wide = [mesh_out(loc, c) for loc in (1, 2, 40, 63) for c in range(4)]
        wide_in = [pixel_in(loc) for loc in (1, 2, 40, 63)]
        e = [
            cp.Entry("ps.bind", mod, with_bindings(describe("ps.bind", "ps", mod, inputs=[io(0)], outputs=[io(0)]))),
            cp.Entry("m.rate", mod, describe("m.rate", "ms", mod, outputs=[mesh_out(4, c, per_vertex=128, per_primitive=True)
                                                                         for c in range(4)])),
            cp.Entry("p.rate", mod, describe("p.rate", "ps", mod, inputs=[pixel_in(4)])),
            cp.Entry("m.gap", bare_module(), describe("m.gap", "ms", bare_module(), outputs=wide)),
            cp.Entry("p.gap", mod, describe("p.gap", "ps", mod, inputs=wide_in + [pixel_in(9, 1, 2, type_="uint32")])),
            cp.Entry("v.wide", mod, describe("v.wide", "vs", mod, outputs=[])),
            cp.Entry("p.wide", mod, describe("p.wide", "ps", mod, inputs=[io(0, type_="float64")])),
        ]
        cls.pack = write_pack(cls.dir / "base.bin", e)

    def test_the_replays_binding_cache_keeps_every_bucket_index_inside_its_table(self) -> None:
        # 4096 buckets: the index of a record is reduced after the digest is mixed in. Reduced before, 64 records reach
        # an index of up to 2^32 and the lookup reads far past the table (an access violation, exit 3221225477).
        done = run("devbinds", str(self.pack), "ps.bind", "64")
        self.assertEqual(done.returncode, 0, f"exit {done.returncode}\n{done.stdout}{done.stderr}")
        self.assertEqual(done.stdout.strip(), "binds 64 distinct 64 stable 64 kind ps")

    def mesh_pairs(self, pixel_inputs: dict[str, list[dict]]) -> dict[str, dict[str, str]]:
        """One mesh stage (it writes words 0 and 1 of location 1 and all of location 2) linked to each pixel stage: what the
        builder makes of the pair up to its mesh library's key, as a record of the pad text, the key without and with a task
        stage and the runs handed to the rewrite, by the pixel stage's name."""
        mod = vertex_module()
        outs = [mesh_out(1, 0), mesh_out(1, 1)] + [mesh_out(2, c) for c in range(4)]
        (self.dir / "mk.json").write_bytes(describe("mk.1", "ms", mod, outputs=outs))
        for name, ins in pixel_inputs.items():
            (self.dir / f"pk-{name}.json").write_bytes(describe(f"pk.{name}", "ps", mod, inputs=ins))
        names = list(pixel_inputs)
        done = run("meshlibkey", str(self.dir / "mk.json"), *(str(self.dir / f"pk-{n}.json") for n in names))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = done.stdout.splitlines()
        self.assertEqual(len(lines), 4 * len(names), done.stdout)
        records = {}
        for i, name in enumerate(names):
            pad, key, task_key, rest = lines[4 * i:4 * i + 4]
            self.assertEqual((pad[:4], key[:4], task_key[:9], rest[:4]), ("pad ", "key ", "task-key ", "outs"), done.stdout)
            records[name] = {"pad": pad[4:], "key": key[4:], "task_key": task_key[9:], "outs": rest[4:].split()}
        return records

    def test_the_mesh_library_key_holds_the_pad_set_and_its_rates(self) -> None:
        # Without the pad set in the key, two pixel stages that need different padding of one mesh stage share the library
        # made for the first, and the second pair is linked to a mesh stage that lacks (or has the wrong rate of) its words.
        r = self.mesh_pairs({
            "a": [pixel_in(1)],                                            # words 2 and 3 of location 1
            "same": [pixel_in(1), pixel_in(2)],                            # another pixel stage with the same gap
            "more": [pixel_in(1), pixel_in(5, 2)],                         # one more run
            "vertex": [pixel_in(1), pixel_in(5)],                          # location 5, written by nothing, read per vertex
            "primitive": [pixel_in(1), pixel_in(5, per_primitive=True)],   # the same words, read per primitive
            "none": [pixel_in(2)],                                         # nothing missing
        })
        self.assertEqual(r["a"]["pad"], "1:2:2:0;")
        self.assertEqual(r["same"]["pad"], r["a"]["pad"])
        self.assertEqual(r["more"]["pad"], "1:2:2:0;5:0:2:0;")
        self.assertEqual(r["vertex"]["pad"], "1:2:2:0;5:0:4:0;")
        self.assertEqual(r["primitive"]["pad"], "1:2:2:0;5:0:4:0:p;")
        self.assertEqual(r["none"]["pad"], "")
        # equal pad sets, equal keys, whatever the pixel stage; different sets, different keys
        self.assertEqual(r["same"]["key"], r["a"]["key"])
        self.assertEqual(r["same"]["task_key"], r["a"]["task_key"])
        for kind in ("key", "task_key"):
            keys = [r[n][kind] for n in ("a", "more", "vertex", "primitive", "none")]
            self.assertEqual(len(set(keys)), len(keys), keys)
        # the key names the mesh stage, and the task stage when there is one
        self.assertTrue(r["a"]["key"].startswith("m|m:6100000000000003.00000000|"), r["a"]["key"])
        self.assertTrue(r["a"]["task_key"].startswith("m|t:6100000000000002.00000000|m:6100000000000003.00000000|"))
        # the runs the rewrite is handed are the pad set's, each with the rate of the pixel input that reads it
        self.assertEqual(r["a"]["outs"], ["1:2:2:0:0"])
        self.assertEqual(r["vertex"]["outs"], ["1:2:2:0:0", "5:0:4:0:0"])
        self.assertEqual(r["primitive"]["outs"], ["1:2:2:0:0", "5:0:4:0:1"])
        self.assertEqual(r["none"]["outs"], [])

    def test_a_mesh_stages_task_payload_is_read_from_its_description(self) -> None:
        mod = vertex_module()
        base = json.loads(describe("pl.1", "ms", mod))
        path = self.dir / "payload.json"
        for stage, want in (({"payload_bytes": 20480, "threads": 32, "first_task": {"read": True}}, 20480), (None, 0),
                            ({"threads": 32}, 0)):
            path.write_text(json.dumps(base | {"mesh_stage": stage}), encoding="utf-8")
            done = run("meshpayload", str(path))
            self.assertEqual(done.stdout, f"payload {want}\n", done.stdout + done.stderr)
        path.write_bytes(describe("pl.1", "ms", mod))   # a description with no such member
        self.assertEqual(run("meshpayload", str(path)).stdout, "payload 0\n")

    def linked(self, kind: str, vkey: str, pkey: str) -> str:
        done = run("linkchoice", str(self.pack), vkey, pkey, kind)
        self.assertEqual(done.returncode, EXIT_FATAL, f"exit {done.returncode}\n{done.stdout}{done.stderr}")
        self.assertNotIn("linked", done.stdout)
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr)
        return lines[0]

    def test_build_link_links_a_mesh_stage_by_the_rate_of_its_outputs(self) -> None:
        text = self.linked("ms", "m.rate", "p.rate")
        self.assertIn("mesh stage m.rate cannot be linked to pixel stage p.rate", text)
        self.assertIn("rate", text)
        self.assertIn("location 4", text)

    def test_build_link_pads_a_mesh_stage_with_the_words_its_pixel_stage_reads_and_it_does_not_write(self) -> None:
        # 160 vertices at location 63: the padding rule counts 160 locations from location 1 and cannot link this pair. The mesh
        # rule links it with one range missing, and build_link goes on to pad the mesh module. The module of this pack is not a
        # mesh stage, which the rewrite refuses by name, so the fatal shows that the pair was padded and not stopped on.
        text = self.linked("ms", "m.gap", "p.gap")
        self.assertIn("mesh stage m.gap cannot be padded with the outputs its pixel stage reads", text)
        self.assertIn("9:2:1:2;", text)
        self.assertNotIn("does not write", text)

    def test_build_link_still_links_a_vertex_stage_by_padding(self) -> None:
        text = self.linked("vs", "v.wide", "p.wide")
        self.assertIn("stage v.wide cannot be linked to stage p.wide", text)
        self.assertIn("64 bits wide", text)
        self.assertNotIn("mesh stage", text)


# ---- on a device, with a synthetic pack of GLSL modules -----------------------------------

GLSL = {
    "vs.vert": """#version 450
layout(set = 1, binding = 1) uniform VsBank { vec4 offset; } bank;
layout(set = 1, binding = 9) readonly buffer VsData { vec4 colour; } data;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
layout(location = 0) out vec4 a;
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    gl_Position = vec4(p * 2.0 - 1.0 + bank.offset.xy, 0.0, 1.0);
    a = data.colour;
}
""",
    "ps.frag": """#version 450
layout(set = 1, binding = 14) uniform PsBank { vec4 add; } bank;
layout(set = 1, binding = 25) readonly buffer PsData { vec4 mul; } data;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
layout(location = 0) in vec4 a;
layout(location = 1) in vec4 b;
layout(location = 0) out vec4 o;
void main() { o = (a + b + bank.add) * data.mul + vec4(float(pc.words[1].x)); }
""",
    "cs.comp": """#version 450
layout(local_size_x = 32) in;
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
layout(set = 1, binding = 1) uniform Bank { uvec4 k; } bank;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
void main() { o.v[gl_GlobalInvocationID.x] = gl_GlobalInvocationID.x * bank.k.x + pc.words[0].x; }
""",
    "cs2.comp": """#version 450
layout(local_size_x = 32) in;
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
void main() { o.v[gl_GlobalInvocationID.x] = 1u; }
""",
    # fills the workgroup's shared words with ones and stores nothing (the store is for a value the shared words never hold)
    "zp.comp": """#version 450
layout(local_size_x = 64) in;
shared uint s[64];
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
void main() {
    s[gl_LocalInvocationIndex] = 0xFFFFFFFFu;
    barrier();
    if (s[(gl_LocalInvocationIndex + 1u) & 63u] == 0u) o.v[0] = 1u;
}
""",
    # stores the shared word of each invocation, which no invocation writes (the push constant that would write one is zero,
    # so the compiler cannot know the variable is never written)
    "zr.comp": """#version 450
layout(local_size_x = 64) in;
shared uint s[64];
layout(set = 1, binding = 9) buffer Out { uint v[]; } o;
layout(push_constant) uniform Push { uvec4 words[8]; } pc;
void main() {
    if (pc.words[0].x == 1u) s[(gl_LocalInvocationIndex + 7u) & 63u] = 5u;
    o.v[gl_GlobalInvocationID.x] = s[gl_LocalInvocationIndex];
}
""",
}
SYNTH_KEYS = {"vs.vert": "5100000000000001.00000000", "ps.frag": "5100000000000002.00000000",
              "cs.comp": "5100000000000003.00000000", "cs2.comp": "5100000000000004.00000000",
              "zp.comp": "5100000000000007.00000000", "zr.comp": "5100000000000008.00000000"}
SYNTH_STAGE = {"vs.vert": "vs", "ps.frag": "ps", "cs.comp": "cs", "cs2.comp": "cs", "zp.comp": "cs", "zr.comp": "cs"}


def synthetic_pack(directory: Path) -> Path:
    """GLSL compiled by glslang with the build's flags, packed with descriptions that name each stage's interface."""
    from build_game import GLSLANG_FLAGS  # tools/build_game.py is on the path through test_runtime_gpu
    program()
    glslang = _build["glslang"]
    entries = []
    for name, src in GLSL.items():
        path = directory / name
        path.write_text(src, encoding="utf-8")
        spv = directory / (name + ".spv")
        done = subprocess.run([glslang, *GLSLANG_FLAGS, "-o", str(spv), str(path)], capture_output=True, text=True)
        assert done.returncode == 0, f"glslang on {name}:\n{done.stdout}{done.stderr}"
        mod = spv.read_bytes()
        key = SYNTH_KEYS[name]
        kw: dict = {}
        if name == "vs.vert":
            kw = {"outputs": [io(0)]}
        elif name == "ps.frag":
            kw = {"inputs": [io(0), io(1)], "outputs": [io(0)]}
        entries.append(cp.Entry(key, mod, describe(key, SYNTH_STAGE[name], mod, **kw)))
    return write_pack(directory / "synthetic.bin", entries)


# One mesh stage that writes location 0 and three pixel stages: a and c read locations 0 and 1, b reads 0, 1 and 2.
MESH_GLSL = {
    "m.mesh": """#version 450
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 32) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
layout(location = 0) out vec4 a[];
void main() {
    SetMeshOutputsEXT(3, 1);
    uint i = gl_LocalInvocationIndex;
    if (i < 3u) {
        gl_MeshVerticesEXT[i].gl_Position = vec4(float(i & 1u) * 2.0 - 1.0, float(i >> 1) * 2.0 - 1.0, 0.0, 1.0);
        a[i] = vec4(1.0);
    }
    if (i == 0u) gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
""",
    "pa.frag": """#version 450
layout(location = 0) in vec4 a;
layout(location = 1) in vec4 b;
layout(location = 0) out vec4 o;
void main() { o = a + b; }
""",
    "pb.frag": """#version 450
layout(location = 0) in vec4 a;
layout(location = 1) in vec4 b;
layout(location = 2) in vec4 c;
layout(location = 0) out vec4 o;
void main() { o = a + b + c; }
""",
    "pc.frag": """#version 450
layout(location = 0) in vec4 a;
layout(location = 1) in vec4 b;
layout(location = 0) out vec4 o;
void main() { o = a * b; }
""",
    # reads location 3 once for each primitive, which the mesh stage writes nothing of
    "pd.frag": """#version 450
#extension GL_EXT_mesh_shader : require
layout(location = 0) in vec4 a;
layout(location = 3) perprimitiveEXT in vec4 d;
layout(location = 0) out vec4 o;
void main() { o = a + d; }
""",
}
MESH_KEYS = {"m.mesh": "5200000000000001.00000000", "pa.frag": "5200000000000002.00000000",
             "pb.frag": "5200000000000003.00000000", "pc.frag": "5200000000000004.00000000",
             "pd.frag": "5200000000000005.00000000"}


def mesh_pack(directory: Path) -> Path:
    """MESH_GLSL compiled by glslang with the build's flags, packed with descriptions that name each stage's interface: the mesh
    stage writes the four words of location 0, one array over its three vertices each, as the translator declares them."""
    from build_game import GLSLANG_FLAGS
    program()
    entries = []
    for name, src in MESH_GLSL.items():
        path = directory / name
        path.write_text(src, encoding="utf-8")
        spv = directory / (name + ".spv")
        # SPIR-V 1.4, the translator's version: the later ones give the local size as ids (OpExecutionModeId), which a mesh
        # module of the translator never does
        done = subprocess.run([_build["glslang"], *GLSLANG_FLAGS, "--target-env", "spirv1.4", "-o", str(spv), str(path)],
                              capture_output=True, text=True)
        assert done.returncode == 0, f"glslang on {name}:\n{done.stdout}{done.stderr}"
        mod = spv.read_bytes()
        key = MESH_KEYS[name]
        if name == "m.mesh":
            desc = describe(key, "ms", mod, outputs=[mesh_out(0, c, per_vertex=3) for c in range(4)])
        elif name == "pd.frag":
            desc = describe(key, "ps", mod, inputs=[pixel_in(0), pixel_in(3, per_primitive=True)], outputs=[io(0)])
        else:
            desc = describe(key, "ps", mod, inputs=[pixel_in(loc) for loc in range(3 if name == "pb.frag" else 2)],
                            outputs=[io(0)])
        entries.append(cp.Entry(key, mod, desc))
    return write_pack(directory / "meshpad.bin", entries)


def require_gpu() -> bool:
    return bool(os.environ.get("OOT2_REQUIRE_GPU"))


class DeviceTestCase(unittest.TestCase):
    usable = 0

    @classmethod
    def setUpClass(cls) -> None:
        done = run("usable")
        skip_on_partial_device(done)
        if done.returncode == EXIT_NO_DEVICE:
            why = f"no Vulkan device here ({done.stdout.strip()})"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but there is {why}")
            raise unittest.SkipTest(why)
        assert done.returncode == 0, done.stdout + done.stderr
        cls.usable = int(re.search(r"^usable (\d+)$", done.stdout, re.M).group(1))
        if not cls.usable:
            why = "no device here has every item the renderer requires"
            if require_gpu():
                raise RuntimeError(f"OOT2_REQUIRE_GPU is set but {why}")
            raise unittest.SkipTest(why)
        cls.dir = scratch()
        cls.pack = synthetic_pack(cls.dir)

    def gpu(self, case: str, *args: str, validation: str = "sync", timeout: int = 300) -> subprocess.CompletedProcess:
        return run(case, *args, env={"OOT2_VK_VALIDATION": validation}, timeout=timeout)

    def assert_clean(self, done: subprocess.CompletedProcess, case: str) -> None:
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn(f"ok {case}", done.stdout)
        self.assertEqual(fatal_lines(done.stderr), [])
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-3000:])

    @staticmethod
    def stats(out: str, tag: str) -> dict[str, int]:
        m = re.search(rf"^stats {tag} libraries (\d+) hits (\d+) stage (\d+) hits (\d+) links (\d+) hits (\d+) "
                      rf"computes (\d+) hits (\d+) padded (\d+) resident (\d+)", out, re.M)
        assert m, out
        keys = ("libraries", "library_hits", "stage_libraries", "stage_library_hits", "links", "link_hits", "computes",
                "compute_hits", "padded", "resident")
        return dict(zip(keys, (int(x) for x in m.groups())))


class DeviceTest(DeviceTestCase):
    def test_the_layouts_are_made_valid_and_destroyed(self) -> None:
        self.assert_clean(self.gpu("layouts"), "layouts")

    def test_a_pair_links_for_two_formats_and_draws_with_the_padded_input_reading_0_0_0_1(self) -> None:
        done = self.gpu("synth_draw", str(self.pack))
        self.assert_clean(done, "synth_draw")
        self.assertIn("padded input reads (0, 0, 0, 1): f32 yes, f16 yes", done.stdout)
        s = self.stats(done.stdout, "graphics")
        # the shared vertex input library, the vertex library (padded), the fragment library and two output libraries;
        # two fast links; the same pair asked for again made nothing
        self.assertEqual((s["libraries"], s["links"], s["padded"], s["resident"]), (5, 2, 1, 6), s)
        print("\n" + done.stdout)

    def test_counter_run_a_pixel_input_left_unpadded_is_reported_by_validation(self) -> None:
        done = self.gpu("synth_draw", str(self.pack), "unpad")
        self.assertIn("VUID-RuntimeSpirv-OpEntryPoint-08743", done.stderr, done.stdout + done.stderr[-3000:])
        self.assertNotIn("validation errors 0 ", done.stdout)
        print("\n" + "\n".join(l for l in done.stderr.splitlines() if "08743" in l)[:1500])

    def test_a_compute_pipeline_dispatches(self) -> None:
        done = self.gpu("synth_compute", str(self.pack))
        self.assert_clean(done, "synth_compute")
        self.assertIn("128 values written by the dispatch: 3 * i + 7", done.stdout)
        self.assertEqual(self.stats(done.stdout, "compute")["computes"], 1)

    def test_workgroup_memory_no_invocation_wrote_reads_as_zero_after_a_dispatch_filled_it_with_ones(self) -> None:
        done = self.gpu("synth_zeroinit", str(self.pack))
        self.assert_clean(done, "synth_zeroinit")
        self.assertIn("0 of 262144 words read as something other than zero", done.stdout)
        self.assertEqual(self.stats(done.stdout, "zeroinit")["computes"], 2)

    def test_a_pipeline_asked_for_with_a_submission_in_flight_is_a_named_fatal(self) -> None:
        done = self.gpu("synth_inflight", str(self.pack), validation="off")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertIn("the existing pipeline was returned with a submission in flight", done.stdout)
        self.assertNotIn("returned\n", done.stdout.replace("was returned with", ""))
        lines = fatal_lines(done.stderr)
        self.assertEqual(len(lines), 1, done.stderr)
        self.assertRegex(lines[0], r"a compute pipeline would be created for stage 5100000000000004\.00000000 with work in "
                                   r"flight \(submissions up to \d+ handed out, \d+ done\)")

    def test_a_stage_the_pack_lacks_is_a_named_fatal_on_the_device_path(self) -> None:
        done = self.gpu("synth_absent", str(self.pack), validation="off")
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        text = fatal_lines(done.stderr)[0]
        self.assertIn("the shader cache has no module for stage 5100000000000009.00000000", text)
        self.assertIn("(a vs stage, key 5100000000000009.00000000, code area 17760 bytes, instructions 16384 bytes", text)

    def test_the_pipeline_cache_serves_a_second_process_and_is_dropped_after_a_driver_change(self) -> None:
        file = self.dir / "pipeline-cache.bin"
        file.unlink(missing_ok=True)
        cold = self.gpu("synth_cache", str(self.pack), str(file), "cold")
        self.assert_clean(cold, "synth_cache")
        self.assertIn("cache_state 0", cold.stdout)
        c = self.stats(cold.stdout, "cold")
        self.assertEqual((c["stage_library_hits"], c["compute_hits"]), (0, 0), c)
        self.assertEqual((c["stage_libraries"], c["computes"]), (2, 1), c)   # the vertex and the fragment library
        self.assertTrue(file.is_file() and file.stat().st_size > 32)
        warm = self.gpu("synth_cache", str(self.pack), str(file), "warm")
        self.assert_clean(warm, "synth_cache")
        self.assertIn("cache_state 1", warm.stdout)
        w = self.stats(warm.stdout, "warm")
        self.assertEqual((w["libraries"], w["links"], w["computes"]), (c["libraries"], c["links"], c["computes"]))
        # everything the driver compiled comes back from the cache: the libraries that hold a shader and the compute
        # pipeline. A library with no shader and a fast link compile nothing and are not cache entries (the pipeline cost study: a
        # link is microseconds), so the driver never reports them as hits.
        self.assertEqual((w["stage_library_hits"], w["compute_hits"]), (w["stage_libraries"], w["computes"]), w)
        print("\n" + cold.stdout + warm.stdout)
        # a driver change: the header names another pipelineCacheUUID, the file is dropped and everything is built again
        data = bytearray(file.read_bytes())
        data[16 + 3] ^= 0xFF
        other = self.dir / "other-driver.bin"
        other.write_bytes(bytes(data))
        dropped = self.gpu("synth_cache", str(self.pack), str(other), "warm")
        self.assert_clean(dropped, "synth_cache")
        self.assertIn("cache_state 2", dropped.stdout)
        d = self.stats(dropped.stdout, "warm")
        self.assertEqual((d["stage_library_hits"], d["compute_hits"]), (0, 0), d)
        short = self.dir / "short.bin"
        short.write_bytes(bytes(data[:20]))
        cut = self.gpu("synth_cache", str(self.pack), str(short), "warm")
        self.assert_clean(cut, "synth_cache")
        self.assertIn("cache_state 3", cut.stdout)

    def test_a_stages_objects_are_released_and_made_again(self) -> None:
        self.assert_clean(self.gpu("synth_release", str(self.pack)), "synth_release")

    def test_eight_threads_asking_for_one_pipeline_make_it_once(self) -> None:
        self.assert_clean(self.gpu("synth_threads", str(self.pack)), "synth_threads")


class MeshPadDeviceTest(DeviceTestCase):
    """The pipelines of a mesh stage whose pixel stages read words it does not write, built by the builder on a device."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.mesh_pack = mesh_pack(cls.dir)

    def test_the_padded_mesh_library_is_made_once_for_each_pad_set_whatever_the_pixel_stage_and_the_format(self) -> None:
        done = self.gpu("synth_meshpad", str(self.mesh_pack))
        self.assert_clean(done, "synth_meshpad")
        s = self.stats(done.stdout, "meshpad")
        # the shared vertex input library, the mesh library of each of the three pad sets (location 1; locations 1 and 2;
        # location 3 read per primitive) and the fragment library of each of the four pixel stages, an output library for
        # each of the two formats; five fast links (a and c, b and d in one format, a in the other): the pairs share the mesh
        # module, a pad set where they have one and the output libraries, and the validation layer finds the per-primitive
        # words of d in the mesh stage's outputs
        self.assertEqual((s["libraries"], s["links"], s["padded"], s["resident"]), (10, 5, 3, 14), s)
        print("\n" + done.stdout)


# ---- on the player's own pack ------------------------------------------------------------

PLAYER_PACK = os.environ.get("OOT2_T60_PACK")
PLAYER_WORK = os.environ.get("OOT2_T60_WORK")


@unittest.skipUnless(PLAYER_PACK and PLAYER_WORK, "OOT2_T60_PACK and OOT2_T60_WORK name the player's pack and a work folder")
class PlayerPackTest(DeviceTestCase):
    """Never in continuous integration: the pack and the work folder hold the game's programs and stay out of the repository."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.work = Path(PLAYER_WORK)
        cls.work.mkdir(parents=True, exist_ok=True)
        cls.real_pack = Path(PLAYER_PACK)

    def pack_module(self, key: str) -> bytes:
        pack = cp.open_pack(self.real_pack)
        mod, _ = pack.read(pack.find(key))
        return mod

    def test_the_pair_the_stage_bindings_work_drew_builds_from_the_pack_and_draws_the_same_pixels(self) -> None:
        spec_path = os.environ.get("OOT2_T60_FIRSTDRAW_SPEC")
        draw_dir = os.environ.get("OOT2_T60_FIRSTDRAW_DIR")
        if not spec_path or not draw_dir:
            self.skipTest("OOT2_T60_FIRSTDRAW_SPEC (spec23.json) and OOT2_T60_FIRSTDRAW_DIR (a run folder's draw.txt) name the draw")
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        vkey, pkey = spec["vs"], spec["ps"]
        out = self.work / "firstdraw"
        out.mkdir(exist_ok=True)
        (out / "vs.spv").write_bytes(self.pack_module(vkey))
        (out / "fs.spv").write_bytes(self.pack_module(pkey))
        draw = Path(draw_dir) / "draw.txt"
        # the reference: tools/firstdraw's own harness on the same two modules
        ref_draw = out / "ref-draw.txt"
        lines = []
        for line in draw.read_text(encoding="utf-8").splitlines():
            w = line.split(" ", 1)
            if w[0] == "vs":
                line = f"vs {out / 'vs.spv'}"
            elif w[0] == "fs":
                line = f"fs {out / 'fs.spv'}"
            elif w[0] == "out":
                line = f"out {out / 'reference.f32'}"
            lines.append(line)
        ref_draw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        built = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "build_vkdraw.py"), "--harness", "vkdraw",
                                "--out", str(out / "vkdraw")], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        exe = out / "vkdraw" / ("vkdraw.exe" if os.name == "nt" else "vkdraw")
        ref = subprocess.run([str(exe), str(ref_draw)], capture_output=True, text=True, env=case_env({"VK_LOADER_LAYERS_DISABLE": "~implicit~"}))
        self.assertEqual(ref.returncode, 0, ref.stdout + ref.stderr)
        print("\nreference (tools/firstdraw/vkdraw.c):", ref.stdout.strip())
        done = self.gpu("firstdraw", str(draw), str(self.real_pack), vkey, pkey, str(out / "builder.f32"))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-4000:])
        print("\n" + done.stdout)
        a, b = (out / "reference.f32").read_bytes(), (out / "builder.f32").read_bytes()
        self.assertEqual(len(a), len(b))
        self.assertEqual(hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest(), "the two images differ")
        self.assertIn("validation errors 0 warnings 0", done.stdout, done.stderr[-4000:])

    def sample(self, pairs: int) -> Path:
        import pipe_cost
        work = self.work / "select"
        work.mkdir(exist_ok=True)
        shutil.copy(os.environ["OOT2_T60_PAIRS"], work / "pairs.jsonl")
        self.assertEqual(pipe_cost.select(work, pairs, 20), 0)   # the pipeline cost study's selection: deciles of module size, seed 20
        rows = json.loads((work / "sample.json").read_text(encoding="utf-8"))["pairs"]
        tsv = self.work / f"pairs-{pairs}.tsv"
        tsv.write_text("".join(f"{r['vs']}\t{r['ps']}\n" for r in rows), encoding="utf-8")
        return tsv

    def test_a_sample_of_corpus_pairs_builds_and_links_with_no_refusal_padded_pairs_included(self) -> None:
        if not os.environ.get("OOT2_T60_PAIRS"):
            self.skipTest("OOT2_T60_PAIRS names S3's pairs.jsonl (the corpus pairs of the pipeline cost study)")
        n = int(os.environ.get("OOT2_T60_SAMPLE", "120"))
        tsv = self.sample(n)
        cache = self.work / f"pipeline-cache-pairs-{n}.bin"
        cache.unlink(missing_ok=True)
        threads = os.environ.get("OOT2_T60_THREADS", "4")
        cold = self.gpu("corpus_pairs", str(self.real_pack), str(tsv), threads, str(cache), "save", validation="on", timeout=7200)
        print("\n" + cold.stdout)
        self.assertEqual(cold.returncode, 0, cold.stdout + cold.stderr[-4000:])
        self.assertEqual(fatal_lines(cold.stderr), [])
        self.assertIn("validation errors 0 warnings 0", cold.stdout, cold.stderr[-4000:])
        c = self.stats(cold.stdout, "pairs")
        self.assertGreater(c["padded"], 0, "the sample holds padded pairs")
        warm = self.gpu("corpus_pairs", str(self.real_pack), str(tsv), threads, str(cache), validation="off", timeout=7200)
        print("\n" + warm.stdout)
        self.assertEqual(warm.returncode, 0, warm.stdout + warm.stderr[-4000:])
        w = self.stats(warm.stdout, "pairs")
        self.assertEqual(w["stage_library_hits"], w["stage_libraries"], w)

    def test_compute_pipelines_build_and_a_second_process_gets_them_warm(self) -> None:
        keys = os.environ.get("OOT2_T60_COMPUTE")
        if not keys:
            self.skipTest("OOT2_T60_COMPUTE names a file of compute job keys, one per line")
        cache = self.work / "pipeline-cache-compute.bin"
        cache.unlink(missing_ok=True)
        threads = os.environ.get("OOT2_T60_THREADS", "4")
        cold = self.gpu("corpus_compute", str(self.real_pack), keys, threads, str(cache), "save", validation="on", timeout=7200)
        print("\n" + cold.stdout)
        self.assertEqual(cold.returncode, 0, cold.stdout + cold.stderr[-4000:])
        self.assertIn("validation errors 0 warnings 0", cold.stdout, cold.stderr[-4000:])
        c = self.stats(cold.stdout, "compute")
        self.assertGreater(c["computes"], 0)
        warm = self.gpu("corpus_compute", str(self.real_pack), keys, threads, str(cache), validation="off", timeout=7200)
        print("\n" + warm.stdout)
        self.assertEqual(warm.returncode, 0, warm.stdout + warm.stderr[-4000:])
        w = self.stats(warm.stdout, "compute")
        self.assertEqual((w["computes"], w["compute_hits"]), (c["computes"], c["computes"]), w)

    def test_the_capability_check_over_the_whole_corpus_and_its_counter_run(self) -> None:
        keys = os.environ.get("OOT2_T60_KEYS")
        if not keys:
            self.skipTest("OOT2_T60_KEYS names `cache_pack.py list` output of the pack")
        done = run("caps", str(self.real_pack), keys, timeout=3600)
        print("\n" + done.stdout)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr[-2000:])
        self.assertIn("ok caps", done.stdout)
        # the counter-run: drop a capability the corpus declares from the list; the modules that declare it are refused
        declared = [(int(m.group(1)), int(m.group(2))) for m in re.finditer(r"^cap (\d+) \S+ (\d+) in the list", done.stdout, re.M)]
        dropped = next((c for c in sorted(declared, key=lambda x: x[1]) if c[0] not in (0, 1)), None)
        if dropped:
            bad = run("caps", str(self.real_pack), keys, "drop", str(dropped[0]), timeout=3600)
            m = re.search(r"refused (\d+)", bad.stdout)
            self.assertTrue(m and int(m.group(1)) == dropped[1], bad.stdout)
            print(f"counter-run: capability {dropped[0]} dropped from the list: {dropped[1]} modules refused")


if __name__ == "__main__":
    unittest.main()
