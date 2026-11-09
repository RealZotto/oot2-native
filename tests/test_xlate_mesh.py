# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the task and mesh emission of the translator (the mesh emission): a task or mesh program becomes a TaskEXT or
MeshEXT module, not a compute shader.

The programs are assembled here from a few instruction words, never read from game files, and the modules are checked by
structure with a small reader of the SPIR-V (execution model and modes, capabilities, the payload, the declared outputs,
the group count's word), since the CI runners have no spirv-val; where spirv-val is present every module is validated too.

The tests skip when the translator has not been built (py -3 tools/build_xlate.py), as test_xlate_host does.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from test_xlate_host import EXIT, ROOT, SPIRV_VAL, Harness, ampere, bit, mesh_fields, validate
from test_xlate_uni import mov_cbuf, scheduled

sys.path.insert(0, str(ROOT / "tools" / "firstdraw"))
import mesh_programs as mp  # noqa: E402

OP_EXTENSION, OP_ENTRY_POINT, OP_EXECUTION_MODE, OP_CAPABILITY = 10, 15, 16, 17
OP_TYPE_ARRAY, OP_TYPE_POINTER, OP_CONSTANT, OP_FUNCTION, OP_FUNCTION_END = 28, 32, 43, 54, 56
OP_VARIABLE, OP_LOAD, OP_STORE, OP_ACCESS_CHAIN, OP_DECORATE, OP_RETURN = 59, 61, 62, 65, 71, 253
OP_SELECT, OP_CONTROL_BARRIER = 169, 224
OP_EMIT_MESH_TASKS, OP_SET_MESH_OUTPUTS = 5294, 5295
OP_MEMORY_MODEL, OP_COMPOSITE_CONSTRUCT, OP_COMPOSITE_EXTRACT, OP_CONVERT_U_TO_PTR, OP_BITCAST, OP_IADD = 14, 80, 81, 120, 124, 128
OP_FUNCTION_CALL = 57

CAPABILITY_MESH_SHADING = 5283
CAPABILITY_INT64, CAPABILITY_PHYSICAL_STORAGE = 11, 5347
ADDRESSING_PHYSICAL_STORAGE_64, STORAGE_PUSH_CONSTANT, STORAGE_PHYSICAL, STORAGE_FUNCTION = 5348, 9, 5349, 7
BUILTIN_WORKGROUP_ID, MEMORY_ACCESS_ALIGNED = 26, 2
MODEL_TASK, MODEL_MESH = 5364, 5365
STORAGE_OUTPUT, STORAGE_PAYLOAD = 3, 5402
DECORATION_BUILTIN, DECORATION_LOCATION, DECORATION_COMPONENT, DECORATION_PER_PRIMITIVE = 11, 30, 31, 5271
BUILTIN_POSITION, BUILTIN_TRIANGLE_INDICES = 0, 5296
MODE_LOCAL_SIZE, MODE_OUTPUT_VERTICES, MODE_OUTPUT_PRIMITIVES, MODE_OUTPUT_TRIANGLES = 17, 26, 5270, 5298
SPIRV_1_4 = 0x00010400

# ISBEWR (0x927): the data register in bits 32..39 (Rb), the byte address Ra + the offset in bits 40..63, the mode in bits 72..80.
# Task: 0xb8 a payload word. Mesh: 0xf8 an attribute word, 0x88 the primitive count (a 32-bit store at byte 3), 0x80 a map byte.
# ISBERD (0x923): the destination Rd, the same address and mode (0x38 a payload word, 0x78 the workgroup's number).


def isbewr(mode: int, offset: int, value_reg: int) -> bytes:
    return scheduled(ampere(0x927, ra=255, rb=value_reg, low_bits=offset << 40, high_bits=bit(72, mode)))


def isberd(mode: int, rd: int, offset: int) -> bytes:
    return scheduled(ampere(0x923, rd=rd, ra=255, low_bits=offset << 40, high_bits=bit(72, mode)))


# EXIT under P0 (the condition predicate in bits 87..90, here P0 not negated)
EXIT_IF_P0 = struct.pack("<QQ", 0x94D | (7 << 12), 0)

# A task program: the group count (from a constant bank word, so unknown here) to byte 4 of its payload, one record word at 0x10
TASK = [mov_cbuf(2, 3, 0x10), isbewr(0xB8, 4, 2), mov_cbuf(3, 3, 0x14), isbewr(0xB8, 0x10, 3), EXIT]

# A mesh program: the payload word at 0x10 into the position's first component of vertex 0, a bank word into its second line, the
# primitive count (byte 3) and a map byte (the first vertex index, byte 4)
MESH = [mov_cbuf(2, 3, 0x10), isberd(0x38, 4, 0x10), isbewr(0xF8, 0, 4), isbewr(0xF8, 0x80, 2),
        mov_cbuf(5, 3, 0x14), isbewr(0x88, 3, 5), mov_cbuf(6, 3, 0x18), isbewr(0x80, 4, 6), EXIT]

POSITION = [[-1, c] for c in range(4)]


class Module:
    """A SPIR-V module read into its instructions: just what the structure tests ask."""

    def __init__(self, data: bytes):
        words = struct.unpack(f"<{len(data) // 4}I", data)
        assert words[0] == 0x07230203, "not a SPIR-V module"
        self.version = words[1]
        self.insts: list[tuple[int, list[int]]] = []
        i = 5
        while i < len(words):
            count = words[i] >> 16
            self.insts.append((words[i] & 0xFFFF, list(words[i + 1:i + count])))
            i += count
        self.constants = {ops[1]: ops[2] for op, ops in self.insts if op == OP_CONSTANT}
        self.pointers = {ops[0]: (ops[1], ops[2]) for op, ops in self.insts if op == OP_TYPE_POINTER}
        self.arrays = {ops[0]: (ops[1], ops[2]) for op, ops in self.insts if op == OP_TYPE_ARRAY}
        # What defines a result id, for the instructions whose result is the second operand
        self.defs = {ops[1]: (op, ops) for op, ops in self.insts
                     if op in (OP_CONSTANT, OP_VARIABLE, OP_LOAD, OP_ACCESS_CHAIN, OP_SELECT, OP_COMPOSITE_CONSTRUCT, OP_COMPOSITE_EXTRACT,
                               OP_CONVERT_U_TO_PTR, OP_BITCAST, OP_IADD)}
        first = next(n for n, (op, _) in enumerate(self.insts) if op == OP_FUNCTION)
        # A global variable: before the first function; (result type, id, storage class)
        self.variables = {ops[1]: (ops[0], ops[2]) for op, ops in self.insts[:first] if op == OP_VARIABLE}

    def ops(self, code: int) -> list[list[int]]:
        return [ops for op, ops in self.insts if op == code]

    def local_stores(self) -> dict[int, int]:
        """The value stored to each function-local variable that is stored to exactly once: the generator keeps the program's temporaries there."""
        local = {ops[1] for op, ops in self.insts if op == OP_VARIABLE and ops[2] == STORAGE_FUNCTION}
        stored: dict[int, list[int]] = {}
        for op, ops in self.insts:
            if op == OP_STORE and ops[0] in local:
                stored.setdefault(ops[0], []).append(ops[1])
        return {pointer: values[0] for pointer, values in stored.items() if len(values) == 1}

    @staticmethod
    def text(words: list[int]) -> tuple[str, int]:
        """A literal string and the number of words it takes."""
        data = b""
        for n, w in enumerate(words, 1):
            data += struct.pack("<I", w)
            if 0 in struct.pack("<I", w):
                return data.split(b"\0")[0].decode(), n
        raise AssertionError("an unterminated string")

    def capabilities(self) -> set[int]:
        return {ops[0] for ops in self.ops(OP_CAPABILITY)}

    def extensions(self) -> set[str]:
        return {self.text(ops)[0] for ops in self.ops(OP_EXTENSION)}

    def entry(self) -> tuple[int, int, str, list[int]]:
        """(execution model, function id, name, interface) of the one entry point."""
        (ops,) = self.ops(OP_ENTRY_POINT)
        name, used = self.text(ops[2:])
        return ops[0], ops[1], name, ops[2 + used:]

    def modes(self, function: int) -> dict[int, list[int]]:
        return {ops[1]: ops[2:] for ops in self.ops(OP_EXECUTION_MODE) if ops[0] == function}

    def decorations(self, target: int) -> dict[int, list[int]]:
        return {ops[1]: ops[2:] for ops in self.ops(OP_DECORATE) if ops[0] == target}

    def variables_in(self, storage: int) -> list[int]:
        return sorted(v for v, (_, s) in self.variables.items() if s == storage)

    def array_length(self, variable: int) -> int:
        """The length of the array a variable points to."""
        pointee = self.pointers[self.variables[variable][0]][1]
        _, length = self.arrays[pointee]
        return self.constants[length]

    def body(self, function: int) -> list[tuple[int, list[int]]]:
        start = next(n for n, (op, ops) in enumerate(self.insts) if op == OP_FUNCTION and ops[1] == function)
        end = next(n for n in range(start, len(self.insts)) if self.insts[n][0] == OP_FUNCTION_END)
        return self.insts[start + 1:end]


class MeshHarness(Harness):
    def code(self, name: str, instructions: list[bytes]) -> Path:
        path = self.dir / f"{name}.bin"
        path.write_bytes(b"".join(instructions))
        return path

    def job(self, key: str, stage: str, instructions: list[bytes], **fields) -> dict:
        return {"key": key, "stage": stage, "arch": "sm86", "code": str(self.code(key, instructions)), "c1": "",
                **mesh_fields(stage, **fields)}

    def translate(self, job: dict) -> tuple[Module, dict]:
        key = job["key"]
        got = self.run_jobs([job])[key]
        self.assertEqual((got.status, got.message), ("ok", ""))
        if SPIRV_VAL:
            v = validate.validate_one(SPIRV_VAL, key, self.out / f"{key}.spv")
            self.assertTrue(v.ok, v.message)
        return Module((self.out / f"{key}.spv").read_bytes()), json.loads((self.out / f"{key}.res.json").read_text(encoding="utf-8"))

    def refuse(self, job: dict, kind: str, needle: str) -> None:
        got = self.run_jobs([job])[job["key"]]
        self.assertEqual(got.status, "fail", got)
        self.assertTrue(got.message.startswith(kind + ":"), got.message)
        self.assertIn(needle, got.message)

    def chains_on(self, module: Module, body, variable: int) -> set[int]:
        """The ids of the access chains that start at the variable."""
        return {ops[1] for op, ops in body if op == OP_ACCESS_CHAIN and ops[2] == variable}


class TaskEmissionTest(MeshHarness):
    def test_a_task_writes_its_payload_and_launches_its_mesh_workgroups(self):
        module, res = self.translate(self.job("t1", "ts", TASK))
        model, function, name, interface = module.entry()
        self.assertEqual((model, name), (MODEL_TASK, "main"))
        self.assertEqual(res["exec_model"], "TaskEXT")
        self.assertEqual(module.version, SPIRV_1_4)
        self.assertIn(CAPABILITY_MESH_SHADING, module.capabilities())
        self.assertIn("SPV_EXT_mesh_shader", module.extensions())
        self.assertEqual(module.modes(function), {MODE_LOCAL_SIZE: [32, 1, 1]})

        # The payload is the task's output region: one TaskPayloadWorkgroupEXT array of the declared 256 bytes
        (payload,) = module.variables_in(STORAGE_PAYLOAD)
        self.assertEqual(module.array_length(payload), 64)
        self.assertIn(payload, interface)

        body = module.body(function)
        chains = self.chains_on(module, body, payload)
        stores = [ops for op, ops in body if op == OP_STORE and ops[0] in chains]
        self.assertGreaterEqual(len(stores), 2)

        # One terminator, the last instruction: OpEmitMeshTasksEXT count 1 1 payload, no OpReturn
        self.assertEqual([op for op, _ in body].count(OP_EMIT_MESH_TASKS), 1)
        self.assertEqual(body[-1][0], OP_EMIT_MESH_TASKS)
        self.assertNotIn(OP_RETURN, [op for op, _ in body])
        self.assertEqual([op for op, _ in body].count(OP_CONTROL_BARRIER), 1)      # before the payload is handed on
        x, y, z, used = body[-1][1]
        self.assertEqual((module.constants[y], module.constants[z], used), (1, 1, payload))

    def test_the_group_count_is_the_payload_word_at_byte_4(self):
        module, _ = self.translate(self.job("t2", "ts", TASK))
        _, function, _, _ = module.entry()
        body = module.body(function)
        (payload,) = module.variables_in(STORAGE_PAYLOAD)
        count = body[-1][1][0]
        load = next(ops for op, ops in body if op == OP_LOAD and ops[1] == count)
        chain = next(ops for op, ops in body if op == OP_ACCESS_CHAIN and ops[1] == load[2])
        self.assertEqual(chain[2], payload)
        self.assertEqual([module.constants[i] for i in chain[3:]], [1])

    def test_every_exit_reaches_the_one_epilogue(self):
        code = [mov_cbuf(2, 3, 0x10), EXIT_IF_P0, isbewr(0xB8, 4, 2), EXIT]
        module, _ = self.translate(self.job("t3", "ts", code))
        _, function, _, _ = module.entry()
        body = module.body(function)
        self.assertEqual([op for op, _ in body].count(OP_EMIT_MESH_TASKS), 1)
        self.assertEqual(body[-1][0], OP_EMIT_MESH_TASKS)
        self.assertNotIn(OP_RETURN, [op for op, _ in body])

    def test_the_isberd_modes_a_task_reads(self):
        for mode in (0x78, 0x38, 0xB8):
            with self.subTest(mode=hex(mode)):
                module, _ = self.translate(self.job(f"r{mode:x}", "ts", [isbewr(0xB8, 0x10, 255), isberd(mode, 4, 0x10), isbewr(0xB8, 0x14, 4), EXIT]))
                self.assertEqual(module.entry()[0], MODEL_TASK)

    def test_a_task_store_in_another_mode_is_refused(self):
        self.refuse(self.job("t4", "ts", [isbewr(0xF8, 0, 2), EXIT]), "unsupported_opcode", "stage output store mode f8 outside a task shader")


class MeshEmissionTest(MeshHarness):
    VERTEX = POSITION + [[1, 0], [1, 1]]
    PRIMITIVE = [[2, 0]]

    def mesh_job(self, key: str, code: list[bytes] = MESH, **fields) -> dict:
        fields = {"payload": 8, "max_vertices": 3, "max_primitives": 1, "vertex": self.VERTEX, "primitive": self.PRIMITIVE, **fields}
        return self.job(key, "ms", code, **fields)

    def test_a_mesh_reads_the_payload_and_emits_three_vertices_and_one_primitive(self):
        module, res = self.translate(self.mesh_job("m1"))
        model, function, name, interface = module.entry()
        self.assertEqual((model, name), (MODEL_MESH, "main"))
        self.assertEqual(res["exec_model"], "MeshEXT")
        self.assertEqual(module.version, SPIRV_1_4)
        self.assertIn(CAPABILITY_MESH_SHADING, module.capabilities())
        self.assertIn("SPV_EXT_mesh_shader", module.extensions())
        self.assertEqual(module.modes(function), {MODE_LOCAL_SIZE: [32, 1, 1], MODE_OUTPUT_VERTICES: [3],
                                                  MODE_OUTPUT_PRIMITIVES: [1], MODE_OUTPUT_TRIANGLES: []})

        # From SPIR-V 1.4 the interface lists every global variable
        self.assertLessEqual(set(module.variables), set(interface))

        # The payload the task wrote, read by the program
        (payload,) = module.variables_in(STORAGE_PAYLOAD)
        self.assertEqual(module.array_length(payload), 2)
        body = module.body(function)
        chains = self.chains_on(module, body, payload)
        self.assertTrue([ops for op, ops in body if op == OP_LOAD and ops[2] in chains])

        # OpSetMeshOutputsEXT once, with the declared maximum vertex count, before the first write to an output
        sets = [(n, ops) for n, (op, ops) in enumerate(body) if op == OP_SET_MESH_OUTPUTS]
        self.assertEqual(len(sets), 1)
        self.assertEqual(module.constants[sets[0][1][0]], 3)
        outputs = set(module.variables_in(STORAGE_OUTPUT))
        writes = [n for n, (op, ops) in enumerate(body) if op == OP_ACCESS_CHAIN and ops[2] in outputs]
        self.assertTrue(writes and min(writes) > sets[0][0])

        # Every EXIT goes to one return
        self.assertEqual([op for op, _ in body].count(OP_RETURN), 1)

    def test_a_mesh_zeroes_its_count_before_its_first_barrier(self):
        # Every invocation stores the zero count, then one barrier, before the program's first instruction; the epilogue has the other
        module, _ = self.translate(self.mesh_job("m8"))
        body = module.body(module.entry()[1])
        barriers = [n for n, (op, _) in enumerate(body) if op == OP_CONTROL_BARRIER]
        self.assertEqual(len(barriers), 2)
        stores = [ops for op, ops in body[:barriers[0]] if op == OP_STORE]
        self.assertEqual(len(stores), 1)
        self.assertEqual(module.constants[stores[0][1]], 0)
        chain = module.defs[stores[0][0]][1]
        self.assertEqual(module.variables[chain[2]][1], 4)                      # a Workgroup variable: the count

    def test_the_primitive_count_is_clamped_to_the_declared_maximum(self):
        module, _ = self.translate(self.mesh_job("m9"))
        body = module.body(module.entry()[1])
        (count,) = [ops[1] for op, ops in body if op == OP_SET_MESH_OUTPUTS]
        kind, select = module.defs[count]
        self.assertEqual(kind, OP_SELECT)
        self.assertIn(1, [module.constants[i] for i in select[3:] if i in module.constants])   # max_primitives is 1

    def test_the_outputs_are_the_components_of_the_map_and_no_others(self):
        module, _ = self.translate(self.mesh_job("m2"))
        builtins, components = {}, {}
        for variable in module.variables_in(STORAGE_OUTPUT):
            decorations = module.decorations(variable)
            if DECORATION_BUILTIN in decorations:
                builtins[decorations[DECORATION_BUILTIN][0]] = module.array_length(variable)
            else:
                key = (decorations[DECORATION_LOCATION][0], decorations.get(DECORATION_COMPONENT, [0])[0])
                components[key] = (DECORATION_PER_PRIMITIVE in decorations, module.array_length(variable))

        # The position (whole), the triangle indices, and one array of float for each other component of the map
        self.assertEqual(builtins, {BUILTIN_POSITION: 3, BUILTIN_TRIANGLE_INDICES: 1})
        self.assertEqual(components, {(1, 0): (False, 3), (1, 1): (False, 3), (2, 0): (True, 1)})

    def test_a_mesh_without_a_task_has_no_payload(self):
        module, res = self.translate(self.mesh_job("m3", has_task=False))
        self.assertEqual(module.variables_in(STORAGE_PAYLOAD), [])
        self.assertFalse(res["mesh_stage"]["has_task"])
        self.assertEqual(res["mesh_stage"]["payload_bytes"], 0)

    def test_every_exit_reaches_the_one_epilogue(self):
        code = [mov_cbuf(5, 3, 0x14), EXIT_IF_P0, isbewr(0x88, 3, 5), EXIT]
        module, _ = self.translate(self.mesh_job("m4", code))
        _, function, _, _ = module.entry()
        body = module.body(function)
        self.assertEqual([op for op, _ in body].count(OP_SET_MESH_OUTPUTS), 1)
        self.assertEqual([op for op, _ in body].count(OP_RETURN), 1)

    def test_a_count_store_away_from_byte_3_is_refused(self):
        code = [mov_cbuf(5, 3, 0x14), isbewr(0x88, 7, 5), EXIT]
        self.refuse(self.mesh_job("m5", code), "unsupported_opcode", "32-bit stage buffer map store at an address other than byte 3")

    def test_a_task_store_in_a_mesh_program_is_refused(self):
        self.refuse(self.mesh_job("m6", [isbewr(0xB8, 0x10, 2), EXIT]), "unsupported_opcode", "stage output store mode b8 outside a mesh shader")

    def test_a_zero_count_store_is_dropped_and_the_program_still_translates(self):
        module, _ = self.translate(self.mesh_job("m7", [isbewr(0x88, 3, 255), EXIT]))
        self.assertEqual(module.entry()[0], MODEL_MESH)


class MeshLimitsTest(MeshHarness):
    def wide(self, key: str, locations: int, **extra) -> dict:
        """160 vertices and 128 primitives of one component in each of `locations` locations: 16 bytes a location and vertex."""
        job = self.job(key, "ms", [EXIT], max_vertices=160, max_primitives=128, vertex=[[n, 0] for n in range(locations)], payload=128)
        job.update(extra)
        return job

    def test_a_program_past_the_limits_is_refused_by_name(self):
        over = [
            (self.job("l1", "ms", [EXIT], max_vertices=300), "maxMeshOutputVertices"),
            (self.job("l2", "ms", [EXIT], max_primitives=300), "maxMeshOutputPrimitives"),
            (self.wide("l3", 13), "maxMeshOutputMemorySize"),                              # 13 x 16 x 160 = 33,280 of 32,768
            (self.job("l4", "ts", [EXIT], payload=16388), "maxTaskPayloadSize"),
            (self.job("l5", "ms", [EXIT], payload=29000, vertex=POSITION, max_vertices=3, max_primitives=1),
             "maxMeshPayloadAndSharedMemorySize"),
        ]
        for job, name in over:
            with self.subTest(name=name):
                self.refuse(job, "over_limit", name)

    def test_the_invocation_count_is_a_limit(self):
        for key, stage, name in (("i1", "ts", "maxTaskWorkGroupInvocations"), ("i2", "ms", "maxMeshWorkGroupInvocations")):
            with self.subTest(name=name):
                job = self.job(key, stage, [EXIT])
                job["local"] = [256, 1, 1]
                job["mesh"]["threads"] = 256
                self.refuse(job, "over_limit", name)

    def test_the_largest_program_that_fits_is_accepted(self):
        _, res = self.translate(self.wide("l6", 12))                                       # 12 x 16 x 160 = 30,720 of 32,768
        self.assertEqual(res["mesh_stage"]["output_bytes"], 30720)

    def test_the_limits_are_an_input_and_are_recorded(self):
        _, res = self.translate(self.wide("l7", 13, options={"mesh_limits": {"max_mesh_output_memory": 40000}}))
        self.assertEqual(res["mesh_stage"]["limits"]["max_mesh_output_memory"], 40000)
        _, res = self.translate(self.wide("l8", 1))
        self.assertEqual(res["mesh_stage"]["limits"], {
            "max_task_invocations": 128, "max_mesh_invocations": 128, "max_mesh_output_vertices": 256,
            "max_mesh_output_primitives": 256, "max_mesh_output_components": 128, "max_mesh_output_memory": 32768,
            "max_mesh_shared_memory": 28672, "max_mesh_payload_and_shared_memory": 28672,
            "max_mesh_payload_and_output_memory": 48128, "max_task_payload": 16384, "max_task_payload_and_shared_memory": 32768})

    def test_a_limit_must_be_named_and_a_number(self):
        for key, limits, needle in (("b1", {"max_everything": 1}, "unknown limit"), ("b2", {"max_task_payload": "big"}, "is not an integer")):
            with self.subTest(key=key):
                job = self.job(key, "ts", [EXIT])
                job["options"] = {"mesh_limits": limits}
                self.refuse(job, "bad_job", needle)

    def test_shared_memory_of_its_own_is_refused(self):
        self.refuse(self.job("s1", "ms", [EXIT], own_shared=True), "unsupported_stage", "shared memory of its own")


class MeshDescriptionTest(MeshHarness):
    def test_the_description_of_the_worst_mesh_program_of_the_dump(self):
        # 160 vertices, 128 primitives, the position and ten generic locations with 22 components: 26 lines. The mesh shader study: 28,160 of
        # 32,768 bytes of output by Vulkan's formula, 18,196 bytes of staging, with a payload of 128 bytes 28,288 of 48,128
        vertex = POSITION + [[n, c] for n in range(10) for c in range(3 if n >= 8 else 2)]
        _, res = self.translate(self.job("d1", "ms", [EXIT], max_vertices=160, max_primitives=128, vertex=vertex, payload=128))
        stage = res["mesh_stage"]
        self.assertEqual((stage["kind"], stage["threads"], stage["max_vertices"], stage["max_primitives"]), ("mesh", 32, 160, 128))
        self.assertEqual((stage["vertex_locations"], stage["vertex_components"], stage["primitive_locations"]), (11, 26, 0))
        self.assertEqual((stage["payload_bytes"], stage["output_bytes"], stage["shared_bytes"]), (128, 28160, 18196))
        self.assertTrue(stage["requires_subgroup_32"])
        self.assertIsNone(res["compute"])

    def test_the_description_of_a_task_program(self):
        _, res = self.translate(self.job("d2", "ts", TASK))
        stage = res["mesh_stage"]
        self.assertEqual((stage["kind"], stage["payload_bytes"], stage["output_bytes"], stage["max_vertices"]), ("task", 256, 0, 0))
        self.assertTrue(stage["requires_subgroup_32"])
        self.assertEqual(res["capabilities"].count("MeshShadingEXT"), 1)
        self.assertIn("SPV_EXT_mesh_shader", res["extensions"])

    def wide_mesh(self, key: str, vertex: list, primitive: list) -> dict:
        return self.job(key, "ms", [EXIT], max_vertices=160, max_primitives=128, vertex=POSITION + vertex, primitive=primitive,
                        payload=128)

    def pixel(self, key: str) -> dict:
        """The synthetic pixel program of tools/firstdraw: generic attribute 0 read in its four components."""
        path = self.dir / f"{key}.bin"
        path.write_bytes(mp.pixel_program())
        return {"key": key, "stage": "ps", "arch": "sm86", "code": str(path), "c1": ""}

    def test_each_mesh_output_gives_its_rate_and_its_array_depth(self):
        # Every output is an array over the 160 vertices or the 128 primitives, around one float at its location and component
        _, res = self.translate(self.wide_mesh("d4", [[1, 0], [1, 1], [3, 2]], [[2, 0]]))
        got = [(o["location"], o["component"], o["components"], o["type"], o["array_length"], o["array_depth"], o["per_primitive"])
               for o in res["interface"]["outputs"]]
        self.assertEqual(got, [(1, 0, 1, "float32", 160, 1, False), (1, 1, 1, "float32", 160, 1, False),
                               (2, 0, 1, "float32", 128, 1, True), (3, 2, 1, "float32", 160, 1, False)])

    def test_a_pixel_input_is_read_per_vertex_and_is_no_array(self):
        _, res = self.translate(self.pixel("p4"))
        got = [(i["location"], i["component"], i["components"], i["array_depth"], i["per_primitive"], "array_length" in i)
               for i in res["interface"]["inputs"]]
        self.assertEqual(got, [(0, 0, 4, 0, False, False)])

    def test_the_runtime_links_these_descriptions_whatever_the_vertex_count(self):
        # The runtime's own rule (shd_mesh_link, through tests/runtime/vk_pipeline_test.c) on the translator's descriptions:
        # a float[160] at location 0 takes location 0 alone; counted as 160 locations it would reach past the builder's 64.
        import test_runtime_vk_pipeline as vp
        self.translate(self.wide_mesh("d5", [[0, c] for c in range(4)], []))
        self.translate(self.pixel("p5"))
        self.translate(self.wide_mesh("d6", [], [[0, c] for c in range(4)]))
        link = lambda mesh: vp.run("meshlink", str(self.out / f"{mesh}.res.json"), str(self.out / "p5.res.json"))  # noqa: E731
        done = link("d5")
        self.assertEqual((done.returncode, done.stdout), (0, "missing 0 \n"), done.stderr)
        done = link("d6")   # written per primitive, read per vertex
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue(done.stdout.startswith("refused: rate: pixel input at location 0 component 0 is read per vertex"), done.stdout)

    def test_other_stages_have_no_mesh_description(self):
        from test_xlate_host import compute_job
        self.assertEqual(self.run_jobs([compute_job("c1", self.code("c1", [EXIT]))])["c1"].status, "ok")
        res = json.loads((self.out / "c1.res.json").read_text(encoding="utf-8"))
        self.assertIsNone(res["mesh_stage"])
        self.assertEqual(res["exec_model"], "GLCompute")


# A task program that reads its workgroup number (ISBERD 0x78) and stores it as payload word 4; a mesh program that reads it and
# writes it to the first component of vertex 0. A mesh program under a task reads the payload word the task wrote.
TASK_NUMBER = [isberd(0x78, 2, 0), isbewr(0xB8, 0x10, 2), mov_cbuf(3, 3, 0x10), isbewr(0xB8, 4, 3), EXIT]
MESH_NUMBER = [isberd(0x78, 4, 0), isbewr(0xF8, 0, 4), mov_cbuf(5, 3, 0x14), isbewr(0x88, 3, 5), EXIT]


class FirstTaskTest(MeshHarness):
    """The draw's first task (the mesh bindings work): the word at the device address in push constant words 6 and 7, added to the workgroup number of a
    task, or of a mesh stage that has no task. One mechanism for the direct and the indirect draw: the module only knows an address."""

    def first_task(self, module: Module) -> tuple[int, list[int]]:
        """The one load that reads the first task, as (its result id, the push words of its address): found by structure, from the load through
        a physical storage buffer pointer back to the two push constant words that make the address."""
        (push,) = module.variables_in(STORAGE_PUSH_CONSTANT)
        found = []
        for op, ops in module.insts:
            if op != OP_LOAD or ops[2] not in module.defs:
                continue
            kind, pointer = module.defs[ops[2]]
            if kind != OP_CONVERT_U_TO_PTR:
                continue
            self.assertEqual(module.pointers[pointer[0]][0], STORAGE_PHYSICAL)
            self.assertEqual(ops[3:5], [MEMORY_ACCESS_ALIGNED, 4], "a physical load states its alignment")
            kind, pair = self.value(module, pointer[2])
            self.assertEqual(kind, OP_COMPOSITE_CONSTRUCT)
            words = []
            for part in pair[2:]:
                kind, load = self.value(module, part)
                self.assertEqual(kind, OP_LOAD)
                kind, chain = module.defs[load[2]]
                self.assertEqual((kind, chain[2]), (OP_ACCESS_CHAIN, push))
                member, vector, component = (module.constants[i] for i in chain[3:])
                self.assertEqual(member, 0)
                words.append(4 * vector + component)
            found.append((ops[1], words))
        self.assertEqual(len(found), 1, "one load of the first task per ISBERD of the workgroup number")
        return found[0]

    @staticmethod
    def value(module: Module, ident: int) -> tuple[int, list[int]]:
        """The instruction that made a value, through bitcasts and the function-local variables the generator keeps its temporaries in."""
        stores = module.local_stores()
        while True:
            if ident not in module.defs:
                return 0, [0, ident]            # made by an instruction this reader does not follow
            kind, ops = module.defs[ident]
            if kind == OP_BITCAST:
                ident = ops[2]
            elif kind == OP_LOAD and ops[2] in stores:
                ident = stores[ops[2]]
            else:
                return kind, ops

    def adds_to_workgroup_number(self, module: Module, first: int) -> None:
        function = module.entry()[1]
        sums = []
        for op, ops in module.body(function):
            if op == OP_IADD:
                terms = [self.value(module, ops[2]), self.value(module, ops[3])]
                if first in [ops_[1] for _, ops_ in terms]:
                    sums.append(terms)
        self.assertEqual(len(sums), 1)
        (kind, ops), = [t for t in sums[0] if t[1][1] != first]
        self.assertEqual(kind, OP_LOAD)
        kind, chain = module.defs[ops[2]]
        self.assertEqual(module.decorations(chain[2]).get(DECORATION_BUILTIN), [BUILTIN_WORKGROUP_ID], "the other term is the workgroup id")

    def physical(self, module: Module) -> bool:
        return CAPABILITY_PHYSICAL_STORAGE in module.capabilities()

    def test_a_task_adds_the_first_task_to_its_workgroup_number(self):
        module, res = self.translate(self.job("f1", "ts", TASK_NUMBER))
        first, words = self.first_task(module)
        self.assertEqual(words, [6, 7])
        self.adds_to_workgroup_number(module, first)
        self.assertEqual(module.ops(OP_MEMORY_MODEL)[0][0], ADDRESSING_PHYSICAL_STORAGE_64)
        self.assertLessEqual({CAPABILITY_PHYSICAL_STORAGE, CAPABILITY_INT64}, module.capabilities())
        self.assertIn("SPV_KHR_physical_storage_buffer", module.extensions())
        # Not a guest access: no helper function, no page table
        self.assertEqual([op for op, _ in module.body(module.entry()[1]) if op == OP_FUNCTION_CALL], [])
        self.assertFalse(res["raw_access"]["page_table"])
        self.assertTrue(res["raw_access"]["physical_pointers"])

    def test_a_mesh_without_a_task_adds_it_too(self):
        module, res = self.translate(self.job("f2", "ms", MESH_NUMBER, payload=8, max_vertices=3, max_primitives=1, has_task=False,
                                              vertex=POSITION))
        first, words = self.first_task(module)
        self.assertEqual(words, [6, 7])
        self.adds_to_workgroup_number(module, first)
        self.assertTrue(res["mesh_stage"]["first_task"]["read"])

    def test_a_mesh_under_a_task_adds_nothing(self):
        # Its number is within its task's launch; the task has the first task
        module, res = self.translate(self.job("f3", "ms", MESH_NUMBER, payload=8, max_vertices=3, max_primitives=1, has_task=True,
                                              vertex=POSITION))
        self.assertFalse(self.physical(module))
        self.assertEqual(module.variables_in(STORAGE_PUSH_CONSTANT), [])
        self.assertFalse(res["mesh_stage"]["first_task"]["read"])
        self.assertEqual(module.ops(OP_MEMORY_MODEL)[0][0], 0)       # Logical addressing, as before

    def test_a_program_that_never_reads_its_number_carries_no_physical_addressing(self):
        module, res = self.translate(self.job("f4", "ts", TASK))
        self.assertFalse(self.physical(module))
        self.assertFalse(res["raw_access"]["physical_pointers"])
        self.assertFalse(res["mesh_stage"]["first_task"]["read"])

    def test_a_task_that_reads_its_own_payload_back_does_not_read_the_first_task(self):
        code = [isbewr(0xB8, 0x10, 255), isberd(0xB8, 4, 0x10), isbewr(0xB8, 0x14, 4), EXIT]
        module, res = self.translate(self.job("f5", "ts", code))
        self.assertFalse(self.physical(module))
        self.assertFalse(res["mesh_stage"]["first_task"]["read"])

    def test_the_description_names_where_the_first_task_goes(self):
        _, res = self.translate(self.job("f6", "ts", TASK_NUMBER))
        self.assertEqual(res["mesh_stage"]["first_task"], {"read": True, "form": "device_address", "push_offset": 24,
                                                           "address_bytes": 8, "word_bytes": 4})
        # Inside the 12 bytes the uni layout keeps for it, and clear of the page table words
        first, length = res["layout"]["first_task_bytes"]
        self.assertEqual((first, length), (24, 12))
        self.assertLessEqual(res["mesh_stage"]["first_task"]["push_offset"] + res["mesh_stage"]["first_task"]["address_bytes"], first + length)
        self.assertGreaterEqual(res["mesh_stage"]["first_task"]["push_offset"], res["raw_access"]["push_offset"] + res["raw_access"]["push_bytes"])
