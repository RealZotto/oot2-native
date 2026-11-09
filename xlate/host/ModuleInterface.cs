// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Oot2.Xlate;

/// <summary>One user-defined input or output variable of a module (a Location decoration).</summary>
internal sealed class InterfaceVariable
{
    public int Location { get; init; }
    public int Component { get; init; }
    public int Components { get; init; }
    public string Type { get; init; }
    /// <summary>The outer array's length (0 for a variable that is not an array, -1 when the length is not a constant).</summary>
    public int ArrayLength { get; init; }

    /// <summary>
    /// How many array levels the variable's type has (0 for none). A mesh stage's output has one level more than what it
    /// holds: the outer array is over its vertices or primitives and takes no location of its own.
    /// </summary>
    public int ArrayDepth { get; init; }

    /// <summary>The variable is decorated PerPrimitiveEXT: a mesh output written, or a pixel input read, per primitive.</summary>
    public bool PerPrimitive { get; init; }

    public bool Used { get; init; }
}

/// <summary>
/// What a module's interface is, read from the module itself (the final SPIR-V): its user inputs and outputs by location,
/// the built-in variables it declares, and which fields of the support buffer it reads. The host builds vertex input state
/// from the inputs of a vertex module, pads a vertex module with the outputs a pixel module reads but it does not write
/// (they read (0, 0, 0, 1) on the console; spike S3, finding G), and fills the support buffer fields a module reads.
/// </summary>
internal sealed class ModuleInterface
{
    private const uint OpEntryPoint = 15, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypeArray = 28;
    private const uint OpTypeStruct = 30, OpTypePointer = 32, OpConstant = 43, OpFunction = 54, OpVariable = 59;
    private const uint OpLoad = 61, OpStore = 62, OpCopyMemory = 63, OpAccessChain = 65, OpInBoundsAccessChain = 66;
    private const uint OpDecorate = 71, OpMemberDecorate = 72;
    private const uint DecorationBuiltIn = 11, DecorationLocation = 30, DecorationComponent = 31;
    private const uint DecorationBinding = 33, DecorationDescriptorSet = 34, DecorationPerPrimitive = 5271;
    private const uint StorageInput = 1, StorageUniform = 2, StorageOutput = 3;

    public List<InterfaceVariable> Inputs { get; } = new();
    public List<InterfaceVariable> Outputs { get; } = new();
    public SortedSet<string> BuiltInsIn { get; } = new(StringComparer.Ordinal);
    public SortedSet<string> BuiltInsOut { get; } = new(StringComparer.Ordinal);

    /// <summary>Indices of the support buffer's structure members that an access chain reaches.</summary>
    public SortedSet<int> SupportFieldsRead { get; } = new();

    public static ModuleInterface Read(byte[] module, int supportSet, int supportBinding)
    {
        ReadOnlySpan<uint> w = MemoryMarshal.Cast<byte, uint>(module);
        ModuleInterface mi = new();

        Dictionary<uint, (uint Op, int At)> types = new();
        Dictionary<uint, uint> constants = new();
        Dictionary<uint, (uint Storage, uint Pointee)> pointers = new();
        Dictionary<uint, uint> variables = new();       // variable id -> pointer type id
        Dictionary<uint, int> location = new(), component = new(), set = new(), binding = new();
        Dictionary<uint, uint> builtIn = new();
        Dictionary<uint, List<uint>> memberBuiltIns = new();
        HashSet<uint> used = new(), perPrimitive = new();
        bool inFunctions = false;

        for (int i = 5; i < w.Length;)
        {
            uint op = w[i] & 0xffff;
            int n = (int)(w[i] >> 16);

            if (n == 0 || i + n > w.Length)
            {
                throw new FormatException($"malformed instruction at word {i}");
            }

            ReadOnlySpan<uint> inst = w.Slice(i, n);

            switch (op)
            {
                case OpDecorate when n >= 3:
                    switch (inst[2])
                    {
                        case DecorationLocation when n >= 4: location[inst[1]] = (int)inst[3]; break;
                        case DecorationComponent when n >= 4: component[inst[1]] = (int)inst[3]; break;
                        case DecorationBinding when n >= 4: binding[inst[1]] = (int)inst[3]; break;
                        case DecorationDescriptorSet when n >= 4: set[inst[1]] = (int)inst[3]; break;
                        case DecorationBuiltIn when n >= 4: builtIn[inst[1]] = inst[3]; break;
                        case DecorationPerPrimitive: perPrimitive.Add(inst[1]); break;
                    }

                    break;
                case OpMemberDecorate when n >= 5 && inst[3] == DecorationBuiltIn:
                    if (!memberBuiltIns.TryGetValue(inst[1], out List<uint> list))
                    {
                        memberBuiltIns[inst[1]] = list = new();
                    }

                    list.Add(inst[4]);
                    break;
                case OpTypeInt or OpTypeFloat or OpTypeVector or OpTypeArray or OpTypeStruct when n >= 2:
                    types[inst[1]] = (op, i);
                    break;
                case OpTypePointer when n >= 4:
                    pointers[inst[1]] = (inst[2], inst[3]);
                    break;
                case OpConstant when n >= 4:
                    constants[inst[2]] = inst[3];
                    break;
                case OpVariable when n >= 4 && !inFunctions:
                    variables[inst[2]] = inst[1];
                    break;
                case OpFunction:
                    inFunctions = true;
                    break;
                case OpLoad when n >= 4:
                    used.Add(inst[3]);
                    break;
                case OpStore when n >= 3:
                    used.Add(inst[1]);
                    break;
                case OpCopyMemory when n >= 3:
                    used.Add(inst[1]);
                    used.Add(inst[2]);
                    break;
                case OpAccessChain or OpInBoundsAccessChain when n >= 4:
                    used.Add(inst[3]);
                    break;
            }

            i += n;
        }

        // The support buffer: the Uniform variable at its set and binding; each access chain from it names a member.
        uint support = 0;

        foreach ((uint id, uint ptr) in variables)
        {
            if (pointers.TryGetValue(ptr, out var p) && p.Storage == StorageUniform &&
                set.TryGetValue(id, out int s) && s == supportSet && binding.TryGetValue(id, out int b) && b == supportBinding)
            {
                support = id;
            }
        }

        for (int i = 5; i < w.Length; i += (int)(w[i] >> 16))
        {
            uint op = w[i] & 0xffff;
            int n = (int)(w[i] >> 16);

            if ((op == OpAccessChain || op == OpInBoundsAccessChain) && n >= 5 && support != 0 && w[i + 3] == support &&
                constants.TryGetValue(w[i + 4], out uint member))
            {
                mi.SupportFieldsRead.Add((int)member);
            }
        }

        foreach ((uint id, uint ptr) in variables)
        {
            if (!pointers.TryGetValue(ptr, out var p) || (p.Storage != StorageInput && p.Storage != StorageOutput))
            {
                continue;
            }

            bool input = p.Storage == StorageInput;
            SortedSet<string> builtIns = input ? mi.BuiltInsIn : mi.BuiltInsOut;

            if (builtIn.TryGetValue(id, out uint bi))
            {
                builtIns.Add(BuiltInName(bi));

                continue;
            }

            uint pointee = Unarray(w, types, p.Pointee, out int arrayLength, out int arrayDepth, constants);

            if (types.TryGetValue(pointee, out var t) && t.Op == OpTypeStruct)
            {
                // A block of built-ins (gl_PerVertex)
                if (memberBuiltIns.TryGetValue(pointee, out List<uint> members))
                {
                    foreach (uint m in members)
                    {
                        builtIns.Add(BuiltInName(m));
                    }
                }

                continue;
            }

            if (!location.TryGetValue(id, out int loc))
            {
                continue;
            }

            (int count, string scalar) = Describe(w, types, pointee);

            (input ? mi.Inputs : mi.Outputs).Add(new InterfaceVariable
            {
                Location = loc,
                Component = component.TryGetValue(id, out int c) ? c : 0,
                Components = count,
                Type = scalar,
                ArrayLength = arrayLength,
                ArrayDepth = arrayDepth,
                PerPrimitive = perPrimitive.Contains(id),
                Used = used.Contains(id),
            });
        }

        mi.Inputs.Sort((a, b) => (a.Location, a.Component).CompareTo((b.Location, b.Component)));
        mi.Outputs.Sort((a, b) => (a.Location, a.Component).CompareTo((b.Location, b.Component)));

        return mi;
    }

    // The element type under any arrays (the per-vertex arrays of mesh outputs), with the outer array's length and the
    // number of array levels.
    private static uint Unarray(ReadOnlySpan<uint> w, Dictionary<uint, (uint Op, int At)> types, uint type, out int length,
        out int depth, Dictionary<uint, uint> constants)
    {
        length = 0;
        depth = 0;

        while (types.TryGetValue(type, out var t) && t.Op == OpTypeArray)
        {
            depth++;

            if (length == 0)
            {
                length = constants.TryGetValue(w[t.At + 3], out uint l) ? (int)l : -1;
            }

            type = w[t.At + 2];
        }

        return type;
    }

    private static (int Count, string Scalar) Describe(ReadOnlySpan<uint> w, Dictionary<uint, (uint Op, int At)> types, uint type)
    {
        int count = 1;

        if (types.TryGetValue(type, out var t) && t.Op == OpTypeVector)
        {
            count = (int)w[t.At + 3];
            type = w[t.At + 2];
        }

        if (!types.TryGetValue(type, out t))
        {
            return (count, "unknown");
        }

        return t.Op switch
        {
            OpTypeFloat => (count, $"float{w[t.At + 2]}"),
            OpTypeInt => (count, (w[t.At + 3] != 0 ? "sint" : "uint") + w[t.At + 2]),
            _ => (count, "unknown"),
        };
    }

    public static string BuiltInName(uint builtIn) => builtIn switch
    {
        0 => "Position", 1 => "PointSize", 3 => "ClipDistance", 4 => "CullDistance", 5 => "VertexId", 6 => "InstanceId",
        7 => "PrimitiveId", 8 => "InvocationId", 9 => "Layer", 10 => "ViewportIndex", 15 => "FragCoord", 16 => "PointCoord",
        17 => "FrontFacing", 18 => "SampleId", 19 => "SamplePosition", 20 => "SampleMask", 22 => "FragDepth",
        23 => "HelperInvocation", 24 => "NumWorkgroups", 25 => "WorkgroupSize", 26 => "WorkgroupId",
        27 => "LocalInvocationId", 28 => "GlobalInvocationId", 29 => "LocalInvocationIndex", 36 => "SubgroupSize",
        38 => "NumSubgroups", 40 => "SubgroupId", 41 => "SubgroupLocalInvocationId", 42 => "VertexIndex",
        43 => "InstanceIndex", 4416 => "SubgroupEqMask", 4417 => "SubgroupGeMask", 4418 => "SubgroupGtMask",
        4419 => "SubgroupLeMask", 4420 => "SubgroupLtMask", 4424 => "BaseVertex", 4425 => "BaseInstance",
        4426 => "DrawIndex", 5294 => "PrimitivePointIndicesEXT", 5295 => "PrimitiveLineIndicesEXT",
        5296 => "PrimitiveTriangleIndicesEXT", 5299 => "CullPrimitiveEXT",
        _ => $"BuiltIn{builtIn}",
    };
}
