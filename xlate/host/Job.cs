// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Oot2.Xlate;

/// <summary>The failure kinds of the job contract; they are the word before the colon in a "fail" reply.</summary>
internal static class Kinds
{
    public const string BadJob = "bad_job";
    public const string Io = "io";
    public const string UnsupportedStage = "unsupported_stage";
    public const string UnsupportedOpcode = "unsupported_opcode";
    public const string Decode = "decode";
    public const string Translate = "translate";
    public const string Validate = "validate";

    // A task or mesh program past a limit the device (or the job's mesh_limits) gives; the message names the limit
    public const string OverLimit = "over_limit";
}

/// <summary>A failure that already knows its contract kind.</summary>
internal sealed class JobException : Exception
{
    public string Kind { get; }

    public JobException(string kind, string message) : base(message)
    {
        Kind = kind;
    }
}

/// <summary>
/// Answers keyed by a slot number, with the contract's "*" entry for every slot
/// that is not named. A wildcard applies to slots the host could decode, never
/// to a handle that decodes to no slot at all.
/// </summary>
internal sealed class AnswerMap<T>
{
    public SortedDictionary<int, T> Named { get; } = new();
    public bool HasStar { get; private set; }
    public T Star { get; private set; }

    public void SetStar(T value)
    {
        HasStar = true;
        Star = value;
    }

    public bool TryGet(int slot, out T value)
    {
        if (slot >= 0 && Named.TryGetValue(slot, out value))
        {
            return true;
        }

        if (slot >= 0 && HasStar)
        {
            value = Star;
            return true;
        }

        value = default;
        return false;
    }

    public bool IsEmpty => Named.Count == 0 && !HasStar;
}

/// <summary>An "img" answer: the image type and the format name of one storage image slot.</summary>
internal readonly record struct ImageAnswer(SamplerType Type, string TypeName, TextureFormat Format);

/// <summary>The "options" object of a job; every field is part of the cache key.</summary>
internal sealed class JobOptions
{
    // Whether the AST optimizer runs in its debug mode (nothing propagated, nothing dropped). The default is the
    // sound form, debug false with this project's AstOptimizer rule; `debug: true` asks for the old form, which
    // only A/B work wants. A different default is a different module for every job, so it is a different translator
    // executable and therefore a different identity for every pack and cache that keys on it.
    public bool Debug { get; init; }
    public bool Reduced { get; init; }
    public bool Float64 { get; init; } = true;
    public bool NonUniformIndexing { get; init; } = true;
    public bool CapabilityFixup { get; init; } = true;
    public string Variant { get; init; } = "";

    // The limits a task or mesh program is checked against (the RTX 3060 Ti's unless prepare reads the device)
    public MeshLimits MeshLimits { get; init; } = MeshLimits.Default;
}

/// <summary>
/// One translation job, parsed and validated against job contract v1. Parsing
/// never touches the file system: the code file is read later, so that every
/// shape error is reported as bad_job before any I/O error.
/// </summary>
internal sealed class Job
{
    public const int DefaultHeapImages = 33025;
    public const int DefaultHeapSamplers = 4096;
    private const int MaxSharedBytes = 1 << 20;

    // Both are checked against this limit: a Vulkan workgroup cannot be larger.
    private const int MaxThreads = 1024;

    private static readonly Regex KeyPattern = new("^[A-Za-z0-9._-]{1,128}$", RegexOptions.CultureInvariant);

    private static readonly HashSet<string> KnownFields = new(StringComparer.Ordinal)
    {
        "key", "stage", "arch", "code", "code_offset", "code_length", "c1", "nvn2_cs", "local", "shared",
        "stage_memory", "mesh", "outputs", "tex", "img", "attr", "fout", "early_fragment_tests", "heap_images",
        "heap_samplers", "options",
    };

    private static readonly HashSet<string> KnownOptions = new(StringComparer.Ordinal)
    {
        "debug", "reduced", "float64", "nonuniform_indexing", "capability_fixup", "variant", "mesh_limits",
    };

    public string Key { get; private set; }
    public string Stage { get; private set; }
    public string CodePath { get; private set; }
    public long CodeOffset { get; private set; }
    public long? CodeLength { get; private set; }
    public byte[] C1 { get; private set; } = Array.Empty<byte>();
    public int[] Nvn2Cs { get; private set; }
    public int[] Local { get; private set; }
    public int Shared { get; private set; }
    public int[] StageMemory { get; private set; }
    public JsonElement? Mesh { get; private set; }

    // The output components a task or mesh program writes, as the splitter read them from the program headers
    // (null when the job carries no map): [location, component] pairs, location -1 for the position. A mesh
    // emission declares exactly these and no other component. Echoed into res.json; no pass reads it yet.
    public int[][] OutputsVertex { get; private set; }
    public int[][] OutputsPrimitive { get; private set; }

    // A task or mesh job as the stage the translator emits: built from its mesh object, local size and output map (null for other stages).
    // MeshPayloadBytes is the payload the control section declares, MeshOwnShared whether the program uses shared memory of its own
    public MeshStageInfo MeshStage { get; private set; }
    public int MeshPayloadBytes { get; private set; }
    public bool MeshOwnShared { get; private set; }

    public AnswerMap<SamplerType> Tex { get; } = new();
    public AnswerMap<ImageAnswer> Img { get; } = new();
    public AnswerMap<AttributeType> Attr { get; } = new();
    public AnswerMap<AttributeType> Fout { get; } = new();
    public bool EarlyFragmentTests { get; private set; }
    public int HeapImages { get; private set; } = DefaultHeapImages;
    public int HeapSamplers { get; private set; } = DefaultHeapSamplers;
    public JobOptions Options { get; private set; } = new();

    public bool IsCompute => Stage is "cs" or "ts" or "ms";

    /// <summary>
    /// Whether the key is safe to echo in a reply line, which is split on
    /// spaces: printable ASCII only, no space, at most 128 characters.
    /// </summary>
    public static bool IsEchoable(string key)
    {
        if (string.IsNullOrEmpty(key) || key.Length > 128)
        {
            return false;
        }

        foreach (char c in key)
        {
            if (c <= ' ' || c >= 0x7f)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>The key of a job object even when the rest of it is invalid, or null.</summary>
    public static string PeekKey(JsonElement obj)
    {
        if (obj.ValueKind == JsonValueKind.Object &&
            obj.TryGetProperty("key", out JsonElement key) &&
            key.ValueKind == JsonValueKind.String)
        {
            string s = key.GetString();
            return IsEchoable(s) ? s : null;
        }

        return null;
    }

    public static Job Parse(JsonElement obj)
    {
        if (obj.ValueKind != JsonValueKind.Object)
        {
            throw Bad("job is not an object");
        }

        HashSet<string> seen = new(StringComparer.Ordinal);

        foreach (JsonProperty p in obj.EnumerateObject())
        {
            if (!KnownFields.Contains(p.Name))
            {
                throw Bad($"unknown field '{Clip(p.Name)}'");
            }

            if (!seen.Add(p.Name))
            {
                throw Bad($"duplicate field '{p.Name}'");
            }
        }

        Job job = new();

        job.Key = RequireString(obj, "key");

        if (!KeyPattern.IsMatch(job.Key))
        {
            throw Bad("key does not match ^[A-Za-z0-9._-]{1,128}$");
        }

        job.Stage = RequireString(obj, "stage");

        if (job.Stage is "gs" or "tcs" or "tes" or "tess" or "hs" or "ds")
        {
            throw new JobException(Kinds.UnsupportedStage, $"stage '{job.Stage}' has no translation in this host");
        }

        if (job.Stage is not ("vs" or "ps" or "cs" or "ts" or "ms"))
        {
            throw new JobException(Kinds.UnsupportedStage, $"unknown stage '{Clip(job.Stage)}'");
        }

        string arch = RequireString(obj, "arch");

        if (arch != "sm86")
        {
            throw Bad($"arch '{Clip(arch)}' is not sm86");
        }

        job.CodePath = RequireString(obj, "code");

        if (job.CodePath.Length == 0)
        {
            throw Bad("code is empty");
        }

        if (obj.TryGetProperty("code_offset", out JsonElement off))
        {
            job.CodeOffset = ReadInt(off, "code_offset", 0, int.MaxValue);
        }

        if (obj.TryGetProperty("code_length", out JsonElement len))
        {
            long n = ReadInt(len, "code_length", 1, int.MaxValue);

            if (n % 16 != 0)
            {
                throw Bad($"code_length {n} is not a multiple of 16");
            }

            job.CodeLength = n;
        }

        job.C1 = ReadHex(RequireString(obj, "c1"));
        ParseStageFields(obj, job);
        ParseAnswers(obj, job);
        ParseOptions(obj, job);

        return job;
    }

    private static void ParseStageFields(JsonElement obj, Job job)
    {
        bool cs = job.Stage == "cs";
        bool mesh = job.Stage is "ts" or "ms";

        RejectUnless(obj, "nvn2_cs", cs, job.Stage);
        RejectUnless(obj, "local", cs || mesh, job.Stage);
        RejectUnless(obj, "shared", cs || mesh, job.Stage);
        RejectUnless(obj, "stage_memory", mesh, job.Stage);
        RejectUnless(obj, "mesh", mesh, job.Stage);
        RejectUnless(obj, "outputs", mesh, job.Stage);

        if (cs)
        {
            job.Nvn2Cs = ReadIntArray(Require(obj, "nvn2_cs"), "nvn2_cs", 7, 0, 65535);
        }

        if (cs || mesh)
        {
            job.Local = ReadIntArray(Require(obj, "local"), "local", 3, 1, MaxThreads);

            if (job.Local[0] * job.Local[1] * job.Local[2] > MaxThreads)
            {
                throw Bad($"local {job.Local[0]}x{job.Local[1]}x{job.Local[2]} has more than {MaxThreads} threads");
            }

            job.Shared = (int)ReadInt(Require(obj, "shared"), "shared", 0, MaxSharedBytes);
        }

        if (mesh)
        {
            job.StageMemory = ReadIntArray(Require(obj, "stage_memory"), "stage_memory", 4, 0, MaxSharedBytes);

            if (obj.TryGetProperty("mesh", out JsonElement m))
            {
                if (m.ValueKind != JsonValueKind.Object)
                {
                    throw Bad("field 'mesh' is not an object");
                }

                // Echoed into res.json; the translator does not read it. Cloned
                // so that it outlives the document of the request.
                job.Mesh = m.Clone();
            }

            if (obj.TryGetProperty("outputs", out JsonElement outputs))
            {
                if (outputs.ValueKind != JsonValueKind.Object)
                {
                    throw Bad("field 'outputs' is not an object");
                }

                foreach (JsonProperty p in outputs.EnumerateObject())
                {
                    if (p.Name is not ("vertex" or "primitive"))
                    {
                        throw Bad($"outputs: unknown field '{Clip(p.Name)}'");
                    }
                }

                job.OutputsVertex = ReadOutputList(Require(outputs, "vertex"), "outputs.vertex", true);
                job.OutputsPrimitive = ReadOutputList(Require(outputs, "primitive"), "outputs.primitive", false);
            }

            BuildMeshStage(job);
        }

        if (job.Stage is "vs" or "ps" && job.CodeLength is long cl && cl < 0x90)
        {
            throw Bad($"code_length {cl} leaves no room for a 0x80-byte header and an instruction");
        }
    }

    // The stage description of a task or mesh job (MeshStageInfo), from what split.py derives out of the control section:
    // the mesh object (kind, threads, payload bytes, and for a mesh program has_task, max_vertices, max_primitives) and the
    // output map. The sizes come from here, never from the program header. A mesh program needs its map; a task program's
    // output is its payload, so a map on a task job must be empty (all 20 task jobs of the dump are).
    private static void BuildMeshStage(Job job)
    {
        if (job.Mesh is not JsonElement m)
        {
            throw Bad($"a {job.Stage} job needs its 'mesh' object");
        }

        bool task = job.Stage == "ts";
        string kind = MeshString(m, "kind");

        if (kind != job.Stage)
        {
            throw Bad($"mesh.kind '{Clip(kind)}' is not the job's stage {job.Stage}");
        }

        int threads = MeshInt(m, "threads", 1, MaxThreads);

        if (job.Local[0] != threads || job.Local[1] != 1 || job.Local[2] != 1)
        {
            throw Bad($"local {job.Local[0]}x{job.Local[1]}x{job.Local[2]} is not mesh.threads {threads} by 1 by 1");
        }

        int payload = MeshInt(m, "payload", 0, MaxSharedBytes);
        bool hasTask = false;
        int maxVertices = 0, maxPrimitives = 0;

        if (!task)
        {
            if (job.OutputsVertex == null)
            {
                throw Bad("a mesh job needs its 'outputs' map");
            }

            hasTask = MeshBool(m, "has_task");
            maxVertices = MeshInt(m, "max_vertices", 1, 4096);
            maxPrimitives = MeshInt(m, "max_primitives", 1, 4096);

            foreach ((string field, int[][] list) in new[] { ("vertex_components", job.OutputsVertex), ("primitive_components", job.OutputsPrimitive) })
            {
                if (m.TryGetProperty(field, out JsonElement c) && (c.ValueKind != JsonValueKind.Number || !c.TryGetInt32(out int n) || n != list.Length))
                {
                    throw Bad($"mesh.{field} does not match the outputs map's {list.Length} components");
                }
            }
        }
        else if (job.OutputsVertex != null && (job.OutputsVertex.Length != 0 || job.OutputsPrimitive.Length != 0))
        {
            throw Bad("a task job's output map must be empty: its output is its payload");
        }

        MeshComponent[] Lines(int[][] list)
        {
            MeshComponent[] lines = new MeshComponent[list?.Length ?? 0];

            for (int i = 0; i < lines.Length; i++)
            {
                lines[i] = new MeshComponent(list[i][0], list[i][1]);
            }

            return lines;
        }

        job.MeshPayloadBytes = payload;
        job.MeshOwnShared = m.TryGetProperty("own_shared", out JsonElement own) && own.ValueKind == JsonValueKind.True;
        job.MeshStage = new MeshStageInfo
        {
            Kind = task ? MeshStageKind.Task : MeshStageKind.Mesh,
            Threads = threads,

            // A task's output region holds the group count at byte 4, so it has at least two words
            PayloadWords = Math.Max(task ? 2 : 1, (payload + 3) / 4),
            HasTask = hasTask,
            MaxVertices = maxVertices,
            MaxPrimitives = maxPrimitives,
            VertexLines = task ? Array.Empty<MeshComponent>() : Lines(job.OutputsVertex),
            PrimitiveLines = task ? Array.Empty<MeshComponent>() : Lines(job.OutputsPrimitive),
        };
    }

    private static string MeshString(JsonElement m, string field)
    {
        return m.TryGetProperty(field, out JsonElement v) ? ReadString(v, "mesh." + field) : throw Bad($"missing field 'mesh.{field}'");
    }

    private static int MeshInt(JsonElement m, string field, long min, long max)
    {
        return m.TryGetProperty(field, out JsonElement v) ? (int)ReadInt(v, "mesh." + field, min, max) : throw Bad($"missing field 'mesh.{field}'");
    }

    private static bool MeshBool(JsonElement m, string field)
    {
        return m.TryGetProperty(field, out JsonElement v) ? ReadBool(v, "mesh." + field) : throw Bad($"missing field 'mesh.{field}'");
    }

    private static void ParseAnswers(JsonElement obj, Job job)
    {
        if (obj.TryGetProperty("tex", out JsonElement tex))
        {
            ReadMap(tex, "tex", 4095, job.Tex, (v, where) =>
            {
                string s = ReadString(v, where);

                if (!TypeNames.TryParse(s, out SamplerType t))
                {
                    throw Bad($"{where}: unknown texture type '{Clip(s)}'");
                }

                return t;
            });
        }

        if (obj.TryGetProperty("img", out JsonElement img))
        {
            ReadMap(img, "img", 4095, job.Img, (v, where) =>
            {
                if (v.ValueKind != JsonValueKind.Object)
                {
                    throw Bad($"{where}: not an object");
                }

                SamplerType type = SamplerType.Texture2D;
                string typeName = "2d";
                TextureFormat format = TextureFormat.R8G8B8A8Unorm;

                foreach (JsonProperty p in v.EnumerateObject())
                {
                    switch (p.Name)
                    {
                        case "type":
                            typeName = ReadString(p.Value, where + ".type");

                            if (!TypeNames.TryParse(typeName, out type))
                            {
                                throw Bad($"{where}.type: unknown image type '{Clip(typeName)}'");
                            }

                            break;
                        case "fmt":
                            string f = ReadString(p.Value, where + ".fmt");

                            if (f.Length == 0 || char.IsDigit(f[0]) || !Enum.TryParse(f, false, out format) || !Enum.IsDefined(format))
                            {
                                throw Bad($"{where}.fmt: unknown texture format '{Clip(f)}'");
                            }

                            break;
                        default:
                            throw Bad($"{where}: unknown field '{Clip(p.Name)}'");
                    }
                }

                return new ImageAnswer(type, typeName, format);
            });
        }

        if (obj.TryGetProperty("attr", out JsonElement attr))
        {
            ReadMap(attr, "attr", 31, job.Attr, ReadAttributeType);
        }

        if (obj.TryGetProperty("fout", out JsonElement fout))
        {
            ReadMap(fout, "fout", 7, job.Fout, ReadAttributeType);
        }

        if (obj.TryGetProperty("early_fragment_tests", out JsonElement early))
        {
            job.EarlyFragmentTests = ReadBool(early, "early_fragment_tests");
        }

        if (obj.TryGetProperty("heap_images", out JsonElement hi))
        {
            job.HeapImages = (int)ReadInt(hi, "heap_images", 1, 1 << 24);
        }

        if (obj.TryGetProperty("heap_samplers", out JsonElement hs))
        {
            job.HeapSamplers = (int)ReadInt(hs, "heap_samplers", 1, 1 << 24);
        }
    }

    private static void ParseOptions(JsonElement obj, Job job)
    {
        if (!obj.TryGetProperty("options", out JsonElement opt))
        {
            return;
        }

        if (opt.ValueKind != JsonValueKind.Object)
        {
            throw Bad("field 'options' is not an object");
        }

        bool debug = false, reduced = false, float64 = true, nonUniform = true, fixup = true;
        string variant = "";
        MeshLimits limits = MeshLimits.Default;

        foreach (JsonProperty p in opt.EnumerateObject())
        {
            if (!KnownOptions.Contains(p.Name))
            {
                throw Bad($"unknown option '{Clip(p.Name)}'");
            }

            string where = "options." + p.Name;

            switch (p.Name)
            {
                case "debug": debug = ReadBool(p.Value, where); break;
                case "reduced": reduced = ReadBool(p.Value, where); break;
                case "float64": float64 = ReadBool(p.Value, where); break;
                case "nonuniform_indexing": nonUniform = ReadBool(p.Value, where); break;
                case "capability_fixup": fixup = ReadBool(p.Value, where); break;
                case "variant": variant = ReadString(p.Value, where); break;
                case "mesh_limits": limits = ReadMeshLimits(p.Value, where); break;
            }
        }

        job.Options = new JobOptions
        {
            Debug = debug,
            Reduced = reduced,
            Float64 = float64,
            NonUniformIndexing = nonUniform,
            CapabilityFixup = fixup,
            Variant = variant,
            MeshLimits = limits,
        };
    }

    // The device's task and mesh limits (MeshLimits.Names); a name left out keeps the RTX 3060 Ti's value
    private static MeshLimits ReadMeshLimits(JsonElement v, string where)
    {
        if (v.ValueKind != JsonValueKind.Object)
        {
            throw Bad($"{where} is not an object");
        }

        MeshLimits limits = MeshLimits.Default;

        foreach (JsonProperty p in v.EnumerateObject())
        {
            if (Array.IndexOf(MeshLimits.Names, p.Name) < 0)
            {
                throw Bad($"{where}: unknown limit '{Clip(p.Name)}'");
            }

            limits = limits.With(p.Name, (int)ReadInt(p.Value, $"{where}.{p.Name}", 1, 1 << 24));
        }

        return limits;
    }

    private static AttributeType ReadAttributeType(JsonElement v, string where)
    {
        return ReadString(v, where) switch
        {
            "float" => AttributeType.Float,
            "sint" => AttributeType.Sint,
            "uint" => AttributeType.Uint,
            string s => throw Bad($"{where}: '{Clip(s)}' is not float, sint or uint"),
        };
    }

    private static void ReadMap<T>(JsonElement el, string name, int maxKey, AnswerMap<T> into, Func<JsonElement, string, T> read)
    {
        if (el.ValueKind != JsonValueKind.Object)
        {
            throw Bad($"field '{name}' is not an object");
        }

        foreach (JsonProperty p in el.EnumerateObject())
        {
            string where = $"{name}[{Clip(p.Name)}]";
            T value = read(p.Value, where);

            if (p.Name == "*")
            {
                if (into.HasStar)
                {
                    throw Bad($"{name}: duplicate '*'");
                }

                into.SetStar(value);
            }
            else
            {
                if (!int.TryParse(p.Name, NumberStyles.None, CultureInfo.InvariantCulture, out int slot) || slot > maxKey)
                {
                    throw Bad($"{name}: key '{Clip(p.Name)}' is not a slot number 0..{maxKey} or '*'");
                }

                if (!into.Named.TryAdd(slot, value))
                {
                    throw Bad($"{name}: slot {slot} is given twice");
                }
            }
        }
    }

    // A list of [location, component] pairs: location -1 (the position, when allowed) or 0..31, component 0..3,
    // each pair once.
    private static int[][] ReadOutputList(JsonElement v, string where, bool position)
    {
        if (v.ValueKind != JsonValueKind.Array || v.GetArrayLength() > 4 + 32 * 4)
        {
            throw Bad($"{where} is not an array of at most 132 pairs");
        }

        List<int[]> list = new();
        HashSet<int> seen = new();

        foreach (JsonElement e in v.EnumerateArray())
        {
            int[] pair = ReadIntArray(e, $"{where}[{list.Count}]", 2, position ? -1 : 0, 31);

            if (pair[1] < 0 || pair[1] > 3)
            {
                throw Bad($"{where}[{list.Count}]: component {pair[1]} is outside 0..3");
            }

            if (!seen.Add(pair[0] * 4 + pair[1]))
            {
                throw Bad($"{where}: component {pair[1]} of location {pair[0]} is listed twice");
            }

            list.Add(pair);
        }

        return list.ToArray();
    }

    private static void RejectUnless(JsonElement obj, string field, bool allowed, string stage)
    {
        if (!allowed && obj.TryGetProperty(field, out _))
        {
            throw Bad($"field '{field}' is not valid for stage {stage}");
        }
    }

    private static JsonElement Require(JsonElement obj, string field)
    {
        if (!obj.TryGetProperty(field, out JsonElement v))
        {
            throw Bad($"missing field '{field}'");
        }

        return v;
    }

    private static string RequireString(JsonElement obj, string field)
    {
        return ReadString(Require(obj, field), $"field '{field}'");
    }

    private static string ReadString(JsonElement v, string where)
    {
        if (v.ValueKind != JsonValueKind.String)
        {
            throw Bad($"{where} is not a string");
        }

        return v.GetString();
    }

    private static bool ReadBool(JsonElement v, string where)
    {
        if (v.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw Bad($"{where} is not a boolean");
        }

        return v.GetBoolean();
    }

    private static long ReadInt(JsonElement v, string where, long min, long max)
    {
        if (v.ValueKind != JsonValueKind.Number || !v.TryGetInt64(out long n))
        {
            throw Bad($"{where} is not an integer");
        }

        if (n < min || n > max)
        {
            throw Bad($"{where} = {n} is outside {min}..{max}");
        }

        return n;
    }

    private static int[] ReadIntArray(JsonElement v, string where, int length, long min, long max)
    {
        if (v.ValueKind != JsonValueKind.Array || v.GetArrayLength() != length)
        {
            throw Bad($"field '{where}' is not an array of {length} integers");
        }

        int[] result = new int[length];
        int i = 0;

        foreach (JsonElement e in v.EnumerateArray())
        {
            result[i] = (int)ReadInt(e, $"{where}[{i}]", min, max);
            i++;
        }

        return result;
    }

    private static byte[] ReadHex(string s)
    {
        if (s.Length % 2 != 0)
        {
            throw Bad("c1 has an odd number of hex digits");
        }

        try
        {
            return Convert.FromHexString(s);
        }
        catch (FormatException)
        {
            throw Bad("c1 is not a hex string");
        }
    }

    private static string Clip(string s)
    {
        return s.Length <= 40 ? s : s.Substring(0, 40) + "...";
    }

    private static JobException Bad(string message)
    {
        return new JobException(Kinds.BadJob, message);
    }
}
