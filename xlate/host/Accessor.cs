// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Common.Memory;
using Ryujinx.Graphics.Shader;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Oot2.Xlate;

/// <summary>One distinct accessor query, with how many times the translator asked it.</summary>
internal sealed class QueryRecord
{
    public string Kind { get; init; }
    public int Handle { get; init; }
    public int CbufSlot { get; init; }
    public int Slot { get; init; }
    public int ImageSlot { get; init; }
    public string Answer { get; init; }
    public string From { get; init; }
    public int Count { get; set; } = 1;
}

/// <summary>
/// The translator's view of the machine and of the draw: every question the
/// shader translator asks, answered from the job or from the host's fixed
/// Vulkan 1.3 profile (docs/architecture.md, D13 and D21). One instance serves
/// one translation pass; the second pass of the texture type recovery gets a
/// fresh instance, because translation mutates the decoded program.
/// Adapted from nxnative dm/gfx/tools/dmsh_xlate/Program.cs, class Accessor (MIT):
/// the code and c1 answers, the slot-based type tables and the two-pass recovery.
/// The depth clip control and non-constant texture offset answers follow DKB
/// tools/xlate/Program.cs (MIT).
/// </summary>
internal sealed class Accessor : IGpuAccessor
{
    // Bindings follow the layout `uni` (UniLayout): buffers and storage images in fixed
    // slots of the stage at set 1 (set 2 for a task program), bound textures and texture
    // arrays in push constant slots read from the global arrays of set 0, and the
    // translator's pool arrays (a texture array as long as the global image heap, or the
    // sampler heap) at set 0, bindings 0 and 1.

    // Handle words in c[2]: one every 8 bytes up to the end of the separate
    // sampler table (128 slots), 301 in all. An array of c[2] handles that a
    // shader indexes at run time has one element per word.
    private const int C2HandleWords = (0x568 + 8 * 128) / 8;
    private const int OtherBufferArrayLength = 32;
    private const int MaxQueryRecords = 20000;
    private const int MaxNotes = 2000;

    private readonly Job _job;
    private readonly byte[] _code;
    private readonly byte[] _c1;
    private readonly Dictionary<int, SamplerType> _imageTypes;
    private readonly Dictionary<int, SamplerType> _textureTypes;
    private readonly HashSet<int> _guessedHandles = new();
    private readonly Dictionary<string, QueryRecord> _queryIndex = new();
    private readonly UniLayout _layout;
    private int _droppedNotes;
    private int _overflowNoted;

    public List<string> Notes { get; } = new();
    public List<QueryRecord> Queries { get; } = new();
    public SortedSet<int> Fetch1D { get; } = new();
    public int DroppedQueries { get; private set; }

    /// <summary>A query asked for the type of an image handle and nothing in the job answered it.</summary>
    public bool ImageQueryGuessed { get; private set; }

    /// <param name="imageTypes">Types of image handles found by an earlier pass, by handle; or null.</param>
    /// <param name="textureTypes">Types of texture slots found by an earlier pass; or null.</param>
    public Accessor(Job job, byte[] code, Dictionary<int, SamplerType> imageTypes = null, Dictionary<int, SamplerType> textureTypes = null)
    {
        _job = job;
        _code = code;
        _c1 = job.C1;
        _imageTypes = imageTypes;
        _textureTypes = textureTypes;
        _layout = new UniLayout(job.Stage);
    }

    /// <summary>The bindings this pass gave, and what did not fit its stage's slots.</summary>
    public UniLayout Layout => _layout;

    public int DroppedNotes => _droppedNotes;

    // ---- recovery of texture types from the shader's own instructions ----

    // A texture query on an image handle (TXQ on c2 0x120 + 8 r) asks the type of
    // that handle, which a job without answers does not give, and the 2D default
    // would take a 3D image's depth from its width. The shader's own surface
    // instructions give the image's type: the first pass records that a query
    // went unanswered (ImageQueryGuessed) and the host translates again with the
    // types of the images it found, by handle.
    public static Dictionary<int, SamplerType> ImageTypesOf(ShaderProgram p)
    {
        Dictionary<int, SamplerType> types = new();

        foreach (TextureDescriptor im in p.Info.Images)
        {
            if (im.CbufSlot == 2 || im.CbufSlot < 0)
            {
                types[im.HandleIndex] = im.Type & (SamplerType.Mask | SamplerType.Array);
            }
        }

        return types;
    }

    // The same for a texture: a handle ORs a texture word and a sampler word and
    // a size query has another sampler word than the sample, so the translator's
    // merge by handle keeps them apart and the query gets its own, 2D-typed
    // binding. So the types go by texture slot. A binding whose handle a query
    // guessed counts only when it is not the guess: then an instruction typed it.
    // A slot two instructions give different types is left out. Only a slot a
    // query guessed and another instruction types otherwise is returned.
    public Dictionary<int, SamplerType> TextureTypesOf(ShaderProgram p)
    {
        Dictionary<int, SamplerType> types = new();
        HashSet<int> mixed = new();

        foreach (TextureDescriptor t in p.Info.Textures)
        {
            if (t.ArrayLength != 1)
            {
                continue;
            }

            SamplerType ty = t.Type & (SamplerType.Mask | SamplerType.Array | SamplerType.Multisample);
            int slot = HandleSlots.Slot(t.HandleIndex, t.CbufSlot, false);

            if (slot < 0 || (ty == SamplerType.Texture2D && _guessedHandles.Contains(t.HandleIndex)))
            {
                continue;
            }

            if (types.TryGetValue(slot, out SamplerType seen) && seen != ty)
            {
                mixed.Add(slot);
            }
            else
            {
                types[slot] = ty;
            }
        }

        HashSet<int> guessed = new(_guessedHandles.Select(h => HandleSlots.Slot(h, -1, false)));

        return types
            .Where(kv => !mixed.Contains(kv.Key) && guessed.Contains(kv.Key) && kv.Value != SamplerType.Texture2D)
            .ToDictionary(kv => kv.Key, kv => kv.Value);
    }

    // ---- code and driver banks ----

    // An address past the code is no code: the jump table scan ends on it, and the
    // decoder stops there.
    public ReadOnlySpan<ulong> GetCode(ulong address, int minimumSize)
    {
        if (address >= (ulong)_code.Length)
        {
            return default;
        }

        return MemoryMarshal.Cast<byte, ulong>(new ReadOnlySpan<byte>(_code)[(int)address..]);
    }

    // c1 holds the jump tables of indirect branches (BRX); without it the
    // translator cannot resolve their targets.
    public int QueryConstantBuffer1Size() => _c1.Length;

    public uint ConstantBuffer1Read(int offset)
    {
        if (offset >= 0 && offset + 4 <= _c1.Length)
        {
            return BitConverter.ToUInt32(_c1, offset);
        }

        AddNote($"c1 read out of bounds +{offset:X} (c1 length {_c1.Length})");

        return 0;
    }

    public int[] QueryAmpereComputeC0Layout() => _job.Nvn2Cs;

    // A task or mesh job, as the stage the translator emits (Job.BuildMeshStage); the job's stage_memory offsets are no longer read
    public MeshStageInfo QueryAmpereMeshStage() => _job.MeshStage;

    // ---- bindings ----

    public SetBindingPair CreateConstantBufferBinding(int index) => Fit(_layout.NextUniform(index));

    public SetBindingPair CreateStorageBufferBinding(int index) => Fit(_layout.NextStorage(index));

    public SetBindingPair CreateTextureBinding(int count, bool isBuffer)
    {
        if (count > 1 && count == _job.HeapImages)
        {
            return new(UniLayout.GlobalSet, UniLayout.ImagesBinding);
        }

        if (count > 1 && count == _job.HeapSamplers)
        {
            return new(UniLayout.GlobalSet, UniLayout.SamplersBinding);
        }

        return Fit(isBuffer ? _layout.TexelBuffer() : _layout.NextTextureSlot(count));
    }

    public SetBindingPair CreateImageBinding(int count, bool isBuffer) => Fit(_layout.NextImage(count, isBuffer));

    public int QuerySupportBufferSet() => UniLayout.PushSet;

    public PushTextureLayout QueryPushTextureLayout() => UniLayout.PushTextures(_job.HeapImages, _job.HeapSamplers);

    private SetBindingPair Fit(SetBindingPair binding)
    {
        if (_layout.Overflow.Count > _overflowNoted)
        {
            AddNote("layout uni: " + _layout.Overflow[_overflowNoted++]);
        }

        return binding;
    }

    // Pool arrays reach set 0 through CreateTextureBinding by their length, so
    // no extra set is requested per array.
    public int CreateExtraSet() => -1;

    public int QueryTextureArrayLengthFromPool() => _job.HeapImages;

    public int QuerySamplerArrayLengthFromPool() => _job.HeapSamplers;

    public int QueryTextureArrayLengthFromBuffer(int slot) => slot == 2 ? C2HandleWords : OtherBufferArrayLength;

    // ---- compute shape ----

    public int QueryComputeLocalSizeX() => _job.Local?[0] ?? 1;

    public int QueryComputeLocalSizeY() => _job.Local?[1] ?? 1;

    public int QueryComputeLocalSizeZ() => _job.Local?[2] ?? 1;

    public int QueryComputeSharedMemorySize() => _job.Shared;

    // ---- types and formats ----

    public SamplerType QuerySamplerType(int handle, int cbufSlot = -1)
    {
        int slot = HandleSlots.Slot(handle, cbufSlot, false);
        int islot = HandleSlots.Slot(handle, cbufSlot, true);
        string from = "job";
        SamplerType? answer = null;

        // A handle in the image table is answered from "img" only. HandleSlots.Slot also
        // maps it to texture slots 32..39 (nxnative's combined range runs through the image
        // table), and a "tex" wildcard must not answer for an image.
        if (islot >= 0)
        {
            if (_job.Img.TryGet(islot, out ImageAnswer ia))
            {
                answer = ia.Type;
            }
        }
        else if (slot >= 0 && _job.Tex.TryGet(slot, out SamplerType tt))
        {
            answer = tt;
        }

        if (answer == null && islot >= 0 && cbufSlot < 0)
        {
            if (_imageTypes != null && _imageTypes.TryGetValue(handle, out SamplerType it))
            {
                AddNote($"texture handle {handle:X} is image slot {islot}: type {TypeNames.Describe(it)} from the shader's image instructions");
                Record("sampler_type", handle, cbufSlot, slot, islot, TypeNames.Describe(it), "recovered");

                return it;
            }

            ImageQueryGuessed = true;
        }

        if (answer == null && slot >= 0 && islot < 0)
        {
            if (_textureTypes != null && _textureTypes.TryGetValue(slot, out SamplerType rt))
            {
                AddNote($"texture handle {handle:X} is texture slot {slot}: type {TypeNames.Describe(rt)} from the shader's other instructions");
                Record("sampler_type", handle, cbufSlot, slot, islot, TypeNames.Describe(rt), "recovered");

                return rt;
            }

            _guessedHandles.Add(handle);
        }

        if (answer == null)
        {
            AddNote($"texture handle {handle:X} (constant buffer {cbufSlot}, slot {slot}) has no type in the job, using 2D");
            answer = SamplerType.Texture2D;
            from = "default";
        }

        Record("sampler_type", handle, cbufSlot, slot, islot, TypeNames.Describe(answer.Value), from);

        return answer.Value;
    }

    // A 1D integer fetch the translator made a wrapped 2D read of, for want of a
    // type answer: the slot may hold a 1D texture or a texture buffer. The answer
    // "buffer" in the job's tex table is the variant that settles it.
    public void RecordTexelFetch1D(int handle)
    {
        int slot = HandleSlots.Slot(handle, -1, false);

        if (slot >= 0)
        {
            Fetch1D.Add(slot);
        }

        Record("fetch1d", handle, -1, slot, -1, "wrapped_2d", "default");
    }

    public TextureFormat QueryTextureFormat(int handle, int cbufSlot = -1)
    {
        int islot = HandleSlots.Slot(handle, cbufSlot, true);

        if (_job.Img.TryGet(islot, out ImageAnswer ia))
        {
            Record("texture_format", handle, cbufSlot, -1, islot, ia.Format.ToString(), "job");

            return ia.Format;
        }

        AddNote($"image handle {handle:X} slot {islot} has no format in the job, using R8G8B8A8Unorm");
        Record("texture_format", handle, cbufSlot, -1, islot, nameof(TextureFormat.R8G8B8A8Unorm), "default");

        return TextureFormat.R8G8B8A8Unorm;
    }

    public void RegisterTexture(int handle, int cbufSlot)
    {
        Record("register_texture", handle, cbufSlot, HandleSlots.Slot(handle, cbufSlot, false), HandleSlots.Slot(handle, cbufSlot, true), "", "");
    }

    // ---- graphics state ----

    public GpuGraphicsState QueryGraphicsState()
    {
        Array32<AttributeType> attr = default;
        Array8<AttributeType> fout = default;

        for (int i = 0; i < 32; i++)
        {
            attr[i] = _job.Attr.TryGet(i, out AttributeType a) ? a : AttributeType.Float;
        }

        for (int i = 0; i < 8; i++)
        {
            fout[i] = _job.Fout.TryGet(i, out AttributeType f) ? f : AttributeType.Float;
        }

        // The depth convention is host state (VK_EXT_depth_clip_control), the
        // window origin is upper left, Y is flipped by the viewport, and alpha
        // test, dual source blending and point size are per-draw variants.
        return new GpuGraphicsState(
            _job.EarlyFragmentTests,
            InputTopology.Triangles,
            false,
            TessPatchType.Triangles,
            TessSpacing.EqualSpacing,
            false,
            false,
            false,
            false,
            false,
            1f,
            AlphaTestOp.Always,
            0f,
            attr,
            false,
            fout,
            false,
            false,
            true,
            false);
    }

    // ---- host capabilities: Vulkan 1.3 on an NVIDIA desktop GPU ----

    // The translator declares FragmentShaderPixelInterlockEXT for every fragment
    // shader when this is true; its only source (Maxwell function matching) does
    // not run for Ampere code, so a device feature would be asked for nothing.
    public bool QueryHostSupportsFragmentShaderInterlock() => false;

    // Constant texture offsets stay ConstOffset; only computed offsets are emulated.
    public bool QueryHostSupportsNonConstantTextureOffset() => false;

    public bool QueryHostSupportsDepthClipControl() => true;

    public bool QueryHostSupportsShaderFloat64() => _job.Options.Float64;

    public bool QueryHostReducedPrecision() => _job.Options.Reduced;

    // HMMA on VK_KHR_cooperative_matrix (the job option hmma, JobOptions.Hmma)
    public bool QueryHostCooperativeHmma() => _job.Options.Hmma == JobOptions.HmmaCoop;

    public bool QueryHostSupportsShaderNonUniformIndexing() => _job.Options.NonUniformIndexing;

    // The host binds each storage buffer at its guest address rounded down to the
    // device's minimum offset alignment (16 on NVIDIA).
    public bool QueryHasUnalignedStorageBuffer() => true;

    public int QueryHostStorageBufferOffsetAlignment() => 16;

    public int QueryHostSubgroupSize() => 32;

    // No geometry stage exists in the game; the answer only matters for one.
    public bool QueryHostSupportsGeometryShaderPassthrough() => false;

    public void Log(string message)
    {
        AddNote("translator: " + message);
    }

    public void AddNote(string note)
    {
        if (Notes.Count >= MaxNotes)
        {
            _droppedNotes++;

            return;
        }

        Notes.Add(note);
    }

    private void Record(string kind, int handle, int cbufSlot, int slot, int islot, string answer, string from)
    {
        string key = $"{kind}|{handle}|{cbufSlot}|{answer}|{from}";

        if (_queryIndex.TryGetValue(key, out QueryRecord existing))
        {
            existing.Count++;

            return;
        }

        if (Queries.Count >= MaxQueryRecords)
        {
            DroppedQueries++;

            return;
        }

        QueryRecord record = new()
        {
            Kind = kind,
            Handle = handle,
            CbufSlot = cbufSlot,
            Slot = slot,
            ImageSlot = islot,
            Answer = answer,
            From = from,
        };

        _queryIndex[key] = record;
        Queries.Add(record);
    }
}
