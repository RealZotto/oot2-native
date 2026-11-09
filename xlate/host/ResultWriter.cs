// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;

namespace Oot2.Xlate;

/// <summary>The measurements of one job that go into res.json and the reply.</summary>
internal sealed class JobTiming
{
    public long TranslatorUs;
    public long DecodeUs;
    public long TranslateUs;
    public long PostUs;
    public long TotalUs;
    public long AllocBytes;
    public int[] Gc = new int[3];
    public bool Cold;
}

/// <summary>Builds <key>.res.json (schema 2) and writes a job's two files.</summary>
internal static class ResultWriter
{
    public const int Schema = 2;
    private const int ConstantBufferBytes = 65536;
    private const int SupportBufferBytes = 1392;

    public static string Sha256Hex(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    public static byte[] BuildResJson(Job job, TranslationOutcome outcome, FixupResult fix, byte[] spirv, string translatorId, JobTiming timing)
    {
        using MemoryStream ms = new();
        using (Utf8JsonWriter w = new(ms, new JsonWriterOptions { Indented = true }))
        {
            ShaderProgramInfo info = outcome.Info;
            Accessor acc = outcome.Accessor;
            UniLayout layout = acc.Layout;
            ModuleInterface io = ModuleInterface.Read(spirv, UniLayout.PushSet, UniLayout.SupportBinding);

            w.WriteStartObject();
            w.WriteNumber("schema", Schema);
            w.WriteString("key", job.Key);
            w.WriteString("stage", job.Stage);
            w.WriteString("exec_model", SpirvFixup.ExecutionModelName(fix.ExecutionModel));
            w.WriteString("entry", "main");
            w.WriteString("translator_id", translatorId);
            WriteOptions(w, job);
            WriteAnswers(w, job);
            w.WriteNumber("spv_bytes", spirv.Length);
            w.WriteString("spv_sha256", Sha256Hex(spirv));

            WriteLayout(w, layout, info);
            WriteSupportBuffer(w, io);

            WriteStringArray(w, "capabilities", fix.Capabilities, SpirvFixup.CapabilityName);
            WriteStringArray(w, "capabilities_added", fix.Added, SpirvFixup.CapabilityName);
            WriteStringArray(w, "extensions", fix.Extensions, s => s);

            w.WriteStartArray("cbufs");

            foreach (BufferDescriptor b in info.CBuffers)
            {
                w.WriteStartObject();
                w.WriteNumber("set", b.Set);
                w.WriteNumber("binding", b.Binding);
                w.WriteNumber("slot", b.Slot);
                w.WriteNumber("ordinal", Ordinal(b.Set, b.Binding, layout.BufferSet, layout.Uniform));
                w.WriteNumber("size", ConstantBufferBytes);
                w.WriteString("role", b.Slot switch { 0 => "driver0", 2 => "driver2", 24 => "push", _ => "user" });
                w.WriteEndObject();
            }

            w.WriteEndArray();

            w.WriteStartArray("sbufs");

            foreach (BufferDescriptor b in info.SBuffers)
            {
                w.WriteStartObject();
                w.WriteNumber("set", b.Set);
                w.WriteNumber("binding", b.Binding);
                w.WriteNumber("ordinal", b.Slot);
                w.WriteNumber("cb_slot", b.SbCbSlot);
                w.WriteNumber("cb_word_offset", b.SbCbOffset);
                w.WriteNumber("cb_byte_offset", b.SbCbOffset * 4);
                w.WriteString("role", "guest");
                // The storage buffer contract (storage_buffer_contract below)
                w.WriteNumber("address_byte_offset", b.SbCbOffset * 4);
                w.WriteNumber("size_byte_offset", b.SbCbOffset * 4 + 8);
                w.WriteBoolean("write", (b.Flags & BufferUsageFlags.Write) != 0);
                WriteAccess(w, Find(fix, b.Set, b.Binding, image: false));
                w.WriteEndObject();
            }

            w.WriteEndArray();

            WriteStorageBufferContract(w);

            // Raw global access (architecture D15): the translator emits physical-pointer code that walks the page table whose device address and
            // span are the first four push constant words (RawTableAbi, mirrored by runtime/include/oot2/raw_table.h). No buffer is bound for it.
            // page_table: the module reads those words and the table; store_bits: a store or an atomic through it sets its leaf slot's store bit;
            // miss_log: a miss of any access, a load included, takes an entry of the table's miss log, so the table is written whatever the module does
            w.WriteStartObject("raw_access");
            w.WriteBoolean("physical_pointers", fix.Capabilities.Contains(SpirvFixup.CapPhysicalStorageBufferAddresses));
            w.WriteBoolean("page_table", info.UsesRawTable);
            w.WriteBoolean("store_bits", info.RawStores);
            w.WriteBoolean("miss_log", info.UsesRawTable);
            w.WriteNumber("abi_version", RawTableAbi.Version);
            w.WriteNumber("push_offset", RawTableAbi.PushTableLow * 4);
            w.WriteNumber("push_bytes", RawTableAbi.PushBytes);
            w.WriteEndObject();

            WriteTextures(w, "textures", info.Textures, false, fix);
            WriteTextures(w, "images", info.Images, true, fix);

            // Whether a storage buffer or image is declared writable: a vertex or
            // fragment module then needs vertexPipelineStoresAndAtomics or
            // fragmentStoresAndAtomics (VUID-RuntimeSpirv-NonWritable-06341, -06340).
            bool writable = false;
            foreach (DescriptorAccess a in fix.Access) { writable |= !a.NonWritable; }
            // The raw helper's miss path (a load's included) writes the table's miss log, so a raw-access module is writable whatever it does
            writable |= info.UsesRawTable;
            w.WriteBoolean("writable_storage", writable);

            // Descriptors the module's draw pushes at set 1 (the support buffer, buffers, storage images), those a task
            // program has in its own set 2, and the push constant slots it reads
            int pushed = 1, taskSet = 0, slots = 0;

            void Tally(int set)
            {
                if (set == UniLayout.TaskSet) { taskSet++; } else { pushed++; }
            }

            foreach (BufferDescriptor b in info.CBuffers) { Tally(b.Set); }
            foreach (BufferDescriptor b in info.SBuffers) { Tally(b.Set); }
            foreach (TextureDescriptor t in info.Textures) { if (t.Set == UniLayout.SlotSet) { slots++; } else if (t.Set != UniLayout.GlobalSet) { Tally(t.Set); } }
            foreach (TextureDescriptor t in info.Images) { if (t.Set != UniLayout.GlobalSet) { Tally(t.Set); } }

            w.WriteNumber("push_descriptors", pushed);
            w.WriteNumber("task_set_descriptors", taskSet);
            w.WriteNumber("texture_slots", slots);

            // A task or mesh program is a stage of its own (mesh_stage below), not a compute shader
            if (job.Stage == "cs")
            {
                w.WriteStartObject("compute");
                w.WriteStartArray("local_size");
                foreach (int n in job.Local) { w.WriteNumberValue(n); }
                w.WriteEndArray();
                w.WriteNumber("shared_bytes", job.Shared);
                w.WriteEndObject();
            }
            else
            {
                w.WriteNull("compute");
            }

            if (job.Mesh is JsonElement mesh)
            {
                w.WritePropertyName("mesh");
                mesh.WriteTo(w);
            }
            else
            {
                w.WriteNull("mesh");
            }

            WriteMeshStage(w, job, info);

            if (job.OutputsVertex != null)
            {
                w.WriteStartObject("outputs");
                WriteOutputList(w, "vertex", job.OutputsVertex);
                WriteOutputList(w, "primitive", job.OutputsPrimitive);
                w.WriteEndObject();
            }
            else
            {
                w.WriteNull("outputs");
            }

            WriteInterface(w, job, io);

            // uses_frag_coord also counts a module that only reads FragCoord.w (every perspective
            // interpolation does): the translator's own flag leaves those out (spike S2, B6)
            w.WriteStartObject("graphics");
            w.WriteBoolean("uses_frag_coord", info.UsesFragCoord || io.BuiltInsIn.Contains("FragCoord"));
            w.WriteBoolean("uses_instance_id", info.UsesInstanceId);
            w.WriteBoolean("uses_draw_parameters", info.UsesDrawParameters);
            w.WriteBoolean("uses_rt_layer", info.UsesRtLayer);
            w.WriteNumber("clip_distances_written", info.ClipDistancesWritten);
            w.WriteNumber("frag_out_map", info.FragmentOutputMap);
            w.WriteEndObject();

            w.WriteStartArray("fetch1d");
            foreach (int slot in acc.Fetch1D) { w.WriteNumberValue(slot); }
            w.WriteEndArray();

            w.WriteStartArray("queries");

            foreach (QueryRecord q in acc.Queries)
            {
                w.WriteStartObject();
                w.WriteString("kind", q.Kind);
                w.WriteNumber("handle", q.Handle);
                w.WriteNumber("cbuf_slot", q.CbufSlot);
                w.WriteNumber("slot", q.Slot);
                w.WriteNumber("image_slot", q.ImageSlot);
                w.WriteString("answer", q.Answer);
                w.WriteString("answered_from", q.From);
                w.WriteNumber("count", q.Count);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteNumber("queries_dropped", acc.DroppedQueries);

            w.WriteStartArray("notes");
            foreach (string n in acc.Notes) { w.WriteStringValue(n); }
            w.WriteEndArray();
            w.WriteNumber("notes_dropped", acc.DroppedNotes);

            w.WriteNumber("passes", outcome.Passes);

            w.WriteStartObject("timing");
            w.WriteNumber("translator_us", timing.TranslatorUs);
            w.WriteNumber("decode_us", timing.DecodeUs);
            w.WriteNumber("translate_us", timing.TranslateUs);
            w.WriteNumber("post_us", timing.PostUs);
            w.WriteNumber("total_us", timing.TotalUs);
            w.WriteNumber("alloc_bytes", timing.AllocBytes);
            w.WriteStartArray("gc");
            foreach (int g in timing.Gc) { w.WriteNumberValue(g); }
            w.WriteEndArray();
            w.WriteBoolean("cold", timing.Cold);
            w.WriteEndObject();

            w.WriteEndObject();
        }

        return ms.ToArray();
    }

    private static void WriteOutputList(Utf8JsonWriter w, string name, int[][] pairs)
    {
        w.WriteStartArray(name);

        foreach (int[] pair in pairs)
        {
            w.WriteStartArray();
            w.WriteNumberValue(pair[0]);
            w.WriteNumberValue(pair[1]);
            w.WriteEndArray();
        }

        w.WriteEndArray();
    }

    private static void WriteOptions(Utf8JsonWriter w, Job job)
    {
        JobOptions o = job.Options;

        w.WriteStartObject("options");
        w.WriteBoolean("debug", o.Debug);
        w.WriteBoolean("reduced", o.Reduced);
        w.WriteBoolean("float64", o.Float64);
        w.WriteBoolean("nonuniform_indexing", o.NonUniformIndexing);
        w.WriteBoolean("capability_fixup", o.CapabilityFixup);
        w.WriteString("variant", o.Variant);
        w.WriteEndObject();
    }

    private static void WriteAnswers(Utf8JsonWriter w, Job job)
    {
        w.WriteStartObject("answers");
        WriteMap(w, "tex", job.Tex, (writer, v) => writer.WriteStringValue(TypeNames.Describe(v)));
        WriteMap(w, "img", job.Img, (writer, v) =>
        {
            writer.WriteStartObject();
            writer.WriteString("type", v.TypeName);
            writer.WriteString("fmt", v.Format.ToString());
            writer.WriteEndObject();
        });
        WriteMap(w, "attr", job.Attr, (writer, v) => writer.WriteStringValue(v.ToString().ToLowerInvariant()));
        WriteMap(w, "fout", job.Fout, (writer, v) => writer.WriteStringValue(v.ToString().ToLowerInvariant()));
        w.WriteBoolean("early_fragment_tests", job.EarlyFragmentTests);
        w.WriteNumber("heap_images", job.HeapImages);
        w.WriteNumber("heap_samplers", job.HeapSamplers);
        w.WriteEndObject();
    }

    private static void WriteMap<T>(Utf8JsonWriter w, string name, AnswerMap<T> map, Action<Utf8JsonWriter, T> writeValue)
    {
        w.WriteStartObject(name);

        if (map.HasStar)
        {
            w.WritePropertyName("*");
            writeValue(w, map.Star);
        }

        foreach (KeyValuePair<int, T> kv in map.Named)
        {
            w.WritePropertyName(kv.Key.ToString(System.Globalization.CultureInfo.InvariantCulture));
            writeValue(w, kv.Value);
        }

        w.WriteEndObject();
    }

    private static void WriteStringArray<T>(Utf8JsonWriter w, string name, IEnumerable<T> items, Func<T, string> text)
    {
        w.WriteStartArray(name);

        foreach (T item in items)
        {
            w.WriteStringValue(text(item));
        }

        w.WriteEndArray();
    }

    /// <summary>The post-pass's view of a storage buffer or image variable, or null when the module has none at that binding.</summary>
    public static DescriptorAccess Find(FixupResult fix, int set, int binding, bool image)
    {
        foreach (DescriptorAccess a in fix.Access)
        {
            if (a.Set == set && a.Binding == binding && a.Image == image)
            {
                return a;
            }
        }

        return null;
    }

    // The support buffer's fields (Ryujinx SupportBuffer, std140), in member order: name, byte offset, bytes, and what the
    // host writes there for a draw of this project (spike S2, B5): no alpha test, no BGRA target, the viewport's inverse
    // and size, every render scale 1.0 (or FragCoord X and Y become infinite), no transform feedback.
    private static readonly (string Name, int Offset, int Bytes, string Host)[] SupportFields =
    {
        ("alpha_test", SupportBuffer.FragmentAlphaTestOffset, 16, "0: no alpha test (a draw variant tests alpha instead)"),
        ("is_bgra", SupportBuffer.FragmentIsBgraOffset, 16 * SupportBuffer.FragmentIsBgraCount, "0 for every colour target"),
        ("viewport_inverse", SupportBuffer.ViewportInverseOffset, 16, "2 / width, 2 / height, 1, 0"),
        ("viewport_size", SupportBuffer.ViewportSizeOffset, 16, "width, height, 1, 0"),
        ("frag_scale_count", SupportBuffer.FragmentRenderScaleCountOffset, 16, "0"),
        ("render_scale", SupportBuffer.GraphicsRenderScaleOffset, 16 * SupportBuffer.RenderScaleMaxCount, "1.0 in every element"),
        ("tfe_offset", SupportBuffer.TfeOffsetOffset, 16, "0"),
        ("tfe_vertex_count", SupportBuffer.TfeVertexCountOffset, 16, "0"),
    };

    private static void WriteLayout(Utf8JsonWriter w, UniLayout layout, ShaderProgramInfo info)
    {
        w.WriteStartObject("layout");
        w.WriteString("name", UniLayout.Name);
        w.WriteString("role", layout.RoleName);
        w.WriteNumber("global_set", UniLayout.GlobalSet);
        w.WriteNumber("images_binding", UniLayout.ImagesBinding);
        w.WriteNumber("samplers_binding", UniLayout.SamplersBinding);
        w.WriteNumber("push_set", UniLayout.PushSet);
        w.WriteNumber("buffer_set", layout.BufferSet);
        WriteRange(w, "uniform_slots", layout.Uniform);
        WriteRange(w, "storage_slots", layout.Storage);
        WriteRange(w, "storage_image_slots", layout.Image);
        w.WriteNumber("push_constant_bytes", UniLayout.PushConstantBytes);
        w.WriteNumber("texture_slots_offset", UniLayout.SlotsOffset);
        WriteRange(w, "texture_slots", layout.Slots);
        // Push constant bytes kept for the draw's first task (task and mesh): words 6 and 7 are the device address of the word that holds it
        // (MeshAbi), read by a task and by a mesh stage without a task (mesh_stage.first_task)
        WriteRange(w, "first_task_bytes", layout.FirstTask);
        w.WriteBoolean("fits", layout.Overflow.Count == 0);
        WriteStringArray(w, "overflow", layout.Overflow, s => s);
        w.WriteEndObject();
    }

    private static void WriteRange(Utf8JsonWriter w, string name, (int First, int Count) range)
    {
        w.WriteStartArray(name);
        w.WriteNumberValue(range.First);
        w.WriteNumberValue(range.Count);
        w.WriteEndArray();
    }

    // The position of a buffer in its stage's slots (the n-th constant bank in the n-th uniform slot), or -1 past them.
    private static int Ordinal(int set, int binding, int bufferSet, (int First, int Count) range)
    {
        return set == bufferSet && binding >= range.First && binding < range.First + range.Count ? binding - range.First : -1;
    }

    private static void WriteSupportBuffer(Utf8JsonWriter w, ModuleInterface io)
    {
        w.WriteStartObject("support_buffer");
        w.WriteNumber("set", UniLayout.PushSet);
        w.WriteNumber("binding", UniLayout.SupportBinding);
        w.WriteNumber("size", SupportBufferBytes);
        w.WriteStartArray("fields");

        for (int i = 0; i < SupportFields.Length; i++)
        {
            w.WriteStartObject();
            w.WriteString("name", SupportFields[i].Name);
            w.WriteNumber("offset", SupportFields[i].Offset);
            w.WriteNumber("bytes", SupportFields[i].Bytes);
            w.WriteString("host_value", SupportFields[i].Host);
            w.WriteBoolean("read", io.SupportFieldsRead.Contains(i));
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
    }

    // How the host binds a storage buffer whose address the program reads from a constant bank (Accessor,
    // QueryHasUnalignedStorageBuffer): the 64-bit address at address_byte_offset of bank cb_slot and the 32-bit size
    // at size_byte_offset; the descriptor starts at the address rounded down to the alignment and covers the size plus
    // the bytes rounded off; the shader subtracts that base from the low 32 bits of the address it reads.
    private static void WriteStorageBufferContract(Utf8JsonWriter w)
    {
        w.WriteStartObject("storage_buffer_contract");
        w.WriteNumber("alignment", 16);
        w.WriteString("address", "u64 at address_byte_offset of constant bank cb_slot");
        w.WriteString("size", "u32 at size_byte_offset (address_byte_offset + 8)");
        w.WriteString("descriptor_offset", "address & ~(alignment - 1)");
        w.WriteString("descriptor_range", "size + (address & (alignment - 1))");
        w.WriteString("shader_reads", "the low 32 bits of the address, less the descriptor offset's low 32 bits");
        w.WriteEndObject();
    }

    // A task or mesh program as VK_EXT_mesh_shader stage (null for other stages): what the pipeline builder and the draw need to know
    // without reading the module. payload_bytes is what the stage declares (the control section's payload, at least 8 for a task, 0
    // for a mesh program without a task); output_bytes is Vulkan's formula for the mesh outputs (every location four scalars, counts
    // rounded to 32); shared_bytes the staging in workgroup memory; the limits are those the program was checked against.
    // requires_subgroup_32 is a pipeline creation flag the back end sets for these stages only (the mesh shader study)
    private static void WriteMeshStage(Utf8JsonWriter w, Job job, ShaderProgramInfo info)
    {
        if (job.MeshStage is not MeshStageInfo mesh)
        {
            w.WriteNull("mesh_stage");

            return;
        }

        MeshSizes sizes = MeshChecks.Measure(mesh);

        w.WriteStartObject("mesh_stage");
        w.WriteString("kind", mesh.Kind == MeshStageKind.Task ? "task" : "mesh");
        w.WriteNumber("threads", mesh.Threads);
        w.WriteBoolean("has_task", mesh.HasTask);
        w.WriteNumber("max_vertices", mesh.MaxVertices);
        w.WriteNumber("max_primitives", mesh.MaxPrimitives);
        w.WriteNumber("payload_bytes", sizes.PayloadBytes);
        w.WriteNumber("output_bytes", sizes.OutputBytes);
        w.WriteNumber("vertex_locations", sizes.VertexLocations);
        w.WriteNumber("primitive_locations", sizes.PrimitiveLocations);
        w.WriteNumber("vertex_components", mesh.VertexLines.Count);
        w.WriteNumber("primitive_components", mesh.PrimitiveLines.Count);
        w.WriteNumber("shared_bytes", sizes.SharedBytes);
        w.WriteBoolean("requires_subgroup_32", true);

        // The draw's first task (MeshAbi): carried as the device address of one 32-bit word, in the push constants at push_offset; read is whether this module
        // adds it to its workgroup number (a task, or a mesh stage without a task, that reads the number), in which case it also carries physical addressing
        w.WriteStartObject("first_task");
        w.WriteBoolean("read", info.UsesFirstTask);
        w.WriteString("form", "device_address");
        w.WriteNumber("push_offset", MeshAbi.FirstTaskPushOffset);
        w.WriteNumber("address_bytes", MeshAbi.FirstTaskAddressBytes);
        w.WriteNumber("word_bytes", MeshAbi.FirstTaskWordBytes);
        w.WriteEndObject();

        w.WriteStartObject("limits");

        foreach (string name in MeshLimits.Names)
        {
            w.WriteNumber(name, job.Options.MeshLimits.Get(name));
        }

        w.WriteEndObject();
        w.WriteEndObject();
    }

    private static void WriteInterface(Utf8JsonWriter w, Job job, ModuleInterface io)
    {
        w.WriteStartObject("interface");
        WriteVariables(w, "inputs", io.Inputs, null);
        WriteVariables(w, "outputs", io.Outputs, null);
        WriteStringArray(w, "builtins_in", io.BuiltInsIn, s => s);
        WriteStringArray(w, "builtins_out", io.BuiltInsOut, s => s);
        w.WriteEndObject();

        // A vertex program's inputs are its vertex attributes: the host builds the vertex input state from them, with the
        // attribute type the job answered (attr) beside the module's own declaration.
        if (job.Stage == "vs")
        {
            WriteVariables(w, "vertex_inputs", io.Inputs, job);
        }
        else
        {
            w.WriteNull("vertex_inputs");
        }
    }

    private static void WriteVariables(Utf8JsonWriter w, string name, List<InterfaceVariable> list, Job job)
    {
        w.WriteStartArray(name);

        foreach (InterfaceVariable v in list)
        {
            w.WriteStartObject();
            w.WriteNumber("location", v.Location);
            w.WriteNumber("component", v.Component);
            w.WriteNumber("components", v.Components);
            w.WriteString("type", v.Type);

            if (v.ArrayLength != 0)
            {
                w.WriteNumber("array_length", v.ArrayLength);
            }

            // The runtime links a mesh stage by these two (runtime/gpu/shader_desc.c, shd_mesh_link): a mesh output's outer
            // array is over its vertices or primitives and takes one location, whatever its length.
            w.WriteNumber("array_depth", v.ArrayDepth);
            w.WriteBoolean("per_primitive", v.PerPrimitive);
            w.WriteBoolean("used", v.Used);

            if (job != null)
            {
                w.WriteString("answer", job.Attr.TryGet(v.Location, out AttributeType a) ? a.ToString().ToLowerInvariant() : "float");
            }

            w.WriteEndObject();
        }

        w.WriteEndArray();
    }

    // What the module's instructions do with the variable and what it declares
    // (NonWritable, NonReadable); all false when the post-pass did not find it.
    private static void WriteAccess(Utf8JsonWriter w, DescriptorAccess a)
    {
        w.WriteBoolean("module_reads", a?.Reads ?? false);
        w.WriteBoolean("module_writes", a?.Writes ?? false);
        w.WriteBoolean("non_writable", a?.NonWritable ?? false);
        w.WriteBoolean("non_readable", a?.NonReadable ?? false);
    }

    private static void WriteTextures(Utf8JsonWriter w, string name, IEnumerable<TextureDescriptor> list, bool image, FixupResult fix)
    {
        w.WriteStartArray(name);

        foreach (TextureDescriptor t in list)
        {
            (int tex, int smp, TextureHandleType handleType) = TextureHandle.UnpackOffsets(t.HandleIndex);
            (int slotLo, int slotHi) = t.CbufSlot >= 0 ? TextureHandle.UnpackSlots(t.CbufSlot, -1) : (-1, -1);
            bool sampler = (t.Type & SamplerType.Mask) == SamplerType.None;
            bool pushSlot = !image && t.Set == UniLayout.SlotSet;

            w.WriteStartObject();

            if (pushSlot)
            {
                // Read from the global arrays at the handle in a push constant slot (UniLayout)
                w.WriteNumber("set", UniLayout.GlobalSet);
                w.WriteNumber("binding", sampler ? UniLayout.SamplersBinding : UniLayout.ImagesBinding);
                w.WriteString("via", "push_slot");
                w.WriteNumber("push_slot", t.Binding);
                w.WriteNumber("push_offset", UniLayout.SlotsOffset + 4 * t.Binding);
            }
            else
            {
                w.WriteNumber("set", t.Set);
                w.WriteNumber("binding", t.Binding);
                w.WriteString("via", t.Set == UniLayout.GlobalSet ? "pool" : "descriptor");
                w.WriteNumber("push_slot", -1);
                w.WriteNumber("push_offset", -1);
            }

            w.WriteNumber("type_raw", (int)t.Type);
            w.WriteString("type", TypeNames.Describe(t.Type));
            w.WriteBoolean("shadow", (t.Type & SamplerType.Shadow) != 0);
            w.WriteString("kind", sampler ? "sampler" : t.Separate ? "separate_texture" : "combined");
            w.WriteNumber("array_length", t.ArrayLength);
            w.WriteBoolean("separate", t.Separate);
            w.WriteNumber("cbuf_slot_raw", t.CbufSlot);
            w.WriteNumber("cbuf_slot_lo", slotLo);
            w.WriteNumber("cbuf_slot_hi", slotHi);
            w.WriteNumber("handle_raw", t.HandleIndex);
            w.WriteNumber("tex_word_offset", tex);
            w.WriteNumber("sampler_word_offset", smp);
            w.WriteString("handle_type", handleType.ToString());
            w.WriteNumber("c2_slot", sampler ? HandleSlots.SamplerSlot(t.HandleIndex) : HandleSlots.Slot(t.HandleIndex, t.CbufSlot, image));
            w.WriteString("format", t.Format.ToString());
            w.WriteNumber("flags_raw", (int)t.Flags);
            w.WriteBoolean("image_store", (t.Flags & TextureUsageFlags.ImageStore) != 0);

            if (image)
            {
                WriteAccess(w, Find(fix, t.Set, t.Binding, image: true));
            }

            w.WriteEndObject();
        }

        w.WriteEndArray();
    }
}
