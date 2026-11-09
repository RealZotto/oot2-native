// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using Ryujinx.Graphics.Shader.Translation;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace Oot2.Xlate;

/// <summary>Translator-only time of one job, summed over its passes, in Stopwatch ticks.</summary>
internal sealed class TranslatorTiming
{
    public long DecodeTicks;
    public long TranslateTicks;

    public long TotalTicks => DecodeTicks + TranslateTicks;

    public long Micros => ToMicros(TotalTicks);

    public static long ToMicros(long ticks) => ticks * 1_000_000 / Stopwatch.Frequency;
}

/// <summary>What the translator produced for a job, before the post-pass.</summary>
internal sealed class TranslationOutcome
{
    public byte[] Spirv { get; init; }
    public ShaderProgramInfo Info { get; init; }
    public Accessor Accessor { get; init; }
    public int Passes { get; init; }
}

/// <summary>
/// Drives the vendored translator for one job: the flags per stage, the two-pass
/// recovery of texture types, the timing bracket and the error classification.
/// </summary>
internal static class Translation
{
    /// <summary>The environment variable that names the folder of the IR dump.</summary>
    public const string IrDumpVariable = "OOT2_XLATE_IR_DUMP";

    public static TranslationOutcome Run(Job job, byte[] code, TranslatorTiming timing)
    {
        TranslationFlags flags = TranslationFlags.Ampere;

        if (job.IsCompute)
        {
            flags |= TranslationFlags.Compute;
        }

        // DebugMode turns the AST optimizer off: nothing is propagated or
        // dropped, which is the form every module had until the sound form
        // became the default. Without it the optimizer follows this project's
        // rule (AstOptimizer.cs): an operation with an effect stays where it
        // is, and a read or a shuffle is carried only to the next statement
        // that uses it. The job option `debug` defaults to false; true asks
        // for the old form (A/B work). The default is part of the executable,
        // so changing it changes the translator identity and with it every
        // key built on that identity.
        if (job.Options.Debug)
        {
            flags |= TranslationFlags.DebugMode;
        }

        TranslationOptions options = new(TargetLanguage.Spirv, TargetApi.Vulkan, flags);

        Dictionary<int, SamplerType> imageTypes = null;
        Dictionary<int, SamplerType> textureTypes = null;
        Accessor accessor = null;
        ShaderProgram program = null;
        int passes = 0;

        // The second pass runs only when the first one guessed a texture type and
        // the shader's own instructions say better. A fresh context is needed
        // because translation mutates the decoded program.
        while (passes < 2)
        {
            passes++;
            accessor = new Accessor(job, code, imageTypes, textureTypes);

            // A text dump of the translator's IR, one folder per job key in the folder the
            // variable names (see IrDump). It is for diagnosis; unset, nothing is written.
            string dumpFolder = Environment.GetEnvironmentVariable(IrDumpVariable);

            if (!string.IsNullOrEmpty(dumpFolder))
            {
                IrDump.Begin(Path.Combine(dumpFolder, job.Key), passes);
            }

            try
            {
                program = TranslateOnce(job, accessor, options, timing);
            }
            finally
            {
                IrDump.End();
            }

            if (passes == 2)
            {
                break;
            }

            imageTypes = accessor.ImageQueryGuessed ? Accessor.ImageTypesOf(program) : null;

            if (imageTypes is { Count: 0 })
            {
                imageTypes = null;
            }

            textureTypes = accessor.TextureTypesOf(program);

            if (textureTypes.Count == 0)
            {
                textureTypes = null;
            }

            if (imageTypes == null && textureTypes == null)
            {
                break;
            }
        }

        if (passes == 2)
        {
            accessor.AddNote($"second pass: types recovered for {imageTypes?.Count ?? 0} image handle(s) and {textureTypes?.Count ?? 0} texture slot(s)");
        }

        return new TranslationOutcome
        {
            Spirv = program.BinaryCode,
            Info = program.Info,
            Accessor = accessor,
            Passes = passes,
        };
    }

    private static ShaderProgram TranslateOnce(Job job, Accessor accessor, TranslationOptions options, TranslatorTiming timing)
    {
        TranslatorContext context;
        long t0 = Stopwatch.GetTimestamp();

        try
        {
            context = Translator.CreateContext(0, accessor, options);
        }
        catch (Exception e) when (e is not JobException)
        {
            throw Classify(e, Kinds.Decode);
        }
        finally
        {
            timing.DecodeTicks += Stopwatch.GetTimestamp() - t0;
        }

        CheckStage(job, context.Stage);

        long t1 = Stopwatch.GetTimestamp();

        try
        {
            ShaderProgram program = context.Translate();

            if (program.BinaryCode == null || program.BinaryCode.Length == 0)
            {
                throw new JobException(Kinds.Translate, "the translator returned no SPIR-V");
            }

            return program;
        }
        catch (Exception e) when (e is not JobException)
        {
            throw Classify(e, Kinds.Translate);
        }
        finally
        {
            timing.TranslateTicks += Stopwatch.GetTimestamp() - t1;
        }
    }

    // The real stage of a graphics program is the one in its header, not the job's.
    private static void CheckStage(Job job, ShaderStage actual)
    {
        ShaderStage expected = job.Stage switch
        {
            "vs" => ShaderStage.Vertex,
            "ps" => ShaderStage.Fragment,
            _ => ShaderStage.Compute,
        };

        if (actual == expected)
        {
            return;
        }

        if (actual is ShaderStage.Geometry or ShaderStage.TessellationControl or ShaderStage.TessellationEvaluation)
        {
            throw new JobException(Kinds.UnsupportedStage, $"the program header says {actual}, which this host does not translate");
        }

        throw new JobException(Kinds.BadJob, $"job stage {job.Stage} but the program header says {actual}");
    }

    /// <summary>
    /// A translator exception as a job failure. "Ampere ..." NotSupportedExceptions
    /// are the frontend's own refusals (an opcode or a form it does not emit) and
    /// keep their message, which names the opcode; everything else is a decode or
    /// translate failure by the phase it came from.
    /// </summary>
    internal static JobException Classify(Exception e, string phase)
    {
        if (e is NotSupportedException && e.Message.StartsWith("Ampere ", StringComparison.Ordinal))
        {
            return new JobException(Kinds.UnsupportedOpcode, e.Message);
        }

        return new JobException(phase, $"{e.GetType().Name}: {e.Message}");
    }
}
