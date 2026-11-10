using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation;
using System;
using System.Collections.Generic;
using static Spv.Specification;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    using SpvInstruction = Spv.Generator.Instruction;
    using SpvLiteralInteger = Spv.Generator.LiteralInteger;

    // oot2-native: Ampere HMMA.16816 with an f16 accumulator (D = A x B + C; A 16 x 16, B 16 x 8, C and D 16 x 8, all halves) on
    // VK_KHR_cooperative_matrix, the body of the function dmsh_hmma16816_coop (Translation/HelperFunctionManager.cs, Transforms/TensorPass.cs).
    //
    // The fragments stay in the lanes: no workgroup memory, no barrier. A lane's registers hold the console's fragments (PTX mma.m16n8k16:
    // lane L, g = L / 4, t = L % 4; A word r: row g + 8 (r & 1), k 2t + 8 (r >> 1) and + 1; B word r: k 2t + 8 r and + 1 of column g; C / D
    // word r: row g + 8 r, columns 2t and 2t + 1; the low half of a word first), and a cooperative matrix's components are held by the lanes
    // in an order the extension leaves to the implementation. On NVIDIA the lanes hold the same positions as the console's fragments, in an
    // order of their own (measured at start-up by the runtime's probe, runtime/gpu/vk_hmma.c, which compares this order with the device's
    // and, on small integers whose sums are exact, the native result with the helper's and the exact sums before any module runs: the
    // wiring and the order, not the rounding, which differs between the forms on general inputs). Each console element e moves into component
    // Order[e] with OpCompositeInsert, one OpCooperativeMatrixMulAddKHR multiplies, and D's components move back out with
    // OpCompositeExtract in the accumulator's order. The workgroup-memory form (stores, OpCooperativeMatrixLoadKHR, a barrier) would need
    // no order but keeps the helper's stores, loads and both synchronisations, the cost that made the helper-body swap gain 1.5 times
    // where operands held in matrices gained 17 (the HMMA cost study).
    //
    // A module that holds this function declares Float16, CooperativeMatrixKHR and the Vulkan memory model, which SPV_KHR_cooperative_matrix
    // requires of a Shader module. The memory model changes no operation of the program but this: its workgroup memory, which GLSL450
    // treats as coherent within the workgroup, gets NonPrivatePointer and MakePointerAvailable or MakePointerVisible at workgroup scope on
    // every load and store (what glslang writes for a `shared` variable under the same model); storage buffers, physical storage buffers
    // and images stay private, as GLSL450 left them non-coherent. A module with a Coherent image, which the Vulkan memory model bans as a
    // decoration, keeps the helper: the function becomes a call of it, and nothing of the native form is declared.
    static class CoopHmma
    {
        // The component of the device's cooperative matrix that holds each console element of a lane's fragment: element e is the low (e even)
        // or high half of register e / 2. A: rows g, g + 8 at k 2t, 2t + 1, then at k 2t + 8, 2t + 9 (the console's), against the device's row
        // g at all four k first; B, C and D in the console's order. The runtime's probe measured it on the reference RTX 3060 Ti, and two
        // public recompilations report the same on an RTX 4070 and an RTX 5090 (A's registers in the order 0, 2, 1, 3); the runtime keeps a
        // copy of these three lines (runtime/include/oot2/vk_hmma.h) and its probe measures the device's before a module with this function is built
        public static readonly int[] OrderA = [0, 1, 4, 5, 2, 3, 6, 7];
        public static readonly int[] OrderB = [0, 1, 2, 3];
        public static readonly int[] OrderC = [0, 1, 2, 3];

        // SPV_KHR_cooperative_matrix's uses (MatrixAKHR, MatrixBKHR, MatrixAccumulatorKHR)
        private const int UseA = 0, UseB = 1, UseAccumulator = 2;

        private const Capability VulkanMemoryModel = Capability.VulkanMemoryModel;
        private const Capability VulkanMemoryModelDeviceScope = Capability.VulkanMemoryModelDeviceScope;

        /// <summary>The index of the program's function named name, or -1.</summary>
        public static int FindFunction(StructuredProgramInfo info, string name)
        {
            for (int i = 0; i < info.Functions.Count; i++)
            {
                if (info.Functions[i].Name == name)
                {
                    return i;
                }
            }

            return -1;
        }

        /// <summary>Whether the module gets the native form: it calls the native function, holds the helper, and declares no Coherent image.</summary>
        public static bool IsNative(CodeGenContext context, StructuredProgramInfo info)
        {
            if (FindFunction(info, HelperFunctionManager.HmmaCoopName) < 0 || FindFunction(info, HelperFunctionManager.HmmaHelperName) < 0)
            {
                return false;
            }

            foreach (TextureDefinition image in context.Properties.Images.Values)
            {
                if (image.Flags.HasFlag(TextureUsageFlags.ImageCoherent))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>The capabilities and extensions of the native form; the caller sets the Vulkan memory model.</summary>
        public static void Declare(CodeGenContext context)
        {
            context.AddCapability(Capability.Float16);
            context.AddCapability(Spv.Generator.Module.CooperativeMatrixKHR);
            context.AddCapability(VulkanMemoryModel);
            context.AddCapability(VulkanMemoryModelDeviceScope);   // the atomics and memory barriers at device scope keep it
            context.AddExtension("SPV_KHR_cooperative_matrix");
            context.AddExtension("SPV_KHR_vulkan_memory_model");
        }

        /// <summary>The body of dmsh_hmma16816_coop, after its parameters (all pointers to 32-bit words, as every function of the
        /// translator's): arguments 2..5 A, 6..7 B, 8..9 C; D's first word returned, its second stored through argument 10. The lane and
        /// the block offset (arguments 0 and 1) are the helper's and unused here.</summary>
        public static void EmitBody(CodeGenContext context)
        {
            context.AddLabel(context.Label());

            SpvInstruction s32 = context.TypeS32();
            SpvInstruction u32 = context.TypeU32();
            SpvInstruction f16 = context.TypeFloat(16);
            SpvInstruction h2 = context.TypeVector(f16, 2);

            // the constants first: a type's operands must be declared before it, and the module orders declarations by id
            SpvInstruction scope = context.Constant(u32, (int)Scope.Subgroup);
            SpvInstruction n16 = context.Constant(u32, 16);
            SpvInstruction n8 = context.Constant(u32, 8);
            SpvInstruction useA = context.Constant(u32, UseA);
            SpvInstruction useB = context.Constant(u32, UseB);
            SpvInstruction useC = context.Constant(u32, UseAccumulator);
            SpvInstruction zero = context.Constant(f16, 0);

            SpvInstruction typeA = context.TypeCooperativeMatrixKHR(f16, scope, n16, n16, useA);
            SpvInstruction typeB = context.TypeCooperativeMatrixKHR(f16, scope, n16, n8, useB);
            SpvInstruction typeC = context.TypeCooperativeMatrixKHR(f16, scope, n16, n8, useC);

            SpvInstruction Fill(SpvInstruction type, int firstArgument, int words, int[] order)
            {
                SpvInstruction matrix = context.ConstantComposite(type, zero);

                for (int w = 0; w < words; w++)
                {
                    SpvInstruction word = context.Load(s32, context.GetArgumentPointer(firstArgument + w));
                    SpvInstruction pair = context.Bitcast(h2, word);

                    for (int h = 0; h < 2; h++)
                    {
                        SpvInstruction half = context.CompositeExtract(f16, pair, (SpvLiteralInteger)h);
                        matrix = context.CompositeInsert(type, half, matrix, (SpvLiteralInteger)order[2 * w + h]);
                    }
                }

                return matrix;
            }

            SpvInstruction a = Fill(typeA, 2, 4, OrderA);
            SpvInstruction b = Fill(typeB, 6, 2, OrderB);
            SpvInstruction c = Fill(typeC, 8, 2, OrderC);
            SpvInstruction d = context.CooperativeMatrixMulAddKHR(typeC, a, b, c);

            SpvInstruction Word(int first)
            {
                SpvInstruction lo = context.CompositeExtract(f16, d, (SpvLiteralInteger)OrderC[first]);
                SpvInstruction hi = context.CompositeExtract(f16, d, (SpvLiteralInteger)OrderC[first + 1]);

                return context.Bitcast(s32, context.CompositeConstruct(h2, lo, hi));
            }

            SpvInstruction d0 = Word(0);
            SpvInstruction d1 = Word(2);

            context.Store(context.GetArgumentPointer(10), d1);
            context.ReturnValue(d0);
        }

        /// <summary>The body of dmsh_hmma16816_coop in a module that keeps the helper: a call of the helper with the same arguments, through
        /// locals of this function (a call's pointer arguments are memory object declarations).</summary>
        public static void EmitHelperCall(CodeGenContext context, SpvInstruction helper)
        {
            context.AddLabel(context.Label());

            SpvInstruction s32 = context.TypeS32();
            SpvInstruction pointer = context.TypePointer(StorageClass.Function, s32);
            SpvInstruction[] locals = new SpvInstruction[11];

            for (int i = 0; i < locals.Length; i++)
            {
                locals[i] = context.Variable(pointer, StorageClass.Function);
                context.AddLocalVariable(locals[i]);
            }

            for (int i = 0; i < 10; i++)
            {
                context.Store(locals[i], context.Load(s32, context.GetArgumentPointer(i)));
            }

            SpvInstruction d0 = context.FunctionCall(s32, helper, locals);

            context.Store(context.GetArgumentPointer(10), context.Load(s32, locals[10]));
            context.ReturnValue(d0);
        }

        // SPIR-V opcodes, storage classes and decorations the conversion reads
        private const int OpMemoryModel = 14, OpFunction = 54, OpVariable = 59, OpLoad = 61, OpStore = 62, OpCopyMemory = 63,
            OpCopyMemorySized = 64, OpAccessChain = 65, OpInBoundsAccessChain = 66, OpPtrAccessChain = 67, OpDecorate = 71,
            OpMemberDecorate = 72, OpCopyObject = 83, OpBitcast = 124, OpTypePointer = 32, OpInBoundsPtrAccessChain = 70;
        private const uint StorageWorkgroup = 4, DecorationVolatile = 21, DecorationCoherent = 23;
        private const uint AccessAligned = 0x2, AccessMakeAvailable = 0x8, AccessMakeVisible = 0x10, AccessNonPrivate = 0x20;

        /// <summary>
        /// The module under the Vulkan memory model's rules for what it does under GLSL450: every load and store through a pointer to workgroup
        /// memory gets NonPrivatePointer and, at the scope whose constant id is workgroupScope, MakePointerVisible (a load) or
        /// MakePointerAvailable (a store). The memory model instruction itself is the generator's. A Coherent or Volatile decoration, or a
        /// memory copy, which the translator never writes and the conversion does not cover, is an error by name.
        /// </summary>
        public static byte[] ToVulkanMemoryModel(byte[] module, uint workgroupScope)
        {
            uint[] w = new uint[module.Length / 4];
            Buffer.BlockCopy(module, 0, w, 0, w.Length * 4);

            HashSet<uint> pointerTypes = [];
            HashSet<uint> pointers = [];
            List<uint> output = new(w.Length + w.Length / 8);

            for (int i = 0; i < 5; i++)
            {
                output.Add(w[i]);
            }

            for (int i = 5; i < w.Length;)
            {
                int count = (int)(w[i] >> 16);
                int op = (int)(w[i] & 0xffff);

                if (count == 0 || i + count > w.Length)
                {
                    throw new InvalidOperationException($"the module's instruction at word {i} runs past its end");
                }

                switch (op)
                {
                    case OpTypePointer when w[i + 2] == StorageWorkgroup:
                        pointerTypes.Add(w[i + 1]);
                        break;
                    case OpVariable or OpAccessChain or OpInBoundsAccessChain or OpPtrAccessChain or OpInBoundsPtrAccessChain or OpCopyObject or OpBitcast
                        when pointerTypes.Contains(w[i + 1]):
                        pointers.Add(w[i + 2]);
                        break;
                    case OpDecorate when count >= 3 && (w[i + 2] == DecorationCoherent || w[i + 2] == DecorationVolatile):
                    case OpMemberDecorate when count >= 4 && (w[i + 3] == DecorationCoherent || w[i + 3] == DecorationVolatile):
                        throw new InvalidOperationException("a Coherent or Volatile decoration under the Vulkan memory model");
                    case OpCopyMemory or OpCopyMemorySized:
                        throw new InvalidOperationException("a memory copy under the Vulkan memory model");
                    case OpLoad when pointers.Contains(w[i + 3]):
                        Rewrite(output, w, i, 4, AccessMakeVisible, workgroupScope);
                        i += count;
                        continue;
                    case OpStore when pointers.Contains(w[i + 1]):
                        Rewrite(output, w, i, 3, AccessMakeAvailable, workgroupScope);
                        i += count;
                        continue;
                }

                for (int k = 0; k < count; k++)
                {
                    output.Add(w[i + k]);
                }

                i += count;
            }

            byte[] result = new byte[output.Count * 4];
            Buffer.BlockCopy(output.ToArray(), 0, result, 0, result.Length);

            return result;
        }

        // A load or store whose memory operands start at word `fixed` of the instruction, with NonPrivatePointer and `make` at `scope` added.
        // Operands follow the mask in the order of their bits: Aligned's literal, then MakePointerAvailable's scope, then MakePointerVisible's.
        private static void Rewrite(List<uint> output, uint[] w, int at, int fixedWords, uint make, uint scope)
        {
            int count = (int)(w[at] >> 16);
            uint mask = count > fixedWords ? w[at + fixedWords] : 0;

            if ((mask & ~AccessAligned) != 0)
            {
                throw new InvalidOperationException($"a workgroup access with memory operands {mask:x} under the Vulkan memory model");
            }

            List<uint> operands = [mask | make | AccessNonPrivate];

            if ((mask & AccessAligned) != 0)
            {
                operands.Add(w[at + fixedWords + 1]);
            }

            operands.Add(scope);

            output.Add((uint)((fixedWords + operands.Count) << 16) | (w[at] & 0xffff));

            for (int k = 1; k < fixedWords; k++)
            {
                output.Add(w[at + k]);
            }

            output.AddRange(operands);
        }
    }
}
