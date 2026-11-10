using Ryujinx.Common;
using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using Ryujinx.Graphics.Shader.Translation;
using System;
using System.Collections.Generic;
using System.Threading;
using static Spv.Specification;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    using SpvInstruction = Spv.Generator.Instruction;
    using SpvInstructionPool = Spv.Generator.GeneratorPool<Spv.Generator.Instruction>;
    using SpvLiteralInteger = Spv.Generator.LiteralInteger;
    using SpvLiteralIntegerPool = Spv.Generator.GeneratorPool<Spv.Generator.LiteralInteger>;

    static class SpirvGenerator
    {
        // Resource pools for Spirv generation. Note: Increase count when more threads are being used.
        private const int GeneratorPoolCount = 1;
        private static readonly ObjectPool<SpvInstructionPool> _instructionPool;
        private static readonly ObjectPool<SpvLiteralIntegerPool> _integerPool;
        private static readonly Lock _poolLock = new();

        static SpirvGenerator()
        {
            _instructionPool = new(() => new SpvInstructionPool(), GeneratorPoolCount);
            _integerPool = new(() => new SpvLiteralIntegerPool(), GeneratorPoolCount);
        }

        private const HelperFunctionsMask NeedsInvocationIdMask = HelperFunctionsMask.SwizzleAdd;

        // oot2-native: the call sites from which a raw-access helper is kept a call rather than inlined (Generate)
        private const int RawHelperCallMinSites = 64;

        // oot2-native: how many times each function is called, over every function's body; a call that returns a value sits in an assignment's source
        private static int[] CountCallSites(StructuredProgramInfo info)
        {
            int[] calls = new int[info.Functions.Count];

            void Count(IAstNode node)
            {
                if (node is AstAssignment assignment)
                {
                    Count(assignment.Source);
                }
                else if (node is AstOperation operation)
                {
                    if (operation.Inst == Instruction.Call && operation.GetSource(0) is AstOperand { Type: OperandType.Constant } id)
                    {
                        calls[id.Value]++;
                    }

                    for (int index = 0; index < operation.SourcesCount; index++)
                    {
                        Count(operation.GetSource(index));
                    }
                }
            }

            foreach (StructuredFunction function in info.Functions)
            {
                foreach (IAstNode node in new AstBlockVisitor(function.MainBlock).Visit())
                {
                    Count(node);
                }
            }

            return calls;
        }

        public static byte[] Generate(StructuredProgramInfo info, CodeGenParameters parameters)
        {
            SpvInstructionPool instPool;
            SpvLiteralIntegerPool integerPool;

            lock (_poolLock)
            {
                instPool = _instructionPool.Allocate();
                integerPool = _integerPool.Allocate();
            }

            CodeGenContext context = new(info, parameters, instPool, integerPool);

            context.AddCapability(Capability.Shader);

            // oot2-native: HMMA on VK_KHR_cooperative_matrix (CoopHmma) takes the Vulkan memory model, which that extension requires of a Shader module
            bool nativeHmma = CoopHmma.IsNative(context, info);
            MemoryModel memoryModel = nativeHmma ? MemoryModel.Vulkan : MemoryModel.GLSL450;

            // nxnative: raw global memory goes through physical storage buffer pointers (ShaderProperties.UsesPhysicalPointers), which take 64-bit physical addressing
            // oot2-native: the draw's first task is read by device address too (MeshAbi)
            if (context.Properties.UsesPhysicalPointers || context.Properties.UsesFirstTask)
            {
                context.AddCapability(Capability.PhysicalStorageBufferAddresses);
                context.AddCapability(Capability.Int64);
                context.AddExtension("SPV_KHR_physical_storage_buffer");
                context.SetMemoryModel(AddressingModel.PhysicalStorageBuffer64, memoryModel);
            }
            else
            {
                context.SetMemoryModel(AddressingModel.Logical, memoryModel);
            }

            uint workgroupScope = 0;

            if (nativeHmma)
            {
                CoopHmma.Declare(context);
                workgroupScope = context.Constant(context.TypeU32(), (int)Scope.Workgroup).Id;
            }

            context.AddCapability(Capability.GroupNonUniformBallot);
            context.AddCapability(Capability.GroupNonUniformShuffle);
            context.AddCapability(Capability.GroupNonUniformVote);
            context.AddCapability(Capability.ImageBuffer);
            context.AddCapability(Capability.ImageGatherExtended);
            context.AddCapability(Capability.ImageQuery);
            context.AddCapability(Capability.SampledBuffer);

            if (parameters.HostCapabilities.SupportsShaderFloat64)
            {
                context.AddCapability(Capability.Float64);
            }

            if (parameters.Definitions.TransformFeedbackEnabled && parameters.Definitions.LastInVertexPipeline)
            {
                context.AddCapability(Capability.TransformFeedback);
            }

            if (parameters.Definitions.Stage == ShaderStage.Fragment)
            {
                if (context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Input, IoVariable.Layer)) ||
                    context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Input, IoVariable.PrimitiveId)))
                {
                    context.AddCapability(Capability.Geometry);
                }

                // nxnative: the sample index (PIXLD.MYINDEX) is a SampleRateShading input
                if (context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Input, IoVariable.SampleId)))
                {
                    context.AddCapability(Capability.SampleRateShading);
                }

                if (context.HostCapabilities.SupportsFragmentShaderInterlock)
                {
                    context.AddCapability(Capability.FragmentShaderPixelInterlockEXT);
                    context.AddExtension("SPV_EXT_fragment_shader_interlock");
                }
            }
            else if (parameters.Definitions.Stage == ShaderStage.Geometry)
            {
                context.AddCapability(Capability.Geometry);

                if (parameters.Definitions.GpPassthrough && context.HostCapabilities.SupportsGeometryShaderPassthrough)
                {
                    context.AddExtension("SPV_NV_geometry_shader_passthrough");
                    context.AddCapability(Capability.GeometryShaderPassthroughNV);
                }
            }
            else if (parameters.Definitions.Stage is ShaderStage.TessellationControl or
                     ShaderStage.TessellationEvaluation)
            {
                context.AddCapability(Capability.Tessellation);
            }
            else if (parameters.Definitions.Stage == ShaderStage.Vertex)
            {
                context.AddCapability(Capability.DrawParameters);
            }

            if (context.Definitions.Stage != ShaderStage.Fragment &&
                context.Definitions.Stage != ShaderStage.Geometry &&
                context.Definitions.Stage != ShaderStage.Compute &&
                (context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Output, IoVariable.Layer)) ||
                context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Output, IoVariable.ViewportIndex))))
            {
                context.AddExtension("SPV_EXT_shader_viewport_index_layer");
                context.AddCapability(Capability.ShaderViewportIndexLayerEXT);
            }

            if (context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Output, IoVariable.ViewportMask)))
            {
                context.AddExtension("SPV_NV_viewport_array2");
                context.AddCapability(Capability.ShaderViewportMaskNV);
            }

            if ((info.HelperFunctionsMask & NeedsInvocationIdMask) != 0)
            {
                info.IoDefinitions.Add(new IoDefinition(StorageKind.Input, IoVariable.SubgroupLaneId));
            }

            // oot2-native: a task or mesh stage (SPV_EXT_mesh_shader). It stays a Compute stage until here, where it takes its own execution model (MeshEmission)
            if (context.Definitions.MeshStage != null)
            {
                context.AddCapability(Capability.MeshShadingEXT);
                context.AddExtension("SPV_EXT_mesh_shader");

                if (context.Definitions.MeshStage.Kind == MeshStageKind.Mesh)
                {
                    // the epilogue copies the lines out by invocation
                    info.IoDefinitions.Add(new IoDefinition(StorageKind.Input, IoVariable.ThreadId));
                }
            }

            Declarations.DeclareAll(context, info);

            int[] callSites = CountCallSites(info);

            for (int funcIndex = 0; funcIndex < info.Functions.Count; funcIndex++)
            {
                StructuredFunction function = info.Functions[funcIndex];
                SpvInstruction retType = context.GetType(function.ReturnType);

                SpvInstruction[] funcArgs = new SpvInstruction[function.InArguments.Length + function.OutArguments.Length];

                for (int argIndex = 0; argIndex < funcArgs.Length; argIndex++)
                {
                    SpvInstruction argType = context.GetType(function.GetArgumentType(argIndex));
                    SpvInstruction argPointerType = context.TypePointer(StorageClass.Function, argType);
                    funcArgs[argIndex] = argPointerType;
                }

                SpvInstruction funcType = context.TypeFunction(retType, false, funcArgs);

                // oot2-native: a raw-access helper called from many sites stays a call. Most raw-access programs call it 253 times, and the driver
                // (617.14) inlines every call unless told not to: inlined there, it made a module about twice as slow to build as with one copy of it
                // (and 14 times as slow with nxnative's pool search). With few sites the call costs more than it saves (up to 1.5 times the build
                // time under 52 sites): the measured crossover lies between 51 and 53 sites (the raw-access compile fix)
                FunctionControlMask control =
                    function.Name.EndsWith(Translation.Optimizations.GlobalToStorage.RawHelperSuffix, StringComparison.Ordinal) && callSites[funcIndex] >= RawHelperCallMinSites
                    ? FunctionControlMask.DontInline
                    : FunctionControlMask.MaskNone;
                SpvInstruction spvFunc = context.Function(retType, control, funcType);

                // nxnative: helpers whose body the shader build replaces in the Metal source keep their name (dmsh_hmma16816_f16)
                if (function.Name.StartsWith("dmsh_"))
                {
                    context.Name(spvFunc, function.Name);
                }

                context.DeclareFunction(funcIndex, function, spvFunc);
            }

            for (int funcIndex = 0; funcIndex < info.Functions.Count; funcIndex++)
            {
                if (info.Functions[funcIndex].Name == HelperFunctionManager.HmmaCoopName)
                {
                    GenerateCoopHmma(context, info, funcIndex, nativeHmma);
                }
                else
                {
                    Generate(context, info, funcIndex);
                }
            }

            if (context.Definitions.MeshStage != null)
            {
                // oot2-native: declared when every function exists, so that its interface can list every global variable (SPIR-V 1.4)
                MeshEmission.AddEntryPoint(context, context.GetFunction(0).Item2);
            }

            byte[] result = context.Generate();

            lock (_poolLock)
            {
                _instructionPool.Release(instPool);
                _integerPool.Release(integerPool);
            }

            // oot2-native: the workgroup accesses under the Vulkan memory model's rules (CoopHmma.ToVulkanMemoryModel)
            return nativeHmma ? CoopHmma.ToVulkanMemoryModel(result, workgroupScope) : result;
        }

        // oot2-native: the HMMA function on VK_KHR_cooperative_matrix, whose body the IR does not hold (HelperFunctionManager.GenerateHmma16816F16Coop):
        // the native body, or in a module that keeps the helper (CoopHmma.IsNative) a call of the helper
        private static void GenerateCoopHmma(CodeGenContext context, StructuredProgramInfo info, int funcIndex, bool native)
        {
            (StructuredFunction function, SpvInstruction spvFunc) = context.GetFunction(funcIndex);

            context.CurrentFunction = function;
            context.AddFunction(spvFunc);
            context.StartFunction(isMainFunction: false);

            Declarations.DeclareParameters(context, function);

            if (native)
            {
                CoopHmma.EmitBody(context);
            }
            else
            {
                CoopHmma.EmitHelperCall(context, context.GetFunction(CoopHmma.FindFunction(info, HelperFunctionManager.HmmaHelperName)).Item2);
            }

            context.FunctionEnd();
        }

        private static void Generate(CodeGenContext context, StructuredProgramInfo info, int funcIndex)
        {
            (StructuredFunction function, SpvInstruction spvFunc) = context.GetFunction(funcIndex);

            context.CurrentFunction = function;
            context.AddFunction(spvFunc);
            context.StartFunction(isMainFunction: funcIndex == 0);

            Declarations.DeclareParameters(context, function);

            context.EnterBlock(function.MainBlock);

            Declarations.DeclareLocals(context, function);

            bool meshMain = funcIndex == 0 && context.Definitions.MeshStage != null;

            if (meshMain)
            {
                MeshEmission.Prologue(context);
            }

            Generate(context, function.MainBlock);

            // Functions must always end with a return.
            if (function.MainBlock.Last is not AstOperation operation ||
                (operation.Inst != Instruction.Return && operation.Inst != Instruction.Discard))
            {
                if (meshMain)
                {
                    MeshEmission.Epilogue(context);
                }
                else
                {
                    context.Return();
                }
            }

            context.FunctionEnd();

            if (funcIndex == 0 && context.Definitions.MeshStage == null)
            {
                context.AddEntryPoint(context.Definitions.Stage.Convert(), spvFunc, "main", context.GetMainInterface());

                if (context.Definitions.Stage == ShaderStage.TessellationControl)
                {
                    context.AddExecutionMode(spvFunc, ExecutionMode.OutputVertices, (SpvLiteralInteger)context.Definitions.ThreadsPerInputPrimitive);
                }
                else if (context.Definitions.Stage == ShaderStage.TessellationEvaluation)
                {
                    switch (context.Definitions.TessPatchType)
                    {
                        case TessPatchType.Isolines:
                            context.AddExecutionMode(spvFunc, ExecutionMode.Isolines);
                            break;
                        case TessPatchType.Triangles:
                            context.AddExecutionMode(spvFunc, ExecutionMode.Triangles);
                            break;
                        case TessPatchType.Quads:
                            context.AddExecutionMode(spvFunc, ExecutionMode.Quads);
                            break;
                    }

                    switch (context.Definitions.TessSpacing)
                    {
                        case TessSpacing.EqualSpacing:
                            context.AddExecutionMode(spvFunc, ExecutionMode.SpacingEqual);
                            break;
                        case TessSpacing.FractionalEventSpacing:
                            context.AddExecutionMode(spvFunc, ExecutionMode.SpacingFractionalEven);
                            break;
                        case TessSpacing.FractionalOddSpacing:
                            context.AddExecutionMode(spvFunc, ExecutionMode.SpacingFractionalOdd);
                            break;
                    }

                    bool tessCw = context.Definitions.TessCw;

                    if (context.TargetApi == TargetApi.Vulkan)
                    {
                        // We invert the front face on Vulkan backend, so we need to do that here as well.
                        tessCw = !tessCw;
                    }

                    if (tessCw)
                    {
                        context.AddExecutionMode(spvFunc, ExecutionMode.VertexOrderCw);
                    }
                    else
                    {
                        context.AddExecutionMode(spvFunc, ExecutionMode.VertexOrderCcw);
                    }
                }
                else if (context.Definitions.Stage == ShaderStage.Geometry)
                {
                    context.AddExecutionMode(spvFunc, context.Definitions.InputTopology switch
                    {
                        InputTopology.Points => ExecutionMode.InputPoints,
                        InputTopology.Lines => ExecutionMode.InputLines,
                        InputTopology.LinesAdjacency => ExecutionMode.InputLinesAdjacency,
                        InputTopology.Triangles => ExecutionMode.Triangles,
                        InputTopology.TrianglesAdjacency => ExecutionMode.InputTrianglesAdjacency,
                        _ => throw new InvalidOperationException($"Invalid input topology \"{context.Definitions.InputTopology}\"."),
                    });

                    context.AddExecutionMode(spvFunc, ExecutionMode.Invocations, (SpvLiteralInteger)context.Definitions.ThreadsPerInputPrimitive);

                    context.AddExecutionMode(spvFunc, context.Definitions.OutputTopology switch
                    {
                        OutputTopology.PointList => ExecutionMode.OutputPoints,
                        OutputTopology.LineStrip => ExecutionMode.OutputLineStrip,
                        OutputTopology.TriangleStrip => ExecutionMode.OutputTriangleStrip,
                        _ => throw new InvalidOperationException($"Invalid output topology \"{context.Definitions.OutputTopology}\"."),
                    });

                    context.AddExecutionMode(spvFunc, ExecutionMode.OutputVertices, (SpvLiteralInteger)context.Definitions.MaxOutputVertices);
                }
                else if (context.Definitions.Stage == ShaderStage.Fragment)
                {
                    context.AddExecutionMode(spvFunc, context.Definitions.OriginUpperLeft
                        ? ExecutionMode.OriginUpperLeft
                        : ExecutionMode.OriginLowerLeft);

                    if (context.Info.IoDefinitions.Contains(new IoDefinition(StorageKind.Output, IoVariable.FragmentOutputDepth)))
                    {
                        context.AddExecutionMode(spvFunc, ExecutionMode.DepthReplacing);
                    }

                    if (context.Definitions.EarlyZForce)
                    {
                        context.AddExecutionMode(spvFunc, ExecutionMode.EarlyFragmentTests);
                    }

                    if ((info.HelperFunctionsMask & HelperFunctionsMask.FSI) != 0 &&
                        context.HostCapabilities.SupportsFragmentShaderInterlock)
                    {
                        context.AddExecutionMode(spvFunc, ExecutionMode.PixelInterlockOrderedEXT);
                    }
                }
                else if (context.Definitions.Stage == ShaderStage.Compute)
                {
                    SpvLiteralInteger localSizeX = (SpvLiteralInteger)context.Definitions.ComputeLocalSizeX;
                    SpvLiteralInteger localSizeY = (SpvLiteralInteger)context.Definitions.ComputeLocalSizeY;
                    SpvLiteralInteger localSizeZ = (SpvLiteralInteger)context.Definitions.ComputeLocalSizeZ;

                    context.AddExecutionMode(
                        spvFunc,
                        ExecutionMode.LocalSize,
                        localSizeX,
                        localSizeY,
                        localSizeZ);
                }

                if (context.Definitions.TransformFeedbackEnabled && context.Definitions.LastInVertexPipeline)
                {
                    context.AddExecutionMode(spvFunc, ExecutionMode.Xfb);
                }
            }
        }

        private static void Generate(CodeGenContext context, AstBlock block)
        {
            AstBlockVisitor visitor = new(block);

            Dictionary<AstBlock, (SpvInstruction, SpvInstruction)> loopTargets = new();

            context.LoopTargets = loopTargets;

            visitor.BlockEntered += (sender, e) =>
            {
                AstBlock mergeBlock = e.Block.Parent;

                if (e.Block.Type == AstBlockType.If)
                {
                    AstBlock ifTrueBlock = e.Block;
                    AstBlock ifFalseBlock;

                    if (AstHelper.Next(e.Block) is AstBlock nextBlock && nextBlock.Type == AstBlockType.Else)
                    {
                        ifFalseBlock = nextBlock;
                    }
                    else
                    {
                        ifFalseBlock = mergeBlock;
                    }

                    SpvInstruction condition = context.Get(AggregateType.Bool, e.Block.Condition);

                    context.SelectionMerge(context.GetNextLabel(mergeBlock), SelectionControlMask.MaskNone);
                    context.BranchConditional(condition, context.GetNextLabel(ifTrueBlock), context.GetNextLabel(ifFalseBlock));
                }
                else if (e.Block.Type == AstBlockType.DoWhile)
                {
                    SpvInstruction continueTarget = context.Label();

                    loopTargets.Add(e.Block, (context.NewBlock(), continueTarget));

                    context.LoopMerge(context.GetNextLabel(mergeBlock), continueTarget, LoopControlMask.MaskNone);
                    context.Branch(context.GetFirstLabel(e.Block));
                }

                context.EnterBlock(e.Block);
            };

            visitor.BlockLeft += (sender, e) =>
            {
                if (e.Block.Parent != null)
                {
                    if (e.Block.Type == AstBlockType.DoWhile)
                    {
                        // This is a loop, we need to jump back to the loop header
                        // if the condition is true.
                        AstBlock mergeBlock = e.Block.Parent;

                        (SpvInstruction loopTarget, SpvInstruction continueTarget) = loopTargets[e.Block];

                        context.Branch(continueTarget);
                        context.AddLabel(continueTarget);

                        SpvInstruction condition = context.Get(AggregateType.Bool, e.Block.Condition);

                        context.BranchConditional(condition, loopTarget, context.GetNextLabel(mergeBlock));
                    }
                    else
                    {
                        // We only need a branch if the last instruction didn't
                        // already cause the program to exit or jump elsewhere.
                        bool lastIsCf = e.Block.Last is AstOperation lastOp &&
                            (lastOp.Inst == Instruction.Discard ||
                             lastOp.Inst == Instruction.LoopBreak ||
                             lastOp.Inst == Instruction.LoopContinue ||
                             lastOp.Inst == Instruction.Return);

                        if (!lastIsCf)
                        {
                            context.Branch(context.GetNextLabel(e.Block.Parent));
                        }
                    }

                    bool hasElse = AstHelper.Next(e.Block) is AstBlock nextBlock &&
                        (nextBlock.Type == AstBlockType.Else ||
                         nextBlock.Type == AstBlockType.ElseIf);

                    // Re-enter the parent block.
                    if (e.Block.Parent != null && !hasElse)
                    {
                        context.EnterBlock(e.Block.Parent);
                    }
                }
            };

            foreach (IAstNode node in visitor.Visit())
            {
                if (node is AstAssignment assignment)
                {
                    AstOperand dest = (AstOperand)assignment.Destination;

                    if (dest.Type == OperandType.LocalVariable)
                    {
                        SpvInstruction source = context.Get(dest.VarType, assignment.Source);
                        context.Store(context.GetLocalPointer(dest), source);
                    }
                    else if (dest.Type == OperandType.Argument)
                    {
                        SpvInstruction source = context.Get(dest.VarType, assignment.Source);
                        context.Store(context.GetArgumentPointer(dest), source);
                    }
                    else
                    {
                        throw new NotImplementedException(dest.Type.ToString());
                    }
                }
                else if (node is AstOperation operation)
                {
                    Instructions.Generate(context, operation);
                }
            }
        }
    }
}
