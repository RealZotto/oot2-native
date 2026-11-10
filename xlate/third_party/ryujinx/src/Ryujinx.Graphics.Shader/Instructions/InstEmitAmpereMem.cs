using Ryujinx.Graphics.Shader.Decoders;
using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.Translation;
using System.Collections.Generic;
using System.Numerics;
using static Ryujinx.Graphics.Shader.Instructions.InstEmitAluHelper;
using static Ryujinx.Graphics.Shader.Instructions.InstEmitHelper;
using static Ryujinx.Graphics.Shader.IntermediateRepresentation.OperandHelper;

namespace Ryujinx.Graphics.Shader.Instructions
{
    // Ampere attribute, memory and texture instructions. Field positions: Mesa NAK sm70_encode.rs (OpIpa, OpALd, OpASt, OpLd, OpSt, OpAtom, OpTex ...).
    static partial class InstEmit
    {
        public static void RegisterAmpereMemory(AmpereRegistrar x)
        {
            const InstProps N = InstProps.None;

            x(AmpereOpcodes.Ipa, InstName.Ipa, AIpa, N);
            x(AmpereOpcodes.IpaIdx, InstName.Ipa, AIpa, N);
            x(AmpereOpcodes.Ald, InstName.Ald, AAld, N);
            x(AmpereOpcodes.Ast, InstName.Ast, AAst, N, writesRd: false);

            x(0x981, InstName.Ldg, ALdg, N);
            x(0x381, InstName.Ldg, ALdg, N);
            x(0xf8c, InstName.Atoms, AAtoms, N);
            x(0x986, InstName.Stg, AStg, N, writesRd: false);
            x(0x386, InstName.Stg, AStg, N, writesRd: false);
            x(0x98e, InstName.Red, ARed, N, writesRd: false);
            x(0x9a8, InstName.Atom, AAtomg, N);
            x(0x3a8, InstName.Atom, AAtomg, N);
            x(0x3a9, InstName.AtomCas, AAtomgCas, N);   // oot2-native: ATOMG.CAS, from nxnative pull request 1275
            x(AmpereOpcodes.Ldl, InstName.Ldl, ALdLocalShared, N);
            x(0x984, InstName.Lds, ALdLocalShared, N);
            x(0x387, InstName.Stl, AStLocalShared, N, writesRd: false);
            x(0x987, InstName.Stl, AStLocalShared, N, writesRd: false);
            x(0x388, InstName.Sts, AStLocalShared, N, writesRd: false);
            x(0x988, InstName.Sts, AStLocalShared, N, writesRd: false);
            x(0x38c, InstName.Atoms, AAtoms, N);
            x(0x38d, InstName.Atoms, AAtomsCas, N);
            x(0x992, InstName.Membar, AMembar, N, writesRd: false);
            x(0x9ab, InstName.Membar, AErrbar, N, writesRd: false);

            x(0xb60, InstName.Tex, ATex, InstProps.Tex);
            x(0x361, InstName.Tex, ATex, InstProps.Tex);
            x(0x364, InstName.Tld4, ATex, InstProps.Tex);
            x(0x367, InstName.Tld, ATex, InstProps.Tex);
            x(0xf60, InstName.Tex, ATex, InstProps.Tex);
            x(0xf63, InstName.Tld4, ATex, InstProps.Tex);
            x(0xf6c, InstName.Txd, ATex, InstProps.Tex);
            x(0xb66, InstName.Tld, ATex, InstProps.Tex);
            x(0xb63, InstName.Tld4, ATex, InstProps.Tex);
            x(0xb69, InstName.Tmml, ATex, InstProps.Tex);
            x(0xb6c, InstName.Txd, ATex, InstProps.Tex);
            x(0xb6f, InstName.Txq, ATxq, InstProps.Tex);
            x(0x370, InstName.Txq, ATxq, InstProps.Tex);
            x(0xb97, InstName.Suld, ASurface, N);
            x(0xb99, InstName.Suld, ASurface, N);
            x(0xb9b, InstName.Sust, ASurface, N, writesRd: false);
            x(0xb9d, InstName.Sust, ASurface, N, writesRd: false);
            x(0x593, InstName.Suatom, ASuatom, N);
            x(0x59f, InstName.Sured, ASured, N, writesRd: false);
            x(0xb1d, InstName.Bar, ABar, N, writesRd: false);
        }

        // ---- attributes ----

        // IPA Rd, a[addr] (0x326) or a[URa + addr] (0x526, the index a uniform register at 24..30): addr 64..72 in words, sample location 76..78 (default,
        // centroid, offset), frequency 78..80 (pass, constant, state). Like Maxwell's IPA.PASS, a perspective attribute is read divided by w: the shader
        // multiplies by 1 / a[0x7c] itself.
        public static void AIpa(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            int addr = x.Bits(64, 72) << 2;

            if (x.Bits(76, 78) == 2)
            {
                throw AFail(context, "IPA at an offset");
            }

            Operand idxAddress = null;

            if (x.Opcode == AmpereOpcodes.IpaIdx)
            {
                idxAddress = context.IAdd(Register(x.Bits(24, 30), RegisterType.Uniform), Const(addr));
            }

            IpaOp ipaOp = x.Bits(78, 80) == 1 ? IpaOp.Constant : IpaOp.Pass;

            EmitIpa(context, idxAddress, addr, ipaOp, RegisterConsts.RegisterZeroIndex, false, x.Rd);
        }

        // Geometry and tessellation stages. OUT (opcode 0x124 in the ALU forms): Rd = the new output handle, Ra the old, the stream in the second source; 78..80 emit (1),
        // cut (2) or both, none being OUTFINAL. ISBERD Rd, [Ra]: the stage buffer entry of a vertex, which only ALD reads: Ra passes through, as on Maxwell. A geometry shader
        // that ends its output through an exit stub sets it up first: LEPC R, R + offset → 0x357 installs it as the address EXIT continues at, and the stub runs OUTFINAL
        // and 0x957 before its own EXIT. Per thread, with no output to finish, EXIT just ends the invocation: LEPC writes its own address (only the stub setup reads it),
        // and 0x357 and 0x957 do nothing (Donkey Kong Bananza's geometry shader 5d537f7630ec3671, which writes its primitives to memory with ATOMG / STG and emits none)
        public static void RegisterAmpereGeometry(AmpereAluRegistrar alu, AmpereRegistrar x)
        {
            const InstProps N = InstProps.None;

            alu(0x124, InstName.Out, AOut);
            x(0x923, InstName.Isberd, AIsberd, N);
            x(0x927, InstName.Ast, AStageOut, N, writesRd: false);
            x(0x34e, InstName.Lepc, ALepc, N);
            x(0x357, InstName.Nop, ANop, N, writesRd: false);
            x(0x957, InstName.Nop, ANop, N, writesRd: false);
        }

        public static void AOut(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            int type = x.Bits(78, 80);

            if (type != 0)
            {
                EmitOut(context, (type & 1) != 0, (type & 2) != 0);
            }

            if (x.Rd < RegisterConsts.RegisterZeroIndex)
            {
                context.Copy(Register(x.Rd, RegisterType.Gpr), AGpr(x.Ra));
            }
        }

        public static void AIsberd(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            MeshStageInfo mesh = context.TranslatorContext.GpuAccessor.QueryAmpereMeshStage();

            if (mesh != null)
            {
                EmitMeshStageRead(context, x, mesh);
                return;
            }

            if (x.Rd < RegisterConsts.RegisterZeroIndex)
            {
                context.Copy(Register(x.Rd, RegisterType.Gpr), AGpr(x.Ra));
            }
        }

        // oot2-native, Lichii's model (CodeGen/Spirv/MeshEmission.cs): ISBERD in a task or mesh program, [Ra + offset] a byte address in the region. Mode 0x78: the workgroup's
        // number; 0x38: a word of the payload the task wrote, whose word 0 is the workgroup's number; 0xb8: a word of the payload this task writes, read back (from nxnative pull
        // request 1300, commit 06f47f57, MIT: Ocarina of Time's agl task shader stores a word of its payload and reads it back). The draw's first task (MeshAbi) is added to the
        // workgroup's number in a task, or in a mesh stage without a task: this is the one place it goes
        private static void EmitMeshStageRead(EmitterContext context, AmpereOp x, MeshStageInfo mesh)
        {
            int mode = x.Bits(72, 80);
            bool task = mesh.Kind == MeshStageKind.Task;

            if (mode != 0x78 && mode != 0x38 && !(mode == 0xb8 && task))
            {
                throw AFail(context, $"ISBERD mode {mode:x2} in a {(task ? "task" : "mesh")} shader");
            }

            Operand workgroup = context.Load(StorageKind.Input, IoVariable.CtaId, null, Const(0));

            // The draw's first task counts in the number of a task workgroup, and in a mesh workgroup's when there is no task stage: a mesh
            // workgroup under a task is numbered within its task's launch
            if (mode != 0xb8 && (task || !mesh.HasTask))
            {
                workgroup = context.IAdd(workgroup, FirstTask(context));
            }

            Operand value = workgroup;

            if (mode == 0xb8 || (mode == 0x38 && !task))
            {
                int payload = context.ResourceManager.MeshPayloadId;
                Operand address = context.IAdd(AGpr(x.Ra), Const((int)x.SignedField(40, 64)));

                // A mesh stage without a task has no payload: its words read as zero, as nxnative's stage memory does
                Operand word = payload >= 0 ? context.Load(StorageKind.SharedMemory, payload, MeshWord(context, address, mesh.PayloadWords)) : Const(0);

                value = mode == 0xb8 ? word : context.ConditionalSelect(context.ICompareEqual(address, Const(0)), workgroup, word);
            }

            if (x.Rd < RegisterConsts.RegisterZeroIndex)
            {
                context.Copy(Register(x.Rd, RegisterType.Gpr), value);
            }
        }

        // oot2-native: the draw's first task, one 32-bit word at the device address in push constant words 6 and 7 (MeshAbi). A direct draw points the address at a word it
        // wrote, an indirect draw at the word its conversion pass wrote beside the record, so one module serves both forms; the access is a plain load that no pass rewrites
        // (DeviceMemory), not a guest access through the page table
        private static Operand FirstTask(EmitterContext context)
        {
            context.ResourceManager.Properties.UsesFirstTask = true;

            Operand low = context.Load(StorageKind.PushConstant, MeshAbi.FirstTaskPushWord);
            Operand high = context.Load(StorageKind.PushConstant, MeshAbi.FirstTaskPushWord + 1);

            return context.Load(StorageKind.DeviceMemory, low, high);
        }

        // The word of a stage memory region that a byte address names, held inside the region (Lichii clamps every index)
        private static Operand MeshWord(EmitterContext context, Operand byteAddress, int words)
        {
            return context.IMinimumU32(context.ShiftRightU32(byteAddress, Const(2)), Const(words - 1));
        }

        // oot2-native, Lichii's model (CodeGen/Spirv/MeshEmission.cs): task and mesh shaders' stage output (0x927), [Ra + offset] a byte address in the region, Rb, mode 72..79: 0xb8 a word
        // of a task's payload; 0xf8 a word of a mesh shader's attribute lines (each component 0x80 bytes for 32 vertices or primitives); 0x80 a map byte, kept whole in a word of its own
        // (so that neighbouring lanes need no compare-and-swap); 0x88 the primitive count, a 32-bit store at map byte 3. A mesh workgroup of 32 invocations is one warp on the console,
        // all at the same instruction, so the zero count each writes first (Mario Kart World: at its start) cannot pass the count one writes later; here they are not in step, so a
        // store from the zero register is left out and the prologue zeroes the count before a barrier
        public static void AStageOut(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            MeshStageInfo mesh = context.TranslatorContext.GpuAccessor.QueryAmpereMeshStage();
            int mode = x.Bits(72, 80);
            bool task = mesh?.Kind == MeshStageKind.Task;

            if (mesh == null || !(task ? mode == 0xb8 : mode is 0xf8 or 0x88 or 0x80))
            {
                throw AFail(context, $"stage output store mode {mode:x2} outside a {(mesh == null ? "task or mesh" : task ? "task" : "mesh")} shader");
            }

            ResourceManager resources = context.ResourceManager;
            int offset = (int)x.SignedField(40, 64);
            Operand address = context.IAdd(AGpr(x.Ra), Const(offset));
            Operand value = AGpr(x.Rb);

            switch (mode)
            {
                case 0xb8:
                    context.Store(StorageKind.SharedMemory, resources.MeshPayloadId, MeshWord(context, address, mesh.PayloadWords), value);
                    break;
                case 0xf8:
                    context.Store(StorageKind.SharedMemory, resources.MeshAttributesId, MeshWord(context, address, mesh.AttributeWords), value);
                    break;
                case 0x80:
                    context.Store(StorageKind.SharedMemory, resources.MeshMapId, context.IMinimumU32(address, Const(mesh.MapWords - 1)), context.BitwiseAnd(value, Const(0xff)));
                    break;
                default:
                    if (x.Rb == RegisterConsts.RegisterZeroIndex)
                    {
                        return;
                    }

                    if (x.Ra != RegisterConsts.RegisterZeroIndex || offset != 3)
                    {
                        throw AFail(context, "32-bit stage buffer map store at an address other than byte 3");
                    }

                    context.Store(StorageKind.SharedMemory, resources.MeshCountId, Const(0), value);
                    break;
            }
        }

        public static void ALepc(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            if (x.Rd < RegisterConsts.RegisterZeroIndex)
            {
                context.Copy(Register(x.Rd, RegisterType.Gpr), Const((int)context.CurrOp.Address));
            }

            if (x.Rd + 1 < RegisterConsts.RegisterZeroIndex)
            {
                context.Copy(Register(x.Rd + 1, RegisterType.Gpr), Const(0));
            }
        }

        // ALD Rd, a[Ra + addr], vtx: addr 40..50, vertex 32..40, count - 1 74..76, patch 76, physical (Ra the whole address) 77, output 79
        public static void AAld(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            if (!x.Bit(77) && x.Ra != RegisterConsts.RegisterZeroIndex)
            {
                throw AFail(context, "ALD with an offset register and an address");
            }

            EmitAld(context, x.Rd, x.Ra, x.Rb, x.Bits(40, 50), x.Bits(74, 76), x.Bit(76), x.Bit(79), x.Bit(77));
        }

        // AST a[Ra + addr], data (32..40), vtx (64..72)
        public static void AAst(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            if (!x.Bit(77) && x.Ra != RegisterConsts.RegisterZeroIndex)
            {
                throw AFail(context, "AST with an offset register and an address");
            }

            EmitAst(context, x.Ra, x.Rb, x.Bits(40, 50), x.Bits(74, 76), x.Bit(76), x.Bit(77));
        }

        // ---- global memory ----

        // The 64-bit address of a global access: Ra (a pair with bit raPairBit, else 32 bits) plus a uniform register pair at urLo (a pair with bit 72) in
        // the uniform form (bit 91), plus the signed offset (40..64). The low word stays (UR + Ra + offset), the form Ryujinx's global-to-storage pass
        // traces back to a storage buffer table entry in c[0].
        private static (Operand Low, Operand High) AGlobalAddress(EmitterContext context, AmpereOp x, int raPairBit, int urLo, int extraOffset)
        {
            bool uniformForm = x.Bit(91);
            bool raPair = uniformForm ? x.Bit(raPairBit) : x.Bit(72);
            int offset = (int)x.SignedField(40, 64) + extraOffset;

            Operand lo = Register(x.Ra, RegisterType.Gpr);
            Operand hi = raPair && x.Ra != RegisterConsts.RegisterZeroIndex ? Register(x.Ra + 1, RegisterType.Gpr) : Const(0);

            if (uniformForm)
            {
                int ur = x.Bits(urLo, urLo + 6);

                if (ur != RegisterConsts.UniformZeroIndex)
                {
                    Operand uLo = Register(ur, RegisterType.Uniform);
                    Operand uHi = x.Bit(72) ? Register(ur + 1, RegisterType.Uniform) : Const(0);

                    Operand sum = context.IAdd(uLo, lo);
                    Operand carry = context.ConditionalSelect(context.ICompareLessUnsigned(sum, uLo), Const(1), Const(0));

                    hi = context.IAdd(context.IAdd(uHi, hi), carry);
                    lo = sum;
                }
            }

            if (offset != 0)
            {
                Operand sum = context.IAdd(lo, Const(offset));
                Operand carry = offset > 0
                    ? context.ConditionalSelect(context.ICompareLessUnsigned(sum, lo), Const(1), Const(0))
                    : context.ConditionalSelect(context.ICompareGreaterUnsigned(sum, lo), Const(-1), Const(0));

                hi = context.IAdd(hi, carry);
                lo = sum;
            }

            return (lo, hi);
        }

        // LDG Rd, [Ra + UR + offset]: size 73..76; the uniform form (0x981) loads only where its predicate (64..67, stored as 7 - P; 67 inverts) holds,
        // and is zero elsewhere
        public static void ALdg(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            LsSize size = (LsSize)x.Bits(73, 76);
            int count = GetVectorCount(size);
            StorageKind storageKind = GetStorageKind(size);

            Operand pred = null;

            if (x.Opcode == 0x981)
            {
                int p = 7 - x.Bits(64, 67);

                if (p != RegisterConsts.PredicateTrueIndex || x.Bit(67))
                {
                    Operand pr = Register(p, RegisterType.Predicate);
                    pred = x.Bit(67) ? context.BitwiseNot(pr) : pr;
                }
            }

            // The address is read once, before any result is written, the zeros of a false predicate too: the destination may be the address register (LDG.128 R0, [R0.64]; Unreal Engine 5.4's bounds-checked
            // reads of its scene data, LDG.P R5, [R5 + UR4], which read at the table's start when R5 was zeroed first: every instance at the first one's transform, and a black scene)
            (Operand[] los, Operand[] his) = (new Operand[count], new Operand[count]);

            for (int index = 0; index < count; index++)
            {
                (Operand lo, Operand hi) = AGlobalAddress(context, x, 90, 32, index * 4);
                (los[index], his[index]) = (context.Copy(lo), context.Copy(hi));
            }

            Operand lblSkip = null;

            if (pred != null)
            {
                for (int index = 0; index < count && x.Rd + index < RegisterConsts.RegisterZeroIndex; index++)
                {
                    context.Copy(Register(x.Rd + index, RegisterType.Gpr), Const(0));
                }

                lblSkip = Label();
                context.BranchIfFalse(lblSkip, pred);
            }

            for (int index = 0; index < count; index++)
            {
                if (x.Rd + index >= RegisterConsts.RegisterZeroIndex)
                {
                    break;
                }

                context.Copy(Register(x.Rd + index, RegisterType.Gpr), context.Load(storageKind, los[index], his[index]));
            }

            if (lblSkip != null)
            {
                context.MarkLabel(lblSkip);
            }
        }

        // STG [Ra + UR + offset], Rb: UR at 64 in the uniform form (0x986)
        public static void AStg(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            LsSize size = (LsSize)x.Bits(73, 76);

            if (size > LsSize.B128)
            {
                throw AFail(context, $"STG size {size}");
            }

            int count = GetVectorCount(size);
            StorageKind storageKind = GetStorageKind(size);

            for (int index = 0; index < count; index++)
            {
                bool isRz = x.Rb + index >= RegisterConsts.RegisterZeroIndex;
                Operand value = isRz ? Const(0) : Register(x.Rb + index, RegisterType.Gpr);

                (Operand lo, Operand hi) = AGlobalAddress(context, x, 90, 64, index * 4);

                context.Store(storageKind, lo, hi, value);
            }
        }

        // RED [Ra + UR + offset], Rb: type 73..76, operation 87..90
        public static void ARed(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            (Operand lo, Operand hi) = AGlobalAddress(context, x, 90, 64, 0);

            EmitAtomicOp(context, StorageKind.GlobalMemory, (AtomOp)x.Bits(87, 90), (AtomSize)x.Bits(73, 76), lo, hi, Register(x.Rb, RegisterType.Gpr));
        }

        // ATOMG Rd, [Ra + UR + offset], Rb: operation 87..91; the uniform form keeps Ra's pair bit at 70
        public static void AAtomg(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            (Operand lo, Operand hi) = AGlobalAddress(context, x, 70, 64, 0);

            Operand res = EmitAtomicOp(context, StorageKind.GlobalMemory, (AtomOp)x.Bits(87, 91), (AtomSize)x.Bits(73, 76), lo, hi, Register(x.Rb, RegisterType.Gpr));

            ASet(context, x, res);
        }

        // oot2-native: ported from nxnative pull request 1275 (commit 8fd3a0dc, MIT), unchanged but for this comment.
        // ATOMG.CAS Rd, [Ra + offset], Rb, Rc (0x3a9): compare with Rb, swap in Rc (64..72), the old value to Rd (Mesa NAK sm70_encode.rs OpAtom, AtomOp::CmpExch:
        // compare at 32, data at 64); address as ATOMG's (.E pair bit 72). Only the 32-bit form (type 73..76 of 0 or 1) in the non-uniform form is met: others are refused
        public static void AAtomgCas(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            if (x.Bits(73, 76) > 1 || x.Bit(91))
            {
                throw AFail(context, $"ATOMG.CAS type {x.Bits(73, 76)} or uniform address");
            }

            (Operand lo, Operand hi) = AGlobalAddress(context, x, 72, 64, 0);
            Operand res = context.AtomicCompareAndSwap(StorageKind.GlobalMemory, lo, hi, AGpr(x.Rb), AGpr(x.Bits(64, 72)));

            ASet(context, x, res);
        }

        // MEMBAR: the order of this thread's memory operations
        public static void AMembar(EmitterContext context)
        {
            context.MemoryBarrier();
        }

        // ERRBAR, after a MEMBAR: waits for errors of earlier memory operations to be reported, which here is nothing to do
        public static void AErrbar(EmitterContext context)
        {
        }

        // ATOMS.CAS Rd, [Ra + offset], Rb (expected), Rc (new value): Rd = 1 when the swap happened, not the old value (Fire Emblem Fortune's Weave's only use is a retry loop for a float add that repeats until Rd == 1).
        // Only the 32-bit form (operation 3 at 87..91, size 0) occurs
        public static void AAtomsCas(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            Operand ra = x.Ra >= RegisterConsts.RegisterZeroIndex ? Const(0) : AStride(context, x, Register(x.Ra, RegisterType.Gpr));
            Operand offset = context.ShiftRightU32(context.IAdd(ra, Const((int)x.SignedField(40, 64))), Const(2));
            Operand expected = context.Copy(x.Rb >= RegisterConsts.RegisterZeroIndex ? Const(0) : Register(x.Rb, RegisterType.Gpr));
            Operand value = x.Rc >= RegisterConsts.RegisterZeroIndex ? Const(0) : Register(x.Rc, RegisterType.Gpr);
            Operand old = context.AtomicCompareAndSwap(StorageKind.SharedMemory, Const(context.ResourceManager.SharedMemoryId), offset, expected, value);

            ASet(context, x, context.ConditionalSelect(context.ICompareEqual(old, expected), Const(1), Const(0)));
        }

        // ---- local and shared memory ----

        // A shared or local address register's stride, bits 78..79: it counts elements of 1, 4, 8 or 16 bytes (Fire Emblem Fortune's Weave has 1 on all its 32-bit and 2 on all its 64-bit accesses with a
        // register, whose code bounds that register by the element count). The field is a scale, not the access size: Pokemon Pokopia has 3 on 3,312 shared accesses of 4, 8 and 16 bytes
        private static Operand AStride(EmitterContext context, AmpereOp x, Operand ra)
        {
            int stride = x.Bits(78, 80);

            return stride == 0 ? ra : context.ShiftLeft(ra, Const(stride + 1));
        }

        // The 32-bit address of a local / shared access: Ra by its stride, plus the uniform register (at urLo in the uniform form, bit 91). LDSM keeps its mode in the stride's bits, so it passes
        // stride: false: scaled by that mode, Donkey Kong Bananza's upscaler read its transposed matrices (LDSM.MT88) four times too far along and the picture came out as an 8 x 8 lattice of garbage
        private static Operand ALocalAddress(EmitterContext context, AmpereOp x, int urLo, bool stride = true)
        {
            Operand ra = Register(x.Ra, RegisterType.Gpr);
            Operand addr = x.Ra == RegisterConsts.RegisterZeroIndex ? Const(0) : stride ? AStride(context, x, ra) : ra;

            if (x.Bit(91) && x.Bits(urLo, urLo + 6) != RegisterConsts.UniformZeroIndex)
            {
                addr = context.IAdd(addr, Register(x.Bits(urLo, urLo + 6), RegisterType.Uniform));
            }

            return addr;
        }

        // LDL / LDS Rd, [Ra + UR + offset]: UR at 32, size 73..76, offset 40..64
        public static void ALdLocalShared(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            StorageKind kind = x.Opcode == AmpereOpcodes.Ldl ? StorageKind.LocalMemory : StorageKind.SharedMemory;

            EmitLoad(context, kind, (LsSize2)x.Bits(73, 76), ALocalAddress(context, x, 32), x.Rd, (int)x.SignedField(40, 64));
        }

        // STL / STS [Ra + UR + offset], Rb: UR at 64
        public static void AStLocalShared(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            StorageKind kind = x.Opcode is 0x387 or 0x987 ? StorageKind.LocalMemory : StorageKind.SharedMemory;

            EmitStore(context, kind, (LsSize2)x.Bits(73, 76), ALocalAddress(context, x, 64), x.Rb, (int)x.SignedField(40, 64));
        }

        // ATOMS Rd, [Ra + offset], Rb: an atomic on shared memory, its operation (87..91) and size (73..76) as ATOMG's, its offset in bytes as STS's (Pokemon Pokopia's 51 compute shaders: ADD on
        // counters at the end of their shared memory, MAX; Splatoon Raiders has 6)
        public static void AAtoms(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            bool one = x.Opcode == 0xf8c;

            // 0xf8c, ATOMS.POPC.INC: an increment by 1 with no source register, the operation field 10 (the hardware sums the warp's lanes on one address; per lane an add of 1). Its one user is
            // Pokemon Pokopia's luminance histogram, hist[bin]++ (2118ab534fac88c6): the field for Rb names a register the shader never wrote on three of its four paths. Other encodings stop here
            if (one ? x.Bits(87, 91) != 10 || !x.Bit(91) || x.Rd != RegisterConsts.RegisterZeroIndex : x.Bit(91))
            {
                throw AFail(context, "ATOMS with a uniform register");
            }

            Operand offset = context.ShiftRightU32(context.IAdd(ALocalAddress(context, x, 64), Const((int)x.SignedField(40, 64))), Const(2));
            Operand id = Const(context.ResourceManager.SharedMemoryId);

            if (one)
            {
                EmitAtomicOp(context, StorageKind.SharedMemory, AtomOp.Add, (AtomSize)x.Bits(73, 76), id, offset, Const(1));
                return;
            }

            Operand value = x.Rb >= RegisterConsts.RegisterZeroIndex ? Const(0) : Register(x.Rb, RegisterType.Gpr);
            Operand res = EmitAtomicOp(context, StorageKind.SharedMemory, (AtomOp)x.Bits(87, 91), (AtomSize)x.Bits(73, 76), id, offset, value);

            ASet(context, x, res);
        }

        // ---- surfaces ----

        // SULD Rd, [Ra] (0xb97 formatted with a channel mask at 72..76, 0xb99 binary with a size at 73..76) and SUST [Ra], Rb (0xb9b, 0xb9d): the handle as
        // a texture's (40..59), the dimension at 61..64 (1D, 1D buffer, 1D array, 2D, 2D array, 3D: Maxwell's SuDim)
        public static void ASurface(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            (int handle, int cbufSlot) = ATexHandle(context, x);
            bool formatted = x.Opcode is 0xb97 or 0xb9b;
            SuDim dim = (SuDim)x.Bits(61, 64);
            SuSize size = formatted ? 0 : (SuSize)x.Bits(73, 76);
            SuRgba mask = formatted ? (SuRgba)x.Bits(72, 76) : 0;

            if (x.Opcode is 0xb97 or 0xb99)
            {
                EmitSuld(context, CacheOpLd.Ca, dim, size, handle, mask, x.Ra, x.Rd, 0, formatted, false, false, cbufSlot);
            }
            else
            {
                EmitSust(context, CacheOpSt.Wb, dim, size, handle, mask, x.Ra, x.Rb, 0, formatted, false, false, cbufSlot);
            }
        }

        // SURED [Ra], Rb (0x59f): an image reduction through a handle, laid out as SUATOM with no result: the coordinates from Ra, the value in Rb, the handle at 40..59, the dimension
        // at 61..64, the size at 73..76 and the operation at 87..91 (Eglantine's lowering to Maxwell names the fields). Mario Kart World's compute shader adds 1 at (R4, R5) of a 2D
        // image; its word 2 is SUATOM's in Donkey Kong Bananza's encoding less the predicate result. Yoshi and the Mysterious Book's compute shaders take other operations in SUATOM's field (MAX, OR, MIN on
        // 32 bits); a size other than 32 bits (its Nanite shaders' MAX.U64) or another form is refused
        public static void ASured(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            if ((x.Bits(64, 96) & ~(0xf << 23)) != 0x0010e000 || x.Bits(59, 61) != 0)
            {
                throw AFail(context, $"SURED form {x.Bits(64, 96):x8}");
            }

            (int handle, int cbufSlot) = ATexHandle(context, x);

            EmitSured(context, (SuDim)x.Bits(61, 64), 0, (RedOp)x.Bits(87, 91), SuatomSize.U32, handle, x.Ra, x.Rb, 0, false, isBindless: false, cbufSlot: cbufSlot);
        }

        // SUATOM Rd, [Ra], Rb (0x593): an atomic on an image through a handle, laid out as SUST's (handle 40..59, dimension 61..64) and as ATOMG's (operation
        // 87..91, size 73..76, Pu 81..83). Donkey Kong Bananza's append compute shaders take a counter with it (SUATOM.ADD.U32 on a 1D image, the old value
        // shuffled to the warp) before their SUST. The handle may be in the game's own uniform buffer, as a texture's. A predicate result is not met in a game: refused.
        public static void ASuatom(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            (int handle, int cbufSlot) = ATexHandle(context, x);

            if (x.Bits(81, 84) != 7)
            {
                throw AFail(context, "SUATOM with a predicate result");
            }

            EmitSuatom(context, (SuDim)x.Bits(61, 64), 0, (SuatomOp)x.Bits(87, 91), (SuatomSize)x.Bits(73, 76), handle, x.Ra, x.Rb, 0, x.Rd, false, isBindless: false, compareAndSwap: false, cbufSlot: cbufSlot);
        }

        // BAR.SYNC: the workgroup barrier
        public static void ABar(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);

            if (x.Bits(77, 80) != 0 && x.Bits(77, 80) != 4)
            {
                throw AFail(context, $"BAR mode {x.Bits(77, 80)}");
            }

            context.Barrier();
        }

        // ---- textures ----

        // Texture dimension (61..64) → Maxwell's TexDim
        private static TexDim ATexDim(EmitterContext context, int dim)
        {
            return dim switch
            {
                0 => TexDim._1d,
                4 => TexDim.Array1d,
                1 => TexDim._2d,
                5 => TexDim.Array2d,
                2 => TexDim._3d,
                3 => TexDim.Cube,
                7 => TexDim.ArrayCube,
                _ => throw AFail(context, $"texture dimension {dim}"),
            };
        }

        // The constant buffer word that holds the texture handle (40..59: word offset 0..14, bank 14..19). NVN keeps the handle table in c[2]
        // (0x20 + 8 slot), the texture buffer Ryujinx assumes; NVN2 compute's handles in c[0] are mapped there (AMapC0); a handle in another bank is passed
        // with its bank.
        private static (int Handle, int CbufSlot) ATexHandle(EmitterContext context, AmpereOp x)
        {
            (int bank, int offset) = AMapC0(context, x.Bits(54, 59), x.Bits(40, 54) * 4);

            return (offset >> 2, bank == 2 ? TextureOperation.DefaultCbufSlot : bank);
        }

        // The two source blocks of a texture instruction from Ra and Rb (n0 and n1 registers). With .SCR (60) the whole sequence was spread over the two
        // registers instead (4 → 2 + 2, 3 → 2 + 1, 2 → 1 + 1; NAK try_scalarize_tex).
        private static (List<Operand> A, List<Operand> B) ATexBlocks(EmitterContext context, AmpereOp x, int n0, int n1)
        {
            List<Operand> seq = [];

            void Take(int reg, int n)
            {
                for (int k = 0; k < n; k++)
                {
                    seq.Add(reg + k >= RegisterConsts.RegisterZeroIndex ? Const(0) : context.Copy(Register(reg + k, RegisterType.Gpr)));
                }
            }

            if (x.Bit(60))
            {
                int total = n0 + n1;

                if (total > 4)
                {
                    throw AFail(context, "texture .SCR with more than four sources");
                }

                int k0 = total >= 3 ? 2 : 1;

                Take(x.Ra, System.Math.Min(k0, total));
                Take(x.Rb, total - System.Math.Min(k0, total));
            }
            else
            {
                Take(x.Ra, n0);
                Take(x.Rb, n1);
            }

            return (seq.GetRange(0, n0), seq.GetRange(n0, seq.Count - n0));
        }

        // The results (mask 72..76) of a texture instruction to their registers: in order to Rd, Rd + 1, ...; with a second destination (64..72) the first
        // two to Rd and the others to it. .F16 (79): two results a register, packed as halves (the first two to Rd, the others to the second destination or
        // Rd + 1).
        private static void ATexResults(EmitterContext context, AmpereOp x, Operand[] results, bool f16)
        {
            int d0 = x.Rd;
            int d1 = x.Rc;
            bool split = d1 != RegisterConsts.RegisterZeroIndex;

            void Write(int reg, Operand value)
            {
                if (reg < RegisterConsts.RegisterZeroIndex)
                {
                    context.Copy(Register(reg, RegisterType.Gpr), value);
                }
            }

            if (d0 == RegisterConsts.RegisterZeroIndex)
            {
                return;
            }

            if (f16)
            {
                for (int pair = 0; pair * 2 < results.Length; pair++)
                {
                    Operand lo = results[pair * 2];
                    Operand hi = pair * 2 + 1 < results.Length ? results[pair * 2 + 1] : ConstF(0);
                    Write(pair == 0 ? d0 : split ? d1 : d0 + 1, context.PackHalf2x16(lo, hi));
                }

                return;
            }

            for (int i = 0; i < results.Length; i++)
            {
                Write(!split || i < 2 ? d0 + i : d1 + i - 2, results[i]);
            }
        }

        // TEX, TLD, TLD4, TMML, TXD. Source blocks (NAK nak_nir_lower_tex, SM70): A = array index, coordinates (TXD: coordinates, then the array index
        // with the offsets in its high half); B = LOD / bias, offset, depth reference, sample (TXD: the derivatives, x and y interleaved). The results are
        // read into locals, then written in Ampere's layout.
        public static void ATex(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            int opcode = x.Opcode;

            // TEX through a handle in a register, not named by its offset: the compiler's form for a handle it read itself, usually from a table at a computed offset. BindlessElimination takes the
            // handle from there, as for Maxwell's TEX.B. The other fields are 0xb60's. 0xf60, 0xf63 and 0xf6c (TEX, TLD4 and TXD, whose other fields are 0xb60's, 0xb63's and 0xb6c's): a uniform register at
            // 40..46. 0x361, 0x364 and 0x367 (TEX, TLD4 and TLD, bit 59 set): a register, the first of the second source block,
            // before the LOD (a census of Fire Emblem Fortune's Weave's 3,327: of the sources that an LDC just wrote, always that place, with and without .SCR). Rc stays the second destination in both
            Operand boundHandle = null;
            bool handleInB = opcode is 0x361 or 0x364 or 0x367;
            bool bindless = handleInB || opcode is 0xf60 or 0xf63 or 0xf6c;

            if (opcode is 0xf60 or 0xf63 or 0xf6c)
            {
                int ur = x.Bits(40, 46);

                boundHandle = ur == RegisterConsts.UniformZeroIndex ? Const(0) : context.Copy(Register(ur, RegisterType.Uniform));
            }

            if (bindless)
            {
                opcode = opcode is 0xf63 or 0xf6c ? opcode - 0x400 : opcode is 0x364 or 0x367 ? opcode + 0x7ff : 0xb60;
            }

            TexDim dim = ATexDim(context, x.Bits(61, 64));
            SamplerType type = ConvertSamplerType(dim);
            bool isArray = type.HasFlag(SamplerType.Array);
            int coords = type.Dimensions;
            int mask = x.Bits(72, 76);
            bool f16 = x.Bit(79) && opcode != 0xb6c;

            if (x.Bits(81, 84) != RegisterConsts.PredicateTrueIndex && opcode != 0xb69)
            {
                throw AFail(context, "texture residency predicate");
            }

            TextureFlags flags = TextureFlags.None;
            int lod = 0;
            bool offset = false, depth = false, ms = false, ptp = false;
            int n1 = 0;

            switch (opcode)
            {
                case 0xb60: // TEX: lod 87..90 (0 auto, 1 zero, 2 bias, 3 level), AOFFI 76, depth reference 78
                    lod = x.Bits(87, 90);
                    offset = x.Bit(76);
                    depth = x.Bit(78);
                    if (lod > 3)
                    {
                        throw AFail(context, $"TEX lod mode {lod}");
                    }

                    n1 = (lod >= 2 ? 1 : 0) + (offset ? 1 : 0) + (depth ? 1 : 0);
                    break;
                case 0xb66: // TLD: lod 87..90 (1 zero, 3 level), AOFFI 76, multisample 78
                    lod = x.Bits(87, 90);
                    offset = x.Bit(76);
                    ms = x.Bit(78);
                    if (lod is not 1 and not 3)
                    {
                        throw AFail(context, $"TLD lod mode {lod}");
                    }

                    flags |= TextureFlags.IntCoords;
                    n1 = (lod == 3 ? 1 : 0) + (offset ? 1 : 0) + (ms ? 1 : 0);
                    break;
                case 0xb63: // TLD4: offset mode 76..78 (1 AOFFI, 2 per texel), depth reference 78, component 87..89
                    int offMode = x.Bits(76, 78);
                    depth = x.Bit(78);
                    if (offMode == 3)
                    {
                        throw AFail(context, "TLD4 offset mode 3");
                    }

                    offset = offMode != 0;
                    ptp = offMode == 2;
                    flags |= TextureFlags.Gather;
                    n1 = (offMode == 2 ? 2 : offMode) + (depth ? 1 : 0);
                    break;
                case 0xb6c: // TXD: AOFFI 76
                    offset = x.Bit(76);
                    flags |= TextureFlags.Derivatives;
                    break;
            }

            (int handle, int cbufSlot) = bindless ? (0, TextureOperation.DefaultCbufSlot) : ATexHandle(context, x);
            List<Operand> sources = [];
            Operand arrayIndex = null;
            bool is1DTo2D = false;

            if (bindless)
            {
                flags |= TextureFlags.Bindless;
            }

            if (opcode == 0xb6c)
            {
                int n0 = coords + (isArray || offset ? 1 : 0);
                (List<Operand> a, List<Operand> b) = ATexBlocks(context, x, n0, 2 * coords);

                if (bindless)
                {
                    sources.Add(boundHandle);
                }

                sources.AddRange(a.GetRange(0, coords));

                if (Sample1DAs2D && (type & SamplerType.Mask) == SamplerType.Texture1D)
                {
                    sources.Add(ConstF(0));
                    type = SamplerType.Texture2D | (type & SamplerType.Array);
                    is1DTo2D = true;
                }

                if (isArray)
                {
                    arrayIndex = context.BitwiseAnd(a[coords], Const(0xffff));
                }

                if (isArray)
                {
                    sources.Add(arrayIndex);
                }

                sources.AddRange(b);

                if (is1DTo2D)
                {
                    sources.Add(ConstF(0));
                    sources.Add(ConstF(0));
                }

                if (offset)
                {
                    for (int i = 0; i < coords; i++)
                    {
                        sources.Add(context.BitfieldExtractS32(a[coords], Const(16 + i * 4), Const(4)));
                    }

                    if (is1DTo2D)
                    {
                        sources.Add(Const(0));
                    }

                    flags |= TextureFlags.Offset;
                }
            }
            else
            {
                int n0 = coords + (isArray ? 1 : 0);
                (List<Operand> a, List<Operand> b) = ATexBlocks(context, x, n0, n1 + (handleInB ? 1 : 0));
                int ia = 0, ib = 0;

                if (handleInB)
                {
                    boundHandle = b[ib++];
                }

                if (bindless)
                {
                    sources.Add(boundHandle);
                }

                if (isArray)
                {
                    arrayIndex = a[ia++];
                }

                for (int i = 0; i < coords; i++)
                {
                    sources.Add(a[ia++]);
                }

                // nxnative: as on Maxwell (InstEmitTexture.cs), the descriptor decides between a 1D texture and a texture buffer, and with no type answer (no game's NVN2 shaders have one)
                // a plain fetch reads 2D at (x & 0x3FFF, x >> 14): a 1D texture, one row on the host, reads (x, 0) as before, and the host binds a texture buffer as rows of 16384 texels over
                // the same memory (nvn_mtl.m tbuf_as_2d). Every other 1D form cannot be a texture buffer and reads row 0. Pokemon Pokopia's KTGL light lists read 0 before this, as a dummy
                if (Sample1DAs2D && (type & SamplerType.Mask) == SamplerType.Texture1D)
                {
                    bool plainFetch = opcode == 0xb66 && !isArray && lod == 1 && !offset && !ms;

                    if (!plainFetch || context.TranslatorContext.GpuAccessor.QuerySamplerType(handle) != SamplerType.TextureBuffer)
                    {
                        if (plainFetch)
                        {
                            Operand fetchX = sources[^1];
                            sources[^1] = context.BitwiseAnd(fetchX, Const(0x3FFF));
                            sources.Add(context.ShiftRightU32(fetchX, Const(14)));
                            context.TranslatorContext.GpuAccessor.RecordTexelFetch1D(handle);
                        }
                        else
                        {
                            sources.Add(opcode == 0xb66 ? Const(0) : ConstF(0));
                        }

                        type = SamplerType.Texture2D | (type & SamplerType.Array);
                        is1DTo2D = true;
                    }
                }

                if (isArray)
                {
                    sources.Add(arrayIndex);
                }

                Operand lodValue = null;

                if ((opcode == 0xb60 && lod >= 2) || (opcode == 0xb66 && lod == 3))
                {
                    lodValue = b[ib++];
                }

                Operand off0 = offset ? b[ib++] : null;
                Operand off1 = ptp ? b[ib++] : null;

                if (depth)
                {
                    sources.Add(b[ib++]);
                    type |= SamplerType.Shadow;
                }

                if (opcode == 0xb60 || opcode == 0xb66)
                {
                    if (!ms && (lod == 1 || lod == 3))
                    {
                        sources.Add(lodValue ?? (opcode == 0xb66 ? Const(0) : ConstF(0)));
                        flags |= TextureFlags.LodLevel;
                    }
                }

                if (offset && opcode != 0xb63)
                {
                    for (int i = 0; i < coords; i++)
                    {
                        sources.Add(context.BitfieldExtractS32(off0, Const(i * 4), Const(4)));
                    }

                    if (is1DTo2D)
                    {
                        sources.Add(Const(0));
                    }

                    flags |= TextureFlags.Offset;
                }
                else if (offset)
                {
                    int texels = ptp ? 4 : 1;

                    for (int i = 0; i < coords * texels; i++)
                    {
                        Operand packed = (i >> 2 & 1) == 0 ? off0 : off1;
                        sources.Add(context.BitfieldExtractS32(packed, Const((i & 3) * 8), Const(6)));
                    }

                    flags |= ptp ? TextureFlags.Offsets : TextureFlags.Offset;
                }

                if (opcode == 0xb60 && lod == 2)
                {
                    sources.Add(lodValue);
                    flags |= TextureFlags.LodBias;
                }

                if (ms)
                {
                    sources.Add(b[ib++]);
                    type |= SamplerType.Multisample;
                }

                if (opcode == 0xb63 && !depth)
                {
                    sources.Add(Const(x.Bits(87, 89)));
                }

                if (opcode == 0xb66 && type == SamplerType.Texture1D && lod == 1 && !offset)
                {
                    if (context.TranslatorContext.GpuAccessor.QuerySamplerType(handle) == SamplerType.TextureBuffer)
                    {
                        type = SamplerType.TextureBuffer;
                        sources.RemoveAt(sources.Count - 1);
                        flags &= ~TextureFlags.LodLevel;
                    }
                }
            }

            int count = BitOperations.PopCount((uint)mask);
            Operand[] results = new Operand[count];

            for (int i = 0; i < count; i++)
            {
                results[i] = Local();
            }

            if (opcode == 0xb69)
            {
                ATmml(context, type, flags, handle, cbufSlot, mask, results, sources.ToArray());
            }
            else
            {
                SetBindingPair setAndBinding = bindless ? default : context.ResourceManager.GetTextureOrImageBinding(
                    Instruction.TextureSample,
                    type,
                    TextureFormat.Unknown,
                    flags,
                    cbufSlot,
                    handle);

                context.TextureSample(type, flags, setAndBinding, mask, results, sources.ToArray());
            }

            ATexResults(context, x, results, f16);
        }

        // TMML: the LOD a sample would use, x and y in 8.8 fixed point (as Maxwell's)
        private static void ATmml(EmitterContext context, SamplerType type, TextureFlags flags, int handle, int cbufSlot, int mask, Operand[] results, Operand[] sources)
        {
            SetBindingPair setAndBinding = context.ResourceManager.GetTextureOrImageBinding(Instruction.Lod, type, TextureFormat.Unknown, flags, cbufSlot, handle);
            int r = 0;

            for (int comp = 0; comp < 4; comp++)
            {
                if ((mask & (1 << comp)) == 0)
                {
                    continue;
                }

                if (comp >= 2)
                {
                    context.Copy(results[r++], Const(0));
                    continue;
                }

                Operand res = context.Lod(type, flags, setAndBinding, comp ^ 1, sources);

                // oot2-native: the level is what the driver's query returns, so it is bounded before the conversion (a NaN gives 0)
                context.Copy(results[r++], FP32ToS32Bounded(context, context.FPMultiply(res, ConstF(256.0f))));
            }
        }

        // TXQ Rd, Ra: query 62..64 (dimensions with Ra the LOD, texture type); results as the other texture instructions
        public static void ATxq(EmitterContext context)
        {
            AmpereOp x = new(context.CurrOp);
            int query = x.Bits(62, 64);
            int mask = x.Bits(72, 76);
            // 0x370 is to 0xb6f as 0x361 to 0xb60: the handle in a register, Ra, with the LOD after it (Fire Emblem Fortune's Weave's lighting: MOV R1c, R64; MOV R1d, RZ; TXQ R1d, R1c). The shape
            // has no field here and the size query does not depend on it, so 2D, as Maxwell's bindless TXQ
            bool bindless = x.Opcode == 0x370;
            (int handle, int cbufSlot) = bindless ? (0, TextureOperation.DefaultCbufSlot) : ATexHandle(context, x);
            SamplerType type = bindless ? SamplerType.Texture2D : context.TranslatorContext.GpuAccessor.QuerySamplerType(handle);
            TextureFlags flags = bindless ? TextureFlags.Bindless : TextureFlags.None;

            int count = BitOperations.PopCount((uint)mask);
            Operand[] results = new Operand[count];

            for (int i = 0; i < count; i++)
            {
                results[i] = Local();
            }

            Operand Src(int reg) => reg >= RegisterConsts.RegisterZeroIndex ? Const(0) : context.Copy(Register(reg, RegisterType.Gpr));

            Operand[] sources = bindless ? [Src(x.Ra), Src(x.Ra + 1)] : [Src(x.Ra)];

            if (query == 0)
            {
                SetBindingPair setAndBinding = bindless ? default : context.ResourceManager.GetTextureOrImageBinding(
                    Instruction.TextureQuerySize, type, TextureFormat.Unknown, TextureFlags.None, cbufSlot, handle);
                int r = 0;

                for (int comp = 0; comp < 4; comp++)
                {
                    if ((mask & (1 << comp)) != 0)
                    {
                        context.Copy(results[r++], context.TextureQuerySize(type, flags, setAndBinding, comp, sources));
                    }
                }
            }
            else if (query == 1)
            {
                SetBindingPair setAndBinding = bindless ? default : context.ResourceManager.GetTextureOrImageBinding(
                    Instruction.TextureQuerySamples, type, TextureFormat.Unknown, TextureFlags.None, cbufSlot, handle);
                int r = 0;

                for (int comp = 0; comp < 4; comp++)
                {
                    if ((mask & (1 << comp)) != 0)
                    {
                        context.Copy(results[r++], comp == 2 ? context.TextureQuerySamples(type, flags, setAndBinding, sources) : Const(0));
                    }
                }
            }
            else
            {
                throw AFail(context, $"TXQ query {query}");
            }

            ATexResults(context, x, results, false);
        }
    }
}
