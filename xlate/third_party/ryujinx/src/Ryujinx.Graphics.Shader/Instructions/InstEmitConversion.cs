using Ryujinx.Graphics.Shader.Decoders;
using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.Translation;
using System;
using static Ryujinx.Graphics.Shader.Instructions.InstEmitAluHelper;
using static Ryujinx.Graphics.Shader.Instructions.InstEmitHelper;
using static Ryujinx.Graphics.Shader.IntermediateRepresentation.OperandHelper;

namespace Ryujinx.Graphics.Shader.Instructions
{
    static partial class InstEmit
    {
        public static void F2fR(EmitterContext context)
        {
            InstF2fR op = context.GetOp<InstF2fR>();

            Operand src = UnpackReg(context, op.SrcFmt, op.Sh, op.SrcB);

            EmitF2F(context, op.SrcFmt, op.DstFmt, op.RoundMode, src, op.Dest, op.AbsB, op.NegB, op.Sat);
        }

        public static void F2fI(EmitterContext context)
        {
            InstF2fI op = context.GetOp<InstF2fI>();

            Operand src = UnpackImm(context, op.SrcFmt, op.Sh, Imm20ToFloat(op.Imm20));

            EmitF2F(context, op.SrcFmt, op.DstFmt, op.RoundMode, src, op.Dest, op.AbsB, op.NegB, op.Sat);
        }

        public static void F2fC(EmitterContext context)
        {
            InstF2fC op = context.GetOp<InstF2fC>();

            Operand src = UnpackCbuf(context, op.SrcFmt, op.Sh, op.CbufSlot, op.CbufOffset);

            EmitF2F(context, op.SrcFmt, op.DstFmt, op.RoundMode, src, op.Dest, op.AbsB, op.NegB, op.Sat);
        }

        public static void F2iR(EmitterContext context)
        {
            InstF2iR op = context.GetOp<InstF2iR>();

            Operand src = UnpackReg(context, op.SrcFmt, op.Sh, op.SrcB);

            EmitF2I(context, op.SrcFmt, op.IDstFmt, op.RoundMode, src, op.Dest, op.AbsB, op.NegB);
        }

        public static void F2iI(EmitterContext context)
        {
            InstF2iI op = context.GetOp<InstF2iI>();

            Operand src = UnpackImm(context, op.SrcFmt, op.Sh, Imm20ToFloat(op.Imm20));

            EmitF2I(context, op.SrcFmt, op.IDstFmt, op.RoundMode, src, op.Dest, op.AbsB, op.NegB);
        }

        public static void F2iC(EmitterContext context)
        {
            InstF2iC op = context.GetOp<InstF2iC>();

            Operand src = UnpackCbuf(context, op.SrcFmt, op.Sh, op.CbufSlot, op.CbufOffset);

            EmitF2I(context, op.SrcFmt, op.IDstFmt, op.RoundMode, src, op.Dest, op.AbsB, op.NegB);
        }

        public static void I2fR(EmitterContext context)
        {
            InstI2fR op = context.GetOp<InstI2fR>();

            Operand src = GetSrcReg(context, op.SrcB);

            EmitI2F(context, op.ISrcFmt, op.DstFmt, src, op.ByteSel, op.Dest, op.AbsB, op.NegB);
        }

        public static void I2fI(EmitterContext context)
        {
            InstI2fI op = context.GetOp<InstI2fI>();

            Operand src = GetSrcImm(context, Imm20ToSInt(op.Imm20));

            EmitI2F(context, op.ISrcFmt, op.DstFmt, src, op.ByteSel, op.Dest, op.AbsB, op.NegB);
        }

        public static void I2fC(EmitterContext context)
        {
            InstI2fC op = context.GetOp<InstI2fC>();

            Operand src = GetSrcCbuf(context, op.CbufSlot, op.CbufOffset);

            EmitI2F(context, op.ISrcFmt, op.DstFmt, src, op.ByteSel, op.Dest, op.AbsB, op.NegB);
        }

        public static void I2iR(EmitterContext context)
        {
            InstI2iR op = context.GetOp<InstI2iR>();

            Operand src = GetSrcReg(context, op.SrcB);

            EmitI2I(context, op.ISrcFmt, op.IDstFmt, src, op.ByteSel, op.Dest, op.AbsB, op.NegB, op.Sat, op.WriteCC);
        }

        public static void I2iI(EmitterContext context)
        {
            InstI2iI op = context.GetOp<InstI2iI>();

            Operand src = GetSrcImm(context, Imm20ToSInt(op.Imm20));

            EmitI2I(context, op.ISrcFmt, op.IDstFmt, src, op.ByteSel, op.Dest, op.AbsB, op.NegB, op.Sat, op.WriteCC);
        }

        public static void I2iC(EmitterContext context)
        {
            InstI2iC op = context.GetOp<InstI2iC>();

            Operand src = GetSrcCbuf(context, op.CbufSlot, op.CbufOffset);

            EmitI2I(context, op.ISrcFmt, op.IDstFmt, src, op.ByteSel, op.Dest, op.AbsB, op.NegB, op.Sat, op.WriteCC);
        }

        private static void EmitF2F(
            EmitterContext context,
            DstFmt srcType,
            DstFmt dstType,
            IntegerRound roundingMode,
            Operand src,
            int rd,
            bool absolute,
            bool negate,
            bool saturate)
        {
            Operand srcB = context.FPAbsNeg(src, absolute, negate, srcType.ToInstFPType());

            if (srcType == dstType)
            {
                srcB = roundingMode switch
                {
                    IntegerRound.Round => context.FPRound(srcB, srcType.ToInstFPType()),
                    IntegerRound.Floor => context.FPFloor(srcB, srcType.ToInstFPType()),
                    IntegerRound.Ceil => context.FPCeiling(srcB, srcType.ToInstFPType()),
                    IntegerRound.Trunc => context.FPTruncate(srcB, srcType.ToInstFPType()),
                    _ => srcB,
                };
            }

            // We don't need to handle conversions between FP16 <-> FP32
            // since we do FP16 operations as FP32 directly.
            // FP16 <-> FP64 conversions are invalid.
            if (srcType == DstFmt.F32 && dstType == DstFmt.F64)
            {
                srcB = context.FP32ConvertToFP64(srcB);
            }
            else if (srcType == DstFmt.F64 && dstType == DstFmt.F32)
            {
                srcB = context.FP64ConvertToFP32(srcB);
            }

            srcB = context.FPSaturate(srcB, saturate, dstType.ToInstFPType());

            WriteFP(context, dstType, srcB, rd);

            // TODO: CC.
        }

        private static void EmitF2I(
            EmitterContext context,
            DstFmt srcType,
            IDstFmt dstType,
            RoundMode2 roundingMode,
            Operand src,
            int rd,
            bool absolute,
            bool negate,
            bool toByte = false)
        {
            if (dstType == IDstFmt.U64)
            {
                context.TranslatorContext.GpuAccessor.Log("Unimplemented 64-bits F2I.");
            }

            Instruction fpType = srcType.ToInstFPType();

            bool isSignedInt = dstType is IDstFmt.S16 or IDstFmt.S32 or IDstFmt.S64;
            bool isSmallInt = dstType is IDstFmt.U16 or IDstFmt.S16;

            Operand srcB = context.FPAbsNeg(src, absolute, negate, fpType);

            if (srcType != DstFmt.F64)
            {
                // oot2-native: a single or a half source is bounded before it is converted (FP32ToIntSaturated). What follows is the double source, which no instruction of the
                // console reaches and which is as it was: its NaN and its values outside the range are still left to the driver.
                context.Copy(GetDest(rd), FP32ToIntSaturated(context, srcB, roundingMode, isSignedInt, toByte ? 8 : isSmallInt ? 16 : 32));

                return;
            }

            srcB = roundingMode switch
            {
                RoundMode2.Round => context.FPRound(srcB, fpType),
                RoundMode2.Floor => context.FPFloor(srcB, fpType),
                RoundMode2.Ceil => context.FPCeiling(srcB, fpType),
                RoundMode2.Trunc => context.FPTruncate(srcB, fpType),
                _ => srcB,
            };

            if (!isSignedInt)
            {
                // Negative float to uint cast is undefined, so we clamp the value before conversion.
                Operand c0 = srcType == DstFmt.F64 ? context.PackDouble2x32(0.0) : ConstF(0);

                srcB = context.FPMaximum(srcB, c0, fpType);
            }

            if (srcType == DstFmt.F64)
            {
                srcB = isSignedInt
                    ? context.FP64ConvertToS32(srcB)
                    : context.FP64ConvertToU32(srcB);
            }
            else
            {
                srcB = isSignedInt
                    ? context.FP32ConvertToS32(srcB)
                    : context.FP32ConvertToU32(srcB);
            }

            if (isSmallInt)
            {
                int min = (int)GetIntMin(dstType);
                int max = (int)GetIntMax(dstType);

                srcB = isSignedInt
                    ? context.IClampS32(srcB, Const(min), Const(max))
                    : context.IClampU32(srcB, Const(min), Const(max));
            }

            Operand dest = GetDest(rd);

            context.Copy(dest, srcB);

            // TODO: CC.
        }

        // oot2-native: F2I from a single (a half is widened first) to an integer of 32, 16 or 8 bits, as the console converts: the float is rounded as the instruction says, a value
        // outside the integer's range gives the nearest bound (an infinity too) and a NaN gives 0. OpConvertFToS and OpConvertFToU are undefined for a NaN, an infinity and a value
        // outside the range, and a select after the conversion does not take that back, so the rounded float is clamped between two floats inside the range first and the conversion
        // only ever sees a number the integer holds. What it costs is what the clamp cannot give:
        //   - the lower bound (-2^31, -32768, -128, 0) is a float, so the clamp alone saturates below;
        //   - 32767, 65535, 127 and 255 are floats, so the clamp alone saturates a 16-bit or an 8-bit integer above, and no integer clamp follows the conversion;
        //   - 2^31 - 1 and 2^32 - 1 are not floats (as float literals they round up to 2^31 and 2^32, which are outside): the clamp stops at the largest float below, 2147483520 or
        //     4294967040, and the integer's top value is selected where the source is above that float, which is where it is 2^31 or 2^32 and up. The source and the rounded float are
        //     on the same side of it, every float that large being a whole number, so the comparison reads the source and the rounded float keeps a single use;
        //   - NClamp gives its lower bound for a NaN: that is the 0 of an unsigned integer, and a signed one selects its 0.
        // So: unsigned 16 and 8 bits, a clamp; unsigned 32 bits, a clamp, a comparison and a select; signed 16 and 8 bits, a clamp, a NaN test and a select; signed 32 bits, both.
        // No branch, and a value in range converts to what it did. Toward zero has no rounding instruction: OpConvertFToS and OpConvertFToU round toward zero themselves, and
        // the clamp's bounds are whole numbers, so the clamped float truncates to exactly what the truncated float clamps to (a NaN is the lower bound either way).
        private static Operand FP32ToIntSaturated(EmitterContext context, Operand value, RoundMode2 roundingMode, bool signed, int bits)
        {
            Operand rounded = roundingMode switch
            {
                RoundMode2.Round => context.FPRound(value),
                RoundMode2.Floor => context.FPFloor(value),
                RoundMode2.Ceil => context.FPCeiling(value),
                _ => value,
            };

            float low = signed ? -(float)(1L << (bits - 1)) : 0f;
            float top = bits == 32 ? (signed ? 2147483520f : 4294967040f) : (1 << (signed ? bits - 1 : bits)) - 1;

            Operand bounded = context.Add(Instruction.FP32 | Instruction.Clamp, Local(), rounded, ConstF(low), ConstF(top));
            Operand res = signed ? context.FP32ConvertToS32(bounded) : context.FP32ConvertToU32(bounded);

            if (bits == 32)
            {
                res = context.ConditionalSelect(context.FPCompareLess(ConstF(top), value), Const(signed ? int.MaxValue : -1), res);
            }

            return signed ? context.ConditionalSelect(context.IsNan(value), Const(0), res) : res;
        }

        // oot2-native: a float to a signed 32-bit integer for the conversions the translator adds itself (a level of detail in fixed point, a coordinate or a size times the render
        // scale), whose float is whatever the driver or the host gave. OpConvertFToS is undefined for a NaN, an infinity and a value outside the integer's range, and a select after it
        // does not take that back, so the float is bounded before it is converted. Both bounds are floats inside the range: -2^31 is one, and 2147483520 is the largest below 2^31 (the
        // literal 2147483647 would round up to 2^31, which is outside). NClamp gives its lower bound for a NaN, so a NaN is 0 at no cost where the caller knows the value is not
        // negative (nonNegative, which makes 0 the lower bound) and costs a test and a select otherwise. Nothing changes for a value in range. F2I itself does not come through here:
        // it has its rounding and its smaller integers, and its top value is exact (FP32ToIntSaturated).
        public static Operand FP32ToS32Bounded(EmitterContext context, Operand value, bool nonNegative = false)
        {
            Operand bounded = context.Add(Instruction.FP32 | Instruction.Clamp, Local(), value, ConstF(nonNegative ? 0f : -2147483648f), ConstF(2147483520f));
            Operand res = context.FP32ConvertToS32(bounded);

            return nonNegative ? res : context.ConditionalSelect(context.IsNan(value), Const(0), res);
        }

        private static void EmitI2F(
            EmitterContext context,
            ISrcFmt srcType,
            DstFmt dstType,
            Operand src,
            ByteSel byteSelection,
            int rd,
            bool absolute,
            bool negate)
        {
            bool isSignedInt =
                srcType is ISrcFmt.S8 or
                ISrcFmt.S16 or
                ISrcFmt.S32 or
                ISrcFmt.S64;
            bool isSmallInt =
                srcType is ISrcFmt.U16 or
                ISrcFmt.S16 or
                ISrcFmt.U8 or
                ISrcFmt.S8;

            // TODO: Handle S/U64.

            Operand srcB = context.IAbsNeg(src, absolute, negate);

            if (isSmallInt)
            {
                int size = srcType is ISrcFmt.U16 or ISrcFmt.S16 ? 16 : 8;

                srcB = isSignedInt
                    ? context.BitfieldExtractS32(srcB, Const((int)byteSelection * 8), Const(size))
                    : context.BitfieldExtractU32(srcB, Const((int)byteSelection * 8), Const(size));
            }

            if (dstType == DstFmt.F64)
            {
                srcB = isSignedInt
                    ? context.IConvertS32ToFP64(srcB)
                    : context.IConvertU32ToFP64(srcB);
            }
            else
            {
                srcB = isSignedInt
                    ? context.IConvertS32ToFP32(srcB)
                    : context.IConvertU32ToFP32(srcB);
            }

            WriteFP(context, dstType, srcB, rd);

            // TODO: CC.
        }

        private static void EmitI2I(
            EmitterContext context,
            ISrcDstFmt srcType,
            ISrcDstFmt dstType,
            Operand src,
            ByteSel byteSelection,
            int rd,
            bool absolute,
            bool negate,
            bool saturate,
            bool writeCC)
        {
            if ((srcType & ~ISrcDstFmt.S8) > ISrcDstFmt.U32 || (dstType & ~ISrcDstFmt.S8) > ISrcDstFmt.U32)
            {
                context.TranslatorContext.GpuAccessor.Log("Invalid I2I encoding.");
                return;
            }

            bool srcIsSignedInt =
                srcType is ISrcDstFmt.S8 or
                ISrcDstFmt.S16 or
                ISrcDstFmt.S32;
            bool dstIsSignedInt =
                dstType is ISrcDstFmt.S8 or
                ISrcDstFmt.S16 or
                ISrcDstFmt.S32;
            bool srcIsSmallInt =
                srcType is ISrcDstFmt.U16 or
                ISrcDstFmt.S16 or
                ISrcDstFmt.U8 or
                ISrcDstFmt.S8;

            if (srcIsSmallInt)
            {
                int size = srcType is ISrcDstFmt.U16 or ISrcDstFmt.S16 ? 16 : 8;

                src = srcIsSignedInt
                    ? context.BitfieldExtractS32(src, Const((int)byteSelection * 8), Const(size))
                    : context.BitfieldExtractU32(src, Const((int)byteSelection * 8), Const(size));
            }

            src = context.IAbsNeg(src, absolute, negate);

            if (saturate)
            {
                int min = (int)GetIntMin(dstType);
                int max = (int)GetIntMax(dstType);

                src = dstIsSignedInt
                    ? context.IClampS32(src, Const(min), Const(max))
                    : context.IClampU32(src, Const(min), Const(max));
            }

            context.Copy(GetDest(rd), src);

            SetZnFlags(context, src, writeCC);
        }

        private static Operand UnpackReg(EmitterContext context, DstFmt floatType, bool h, int reg)
        {
            if (floatType == DstFmt.F32)
            {
                return GetSrcReg(context, reg);
            }
            else if (floatType == DstFmt.F16)
            {
                return GetHalfUnpacked(context, GetSrcReg(context, reg), HalfSwizzle.F16)[h ? 1 : 0];
            }
            else if (floatType == DstFmt.F64)
            {
                return GetSrcReg(context, reg, isFP64: true);
            }

            throw new ArgumentException($"Invalid floating point type \"{floatType}\".");
        }

        private static Operand UnpackCbuf(EmitterContext context, DstFmt floatType, bool h, int cbufSlot, int cbufOffset)
        {
            if (floatType == DstFmt.F32)
            {
                return GetSrcCbuf(context, cbufSlot, cbufOffset);
            }
            else if (floatType == DstFmt.F16)
            {
                return GetHalfUnpacked(context, GetSrcCbuf(context, cbufSlot, cbufOffset), HalfSwizzle.F16)[h ? 1 : 0];
            }
            else if (floatType == DstFmt.F64)
            {
                return GetSrcCbuf(context, cbufSlot, cbufOffset, isFP64: true);
            }

            throw new ArgumentException($"Invalid floating point type \"{floatType}\".");
        }

        private static Operand UnpackImm(EmitterContext context, DstFmt floatType, bool h, int imm)
        {
            if (floatType == DstFmt.F32)
            {
                return GetSrcImm(context, imm);
            }
            else if (floatType == DstFmt.F16)
            {
                return GetHalfUnpacked(context, GetSrcImm(context, imm), HalfSwizzle.F16)[h ? 1 : 0];
            }
            else if (floatType == DstFmt.F64)
            {
                return GetSrcImm(context, imm, isFP64: true);
            }

            throw new ArgumentException($"Invalid floating point type \"{floatType}\".");
        }

        private static void WriteFP(EmitterContext context, DstFmt type, Operand srcB, int rd)
        {
            Operand dest = GetDest(rd);

            if (type == DstFmt.F32)
            {
                context.Copy(dest, srcB);
            }
            else if (type == DstFmt.F16)
            {
                context.Copy(dest, context.PackHalf2x16(srcB, ConstF(0)));
            }
            else /* if (type == FPType.FP64) */
            {
                Operand dest2 = GetDest2(rd);

                context.Copy(dest, context.UnpackDouble2x32Low(srcB));
                context.Copy(dest2, context.UnpackDouble2x32High(srcB));
            }
        }

        private static Instruction ToInstFPType(this DstFmt type)
        {
            return type == DstFmt.F64 ? Instruction.FP64 : Instruction.FP32;
        }
    }
}
