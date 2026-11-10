// oot2-native: the two instructions of SPV_KHR_cooperative_matrix the translator emits (its type and its multiply-add),
// which the vendored grammar (Core 1.x, the NV names only) does not have. MIT, like the rest of Spv.Generator.

using static Spv.Specification;

namespace Spv.Generator
{
    public partial class Module
    {
        // The opcodes of SPV_KHR_cooperative_matrix (SPIR-V registry, extensions/KHR/SPV_KHR_cooperative_matrix)
        private const Op OpTypeCooperativeMatrixKHR = (Op)4456;
        private const Op OpCooperativeMatrixMulAddKHR = (Op)4459;

        /// <summary>The capability CooperativeMatrixKHR (6022).</summary>
        public const Capability CooperativeMatrixKHR = (Capability)6022;

        /// <summary>OpTypeCooperativeMatrixKHR: a matrix of rows x columns components shared by the invocations of scope; use is
        /// MatrixAKHR (0), MatrixBKHR (1) or MatrixAccumulatorKHR (2). Scope, rows, columns and use are constant ids.</summary>
        public Instruction TypeCooperativeMatrixKHR(Instruction componentType, Instruction scope, Instruction rows, Instruction columns, Instruction use, bool forceIdAllocation = false)
        {
            Instruction result = NewInstruction(OpTypeCooperativeMatrixKHR);

            result.AddOperand(componentType);
            result.AddOperand(scope);
            result.AddOperand(rows);
            result.AddOperand(columns);
            result.AddOperand(use);
            AddTypeDeclaration(result, forceIdAllocation);

            return result;
        }

        /// <summary>OpCooperativeMatrixMulAddKHR: a times b plus c, without the optional cooperative matrix operands (no
        /// saturation, floating-point components).</summary>
        public Instruction CooperativeMatrixMulAddKHR(Instruction resultType, Instruction a, Instruction b, Instruction c)
        {
            Instruction result = NewInstruction(OpCooperativeMatrixMulAddKHR, GetNewId(), resultType);

            result.AddOperand(a);
            result.AddOperand(b);
            result.AddOperand(c);
            AddToFunctionDefinitions(result);

            return result;
        }
    }
}
