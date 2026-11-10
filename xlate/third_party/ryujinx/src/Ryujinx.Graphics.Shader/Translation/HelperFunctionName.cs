namespace Ryujinx.Graphics.Shader.Translation
{
    enum HelperFunctionName
    {
        Invalid,

        ConvertDoubleToFloat,
        ConvertFloatToDouble,
        Hmma16816F16,
        // oot2-native: HMMA.16816 with an f16 accumulator on VK_KHR_cooperative_matrix (CodeGen/Spirv/CoopHmma.cs)
        Hmma16816F16Coop,
        SharedAtomicMaxS32,
        SharedAtomicMinS32,
        SharedStore8,
        SharedStore16,
        Shuffle,
        ShuffleDown,
        ShuffleUp,
        ShuffleXor,
        TexelFetchScale,
        TextureSizeUnscale,
    }
}
