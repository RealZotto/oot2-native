// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// HMMA on VK_KHR_cooperative_matrix (runtime/gpu/vk_hmma.c). A translated compute program with HMMA.16816 (an f16
// accumulator) calls the function dmsh_hmma16816_coop, which moves each lane's fragment registers into cooperative
// matrices, multiplies with one OpCooperativeMatrixMulAddKHR and moves D back out; the module also holds the
// scratch-memory helper dmsh_hmma16816_f16 with the same arguments, uncalled (the translator's CodeGen/Spirv/CoopHmma.cs).
// Which component of a matrix a lane holds is the implementation's choice, so the translator writes the moves in an order
// measured on NVIDIA (below), and at start-up this probe decides whether the device runs that form:
//
//   the gate   the device lists a 16x8x16 subgroup matrix with f16 A, B, C and result, for compute, and was made with
//              16-bit floats (the optional feature shader_float16) and the Vulkan memory model (an item);
//   the order  a probe loads row-major matrices whose components are their positions and reads back every lane's
//              components: each console element of every lane must sit in the component the translator's order names;
//   the result the same lanes' fragments of small integers multiplied by the native form and by the helper's algorithm
//              must give the same D, bit for bit, and the exact sums.
//
// When all three hold the modules run as translated. Otherwise, or with OOT2_VK_HMMA=helper, the pipeline builder makes
// every call of the native function a call of the helper, found by name, and takes the native function and the matrix
// declarations out of the module before it creates the pipeline (spv_hmma_fallback, oot2/spirv.h): the helper path, the
// form every HMMA had before. The driver's pipeline cache keys on the module it is given, so the two forms never share an
// entry. One line in the log names the decision and its reason, with the order the device showed.
#pragma once
#include <stddef.h>

// The component of the device's cooperative matrix that holds each console element of a lane's fragment (element e: the
// low half of register e / 2 for e even, the high half for e odd). A copy of the translator's CoopHmma.OrderA, OrderB
// and OrderC (xlate/third_party/ryujinx/src/Ryujinx.Graphics.Shader/CodeGen/Spirv/CoopHmma.cs); a test reads both.
#define VKH_ORDER_A {0, 1, 4, 5, 2, 3, 6, 7}
#define VKH_ORDER_B {0, 1, 2, 3}
#define VKH_ORDER_C {0, 1, 2, 3}

// Run the gate and the probe and log the decision: the pipeline builder's open (vkp_open) calls it, after the device and its
// timeline exist and before any module is built, so the game and prepare's pipeline step decide alike.
void vkh_probe(void);
// 1 when the modules' native HMMA runs as translated, 0 when the pipeline builder turns it into the helper.
int vkh_native(void);
// Why, in a few words ("OOT2_VK_HMMA=helper", "the device lists no 16x8x16 f16 subgroup matrix", ...); "" when native.
const char *vkh_reason(void);
