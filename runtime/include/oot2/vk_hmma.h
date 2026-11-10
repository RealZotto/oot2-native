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
//              16-bit floats (the optional feature shader_float16), the Vulkan memory model (an item) and full subgroups in
//              compute (an item: the pipeline of a module that keeps the native function is created with the
//              full-subgroups requirement, which Vulkan asks of a SPIR-V 1.3 module with cooperative matrices, and the
//              feature computeFullSubgroups it needs);
//   the order  a probe loads row-major matrices whose components are their positions and reads back every lane's
//              components: each console element of every lane must sit in the component the translator's order names;
//   the result the same lanes' fragments of small integers, whose every sum f16 holds exactly whatever the order or the
//              width of the additions, multiplied by the native form and by the helper's algorithm must give the same D,
//              bit for bit, and the exact sums; and each lane must hold 8, 4 and 4 components of A, B and C.
//
// The third check proves the wiring end to end on the device (each lane's D comes from the positions the order names,
// through the moves in and out) and nothing about rounding: on general inputs the two forms differ, the native one rounding
// as the tensor cores do.
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

// Tests only: a fault written into the probe's read-back before the third check judges it (the counter-run of that check):
// VKH_FAULT_LENGTHS says a lane holds 7 components of A, VKH_FAULT_HELPER changes one half of the helper's D, and
// VKH_FAULT_SUMS changes that half of both forms' D alike, so that they agree and miss the exact sum. 0: none.
enum { VKH_FAULT_LENGTHS = 1, VKH_FAULT_HELPER, VKH_FAULT_SUMS };
extern int vkh_fault;
