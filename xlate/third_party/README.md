# Third-party code of the shader translator

Code in this folder was written by others and is kept unmodified unless this
file lists a change. Each component keeps its own licence file. The repository
notices ([THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)) carry one row
per component.

## ryujinx/

The shader translation library of Ryujinx 1.3.2, three projects
(`Ryujinx.Graphics.Shader`, `Ryujinx.Common`, `Spv.Generator`), as vendored by
nxnative, which added a decoder for Ampere (sm_86) code and the other changes
listed in [ryujinx/README.md](ryujinx/README.md).

- Source: https://github.com/seedds/nxnative, folder `dm/third_party/ryujinx`.
- Commit: `865cc35fad7034d17c2190f8ce349f61f5bcaba1` (2026-11-06).
- Licence: MIT, [ryujinx/LICENSE.txt](ryujinx/LICENSE.txt); `Spv.Generator`
  also carries its own copy in `src/Spv.Generator/LICENSE`.
- Copied as is at first: every file was byte-identical to the blob at that
  commit. The files listed below have changed since.

### Changes made here

A change to a file in `ryujinx/` gets a line in this list (file, function, what
and why) and an `oot2-native:` comment at the changed code, so that the folder
can be compared with a later nxnative commit and the changes carried over. Paths
are under `ryujinx/src/Ryujinx.Graphics.Shader/`.

Handlers ported from nxnative pull requests not merged at the commit above
(MIT, same author and licence as the vendored tree):

- `Instructions/InstEmitAmpereAlu.cs`, the ALU table and `AF2ip`: F2IP
  (opcode 0x243), the float to 8-bit integer convert and pack, both forms the
  game uses (round to nearest of two registers; truncation with source A `RZ`).
  From pull request 1300, commits `06f47f57` and `e3f8ef8c`. Without it, 26
  programs did not translate.
- `Instructions/InstEmitAmpereMem.cs`, `AIsberd`: ISBERD mode 0xb8 in a task
  shader, the read-back of a payload word the task itself stored (the outputs
  at `stage[2]`, where `AStageOut` writes it). From pull request 1300, commit
  `06f47f57`. Without it, 1 task program did not translate.
- `Instructions/InstEmitAmpereMem.cs`, the memory table and `AAtomgCas`, and
  `Decoders/DecoderAmpere.cs`, the store feature list: ATOMG.CAS (opcode
  0x3a9), 32-bit compare and swap on global memory. From pull request 1275,
  commit `8fd3a0dc` (only the ATOMG.CAS parts of that commit). Without it, 6
  compute programs did not translate.

Changes of this project:

- `Instructions/InstEmitMemory.cs`, `EmitAtomicOp`;
  `Translation/EmitterContextInsts.cs`, `AtomicAddF32`;
  `Translation/Optimizations/GlobalToStorage.cs`, `TryGenerateStorageOp`;
  `CodeGen/Spirv/CodeGenContext.cs`, `FloatStorageViews`;
  `CodeGen/Spirv/Instructions.cs`, `GenerateAtomicAdd`,
  `GenerateAtomicFloatAdd`, `GetFloatStorageView`: the float add reduction
  (`RED.F32.FTZ.RN`, and ATOM of the same type) on global memory, which the tree
  logged ("Invalid reduction type") and dropped, becomes `OpAtomicFAddEXT`
  (`SPV_EXT_shader_atomic_float_add`), through a float view of the storage
  buffer at the same set and binding, or through a physical storage buffer
  pointer for a raw address. The device needs `shaderBufferFloat32AtomicAdd`
  (`VK_EXT_shader_atomic_float`). 55 programs lost an operation without it.
- `StructuredIr/ShaderProperties.cs`, `TextureVariants`;
  `CodeGen/Spirv/CodeGenContext.cs`, `SamplerVariants`;
  `CodeGen/Spirv/Declarations.cs`, `SamplersToDeclare`, `DeclareSamplers`;
  `CodeGen/Spirv/Instructions.cs`, `GetSamplerDeclaration`, `GetSamplerType`:
  a texture pool used with several types (the host binds one pool, set 0
  binding 0, for all of them) is declared once per type, all aliasing that
  binding, and each texture operation uses the variable of its own type. With
  one variable, the last type served every operation, and a 2D sample of a 2D
  array image lost its layer coordinate (3,629 invalid pixel modules).
- `CodeGen/Spirv/Instructions.cs`, `GenerateTextureQuerySize`: a size
  component the declared texture type does not have (component 2 of a 2D
  texture) is the constant 1, and the translator logs the query, instead of an
  extract past the end of the size vector (74 invalid modules). The right value
  comes with the texture's real type (architecture D21).
- `Instructions/AttributeMap.cs`, `GenerateAttributeStore`: a store to
  attribute 0x5C, the variable pixel rate shading table index, is dropped on
  purpose with its own log line, instead of the "not valid" line for unknown
  offsets: the game enables no shading rate, so the index selects nothing.
- `PushTextureLayout.cs` (new); `IGpuAccessor.cs`, `QuerySupportBufferSet`,
  `QueryPushTextureLayout` (default methods: set 0, none);
  `Translation/ResourceManager.cs`, the constructor;
  `StructuredIr/ShaderProperties.cs`, `PushTextures`;
  `CodeGen/Spirv/SamplerDeclaration.cs`, `PushSlot`;
  `CodeGen/Spirv/CodeGenContext.cs`, `PushTexture*`;
  `CodeGen/Spirv/Declarations.cs`, `DeclareSamplers`, `DeclarePushTextures`,
  `DeclareGlobalArray`;
  `CodeGen/Spirv/Instructions.cs`, `GenerateSampledImageLoad`,
  `GeneratePushSlotSampledImageLoad`, `LoadPushSlotSampler`, `LoadPushSlot`,
  `PushSlotIndex`: the descriptor layout `uni` (architecture D13, the stage bindings work).
  The host names the support buffer's set, and a texture the host gives a
  binding in `PushTextureLayout.SlotSet` is read through a push constant slot
  instead of a descriptor of its own: the slot holds an NVN texture handle,
  whose texture ID indexes the global image array (one view per image type,
  sharing the pool's) and whose sampler ID indexes the global sampler array,
  both clamped to their arrays; element i of an array of textures adds i to
  both. Without it, every pair of programs claimed the same set 0 bindings
  for different things in its two stages (spike S2, B3), and 13 percent of
  pairs needed more than the 32 push descriptors of the reference GPU (B4).
- `Translation/Optimizations/GlobalToStorage.cs`, `TryGenerateRawStorageOp`
  and the constants before it: a raw global access (an address traced to no
  constant buffer) is looked up in a two-level page table (one word per 2 MiB
  chunk, then the 4 KiB page's 64-bit delta; architecture D15) instead of
  nxnative's sorted pool table, and a store or an atomic sets its leaf slot's
  store bit instead of inserting its page in a hash set. The table's device
  address and span are the first four push constant words, and the table holds
  the store bits and a miss log that names the address of every miss, a load's
  too (`RawTableAbi.cs`, mirrored by `runtime/include/oot2/raw_table.h`; no
  storage buffer is bound, nxnative's sentinel slot 255 is gone).
  `StorageKind.PushConstant`, `Declarations.DeclarePushConstants` and
  `Instructions.PushWordPointer`: a load of a word of the push constant block
  by constant index, one block shared by the raw words and the texture slots.
  `CodeGen/Spirv/SpirvGenerator.cs`, `Generate` and `CountCallSites`: a raw
  helper (its name ends with `GlobalToStorage.RawHelperSuffix`) called from 64
  sites or more is declared `DontInline`. Most raw-access programs call it 253
  times and the driver inlined every call: the 12-step search made the helper
  893 SPIR-V instructions and such a module about 240,000 once inlined, 7 to
  10 s to build on the reference GPU; the walk is 167 instructions, and kept a
  call the module builds in about 0.5 s. Under 52 sites the call made builds
  slower, so those helpers are left to inline.
- `Translation/Optimizations/GlobalToStorage.cs`, `TryGenerateMultiTargetStorageOp`
  and `HoldsPointer`: a global address that is a phi of a constant buffer
  pointer and an address traced to no buffer (a pointer the shader loaded
  through another buffer) got a helper over the traced buffers alone, which
  gives a load zero at every other address, so the loaded path read zeros. Such
  an access now takes the raw helper, as one through a select does, and a note
  says so ("mixes an address traced to a storage buffer with one traced to
  none"). A constant or undefined phi source holds no pointer (a register one
  path never wrote enters a phi that way) and is left out as before. Lichii
  fixed the same defect in its own tree (pull request 410); the code here is
  this project's. On the 56,381 programs of the corpus 132 modules change (121
  mesh and 11 compute, 72 of them in the set the updated game loads), and
  nothing else does.
- `StructuredIr/AstOptimizer.cs`, `Optimize`, `Classify`, `CanMove`, `Mentions`:
  with the debug mode off, Ryujinx moves every expression with one use to that
  use and removes an assignment nobody reads. Through this translator, on the
  56,381 programs of the corpus, 9,897 (17.6 percent) fail to translate and
  of the 46,484 that do, 114 lose an atomic or a subgroup operation, 39 have a
  read moved past a store and 738 have a shuffle, a vote or a sample moved
  into a branch: an atomic or a raw-access call whose result is discarded
  disappears, a read of shared, storage or device memory is moved past a store
  that came between, a shuffle or a sample is moved into a branch, and an
  expression is folded into a call's argument (the SPIR-V generator casts each
  argument to a variable, so the translation fails). The optimizer now sorts an
  expression into pinned (an atomic, a call, a store to memory, a barrier, an
  image write), reading (memory a store can change) and convergent (a shuffle,
  a vote, a derivative, a sample). A pinned one stays a statement and is never
  removed; a reading or convergent one with a single mention moves only to the
  next statement of its own block that uses it, over plain statements only
  (none pinned when it reads), and never into a loop's condition or a call. The
  host's job option `debug` now defaults to false, so this rule is the form
  every job gets; `debug: true` asks for the old form (nothing propagated or
  dropped), which this file does not touch. The rule is
  this project's code. Two public translators carry the same defects and their
  fixes are evidence only, neither copied: TotK Recomp's own patch keeping
  side-effect assignments, and one of Jaigner's keeping memory reads in place
  (MIT; its patch carries no licence header and quotes a game program's
  identifier). Hookshot's shader path is a C++ recompiler without this
  optimizer, under no stated licence.
- `CodeGen/Spirv/Instructions.cs`, `MarkNonUniform`: the first non-uniform
  access no longer declares `StorageImageArrayNonUniformIndexing` next to
  `SampledImageArrayNonUniformIndexing`. The generator marks every indexed
  access non-uniform and no module of the corpus indexes an array of storage
  images, so the declaration only asked the device for a feature nothing used
  (9,533 of the 55,527 modules the corpus split translates, and 22 of its 854
  mesh programs re-run with a synthesized interface; the only difference in
  any module is that one declaration, 8 bytes). The host's post-pass
  (`xlate/host/SpirvFixup.cs`) declares it where a storage image array is
  indexed with a non-uniform index.
- `Translation/IrDump.cs` (new), with calls in `Translation/TranslatorContext.cs`,
  `Translation/Optimizations/Optimizer.cs`,
  `Translation/Transforms/TransformPasses.cs`, `StructuredIr/StructuredProgram.cs`
  and the host (`xlate/host/Translation.cs`): a text dump of the IR of one
  program, written only when the environment variable `OOT2_XLATE_IR_DUMP`
  names a folder (off by default; nothing is written and no translation
  changes). One file per stage and function: after the emitters, after the
  SSA rename, after each optimizer step and each sweep of its cleanup loop,
  after each transform pass, before the structured form, and the structured
  tree before and after the AST optimizer, with a `counts.tsv` of the shared
  memory, barrier, shuffle, HMMA and call operations at each stage, which
  shows the stage where a kind of operation goes missing.
- `Translation/Optimizations/Optimizer.cs`, `RemoveNode`: when a removed
  node was the last reader of a result, the cleanup went on into the node that
  made the result and dropped that node's uses of its own sources, so that the
  next sweep could delete their producers. It did so for any producer without
  an effect, also one with several results of which another is still read: an
  operation that stays in the block then lost its claim on its sources. It now
  goes on only into a producer whose every result is unused (`IsUnused`). The
  matrix multiply-add of HMMA has two results, and a program that read one of
  them kept its last HMMA while the earlier HMMA and the shared memory loads
  of LDS and LDSM that fed it were deleted, the kept one reading values nothing
  defined. On the 4,125 compute programs of the corpus split 8 modules change:
  the 5 that lose HMMA (one lost 45 of its 46), which now have as many helper
  calls as the program has HMMA, and 3 without HMMA that get back arithmetic
  fed by one result of an image load.
- `CodeGen/Spirv/Instructions.cs`, `GenerateMaximum`, `GenerateMinimum`,
  `GenerateClamp`; `CodeGen/Spirv/SpirvDelegates.cs`, `GlslNMax`, `GlslNMin`,
  `GlslNClamp`; `Translation/Optimizations/ConstantFolding.cs`, the float clamp,
  `NMax`, `NMin`; `Instructions/InstEmitConversion.cs`, `EmitF2I`;
  `Instructions/InstEmitAmpereAlu.cs`, `AF2i64`, `AHalfResult`: the console's
  NaN rules. A float maximum, minimum or clamp (FMNMX, HMNMX2, the saturation
  of FADD, FMUL, FFMA, F2F, HADD2 and the other arithmetic, the pairs F2IP
  builds) was emitted as `FMax`, `FMin` and `FClamp`, which Vulkan leaves to the
  driver when an operand is NaN: NVIDIA's returns the other operand, AMD's can
  return the NaN. The console's rule is "the other operand" (the NaN only when
  both are NaN), and a saturated NaN is 0, so they are now `NMax`, `NMin` and
  `NClamp` (`GLSL.std.450`, no capability), and the constant folder follows the
  same rule instead of .NET's NaN-propagating `Math.Clamp`. F2I of a NaN is 0
  for an integer of 32 bits or less (`OpConvertFToS` and `OpConvertFToU` leave
  it undefined): `EmitF2I` gives 0 for a NaN source, by a select after the
  conversion when the count below was made and by the bounded sequence of the
  entry on `FP32ToIntSaturated` since; the 64-bit form takes
  the integer's top bit, as NVIDIA's documentation gives. The `.RELU` of HFMA2
  and HMUL2 keeps a NaN a NaN, which `NMax` against 0 would turn into 0. Over
  the 56,381 programs of the corpus, 50,926 modules change and nothing else:
  8,041 only by the instruction number of the minimum, maximum or clamp, 42,885
  also by a select around each float to integer conversion; all of them pass
  spirv-val and none of their descriptions changes but the module's hash and
  size.
- `Instructions/InstEmitAmpereAlu.cs`, `AFmnmx`: a FMNMX word with one of bits
  74 to 79 or 81 to 86 is refused by name. Those bits select modes that are not
  translated (the minimum and maximum that give a NaN for a NaN operand among
  them, at bit 81 by the reading of other lowerers, which no word seen
  confirms), and the word was translated as the plain minimum or maximum. Bit
  80, `.FTZ`, is on every FMNMX word of the corpus and stays dropped; none of
  those words has a refused bit.
- `Instructions/InstEmitConversion.cs`, `FP32ToS32Bounded` (new, beside
  `EmitF2I`); `Instructions/InstEmitAmpereMem.cs`, `ATmml`;
  `Instructions/InstEmitAmpereAlu.cs`, `AF2i64`;
  `Translation/HelperFunctionManager.cs`, `GenerateTexelFetchScaleFunction`,
  `GenerateTextureSizeUnscaleFunction`; `Translation/Transforms/TexturePass.cs`,
  `InsertTextureLod`: the float to integer conversions around F2I bound their
  float before they convert it. `OpConvertFToS` and `OpConvertFToU` are
  undefined for a NaN, an infinity and a value outside the integer's range, and
  these floats are a driver's level of detail query, a product or quotient with
  the host's render scale, or the program's own operand. Each is clamped
  (`NClamp`) between floats that are inside the range, -2^31 or 0 below and
  2147483520, the largest float below 2^31, above; a NaN gives 0, by the
  clamp's lower bound where the value cannot be negative (a size, a level) and
  by a select otherwise. The 64-bit conversion clamps from 0 to the largest
  float below 2^64 and saturates as the console's does: 0 for a negative value,
  all ones from 2^64 up, the top bit for a NaN. A value in range converts as
  before. F2I's own sequence (`EmitF2I`) is not part of this change; it is the
  next entry.
- `Instructions/InstEmitConversion.cs`, `EmitF2I` and `FP32ToIntSaturated`
  (new); `Instructions/InstEmitAmpereAlu.cs`, `AF2i`: F2I of a single or of a
  half bounds its float before it converts it. Its select of 0 for a NaN came
  after the conversion, and a select does not take back a conversion that was
  undefined: a NaN, an infinity or a value outside the integer's range still
  went into `OpConvertFToS` or `OpConvertFToU`. The console saturates such a
  value to the integer's range and gives 0 for a NaN. The rounded float is now
  clamped (`NClamp`) between two floats the integer holds and then converted:
  -2^31 and 2147483520 for a signed 32-bit integer, 0 and 4294967040 for an
  unsigned one (the largest floats below 2^31 and 2^32, since 2^31 - 1 and
  2^32 - 1 are not floats), and the integer's own bounds for 16 and 8 bits,
  which takes the place of the integer clamp those had after the conversion.
  A 32-bit result then takes its top value by a select where the source is
  above the clamp's upper bound, which is where it is 2^31 or 2^32 and up; a
  signed result takes 0 by a select where the source is NaN, and an unsigned
  one has its 0 from the clamp's lower bound, with no NaN test. There is no
  branch, and a value in range converts to what it did. The double source,
  which no instruction of the console reaches, is as it was. Rounding toward
  zero no longer emits `Trunc`: the conversion rounds that way itself and the
  clamp's bounds are whole numbers, so the clamped float truncates to what
  the truncated float clamps to. On synthetic modules that convert a computed
  value, the function has 4 instructions more for a signed 32-bit integer (a
  load of the source, the clamp, a comparison, a select), as many as before
  for an unsigned 32-bit and for a signed 16-bit one, 1 fewer for a signed
  8-bit one, 9 fewer for an unsigned 16-bit one and 10 fewer for an unsigned
  8-bit one; toward zero each has 1 fewer than that, so 3 more for a signed
  32-bit integer and 1 fewer than before for an unsigned one. A signed 32-bit
  site also has one more `NoContraction` decoration, on the clamp. Over the
  56,381 programs of the corpus, which all translate with and without this
  change, 42,885 modules change, the 42,885 that hold a F2I word, and nothing
  else; all of them pass spirv-val and none of their descriptions changes but
  the module's hash and size. 38,533 grow, 4,322 shrink and 30 keep their
  size; together they have 0.22 percent more words. The corpus converts to a
  signed or unsigned 32-bit integer and to an unsigned 16-bit one, from a
  single; the other forms (a signed 16-bit integer, either 8-bit one, a half
  source) have the synthetic tests only. Four pixel programs of the corpus
  convert the constant 0x7FFFFFFF, a NaN, to an unsigned 32-bit integer and
  give the result to an atomic minimum and maximum: the clamp folds to 0
  there, where the module held `OpConvertFToU` of the NaN before. What the
  sequence gives for a NaN or an
  infinity, here and in the conversions of the entry above, is what SPIR-V
  says, and Vulkan does not hold a driver to it: the modules do not declare
  `SignedZeroInfNanPreserve`, and without it a driver may assume that no
  operand is a NaN or an infinity, for `NClamp`, for `OpIsNan` and for the
  comparison alike (only what moves bits, `OpSelect` among it, is outside
  that); with it declared the assumption still stands for `NClamp`. No device
  has been measured.
- `CodeGen/Spirv/CoopHmma.cs` (new), `CodeGen/Spirv/SpirvGenerator.cs`,
  `Generate` and `GenerateCoopHmma`, `CodeGen/Spirv/CodeGenContext.cs`,
  `GetArgumentPointer`; `Translation/HelperFunctionManager.cs`,
  `GenerateHmma16816F16Coop`; `Translation/HelperFunctionName.cs`;
  `Translation/Transforms/TensorPass.cs`, `RunPass`; `IGpuAccessor.cs`,
  `QueryHostCooperativeHmma` (default false); `Spv.Generator/CoopMatrixGrammar.cs`
  (new, under `ryujinx/src/`): HMMA.16816 with an f16 accumulator in a SPIR-V
  compute program calls `dmsh_hmma16816_coop`, which moves each lane's fragment
  registers into `VK_KHR_cooperative_matrix` matrices in the device's component
  order, multiplies with one `OpCooperativeMatrixMulAddKHR` and moves D back out,
  instead of the scratch-memory helper `dmsh_hmma16816_f16`, which stays in the
  module uncalled with the same arguments: the runtime turns every call of the
  one into a call of the other, by name, on a device whose start-up probe refuses
  the native form. A module with the native function takes the Vulkan memory
  model, which the extension requires of a Shader module; its workgroup loads and
  stores get `NonPrivatePointer` and `MakePointerVisible` or
  `MakePointerAvailable` at workgroup scope, as glslang writes a `shared`
  variable, and nothing else of the module changes. A module with a Coherent
  image keeps the helper (its native function is a call of the helper), and so
  does a program whose workgroup width is not a multiple of 32, since a subgroup
  matrix needs every lane of its subgroup. The host's job option `hmma`
  chooses: `coop` (the default) or `helper`, the form every module had before. On the movie's ten tensor programs every module
  passes spirv-val (Vulkan 1.3), and the driver's code for the largest falls
  from 1,621,632 to 221,568 bytes.
