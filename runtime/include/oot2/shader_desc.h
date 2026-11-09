// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the pipeline builder reads of a module's resource description (schema 2, xlate/host/ResultWriter.cs; the text is
// the translator's JSON, stored unchanged in the shader cache pack, oot2/shader_cache.h): the stage, whether the module
// fits the `uni` layout, and its user interface by location, which decides whether a vertex module has to be padded
// for the pixel module it is linked to (the pipeline cost study). Nothing here needs Vulkan.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/bindless.h"

// The stage names of the description ("stage": vs, ps, cs, ts, ms), in the order of the runtime's NVN_STAGE_* kinds.
enum { SHD_VS, SHD_PS, SHD_CS, SHD_TS, SHD_MS, SHD_KINDS };
// The scalar types of an interface variable ("float32", "sint32", "uint32").
enum { SHD_FLOAT, SHD_SINT, SHD_UINT };

#define SHD_MAX_IO 64          // user interface entries a description may list per direction
#define SHD_MAX_LOCATIONS 64   // locations an entry or an array of entries may reach

// One user input or output variable (a Location decoration). An array of length n takes n locations in a row, except
// the outer array of a mesh stage's output: it holds one element per vertex or per primitive, all at the variable's own
// location, so a mesh output takes one location whatever its vertex count (shd_mesh_link).
typedef struct ShdIo {
    int32_t location;
    uint8_t component;     // first component used at the location (0 to 3)
    uint8_t components;    // 1 to 4
    uint8_t type;          // SHD_FLOAT, SHD_SINT or SHD_UINT
    uint8_t bits;          // 32 for every variable the translator declares today
    int32_t array_length;  // the outer array's length; 1 for a variable that is not an array
    uint8_t used;
    int8_t array_depth;    // the array levels of the variable's type (0 for none); -1 when the description does not say
    int8_t per_primitive;  // 1 for a PerPrimitiveEXT variable, 0 for one per vertex; -1 when the description does not say
} ShdIo;

typedef struct ShaderDesc {
    uint32_t schema;
    int kind;              // SHD_*, or -1 when the description names no known stage
    int fits;              // layout.fits: the module's resources fit the stage's slots of the layout
    int raw_pointers;      // raw_access.physical_pointers
    uint32_t ninputs, noutputs;
    ShdIo inputs[SHD_MAX_IO], outputs[SHD_MAX_IO];
} ShaderDesc;

const char *shd_kind_name(int kind);   // "vs", "ps", "cs", "ts", "ms"; "?" for another value

// Parse the description text json[0..n) (not NUL-terminated). 1 with *out filled, or 0 with one line in why. A
// description of another schema than the one the runtime reads, or of no stage, or with more interface entries than
// SHD_MAX_IO, is refused.
int shd_parse(const char *json, size_t n, uint32_t schema, ShaderDesc *out, char *why, size_t cap);

// A vertex output a pixel module reads and the vertex module does not write, as one variable of consecutive
// components of one location.
typedef struct ShdPad {
    uint16_t location;
    uint8_t component, components, type;
} ShdPad;

// The padding a vertex module needs to be linked to a pixel module: every component word of a location the pixel
// module declares as input and the vertex module declares no output for, as runs of consecutive components of one type,
// in location and component order. Returns how many (at most max), or -1 with the reason in why when a variable cannot
// be padded (more locations than SHD_MAX_LOCATIONS, a type that is not 32 bits, more runs than max).
int shd_pad_set(const ShaderDesc *vs, const ShaderDesc *ps, ShdPad *out, int max, char *why, size_t cap);
// The rule a mesh module is linked to a pixel module by. Each mesh output is an array with one element per vertex (or,
// marked per_primitive, per primitive) of a scalar or vector at its one location; each pixel input reads its words at
// the rate it is marked with. Returns the runs of words the pixel module reads and the mesh module does not write, as
// shd_pad_set does (a mesh module is never padded: the caller stops on any), or -1 with the reason in why: an entry
// whose description does not give its array depth or rate (a description older than these members), a shape the
// builder does not link (a mesh output that is not one array level around its element, a pixel input array over
// locations the mesh module writes), a word written at one rate and read at the other ("rate"), a word read as another
// type or width than it is written ("type"), a location past SHD_MAX_LOCATIONS, or more runs than max.
int shd_mesh_link(const ShaderDesc *ms, const ShaderDesc *ps, ShdPad *missing, int max, char *why, size_t cap);
// The canonical text of a pad set ("l:c:n:t;" per run, empty for none), which keys the padded library.
void shd_pad_text(const ShdPad *pad, int n, char *out, size_t cap);

// ---- bindings
//
// Where a module reads its resources, for the replay of recorded NVN2 commands: which constant banks, storage buffers,
// textures, storage images and vertex attributes the module declares, and at which set and binding of the layout
// (xlate/host/UniLayout.cs) each one is. The caps are generous: the stage bindings study counted at most 11 user banks plus c0, c1, c2
// and c24 in a pixel module, 14 storage buffers and 12 textures (one an array of 32) in a compute module, 23 textures in
// a pixel module and 6 images over the corpus.
#define SHD_MAX_CBUFS 32
#define SHD_MAX_SBUFS 32
#define SHD_MAX_TEXS 64
#define SHD_MAX_IMGS 16
#define SHD_MAX_VINS 32

// How a texture reaches the module ("via").
enum { SHD_VIA_PUSH_SLOT, SHD_VIA_POOL, SHD_VIA_DESCRIPTOR };
// What a texture entry is ("kind"): a combined image sampler, a texture without its sampler, or a bare sampler.
enum { SHD_TEX_COMBINED, SHD_TEX_SEPARATE, SHD_TEX_SAMPLER };

// A constant bank: the module's n-th bank, c[slot] of the game's command buffer, at (set, binding).
typedef struct ShdCbuf {
    uint8_t slot;
    uint8_t set;
    uint16_t binding;
} ShdCbuf;

// A guest storage buffer. Its address (a u64) and size (a u32 eight bytes on) are in constant bank cb_slot at
// address_byte_offset (storage_buffer_contract); ordinal is the buffer's number in the module. write: the module may
// write it (the translator's own flag or the post-pass's finding that an instruction does).
typedef struct ShdSbuf {
    uint8_t set, ordinal, cb_slot, write;
    uint16_t binding, address_byte_offset;
} ShdSbuf;

// A texture, or a bare sampler. via PUSH_SLOT: the module reads the handle from push constant word push_slot, at byte
// push_offset (set 0, binding 0 for a texture, 1 for a sampler: the game's pool arrays); POOL: it reads a handle from
// memory at the pool (push_slot and push_offset are -1); DESCRIPTOR: a descriptor of its own at (set, binding).
// c2_slot is the NVN handle table's slot in constant bank 2 the handle is read from (texture 0 to 127, separate
// texture 128 and on, a sampler's own number), -1 when the translator could not tell. view is the BV_* view type the
// type string asks for (multisampled types take the 2D types), BV_NONE for a buffer texture and for a bare sampler.
typedef struct ShdTex {
    int8_t via, kind, view;
    uint8_t set, shadow;
    uint16_t binding, array_length;
    int16_t push_slot, push_offset, c2_slot;
} ShdTex;

// A storage image. write: the module may store to it (the translator saw a store, or the post-pass did not declare it
// NonWritable). view as for a texture.
typedef struct ShdImg {
    uint8_t set, write;
    int8_t view;
    uint16_t binding;
    int16_t c2_slot;
} ShdImg;

// A vertex attribute the module reads: the variable at `location`, from `component`, of `components` scalars of type
// SHD_FLOAT, SHD_SINT or SHD_UINT, 32 bits each. An array takes one entry per location.
typedef struct ShdVin {
    uint8_t location, component, components, type;
} ShdVin;

typedef struct ShaderBinds {
    int kind;                 // SHD_*
    int fits;                 // layout.fits
    uint8_t raw;              // raw_access.page_table or physical_pointers: the module reads the page table
    uint8_t support_read;     // any field of the support buffer is read
    uint32_t ncb, nsb, ntex, nimg, nvin;
    ShdCbuf cb[SHD_MAX_CBUFS];
    ShdSbuf sb[SHD_MAX_SBUFS];
    ShdTex tex[SHD_MAX_TEXS];
    ShdImg img[SHD_MAX_IMGS];
    ShdVin vin[SHD_MAX_VINS];
    uint32_t local_size[3];   // a compute module's workgroup size; 0 for another stage
} ShaderBinds;

// Parse the bindings of description json[0..n) (not NUL-terminated): 1 with *out filled, or 0 with one line in why: a
// schema other than `schema`, a list past its cap, a member missing or of the wrong type or out of range, a via, kind,
// texture type or vertex input type the replay does not know. A compute module needs its local_size and a vertex module
// its vertex_inputs.
int shd_parse_binds(const char *json, size_t n, uint32_t schema, ShaderBinds *out, char *why, size_t cap);
