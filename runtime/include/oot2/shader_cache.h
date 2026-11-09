// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The shader cache pack (architecture D21): every translated module with its resource description, one file per
// role, mapped read-only and never read whole. tools/shaders/cache_pack.py writes it and owns the layout (its
// docstring gives every field); runtime/gpu/shader_cache.c reads it. Open checks the header and the index once; a
// lookup is a binary search over the index that reads no entry but the one it returns. A pack is never written while
// it is open, so lookups need no lock.
//
// Two packs, one layout: the base pack holds every job translated with the default answers (answers digest zero), and
// the variants pack holds jobs translated again with projected ledger answers. The variant map connects a stage key
// and the full ledger-row digest to a projected digest or the base row. A missing map entry safely takes the base row.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define SHC_MAGIC "OOT2SHC1"
#define SHC_VERSION 1u
#define SHC_HEADER_BYTES 128u
#define SHC_ROW_BYTES 128u
#define SHC_KEY_BYTES 32u        // 1 to 31 key characters, padded with NUL
#define SHC_DIGEST_BYTES 32u
#define SHC_ENTRY_ALIGN 16u
#define SHC_DESC_SCHEMA 2u       // the description schema this runtime reads: xlate/host/ResultWriter.cs Schema
#define SHC_MAP_MAGIC "OOT2VM1"
#define SHC_MAP_VERSION 1u
#define SHC_MAP_HEADER_BYTES 96u
#define SHC_MAP_ROW_BYTES 96u

enum { SHC_ROLE_BASE = 1, SHC_ROLE_VARIANTS = 2 };
enum { SHC_TRANSLATED = 0, SHC_FAILED = 1 };   // a row's status; a failed row holds the reason, no module

// Why a pack was refused. shc_error_name gives the name cache_pack.py uses for the same refusal.
typedef enum ShcError {
    SHC_OK = 0,
    SHC_E_OPEN,              // the file cannot be opened or mapped
    SHC_E_SHORT,             // shorter than a header
    SHC_E_MAGIC,
    SHC_E_VERSION,
    SHC_E_HEADER,            // sizes, offsets or role that format version 1 does not give
    SHC_E_SCHEMA,            // descriptions of another schema than SHC_DESC_SCHEMA (or the one asked for)
    SHC_E_ROLE,              // a base pack where a variants pack was expected, or the reverse
    SHC_E_TRANSLATOR,        // a variants pack written by another translator than the base pack
    SHC_E_MAP,               // a malformed or inconsistent ledger-to-variant map
    SHC_E_INDEX_TRUNCATED,   // the index ends past the end of the file
    SHC_E_INDEX_HASH,        // the index does not match its SHA-256
    SHC_E_ROW,               // a malformed row: key, digest for the role, status, sizes, alignment
    SHC_E_ORDER,             // rows not strictly ascending by key and digest
    SHC_E_ENTRY_PAST_END,    // an entry that ends past the end of the file
} ShcError;

typedef struct ShcPack {
    const uint8_t *map;          // the whole file, read-only; NULL when closed
    uint64_t size;
    uint64_t count;              // rows
    uint32_t schema, role;
    const uint8_t *translator;   // SHA-256 of the translator executable, 32 bytes inside the header
    int owns_map;                // shc_open mapped the file, so shc_close unmaps it
} ShcPack;

typedef struct ShcEntry {
    const char *key;               // NUL-terminated, inside the index
    const uint8_t *answers;        // 32 bytes; all zero for the default answers
    uint32_t status;               // SHC_TRANSLATED or SHC_FAILED
    const uint32_t *module;        // the SPIR-V words; NULL in a failure record
    uint32_t module_bytes;
    const char *description;       // the resource description (JSON), or the failure reason; not NUL-terminated
    uint32_t description_bytes;
    const uint8_t *sha256;         // of the module and the description together, 32 bytes
} ShcEntry;

const char *shc_error_name(ShcError e);

// Map path and check it as a pack of the given role holding descriptions of the given schema. On a refusal the pack
// stays closed and why holds one line naming the file and the reason.
ShcError shc_open(ShcPack *p, const char *path, uint32_t role, uint32_t schema, char *why, size_t cap);
// The same checks on a view the caller mapped and keeps (tests, and a caller that maps its own way); name is for
// the messages.
ShcError shc_open_view(ShcPack *p, const void *map, uint64_t size, const char *name, uint32_t role, uint32_t schema,
                       char *why, size_t cap);
void shc_close(ShcPack *p);

// The row of (key, answers); answers NULL means the default answers. 1 when found.
int shc_find(const ShcPack *p, const char *key, const uint8_t *answers, ShcEntry *out);
// 1 when an entry's bytes match their SHA-256 (reads the whole entry; lookups never call it).
int shc_entry_intact(const ShcEntry *e);

typedef struct ShcSet {
    ShcPack base, variants;
    int has_variants;
    const uint8_t *variant_map;
    uint64_t variant_map_size, variant_map_count;
    int owns_variant_map, has_variant_map;
} ShcSet;
enum { SHC_MISS = 0, SHC_FROM_BASE = 1, SHC_FROM_VARIANTS = 2 };

// Open the base pack and, when variants is not NULL, the variants pack; both of the given schema and from one
// translator. A variants pack is named only once the ledger step wrote it: a name that cannot be opened is refused.
ShcError shc_set_open(ShcSet *s, const char *base, const char *variants, uint32_t schema, char *why, size_t cap);
// Open the same packs and optionally the prepare step's full-ledger-digest map.
ShcError shc_set_open_map(ShcSet *s, const char *base, const char *variants, const char *map, uint32_t schema,
                          char *why, size_t cap);
// The variants pack's row of (key, answers) when answers is not zero and the row exists, else the base pack's row of
// key. Returns where the entry came from, or SHC_MISS.
int shc_set_find(const ShcSet *s, const char *key, const uint8_t *answers, ShcEntry *out);
// Resolve a full ledger-row digest through the map, returning its projected digest or zero for base. On a missing
// mapping, returns 0 and also writes zero to target.
int shc_set_resolve_ledger(const ShcSet *s, const char *key, const uint8_t *ledger_digest, uint8_t target[32]);
// Resolve a full ledger-row digest and return the matching variant or base pack entry. `mapped` reports whether a
// map row was found, including rows that explicitly select base.
int shc_set_find_ledger(const ShcSet *s, const char *key, const uint8_t *ledger_digest, ShcEntry *out, int *mapped);
void shc_set_close(ShcSet *s);
