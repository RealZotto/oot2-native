// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A read-only view of the game's store of named state values (a flag, a counter, a scene name), for the markers of the
// input file (oot2/hid_replay.h). The game keeps all such values in one global store: tables of entries, one table per
// type, found by a 32-bit hash of the value's name, hanging off one object that a global pointer cell leads to. Where the
// cell is, how the tables are laid out and the hash of each name a person asked to wait for are not in this program: the
// prepare step writes them to oot2-store.txt in the game folder, from the player's own copy of the game and a names file
// of the player's own (tools/prepare/store.py has the file's format). This reads that file and, on request, one value from
// guest memory. It never writes guest memory.
//
// A value is read through the chain the description gives: the cell (an offset in the main module's image) holds the address
// of a variable, the variable holds the address of the store's object, the object holds a table pointer, a capacity, an
// entry count and an entry array pointer for each type. The reader takes a table to be open addressing: the probe starts at
// hash mod capacity, steps one cell at a time and wraps, a cell whose key is 0 is empty, and a cell holds the key `key`
// bytes in and the number of the entry `index` bytes in (both are in the description, read by the prepare step from the
// game's own code). A role finds its table by its kind (a bool is read as a byte, an int or uint as a word, a float as
// a single float, a string as a pointer to characters) and by the hash being in that table.
//
// Safe against a description that does not fit the game. The file names the build of the main module it was made from and
// is refused (store_load) when the loaded module is another. Every address read is checked against the guest memory the
// runtime has committed: an address that lies in no guest region is an error (the layout is wrong), one in a region
// that is not committed yet, or a null pointer, means the game has not built the store yet (STORE_WAIT).
#pragma once
#include <stddef.h>
#include <stdint.h>

#define STORE_FILE "oot2-store.txt"

typedef enum StoreKind { STORE_BOOL, STORE_INT, STORE_UINT, STORE_FLOAT, STORE_STRING } StoreKind;

typedef struct StoreValue {
    int64_t i;      // a bool (0 or 1), an int or a uint
    double f;       // a float
    char s[64];     // a string, cut at 63 characters
} StoreValue;

typedef enum StoreRead {
    STORE_OK,        // *out is the value
    STORE_WAIT,      // the game has not built the store, or the table that would hold the value, yet
    STORE_MISSING,   // every table of the kind is built and none has the value
    STORE_BAD        // an address left the guest's memory: the description does not fit; why says how
} StoreRead;

// Read and check the description in `path` against the loaded main module. 1 on success; 0 with one line in why when the
// file is missing, malformed, of another format (a file of format 1, which had no key displacement, is refused by name), or
// made from another build of the main module. A second call replaces the
// first. Call after rt_load_modules.
int store_load(const char *path, char *why, size_t why_size);

// The roles the description holds, in file order: their number, their name and their kind. store_role finds one by name
// (case-insensitive): its number, or -1.
int store_role_count(void);
const char *store_role_name(int role);
StoreKind store_role_kind(int role);
int store_role(const char *name);

// Read the value of a role now. Reads guest memory only, and only the words the chain names. On STORE_BAD why holds one line.
StoreRead store_read(int role, StoreValue *out, char *why, size_t why_size);
