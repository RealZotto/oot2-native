# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Where the census reads the game's files from: the RomFS index of a prepared folder and the image it points into.

The index (tools/prepare/extract.py) lists every RomFS path with the offset and size of its bytes in the image and the
SHA-256 of those bytes. The image is the player's .nup when prepare kept the RomFS in place, or romfs/romfs.bin when it
copied it. A file is read by seeking to its extent, so a census touches the few hundred files it needs and never the
whole 25 GiB. Every file read is checked against the SHA-256 its row gives, so an image that is not the one the index
was made from stops the census at the first file, by name.

A file stored as one zstd frame is expanded by magic, never by name; a frame that names a dictionary gets it from the
game's dictionary pack the way tools/shaders/split.py does. Python 3.14's compression.zstd is needed only when such a
frame is met.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from prepare import state as statefile
from prepare.extract import INDEX, read_index

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


class CensusError(Exception):
    """The census cannot go on; the message names the file or the setting at fault."""


def locate_image(game: Path, override: Path | None = None) -> Path:
    """The image the index of a prepared folder points into."""
    if override is not None:
        return Path(override)
    copy = Path(game) / "romfs" / "romfs.bin"
    if copy.is_file():
        return copy
    try:
        state = statefile.load(Path(game))
    except statefile.StateError as e:
        raise CensusError(str(e)) from None
    record = ((state or {}).get("steps") or {}).get("extract") or {}
    nup = record.get("nup")
    if isinstance(nup, dict) and isinstance(nup.get("path"), str):
        return Path(nup["path"])
    raise CensusError(f"{game} holds neither romfs/romfs.bin nor a record of the dump the RomFS is read from in place; "
                      "run prepare first, or name the image with --image")


class Source:
    """The files of a RomFS, by path."""

    def __init__(self, index: Path, image: Path):
        self.index_path, self.image_path = Path(index), Path(image)
        try:
            self.header, rows = read_index(self.index_path)
        except OSError as e:
            raise CensusError(f"the RomFS index {self.index_path} cannot be read: {e.strerror or e}") from None
        except ValueError as e:
            raise CensusError(f"the RomFS index {self.index_path} is unusable: {e}") from None
        self.rows = {path: (off, size, sha) for path, off, size, sha in rows}
        self.paths = sorted(self.rows)
        try:
            self._image = open(self.image_path, "rb")
            self.image_size = self.image_path.stat().st_size
        except OSError as e:
            raise CensusError(f"the image {self.image_path} cannot be opened: {e.strerror or e}") from None
        self._dec = None

    def close(self) -> None:
        self._image.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def has(self, path: str) -> bool:
        return path in self.rows

    def size(self, path: str) -> int:
        return self.rows[path][1]

    def read(self, path: str) -> bytes:
        """The stored bytes of one file, checked against the index."""
        row = self.rows.get(path)
        if row is None:
            raise CensusError(f"{path} is not listed in the RomFS index")
        off, size, sha = row
        if off + size > self.image_size:
            raise CensusError(f"{path} cannot be read: the index puts it at {off}+{size} bytes but the image "
                              f"{self.image_path.name} ends at {self.image_size}")
        try:
            self._image.seek(off)
            data = self._image.read(size)
        except OSError as e:
            raise CensusError(f"{path} cannot be read from the image: {e.strerror or e}") from None
        if hashlib.sha256(data).digest() != sha:
            raise CensusError(f"{path} does not match the SHA-256 the index holds for it: the image is not the one "
                              "the index was made from")
        return data

    def data(self, path: str) -> bytes:
        """The file's contents with a zstd frame expanded."""
        raw = self.read(path)
        if raw[:4] != ZSTD_MAGIC:
            return raw
        try:
            if self._dec is None:
                from shaders.split import Decompressor
                self._dec = Decompressor(self)
            return bytes(self._dec.decompress(raw))
        except CensusError as e:
            raise CensusError(f"{path} is a zstd frame and cannot be expanded: {e}") from None
        except Exception as e:                       # InputError, ContainerError, ZstdError, ImportError
            raise CensusError(f"{path} is a zstd frame and cannot be expanded: {e}") from None


def open_game(game: Path, image: Path | None = None) -> Source:
    index = Path(game) / "romfs" / INDEX
    if not index.is_file():
        raise CensusError(f"{index} does not exist: prepare has not extracted the RomFS index there")
    return Source(index, locate_image(game, image))
