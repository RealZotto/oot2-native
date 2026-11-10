# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The font step: the font the runtime serves for the console's shared font calls (nn::pl, runtime/hos/ov_system.c).

The console's own fonts are not part of this program, are never shipped and are never read from a dump. The runtime
serves one open-licence font file for every shared font type, from fonts/shared-font.ttf in the game folder. This step
places that file. No font is in this repository: the default is fetched on the player's machine at a pinned commit,
checked, and written beside the licence text it is distributed under.

    BIZ UDPGothic Regular   https://github.com/googlefonts/morisawa-biz-ud-gothic
    Copyright 2022 The BIZ UDGothic Project Authors, SIL Open Font License 1.1 (OFL.txt of that repository)
    commit                  0d3e2d57dbb3cf7480aefaa7fa530dc9978667d1

It covers Latin and Japanese, which the game's two text languages need. The file is placed unchanged, under the name
the runtime reads.

Where the files come from, in this order:

    --shared-font FILE   a TrueType or OpenType font of the player's own choice, placed as it is. Its licence is the
                         player's matter, and no licence text is written. The runtime's own limits are checked here: a
                         plain sfnt (by its first four bytes) of at most 16 MiB.
    fetched/font/        beside tools/, when it holds the two pinned files (BIZUDPGothic-Regular.ttf and OFL.txt) with
                         the pinned hashes: a machine without network access gets them there by hand.
    the network          the two files at the pinned commit, each checked by size and SHA-256 before anything is written.

The output is DEST/fonts/shared-font.ttf and, for the pinned font, DEST/fonts/shared-font-OFL.txt.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import urllib.error
import urllib.request
from pathlib import Path

from .runner import Context, PrepareError, Step, file_sha256, fingerprint

COMMIT = "0d3e2d57dbb3cf7480aefaa7fa530dc9978667d1"
BASE = f"https://raw.githubusercontent.com/googlefonts/morisawa-biz-ud-gothic/{COMMIT}/"
NAME = "BIZ UDPGothic Regular"
# kit file name: (path under BASE, size, SHA-256, name in DEST/fonts)
PIN = {
    "BIZUDPGothic-Regular.ttf": ("fonts/ttf/BIZUDPGothic-Regular.ttf", 4669688,
                                 "258d7156c165f2ff774b6efee637c22c3b950de0d8a10e501137061bc8085d01", "shared-font.ttf"),
    "OFL.txt": ("OFL.txt", 4408, "e753d7155d53c747d037a445e584c8ecfca6dd79846db610417e282a736b28bc",
                "shared-font-OFL.txt"),
}
FONT_FILE, LICENCE_FILE = "shared-font.ttf", "shared-font-OFL.txt"
KIT = Path(__file__).resolve().parents[2] / "fetched" / "font"
MAX_FONT = 16 << 20                      # FONT_MAX_SIZE of runtime/hos/ov_system.c
SFNT = (b"\0\1\0\0", b"OTTO", b"true", b"ttcf")   # font_is_sfnt of runtime/hos/ov_system.c
DOWNLOAD_TIMEOUT = 60


def is_sfnt(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] in SFNT


def download(url: str, limit: int) -> bytes:
    """The bytes at url, at most limit of them plus one (the caller refuses a longer answer by its size)."""
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r:
        return r.read(limit + 1)


def kit_files(kit: Path, pin: dict) -> dict[str, bytes] | None:
    """The pinned files of a kit folder by kit name, or None when one is missing or is not the pinned file."""
    out: dict[str, bytes] = {}
    for name, (_, size, sha, _) in pin.items():
        path = kit / name
        try:
            if path.stat().st_size != size:
                return None
            data = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != sha:
            return None
        out[name] = data
    return out


def fetch(pin: dict, base: str, kit: Path, get=None) -> dict[str, bytes]:
    """The pinned files from the network, each checked by size and hash. Nothing is written here."""
    get = get or download
    out: dict[str, bytes] = {}
    for name, (path, size, sha, _) in pin.items():
        url = base + path
        try:
            data = get(url, size)
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            raise PrepareError(
                f"the shared font could not be downloaded from {url} ({getattr(e, 'reason', None) or e!r}). The game asks the "
                f"console for a font and this program ships none. Download {', '.join(base + p[0] for p in pin.values())} "
                f"yourself into {kit} (as {', '.join(pin)}), or give an open-licence TrueType or OpenType font of your "
                "own with --shared-font FILE.") from None
        got = hashlib.sha256(data).hexdigest()
        if len(data) != size or got != sha:
            raise PrepareError(
                f"the file at {url} is not the pinned one: {len(data):,} bytes with SHA-256 {got}, the tools expect "
                f"{size:,} bytes with {sha}. Nothing was placed. Give an open-licence TrueType or OpenType font of "
                "your own with --shared-font FILE.")
        out[name] = data
    return out


def own_font(path: Path) -> bytes:
    """The player's own font file, refused by name when the runtime would refuse it."""
    try:
        size = path.stat().st_size
        if size > MAX_FONT:
            raise PrepareError(f"the font {path} has {size:,} bytes; the runtime serves at most {MAX_FONT >> 20} MiB")
        data = path.read_bytes()
    except OSError as e:
        raise PrepareError(f"the font {path} cannot be read ({e.strerror or e})") from None
    if not is_sfnt(data):
        raise PrepareError(f"the font {path} is not a TrueType or OpenType font (it has {len(data):,} bytes and starts "
                           f"with {data[:4].hex() or 'nothing'}); the game's font loader takes no other form")
    return data


def write(path: Path, data: bytes) -> None:
    """Write through a temporary name, so a run stopped halfway leaves no short file under the real one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class Font(Step):
    id = "font"
    pin = PIN
    base = BASE
    kit = KIT
    get = None   # tests: a function in place of download

    def named(self, ctx: Context) -> Path | None:
        return getattr(ctx.options, "shared_font", None)

    def key(self, ctx: Context) -> str:
        named = self.named(ctx)
        pin = sorted((n, *v) for n, v in self.pin.items())
        return fingerprint("font", Path(__file__), named if named else f"pin {COMMIT} {pin}")

    def outputs(self, record: dict) -> dict[str, tuple[int, str]]:
        return {name: (int(size), sha) for name, (size, sha) in record.get("files", {}).items()}

    def current(self, ctx: Context, record: dict) -> bool:
        outs = self.outputs(record)
        if FONT_FILE not in outs:
            return False
        try:
            return all((ctx.dest / "fonts" / n).stat().st_size == size for n, (size, _) in outs.items())
        except OSError:
            return False

    def verify(self, ctx: Context, record: dict) -> bool:
        try:
            return all(file_sha256(ctx.dest / "fonts" / n) == sha for n, (_, sha) in self.outputs(record).items())
        except OSError:
            return False

    def run(self, ctx: Context) -> dict:
        rep, folder = ctx.reporter, ctx.dest / "fonts"
        named = self.named(ctx)
        if named:
            placed = {FONT_FILE: own_font(Path(named))}
            source = str(named)
            rep.say(f"font: your own font {named}, placed as it is; its licence is yours to keep")
        else:
            files = kit_files(self.kit, self.pin)
            if files is not None:
                source = str(self.kit)
                rep.say(f"font: {NAME} from {self.kit}")
            else:
                source = self.base
                rep.say(f"font: downloading {NAME} ({COMMIT[:7]}, SIL Open Font License 1.1) from {self.base}")
                files = fetch(self.pin, self.base, self.kit, self.get)
            placed = {self.pin[name][3]: data for name, data in files.items()}
        for name in (FONT_FILE, LICENCE_FILE):
            if name in placed:
                write(folder / name, placed[name])
            else:
                (folder / name).unlink(missing_ok=True)   # a licence text of another font must not stay beside this one
        rep.say(f"font: {folder / FONT_FILE}, {len(placed[FONT_FILE]):,} bytes"
                + (f", with its licence in {LICENCE_FILE}" if LICENCE_FILE in placed else ""))
        return {"source": source, "bytes": sum(len(d) for d in placed.values()),
                "files": {n: [len(d), hashlib.sha256(d).hexdigest()] for n, d in placed.items()}}
