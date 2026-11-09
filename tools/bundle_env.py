# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Where the release bundle keeps its tools, for the steps of `oot2 prepare` that look for one (architecture D24c).

A release bundle (tools/bundle.py builds it) is one folder with `bundle.json` at its root, `tools/` (this file's folder)
inside it, and beside `tools/` the pieces a player's machine has no other way to get: an LLVM-MinGW subset, Python,
Ninja, the shader translator, the prebuilt runtime library and the Vulkan headers. Run from a git checkout there is no
`bundle.json` one level above `tools/`, every function here answers None, and prepare finds its tools as it always has
(an option, an environment variable, PATH).

Order of choice, for the compiler and the translator: the option, then the environment variable that names it, then
the bundle, then PATH. A bundle's own tool comes before whatever is on PATH, so that a player's other clang, Ninja or
Python never changes what a bundle builds. The launchers put the bundle's folders first on PATH as well, for the
programs the tools start.

Nothing here reads the network or the game.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
MANIFEST = "bundle.json"
EXE = ".exe" if os.name == "nt" else ""

# Where each piece sits inside a bundle, relative to its root (tools/bundle.py writes this layout).
LLVM_BIN = "llvm/bin"
CLANG = f"llvm/bin/clang{EXE}"
NINJA = f"bin/ninja{EXE}"
XLATE = f"bin/oot2-xlate{EXE}"
PIPEBUILD = f"bin/pipebuild{EXE}"      # the pipelines step's builder, made by tools/bundle.py with the bundle's compiler
VC_RUNTIME = "bin"                     # Microsoft's Visual C++ runtime DLLs sit beside the tools; the launchers put bin first on PATH
PYTHON_HOME = "python"
RUNTIME_LIB = "runtime/liboot2rt.a"
DAV1D_KIT = "runtime/dav1d"
OPUS_KIT = "runtime/opus"
SDL3_KIT = "runtime/sdl3"
VULKAN_SDK = "vulkan"
ZBIC_ARCHIVE = "third_party/zbic-source.tar.gz"


def root() -> Path | None:
    """The bundle's root folder when this code runs from inside one (OOT2_BUNDLE_ROOT overrides, for tests)."""
    named = os.environ.get("OOT2_BUNDLE_ROOT")
    base = Path(named) if named else TOOLS.parent
    return base.resolve() if (base / MANIFEST).is_file() else None


def manifest() -> dict | None:
    base = root()
    if base is None:
        return None
    try:
        return json.loads((base / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def file(relative: str) -> Path | None:
    """A file of the bundle by its path inside it, or None when this is no bundle or the file is not there."""
    base = root()
    if base is None:
        return None
    path = base / relative
    return path if path.is_file() else None


def folder(relative: str) -> Path | None:
    base = root()
    if base is None:
        return None
    path = base / relative
    return path if path.is_dir() else None


def compiler() -> str | None:
    found = file(CLANG)
    return str(found) if found else None


def ninja() -> str | None:
    found = file(NINJA)
    return str(found) if found else None


def translator() -> Path | None:
    return file(XLATE)


def pipebuild() -> Path | None:
    return file(PIPEBUILD)


def runtime_lib() -> Path | None:
    return file(RUNTIME_LIB)


def dav1d_kit() -> Path | None:
    return folder(DAV1D_KIT)


def opus_kit() -> Path | None:
    return folder(OPUS_KIT)


def sdl3_kit() -> Path | None:
    return folder(SDL3_KIT)


def vulkan_sdk() -> Path | None:
    return folder(VULKAN_SDK)


def zbic_archive() -> Path | None:
    return file(ZBIC_ARCHIVE)


def path_folders() -> list[str]:
    """The folders a bundle's launchers put first on PATH: the compiler's, Ninja's and the translator's, and Python's."""
    base = root()
    if base is None:
        return []
    return [str(base / p) for p in (LLVM_BIN, "bin", PYTHON_HOME) if (base / p).is_dir()]


def environment(env: dict[str, str] | None = None) -> dict[str, str]:
    """env (default: this process's) with the bundle's folders first on PATH, and the variables that would point at
    another toolchain removed, so that a child process of a bundle run finds the bundle's tools and no others."""
    out = dict(os.environ if env is None else env)
    folders = path_folders()
    if not folders:
        return out
    out["PATH"] = os.pathsep.join(folders + [p for p in out.get("PATH", "").split(os.pathsep) if p])
    for name in ("VULKAN_SDK", "VK_SDK_PATH", "OOT2_CC", "OOT2_CXX", "OOT2_NINJA", "OOT2_XLATE", "OOT2_GLSLANG",
                 "OOT2_IMGUI", "OOT2_RUNTIME_LIB", "DAV1D_DIR", "OPUS_DIR", "SDL3_DIR", "PYTHONPATH", "PYTHONHOME"):
        out.pop(name, None)
    return out
