# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The check step of `oot2 prepare` (architecture D24c). It runs first, every time, and writes nothing but its own
scratch files and the state file.

In order, cheapest first, so a run that cannot succeed stops in seconds:

1. The output folder: a folder, and empty on a first run, so prepare never mixes its outputs with the player's files.
2. The dump: the file is there.
3. The disk: about 38 GiB free where the outputs go (architecture section 3), less what the kept steps already wrote.
4. The toolchain and the CPU: the host probe (hostcheck.c) is built with the C compiler the compile step will use, so
   a broken compiler fails here rather than an hour later, and run. The CPU must have every feature of x86-64-v3
   (AVX2, BMI1, BMI2, FMA, F16C, LZCNT, MOVBE and the operating system saving the YMM registers), which the generated
   code is compiled for (D4) and runtime/include/oot2/cpu.h requires. Ninja, an archiver and the compiler's C++ driver
   (clang++, which also links the program) must be there too: the compile and link steps run them. So must the Vulkan
   headers and Volk, which the runtime's Vulkan back end compiles
   against, and glslang, which compiles the runtime's own shaders (tools/build_game.py, find_toolchain), and the Python
   package capstone, which the lift step's spin-wait search (tools/lift/spin_waits.py) decodes with. The translate
   step needs Python's compression.zstd (Python 3.14 and later: the shader archives are zstd frames with
   dictionaries) and the shader translator (--xlate, else OOT2_XLATE, else oot2-xlate on PATH), which must start and
   give, through --version, the identity of its own file and the line protocol the pool speaks.
   The memory is reported, never required: the installed and the free amount, with a warning below the reference
   machine's 16 GB (prepare then runs fewer jobs at once, and the game has not been measured with less).
5. Vulkan: the device items of runtime/include/oot2/vk_required.h (architecture section 1), as the probe judged them
   with that header's own tests, which the runtime's device creation uses too. Reported, not required, while the null
   back end is oot2.exe's default (REQUIRE_VULKAN_ITEMS below). The pipelines step needs such a device to build
   with: while the items are only reported it skips, with a warning that names what the GPU lacks, and once they are
   required the same finding fails it (tools/prepare/pipelines.py).
6. The dump's SHA-256 and size. They identify the dump: the steps that read it take them into their keys, so another
   dump, or a damaged one, makes those steps run again. Hashing 25 GB takes minutes, so the hash is recorded with the
   file's size and modification time and computed again only when either changes, or with --verify.

The repository carries no expected hash of the game's files, so a first run accepts any file as the dump and the
extraction step is what finds a wrong one. Whether a table of expected hashes may ship, as TotK Recomp and
oot-pc-recomp ship theirs, is the maintainer's decision.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .runner import Context, PrepareError, Step, file_sha256

TOOLS = Path(__file__).resolve().parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import bundle_env  # noqa: E402  (tools/bundle_env.py)

GiB = 1 << 30
# Architecture section 3. The outputs without shaders, with the RomFS copied, measured on the player's dump: 25.3 GiB,
# 23.5 of it the image. The translate step keeps about 6 GiB (its pack and the split) and holds about 5.5 GiB more
# while it runs (its spill of the translator's outputs, until the pack is complete), so the peak, during that step, is
# about 35 GiB: an inference from S1's modules of the .111 set, 5.3 GB of SPIR-V. 38 GiB leaves about 3 GiB above it.
OUTPUT_BYTES = 38 * GiB
# A 16 GB machine shows about 15.9 GiB to Windows; below 15 GiB is less than the reference machine (README).
LOW_RAM = 15 * GiB
PROBE_SOURCE = Path(__file__).resolve().parent / "hostcheck.c"
TEMP = "temp"                      # dest/temp: the steps' scratch files
PROBE_TIMEOUT = 120                # seconds; a broken driver can hang instance creation
XLATE_PROTOCOL = 2                 # the translator's line protocol (xlate/host/Host.cs, tools/shaders/pool.py)
# The form of the AST optimizer a job gets when it names none, as `--version` says it (optimizer=, xlate/host/Program.cs).
# Prepare's packs hold the sound form only (debug false with AstOptimizer.cs's rule): a translator whose default is the
# old form is refused, so a pack never mixes the two. Together with the identity, which is the SHA-256 of the
# executable, this is what makes the switch of the default a new identity for every key built on it.
XLATE_OPTIMIZER = "sound"
XLATE_TIMEOUT = 60                 # seconds for the translator's --version

CPU_V2 = (("sse3", "SSE3"), ("ssse3", "SSSE3"), ("sse4_1", "SSE4.1"), ("sse4_2", "SSE4.2"), ("popcnt", "POPCNT"),
          ("cx16", "CMPXCHG16B"), ("lahf", "LAHF/SAHF"))
CPU_V3 = CPU_V2 + (("avx", "AVX"), ("avx2", "AVX2"), ("bmi1", "BMI1"), ("bmi2", "BMI2"), ("fma", "FMA"),
                   ("f16c", "F16C"), ("lzcnt", "LZCNT"), ("movbe", "MOVBE"),
                   ("os_ymm", "operating-system support for the YMM registers"))

# The include folder of the runtime, which the probe takes the list of Vulkan device items from (oot2/vk_required.h).
RUNTIME_INCLUDE = PROBE_SOURCE.parents[2] / "runtime" / "include"

# Whether the check requires every Vulkan device item of runtime/include/oot2/vk_required.h, or reports what the device
# lacks as warnings. It turns True in the commit that makes the Vulkan back end oot2.exe's default
# (GFX_DEFAULT_BACKEND in runtime/include/oot2/gpu.h), and not later: tests/test_prepare_check.py ties the two.
REQUIRE_VULKAN_ITEMS = False

DEVICE_TYPES = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "CPU"}


class Check(Step):
    id = "check"
    always = True

    def run(self, ctx: Context) -> dict:
        check_dest(ctx)
        dump = dump_path(ctx)
        check_disk(ctx)
        host = probe_host(ctx)
        missing = cpu_missing(host.get("cpu") or {})
        if missing:
            raise PrepareError(
                f"this CPU ({cpu_name(host)}) lacks {', '.join(missing)}. The game is compiled for x86-64-v3 "
                "(AVX2 and the instructions that came with it, Intel Haswell or AMD Excavator and later).")
        ctx.reporter.say(f"check: CPU {cpu_name(host)}: x86-64-v3")
        line, warnings = memory_report(*ram())
        ctx.reporter.say("check: " + line)
        for w in warnings:
            ctx.reporter.warning(w)
        ctx.reporter.say(f"check: build tools: {build_tools(ctx)}")
        ctx.reporter.say(f"check: menu library: {imgui_note(ctx)}")
        ctx.reporter.say(f"check: window library: {sdl3_note(ctx)}")
        ctx.reporter.say(f"check: Opus decoder: {opus_note(ctx)}")
        ctx.reporter.say(f"check: lift tools: {lift_tools()}")
        ctx.reporter.say(f"check: shader tools: {shader_tools(ctx)}")
        lines, warnings = vulkan_report(host.get("vulkan"), host.get("vulkan_items") or [])
        for line in lines:
            ctx.reporter.say("check: " + line)
        for w in warnings:
            ctx.reporter.warning(w)
        ctx.state["host"] = host
        identify_dump(ctx, dump)
        return {}


def check_dest(ctx: Context) -> None:
    dest = ctx.dest
    if dest.exists() and not dest.is_dir():
        raise PrepareError(f"the output folder {dest} is a file. Choose a folder with --dest.")
    if ctx.fresh and dest.is_dir():
        others = [p.name for p in dest.iterdir() if p.name != TEMP and not p.name.startswith("oot2-prepare.json")]
        if others:
            raise PrepareError(
                f"the output folder {dest} is not empty and holds no prepare state (it has {', '.join(sorted(others)[:3])}"
                f"{', ...' if len(others) > 3 else ''}). Choose an empty or new folder with --dest, so prepare never "
                "mixes its files with others.")


def dump_path(ctx: Context) -> Path:
    given = getattr(ctx.options, "dump", None)
    recorded = ctx.state["inputs"].get("dump", {}).get("path")
    if given is None and recorded is None:
        raise PrepareError("give the dump with --dump FILE (your own .nup of the game).")
    path = Path(given).resolve() if given is not None else Path(recorded)
    if not path.is_file():
        how = "given with --dump" if given is not None else "recorded by the last run; give it again with --dump"
        raise PrepareError(f"the dump {path} ({how}) is not a file.")
    return path


def free_bytes(path: Path) -> int:
    """Free bytes on the disk that holds path, or would hold it once created."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def check_disk(ctx: Context) -> None:
    kept = sum(int(r.get("bytes", 0)) for r in ctx.state["steps"].values())
    need = max(OUTPUT_BYTES - kept, 0)
    free = free_bytes(ctx.dest)
    ctx.reporter.say(f"check: disk of {ctx.dest}: {free / GiB:.1f} GiB free, {need / GiB:.1f} GiB needed")
    if free < need:
        raise PrepareError(
            f"the disk of {ctx.dest} has {free / GiB:.1f} GiB free and prepare needs about {need / GiB:.1f} GiB "
            "there. Free some space, or choose a folder on another disk with --dest.")


def ram() -> tuple[int | None, int | None]:
    """Installed and free memory in bytes, None where the system does not say."""
    if sys.platform == "win32":
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "tpf", "apf", "tv", "av", "aev")]
        st = MemoryStatusEx()
        st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.total, st.avail
        return None, None
    found = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                name, _, rest = line.partition(":")
                if name in ("MemTotal", "MemAvailable"):
                    found[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return found.get("MemTotal"), found.get("MemAvailable")


def memory_report(total: int | None, free: int | None) -> tuple[str, list[str]]:
    """The memory line of the check and its warning: reported, not required (no step has a measured minimum)."""
    if not total:
        return "memory: the system did not say how much", []
    line = f"memory: {total / GiB:.1f} GiB installed" + (f", {free / GiB:.1f} GiB free" if free else "")
    if total < LOW_RAM:
        return line, [f"this PC has {total / GiB:.1f} GiB of memory, below the reference machine's 16 GB: prepare runs "
                      "fewer jobs at once and takes longer, and the game has not been measured with less"]
    return line, []


def compiler(ctx: Context) -> str:
    """The C compiler: --cc, else the OOT2_CC variable, else the release bundle's clang, else clang on PATH. The
    bundle's comes before PATH, so that a player's own clang never decides what a bundle builds."""
    name = getattr(ctx.options, "cc", None) or os.environ.get("OOT2_CC") or bundle_env.compiler() or "clang"
    found = shutil.which(name)
    if not found:
        raise PrepareError(f"no C compiler named {name!r} was found. Give one with --cc or the OOT2_CC variable; "
                           "the compile step needs it too.")
    return found


def runtime_lib(ctx: Context) -> Path | None:
    """The prebuilt runtime library the program is linked with instead of building one: --runtime-lib, else the
    OOT2_RUNTIME_LIB variable, else the release bundle's (tools/bundle.py makes it with the same build_game.py, the
    menu library and the shaders inside). None builds the runtime from runtime/ in the checkout. A named library that is
    not there is a failure by name, not a silent build from source."""
    named = getattr(ctx.options, "runtime_lib", None) or os.environ.get("OOT2_RUNTIME_LIB")
    found = Path(named) if named else bundle_env.runtime_lib()
    if found is not None and not Path(found).is_file():
        raise PrepareError(f"the prebuilt runtime library {found} is not a file (--runtime-lib or OOT2_RUNTIME_LIB)")
    return Path(found).resolve() if found is not None else None


def build_tools(ctx: Context) -> str:
    """Ninja, the archiver, the C++ driver, the Vulkan headers with Volk and glslang, which the compile and link steps
    use (tools/build_game.py), found now so that a missing one stops the run here rather than after the extraction and
    the lift."""
    # tools/ is on the path (tools/oot2.py, prepare/program.py)
    from build_game import BuildError, find_toolchain, vulkan_header_version
    try:
        lib = runtime_lib(ctx)
        tools = find_toolchain(compiler(ctx), vulkan_sdk=getattr(ctx.options, "vulkan_sdk", None),
                               dav1d_dir=getattr(ctx.options, "dav1d_dir", None),
                               sdl3_dir=getattr(ctx.options, "sdl3_dir", None),
                               opus_dir=getattr(ctx.options, "opus_dir", None), prebuilt=lib is not None)
    except BuildError as e:
        raise PrepareError(f"{e}.") from None
    if lib is not None:
        return (f"compiler {compiler(ctx)}, C++ driver {tools.cxx}, Ninja {tools.ninja}, archiver {tools.ar}, "
                f"Vulkan headers {vulkan_header_version(tools.vulkan)} in {tools.vulkan} (the host probe's), the "
                f"prebuilt runtime library {lib}, {av1_note(tools)}")
    return (f"C++ driver {tools.cxx}, Ninja {tools.ninja}, archiver {tools.ar}, Vulkan headers "
            f"{vulkan_header_version(tools.vulkan)} and Volk "
            f"in {tools.vulkan}, glslang {tools.glslang}, {av1_note(tools)}")


def imgui_note(ctx: Context) -> str:
    """Where the runtime's menu gets Dear ImGui (tools/fetch_imgui.py): a folder already laid out as the pin (--imgui,
    OOT2_IMGUI, or fetched/imgui), or the fetch the link step makes. Stops here when the folder named is not the pin, or
    when the fetch is due and there is no git to make it, so that a missing piece stops the run before the extraction."""
    import build_game
    import fetch_imgui
    if runtime_lib(ctx) is not None:
        return "inside the prebuilt runtime library (nothing to fetch)"
    named = getattr(ctx.options, "imgui", None) or os.environ.get("OOT2_IMGUI")
    try:
        folder = build_game.find_imgui(named) if named else None
    except build_game.BuildError as e:
        raise PrepareError(f"{e}.") from None
    if folder is None and not fetch_imgui.problems(fetch_imgui.DEFAULT_DEST):
        folder = fetch_imgui.DEFAULT_DEST
    if folder is not None:
        return f"Dear ImGui {fetch_imgui.VERSION} in {folder}"
    if not shutil.which("git"):
        raise PrepareError("the menu is built on Dear ImGui, which prepare fetches with git (tools/fetch_imgui.py) and "
                           "no git was found on PATH. Install git, or give a fetched folder with --imgui.")
    return (f"Dear ImGui {fetch_imgui.VERSION} is fetched into {fetch_imgui.DEFAULT_DEST} when the program is linked "
            "(git and network access needed)")


def av1_note(tools) -> str:
    """The AV1 decoder of the movies (runtime/video) comes from a dav1d kit (tools/fetch_dav1d.py). Without one the game
    builds and runs up to its first movie, where it stops by name: said here, not refused, so a build for other work
    needs no kit."""
    if tools.dav1d:
        return f"AV1 decoder dav1d kit {tools.dav1d}"
    return "no dav1d kit (the movies cannot play: run tools/fetch_dav1d.py and set DAV1D_DIR)"


def opus_note(ctx: Context) -> str:
    """The decoder behind the hardware Opus override (runtime/audio) comes from a libopus kit (tools/fetch_opus.py).
    Said here, not refused, when there is none: the game then builds and runs with no sound from Opus streams, so a
    build for other work needs no kit."""
    import build_game
    kit = build_game.find_opus(getattr(ctx.options, "opus_dir", None))
    if kit:
        return f"libopus kit {kit}"
    return "no libopus kit (Opus streams make no sound: run tools/fetch_opus.py and set OPUS_DIR)"


def sdl3_note(ctx: Context) -> str:
    """The window layer of the game (window, Vulkan surface, input) comes from an SDL3 kit (tools/fetch_sdl3.py). Said
    here, not refused, when there is none: the game then builds with no window, so a build for other work needs no kit.
    Stops here when the kit named is not exactly the layout that script lays out."""
    import build_game
    try:
        kit = build_game.find_sdl3(getattr(ctx.options, "sdl3_dir", None))
    except build_game.BuildError as e:
        raise PrepareError(f"{e}.") from None
    if kit:
        return f"SDL3 kit {kit}"
    return "no SDL3 kit (the game shows no window: run tools/fetch_sdl3.py and set SDL3_DIR)"


def lift_tools() -> str:
    """capstone, which the lift step's spin-wait search imports (tools/lift/spin_waits.py, run by this interpreter),
    found now so that a missing one stops the run here rather than after the extraction."""
    try:
        import capstone
    except ImportError as e:
        raise PrepareError(f"the Python package capstone cannot be imported ({e}), and the lift step's spin-wait "
                           f"search needs it. Install it with: {sys.executable} -m pip install capstone==5.0.7") \
            from None
    return f"capstone {getattr(capstone, '__version__', '(version unknown)')}"


def translator(ctx: Context) -> Path:
    """The shader translator the translate step runs: --xlate, else the OOT2_XLATE variable, else oot2-xlate on PATH,
    the way the compiler is found (tools/build_xlate.py builds it)."""
    bundled = bundle_env.translator()
    name = getattr(ctx.options, "xlate", None) or os.environ.get("OOT2_XLATE") or (str(bundled) if bundled else None)         or "oot2-xlate"
    given = Path(name)
    if given.is_file():
        return given.resolve()
    found = shutil.which(str(name))
    if not found:
        raise PrepareError(f"no shader translator named {str(name)!r} was found. Give it with --xlate or the OOT2_XLATE "
                           "variable (tools/build_xlate.py builds it); the translate step needs it.")
    return Path(found).resolve()


def translator_command(path: Path) -> list[str]:
    """How the translator starts: the executable itself, or this interpreter for a stand-in script (.py) that speaks
    the same protocol, as the tests use."""
    return [sys.executable, "-I", str(path)] if path.suffix == ".py" else [str(path)]


def translator_identity(ctx: Context) -> str:
    """The translator's identity, the SHA-256 of its file (xlate/host/Program.cs), hashed again only when the file's
    path, size or modification time changes."""
    path = translator(ctx)
    st = path.stat()
    seen = ctx.facts.setdefault("translator", {})
    sig = [str(path), st.st_size, st.st_mtime_ns]
    if seen.get("sig") != sig:
        seen["sig"], seen["sha256"] = sig, file_sha256(path)
    return seen["sha256"]


def shader_tools(ctx: Context) -> str:
    """Python's compression.zstd, which the translate step reads the shader archives with (zstd frames with
    dictionaries, tools/shaders/split.py), and the translator, which must start and answer --version with the
    identity of its own file (the pack records it and every description repeats it) and the protocol the pool speaks,
    found now so that a missing one stops the run here rather than after the extraction."""
    try:
        from compression import zstd  # noqa: F401  (Python 3.14)
    except ImportError:
        raise PrepareError(f"this Python ({platform.python_version()}, {sys.executable}) has no compression.zstd, "
                           "which the translate step reads the game's shader archives with (zstd frames with "
                           "dictionaries). Run oot2 prepare with Python 3.14 or later.") from None
    path = translator(ctx)
    identity = translator_identity(ctx)
    try:
        r = subprocess.run([*translator_command(path), "--version"], capture_output=True, text=True,
                           errors="replace", timeout=XLATE_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise PrepareError(f"the shader translator {path} could not be run ({e}).") from None
    words = r.stdout.split()
    if r.returncode or len(words) < 2 or words[0] != "oot2-xlate":
        raise PrepareError(f"the shader translator {path} did not answer --version (exit code {r.returncode}): "
                           f"{(r.stderr or r.stdout).strip()[-300:]}")
    if words[1] != identity:
        raise PrepareError(f"the shader translator {path} gives the identity {words[1]}, but the SHA-256 of its file "
                           f"is {identity}, so its translations would not match the pack. Rebuild it "
                           "(tools/build_xlate.py) or give another with --xlate.")
    fields = dict(w.split("=", 1) for w in words[2:] if "=" in w)
    if fields.get("proto") != str(XLATE_PROTOCOL):
        raise PrepareError(f"the shader translator {path} speaks protocol {fields.get('proto')}, and this prepare "
                           f"speaks protocol {XLATE_PROTOCOL}. Use the translator of the same release.")
    if fields.get("optimizer") != XLATE_OPTIMIZER:
        raise PrepareError(f"the shader translator {path} translates with the optimizer form "
                           f"{fields.get('optimizer') or 'of a build older than the sound form (it names none)'} by "
                           f"default, and this prepare packs the {XLATE_OPTIMIZER} form only, so an old pack and a new "
                           "one are never mixed. Rebuild it (tools/build_xlate.py) or give another with --xlate.")
    return (f"Python {platform.python_version()} with compression.zstd, translator {path} (protocol {XLATE_PROTOCOL}, "
            f"identity {identity[:16]})")


def vulkan_include(ctx: Context) -> Path | None:
    """The Include folder of the Vulkan SDK (--vulkan-sdk, else VULKAN_SDK) with the Vulkan headers and Volk, found as
    the build finds it (tools/build_game.py, vulkan_include), or None: the probe then reports the CPU and the list."""
    from build_game import vulkan_include as find   # tools/ is on the path (tools/oot2.py, prepare/program.py)
    return find(getattr(ctx.options, "vulkan_sdk", None), need_volk=runtime_lib(ctx) is None)


def build_probe(cc: str, out: Path, include: Path | None, extra: tuple[str, ...] = ()) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    exe = out / ("hostcheck.exe" if os.name == "nt" else "hostcheck")
    cmd = [cc, "-std=c11", "-O1", "-I", str(RUNTIME_INCLUDE), *extra, str(PROBE_SOURCE), "-o", str(exe)]
    if include is not None:
        cmd[1:1] = ["-I", str(include)]
    if os.name != "nt":
        cmd.append("-ldl")
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode or not exe.is_file():
        detail = (r.stderr or r.stdout).strip()[-600:]
        raise PrepareError(f"the C compiler {cc} could not build the host probe:\n{detail}")
    return exe


def run_probe(exe: Path) -> dict:
    # Third-party implicit layers (overlays, recorders) can flood or hang a bare query; the probe needs none.
    env = dict(os.environ, VK_LOADER_LAYERS_DISABLE="~implicit~")
    try:
        r = subprocess.run([str(exe)], capture_output=True, text=True, errors="replace", env=env,
                           timeout=PROBE_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PrepareError(f"the host probe did not finish in {PROBE_TIMEOUT} s (a Vulkan driver hung?).") from None
    try:
        host = json.loads(r.stdout)
    except ValueError:
        host = None
    if not isinstance(host, dict) or not isinstance(host.get("cpu"), dict):
        raise PrepareError(f"the host probe failed (exit code {r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    return host


def probe_host(ctx: Context) -> dict:
    exe = build_probe(compiler(ctx), ctx.dest / TEMP / "check", vulkan_include(ctx))
    return run_probe(exe)


def cpu_name(host: dict) -> str:
    return (host.get("cpu") or {}).get("brand", "").strip() or "unknown CPU"


def cpu_missing(cpu: dict) -> list[str]:
    """The x86-64-v3 features this CPU lacks, by name."""
    return [label for flag, label in CPU_V3 if not cpu.get(flag)]


def pick_device(devices: list[dict]) -> dict | None:
    """The device the game would use: a Vulkan 1.3 one, discrete first, else the first."""
    if not devices:
        return None
    ranked = sorted(devices, key=lambda d: (tuple(d["api"][:2]) < (1, 3), d.get("type") != 2))
    return ranked[0]


def vulkan_report(vk: dict | None, items: list, require: bool | None = None) -> tuple[list[str], list[str]]:
    """Lines describing the Vulkan devices, and one warning per device item the device the game would use lacks.

    items is the probe's "vulkan_items": the [id, what] pairs of runtime/include/oot2/vk_required.h; each device's
    "items" holds the probe's verdict per id, judged by that header's tests. With require (default
    REQUIRE_VULKAN_ITEMS), a missing loader, device or item fails the check instead (PrepareError)."""
    require = REQUIRE_VULKAN_ITEMS if require is None else require

    def problem(text: str) -> tuple[list[str], list[str]]:
        if require:
            raise PrepareError(text)
        return [], [text]

    if vk is None:
        return problem("Vulkan was not probed: the host probe was built without the Vulkan headers "
                       "(give --vulkan-sdk or set VULKAN_SDK).")
    if not vk.get("loader"):
        return problem("no Vulkan loader was found (vulkan-1.dll); the game will need one and a Vulkan 1.3 GPU.")
    if vk.get("instance", 0) != 0:
        return problem(f"the Vulkan loader refused to create an instance (VkResult {vk.get('instance')}).")
    devices = vk.get("devices") or []
    if not devices:
        return problem("Vulkan found no GPU.")
    if not items:
        return problem("the host probe printed no Vulkan device items (runtime/include/oot2/vk_required.h).")
    lines = []
    for d in devices:
        driver = d.get("properties", {}).get("driver_info", "")
        lines.append(f"Vulkan device: {d['name']} ({DEVICE_TYPES.get(d.get('type'), 'other')}), "
                     f"API {'.'.join(map(str, d['api']))}" + (f", driver {driver}" if driver else ""))
    dev = pick_device(devices)
    heaps = [h for h in dev.get("heaps", ()) if h.get("device_local")]
    if heaps:
        big = max(heaps, key=lambda h: h["size"])
        budget = f", budget {big['budget'] / GiB:.2f} GiB" if "budget" in big else ""
        lines.append(f"{dev['name']}: device-local memory {big['size'] / GiB:.2f} GiB{budget}")
    verdicts = dev.get("items") or {}
    lack = [what for ident, what in items if not verdicts.get(ident)]
    lines.append(f"{dev['name']}: has {len(items) - len(lack)} of the {len(items)} features the renderer will rely on")
    if require and lack:
        raise PrepareError(f"{dev['name']} lacks {', '.join(lack)}, which the renderer requires (architecture "
                           "section 1).")
    warnings = [f"{dev['name']} lacks {what} (architecture section 1; not required yet)" for what in lack]
    return lines, warnings


def identify_dump(ctx: Context, path: Path) -> None:
    """Record the dump's size and SHA-256 in the state and in ctx.facts["dump"], hashing only when needed."""
    rep = ctx.reporter
    st = path.stat()
    rec = ctx.state["inputs"].get("dump")
    same_file = bool(rec) and rec.get("path") == str(path) and rec.get("size") == st.st_size \
        and rec.get("mtime_ns") == st.st_mtime_ns
    if same_file and not ctx.verify:
        sha256 = rec["sha256"]
        rep.say(f"check: dump {path.name}: {st.st_size:,} bytes, SHA-256 as recorded")
    else:
        def progress(done, total):
            rep.progress("dump-hash", f"check: hashing the dump: {done / GiB:.1f} of {total / GiB:.1f} GiB",
                         final=done >= total)
        sha256 = file_sha256(path, progress)
        rep.say(f"check: dump {path.name}: {st.st_size:,} bytes, SHA-256 {sha256}")
        if same_file and sha256 != rec["sha256"]:
            rep.warning("the dump's contents changed although its size and date did not; the steps that read it "
                        "run again")
        elif rec and rec.get("sha256") not in (None, sha256):
            rep.say("check: this is not the dump of the last run; the steps that read it run again")
    record = {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256}
    ctx.state["inputs"]["dump"] = record
    ctx.facts["dump"] = record
    ctx.save()
