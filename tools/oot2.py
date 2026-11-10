#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The oot2 command line. Two commands:

    py -3 tools/oot2.py prepare --dump FILE --dest DIR [--check] [--verify] [--cc CC] [--vulkan-sdk DIR] [--jobs N]
                                [--xlate FILE] [--translate-jobs N] [--plain-modules DIR] [--zbic-archive FILE]
                                [--names FILE]

prepare builds the game from the player's own dump into DIR, one step after the
other (tools/prepare/). DIR must be empty or new on the first run, and the disk
that holds it needs about 38 GiB free. A run that stops, whether by Ctrl+C, a
failure, a crash or a power cut, keeps the steps that finished, and the next run
with the same DIR starts at the first step that did not; --dump may then be left
out. Running it again after an update of the tools rebuilds only what the update
changed.

--runtime-lib names a runtime library built before, which a release bundle ships; the link step then neither builds
the runtime nor needs Dear ImGui, Volk or glslang.

--check runs the check step alone (the dump, the disk, the CPU, the toolchain,
the shader translator and Vulkan) and says which of the other steps would run and why. It records the
dump's hash and what it found in DIR, so a later run does not hash the dump again.

--verify also hashes the dump again and fully checks every kept output.

--jobs sets how many files the compile step compiles at once (default: the CPU
count, capped by the free memory). The program it builds is DIR/oot2.exe.

--xlate gives the shader translator (default: the OOT2_XLATE variable, else
oot2-xlate on PATH), and --translate-jobs how many copies of it the translate
step runs at once (default: the CPU count, capped by the free memory). The
translated shaders go to DIR/shaders/shader-base.bin. The next step, ledger, reads DIR/ledger.jsonl (the game writes
it when a run ends) and translates the shader answers it recorded as variants into DIR/shaders/shader-variants.bin.

The pipelines step, after ledger, builds with the GPU the game will use every pipeline library and compute pipeline the
shader cache holds into DIR/pipelines/pipeline.cache (tools/prepare/pipelines.py). --pipelines-jobs sets its builder
processes (default: the CPU count, capped by the free memory), --pipelines-list limits it to the job keys in a file,
and --pipelines-cold turns the driver's own cache off for a timing. --skip-pipelines leaves the step out (it takes
hours on a first run, and a run that only needs the program does not need its output). A GPU without every device item
of the renderer makes it skip with a warning, until the Vulkan back end is the default.

--plain-modules gives modules the player already has in plain form (files named main, sdk, ...): they are checked
against the dump's and used instead of decoding, so the ZBIC decoder is not fetched for them. --zbic-archive gives
the decoder's source archive (hexkyz/zstd at the pinned commit, tools/prepare/zbic.py) instead of downloading it.

--names gives a names file of your own, outside the repository, and adds the store step after the modules step: it finds where
the game keeps its named state values in your copy of the main module and writes oot2-store.txt beside the program, with the
hash of each name in your file (never the name), so that lines of an input file (oot2/hid_replay.h) can wait for a value of the
game. A line of the file is `role = name`: the roles are scene, next_scene, entrance, room, age, health, max_health, boss,
paused, loading and menu, or `<kind>.<label>` of your own with a kind of bool, int, uint, float or string; the name is the
game's. Without the option no such file is written and nothing else changes (tools/prepare/store.py).

Exit codes: 0 done, 1 failed (the message says why), 2 bad command line,
3 cancelled.

    py -3 tools/oot2.py census --game DIR --rules FILE [--out DIR] [--image FILE]

census lists the game's scenes from the player's own RomFS, in a folder prepare has already filled (DIR): which scenes
the game has, their places, the ages they exist in, their entrances, the ways from one into another, their events,
movies and tags, and the menus (tools/census/). It reads DIR/romfs/romfs-index.bin and the image that index points
into (the .nup prepare recorded, or romfs/romfs.bin), a few hundred files in all, and writes census.json and
scenes.tsv, the same bytes on every run over the same inputs, into --out (default DIR/census). The census is the game's
data and stays on the player's disk. It is not a prepare step: it changes nothing prepare keeps, so it can run any time
after the extract step, as often as wanted, and into any folder. --rules names a JSON file of the player's own that says
where the scene files are and which keys, words and events mean an age or a tag; the tool has no game name in it, and
tools/census/rules.py describes the file. A file the index lists that cannot be read stops the run with its name.
Exit codes: 0 done, 1 failed (the message says why), 2 bad command line.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare import all_steps  # noqa: E402
from prepare import power  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, Reporter, run_steps  # noqa: E402

EXIT = {"ok": 0, "failed": 1, "cancelled": 3}


def cancel(signum, frame):
    raise KeyboardInterrupt


def positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of at least 1, got {text!r}")
    return n


def steps_for(args: argparse.Namespace) -> list:
    """The steps of this run: all of them, and the store step after the modules step when a names file is given."""
    steps = all_steps()
    if getattr(args, "names", None):
        from prepare.store import Store
        steps.insert([s.id for s in steps].index("modules") + 1, Store())
    return steps


def prepare(args: argparse.Namespace) -> int:
    rep = Reporter()
    dest = args.dest.resolve()
    if args.names:   # a names file that cannot be taken stops the run now, not after the lift
        from prepare import store
        try:
            store.parse_names(args.names.read_bytes())
        except OSError as e:
            rep.error(f"the names file {args.names} cannot be read ({e.strerror or e})")
            return EXIT["failed"]
        except store.NamesError as e:
            rep.error(f"the names file {args.names}, {e}")
            return EXIT["failed"]
    try:
        state = statefile.load(dest)
    except statefile.StateError as e:
        rep.error(str(e))
        return EXIT["failed"]
    ctx = Context(dest, args, rep, state=state, verify=args.verify)
    steps = steps_for(args)
    rep.say(f"oot2 prepare: {'first run' if ctx.fresh else 'resuming'} in {dest}; "
            f"steps: {', '.join(s.id for s in steps)}")
    # The PC stays awake for the run (a first one takes an hour or more); the guard ends with it.
    with power.keep_awake() if power.wanted() else contextlib.nullcontext():
        result = run_steps(steps, ctx, only={"check"} if args.check else None)
    rep.say(f"oot2 prepare: {result}")
    return EXIT[result]


def census_command(args: argparse.Namespace) -> int:
    import census as censusmod
    try:
        censusmod.run(args.game, args.out, args.rules, args.image)
    except censusmod.CensusError as e:
        print(f"oot2 census: {e}", file=sys.stderr)
        return EXIT["failed"]
    return EXIT["ok"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oot2", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="build the game from your own dump",
                       description="Build the game from your own dump, resuming where the last run stopped.")
    p.add_argument("--dump", type=Path, help="your own dump of the game (.nup, update 1.1.1)")
    p.add_argument("--dest", type=Path, required=True, help="output folder, empty or new on the first run")
    p.add_argument("--check", action="store_true", help="run the check step only, and say what the others would do")
    p.add_argument("--verify", action="store_true", help="hash the dump again and fully check every kept output")
    p.add_argument("--cc", help="C compiler (default: the OOT2_CC variable, else clang on PATH)")
    p.add_argument("--vulkan-sdk", type=Path,
                   help="Vulkan SDK whose headers, Volk and glslang the host probe and the build use (default: the "
                        "VULKAN_SDK variable)")
    p.add_argument("--dav1d-dir", type=Path,
                   help="dav1d kit made by tools/fetch_dav1d.py, for the AV1 decoder of the movies (default: the "
                        "DAV1D_DIR variable)")
    p.add_argument("--opus-dir", type=Path,
                   help="libopus kit made by tools/fetch_opus.py, for the game's Opus audio (default: the OPUS_DIR "
                        "variable; without one Opus streams make no sound)")
    p.add_argument("--sdl3-dir", type=Path,
                   help="SDL3 kit made by tools/fetch_sdl3.py, for the window, the Vulkan surface and the input devices "
                        "(default: the SDL3_DIR variable; without one the game shows no window)")
    p.add_argument("--imgui", type=Path,
                   help="Dear ImGui folder made by tools/fetch_imgui.py, for the menu (default: the OOT2_IMGUI "
                        "variable, else fetched/imgui, which prepare fetches itself when it is not there)")
    p.add_argument("--shared-font", type=Path,
                   help="a TrueType or OpenType font of your own for the game's text, placed as fonts/shared-font.ttf "
                        "(default: BIZ UDPGothic, SIL Open Font License 1.1, from fetched/font or downloaded at its "
                        "pinned commit)")
    p.add_argument("--runtime-lib", type=Path,
                   help="a runtime library built before (liboot2rt.a) to link instead of building one (default: the "
                        "OOT2_RUNTIME_LIB variable, else the release bundle's, else the runtime is built from runtime/)")
    p.add_argument("--jobs", type=positive, help="compile jobs (default: the CPU count, capped by the free memory)")
    p.add_argument("--xlate", type=Path, help="shader translator (default: the OOT2_XLATE variable, else oot2-xlate on "
                                             "PATH)")
    p.add_argument("--translate-jobs", type=positive,
                   help="translator processes (default: the CPU count, capped by the free memory)")
    p.add_argument("--pipelines-jobs", type=positive,
                   help="pipeline builder processes (default: the CPU count, capped by the free memory)")
    p.add_argument("--pipelines-list", type=Path,
                   help="file of shader job keys, one per line, to build pipelines for instead of every program")
    p.add_argument("--pipelines-cold", action="store_true",
                   help="build with the driver's own pipeline cache disabled (a measurement, not for play)")
    p.add_argument("--skip-pipelines", action="store_true",
                   help="do not build the pipelines (a development run that needs the program and not the cache)")
    p.add_argument("--plain-modules", type=Path,
                   help="folder of plain modules you already have (main, sdk, ...), used instead of decoding")
    p.add_argument("--zbic-archive", type=Path,
                   help="source archive of the ZBIC decoder, instead of downloading it")
    p.add_argument("--names", type=Path,
                   help="your own file of `role = name` lines (outside the repository): adds the store step, which lets "
                        "lines of an input file wait for a state value of the game")
    c = sub.add_parser("census", help="list the game's scenes from your own RomFS",
                       description="List the game's scenes, entrances, events and menus from the RomFS of a prepared "
                                   "folder, into files that stay on your disk.")
    c.add_argument("--game", type=Path, required=True, help="the prepared folder (after the extract step)")
    c.add_argument("--rules", type=Path, required=True, help="your rules file (tools/census/rules.py describes it)")
    c.add_argument("--out", type=Path, help="output folder (default: GAME/census)")
    c.add_argument("--image", type=Path, help="the image the index points into, when it is not where prepare recorded it")
    args = ap.parse_args(argv)
    if args.command == "census":
        return census_command(args)
    if power.wanted():
        power.install()   # children of this run are not throttled either
    # Ctrl+Break and a terminate request end the run the way Ctrl+C does: the finished steps stay recorded.
    for name in ("SIGBREAK", "SIGTERM"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), cancel)
    return prepare(args)


if __name__ == "__main__":
    sys.exit(main())
