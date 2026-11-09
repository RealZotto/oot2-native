# oot2-native

**oot2-native** is a static recompilation of *The Legend of Zelda: Ocarina of Time* for Nintendo Switch 2 (update 1.1.1) into a native x86-64 application for Windows, with its own NVN2 front end and a Vulkan renderer ([docs/architecture.md](docs/architecture.md)).

> [!IMPORTANT]
> **No game files are included.** The repository holds tools and a runtime and nothing taken from the game. You need your own legally obtained copy of *Ocarina of Time* for Switch 2, dumped from your own console: the base game with its 1.1.1 update, as one `.nup` file. Instructions for obtaining that file will not be provided.
>
> `oot2 prepare` builds the game from your dump on your machine, so the result contains the game. It is for your own use. Do not share it, and never attach game files, keys, captures or screenshots to an issue or a pull request.

## Status

Developer preview. The game is not playable yet, and there is no release yet. `oot2 prepare` builds the program from a dump. The program runs the game's start-up, decodes its start-up movies and reaches the title menu's draws on a null graphics back end, which has no window and shows no picture. The Vulkan back end is built in parts (device, pipelines, the replay of the game's draws and dispatches, texture upload) and tested on a GPU, but it has no window yet and does not draw task and mesh programs or compressed textures, so nothing is shown. The first milestone is the title screen with its movie on the Vulkan back end at 60 fps.

A release follows when the prebuilt bundle described under Planned exists. Until then the game is prepared from source, as described below.

Report problems as issues, following [CONTRIBUTING.md](CONTRIBUTING.md).

## Help wanted

Open work that nobody has taken. Say in an issue that you are taking an item before you start, so that nobody does it twice. Every item needs your own dump to test against, and none of them needs a game file in a commit.

- Audio: serve the console's audio services with a mixer and a platform output stream, in place of the silent stand-in the game starts with today
- Shader translator: the console's NaN rules for min, max, clamp, saturate and float-to-integer conversion
- `oot2 prepare`: return each shader translation over the translator's line protocol, so that the translate step writes no loose file (a real-time antivirus scanner slows every file that is reopened)
- Vulkan back end: draw task and mesh programs, direct and indirect (needs a GPU with `VK_EXT_mesh_shader`)
- Vulkan back end: fit the bindless tables to the game: custom border colours, the highest texture ID it registers, and two pools in one submission

## Features

- Native code, not emulation: the game's AArch64 code is lifted to C, compiled on your machine and linked with a runtime that stands in for the console's operating system, graphics API and services
- Vulkan 1.3 renderer behind an NVN2 front end, with task and mesh shaders; NVIDIA GPUs are the target of the first release
- The game's shaders are translated, and their pipelines built, when you prepare the game, not during play
- Movies decoded with dav1d (AV1)
- Sound: the console's audio services are served on the PC, the game's Opus streams are decoded with libopus, and the mix plays through SDL3
- A run that stops keeps its finished steps and starts again where it left off; running it again after an update rebuilds only what the update changed
- A setup window over `oot2 prepare`: it asks for the dump and the folder, checks the PC, shows the progress of each step and starts the game when it is prepared
- `oot2 prepare --check` reports what is missing (dump, disk, CPU, toolchain, shader translator, Vulkan) before anything is built

## Planned

- A prebuilt bundle with the toolchain, the shader translator, the pipeline builder, the runtime library and Microsoft's Visual C++ runtime DLLs, so that nothing has to be installed first. It will contain no file derived from the game either
- The title screen with its movie on the Vulkan back end, then the game itself: the target is 60 fps in Kokiri Forest, docked, at 2560 by 1440, on the reference machine below

## Installing

The game is prepared from your dump with `oot2 prepare`, from the command line or through a setup window. Both need the tools below until the prebuilt bundle exists.

### Windows

Requirements:

- Windows 10 or 11, x86-64 with the `x86-64-v3` features (AVX2, BMI1, BMI2, FMA, F16C, LZCNT, MOVBE).
- A Vulkan 1.3 GPU. NVIDIA is the target of the first release: the mesh path assumes a subgroup size of 32. Reference machine: Ryzen 5 5600X, 16 GB RAM, GeForce RTX 3060 Ti 8 GB, driver 617.14.
- Your own dump: one `.nup` file holding the base program and its update (about 25 GiB), and about 38 GiB free on the disk that receives the build.
- Until the prebuilt bundle exists: Python 3.14 with `numpy` and `capstone`; clang and clang++ from [LLVM-MinGW](https://github.com/mstorsjo/llvm-mingw) (x86_64; CI and the reference machine use 22.1.8), Ninja and git; the Vulkan headers, Volk and glslang from the Vulkan SDK 1.4.357 or from `tools/fetch_vulkan.py`; the .NET 10 SDK and the Visual Studio Build Tools (C++) for the shader translator; a dav1d kit from `tools/fetch_dav1d.py` (`DAV1D_DIR`) for the movies, without which the check step says so.

Steps:

1. Clone the repository and install the tools above.
2. Lay out the Vulkan headers, Volk and glslang once, unless the Vulkan SDK is installed, and build the shader translator:

   ```bat
   py -3.14 tools/fetch_vulkan.py --dest vk
   set VULKAN_SDK=vk
   py -3.14 tools/build_xlate.py
   set OOT2_XLATE=build\xlate\publish\oot2-xlate.exe
   ```

3. Check the machine and the dump. `GAME.nup` is your dump and `OUT` an empty or new folder:

   ```bat
   py -3.14 tools/oot2.py prepare --dump GAME.nup --dest OUT --check
   ```

4. Build the game, then run `OUT\oot2.exe`:

   ```bat
   py -3.14 tools/oot2.py prepare --dump GAME.nup --dest OUT
   ```

   Or use the setup window instead of steps 3 and 4: `pyw -3.14 tools/oot2_setup.pyw`, or a double-click on that file. Its logs are in `%LOCALAPPDATA%\oot2-native\logs`.

On a 6-core desktop the lift and compile take about half an hour and the shader translation about ten minutes. The pipeline step takes hours on a first run; `--skip-pipelines` leaves it out. A run that stops keeps its finished steps: start it again with the same `--dest`. [tools/oot2.py](tools/oot2.py) documents every option.

## Credits

oot2-native is written for this project, with parts adapted from earlier recompilations of Switch and Switch 2 games and strongly inspired by them. What was taken from each, with its source, commit and licence, is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [oot-pc-recomp](https://github.com/philneitz/oot-pc-recomp), a fork of [Chiuauaa's oot-pc-recomp](https://github.com/Chiuauaa/oot-pc-recomp)
- [DKB Recomp](https://github.com/Lucas-blip-png/dkb-pc-recomp)
- [TotK Recomp](https://github.com/valentderah/totk-pc-recomp), whose legal model this project follows
- [LASRecomp](https://github.com/peiikai/LASRecomp)
- [nxnative](https://github.com/seedds/nxnative) and [MacNativeGames](https://github.com/ly05010419/MacNativeGames)
- [Lichii](https://github.com/zortos293/Lichii) and [Eglantine](https://github.com/Djihads80/Eglantine)
- [Ryujinx](https://github.com/Ryujinx/Ryujinx), whose shader translator is vendored as nxnative carries it
- [Dear ImGui](https://github.com/ocornut/imgui), [dav1d](https://code.videolan.org/videolan/dav1d), [libopus](https://opus-codec.org/), [Volk](https://github.com/zeux/volk), [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers), [glslang](https://github.com/KhronosGroup/glslang) and [hexkyz/zstd](https://github.com/hexkyz/zstd), fetched at pinned versions and checked by hash rather than copied into the repository

Each keeps its own licence and header, and its licence text is in [licenses/](licenses/) or next to its files.

## Licence

oot2-native is licensed GPL-3.0-or-later: the text is in [LICENSE](LICENSE), and every source file written for the project carries `SPDX-License-Identifier: GPL-3.0-or-later`. This licence covers the project's source and grants no right in the game.

Parts that come from other projects keep their own licence and header (MIT, BSD, ISC, zlib or GPL-3.0-or-later, all compatible with GPL-3.0-or-later). Each is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) with its source, commit and modifications. Third-party code that is not in the repository keeps its own licence and is listed in the same file.

oot2-native is an independent project, not affiliated with or endorsed by Nintendo. *The Legend of Zelda*, *Ocarina of Time*, *Nintendo Switch* and *Nintendo Switch 2* are trademarks of Nintendo. All other trademarks belong to their owners.

## Building from source

The runtime, the lifter and the translator are built by the steps above. The tests are synthetic and need no game file:

```sh
python tools/check_content.py --all
python -m unittest discover -s tests -v
```

[CONTRIBUTING.md](CONTRIBUTING.md) covers the content check that runs before every commit and in CI, the tests, licence headers and the rules for code taken from other projects. Nothing the build writes is committed: the content check refuses it. [docs/architecture.md](docs/architecture.md) holds the design and the decisions.
