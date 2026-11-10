oot2-native, release bundle

This folder holds the tools that turn your own dump of The Legend of Zelda: Ocarina of Time (Switch 2, update 1.1.1)
into a game that runs on this PC. It holds no part of the game: no file, key, shader, texture or code of it. The
preparation reads your dump and writes everything it makes into a folder you choose.

To start: double-click oot2-setup.cmd, choose your dump and an empty folder, press Check this PC, then Prepare.
The first preparation takes an hour or more and about 38 GiB of disk. It can be stopped and started again; it resumes
at the first step that did not finish. From a command line: oot2 prepare --dump FILE --dest FOLDER (oot2.cmd).

What this PC needs: Windows 10 or 11 (64-bit), a CPU with AVX2 (Intel Haswell or AMD Excavator and later), 16 GB of
memory, a Vulkan 1.3 GPU and its driver, and no other software. The compiler, Python, Ninja, the shader translator, the
pipeline builder, the runtime library and Microsoft's Visual C++ runtime (vcruntime140.dll, vcruntime140_1.dll,
msvcp140.dll) are in this folder (llvm, python, bin, runtime) and are the only ones used.

One download: the preparation fetches the font for the game's text, BIZ UDPGothic Regular (4.7 MB, SIL Open Font
License 1.1), from raw.githubusercontent.com and places it in the folder you chose, beside its licence text. No font is
in this folder. On a PC without a network connection the font step stops and names the two files to put into
fetched\font in this folder by hand. A TrueType or OpenType font of your own (at most 16 MiB), put into the folder you
chose as fonts\shared-font.ttf, is used instead: the preparation keeps a font that is there, so start it again.

Licences: oot2-native is GPL-3.0-or-later (LICENSE). Every piece in this folder, with its source, version and licence,
has a row in THIRD_PARTY_NOTICES.md (rows whose first cell starts with bundle/), and its licence text is in licenses/.
CONTENTS.tsv lists every file with its size, SHA-256 and the row it belongs to.
