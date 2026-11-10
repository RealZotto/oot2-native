# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The font step of `oot2 prepare` (tools/prepare/font.py): the font the runtime serves for the console's shared font
calls, placed in the game folder as fonts/shared-font.ttf.

No font is in this repository and no test reads the network: the step is given a made-up pin of two small files and a
function in place of the download, which counts the fetches. What is checked: the pinned files from the kit folder or
from the network, each by size and SHA-256, with nothing placed when one is not the pin; the player's own font, placed
without a licence text and refused by name when the runtime would refuse it; what is in place already (the pinned files
taken as they are, any other font the runtime would take kept, a file it would not take replaced); the step's record
(kept, missing, changed, damaged); and that the tool's limits are the runtime's (runtime/hos/ov_system.c) and its step
is in prepare's list and in the setup window's.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import re
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import oot2  # noqa: E402
from installer import logic  # noqa: E402
from prepare import all_steps, font  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, run_steps  # noqa: E402

TTF = b"\0\1\0\0" + bytes(range(200))          # made up: the four bytes of a TrueType file and no font behind them
OTF = b"OTTO" + b"own font" * 40
TEXT = b"a made-up licence text\n"
BASE = "https://fonts.invalid/pin/"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


PIN = {
    "Made-Up.ttf": ("ttf/Made-Up.ttf", len(TTF), sha(TTF), font.FONT_FILE),
    "OFL.txt": ("OFL.txt", len(TEXT), sha(TEXT), font.LICENCE_FILE),
}
SERVED = {BASE + "ttf/Made-Up.ttf": TTF, BASE + "OFL.txt": TEXT}


class FontTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dest = self.root / "game"
        self.kit = self.root / "kit"
        self.served = dict(SERVED)
        self.fetched: list[str] = []

    def get(self, url: str, limit: int) -> bytes:
        self.fetched.append(url)
        data = self.served[url]
        if isinstance(data, Exception):
            raise data
        return data[:limit + 1]

    def step(self) -> font.Font:
        s = font.Font()
        s.pin, s.base, s.kit, s.get = PIN, BASE, self.kit, self.get
        return s

    def run_step(self, shared_font: Path | None = None, verify: bool = False) -> tuple[str, str]:
        out = io.StringIO()
        ctx = Context(self.dest, argparse.Namespace(shared_font=shared_font), Reporter(out=out, err=out),
                      state=statefile.load(self.dest), verify=verify)
        return run_steps([self.step()], ctx), out.getvalue()

    def placed(self) -> dict[str, bytes]:
        folder = self.dest / "fonts"
        return {p.name: p.read_bytes() for p in sorted(folder.iterdir())} if folder.is_dir() else {}


class PinnedFontTest(FontTestBase):
    def test_the_pinned_font_is_downloaded_checked_and_placed_once(self) -> None:
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF, font.LICENCE_FILE: TEXT})
        self.assertEqual(self.fetched, list(SERVED))
        self.assertIn(f"font: downloading {font.NAME}", text)
        self.assertIn(f"{len(TTF):,} bytes, with its licence in {font.LICENCE_FILE}", text)
        record = statefile.load(self.dest)["steps"]["font"]
        self.assertEqual(record["source"], BASE)
        self.assertEqual(record["bytes"], len(TTF) + len(TEXT))
        self.assertEqual(record["files"], {font.FONT_FILE: [len(TTF), sha(TTF)], font.LICENCE_FILE: [len(TEXT), sha(TEXT)]})
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertIn("font: kept (done, inputs unchanged)", text)
        self.assertEqual(len(self.fetched), 2)                       # a kept step asks the network nothing

    def test_the_kit_folder_is_used_before_the_network(self) -> None:
        self.kit.mkdir()
        (self.kit / "Made-Up.ttf").write_bytes(TTF)
        (self.kit / "OFL.txt").write_bytes(TEXT)
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertEqual(self.fetched, [])
        self.assertIn(f"font: {font.NAME} from {self.kit}", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF, font.LICENCE_FILE: TEXT})
        self.assertEqual(statefile.load(self.dest)["steps"]["font"]["source"], str(self.kit))

    def test_a_kit_that_is_not_the_pin_is_not_used(self) -> None:
        self.kit.mkdir()
        (self.kit / "OFL.txt").write_bytes(TEXT)
        for wrong in (None, TTF[:-1], TTF[:-1] + b"\xff"):           # missing, another size, the size and another hash
            with self.subTest(wrong=wrong and len(wrong)):
                if wrong is not None:
                    (self.kit / "Made-Up.ttf").write_bytes(wrong)
                self.assertIsNone(font.kit_files(self.kit, PIN))
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertIn(f"font: {self.kit} does not hold Made-Up.ttf and OFL.txt as pinned", text)
        self.assertEqual(len(self.fetched), 2)
        self.assertEqual(self.placed()[font.FONT_FILE], TTF)
        self.assertEqual((self.kit / "Made-Up.ttf").read_bytes(), TTF[:-1] + b"\xff")   # the kit is the player's: left

    def test_a_download_that_is_not_the_pin_places_nothing(self) -> None:
        for name, wrong in (("longer", TTF + b"more"), ("shorter", TTF[:50]), ("other", TTF[:-1] + b"\xff")):
            with self.subTest(answer=name):
                self.served[BASE + "ttf/Made-Up.ttf"] = wrong
                result, text = self.run_step()
                self.assertEqual(result, "failed", text)
                self.assertIn("oot2 prepare: step font: what https://fonts.invalid/pin/ttf/Made-Up.ttf answered is not "
                              "the pinned file", text)
                read = wrong[:len(TTF) + 1]                          # one byte past the pin is all that is read
                self.assertIn(f"{len(read):,} bytes were read, with SHA-256 {sha(read)}", text)
                self.assertIn(f"the tools expect {len(TTF):,} bytes with {sha(TTF)}", text)
                self.assertIn("run prepare again", text)
                self.assertIn(f"into {self.kit} yourself (as Made-Up.ttf, OFL.txt)", text)
                self.assertIn("--shared-font FILE", text)
                self.assertEqual(self.placed(), {})
                self.assertIsNone(statefile.load(self.dest))        # nothing recorded, and no folder made
                self.assertFalse(self.dest.exists())

    def test_a_wrong_licence_text_places_no_font(self) -> None:
        self.served[BASE + "OFL.txt"] = b"another text entirely\n"
        result, text = self.run_step()
        self.assertEqual(result, "failed", text)
        self.assertEqual(self.placed(), {})                          # the font was fetched and matched; it is not placed alone

    def test_no_network_says_what_to_do(self) -> None:
        for error in (urllib.error.URLError("no route to host"), TimeoutError("timed out"), ConnectionResetError()):
            with self.subTest(error=type(error).__name__):
                self.served[BASE + "ttf/Made-Up.ttf"] = error
                result, text = self.run_step()
                self.assertEqual(result, "failed", text)
                self.assertIn("the shared font could not be downloaded from https://fonts.invalid/pin/ttf/Made-Up.ttf",
                              text)
                self.assertIn(f"yourself into {self.kit} (as Made-Up.ttf, OFL.txt)", text)
                self.assertIn(f"--shared-font FILE, or put into the game folder as fonts/{font.FONT_FILE}", text)
                self.assertIn("running oot2 prepare again resumes at this step", text)
                self.assertEqual(self.placed(), {})
        self.served[BASE + "ttf/Made-Up.ttf"] = TTF                  # the network is back: the next run places the font
        self.assertEqual(self.run_step()[0], "ok")
        self.assertEqual(self.placed()[font.FONT_FILE], TTF)

    def test_the_record_follows_the_placed_files(self) -> None:
        self.assertEqual(self.run_step()[0], "ok")
        (self.dest / "fonts" / font.LICENCE_FILE).unlink()
        result, text = self.run_step()
        self.assertIn("font: started (its output is missing)", text)
        self.assertEqual(self.placed()[font.LICENCE_FILE], TEXT)
        (self.dest / "fonts" / font.FONT_FILE).write_bytes(TTF[:-1] + b"\xff")   # the same size: only the full check sees it
        self.assertIn("font: kept", self.run_step()[1])
        result, text = self.run_step(verify=True)
        self.assertEqual(result, "ok", text)
        self.assertIn("warning: the output of step font failed its full check; it runs again", text)
        self.assertIn(f"was {font.NAME} when this step placed it and its bytes have changed since", text)
        self.assertNotIn("keeping", text)                            # so it is not taken for the player's own
        self.assertEqual(self.placed()[font.FONT_FILE], TTF)
        self.assertEqual(len(self.fetched), 6)

    def test_the_pinned_files_already_in_place_are_taken_as_they_are(self) -> None:
        folder = self.dest / "fonts"
        folder.mkdir(parents=True)
        (folder / font.FONT_FILE).write_bytes(TTF)
        (folder / font.LICENCE_FILE).write_bytes(TEXT)
        self.served[BASE + "ttf/Made-Up.ttf"] = urllib.error.URLError("no route to host")
        result, text = self.run_step()                               # nothing is recorded, and there is no network
        self.assertEqual(result, "ok", text)
        self.assertIn(f"font: {font.NAME} and its licence text are in place already, unchanged", text)
        self.assertEqual(self.fetched, [])
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF, font.LICENCE_FILE: TEXT})
        record = statefile.load(self.dest)["steps"]["font"]
        self.assertEqual(record["source"], str(folder))
        self.assertEqual(record["files"], {font.FONT_FILE: [len(TTF), sha(TTF)], font.LICENCE_FILE: [len(TEXT), sha(TEXT)]})
        self.assertIn("font: kept", self.run_step()[1])
        (folder / font.LICENCE_FILE).unlink()                        # the pinned font alone: both are asked for again,
        result, text = self.run_step()                               # and with no network the usable font is kept
        self.assertEqual(result, "ok", text)
        self.assertIn("font: started (its output is missing)", text)
        self.assertIn(f"font: the download did not succeed; keeping {font.NAME} already at "
                      f"{folder / font.FONT_FILE}, though its licence text is not beside it as pinned", text)
        self.assertEqual(len(self.fetched), 1)
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF})
        self.assertEqual(statefile.load(self.dest)["steps"]["font"]["files"], {font.FONT_FILE: [len(TTF), sha(TTF)]})
        self.assertIn("font: kept", self.run_step()[1])

    def test_the_pinned_font_without_its_licence_text_gets_it_again_from_the_kit(self) -> None:
        folder = self.dest / "fonts"
        folder.mkdir(parents=True)
        (folder / font.FONT_FILE).write_bytes(TTF)
        self.kit.mkdir()
        (self.kit / "Made-Up.ttf").write_bytes(TTF)
        (self.kit / "OFL.txt").write_bytes(TEXT)
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertIn(f"font: {font.NAME} from {self.kit}", text)
        self.assertNotIn("keeping", text)
        self.assertEqual(self.fetched, [])
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF, font.LICENCE_FILE: TEXT})

    def test_a_font_put_over_the_pinned_one_is_kept_though_the_licence_text_is_beside_it(self) -> None:
        self.assertEqual(self.run_step()[0], "ok")                   # the pinned pair, placed by the step
        folder = self.dest / "fonts"
        (folder / font.FONT_FILE).write_bytes(OTF)                   # the player's own font, copied over it by hand
        self.kit.mkdir()                                             # the kit and the network could both give the pin
        (self.kit / "Made-Up.ttf").write_bytes(TTF)
        (self.kit / "OFL.txt").write_bytes(TEXT)
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertIn("font: started (its output is missing)", text)
        self.assertIn(f"font: keeping the font already at {folder / font.FONT_FILE}, a font of your own", text)
        self.assertIn(f"The {font.LICENCE_FILE} beside it is {font.NAME}'s licence text and says nothing about this "
                      "font", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF, font.LICENCE_FILE: TEXT})
        self.assertEqual(len(self.fetched), 2)                       # the first run's two, and none since
        self.assertEqual(statefile.load(self.dest)["steps"]["font"]["files"], {font.FONT_FILE: [len(OTF), sha(OTF)]})
        self.assertIn("font: kept", self.run_step()[1])

    def test_a_file_in_place_that_is_no_font_is_replaced_by_the_pinned_one(self) -> None:
        folder = self.dest / "fonts"
        folder.mkdir(parents=True)
        for data in (b"<html>not a font</html>", b"OTTO", b""):
            with self.subTest(data=data):
                (folder / font.FONT_FILE).write_bytes(data)
                self.assertIsNone(font.font_in_place(folder / font.FONT_FILE))
        with mock.patch.object(font, "MAX_FONT", len(OTF) - 1):
            (folder / font.FONT_FILE).write_bytes(OTF)               # larger than the runtime serves
            self.assertIsNone(font.font_in_place(folder / font.FONT_FILE))
            result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertNotIn("keeping", text)
        self.assertIn(f"font: the file at {folder / font.FONT_FILE} is not a font the runtime takes", text)
        self.assertIn(f"it is replaced by {font.NAME}", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF, font.LICENCE_FILE: TEXT})
        self.assertIsNone(font.font_in_place(folder / "no-such-file.ttf"))
        self.assertIsNone(font.font_in_place(folder))                # a folder is no font file

    def test_download_reads_no_more_than_one_byte_past_the_pinned_size(self) -> None:
        big = self.root / "big.bin"
        big.write_bytes(bytes(1000))
        self.assertEqual(len(font.download(big.as_uri(), 10)), 11)
        self.assertEqual(len(font.download(big.as_uri(), 5000)), 1000)


class OwnFontTest(FontTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.own = self.root / "my font.otf"
        self.own.write_bytes(OTF)

    def test_the_players_own_font_is_placed_as_it_is_without_a_licence_text(self) -> None:
        self.assertEqual(self.run_step()[0], "ok")                   # the pinned font first, with its licence text
        result, text = self.run_step(self.own)
        self.assertEqual(result, "ok", text)
        self.assertIn("font: started (its inputs changed)", text)
        self.assertIn(f"font: your own font {self.own}, placed as it is", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF})       # the other font's licence text does not stay
        self.assertEqual(len(self.fetched), 2)
        self.assertIn("font: kept", self.run_step(self.own)[1])
        self.own.write_bytes(OTF + b"edited")
        result, text = self.run_step(self.own)
        self.assertIn("font: started (its inputs changed)", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF + b"edited"})
        result, text = self.run_step()                               # without the option, as the setup window runs: kept
        self.assertEqual(result, "ok", text)
        self.assertIn("font: started (its inputs changed)", text)
        self.assertIn(f"font: keeping the font already at {self.dest / 'fonts' / font.FONT_FILE}, a font of your own",
                      text)
        self.assertIn(f"Delete the file for {font.NAME}", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF + b"edited"})
        self.assertEqual(len(self.fetched), 2)
        record = statefile.load(self.dest)["steps"]["font"]
        self.assertEqual(record["files"], {font.FONT_FILE: [len(OTF) + 6, sha(OTF + b"edited")]})
        self.assertIn("font: kept", self.run_step()[1])
        (self.dest / "fonts" / font.FONT_FILE).unlink()              # deleted: the pinned font again
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF, font.LICENCE_FILE: TEXT})

    def test_the_pinned_font_given_as_the_players_own_is_kept_by_a_run_without_the_option(self) -> None:
        self.own.write_bytes(TTF)                                    # the pinned font's bytes, fetched by hand elsewhere
        self.served[BASE + "ttf/Made-Up.ttf"] = urllib.error.URLError("no route to host")
        result, text = self.run_step(self.own)
        self.assertEqual(result, "ok", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: TTF})
        for _ in range(2):                                           # one try for the licence text, then it is kept
            result, text = self.run_step()
            self.assertEqual(result, "ok", text)
            self.assertEqual(len(self.fetched), 1)
            self.assertEqual(self.placed(), {font.FONT_FILE: TTF})
        self.assertIn("font: kept", text)

    def test_a_file_in_place_that_is_no_font_is_named_above_the_failed_download(self) -> None:
        folder = self.dest / "fonts"
        folder.mkdir(parents=True)
        (folder / font.FONT_FILE).write_bytes(b"wOF2" + bytes(300))  # a web font: the runtime takes none
        self.served[BASE + "ttf/Made-Up.ttf"] = urllib.error.URLError("no route to host")
        result, text = self.run_step()
        self.assertEqual(result, "failed", text)
        self.assertIn("is not a font the runtime takes", text)
        self.assertLess(text.index("is not a font the runtime takes"), text.index("could not be downloaded"))
        self.assertEqual(self.placed(), {font.FONT_FILE: b"wOF2" + bytes(300)})   # left as it was: nothing replaced it

    def test_a_font_of_the_players_own_changed_at_the_same_size_is_kept_by_the_full_check(self) -> None:
        self.assertEqual(self.run_step(self.own)[0], "ok")
        (self.dest / "fonts" / font.FONT_FILE).write_bytes(OTF[:-1] + b"!")
        result, text = self.run_step(self.own, verify=True)          # with the option: the named file is placed again
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF})
        self.assertEqual(self.run_step()[0], "ok")                   # now recorded as a kept font of the player's own
        (self.dest / "fonts" / font.FONT_FILE).write_bytes(OTF[:-1] + b"!")
        result, text = self.run_step(verify=True)
        self.assertEqual(result, "ok", text)
        self.assertIn("warning: the output of step font failed its full check; it runs again", text)
        self.assertIn("font: keeping the font already at ", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF[:-1] + b"!"})
        self.assertEqual(self.fetched, [])

    def test_a_font_placed_by_hand_is_kept_and_needs_no_network(self) -> None:
        folder = self.dest / "fonts"
        folder.mkdir(parents=True)
        (folder / font.FONT_FILE).write_bytes(OTF)
        self.served[BASE + "ttf/Made-Up.ttf"] = urllib.error.URLError("no route to host")
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertIn("font: started (not done yet)", text)
        self.assertIn("font: keeping the font already at ", text)
        self.assertEqual(self.fetched, [])
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF})       # as it was, and no licence text is written beside it
        self.assertEqual(statefile.load(self.dest)["steps"]["font"]["source"], str(folder / font.FONT_FILE))
        (folder / font.FONT_FILE).write_bytes(OTF + b"another")      # swapped by hand for another: kept again
        result, text = self.run_step()
        self.assertEqual(result, "ok", text)
        self.assertIn("font: started (its output is missing)", text)
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF + b"another"})
        self.assertEqual(self.fetched, [])

    def test_a_file_the_runtime_would_refuse_is_refused_here_by_name(self) -> None:
        cases = (
            (b"<html>not a font</html>", "is not a TrueType or OpenType font (it has 23 bytes and starts with 3c68746d)"),
            (b"OTTO", "is not a TrueType or OpenType font (it has 4 bytes and starts with 4f54544f)"),   # under 12 bytes
            (b"", "is not a TrueType or OpenType font (it has 0 bytes and starts with nothing)"),
        )
        for data, message in cases:
            with self.subTest(data=data[:8]):
                self.own.write_bytes(data)
                result, text = self.run_step(self.own)
                self.assertEqual(result, "failed", text)
                self.assertIn(f"oot2 prepare: step font: the font {self.own} {message}", text)
                self.assertEqual(self.placed(), {})
        self.own.write_bytes(OTF)
        with mock.patch.object(font, "MAX_FONT", len(OTF) - 1):
            with self.assertRaisesRegex(PrepareError, "has 324 bytes; the runtime serves at most 0 MiB"):
                font.own_font(self.own)
        with mock.patch.object(font, "MAX_FONT", len(OTF)):
            self.assertEqual(font.own_font(self.own), OTF)
        result, text = self.run_step(self.root / "no-such-font.ttf")
        self.assertEqual(result, "failed", text)
        self.assertIn("no-such-font.ttf cannot be read", text)
        self.assertEqual(self.fetched, [])                           # a font that was named is never replaced by the pin

    def test_a_folder_or_an_oversized_file_is_refused_by_name_and_never_hashed(self) -> None:
        many = self.root / "a folder"
        many.mkdir()
        (many / "inside.ttf").write_bytes(OTF)
        hashed: list[str] = []

        def hash_of(path: Path, progress=None) -> str:
            hashed.append(path.name)
            return sha(path.read_bytes())

        with mock.patch("prepare.runner.file_sha256", side_effect=hash_of):
            result, text = self.run_step(many)
            self.assertEqual(result, "failed", text)
            self.assertIn(f"oot2 prepare: step font: the font {many} is a folder; name the font file itself", text)
            with mock.patch.object(font, "MAX_FONT", len(OTF) - 1):
                result, text = self.run_step(self.own)
            self.assertEqual(result, "failed", text)
            self.assertIn("has 324 bytes; the runtime serves at most 0 MiB", text)
            self.assertEqual(set(hashed), {"font.py"})               # the key is made of the tool alone
            self.assertEqual(self.run_step(self.own)[0], "ok")       # within the limit: the file is part of the key
            self.assertEqual(set(hashed), {"font.py", self.own.name})
        self.assertEqual(self.placed(), {font.FONT_FILE: OTF})

    def test_the_command_line_refuses_a_font_it_cannot_take_before_any_step(self) -> None:
        self.own.write_bytes(b"<html>not a font</html>")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(oot2, "run_steps", side_effect=AssertionError("a step ran")):
            code = oot2.main(["prepare", "--dest", str(self.dest), "--shared-font", str(self.own)])
        self.assertEqual(code, oot2.EXIT["failed"], err.getvalue())
        self.assertIn(f"the font {self.own} is not a TrueType or OpenType font", err.getvalue())
        self.assertFalse(self.dest.exists())

    def test_every_form_the_runtime_takes_is_taken(self) -> None:
        for magic in font.SFNT:
            self.own.write_bytes(magic + bytes(8))
            self.assertEqual(font.own_font(self.own), magic + bytes(8))


class FontStepPlaceTest(unittest.TestCase):
    def test_the_step_is_in_prepares_list_and_in_the_setup_windows(self) -> None:
        ids = [s.id for s in all_steps()]
        self.assertEqual(ids[:3], ["check", "extract", "font"])      # early: a machine with no network stops in minutes
        step = all_steps()[2]
        self.assertEqual(step.needs, ())
        self.assertEqual((step.pin, step.base, step.kit, step.get), (font.PIN, font.BASE, font.KIT, None))
        self.assertEqual(list(logic.STEP_ORDER), ids)
        self.assertEqual(font.KIT, ROOT / "fetched" / "font")
        ignored = (ROOT / ".gitignore").read_text("utf-8").splitlines()
        self.assertIn("fetched/", ignored)                           # the kit folder is never committed

    def test_the_command_line_takes_a_font(self) -> None:
        seen = []
        with mock.patch.object(oot2, "prepare", lambda a: seen.append(a.shared_font) or 0):
            self.assertEqual(oot2.main(["prepare", "--dest", "x", "--shared-font", "my.ttf"]), 0)
            self.assertEqual(oot2.main(["prepare", "--dest", "x"]), 0)
        self.assertEqual(seen, [Path("my.ttf"), None])

    def test_the_pin_is_one_font_and_its_licence_text_at_one_commit(self) -> None:
        self.assertRegex(font.COMMIT, r"^[0-9a-f]{40}$")
        self.assertEqual(font.BASE, f"https://raw.githubusercontent.com/googlefonts/morisawa-biz-ud-gothic/{font.COMMIT}/")
        self.assertEqual(sorted(v[3] for v in font.PIN.values()), [font.LICENCE_FILE, font.FONT_FILE])
        for name, (path, size, digest, _) in font.PIN.items():
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)
            self.assertTrue(0 < size <= font.MAX_FONT, name)
            self.assertEqual(Path(path).name, name)
        for name, (_, size, digest, _) in font.PIN.items():        # a developer's kit folder, whatever part of it is there
            if (font.KIT / name).is_file():
                data = (font.KIT / name).read_bytes()
                self.assertEqual((len(data), sha(data)), (size, digest), f"{font.KIT / name} is not the pinned file")

    def test_the_limits_are_the_runtimes(self) -> None:
        source = (ROOT / "runtime" / "hos" / "ov_system.c").read_text("utf-8")
        size = re.search(r"FONT_MAX_SIZE = (\d+) << (\d+)", source)
        self.assertEqual(int(size[1]) << int(size[2]), font.MAX_FONT)
        body = re.search(r"static int font_is_sfnt\(const uint8_t \*b\) \{\n(.*?)\n\}", source, re.S)[1]
        magics = [m.encode().decode("unicode_escape").encode("latin-1") for m in re.findall(r'memcmp\(b, "(.*?)", 4\)', body)]
        self.assertEqual(sorted(magics), sorted(font.SFNT))
        self.assertIn("len < 12 || !font_is_sfnt(data)", source)     # is_sfnt's own length
        self.assertIn(f'"%s/fonts/{font.FONT_FILE}"', source)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        unittest.main()
