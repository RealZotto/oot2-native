# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for the setup window over `oot2 prepare` (tools/installer/), without a display.

The lines the window parses come from the real `oot2 prepare` code: tools/oot2.py's prepare() with synthetic steps in
place of all_steps(), printing through the real Reporter and run_steps, so a change to prepare's line format fails here.
The cancel is a real Ctrl+Break (SIGTERM elsewhere) sent to a real child process. Only the last test draws the window,
and it skips where Tk cannot open one. No game file is read.

Run: python -m unittest discover -s tests -p test_installer.py -v
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import oot2  # noqa: E402
from installer import logic  # noqa: E402
from installer.logic import (BAD, CANCELLED, DONE, FAILED, KEPT, OK, PENDING, RUNNING, WARN, WOULD, Form,  # noqa: E402
                             Model, Process, Settings)
from prepare import check  # noqa: E402
from prepare import state as statefile  # noqa: E402
from prepare.runner import Context, PrepareError, Reporter, Step  # noqa: E402

IDS = logic.STEP_ORDER


class Fake(Step):
    """A step that says a progress line, a warning in the check, and stops as the test asks."""

    def __init__(self, id, world):
        self.id, self.world = id, world
        self.always = id == "check"
        self.needs = ()

    def key(self, ctx):
        return self.world["keys"].get(self.id, "v1")

    def run(self, ctx):
        stop = self.world["stop"].get(self.id)
        ctx.reporter.progress(self.id, f"{self.id}: 3 of 7 pieces")
        if self.id == "check":
            ctx.reporter.say("check: memory: 15.9 GiB installed, 9.0 GiB free")
            ctx.reporter.warning("Test GPU lacks memory budget (architecture section 1; not required yet)")
        if stop == "fail":
            raise PrepareError(f"the disk of {ctx.dest} has 1.0 GiB free and prepare needs about 38.0 GiB there. "
                               "Free some space, or choose a folder on another disk with --dest.")
        if stop == "interrupt":
            raise KeyboardInterrupt
        if stop == "crash":
            raise RuntimeError("synthetic bug")
        return {"bytes": 1}


def run_prepare(dest: Path, world: dict, *extra: str) -> tuple[int, list[str]]:
    """tools/oot2.py prepare with the fake steps; returns the exit code and its stdout and stderr as one stream of
    lines, in the order printed, as the window reads them from one pipe."""
    out = io.StringIO()
    steps = [Fake(i, world) for i in IDS]
    with mock.patch.object(oot2, "all_steps", lambda: steps), contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(out):
        code = oot2.main(["prepare", "--dest", str(dest), *extra])
    return code, out.getvalue().splitlines()


def fold(code: int, lines: list[str]) -> Model:
    m = Model()
    for line in lines:
        m.feed(line)
    m.finish(code)
    return m


def states(m: Model) -> dict[str, str]:
    return {s: r.state for s, r in m.rows.items()}


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-setup-")
        self.root = Path(self.tmp.name)
        self.dest = self.root / "game"
        self.world = {"keys": {}, "stop": {}}

    def tearDown(self):
        self.tmp.cleanup()


class PrepareLinesTest(TempCase):
    """prepare's real output folded into step rows."""

    def test_step_lines_parse(self):
        code, lines = run_prepare(self.dest, self.world)
        m, seen = Model(), []
        for line in lines:                      # every step is shown running, then done, in order
            m.feed(line)
            if m.current and (not seen or seen[-1] != (m.current, m.rows[m.current].state)):
                seen.append((m.current, m.rows[m.current].state))
        m.finish(code)
        self.assertEqual(seen, [(s, RUNNING) for s in IDS])
        self.assertEqual(code, 0)
        self.assertEqual(tuple(m.rows), IDS)
        self.assertTrue(m.fresh)
        self.assertEqual(set(states(m).values()), {DONE})
        self.assertTrue(all(r.seconds is not None for r in m.rows.values()))
        self.assertEqual((m.result, m.error, m.current), ("ok", None, None))
        self.assertEqual(m.summary(), "Done: every step is finished.")
        self.assertEqual(m.raw, [])

    def test_check_alone_lists_the_machine_and_what_would_run(self):
        code, lines = run_prepare(self.dest, self.world, "--check")
        m = fold(code, lines)
        self.assertEqual(states(m), {"check": DONE, **{s: WOULD for s in IDS[1:]}})
        marks = dict((text, mark) for mark, text in m.checks)
        self.assertEqual(marks["memory: 15.9 GiB installed, 9.0 GiB free"], OK)
        self.assertEqual(marks["Test GPU lacks memory budget (architecture section 1; not required yet)"], WARN)
        self.assertEqual(m.summary(), "The check passed. Prepare would run 9 of 10 steps.")

    def test_resume_keeps_the_finished_steps(self):
        self.world["stop"]["lift"] = "interrupt"
        code, lines = run_prepare(self.dest, self.world)
        m = fold(code, lines)
        self.assertEqual(code, 3)
        self.assertEqual(m.result, "cancelled")
        self.assertEqual(m.rows["lift"].state, CANCELLED)
        self.assertEqual(m.rows["modules"].state, DONE)
        self.assertEqual(m.rows["compile"].state, PENDING)
        self.assertIn("finished steps are kept", m.summary())
        folder = logic.look(self.dest)
        self.assertEqual(folder.kind, "stopped")
        self.assertIn("Prepare resumes it", folder.message)

        del self.world["stop"]["lift"]
        code, lines = run_prepare(self.dest, self.world)
        m = fold(code, lines)
        self.assertEqual(code, 0)
        self.assertFalse(m.fresh)
        self.assertEqual(m.rows["modules"].state, KEPT)
        self.assertEqual(m.rows["lift"].state, DONE)

    def test_failure_names_the_step_in_the_windows_words(self):
        self.world["stop"]["check"] = "fail"
        code, lines = run_prepare(self.dest, self.world)
        m = fold(code, lines)
        m.finish(code, self.root / "prepare.log")
        self.assertEqual(code, 1)
        self.assertEqual(m.rows["check"].state, FAILED)
        self.assertEqual(m.error_step, "check")
        self.assertNotIn("--", m.error)
        self.assertIn("choose an install folder on another disk", m.error)
        self.assertEqual(m.checks[-1][0], BAD)
        self.assertTrue(m.summary().startswith("Check this PC and the dump failed. The disk of"))

    def test_a_crash_is_called_a_bug_with_the_log(self):
        self.world["stop"]["compile"] = "crash"
        code, lines = run_prepare(self.dest, self.world)
        m = Model()
        for line in lines:
            m.feed(line)
        m.finish(code, Path("prepare.log"))
        self.assertEqual(m.rows["compile"].state, FAILED)
        self.assertTrue(m.error.startswith('An unexpected error stopped the step "Compile". This is a bug'))
        self.assertTrue(m.error.endswith("The whole output is in prepare.log."))
        self.assertTrue(any("Traceback" in r for r in m.raw))

    def test_progress_shows_on_the_running_step(self):
        m = Model()
        m.feed("[    1.0 s] oot2 prepare: first run in X; steps: check, lift")
        m.feed("[    1.0 s] lift: started (not done yet)")
        m.feed("[   61.5 s] lift: 1,200 of 247,520 functions")
        self.assertEqual(m.rows["lift"].state, RUNNING)
        self.assertEqual(m.rows["lift"].detail, "1,200 of 247,520 functions")
        self.assertEqual(m.elapsed, 61.5)
        self.assertEqual(m.summary(), "Lift the game's code to C...")

    def test_a_child_that_dies_silently_is_a_failure(self):
        m = Model()
        m.feed("[    1.0 s] lift: started (not done yet)")
        m.finish(-1073741819)
        self.assertEqual((m.result, m.rows["lift"].state), ("failed", FAILED))
        self.assertIn("without saying why", m.error)

    def test_unknown_steps_and_lines_are_kept(self):
        m = Model()
        m.feed("[    0.0 s] oot2 prepare: resuming in X; steps: check, newstep")
        m.feed("[    0.0 s] newstep: started (its inputs changed)")
        self.assertEqual(m.rows["newstep"].label, "newstep")
        m.feed("usage: oot2 prepare [-h]")
        self.assertEqual(m.raw, ["usage: oot2 prepare [-h]"])


class StatusTest(TempCase):
    def test_fraction_and_bar(self):
        self.assertAlmostEqual(logic.fraction("41,200 of 118,000 (3.1 per second)"), 41200 / 118000)
        self.assertAlmostEqual(logic.fraction("hashing the dump: 2.3 of 23.9 GiB"), 2.3 / 23.9)
        self.assertIsNone(logic.fraction("started"))
        m = Model()
        m.feed("[    0.0 s] oot2 prepare: first run in X; steps: check, lift, link")
        m.feed("[    0.0 s] check: done in 1.0 s")
        m.feed("[   90.0 s] lift: started (not done yet)")
        m.feed("[   90.0 s] lift: 50 of 100 functions")
        share, text = logic.progress(m)
        self.assertAlmostEqual(share, 1.5 / 3)
        self.assertEqual(text, "Step 2 of 3: Lift the game's code to C, 50 of 100 functions, 2 min since the start")

    def test_status_tones(self):
        none, new = logic.look(None), logic.look(self.dest)
        self.assertEqual(logic.status(Model(), none, False, False), ("unknown", "No dump chosen."))
        self.assertEqual(logic.status(Model(), none, True, False)[0], "unknown")
        self.assertEqual(logic.status(Model(), new, True, False)[0], "pending")
        self.assertEqual(logic.status(Model(), new, True, True), ("busy", "Starting oot2 prepare..."))
        code, lines = run_prepare(self.dest, {"keys": {}, "stop": {"check": "fail"}})
        self.assertEqual(logic.status(fold(code, lines), new, True, False)[0], "bad")
        code, lines = run_prepare(self.dest, {"keys": {}, "stop": {}}, "--check")
        self.assertEqual(logic.status(fold(code, lines), logic.look(self.dest), True, False)[0], "ready")


class CompactTest(unittest.TestCase):
    def test_check_lines_are_shortened_for_the_list(self):
        line = (r"build tools: C++ driver C:\Tools\llvm\bin\clang++.EXE, Ninja C:\windows\ninja.EXE, Vulkan headers "
                r"1.4.357 and Volk in C:\VulkanSDK\Include, glslang C:\VulkanSDK\Bin\glslang.exe")
        self.assertEqual(logic.compact(line), "build tools: C++ driver clang++.EXE, Ninja ninja.EXE, Vulkan headers "
                                              "1.4.357 and Volk in Include, glslang glslang.exe")
        self.assertEqual(logic.compact(r"disk of D:\Games\oot2: 10.0 GiB free", r"D:\Games\oot2"),
                         "disk of the install folder: 10.0 GiB free")
        self.assertEqual(logic.compact("dump x.nup: 3 bytes, SHA-256 " + "ab" * 32),
                         "dump x.nup: 3 bytes, SHA-256 abababababab...")

    def test_progress_lines_replace_each_other(self):
        m = Model()
        m.feed("[    0.0 s] check: started (runs every time)")
        for done in ("0.0", "2.3", "23.9"):
            m.feed(f"[    1.0 s] check: hashing the dump: {done} of 23.9 GiB")
        m.feed("[    1.0 s] check: NVIDIA X: device-local memory 7.84 GiB")
        m.feed("[    1.0 s] check: NVIDIA X: has 32 of the 32 features the renderer will rely on")
        self.assertEqual([t for _, t in m.checks], ["hashing the dump: 23.9 of 23.9 GiB",
                                                     "NVIDIA X: device-local memory 7.84 GiB",
                                                     "NVIDIA X: has 32 of the 32 features the renderer will rely on"])


class PlainWordsTest(TempCase):
    """prepare's real messages, with their command-line options, in the window's words."""

    def ctx(self, dump=None):
        opts = argparse.Namespace(dump=dump, cc=None, vulkan_sdk=None)
        return Context(self.dest, opts, Reporter(io.StringIO(), io.StringIO()), state=statefile.load(self.dest))

    def message(self, fn, *args) -> str:
        with self.assertRaises(PrepareError) as cm:
            fn(*args)
        return str(cm.exception)

    def test_every_option_named_by_the_check_is_replaced(self):
        self.dest.mkdir()
        (self.dest / "mine.txt").write_text("x")
        texts = [self.message(check.check_dest, self.ctx()), self.message(check.dump_path, self.ctx())]
        (self.dest / "mine.txt").unlink()
        texts.append(self.message(check.dump_path, self.ctx(self.root / "absent.nup")))
        with mock.patch.object(check, "free_bytes", return_value=1 << 30):
            texts.append(self.message(check.check_disk, self.ctx()))
        for text in texts:
            self.assertRegex(text, r"--(dest|dump)")
            words = logic.plain(text)
            self.assertNotRegex(words, r"--[a-z]", words)
            self.assertTrue(words[0].isupper(), words)


class FolderAndFormTest(TempCase):
    def test_folder_kinds(self):
        self.assertEqual(logic.look(None).kind, "none")
        self.assertEqual(logic.look(self.dest).kind, "new")
        self.dest.mkdir()
        (self.dest / "temp").mkdir()
        self.assertEqual(logic.look(self.dest).kind, "new")
        (self.dest / "photo.jpg").write_bytes(b"")
        self.assertEqual(logic.look(self.dest).kind, "other")
        self.assertEqual(logic.look(self.dest / "photo.jpg").kind, "file")
        (self.dest / "photo.jpg").unlink()
        (self.dest / statefile.NAME).write_text("{not json")
        damaged = logic.look(self.dest)
        self.assertEqual(damaged.kind, "damaged")
        self.assertIn("is damaged", damaged.message)
        record = {"key": "k", "finished": "2026-11-07T00:00:00Z", "seconds": 1.0}
        statefile.save(self.dest, {"format": 1, "inputs": {"dump": {"path": "D:/my.nup"}},
                                   "steps": {"check": record, "link": record}})
        self.assertEqual(logic.look(self.dest).kind, "stopped")
        (self.dest / logic.EXE_NAME).write_bytes(b"")
        ready = logic.look(self.dest)
        self.assertEqual((ready.kind, ready.dump), ("ready", "D:/my.nup"))

    def test_problems_before_a_run(self):
        self.assertEqual(len(Form().problems()), 2)
        dump = self.root / "my.nup"
        dump.write_bytes(b"x")
        self.assertEqual(Form(str(dump), str(self.dest)).problems(), [])
        self.assertIn("is not a file", Form(str(self.root / "no.nup"), str(self.dest)).problems()[0])
        self.dest.mkdir()
        statefile.save(self.dest, {"format": 1, "inputs": {"dump": {"path": str(dump)}}, "steps": {}})
        self.assertEqual(Form("", str(self.dest)).problems(), [])   # resuming: the recorded dump is used

    def test_command(self):
        cmd = Form(" D:/d/my.nup ", "E:/oot2", skip_pipelines=True).command(check=True, python="py")
        self.assertEqual(cmd[:6], ["py", "-I", "-X", "utf8", str(logic.OOT2), "prepare"])
        self.assertEqual(cmd[6:], ["--dest", str(Path("E:/oot2")), "--dump", str(Path("D:/d/my.nup")), "--check",
                                   "--skip-pipelines"])
        self.assertNotIn("--dump", Form("", "E:/oot2").command(python="py"))
        with mock.patch.object(oot2, "prepare", lambda a: 0):   # the real parser takes every option given
            self.assertEqual(oot2.main(cmd[5:]), 0)

    def test_child_runs_on_the_console_interpreter(self):
        (self.root / "python.exe").write_bytes(b"")
        self.assertEqual(logic.child_python(str(self.root / "pythonw.exe")), str(self.root / "python.exe"))
        self.assertEqual(logic.child_python(str(self.root / "python.exe")), str(self.root / "python.exe"))

    def test_settings_keep_three_fields_and_survive_a_bad_file(self):
        s = Settings(self.root / "p" / "setup.json")
        self.assertEqual(s.load(), Form())
        s.save(Form("D:/my.nup", "E:/oot2", True))
        self.assertEqual(s.load(), Form("D:/my.nup", "E:/oot2", True))
        self.assertEqual(set(json.loads(s.path.read_text(encoding="utf-8"))), set(Settings.FIELDS))
        s.path.write_text("[1, 2]")
        self.assertEqual(s.load(), Form())

    def test_log_and_settings_live_in_the_profile(self):
        with mock.patch.dict(os.environ, {"OOT2_SETUP_HOME": str(self.root)}):
            self.assertEqual(Settings().path, self.root / "setup.json")
            self.assertEqual(logic.log_path().parent, self.root / "logs")

    def test_game_command_and_exit_words(self):
        self.assertEqual(logic.game_command("E:/oot2"), [str(Path("E:/oot2") / logic.EXE_NAME)])
        self.assertEqual(logic.game_exit(0), "The game ended normally.")
        self.assertIn("exit code 70", logic.game_exit(70))
        self.assertIn("unexpectedly", logic.game_exit(5))


class ProcessTest(TempCase):
    CHILD = (
        "import signal, sys, time\n"
        "def stop(*_):\n"
        "    print('cancelled during step lift; the steps finished before it are kept', flush=True)\n"
        "    sys.exit(3)\n"
        "for name in ('SIGBREAK', 'SIGTERM'):\n"
        "    if hasattr(signal, name): signal.signal(getattr(signal, name), stop)\n"
        "print('[    0.0 s] lift: started (not done yet)', flush=True)\n"
        "print('caf\\u00e9', flush=True)\n"
        "for _ in range(600):\n"
        "    time.sleep(0.1)\n"   # one long sleep on Windows wakes for Ctrl+C only, not for the Ctrl+Break handler
    )

    def wait(self, p: Process, limit: float = 30.0):
        """Drain the lines as the window does (poll() answers only once they are drained) until the child ends."""
        t0, lines = time.monotonic(), []
        while True:
            while not p.lines.empty():
                lines.append(p.lines.get_nowait())
            code = p.poll()
            if code is not None:
                return code, lines
            if time.monotonic() - t0 > limit:
                p.kill()
                self.fail("the child did not end")
            time.sleep(0.05)

    def started(self, p: Process):
        """Wait for the child's two lines in the log, so its signal handler is in place before the cancel."""
        self.addCleanup(p.kill)
        log, t0 = self.root / "run.log", time.monotonic()
        while not (log.is_file() and log.read_text(encoding="utf-8").count("\n") >= 3):
            if time.monotonic() - t0 > 30:
                self.fail("the child printed nothing")
            time.sleep(0.05)

    def test_cancel_reaches_the_child_and_the_log_has_its_lines(self):
        log = self.root / "run.log"
        p = Process([sys.executable, "-I", "-X", "utf8", "-c", self.CHILD], log=log)   # as Form.command
        self.started(p)
        self.assertTrue(p.cancel())
        code, lines = self.wait(p)
        self.assertEqual(code, 3)
        self.assertEqual(lines[-1], "cancelled during step lift; the steps finished before it are kept")
        self.assertIn("café", lines)
        text = log.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("$ "))
        self.assertIn("lift: started", text)
        self.assertFalse(p.cancel())   # once only

    def test_kill_ends_a_child_that_ignores_the_cancel(self):
        stubborn = self.CHILD.replace("sys.exit(3)", "pass")
        p = Process([sys.executable, "-I", "-c", stubborn], log=self.root / "run.log")
        self.started(p)
        p.cancel()
        p.kill()
        code, _ = self.wait(p)
        self.assertNotEqual(code, 0)


class NoConnectionTest(unittest.TestCase):
    def test_the_window_opens_no_connection(self):
        """Nothing of the player's game leaves the PC through the window: it imports no network module (prepare's own
        downloads of pinned sources are prepare's, checked by hash)."""
        files = sorted((ROOT / "tools" / "installer").glob("*.py")) + [ROOT / "tools" / "oot2_setup.pyw"]
        self.assertGreaterEqual(len(files), 5)
        net = re.compile(r"^\s*(?:import|from)\s+(socket|ssl|urllib|http|requests|ftplib|smtplib|webbrowser)\b", re.M)
        for f in files:
            self.assertIsNone(net.search(f.read_text(encoding="utf-8")), f.name)


class MemoryLineTest(unittest.TestCase):
    def test_memory_is_reported_and_warned_below_the_reference(self):
        GiB = 1 << 30
        line, warnings = check.memory_report(int(15.9 * GiB), 9 * GiB)
        self.assertEqual((line, warnings), ("memory: 15.9 GiB installed, 9.0 GiB free", []))
        line, warnings = check.memory_report(8 * GiB, None)
        self.assertEqual(line, "memory: 8.0 GiB installed")
        self.assertIn("below the reference machine's 16 GB", warnings[0])
        self.assertEqual(check.memory_report(None, None)[1], [])
        total, free = check.ram()
        self.assertTrue(total is None or total > 0)


class WindowTest(unittest.TestCase):
    def test_selftest_draws_every_screen(self):
        try:
            import tkinter
            tkinter.Tk().destroy()
        except Exception as e:   # no Tk in this Python, or no display (Linux CI)
            self.skipTest(f"no window here: {e}")
        from installer import window
        with mock.patch.dict(os.environ):
            self.assertEqual(window.selftest(keep_open_ms=0), 0)


if __name__ == "__main__":
    unittest.main()
