# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""What the setup window decides, without Tk, so tests/test_installer.py runs it headless.

    Form        what the player chose (dump, install folder, the pipeline checkbox); problems() and command()
    Folder      what the install folder already holds: nothing, a stopped run to resume, or a prepared game
    parse()     one line of `oot2 prepare` output (tools/prepare/runner.py, Reporter) as an Event
    Model       the events folded into step rows, the machine check's list, warnings, the error and the result
    plain()     prepare's message with its command-line flags replaced by the window's names
    status(), progress()   the window's one status line with its tone, and the bar's share and label
    Process     the prepare child process: its lines to a queue and to the log, and its cancel by Ctrl+Break
    Settings    the paths remembered between runs, in the player's own profile
    game_command(), game_exit()   start the prepared oot2.exe, and say how it ended

prepare's output is read as it is printed for a person (no event format of its own): each stamped line is
"[   12.3 s] text", a failure is "oot2 prepare: step ID: message" on stderr. The tests produce those lines with the real
Reporter and run_steps, so a change to that format fails them rather than this parse.

Nothing here opens a connection. The only files it writes are the log of each run and the remembered paths, both
under the player's profile (LOG_DIR, Settings.path).

Process, send_break and the break helper are adapted from TotK Recomp installer/guilogic.py at 0ae47a5 (MIT,
Copyright (c) 2026 Valiantsin Dzerakh, licenses/totk-MIT.txt). Changed: the child is `tools/oot2.py prepare` with its
plain lines instead of `python -m installer` with JSON events; the reader also writes the log; the child runs on
python.exe when the window runs on pythonw.exe (a console process, so Ctrl+Break reaches it); kill() ends the whole
process tree.
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
OOT2 = TOOLS / "oot2.py"
EXE_NAME = "oot2.exe" if os.name == "nt" else "oot2"   # tools/build_game.py EXE_NAME
STATE_NAME = "oot2-prepare.json"                       # tools/prepare/state.py NAME
SCRATCH = "temp"                                       # tools/prepare/check.py TEMP

# The step ids of tools/prepare/__init__.py all_steps(), in the words the window shows. A step this table lacks (a
# newer prepare) shows its id.
STEP_LABELS = {
    "check": "Check this PC and the dump",
    "extract": "Extract the game from the dump",
    "font": "Place the font for the game's text",
    "translate": "Translate the shaders",
    "ledger": "Translate the shader variants the game recorded",
    "pipelines": "Build the GPU pipelines",
    "modules": "Decode the program modules",
    "lift": "Lift the game's code to C",
    "compile": "Compile",
    "link": "Link the program",
}
STEP_ORDER = tuple(STEP_LABELS)

PENDING, RUNNING, DONE, KEPT, WOULD, FAILED, CANCELLED = (
    "pending", "running", "done", "kept", "would run", "failed", "cancelled")
STATE_WORDS = {PENDING: "waiting", RUNNING: "running", DONE: "done", KEPT: "already done", WOULD: "to do",
               FAILED: "failed", CANCELLED: "stopped"}
OK, WARN, BAD = "ok", "warning", "error"   # marks of the machine check's lines

# tools/oot2.py's exit codes
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_CANCELLED = 0, 1, 2, 3
# runtime/include/oot2/runtime.h and runtime/program/main.c
GAME_EXITS = {
    0: "The game ended normally.",
    2: "The game refused its command line or its settings file.",
    70: "The game stopped on an error it names (a fatal error).",
    71: "The game reached something oot2-native does not do yet.",
    72: "The game stopped itself.",
}

ID = r"[a-z][a-z0-9_]*"
STAMP = re.compile(r"^\[\s*(\d+(?:\.\d+)?) s\] (.*)$")
PLAN = re.compile(r"^oot2 prepare: (first run|resuming) in (.*); steps: (.*)$")
RESULT = re.compile(r"^oot2 prepare: (ok|failed|cancelled)$")
STEP = re.compile(rf"^({ID}): (started|kept|would run) \((.*)\)$")
DONE_IN = re.compile(rf"^({ID}): done in (\d+(?:\.\d+)?) s$")
CANCEL = re.compile(rf"^cancelled during step ({ID}|None);")
INFO = re.compile(rf"^({ID}): (.+)$")
PROGRESS = re.compile(r"^(.*): [\d.,]+ of [\d.,]+\b")
WIN_PATH = re.compile(r"\b[A-Za-z]:[\\/](?:[^,;()\\/]*[\\/])*([^,;()\\/\s]+)")
SHA256 = re.compile(r"\b([0-9a-f]{12})[0-9a-f]{52}\b")
STEP_ERROR = re.compile(rf"^oot2 prepare: step ({ID}): (.*)$")
ERROR = re.compile(r"^oot2 prepare: (?:error: )?(.*)$")


@dataclass
class Event:
    kind: str            # plan, started, done, kept, would, info, warning, cancelled, result, error, raw
    step: str | None = None
    text: str = ""
    seconds: float | None = None   # a finished step's duration
    steps: tuple[str, ...] = ()
    at: float | None = None        # the line's time since the run started


def parse(line: str) -> Event:
    """One line of prepare's output, stdout and stderr alike (the two are read as one stream)."""
    line = line.rstrip("\r\n")
    m = STAMP.match(line)
    if not m:
        m = STEP_ERROR.match(line)
        if m:
            return Event("error", m.group(1), m.group(2))
        m = ERROR.match(line)
        if m:
            return Event("error", None, m.group(1))
        return Event("raw", text=line)
    at, text = float(m.group(1)), m.group(2)
    if m := PLAN.match(text):
        steps = tuple(s.strip() for s in m.group(3).split(",") if s.strip())
        return Event("plan", text=m.group(1), steps=steps, at=at)
    if m := RESULT.match(text):
        return Event("result", text=m.group(1), at=at)
    if m := STEP.match(text):
        kind = {"started": "started", "kept": "kept", "would run": "would"}[m.group(2)]
        return Event(kind, m.group(1), m.group(3), at=at)
    if m := DONE_IN.match(text):
        return Event("done", m.group(1), seconds=float(m.group(2)), at=at)
    if m := CANCEL.match(text):
        return Event("cancelled", None if m.group(1) == "None" else m.group(1), text, at=at)
    if text.startswith("warning: "):
        return Event("warning", text=text[len("warning: "):], at=at)
    if m := INFO.match(text):
        return Event("info", m.group(1), m.group(2), at=at)
    return Event("raw", text=text, at=at)


# prepare's messages name its command-line options; the window has fields instead. Longest first.
FLAG_WORDS = (
    ("Choose an empty or new folder with --dest", "Choose an empty or new install folder"),
    ("choose a folder on another disk with --dest", "choose an install folder on another disk"),
    ("Choose a folder with --dest", "Choose an install folder"),
    ("give the dump with --dump FILE", "choose your dump"),
    ("recorded by the last run; give it again with --dump", "recorded by the last run; choose it again"),
    ("given with --dump", "the dump you chose"),
    ("--dump FILE", "your dump"),
    ("--dest", "the install folder"),
    ("--dump", "your dump"),
)


def plain(message: str) -> str:
    """prepare's message for the window: the options it names for the command line become the window's fields, and
    the first letter is a capital."""
    for flag, words in FLAG_WORDS:
        message = message.replace(flag, words)
    return message[:1].upper() + message[1:]


def compact(text: str, dest: str | os.PathLike | None = None) -> str:
    """A check line for the window's short list: the install folder by that name, other paths by their last part, a
    SHA-256 by its first 12 digits. The log keeps the whole line."""
    if dest:
        for form in {str(dest), str(Path(dest)), str(Path(dest)).replace("\\", "/")}:
            text = text.replace(form, "the install folder")
    text = WIN_PATH.sub(lambda m: m.group(1), text)
    return SHA256.sub(lambda m: m.group(1) + "...", text)


def label(step: str | None) -> str:
    return STEP_LABELS.get(step or "", step or "the run")


@dataclass
class Row:
    step: str
    state: str = PENDING
    detail: str = ""
    seconds: float | None = None

    @property
    def label(self) -> str:
        return label(self.step)


@dataclass
class Model:
    """The events of one run folded into what the window shows."""
    rows: dict[str, Row] = field(default_factory=lambda: {s: Row(s) for s in STEP_ORDER})
    checks: list[tuple[str, str]] = field(default_factory=list)   # (OK|WARN|BAD, text) of the machine check
    warnings: list[str] = field(default_factory=list)
    error: str | None = None          # the plain-word failure, when the run failed
    error_step: str | None = None
    result: str | None = None         # ok, failed, cancelled, once the run said so
    fresh: bool | None = None         # True on a first run in the folder, False when resuming
    current: str | None = None
    elapsed: float = 0.0
    raw: list[str] = field(default_factory=list)   # lines that are none of the above (a traceback, argparse's usage)

    def feed(self, line: str) -> Event:
        e = parse(line)
        if e.at is not None:
            self.elapsed = e.at
        if e.kind == "plan":
            self.fresh = e.text == "first run"
            self.rows = {s: self.rows.get(s) or Row(s) for s in e.steps}
        elif e.kind in ("started", "kept", "would", "done"):
            row = self.row(e.step)
            if e.kind == "started":
                row.state, row.detail, self.current = RUNNING, "", e.step
            elif e.kind == "kept":
                row.state, row.detail = KEPT, ""
            elif e.kind == "would":
                row.state, row.detail = WOULD, e.text
            else:
                row.state, row.seconds, row.detail = DONE, e.seconds, ""
                if self.current == e.step:
                    self.current = None
        elif e.kind == "info":
            row = self.row(e.step)
            row.detail = e.text
            if e.step == "check":
                head = PROGRESS.match(e.text)
                last = PROGRESS.match(self.checks[-1][1]) if self.checks else None
                if head and last and head.group(1) == last.group(1):
                    self.checks[-1] = (OK, e.text)      # "hashing the dump: 4.7 of 23.9 GiB" replaces 2.3 of 23.9
                else:
                    self.checks.append((OK, e.text))
        elif e.kind == "warning":
            self.warnings.append(e.text)
            if self.current == "check":
                self.checks.append((WARN, e.text))
        elif e.kind == "cancelled":
            step = e.step or self.current
            if step:
                self.row(step).state = CANCELLED
            self.current = None
        elif e.kind == "error":
            step = e.step or self.current or next((s for s, r in self.rows.items() if r.state == FAILED), None)
            self.error_step = step
            self.error = plain(e.text)
            if step:
                row = self.row(step)
                row.state, row.detail = FAILED, self.error
                if step == "check":
                    self.checks.append((BAD, self.error))
        elif e.kind == "result":
            self.result = e.text
            if e.text == "failed" and self.current:
                self.row(self.current).state = FAILED
            self.current = None
        else:
            self.raw.append(e.text)
        return e

    def row(self, step: str) -> Row:
        if step not in self.rows:
            self.rows[step] = Row(step)
        return self.rows[step]

    def finish(self, code: int, log: Path | None = None) -> None:
        """The child ended with code: fill in what its own lines did not say (a crash prints no result line)."""
        if self.result is None:
            self.result = {EXIT_OK: "ok", EXIT_CANCELLED: "cancelled"}.get(code, "failed")
        if self.current:
            self.row(self.current).state = CANCELLED if self.result == "cancelled" else FAILED
            self.current = None
        if self.result == "failed" and self.error is None:
            if code == EXIT_USAGE:
                self.error = "oot2 prepare refused the command the window gave it. This is a bug in the window."
            else:
                self.error = f"oot2 prepare stopped without saying why (exit code {code})."
        if self.result == "failed" and self.error and self.error.startswith("Internal error"):
            self.error = (f"An unexpected error stopped the step \"{label(self.error_step)}\". This is a bug in "
                          f"oot2 prepare, not a problem with your files: {self.error}")
        if self.result == "failed" and log is not None:
            self.error += f" The whole output is in {log}."

    def summary(self) -> str:
        """One line on the run's outcome, for the window's status."""
        if self.result == "ok":
            todo = [r for r in self.rows.values() if r.state == WOULD]
            if todo:   # the step list beside the status names them
                return f"The check passed. Prepare would run {len(todo)} of {len(self.rows)} steps."
            return "Done: every step is finished."
        if self.result == "cancelled":
            return "Stopped. The finished steps are kept: Prepare resumes from the first unfinished one."
        if self.result == "failed":
            if self.error_step and self.error and not self.error.startswith("An unexpected error"):
                return f"{label(self.error_step)} failed. {self.error}"
            return self.error or "Failed."
        if self.current:
            return f"{label(self.current)}..."
        return ""


FRACTION = re.compile(r"([\d,]+(?:\.\d+)?) of ([\d,]+(?:\.\d+)?)")


def fraction(text: str) -> float | None:
    """The share a progress line reports ("41,200 of 118,000 (...)", "2.3 of 23.9 GiB"), or None."""
    m = FRACTION.search(text or "")
    if not m:
        return None
    done, total = (float(g.replace(",", "")) for g in m.groups())
    return min(done / total, 1.0) if total > 0 else None


def progress(m: Model) -> tuple[float, str]:
    """The bar's share of the whole run (finished steps, plus the running step's own share) and its label."""
    total = len(m.rows) or 1
    finished = sum(r.state in (DONE, KEPT) for r in m.rows.values())
    if m.current and m.current in m.rows:
        row = m.rows[m.current]
        text = f"Step {list(m.rows).index(m.current) + 1} of {total}: {row.label}"
        if row.detail:
            text += f", {row.detail}"
        if m.elapsed >= 60:
            text += f", {m.elapsed / 60:.0f} min since the start"
        return (finished + (fraction(row.detail) or 0.0)) / total, text
    if m.result == "ok" and finished == total:
        return 1.0, "Every step is finished."
    return finished / total, ""


def status(m: Model, folder: "Folder", has_dump: bool, running: bool) -> tuple[str, str]:
    """The window's status line and its tone: ready, bad, busy, notice, pending or unknown."""
    if running or (m.current and m.result is None):
        return "busy", m.summary() or "Starting oot2 prepare..."
    if m.result == "failed":
        return "bad", m.summary()
    if m.result == "cancelled":
        return "notice", m.summary()
    if m.result == "ok":
        if folder.kind == "ready" and not any(r.state == WOULD for r in m.rows.values()):
            return "ready", "The game is prepared. Play starts it."
        return "ready", m.summary()
    if folder.kind in ("file", "other", "damaged"):
        return "bad", folder.message
    if folder.kind == "ready":
        return "ready", "The game is prepared. Play starts it."
    if folder.kind == "stopped" and folder.finished:
        return "pending", "A stopped run waits here. Resume continues it."
    if not has_dump and not folder.dump:
        return "unknown", "No dump chosen."
    if folder.kind == "none":
        return "unknown", "No install folder chosen."
    return "pending", "Ready: check this PC, then prepare."


@dataclass
class Folder:
    """What an install folder holds before a run."""
    kind: str                         # none, new, other, stopped, ready, damaged, file
    message: str
    dump: str | None = None           # the dump the last run recorded
    finished: tuple[str, ...] = ()


def look(dest: str | os.PathLike | None) -> Folder:
    if not dest:
        return Folder("none", "Choose an install folder: an empty or new one, about 38 GiB.")
    dest = Path(dest)
    if dest.exists() and not dest.is_dir():
        return Folder("file", f"{dest} is a file. Choose a folder.")
    state_file = dest / STATE_NAME
    if not state_file.is_file():
        if dest.is_dir() and any(p.name != SCRATCH and not p.name.startswith(STATE_NAME) for p in dest.iterdir()):
            return Folder("other", "This folder holds other files. Choose an empty or new folder, so prepare never "
                                   "mixes its files with yours.")
        return Folder("new", "A first run: every step runs here.")
    statefile = prepare_state()
    try:
        data = statefile.load(dest)
    except statefile.StateError as e:
        return Folder("damaged", plain(str(e)))
    finished = tuple(data["steps"])
    dump = (data.get("inputs", {}).get("dump") or {}).get("path")
    if "link" in finished and (dest / EXE_NAME).is_file():
        return Folder("ready", f"The game is prepared here ({EXE_NAME}). Prepare again checks it and rebuilds only "
                               "what changed.", dump, finished)
    if not finished:
        return Folder("stopped", "Checked here before; no step has finished yet.", dump, finished)
    return Folder("stopped", f"A stopped run is here, {len(finished)} step{'s' * (len(finished) != 1)} finished. "
                             "Prepare resumes it.", dump, finished)


def prepare_state():
    """tools/prepare/state.py, the reader of the state file prepare writes."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from prepare import state
    return state


@dataclass
class Form:
    dump: str = ""
    dest: str = ""
    skip_pipelines: bool = False

    def problems(self) -> list[str]:
        """What stops a run before it starts, in the window's words. prepare's own check says the rest."""
        folder = look(self.dest.strip() or None)
        out = []
        if folder.kind in ("none", "file", "other", "damaged"):
            out.append(folder.message)
        dump = self.dump.strip()
        if dump:
            if not Path(dump).is_file():
                out.append(f"The dump {dump} is not a file.")
        elif not folder.dump:
            out.append("Choose your dump: your own .nup of the game, base program and update 1.1.1.")
        return out

    def command(self, check: bool = False, python: str | None = None) -> list[str]:
        # -I ignores the PYTHON* variables Process sets, so UTF-8 output (a path with an accent) is asked for by -X.
        cmd = [python or child_python(), "-I", "-X", "utf8", str(OOT2), "prepare", "--dest",
               str(Path(self.dest.strip()))]
        if self.dump.strip():
            cmd += ["--dump", str(Path(self.dump.strip()))]
        if check:
            cmd.append("--check")
        if self.skip_pipelines:
            cmd.append("--skip-pipelines")
        return cmd


def child_python(exe: str | None = None) -> str:
    """The interpreter for the child: this one, except that pythonw.exe (a window, no console) is swapped for the
    python.exe beside it, since Ctrl+Break reaches only a process with a console."""
    exe = Path(exe or sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").is_file():
        return str(exe.parent / "python.exe")
    return str(exe)


def profile_dir() -> Path:
    """Where the window keeps its log and settings: OOT2_SETUP_HOME, else %LOCALAPPDATA%\\oot2-native."""
    if os.environ.get("OOT2_SETUP_HOME"):
        return Path(os.environ["OOT2_SETUP_HOME"])
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "oot2-native"


def log_path(now: datetime.datetime | None = None) -> Path:
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    return profile_dir() / "logs" / f"prepare-{stamp}.log"


class Settings:
    """The dump, the install folder and the checkbox of the last run, in profile_dir()/setup.json. Never anything
    from the game: paths the player typed or picked, and one flag."""
    FIELDS = ("dump", "dest", "skip_pipelines")

    def __init__(self, path: Path | None = None):
        self.path = path or profile_dir() / "setup.json"

    def load(self) -> Form:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return Form()
        if not isinstance(data, dict):
            return Form()
        return Form(dump=str(data.get("dump") or ""), dest=str(data.get("dest") or ""),
                    skip_pipelines=data.get("skip_pipelines") is True)

    def save(self, form: Form) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"dump": form.dump, "dest": form.dest,
                                       "skip_pipelines": form.skip_pipelines}, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass   # remembering is a convenience; a read-only profile must not stop a run


# ---- the prepare child process -------------------------------------------------------------------------------------

CREATE_NEW_PROCESS_GROUP = 0x200
CREATE_NO_WINDOW = 0x08000000

# Run by a short helper process: it attaches to the child's console (the window may have none) and sends Ctrl+Break
# to the child's process group, which tools/oot2.py ends like Ctrl+C, keeping the finished steps.
_BREAK = """
import ctypes, sys, time
k = ctypes.windll.kernel32
pid = int(sys.argv[1])
k.FreeConsole()
if k.AttachConsole(pid):
    k.SetConsoleCtrlHandler(None, True)
    k.GenerateConsoleCtrlEvent(1, pid)
    time.sleep(1)
    k.FreeConsole()
else:
    sys.exit(1)
"""


def send_break(proc: subprocess.Popen, python: str | None = None) -> bool:
    """Ctrl+Break to the process group of proc (started with CREATE_NEW_PROCESS_GROUP); elsewhere SIGTERM to its
    session, which tools/oot2.py handles the same way."""
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            return True
        except OSError:
            return False
    helper = [child_python(python), "-I", "-c", _BREAK, str(proc.pid)]
    r = subprocess.run(helper, creationflags=CREATE_NO_WINDOW, capture_output=True)
    return r.returncode == 0


class Process:
    """A child whose output lines go to self.lines (a queue) and to the log file.

    poll() returns the exit code once the child ended and its output is drained, else None."""

    def __init__(self, command: list[str], log: Path | None = None, cwd: str | None = None):
        self.command = command
        self.lines: queue.Queue[str] = queue.Queue()
        self.log = log
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        if os.name == "nt":
            group = {"creationflags": CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW}
        else:
            group = {"start_new_session": True}
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, **group)
        self.cancelled_at: float | None = None
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self) -> None:
        log = open(self.log, "a", encoding="utf-8", newline="\n") if self.log is not None else None
        try:
            if log:
                log.write(f"$ {subprocess.list2cmdline(self.command)}\n")
            for raw in iter(self.proc.stdout.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if log:
                    log.write(text + "\n")
                    log.flush()
                self.lines.put(text)
        finally:
            self.proc.stdout.close()
            if log:
                log.close()

    def cancel(self) -> bool:
        if self.cancelled_at is None and self.proc.poll() is None:
            self.cancelled_at = time.monotonic()
            return send_break(self.proc)
        return False

    def poll(self) -> int | None:
        code = self.proc.poll()
        if code is not None and not self.reader.is_alive() and self.lines.empty():
            return code
        return None

    def kill(self) -> None:
        """End the child and everything it started (the compiler, the translator), at once. The state file stays
        whole: prepare writes it only between steps, atomically."""
        if self.proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], creationflags=CREATE_NO_WINDOW,
                           capture_output=True)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
        self.proc.kill()


def game_command(dest: str | os.PathLike) -> list[str]:
    """The prepared program, which finds the game in its own folder."""
    return [str(Path(dest) / EXE_NAME)]


def game_exit(code: int) -> str:
    """The game's exit code in plain words."""
    words = GAME_EXITS.get(code, "The game stopped unexpectedly.")
    return words if code == 0 else f"{words} (exit code {code}; census.tsv in the install folder lists what ran)"
