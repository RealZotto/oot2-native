# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The setup window (tkinter): tools/oot2_setup.pyw opens it, or `python tools/installer/window.py`.

It draws what logic.py decides and nothing else. A column of actions on the left (select your dump, install folder,
check this PC, prepare, play, settings, quit) and, on the right, the steps of the run and the machine check's lines;
along the bottom, one coloured status line, a notice for the check's warnings, and the log. "Check this PC" runs
`oot2 prepare --check`, "Prepare" runs every step (it reads "Resume" over a stopped run and turns into "Cancel" while
one runs), and "Play" starts the prepared oot2.exe. Each run's whole output goes to a log in the player's profile.

    --selftest [--shot DIR]   draw each screen from sample lines (no dump, no child process) and close; with --shot,
                              a PNG of each screen in DIR (Windows)

The layout follows the structure of Dusklight's launcher, which the maintainer holds as the reference for this
project's menus: a dark warm screen, a title over a list of large capitalised actions whose focused entry lights up
with a fading accent, the state in one coloured line at the bottom left, notices in a bordered box at the bottom centre,
and progress as a label over a bar. Nothing of Dusklight's is copied: no code, art, font or text; the colours and sizes
here are this file's own.

tkinter because the bundle's Python carries Tcl/Tk for it (as TotK Recomp's installer bundle does), the window is a
front end over a Python command line, and it needs no GPU: it must work on a PC whose Vulkan check fails.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "installer"

import tkinter as tk  # noqa: E402
from tkinter import filedialog, font as tkfont, messagebox  # noqa: E402

from installer import logic  # noqa: E402
from installer.logic import (BAD, CANCELLED, DONE, FAILED, KEPT, OK, PENDING, RUNNING, STATE_WORDS, WARN,  # noqa: E402
                             WOULD, Form, Model, Process, Settings)

POLL_MS = 100
STOP_AFTER = 20.0          # seconds after Cancel before "Stop now" is offered

# A dark warm palette with one light accent.
BG = "#14110f"
PANEL = "#1d1916"
LINE = "#3a322b"
TEXT = "#f3eee8"
MUTED = "#a09a93"
DIM = "#6c665f"
ACCENT = "#f6d97c"
READY = "#c9e98f"
ERROR = "#f6b6b6"
NOTICE = "#f8cd98"

GLYPH = {PENDING: "\u25cb", RUNNING: "\u25b6", DONE: "\u2714", KEPT: "\u2714", WOULD: "\u25cb", FAILED: "\u2716",
         CANCELLED: "\u25a0", OK: "\u2714", WARN: "\u26a0", BAD: "\u2716"}
COLOUR = {PENDING: DIM, RUNNING: TEXT, DONE: READY, KEPT: READY, WOULD: MUTED, FAILED: ERROR, CANCELLED: NOTICE,
          OK: READY, WARN: NOTICE, BAD: ERROR}
TONE = {"ready": (READY, "\u2714"), "bad": (ERROR, "\u2716"), "busy": (TEXT, "\u25b6"), "notice": (NOTICE, "\u26a0"),
        "pending": (ACCENT, "\u25cf"), "unknown": (MUTED, "\u25cb")}


def dpi_aware() -> None:
    """System DPI awareness before the first window, so Windows does not stretch a blurry one at 125% and above."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def fix_tcl_paths() -> None:
    """The embeddable Python of the bundle keeps Tcl/Tk next to python.exe (python/tcl); point Tcl at it."""
    tcl = Path(sys.executable).resolve().parent / "tcl"
    if tcl.is_dir():
        for sub in sorted(tcl.iterdir()):
            for prefix, var in (("tcl", "TCL_LIBRARY"), ("tk", "TK_LIBRARY")):
                if sub.is_dir() and sub.name.startswith(prefix) and sub.name[len(prefix):][:1].isdigit():
                    os.environ.setdefault(var, str(sub))


def mix(a: str, b: str, t: float) -> str:
    """The colour t of the way from a to b."""
    ca, cb = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (a, b))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def fade(root: tk.Misc, width: int, height: int, colour: str, back: str) -> tk.PhotoImage:
    """A band that fades from colour on the left to back on the right."""
    img = tk.PhotoImage(master=root, width=width, height=height)
    row = " ".join(mix(colour, back, (x / max(width - 1, 1)) ** 1.6) for x in range(width))
    img.put("{" + row + "}", to=(0, 0, width, height))
    return img


class Action(tk.Canvas):
    """One entry of the action list: capitalised text that lights up with a fading band when hovered or focused,
    dimmed when disabled. Enter and Space press it, as a click does; the arrow keys move along the list."""

    def __init__(self, master, app: "App", text: str, command, width: int = 360, height: int = 46):
        super().__init__(master, width=width, height=height, bg=BG, highlightthickness=0, takefocus=1,
                         cursor="hand2")
        self.app, self.command, self.enabled, self.lit = app, command, True, False
        self.band = self.create_image(0, 0, image=app.band, anchor="nw", state="hidden")
        self.label = self.create_text(16, height // 2, text=text.upper(), anchor="w", font=app.f_action, fill=TEXT)
        for ev, fn in (("<Enter>", lambda e: self.light(True)),
                       ("<Leave>", lambda e: self.light(self.focus_get() is self)),
                       ("<FocusIn>", lambda e: self.light(True)), ("<FocusOut>", lambda e: self.light(False)),
                       ("<Button-1>", lambda e: self.press()), ("<Return>", lambda e: self.press()),
                       ("<space>", lambda e: self.press()), ("<Down>", lambda e: self.app.move_focus(self, 1)),
                       ("<Up>", lambda e: self.app.move_focus(self, -1))):
            self.bind(ev, fn)

    def set(self, text: str | None = None, enabled: bool | None = None) -> None:
        if text is not None:
            self.itemconfigure(self.label, text=text.upper())
        if enabled is not None:
            self.enabled = enabled
            self.configure(cursor="hand2" if enabled else "arrow", takefocus=1 if enabled else 0)
        self.light(self.lit)

    def light(self, on: bool) -> None:
        self.lit = on and self.enabled
        self.itemconfigure(self.band, state="normal" if self.lit else "hidden")
        self.itemconfigure(self.label, fill=BG if self.lit else (TEXT if self.enabled else DIM))

    def press(self) -> None:
        if self.enabled:
            self.command()


class App:
    def __init__(self, root: tk.Tk, settings: Settings | None = None):
        self.root = root
        self.settings = settings or Settings()
        self.model = Model()
        self.process: Process | None = None
        self.log: Path | None = None
        self.game: subprocess.Popen | None = None
        self.closing = False
        self.view = "progress"
        form = self.settings.load()
        self.dump = tk.StringVar(value=form.dump)
        self.dest = tk.StringVar(value=form.dest)
        self.skip = tk.BooleanVar(value=form.skip_pipelines)
        self.message: tuple[str, str] | None = None   # (tone, text) the window says itself, over the model's status
        self.fonts()
        self.build()
        self.dest.trace_add("write", lambda *_: self.render())
        self.dump.trace_add("write", lambda *_: self.render())
        self.render()
        root.protocol("WM_DELETE_WINDOW", self.close)

    # ---- layout --------------------------------------------------------------------------------------------------

    def fonts(self) -> None:
        families = set(tkfont.families(self.root))
        body = "Segoe UI" if "Segoe UI" in families else "TkDefaultFont"
        title = next((f for f in ("Georgia", "Cambria", "Times New Roman") if f in families), body)
        semi = "Segoe UI Semibold" if "Segoe UI Semibold" in families else body
        self.f_eyebrow = (title, 13, "bold")
        self.f_title = (title, 30, "bold")
        self.f_action = (semi, 15)
        self.f_head = (semi, 9)
        self.f_body = (body, 10)
        self.f_small = (body, 9)
        self.f_status = (body, 11)
        self.band = fade(self.root, 360, 46, ACCENT, BG)

    def build(self) -> None:
        r = self.root
        r.title("oot2-native setup")
        r.configure(bg=BG)
        r.minsize(980, 620)
        r.columnconfigure(1, weight=1)
        r.rowconfigure(0, weight=1)

        left = tk.Frame(r, bg=BG)
        left.grid(row=0, column=0, sticky="nsw", padx=(48, 24), pady=(40, 12))
        tk.Label(left, text="OOT2-NATIVE", font=self.f_eyebrow, fg=MUTED, bg=BG).pack(anchor="w")
        tk.Label(left, text="Ocarina of Time", font=self.f_title, fg=ACCENT, bg=BG).pack(anchor="w")
        tk.Label(left, text="for this PC, prepared from your own dump. Nothing from it is uploaded.",
                 font=self.f_body, fg=MUTED, bg=BG, wraplength=360, justify="left").pack(anchor="w", pady=(0, 22))
        self.actions = {}
        for key, text, cmd in (("dump", "Select your dump", self.pick_dump),
                               ("dest", "Install folder", self.pick_dest),
                               ("check", "Check this PC", lambda: self.start(check=True)),
                               ("prepare", "Prepare", self.prepare_or_cancel),
                               ("play", "Play", self.play),
                               ("settings", "Settings", self.toggle_settings),
                               ("quit", "Quit", self.close)):
            a = Action(left, self, text, cmd)
            a.pack(anchor="w", pady=2)
            self.actions[key] = a
        self.order = list(self.actions.values())
        info = tk.Frame(left, bg=BG)
        info.pack(anchor="w", fill="x", pady=(20, 0))
        self.dump_info = tk.Label(info, font=self.f_small, fg=MUTED, bg=BG, wraplength=360, justify="left")
        self.dump_info.pack(anchor="w")
        self.dest_info = tk.Label(info, font=self.f_small, fg=MUTED, bg=BG, wraplength=360, justify="left")
        self.dest_info.pack(anchor="w", pady=(4, 0))

        self.right = tk.Frame(r, bg=PANEL, highlightthickness=1, highlightbackground=LINE)
        self.right.grid(row=0, column=1, sticky="nsew", padx=(0, 40), pady=(40, 12))
        self.right.columnconfigure(0, weight=1)
        self.right.rowconfigure(0, weight=1)
        self.build_progress(self.right)
        self.build_settings(self.right)

        bottom = tk.Frame(r, bg=BG)
        bottom.grid(row=1, column=0, columnspan=2, sticky="we", padx=(48, 40), pady=(4, 22))
        bottom.columnconfigure(1, weight=1)
        self.status_icon = tk.Label(bottom, font=self.f_status, bg=BG)
        self.status_icon.grid(row=0, column=0, sticky="nw")
        self.status = tk.Label(bottom, font=self.f_status, bg=BG, fg=TEXT, wraplength=600, justify="left", anchor="w")
        self.status.grid(row=0, column=1, sticky="nw", padx=(6, 16))
        self.toast = tk.Frame(bottom, bg=PANEL, highlightthickness=1, highlightbackground=ACCENT)
        self.toast_title = tk.Label(self.toast, font=self.f_head, fg=ACCENT, bg=PANEL, anchor="w")
        self.toast_title.pack(anchor="w", padx=14, pady=(8, 0))
        self.toast_text = tk.Label(self.toast, font=self.f_small, fg=TEXT, bg=PANEL, wraplength=360, justify="left",
                                   anchor="w")
        self.toast_text.pack(anchor="w", padx=14, pady=(2, 9))
        self.log_link = tk.Label(bottom, text="Open the log", font=self.f_small, fg=MUTED, bg=BG, cursor="hand2")
        self.log_link.grid(row=0, column=3, sticky="ne")
        self.log_link.bind("<Button-1>", lambda e: self.open_log())

    def heading(self, parent, text: str, row: int, pad: int = 16) -> None:
        tk.Label(parent, text=text.upper(), font=self.f_head, fg=ACCENT, bg=PANEL).grid(
            row=row, column=0, sticky="w", padx=20, pady=(pad, 4))

    def text_box(self, parent, height: int) -> tk.Text:
        t = tk.Text(parent, height=height, wrap="word", relief="flat", borderwidth=0, bg=PANEL, fg=TEXT,
                    font=self.f_body, highlightthickness=0, cursor="arrow", spacing1=2, spacing3=2)
        for mark, colour in COLOUR.items():
            t.tag_configure(mark, foreground=colour)
        t.tag_configure("muted", foreground=MUTED)
        t.tag_configure("detail", foreground=MUTED, lmargin1=26, lmargin2=26)
        return t

    def build_progress(self, panel) -> None:
        f = self.progress_view = tk.Frame(panel, bg=PANEL)
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=3)
        f.rowconfigure(5, weight=2)
        self.heading(f, "Steps", 0, pad=18)
        self.steps = self.text_box(f, len(logic.STEP_ORDER) + 1)   # every step, and the running one's detail line
        self.steps.grid(row=1, column=0, sticky="nsew", padx=20)
        self.bar_label = tk.Label(f, font=self.f_small, fg=MUTED, bg=PANEL, anchor="w")
        self.bar_label.grid(row=2, column=0, sticky="we", padx=20, pady=(8, 2))
        self.bar = tk.Canvas(f, height=6, bg=LINE, highlightthickness=0)
        self.bar.grid(row=3, column=0, sticky="we", padx=20)
        self.bar_fill = self.bar.create_rectangle(0, 0, 0, 6, fill=ACCENT, width=0)
        self.heading(f, "This PC", 4)
        pc = tk.Frame(f, bg=PANEL)
        pc.grid(row=5, column=0, sticky="nsew", padx=(20, 8), pady=(0, 16))
        pc.columnconfigure(0, weight=1)
        pc.rowconfigure(0, weight=1)
        self.checks = self.text_box(pc, 6)
        scroll = tk.Scrollbar(pc, command=self.checks.yview, bg=PANEL, troughcolor=PANEL, relief="flat", width=10)
        self.checks.configure(yscrollcommand=scroll.set)
        self.checks.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        f.grid(row=0, column=0, sticky="nsew")

    def build_settings(self, panel) -> None:
        f = self.settings_view = tk.Frame(panel, bg=PANEL)
        f.columnconfigure(0, weight=1)
        self.heading(f, "Settings", 0, pad=18)
        self.skip_box = tk.Checkbutton(
            f, variable=self.skip, text="Leave out the pipeline step", font=self.f_status, fg=TEXT, bg=PANEL,
            activebackground=PANEL, activeforeground=ACCENT, selectcolor=BG, highlightthickness=0, anchor="w",
            command=self.save_settings)
        self.skip_box.grid(row=1, column=0, sticky="w", padx=16, pady=(6, 0))
        tk.Label(f, text="The pipeline step builds every GPU pipeline the game will use, so it never waits for one "
                         "while you play. It takes hours on a first run. Left out, the game builds each pipeline the "
                         "first time it meets it, with a stutter.", font=self.f_body, fg=MUTED, bg=PANEL,
                 wraplength=460, justify="left").grid(row=2, column=0, sticky="w", padx=44, pady=(2, 18))
        self.heading(f, "Where things are kept", 3)
        tk.Label(f, text=f"Logs and the paths you chose: {logic.profile_dir()}. The prepared game stays in the install "
                         "folder; keep it to yourself.", font=self.f_body, fg=MUTED, bg=PANEL, wraplength=460,
                 justify="left").grid(row=4, column=0, sticky="w", padx=20)

    # ---- the player's choices ------------------------------------------------------------------------------------

    def form(self) -> Form:
        return Form(self.dump.get(), self.dest.get(), bool(self.skip.get()))

    def pick_dump(self) -> None:
        start = Path(self.dump.get()).parent if self.dump.get() else None
        path = filedialog.askopenfilename(parent=self.root, title="Your dump of the game",
                                          initialdir=str(start) if start and start.is_dir() else None,
                                          filetypes=[("Switch 2 dump", "*.nup"), ("All files", "*.*")])
        if path:
            self.message = None
            self.dump.set(str(Path(path)))
            self.settings.save(self.form())

    def pick_dest(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="Install folder (empty or new, about 38 GiB)",
                                       mustexist=False)
        if path:
            self.message = None
            self.dest.set(str(Path(path)))
            self.settings.save(self.form())

    def save_settings(self) -> None:
        self.settings.save(self.form())

    def toggle_settings(self) -> None:
        self.view = "progress" if self.view == "settings" else "settings"
        self.render()

    def move_focus(self, current: Action, step: int) -> None:
        live = [a for a in self.order if a.enabled]
        if current in live:
            live[(live.index(current) + step) % len(live)].focus_set()

    # ---- runs ----------------------------------------------------------------------------------------------------

    def start(self, check: bool) -> None:
        form = self.form()
        problems = form.problems()
        if problems:
            self.message = ("bad", " ".join(problems))
            self.render()
            return
        self.settings.save(form)
        self.model = Model()
        self.message = None
        self.view = "progress"
        self.log = logic.log_path()
        try:
            self.process = Process(form.command(check=check), log=self.log)
        except OSError as e:
            self.process = None
            self.message = ("bad", f"oot2 prepare could not start: {e}.")
        self.render()
        if self.process is not None:
            self.root.after(POLL_MS, self.poll)

    def prepare_or_cancel(self) -> None:
        if self.process is None:
            self.start(check=False)
        else:
            self.cancel()

    def poll(self) -> None:
        p = self.process
        if p is None:
            return
        while True:
            try:
                self.model.feed(p.lines.get_nowait())
            except Exception:
                break
        code = p.poll()
        if code is not None:
            self.model.finish(code, self.log)
            self.process = None
            self.message = None
            if self.closing:
                self.root.destroy()
                return
        self.render()
        if self.process is not None:
            self.root.after(POLL_MS, self.poll)

    def cancel(self) -> None:
        p = self.process
        if p is None:
            return
        if p.cancelled_at is None:
            p.cancel()
            self.message = ("busy", "Stopping after the current piece of work; the finished steps are kept...")
        elif time.monotonic() - p.cancelled_at > STOP_AFTER:
            p.kill()
        self.render()

    def play(self) -> None:
        dest = self.dest.get().strip()
        out = logic.profile_dir() / "logs" / f"game-{time.strftime('%Y%m%d-%H%M%S')}.log"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "wb") as f:
                self.game = subprocess.Popen(logic.game_command(dest), cwd=dest, stdin=subprocess.DEVNULL,
                                             stdout=f, stderr=subprocess.STDOUT)
        except OSError as e:
            self.game = None
            self.message = ("bad", f"The game could not start: {e}.")
            self.render()
            return
        self.message = ("busy", "The game is running.")
        self.render()
        self.root.after(500, self.watch_game)

    def watch_game(self) -> None:
        if self.game is None:
            return
        code = self.game.poll()
        if code is None:
            self.root.after(500, self.watch_game)
            return
        self.game = None
        self.message = ("ready" if code == 0 else "bad", logic.game_exit(code))
        self.render()

    def open_log(self) -> None:
        folder = self.log.parent if self.log else logic.profile_dir() / "logs"
        if os.name == "nt" and folder.is_dir():
            os.startfile(folder)   # the folder, so the player sees every log

    def close(self) -> None:
        if self.process is None:
            self.root.destroy()
            return
        if messagebox.askyesno("oot2-native setup", "A run is in progress. Stop it and close? The finished steps "
                                                    "are kept.", parent=self.root):
            self.closing = True
            self.cancel()

    # ---- drawing -------------------------------------------------------------------------------------------------

    def render(self) -> None:
        m = self.model
        dest = self.dest.get().strip()
        folder = logic.look(dest or None)
        running = self.process is not None

        a = self.actions
        for key in ("dump", "dest", "check"):
            a[key].set(enabled=not running)
        if running:
            late = self.process.cancelled_at is not None and time.monotonic() - self.process.cancelled_at > STOP_AFTER
            a["prepare"].set("Stop now" if late else "Cancel", enabled=True)
        else:
            a["prepare"].set({"stopped": "Resume" if folder.finished else "Prepare",
                              "ready": "Prepare again"}.get(folder.kind, "Prepare"), enabled=True)
        a["play"].set(enabled=folder.kind == "ready" and not running and self.game is None)
        a["settings"].set("Back" if self.view == "settings" else "Settings", enabled=not running)

        dump = self.dump.get().strip()
        if dump:
            self.dump_info.configure(text=f"Dump:  {Path(dump).name}")
        elif folder.dump:
            self.dump_info.configure(text=f"Dump:  {Path(folder.dump).name} (from the last run)")
        else:
            self.dump_info.configure(text="Dump:  none chosen")
        self.dest_info.configure(text=f"Folder:  {dest}\n{folder.message}" if dest else f"Folder:  {folder.message}")

        if self.view == "settings":
            self.progress_view.grid_remove()
            self.settings_view.grid(row=0, column=0, sticky="nsew")
        else:
            self.settings_view.grid_remove()
            self.progress_view.grid()
        self.draw_steps(m)
        self.draw_checks(m, dest)
        self.draw_bar(m)
        tone, text = self.message or logic.status(m, folder, bool(dump), running)
        colour, icon = TONE[tone]
        self.status_icon.configure(text=icon, fg=colour)
        self.status.configure(text=text, fg=colour)
        self.draw_toast(m)

    def draw_steps(self, m: Model) -> None:
        t = self.steps
        t.configure(state="normal")
        t.delete("1.0", "end")
        for row in m.rows.values():
            when = "" if row.seconds is None else (f"{row.seconds:.0f} s" if row.seconds < 120
                                                    else f"{row.seconds / 60:.0f} min")
            t.insert("end", f"{GLYPH[row.state]}  ", (row.state,))
            t.insert("end", row.label, (row.state,) if row.state in (RUNNING, FAILED) else ())
            t.insert("end", f"   {STATE_WORDS[row.state]}" + (f", {when}" if when else "") + "\n", ("muted",))
            if row.state == RUNNING and row.detail:
                t.insert("end", row.detail + "\n", ("detail",))
        t.configure(state="disabled")

    def draw_checks(self, m: Model, dest: str) -> None:
        t = self.checks
        t.configure(state="normal")
        t.delete("1.0", "end")
        if not m.checks:
            t.insert("end", "Check this PC runs the check step alone: disk, memory, CPU, build tools, the GPU and "
                            "its Vulkan features, and your dump.", ("muted",))
        for mark, text in m.checks:
            t.insert("end", f"{GLYPH[mark]}  ", (mark,))
            t.insert("end", logic.compact(text, dest) + "\n")
        t.configure(state="disabled")

    def draw_bar(self, m: Model) -> None:
        share, text = logic.progress(m)
        self.bar_label.configure(text=text)
        self.bar.update_idletasks()
        self.bar.coords(self.bar_fill, 0, 0, int(self.bar.winfo_width() * share), 6)

    def draw_toast(self, m: Model) -> None:
        if not m.warnings:
            self.toast.grid_remove()
            return
        n = len(m.warnings)
        self.toast_title.configure(text=("1 warning from the check" if n == 1 else f"{n} warnings from the check")
                                   .upper())
        self.toast_text.configure(text=logic.plain(m.warnings[0]) + ("" if n == 1 else " More in the log."))
        self.toast.grid(row=0, column=2, sticky="n", padx=(0, 24))


# ---- self-test -------------------------------------------------------------------------------------------------------

# Lines in prepare's format (tools/prepare/runner.py, check.py) with made-up paths and figures; no game file is read.
SAMPLE_DEST = r"D:\Games\oot2"
SAMPLE_DUMP = r"D:\Dumps\my-dump.nup"
SAMPLE_CHECK = [
    f"[    0.0 s] oot2 prepare: first run in {SAMPLE_DEST}; steps: check, extract, font, translate, ledger, "
    "pipelines, modules, lift, compile, link",
    "[    0.0 s] check: started (runs every time)",
    f"[    0.0 s] check: disk of {SAMPLE_DEST}: 412.6 GiB free, 38.0 GiB needed",
    "[    1.9 s] check: CPU AMD Ryzen 5 5600X 6-Core Processor: x86-64-v3",
    "[    1.9 s] check: memory: 15.9 GiB installed, 9.4 GiB free",
    r"[    2.4 s] check: build tools: C++ driver C:\oot2\llvm\bin\clang++.exe, Ninja C:\oot2\bin\ninja.exe, "
    r"Vulkan headers 1.4.357 and Volk in C:\oot2\vulkan\Include, glslang C:\oot2\vulkan\Bin\glslang.exe",
    "[    2.4 s] check: Vulkan device: NVIDIA GeForce RTX 3060 Ti (discrete), API 1.4.351, driver 617.14",
    "[    2.4 s] check: NVIDIA GeForce RTX 3060 Ti: device-local memory 7.84 GiB, budget 7.09 GiB",
    "[    2.4 s] check: NVIDIA GeForce RTX 3060 Ti: has 32 of the 32 features the renderer will rely on",
    "[   95.0 s] check: dump my-dump.nup: 25,104,384,000 bytes, SHA-256 as recorded",
    "[   95.0 s] check: done in 95.0 s",
] + [f"[   95.0 s] {s}: would run (not done yet)" for s in ("extract", "font", "translate", "ledger", "pipelines",
                                                           "modules", "lift", "compile", "link")] + [
    "[   95.0 s] oot2 prepare: ok"]
SAMPLE_RUN = SAMPLE_CHECK[:10] + [
    "[   95.0 s] check: done in 95.0 s",
    "[   95.0 s] extract: started (not done yet)",
    "[  412.3 s] extract: done in 317.3 s",
    "[  412.3 s] font: started (not done yet)",
    "[  414.0 s] font: done in 1.7 s",
    "[  414.0 s] translate: started (not done yet)",
    "[  930.1 s] translate: done in 516.1 s",
    "[  930.1 s] ledger: kept (done, inputs unchanged)",
    "[  930.1 s] pipelines: started (not done yet)",
    "[ 1504.6 s] pipelines: 41,200 of 118,000 (3.1 per second)",
]
SAMPLE_FAILED = SAMPLE_CHECK[:2] + [
    f"[    0.0 s] check: disk of {SAMPLE_DEST}: 21.4 GiB free, 38.0 GiB needed",
    "[    0.0 s] warning: this PC has 8.0 GiB of memory, below the reference machine's 16 GB: prepare runs fewer jobs "
    "at once and takes longer, and the game has not been measured with less",
    f"oot2 prepare: step check: the disk of {SAMPLE_DEST} has 21.4 GiB free and prepare needs about 38.0 GiB there. "
    "Free some space, or choose a folder on another disk with --dest.",
    "[    0.0 s] oot2 prepare: failed",
]
SCREENS = (("start", [], "progress"), ("check", SAMPLE_CHECK, "progress"), ("running", SAMPLE_RUN, "progress"),
           ("failed", SAMPLE_FAILED, "progress"), ("settings", [], "settings"))


def selftest(shot_dir: str | None = None, keep_open_ms: int = 150) -> int:
    """Draw each screen from sample lines and close. Returns 0, or 1 when a screen's sample did not draw."""
    home = tempfile.mkdtemp(prefix="oot2-setup-selftest-")
    os.environ["OOT2_SETUP_HOME"] = home
    root = tk.Tk()
    app = App(root, Settings(Path(home) / "setup.json"))
    root.geometry("1180x760+40+40")
    failures = 0
    for name, lines, view in SCREENS:
        app.dump.set("" if name == "start" else SAMPLE_DUMP)
        app.dest.set("" if name == "start" else SAMPLE_DEST)
        app.model = Model()
        app.view = view
        for line in lines:
            app.model.feed(line)
        if name == "failed":
            app.model.finish(1, Path(home) / "logs" / "prepare-20261107-191500.log")
        root.update()
        app.render()
        lit = {"start": "dump", "check": "prepare", "running": "prepare", "failed": "dest", "settings": "settings"}
        for key, action in app.actions.items():
            action.light(key == lit[name])
        if name in ("check", "running", "failed") and app.steps.get("1.0", "end").count("\n") < 9:
            failures += 1
        root.update()
        root.lift()
        root.attributes("-topmost", True)
        root.update()
        root.after(keep_open_ms)
        root.update()
        if shot_dir:
            from installer import shot
            shot.save(Path(shot_dir) / f"setup-{name}.png", root.winfo_rootx(), root.winfo_rooty(),
                      root.winfo_width(), root.winfo_height())
    root.destroy()
    shutil.rmtree(home, ignore_errors=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="oot2-setup", description="The setup window over oot2 prepare.")
    ap.add_argument("--selftest", action="store_true", help="draw each screen from sample lines and close")
    ap.add_argument("--shot", help="with --selftest: write a PNG of each screen into this folder")
    args = ap.parse_args(argv)
    dpi_aware()
    fix_tcl_paths()
    if args.selftest:
        return selftest(args.shot)
    root = tk.Tk()
    root.geometry("1180x760")
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
