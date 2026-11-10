# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests of the M1 capture checker (tools/check_capture.py) on synthetic runs.

Every capture here is written by synth() below, in the format of
runtime/include/oot2/capture.h: presents at a stated rate per segment of host
time, one frame line per present, memory, count and cost lines once a second,
and the closing record. No game, no runtime: tests/test_runtime_capture.py
checks that the runtime writes the same format.

Each failing case has a passing counterpart that differs in one thing only, and
the runs are shaped so that a checker judging the wrong thing passes them: the
57 fps window sits between 60 fps stretches, so an average over the whole
capture or over the first 60 s is above 58; the memory over the limit at 60 s is
under it at every other sample, and the other way round.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import check_capture as cc  # noqa: E402

S = 1_000_000_000
GAME, WORKER = 0x8001, 0x10002
GIB = 1 << 30


def synth(*, segments=((30, 60.0), (60, 60.0), (5, 60.0)), title_segment: int | None = 1, misses=(),
          memory=None, end_code: int = 0, end_reason: str = "exit", closing: bool = True, census: bool = False,
          collect: bool | None = None, header_drop=(), extra=(), test_movie_blit: str | None = None) -> str:
    """A capture as text. segments: (seconds, fps) one after the other from t = 0; the title mark is the first present
    of segments[title_segment]. collect: None leaves the collect_variants header key out (a capture of an older
    writer), else writes it as 1 or 0. test_movie_blit: the header key of that test switch ("0", or "1 (...)"), left
    out when None (a capture of a build without the switch). misses: (counter index, seconds) pairs, each adding one.
    memory(t_ns) -> (working set, private) for the sample at t (default 2 and 3 GiB). extra: lines put before the
    closing record."""
    head = {"format": "1", "back_end": "null", "driver": "none (null back end)", "display_mode": "none (null back end)",
            "implicit_vulkan_layers": "none (null back end)", "ram_total": str(16 * GIB),
            "ram_available": str(9 * GIB), "census_mode": "1" if census else "0", "title_mark_present": "0"}
    if test_movie_blit is not None:
        head["test_movie_blit"] = test_movie_blit
    presents: list[tuple[int, int]] = []   # (seq, t)
    t_seg = 0
    title_seq = title_t = None
    for i, (secs, fps) in enumerate(segments):
        n = round(secs * fps)
        for k in range(n):
            seq = len(presents) + 1
            t = t_seg + round(k * S / fps)
            presents.append((seq, t))
            if i == title_segment and k == 0:
                title_seq, title_t = seq, t
        t_seg += secs * S
    if title_seq is not None:
        head["title_mark_present"] = str(title_seq)
    if collect is not None:
        head["collect_variants"] = "1" if collect else "0"
    for k in header_drop:
        head.pop(k)
    mem = memory or (lambda t: (2 * GIB, 3 * GIB))

    def counters_at(t: int) -> list[int]:
        c = [0] * len(cc.COUNTERS)
        for i, at in misses:
            if at * S <= t:
                c[i] += 1
        return c

    out = ["oot2-capture\t1"] + [f"header\t{k}\t{v}" for k, v in head.items()]
    out.append(f"thread\t{GAME:#x}\t0x7100001000\t44")
    out.append(f"thread\t{WORKER:#x}\t0x7100002000\t50")
    next_second = S
    due = 0
    prev_t = None
    for seq, t in presents:
        while t >= next_second:   # the sampler's lines once a second, with the newest present then
            ws, priv = mem(next_second)
            last = max((s for s, pt in presents if pt < next_second), default=0)
            out.append(f"memory\t{last}\t{next_second}\t{ws}\t{priv}\t{8 * GIB}\t{1000 + next_second // S}")
            out.append(f"count\t{last}\t{next_second}\t" + "\t".join(map(str, counters_at(next_second))))
            out.append(f"cost\t{last}\t{next_second}\t{last}\t{last * 20000}\t60000\t1000000")
            next_second += S
        due = seq - 1
        frame_ns = t - prev_t if prev_t is not None else 0
        prev_t = t
        out.append(f"present\t{seq}\t{t}\t{due}\t0\t1\t0\t{frame_ns}\t0")
        out.append(f"frame\t{seq}\t{GAME:#x}\t3\t1\t{GAME:#x}\t{GAME:#x}\t7000000\t8000000\t{WORKER:#x}\t20000000\t21000000")
        if seq == title_seq:
            out.append(f"title\t{seq}\t{t}\t" + "\t".join(map(str, counters_at(t))))
    out.extend(extra)
    if closing:
        last_seq, last_t = presents[-1]
        t_end = last_t + 1_000_000
        ws, priv = mem(t_end)
        fatals = 1 if end_code in (70, 71, 72) else 0
        c = counters_at(t_end)
        out.append(f"end\t{end_code}\t{end_reason}\t{last_seq}\t{t_end}\t{fatals}\t" + "\t".join(map(str, c)) +
                   f"\t{ws}\t{priv}\t{8 * GIB}\t5000\t{last_seq}\t{last_seq * 20000}\t60000\t1000000")
    return "\n".join(out) + "\n"


class CheckerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="oot2-capcheck-")
        self.addCleanup(self.tmp.cleanup)
        self.n = 0

    def write(self, text: str) -> Path:
        self.n += 1
        p = Path(self.tmp.name) / f"capture-{self.n}.tsv"
        p.write_bytes(text.encode("utf-8"))
        return p

    def verdict(self, text: str) -> cc.Verdict:
        return cc.check(cc.parse(self.write(text)))

    def failed(self, v: cc.Verdict) -> dict[str, str]:
        return {c.name: c.value for c in v.conditions if not c.ok}

    def test_a_clean_title_run_passes(self) -> None:
        v = self.verdict(synth())
        self.assertTrue(v.passed, self.failed(v))
        self.assertEqual(v.figures["late ticks in the window"], 0)
        self.assertEqual(v.figures["requested intervals"], {"1": 3600})

    def test_a_57_fps_window_fails_while_the_stretches_around_it_run_at_60(self) -> None:
        # Over the whole capture: (1800 + 3420 + 7200) / 210 s = 59.1 fps; over the first 60 s: 58.5. Both pass.
        v = self.verdict(synth(segments=((30, 60.0), (60, 57.0), (120, 60.0))))
        self.assertEqual(set(self.failed(v)), {"frame rate"})
        self.assertIn("57.00 fps (3420 presents)", self.failed(v)["frame rate"])
        # The counterpart: 58 fps in the window passes.
        self.assertTrue(self.verdict(synth(segments=((30, 60.0), (60, 58.0), (120, 60.0)))).passed)

    def test_the_window_is_the_60_s_after_the_mark_not_the_one_after_it(self) -> None:
        v = self.verdict(synth(segments=((30, 60.0), (60, 60.0), (60, 50.0))))
        self.assertTrue(v.passed, self.failed(v))

    def test_one_miss_of_any_kind_fails(self) -> None:
        for i, name in enumerate(cc.COUNTERS):
            with self.subTest(counter=name):
                v = self.verdict(synth(misses=[(i, 45)]))
                self.assertEqual(set(self.failed(v)), {name})
                self.assertEqual(self.failed(v)[name], "1")

    def test_a_miss_before_the_mark_fails_and_is_named_so(self) -> None:
        v = self.verdict(synth(misses=[(cc.COUNTERS.index("pipeline misses"), 5)]))
        self.assertEqual(self.failed(v), {"pipeline misses": "1 (1 before the mark)"})

    def test_a_named_fatal_fails(self) -> None:
        for code, reason in ((70, "fatal"), (71, "unimplemented"), (72, "guest-stop")):
            with self.subTest(code=code):
                v = self.verdict(synth(end_code=code, end_reason=reason))
                self.assertEqual(set(self.failed(v)), {"named fatals"})
                self.assertIn(f"{code}, {reason}", self.failed(v)["named fatals"])

    def test_a_census_mode_run_is_not_a_judged_run(self) -> None:
        v = self.verdict(synth(census=True, end_code=73, end_reason="census-limit"))
        self.assertEqual(set(self.failed(v)), {"judged run", "named fatals"})

    def test_a_collection_run_is_not_a_judged_run(self) -> None:
        # collect_variants skips draws that stop on a ledger answer: the picture is wrong, the run is no proof. The
        # mark alone fails it, whatever the counters say; with the key at 0 or absent the same run passes.
        v = self.verdict(synth(collect=True))
        self.assertEqual(set(self.failed(v)), {"judged run"})
        self.assertIn("collect_variants", self.failed(v)["judged run"])
        self.assertTrue(self.verdict(synth(collect=False)).passed)
        self.assertTrue(self.verdict(synth()).passed)
        # its skipped draws fail the counter as well, so both name it
        v = self.verdict(synth(collect=True, misses=[(cc.COUNTERS.index("skipped draws"), 5)]))
        self.assertEqual(set(self.failed(v)), {"judged run", "skipped draws"})

    def test_a_run_with_a_test_switch_on_is_refused_whatever_else_it_did(self) -> None:
        # a run that never reached a movie and missed nothing: only the header says that it had the switch on
        v = self.verdict(synth(test_movie_blit="1 (a test switch that changes the pictures of the movies: not a proof run)"))
        self.assertEqual(set(self.failed(v)), {"test settings"})
        self.assertIn("on: test_movie_blit = 1 (a test switch", self.failed(v)["test settings"])
        self.assertFalse(v.passed)
        # the counterparts: the key at 0, and a capture of a build that has no such switch, pass
        self.assertTrue(self.verdict(synth(test_movie_blit="0")).passed)
        self.assertTrue(self.verdict(synth()).passed)
        # the exit code of the tool is the verdict's
        for text, code, shown in ((synth(test_movie_blit="1 (x)"), 1, True), (synth(test_movie_blit="0"), 0, False)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cc.main([str(self.write(text))]), code)
            self.assertEqual("FAIL test settings: on: test_movie_blit = 1 (x)" in out.getvalue(), shown, out.getvalue())

    def test_a_working_set_over_5_gib_at_60_s_fails(self) -> None:
        def at_60(over: int):
            # 60 s after the mark is t = 90 s: the sample there holds `over`, every other sample 7 GiB before it and
            # 1 GiB after it, so a checker that reads the largest or the last sample judges something else.
            return lambda t: (over, 3 * GIB) if t == 90 * S else ((7 * GIB, 3 * GIB) if t < 90 * S else (GIB, 3 * GIB))
        v = self.verdict(synth(memory=at_60(5 * GIB + 4096)))
        self.assertEqual(set(self.failed(v)), {"working set at 60 s"})
        self.assertIn("5.000 GiB (sample at 90.000 s)", self.failed(v)["working set at 60 s"])
        self.assertTrue(self.verdict(synth(memory=at_60(5 * GIB))).passed)

    def test_a_private_commit_over_6_gib_at_60_s_fails(self) -> None:
        v = self.verdict(synth(memory=lambda t: (2 * GIB, 6 * GIB + 1 if t == 90 * S else 3 * GIB)))
        self.assertEqual(set(self.failed(v)), {"private commit at 60 s"})
        self.assertTrue(self.verdict(synth(memory=lambda t: (2 * GIB, 6 * GIB))).passed)

    def test_a_truncated_capture_fails(self) -> None:
        text = synth()
        lines = text.split("\n")
        self.assertTrue(lines[-2].startswith("end\t"))
        cases = {
            "no closing record": "\n".join(lines[:-2]) + "\n",
            "cut inside the last present's frame line": "\n".join(lines[:-2])[:-7],
            "cut inside the closing record": "\n".join(lines[:-1])[:-9],
        }
        for what, cut in cases.items():
            with self.subTest(what=what):
                v = self.verdict(cut)
                self.assertIn("closing record", self.failed(v))
                self.assertFalse(v.passed)

    def test_records_after_the_closing_record_fail(self) -> None:
        v = self.verdict(synth() + "present\t99999\t1\t1\t0\t1\t0\t0\t0\n")
        self.assertEqual(self.failed(v), {"closing record": "records after it"})

    def test_the_title_never_reached_fails(self) -> None:
        v = self.verdict(synth(segments=((20, 60.0),), title_segment=None))
        self.assertIn("title reached", self.failed(v))
        self.assertFalse(v.passed)

    def test_a_window_shorter_than_60_s_fails(self) -> None:
        v = self.verdict(synth(segments=((30, 60.0), (30, 60.0)), memory=None))
        self.assertIn("window complete", self.failed(v))
        self.assertIn("frame rate", self.failed(v))

    def test_a_header_without_the_driver_fails(self) -> None:
        v = self.verdict(synth(header_drop=("driver", "implicit_vulkan_layers")))
        self.assertEqual(self.failed(v), {"header": "lacks driver, implicit_vulkan_layers"})

    def test_header_keys_and_record_kinds_it_does_not_know_are_kept_or_skipped(self) -> None:
        extra = ["header\tsetting gfx\tvulkan (command line)", "event\t200\tgfx\tnull\tvulkan\tcommand-line"]
        cap = cc.parse(self.write(synth(extra=extra)))
        self.assertEqual(cap.header["setting gfx"], "vulkan (command line)")
        self.assertTrue(cc.check(cap).passed)

    def test_the_render_thread_busy_time_is_the_submitters_core_time(self) -> None:
        v = self.verdict(synth())
        self.assertEqual(v.figures["render thread busy p50/p99/max ms"], [8.0, 8.0, 8.0])
        self.assertEqual(v.figures["thread CPU ms per frame, top 5"], {hex(WORKER): 20.0, hex(GAME): 7.0})

    def test_a_file_that_is_not_a_capture_is_unreadable(self) -> None:
        for text in ("hello\n", "", "oot2-capture\t1\npresent\t1\t2\n", "oot2-capture\t1\npresent\t1\tx\t0\t0\t1\t0\t0\t0\n"):
            with self.subTest(text=text[:20]):
                with self.assertRaises(cc.CaptureError):
                    cc.parse(self.write(text))

    def test_the_command_line_exit_codes(self) -> None:
        def run(path: Path, *args: str) -> tuple[int, str]:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                code = cc.main([str(path), *args])
            return code, out.getvalue()
        code, out = run(self.write(synth()))
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("PASS "), out)
        code, out = run(self.write(synth(end_code=71, end_reason="unimplemented")))
        self.assertEqual(code, 1)
        self.assertIn("FAIL named fatals", out)
        code, out = run(self.write(synth()), "--json")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)["passed"])
        self.assertEqual(run(self.write("not a capture\n"))[0], 2)


if __name__ == "__main__":
    unittest.main()
