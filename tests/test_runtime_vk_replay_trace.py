# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The replay's trace (OOT2_REPLAY_TRACE, OOT2_REPLAY_TRACE_FILE; runtime/gpu/vk_replay_trace.c) and the counters of the work the
replay drops without a stop (the "silent" group of OOT2_REPLAY_COUNTERS), through tests/runtime/vk_replay_test.c: its cases are
the replay's contract tests, with the real recorder writing every packet as a line and no device.

What is proved here.
- The trace changes nothing the replay does. Every case of the replay runs twice, with the trace off and with it on (every present
  in the window, so every row is formatted); what a case shows is its standard output (the packets, in order, with the pipeline
  handles, scratch ranges and descriptor writes the replay chose, and the counts it prints: draws, dispatches, render off,
  skipped, lost), its exit code and its standard error less the trace's own lines. The runs must be equal; the draws, dispatches
  and render-off decisions are in what is compared. The same for the counters of the "silent" group.
- What the lines say. The summary and the rows of a case whose answer is known: a draw skipped for a pipeline, a draw
  turned off by RenderEnable, two by a condition (each with its cause), the S line's accounting (draw_ops against what was made,
  lost 0), the P line of each pass, the D line of each kind of draw and dispatch with its stage keys, count, instances recorded
  and replayed, a record read from the CPU's view, the texture slots, the stages that carry raw access and the stage keys that
  carried it in the S line, the bindings made as zeros, the Q line of each occlusion query with its samples, the M line of a
  raw-access miss report, and the window and every-N rules of the spec.
- A spec it does not know is a fatal at start-up, and with no file the lines go to the log.

What this cannot see. The device is not there: the texture entry, the zero regions and the occlusion counts are the stand-ins'
(tests/runtime/vk_replay_test.c); no run of the real game is read; the cost of the trace when it is off (a load and a branch) is
not measured here.

The counter-runs (run once, then reverted): a trace hook that sets the render state (the "same replay" test goes red on the first
draw that rendering off should have dropped), a trace that formats the rows of every present whatever the window (the window
test goes red).

Run: python -m unittest discover -s tests -p test_runtime_vk_replay_trace.py -v
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, case_env  # sets the path of tools/
from test_runtime_vk_replay import ReplayBase, tearDownModule  # noqa: F401  (the module's teardown frees the program)

TRACE, FILE, COUNTERS = "OOT2_REPLAY_TRACE", "OOT2_REPLAY_TRACE_FILE", "OOT2_REPLAY_COUNTERS"

PASSING = [("graphics",), ("depth",), ("compute",), ("indirect",), ("draw_indirect",), ("features",), ("mesh",),
           ("mesh_restart",), ("storage_image",), ("transfers",), ("skip",), ("skip_render_off",), ("conflict_read_first",),
           ("conflict_write_first",), ("raw_first",), ("full_rate",), ("reports",),
           ("graphics", "--mutant", "nosnapshot"), ("conflict_read_first", "--mutant", "nodrop")]
STOPPING = [("dual_two_targets",), ("broken",)]


def without_trace_lines(text: str) -> str:
    return "".join(line + "\n" for line in text.splitlines() if not line.startswith("[rt] trace "))


def rows(text: str) -> dict[str, list[list[str]]]:
    """The trace's lines by kind (the first field), each split at its tabs; the # header lines under "#"."""
    out: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        out.setdefault("#" if line.startswith("#") else fields[0], []).append(fields)
    return out


def fields(row: list[str]) -> dict[str, str]:
    """The key=value fields of a row, tex fields kept as a list under "tex"."""
    d: dict[str, str] = {}
    for f in row:
        if "=" in f:
            k, v = f.split("=", 1)
            if k == "tex":
                d.setdefault("tex", []).append(v)  # type: ignore[arg-type]
            else:
                d[k] = v
    return d


class TraceBase(ReplayBase):
    def shown(self, args: tuple[str, ...], spec: str | None, counters: bool = False, file: bool = True,
              extra: dict[str, str] | None = None) -> tuple[subprocess.CompletedProcess, str]:
        env = {k: v for k, v in case_env().items() if k not in (TRACE, FILE, COUNTERS)}
        with tempfile.TemporaryDirectory(prefix="oot2-trace-") as tmp:
            path = Path(tmp) / "trace.tsv"
            if spec is not None:
                env[TRACE] = spec
                if file:
                    env[FILE] = str(path)
            if counters:
                env[COUNTERS] = "1"
            env.update(extra or {})
            done = subprocess.run([str(self.exe), *args], capture_output=True, text=True, timeout=60, env=env)
            text = path.read_text(encoding="utf-8") if path.exists() else ""
        return done, text


class TraceChangesNothingTest(TraceBase):
    def compare(self, args: tuple[str, ...]) -> str:
        off1, _ = self.shown(args, None)
        off2, _ = self.shown(args, None)
        on, text = self.shown(args, "on,1-1000000")
        a = (off1.returncode, off1.stdout, without_trace_lines(off1.stderr))
        b = (off2.returncode, off2.stdout, without_trace_lines(off2.stderr))
        c = (on.returncode, on.stdout, without_trace_lines(on.stderr))
        if a != b and os.name != "nt":
            raise unittest.SkipTest(f"{args}: the output varies from run to run here (randomized host addresses)")
        self.assertEqual(a, b, f"{args}: two runs with the trace off differ (the output is not repeatable)")
        self.assertEqual(a, c, f"{args}: the trace changed what the replay did")
        self.assertEqual(off1.stderr.count("[rt] trace "), 0, args)
        return text

    def test_the_packets_exit_codes_counts_and_messages_are_the_same_with_the_trace_on(self) -> None:
        draw_rows = 0
        for args in PASSING + STOPPING:
            with self.subTest(args=args):
                text = self.compare(args)
                self.assertTrue(text.startswith("# oot2 replay trace v1"), text[:80])
                draw_rows += len(rows(text).get("D", []))
        self.assertGreater(draw_rows, 40)   # the comparison was over real rows

    def test_the_rows_account_for_the_draws_and_dispatches_the_case_counts(self) -> None:
        for args in (("skip_render_off",), ("skip",)):
            done, text = self.shown(args, "on,1-9")
            stats = re.search(r"stats draw_ops (\d+) draws (\d+) skipped (\d+) render_off (\d+) lost (\d+)", done.stdout)
            self.assertIsNotNone(stats, done.stdout)
            draw_ops, draws, skipped, off, lost = (int(x) for x in stats.groups())
            d = rows(text)["D"]
            self.assertEqual(len(d), draw_ops, args)
            self.assertEqual([r[5] for r in d].count("drawn"), draws)
            self.assertEqual([r[5] for r in d].count("skipped-pipeline-not-ready"), skipped)
            self.assertEqual(sum(r[5].startswith("off-") for r in d), off)
            self.assertEqual(lost, 0)

    def test_the_silent_counters_change_nothing_either(self) -> None:
        for args in (("reports",), ("skip_render_off",), ("graphics",)):
            with self.subTest(args=args):
                off, _ = self.shown(args, None)
                on, _ = self.shown(args, None, counters=True)
                lines = [x for x in on.stderr.splitlines() if x.startswith("[rt] rc ")]
                self.assertTrue(any("group=silent" in x for x in lines))
                self.assertEqual((off.returncode, off.stdout), (on.returncode, on.stdout))
                self.assertEqual([x for x in off.stderr.splitlines() if not x.startswith("[rt] rc ")],
                                 [x for x in on.stderr.splitlines() if not x.startswith("[rt] rc ")])


class SilentCountersTest(TraceBase):
    def silent(self, args: tuple[str, ...]) -> dict[str, int]:
        done, _ = self.shown(args, None, counters=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        line = [x for x in done.stderr.splitlines() if "kind=exit group=silent" in x][0]
        return {k: int(v) for k, v in re.findall(r" (\w+)=(\d+)", line)}

    def test_render_off_is_split_by_cause_and_the_conditions_are_counted(self) -> None:
        s = self.silent(("reports",))
        self.assertEqual((s["vkr_render_off_enable"], s["vkr_render_off_cond"]), (0, 2))   # two draws a condition turned off
        self.assertEqual((s["rep_conditions"], s["rep_conditions_false"]), (4, 2))
        self.assertEqual((s["rep_waits"], s["rep_waits_unsatisfied"], s["rep_waits_unsat_first_per_site_frame"]), (2, 1, 1))
        self.assertEqual(s["rep_reports_dropped"], 1)                                       # the report into GPU-only memory
        s = self.silent(("skip_render_off",))
        self.assertEqual((s["vkr_render_off_enable"], s["vkr_render_off_cond"]), (1, 0))   # the game's RenderEnable of 0
        for key in ("bind_wrap_approx", "bind_border_approx", "tex_zero_regions_uploaded", "texsync_zero_regions",
                    "img_skipped_cpu", "img_skipped_format", "vkr_clears_off"):
            self.assertIn(key, s)

    def test_the_counter_lines_name_the_old_counters_once(self) -> None:
        done, _ = self.shown(("graphics",), None, counters=True)
        text = done.stderr
        for key in ("vkr_render_off", "vkr_unregistered_textures"):
            self.assertEqual(len(re.findall(rf" {key}=", text)), 1, key)   # printed already: not printed twice


class TraceContentTest(TraceBase):
    def test_a_pipeline_skip_a_render_enable_off_and_the_accounting(self) -> None:
        _, text = self.shown(("skip_render_off",), "on,1-9")
        r = rows(text)
        d = r["D"]
        self.assertEqual([x[5] for x in d], ["skipped-pipeline-not-ready", "drawn", "off-enable"])
        self.assertEqual(d[0][4], "d")
        self.assertIn("key0=00000000000000a1.00000001", d[0])
        drawn = fields(d[1])
        self.assertEqual((drawn["count"], drawn["first"], drawn["inst"], drawn["raw"]), ("3", "3", "1>1", "0"))
        self.assertEqual(fields(d[2]).get("count"), None)       # a draw turned off has no stages and no count
        s = fields(r["S"][0])
        self.assertEqual((s["draw_ops"], s["draws"], s["skipped_pipeline"], s["off_enable"], s["off_cond"], s["lost"]),
                         ("3", "1", "1", "1", "0", "0"))
        p = r["P"][0]
        self.assertEqual((p[3], p[4]), ("1", "64x32"))
        self.assertEqual(fields(p)["draws"], "1")
        pres = fields(r["PRESENT"][0])
        self.assertEqual((pres["draws"], pres["skipped_pipeline"], pres["off_enable"], pres["off_cond"], pres["subs"],
                          pres["passes"]), ("1", "1", "1", "0", "1", "1"))
        self.assertTrue(r["PRESENT"][0][1] == "1")

    def test_a_condition_turns_draws_off_and_each_query_reports_its_samples(self) -> None:
        done, text = self.shown(("reports",), "on,1-9")
        r = rows(text)
        status = [x[5] for x in r["D"]]
        self.assertEqual(status.count("off-cond"), 2)
        self.assertEqual(status.count("off-enable"), 0)
        self.assertIn("render_off_by_cause 0 2", done.stdout)
        q = r["Q"]
        samples = [(x[3], x[4], x[5]) for x in q]                  # pass, query, samples (10 * (i + 1), the stand-in's)
        self.assertEqual([int(fields(x)["samples"]) for x in q], [10, 20, 30, 10])
        self.assertTrue(all(int(fields(x)["pass_draws"]) > 0 for x in q), samples)
        pres = fields(r["PRESENT"][0])
        self.assertEqual((pres["queries"], pres["queries_zero_samples"], pres["queries_zero_samples_with_draws"]), ("4", "0", "0"))
        self.assertEqual(pres["off_cond"], "2")
        for s in r["S"]:
            self.assertEqual(fields(s)["lost"], "0")

    def test_every_kind_of_draw_and_dispatch_has_its_row(self) -> None:
        _, text = self.shown(("graphics",), "on,1-9")
        d = rows(text)["D"]
        self.assertEqual([x[4] for x in d], ["d", "i", "I", "d"])
        indexed = fields(d[1])
        self.assertEqual((indexed["count"], indexed["inst"]), ("6", "2>2"))
        direct = fields(d[0])
        self.assertEqual(direct["inst"], "0>1")                     # recorded 0, replayed as 1
        self.assertEqual(fields(d[2])["indirect"], "gpu")           # the record is in memory the GPU wrote or the CPU cannot see
        tex = direct["tex"][0]
        self.assertTrue(tex.startswith("ps5:id=7:smp=3:entry=1:fmt=0x2c:astc=0:slot=1057:"), tex)
        _, text = self.shown(("compute",), "on,1-9")
        d = rows(text)["D"]
        self.assertEqual([x[4] for x in d], ["c", "C"])
        self.assertEqual(fields(d[0])["groups"], "4,2,1")
        _, text = self.shown(("mesh",), "on,1-9")
        d = rows(text)["D"]
        self.assertEqual([x[4] for x in d], ["m", "m", "m", "M", "M"])
        self.assertEqual([fields(x).get("count") for x in d[:3]], ["1", "2", "0"])
        self.assertIn("key2=00000000000000d1.00000001", d[1])      # the task stage
        s = fields(rows(text)["TOTAL"][0])                         # the case makes no present: the totals at the exit
        self.assertEqual((s["mesh_draws"], s["zero_work"], s["indirect"]), ("5", "1", "2"))
        _, text = self.shown(("draw_indirect",), "on,1-9")
        d = rows(text)["D"]
        read = fields(d[0])
        self.assertEqual((read["indirect"], read["count"], read["inst"]), ("read", "6", "2"))   # a record the CPU can read
        self.assertEqual(fields(d[1])["indirect"], "gpu")

    def test_raw_access_stages_and_the_bindings_made_as_zeros_are_said(self) -> None:
        _, text = self.shown(("raw_first",), "on,1-9")
        r = rows(text)
        self.assertEqual([fields(x)["raw"] for x in r["D"]], ["1", "1", "1"])
        self.assertEqual(fields(r["S"][0])["raw_stage_keys"], "00000000000000c4.00000001")
        self.assertEqual(fields(r["S"][1])["raw_stage_keys"], "00000000000000a2.00000001")
        self.assertEqual(fields(r["TOTAL"][0])["raw_access"], "3")
        _, text = self.shown(("storage_image",), "on,1-9")
        r = rows(text)
        self.assertTrue(all("unbound_as_zeros=c1:0,storage:1,uniform:0" in "\t".join(x) for x in r["D"]))
        self.assertEqual(fields(r["TOTAL"][0])["unbound_storage"], "3")

    def test_the_raw_access_miss_line_says_the_kinds_and_reasons_of_the_kept_entries(self) -> None:
        _, text = self.shown(("trace_misses",), "on")
        m = [fields(x) | {"seq": x[1]} for x in rows(text)["M"]]
        self.assertEqual(len(m), 2)                                  # the second call had nothing new
        first = m[0]
        self.assertEqual((first["seq"], first["submission"], first["misses_total"], first["earlier_miss_submissions_not_shown"]),
                         ("9", "77", "130", "2"))
        self.assertEqual([first[k] for k in ("kept", "load", "store", "atomic", "outside_span", "no_chunk", "no_page")],
                         ["5", "2", "1", "2", "1", "1", "3"])
        self.assertEqual((m[1]["seq"], m[1]["submission"], m[1]["load"], m[1]["earlier_miss_submissions_not_shown"]),
                         ("11", "90", "1", "0"))


class TraceSpecTest(TraceBase):
    def test_the_rows_are_only_in_the_window_and_the_summary_follows_every(self) -> None:
        _, text = self.shown(("skip_render_off",), "on")                    # no window: a summary and no rows
        r = rows(text)
        self.assertEqual(set(r) - {"#"}, {"PRESENT", "TOTAL"})   # rows only in a window
        _, text = self.shown(("skip_render_off",), "2-3")                   # the submission belongs to present 1
        r = rows(text)
        self.assertNotIn("D", r)
        self.assertNotIn("S", r)
        self.assertIn("PRESENT", r)
        _, text = self.shown(("skip_render_off",), "1")                     # a single present is its own window
        self.assertIn("D", rows(text))
        _, text = self.shown(("skip_render_off",), "every=2")               # present 1 is not a multiple of 2
        self.assertNotIn("PRESENT", rows(text))
        _, text = self.shown(("skip_render_off",), "every=1,1-1")
        r = rows(text)
        self.assertEqual(len(r["PRESENT"]), 1)
        self.assertIn("D", r)

    def test_the_exit_prints_the_totals(self) -> None:
        _, text = self.shown(("skip_render_off",), "on")
        total = fields(rows(text)["TOTAL"][0])
        self.assertEqual((total["draws"], total["off_enable"], total["skipped_pipeline"]), ("1", "1", "1"))

    def test_no_file_means_the_log_and_unset_means_off(self) -> None:
        done, text = self.shown(("skip_render_off",), "1-9", file=False)
        self.assertEqual(text, "")
        self.assertGreater(done.stderr.count("[rt] trace D\t"), 2)
        self.assertIn("[rt] trace # oot2 replay trace v1", done.stderr)
        done, text = self.shown(("skip_render_off",), None)
        self.assertNotIn("[rt] trace", done.stderr)
        for off in ("0", "off", ""):
            done, text = self.shown(("skip_render_off",), off)
            self.assertEqual((done.returncode, text, done.stderr.count("[rt] trace")), (0, "", 0), off)

    def test_a_spec_it_does_not_know_is_a_named_fatal_at_start_up(self) -> None:
        for spec in ("sometimes", "5-3", "every=0", "0-5", "1-x"):
            with self.subTest(spec=spec):
                done, _ = self.shown(("skip_render_off",), spec)
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertIn("OOT2_REPLAY_TRACE", done.stderr)
                self.assertNotIn("ok skip_render_off", done.stdout)


class ScanTest(TraceBase):
    """The scan of readback texels, on invented data (case trace_scan)."""

    def scans(self) -> dict[str, list[dict[str, float]]]:
        done, _ = self.shown(("trace_scan",), None)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ok trace_scan", done.stdout)
        out: dict[str, list[dict[str, float]]] = {}
        for line in done.stdout.splitlines():
            m = re.match(r"scan (\w+) nch=(\d+)(.*)", line)
            if not m or not m.group(3):
                continue
            chans = []
            for part in m.group(3).split("|")[1:]:
                chans.append({k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.e+-]+)", part)})
            out[m.group(1)] = chans
        return out

    def test_float32_counts_nan_inf_and_the_values_past_the_threshold(self) -> None:
        c = self.scans()["f32"]
        self.assertEqual(len(c), 4)
        self.assertEqual((c[0]["min"], c[0]["max"], c[0]["nonfinite"], c[0]["above"]), (-1.0, 0.75, 0, 0))
        self.assertAlmostEqual(c[0]["mean"], 0.0, places=6)
        self.assertEqual((c[1]["max"], c[1]["above"]), (70000.0, 1))
        self.assertAlmostEqual(c[1]["mean"], 70001.0 / 3.0, delta=0.1)
        self.assertEqual((c[2]["finite"], c[2]["nonfinite"], c[2]["min"], c[2]["max"], c[2]["above"]), (2, 1, 2.0, 100000.0, 1))
        self.assertAlmostEqual(c[2]["mean"], 50001.0, places=3)             # the NaN is left out of the mean
        self.assertEqual((c[3]["finite"], c[3]["nonfinite"], c[3]["max"], c[3]["above"]), (2, 1, 4.0, 0))

    def test_half_floats_and_the_packed_floats_have_their_non_finite_codes(self) -> None:
        s = self.scans()
        h = s["f16"]
        self.assertEqual((h[0]["min"], h[0]["max"], h[0]["mean"]), (-2.0, 1.0, -0.5))
        self.assertEqual((h[1]["finite"], h[1]["nonfinite"]), (1, 1))        # infinity
        self.assertEqual((h[2]["finite"], h[2]["nonfinite"], h[2]["max"]), (1, 1, 2.0))   # NaN
        self.assertEqual((h[3]["max"], h[3]["min"], h[3]["above"]), (65504.0, 0.5, 1))
        p = s["b10g11r11"]
        self.assertEqual((p[0]["min"], p[0]["max"], p[0]["nonfinite"]), (0.0, 1.0, 1))
        self.assertEqual((p[1]["min"], p[1]["max"], p[1]["nonfinite"], p[1]["above"]), (2.0, 65024.0, 1, 1))
        self.assertEqual((p[2]["min"], p[2]["max"], p[2]["nonfinite"]), (0.0, 0.5, 1))

    def test_unorm_bytes_and_a_format_with_no_decoder(self) -> None:
        u = self.scans()["unorm8"]
        self.assertEqual((u[0]["min"], u[0]["max"], u[0]["above"]), (0.0, 1.0, 1))
        self.assertAlmostEqual(u[2]["max"], 0.2, places=6)
        done, _ = self.shown(("trace_scan",), None)
        self.assertIn("scan bc1 refused", done.stdout)


def r_rows(text: str) -> list[tuple[list[str], dict[str, str]]]:
    return [(r, fields(r)) for r in rows(text).get("R", [])]


class ReadbackTest(TraceBase):
    """The readback of a pass's colour target (OOT2_REPLAY_TRACE rb=...), with the stand-in device's copy writing a pattern
    (texel i is bytes i mod 256, 255, 16 z + 8 layer, 128)."""

    def run_rb(self, case: str, spec: str, **kw) -> tuple[subprocess.CompletedProcess, str]:
        done, text = self.shown((case,), spec, **kw)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return done, text

    def test_the_statistics_of_the_target_after_its_pass(self) -> None:
        _, text = self.run_rb("trace_readback", "1-9,rb=64x32,rbthr=0.5")
        got = r_rows(text)
        self.assertEqual(len(got), 1, text)
        row, f = got[0]
        self.assertEqual((row[5], f["pass"], f["att"], f["fmt"], f["z"], f["level"], f["layer"], f["texels"]),
                         ("64x32", "1", "0", "37", "0", "0", "0", "2048"))
        self.assertEqual((float(f["c0.min"]), float(f["c0.max"])), (0.0, 1.0))
        self.assertAlmostEqual(float(f["c0.mean"]), 0.5, places=6)                 # 8 cycles of 0..255
        self.assertEqual((f["c0.above"], f["c0.nonfinite"]), ("1024", "0"))        # bytes 128 to 255 are above one half
        self.assertEqual((float(f["c1.min"]), float(f["c1.max"]), f["c1.above"]), (1.0, 1.0, "2048"))
        self.assertEqual((float(f["c2.max"]), f["c2.above"]), (0.0, "0"))
        self.assertAlmostEqual(float(f["c3.mean"]), 128 / 255, places=6)
        self.assertEqual(f["c3.above"], "2048")
        total = fields(rows(text)["TOTAL"][0])
        self.assertEqual((total["readbacks"], total["readbacks_skipped"]), ("1", "0"))

    def test_the_selection_by_extent_format_pass_attachment_and_count(self) -> None:
        cases = [("1-9,rb=64x32", ["1"]), ("1-9,rb=16x8", ["2"]), ("1-9,rb=64x32/37", ["1"]), ("1-9,rb=64x32/38", []),
                 ("1-9,rb=64x32,rb=16x8", ["1", "2"]), ("1-9,rbpass=2", ["2"]), ("1-9,rbpass=1-2", ["1", "2"]),
                 ("1-9,rbpass=3", []), ("1-9,rbn=1,rbpass=1-2", ["1"]), ("1-9,rbatt=1,rbpass=1-2", []),
                 ("1-9,rbatt=all,rbpass=1-2", ["1", "2"]), ("1-9,rb=64x32,rb=16x8,rbn=1", ["1"])]
        for spec, passes in cases:
            with self.subTest(spec=spec):
                _, text = self.run_rb("trace_readback_two", spec)
                self.assertEqual([f["pass"] for _, f in r_rows(text)], passes, text)

    def test_only_the_presents_of_the_windows_are_read(self) -> None:
        _, text = self.run_rb("trace_readback_two", "rb=64x32")                    # no window: no present is chosen
        self.assertEqual(r_rows(text), [])
        _, text = self.run_rb("trace_readback_two", "2-9,rb=64x32")                # the submission belongs to present 1
        self.assertEqual(r_rows(text), [])
        done, text = self.run_rb("trace_readback_two", "1-9,rb=64x32")
        self.assertEqual(done.stdout.count("pkt CopyImageToBuffer"), 1)
        done, text = self.run_rb("trace_readback_two", "2-9,rb=64x32")
        self.assertEqual(done.stdout.count("pkt CopyImageToBuffer"), 0)           # nothing was even recorded

    def test_a_slice_of_a_volume_and_a_slice_that_is_not_there(self) -> None:
        _, text = self.run_rb("trace_readback_volume", "1-9,rb=64x32,rbz=2")
        (_, f), = r_rows(text)
        self.assertEqual((f["z"], f["layer"]), ("2", "2"))
        self.assertAlmostEqual(float(f["c2.min"]), 32 / 255, places=6)             # the copy named slice 2: 16 z
        self.assertAlmostEqual(float(f["c2.max"]), 32 / 255, places=6)
        _, text = self.run_rb("trace_readback_volume", "1-9,rb=64x32,rbz=9")
        (_, f), = r_rows(text)
        self.assertEqual(f["skipped"], "slice_past_the_volume")
        _, text = self.run_rb("trace_readback", "1-9,rb=64x32,rbz=1")              # a plain 2D target has one slice
        (_, f), = r_rows(text)
        self.assertEqual(f["skipped"], "slice_past_the_layers")
        self.assertEqual(fields(rows(text)["TOTAL"][0])["readbacks_skipped"], "1")

    def test_the_replay_is_the_same_with_the_readback_on_but_for_its_own_copy(self) -> None:
        off, _ = self.shown(("trace_readback_two",), None)
        on, text = self.shown(("trace_readback_two",), "1-9,rb=64x32")
        self.assertEqual(on.returncode, 0, on.stdout + on.stderr)

        def stripped(out: str) -> tuple[list[str], int]:
            lines, kept, copies, i = out.splitlines(), [], 0, 0
            while i < len(lines):
                if (lines[i].startswith("pkt PipelineBarrier2") and i + 2 < len(lines)
                        and lines[i + 1].startswith("pkt CopyImageToBuffer") and lines[i + 2].startswith("pkt PipelineBarrier2")):
                    copies, i = copies + 1, i + 3
                    continue
                if not lines[i].startswith("frame "):                       # the frame's packet and byte counts include the copy
                    # the copy's destination takes scratch memory, so every later scratch offset moves by its size
                    kept.append(re.sub(r"(\+|off=)0x[0-9a-f]+", r"\1OFF", lines[i]))
                i += 1
            return kept, copies

        a, n_off = stripped(off.stdout)
        b, n_on = stripped(on.stdout)
        self.assertEqual((n_off, n_on), (0, 1))
        self.assertEqual(a, b)                                                  # every other packet, in order, is the same
        self.assertEqual(without_trace_lines(off.stderr), without_trace_lines(on.stderr))

    def test_a_malformed_readback_item_is_a_named_fatal(self) -> None:
        for spec in ("1-9,rb=64", "1-9,rb=0x4", "1-9,rb=4x4/", "1-9,rbpass=0", "1-9,rbpass=3-2", "1-9,rbatt=9", "1-9,rbn=0",
                     "1-9,rbthr=x", "1-9,rbz=-"):
            with self.subTest(spec=spec):
                done, _ = self.shown(("trace_readback",), spec)
                self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                self.assertIn("OOT2_REPLAY_TRACE", done.stderr)


class ColumnsTest(TraceBase):
    """The columns the first trace lacked: state, targets, writes, and the raw-access miss rows (cases trace_cols and
    trace_misses_keys)."""

    def test_the_draw_row_carries_blend_mask_depth_point_size_and_the_buffer_it_writes(self) -> None:
        _, text = self.shown(("trace_cols",), "1-9")
        r = rows(text)
        draws = [(x, fields(x)) for x in r["D"] if x[4] == "d"]
        self.assertEqual(len(draws), 1)
        _, f = draws[0]
        self.assertEqual((f["ncolor"], f["blend_en"], f["wmask"]), ("2", "0x1", "0x83"))     # target 1 writes only alpha
        self.assertEqual(f["blend0"], "5/6/2/1/1/1")                                      # src alpha, 1 - src alpha, 1, 0, add, add
        self.assertNotIn("blend1", f)                                                     # the game never set one for target 1
        self.assertEqual(f["ds"], "test:1,write:0,func:3,stencil:0")
        self.assertEqual((f["point_size"], f["line_width"]), ("6.5", "2"))
        self.assertRegex(f["writes"], r"^buf:0x[0-9a-f]+\+0x[0-9a-f]+(,buf:.*)?$")

    def test_a_draw_with_no_state_set_says_so_and_a_dispatch_names_the_image_it_stores_to(self) -> None:
        _, text = self.shown(("trace_cols",), "1-9")
        disp = [(x, fields(x)) for x in rows(text)["D"] if x[4] == "c"]
        self.assertEqual(len(disp), 1)
        self.assertRegex(disp[0][1]["writes"], r"^img:id9@0x[0-9a-f]+\+0x4000$")        # the stand-in's image 9, 0x4000 bytes
        self.assertNotIn("blend0", disp[0][1])
        _, text = self.shown(("graphics",), "1-9")
        for x in rows(text)["D"]:
            f = fields(x)
            self.assertEqual(f["point_size"], "unset")                                    # graphics never set a point size
            self.assertEqual(f["ds"], "unset")

    def test_the_pass_row_names_the_storage_format_extent_level_and_layer_of_each_target(self) -> None:
        _, text = self.shown(("trace_cols",), "1-9")
        passes = [fields(x) for x in rows(text)["P"]]
        self.assertTrue(passes)
        p = passes[0]
        self.assertRegex(p["t0"], r"^storage:(0x[0-9a-f]+|0):fmt:37:64x32:level:0:layer:0:layers:1$")
        self.assertRegex(p["t1"], r"^storage:(0x[0-9a-f]+|0):fmt:37:64x32:level:0:layer:0:layers:1$")
        self.assertIn("d0", p)

    def test_each_kept_miss_has_a_row_with_its_address_class_and_the_stage_keys_that_could_have_missed(self) -> None:
        done, text = self.shown(("trace_misses_keys",), "1-9")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        r = rows(text)
        keys = fields(r["S"][0])["raw_stage_keys"]
        self.assertTrue(keys)
        m = [fields(x) for x in r["m"]]
        self.assertEqual(len(m), 2)
        self.assertEqual((m[0]["addr"], m[0]["kind"], m[0]["reason"], m[0]["region"], m[0]["page"]),
                         ("0x1234567890", "load", "no_page", "0x1234", "0x1234567"))
        self.assertEqual((m[1]["addr"], m[1]["kind"], m[1]["reason"], m[1]["region"]), ("0x2000000000", "store", "outside_span", "0x2000"))
        self.assertEqual(m[0]["stage_keys"], keys)
        self.assertEqual(m[1]["stage_keys"], keys)

    def test_no_window_no_miss_rows_and_an_unknown_submission_is_a_question_mark(self) -> None:
        _, text = self.shown(("trace_misses_keys",), "on")
        self.assertNotIn("m", rows(text))
        self.assertIn("M", rows(text))
        _, text = self.shown(("trace_misses",), "1-9")
        for x in rows(text)["m"]:
            self.assertEqual(fields(x)["stage_keys"], "?")


if __name__ == "__main__":
    unittest.main()
