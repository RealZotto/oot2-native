# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the audio renderer and its services (runtime/audio, runtime/hos/srv_audio.c).

Builds tests/runtime/audio_test.c with runtime/core, runtime/nvn, runtime/hos and
runtime/audio against a synthetic layout (one module of 0x8000 bytes, main), at -O1
and -O2 with -march=x86-64-v3, then runs each case in its own process with a fresh
game folder. A guest thread written as a host function opens a renderer through the
services, builds update blocks in the layouts of tests/runtime/audren_blocks.h
(numbers written out, not the reader's structures) and reads the answers; the frame
clock is driven by the test, so a release or a frame signal is pinned to the instant.
Nothing is read from a dump.

A passing case must print "ok <case>"; a failing case must end in exactly one FATAL
line with the expected exit code and text, and leave census.tsv with its stop row.

Needs clang for x86-64 (OOT2_CC, else clang on PATH). Without one the tests skip,
unless OOT2_REQUIRE_CLANG is set, as continuous integration sets it.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import layout  # noqa: E402

RUNTIME = ROOT / "runtime"
TEST_SOURCE = ROOT / "tests" / "runtime" / "audio_test.c"
END_SOURCE = ROOT / "tests" / "runtime" / "endmark.c"   # linked after the test: the end of its lifted code
LEVELS = ("-O1", "-O2")
MODULE_SIZE = 0x8000
TIMEOUT = 120

EXIT_UNIMPLEMENTED = 71

PASSING = ("release_clock", "release_clock_32k", "cadence_running_stopped", "cadence_period", "cadence_no_burst",
           "parse_ok", "flush_stop", "device_basic", "auddmg_basic", "mix_voice", "mix_voice_slots", "mix_voice_slots_old",
           "mix_formats", "mix_resample", "mix_adpcm", "mix_pool_refused", "mix_route", "mix_device_volume", "mix_biquad",
           "mix_loop", "mix_stereo", "mix_sink6", "starve_empty_loop", "stall_lock", "sink_null", "sink_capture",
           "stall_report", "stall_healthy", "stall_names", "stall_no_update", "sysmix_stream", "opus_fake")

SERVICE = "service\taud:u\timplemented\t1\t"
UPDATE = "FATAL: unimplemented audio: RequestUpdate: "

# case: (text the FATAL line must contain, a row census.tsv must hold)
FAILING = {
    "open_manual": ("FATAL: unimplemented audio: OpenAudioRenderer: rendering device 0 and execution mode 1; the "
                    "renderer runs on the DSP in automatic mode only", SERVICE),
    "open_device": ("FATAL: unimplemented audio: OpenAudioRenderer: rendering device 1 and execution mode 0", SERVICE),
    "open_rate": ("FATAL: unimplemented audio: GetAudioRendererWorkBufferSize: the output rate is neither 48000 nor "
                  "32000 Hz", SERVICE),
    "open_tag": ("FATAL: unimplemented audio: GetAudioRendererWorkBufferSize: the revision tag names no revision from "
                 "1 to 16", SERVICE),
    "open_notmem": ("FATAL: unimplemented audio: OpenAudioRenderer: no transfer memory handle came with the request, "
                    "and the work buffer states 65536 bytes", SERVICE),
    "event_twice": ("FATAL: unimplemented audio: IAudioRenderer QuerySystemEvent twice: one readable end per event",
                    SERVICE),
    "manual_execution": ("FATAL: unimplemented service command: IAudioRenderer cmif command 11 (0 bytes of arguments",
                         SERVICE),
    "auddmg_open": ("FATAL: unimplemented service command: auddmg:u cmif command 1 (0 bytes of arguments",
                    "command\tauddmg:u cmif 1\tnamed-fatal\t1\t"),
    "auddmg_stream": ("FATAL: unimplemented service command: IAudioVirtualDM cmif command 900 (16 bytes of "
                      "arguments", "command\tIAudioVirtualDM cmif 900\tnamed-fatal\t1\t"),
    "aud_unknown": ("FATAL: unimplemented service command: aud:u cmif command 0 (0 bytes of arguments",
                    "command\taud:u cmif 0\tnamed-fatal\t1\t"),
    "parse_voice_stride": (UPDATE + "voice section is 832 bytes: 2 voices of 392 (REV16) take 784; the section holds "
                                    "416 per voice, remainder 0", SERVICE),
    "parse_voice_extra": (UPDATE + "voice section is 792 bytes: 2 voices of 392 (REV16) take 784; the section holds "
                                   "396 per voice, remainder 0", SERVICE),
    "parse_voice_old": (UPDATE + "voice section is 736 bytes: 2 voices of 392 (REV16) take 784; the section holds 368 "
                                 "per voice, remainder 0", SERVICE),
    "parse_pools": (UPDATE + "memory pool section is 320 bytes, 9 pools of 32 take 288", SERVICE),
    "parse_effects": (UPDATE + "effect section is 384 bytes: 1 effects of 192 take 192", SERVICE),
    "parse_sinks": (UPDATE + "sink section is 352 bytes: 1 sinks of 320 take 320", SERVICE),
    "parse_behavior": (UPDATE + "behavior section is 32 bytes, REV16 takes 16", SERVICE),
    "parse_total": (UPDATE + "the update's total ", SERVICE),
    "parse_leftover": (UPDATE + "16 bytes of the update are not accounted for by its sections", SERVICE),
    "parse_out_small": (UPDATE + "the output block needs 624 bytes (REV16, effect status 144 bytes each), the client "
                                 "gave 256", SERVICE),
    "parse_voice_id": (UPDATE + "voice record 1 carries the id 9, the renderer has 2 voices", SERVICE),
    "parse_wave_index": (UPDATE + "voice 0 has the wave buffer index 4 (of 4)", SERVICE),
    "parse_revision": (UPDATE + "the update's revision tag 0x30564552 names no revision from 1 to 16", SERVICE),
    "parse_mixes": (UPDATE + "mix section is 2360 bytes for 2 mixes (2352 each in full, or 0x20 and a multiple of it)",
                    SERVICE),
    "parse_splitter_magic": (UPDATE + "splitter header magic 0x58 is not SNDH", SERVICE),
}


def find_compiler() -> str | None:
    return shutil.which(os.environ.get("OOT2_CC", "clang"))


class AudioTest(unittest.TestCase):
    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = find_compiler()
        if not cc:
            if os.environ.get("OOT2_REQUIRE_CLANG"):
                raise RuntimeError("OOT2_REQUIRE_CLANG is set but no clang was found (OOT2_CC or PATH)")
            raise unittest.SkipTest("no clang on PATH (set OOT2_CC)")
        cls.cc = cc
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-audio-", ignore_cleanup_errors=True)
        out = Path(cls.tmp.name)
        cls.out = out
        layout.write(layout.place({"main": MODULE_SIZE}), out)
        sources = [str(TEST_SOURCE), str(END_SOURCE)]
        for part in ("core", "nvn", "hos", "audio"):
            sources += sorted(str(p) for p in (RUNTIME / part).glob("*.c"))
        cls.sources = sources
        # Windows: without high-entropy ASLR the host's own stacks stay below 4 GiB, out of the fixed module region
        # (runtime/core/memory.c, rt_mem_init); the program is linked the same way.
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        libs = [] if os.name == "nt" else ["-lm"]   # lrintf in the sink and the tests is not always inlined
        for level in LEVELS:
            exe = out / f"audio_test{level}{'.exe' if os.name == 'nt' else ''}"
            cmd = [cc, "-std=gnu11", level, "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", *extra,
                   "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "audio"),
                   "-I", str(out), "-I", str(ROOT / "tests" / "runtime"), *sources, "-o", str(exe), *libs]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"build at {level} failed:\n{done.stdout}{done.stderr}")
            cls.exes[level] = exe

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str) -> tuple[subprocess.CompletedProcess, Path]:
        folder = Path(tempfile.mkdtemp(prefix=f"{case}{level}-", dir=self.tmp.name))
        env = {k: v for k, v in os.environ.items() if not k.startswith("OOT2_CENSUS")}   # a default run
        try:
            done = subprocess.run([str(self.exes[level]), case, str(folder)], capture_output=True, text=True,
                                  timeout=TIMEOUT, env=env)
        except subprocess.TimeoutExpired as e:
            self.fail(f"{case} at {level} ran past {TIMEOUT} s:\n{e.stdout or ''}{e.stderr or ''}")
        return done, folder

    def test_release_clock(self) -> None:
        for level in LEVELS:
            for case in ("release_clock", "release_clock_32k"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)

    def test_cadence(self) -> None:
        for level in LEVELS:
            for case in ("cadence_running_stopped", "cadence_period", "cadence_no_burst"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_parse_and_voice_lifetime(self) -> None:
        for level in LEVELS:
            for case in ("parse_ok", "flush_stop", "device_basic", "auddmg_basic"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_mixing(self) -> None:
        for level in LEVELS:
            for case in ("mix_voice", "mix_formats", "mix_resample", "mix_adpcm", "mix_route", "mix_device_volume", "mix_biquad",
                         "mix_loop", "mix_stereo", "mix_sink6"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)
                    self.assertNotIn("is not read", done.stderr)

    def test_voice_slots_do_not_silence_other_channels(self) -> None:
        for level in LEVELS:
            for case in ("mix_voice_slots", "mix_voice_slots_old"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_a_wave_buffer_outside_the_attached_pools_is_not_read(self) -> None:
        done, _ = self.run_case("-O2", "mix_pool_refused")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("ok mix_pool_refused", done.stdout)
        lines = [line for line in done.stderr.splitlines() if "is not read" in line]
        self.assertEqual(len(lines), 1, done.stderr)
        self.assertRegex(lines[0], r"audio: wave buffer of voice 0 \(slot 0\) at 0x[0-9a-f]+, 480 bytes, is not read: "
                                   r"not inside an attached memory pool")

    def test_stall_cases(self) -> None:
        for level in LEVELS:
            for case in ("stall_report", "stall_healthy", "stall_names", "stall_no_update", "stall_lock"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertNotIn("FATAL", done.stderr)

    def test_a_buffer_with_no_samples_that_loops_for_ever_does_not_hold_up_the_frame(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "starve_empty_loop")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok starve_empty_loop", done.stdout)
                self.assertNotIn("FATAL", done.stderr)

    def test_a_held_renderer_lock_is_named_not_waited_for(self) -> None:
        done, _ = self.run_case("-O2", "stall_lock")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = [line for line in done.stderr.splitlines() if "audio: stall" in line]
        self.assertEqual(len(lines), 1, done.stderr)
        self.assertRegex(lines[0], r"^\[rt\] audio: stall: the renderer's thread has not released the renderer's lock for 1000 ms "
                                   r"\(a frame, an update or a sink write does not return\); renderer stopped, 0 frames rendered, "
                                   r"no frame yet$")

    def test_a_build_with_the_sdl3_kit_but_no_audio_driver_file_still_links(self) -> None:
        """tools/build_game.py compiles every runtime file with OOT2_WITH_SDL3=1 whenever a kit is present and links the
        runtime whole: the sink must not call a driver that no file defines. The same objects link with that flag alone; the
        sink built as if the driver file existed (OOT2_AUDIO_SDL3) does not link, and the linker names the driver."""
        flags = [self.cc, "-std=gnu11", "-O0", "-march=x86-64-v3", "-Wall", "-Wextra", "-Werror", "-DOOT2_WITH_SDL3=1",
                 "-I", str(RUNTIME / "include"), "-I", str(RUNTIME / "core"), "-I", str(RUNTIME / "audio"),
                 "-I", str(self.out), "-I", str(ROOT / "tests" / "runtime")]
        folder = Path(tempfile.mkdtemp(prefix="sdl3-", dir=self.out))
        objects, sink = [], None
        for at, source in enumerate(self.sources):
            obj = folder / f"{at}.o"
            if Path(source) == RUNTIME / "audio" / "sink.c":
                sink = source
                continue
            done = subprocess.run([*flags, "-c", source, "-o", str(obj)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            objects.append(str(obj))
        self.assertIsNotNone(sink)
        extra = ["-Wl,--disable-high-entropy-va"] if os.name == "nt" else ["-pthread"]
        libs = [] if os.name == "nt" else ["-lm"]
        for define, links in (([], True), (["-DOOT2_AUDIO_SDL3=1"], False)):
            sink_obj = folder / ("sink_%d.o" % len(define))
            done = subprocess.run([*flags, *define, "-c", sink, "-o", str(sink_obj)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            exe = folder / ("linked_%d%s" % (len(define), ".exe" if os.name == "nt" else ""))
            done = subprocess.run([self.cc, *extra, *objects, str(sink_obj), "-o", str(exe), *libs], capture_output=True, text=True)
            if links:
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            else:
                self.assertNotEqual(done.returncode, 0)
                self.assertIn("audio_sdl3_driver", done.stdout + done.stderr)

    def test_a_waiter_on_a_healthy_renderer_gets_no_report(self) -> None:
        done, _ = self.run_case("-O2", "stall_healthy")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertNotIn("audio: stall", done.stderr)

    def test_the_event_report_names_the_thread_and_repeats_once_in_ten_seconds(self) -> None:
        done, _ = self.run_case("-O2", "stall_report")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = [line for line in done.stderr.splitlines() if "audio: stall" in line]
        self.assertEqual(len(lines), 2, done.stderr)
        for line in lines:
            self.assertRegex(line, r"^\[rt\] audio: stall: guest thread 0x[0-9a-f]+ \(entry 0x[0-9a-f]+ \(main\+0x200\)\) has waited "
                                   r"(1000|11000) ms on the renderer event; renderer running, 0 frames rendered, no frame yet$")

    def test_the_buffer_report_names_the_voice_the_slot_and_the_submitting_thread(self) -> None:
        done, _ = self.run_case("-O2", "stall_names")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = [line for line in done.stderr.splitlines() if "audio: stall" in line]
        self.assertEqual(len(lines), 2, done.stderr)
        self.assertRegex(lines[0], r"^\[rt\] audio: stall: voice 0 wave buffer 0 \(2400 samples\) submitted by guest thread "
                                   r"0x[0-9a-f]+ \(entry 0x[0-9a-f]+ \(main\+0x100\)\) 1050 ms ago \(0 frames rendered since\), not released; "
                                   r"voice started, renderer stopped$")
        self.assertRegex(lines[1], r"^\[rt\] audio: stall: voice 0 wave buffer 1 \(4800 samples\) submitted by guest thread "
                                   r"0x[0-9a-f]+ \(entry 0x[0-9a-f]+ \(main\+0x100\)\) 1150 ms ago \(0 frames rendered since\), not released; "
                                   r"voice started, renderer stopped$")

    def test_a_missing_update_is_named(self) -> None:
        done, _ = self.run_case("-O2", "stall_no_update")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        lines = [line for line in done.stderr.splitlines() if "audio: stall" in line]
        self.assertEqual(len(lines), 1, done.stderr)
        self.assertRegex(lines[0], r"^\[rt\] audio: stall: no update for 1000 ms; renderer running, 0 frames rendered; the last "
                                   r"update came from guest thread 0x[0-9a-f]+ \(entry 0x[0-9a-f]+ \(main\+0x100\)\)$")

    def test_the_sound_mixer_takes_a_streams_frames(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "sysmix_stream")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok sysmix_stream", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: sysmix stream 1 open: int16 stereo 48000, 1024 frames, sink "
                                              r"capture \(")
                self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: sysmix stream 1 fault: the ring's indexes are out of range or "
                                              r"the read index moved$")

    def test_the_opus_overrides_follow_the_frame_contract(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, _ = self.run_case(level, "opus_fake")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                self.assertIn("ok opus_fake", done.stdout)
                self.assertNotIn("FATAL", done.stderr)
                lines = [line for line in done.stderr.splitlines() if "the final range" in line]
                self.assertEqual(lines, ["[rt] audio: hardware Opus: the final range 0x3 is not the stream's 0x63 (logged once)"])

    def test_sinks(self) -> None:
        for level in LEVELS:
            for case in ("sink_null", "sink_capture"):
                with self.subTest(level=level, case=case):
                    done, _ = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)

    def test_the_open_and_the_first_update_are_logged(self) -> None:
        done, _ = self.run_case("-O2", "parse_ok")
        self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: work buffer size asked for 48000 Hz x240, 4 voices, 1 effects, "
                                      r"1 submixes, 1 sinks, 0 splitters, REV16: answered 65536 bytes$")
        self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: renderer open: 48000 Hz x240, 8 mix buffers, 4 voices, 1 effects, "
                                      r"1 submixes, 1 sinks, 0/0 splitters, tag REV@, REV16, device 0, mode 0, work buffer "
                                      r"65536 \(asked\) 65536 \(stated\)$")
        self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: sink: null 48000 Hz stereo float, no device: this build has no "
                                      r"SDL3 audio \(tools/fetch_sdl3.py, SDL3_DIR\)$")
        self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: update 1: REV16 \(tag REV@\), behavior 16 pools 416 voices 1176 "
                                      r"voice_resources 336 effects 192 mixes 4704 sinks 320 performance 0 splitters 0 "
                                      r"render 16 total \d+, in \d+ out 8192; per-section size / count, remainder: pools 32 "
                                      r"r0, resources 3 r0 \(of 112\), voices 392 r0, effects 192 r0, sinks 320 r0$")
        self.assertRegex(done.stderr, r"(?m)^\[rt\] audio: renderer closed after 0 frames \(0 skipped\), 0 wave buffers "
                                      r"released, \d+ updates;")

    def test_fatal_cases(self) -> None:
        for level in LEVELS:
            for case, (text, row) in FAILING.items():
                with self.subTest(level=level, case=case):
                    done, folder = self.run_case(level, case)
                    self.assertEqual(done.returncode, EXIT_UNIMPLEMENTED, done.stdout + done.stderr)
                    self.assertIn(text, done.stderr)
                    self.assertNotIn(f"ok {case}", done.stdout)
                    fatal_lines = [line for line in done.stderr.splitlines() if line.startswith("FATAL")]
                    self.assertEqual(len(fatal_lines), 1, done.stderr)
                    census = (folder / "census.tsv").read_text(encoding="utf-8")
                    self.assertIn("\n" + row, census)
                    stop = [line for line in census.splitlines() if line.startswith("stop\t")]
                    self.assertEqual(len(stop), 1, census)
                    self.assertTrue(stop[0].startswith(f"stop\texit {EXIT_UNIMPLEMENTED}\tnamed-fatal\t1\t\t{fatal_lines[0]}"),
                                    stop[0])


if __name__ == "__main__":
    unittest.main()
