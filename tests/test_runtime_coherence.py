# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Contract tests of the coherence of VRAM mirrors with the CPU's view (runtime/gpu/pagegen.c, the mirrors of
runtime/gpu/backing.c; runtime/include/oot2/pagegen.h, runtime/include/oot2/backing.h) that need no GPU: one owner of
the write watch whose generations two consumers read alike, a CPU write read by the next submission, the admission rule
and its fallback to an import, a GPU write written back only once its submission was reached, a mirror replaced only
after the submission that wrote it, a bind in a raw-access submission that grows past a mirror its page table maps
(the mirror keeps the part the bind leaves, or grows in place if already bound, without creating two copies: by the
bind's pages only, whatever the imports it overlaps and the mirrors' total are), an import such a submission binds a
range larger than a uniform's of that becomes a mirror at the next raw-access begin (brought current, in the page
table, left alone when the submission bound it before its raw stage or the admission refuses, and not past the cap on
the mirrors that the raw write-back's traffic sets), pages the GPU wrote into guest memory itself (the raw path's sinks)
told from the CPU's writes by the page generations, and the binds and waits the map refuses by name, a mirror's growth
with the reason it could not.
A growth also takes pages from an import the submission used when it can do so exactly (the import is trimmed in place:
raw_grow_used_*), waits for the submissions issued before it, and hands what the submission knew of imports it replaces to
the import that replaces them (raw_grow_heir*); every import it cannot take pages of stops by the name of the rule that
refused it.
The device and the write watch stand in (tests/runtime/coherence_test.c says how); the `watch` case uses the real write
watch on Windows and prints a skip line elsewhere. The device's own tests are tests/test_runtime_vk.py.

Also, without building anything: the runtime has one caller of rt_mem_written (a query resets the marks it reports, so a
second caller would take pages from the first).

Builds tests/runtime/coherence_test.c like tests/test_runtime_gpu.py's programs (build_program), at -O1 and -O2, and runs
each case in its own process. Needs clang and a Vulkan SDK with the headers and Volk, as tests/test_runtime_gpu.py does.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_runtime_gpu import EXIT_FATAL, LEVELS, ROOT, build_program, case_env, clang_or_skip, fatal_lines, \
    vulkan_or_skip

TEST_SOURCE = ROOT / "tests" / "runtime" / "coherence_test.c"

PASSING = ("generations", "upload", "admission", "guarded_admission", "writeback", "writeback_no_reupload",
           "writeback_partial_page", "writeback_replaced", "writeback_long_run", "gpu_mark", "gpu_mark_direct", "replace",
           "raw_grow", "raw_grow_head", "raw_grow_whole", "raw_grow_beside", "raw_grow_taken", "raw_grow_neighbour",
           "raw_grow_taken_wait", "raw_grow_taken_head", "raw_grow_taken_layout", "raw_grow_trimmed",
           "raw_grow_guarded_head", "raw_grow_guarded_tail", "raw_grow_over_total", "raw_grow_head_import",
           "raw_grow_tail_import", "raw_grow_wait", "raw_sync_wait", "raw_sync_nowait",
           "raw_promote", "raw_promote_small", "raw_promote_wait", "raw_promote_cap",
           "raw_dry_grow", "raw_dry_g4", "raw_dry_cap", "raw_dry_kind", "raw_dry_none", "raw_dry_g4_on", "raw_dry_g8",
           "raw_dry_g5", "raw_dry_g1", "raw_dry_g2",
           "raw_grow_used_import", "raw_grow_used_edge", "raw_grow_used_wait", "raw_grow_heir", "raw_grow_heir_old",
           "raw_grow_used_decommit",
           "union_begin_straddle", "union_cap_demote", "union_cap_exempt", "union_exact_bytes", "union_order",
           "union_promote_chain", "union_quiet", "union_other_cell", "union_left_hot", "union_hot_cap", "union_hot_rule",
           "union_hot_admit")

# case: text the one FATAL line must hold
FAILING = {
    "unwritten": "into a mirror was not written back by the time its value was done (a CPU wait for a fence)",
    "replace_written": "which submission 1 wrote on the GPU while it is being recorded, would be replaced by a backing "
                       "over",
    "gpu_mark_direct_partial": "a direct write-back mark of [",
    # A mirror that cannot grow stops with the reason, after the account of the bind (block offsets are the block's own).
    "raw_grow_bound_import": "which this submission bound, lies on the pages it would add",
    "raw_grow_fragmented": "the mirror cannot grow over them: its block has no free space after its part for "
                           "[0x60000, 0x150000)",
    "raw_grow_guarded_blocked": "the mirror cannot grow over them: its block has no free space after its part for "
                                "[0x30000, 0x40000)",
    "raw_grow_untracked": "the mirror cannot grow over them: the write watch does not cover [",
    "raw_grow_cap": "KiB, more than the 1024 KiB one growth may add",
    # The same stop with its dump before it (test_the_stop_of_a_straddling_bind_prints_what_it_is_made_of).
    "raw_diag_hull": "lies on the pages it would add: it lies above the part of it this submission reached",
    "raw_diag_dry": "lies on the pages it would add: it lies above the part of it this submission reached",
    "raw_diag_birth": "lies on the pages it would add: it lies above the part of it this submission reached",
    # The same stop with the event ring in the dump (test_the_event_ring_*).
    "raw_ev_miss": "which this submission bound, lies on the pages it would add",
    "raw_ev_survives_hits": "which this submission bound, lies on the pages it would add",
    "raw_ev_lost": "which this submission bound, lies on the pages it would add",
    "raw_ev_origin": "which this submission bound, lies on the pages it would add",
    # An import the submission used, which the growth cannot take pages of: each rule that refuses it, named after the head
    # (test_a_used_import_is_refused_by_the_rule_the_dump_names).
    "raw_grow_used_written": "lies on the pages it would add: it was written on the GPU by this submission",
    "raw_grow_used_hull": "which this submission bound, lies on the pages it would add",
    "raw_grow_used_hull_edge": "which this submission bound, lies on the pages it would add",
    "raw_grow_used_loose": "lies on the pages it would add: it was reached by a resolve whose extent is not stated",
    "raw_grow_used_up": "lies on the pages it would add: it lies above the part of it this submission reached, and how "
                        "far a resolve reaches upward is not bounded",
    "raw_grow_heir_hull": "which this submission bound, lies on the pages it would add",
    "raw_grow_heir_written": "lies on the pages it would add: it was written on the GPU by this submission",
    "raw_grow_heir_loose": "lies on the pages it would add: it was reached by a resolve whose extent is not stated",
    # The dry verdict names the rule that refused (test_the_verdict_names_the_rule_that_refused).
    "raw_ev_g4": "which this submission bound, lies on the pages it would add",
    "raw_ev_g4_hull": "which this submission bound, lies on the pages it would add",
    # The union at the begin leaves a chain a member of which the submission wrote on the GPU: the stop as it was.
    "union_written_skip": "lies on the pages it would add: it lies above the part of it this submission reached",
    # The union made an import, and the event of the pass in the dump says for which reason (test_the_union_event_*).
    "union_cap_stop": "lies on the pages it would add: it lies above the part of it this submission reached",
    "union_admit_stop": "lies on the pages it would add: it lies above the part of it this submission reached",
    "union_rule_stop": "lies on the pages it would add: it lies above the part of it this submission reached",
}
UNION_WHY = {
    "union_cap_stop": "made one import over them (the mirrors would pass the cap)",
    "union_admit_stop": "made one import over them (the admission refused the mirror)",
    "union_rule_stop": "made one import over them (the rule says import)",
}

# The rule of import_gives that refuses the import in each of those stops, and the reason the text adds after the head (none
# for the hull rule: its text is the head alone, as before the rule had names).
HEAD_ENDS = "lies on the pages it would add"
WHY = {
    "written": "was written on the GPU by this submission",
    "loose": "was reached by a resolve whose extent is not stated",
    "up": "lies above the part of it this submission reached, and how far a resolve reaches upward is not bounded",
}
RULE_OF = {
    "raw_grow_bound_import": "hull", "raw_grow_used_hull": "hull", "raw_grow_used_hull_edge": "hull",
    "raw_grow_heir_hull": "hull", "raw_grow_heir_loose": "loose", "raw_grow_used_written": "written", "raw_grow_heir_written": "written",
    "raw_grow_used_loose": "loose", "raw_grow_used_up": "up", "raw_diag_hull": "up", "raw_diag_dry": "up",
    "raw_diag_birth": "up", "raw_ev_miss": "hull", "raw_ev_survives_hits": "hull", "raw_ev_lost": "hull",
    "raw_ev_origin": "hull", "raw_ev_g4": "hull", "raw_ev_g4_hull": "hull", "union_written_skip": "up",
}
VERDICT_RULES = {"raw_ev_g4": ("G4[up],G4_hull_off", "up"), "raw_ev_g4_hull": ("G4[hull]", "hull")}
RULE_LINE = re.compile(r"guard G4 rule: (\w+)(?:, loose)?(?:: it (.*))?$")

# The dry growth's refusals, in the order of BackingStats raw_dry_g, as the exit line names them; the case that has each.
DRY_REFUSALS = ["rule_kind", "G1", "G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G8", "G9", "cap", "G4_hull_off"]
# The refusals each case ends with. A bind can count in several: G4 comes with G4_hull_off when the import it refuses on has
# its hull off the pages the growth adds; G1 comes with G5 (a head over 1 MiB is also past the 64 KiB the mirror reserves).
DRY_CASES = {"raw_dry_grow": (), "raw_dry_g4": ("G4", "G4_hull_off"), "raw_dry_g4_on": ("G4",), "raw_dry_cap": ("cap",),
             "raw_dry_kind": ("rule_kind",), "raw_dry_g8": ("G8",), "raw_dry_g5": ("G5",), "raw_dry_g1": ("G1", "G5"),
             "raw_dry_g2": ("G2", "G7a")}
EXIT_LINE = "[rt] gfx: vulkan: backing map: dry growth at exit: "
TOTALS_LINE = "[rt] gfx: vulkan: backing map: mirror totals at exit: "

# The dump's lines, as the log writes them: one prefix, the text after it.
DUMP = "[rt] gfx: vulkan: backing map: straddle dump: "

# Where rt_mem_written may appear outside comments: its declaration, its definition, and its one caller.
WRITE_WATCH_FILES = {"runtime/include/oot2/memory.h", "runtime/core/memory.c", "runtime/gpu/pagegen.c"}


def code_mentions(path: Path, name: str) -> list[int]:
    """Lines of path that name `name` outside // comments."""
    found = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = line.split("//", 1)[0]
        if re.search(rf"\b{name}\b", code):
            found.append(n)
    return found


def dump_lines(stderr: str) -> list[str]:
    return [line[len(DUMP):] for line in stderr.splitlines() if line.startswith(DUMP)]


def printed(stdout: str, case: str) -> dict[str, int]:
    """The key=value line the case printed before its stop, numbers read as the case wrote them (0x... or decimal)."""
    line = next(line for line in stdout.splitlines() if line.startswith(case + ":"))
    return {key: int(value, 0) for key, value in re.findall(r"(\w+)=(0x[0-9a-f]+|\d+)", line)}


def hx(n: int) -> str:
    return f"{n:#x}"


EVENT = re.compile(r"event (\d+): submission (\d+) (\w+) (\w+) \[bind (\d+)\]: (.*)$")


def events(lines: list[str]) -> list[dict]:
    """The dump's events in the order printed: seq, value, phase, what, bind (the number of the bind it was made for), the
    text, and the lines that follow it for the same seq (the backings it recorded)."""
    found: list[dict] = []
    for line in lines:
        m = EVENT.match(line)
        if m:
            found.append({"seq": int(m[1]), "value": int(m[2]), "phase": m[3], "what": m[4], "bind": int(m[5]),
                          "text": m[6], "sub": []})
            continue
        m = re.match(r"event (\d+): (.*)$", line)
        if m and found and found[-1]["seq"] == int(m[1]):
            found[-1]["sub"].append(m[2])
    return found


class DryDigestTest(unittest.TestCase):
    """The digest the dry evaluation takes of the map before and after must not read what another thread changes under
    another lock: the completion thread rewrites the list of GPU writes to write back under the coherence lock alone, while
    the evaluation holds the map's lock, so a digest that read the list could differ with nothing mutated."""

    def test_the_digest_leaves_out_the_pending_writes(self) -> None:
        text = (ROOT / "runtime" / "gpu" / "backing.c").read_text(encoding="utf-8")
        start = text.index("static uint64_t diag_digest(void) {")
        body = text[start:text.index("\n}", start)]
        code = "\n".join(line.split("//", 1)[0] for line in body.splitlines())
        self.assertIn("g_ndef", code)   # it still reads the queue the map's lock guards
        self.assertNotIn("g_npend", code)
        self.assertNotIn("g_pend", code)


class DryCountersTest(unittest.TestCase):
    """The dry evaluation of the growth's guards and of the verdict must write none of the counters of the caps and of the
    unions: the digest it takes of the map before and after covers BackingStats, and the evaluation is not allowed to move it."""

    COUNTERS = ("raw_union_", "raw_grow_over_cap", "mirror_bytes_peak", "mirror_overshoot_peak", "mirror_note(", "union_make(",
                "union_pass(")

    def body(self, name: str) -> str:
        text = (ROOT / "runtime" / "gpu" / "backing.c").read_text(encoding="utf-8")
        start = text.index(f"static void {name}(")
        body = text[start:text.index("\n}\n", start)]
        return "\n".join(line.split("//", 1)[0] for line in body.splitlines())

    def test_the_dry_evaluation_does_not_write_the_cap_counters(self) -> None:
        for name in ("diag_guards", "dry_growth", "diag_stop"):
            code = self.body(name)
            for counter in self.COUNTERS:
                with self.subTest(function=name, counter=counter):
                    self.assertNotIn(counter, code)


class WriteWatchOwnerTest(unittest.TestCase):
    """Nothing in the runtime but the page generations calls rt_mem_written."""

    def test_only_the_page_generations_query_the_write_watch(self) -> None:
        users = {}
        for path in sorted((ROOT / "runtime").rglob("*")):
            if path.suffix in (".c", ".h", ".cpp") and path.is_file():
                lines = code_mentions(path, "rt_mem_written")
                if lines:
                    users[path.relative_to(ROOT).as_posix()] = lines
        self.assertEqual(set(users), WRITE_WATCH_FILES, users)
        self.assertEqual(len(users["runtime/gpu/pagegen.c"]), 1, users)


class CoherenceTest(unittest.TestCase):
    """coherence_test.c with the runtime's core, NVN2 front end and gpu module, without a device."""

    exes: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cc = clang_or_skip()
        include, glslang = vulkan_or_skip()
        cls.tmp = tempfile.TemporaryDirectory(prefix="oot2-coherence-", ignore_cleanup_errors=True)
        cls.exes = build_program(cc, include, glslang, TEST_SOURCE, Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "tmp"):
            cls.tmp.cleanup()

    def run_case(self, level: str, case: str, timeout: int = 120) -> subprocess.CompletedProcess:
        return subprocess.run([str(self.exes[level]), case], capture_output=True, text=True, timeout=timeout,
                              env=case_env())

    def test_passing_cases(self) -> None:
        for case in PASSING:
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    self.assertIn(f"ok {case}", done.stdout)
                    self.assertEqual(fatal_lines(done.stderr), [])

    def test_the_real_write_watch_keeps_its_pages_for_the_owner(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "watch")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                if os.name != "nt":
                    self.assertIn("skip watch", done.stdout)
                    self.skipTest("no write watch on this platform")
                self.assertIn("ok watch", done.stdout)
                self.assertEqual(fatal_lines(done.stderr), [])

    def test_what_the_map_cannot_keep_coherent_stops_by_name(self) -> None:
        for case, text in FAILING.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
                    self.assertNotIn("returned", done.stdout)
                    fatal = fatal_lines(done.stderr)
                    self.assertEqual(len(fatal), 1, done.stderr)
                    self.assertIn(text, fatal[0])

    def test_a_used_import_is_refused_by_the_rule_the_dump_names(self) -> None:
        """The stop and the dry evaluation of the guard in the dump agree on which rule refused the import, and the stop
        names it: the text after the head is the rule's reason (the hull rule keeps the head alone), and the refusals the
        run made so far are counted in the dump."""
        for case, rule in RULE_OF.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done, lines = self.stop_with_dump(level, case)
                    fatal = fatal_lines(done.stderr)[0]
                    named = [r for r, why in WHY.items() if fatal.endswith(f"{HEAD_ENDS}: it {why}")]
                    if rule == "hull":
                        self.assertTrue(fatal.endswith(HEAD_ENDS), fatal)
                        self.assertEqual(named, [])
                    else:
                        self.assertEqual(named, [rule], fatal)
                    seen = [RULE_LINE.match(line) for line in lines if line.startswith("guard G4 rule: ")]
                    self.assertEqual(len(seen), 1, lines)
                    self.assertEqual(seen[0][1], rule)
                    if rule != "hull":
                        self.assertEqual(seen[0][2], WHY[rule])
                    counts = [line for line in lines if line.startswith("growths so far: ")]
                    self.assertEqual(len(counts), 1, lines)
                    self.assertIn(f" {rule} 1", counts[0].split("import refusals:")[1])

    def test_a_growth_over_a_used_import_logs_what_it_did(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "raw_grow_used_import")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                logs = [line for line in done.stderr.splitlines() if "backing map: raw growth" in line]
                self.assertEqual(len(logs), 2, done.stderr)
                self.assertRegex(logs[0], r"raw growth 1 in submission 2: the mirror \[0x[0-9a-f]+, 0x[0-9a-f]+\) grew to "
                                 r"\[0x[0-9a-f]+, 0x[0-9a-f]+\), adding 0x1000 bytes; 1 used imports trimmed, 0 unused "
                                 r"deferred; mirror_bytes \d+; trimmed \[0x[0-9a-f]+, 0x[0-9a-f]+\) used_by 2 written 0 "
                                 r"hull \[0x[0-9a-f]+, 0x[0-9a-f]+\)")
                self.assertTrue(logs[1].endswith(
                    "raw growth at exit: grows 1 (4096 bytes), 1 trimmed a used import (1 imports, 4096 bytes), 0 waited, "
                    "heirs 0 (0 with a GPU write); import refusals: newer 0 written 0 no_hull 0 loose 0 hull 0 up 0"), logs[1])

    def stop_with_dump(self, level: str, case: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        done = self.run_case(level, case)
        self.assertEqual(done.returncode, EXIT_FATAL, done.stdout + done.stderr)
        self.assertNotIn("returned", done.stdout)
        fatal = fatal_lines(done.stderr)
        self.assertEqual(len(fatal), 1, done.stderr)   # the dump is log lines: the stop is still the one FATAL
        self.assertIn(FAILING[case], fatal[0])
        lines = dump_lines(done.stderr)
        self.assertTrue(lines, done.stderr)
        last = done.stderr.splitlines().index(DUMP + lines[-1])
        self.assertTrue(done.stderr.splitlines()[last + 1].startswith("FATAL"), done.stderr)   # nothing between
        return done, lines

    def backing_line(self, lines: list[str], kind: str, lo: int, hi: int) -> str:
        found = [line for line in lines if line.startswith(f"backing {kind} [{hx(lo)}, {hx(hi)}) ")]
        self.assertEqual(len(found), 1, (kind, hx(lo), hx(hi), lines))
        return found[0]

    def test_the_stop_of_a_straddling_bind_prints_what_it_is_made_of(self) -> None:
        """The hull of what the submission resolved in the import and in the mirror, the binds in order with their phase."""
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_diag_hull")
                k = printed(done.stdout, "raw_diag_hull")
                p, v, v0 = k["p"], k["v"], k["v0"]
                imp = self.backing_line(lines, "import", p - 0x20000, p)
                self.assertIn(f"res (value {v}, [{hx(p - 0x10000)}, {hx(p - 0x10000 + 4)}), n 1) born fill_pool for {v} ", imp)
                mirror = self.backing_line(lines, "mirror", p, p + 0x40000)
                self.assertIn(f"res (value {v}, [{hx(p + 0x100)}, {hx(p + 0x3004)}), n 2) born bind for {v0} "
                              f"was [{hx(p)}, {hx(p + 0x40000)})", mirror)
                self.assertEqual(sum("<- the mirror the bind reaches" in line for line in lines), 1)
                bind = [line for line in lines if line.startswith("bind [")]
                self.assertEqual(len(bind), 1, lines)
                self.assertIn(f"bind [{hx(k['bind_addr'])}, {hx(k['bind_addr'] + k['bind_size'])}) pages "
                              f"[{hx(p - 0x2000)}, {hx(p + 0x4000)}) ", bind[0])
                self.assertTrue(lines[0].startswith(f"growth refused in submission {v} "), lines[0])
                head = next(i for i, line in enumerate(lines) if line.startswith(f"the binds of submission {v},"))
                ring = []
                for line in lines[head + 1:]:
                    if line.startswith("the last "):
                        break
                    ring.append(line)
                want = [f"submission {v} before_begin hit mirror addr {hx(p + 0x3000)} size 0x4 ",
                        f"submission {v} after_begin hit mirror addr {hx(p + 0x100)} size 0x4 ",
                        f"submission {v} after_begin hit import addr {hx(p - 0x10000)} size 0x4 ",
                        f"submission {v} after_begin miss none addr {hx(k['bind_addr'])} size {hx(k['bind_size'])} "]
                self.assertEqual(len(ring), len(want), ring)
                for line, text in zip(ring, want):
                    self.assertIn(text, line)
                self.assertIn(f"stop 1 of this run follows", lines[-1])

    def test_the_dry_evaluation_of_the_growth_leaves_the_map_as_it_was(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_diag_dry")
                k = printed(done.stdout, "raw_diag_dry")
                p, v = k["p"], k["v"]
                left = [line for line in lines if line.startswith("the dry evaluation left the map ")]
                self.assertEqual(len(left), 1, lines)
                match = re.fullmatch(r"the dry evaluation left the map unchanged \(digest (0x[0-9a-f]+) before, "
                                     r"(0x[0-9a-f]+) after\)", left[0])
                self.assertTrue(match, left[0])
                self.assertEqual(match.group(1), match.group(2))
                guards = {}
                for line in lines:
                    m = re.match(r"guard (G\d\w?): (passes|refuses)(?:: (.*))?$", line)
                    if m:
                        guards.setdefault(m.group(1), []).append((m.group(2), m.group(3) or ""))
                self.assertEqual(sorted(guards), ["G1", "G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G8", "G9"])
                for tag in guards:
                    verdict = [d[0] for d in guards[tag]]
                    self.assertEqual(verdict, ["refuses"] if tag == "G4" else ["passes"], (tag, guards[tag]))
                self.assertEqual(guards["G4"][0][1],
                                 f"the import [{hx(p - 0x20000)}, {hx(p)}) used_by {v} written 0 hull "
                                 f"[{hx(p - 0x10000)}, {hx(p - 0x10000 + 4)}) hull_on_added_pages no hull_on_bind no")

    def test_the_dump_says_where_each_backing_came_from(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_diag_birth")
                k = printed(done.stdout, "raw_diag_birth")
                p, T, P, v0, v1, v = k["p"], k["T"], k["P"], k["v0"], k["v1"], k["v"]
                self.assertIn(f"born grow for {v} was [{hx(p)}, {hx(p + 0x40000)})",
                              self.backing_line(lines, "mirror", p - 0x2000, p + 0x40000))
                self.assertIn(f"born fill_pool for {v} ", self.backing_line(lines, "import", p - 0x20000, p - 0x2000))
                self.assertIn(f"born beside_front for {v} was [{hx(T)}, {hx(T + 0x40000)})",
                              self.backing_line(lines, "mirror", T, T + 0x3F000))
                promoted = self.backing_line(lines, "mirror", P, P + 0x20000)
                self.assertIn(f"born promote for {v} was [{hx(P)}, {hx(P + 0x20000)})", promoted)
                # nothing resolved it: the hull is empty, as a new backing's is (a low end above the high end)
                self.assertIn("res (value 0, empty (lo 0xffffffffffffffff, hi 0), n 0)", promoted)
                # how each bind of the submission was served: the cut one made an import, the grown one grew the mirror
                ring = [line for line in lines if line.startswith("ring ") and f"submission {v} " in line]
                self.assertTrue(any(f"made import addr {hx(T + 0x3F000)} size 0x20000 " in line for line in ring), ring)
                self.assertTrue(any(f"grew mirror addr {hx(p - 0x1800)} size 0x5800 " in line for line in ring), ring)
                head = next(i for i, line in enumerate(lines) if line.startswith("the last "))
                self.assertEqual(lines[head], f"the last 1 binds of the raw-access submission before it ({v1})")
                self.assertIn(f"submission {v1} after_begin hit import addr {hx(P)} size 0x20000 ", lines[head + 1])

    def test_the_event_ring_holds_the_bind_that_cut_the_mirror_and_its_verdict(self) -> None:
        """B0 is a miss, not a hit: its event, the dry verdict right after it, the cut and what it made are all in the dump,
        tied by the number of the bind, with the backings the growth would have replaced as they were before the cut."""
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_ev_miss")
                k = printed(done.stdout, "raw_ev_miss")
                p, v0, v = k["p"], k["v0"], k["v"]
                evs = [e for e in events(lines) if e["value"] == v]
                self.assertEqual([e["what"] for e in evs], ["begin", "fill", "miss", "verdict", "cut", "made", "miss"])
                miss, verdict, cut, made = evs[2:6]
                self.assertEqual({miss["bind"], verdict["bind"], cut["bind"], made["bind"]}, {miss["bind"]})
                self.assertGreater(miss["bind"], 0)
                self.assertEqual(evs[6]["bind"], miss["bind"] + 1)
                self.assertEqual(miss["phase"], "after_begin")
                self.assertEqual(miss["text"], f"addr {hx(k['b0_addr'])} size 0x1000 pages [{hx(p - 0x1000)}, "
                                               f"{hx(p + 0x1000)}) committed run [{hx(p - 0x20000)}, {hx(p + 0x40000)}) "
                                               f"rule says mirror")
                self.assertIn(f"the mirror [{hx(p)}, {hx(p + 0x40000)}) would grow to [{hx(p - 0x1000)}, "
                              f"{hx(p + 0x40000)}), adding 0x1000 bytes (head 0x1000, ", verdict["text"])
                self.assertIn("rule says mirror;", verdict["text"])
                self.assertIn("the evaluation left the map unchanged (its digest leaves out the list of GPU writes); refuses: nothing "
                              "(it would grow)", verdict["text"])
                self.assertNotIn("not evaluated", verdict["text"])
                self.assertIn(" hull_on_added_pages unknown (no hull of this submission)", verdict["sub"][1])
                self.assertEqual(len(verdict["sub"]), 2, verdict["sub"])
                self.assertTrue(verdict["sub"][0].startswith(
                    f"the mirror it would grow mirror [{hx(p)}, {hx(p + 0x40000)}) used_by {v0} (before submission {v})"),
                    verdict["sub"][0])
                self.assertIn(f"born0 bind for {v0} was [{hx(p)}, {hx(p + 0x40000)})", verdict["sub"][0])
                self.assertTrue(verdict["sub"][1].startswith(
                    f"would be replaced import [{hx(p - 0x20000)}, {hx(p)}) used_by 0 (before submission {v}) hull empty "
                    f"res_n 0 "), verdict["sub"][1])
                self.assertIn(f"born0 fill_pool for {v} was [{hx(p - 0x20000)}, {hx(p)})", verdict["sub"][1])
                self.assertEqual(cut["text"], f"mirror [{hx(p)}, {hx(p + 0x40000)}) kept its end (the bind took its start): "
                                              f"now [{hx(p + 0x1000)}, {hx(p + 0x40000)})")
                self.assertIn(f"born0 bind for {v0} ", cut["sub"][0])
                self.assertEqual(made["text"], f"import [{hx(p - 0x20000)}, {hx(p + 0x1000)}), replacing 1 recorded and "
                                               f"0 more backings")
                self.assertTrue(made["sub"][0].startswith(f"replaced import [{hx(p - 0x20000)}, {hx(p)}) used_by 0 "),
                                made["sub"][0])
                # the import it made is the one the dump lists, and the event is as old as the miss, not the stop
                self.assertIn(f"born bind for {v} ", self.backing_line(lines, "import", p - 0x20000, p + 0x1000))
                self.assertIn(f"event ring overwritten: none of the events of these two submissions (submission {v}: 0 of "
                              f"7)", lines[-2])
                self.assertEqual(done.stderr.count(EXIT_LINE), 0)

    def test_the_event_ring_keeps_the_miss_when_a_thousand_hits_follow_it(self) -> None:
        """The ring of binds takes every hit and loses B0; the event ring takes none and keeps it."""
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_ev_survives_hits")
                k = printed(done.stdout, "raw_ev_survives_hits")
                p, v = k["p"], k["v"]
                b0 = f"addr {hx(k['b0_addr'])} size 0x1000 "
                ring = [line for line in lines if line.startswith("ring ")]
                self.assertEqual(len(ring), 512, len(ring))
                self.assertFalse([line for line in ring if "miss none " + b0 in line], "B0 should have left the bind ring")
                evs = [e for e in events(lines) if e["value"] == v]
                self.assertEqual([e["what"] for e in evs], ["begin", "fill", "miss", "verdict", "cut", "made", "miss"])
                self.assertIn(b0, evs[2]["text"])
                self.assertIn("refuses: nothing (it would grow)", evs[3]["text"])
                self.assertIn(f"now [{hx(p + 0x1000)}, {hx(p + 0x40000)})", evs[4]["text"])

    def test_the_event_ring_says_how_many_events_it_already_lost(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_ev_lost")
                k = printed(done.stdout, "raw_ev_lost")
                v = k["v"]
                evs = [e for e in events(lines) if e["value"] == v]
                m = re.fullmatch(rf"event ring overwritten: (\d+) events of these two submissions \(submission {v}: "
                                 rf"(\d+) of (\d+)\)", lines[-2])
                self.assertTrue(m, lines[-2])
                lost, lost_v, total = int(m[1]), int(m[2]), int(m[3])
                self.assertEqual(lost, lost_v)
                self.assertGreater(lost, 0)
                self.assertEqual(len(evs), 256)   # the ring is full of this submission's events
                self.assertEqual(lost + len(evs), total)
                self.assertNotEqual(evs[0]["what"], "begin")   # the oldest ones are the lost ones
                self.assertEqual(evs[-1]["what"], "miss")
                self.assertIn(f"of {total + 2} events", next(line for line in lines if line.startswith("events of ")))

    def test_the_union_event_names_the_reason_the_union_is_an_import(self) -> None:
        """The dump of a stop says why the pass made an import of a chain with a mirror in it: the cap, the rule or the
        admission, not one of them for all."""
        for case, why in UNION_WHY.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done, lines = self.stop_with_dump(level, case)
                    unions = [e for e in events(lines) if e["what"] == "union"]
                    self.assertEqual(len(unions), 1, lines)
                    self.assertIn(": " + why + ";", unions[0]["text"])

    def test_a_hot_chain_over_the_cap_says_so_once(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level):
                done = self.run_case(level, "union_hot_cap")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                said = [line for line in done.stderr.splitlines() if "the mirrors would pass 128 MiB, the most a promotion "
                        "makes; imported" in line]
                self.assertEqual(len(said), 1, done.stderr)

    def test_the_event_ring_keeps_the_origin_of_a_cut_mirror(self) -> None:
        """A mirror a promotion made, then cut: the dump says both. The events of the raw submission before come first."""
        for level in LEVELS:
            with self.subTest(level=level):
                done, lines = self.stop_with_dump(level, "raw_ev_origin")
                k = printed(done.stdout, "raw_ev_origin")
                P, v1, v = k["P"], k["v1"], k["v"]
                evs = events(lines)
                self.assertEqual([(e["value"], e["what"]) for e in evs],
                                 [(v1, "begin"), (v1, "fill"), (v, "begin"), (v, "fill"), (v, "promote"), (v, "miss"),
                                  (v, "verdict"), (v, "cut"), (v, "made"), (v, "miss")])
                self.assertIn(f"import [{hx(P)}, {hx(P + 0x20000)}) made a mirror", evs[4]["text"])
                cut = evs[7]
                self.assertIn(f"now [{hx(P + 0x1000)}, {hx(P + 0x20000)})", cut["text"])
                self.assertIn(f"born promote for {v} born0 promote for {v} was [{hx(P)}, {hx(P + 0x20000)})", cut["sub"][0])
                mirror = self.backing_line(lines, "mirror", P + 0x1000, P + 0x20000)
                self.assertIn(f"born beside_end for {v} was [{hx(P)}, {hx(P + 0x20000)}) born0 promote for {v} was "
                              f"[{hx(P)}, {hx(P + 0x20000)})", mirror)
                self.assertIn(f"(submission {v1}: 0 of 2; submission {v}: 0 of 8)", lines[-2])

    def test_the_verdict_names_the_rule_that_refused(self) -> None:
        """The verdict of the dry growth says which answer of import_gives refused the import (G4[up], G4[hull]), in its
        refusals and on the line of the import it would replace; the stop that follows names its own rule."""
        for case, (refuses, rule) in VERDICT_RULES.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done, lines = self.stop_with_dump(level, case)
                    verdicts = [e for e in events(dump_lines(done.stderr)) if e["what"] == "verdict"]
                    self.assertEqual(len(verdicts), 1, lines)
                    self.assertTrue(verdicts[0]["text"].endswith(f"refuses: {refuses}"), verdicts[0]["text"])
                    replaced = [sub for sub in verdicts[0]["sub"] if sub.startswith("would be replaced import ")]
                    self.assertEqual(len(replaced), 1, verdicts[0]["sub"])
                    self.assertTrue(replaced[0].endswith(f" G4 rule {rule}"), replaced[0])
                    # the dry guards of the stop name the rule of the stopping bind on a line of their own, as before
                    self.assertEqual(len([line for line in lines if line.startswith("guard G4 rule: ")]), 1)

    def test_the_caps_and_the_unions_are_counted_once_at_exit_and_when_a_growth_passes_the_cap(self) -> None:
        for level in LEVELS:
            with self.subTest(level=level, case="union_cap_exempt"):
                done = self.run_case(level, "union_cap_exempt")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                found = [line for line in done.stderr.splitlines() if line.startswith(TOTALS_LINE)]
                self.assertEqual(len(found), 1, done.stderr)
                self.assertRegex(found[0][len(TOTALS_LINE):],
                                 r"mirror_bytes \d+ \(peak \d+, a promotion leaves at most 134217728, peak overshoot \d+\), "
                                 r"growths that left it over the cap 0; unions 10 \(8 mirrors, 2 imports, 11862016 bytes, "
                                 r"mirrors moved by 8126464 bytes, 8388608 of the 8388608 bytes of the cap exemption used\), "
                                 r"chains left alone: written 0, other cell 0, past the cell 0$")
            with self.subTest(level=level, case="raw_grow_over_total"):
                done = self.run_case(level, "raw_grow_over_total")
                self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                once = [line for line in done.stderr.splitlines() if "a growth of a mirror left the mirrors at" in line]
                self.assertEqual(len(once), 1, done.stderr)
                self.assertTrue(once[0].endswith("growths over the cap so far 1, mirror_bytes peak 138420224, peak overshoot "
                                                 "4202496"), once[0])
                found = [line for line in done.stderr.splitlines() if line.startswith(TOTALS_LINE)]
                self.assertEqual(len(found), 1, done.stderr)
                self.assertIn("(peak 138420224, a promotion leaves at most 134217728, peak overshoot 4202496), growths that "
                              "left it over the cap 1; unions 0 ", found[0])

    def test_a_run_that_does_not_stop_prints_the_dry_counters_once_at_exit(self) -> None:
        for case, refusals in DRY_CASES.items():
            for level in LEVELS:
                with self.subTest(case=case, level=level):
                    done = self.run_case(level, case)
                    self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
                    found = [line for line in done.stderr.splitlines() if line.startswith(EXIT_LINE)]
                    self.assertEqual(len(found), 1, done.stderr)
                    refused = int(bool(refusals))
                    want = ", ".join(f"{name} {int(name in refusals)}" for name in DRY_REFUSALS)
                    self.assertEqual(found[0][len(EXIT_LINE):],
                                     f"1 raw-access submissions, 1 binds cut a mirror the submission had not resolved "
                                     f"there, {1 - refused} would have grown it, {refused} refused (refusals: {want}), "
                                     f"the evaluation changed the map 0 times")


if __name__ == "__main__":
    unittest.main()
