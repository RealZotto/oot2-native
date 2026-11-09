# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for tools/shaders/corpus.py.

The passes run against a stand-in translator written in Python that speaks the
pool's line protocol and writes a module whose bytes depend on the answers a
real translator would read: no game data, no .NET build, no Vulkan SDK.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import corpus  # noqa: E402

# Vertex programs read the attribute answers, pixel programs the output answers,
# and a program marked "tex" the texture answers; "act": "fail" fails the job.
FAKE = textwrap.dedent("""
    import json, os, sys, base64
    for line in sys.stdin:
        req = json.loads(line)
        job = req["job"]
        key, stage = job["key"], job["stage"]
        if job.get("act") == "fail":
            print(f"fail {key} 7 unsupported_opcode: Ampere op 0x1a2b at 0x340 is not implemented", flush=True)
            continue
        body = {"key": key, "stage": stage, "fixup": job.get("options", {}).get("capability_fixup", True),
                "attr": job.get("attr") if stage == "vs" else None,
                "fout": job.get("fout") if stage == "ps" else None,
                "tex": job.get("tex") if job.get("act") == "tex" else None}
        data = json.dumps(body, sort_keys=True).encode()
        res = {"notes": [], "capabilities_added": [], "passes": 1, "raw_access": {"physical_pointers": False}}
        if stage == "vs":
            res["capabilities_added"] = ["ClipDistance", "CullDistance"]
        if stage == "ps":
            res["passes"] = 2
        if job.get("act") == "tex":
            res["notes"] = ["translator: Invalid reduction type: F32FtzRn.",
                            "texture handle 1A2B (constant buffer 2, slot 3) has no type in the job, using 2D"]
            res["raw_access"]["physical_pointers"] = True
        module = base64.b64encode(data)
        desc = base64.b64encode(json.dumps(res, sort_keys=True).encode())
        print(f"ok {key} 100 {module.decode()} {desc.decode()}", flush=True)
""")

JOBS = [
    {"key": "a.00000001", "stage": "vs"},
    {"key": "b.00000002", "stage": "ps"},
    {"key": "c.00000003", "stage": "cs", "act": "tex"},
    {"key": "d.00000004", "stage": "cs"},
    {"key": "e.00000005", "stage": "cs", "act": "fail"},
]


class CorpusRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        cls.split = base / "split"
        cls.split.mkdir()
        (cls.split / "jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in JOBS), encoding="utf-8")
        (cls.split / "programs.tsv").write_text(
            "job_key\tstage\tgroup\tonly_product100\n"
            + "".join(f"{j['key']}\t{j['stage']}\teffect\t0\n" for j in JOBS), encoding="utf-8")
        cls.xlate = base / "fake_xlate.py"
        cls.xlate.write_text(FAKE, encoding="utf-8")
        cls.work = base / "work"
        args = ["run", "--split", str(cls.split), "--xlate", str(cls.xlate), "--work", str(cls.work),
                "--workers", "2", "--no-validate"]
        for name in ("default", "raw", "repeat", "tex", "attr", "fout"):
            args += ["--pass", name]
        cls.code = corpus.main(args)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def summary(self, name):
        return json.loads((self.work / f"summary-{name}.json").read_text(encoding="utf-8"))

    def test_run_succeeds(self):
        self.assertEqual(self.code, 0)

    def test_default_pass_counts_and_failures(self):
        s = self.summary("default")
        self.assertEqual(s["jobs"], 5)
        self.assertEqual(s["status"], {"fail": 1, "ok": 4})
        self.assertEqual(s["per_stage"]["cs"]["status"], {"fail": 1, "ok": 2})
        (failure,) = s["failures"]
        self.assertEqual((failure["kind"], failure["count"], failure["stages"]), ("unsupported_opcode", 1, {"cs": 1}))
        self.assertNotIn("1a2b", failure["message"])
        self.assertNotIn("340", failure["message"])
        self.assertEqual(s["by_group"][0]["group"], "effect")

    def test_answer_passes_change_only_the_programs_that_read_the_answer(self):
        self.assertEqual(self.summary("attr")["compare"]["changed"], {"vs": 1})
        self.assertEqual(self.summary("attr")["jobs"], 1)
        self.assertEqual(self.summary("fout")["compare"]["changed"], {"ps": 1})
        self.assertEqual(self.summary("tex")["compare"]["changed"], {"cs": 1})
        self.assertEqual(self.summary("tex")["compare"]["same"], {"cs": 1, "ps": 1, "vs": 1})

    def test_repeat_is_byte_identical(self):
        c = self.summary("repeat")["compare"]
        self.assertEqual(c["changed"], {})
        self.assertEqual(sum(c["same"].values()), 4)

    def test_raw_pass_turns_the_post_pass_off(self):
        spv = (self.work / "raw" / "spv" / "a.00000001.spv").read_text(encoding="utf-8")
        self.assertFalse(json.loads(spv)["fixup"])

    def test_notes_count_dropped_operations_and_defaults(self):
        self.assertEqual(corpus.main(["notes", "--split", str(self.split), "--work", str(self.work)]), 0)
        n = json.loads((self.work / "notes-default.json").read_text(encoding="utf-8"))
        by_note = {x["note"]: x for x in n["notes"]}
        self.assertEqual(by_note["translator: Invalid reduction type: F32FtzRn."]["stages"], {"cs": 1})
        self.assertIn("texture handle N (constant buffer N, slot N) has no type in the job, using 2D", by_note)
        self.assertNotIn("1A2B", json.dumps(n))
        self.assertEqual(n["capabilities_added"], {"ClipDistance": {"vs": 1}, "CullDistance": {"vs": 1}})
        self.assertIn({"stage": "ps", "passes": 2, "programs": 1}, n["passes"])
        self.assertEqual(n["raw_access"], {"cs": 1})
        self.assertEqual(n["unreadable"], 0)

    def test_resume_translates_nothing_twice(self):
        args = ["run", "--split", str(self.split), "--xlate", str(self.xlate), "--work", str(self.work),
                "--workers", "1", "--no-validate", "--pass", "default"]
        self.assertEqual(corpus.main(args), 0)
        self.assertEqual(self.summary("default")["run"]["translated_now"], 0)
        self.assertEqual(self.summary("default")["jobs"], 5)


class CorpusHelpersTest(unittest.TestCase):
    def test_work_folder_inside_the_repository_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = Path(tmp)
            (split / "jobs.jsonl").write_text("", encoding="utf-8")
            code = corpus.main(["run", "--split", str(split), "--xlate", sys.executable,
                                "--work", str(ROOT / "work-s1"), "--no-validate"])
        self.assertEqual(code, 2)
        self.assertFalse((ROOT / "work-s1").exists())

    def test_normalise_drops_numbers_and_ids(self):
        self.assertEqual(corpus.normalise("error: line 812: ID '%42' has type 0x1F and 3 members"),
                         "error: ID '%id' has type <hex> and N members")
        self.assertEqual(corpus.normalise("texture handle 1A2B (constant buffer 2, slot 3) has no type, using 2D"),
                         "texture handle N (constant buffer N, slot N) has no type, using 2D")
        self.assertEqual(corpus.normalise("translator: Invalid reduction type: F32FtzRn."),
                         "translator: Invalid reduction type: F32FtzRn.")

    def test_normalise_keeps_the_opcode_and_drops_the_instruction_words(self):
        msg = ("unsupported_opcode: Ampere instruction 98e at 0x1c0 "
               "(000fe2000c10e3800123456789abcdef) is not implemented")
        self.assertEqual(corpus.normalise(msg), "unsupported_opcode: Ampere instruction 98e at <hex> (N) is not implemented")

    def test_fail_kind(self):
        self.assertEqual(corpus.fail_kind("translate: index out of range"), "translate")
        self.assertEqual(corpus.fail_kind("Something odd: x"), "other")

    def test_variant_job_merges_options_without_touching_the_original(self):
        job = {"key": "k", "options": {"debug": True}}
        out = corpus.variant_job(job, {"options": {"capability_fixup": False}, "attr": {"*": "sint"}})
        self.assertEqual(out["options"], {"debug": True, "capability_fixup": False})
        self.assertEqual(out["attr"], {"*": "sint"})
        self.assertEqual(job, {"key": "k", "options": {"debug": True}})

    def test_nodebug_pass_turns_the_optimizer_debug_mode_off_and_keeps_the_other_options(self):
        change, stages = corpus.VARIANTS["nodebug"]
        self.assertEqual(set(stages), set(corpus.STAGES))
        job = {"key": "k", "stage": "ps", "options": {"capability_fixup": True}}
        out = corpus.variant_job(job, change)
        self.assertEqual(out["options"], {"capability_fixup": True, "debug": False})
        self.assertEqual(job["options"], {"capability_fixup": True})
        self.assertIn("nodebug", corpus.PASSES)

    def test_debug_pass_asks_for_the_old_optimizer_form_and_the_default_pass_names_no_option(self):
        """The sound form is the translator's default, so `default` carries no option and `debug` is the explicit way to
        the old form for A/B work (opcheck compares the two)."""
        change, stages = corpus.VARIANTS["debug"]
        self.assertEqual(set(stages), set(corpus.STAGES))
        job = {"key": "k", "stage": "ps", "options": {"capability_fixup": True}}
        out = corpus.variant_job(job, change)
        self.assertEqual(out["options"], {"capability_fixup": True, "debug": True})
        self.assertEqual(job["options"], {"capability_fixup": True})
        self.assertIn("debug", corpus.PASSES)
        self.assertEqual(corpus.variant_job(job, corpus.VARIANTS["default"][0]), job)

    def test_repeat_sample_is_spread_and_stable(self):
        jobs = {f"k{i:05d}": {"stage": "vs"} for i in range(corpus.REPEAT_SAMPLE * 3)}
        keys = corpus.select(jobs, "repeat", corpus.STAGES, None)
        self.assertEqual(len(keys), corpus.REPEAT_SAMPLE)
        self.assertEqual(keys, corpus.select(jobs, "repeat", corpus.STAGES, None))
        self.assertEqual(keys[0], "k00000")
        self.assertGreater(keys[-1], "k05000")


if __name__ == "__main__":
    unittest.main()
