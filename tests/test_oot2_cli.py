# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Tests for `oot2 ledger-absorb` (tools/oot2.py): the verb that merges the ledgers of other runs into the prepared
folder's ledger.jsonl, with the checks of the ledger step (tools/prepare/ledger.py absorb).

Run: python -m unittest discover -s tests -p test_oot2_cli.py -v
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))

import cache_pack as cp  # noqa: E402
import oot2  # noqa: E402
from prepare import ledger  # noqa: E402

KEY_A = "0123456789abcdef.0"
KEY_B = "fedcba9876543210.0"


def row(key: str, kind: str, **answers: dict) -> dict:
    return {"key": key, "kind": kind, "answers": answers, "digest": cp.digest_text(cp.answers_digest(answers))}


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join([json.dumps(ledger.HEADER, separators=(",", ":"))] +
                     [json.dumps(r, separators=(",", ":")) for r in rows]) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return path


class LedgerAbsorbCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="oot2-cli-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "game"
        self.target = self.out / ledger.LEDGER_FILE

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            rc = oot2.main(["ledger-absorb", *argv])
        return rc, o.getvalue(), e.getvalue()

    def test_it_merges_reports_and_keeps_a_backup(self):
        write(self.target, [row(KEY_A, "vs", tex={"0": "3d"})])
        run = write(self.tmp / "r" / ledger.LEDGER_FILE, [row(KEY_A, "vs", tex={"0": "3d"}),
                                                          row(KEY_B, "ps", tex={"1": "cube"})])
        rc, out, err = self.run_cli("--out", str(self.out), str(run))
        self.assertEqual((rc, err), (0, ""))
        self.assertIn("added 1 rows (1 new program and answers variants)", out)
        self.assertIn(f"{run}: 2 rows, 1 new", out)
        self.assertEqual(len(ledger.read_ledger(self.target)[0]), 2)
        self.assertTrue(self.target.with_name("ledger.jsonl.bak").exists())

    def test_a_second_call_with_the_same_rows_changes_nothing(self):
        run = write(self.tmp / "r" / ledger.LEDGER_FILE, [row(KEY_A, "vs", tex={"0": "3d"})])
        self.run_cli("--out", str(self.out), str(run))
        sha = hashlib.sha256(self.target.read_bytes()).hexdigest()
        rc, out, _ = self.run_cli("--out", str(self.out), str(run))
        self.assertEqual(rc, 0)
        self.assertIn("unchanged, nothing written", out)
        self.assertEqual(hashlib.sha256(self.target.read_bytes()).hexdigest(), sha)

    def test_dry_run_prints_the_counts_and_writes_nothing(self):
        run = write(self.tmp / "r" / ledger.LEDGER_FILE, [row(KEY_A, "vs", tex={"0": "3d"})])
        rc, out, _ = self.run_cli("--out", str(self.out), "--dry-run", str(run))
        self.assertEqual(rc, 0)
        self.assertIn("would add 1 rows", out)
        self.assertFalse(self.out.exists())

    def test_a_bad_input_exits_1_names_file_and_line_and_writes_nothing(self):
        write(self.target, [row(KEY_A, "vs", tex={"0": "3d"})])
        before = self.target.read_bytes()
        bad = self.tmp / "bad" / ledger.LEDGER_FILE
        write(bad, [row(KEY_B, "ps", tex={"1": "cube"})])
        bad.write_text(bad.read_text(encoding="utf-8") + "{oops\n", encoding="utf-8")
        rc, out, err = self.run_cli("--out", str(self.out), str(bad))
        self.assertEqual(rc, 1)
        self.assertIn(str(bad), err)
        self.assertIn("line 3", err)
        self.assertEqual(self.target.read_bytes(), before)

    def test_no_input_is_a_bad_command_line(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            oot2.main(["ledger-absorb", "--out", str(self.out)])
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
