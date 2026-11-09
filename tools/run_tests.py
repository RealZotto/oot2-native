#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""Run the unit tests one module per process, several modules at a time.

`python -m unittest discover -s tests` runs every test on one core: about 20
minutes on the Windows runner and 15 on the Linux one, where almost all of a CI
job's time went. Most of that is clang compiling and running the runtime's C
tests, one module after another, while the runner's other cores wait. Here each
tests/test_*.py module runs as its own `unittest discover -p <file>` process
and up to --jobs of them run at once. The output of a module is printed in one
block when it ends, so a log still reads module by module, and the summary
ends with the same `Ran N tests` and `OK (skipped=N)` lines a plain run prints.

A module is a failure when its process exits non-zero, when it prints no `Ran`
line (it crashed before unittest could report), or when it outlives --timeout.
The modules named in SERIAL run alone, after the others, because they assert a
wall-clock bound that a busy machine could break; LONGEST start first so the
last module to finish is a short one.

Usage:
  python tools/run_tests.py [--jobs N] [--timeout S] [--tests-dir DIR] [MODULE ...]

MODULE is a file name (test_x.py) or a module name (test_x); none means all.
Exit status: 0 when every module passed, 1 when one failed, 2 on bad usage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Modules that assert a wall-clock bound on real threads, so they run one at a
# time after the parallel part (see the docstring). Each with the bound a busy
# machine could break, read from the C test (2026-11):
#   codec      submits must return under 400 ms while a 600 ms decode runs
#   coherence  a helper's 200 ms sleep, measured from another thread, >= 150 ms
#   gpu        a wait on a value never handed out returns under 100 ms
#   kernel     a first timed-out wait starts within 20 ms
#   nvn_sync   a real-clock 5 ms wait expires about 50 ms before the next tick
#   vk_miss    a 100 ms bound must skip a 500 ms build before it finishes
#   vsync      four waiters on the real clock keep a 4 ms p99 lateness
SERIAL: tuple[str, ...] = (
    "test_runtime_codec", "test_runtime_coherence", "test_runtime_gpu", "test_runtime_kernel",
    "test_runtime_nvn_sync", "test_runtime_vk_miss", "test_runtime_vsync",
)

# The longest modules, measured on both runners (2026-11): started first.
LONGEST: tuple[str, ...] = (
    "test_build_game", "test_runtime_untile", "test_runtime_vk_rec", "test_runtime_nvn_target",
    "test_runtime_audio", "test_runtime_backing", "test_pre_push", "test_pipebuild",
    "test_runtime_texmem", "test_runtime_capture", "test_runtime_vk_window", "test_runtime_hos",
)

RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
SKIPPED = re.compile(r"skipped=(\d+)")
VERDICT = re.compile(r"^(OK|FAILED)\b.*$", re.M)


@dataclass
class Result:
    module: str
    returncode: int | None
    output: str
    seconds: float
    ran: int | None = None
    skipped: int = 0
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        # A module cut at the timeout has no return code. A process that exits 0
        # before unittest reports (os._exit(0) in an import) has no Ran line.
        return self.returncode == 0 and self.ran is not None


def parse(result: Result) -> Result:
    """Read the counts of one module's unittest output into its result."""
    ran = RAN.findall(result.output)
    result.ran = int(ran[-1]) if ran else None
    verdicts = [m.group(0) for m in VERDICT.finditer(result.output)]
    skipped = SKIPPED.search(verdicts[-1]) if verdicts else None
    result.skipped = int(skipped.group(1)) if skipped else 0
    return result


def run_module(module: str, tests_dir: Path, timeout: float) -> Result:
    """Run one test module in its own interpreter and capture what it prints."""
    cmd = [sys.executable]
    if sys.flags.isolated:
        cmd.append("-I")
    cmd += ["-m", "unittest", "discover", "-s", str(tests_dir), "-p", module + ".py", "-v"]
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.output.decode("utf-8", "replace") if exc.output else ""
        return parse(Result(module, None, out, time.monotonic() - start, timed_out=True))
    out = proc.stdout.decode("utf-8", "replace")
    return parse(Result(module, proc.returncode, out, time.monotonic() - start))


def order(modules: list[str]) -> tuple[list[str], list[str]]:
    """Split the modules into the parallel part (longest first) and the serial part."""
    serial = [m for m in modules if m in SERIAL]
    rest = [m for m in modules if m not in SERIAL]
    rank = {m: i for i, m in enumerate(LONGEST)}
    rest.sort(key=lambda m: (rank.get(m, len(LONGEST)), m))
    return rest, sorted(serial)


def report(result: Result) -> None:
    state = "ok" if result.passed else ("TIMED OUT" if result.timed_out else "FAILED")
    print(f"===== {result.module}: {state} in {result.seconds:.1f}s", flush=True)
    sys.stdout.write(result.output if result.output.endswith("\n") or not result.output
                     else result.output + "\n")
    sys.stdout.flush()


def find_modules(tests_dir: Path, names: list[str]) -> list[str]:
    found = sorted(p.stem for p in tests_dir.glob("test_*.py"))
    if not names:
        return found
    wanted = [n[:-3] if n.endswith(".py") else n for n in names]
    missing = [n for n in wanted if n not in found]
    if missing:
        raise SystemExit(f"run_tests: no such module in {tests_dir}: {', '.join(missing)}")
    return wanted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", "-j", type=int, default=min(4, os.cpu_count() or 1),
                    help="modules run at once (default: the cores, at most 4)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds one module may take")
    ap.add_argument("--tests-dir", type=Path, default=ROOT / "tests")
    ap.add_argument("modules", nargs="*")
    args = ap.parse_args(argv)
    if args.jobs < 1 or args.timeout <= 0:
        ap.error("--jobs and --timeout must be positive")
    tests_dir = args.tests_dir.resolve()
    if not tests_dir.is_dir():
        ap.error(f"{tests_dir} is not a folder")
    parallel, serial = order(find_modules(tests_dir, args.modules))

    start = time.monotonic()
    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_module, m, tests_dir, args.timeout) for m in parallel]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
            report(results[-1])
    for m in serial:
        results.append(run_module(m, tests_dir, args.timeout))
        report(results[-1])
    elapsed = time.monotonic() - start

    failed = [r for r in results if not r.passed]
    ran = sum(r.ran or 0 for r in results)
    skipped = sum(r.skipped for r in results)
    print("=" * 70)
    print(f"{len(results)} modules, {args.jobs} at a time; the slowest: "
          + ", ".join(f"{r.module} {r.seconds:.0f}s"
                      for r in sorted(results, key=lambda r: (-r.seconds, r.module))[:5]))
    for r in sorted(failed, key=lambda r: r.module):
        why = "timed out" if r.timed_out else ("no Ran line" if r.ran is None else f"exit {r.returncode}")
        print(f"FAILED module {r.module} ({why})")
    print("-" * 70)
    print(f"Ran {ran} tests in {elapsed:.3f}s")
    print()
    if failed:
        print(f"FAILED (modules={len(failed)}, skipped={skipped})")
        return 1
    print(f"OK (skipped={skipped})" if skipped else "OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
