# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The oot2-native contributors
"""The seed of the pipelines step (tools/prepare/pipeseed.py, pipelines.py), with no GPU: when the step's key changes the
previous pipeline cache is kept as the seed of the next build, after checks, and a cache that fails one is dropped by
name. The stand-in builder of test_prepare_pipelines.py speaks the real builder's protocol (a cache file that begins
with a real pipeline cache header, `--cache` on the command line, `cache=` and `seeded=` in the ready line); the real
builder serving a seeded cache on a device is test_pipebuild.py.

What is held here: the header rule (version one, this vendor, device and cache UUID, a file longer than its header),
the record rule (size, runtime, device and driver of the run that wrote the cache), what happens to the file (moved to
the seed folder, deleted when dropped, removed once merged), that the builders are given it, that a builder which refuses
it is reported, that an interrupted run keeps it for the next, that a cold build takes none, that a cache the full
check failed on is not one, that a seeded builder is not flushed on a timer (a flush empties its cache, seed included),
and that the step counts the seed in the memory it gives each worker.
"""

from __future__ import annotations

import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "shaders"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare import pipelines, pipesched, pipeseed  # noqa: E402
from prepare.pipelines import Pipelines  # noqa: E402
from prepare.runner import Context, Reporter, run_steps  # noqa: E402
from test_prepare_pipelines import Env, FakeTranslate, cache_keys, device, host  # noqa: E402

UUID = "ab" * 16
EXPECT = {"vendor": "10de", "device": "2486", "driver": "0009a000", "cache_uuid": UUID}
IDENTITY = f"10de:2486:0009a000:{UUID}:1.4.351"
RUNTIME = {"zero_workgroup": "1"}


def header(*, size=32, version=1, vendor=0x10DE, dev=0x2486, uuid=UUID) -> bytes:
    return struct.pack("<IIII16s", size, version, vendor, dev, bytes.fromhex(uuid))


class SettleTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="oot2-seed-")
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name) / "pipelines" / "pipeline.cache"
        self.out.parent.mkdir()
        self.dir = Path(tmp.name) / "temp" / "pipelines-seed"

    def put(self, data: bytes) -> dict:
        self.out.write_bytes(data)
        return {"cache_bytes": len(data), "runtime": dict(RUNTIME), "device": {"identity": IDENTITY}}

    def settle(self, record, **kw) -> pipeseed.Seed:
        args = dict(expect=EXPECT, identity=IDENTITY, runtime=RUNTIME, previous=record)
        args.update(kw)
        return pipeseed.settle(self.out, self.dir, **args)

    def assert_dropped(self, seed: pipeseed.Seed, *words: str) -> None:
        self.assertEqual(seed.outcome, pipeseed.DROPPED)
        for w in words:
            self.assertIn(w, seed.reason)
        self.assertFalse(self.out.exists())                  # deleted, as the step always deleted the old cache
        self.assertFalse(self.dir.exists())                  # and nothing is left for a builder to read
        self.assertFalse(seed.used)

    def test_a_cache_of_this_device_and_driver_is_moved_to_the_seed_folder_and_kept(self) -> None:
        data = header() + b"keys"
        seed = self.settle(self.put(data))
        self.assertEqual((seed.outcome, seed.bytes, seed.path), (pipeseed.KEPT, len(data), self.dir / "pipeline.cache"))
        self.assertTrue(seed.used)
        self.assertEqual(seed.path.read_bytes(), data)
        self.assertFalse(self.out.exists())                  # the step writes pipelines/pipeline.cache anew
        self.assertTrue((self.dir / "seed.json").is_file())

    def test_a_first_run_has_no_record_and_the_header_alone_decides(self) -> None:
        self.assertEqual(self.settle(None).outcome, pipeseed.NONE)
        self.put(header() + b"x")
        self.assertEqual(self.settle(None).outcome, pipeseed.KEPT)

    def test_a_cache_shorter_than_its_header_is_dropped_and_named(self) -> None:
        for n in (0, 1, 31):
            seed = self.settle(self.put(header()[:n]))
            self.assert_dropped(seed, "shorter than the 32-byte header")

    def test_a_header_that_claims_more_than_the_file_holds_is_dropped(self) -> None:
        seed = self.settle(self.put(header(size=4096) + b"x"))
        self.assert_dropped(seed, "4096 bytes of header", "cut short")

    def test_a_header_of_another_version_is_dropped(self) -> None:
        self.assert_dropped(self.settle(self.put(header(version=2) + b"x")), "header version is 2")

    def test_a_cache_of_another_device_or_driver_is_dropped_by_its_uuid_vendor_and_device(self) -> None:
        for kw in ({"uuid": "cd" * 16}, {"vendor": 0x1002}, {"dev": 0x1111}):
            seed = self.settle(self.put(header(**kw) + b"x"))
            self.assert_dropped(seed, "another device or driver", UUID)

    def test_a_cache_cut_short_since_the_record_was_written_is_dropped(self) -> None:
        record = self.put(header() + b"x" * 100)
        self.out.write_bytes(header() + b"x" * 40)            # the header is whole, the rest is gone
        self.assert_dropped(self.settle(record), "says 132", "cut short or changed")

    def test_a_cache_of_another_runtime_is_dropped(self) -> None:
        record = self.put(header() + b"x")
        for said in ({}, {"zero_workgroup": "0"}, None):
            self.put(header() + b"x")
            record["runtime"] = said
            self.assert_dropped(self.settle(record), "another runtime")

    def test_a_cache_of_another_driver_version_is_dropped_though_its_uuid_is_the_same(self) -> None:
        record = self.put(header() + b"x")
        record["device"] = {"identity": f"10de:2486:0009b000:{UUID}:1.4.351"}
        self.assert_dropped(self.settle(record), "another device or driver", "0009b000")

    def test_a_skipped_record_holds_the_cache_to_nothing_but_its_header(self) -> None:
        self.put(header() + b"x")
        self.assertEqual(self.settle({"skipped": "why"}).outcome, pipeseed.KEPT)

    def test_a_cache_the_full_check_just_failed_on_is_dropped_by_name_whatever_its_header_says(self) -> None:
        record = self.put(header() + b"x")
        self.assert_dropped(self.settle(record, damaged=True), "full check failed")
        self.assertEqual(self.settle(self.put(header() + b"x")).outcome, pipeseed.KEPT)     # the same cache, no failed check

    def test_a_cold_build_takes_no_seed_and_deletes_the_old_cache(self) -> None:
        seed = self.settle(self.put(header() + b"x"), cold=True)
        self.assertEqual(seed.outcome, pipeseed.OFF)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.dir.exists())

    def test_the_seed_of_an_interrupted_run_is_judged_again_by_what_was_claimed_when_it_was_kept(self) -> None:
        data = header() + b"x" * 10
        self.assertTrue(self.settle(self.put(data)).used)
        again = self.settle(None)                              # the next run: no output, the seed folder alone
        self.assertEqual((again.outcome, again.source), (pipeseed.KEPT, "the seed of an interrupted run"))
        other = self.settle(None, identity=f"10de:2486:0009b000:{UUID}:1.4.351")    # and the driver was updated since
        self.assert_dropped(other, "another device or driver")

    def test_a_seed_whose_claims_are_gone_is_dropped(self) -> None:
        self.assertTrue(self.settle(self.put(header() + b"x")).used)
        (self.dir / "seed.json").unlink()
        self.assert_dropped_in_place(self.settle(None))

    def assert_dropped_in_place(self, seed: pipeseed.Seed) -> None:
        self.assertEqual(seed.outcome, pipeseed.DROPPED)
        self.assertIn("claimed", seed.reason)
        self.assertFalse(self.dir.exists())


class StepSeedTest(unittest.TestCase):
    """The step end to end with the stand-in builder: a change of its key seeds the next build."""

    def setUp(self) -> None:
        self.env = Env(self)
        self.marker = self.env.tmp / "marker.txt"
        base = pipelines.key_files()
        patch = mock.patch.object(pipelines, "key_files", lambda: [*base, self.marker])
        patch.start()
        self.addCleanup(patch.stop)
        self.marker.write_text("one")
        self.status, self.ctx = self.env.run()
        self.assertEqual(self.status, "ok", self.env.say(self.ctx))
        self.cache = self.env.dest / "pipelines" / "pipeline.cache"
        self.first_keys = sorted(cache_keys(self.cache))

    def change_key(self, text: str = "two") -> None:
        self.marker.write_text(text)

    def serves(self) -> list[str]:
        return [line for line in self.env.log.read_text().splitlines() if line.startswith("serve ")]

    def hits(self) -> dict[str, str]:
        rows = (self.env.dest / "pipelines" / "times.tsv").read_text().splitlines()
        head = rows[0].split("\t")
        return {r.split("\t")[1]: r.split("\t")[head.index("hit")] for r in rows[1:]}

    def test_a_first_build_has_no_seed(self) -> None:
        rec = self.ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["seed"]["outcome"], pipeseed.NONE)
        self.assertEqual(self.serves(), ["serve no-cache none"])
        self.assertEqual(set(self.hits().values()), {"0"})

    def test_a_key_change_serves_every_unchanged_program_from_the_seed_and_builds_the_changed_one_cold(self) -> None:
        changed = "c000000000000002.00000000"
        self.change_key()
        self.env.standin(cold_keys=[changed])                  # the one module whose create info changed
        status, ctx = self.env.run(self.ctx)
        said = self.env.say(ctx)
        self.assertEqual(status, "ok", said)
        self.assertIn("pipelines: started (its inputs changed)", said)
        self.assertEqual(self.serves()[1:], ["serve --cache loaded"])
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["seed"]["outcome"], rec["seed"]["starts_loaded"]), (pipeseed.KEPT, 1))
        self.assertGreater(rec["seed"]["bytes"], 32)
        hits = self.hits()
        self.assertEqual(len(hits), 10)
        self.assertEqual({k for k, v in hits.items() if v == "0"}, {changed})        # the changed one built cold
        self.assertEqual({v for k, v in hits.items() if k != changed}, {"1"})        # every other one was served
        self.assertEqual(rec["cache_hits"], 9)
        self.assertEqual(sorted(cache_keys(self.cache)), self.first_keys)            # the new cache holds the seed's entries
        self.assertFalse((self.env.dest / "temp" / "pipelines-seed").exists())       # merged, so removed
        self.assertFalse((self.env.dest / "temp" / "pipelines").exists())
        self.assertIn("is the seed", said)

    def test_a_foreign_seed_is_dropped_by_name_and_the_step_builds_from_an_empty_cache(self) -> None:
        data = bytearray(self.cache.read_bytes())
        data[16] ^= 0xFF                                       # the first byte of the header's cache UUID
        self.cache.write_bytes(bytes(data))
        self.change_key()
        status, ctx = self.env.run(self.ctx)
        said = self.env.say(ctx)
        self.assertEqual(status, "ok", said)
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["seed"]["outcome"], pipeseed.DROPPED)
        self.assertIn("another device or driver", rec["seed"]["reason"])
        self.assertIn("warning: pipelines: the previous cache", said)
        self.assertIn("another device or driver", said)
        self.assertEqual(self.serves()[1:], ["serve no-cache none"])
        self.assertEqual(set(self.hits().values()), {"0"})
        self.assertEqual(rec["cache_hits"], 0)
        self.assertEqual(sorted(cache_keys(self.cache)), self.first_keys)              # a whole cache, built again

    def test_a_truncated_seed_is_dropped_by_name(self) -> None:
        for cut, word in ((20, "shorter than the 32-byte header"), (self.cache.stat().st_size - 3, "cut short or changed")):
            data = self.cache.read_bytes()[:cut]
            self.cache.write_bytes(data)
            self.change_key(f"cut {cut}")
            status, ctx = self.env.run(self.ctx)
            said = self.env.say(ctx)
            self.assertEqual(status, "ok", said)
            rec = ctx.state["steps"]["pipelines"]
            self.assertEqual(rec["seed"]["outcome"], pipeseed.DROPPED, said)
            self.assertIn(word, rec["seed"]["reason"])
            self.assertIn(word, said)
            self.assertEqual(self.serves()[-1], "serve no-cache none")
            self.assertEqual(sorted(cache_keys(self.cache)), self.first_keys)

    def test_a_driver_update_drops_the_seed_by_the_record_of_the_run_that_wrote_it(self) -> None:
        self.ctx.state["host"] = host(device(driver=0x9B000))     # the same cache UUID, another driver version
        self.env.standin(driver="0009b000")
        status, ctx = self.env.run(self.ctx)
        self.assertEqual(status, "ok", self.env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["seed"]["outcome"], pipeseed.DROPPED)
        self.assertIn("0009a000", rec["seed"]["reason"])

    def test_a_builder_that_refuses_the_seed_is_reported(self) -> None:
        self.change_key()
        self.env.standin(cache_state="other_driver")
        status, ctx = self.env.run(self.ctx)
        said = self.env.say(ctx)
        self.assertEqual(status, "ok", said)
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["seed"]["outcome"], rec["seed"]["starts_loaded"]), (pipeseed.DROPPED, 0))
        self.assertIn("other_driver", rec["seed"]["reason"])
        self.assertIn("the seed was not used by every builder", said)

    def test_a_cold_build_takes_no_seed(self) -> None:
        self.change_key()
        status, ctx = self.env.run(self.env.context(pipelines_cold=True))
        self.assertEqual(status, "ok", self.env.say(ctx))
        self.assertEqual(ctx.state["steps"]["pipelines"]["seed"]["outcome"], pipeseed.OFF)
        self.assertEqual(self.serves()[1:], ["serve no-cache none"])
        self.assertEqual(set(self.hits().values()), {"0"})

    def test_an_interrupted_run_keeps_the_seed_for_the_next_run_even_after_another_key_change(self) -> None:
        self.change_key("two")
        with mock.patch.object(pipesched.Pool, "run", side_effect=KeyboardInterrupt):
            ctx = self.env.context()
            self.assertEqual(run_steps([FakeTranslate(), Pipelines()], ctx), "cancelled")
        seed_dir = self.env.dest / "temp" / "pipelines-seed"
        self.assertTrue((seed_dir / "pipeline.cache").is_file())      # the output was taken as the seed and not given back
        self.assertFalse(self.cache.exists())
        self.change_key("three")                                      # the key changes again before the run is resumed
        status, ctx = self.env.run(self.env.context())
        self.assertEqual(status, "ok", self.env.say(ctx))
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual((rec["seed"]["outcome"], rec["seed"]["source"]),
                         (pipeseed.KEPT, "the seed of an interrupted run"))
        self.assertEqual(set(self.hits().values()), {"1"})
        self.assertFalse(seed_dir.exists())

    def run_spied(self):
        """Run the step after a key change; return the pools it made and what size_workers was called with."""
        self.change_key()
        pools, sized = [], []
        real_pool, real_size = pipesched.Pool, pipesched.size_workers

        class Spy(real_pool):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                pools.append(self)

        def size_spy(*args, **kw):
            sized.append((args, kw))
            return real_size(*args, **kw)

        with mock.patch.object(pipesched, "Pool", Spy), mock.patch.object(pipesched, "size_workers", size_spy):
            status, ctx = self.env.run(self.ctx)
        self.assertEqual(status, "ok", self.env.say(ctx))
        return pools, sized

    def test_a_seeded_builder_does_not_flush_on_a_timer_so_the_seed_serves_every_unchanged_program(self) -> None:
        # The real builder goes on with an empty cache after a flush (vkp_cache_flush), as the stand-in does: a builder
        # flushed on a timer would serve the seed to its first program and build the rest cold.
        changed = "c000000000000002.00000000"
        self.change_key()
        self.env.standin(cold_keys=[changed])
        with mock.patch.object(pipesched, "FLUSH_SECONDS", 0.0):          # a flush after every program, were it asked
            status, ctx = self.env.run(self.ctx)
        self.assertEqual(status, "ok", self.env.say(ctx))
        hits = self.hits()
        self.assertEqual({k for k, v in hits.items() if v == "0"}, {changed}, hits)
        self.assertEqual(ctx.state["steps"]["pipelines"]["segments"], 1)   # the final flush alone
        self.assertEqual(sorted(cache_keys(self.cache)), self.first_keys)

    def test_a_build_with_no_seed_still_flushes_on_its_timer(self) -> None:
        self.change_key()
        data = bytearray(self.cache.read_bytes())
        data[16] ^= 0xFF                                       # a foreign seed is dropped, the build is unseeded
        self.cache.write_bytes(bytes(data))
        with mock.patch.object(pipesched, "FLUSH_SECONDS", 0.0):
            status, ctx = self.env.run(self.ctx)
        self.assertEqual(status, "ok", self.env.say(ctx))
        self.assertEqual(ctx.state["steps"]["pipelines"]["seed"]["outcome"], pipeseed.DROPPED)
        self.assertEqual(ctx.state["steps"]["pipelines"]["segments"], 10)  # one a program

    def test_the_seed_is_counted_in_the_memory_a_builder_may_hold_before_it_is_replaced(self) -> None:
        seed_bytes = self.cache.stat().st_size
        pools, _ = self.run_spied()
        self.assertEqual(pools[0].recycle_bytes, pipesched.RECYCLE_BYTES + seed_bytes)

    def test_the_seed_is_counted_in_the_memory_each_worker_is_given(self) -> None:
        seed_bytes = self.cache.stat().st_size
        _, sized = self.run_spied()
        args, kw = sized[0]
        self.assertEqual(kw.get("seed_bytes", args[3] if len(args) > 3 else None), seed_bytes)

    def test_a_cache_the_full_check_failed_on_is_not_the_seed_of_the_run_that_follows(self) -> None:
        data = self.cache.read_bytes()
        self.cache.write_bytes(data[:32] + b"x" * (len(data) - 32))     # the header and the size are whole, the keys are not
        ctx = Context(self.env.dest, self.env.options(), Reporter(io.StringIO(), io.StringIO()), state=self.ctx.state,
                      verify=True)
        status, ctx = self.env.run(ctx)
        said = self.env.say(ctx)
        self.assertEqual(status, "ok", said)
        self.assertIn("pipelines: started (its output failed the full check)", said)
        rec = ctx.state["steps"]["pipelines"]
        self.assertEqual(rec["seed"]["outcome"], pipeseed.DROPPED, said)
        self.assertIn("full check failed", rec["seed"]["reason"])
        self.assertEqual(self.serves()[-1], "serve no-cache none")      # (the full check's own warm pass is the one before)
        self.assertEqual(set(self.hits().values()), {"0"})


class SizingSeedTest(unittest.TestCase):
    def test_every_worker_counts_the_seed_it_loads(self) -> None:
        free = 8 << 30
        plain = pipesched.size_workers(free, 12)[0]
        seeded = pipesched.size_workers(free, 12, seed_bytes=1 << 30)[0]
        self.assertLess(seeded, plain)
        self.assertEqual(pipesched.size_workers(free, 12, given=5, seed_bytes=1 << 30)[0], 5)   # a number given is kept


if __name__ == "__main__":
    unittest.main()
