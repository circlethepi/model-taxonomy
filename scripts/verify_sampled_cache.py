#!/usr/bin/env python
"""Check that every cached draw still reproduces the rows it was recorded with.

Draws are stored as source indices rather than row text, so the cache is only as
trustworthy as the upstream dataset staying put.  Nothing about that is self-announcing:
if ``yahoo_answers_topics`` were re-uploaded in a different order, every index in the
cache would quietly point at the wrong row.  This script is what makes that loud.

Two depths:

    --fast   descriptors only.  Confirms each source still resolves, and that its
             revision and row count match what was recorded.  Seconds.
    --full   also rehydrates every draw and checks its rows_sha256.  This is the
             authoritative check — the one that catches an upstream change the
             revision and row count did not.  Minutes.

Run it after a migration, before trusting a long pipeline run, or on a schedule if the
cache matters.  ``--full`` is also what the identity migration calls to verify itself,
so the conversion and the audit cannot drift apart.

Usage::

    python scripts/verify_sampled_cache.py --fast
    python scripts/verify_sampled_cache.py --full
    python scripts/verify_sampled_cache.py --full --recipe 1a2b3c4d5e6f7890
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_ROOT = REPO / "results" / "shared_cache"


def iter_draws(datasets_root: Path, only: str | None = None):
    """Yield ``(recipe_hash, path)`` for every draw manifest in the cache."""
    for directory in sorted(datasets_root.iterdir()):
        if not directory.is_dir():
            continue
        if only and directory.name != only:
            continue
        for path in sorted(directory.glob("n*_s*.json")):
            yield directory.name, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--recipe", help="check only this recipe hash")
    depth = parser.add_mutually_exclusive_group(required=True)
    depth.add_argument("--fast", action="store_true", help="descriptors only")
    depth.add_argument("--full", action="store_true", help="rehydrate and checksum")
    args = parser.parse_args()

    datasets_root = args.root / "01_datasets"
    if not datasets_root.exists():
        print(f"No dataset cache at {datasets_root}")
        return 1

    from src.cache.sampled_dataset_cache import SampledDatasetCache, rows_checksum
    from src.datasets import source_registry

    cache = SampledDatasetCache(args.root)
    draws = list(iter_draws(datasets_root, args.recipe))
    print(f"{len(draws)} draw(s) under {datasets_root}")
    print(f"mode: {'full (rehydrate + checksum)' if args.full else 'fast (descriptors)'}\n")

    ok = legacy = failed = 0
    t0 = time.time()

    for i, (recipe_hash, path) in enumerate(draws, 1):
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            # Pre-index draw: the rows are right there, so there is nothing upstream to
            # verify against.  Reported rather than passed over, since after a migration
            # any remaining one is a draw that was missed.
            legacy += 1
            print(f"  LEGACY  {recipe_hash}/{path.name} — v1 rows, not index-backed")
            continue

        try:
            for desc in payload["sources"]:
                ds = source_registry.get(
                    desc["dataset_id"], desc.get("subset"), desc.get("split", "train"),
                    revision=desc.get("revision"),
                )
                source_registry.validate(
                    desc,
                    source_registry.describe(
                        ds, desc["dataset_id"], desc.get("subset"),
                        desc.get("split", "train"),
                    ),
                )

            if args.full:
                n_samples, seed = payload["n_samples"], payload["seed"]
                rows = cache.get(recipe_hash, n_samples, seed)
                if rows is None:
                    raise RuntimeError("draw vanished between listing and reading")
                # cache.get already raises on a checksum mismatch; re-checking here
                # keeps this script honest if that ever becomes lenient.
                recorded = payload.get("rows_sha256")
                if recorded and rows_checksum(rows) != recorded:
                    raise RuntimeError("rows_sha256 mismatch after rehydration")
            ok += 1
        except Exception as exc:  # noqa: BLE001 — the report is the point
            failed += 1
            print(f"  FAIL    {recipe_hash}/{path.name}: {exc}")

        if args.full and i % 50 == 0:
            print(f"  ... {i}/{len(draws)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{ok} ok, {legacy} legacy (v1), {failed} failed  "
          f"[{time.time() - t0:.0f}s]")
    if failed:
        print("\nA failure means the upstream data no longer reproduces the recorded "
              "draw. The rows themselves are not recoverable from the cache — re-sample "
              "and accept new rows, or restore the pinned dataset revision.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
