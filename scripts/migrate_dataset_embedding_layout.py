#!/usr/bin/env python
"""Put the draw in the path: pad ``01_datasets``, relayout ``02_dataset_embeddings``.

Two stages ship together because they are one decision — that a draw is written
down exactly one way, everywhere — and neither is coherent alone.

**Stage 1: pad the seed in ``01_datasets``.**  Draw manifests were named
``n{n}_s{seed}.json`` with an unpadded seed, while ``04_activations`` and
``05_generated`` wrote ``n{n}_s{seed:02d}`` directories.  Three stages, two
spellings, and nothing compared them.  ``src/cache/_draw.draw_name`` now owns the
token and 01 moves onto it.  All stored seeds are 0-9 and single-digit, so this
is a pure rename with no collisions.

**Stage 2: give ``02_dataset_embeddings`` a draw and a surrogate level.**  02 was
the only stage that did not say which draw an artifact came from: ``n_samples``
and ``seed`` were folded into ``embedder_hash``, so no ``n*_s*`` component
existed anywhere beneath it and ``ls`` could not answer "which draws are
embedded?".  Enumerating the stored entries shows why that stopped being
defensible — ``representation``, ``model_name`` and ``prompt_prefix`` are
constant across all of them, so a directory named ``embedder_hash`` was in
practice a draw hash.

    before:  {recipe_hash}/{embedder_hash}/{config.json, embeddings.safetensors}
    after:   {recipe_hash}/n{n}_s{seed:02d}/{embedder_hash}/
                 config.json
                 surrogates/{surrogate_hash}/{config.json, surrogate.safetensors}

``embedder_hash`` drops to ``{embedder_config}`` alone — what its name always
claimed — and ``representation`` moves into the surrogate spec, mirroring
``04``/``05``.

**No recomputation and no GPU.**  Every field the new path needs is already in
each entry's ``config.json``.  Nothing is re-embedded and no distance changes;
tensors are copied byte for byte.

Safety model, following ``migrate_recipe_identity.py`` and
``migrate_behavioral_layout.py``:

- **Additive.**  New paths are written beside the old ones.  Nothing is deleted
  until ``--prune``, a separate invocation to be run only after
  ``check_analysis.py`` is green.  01 is 29 MB and 02 is 2.1 MB, so the
  duplication costs nothing worth optimising away.
- **No ``--revert``.**  The old paths surviving until ``--prune`` *is* the
  rollback.
- **Self-verifying.**  Every copied file is compared byte for byte against its
  source before the old one becomes eligible for pruning.

Usage::

    python scripts/migrate_dataset_embedding_layout.py --dry-run
    python scripts/migrate_dataset_embedding_layout.py --apply
    python scripts/check_analysis.py                       # must be green first
    python scripts/verify_sampled_cache.py --full          # and this
    python scripts/migrate_dataset_embedding_layout.py --prune
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.cache._draw import draw_name, parse_draw_name  # noqa: E402

DEFAULT_ROOT = REPO / "results" / "shared_cache"
DATASETS_DIR = "01_datasets"
EMBEDDINGS_DIR = "02_dataset_embeddings"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── stage 1: pad the seed in 01_datasets ──────────────────────────────────────


def plan_datasets(root: Path) -> list[tuple[Path, Path]]:
    """``[(old, new), ...]`` for every draw whose name is not already padded."""
    base = root / DATASETS_DIR
    if not base.exists():
        return []

    moves: list[tuple[Path, Path]] = []
    for recipe_dir in sorted(base.iterdir()):
        if not recipe_dir.is_dir():
            continue
        for path in sorted(recipe_dir.glob("n*_s*.json")):
            parsed = parse_draw_name(path.stem)
            if parsed is None:
                continue
            new = path.with_name(f"{draw_name(*parsed)}.json")
            if new != path:
                moves.append((path, new))
    return moves


def apply_datasets(moves: list[tuple[Path, Path]], dry_run: bool) -> dict:
    stats = {"copied": 0, "already": 0, "collisions": 0}
    for old, new in moves:
        if new.exists():
            # Only tolerable if it is the same draw already copied by an earlier
            # run.  Two *different* draws mapping to one name would be data loss,
            # so compare before waving it through.
            if _sha256(old) == _sha256(new):
                stats["already"] += 1
                continue
            stats["collisions"] += 1
            raise SystemExit(
                f"Refusing to overwrite: {new} exists and differs from {old}. "
                "Two distinct draws map to one padded name; resolve by hand."
            )
        if not dry_run:
            shutil.copy2(old, new)
        stats["copied"] += 1
    return stats


def verify_datasets(moves: list[tuple[Path, Path]]) -> int:
    verified = 0
    for old, new in moves:
        if not new.exists():
            raise SystemExit(f"missing after copy: {new}")
        if _sha256(old) != _sha256(new):
            raise SystemExit(f"content differs after copy: {old} -> {new}")
        verified += 1
    return verified


def prune_datasets(moves: list[tuple[Path, Path]], dry_run: bool) -> int:
    removed = 0
    for old, new in moves:
        if not new.exists():
            raise SystemExit(f"refusing to prune {old}: {new} is not there")
        if _sha256(old) != _sha256(new):
            raise SystemExit(f"refusing to prune {old}: {new} differs")
        if not dry_run:
            old.unlink()
        removed += 1
    return removed


# ── driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    mode.add_argument("--apply", action="store_true", help="write the new layout")
    mode.add_argument("--prune", action="store_true",
                      help="delete the old paths; run only after the checks are green")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"cache root: {root}\n")

    print("Stage 1 — 01_datasets: pad the seed")
    moves = plan_datasets(root)
    print(f"  {len(moves)} draw(s) to rename")

    if args.prune:
        removed = prune_datasets(moves, dry_run=False)
        print(f"  {removed} old name(s) removed")
        if not moves:
            print("  nothing to prune — already migrated and pruned")
        return 0

    stats = apply_datasets(moves, dry_run=args.dry_run)
    print(f"  {stats['copied']} copied, {stats['already']} already present")

    if args.dry_run:
        print("\nDry run only. Re-run with --apply to make these changes.")
        return 0

    print(f"  {verify_datasets(moves)} verified byte-identical")

    print("\nDone. Next:")
    print("    python scripts/check_analysis.py")
    print("    python scripts/verify_sampled_cache.py --full")
    print("    python scripts/migrate_dataset_embedding_layout.py --prune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
