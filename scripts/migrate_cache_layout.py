"""One-shot migration: give the shared cache directories numeric stage prefixes.

The shared cache used to sit flat, so nothing in a directory listing said which
stages came first and which were derived from them.  Directories sharing a number
sit at the same pipeline stage; a letter suffix (``03A``) means "analysis of the
objects at that stage".

    sampled_datasets    → 01_datasets
    dataset_embeddings  → 02_dataset_embeddings
    adapters            → 03_adapters
    adapter_alignments  → 03A_adapter_alignments
    collections         → 07_collections
    representations     → deleted (empty; split into 04_activations / 05_generated,
                          both created on first write)

There is deliberately no compatibility shim in the cache classes: a call site that
was missed must fail loudly rather than quietly keep reading the old directory.

**A later renumbering moved collections again.**  This script originally produced
``06_collections``; ``scripts/migrate_pairwise_layout.py`` then moved that to
``07_collections`` to free ``06`` for the pairwise-distance store, which
collections are assembled from.  The mapping above names the *current* stage
rather than the historical one, so running this on an old cache lands it in
today's layout in one step instead of a layout that immediately needs migrating
again.

This also gives recipes a hash-indexed home.  Until now the only way to resolve a
``recipe_hash`` to its recipe was to reach into the dataset-embedding cache, which
fails for any recipe that was sampled but never embedded.  ``--apply`` mirrors a
``recipe.json`` into every ``01_datasets/{recipe_hash}/``.

Only the shared cache is migrated.  Legacy per-experiment cache trees under
``results/<experiment>/cache/`` keep the old names and are not readable by current
code; they are superseded work.

Usage:
    python scripts/migrate_cache_layout.py --dry-run
    python scripts/migrate_cache_layout.py --apply
    python scripts/migrate_cache_layout.py --revert
    python scripts/migrate_cache_layout.py --root some/other/cache --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.atomic import atomic_write_json  # noqa: E402

REPO = Path(__file__).parent.parent

DEFAULT_ROOT = REPO / "results" / "shared_cache"

#: old directory name → new directory name.
RENAMES = {
    "sampled_datasets": "01_datasets",
    "dataset_embeddings": "02_dataset_embeddings",
    "adapters": "03_adapters",
    "adapter_alignments": "03A_adapter_alignments",
    "collections": "07_collections",
}

#: Emptied by the 04/05 split.  Removed rather than moved; the two replacements are
#: created on first write by their DiskCache instances.
RETIRED = "representations"

DATASETS_DIR = RENAMES["sampled_datasets"]
EMBEDDINGS_DIR = RENAMES["dataset_embeddings"]


# ── Directory renames ─────────────────────────────────────────────────────────


def plan_renames(root: Path, revert: bool = False) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return the (src, dst) pairs to rename, plus human-readable notes on the rest.

    Refuses to plan a rename whose destination already exists — that means a
    partially applied migration, and merging two trees silently is worse than
    stopping.
    """
    pairs: list[tuple[Path, Path]] = []
    notes: list[str] = []

    mapping = {v: k for k, v in RENAMES.items()} if revert else dict(RENAMES)

    for old, new in mapping.items():
        src, dst = root / old, root / new
        if not src.exists():
            notes.append(f"skip   {old}/ — not present")
            continue
        if dst.exists():
            raise SystemExit(
                f"refusing to migrate: both {src} and {dst} exist. "
                "Resolve by hand — merging two cache trees silently would lose data."
            )
        pairs.append((src, dst))

    return pairs, notes


def apply_renames(pairs: list[tuple[Path, Path]]) -> None:
    """``os.rename`` each pair.  Within one filesystem this is a metadata operation,
    so it is effectively instant regardless of how many directories are inside."""
    for src, dst in pairs:
        os.rename(src, dst)
        print(f"  moved  {src.name}/ → {dst.name}/")


def retire_representations(root: Path, dry_run: bool) -> None:
    """Remove the emptied ``representations/`` DiskCache directory.

    Refuses if it is not empty: content there would be functional or behavioral
    representations that cannot be sorted into 04/05 without reading each entry.
    """
    path = root / RETIRED
    if not path.exists():
        print(f"  skip   {RETIRED}/ — not present")
        return

    entries = list(path.rglob("*"))
    if entries:
        print(
            f"  KEEP   {RETIRED}/ — {len(entries)} entr(ies) present, not empty. "
            "Its contents are content-hashed and cannot be sorted into "
            "04_activations / 05_generated without reading each one. Left in place."
        )
        return

    if dry_run:
        print(f"  would remove  {RETIRED}/ (empty)")
    else:
        path.rmdir()
        print(f"  removed  {RETIRED}/ (empty)")


# ── Recipe backfill ───────────────────────────────────────────────────────────


def _experiment_recipes() -> dict[str, dict]:
    """``recipe_hash → recipe dict`` for every recipe written by build_datasets.py.

    These live at ``results/<experiment>/datasets/{name}.recipe.json`` and are the
    only source for recipes that were sampled but never embedded.  The hash is
    recomputed rather than trusted from the file, so a stale ``recipe_hash`` field
    cannot introduce a wrong mapping.
    """
    from src.datasets.class_recipe import ClassAwareDatasetRecipe
    from src.datasets.recipe import DatasetRecipe

    found: dict[str, dict] = {}
    for path in sorted((REPO / "results").glob("*/datasets/*.recipe.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cls = (
            ClassAwareDatasetRecipe
            if payload.get("recipe_type") == "class_aware"
            else DatasetRecipe
        )
        try:
            recipe = cls.load(path)
        except (KeyError, TypeError, ValueError):
            continue
        found.setdefault(recipe.recipe_hash(), recipe.to_dict())
    return found


def _resolve(root: Path, new_name: str, old_name: str) -> Path:
    """The new directory if it exists, else the old one.

    Lets ``--dry-run`` preview the backfill before the rename has happened.
    """
    new = root / new_name
    return new if new.exists() else root / old_name


def backfill_recipes(root: Path, dry_run: bool) -> None:
    """Mirror a ``recipe.json`` into every ``01_datasets/{recipe_hash}/``.

    Preferred source is the matching copy already in the dataset-embedding cache.
    Anything left over is resolved by recomputing hashes over the experiment recipe
    files.  Unresolved hashes are reported, not raised: a sampled dataset whose
    recipe was never persisted anywhere is a gap in older runs, not a migration
    failure.
    """
    datasets_root = _resolve(root, DATASETS_DIR, "sampled_datasets")
    embeddings_root = _resolve(root, EMBEDDINGS_DIR, "dataset_embeddings")

    if not datasets_root.exists():
        print(f"  skip   recipe backfill — {DATASETS_DIR}/ not present")
        return

    hashes = sorted(d.name for d in datasets_root.iterdir() if d.is_dir())
    from_embeddings = from_experiments = already = 0
    unresolved: list[str] = []
    experiment_recipes: dict[str, dict] | None = None

    for recipe_hash in hashes:
        target = datasets_root / recipe_hash / "recipe.json"
        if target.exists():
            already += 1
            continue

        source = embeddings_root / recipe_hash / "recipe.json"
        if source.exists():
            if not dry_run:
                _write_json(target, json.loads(source.read_text()))
            from_embeddings += 1
            continue

        if experiment_recipes is None:
            experiment_recipes = _experiment_recipes()
        recipe = experiment_recipes.get(recipe_hash)
        if recipe is not None:
            if not dry_run:
                _write_json(target, recipe)
            from_experiments += 1
            continue

        unresolved.append(recipe_hash)

    verb = "would write" if dry_run else "wrote"
    print(
        f"  {verb} recipe.json for {from_embeddings + from_experiments} of "
        f"{len(hashes)} hash(es) in {datasets_root.name}/"
    )
    print(f"    from {embeddings_root.name}/: {from_embeddings}")
    print(f"    from results/*/datasets/*.recipe.json: {from_experiments}")
    if already:
        print(f"    already present: {already}")
    if unresolved:
        print(
            f"    UNRESOLVED: {len(unresolved)} hash(es) have sampled rows but no "
            f"recipe anywhere, e.g. {unresolved[0]}"
        )


def _write_json(path: Path, payload: dict) -> None:
    """Atomic write, consistent with the cache classes."""
    atomic_write_json(path, payload)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Give the shared cache directories numeric stage prefixes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Cache root to migrate (default: {DEFAULT_ROOT}).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print the plan, change nothing.")
    mode.add_argument("--apply", action="store_true", help="Rename directories and backfill recipes.")
    mode.add_argument("--revert", action="store_true", help="Rename the new names back to the old ones.")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"cache root not found: {root}")

    print(f"=== Cache layout migration ===\n  root: {root}")
    if args.revert:
        print("  mode: revert (new names → old names)\n")
    else:
        print(f"  mode: {'dry run' if args.dry_run else 'apply'}\n")

    pairs, notes = plan_renames(root, revert=args.revert)

    for note in notes:
        print(f"  {note}")
    if args.dry_run:
        for src, dst in pairs:
            print(f"  would move  {src.name}/ → {dst.name}/")

    if not pairs and not args.revert:
        print("\n  Nothing to rename — the cache is already migrated.")

    if args.dry_run:
        print()
        retire_representations(root, dry_run=True)
        print()
        backfill_recipes(root, dry_run=True)
        print("\nDry run only. Re-run with --apply to make these changes.")
        return

    if args.revert:
        apply_renames(pairs)
        print(
            "\nReverted. Note that recipe.json files written into "
            f"{DATASETS_DIR}/ are left in place — they are additive and harmless."
        )
        return

    apply_renames(pairs)
    print()
    retire_representations(root, dry_run=False)
    print()
    backfill_recipes(root, dry_run=False)
    print("\nDone. Run `python scripts/check_analysis.py` to confirm no path was missed.")


if __name__ == "__main__":
    main()
