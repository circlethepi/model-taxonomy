"""One-shot migration: make room for the pairwise-distance store.

    06_collections  → 07_collections
    —               → 06_pairwise   (created empty)

Collections are *assembled from* pairs, so they sit downstream of them: the
number ``06`` belongs to the store that everything else is derived from, and
``07`` to the whole matrices and MDS fits built out of it.  The convention this
follows is stated in ``scripts/migrate_cache_layout.py``: directories sharing a
number sit at the same pipeline stage, and a letter suffix means "analysis of
the objects at that stage".

Renaming is cheap in a way worth stating, because the size of the directory
suggests otherwise: ``os.rename`` within one filesystem is a metadata operation,
so a tree holding 88 distance matrices moves as fast as an empty one.

There is deliberately **no compatibility shim** in the cache classes, following
the stance ``migrate_cache_layout.py`` already takes: a call site that was
missed must fail loudly rather than quietly keep reading the old directory.

This is independent of ``scripts/migrate_logprob_stage.py``.  The two touch
disjoint directories and disjoint call sites, so either order works and neither
needs the other to have run.  The intermediate state *looks* alarming and is
not: after this script alone, ``07_collections`` and ``05a_logprobs`` coexist
and nothing resolves a stage by number.

Usage:
    python scripts/migrate_pairwise_layout.py --dry-run
    python scripts/migrate_pairwise_layout.py --apply
    python scripts/migrate_pairwise_layout.py --revert
    python scripts/migrate_pairwise_layout.py --root some/other/cache --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent

DEFAULT_ROOT = REPO / "results" / "shared_cache"

#: old directory name → new directory name.
RENAMES = {
    "06_collections": "07_collections",
}

#: Created empty by ``--apply``; removed by ``--revert`` only if still empty.
CREATED = "06_pairwise"


def plan_renames(root: Path, revert: bool = False) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return the (src, dst) pairs to rename, plus human-readable notes on the rest.

    Refuses to plan a rename whose destination already exists — that means a
    partially applied migration, and merging two cache trees silently is worse
    than stopping.
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
    for src, dst in pairs:
        os.rename(src, dst)
        print(f"  moved  {src.name}/ → {dst.name}/")


def create_store(root: Path, dry_run: bool) -> None:
    """Create ``06_pairwise/`` so the new stage is visible in a listing.

    ``PairwiseCache`` creates it on first write anyway; making it here means the
    migration's result is what the layout diagram says it is, rather than a
    directory that appears only after something happens to run.
    """
    path = root / CREATED
    if path.exists():
        print(f"  skip   {CREATED}/ — already present")
        return
    if dry_run:
        print(f"  would create  {CREATED}/")
    else:
        path.mkdir(parents=True)
        print(f"  created  {CREATED}/")


def remove_store(root: Path, dry_run: bool) -> None:
    """Remove ``06_pairwise/`` on revert, but only while it is empty.

    A populated store is real cached work, and no part of reverting a *rename*
    justifies deleting it.  Refusing here is the same stance ``retire_representations``
    takes in ``migrate_cache_layout.py``.
    """
    path = root / CREATED
    if not path.exists():
        print(f"  skip   {CREATED}/ — not present")
        return

    entries = list(path.rglob("*"))
    if entries:
        print(
            f"  KEEP   {CREATED}/ — {len(entries)} entr(ies) present, not empty. "
            "Those are computed distances; reverting a rename does not justify "
            "deleting them. Left in place."
        )
        return

    if dry_run:
        print(f"  would remove  {CREATED}/ (empty)")
    else:
        path.rmdir()
        print(f"  removed  {CREATED}/ (empty)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="print what would happen and change nothing")
    mode.add_argument("--apply", action="store_true",
                      help="perform the rename and create 06_pairwise/")
    mode.add_argument("--revert", action="store_true",
                      help="undo the rename; removes 06_pairwise/ only if empty")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"the shared cache root (default: {DEFAULT_ROOT})")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"no such cache root: {root}")

    print(f"cache root: {root}")
    pairs, notes = plan_renames(root, revert=args.revert)
    for note in notes:
        print(f"  {note}")

    if args.dry_run:
        for src, dst in pairs:
            print(f"  would move  {src.name}/ → {dst.name}/")
        if args.revert:
            remove_store(root, dry_run=True)
        else:
            create_store(root, dry_run=True)
        print("dry run — nothing changed")
        return

    apply_renames(pairs)
    if args.revert:
        remove_store(root, dry_run=False)
    else:
        create_store(root, dry_run=False)
    print("done")


if __name__ == "__main__":
    main()
