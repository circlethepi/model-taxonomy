"""One-shot migration: renumber the log-prob stage from ``07`` to ``05a``.

    07_logprobs → 05a_logprobs

``07`` put the log-probs after ``06_collections`` in every sorted listing, which
misdescribes where they come from.  They are not derived from collections: they
are a ride-along artifact of the same generation pass that writes
``05_generated``.  ``BehavioralTaxonomy`` with ``collect_logprobs=True`` fills
both trees in one pass, under the same draw key and the same variant token, so a
log-prob file and the generations it describes join by *filename* with no lookup.
``05a`` puts the stage next to the generations it annotates, and ahead of the
collections that are built on top of the inference stages.

The letter suffix is a mild stretch of the convention ``03A_adapter_alignments``
set — there it means "analysis of the objects at that stage", and log-probs are a
parallel artifact rather than an analysis.  The adjacency is worth the stretch.

There is deliberately no compatibility shim in ``LogProbCache``: ``_STAGE_DIR``
moves outright.  A cache root that was never migrated finds no entries and
re-extracts, which is loud and correct; a fallback read would silently split one
experiment's results across two trees.

The move is a single ``os.rename`` within one filesystem, so it is a metadata
operation — instant regardless of how large the tree is.

Only a shared-cache-shaped root is migrated.  Legacy per-experiment cache trees
under ``results/<experiment>/cache/`` keep the old names and are not readable by
current code anyway; none of them contains a log-prob stage.

Usage:
    python scripts/migrate_logprob_stage.py             # dry run (the default)
    python scripts/migrate_logprob_stage.py --apply
    python scripts/migrate_logprob_stage.py --revert
    python scripts/migrate_logprob_stage.py --root some/other/cache --dry-run
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

OLD_NAME = "07_logprobs"
NEW_NAME = "05a_logprobs"


def plan_rename(root: Path, revert: bool = False) -> tuple[Path, Path] | None:
    """Return the ``(src, dst)`` pair to rename, or ``None`` if there is nothing to do.

    Refuses when both names exist — that is a partially applied migration, and
    merging two cache trees silently is worse than stopping.  Entries under the
    two names are addressed identically below the stage directory, so a merge
    would look plausible and quietly leave half the draws unreachable.
    """
    old, new = (NEW_NAME, OLD_NAME) if revert else (OLD_NAME, NEW_NAME)
    src, dst = root / old, root / new

    if src.exists() and dst.exists():
        raise SystemExit(
            f"refusing to migrate: both {src} and {dst} exist. "
            "Resolve by hand — merging two cache trees silently would lose data."
        )
    if not src.exists():
        return None
    return src, dst


def apply_rename(pair: tuple[Path, Path]) -> None:
    """``os.rename`` the stage directory.  Contents are untouched."""
    src, dst = pair
    os.rename(src, dst)
    print(f"  moved  {src.name}/ → {dst.name}/")


def describe(path: Path) -> str:
    """File count and top-level base-model names, for confirming the move landed."""
    if not path.exists():
        return "not present"
    files = sum(1 for p in path.rglob("*") if p.is_file())
    bases = sorted(p.name for p in path.iterdir() if p.is_dir())
    shown = ", ".join(bases) if bases else "none"
    return f"{files} file(s) under base model(s): {shown}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Rename the log-prob cache stage {OLD_NAME} → {NEW_NAME}."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Cache root to migrate (default: {DEFAULT_ROOT}).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print the plan, change nothing. The default.")
    mode.add_argument("--apply", action="store_true",
                      help=f"Rename {OLD_NAME}/ to {NEW_NAME}/.")
    mode.add_argument("--revert", action="store_true",
                      help=f"Rename {NEW_NAME}/ back to {OLD_NAME}/.")
    args = parser.parse_args()

    # Dry run is the default: this moves every log-prob result there is, and the
    # harmless mode is the one you get by forgetting a flag.
    dry_run = not (args.apply or args.revert)

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"cache root not found: {root}")

    print(f"=== Log-prob stage migration ===\n  root: {root}")
    if args.revert:
        print(f"  mode: revert ({NEW_NAME} → {OLD_NAME})\n")
    else:
        print(f"  mode: {'dry run' if dry_run else 'apply'}\n")

    pair = plan_rename(root, revert=args.revert)

    if pair is None:
        src_name = NEW_NAME if args.revert else OLD_NAME
        dst_name = OLD_NAME if args.revert else NEW_NAME
        print(f"  skip   {src_name}/ — not present")
        print(f"  {dst_name}/: {describe(root / dst_name)}")
        print("\n  Nothing to rename — already migrated, or never populated.")
        return

    src, dst = pair
    print(f"  {src.name}/: {describe(src)}")

    if dry_run:
        print(f"\n  would move  {src.name}/ → {dst.name}/")
        print("\nDry run only. Re-run with --apply to make this change.")
        return

    apply_rename(pair)
    print(f"  {dst.name}/: {describe(dst)}")
    print(
        "\nDone. Run `python scripts/check_analysis.py` to confirm no path was "
        "missed; there is no compatibility shim, so a missed call site fails loudly."
    )


if __name__ == "__main__":
    main()
