"""One-shot migration: move the log-prob stage to ``05A_logprobs``.

    07_logprobs  → 05A_logprobs      (the original move, from a cold cache root)
    05a_logprobs → 05A_logprobs      (the case fix, for a root already moved once)

``07`` put the log-probs after the collections in every sorted listing, which
misdescribes where they come from.  They are not derived from collections: they
are a ride-along artifact of the same generation pass that writes
``05_generated``.  ``BehavioralTaxonomy`` with ``collect_logprobs=True`` fills
both trees in one pass, under the same draw key and the same variant token, so a
log-prob file and the generations it describes join by *filename* with no lookup.
``05`` puts the stage next to the generations it annotates, and ahead of the
collections built on top of the inference stages.

**On the case of the suffix.**  The letter suffix is a mild stretch of the
convention ``03A_adapter_alignments`` set — there it means "analysis of the
objects at that stage", and log-probs are a parallel artifact rather than an
analysis.  The adjacency is worth the stretch.  Given the stretch is being made,
the suffix is spelled the way the only other letter-suffixed stage spells it:
**uppercase**.  A first pass of this migration wrote ``05a``, so both source
spellings are accepted here and a root in either state converges on ``05A``.

There is deliberately no compatibility shim in ``LogProbCache``: ``_STAGE_DIR``
moves outright.  A cache root that was never migrated finds no entries and
re-extracts, which is loud and correct; a fallback read would silently split one
experiment's results across two trees.

The move is a single ``os.rename`` within one filesystem, so it is a metadata
operation — instant regardless of how large the tree is.

This is **independent** of ``scripts/migrate_pairwise_layout.py``: the two touch
disjoint directories and disjoint call sites, so either order works and neither
needs the other to have run.  Kept separate so a failure in either is
diagnosable without unpicking the other.

Only a shared-cache-shaped root is migrated.  Legacy per-experiment cache trees
under ``results/<experiment>/cache/`` keep the old names and are not readable by
current code anyway; none of them contains a log-prob stage.

Note for anyone running the case fix on another machine: a case-only rename is a
silent no-op on a case-insensitive filesystem, so ``--apply`` verifies the
resulting name in a directory listing rather than trusting ``os.rename``.

Usage:
    python scripts/migrate_logprob_stage.py --dry-run
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

NEW = "05A_logprobs"
#: Accepted source spellings, oldest first.  Both converge on NEW.
OLD = ("07_logprobs", "05a_logprobs")
#: What ``--revert`` restores: the stage as it was before any of this.
ORIGINAL = OLD[0]


def _names(root: Path) -> list[str]:
    """Directory names as the filesystem actually spells them.

    Read by listing the parent rather than by ``Path.exists()``, which on a
    case-insensitive filesystem answers ``True`` for either spelling of the
    suffix and so cannot tell a completed rename from an unstarted one.
    """
    return [p.name for p in root.iterdir() if p.is_dir()]


def plan(root: Path, revert: bool) -> tuple[Path, Path] | None:
    names = _names(root)

    if revert:
        if NEW not in names:
            print(f"  skip   {NEW}/ — not present")
            return None
        if ORIGINAL in names:
            raise SystemExit(
                f"refusing to revert: both {root / NEW} and {root / ORIGINAL} "
                "exist. Resolve by hand — merging two cache trees silently "
                "would lose data."
            )
        return root / NEW, root / ORIGINAL

    present = [name for name in OLD if name in names]
    if not present:
        if NEW in names:
            print(f"  skip   — already at {NEW}/")
        else:
            print(f"  skip   — no log-prob stage under any of {list(OLD)}")
        return None
    if len(present) > 1:
        raise SystemExit(
            f"refusing to migrate: {present} both exist under {root}. Resolve "
            "by hand — merging two cache trees silently would lose data."
        )
    if NEW in names:
        raise SystemExit(
            f"refusing to migrate: both {root / present[0]} and {root / NEW} "
            "exist. Resolve by hand — merging two cache trees silently would "
            "lose data."
        )
    return root / present[0], root / NEW


def apply_rename(root: Path, pair: tuple[Path, Path]) -> None:
    src, dst = pair
    os.rename(src, dst)
    if dst.name not in _names(root):
        raise SystemExit(
            f"rename reported success but {dst.name}/ is not in the listing. "
            "This is what a case-insensitive filesystem does with a case-only "
            "rename; the cache root must be on a case-sensitive filesystem."
        )
    print(f"  moved  {src.name}/ → {dst.name}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true",
                      help="print what would happen and change nothing")
    mode.add_argument("--apply", action="store_true",
                      help=f"rename 07_logprobs/ or 05a_logprobs/ → {NEW}/")
    mode.add_argument("--revert", action="store_true",
                      help=f"rename {NEW}/ → {ORIGINAL}/")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"the shared cache root (default: {DEFAULT_ROOT})")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"no such cache root: {root}")

    print(f"cache root: {root}")
    pair = plan(root, revert=args.revert)

    if args.dry_run:
        if pair is not None:
            print(f"  would move  {pair[0].name}/ → {pair[1].name}/")
        print("dry run — nothing changed")
        return

    if pair is not None:
        apply_rename(root, pair)
    print("done")


if __name__ == "__main__":
    main()
