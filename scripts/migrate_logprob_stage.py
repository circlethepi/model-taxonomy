"""One-shot migration: fix the case of the log-prob stage directory.

    05a_logprobs → 05A_logprobs

Log-probs are a second artifact of the *generation* pass, written beside the
generations (``src/taxonomy/behavioral.py``), so they belong at stage 05 rather
than at a stage of their own — which is why the directory was moved off
``07_logprobs`` already.  What that move did not settle is the case of the
suffix.  ``scripts/migrate_cache_layout.py`` defines the convention and the only
other letter-suffixed stage spells it uppercase (``03A_adapter_alignments``), so
``05A`` is the spelling that matches.  Two stages disagreeing on the case of a
suffix is the kind of detail that is free to fix now and permanent later.

This is **independent** of ``scripts/migrate_pairwise_layout.py``: the two touch
disjoint directories and disjoint call sites, so either order works and neither
needs the other to have run.  Kept separate so a failure in either is
diagnosable without unpicking the other.

Note for anyone running this elsewhere: the rename is case-only, so on a
case-insensitive filesystem it is a no-op that may appear to succeed while
changing nothing.  ``--apply`` verifies the resulting name rather than trusting
``os.rename``.

There is deliberately no compatibility shim in ``src/cache/logprob_cache.py``: a
missed call site must fail loudly rather than quietly read the old directory.

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

OLD = "05a_logprobs"
NEW = "05A_logprobs"


def _names(root: Path) -> list[str]:
    """Directory names as the filesystem actually spells them.

    Read by listing the parent rather than by ``Path.exists()``, which on a
    case-insensitive filesystem answers ``True`` for either spelling and so
    cannot tell a completed rename from an unstarted one.
    """
    return [p.name for p in root.iterdir() if p.is_dir()]


def plan(root: Path, revert: bool) -> tuple[Path, Path] | None:
    src_name, dst_name = (NEW, OLD) if revert else (OLD, NEW)
    names = _names(root)

    if src_name not in names:
        if dst_name in names:
            print(f"  skip   {src_name}/ — not present; {dst_name}/ already is")
        else:
            print(f"  skip   {src_name}/ — not present")
        return None
    if dst_name in names:
        raise SystemExit(
            f"refusing to migrate: both {root / src_name} and {root / dst_name} "
            "exist. Resolve by hand — merging two cache trees silently would "
            "lose data."
        )
    return root / src_name, root / dst_name


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
    mode.add_argument("--apply", action="store_true", help=f"rename {OLD}/ → {NEW}/")
    mode.add_argument("--revert", action="store_true", help=f"rename {NEW}/ → {OLD}/")
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
