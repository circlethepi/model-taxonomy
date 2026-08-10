"""Quarantine the collections written under the old, selector-blind key.

``CollectionCache`` used to name a directory after ``(sorted model IDs,
taxonomy, metric)``.  That triple does not determine a distance matrix — the
draw, embedder, view, normalization, pooling and replicate reduction all do too
— so a collection built under one selector was returned unchanged for another,
silently.  That is ``docs/notes/TODO.md`` item 14.

The key is now ``{taxonomy}/{collection_key}/{metric}_{surrogate_key}``, built
from each model's resolved artifact path and surrogate hash.  Nothing written
under the old key is readable under the new one.

**Why this quarantines rather than migrates.**  A faithful rehash needs to know
what each entry was built from, and the old entries do not record it:

* the one ``functional`` entry has ``"slice": {}`` — no provenance at all, and it
  is known to differ by 1.13e-03 from a rebuild under today's ``layer`` default;
* the three ``behavioral`` entries record pre-item-13 vocabulary
  (``behavioral_config_hash``, and ``representation``/``normalize`` in the
  item-12 sense) that no longer maps onto today's selector keys.

Translating those would be guessing, and a guess baked into a content hash is
worse than no entry: it would claim a provenance the data does not support.  So
they are moved aside, kept, and rebuilt on next read.  The whole directory is
52 KB and rebuilding needs no GPU, so the "invalidates every stored collection"
cost that deferred this item is not a real cost.

Layout::

    before:  06_collections/{old_hash}/{collection_info.json,distance_matrix.safetensors,coordinates/}
    after:   06_collections/_legacy/{old_hash}/...          ← untouched, still readable
             06_collections/{taxonomy}/{collection_key}/... ← written fresh on next read

Safety model
------------
* Nothing is deleted.  ``--apply`` moves; ``--revert`` moves back exactly.
* ``_legacy`` is skipped by ``CollectionCache.list_collections``, which ignores
  any top-level directory starting with ``_``, so quarantined entries cannot be
  mistaken for live ones.
* ``--compare`` rebuilds and reports **max |Δ|** without writing or deleting
  anything, so the numbers can be reviewed before ``--prune``.
* ``--prune`` deletes a quarantined entry only when its rebuild matched.

Note that ``--compare`` reports a **tolerance**, not byte identity.  A rebuild
recomputes floating-point distances rather than moving files: the TODO records
the functional collection reproducing to 4.09e-10 under its original
normalization, which is agreement, not equality.  The byte-identity checks the
other migration scripts use are right for moves and wrong here.

Usage::

    python scripts/migrate_collection_key.py                    # dry run
    python scripts/migrate_collection_key.py --apply
    python scripts/check_analysis.py
    python scripts/migrate_collection_key.py --compare
    python scripts/migrate_collection_key.py --prune            # only what matched
    python scripts/migrate_collection_key.py --revert           # undo the move
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_ROOT = REPO / "results/shared_cache"
LEGACY = "_legacy"

#: Old entries whose rebuild is expected to differ, and why.  Printed alongside
#: the comparison so a mismatch is read as the measurement it is rather than as a
#: failure of the migration.
EXPECTED_MISMATCH = {
    "functional": (
        "slice is empty, so its selector is unrecoverable; TODO.md records it "
        "was built under normalize='global' and differs by ~1.13e-03 under "
        "today's 'layer' default"
    ),
    "behavioral": (
        "slice uses pre-item-13 vocabulary (behavioral_config_hash), which does "
        "not map onto today's selector keys"
    ),
}


def _old_entries(root: Path) -> list[Path]:
    """Top-level directories written under the old flat key.

    Recognised by shape, not by name: a 16-hex directory holding a
    ``distance_matrix.safetensors``.  The new layout never puts one at this
    depth, so there is no ambiguity.
    """
    base = root / "06_collections"
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "distance_matrix.safetensors").exists():
            out.append(d)
    return out


def _quarantined(root: Path) -> list[Path]:
    base = root / "06_collections" / LEGACY
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir())


def plan(root: Path) -> list[tuple[Path, Path]]:
    base = root / "06_collections"
    return [(d, base / LEGACY / d.name) for d in _old_entries(root)]


def apply_moves(moves: list[tuple[Path, Path]], dry_run: bool) -> int:
    n = 0
    for old, new in moves:
        if new.exists():
            raise SystemExit(
                f"refusing to overwrite {new}: something is already quarantined "
                f"under that name. Inspect both and decide, rather than merging."
            )
        if not dry_run:
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(new))
        n += 1
    # The old index describes only old entries, so it moves with them; a fresh
    # one is written by the first save under the new key.
    idx = root_index(moves)
    if idx is not None:
        old_idx, new_idx = idx
        if old_idx.exists() and not dry_run:
            new_idx.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_idx), str(new_idx))
    return n


def root_index(moves: list[tuple[Path, Path]]) -> tuple[Path, Path] | None:
    if not moves:
        return None
    base = moves[0][0].parent
    return base / "index.json", base / LEGACY / "index.json"


def revert(root: Path, dry_run: bool) -> int:
    base = root / "06_collections"
    n = 0
    for d in _quarantined(root):
        target = base / d.name
        if target.exists():
            raise SystemExit(
                f"refusing to revert {d} onto the existing {target}. A rebuild "
                "has already written there; remove it first if that is intended."
            )
        if not dry_run:
            shutil.move(str(d), str(target))
        n += 1
    legacy_idx = base / LEGACY / "index.json"
    if legacy_idx.exists() and not dry_run:
        shutil.move(str(legacy_idx), str(base / "index.json"))
    return n


def describe(root: Path) -> None:
    """Print what each quarantined entry was, so the plan can be read."""
    for d in _quarantined(root) or _old_entries(root):
        info = json.loads((d / "collection_info.json").read_text())
        tax = info.get("taxonomy")
        note = EXPECTED_MISMATCH.get(tax, "")
        print(f"    {d.name}  {tax}/{info.get('metric')}")
        print(f"        label: {info.get('label')!r}")
        print(f"        slice: {info.get('slice')}")
        if note:
            print(f"        expected to differ on rebuild: {note}")


def compare(root: Path) -> int:
    """Rebuild each quarantined entry and report max |Δ| against the stored one.

    Reports; never writes to the quarantine and never deletes.  Behavioral
    entries are skipped while their inputs are still being written by the
    replicate re-extraction — rebuilding against a moving ``05_generated``
    measures nothing.
    """
    import numpy as np
    from safetensors.numpy import load_file

    matched = 0
    for d in _quarantined(root):
        info = json.loads((d / "collection_info.json").read_text())
        tax = info.get("taxonomy")
        stored = load_file(str(d / "distance_matrix.safetensors"))["matrix"]
        print(f"    {d.name}  {tax}/{info.get('metric')}  stored {stored.shape}")
        if tax != "functional":
            print(
                f"        SKIP — {tax} rebuild deferred: its inputs are still "
                "being written by the replicate re-extraction (TODO.md item 16)"
            )
            continue
        rebuilt = _rebuild_functional(root, info)
        if rebuilt is None:
            print("        SKIP — could not resolve the models it was built from")
            continue
        for norm, matrix in rebuilt.items():
            delta = float(np.abs(matrix - stored).max())
            verdict = "match" if delta < 1e-6 else "DIFFERS"
            print(f"        normalize={norm:<7} max|Δ| = {delta:.3e}  {verdict}")
            if delta < 1e-6:
                matched += 1
    return matched


def _rebuild_functional(root: Path, info: dict) -> dict | None:
    """Rebuild one functional collection at both normalizations.

    Both, because the entry records no selector: ``global`` is what TODO.md says
    it was built under and ``layer`` is today's default, and reporting the pair
    is what distinguishes "the stored bytes are intact" from "the default moved
    underneath it".
    """
    from src.analysis import scan_cache
    from src.analysis.comparison import _compute_distance_matrix

    wanted = [e["model_id"] for e in info.get("model_entries", [])]
    index = scan_cache(root).with_available("functional_repr")
    # The stored model_ids are recipe IDs (id_scheme="recipe_id", the default),
    # so select on that rather than on the adapter name.
    sub = index.filter(recipe_id=wanted)
    ids = [e.recipe_id or e.model_id for e in sub.entries]
    if sorted(ids) != sorted(wanted):
        return None
    out = {}
    for norm in ("global", "layer"):
        dm = _compute_distance_matrix(
            sub, "functional", info.get("metric", "cka"), ids,
            functional_selector={"normalize": norm},
        )
        out[norm] = dm.matrix
    return out


def prune(root: Path, dry_run: bool) -> int:
    """Delete quarantined entries whose rebuild matched.

    Deliberately conservative: it re-runs the comparison rather than trusting a
    previous run's output, and keeps anything that differs.
    """
    import numpy as np
    from safetensors.numpy import load_file

    n = 0
    for d in _quarantined(root):
        info = json.loads((d / "collection_info.json").read_text())
        if info.get("taxonomy") != "functional":
            continue
        stored = load_file(str(d / "distance_matrix.safetensors"))["matrix"]
        rebuilt = _rebuild_functional(root, info)
        if not rebuilt:
            continue
        if any(float(np.abs(m - stored).max()) < 1e-6 for m in rebuilt.values()):
            print(f"    removing {d.name} — rebuild reproduces it")
            if not dry_run:
                shutil.rmtree(d)
            n += 1
        else:
            print(f"    keeping  {d.name} — rebuild differs; see --compare")
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true",
                   help="report and change nothing (default)")
    g.add_argument("--apply", action="store_true", help="move old entries to _legacy/")
    g.add_argument("--revert", action="store_true", help="move them back")
    g.add_argument("--compare", action="store_true",
                   help="rebuild and report max|Δ|; writes nothing")
    g.add_argument("--prune", action="store_true",
                   help="delete quarantined entries whose rebuild matched")
    args = p.parse_args()

    root = args.root.resolve()
    print(f"cache root: {root}\n")

    if args.compare:
        print("Comparing quarantined entries against a rebuild")
        matched = compare(root)
        print(f"\n  {matched} rebuild(s) reproduced the stored matrix")
        return 0

    if args.prune:
        print("Pruning quarantined entries that a rebuild reproduces")
        n = prune(root, dry_run=False)
        print(f"\n  {n} entry(ies) removed")
        return 0

    if args.revert:
        n = revert(root, dry_run=False)
        print(f"  {n} entry(ies) moved back out of {LEGACY}/")
        return 0

    moves = plan(root)
    print(f"Old-key collections found: {len(moves)}")
    describe(root)
    if not moves:
        print("\nNothing to quarantine.")
        return 0

    n = apply_moves(moves, dry_run=not args.apply)
    if not args.apply:
        print(f"\nDry run only: {n} entry(ies) would move to {LEGACY}/.")
        print("Re-run with --apply to make these changes.")
        return 0

    print(f"\n  {n} entry(ies) moved to {LEGACY}/")
    print("\nDone. Next:")
    print("    python scripts/check_analysis.py")
    print("    python scripts/migrate_collection_key.py --compare")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
