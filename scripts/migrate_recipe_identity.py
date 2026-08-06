#!/usr/bin/env python
"""Re-key the shared cache onto content-addressed recipe hashes, and convert draws to indices.

Two changes land together because neither is reachable without the other.

**Recipe identity.**  ``recipe_hash`` used to be a SHA-256 of ``{name, datasets}``, and
the name carried ``_n{n}_s{seed:02d}``.  Every (mixture, n, seed) was therefore its own
recipe: 564 hashes describing 5 mixtures, with 38 pairs of *byte-identical* row files
stored twice because their names differed.  The hash is now over ``{recipe_type,
datasets}`` alone, so one mixture is one recipe and n and seed move into the draw
filename, ``n{n}_s{seed}.json``.

**Draw storage.**  A draw used to be the rows themselves — 722 bytes per row of text
duplicated out of a dataset already sitting in the HuggingFace cache.  It is now the
source indices those rows came from: 2.07 GiB becomes ~40 MiB.

Indices are recovered by **matching stored rows against the source**, not by re-running
the sampler.  That distinction is load-bearing.  Several large draws predate the
proportional class scale-down in ``ClassMixedDataset._load_entry`` and re-running
produces materially different rows — the n=265000 draw of ``yahoo_075t0_025t1`` has
206,250 rows at a 68/32 topic split where today's sampler yields 186,666 at 75/25.
Content-matching preserves what was actually drawn; resampling would silently replace
it.

Safety model, which differs deliberately from ``migrate_cache_layout.py``:

- **Additive.**  New hash directories are written alongside the old ones.  Nothing is
  deleted until ``--prune``, which is a separate invocation to be run only after
  ``check_analysis.py`` is green.
- **No ``--revert``.**  Rows-to-indices is not a reversible rename.  The old directories
  surviving until ``--prune`` *is* the rollback.
- **Self-verifying.**  Every converted draw is rehydrated from its new indices and
  required to match the original bytes before the manifest is written.  A draw that
  cannot be reproduced keeps its rows verbatim rather than being converted.

Usage::

    python scripts/migrate_recipe_identity.py --dry-run
    python scripts/migrate_recipe_identity.py --apply
    python scripts/check_analysis.py                  # must be green first
    python scripts/migrate_recipe_identity.py --prune
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_ROOT = REPO / "results" / "shared_cache"
DATASETS_DIR = "01_datasets"
EMBEDDINGS_DIR = "02_dataset_embeddings"
ADAPTERS_DIR = "03_adapters"

# Legacy draw filename: {n_samples}_{seed:010d}.json
_LEGACY_DRAW_RE = re.compile(r"^(?P<n>\d+)_(?P<seed>\d+)$")
# Expanded config-block name, which is what the old recipe name was.
_NAME_RE = re.compile(r"^(?P<mixture>.+?)(?:_n(?P<n>\d+))?_s(?P<seed>\d+)$")
# Adapter directory: {expanded_block_name}_r{rank}[_i{init}].  Mirrors
# src.analysis.discovery._ADAPTER_DIR_RE.
_ADAPTER_DIR_RE = re.compile(r"^(?P<name>.+?)_r(?P<rank>\d+)(?:_i(?P<init>\d+))?$")


# ── Phase 1: the hash map ─────────────────────────────────────────────────────


def _load_recipe(path: Path):
    from src.datasets.class_recipe import ClassAwareDatasetRecipe
    from src.datasets.recipe import DatasetRecipe

    payload = json.loads(path.read_text())
    cls = (
        ClassAwareDatasetRecipe
        if payload.get("recipe_type") == "class_aware"
        else DatasetRecipe
    )
    return cls.load(path)


def build_hash_map(root: Path) -> tuple[dict[str, dict], list[str]]:
    """``old_hash -> {new_hash, name, recipe}`` over every recipe findable on disk.

    Three sources, widest last: the dataset cache, the embedding cache, and the
    per-experiment recipe files under ``results/*/datasets/``.  The third is what
    resolves the five adapter hashes that no shared-cache directory covers — they were
    trained under the pre-rename naming (``yahoo_25t0_75t1``) and their rows were never
    sampled into the shared cache, but their recipes were persisted by build_datasets.py.

    The new hash is always *recomputed*, never read from the file, so a stale stored
    ``recipe_hash`` cannot introduce a wrong mapping.
    """
    mapping: dict[str, dict] = {}
    notes: list[str] = []

    sources = [
        (root / DATASETS_DIR, "*/recipe.json"),
        (root / EMBEDDINGS_DIR, "*/recipe.json"),
        (REPO / "results", "*/datasets/*.recipe.json"),
    ]

    for base, pattern in sources:
        if not base.exists():
            continue
        found = 0
        for path in sorted(base.glob(pattern)):
            try:
                payload = json.loads(path.read_text())
                recipe = _load_recipe(path)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            # The old hash is the one the cache is keyed by.  For files under a hash
            # directory that is the directory name; otherwise it is the stored field.
            old_hash = (
                path.parent.name
                if base.name in (DATASETS_DIR, EMBEDDINGS_DIR)
                else payload.get("recipe_hash")
            )
            if not old_hash:
                continue
            entry = mapping.setdefault(
                old_hash,
                {"new_hash": recipe.recipe_hash(), "names": set(), "recipe": recipe},
            )
            if payload.get("name"):
                entry["names"].add(payload["name"])
            found += 1
        notes.append(f"  {base.name}/{pattern}: {found} recipe file(s)")

    return mapping, notes


def check_adapter_coverage(root: Path, mapping: dict[str, dict]) -> list[str]:
    """Every adapter's recipe_hash must be in the map before anything is written."""
    problems = []
    adapters = root / ADAPTERS_DIR
    if not adapters.exists():
        return problems
    for meta_path in sorted(adapters.glob("*/*/experiment_meta.json")):
        meta = json.loads(meta_path.read_text())
        old_hash = meta.get("recipe_hash")
        if old_hash and old_hash not in mapping:
            problems.append(
                f"  {meta_path.parent.name}: recipe_hash {old_hash} resolves to no recipe"
            )
    return problems


# ── Phase 2: draws ────────────────────────────────────────────────────────────


def convert_draws(root: Path, mapping: dict[str, dict], dry_run: bool) -> dict:
    """Rewrite every draw under its new hash, as indices.

    Each draw is converted, rehydrated from the result, and compared byte-for-byte with
    the original before anything is written.  A draw that will not reproduce keeps its
    rows: correctness of the stored data outranks the space saving.
    """
    from src.cache.sampled_dataset_cache import SampledDatasetCache, rows_checksum
    from src.datasets import source_registry

    datasets_root = root / DATASETS_DIR
    cache = SampledDatasetCache(root)
    stats = {"converted": 0, "kept_rows": 0, "duplicate": 0,
             "old_bytes": 0, "new_bytes": 0, "unmapped": 0}
    t0 = time.time()

    draws = []
    for directory in sorted(datasets_root.iterdir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            m = _LEGACY_DRAW_RE.match(path.stem)
            if path.suffix == ".json" and m:
                draws.append((directory.name, path, int(m.group("n")), int(m.group("seed"))))

    print(f"  {len(draws)} legacy draw file(s) to convert")

    for i, (old_hash, path, n_samples, seed) in enumerate(draws, 1):
        entry = mapping.get(old_hash)
        if entry is None:
            stats["unmapped"] += 1
            print(f"  UNMAPPED {old_hash}/{path.name} — no recipe, left alone")
            continue

        new_hash = entry["new_hash"]
        target = cache._path(new_hash, n_samples, seed)
        stats["old_bytes"] += path.stat().st_size

        if target.exists():
            # The 38 byte-identical pairs collapse here.  Refuse to merge differing
            # content silently — that would destroy one of the two draws.
            existing = json.loads(target.read_text())
            if existing.get("rows_sha256") != rows_checksum(json.loads(path.read_text())):
                raise SystemExit(
                    f"Refusing to merge: {old_hash}/{path.name} and an already-written "
                    f"{new_hash}/{target.name} hold different rows under one hash."
                )
            stats["duplicate"] += 1
            continue

        stored_text = path.read_text()
        stored = json.loads(stored_text)
        recipe = entry["recipe"]

        sources = []
        for recipe_entry in recipe.datasets:
            ds = source_registry.get(
                recipe_entry.dataset_id, recipe_entry.subset, recipe_entry.split
            )
            sources.append(source_registry.describe(
                ds, recipe_entry.dataset_id, recipe_entry.subset, recipe_entry.split
            ))

        indices = source_registry.locate_rows(stored, sources)
        rebuilt_ok = False
        if indices is not None:
            wanted: dict[int, list[int]] = {}
            for s, r in indices:
                wanted.setdefault(s, []).append(r)
            picked: dict[int, dict[int, dict]] = {}
            for s, row_indices in wanted.items():
                desc = sources[s]
                ds = source_registry.get(desc["dataset_id"], desc["subset"], desc["split"])
                unique = sorted(set(row_indices))
                picked[s] = {j: dict(row) for j, row in zip(unique, ds.select(unique))}
            rebuilt_ok = json.dumps([picked[s][r] for s, r in indices]) == stored_text

        if not rebuilt_ok:
            stats["kept_rows"] += 1
            print(f"  KEEP-ROWS {old_hash}/{path.name} — cannot be reproduced from "
                  f"source; storing rows verbatim")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_json(target, stored)
            stats["new_bytes"] += len(stored_text)
            continue

        stats["converted"] += 1
        stats["new_bytes"] += len(json.dumps({
            "schema_version": "2", "n_samples": n_samples, "seed": seed,
            "sources": sources, "indices": [list(p) for p in indices],
            "rows_sha256": rows_checksum(stored),
        }))
        if not dry_run:
            # Write through the cache rather than hand-rolling the manifest, so the
            # migration and normal operation cannot drift in format.  It matters more
            # than it looks: this file's _write_json indents, and indenting a
            # 265,000-element index list costs 2.6x the bytes.
            cache.put(new_hash, n_samples, seed,
                      rows=stored, indices=indices, sources=sources)
            cache.put_recipe(new_hash, recipe.to_dict())
            for name in sorted(entry["names"]):
                cache.add_name(new_hash, name)

        if i % 50 == 0:
            print(f"  ... {i}/{len(draws)} ({time.time() - t0:.0f}s)", flush=True)

    return stats


# ── Phase 3: dataset embeddings ───────────────────────────────────────────────


def _legacy_embedder_hash(
    embedder_config: dict, representation: str, n_samples: int, seed: int | None
) -> str:
    """``DatasetEmbeddingCache.embedder_hash`` as it stood when this script ran.

    Frozen deliberately.  The live function now keys on ``embedder_config``
    alone, so calling it here would silently rewrite what this migration is a
    record of.  A migration script describes a transition that already happened;
    it must keep computing the digests it actually wrote.
    """
    payload = json.dumps(
        {
            "embedder_config": embedder_config,
            "representation": representation,
            "n_samples": n_samples,
            "seed": seed,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def migrate_embeddings(root: Path, mapping: dict[str, dict], dry_run: bool) -> dict:
    """Re-key ``02_dataset_embeddings`` on both levels.

    The recipe hash changes, and so does the embedder hash — it now includes the seed,
    without which every seed of a mixture would collide onto one entry.  The seed is
    read from the *old* recipe name, which still carries ``_s{seed}`` at this point;
    after the migration nothing else records it.

    The embedder-hash signature described above is **this migration's**, not the
    live one.  Item 15 later moved the draw into the path and the representation
    into a surrogate spec, leaving ``DatasetEmbeddingCache.embedder_hash`` keyed
    on ``embedder_config`` alone.  This script is a completed one-shot whose
    record of what it did must not change meaning when the live signature does,
    so it computes the old four-field digest itself — see
    :func:`_legacy_embedder_hash` below.  Do not re-point it at the live one.
    """
    embeddings_root = root / EMBEDDINGS_DIR
    stats = {"moved": 0, "skipped": 0, "collisions": 0, "duplicate": 0}
    if not embeddings_root.exists():
        return stats

    planned: dict[tuple[str, str], Path] = {}

    for directory in sorted(embeddings_root.iterdir()):
        if not directory.is_dir():
            continue
        entry = mapping.get(directory.name)
        if entry is None:
            stats["skipped"] += 1
            continue
        new_hash = entry["new_hash"]

        seed = None
        for name in entry["names"]:
            m = _NAME_RE.match(name)
            if m:
                seed = int(m.group("seed"))
                break

        for emb_dir in sorted(directory.iterdir()):
            config_path = emb_dir / "config.json"
            if emb_dir.is_dir() and config_path.exists():
                config = json.loads(config_path.read_text())
                new_emb = _legacy_embedder_hash(
                    config["embedder_config"], config["representation"],
                    config["n_samples"], seed,
                )
                key = (new_hash, new_emb)
                if key in planned:
                    # Two old hashes reaching one key is expected: the pairs whose only
                    # difference was a recipe *name* (yahoo_x_n1000_s09 vs yahoo_x_s09)
                    # embedded the same draw twice.  Collapse them when the vectors
                    # agree, and refuse only when they genuinely differ.
                    if _embeddings_agree(planned[key], emb_dir):
                        stats["duplicate"] += 1
                        continue
                    stats["collisions"] += 1
                    raise SystemExit(
                        f"Refusing to merge: {directory.name}/{emb_dir.name} and "
                        f"{planned[key]} both map to {new_hash}/{new_emb} but hold "
                        f"different embeddings. One would overwrite the other."
                    )
                planned[key] = emb_dir

                if not dry_run:
                    target = embeddings_root / new_hash / new_emb
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copytree(emb_dir, target)
                    config["schema_version"] = "2"
                    config["recipe_hash"] = new_hash
                    config["seed"] = seed
                    _write_json(target / "config.json", config)
                    recipe_path = embeddings_root / new_hash / "recipe.json"
                    if not recipe_path.exists():
                        _write_json(recipe_path, entry["recipe"].to_dict())
                stats["moved"] += 1

    return stats


# ── Phase 4: adapters ─────────────────────────────────────────────────────────


def migrate_adapters(root: Path, mapping: dict[str, dict], dry_run: bool) -> dict:
    """Point each adapter's ``experiment_meta.json`` at the new hash, and record its seed.

    ``training.seed`` is added because after this migration the recipe no longer encodes
    it and ``CacheIndex`` has no other fallback.  It is parsed from ``dataset_name``,
    which is the expanded config-block name and still carries ``_s{seed}``.
    """
    adapters = root / ADAPTERS_DIR
    stats = {"updated": 0, "unmapped": 0, "seed_added": 0}
    if not adapters.exists():
        return stats

    for meta_path in sorted(adapters.glob("*/*/experiment_meta.json")):
        meta = json.loads(meta_path.read_text())
        old_hash = meta.get("recipe_hash")
        entry = mapping.get(old_hash) if old_hash else None
        if entry is None:
            stats["unmapped"] += 1
            print(f"  UNMAPPED adapter {meta_path.parent.name} (hash {old_hash})")
            continue

        meta["recipe_hash"] = entry["new_hash"]
        training = meta.setdefault("training", {})
        if "seed" not in training:
            # dataset_name is the expanded block name for anything recent.  The five
            # oldest adapters predate that convention and carry a bare mixture name, but
            # their directory is {block_name}_r{rank}_i{init} and still has the seed.
            m = _NAME_RE.match(meta.get("dataset_name") or "")
            if not m:
                dir_m = _ADAPTER_DIR_RE.match(meta_path.parent.name)
                if dir_m:
                    m = _NAME_RE.match(dir_m.group("name"))
            if m:
                training["seed"] = int(m.group("seed"))
                stats["seed_added"] += 1
        stats["updated"] += 1
        if not dry_run:
            _write_json(meta_path, meta)

    return stats


# ── Prune ─────────────────────────────────────────────────────────────────────


def prune(root: Path, mapping: dict[str, dict], dry_run: bool) -> None:
    """Delete the pre-migration directories, once the new ones are proven good.

    Separate from ``--apply`` on purpose: until this runs, the old tree is the rollback.
    """
    live = {e["new_hash"] for e in mapping.values()}
    removed = kept = 0
    for base in (root / DATASETS_DIR, root / EMBEDDINGS_DIR):
        if not base.exists():
            continue
        for directory in sorted(base.iterdir()):
            if not directory.is_dir():
                continue
            if directory.name in live:
                kept += 1
                continue
            if directory.name not in mapping:
                # Not something this migration created or replaced; leave it be.
                kept += 1
                continue
            removed += 1
            print(f"  {'would remove' if dry_run else 'removed'} {base.name}/{directory.name}")
            if not dry_run:
                shutil.rmtree(directory)
    print(f"\n  {removed} old directory(ies), {kept} kept")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _embeddings_agree(a_dir: Path, b_dir: Path) -> bool:
    """Do two embedding entries hold the same vectors?

    ``_meta_json`` is excluded on purpose: it carries the recipe *name*, which is
    precisely what differs between the duplicate pairs this migration collapses, and
    which is not part of the embedding.  Compared with a float tolerance because the
    two were computed in separate runs — measured agreement on the real pairs is
    ~1e-8 absolute, cosine 1.0, i.e. the same vector recomputed.
    """
    import numpy as np
    from safetensors.numpy import load_file

    try:
        a, b = load_file(a_dir / "embeddings.safetensors"), load_file(b_dir / "embeddings.safetensors")
    except OSError:
        return False
    if set(a) != set(b):
        return False
    for key in a:
        if key == "_meta_json":
            continue
        if a[key].shape != b[key].shape or not np.allclose(a[key], b[key], rtol=1e-5, atol=1e-6):
            return False
    return True


def _write_json(path: Path, payload) -> None:
    """Atomic write, consistent with the cache classes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) if isinstance(payload, dict)
                   else json.dumps(payload))
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    mode.add_argument("--apply", action="store_true", help="write the new layout")
    mode.add_argument("--prune", action="store_true",
                      help="delete the pre-migration directories (run after verifying)")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"No cache at {args.root}")
        return 1

    dry_run = args.dry_run
    print(f"Cache root: {args.root}")
    print(f"Mode: {'dry-run' if dry_run else 'prune' if args.prune else 'apply'}\n")

    print("Phase 1 — hash map")
    mapping, notes = build_hash_map(args.root)
    for note in notes:
        print(note)
    new_hashes = {e["new_hash"] for e in mapping.values()}
    print(f"  {len(mapping)} old hash(es) -> {len(new_hashes)} new hash(es)")

    problems = check_adapter_coverage(args.root, mapping)
    if problems:
        print("\nRefusing to migrate — adapters reference recipes that cannot be resolved:")
        for problem in problems:
            print(problem)
        return 1
    print("  all adapter recipe_hash values resolve")

    if args.prune:
        print("\nPrune")
        prune(args.root, mapping, dry_run=False)
        return 0

    print("\nPhase 2 — draws")
    draw_stats = convert_draws(args.root, mapping, dry_run)
    print(f"  converted {draw_stats['converted']}, duplicates collapsed "
          f"{draw_stats['duplicate']}, rows kept {draw_stats['kept_rows']}, "
          f"unmapped {draw_stats['unmapped']}")
    if draw_stats["new_bytes"]:
        print(f"  {draw_stats['old_bytes']/2**30:.2f} GiB -> "
              f"{draw_stats['new_bytes']/2**20:.1f} MiB")

    print("\nPhase 3 — dataset embeddings")
    emb_stats = migrate_embeddings(args.root, mapping, dry_run)
    print(f"  {emb_stats['moved']} entry(ies) re-keyed, "
          f"{emb_stats['duplicate']} duplicate(s) collapsed, "
          f"{emb_stats['skipped']} skipped")

    print("\nPhase 4 — adapters")
    adapter_stats = migrate_adapters(args.root, mapping, dry_run)
    print(f"  {adapter_stats['updated']} updated "
          f"({adapter_stats['seed_added']} gained training.seed), "
          f"{adapter_stats['unmapped']} unmapped")

    print("\nNext:")
    if dry_run:
        print("  python scripts/migrate_recipe_identity.py --apply")
    else:
        print("  python scripts/verify_sampled_cache.py --full")
        print("  python scripts/check_analysis.py")
        print("  python scripts/migrate_recipe_identity.py --prune   # only once green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
