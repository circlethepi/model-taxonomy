"""Step 3: extract activations and/or outputs for a set of models.

Pre-populates the DiskCache so run_taxonomy.py can load representations
without re-running inference.

Usage:
    python scripts/extract_reprs.py experiments/example.yaml
    python scripts/extract_reprs.py experiments/example.yaml --taxonomy functional
    python scripts/extract_reprs.py experiments/example.yaml --taxonomy behavioral functional
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._utils import (
    load_config,
    expand_dataset_seeds,
    expand_dataset_n_samples,
    get_cache_dir,
    resolve_model_ids,
    make_repr_cache,
    make_queries,
    make_functional_taxonomy,
    make_behavioral_taxonomy,
    make_dataset_embedding_cache,
    make_sampled_dataset_cache,
    make_dataset_embedding_taxonomy,
)


def extract_representations(cfg: dict, only_taxonomies: list[str] | None = None) -> None:
    """Extract and cache representations for all configured models and taxonomies."""
    ext_cfg = cfg.get("extraction", {})
    tax_cfgs = ext_cfg.get("taxonomies", {})
    enabled = set(only_taxonomies) if only_taxonomies else None

    model_ids = resolve_model_ids(cfg, section_key="extraction")
    if not model_ids and (enabled is None or enabled - {"dataset_embedding"}):
        print("  No models to extract. Check extraction.models in your config.")
        return

    if model_ids:
        print(f"  Models ({len(model_ids)}):")
        for mid in model_ids:
            print(f"    {mid}")

    cache_dir = get_cache_dir(cfg)
    cache = make_repr_cache(cache_dir)

    # Only load queries when functional or behavioral extraction is actually enabled —
    # dataset_embedding does not use queries.
    fcfg = tax_cfgs.get("functional", {})
    bcfg = tax_cfgs.get("behavioral", {})
    need_queries = (
        fcfg.get("enabled", True) and (enabled is None or "functional" in enabled)
    ) or (
        bcfg.get("enabled", True) and (enabled is None or "behavioral" in enabled)
    )

    queries: list[str] = []
    if model_ids and need_queries:
        print("  Loading queries...")
        queries = make_queries(cfg)
        print(f"  Got {len(queries)} queries.")

    # ── Functional taxonomy ────────────────────────────────────────────────────
    if fcfg.get("enabled", True) and (enabled is None or "functional" in enabled):
        print(f"\n  [functional]  layers={fcfg.get('layer_indices', [-1, -4, -8])}"
              f"  mode={fcfg.get('activation_mode', 'input')}")
        taxonomy = make_functional_taxonomy(cfg, queries, cache=cache)
        for i, model_id in enumerate(model_ids, 1):
            print(f"    [{i}/{len(model_ids)}] {model_id}", end=" ... ", flush=True)
            rep = taxonomy.extract(model_id)
            print(f"shape={rep.matrix.shape}  key={rep.cache_key}")

    # ── Behavioral taxonomy ────────────────────────────────────────────────────
    if bcfg.get("enabled", True) and (enabled is None or "behavioral" in enabled):
        print(f"\n  [behavioral]  max_new_tokens={bcfg.get('max_new_tokens', 64)}")
        taxonomy = make_behavioral_taxonomy(cfg, queries, cache=cache)
        for i, model_id in enumerate(model_ids, 1):
            print(f"    [{i}/{len(model_ids)}] {model_id}", end=" ... ", flush=True)
            rep = taxonomy.extract(model_id)
            print(f"shape={rep.matrix.shape}  key={rep.cache_key}")

    # ── Dataset Embedding taxonomy ─────────────────────────────────────────────
    decfg = tax_cfgs.get("dataset_embedding", {})
    if decfg.get("enabled", False) and (enabled is None or "dataset_embedding" in enabled):
        print(f"\n  [dataset_embedding]  representation={decfg.get('representation', 'matrix')}")
        de_cache = make_dataset_embedding_cache(cache_dir)
        sample_cache = make_sampled_dataset_cache(cache_dir)
        taxonomy = make_dataset_embedding_taxonomy(cfg, cache=de_cache, sample_cache=sample_cache)
        recipe_ids = taxonomy.recipe_ids()
        for i, recipe_id in enumerate(recipe_ids, 1):
            print(f"    [{i}/{len(recipe_ids)}] {recipe_id}", end=" ... ", flush=True)
            rep = taxonomy.extract(recipe_id)
            print(f"shape={rep.matrix.shape}  key={rep.cache_key}")


def main(cfg: dict, only_taxonomies: list[str] | None = None) -> None:
    print("=== Step 3: Extract representations ===")
    extract_representations(cfg, only_taxonomies=only_taxonomies)
    print("\nDone. Representations cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract model representations from an experiment YAML."
    )
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--taxonomy",
        nargs="+",
        metavar="NAME",
        help="Only extract these taxonomies (e.g. --taxonomy functional behavioral).",
    )
    args = parser.parse_args()
    cfg = expand_dataset_seeds(expand_dataset_n_samples(load_config(args.config)))
    main(cfg, only_taxonomies=args.taxonomy)
