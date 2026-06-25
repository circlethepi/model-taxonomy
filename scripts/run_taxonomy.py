"""Step 4: compute distance matrices and geometry embeddings for a model collection.

Reads cached representations produced by extract_reprs.py, then runs
TaxonomyAnalyzer to produce pairwise distance matrices and (optionally)
low-dimensional coordinate embeddings.  Results are saved to:
    {output_dir}/taxonomy/

Usage:
    python scripts/run_taxonomy.py experiments/example.yaml
    python scripts/run_taxonomy.py experiments/example.yaml --taxonomy functional behavioral
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._utils import (
    load_config,
    hf_token,
    resolve_model_ids,
    make_repr_cache,
    make_queries,
    make_functional_taxonomy,
    make_behavioral_taxonomy,
    make_structural_taxonomy,
    make_dataset_embedding_cache,
    make_dataset_embedding_taxonomy,
    make_metric,
    make_geometry,
    read_adapter_meta,
)
from src.compute.local import LocalBackend
from src.core.analysis import TaxonomyAnalyzer, ModelTaxonomyProfile


def _to_list(v) -> list[str]:
    return v if isinstance(v, list) else [v]


def _make_lora_cache(cfg: dict):
    from src.cache.lora_cache import LoRACache
    output_dir = Path(cfg["output_dir"])
    return LoRACache(output_dir / "cache" / "lora")


def run_taxonomy(cfg: dict, only_taxonomies: list[str] | None = None) -> ModelTaxonomyProfile:
    """Compute distance matrices and geometry for the configured model collection.

    Returns the populated ModelTaxonomyProfile.
    """
    output_dir = Path(cfg["output_dir"])
    tax_cfg = cfg.get("taxonomy", {})
    configured_taxonomies = tax_cfg.get("taxonomies", ["functional", "behavioral"])
    geometry_names = tax_cfg.get("geometry", ["pca"])
    metrics_cfg = tax_cfg.get("metrics", {})

    if only_taxonomies:
        configured_taxonomies = [t for t in configured_taxonomies if t in only_taxonomies]

    model_ids = resolve_model_ids(cfg, section_key="taxonomy")
    if not model_ids and (set(configured_taxonomies) - {"dataset_embedding"}):
        print("  No models configured. Check taxonomy.models in your config.")
        return ModelTaxonomyProfile(model_ids=[])

    if model_ids:
        print(f"  Models ({len(model_ids)}):")
        for mid in model_ids:
            print(f"    {mid}")

    backend = LocalBackend(n_jobs=1)
    repr_cache = make_repr_cache(output_dir)
    profile = ModelTaxonomyProfile(model_ids=model_ids)

    # ── Functional ─────────────────────────────────────────────────────────────
    if "functional" in configured_taxonomies:
        print("\n  [functional]")
        queries = make_queries(cfg)
        taxonomy = make_functional_taxonomy(cfg, queries, cache=repr_cache)
        metric_names = _to_list(metrics_cfg.get("functional", "cka"))

        for metric_name in metric_names:
            metric = make_metric(metric_name)
            analyzer = TaxonomyAnalyzer(taxonomy, metric, backend)
            analysis = analyzer.fit(model_ids)

            for geo_name in geometry_names:
                geo = make_geometry(geo_name)
                analysis.geometry = geo.fit(analysis.distance_matrix)
                print(f"    [{metric_name}] geometry [{geo_name}] computed  coords={analysis.geometry.coordinates.shape}")

            if len(metric_names) > 1:
                analysis.taxonomy_name = f"functional_{metric_name}"

            profile.add(analysis)

    # ── Behavioral ─────────────────────────────────────────────────────────────
    if "behavioral" in configured_taxonomies:
        print("\n  [behavioral]")
        queries = make_queries(cfg)
        taxonomy = make_behavioral_taxonomy(cfg, queries, cache=repr_cache)
        metric_names = _to_list(metrics_cfg.get("behavioral", "frobenius"))

        for metric_name in metric_names:
            metric = make_metric(metric_name)
            analyzer = TaxonomyAnalyzer(taxonomy, metric, backend)
            analysis = analyzer.fit(model_ids)

            for geo_name in geometry_names:
                geo = make_geometry(geo_name)
                analysis.geometry = geo.fit(analysis.distance_matrix)
                print(f"    [{metric_name}] geometry [{geo_name}] computed  coords={analysis.geometry.coordinates.shape}")

            if len(metric_names) > 1:
                analysis.taxonomy_name = f"behavioral_{metric_name}"

            profile.add(analysis)

    # ── Dataset Embedding ──────────────────────────────────────────────────────
    if "dataset_embedding" in configured_taxonomies:
        print("\n  [dataset_embedding]")
        de_cache = make_dataset_embedding_cache(output_dir)
        taxonomy = make_dataset_embedding_taxonomy(cfg, cache=de_cache)
        recipe_ids = taxonomy.recipe_ids()

        de_repr = cfg.get("extraction", {}).get("taxonomies", {}).get("dataset_embedding", {}).get("representation", "matrix")
        default_de_metric = "cosine" if de_repr == "mean" else "frobenius"
        metric_names = _to_list(metrics_cfg.get("dataset_embedding", default_de_metric))

        for metric_name in metric_names:
            metric = make_metric(metric_name)
            analyzer = TaxonomyAnalyzer(taxonomy, metric, backend)
            analysis = analyzer.fit(recipe_ids)
            analysis.representations = []  # already cached in DatasetEmbeddingCache

            for geo_name in geometry_names:
                geo = make_geometry(geo_name)
                analysis.geometry = geo.fit(analysis.distance_matrix)
                print(f"    [{metric_name}] geometry [{geo_name}] computed  coords={analysis.geometry.coordinates.shape}")

            if len(metric_names) > 1:
                analysis.taxonomy_name = f"dataset_embedding_{metric_name}"

            profile.add(analysis)

        other_model_taxonomies = {"functional", "behavioral", "structural"}
        if not any(t in configured_taxonomies for t in other_model_taxonomies):
            profile.model_ids = recipe_ids

    # ── Structural ─────────────────────────────────────────────────────────────
    if "structural" in configured_taxonomies:
        print("\n  [structural]")
        lora_cache = _make_lora_cache(cfg)

        from src.taxonomy.structural import StructuralTaxonomy
        from src.core.representation import ModelRepresentation
        from src.core.distance import DistanceMatrix
        from src.core.analysis import TaxonomyAnalysis
        import numpy as np

        representations: list[ModelRepresentation] = []

        for model_id in model_ids:
            meta = read_adapter_meta(model_id)
            base_model_id = meta.get("base_model_id")
            stax = StructuralTaxonomy(
                lora_only=True,
                cache=repr_cache,
                lora_cache=lora_cache,
                base_model_id=base_model_id,
                hf_token=hf_token(cfg),
            )
            print(f"    Extracting {model_id}", end=" ... ", flush=True)
            try:
                rep = stax.extract(model_id)
                representations.append(rep)
                print(f"shape={rep.matrix.shape}")
            except ValueError as e:
                print(f"SKIPPED ({e})")

        if len(representations) < 2:
            print("    Not enough models with LoRA adapters for structural taxonomy — skipping.")
        else:
            valid_ids = [r.model_id for r in representations]
            metric_names = _to_list(metrics_cfg.get("structural", "frobenius"))

            for metric_name in metric_names:
                metric = make_metric(metric_name)
                n = len(representations)
                dist_arr = np.zeros((n, n), dtype=np.float64)
                for i in range(n):
                    for j in range(i + 1, n):
                        d = metric.compute(representations[i], representations[j])
                        dist_arr[i, j] = dist_arr[j, i] = d

                tax_name = "structural" if len(metric_names) == 1 else f"structural_{metric_name}"
                dm = DistanceMatrix(
                    matrix=dist_arr,
                    model_ids=valid_ids,
                    metric=metric.metric_name,
                    taxonomy="structural",
                )
                # representations=[] avoids re-saving weights already in lora_cache
                analysis = TaxonomyAnalysis(
                    taxonomy_name=tax_name,
                    model_ids=valid_ids,
                    representations=[],
                    distance_matrix=dm,
                )

                for geo_name in geometry_names:
                    if len(representations) < 3 and geo_name == "umap":
                        print(f"    geometry [{geo_name}] skipped (need ≥3 models)")
                        continue
                    geo = make_geometry(geo_name)
                    analysis.geometry = geo.fit(dm)
                    print(f"    [{metric_name}] geometry [{geo_name}] computed  coords={analysis.geometry.coordinates.shape}")

                profile.add(analysis)

    # ── Save all results under taxonomy/ ──────────────────────────────────────
    if profile.taxonomy_names():
        tax_path = output_dir / "taxonomy"
        profile.save(tax_path)
        print(f"\n  Taxonomy results saved to {tax_path}")

    return profile


def main(cfg: dict, only_taxonomies: list[str] | None = None) -> None:
    print("=== Step 4: Run taxonomy analysis ===")
    profile = run_taxonomy(cfg, only_taxonomies=only_taxonomies)
    print(f"\nDone. Taxonomies: {profile.taxonomy_names()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute taxonomy distance matrices from an experiment YAML."
    )
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--taxonomy",
        nargs="+",
        metavar="NAME",
        help="Only run these taxonomies (e.g. --taxonomy functional behavioral).",
    )
    args = parser.parse_args()
    main(load_config(args.config), only_taxonomies=args.taxonomy)
