"""Shared helpers used by all experiment scripts."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import re
import warnings

import torch
import yaml


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _nice_sizes(k: int) -> list[int]:
    """Sorted {1,2,5} × 10^j for j = 0 … k."""
    return sorted({c * 10**j for c in (1, 2, 5) for j in range(k + 1)})


def _tens_sizes(k: int) -> list[int]:
    """[10^j for j = 0 … k]."""
    return [10**j for j in range(k + 1)]


def expand_dataset_n_samples(cfg: dict) -> dict:
    """Expand dataset blocks that carry an 'n_samples_sweep' field.

    n_samples_sweep: nice          → {1,2,5}×10^j for j=0..4  (15 values)
    n_samples_sweep: nice 3        → {1,2,5}×10^j for j=0..3  (12 values)
    n_samples_sweep: tens 4        → 10^j for j=0..4           (5 values)
    n_samples_sweep: [1, 10, 100]  → explicit list

    An optional ``max_samples`` key on the dataset block pre-filters the
    generated list at parse time: any size above the cap is skipped with a
    UserWarning.

    Each value n produces an entry named ``{base_name}_n{n}`` with
    ``n_samples`` set to n.  Blocks without ``n_samples_sweep`` pass through
    unchanged.  Call this before expand_dataset_seeds so final names follow
    the pattern ``{base}_n{n}_s{seed:02d}``.

    Returns a new cfg dict (does not mutate the input).
    """
    cfg = copy.deepcopy(cfg)
    expanded: list[dict] = []
    for ds in cfg.get("datasets", []):
        sweep = ds.pop("n_samples_sweep", None)
        if sweep is None:
            expanded.append(ds)
            continue
        if isinstance(sweep, str):
            parts = sweep.split()
            k = int(parts[1]) if len(parts) > 1 else 4
            sizes = _nice_sizes(k) if parts[0] == "nice" else _tens_sizes(k)
        else:
            sizes = sorted(sweep)
        # Optional parse-time cap via max_samples key
        cap = ds.pop("max_samples", None)
        if cap is not None:
            kept = [n for n in sizes if n <= cap]
            skipped = [n for n in sizes if n > cap]
            if skipped:
                warnings.warn(
                    f"Dataset '{ds['name']}': skipping n_samples={skipped} "
                    f"(exceed max_samples={cap}).",
                    UserWarning, stacklevel=2,
                )
            sizes = kept
        base_name = ds["name"]
        for n in sizes:
            entry = copy.deepcopy(ds)
            entry["name"] = f"{base_name}_n{n}"
            entry["n_samples"] = n
            expanded.append(entry)
    cfg["datasets"] = expanded
    return cfg


def expand_dataset_seeds(cfg: dict) -> dict:
    """Expand dataset blocks that carry a 'seeds' list into one block per seed.

    A dataset block with ``seeds: [0, 1, 2]`` is replaced by three blocks whose
    names are ``{base_name}_s00``, ``{base_name}_s01``, ``{base_name}_s02`` with
    ``seed`` set to the corresponding value.  Blocks without a ``seeds`` key are
    passed through unchanged (backward compatible).

    Returns a new cfg dict (does not mutate the input).
    """
    cfg = copy.deepcopy(cfg)
    expanded: list[dict] = []
    for ds in cfg.get("datasets", []):
        seeds = ds.pop("seeds", None)
        if seeds is None:
            expanded.append(ds)
        else:
            base_name = ds["name"]
            for seed_val in seeds:
                entry = copy.deepcopy(ds)
                entry["name"] = f"{base_name}_s{seed_val:02d}"
                entry["seed"] = seed_val
                expanded.append(entry)
    cfg["datasets"] = expanded
    return cfg


def build_recipe_from_cfg(ds_cfg: dict):
    """Build a DatasetRecipe or ClassAwareDatasetRecipe from a YAML dataset block."""
    name = ds_cfg["name"]
    recipe_type = ds_cfg.get("recipe_type", "simple")
    if recipe_type == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
        entries = [
            ClassDatasetEntry(
                dataset_id=e["dataset_id"],
                split=e.get("split", "train"),
                weight=e.get("weight", 1.0),
                text_field=e.get("text_field", "text"),
                class_field=e.get("class_field", "label"),
                subset=e.get("subset"),
                class_filter=e.get("class_filter"),
                class_weights=e.get("class_weights"),
            )
            for e in ds_cfg["entries"]
        ]
        return ClassAwareDatasetRecipe(name=name, datasets=entries)
    else:
        from src.datasets.recipe import DatasetRecipe, DatasetEntry
        entries = [
            DatasetEntry(
                dataset_id=e["dataset_id"],
                split=e.get("split", "train"),
                weight=e.get("weight", 1.0),
                text_field=e.get("text_field", "text"),
                subset=e.get("subset"),
            )
            for e in ds_cfg["entries"]
        ]
        return DatasetRecipe(name=name, datasets=entries)


def apply_dataset_size_caps(
    cfg: dict,
    hf_token: str | None = None,
) -> dict:
    """Probe each recipe's true capacity and trim the expanded dataset list.

    Call this after ``expand_dataset_n_samples`` and ``expand_dataset_seeds``.

    For each group of entries that share the same underlying recipe (same
    ``recipe_hash``), the dataset is loaded once at ``total_samples=max_n``
    to measure the true capacity.  If ``capacity < max_n``:

    * All entries with ``n_samples > capacity`` are removed (with a warning).
    * If no remaining entry has ``n_samples == capacity``, one entry per
      unique name-prefix is inserted at that size (one per seed variant).

    Datasets that have no ``n_samples`` (i.e. not from a sweep) and datasets
    whose config lacks an ``entries`` key are passed through unchanged.

    Returns a new cfg dict (does not mutate the input).
    """
    from collections import defaultdict

    cfg = copy.deepcopy(cfg)
    datasets = cfg.get("datasets", [])
    if not datasets:
        return cfg

    recipe_by_hash: dict[str, Any] = {}
    entry_hash: list[str | None] = []

    for ds_cfg in datasets:
        if "entries" not in ds_cfg or ds_cfg.get("n_samples") is None:
            entry_hash.append(None)
            continue
        try:
            recipe = build_recipe_from_cfg(ds_cfg)
        except Exception:
            entry_hash.append(None)
            continue
        rh = recipe.recipe_hash()
        recipe_by_hash[rh] = recipe
        entry_hash.append(rh)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, rh in enumerate(entry_hash):
        if rh is not None:
            groups[rh].append(i)

    to_skip: set[int] = set()
    insert_before: dict[int, list[dict]] = defaultdict(list)

    for rh, indices in groups.items():
        n_vals = [datasets[i]["n_samples"] for i in indices]
        max_n = max(n_vals)
        recipe = recipe_by_hash[rh]

        # Load at max_n: if capacity < max_n we get fewer samples back
        probe = make_mixed_dataset(recipe, total_samples=max_n, seed=42, hf_token=hf_token)
        capacity = len(probe)

        if capacity >= max_n:
            continue

        above = [i for i in indices if datasets[i]["n_samples"] > capacity]
        below = [i for i in indices if datasets[i]["n_samples"] <= capacity]

        oversized = sorted({datasets[i]["n_samples"] for i in above})
        warnings.warn(
            f"Dataset cap: recipe capacity is {capacity}. "
            f"Skipping n_samples={oversized} for '{datasets[indices[0]]['name']}' (and siblings).",
            UserWarning, stacklevel=2,
        )
        to_skip.update(above)

        remaining = {datasets[i]["n_samples"] for i in below}
        if capacity in remaining:
            continue

        # Insert one entry at capacity per unique name-prefix (seed variant)
        seen_bases: set[str] = set()
        first_above = min(above)
        for i in above:
            base = re.sub(r"_n\d+(?=(_s\d+)?$)", "", datasets[i]["name"])
            if base in seen_bases:
                continue
            seen_bases.add(base)
            new_entry = copy.deepcopy(datasets[i])
            if re.search(r"_s\d+$", base):
                new_entry["name"] = re.sub(r"(_s\d+)$", f"_n{capacity}\\1", base)
            else:
                new_entry["name"] = f"{base}_n{capacity}"
            new_entry["n_samples"] = capacity
            insert_before[first_above].append(new_entry)

        if insert_before.get(first_above):
            warnings.warn(
                f"Inserting {len(insert_before[first_above])} dataset(s) at "
                f"n_samples={capacity} to fill the capacity gap.",
                UserWarning, stacklevel=2,
            )

    new_datasets: list[dict] = []
    for i, ds_cfg in enumerate(datasets):
        for new_entry in insert_before.get(i, []):
            new_datasets.append(new_entry)
        if i not in to_skip:
            new_datasets.append(ds_cfg)

    cfg["datasets"] = new_datasets
    return cfg


def load_recipe(path: Path | str):
    """Load DatasetRecipe or ClassAwareDatasetRecipe by inspecting recipe_type."""
    import json as _json
    data = _json.loads(Path(path).read_text())
    if data.get("recipe_type") == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe
        return ClassAwareDatasetRecipe.load(path)
    from src.datasets.recipe import DatasetRecipe
    return DatasetRecipe.load(path)


def make_mixed_dataset(
    recipe,
    total_samples: int,
    seed: int = 42,
    hf_token: str | None = None,
    sample_cache=None,
):
    """Instantiate MixedDataset or ClassMixedDataset depending on recipe type.

    If *sample_cache* is provided, checks for a cached list[dict] keyed by
    ``(recipe_hash, total_samples, seed)`` and returns a ``CachedMixedDataset``
    on a hit.  On a miss the dataset is loaded from HuggingFace and the rows
    are written to the cache for future calls.
    """
    from src.datasets.class_recipe import ClassAwareDatasetRecipe

    if sample_cache is not None:
        cached_rows = sample_cache.get(recipe.recipe_hash(), total_samples, seed)
        if cached_rows is not None:
            from src.datasets.mixed_dataset import CachedMixedDataset
            return CachedMixedDataset(cached_rows, recipe)

    if isinstance(recipe, ClassAwareDatasetRecipe):
        from src.datasets.mixed_dataset import ClassMixedDataset
        ds = ClassMixedDataset(recipe, total_samples=total_samples, seed=seed, hf_token=hf_token)
    else:
        from src.datasets.mixed_dataset import MixedDataset
        ds = MixedDataset(recipe, total_samples=total_samples, seed=seed, hf_token=hf_token)

    if sample_cache is not None:
        sample_cache.put(recipe.recipe_hash(), total_samples, seed, list(ds))

    return ds


def hf_token(cfg: dict) -> str | None:
    return cfg.get("hf_token") or os.environ.get("HF_TOKEN") or None


# ── Cache dir resolution ───────────────────────────────────────────────────────

def get_cache_dir(cfg: dict) -> Path:
    """Return the cache root for this experiment.

    If the experiment YAML contains a top-level ``cache_dir`` key the value is
    used as-is, allowing multiple experiments to share a single cache tree.
    Otherwise falls back to ``{output_dir}/cache`` (the original per-experiment
    behaviour, fully backward compatible).
    """
    if "cache_dir" in cfg:
        return Path(cfg["cache_dir"])
    return Path(cfg["output_dir"]) / "cache"


# ── Model ID resolution ────────────────────────────────────────────────────────

def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def adapter_dir(output_dir: Path, base_model_id: str, dataset_name: str, lora_rank: int) -> Path:
    return output_dir / "adapters" / _model_slug(base_model_id) / f"{dataset_name}_r{lora_rank}"


def discover_adapter_paths(output_dir: Path) -> list[str]:
    """Return local paths of all trained adapters (those with experiment_meta.json)."""
    adapters_root = output_dir / "adapters"
    if not adapters_root.exists():
        return []
    paths = []
    for meta_file in sorted(adapters_root.rglob("experiment_meta.json")):
        paths.append(str(meta_file.parent))
    return paths


def read_adapter_meta(adapter_path: str | Path) -> dict:
    meta_file = Path(adapter_path) / "experiment_meta.json"
    if meta_file.exists():
        return json.loads(meta_file.read_text())
    return {}


def resolve_model_ids(cfg: dict, section_key: str = "models") -> list[str]:
    """Resolve model IDs for a given section (extraction or taxonomy).

    Supports three tokens in the 'models' list:
      - "base_models"  → the base_models list from cfg
      - "fine_tuned"   → all adapter paths discovered under output_dir/adapters/
      - any other str  → treated as an explicit HF ID or local path
    """
    output_dir = Path(cfg["output_dir"])
    section = cfg.get(section_key, cfg)  # extraction or taxonomy sub-dict, or cfg itself
    model_tokens = section.get("models", ["base_models"])

    model_ids: list[str] = []
    seen: set[str] = set()

    for token in model_tokens:
        if token == "base_models":
            for mid in cfg.get("base_models", []):
                if mid not in seen:
                    model_ids.append(mid)
                    seen.add(mid)
        elif token == "fine_tuned":
            for path in discover_adapter_paths(output_dir):
                if path not in seen:
                    model_ids.append(path)
                    seen.add(path)
        else:
            if token not in seen:
                model_ids.append(token)
                seen.add(token)

    return model_ids


# ── Dtype helper ──────────────────────────────────────────────────────────────

def parse_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


# ── Cache factories ───────────────────────────────────────────────────────────

def make_repr_cache(cache_dir: Path):
    from src.cache.disk import DiskCache
    return DiskCache(cache_dir / "representations")


def make_dataset_embedding_cache(cache_dir: Path):
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache
    return DatasetEmbeddingCache(cache_dir)


def make_sampled_dataset_cache(cache_dir: Path):
    from src.cache.sampled_dataset_cache import SampledDatasetCache
    return SampledDatasetCache(cache_dir / "sampled_datasets")


# ── Taxonomy / metric / geometry factories ────────────────────────────────────

def make_queries(cfg: dict) -> list[str]:
    """Load query strings from the configured queries_dataset."""
    output_dir = Path(cfg["output_dir"])
    ext_cfg = cfg.get("extraction", {})
    dataset_name = ext_cfg.get("queries_dataset")
    n_queries = ext_cfg.get("n_queries", 128)

    if dataset_name is None:
        raise ValueError("extraction.queries_dataset must be set in the config.")

    recipe_path = output_dir / "datasets" / f"{dataset_name}.recipe.json"
    if not recipe_path.exists():
        raise FileNotFoundError(
            f"Recipe not found at {recipe_path}. Run build_datasets.py first."
        )

    seed = next(
        (d.get("seed", 42) for d in cfg.get("datasets", []) if d["name"] == dataset_name),
        42,
    )
    recipe = load_recipe(recipe_path)
    mixed = make_mixed_dataset(recipe, total_samples=n_queries, seed=seed, hf_token=hf_token(cfg))
    return mixed.to_queries(n=n_queries)


def make_functional_taxonomy(cfg: dict, queries: list[str], cache=None):
    from src.taxonomy.functional import FunctionalTaxonomy

    ext_cfg = cfg.get("extraction", {})
    fcfg = ext_cfg.get("taxonomies", {}).get("functional", {})

    return FunctionalTaxonomy(
        queries=queries,
        layer_indices=fcfg.get("layer_indices", [-1, -4, -8]),
        cache=cache,
        device=ext_cfg.get("device", "cuda"),
        batch_size=ext_cfg.get("batch_size", 8),
        torch_dtype=parse_dtype(ext_cfg.get("torch_dtype", "float16")),
        hf_token=hf_token(cfg),
        pooling=fcfg.get("pooling", "mean"),
        normalize_activations=fcfg.get("normalize_activations", True),
        activation_mode=fcfg.get("activation_mode", "input"),
        max_new_tokens=fcfg.get("max_new_tokens", 32),
        representation=fcfg.get("representation", "gram"),
    )


def make_behavioral_taxonomy(cfg: dict, queries: list[str], cache=None):
    from src.taxonomy.behavioral import BehavioralTaxonomy
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder

    ext_cfg = cfg.get("extraction", {})
    bcfg = ext_cfg.get("taxonomies", {}).get("behavioral", {})
    ecfg = bcfg.get("embedder", {})

    embedder = SentenceTransformerEmbedder(
        model_name=ecfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device="cpu",
        use_generated_text=True,
        normalize_embeddings=ecfg.get("normalize_embeddings", True),
        trust_remote_code=ecfg.get("trust_remote_code", False),
        prompt_name=ecfg.get("prompt_name"),
    )
    return BehavioralTaxonomy(
        queries=queries,
        embedder=embedder,
        cache=cache,
        device=ext_cfg.get("device", "cuda"),
        batch_size=ext_cfg.get("batch_size", 8),
        max_new_tokens=bcfg.get("max_new_tokens", 64),
        torch_dtype=parse_dtype(ext_cfg.get("torch_dtype", "float16")),
        hf_token=hf_token(cfg),
    )


def make_structural_taxonomy(cfg: dict, cache=None, lora_cache=None):
    from src.taxonomy.structural import StructuralTaxonomy

    return StructuralTaxonomy(
        lora_only=True,
        cache=cache,
        lora_cache=lora_cache,
        hf_token=hf_token(cfg),
    )


def make_metric(name: str):
    if name == "cka":
        from src.metrics.cka import CKADistanceMetric
        return CKADistanceMetric()
    elif name == "frobenius":
        from src.metrics.frobenius import FrobeniusDistanceMetric
        return FrobeniusDistanceMetric()
    elif name == "cosine":
        from src.metrics.vector import CosineDistanceMetric
        return CosineDistanceMetric()
    elif name == "dot_product":
        from src.metrics.vector import DotProductDistanceMetric
        return DotProductDistanceMetric()
    else:
        raise ValueError(f"Unknown metric: {name!r}. Choose from cka, frobenius, cosine, dot_product.")


def make_geometry(name: str):
    if name == "pca":
        from src.geometry_methods.pca import PCAGeometry
        return PCAGeometry(n_components=2)
    elif name == "mds":
        from src.geometry_methods.mds import MDSGeometry
        return MDSGeometry(n_components=2)
    elif name == "umap":
        from src.geometry_methods.umap import UMAPGeometry
        return UMAPGeometry(n_components=2)
    else:
        raise ValueError(f"Unknown geometry method: {name!r}. Choose from pca, mds, umap.")


def make_dataset_embedding_taxonomy(cfg: dict, cache=None, sample_cache=None):
    from src.taxonomy.dataset_embedding import DatasetEmbeddingTaxonomy
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder

    output_dir = Path(cfg["output_dir"])
    ext_cfg = cfg.get("extraction", {})
    decfg = ext_cfg.get("taxonomies", {}).get("dataset_embedding", {})
    ecfg = decfg.get("embedder", {})
    n_samples = decfg.get("n_samples", 200)
    global_seed = decfg.get("seed", 42)

    # Build datasets dict with per-dataset seeds (3-tuple).
    # Per-dataset seed comes from the expanded YAML's 'seed' field; falls back
    # to the global embedding seed if a dataset block has no seed.
    datasets: dict[str, Any] = {}
    for ds in cfg.get("datasets", []):
        recipe_path = output_dir / "datasets" / f"{ds['name']}.recipe.json"
        per_ds_seed = ds.get("seed", global_seed)
        ds_n_samples = ds.get("n_samples", n_samples)
        datasets[ds["name"]] = (load_recipe(recipe_path), ds_n_samples, per_ds_seed)

    embedder = SentenceTransformerEmbedder(
        model_name=ecfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device="cpu",
        use_generated_text=False,
        normalize_embeddings=ecfg.get("normalize_embeddings", True),
        trust_remote_code=ecfg.get("trust_remote_code", False),
        prompt_name=ecfg.get("prompt_name"),
    )
    return DatasetEmbeddingTaxonomy(
        embedder=embedder,
        datasets=datasets,
        representation=decfg.get("representation", "matrix"),
        cache=cache,
        seed=global_seed,
        hf_token=hf_token(cfg),
        sample_cache=sample_cache,
    )
