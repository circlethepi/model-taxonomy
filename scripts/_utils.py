"""Shared helpers used by all experiment scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import yaml


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_recipe(path: Path | str):
    """Load DatasetRecipe or ClassAwareDatasetRecipe by inspecting recipe_type."""
    import json as _json
    data = _json.loads(Path(path).read_text())
    if data.get("recipe_type") == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe
        return ClassAwareDatasetRecipe.load(path)
    from src.datasets.recipe import DatasetRecipe
    return DatasetRecipe.load(path)


def make_mixed_dataset(recipe, total_samples: int, seed: int = 42, hf_token: str | None = None):
    """Instantiate MixedDataset or ClassMixedDataset depending on recipe type."""
    from src.datasets.class_recipe import ClassAwareDatasetRecipe
    if isinstance(recipe, ClassAwareDatasetRecipe):
        from src.datasets.mixed_dataset import ClassMixedDataset
        return ClassMixedDataset(recipe, total_samples=total_samples, seed=seed, hf_token=hf_token)
    from src.datasets.mixed_dataset import MixedDataset
    return MixedDataset(recipe, total_samples=total_samples, seed=seed, hf_token=hf_token)


def hf_token(cfg: dict) -> str | None:
    return cfg.get("hf_token") or os.environ.get("HF_TOKEN") or None


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


# ── Taxonomy / metric / geometry factories ────────────────────────────────────

def make_repr_cache(output_dir: Path):
    from src.cache.disk import DiskCache
    return DiskCache(output_dir / "cache" / "representations")


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
    else:
        raise ValueError(f"Unknown metric: {name!r}. Choose 'cka' or 'frobenius'.")


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
