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
    """Sorted {1,2,5} × 10^j up to 10^k."""
    cap = 10**k
    return sorted({c * 10**j for c in (1, 2, 5) for j in range(k + 1) if c * 10**j <= cap})


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
            # The block name must stay unique — it keys the datasets dict, the taxonomy
            # labels and the per-experiment recipe filename.  'mixture' remembers the
            # name before expansion, and that is what names the recipe, so all n of one
            # mixture share a single content-addressed recipe.
            entry.setdefault("mixture", base_name)
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
                # setdefault, not assignment: when expand_dataset_n_samples ran first
                # base_name already carries _n{n}, and the mixture it recorded is the
                # one we want to keep.
                entry.setdefault("mixture", base_name)
                expanded.append(entry)
    cfg["datasets"] = expanded
    return cfg


def compute_recipe_capacity(recipe, hf_token: str | None = None) -> int:
    """Return the maximum sample count this recipe can deliver while maintaining
    all class/entry weight ratios.

    For each entry, finds the most-constrained class (or the dataset itself for
    simple entries) and derives the effective total as ``min_c(size_c / w_c)``.
    Across entries, takes the minimum of ``entry_capacity / entry_weight``.
    """
    from collections import Counter
    from datasets import load_dataset  # type: ignore[import]
    from src.datasets.class_recipe import ClassDatasetEntry

    effective: float = float("inf")

    for entry, w in zip(recipe.datasets, recipe.normalized_weights):
        if w <= 0:
            continue
        ds = load_dataset(
            entry.dataset_id,
            getattr(entry, "subset", None),
            split=getattr(entry, "split", "train"),
            token=hf_token,
        )

        if isinstance(entry, ClassDatasetEntry):
            if entry.class_filter is not None:
                allowed = set(entry.class_filter)
                ds = ds.filter(lambda row: row[entry.class_field] in allowed)
            if len(ds) == 0:
                return 0

            class_norm_w = entry.normalized_class_weights
            if class_norm_w is None:
                present = list(set(ds[entry.class_field]))
                class_norm_w = {c: 1.0 / len(present) for c in present}

            class_sizes: Counter = Counter(ds[entry.class_field])
            entry_cap = min(
                (class_sizes.get(c, 0) / cw for c, cw in class_norm_w.items() if cw > 0),
                default=0.0,
            )
        else:
            entry_cap = float(len(ds))

        effective = min(effective, entry_cap / w)

    return int(effective) if effective != float("inf") else 0


def load_recipe(path: Path | str):
    """Load DatasetRecipe or ClassAwareDatasetRecipe by inspecting recipe_type."""
    import json as _json
    data = _json.loads(Path(path).read_text())
    if data.get("recipe_type") == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe
        return ClassAwareDatasetRecipe.load(path)
    from src.datasets.recipe import DatasetRecipe
    return DatasetRecipe.load(path)


def build_recipe_from_cfg(ds_cfg: dict):
    """Build a DatasetRecipe or ClassAwareDatasetRecipe from an expanded dataset config block.

    The recipe is named for the *mixture*, not the expanded block.  The block name
    carries ``_n{n}_s{seed}`` so it can key the datasets dict and label taxonomy points;
    the recipe underneath is the same mixing spec at every n and seed, and the recipe
    hash is content-addressed anyway, so giving it the expanded name would only put a
    misleading label on a shared object.  Blocks written by hand, with no ``mixture``
    key, fall back to their own name.
    """
    rtype = ds_cfg.get("recipe_type", "simple")
    name = ds_cfg.get("mixture", ds_cfg["name"])
    entries_raw = ds_cfg.get("entries", [])
    if rtype == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
        entries = [ClassDatasetEntry.from_dict(e) for e in entries_raw]
        return ClassAwareDatasetRecipe(name=name, datasets=entries)
    from src.datasets.recipe import DatasetRecipe, DatasetEntry
    entries = [DatasetEntry.from_dict(e) for e in entries_raw]
    return DatasetRecipe(name=name, datasets=entries)


def apply_dataset_capacity_caps(cfg: dict) -> dict:
    """Cap each dataset's n_samples to the true recipe capacity and rename it.

    After ``expand_dataset_n_samples`` produces multiple blocks at different
    sizes, some may exceed how many samples the underlying data can actually
    deliver while maintaining class/entry proportions.  This function:

    1. Computes the capacity for each unique recipe (loading dataset sizes from
       HuggingFace cache; typically free after the first ``build`` step).
    2. For any block whose n_samples exceeds the capacity, reduces n_samples
       to the capacity and renames the block (``_n200000_`` → ``_n186667_``).
    3. Removes duplicate blocks that collapse to the same name after renaming,
       keeping only the first occurrence (lowest original n_samples).
    4. Emits a UserWarning for every change and every removed duplicate.

    Returns a new cfg dict (does not mutate the input).
    """
    cfg = copy.deepcopy(cfg)
    output_dir = Path(cfg["output_dir"])
    token = hf_token(cfg)

    recipe_capacity_cache: dict[str, int] = {}
    updated: list[dict] = []
    seen_names: set[str] = set()

    for ds in cfg.get("datasets", []):
        recipe_path = output_dir / "datasets" / f"{ds['name']}.recipe.json"

        if not recipe_path.exists():
            if ds["name"] not in seen_names:
                seen_names.add(ds["name"])
                updated.append(ds)
            continue

        recipe = load_recipe(recipe_path)
        rhash = recipe.recipe_hash()

        if rhash not in recipe_capacity_cache:
            recipe_capacity_cache[rhash] = compute_recipe_capacity(recipe, hf_token=token)
        capacity = recipe_capacity_cache[rhash]

        n_samples = ds.get("n_samples")
        if n_samples is not None and capacity > 0 and n_samples > capacity:
            old_name = ds["name"]
            new_name = re.sub(r"_n\d+(?=_|$)", f"_n{capacity}", old_name)
            warnings.warn(
                f"Dataset cap: recipe capacity is {capacity}. "
                f"Reducing n_samples from {n_samples} to {capacity} "
                f"and renaming '{old_name}' → '{new_name}'.",
                stacklevel=2,
            )
            ds = dict(ds)
            ds["name"] = new_name
            ds["n_samples"] = capacity

        if ds["name"] in seen_names:
            warnings.warn(
                f"Removing duplicate dataset '{ds['name']}' "
                f"(collapsed to same capacity as an earlier entry).",
                stacklevel=2,
            )
            continue

        seen_names.add(ds["name"])
        updated.append(ds)

    cfg["datasets"] = updated
    return cfg


def make_mixed_dataset(
    recipe,
    total_samples: int,
    seed: int = 42,
    hf_token: str | None = None,
    sample_cache=None,
    name: str | None = None,
):
    """Instantiate MixedDataset or ClassMixedDataset depending on recipe type.

    If *sample_cache* is provided, checks for a cached draw keyed by
    ``(recipe_hash, total_samples, seed)`` and returns a ``CachedMixedDataset``
    on a hit.  On a miss the dataset is loaded from HuggingFace and the draw
    is written to the cache for future calls.

    The recipe itself is mirrored alongside its draws on either path, so the cache
    stays the hash-indexed home for recipes rather than depending on the dataset
    having also been embedded.  *name*, when given, is recorded as a label for the
    recipe: the hash is content-addressed, so several config-block names legitimately
    resolve to one recipe and the cache keeps all of them.
    """
    from src.datasets.class_recipe import ClassAwareDatasetRecipe

    recipe_hash = recipe.recipe_hash()

    if sample_cache is not None:
        sample_cache.put_recipe(recipe_hash, recipe.to_dict())
        sample_cache.add_name(recipe_hash, name)
        cached_rows = sample_cache.get(recipe_hash, total_samples, seed, hf_token=hf_token)
        if cached_rows is not None:
            from src.datasets.mixed_dataset import CachedMixedDataset
            manifest = sample_cache.get_manifest(recipe_hash, total_samples, seed) or {}
            return CachedMixedDataset(
                cached_rows,
                recipe,
                source_indices=[tuple(p) for p in manifest.get("indices", [])] or None,
                sources=manifest.get("sources"),
            )

    if isinstance(recipe, ClassAwareDatasetRecipe):
        from src.datasets.mixed_dataset import ClassMixedDataset
        ds = ClassMixedDataset(recipe, total_samples=total_samples, seed=seed, hf_token=hf_token)
    else:
        from src.datasets.mixed_dataset import MixedDataset
        ds = MixedDataset(recipe, total_samples=total_samples, seed=seed, hf_token=hf_token)

    if sample_cache is not None:
        rows = list(ds)
        sample_cache.put(
            recipe_hash, total_samples, seed,
            rows=rows, indices=ds.source_indices, sources=ds.sources,
        )

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


def get_adapter_root(cfg: dict) -> Path:
    """Return the root directory for raw PEFT adapter files.

    When a shared ``cache_dir`` is configured, adapters are stored under
    ``{cache_dir}/03_adapters/`` so they are shared across experiments.
    Falls back to ``{output_dir}/adapters/`` for backward compatibility — the
    numeric prefixes describe shared-cache pipeline stages, and an experiment
    output directory has no such ordering, so that name is left unnumbered.
    """
    if "cache_dir" in cfg:
        return Path(cfg["cache_dir"]) / "03_adapters"
    return Path(cfg["output_dir"]) / "adapters"


def adapter_dir(adapter_root: Path, base_model_id: str, dataset_name: str, lora_rank: int, lora_init_seed: int = 0) -> Path:
    return adapter_root / _model_slug(base_model_id) / f"{dataset_name}_r{lora_rank}_i{lora_init_seed:02d}"


def discover_adapter_paths(adapter_root: Path) -> list[str]:
    """Return local paths of all trained adapters (those with experiment_meta.json)."""
    if not adapter_root.exists():
        return []
    paths = []
    for meta_file in sorted(adapter_root.rglob("experiment_meta.json")):
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
      - "fine_tuned"   → all adapter paths discovered under get_adapter_root(cfg)
      - any other str  → treated as an explicit HF ID or local path
    """
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
            for path in discover_adapter_paths(get_adapter_root(cfg)):
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

#: Where each taxonomy's representations live under the shared cache.  Structural is
#: absent on purpose: its representations are written by LoRACache, beside the adapter
#: weights they were derived from, rather than into a stage directory of their own.
#: behavioral is absent: it outgrew DiskCache and has its own GeneratedTextCache,
#: which owns ``05_generated`` and lays it out differently.  Leaving the entry here
#: would let make_repr_cache hand back a DiskCache pointed at the same directory,
#: giving that tree two writers with incompatible layouts.
REPR_CACHE_DIRS = {
    "functional": "04_activations",
}


def make_repr_cache(cache_dir: Path, taxonomy: str):
    """DiskCache for one taxonomy's representations.

    Functional's provisional home: when that level is built out it is expected to
    grow its own cache class the way structural has LoRACache and behavioral now
    has GeneratedTextCache.
    """
    from src.cache.disk import DiskCache

    try:
        subdir = REPR_CACHE_DIRS[taxonomy]
    except KeyError:
        raise ValueError(
            f"no representation cache directory for taxonomy {taxonomy!r}; "
            f"choose from {sorted(REPR_CACHE_DIRS)}"
        ) from None
    return DiskCache(cache_dir / subdir)


def make_dataset_embedding_cache(cache_dir: Path):
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache
    return DatasetEmbeddingCache(cache_dir)


def make_sampled_dataset_cache(cache_dir: Path):
    from src.cache.sampled_dataset_cache import SampledDatasetCache
    return SampledDatasetCache(cache_dir)


def make_generated_text_cache(cache_dir: Path):
    from src.cache.generated_text_cache import GeneratedTextCache
    return GeneratedTextCache(cache_dir)


# ── Taxonomy / metric / geometry factories ────────────────────────────────────

def resolve_device(cfg: dict, override: str | None = None) -> str:
    """Device for an embedder, most specific setting first.

    1. *override* — a per-embedder ``device:`` from the taxonomy's own block.
    2. ``extraction.device`` — the experiment-wide setting.
    3. auto-detect.

    Auto-detection is the tier that matters: the old default handed the literal
    string ``"cuda"`` to whatever machine happened to be running, so a CPU-only
    host failed at model-load time rather than degrading to CPU.

    ``device`` is deliberately absent from
    :meth:`SentenceTransformerEmbedder.config_dict`, so changing it does not
    change any cache key and does not invalidate existing embeddings.  The flip
    side is that CPU- and GPU-computed embeddings share a key and differ in the
    low-order bits (different reduction orders, fused multiply-add, TF32 on
    Ampere+).  That is ~1e-6 on L2-normalized vectors — far below anything the
    comparison layer treats as signal — but it does mean byte-level equality
    across machines is not a property this cache has.
    """
    if override:
        return override
    configured = cfg.get("extraction", {}).get("device")
    if configured:
        return configured
    return "cuda" if torch.cuda.is_available() else "cpu"

def make_queries(cfg: dict) -> tuple[list[str], dict]:
    """Load query strings from the configured queries_dataset.

    Returns ``(queries, query_key)``, where *query_key* is the
    ``{recipe_hash, n_samples, seed}`` triple identifying the draw in
    ``01_datasets``.  Callers that cache representations keyed on the query set
    should store the key rather than the strings themselves: the strings are
    derived data, and hashing all of them makes a cache entry sensitive to every
    upstream change that shifts the draw, with no way to tell from the key which
    draw an entry belonged to.

    The draw is requested at exactly *n_queries*.  ``n`` enters the sampler, so a
    32-row draw is not the first 32 rows of a 64-row one — never slice a larger
    draw down.
    """
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
    sample_cache = make_sampled_dataset_cache(get_cache_dir(cfg))
    mixed = make_mixed_dataset(
        recipe,
        total_samples=n_queries,
        seed=seed,
        hf_token=hf_token(cfg),
        sample_cache=sample_cache,
        name=dataset_name,
    )
    query_key = {
        "recipe_hash": recipe.recipe_hash(),
        "n_samples": n_queries,
        "seed": seed,
    }
    return mixed.to_queries(n=n_queries), query_key


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


def make_behavioral_taxonomy(cfg: dict, queries: list[str], query_key: dict | None = None, cache=None):
    from src.taxonomy.behavioral import BehavioralTaxonomy
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder

    ext_cfg = cfg.get("extraction", {})
    bcfg = ext_cfg.get("taxonomies", {}).get("behavioral", {})
    ecfg = bcfg.get("embedder", {})

    embedder = SentenceTransformerEmbedder(
        model_name=ecfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=resolve_device(cfg, ecfg.get("device")),
        use_generated_text=True,
        normalize_embeddings=ecfg.get("normalize_embeddings", True),
        trust_remote_code=ecfg.get("trust_remote_code", False),
        prompt_name=ecfg.get("prompt_name"),
    )
    return BehavioralTaxonomy(
        queries=queries,
        embedder=embedder,
        query_key=query_key,
        cache=cache,
        device=ext_cfg.get("device", "cuda"),
        batch_size=ext_cfg.get("batch_size", 8),
        max_new_tokens=bcfg.get("max_new_tokens", 64),
        torch_dtype=parse_dtype(ext_cfg.get("torch_dtype", "float16")),
        hf_token=hf_token(cfg),
    )


def make_structural_taxonomy(cfg: dict, lora_cache=None):
    from src.taxonomy.structural import StructuralTaxonomy

    return StructuralTaxonomy(
        lora_only=True,
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


def make_geometry(name: str, n_components: int = 2):
    """Build a geometry method at a chosen dimension.

    The dimension used to be fixed at 2, which is right for a plot but rules out
    everything else: a barycentric projection onto a simplex with k vertices is
    undefined below k-1 dimensions, so a three-way dataset mixture could not be
    analysed at all through this path.  See ``taxonomy.n_components`` in the
    experiment YAML, and :func:`src.analysis.bridge.fit_geometry` for the
    library-level equivalent.
    """
    if name == "pca":
        from src.geometry_methods.pca import PCAGeometry
        return PCAGeometry(n_components=n_components)
    elif name == "mds":
        from src.geometry_methods.mds import MDSGeometry
        return MDSGeometry(n_components=n_components)
    elif name == "umap":
        from src.geometry_methods.umap import UMAPGeometry
        return UMAPGeometry(n_components=n_components)
    else:
        raise ValueError(f"Unknown geometry method: {name!r}. Choose from pca, mds, umap.")


def geometry_dims(cfg: dict) -> list[int]:
    """Dimensions to embed at, from ``taxonomy.n_components`` in the YAML.

    Accepts a scalar or a list; defaults to ``[2]``, which is what every existing
    config produces.
    """
    raw = cfg.get("taxonomy", {}).get("n_components", 2)
    dims = [raw] if isinstance(raw, int) else list(raw)
    if not dims or any(int(d) < 1 for d in dims):
        raise ValueError(f"taxonomy.n_components must be >= 1, got {raw!r}")
    return sorted({int(d) for d in dims})


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
        device=resolve_device(cfg, ecfg.get("device")),
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
