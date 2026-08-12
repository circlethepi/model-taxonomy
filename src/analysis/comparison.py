"""Comparing taxonomy levels to each other and to the recipe's ground truth.

:mod:`src.analysis.matrices`, :mod:`~src.analysis.configurations` and
:mod:`~src.analysis.simplex` each answer one question about a *pair* of objects.
This module runs the whole battery over a whole collection and puts the answers
in one place, so "which level of abstraction best recovers the training mixture?"
becomes a single call rather than a notebook of bookkeeping.

Two comparisons, and why both are here
--------------------------------------
**Barycentric** — anchor the measured embedding at the models trained on pure
recipes, read off every other model's mixing proportion, and correlate against
what the recipe says it was.  Directly interpretable ("the geometry thinks this
adapter is 71% topic 0, and it was trained at 75%"), but it needs those pure
anchors to exist, and it only sees the simplex.

**Procrustes** — build the ground-truth simplex as an actual point configuration
and superimpose each taxonomy's embedding onto it.  Needs no anchors and uses the
whole configuration, but reports a shape mismatch rather than a per-model
proportion.

They fail in different ways, so agreement between them is worth something and
disagreement is a finding.  Everything is reported per taxonomy so the levels are
directly comparable.

Identifiers
-----------
The model-level taxonomies are keyed by adapter path and the dataset-level one by
recipe ID.  Everything here works in the **recipe-ID namespace**:
:func:`build_taxonomy_artifacts` emits it directly (the mapping is recorded in
each adapter's ``experiment_meta.json``, so it is looked up rather than guessed),
and :func:`compare_taxonomies` applies
:func:`src.analysis.identity.recipe_id_for` by default, which is a no-op on
identifiers already in that namespace and the correct translation otherwise.

Honesty about the recovery number
---------------------------------
Anchors receive one-hot weights *by construction*, so a correlation computed over
them partly measures the projection's own definition rather than the geometry.
Every recovery statistic is therefore reported twice: over all models, and over
the non-anchor evaluation points alone.  The second is the one to quote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult
from src.core.protocols import ModelID

from .configurations import ProcrustesResult, ProtestResult, procrustes_compare, protest
from .discovery import CacheIndex
from .ground_truth import (
    evaluation_points,
    ground_truth_weights,
    pure_anchors,
    simplex_distance_matrix,
    simplex_geometry,
    truth_matrix,
)
from .identity import recipe_id_for, relabel
from .matrices import DcorResult, correlation_table, dcor_test, matrix_correlation
from .quality import kruskal_stress
from .simplex import (
    RecoveryResult,
    SimplexComparison,
    SimplexProjection,
    anchor_weight_vs_truth,
    barycentric,
    compare_simplices,
)

__all__ = [
    "build_taxonomy_artifacts",
    "TaxonomyComparison",
    "compare_taxonomies",
    "compare_all_slices",
]

_MODEL_LEVEL = ("structural", "functional", "behavioral")


# ── stage 1: get a DistanceMatrix per taxonomy ─────────────────────────────────

def build_taxonomy_artifacts(
    index: CacheIndex,
    taxonomy: str,
    metric: str = "cosine",
    cache_root: Path | str | None = None,
    n_components: Sequence[int] = (2,),
    label: str | None = None,
    slice_key: dict | None = None,
    layers: int | list[int] | None = None,
    projections: str | list[str] | None = None,
    embedder_hash: str | None = None,
    dataset_selector: dict | None = None,
    behavioral_selector: dict | None = None,
    functional_selector: dict | None = None,
    id_scheme: str = "recipe_id",
    mds_kwargs: Mapping[str, Any] | None = None,
    use_cache: bool = True,
) -> tuple[DistanceMatrix, dict[str, GeometryResult]]:
    """Distance matrix plus embeddings for one taxonomy over one collection.

    Reads from :class:`~src.cache.collection_cache.CollectionCache` when the same
    collection has been built before, and writes back on a miss — so repeating a
    comparison, or running one slice of a sweep after another, does not recompute
    what is already on disk.

    Representations and distances come from different places per taxonomy:

    ==================  ==========================================  ==========================
    taxonomy            representation                              distance
    ==================  ==========================================  ==========================
    structural          LoRA A/B factors from the adapter files     low-rank builders
    dataset_embedding   ``DatasetEmbeddingCache``                   ``src.metrics`` pairwise
    functional          ``ActivationCache``                         ``src.metrics`` pairwise
    behavioral          ``GeneratedTextCache``                      ``src.metrics`` pairwise
    ==================  ==========================================  ==========================

    The structural row is the one exception to using :mod:`src.metrics`.  Its
    builders in :mod:`src.notebook.structure` work directly on the rank-``r``
    factors and never form the ``d × d`` product ``B @ A`` — which for these
    adapters would be ~37 MB each — and they sidestep the variable-row-length
    limitation recorded in :mod:`src.metrics.cka` and :mod:`src.metrics.frobenius`.
    ``scripts/check_analysis.py::t_cosine_equivalence`` pins the two to identical
    numbers, so this is a cost and robustness choice, not a different metric.

    Parameters
    ----------
    n_components:
        Dimensions to embed at.  A simplex projection needs ``k-1``; a plot needs
        2.  Both are usually wanted, and the cache now keeps them apart.
    behavioral_selector:
        Which cached behavioral representations to read and how to view them:
        ``{"draw", "max_new_tokens", "replicates", "sampling_hash",
        "embedder_hash", "replicate_reduction", "view", "normalize",
        "representation", "renormalize"}``, all optional.  ``draw`` resolves to the
        one every model shares and the variant to the one present within it;
        several of either is an error naming the options, since two draws are
        different query sets, two embedders are different vector spaces, and two
        sampling settings are different generated text.

        ``replicate_reduction`` sits between addressing and pooling.  A stored
        matrix is ``(n_queries * replicates, d)`` in query-major order;
        ``"mean"`` averages each query's replicates back to ``(n_queries, d)``,
        while the default ``"all"`` keeps them, so a distance reflects the
        within-query spread instead of averaging it away first.  It is applied
        inside the cache, before ``normalize`` and before ``view``, because a
        ``gram`` of the unreduced matrix is a kernel over replicates and
        averaging *that* is a different quantity.

        The last two keys are a *read-time pooling* step applied after the cache
        hands back a representation, and they compose with the ones above rather
        than replacing them.  ``representation="matrix"`` (the default) is the
        tensor as addressed; ``"mean"`` pools it to a ``(1, d)``
        centroid, matching
        :class:`~src.taxonomy.dataset_embedding.DatasetEmbeddingTaxonomy`'s
        ``representation="mean"``.  ``renormalize`` L2-normalizes that centroid.
        It is deliberately a separate knob from ``normalize``, and deliberately
        not the same thing: ``normalize`` is the cache's own row-wise mode
        (``"layer" | "global" | "none"``, folded into the surrogate hash and
        applied *before* pooling), whereas ``renormalize`` acts *after* it.  The
        dataset level does not renormalize, so the default is ``False``; but rows
        here are already unit-norm, their mean is not, and leaving it that way
        puts a large constant offset into ``dot_product`` distances.

        Pooling is read-time only —
        :class:`~src.taxonomy.behavioral.BehavioralTaxonomy` still stores the full
        matrix either way — and ``representation="mean"`` is rejected for kernel
        views such as ``view="gram"``, whose rows are query similarities rather
        than features and whose mean is not a kernel.
    functional_selector:
        The same shape, for activations:
        ``{"draw", "mode", "pooling", "layers", "view", "normalize"}``.
        ``layers`` is a read-time choice — a run stores every layer — so changing
        it costs a surrogate build, not a GPU pass.

    use_cache:
        ``False`` skips ``06_collections`` entirely — nothing is read and nothing
        is written.  ``cache_root=None`` does **not** do this: it falls back to
        ``index.cache_root``, which a ``scan_cache`` index always carries, so a
        caller meaning "just compute" wrote to the shared cache anyway.  That
        fallback is load-bearing for callers that rely on inheriting the root, so
        the escape hatch is its own flag rather than a change of meaning.

    .. note::
       **The collection key sees the selectors, via what they resolve to.**
       A collection is keyed on ``{taxonomy}/{collection_key}/{metric}_{surrogate_key}``,
       where ``collection_key`` hashes each model's stored artifact path and
       ``surrogate_key`` hashes the per-model surrogate hashes.  So two calls
       differing in draw, embedder, view, normalization, pooling or replicate
       reduction land in different directories, and two calls that read the same
       tensors share one even if they spelled the selector differently.

       This was ``docs/notes/TODO.md`` item 14: the key used to be
       ``(ids, taxonomy, metric)``, so a collection built under one selector was
       returned unchanged for another, silently.  Entries written under the old
       key are not readable under the new one and were quarantined rather than
       migrated — their recorded provenance was too thin to rehash faithfully.
    id_scheme:
        ``"recipe_id"`` (default) keys the result by training recipe, which is the
        namespace every taxonomy shares.  Use ``"model_id"`` when a collection
        sweeps LoRA rank or init seed, since several adapters then share one
        recipe and the default would make the rows ambiguous.
    """
    if not len(index):
        raise ValueError(f"no models in the collection for taxonomy {taxonomy!r}")
    if id_scheme not in ("recipe_id", "model_id"):
        raise ValueError(f"id_scheme must be 'recipe_id' or 'model_id', got {id_scheme!r}")

    ids = _resolve_ids(index, id_scheme)
    cache = _collection_cache(cache_root or index.cache_root) if use_cache else None

    # Resolve first, then key.  The representations have to be loaded either way,
    # and only once they are do we know which artifact and which view each model
    # actually resolved to — which is what the collection is keyed on.  The cache
    # then saves the pairwise computation, which is the expensive part.
    reps, identity = _resolve_representations(
        index, taxonomy,
        layers=layers, projections=projections, embedder_hash=embedder_hash,
        dataset_selector=dataset_selector,
        behavioral_selector=behavioral_selector,
        functional_selector=functional_selector,
    )
    model_entries = [{**ident, "model_id": i} for i, ident in zip(ids, identity)]

    dm: DistanceMatrix | None = None
    handle: str | None = None
    if cache is not None:
        # The metric name is resolved once here.  Looking up under the caller's
        # spelling ("cka") while saving under the metric's own ("cka_linear") meant
        # such a collection was stored where it was never sought, so it never hit
        # and rewrote its directory on every run.
        metric_name = _resolve_metric(metric).metric_name
        handle = cache.handle(
            taxonomy,
            cache.collection_key(model_entries),
            metric_name,
            cache.surrogate_key([e["surrogate_hash"] for e in model_entries]),
        )
        if cache.exists(handle):
            dm = cache.load_distance_matrix(handle)

    if dm is None:
        dm = _distances(index, taxonomy, metric, ids, reps, layers, projections)
        if cache is not None:
            cache.save_distance_matrix(
                dm,
                handle,
                model_entries=model_entries,
                label=label,
                slice_key=slice_key,
                config=_leaf_config(
                    taxonomy, reps, model_entries,
                    layers=layers, projections=projections,
                    embedder_hash=embedder_hash,
                    dataset_selector=dataset_selector,
                    behavioral_selector=behavioral_selector,
                    functional_selector=functional_selector,
                ),
            )

    geometries = _fit_geometries(dm, n_components, cache, handle, mds_kwargs)
    return dm, geometries


def _leaf_config(taxonomy, reps, model_entries, **selectors) -> dict:
    """What the leaf ``config.json`` records, so a collection stays traceable.

    The directory name is a digest of the per-model surrogate hashes, so the list
    itself has to be written down: it is the only way back from a collection to
    the exact tensors it was built from.
    """
    return {
        "taxonomy": taxonomy,
        "selectors": {k: v for k, v in selectors.items() if v is not None},
        "representations": [
            {
                "model_id": e["model_id"],
                "artifact_path": e["artifact_path"],
                "surrogate_hash": e["surrogate_hash"],
            }
            for e in model_entries
        ],
        "resolved": (reps[0].metadata if reps else {}) and {
            k: v
            for k, v in (reps[0].metadata or {}).items()
            # The shared read-time view.  Per-model values (paths, hashes, the
            # generated text itself) are listed above or are far too large.
            if k in ("mode", "pooling", "layers", "view", "normalize",
                     "replicate_reduction", "embedder_hash", "max_new_tokens",
                     "replicates", "sampling_hash", "is_kernel")
        },
    }


def _resolve_ids(index: CacheIndex, id_scheme: str) -> list[ModelID]:
    ids = [getattr(e, id_scheme) or e.model_id for e in index.entries]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"id_scheme={id_scheme!r} maps distinct models onto the same "
            f"identifier: {duplicated}. This happens when a collection sweeps a "
            "parameter the identifier does not record — LoRA rank or init seed, "
            "typically. Pass id_scheme='model_id' to keep them apart."
        )
    return ids


def _metric_name(metric: Any) -> str:
    return metric if isinstance(metric, str) else metric.metric_name


def _resolve_metric(metric: Any):
    """Name → metric instance.

    Mirrors ``scripts/_utils.make_metric``, but resolved here so that
    :mod:`src` never imports from ``scripts``.
    """
    if not isinstance(metric, str):
        return metric

    from src.metrics import (
        BuresWassersteinDistanceMetric,
        CKADistanceMetric,
        CosineDistanceMetric,
        DotProductDistanceMetric,
        FrobeniusDistanceMetric,
    )

    table = {
        "cka": CKADistanceMetric,
        "frobenius": FrobeniusDistanceMetric,
        "cosine": CosineDistanceMetric,
        "dot_product": DotProductDistanceMetric,
        # Selectable on the representation path as well as the structural one.
        # `_structural_matrix` gates on the *reported* name, which is the same
        # string, so a structural comparison still routes to the low-rank builder
        # in `src.notebook.structure` rather than through this class.
        "bures_wasserstein": BuresWassersteinDistanceMetric,
    }
    if metric in table:
        return table[metric]()

    # A metric answers to two names: the token you select it by ("cka") and the
    # one it reports ("cka_linear"), which is what gets stored on a DistanceMatrix.
    # Accepting both is what lets a stored matrix be re-resolved from its own
    # record. Not doing so is how the collection cache came to look collections up
    # under one spelling and save them under the other, so that they never hit.
    by_reported = {cls().metric_name: cls for cls in table.values()}
    if metric in by_reported:
        return by_reported[metric]()

    raise ValueError(
        f"unknown metric {metric!r}. Choose from {sorted(table)} "
        f"(or a reported name: {sorted(by_reported)})."
    )


def _collection_cache(cache_root: Path | str | None):
    if cache_root is None:
        return None
    from src.cache.collection_cache import CollectionCache

    return CollectionCache(cache_root)


def _fit_geometries(
    dm: DistanceMatrix,
    n_components: Sequence[int],
    cache,
    chash: str | None,
    mds_kwargs: Mapping[str, Any] | None,
) -> dict[str, GeometryResult]:
    """Embed at each requested dimension, using the cache where possible."""
    from .bridge import fit_geometry

    kwargs = dict(mds_kwargs or {})
    kwargs.setdefault("random_state", 0)

    out: dict[str, GeometryResult] = {}
    for n in sorted({int(n) for n in n_components}):
        if n >= len(dm.model_ids):
            # MDS cannot place n models in n or more dimensions meaningfully.
            continue
        key = f"mds_{n}d"
        geo = None
        if cache is not None and chash is not None:
            try:
                geo = cache.load_geometry(chash, "mds", n, mds_kwargs=kwargs)
            except (FileNotFoundError, ValueError, KeyError):
                geo = None
        if geo is None:
            geo = fit_geometry(dm, method="mds", n_components=n, **kwargs)
            if cache is not None and chash is not None:
                cache.save_geometry(chash, geo, mds_kwargs=kwargs)
        out[key] = geo
    if not out:
        raise ValueError(
            f"could not embed {len(dm.model_ids)} models at any of "
            f"{sorted(n_components)} dimensions"
        )
    return out


def _compute_distance_matrix(
    index: CacheIndex,
    taxonomy: str,
    metric: Any,
    ids: Sequence[ModelID],
    layers=None,
    projections=None,
    embedder_hash: str | None = None,
    dataset_selector: dict | None = None,
    behavioral_selector: dict | None = None,
    functional_selector: dict | None = None,
) -> DistanceMatrix:
    """Distances for one taxonomy, always recomputed.

    The selector-faithful path: it honours every selector on every call and never
    reads or writes ``06_collections``.  :func:`build_taxonomy_artifacts` goes
    through the cache instead; this is what to call to sweep selectors without
    populating it.
    """
    reps, _ = _resolve_representations(
        index, taxonomy, layers=layers, projections=projections,
        embedder_hash=embedder_hash, dataset_selector=dataset_selector,
        behavioral_selector=behavioral_selector,
        functional_selector=functional_selector,
    )
    return _distances(index, taxonomy, metric, ids, reps, layers, projections)


#: Selector keys per taxonomy, for resolving one model's representation.
_TAXONOMIES = ("structural", "dataset_embedding", "functional", "behavioral")


def _resolve_representations(
    index: CacheIndex,
    taxonomy: str,
    *,
    layers=None,
    projections=None,
    embedder_hash: str | None = None,
    dataset_selector: dict | None = None,
    behavioral_selector: dict | None = None,
    functional_selector: dict | None = None,
) -> tuple[list | None, list[dict]]:
    """Resolve what each model's representation *is*, before any distance is computed.

    Returns ``(reps, identity)``.  ``identity`` is one dict per model in
    ``index.entries`` order, carrying the ``artifact_path`` and
    ``surrogate_hash`` that key the collection — this is the ground truth the
    collection cache is keyed on, rather than the caller's selector, which may be
    underspecified (``{}`` and ``{"draw": ...}`` are the same read).

    ``reps`` is ``None`` for **structural** only.  Structural gets the same
    resolution seam and the same identity shape as the other three, but its
    distances still come from the low-rank builders in
    :mod:`src.notebook.structure`, which read the LoRA factors themselves and
    never form ``B @ A``.  That is the one remaining special case, and it closes
    with ``docs/notes/TODO.md`` item 8: once :mod:`src.metrics` handles
    variable-length representation rows, structural returns real representations
    here and only :func:`_distances` changes.
    """
    if taxonomy == "structural":
        return None, _structural_identity(index, layers, projections)
    if taxonomy == "dataset_embedding":
        reps = _dataset_embedding_reps(index, embedder_hash, dataset_selector)
    elif taxonomy == "behavioral":
        reps = _behavioral_reps(index, behavioral_selector)
    elif taxonomy == "functional":
        reps = _functional_reps(index, functional_selector)
    else:
        raise ValueError(
            f"unknown taxonomy {taxonomy!r}. Choose from {', '.join(_TAXONOMIES)}."
        )
    return reps, _identity_from_reps(index, reps, taxonomy)


def _identity_from_reps(
    index: CacheIndex, reps: Sequence, taxonomy: str
) -> list[dict]:
    """One identity dict per model, read off what the caches actually resolved."""
    out = []
    for entry, rep in zip(index.entries, reps):
        meta = rep.metadata or {}
        path = meta.get("artifact_path")
        if path is None:
            raise ValueError(
                f"the {taxonomy} cache returned a representation for "
                f"{entry.model_id!r} with no 'artifact_path' in its metadata, so "
                "the collection it belongs to cannot be keyed on what it was "
                "built from. Every cache's load() must surface artifact_path and "
                "surrogate_hash — see TODO.md item 14."
            )
        out.append(
            {
                "model_id": None,  # filled in by the caller, which knows id_scheme
                "entry_type": "lora_adapter" if entry.adapter_dir else "recipe",
                "adapter_name": entry.adapter_name,
                "recipe_hash": entry.recipe_hash,
                "artifact_path": path,
                "surrogate_hash": meta.get("surrogate_hash"),
            }
        )
    return out


def _structural_identity(index: CacheIndex, layers, projections) -> list[dict]:
    """Identity for the structural level, from the adapter files it reads.

    Structural has no surrogate, so the read-time choice (*layers*,
    *projections*) plays that role: it is what two structural collections over
    the same adapters can differ by.
    """
    from src.cache._draw_keyed import DrawKeyedCache

    view = DrawKeyedCache.config_hash(
        {
            "kind": "structural_view",
            "layers": layers if layers is not None else "last",
            "projections": projections if projections is not None else "o",
        }
    )
    return [
        {
            "model_id": None,
            "entry_type": "lora_adapter" if entry.adapter_dir else "recipe",
            "adapter_name": entry.adapter_name,
            "recipe_hash": entry.recipe_hash,
            "artifact_path": f"03_adapters/{entry.adapter_name}",
            "surrogate_hash": view,
        }
        for entry in index.entries
    ]


def _distances(
    index: CacheIndex,
    taxonomy: str,
    metric: Any,
    ids: Sequence[ModelID],
    reps: Sequence | None,
    layers=None,
    projections=None,
) -> DistanceMatrix:
    """Pairwise distances from resolved representations."""
    if reps is None:
        return _structural_matrix(index, metric, ids, layers, projections)

    metric_obj = _resolve_metric(metric)
    n = len(reps)
    arr = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            arr[i, j] = arr[j, i] = metric_obj.compute(reps[i], reps[j])

    return DistanceMatrix(
        matrix=arr,
        model_ids=list(ids),
        metric=metric_obj.metric_name,
        taxonomy=taxonomy,
    )


def _structural_matrix(
    index: CacheIndex, metric: Any, ids: Sequence[ModelID], layers, projections
) -> DistanceMatrix:
    """Low-rank distances over the LoRA factors, relabelled into the shared namespace."""
    from src.notebook.lora_weights import load_lora_weights

    from .bridge import lora_distance_matrix

    missing = [e.adapter_name for e in index.entries if not e.has("structural_weights")]
    if missing:
        raise ValueError(
            f"{len(missing)} adapter(s) have no adapter_model.safetensors, e.g. "
            f"{missing[0]}. Filter with index.with_available('structural_weights')."
        )

    root = index.cache_root
    if root is None:
        raise ValueError("index has no cache_root; cannot locate adapter weights")

    weights = load_lora_weights(
        [e.adapter_name for e in index.entries],
        adapter_root=Path(root) / "03_adapters",
        layer_indices=layers if layers is not None else "last",
        projections=projections if projections is not None else "o",
    )
    kind = _metric_name(metric)
    if kind not in ("cosine", "frobenius", "bures_wasserstein", "cka"):
        raise ValueError(
            f"structural distances support cosine, frobenius, bures_wasserstein "
            f"and cka; got {kind!r}"
        )
    dm = lora_distance_matrix(weights, kind=kind, layers=layers, projections=projections)

    # lora_distance_matrix keys by adapter folder name; map onto the shared
    # namespace using the authoritative record rather than by parsing the name.
    rename = {e.adapter_name: i for e, i in zip(index.entries, ids)}
    return relabel(dm, rename)


def _dataset_embedding_reps(
    index: CacheIndex,
    embedder_hash: str | None,
    dataset_selector: dict | None = None,
) -> list:
    """Resolve each model's cached dataset embedding.

    Returns the representations rather than a distance matrix: the collection
    key is built from what these resolved to, so resolution has to happen before
    the cache is consulted (``docs/notes/TODO.md`` item 14).
    """
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

    root = index.cache_root
    if root is None:
        raise ValueError("index has no cache_root; cannot locate dataset embeddings")

    cache = DatasetEmbeddingCache(root)
    sel = dict(dataset_selector or {})
    if embedder_hash and not sel.get("embedder_hash"):
        sel["embedder_hash"] = embedder_hash

    # One embedder hash now serves every draw, because the hash no longer bundles
    # n_samples: agreeing on it is agreeing on the embedder, not on a sample
    # count.  Only the representation still has to be chosen.
    chosen = sel.get("embedder_hash") or _embedder_choice(cache, index)
    spec = DatasetEmbeddingCache.spec_for(sel.get("representation", "mean"))

    reps = []
    for entry in index.entries:
        if not entry.recipe_hash:
            raise ValueError(f"{entry.adapter_name} has no recipe_hash to look up")
        if not cache.exists(
            entry.recipe_hash, entry.n_samples, entry.seed, chosen, spec
        ):
            raise ValueError(
                f"no dataset embedding for recipe {entry.recipe_id!r} at draw "
                f"n{entry.n_samples}_s{entry.seed} under embedder {chosen} and "
                f"representation {spec['representation']!r}. Embedders stored for "
                f"that draw: "
                f"{cache.list_embedder_hashes(entry.recipe_hash, entry.n_samples, entry.seed)}"
            )
        reps.append(
            cache.load(entry.recipe_hash, entry.n_samples, entry.seed, chosen, spec)
        )

    return reps


def _behavioral_reps(
    index: CacheIndex, selector: dict | None
) -> list:
    """Resolve each model's cached behavioral representation.

    Mirrors :func:`_functional_reps`, because since the re-key the two levels
    are addressed identically.  A selector names the draw and the variant, then
    optionally pools what comes back::

        {"draw": {"recipe_hash": ..., "n_samples": 64, "seed": 0},
         "max_new_tokens": 128, "replicates": 8, "sampling_hash": ...,
         "embedder_hash": ...,
         "replicate_reduction": "all",            # all | mean
         "view": "matrix", "normalize": "none",   # matrix | gram
         "representation": "matrix", "renormalize": False}   # matrix | mean

    Every field has a default.  ``draw=None`` resolves to the one draw every
    model shares, and any unnamed part of ``(max_new_tokens, replicates,
    sampling_hash, embedder_hash)`` to the one variant present within it — both
    with a "several exist, name one" error rather than a silent pick, since two
    draws are different query sets, two embedders are different vector spaces,
    and two sampling settings are different text.

    ``replicate_reduction`` is the odd one out: it does not select a stored
    entry, it averages the replicates of each query at read time, giving back the
    ``(n_queries, d)`` shape a single-sample run produced.  Leaving it at
    ``"all"`` compares ``(n_queries * replicates, d)`` matrices, where a distance
    includes the within-query spread rather than averaging it away first.

    **This replaced a single ``config_hash``**, which selected a whole run at
    once.  The disambiguation did not disappear with it; it split in two — which
    draw, and which variant within the draw — and gained the ability to say "the
    same queries, re-embedded" instead of treating that as an unrelated run.

    The last two fields are a separate, later step: the cache addresses and views
    a representation, then :func:`_behavioral_view` pools it.  See
    :func:`build_taxonomy_artifacts` for why ``renormalize`` is not ``normalize``.
    """
    from src.cache.generated_text_cache import GeneratedTextCache

    root = index.cache_root
    if root is None:
        raise ValueError("index has no cache_root; cannot locate behavioral representations")

    cache = GeneratedTextCache(root)
    sel = dict(selector or {})
    unknown = set(sel) - {
        "draw", "max_new_tokens", "replicates", "sampling_hash", "embedder_hash",
        "replicate_reduction", "view", "normalize",
        "representation", "renormalize",
    }
    if unknown:
        raise ValueError(
            f"unknown behavioral_selector key(s) {sorted(unknown)}. Choose from "
            "'draw', 'max_new_tokens', 'replicates', 'sampling_hash', "
            "'embedder_hash', 'replicate_reduction', 'view', 'normalize', "
            "'representation', 'renormalize'."
        )
    view = sel.get("view", "matrix")
    normalize = sel.get("normalize", "none")
    replicate_reduction = sel.get("replicate_reduction", "all")

    draw = sel.get("draw")
    if draw is None:
        draw = _draw_choice(cache, index, "behavioral", "behavioral")

    max_new_tokens = sel.get("max_new_tokens")
    replicates = sel.get("replicates")
    sampling_hash = sel.get("sampling_hash")
    embedder_hash = sel.get("embedder_hash")
    if None in (max_new_tokens, replicates, sampling_hash, embedder_hash):
        max_new_tokens, replicates, sampling_hash, embedder_hash = (
            _behavioral_variant_choice(
                cache, index, draw, max_new_tokens, replicates,
                sampling_hash, embedder_hash,
            )
        )

    reps = []
    for entry in index.entries:
        base_id = entry.base_model_id
        if not base_id:
            raise ValueError(
                f"{entry.adapter_name} has no base_model_id; behavioral representations "
                "are keyed by (base model, adapter) and cannot be located without it"
            )
        try:
            rep = cache.load(
                base_id, entry.model_id, draw,
                max_new_tokens=max_new_tokens,
                replicates=replicates,
                sampling_hash=sampling_hash,
                embedder_hash=embedder_hash,
                view=view, normalize=normalize,
                replicate_reduction=replicate_reduction,
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"no behavioral representation for {entry.adapter_name!r} under draw "
                f"{GeneratedTextCache.draw_name(draw)} "
                f"({cache.variant_token(max_new_tokens, replicates, sampling_hash)}, "
                f"embedder {embedder_hash}): {e}. "
                "Filter with index.with_available('behavioral_repr')."
            ) from None
        reps.append(_behavioral_view(rep, sel, view=view))

    return reps


_BEHAVIORAL_REPRESENTATIONS = ("matrix", "mean")


def _behavioral_view(rep, selector: dict, view: str = "matrix"):
    """Pool one loaded representation, per the selector's read-time keys.

    Runs *after* :class:`GeneratedTextCache` has already addressed and viewed the
    representation, so it reads only ``representation`` and ``renormalize``; the
    addressing keys are the caller's business and are validated there.  Kept out
    of the cache deliberately: what is on disk stays the superset, and choosing a
    pooling costs no re-extraction.

    *view* is the cache view the matrix came back as, needed only to refuse to
    pool a kernel.
    """
    representation = selector.get("representation", "matrix")
    if representation not in _BEHAVIORAL_REPRESENTATIONS:
        raise ValueError(
            f"unknown behavioral representation {representation!r}. "
            f"Choose from {list(_BEHAVIORAL_REPRESENTATIONS)}."
        )
    renormalize = bool(selector.get("renormalize", False))

    if representation == "matrix" and not renormalize:
        return rep

    from src.cache.generated_text_cache import GeneratedTextCache

    if representation == "mean" and view in GeneratedTextCache.KERNEL_VIEWS:
        # A gram's rows are query-to-query similarities, not features: their mean
        # is a row of column sums, and the (1, n) result is not a kernel at all.
        raise ValueError(
            f"representation='mean' is meaningless for kernel view {view!r}: its "
            "rows are query similarities, not features. Use view='matrix' to pool, "
            "or drop representation='mean' to compare grams."
        )

    from dataclasses import replace

    matrix = rep.matrix.astype(np.float64)
    if representation == "mean":
        # keepdims so the result stays 2-D, matching DatasetEmbeddingTaxonomy's
        # (1, d) centroid rather than collapsing to a bare vector.
        matrix = matrix.mean(axis=0, keepdims=True)
    if renormalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # A zero row cannot be given a direction; leave it at zero rather than
        # inventing one, and let the metric decide what that means.
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 1e-10)

    return replace(
        rep,
        matrix=matrix.astype(np.float32),
        metadata={
            **rep.metadata,
            "representation": representation,
            "renormalize": renormalize,
        },
    )


def _functional_reps(
    index: CacheIndex, selector: dict | None
) -> list:
    """Resolve each model's cached activations.

    Unlike behavioral, functional is keyed by *model* rather than by run, so the
    choice is not a single config hash.  A selector names the draw and the view::

        {"draw": {"recipe_hash": ..., "n_samples": 64, "seed": 0},
         "mode": "input", "pooling": "mean", "layers": None,
         "view": "concat", "normalize": "layer"}   # layer | global | none

    Every field has a default, and ``draw=None`` resolves to the one draw present
    when there is exactly one — with the same "several exist, name one" error
    behavioral raises, since two draws are different query sets and comparing
    across them is meaningless rather than merely imprecise.
    """
    from src.cache.activation_cache import ActivationCache

    root = index.cache_root
    if root is None:
        raise ValueError("index has no cache_root; cannot locate activations")

    cache = ActivationCache(root)
    sel = dict(selector or {})
    unknown = set(sel) - {
        "draw", "mode", "pooling", "layers", "view", "normalize", "max_new_tokens",
    }
    if unknown:
        # Mirrors the behavioral check below.  Without it a misspelled key —
        # "normalise", or "layer" for "layers" — silently fell through to the
        # default, so you measured all 29 layers concatenated while believing you
        # had asked for one.  That now also produces a *correct* cache key for the
        # default view, which would make the wrong answer permanent.
        raise ValueError(
            f"unknown functional_selector key(s) {sorted(unknown)}. Choose from "
            "'draw', 'mode', 'pooling', 'layers', 'view', 'normalize', "
            "'max_new_tokens'."
        )
    mode = sel.get("mode", "input")
    pooling = sel.get("pooling", "mean")
    view = sel.get("view", "concat")
    normalize = sel.get("normalize", "layer")
    layers = sel.get("layers")
    max_new_tokens = sel.get("max_new_tokens")

    draw = sel.get("draw")
    if draw is None:
        draw = _draw_choice(cache, index, "functional", "functional")

    reps = []
    for entry in index.entries:
        base_id = entry.base_model_id
        if not base_id:
            raise ValueError(
                f"{entry.adapter_name} has no base_model_id; activations are keyed "
                "by (base model, adapter) and cannot be located without it"
            )
        try:
            reps.append(
                cache.load(
                    base_id, entry.model_id, draw,
                    mode=mode, pooling=pooling, layers=layers,
                    view=view, normalize=normalize, max_new_tokens=max_new_tokens,
                )
            )
        except FileNotFoundError as e:
            raise ValueError(
                f"no functional representation for {entry.adapter_name!r} under draw "
                f"{ActivationCache.draw_name(draw)} ({mode}/{pooling}): {e}. "
                "Filter with index.with_available('functional_repr')."
            ) from None

    return reps


def _draw_choice(cache, index: CacheIndex, taxonomy: str, extract_flag: str) -> dict:
    """The single draw every entry has a representation under, or an error naming the options.

    Serves both inference levels: since ``05_generated`` was re-keyed to match
    ``04_activations``, ``list_draws`` means the same thing on both caches, and
    "which query draw do all these models share?" is one question asked of a
    different tree.
    """
    per_entry = []
    for entry in index.entries:
        if not entry.base_model_id:
            continue
        draws = cache.list_draws(entry.base_model_id, entry.model_id)
        per_entry.append({(d["recipe_hash"], d["n_samples"], d["seed"]) for d in draws})

    shared = set.intersection(*per_entry) if per_entry else set()
    if not shared:
        raise ValueError(
            f"no single query draw has {taxonomy} representations for every model in "
            f"this collection. Run scripts/extract_reprs.py --taxonomy {extract_flag}, "
            f"or pass {taxonomy}_selector={{'draw': {{...}}}} to name one."
        )
    if len(shared) > 1:
        listed = sorted(f"{r}/n{n}_s{s:02d}" for r, n, s in shared)
        raise ValueError(
            f"{len(listed)} query draws are cached for every model: {listed}. "
            "Different draws are different query sets and are not comparable, so "
            f"pass {taxonomy}_selector={{'draw': {{...}}}} to choose one."
        )
    recipe_hash, n_samples, seed = next(iter(shared))
    return {"recipe_hash": recipe_hash, "n_samples": n_samples, "seed": seed}


def _behavioral_variant_choice(
    cache, index: CacheIndex, draw: dict,
    max_new_tokens: int | None, replicates: int | None,
    sampling_hash: str | None, embedder_hash: str | None,
) -> tuple[int, int, str, str]:
    """The one variant every entry shares under *draw*.

    A variant is ``(max_new_tokens, replicates, sampling_hash, embedder_hash)``:
    the four axes that make two stored generations different numbers rather than
    the same numbers addressed twice.

    Modelled on :func:`_embedder_choice`, and like it this is a **directory
    listing, not a file open** — the variant is entirely in the filename, so
    answering "what has been computed here?" costs one ``glob`` per model rather
    than one safetensors load.  It parses the filename through the cache's own
    :data:`~src.cache.generated_text_cache._GEN_RE` rather than slicing the
    string, which is what an earlier version did and what stopped working the
    moment the stem grew a second component.

    Partial selectors are honoured: naming only ``embedder_hash`` narrows the
    candidates and then requires the rest to be unambiguous, which is what makes
    "the same draw, re-embedded" expressible.
    """
    wanted = (max_new_tokens, replicates, sampling_hash, embedder_hash)
    per_entry = []
    for entry in index.entries:
        if not entry.base_model_id:
            continue
        variants = set()
        for mode_token, reps, samp, emb in cache.list_variants(
            entry.base_model_id, entry.model_id, draw
        ):
            if not mode_token.startswith("generation"):
                continue
            candidate = (int(mode_token[len("generation"):]), reps, samp, emb)
            if all(w is None or w == c for w, c in zip(wanted, candidate)):
                variants.add(candidate)
        per_entry.append(variants)

    shared = set.intersection(*per_entry) if per_entry else set()
    if not shared:
        raise ValueError(
            "no single (max_new_tokens, replicates, sampling, embedder) variant "
            "has behavioral representations for every model under draw "
            f"{cache.draw_name(draw)}. Run scripts/extract_reprs.py --taxonomy "
            "behavioral, or name one with behavioral_selector="
            "{'max_new_tokens': ..., 'replicates': ..., 'sampling_hash': ..., "
            "'embedder_hash': ...}."
        )
    if len(shared) > 1:
        listed = sorted(
            f"generation{m}_{r}r_{s}_{e}" for m, r, s, e in shared
        )
        raise ValueError(
            f"{len(listed)} behavioral variants are cached for every model: {listed}. "
            "They differ in generation length, replicate count, sampling settings "
            "or embedder and are not comparable, so pass behavioral_selector="
            "{'max_new_tokens': ..., 'replicates': ..., 'sampling_hash': ..., "
            "'embedder_hash': ...} to choose one."
        )
    return next(iter(shared))


def _draw_key(entry) -> tuple:
    """What identifies the dataset embedding an entry needs.

    Not the recipe hash alone.  That hash is content-addressed, so every n and seed of
    one mixture shares it; keying on it would hand two seeds the same embedding and
    quietly collapse a seed sweep to a single point.
    """
    return (entry.recipe_hash, entry.n_samples, entry.seed)


def _embedder_choice(cache, index: CacheIndex) -> str:
    """The one embedder hash every entry in this collection has, as a string.

    This used to return a hash *per draw* and take fifty lines to do it, because
    ``embedder_hash`` bundled ``n_samples`` and ``seed``: requiring one shared
    hash would have required one shared sample count, which rules out exactly the
    pooled comparison that spans the sample-size sweep.  The workaround was to
    agree on an ``(embedder_config, representation)`` signature instead and let
    each recipe contribute its own hash.

    Item 15 removed the reason.  The hash now keys the embedder alone, so one
    hash genuinely is common to every draw and the signature bookkeeping
    collapses into a set intersection over hashes.

    The two errors below are kept.  They stop firing for the sweep case they were
    written to work around, but they remain reachable — and necessary — the
    moment a second embedder exists in the cache.
    """
    available: list[set[str]] = []
    for entry in index.entries:
        if not entry.recipe_hash:
            raise ValueError(f"{entry.adapter_name} has no recipe_hash to look up")
        hashes = set(
            cache.list_embedder_hashes(
                entry.recipe_hash, entry.n_samples, entry.seed
            )
        )
        if not hashes:
            raise ValueError(
                f"no cached dataset embedding for recipe {entry.recipe_id!r} at "
                f"draw n{entry.n_samples}_s{entry.seed}"
            )
        available.append(hashes)

    common = set.intersection(*available)
    if not common:
        raise ValueError(
            "the models in this collection share no embedder configuration, so "
            "their dataset embeddings cannot be compared on one axis. Embed them "
            "all with the same embedder first."
        )
    if len(common) > 1:
        raise ValueError(
            f"{len(common)} embedder configurations are available for every "
            f"model: {sorted(common)}. Pass dataset_selector={{'embedder_hash': ...}} "
            "to choose."
        )
    return next(iter(common))


# ── stage 2: the comparison ────────────────────────────────────────────────────

@dataclass
class TaxonomyComparison:
    """Every comparison for one collection: taxonomy vs. taxonomy, and vs. the truth."""

    slice_key: dict
    model_ids: list[ModelID]
    vertices: list[str]
    anchors: list[ModelID]
    eval_points: list[ModelID]
    taxonomies: list[str]

    distance_matrices: dict[str, DistanceMatrix] = field(default_factory=dict)
    geometries: dict[str, dict[str, GeometryResult]] = field(default_factory=dict)
    ground_truth: GeometryResult | None = None
    ground_truth_matrix: DistanceMatrix | None = None
    ground_truth_weights: np.ndarray | None = None

    #: Dimension of the ground-truth simplex, ``k-1``.  This is the *minimum* an
    #: embedding must have for :func:`~src.analysis.simplex.barycentric` to work.
    simplex_dim: int = 0
    #: Dimension the barycentric projection and the Procrustes comparison actually
    #: used.  Deliberately allowed to exceed :attr:`simplex_dim`: projecting from
    #: exactly ``k-1`` makes the anchors' affine hull the whole space, so every
    #: residual is identically zero and the "how far off the simplex is this
    #: model?" diagnostic is lost.
    projection_dim: int = 0

    stress: dict[str, float] = field(default_factory=dict)
    projections: dict[str, SimplexProjection] = field(default_factory=dict)
    recovery: dict[str, RecoveryResult] = field(default_factory=dict)
    recovery_eval_only: dict[str, RecoveryResult] = field(default_factory=dict)
    edge_deviation: dict[str, np.ndarray] = field(default_factory=dict)
    procrustes_vs_truth: dict[str, ProcrustesResult] = field(default_factory=dict)
    protest_vs_truth: dict[str, ProtestResult] = field(default_factory=dict)
    matrix_corr_vs_truth: dict[str, float] = field(default_factory=dict)
    #: Distance correlation against the ground-truth matrix, with an exact
    #: permutation p-value.  Unlike ``matrix_corr_vs_truth`` this is a
    #: calibrated test, and unlike ``protest_vs_truth`` it needs no embedding.
    dcor_vs_truth: dict[str, DcorResult] = field(default_factory=dict)

    pairwise_procrustes: dict[tuple[str, str], ProcrustesResult] = field(default_factory=dict)
    pairwise_simplex: dict[tuple[str, str], SimplexComparison] = field(default_factory=dict)
    matrix_correlations: tuple[list[str], np.ndarray] | None = None

    #: Populated by :meth:`to_report`, and the only form :meth:`load` restores the
    #: statistics in — the heavyweight result objects are not re-serialised.
    report: dict = field(default_factory=dict)

    # ── summarising ───────────────────────────────────────────────────────────

    def to_report(self) -> dict:
        """JSON-serialisable summary — the durable form of this comparison."""
        labels, table = self.matrix_correlations or ([], np.zeros((0, 0)))
        return {
            "schema_version": "1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "slice": self.slice_key,
            "n_models": len(self.model_ids),
            "model_ids": list(self.model_ids),
            "vertices": list(self.vertices),
            "anchors": list(self.anchors),
            "eval_points": list(self.eval_points),
            "taxonomies": list(self.taxonomies),
            "simplex_dim": self.simplex_dim,
            "projection_dim": self.projection_dim,
            "ground_truth_weights": {
                m: [float(v) for v in w]
                for m, w in zip(self.model_ids, self.ground_truth_weights)
            }
            if self.ground_truth_weights is not None
            else {},
            "per_taxonomy": {
                t: {
                    "stress": _f(self.stress.get(t)),
                    "recovery_all": _recovery_dict(self.recovery.get(t)),
                    "recovery_eval_only": _recovery_dict(self.recovery_eval_only.get(t)),
                    "edge_deviation": {
                        m: float(v)
                        for m, v in zip(self.eval_points, self.edge_deviation.get(t, []))
                    },
                    "procrustes_vs_truth": _f(
                        getattr(self.procrustes_vs_truth.get(t), "disparity", None)
                    ),
                    # The one scalar of the fitted map worth having in JSON; the
                    # rotation and centroids live in procrustes/vs_truth/{t}/.
                    "procrustes_scale": _f(
                        getattr(self.procrustes_vs_truth.get(t), "scale", None)
                    ),
                    "protest_p_value": _f(
                        getattr(self.protest_vs_truth.get(t), "p_value", None)
                    ),
                    "matrix_corr_vs_truth": _f(self.matrix_corr_vs_truth.get(t)),
                    "dcor_vs_truth": _f(
                        getattr(self.dcor_vs_truth.get(t), "statistic", None)
                    ),
                    "dcor_p_value": _f(
                        getattr(self.dcor_vs_truth.get(t), "p_value", None)
                    ),
                }
                for t in self.taxonomies
            },
            "pairwise": {
                f"{a}|{b}": {
                    "procrustes_disparity": _f(
                        getattr(self.pairwise_procrustes.get((a, b)), "disparity", None)
                    ),
                    "mean_total_variation": _f(
                        getattr(
                            self.pairwise_simplex.get((a, b)), "mean_total_variation", None
                        )
                    ),
                }
                for (a, b) in self.pairwise_procrustes
            },
            "matrix_correlations": {
                "labels": list(labels),
                "table": [[_f(v) for v in row] for row in np.asarray(table)],
            },
        }

    def to_markdown(self) -> str:
        """Human-readable summary of the same numbers."""
        rep = self.report or self.to_report()
        slice_txt = (
            ", ".join(f"{k}={v}" for k, v in self.slice_key.items()) or "pooled"
        )
        lines = [
            f"# Taxonomy comparison — {slice_txt}",
            "",
            f"- **models**: {len(self.model_ids)}",
            f"- **simplex vertices** ({len(self.vertices)}): {', '.join(self.vertices)}",
            f"- **anchors**: {', '.join(self.anchors)}",
            f"- **evaluation points**: {', '.join(self.eval_points) or '(none)'}",
            f"- **ground-truth simplex dimension**: {self.simplex_dim}; "
            f"**projected from**: {self.projection_dim}-D MDS",
            "",
            "## Each taxonomy vs. the recipe ground truth",
            "",
            "Recovery is quoted over the evaluation points only — anchors are one-hot",
            "by construction, so including them measures the projection, not the geometry.",
            "",
            "| taxonomy | stress | recovery r | recovery rho | mean L1 | max residual "
            "| Procrustes | PROTEST p | matrix corr | dCor* | dCor p |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in self.taxonomies:
            r = rep["per_taxonomy"][t]
            e = r["recovery_eval_only"] or {}
            lines.append(
                f"| {t} | {_fmt(r['stress'])} | {_fmt(e.get('pearson_mean'))} | "
                f"{_fmt(e.get('spearman_mean'))} | {_fmt(e.get('mean_l1'))} | "
                f"{_fmt(e.get('max_residual'))} | {_fmt(r['procrustes_vs_truth'])} | "
                f"{_fmt(r['protest_p_value'])} | {_fmt(r['matrix_corr_vs_truth'])} | "
                f"{_fmt(r['dcor_vs_truth'])} | {_fmt(r['dcor_p_value'])} |"
            )

        if self.eval_points:
            lines += ["", "## Where each mixture landed", ""]
            header = "| model | truth | " + " | ".join(self.taxonomies) + " |"
            lines += [header, "|---" * (len(self.taxonomies) + 2) + "|"]
            for i, m in enumerate(self.eval_points):
                truth = rep["ground_truth_weights"].get(m, [])
                cells = []
                for t in self.taxonomies:
                    proj = self.projections.get(t)
                    cells.append(
                        _vec(proj.weight_for(m)) if proj is not None else "-"
                    )
                lines.append(f"| {m} | {_vec(truth)} | " + " | ".join(cells) + " |")

        if self.pairwise_procrustes:
            lines += [
                "",
                "## Taxonomy vs. taxonomy",
                "",
                "| pair | Procrustes disparity | mean total variation |",
                "|---|---|---|",
            ]
            for (a, b), res in self.pairwise_procrustes.items():
                tv = getattr(self.pairwise_simplex.get((a, b)), "mean_total_variation", None)
                lines.append(f"| {a} ↔ {b} | {_fmt(res.disparity)} | {_fmt(tv)} |")

        return "\n".join(lines) + "\n"

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> Path:
        """Write the report and every array needed to plot or re-analyse this later."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.report = self.to_report()
        (path / "report.json").write_text(json.dumps(self.report, indent=2))
        (path / "report.md").write_text(self.to_markdown())

        if self.ground_truth is not None:
            gt = path / "ground_truth"
            self.ground_truth.save(gt)
            (gt / "weights.json").write_text(
                json.dumps(
                    {
                        "vertices": list(self.vertices),
                        "weights": self.report["ground_truth_weights"],
                    },
                    indent=2,
                )
            )
        if self.ground_truth_matrix is not None:
            self.ground_truth_matrix.save(path / "ground_truth" / "distance_matrix")

        for tax, dm in self.distance_matrices.items():
            dm.save(path / "distance_matrices" / tax)
        for tax, geos in self.geometries.items():
            for key, geo in geos.items():
                geo.save(path / "geometries" / tax / key)
        for tax, proj in self.projections.items():
            proj.save(path / "simplex" / tax)
        # The fitted map onto the ground-truth simplex, not just its score — so a
        # later figure or convergence study can apply it rather than refit it.
        for tax, res in self.procrustes_vs_truth.items():
            res.save(path / "procrustes" / "vs_truth" / tax)

        return path

    @classmethod
    def load(cls, path: Path | str) -> "TaxonomyComparison":
        """Restore a saved comparison.

        Restored in full: the distance matrices, the geometries, the ground truth,
        the simplex projections, and the fitted Procrustes map onto the ground
        truth — including its rotation, centroids and scale, so
        :meth:`~src.analysis.configurations.ProcrustesResult.transform` works on a
        reloaded comparison.

        Not restored as objects: :attr:`pairwise_procrustes` and
        :attr:`protest_vs_truth`.  Their numbers are in :attr:`report`; a
        permutation null is cheaper to recompute than to store, and the pairwise
        maps are derivable from the restored geometries.
        """
        path = Path(path)
        report = json.loads((path / "report.json").read_text())

        obj = cls(
            slice_key=report.get("slice", {}),
            model_ids=list(report.get("model_ids", [])),
            vertices=list(report.get("vertices", [])),
            anchors=list(report.get("anchors", [])),
            eval_points=list(report.get("eval_points", [])),
            taxonomies=list(report.get("taxonomies", [])),
            simplex_dim=int(report.get("simplex_dim", 0)),
            projection_dim=int(report.get("projection_dim", 0)),
            report=report,
        )

        weights = report.get("ground_truth_weights", {})
        if weights and obj.model_ids:
            obj.ground_truth_weights = np.array(
                [weights[m] for m in obj.model_ids], dtype=np.float64
            )

        gt_dir = path / "ground_truth"
        if (gt_dir / "geometry.safetensors").exists():
            obj.ground_truth = GeometryResult.load(gt_dir)
        if (gt_dir / "distance_matrix" / "distance_matrix.safetensors").exists():
            obj.ground_truth_matrix = DistanceMatrix.load(gt_dir / "distance_matrix")

        dm_root = path / "distance_matrices"
        if dm_root.exists():
            for d in sorted(dm_root.iterdir()):
                if (d / "distance_matrix.safetensors").exists():
                    obj.distance_matrices[d.name] = DistanceMatrix.load(d)

        geo_root = path / "geometries"
        if geo_root.exists():
            for tax_dir in sorted(geo_root.iterdir()):
                if not tax_dir.is_dir():
                    continue
                obj.geometries[tax_dir.name] = {
                    g.name: GeometryResult.load(g)
                    for g in sorted(tax_dir.iterdir())
                    if (g / "geometry.safetensors").exists()
                }

        simplex_root = path / "simplex"
        if simplex_root.exists():
            for d in sorted(simplex_root.iterdir()):
                if (d / "simplex.safetensors").exists():
                    obj.projections[d.name] = SimplexProjection.load(d)

        proc_root = path / "procrustes" / "vs_truth"
        if proc_root.exists():
            for d in sorted(proc_root.iterdir()):
                if (d / "procrustes.safetensors").exists():
                    obj.procrustes_vs_truth[d.name] = ProcrustesResult.load(d)

        obj.stress = {
            t: v["stress"] for t, v in report.get("per_taxonomy", {}).items()
        }
        return obj


def compare_taxonomies(
    distance_matrices: Mapping[str, DistanceMatrix],
    recipes: Mapping[ModelID, Any],
    anchors: Sequence[ModelID] | None = None,
    n_components: Sequence[int] | None = None,
    projection_dim: int | None = None,
    key: Callable[[ModelID], str] | None = recipe_id_for,
    slice_key: Mapping[str, Any] | None = None,
    prefer_anchors: Sequence[ModelID] | None = None,
    n_permutations: int = 999,
    run_protest: bool = True,
    mds_kwargs: Mapping[str, Any] | None = None,
) -> TaxonomyComparison:
    """Run every comparison over one collection.

    Parameters
    ----------
    distance_matrices:
        ``{taxonomy_name: DistanceMatrix}``.  Model sets may differ; the
        intersection is used.
    recipes:
        Model ID → recipe, in any form :func:`~src.analysis.ground_truth.mixture_weights`
        accepts.  This is where the ground truth comes from.
    anchors:
        Override the automatic choice of pure-recipe anchors.
    n_components:
        Dimensions to embed at.  Defaults to ``{2, k-1}`` — ``k-1`` is the least
        the simplex admits, and 2 is what a plot needs.
    projection_dim:
        Dimension to run the barycentric projection and the Procrustes
        comparison from.  Defaults to the **largest** available, not ``k-1``.
        Projecting from exactly ``k-1`` makes the anchors' affine hull the entire
        space, so every residual is identically zero and the off-simplex
        diagnostic silently disappears — and it forces MDS to discard whatever
        structure did not fit in ``k-1`` dimensions.  ``k-1`` is the minimum the
        projection is *defined* at, not the right place to read it from.
    prefer_anchors:
        Tie-break ordering when several models share a vertex, which happens in a
        pooled collection spanning several sample sizes.
    run_protest:
        PROTEST is a permutation test and costs ``n_permutations`` superpositions
        per taxonomy; turn it off for a quick pass.
    """
    if not distance_matrices:
        raise ValueError("no distance matrices given")

    relabelled = {
        t: (relabel(dm, key) if key is not None else dm)
        for t, dm in distance_matrices.items()
    }
    recipes = {(key(m) if key is not None else m): r for m, r in recipes.items()}

    common = set.intersection(
        *(set(dm.model_ids) for dm in relabelled.values()), set(recipes)
    )
    if len(common) < 2:
        raise ValueError(
            "fewer than 2 models are shared by every distance matrix and the "
            f"recipe set (found {sorted(common)}). Check the identifier schemes "
            "line up — src.analysis.id_overlap(..., key=recipe_id_for) diagnoses this."
        )
    first = next(iter(relabelled.values()))
    model_ids = [m for m in first.model_ids if m in common]

    vertices, weights = ground_truth_weights({m: recipes[m] for m in model_ids})
    W = truth_matrix(weights, model_ids)
    k = len(vertices)
    simplex_dim = k - 1

    dims = sorted({2, simplex_dim} if n_components is None else {int(n) for n in n_components})
    dims = [d for d in dims if 1 <= d < len(model_ids)]
    usable = [d for d in dims if d >= simplex_dim]
    if not usable:
        raise ValueError(
            f"the ground-truth simplex has {k} vertices, so barycentric "
            f"projection needs an embedding of at least {simplex_dim} dimensions; "
            f"the available dimensions are {dims} for {len(model_ids)} models. "
            "Request a higher n_components, or add more models."
        )
    projection_dim = max(usable) if projection_dim is None else int(projection_dim)
    if projection_dim < simplex_dim:
        raise ValueError(
            f"projection_dim={projection_dim} is below the simplex dimension "
            f"{simplex_dim}; the anchors would be affinely dependent."
        )
    if projection_dim not in dims:
        dims = sorted(set(dims) | {projection_dim})

    anchors = list(anchors) if anchors else pure_anchors(vertices, weights, prefer=prefer_anchors)
    evals = evaluation_points(model_ids, anchors)

    result = TaxonomyComparison(
        slice_key=dict(slice_key or {}),
        model_ids=model_ids,
        vertices=vertices,
        anchors=anchors,
        eval_points=evals,
        taxonomies=sorted(relabelled),
        ground_truth_weights=W,
        simplex_dim=simplex_dim,
        projection_dim=projection_dim,
    )
    result.ground_truth = simplex_geometry(W, model_ids, vertices)
    result.ground_truth_matrix = simplex_distance_matrix(W, model_ids, vertices)

    truth_by_model = {m: W[i] for i, m in enumerate(model_ids)}

    from .bridge import fit_geometry

    kwargs = dict(mds_kwargs or {})
    kwargs.setdefault("random_state", 0)

    for tax in result.taxonomies:
        dm = _restrict(relabelled[tax], model_ids)
        result.distance_matrices[tax] = dm
        result.geometries[tax] = {
            f"mds_{d}d": fit_geometry(dm, method="mds", n_components=d, **kwargs)
            for d in dims
        }

        geo = result.geometries[tax][f"mds_{projection_dim}d"]
        result.stress[tax] = kruskal_stress(dm, geo)

        proj = barycentric(geo, anchors)
        result.projections[tax] = proj
        result.recovery[tax] = anchor_weight_vs_truth(proj, truth_by_model)
        if evals:
            result.recovery_eval_only[tax] = anchor_weight_vs_truth(
                proj, {m: truth_by_model[m] for m in evals}
            )
            found = np.vstack([proj.weight_for(m) for m in evals])
            truth = np.vstack([truth_by_model[m] for m in evals])
            # Total variation: the bounded [0, 1] distance between two weight
            # vectors, matching what compare_simplices reports between taxonomies.
            result.edge_deviation[tax] = 0.5 * np.abs(found - truth).sum(axis=1)

        # Argument order matters for the *map*, though not for the score:
        # procrustes_compare superimposes its second argument onto its first, so
        # the ground truth goes first and the taxonomy second. That makes the
        # fitted map run taxonomy -> ground-truth frame, which is the direction
        # worth keeping — `.transform(mds_coords)` then lands on the simplex.
        # Disparity is symmetric under the swap, so no reported number changes.
        #
        # The ground truth is (k-1)-dimensional and the embedding may be higher;
        # procrustes_compare zero-pads the smaller, which embeds it without
        # changing any of its internal distances.
        result.procrustes_vs_truth[tax] = procrustes_compare(result.ground_truth, geo)
        if run_protest and len(model_ids) >= 3:
            result.protest_vs_truth[tax] = protest(
                result.ground_truth, geo, n_permutations=n_permutations
            )
        result.matrix_corr_vs_truth[tax] = matrix_correlation(
            dm, result.ground_truth_matrix
        )
        # dCor is the matrix-level test that replaces Mantel's p-value: it reads
        # the matrices directly, so unlike PROTEST it carries none of the MDS
        # distortion `stress` measures.  The bias-corrected form needs 4 models.
        if len(model_ids) >= 4:
            result.dcor_vs_truth[tax] = dcor_test(
                dm, result.ground_truth_matrix, n_permutations=n_permutations
            )

    for i, a in enumerate(result.taxonomies):
        for b in result.taxonomies[i + 1 :]:
            result.pairwise_procrustes[(a, b)] = procrustes_compare(
                result.geometries[a][f"mds_{projection_dim}d"],
                result.geometries[b][f"mds_{projection_dim}d"],
            )
            result.pairwise_simplex[(a, b)] = compare_simplices(
                result.projections[a], result.projections[b]
            )

    result.matrix_correlations = correlation_table(result.distance_matrices)
    result.report = result.to_report()
    return result


def compare_all_slices(
    index: CacheIndex,
    taxonomies: Mapping[str, str],
    output_dir: Path | str,
    cache_root: Path | str | None = None,
    groupings: Sequence[Sequence[str]] = (("n_samples", "seed"), ("n_samples",), ("seed",), ()),
    n_components: Sequence[int] | None = None,
    min_models: int = 3,
    embedder_hash: str | None = None,
    behavioral_selector: dict | None = None,
    functional_selector: dict | None = None,
    **compare_kwargs,
) -> dict[tuple[str, tuple], TaxonomyComparison]:
    """Compare every slice of a collection, and the pooled whole, writing reports.

    The four default groupings answer different questions from the same cache:
    ``(n_samples, seed)`` isolates one experimental cell, ``(n_samples,)`` varies
    the seed within a size, ``(seed,)`` varies the size within a seed — the axis a
    convergence-in-``n`` study needs — and ``()`` pools everything.

    A slice that cannot be compared (too few models, a missing pure anchor, no
    cached embeddings) is recorded in the manifest with the reason and skipped,
    rather than aborting the sweep.

    Parameters
    ----------
    taxonomies:
        ``{taxonomy_name: metric}``, e.g. ``{"structural": "cosine",
        "dataset_embedding": "cosine"}``.
    embedder_hash, behavioral_selector, functional_selector:
        Forwarded to :func:`build_taxonomy_artifacts` for every slice.

        These were **missing** before, which meant a sweep could not name the
        draw, view or embedder it wanted: each slice silently took whatever the
        auto-resolution picked, and a sweep spanning two draws failed on the
        ambiguity instead of being told which to use.  They are per-collection
        choices, not per-slice ones, so one value applies to the whole sweep.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = cache_root or index.cache_root

    results: dict[tuple[str, tuple], TaxonomyComparison] = {}
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root),
        "taxonomies": dict(taxonomies),
        "groupings": {},
    }

    for by in groupings:
        by = tuple(by)
        group_name = "by_" + ("_and_".join(by) if by else "pooled")
        manifest["groupings"][group_name] = {}

        for slice_key, sub in index.slices(by=by).items():
            label = sub.slice_label(slice_key, by)
            record: dict[str, Any] = {"n_models": len(sub), "label": label}

            if len(sub) < min_models:
                record["skipped"] = f"only {len(sub)} model(s), need {min_models}"
                manifest["groupings"][group_name][label] = record
                continue

            slice_dict = dict(zip(by, slice_key))
            try:
                matrices = {}
                for tax, metric in taxonomies.items():
                    dm, _ = build_taxonomy_artifacts(
                        sub, tax, metric,
                        cache_root=cache_root,
                        n_components=n_components or (2,),
                        label=f"{group_name}/{label}",
                        slice_key=slice_dict,
                        embedder_hash=embedder_hash,
                        behavioral_selector=behavioral_selector,
                        functional_selector=functional_selector,
                    )
                    matrices[tax] = dm

                # In a pooled slice several adapters sit at the same simplex
                # vertex; prefer the largest sample size, which is the best-
                # estimated of them, and record that this happened.
                prefer = [
                    e.recipe_id or e.model_id
                    for e in sorted(
                        sub.entries, key=lambda x: -(x.n_samples or 0)
                    )
                ]
                comparison = compare_taxonomies(
                    matrices,
                    sub.recipes(),
                    n_components=n_components,
                    slice_key=slice_dict,
                    prefer_anchors=prefer,
                    **compare_kwargs,
                )
            except (ValueError, NotImplementedError, FileNotFoundError) as e:
                record["skipped"] = f"{type(e).__name__}: {e}"
                manifest["groupings"][group_name][label] = record
                continue

            comparison.save(output_dir / group_name / label)
            results[(group_name, slice_key)] = comparison
            record["report"] = f"{group_name}/{label}/report.md"
            record["taxonomies"] = comparison.taxonomies
            manifest["groupings"][group_name][label] = record

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return results


# ── helpers ────────────────────────────────────────────────────────────────────

def _restrict(dm: DistanceMatrix, model_ids: Sequence[ModelID]) -> DistanceMatrix:
    """Reindex a distance matrix onto *model_ids*, in that order."""
    if list(dm.model_ids) == list(model_ids):
        return dm
    pos = {m: i for i, m in enumerate(dm.model_ids)}
    idx = np.array([pos[m] for m in model_ids], dtype=int)
    return DistanceMatrix(
        matrix=np.asarray(dm.matrix, dtype=np.float64)[np.ix_(idx, idx)],
        model_ids=list(model_ids),
        metric=dm.metric,
        taxonomy=dm.taxonomy,
    )


def _recovery_dict(rec: RecoveryResult | None) -> dict | None:
    if rec is None:
        return None
    return {
        "columns": list(rec.columns),
        "pearson": [_f(v) for v in np.atleast_1d(rec.pearson)],
        "spearman": [_f(v) for v in np.atleast_1d(rec.spearman)],
        "pearson_mean": _f(np.nanmean(rec.pearson)) if rec.pearson.size else None,
        "spearman_mean": _f(np.nanmean(rec.spearman)) if rec.spearman.size else None,
        "mean_l1": _f(rec.mean_l1),
        "max_residual": _f(float(np.max(rec.residuals)) if rec.residuals.size else None),
        "n_models": len(rec.model_ids),
    }


def _f(value) -> float | None:
    """Coerce to a JSON-safe float; NaN and None both become None."""
    if value is None:
        return None
    v = float(value)
    return None if np.isnan(v) else v


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _vec(values) -> str:
    return "(" + ", ".join(f"{float(v):.3f}" for v in values) + ")"
