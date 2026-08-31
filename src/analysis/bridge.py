"""Adapters that let the notebook LoRA-weight path speak the core pipeline types.

The repository grows distance matrices in two places:

* the config-driven pipeline (``scripts/run_taxonomy.py`` → :class:`TaxonomyAnalyzer`),
  which yields a typed :class:`~src.core.distance.DistanceMatrix`; and
* the low-rank LoRA builders in :mod:`src.notebook.structure`, which yield a bare
  ``(names, ndarray)`` tuple and are never persisted.

Everything here converts the second into the first, so a single analysis layer —
:mod:`src.analysis.matrices`, :mod:`src.analysis.configurations`,
:mod:`src.analysis.quality`, :mod:`src.analysis.simplex` — serves both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult
from src.core.protocols import ModelID

DistanceKind = Literal["cosine", "frobenius", "bures_wasserstein", "cka"]
GeometryName = Literal["mds", "pca", "umap"]


def as_distance_matrix(
    names: Sequence[ModelID],
    matrix: np.ndarray,
    metric: str,
    taxonomy: str = "structural",
    similarity: bool = False,
) -> DistanceMatrix:
    """Wrap a bare ``(names, matrix)`` pair as a :class:`DistanceMatrix`.

    Parameters
    ----------
    names:
        Model/adapter identifiers in matrix row order.
    matrix:
        Symmetric ``(n, n)`` array.
    metric:
        Name recorded on the result, e.g. ``"cosine"``.
    taxonomy:
        Taxonomy level the matrix belongs to.
    similarity:
        Set when *matrix* holds similarities rather than distances; the values
        are converted with ``1 - S``.  :func:`src.notebook.structure.cosine_similarity_matrix`
        is the case that needs this — it returns similarity in ``[-1, 1]``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if similarity:
        arr = 1.0 - arr
    # The builders fill only the upper triangle before mirroring; guard against
    # accumulated float asymmetry so downstream squareform() calls stay happy.
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    return DistanceMatrix(
        matrix=arr,
        model_ids=list(names),
        metric=metric,
        taxonomy=taxonomy,
    )


def lora_distance_matrix(
    weights,
    kind: DistanceKind = "cosine",
    layers: int | list[int] | None = None,
    projections: str | list[str] | None = None,
    align: bool = False,
    cache_dir: Path | None = None,
    taxonomy: str = "structural",
) -> DistanceMatrix:
    """Build a :class:`DistanceMatrix` directly from raw LoRA A/B factors.

    Dispatches to the low-rank builders in :mod:`src.notebook.structure`, which
    never form the ``d × d`` product ``B @ A`` — the only tractable route at
    28 layers × 4 projections × 3072².  The maths is not reimplemented here; see
    ``docs/notes/frobenius_bw_generalization.md`` for why the multi-block
    generalisation is exact.

    Parameters
    ----------
    weights:
        A :class:`~src.notebook.lora_weights.LoRAWeightCollection`.
    kind:
        Which builder to use.  ``"cka"`` accepts only a single ``(layer, proj)``
        block — pass a bare int for *layers* and a single string for *projections*.
    layers, projections:
        Blocks to include; ``None`` means every layer/projection loaded in
        *weights*.
    align:
        Forwarded to the builder.  This is the Procrustes alignment *of the LoRA
        factors themselves* offered by :mod:`src.notebook.structure`, and is
        unrelated to the configuration-level alignment in
        :mod:`src.analysis.configurations`.  Supported for a single block only,
        and a no-op for ``"cka"`` (linear CKA is orthogonally invariant).
    cache_dir:
        Where to cache alignment matrices when ``align=True``.
    """
    from src.notebook import structure as _structure

    if kind == "cosine":
        names, mat = _structure.cosine_similarity_matrix(
            weights, layers=layers, projections=projections
        )
        return as_distance_matrix(names, mat, "cosine", taxonomy, similarity=True)

    if kind == "frobenius":
        names, mat = _structure.frobenius_distance_matrix(
            weights, layers=layers, projections=projections,
            align=align, cache_dir=cache_dir,
        )
        return as_distance_matrix(names, mat, "frobenius", taxonomy)

    if kind == "bures_wasserstein":
        names, mat = _structure.bures_wasserstein_distance_matrix(
            weights, layers=layers, projections=projections,
            align=align, cache_dir=cache_dir,
        )
        return as_distance_matrix(names, mat, "bures_wasserstein", taxonomy)

    if kind == "cka":
        layer, proj = _single_block(weights, layers, projections)
        names, mat = _structure.cka_distance_matrix(
            weights, layer=layer, proj=proj, align=align, cache_dir=cache_dir,
        )
        return as_distance_matrix(names, mat, "cka_linear", taxonomy)

    raise ValueError(
        f"Unknown kind {kind!r}. Choose from cosine, frobenius, "
        "bures_wasserstein, cka."
    )


def _single_block(
    weights,
    layers: int | list[int] | None,
    projections: str | list[str] | None,
) -> tuple[int, str]:
    """Resolve (layers, projections) to exactly one (layer, proj) pair for CKA."""
    from src.notebook.structure import _normalize_layers, _normalize_projections

    _layers = _normalize_layers(weights, layers)
    _projs = _normalize_projections(weights, projections)
    if len(_layers) != 1 or len(_projs) != 1:
        raise ValueError(
            f"kind='cka' supports a single (layer, projection) block; got "
            f"{len(_layers)} layer(s) and {len(_projs)} projection(s). "
            "Linear CKA cannot be summed across blocks — its kernel lives on "
            "the d_out axis, which differs between projections under GQA. "
            "See docs/notes/cka_notes.md."
        )
    return _layers[0], _projs[0]


def fit_geometry(
    dm: DistanceMatrix,
    method: GeometryName = "mds",
    n_components: int = 2,
    **kwargs: Any,
) -> GeometryResult:
    """Embed a distance matrix into coordinates.

    Mirrors ``scripts/_utils.make_geometry`` but leaves *n_components* free —
    that factory hardcodes 2, which rules out 1-D embeddings and simplex
    projection in the full embedding dimension.

    Extra keyword arguments are forwarded to the underlying geometry class
    (e.g. ``metric=False`` for non-metric MDS, ``random_state=42``).
    """
    if method == "mds":
        from src.geometry_methods.mds import MDSGeometry

        return MDSGeometry(n_components=n_components, **kwargs).fit(dm)

    if method == "pca":
        from src.geometry_methods.pca import PCAGeometry

        if kwargs:
            raise TypeError(f"PCAGeometry takes no extra arguments; got {sorted(kwargs)}")
        return PCAGeometry(n_components=n_components).fit(dm)

    if method == "umap":
        try:
            from src.geometry_methods.umap import UMAPGeometry
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "UMAPGeometry requires the 'umap-learn' package, which is not "
                "installed in this environment. Use method='mds' or 'pca'."
            ) from e
        return UMAPGeometry(n_components=n_components, **kwargs).fit(dm)

    raise ValueError(f"Unknown geometry method {method!r}. Choose from mds, pca, umap.")


def save_collection(
    dm: DistanceMatrix,
    geometries: Iterable[GeometryResult] = (),
    cache_root: Path | str = "results/shared_cache",
    model_entries: list[dict] | None = None,
    label: str | None = None,
    slice_key: dict | None = None,
) -> str:
    """Persist a distance matrix and its geometries via :class:`CollectionCache`.

    Gives notebook-built structural matrices the same on-disk life as pipeline
    results, instead of existing only until the kernel restarts.

    *model_entries* should carry an ``artifact_path`` per model, as
    :func:`~src.analysis.comparison.build_taxonomy_artifacts` supplies — that is
    what keys the collection to the tensors it was built from.  Without it the
    entries key on model IDs alone, which is the item-14 blind spot: two matrices
    over the same models under different views would collide.  A matrix built by
    hand in a notebook usually has no such path, so this is a warning rather than
    an error, but prefer ``build_taxonomy_artifacts`` where you can.

    Returns the handle (the path under ``{cache_root}/07_collections/``).
    """
    import warnings

    from src.cache.collection_cache import CollectionCache

    cc = CollectionCache(cache_root)
    entries = model_entries or [
        {"model_id": mid, "entry_type": "base_model"} for mid in dm.model_ids
    ]
    if any(e.get("artifact_path") is None for e in entries):
        warnings.warn(
            "save_collection: model_entries carry no 'artifact_path', so this "
            "collection is keyed on model IDs alone and will collide with any "
            "other view of the same models (TODO.md item 14). Prefer "
            "build_taxonomy_artifacts, which supplies them.",
            stacklevel=2,
        )
    handle = cc.handle(
        dm.taxonomy,
        cc.collection_key(entries),
        dm.metric,
        cc.surrogate_key([e.get("surrogate_hash") for e in entries]),
    )
    cc.save_distance_matrix(
        dm, handle, model_entries=entries, label=label, slice_key=slice_key
    )
    for geo in geometries:
        cc.save_geometry(handle, geo)
    return handle
