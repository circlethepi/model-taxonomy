"""Level 1 — comparison of distance matrices.

Asks whether two taxonomy levels rank *model pairs* the same way.  Nothing here
looks at coordinates; see :mod:`src.analysis.configurations` for that.

Model membership genuinely differs between taxonomies — ``run_taxonomy.py``
drops models that have no LoRA adapter, so ``structural`` may cover five models
where ``behavioral`` covers seven, in a different order.  Every function here
therefore reindexes to the common set first, via :func:`match_models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult
from src.core.protocols import ModelID

CorrelationMethod = Literal["spearman", "pearson"]


def match_models(*objs, key: Callable[[ModelID], str] | None = None) -> tuple[list[ModelID], list[np.ndarray]]:
    """Reindex several objects onto their common models, in one shared order.

    This is bookkeeping — a set intersection plus a permutation — and has
    nothing to do with Procrustes alignment.

    Accepts any mix of :class:`DistanceMatrix` (reindexed on both axes) and
    :class:`GeometryResult` (reindexed on rows only).  The shared order follows
    the first object's ``model_ids``, restricted to the intersection, so results
    are deterministic rather than set-ordering dependent.

    Parameters
    ----------
    key:
        Optional normalisation applied to every identifier before intersecting.
        Two taxonomy levels can describe the same models under different naming
        schemes — the dataset-level and model-level taxonomies being the case
        that arises in practice — and a *key* reconciles them without touching
        stored results.  Pass
        :func:`src.analysis.identity.recipe_id_for` for that one.  The returned
        ``model_ids`` are the normalised identifiers.

    Returns
    -------
    model_ids:
        The common models, in the shared order.
    arrays:
        One reindexed array per input, in input order.
    """
    if len(objs) < 1:
        raise ValueError("match_models needs at least one object")

    id_lists = [[key(m) for m in _model_ids(o)] if key else list(_model_ids(o)) for o in objs]
    for obj, ids in zip(objs, id_lists):
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"{type(obj).__name__} has duplicate identifiers after applying "
                f"key — its rows would be ambiguous. See src.analysis.relabel."
            )
    common = set(id_lists[0]).intersection(*(set(x) for x in id_lists[1:]))
    if not common:
        raise ValueError(
            "No models in common. First object has "
            f"{id_lists[0][:3]}...; second has {id_lists[1][:3] if len(id_lists) > 1 else '-'}..."
        )
    order = [m for m in id_lists[0] if m in common]

    arrays: list[np.ndarray] = []
    for obj, ids in zip(objs, id_lists):
        pos = {m: i for i, m in enumerate(ids)}
        idx = np.array([pos[m] for m in order], dtype=int)
        if isinstance(obj, DistanceMatrix):
            arrays.append(np.asarray(obj.matrix, dtype=np.float64)[np.ix_(idx, idx)])
        elif isinstance(obj, GeometryResult):
            arrays.append(np.asarray(obj.coordinates, dtype=np.float64)[idx])
        else:
            raise TypeError(
                f"match_models accepts DistanceMatrix or GeometryResult, got {type(obj).__name__}"
            )
    return order, arrays


def _model_ids(obj) -> Sequence[ModelID]:
    ids = getattr(obj, "model_ids", None)
    if ids is None:
        raise TypeError(f"{type(obj).__name__} has no model_ids")
    return ids


def offdiag(matrix: np.ndarray) -> np.ndarray:
    """Return the ``n(n-1)/2`` off-diagonal entries as a flat vector.

    Symmetrises first: ``DistanceMatrix.save`` writes float32 and ``load``
    returns float64, so a round-tripped matrix can be very slightly asymmetric
    and ``squareform`` would reject it.
    """
    from scipy.spatial.distance import squareform

    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {m.shape}")
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    return squareform(m, checks=False)


def matrix_correlation(
    dm_a: DistanceMatrix,
    dm_b: DistanceMatrix,
    method: CorrelationMethod = "spearman",
    key: Callable[[ModelID], str] | None = None,
) -> float:
    """Correlation between the two matrices' off-diagonal vectors.

    For five common models that is a correlation between two 10-element
    vectors, one entry per model pair: *do the two taxonomies rank model-pair
    similarity the same way?*

    *key* is passed to :func:`match_models` to reconcile differing identifier
    schemes.
    """
    _, (a, b) = match_models(dm_a, dm_b, key=key)
    return _corr(offdiag(a), offdiag(b), method)


def _corr(x: np.ndarray, y: np.ndarray, method: CorrelationMethod) -> float:
    from scipy.stats import pearsonr, spearmanr

    if x.size < 2:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(f"Unknown method {method!r}. Choose from spearman, pearson.")


@dataclass
class MantelResult:
    """Outcome of a Mantel permutation test between two distance matrices."""

    statistic: float
    p_value: float
    n_permutations: int
    n_models: int
    method: CorrelationMethod
    null: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"MantelResult(statistic={self.statistic:.4f}, p_value={self.p_value:.4g}, "
            f"n_models={self.n_models}, n_permutations={self.n_permutations}, "
            f"method={self.method!r})"
        )


def mantel_test(
    dm_a: DistanceMatrix,
    dm_b: DistanceMatrix,
    n_permutations: int = 9999,
    method: CorrelationMethod = "spearman",
    random_state: int | None = 0,
    key: Callable[[ModelID], str] | None = None,
) -> MantelResult:
    """Permutation test for correspondence between two distance matrices.

    The null is built by permuting the **row/column index jointly** — relabelling
    which model is which — never by shuffling the off-diagonal entries.  Each
    matrix therefore keeps its internal structure completely intact, and the only
    thing destroyed is the model-to-model correspondence between the two.  That
    is precisely the null hypothesis "these two taxonomies' notions of similarity
    are unrelated".

    The p-value is one-sided (``P(null >= observed)``) with the observed value
    included in the count, so it can never be zero and is bounded below by
    ``1 / (n_permutations + 1)``.

    *key* is passed to :func:`match_models` to reconcile differing identifier
    schemes.
    """
    ids, (a, b) = match_models(dm_a, dm_b, key=key)
    n = len(ids)
    if n < 3:
        raise ValueError(f"Mantel test needs at least 3 common models, got {n}")

    va = offdiag(a)
    observed = _corr(va, offdiag(b), method)

    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        null[i] = _corr(va, offdiag(b[np.ix_(perm, perm)]), method)

    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return MantelResult(
        statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        n_models=n,
        method=method,
        null=null,
    )


def correlation_table(
    analyses,
    method: CorrelationMethod = "spearman",
    min_models: int = 3,
    key: Callable[[ModelID], str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """All-pairs correlation across taxonomy levels.

    Parameters
    ----------
    analyses:
        A :class:`~src.core.analysis.ModelTaxonomyProfile`, or any mapping of
        ``{label: DistanceMatrix}``.
    method:
        Correlation to use for each pair.
    min_models:
        Pairs sharing fewer than this many models are reported as ``nan``
        rather than as a correlation computed from too little data.
    key:
        Identifier normalisation, passed to :func:`match_models`.  Pass
        :func:`src.analysis.identity.recipe_id_for` to bring
        ``dataset_embedding`` into the table — see Notes.

    Returns
    -------
    labels, table:
        ``table`` is ``(n_taxonomies, n_taxonomies)`` with 1.0 on the diagonal.
        Feed it to :func:`src.plots.plot_distance_heatmap` by wrapping with
        :func:`src.analysis.bridge.as_distance_matrix`.

    Notes
    -----
    Entries are ``nan`` where two levels cannot be compared, rather than raising
    — one incomparable pair should not destroy the whole table.

    With ``key=None`` that includes ``dataset_embedding``, which
    :class:`~src.taxonomy.dataset_embedding.DatasetEmbeddingTaxonomy` keys by
    *recipe* ID (``yahoo_topic0_only``) while the model-level taxonomies key by
    adapter path (``.../yahoo_topic0_only_r16``).  The two describe the same
    experimental objects, so passing ``key=recipe_id_for`` maps the adapter
    paths onto their training recipes and the whole table fills in.  A
    dataset-level taxonomy can legitimately carry entries no adapter was trained
    on — a held-out probe set, say — and those simply fall out of the
    intersection.
    """
    mapping = _as_matrix_mapping(analyses)
    labels = list(mapping)
    n = len(labels)
    table = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = mapping[labels[i]], mapping[labels[j]]
            try:
                ids, _ = match_models(a, b, key=key)
            except ValueError:
                c = np.nan
            else:
                c = (
                    matrix_correlation(a, b, method, key=key)
                    if len(ids) >= min_models
                    else np.nan
                )
            table[i, j] = table[j, i] = c
    return labels, table


def _as_matrix_mapping(analyses) -> dict[str, DistanceMatrix]:
    if isinstance(analyses, Mapping):
        out = {}
        for k, v in analyses.items():
            out[k] = v.distance_matrix if hasattr(v, "distance_matrix") else v
        return out
    # ModelTaxonomyProfile
    if hasattr(analyses, "analyses"):
        return {k: v.distance_matrix for k, v in analyses.analyses.items()}
    raise TypeError(
        "Expected a ModelTaxonomyProfile or a mapping of {label: DistanceMatrix}, "
        f"got {type(analyses).__name__}"
    )
