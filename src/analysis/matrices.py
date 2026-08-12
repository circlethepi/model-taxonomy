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
    """Outcome of a Mantel permutation test between two distance matrices.

    :attr:`statistic` is a descriptive correlation and is fine to quote.
    :attr:`p_value` is **not** calibrated — see the warning on :func:`mantel_test`.
    """

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

    .. warning::

       **Treat the statistic as descriptive and do not read the p-value as
       evidence.**  The ``n(n-1)/2`` off-diagonal entries are derived from ``n``
       points, so each shares a point with ``n-2`` others.  Under the spatial
       autocorrelation that geometric distances between related models certainly
       have, that dependence inflates the statistic's variance and the
       permutation null does not absorb it — the literature reports inflated
       type-I error for exactly this reason.  Use :func:`dcor_test` (matrix
       level, no embedding) or :func:`~src.analysis.configurations.protest`
       (configuration level, better power and calibration —
       Peres-Neto & Jackson 2001) for inference.  Mantel is kept because
       existing figures reference it.

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


# ── distance correlation ─────────────────────────────────────────────────────
#
# Székely, Rizzo & Bakirov (2007) for dCor; Székely & Rizzo (2013) for the
# bias-corrected (U-centred) form.  The appeal here is that it consumes
# distance matrices *directly* — no MDS step, so unlike PROTEST it does not
# inherit the embedding's distortion — and it detects non-monotone dependence,
# which a rank correlation cannot.


def _double_center(d: np.ndarray) -> np.ndarray:
    """V-statistic (biased) centring: subtract row, column and grand means."""
    row = d.mean(axis=0, keepdims=True)
    col = d.mean(axis=1, keepdims=True)
    return d - row - col + d.mean()


def _u_center(d: np.ndarray) -> np.ndarray:
    """U-statistic (bias-corrected) centring of Székely & Rizzo (2013).

    Uses ``n-2`` and ``(n-1)(n-2)`` denominators instead of ``n`` and ``n**2``
    and zeroes the diagonal, which makes the resulting inner product an
    unbiased estimator of the population ``dCov**2``.  Needs ``n >= 4``.
    """
    n = d.shape[0]
    if n < 4:
        raise ValueError(f"bias-corrected dCor needs at least 4 models, got {n}")
    row = d.sum(axis=0, keepdims=True) / (n - 2)
    col = d.sum(axis=1, keepdims=True) / (n - 2)
    out = d - row - col + d.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(out, 0.0)
    return out


def _clean(d: np.ndarray) -> np.ndarray:
    """Symmetrise, zero the diagonal, and widen to float64."""
    m = np.asarray(d, dtype=np.float64)
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    return m


def _center(d: np.ndarray, bias_corrected: bool) -> np.ndarray:
    return _u_center(d) if bias_corrected else _double_center(d)


def _dcor_from_centered(A: np.ndarray, B: np.ndarray, bias_corrected: bool) -> float:
    """Distance correlation from two already-centred matrices.

    Split out from :func:`_dcor_from_matrices` because centring commutes with a
    joint row/column relabelling — ``center(P B Pᵀ) == P center(B) Pᵀ`` — so the
    permutation null in :func:`dcor_test` can centre once and permute the
    centred matrix, rather than re-centring on every draw.
    """
    if bias_corrected:
        n = A.shape[0]
        # The diagonals are already zero, so the full sum *is* the j != k sum,
        # and the 1/(n(n-3)) scale cancels out of the ratio below.
        cov = float((A * B).sum())
        var_a = float((A * A).sum())
        var_b = float((B * B).sum())
        denom = var_a * var_b
        if denom <= 0.0:
            return float("nan")
        # dCor* is the ratio on the squared scale, and unlike dCor it may be
        # negative — that is the price of unbiasedness, not a bug.
        return float(cov / np.sqrt(denom))

    # dCor**2 = dCov**2 / sqrt(dVar_a**2 * dVar_b**2), so `denom` is already a
    # square root and only one more is taken.
    cov = float((A * B).mean())
    var_a = float((A * A).mean())
    var_b = float((B * B).mean())
    denom = np.sqrt(var_a * var_b)
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(max(cov, 0.0) / denom))


def _dcor_from_matrices(a: np.ndarray, b: np.ndarray, bias_corrected: bool) -> float:
    A = _center(_clean(a), bias_corrected)
    B = _center(_clean(b), bias_corrected)
    return _dcor_from_centered(A, B, bias_corrected)


def distance_correlation(
    dm_a: DistanceMatrix,
    dm_b: DistanceMatrix,
    bias_corrected: bool = True,
    key: Callable[[ModelID], str] | None = None,
) -> float:
    """Distance correlation between two matrices, computed on the matrices.

    With *bias_corrected* (the default) this is Székely & Rizzo's ``dCor*``, an
    unbiased estimator on the **squared** scale: it lies in ``[-1, 1]``, may be
    negative, and is not the square root of anything.  It is the right default
    at the sizes this repo works at — the classical V-statistic ``dCor`` is
    badly inflated for a handful of models and would report a large value
    between independent matrices.  Pass ``bias_corrected=False`` for the
    classical statistic in ``[0, 1]``.

    .. warning::

       **dCor is unsigned: it measures dependence, not agreement.**  A taxonomy
       whose geometry is the exact *reversal* of the truth scores
       ``dCor = 1.0``, identically to a perfect one.

       This is not a defect of dCor specifically — **no matrix-level statistic
       can see that inversion**, because a distance matrix has no notion of
       direction to begin with.  ``matrix_correlation`` does not rescue it
       either: on the real five-adapter slice the behavioral level recovers the
       mixing order backwards, and its ``matrix_corr_vs_truth`` is nonetheless
       ``+0.76``.  What catches it is the *recovery* correlation, downstream of
       MDS and the barycentric projection, where the mixture weights finally
       have a sign (behavioral scores ``r = -0.9995`` there; see the 2026-08-05
       table in ``docs/notes/TODO.md``).

       So read dCor as "how much of the truth's structure is present", and go to
       :class:`~src.analysis.comparison.TaxonomyComparison.recovery` to ask
       whether it points the right way.  Pinned by ``t_dcor_unsigned`` in
       ``scripts/check_analysis.py``.

    .. note::

       ``dCor = 0`` characterises independence only for metrics of strong
       negative type (Lyons 2013).  Euclidean distance qualifies; the cosine
       and CKA distances used here are not known to, so read a value near zero
       as "no dependence detected", not as proof of independence.

    *key* is passed to :func:`match_models` to reconcile differing identifier
    schemes.
    """
    _, (a, b) = match_models(dm_a, dm_b, key=key)
    return _dcor_from_matrices(a, b, bias_corrected)


@dataclass
class DcorResult:
    """Outcome of a permutation test on the distance correlation."""

    statistic: float
    p_value: float
    n_permutations: int
    n_models: int
    bias_corrected: bool
    #: True when every one of the ``n!`` relabellings was enumerated, so the
    #: p-value is exact rather than sampled.
    exact: bool
    null: np.ndarray

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"DcorResult(statistic={self.statistic:.4f}, p_value={self.p_value:.4g}, "
            f"n_models={self.n_models}, n_permutations={self.n_permutations}, "
            f"bias_corrected={self.bias_corrected}, exact={self.exact})"
        )


def dcor_test(
    dm_a: DistanceMatrix,
    dm_b: DistanceMatrix,
    n_permutations: int = 9999,
    bias_corrected: bool = True,
    random_state: int | None = 0,
    key: Callable[[ModelID], str] | None = None,
) -> DcorResult:
    """Permutation test on :func:`distance_correlation`.

    The null is the same one :func:`mantel_test` uses — permute the row/column
    index jointly, destroying only the model-to-model correspondence — but the
    statistic is not, and that is the point: dCor is computed from the
    doubly-centred matrices rather than from the dependent off-diagonal vector,
    so the permutation null is the whole of the inference and there is no
    embedding step to distort it.

    When ``n!`` is no larger than *n_permutations* every relabelling is
    enumerated and the p-value is **exact** (:attr:`DcorResult.exact`).  That
    matters at the sizes here: with five models there are only 120 distinct
    relabellings, so 9,999 random draws would resample the same 120 values and
    the smallest attainable p-value is 1/120 ≈ 0.0083 either way.

    .. warning::

       **At five models, exact agreement is *penalised*, and the p-value is too
       coarse to threshold at 0.05.**  U-centring adds symmetry the raw matrix
       does not have: on the evenly-spaced 1-D ground truth these slices use,
       the raw and doubly-centred matrices each have 2 automorphisms among the
       120 relabellings, but the U-centred one has **8** — it can no longer tell
       either endpoint from its neighbour.  All 8 tie at ``dCor = 1``, and ties
       count toward a one-sided p-value, so a taxonomy reproducing the truth
       *exactly* scores ``p = 8/120 ≈ 0.067`` while a merely good one can score
       **lower**: on the real slice ``functional`` reaches ``4/120 ≈ 0.033``.

       So this is not a floor — ``p < 0.05`` is attainable — but the statistic
       is not monotone in agreement near the top, and the resolution is 1/120.
       **Rank the levels by** :attr:`DcorResult.statistic`, **not by p**, and
       treat the whole effect as an argument for more adapters per slice.
       Pinned by ``t_dcor_u_centering_symmetry``.

    *key* is passed to :func:`match_models` to reconcile differing identifier
    schemes.
    """
    from itertools import permutations
    from math import factorial

    ids, (a, b) = match_models(dm_a, dm_b, key=key)
    n = len(ids)
    minimum = 4 if bias_corrected else 3
    if n < minimum:
        raise ValueError(f"dcor_test needs at least {minimum} common models, got {n}")

    A = _center(_clean(a), bias_corrected)
    B = _center(_clean(b), bias_corrected)
    observed = _dcor_from_centered(A, B, bias_corrected)

    exact = factorial(n) <= n_permutations
    if exact:
        perms = (np.asarray(p) for p in permutations(range(n)))
        total = factorial(n)
    else:
        rng = np.random.default_rng(random_state)
        perms = (rng.permutation(n) for _ in range(n_permutations))
        total = n_permutations

    null = np.fromiter(
        (
            _dcor_from_centered(A, B[np.ix_(perm, perm)], bias_corrected)
            for perm in perms
        ),
        dtype=np.float64,
        count=total,
    )

    if exact:
        # The identity relabelling is in the enumeration, so the observed value
        # is already counted and the p-value cannot be zero.
        p_value = float(np.sum(null >= observed) / total)
    else:
        p_value = float((np.sum(null >= observed) + 1) / (total + 1))

    return DcorResult(
        statistic=observed,
        p_value=p_value,
        n_permutations=total,
        n_models=n,
        bias_corrected=bias_corrected,
        exact=exact,
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
