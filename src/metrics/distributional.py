"""Distances between representations read as *distributions* of rows.

The metrics in :mod:`src.metrics.frobenius` and :mod:`src.metrics.cka` compare
two indexed lists of vectors: row *i* of one model is assumed to mean the same
thing as row *i* of the other, and permuting one input changes the answer.  That
assumption is right when rows are queries in a shared draw and wrong as soon as
they are not.

These two treat each representation as a sample from a distribution over the
feature space, and so are invariant to row order and tolerant of unequal row
counts — the same property :class:`~src.metrics.bures_wasserstein.BuresWassersteinDistanceMetric`
has, but without its restriction to second moments.  BW compares two clouds
through their covariances alone, which cannot see a difference in shape at equal
covariance; MMD with a characteristic kernel and the energy distance both can.

Where this matters here
-----------------------
The behavioral level stores ``(n_queries * replicates, d)`` in query-major order.
Under sampling, the *spread* of a model's 16 replicates for one question is part
of what distinguishes it, and pairing replicate 3 of model A with replicate 3 of
model B is meaningless — they are independent draws.  A distributional distance
is the honest reading of that matrix.

Both estimators cost O(n_a * n_b * d) per pair, dominated by the cross Gram.  At
the sizes here (1600 behavioral rows, 1000 dataset rows) that is well under a
second per pair.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from src.core.protocols import DistanceMetric
from src.core.representation import ModelRepresentation


def _rows(rep: ModelRepresentation, who: str) -> np.ndarray:
    m = np.atleast_2d(rep.matrix.astype(np.float64))
    if m.shape[0] < 2:
        raise ValueError(
            f"{who} ({rep.model_id!r}) has {m.shape[0]} row(s). A distributional "
            "distance compares two samples, so it needs more than one row per "
            "model — use representation='matrix' rather than a pooled mean."
        )
    if rep.metadata.get("is_kernel"):
        raise ValueError(
            f"{rep.model_id!r} is a kernel matrix "
            f"(view={rep.metadata.get('view')!r}), not a sample of feature "
            "vectors. Its rows are not points in the space this metric measures. "
            "Use view='concat'."
        )
    return m


def _check_dims(ma: np.ndarray, mb: np.ndarray, name: str) -> None:
    if ma.shape[1] != mb.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: {ma.shape} vs {mb.shape}. {name} "
            "compares two samples drawn in one space, so the feature dimension "
            "must match (row counts need not)."
        )


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between every row of *X* and every row of *Y*.

    Via the expansion rather than a broadcast difference: the (n, m, d) array the
    latter builds is 768x larger than the (n, m) result, and at 1600 rows that is
    the difference between 20 MB and 15 GB.  Clipped at zero because the
    expansion can go slightly negative for near-identical rows.
    """
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    yy = np.einsum("ij,ij->i", Y, Y)[None, :]
    return np.maximum(xx + yy - 2.0 * (X @ Y.T), 0.0)


def _offdiag_mean(D: np.ndarray) -> float:
    """Mean of a square matrix off the diagonal — the U-statistic denominator.

    Including the diagonal would fold in each point's zero distance to itself,
    biasing the within-sample term toward zero by a factor of (n-1)/n and
    shrinking every distance in the collection by a model-size-dependent amount.
    """
    n = D.shape[0]
    return float((D.sum() - np.trace(D)) / (n * (n - 1)))


class EnergyDistanceMetric(DistanceMetric):
    """Energy distance between the two row clouds (Székely & Rizzo).

    ``E(A, B)² = 2·E‖x − y‖ − E‖x − x'‖ − E‖y − y'‖`` with x, x' ~ A and
    y, y' ~ B, estimated by U-statistics (self-pairs excluded).  This is
    non-negative, and zero exactly when the two distributions coincide, so the
    square root reported here is a true metric on distributions.

    No bandwidth, no kernel choice — which is the reason to prefer it as the
    default distributional reading.  It is equivalent to MMD with the (negative
    definite) Euclidean-distance kernel, so it and
    :class:`MMDDistanceMetric` are the same family, not independent evidence.

    Scale-sensitive by construction: ``X → cX`` scales the distance by ``|c|``.
    """

    @property
    def metric_name(self) -> str:
        return "energy"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        ma = _rows(a, "a")
        mb = _rows(b, "b")
        _check_dims(ma, mb, "The energy distance")

        cross = float(np.sqrt(_sq_dists(ma, mb)).mean())
        within_a = _offdiag_mean(np.sqrt(_sq_dists(ma, ma)))
        within_b = _offdiag_mean(np.sqrt(_sq_dists(mb, mb)))

        # Non-negative in exact arithmetic; the U-statistic can land a few ulps
        # below zero when the two samples are the same points.
        return float(np.sqrt(max(2.0 * cross - within_a - within_b, 0.0)))


class MMDDistanceMetric(DistanceMetric):
    """Maximum Mean Discrepancy between the two row clouds.

    ``MMD²(A, B) = E k(x, x') + E k(y, y') − 2 E k(x, y)``, with the unbiased
    estimator (self-pairs excluded from the two within-sample terms).  Reported
    as ``sqrt(max(MMD², 0))``.

    The clamp is not cosmetic and the unbiased estimator is not interchangeable
    with the biased one here: under the null the unbiased MMD² is centred on zero
    and is negative about half the time.  Taking a square root of that is
    undefined, so it is clamped — which means a value of exactly 0.0 should be
    read as "at or below the noise floor", not as "identical".  The biased
    estimator avoids the clamp by being positive even under the null, which would
    put a floor under every distance in the collection that grows as row counts
    shrink; that is worse, because it varies across the very models being
    compared.

    Parameters
    ----------
    kernel:
        ``"rbf"`` (Gaussian) is characteristic, so MMD = 0 implies the
        distributions are equal.  ``"linear"`` sees only the difference of means
        and is offered as the explicit degenerate case: it is what a cosine or
        Frobenius comparison of pooled centroids already measures.
    sigma:
        RBF bandwidth.  ``None`` selects the median heuristic — the median
        pairwise distance over the two samples *pooled*.  It is recomputed for
        each pair rather than fixed once over the collection, because a bandwidth
        derived from one model's scale would make ``d(a, b) != d(b, a)``.  A
        consequence worth stating: with ``sigma=None`` the metric is symmetric
        but not a metric across the collection, since different pairs are
        measured with different kernels.  Pass an explicit *sigma* when a
        distance matrix needs to be internally consistent — an MDS embedding of
        one is exactly that case.
    """

    def __init__(
        self,
        kernel: Literal["rbf", "linear"] = "rbf",
        sigma: float | None = None,
    ) -> None:
        if kernel not in ("rbf", "linear"):
            raise ValueError(f"kernel must be 'rbf' or 'linear', got {kernel!r}")
        if sigma is not None and sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.kernel = kernel
        self.sigma = sigma

    @property
    def metric_name(self) -> str:
        return f"mmd_{self.kernel}"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        ma = _rows(a, "a")
        mb = _rows(b, "b")
        _check_dims(ma, mb, "MMD")

        if self.kernel == "linear":
            kaa, kbb, kab = ma @ ma.T, mb @ mb.T, ma @ mb.T
        else:
            daa, dbb, dab = (_sq_dists(ma, ma), _sq_dists(mb, mb),
                             _sq_dists(ma, mb))
            sigma = self.sigma if self.sigma is not None else _median_sigma(daa, dbb, dab)
            gamma = 1.0 / (2.0 * sigma ** 2)
            kaa, kbb, kab = (np.exp(-gamma * daa), np.exp(-gamma * dbb),
                             np.exp(-gamma * dab))

        mmd2 = _offdiag_mean(kaa) + _offdiag_mean(kbb) - 2.0 * float(kab.mean())
        return float(np.sqrt(max(mmd2, 0.0)))


def _median_sigma(daa: np.ndarray, dbb: np.ndarray, dab: np.ndarray) -> float:
    """Median heuristic bandwidth over the pooled pair, from squared distances.

    Symmetric in its two samples by construction — every within- and between-
    sample distance enters once.  Falls back to 1.0 when the median is zero,
    which happens only when the pooled sample is a single repeated point; a zero
    bandwidth would make gamma infinite and every kernel entry NaN.
    """
    pooled = np.concatenate([
        np.sqrt(daa[np.triu_indices_from(daa, k=1)]),
        np.sqrt(dbb[np.triu_indices_from(dbb, k=1)]),
        np.sqrt(dab).ravel(),
    ])
    med = float(np.median(pooled))
    return med if med > 0 else 1.0
