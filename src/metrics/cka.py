from __future__ import annotations

from typing import Literal

import numpy as np

from src.core.protocols import DistanceMetric
from src.core.representation import ModelRepresentation


class CKADistanceMetric(DistanceMetric):
    """Centered Kernel Alignment distance (Kornblith et al., 2019).

    CKA(A, B) in [0, 1] measures representational similarity; 1 = identical geometry.
    Distance = 1 - CKA(A, B), so 0 means identical.

    Invariant to orthogonal transformations and isotropic scaling, making it
    more geometrically meaningful than Frobenius norm for comparing representations
    of different dimensionality or scale.
    """

    def __init__(
        self,
        kernel: Literal["linear", "rbf"] = "linear",
        sigma: float | None = None,
        unbiased: bool = True,
    ) -> None:
        self.kernel = kernel
        self.sigma = sigma
        self.unbiased = unbiased

    @property
    def metric_name(self) -> str:
        return f"cka_{self.kernel}"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        if a.n_queries != b.n_queries:
            raise ValueError(
                f"Query count mismatch: {a.n_queries} vs {b.n_queries}. "
                "Both representations must use the same query set."
            )
        X = a.matrix.astype(np.float64)
        Y = b.matrix.astype(np.float64)

        if self.kernel == "linear":
            K = X @ X.T
            L = Y @ Y.T
        else:
            sigma = self.sigma or _median_bandwidth(X, Y)
            K = _rbf_kernel(X, sigma)
            L = _rbf_kernel(Y, sigma)

        cka_val = _cka(K, L, unbiased=self.unbiased)
        return float(1.0 - cka_val)


def _center(K: np.ndarray) -> np.ndarray:
    """Double-center a kernel matrix."""
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    total_mean = K.mean()
    return K - row_mean - col_mean + total_mean


def _hsic(K: np.ndarray, L: np.ndarray, unbiased: bool) -> float:
    """Hilbert-Schmidt Independence Criterion."""
    n = K.shape[0]
    if unbiased:
        # Unbiased estimator (Song et al., 2012)
        K_ = K.copy()
        L_ = L.copy()
        np.fill_diagonal(K_, 0)
        np.fill_diagonal(L_, 0)
        KL = K_ @ L_
        trace_KL = np.trace(KL)
        sum_K = K_.sum()
        sum_L = L_.sum()
        sum_KL = KL.sum()
        return (
            trace_KL
            + sum_K * sum_L / ((n - 1) * (n - 2))
            - 2 * sum_KL / (n - 2)
        ) / (n * (n - 3))
    else:
        Kc = _center(K)
        Lc = _center(L)
        return float(np.sum(Kc * Lc) / (n - 1) ** 2)


def _cka(K: np.ndarray, L: np.ndarray, unbiased: bool) -> float:
    hsic_kl = _hsic(K, L, unbiased)
    hsic_kk = _hsic(K, K, unbiased)
    hsic_ll = _hsic(L, L, unbiased)
    denom = np.sqrt(max(hsic_kk, 0.0) * max(hsic_ll, 0.0))
    if denom < 1e-10:
        return 0.0
    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


def _rbf_kernel(X: np.ndarray, sigma: float) -> np.ndarray:
    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        + np.sum(X ** 2, axis=1)
        - 2 * X @ X.T
    )
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2 * sigma ** 2))


def _median_bandwidth(*matrices: np.ndarray) -> float:
    """Median heuristic: sigma = median of pairwise distances in the combined data."""
    combined = np.vstack(matrices)
    sq_dists = (
        np.sum(combined ** 2, axis=1, keepdims=True)
        + np.sum(combined ** 2, axis=1)
        - 2 * combined @ combined.T
    )
    sq_dists = np.maximum(sq_dists, 0.0)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    return float(np.sqrt(np.median(upper))) if len(upper) > 0 else 1.0
