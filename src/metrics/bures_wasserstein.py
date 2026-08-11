from __future__ import annotations

import numpy as np

from src.core.protocols import DistanceMetric
from src.core.representation import ModelRepresentation


class BuresWassersteinDistanceMetric(DistanceMetric):
    """Bures-Wasserstein distance between the uncentered covariances Σ = XᵀX.

    The 2-Wasserstein distance between two zero-mean Gaussians with covariances
    Σ_a and Σ_b::

        d² = tr Σ_a + tr Σ_b - 2 tr((Σ_a^½ Σ_b Σ_a^½)^½)

    Computed here without ever forming a d×d covariance or a matrix square root.
    Writing Σ = XᵀX and taking thin SVDs X = U S Vᵀ, the singular values of
    Σ_a^½ Σ_b^½ are those of S_a V_aᵀ V_b S_b, which are in turn those of
    A Bᵀ — so the trace term is a nuclear norm of the *cross* matrix::

        d²(A, B) = ‖A‖_F² + ‖B‖_F² - 2‖A Bᵀ‖_*

    This is the same identity :func:`src.notebook.structure.bures_wasserstein_distance_matrix`
    uses on the LoRA factors, where the factor is ``M = R A`` from ``B = QR``.  Here
    the representation matrix is already such a factor, so it is used directly.
    ``docs/notes/frobenius_bw_generalization.md`` carries the full derivation.

    **The two inputs need not have the same number of rows, and row order does not
    matter.**  Σ = XᵀX is invariant to permuting the rows of X, so unlike
    :class:`~src.metrics.cka.CKADistanceMetric` and
    :class:`~src.metrics.frobenius.FrobeniusDistanceMetric` this metric makes no
    assumption that row *i* means the same thing in both models.  It compares the
    second moments of two clouds of vectors, not two indexed lists of them.  What it
    does require is a shared feature dimension, since that is what the covariance
    lives in.

    Not scale-invariant, deliberately: X → cX scales the distance by |c|, because
    the size of a covariance is part of what BW measures.  Pair it with a
    renormalized representation if that is not wanted.
    """

    @property
    def metric_name(self) -> str:
        return "bures_wasserstein"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        ma = np.atleast_2d(a.matrix.astype(np.float64))
        mb = np.atleast_2d(b.matrix.astype(np.float64))

        if ma.shape[1] != mb.shape[1]:
            raise ValueError(
                f"feature dimension mismatch: {ma.shape} vs {mb.shape}. "
                "Bures-Wasserstein compares covariances XᵀX, so the two "
                "representations must share a feature dimension (row counts may "
                "differ)."
            )

        # min(n_a, n_b) singular values of the cross matrix; no d×d product formed.
        cross = np.linalg.svd(ma @ mb.T, compute_uv=False)

        d2 = (
            float(np.sum(ma * ma))
            + float(np.sum(mb * mb))
            - 2.0 * float(cross.sum())
        )
        # d² is non-negative in exact arithmetic; cancellation can put it a few
        # ulps below zero when the two covariances nearly coincide.
        return float(np.sqrt(max(d2, 0.0)))
