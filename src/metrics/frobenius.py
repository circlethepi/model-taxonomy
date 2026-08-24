# TODO (full-pipeline): assumes ModelRepresentation.matrix is a uniform 2D array
# (N_layers × d). Needs updating before structural taxonomy representations can
# flow through this path (rows may have variable natural lengths once truncation
# is removed; see src/taxonomy/structural.py and memory/project_full_pipeline_metrics.md).
from __future__ import annotations

import numpy as np

from src.core.protocols import DistanceMetric
from src.core.representation import ModelRepresentation


class FrobeniusDistanceMetric(DistanceMetric):
    """Frobenius norm of the difference between two representation matrices.

    With normalize=True (default), each row is L2-normalized before subtraction,
    making the metric invariant to embedding scale.
    The result is divided by sqrt(N) so it does not grow with the query set size.

    ``normalize=False`` is plain row-wise Euclidean distance, and reports itself
    under the separate name ``"euclidean"`` because the two are genuinely
    different measurements and a stored collection must not be ambiguous about
    which it holds.  The distinction matters most alongside the centering
    transforms in :mod:`src.analysis.surrogates`: Euclidean distance is
    translation-invariant, so centering is exactly a no-op for it, whereas the
    normalized form discards each row's magnitude — and after centering the
    magnitude is precisely the *distance from the fleet centroid*, which is
    usually the largest part of what distinguishes two models.  Pairing a
    centered representation with a scale-invariant metric therefore throws away
    what the centering just exposed.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    @property
    def metric_name(self) -> str:
        return "frobenius" if self.normalize else "euclidean"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        if a.n_queries != b.n_queries:
            raise ValueError(
                f"Query count mismatch: {a.n_queries} vs {b.n_queries}. "
                "Both representations must use the same query set."
            )
        ma = a.matrix.astype(np.float64)
        mb = b.matrix.astype(np.float64)

        if self.normalize:
            ma = _row_normalize(ma)
            mb = _row_normalize(mb)

        diff = ma - mb
        return float(np.linalg.norm(diff, "fro") / np.sqrt(a.n_queries))


def _row_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return m / norms
