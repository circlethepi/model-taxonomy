from __future__ import annotations

import numpy as np

from src.core.protocols import GeometryMethod
from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult


class PCAGeometry(GeometryMethod):
    """PCA on the double-centered distance matrix (equivalent to classical MDS).

    Uses eigendecomposition of the double-centered Gram matrix derived from
    the distance matrix. Faster than iterative MDS for large collections.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components

    @property
    def method_name(self) -> str:
        return "pca"

    def fit(self, distance_matrix: DistanceMatrix) -> GeometryResult:
        D = distance_matrix.matrix.astype(np.float64)
        n = D.shape[0]

        D_sq = D ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J

        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # eigh returns ascending order; reverse for descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        pos_mask = eigenvalues > 0
        k = min(self.n_components, pos_mask.sum())
        coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0))

        if k < self.n_components:
            padding = np.zeros((n, self.n_components - k))
            coords = np.hstack([coords, padding])

        explained = eigenvalues[:self.n_components] / eigenvalues[pos_mask].sum()

        return GeometryResult(
            coordinates=coords.astype(np.float32),
            model_ids=distance_matrix.model_ids,
            method=self.method_name,
            taxonomy=distance_matrix.taxonomy,
            n_components=self.n_components,
            stress=None,
            metadata={"explained_variance_ratio": explained.tolist()},
        )
