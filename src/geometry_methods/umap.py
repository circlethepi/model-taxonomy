from __future__ import annotations

import numpy as np

from src.core.protocols import GeometryMethod
from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult


class UMAPGeometry(GeometryMethod):
    """UMAP embedding from a precomputed distance matrix.

    Requires umap-learn (optional dependency: pip install model-taxonomy[umap]).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    @property
    def method_name(self) -> str:
        return "umap"

    def fit(self, distance_matrix: DistanceMatrix) -> GeometryResult:
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAPGeometry. "
                "Install it with: pip install umap-learn"
            )

        reducer = UMAP(
            n_components=self.n_components,
            n_neighbors=min(self.n_neighbors, len(distance_matrix.model_ids) - 1),
            min_dist=self.min_dist,
            metric="precomputed",
            random_state=self.random_state,
        )
        coords = reducer.fit_transform(distance_matrix.matrix)

        return GeometryResult(
            coordinates=coords.astype(np.float32),
            model_ids=distance_matrix.model_ids,
            method=self.method_name,
            taxonomy=distance_matrix.taxonomy,
            n_components=self.n_components,
            stress=None,
            metadata={"n_neighbors": self.n_neighbors, "min_dist": self.min_dist},
        )
