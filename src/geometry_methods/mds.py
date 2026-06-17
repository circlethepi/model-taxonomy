from __future__ import annotations

import numpy as np

from src.core.protocols import GeometryMethod
from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult


class MDSGeometry(GeometryMethod):
    """Multidimensional Scaling via sklearn.manifold.MDS.

    metric=True  → classical (metric) MDS: preserves distances faithfully.
    metric=False → non-metric (ordinal) MDS (Kruskal): preserves rank order.
    Stress measures goodness-of-fit; lower is better.
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        max_iter: int = 300,
        n_init: int = 4,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.metric = metric
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    @property
    def method_name(self) -> str:
        return "mds"

    def fit(self, distance_matrix: DistanceMatrix) -> GeometryResult:
        from sklearn.manifold import MDS

        mds = MDS(
            n_components=self.n_components,
            metric_mds=self.metric,
            metric="precomputed",
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
            normalized_stress="auto",
            init="random",
        )
        coords = mds.fit_transform(distance_matrix.matrix)

        return GeometryResult(
            coordinates=coords.astype(np.float32),
            model_ids=distance_matrix.model_ids,
            method=self.method_name,
            taxonomy=distance_matrix.taxonomy,
            n_components=self.n_components,
            stress=float(mds.stress_),
            metadata={"metric": self.metric},
        )
