from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from src.core.protocols import Taxonomy, DistanceMetric, ComputeBackend, ModelID
from src.core.representation import ModelRepresentation


class LocalBackend(ComputeBackend):
    """Runs extraction and distance computation on the local machine.

    n_jobs=1  → sequential for-loop (required when models need a single GPU,
                since only one model can be in GPU memory at a time).
    n_jobs>1  → joblib.Parallel with loky backend; useful for CPU-bound
                distance computation or CPU-only models.
    n_jobs=-1 → use all available CPU cores.
    """

    def __init__(self, n_jobs: int = 1) -> None:
        self.n_jobs = n_jobs

    def map_extract(
        self,
        taxonomy: Taxonomy,
        model_ids: Sequence[ModelID],
    ) -> list[ModelRepresentation]:
        if self.n_jobs == 1:
            return [taxonomy.extract(mid) for mid in model_ids]

        from joblib import Parallel, delayed

        return Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(taxonomy.extract)(mid) for mid in model_ids
        )

    def map_distances(
        self,
        metric: DistanceMetric,
        representations: Sequence[ModelRepresentation],
    ) -> np.ndarray:
        reps = list(representations)
        n = len(reps)
        matrix = np.zeros((n, n), dtype=np.float64)

        pairs = list(combinations(range(n), 2))

        if self.n_jobs == 1:
            for i, j in pairs:
                d = metric.compute(reps[i], reps[j])
                matrix[i, j] = d
                matrix[j, i] = d
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(metric.compute)(reps[i], reps[j]) for i, j in pairs
            )
            for (i, j), d in zip(pairs, results):
                matrix[i, j] = d
                matrix[j, i] = d

        return matrix
