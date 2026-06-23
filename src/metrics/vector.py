from __future__ import annotations

import numpy as np

from src.core.protocols import DistanceMetric
from src.core.representation import ModelRepresentation


class CosineDistanceMetric(DistanceMetric):
    """Cosine distance = 1 - cosine_similarity.

    Scale-invariant; works regardless of whether embeddings are pre-normalized.
    Natural companion to the ``"mean"`` representation in DatasetEmbeddingTaxonomy,
    but can be applied to any flat representation.

    Both matrices are flattened to 1-D before comparison, so this metric works
    with any shape produced by the taxonomy (1×d mean, N×d matrix, etc.).
    """

    @property
    def metric_name(self) -> str:
        return "cosine"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        va = a.matrix.flatten().astype(np.float64)
        vb = b.matrix.flatten().astype(np.float64)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 1.0
        cos_sim = np.dot(va, vb) / (norm_a * norm_b)
        return float(1.0 - np.clip(cos_sim, -1.0, 1.0))


class DotProductDistanceMetric(DistanceMetric):
    """Distance based on dot product similarity: 1 - dot(a, b).

    Assumes pre-normalized embeddings (i.e., ``normalize_embeddings=True`` in the
    embedder config).  For normalized vectors this is equivalent to cosine distance.
    Both matrices are flattened to 1-D before comparison.
    """

    @property
    def metric_name(self) -> str:
        return "dot_product"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        va = a.matrix.flatten().astype(np.float64)
        vb = b.matrix.flatten().astype(np.float64)
        return float(1.0 - np.clip(np.dot(va, vb), -1.0, 1.0))
