from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .protocols import ModelID
from .distance import DistanceMatrix


@dataclass
class GeometryResult:
    """Low-dimensional coordinate embedding of a model collection."""

    coordinates: np.ndarray
    model_ids: list[ModelID]
    method: str
    taxonomy: str
    n_components: int
    stress: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.model_ids)
        if self.coordinates.shape != (n, self.n_components):
            raise ValueError(
                f"coordinates shape {self.coordinates.shape} expected ({n}, {self.n_components})"
            )

    def nearest_neighbors(self, model_id: ModelID, k: int = 3) -> list[ModelID]:
        """Return k nearest neighbors in coordinate space (Euclidean)."""
        idx = self.model_ids.index(model_id)
        dists = np.linalg.norm(self.coordinates - self.coordinates[idx], axis=1)
        dists[idx] = np.inf
        neighbor_indices = np.argsort(dists)[:k]
        return [self.model_ids[i] for i in neighbor_indices]

    def to_networkx(self, distance_matrix: DistanceMatrix | None = None) -> nx.Graph:
        """Build a graph where nodes are models; edge weights are distances.

        If distance_matrix is provided, edge weights come from it.
        Otherwise, pairwise Euclidean distances in coordinate space are used.
        """
        g = nx.Graph()
        g.add_nodes_from(self.model_ids)
        n = len(self.model_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if distance_matrix is not None:
                    w = distance_matrix[(self.model_ids[i], self.model_ids[j])]
                else:
                    w = float(np.linalg.norm(self.coordinates[i] - self.coordinates[j]))
                g.add_edge(self.model_ids[i], self.model_ids[j], weight=w)
        return g

    def save(self, path: Path) -> None:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_ids": self.model_ids,
            "method": self.method,
            "taxonomy": self.taxonomy,
            "n_components": self.n_components,
            "stress": self.stress,
            "metadata": self.metadata,
        }
        meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        save_file(
            {
                "coordinates": np.ascontiguousarray(self.coordinates.astype(np.float32)),
                "_meta_json": meta_bytes,
            },
            str(path / "geometry.safetensors"),
        )

    @classmethod
    def load(cls, path: Path) -> "GeometryResult":
        from safetensors.numpy import load_file

        path = Path(path)
        tensors = load_file(str(path / "geometry.safetensors"))
        coordinates = tensors["coordinates"]
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return cls(coordinates=coordinates, **meta)
