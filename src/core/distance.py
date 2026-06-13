from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .protocols import ModelID


@dataclass
class DistanceMatrix:
    """Pairwise distance matrix over a model collection at one taxonomy level."""

    matrix: np.ndarray
    model_ids: list[ModelID]
    metric: str
    taxonomy: str

    def __post_init__(self) -> None:
        n = len(self.model_ids)
        if self.matrix.shape != (n, n):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match {n} model_ids"
            )

    def __getitem__(self, pair: tuple[ModelID, ModelID]) -> float:
        a, b = pair
        i = self.model_ids.index(a)
        j = self.model_ids.index(b)
        return float(self.matrix[i, j])

    def sorted_neighbors(self, model_id: ModelID) -> list[tuple[ModelID, float]]:
        """Return all other models sorted by ascending distance to model_id."""
        idx = self.model_ids.index(model_id)
        row = self.matrix[idx]
        pairs = [
            (self.model_ids[j], float(row[j]))
            for j in range(len(self.model_ids))
            if j != idx
        ]
        return sorted(pairs, key=lambda x: x[1])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "matrix.npy", self.matrix)
        meta = {"model_ids": self.model_ids, "metric": self.metric, "taxonomy": self.taxonomy}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DistanceMatrix":
        path = Path(path)
        matrix = np.load(path / "matrix.npy")
        meta = json.loads((path / "meta.json").read_text())
        return cls(matrix=matrix, **meta)
