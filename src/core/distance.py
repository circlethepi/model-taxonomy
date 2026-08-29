from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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

    def reindex(self, model_ids: Sequence[ModelID]) -> "DistanceMatrix":
        """Return this matrix with its rows and columns in *model_ids* order.

        The guard on every read from the collection cache.  ``collection_key``
        sorts the model entries before hashing, so **row order is not part of the
        handle**: a matrix written in ``sort_by_mixture`` order and one written in
        cache-scan order land on the same key.  The stored ``model_ids`` are
        self-describing, so the data on disk is right either way — but handing a
        cache hit back unpermuted gives the caller a matrix whose labels no longer
        describe its rows, which is exactly the defect
        ``docs/notes/row_order_bug.md`` records, and through a cache it would be
        stable across runs rather than moving with the inputs.

        *model_ids* may name a **subset**, which is how a superset collection on
        disk becomes a legitimate hit: select the rows first, then use it.  An id
        this matrix does not hold raises, rather than being dropped — a silently
        shorter matrix is the same class of bug one step along.

        Permuting a symmetric distance matrix is exact.  A *geometry* is not:
        an MDS fit of a superset, restricted to some of its points, is not the
        fit of those points, and a fit is only defined up to rotation anyway.
        Coordinates are therefore refitted when the ids do not match, never
        permuted into place.
        """
        wanted = list(model_ids)
        if len(set(wanted)) != len(wanted):
            dupes = sorted({m for m in wanted if wanted.count(m) > 1})
            raise ValueError(
                f"reindex was asked for duplicate ids {dupes}: a distance matrix "
                "row is one model, so a repeated id has no meaning here."
            )
        position = {mid: i for i, mid in enumerate(self.model_ids)}
        missing = [m for m in wanted if m not in position]
        if missing:
            raise ValueError(
                f"reindex asked for {missing}, which this matrix does not hold. "
                f"It has {list(self.model_ids)}."
            )
        take = [position[m] for m in wanted]
        return DistanceMatrix(
            matrix=self.matrix[np.ix_(take, take)],
            model_ids=wanted,
            metric=self.metric,
            taxonomy=self.taxonomy,
        )

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
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"model_ids": self.model_ids, "metric": self.metric, "taxonomy": self.taxonomy}
        meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        save_file(
            {
                "matrix": np.ascontiguousarray(self.matrix.astype(np.float32)),
                "_meta_json": meta_bytes,
            },
            str(path / "distance_matrix.safetensors"),
        )

    @classmethod
    def load(cls, path: Path) -> "DistanceMatrix":
        from safetensors.numpy import load_file

        path = Path(path)
        tensors = load_file(str(path / "distance_matrix.safetensors"))
        matrix = tensors["matrix"].astype(np.float64)
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return cls(matrix=matrix, **meta)
