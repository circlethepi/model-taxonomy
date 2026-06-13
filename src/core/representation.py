from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocols import ModelID


def _config_hash(config: dict[str, Any]) -> str:
    """Deterministic SHA-256 hash of a config dict, truncated to 16 hex chars."""
    payload = repr(sorted(config.items())).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _cache_key(model_id: ModelID, config_hash: str) -> str:
    payload = f"{model_id}::{config_hash}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass
class ModelRepresentation:
    """Matrix representation of a model at a given taxonomy level.

    matrix shape: (N_probes, d) where d is the embedding dimension.
    """

    model_id: ModelID
    taxonomy: str
    matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_key: str = ""

    def __post_init__(self) -> None:
        if self.matrix.dtype != np.float32:
            self.matrix = self.matrix.astype(np.float32)

    @classmethod
    def create(
        cls,
        model_id: ModelID,
        taxonomy: str,
        matrix: np.ndarray,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> "ModelRepresentation":
        ch = _config_hash(config)
        key = _cache_key(model_id, ch)
        return cls(
            model_id=model_id,
            taxonomy=taxonomy,
            matrix=matrix.astype(np.float32),
            metadata=metadata or {},
            cache_key=key,
        )

    @property
    def n_probes(self) -> int:
        return self.matrix.shape[0]

    @property
    def embedding_dim(self) -> int:
        return self.matrix.shape[1]
