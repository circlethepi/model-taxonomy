from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from .representation import ModelRepresentation
    from .distance import DistanceMatrix
    from .geometry import GeometryResult

ModelID = str


class Taxonomy(ABC):
    """Step 1: extract a matrix representation for a single model."""

    @abstractmethod
    def extract(self, model_id: ModelID) -> "ModelRepresentation":
        """Load the model, extract its representation, unload it, return result."""
        ...

    @property
    @abstractmethod
    def taxonomy_name(self) -> str: ...

    @abstractmethod
    def config_dict(self) -> dict[str, Any]:
        """Return a deterministic dict of all config parameters for cache keying."""
        ...


class Embedder(ABC):
    """Converts raw model output into a fixed-size vector."""

    @abstractmethod
    def embed(self, model_output: Any, probe: str) -> np.ndarray:
        """Return a 1-D float32 array of shape (d,)."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int | None:
        """Return embedding dimension, or None if unknown until first call."""
        ...

    @abstractmethod
    def config_dict(self) -> dict[str, Any]:
        """Return a deterministic dict of all config parameters for cache keying."""
        ...


class DistanceMetric(ABC):
    """Step 2: scalar distance between two representations."""

    @abstractmethod
    def compute(self, a: "ModelRepresentation", b: "ModelRepresentation") -> float:
        """Return a non-negative float; 0 means identical."""
        ...

    @property
    @abstractmethod
    def metric_name(self) -> str: ...


class GeometryMethod(ABC):
    """Step 3: embed a DistanceMatrix into a low-dimensional coordinate space."""

    @abstractmethod
    def fit(self, distance_matrix: "DistanceMatrix") -> "GeometryResult":
        """Return coordinates of shape (N, n_components)."""
        ...

    @property
    @abstractmethod
    def method_name(self) -> str: ...


class ComputeBackend(ABC):
    """Schedules and executes extraction and distance computation across a model collection."""

    @abstractmethod
    def map_extract(
        self,
        taxonomy: Taxonomy,
        model_ids: Sequence[ModelID],
    ) -> list["ModelRepresentation"]: ...

    @abstractmethod
    def map_distances(
        self,
        metric: DistanceMetric,
        representations: Sequence["ModelRepresentation"],
    ) -> np.ndarray:
        """Return symmetric NxN distance matrix."""
        ...
