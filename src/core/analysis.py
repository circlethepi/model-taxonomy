from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .protocols import Taxonomy, DistanceMetric, GeometryMethod, ComputeBackend, ModelID
from .representation import ModelRepresentation
from .distance import DistanceMatrix
from .geometry import GeometryResult


@dataclass
class TaxonomyAnalysis:
    """Complete result for one taxonomy applied to a model collection."""

    taxonomy_name: str
    model_ids: list[ModelID]
    representations: list[ModelRepresentation]
    distance_matrix: DistanceMatrix
    geometry: GeometryResult | None = None

    def save(self, path: Path) -> None:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.distance_matrix.save(path / "distance_matrix")
        if self.geometry is not None:
            self.geometry.save(path / "geometry")
        for rep in self.representations:
            rep_dir = path / "representations" / rep.cache_key
            rep_dir.mkdir(parents=True, exist_ok=True)
            meta_payload = {
                "model_id": rep.model_id,
                "taxonomy": rep.taxonomy,
                "cache_key": rep.cache_key,
                "metadata": rep.metadata,
            }
            meta_bytes = np.frombuffer(
                json.dumps(meta_payload).encode("utf-8"), dtype=np.uint8
            )
            save_file(
                {
                    "matrix": np.ascontiguousarray(rep.matrix),
                    "_meta_json": meta_bytes,
                },
                str(rep_dir / "representation.safetensors"),
            )
        meta = {"taxonomy_name": self.taxonomy_name, "model_ids": self.model_ids}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TaxonomyAnalysis":
        from safetensors.numpy import load_file

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        distance_matrix = DistanceMatrix.load(path / "distance_matrix")
        geometry: GeometryResult | None = None
        if (path / "geometry").exists():
            geometry = GeometryResult.load(path / "geometry")
        representations: list[ModelRepresentation] = []
        rep_root = path / "representations"
        if rep_root.exists():
            for rep_dir in sorted(rep_root.iterdir()):
                if not rep_dir.is_dir():
                    continue
                tensors = load_file(str(rep_dir / "representation.safetensors"))
                matrix = tensors["matrix"]
                m = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
                representations.append(
                    ModelRepresentation(
                        model_id=m["model_id"],
                        taxonomy=m["taxonomy"],
                        matrix=matrix,
                        metadata=m.get("metadata", {}),
                        cache_key=m["cache_key"],
                    )
                )
        return cls(
            taxonomy_name=meta["taxonomy_name"],
            model_ids=meta["model_ids"],
            representations=representations,
            distance_matrix=distance_matrix,
            geometry=geometry,
        )


@dataclass
class ModelTaxonomyProfile:
    """A model collection with results across multiple taxonomy levels."""

    model_ids: list[ModelID]
    analyses: dict[str, TaxonomyAnalysis] = field(default_factory=dict)

    def get(self, taxonomy_name: str) -> TaxonomyAnalysis:
        if taxonomy_name not in self.analyses:
            raise KeyError(f"No analysis for taxonomy '{taxonomy_name}'")
        return self.analyses[taxonomy_name]

    def taxonomy_names(self) -> list[str]:
        return list(self.analyses.keys())

    def add(self, analysis: TaxonomyAnalysis) -> None:
        self.analyses[analysis.taxonomy_name] = analysis

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps({"model_ids": self.model_ids, "taxonomy_names": self.taxonomy_names()}, indent=2)
        )
        for name, analysis in self.analyses.items():
            analysis.save(path / name)

    @classmethod
    def load(cls, path: Path) -> "ModelTaxonomyProfile":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        analyses = {name: TaxonomyAnalysis.load(path / name) for name in meta["taxonomy_names"]}
        return cls(model_ids=meta["model_ids"], analyses=analyses)


class TaxonomyAnalyzer:
    """Runs the three-step pipeline (extraction → distances → geometry) for one taxonomy."""

    def __init__(
        self,
        taxonomy: Taxonomy,
        metric: DistanceMetric,
        backend: ComputeBackend,
        geometry_method: GeometryMethod | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.metric = metric
        self.backend = backend
        self.geometry_method = geometry_method

    def fit(self, model_ids: Sequence[ModelID]) -> TaxonomyAnalysis:
        model_ids = list(model_ids)

        representations = self.backend.map_extract(self.taxonomy, model_ids)

        shapes = {(r.n_queries, r.embedding_dim) for r in representations}
        if len(shapes) > 1:
            raise ValueError(
                f"Representations have inconsistent shapes: {shapes}. "
                "All models must use the same queries and embedder."
            )

        raw_matrix = self.backend.map_distances(self.metric, representations)
        dist_matrix = DistanceMatrix(
            matrix=raw_matrix,
            model_ids=[r.model_id for r in representations],
            metric=self.metric.metric_name,
            taxonomy=self.taxonomy.taxonomy_name,
        )

        geometry: GeometryResult | None = None
        if self.geometry_method is not None:
            geometry = self.geometry_method.fit(dist_matrix)

        return TaxonomyAnalysis(
            taxonomy_name=self.taxonomy.taxonomy_name,
            model_ids=model_ids,
            representations=representations,
            distance_matrix=dist_matrix,
            geometry=geometry,
        )
