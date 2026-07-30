"""Top-level exports, resolved lazily.

Every name below is importable exactly as before — ``from src import DiskCache``,
``src.MDSGeometry``, ``from src import *`` — but the module that defines it is not
imported until the name is first touched.

This matters because the package spans both ends of the dependency graph.  Reading a
cached distance matrix needs numpy and nothing else; ``BehavioralTaxonomy`` needs
transformers, ``SentenceTransformerEmbedder`` needs sentence-transformers, and
``UMAPGeometry`` needs umap/numba.  While these were imported eagerly, *any* ``import
src.x`` paid for all of them — measured at ~470 MB and ~7 s before a single line of
work, which made light scripts (``scripts/check_analysis.py``, notebook first cells)
far heavier than what they actually do.

PEP 562 module ``__getattr__`` is the mechanism; ``_EXPORTS`` is the name → module map.
Adding an export means adding it here, not writing an import statement.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

#: submodule (relative to ``src``) → the names it provides.
_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "core.protocols": ("ModelID",),
    "core.representation": ("ModelRepresentation",),
    "core.distance": ("DistanceMatrix",),
    "core.geometry": ("GeometryResult",),
    "core.analysis": ("TaxonomyAnalysis", "TaxonomyAnalyzer", "ModelTaxonomyProfile"),
    "models.collection": ("ModelCollection",),
    "taxonomy.behavioral": ("BehavioralTaxonomy",),
    "taxonomy.functional": ("FunctionalTaxonomy",),
    "taxonomy.structural": ("StructuralTaxonomy",),
    "taxonomy.training_data": ("TrainingDataTaxonomy",),
    "taxonomy.dataset_embedding": ("DatasetEmbeddingTaxonomy",),
    "embedders.hidden_state": ("HiddenStateEmbedder",),
    "embedders.sentence_transformer": ("SentenceTransformerEmbedder",),
    "metrics.frobenius": ("FrobeniusDistanceMetric",),
    "metrics.cka": ("CKADistanceMetric",),
    "geometry_methods.mds": ("MDSGeometry",),
    "geometry_methods.pca": ("PCAGeometry",),
    "geometry_methods.umap": ("UMAPGeometry",),
    "compute.local": ("LocalBackend",),
    "compute.slurm": ("SlurmBackend",),
    "cache.disk": ("DiskCache",),
    "cache.dataset_embedding_cache": ("DatasetEmbeddingCache",),
    "datasets": (
        "DatasetEntry",
        "DatasetRecipe",
        "ClassDatasetEntry",
        "ClassAwareDatasetRecipe",
        "MixedDataset",
        "ClassMixedDataset",
    ),
    "analysis": (
        "as_distance_matrix",
        "lora_distance_matrix",
        "fit_geometry",
        "save_collection",
        "recipe_id_for",
        "relabel",
        "match_models",
        "matrix_correlation",
        "mantel_test",
        "correlation_table",
        "procrustes_compare",
        "per_point_residuals",
        "protest",
        "align_to_reference",
        "point_dispersion",
        "kruskal_stress",
        "shepard",
        "SimplexProjection",
        "barycentric",
        "compare_simplices",
        "anchor_weight_vs_truth",
    ),
}

#: exported name → submodule that defines it.
_EXPORTS: dict[str, str] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Import the defining submodule on first access (PEP 562)."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'src' has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f"src.{module}"), name)
    globals()[name] = value  # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


# Type checkers and IDEs cannot follow __getattr__, so give them the real thing.
# This block never runs.
if TYPE_CHECKING:
    from src.analysis import (
        SimplexProjection,
        align_to_reference,
        anchor_weight_vs_truth,
        as_distance_matrix,
        barycentric,
        compare_simplices,
        correlation_table,
        fit_geometry,
        kruskal_stress,
        lora_distance_matrix,
        mantel_test,
        match_models,
        matrix_correlation,
        per_point_residuals,
        point_dispersion,
        procrustes_compare,
        protest,
        recipe_id_for,
        relabel,
        save_collection,
        shepard,
    )
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache
    from src.cache.disk import DiskCache
    from src.compute.local import LocalBackend
    from src.compute.slurm import SlurmBackend
    from src.core.analysis import (
        ModelTaxonomyProfile,
        TaxonomyAnalysis,
        TaxonomyAnalyzer,
    )
    from src.core.distance import DistanceMatrix
    from src.core.geometry import GeometryResult
    from src.core.protocols import ModelID
    from src.core.representation import ModelRepresentation
    from src.datasets import (
        ClassAwareDatasetRecipe,
        ClassDatasetEntry,
        ClassMixedDataset,
        DatasetEntry,
        DatasetRecipe,
        MixedDataset,
    )
    from src.embedders.hidden_state import HiddenStateEmbedder
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder
    from src.geometry_methods.mds import MDSGeometry
    from src.geometry_methods.pca import PCAGeometry
    from src.geometry_methods.umap import UMAPGeometry
    from src.metrics.cka import CKADistanceMetric
    from src.metrics.frobenius import FrobeniusDistanceMetric
    from src.models.collection import ModelCollection
    from src.taxonomy.behavioral import BehavioralTaxonomy
    from src.taxonomy.dataset_embedding import DatasetEmbeddingTaxonomy
    from src.taxonomy.functional import FunctionalTaxonomy
    from src.taxonomy.structural import StructuralTaxonomy
    from src.taxonomy.training_data import TrainingDataTaxonomy
