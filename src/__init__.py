from src.core.protocols import ModelID
from src.core.representation import ModelRepresentation
from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult
from src.core.analysis import TaxonomyAnalysis, TaxonomyAnalyzer, ModelTaxonomyProfile

from src.models.collection import ModelCollection

from src.taxonomy.behavioral import BehavioralTaxonomy
from src.taxonomy.functional import FunctionalTaxonomy
from src.taxonomy.structural import StructuralTaxonomy
from src.taxonomy.training_data import TrainingDataTaxonomy

from src.embedders.hidden_state import HiddenStateEmbedder
from src.embedders.sentence_transformer import SentenceTransformerEmbedder

from src.metrics.frobenius import FrobeniusDistanceMetric
from src.metrics.cka import CKADistanceMetric

from src.geometry_methods.mds import MDSGeometry
from src.geometry_methods.pca import PCAGeometry
from src.geometry_methods.umap import UMAPGeometry

from src.compute.local import LocalBackend
from src.compute.slurm import SlurmBackend

from src.cache.disk import DiskCache
from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

from src.taxonomy.dataset_embedding import DatasetEmbeddingTaxonomy

from src.datasets import (
    DatasetEntry,
    DatasetRecipe,
    ClassDatasetEntry,
    ClassAwareDatasetRecipe,
    MixedDataset,
    ClassMixedDataset,
)

__all__ = [
    # core types
    "ModelID",
    "ModelRepresentation",
    "DistanceMatrix",
    "GeometryResult",
    "TaxonomyAnalysis",
    "TaxonomyAnalyzer",
    "ModelTaxonomyProfile",
    # model collection
    "ModelCollection",
    # taxonomies
    "BehavioralTaxonomy",
    "FunctionalTaxonomy",
    "StructuralTaxonomy",
    "TrainingDataTaxonomy",
    "DatasetEmbeddingTaxonomy",
    # embedders
    "HiddenStateEmbedder",
    "SentenceTransformerEmbedder",
    # distance metrics
    "FrobeniusDistanceMetric",
    "CKADistanceMetric",
    # geometry methods
    "MDSGeometry",
    "PCAGeometry",
    "UMAPGeometry",
    # compute backends
    "LocalBackend",
    "SlurmBackend",
    # cache
    "DiskCache",
    "DatasetEmbeddingCache",
    # datasets
    "DatasetEntry",
    "DatasetRecipe",
    "ClassDatasetEntry",
    "ClassAwareDatasetRecipe",
    "MixedDataset",
    "ClassMixedDataset",
]
