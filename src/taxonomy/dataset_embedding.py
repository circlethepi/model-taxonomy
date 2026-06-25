from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.core.protocols import Taxonomy, Embedder, ModelID
from src.core.representation import ModelRepresentation
from src.datasets.recipe import DatasetRecipe
from src.datasets.class_recipe import ClassAwareDatasetRecipe
from src.datasets.mixed_dataset import MixedDataset, ClassMixedDataset, CachedMixedDataset

if TYPE_CHECKING:
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache
    from src.cache.sampled_dataset_cache import SampledDatasetCache

_AnyRecipe = DatasetRecipe | ClassAwareDatasetRecipe

# datasets dict values may be 2-tuples (recipe, n_samples) for backward compat,
# or 3-tuples (recipe, n_samples, seed) when per-dataset seeds are needed.
_DatasetEntry = tuple[_AnyRecipe, int] | tuple[_AnyRecipe, int, int]


class DatasetEmbeddingTaxonomy(Taxonomy):
    """Compare datasets by embedding their elements with a chosen Embedder.

    For each registered recipe, ``extract()`` loads the dataset via
    ``MixedDataset`` / ``ClassMixedDataset``, embeds every text element, and
    returns a ``ModelRepresentation`` whose ``model_id`` is the recipe hash.

    Two representation modes are available:

    ``"matrix"``
        Raw ``(N_samples, d)`` embedding matrix.  Compatible with both
        ``FrobeniusDistanceMetric`` and ``CKADistanceMetric``.

    ``"gram"``
        Gram matrix ``E @ E.T`` upper-triangle, stored as
        ``(1, N_samples*(N_samples+1)//2)``.  Mirrors what
        ``FunctionalTaxonomy`` does per layer.  Use with
        ``FrobeniusDistanceMetric``.

    Because this class implements the ``Taxonomy`` protocol, it plugs directly
    into ``TaxonomyAnalyzer`` alongside ``BehavioralTaxonomy``,
    ``FunctionalTaxonomy``, and ``StructuralTaxonomy``.  Pass
    ``taxonomy.recipe_ids()`` as the ``model_ids`` argument to
    ``analyzer.fit()``.

    Embedder note: use ``SentenceTransformerEmbedder(use_generated_text=False)``
    so the embedder operates on the dataset text rather than model-generated
    continuations.
    """

    def __init__(
        self,
        embedder: Embedder,
        datasets: dict[str, _DatasetEntry],
        representation: Literal["matrix", "gram", "mean"] = "matrix",
        cache: DatasetEmbeddingCache | None = None,
        seed: int = 42,
        hf_token: str | None = None,
        sample_cache: SampledDatasetCache | None = None,
    ) -> None:
        """
        Args:
            embedder: Embedder to call per element.
            datasets: Mapping from ``recipe_id`` → ``(recipe, n_samples)`` or
                      ``(recipe, n_samples, seed)``.  The optional third element
                      overrides ``seed`` for that specific dataset; use this when
                      running multi-seed experiments.
            representation: ``"matrix"``, ``"gram"``, or ``"mean"``.
            cache: Optional ``DatasetEmbeddingCache`` for embedding persistence.
            seed: Global fallback seed for dataset shuffling (used when a dataset
                  entry does not carry its own seed).
            hf_token: HuggingFace API token for gated datasets.
            sample_cache: Optional ``SampledDatasetCache`` for raw-sample persistence.
                          Populated on first load; avoids HF re-loading on reruns.
        """
        self.embedder = embedder
        self._datasets = datasets
        self.representation = representation
        self.cache = cache
        self.seed = seed
        self.hf_token = hf_token
        self.sample_cache = sample_cache

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_recipes(
        cls,
        recipes: list[_AnyRecipe],
        n_samples: int,
        embedder: Embedder,
        representation: Literal["matrix", "gram", "mean"] = "matrix",
        cache: DatasetEmbeddingCache | None = None,
        seed: int = 42,
        hf_token: str | None = None,
        sample_cache: SampledDatasetCache | None = None,
    ) -> DatasetEmbeddingTaxonomy:
        """Convenience constructor: build from a list of recipes and a shared sample count.

        All recipes are registered under their ``recipe_hash()`` as the
        recipe ID.  Duplicate hashes (identical recipes) collapse to one entry.
        """
        datasets = {r.recipe_hash(): (r, n_samples) for r in recipes}
        return cls(
            embedder=embedder,
            datasets=datasets,
            representation=representation,
            cache=cache,
            seed=seed,
            hf_token=hf_token,
            sample_cache=sample_cache,
        )

    def recipe_ids(self) -> list[str]:
        """Return the list of recipe IDs to pass to ``TaxonomyAnalyzer.fit()``."""
        return list(self._datasets.keys())

    # ------------------------------------------------------------------
    # Taxonomy protocol
    # ------------------------------------------------------------------

    @property
    def taxonomy_name(self) -> str:
        return "dataset_embedding"

    def config_dict(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.config_dict(),
            "representation": self.representation,
            "seed": self.seed,
        }

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        """Embed all elements of the recipe and return a ``ModelRepresentation``.

        ``model_id`` here is a recipe ID returned by ``recipe_ids()``.
        """
        if model_id not in self._datasets:
            raise KeyError(
                f"Recipe ID {model_id!r} not registered in this taxonomy. "
                f"Known IDs: {list(self._datasets.keys())}"
            )

        entry = self._datasets[model_id]
        recipe, n_samples = entry[0], entry[1]
        seed = entry[2] if len(entry) > 2 else self.seed  # type: ignore[misc]

        # Use recipe_hash (not model_id) as the stable cache key — model_id may be
        # a dataset name rather than a hash when called from make_dataset_embedding_taxonomy.
        recipe_hash = recipe.recipe_hash()

        # Embedding cache lookup
        if self.cache is not None:
            from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

            emb_hash = DatasetEmbeddingCache.embedder_hash(
                self.embedder.config_dict(), self.representation, n_samples
            )
            if self.cache.exists(recipe_hash, emb_hash):
                return self.cache.load(recipe_hash, emb_hash)

        # Sample cache lookup / population
        mixed: MixedDataset | ClassMixedDataset | CachedMixedDataset
        if self.sample_cache is not None:
            cached_rows = self.sample_cache.get(recipe_hash, n_samples, seed)
            if cached_rows is not None:
                mixed = CachedMixedDataset(cached_rows, recipe)
            else:
                mixed = self._load_mixed(recipe, n_samples, seed)
                self.sample_cache.put(recipe_hash, n_samples, seed, list(mixed))
        else:
            mixed = self._load_mixed(recipe, n_samples, seed)

        texts = mixed.to_queries()

        # Embed each element
        vecs: list[np.ndarray] = []
        for text in texts:
            output = types.SimpleNamespace(generated_text=text)
            vecs.append(self.embedder.embed(output, query=text))
        E = np.stack(vecs).astype(np.float32)  # (N, d)

        # Build matrix in chosen mode
        if self.representation == "gram":
            G = (E @ E.T).astype(np.float32)  # (N, N)
            triu_idx = np.triu_indices(len(texts))
            matrix = G[triu_idx][np.newaxis, :]  # (1, N*(N+1)//2)
        elif self.representation == "mean":
            matrix = E.mean(axis=0, keepdims=True)  # (1, d)
        else:
            matrix = E  # (N, d)

        metadata: dict[str, Any] = {
            "recipe": recipe.to_dict(),
            "n_samples": n_samples,
            "representation": self.representation,
            "embedder": self.embedder.config_dict(),
        }

        rep = ModelRepresentation(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            metadata=metadata,
        )

        # Persist embeddings to cache
        if self.cache is not None:
            self.cache.save(
                recipe=recipe,
                rep=rep,
                embedder_config=self.embedder.config_dict(),
                representation=self.representation,
                n_samples=n_samples,
            )

        return rep

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_mixed(
        self,
        recipe: _AnyRecipe,
        n_samples: int,
        seed: int,
    ) -> MixedDataset | ClassMixedDataset:
        if isinstance(recipe, ClassAwareDatasetRecipe):
            return ClassMixedDataset(
                recipe, total_samples=n_samples, seed=seed, hf_token=self.hf_token
            )
        return MixedDataset(
            recipe, total_samples=n_samples, seed=seed, hf_token=self.hf_token
        )
