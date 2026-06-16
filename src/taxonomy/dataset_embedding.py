from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.core.protocols import Taxonomy, Embedder, ModelID
from src.core.representation import ModelRepresentation
from src.datasets.recipe import DatasetRecipe
from src.datasets.class_recipe import ClassAwareDatasetRecipe
from src.datasets.mixed_dataset import MixedDataset, ClassMixedDataset

if TYPE_CHECKING:
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

_AnyRecipe = DatasetRecipe | ClassAwareDatasetRecipe


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
        datasets: dict[str, tuple[_AnyRecipe, int]],
        representation: Literal["matrix", "gram"] = "matrix",
        cache: DatasetEmbeddingCache | None = None,
        seed: int = 42,
        hf_token: str | None = None,
    ) -> None:
        """
        Args:
            embedder: Embedder to call per element.
            datasets: Mapping from ``recipe_id`` → ``(recipe, n_samples)``.
                      Use ``from_recipes()`` to build this from a plain list.
            representation: ``"matrix"`` or ``"gram"``.
            cache: Optional ``DatasetEmbeddingCache`` for persistence.
            seed: Random seed for dataset shuffling.
            hf_token: HuggingFace API token for gated datasets.
        """
        self.embedder = embedder
        self._datasets = datasets
        self.representation = representation
        self.cache = cache
        self.seed = seed
        self.hf_token = hf_token

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_recipes(
        cls,
        recipes: list[_AnyRecipe],
        n_samples: int,
        embedder: Embedder,
        representation: Literal["matrix", "gram"] = "matrix",
        cache: DatasetEmbeddingCache | None = None,
        seed: int = 42,
        hf_token: str | None = None,
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

        ``model_id`` here is a recipe hash returned by ``recipe_ids()``.
        """
        if model_id not in self._datasets:
            raise KeyError(
                f"Recipe ID {model_id!r} not registered in this taxonomy. "
                f"Known IDs: {list(self._datasets.keys())}"
            )

        recipe, n_samples = self._datasets[model_id]

        # Cache lookup
        if self.cache is not None:
            from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

            emb_hash = DatasetEmbeddingCache.embedder_hash(
                self.embedder.config_dict(), self.representation, n_samples
            )
            if self.cache.exists(model_id, emb_hash):
                return self.cache.load(model_id, emb_hash)

        # Build MixedDataset / ClassMixedDataset and sample texts
        if isinstance(recipe, ClassAwareDatasetRecipe):
            mixed = ClassMixedDataset(
                recipe, total_samples=n_samples, seed=self.seed, hf_token=self.hf_token
            )
        else:
            mixed = MixedDataset(
                recipe, total_samples=n_samples, seed=self.seed, hf_token=self.hf_token
            )

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

        # Persist to cache
        if self.cache is not None:
            self.cache.save(
                recipe=recipe,
                rep=rep,
                embedder_config=self.embedder.config_dict(),
                representation=self.representation,
                n_samples=n_samples,
            )

        return rep
