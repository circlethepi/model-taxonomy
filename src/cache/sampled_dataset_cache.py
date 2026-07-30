"""Persistent cache for sampled dataset rows, keyed by (recipe_hash, n_samples, seed).

Stores the post-shuffle/filter list[dict] produced by MixedDataset / ClassMixedDataset
so that downstream steps (embedding, fine-tuning) can skip HuggingFace re-loading when
the same recipe+seed+n_samples combination is requested again — even from a different
experiment.

Directory layout::

    cache_root/01_datasets/{recipe_hash}/
        recipe.json                     ← the mixing spec this hash identifies
        {n_samples}_{seed:010d}.json    ← one sampled draw

The sample files are lists of row dicts as produced by the dataset's _load() method.
Writes are atomic (temp-file rename), consistent with other cache classes.

``recipe.json`` makes this the hash-indexed home for recipes.  Previously the only
way to resolve a ``recipe_hash`` to its recipe was to read it out of the
dataset-embedding cache, which fails for any recipe sampled but never embedded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class SampledDatasetCache:
    """File-backed cache for sampled dataset rows and their recipes.

    Sample keys are ``(recipe_hash, n_samples, seed)`` triples; values are lists of
    row dicts (all original columns) as returned by ``MixedDataset._load()``.
    Calling code is responsible for extracting the desired text field.

    Takes the cache root and appends its own directory name, the same contract as
    :class:`~src.cache.lora_cache.LoRACache` and the other cache classes.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root) / "01_datasets"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, recipe_hash: str, n_samples: int, seed: int) -> Path:
        return self.root / recipe_hash / f"{n_samples}_{seed:010d}.json"

    def _recipe_path(self, recipe_hash: str) -> Path:
        return self.root / recipe_hash / "recipe.json"

    def exists(self, recipe_hash: str, n_samples: int, seed: int) -> bool:
        return self._path(recipe_hash, n_samples, seed).exists()

    def get(self, recipe_hash: str, n_samples: int, seed: int) -> list[dict] | None:
        path = self._path(recipe_hash, n_samples, seed)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, recipe_hash: str, n_samples: int, seed: int, samples: list[dict]) -> None:
        self._write(self._path(recipe_hash, n_samples, seed), json.dumps(samples))

    # ------------------------------------------------------------------
    # Recipes
    # ------------------------------------------------------------------

    def get_recipe(self, recipe_hash: str) -> dict | None:
        """The recipe dict this hash identifies, or None if it was never written."""
        path = self._recipe_path(recipe_hash)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put_recipe(self, recipe_hash: str, recipe: dict) -> None:
        """Mirror a recipe next to its samples.

        Idempotent — an existing recipe.json is left alone, since the hash pins the
        content and rewriting it would only churn the file.
        """
        path = self._recipe_path(recipe_hash)
        if path.exists():
            return
        self._write(path, json.dumps(recipe, indent=2))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _write(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)
