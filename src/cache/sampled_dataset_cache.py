"""Persistent cache for sampled dataset rows, keyed by (recipe_hash, n_samples, seed).

Stores the post-shuffle/filter list[dict] produced by MixedDataset / ClassMixedDataset
so that downstream steps (embedding, fine-tuning) can skip HuggingFace re-loading when
the same recipe+seed+n_samples combination is requested again — even from a different
experiment.

Directory layout::

    cache_root/sampled_datasets/{recipe_hash}/{n_samples}_{seed:010d}.json

The JSON file is a list of row dicts as produced by the dataset's _load() method.
Writes are atomic (temp-file rename), consistent with other cache classes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class SampledDatasetCache:
    """File-backed cache for sampled dataset rows.

    Keys are ``(recipe_hash, n_samples, seed)`` triples. Values are lists of
    row dicts (all original columns) as returned by ``MixedDataset._load()``.
    Calling code is responsible for extracting the desired text field.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, recipe_hash: str, n_samples: int, seed: int) -> Path:
        return self.root / recipe_hash / f"{n_samples}_{seed:010d}.json"

    def exists(self, recipe_hash: str, n_samples: int, seed: int) -> bool:
        return self._path(recipe_hash, n_samples, seed).exists()

    def get(self, recipe_hash: str, n_samples: int, seed: int) -> list[dict] | None:
        path = self._path(recipe_hash, n_samples, seed)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def put(self, recipe_hash: str, n_samples: int, seed: int, samples: list[dict]) -> None:
        path = self._path(recipe_hash, n_samples, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(samples))
        os.replace(tmp, path)
