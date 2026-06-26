from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterator

import numpy as np

from src.datasets.recipe import DatasetRecipe
from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry

if TYPE_CHECKING:
    from src.datasets.recipe import DatasetRecipe as _AnyRecipe


def _allocate_counts(weights: list[float], total: int) -> list[int]:
    """Distribute *total* samples across buckets by normalized weights.

    Uses largest-remainder (Hamilton) method so the counts sum exactly to
    *total* without rounding drift.
    """
    exact = [w * total for w in weights]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    # Assign leftover slots to the buckets with the largest fractional parts
    fracs = [(exact[i] - floors[i], i) for i in range(len(weights))]
    fracs.sort(reverse=True)
    for _, i in fracs[:remainder]:
        floors[i] += 1
    return floors


class MixedDataset:
    """Weighted mixture of HuggingFace datasets.

    Datasets are lazy-loaded on first access; HuggingFace's own disk cache
    handles repeated loads.  Samples are drawn deterministically from each
    dataset using *seed*, then interleaved in a shuffled order.

    Usage::

        recipe = DatasetRecipe("qa_mix", [
            DatasetEntry("squad", weight=2.0, text_field="question"),
            DatasetEntry("trivia_qa", subset="unfiltered", weight=1.0,
                         text_field="question"),
        ])
        mixed = MixedDataset(recipe, total_samples=300, seed=0)

        queries = mixed.to_queries()          # list[str] for inference
        for sample in mixed.for_finetuning():  # full dicts for training
            ...
    """

    def __init__(
        self,
        recipe: DatasetRecipe,
        total_samples: int,
        seed: int = 42,
        hf_token: str | None = None,
    ) -> None:
        self.recipe = recipe
        self.total_samples = total_samples
        self.seed = seed
        self.hf_token = hf_token
        self._samples: list[dict] | None = None

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        from datasets import load_dataset  # type: ignore[import]

        counts = _allocate_counts(self.recipe.normalized_weights, self.total_samples)
        all_samples: list[dict] = []

        for entry, count in zip(self.recipe.datasets, counts):
            if count == 0:
                continue
            ds = load_dataset(
                entry.dataset_id,
                entry.subset,
                split=entry.split,
                token=self.hf_token,
            )
            ds = ds.shuffle(seed=self.seed)
            if count > len(ds):
                warnings.warn(
                    f"MixedDataset: '{entry.dataset_id}' has only {len(ds)} rows "
                    f"but {count} were requested; capping to {len(ds)}.",
                    UserWarning, stacklevel=4,
                )
            count = min(count, len(ds))
            ds = ds.select(range(count))
            for row in ds:
                all_samples.append(dict(row))

        # Shuffle the merged list deterministically
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(all_samples)).tolist()
        return [all_samples[i] for i in idx]

    def _ensure_loaded(self) -> list[dict]:
        if self._samples is None:
            self._samples = self._load()
        return self._samples

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_queries(self, n: int | None = None) -> list[str]:
        """Return text strings suitable for use as inference queries.

        If *n* is given, return the first *n* samples (must be ≤ total_samples).
        The text is taken from each entry's configured ``text_field``.
        """
        samples = self._ensure_loaded()
        if n is not None:
            if n > len(samples):
                raise ValueError(
                    f"Requested {n} queries but only {len(samples)} samples available."
                )
            samples = samples[:n]

        # Build a field lookup keyed by dataset_id for fast access
        field_map = {e.dataset_id: e.text_field for e in self.recipe.datasets}
        default_field = self.recipe.datasets[0].text_field

        texts: list[str] = []
        for row in samples:
            # Try each configured text_field in order of recipe entries
            text = None
            for tf in field_map.values():
                if tf in row:
                    text = str(row[tf])
                    break
            texts.append(text if text is not None else str(row.get(default_field, "")))
        return texts

    def for_finetuning(self) -> Iterator[dict]:
        """Yield sample dicts (all original columns) for fine-tuning."""
        yield from self._ensure_loaded()

    def recipe_metadata_dict(self) -> dict:
        """Return a dict suitable for embedding in ModelRepresentation.metadata."""
        return {"dataset_recipe": self.recipe.to_dict()}

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __iter__(self) -> Iterator[dict]:
        yield from self._ensure_loaded()


class CachedMixedDataset:
    """Wraps a pre-loaded list[dict] from SampledDatasetCache with the same interface
    as MixedDataset / ClassMixedDataset, so it can be used as a drop-in replacement.
    """

    def __init__(self, samples: list[dict], recipe: DatasetRecipe | ClassAwareDatasetRecipe) -> None:
        self._samples = samples
        self.recipe = recipe
        self.total_samples = len(samples)
        self.seed: int | None = None
        self.hf_token: str | None = None

    def to_queries(self, n: int | None = None) -> list[str]:
        samples = self._samples[:n] if n is not None else self._samples
        field_map = {e.dataset_id: e.text_field for e in self.recipe.datasets}
        default_field = self.recipe.datasets[0].text_field
        texts: list[str] = []
        for row in samples:
            text = None
            for tf in field_map.values():
                if tf in row:
                    text = str(row[tf])
                    break
            texts.append(text if text is not None else str(row.get(default_field, "")))
        return texts

    def for_finetuning(self) -> Iterator[dict]:
        yield from self._samples

    def recipe_metadata_dict(self) -> dict:
        return {"dataset_recipe": self.recipe.to_dict()}

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[dict]:
        yield from self._samples


class ClassMixedDataset:
    """Weighted mixture of HuggingFace datasets with per-class proportion control.

    Extends the simple mixing from ``MixedDataset`` with two additional knobs:

    - ``class_filter`` restricts which class values are included per dataset.
    - ``class_weights`` controls the proportion drawn from each class within a
      dataset (independent of the dataset-level mixing weight).

    Usage::

        recipe = ClassAwareDatasetRecipe("balanced_sentiment", [
            ClassDatasetEntry(
                "imdb", text_field="text", class_field="label",
                class_weights={0: 1.0, 1: 1.0},   # 50/50 positive/negative
            ),
        ])
        mixed = ClassMixedDataset(recipe, total_samples=200, seed=0)
        queries = mixed.to_queries()
    """

    def __init__(
        self,
        recipe: ClassAwareDatasetRecipe,
        total_samples: int,
        seed: int = 42,
        hf_token: str | None = None,
    ) -> None:
        self.recipe = recipe
        self.total_samples = total_samples
        self.seed = seed
        self.hf_token = hf_token
        self._samples: list[dict] | None = None

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_entry(self, entry: ClassDatasetEntry, count: int) -> list[dict]:
        """Load *count* samples from *entry*, respecting class weights/filter."""
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset(
            entry.dataset_id,
            entry.subset,
            split=entry.split,
            token=self.hf_token,
        )

        # Apply class_filter
        if entry.class_filter is not None:
            allowed = set(entry.class_filter)
            ds = ds.filter(lambda row: row[entry.class_field] in allowed)

        if len(ds) == 0:
            return []

        # Determine per-class normalized weights
        if entry.normalized_class_weights is not None:
            class_norm_w = entry.normalized_class_weights
        else:
            # Uniform over all present classes
            present = list({row[entry.class_field] for row in ds})
            class_norm_w = {c: 1.0 / len(present) for c in present}

        # Allocate count per class
        classes = list(class_norm_w.keys())
        w_list = [class_norm_w[c] for c in classes]
        per_class_counts = _allocate_counts(w_list, count)

        rng = np.random.default_rng(self.seed)
        samples: list[dict] = []
        for cls_val, cls_count in zip(classes, per_class_counts):
            if cls_count == 0:
                continue
            cls_ds = ds.filter(lambda row, cv=cls_val: row[entry.class_field] == cv)
            cls_ds = cls_ds.shuffle(seed=int(rng.integers(0, 2**31)))
            if cls_count > len(cls_ds):
                warnings.warn(
                    f"ClassMixedDataset: class {cls_val!r} in '{entry.dataset_id}' "
                    f"has only {len(cls_ds)} rows but {cls_count} were requested; "
                    f"capping to {len(cls_ds)}.",
                    UserWarning, stacklevel=4,
                )
            cls_count = min(cls_count, len(cls_ds))
            cls_ds = cls_ds.select(range(cls_count))
            for row in cls_ds:
                samples.append(dict(row))
        return samples

    def _load(self) -> list[dict]:
        counts = _allocate_counts(self.recipe.normalized_weights, self.total_samples)
        all_samples: list[dict] = []
        for entry, count in zip(self.recipe.datasets, counts):
            if count > 0:
                all_samples.extend(self._load_entry(entry, count))

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(all_samples)).tolist()
        return [all_samples[i] for i in idx]

    def _ensure_loaded(self) -> list[dict]:
        if self._samples is None:
            self._samples = self._load()
        return self._samples

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def to_queries(self, n: int | None = None) -> list[str]:
        """Return text strings suitable for use as inference queries."""
        samples = self._ensure_loaded()
        if n is not None:
            if n > len(samples):
                raise ValueError(
                    f"Requested {n} queries but only {len(samples)} samples available."
                )
            samples = samples[:n]

        field_map = {e.dataset_id: e.text_field for e in self.recipe.datasets}
        default_field = self.recipe.datasets[0].text_field

        texts: list[str] = []
        for row in samples:
            text = None
            for tf in field_map.values():
                if tf in row:
                    text = str(row[tf])
                    break
            texts.append(text if text is not None else str(row.get(default_field, "")))
        return texts

    def for_finetuning(self) -> Iterator[dict]:
        """Yield sample dicts for fine-tuning."""
        yield from self._ensure_loaded()

    def recipe_metadata_dict(self) -> dict:
        """Return a dict suitable for embedding in ModelRepresentation.metadata."""
        return {"dataset_recipe": self.recipe.to_dict()}

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __iter__(self) -> Iterator[dict]:
        yield from self._ensure_loaded()
