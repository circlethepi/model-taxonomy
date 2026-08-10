from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterator

import numpy as np

from src.datasets.recipe import DatasetRecipe
from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
from src.datasets._text_projection import row_text

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
        #: Source descriptors, one per recipe entry — see src.datasets.source_registry.
        self.sources: list[dict] | None = None
        #: ``(entry_index, row_index)`` per returned row, in the same order.  This is
        #: what SampledDatasetCache persists in place of the row text.
        self.source_indices: list[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        import warnings

        from src.datasets import source_registry

        weights = self.recipe.normalized_weights

        # Probe each entry's size and find the effective total that keeps all
        # entry proportions intact even when one dataset is data-limited.
        effective_total = self.total_samples
        sources: list[dict] = []
        for entry, w in zip(self.recipe.datasets, weights):
            ds = source_registry.get(
                entry.dataset_id, entry.subset, entry.split, token=self.hf_token
            )
            sources.append(
                source_registry.describe(ds, entry.dataset_id, entry.subset, entry.split)
            )
            if w > 0:
                effective_total = min(effective_total, int(len(ds) / w))

        if effective_total < self.total_samples:
            warnings.warn(
                f"MixedDataset: recipe capacity is {effective_total} "
                f"(requested {self.total_samples}); scaling entry counts "
                f"proportionally to maintain ratios.",
                stacklevel=3,
            )

        counts = _allocate_counts(weights, effective_total)
        all_samples: list[dict] = []
        all_indices: list[tuple[int, int]] = []

        for entry_index, (entry, count) in enumerate(zip(self.recipe.datasets, counts)):
            if count == 0:
                continue
            ds = source_registry.get(
                entry.dataset_id, entry.subset, entry.split, token=self.hf_token
            )
            # The index column is attached before shuffling, so it records a position
            # in the original split.  Verified not to change which rows are selected.
            ds = source_registry.with_row_index(ds)
            ds = ds.shuffle(seed=self.seed)
            ds = ds.select(range(count))
            for row in ds:
                row = dict(row)
                all_indices.append((entry_index, row.pop(source_registry.ROW_INDEX_COLUMN)))
                all_samples.append(row)

        # Shuffle the merged list deterministically; terminal cap enforces hard limit.
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(all_samples)).tolist()
        self.sources = sources
        self.source_indices = [all_indices[i] for i in idx][:self.total_samples]
        return [all_samples[i] for i in idx][:self.total_samples]

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
        The text is each entry's configured ``text_field``, or its composed
        ``text_fields`` — see :func:`src.datasets._text_projection.row_text`.
        """
        samples = self._ensure_loaded()
        if n is not None:
            if n > len(samples):
                raise ValueError(
                    f"Requested {n} queries but only {len(samples)} samples available."
                )
            samples = samples[:n]

        return [row_text(self.recipe, row) for row in samples]

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

    def __init__(
        self,
        samples: list[dict],
        recipe: DatasetRecipe | ClassAwareDatasetRecipe,
        source_indices: list[tuple[int, int]] | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        self._samples = samples
        self.recipe = recipe
        self.total_samples = len(samples)
        self.seed: int | None = None
        self.hf_token: str | None = None
        # Carried through from the manifest so a draw read from one cache can be
        # written to another.  Without these a round-trip through the cache would lose
        # its provenance, and SampledDatasetCache.put would refuse the re-write.
        self.sources = sources
        self.source_indices = source_indices

    def to_queries(self, n: int | None = None) -> list[str]:
        samples = self._samples[:n] if n is not None else self._samples
        return [row_text(self.recipe, row) for row in samples]

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
        #: Source descriptors, one per recipe entry — see src.datasets.source_registry.
        self.sources: list[dict] | None = None
        #: ``(entry_index, row_index)`` per returned row, in the same order.  This is
        #: what SampledDatasetCache persists in place of the row text.
        self.source_indices: list[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_entry(self, entry: ClassDatasetEntry, count: int) -> tuple[list[dict], list[int]]:
        """Load *count* samples from *entry*, respecting class weights/filter.

        If the most-constrained class cannot meet its proportional quota, the
        effective total is scaled down so that all class ratios are maintained.

        Returns the rows and their positions in the *unfiltered* source split.  The
        index column is attached before either filter, so it survives both and still
        refers to the original split rather than the filtered view.
        """
        import warnings
        from collections import Counter

        from src.datasets import source_registry

        ds = source_registry.get(
            entry.dataset_id, entry.subset, entry.split, token=self.hf_token
        )
        ds = source_registry.with_row_index(ds)

        # Apply class_filter
        if entry.class_filter is not None:
            allowed = set(entry.class_filter)
            ds = ds.filter(lambda row: row[entry.class_field] in allowed)

        if len(ds) == 0:
            return [], []

        # Determine per-class normalized weights
        if entry.normalized_class_weights is not None:
            class_norm_w = entry.normalized_class_weights
        else:
            # Uniform over all present classes
            present = list({row[entry.class_field] for row in ds})
            class_norm_w = {c: 1.0 / len(present) for c in present}

        classes = list(class_norm_w.keys())
        w_list = [class_norm_w[c] for c in classes]

        # Read all class sizes in one Arrow column scan (fast, no per-class filter).
        class_sizes: Counter = Counter(ds[entry.class_field])

        # Find the constraining class and scale the effective total down so that
        # all class proportions are preserved even when one class is data-limited.
        effective_count = count
        for cls_val, w in zip(classes, w_list):
            if w > 0:
                size = class_sizes.get(cls_val, 0)
                effective_count = min(effective_count, int(size / w))

        if effective_count < count:
            warnings.warn(
                f"ClassMixedDataset: recipe capacity for '{entry.dataset_id}' is "
                f"{effective_count} (requested {count}); scaling all class counts "
                f"proportionally to maintain ratios.",
                stacklevel=4,
            )

        per_class_counts = _allocate_counts(w_list, effective_count)

        rng = np.random.default_rng(self.seed)
        samples: list[dict] = []
        indices: list[int] = []
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
            cls_count = min(cls_count, len(cls_ds))  # safety net
            cls_ds = cls_ds.select(range(cls_count))
            for row in cls_ds:
                row = dict(row)
                indices.append(row.pop(source_registry.ROW_INDEX_COLUMN))
                samples.append(row)
        return samples, indices

    def _load(self) -> list[dict]:
        from src.datasets import source_registry

        counts = _allocate_counts(self.recipe.normalized_weights, self.total_samples)
        all_samples: list[dict] = []
        all_indices: list[tuple[int, int]] = []
        sources: list[dict] = []
        for entry_index, (entry, count) in enumerate(zip(self.recipe.datasets, counts)):
            ds = source_registry.get(
                entry.dataset_id, entry.subset, entry.split, token=self.hf_token
            )
            sources.append(
                source_registry.describe(ds, entry.dataset_id, entry.subset, entry.split)
            )
            if count > 0:
                rows, row_indices = self._load_entry(entry, count)
                all_samples.extend(rows)
                all_indices.extend((entry_index, i) for i in row_indices)

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(all_samples)).tolist()
        self.sources = sources
        self.source_indices = [all_indices[i] for i in idx][:self.total_samples]
        return [all_samples[i] for i in idx][:self.total_samples]

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

        return [row_text(self.recipe, row) for row in samples]

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
