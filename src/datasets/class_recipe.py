from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.atomic import atomic_write_json
from src.datasets._text_projection import (
    DEFAULT_SEPARATOR,
    composition_dict,
    read_composition,
)


#: The two ways an entry's rows may be spread over its classes.
CLASS_SAMPLING_MODES = ("stratified", "pooled")
DEFAULT_CLASS_SAMPLING = "stratified"


def class_sampling_dict(class_sampling: str) -> dict:
    """The ``class_sampling`` key for ``to_dict()``, or nothing at all.

    Empty at the default, for the same reason
    :func:`src.datasets._text_projection.composition_dict` is: ``to_dict()`` is
    what ``recipe_hash`` is computed over, so emitting the key unconditionally
    would move every hash already in ``01_datasets`` at once.
    """
    if class_sampling == DEFAULT_CLASS_SAMPLING:
        return {}
    return {"class_sampling": class_sampling}


@dataclass
class ClassDatasetEntry:
    """One constituent dataset in a class-aware mixing recipe.

    ``class_field`` names the column whose values distinguish classes (e.g.
    ``"label"``).  ``class_filter`` restricts sampling to a subset of class
    values.  ``class_weights`` controls the proportion drawn from each class;
    if omitted, classes are sampled uniformly.

    ``class_sampling`` decides how the entry's row quota is spread over those
    classes.  Both modes allocate the *entry's* share of the draw identically —
    that is the recipe's dataset-level weights — and differ only in the second
    stage, within the entry:

    ``"stratified"`` (default)
        Exact per-class quotas: each class gets ``count / n_classes`` rows, or
        its ``class_weights`` share, by largest-remainder allocation.  Random
        choice picks *which* rows within a class, never *how many*.  A two-entry
        recipe over three classes each therefore yields exactly ``n/6`` per
        class, every time, for every seed.
    ``"pooled"``
        The classes are one pool and ``count`` rows are drawn uniformly without
        replacement from it.  Per-class counts are hypergeometric: they vary with
        the seed and track the classes' natural sizes rather than being forced
        equal.  This is what "sample uniformly from the combination of these
        topics" means, and it is not expressible by any choice of
        ``class_weights`` — weights move the expectation, not the determinism.

        ``class_weights`` is rejected alongside ``pooled``.  The combination is
        contradictory, and silently ignoring the weights is exactly the failure
        this mode exists to remove.

        Note ``pooled`` is meaningful without ``class_filter`` too: it then means
        "do not stratify at all", which is otherwise unreachable — an entry with
        neither ``class_filter`` nor ``class_weights`` still gets stratified
        uniformly over whichever classes happen to be present.

    ``text_fields`` composes several columns into the entry's text instead of
    taking one, joined by ``text_separator`` — see
    :mod:`src.datasets._text_projection` for why that exists and why an entry
    that does not use it must serialize exactly as it did before.
    """

    dataset_id: str
    split: str = "train"
    weight: float = 1.0
    text_field: str = "text"
    class_field: str = "label"
    subset: str | None = None
    class_filter: list | None = None
    class_weights: dict | None = None
    class_sampling: str = DEFAULT_CLASS_SAMPLING
    text_fields: list[str] | None = None
    text_separator: str = DEFAULT_SEPARATOR

    # Derived in __post_init__
    normalized_class_weights: dict | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.class_sampling not in CLASS_SAMPLING_MODES:
            raise ValueError(
                f"class_sampling must be one of {CLASS_SAMPLING_MODES}, "
                f"got {self.class_sampling!r}."
            )

        active_classes = self.class_filter
        if self.class_sampling == "pooled":
            if self.class_weights is not None:
                raise ValueError(
                    "class_sampling='pooled' draws uniformly from the union of the "
                    "classes, so per-class weights cannot be honoured. Drop "
                    "class_weights, or use class_sampling='stratified'."
                )
            # No per-class quotas at all: the sampler takes the pooled branch and
            # per-class counts fall out of the draw.
            self.normalized_class_weights = None
        elif self.class_weights is not None:
            # If class_filter is set, restrict weights to those classes
            cw = {
                k: v
                for k, v in self.class_weights.items()
                if active_classes is None or k in active_classes
            }
            if not cw:
                raise ValueError(
                    "class_weights has no entries matching class_filter."
                )
            total = sum(cw.values())
            if total <= 0:
                raise ValueError("class_weights must sum to a positive number.")
            self.normalized_class_weights = {k: v / total for k, v in cw.items()}
        elif active_classes is not None:
            # Uniform over filtered classes
            n = len(active_classes)
            self.normalized_class_weights = {c: 1.0 / n for c in active_classes}
        else:
            self.normalized_class_weights = None  # determined at load time

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "weight": self.weight,
            "text_field": self.text_field,
            "class_field": self.class_field,
            "subset": self.subset,
            "class_filter": self.class_filter,
            "class_weights": self.class_weights,
            "normalized_class_weights": self.normalized_class_weights,
            # Both spliced in only when non-default: this dict is what recipe_hash
            # is computed over, so an unconditional key would move every existing
            # hash.
            **class_sampling_dict(self.class_sampling),
            **composition_dict(self.text_fields, self.text_separator),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ClassDatasetEntry:
        class_filter = d.get("class_filter")
        class_weights = d.get("class_weights")
        # JSON keys are always strings; coerce to match class_filter element types
        # so that the membership check in __post_init__ works correctly.
        if class_weights is not None and class_filter:
            try:
                elem_type = type(class_filter[0])
                class_weights = {elem_type(k): v for k, v in class_weights.items()}
            except (ValueError, TypeError):
                pass
        text_fields, text_separator = read_composition(d)
        return cls(
            dataset_id=d["dataset_id"],
            split=d.get("split", "train"),
            weight=d.get("weight", 1.0),
            text_field=d.get("text_field", "text"),
            class_field=d.get("class_field", "label"),
            subset=d.get("subset"),
            class_filter=class_filter,
            class_weights=class_weights,
            class_sampling=d.get("class_sampling", DEFAULT_CLASS_SAMPLING),
            text_fields=text_fields,
            text_separator=text_separator,
        )


@dataclass
class ClassAwareDatasetRecipe:
    """Weighted mixture of HuggingFace datasets with per-class weight control.

    Each entry specifies a dataset and, optionally, how to weight the classes
    within it.  Dataset-level weights are normalized just like ``DatasetRecipe``.
    Serializes to the same ``.recipe.json`` format with ``recipe_type =
    "class_aware"``.
    """

    name: str
    datasets: list[ClassDatasetEntry]
    normalized_weights: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError(
                "ClassAwareDatasetRecipe requires at least one ClassDatasetEntry."
            )
        total = sum(e.weight for e in self.datasets)
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        self.normalized_weights = [e.weight / total for e in self.datasets]

    # ------------------------------------------------------------------
    # Hashing & serialization
    # ------------------------------------------------------------------

    def _canonical(self) -> str:
        """Deterministic JSON string used to derive the recipe hash.

        Content only — see :meth:`src.datasets.recipe.DatasetRecipe._canonical` for why
        ``name`` is excluded and ``recipe_type`` included.  Both classes must apply the
        same rule; they are structural duplicates by design.
        """
        return json.dumps(
            {
                "recipe_type": "class_aware",
                "datasets": [e.to_dict() for e in self.datasets],
            },
            sort_keys=True,
        )

    def recipe_hash(self) -> str:
        """16-char SHA-256 prefix that uniquely identifies this recipe's content."""
        return hashlib.sha256(self._canonical().encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        # schema_version 2 = content-addressed hash; see DatasetRecipe.to_dict.
        return {
            "schema_version": "2",
            "recipe_type": "class_aware",
            "name": self.name,
            "recipe_hash": self.recipe_hash(),
            "datasets": [e.to_dict() for e in self.datasets],
            "normalized_weights": self.normalized_weights,
        }

    def save(self, path: Path | str) -> None:
        """Write recipe to a ``.recipe.json`` file atomically.

        Concurrency matters here more than anywhere else in the cache: every job
        in a suite builds the *same* recipe, so a mass submission has dozens of
        processes writing this one path within the same second.
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> ClassAwareDatasetRecipe:
        """Reconstruct a ClassAwareDatasetRecipe from a ``.recipe.json`` file."""
        data = json.loads(Path(path).read_text())
        if data.get("recipe_type") != "class_aware":
            raise ValueError(
                f"Expected recipe_type='class_aware', got {data.get('recipe_type')!r}. "
                "Use DatasetRecipe.load() for simple recipes."
            )
        return cls(
            name=data["name"],
            datasets=[ClassDatasetEntry.from_dict(d) for d in data["datasets"]],
        )
