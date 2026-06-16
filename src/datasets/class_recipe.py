from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClassDatasetEntry:
    """One constituent dataset in a class-aware mixing recipe.

    ``class_field`` names the column whose values distinguish classes (e.g.
    ``"label"``).  ``class_filter`` restricts sampling to a subset of class
    values.  ``class_weights`` controls the proportion drawn from each class;
    if omitted, classes are sampled uniformly.
    """

    dataset_id: str
    split: str = "train"
    weight: float = 1.0
    text_field: str = "text"
    class_field: str = "label"
    subset: str | None = None
    class_filter: list | None = None
    class_weights: dict | None = None

    # Derived in __post_init__
    normalized_class_weights: dict | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        active_classes = self.class_filter
        if self.class_weights is not None:
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
        return cls(
            dataset_id=d["dataset_id"],
            split=d.get("split", "train"),
            weight=d.get("weight", 1.0),
            text_field=d.get("text_field", "text"),
            class_field=d.get("class_field", "label"),
            subset=d.get("subset"),
            class_filter=class_filter,
            class_weights=class_weights,
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
        return json.dumps(
            {
                "name": self.name,
                "datasets": [e.to_dict() for e in self.datasets],
            },
            sort_keys=True,
        )

    def recipe_hash(self) -> str:
        """16-char SHA-256 prefix that uniquely identifies this recipe."""
        return hashlib.sha256(self._canonical().encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "schema_version": "1",
            "recipe_type": "class_aware",
            "name": self.name,
            "recipe_hash": self.recipe_hash(),
            "datasets": [e.to_dict() for e in self.datasets],
            "normalized_weights": self.normalized_weights,
        }

    def save(self, path: Path | str) -> None:
        """Write recipe to a ``.recipe.json`` file atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        os.replace(tmp, path)

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
