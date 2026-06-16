from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetEntry:
    """One constituent dataset in a mixing recipe."""

    dataset_id: str
    split: str = "train"
    weight: float = 1.0
    text_field: str = "text"
    subset: str | None = None

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "weight": self.weight,
            "text_field": self.text_field,
            "subset": self.subset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatasetEntry:
        return cls(
            dataset_id=d["dataset_id"],
            split=d.get("split", "train"),
            weight=d.get("weight", 1.0),
            text_field=d.get("text_field", "text"),
            subset=d.get("subset"),
        )


@dataclass
class DatasetRecipe:
    """Weighted mixture of HuggingFace datasets.

    Weights are normalized to sum to 1 in ``__post_init__``.  Serializes to a
    human-readable ``.recipe.json`` file so recipes can be stored alongside
    ``DiskCache`` outputs or embedded in ``LoRACache`` config.json.
    """

    name: str
    datasets: list[DatasetEntry]
    normalized_weights: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("DatasetRecipe requires at least one DatasetEntry.")
        total = sum(e.weight for e in self.datasets)
        if total <= 0:
            raise ValueError("Weights must sum to a positive number.")
        self.normalized_weights = [e.weight / total for e in self.datasets]

    # ------------------------------------------------------------------
    # Hashing & serialization
    # ------------------------------------------------------------------

    def _canonical(self) -> str:
        """Deterministic JSON string used to derive the recipe hash."""
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
            "recipe_type": "simple",
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
    def load(cls, path: Path | str) -> DatasetRecipe:
        """Reconstruct a DatasetRecipe from a ``.recipe.json`` file."""
        data = json.loads(Path(path).read_text())
        if data.get("recipe_type") != "simple":
            raise ValueError(
                f"Expected recipe_type='simple', got {data.get('recipe_type')!r}. "
                "Use ClassAwareDatasetRecipe.load() for class-aware recipes."
            )
        return cls(
            name=data["name"],
            datasets=[DatasetEntry.from_dict(d) for d in data["datasets"]],
        )
