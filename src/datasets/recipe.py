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


@dataclass
class DatasetEntry:
    """One constituent dataset in a mixing recipe.

    ``text_fields`` composes several columns into the entry's text instead of
    taking one, joined by ``text_separator``; see
    :mod:`src.datasets._text_projection`.  Kept in step with
    :class:`~src.datasets.class_recipe.ClassDatasetEntry`, which is where the
    composition is actually used — the two are structural duplicates by design
    and drift between them is the recurring hazard.
    """

    dataset_id: str
    split: str = "train"
    weight: float = 1.0
    text_field: str = "text"
    subset: str | None = None
    text_fields: list[str] | None = None
    text_separator: str = DEFAULT_SEPARATOR

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "weight": self.weight,
            "text_field": self.text_field,
            "subset": self.subset,
            # Only when set — this dict is hashed into recipe_hash.
            **composition_dict(self.text_fields, self.text_separator),
        }

    @classmethod
    def from_dict(cls, d: dict) -> DatasetEntry:
        text_fields, text_separator = read_composition(d)
        return cls(
            dataset_id=d["dataset_id"],
            split=d.get("split", "train"),
            weight=d.get("weight", 1.0),
            text_field=d.get("text_field", "text"),
            subset=d.get("subset"),
            text_fields=text_fields,
            text_separator=text_separator,
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
        """Deterministic JSON string used to derive the recipe hash.

        Content only — ``name`` is deliberately *not* hashed.  The entries fully
        determine the sampling distribution, so two recipes with the same entries
        produce the same draws and are the same recipe however they are labelled.
        Keeping the name out means one mixture has one hash regardless of how many
        ``_n{n}_s{seed}`` variants of its name exist in configs.

        ``recipe_type`` *is* hashed: it is the only field that distinguishes this
        from :class:`~src.datasets.class_recipe.ClassAwareDatasetRecipe`, whose
        canonical form is otherwise structurally identical.
        """
        return json.dumps(
            {
                "recipe_type": "simple",
                "datasets": [e.to_dict() for e in self.datasets],
            },
            sort_keys=True,
        )

    def recipe_hash(self) -> str:
        """16-char SHA-256 prefix that uniquely identifies this recipe's content."""
        return hashlib.sha256(self._canonical().encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        # schema_version 2 = content-addressed hash (name no longer hashed).  A "1"
        # file's stored recipe_hash is a legacy hash this code will not reproduce, so
        # the version has to be distinguishable rather than silently recomputed.
        # Versioned independently of the draw-manifest schema in SampledDatasetCache.
        return {
            "schema_version": "2",
            "recipe_type": "simple",
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
