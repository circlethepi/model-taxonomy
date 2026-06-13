from __future__ import annotations

from typing import Any

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation


class TrainingDataTaxonomy(Taxonomy):
    """Extracts representations from a model's fine-tuning dataset.

    Not yet implemented. This taxonomy compares models at the level of their
    training data distribution, using HuggingFace datasets.
    """

    @property
    def taxonomy_name(self) -> str:
        return "training_data"

    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError("TrainingDataTaxonomy is not yet implemented.")

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        raise NotImplementedError("TrainingDataTaxonomy is not yet implemented.")
