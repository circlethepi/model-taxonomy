from __future__ import annotations

from typing import Any

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation


class StructuralTaxonomy(Taxonomy):
    """Extracts structural representations from model weight matrices.

    Not yet implemented. This taxonomy captures the geometry of the model's
    parameters directly, without requiring any input data.
    """

    @property
    def taxonomy_name(self) -> str:
        return "structural"

    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError("StructuralTaxonomy is not yet implemented.")

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        raise NotImplementedError("StructuralTaxonomy is not yet implemented.")
