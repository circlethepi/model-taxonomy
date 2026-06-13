from __future__ import annotations

from typing import Any

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation


class FunctionalTaxonomy(Taxonomy):
    """Extracts functional representations from internal model activations.

    Not yet implemented. This taxonomy captures how a model processes inputs
    internally, independent of its final outputs.
    """

    @property
    def taxonomy_name(self) -> str:
        return "functional"

    def config_dict(self) -> dict[str, Any]:
        raise NotImplementedError("FunctionalTaxonomy is not yet implemented.")

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        raise NotImplementedError("FunctionalTaxonomy is not yet implemented.")
