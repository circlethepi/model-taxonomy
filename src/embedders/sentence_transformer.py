from __future__ import annotations

from typing import Any

import numpy as np

from src.core.protocols import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embeds a probe using a separate sentence-transformers model.

    The sentence-transformer is loaded once at construction time on a separate
    device (usually CPU) and is kept alive across all probes and models.
    The LM under analysis generates text; this embedder then encodes that text.
    If use_generated_text=False, the raw probe string is encoded instead.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_generated_text: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_generated_text = use_generated_text
        self.normalize_embeddings = normalize_embeddings
        self._st_model = None
        self._embedding_dim: int | None = None

    def _load(self) -> None:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            self._embedding_dim = self._st_model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def embed(self, model_output: Any, probe: str) -> np.ndarray:
        self._load()
        if self.use_generated_text:
            text = getattr(model_output, "generated_text", None)
            if text is None:
                raise ValueError(
                    "model_output.generated_text is None. "
                    "Set max_new_tokens > 0 in BehavioralTaxonomy or use use_generated_text=False."
                )
        else:
            text = probe

        vec = self._st_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def config_dict(self) -> dict[str, Any]:
        return {
            "embedder_class": "SentenceTransformerEmbedder",
            "model_name": self.model_name,
            "use_generated_text": self.use_generated_text,
            "normalize_embeddings": self.normalize_embeddings,
        }
