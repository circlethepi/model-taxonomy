from __future__ import annotations

from typing import Any, Literal

import numpy as np

from src.core.protocols import Embedder


class HiddenStateEmbedder(Embedder):
    """Embeds a query by extracting the model's own hidden states or logits.

    No second model is needed; operates entirely within the model under analysis.
    """

    def __init__(
        self,
        strategy: Literal["hidden_states", "logits"] = "hidden_states",
        layer_index: int = -1,
        pooling: Literal["mean", "last_token", "cls"] = "mean",
    ) -> None:
        self.strategy = strategy
        self.layer_index = layer_index
        self.pooling = pooling
        self._embedding_dim: int | None = None

    @property
    def embedding_dim(self) -> int | None:
        return self._embedding_dim

    def embed(self, model_output: Any, query: str) -> np.ndarray:
        """Extract and pool a vector from model_output.

        model_output must have .hidden_states (tuple of tensors) or .logits.
        """
        import torch

        if self.strategy == "hidden_states":
            if model_output.hidden_states is None:
                raise ValueError(
                    "model_output.hidden_states is None. "
                    "Pass output_hidden_states=True when calling the model."
                )
            h = model_output.hidden_states[self.layer_index]  # (batch, seq_len, hidden_dim)
            h = h.squeeze(0)  # (seq_len, hidden_dim)
            vec = self._pool(h)  # (hidden_dim,)
        else:
            logits = model_output.logits.squeeze(0)  # (seq_len, vocab_size)
            vec = self._pool(logits)  # (vocab_size,)

        result = vec.float().cpu().numpy()
        self._embedding_dim = result.shape[0]
        return result

    def _pool(self, tensor: "torch.Tensor") -> "torch.Tensor":
        import torch

        if self.pooling == "mean":
            return tensor.mean(dim=0)
        elif self.pooling == "last_token":
            return tensor[-1]
        elif self.pooling == "cls":
            return tensor[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling!r}")

    def config_dict(self) -> dict[str, Any]:
        return {
            "embedder_class": "HiddenStateEmbedder",
            "strategy": self.strategy,
            "layer_index": self.layer_index,
            "pooling": self.pooling,
        }
