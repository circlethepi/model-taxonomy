from __future__ import annotations

import gc
import os
from typing import Any, Literal, Sequence

import numpy as np
import torch

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation
from src.cache.disk import DiskCache


class FunctionalTaxonomy(Taxonomy):
    """Compares models via the covariance structure of their internal activations.

    For each probe input and each specified layer, the model's hidden states are
    pooled to a single vector. The (N_probes, hidden_dim) activation matrix is then
    used to compute a Gram matrix G = H @ H.T, where G[i,j] is the dot product of
    the activation of probe i with the activation of probe j at that layer.

    The representation for a model is the stacked upper triangles of its per-layer
    Gram matrices: shape (N_layers, N_probes*(N_probes+1)//2).
    """

    def __init__(
        self,
        probes: Sequence[str],
        layer_indices: list[int],
        cache: DiskCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        pooling: Literal["mean", "last_token", "cls"] = "mean",
        normalize_activations: bool = True,
    ) -> None:
        self.probes = list(probes)
        self.layer_indices = list(layer_indices)
        self.cache = cache
        self.device = device
        self.batch_size = batch_size
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.pooling = pooling
        self.normalize_activations = normalize_activations

    @property
    def taxonomy_name(self) -> str:
        return "functional"

    def config_dict(self) -> dict[str, Any]:
        return {
            "taxonomy": "functional",
            "probes": self.probes,
            "layer_indices": self.layer_indices,
            "pooling": self.pooling,
            "normalize_activations": self.normalize_activations,
            "torch_dtype": str(self.torch_dtype),
        }

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        cache_key = DiskCache.key_for(model_id, self.config_dict()) if self.cache else ""

        if self.cache is not None and self.cache.exists(cache_key):
            return self.cache.load(cache_key)

        rep = self._extract_fresh(model_id, cache_key)

        if self.cache is not None:
            self.cache.save(cache_key, rep)

        return rep

    def _extract_fresh(self, model_id: ModelID, cache_key: str) -> ModelRepresentation:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=self.hf_token, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            token=self.hf_token,
            trust_remote_code=True,
        )
        model.eval()

        # per_layer_vecs[k] accumulates one (d,) vector per probe for layer_indices[k]
        per_layer_vecs: list[list[np.ndarray]] = [[] for _ in self.layer_indices]

        try:
            for i in range(0, len(self.probes), self.batch_size):
                batch_probes = self.probes[i : i + self.batch_size]
                batch_vecs = self._process_batch(model, tokenizer, batch_probes)
                for probe_vecs in batch_vecs:
                    for k, vec in enumerate(probe_vecs):
                        per_layer_vecs[k].append(vec)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        triu_idx = np.triu_indices(len(self.probes))
        gram_rows: list[np.ndarray] = []
        for vecs in per_layer_vecs:
            H = np.stack(vecs, axis=0).astype(np.float64)  # (N_probes, d)
            if self.normalize_activations:
                norms = np.linalg.norm(H, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                H = H / norms
            G = H @ H.T  # (N_probes, N_probes)
            gram_rows.append(G[triu_idx].astype(np.float32))

        matrix = np.stack(gram_rows, axis=0)  # (N_layers, N_probes*(N_probes+1)//2)

        return ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={
                "n_probes": len(self.probes),
                "n_layers": len(self.layer_indices),
                "layer_indices": self.layer_indices,
            },
        )

    def _process_batch(
        self,
        model: Any,
        tokenizer: Any,
        probes: list[str],
    ) -> list[list[np.ndarray]]:
        """Return one list per probe, each containing N_layers activation vectors."""
        inputs = tokenizer(
            probes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        batch_results: list[list[np.ndarray]] = []
        for probe_idx in range(len(probes)):
            probe_layer_vecs: list[np.ndarray] = []
            for layer_idx in self.layer_indices:
                h = out.hidden_states[layer_idx][probe_idx]  # (seq_len, d)
                vec = self._pool(h).float().cpu().numpy()     # (d,)
                probe_layer_vecs.append(vec)
            batch_results.append(probe_layer_vecs)

        return batch_results

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return h.mean(dim=0)
        elif self.pooling == "last_token":
            return h[-1]
        elif self.pooling == "cls":
            return h[0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling!r}")
