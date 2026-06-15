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

    Activation modes
    ----------------
    ``"input"`` (default)
        Activations from the forward pass on the input prompt only.
    ``"generation"``
        Activations collected during auto-regressive generation.  At each
        decoding step the last-token hidden state is extracted per layer; these
        are mean-pooled across all generated steps to yield one vector per
        (probe, layer).  Requires ``max_new_tokens > 0``.
    ``"both"``
        Runs both input and generation passes and stacks their Gram rows,
        producing a ``(2*N_layers, features)`` matrix.  The first N_layers rows
        correspond to input activations; the next N_layers to generation
        activations.
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
        activation_mode: Literal["input", "generation", "both"] = "input",
        max_new_tokens: int = 32,
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
        self.activation_mode = activation_mode
        self.max_new_tokens = max_new_tokens

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
            "activation_mode": self.activation_mode,
            "max_new_tokens": self.max_new_tokens,
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

        # Number of layer-vector slots per probe depends on activation_mode
        n_layer_vecs = (2 if self.activation_mode == "both" else 1) * len(self.layer_indices)
        per_layer_vecs: list[list[np.ndarray]] = [[] for _ in range(n_layer_vecs)]

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

        matrix = np.stack(gram_rows, axis=0)  # (n_layer_vecs, features)

        return ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={
                "n_probes": len(self.probes),
                "n_layers": n_layer_vecs,
                "layer_indices": self.layer_indices,
                "activation_mode": self.activation_mode,
            },
        )

    def _process_batch(
        self,
        model: Any,
        tokenizer: Any,
        probes: list[str],
    ) -> list[list[np.ndarray]]:
        """Return one list per probe, each containing n_layer_vecs activation vectors."""
        inputs = tokenizer(
            probes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.activation_mode == "input":
                return self._extract_input_activations(model, inputs, len(probes))
            elif self.activation_mode == "generation":
                return self._extract_generation_activations(
                    model, tokenizer, inputs, len(probes)
                )
            elif self.activation_mode == "both":
                input_vecs = self._extract_input_activations(model, inputs, len(probes))
                gen_vecs = self._extract_generation_activations(
                    model, tokenizer, inputs, len(probes)
                )
                # Concatenate per-probe: [input_layers..., gen_layers...]
                return [iv + gv for iv, gv in zip(input_vecs, gen_vecs)]
            else:
                raise ValueError(f"Unknown activation_mode: {self.activation_mode!r}")

    def _extract_input_activations(
        self,
        model: Any,
        inputs: dict,
        n_probes: int,
    ) -> list[list[np.ndarray]]:
        """Forward pass on the input prompt; return one vector per (probe, layer)."""
        out = model(**inputs, output_hidden_states=True)
        batch_results: list[list[np.ndarray]] = []
        for probe_idx in range(n_probes):
            probe_layer_vecs: list[np.ndarray] = []
            for layer_idx in self.layer_indices:
                h = out.hidden_states[layer_idx][probe_idx]  # (seq_len, d)
                vec = self._pool(h).float().cpu().numpy()
                probe_layer_vecs.append(vec)
            batch_results.append(probe_layer_vecs)
        return batch_results

    def _extract_generation_activations(
        self,
        model: Any,
        tokenizer: Any,
        inputs: dict,
        n_probes: int,
    ) -> list[list[np.ndarray]]:
        """Generation-phase activations, mean-pooled across decoding steps.

        At each step the last-token hidden state is extracted for each selected
        layer (shape: batch × d) and accumulated.  After all steps the per-step
        tensors are stacked and averaged, yielding one (d,) vector per
        (probe, layer).
        """
        gen_out = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        # gen_out.hidden_states: tuple[step] of tuple[layer] of (batch, seq_at_step, d)
        # For step 0: seq_at_step == input_len + 1.  For steps 1+: seq_at_step == 1.
        # Taking [:, -1, :] always selects the most-recently generated token.

        per_layer_per_step: list[list[np.ndarray]] = [[] for _ in self.layer_indices]
        for step_hs in gen_out.hidden_states:
            for k, layer_idx in enumerate(self.layer_indices):
                h = step_hs[layer_idx][:, -1, :]   # (batch, d)
                per_layer_per_step[k].append(h.float().cpu().numpy())

        batch_results: list[list[np.ndarray]] = [[] for _ in range(n_probes)]
        for k in range(len(self.layer_indices)):
            stacked = np.stack(per_layer_per_step[k], axis=0)  # (n_steps, batch, d)
            mean_over_steps = stacked.mean(axis=0)              # (batch, d)
            for probe_idx in range(n_probes):
                batch_results[probe_idx].append(mean_over_steps[probe_idx])

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
