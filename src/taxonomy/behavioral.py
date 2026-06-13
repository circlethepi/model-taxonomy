from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from src.core.protocols import Taxonomy, Embedder, ModelID
from src.core.representation import ModelRepresentation
from src.cache.disk import DiskCache


@dataclass
class _InferenceOutput:
    """Unified container for a single forward pass over one probe."""

    hidden_states: tuple | None
    logits: torch.Tensor
    generated_text: str | None


class BehavioralTaxonomy(Taxonomy):
    """Extracts behavioral representations of HuggingFace language models.

    For each model, runs inference over a set of probe strings and uses the
    provided embedder to convert each output into a fixed-size vector.
    The stacked vectors form the (N_probes, d) matrix representation.
    """

    def __init__(
        self,
        probes: Sequence[str],
        embedder: Embedder,
        cache: DiskCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 0,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
    ) -> None:
        self.probes = list(probes)
        self.embedder = embedder
        self.cache = cache
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

    @property
    def taxonomy_name(self) -> str:
        return "behavioral"

    def config_dict(self) -> dict[str, Any]:
        return {
            "taxonomy": "behavioral",
            "probes": self.probes,
            "embedder": self.embedder.config_dict(),
            "max_new_tokens": self.max_new_tokens,
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

        vectors: list[np.ndarray] = []
        try:
            for i in range(0, len(self.probes), self.batch_size):
                batch_probes = self.probes[i : i + self.batch_size]
                batch_vectors = self._process_batch(model, tokenizer, batch_probes)
                vectors.extend(batch_vectors)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        matrix = np.stack(vectors, axis=0)  # (N_probes, d)
        return ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={"n_probes": len(self.probes)},
        )

    def _process_batch(
        self,
        model: Any,
        tokenizer: Any,
        probes: list[str],
    ) -> list[np.ndarray]:
        from transformers import GenerationConfig

        inputs = tokenizer(
            probes,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.max_new_tokens > 0:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated_texts = tokenizer.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                forward_out = model(
                    **inputs,
                    output_hidden_states=True,
                )
            else:
                generated_texts = [None] * len(probes)
                forward_out = model(
                    **inputs,
                    output_hidden_states=True,
                )

        vectors = []
        for idx, (probe, gen_text) in enumerate(zip(probes, generated_texts)):
            # Build per-probe output by slicing batch dimension
            per_probe = _InferenceOutput(
                hidden_states=(
                    tuple(h[idx : idx + 1] for h in forward_out.hidden_states)
                    if forward_out.hidden_states
                    else None
                ),
                logits=forward_out.logits[idx : idx + 1],
                generated_text=gen_text,
            )
            vec = self.embedder.embed(per_probe, probe)
            vectors.append(vec)

        return vectors
