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
    """Unified container for the output of one generation call over one query."""

    hidden_states: tuple | None
    logits: "torch.Tensor | None"
    generated_text: str | None


class BehavioralTaxonomy(Taxonomy):
    """Extracts behavioral representations of HuggingFace language models.

    For each model, generates continuations for a set of query strings and uses
    the provided embedder to convert each generated output into a fixed-size
    vector.  The stacked vectors form the (N_queries, d) matrix representation.

    This taxonomy operates **exclusively on generated text output** — it does not
    collect hidden states or logits during the generation pass.  Use
    :class:`FunctionalTaxonomy` if you need activation-based comparison.

    Generated texts are stored in ``ModelRepresentation.metadata["generated_texts"]``
    so you can audit outputs without re-running the model.

    Parameters
    ----------
    max_new_tokens:
        Number of tokens to generate per query.  Must be > 0 — this is what
        distinguishes behavioral (output-based) comparison from functional
        (activation-based) comparison.
    """

    def __init__(
        self,
        queries: Sequence[str],
        embedder: Embedder,
        cache: DiskCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 64,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "BehavioralTaxonomy requires max_new_tokens > 0. "
                "Behavioral comparison is based on generated text output. "
                "For activation-based comparison use FunctionalTaxonomy instead."
            )
        self.queries = list(queries)
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
            "queries": self.queries,
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
        tokenizer.padding_side = "left"
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
        all_generated_texts: list[str] = []
        try:
            for i in range(0, len(self.queries), self.batch_size):
                batch_queries = self.queries[i : i + self.batch_size]
                batch_vectors, batch_texts = self._process_batch(model, tokenizer, batch_queries)
                vectors.extend(batch_vectors)
                all_generated_texts.extend(batch_texts)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        matrix = np.stack(vectors, axis=0)  # (N_queries, d)
        return ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={
                "n_queries": len(self.queries),
                "generated_texts": all_generated_texts,
            },
        )

    def _process_batch(
        self,
        model: Any,
        tokenizer: Any,
        queries: list[str],
    ) -> tuple[list[np.ndarray], list[str]]:
        inputs = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_texts = tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )

        vectors = []
        for query, gen_text in zip(queries, generated_texts):
            output_obj = _InferenceOutput(
                hidden_states=None,    # behavioral is output-only; no hidden states collected
                logits=None,
                generated_text=gen_text,
            )
            vec = self.embedder.embed(output_obj, query)
            vectors.append(vec)

        return vectors, generated_texts
