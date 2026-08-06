from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from src.core.protocols import Taxonomy, Embedder, ModelID
from src.core.representation import ModelRepresentation
from src.cache.generated_text_cache import GeneratedTextCache
from src.taxonomy._hf_inference import HFInferenceTaxonomy


@dataclass
class _InferenceOutput:
    """Unified container for the output of one generation call over one query."""

    hidden_states: tuple | None
    logits: "torch.Tensor | None"
    generated_text: str | None


class BehavioralTaxonomy(HFInferenceTaxonomy, Taxonomy):
    """Extracts behavioral representations of HuggingFace language models.

    For each model, generates continuations for a set of query strings and uses
    the provided embedder to convert each generated output into a fixed-size
    vector.  The stacked vectors form the (N_queries, d) matrix representation.

    This taxonomy operates **exclusively on generated text output** — it does not
    collect hidden states or logits during the generation pass.  Use
    :class:`FunctionalTaxonomy` if you need activation-based comparison.

    Generated texts are stored in ``ModelRepresentation.metadata["generated_texts"]``
    so you can audit outputs without re-running the model.

    Model loading — base-model reuse, adapter swapping, ``close()`` — comes from
    :class:`~src.taxonomy._hf_inference.HFInferenceTaxonomy`, shared with
    :class:`~src.taxonomy.functional.FunctionalTaxonomy`.

    Parameters
    ----------
    query_key:
        The ``{recipe_hash, n_samples, seed}`` triple identifying the query draw in
        ``01_datasets``.  This — not the query strings — is what goes into
        :meth:`config_dict`.  Hashing the strings would make every cache entry
        sensitive to any upstream change that shifts the draw, and would leave no
        way to tell from a cache key which draw an entry belonged to.
    max_new_tokens:
        Number of tokens to generate per query.  Must be > 0 — this is what
        distinguishes behavioral (output-based) comparison from functional
        (activation-based) comparison.
    """

    def __init__(
        self,
        queries: Sequence[str],
        embedder: Embedder,
        query_key: dict | None = None,
        cache: GeneratedTextCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 64,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        source_indices: list | None = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "BehavioralTaxonomy requires max_new_tokens > 0. "
                "Behavioral comparison is based on generated text output. "
                "For activation-based comparison use FunctionalTaxonomy instead."
            )
        super().__init__(
            device=device,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
        )
        self.queries = list(queries)
        self.embedder = embedder
        self.query_key = dict(query_key or {})
        self.cache = cache
        self.max_new_tokens = max_new_tokens
        # Which source row of 01_datasets is query i.  A denormalized convenience
        # -- the draw file holds the same list -- and deliberately outside
        # config_dict(), so supplying it does not fragment the cache.
        self.source_indices = source_indices

    @property
    def taxonomy_name(self) -> str:
        return "behavioral"

    def config_dict(self) -> dict[str, Any]:
        # "taxonomy" is not required by the Taxonomy protocol — config_dict only
        # has to be deterministic — but it keeps a behavioral config from ever
        # hashing equal to a functional one, and makes config.json self-describing.
        return {
            "taxonomy": "behavioral",
            "query_key": self.query_key,
            "n_queries": len(self.queries),
            "embedder": self.embedder.config_dict(),
            "max_new_tokens": self.max_new_tokens,
            "torch_dtype": str(self.torch_dtype),
        }

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        config = self.config_dict()
        config_hash = GeneratedTextCache.config_hash(config) if self.cache else ""

        if self.cache is None:
            return self._extract_fresh(model_id, config_hash)

        # (base, adapter) comes from the shared HFInferenceTaxonomy helper, so a
        # behavioral entry lands at the same coordinates as the functional entry
        # for the same model — that is the whole point of TODO item 13.
        base_model_id, adapter_id = self._model_key(model_id)
        embedder_hash = GeneratedTextCache.embedder_hash(self.embedder.config_dict())

        if self.cache.exists(
            base_model_id,
            adapter_id,
            self.query_key,
            self.max_new_tokens,
            embedder_hash,
        ):
            return self.cache.load(
                base_model_id,
                adapter_id,
                self.query_key,
                self.max_new_tokens,
                embedder_hash,
            )

        rep = self._extract_fresh(model_id, config_hash)

        self.cache.save(
            base_model_id,
            adapter_id,
            self.query_key,
            rep,
            max_new_tokens=self.max_new_tokens,
            embedder_hash=embedder_hash,
            config=config,
            source_indices=self.source_indices,
        )

        return rep

    def _extract_fresh(self, model_id: ModelID, config_hash: str) -> ModelRepresentation:
        model, shared = self._get_model(model_id)
        tokenizer = self._load_tokenizer(model_id, self._resolve_base_model_id(model_id))

        vectors: list[np.ndarray] = []
        all_generated_texts: list[str] = []
        try:
            for i in range(0, len(self.queries), self.batch_size):
                batch_queries = self.queries[i : i + self.batch_size]
                batch_vectors, batch_texts = self._process_batch(model, tokenizer, batch_queries)
                vectors.extend(batch_vectors)
                all_generated_texts.extend(batch_texts)
        finally:
            if not shared:
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
                # Provenance, deliberately outside config_dict() so it does not
                # fragment the cache: greedy decoding is not reproducible across GPU
                # architectures — different fp16 kernels flip the argmax on near-ties
                # — so knowing which device produced a generation is the difference
                # between "the code changed" and "it ran on a different node".
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
                "batch_size": self.batch_size,
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
