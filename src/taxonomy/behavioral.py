from __future__ import annotations

import gc
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from src.core.protocols import Taxonomy, Embedder, ModelID
from src.core.representation import ModelRepresentation
from src.cache.generated_text_cache import GeneratedTextCache, model_slug


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

    The base model is loaded **once** and adapters are attached to it, rather than
    reloading it per model.  Across 25 adapters that is the difference between one
    6 GB read and twenty-five of them, and the ratio only gets worse at 8B.  Call
    :meth:`close` when done to release it.

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
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError(
                "BehavioralTaxonomy requires max_new_tokens > 0. "
                "Behavioral comparison is based on generated text output. "
                "For activation-based comparison use FunctionalTaxonomy instead."
            )
        self.queries = list(queries)
        self.embedder = embedder
        self.query_key = dict(query_key or {})
        self.cache = cache
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Shared across extractions, freed by close().
        self._base_model: Any = None
        self._base_model_id: str | None = None
        self._peft_model: Any = None

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

        if self.cache is not None and self.cache.exists(config_hash, model_id):
            return self.cache.load(config_hash, model_id)

        rep = self._extract_fresh(model_id, config_hash)

        if self.cache is not None:
            self.cache.save(
                config_hash,
                rep,
                config=config,
                queries=self.queries,
                query_key=self.query_key,
            )

        return rep

    # ------------------------------------------------------------------
    # Model loading — base held across calls, adapters swapped
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_base_model_id(model_id: ModelID) -> str | None:
        """The base model an adapter directory was trained on, or None.

        None means *model_id* is not a local adapter directory — a plain
        HuggingFace model ID, or a full checkpoint — and should be loaded whole.
        """
        path = Path(model_id)
        meta = path / "experiment_meta.json"
        if meta.exists():
            base = json.loads(meta.read_text()).get("base_model_id")
            if base:
                return base
        adapter_cfg = path / "adapter_config.json"
        if adapter_cfg.exists():
            return json.loads(adapter_cfg.read_text()).get("base_model_name_or_path")
        return None

    def _load_tokenizer(self, model_id: ModelID, base_model_id: str | None) -> Any:
        """Tokenizer from the adapter dir, falling back to the base model.

        finetune_lora.py does not call ``tokenizer.save_pretrained`` explicitly; it
        relies on ``Trainer.save_model`` persisting the processing class, so the
        files are usually but not certainly there.
        """
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=self.hf_token, trust_remote_code=True
            )
        except Exception:
            if base_model_id is None:
                raise
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_id, token=self.hf_token, trust_remote_code=True
            )

        # Left padding is what makes batched greedy generation agree with
        # batch_size=1: with right padding the pads land between the prompt and
        # the first generated token and every short sequence in the batch decodes
        # from the wrong position.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_model(self, model_id: ModelID) -> tuple[Any, bool]:
        """Return ``(model, shared)``.

        ``shared`` is False when the model was loaded standalone and the caller
        should free it; True when it is the shared base with an adapter attached
        and must outlive this extraction.
        """
        from transformers import AutoModelForCausalLM

        base_model_id = self._resolve_base_model_id(model_id)

        if base_model_id is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
            )
            model.eval()
            return model, False

        # Adapter swapping is only valid across adapters sharing one base.  A
        # mixed-base collection degrades to standalone loads rather than silently
        # generating from the wrong weights.
        if self._base_model is not None and base_model_id != self._base_model_id:
            warnings.warn(
                f"{model_id} has base {base_model_id!r} but this taxonomy already holds "
                f"{self._base_model_id!r}; loading it standalone. Base-model reuse only "
                f"applies within one base model.",
                stacklevel=2,
            )
            from peft import PeftModel

            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, str(model_id))
            model.eval()
            return model, False

        from peft import PeftModel

        if self._base_model is None:
            # Loading the base explicitly, rather than handing transformers an
            # adapter directory and relying on it to detect the adapter and
            # resolve base_model_name_or_path, keeps us in control of which base
            # is used and separates "base failed to load" from "adapter failed to
            # attach".  It is also what makes reuse possible at all.
            self._base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                token=self.hf_token,
                trust_remote_code=True,
            )
            self._base_model_id = base_model_id

        adapter_name = model_slug(str(model_id))
        if self._peft_model is None:
            self._peft_model = PeftModel.from_pretrained(
                self._base_model, str(model_id), adapter_name=adapter_name
            )
        elif adapter_name not in self._peft_model.peft_config:
            self._peft_model.load_adapter(str(model_id), adapter_name=adapter_name)

        self._peft_model.set_adapter(adapter_name)
        self._peft_model.eval()
        return self._peft_model, True

    def close(self) -> None:
        """Release the shared base model.

        The per-extraction ``finally`` frees only standalone models; the base is
        held on purpose and would otherwise survive until interpreter exit.
        """
        self._peft_model = None
        self._base_model = None
        self._base_model_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "BehavioralTaxonomy":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

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
