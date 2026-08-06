from __future__ import annotations

import gc
import hashlib
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
    vector.  The stacked vectors form the (N_queries * replicates, d) matrix
    representation, in query-major order.

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
    replicates:
        How many continuations to draw per query.  The matrix is then
        ``(n_queries * replicates, d)`` in query-major order, and
        ``metadata["generated_texts"][q][r]`` is replicate *r* of query *q*.
        Replicates only mean anything under sampling: with ``do_sample=False``
        every replicate is the same continuation, so ``replicates > 1`` and
        greedy decoding together are rejected rather than silently producing
        ``R`` copies of one row.
    do_sample, temperature, top_p, top_k, generation_seed:
        Decoding settings.  All of them change the generated text, so all of them
        are in :meth:`config_dict` *and* in the stored filename, via
        :meth:`GeneratedTextCache.sampling_hash` — two temperatures over one draw
        are two entries, not one entry silently reused.

    **Reproducibility is conditional on ``batch_size``, and this is new.**
    Under greedy decoding, batch size only flipped ``argmax`` on fp16 near-ties —
    a last-bit effect, measured at 6/8 sequences byte-identical between batch 1
    and batch 8.  Under sampling it is first-order: one generator serves the whole
    batch, so how the RNG stream is consumed depends on the batch shape.  A re-run
    at the same ``batch_size`` and ``generation_seed`` reproduces exactly; at a
    different ``batch_size`` it does not.  ``batch_size`` deliberately stays out of
    :meth:`config_dict` — putting it in would fragment the cache along an axis
    that is a machine detail — and stays in ``metadata`` and the run record, so
    the discrepancy is detectable afterwards.
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
        replicates: int = 1,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        generation_seed: int = 0,
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
        if int(replicates) < 1:
            raise ValueError(f"replicates must be >= 1, got {replicates!r}")
        if int(replicates) > 1 and not do_sample:
            raise ValueError(
                f"replicates={replicates} with do_sample=False would store "
                f"{replicates} copies of one greedy continuation. Set "
                "do_sample=True, or leave replicates=1."
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
        self.replicates = int(replicates)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.generation_seed = generation_seed
        # Which source row of 01_datasets is query i.  A denormalized convenience
        # -- the draw file holds the same list -- and deliberately outside
        # config_dict(), so supplying it does not fragment the cache.
        self.source_indices = source_indices

    @property
    def taxonomy_name(self) -> str:
        return "behavioral"

    def sampling_config(self) -> dict[str, Any]:
        """The decoding settings, in the shape ``GeneratedTextCache`` hashes.

        Greedy is stored with the other fields nulled rather than carrying
        whatever ``temperature`` happened to be set: a temperature that was never
        applied must not change the digest, or one greedy run would be
        unreachable from another.
        """
        if not self.do_sample:
            return dict(GeneratedTextCache.GREEDY_SAMPLING)
        return {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "generation_seed": self.generation_seed,
        }

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
            "replicates": self.replicates,
            "sampling": self.sampling_config(),
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
        sampling = self.sampling_config()
        sampling_hash = GeneratedTextCache.sampling_hash(sampling)

        if self.cache.exists(
            base_model_id,
            adapter_id,
            self.query_key,
            self.max_new_tokens,
            self.replicates,
            sampling_hash,
            embedder_hash,
        ):
            return self.cache.load(
                base_model_id,
                adapter_id,
                self.query_key,
                self.max_new_tokens,
                self.replicates,
                sampling_hash,
                embedder_hash,
            )

        rep = self._extract_fresh(model_id, config_hash)

        self.cache.save(
            base_model_id,
            adapter_id,
            self.query_key,
            rep,
            max_new_tokens=self.max_new_tokens,
            replicates=self.replicates,
            sampling=sampling,
            embedder_hash=embedder_hash,
            config=config,
            source_indices=self.source_indices,
        )

        return rep

    def _extract_fresh(self, model_id: ModelID, config_hash: str) -> ModelRepresentation:
        model, shared = self._get_model(model_id)
        tokenizer = self._load_tokenizer(model_id, self._resolve_base_model_id(model_id))

        vectors: list[np.ndarray] = []
        all_generated_texts: list[list[str]] = []
        try:
            for i in range(0, len(self.queries), self.batch_size):
                batch_queries = self.queries[i : i + self.batch_size]
                batch_vectors, batch_texts = self._process_batch(
                    model, tokenizer, batch_queries, batch_start=i
                )
                vectors.extend(batch_vectors)
                all_generated_texts.extend(batch_texts)
        finally:
            if not shared:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        matrix = np.stack(vectors, axis=0)  # (N_queries * replicates, d)
        return ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={
                "n_queries": len(self.queries),
                "replicates": self.replicates,
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
                # What `generate` actually saw. Replicates multiply the batch, so
                # a batch_size that fit before may not at R > 1, and the KV-cache
                # reasoning in the experiment YAMLs is about this number.
                "effective_batch": self.batch_size * self.replicates,
            },
        )

    def _seed_for_batch(self, batch_start: int) -> int:
        """A generator seed for one batch, derived from the run's seed.

        Derived per batch rather than set once for the whole extraction so that
        the *n*-th batch draws the same stream no matter what ran before it —
        without this, adding one query to the front of the draw would change
        every generation after it.  Hashed rather than added so that two runs
        whose seeds differ by one do not share their streams offset by one.
        """
        digest = hashlib.sha256(
            f"{self.generation_seed}:{batch_start}".encode()
        ).hexdigest()[:8]
        return int(digest, 16)

    def _process_batch(
        self,
        model: Any,
        tokenizer: Any,
        queries: list[str],
        batch_start: int = 0,
    ) -> tuple[list[np.ndarray], list[list[str]]]:
        inputs = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        if self.do_sample:
            # Seeds the global RNG rather than passing a generator, because
            # `generate` threads one generator through the whole batch anyway —
            # per-sequence streams would need batch_size=1. See the class
            # docstring on what that costs.
            seed = self._seed_for_batch(batch_start)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        sampling_kwargs: dict[str, Any] = {"do_sample": self.do_sample}
        if self.do_sample:
            sampling_kwargs.update(temperature=self.temperature, top_p=self.top_p)
            if self.top_k is not None:
                sampling_kwargs["top_k"] = self.top_k

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.replicates,
                pad_token_id=tokenizer.pad_token_id,
                **sampling_kwargs,
            )

        # `generate` returns (n_queries * R) rows, query-major: the R
        # continuations of query 0, then those of query 1, and so on. That is
        # exactly the row order GeneratedTextCache stores, so no regrouping is
        # needed here beyond nesting the text.
        flat_texts = tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
        )

        R = self.replicates
        vectors = []
        generated_texts: list[list[str]] = []
        for q_index, query in enumerate(queries):
            per_query = flat_texts[q_index * R : (q_index + 1) * R]
            generated_texts.append(per_query)
            for gen_text in per_query:
                output_obj = _InferenceOutput(
                    hidden_states=None,  # behavioral is output-only; no hidden states collected
                    logits=None,
                    generated_text=gen_text,
                )
                vectors.append(self.embedder.embed(output_obj, query))

        return vectors, generated_texts
