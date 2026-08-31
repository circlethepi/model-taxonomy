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
from src.cache.logprob_cache import LogProbCache
from src.taxonomy._hf_inference import HFInferenceTaxonomy
from src.taxonomy.logprob import _pad_rows


@dataclass
class _InferenceOutput:
    """Unified container for the output of one generation call over one query."""

    hidden_states: tuple | None
    logits: "torch.Tensor | None"
    generated_text: str | None


#: The token a reasoning model emits to end its thinking block.  Spelled once
#: here rather than in a per-model profile: it is derivable from any tokenizer
#: that has one, so a new reasoning model is instrumented without a config edit,
#: and every model without one yields None and disables the whole code path.
_THINK_CLOSE_TOKEN = "</think>"


def _think_close_token_id(tokenizer) -> int | None:
    """The id of ``</think>``, or None if this tokenizer has no such token.

    Guarded against ``unk``: ``convert_tokens_to_ids`` maps anything it does not
    know to the unknown-token id rather than failing, and treating that as a real
    match would make every generation look like it closed a reasoning block on
    whatever token unk happens to be.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        tid = convert(_THINK_CLOSE_TOKEN)
    except Exception:
        return None
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(tid)


def stop_token_ids(model, tokenizer) -> set[int]:
    """Every id that ends a generated row, as *this* checkpoint defines ending.

    One definition, because two callers have to agree: ``generate`` is invoked
    without an explicit ``eos_token_id`` and therefore stops on whatever
    ``model.generation_config.eos_token_id`` names, while the trim below decides
    where a row *ended*.  Derive them from different places and they disagree by
    construction -- the same argument :func:`render_prompt` makes for training
    and extraction prompts.

    Three sources, unioned:

    ``pad_token_id`` and ``tokenizer.eos_token_id`` are the pair this used to
    consider alone.  They are usually the same token (the tokenizer falls back to
    eos for pad) but need not be: with a distinct pad, the terminating eos is a
    real choice and the run of pads after it is not.

    ``generation_config.eos_token_id`` is the addition, and it is what makes this
    correct for an instruct checkpoint.  A chat model ends a *turn* with one token
    and a *sequence* with another -- Llama-3.x emits ``<|eot_id|>`` while
    ``tokenizer.eos_token_id`` is ``<|end_of_text|>`` -- and the generation config
    lists both.  Without it, a row that ended at its turn boundary is scored as
    running to the full token budget, silently, in every log-prob-bearing job.
    HuggingFace allows a scalar or a list here, so both shapes are normalised.

    Nothing model-specific is named: a checkpoint that declares one stop token
    yields one, and a raw base model with neither pad nor eos yields the empty
    set, which disables the trim exactly as before.
    """
    ids: set[int] = set()
    for tid in (getattr(tokenizer, "pad_token_id", None),
                getattr(tokenizer, "eos_token_id", None)):
        if tid is not None:
            ids.add(int(tid))

    gen_cfg = getattr(model, "generation_config", None)
    declared = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if declared is not None:
        if isinstance(declared, (list, tuple, set)):
            ids.update(int(t) for t in declared if t is not None)
        else:
            ids.add(int(declared))
    return ids


class BehavioralTaxonomy(HFInferenceTaxonomy, Taxonomy):
    """Extracts behavioral representations of HuggingFace language models.

    For each model, generates continuations for a set of query strings and uses
    the provided embedder to convert each generated output into a fixed-size
    vector.  The stacked vectors form the (N_queries * replicates, d) matrix
    representation, in query-major order.

    This taxonomy's *representation* is built **exclusively from generated text**
    — it collects no hidden states.  Use :class:`FunctionalTaxonomy` if you need
    activation-based comparison.  With ``collect_logprobs=True`` it additionally
    records the per-token log-probabilities of the text it drew, into
    ``05a_logprobs`` beside the generations; that is a second artifact from the
    same pass, not a change to what the behavioral matrix contains.

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
    collect_logprobs, logprob_cache:
        Collect the per-token log-probability and entropy of every generated
        token and store them in ``05a_logprobs`` under the *same* variant token as
        the generations, so the two files join by name.

        **Two distributions are stored, not one.**  ``generate`` runs the raw
        logits ``z`` through a chain of ``LogitsProcessor``s before sampling —
        temperature divides, top-p/top-k mask — so ``scores`` is the distribution
        the token was actually drawn from and ``logits`` is the model's own
        belief.  Across a temperature sweep only the second is comparable
        between settings, and the first is not recoverable from it: recovering
        ``log softmax(z/T)`` needs ``logsumexp`` over the whole vocabulary, which
        is discarded.  So both are kept — ``logprob``/``entropy`` from ``scores``,
        ``logprob_raw``/``entropy_raw`` from ``logits`` — for one extra gather and
        no extra forward pass.  At ``T=1.0, top_p=1.0, top_k=None`` every
        processor is the identity and the two must coincide, which is a free
        consistency check on a sweep.

        The unprocessed quantity is the *same* quantity
        :class:`~src.taxonomy.logprob.LogProbTaxonomy` stores in input mode, so
        the two sit on one scale.

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
        collect_logprobs: bool = False,
        logprob_cache: LogProbCache | None = None,
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
        if collect_logprobs and logprob_cache is None:
            raise ValueError(
                "collect_logprobs=True requires a logprob_cache. The arrays are "
                "the whole point of collecting them and there is nowhere else to "
                "put them; failing here beats discovering it after the decode."
            )
        self.collect_logprobs = bool(collect_logprobs)
        self.logprob_cache = logprob_cache
        # Filled by _extract_fresh when collecting; read by extract() to write
        # the 05a_logprobs entry beside the generations it describes.
        self._logprob_arrays: dict[str, np.ndarray] | None = None

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
        ) and self._logprobs_on_disk(base_model_id, adapter_id, sampling_hash):
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

        if self.collect_logprobs and self._logprob_arrays is not None:
            self.logprob_cache.save_logprobs(
                base_model_id,
                adapter_id,
                self.query_key,
                "generation",
                self._logprob_arrays,
                max_new_tokens=self.max_new_tokens,
                replicates=self.replicates,
                sampling=sampling,
                model_id=str(model_id),
                config=config,
                run_metadata={
                    "batch_size": self.batch_size,
                    "effective_batch": self.batch_size * self.replicates,
                    "device_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else "cpu"
                    ),
                },
                source_indices=self.source_indices,
            )

        return rep

    def _logprobs_on_disk(
        self, base_model_id: str, adapter_id: str, sampling_hash: str
    ) -> bool:
        """Whether the log-prob half of this entry is already stored.

        Not collecting log-probs makes this vacuously true, which is what keeps
        every existing suite's cache hit exactly as it was.  When collecting, it
        has to be part of the hit test: the generations and their embedding are
        written by a different cache, so an entry from an earlier run without
        ``collect_logprobs`` would otherwise short-circuit here, return the
        cached text and silently produce no log-probs at all.

        Re-generating on that miss is exact rather than approximate — greedy is
        deterministic, and sampling reproduces at the same ``generation_seed``
        **and the same ``batch_size``**, which is why that value is pinned in the
        sweep configs and stays out of the cache key.
        """
        if not self.collect_logprobs:
            return True
        return self.logprob_cache.exists(
            base_model_id,
            adapter_id,
            self.query_key,
            "generation",
            max_new_tokens=self.max_new_tokens,
            replicates=self.replicates,
            sampling_hash=sampling_hash,
        )

    def _extract_fresh(self, model_id: ModelID, config_hash: str) -> ModelRepresentation:
        model, shared = self._get_model(model_id)
        tokenizer = self._load_tokenizer(model_id, self._resolve_base_model_id(model_id))

        # Resolved once per extraction, from the tokenizer rather than from a
        # per-model table: a model that grows a thinking mode gets instrumented
        # without a config edit, and every model that has no such token gets
        # None, which makes the whole block below a no-op.  That is what keeps
        # the existing Llama suite bit-identical.
        self._think_close_id = _think_close_token_id(tokenizer)

        vectors: list[np.ndarray] = []
        all_generated_texts: list[list[str]] = []
        all_closed_at: list[list[int | None]] = []
        lp_rows: list[dict[str, np.ndarray]] = []
        lp_lengths: list[int] = []
        try:
            for i in range(0, len(self.queries), self.batch_size):
                batch_queries = self.queries[i : i + self.batch_size]
                batch_vectors, batch_texts, batch_closed, batch_lp, batch_lp_len = (
                    self._process_batch(
                        model, tokenizer, batch_queries, batch_start=i
                    )
                )
                vectors.extend(batch_vectors)
                all_generated_texts.extend(batch_texts)
                all_closed_at.extend(batch_closed)
                lp_rows.extend(batch_lp)
                lp_lengths.extend(batch_lp_len)
        finally:
            if not shared:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Padded here rather than per batch: a batch whose sequences all hit EOS
        # early stops short, so widths differ between batches and only the whole
        # run knows T_max.  Rows are query-major, the same order as `matrix`.
        self._logprob_arrays = _pad_rows(lp_rows, lp_lengths) if lp_rows else None

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
                # Validity statistic for reasoning models; None-valued and inert
                # for every other model.  Outside config_dict() for the same
                # reason device_name is: a diagnostic must never fragment the
                # cache.  See _think_closure_summary for what it is for.
                "think_closure": self._think_closure_summary(all_closed_at),
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
    ) -> tuple[
        list[np.ndarray],
        list[list[str]],
        list[list[int | None]],
        list[dict[str, np.ndarray]],
        list[int],
    ]:
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

        if self.collect_logprobs:
            # `scores` is what the sampler saw, `logits` what the model emitted.
            # Both, because they are different quantities under a temperature and
            # neither is recoverable from the other — see the class docstring.
            sampling_kwargs.update(
                return_dict_in_generate=True, output_scores=True, output_logits=True
            )

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.replicates,
                pad_token_id=tokenizer.pad_token_id,
                **sampling_kwargs,
            )

        if self.collect_logprobs:
            output_ids = generated.sequences
            lp_rows, lp_lengths = self._gather_logprobs(
                generated, output_ids[:, input_len:], tokenizer, model
            )
            del generated
        else:
            output_ids = generated
            lp_rows, lp_lengths = [], []

        # `generate` returns (n_queries * R) rows, query-major: the R
        # continuations of query 0, then those of query 1, and so on. That is
        # exactly the row order GeneratedTextCache stores, so no regrouping is
        # needed here beyond nesting the text.
        gen_ids = output_ids[:, input_len:]
        think_close_id = getattr(self, "_think_close_id", None)

        if think_close_id is None:
            flat_closed: list[int | None] = [None] * gen_ids.shape[0]
            flat_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        else:
            # Two jobs, both done on raw ids before decoding.
            #
            # 1. Record *where* the model closed its reasoning block, if it did.
            #    Read off ids rather than matched in text because it is exact and
            #    needs no string search, and because it is the statistic that
            #    says whether this level measured answers or reasoning traces.
            #
            # 2. Embed the answer, not the reasoning.  `</think>` is in the added
            #    vocabulary but NOT in all_special_tokens, so skip_special_tokens
            #    does not remove it: without this split every stored generation
            #    would carry a "</think>" preamble into the embedder, identical
            #    across all adapters and absent from the raw-prompted suites.
            #    A sequence that never closed is kept whole — reasoning is all it
            #    produced, and silently emitting an empty string would be worse
            #    than a low closure_rate that says so.
            flat_closed = []
            pieces = []
            hits = gen_ids == think_close_id
            for row in range(gen_ids.shape[0]):
                nz = hits[row].nonzero()
                if nz.numel():
                    k = int(nz[0, 0])
                    flat_closed.append(k)
                    pieces.append(gen_ids[row, k + 1 :])
                else:
                    flat_closed.append(None)
                    pieces.append(gen_ids[row])
            flat_texts = [
                tokenizer.decode(p, skip_special_tokens=True).strip() for p in pieces
            ]

        R = self.replicates
        vectors = []
        generated_texts: list[list[str]] = []
        closed_at: list[list[int | None]] = []
        for q_index, query in enumerate(queries):
            per_query = flat_texts[q_index * R : (q_index + 1) * R]
            generated_texts.append(per_query)
            closed_at.append(flat_closed[q_index * R : (q_index + 1) * R])
            for gen_text in per_query:
                output_obj = _InferenceOutput(
                    hidden_states=None,  # behavioral is output-only; no hidden states collected
                    logits=None,
                    generated_text=gen_text,
                )
                vectors.append(self.embedder.embed(output_obj, query))

        return vectors, generated_texts, closed_at, lp_rows, lp_lengths

    def _gather_logprobs(
        self, generated: Any, gen_ids: torch.Tensor, tokenizer: Any, model: Any
    ) -> tuple[list[dict[str, np.ndarray]], list[int]]:
        """Per-token log-probs and entropies of one batch's generated tokens.

        One row per generated sequence, in the query-major order ``generate``
        returns and the behavioral matrix stores, so row *i* means the same thing
        in ``05_generated`` and ``05a_logprobs``.

        Reduced **step by step**: each step's ``(rows, V)`` distribution is turned
        into two scalars per row and dropped, so the peak here is one step, not
        the whole 128-step stack.  ``generate`` has already accumulated both
        stacks on device — that is the memory the sweep configs budget for — and
        the point of reducing eagerly is not to add a third copy on top of them.

        ``lengths`` stops at the first stop token -- see :func:`stop_token_ids`.
        Once a sequence has finished, ``generate`` keeps stepping it with pad and
        the distributions from those steps describe nothing the model chose; the
        token that ended it *is* a real choice and is counted.
        """
        scores = generated.scores
        raw = generated.logits
        n_steps = len(scores)
        rows = gen_ids.shape[0]

        cols = {
            "logprob": np.zeros((rows, n_steps), dtype=np.float32),
            "entropy": np.zeros((rows, n_steps), dtype=np.float32),
            "logprob_raw": np.zeros((rows, n_steps), dtype=np.float32),
            "entropy_raw": np.zeros((rows, n_steps), dtype=np.float32),
            "token_id": np.zeros((rows, n_steps), dtype=np.int64),
        }
        for step in range(n_steps):
            tok = gen_ids[:, step]
            cols["token_id"][:, step] = tok.cpu().numpy()
            for src, lp_key, ent_key in (
                (scores[step], "logprob", "entropy"),
                (raw[step], "logprob_raw", "entropy_raw"),
            ):
                z = src.float()
                logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
                cols[lp_key][:, step] = (
                    logp.gather(-1, tok[:, None]).squeeze(-1).cpu().numpy()
                )
                # -inf log-probs are real: top-p/top-k mask tokens out, and a
                # masked token contributes 0 to the entropy rather than NaN.
                p = logp.exp()
                cols[ent_key][:, step] = (
                    -(torch.where(p > 0, p * logp, torch.zeros_like(p)))
                    .sum(dim=-1)
                    .cpu()
                    .numpy()
                )
                del z, logp, p

        # Whatever this checkpoint says ends a row -- pad, tokenizer eos, and the
        # turn-end tokens the generation config declares.  See stop_token_ids for
        # why all three, and why a turn end is not the same thing as a sequence end.
        stop_ids = stop_token_ids(model, tokenizer)
        ids = gen_ids.cpu().numpy()
        is_stop = np.isin(ids, list(stop_ids)) if stop_ids else np.zeros_like(ids, bool)
        lengths = []
        for r in range(rows):
            hit = np.nonzero(is_stop[r])[0]
            lengths.append(int(hit[0]) + 1 if hit.size else n_steps)

        per_row = [
            {k: v[r] for k, v in cols.items()} for r in range(rows)
        ]
        return per_row, lengths

    def _think_closure_summary(self, closed_at: list[list[int | None]]) -> dict | None:
        """Did the model finish reasoning inside its token budget?

        ``None`` for every model without a ``</think>`` token, which keeps this
        absent from the existing suites' metadata entirely.

        Why it matters: the behavioral representation is the *embedding of the
        generated text*.  If a sequence spends its whole budget reasoning, what
        nomic embeds is a reasoning trace, not an answer — a coherent thing to
        measure, but a different construct from the one the raw-prompted suites
        measured, so a mixture of the two averages two constructs together and
        the cross-suite comparison stops being like-for-like.  This is therefore
        a validity statistic for the level, not a performance metric, and it is
        reported rather than enforced.

        ``closed_at`` is nested exactly like ``generated_texts`` — one list per
        query, one entry per replicate — so the two can be read side by side.
        """
        if getattr(self, "_think_close_id", None) is None:
            return None
        flat = [c for per_query in closed_at for c in per_query]
        closed = [c for c in flat if c is not None]
        return {
            "token_id": self._think_close_id,
            "closed_at": closed_at,
            "closure_rate": (len(closed) / len(flat)) if flat else None,
            "median_close_index": (
                float(np.median(closed)) if closed else None
            ),
            "max_new_tokens": self.max_new_tokens,
        }
