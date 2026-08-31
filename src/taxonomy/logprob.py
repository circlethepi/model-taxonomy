"""The log-probability level: what each model *believes* about a shared draw.

The third :class:`~src.taxonomy._hf_inference.HFInferenceTaxonomy` subclass, peer
of :class:`~src.taxonomy.functional.FunctionalTaxonomy` and
:class:`~src.taxonomy.behavioral.BehavioralTaxonomy`.  Those two read a hidden
state and a generated string; this one reads the probability the model assigned.

``mode="input"`` only.  It teacher-forces the shared query text through one
masked forward pass — the same pass the functional level already runs, with no
decoding — and records, at every position, the log-probability of the token that
actually came next and the entropy of the full next-token distribution.  The
generation-mode counterpart is not a second class: it rides along with
:class:`~src.taxonomy.behavioral.BehavioralTaxonomy`'s existing ``generate``
call, because the distributions it needs exist only inside that call.

Why the input mode is worth having on its own: it is defined for every model
over every query, needs no sampling, and compares models on one scale that
nothing else in the pipeline measures.  A generation-based comparison asks "do
these two models say similar things"; this asks "do these two models find the
same text likely", which stays well-defined where the generations are noisy.

**Memory, not time, is the cost.**  Per-token log-probs need logits at every
position, and a modern vocabulary is ~250k wide: at batch 16 × seq 512 that
tensor is ~4 GB in bf16 and a float32 ``log_softmax`` of it ~8 GB more.  So the
softmax is chunked over the sequence axis and the realized token gathered per
chunk (:meth:`_score_chunked`) — identical numbers, bounded memory.
"""

from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch

from src.cache.logprob_cache import LogProbCache
from src.core.protocols import ModelID, Taxonomy
from src.core.representation import ModelRepresentation
from src.taxonomy._hf_inference import HFInferenceTaxonomy


class LogProbTaxonomy(HFInferenceTaxonomy, Taxonomy):
    """Teacher-forced per-token log-probabilities and entropies over one draw.

    Parameters
    ----------
    query_key:
        The ``{recipe_hash, n_samples, seed[, prompt_format_id]}`` key identifying
        the draw in ``01_datasets``.  This — not the query strings — is what goes
        into :meth:`config_dict` and keys the cache, exactly as at the other two
        inference levels.
    batch_size:
        Rows per forward pass.  Bounded by the vocabulary-wide softmax rather
        than by the model: see the module docstring.  It does **not** change the
        stored numbers — every reduction here is mask-aware and every row is
        scored independently — so it stays out of the cache key and lands in the
        run record instead.
    seq_chunk:
        How many sequence positions to run the ``log_softmax`` over at once.  A
        pure memory/throughput knob with no effect on the result.

    What ``extract`` returns
    ------------------------
    A ``(n_queries, 2)`` representation whose columns are the per-query mean
    log-probability and mean entropy over the content positions.  The stored
    artifact is the full per-token detail; this is the smallest summary that
    makes the level usable by the ordinary representation machinery, and it is
    recoverable from the rows rather than the other way round — the same
    superset rule ``docs/notes/TODO.md`` item 12 states for replicates.
    """

    def __init__(
        self,
        queries: Sequence[str],
        query_key: dict | None = None,
        cache: LogProbCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        mode: str = "input",
        max_length: int = 512,
        seq_chunk: int = 64,
        source_indices: list | None = None,
    ) -> None:
        if mode != "input":
            raise ValueError(
                f"LogProbTaxonomy stores mode='input' only, got {mode!r}. "
                "Generation-mode log-probs are collected by BehavioralTaxonomy "
                "with collect_logprobs=True — they exist only inside the "
                "generate() call that drew the tokens."
            )
        super().__init__(
            device=device,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
        )
        self.queries = list(queries)
        self.query_key = dict(query_key or {})
        self.cache = cache
        self.mode = mode
        self.max_length = int(max_length)
        self.seq_chunk = int(seq_chunk)
        self.source_indices = source_indices

    @property
    def taxonomy_name(self) -> str:
        return "logprob"

    def config_dict(self) -> dict[str, Any]:
        # Same shape as the other two levels', and "taxonomy" keeps this from
        # ever hashing equal to a functional config over the same draw.
        # max_length is in because truncating at a different point scores a
        # different span; seq_chunk and batch_size are out because they cannot
        # change the numbers.
        return {
            "taxonomy": "logprob",
            "query_key": self.query_key,
            "n_queries": len(self.queries),
            "mode": self.mode,
            "max_length": self.max_length,
            "torch_dtype": str(self.torch_dtype),
        }

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        if self.cache is None:
            raise ValueError(
                "LogProbTaxonomy requires a LogProbCache. The per-token arrays "
                "are the stored artefact and the returned representation is a "
                "summary of them; there is no in-memory-only path."
            )
        if not self.query_key:
            raise ValueError(
                "LogProbTaxonomy requires query_key — the "
                "{recipe_hash, n_samples, seed} key identifying the draw. It is "
                "what keys the cache."
            )

        base_id, adapter_id = self._model_key(model_id)

        if not self.cache.exists(base_id, adapter_id, self.query_key, self.mode):
            self._extract_fresh(model_id, base_id, adapter_id)

        return self._load(model_id, base_id, adapter_id)

    def _load(self, model_id: ModelID, base_id: str, adapter_id: str) -> ModelRepresentation:
        arrays, meta = self.cache.load_logprobs(
            base_id, adapter_id, self.query_key, self.mode
        )
        lengths = arrays["lengths"]
        start = arrays.get("content_start")
        matrix = np.stack(
            [
                self.cache.masked_mean(arrays["logprob"], lengths, start),
                self.cache.masked_mean(arrays["entropy"], lengths, start),
            ],
            axis=1,
        )
        return ModelRepresentation(
            model_id=str(model_id),
            taxonomy="logprob",
            matrix=matrix.astype(np.float32),
            metadata={
                "base_model_id": base_id,
                "query_key": self.query_key,
                "mode": self.mode,
                "columns": ["mean_logprob", "mean_entropy"],
                "n_queries": int(matrix.shape[0]),
                "scored_positions": int(np.asarray(lengths).sum()),
                "artifact_path": self.cache.artifact_path(
                    base_id, adapter_id, self.query_key
                ),
                "stored_keys": sorted(arrays),
                "run_metadata": meta.get("metadata", {}),
            },
            cache_key=self.cache.cache_key(adapter_id, self.query_key, self.mode),
        )

    def _extract_fresh(self, model_id: ModelID, base_id: str, adapter_id: str) -> None:
        model, shared = self._get_model(model_id)
        tokenizer = self._load_tokenizer(model_id, self._resolve_base_model_id(model_id))

        content_start = self._content_start(tokenizer)

        rows: list[dict[str, np.ndarray]] = []
        try:
            for i in range(0, len(self.queries), self.batch_size):
                batch = self.queries[i : i + self.batch_size]
                rows.extend(self._process_batch(model, tokenizer, batch))
        finally:
            if not shared:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        arrays = _pad_rows(rows)
        arrays["content_start"] = np.minimum(
            np.full(arrays["lengths"].shape, content_start, dtype=np.int64),
            arrays["lengths"],
        )

        self.cache.save_logprobs(
            base_id,
            adapter_id,
            self.query_key,
            self.mode,
            arrays,
            model_id=str(model_id),
            config=self.config_dict(),
            run_metadata={
                # Provenance, deliberately outside config_dict() so it does not
                # fragment the cache — the same rule the other two levels follow.
                "batch_size": self.batch_size,
                "seq_chunk": self.seq_chunk,
                "device_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
                "content_start_tokens": int(content_start),
            },
            source_indices=self.source_indices,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _content_start(self, tokenizer: Any) -> int:
        """Where the query's own text begins, in stored-row coordinates.

        Under a chat template every prompt in a draw opens with the *same*
        scaffolding, so the longest common token prefix across the draw is
        exactly that preamble — measured rather than configured, which means a
        new prompt format needs no edit here and a raw-prompted draw correctly
        reports 0.

        The shift is why this is not simply the prefix length: stored index *j*
        scores the token at real position ``j + 1``, so the first stored index
        that scores a content token is ``prefix - 1``.
        """
        ids = [
            tokenizer(q, add_special_tokens=True, truncation=True,
                      max_length=self.max_length)["input_ids"]
            for q in self.queries
        ]
        if not ids:
            return 0
        prefix = 0
        shortest = min(len(x) for x in ids)
        while prefix < shortest and len({x[prefix] for x in ids}) == 1:
            prefix += 1
        return max(prefix - 1, 0)

    def _process_batch(
        self, model: Any, tokenizer: Any, queries: list[str]
    ) -> list[dict[str, np.ndarray]]:
        """One masked forward pass; one left-aligned row per query.

        The tokenizer pads on the **left** (pinned in ``_load_tokenizer`` so
        batched generation is batch-invariant), which puts the real tokens at the
        right-hand end of each row.  The stored rows are re-packed left-aligned so
        that index 0 is the first scored position and ``lengths`` means what it
        says; without that, ``lengths`` would have to be read against a per-row
        offset that depends on which other queries shared the batch.
        """
        inputs = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        with torch.no_grad():
            logits = model(**inputs).logits

        # Position t predicts token t+1, so the scored span is one shorter.
        targets = input_ids[:, 1:]
        logprob, entropy = self._score_chunked(logits[:, :-1, :], targets)
        valid = (mask[:, :-1] * mask[:, 1:]).bool()  # both context and target real

        out = []
        for r in range(input_ids.shape[0]):
            keep = valid[r]
            out.append(
                {
                    "logprob": logprob[r][keep].float().cpu().numpy(),
                    "entropy": entropy[r][keep].float().cpu().numpy(),
                    "token_id": targets[r][keep].cpu().numpy(),
                }
            )
        return out

    def _score_chunked(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(logprob, entropy)`` of shape ``(batch, seq)`` from ``(batch, seq, V)``.

        Chunked over the sequence axis, in float32, gathering the realized token
        inside each chunk so the full distribution is never held for more than
        ``seq_chunk`` positions at a time.  The arithmetic is the ordinary
        ``log_softmax``; the chunking is only about peak memory, and the numbers
        are identical to the unchunked form.
        """
        b, s, _ = logits.shape
        logprob = torch.empty((b, s), dtype=torch.float32, device=logits.device)
        entropy = torch.empty((b, s), dtype=torch.float32, device=logits.device)
        for lo in range(0, s, self.seq_chunk):
            hi = min(lo + self.seq_chunk, s)
            z = logits[:, lo:hi, :].float()
            logp = z - torch.logsumexp(z, dim=-1, keepdim=True)
            logprob[:, lo:hi] = logp.gather(-1, targets[:, lo:hi, None]).squeeze(-1)
            # p·log p with p = exp(logp); summed in float32 over the full vocab.
            entropy[:, lo:hi] = -(logp.exp() * logp).sum(dim=-1)
            del z, logp
        return logprob, entropy


def _pad_rows(
    rows: list[dict[str, np.ndarray]], lengths: list[int] | None = None
) -> dict[str, np.ndarray]:
    """Stack variable-length left-aligned rows into ``(rows, T_max)`` arrays.

    Shared with :class:`~src.taxonomy.behavioral.BehavioralTaxonomy`'s
    ride-along collection, so both modes of ``05a_logprobs`` are padded by one
    piece of code and cannot disagree about the convention.

    Padding is zero and is *not* meaningful: ``lengths`` is the only thing that
    says where a row ends, which is why it is stored rather than inferred from a
    sentinel value a real log-prob could collide with.  Pass *lengths* when a row
    is stored at full width but only part of it is real — which is what a decode
    that hit EOS early produces.
    """
    if not rows:
        raise ValueError("no rows to store; the draw produced no scored positions")
    keys = list(rows[0])
    width = max(len(r[keys[0]]) for r in rows)
    out = {
        k: np.zeros(
            (len(rows), width),
            dtype=np.int64 if k == "token_id" else np.float32,
        )
        for k in keys
    }
    real = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        n = len(row[keys[0]])
        real[i] = n if lengths is None else int(lengths[i])
        for k in keys:
            out[k][i, :n] = row[k]
    out["lengths"] = real
    return out
