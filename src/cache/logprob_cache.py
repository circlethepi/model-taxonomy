"""``05a_logprobs`` — per-token log-probabilities and entropies.

The third stage keyed by :class:`~src.cache._draw_keyed.DrawKeyedCache`, and
the first that stores what the model *believes* rather than where it sits or
what it said.  ``04_activations`` holds a hidden state, ``05_generated`` holds
text and its embedding; neither reads the probability the model assigned.

Directory layout::

    cache_root/05a_logprobs/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}[_f{fmt}]/
        queries.json                        ← query_key + source row indices
        runs/{config_hash}.json             ← extraction provenance
        logprobs/input.safetensors
        logprobs/{variant_token}.safetensors

The prefix through the draw directory is byte-identical to the functional and
behavioral stages', so one model under one draw sits at the same coordinates in
all three trees and they can be read side by side.  That co-location is the
point of ``docs/notes/TODO.md`` item 13.

**Generation filenames reuse
:meth:`~src.cache.generated_text_cache.GeneratedTextCache.variant_token`
verbatim** — the attribute below is the same function object, not a
reimplementation of the same spelling.  A log-prob file therefore carries the
same token as the ``generations/{token}.json`` it describes, and the two join by
name with no lookup.  Every result-changing axis (token budget, replicate count,
sampling settings) is in that name, which is the invariant ``sampling_hash``
exists to protect: :meth:`save` is idempotent on filename, so an axis left out
of the name makes a second run at a different setting a silent no-op that
returns the first run's numbers.

**Two distributions are stored for generation mode, not one.**  ``logprob`` /
``entropy`` come from the *processed* logits — temperature- and top-p-warped,
i.e. the distribution the token was actually drawn from — and ``logprob_raw`` /
``entropy_raw`` from the unprocessed model output.  Across a temperature sweep
these are different quantities and only the raw pair is comparable between
settings; the warped pair is not recoverable from it, because
``log softmax(z/T)[i] = z[i]/T - logsumexp(z/T)`` needs the whole 248,320-token
logit vector, which is discarded.  Input mode is teacher-forced with no decoding
at all, so it has no processed/unprocessed split: its ``logprob`` *is* the raw
quantity, on the same scale as ``logprob_raw`` above.

**Rows are padded and query-major**, matching the behavioral matrix layout
exactly, so row *i* means the same thing in both stages.  ``lengths`` gives the
real extent of each row; positions at or beyond it are padding and carry no
meaning.  ``content_start`` (input mode) is the first non-scaffolding position
under the chat template.  Storing the index rather than pre-trimming keeps the
cache the superset — TODO item 12's principle: the trimmed view is recoverable
from the full rows, the full rows are not recoverable from the trimmed view.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.cache._draw_keyed import DrawKeyedCache, adapter_slug
from src.cache.generated_text_cache import GeneratedTextCache

__all__ = ["LogProbCache"]


class LogProbCache(DrawKeyedCache):
    """Cache for the log-probability level: ``05a_logprobs``."""

    _STAGE_DIR = "05a_logprobs"
    _ARTIFACT_DIR = "logprobs"

    # ------------------------------------------------------------------
    # Filename spelling, shared with 05_generated
    # ------------------------------------------------------------------

    #: The generation variant token, *the same function object* as
    #: ``GeneratedTextCache.variant_token``.  Bound here rather than respelled so
    #: the two stages cannot drift apart the way ``mode_token`` once could;
    #: ``scripts/check_analysis.py`` asserts the identity.
    variant_token = GeneratedTextCache.variant_token

    #: Likewise for the decoding-settings hash and the keys it accepts.  A
    #: log-prob file and the generations file it describes must agree on the
    #: 8-hex token or the join by name silently misses.
    SAMPLING_KEYS = GeneratedTextCache.SAMPLING_KEYS
    GREEDY_SAMPLING = GeneratedTextCache.GREEDY_SAMPLING
    sampling_hash = GeneratedTextCache.sampling_hash

    #: Arrays every entry must carry.  ``token_id`` is stored even though the
    #: text is derivable elsewhere: it is what makes an entry self-checking, and
    #: for generation mode it is the only record of *which* token each log-prob
    #: scores without re-tokenizing the generations file.
    REQUIRED_KEYS = ("logprob", "entropy", "token_id", "lengths")

    #: Accepted beyond the required set.  ``content_start`` is input mode's
    #: scaffolding boundary; the ``_raw`` pair is generation mode's unprocessed
    #: distribution.  Anything else is rejected rather than written, so a typo
    #: cannot land a silently-ignored array in a stored entry.
    OPTIONAL_KEYS = ("content_start", "logprob_raw", "entropy_raw")

    #: Which arrays are integer-valued.  Everything else is float32.
    _INT_KEYS = frozenset({"token_id", "lengths", "content_start"})

    #: Arrays shaped ``(rows,)`` rather than ``(rows, T)``.
    _PER_ROW_KEYS = frozenset({"lengths", "content_start"})

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def artifact_name(
        self,
        mode: str,
        *,
        max_new_tokens: int | None = None,
        replicates: int | None = None,
        sampling_hash: str | None = None,
    ) -> str:
        """``input`` or ``generation{N}_{R}r_{sampling8}``, plus the extension.

        Input mode needs no variant token: teacher-forced scoring has no
        decoding settings to vary, so one draw admits exactly one input entry.
        """
        if mode == "input":
            return "input.safetensors"
        if mode == "generation":
            if replicates is None or sampling_hash is None:
                raise ValueError(
                    "generation mode requires replicates and sampling_hash; they "
                    "are in the filename, and omitting them would let two "
                    "decoding settings overwrite each other"
                )
            token = self.variant_token(max_new_tokens, replicates, sampling_hash)
            return f"{token}.safetensors"
        raise ValueError(
            f"{mode!r} is not a stored mode; only 'input' and 'generation' are "
            "written to disk."
        )

    def logprob_path(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        *,
        max_new_tokens: int | None = None,
        replicates: int | None = None,
        sampling_hash: str | None = None,
    ) -> Path:
        name = self.artifact_name(
            mode,
            max_new_tokens=max_new_tokens,
            replicates=replicates,
            sampling_hash=sampling_hash,
        )
        return (
            self.draw_dir(base_model_id, adapter_id, query_key)
            / self._ARTIFACT_DIR
            / name
        )

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    def exists(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        *,
        max_new_tokens: int | None = None,
        replicates: int | None = None,
        sampling_hash: str | None = None,
    ) -> bool:
        return self.logprob_path(
            base_model_id,
            adapter_id,
            query_key,
            mode,
            max_new_tokens=max_new_tokens,
            replicates=replicates,
            sampling_hash=sampling_hash,
        ).exists()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_logprobs(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        arrays: dict[str, np.ndarray],
        *,
        max_new_tokens: int | None = None,
        replicates: int | None = None,
        sampling: dict | None = None,
        model_id: str | None = None,
        config: dict | None = None,
        run_metadata: dict | None = None,
        source_indices: list | None = None,
    ) -> None:
        """Write one entry, atomically, leaving any existing one alone.

        Idempotent on filename, like every other stage here: a second run at the
        same mode and variant is a no-op.  That is safe precisely because every
        axis that changes the numbers is in the filename — see the module
        docstring.
        """
        from filelock import FileLock

        arrays = self._validate(arrays, mode)
        sampling_hash = (
            self.sampling_hash(sampling) if mode == "generation" else None
        )

        draw = self.draw_dir(base_model_id, adapter_id, query_key)
        (draw / self._ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
        (draw / "runs").mkdir(parents=True, exist_ok=True)

        with FileLock(str(draw / ".lock")):
            path = self.logprob_path(
                base_model_id,
                adapter_id,
                query_key,
                mode,
                max_new_tokens=max_new_tokens,
                replicates=replicates,
                sampling_hash=sampling_hash,
            )
            if path.exists():
                return

            self._write_queries_once(draw, query_key, source_indices)

            if config is not None:
                self._write_run_record(
                    draw,
                    config,
                    {
                        "mode": mode,
                        "max_new_tokens": max_new_tokens,
                        "replicates": replicates,
                        "sampling": dict(sampling) if sampling else None,
                        "sampling_hash": sampling_hash,
                        "model_id": model_id,
                        "stored_keys": sorted(arrays),
                        **(run_metadata or {}),
                    },
                )

            payload = dict(arrays)
            payload["_meta_json"] = np.frombuffer(
                json.dumps(
                    {
                        "schema_version": "1",
                        "model_id": model_id,
                        "base_model_id": base_model_id,
                        "taxonomy": "logprob",
                        "query_key": query_key,
                        "mode": mode,
                        "max_new_tokens": max_new_tokens,
                        "replicates": replicates,
                        "sampling": dict(sampling) if sampling else None,
                        "sampling_hash": sampling_hash,
                        "keys": sorted(arrays),
                        "metadata": dict(run_metadata or {}),
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            )
            _atomic_save_tensors(path, payload)

    def _validate(self, arrays: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
        """Reject unknown keys, disagreeing shapes and the wrong dtypes.

        Unknown keys are an error rather than a silent drop for the same reason
        :meth:`sampling_hash` rejects them: a stored entry that quietly lacks an
        array reads as "this quantity was not measured", which is indistinguishable
        from a run that genuinely did not measure it.
        """
        unknown = set(arrays) - set(self.REQUIRED_KEYS) - set(self.OPTIONAL_KEYS)
        if unknown:
            raise ValueError(
                f"unknown log-prob array(s) {sorted(unknown)}; expected a subset of "
                f"{list(self.REQUIRED_KEYS + self.OPTIONAL_KEYS)}"
            )
        missing = set(self.REQUIRED_KEYS) - set(arrays)
        if missing:
            raise ValueError(f"missing required log-prob array(s) {sorted(missing)}")
        if mode == "input" and ("logprob_raw" in arrays or "entropy_raw" in arrays):
            raise ValueError(
                "input mode has no processed/unprocessed split — its logprob is "
                "already the unwarped quantity; storing a '_raw' copy would "
                "suggest the two differ"
            )

        rows = int(np.asarray(arrays["lengths"]).shape[0])
        width = int(np.asarray(arrays["logprob"]).shape[1])
        out: dict[str, np.ndarray] = {}
        for key, arr in arrays.items():
            a = np.asarray(arr)
            want = (rows,) if key in self._PER_ROW_KEYS else (rows, width)
            if a.shape != want:
                raise ValueError(
                    f"{key!r} has shape {a.shape}, expected {want}; every array in "
                    "one entry describes the same rows"
                )
            dtype = np.int32 if key in self._INT_KEYS else np.float32
            out[key] = np.ascontiguousarray(a.astype(dtype))
        return out

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_logprobs(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        *,
        max_new_tokens: int | None = None,
        replicates: int | None = None,
        sampling_hash: str | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """The stored arrays and the ``_meta_json`` blob beside them."""
        from safetensors.numpy import load_file

        path = self.logprob_path(
            base_model_id,
            adapter_id,
            query_key,
            mode,
            max_new_tokens=max_new_tokens,
            replicates=replicates,
            sampling_hash=sampling_hash,
        )
        if not path.exists():
            raise FileNotFoundError(
                f"no stored log-probs for {adapter_id} under draw "
                f"{self.draw_name(query_key)} "
                f"({self.artifact_name(mode, max_new_tokens=max_new_tokens, replicates=replicates, sampling_hash=sampling_hash)}) "
                f"at {path}"
            )
        tensors = load_file(str(path))
        meta = json.loads(tensors.pop("_meta_json").tobytes().decode("utf-8"))
        return tensors, meta

    @staticmethod
    def masked_mean(
        values: np.ndarray, lengths: np.ndarray, start: np.ndarray | None = None
    ) -> np.ndarray:
        """Per-row mean over the real positions, ignoring padding.

        The one reduction every reader of this stage needs, kept here so the
        padding convention is applied in one place rather than re-derived at each
        call site.  ``start`` trims leading scaffolding (``content_start``); the
        stored rows keep it, and this is how a caller drops it.
        """
        values = np.asarray(values, dtype=np.float64)
        rows, width = values.shape
        idx = np.arange(width)[None, :]
        lo = np.zeros((rows, 1)) if start is None else np.asarray(start).reshape(-1, 1)
        mask = (idx >= lo) & (idx < np.asarray(lengths).reshape(-1, 1))
        counts = mask.sum(axis=1)
        totals = np.where(mask, values, 0.0).sum(axis=1)
        return np.where(counts > 0, totals / np.where(counts > 0, counts, 1), np.nan)

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_entries(
        self, base_model_id: str, adapter_id: str, query_key: dict
    ) -> list[str]:
        """Artifact stems stored for one draw — ``input``, ``generation128_8r_…``.

        A directory listing, no file opens: the peer of
        ``GeneratedTextCache.list_variants``.
        """
        d = self.draw_dir(base_model_id, adapter_id, query_key) / self._ARTIFACT_DIR
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.safetensors"))

    def cache_key(
        self,
        adapter_id: str,
        query_key: dict,
        mode: str,
        *,
        max_new_tokens: int | None = None,
        replicates: int | None = None,
        sampling_hash: str | None = None,
    ) -> str:
        """``{adapter}/{draw}/{stem}`` — how one entry names itself to a reader."""
        stem = self.artifact_name(
            mode,
            max_new_tokens=max_new_tokens,
            replicates=replicates,
            sampling_hash=sampling_hash,
        ).removesuffix(".safetensors")
        return f"{adapter_slug(adapter_id)}/{self.draw_name(query_key)}/{stem}"


def _atomic_save_tensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write several named arrays as one safetensors file, atomically."""
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp))
    os.replace(tmp, path)
