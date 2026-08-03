from __future__ import annotations

import gc
from typing import Any, Literal, Sequence

import numpy as np
import torch

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation
from src.cache.activation_cache import ActivationCache
from src.taxonomy._hf_inference import HFInferenceTaxonomy


class FunctionalTaxonomy(HFInferenceTaxonomy, Taxonomy):
    """Compares models via their internal activations on a shared query set.

    For each query and each layer, the model's hidden states are pooled to a
    single vector.  The pooled ``(n_queries, d)`` matrix for each layer is stored
    **separately** by :class:`~src.cache.activation_cache.ActivationCache`, and
    the representation handed to the comparison layer is a *view* assembled at
    read time.

    That split is the point.  One forward pass produces every layer at once, so
    "which layers am I comparing?" is a question about reading, not about running
    a model.  By default extraction stores all of them (~23 MB for a 3B model at
    64 queries) and ``load`` concatenates across them.

    Views
    -----
    ``"concat"`` (default)
        Activations concatenated across the selected layers, ``(n_queries,
        L·d)``.  Row *i* is query *i*.  This is what feeds the distance metrics.
    ``"gram"``
        ``G = H Hᵀ`` of that concatenation, ``(n_queries, n_queries)``.  Rows are
        still queries.  This is a **kernel**, not a feature matrix: handing it to
        a metric that forms its own kernel computes ``(H Hᵀ)²``.  Representations
        built from it are tagged ``metadata["is_kernel"]`` so metrics can refuse.

    Activation modes
    ----------------
    ``"input"`` (default)
        Forward pass on the prompt only.
    ``"generation"``
        Activations collected during decoding; at each step the last-token hidden
        state is taken per layer and mean-pooled across steps.
    ``"both"``
        Runs both passes and stores them as separate files, combined at read
        time.  Because the two are stored separately, a draw that already has
        ``input`` can gain ``generation`` later without recomputing either.

    Parameters
    ----------
    query_key:
        The ``{recipe_hash, n_samples, seed}`` triple identifying the query draw
        in ``01_datasets``.  This — not the query strings — is what goes into
        :meth:`config_dict` and what keys the cache.
    layer_indices:
        Layers to store.  ``None`` (the default) stores every hidden state, which
        is what makes layer choice a read-time decision.  Negative indices are
        resolved against the model's actual hidden-state count before anything
        touches disk.
    normalize_activations:
        Applies at *read* time, not extraction: raw activations are stored, and
        normalization is part of a surrogate's identity.  Kept here as the
        default passed to :meth:`ActivationCache.load`.
    """

    def __init__(
        self,
        queries: Sequence[str],
        layer_indices: list[int] | None = None,
        query_key: dict | None = None,
        cache: ActivationCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        pooling: Literal["mean", "last_token", "cls"] = "mean",
        normalize_activations: bool = True,
        activation_mode: Literal["input", "generation", "both"] = "input",
        max_new_tokens: int = 32,
        view: Literal["concat", "gram"] = "concat",
        source_indices: list | None = None,
    ) -> None:
        if activation_mode not in ("input", "generation", "both"):
            raise ValueError(f"Unknown activation_mode: {activation_mode!r}")
        if activation_mode in ("generation", "both") and max_new_tokens <= 0:
            raise ValueError(
                f"activation_mode={activation_mode!r} requires max_new_tokens > 0."
            )

        super().__init__(
            device=device,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            hf_token=hf_token,
        )
        self.queries = list(queries)
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.query_key = dict(query_key or {})
        self.cache = cache
        self.pooling = pooling
        self.normalize_activations = normalize_activations
        self.activation_mode = activation_mode
        self.max_new_tokens = max_new_tokens
        self.view = view
        self.source_indices = source_indices

    @property
    def taxonomy_name(self) -> str:
        return "functional"

    def config_dict(self) -> dict[str, Any]:
        # "taxonomy" keeps a functional config from ever hashing equal to a
        # behavioral one and makes the run record self-describing.  The query
        # *key* rather than the strings: hashing the strings would make every
        # entry sensitive to any upstream change that shifts the draw, with no
        # way to tell from a key which draw it belonged to.
        return {
            "taxonomy": "functional",
            "query_key": self.query_key,
            "n_queries": len(self.queries),
            "layer_indices": self.layer_indices,
            "pooling": self.pooling,
            "activation_mode": self.activation_mode,
            "max_new_tokens": self.max_new_tokens if self.activation_mode != "input" else None,
            "torch_dtype": str(self.torch_dtype),
        }

    # ------------------------------------------------------------------
    # Cache coordinates
    # ------------------------------------------------------------------

    def _model_key(self, model_id: ModelID) -> tuple[str, str]:
        """``(base_model_id, adapter_id)`` for cache addressing.

        A plain HuggingFace model has no adapter and lands under ``_base``.
        """
        base = self._resolve_base_model_id(model_id)
        if base is None:
            return str(model_id), "_base"
        return base, str(model_id)

    @property
    def _stored_modes(self) -> list[str]:
        return ["input", "generation"] if self.activation_mode == "both" else [self.activation_mode]

    def _mnt(self, mode: str) -> int | None:
        return None if mode == "input" else self.max_new_tokens

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        if self.cache is None:
            raise ValueError(
                "FunctionalTaxonomy requires an ActivationCache. Per-layer "
                "activations are the stored artefact and views are assembled "
                "from them; there is no in-memory-only path."
            )
        if not self.query_key:
            raise ValueError(
                "FunctionalTaxonomy requires query_key — the "
                "{recipe_hash, n_samples, seed} triple identifying the draw. It "
                "is what keys the cache."
            )

        base_id, adapter_id = self._model_key(model_id)

        if not self._all_on_disk(base_id, adapter_id):
            self._extract_fresh(model_id, base_id, adapter_id)

        return self.cache.load(
            base_id,
            adapter_id,
            self.query_key,
            mode=self.activation_mode,
            pooling=self.pooling,
            layers=None if self.layer_indices is None else self._stored_selection(base_id, adapter_id),
            view=self.view,
            normalize=self.normalize_activations,
            max_new_tokens=self._mnt(self.activation_mode),
        )

    def _stored_n_hidden_states(self, base_id: str, adapter_id: str) -> int | None:
        """Hidden-state count recorded by an earlier run on this draw, if any.

        Lets a negative ``layer_indices`` be resolved — and therefore a cache hit
        be detected — without loading the model.
        """
        for run_hash in self.cache.list_runs(base_id, adapter_id, self.query_key):
            rec = self.cache.load_config(base_id, adapter_id, self.query_key, run_hash)
            if rec.get("n_hidden_states"):
                return int(rec["n_hidden_states"])
        return None

    def _resolve_layers(self, n_hidden_states: int) -> list[int]:
        """Configured indices as absolute positions into ``hidden_states``.

        ``-1`` and ``28`` name the same layer; storing files under the configured
        index would write it twice under two names and let the copies drift.
        """
        if self.layer_indices is None:
            return list(range(n_hidden_states))
        out = []
        for ell in self.layer_indices:
            resolved = ell if ell >= 0 else n_hidden_states + ell
            if not 0 <= resolved < n_hidden_states:
                raise IndexError(
                    f"layer index {ell} resolves to {resolved}, outside the "
                    f"{n_hidden_states} hidden states this model produces."
                )
            out.append(resolved)
        return sorted(set(out))

    def _stored_selection(self, base_id: str, adapter_id: str) -> list[int] | None:
        n = self._stored_n_hidden_states(base_id, adapter_id)
        return self._resolve_layers(n) if n is not None else None

    def _all_on_disk(self, base_id: str, adapter_id: str) -> bool:
        """True when every requested (mode, layer) is already stored.

        A hit here means the model is never loaded — which is the whole reason
        activations are keyed model-wise rather than run-wise.
        """
        n = self._stored_n_hidden_states(base_id, adapter_id)
        for mode in self._stored_modes:
            stored = self.cache.list_layers(
                base_id, adapter_id, self.query_key, mode, self.pooling, self._mnt(mode)
            )
            if not stored:
                return False
            if self.layer_indices is None:
                # "All layers" is only satisfied by a run that stored all of
                # them; without a recorded count we cannot tell, so re-extract.
                if n is None or sorted(stored) != list(range(n)):
                    return False
            else:
                if n is None or not set(self._resolve_layers(n)).issubset(stored):
                    return False
        return True

    def _extract_fresh(self, model_id: ModelID, base_id: str, adapter_id: str) -> None:
        model, shared = self._get_model(model_id)
        tokenizer = self._load_tokenizer(model_id, self._resolve_base_model_id(model_id))

        try:
            n_hidden = int(model.config.num_hidden_layers) + 1
            layers = self._resolve_layers(n_hidden)

            # {mode: {layer: [ (batch, d) arrays ]}}
            acc: dict[str, dict[int, list[np.ndarray]]] = {
                mode: {ell: [] for ell in layers} for mode in self._stored_modes
            }

            for i in range(0, len(self.queries), self.batch_size):
                batch = self.queries[i : i + self.batch_size]
                for mode, per_layer in self._process_batch(model, tokenizer, batch, layers).items():
                    for ell, arr in per_layer.items():
                        acc[mode][ell].append(arr)
        finally:
            if not shared:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        config = self.config_dict()
        run_metadata = {
            "n_hidden_states": n_hidden,
            # Provenance, deliberately outside config_dict() so it does not
            # fragment the cache.  batch_size in particular must not key the
            # cache — with mask-aware pooling it does not change the numbers,
            # and recording it is how a regression there would be traceable.
            "batch_size": self.batch_size,
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        }

        for mode in self._stored_modes:
            stacked = {
                ell: np.concatenate(chunks, axis=0) for ell, chunks in acc[mode].items()
            }
            self.cache.save_activations(
                base_id,
                adapter_id,
                self.query_key,
                mode,
                self.pooling,
                stacked,
                max_new_tokens=self._mnt(mode),
                config=config,
                run_metadata=run_metadata,
                source_indices=self.source_indices,
            )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _process_batch(
        self,
        model: Any,
        tokenizer: Any,
        queries: list[str],
        layers: list[int],
    ) -> dict[str, dict[int, np.ndarray]]:
        """``{mode: {layer: (batch, d) array}}`` for one batch of queries."""
        inputs = tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        mask = inputs["attention_mask"]

        out: dict[str, dict[int, np.ndarray]] = {}
        with torch.no_grad():
            if "input" in self._stored_modes:
                out["input"] = self._input_activations(model, inputs, mask, layers)
            if "generation" in self._stored_modes:
                out["generation"] = self._generation_activations(
                    model, tokenizer, inputs, layers
                )
        return out

    def _input_activations(
        self, model: Any, inputs: dict, mask: torch.Tensor, layers: list[int]
    ) -> dict[int, np.ndarray]:
        hs = model(**inputs, output_hidden_states=True).hidden_states
        return {ell: self._pool(hs[ell], mask).float().cpu().numpy() for ell in layers}

    def _generation_activations(
        self, model: Any, tokenizer: Any, inputs: dict, layers: list[int]
    ) -> dict[int, np.ndarray]:
        """Generation-phase activations, mean-pooled across decoding steps.

        At each step the last-token hidden state is taken per layer and
        accumulated; the per-step tensors are then averaged.  Every step's last
        position is a real generated token, so no masking applies here — but note
        greedy decoding runs to ``max_new_tokens`` for every query regardless of
        EOS, so a query that finished early contributes post-EOS states.
        """
        gen = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        sums: dict[int, np.ndarray] = {}
        n_steps = 0
        for step_hs in gen.hidden_states:
            n_steps += 1
            for ell in layers:
                h = step_hs[ell][:, -1, :].float().cpu().numpy()
                sums[ell] = h if ell not in sums else sums[ell] + h
        return {ell: v / max(n_steps, 1) for ell, v in sums.items()}

    def _pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool ``(batch, seq, d)`` to ``(batch, d)``, ignoring padding.

        Masking is what makes a pooled vector a function of its query alone.
        ``padding=True`` pads each batch to its own longest sequence, so an
        unmasked mean would average in pad-position hidden states — which are not
        zero, since the model computes a residual-stream vector at every position
        even though nothing attends to them — and how many of those a query gets
        would depend on which other queries shared its batch.  Reordering the
        queries or changing ``batch_size`` would then shift every vector, and the
        cache could not notice because neither is part of the key.
        """
        m = mask.unsqueeze(-1).to(h.dtype)
        if self.pooling == "mean":
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        if self.pooling == "last_token":
            # Left padding puts the last real token at -1, but index it from the
            # mask anyway so this stays correct if padding_side ever changes.
            idx = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
            return h[torch.arange(h.shape[0], device=h.device), idx]
        if self.pooling == "cls":
            idx = mask.argmax(dim=1)  # first unmasked position
            return h[torch.arange(h.shape[0], device=h.device), idx]
        raise ValueError(f"Unknown pooling: {self.pooling!r}")
