from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from src.cache._draw_keyed import (
    DrawKeyedCache,
    _atomic_save_tensor,
    _atomic_write_json,
    _row_normalize,
    _slug,
    adapter_slug,
)
from src.core.representation import ModelRepresentation

# Re-exported so existing importers of ``activation_cache._slug`` /
# ``adapter_slug`` keep working; the definitions now live on the shared base so
# the two inference caches cannot drift apart.  See ``_draw_keyed`` for why that
# matters — the drift is exactly what orphaned the behavioral cache.
__all__ = [
    "ActivationCache",
    "_slug",
    "adapter_slug",
    "_row_normalize",
    "_atomic_write_json",
    "_atomic_save_tensor",
]


#: Filenames look like ``input_mean_layer028.safetensors``.  Everything that
#: changes the stored numbers is in the name rather than in a side file, so a
#: directory listing is a complete description of what was computed.
_ACT_RE = re.compile(r"^(?P<mode>[a-z]+\d*)_(?P<pooling>[a-z_]+)_layer(?P<layer>\d+)$")


class ActivationCache(DrawKeyedCache):
    """Cache for functional representations: pooled per-layer activations.

    Directory layout::

        cache_root/04_activations/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
            queries.json                              ← query_key + source row indices
            runs/{config_hash}.json                   ← extraction provenance
            activations/{mode}_{pooling}_layer{NNN}.safetensors
            surrogates/{surrogate_hash}/
                config.json
                surrogate.safetensors

    The addressing above the artifact is :class:`DrawKeyedCache`; see that module
    for why it is shared with ``05_generated``.

    One file per ``(mode, pooling, layer)`` makes writes purely additive — a
    later run that adds a mode never rewrites an existing file.

    ``mode`` is ``input`` or ``generation{max_new_tokens}``.  There is no stored
    ``both``: that is a read-time combination of the two, which is what lets a
    run that already did ``input`` add ``generation`` later without redoing it.

    Everything that changes the stored numbers — mode, pooling, and the token
    budget for generation — is in the filename.  Normalization is *not*: raw
    activations are stored and normalization is a property of a surrogate, so
    differently normalized views coexist without re-running inference.

    **One thing is deliberately not in the filename, and it is a hazard:**
    ``torch_dtype``.  An fp16 run and an fp32 run over the same draw write the
    same filenames, so the second is a no-op skip rather than a second entry —
    the numbers on disk are whichever ran first.  It is detectable after the
    fact, because each run leaves its own ``runs/{config_hash}.json`` and the
    dtype is in there; it is not prevented.  ``05_generated`` has the same
    property for the same reason, and the symmetry is intentional.

    Normalization has two modes, both applied to *rows* (a row is a query):

    ``layer`` (the default)
        Normalize each ``(mode, layer)`` block's rows, concatenate, then
        normalize the row again.  Every layer contributes equally to a dot
        product.
    ``global``
        Concatenate first, normalize the row once.  Layers then contribute in
        proportion to their own scale.

    ``layer`` is the default because under ``global`` a layer's weight in the
    comparison is an accident of its activation scale rather than a choice.
    **Measured** on Llama-3.2-3B, mean-pooled over 64 queries: the transformer
    blocks are within ~1.6x of each other (row norms 56 → 71, rising with depth),
    so mid-and-late layers are *not* badly skewed — but the **embedding layer and
    layer 1 are two orders of magnitude smaller** (0.36 and 2.41), giving them
    shares of 0.00% and 0.01% of a row's squared norm.  Under ``global`` those
    two layers are effectively absent from the comparison; under ``layer`` they
    are full members at 1/29 each.

    That cuts both ways, so it is worth knowing which you asked for: ``layer``
    includes information ``global`` silently drops, and equally it amplifies a
    near-zero-norm layer — and whatever noise it carries — up to parity.

    ``none`` stores the raw concatenation.  See ``docs/notes/gram_and_cka.md``.
    """

    _STAGE_DIR = "04_activations"
    _ARTIFACT_DIR = "activations"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def activation_path(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        pooling: str,
        layer: int,
        max_new_tokens: int | None = None,
    ) -> Path:
        if layer < 0:
            raise ValueError(
                f"layer index {layer} is negative; resolve it against the model's "
                "hidden-state count before addressing the cache, or -1 and 28 will "
                "be stored twice under two names and be free to drift apart."
            )
        name = f"{self.mode_token(mode, max_new_tokens)}_{pooling}_layer{layer:03d}.safetensors"
        return self.draw_dir(base_model_id, adapter_id, query_key) / "activations" / name

    # ------------------------------------------------------------------
    # Existence
    # ------------------------------------------------------------------

    def exists(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        pooling: str,
        layer: int,
        max_new_tokens: int | None = None,
    ) -> bool:
        return self.activation_path(
            base_model_id, adapter_id, query_key, mode, pooling, layer, max_new_tokens
        ).exists()

    def has_all(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        pooling: str,
        layers: list[int],
        max_new_tokens: int | None = None,
    ) -> bool:
        """True when every requested layer is already on disk.

        This is what lets extraction skip loading the model at all.
        """
        return bool(layers) and all(
            self.exists(
                base_model_id, adapter_id, query_key, mode, pooling, ell, max_new_tokens
            )
            for ell in layers
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_activations(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        pooling: str,
        layers: dict[int, np.ndarray],
        *,
        max_new_tokens: int | None = None,
        config: dict | None = None,
        run_metadata: dict | None = None,
        source_indices: list | None = None,
    ) -> None:
        """Write one ``(n_queries, d)`` array per resolved layer, atomically.

        Existing layer files are left alone — writes are additive, so re-running
        with an extra mode or extra layers never rewrites what is already there.
        """
        from filelock import FileLock

        draw = self.draw_dir(base_model_id, adapter_id, query_key)
        (draw / "activations").mkdir(parents=True, exist_ok=True)
        (draw / "runs").mkdir(parents=True, exist_ok=True)

        with FileLock(str(draw / ".lock")):
            self._write_queries_once(draw, query_key, source_indices)

            if config is not None:
                self._write_run_record(
                    draw,
                    config,
                    {
                        "mode": mode,
                        "pooling": pooling,
                        "max_new_tokens": max_new_tokens,
                        "resolved_layers": sorted(layers),
                        **(run_metadata or {}),
                    },
                )

            for layer, arr in layers.items():
                path = self.activation_path(
                    base_model_id, adapter_id, query_key, mode, pooling, layer, max_new_tokens
                )
                if path.exists():
                    continue
                _atomic_save_tensor(path, np.ascontiguousarray(arr.astype(np.float32)))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_layers(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str,
        pooling: str,
        max_new_tokens: int | None = None,
    ) -> list[int]:
        """Resolved layer indices stored for one ``(mode, pooling)``, ascending."""
        act_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "activations"
        if not act_dir.exists():
            return []
        want_mode = self.mode_token(mode, max_new_tokens)
        found = []
        for p in act_dir.glob("*.safetensors"):
            m = _ACT_RE.match(p.stem)
            if m and m["mode"] == want_mode and m["pooling"] == pooling:
                found.append(int(m["layer"]))
        return sorted(found)

    def load_layers(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str = "input",
        pooling: str = "mean",
        layers: list[int] | None = None,
        max_new_tokens: int | None = None,
    ) -> dict[int, np.ndarray]:
        """``{resolved_layer: (n_queries, d) array}``.

        ``layers=None`` reads everything stored for this ``(mode, pooling)``.
        """
        from safetensors.numpy import load_file

        if layers is None:
            layers = self.list_layers(
                base_model_id, adapter_id, query_key, mode, pooling, max_new_tokens
            )
        out: dict[int, np.ndarray] = {}
        for ell in sorted(layers):
            path = self.activation_path(
                base_model_id, adapter_id, query_key, mode, pooling, ell, max_new_tokens
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"no stored activations for layer {ell} of {adapter_id} "
                    f"({mode}/{pooling}) at {path}"
                )
            out[ell] = load_file(str(path))["activations"]
        return out

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def load(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        mode: str = "input",
        pooling: str = "mean",
        layers: list[int] | None = None,
        view: str = "concat",
        normalize: str | bool = "layer",
        max_new_tokens: int | None = None,
    ) -> ModelRepresentation:
        """A model's functional representation, defaulting to the concatenated view.

        With no arguments beyond the model key and draw this returns the
        concatenation across **all stored layers** of ``input``-mode mean-pooled
        activations — ``(n_queries, L·d)``, row *i* being query *i* —
        layer-normalized, so every layer weighs the same.

        Views are resolved through ``surrogates/`` first and computed only on a
        miss, then written back, so a given ``(draw, mode, pooling, layers, view,
        normalize)`` is computed at most once.
        """
        normalize = self.canon_normalize(normalize)

        if mode == "both":
            stored_modes = ["input", "generation"]
        else:
            stored_modes = [mode]

        resolved: dict[str, list[int]] = {}
        for m in stored_modes:
            avail = self.list_layers(
                base_model_id, adapter_id, query_key, m, pooling, max_new_tokens
            )
            resolved[m] = sorted(avail if layers is None else [ell for ell in layers if ell in avail])
            if not resolved[m]:
                raise FileNotFoundError(
                    f"no stored {m} activations for {adapter_id} under draw "
                    f"{self.draw_name(query_key)} (pooling={pooling})"
                )

        spec = {
            "kind": "functional_surrogate",
            "query_key": query_key,
            "mode": mode,
            "pooling": pooling,
            "layers": {m: resolved[m] for m in stored_modes},
            "view": view,
            "normalize": normalize,
            "max_new_tokens": max_new_tokens if mode != "input" else None,
        }

        matrix = self.load_surrogate(base_model_id, adapter_id, query_key, spec)
        cached = matrix is not None
        if not cached:
            matrix = self._build_view(
                base_model_id, adapter_id, query_key, resolved,
                pooling, view, normalize, max_new_tokens,
            )
            self.save_surrogate(base_model_id, adapter_id, query_key, spec, matrix)

        return ModelRepresentation(
            model_id=adapter_id,
            taxonomy="functional",
            matrix=matrix,
            metadata={
                "base_model_id": base_model_id,
                "query_key": query_key,
                "mode": mode,
                "pooling": pooling,
                "layers": {m: resolved[m] for m in stored_modes},
                "view": view,
                "normalize": normalize,
                "n_queries": int(matrix.shape[0]),
                # Tagged so a metric that forms its own kernel can refuse rather
                # than silently computing (H Hᵀ)² — see KERNEL_VIEWS.
                "is_kernel": view in self.KERNEL_VIEWS,
                "surrogate_cached": cached,
                # What identifies this read to CollectionCache: where the stored
                # artifact lives (relative to the cache root) and which view of it
                # was taken.  Surfaced here so the collection key is built from
                # what was actually resolved, rather than from the caller's
                # possibly-underspecified selector.
                "artifact_path": self.artifact_path(
                    base_model_id, adapter_id, query_key
                ),
                "surrogate_hash": self.config_hash(spec),
            },
            cache_key=f"{adapter_slug(adapter_id)}/{self.draw_name(query_key)}/{self.config_hash(spec)}",
        )

    def _build_view(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        resolved: dict[str, list[int]],
        pooling: str,
        view: str,
        normalize: str,
        max_new_tokens: int | None,
    ) -> np.ndarray:
        """Concatenate the requested layers, then apply the view.

        ``both`` concatenates the input and generation halves along the feature
        axis, so a row stays one query.

        Under ``normalize="layer"`` each block is row-normalized *before* the
        concatenation, which is what equalizes the layers; the row is then
        normalized again so the result is unit-norm either way, and ``gram``'s
        diagonal is 1 under both modes.  Blocks are per ``(mode, layer)``, so an
        input-mode and a generation-mode block of the same layer are scaled
        independently — they have no reason to share a scale.
        """
        blocks: list[np.ndarray] = []
        for m, lays in resolved.items():
            per_layer = self.load_layers(
                base_model_id, adapter_id, query_key, m, pooling, lays, max_new_tokens
            )
            blocks.extend(per_layer[ell].astype(np.float64) for ell in sorted(per_layer))

        if normalize == "layer":
            blocks = [_row_normalize(b) for b in blocks]

        H = np.concatenate(blocks, axis=1)

        if normalize in ("layer", "global"):
            H = _row_normalize(H)

        if view == "concat":
            return H.astype(np.float32)
        if view == "gram":
            # Gram of the *concatenated* matrix: (n_queries, n_queries), rows =
            # queries.  Not a stack of per-layer triangles — that older form made
            # a row a layer, which is a different object with a different meaning.
            return (H @ H.T).astype(np.float32)
        raise ValueError(f"unknown view {view!r}; expected 'concat' or 'gram'")
