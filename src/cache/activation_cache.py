from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.core.representation import ModelRepresentation


def _slug(model_id: str) -> str:
    """Filesystem-safe slug for a model or adapter ID.

    Deliberately the same scheme as :func:`src.cache.lora_cache._slug` — a
    functional entry sits at the same ``{base}/{adapter}`` coordinates as the
    structural entry for the same adapter, so the two trees can be read together.

    Note this differs from :func:`src.cache.generated_text_cache.model_slug`,
    which hashes the *full path*.  That makes a behavioral entry keyed to where
    the adapter directory happened to live on disk; this one is not.  The
    divergence is recorded in ``docs/notes/TODO.md``.
    """
    return str(model_id).replace("/", "--")


def adapter_slug(model_id: str) -> str:
    """Slug for the adapter component of a model key.

    Local adapter directories are absolute paths, so slugging the whole path
    would bury the name under the tree it lives in.  Take the directory name.
    """
    return _slug(Path(str(model_id)).name)


#: Filenames look like ``input_mean_layer028.safetensors``.  Everything that
#: changes the stored numbers is in the name rather than in a side file, so a
#: directory listing is a complete description of what was computed.
_ACT_RE = re.compile(r"^(?P<mode>[a-z]+\d*)_(?P<pooling>[a-z_]+)_layer(?P<layer>\d+)$")


class ActivationCache:
    """Cache for functional representations: pooled per-layer activations.

    Directory layout::

        cache_root/04_activations/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
            queries.json                              ← query_key + source row indices
            runs/{config_hash}.json                   ← extraction provenance
            activations/{mode}_{pooling}_layer{NNN}.safetensors
            surrogates/{surrogate_hash}/
                config.json
                surrogate.safetensors

    Keyed **model-wise then draw-wise**, following :class:`LoRACache`, rather
    than run-wise like :class:`GeneratedTextCache`.  The reason is that one
    forward pass produces *every* layer at once, so which layers you look at
    should be a read-time choice: a run stores all of them and re-observing a
    different subset costs nothing.  Under a run-wise ``{config_hash}/`` layout,
    changing ``layer_indices`` changes the hash and re-runs inference to
    recompute tensors already on disk.

    One file per ``(mode, pooling, layer)`` makes writes purely additive — a
    later run that adds a mode never rewrites an existing file.

    ``mode`` is ``input`` or ``generation{max_new_tokens}``.  There is no stored
    ``both``: that is a read-time combination of the two, which is what lets a
    run that already did ``input`` add ``generation`` later without redoing it.

    Everything that changes the stored numbers — mode, pooling, and the token
    budget for generation — is in the filename.  Normalization is *not*: raw
    activations are stored and normalization is a property of a surrogate, so
    differently normalized views coexist without re-running inference.

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

    #: Accepted normalization modes.  Bools are accepted too and canonicalized
    #: by :meth:`_canon_normalize`, so ``True`` and ``"layer"`` cannot produce
    #: two surrogates for one request.
    NORM_MODES = frozenset({"layer", "global", "none"})

    #: Views that are kernel matrices rather than feature matrices.  Handing one
    #: to a metric that forms its own kernel computes a different quantity
    #: silently, so representations built from these are tagged (see ``load``).
    KERNEL_VIEWS = frozenset({"gram"})

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._base = self.root / "04_activations"

    # ------------------------------------------------------------------
    # Path helpers — everything else must call these rather than rebuild paths
    # ------------------------------------------------------------------

    @staticmethod
    def draw_name(query_key: dict) -> str:
        """``n{n}_s{seed}``, matching the draw filenames in ``01_datasets``."""
        return f"n{query_key['n_samples']}_s{int(query_key['seed']):02d}"

    def draw_dir(self, base_model_id: str, adapter_id: str, query_key: dict) -> Path:
        return (
            self._base
            / _slug(base_model_id)
            / adapter_slug(adapter_id)
            / query_key["recipe_hash"]
            / self.draw_name(query_key)
        )

    @staticmethod
    def mode_token(mode: str, max_new_tokens: int | None = None) -> str:
        """The stored-mode component of an activation filename.

        ``generation`` carries its token budget because generating 32 tokens and
        generating 128 produce different mean-pooled vectors; storing both under
        one name would let them overwrite each other.
        """
        if mode == "input":
            return "input"
        if mode == "generation":
            if not max_new_tokens:
                raise ValueError("generation mode requires max_new_tokens > 0")
            return f"generation{int(max_new_tokens)}"
        raise ValueError(
            f"{mode!r} is not a stored mode; only 'input' and 'generation' are "
            "written to disk. 'both' is a read-time combination of the two."
        )

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
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def config_hash(config: dict) -> str:
        """16-char SHA-256 prefix identifying a config dict."""
        payload = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @classmethod
    def canon_normalize(cls, normalize: str | bool) -> str:
        """One spelling per normalization request.

        ``normalize=True`` and ``normalize="layer"`` are the same thing asked
        two ways.  They must reach the surrogate spec identically or they hash
        differently and the same view is computed and stored twice.
        """
        if normalize is True:
            return "layer"
        if normalize is False:
            return "none"
        if normalize in cls.NORM_MODES:
            return str(normalize)
        raise ValueError(
            f"unknown normalize {normalize!r}; expected one of "
            f"{sorted(cls.NORM_MODES)}, or a bool (True→'layer', False→'none')"
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
            # queries.json — which source row is row i of every matrix here.
            # The draw in 01_datasets is canonical and already stores source
            # indices, and (recipe_hash, n_samples, seed) determines the text
            # completely, so there is no reason to duplicate the text.
            queries_path = draw / "queries.json"
            if not queries_path.exists():
                _atomic_write_json(
                    queries_path,
                    {
                        "schema_version": "1",
                        "query_key": query_key,
                        "source_indices": source_indices or [],
                    },
                )

            if config is not None:
                run_path = draw / "runs" / f"{self.config_hash(config)}.json"
                if not run_path.exists():
                    _atomic_write_json(
                        run_path,
                        {
                            "schema_version": "1",
                            "config": config,
                            "mode": mode,
                            "pooling": pooling,
                            "max_new_tokens": max_new_tokens,
                            "resolved_layers": sorted(layers),
                            **(run_metadata or {}),
                            "written_at": datetime.now(timezone.utc).isoformat(),
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

    def load_config(
        self, base_model_id: str, adapter_id: str, query_key: dict, config_hash: str
    ) -> dict:
        path = self.draw_dir(base_model_id, adapter_id, query_key) / "runs" / f"{config_hash}.json"
        return json.loads(path.read_text())

    def load_queries(self, base_model_id: str, adapter_id: str, query_key: dict) -> dict:
        path = self.draw_dir(base_model_id, adapter_id, query_key) / "queries.json"
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Surrogates — computed on demand, written back
    # ------------------------------------------------------------------

    def surrogate_dir(
        self, base_model_id: str, adapter_id: str, query_key: dict, spec: dict
    ) -> Path:
        return (
            self.draw_dir(base_model_id, adapter_id, query_key)
            / "surrogates"
            / self.config_hash(spec)
        )

    def save_surrogate(
        self,
        base_model_id: str,
        adapter_id: str,
        query_key: dict,
        spec: dict,
        matrix: np.ndarray,
    ) -> None:
        from filelock import FileLock

        d = self.surrogate_dir(base_model_id, adapter_id, query_key, spec)
        d.mkdir(parents=True, exist_ok=True)
        with FileLock(str(d / ".lock")):
            if (d / "surrogate.safetensors").exists():
                return
            _atomic_write_json(d / "config.json", {"schema_version": "1", **spec})
            _atomic_save_tensor(
                d / "surrogate.safetensors",
                np.ascontiguousarray(matrix.astype(np.float32)),
                key="surrogate",
            )

    def load_surrogate(
        self, base_model_id: str, adapter_id: str, query_key: dict, spec: dict
    ) -> np.ndarray | None:
        from safetensors.numpy import load_file

        path = (
            self.surrogate_dir(base_model_id, adapter_id, query_key, spec)
            / "surrogate.safetensors"
        )
        if not path.exists():
            return None
        return load_file(str(path))["surrogate"]

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

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_base_models(self) -> list[str]:
        if not self._base.exists():
            return []
        return [d.name.replace("--", "/") for d in sorted(self._base.iterdir()) if d.is_dir()]

    def list_models(self, base_model_id: str | None = None) -> list[tuple[str, str]]:
        """``[(base_slug, adapter_slug), ...]`` present in the cache."""
        if not self._base.exists():
            return []
        bases = (
            [self._base / _slug(base_model_id)]
            if base_model_id is not None
            else [d for d in sorted(self._base.iterdir()) if d.is_dir()]
        )
        out = []
        for bd in bases:
            if not bd.exists():
                continue
            out.extend((bd.name, ad.name) for ad in sorted(bd.iterdir()) if ad.is_dir())
        return out

    def list_draws(self, base_model_id: str, adapter_id: str) -> list[dict]:
        """``[{recipe_hash, n_samples, seed}, ...]`` stored for one model."""
        adapter_dir = self._base / _slug(base_model_id) / adapter_slug(adapter_id)
        if not adapter_dir.exists():
            return []
        draws = []
        for recipe_dir in sorted(adapter_dir.iterdir()):
            if not recipe_dir.is_dir():
                continue
            for draw_dir in sorted(recipe_dir.iterdir()):
                m = re.match(r"^n(\d+)_s(\d+)$", draw_dir.name)
                if draw_dir.is_dir() and m:
                    draws.append({
                        "recipe_hash": recipe_dir.name,
                        "n_samples": int(m.group(1)),
                        "seed": int(m.group(2)),
                    })
        return draws

    def list_runs(self, base_model_id: str, adapter_id: str, query_key: dict) -> list[str]:
        runs_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "runs"
        if not runs_dir.exists():
            return []
        return sorted(p.stem for p in runs_dir.glob("*.json"))

    def has_draw(self, base_model_id: str, adapter_id: str, query_key: dict) -> bool:
        """True when any activations are stored for this model under this exact draw.

        The exact availability test, as opposed to :meth:`has_any`.
        """
        act_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "activations"
        return act_dir.exists() and any(act_dir.glob("*.safetensors"))

    def has_any(self, base_model_id: str, adapter_id: str) -> bool:
        """True when any activations exist for this model, under any draw.

        The coarse availability test discovery falls back to when it has not been
        told which draw to look for.
        """
        adapter_dir = self._base / _slug(base_model_id) / adapter_slug(adapter_id)
        if not adapter_dir.exists():
            return False
        return any(adapter_dir.glob("*/*/activations/*.safetensors"))


def _row_normalize(m: np.ndarray) -> np.ndarray:
    """L2-normalize each row, leaving zero rows at zero rather than NaN.

    Applied at two scales — per layer block and to the whole concatenation — so
    it is a helper rather than an inline expression.
    """
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms < 1e-12, 1.0, norms)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _atomic_save_tensor(path: Path, arr: np.ndarray, key: str = "activations") -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file({key: arr}, str(tmp))
    os.replace(tmp, path)
