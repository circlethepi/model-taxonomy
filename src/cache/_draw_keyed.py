"""Shared addressing for the caches that store one representation per model per draw.

``04_activations`` (:class:`~src.cache.activation_cache.ActivationCache`) and
``05_generated`` (:class:`~src.cache.generated_text_cache.GeneratedTextCache`)
hold the same *kind* of thing: a per-model representation computed over a shared
query draw.  They used to spell that coordinate two different ways, and one of
the spellings could lose data — ``05_generated`` keyed on a hash of the adapter's
**path**, so the entries were reachable only from the working directory the
extraction job happened to run in.  Every write still succeeded; the cache simply
read as empty.  ``docs/notes/TODO.md`` item 13 records the whole story.

The fix is not just to correct the one slug but to make the two caches share the
addressing code, so they cannot drift apart again:

    {stage_dir}/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
        queries.json            ← which draw this is; no query text (see below)
        runs/{config_hash}.json ← provenance for one extraction run
        {artifact_dir}/…        ← the subclass's stored tensors
        surrogates/{hash}/      ← read-time views, computed once and written back

Subclasses supply ``_STAGE_DIR`` and ``_ARTIFACT_DIR`` and the artifact naming;
everything above the artifact is here.  The precedent is
:mod:`src.taxonomy._hf_inference`: the two taxonomies loaded models identically,
so the loading half was extracted.  These two are *addressed* identically, so the
addressing half is extracted too.

**No query text is stored at either level.**  ``(recipe_hash, n_samples, seed)``
determines the text completely, because ``text_field`` is part of the recipe and
therefore part of ``recipe_hash`` — see ``ClassDatasetEntry.to_dict`` and
``ClassAwareDatasetRecipe._canonical``.  ``01_datasets`` is canonical; these
caches store a pointer to it, not a copy of it.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.cache._draw import DRAW_RE
from src.cache._draw import draw_name as _draw_name


def _slug(model_id: str) -> str:
    """Filesystem-safe slug for a model or adapter ID.

    Deliberately the same scheme as :func:`src.cache.lora_cache._slug`, so a
    functional or behavioral entry sits at the same ``{base}/{adapter}``
    coordinates as the structural entry for the same adapter and the trees can be
    read together.

    Note what this does **not** do: it does not hash, and it does not look at
    anything but the string it is given.  An earlier ``generated_text_cache``
    helper hashed the *full path*, which made an entry reachable only from the
    directory the writer ran in.  Do not reintroduce that.
    """
    return str(model_id).replace("/", "--")


def adapter_slug(model_id: str) -> str:
    """Slug for the adapter component of a model key.

    Local adapter directories are absolute paths, so slugging the whole path
    would bury the name under the tree it lives in.  Take the directory name —
    and note this is exactly why the *base* model is a separate path component:
    adapter basenames collide across base models, and the nesting is what keeps
    them apart.
    """
    return _slug(Path(str(model_id)).name)


#: ``n{n}_s{seed}`` directories, parsed when enumerating what a model has.
#: Re-exported from :mod:`src.cache._draw`, which owns the spelling for every
#: stage; the alias is kept so existing references here keep reading.
_DRAW_RE = DRAW_RE


class DrawKeyedCache:
    """Base for caches keyed ``{base}/{adapter}/{recipe_hash}/n{n}_s{seed}``.

    Keyed **model-wise then draw-wise**, following
    :class:`~src.cache.lora_cache.LoRACache`.  For activations the reason is that
    one forward pass produces every layer at once, so which layers you look at
    should be a read-time choice; a run stores all of them and re-observing a
    different subset costs nothing.  For generations the reason is weaker but the
    consistency is worth more than the marginal saving: the same model under the
    same draw should be at the same coordinates whatever stage you are reading.
    """

    #: ``04_activations`` / ``05_generated``.  Set by the subclass.
    _STAGE_DIR: str = ""

    #: The per-draw subdirectory holding the stored tensors.
    _ARTIFACT_DIR: str = ""

    #: Glob matching a *complete* artifact inside ``_ARTIFACT_DIR``.  Deliberately
    #: not ``*``: an interrupted write leaves a ``.safetensors.tmp`` behind, and a
    #: bare ``*`` would report that draw as present when it holds nothing usable.
    _ARTIFACT_GLOB: str = "*.safetensors"

    #: Accepted normalization modes.  Bools are accepted too and canonicalized by
    #: :meth:`canon_normalize`, so ``True`` and ``"layer"`` cannot produce two
    #: surrogates for one request.
    NORM_MODES = frozenset({"layer", "global", "none"})

    #: Views that are kernel matrices rather than feature matrices.  Handing one
    #: to a metric that forms its own kernel computes a different quantity
    #: silently, so representations built from these are tagged.
    KERNEL_VIEWS = frozenset({"gram"})

    def __init__(self, cache_root: Path | str) -> None:
        if not self._STAGE_DIR:
            raise TypeError(
                f"{type(self).__name__} must set _STAGE_DIR; DrawKeyedCache is abstract"
            )
        self.root = Path(cache_root)
        self._base = self.root / self._STAGE_DIR

    # ------------------------------------------------------------------
    # Path helpers — everything else must call these rather than rebuild paths
    # ------------------------------------------------------------------

    @staticmethod
    def draw_name(query_key: dict) -> str:
        """``n{n}_s{seed}`` for a query key, via :func:`src.cache._draw.draw_name`.

        Takes the whole query key rather than two ints because every caller here
        already holds one.  The spelling itself is not decided in this class —
        :mod:`src.cache._draw` owns it, so ``01``, ``02`` and the inference
        stages cannot drift apart again.  They *had* drifted: this token was
        documented as "matching the draw filenames in ``01_datasets``" while
        ``01`` was in fact writing an unpadded seed.  Item 15 moved ``01`` onto
        this spelling and made the claim true.
        """
        return _draw_name(query_key["n_samples"], query_key["seed"])

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
        """The stored-mode component of an artifact filename.

        ``generation`` carries its token budget because generating 32 tokens and
        generating 128 produce different results; storing both under one name
        would let them overwrite each other.
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

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def config_hash(config: dict) -> str:
        """16-char SHA-256 prefix identifying a config dict.

        ``sort_keys`` makes this independent of dict ordering, so a config built
        in a different order is still the same run.
        """
        payload = json.dumps(config, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @classmethod
    def canon_normalize(cls, normalize: str | bool) -> str:
        """One spelling per normalization request.

        ``normalize=True`` and ``normalize="layer"`` are the same thing asked two
        ways.  They must reach the surrogate spec identically or they hash
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
    # Per-draw side files, written once
    # ------------------------------------------------------------------

    def _write_queries_once(
        self, draw: Path, query_key: dict, source_indices: list | None
    ) -> None:
        """Record which draw this directory holds, and which source row is row *i*.

        Deliberately no query text: ``01_datasets`` is canonical and
        ``(recipe_hash, n_samples, seed)`` determines the text completely, since
        ``text_field`` is inside the recipe and therefore inside ``recipe_hash``.
        ``source_indices`` is a denormalized convenience — the same list is in the
        draw file — and may be empty when the writer did not have it.
        """
        path = draw / "queries.json"
        if path.exists():
            return
        _atomic_write_json(
            path,
            {
                "schema_version": "2",
                "query_key": query_key,
                "source_indices": list(source_indices or []),
            },
        )

    def _write_run_record(self, draw: Path, config: dict, extra: dict) -> None:
        """One JSON per extraction run, named by its config hash.

        Runs are additive: a later run that adds a mode or a layer leaves the
        earlier record alone, so ``runs/`` is a log of everything that has touched
        this draw.  That is what makes two runs differing only in a field outside
        the filename (``torch_dtype``, for instance) *detectable* after the fact.
        """
        (draw / "runs").mkdir(parents=True, exist_ok=True)
        path = draw / "runs" / f"{self.config_hash(config)}.json"
        if path.exists():
            return
        _atomic_write_json(
            path,
            {
                "schema_version": "1",
                "config": config,
                **extra,
                "written_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Read — side files
    # ------------------------------------------------------------------

    def load_queries(self, base_model_id: str, adapter_id: str, query_key: dict) -> dict:
        path = self.draw_dir(base_model_id, adapter_id, query_key) / "queries.json"
        return json.loads(path.read_text())

    def load_config(
        self, base_model_id: str, adapter_id: str, query_key: dict, config_hash: str
    ) -> dict:
        path = (
            self.draw_dir(base_model_id, adapter_id, query_key) / "runs" / f"{config_hash}.json"
        )
        return json.loads(path.read_text())

    def list_runs(self, base_model_id: str, adapter_id: str, query_key: dict) -> list[str]:
        runs_dir = self.draw_dir(base_model_id, adapter_id, query_key) / "runs"
        if not runs_dir.exists():
            return []
        return sorted(p.stem for p in runs_dir.glob("*.json"))

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
                m = _DRAW_RE.match(draw_dir.name)
                if draw_dir.is_dir() and m:
                    draws.append({
                        "recipe_hash": recipe_dir.name,
                        "n_samples": int(m.group(1)),
                        "seed": int(m.group(2)),
                    })
        return draws

    def has_draw(self, base_model_id: str, adapter_id: str, query_key: dict) -> bool:
        """True when any artifact is stored for this model under this exact draw.

        The exact availability test, as opposed to :meth:`has_any`.
        """
        art = self.draw_dir(base_model_id, adapter_id, query_key) / self._ARTIFACT_DIR
        return art.exists() and any(art.glob(self._ARTIFACT_GLOB))

    def has_any(self, base_model_id: str, adapter_id: str) -> bool:
        """True when any artifact exists for this model, under any draw.

        The coarse availability test discovery falls back to when it has not been
        told which draw to look for.
        """
        adapter_dir = self._base / _slug(base_model_id) / adapter_slug(adapter_id)
        if not adapter_dir.exists():
            return False
        return any(adapter_dir.glob(f"*/*/{self._ARTIFACT_DIR}/{self._ARTIFACT_GLOB}"))


def _row_normalize(m: np.ndarray) -> np.ndarray:
    """L2-normalize each row, leaving zero rows at zero rather than NaN.

    Applied at two scales — per layer block and to the whole concatenation — so
    it is a helper rather than an inline expression.
    """
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms < 1e-12, 1.0, norms)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def _atomic_save_tensor(path: Path, arr: np.ndarray, key: str = "activations") -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".safetensors.tmp")
    save_file({key: arr}, str(tmp))
    os.replace(tmp, path)
