from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.cache._draw import draw_name, parse_draw_name
from src.cache._draw_keyed import DrawKeyedCache
from src.core.representation import ModelRepresentation

if TYPE_CHECKING:
    from src.datasets.recipe import DatasetRecipe
    from src.datasets.class_recipe import ClassAwareDatasetRecipe


class DatasetEmbeddingCache:
    """Hierarchical cache for dataset embedding representations.

    Directory layout::

        cache_root/02_dataset_embeddings/{recipe_hash}/
            recipe.json                     ← human-readable recipe (plain-text)
            n{n}_s{seed}/
                {embedder_hash}/
                    config.json             ← the embedder config
                    surrogates/{surrogate_hash}/
                        config.json         ← {"representation": ...}
                        surrogate.safetensors

    The recipe is always written as a standalone human-readable file so recipes
    can be inspected or reconstructed without loading the tensor data.  It sits
    at the **recipe** level because it describes the mixture — which is exactly
    what ``recipe_hash`` identifies — and is shared across every draw of it.

    **The draw is a path component, not part of a hash.**  It briefly was part of
    one: when ``recipe_hash`` became content-addressed over
    ``{recipe_type, datasets}``, ``_s{seed}`` left the recipe *name*, and seed had
    to go somewhere or every seed of a mixture would collapse onto one entry and
    a seed sweep would silently read one draw for all seeds.  Folding it into
    ``embedder_hash`` fixed that, but it left this the only stage where "which
    draws are embedded?" could not be answered by looking — 520 opaque directory
    names encoding what ``01``, ``04`` and ``05`` all write in plain text.  The
    guarantee is unchanged; only its location is.

    **A surrogate here is authored, not derived — and that differs from
    ``04``/``05``.**  In the inference caches a surrogate is a read-time *view* of
    a stored base artifact: the raw activations are on disk, so a new view is a
    recomputation that costs nothing but CPU.  This stage has no base artifact.
    ``representation`` is chosen *before* embedding and only its result is
    stored, because the true base — the full ``(N, 768)`` per-element embeddings —
    would cost 6.1 GB across the stored cache and a GPU re-embed of ~2M texts,
    and ``mean`` is not invertible.  So the directory shape matches ``04``/``05``
    exactly while the guarantee behind it is weaker: **adding a representation
    here means re-embedding, not a read-time rebuild.**  Do not write code that
    assumes a missing surrogate can be reconstructed from a sibling.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._base = self.root / "02_dataset_embeddings"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _recipe_dir(self, recipe_hash: str) -> Path:
        return self._base / recipe_hash

    def draw_dir(self, recipe_hash: str, n_samples: int, seed: int) -> Path:
        return self._recipe_dir(recipe_hash) / draw_name(n_samples, seed)

    def entry_dir(
        self, recipe_hash: str, n_samples: int, seed: int, embedder_hash: str
    ) -> Path:
        return self.draw_dir(recipe_hash, n_samples, seed) / embedder_hash

    def surrogate_dir(
        self,
        recipe_hash: str,
        n_samples: int,
        seed: int,
        embedder_hash: str,
        spec: dict,
    ) -> Path:
        return (
            self.entry_dir(recipe_hash, n_samples, seed, embedder_hash)
            / "surrogates"
            / self.surrogate_hash(spec)
        )

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def embedder_hash(embedder_config: dict) -> str:
        """16-char SHA-256 prefix identifying an embedder configuration.

        **Only the embedder**, which is what the name has always claimed.  It
        used to carry ``representation``, ``n_samples`` and ``seed`` as well —
        and since the embedder axis never actually varied in the stored cache,
        that made a directory named ``embedder_hash`` in practice a draw hash.
        The draw is now a path component and the representation is the surrogate
        spec, so this key finally means what it says.

        Every field of the embedder config counts, including ``prompt_prefix``:
        bare and correctly-prefixed embeddings live on different scales, so a key
        that could not tell them apart would let the cache hand back one where
        the other was asked for.
        """
        return DrawKeyedCache.config_hash(embedder_config)

    @staticmethod
    def surrogate_hash(spec: dict) -> str:
        """16-char SHA-256 prefix identifying a representation spec.

        Deliberately :meth:`DrawKeyedCache.config_hash`, not a private peer, so a
        spec dict hashes identically here and at ``04``/``05``.  Two hashing
        schemes for one concept is how the draw token drifted; do not start a
        second one.
        """
        return DrawKeyedCache.config_hash(spec)

    @staticmethod
    def spec_for(representation: str) -> dict:
        """The surrogate spec naming one representation mode."""
        return {"representation": representation}

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def exists(
        self,
        recipe_hash: str,
        n_samples: int,
        seed: int,
        embedder_hash: str,
        spec: dict,
    ) -> bool:
        d = self.surrogate_dir(recipe_hash, n_samples, seed, embedder_hash, spec)
        return (d / "config.json").exists() and (d / "surrogate.safetensors").exists()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        recipe: DatasetRecipe | ClassAwareDatasetRecipe,
        rep: ModelRepresentation,
        embedder_config: dict,
        representation: str,
        n_samples: int,
        seed: int,
    ) -> None:
        """Atomically write recipe.json, the entry config, and the surrogate.

        Idempotent: returns immediately if the surrogate already exists.
        Thread-safe via FileLock (safe on shared network filesystems).

        ``seed`` is required.  It was optional while it lived inside a hash,
        where ``None`` was merely one more value to digest; now that it names a
        directory, ``None`` would render as the literal ``sNone`` — a path that
        describes no draw and silently pools everything seedless into one entry.
        """
        from filelock import FileLock
        from safetensors.numpy import save_file

        recipe_hash = recipe.recipe_hash()
        emb_hash = self.embedder_hash(embedder_config)
        spec = self.spec_for(representation)

        entry_dir = self.entry_dir(recipe_hash, n_samples, seed, emb_hash)
        surr_dir = self.surrogate_dir(recipe_hash, n_samples, seed, emb_hash, spec)
        surr_dir.mkdir(parents=True, exist_ok=True)

        lock_path = self._recipe_dir(recipe_hash) / ".lock"
        with FileLock(str(lock_path)):
            if self.exists(recipe_hash, n_samples, seed, emb_hash, spec):
                return

            # recipe.json — human-readable, shared across draws and embedders
            recipe_path = self._recipe_dir(recipe_hash) / "recipe.json"
            if not recipe_path.exists():
                tmp = recipe_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(recipe.to_dict(), indent=2))
                os.replace(tmp, recipe_path)

            # The entry config describes the embedder and the draw it ran over.
            # n_samples and seed are recorded even though the path already says
            # them: a file that cannot be interpreted without its own path is a
            # file that cannot survive being moved, and this stage has now been
            # moved twice.
            config = {
                "schema_version": "3",
                "recipe_hash": recipe_hash,
                "embedder_config": embedder_config,
                "n_samples": n_samples,
                "seed": seed,
                "embedded_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_write_json(entry_dir / "config.json", config)

            _atomic_write_json(
                surr_dir / "config.json", {"schema_version": "1", **spec}
            )

            st_path = surr_dir / "surrogate.safetensors"
            tmp_st = surr_dir / "surrogate.safetensors.tmp"
            meta_bytes = np.frombuffer(
                json.dumps(
                    {
                        "model_id": rep.model_id,
                        "taxonomy": rep.taxonomy,
                        "metadata": rep.metadata,
                    }
                ).encode("utf-8"),
                dtype=np.uint8,
            )
            save_file(
                {
                    "matrix": np.ascontiguousarray(rep.matrix.astype(np.float32)),
                    "_meta_json": meta_bytes,
                },
                str(tmp_st),
            )
            os.replace(tmp_st, st_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(
        self,
        recipe_hash: str,
        n_samples: int,
        seed: int,
        embedder_hash: str,
        spec: dict,
    ) -> ModelRepresentation:
        """Reconstruct a ModelRepresentation from a cached surrogate."""
        from safetensors.numpy import load_file

        d = self.surrogate_dir(recipe_hash, n_samples, seed, embedder_hash, spec)
        tensors = load_file(str(d / "surrogate.safetensors"))
        matrix = tensors["matrix"]
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return ModelRepresentation(
            model_id=meta["model_id"],
            taxonomy=meta["taxonomy"],
            matrix=matrix,
            metadata=meta.get("metadata", {}),
            cache_key="",
        )

    def load_recipe(self, recipe_hash: str) -> dict:
        """Return the raw recipe dict stored in recipe.json."""
        path = self._recipe_dir(recipe_hash) / "recipe.json"
        return json.loads(path.read_text())

    def load_config(
        self, recipe_hash: str, n_samples: int, seed: int, embedder_hash: str
    ) -> dict:
        """Return the config.json dict for one embedder entry."""
        path = self.entry_dir(recipe_hash, n_samples, seed, embedder_hash) / "config.json"
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_recipes(self) -> list[str]:
        """Return all recipe_hashes present in the cache."""
        if not self._base.exists():
            return []
        return [
            d.name
            for d in sorted(self._base.iterdir())
            if d.is_dir() and (d / "recipe.json").exists()
        ]

    def list_draws(self, recipe_hash: str) -> list[tuple[int, int]]:
        """The ``(n_samples, seed)`` draws embedded under a recipe hash.

        The point of the whole relayout: this used to require opening one JSON
        per entry, because the answer was inside a hash rather than in the tree.
        It is now a directory listing, which is how ``01`` and ``04`` have always
        answered the same question.
        """
        recipe_dir = self._recipe_dir(recipe_hash)
        if not recipe_dir.exists():
            return []
        draws = []
        for d in sorted(recipe_dir.iterdir()):
            parsed = parse_draw_name(d.name) if d.is_dir() else None
            if parsed:
                draws.append(parsed)
        return draws

    def list_embedder_hashes(
        self, recipe_hash: str, n_samples: int, seed: int
    ) -> list[str]:
        """The embedder hashes stored for one draw, without reading configs."""
        d = self.draw_dir(recipe_hash, n_samples, seed)
        if not d.exists():
            return []
        return [
            e.name for e in sorted(d.iterdir())
            if e.is_dir() and (e / "config.json").exists()
        ]

    def list_surrogates(
        self, recipe_hash: str, n_samples: int, seed: int, embedder_hash: str
    ) -> list[dict]:
        """The stored representation specs for one embedder entry."""
        d = self.entry_dir(recipe_hash, n_samples, seed, embedder_hash) / "surrogates"
        if not d.exists():
            return []
        specs = []
        for s in sorted(d.iterdir()):
            cfg = s / "config.json"
            if s.is_dir() and cfg.exists():
                specs.append(json.loads(cfg.read_text()))
        return specs

    def list_embedder_configs(
        self,
        recipe_hash: str,
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> list[dict]:
        """Stored embedder configs, for one draw or across all of them.

        Each returned config carries its own ``n_samples`` and ``seed``, so a
        caller sweeping every draw of a recipe can still tell them apart.
        """
        if n_samples is not None and seed is not None:
            draws = [(n_samples, seed)]
        else:
            draws = self.list_draws(recipe_hash)

        configs = []
        for n, s in draws:
            d = self.draw_dir(recipe_hash, n, s)
            if not d.exists():
                continue
            for e in sorted(d.iterdir()):
                cfg = e / "config.json"
                if e.is_dir() and cfg.exists():
                    configs.append(json.loads(cfg.read_text()))
        return configs


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
