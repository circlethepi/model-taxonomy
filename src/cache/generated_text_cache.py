from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.core.representation import ModelRepresentation


def model_slug(model_id: str) -> str:
    """Filesystem-safe, collision-free name for a model.

    Model IDs at this level are usually *absolute paths* to adapter directories,
    so the ``/`` → ``--`` scheme used for HuggingFace repo IDs would produce
    unreadable names carrying the whole path.  Take the adapter directory name
    for legibility and a hash prefix of the full ID for uniqueness.

    This is public and shared on purpose: :meth:`GeneratedTextCache.save` uses it
    to decide where a file goes and ``src.analysis.discovery.scan_cache`` uses it
    to find that file again.  If the two ever computed the slug separately they
    could drift, and the failure mode is silent — every write succeeds while the
    cache reads as empty.  ``discovery._sampled_rows_exist`` is that mistake
    already present in this codebase; do not add a second one.
    """
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:8]
    return f"{Path(model_id).name}__{digest}"


class GeneratedTextCache:
    """Cache for behavioral representations: generated text and its embeddings.

    Directory layout::

        cache_root/05_generated/{config_hash}/
            config.json                         ← embedder + generation config + query key
            queries.json                        ← the resolved query set, in order
            generations/{model_slug}.json       ← raw generated text per model
            embeddings/{model_slug}.safetensors ← (n_queries, d) float32 + _meta_json

    Text and tensors are split because they are read by different readers.
    Auditing what a model actually generated — the first thing worth doing after
    a run — is then a plain JSON open, with no safetensors load, no numpy and no
    GPU.  :meth:`load` reassembles the two into the single
    :class:`ModelRepresentation` the rest of the pipeline expects, so the split
    is an on-disk detail rather than an API change.

    Replaces the flat ``DiskCache`` this level used to use, which hashed the
    whole config — including every query string — into one opaque filename.  That
    made "does a representation exist for this model?" unanswerable without
    already holding the exact config that produced it.  Here availability is a
    file-existence test again, parameterized by a config hash.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._base = self.root / "05_generated"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _config_dir(self, config_hash: str) -> Path:
        return self._base / config_hash

    def generations_path(self, config_hash: str, model_id: str) -> Path:
        return self._config_dir(config_hash) / "generations" / f"{model_slug(model_id)}.json"

    def embeddings_path(self, config_hash: str, model_id: str) -> Path:
        return self._config_dir(config_hash) / "embeddings" / f"{model_slug(model_id)}.safetensors"

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def config_hash(config: dict) -> str:
        """16-char SHA-256 prefix identifying a taxonomy config.

        ``sort_keys`` makes this independent of dict ordering, so a config built
        in a different order is still the same run.
        """
        payload = json.dumps(config, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def exists(self, config_hash: str, model_id: str) -> bool:
        return (
            self.embeddings_path(config_hash, model_id).exists()
            and self.generations_path(config_hash, model_id).exists()
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(
        self,
        config_hash: str,
        rep: ModelRepresentation,
        *,
        config: dict | None = None,
        queries: list[str] | None = None,
        query_key: dict | None = None,
    ) -> None:
        """Atomically write config, queries, generations and embeddings.

        Idempotent: returns immediately if this model's entry already exists.
        Note the consequence — there is no invalidation path.  Anything inside
        *config* that changes produces a new ``config_hash`` and so a new
        directory, which is the behaviour you want and is legible on disk.
        Anything *outside* it (``device``, ``batch_size``) silently reuses the
        existing entry.
        """
        from filelock import FileLock
        from safetensors.numpy import save_file

        config_dir = self._config_dir(config_hash)
        (config_dir / "generations").mkdir(parents=True, exist_ok=True)
        (config_dir / "embeddings").mkdir(parents=True, exist_ok=True)

        lock_path = config_dir / ".lock"
        with FileLock(str(lock_path)):
            if self.exists(config_hash, rep.model_id):
                return

            # config.json — the identity of the run, shared by every model in it.
            config_path = config_dir / "config.json"
            if config is not None and not config_path.exists():
                payload = {
                    "schema_version": "1",
                    "config_hash": config_hash,
                    "query_key": query_key,
                    **config,
                    "written_at": datetime.now(timezone.utc).isoformat(),
                }
                _atomic_write_json(config_path, payload)

            # queries.json — convenience copy so a generation can be read beside
            # the prompt that produced it.  01_datasets stays canonical; the key
            # in config.json is what proves which draw this came from.
            queries_path = config_dir / "queries.json"
            if queries is not None and not queries_path.exists():
                _atomic_write_json(queries_path, {"query_key": query_key, "queries": queries})

            # generations/{slug}.json
            metadata = dict(rep.metadata or {})
            generated_texts = metadata.pop("generated_texts", None)
            _atomic_write_json(
                self.generations_path(config_hash, rep.model_id),
                {"model_id": rep.model_id, "generated_texts": generated_texts or []},
            )

            # embeddings/{slug}.safetensors
            st_path = self.embeddings_path(config_hash, rep.model_id)
            tmp_st = st_path.with_suffix(".safetensors.tmp")
            meta_bytes = np.frombuffer(
                json.dumps(
                    {
                        "model_id": rep.model_id,
                        "taxonomy": rep.taxonomy,
                        "metadata": metadata,
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

    def load(self, config_hash: str, model_id: str) -> ModelRepresentation:
        """Reconstruct a ModelRepresentation, generations folded back in.

        ``metadata["generated_texts"]`` is restored from ``generations/`` so the
        returned object is indistinguishable from one that was never cached.
        """
        from safetensors.numpy import load_file

        tensors = load_file(str(self.embeddings_path(config_hash, model_id)))
        matrix = tensors["matrix"]
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))

        metadata = dict(meta.get("metadata", {}))
        metadata["generated_texts"] = self.load_generations(config_hash, model_id)

        return ModelRepresentation(
            model_id=meta["model_id"],
            taxonomy=meta["taxonomy"],
            matrix=matrix,
            metadata=metadata,
            cache_key=f"{config_hash}/{model_slug(model_id)}",
        )

    def load_generations(self, config_hash: str, model_id: str) -> list[str]:
        """The generated text for one model, without touching the tensors."""
        path = self.generations_path(config_hash, model_id)
        if not path.exists():
            return []
        return json.loads(path.read_text()).get("generated_texts", [])

    def load_config(self, config_hash: str) -> dict:
        return json.loads((self._config_dir(config_hash) / "config.json").read_text())

    def load_queries(self, config_hash: str) -> dict:
        return json.loads((self._config_dir(config_hash) / "queries.json").read_text())

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_configs(self) -> list[str]:
        """Every config_hash present in the cache."""
        if not self._base.exists():
            return []
        return [
            d.name
            for d in sorted(self._base.iterdir())
            if d.is_dir() and (d / "config.json").exists()
        ]

    def list_models(self, config_hash: str) -> list[str]:
        """Model slugs with embeddings stored under *config_hash*.

        Slugs, not model IDs — the full ID is inside each file's ``_meta_json``,
        and the point of this method is to answer "what is here?" with a
        directory listing rather than N safetensors loads.
        """
        emb_dir = self._config_dir(config_hash) / "embeddings"
        if not emb_dir.exists():
            return []
        return sorted(p.stem for p in emb_dir.glob("*.safetensors"))


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)
