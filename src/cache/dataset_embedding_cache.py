from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.core.representation import ModelRepresentation

if TYPE_CHECKING:
    from src.datasets.recipe import DatasetRecipe
    from src.datasets.class_recipe import ClassAwareDatasetRecipe


class DatasetEmbeddingCache:
    """Hierarchical cache for dataset embedding representations.

    Directory layout::

        cache_root/dataset_embeddings/{recipe_hash}/
            recipe.json                ← human-readable recipe (plain-text)
            {embedder_hash}/
                config.json            ← embedder config + representation + n_samples
                embeddings.safetensors ← (N, d) or (1, gram_features) float32

    The recipe is always written as a standalone human-readable file so recipes
    can be inspected or reconstructed without loading the tensor data.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._base = self.root / "dataset_embeddings"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _recipe_dir(self, recipe_hash: str) -> Path:
        return self._base / recipe_hash

    def _entry_dir(self, recipe_hash: str, embedder_hash: str) -> Path:
        return self._base / recipe_hash / embedder_hash

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def embedder_hash(
        embedder_config: dict,
        representation: str,
        n_samples: int,
    ) -> str:
        """16-char SHA-256 prefix identifying a (embedder, mode, n_samples) triple."""
        payload = json.dumps(
            {
                "embedder_config": embedder_config,
                "representation": representation,
                "n_samples": n_samples,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Existence checks
    # ------------------------------------------------------------------

    def exists(self, recipe_hash: str, embedder_hash: str) -> bool:
        d = self._entry_dir(recipe_hash, embedder_hash)
        return (d / "config.json").exists() and (d / "embeddings.safetensors").exists()

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
    ) -> None:
        """Atomically write recipe.json, config.json, and embeddings.safetensors.

        Idempotent: returns immediately if the entry already exists.
        Thread-safe via FileLock (safe on shared network filesystems).
        """
        from filelock import FileLock
        from safetensors.numpy import save_file

        recipe_hash = recipe.recipe_hash()
        emb_hash = self.embedder_hash(embedder_config, representation, n_samples)

        entry_dir = self._entry_dir(recipe_hash, emb_hash)
        entry_dir.mkdir(parents=True, exist_ok=True)

        lock_path = self._recipe_dir(recipe_hash) / ".lock"
        with FileLock(str(lock_path)):
            if self.exists(recipe_hash, emb_hash):
                return

            # Write recipe.json (human-readable, shared across embedder configs)
            recipe_path = self._recipe_dir(recipe_hash) / "recipe.json"
            if not recipe_path.exists():
                tmp = recipe_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(recipe.to_dict(), indent=2))
                os.replace(tmp, recipe_path)

            # Write config.json
            config = {
                "schema_version": "1",
                "recipe_hash": recipe_hash,
                "embedder_config": embedder_config,
                "representation": representation,
                "n_samples": n_samples,
                "embedded_at": datetime.now(timezone.utc).isoformat(),
            }
            config_path = entry_dir / "config.json"
            tmp_cfg = entry_dir / "config.json.tmp"
            tmp_cfg.write_text(json.dumps(config, indent=2))
            os.replace(tmp_cfg, config_path)

            # Write embeddings.safetensors
            st_path = entry_dir / "embeddings.safetensors"
            tmp_st = entry_dir / "embeddings.safetensors.tmp"
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

    def load(self, recipe_hash: str, embedder_hash: str) -> ModelRepresentation:
        """Reconstruct a ModelRepresentation from cached embeddings."""
        from safetensors.numpy import load_file

        entry_dir = self._entry_dir(recipe_hash, embedder_hash)
        tensors = load_file(str(entry_dir / "embeddings.safetensors"))
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

    def load_config(self, recipe_hash: str, embedder_hash: str) -> dict:
        """Return the config.json dict for a specific embedding entry."""
        path = self._entry_dir(recipe_hash, embedder_hash) / "config.json"
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

    def list_embedder_configs(self, recipe_hash: str) -> list[dict]:
        """Return all stored embedder configs for a given recipe_hash."""
        recipe_dir = self._recipe_dir(recipe_hash)
        if not recipe_dir.exists():
            return []
        configs = []
        for d in sorted(recipe_dir.iterdir()):
            cfg_path = d / "config.json"
            if d.is_dir() and cfg_path.exists():
                configs.append(json.loads(cfg_path.read_text()))
        return configs
