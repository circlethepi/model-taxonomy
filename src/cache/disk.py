from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from src.core.protocols import ModelID
from src.core.representation import ModelRepresentation


class DiskCache:
    """File-backed cache for ModelRepresentation objects.

    Atomic writes (via os.replace) and per-key file locks prevent corruption
    when multiple SLURM jobs write to a shared network filesystem simultaneously.

    Formats:
        "safetensors" (default): memory-mappable, pickle-free, fast load.
        "npz": NumPy zip archive (backward compat).
        "pt": PyTorch pickle format (backward compat, preserves bfloat16).
    """

    def __init__(
        self,
        cache_dir: Path | str,
        format: Literal["npz", "pt", "safetensors"] = "safetensors",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.format = format
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.{self.format}"

    def _lock_path(self, key: str) -> Path:
        subdir = self.cache_dir / key[:2]
        return subdir / f"{key}.lock"

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def load(self, key: str) -> ModelRepresentation:
        path = self._path(key)
        if self.format == "npz":
            data = np.load(path, allow_pickle=True)
            matrix = data["matrix"]
            meta = json.loads(str(data["meta"]))
        elif self.format == "pt":
            import torch

            data = torch.load(path, map_location="cpu", weights_only=False)
            matrix = data["matrix"]
            meta = data["meta"]
        else:  # safetensors
            from safetensors.numpy import load_file

            tensors = load_file(str(path))
            matrix = tensors["matrix"]
            meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))

        return ModelRepresentation(
            model_id=meta["model_id"],
            taxonomy=meta["taxonomy"],
            matrix=matrix,
            metadata=meta.get("metadata", {}),
            cache_key=key,
        )

    def save(self, key: str, rep: ModelRepresentation) -> None:
        from filelock import FileLock

        lock_path = self._lock_path(key)
        with FileLock(str(lock_path)):
            if self.exists(key):
                return
            path = self._path(key)
            tmp_path = path.with_suffix(".tmp")
            meta = {
                "model_id": rep.model_id,
                "taxonomy": rep.taxonomy,
                "metadata": rep.metadata,
            }
            if self.format == "npz":
                # np.savez always appends .npz, so pass the stem without extension
                # to avoid collisions (subdir/key.tmp → subdir/key.tmp.npz).
                tmp_stem = path.parent / f"{path.stem}.tmp"
                np.savez(tmp_stem, matrix=rep.matrix, meta=json.dumps(meta))
                os.replace(str(tmp_stem) + ".npz", path)
            elif self.format == "pt":
                import torch

                torch.save({"matrix": rep.matrix, "meta": meta}, tmp_path)
                os.replace(tmp_path, path)
            else:  # safetensors
                from safetensors.numpy import save_file

                meta_bytes = np.frombuffer(
                    json.dumps(meta).encode("utf-8"), dtype=np.uint8
                )
                save_file(
                    {
                        "matrix": np.ascontiguousarray(rep.matrix.astype(np.float32)),
                        "_meta_json": meta_bytes,
                    },
                    str(tmp_path),
                )
                os.replace(tmp_path, path)

    @staticmethod
    def key_for(model_id: ModelID, config: dict) -> str:
        """Derive a cache key from a model ID and a config dict."""
        config_hash = hashlib.sha256(
            repr(sorted(config.items())).encode()
        ).hexdigest()[:16]
        payload = f"{model_id}::{config_hash}".encode()
        return hashlib.sha256(payload).hexdigest()[:16]
