from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.core.protocols import ModelID
from src.core.representation import ModelRepresentation


def _slug(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe slug."""
    return model_id.replace("/", "--")


class LoRACache:
    """Hierarchical cache for LoRA adapter representations.

    Directory layout::

        cache_root/loras/{base_model_slug}/{adapter_slug}/
            config.json                 ← fine-tuning details + dataset_recipe stub
            representation.safetensors  ← extracted LoRA representation matrix

    The base model slug and adapter slug are derived by replacing '/' with '--'
    in the HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B' →
    'meta-llama--Llama-3.1-8B'), matching HuggingFace's own naming convention.

    This cache stores extracted representations, not raw LoRA weight tensors.
    Raw weights remain in HuggingFace's own download cache.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._loras_dir = self.root / "loras"

    def _adapter_dir(self, base_model_id: str, adapter_id: str) -> Path:
        return self._loras_dir / _slug(base_model_id) / _slug(adapter_id)

    def exists(self, base_model_id: str, adapter_id: str) -> bool:
        d = self._adapter_dir(base_model_id, adapter_id)
        return (d / "config.json").exists() and (d / "representation.safetensors").exists()

    def save(
        self,
        base_model_id: str,
        adapter_id: str,
        rep: ModelRepresentation,
        training_config: dict,
        extraction_config: dict,
    ) -> None:
        """Write config.json and representation.safetensors atomically."""
        from filelock import FileLock
        from safetensors.numpy import save_file

        adapter_dir = self._adapter_dir(base_model_id, adapter_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        lock_path = adapter_dir / ".lock"
        with FileLock(str(lock_path)):
            if self.exists(base_model_id, adapter_id):
                return

            config = {
                "schema_version": "1",
                "base_model_id": base_model_id,
                "adapter_id": adapter_id,
                "adapter_type": "lora",
                "training_config": training_config,
                "dataset_recipe": {
                    "_note": "stub — populate with actual dataset details",
                    "dataset_ids": [],
                    "split": None,
                    "num_samples": None,
                },
                "extraction_config": extraction_config,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

            # Write config.json atomically
            config_tmp = adapter_dir / "config.json.tmp"
            config_tmp.write_text(json.dumps(config, indent=2))
            os.replace(config_tmp, adapter_dir / "config.json")

            # Write representation.safetensors atomically
            st_tmp = adapter_dir / "representation.safetensors.tmp"
            save_file(
                {"matrix": np.ascontiguousarray(rep.matrix.astype(np.float32))},
                str(st_tmp),
            )
            os.replace(st_tmp, adapter_dir / "representation.safetensors")

    def load(self, base_model_id: str, adapter_id: str) -> ModelRepresentation:
        """Read representation.safetensors and reconstruct a ModelRepresentation."""
        from safetensors.numpy import load_file

        adapter_dir = self._adapter_dir(base_model_id, adapter_id)
        config = json.loads((adapter_dir / "config.json").read_text())
        tensors = load_file(str(adapter_dir / "representation.safetensors"))
        matrix = tensors["matrix"]
        return ModelRepresentation(
            model_id=adapter_id,
            taxonomy="structural",
            matrix=matrix,
            metadata={
                "base_model_id": base_model_id,
                "extraction_config": config.get("extraction_config", {}),
            },
            cache_key="",
        )

    def load_config(self, base_model_id: str, adapter_id: str) -> dict:
        """Return the full config.json dict for an adapter."""
        adapter_dir = self._adapter_dir(base_model_id, adapter_id)
        return json.loads((adapter_dir / "config.json").read_text())

    def list_adapters(self, base_model_id: str) -> list[str]:
        """Return all adapter IDs known for a given base model."""
        base_dir = self._loras_dir / _slug(base_model_id)
        if not base_dir.exists():
            return []
        return [
            d.name.replace("--", "/")
            for d in sorted(base_dir.iterdir())
            if d.is_dir() and (d / "config.json").exists()
        ]

    def list_base_models(self) -> list[str]:
        """Return all base model IDs present in the cache."""
        if not self._loras_dir.exists():
            return []
        return [
            d.name.replace("--", "/")
            for d in sorted(self._loras_dir.iterdir())
            if d.is_dir()
        ]

    @staticmethod
    def detect_base_model(adapter_id: str, hf_token: str | None = None) -> str:
        """Read PEFT adapter_config.json from the Hub and return the base model ID."""
        cfg = LoRACache._read_peft_adapter_config(adapter_id, hf_token)
        return cfg["base_model_name_or_path"]

    @staticmethod
    def _read_peft_adapter_config(
        adapter_id: str, hf_token: str | None = None
    ) -> dict:
        """Download and parse adapter_config.json from a HuggingFace PEFT adapter."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(adapter_id, "adapter_config.json", token=hf_token)
        return json.loads(Path(path).read_text())
