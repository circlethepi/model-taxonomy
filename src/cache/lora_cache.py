from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.core.protocols import ModelID
from src.core.representation import ModelRepresentation


def _slug(model_id: str) -> str:
    """Convert a HuggingFace model ID to a filesystem-safe slug."""
    return model_id.replace("/", "--")


class LoRACache:
    """Hierarchical cache for LoRA adapter representations.

    Directory layout::

        cache_root/adapters/{base_model_slug}/{adapter_slug}/
            adapter_model.safetensors   ← raw PEFT weights (untouched)
            adapter_config.json         ← raw PEFT config (untouched)
            {config_hash}/
                config.json             ← extraction metadata + layer_lengths
                representation.safetensors  ← extracted matrix (N_layers, max_len)

    Multiple extraction configs (e.g. different projection subsets) are stored in
    separate ``{config_hash}`` subdirectories under the same adapter folder.
    ``config_hash`` is a 16-char SHA-256 hex digest of the extraction config dict.

    Raw PEFT adapters (``adapter_model.safetensors`` present, no config-hash subdirs)
    are visible via ``list_raw_adapters()`` / ``adapter_status()``.
    """

    def __init__(self, cache_root: Path | str) -> None:
        self.root = Path(cache_root)
        self._loras_dir = self.root / "adapters"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adapter_dir(self, base_model_id: str, adapter_id: str) -> Path:
        return self._loras_dir / _slug(base_model_id) / _slug(adapter_id)

    @staticmethod
    def _config_hash(extraction_config: dict) -> str:
        """16-char SHA-256 hex digest of a sorted extraction config dict."""
        return hashlib.sha256(
            repr(sorted(extraction_config.items())).encode()
        ).hexdigest()[:16]

    def _rep_dir(self, base_model_id: str, adapter_id: str, extraction_config: dict) -> Path:
        return self._adapter_dir(base_model_id, adapter_id) / self._config_hash(extraction_config)

    @staticmethod
    def _is_rep_dir(path: Path) -> bool:
        """True if ``path`` looks like a config-hash representation subdir."""
        return path.is_dir() and (path / "config.json").exists()

    # ------------------------------------------------------------------
    # Core cache operations
    # ------------------------------------------------------------------

    def exists(self, base_model_id: str, adapter_id: str, extraction_config: dict) -> bool:
        d = self._rep_dir(base_model_id, adapter_id, extraction_config)
        return (d / "config.json").exists() and (d / "representation.safetensors").exists()

    def save(
        self,
        base_model_id: str,
        adapter_id: str,
        rep: ModelRepresentation,
        training_config: dict,
        extraction_config: dict,
        layer_lengths: list[int] | None = None,
        dataset_recipe: dict | None = None,
    ) -> None:
        """Write config.json and representation.safetensors into a config-hash subdir.

        Pass ``dataset_recipe=recipe.to_dict()`` to store the full mixing
        recipe alongside the LoRA tensors.  When omitted a placeholder stub
        is written instead.
        """
        from filelock import FileLock
        from safetensors.numpy import save_file

        rep_dir = self._rep_dir(base_model_id, adapter_id, extraction_config)
        rep_dir.mkdir(parents=True, exist_ok=True)

        lock_path = rep_dir / ".lock"
        with FileLock(str(lock_path)):
            if self.exists(base_model_id, adapter_id, extraction_config):
                return

            stored_recipe = dataset_recipe if dataset_recipe is not None else {
                "_note": "stub — populate with actual dataset details",
                "dataset_ids": [],
                "split": None,
                "num_samples": None,
            }

            stored_extraction = dict(extraction_config)
            if layer_lengths is not None:
                stored_extraction["layer_lengths"] = layer_lengths

            config = {
                "schema_version": "2",
                "base_model_id": base_model_id,
                "adapter_id": adapter_id,
                "adapter_type": "lora",
                "training_config": training_config,
                "dataset_recipe": stored_recipe,
                "extraction_config": stored_extraction,
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            }

            config_tmp = rep_dir / "config.json.tmp"
            config_tmp.write_text(json.dumps(config, indent=2))
            os.replace(config_tmp, rep_dir / "config.json")

            st_tmp = rep_dir / "representation.safetensors.tmp"
            save_file(
                {"matrix": np.ascontiguousarray(rep.matrix.astype(np.float32))},
                str(st_tmp),
            )
            os.replace(st_tmp, rep_dir / "representation.safetensors")

    def load(self, base_model_id: str, adapter_id: str, extraction_config: dict) -> ModelRepresentation:
        """Read representation.safetensors and reconstruct a ModelRepresentation."""
        from safetensors.numpy import load_file

        rep_dir = self._rep_dir(base_model_id, adapter_id, extraction_config)
        config = json.loads((rep_dir / "config.json").read_text())
        tensors = load_file(str(rep_dir / "representation.safetensors"))
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

    def load_config(self, base_model_id: str, adapter_id: str, extraction_config: dict) -> dict:
        """Return the full config.json dict for a specific extraction config."""
        rep_dir = self._rep_dir(base_model_id, adapter_id, extraction_config)
        return json.loads((rep_dir / "config.json").read_text())

    # ------------------------------------------------------------------
    # Listing methods
    # ------------------------------------------------------------------

    def list_adapters(self, base_model_id: str) -> list[str]:
        """Return adapter IDs that have at least one extracted representation."""
        base_dir = self._loras_dir / _slug(base_model_id)
        if not base_dir.exists():
            return []
        return [
            d.name.replace("--", "/")
            for d in sorted(base_dir.iterdir())
            if d.is_dir() and any(self._is_rep_dir(sub) for sub in d.iterdir() if sub.is_dir())
        ]

    def list_raw_adapters(self, base_model_id: str) -> list[str]:
        """Return adapter IDs that have raw PEFT weights but no extracted representation."""
        base_dir = self._loras_dir / _slug(base_model_id)
        if not base_dir.exists():
            return []
        return [
            d.name.replace("--", "/")
            for d in sorted(base_dir.iterdir())
            if d.is_dir()
            and (d / "adapter_model.safetensors").exists()
            and not any(self._is_rep_dir(sub) for sub in d.iterdir() if sub.is_dir())
        ]

    def adapter_status(self, base_model_id: str) -> dict[str, list[str]]:
        """Return ``{"processed": [...], "raw": [...]}`` for a base model."""
        return {
            "processed": self.list_adapters(base_model_id),
            "raw": self.list_raw_adapters(base_model_id),
        }

    def adapter_count(self, base_model_id: str) -> dict[str, int]:
        """Return ``{"processed": N, "raw": M}`` counts for a base model."""
        status = self.adapter_status(base_model_id)
        return {"processed": len(status["processed"]), "raw": len(status["raw"])}

    def list_representations(
        self, base_model_id: str, adapter_id: str
    ) -> list[tuple[str, dict]]:
        """Return ``[(config_hash, extraction_config), ...]`` for one adapter.

        Each entry is one previously extracted representation with a different
        extraction configuration.
        """
        adapter_dir = self._adapter_dir(base_model_id, adapter_id)
        if not adapter_dir.exists():
            return []
        results = []
        for sub in sorted(adapter_dir.iterdir()):
            if not self._is_rep_dir(sub):
                continue
            cfg = json.loads((sub / "config.json").read_text())
            results.append((sub.name, cfg.get("extraction_config", {})))
        return results

    def list_base_models(self) -> list[str]:
        """Return all base model IDs present in the cache."""
        if not self._loras_dir.exists():
            return []
        return [
            d.name.replace("--", "/")
            for d in sorted(self._loras_dir.iterdir())
            if d.is_dir()
        ]

    # ------------------------------------------------------------------
    # Product vectorization
    # ------------------------------------------------------------------

    _PROJ_LONG_TO_SHORT = {"k_proj": "k", "q_proj": "q", "v_proj": "v", "o_proj": "o"}

    def vectorize_products(
        self,
        base_model_id: str,
        weights,
        layers: list[int] | None = None,
        projections: str | list[str] | None = None,
        adapter_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute and cache per-adapter LoRA product vectors (B @ A) across layers and projections.

        For each adapter in *weights*, products are computed for every (layer, projection)
        pair in sorted order and concatenated into a single 1-D float32 array.

        Results are stored under the same adapter directory layout used by :meth:`save`:
        ``adapters/{base_model_slug}/{adapter_slug}/{config_hash}/`` with a ``config.json``
        and ``vector.npy``.  Existing entries are returned without recomputation.

        Parameters
        ----------
        base_model_id : base model string (e.g. ``"meta-llama/Llama-3.2-3B"``)
        weights : LoRAWeightCollection
        layers : layer indices to include; default is all layers loaded in *weights*
        projections : projections to include, e.g. ``"o"`` or ``["k", "v"]``;
            accepts short (``"o"``) or long (``"o_proj"``) forms.
            Default is all projections loaded in *weights*.
        adapter_names : subset of adapter names to process; default is all adapters in *weights*

        Returns
        -------
        dict mapping adapter name → 1-D float32 array of length
        ``n_layers × n_projections × out_features × in_features``
        """
        _layers = sorted(weights.layers if layers is None else layers)

        if projections is None:
            _projs = sorted(weights.projections)
        else:
            raw = [projections] if isinstance(projections, str) else list(projections)
            _projs = sorted({self._PROJ_LONG_TO_SHORT.get(p.lower(), p.lower()) for p in raw})

        vectorization_config = {
            "type": "vectorized_product",
            "layers": _layers,
            "projections": _projs,
        }
        config_hash = self._config_hash(vectorization_config)

        names = adapter_names if adapter_names is not None else weights.keys()

        result: dict[str, np.ndarray] = {}
        for name in tqdm(names, desc="vectorize products"):
            vec_dir = self._adapter_dir(base_model_id, name) / config_hash
            config_path = vec_dir / "config.json"
            vector_path = vec_dir / "vector.npy"

            if config_path.exists() and vector_path.exists():
                result[name] = np.load(str(vector_path))
                continue

            vec = np.concatenate([
                weights[name].product(layer, proj).ravel()
                for layer in _layers
                for proj in _projs
            ]).astype(np.float32)

            vec_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(vectorization_config, indent=2))
            np.save(str(vector_path), vec)

            result[name] = vec

        return result

    # ------------------------------------------------------------------
    # Hub helpers
    # ------------------------------------------------------------------

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
