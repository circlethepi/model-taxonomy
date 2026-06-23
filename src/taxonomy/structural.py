from __future__ import annotations

import gc
import os
from typing import Any, Sequence

import numpy as np

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation
from src.cache.disk import DiskCache
from src.cache.lora_cache import LoRACache


def _find_lora_pairs(
    model: Any,
    layer_names: list[str] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {module_name: (A_matrix, B_matrix)} for all detected LoRA adapters.

    Matches parameters whose names contain '.lora_A.' with their paired '.lora_B.'
    counterparts. If layer_names is provided, only modules whose name starts with
    one of the given prefixes are included.
    """
    params = dict(model.named_parameters())
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, param in params.items():
        if ".lora_A." not in name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        if b_name not in params:
            continue
        module = name.split(".lora_A.")[0]
        if layer_names is not None and not any(module.startswith(ln) for ln in layer_names):
            continue
        pairs[module] = (
            param.detach().float().cpu().numpy(),
            params[b_name].detach().float().cpu().numpy(),
        )
    return pairs


def _truncate_pad(v: np.ndarray, n: int) -> np.ndarray:
    if len(v) >= n:
        return v[:n]
    return np.pad(v, (0, n - len(v)))


class StructuralTaxonomy(Taxonomy):
    """Compares models via the geometry of their weight matrices.

    By default (`lora_only=True`) only LoRA adapter matrices are used, making this
    practical for comparing fine-tuned variants without storing full weight matrices.
    For each LoRA module the adapter matrices A (rank × in) and B (out × rank) are
    concatenated and truncated/padded to `n_components` values.

    For base models or when `lora_only=False`, explicit `layer_names` can be given
    to select which named parameters to compare; if omitted, all 2-D weight matrices
    larger than 1024 elements are included automatically.

    The representation matrix has shape (N_layers, n_components), where each row
    corresponds to one weight layer/adapter.

    Cache priority:
        1. ``lora_cache`` (LoRACache) — checked first when set; organises
           representations under ``base_model_id → adapter_id``.
        2. ``cache`` (DiskCache) — flat hash-keyed fallback.
    """

    def __init__(
        self,
        layer_names: list[str] | None = None,
        n_components: int = 256,
        lora_only: bool = True,
        use_lora_product: bool = True,
        cache: DiskCache | None = None,
        lora_cache: LoRACache | None = None,
        base_model_id: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.layer_names = layer_names
        self.n_components = n_components
        self.lora_only = lora_only
        self.use_lora_product = use_lora_product
        self.cache = cache
        self.lora_cache = lora_cache
        self.base_model_id = base_model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

    @property
    def taxonomy_name(self) -> str:
        return "structural"

    def config_dict(self) -> dict[str, Any]:
        return {
            "taxonomy": "structural",
            "layer_names": sorted(self.layer_names) if self.layer_names is not None else None,
            "n_components": self.n_components,
            "lora_only": self.lora_only,
            "use_lora_product": self.use_lora_product,
        }

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        # Try LoRACache first (hierarchical, base_model-aware)
        if self.lora_cache is not None:
            base_id = self._resolve_base_model(model_id)
            if self.lora_cache.exists(base_id, model_id):
                return self.lora_cache.load(base_id, model_id)

        # Fall back to flat DiskCache
        cache_key = DiskCache.key_for(model_id, self.config_dict()) if self.cache else ""
        if self.cache is not None and self.cache.exists(cache_key):
            return self.cache.load(cache_key)

        rep = self._extract_fresh(model_id, cache_key)

        # Persist to whichever cache is configured
        if self.lora_cache is not None:
            base_id = self._resolve_base_model(model_id)
            training_config = self._read_training_config(model_id)
            self.lora_cache.save(
                base_model_id=base_id,
                adapter_id=model_id,
                rep=rep,
                training_config=training_config,
                extraction_config={
                    "n_components": self.n_components,
                    "use_lora_product": self.use_lora_product,
                    "layer_names": self.layer_names,
                },
            )
        elif self.cache is not None:
            self.cache.save(cache_key, rep)

        return rep

    def _resolve_base_model(self, adapter_id: ModelID) -> str:
        if self.base_model_id is not None:
            return self.base_model_id
        return LoRACache.detect_base_model(adapter_id, self.hf_token)

    def _read_training_config(self, adapter_id: ModelID) -> dict:
        try:
            peft_cfg = LoRACache._read_peft_adapter_config(adapter_id, self.hf_token)
            return {
                "lora_rank": peft_cfg.get("r"),
                "lora_alpha": peft_cfg.get("lora_alpha"),
                "target_modules": peft_cfg.get("target_modules"),
                "lora_dropout": peft_cfg.get("lora_dropout"),
            }
        except Exception:
            return {}

    def _extract_fresh(self, model_id: ModelID, cache_key: str) -> ModelRepresentation:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="float32",
            device_map="cpu",
            token=self.hf_token,
            trust_remote_code=True,
        )

        try:
            vectors, layer_labels = self._build_vectors(model)
        finally:
            del model
            gc.collect()

        if not vectors:
            raise ValueError(
                f"No weight layers found for model '{model_id}' with the current "
                f"configuration (lora_only={self.lora_only}, layer_names={self.layer_names}). "
                "Check that the model has LoRA adapters or provide explicit layer_names."
            )

        matrix = np.stack(vectors, axis=0)  # (N_layers, n_components)

        return ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={
                "n_layers": len(vectors),
                "layer_labels": layer_labels,
                "lora_only": self.lora_only,
            },
        )

    def _build_vectors(self, model: Any) -> tuple[list[np.ndarray], list[str]]:
        lora_pairs = _find_lora_pairs(model, self.layer_names)
        has_lora = len(lora_pairs) > 0

        vectors: list[np.ndarray] = []
        labels: list[str] = []

        if has_lora and self.lora_only:
            for module, (A, B) in sorted(lora_pairs.items()):
                if self.use_lora_product:
                    v = (B @ A).flatten()
                else:
                    v = np.concatenate([A.flatten(), B.flatten()])
                vectors.append(_truncate_pad(v, self.n_components))
                labels.append(module)

        elif not has_lora and self.lora_only:
            raise ValueError(
                "lora_only=True but the model has no LoRA adapter parameters. "
                "Use lora_only=False to compare full weight matrices instead."
            )

        elif self.layer_names is not None:
            params = dict(model.named_parameters())
            for name in self.layer_names:
                if name not in params:
                    raise ValueError(
                        f"layer_names entry '{name}' not found in model parameters."
                    )
                v = params[name].detach().float().cpu().numpy().flatten()
                vectors.append(_truncate_pad(v, self.n_components))
                labels.append(name)

        else:
            # Auto-select all 2-D weight matrices above 1024 elements
            for name, param in model.named_parameters():
                if param.ndim == 2 and param.numel() >= 1024:
                    v = param.detach().float().cpu().numpy().flatten()
                    vectors.append(_truncate_pad(v, self.n_components))
                    labels.append(name)

        return vectors, labels
