from __future__ import annotations

import gc
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation
from src.cache.disk import DiskCache
from src.cache.lora_cache import LoRACache

# ---------------------------------------------------------------------------
# Module-level constants shared by filter helpers
# ---------------------------------------------------------------------------

_LAYER_RE   = re.compile(r"\.layers\.(\d+)\.")
_PROJ_RE    = re.compile(r"\.(k_proj|q_proj|v_proj|o_proj)(?:\.|$)")
_PROJ_SHORT = {"k": "k_proj", "q": "q_proj", "v": "v_proj", "o": "o_proj"}


# ---------------------------------------------------------------------------
# Filter helpers — used by both _find_lora_pairs and _extract_from_safetensors
# ---------------------------------------------------------------------------

def _build_filter(
    layer_names: list[str] | None,
    layer_indices: int | list[int] | Literal["last"] | None,
    projections: str | list[str] | None,
) -> tuple[list[str] | None, set[str] | None, set[int] | None, bool]:
    """Normalise filter params to (layer_names, proj_long, layer_set, use_last)."""
    proj_long: set[str] | None = None
    if projections is not None:
        proj_list = [projections] if isinstance(projections, str) else list(projections)
        proj_long = {_PROJ_SHORT.get(p, p) for p in proj_list}

    layer_set: set[int] | None = None
    use_last = False
    if layer_indices == "last":
        use_last = True
    elif layer_indices is not None:
        idxs = [layer_indices] if isinstance(layer_indices, int) else list(layer_indices)
        layer_set = set(idxs)

    return layer_names, proj_long, layer_set, use_last


def _module_passes_filter(
    module: str,
    layer_names: list[str] | None,
    proj_long: set[str] | None,
    layer_set: set[int] | None,
) -> bool:
    """True if *module* passes the normalised filter (use_last handled separately)."""
    if layer_names is not None:
        return any(module.startswith(ln) for ln in layer_names)
    if proj_long is not None:
        m = _PROJ_RE.search(module)
        if m is None or m.group(1) not in proj_long:
            return False
    if layer_set is not None:
        m = _LAYER_RE.search(module)
        if m is None or int(m.group(1)) not in layer_set:
            return False
    return True


def _apply_last_filter(
    pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Keep only modules belonging to the highest layer index."""
    def _layer_idx(module: str) -> int:
        m = _LAYER_RE.search(module)
        return int(m.group(1)) if m else -1

    max_idx = max(_layer_idx(m) for m in pairs)
    return {m: v for m, v in pairs.items() if _layer_idx(m) == max_idx}


# ---------------------------------------------------------------------------
# LoRA pair extraction from a loaded model
# ---------------------------------------------------------------------------

def _find_lora_pairs(
    model: Any,
    layer_names: list[str] | None,
    layer_indices: int | list[int] | Literal["last"] | None,
    projections: str | list[str] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {module_name: (A_matrix, B_matrix)} for all detected LoRA adapters."""
    ln, proj_long, layer_set, use_last = _build_filter(layer_names, layer_indices, projections)

    params = dict(model.named_parameters())
    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, param in params.items():
        if ".lora_A." not in name:
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        if b_name not in params:
            continue
        module = name.split(".lora_A.")[0]
        if not _module_passes_filter(module, ln, proj_long, layer_set):
            continue
        pairs[module] = (
            param.detach().float().cpu().numpy(),
            params[b_name].detach().float().cpu().numpy(),
        )

    if use_last and pairs:
        pairs = _apply_last_filter(pairs)

    return pairs


# ---------------------------------------------------------------------------
# Shared post-processing: pad → stack → ModelRepresentation
# ---------------------------------------------------------------------------

def _vectors_to_rep(
    vectors: list[np.ndarray],
    labels: list[str],
    model_id: ModelID,
    taxonomy_name: str,
    config: dict,
    lora_only: bool,
) -> tuple[ModelRepresentation, list[int]]:
    """Pad vectors to uniform length, stack, and create a ModelRepresentation."""
    layer_lengths = [len(v) for v in vectors]

    # TODO (full-pipeline): rows padded to max_len may differ per projection type
    # (e.g. q/o_proj vs k/v_proj in GQA). Before wiring into src/metrics/, decide
    # on a final strategy and update cka.py, frobenius.py. src/notebook/structure.py
    # is unaffected — it works directly from raw A/B matrices.
    max_len = max(layer_lengths)
    padded = [np.pad(v, (0, max_len - len(v))) if len(v) < max_len else v for v in vectors]
    matrix = np.stack(padded, axis=0)  # (N_layers, max_len)

    rep = ModelRepresentation.create(
        model_id=model_id,
        taxonomy=taxonomy_name,
        matrix=matrix,
        config=config,
        metadata={
            "n_layers": len(vectors),
            "layer_labels": labels,
            "lora_only": lora_only,
            "layer_lengths": layer_lengths,
        },
    )
    return rep, layer_lengths


# ---------------------------------------------------------------------------
# StructuralTaxonomy
# ---------------------------------------------------------------------------

class StructuralTaxonomy(Taxonomy):
    """Compares models via the geometry of their weight matrices.

    By default (``lora_only=True``) only LoRA adapter matrices are used, making
    this practical for comparing fine-tuned variants without storing full weight
    matrices. For each LoRA module the adapter product B @ A (or raw A/B
    concatenation when ``use_lora_product=False``) is stored at its full natural
    length — no truncation. Rows of different length are zero-padded so the matrix
    can be stacked; the original unpadded lengths are preserved in the cache config.

    Layer/projection selection (shorthand, architecture-agnostic):
        ``layer_indices`` — int, list[int], ``"last"``, or None (all layers)
        ``projections``   — ``"k"``/``"q"``/``"v"``/``"o"``, list thereof, or None (all)
    For full control, pass ``layer_names`` as explicit module-name prefixes
    (takes precedence over the shorthand params when set).

    For base models or when ``lora_only=False``, explicit ``layer_names`` can be
    given to select which named parameters to compare; if omitted, all 2-D weight
    matrices larger than 1024 elements are included automatically.

    The representation matrix has shape (N_layers, max_len), where each row
    corresponds to one weight layer/adapter, zero-padded to the longest row.

    Extraction priority:
        1. LoRACache hit         — return existing representation immediately.
        2. DiskCache hit         — return existing representation immediately.
        3. Local safetensors     — read A/B tensors directly from the PEFT
                                   ``adapter_model.safetensors`` file stored in the
                                   LoRACache adapter directory (no base model load).
        4. Full model load       — ``AutoModelForCausalLM.from_pretrained`` fallback
                                   for when raw PEFT files are not available locally.

    Cache storage:
        ``lora_cache`` (LoRACache) — checked first when set; organises representations
        under ``base_model_id → adapter_id → config_hash``.
        ``cache`` (DiskCache) — flat hash-keyed fallback.
    """

    def __init__(
        self,
        layer_names: list[str] | None = None,
        layer_indices: int | list[int] | Literal["last"] | None = None,
        projections: str | list[str] | None = None,
        lora_only: bool = True,
        use_lora_product: bool = True,
        cache: DiskCache | None = None,
        lora_cache: LoRACache | None = None,
        base_model_id: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.layer_names = layer_names
        self.layer_indices = layer_indices
        self.projections = projections
        self.lora_only = lora_only
        self.use_lora_product = use_lora_product
        self.cache = cache
        self.lora_cache = lora_cache
        self.base_model_id = base_model_id
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

    @property
    def taxonomy_name(self) -> str:
        return "structural"

    def config_dict(self) -> dict:
        proj = self.projections
        if isinstance(proj, str):
            proj = [proj]
        idx = self.layer_indices
        if isinstance(idx, int):
            idx = [idx]
        return {
            "taxonomy": "structural",
            "layer_names": sorted(self.layer_names) if self.layer_names is not None else None,
            "layer_indices": idx,
            "projections": proj,
            "lora_only": self.lora_only,
            "use_lora_product": self.use_lora_product,
        }

    def _extraction_config(self) -> dict:
        """Subset of config_dict used as the LoRACache key (excludes taxonomy name)."""
        d = self.config_dict()
        d.pop("taxonomy")
        return d

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        extraction_config = self._extraction_config()

        # 1. LoRACache hit
        if self.lora_cache is not None:
            base_id = self._resolve_base_model(model_id)
            if self.lora_cache.exists(base_id, model_id, extraction_config):
                return self.lora_cache.load(base_id, model_id, extraction_config)

        # 2. DiskCache hit
        cache_key = DiskCache.key_for(model_id, self.config_dict()) if self.cache else ""
        if self.cache is not None and self.cache.exists(cache_key):
            return self.cache.load(cache_key)

        # 3. Fast path: read directly from local PEFT safetensors (no base model load)
        vectors: list[np.ndarray] | None = None
        labels: list[str] | None = None
        local_adapter_dir: Path | None = None

        if self.lora_only and self.lora_cache is not None:
            base_id = self._resolve_base_model(model_id)
            local_adapter_dir = self.lora_cache._adapter_dir(base_id, model_id)
            st_path = local_adapter_dir / "adapter_model.safetensors"
            if st_path.exists():
                vectors, labels = self._extract_from_safetensors(st_path)

        # 4. Full model load fallback
        if vectors is None:
            vectors, labels = self._extract_via_model(model_id)

        if not vectors:
            raise ValueError(
                f"No weight layers found for model '{model_id}' with the current "
                f"configuration (lora_only={self.lora_only}, layer_names={self.layer_names}, "
                f"layer_indices={self.layer_indices}, projections={self.projections}). "
                "Check that the model has LoRA adapters or provide explicit layer_names."
            )

        rep, layer_lengths = _vectors_to_rep(
            vectors, labels, model_id, self.taxonomy_name, self.config_dict(), self.lora_only
        )

        # Persist
        if self.lora_cache is not None:
            base_id = self._resolve_base_model(model_id)
            training_config = self._read_training_config(model_id, local_adapter_dir)
            self.lora_cache.save(
                base_model_id=base_id,
                adapter_id=model_id,
                rep=rep,
                training_config=training_config,
                extraction_config=extraction_config,
                layer_lengths=layer_lengths,
            )
        elif self.cache is not None:
            self.cache.save(cache_key, rep)

        return rep

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_base_model(self, adapter_id: ModelID) -> str:
        if self.base_model_id is not None:
            return self.base_model_id
        return LoRACache.detect_base_model(adapter_id, self.hf_token)

    def _read_training_config(
        self, adapter_id: ModelID, local_adapter_dir: Path | None = None
    ) -> dict:
        # Check local adapter_config.json first (avoids a Hub round-trip)
        if local_adapter_dir is not None:
            local_cfg = local_adapter_dir / "adapter_config.json"
            if local_cfg.exists():
                peft_cfg = json.loads(local_cfg.read_text())
                return {
                    "lora_rank": peft_cfg.get("r"),
                    "lora_alpha": peft_cfg.get("lora_alpha"),
                    "target_modules": peft_cfg.get("target_modules"),
                    "lora_dropout": peft_cfg.get("lora_dropout"),
                }
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

    def _extract_from_safetensors(
        self, safetensors_path: Path
    ) -> tuple[list[np.ndarray], list[str]]:
        """Read LoRA A/B tensors directly from a PEFT safetensors file.

        Does not load the base model. Uses the same filter logic as
        _find_lora_pairs so results are identical to the full-model-load path.
        """
        from safetensors import safe_open

        ln, proj_long, layer_set, use_last = _build_filter(
            self.layer_names, self.layer_indices, self.projections
        )

        A_mats: dict[str, np.ndarray] = {}
        B_mats: dict[str, np.ndarray] = {}

        with safe_open(str(safetensors_path), framework="pt") as f:
            for key in sorted(f.keys()):
                if ".lora_A." in key:
                    module = key.split(".lora_A.")[0]
                elif ".lora_B." in key:
                    module = key.split(".lora_B.")[0]
                else:
                    continue
                if not _module_passes_filter(module, ln, proj_long, layer_set):
                    continue
                arr = f.get_tensor(key).numpy().astype(np.float32)
                if ".lora_A." in key:
                    A_mats[module] = arr
                else:
                    B_mats[module] = arr

        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {
            m: (A_mats[m], B_mats[m]) for m in A_mats if m in B_mats
        }
        if use_last and pairs:
            pairs = _apply_last_filter(pairs)

        vectors: list[np.ndarray] = []
        labels: list[str] = []
        for module, (A, B) in sorted(pairs.items()):
            v = (B @ A).flatten() if self.use_lora_product else np.concatenate([A.flatten(), B.flatten()])
            vectors.append(v)
            labels.append(module)

        return vectors, labels

    def _extract_via_model(self, model_id: ModelID) -> tuple[list[np.ndarray], list[str]]:
        """Load the full model and extract vectors. Fallback when no local PEFT file."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="float32",
            device_map="cpu",
            token=self.hf_token,
            trust_remote_code=True,
        )
        try:
            return self._build_vectors(model)
        finally:
            del model
            gc.collect()

    def _build_vectors(self, model: Any) -> tuple[list[np.ndarray], list[str]]:
        lora_pairs = _find_lora_pairs(
            model, self.layer_names, self.layer_indices, self.projections
        )
        has_lora = len(lora_pairs) > 0

        vectors: list[np.ndarray] = []
        labels: list[str] = []

        if has_lora and self.lora_only:
            for module, (A, B) in sorted(lora_pairs.items()):
                v = (B @ A).flatten() if self.use_lora_product else np.concatenate([A.flatten(), B.flatten()])
                vectors.append(v)
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
                vectors.append(v)
                labels.append(name)

        else:
            for name, param in model.named_parameters():
                if param.ndim == 2 and param.numel() >= 1024:
                    v = param.detach().float().cpu().numpy().flatten()
                    vectors.append(v)
                    labels.append(name)

        return vectors, labels
