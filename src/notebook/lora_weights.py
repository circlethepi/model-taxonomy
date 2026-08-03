from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np

# Matches: ...layers.{i}.self_attn.{proj}.lora_{A|B}.weight
_KEY_RE = re.compile(
    r"\.layers\.(\d+)\.self_attn\.(k_proj|q_proj|v_proj|o_proj)\.lora_(A|B)\.weight$"
)
_PROJ_SHORT = {"k": "k_proj", "q": "q_proj", "v": "v_proj", "o": "o_proj"}
_PROJ_LONG  = {v: k for k, v in _PROJ_SHORT.items()}

MatrixChoice = Literal["A", "B", "both"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class LayerMatrices:
    """A and/or B matrices for one (layer_idx, projection) pair."""

    def __init__(self, layer: int, proj: str, A: np.ndarray | None, B: np.ndarray | None) -> None:
        self.layer = layer
        self.proj  = proj
        self.A     = A   # (rank, in_features)  or None
        self.B     = B   # (out_features, rank) or None

    def get(self, which: MatrixChoice = "both") -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if which == "A":
            return self.A
        if which == "B":
            return self.B
        return self.A, self.B

    def product(self) -> np.ndarray:
        """Return the low-rank weight delta B @ A, shape (out_features, in_features)."""
        if self.A is None or self.B is None:
            raise ValueError(
                f"Both A and B must be loaded to compute the product "
                f"(layer={self.layer}, proj={self.proj!r})."
            )
        return self.B @ self.A

    def __repr__(self) -> str:
        parts = []
        if self.A is not None:
            parts.append(f"A={self.A.shape}")
        if self.B is not None:
            parts.append(f"B={self.B.shape}")
        return f"LayerMatrices(layer={self.layer}, proj={self.proj!r}, {', '.join(parts)})"


class AdapterWeights:
    """Raw LoRA A/B matrices for one adapter, keyed by (layer_idx, proj_short)."""

    def __init__(self, name: str, data: dict[tuple[int, str], LayerMatrices]) -> None:
        self.name  = name
        self._data = data

    # --- introspection ---

    @property
    def layers(self) -> list[int]:
        return sorted({k[0] for k in self._data})

    @property
    def projections(self) -> list[str]:
        return sorted({k[1] for k in self._data})

    # --- access ---

    def get_layer(self, idx: int) -> dict[str, LayerMatrices]:
        """Return {proj_short: LayerMatrices} for all projections at *idx*."""
        return {k[1]: v for k, v in self._data.items() if k[0] == idx}

    def get(self, layer: int, proj: str) -> LayerMatrices:
        """Return LayerMatrices for a specific (layer, proj_short) pair."""
        return self._data[_resolve_proj_key(layer, proj)]

    def matrix(
        self, layer: int, proj: str, which: MatrixChoice = "both"
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return the A matrix, B matrix, or both for a specific (layer, proj)."""
        return self.get(layer, proj).get(which)

    def product(self, layer: int, proj: str) -> np.ndarray:
        """Return B @ A for a specific (layer, proj_short) pair."""
        return self.get(layer, proj).product()

    def __getitem__(self, key: tuple[int, str]) -> LayerMatrices:
        layer, proj = key
        return self.get(layer, proj)

    def __repr__(self) -> str:
        return (
            f"AdapterWeights(name={self.name!r}, "
            f"layers={self.layers}, projections={self.projections})"
        )


class LoRAWeightCollection:
    """Dict-like collection of AdapterWeights, one per adapter name."""

    def __init__(self, adapters: dict[str, AdapterWeights]) -> None:
        self._adapters = adapters

    def __getitem__(self, name: str) -> AdapterWeights:
        return self._adapters[name]

    def __iter__(self):
        return iter(self._adapters)

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def keys(self) -> list[str]:
        return list(self._adapters)

    @property
    def layers(self) -> list[int]:
        return sorted({layer for aw in self._adapters.values() for layer in aw.layers})

    @property
    def projections(self) -> list[str]:
        return sorted({proj for aw in self._adapters.values() for proj in aw.projections})

    def get_all(
        self,
        layer: int,
        proj: str,
        which: MatrixChoice = "both",
    ) -> dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]]:
        """Return {adapter_name: matrix} across all adapters for (layer, proj)."""
        return {name: aw.matrix(layer, proj, which) for name, aw in self._adapters.items()}

    def get_all_layers(
        self,
        proj: str,
        which: MatrixChoice = "both",
    ) -> dict[str, dict[int, np.ndarray | tuple[np.ndarray, np.ndarray]]]:
        """Return {adapter_name: {layer: matrix}} across all adapters and loaded layers."""
        return {
            name: {layer: aw.matrix(layer, proj, which) for layer in aw.layers}
            for name, aw in self._adapters.items()
        }

    def get_all_products(
        self,
        layer: int,
        proj: str,
    ) -> dict[str, np.ndarray]:
        """Return {adapter_name: B @ A} across all adapters for (layer, proj)."""
        return {name: aw.product(layer, proj) for name, aw in self._adapters.items()}

    def get_all_layers_product(
        self,
        proj: str,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Return {adapter_name: {layer: B @ A}} across all adapters and loaded layers."""
        return {
            name: {layer: aw.product(layer, proj) for layer in aw.layers}
            for name, aw in self._adapters.items()
        }

    def __repr__(self) -> str:
        return (
            f"LoRAWeightCollection({list(self._adapters)}, "
            f"layers={self.layers}, projections={self.projections})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_lora_weights(
    model_names: str | list[str],
    adapter_root: str | Path = "results/shared_cache/03_adapters",
    layer_indices: int | list[int] | Literal["last"] = "last",
    projections: str | list[str] = "o",
    matrices: MatrixChoice = "both",
) -> LoRAWeightCollection:
    """Load raw LoRA A/B weight tensors from PEFT adapter safetensors files.

    Parameters
    ----------
    model_names:
        One or more adapter folder names to load (e.g. ``"yahoo_100t0_n1000_s42"``).
    adapter_root:
        Root directory to search for adapter folders. Each adapter is expected
        at ``{adapter_root}/{name}/adapter_model.safetensors`` (flat) or
        ``{adapter_root}/*/{name}/adapter_model.safetensors`` (one level nested,
        e.g. grouped by base-model slug).
    layer_indices:
        Transformer layer index/indices to extract. ``"last"`` picks the highest
        layer index found in the file.
    projections:
        Which attention projections to include: ``"k"``, ``"q"``, ``"v"``, ``"o"``
        (short form) or ``"k_proj"`` etc. (long form). Default ``"o"``.
    matrices:
        Which PEFT matrices to load: ``"A"``, ``"B"``, or ``"both"``.

    Returns
    -------
    LoRAWeightCollection
        Dict-like object keyed by adapter name.
    """
    from safetensors import safe_open

    adapter_root = Path(adapter_root)
    names  = [model_names] if isinstance(model_names, str) else list(model_names)
    projs  = _normalise_projections(projections)

    loaded: dict[str, AdapterWeights] = {}
    for name in names:
        st_path = _find_safetensors(adapter_root, name)
        # framework="numpy" keeps torch off the distance-computation import path
        # entirely — measured ~587 MB of import overhead against ~190 MB.  Safe
        # because PEFT adapter factors are float32, which numpy has; a bfloat16
        # adapter would need framework="pt" and a .float() before .numpy().
        with safe_open(str(st_path), framework="numpy") as f:
            all_keys = list(f.keys())
            n_layers = _max_layer(all_keys) + 1
            indices  = _resolve_layer_indices(layer_indices, n_layers)
            data: dict[tuple[int, str], LayerMatrices] = {}
            for key in all_keys:
                m = _KEY_RE.search(key)
                if m is None:
                    continue
                layer_idx = int(m.group(1))
                proj_long = m.group(2)
                ab        = m.group(3)
                proj_short = _PROJ_LONG[proj_long]
                if layer_idx not in indices or proj_short not in projs:
                    continue
                if matrices == "A" and ab != "A":
                    continue
                if matrices == "B" and ab != "B":
                    continue
                cell_key = (layer_idx, proj_short)
                if cell_key not in data:
                    data[cell_key] = LayerMatrices(layer_idx, proj_short, None, None)
                arr = f.get_tensor(key)
                if ab == "A":
                    data[cell_key].A = arr
                else:
                    data[cell_key].B = arr
        loaded[name] = AdapterWeights(name, data)

    return LoRAWeightCollection(loaded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_safetensors(root: Path, name: str) -> Path:
    """Locate adapter_model.safetensors under *root* for adapter *name*."""
    # flat: root/name/adapter_model.safetensors
    direct = root / name / "adapter_model.safetensors"
    if direct.exists():
        return direct
    # one level nested: root/*/name/adapter_model.safetensors
    for candidate in sorted(root.iterdir()):
        nested = candidate / name / "adapter_model.safetensors"
        if nested.exists():
            return nested
    raise FileNotFoundError(
        f"No adapter_model.safetensors found for {name!r} under {root}.\n"
        "Check that adapter_root points to a directory containing raw PEFT adapter folders\n"
        "(not the shared_cache extracted representations)."
    )


def _normalise_projections(projections: str | list[str]) -> set[str]:
    raw = [projections] if isinstance(projections, str) else list(projections)
    out = set()
    for p in raw:
        p = p.lower()
        if p in _PROJ_SHORT:
            out.add(p)
        elif p in _PROJ_LONG:
            out.add(_PROJ_LONG[p])
        else:
            raise ValueError(
                f"Unknown projection {p!r}. Use 'k', 'q', 'v', 'o' "
                "(or 'k_proj', 'q_proj', 'v_proj', 'o_proj')."
            )
    return out


def _max_layer(keys: list[str]) -> int:
    indices = [int(m.group(1)) for k in keys if (m := _KEY_RE.search(k))]
    if not indices:
        raise ValueError("No LoRA layer keys found in safetensors file.")
    return max(indices)


def _resolve_layer_indices(
    layer_indices: int | list[int] | Literal["last"],
    n_layers: int,
) -> set[int]:
    if layer_indices == "last":
        return {n_layers - 1}
    if isinstance(layer_indices, int):
        return {layer_indices}
    return set(layer_indices)


def _resolve_proj_key(layer: int, proj: str) -> tuple[int, str]:
    p = proj.lower()
    if p in _PROJ_SHORT:        # short form: "k", "q", "v", "o"
        return (layer, p)
    if p in _PROJ_LONG:         # long form: "k_proj" -> "k"
        return (layer, _PROJ_LONG[p])
    raise ValueError(f"Unknown projection {proj!r}")
