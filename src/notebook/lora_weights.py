from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np

# Matches: ...layers.{i}.{family}.{proj}.lora_{A|B}.weight
#
# Two attention families, because hybrid models carry both.  Qwen3.5 sets
# `full_attention_interval: 4`, so 3 of every 4 layers are `linear_attn`
# (a gated-delta-rule recurrence) and the 4th is ordinary softmax `self_attn`.
# Matching only `self_attn` — as this pattern did until now — silently sees
# 64 of a Qwen adapter's 208 tensors and drops the three largest matrices.
# Llama-style models have only `self_attn`, so nothing about them changes.
_KEY_RE = re.compile(
    r"\.layers\.(\d+)\.(self_attn|linear_attn)\."
    r"(k_proj|q_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj)"
    r"\.lora_(A|B)\.weight$"
)

#: short name -> (family, tensor name, row-slice kind)
#:
#: The slice kind is None for a whole tensor.  ``q_query``/``q_gate`` are
#: *pseudo-projections*: on a model with `attn_output_gate: true` the single
#: `q_proj` tensor carries the queries and an output gate fused together, and
#: these two names address the halves separately.  Plain ``q`` remains the
#: whole tensor on every model, so existing callers are unaffected.
_PROJ_SPECS: dict[str, tuple[str, str, str | None]] = {
    "k":       ("self_attn",   "k_proj",      None),
    "q":       ("self_attn",   "q_proj",      None),
    "v":       ("self_attn",   "v_proj",      None),
    "o":       ("self_attn",   "o_proj",      None),
    "q_query": ("self_attn",   "q_proj",      "query"),
    "q_gate":  ("self_attn",   "q_proj",      "gate"),
    "qkv":     ("linear_attn", "in_proj_qkv", None),
    "z":       ("linear_attn", "in_proj_z",   None),
    "out":     ("linear_attn", "out_proj",    None),
}

#: short -> long, for the whole-tensor names only (the pseudo-projections have
#: no distinct tensor of their own).
_PROJ_SHORT = {s: t for s, (_, t, sl) in _PROJ_SPECS.items() if sl is None}
#: long -> short.  ``q_proj`` maps back to ``q``, never to a gated half.
_PROJ_LONG  = {t: s for s, t in _PROJ_SHORT.items()}

#: pseudo-projections, which need `attn_num_heads` to be resolvable
_GATED_PROJS = {s for s, (_, _, sl) in _PROJ_SPECS.items() if sl is not None}

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

    @property
    def cells(self) -> set[tuple[int, str]]:
        """The (layer, proj_short) pairs actually held.

        On a hybrid model this is a strict subset of ``layers × projections``,
        since each layer carries only its own family's projections.
        """
        return set(self._data)

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
    attn_num_heads: int | None = None,
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
        Which projections to include. Softmax-attention blocks: ``"k"``, ``"q"``,
        ``"v"``, ``"o"`` (or the long forms ``"k_proj"`` etc.). Linear-attention
        blocks: ``"qkv"`` (``in_proj_qkv``), ``"z"`` (``in_proj_z``), ``"out"``
        (``out_proj``). Default ``"o"``.

        ``"q_query"`` and ``"q_gate"`` address the two halves of a fused, gated
        ``q_proj`` separately and require *attn_num_heads*; plain ``"q"`` is
        always the whole tensor.

        A hybrid model carries both families on different layers, so a
        (layer, projection) pair that does not exist is simply absent from the
        result rather than an error.
    matrices:
        Which PEFT matrices to load: ``"A"``, ``"B"``, or ``"both"``.
    attn_num_heads:
        Number of attention heads, needed only to resolve ``"q_query"`` /
        ``"q_gate"``, whose halves are interleaved per head.

    Returns
    -------
    LoRAWeightCollection
        Dict-like object keyed by adapter name.
    """
    from safetensors import safe_open

    adapter_root = Path(adapter_root)
    names  = [model_names] if isinstance(model_names, str) else list(model_names)
    projs  = _normalise_projections(projections)

    gated = projs & _GATED_PROJS
    if gated and attn_num_heads is None:
        raise ValueError(
            f"{sorted(gated)} address halves of a fused gated q_proj, whose rows "
            "are interleaved per head, so attn_num_heads must be given."
        )

    # Which underlying tensor each requested short name reads from.  Several
    # short names can share one tensor (q_query and q_gate both read q_proj),
    # so this is a tensor -> [(short, slice_kind)] fan-out.
    wanted: dict[str, list[tuple[str, str | None]]] = {}
    for short in projs:
        _family, tensor, slice_kind = _PROJ_SPECS[short]
        wanted.setdefault(tensor, []).append((short, slice_kind))

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
                tensor    = m.group(3)
                ab        = m.group(4)
                if layer_idx not in indices or tensor not in wanted:
                    continue
                if matrices == "A" and ab != "A":
                    continue
                if matrices == "B" and ab != "B":
                    continue
                arr = f.get_tensor(key)
                for proj_short, slice_kind in wanted[tensor]:
                    # Only B carries output features, so only B is sliced; A is
                    # shared between the query and gate halves.
                    value = arr
                    if slice_kind is not None and ab == "B":
                        value = _split_gated_q(arr, slice_kind, attn_num_heads)
                    cell_key = (layer_idx, proj_short)
                    if cell_key not in data:
                        data[cell_key] = LayerMatrices(layer_idx, proj_short, None, None)
                    if ab == "A":
                        data[cell_key].A = value
                    else:
                        data[cell_key].B = value
        if not data:
            # An empty result is almost always a family mismatch — asking for a
            # linear-attention projection at a layer that runs softmax attention,
            # or vice versa.  Silently returning nothing would surface much later
            # as an unexplained empty distance matrix.
            present = sorted({
                (int(mm.group(1)), mm.group(3))
                for k in all_keys if (mm := _KEY_RE.search(k))
            })
            raise ValueError(
                f"No LoRA tensors matched for adapter {name!r}: "
                f"layers={sorted(indices)}, projections={sorted(projs)}.\n"
                f"That adapter has {len(present)} (layer, tensor) pairs, e.g. "
                f"{present[:6]}."
            )
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
        if p in _PROJ_SPECS:
            out.add(p)
        elif p in _PROJ_LONG:
            out.add(_PROJ_LONG[p])
        else:
            raise ValueError(
                f"Unknown projection {p!r}. Use one of {sorted(_PROJ_SPECS)} "
                f"or a long form in {sorted(_PROJ_LONG)}."
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
    if p in _PROJ_SPECS:        # short form: "k", "q", "qkv", "q_gate", ...
        return (layer, p)
    if p in _PROJ_LONG:         # long form: "k_proj" -> "k"
        return (layer, _PROJ_LONG[p])
    raise ValueError(f"Unknown projection {proj!r}")


def _split_gated_q(arr: np.ndarray, which: str, n_heads: int) -> np.ndarray:
    """Take the query or gate half of a fused, gated ``q_proj`` tensor.

    With ``attn_output_gate: true`` the model computes

        query, gate = torch.chunk(q_proj(x).view(..., n_heads, head_dim * 2), 2, dim=-1)

    (`transformers/models/qwen3_5/modeling_qwen3_5.py:683`), so the two halves are
    **interleaved per head** — each head owns ``head_dim`` query rows followed by
    ``head_dim`` gate rows — and a contiguous top/bottom split would mix them.

    Only the ``B`` factor is split: it is the one whose rows are output features.
    ``A`` is shared by both halves and is returned unchanged by the caller.
    """
    if which not in ("query", "gate"):
        raise ValueError(f"which must be 'query' or 'gate', got {which!r}")
    out_features = arr.shape[0]
    if out_features % (2 * n_heads) != 0:
        raise ValueError(
            f"q_proj has {out_features} output rows, which is not divisible by "
            f"2 * attn_num_heads ({2 * n_heads}). Either this model is not "
            f"output-gated, or attn_num_heads is wrong."
        )
    head_dim = out_features // (2 * n_heads)
    reshaped = arr.reshape(n_heads, 2 * head_dim, *arr.shape[1:])
    part = reshaped[:, :head_dim] if which == "query" else reshaped[:, head_dim:]
    return part.reshape(n_heads * head_dim, *arr.shape[1:])
