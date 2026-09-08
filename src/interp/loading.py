"""Architecture-agnostic reading of PEFT LoRA adapters.

This exists alongside :mod:`src.notebook.lora_weights` rather than extending it.
That loader hard-codes the four full-attention projections::

    _KEY_RE = r"\\.layers\\.(\\d+)\\.self_attn\\.(k_proj|q_proj|v_proj|o_proj)\\.lora_(A|B)\\.weight$"

which is correct for Llama-style stacks but silently drops most of a *hybrid*
attention model.  On the Qwen3.5-4B adapters this package was written for, 24 of
the 32 layers are gated-delta ``linear_attn`` blocks (``in_proj_qkv``,
``in_proj_z``, ``out_proj``) and only 8 are ``self_attn``; that regex matches 64
of 208 tensors, so 72 of 104 modules would vanish without an error.  The reader
here matches any ``layers.{i}.{path}.lora_{A|B}.weight`` key and *asserts* that
the resulting ``(layer, module)`` keys are unique, so an architecture it cannot
represent fails loudly instead of quietly losing weights.

Nothing in this module writes to the adapter tree.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ADAPTER_LEAF_RE",
    "KEY_RE",
    "AdapterRef",
    "adapter_scale",
    "discover_adapters",
    "load_adapter",
    "mixture_proportions",
    "parse_adapter_leaf",
    "read_base_model_id",
    "weights_digest",
]

# scripts/_utils.adapter_dir builds an adapter directory name as
#   {dataset_name}_r{rank}[_i{init_seed}][_b{samples_seen}][_f{prompt_format_id}]
# Read wide: every segment after the rank is optional, because they were each
# introduced at a different point and older adapters simply lack them.  The
# repo's own _draw.py sets this precedent explicitly ("writing is narrow,
# reading is wide").  The copies in src/analysis/{identity,discovery}.py stop at
# _b and so fail outright on any adapter carrying the _f suffix.
ADAPTER_LEAF_RE = re.compile(
    r"^(?P<name>.+?)_r(?P<rank>\d+)(?:_i(?P<init>\d+))?"
    r"(?:_b(?P<budget>\d+))?(?:_f(?P<fmt>[0-9a-f]+))?$"
)

# Non-greedy {path} so deeply nested modules (MoE experts, say) still parse; the
# module name is the last dotted segment.
KEY_RE = re.compile(
    r"\.layers\.(?P<layer>\d+)\.(?P<path>.+?)\.lora_(?P<ab>A|B)\.weight$"
)

# A mixture segment: "025g1" -> ("025", "g", "1"), "100t0" -> ("100", "t", "0").
# Covers both the 3-group simplex naming and the older 2-topic naming.
_MIXTURE_RE = re.compile(r"(\d+)([a-z]+)(\d+)")


@dataclass(frozen=True)
class AdapterRef:
    """One adapter on disk."""

    name: str
    path: Path
    base_model_slug: str

    @property
    def recipe(self) -> str:
        """The dataset name, with the training-suffix segments stripped."""
        parsed = parse_adapter_leaf(self.name)
        return parsed["name"] if parsed else self.name


def parse_adapter_leaf(name: str) -> dict[str, str | None] | None:
    """Split an adapter directory name into its recorded fields.

    Returns ``{name, rank, init, budget, fmt}`` or ``None`` if *name* does not
    carry a ``_r{rank}`` segment at all.
    """
    m = ADAPTER_LEAF_RE.match(name)
    return dict(m.groupdict()) if m else None


def discover_adapters(
    adapter_root: str | Path,
    base_model_slug: str | None = None,
) -> list[AdapterRef]:
    """Find every adapter directory under *adapter_root*.

    Handles both the flat layout (``root/{adapter}``) and the cache layout
    ``root/{base_model_slug}/{adapter}`` that :class:`~src.cache.lora_cache.LoRACache`
    writes.  With *base_model_slug* given, only that subtree is scanned.
    """
    root = Path(adapter_root)
    if not root.is_dir():
        raise FileNotFoundError(f"adapter_root does not exist: {root}")

    def _leaves(parent: Path, slug: str) -> list[AdapterRef]:
        out = []
        for child in sorted(parent.iterdir()):
            if (child / "adapter_model.safetensors").is_file():
                out.append(AdapterRef(child.name, child, slug))
        return out

    if base_model_slug is not None:
        subtree = root / base_model_slug.replace("/", "--")
        if not subtree.is_dir():
            raise FileNotFoundError(
                f"No subtree for base model {base_model_slug!r} under {root}. "
                f"Present: {sorted(p.name for p in root.iterdir() if p.is_dir())}"
            )
        return _leaves(subtree, subtree.name)

    found = _leaves(root, root.name)
    if found:
        return found
    for child in sorted(root.iterdir()):
        if child.is_dir():
            found.extend(_leaves(child, child.name))
    if not found:
        raise FileNotFoundError(
            f"No adapter_model.safetensors found under {root} (checked {root} "
            "and one level below it)."
        )
    return found


def load_adapter(path: str | Path) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Read every LoRA factor pair from an adapter directory.

    Returns ``{(layer, module): {"A": (r, d_in), "B": (d_out, r)}}`` in float64.

    Raises if two distinct module paths collapse onto the same ``(layer,
    module)`` key — that would mean the last one read silently overwrote the
    others, which is exactly the failure mode this module exists to prevent.
    """
    from safetensors import safe_open

    st = Path(path) / "adapter_model.safetensors"
    if not st.is_file():
        raise FileNotFoundError(f"No adapter_model.safetensors in {path}")

    data: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    origin: dict[tuple[int, str], str] = {}
    skipped: list[str] = []

    # framework="numpy" for the reason src/notebook/lora_weights.py:224 records
    # (keeps torch off the import path) and because safe_open(framework="pt") is
    # unusable in this repo's conda env — torch fails to load with an undefined
    # ncclCommResume symbol.  PEFT factors are float32, which numpy has.
    with safe_open(str(st), framework="numpy") as f:
        for key in f.keys():
            m = KEY_RE.search(key)
            if m is None:
                skipped.append(key)
                continue
            layer = int(m.group("layer"))
            module = m.group("path").rsplit(".", 1)[-1]
            cell = (layer, module)
            if cell in origin and origin[cell] != m.group("path"):
                raise ValueError(
                    f"Ambiguous module key {cell!r} in {path}: both "
                    f"{origin[cell]!r} and {m.group('path')!r} reduce to it. "
                    "Keying by the trailing dotted segment is not unique for "
                    "this architecture."
                )
            origin[cell] = m.group("path")
            data.setdefault(cell, {})[m.group("ab")] = f.get_tensor(key).astype(np.float64)

    if not data:
        raise ValueError(f"No LoRA keys matched in {st}")

    incomplete = [k for k, v in data.items() if "A" not in v or "B" not in v]
    if incomplete:
        raise ValueError(f"Blocks missing an A or B factor in {path}: {sorted(incomplete)}")
    if skipped:
        raise ValueError(
            f"{len(skipped)} tensor(s) in {st} did not match the LoRA key pattern, "
            f"e.g. {skipped[0]!r}. Refusing to proceed with a partial read."
        )
    return data


def adapter_scale(path: str | Path) -> tuple[float, int, int]:
    """Return ``(alpha / r, r, alpha)`` from ``adapter_config.json``.

    PEFT applies ``ΔW = (lora_alpha / r) · B @ A`` at inference, so this factor
    is part of the effective update, not a display convention.  ``use_rslora``
    would change it to ``alpha / sqrt(r)``; we refuse rather than guess.
    """
    cfg_path = Path(path) / "adapter_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"No adapter_config.json in {path}")
    cfg = json.loads(cfg_path.read_text())
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    if cfg.get("use_rslora"):
        raise NotImplementedError(
            f"{path} sets use_rslora; the effective scale is alpha/sqrt(r), not "
            "alpha/r. Add explicit support before using this adapter."
        )
    if cfg.get("use_dora"):
        raise NotImplementedError(
            f"{path} sets use_dora; the update is not a plain low-rank product."
        )
    return alpha / r, r, alpha


def read_base_model_id(path: str | Path) -> str | None:
    """Base model this adapter was trained on, from adapter_config.json."""
    cfg_path = Path(path) / "adapter_config.json"
    if not cfg_path.is_file():
        return None
    return json.loads(cfg_path.read_text()).get("base_model_name_or_path")


def mixture_proportions(path: str | Path) -> tuple[list[str], np.ndarray] | None:
    """Recover the training mixture from an adapter's recorded dataset name.

    Returns ``(component_labels, weights)`` with *weights* summing to 1, or
    ``None`` if the name encodes no mixture.

    Prefers ``experiment_meta.json["dataset_name"]``, which ``finetune_lora.py``
    writes and is therefore exact; falls back to the directory name.

    **Weights are normalized by their own sum, never by 100.**  The generator
    (``scripts/gen_simplex3.py``) rounds only the *label*: the even three-way
    mixture is written ``033g1_033g2_033g3``, which sums to 99 while the recipe
    carries exact 1:1:1.  Dividing by 99 gives exactly 1/3; dividing by 100
    would put a 1% error into the centroid, the single most informative point.
    """
    p = Path(path)
    name = None
    meta = p / "experiment_meta.json"
    try:
        if meta.is_file():
            recorded = json.loads(meta.read_text()).get("dataset_name")
            if isinstance(recorded, str) and recorded:
                name = recorded
    except (OSError, json.JSONDecodeError):
        pass  # unreadable or malformed — fall through to the directory name

    if name is None:
        parsed = parse_adapter_leaf(p.name)
        name = parsed["name"] if parsed else p.name

    parts = _MIXTURE_RE.findall(name)
    if not parts:
        return None

    labels = [f"{letter}{idx}" for _, letter, idx in parts]
    values = np.array([float(pct) for pct, _, _ in parts], dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return None
    return labels, values / total


def weights_digest(path: str | Path) -> str:
    """A cheap fingerprint of an adapter's weight file.

    Digests the safetensors *header* (its length prefix plus the JSON block
    naming every tensor with its dtype, shape and byte offsets) together with
    the file size and mtime — a few milliseconds, versus ~50 MB of reading for a
    full content hash.

    This is needed because a config-only cache key is unsafe here:
    ``scripts/_utils.retag_adapter_dir`` renames adapter directories in place
    after training, so the same path can hold different weights over time.
    """
    import hashlib

    st = Path(path) / "adapter_model.safetensors"
    with open(st, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = f.read(n)
    stat = st.stat()
    h = hashlib.sha256()
    h.update(n.to_bytes(8, "little"))
    h.update(header)
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()[:16]
