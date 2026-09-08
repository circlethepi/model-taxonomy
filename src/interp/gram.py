"""The Gram matrix of effective LoRA updates, computed without ever forming one.

Everything this package measures — interpolation residuals, best-fit
coefficients, cosines, distances — is a function of the inner products

    G[i,j] = <ΔW_i, ΔW_j>_F ,    ΔW_i = (lora_alpha / r) · B_i A_i

summed over modules.  Forming ΔW is not an option: for the Qwen3.5-4B adapters
here it is 1.3e9 parameters per adapter -- 5.2 GB at float32, 10.4 GB at the
float64 this module computes in -- and there are 16 of them.  It is also unnecessary, because the inner product collapses into rank
space by the cyclic property of the trace:

    <B_i A_i, B_j A_j>_F = tr(A_i^T B_i^T B_j A_j)
                         = tr((B_i^T B_j)(A_j A_i^T))
                         = sum( (B_i^T B_j) ⊙ (A_i A_j^T) )        [all r×r]

Both factors are r×r, so cost per pair is O(r²(d_out + d_in)) instead of
O(d_out·d_in).  ``src/notebook/structure.py`` uses the same identity for its
pairwise builders; the difference here is that all pairs for a block are
obtained from **two** GEMMs rather than n² small ones (see :func:`compute_gram`).

Two conventions differ from the rest of the repo, both deliberate:

* **The ``alpha / r`` scale is applied.**  ``src/notebook/structure.py`` and
  ``src/metrics/frobenius.py`` operate on the unscaled ``B @ A``.  This module
  is about the *effective* update PEFT actually adds at inference, so it carries
  the factor.  Since alpha/r is shared by every adapter in a collection it
  cancels from cosines, normalized coefficients and relative errors, and scales
  squared distances by exactly ``scale²`` (4.0 here).  Do not "reconcile" the
  two — the difference is intended and this is where it is recorded.

* **Storage is float64, not float32.**  ``DistanceMatrix.save`` and
  ``SimplexProjection.save`` both downcast.  Here the Gram is not an end
  product: a 3×3 submatrix of it is inverted to produce every coefficient, so
  the digits are load-bearing.  The whole per-block tensor is 213 KB.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["LoRAGram", "compute_gram", "gram_cache_key", "load_or_compute_gram"]

SCHEMA_VERSION = "1"


def _threadpool_limits(n: int):
    """threadpool_limits(n) if threadpoolctl is importable, else a no-op.

    This matters far more than it looks.  The BLAS on this cluster reports 128
    threads and OpenMP 256; the GEMMs here are 256×8192 by 8192×256, small
    enough that thread dispatch dominates.  Measured on the real 16-adapter,
    104-block collection: **4.0 s pinned to one thread, 77 s unpinned.**

    threadpoolctl is not in environment.yml — it arrives as a scikit-learn
    dependency — so import it defensively, exactly as scripts/check_analysis.py
    does around its own run loop.
    """
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(n)
    except ImportError:  # pragma: no cover - depends on the environment
        from contextlib import nullcontext

        return nullcontext()


@dataclass
class LoRAGram:
    """Per-module Gram matrix of effective LoRA updates over a set of adapters.

    ``per_block[b, i, j]`` is ``<ΔW_i, ΔW_j>_F`` restricted to block ``b``;
    summing over ``b`` gives the whole-model Gram (:attr:`total`).  Keeping the
    per-block breakdown costs 213 KB and is what makes layer/module subsetting
    and the per-block error profile free after the fact.
    """

    per_block: np.ndarray            # (n_blocks, n, n) float64
    adapter_names: list[str]
    blocks: list[tuple[int, str]]
    scale: float
    base_model_id: str
    rank: int
    lora_alpha: int
    _total: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = len(self.adapter_names)
        expected = (len(self.blocks), n, n)
        if self.per_block.shape != expected:
            raise ValueError(
                f"per_block shape {self.per_block.shape} does not match "
                f"{len(self.blocks)} blocks x {n} adapters {expected}"
            )

    @property
    def total(self) -> np.ndarray:
        """(n, n) Gram summed over every block."""
        if self._total is None:
            self._total = self.per_block.sum(axis=0)
        return self._total

    def index(self, name: str) -> int:
        try:
            return self.adapter_names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not in this Gram. Present: {self.adapter_names}"
            ) from None

    def sub(self, names) -> np.ndarray:
        """The Gram restricted to *names*, in the order given."""
        idx = [self.index(n) for n in names]
        return self.total[np.ix_(idx, idx)]

    def norms(self) -> np.ndarray:
        """‖ΔW_i‖_F for each adapter."""
        return np.sqrt(np.clip(np.diag(self.total), 0.0, None))

    def restrict(self, layers=None, modules=None) -> "LoRAGram":
        """A new Gram over the subset of blocks matching *layers* / *modules*."""
        keep = [
            b for b, (layer, module) in enumerate(self.blocks)
            if (layers is None or layer in set(layers))
            and (modules is None or module in set(modules))
        ]
        if not keep:
            raise ValueError(
                f"No blocks match layers={layers} modules={modules}. "
                f"Available modules: {sorted({m for _, m in self.blocks})}"
            )
        return LoRAGram(
            per_block=self.per_block[keep],
            adapter_names=list(self.adapter_names),
            blocks=[self.blocks[b] for b in keep],
            scale=self.scale,
            base_model_id=self.base_model_id,
            rank=self.rank,
            lora_alpha=self.lora_alpha,
        )

    def distance_matrix(self) -> np.ndarray:
        """(n, n) Frobenius distances ‖ΔW_i − ΔW_j‖_F.

        Exact, from d²ᵢⱼ = Gᵢᵢ + Gⱼⱼ − 2Gᵢⱼ.  The Gram is PSD, so this is a true
        Euclidean distance matrix on the 1.3e9-dimensional weight vectors.
        """
        g = self.total
        d2 = np.diag(g)[:, None] + np.diag(g)[None, :] - 2.0 * g
        np.fill_diagonal(d2, 0.0)
        return np.sqrt(np.clip(d2, 0.0, None))

    def as_distance_matrix(self, taxonomy: str = "structural"):
        """The same distances as a repo :class:`~src.core.distance.DistanceMatrix`."""
        from src.core.distance import DistanceMatrix

        return DistanceMatrix(
            matrix=self.distance_matrix(),
            model_ids=list(self.adapter_names),
            metric="frobenius_delta_w",
            taxonomy=taxonomy,
        )

    def cosine_matrix(self) -> np.ndarray:
        """(n, n) cosine similarity between the ΔW vectors."""
        d = np.sqrt(np.clip(np.diag(self.total), 0.0, None))
        return np.clip(self.total / np.outer(d, d), -1.0, 1.0)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "adapter_names": self.adapter_names,
            "blocks": [[int(l), m] for l, m in self.blocks],
            "scale": float(self.scale),
            "base_model_id": self.base_model_id,
            "rank": int(self.rank),
            "lora_alpha": int(self.lora_alpha),
        }
        save_file(
            {
                "per_block": np.ascontiguousarray(self.per_block, dtype=np.float64),
                "_meta_json": np.frombuffer(
                    json.dumps(meta).encode("utf-8"), dtype=np.uint8
                ),
            },
            str(path / "gram.safetensors"),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LoRAGram":
        from safetensors.numpy import load_file

        tensors = load_file(str(Path(path) / "gram.safetensors"))
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"gram schema {meta.get('schema_version')!r} != {SCHEMA_VERSION!r}; "
                "recompute with --recompute"
            )
        return cls(
            per_block=tensors["per_block"].astype(np.float64),
            adapter_names=list(meta["adapter_names"]),
            blocks=[(int(l), m) for l, m in meta["blocks"]],
            scale=float(meta["scale"]),
            base_model_id=meta["base_model_id"],
            rank=int(meta["rank"]),
            lora_alpha=int(meta["lora_alpha"]),
        )


def compute_gram(
    weights: dict[str, dict[tuple[int, str], dict[str, np.ndarray]]],
    scale: float,
    base_model_id: str = "",
    rank: int = 0,
    lora_alpha: int = 0,
    threads: int = 1,
    progress: bool = True,
) -> LoRAGram:
    """Build the per-block Gram from loaded adapter factors.

    *weights* maps adapter name to ``{(layer, module): {"A", "B"}}``, i.e. the
    output of :func:`src.interp.loading.load_adapter`.  Every adapter must
    expose the same block set.

    Fast path: for a block where all adapters share a rank, stack the factors
    once and take two GEMMs —

        B_all = vstack(B_iᵀ)  (n·r, d_out) ;  BB = B_all @ B_allᵀ  (n·r, n·r)
        A_all = vstack(A_i)   (n·r, d_in)  ;  AA = A_all @ A_allᵀ  (n·r, n·r)
        G_block = einsum('iajb,iajb->ij', BB.reshape(n,r,n,r), AA.reshape(n,r,n,r))

    which is the identity above for all n² pairs at once.  Ranks may legitimately
    differ (a combination of rank-r adapters has rank up to k·r), so a mixed
    block falls back to the pairwise loop; both agree to ~1e-15.
    """
    names = list(weights)
    if not names:
        raise ValueError("No adapters given.")
    n = len(names)

    blocks = sorted(weights[names[0]])
    for name in names[1:]:
        if sorted(weights[name]) != blocks:
            missing = set(blocks) ^ set(weights[name])
            raise ValueError(
                f"Adapter {name!r} has a different block set; symmetric "
                f"difference has {len(missing)} entries, e.g. {sorted(missing)[:3]}"
            )

    scale_sq = float(scale) ** 2
    per_block = np.zeros((len(blocks), n, n), dtype=np.float64)

    iterator = blocks
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(blocks, desc="lora gram")
        except ImportError:  # pragma: no cover
            pass

    with _threadpool_limits(threads):
        for bi, blk in enumerate(iterator):
            ranks = {weights[nm][blk]["A"].shape[0] for nm in names}
            if len(ranks) == 1:
                r = ranks.pop()
                B_all = np.concatenate([weights[nm][blk]["B"].T for nm in names], axis=0)
                A_all = np.concatenate([weights[nm][blk]["A"] for nm in names], axis=0)
                BB = (B_all @ B_all.T).reshape(n, r, n, r)
                AA = (A_all @ A_all.T).reshape(n, r, n, r)
                per_block[bi] = np.einsum("iajb,iajb->ij", BB, AA)
            else:
                for i in range(n):
                    Ai, Bi = weights[names[i]][blk]["A"], weights[names[i]][blk]["B"]
                    for j in range(i, n):
                        Aj, Bj = weights[names[j]][blk]["A"], weights[names[j]][blk]["B"]
                        v = float(np.sum((Bi.T @ Bj) * (Ai @ Aj.T)))
                        per_block[bi, i, j] = v
                        per_block[bi, j, i] = v
    per_block *= scale_sq

    return LoRAGram(
        per_block=per_block,
        adapter_names=names,
        blocks=blocks,
        scale=float(scale),
        base_model_id=base_model_id,
        rank=rank,
        lora_alpha=lora_alpha,
    )


def gram_cache_key(config: dict) -> str:
    """16-char content hash of the cache-key payload.

    Same recipe as :meth:`src.cache.lora_cache.LoRACache._config_hash`, so the
    two caches are keyed the same way.
    """
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def load_or_compute_gram(
    refs,
    cache_dir: str | Path | None = None,
    recompute: bool = False,
    threads: int = 1,
    progress: bool = True,
) -> tuple[LoRAGram, dict]:
    """Return the Gram over *refs*, from cache when it is still valid.

    *refs* is a sequence of :class:`~src.interp.loading.AdapterRef`.  Returns
    ``(gram, provenance)``; *provenance* records the cache key, whether it hit,
    and the per-adapter weight digests.

    The key includes a digest of each adapter's weight file, not just its name.
    Adapter directories get renamed in place after training
    (``scripts/_utils.retag_adapter_dir``), so a name-and-config key would go
    stale silently.  The key is over the **full** block set; subsetting happens
    in memory via :meth:`LoRAGram.restrict`, so a projection subset never causes
    a recompute.
    """
    from src.interp.loading import (
        adapter_scale,
        load_adapter,
        read_base_model_id,
        weights_digest,
    )

    refs = list(refs)
    if not refs:
        raise ValueError("No adapters given.")

    scale, rank, alpha = adapter_scale(refs[0].path)
    for ref in refs[1:]:
        s, r, a = adapter_scale(ref.path)
        if (s, r, a) != (scale, rank, alpha):
            raise ValueError(
                f"{ref.name} has scale/rank/alpha ({s}, {r}, {a}), expected "
                f"({scale}, {rank}, {alpha}). A shared Gram needs one convention."
            )
    base_model_id = read_base_model_id(refs[0].path) or refs[0].base_model_slug

    digests = {ref.name: weights_digest(ref.path) for ref in refs}
    key_payload = {
        "schema_version": SCHEMA_VERSION,
        "base_model_id": base_model_id,
        "adapters": [ref.name for ref in refs],   # order is matrix order
        "scale": scale,
        "dtype": "float64",
        "weights_digest": digests,
    }
    key = gram_cache_key(key_payload)
    provenance = {"gram_hash": key, "cache_hit": False, "weights_digest": digests}

    target = Path(cache_dir) / key if cache_dir is not None else None
    if target is not None and not recompute and (target / "gram.safetensors").is_file():
        gram = LoRAGram.load(target)
        if gram.adapter_names == [ref.name for ref in refs]:
            provenance["cache_hit"] = True
            provenance["cache_path"] = str(target)
            return gram, provenance

    weights = {ref.name: load_adapter(ref.path) for ref in refs}
    gram = compute_gram(
        weights,
        scale=scale,
        base_model_id=base_model_id,
        rank=rank,
        lora_alpha=alpha,
        threads=threads,
        progress=progress,
    )

    if target is not None:
        gram.save(target)
        (target / "config.json").write_text(json.dumps(key_payload, indent=2))
        provenance["cache_path"] = str(target)
    return gram, provenance
