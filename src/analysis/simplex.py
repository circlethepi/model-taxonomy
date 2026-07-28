"""Level 3 — projection onto the simplex spanned by anchor models.

Pick ``k`` models as *anchors*; they span a ``(k-1)``-dimensional simplex inside
whatever ``d``-dimensional space the embedding used.  Two anchors give the line
segment between them, three a triangle, and so on — ``k`` and ``d`` are
independent.  Every model is then written in barycentric coordinates of that
simplex: a weight per anchor, summing to one, plus a residual recording how far
off the simplex it actually sits.

Why this is the right tool for comparing geometries
---------------------------------------------------
An embedding's coordinates are not meaningful on their own — MDS fixes a
configuration only up to rotation, reflection, translation and overall scale, and
picks among those arbitrarily.  Barycentric coordinates quotient that ambiguity
out, so two geometries built from *different* taxonomies become directly
comparable with no Procrustes superposition in between.

Precisely: for a point lying **in** the affine hull of the anchors, barycentric
coordinates are invariant under *any* invertible affine map applied to the
anchors and the point together.  Points off the hull are handled by orthogonal
least-squares projection, which depends on the metric and so is invariant under
*similarity* transforms — rotation, reflection, translation, uniform scale — but
not under shear or anisotropic scaling.  Similarity transforms are exactly the
ambiguity class of an MDS configuration, so the guarantee covers the case this
package needs; ``residuals`` reports how far off the hull each point was, and
hence how much of the guarantee is doing the weaker, projection-based work.

The concrete case this was written for: adapters fine-tuned on mixtures of two
topics, anchored at the two pure endpoints.  The weight on the ``100% topic 0``
anchor is the geometry's own estimate of an adapter's mixing proportion — a
quantity the extraction → distance → embedding chain never saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.core.geometry import GeometryResult
from src.core.protocols import ModelID


# ── container ─────────────────────────────────────────────────────────────────

@dataclass
class SimplexProjection:
    """Barycentric coordinates of a model collection w.r.t. a set of anchors."""

    weights: np.ndarray          # (n, k), rows sum to 1
    model_ids: list[ModelID]
    anchor_ids: list[ModelID]
    residuals: np.ndarray        # (n,)
    taxonomy: str = ""
    method: str = ""
    clipped: bool = True

    def __post_init__(self) -> None:
        n, k = len(self.model_ids), len(self.anchor_ids)
        if self.weights.shape != (n, k):
            raise ValueError(
                f"weights shape {self.weights.shape} expected ({n}, {k})"
            )
        if self.residuals.shape != (n,):
            raise ValueError(
                f"residuals shape {self.residuals.shape} expected ({n},)"
            )

    def weight_for(self, model_id: ModelID) -> np.ndarray:
        """Barycentric coordinates of one model."""
        return self.weights[self.model_ids.index(model_id)]

    def anchor_column(self, anchor: int | ModelID = 0) -> np.ndarray:
        """The weight every model puts on one anchor, as a ``(n,)`` array."""
        idx = anchor if isinstance(anchor, int) else self.anchor_ids.index(anchor)
        return self.weights[:, idx]

    def save(self, path: Path | str) -> None:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_ids": self.model_ids,
            "anchor_ids": self.anchor_ids,
            "taxonomy": self.taxonomy,
            "method": self.method,
            "clipped": self.clipped,
        }
        meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        save_file(
            {
                "weights": np.ascontiguousarray(self.weights.astype(np.float32)),
                "residuals": np.ascontiguousarray(self.residuals.astype(np.float32)),
                "_meta_json": meta_bytes,
            },
            str(path / "simplex.safetensors"),
        )

    @classmethod
    def load(cls, path: Path | str) -> "SimplexProjection":
        from safetensors.numpy import load_file

        path = Path(path)
        tensors = load_file(str(path / "simplex.safetensors"))
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return cls(
            weights=tensors["weights"].astype(np.float64),
            residuals=tensors["residuals"].astype(np.float64),
            **meta,
        )


# ── projection ────────────────────────────────────────────────────────────────

def barycentric(
    geometry: GeometryResult,
    anchors: Sequence[ModelID],
    clip: bool = True,
) -> SimplexProjection:
    """Express every model in barycentric coordinates of the anchor simplex.

    For each point ``x`` this solves ``min_w ‖Aᵀw − x‖₂`` subject to ``Σw = 1``,
    where ``A`` holds the anchor coordinates.  The constraint is eliminated
    exactly by writing the point relative to the first anchor, so no penalty
    weight or tolerance is involved.

    Parameters
    ----------
    geometry:
        Any :class:`GeometryResult` — MDS, PCA, or otherwise — in any number of
        dimensions.
    anchors:
        Two or more model IDs present in *geometry*, in the order the weight
        columns should appear.
    clip:
        When ``True`` (default) the weights are additionally constrained to be
        non-negative, so every model lands *inside* the simplex and
        ``residuals`` measures the distance to the simplex itself.  With
        ``False`` weights may fall outside ``[0, 1]`` — meaning the model sits
        beyond the anchors along the simplex axis — and ``residuals`` measures
        the distance to the simplex's affine hull instead.

    Notes
    -----
    Anchors are fixed points: an anchor always receives a one-hot weight vector
    and zero residual, whatever the geometry looks like.

    A point with ``residual == 0`` lies in the anchors' affine hull, and its
    weights are invariant under any invertible affine map of the configuration.
    A point with a non-zero residual was projected onto that hull, and its
    weights are invariant only under similarity transforms — which still covers
    the rotation/reflection/translation/scale ambiguity of an MDS solution.
    """
    anchors = list(anchors)
    if len(anchors) < 2:
        raise ValueError(f"need at least 2 anchors, got {len(anchors)}")
    if len(set(anchors)) != len(anchors):
        raise ValueError(f"anchors must be distinct, got {anchors}")

    ids = list(geometry.model_ids)
    missing = [a for a in anchors if a not in ids]
    if missing:
        raise ValueError(
            f"anchors not present in geometry: {missing}. "
            f"Available (first 5): {ids[:5]}"
        )

    coords = np.asarray(geometry.coordinates, dtype=np.float64)
    k = len(anchors)
    anchor_idx = [ids.index(a) for a in anchors]
    A = coords[anchor_idx]                       # (k, d)

    origin = A[0]
    edges = (A[1:] - origin).T                   # (d, k-1)

    rank = np.linalg.matrix_rank(edges)
    if rank < k - 1:
        raise ValueError(
            f"anchors are affinely dependent (edge rank {rank}, need {k - 1}). "
            "Two anchors coincide, or three or more are collinear, so they do "
            "not span a simplex. Choose anchors that are genuinely spread out, "
            "or embed with more components via fit_geometry(n_components=...)."
        )

    rhs = (coords - origin).T                    # (d, n)
    z, *_ = np.linalg.lstsq(edges, rhs, rcond=None)   # (k-1, n)
    weights = np.empty((len(ids), k), dtype=np.float64)
    weights[:, 0] = 1.0 - z.sum(axis=0)
    weights[:, 1:] = z.T

    if clip:
        weights = _clip_to_simplex(A, coords, weights)

    fitted = weights @ A                         # (n, d)
    residuals = np.linalg.norm(coords - fitted, axis=1)

    return SimplexProjection(
        weights=weights,
        model_ids=ids,
        anchor_ids=anchors,
        residuals=residuals,
        taxonomy=geometry.taxonomy,
        method=geometry.method,
        clipped=clip,
    )


def _clip_to_simplex(
    A: np.ndarray, coords: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Re-solve with ``w >= 0`` for the points whose affine solution went negative.

    Uses NNLS on the system augmented with a heavily weighted ``Σw = 1`` row,
    which is the standard way to add an equality constraint to a non-negative
    least-squares solve.  The penalty is scaled to the anchor magnitudes so it
    dominates the fit regardless of units; the result is renormalised so the
    weights sum to exactly one.
    """
    from scipy.optimize import nnls

    needs_clip = np.any(weights < 0.0, axis=1)
    if not needs_clip.any():
        return weights

    penalty = 1e6 * max(float(np.abs(A).max()), 1.0)
    design = np.vstack([A.T, np.full((1, A.shape[0]), penalty)])   # (d+1, k)

    out = weights.copy()
    for i in np.flatnonzero(needs_clip):
        target = np.concatenate([coords[i], [penalty]])
        w, _ = nnls(design, target)
        total = w.sum()
        out[i] = w / total if total > 1e-12 else np.full(A.shape[0], 1.0 / A.shape[0])
    return out


# ── comparison of two projections ─────────────────────────────────────────────

@dataclass
class SimplexComparison:
    """How differently two geometries place models within the same simplex."""

    model_ids: list[ModelID]
    anchor_ids: list[ModelID]
    l1: np.ndarray                 # (n,)
    total_variation: np.ndarray    # (n,)  == l1 / 2
    l2: np.ndarray                 # (n,)
    per_anchor_spearman: np.ndarray  # (k,)
    weights_a: np.ndarray = field(repr=False)
    weights_b: np.ndarray = field(repr=False)

    @property
    def mean_total_variation(self) -> float:
        return float(self.total_variation.mean())

    @property
    def max_total_variation(self) -> float:
        return float(self.total_variation.max())

    def sorted_models(self) -> list[tuple[ModelID, float]]:
        """Models ranked by disagreement, largest first."""
        pairs = list(zip(self.model_ids, (float(v) for v in self.total_variation)))
        return sorted(pairs, key=lambda x: -x[1])

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"SimplexComparison(n_models={len(self.model_ids)}, "
            f"anchors={self.anchor_ids}, "
            f"mean_tv={self.mean_total_variation:.4f}, "
            f"max_tv={self.max_total_variation:.4f})"
        )


def compare_simplices(
    a: SimplexProjection,
    b: SimplexProjection,
) -> SimplexComparison:
    """Compare where two geometries place each model inside the same simplex.

    With anchors ``[100t0, 000t0]``, an adapter might land at ``[0.71, 0.29]``
    under the structural geometry and ``[0.68, 0.32]`` under the
    dataset-embedding one.  Per model this reports

    * ``l1`` — ``|0.71-0.68| + |0.29-0.32|`` = ``0.06``;
    * ``total_variation`` — half of that, ``0.03``, the usual bounded ``[0, 1]``
      measure of how far apart two weight vectors are;
    * ``l2`` — the same idea, Euclidean.

    Then per anchor column, across all models, a Spearman correlation: *do the
    two geometries order the models the same way along that axis?*

    Both projections must use the same anchors in the same order; the model sets
    are intersected automatically.
    """
    if a.anchor_ids != b.anchor_ids:
        raise ValueError(
            "both projections must use the same anchors in the same order; got "
            f"{a.anchor_ids} and {b.anchor_ids}"
        )

    ids, (wa, wb) = _match_projections(a, b)

    diff = wa - wb
    l1 = np.abs(diff).sum(axis=1)
    l2 = np.linalg.norm(diff, axis=1)

    from scipy.stats import spearmanr

    per_anchor = np.array(
        [
            float(spearmanr(wa[:, j], wb[:, j]).statistic) if len(ids) > 2 else float("nan")
            for j in range(wa.shape[1])
        ]
    )

    return SimplexComparison(
        model_ids=ids,
        anchor_ids=list(a.anchor_ids),
        l1=l1,
        total_variation=0.5 * l1,
        l2=l2,
        per_anchor_spearman=per_anchor,
        weights_a=wa,
        weights_b=wb,
    )


def _match_projections(
    a: SimplexProjection, b: SimplexProjection
) -> tuple[list[ModelID], tuple[np.ndarray, np.ndarray]]:
    """Intersect two projections' model sets, preserving *a*'s ordering."""
    common = set(a.model_ids) & set(b.model_ids)
    if not common:
        raise ValueError("the two projections have no models in common")
    order = [m for m in a.model_ids if m in common]
    ia = [a.model_ids.index(m) for m in order]
    ib = [b.model_ids.index(m) for m in order]
    return order, (a.weights[ia], b.weights[ib])


# ── validation against a known quantity ───────────────────────────────────────

@dataclass
class RecoveryResult:
    """Recovered vs. known values for an anchored quantity."""

    true: np.ndarray            # (n,) or (n, k)
    recovered: np.ndarray       # same shape
    pearson: np.ndarray         # (n_columns,)
    spearman: np.ndarray        # (n_columns,)
    residuals: np.ndarray       # (n,)
    model_ids: list[ModelID]
    columns: list[ModelID]
    mean_l1: float | None = None

    @property
    def r(self) -> float:
        """Pearson r, when a single column was compared."""
        if self.pearson.size != 1:
            raise ValueError(f"{self.pearson.size} columns compared; use .pearson")
        return float(self.pearson[0])

    @property
    def rho(self) -> float:
        """Spearman rho, when a single column was compared."""
        if self.spearman.size != 1:
            raise ValueError(f"{self.spearman.size} columns compared; use .spearman")
        return float(self.spearman[0])

    def pairs(self) -> list[tuple[ModelID, float, float]]:
        """``(model_id, true, recovered)`` triples, for a scatter plot."""
        if self.true.ndim != 1:
            raise ValueError("pairs() is for single-column comparisons; use .true/.recovered")
        return [
            (m, float(t), float(rec))
            for m, t, rec in zip(self.model_ids, self.true, self.recovered)
        ]

    def __repr__(self) -> str:  # pragma: no cover - display only
        head = (
            f"RecoveryResult(n_models={len(self.model_ids)}, "
            f"columns={self.columns}, "
        )
        if self.pearson.size == 1:
            return head + f"r={self.r:.4f}, rho={self.rho:.4f}, max_residual={float(self.residuals.max()):.4g})"
        return (
            head
            + f"pearson={np.round(self.pearson, 4).tolist()}, "
            + f"mean_l1={self.mean_l1:.4f})"
        )


def anchor_weight_vs_truth(
    projection: SimplexProjection,
    true_values: Mapping[ModelID, float] | Mapping[ModelID, Sequence[float]] | np.ndarray,
    anchor: int | ModelID = 0,
) -> RecoveryResult:
    """Check a geometry's recovered anchor weights against known values.

    This is the only analysis in the package that compares a geometry to
    something **external to the pipeline**.  Mantel, PROTEST and stress all
    compare one derived quantity to another, so they can only establish internal
    consistency; this one can say the geometry is *right*.

    The setup it was built for: each adapter was fine-tuned on a dataset with a
    known topic mixture, recorded in its name — a generating parameter, chosen
    before training and never seen by extraction, distance or embedding.  With
    the two pure endpoints as anchors, the barycentric weight on the ``100%``
    anchor is the geometry's own estimate of that proportion.  Correlating
    recovered against true asks whether the geometry recovered the thing that
    actually varied.

    Interpreting the result:

    * ``rho == 1`` — the models are ordered correctly along the mixing axis.
    * ``r ≈ 1`` with points on ``y = x`` — the *spacing* is right too, so
      geometric distance is linear in the mixture change.
    * ``r ≈ 1`` but bowed — monotone yet saturating; the taxonomy over-separates
      the pure cases.
    * low ``r`` — the geometry is not tracking the mixture at all.
    * large ``residuals`` — the models do not lie near the simplex to begin
      with, so the projection is a lossy summary and a high ``r`` should not be
      over-read.  This is why residuals come back alongside the correlations.

    Run it per taxonomy and the scores are directly comparable: *which level of
    abstraction best recovers the training mixture?*

    Parameters
    ----------
    true_values:
        Known values, supplied by the caller — nothing is parsed from model
        names here, so this works for any anchored quantity (mixing proportion,
        dataset size, LoRA rank, ...).  Either a mapping keyed by model ID, or
        an array aligned to ``projection.model_ids``.  Shape ``(n,)`` compares a
        single anchor column; shape ``(n, k)`` compares the full barycentric
        vector, which is the natural form when three or more topics were mixed —
        the ground truth is then itself a barycentric vector and no projection
        onto a single axis is needed.
    anchor:
        Which anchor column to compare, when *true_values* is one-dimensional.
        Accepts an index or a model ID.
    """
    true_arr, ids = _resolve_true_values(true_values, projection)
    keep = [projection.model_ids.index(m) for m in ids]
    residuals = projection.residuals[keep]

    if true_arr.ndim == 1:
        col = anchor if isinstance(anchor, int) else projection.anchor_ids.index(anchor)
        recovered = projection.weights[keep, col]
        columns = [projection.anchor_ids[col]]
        mean_l1 = None
        true_cols, rec_cols = true_arr[:, None], recovered[:, None]
    else:
        if true_arr.shape[1] != len(projection.anchor_ids):
            raise ValueError(
                f"true_values has {true_arr.shape[1]} columns but the projection "
                f"has {len(projection.anchor_ids)} anchors"
            )
        recovered = projection.weights[keep]
        columns = list(projection.anchor_ids)
        mean_l1 = float(np.abs(true_arr - recovered).sum(axis=1).mean())
        true_cols, rec_cols = true_arr, recovered

    from scipy.stats import pearsonr, spearmanr

    n_cols = true_cols.shape[1]
    pearson = np.full(n_cols, np.nan)
    spearman = np.full(n_cols, np.nan)
    if len(ids) >= 3:
        for j in range(n_cols):
            t, r_ = true_cols[:, j], rec_cols[:, j]
            # A constant column has no variance, so correlation is undefined.
            if np.ptp(t) > 1e-12 and np.ptp(r_) > 1e-12:
                pearson[j] = float(pearsonr(t, r_).statistic)
                spearman[j] = float(spearmanr(t, r_).statistic)

    return RecoveryResult(
        true=true_arr,
        recovered=recovered,
        pearson=pearson,
        spearman=spearman,
        residuals=residuals,
        model_ids=ids,
        columns=columns,
        mean_l1=mean_l1,
    )


def _resolve_true_values(
    true_values, projection: SimplexProjection
) -> tuple[np.ndarray, list[ModelID]]:
    """Normalise *true_values* to an array plus the model order it refers to."""
    if isinstance(true_values, Mapping):
        ids = [m for m in projection.model_ids if m in true_values]
        if not ids:
            raise ValueError(
                "no overlap between true_values keys and projection.model_ids"
            )
        arr = np.asarray([true_values[m] for m in ids], dtype=np.float64)
        return arr, ids

    arr = np.asarray(true_values, dtype=np.float64)
    if arr.shape[0] != len(projection.model_ids):
        raise ValueError(
            f"true_values has {arr.shape[0]} rows but the projection covers "
            f"{len(projection.model_ids)} models; pass a mapping keyed by model "
            "ID if the sets differ"
        )
    return arr, list(projection.model_ids)
