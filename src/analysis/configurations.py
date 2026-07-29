"""Level 2 — comparison of the point configurations themselves.

Where :mod:`src.analysis.matrices` compares distance matrices, this module
compares the coordinates an embedding produced.  All comparisons are Procrustes
based: each configuration is translated to the origin, optionally scaled to unit
Frobenius norm, and then optimally rotated onto the other.  The result therefore
depends on *shape alone* — not on the arbitrary orientation, reflection or scale
that MDS happened to pick, none of which carry meaning.

Two geometries need not share dimensionality; the lower-dimensional one is
padded with zero columns, which embeds it in the larger space without changing
any of its internal distances.

``disparity`` follows SciPy's convention — on unit-normalised inputs it is
``1 - (Σσ)²`` and therefore lies in ``[0, 1]``, 0 meaning identical shape.
``scripts/check_analysis.py::t_procrustes_vs_scipy`` pins it there against
:func:`scipy.spatial.procrustes` so the two cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.core.geometry import GeometryResult
from src.core.protocols import ModelID

from .matrices import match_models


# ── low-level Procrustes machinery ────────────────────────────────────────────

def _pad_to(a: np.ndarray, d: int) -> np.ndarray:
    """Right-pad a coordinate array with zero columns to width *d*."""
    if a.shape[1] == d:
        return a
    return np.hstack([a, np.zeros((a.shape[0], d - a.shape[1]), dtype=a.dtype)])


def _standardize(a: np.ndarray, scaling: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Center (and optionally unit-normalise) a configuration.

    Returns the standardized array, the centroid that was removed, and the
    Frobenius norm that was divided out (1.0 when ``scaling=False``).
    """
    centroid = a.mean(axis=0)
    out = a - centroid
    norm = float(np.linalg.norm(out))
    if not scaling:
        return out, centroid, 1.0
    if norm < 1e-12:
        raise ValueError("degenerate configuration: all points coincide")
    return out / norm, centroid, norm


# Why this is hand-written rather than a call into SciPy.
#
# SciPy offers two entry points, and neither covers what this module needs:
#
#   scipy.spatial.procrustes(data1, data2) -> (mtx1, mtx2, disparity)
#       Takes no options at all.  It never returns the rotation or the scale, so
#       `ProcrustesResult` could not expose them and `align_to_reference` would
#       have no transform to apply; it always standardises, so `scaling=False`
#       is unreachable; it always permits reflection; it rejects inputs of
#       differing shape ("Input matrices must be of same shape"), which rules
#       out comparing a 2-D embedding against a 3-D one; and it is pairwise, so
#       fitting N geometries onto one frame is not expressible.
#
#   scipy.linalg.orthogonal_procrustes(A, B) -> (R, sum of singular values)
#       Exactly what the `reflection=True` branch below computes — verified
#       identical to the bit, in both R and the scale.  It has no notion of
#       forbidding reflection, at this or any other level of the library.
#
# So the only genuinely custom piece is the det(R) = +1 correction.  It is not
# split out to delegate the other half to SciPy because that branch needs the
# singular values anyway: one SVD serves both cases here, whereas delegating
# would leave two independent implementations of the same quantity.
#
# `scaling=False` is load-bearing rather than a convenience.  Under unit-norm
# standardisation, displacing a single model redistributes the residual across
# the whole configuration, which is enough to make `per_point_residuals` blame
# the wrong model.  Isolating one point's movement requires switching scaling
# off, and SciPy's interface cannot.

def _optimal_rotation(
    source: np.ndarray, target: np.ndarray, reflection: bool
) -> tuple[np.ndarray, float]:
    """Orthogonal ``R`` and scale ``s`` minimising ``‖s · source @ R - target‖_F``.

    ``s`` is the sum of the singular values of ``sourceᵀ target``; for
    unit-norm inputs the resulting disparity is ``1 - s²``.

    With ``reflection=True`` this is :func:`scipy.linalg.orthogonal_procrustes`.
    """
    u, sv, vt = np.linalg.svd(source.T @ target)
    if not reflection and np.linalg.det(u @ vt) < 0:
        # Flip the least-important singular direction to force det(R) = +1;
        # that direction's contribution to the scale flips sign with it.
        u = u.copy()
        u[:, -1] *= -1.0
        sv = sv.copy()
        sv[-1] *= -1.0
    return u @ vt, float(sv.sum())


def _superimpose(
    target: np.ndarray, source: np.ndarray, reflection: bool, apply_scale: bool
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit *source* onto *target*; return (aligned, rotation, scale, disparity)."""
    rotation, scale = _optimal_rotation(source, target, reflection)
    if not apply_scale:
        scale = 1.0
    aligned = scale * (source @ rotation)
    disparity = float(np.sum((target - aligned) ** 2)) # sum of squares distances
    return aligned, rotation, scale, disparity


@dataclass
class ProcrustesResult:
    """Superposition of two point configurations, and the map that produced it.

    ``disparity`` scores the fit; the remaining fields *are* the fit, and together
    they reconstruct it exactly.  The centroid and norm matter as much as the
    rotation: a superposition standardises both configurations before rotating
    one, so without the shift and scale that were divided out, the map can be
    inspected but not applied to any point that was not in the original fit.
    :meth:`transform` is what they buy.
    """

    disparity: float
    aligned_a: GeometryResult
    aligned_b: GeometryResult
    rotation: np.ndarray
    scale: float
    model_ids: list[ModelID]
    #: Centroids removed by ``_standardize`` before fitting.
    centroid_a: np.ndarray | None = None
    centroid_b: np.ndarray | None = None
    #: Frobenius norms divided out; 1.0 when ``scaling=False``.
    norm_a: float = 1.0
    norm_b: float = 1.0
    #: The options the fit was made under, recorded so the map is interpretable.
    scaling: bool = True
    reflection: bool = True

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ProcrustesResult(disparity={self.disparity:.6f}, "
            f"n_models={len(self.model_ids)}, dim={self.rotation.shape[0]})"
        )

    # ── applying the fitted map ───────────────────────────────────────────────

    def transform(self, coords: np.ndarray, which: str = "b") -> np.ndarray:
        """Put coordinates into the superimposed frame using the stored map.

        For ``which="b"`` this is the full fitted map,
        ``((X - centroid_b) / norm_b) @ rotation * scale``.  For ``which="a"`` it
        is only ``(X - centroid_a) / norm_a``, because *a* is the target frame —
        it was standardised but never rotated.

        The point of exposing this rather than only ``aligned_b`` is that it
        applies to coordinates that were **not** part of the fit: a newly trained
        adapter, a held-out mixture, or the same models embedded at a different
        sample size.  Comparing maps fitted on different collections by applying
        them to one common set of points is what makes them comparable at all.

        *coords* is ``(n, d)`` and is zero-padded to the fitted width if narrower,
        which embeds it without changing any of its internal distances.
        """
        if which not in ("a", "b"):
            raise ValueError(f"which must be 'a' or 'b', got {which!r}")
        centroid = self.centroid_a if which == "a" else self.centroid_b
        if centroid is None:
            raise ValueError(
                "this ProcrustesResult predates the stored centroid/norm and so "
                "cannot be applied to new points; recompute it with "
                "procrustes_compare()."
            )

        width = self.rotation.shape[0]
        x = np.asarray(coords, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] > width:
            raise ValueError(
                f"coords have {x.shape[1]} dimensions but the map was fitted in "
                f"{width}"
            )
        x = _pad_to(x, width)

        norm = self.norm_a if which == "a" else self.norm_b
        centered = (x - centroid) / norm
        if which == "a":
            return centered
        return self.scale * (centered @ self.rotation)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Persist the map and both aligned configurations."""
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "disparity": self.disparity,
            "scale": self.scale,
            "model_ids": self.model_ids,
            "norm_a": self.norm_a,
            "norm_b": self.norm_b,
            "scaling": self.scaling,
            "reflection": self.reflection,
        }
        width = self.rotation.shape[0]
        tensors = {
            "rotation": np.ascontiguousarray(self.rotation.astype(np.float64)),
            "centroid_a": np.ascontiguousarray(
                (self.centroid_a if self.centroid_a is not None else np.zeros(width)
                 ).astype(np.float64)
            ),
            "centroid_b": np.ascontiguousarray(
                (self.centroid_b if self.centroid_b is not None else np.zeros(width)
                 ).astype(np.float64)
            ),
            "_meta_json": np.frombuffer(
                json.dumps(meta).encode("utf-8"), dtype=np.uint8
            ),
        }
        save_file(tensors, str(path / "procrustes.safetensors"))
        self.aligned_a.save(path / "aligned_a")
        self.aligned_b.save(path / "aligned_b")

    @classmethod
    def load(cls, path: Path | str) -> "ProcrustesResult":
        from safetensors.numpy import load_file

        path = Path(path)
        tensors = load_file(str(path / "procrustes.safetensors"))
        meta = json.loads(tensors["_meta_json"].tobytes().decode("utf-8"))
        return cls(
            aligned_a=GeometryResult.load(path / "aligned_a"),
            aligned_b=GeometryResult.load(path / "aligned_b"),
            rotation=tensors["rotation"],
            centroid_a=tensors["centroid_a"],
            centroid_b=tensors["centroid_b"],
            **meta,
        )


def procrustes_compare(
    geom_a: GeometryResult,
    geom_b: GeometryResult,
    scaling: bool = True,
    reflection: bool = True,
    key: Callable[[ModelID], str] | None = None,
) -> ProcrustesResult:
    """Superimpose *geom_b* onto *geom_a* and report the leftover mismatch.

    Parameters
    ----------
    scaling:
        When ``True`` (default) both configurations are scaled to unit Frobenius
        norm and an optimal scale factor is fitted, making the comparison
        scale-free and putting ``disparity`` on a ``[0, 1]`` scale where 0 means
        identical shape.  This matches :func:`scipy.spatial.procrustes`.  With
        ``False`` the configurations are only centered, no scale is fitted, and
        ``disparity`` is in raw squared coordinate units.
    reflection:
        Whether a reflection is permitted in addition to rotation.  MDS output
        is only defined up to reflection, so the default ``True`` is almost
        always what you want, and no caller in this repo sets it otherwise.
        ``False`` is for configurations where chirality is meaningful — where a
        mirrored arrangement should count as a genuine disagreement rather than
        the same shape seen the other way round.

    The returned ``aligned_a`` / ``aligned_b`` are new :class:`GeometryResult`
    objects sharing one frame, ready to plot on the same axes.

    *key* is passed to :func:`~src.analysis.matrices.match_models` to reconcile
    differing identifier schemes across taxonomy levels.
    """
    ids, (a_raw, b_raw) = match_models(geom_a, geom_b, key=key)
    if len(ids) < 2:
        raise ValueError(f"Procrustes needs at least 2 common models, got {len(ids)}")

    d = max(a_raw.shape[1], b_raw.shape[1])
    # The centroid and norm are kept rather than dropped: without them the fitted
    # map cannot be applied to any point outside this call. See
    # ProcrustesResult.transform.
    a_std, centroid_a, norm_a = _standardize(_pad_to(a_raw, d), scaling)
    b_std, centroid_b, norm_b = _standardize(_pad_to(b_raw, d), scaling)

    b_aligned, rotation, scale, disparity = _superimpose(a_std, b_std, reflection, scaling)

    return ProcrustesResult(
        disparity=disparity,
        aligned_a=_as_geometry(a_std, ids, geom_a, "procrustes"),
        aligned_b=_as_geometry(b_aligned, ids, geom_b, "procrustes"),
        rotation=rotation,
        scale=scale,
        model_ids=ids,
        centroid_a=centroid_a,
        centroid_b=centroid_b,
        norm_a=norm_a,
        norm_b=norm_b,
        scaling=scaling,
        reflection=reflection,
    )


def _as_geometry(
    coords: np.ndarray,
    ids: list[ModelID],
    source: GeometryResult,
    tag: str,
) -> GeometryResult:
    meta = dict(source.metadata)
    meta["aligned_by"] = tag
    meta["source_method"] = source.method
    return GeometryResult(
        coordinates=coords.astype(np.float32),
        model_ids=list(ids),
        method=source.method,
        taxonomy=source.taxonomy,
        n_components=coords.shape[1],
        stress=source.stress,
        metadata=meta,
    )


def per_point_residuals(result: ProcrustesResult) -> np.ndarray:
    """Per-model distance between the two superimposed positions.

    The *which models disagree* diagnostic.  A low overall ``disparity`` paired
    with one large residual means the two taxonomies agree about everything
    except that one model — which the single summary number hides entirely.

    Returns a ``(n_models,)`` array in ``result.model_ids`` order.
    """
    a = np.asarray(result.aligned_a.coordinates, dtype=np.float64)
    b = np.asarray(result.aligned_b.coordinates, dtype=np.float64)
    return np.linalg.norm(a - b, axis=1)


@dataclass
class ProtestResult:
    """Outcome of a PROTEST permutation test between two configurations."""

    disparity: float
    p_value: float
    n_permutations: int
    n_models: int
    null: np.ndarray = field(repr=False)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ProtestResult(disparity={self.disparity:.6f}, p_value={self.p_value:.4g}, "
            f"n_models={self.n_models}, n_permutations={self.n_permutations})"
        )


def protest(
    geom_a: GeometryResult,
    geom_b: GeometryResult,
    n_permutations: int = 9999,
    random_state: int | None = 0,
    scaling: bool = True,
    reflection: bool = True,
    key: Callable[[ModelID], str] | None = None,
) -> ProtestResult:
    """Permutation test on the Procrustes disparity (PROTEST).

    The configuration-level counterpart to
    :func:`src.analysis.matrices.mantel_test`, and built on the same null:
    permute which model is which in one configuration, re-superimpose, and see
    how often random labelling fits as well as the real correspondence.  Each
    configuration's internal shape is untouched.

    Because low disparity means good agreement, the p-value counts
    ``P(null <= observed)``; it is bounded below by ``1 / (n_permutations + 1)``.
    """
    ids, (a_raw, b_raw) = match_models(geom_a, geom_b, key=key)
    n = len(ids)
    if n < 3:
        raise ValueError(f"PROTEST needs at least 3 common models, got {n}")

    d = max(a_raw.shape[1], b_raw.shape[1])
    a_std, _, _ = _standardize(_pad_to(a_raw, d), scaling)
    b_std, _, _ = _standardize(_pad_to(b_raw, d), scaling)

    def _disparity(b: np.ndarray) -> float:
        return _superimpose(a_std, b, reflection, scaling)[3]

    observed = _disparity(b_std)

    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        null[i] = _disparity(b_std[rng.permutation(n)])

    p_value = float((np.sum(null <= observed) + 1) / (n_permutations + 1))
    return ProtestResult(
        disparity=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        n_models=n,
        null=null,
    )


def align_to_reference(
    geometries: Sequence[GeometryResult],
    reference: GeometryResult | None = None,
    scaling: bool = True,
    reflection: bool = True,
    key: Callable[[ModelID], str] | None = None,
) -> list[GeometryResult]:
    """Superimpose several configurations onto one common frame.

    Use this before plotting geometries side by side, or before
    :func:`point_dispersion`.  It replaces hand-tuned flips and rotations such
    as :func:`src.notebook.utils.transform_geometry`, which has to be re-guessed
    for every new figure.

    *reference* defaults to the first geometry.  All returned geometries —
    including the one matching the reference — are expressed in the reference's
    standardized frame, restricted to the models common to every input.
    """
    geoms = list(geometries)
    if not geoms:
        raise ValueError("align_to_reference needs at least one geometry")
    ref = geoms[0] if reference is None else reference

    ids, arrays = match_models(ref, *geoms, key=key)
    ref_raw, others = arrays[0], arrays[1:]

    d = max(a.shape[1] for a in arrays)
    ref_std, _, _ = _standardize(_pad_to(ref_raw, d), scaling)

    out: list[GeometryResult] = []
    for geom, raw in zip(geoms, others):
        std, _, _ = _standardize(_pad_to(raw, d), scaling)
        aligned, _, _, _ = _superimpose(ref_std, std, reflection, scaling)
        out.append(_as_geometry(aligned, ids, geom, "align_to_reference"))
    return out


@dataclass
class DispersionResult:
    """How stably each model is placed across a set of configurations."""

    per_model: np.ndarray
    model_ids: list[ModelID]
    mean_disparity: float
    n_geometries: int

    def sorted_models(self) -> list[tuple[ModelID, float]]:
        """Models ranked from least to most stable."""
        pairs = list(zip(self.model_ids, (float(v) for v in self.per_model)))
        return sorted(pairs, key=lambda x: x[1])

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"DispersionResult(n_geometries={self.n_geometries}, "
            f"n_models={len(self.model_ids)}, "
            f"mean_disparity={self.mean_disparity:.6f}, "
            f"max_dispersion={float(self.per_model.max()):.6f})"
        )


def point_dispersion(
    geometries: Sequence[GeometryResult],
    reference: GeometryResult | None = None,
    scaling: bool = True,
    reflection: bool = True,
    key: Callable[[ModelID], str] | None = None,
) -> DispersionResult:
    """Per-model positional spread across several configurations.

    Superimposes every geometry onto a common frame, after which each model has
    one point per geometry.  Its dispersion is the RMS distance of those points
    from their own centroid: *which models sit in a stable place across seeds,
    and which jump around?*

    ``mean_disparity`` is the mean pairwise Procrustes disparity over the set —
    a single number for how much the configurations agree overall.
    """
    geoms = list(geometries)
    if len(geoms) < 2:
        raise ValueError(f"point_dispersion needs at least 2 geometries, got {len(geoms)}")

    aligned = align_to_reference(geoms, reference, scaling, reflection, key=key)
    stack = np.stack([np.asarray(g.coordinates, dtype=np.float64) for g in aligned])

    centroids = stack.mean(axis=0)
    per_model = np.sqrt(((stack - centroids) ** 2).sum(axis=2).mean(axis=0))

    disparities = [
        float(np.sum((stack[i] - stack[j]) ** 2))
        for i in range(len(stack))
        for j in range(i + 1, len(stack))
    ]

    return DispersionResult(
        per_model=per_model,
        model_ids=list(aligned[0].model_ids),
        mean_disparity=float(np.mean(disparities)) if disparities else 0.0,
        n_geometries=len(geoms),
    )
