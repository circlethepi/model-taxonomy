"""Embedding fidelity — how well a geometry represents its own distance matrix.

Distinct from :mod:`src.analysis.configurations`, which compares two geometries
to each other.  The question here is narrower and comes first: before reading
anything off a 2-D picture, how much did the embedding distort?
"""

from __future__ import annotations

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult

from .matrices import match_models, offdiag


def _paired_distances(
    dm: DistanceMatrix, geometry: GeometryResult
) -> tuple[np.ndarray, np.ndarray]:
    """Return (original, embedded) off-diagonal distance vectors, model-matched."""
    _, (dist, coords) = match_models(dm, geometry)
    original = offdiag(dist)

    from scipy.spatial.distance import pdist

    embedded = pdist(coords)
    return original, embedded


def kruskal_stress(dm: DistanceMatrix, geometry: GeometryResult) -> float:
    """Kruskal stress-1 of *geometry* against the distances it came from.

    ``sqrt(Σ (d_ij - ẟ_ij)² / Σ d_ij²)`` over all model pairs, where ``d_ij`` is
    the original distance and ``ẟ_ij`` the distance between the same two points
    in the embedding.  Lower is better; 0 is a perfect embedding.

    Recomputing it here rather than reading :attr:`GeometryResult.stress` is
    what makes methods comparable — only :class:`~src.geometry_methods.mds.MDSGeometry`
    populates that attribute, so PCA and UMAP results carry ``None`` and cannot
    otherwise be scored on the same axis.

    Rough reading (Kruskal's own guidance, for ordinal MDS): <0.05 excellent,
    <0.10 good, <0.20 fair, above that the picture is not trustworthy.
    """
    original, embedded = _paired_distances(dm, geometry)
    denom = float(np.sum(original**2))
    if denom < 1e-24:
        return 0.0
    return float(np.sqrt(np.sum((original - embedded) ** 2) / denom))


def shepard(
    dm: DistanceMatrix, geometry: GeometryResult
) -> tuple[np.ndarray, np.ndarray]:
    """Data for a Shepard diagram: original vs embedded distance, per model pair.

    Returns ``(original, embedded)``, each of length ``n(n-1)/2`` in
    :func:`scipy.spatial.distance.squareform` order.  Plot them as a scatter
    with *original* on x.

    How to read it.  A faithful metric embedding puts every point on the line
    ``y = x``.  Vertical spread is distortion.  A systematic bend below the line
    at large x means the embedding is compressing big distances to fit the
    dimensions available — the classic symptom of forcing high-dimensional
    structure into two dimensions.  Individual points far off the trend name the
    specific model pairs the picture misrepresents.

    This is the per-pair detail behind the single :func:`kruskal_stress` number,
    and it is worth looking at before trusting any 2-D scatter.
    """
    return _paired_distances(dm, geometry)
