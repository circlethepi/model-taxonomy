"""Analysis layer over distance matrices and model geometries.

Three levels of comparison, all built on the containers in :mod:`src.core`:

* :mod:`~src.analysis.matrices` — distance matrices, pair by pair.
* :mod:`~src.analysis.configurations` — the embedded point configurations.
* :mod:`~src.analysis.simplex` — positions within a simplex of anchor models.

plus :mod:`~src.analysis.quality` for how faithfully an embedding represents the
distances it came from, and :mod:`~src.analysis.bridge` for turning raw LoRA
weights into the same typed containers so notebook work and pipeline results can
be analysed with one set of tools.
"""

from .bridge import (
    as_distance_matrix,
    fit_geometry,
    lora_distance_matrix,
    save_collection,
)
from .configurations import (
    DispersionResult,
    ProcrustesResult,
    ProtestResult,
    align_to_reference,
    per_point_residuals,
    point_dispersion,
    procrustes_compare,
    protest,
)
from .matrices import (
    MantelResult,
    correlation_table,
    mantel_test,
    match_models,
    matrix_correlation,
    offdiag,
)
from .quality import kruskal_stress, shepard
from .simplex import (
    RecoveryResult,
    SimplexComparison,
    SimplexProjection,
    anchor_weight_vs_truth,
    barycentric,
    compare_simplices,
)

__all__ = [
    # bridge
    "as_distance_matrix",
    "lora_distance_matrix",
    "fit_geometry",
    "save_collection",
    # matrices
    "match_models",
    "offdiag",
    "matrix_correlation",
    "mantel_test",
    "MantelResult",
    "correlation_table",
    # configurations
    "procrustes_compare",
    "ProcrustesResult",
    "per_point_residuals",
    "protest",
    "ProtestResult",
    "align_to_reference",
    "point_dispersion",
    "DispersionResult",
    # quality
    "kruskal_stress",
    "shepard",
    # simplex
    "SimplexProjection",
    "barycentric",
    "compare_simplices",
    "SimplexComparison",
    "anchor_weight_vs_truth",
    "RecoveryResult",
]
