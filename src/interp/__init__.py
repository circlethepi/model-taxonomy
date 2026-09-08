"""Interpolation of LoRA adapters between simplex corners.

Asks how well an adapter trained on a mixture of dataset components is
reproduced by combining the adapters trained on the pure components, and which
coefficients fit best.  Everything reduces to the Gram matrix of effective
updates ``ΔW = (alpha/r)·B@A``, computed in rank space so the dense ΔW (1.3e9
parameters per adapter for Qwen3.5-4B, 10.4 GB at float64) is never formed.

See ``src/interp/notes/lora_interpolation.md`` for the derivation.
"""

from src.interp.fitting import (
    InterpolationFit,
    fit_interpolation,
    least_squares_weights,
    random_simplex_baseline,
    residual_sq,
    simplex_weights,
)
from src.interp.geometry import (
    CoefficientGeometry,
    CoefficientGeometryReport,
    coefficient_distance_matrix,
    compare_coefficients,
    compare_from_fits,
    compare_from_report,
)
from src.interp.gram import LoRAGram, compute_gram, load_or_compute_gram
from src.interp.loading import (
    AdapterRef,
    adapter_scale,
    discover_adapters,
    load_adapter,
    mixture_proportions,
)
from src.interp.report import (
    InterpolationReport,
    build_report,
    detect_corners,
    load_fits,
)

__all__ = [
    "AdapterRef",
    "CoefficientGeometry",
    "CoefficientGeometryReport",
    "InterpolationFit",
    "InterpolationReport",
    "LoRAGram",
    "adapter_scale",
    "build_report",
    "coefficient_distance_matrix",
    "compare_coefficients",
    "compare_from_fits",
    "compare_from_report",
    "compute_gram",
    "detect_corners",
    "discover_adapters",
    "fit_interpolation",
    "least_squares_weights",
    "load_adapter",
    "load_fits",
    "load_or_compute_gram",
    "mixture_proportions",
    "random_simplex_baseline",
    "residual_sq",
    "simplex_weights",
]
