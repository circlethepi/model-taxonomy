"""Fitting a trained mixture adapter as a combination of the corner adapters.

Given a target adapter ``T`` trained on a mixture and ``k`` corner adapters each
trained on a single pure component, two different questions get asked:

1. **Fidelity.**  How close is ``Σ wₖ ΔWₖ`` — the corners combined at the *true*
   data proportions — to ``ΔW_T``?
2. **Recovery.**  Which coefficients fit best, and how far are they from the
   true proportions?

Both are closed form in the Gram.  Writing ``G_TT = <ΔW_T, ΔW_T>``,
``g = (<ΔW_T, ΔWₖ>)ₖ`` and ``Gc = (<ΔWᵢ, ΔWⱼ>)ᵢⱼ``:

    ‖ΔW_T − Σ xₖ ΔWₖ‖²  =  G_TT − 2 xᵀg + xᵀ Gc x

so the unconstrained minimizer is ``x* = Gc⁻¹ g`` and every reported quantity
follows without touching a weight tensor again.

**The primary coefficient estimate is x*/Σx*, not the simplex-constrained fit.**
That is an empirical finding, not a stylistic preference.  On the real
collection the unconstrained fit consistently sums to γ ≈ 0.73 rather than 1:
the trained mixture adapter has systematically *less* mass along the corner
directions than the corners themselves do.  Forcing Σw = 1 fights that shrinkage
and pushes the surplus onto the components that should be zero, roughly doubling
the coefficient error (mean L2 0.053 normalized vs 0.117 constrained).  So γ is
reported as its own scalar — it is a real property of the geometry — and the
direction is read off the normalized fit.  The constrained fit is still computed
and reported, since it is the answer to "best mixture *as a mixture*".
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = [
    "InterpolationFit",
    "fit_interpolation",
    "least_squares_weights",
    "random_simplex_baseline",
    "residual_sq",
    "simplex_weights",
]


def residual_sq(G_TT: float, g: np.ndarray, Gc: np.ndarray, w: np.ndarray) -> float:
    """‖ΔW_T − Σ wₖ ΔWₖ‖²_F, clipped at zero.

    Exact in the Gram; the clip only absorbs float error when the residual is
    genuinely ~0 (a corner fitting itself).
    """
    w = np.asarray(w, dtype=np.float64)
    return float(max(G_TT - 2.0 * (w @ g) + w @ Gc @ w, 0.0))


def least_squares_weights(Gc: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Unconstrained minimizer ``Gc⁻¹ g`` of the residual.

    Uses ``solve`` rather than forming an inverse.  ``Gc`` is a Gram matrix, so
    it is PSD, and singular only if the corner updates are linearly dependent —
    which would mean two corners span the same direction, worth an explicit
    error rather than a pseudo-inverse that quietly picks one of infinitely many
    answers.
    """
    try:
        return np.linalg.solve(Gc, g)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            f"Corner Gram is singular (cond={np.linalg.cond(Gc):.3e}); the "
            "corner updates are linearly dependent, so the coefficients are "
            "not identifiable."
        ) from e


def simplex_weights(Gc: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Minimizer subject to ``w ≥ 0`` and ``Σw = 1``.

    Returns ``(w, face)`` where *face* is the support of the solution.

    Solved exactly by enumerating the ``2ᵏ−1`` faces of the simplex rather than
    by iterative QP.  On each face the inequality constraints are inactive by
    construction, so the problem is an equality-constrained least squares with
    the KKT system

        [ Gs  1 ] [ x ]   [ gs ]
        [ 1ᵀ  0 ] [ λ ] = [ 1  ]

    The global optimum lies in the relative interior of exactly one face, so
    taking the best feasible solution over all of them is exact.  For k=3 that
    is 7 tiny solves — cheaper and more accurate than any iterative method.
    """
    k = Gc.shape[0]
    best_obj, best_w, best_face = np.inf, None, ()
    for size in range(1, k + 1):
        for face in combinations(range(k), size):
            idx = list(face)
            Gs, gs = Gc[np.ix_(idx, idx)], g[idx]
            m = len(idx)
            K = np.zeros((m + 1, m + 1))
            K[:m, :m] = Gs
            K[:m, m] = K[m, :m] = 1.0
            rhs = np.zeros(m + 1)
            rhs[:m] = gs
            rhs[m] = 1.0
            try:
                sol = np.linalg.solve(K, rhs)[:m]
            except np.linalg.LinAlgError:
                continue
            if np.any(sol < -1e-9):
                continue
            w = np.zeros(k)
            w[idx] = np.clip(sol, 0.0, None)
            obj = -2.0 * (w @ g) + w @ Gc @ w      # G_TT is constant, drop it
            if obj < best_obj:
                best_obj, best_w, best_face = obj, w, face
    if best_w is None:  # pragma: no cover - needs a degenerate Gram on every face
        raise np.linalg.LinAlgError("No feasible face; the corner Gram is degenerate.")
    return best_w, best_face


def random_simplex_baseline(
    true: np.ndarray, n_samples: int = 20000, seed: int = 0
) -> float:
    """Mean L2 distance from a uniformly random simplex point to *true*.

    The scale against which a coefficient error should be read: it is what
    guessing would score.  Without it "L2 = 0.05" is a number with no units.
    """
    rng = np.random.default_rng(seed)
    pts = rng.dirichlet(np.ones(len(true)), n_samples)
    return float(np.linalg.norm(pts - np.asarray(true, dtype=np.float64), axis=1).mean())


@dataclass
class InterpolationFit:
    """Every measured quantity for one target adapter."""

    target: str
    corners: list[str]
    true: np.ndarray                 # (k,) true data proportions, sum 1
    ols: np.ndarray                  # (k,) unconstrained fit
    ols_normalized: np.ndarray       # (k,) ols / gamma  ← primary estimate
    simplex: np.ndarray              # (k,) constrained fit
    face: tuple[int, ...]
    gamma: float                     # sum(ols): the shrinkage
    norm_sq: float                   # ‖ΔW_T‖²
    residual_true: float
    residual_ols: float
    residual_simplex: float
    rel_true: float                  # residual / ‖ΔW_T‖
    rel_ols: float
    rel_simplex: float
    cosine: float                    # cos(ΔW_T, Σ wₖ ΔWₖ) at the TRUE weights
    coeff_l2: float                  # ‖ols_normalized − true‖₂
    coeff_l1: float
    coeff_l2_simplex: float

    @property
    def is_corner(self) -> bool:
        """True when this target is itself one of the corners."""
        return self.target in self.corners


def fit_interpolation(
    gram,
    target: str,
    corners: list[str],
    true: np.ndarray,
) -> InterpolationFit:
    """Fit one target against the corner adapters.

    *gram* is a :class:`~src.interp.gram.LoRAGram`; *true* is the target's
    actual training proportions, in corner order.
    """
    t = gram.index(target)
    ci = [gram.index(c) for c in corners]
    G = gram.total
    G_TT = float(G[t, t])
    g = G[t, ci].astype(np.float64)
    Gc = G[np.ix_(ci, ci)].astype(np.float64)

    true = np.asarray(true, dtype=np.float64)
    if true.shape != (len(corners),):
        raise ValueError(f"true has shape {true.shape}, expected ({len(corners)},)")

    ols = least_squares_weights(Gc, g)
    gamma = float(ols.sum())
    # gamma is ~0.73 in practice and cannot be 0 unless the target is orthogonal
    # to the entire corner span, in which case the direction is meaningless.
    ols_norm = ols / gamma if abs(gamma) > 1e-12 else np.full_like(ols, np.nan)
    simplex, face = simplex_weights(Gc, g)

    r_true = residual_sq(G_TT, g, Gc, true)
    r_ols = residual_sq(G_TT, g, Gc, ols)
    r_simplex = residual_sq(G_TT, g, Gc, simplex)
    norm = np.sqrt(G_TT)

    interp_norm_sq = float(true @ Gc @ true)
    cosine = (
        float((true @ g) / np.sqrt(G_TT * interp_norm_sq))
        if G_TT > 0 and interp_norm_sq > 0
        else float("nan")
    )

    return InterpolationFit(
        target=target,
        corners=list(corners),
        true=true,
        ols=ols,
        ols_normalized=ols_norm,
        simplex=simplex,
        face=face,
        gamma=gamma,
        norm_sq=G_TT,
        residual_true=float(np.sqrt(r_true)),
        residual_ols=float(np.sqrt(r_ols)),
        residual_simplex=float(np.sqrt(r_simplex)),
        rel_true=float(np.sqrt(r_true) / norm),
        rel_ols=float(np.sqrt(r_ols) / norm),
        rel_simplex=float(np.sqrt(r_simplex) / norm),
        cosine=cosine,
        coeff_l2=float(np.linalg.norm(ols_norm - true)),
        coeff_l1=float(np.abs(ols_norm - true).sum()),
        coeff_l2_simplex=float(np.linalg.norm(simplex - true)),
    )
