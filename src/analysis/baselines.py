"""Uninformed baselines: what a taxonomy that learned nothing would score.

`dcor_vs_truth` and `disparity_vs_truth` are reported with no reference point,
so a reader cannot tell a real recovery from a number any arrangement of points
would have produced.  This module supplies that reference.

Terminology, fixed in ``docs/notes/chance_baselines.md`` and used throughout:

**Uninformed baseline**
    The reference level a taxonomy that learned nothing would score.
**Uninformed band**
    Its 5-95 interval -- drawn as a shaded region with a dashed centre line.
**Structure null**
    Replace the taxonomy with a structureless configuration, keep the truth.
    *Is this level doing anything?*  The only null implemented here.
**Label null**
    Keep both real geometries, permute which model is which.  Not implemented
    here; :func:`src.analysis.configurations.protest` and
    :func:`src.analysis.matrices.dcor_test` are the pieces it is built from.

Three routes to the same structure null, in decreasing generality:

``*_null_direct``
    Monte Carlo the whole scoring path -- draw ``X``, standardise, superimpose
    (or U-centre), read the score.  Correct for every generator and every ``n``,
    and slow.  The other two are checked against this one.
``disparity_null_analytic``
    Exact for the Gaussian generator, and cheap: all the ``n``-dependence
    isolates into one chi-square draw.  See the derivation below.
``disparity_null_approx``
    The same expression with the denominator replaced by its mean.  A **large-n
    approximation only** -- see its docstring for where it fails and why.

One caveat carried by all three, and repeated in every figure that draws a
band: the structure null is scored **directly as a configuration**, never
through MDS.  The real disparity is MDS-mediated and therefore handicapped in a
way the baseline is not, so the baseline is mildly optimistic.  Scoring the null
directly is also exactly what buys the closed form -- an MDS step inside the
loop would destroy it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .ground_truth import simplex_vertices

__all__ = [
    "BAND_Q",
    "GENERATORS",
    "band_quantiles",
    "dcor_band_from_c",
    "dcor_null_direct",
    "disparity_null_analytic",
    "disparity_null_approx",
    "disparity_null_direct",
    "draw_configuration",
    "fit_dcor_c",
    "table_key",
    "truth_singular_values",
    "uninformed_band",
]

#: Smallest ``n`` at which :func:`disparity_null_approx` is within 0.01 of the
#: exact form on every band edge.  Measured, not assumed -- and measured to be
#: the same at ``d=2`` and ``d=3``, which is why it is one number.  Below it the
#: error is entirely in the *lower* edge and grows fast: 0.02 at n=20, 0.09 at
#: n=10, 0.52 at n=5, where the approximation returns a negative disparity.
APPROX_MIN_N = 30

#: The band's lower and upper quantiles.  5-95, so the band is what an
#: uninformed taxonomy lands inside nine times in ten.
BAND_Q = (5.0, 95.0)

_N_MC = 20_000


# ── generators ────────────────────────────────────────────────────────────────

def _gaussian(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """iid N(0, I) in R^(k-1).  Reads as: no geometry at all."""
    return rng.standard_normal((n, k - 1))


def _dirichlet(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Dirichlet(1) weights on the regular simplex.

    Reads as: recovered *a* valid recipe, just not this one.  Dirichlet(1) is
    the uniform distribution over the simplex, so there is deliberately no
    separate "uniform" generator -- it would be this one written twice.

    Goes through :func:`~src.analysis.ground_truth.simplex_vertices` rather than
    reimplementing the barycentric map, so a generated configuration lives in
    the same coordinates as a real ground truth.
    """
    return rng.dirichlet(np.ones(k), size=n) @ simplex_vertices(k)


#: Generator name -> ``(n, k, rng) -> (n, k-1) coordinates``.
GENERATORS: dict[str, Callable[[int, int, np.random.Generator], np.ndarray]] = {
    "gaussian": _gaussian,
    "dirichlet": _dirichlet,
}


def draw_configuration(
    generator: str, n: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """One structureless configuration of *n* points, ``(n, k-1)``."""
    try:
        fn = GENERATORS[generator]
    except KeyError:
        raise ValueError(
            f"unknown generator {generator!r}; have {sorted(GENERATORS)}"
        ) from None
    return fn(n, k, rng)


# ── the truth's shape ─────────────────────────────────────────────────────────

def truth_singular_values(coords: np.ndarray) -> np.ndarray:
    """The standardised truth's singular values ``L``, with ``sum(l^2) == 1``.

    Centre, divide by the Frobenius norm, take the singular values -- the same
    standardisation :func:`src.analysis.configurations._standardize` applies
    before superimposing, which is what makes ``L`` the right thing to feed the
    analytic form.  ``uninformed lambda matches standardize`` pins the two
    against each other rather than trusting the reading.

    Trailing zeros are kept: a truth that is rank-deficient in ``d`` dimensions
    has a zero singular value, and dropping it would silently change ``d``.
    """
    a = np.asarray(coords, dtype=np.float64)
    a = a - a.mean(axis=0)
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError("degenerate truth: all points coincide")
    return np.linalg.svd(a / norm, compute_uv=False)


# ── the disparity's structure null ────────────────────────────────────────────

def _joint_sample(
    lam: np.ndarray, n_mc: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo the pair ``(||L Z||_*^2, ||Z||_F^2)`` over d x d Gaussian ``Z``.

    Carries no ``n`` -- that is the whole point.  ``d`` is 2 or 3, so this is
    milliseconds even at 20k draws.
    """
    d = len(lam)
    z = rng.standard_normal((n_mc, d, d))
    lz = lam[None, :, None] * z
    nuclear = np.linalg.svd(lz, compute_uv=False).sum(axis=1)
    return nuclear ** 2, (z ** 2).sum(axis=(1, 2))


def _chi2_df(n: int, d: int) -> int:
    df = (n - 1 - d) * d
    if df <= 0:
        raise ValueError(
            f"n={n} at d={d} leaves {df} degrees of freedom outside the truth's "
            "span: a configuration of that few points spans the truth exactly, "
            "so the disparity is 0 by construction and there is no null to draw."
        )
    return df


def disparity_null_analytic(
    n: int, lam: Sequence[float], *, n_mc: int = _N_MC, seed: int = 0
) -> np.ndarray:
    """Sample the Gaussian structure null of the Procrustes disparity at *n*.

    With the thin SVD ``T = U L V^T`` of the standardised truth and ``Z = U^T X``
    -- exactly a d x d standard Gaussian matrix, because ``U``'s columns are
    orthonormal and orthogonal to ``1``, so centring ``X`` leaves ``Z`` alone --
    ``disparity = 1 - (sum_i sigma_i(X^T T))^2`` becomes

    .. code-block:: text

        disparity = 1 - ||L Z||_*^2 / (||Z||_F^2 + W_n)

        Z   ~ d x d standard Gaussian            -- carries no n
        W_n ~ chi^2_{(n-1-d)d}, independent of Z -- carries all the n

    where ``||.||_*`` is the nuclear norm and the split is
    ``||X_c||_F^2 = ||Z||_F^2 + W_n``: ``W_n`` is the mass outside ``U``'s span.
    Exact for the Gaussian generator at every ``n``.  For ``dirichlet`` use
    :func:`disparity_null_direct` -- ``X`` is not Gaussian there, and while the
    CLT recovers this expression for large ``n``, the band is wanted at small
    ``n`` precisely where the CLT has not arrived.

    Returns the sample, not a summary; call :func:`band_quantiles` on it.
    """
    lam = np.asarray(lam, dtype=np.float64)
    rng = np.random.default_rng(seed)
    nuc2, fro2 = _joint_sample(lam, n_mc, rng)
    w = rng.chisquare(_chi2_df(n, len(lam)), size=n_mc)
    return 1.0 - nuc2 / (fro2 + w)


def disparity_null_approx(
    n: int, lam: Sequence[float], *, n_mc: int = _N_MC, seed: int = 0
) -> np.ndarray:
    """``1 - ||L Z||_*^2 / ((n-1)d - 2)`` -- a **large-n approximation only**.

    This replaces the random denominator of :func:`disparity_null_analytic` with
    (near enough) its mean: ``E[||Z||_F^2 + W_n] = d^2 + (n-1-d)d = (n-1)d``,
    less the usual small correction for the fact that the reciprocal of a
    chi-square is not the reciprocal of its mean.  It is kept because it is a
    useful back-of-envelope and an independent cross-check on the sampler above.

    It is **wrong at small n**, in two ways that compound:

    * ``Z`` is on both sides.  ``||X_c||_F^2 = ||Z||_F^2 + W_n``, so the
      numerator's ``Z`` is part of its own denominator.  At ``n=5, d=2`` the
      outside-span term ``W_n`` carries ``(n-1-d)d = 4`` degrees of freedom while
      ``||Z||_F^2`` carries ``d^2 = 4``: ``Z`` is *half its own normaliser*.
    * Freezing a random denominator understates the 5-95 width, worst at the
      same end.

    Both failures land exactly where the band matters most.  Measured error
    against the exact form, worst band edge, at both ``d=2`` and ``d=3``:

    .. code-block:: text

        n      5      7     10     15     20     30     50    100
        err  0.52   0.21   0.09   0.04   0.02   0.008  0.003  0.001

    -- all of it in the lower edge, and at ``n=5`` far enough out to report a
    *negative* disparity.  :data:`APPROX_MIN_N` is where that crosses 0.01.  The
    ``uninformed approximation threshold`` check pins agreement above it and
    *disagreement* below it, so the threshold is a tested claim rather than a
    comment.
    """
    lam = np.asarray(lam, dtype=np.float64)
    d = len(lam)
    rng = np.random.default_rng(seed)
    nuc2, _ = _joint_sample(lam, n_mc, rng)
    return 1.0 - nuc2 / ((n - 1) * d - 2)


def disparity_null_direct(
    n: int,
    truth_coords: np.ndarray,
    *,
    generator: str = "gaussian",
    n_mc: int = 2_000,
    seed: int = 0,
) -> np.ndarray:
    """Sample the structure null by running the actual scoring path.

    Draws ``X`` from *generator*, standardises both configurations and
    superimposes, exactly as
    :func:`src.analysis.configurations.procrustes_compare` does -- so this
    validates the analytic form rather than restating it, and is the only
    correct route for a non-Gaussian generator.

    *truth_coords* is used at its own ``n``: only its first *n* rows are taken,
    so pass a truth already cut to the collection being scored.
    """
    from .configurations import _standardize, _superimpose

    t = np.asarray(truth_coords, dtype=np.float64)
    if t.shape[0] < n:
        raise ValueError(f"truth has {t.shape[0]} rows, need at least {n}")
    t_std, _, _ = _standardize(t[:n], True)
    k = t.shape[1] + 1

    rng = np.random.default_rng(seed)
    out = np.empty(n_mc, dtype=np.float64)
    for i in range(n_mc):
        x_std, _, _ = _standardize(draw_configuration(generator, n, k, rng), True)
        out[i] = _superimpose(t_std, x_std, True, True)[3]
    return out


# ── dCor's structure null ─────────────────────────────────────────────────────

def dcor_null_direct(
    n: int,
    truth_dm: np.ndarray,
    *,
    k: int,
    generator: str = "gaussian",
    n_mc: int = 2_000,
    seed: int = 0,
) -> np.ndarray:
    """Sample the structure null of dCor* at *n*, by running the scoring path.

    The centre needs no sampling: the bias-corrected numerator is unbiased for
    ``dCov^2``, which is 0 under independence, so the centre sits at ~0 for every
    ``n`` and every generator.  (dCor* is a ratio with a random denominator, so
    ~0, not exactly 0.)  What is sampled here is the *width*: under independence
    ``n * dCov^2_n`` converges to a weighted sum of chi-squares whose weights
    depend on the truth's own distance structure, so there is no closed form to
    fall back on.  :func:`fit_dcor_c` turns a handful of these samples into the
    power law the table stores.

    *k* is the number of simplex vertices.  It is a separate argument because
    this null is **matrix-level** -- a distance matrix does not say how many
    vertices the configuration behind it had, and the generator needs to know.
    """
    from scipy.spatial.distance import pdist, squareform

    from .matrices import _clean, _dcor_from_centered, _u_center

    t = np.asarray(truth_dm, dtype=np.float64)
    if t.shape[0] < n:
        raise ValueError(f"truth matrix is {t.shape[0]}x{t.shape[0]}, need {n}")
    b = _u_center(_clean(t[:n, :n]))

    rng = np.random.default_rng(seed)
    out = np.empty(n_mc, dtype=np.float64)
    for i in range(n_mc):
        x = draw_configuration(generator, n, k, rng)
        a = _u_center(_clean(squareform(pdist(x))))
        out[i] = _dcor_from_centered(a, b, True)
    return out


def fit_dcor_c(
    ns: Sequence[int], widths: Sequence[float]
) -> tuple[float, float]:
    """Least-squares ``(c, p)`` in ``width ~ c * n**(-p)``, fitted in log-log.

    **Fitted, not derived** -- including the exponent, and that is the point.

    ``docs/notes/chance_baselines.md`` predicted ``c/sqrt(n)``, i.e. ``p = 0.5``,
    from the CLT rate for the statistic.  Measured against
    :func:`dcor_null_direct` over ``n = 5..500``, ``p`` comes out at **1.09**,
    and identically so at ``k=3`` and ``k=4`` and under both generators:

    .. code-block:: text

        k=3 gaussian   -1.087      k=4 gaussian   -1.089
        k=3 dirichlet  -1.085      k=4 dirichlet  -1.085

    The note was reasoning about the classical dCor, which does decay at the
    ``sqrt`` rate.  ``dCor*`` is the bias-corrected estimator and lives on the
    **squared** scale -- it is a ratio of ``dCov^2`` to ``dVar``, never square
    rooted -- so its null width decays like the square of that, ``1/n``.  Half a
    decade of ``n`` separates the two predictions by a factor of three, so this
    is not a detail: assuming ``p = 0.5`` would draw a band roughly three times
    too wide at ``n = 500``.

    ``p`` is fitted rather than pinned at 1 because the measured value is
    consistently a little steeper, and a fitted exponent that is checked against
    the samples is worth more than a round number that is not.
    """
    x = np.log(np.asarray(ns, dtype=np.float64))
    y = np.log(np.asarray(widths, dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope)


def dcor_band_from_c(n: float, c: float, p: float) -> tuple[float, float, float]:
    """``(lo, mid, hi)`` for dCor* at *n*: centre 0, half-width ``c * n**-p``.

    The centre is exact, not fitted: the bias-corrected numerator is unbiased
    for ``dCov^2``, which is 0 under independence.  Only the half-width comes
    from :func:`fit_dcor_c`.
    """
    h = c * float(n) ** (-p)
    return (-h, 0.0, h)


# ── reading a band ────────────────────────────────────────────────────────────

def band_quantiles(sample: np.ndarray) -> tuple[float, float, float]:
    """``(lo, mid, hi)`` -- the :data:`BAND_Q` quantiles and the median."""
    lo, hi = np.percentile(sample, BAND_Q)
    return float(lo), float(np.median(sample)), float(hi)


def table_key(k: int, d: int, generator: str, score: str) -> str:
    """The one spelling of a table key, so writer and reader cannot drift."""
    return f"{k}|{d}|{generator}|{score}"


def uninformed_band(
    n: float,
    *,
    table: dict,
    score: str,
    generator: str,
    k: int,
    d: int,
) -> tuple[float, float, float]:
    """``(lo, mid, hi)`` for one ``n``, read from a precomputed table.

    *table* is the parsed ``results/baselines/constants.json``, written by
    ``scripts/make_baselines.py``.  Between stored grid points the band is
    interpolated **in log n**, which is the axis the collection-size figure
    uses; outside the stored range it raises rather than extrapolating, because
    both scores' n-dependence is exactly what the band exists to show and a
    guessed tail would misreport it.
    """
    key = table_key(k, d, generator, score)
    try:
        entry = table["entries"][key]
    except KeyError:
        have = sorted(table.get("entries", {}))
        raise KeyError(
            f"no uninformed baseline for {key!r}; the table has {have}. "
            "Rebuild it with `python scripts/make_baselines.py`."
        ) from None

    grid = np.asarray(entry["n"], dtype=np.float64)
    if n < grid[0] or n > grid[-1]:
        raise ValueError(
            f"n={n} is outside the tabulated range [{grid[0]:g}, {grid[-1]:g}] "
            f"for {key!r}; extend --n-grid in scripts/make_baselines.py rather "
            "than extrapolating."
        )
    lx = np.log(grid)
    at = np.log(float(n))
    return tuple(
        float(np.interp(at, lx, np.asarray(entry[q], dtype=np.float64)))
        for q in ("q05", "q50", "q95")
    )
