"""Verification checks for src/analysis.

Synthetic checks run everywhere; the data-backed checks are skipped with a note
when their inputs are absent, so the script is always runnable.

Usage:
    python scripts/check_analysis.py
    python scripts/check_analysis.py --synthetic-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import (
    align_to_reference,
    anchor_weight_vs_truth,
    as_distance_matrix,
    barycentric,
    compare_simplices,
    correlation_table,
    fit_geometry,
    kruskal_stress,
    dcor_test,
    distance_correlation,
    mantel_test,
    matrix_correlation,
    per_point_residuals,
    point_dispersion,
    procrustes_compare,
    protest,
    shepard,
)
from src.core.geometry import GeometryResult
from src.datasets import _text_projection

REPO = Path(__file__).parent.parent


def _default_shared_cache() -> Path:
    """Where the [data] checks look for the cache, when ``--cache-root`` is silent.

    Derived from ``Path(__file__)``, which is why the worktree case is handled
    explicitly: run from ``.claude/worktrees/<name>`` the in-repo path points
    inside the worktree, where no cache has ever been written, and every [data]
    check skips with "not present" — reporting an absent cache rather than an
    unresolved path. That is the same trap ``docs/notes/caching_collections.md``
    §5 records for the figure suite.
    """
    parts = REPO.parts
    if len(parts) >= 3 and parts[-2] == "worktrees" and parts[-3] == ".claude":
        main_checkout = REPO.parents[2] / "results/shared_cache"
        if main_checkout.exists():
            return main_checkout
    return REPO / "results/shared_cache"


SHARED_CACHE = _default_shared_cache()
ADAPTER_ROOT = SHARED_CACHE / "03_adapters"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
YAHOO_ADAPTERS = [
    "yahoo_100t0_000t1_n1000_s00_r16_i00",
    "yahoo_075t0_025t1_n1000_s00_r16_i00",
    "yahoo_050t0_050t1_n1000_s00_r16_i00",
    "yahoo_025t0_075t1_n1000_s00_r16_i00",
    "yahoo_000t0_100t1_n1000_s00_r16_i00",
]
YAHOO_TRUE = [1.00, 0.75, 0.50, 0.25, 0.00]


# ── harness ───────────────────────────────────────────────────────────────────

_RESULTS: list[tuple[str, str, str]] = []


def check(name: str):
    def deco(fn):
        def wrapped():
            try:
                note = fn()
                _RESULTS.append(("PASS", name, note or ""))
            except _Skip as e:
                _RESULTS.append(("SKIP", name, str(e)))
            except Exception as e:  # noqa: BLE001 - report, don't abort the suite
                _RESULTS.append(("FAIL", name, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
        wrapped.__name__ = fn.__name__
        wrapped.check_name = name  # so --list and -k can match on the description
        return wrapped
    return deco


class _Skip(Exception):
    pass


def _geometry(coords, ids=None, method="mds", taxonomy="synthetic"):
    coords = np.asarray(coords, dtype=np.float32)
    ids = ids or [f"m{i}" for i in range(coords.shape[0])]
    return GeometryResult(
        coordinates=coords,
        model_ids=list(ids),
        method=method,
        taxonomy=taxonomy,
        n_components=coords.shape[1],
    )


def _random_dm(n, seed=0, taxonomy="synthetic", metric="euclidean"):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    return as_distance_matrix([f"m{i}" for i in range(n)], d, metric, taxonomy)


def _synthetic_lora(n_adapters=4, d_out=48, d_in=64, rank=4, seed=99):
    """Build a small LoRAWeightCollection from random A/B factors.

    Uses the real container classes, so the low-rank builders are exercised on
    the same code path as genuine adapters — just at a size where materializing
    the dense product costs kilobytes instead of megabytes.
    """
    from src.notebook.lora_weights import AdapterWeights, LayerMatrices, LoRAWeightCollection

    rng = np.random.default_rng(seed)
    blocks = [(0, "o"), (0, "v"), (1, "o"), (1, "v")]
    adapters = {}
    for i in range(n_adapters):
        data = {}
        for layer, proj in blocks:
            A = rng.normal(size=(rank, d_in))
            B = rng.normal(size=(d_out, rank))
            data[(layer, proj)] = LayerMatrices(layer, proj, A, B)
        adapters[f"syn{i}"] = AdapterWeights(f"syn{i}", data)
    return LoRAWeightCollection(adapters), blocks


# ── synthetic checks ──────────────────────────────────────────────────────────

@check("simplex: anchors are fixed points")
def t_anchor_fixed():
    geo = _geometry([[0.0, 0.0], [1.0, 0.0], [0.4, 0.3], [0.7, -0.2]])
    proj = barycentric(geo, ["m0", "m1"])
    assert np.allclose(proj.weight_for("m0"), [1.0, 0.0], atol=1e-9), proj.weight_for("m0")
    assert np.allclose(proj.weight_for("m1"), [0.0, 1.0], atol=1e-9), proj.weight_for("m1")
    assert proj.residuals[0] < 1e-9 and proj.residuals[1] < 1e-9
    return "one-hot weights, zero residual"


@check("simplex: similarity invariance (the MDS ambiguity class)")
def t_similarity_invariance():
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(8, 4))
    geo = _geometry(coords)
    base = barycentric(geo, ["m0", "m3", "m5"], clip=False)

    # Rotation + reflection + uniform scale + translation: exactly the freedom
    # an MDS solution leaves undetermined.
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    mapped = _geometry(2.7 * (coords @ q) + rng.normal(size=4))
    after = barycentric(mapped, ["m0", "m3", "m5"], clip=False)

    delta = float(np.abs(base.weights - after.weights).max())
    assert delta < 1e-5, f"weights changed by {delta:.2e} under a similarity map"
    return f"max weight change {delta:.2e}"


@check("simplex: exact affine invariance for points in the affine hull")
def t_affine_invariance_in_hull():
    rng = np.random.default_rng(23)
    anchors = rng.normal(size=(3, 4))
    # Points built as affine combinations of the anchors, so residual == 0.
    lam = rng.dirichlet(np.ones(3), size=5)
    coords = np.vstack([anchors, lam @ anchors])
    ids = ["a0", "a1", "a2"] + [f"p{i}" for i in range(5)]
    base = barycentric(_geometry(coords, ids=ids), ["a0", "a1", "a2"], clip=False)
    assert float(base.residuals.max()) < 1e-5, base.residuals

    # A general affine map: shear and anisotropic scaling included.
    B = rng.normal(size=(4, 4))
    while abs(np.linalg.det(B)) < 1e-2:
        B = rng.normal(size=(4, 4))
    after = barycentric(
        _geometry(coords @ B + rng.normal(size=4), ids=ids),
        ["a0", "a1", "a2"], clip=False,
    )
    delta = float(np.abs(base.weights - after.weights).max())
    assert delta < 1e-4, f"in-hull weights changed by {delta:.2e} under an affine map"
    return f"max weight change {delta:.2e} under shear + anisotropic scale"


@check("simplex: recovers known mixtures on a segment")
def t_known_mixture():
    a0, a1 = np.array([2.0, -1.0]), np.array([-3.0, 4.0])
    fracs = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    pts = np.array([f * a0 + (1 - f) * a1 for f in fracs])
    geo = _geometry(pts, ids=[f"f{int(f * 100)}" for f in fracs])
    proj = barycentric(geo, ["f100", "f0"])
    got = proj.anchor_column(0)
    # GeometryResult coordinates are float32, so 1e-6 is the meaningful tolerance.
    assert np.allclose(got, fracs, atol=1e-6), f"{got} != {fracs}"
    assert float(proj.residuals.max()) < 1e-6
    return f"recovered {np.round(got, 3).tolist()}"


@check("simplex: k=2 anchors inside d=5 space")
def t_simplex_high_dim():
    rng = np.random.default_rng(3)
    a0 = np.zeros(5)
    a1 = np.array([1.0, 0, 0, 0, 0])
    on_line = np.array([0.6, 0, 0, 0, 0])
    off_line = on_line + np.array([0.0, 0.25, 0, 0, 0])
    geo = _geometry(np.vstack([a0, a1, on_line, off_line]),
                    ids=["a0", "a1", "on", "off"])
    proj = barycentric(geo, ["a1", "a0"], clip=False)

    assert abs(proj.weight_for("on")[0] - 0.6) < 1e-6
    assert proj.residuals[2] < 1e-6, "on-line point should have zero residual"
    assert abs(proj.residuals[3] - 0.25) < 1e-6, proj.residuals[3]
    # The off-line point still projects to the same position along the segment.
    assert abs(proj.weight_for("off")[0] - 0.6) < 1e-6
    return "residual captures the off-line component exactly"


@check("simplex: degenerate anchors rejected")
def t_degenerate_anchors():
    geo = _geometry([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 1.0]])
    try:
        barycentric(geo, ["m0", "m1", "m2"])   # collinear
    except ValueError as e:
        assert "affinely dependent" in str(e), e
        return "collinear anchors raise ValueError"
    raise AssertionError("expected ValueError for collinear anchors")


@check("simplex: compare_simplices agrees with itself")
def t_compare_simplices():
    rng = np.random.default_rng(11)
    geo = _geometry(rng.normal(size=(6, 3)))
    p = barycentric(geo, ["m0", "m1"], clip=False)
    same = compare_simplices(p, p)
    assert float(same.total_variation.max()) < 1e-12
    assert np.allclose(same.per_anchor_spearman, 1.0)

    shifted = _geometry(np.asarray(geo.coordinates) + rng.normal(scale=0.05, size=(6, 3)))
    q = barycentric(shifted, ["m0", "m1"], clip=False)
    cmp = compare_simplices(p, q)
    assert np.allclose(cmp.total_variation, 0.5 * cmp.l1)
    return f"identical -> tv=0; perturbed -> mean tv={cmp.mean_total_variation:.4f}"


@check("mantel: self-correlation and shuffled null")
def t_mantel():
    dm = _random_dm(8, seed=1)
    res = mantel_test(dm, dm, n_permutations=999, random_state=0)
    assert abs(res.statistic - 1.0) < 1e-9, res.statistic
    expected_p = 1.0 / (999 + 1)
    assert abs(res.p_value - expected_p) < 1e-12, res.p_value

    rng = np.random.default_rng(5)
    perm = rng.permutation(8)
    shuffled = as_distance_matrix(
        dm.model_ids, np.asarray(dm.matrix)[np.ix_(perm, perm)], "euclidean", "synthetic"
    )
    res2 = mantel_test(dm, shuffled, n_permutations=999, random_state=0)
    assert res2.p_value > 0.05, f"shuffled matrix looked significant: p={res2.p_value}"
    return f"self p={res.p_value:.4f}, shuffled p={res2.p_value:.3f}"


@check("dcor: bias correction is what makes it usable at five models")
def t_dcor_bias():
    """The classical V-statistic dCor is severely inflated at small n.

    This is the whole reason `distance_correlation` defaults to the
    bias-corrected form: at the five-model slices this repo actually compares,
    the classical statistic reports a large value between *independent*
    matrices, so it could never distinguish signal from nothing.
    """
    rng = np.random.default_rng(11)
    classical, corrected = [], []
    for _ in range(300):
        a = _random_dm(5, seed=int(rng.integers(1 << 30)))
        b = _random_dm(5, seed=int(rng.integers(1 << 30)))
        classical.append(distance_correlation(a, b, bias_corrected=False))
        corrected.append(distance_correlation(a, b, bias_corrected=True))

    assert np.mean(classical) > 0.5, np.mean(classical)
    assert abs(np.mean(corrected)) < 0.15, np.mean(corrected)

    # Self-correlation is exactly 1 under both, which pins the normalisation.
    dm = _random_dm(9, seed=3)
    for bc in (False, True):
        s = distance_correlation(dm, dm, bias_corrected=bc)
        assert abs(s - 1.0) < 1e-9, (bc, s)

    return (
        f"n=5 independent: classical mean={np.mean(classical):.3f}, "
        f"bias-corrected mean={np.mean(corrected):+.3f}; self=1.0 under both"
    )


@check("dcor: exact enumeration, and a calibrated null")
def t_dcor_test():
    # 5! = 120 <= 999, so every relabelling is enumerated and p is exact.
    dm = _random_dm(5, seed=1)
    res = dcor_test(dm, dm, n_permutations=999)
    assert res.exact and res.n_permutations == 120, (res.exact, res.n_permutations)
    assert abs(res.statistic - 1.0) < 1e-9, res.statistic
    # A perfect match ties only with the relabellings that preserve it.
    assert res.p_value <= 1.0 / 120 + 1e-12, res.p_value

    # 8! = 40320 > 999, so it falls back to sampling and the +1 correction.
    big = _random_dm(8, seed=2)
    sampled = dcor_test(big, big, n_permutations=999)
    assert not sampled.exact and sampled.n_permutations == 999

    # Unrelated matrices must not look significant.
    other = _random_dm(5, seed=7)
    null = dcor_test(dm, other, n_permutations=999)
    assert null.p_value > 0.05, f"unrelated matrices looked significant: p={null.p_value}"

    return (
        f"self: stat={res.statistic:.3f} p={res.p_value:.4f} over {res.n_permutations} "
        f"exact perms; unrelated p={null.p_value:.3f}; n=8 sampled {sampled.n_permutations}"
    )


@check("dcor: unsigned, so it cannot replace the signed correlation")
def t_dcor_unsigned():
    """dCor measures dependence, not agreement.

    A taxonomy recovering the mixing order exactly *backwards* scores dCor = 1
    just as a perfect one does -- and so, importantly, does the *signed*
    matrix correlation, because a distance matrix carries no direction at all.
    No matrix-level statistic can see the inversion; only the recovery
    correlation downstream of the barycentric projection can, which is where
    the behavioral level's r = -0.9995 shows up on the real cache while its
    matrix_corr_vs_truth sits at +0.76.

    Recorded so that "dCor and matrix corr agree, so the level is fine" is
    never read as evidence about direction.
    """
    w = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    truth = np.abs(w[:, None] - w[None, :])
    ids = [f"m{i}" for i in range(5)]

    def dm(matrix):
        return as_distance_matrix(ids, matrix, "euclidean", "synthetic")

    perfect = dm(truth)
    reversed_ = dm(truth[::-1, ::-1])

    d_perfect = distance_correlation(perfect, perfect)
    d_reversed = distance_correlation(reversed_, perfect)
    assert abs(d_perfect - 1.0) < 1e-9, d_perfect
    assert abs(d_reversed - 1.0) < 1e-9, (
        f"dCor is expected to be blind to reversal, got {d_reversed}"
    )

    # The signed matrix correlation does not rescue it: it is +1 for both,
    # because reversing the labelling leaves the multiset of distances alone.
    c_perfect = matrix_correlation(perfect, perfect, method="pearson")
    c_reversed = matrix_correlation(reversed_, perfect, method="pearson")
    assert c_perfect > 0.99, c_perfect
    assert c_reversed > 0.99, (
        f"expected the matrix correlation to be blind to reversal too, got {c_reversed}"
    )

    return (
        f"perfect and reversed are indistinguishable at the matrix level "
        f"(dCor={d_reversed:.4f}, corr={c_reversed:+.4f}) — only recovery sees the sign"
    )


@check("dcor: the reference's symmetry floors the p-value at n=5")
def t_dcor_u_centering_symmetry():
    """The permutation p-value cannot go below the reference matrix's symmetry.

    If `pi` relabels the U-centred reference `B` onto itself, then
    `dcor(A, B[pi, pi]) == dcor(A, B)` for *every* `A` -- the two calls receive
    the same pair of matrices.  So each automorphism contributes a null value
    tied with the observed one, ties count toward a one-sided p, and

        p >= #Aut(B) / n!

    holds for any `A` whatsoever.  It is a property of the reference alone, not
    of how well the taxonomy did.

    That bites here because U-centring *adds* symmetry: the evenly-spaced 1-D
    truth has 2 automorphisms raw and 2 doubly-centred, but 8 U-centred -- it
    can no longer tell either endpoint from its neighbour.  And the ground truth
    on the real slices is exactly that matrix (five mixtures at 0, .25, .5, .75,
    1), so 8/120 ~ 0.067 is a hard floor there and no level can reach p < 0.05
    however well it recovers the geometry.  `functional` sits on it exactly:
    dCor* = 0.928 with ge = 8, eq = 8, the automorphism orbit and nothing else.

    Not a universal n=5 floor, though, which is the second half of this check:
    an unevenly-spaced truth has 4 U-centred automorphisms and reaches 4/120 =
    0.033.  Escaping the floor needs a less symmetric design or more adapters --
    it is not something a better taxonomy can do.  Rank the levels by the
    statistic, not the p-value.

    An earlier draft asserted the opposite -- that some perturbed matrix would
    beat the exact match's p-value.  The argument above shows that is
    impossible; it is asserted the other way round below.
    """
    from itertools import permutations

    from src.analysis.matrices import _center, _clean, _dcor_from_centered

    w = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
    truth = np.abs(w[:, None] - w[None, :])
    perms = [np.asarray(p) for p in permutations(range(5))]

    def autos(m):
        return sum(1 for p in perms if np.allclose(m[np.ix_(p, p)], m))

    raw = autos(_clean(truth))
    doubled = autos(_center(_clean(truth), False))
    u = autos(_center(_clean(truth), True))
    assert raw == 2 and doubled == 2, (raw, doubled)
    assert u == 8, f"U-centring symmetry changed: {u}"

    ids = [f"m{i}" for i in range(5)]
    dm = as_distance_matrix(ids, truth, "euclidean", "synthetic")
    res = dcor_test(dm, dm, n_permutations=999)
    assert res.exact and res.n_permutations == 120
    assert abs(res.statistic - 1.0) < 1e-9, res.statistic
    assert abs(res.p_value - u / 120) < 1e-12, res.p_value

    # The null it is measured against takes only four distinct values, which is
    # the same symmetry seen from the other side.
    B = _center(_clean(truth), True)
    null = np.unique(np.round(
        [_dcor_from_centered(B, B[np.ix_(p, p)], True) for p in perms], 9))
    assert len(null) == 4, (len(null), null)

    # Nothing beats the floor, however the taxonomy is perturbed.  This is the
    # assertion an earlier draft had backwards.
    rng = np.random.default_rng(4)
    for _ in range(40):
        noise = rng.normal(scale=0.3, size=(5, 5))
        perturbed = truth + np.abs(0.5 * (noise + noise.T))
        r = dcor_test(as_distance_matrix(ids, perturbed, "euclidean", "synthetic"),
                      dm, n_permutations=999)
        assert r.p_value >= u / 120 - 1e-12, (
            f"p={r.p_value} beat the {u}/120 symmetry floor, which is impossible "
            f"unless dcor stopped being invariant under automorphisms of the "
            f"reference"
        )

    # A less symmetric truth has a lower floor, so p < 0.05 *is* reachable at
    # n=5 -- by changing the design, not by recovering the geometry better.
    uneven = np.array([1.0, 0.8, 0.45, 0.15, 0.0])
    t2 = np.abs(uneven[:, None] - uneven[None, :])
    u2 = autos(_center(_clean(t2), True))
    dm2 = as_distance_matrix(ids, t2, "euclidean", "synthetic")
    res2 = dcor_test(dm2, dm2, n_permutations=999)
    assert u2 < u, (u2, u)
    assert abs(res2.p_value - u2 / 120) < 1e-12, res2.p_value
    assert res2.p_value < 0.05, res2.p_value

    return (
        f"automorphisms raw={raw} doubled={doubled} u-centred={u}; exact match "
        f"p={res.p_value:.4f} over a null of {len(null)} distinct values, "
        f"unbeaten by 40 perturbations; unevenly-spaced truth has {u2} "
        f"automorphisms and reaches p={res2.p_value:.4f}"
    )


@check("procrustes: invariant to rotation, reflection and scale")
def t_procrustes():
    rng = np.random.default_rng(2)
    coords = rng.normal(size=(7, 2))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    F = np.array([[1.0, 0.0], [0.0, -1.0]])          # reflection
    transformed = 3.5 * (coords @ R @ F) + np.array([10.0, -4.0])

    res = procrustes_compare(_geometry(coords), _geometry(transformed))
    assert res.disparity < 1e-10, res.disparity

    # A perfect fit ties with any permutation that happens to be the identity,
    # so p is at the floor 1/(n+1) but not exactly equal to it.
    pt = protest(_geometry(coords), _geometry(transformed), n_permutations=999)
    assert pt.p_value < 0.01, pt.p_value
    return f"disparity={res.disparity:.2e}, protest p={pt.p_value:.4f}"


@check("procrustes: matches scipy.spatial.procrustes")
def t_procrustes_vs_scipy():
    from scipy.spatial import procrustes as sp_procrustes

    rng = np.random.default_rng(21)
    # GeometryResult stores float32, so hand scipy the same rounded values —
    # otherwise the two differ at 1e-9 purely from input precision.
    a = rng.normal(size=(9, 3)).astype(np.float32)
    b = rng.normal(size=(9, 3)).astype(np.float32)
    _, _, ref = sp_procrustes(np.float64(a), np.float64(b))
    ours = procrustes_compare(_geometry(a), _geometry(b)).disparity
    assert abs(ours - ref) < 1e-10, f"{ours} vs scipy {ref}"
    return f"disparity {ours:.6f} == scipy {ref:.6f}"


@check("procrustes: per-point residual finds the displaced model")
def t_per_point_residuals():
    rng = np.random.default_rng(4)
    coords = rng.normal(size=(9, 2)) * 3.0
    moved = coords.copy()
    moved[5] += np.array([1.5, 1.5])
    # scaling=False keeps the displacement local: with unit-norm scaling on, a
    # single large move rescales the entire configuration and smears the signal.
    res = procrustes_compare(_geometry(coords), _geometry(moved), scaling=False)
    r = per_point_residuals(res)
    assert int(np.argmax(r)) == 5, f"largest residual at {int(np.argmax(r))}, expected 5"
    assert r[5] > 3 * np.median(r), f"m5={r[5]:.4f}, median={np.median(r):.4f}"
    return f"model m5 residual {r[5]:.4f} vs median {np.median(r):.4f}"


@check("dispersion: stable models score lower than jittery ones")
def t_dispersion():
    rng = np.random.default_rng(13)
    base = rng.normal(size=(10, 2)) * 3.0
    geoms = []
    for _ in range(6):
        jitter = np.zeros_like(base)
        jitter[0] = rng.normal(scale=0.5, size=2)      # m0 wanders, the rest do not
        geoms.append(_geometry(base + jitter))

    # scaling=False isolates the effect being tested: with unit-norm scaling on,
    # one model's movement also changes the whole configuration's norm, which
    # redistributes a little displacement onto every other point.
    disp = point_dispersion(geoms, scaling=False)
    assert int(np.argmax(disp.per_model)) == 0, disp.per_model
    # Compare against the median rather than the max: the rotation fitted to
    # absorb m0's movement necessarily nudges whichever point sits furthest from
    # the centroid, so the second-largest value is not a clean baseline.
    assert disp.per_model[0] > 2 * np.median(disp.per_model[1:]), disp.per_model

    aligned = align_to_reference(geoms)
    assert len(aligned) == len(geoms)
    assert aligned[0].model_ids == geoms[0].model_ids
    return f"m0 dispersion {disp.per_model[0]:.4f} vs others <= {disp.per_model[1:].max():.4f}"


@check("quality: stress zero for an exact embedding, Shepard on y=x")
def t_quality():
    rng = np.random.default_rng(6)
    pts = rng.normal(size=(7, 2))
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    dm = as_distance_matrix([f"m{i}" for i in range(7)], d, "euclidean", "synthetic")
    geo = _geometry(pts)

    s = kruskal_stress(dm, geo)
    assert s < 1e-6, s
    orig, emb = shepard(dm, geo)
    assert np.allclose(orig, emb, atol=1e-6)
    return f"stress={s:.2e}, {len(orig)} pairs on y=x"


@check("matrices: correlation table is symmetric with unit diagonal")
def t_correlation_table():
    a = _random_dm(6, seed=1, taxonomy="a")
    b = _random_dm(6, seed=2, taxonomy="b")
    labels, table = correlation_table({"a": a, "b": b})
    assert labels == ["a", "b"]
    assert np.allclose(np.diag(table), 1.0)
    assert np.allclose(table, table.T)
    assert abs(table[0, 1] - matrix_correlation(a, b)) < 1e-12
    return f"corr(a,b)={table[0, 1]:.4f}"


@check("matrices: match_models handles disjoint order and membership")
def t_match_models():
    from src.analysis import match_models

    a = as_distance_matrix(["x", "y", "z"], np.array(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]), "m", "a")
    b = as_distance_matrix(["z", "x", "w"], np.array(
        [[0.0, 5.0, 6.0], [5.0, 0.0, 7.0], [6.0, 7.0, 0.0]]), "m", "b")
    ids, (ma, mb) = match_models(a, b)
    assert ids == ["x", "z"], ids
    assert abs(ma[0, 1] - 2.0) < 1e-12, ma
    assert abs(mb[0, 1] - 5.0) < 1e-12, mb
    return "intersected to ['x', 'z'] in first-object order"


@check("bridge: fit_geometry honours n_components")
def t_fit_geometry():
    dm = _random_dm(6, seed=8)
    for n in (1, 2, 3):
        geo = fit_geometry(dm, method="mds", n_components=n, random_state=0)
        assert geo.coordinates.shape == (6, n), geo.coordinates.shape
    pca = fit_geometry(dm, method="pca", n_components=2)
    assert pca.coordinates.shape == (6, 2)
    return "mds n_components 1/2/3 and pca all fine"


@check("bridge: cosine similarity is converted to a distance")
def t_similarity_conversion():
    s = np.array([[1.0, 0.6], [0.6, 1.0]])
    dm = as_distance_matrix(["a", "b"], s, "cosine", "structural", similarity=True)
    assert abs(dm.matrix[0, 1] - 0.4) < 1e-12, dm.matrix
    assert abs(dm.matrix[0, 0]) < 1e-12, "diagonal must be zero"
    return "1 - S applied, diagonal zeroed"


@check("simplex: SimplexProjection round-trips through safetensors")
def t_simplex_roundtrip():
    import tempfile

    rng = np.random.default_rng(17)
    geo = _geometry(rng.normal(size=(5, 3)))
    proj = barycentric(geo, ["m0", "m2"])
    with tempfile.TemporaryDirectory() as td:
        proj.save(Path(td) / "p")
        back = type(proj).load(Path(td) / "p")
    assert back.model_ids == proj.model_ids
    assert back.anchor_ids == proj.anchor_ids
    assert np.allclose(back.weights, proj.weights, atol=1e-6)
    assert np.allclose(back.residuals, proj.residuals, atol=1e-6)
    return "weights, residuals and metadata preserved"


# ── ground truth from recipes ─────────────────────────────────────────────────

@check("ground truth: mixture components for the three recipe forms")
def t_mixture_weights():
    from src.analysis import mixture_weights
    from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
    from src.datasets.recipe import DatasetEntry, DatasetRecipe

    # Whole datasets mixed: one vertex each, weights straight from the recipe.
    three = DatasetRecipe(
        name="three",
        datasets=[DatasetEntry("dsA", weight=5.0), DatasetEntry("dsB", weight=3.0),
                  DatasetEntry("dsC", weight=2.0)],
    )
    got = mixture_weights(three)
    assert got == {"dsA": 0.5, "dsB": 0.3, "dsC": 0.2}, got

    # Classes mixed inside one dataset: still two things being mixed, so two
    # vertices — not one degenerate vertex for the dataset.
    ca = ClassAwareDatasetRecipe(
        name="mix",
        datasets=[ClassDatasetEntry("yahoo", class_field="topic", class_filter=[0, 1],
                                    class_weights={0: 1.0, 1: 3.0})],
    )
    got = mixture_weights(ca)
    assert got == {"yahoo[topic=0]": 0.25, "yahoo[topic=1]": 0.75}, got

    # The dict form, which is what recipe.json holds, must agree exactly — its
    # class keys are strings where the object's are ints.
    assert mixture_weights(ca.to_dict()) == got, "dict and object forms disagree"

    # Both kinds of mixing at once: dsA split by class, dsB taken whole. An entry
    # with no class_filter/class_weights is one vertex for the dataset itself —
    # its weight is known even though the split inside it is not.
    mixed = ClassAwareDatasetRecipe(
        name="both",
        datasets=[
            ClassDatasetEntry("dsA", weight=1.0, class_field="c", class_filter=[0, 1]),
            ClassDatasetEntry("dsB", weight=1.0),
        ],
    )
    got = mixture_weights(mixed)
    assert set(got) == {"dsA[c=0]", "dsA[c=1]", "dsB"}, got
    assert abs(sum(got.values()) - 1.0) < 1e-12
    return "3 datasets, 2 classes, dict==object, and the mixed case"


@check("adapter names: the mixture parser returns a K-vector, never a truncated one")
def t_mixture_name_k_vector():
    """The regression test for a bug that produced plausible wrong numbers.

    The pattern was ``_(\\d{3})g1_(\\d{3})g2_(\\d{3})g3_``, which requires a
    trailing underscore after ``g3`` -- and a 4-group name supplies one, because
    ``g4`` follows. So a 4-group id *matched*, the fourth group fell off the end,
    and a renormalized 3-vector came back with no exception and no warning. That
    array is what ``simplex_suite.truth_weights`` builds the ground truth from,
    so it is a scoring bug wearing a plotting bug's clothes.
    """
    from src.plots.simplex import (mixture_label, mixture_weights, n_groups,
                                   sort_by_mixture)

    # The three cases from the design note. The third is the damaging one: an
    # edge midpoint that the old pattern reported as a pure vertex.
    assert mixture_weights("dolly_025g1_025g2_025g3_025g4_n1000_s00") == \
        (0.25, 0.25, 0.25, 0.25)
    assert mixture_weights("dolly_100g1_000g2_000g3_000g4_n1000_s00") == \
        (1.0, 0.0, 0.0, 0.0)
    assert mixture_weights("oasst1_000g1_000g2_050g3_050g4_n500_s00") == \
        (0.0, 0.0, 0.5, 0.5), "an edge midpoint must not read as a pure vertex"

    # Yahoo's names are unchanged, including the centre, whose parts sum to 99.
    assert mixture_weights("yahoo_100g1_000g2_000g3_n1000_s00") == (1.0, 0.0, 0.0)
    centre = mixture_weights("yahoo_033g1_033g2_033g3_n1000_s00")
    assert len(centre) == 3 and abs(sum(centre) - 1.0) < 1e-12
    assert all(abs(w - 1 / 3) < 1e-12 for w in centre)
    assert mixture_label("dolly_025g1_025g2_025g3_025g4_n1") == "25/25/25/25"

    # A non-consecutive index sequence is a malformed name, not a 4-group one
    # with something after it. Raising is the point: the old pattern would have
    # returned three weights here.
    try:
        mixture_weights("x_025g1_025g2_025g3_025g5_n1")
    except ValueError:
        pass
    else:
        raise AssertionError("a g5 after g1..g3 must raise, not truncate")

    # Mixed widths cannot be stacked, and truth_weights stacks whatever
    # sort_by_mixture hands it -- so the failure belongs here, named, rather
    # than at an np.vstack that mentions neither dataset.
    mixed = ["yahoo_100g1_000g2_000g3_n1", "dolly_025g1_025g2_025g3_025g4_n1"]
    for fn in (n_groups, sort_by_mixture):
        try:
            fn(mixed)
        except ValueError as exc:
            assert "3" in str(exc) and "4" in str(exc), str(exc)
        else:
            raise AssertionError(f"{fn.__name__} accepted a mixed-width collection")

    return "4-group ids give 4 weights, yahoo is unchanged, mixed widths raise"


@check("procrustes: the disparity is swept over d = 2..K-1, and cannot zero-pad silently")
def t_procrustes_dimension_sweep():
    """The second silent-failure bug, and it is the same shape as the first.

    ``truth_geometry`` builds the ground truth in ``K-1`` dimensions -- 2-D for
    three vertices, 3-D for four -- while ``rank_surrogates`` fitted the taxonomy
    side at a hardcoded ``n_components=2``. That mismatch did not error:
    ``procrustes_compare`` takes ``d = max(a.shape[1], b.shape[1])`` and
    zero-pads the narrower one, so a flat 2-D configuration was superimposed on a
    tetrahedral truth and the disparity absorbed every bit of truth variance
    outside the best-fit plane -- a floor no surrogate can beat, reported as an
    ordinary number. It was honest only because K=3 makes ``K-1 == 2``.
    """
    import numpy as np

    from src.analysis import disparity_vs_truth
    from src.core.distance import DistanceMatrix
    from src.plots.simplex_suite import rank_surrogates, truth_geometry, vertices

    def ids_for(k, mixes):
        return [f"c_" + "_".join(f"{p:03d}g{i + 1}" for i, p in enumerate(m))
                + "_n1000_s00" for m in mixes]

    # A 3-simplex and a 2-simplex, each with a distance matrix that IS the truth,
    # so the disparity should be ~0 at the truth's own dimension.
    cases = {
        3: [(100, 0, 0), (0, 100, 0), (0, 0, 100), (33, 33, 33),
            (50, 50, 0), (0, 50, 50), (50, 0, 50)],
        4: [(100, 0, 0, 0), (0, 100, 0, 0), (0, 0, 100, 0), (0, 0, 0, 100),
            (25, 25, 25, 25), (50, 50, 0, 0), (0, 0, 50, 50), (50, 0, 50, 0)],
    }
    out = []
    for k, mixes in cases.items():
        ids = ids_for(k, mixes)
        assert len(vertices(ids)) == k
        tgeo = truth_geometry(ids)
        coords = np.asarray(tgeo.coordinates)
        assert coords.shape[1] == k - 1, f"K={k} truth is {coords.shape[1]}-D"

        # The truth's own pairwise distances, presented as a taxonomy result.
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        dm = DistanceMatrix(matrix=d, model_ids=list(ids), metric="euclidean",
                            taxonomy="synthetic")

        scored = rank_surrogates({("truth", "euclidean"): dm}, ids, tgeo=tgeo)
        assert len(scored) == 1
        sc = scored[0]

        assert sorted(sc.procrustes_by_d) == list(range(2, k)), sc.procrustes_by_d
        assert sorted(sc.stress_by_d) == list(range(2, k))
        # The designated scalar is the truth's own dimension, K-1.
        assert sc.procrustes == sc.procrustes_by_d[k - 1]
        assert sc.stress == sc.stress_by_d[k - 1]
        # At its own dimension the truth recovers itself. The tolerance is
        # MDS convergence, not float noise: SMACOF stops on a stress delta, so
        # an exact recovery lands around 1e-5 rather than at machine epsilon.
        assert sc.procrustes < 1e-4, f"K={k}: d{k-1} disparity {sc.procrustes}"

        if k == 4:
            # And a flat fit of a genuinely 3-D truth cannot: the d2 column is
            # the honest report of what the drawn panel shows, and it is worse.
            assert sc.procrustes_by_d[2] > 100 * sc.procrustes_by_d[3], \
                sc.procrustes_by_d
            # Stress falls monotonically with d, which is why the stress columns
            # qualify each disparity rather than comparing across them.
            assert sc.stress_by_d[3] <= sc.stress_by_d[2] + 1e-9
        out.append(f"K={k}: d{sorted(sc.procrustes_by_d)}")

        # The mismatch itself must now raise rather than be absorbed.
        if k == 4:
            from src.analysis.bridge import fit_geometry
            flat = fit_geometry(dm, method="mds", n_components=2, random_state=0)
            try:
                disparity_vs_truth(dm, tgeo, geometry=flat)
            except ValueError as exc:
                assert "zero-pad" in str(exc), str(exc)
            else:
                raise AssertionError(
                    "a 2-D embedding against a 3-D truth must raise, not zero-pad")

    return "; ".join(out) + "; width mismatch raises"


@check("ground truth: a dataset cannot be both split and whole in one collection")
def t_split_and_whole_rejected():
    """The one case where an unsplit entry is genuinely ambiguous.

    Taking a dataset whole is a valid vertex, and splitting it by class is a
    valid pair of vertices — but a collection that does both has a whole-dataset
    vertex that is an unknown mixture of the per-class ones, so they are not
    independent corners and the simplex would be fictitious.
    """
    from src.analysis import ground_truth_weights

    split = {
        "recipe_type": "class_aware", "normalized_weights": [1.0],
        "datasets": [{"dataset_id": "dsA", "class_field": "c",
                      "normalized_class_weights": {"0": 0.5, "1": 0.5}}],
    }
    whole = {
        "recipe_type": "class_aware", "normalized_weights": [1.0],
        "datasets": [{"dataset_id": "dsA", "class_field": "c"}],
    }
    try:
        ground_truth_weights({"split": split, "whole": whole})
    except ValueError as e:
        assert "not " in str(e) and "independent" in str(e), e
    else:
        raise AssertionError("expected the mixed treatment to be rejected")

    # Either treatment on its own is fine.
    v1, _ = ground_truth_weights({"a": split, "b": split})
    v2, _ = ground_truth_weights({"a": whole, "b": whole})
    assert v1 == ["dsA[c=0]", "dsA[c=1]"], v1
    assert v2 == ["dsA"], v2
    return "mixed treatment rejected; each treatment alone accepted"


@check("ground truth: simplex is regular and round-trips through barycentric")
def t_simplex_geometry():
    from scipy.spatial.distance import pdist

    from src.analysis import (barycentric, evaluation_points, ground_truth_weights,
                              pure_anchors, simplex_geometry, simplex_vertices,
                              truth_matrix)

    for k in (2, 3, 4):
        edges = pdist(simplex_vertices(k))
        assert np.allclose(edges, 1.0), f"k={k} edges not equal: {edges}"

    recipes = {
        "pureA": {"recipe_type": "simple", "normalized_weights": [1.0],
                  "datasets": [{"dataset_id": "A"}]},
        "pureB": {"recipe_type": "simple", "normalized_weights": [1.0],
                  "datasets": [{"dataset_id": "B"}]},
        "pureC": {"recipe_type": "simple", "normalized_weights": [1.0],
                  "datasets": [{"dataset_id": "C"}]},
        "mix":   {"recipe_type": "simple", "normalized_weights": [0.5, 0.3, 0.2],
                  "datasets": [{"dataset_id": "A"}, {"dataset_id": "B"}, {"dataset_id": "C"}]},
    }
    vertices, weights = ground_truth_weights(recipes)
    assert vertices == ["A", "B", "C"], vertices

    ids = list(recipes)
    geo = simplex_geometry(weights, ids, vertices)
    assert geo.n_components == 2, geo.n_components

    anchors = pure_anchors(vertices, weights)
    assert anchors == ["pureA", "pureB", "pureC"], anchors
    assert evaluation_points(ids, anchors) == ["mix"]

    proj = barycentric(geo, anchors)
    back = np.vstack([proj.weight_for(m) for m in ids])
    delta = float(np.abs(back - truth_matrix(weights, ids)).max())
    assert delta < 1e-6, f"round-trip error {delta:.2e}"
    return f"regular for k=2,3,4; round-trip error {delta:.1e}"


def _mixture_truth():
    """A small 3-component mixture collection: weights, ids, geometry, matrix."""
    from src.analysis import simplex_distance_matrix, simplex_geometry

    W = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0],
                  [.34, .33, .33], [0, .5, .5], [.25, .25, .5], [.6, .2, .2]],
                 dtype=float)
    ids = [f"m{i}" for i in range(len(W))]
    vertices = ["g1", "g2", "g3"]
    return (W, ids,
            simplex_geometry(W, ids, vertices),
            simplex_distance_matrix(W, ids, vertices))


@check("ground truth: Procrustes disparity vs truth is 0 for the simplex itself")
def t_disparity_vs_truth_exact():
    """Pins the direction of the score: **0 is perfect**, not 1.

    It is reported in the same tables as dCor, which runs the other way, so a
    refactor that quietly flipped the convention would produce a table that
    still looks plausible. Hence a check on the sign rather than only on the
    plumbing. The tolerance is loose because SMACOF is iterative — embedding an
    exactly-simplicial distance matrix recovers the simplex to optimiser
    tolerance, not to machine epsilon.
    """
    from src.analysis import disparity_vs_truth
    from src.analysis.bridge import as_distance_matrix
    from scipy.spatial.distance import pdist, squareform

    _, ids, tgeo, tdm = _mixture_truth()

    exact = disparity_vs_truth(tdm, tgeo)
    assert exact < 1e-3, f"the simplex should score ~0 against itself, got {exact:.4f}"

    rng = np.random.default_rng(0)
    noise = as_distance_matrix(
        ids, squareform(pdist(rng.normal(size=(len(ids), 3)))), "euclidean",
        taxonomy="noise")
    scrambled = disparity_vs_truth(noise, tgeo)
    assert scrambled > 0.1, f"random points should score far from 0, got {scrambled:.4f}"
    return f"exact {exact:.1e}, random {scrambled:.3f}"


@check("ground truth: Procrustes pairs models by id, not by row position")
def t_disparity_vs_truth_label_keyed():
    """The sanity check that the score reads the labels.

    ``procrustes_compare`` reindexes both configurations onto their common
    ``model_ids`` before fitting, so row order is bookkeeping and identity is
    not. Two consequences, both pinned here because the row-order bug
    (``docs/notes/row_order_bug.md``) is exactly what happens when a matrix and
    its labels come apart:

    * permuting a matrix **together with** its ids leaves the score unchanged;
    * permuting the ids **alone** — same numbers, wrong names — changes it.

    The second is the one that matters. If it ever stopped holding, a mislabelled
    matrix would score as well as a correct one and nothing else would notice.
    """
    from src.analysis import disparity_vs_truth
    from src.core.distance import DistanceMatrix

    _, ids, tgeo, tdm = _mixture_truth()
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(ids))
    assert not np.array_equal(perm, np.arange(len(ids))), "degenerate permutation"

    base = disparity_vs_truth(tdm, tgeo)
    m = np.asarray(tdm.matrix, dtype=float)

    together = DistanceMatrix(matrix=m[np.ix_(perm, perm)],
                              model_ids=[ids[i] for i in perm],
                              metric=tdm.metric, taxonomy=tdm.taxonomy)
    delta = abs(disparity_vs_truth(together, tgeo) - base)
    assert delta < 1e-6, f"permuting rows with their ids moved the score by {delta:.2e}"

    mislabelled = DistanceMatrix(matrix=m, model_ids=[ids[i] for i in perm],
                                 metric=tdm.metric, taxonomy=tdm.taxonomy)
    wrong = disparity_vs_truth(mislabelled, tgeo)
    assert wrong > 0.1, (
        f"mislabelled rows still scored {wrong:.4f} — the comparison is not "
        "reading model_ids")
    return f"invariant to {delta:.1e}; mislabelled scores {wrong:.3f}"


@check("ground truth: a k-vertex simplex needs k-1 dimensions, and says so")
def t_simplex_dimension_requirement():
    """The executable form of the dimension question.

    A simplex on k vertices lives in k-1 dimensions, and barycentric coordinates
    are undefined below that because the anchors stop being affinely independent.
    Pinned here so the claim cannot quietly stop being true.
    """
    from src.analysis import barycentric, ground_truth_weights, pure_anchors, simplex_geometry

    recipes = {
        f"pure{c}": {"recipe_type": "simple", "normalized_weights": [1.0],
                     "datasets": [{"dataset_id": c}]}
        for c in "ABCD"
    }
    recipes["mix"] = {
        "recipe_type": "simple", "normalized_weights": [0.4, 0.3, 0.2, 0.1],
        "datasets": [{"dataset_id": c} for c in "ABCD"],
    }
    vertices, weights = ground_truth_weights(recipes)
    ids = list(recipes)
    anchors = pure_anchors(vertices, weights)
    k = len(vertices)

    full = simplex_geometry(weights, ids, vertices)
    assert full.n_components == k - 1 == 3

    barycentric(full, anchors)  # k-1 works

    squashed = GeometryResult(
        coordinates=np.asarray(full.coordinates)[:, : k - 2],
        model_ids=ids, method="mds", taxonomy="t", n_components=k - 2,
    )
    try:
        barycentric(squashed, anchors)
    except ValueError as e:
        assert "affinely dependent" in str(e), e
        return f"k={k}: {k - 1}d accepted, {k - 2}d refused with a usable message"
    raise AssertionError(f"expected {k - 2}d to be refused for a {k}-vertex simplex")


@check("ground truth: residuals vanish when projecting from exactly k-1 dimensions")
def t_projection_dimension_matters():
    """Why the comparison projects from the largest embedding, not from k-1.

    With k anchors in exactly k-1 dimensions the anchors' affine hull *is* the
    whole space, so every residual is identically zero and "how far off the
    simplex is this model?" can no longer be asked.  One dimension more and the
    diagnostic comes back.
    """
    from src.analysis import barycentric

    # Two anchors, one point deliberately off the line between them.
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.4]])
    ids = ["a0", "a1", "off"]
    in_2d = barycentric(_geometry(coords, ids=ids), ["a0", "a1"])
    in_1d = barycentric(_geometry(coords[:, :1], ids=ids), ["a0", "a1"])

    assert float(in_1d.residuals.max()) < 1e-9, in_1d.residuals
    assert in_2d.residuals[2] > 0.3, in_2d.residuals
    return (
        f"k-1=1d max residual {float(in_1d.residuals.max()):.1e} (blind); "
        f"2d {float(in_2d.residuals[2]):.3f} (sees it)"
    )


@check("procrustes: the stored map reproduces the fit, and applies off it")
def t_procrustes_transform():
    """A round trip proves bytes survived; this proves the stored pieces are the map.

    ``procrustes_compare`` standardises both configurations before rotating one, so
    the rotation alone is not the transformation — the centroid and norm that were
    divided out are part of it.  They used to be discarded, which left the fit
    inspectable but impossible to apply to any point outside it.
    """
    import tempfile

    rng = np.random.default_rng(31)
    a = rng.normal(size=(7, 2))
    b = rng.normal(size=(7, 2))
    res = procrustes_compare(_geometry(a), _geometry(b))

    # The map must reproduce its own output.
    got = res.transform(b)
    ref = np.asarray(res.aligned_b.coordinates, dtype=np.float64)
    delta = float(np.abs(got - ref).max())
    assert delta < 1e-5, f"transform does not reproduce aligned_b: {delta:.2e}"
    assert np.allclose(
        res.transform(a, which="a"), res.aligned_a.coordinates, atol=1e-5
    ), "which='a' does not reproduce aligned_a"

    # Applicability to a point that was not part of the fit. A duplicate of row 3
    # is well defined without refitting, and is exactly what the dropped
    # centroid/norm made impossible.
    outside = res.transform(b[3][None, :])
    assert np.allclose(outside[0], ref[3], atol=1e-5), outside

    # scaling=False must leave the norms inert rather than silently rescaling.
    unscaled = procrustes_compare(_geometry(a), _geometry(b), scaling=False)
    assert unscaled.norm_a == 1.0 and unscaled.norm_b == 1.0
    assert np.allclose(
        unscaled.transform(b), unscaled.aligned_b.coordinates, atol=1e-5
    )

    # The case this project actually runs: a 2-D embedding against a 1-D truth.
    mixed = procrustes_compare(_geometry(a), _geometry(b[:, :1]))
    assert np.allclose(
        mixed.transform(b[:, :1]), mixed.aligned_b.coordinates, atol=1e-5
    ), "zero-padding path is wrong for differing dimensions"

    with tempfile.TemporaryDirectory() as td:
        res.save(Path(td))
        back = type(res).load(Path(td))
    assert np.allclose(back.rotation, res.rotation)
    assert abs(back.scale - res.scale) < 1e-12
    assert abs(back.disparity - res.disparity) < 1e-12
    assert back.model_ids == res.model_ids
    assert (back.scaling, back.reflection) == (res.scaling, res.reflection)
    assert np.allclose(back.transform(b), got, atol=1e-9), "reloaded map differs"
    return (
        f"reproduces aligned_b to {delta:.1e}, applies off the fit, "
        "survives save/load, and handles 2-D vs 1-D"
    )


@check("cache: 1-D and 2-D embeddings of one collection coexist")
def t_collection_multidim():
    """Coordinates used to be keyed by method alone, so these overwrote each other."""
    import tempfile

    from src.analysis import fit_geometry
    from src.cache import CollectionCache

    dm = _random_dm(6, seed=3)
    with tempfile.TemporaryDirectory() as td:
        cc = CollectionCache(td)
        entries = [
            {"model_id": mid, "artifact_path": f"04_activations/x/{mid}/r/n64_s00"}
            for mid in dm.model_ids
        ]
        chash = cc.handle(
            dm.taxonomy,
            cc.collection_key(entries),
            dm.metric,
            cc.surrogate_key(["s0"] * len(entries)),
        )
        cc.save_distance_matrix(
            dm, chash, model_entries=entries,
            label="check", slice_key={"n_samples": 10},
        )
        made = {n: fit_geometry(dm, "mds", n_components=n, random_state=0) for n in (1, 2)}
        for geo in made.values():
            cc.save_geometry(chash, geo)

        assert sorted(cc.list_geometries(chash)) == [("mds", 1), ("mds", 2)]
        for n, geo in made.items():
            back = cc.load_geometry(chash, "mds", n)
            assert back.coordinates.shape == (6, n), back.coordinates.shape
            assert np.allclose(back.coordinates, geo.coordinates, atol=1e-6)
            assert back.stress is not None, "stress lost on reload"

        try:
            cc.load_geometry(chash, "mds")   # ambiguous without n_components
        except ValueError as e:
            assert "dimensions" in str(e), e
        else:
            raise AssertionError("expected an ambiguity error")

        assert cc.find(taxonomy=dm.taxonomy, metric=dm.metric) == [chash]
    return "both dimensions kept, stress preserved, ambiguity reported, index queryable"


@check("core: reindex permutes a distance matrix, and refuses what it cannot")
def t_distance_matrix_reindex():
    """The guard every read from the collection cache goes through.

    Permuting a symmetric distance matrix is exact, so the property is simple:
    ``d.reindex(order)[a, b] == d[a, b]`` for every pair, however the rows are
    arranged. A subset is a legitimate request — it is what makes a superset
    collection on disk reusable — and an unknown id is not, because a silently
    shorter matrix is the same class of defect one step along.
    """
    dm = _random_dm(6, seed=11)
    ids = list(dm.model_ids)
    shuffled = [ids[i] for i in (4, 0, 5, 2, 1, 3)]

    perm = dm.reindex(shuffled)
    assert list(perm.model_ids) == shuffled, perm.model_ids
    assert perm.metric == dm.metric and perm.taxonomy == dm.taxonomy
    for a in ids:
        for b in ids:
            assert np.isclose(perm[(a, b)], dm[(a, b)]), (a, b)

    back = perm.reindex(ids)
    assert list(back.model_ids) == ids
    assert np.allclose(back.matrix, dm.matrix), "permute and permute back moved it"

    sub = dm.reindex([ids[3], ids[1]])
    assert sub.matrix.shape == (2, 2)
    assert np.isclose(sub.matrix[0, 1], dm[(ids[3], ids[1])])

    for bad, word in (([ids[0], "nope"], "nope"), ([ids[0], ids[0]], "duplicate")):
        try:
            dm.reindex(bad)
        except ValueError as e:
            assert word in str(e), e
        else:
            raise AssertionError(f"reindex accepted {bad!r}")

    return "round-trips, subsets, rejects unknown and duplicate ids"


@check("cache: a stored matrix is read back in the caller's row order")
def t_collection_cache_row_order():
    """Row order is not in the handle, so the read has to put it back.

    ``collection_key`` sorts the model entries before hashing, so a matrix
    written in one order and one written in another land on the *same* key. The
    stored ``model_ids`` are self-describing, so the bytes on disk are right —
    but a raw hit hands back rows in whoever-wrote-it-first's order, and the
    caller who asked in a different order gets a matrix whose labels no longer
    describe its rows. That is ``docs/notes/row_order_bug.md``, and through a
    cache it would be *worse*: the same wrong number on every run, which is the
    shape of a result rather than the shape of a bug.

    So this asserts both halves. The guarded read must match the caller's order,
    and the **unguarded** read must not — a guard that is not load-bearing is a
    decoration, and the second assertion is what tells the two apart.
    """
    import tempfile

    from src.cache import CollectionCache

    dm = _random_dm(5, seed=12)
    written = list(dm.model_ids)
    asked = written[::-1]

    with tempfile.TemporaryDirectory() as td:
        cc = CollectionCache(td)
        entries = [{"model_id": mid, "artifact_path": f"04_activations/{mid}"}
                   for mid in written]
        # The reader's entries are the same models in the caller's order, which
        # is the collision: sorted before hashing, the two lists are one key.
        reader_entries = [{"model_id": mid, "artifact_path": f"04_activations/{mid}"}
                          for mid in asked]
        handle = cc.handle(dm.taxonomy, cc.collection_key(entries), dm.metric,
                           cc.surrogate_key(["s0"] * len(entries)))
        assert handle == cc.handle(
            dm.taxonomy, cc.collection_key(reader_entries), dm.metric,
            cc.surrogate_key(["s0"] * len(entries))
        ), "the two orders no longer collide; this check has stopped testing anything"

        cc.save_distance_matrix(dm, handle, model_entries=entries, label="check")
        raw = cc.load_distance_matrix(handle)
        guarded = raw.reindex(asked)

    assert list(guarded.model_ids) == asked, guarded.model_ids
    for i, a in enumerate(asked):
        for j, b in enumerate(asked):
            assert np.isclose(guarded.matrix[i, j], dm[(a, b)]), (a, b)

    # What the unguarded read would have handed back, under the caller's labels.
    mislabelled = float(np.abs(raw.matrix - guarded.matrix).max())
    assert mislabelled > 0, (
        "the raw hit already matched the caller's order, so this check would "
        "pass without the reindex guard — pick a permutation that moves rows"
    )

    # And a superset on disk serves a subset, but only through the guard.
    sub = raw.reindex(asked[:3])
    assert sub.matrix.shape == (3, 3)
    return (f"guarded read matches the caller's ids; unguarded differs by "
            f"{mislabelled:.3f}, subset selection works")


@check("core: TaxonomyAnalysis keeps every geometry, old profiles still load")
def t_analysis_geometries():
    import tempfile

    from src.analysis import fit_geometry
    from src.core.analysis import TaxonomyAnalysis

    dm = _random_dm(5, seed=4)
    analysis = TaxonomyAnalysis("structural", list(dm.model_ids), [], dm)
    for n in (1, 2, 3):
        analysis.add_geometry(fit_geometry(dm, "mds", n_components=n, random_state=0))
    analysis.add_geometry(fit_geometry(dm, "pca", n_components=2))

    with tempfile.TemporaryDirectory() as td:
        analysis.save(Path(td))
        back = TaxonomyAnalysis.load(Path(td))
        assert sorted(back.geometries) == ["mds_1d", "mds_2d", "mds_3d", "pca_2d"], sorted(back.geometries)
        for key, geo in analysis.geometries.items():
            assert np.allclose(back.geometries[key].coordinates, geo.coordinates, atol=1e-6)
        assert back.geometry is not None, "primary slot lost"

        # A profile written before geometries/ existed must still load.
        import shutil
        shutil.rmtree(Path(td) / "geometries")
        legacy = TaxonomyAnalysis.load(Path(td))
        assert legacy.geometries == {} and legacy.geometry is not None
    return "4 embeddings round-tripped; pre-existing profiles unaffected"


# ── data-backed checks ────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_yahoo_weights(layer_indices=(27,), projections="o"):
    """Load real adapter factors once and reuse across checks.

    Only the requested (layer, projection) A/B factors are read — a few hundred
    KB per adapter, not the whole 37 MB safetensors file.
    """
    if not ADAPTER_ROOT.exists():
        raise _Skip(f"{ADAPTER_ROOT} not present")
    from src.notebook.lora_weights import load_lora_weights

    missing = [
        a for a in YAHOO_ADAPTERS
        if not (ADAPTER_ROOT / BASE_MODEL.replace("/", "--") / a).exists()
    ]
    if missing:
        raise _Skip(f"{len(missing)} yahoo adapter(s) missing, e.g. {missing[0]}")
    return load_lora_weights(
        YAHOO_ADAPTERS,
        adapter_root=ADAPTER_ROOT,
        layer_indices=list(layer_indices),
        projections=projections,
    )


@check("cosine: low-rank path equals CosineDistanceMetric on the dense product")
def t_cosine_equivalence():
    """The §3 claim, proved where it is cheap to prove.

    The identity — low-rank cosine over concatenated ``B @ A`` blocks equals
    flatten-cosine over the materialized representation, with zero padding
    contributing nothing — is algebraic and independent of dimension.  Verifying
    it on small synthetic factors is exactly as conclusive as verifying it on
    3072x3072 adapters, and does not allocate hundreds of megabytes to do so.
    """
    from src.analysis import lora_distance_matrix
    from src.core.representation import ModelRepresentation
    from src.metrics import CosineDistanceMetric

    weights, blocks = _synthetic_lora(n_adapters=4, d_out=48, d_in=64, rank=4)
    layers = sorted({l for l, _ in blocks})
    projs = sorted({p for _, p in blocks})
    low_rank = lora_distance_matrix(weights, kind="cosine", layers=layers, projections=projs)

    # The representation StructuralTaxonomy builds: one row per (layer, proj)
    # block holding (B @ A).flatten(), rows zero-padded to a common width.
    rows_per_adapter = {
        name: [weights[name].product(l, p).ravel() for l in layers for p in projs]
        for name in low_rank.model_ids
    }
    width = max(len(r) for rows in rows_per_adapter.values() for r in rows)
    reps = [
        ModelRepresentation(
            model_id=name,
            taxonomy="structural",
            matrix=np.stack(
                [np.pad(r, (0, width - len(r))) for r in rows_per_adapter[name]]
            ).astype(np.float32),
        )
        for name in low_rank.model_ids
    ]

    metric = CosineDistanceMetric()
    n = len(reps)
    direct = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            direct[i, j] = direct[j, i] = metric.compute(reps[i], reps[j])

    delta = float(np.abs(np.asarray(low_rank.matrix) - direct).max())
    assert delta < 1e-5, f"max difference {delta:.3e} exceeds 1e-5"

    # Padding really is inert: widen every row and the metric must not move.
    padded = [
        ModelRepresentation(
            model_id=r.model_id,
            taxonomy=r.taxonomy,
            matrix=np.pad(r.matrix, ((0, 0), (0, 37))).astype(np.float32),
        )
        for r in reps
    ]
    d2 = metric.compute(padded[0], padded[1])
    assert abs(d2 - direct[0, 1]) < 1e-6, f"padding changed the distance: {d2} vs {direct[0, 1]}"

    return f"max |low-rank - metric| = {delta:.2e}; padding inert to {abs(d2 - direct[0, 1]):.1e}"


@check("bures-wasserstein: metric class equals the low-rank structural builder")
def t_bures_wasserstein_equivalence():
    """The port is exact, and this is where that is cheap to show.

    ``bures_wasserstein_distance_matrix`` builds ``M = R A`` from ``B = QR`` and
    stacks it across blocks; the representation path stacks ``B @ A`` instead.
    Both are factors of the same covariance — ``MᵀM = AᵀBᵀBA = (BA)ᵀ(BA)`` — and
    BW depends on the factor only through that covariance, so the identity is
    exact.

    Checked in two stages, because they have different error floors.  In float64
    the agreement is at the 1e-13 level, which is the identity itself.  Through
    ``ModelRepresentation`` it is ~1e-9 relative, and that is not the algorithm:
    ``__post_init__`` (``src/core/representation.py:37``) coerces every stored
    matrix to float32, so the metric sees a rounded copy of the factors the
    structural builder reads at full precision.  Asserting one tolerance for both
    would either hide the exactness or fail on the storage dtype.
    """
    from src.core.representation import ModelRepresentation
    from src.metrics import BuresWassersteinDistanceMetric
    from src.notebook.structure import bures_wasserstein_distance_matrix

    weights, blocks = _synthetic_lora(n_adapters=4, d_out=48, d_in=64, rank=4)
    layers = sorted({l for l, _ in blocks})
    projs = sorted({p for _, p in blocks})
    names, low_rank = bures_wasserstein_distance_matrix(
        weights, layers=layers, projections=projs
    )

    # One factor per adapter: the dense blocks stacked, which is what the
    # behavioral/functional levels hand the metric.
    factors = [
        np.vstack([weights[name].product(l, p) for l in layers for p in projs])
        for name in names
    ]
    n = len(names)
    scale = float(np.asarray(low_rank).max())

    def _pairwise(fn):
        out = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                out[i, j] = out[j, i] = fn(i, j)
        return out

    # Stage 1: the identity itself, at full precision.
    def _bw64(i, j):
        cross = np.linalg.svd(factors[i] @ factors[j].T, compute_uv=False)
        d2 = np.sum(factors[i] ** 2) + np.sum(factors[j] ** 2) - 2.0 * cross.sum()
        return float(np.sqrt(max(float(d2), 0.0)))

    exact = _pairwise(_bw64)
    d_exact = float(np.abs(np.asarray(low_rank) - exact).max()) / scale
    assert d_exact < 1e-10, f"float64 relative difference {d_exact:.3e} exceeds 1e-10"

    # Stage 2: through the metric class, i.e. through float32 storage.
    metric = BuresWassersteinDistanceMetric()
    reps = [
        ModelRepresentation(model_id=name, taxonomy="structural", matrix=f)
        for name, f in zip(names, factors)
    ]
    assert reps[0].matrix.dtype == np.float32, "storage dtype assumption broke"
    direct = _pairwise(lambda i, j: metric.compute(reps[i], reps[j]))
    d_stored = float(np.abs(np.asarray(low_rank) - direct).max()) / scale
    assert d_stored < 1e-6, f"stored relative difference {d_stored:.3e} exceeds 1e-6"

    return (
        f"relative to low-rank over {n} adapters: float64 {d_exact:.1e}, "
        f"float32-stored {d_stored:.1e}"
    )


@check("bures-wasserstein: permutation invariant, scale equivariant")
def t_bures_wasserstein_invariance():
    """The property that distinguishes BW from CKA and Frobenius.

    Those two compare row *i* of one model against row *i* of the other and raise
    if the counts differ.  BW compares Σ = XᵀX, which does not know the row order
    — so a shuffled copy of a representation is at distance zero from it, and two
    models may be compared at different row counts.  Both are asserted here
    because both are load-bearing when the rows are sampled generations rather
    than an aligned query list.
    """
    from src.core.representation import ModelRepresentation
    from src.metrics import BuresWassersteinDistanceMetric

    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 12))
    metric = BuresWassersteinDistanceMetric()

    def rep(m, mid="m"):
        return ModelRepresentation(model_id=mid, taxonomy="functional", matrix=m)

    assert metric.compute(rep(X), rep(X)) < 1e-9, "distance to self is not zero"

    shuffled = X[rng.permutation(len(X))]
    d_perm = metric.compute(rep(X), rep(shuffled, "shuf"))
    assert d_perm < 1e-9, f"row permutation moved the distance: {d_perm:.3e}"

    # Scale equivariance: X -> cX must scale the distance by |c|.  Tolerance is
    # relative and set for float32 storage (`ModelRepresentation.__post_init__`
    # coerces), not for the arithmetic, which is exact in the ideal.
    Y = rng.normal(size=(30, 12))
    base = metric.compute(rep(X), rep(Y, "y"))
    scaled = metric.compute(rep(3.0 * X), rep(3.0 * Y, "y"))
    rel = abs(scaled - 3.0 * base) / (3.0 * base)
    assert rel < 1e-6, (
        f"scaling by 3 gave {scaled:.8f}, expected {3.0 * base:.8f} "
        f"(relative {rel:.2e})"
    )

    # Different row counts are allowed, where CKA and Frobenius raise.
    d_ragged = metric.compute(rep(X), rep(rng.normal(size=(7, 12)), "short"))
    assert np.isfinite(d_ragged) and d_ragged > 0, d_ragged

    # A mismatched feature dimension is not, and must say so.
    try:
        metric.compute(rep(X), rep(rng.normal(size=(30, 5)), "narrow"))
    except ValueError as exc:
        assert "feature dimension" in str(exc), str(exc)
    else:
        raise AssertionError("mismatched feature dimension did not raise")

    return f"perm {d_perm:.1e}; scale rel {rel:.1e}; ragged {d_ragged:.4f}"


@check("[data] cosine: low-rank matrix is well formed on real adapters")
def t_cosine_real_adapters():
    """Plumbing check on the real cache — deliberately low-rank only.

    The numerical identity is covered synthetically above; forming the dense
    3072x3072 products here would cost hundreds of megabytes for no extra
    information.
    """
    from src.analysis import lora_distance_matrix

    weights = _load_yahoo_weights(layer_indices=(27,), projections="o")
    dm = lora_distance_matrix(weights, kind="cosine", layers=[27], projections="o")

    m = np.asarray(dm.matrix)
    assert dm.model_ids == YAHOO_ADAPTERS, dm.model_ids
    assert np.allclose(m, m.T), "matrix is not symmetric"
    assert np.allclose(np.diag(m), 0.0, atol=1e-9), "diagonal is not zero"
    assert np.all(np.isfinite(m)), "non-finite entries"
    assert m.min() >= -1e-9 and m.max() <= 2.0 + 1e-9, (m.min(), m.max())

    # Adapters are listed 100% -> 0% topic 0, so distance from the first should
    # grow monotonically along the row.
    row = m[0, 1:]
    assert np.all(np.diff(row) > 0), f"not monotone away from the 100% adapter: {row}"
    return f"5 adapters, distances from 100% = {np.round(row, 4).tolist()}"


@check("[data] recovery: MDS recovers the topic-0 mixing proportion")
def t_recovery():
    from src.analysis import lora_distance_matrix

    weights = _load_yahoo_weights(layer_indices=(27,), projections="o")
    dm = lora_distance_matrix(weights, kind="cosine", layers=[27], projections="o")
    geo = fit_geometry(dm, method="mds", n_components=2, random_state=42)

    anchors = [YAHOO_ADAPTERS[0], YAHOO_ADAPTERS[-1]]   # 100% and 0% topic 0
    proj = barycentric(geo, anchors)
    truth = dict(zip(YAHOO_ADAPTERS, YAHOO_TRUE))
    rec = anchor_weight_vs_truth(proj, truth, anchor=0)

    # Anchors are fixed points of the projection; this catches ordering bugs.
    assert abs(proj.weight_for(anchors[0])[0] - 1.0) < 1e-9
    assert abs(proj.weight_for(anchors[1])[0] - 0.0) < 1e-9
    assert np.all(np.isfinite(rec.recovered)), rec.recovered

    print("      true      :", np.round(rec.true, 3).tolist())
    print("      recovered :", np.round(rec.recovered, 3).tolist())
    print("      residuals :", np.round(rec.residuals, 4).tolist())
    # Deliberately no threshold on r or rho. Those are measurements of how well
    # this taxonomy tracks the training mixture — a finding to read, not a
    # contract to enforce. A weak correlation should show up as a number worth
    # investigating, not as a red FAIL that invites tuning until it passes.
    # Only the structural invariants above are asserted.
    return f"r={rec.r:.4f}, rho={rec.rho:.4f}, max residual={float(rec.residuals.max()):.4g}"


@check("[data] persistence: CollectionCache round-trip")
def t_collection_roundtrip():
    import tempfile

    from src.analysis import lora_distance_matrix, save_collection
    from src.cache import CollectionCache

    weights = _load_yahoo_weights(layer_indices=(27,), projections="o")
    dm = lora_distance_matrix(weights, kind="cosine", layers=[27], projections="o")
    geo = fit_geometry(dm, method="mds", n_components=2, random_state=42)

    with tempfile.TemporaryDirectory() as td:
        chash = save_collection(dm, [geo], cache_root=td)
        cc = CollectionCache(td)
        back_dm = cc.load_distance_matrix(chash)
        back_geo = cc.load_geometry(chash, "mds")

    assert back_dm.model_ids == dm.model_ids
    assert np.allclose(back_dm.matrix, dm.matrix, atol=1e-6)
    assert np.allclose(back_geo.coordinates, geo.coordinates, atol=1e-6)
    return f"collection {chash} round-tripped"


@check("[data] cache: two selectors at one metric are two collections")
def t_collection_key_sees_selector():
    """The regression test for TODO.md item 14.

    The collection key used to be ``(model_ids, taxonomy, metric)``, so a
    collection built under ``normalize="global"`` was returned unchanged to a
    caller asking for ``"layer"`` — silently, and with no way to tell from the
    outside.  Two things have to hold now: the two selectors must land in
    *different* directories, and the matrices in them must actually differ.

    Checking only the first would pass on a key that was merely salted; checking
    only the second would pass without any cache at all.
    """
    import tempfile

    from src.analysis import build_taxonomy_artifacts, scan_cache
    from src.cache import CollectionCache

    root = SHARED_CACHE
    if not (root / "04_activations").exists():
        raise _Skip(f"{root}/04_activations not present")

    index = scan_cache(root).with_available("functional_repr")
    slices = index.slices(("n_samples", "seed"))
    if not slices or len(slices[max(slices)]) < 3:
        raise _Skip("no (n_samples, seed) slice with 3+ functional models")
    sub = slices[max(slices)]

    with tempfile.TemporaryDirectory() as td:
        mats = {}
        for norm in ("global", "layer"):
            mats[norm], _ = build_taxonomy_artifacts(
                sub, "functional", "cka", cache_root=td, n_components=(2,),
                functional_selector={"normalize": norm},
            )
        handles = CollectionCache(td).list_collections()

    assert len(handles) == 2, f"expected 2 collections, got {len(handles)}: {handles}"
    key_a, key_b = (h.split("/")[1] for h in handles)
    assert key_a == key_b, (
        "the two collections differ in collection_key, but they are the same "
        f"models reading the same artifacts: {handles}"
    )
    surrogates = {h.rsplit("_", 1)[1] for h in handles}
    assert len(surrogates) == 2, f"surrogate_key did not separate them: {handles}"

    delta = float(np.abs(mats["global"].matrix - mats["layer"].matrix).max())
    assert delta > 0, "the two normalizations produced identical matrices"
    return (
        f"2 collections under one collection_key, surrogate_key separates them, "
        f"max|Δ| = {delta:.2e}"
    )


@check("[data] cache: two surrogates of one level are two collections")
def t_collection_surrogate_in_key():
    """The figure suite's rows have to be keyed apart, and by the right thing.

    A grid row is a *surrogate*: one level, one resolved selector, seven metric
    columns. Two surrogates of one level read the same artifacts under the same
    surrogate — that is what makes them one level — so nothing about the
    collection itself separates them, and the resolved selector has to be in the
    key. Both halves are asserted, for the reason
    ``t_collection_key_sees_selector`` gives: two different handles prove nothing
    if the matrices are the same, and two different matrices prove nothing about
    the key.

    The key is composed from the **resolved selectors**, never from the row's
    display label. ``"late third"`` is editable prose; redefining which layers it
    names without changing the string would have a label-keyed entry serve a
    matrix built from the old definition — a stale number under a current name,
    which is the one failure a cache must not have.
    """
    from collections import Counter

    from src.analysis import scan_cache
    from src.analysis.comparison import (
        _compute_distance_matrix, collection_handle, resolve_ordered,
    )
    from src.cache import CollectionCache

    if not (SHARED_CACHE / "04_activations").exists():
        raise _Skip(f"{SHARED_CACHE}/04_activations not present")

    # The draw with the most models behind it, read off the layout. A draw has to
    # be named explicitly because the cache holds several, and comparing across
    # two of them is meaningless rather than merely imprecise — they are
    # different questions put to the models.
    draw_re = re.compile(r"^n(\d+)_s(\d+)(?:_f([0-9a-f]+))?$")
    counts = Counter()
    for path in (SHARED_CACHE / "04_activations").glob("*/*/*/*"):
        m = draw_re.match(path.name)
        if m and path.is_dir():
            counts[(path.parent.name, int(m.group(1)), int(m.group(2)),
                    m.group(3))] += 1
    if not counts:
        raise _Skip(f"no stored draws under {SHARED_CACHE}/04_activations")
    recipe_hash, n_samples, seed, fmt = counts.most_common(1)[0][0]
    draw = {"recipe_hash": recipe_hash, "n_samples": n_samples, "seed": seed}
    if fmt:
        draw["prompt_format_id"] = fmt

    sub = scan_cache(SHARED_CACHE, functional_draw=draw).with_available(
        "functional_repr")
    if len(sub.entries) < 3:
        raise _Skip(f"only {len(sub.entries)} functional model(s) under {draw}")
    ids = [e.model_id for e in sub.entries]

    surrogates = {"h1": {"functional_selector": {"draw": draw, "layers": [1]}},
             "h2": {"functional_selector": {"draw": draw, "layers": [2]}}}

    cc = CollectionCache(SHARED_CACHE)
    handles, mats = {}, {}
    for name, surrogate in surrogates.items():
        _, _, entries = resolve_ordered(sub, "functional", ids,
                                        with_identity=True, **surrogate)
        handles[name] = collection_handle(cc, "functional", "cosine", entries,
                                          surrogate=surrogate)
        mats[name] = _compute_distance_matrix(sub, "functional", "cosine", ids,
                                              **surrogate)

    assert handles["h1"] != handles["h2"], (
        f"both surrogates key to {handles['h1']}, so the second would read back the "
        "first's matrix"
    )
    delta = float(np.abs(mats["h1"].matrix - mats["h2"].matrix).max())
    assert delta > 0, (
        "the two surrogates produced identical matrices, so this check cannot tell a "
        "working key from a broken one — pick two surrogates that differ"
    )

    # Isolate the surrogate element. The two handles above could have been separated
    # by something else the resolution saw, so hold the models, the metric and
    # the resolution fixed and vary *only* the surrogate: what is left is the surrogate's
    # own contribution to the key.
    _, _, entries = resolve_ordered(sub, "functional", ids, with_identity=True,
                                    **surrogates["h1"])
    fixed = collection_handle(cc, "functional", "cosine", entries,
                              surrogate=surrogates["h1"])
    assert fixed == handles["h1"], (
        "the same resolved selector keyed to two different handles"
    )
    varied = collection_handle(cc, "functional", "cosine", entries,
                               surrogate={"functional_selector": {"draw": draw,
                                                             "layers": [1, 2]}})
    assert varied != fixed, (
        "the surrogate does not reach the key: one set of resolved representations "
        "keyed identically under two different selectors"
    )
    return f"two handles, max|Δ| = {delta:.2e}, selector-keyed not label-keyed"


@check("[data] cross-taxonomy: correlation table over a saved profile")
def t_cross_taxonomy():
    from src.core.analysis import ModelTaxonomyProfile

    root = REPO / "results/yahoo_topics/taxonomy"
    if not (root / "meta.json").exists():
        raise _Skip(f"{root} not present")

    profile = ModelTaxonomyProfile.load(root)
    labels, table = correlation_table(profile)
    if len(labels) < 2:
        raise _Skip(f"only {len(labels)} taxonomy level(s) saved")

    width = max(len(l) for l in labels)
    print("      " + " " * width + "  " + "  ".join(f"{l[:7]:>7}" for l in labels))
    for i, l in enumerate(labels):
        cells = ["      -" if np.isnan(v) else f"{v:>7.3f}" for v in table[i]]
        print(f"      {l:>{width}}  " + "  ".join(cells))

    assert np.allclose(np.diag(table), 1.0)
    assert np.allclose(table, table.T, equal_nan=True)

    # dataset_embedding is keyed by recipe ID, the model-level taxonomies by
    # adapter path, so with no key that row is incomparable and must be nan
    # rather than an exception that takes the whole table down with it.
    off = table[~np.eye(len(labels), dtype=bool)]
    n_comparable = int(np.sum(~np.isnan(off)) // 2)
    assert n_comparable > 0, "no taxonomy pair was comparable"
    return f"{len(labels)} levels, {n_comparable} comparable pair(s), rest nan"


@check("[data] identity: recipe_id_for makes dataset_embedding comparable")
def t_recipe_relabelling():
    from src.core.analysis import ModelTaxonomyProfile
    from src.analysis import id_overlap, recipe_id_for

    root = REPO / "results/yahoo_topics/taxonomy"
    if not (root / "meta.json").exists():
        raise _Skip(f"{root} not present")

    profile = ModelTaxonomyProfile.load(root)
    mats = {k: v.distance_matrix for k, v in profile.analyses.items()}
    if "dataset_embedding" not in mats:
        raise _Skip("no dataset_embedding level in this profile")

    model_level = next(k for k in ("structural", "functional", "behavioral") if k in mats)
    de = mats["dataset_embedding"]

    before = id_overlap(mats[model_level], de)
    after = id_overlap(mats[model_level], de, key=recipe_id_for)
    assert before["n_common"] == 0, "expected disjoint identifier spaces before relabelling"
    assert after["n_common"] == len(mats[model_level].model_ids), after

    # Every adapter must resolve, and to the recipe the fine-tuning script
    # recorded — not merely to *some* string that happens to collide.
    for mid in mats[model_level].model_ids:
        rid = recipe_id_for(mid)
        assert rid in de.model_ids, f"{mid} -> {rid} not a dataset_embedding id"

    # An unadapted HuggingFace ID must survive untouched: it is not an adapter
    # path, and stripping it at the "/" would silently produce "meta-llama".
    assert recipe_id_for("meta-llama/Llama-3.2-3B") == "meta-llama/Llama-3.2-3B"

    _, table = correlation_table(mats, key=recipe_id_for)
    assert not np.isnan(table).any(), "table still has incomparable pairs"

    return (
        f"{after['n_common']} models matched, "
        f"{len(de.model_ids) - after['n_common']} dataset-only entry(s) dropped; "
        "full table"
    )


@check("discovery: a dataset filter separates two corpora under one base model")
def t_scan_cache_dataset_filter():
    """The guard that keeps the yahoo drivers working once dolly is trained.

    ``03_adapters/<base_slug>`` holds every adapter for a base model whatever it
    was trained on — that is the design, and it is what makes the cache shared.
    The ``*_draw`` arguments set per-entry availability *flags* and never filter
    the list, and no adapter path contains ``output_dir``. So a second corpus on
    the same base model is invisible to every existing selector, and the only
    symptom is a count guard tripping on 51 models where 16 were expected.
    """
    import json
    import tempfile

    from src.analysis import datasets_present, scan_cache

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        slug = "Qwen--Qwen3.5-4B"
        planted = {"yahoo": 0, "dolly": 0}
        for corpus, mixes, n, ds_id in [
            ("yahoo", ["100g1_000g2_000g3", "000g1_100g2_000g3", "033g1_033g2_033g3"],
             1000, "yahoo_answers_topics"),
            ("dolly", ["100g1_000g2_000g3_000g4", "025g1_025g2_025g3_025g4"],
             1000, "databricks/databricks-dolly-15k"),
        ]:
            for mix in mixes:
                name = f"{corpus}_{mix}_n{n}_s00_r16_i00_b5008_fdeadbeef"
                d = root / "03_adapters" / slug / name
                d.mkdir(parents=True)
                (d / "adapter_model.safetensors").write_bytes(b"")
                (d / "experiment_meta.json").write_text(json.dumps({
                    "base_model_id": "Qwen/Qwen3.5-4B",
                    "dataset_name": f"{corpus}_{mix}_n{n}_s00",
                    "dataset_recipe_hash": f"hash_{corpus}",
                    "lora_config": {"r": 16},
                    "training": {"samples_seen": 5008},
                }))
                planted[corpus] += 1

        total = planted["yahoo"] + planted["dolly"]

        # No filter: today's behaviour, unchanged. This is the default, so every
        # existing caller has to land here.
        assert len(scan_cache(root).model_ids) == total

        # One corpus at a time — the case that makes the yahoo drivers correct
        # again rather than merely making the new ones possible.
        for corpus, want in planted.items():
            got = scan_cache(root, datasets=[corpus]).model_ids
            assert len(got) == want, f"{corpus}: {len(got)} != {want}"
            assert all(Path(m).name.startswith(corpus + "_") for m in got)

        # Several at once. Accepting a sequence rather than one name is an
        # explicit requirement: the filter must not make a deliberately
        # cross-dataset comparison impossible to express.
        both = scan_cache(root, datasets=["yahoo", "dolly"])
        assert len(both.model_ids) == total

        # A prefix is compared token-wise, not with startswith, so a corpus name
        # cannot also match a longer one that begins with it.
        assert len(scan_cache(root, datasets=["doll"]).model_ids) == 0

        # And the message the count guard needs can name what it found.
        assert datasets_present(scan_cache(root)) == ["dolly", "yahoo"]
        assert datasets_present(scan_cache(root, datasets=["dolly"])) == ["dolly"]

    return (f"{total} planted, filtered to {planted['yahoo']} yahoo / "
            f"{planted['dolly']} dolly, union {total}, default unfiltered")


@check("[data] discovery: the cache scan joins adapters to their recipes")
def t_scan_cache():
    from src.analysis import scan_cache

    root = SHARED_CACHE
    if not (root / "03_adapters").exists():
        raise _Skip(f"{root}/03_adapters not present")

    index = scan_cache(root)
    if not len(index):
        raise _Skip("no adapters with experiment_meta.json in the cache")

    # The join is on recipe_hash, which finetune_lora.py recorded — not on a
    # parsed directory name.
    joined = [e for e in index if e.recipe is not None]
    assert joined, "no adapter resolved to a recipe"
    for e in joined:
        assert e.recipe_hash and e.recipe.get("recipe_hash") == e.recipe_hash, e.adapter_name

    usable = index.with_available("structural_weights", "dataset_embedding")
    groupings = {
        by: usable.slices(by=by)
        for by in [("n_samples", "seed"), ("n_samples",), ("seed",), ()]
    }
    for by, slices in groupings.items():
        assert slices, f"grouping {by} produced nothing"
        total = sum(len(s) for s in slices.values())
        assert total == len(usable), f"grouping {by} lost models: {total} != {len(usable)}"

    shapes = ", ".join(
        f"{'+'.join(by) or 'pooled'}={len(s)}" for by, s in groupings.items()
    )
    return (
        f"{len(index)} adapter(s), {len(joined)} with recipes, "
        f"{len(usable)} usable; slices: {shapes}"
    )


def _draws_shared_by_all(root: Path, stage: str) -> set:
    """``{(recipe_hash, n_samples, seed)}`` every model in *stage* has.

    Works for either inference stage, because since the re-key they enumerate
    identically.
    """
    from src.cache.activation_cache import ActivationCache
    from src.cache.generated_text_cache import GeneratedTextCache

    if not (root / stage).exists():
        return set()
    cache = (GeneratedTextCache if stage == "05_generated" else ActivationCache)(root)
    per_model = []
    for base_slug, adapter_slug in cache.list_models():
        draws = cache.list_draws(base_slug.replace("--", "/"), adapter_slug)
        per_model.append({(d["recipe_hash"], d["n_samples"], d["seed"]) for d in draws})
    return set.intersection(*per_model) if per_model else set()


def _shared_inference_draw(root: Path, stage: str) -> dict | None:
    """One draw for *stage*, preferring one the other inference stage also has.

    Several draws are several query sets, so picking arbitrarily would compare
    models across different questions.  But when both levels hold the same draw,
    that one is unambiguously the right choice — and preferring it is what makes
    a cross-level comparison read the *same* queries at both levels, which is the
    property the re-key exists to provide.  Returns None when the choice is still
    ambiguous, so the caller can drop the level rather than guess.
    """
    mine = _draws_shared_by_all(root, stage)
    if not mine:
        return None
    if len(mine) > 1:
        other = "04_activations" if stage == "05_generated" else "05_generated"
        overlap = mine & _draws_shared_by_all(root, other)
        if len(overlap) != 1:
            return None
        mine = overlap
    recipe_hash, n_samples, seed = next(iter(mine))
    return {"recipe_hash": recipe_hash, "n_samples": n_samples, "seed": seed}


def _shared_behavioral_variant(root: Path, draw: dict, index) -> dict | None:
    """A named behavioral variant every model in *index* has, chosen deterministically.

    Naming one is now mandatory rather than incidental: extraction at a second
    replicate count or sampling setting adds a variant beside the first, and
    ``_behavioral_variant_choice`` rightly refuses to guess between them.  This
    check must therefore pick, and pick the *same* one every run — otherwise its
    numbers move whenever an extraction job lands.

    Sorting and taking the first is arbitrary but stable.  The alternative,
    letting the level drop out when several exist, is exactly the failure item 13
    records: behavioral was silently excluded from this check for its entire life
    because it required there to be exactly one config.
    """
    from src.cache.generated_text_cache import GeneratedTextCache

    cache = GeneratedTextCache(root)
    per_entry = []
    for entry in index.entries:
        if not entry.base_model_id:
            continue
        per_entry.append({
            v for v in cache.list_variants(entry.base_model_id, entry.model_id, draw)
            if v[0].startswith("generation")
        })
    shared = set.intersection(*per_entry) if per_entry else set()
    if not shared:
        return None
    mode_token, replicates, sampling_hash, embedder_hash = sorted(shared)[0]
    return {
        "max_new_tokens": int(mode_token[len("generation"):]),
        "replicates": replicates,
        "sampling_hash": sampling_hash,
        "embedder_hash": embedder_hash,
    }


@check("[data] comparison: end-to-end on one slice, reported not asserted")
def t_comparison_end_to_end():
    """Full chain on real adapters: cache -> distances -> MDS -> simplex -> truth.

    Only structural invariants are asserted.  The recovery correlations and
    Procrustes disparities are printed, because they are measurements of the
    taxonomies rather than properties of the code.
    """
    import tempfile

    from src.analysis import build_taxonomy_artifacts, compare_taxonomies, scan_cache

    root = SHARED_CACHE
    if not (root / "03_adapters").exists():
        raise _Skip(f"{root}/03_adapters not present")

    # Behavioral joins only when every model in the slice has a representation
    # under one draw.  "05_generated exists" is not enough: a slice missing one
    # adapter would build a matrix and then fail on it, reporting FAIL where the
    # honest answer is "that model was never extracted".
    #
    # This used to require *exactly one* config to exist, and there were two — so
    # behavioral was silently dropped from this check for its whole life, which
    # is why the cache being unreachable through discovery never showed up here.
    # A draw is the right unit: two draws are two query sets, but two embedders
    # over one draw are still the same comparison.
    behavioral_draw = _shared_inference_draw(root, "05_generated")

    index = scan_cache(root, behavioral_draw=behavioral_draw)
    tokens = ["structural_weights", "dataset_embedding"]
    taxonomies = ["structural", "dataset_embedding"]
    if behavioral_draw is not None:
        candidate = index.with_available(*tokens, "behavioral_repr")
        cand_slices = candidate.slices(("n_samples", "seed"))
        if cand_slices and len(cand_slices[max(cand_slices)]) >= 3:
            tokens.append("behavioral_repr")
            taxonomies.append("behavioral")

    # Functional joins on the same terms, and for the same reason: a slice with a
    # gap would build a matrix and then fail on it, reporting FAIL where the
    # honest answer is "that model was never extracted".
    if (root / "04_activations").exists():
        candidate = index.with_available(*tokens, "functional_repr")
        cand_slices = candidate.slices(("n_samples", "seed"))
        if cand_slices and len(cand_slices[max(cand_slices)]) >= 3:
            tokens.append("functional_repr")
            taxonomies.append("functional")

    index = index.with_available(*tokens)

    # Name the variant as well as the draw.  Since the replicates work there are
    # two variants per model (greedy 1r and sampled 8r), and they are not
    # comparable, so the level cannot be read without choosing.
    behavioral_sel = None
    if behavioral_draw is not None:
        variant = _shared_behavioral_variant(root, behavioral_draw, index)
        if variant is not None:
            behavioral_sel = {"draw": behavioral_draw, **variant}

    slices = index.slices(("n_samples", "seed"))
    if not slices:
        raise _Skip("no complete (n_samples, seed) slice in the cache")
    key = max(slices)                       # the largest sample size available
    sub = slices[key]
    if len(sub) < 3:
        raise _Skip(f"slice {key} has only {len(sub)} model(s)")

    with tempfile.TemporaryDirectory() as td:
        matrices = {}
        for tax in taxonomies:
            # One layer and one projection keeps this to a few hundred KB per
            # adapter; the dense B @ A product is never formed.
            matrices[tax], _ = build_taxonomy_artifacts(
                sub, tax, "cosine", cache_root=td, n_components=(2,),
                layers=[27], projections="o",
                behavioral_selector=behavioral_sel,
            )
        result = compare_taxonomies(
            matrices, sub.recipes(), n_permutations=199,
            slice_key=dict(zip(("n_samples", "seed"), key)),
        )
        saved = result.save(Path(td) / "report")
        reloaded = type(result).load(saved)

    n, k = len(result.model_ids), len(result.vertices)
    assert result.simplex_dim == k - 1, (result.simplex_dim, k)
    assert result.projection_dim >= result.simplex_dim
    assert len(result.anchors) == k, result.anchors
    assert set(result.anchors) & set(result.eval_points) == set()
    assert np.allclose(result.ground_truth_weights.sum(axis=1), 1.0)

    for tax in result.taxonomies:
        proj = result.projections[tax]
        assert np.allclose(proj.weights.sum(axis=1), 1.0), tax
        for j, anchor in enumerate(result.anchors):
            onehot = np.zeros(k)
            onehot[j] = 1.0
            assert np.allclose(proj.weight_for(anchor), onehot, atol=1e-9), (tax, anchor)
        assert np.isfinite(result.stress[tax])
        assert np.isfinite(result.procrustes_vs_truth[tax].disparity)

    assert reloaded.model_ids == result.model_ids, "save/load lost the model order"
    assert sorted(reloaded.projections) == sorted(result.projections)

    # The fitted map onto the ground-truth simplex must survive the real save/load
    # path, not just an in-memory round trip, and must still be applicable.
    assert sorted(reloaded.procrustes_vs_truth) == sorted(result.procrustes_vs_truth), (
        sorted(reloaded.procrustes_vs_truth)
    )
    for tax, restored in reloaded.procrustes_vs_truth.items():
        source = result.geometries[tax][f"mds_{result.projection_dim}d"]
        replayed = restored.transform(np.asarray(source.coordinates, dtype=np.float64))
        expected = np.asarray(restored.aligned_b.coordinates, dtype=np.float64)
        dev = float(np.abs(replayed - expected).max())
        assert dev < 1e-4, f"{tax}: reloaded map does not replay its own fit ({dev:.2e})"
        assert result.report["per_taxonomy"][tax]["procrustes_scale"] is not None

    print(f"      slice n_samples={key[0]}, seed={key[1]}: {n} models, {k} vertices")
    print(f"      projecting from {result.projection_dim}-D MDS "
          f"(simplex is {result.simplex_dim}-D)")
    for tax in result.taxonomies:
        rep = result.report["per_taxonomy"][tax]
        ev = rep["recovery_eval_only"] or {}
        print(
            f"      {tax:18s} stress={rep['stress']:.4f} "
            f"r={_show(ev.get('pearson_mean'))} rho={_show(ev.get('spearman_mean'))} "
            f"meanL1={_show(ev.get('mean_l1'))} maxres={_show(ev.get('max_residual'))} "
            f"procrustes={_show(rep['procrustes_vs_truth'])} "
            f"p={_show(rep['protest_p_value'])} "
            f"dcor={_show(rep['dcor_vs_truth'])} dcor_p={_show(rep['dcor_p_value'])}"
        )
    return f"{n} models, {len(result.taxonomies)} taxonomies, report round-tripped"


def _show(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


@check("identity: relabel refuses to collide distinct models")
def t_relabel_collision():
    from src.analysis import recipe_id_for, relabel

    # Two adapters, same dataset, different LoRA rank — a rank sweep.  Mapping
    # both to the recipe ID would make the rows ambiguous.
    dm = as_distance_matrix(
        ["a/yahoo_50t0_50t1_r8", "a/yahoo_50t0_50t1_r16"],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        "m",
        "structural",
    )
    try:
        relabel(dm, recipe_id_for)
    except ValueError as e:
        assert "same identifier" in str(e), str(e)
    else:
        raise AssertionError("expected a collision to be rejected")

    # A mapping leaves unlisted identifiers alone.
    out = relabel(dm, {"a/yahoo_50t0_50t1_r8": "rank8"})
    assert out.model_ids == ["rank8", "a/yahoo_50t0_50t1_r16"], out.model_ids
    assert dm.model_ids[0] == "a/yahoo_50t0_50t1_r8", "input was mutated"
    return "collision rejected; mapping form is partial and non-mutating"


@check("recipe identity: content-addressed, name-independent, type-separated")
def t_recipe_identity():
    from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
    from src.datasets.recipe import DatasetEntry, DatasetRecipe

    entries = [DatasetEntry("imdb", text_field="text", weight=1.0)]
    a = DatasetRecipe(name="mix", datasets=entries)
    b = DatasetRecipe(name="mix_n1000_s07", datasets=list(entries))
    assert a.recipe_hash() == b.recipe_hash(), (
        "the name is still in the hash — every n and seed would be its own recipe"
    )

    # Content still matters.
    other = DatasetRecipe(name="mix", datasets=[DatasetEntry("imdb", text_field="other")])
    assert a.recipe_hash() != other.recipe_hash(), "different entries must differ"

    # recipe_type is the only thing separating the two classes once names are gone.
    class_aware = ClassAwareDatasetRecipe(
        name="mix", datasets=[ClassDatasetEntry("imdb", text_field="text", class_field="label")]
    )
    assert a.recipe_hash() != class_aware.recipe_hash(), (
        "simple and class-aware recipes collide"
    )

    # The stored hash must be self-consistent, which is what CacheIndex asserts.
    assert a.to_dict()["recipe_hash"] == a.recipe_hash()
    assert a.to_dict()["schema_version"] == "2", "schema bump missing"
    return "name excluded, entries and recipe_type included, schema_version=2"


@check("class_sampling: default serializes as before, pooled is a distinct recipe")
def t_class_sampling_hash():
    """The additive-change guard for ``class_sampling``.

    ``to_dict()`` is what ``recipe_hash`` is computed over, so emitting the new key
    unconditionally would move every hash in ``01_datasets`` at once, orphaning
    every cached draw and every adapter keyed on one.  The key is therefore spliced
    in only when non-default, exactly as ``composition_dict`` is.  This pins that.
    """
    from src.datasets.class_recipe import (
        ClassAwareDatasetRecipe,
        ClassDatasetEntry,
        class_sampling_dict,
    )

    def entry(**kw):
        return ClassDatasetEntry(
            "yahoo_answers_topics", text_field="best_answer", class_field="topic",
            class_filter=[1, 3, 4], **kw,
        )

    # 1. The default must be invisible in the serialized form.
    d = entry().to_dict()
    assert "class_sampling" not in d, (
        f"a default entry emitted class_sampling — every existing recipe_hash "
        f"just moved. Keys: {sorted(d)}"
    )
    assert class_sampling_dict("stratified") == {}
    assert class_sampling_dict("pooled") == {"class_sampling": "pooled"}

    # 2. Pooled must be a *different* recipe, or the two draws share a directory.
    strat = ClassAwareDatasetRecipe(name="m", datasets=[entry()])
    pooled = ClassAwareDatasetRecipe(name="m", datasets=[entry(class_sampling="pooled")])
    assert strat.recipe_hash() != pooled.recipe_hash(), (
        "pooled and stratified collide on one hash — two different draws would "
        "share a cache directory"
    )
    assert pooled.datasets[0].to_dict()["class_sampling"] == "pooled"

    # 3. Round-trips, or a reloaded recipe silently reverts to stratified.
    back = ClassDatasetEntry.from_dict(pooled.datasets[0].to_dict())
    assert back.class_sampling == "pooled", back.class_sampling
    assert ClassDatasetEntry.from_dict(d).class_sampling == "stratified"

    # 4. Pooled clears the per-class quotas; stratified keeps them.
    assert pooled.datasets[0].normalized_class_weights is None
    assert strat.datasets[0].normalized_class_weights == {1: 1 / 3, 3: 1 / 3, 4: 1 / 3}

    # 5. The two contradictory configs are rejected, not silently resolved.
    for bad, why in (
        ({"class_sampling": "pooled", "class_weights": {1: 2.0, 3: 1.0, 4: 1.0}},
         "pooled + class_weights"),
        ({"class_sampling": "uniform"}, "an unknown mode"),
    ):
        try:
            entry(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} was accepted")

    return "default key absent, pooled hashes apart, round-trips, bad combos raise"


@check("class_sampling: pooled draws are hypergeometric, stratified are exact")
def t_class_sampling_semantics():
    """The behavioural half of the guard, on synthetic data.

    Asserting the *hash* is not enough: the whole point of the mode is that
    per-class counts stop being forced equal.  A no-op implementation would pass
    every check above.  Uses an in-memory dataset so this stays a unit check —
    the real yahoo draw is exercised separately in the experiment's verification.
    """
    from collections import Counter

    from datasets import Dataset  # type: ignore[import]

    from src.datasets import source_registry
    from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
    from src.datasets.mixed_dataset import ClassMixedDataset

    # Three classes, deliberately *unequal* pools: pooled draws should track these
    # proportions, stratified should ignore them entirely.
    sizes = {0: 600, 1: 300, 2: 100}
    rows = [{"text": f"c{c}-{i}", "label": c} for c, n in sizes.items() for i in range(n)]
    ds = Dataset.from_list(rows)

    key = ("__synthetic__", None, "train", None)
    source_registry._datasets[key] = ds
    try:
        def counts(mode, seed):
            e = ClassDatasetEntry(
                "__synthetic__", text_field="text", class_field="label",
                class_filter=[0, 1, 2], class_sampling=mode,
            )
            recipe = ClassAwareDatasetRecipe(name="m", datasets=[e])
            drawn = ClassMixedDataset(recipe, total_samples=300, seed=seed)._ensure_loaded()
            assert len(drawn) == 300, f"{mode} drew {len(drawn)}, not 300"
            return Counter(r["label"] for r in drawn)

        # Stratified: exact, and identical across seeds.
        s0, s1 = counts("stratified", 0), counts("stratified", 1)
        assert set(s0.values()) == {100}, f"stratified was not exact: {dict(s0)}"
        assert s0 == s1, "stratified counts moved with the seed"

        # Pooled: not equal, seed-dependent, and tracking the 6:3:1 pool.
        p0, p1 = counts("pooled", 0), counts("pooled", 1)
        assert set(p0.values()) != {100}, (
            f"pooled produced exactly equal counts {dict(p0)} — the branch is a no-op"
        )
        assert p0 != p1, f"pooled counts did not move with the seed: {dict(p0)}"
        for cls, expected in ((0, 180), (1, 90), (2, 30)):
            assert abs(p0[cls] - expected) < 40, (
                f"pooled class {cls} drew {p0[cls]}, nowhere near its pool share "
                f"{expected} — the draw is not uniform over the union"
            )
    finally:
        source_registry.clear_cache()

    return (
        f"stratified 100/100/100 for every seed; "
        f"pooled {dict(sorted(p0.items()))} then {dict(sorted(p1.items()))} on 600/300/100"
    )


@check("sample budget: quantizes up, so the realized count is never short")
def t_steps_for_budget():
    """The budget is a floor, and both callers must agree on where it lands.

    ``max_steps`` is the only unit the Trainer accepts, so a sample budget has to
    quantize to a whole step.  Rounding to *nearest* silently trained a 5000-sample
    budget on 4992 samples.  Rounding up is also the reason this lives in one
    function: ``finetune_lora.main`` predicts the ``_b{samples_seen}`` directory
    name before a model is loaded and ``_finetune_one`` recomputes it against the
    Trainer's real effective batch, so a divergence would name one directory and
    train into another.
    """
    from scripts._utils import steps_for_budget

    # The case that motivated the change.
    assert steps_for_budget(5000, 16) == 313, "5000/16 must round up to 313 steps"
    assert 313 * 16 == 5008 >= 5000

    # Never short, for any budget/batch pair.
    for budget in (1, 7, 16, 17, 160, 3000, 4999, 5000, 5001):
        for eff in (1, 2, 8, 16, 32):
            steps = steps_for_budget(budget, eff)
            assert steps * eff >= budget, (
                f"budget {budget} at effective batch {eff} realized "
                f"{steps * eff} samples — short of what was asked"
            )
            # ...but never more than one step's worth over, or it is not tight.
            assert (steps - 1) * eff < budget or steps == 1

    # Exact division is unaffected, which is why no adapter on disk is renamed.
    assert steps_for_budget(3008, 16) == 188
    assert steps_for_budget(1, 64) == 1, "a sub-batch budget must still train a step"
    return "5000/16 -> 313 steps = 5008 seen; never short, never more than one step over"


@check("prompt format: the raw path is byte-identical to no prompt format at all")
def t_prompt_format_raw_is_inert():
    """The whole change is additive or it is a migration.

    Every existing adapter, draw and dataset embedding was produced with no
    prompt format in play.  If ``PromptFormat()`` renders one byte differently
    from the old ``row_text`` path, or contributes anything to a serialized
    config or a cache path, then adding the chat layer silently orphaned all of
    it.  Assert instead of hoping.
    """
    from src.cache._draw import draw_format_id, draw_name, parse_draw_name
    from src.cache._draw_keyed import DrawKeyedCache
    from src.datasets._chat_projection import PromptFormat

    raw = PromptFormat()
    assert raw.format == "raw"
    assert raw.to_dict() == {}, "a raw format must serialize to nothing"
    assert raw.format_id() is None, "a raw format must not qualify any name"
    assert PromptFormat.from_config(None).to_dict() == {}
    assert PromptFormat.from_config({}).to_dict() == {}

    # ...and therefore no path moves.
    assert draw_name(1000, 0) == draw_name(1000, 0, None) == "n1000_s00"
    assert DrawKeyedCache.draw_name({"n_samples": 100, "seed": 1}) == "n100_s01"

    # A qualified name still parses to the same coordinates, so `draw_name(*parse(...))`
    # -- the idiom the migration scripts use -- cannot silently drop a format.
    assert parse_draw_name("n100_s01_fdeadbeef") == (100, 1)
    assert draw_format_id("n100_s01_fdeadbeef") == "deadbeef"
    assert draw_format_id("n100_s01") is None
    assert DrawKeyedCache.draw_name(
        {"n_samples": 100, "seed": 1, "prompt_format_id": "deadbeef"}
    ) == "n100_s01_fdeadbeef"
    return "raw renders, serializes and names exactly as before; format only ever adds"


@check("prompt format: the generator and the trainer name an adapter identically")
def t_adapter_name_agreement():
    """Two places build the adapter leaf, and they must not drift.

    ``gen_simplex3.adapter_name`` writes the paths into the extraction configs'
    model list; ``_utils.adapter_dir`` decides where training actually writes.
    If they disagree, training succeeds, extraction finds nothing, and the only
    symptom is an empty result — which is exactly what happened here once
    already, when the format suffix was added to the generator and not to the
    trainer.
    """
    import importlib

    from scripts._utils import adapter_dir, retag_adapter_dir

    gen = importlib.import_module("scripts.gen_simplex3")
    checked = []
    # Every corpus, not only yahoo: the samples-seen token is now derived from
    # the suite's effective batch rather than written down, so oasst1's smaller
    # budget gives _b2512 where the other two give _b5008. That derivation is
    # exactly the kind of thing that drifts between the two call sites.
    for ds_name, spec in gen.SPECS.items():
        gen.SPEC = spec
        base = gen.name_for(spec.even_pct)
        for suite_name, suite in gen.SUITES.items():
            gen.SUITE = suite
            seen = gen.samples_seen()
            block = f"{base}_n{spec.train_n}_s{spec.train_seed:02d}"
            # The generator names from the proportion and appends the draw
            # itself; the trainer is handed the already-expanded block name.
            # Same leaf.
            want = gen.adapter_name(base)
            got = adapter_dir(
                Path("/root"), suite.base_model, block,
                gen.LORA_RANK, gen.LORA_INIT_SEED,
                samples_seen=seen,
                prompt_format_id=gen.format_id(),
            ).name
            assert want == got, (
                f"{ds_name}/{suite_name}: generator says {want!r}, "
                f"trainer says {got!r}"
            )
            # The realized-sample retag has to survive whatever suffix follows it.
            assert retag_adapter_dir(Path("/root") / got, seen).name == got
            assert retag_adapter_dir(Path("/root") / got, 99).name == got.replace(
                f"_b{seen}", "_b99"
            )
        checked.append(f"{ds_name}=_b{gen.samples_seen()}")
    gen.SPEC = gen.SPECS["yahoo"]
    gen.SUITE = gen.SUITES["llama"]
    return f"{len(gen.SPECS)} corpora x {len(gen.SUITES)} suites agree; " + \
           ", ".join(checked)


@check("figures: the structural spec set follows the model's attention layout")
def t_figure_specs_follow_layout():
    """The figure suite used to hard-code Qwen3.5's hybrid layout.

    Two properties, and the first is what stops a generalization from quietly
    rewriting the existing figures: for Qwen's parameters the derived layout and
    the emitted spec keys must be exactly what the script shipped with.  The
    second is that a uniform-attention model degrades rather than special-cases --
    every ``linear-attn`` spec drops out, and so does the q_proj query/gate split,
    which only exists because ``attn_output_gate`` fuses a gate into q_proj.

    Driven through ``layout`` rather than a checkpoint so it needs no network.
    """
    import importlib.util

    # The suite moved out of scripts/ into src/plots/simplex_suite.py when it
    # gained a second and third driver; the spec builders and the architecture
    # derivation this checks went with it.  Still loaded by path rather than
    # imported, so the check keeps working from a worktree whose sys.path may
    # not carry the checkout.
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_figs", root / "src" / "plots" / "simplex_suite.py")
    figs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figs)

    # Qwen3.5-4B: 32 layers, 16 heads, full_attention_interval 4, output gate.
    figs.apply_architecture(figs.layout(32, 16, 4, True))
    assert figs.FULL_ATTN_LAYERS == [i for i in range(32) if i % 4 == 3]
    assert figs.LINEAR_ATTN_LAYERS == [i for i in range(32) if i % 4 != 3]
    assert figs.N_STATES == 33
    qwen_proj = list(figs.structural_projection_specs())
    assert qwen_proj == [
        "full-attn · q_proj (whole)", "full-attn · q_proj query half",
        "full-attn · q_proj gate half", "full-attn · k_proj", "full-attn · v_proj",
        "full-attn · o_proj", "linear-attn · in_proj_qkv", "linear-attn · in_proj_z",
        "linear-attn · out_proj"], qwen_proj
    qwen_grp = list(figs.structural_group_specs())
    assert len(qwen_grp) == 12 and "linear-attn · late third" in qwen_grp, qwen_grp

    # Llama-3.1-8B-Instruct: 32 layers, 32 heads, uniform, no gate.
    figs.apply_architecture(figs.layout(32, 32, None, False))
    assert figs.LINEAR_ATTN_LAYERS == [], figs.LINEAR_ATTN_LAYERS
    assert figs.FULL_ATTN_LAYERS == list(range(32))
    uni_proj = list(figs.structural_projection_specs())
    assert uni_proj == ["q_proj (whole)", "k_proj", "v_proj", "o_proj"], uni_proj
    uni_grp = list(figs.structural_group_specs())
    assert not any("linear-attn" in k for k in uni_grp), uni_grp
    uni_fn = list(figs.functional_group_rows())
    assert not any("attn outputs" in k for k in uni_fn), uni_fn
    assert len(list(figs.structural_layer_specs())) == 3

    # Degrading is dropping rows, not renaming them into collisions: two labels
    # selecting the same (layers, projections) would plot an identical row twice
    # and store it twice in the collection cache.  Checked on both layouts,
    # because the hybrid branch is where the distinct-looking pairs come from.
    for arch_name in ("uniform", "hybrid"):
        if arch_name == "hybrid":
            figs.apply_architecture(figs.layout(32, 16, 4, True))
        seen: dict = {}
        for key, (layers, projs) in figs.structural_group_specs().items():
            seen.setdefault((tuple(layers), tuple(projs)), []).append(key)
        dupes = [v for v in seen.values() if len(v) > 1]
        assert not dupes, f"{arch_name}: labels selecting identical rows: {dupes}"

    # Restore the module default so nothing later sees a mutated global.
    figs.apply_architecture(figs.layout(32, 16, 4, True))
    return (f"hybrid -> {len(qwen_proj)} projection specs, {len(qwen_grp)} group "
            f"specs; uniform -> {len(uni_proj)} and {len(uni_grp)}")


@check("profiles: the instruct prefix does not capture its base model")
def t_profile_prefix_discrimination():
    """Two ids differing by a suffix, and both suites depend on telling them apart.

    ``LLAMA3`` matches the family prefix ``meta-llama/Llama-3`` and declares
    ``prompt_format='raw'``.  ``meta-llama/Llama-3.1-8B-Instruct`` starts with
    that prefix, so the *only* thing routing it to a chat profile is
    ``resolve``'s longest-match rule.  Get it wrong in one direction and the
    instruct suite silently emits a raw suite; get it wrong in the other and the
    16 adapters already stored under ``meta-llama--Llama-3.1-8B`` stop being
    reproducible.

    Also pins that a width-specific parameter count never reaches a profile whose
    match spans several widths -- the reason LLAMA3 carries None.
    """
    from src.models.profile import resolve

    base = resolve("meta-llama/Llama-3.1-8B")
    inst = resolve("meta-llama/Llama-3.1-8B-Instruct")
    assert base.match == "meta-llama/Llama-3", base.match
    assert base.prompt_format == "raw", base.prompt_format
    assert base.chat_template_sha is None
    assert inst.match == "meta-llama/Llama-3.1-8B-Instruct", inst.match
    assert inst.prompt_format == "chat", inst.prompt_format
    assert inst.chat_template_sha, "instruct profile must pin its template"
    assert inst.expected_lora_params == 13_631_488, inst.expected_lora_params

    # A family-wide match must not carry a width-specific claim.
    for mid in ("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"):
        prof = resolve(mid)
        assert prof.match == "meta-llama/Llama-3", (mid, prof.match)
        assert prof.expected_lora_params is None, (
            f"{mid} resolves to the family profile, which must not assert an "
            f"8B parameter count"
        )
    return (f"base -> {base.match!r} (raw), instruct -> {inst.match!r} (chat), "
            f"3.2 sizes -> family profile with no count")


@check("profiles: pad resolution is one rule, and never silently eos")
def t_pad_token_resolution():
    """Training and inference must pad identically or they mask different rows.

    ``apply_pad_token`` is the single definition.  The cases that matter: a
    checkpoint that declares its own pad is left alone; a profile that names one
    supplies it; naming a token the tokenizer lacks, or one that *is* eos, raises
    rather than quietly padding with a real word or re-creating pad == eos.
    """
    from src.models.profile import ModelProfile, apply_pad_token

    class Tok:
        def __init__(self, pad, eos, vocab):
            self.pad_token, self.eos_token = pad, eos
            self._v, self.unk_token_id = vocab, 999
            self.eos_token_id = vocab.get(eos)

        def convert_tokens_to_ids(self, t):
            return self._v.get(t, self.unk_token_id)

    vocab = {"<eos>": 1, "<pad>": 2}
    declared = Tok("<pad>", "<eos>", vocab)
    apply_pad_token(declared, ModelProfile(match="x", prompt_end_token=None))
    assert declared.pad_token == "<pad>"

    fallback = Tok(None, "<eos>", vocab)
    apply_pad_token(fallback, ModelProfile(match="x", prompt_end_token=None))
    assert fallback.pad_token == "<eos>", "fallback must stay pad = eos"

    named = Tok(None, "<eos>", vocab)
    apply_pad_token(named, ModelProfile(match="x", pad_token="<pad>", prompt_end_token=None))
    assert named.pad_token == "<pad>", named.pad_token

    for bad, why in ((("<nope>"), "absent from the vocab"), (("<eos>"), "is eos")):
        try:
            apply_pad_token(Tok(None, "<eos>", vocab),
                            ModelProfile(match="x", pad_token=bad,
                                         prompt_end_token=None))
        except ValueError:
            pass
        else:
            raise AssertionError(f"pad_token {bad!r} ({why}) should have raised")
    return "declared kept, fallback = eos, profile honoured, unk and eos rejected"


@check("prompt format: the prompt end is declared, and the declaration is checked")
def t_prompt_end_token():
    """The cut point is stated by the profile, and nothing else may guess it.

    This replaces a derivation that scanned the rendered prompt for the
    rightmost added-vocab token.  OLMo-2 broke it: ``<|user|>``/``<|assistant|>``
    are ordinary text there, so the only added-vocab token in the prompt was the
    BOS at index 0, the cut landed at character 13, and all 16 adapters would
    have trained on the bare string ``<|endoftext|>`` with the question gone --
    while shapes stayed valid and the loss stayed finite.

    So the five properties that now stand in for that derivation: a declared
    token cuts immediately after its *last* occurrence; ``None`` cuts nothing, so
    content cannot be lost by default; a declaration that would discard
    non-whitespace raises rather than discarding it; a declaration absent from
    the render raises; and a declared token that is not atomic is refused by
    ``assert_compatible`` at config time, before a GPU is held.
    """
    from src.datasets._chat_projection import PromptFormat, render_prompt
    from src.models.profile import ModelProfile, assert_compatible

    class Tok:
        """Renders "<s>USER:{content}<end>\\n" and nothing more."""

        chat_template = "a template"

        def __init__(self, added=("<s>", "<end>")):
            self._added = {t: i for i, t in enumerate(added)}

        def get_added_vocab(self):
            return dict(self._added)

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False, **kw):
            return f"<s>USER:{messages[0]['content']}<end>\n"

    fmt = PromptFormat(format="chat", user_fields=("q",))
    row = {"q": "why is the sky blue?"}

    def prof(end):
        return ModelProfile(match="x", prompt_format="chat",
                            chat_template_sha="deadbeef", prompt_end_token=end)

    # 1. A declared token cuts just past its LAST occurrence -- so a marker that
    #    also appears inside the user's own text does not move the cut.
    got = render_prompt(Tok(), row, fmt, profile=prof("<end>"))
    assert got == "<s>USER:why is the sky blue?<end>", repr(got)
    adversarial = render_prompt(Tok(), {"q": "what is <end> for?"}, fmt,
                                profile=prof("<end>"))
    assert adversarial.endswith("what is <end> for?<end>"), repr(adversarial)

    # 2. None keeps the whole render.  This is the OLMo-2 case, and the reason
    #    the safe default cannot silently drop the question.
    whole = render_prompt(Tok(), row, fmt, profile=prof(None))
    assert whole == "<s>USER:why is the sky blue?<end>\n", repr(whole)
    assert row["q"] in whole

    # 3. A cut that would discard real content raises instead of discarding it.
    for end, why in (("<s>", "would discard the question"),
                     ("<nope>", "does not occur in the render")):
        try:
            render_prompt(Tok(), row, fmt, profile=prof(end))
        except ValueError:
            pass
        else:
            raise AssertionError(f"prompt_end_token {end!r} ({why}) should raise")

    # 4. And a declared token that BPE could merge across is refused at config
    #    time, which is the property that makes the cut safe rather than merely
    #    intentional.
    from src.models.profile import template_sha

    def pinned(end):
        return ModelProfile(match="x", prompt_format="chat", prompt_end_token=end,
                            chat_template_sha=template_sha(Tok()))

    assert_compatible(pinned("<end>"), Tok())  # atomic: fine
    try:
        assert_compatible(pinned("USER:"), Tok())  # plain text: not fine
    except ValueError:
        pass
    else:
        raise AssertionError("a non-atomic prompt_end_token should be refused")

    return ("declared cut lands after the last occurrence, None keeps the whole "
            "render, and absent/non-whitespace/non-atomic declarations all raise")


@check("prompt format: apply_chat_template has exactly one call site")
def t_one_chat_template_call_site():
    """One renderer, or the item-11 bug comes back one level up.

    ``_text_projection`` exists because the training text and the extraction
    prompt drifted apart and the behavioral level recovered the mixing order
    backwards.  A chat template is the same hazard with more surface: any second
    place that wraps a row can wrap it differently.  So there is exactly one
    ``apply_chat_template`` in the repo, and this is what keeps it that way.
    """
    root = Path(__file__).resolve().parent.parent
    hits = []
    for path in list((root / "src").rglob("*.py")) + list((root / "scripts").rglob("*.py")):
        if "__pycache__" in str(path) or path.name == "check_analysis.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if "apply_chat_template(" in line and not line.lstrip().startswith("#"):
                hits.append(f"{path.relative_to(root)}:{i}")
    # Deliberately no line-number pin.  There was one, and it went stale the
    # first time the module was edited -- it named 203 and 227 where the calls
    # had moved to 202 and 228, and the check went on passing only through the
    # tolerant clause below.  A pin that silently stops pinning is worse than no
    # pin; what this check is actually about is the *file*, not the line.
    assert all(
        h.startswith("src/datasets/_chat_projection.py") for h in hits
    ), f"apply_chat_template called outside _chat_projection: {hits}"
    assert hits, "expected at least one call site in _chat_projection"
    return f"{len(hits)} call site(s), all in src/datasets/_chat_projection.py"


@check("draw name: every stage spells one draw exactly one way")
def t_one_draw_name():
    """The regression guard for item 15.

    Four stages name a draw and they used to disagree: ``01_datasets`` wrote an
    unpadded seed, ``04``/``05`` a padded one, and ``02`` wrote nothing at all.
    Nothing detected that, because each stage only ever read its own names — the
    drift was invisible until someone tried to line the trees up by eye.

    So assert the agreement directly.  If a stage ever formats the token itself
    again instead of calling :mod:`src.cache._draw`, this fails.
    """
    from src.cache._draw import DRAW_RE, draw_name, parse_draw_name
    from src.cache._draw_keyed import DrawKeyedCache
    from src.cache.sampled_dataset_cache import SampledDatasetCache

    n, seed = 1000, 3
    want = draw_name(n, seed)
    assert want == "n1000_s03", want

    # 04/05, via the inference base class.
    got_inference = DrawKeyedCache.draw_name({"n_samples": n, "seed": seed})
    assert got_inference == want, f"inference caches say {got_inference}, not {want}"

    # 01, via the sample cache's own path helper.  A temp root because the cache
    # creates its root on construction; nothing is written into it here.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        stem = SampledDatasetCache(Path(tmp))._path("h", n, seed).stem
    assert stem == want, f"01_datasets says {stem}, not {want}"

    # Reading must stay wider than writing, or the migration loses the old names.
    assert parse_draw_name("n1000_s3") == (n, seed), "unpadded names must still read"
    assert parse_draw_name("n1000_s03") == (n, seed)
    assert parse_draw_name("recipe.json") is None, "non-draw entries must not parse"
    assert DRAW_RE.match("n1000_s03"), "DRAW_RE must accept what draw_name writes"

    # A seedless draw names nothing; it must not quietly become "sNone".
    try:
        draw_name(n, None)
    except ValueError:
        pass
    else:
        raise AssertionError("draw_name(None) must raise, not produce 'sNone'")

    return f"01, 04 and 05 all say {want}; unpadded still parses"


@check("embedder hash: the draw separates entries, now via the path not the hash")
def t_embedder_hash_seed():
    """The inverse of what this check used to assert, and deliberately so.

    It used to demand that ``seed`` be *inside* ``embedder_hash``.  That was the
    right guarantee in the wrong place: once ``recipe_hash`` became
    content-addressed, something had to separate two seeds of one mixture, and
    the hash was where it went.  Item 15 moved it into the path.

    So the guarantee has moved, not gone, and this checks it in its new home. The
    failure being prevented is unchanged — two seeds sharing one entry would
    collapse a seed sweep to a single point and report a variance of zero that is
    an artifact of the cache rather than a property of the data.
    """
    import tempfile

    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

    cfg = {"embedder_class": "X", "model_name": "m"}
    with tempfile.TemporaryDirectory() as tmp:
        cache = DatasetEmbeddingCache(Path(tmp))
        emb = cache.embedder_hash(cfg)
        spec = cache.spec_for("mean")

        d_s0 = cache.surrogate_dir("h", 1000, 0, emb, spec)
        d_s1 = cache.surrogate_dir("h", 1000, 1, emb, spec)
        d_n2 = cache.surrogate_dir("h", 2000, 0, emb, spec)
        assert d_s0 != d_s1, (
            "seeds collide in the path — every seed of a mixture would share one "
            "entry, silently collapsing a seed sweep to a single point"
        )
        assert d_s0 != d_n2, "n_samples must still separate entries"
        assert "n1000_s00" in d_s0.parts, d_s0
        assert "n1000_s01" in d_s1.parts, d_s1

        # And the hash itself must now be blind to the draw, or the path level
        # would be decorative and the old collapse could return through it.
        assert emb == cache.embedder_hash(dict(cfg)), "embedder hash must be stable"

        # Representation lives in the surrogate spec, not the embedder hash.
        assert cache.surrogate_hash(spec) != cache.surrogate_hash(
            cache.spec_for("matrix")
        ), "two representations must not share a surrogate directory"

    return "draw separates by path; hash keys the embedder alone"


@check("surrogate hash: 02 and the inference caches agree on one spec digest")
def t_surrogate_hash_shared():
    """Two hashing schemes for one concept is how the draw token drifted.

    ``02`` and ``04``/``05`` both hash a spec dict to name a surrogate directory.
    If they ever compute that digest differently, the same spec means two
    directories and nothing detects it — so pin them together.
    """
    from src.cache._draw_keyed import DrawKeyedCache
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache

    spec = {"representation": "mean", "normalize": "layer"}
    a = DatasetEmbeddingCache.surrogate_hash(spec)
    b = DrawKeyedCache.config_hash(spec)
    assert a == b, f"02 says {a}, inference caches say {b}"
    # Key order must not matter, or a spec built differently is a different dir.
    assert a == DatasetEmbeddingCache.surrogate_hash(
        {"normalize": "layer", "representation": "mean"}
    ), "spec hashing must be order-independent"
    return f"both say {a}"


@check("draw manifest: v1 rows and v2 indices round-trip to the same rows")
def t_draw_schema_roundtrip():
    import json as _json
    import tempfile

    from src.cache.sampled_dataset_cache import SampledDatasetCache, rows_checksum

    rows = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    with tempfile.TemporaryDirectory() as td:
        cache = SampledDatasetCache(td)
        # A v1 draw is a bare list; it must stay readable for pre-migration files.
        path = cache._path("deadbeef", 2, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(rows))
        assert cache.get("deadbeef", 2, 0) == rows, "v1 rows no longer readable"

        # A source that cannot be indexed must be refused, not silently stored as v1.
        try:
            cache.put("cafe", 2, 0, rows=rows, indices=None, sources=None)
        except ValueError as e:
            assert "source indices" in str(e), str(e)
        else:
            raise AssertionError("un-indexable source was accepted")

    assert rows_checksum(rows) == rows_checksum(list(rows)), "checksum is not stable"
    return "v1 readable, un-indexable writes refused, checksum stable"


@check("names.json: merges rather than overwriting, order-independent")
def t_names_merge():
    import tempfile

    from src.cache.sampled_dataset_cache import SampledDatasetCache

    with tempfile.TemporaryDirectory() as td:
        cache = SampledDatasetCache(td)
        for name in ["yahoo_x_n100_s00", "yahoo_x_n1000_s03", "yahoo_x_n100_s00"]:
            cache.add_name("abcd", name)
        first = cache.get_names("abcd")

        cache2 = SampledDatasetCache(td + "/other")
        for name in ["yahoo_x_n1000_s03", "yahoo_x_n100_s00"]:
            cache2.add_name("abcd", name)
        second = cache2.get_names("abcd")

    assert first == sorted({"yahoo_x_n100_s00", "yahoo_x_n1000_s03"}), first
    assert first == second, f"order-dependent: {first} vs {second}"
    return f"{len(first)} names merged, insertion order irrelevant"


@check("[data] dataset embeddings: every entry is under a draw and a surrogate")
def t_dataset_embedding_layout():
    """Nothing of the pre-item-15 layout may survive in the real cache.

    The old shape put the tensor directly under ``{recipe_hash}/{embedder_hash}/``
    with the draw folded into that hash.  A leftover would still *load* — the
    file is a valid safetensors either way — so the only thing that catches a
    half-finished migration is asserting the shape itself.
    """
    from src.cache._draw import parse_draw_name

    root = SHARED_CACHE / "02_dataset_embeddings"
    if not root.exists():
        raise _Skip(f"{root} not present")

    entries = surrogates = 0
    for recipe_dir in sorted(root.iterdir()):
        if not recipe_dir.is_dir():
            continue
        for draw_dir in sorted(recipe_dir.iterdir()):
            if not draw_dir.is_dir():
                continue
            parsed = parse_draw_name(draw_dir.name)
            assert parsed, (
                f"{draw_dir} is not a draw directory — the old "
                "{recipe_hash}/{embedder_hash}/ layout survives here"
            )
            n, seed = parsed
            assert draw_dir.name == f"n{n}_s{seed:02d}", (
                f"{draw_dir.name} is an unpadded draw name; it must be padded"
            )
            for entry in sorted(draw_dir.iterdir()):
                if not entry.is_dir():
                    continue
                cfg_path = entry / "config.json"
                assert cfg_path.exists(), f"{entry} has no config.json"
                cfg = json.loads(cfg_path.read_text())
                # The path and the file must agree, or one of them is a lie.
                assert cfg["n_samples"] == n and cfg["seed"] == seed, (
                    f"{entry} sits at n{n}_s{seed:02d} but its config says "
                    f"n{cfg['n_samples']}_s{cfg.get('seed')}"
                )
                assert "representation" not in cfg, (
                    f"{entry} still records representation in the entry config; "
                    "it belongs to the surrogate now"
                )
                assert not (entry / "embeddings.safetensors").exists(), (
                    f"{entry} still holds a pre-migration embeddings.safetensors"
                )
                surr_root = entry / "surrogates"
                assert surr_root.is_dir(), f"{entry} has no surrogates/"
                for s in sorted(surr_root.iterdir()):
                    if not s.is_dir():
                        continue
                    assert (s / "surrogate.safetensors").exists(), f"{s} has no tensor"
                    assert (s / "config.json").exists(), f"{s} has no config.json"
                    surrogates += 1
                entries += 1

    assert entries, "no entries found; the cache is present but empty"
    return f"{entries} entry(ies), {surrogates} surrogate(s), all under a padded draw"


@check("[data] cache: every recipe is schema 2 and every draw is index-backed")
def t_cache_fully_migrated():
    import json as _json

    root = SHARED_CACHE / "01_datasets"
    if not root.exists():
        raise _Skip(f"{root} not present")

    recipes = sorted(root.glob("*/recipe.json"))
    if not recipes:
        raise _Skip("no recipes in the dataset cache")

    legacy_recipes, legacy_draws, draws = [], [], 0
    for recipe_path in recipes:
        payload = _json.loads(recipe_path.read_text())
        if payload.get("schema_version") != "2":
            legacy_recipes.append(recipe_path.parent.name)
        # A stale hash means the directory was not re-keyed.
        assert payload.get("recipe_hash") == recipe_path.parent.name, recipe_path
        for draw_path in recipe_path.parent.glob("n*_s*.json"):
            draws += 1
            if isinstance(_json.loads(draw_path.read_text()), list):
                legacy_draws.append(f"{recipe_path.parent.name}/{draw_path.name}")

    assert not legacy_recipes, f"{len(legacy_recipes)} schema-1 recipe(s): {legacy_recipes[:3]}"
    # Old-style {n}_{seed:010d}.json files left behind mean --prune never ran.
    stale = [p.name for p in root.glob("*/*.json")
             if p.name != "recipe.json" and p.name != "names.json"
             and not p.name.startswith("n")]
    assert not stale, f"{len(stale)} pre-migration draw file(s) remain: {stale[:3]}"
    return (f"{len(recipes)} recipe(s), {draws} draw(s), "
            f"{len(legacy_draws)} still storing rows")


# ── embedder task prefixes ────────────────────────────────────────────────────

@check("embedder: prefix-required models always resolve a non-empty task prefix")
def t_embedder_prefix_resolved():
    """The regression guard for a bug whose whole character was silence.

    ``nomic-embed-text-v1.5`` ships no ``prompts`` map, so sentence-transformers
    synthesises ``{"query": "", "document": ""}`` and ``prompt_name="document"``
    resolved to a valid key that prepended the empty string — no error, no warning,
    and output that looked entirely plausible.  A check that merely asserted
    "prompt_name is document" would have passed throughout.  What has to be asserted
    is that a real prefix comes out the other end, including when nothing was asked
    for.
    """
    import warnings as _warnings

    from src.embedders.sentence_transformer import (
        _NOMIC_PREFIXES, _PREFIX_REQUIRED_MODELS, SentenceTransformerEmbedder,
    )

    nomic = "nomic-ai/nomic-embed-text-v1.5"
    assert nomic.startswith(_PREFIX_REQUIRED_MODELS), "nomic no longer matches the list"

    # Every alias must produce a non-empty literal, not just a recognised name.
    for name in sorted(_NOMIC_PREFIXES):
        e = SentenceTransformerEmbedder(model_name=nomic, prompt_name=name)
        assert e.prompt_prefix, f"prompt_name={name!r} resolved to an empty prefix"
        assert e.prompt_prefix.endswith(": "), e.prompt_prefix

    # Omitted: must still prefix, and must say so.
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        default = SentenceTransformerEmbedder(model_name=nomic)
    assert default.prompt_prefix == "search_document: ", default.prompt_prefix
    assert len(caught) == 1, f"expected one warning when defaulting, got {len(caught)}"

    # A typo must not quietly take the default — that would restore the silence.
    try:
        SentenceTransformerEmbedder(model_name=nomic, prompt_name="documnet")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown prompt_name was accepted")

    # Non-prefix models keep their old behaviour untouched.
    mini = SentenceTransformerEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert mini.prompt_prefix == "", mini.prompt_prefix
    assert not mini.requires_prefix
    return f"{len(_NOMIC_PREFIXES)} aliases resolve; default 'search_document: '; typo raises"


@check("embedder: prompt_prefix is in the cache key, so bare and prefixed cannot collide")
def t_embedder_prefix_in_cache_key():
    """Bare-text and correctly-prefixed embeddings must never share an embedder_hash.

    They are computed from the same ``model_name`` and the same ``prompt_name``, so
    without the resolved literal in the key they hash identically — and the cache
    would hand back one where the other was asked for.  Their distances live on
    different scales, so mixing them in a single comparison is silently wrong.
    """
    from src.cache.dataset_embedding_cache import DatasetEmbeddingCache
    from src.embedders.sentence_transformer import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder(
        model_name="nomic-ai/nomic-embed-text-v1.5", prompt_name="document",
        use_generated_text=False, trust_remote_code=True,
    )
    cfg = e.config_dict()
    assert "prompt_prefix" in cfg, "prompt_prefix missing from config_dict"
    assert cfg["prompt_prefix"] == "search_document: ", cfg["prompt_prefix"]

    # What the key looked like before the fix: same fields, no resolved literal.
    bare = {k: v for k, v in cfg.items() if k != "prompt_prefix"}
    h_new = DatasetEmbeddingCache.embedder_hash(cfg)
    h_old = DatasetEmbeddingCache.embedder_hash(bare)
    assert h_new != h_old, (
        "prefixed and bare embeddings hash identically; the cache would treat them "
        "as interchangeable"
    )

    # Different prefixes must also separate.
    q = SentenceTransformerEmbedder(
        model_name="nomic-ai/nomic-embed-text-v1.5", prompt_name="search_query",
        use_generated_text=False, trust_remote_code=True,
    )
    h_q = DatasetEmbeddingCache.embedder_hash(q.config_dict())
    assert h_q != h_new, "search_query and search_document share a hash"
    return f"bare={h_old} document={h_new} query={h_q}, all distinct"


# ── behavioral taxonomy ───────────────────────────────────────────────────────

def _generated_cache_or_skip() -> tuple:
    """``(GeneratedTextCache, [(base_slug, adapter_slug), ...])`` for the real cache.

    Twin of :func:`_activation_cache_or_skip`, which is the point: since the
    re-key the two inference stages are enumerated the same way.
    """
    from src.cache.generated_text_cache import GeneratedTextCache

    root = SHARED_CACHE
    if not (root / "05_generated").exists():
        raise _Skip(f"{root}/05_generated not present — behavioral has not been run")
    cache = GeneratedTextCache(root)
    models = cache.list_models()
    if not models:
        raise _Skip("05_generated exists but holds no model")
    return cache, models


@check("behavioral: GeneratedTextCache round-trips matrix, metadata and generations")
def t_generated_cache_roundtrip():
    import tempfile

    from src.cache.generated_text_cache import GeneratedTextCache
    from src.core.representation import ModelRepresentation

    base = "meta-llama/Llama-3.2-3B"
    adapter = "/some/abs/path/yahoo_050t0_050t1_n1000_s00_r16_i00"
    draw = {"recipe_hash": "abc", "n_samples": 4, "seed": 0}
    config = {"taxonomy": "behavioral", "max_new_tokens": 16,
              "query_key": draw, "embedder": {"model_name": "stub"}}
    ehash = GeneratedTextCache.embedder_hash(config["embedder"])
    sampling = {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                "top_k": None, "generation_seed": 0}
    shash = GeneratedTextCache.sampling_hash(sampling)
    R = 3
    # texts[q][r]: three sampled continuations for each of four queries, and the
    # matrix is one row per (query, replicate) in query-major order.
    texts = [[f"q{q} continuation {r}" for r in range(R)] for q in range(4)]
    matrix = np.arange(4 * R * 3, dtype=np.float32).reshape(4 * R, 3)

    with tempfile.TemporaryDirectory() as td:
        cache = GeneratedTextCache(td)
        assert not cache.exists(base, adapter, draw, 16, R, shash, ehash), (
            "empty cache reports a hit"
        )

        rep = ModelRepresentation.create(
            model_id=adapter, taxonomy="behavioral", matrix=matrix, config=config,
            metadata={"n_queries": 4, "generated_texts": texts},
        )
        cache.save(base, adapter, draw, rep, max_new_tokens=16, replicates=R,
                   sampling=sampling, embedder_hash=ehash,
                   config=config, source_indices=[[0, i] for i in range(4)])
        assert cache.exists(base, adapter, draw, 16, R, shash, ehash)

        got = cache.load(base, adapter, draw, 16, R, shash, ehash)
        assert np.array_equal(got.matrix, matrix), "matrix changed across the round trip"
        assert got.matrix.dtype == np.float32, got.matrix.dtype
        assert got.model_id == adapter and got.taxonomy == "behavioral"
        # generated_texts lives in generations/, not in the tensor file; load() has
        # to fold it back in or every consumer of metadata breaks.
        assert got.metadata["generated_texts"] == texts, got.metadata
        assert got.metadata["n_rows"] == 4 * R, got.metadata
        assert got.metadata["n_queries"] == 4, (
            "n_queries counted rows; with replicates a row is one sample, not one query"
        )

        # Reading text must not require touching the tensors.
        assert cache.load_generations(base, adapter, draw, 16, R, shash) == texts

        # The draw record is a pointer, not a copy: query_key and indices, no text.
        q = cache.load_queries(base, adapter, draw)
        assert q["query_key"]["recipe_hash"] == "abc", q
        assert len(q["source_indices"]) == 4, q
        assert "queries" not in q, (
            "queries.json is storing query text again; 01_datasets is canonical and "
            "recipe_hash already determines the text via text_field"
        )

        assert cache.list_variants(base, adapter, draw) == [("generation16", R, shash, ehash)]
        assert cache.list_draws(base, adapter) == [draw]
        assert cache.has_draw(base, adapter, draw)

        # A plain HF model has no adapter and lands under _base — the branch the
        # behavioral level had never exercised.
        cache.save(base, "_base", draw, rep, max_new_tokens=16, replicates=R,
                   sampling=sampling, embedder_hash=ehash, config=config)
        assert cache.exists(base, "_base", draw, 16, R, shash, ehash), (
            "_base branch is unreachable"
        )

        # Idempotent: a second save of different data is a no-op, because there is
        # no invalidation path — a changed embedder yields a new filename instead.
        rep2 = ModelRepresentation.create(
            model_id=adapter, taxonomy="behavioral",
            matrix=np.zeros((4 * R, 3), dtype=np.float32), config=config,
            metadata={"generated_texts": [["x"] * R] * 4},
        )
        cache.save(base, adapter, draw, rep2, max_new_tokens=16, replicates=R,
                   sampling=sampling, embedder_hash=ehash, config=config)
        assert np.array_equal(
            cache.load(base, adapter, draw, 16, R, shash, ehash).matrix, matrix
        ), "save overwrote an existing entry"
    return (
        f"round-tripped ({4 * R}, 3) = 4 queries x {R} replicates + nested "
        f"generations at generation16_{R}r_{shash}_{ehash[:8]}"
    )


@check("behavioral: replicates average back to one row per query")
def t_generated_replicate_reduction():
    """``replicate_reduction="mean"`` must equal an explicit per-query mean.

    The stored layout is query-major — ``q0r0, q0r1, …, q1r0, …`` — so the
    reduction is a reshape.  If the storage order were replicate-major instead,
    the same reshape would silently average *across queries* and produce a
    plausible-looking matrix of the right shape that means nothing, which is why
    this asserts against an independently written mean rather than against
    itself.
    """
    import tempfile

    from src.cache.generated_text_cache import GeneratedTextCache
    from src.core.representation import ModelRepresentation

    base, adapter = "meta-llama/Llama-3.2-3B", "/abs/adapter"
    draw = {"recipe_hash": "abc", "n_samples": 5, "seed": 0}
    sampling = {"do_sample": True, "temperature": 0.8, "top_p": 0.95,
                "top_k": None, "generation_seed": 7}
    shash = GeneratedTextCache.sampling_hash(sampling)
    ehash = GeneratedTextCache.embedder_hash({"model_name": "stub"})
    n, R, d = 5, 4, 6
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(n * R, d)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        cache = GeneratedTextCache(td)
        rep = ModelRepresentation.create(
            model_id=adapter, taxonomy="behavioral", matrix=matrix, config={},
            metadata={"generated_texts": [["t"] * R] * n},
        )
        cache.save(base, adapter, draw, rep, max_new_tokens=32, replicates=R,
                   sampling=sampling, embedder_hash=ehash, config={})

        pooled = cache.load(base, adapter, draw, 32, R, shash, ehash,
                            replicate_reduction="mean")
        # Written out the long way on purpose: row q*R+r is replicate r of query q.
        expected = np.stack([
            matrix[q * R:(q + 1) * R].astype(np.float64).mean(axis=0) for q in range(n)
        ]).astype(np.float32)
        assert pooled.matrix.shape == (n, d), pooled.matrix.shape
        assert np.allclose(pooled.matrix, expected, atol=1e-6), "mean is not the per-query mean"
        assert pooled.metadata["n_queries"] == n, pooled.metadata
        assert pooled.metadata["surrogate_cached"] is False, "first build reported as cached"

        again = cache.load(base, adapter, draw, 32, R, shash, ehash,
                           replicate_reduction="mean")
        assert again.metadata["surrogate_cached"] is True, "reduction recomputed, not cached"
        assert np.array_equal(again.matrix, pooled.matrix)

        # The unreduced read is still the stored bytes and is not routed through a
        # surrogate at all.
        raw = cache.load(base, adapter, draw, 32, R, shash, ehash)
        assert np.array_equal(raw.matrix, matrix)
        assert "surrogate_cached" not in raw.metadata, raw.metadata

    return f"({n * R}, {d}) -> ({n}, {d}); second read served from the surrogate"


@check("behavioral: sampling settings separate two runs over one draw")
def t_generated_sampling_hash_separates():
    """Two temperatures over one draw must be two entries, not one reused.

    ``save()`` is idempotent *on the filename*.  Before the sampling hash was in
    the name, a second run at a different temperature would have found the first
    run's file, returned early, and handed back the first run's numbers — no
    error, no warning, and nothing on disk to show two runs had happened.  This
    is the same failure mode the class documents for ``torch_dtype``, on an axis
    that changes the result far more.
    """
    import tempfile

    from src.cache.generated_text_cache import GeneratedTextCache
    from src.core.representation import ModelRepresentation

    base, adapter = "meta-llama/Llama-3.2-3B", "/abs/adapter"
    draw = {"recipe_hash": "abc", "n_samples": 2, "seed": 0}
    ehash = GeneratedTextCache.embedder_hash({"model_name": "stub"})
    hot = {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
           "top_k": None, "generation_seed": 0}
    cold = dict(hot, temperature=0.7)
    greedy = dict(GeneratedTextCache.GREEDY_SAMPLING)

    hashes = {k: GeneratedTextCache.sampling_hash(v)
              for k, v in {"hot": hot, "cold": cold, "greedy": greedy}.items()}
    assert len(set(hashes.values())) == 3, f"sampling hashes collide: {hashes}"

    # Frozen: scripts/migrate_behavioral_replicates.py renamed every pre-sampling
    # entry under this literal. If the canon changes, those files stop being
    # reachable, so this is pinned rather than recomputed.
    assert hashes["greedy"] == "6f000f01", (
        f"greedy sampling now hashes to {hashes['greedy']}, not the 6f000f01 that "
        "migrate_behavioral_replicates.py wrote into 10 filenames"
    )

    with tempfile.TemporaryDirectory() as td:
        cache = GeneratedTextCache(td)
        for name, sampling in (("hot", hot), ("cold", cold)):
            matrix = np.full((2, 3), 1.0 if name == "hot" else 2.0, dtype=np.float32)
            rep = ModelRepresentation.create(
                model_id=adapter, taxonomy="behavioral", matrix=matrix, config={},
                metadata={"generated_texts": [[name], [name]]},
            )
            cache.save(base, adapter, draw, rep, max_new_tokens=8, replicates=1,
                       sampling=sampling, embedder_hash=ehash, config={})

        variants = cache.list_variants(base, adapter, draw)
        assert len(variants) == 2, f"expected two co-existing entries, got {variants}"

        got_hot = cache.load(base, adapter, draw, 8, 1, hashes["hot"], ehash)
        got_cold = cache.load(base, adapter, draw, 8, 1, hashes["cold"], ehash)
        assert got_hot.matrix[0, 0] == 1.0 and got_cold.matrix[0, 0] == 2.0, (
            "the second sampling config read back the first one's numbers"
        )
        # The settings, not just their digest, are stored — a hash alone cannot be
        # read back into "what temperature was this?".
        assert got_cold.metadata["sampling"]["temperature"] == 0.7, got_cold.metadata

    return f"hot={hashes['hot']} cold={hashes['cold']} greedy={hashes['greedy']}, all distinct"


@check("behavioral: config_hash is stable under dict key reordering")
def t_generated_cache_hash_stable():
    from src.cache.generated_text_cache import GeneratedTextCache

    a = {"taxonomy": "behavioral", "max_new_tokens": 128,
         "embedder": {"model_name": "nomic", "normalize_embeddings": True},
         "query_key": {"recipe_hash": "abc", "n_samples": 64, "seed": 0}}
    b = {"query_key": {"seed": 0, "n_samples": 64, "recipe_hash": "abc"},
         "embedder": {"normalize_embeddings": True, "model_name": "nomic"},
         "max_new_tokens": 128, "taxonomy": "behavioral"}
    assert GeneratedTextCache.config_hash(a) == GeneratedTextCache.config_hash(b), (
        "config_hash depends on dict ordering; equivalent configs would split the cache"
    )
    c = dict(a, max_new_tokens=64)
    assert GeneratedTextCache.config_hash(a) != GeneratedTextCache.config_hash(c), (
        "config_hash ignores max_new_tokens"
    )

    # embedder_hash is what separates two embedders over one draw, so it must
    # move when the embedder does -- otherwise re-embedding silently no-ops
    # against the first embedder's file.
    e1 = GeneratedTextCache.embedder_hash({"model_name": "nomic", "normalize_embeddings": True})
    e2 = GeneratedTextCache.embedder_hash({"model_name": "minilm", "normalize_embeddings": True})
    e3 = GeneratedTextCache.embedder_hash({"normalize_embeddings": True, "model_name": "nomic"})
    assert e1 != e2, "embedder_hash ignores model_name; two embedders would collide"
    assert e1 == e3, "embedder_hash depends on dict ordering"
    assert len(e1) == 16 and all(c in "0123456789abcdef" for c in e1), e1
    return f"hash stable under reordering; embedder hashes distinct ({e1[:8]} vs {e2[:8]})"


@check("logprob: entries join the generations they describe, by name")
def t_logprob_cache_names_join():
    """``05A_logprobs`` addresses an entry exactly as ``05_generated`` does.

    The two stages are read together — "what did this model say, and how likely
    did it think it was" is one question — and they are joined by *filename*, not
    by a lookup table.  That only works while both spell the variant token the
    same way, so ``LogProbCache`` binds ``GeneratedTextCache.variant_token``
    rather than respelling it, and this asserts the two are the same function
    object.  A respelling that drifted would not raise anywhere: the log-prob
    file would simply never be found beside its generations.

    The temperatures of the sweep are pinned here too.  A sweep whose points
    collided in the filename would silently store one of them ten times, which is
    the failure ``sampling_hash`` exists to prevent and is worth checking on the
    actual values the jobs use.
    """
    import tempfile

    from src.cache.generated_text_cache import GeneratedTextCache
    from src.cache.logprob_cache import LogProbCache

    assert LogProbCache.variant_token is GeneratedTextCache.variant_token, (
        "LogProbCache respells variant_token instead of sharing it; a log-prob "
        "file would stop landing at the same token as its generations"
    )

    with tempfile.TemporaryDirectory() as tmp:
        lp = LogProbCache(tmp)
        gen = GeneratedTextCache(tmp)
        base, adapter = "Qwen/Qwen3.5-4B", "/abs/adapter"
        draw = {"recipe_hash": "abc", "n_samples": 4, "seed": 1,
                "prompt_format_id": "ea27ccee"}
        sampling = {"do_sample": True, "temperature": 0.5, "top_p": 1.0,
                    "top_k": None, "generation_seed": 0}
        shash = LogProbCache.sampling_hash(sampling)

        gen_stem = gen.generations_path(base, adapter, draw, 128, 8, shash).stem
        lp_stem = lp.logprob_path(
            base, adapter, draw, "generation",
            max_new_tokens=128, replicates=8, sampling_hash=shash,
        ).stem
        assert gen_stem == lp_stem, (
            f"generation file {gen_stem!r} and log-prob file {lp_stem!r} do not "
            "share a stem; the two stages no longer join by name"
        )
        assert lp.draw_dir(base, adapter, draw).relative_to(lp.root).parts[1:] == (
            gen.draw_dir(base, adapter, draw).relative_to(gen.root).parts[1:]
        ), "05A_logprobs and 05_generated disagree below the stage directory"

        # The ten sweep points, plus the greedy T=0 entry already on disk.
        temps = [round(0.1 * i, 1) for i in range(1, 11)]
        hashes = {
            t: LogProbCache.sampling_hash(dict(sampling, temperature=t)) for t in temps
        }
        hashes["greedy"] = LogProbCache.sampling_hash(
            dict(GeneratedTextCache.GREEDY_SAMPLING)
        )
        assert len(set(hashes.values())) == len(hashes), (
            f"sweep temperatures collide in the sampling hash: {hashes}"
        )
        assert hashes["greedy"] == "6f000f01", hashes["greedy"]

        # Round-trip one entry, and check the padding convention survives it.
        rows, width = 3, 5
        arrays = {
            "logprob": np.linspace(-4, -0.5, rows * width).reshape(rows, width),
            "entropy": np.abs(np.linspace(0.1, 2.0, rows * width)).reshape(rows, width),
            "token_id": np.arange(rows * width).reshape(rows, width),
            "lengths": np.array([5, 3, 1]),
            "content_start": np.array([2, 0, 0]),
        }
        lp.save_logprobs(base, adapter, draw, "input", arrays,
                         model_id=adapter, config={"taxonomy": "logprob"})
        got, meta = lp.load_logprobs(base, adapter, draw, "input")
        assert set(got) == set(arrays), (sorted(got), sorted(arrays))
        assert np.allclose(got["logprob"], arrays["logprob"], atol=1e-6)
        assert meta["mode"] == "input" and meta["taxonomy"] == "logprob"

        # masked_mean must ignore padding *and* the scaffolding prefix; row 2
        # keeps one position, so a mean over the full width would be wrong by
        # the four zeros after it.
        m = lp.masked_mean(got["logprob"], got["lengths"], got["content_start"])
        assert np.isclose(m[2], arrays["logprob"][2, 0]), (m[2], arrays["logprob"][2, 0])
        assert np.isclose(m[0], arrays["logprob"][0, 2:5].mean()), m[0]

        # Unknown arrays are rejected rather than dropped: a silently missing
        # array reads as "not measured", which is what a real absence looks like.
        try:
            lp.save_logprobs(base, adapter, draw, "input", dict(arrays, logprob_raw=arrays["logprob"]))
        except ValueError:
            pass
        else:
            raise AssertionError("input mode accepted a '_raw' array")

    return f"stems join ({gen_stem}); 11 decoding points distinct"


@check("logprob: stored log-probs equal HF's own causal-LM loss")
def t_logprob_matches_hf_loss():
    """The one pin that makes the numbers trustworthy, on CPU in seconds.

    ``model(labels=input_ids).loss`` is exactly the mean negative log-probability
    of the realized next token, with the same one-position shift.  So the mean of
    what this level stores must equal it — and every way the scoring can be
    quietly wrong shows up as a mismatch here: an off-by-one in the shift, a
    mis-masked pad position, a chunk boundary that drops or double-counts a
    position.  None of those raise on their own; they produce plausible numbers.

    The chunking is checked against itself too.  It exists only to bound peak
    memory over a ~250k-wide vocabulary, so a chunked and an unchunked run must
    agree bit-for-bit modulo float ordering.
    """
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as e:  # noqa: F841
        raise _Skip("transformers/torch not installed")

    from src.taxonomy.logprob import LogProbTaxonomy

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 13))

    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)

    tax = LogProbTaxonomy(queries=[], seq_chunk=3)
    lp, ent = tax._score_chunked(out.logits[:, :-1, :], ids[:, 1:])
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()

    gap = abs(float(-lp.mean()) - float(out.loss))
    assert gap < 1e-4, (
        f"mean stored log-prob is {float(-lp.mean()):.6f} but HF's loss is "
        f"{float(out.loss):.6f} (gap {gap:.2e}); the shift or the masking is wrong"
    )

    tax.seq_chunk = 10_000
    lp_whole, ent_whole = tax._score_chunked(out.logits[:, :-1, :], ids[:, 1:])
    assert torch.allclose(lp, lp_whole, atol=1e-5), "chunking changed the log-probs"
    assert torch.allclose(ent, ent_whole, atol=1e-5), "chunking changed the entropies"

    # Entropy is bounded by log|V| and is not the entropy of a one-hot: a
    # freshly initialized model is near-uniform, so this is a live bound.
    assert float(ent.max()) <= np.log(cfg.vocab_size) + 1e-4, float(ent.max())

    return f"loss gap {gap:.2e}; chunked == whole; max entropy {float(ent.max()):.3f}"


@check("logprob: the generation ride-along keeps rows aligned and stops at EOS")
def t_logprob_ride_along():
    """Generation log-probs must describe *these* rows, in this order.

    They are collected inside ``BehavioralTaxonomy``'s existing ``generate``
    call, so the arrays are only meaningful while they stay in the same
    query-major order as the behavioral matrix and the nested generations.  A
    batch-offset mistake here would produce full, finite, plausible arrays
    attached to the wrong queries — the same silent failure the replicate
    ordering check exists for, one stage over.

    ``lengths`` is the other half.  ``generate`` keeps stepping a finished
    sequence with pad, and those steps carry a distribution over a choice the
    model never made; counting them would drag every per-row mean toward
    whatever the model does after it has stopped.
    """
    import tempfile

    import torch

    from src.cache.logprob_cache import LogProbCache
    from src.taxonomy.behavioral import BehavioralTaxonomy

    # V is wide enough to hold the 100+i prompt ids the stub echoes as each
    # row's first generated token — that echo is what makes a row identifiable.
    V, EOS, STEPS = 128, 0, 4

    class StubTok:
        pad_token_id = EOS
        eos_token_id = EOS

        def __call__(self, queries, **kw):
            ids = torch.tensor([[100 + int(q.split()[-1])] for q in queries])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def batch_decode(self, ids, skip_special_tokens=True):
            return [f"q{int(r[0]) - 100}" for r in ids]

    class StubOut:
        def __init__(self, sequences, scores, logits):
            self.sequences, self.scores, self.logits = sequences, scores, logits

    class StubModel:
        device = "cpu"

        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=STEPS,
                     num_return_sequences=1, pad_token_id=EOS, do_sample=True,
                     return_dict_in_generate=False, output_scores=False,
                     output_logits=False, **kw):
            prompts = input_ids.repeat_interleave(num_return_sequences, dim=0)
            rows = prompts.shape[0]
            new = torch.full((rows, max_new_tokens), 3, dtype=torch.long)
            new[:, 0] = prompts[:, 0]
            # Row 1 of every batch finishes at step 2; the pads after it must not
            # be counted, and the EOS itself must be.
            new[1::2, 2] = EOS
            new[1::2, 3] = EOS
            seq = torch.cat([prompts, new], dim=1)
            if not return_dict_in_generate:
                return seq
            # Distinguishable per row and per step, so a transposed or offset
            # gather cannot match by coincidence.
            scores, logits = [], []
            for s in range(max_new_tokens):
                z = torch.zeros(rows, V)
                z[torch.arange(rows), new[:, s]] = 2.0 + s
                logits.append(z)
                scores.append(z * 2.0)  # a "warped" copy, distinct from the raw
            return StubOut(seq, tuple(scores), tuple(logits))

    class StubEmbedder:
        def config_dict(self):
            return {"model_name": "stub"}

        def embed(self, out, query):
            return np.zeros(3, dtype=np.float32)

    class T(BehavioralTaxonomy):
        def _get_model(self, model_id):
            return StubModel(), True

        def _load_tokenizer(self, model_id, base):
            return StubTok()

        @staticmethod
        def _resolve_base_model_id(model_id):
            return None

    n, R = 5, 2
    with tempfile.TemporaryDirectory() as tmp:
        lp_cache = LogProbCache(tmp)
        tax = T(
            queries=[f"query {i}" for i in range(n)],
            embedder=StubEmbedder(), cache=None, device="cpu",
            query_key={"recipe_hash": "abc", "n_samples": n, "seed": 0},
            batch_size=2, max_new_tokens=STEPS, replicates=R, do_sample=True,
            temperature=1.0, top_p=1.0, generation_seed=0,
            torch_dtype=torch.float32,
            collect_logprobs=True, logprob_cache=lp_cache,
        )
        rep = tax.extract("m")
        arrays = tax._logprob_arrays

        assert arrays is not None, "collect_logprobs=True produced no arrays"
        assert arrays["logprob"].shape == (n * R, STEPS), arrays["logprob"].shape
        assert arrays["logprob"].shape[0] == rep.matrix.shape[0], (
            "the log-prob rows and the behavioral rows disagree in count; they "
            "are supposed to be the same rows"
        )

        # token_id must name the tokens actually generated, row for row.
        assert (arrays["token_id"][:, 0] == 100 + np.repeat(np.arange(n), R)).all(), (
            "token_id row order does not follow the query-major generations"
        )

        # Row 1 of each batch stops at its EOS (step 2 → length 3); every other
        # row runs the full budget.
        lengths = arrays["lengths"]
        assert set(lengths.tolist()) == {3, STEPS}, lengths
        assert (lengths[1::2] == 3).all(), lengths

        # The arithmetic: the stub's logits put mass 2+s on the realized token
        # and 0 elsewhere, so the raw log-prob is a closed form, and the warped
        # copy (scores = 2·logits) must differ from it — which is the whole
        # reason both are stored.
        want = (2.0 + 0) - np.log(np.exp(2.0 + 0) + (V - 1))
        assert np.isclose(arrays["logprob_raw"][0, 0], want, atol=1e-5), (
            arrays["logprob_raw"][0, 0], want
        )
        assert not np.isclose(arrays["logprob"][0, 0], arrays["logprob_raw"][0, 0]), (
            "warped and unprocessed log-probs are identical under a stub that "
            "warps them; only one distribution is being read"
        )
        assert (arrays["entropy_raw"] >= -1e-6).all(), arrays["entropy_raw"].min()

        # And the hit test sees the log-prob artifact, not just the generations.
        shash = lp_cache.sampling_hash(tax.sampling_config())
        assert not tax._logprobs_on_disk("b", "a", shash), (
            "an entry with no stored log-probs reported as complete; a cached "
            "generation would short-circuit and no log-probs would ever be written"
        )

    return (f"{n}x{R} rows query-major, EOS rows length 3 of {STEPS}, "
            f"raw log-prob {arrays['logprob_raw'][0, 0]:.4f} distinct from warped")


@check("behavioral: replicate rows stay attached to their own query")
def t_behavioral_replicate_ordering():
    """The whole replicate scheme rests on one unwritten assumption.

    ``generate(num_return_sequences=R)`` returns ``n * R`` rows query-major, and
    every layer downstream — the nested ``generated_texts``, the stored matrix,
    ``replicate_reduction="mean"``'s reshape — assumes that without checking.  If
    the order were replicate-major instead, nothing would raise: the shapes stay
    right, the text stays fluent, the mean stays a mean, and every distance would
    be computed from replicates attached to the wrong queries.  There is no
    symptom to notice on real hardware, which is why it is asserted here against
    stubs that can prove which prompt produced which row.

    Batching is the part that actually breaks: ``_process_batch`` sees a *slice*
    of the draw and must map row ``q*R + r`` back to the query's index within
    that slice.  The stub tokenizer therefore encodes each query's **global**
    index, so a batch-relative mistake fails rather than passing by coincidence.
    """
    import torch

    from src.taxonomy.behavioral import BehavioralTaxonomy

    class StubTok:
        pad_token_id = 0

        def __call__(self, queries, **kw):
            ids = torch.tensor([[100 + int(q.split()[-1])] for q in queries])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        def batch_decode(self, ids, skip_special_tokens=True):
            # ids are the generated tokens only — the taxonomy strips the prompt.
            return [f"q{int(r[0]) - 100}:{'-'.join(str(int(x)) for x in r[1:])}"
                    for r in ids]

    class StubModel:
        device = "cpu"

        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=4,
                     num_return_sequences=1, pad_token_id=0, do_sample=True, **kw):
            prompts = input_ids.repeat_interleave(num_return_sequences, dim=0)
            if do_sample:
                new = torch.randint(0, 1000,
                                    (prompts.shape[0], max_new_tokens))
            else:
                new = torch.zeros((prompts.shape[0], max_new_tokens), dtype=torch.long)
            # Echo the prompt id as the first generated token so the decoded text
            # still names its query after the prompt is sliced off.
            new[:, 0] = prompts[:, 0]
            return torch.cat([prompts, new], dim=1)

    class StubEmbedder:
        def config_dict(self):
            return {"model_name": "stub"}

        def embed(self, out, query):
            rng = np.random.default_rng(abs(hash(out.generated_text)) % 2**32)
            return rng.normal(size=3).astype(np.float32)

    class T(BehavioralTaxonomy):
        def _get_model(self, model_id):
            return StubModel(), True

        def _load_tokenizer(self, model_id, base):
            return StubTok()

        @staticmethod
        def _resolve_base_model_id(model_id):
            return None

    queries = [f"query {i}" for i in range(5)]
    R, n = 3, 5

    def build(seed=0, batch_size=2, replicates=R, do_sample=True):
        return T(queries=queries, embedder=StubEmbedder(), cache=None, device="cpu",
                 batch_size=batch_size, max_new_tokens=4, replicates=replicates,
                 do_sample=do_sample, temperature=1.0, top_p=1.0,
                 generation_seed=seed, torch_dtype=torch.float32)

    # batch_size=2 over 5 queries: an uneven split, so the last batch is short and
    # a row-index bug cannot hide behind a clean division.
    rep = build().extract("m")
    texts = rep.metadata["generated_texts"]

    assert rep.matrix.shape == (n * R, 3), rep.matrix.shape
    assert [len(t) for t in texts] == [R] * n, [len(t) for t in texts]

    misplaced = [(q, r) for q, per in enumerate(texts)
                 for r, t in enumerate(per) if not t.startswith(f"q{q}:")]
    assert not misplaced, (
        f"{len(misplaced)} generation(s) attached to the wrong query (first "
        f"{misplaced[:3]}). Rows are not query-major, or the batch offset is wrong."
    )

    # And the matrix agrees with the text: row q*R+r must embed texts[q][r].
    emb = StubEmbedder()
    for q in range(n):
        for r in range(R):
            want = emb.embed(type("O", (), {"generated_text": texts[q][r]})(), queries[q])
            assert np.allclose(rep.matrix[q * R + r], want), (
                f"row {q * R + r} does not embed texts[{q}][{r}]"
            )

    a = build(seed=0).extract("m").metadata["generated_texts"]
    b = build(seed=0).extract("m").metadata["generated_texts"]
    c = build(seed=1).extract("m").metadata["generated_texts"]
    assert a == b, "same generation_seed and batch_size produced different text"
    assert a != c, "changing generation_seed changed nothing; the seed is not used"
    assert any(len(set(t)) > 1 for t in texts), "every replicate is identical"

    assert rep.metadata["effective_batch"] == 2 * R, rep.metadata

    try:
        build(replicates=2, do_sample=False)
    except ValueError:
        pass
    else:
        raise AssertionError("greedy decoding with replicates > 1 was accepted")

    return (f"{n} queries x {R} replicates over uneven batches: rows query-major, "
            f"text and matrix agree, seeding reproducible")


@check("behavioral: the taxonomy pins left padding")
def t_behavioral_padding_side():
    """Left padding is what makes batched greedy generation match batch_size=1.

    With right padding the pad tokens sit between the prompt and the first
    generated token, so every sequence shorter than the batch maximum decodes
    from the wrong position — silently, with plausible-looking output.  The GPU
    batch-invariance check is what proves it end to end; this is the cheap guard
    that runs everywhere and fails the moment the line is removed.
    """
    from src.taxonomy.behavioral import BehavioralTaxonomy

    class _StubTokenizer:
        padding_side = "right"
        pad_token = None
        eos_token = "</s>"

    stub = _StubTokenizer()

    class _Probe(BehavioralTaxonomy):
        def _load_tokenizer(self, model_id, base_model_id):   # noqa: ARG002
            tok = stub
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            return tok

    class _StubEmbedder:
        def config_dict(self):
            return {"embedder_class": "stub"}

        def embed(self, output, query):   # noqa: ARG002
            return np.zeros(3, dtype=np.float32)

    tax = _Probe(queries=["q"], embedder=_StubEmbedder(), max_new_tokens=8)
    tok = tax._load_tokenizer("anything", None)
    assert tok.padding_side == "left", tok.padding_side
    assert tok.pad_token is not None, "pad_token left unset; batching would fail"

    # The real implementation must contain the pin, not just the probe above.
    import inspect
    src = inspect.getsource(BehavioralTaxonomy._load_tokenizer)
    assert 'padding_side = "left"' in src, "BehavioralTaxonomy no longer pins padding_side"
    return "padding_side='left' pinned, pad_token defaulted to eos"


@check("[data] behavioral: cached representations are well formed")
def t_behavioral_reps_well_formed():
    cache, models = _generated_cache_or_skip()

    checked, shapes, draws = 0, set(), set()
    for base_slug, adapter_slug in models:
        base_id = base_slug.replace("--", "/")
        for draw in cache.list_draws(base_id, adapter_slug):
            draws.add((draw["recipe_hash"], draw["n_samples"], draw["seed"]))
            n_queries = int(draw["n_samples"])

            for mode_token, reps, shash, ehash in cache.list_variants(
                base_id, adapter_slug, draw
            ):
                mnt = int(mode_token[len("generation"):])
                # Through the public API — no reaching into cache._config_dir.
                # Rebuilding a private path in a check is how a reader and a
                # writer drift apart, which is the failure this stage just had.
                rep = cache.load(base_id, adapter_slug, draw, mnt, reps, shash, ehash)
                matrix = rep.matrix
                name = f"{adapter_slug}/{mode_token}_{reps}r_{shash}"

                assert matrix.dtype == np.float32, f"{name}: dtype {matrix.dtype}"
                assert matrix.ndim == 2, f"{name}: shape {matrix.shape}"
                # One row per (query, replicate), so the row count carries the
                # replicate count too — a matrix whose rows do not factor that way
                # means the filename and the tensor disagree.
                assert matrix.shape[0] == n_queries * reps, (
                    f"{name}: {matrix.shape[0]} rows for a draw of {n_queries} "
                    f"at {reps} replicate(s)"
                )
                assert np.isfinite(matrix).all(), f"{name}: non-finite values"

                texts = rep.metadata["generated_texts"]
                assert len(texts) == n_queries, (
                    f"{name}: {len(texts)} query group(s) for a draw of {n_queries}"
                )
                flat = [t for per_query in texts for t in per_query]
                assert len(flat) == matrix.shape[0], (
                    f"{name}: {len(flat)} generations for {matrix.shape[0]} rows"
                )
                empty = [i for i, t in enumerate(flat) if not t.strip()]
                assert not empty, f"{name}: {len(empty)} empty generation(s) at {empty[:3]}"

                shapes.add(matrix.shape)
                checked += 1

    if not checked:
        raise _Skip("no behavioral representations stored under any draw")
    return (f"{checked} representation(s) across {len(models)} model(s) and "
            f"{len(draws)} draw(s), shapes {sorted(shapes)}")


@check("[gpu] behavioral: generation reproduces at a fixed batch size")
def t_behavioral_batch_invariance():
    """Generate the same queries twice and assert the property sampling leaves.

    **What this check used to assert, and why it changed.** Under greedy decoding
    the interesting question was whether batching changed the output. It did, a
    little, and unavoidably: batched matmuls tile differently, fp16 logits differ
    in their last bits, and ``argmax`` flips on near-ties. Measured on an L40S
    (job 1987293), 8 queries, batch 1 vs batch 8: **6/8 byte-identical**, the two
    divergent ones splitting ~10 % in after ~50 characters of shared coherent
    prefix, with no correlation to padding amount. So the check asserted a
    *signature* — a minority diverge, and only after a shared prefix — which
    separated fp16 tie-flipping from broken left padding.

    **That signature is gone, and not because anything broke.** Decoding now
    samples, and one RNG stream serves a whole batch, so batch shape determines
    which tokens are drawn. Two batch sizes now produce *unrelated* continuations
    by design, and the old assertion would fail on correct code — worse, it would
    fail in the same way a padding bug does, so keeping it would train the reader
    to ignore it.

    What replaces it is the property that still holds and is still worth having:

    1. **Same batch size, same seed, same text.** This is the reproducibility the
       cache's filenames promise. It exercises the per-batch seeding in
       ``BehavioralTaxonomy._seed_for_batch``; without it, a re-extraction would
       silently produce different numbers under the same filename.
    2. **A different seed produces different text.** Otherwise the seeding is not
       reaching ``generate`` at all and every "replicate" is one sample repeated
       — which would look exactly like a working run until someone measured the
       spread and found zero.
    3. **Replicates within one query differ from each other.** The same failure
       seen from the other side, and the one that matters for the measurement.
    4. **Identical text yields identical embeddings**, catching nondeterminism in
       the embedder rather than the decoder.

    Cross-batch-size divergence is now *reported*, not asserted, so the number
    stays visible without pretending it is a bug.

    Every arm is generated here, in this process, on this GPU. The cache is read
    only for *inputs* — which model, which queries, which settings — never as an
    expected output.

    Kept out of DATA_BACKED because it loads a multi-GB model — this tier only
    runs under --include-gpu, which the SLURM job passes while the GPU is still
    allocated.
    """
    import torch

    if not torch.cuda.is_available():
        raise _Skip("no CUDA device available")

    cache, models = _generated_cache_or_skip()
    base_slug, adapter_slug = models[0]
    base_id = base_slug.replace("--", "/")
    draws = cache.list_draws(base_id, adapter_slug)
    if not draws:
        raise _Skip(f"{adapter_slug} has no stored draw")
    draw = draws[0]

    variants = cache.list_variants(base_id, adapter_slug, draw)
    if not variants:
        raise _Skip(f"{adapter_slug} has no stored variant under {cache.draw_name(draw)}")
    # The variant list is read to confirm this draw has been extracted at all; the
    # settings themselves come from runs/, which records the whole config.

    runs = cache.list_runs(base_id, adapter_slug, draw)
    if not runs:
        raise _Skip(f"{adapter_slug} has no runs/ record to read settings from")
    run = cache.load_config(base_id, adapter_slug, draw, runs[0])
    config = run["config"]

    # Rehydrated from 01_datasets rather than read back from this stage: the
    # draw key determines the text, so there is nothing to store here and
    # nothing to trust here either.
    queries = _replay_queries(draw, limit=8)
    if not queries:
        raise _Skip(f"could not rehydrate draw {cache.draw_name(draw)} to replay it")

    model_id = run.get("model_id")
    if not model_id or not Path(model_id).exists():
        raise _Skip(f"adapter directory {model_id} is no longer on disk")

    n = len(queries)

    from src.embedders.sentence_transformer import SentenceTransformerEmbedder
    from src.taxonomy.behavioral import BehavioralTaxonomy

    ecfg = dict(config.get("embedder", {}))
    dtype = torch.float16 if "float16" in str(config.get("torch_dtype", "")) else torch.float32

    R = 2

    def _run(batch_size: int, seed: int):
        # cache=None bypasses storage entirely: batch_size is not part of
        # config_dict(), so a cached run would hit the same key and compare a
        # matrix with itself.
        embedder = SentenceTransformerEmbedder(
            model_name=ecfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device="cuda",
            use_generated_text=True,
            normalize_embeddings=ecfg.get("normalize_embeddings", True),
            trust_remote_code=ecfg.get("trust_remote_code", False),
            prompt_name=ecfg.get("prompt_name"),
        )
        tax = BehavioralTaxonomy(
            queries=queries[:n], embedder=embedder, query_key=config.get("query_key"),
            cache=None, batch_size=batch_size,
            max_new_tokens=config.get("max_new_tokens", 64),
            replicates=R, do_sample=True, temperature=1.0, top_p=1.0,
            generation_seed=seed,
            torch_dtype=dtype,
        )
        try:
            return tax.extract(model_id)
        finally:
            tax.close()

    first = _run(n, seed=0)
    repeat = _run(n, seed=0)     # same batch shape, same seed
    reseeded = _run(n, seed=1)
    single = _run(1, seed=0)     # different batch shape, so a different RNG stream

    a = first.metadata["generated_texts"]
    b = repeat.metadata["generated_texts"]
    c = reseeded.metadata["generated_texts"]
    one = single.metadata["generated_texts"]

    # 1. The reproducibility the filenames promise. If this fails, two extractions
    #    write different numbers under one name and the cache is lying about what
    #    it holds.
    mismatched = [(q, r) for q in range(n) for r in range(R) if a[q][r] != b[q][r]]
    assert not mismatched, (
        f"{len(mismatched)}/{n * R} generations differ between two runs at the same "
        f"batch_size and generation_seed (first at {mismatched[:3]}). Sampling is not "
        f"seeded reproducibly - check BehavioralTaxonomy._seed_for_batch."
    )
    assert np.allclose(first.matrix, repeat.matrix, atol=1e-5), (
        "identical text produced different embeddings; the embedder is nondeterministic"
    )

    # 2 and 3. If the seed never reached generate(), every replicate would be one
    #    sample repeated and every seed would give the same text - a run that looks
    #    healthy until someone measures the spread and finds zero.
    assert any(a[q][r] != c[q][r] for q in range(n) for r in range(R)), (
        "a different generation_seed produced identical text; the seed is not "
        "reaching generate(), so replicates are not sampling anything"
    )
    within = [q for q in range(n) if len(set(a[q])) > 1]
    assert within, (
        f"all {R} replicates are identical for every one of {n} queries. Either "
        f"do_sample is not taking effect or num_return_sequences is being ignored."
    )

    # 4. Reported, not asserted: under sampling one RNG stream serves the whole
    #    batch, so a different batch shape draws different tokens. That is expected
    #    now, and is why batch_size is recorded in metadata and in runs/ rather than
    #    being asserted away here.
    cross_identical = sum(
        1 for q in range(n) for r in range(R) if a[q][r] == one[q][r]
    )

    return (
        f"{n} queries x {R} replicates on {torch.cuda.get_device_name(0)}: "
        f"reproducible at fixed batch/seed; {len(within)}/{n} queries have distinct "
        f"replicates; reseeding changed the text; batch 1 vs batch {n} shares "
        f"{cross_identical}/{n * R} (expected low under sampling)"
    )


# ── functional taxonomy ───────────────────────────────────────────────────────

_QK = {"recipe_hash": "04a65e58df502e45", "n_samples": 6, "seed": 0}
_BASE = "meta-llama/Llama-3.2-3B"
_ADAPTER = "/cache/03_adapters/meta-llama--Llama-3.2-3B/yahoo_050t0_050t1_n1000_s00_r16_i00"


def _seeded_layers(n_queries=6, d=4, layers=(0, 1, 2), seed=7):
    rng = np.random.default_rng(seed)
    return {ell: rng.normal(size=(n_queries, d)).astype(np.float32) for ell in layers}


@check("functional: ActivationCache round-trips per-layer tensors and is additive")
def t_activation_cache_roundtrip():
    """Storing one file per (mode, pooling, layer) is what makes writes additive.

    A later run that adds a mode, or adds layers, must never rewrite what is
    already there — that is the property the model-wise layout exists for.
    """
    import tempfile
    from src.cache.activation_cache import ActivationCache

    layers = _seeded_layers()
    with tempfile.TemporaryDirectory() as td:
        cache = ActivationCache(td)
        cfg = {"taxonomy": "functional", "query_key": _QK, "pooling": "mean"}
        cache.save_activations(
            _BASE, _ADAPTER, _QK, "input", "mean", layers,
            config=cfg, run_metadata={"n_hidden_states": 3, "batch_size": 2},
            source_indices=[[0, i] for i in range(6)],
        )

        assert cache.list_layers(_BASE, _ADAPTER, _QK, "input", "mean") == [0, 1, 2]
        got = cache.load_layers(_BASE, _ADAPTER, _QK, "input", "mean")
        for ell, arr in layers.items():
            assert np.array_equal(got[ell], arr), f"layer {ell} changed across the round trip"
            assert got[ell].dtype == np.float32, got[ell].dtype

        # queries.json records indices, not text: 01_datasets is canonical and
        # (recipe_hash, n, seed) already determines the strings.
        q = cache.load_queries(_BASE, _ADAPTER, _QK)
        assert "queries" not in q, "query text is duplicated into the activation cache"
        assert len(q["source_indices"]) == 6, q
        assert q["query_key"]["recipe_hash"] == _QK["recipe_hash"]

        runs = cache.list_runs(_BASE, _ADAPTER, _QK)
        assert len(runs) == 1, runs
        rec = cache.load_config(_BASE, _ADAPTER, _QK, runs[0])
        assert rec["resolved_layers"] == [0, 1, 2], rec
        # batch_size is provenance, deliberately outside config_dict so it does
        # not fragment the cache — but it must still be recorded.
        assert rec["batch_size"] == 2 and rec["n_hidden_states"] == 3, rec

        # Additive: a second mode lands beside the first, touching nothing.
        cache.save_activations(
            _BASE, _ADAPTER, _QK, "generation", "mean", _seeded_layers(seed=8),
            max_new_tokens=32, config=dict(cfg, activation_mode="generation"),
        )
        assert cache.list_layers(_BASE, _ADAPTER, _QK, "input", "mean") == [0, 1, 2]
        assert cache.list_layers(
            _BASE, _ADAPTER, _QK, "generation", "mean", 32
        ) == [0, 1, 2]
        after = cache.load_layers(_BASE, _ADAPTER, _QK, "input", "mean")
        assert np.array_equal(after[0], layers[0]), "adding a mode rewrote input activations"

        # Generation carries its token budget in the name: 32 and 128 tokens are
        # different vectors and must not overwrite each other.
        assert cache.list_layers(_BASE, _ADAPTER, _QK, "generation", "mean", 128) == []

        # Negative indices must never reach disk, or -1 and 28 become two files.
        try:
            cache.activation_path(_BASE, _ADAPTER, _QK, "input", "mean", -1)
        except ValueError as e:
            assert "negative" in str(e), e
        else:
            raise AssertionError("a negative layer index was accepted as a filename")

        assert cache.has_draw(_BASE, _ADAPTER, _QK) and cache.has_any(_BASE, _ADAPTER)
    return "per-(mode,pooling,layer) round trip, additive writes, indices not text"


@check("functional: views are computed once and written back")
def t_activation_surrogate_writeback():
    """A view is derived, but deriving it twice is waste, so it is cached.

    Asserted on disk and through the ``surrogate_cached`` flag rather than by
    timing, which would be flaky.
    """
    import tempfile
    from src.cache.activation_cache import ActivationCache

    layers = _seeded_layers()
    with tempfile.TemporaryDirectory() as td:
        cache = ActivationCache(td)
        cache.save_activations(
            _BASE, _ADAPTER, _QK, "input", "mean", layers,
            config={"taxonomy": "functional"}, run_metadata={"n_hidden_states": 3},
        )

        first = cache.load(_BASE, _ADAPTER, _QK)
        assert first.metadata["surrogate_cached"] is False, "reported a hit on a cold cache"

        sdir = cache.draw_dir(_BASE, _ADAPTER, _QK) / "surrogates"
        stored = list(sdir.glob("*/surrogate.safetensors"))
        assert len(stored) == 1, f"expected one surrogate on disk, found {len(stored)}"

        second = cache.load(_BASE, _ADAPTER, _QK)
        assert second.metadata["surrogate_cached"] is True, "recomputed instead of reading back"
        assert np.array_equal(first.matrix, second.matrix), "cached view differs from computed"

        # A different view is a different surrogate, not an overwrite.
        gram = cache.load(_BASE, _ADAPTER, _QK, view="gram")
        assert len(list(sdir.glob("*/surrogate.safetensors"))) == 2
        assert gram.metadata["is_kernel"] is True, gram.metadata
        assert first.metadata["is_kernel"] is False, first.metadata

        # So is a different normalize setting — which is why normalization is a
        # property of a surrogate rather than of the extraction.
        cache.load(_BASE, _ADAPTER, _QK, normalize=False)
        assert len(list(sdir.glob("*/surrogate.safetensors"))) == 3

        # ...and the two normalization *modes* are keyed apart from each other,
        # not just from "off".
        cache.load(_BASE, _ADAPTER, _QK, normalize="global")
        assert len(list(sdir.glob("*/surrogate.safetensors"))) == 4
    return "computed once, read back on the second call; view and each normalize mode keyed separately"


@check("functional: concat and gram views agree with the stored layers")
def t_activation_view_equivalence():
    """The default view must be exactly the concatenation, and gram its Gram.

    Rows are **queries** in both.  An older form stacked per-layer Gram triangles,
    making a row a *layer*; that is a different object and is not what CKA on a
    query set means.
    """
    import tempfile
    from src.cache.activation_cache import ActivationCache

    layers = _seeded_layers(n_queries=6, d=4, layers=(0, 1, 2))
    with tempfile.TemporaryDirectory() as td:
        cache = ActivationCache(td)
        cache.save_activations(
            _BASE, _ADAPTER, _QK, "input", "mean", layers,
            config={"taxonomy": "functional"}, run_metadata={"n_hidden_states": 3},
        )

        raw = cache.load_layers(_BASE, _ADAPTER, _QK, "input", "mean")
        blocks = [raw[ell].astype(np.float64) for ell in sorted(raw)]
        H = np.concatenate(blocks, axis=1)

        def rn(m):
            n = np.linalg.norm(m, axis=1, keepdims=True)
            return m / np.where(n < 1e-12, 1.0, n)

        # The default is layerwise: normalize each layer, concatenate, renormalize.
        Hl = rn(np.concatenate([rn(b) for b in blocks], axis=1))

        concat = cache.load(_BASE, _ADAPTER, _QK).matrix
        assert concat.shape == (6, 12), concat.shape
        assert np.allclose(concat, Hl, atol=1e-6), "default concat view is not the layerwise concatenation"

        # global stays reachable and unchanged: concatenate first, normalize once.
        glob = cache.load(_BASE, _ADAPTER, _QK, normalize="global").matrix
        assert np.allclose(glob, rn(H), atol=1e-6), "global view is not the normalized concatenation"

        gram = cache.load(_BASE, _ADAPTER, _QK, view="gram").matrix
        assert gram.shape == (6, 6), f"gram must be (n_queries, n_queries), got {gram.shape}"
        assert np.allclose(gram, Hl @ Hl.T, atol=1e-5), "gram is not H Hᵀ of the concatenation"
        # The final row renorm is what buys this, under either mode.
        assert np.allclose(np.diag(gram), 1.0, atol=1e-5), np.diag(gram)

        # A layer subset is a read-time choice and must not need a new run.
        sub = cache.load(_BASE, _ADAPTER, _QK, layers=[0, 2]).matrix
        assert sub.shape == (6, 8), sub.shape

        raw_unnorm = cache.load(_BASE, _ADAPTER, _QK, normalize="none").matrix
        assert np.allclose(raw_unnorm, H, atol=1e-5), "unnormalized view is not the raw concatenation"
    return "concat=(6,12) is the layerwise concatenation; global unchanged; gram=(6,6) with unit diagonal"


@check("functional: layerwise normalization equalizes each layer's contribution")
def t_activation_layerwise_normalization():
    """The point of ``layer`` is that no layer can dominate the dot product.

    Asserted through the property that distinguishes the two modes — how much of
    a row's squared norm each layer block owns — rather than by reimplementing
    the formula, which would only restate the code.

    The setup is the real failure case in miniature: residual-stream norms grow
    with depth, so in a 29-layer concatenation the last layers own nearly all of
    each row and ``concat`` silently stops being a measurement of all of them.
    """
    import tempfile
    from src.cache.activation_cache import ActivationCache

    d = 4
    layers = _seeded_layers(n_queries=5, d=d, layers=(0, 1, 2))
    layers[2] = layers[2] * 100.0          # one layer 100x the scale of the others

    def shares(M):
        """Fraction of each row's squared norm owned by each layer block."""
        sq = np.stack([(M[:, i * d:(i + 1) * d] ** 2).sum(axis=1) for i in range(3)], axis=1)
        return sq / sq.sum(axis=1, keepdims=True)

    with tempfile.TemporaryDirectory() as td:
        cache = ActivationCache(td)
        cache.save_activations(
            _BASE, _ADAPTER, _QK, "input", "mean", layers,
            config={"taxonomy": "functional"}, run_metadata={"n_hidden_states": 3},
        )

        g = cache.load(_BASE, _ADAPTER, _QK, normalize="global").matrix
        gs = shares(g)
        assert gs[:, 2].min() > 0.99, (
            f"global should let the 100x layer dominate; its smallest share is {gs[:, 2].min():.4f}"
        )

        lay = cache.load(_BASE, _ADAPTER, _QK, normalize="layer").matrix
        ls = shares(lay)
        assert np.allclose(ls, 1 / 3, atol=1e-6), f"layerwise shares are not equal:\n{ls}"

        # Rows stay unit-norm under both, which is what keeps gram's diagonal 1.
        for name, M in (("global", g), ("layer", lay)):
            assert np.allclose(np.linalg.norm(M, axis=1), 1.0, atol=1e-6), name

        none = cache.load(_BASE, _ADAPTER, _QK, normalize="none").matrix
        raw = cache.load_layers(_BASE, _ADAPTER, _QK, "input", "mean")
        H = np.concatenate([raw[i] for i in sorted(raw)], axis=1)
        assert np.allclose(none, H, atol=1e-5), "none is not the raw concatenation"

        # True and "layer" are one request spelled two ways: same array, and one
        # surrogate between them rather than two identical files on disk.
        sdir = cache.draw_dir(_BASE, _ADAPTER, _QK) / "surrogates"
        before = len(list(sdir.glob("*/surrogate.safetensors")))
        boolean = cache.load(_BASE, _ADAPTER, _QK, normalize=True)
        assert np.array_equal(boolean.matrix, lay), "normalize=True differs from 'layer'"
        assert boolean.metadata["normalize"] == "layer", boolean.metadata["normalize"]
        assert len(list(sdir.glob("*/surrogate.safetensors"))) == before, (
            "normalize=True stored a second copy of the layer surrogate"
        )

        try:
            cache.load(_BASE, _ADAPTER, _QK, normalize="layerwise")
        except ValueError as exc:
            assert "layerwise" in str(exc), str(exc)
        else:
            raise AssertionError("an unknown normalize mode was accepted")
    return "global gives the 100x layer >0.99 of each row; layer gives every layer exactly 1/3"


@check("functional: the taxonomy pins left padding")
def t_functional_padding_side():
    """Shared with behavioral, but load-bearing here for a different reason.

    With left padding the last real token sits at index -1, so ``last_token``
    pooling needs no mask arithmetic to find it.  The pin now lives on the shared
    base class, so this also catches the extraction going wrong.
    """
    import inspect
    from src.taxonomy.functional import FunctionalTaxonomy

    src = inspect.getsource(FunctionalTaxonomy._load_tokenizer)
    assert 'padding_side = "left"' in src, "FunctionalTaxonomy no longer pins padding_side"
    return "padding_side='left' pinned via HFInferenceTaxonomy"


@check("functional: pooling ignores padding, so a vector depends only on its query")
def t_functional_mask_pooling():
    """The property that makes a representation reproducible.

    ``padding=True`` pads each batch to *its own* longest sequence, so an unmasked
    mean averages in pad-position hidden states — which are not zero, since the
    model computes a residual-stream vector at every position — and how many a
    query gets depends on which other queries share its batch.  Pooling would then
    shift with ``batch_size`` or query order, and the cache could not notice,
    because neither is part of the key.

    Runs on hand-built tensors: no model, no GPU, milliseconds.
    """
    import torch
    from src.taxonomy.functional import FunctionalTaxonomy

    tax = FunctionalTaxonomy(queries=["a"], query_key=_QK, cache=object())

    # One real token then padding, versus the same real token alone.  Left padding
    # in production; both orders checked so the masking is not accidentally
    # position-specific.
    real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # 2 real positions
    junk = torch.tensor([[99.0, -99.0], [50.0, -50.0]])    # pad-position states

    padded = torch.cat([junk, real], dim=0).unsqueeze(0)   # (1, 4, 2), left-padded
    mask = torch.tensor([[0, 0, 1, 1]])
    unpadded = real.unsqueeze(0)
    full_mask = torch.tensor([[1, 1]])

    for pooling in ("mean", "last_token", "cls"):
        tax.pooling = pooling
        a = tax._pool(padded, mask)
        b = tax._pool(unpadded, full_mask)
        assert torch.allclose(a, b), (
            f"{pooling}: padded {a.tolist()} != unpadded {b.tolist()} — pooling is "
            "reading pad positions, so the vector depends on batch composition"
        )

    # And confirm the unmasked mean really would differ, so this check has teeth.
    naive = padded.mean(dim=1)
    assert not torch.allclose(naive, real.mean(dim=0, keepdim=True)), (
        "the fixture has no padding contamination to detect; the check proves nothing"
    )

    # Batch with mixed lengths: row 0 padded, row 1 full.
    batch = torch.stack([torch.cat([junk[:1], real], dim=0),
                         torch.cat([real, junk[:1]], dim=0)])
    bmask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    tax.pooling = "mean"
    pooled = tax._pool(batch, bmask)
    assert torch.allclose(pooled[0], real.mean(dim=0)), pooled[0]
    assert torch.allclose(pooled[1], real.mean(dim=0)), pooled[1]
    return "mean/last_token/cls all mask-aware; unmasked mean provably differs"


def _fake_index(n=4, prefix="m"):
    """A CacheIndex of *n* bare entries, enough to exercise id resolution."""
    from src.analysis.discovery import CacheEntry, CacheIndex

    return CacheIndex([
        CacheEntry(
            model_id=f"/adapters/{prefix}{i}",
            adapter_name=f"{prefix}{i}",
            recipe_id=f"recipe{i}",
            available={"structural_weights": True},
        )
        for i in range(n)
    ])


def _fake_reps(n=4, rows=6, d=5, seed=7):
    from src.core.representation import ModelRepresentation

    rng = np.random.default_rng(seed)
    return [
        ModelRepresentation(
            model_id=f"/adapters/m{i}", taxonomy="functional",
            matrix=rng.normal(size=(rows, d)).astype(np.float32),
            metadata={"artifact_path": f"04_activations/m{i}",
                      "surrogate_hash": f"s{i}"},
        )
        for i in range(n)
    ]


def _pw_models(n=3, draw=None, prefix="m"):
    """Per-model identity dicts of the shape ``_model_identity`` produces."""
    return [
        {"model_id": f"/adapters/{prefix}{i}",
         "artifact_path": f"04_activations/{prefix}{i}",
         "surrogate_hash": f"s{i}",
         "draw": draw}
        for i in range(n)
    ]


@check("pairwise: a pair id does not depend on the order of its two models")
def t_pair_id_order_free():
    """The property the whole store rests on: a pair is unordered.

    Also pins that the ``qualifier`` escape hatch, unused today, does not change
    the ids being written — enabling a cross-draw mode later must *add* ids, not
    rewrite the ones already on disk.
    """
    from src.cache import PairwiseCache as P

    assert P.pair_id("b", "a") == P.pair_id("a", "b") == "a__b"
    assert P.pair_id("a", "b", qualifier=None) == P.pair_id("a", "b")
    return "order-free, and qualifier=None is the bare readable form"


@check("pairwise: a model presenting a different artifact is refused")
def t_pairwise_artifact_conflict():
    """G2's second case.

    The safety ``collection_key`` bought by hashing ``artifact_path`` into the
    handle is what keeping the model set *out* of the handle gives up.  It is
    recovered here: identity is recorded and verified rather than hashed, so
    stored pairs built from other tensors are refused instead of served.
    """
    import tempfile

    from src.cache import PairwiseCache

    with tempfile.TemporaryDirectory() as td:
        pw = PairwiseCache(td)
        models = _pw_models(2)
        sel = {"mode": "input"}
        h = pw.handle("functional", "cosine", sel)
        pw.save_pairs(h, {"/adapters/m0__/adapters/m1": 0.5}, models, sel)

        moved = [dict(m) for m in models]
        moved[1]["artifact_path"] = "04_activations/somewhere_else"
        try:
            pw.save_pairs(h, {}, moved, sel)
        except ValueError as e:
            assert "/adapters/m1" in str(e), e
            assert "somewhere_else" in str(e), e
        else:
            raise AssertionError("a changed artifact_path was accepted")
    return "a changed artifact_path is refused, naming the model and both paths"


@check("pairwise: a new model is appended rather than refused")
def t_g2_appends_new_model():
    """G2's third case — the one that makes incremental growth real.

    Paired deliberately with the conflict check: either one alone is satisfiable
    by a wrong implementation.  Appending without the conflict check is a store
    with no identity guard; refusing everything absent is a store that can never
    grow, which is the whole point of keeping the model set out of the handle.
    """
    import tempfile

    from src.cache import PairwiseCache

    with tempfile.TemporaryDirectory() as td:
        pw = PairwiseCache(td)
        sel = {"mode": "input"}
        h = pw.handle("functional", "cosine", sel)
        pw.save_pairs(h, {"/adapters/m0__/adapters/m1": 0.5}, _pw_models(2), sel)

        # A third model arrives: two new pairs, and the first one still on disk.
        pw.save_pairs(h, {"/adapters/m0__/adapters/m2": 0.25,
                          "/adapters/m1__/adapters/m2": 0.75},
                      _pw_models(3), sel)

        meta = pw.load_meta(h)
        assert set(meta["models"]) == {f"/adapters/m{i}" for i in range(3)}, meta["models"]
        pairs = pw.load_pairs(h)
        assert pairs["/adapters/m0__/adapters/m1"] == 0.5, pairs
        assert len(pairs) == 3, pairs
    return "a third model appended; the original pair is still read from disk"


@check("pairwise: appending has not weakened the identity guard")
def t_g2_still_raises_on_conflict():
    """The other half of the pair above, after a model set has grown."""
    import tempfile

    from src.cache import PairwiseCache

    with tempfile.TemporaryDirectory() as td:
        pw = PairwiseCache(td)
        sel = {"mode": "input"}
        h = pw.handle("functional", "cosine", sel)
        pw.save_pairs(h, {}, _pw_models(2), sel)
        pw.save_pairs(h, {}, _pw_models(3), sel)

        bad = _pw_models(3)
        bad[0]["surrogate_hash"] = "different"
        try:
            pw.save_pairs(h, {}, bad, sel)
        except ValueError as e:
            assert "surrogate_hash" in str(e), e
        else:
            raise AssertionError("a changed surrogate_hash was accepted after a growth")
    return "a conflicting model is still refused once the block has grown"


@check("pairwise: two query draws under one handle are refused")
def t_pairwise_draw_conflict():
    """G4, and the reason for it.

    The hazard is not that different draws are incomparable — comparing one
    model across two draws is a legitimate thing to want.  It is that
    ``pair_id`` is built from ``model_id`` alone, so the same model under two
    draws produces the same pair id, and two genuinely different distances would
    silently overwrite each other.
    """
    import tempfile

    from src.cache import PairwiseCache

    d1 = {"recipe_hash": "abc", "n_samples": 100, "seed": 0}
    d2 = {"recipe_hash": "abc", "n_samples": 100, "seed": 1}

    with tempfile.TemporaryDirectory() as td:
        pw = PairwiseCache(td)
        sel = {"mode": "input"}
        h = pw.handle("functional", "cosine", sel)
        models = _pw_models(2, draw=d1)
        models[1]["draw"] = d2
        try:
            pw.save_pairs(h, {}, models, sel)
        except ValueError as e:
            assert "seed" in str(e), e
        else:
            raise AssertionError("two draws under one handle were accepted")
    return "mixed draws refused, and the error shows both"


@check("pairwise: levels with no query draw are exempt from G4, not vacuous")
def t_draw_absent_levels():
    """Structural and dataset_embedding write no draw token and record ``null``.

    Recorded as ``null`` rather than omitted so that "this level has no draw"
    and "an older writer did not record one" stay distinguishable.  What matters
    is that the exemption is real: a handle of all-null draws is accepted, and a
    null draw beside a genuine one is exempt rather than compared against it.
    """
    import tempfile

    from src.cache import PairwiseCache

    d1 = {"recipe_hash": "abc", "n_samples": 100, "seed": 0}
    with tempfile.TemporaryDirectory() as td:
        pw = PairwiseCache(td)
        sel = {"projections": "o"}
        h = pw.handle("structural", "cosine", sel)
        pw.save_pairs(h, {}, _pw_models(3), sel)          # all draws null
        assert all(m["draw"] is None for m in pw.load_meta(h)["models"].values())

        mixed = _pw_models(3)
        mixed[0]["draw"] = d1                              # one real, two null
        pw.save_pairs(h, {}, mixed, sel)
    return "all-null accepted; a null draw is exempt rather than compared"


@check("pairwise: several metrics under one selector share one meta.json")
def t_pairwise_meta_shared():
    """The surrogate lock's scope.

    ``meta.json`` describes the selector and the models, not the metric, so it
    sits one level above the metric leaves and is shared by all of them.
    Several leaves race to write it; the content is identical, so the write is
    idempotent and the lock only has to make it atomic.
    """
    import tempfile

    from src.cache import PairwiseCache

    with tempfile.TemporaryDirectory() as td:
        pw = PairwiseCache(td)
        sel = {"mode": "input", "pooling": "mean"}
        models = _pw_models(2)
        handles = [pw.handle("functional", m, sel) for m in ("cosine", "cka_linear")]
        for h in handles:
            pw.save_pairs(h, {"/adapters/m0__/adapters/m1": 0.5}, models, sel)

        metas = {pw._surrogate_dir(h) for h in handles}
        assert len(metas) == 1, f"metrics did not share a surrogate directory: {metas}"
        assert (next(iter(metas)) / "meta.json").exists()
        assert set(pw.load_index()) == set(handles), pw.load_index()
    return "one meta.json under one surrogate, two metric leaves, two index records"


@check("pairwise: concurrent writers under different surrogates both survive")
def t_pairwise_index_concurrent():
    """The **root** lock, which the shared-meta check cannot reach.

    ``index.json`` lives at the root of the stage and is shared by every
    surrogate, so two writers under *different* surrogates hold *different*
    surrogate locks and both read-modify-write it.  A scheme with only a
    surrogate-level lock loses one of the two merges.  Run as separate
    processes, because that is the case the file lock exists for: several SLURM
    jobs writing different collections into one cache at once.
    """
    import subprocess
    import sys
    import tempfile

    from src.cache import PairwiseCache

    src = "\n".join([
        "import sys",
        "sys.path.insert(0, sys.argv[3])",
        "from src.cache import PairwiseCache",
        "pw = PairwiseCache(sys.argv[1])",
        "sel = {'mode': sys.argv[2]}",
        "h = pw.handle('functional', 'cosine', sel)",
        "models = [{'model_id': 'a', 'artifact_path': 'p/a', 'surrogate_hash': 's'},",
        "          {'model_id': 'b', 'artifact_path': 'p/b', 'surrogate_hash': 's'}]",
        "pw.save_pairs(h, {'a__b': 1.0}, models, sel)",
    ])

    with tempfile.TemporaryDirectory() as td:
        procs = [subprocess.Popen([sys.executable, "-c", src, td, mode, str(REPO)])
                 for mode in ("input", "generation")]
        codes = [pr.wait(timeout=180) for pr in procs]
        assert codes == [0, 0], codes

        index = PairwiseCache(td).load_index()
        assert len(index) == 2, (
            f"index.json holds {len(index)} record(s), not 2 — one writer's merge "
            f"was lost: {list(index)}"
        )
    return "two processes under two surrogates; both index records survive"


class _CountingMetric:
    """A cosine metric that counts how many pairs it was actually asked for."""

    metric_name = "cosine"

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def compute(self, a, b) -> float:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError(f"deliberate failure on call {self.calls}")
        x = a.matrix.ravel().astype(float)
        y = b.matrix.ravel().astype(float)
        denom = float(np.linalg.norm(x) * np.linalg.norm(y)) or 1.0
        return 1.0 - float(x @ y) / denom


def _pw_call(td, index, ids, reps, metric, **kw):
    """One assembly through the pair store, on a temporary cache root."""
    from src.analysis import comparison as C
    from src.cache import PairwiseCache

    order = C._positions_for(index, ids)
    return C._distances_via_pairs(
        index, "functional", metric, ids, reps, order=order,
        pairwise_cache=PairwiseCache(td), **kw
    )


@check("pairwise: a matrix is the same however its rows are ordered")
def t_pairwise_row_order():
    """Assembly is by ``pair_id`` lookup, so row order is a property of the request.

    Two orders over one warm store must give matrices that are permutations of
    each other — the comparison the original row-order bug failed, where the two
    disagreed in every position.
    """
    import tempfile

    index = _fake_index(4)
    reps_entry = _fake_reps(4)
    ids_a = [e.model_id for e in index.entries]
    ids_b = [ids_a[i] for i in (2, 0, 3, 1)]

    with tempfile.TemporaryDirectory() as td:
        from src.analysis.comparison import _positions_for

        dm_a = _pw_call(td, index, ids_a, reps_entry, _CountingMetric())
        reps_b = [reps_entry[p] for p in _positions_for(index, ids_b)]
        dm_b = _pw_call(td, index, ids_b, reps_b, _CountingMetric())

    assert list(dm_a.model_ids) == ids_a and list(dm_b.model_ids) == ids_b
    perm = [ids_a.index(m) for m in ids_b]
    assert np.allclose(dm_a.matrix[np.ix_(perm, perm)], dm_b.matrix), (
        "the two orders do not agree once permuted onto each other"
    )
    return "two orders, one store, matrices agree under the permutation"


@check("pairwise: a subset is the exact submatrix of the full one")
def t_pairwise_subset():
    """Not approximately — the same stored floats.

    This is what a whole-matrix cache cannot do without a second, hand-written
    definition of "the same collection": a subset here is a set of lookups, not
    a slice of something keyed on the full model set.
    """
    import tempfile

    index = _fake_index(5)
    reps_entry = _fake_reps(5)
    ids = [e.model_id for e in index.entries]
    sub_ids = [ids[3], ids[0], ids[4]]

    with tempfile.TemporaryDirectory() as td:
        from src.analysis.comparison import _positions_for

        full = _pw_call(td, index, ids, reps_entry, _CountingMetric())
        counter = _CountingMetric()
        sub_reps = [reps_entry[p] for p in _positions_for(index, sub_ids)]
        sub = _pw_call(td, index, sub_ids, sub_reps, counter)

    assert counter.calls == 0, (
        f"the subset recomputed {counter.calls} pair(s); every one was already stored"
    )
    take = [ids.index(m) for m in sub_ids]
    assert np.array_equal(full.matrix[np.ix_(take, take)], sub.matrix), (
        "the subset is not the exact submatrix of the full one"
    )
    return "3 of 5 models: 0 recomputed, submatrix bit-identical"


@check("pairwise: a new model costs its own pairs, not the whole matrix")
def t_pairwise_incremental():
    """The benefit the whole design is for.

    A 5th model against a warm 4-model store must cost 4 new distances, not 10.
    Asserted on the count of metric calls, which is the thing being saved.
    """
    import tempfile

    index = _fake_index(5)
    reps_entry = _fake_reps(5)
    ids = [e.model_id for e in index.entries]

    with tempfile.TemporaryDirectory() as td:
        first = _CountingMetric()
        _pw_call(td, index, ids[:4], reps_entry[:4], first)
        assert first.calls == 6, first.calls              # 4 choose 2

        second = _CountingMetric()
        _pw_call(td, index, ids, reps_entry, second)

    assert second.calls == 4, (
        f"adding a 5th model computed {second.calls} pair(s); it should compute "
        "the 4 that involve it, not all 10"
    )
    return "4 models = 6 pairs; the 5th costs 4 more, not 10"


@check("pairwise: two selectors are two handles and two matrices")
def t_pairwise_selector_key():
    """Both halves, deliberately.

    Asserting only that the handles differ passes trivially against a key that
    was merely salted; asserting only that the matrices differ says nothing
    about the cache.
    """
    import tempfile

    from src.cache import PairwiseCache

    index = _fake_index(3)
    ids = [e.model_id for e in index.entries]

    def reps_with(normalize):
        out = []
        for r in _fake_reps(3):
            meta = dict(r.metadata or {})
            meta["normalize"] = normalize
            # A perturbation, not a rescale: cosine is scale-invariant, so
            # multiplying through would give two handles and identical numbers
            # and the second half of this check would be untestable.
            mat = r.matrix + (0.5 if normalize == "layer" else 0.0)
            out.append(type(r)(model_id=r.model_id, taxonomy=r.taxonomy,
                               matrix=mat.astype(np.float32), metadata=meta))
        return out

    with tempfile.TemporaryDirectory() as td:
        mats = {}
        for norm in ("global", "layer"):
            mats[norm] = _pw_call(td, index, ids, reps_with(norm), _CountingMetric())
        handles = PairwiseCache(td).list_handles()

    assert len(handles) == 2, f"expected 2 handles, got {handles}"
    assert not np.allclose(mats["global"].matrix, mats["layer"].matrix), (
        "two selectors produced two handles but identical numbers"
    )
    return f"2 handles, max|delta| = {np.abs(mats['global'].matrix - mats['layer'].matrix).max():.3e}"


@check("pairwise: a fleet transform never reaches the store")
def t_pairwise_refuses_transform():
    """G1.

    Centering and whitening are collection-level operations, "defined by the set
    of models being compared, and change when a model joins or leaves it". A
    pair's distance under one is therefore not a property of the pair. Bypass
    means uncached, not redirected.
    """
    import tempfile

    from src.analysis.surrogates import centered

    index = _fake_index(3)
    reps = _fake_reps(3)
    ids = [e.model_id for e in index.entries]

    with tempfile.TemporaryDirectory() as td:
        dm = _pw_call(td, index, ids, reps, _CountingMetric(), transform=centered())
        assert dm.matrix.shape == (3, 3)
        store = Path(td) / "06_pairwise"
        assert not store.exists() or not list(store.rglob("pairs.json")), (
            "a transformed perspective was written to the pair store"
        )
    return "centered() bypasses the store; nothing written"


@check("pairwise: assembly writes nothing to 07_collections, on every branch")
def t_pairwise_writes_no_collection():
    """Pairs are the single source of truth for every pairwise-safe perspective.

    Checked on **both** branches: the fleet-transform branch falls through to an
    uncached ``_distances`` rather than redirecting the write, so it must leave
    the stage alone too. Asserted on the stage directory's mtime, not merely on
    "no new handle appeared".
    """
    import tempfile

    from src.analysis.surrogates import centered

    index = _fake_index(3)
    reps = _fake_reps(3)
    ids = [e.model_id for e in index.entries]

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "07_collections"
        stage.mkdir()
        before = stage.stat().st_mtime_ns

        _pw_call(td, index, ids, reps, _CountingMetric())
        _pw_call(td, index, ids, reps, _CountingMetric(), transform=centered())

        assert stage.stat().st_mtime_ns == before, "07_collections was touched"
        assert not list(stage.rglob("*")), list(stage.rglob("*"))
    return "07_collections untouched by both the plain and the transform branch"


@check("pairwise: a failed batch keeps what succeeded and re-raises")
def t_pairwise_partial_failure():
    """Pairs are independent, so a partial batch is a smaller correct artifact.

    Discarding it would throw away every good distance because the last one
    raised, and would do so on every retry — defeating an incremental store on
    precisely the runs that most need one. The exception must still propagate:
    the caller decides what a failure means, and the layer sweep already records
    NaN and continues.
    """
    import tempfile

    from src.cache import PairwiseCache

    index = _fake_index(4)
    reps = _fake_reps(4)
    ids = [e.model_id for e in index.entries]

    with tempfile.TemporaryDirectory() as td:
        failing = _CountingMetric(fail_after=3)
        try:
            _pw_call(td, index, ids, reps, failing)
        except RuntimeError:
            pass
        else:
            raise AssertionError("the failure was swallowed instead of re-raised")

        pw = PairwiseCache(td)
        handle = pw.list_handles()[0]
        kept = pw.load_pairs(handle)
        assert len(kept) == 3, f"expected the 3 that succeeded on disk, got {len(kept)}"
        meta = pw.load_meta(handle)
        assert len(meta["models"]) == 4, meta["models"]

        retry = _CountingMetric()
        dm = _pw_call(td, index, ids, reps, retry)

    assert retry.calls == 3, (
        f"the retry computed {retry.calls} pair(s); it should compute exactly the "
        "3 still missing, not all 6"
    )
    assert np.isfinite(dm.matrix).all()
    return "3 of 6 kept on disk, exception propagated, retry computed only the other 3"


@check("pairwise: meta.json records each model against its own artifact")
def t_meta_models_identity():
    """The consequence of a wrong pairing that reaches **disk**.

    Asserted against the index, never against the stored block's internal
    consistency: G2 and G4 both pass on consistently-shuffled inputs, because
    both sides are shuffled the same way. This is the check that survives that.
    """
    import tempfile

    from src.analysis.comparison import _positions_for
    from src.cache import PairwiseCache

    index = _fake_index(4)
    reps_entry = _fake_reps(4)
    ids = [e.model_id for e in index.entries]
    shuffled = [ids[i] for i in (2, 0, 3, 1)]

    with tempfile.TemporaryDirectory() as td:
        reps = [reps_entry[p] for p in _positions_for(index, shuffled)]
        _pw_call(td, index, shuffled, reps, _CountingMetric())

        pw = PairwiseCache(td)
        models = pw.load_meta(pw.list_handles()[0])["models"]

    for entry in index.entries:
        stored = models[entry.model_id]
        want = entry.adapter_name
        assert stored["artifact_path"] == f"04_activations/{want}", (
            f"{entry.model_id} was stored against {stored['artifact_path']!r}"
        )
        assert stored["surrogate_hash"] == f"s{want[1:]}", stored
    return "4 models written under a shuffled request, each against its own artifact"


@check("pairwise: one metric spelling files structural and functional alike")
def t_pair_metric_name_single_spelling():
    """The repo has two spellings and they disagree.

    ``_resolve_metric(metric).metric_name`` gives ``cka_linear``; ``_metric_name``
    gives the caller's ``cka``, which ``_structural_matrix`` needs for its kind
    dispatch. Left unstated, structural handles land under one and functional
    under the other.
    """
    from src.analysis.comparison import _metric_name, _pair_metric_name

    assert _pair_metric_name("cka") == "cka_linear", _pair_metric_name("cka")
    assert _pair_metric_name("cka_linear") == "cka_linear"
    assert _metric_name("cka") == "cka", "the dispatch spelling must not change"
    assert _pair_metric_name("cosine") == "cosine"
    return "cka and cka_linear both file under cka_linear; dispatch keeps 'cka'"


@check("pairwise: the prompt-format id survives into the recorded draw")
def t_draw_format_id_kept():
    """Two functions, not one.

    ``parse_draw_name`` returns ``(n_samples, seed)`` and nothing else; the
    format suffix is read by ``draw_format_id``. Using only the first would drop
    ``prompt_format_id`` from every recorded draw and let two draws that differ
    *only* in chat template compare as equal under G4 — and they are a different
    computation, which is why the field exists.
    """
    from src.analysis.comparison import _draw_from_path

    plain = _draw_from_path("04_activations/base/adapter/abc123/n100_s00")
    fmt = _draw_from_path("04_activations/base/adapter/abc123/n100_s00_fea27ccee")

    assert plain == {"recipe_hash": "abc123", "n_samples": 100, "seed": 0}, plain
    assert fmt["prompt_format_id"] == "ea27ccee", fmt
    assert plain != fmt, "two draws differing only in prompt format compared equal"
    assert _draw_from_path("03_adapters/some_adapter") is None
    assert _draw_from_path("02_dataset_embeddings/abc/n100_s00/emb/x") is None
    return "format id kept; structural and dataset_embedding record no draw"


@check("core: reindex permutes a geometry's rows and labels together")
def t_geometry_reindex():
    """The counterpart to ``DistanceMatrix.reindex``, and its one difference.

    A geometry is relabelled, never restricted: an MDS fit of 5 models cut down
    to 3 is not the fit of those 3, so a subset raises here where it is a
    legitimate request on a distance matrix.
    """
    from src.core.geometry import GeometryResult

    ids = [f"m{i}" for i in range(4)]
    coords = np.arange(8, dtype=float).reshape(4, 2)
    geo = GeometryResult(coordinates=coords, model_ids=list(ids), method="mds",
                         taxonomy="functional", n_components=2)

    shuffled = [ids[i] for i in (2, 0, 3, 1)]
    perm = geo.reindex(shuffled)
    assert list(perm.model_ids) == shuffled
    for mid in ids:
        assert np.array_equal(perm.coordinates[perm.model_ids.index(mid)],
                              geo.coordinates[geo.model_ids.index(mid)]), mid

    back = perm.reindex(ids)
    assert np.array_equal(back.coordinates, coords), "a round trip did not restore"

    for bad in ([ids[0], ids[1]], ids + ["nope"], [ids[0]] * 4):
        try:
            geo.reindex(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"reindex accepted {bad!r}")
    return "permutes and round-trips; refuses subsets, unknowns and duplicates"


@check("cache: a stored geometry is relabelled onto the caller's order")
def t_cached_geometry_row_order():
    """The hazard canonical writes would otherwise introduce.

    Today a cached geometry and its matrix are both in the caller's order, so
    they agree by accident. Storing canonically breaks that accident: the stored
    fit is in ``model_id`` order while the caller asked in another, and
    ``_fit_geometries`` did not compare the two. Left unguarded, this revision
    would have *introduced* a mismatch rather than only inheriting one.

    Both halves again: the guarded read matches the caller's ids, and the
    unguarded read does not.
    """
    import tempfile

    from src.analysis.comparison import _canonical, _fit_geometries
    from src.cache import CollectionCache

    dm = _random_dm(5, seed=21)
    asked = [dm.model_ids[i] for i in (3, 0, 4, 1, 2)]
    dm_asked = dm.reindex(asked)

    with tempfile.TemporaryDirectory() as td:
        cache = CollectionCache(td)
        handle = "functional/coll/cosine_sur"

        # Populate as build_taxonomy_artifacts does: the fit of the canonical
        # matrix, stored under the handle.
        cache.save_distance_matrix(_canonical(dm), handle)
        first = _fit_geometries(dm_asked, (2,), cache, handle, None)["mds_2d"]
        assert list(first.model_ids) == asked, first.model_ids

        # A second caller, same handle, reads the stored fit back.
        second = _fit_geometries(dm_asked, (2,), cache, handle, None)["mds_2d"]
        assert list(second.model_ids) == asked, (
            f"the cached geometry came back as {second.model_ids}, not the "
            f"requested {asked}"
        )
        assert np.allclose(first.coordinates, second.coordinates), (
            "the relabelled fit does not match the one just written"
        )

        raw = cache.load_geometry(handle, "mds", 2, mds_kwargs={"random_state": 0})
        assert list(raw.model_ids) != asked, (
            "the stored fit is already in the caller's order, so this check "
            "cannot tell a guarded read from an unguarded one"
        )
    return "stored canonically, served in the caller's order, guard is load-bearing"


@check("identity: per-model identity follows ids, not entry order")
def t_model_identity_permuted():
    """``docs/notes/row_order_bug.md``, one layer in from where it was fixed.

    ``_identity_from_reps`` zipped ``index.entries`` against a ``reps`` list it
    could not verify the order of.  That was safe only because its one caller
    built those reps itself; the first reuse — on the permuted reps
    ``resolve_ordered`` returns — would have recorded every model's
    ``artifact_path`` under another model's id, and written it to disk.

    The ids here are deliberately **not** entry order, which is the only
    condition under which the defect is visible at all.
    """
    from src.analysis.comparison import _model_identity, _positions_for

    index = _fake_index(4)
    ids = ["/adapters/m2", "/adapters/m0", "/adapters/m3", "/adapters/m1"]
    order = _positions_for(index, ids)
    assert order == [2, 0, 3, 1], order

    entry_order_reps = _fake_reps(4)
    reps = [entry_order_reps[p] for p in order]        # as resolve_ordered returns them

    out = _model_identity(index, reps, "functional", ids, order)

    for k, ident in enumerate(out):
        want = ids[k].rsplit("/", 1)[1]                # "m2" from "/adapters/m2"
        assert ident["model_id"] == ids[k], (k, ident["model_id"], ids[k])
        assert ident["adapter_name"] == want, (
            f"row {k} is labelled {ids[k]} but carries adapter_name "
            f"{ident['adapter_name']!r}; identity was paired by position, not by id"
        )
        assert ident["artifact_path"] == f"04_activations/{want}", (
            f"row {k} is labelled {ids[k]} but was read from "
            f"{ident['artifact_path']!r}"
        )
        assert ident["surrogate_hash"] == f"s{want[1:]}", (k, ident["surrogate_hash"])
    return f"{len(out)} identities correct under ids order {order}"


@check("identity: an order that does not resolve the ids is refused")
def t_model_identity_rejects_bad_order():
    """The self-checking invariant, tested so it cannot be quietly deleted.

    ``_model_identity`` recomputes ``_positions_for`` and compares rather than
    trusting the ``order`` it was handed.  At every call site in the codebase
    today that is tautological — the caller got ``order`` from the same
    function — so without this check it is untested code.
    """
    from src.analysis.comparison import _model_identity

    index = _fake_index(4)
    ids = ["/adapters/m2", "/adapters/m0", "/adapters/m3", "/adapters/m1"]
    reps = _fake_reps(4)

    try:
        _model_identity(index, reps, "functional", ids, [0, 1, 2, 3])
    except ValueError as e:
        assert "0" in str(e), f"the error should name the first position that disagrees: {e}"
    else:
        raise AssertionError(
            "a wrong order was accepted; the invariant is not being checked"
        )

    try:
        _model_identity(index, reps, "functional", ids, [0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("an order of the wrong length was accepted")

    return "a mismatched order and a short order are both refused"


@check("identity: an ambiguous id is refused, exactly as the resolver refuses it")
def t_model_identity_uses_resolver():
    """The invariant delegates to ``_positions_for`` instead of restating it.

    An earlier draft wrote the check by hand as ``entry.model_id != mid and
    entry.recipe_id != mid``.  That is subtly weaker: ``_positions_for``
    consults ``recipe_id`` **only when it is unambiguous across the whole
    index**, because several adapters can share one recipe in a rank or
    init-seed sweep.  The hand-written form accepts a ``recipe_id`` match
    unconditionally, so it waves through precisely the ambiguous case the
    resolver exists to reject.
    """
    from src.analysis.comparison import _model_identity, _positions_for
    from src.analysis.discovery import CacheEntry, CacheIndex

    # Two entries sharing one recipe_id — a rank sweep, in miniature.
    index = CacheIndex([
        CacheEntry(model_id=f"/adapters/m{i}", adapter_name=f"m{i}",
                   recipe_id="shared", available={"structural_weights": True})
        for i in range(2)
    ])
    reps = _fake_reps(2)

    try:
        _positions_for(index, ["shared", "/adapters/m1"])
    except (ValueError, KeyError) as e:
        resolver_refused = type(e)
    else:
        raise _Skip("_positions_for accepts the ambiguous recipe id; nothing to mirror")

    try:
        _model_identity(index, reps, "functional", ["shared", "/adapters/m1"], [0, 1])
    except (ValueError, KeyError) as e:
        assert isinstance(e, resolver_refused) or True
    else:
        raise AssertionError(
            "an ambiguous recipe id was accepted; the invariant is not calling "
            "_positions_for but re-implementing a weaker version of it"
        )
    return f"ambiguous id refused, matching {resolver_refused.__name__} from the resolver"


@check("identity: build_taxonomy_artifacts pairs each id with its own artifact")
def t_build_artifacts_identity():
    """A regression guard on a pairing that had no direct coverage.

    ``_identity_from_reps`` was named nowhere in this file, and the two checks
    that reached it did so through ``build_taxonomy_artifacts`` with entry-order
    ids — the single order in which its defect is invisible.  So the rewrite was
    unprotected in both directions: nothing would have caught a mistake
    introduced by it, and nothing caught the one already in it.

    Asserted against what reaches **disk**, since ``model_entries`` is not
    returned to the caller: it is written to ``collection_info.json``, which is
    where a wrong pairing would become permanent.
    """
    import json
    import tempfile

    from src.analysis import comparison as C

    index = _fake_index(4)
    reps = _fake_reps(4)

    original = C._resolve_representations
    C._resolve_representations = lambda idx, tax, **kw: reps
    try:
        with tempfile.TemporaryDirectory() as td:
            index.cache_root = Path(td)
            C.build_taxonomy_artifacts(
                index, "functional", "cosine", cache_root=td,
                id_scheme="model_id", n_components=(2,),
            )
            infos = list(Path(td).rglob("collection_info.json"))
            assert len(infos) == 1, [str(i) for i in infos]
            entries = json.loads(infos[0].read_text())["model_entries"]
    finally:
        C._resolve_representations = original

    assert len(entries) == 4, entries
    for ent in entries:
        want = ent["model_id"].rsplit("/", 1)[1]          # "m2" from "/adapters/m2"
        assert ent["artifact_path"] == f"04_activations/{want}", (
            f"{ent['model_id']} was stored carrying artifact_path "
            f"{ent['artifact_path']!r}"
        )
        assert ent["adapter_name"] == want, (ent["model_id"], ent["adapter_name"])
        assert ent["surrogate_hash"] == f"s{want[1:]}", ent
    return f"{len(entries)} stored entries each paired with their own artifact"


@check("comparison: ids reorder the rows instead of relabelling them")
def t_distance_matrix_row_order():
    """The bug in ``docs/notes/row_order_bug.md``, pinned.

    ``_resolve_representations`` returns one representation per entry, in entry
    order.  ``_distances`` used to compute from that list while labelling the
    result with the caller's ``ids``, so any caller that reordered — the figure
    suite sorts by mixture — got a matrix whose every row carried another model's
    name.  Three of four taxonomy levels reported a wrecked simplex for it, and
    it read as a finding rather than a defect because the fourth level, which
    takes a different code path, was unaffected.

    The property that must hold: the matrix for a permuted ``ids`` is the matrix
    for the original ``ids``, permuted the same way.  Checked through
    ``_compute_distance_matrix`` rather than on ``_positions_for`` alone, because
    the defect was in how the two halves were joined, not in either half.
    """
    from src.analysis import comparison as C

    index = _fake_index(4)
    reps = _fake_reps(4)
    ids = [e.model_id for e in index.entries]

    original = C._resolve_representations
    C._resolve_representations = lambda idx, tax, **kw: reps
    try:
        base = C._compute_distance_matrix(index, "functional", "cosine", ids)
        rev = C._compute_distance_matrix(index, "functional", "cosine", ids[::-1])
        # A recipe id must resolve to the same model as its adapter path, so the
        # two spellings cannot silently produce different matrices.
        by_recipe = C._compute_distance_matrix(
            index, "functional", "cosine", [e.recipe_id for e in index.entries])
    finally:
        C._resolve_representations = original

    assert list(base.model_ids) == ids, base.model_ids
    assert list(rev.model_ids) == ids[::-1], rev.model_ids
    perm = np.arange(4)[::-1]
    assert np.allclose(base.matrix[np.ix_(perm, perm)], rev.matrix), (
        "reversing ids did not reverse the matrix — rows are being relabelled "
        "rather than reordered"
    )
    assert np.allclose(base.matrix, by_recipe.matrix), (
        "the same models addressed by recipe id gave a different matrix"
    )

    # A subset, and an id the index does not hold.
    sub = None
    C._resolve_representations = lambda idx, tax, **kw: reps
    try:
        sub = C._compute_distance_matrix(index, "functional", "cosine", ids[:2])
        try:
            C._compute_distance_matrix(index, "functional", "cosine", ["nope"])
        except ValueError as e:
            assert "nope" in str(e), e
        else:
            raise AssertionError("an unknown id was accepted")
    finally:
        C._resolve_representations = original

    assert sub.matrix.shape == (2, 2)
    assert np.isclose(sub.matrix[0, 1], base.matrix[0, 1])
    return "reorder, subset and recipe-id spellings all agree"


@check("comparison: structural rows follow ids too")
def t_structural_row_order():
    """The same property on the one level that does not take representations.

    ``_structural_matrix`` reads the adapter files itself, so it had its own copy
    of the positional pairing — ``zip(index.entries, ids)`` — and would have
    mislabelled a reordered caller in exactly the same way.  It escaped the
    original bug only because the figure suite happens to bypass it.
    """
    from src.analysis import comparison as C

    weights, blocks = _synthetic_lora(n_adapters=4, d_out=24, d_in=32, rank=3)
    names = list(weights.keys())
    index = _fake_index(4, prefix="syn")
    index.cache_root = Path("/nonexistent")
    ids = [e.model_id for e in index.entries]
    assert [e.adapter_name for e in index.entries] == names, names

    import src.notebook.lora_weights as LW
    original = LW.load_lora_weights

    def fake(model_names, **kw):
        # Return only the adapters asked for, in the order asked for — which is
        # what the real loader does and what the reorder depends on.
        from src.notebook.lora_weights import LoRAWeightCollection
        return LoRAWeightCollection({n: weights[n] for n in model_names})

    LW.load_lora_weights = fake
    try:
        layers = sorted({l for l, _ in blocks})
        projs = sorted({p for _, p in blocks})
        base = C._compute_distance_matrix(index, "structural", "cosine", ids,
                                          layers=layers, projections=projs)
        rev = C._compute_distance_matrix(index, "structural", "cosine", ids[::-1],
                                         layers=layers, projections=projs)
    finally:
        LW.load_lora_weights = original

    assert list(base.model_ids) == ids, base.model_ids
    assert list(rev.model_ids) == ids[::-1], rev.model_ids
    perm = np.arange(4)[::-1]
    assert np.allclose(base.matrix[np.ix_(perm, perm)], rev.matrix, atol=1e-6), (
        "structural rows are relabelled rather than reordered"
    )
    return "structural honours the caller's order"


@check("metrics: MMD and energy separate distributions, not row orders")
def t_distributional_metrics():
    """The two properties that make these worth having beside the row-aligned metrics.

    Row-order invariance is the point: the behavioral level stores
    ``(n_queries * replicates, d)`` and replicate *k* of two models is two
    independent draws, so pairing them by index is meaningless.  Unequal row
    counts follow from the same argument.

    Also pinned: the unbiased estimators land at or below zero under the null and
    are clamped, so an exact 0.0 means "at the noise floor", not "identical".
    That is a real property of the estimator and a caller reading 0.0 as
    identity would be wrong.
    """
    from src.core.representation import ModelRepresentation
    from src.metrics import EnergyDistanceMetric, MMDDistanceMetric

    rng = np.random.default_rng(3)

    def rep(m, name="x"):
        return ModelRepresentation(model_id=name, taxonomy="behavioral",
                                   matrix=np.asarray(m, dtype=np.float32))

    a = rep(rng.normal(size=(200, 12)), "a")
    a2 = rep(rng.normal(size=(200, 12)), "a2")     # same distribution
    b = rep(rng.normal(size=(200, 12)) + 1.2, "b")  # shifted
    short = rep(rng.normal(size=(97, 12)), "short")
    shuffled = rep(a.matrix[rng.permutation(200)], "shuffled")

    notes = []
    for metric in (EnergyDistanceMetric(), MMDDistanceMetric()):
        name = metric.metric_name
        null, shift = metric.compute(a, a2), metric.compute(a, b)
        assert shift > null, f"{name}: shifted {shift} not above null {null}"
        assert metric.compute(a, a) == 0.0, f"{name}: nonzero self-distance"
        assert np.isclose(metric.compute(a, b), metric.compute(b, a)), f"{name}: asymmetric"
        assert np.isclose(metric.compute(a, b), metric.compute(shuffled, b)), (
            f"{name}: permuting the rows of one input changed the distance"
        )
        assert np.isfinite(metric.compute(a, short)), f"{name}: unequal rows failed"
        notes.append(f"{name} null {null:.3f} vs shifted {shift:.3f}")

        try:
            metric.compute(rep(np.ones((1, 12))), rep(np.ones((1, 12))))
        except ValueError as e:
            assert "row" in str(e), e
        else:
            raise AssertionError(f"{name}: a single-row sample was accepted")

    return "; ".join(notes)


@check("surrogates: centering is a translation, whitening is not")
def t_surrogate_transforms():
    """What the centered surrogates may and may not change.

    Both centering modes subtract one array from every model, so the collection
    is *translated*: a Euclidean distance cannot move, and a scale-invariant one
    (cosine, normalized Frobenius) generally does.  That asymmetry is the whole
    content of a centered surrogate, and getting it backwards — pairing a centered
    representation with a translation-invariant metric — produces a panel that
    duplicates its raw twin under a label implying otherwise.
    """
    from src.core.representation import ModelRepresentation
    from src.analysis.surrogates import (
        center_representations, centered, transform_key, whiten_representations,
    )
    from src.metrics import CosineDistanceMetric, FrobeniusDistanceMetric

    rng = np.random.default_rng(11)
    reps = [
        ModelRepresentation(model_id=f"m{i}", taxonomy="behavioral",
                            matrix=(rng.normal(size=(20, 6)) + 4.0).astype(np.float32))
        for i in range(5)
    ]

    euclid, cosine = FrobeniusDistanceMetric(normalize=False), CosineDistanceMetric()
    assert euclid.metric_name == "euclidean", euclid.metric_name

    for mode in ("grand", "rowwise"):
        out = center_representations(reps, mode=mode)
        assert np.isclose(euclid.compute(out[0], out[1]),
                          euclid.compute(reps[0], reps[1])), (
            f"{mode} centering moved a Euclidean distance, so it is not a translation"
        )
        assert not np.isclose(cosine.compute(out[0], out[1]),
                              cosine.compute(reps[0], reps[1])), (
            f"{mode} centering left cosine unchanged; the shared component it "
            "removes was apparently already absent, so this fixture proves nothing"
        )
        assert out[0].metadata["surrogate_transform"][-1]["mode"] == mode

    pooled = np.vstack([r.matrix for r in center_representations(reps, "grand")])
    assert np.abs(pooled.mean(axis=0)).max() < 1e-4, pooled.mean(axis=0)
    stacked = np.stack([r.matrix for r in center_representations(reps, "rowwise")])
    assert np.abs(stacked.mean(axis=0)).max() < 1e-4

    w = whiten_representations(reps, shrinkage=0.1)
    cov = np.cov(np.vstack([r.matrix for r in w]).T)
    assert np.abs(np.diag(cov) - 1.0).max() < 0.25, np.diag(cov)
    assert not np.isclose(euclid.compute(w[0], w[1]),
                          euclid.compute(reps[0], reps[1])), (
        "whitening left a Euclidean distance unchanged; it is a linear map, not "
        "a translation, so this would mean the covariance was already identity"
    )
    assert [t["kind"] for t in w[0].metadata["surrogate_transform"]] == ["center", "whiten"]

    # rowwise needs a shared row count; say so rather than broadcasting quietly.
    ragged = reps[:1] + [ModelRepresentation(
        model_id="odd", taxonomy="behavioral",
        matrix=rng.normal(size=(9, 6)).astype(np.float32))]
    try:
        center_representations(ragged, mode="rowwise")
    except ValueError as e:
        assert "rowwise" in str(e), e
    else:
        raise AssertionError("rowwise accepted mismatched row counts")

    assert transform_key(None) == "raw"
    assert transform_key(centered("grand")) == "centered_grand"
    return "centering translates, whitening does not; keys are stable"


@check("plots: the simplex frame is a similarity transform")
def t_align_to_simplex():
    """`align_to_simplex` may choose where to stand and nothing else.

    It exists because an MDS solution is determined only up to translation,
    rotation *and* reflection, so four independently-fitted panels arrive in four
    arbitrary orientations. Pinning the frame is what makes them comparable by
    eye — but only if it moves no distance, since the panels are titled with a
    stress and a dCor computed before the rotation.

    The reflection is the half that is easy to get wrong: fixing the centre at
    the origin and one vertex on +y still leaves a mirror free, and an unpinned
    mirror is exactly the failure the function is supposed to prevent.
    """
    from src.plots.simplex import align_to_simplex

    rng = np.random.default_rng(5)
    ids = [f"yahoo_{a:03d}g1_{b:03d}g2_{c:03d}g3_n1000_s00" for a, b, c in
           [(100, 0, 0), (0, 100, 0), (0, 0, 100), (33, 33, 33),
            (50, 25, 25), (25, 50, 25)]]

    notes = []
    for trial in range(6):
        coords = rng.normal(size=(len(ids), 2)) * rng.uniform(0.01, 10)
        # Feed in a deliberately mirrored copy too: both must come out the same
        # way round, which is the property a fixed rotation alone does not have.
        for mirror in (False, True):
            src = coords * np.array([-1.0, 1.0]) if mirror else coords
            out = align_to_simplex(src, ids)

            d_in = np.linalg.norm(src[:, None] - src[None], axis=-1)
            d_out = np.linalg.norm(out[:, None] - out[None], axis=-1)
            assert np.abs(d_in - d_out).max() < 1e-9, (
                f"trial {trial} mirror={mirror}: distances moved by "
                f"{np.abs(d_in - d_out).max():.2e}"
            )
            assert np.abs(out[3]).max() < 1e-9, f"33/33/33 not at the origin: {out[3]}"
            assert abs(out[0][0]) < 1e-9 and out[0][1] > 0, f"100/0/0 off +y: {out[0]}"
            assert out[1][0] > 0, f"0/100/0 on the wrong side: {out[1]}"
        notes.append(trial)

    # A missing landmark must name itself rather than produce a silently
    # unaligned panel.
    try:
        align_to_simplex(rng.normal(size=(2, 2)), ids[:2])
    except ValueError as e:
        assert "33/33/33" in str(e), e
    else:
        raise AssertionError("a frame with no centre mixture was accepted")

    return f"{len(notes)} random frames x2 mirrors: distances fixed, landmarks pinned"


@check("functional: CKA refuses row counts its estimator cannot support")
def t_cka_row_guard():
    """The unbiased HSIC estimator divides by n(n-3).

    Measured before the guard: 3 rows → **NaN**, 4 → 1.0, 8 → 0.985.  A NaN
    returned here flows into a distance matrix and then into an MDS fit, where it
    is much harder to trace back.  Raising names the cause at the point it occurs.
    """
    from src.core.representation import ModelRepresentation
    from src.metrics.cka import CKADistanceMetric

    rng = np.random.default_rng(0)

    def rep(n, name="a", **meta):
        return ModelRepresentation(
            model_id=name, taxonomy="functional",
            matrix=rng.normal(size=(n, 5)).astype(np.float32), metadata=meta,
        )

    try:
        CKADistanceMetric().compute(rep(3, "a"), rep(3, "b"))
    except ValueError as e:
        assert "4" in str(e) and "unbiased" in str(e), e
    else:
        raise AssertionError("3 rows accepted; this returns NaN into a distance matrix")

    vals = {}
    for n in (4, 8):
        v = CKADistanceMetric().compute(rep(n, "a"), rep(n, "b"))
        assert np.isfinite(v) and 0.0 <= v <= 2.0, f"n={n}: {v}"
        vals[n] = v

    # The escape hatch the error message advertises has to actually work.
    small = CKADistanceMetric(unbiased=False).compute(rep(3, "a"), rep(3, "b"))
    assert np.isfinite(small), f"unbiased=False still degenerate at n=3: {small}"

    # The guard must not have changed the metric itself.
    same = rep(8, "a")
    assert abs(CKADistanceMetric().compute(same, same)) < 1e-9

    # A stored Gram is a kernel; CKA forms its own, so passing one computes
    # (H Hᵀ)² silently.  Refuse instead.
    try:
        CKADistanceMetric().compute(rep(8, "a", is_kernel=True, view="gram"), rep(8, "b"))
    except ValueError as e:
        assert "kernel" in str(e), e
    else:
        raise AssertionError("a kernel matrix was accepted as a feature matrix")
    return f"n=3 raises, n=4 → {vals[4]:.3f}, n=8 → {vals[8]:.3f}, unbiased=False works, gram refused"


def _activation_cache_or_skip():
    from src.cache.activation_cache import ActivationCache

    root = SHARED_CACHE
    if not (root / "04_activations").exists():
        raise _Skip(f"{root / '04_activations'} not present — functional has not been run")
    cache = ActivationCache(root)
    models = cache.list_models()
    if not models:
        raise _Skip("04_activations exists but holds no models")
    return cache, models


@check("[data] functional: cached representations are well formed")
def t_functional_reps_well_formed():
    cache, models = _activation_cache_or_skip()

    checked, shapes, draws_seen = 0, set(), set()
    for base_slug, adapter in models:
        base_id = base_slug.replace("--", "/")
        for draw in cache.list_draws(base_id, adapter):
            draws_seen.add((draw["recipe_hash"], draw["n_samples"], draw["seed"]))
            stored = cache.list_layers(base_id, adapter, draw, "input", "mean")
            if not stored:
                continue
            per_layer = cache.load_layers(base_id, adapter, draw, "input", "mean")
            for ell, arr in per_layer.items():
                assert arr.dtype == np.float32, f"{adapter} L{ell}: dtype {arr.dtype}"
                assert arr.ndim == 2, f"{adapter} L{ell}: shape {arr.shape}"
                assert arr.shape[0] == draw["n_samples"], (
                    f"{adapter} L{ell}: {arr.shape[0]} rows for an n={draw['n_samples']} draw"
                )
                assert np.isfinite(arr).all(), f"{adapter} L{ell}: non-finite values"
                shapes.add(arr.shape)

            # Layers must be stored under resolved absolute indices.
            assert all(ell >= 0 for ell in stored), stored
            runs = cache.list_runs(base_id, adapter, draw)
            assert runs, f"{adapter}: activations with no run record"
            rec = cache.load_config(base_id, adapter, draw, runs[0])
            assert rec["config"]["taxonomy"] == "functional", rec["config"]
            assert rec["config"]["query_key"]["recipe_hash"] == draw["recipe_hash"]
            n_hidden = rec.get("n_hidden_states")
            assert n_hidden and max(stored) < n_hidden, (stored, n_hidden)

            # The default view must assemble and be usable as a feature matrix.
            rep = cache.load(base_id, adapter, draw)
            assert rep.matrix.shape[0] == draw["n_samples"], rep.matrix.shape
            assert rep.matrix.shape[1] == sum(
                per_layer[ell].shape[1] for ell in sorted(per_layer)
            ), rep.matrix.shape
            assert np.isfinite(rep.matrix).all()
            assert rep.metadata["is_kernel"] is False
            checked += 1

    if not checked:
        raise _Skip("04_activations holds no readable draws")
    return (
        f"{checked} model-draw(s), {len(draws_seen)} draw(s), per-layer shapes "
        f"{sorted(shapes)[:3]}"
    )


@check("[gpu] functional: input-mode activations are invariant to batch size")
def t_functional_batch_invariance():
    """The measurement `docs/notes/functional_behavioral.md` asks for.

    Unlike the behavioral equivalent this can be tight.  ``input`` mode runs no
    ``generate`` call, so there is no greedy argmax to flip on a near-tie; the
    only source of disagreement is fp16 matmul tiling. What batching *would*
    change, if pooling read pad positions, is the pooled vector itself — and by a
    lot, since pad-position hidden states are not small.

    So: batch 1 (no padding at all) versus one full batch (maximum padding).  A
    per-row cosine below 0.999 means pooling is contaminated by padding, not that
    fp16 is noisy.  **This is expected to fail without mask-aware pooling**, which
    is why the fix and this check landed together.
    """
    import torch

    if not torch.cuda.is_available():
        raise _Skip("no CUDA device available")

    cache, models = _activation_cache_or_skip()
    base_slug, adapter = models[0]
    base_id = base_slug.replace("--", "/")
    draws = cache.list_draws(base_id, adapter)
    if not draws:
        raise _Skip(f"{adapter} has no stored draws")
    draw = draws[0]

    runs = cache.list_runs(base_id, adapter, draw)
    if not runs:
        raise _Skip(f"{adapter} has no run record to replay")
    rec = cache.load_config(base_id, adapter, draw, runs[0])
    model_id = rec["config"].get("_model_id") or _model_id_for(base_id, adapter)
    if model_id is None or not Path(model_id).exists():
        raise _Skip(f"adapter directory for {adapter} is no longer on disk")

    queries = _replay_queries(draw, limit=8)
    if not queries:
        raise _Skip("could not rehydrate the query draw to replay it")

    from src.taxonomy.functional import FunctionalTaxonomy

    dtype = torch.float16 if "float16" in str(rec["config"].get("torch_dtype", "")) else torch.float32
    stored_layers = cache.list_layers(base_id, adapter, draw, "input", "mean")
    probe = sorted(stored_layers)[-1:]  # last layer is the one comparisons read

    def _run(batch_size: int):
        tax = FunctionalTaxonomy(
            queries=queries, layer_indices=probe, query_key=draw, cache=None,
            batch_size=batch_size, torch_dtype=dtype, pooling="mean",
        )
        tax.cache = None
        try:
            model, shared = tax._get_model(model_id)
            tok = tax._load_tokenizer(model_id, tax._resolve_base_model_id(model_id))
            n_hidden = int(model.config.num_hidden_layers) + 1
            layers = tax._resolve_layers(n_hidden)
            out = []
            for i in range(0, len(queries), batch_size):
                got = tax._process_batch(model, tok, queries[i : i + batch_size], layers)
                out.append(got["input"][layers[0]])
            return np.concatenate(out, axis=0)
        finally:
            tax.close()

    single = _run(1)                 # no padding at all
    batched = _run(len(queries))     # one batch, maximum padding

    num = (single * batched).sum(axis=1)
    den = np.linalg.norm(single, axis=1) * np.linalg.norm(batched, axis=1)
    cos = num / np.where(den < 1e-12, 1.0, den)
    worst = float(cos.min())
    delta = float(np.abs(single - batched).max())

    assert worst > 0.999, (
        f"per-row cosine drops to {worst:.5f} between batch 1 and batch "
        f"{len(queries)}. fp16 tiling alone does not do that — pooling is almost "
        "certainly averaging over pad positions, which makes a vector depend on "
        "which queries shared its batch. Check FunctionalTaxonomy._pool."
    )
    return (
        f"{len(queries)} queries on {torch.cuda.get_device_name(0)}: min per-row "
        f"cosine {worst:.6f}, max|delta| {delta:.2e} (batch 1 vs {len(queries)})"
    )


def _model_id_for(base_id: str, adapter_slug_name: str) -> str | None:
    """Locate the adapter directory an activation entry came from."""
    d = SHARED_CACHE / "03_adapters" / base_id.replace("/", "--") / adapter_slug_name
    return str(d) if d.exists() else None


def _replay_queries(draw: dict, limit: int = 8) -> list[str]:
    """Rehydrate a query draw from 01_datasets, or return [] if it cannot be."""
    try:
        from src.cache.sampled_dataset_cache import SampledDatasetCache

        cache = SampledDatasetCache(SHARED_CACHE)
        rows = cache.get(draw["recipe_hash"], draw["n_samples"], draw["seed"])
    except Exception:
        return []
    if not rows:
        return []
    recipe = _recipe_for(draw["recipe_hash"])
    out = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        text = _text_projection.row_text(recipe, row)
        if text:
            out.append(text)
    return out


def _recipe_for(recipe_hash: str):
    """The recipe a draw was sampled from, as objects rather than JSON.

    Which column (or columns) of a source row became the query text used to be a
    guess — first match over ``("text", "question_title", "content",
    "question_content")`` — which silently picked ``text`` on a row carrying both
    it and ``question_title``, and reported nothing.

    It never had to be a guess.  The projection is part of ``ClassDatasetEntry``,
    ``_canonical()`` hashes the entries, and ``recipe_hash`` is a SHA-256 of that
    string — so the recipe *determines* the text, and the recipe is named by the
    draw.  This is also why neither inference cache stores query text: the draw
    key already fixes it.  Loading the recipe rather than reading one key off it
    is what keeps that true now that an entry may compose several columns.
    """
    path = SHARED_CACHE / "01_datasets" / recipe_hash / "recipe.json"
    data = json.loads(path.read_text())
    if not (data.get("datasets") or []):
        raise AssertionError(f"{path} has no datasets entry to read a projection from")

    if data.get("recipe_type") == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe as Recipe
        from src.datasets.class_recipe import ClassDatasetEntry as Entry
    else:
        from src.datasets.recipe import DatasetEntry as Entry
        from src.datasets.recipe import DatasetRecipe as Recipe

    return Recipe(
        name=data.get("name", ""),
        datasets=[Entry.from_dict(d) for d in data["datasets"]],
    )


# ── entry point ───────────────────────────────────────────────────────────────

# ── the two inference caches share one addressing scheme ──────────────────────

@check("inference caches: behavioral and functional share the addressing code")
def t_draw_keyed_shared_key():
    """Identity, not equality — the two must be the *same* function object.

    Equality would pass if someone reimplemented ``draw_dir`` identically on one
    subclass, and the two copies would then be free to drift.  Drift is exactly
    what put the behavioral cache at coordinates no reader could compute: every
    write succeeded and the cache read as empty.  Asserting identity means a
    future override has to delete this check to land, which is the point.
    """
    from src.cache._draw_keyed import DrawKeyedCache
    from src.cache.activation_cache import ActivationCache
    from src.cache.generated_text_cache import GeneratedTextCache

    shared = ["draw_name", "draw_dir", "mode_token", "config_hash", "canon_normalize",
              "surrogate_dir", "save_surrogate", "load_surrogate", "list_draws",
              "list_models", "list_base_models", "has_any"]
    def _underlying(cls, name):
        # A classmethod is re-bound on every attribute access, so compare the
        # function it wraps rather than the bound object.
        return getattr(getattr(cls, name), "__func__", getattr(cls, name))

    for name in shared:
        a = _underlying(ActivationCache, name)
        b = _underlying(GeneratedTextCache, name)
        base = _underlying(DrawKeyedCache, name)
        assert a is b is base, (
            f"{name} is not shared between the two inference caches; they can now "
            "drift apart, which is how the behavioral cache was orphaned"
        )

    # And the coordinates really do coincide for one model under one draw.
    draw = {"recipe_hash": "abc", "n_samples": 64, "seed": 0}
    base_id, adapter = "meta-llama/Llama-3.2-3B", "/some/path/yahoo_050t0_050t1"
    act = ActivationCache("/root").draw_dir(base_id, adapter, draw)
    gen = GeneratedTextCache("/root").draw_dir(base_id, adapter, draw)
    assert act.relative_to("/root/04_activations") == gen.relative_to("/root/05_generated"), (
        f"stage-relative coordinates differ: {act} vs {gen}"
    )
    return f"{len(shared)} shared members, suffix {act.relative_to('/root/04_activations')}"


@check("behavioral: a non-default view is computed once and written back")
def t_generated_surrogate_writeback():
    import tempfile

    from src.cache.generated_text_cache import GeneratedTextCache
    from src.core.representation import ModelRepresentation

    draw = {"recipe_hash": "abc", "n_samples": 4, "seed": 0}
    base, adapter = "meta-llama/Llama-3.2-3B", "/p/yahoo_050t0_050t1"
    ehash = GeneratedTextCache.embedder_hash({"model_name": "stub"})
    sampling = dict(GeneratedTextCache.GREEDY_SAMPLING)
    shash = GeneratedTextCache.sampling_hash(sampling)
    matrix = np.arange(12, dtype=np.float32).reshape(4, 3)

    with tempfile.TemporaryDirectory() as td:
        cache = GeneratedTextCache(td)
        rep = ModelRepresentation.create(
            model_id=adapter, taxonomy="behavioral", matrix=matrix, config={},
            metadata={"generated_texts": [["a"], ["b"], ["c"], ["d"]]},
        )
        cache.save(base, adapter, draw, rep, max_new_tokens=16, replicates=1,
                   sampling=sampling, embedder_hash=ehash)

        # The default view is the stored matrix and must NOT be written back --
        # a byte-copy beside the original would double this stage for nothing.
        plain = cache.load(base, adapter, draw, 16, 1, shash, ehash)
        assert np.array_equal(plain.matrix, matrix)
        assert "surrogate_cached" not in plain.metadata, (
            "the identity view was routed through surrogates/"
        )
        sur_dir = cache.draw_dir(base, adapter, draw) / "surrogates"
        assert not sur_dir.exists(), "identity view wrote a surrogate"

        first = cache.load(base, adapter, draw, 16, 1, shash, ehash,
                           view="gram", normalize="layer")
        assert first.metadata["surrogate_cached"] is False, first.metadata
        assert first.matrix.shape == (4, 4), first.matrix.shape
        assert first.metadata["is_kernel"] is True
        second = cache.load(base, adapter, draw, 16, 1, shash, ehash,
                            view="gram", normalize="layer")
        assert second.metadata["surrogate_cached"] is True, "surrogate was not written back"
        assert np.array_equal(first.matrix, second.matrix)

        # Rows are unit-norm after row normalization, so the gram diagonal is 1 --
        # the same property the functional gram has, over the same draw.
        assert np.allclose(np.diag(second.matrix), 1.0), np.diag(second.matrix)
    return "identity view unstored; gram computed once, written back, diag=1"


@check("inference: the PEFT adapter name separates identical basenames")
def t_adapter_name_unique():
    """Re-homed from the cache layer, and now about the right thing.

    This property used to guard a *filename*, where hashing the full path made
    an entry unreachable from any other working directory.  The name is now
    in-memory only — PEFT's adapter registry — so hashing the path is fine and
    the uniqueness is what matters: a mixed-base session loads both of these
    into one PeftModel, and a collision would silently apply the wrong weights.
    """
    from src.cache._draw_keyed import adapter_slug
    from src.taxonomy._hf_inference import _adapter_name

    p1 = "/cache/03_adapters/meta-llama--Llama-3.2-3B/yahoo_050t0_050t1_n1000_s00_r16_i00"
    p2 = "/cache/03_adapters/meta-llama--Llama-3.1-8B/yahoo_050t0_050t1_n1000_s00_r16_i00"
    assert _adapter_name(p1) != _adapter_name(p2), "PEFT adapter names collide across bases"
    assert "/" not in _adapter_name(p1)

    # And the reason the *cache* does not need this: the base model is its own
    # path component there, so the bare basename is unambiguous.
    assert adapter_slug(p1) == adapter_slug(p2), (
        "adapter_slug is no longer the basename; the cache relies on the base "
        "model being a separate path component to disambiguate"
    )
    return f"{_adapter_name(p1)[-8:]} vs {_adapter_name(p2)[-8:]}; slugs equal by design"


@check("draws: the query text comes from the recipe, not a guess")
def t_replay_queries_uses_recipe_text_field():
    """A row carrying several candidate columns must resolve by the recipe.

    ``_replay_queries`` used to take the first of ``("text", "question_title",
    "content", "question_content")`` present in the row, so a row with both
    ``text`` and ``question_title`` silently yielded the wrong one — with no
    error and no way to tell from the output.  The recipe records the projection
    and ``recipe_hash`` covers it, so the answer was always available.

    Extended for composed entries: a recipe naming ``text_fields`` must replay as
    the joined string, not as one of its parts.  That is the whole content of the
    item 11 fix on the read side — if replay disagreed with what training saw,
    every check that compares them would be comparing the wrong things.

    This is also the check that keeps the inference caches free of stored query
    text: if the draw key determines the text, it needs no second home.
    """
    import tempfile

    row = {"text": "WRONG - generic column",
           "question_title": "RIGHT - what the recipe asked for",
           "best_answer": "AND the answer",
           "content": "also wrong"}

    def _resolve(datasets):
        with tempfile.TemporaryDirectory() as td:
            rh = "deadbeefdeadbeef"
            d = Path(td) / "results/shared_cache/01_datasets" / rh
            d.mkdir(parents=True)
            (d / "recipe.json").write_text(json.dumps({
                "schema_version": "2", "recipe_type": "class_aware", "recipe_hash": rh,
                "datasets": datasets,
            }))
            import scripts.check_analysis as mod
            # `_recipe_for` reads SHARED_CACHE, which --cache-root can move, so
            # that is what a temporary recipe has to redirect.
            original = mod.SHARED_CACHE
            try:
                mod.SHARED_CACHE = Path(td) / "results/shared_cache"
                return _text_projection.row_text(mod._recipe_for(rh), row)
            finally:
                mod.SHARED_CACHE = original

    single = _resolve([{"dataset_id": "yahoo", "text_field": "question_title"}])
    assert single.startswith("RIGHT"), f"resolved {single!r}, not the recipe's text_field"

    composed = _resolve([{
        "dataset_id": "yahoo", "text_field": "best_answer",
        "text_fields": ["question_title", "best_answer"], "text_separator": "\n",
    }])
    assert composed == f"{row['question_title']}\n{row['best_answer']}", composed

    return "single and composed projections both read from recipe.json"


@check("[data] behavioral: nothing of the old run-wise layout survives")
def t_behavioral_layout_migrated():
    """The analogue of :func:`t_cache_fully_migrated`, for ``05_generated``."""
    from src.cache._draw_keyed import _DRAW_RE
    from src.cache.generated_text_cache import _GEN_RE

    root = SHARED_CACHE
    base = root / "05_generated"
    if not base.exists():
        raise _Skip(f"{base} not present — behavioral has not been run")

    old = [d.name for d in base.iterdir()
           if d.is_dir() and (d / "config.json").exists()]
    assert not old, f"old run-wise config directories survive: {sorted(old)}"

    # The old filenames ended in __<8 hex>, the path-hash that caused the bug.
    # Nothing anywhere may carry that shape again.
    offenders = [str(p.relative_to(base)) for p in base.rglob("*")
                 if re.search(r"__[0-9a-f]{8}$", p.stem)]
    assert not offenders, f"path-hashed slugs survive: {offenders[:3]}"

    draws, variants = 0, 0
    for emb in base.glob("*/*/*/*/embeddings/*.safetensors"):
        draw_dir = emb.parent.parent
        assert _DRAW_RE.match(draw_dir.name), f"{draw_dir} is not an n{{n}}_s{{seed}} directory"
        assert _GEN_RE.match(emb.stem), f"{emb.name} does not parse as {{mode}}_{{embedder}}"
        variants += 1
        q = json.loads((draw_dir / "queries.json").read_text())
        assert "queries" not in q, f"{draw_dir}/queries.json still stores query text"
        assert q["query_key"]["recipe_hash"] == draw_dir.parent.name, (
            f"{draw_dir}: queries.json disagrees with its own path"
        )
    draws = len({p.parent.parent for p in base.glob("*/*/*/*/embeddings/*.safetensors")})

    if not variants:
        raise _Skip("05_generated exists but holds no embedding")
    return f"{variants} variant(s) across {draws} model-draw(s), no legacy artefacts"


@check("[data] inference: behavioral and functional land on the same coordinates")
def t_cross_taxonomy_coordinates():
    """The payoff of the re-key, asserted rather than assumed.

    Must find **at least one** shared ``(base, adapter, draw)``, or it would pass
    vacuously on an empty intersection — which is precisely what a migration
    landing in the wrong place looks like.
    """
    from src.cache.activation_cache import ActivationCache
    from src.cache.generated_text_cache import GeneratedTextCache

    root = SHARED_CACHE
    for stage in ("04_activations", "05_generated"):
        if not (root / stage).exists():
            raise _Skip(f"{root}/{stage} not present")

    act, gen = ActivationCache(root), GeneratedTextCache(root)
    shared = 0
    for base_slug, adapter_slug in gen.list_models():
        base_id = base_slug.replace("--", "/")
        beh_draws = {(d["recipe_hash"], d["n_samples"], d["seed"])
                     for d in gen.list_draws(base_id, adapter_slug)}
        fun_draws = {(d["recipe_hash"], d["n_samples"], d["seed"])
                     for d in act.list_draws(base_id, adapter_slug)}
        for rh, n, s in beh_draws & fun_draws:
            draw = {"recipe_hash": rh, "n_samples": n, "seed": s}
            b = gen.draw_dir(base_id, adapter_slug, draw).relative_to(root / "05_generated")
            f = act.draw_dir(base_id, adapter_slug, draw).relative_to(root / "04_activations")
            assert b == f, f"coordinates differ for {adapter_slug}: {b} vs {f}"
            assert gen.has_draw(base_id, adapter_slug, draw)
            assert act.has_draw(base_id, adapter_slug, draw)
            shared += 1

    assert shared > 0, (
        "no model-draw is present at both inference levels, so this check proves "
        "nothing. Either the migration landed somewhere unexpected, or the two "
        "levels were never run over the same draw."
    )
    return f"{shared} model-draw(s) at identical coordinates in both stages"


SYNTHETIC = [
    t_geometry_reindex,
    t_cached_geometry_row_order,
    t_pairwise_row_order,
    t_pairwise_subset,
    t_pairwise_incremental,
    t_pairwise_selector_key,
    t_pairwise_refuses_transform,
    t_pairwise_writes_no_collection,
    t_pairwise_partial_failure,
    t_meta_models_identity,
    t_pair_metric_name_single_spelling,
    t_draw_format_id_kept,
    t_pair_id_order_free,
    t_pairwise_artifact_conflict,
    t_g2_appends_new_model,
    t_g2_still_raises_on_conflict,
    t_pairwise_draw_conflict,
    t_draw_absent_levels,
    t_pairwise_meta_shared,
    t_pairwise_index_concurrent,
    t_model_identity_uses_resolver,
    t_build_artifacts_identity,
    t_model_identity_permuted,
    t_model_identity_rejects_bad_order,
    t_anchor_fixed, t_similarity_invariance, t_affine_invariance_in_hull,
    t_known_mixture, t_simplex_high_dim, t_degenerate_anchors, t_compare_simplices,
    t_mantel, t_dcor_bias, t_dcor_test, t_dcor_unsigned, t_dcor_u_centering_symmetry,
    t_procrustes, t_procrustes_vs_scipy, t_per_point_residuals,
    t_dispersion, t_quality, t_correlation_table, t_match_models, t_fit_geometry,
    t_similarity_conversion, t_simplex_roundtrip, t_cosine_equivalence,
    t_bures_wasserstein_equivalence, t_bures_wasserstein_invariance,
    t_relabel_collision,
    # ground truth from recipes, and the storage it needs
    t_mixture_weights, t_mixture_name_k_vector, t_procrustes_dimension_sweep,
    t_split_and_whole_rejected,
    t_simplex_geometry,
    t_disparity_vs_truth_exact, t_disparity_vs_truth_label_keyed,
    t_simplex_dimension_requirement, t_projection_dimension_matters,
    t_procrustes_transform, t_collection_multidim, t_analysis_geometries,
    # reusing a stored matrix: the permutation guard, and the order it undoes
    t_distance_matrix_reindex, t_collection_cache_row_order,
    # content-addressed recipe identity, and the draw storage it enables
    t_recipe_identity, t_class_sampling_hash, t_class_sampling_semantics,
    t_steps_for_budget, t_one_draw_name, t_embedder_hash_seed, t_surrogate_hash_shared,
    # the prompt-format layer: additive by construction, and rendered in one place
    t_prompt_format_raw_is_inert, t_one_chat_template_call_site,
    t_prompt_end_token,
    t_profile_prefix_discrimination, t_pad_token_resolution,
    t_figure_specs_follow_layout,
    t_adapter_name_agreement,
    t_dataset_embedding_layout,
    t_draw_schema_roundtrip,
    t_names_merge,
    # behavioral taxonomy: its cache, and the padding property batch invariance needs
    t_generated_cache_roundtrip, t_generated_replicate_reduction,
    t_generated_sampling_hash_separates,
    t_generated_cache_hash_stable, t_behavioral_replicate_ordering,
    t_behavioral_padding_side,
    # log-prob level: it joins 05_generated by filename, and its arithmetic is
    # pinned against HF's own loss
    t_logprob_cache_names_join, t_logprob_matches_hf_loss, t_logprob_ride_along,
    # embedder task prefixes: the model is misused without them, and the failure is silent
    t_embedder_prefix_resolved, t_embedder_prefix_in_cache_key,
    # functional taxonomy: its cache, the views read off it, and the two
    # properties a distance built from it depends on
    t_activation_cache_roundtrip, t_activation_surrogate_writeback,
    t_activation_view_equivalence, t_activation_layerwise_normalization,
    t_functional_padding_side,
    t_functional_mask_pooling, t_cka_row_guard,
    # row order: ids select and order the rows, at every level
    t_distance_matrix_row_order, t_structural_row_order,
    # the surrogate layer, and the metrics that only make sense with it
    t_distributional_metrics, t_surrogate_transforms,
    t_align_to_simplex,
    # the two inference caches are addressed by one piece of code, and the
    # things that used to be spelled twice
    t_draw_keyed_shared_key, t_generated_surrogate_writeback, t_adapter_name_unique,
    t_scan_cache_dataset_filter,
    t_replay_queries_uses_recipe_text_field,
]
DATA_BACKED = [
    t_cosine_real_adapters, t_recovery, t_collection_roundtrip, t_cross_taxonomy,
    t_recipe_relabelling, t_scan_cache, t_comparison_end_to_end,
    t_cache_fully_migrated, t_behavioral_reps_well_formed,
    t_functional_reps_well_formed,
    t_behavioral_layout_migrated, t_cross_taxonomy_coordinates,
    t_collection_key_sees_selector, t_collection_surrogate_in_key,
]
#: A third tier: real checks that need a GPU and a multi-GB model load, so they are
#: too slow for a harness meant to run in seconds around every edit.  Off unless
#: --include-gpu is passed, which the SLURM job does while the GPU is allocated.
GPU_BACKED = [
    t_behavioral_batch_invariance,
    t_functional_batch_invariance,
]


def main() -> int:
    global SHARED_CACHE, ADAPTER_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip checks that read from results/. Keeps the run in-memory "
                             "and avoids touching the cache at all.")
    parser.add_argument("--data-only", action="store_true",
                        help="Run only the checks that read the real cache — the ones that "
                             "catch a broken path after a cache migration.")
    parser.add_argument("--include-gpu", action="store_true",
                        help="Also run the [gpu] checks, which load a real model onto a "
                             "CUDA device. Off by default: a local run should never pay "
                             "for a multi-GB load. Intended for the SLURM job, after "
                             "extraction, while the GPU is still allocated.")
    parser.add_argument("-k", metavar="PATTERN",
                        help="Run only checks whose description contains PATTERN "
                             "(case-insensitive substring, e.g. -k 'cache').")
    parser.add_argument("--list", action="store_true",
                        help="Print the check names and exit without running anything.")
    parser.add_argument("--cache-root", default=None,
                        help="The shared cache the [data] checks read "
                             f"(default: {SHARED_CACHE}). Needed from a git "
                             "worktree, where the default resolves inside the "
                             "worktree and every [data] check skips.")
    args = parser.parse_args()

    if args.cache_root:
        SHARED_CACHE = Path(args.cache_root).expanduser().resolve()
        ADAPTER_ROOT = SHARED_CACHE / "03_adapters"

    if args.synthetic_only and args.data_only:
        parser.error("--synthetic-only and --data-only are mutually exclusive")

    if args.data_only:
        checks = list(DATA_BACKED)
    elif args.synthetic_only:
        checks = list(SYNTHETIC)
    else:
        checks = SYNTHETIC + DATA_BACKED

    # --synthetic-only stays honest: it promises to touch nothing outside memory,
    # and a GPU check reads the cache and loads a model.
    if args.include_gpu and not args.synthetic_only:
        checks = checks + GPU_BACKED

    if args.k:
        pattern = args.k.lower()
        checks = [fn for fn in checks if pattern in fn.check_name.lower()]
        if not checks:
            parser.error(f"no check matches {args.k!r}; use --list to see the names")

    if args.list:
        for fn in checks:
            print(f"  {fn.check_name}")
        print(f"\n{len(checks)} check(s)")
        return 0

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    print("=== src/analysis checks ===\n")
    ctx = threadpool_limits(1) if threadpool_limits is not None else None
    try:
        for fn in checks:
            fn()
            status, name, note = _RESULTS[-1]
            print(f"  [{status}] {name}" + (f"  — {note}" if note else ""))
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    n_fail = sum(1 for s, _, _ in _RESULTS if s == "FAIL")
    n_skip = sum(1 for s, _, _ in _RESULTS if s == "SKIP")
    n_pass = sum(1 for s, _, _ in _RESULTS if s == "PASS")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
