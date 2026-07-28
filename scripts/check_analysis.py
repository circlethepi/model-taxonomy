"""Verification checks for src/analysis.

Synthetic checks run everywhere; the data-backed checks are skipped with a note
when their inputs are absent, so the script is always runnable.

Usage:
    python scripts/check_analysis.py
    python scripts/check_analysis.py --synthetic-only
"""

from __future__ import annotations

import argparse
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
    mantel_test,
    matrix_correlation,
    per_point_residuals,
    point_dispersion,
    procrustes_compare,
    protest,
    shepard,
)
from src.core.geometry import GeometryResult

REPO = Path(__file__).parent.parent
ADAPTER_ROOT = REPO / "results/shared_cache/adapters"
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

    print("      true      :", np.round(rec.true, 3).tolist())
    print("      recovered :", np.round(rec.recovered, 3).tolist())
    print("      residuals :", np.round(rec.residuals, 4).tolist())
    assert abs(rec.rho) > 0.85, f"weak rank agreement: rho={rec.rho}"
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
    # adapter path, so that row is legitimately incomparable and must be nan
    # rather than an exception that takes the whole table down with it.
    off = table[~np.eye(len(labels), dtype=bool)]
    n_comparable = int(np.sum(~np.isnan(off)) // 2)
    assert n_comparable > 0, "no taxonomy pair was comparable"
    return f"{len(labels)} levels, {n_comparable} comparable pair(s), rest nan"


# ── entry point ───────────────────────────────────────────────────────────────

SYNTHETIC = [
    t_anchor_fixed, t_similarity_invariance, t_affine_invariance_in_hull,
    t_known_mixture, t_simplex_high_dim, t_degenerate_anchors, t_compare_simplices,
    t_mantel, t_procrustes, t_procrustes_vs_scipy, t_per_point_residuals,
    t_dispersion, t_quality, t_correlation_table, t_match_models, t_fit_geometry,
    t_similarity_conversion, t_simplex_roundtrip, t_cosine_equivalence,
]
DATA_BACKED = [
    t_cosine_real_adapters, t_recovery, t_collection_roundtrip, t_cross_taxonomy,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip checks that read from results/.")
    args = parser.parse_args()

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    checks = SYNTHETIC + ([] if args.synthetic_only else DATA_BACKED)

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
