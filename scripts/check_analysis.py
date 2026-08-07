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
ADAPTER_ROOT = REPO / "results/shared_cache/03_adapters"
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

    root = REPO / "results/shared_cache"
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


@check("[data] discovery: the cache scan joins adapters to their recipes")
def t_scan_cache():
    from src.analysis import scan_cache

    root = REPO / "results/shared_cache"
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

    root = REPO / "results/shared_cache"
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
            f"p={_show(rep['protest_p_value'])}"
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

    root = REPO / "results/shared_cache/02_dataset_embeddings"
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

    root = REPO / "results/shared_cache/01_datasets"
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

    root = REPO / "results/shared_cache"
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

    root = REPO / "results/shared_cache"
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
    d = REPO / "results/shared_cache/03_adapters" / base_id.replace("/", "--") / adapter_slug_name
    return str(d) if d.exists() else None


def _replay_queries(draw: dict, limit: int = 8) -> list[str]:
    """Rehydrate a query draw from 01_datasets, or return [] if it cannot be."""
    try:
        from src.cache.sampled_dataset_cache import SampledDatasetCache

        cache = SampledDatasetCache(REPO / "results/shared_cache")
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
    path = REPO / "results/shared_cache/01_datasets" / recipe_hash / "recipe.json"
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
            original = mod.REPO
            try:
                mod.REPO = Path(td)
                return _text_projection.row_text(mod._recipe_for(rh), row)
            finally:
                mod.REPO = original

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

    root = REPO / "results/shared_cache"
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

    root = REPO / "results/shared_cache"
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
    t_anchor_fixed, t_similarity_invariance, t_affine_invariance_in_hull,
    t_known_mixture, t_simplex_high_dim, t_degenerate_anchors, t_compare_simplices,
    t_mantel, t_procrustes, t_procrustes_vs_scipy, t_per_point_residuals,
    t_dispersion, t_quality, t_correlation_table, t_match_models, t_fit_geometry,
    t_similarity_conversion, t_simplex_roundtrip, t_cosine_equivalence,
    t_relabel_collision,
    # ground truth from recipes, and the storage it needs
    t_mixture_weights, t_split_and_whole_rejected, t_simplex_geometry,
    t_simplex_dimension_requirement, t_projection_dimension_matters,
    t_procrustes_transform, t_collection_multidim, t_analysis_geometries,
    # content-addressed recipe identity, and the draw storage it enables
    t_recipe_identity, t_one_draw_name, t_embedder_hash_seed, t_surrogate_hash_shared,
    t_dataset_embedding_layout,
    t_draw_schema_roundtrip,
    t_names_merge,
    # behavioral taxonomy: its cache, and the padding property batch invariance needs
    t_generated_cache_roundtrip, t_generated_replicate_reduction,
    t_generated_sampling_hash_separates,
    t_generated_cache_hash_stable, t_behavioral_replicate_ordering,
    t_behavioral_padding_side,
    # embedder task prefixes: the model is misused without them, and the failure is silent
    t_embedder_prefix_resolved, t_embedder_prefix_in_cache_key,
    # functional taxonomy: its cache, the views read off it, and the two
    # properties a distance built from it depends on
    t_activation_cache_roundtrip, t_activation_surrogate_writeback,
    t_activation_view_equivalence, t_activation_layerwise_normalization,
    t_functional_padding_side,
    t_functional_mask_pooling, t_cka_row_guard,
    # the two inference caches are addressed by one piece of code, and the
    # things that used to be spelled twice
    t_draw_keyed_shared_key, t_generated_surrogate_writeback, t_adapter_name_unique,
    t_replay_queries_uses_recipe_text_field,
]
DATA_BACKED = [
    t_cosine_real_adapters, t_recovery, t_collection_roundtrip, t_cross_taxonomy,
    t_recipe_relabelling, t_scan_cache, t_comparison_end_to_end,
    t_cache_fully_migrated, t_behavioral_reps_well_formed,
    t_functional_reps_well_formed,
    t_behavioral_layout_migrated, t_cross_taxonomy_coordinates,
    t_collection_key_sees_selector,
]
#: A third tier: real checks that need a GPU and a multi-GB model load, so they are
#: too slow for a harness meant to run in seconds around every edit.  Off unless
#: --include-gpu is passed, which the SLURM job does while the GPU is allocated.
GPU_BACKED = [
    t_behavioral_batch_invariance,
    t_functional_batch_invariance,
]


def main() -> int:
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
    args = parser.parse_args()

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
