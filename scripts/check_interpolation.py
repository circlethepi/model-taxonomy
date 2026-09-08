"""Verification checks for src/interp.

Synthetic checks run everywhere; the data-backed checks are skipped with a note
when the adapter cache is absent, so the script is always runnable.  Mirrors the
harness style of scripts/check_analysis.py, with its own registry so that file
is untouched.

Usage:
    python scripts/check_interpolation.py
    python scripts/check_interpolation.py --synthetic-only
    python scripts/check_interpolation.py --list
    python scripts/check_interpolation.py -k gram
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interp.fitting import (
    fit_interpolation,
    least_squares_weights,
    residual_sq,
    simplex_weights,
)
from src.interp.gram import compute_gram
from src.interp.loading import (
    ADAPTER_LEAF_RE,
    KEY_RE,
    discover_adapters,
    load_adapter,
    mixture_proportions,
    parse_adapter_leaf,
)
from src.interp.geometry import (
    COEFF_KINDS,
    coefficient_distance_matrix,
    compare_coefficients,
    compare_from_fits,
)
from src.interp.report import build_report, load_fits, short_labels

# Overridable so the data-backed checks can point at wherever the cache lives.
QWEN_CACHE = Path(
    os.environ.get("MODEL_TAXONOMY_CACHE", "/exp/nverma/model_taxonomy/shared_cache")
)
QWEN_ADAPTER_ROOT = QWEN_CACHE / "03_adapters"
QWEN_BASE_MODEL = "Qwen/Qwen3.5-4B"


# ── harness ───────────────────────────────────────────────────────────────────

_RESULTS: list[tuple[str, str, str]] = []
_CHECKS: list = []


class _Skip(Exception):
    pass


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
        wrapped.check_name = name
        _CHECKS.append(wrapped)
        return wrapped
    return deco


def _synthetic_factors(n=4, d_out=48, d_in=64, rank=4, n_blocks=3, seed=99):
    """Random A/B factors shaped like a small hybrid adapter.

    Block names deliberately mix the two families this package has to handle, so
    the synthetic path exercises the same key space as the real files.
    """
    rng = np.random.default_rng(seed)
    blocks = [(0, "in_proj_qkv"), (0, "in_proj_z"), (1, "o_proj")][:n_blocks]
    return {
        f"syn{i}": {
            blk: {
                "A": rng.normal(size=(rank, d_in)),
                "B": rng.normal(size=(d_out, rank)),
            }
            for blk in blocks
        }
        for i in range(n)
    }, blocks


def _dense_gram(weights, blocks, scale):
    """The thing compute_gram must equal, by explicit construction."""
    names = list(weights)
    vecs = {
        nm: np.concatenate(
            [(scale * weights[nm][b]["B"] @ weights[nm][b]["A"]).ravel() for b in blocks]
        )
        for nm in names
    }
    n = len(names)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            G[i, j] = float(vecs[names[i]] @ vecs[names[j]])
    return G


# ── synthetic checks ──────────────────────────────────────────────────────────

@check("gram: equals the dense <dW_i, dW_j>_F on the materialized product")
def t_gram_dense_equivalence():
    weights, blocks = _synthetic_factors()
    scale = 2.0
    got = compute_gram(weights, scale=scale, progress=False).total
    want = _dense_gram(weights, blocks, scale)
    rel = np.abs(got - want).max() / np.abs(want).max()
    assert rel < 1e-12, f"relative difference {rel:.3e}"
    return f"max rel diff {rel:.2e} over {len(blocks)} blocks"


@check("gram: the batched and pairwise code paths agree")
def t_gram_batched_vs_pairwise():
    weights, blocks = _synthetic_factors(n=5)
    batched = compute_gram(weights, scale=2.0, progress=False).total
    # An adapter with a different rank in one block forces the pairwise branch.
    rng = np.random.default_rng(3)
    mixed = {k: {b: dict(v) for b, v in d.items()} for k, d in weights.items()}
    blk = blocks[0]
    d_out, d_in = mixed["syn0"][blk]["B"].shape[0], mixed["syn0"][blk]["A"].shape[1]
    for nm in mixed:
        mixed[nm][blk] = {
            "A": np.vstack([weights[nm][blk]["A"], rng.normal(size=(1, d_in))]),
            "B": np.hstack([weights[nm][blk]["B"], rng.normal(size=(d_out, 1))]),
        }
    pairwise = compute_gram(mixed, scale=2.0, progress=False).total
    dense = _dense_gram(mixed, blocks, 2.0)
    rel = np.abs(pairwise - dense).max() / np.abs(dense).max()
    assert rel < 1e-12, f"pairwise branch differs from dense by {rel:.3e}"
    assert not np.allclose(batched, pairwise), "the mixed-rank case was not distinct"
    return f"pairwise branch matches dense to {rel:.2e}"


@check("gram: distances are a valid Euclidean distance matrix")
def t_gram_distances():
    weights, _ = _synthetic_factors(n=5)
    gram = compute_gram(weights, scale=2.0, progress=False)
    D = gram.distance_matrix()
    assert np.allclose(np.diag(D), 0.0), "nonzero diagonal"
    assert np.allclose(D, D.T), "not symmetric"
    for i, j, k in combinations(range(D.shape[0]), 3):
        assert D[i, k] <= D[i, j] + D[j, k] + 1e-9, f"triangle violated at {i},{j},{k}"
    C = gram.cosine_matrix()
    assert np.allclose(np.diag(C), 1.0) and C.max() <= 1.0 + 1e-12
    return f"{D.shape[0]}x{D.shape[0]}, triangle inequality holds"


@check("interpolation: exact recovery of a target inside the corner span")
def t_exact_recovery():
    """The sharpest available check, and it needs no real data.

    Stacking the corner factors gives B_T = [c1 B1 | c2 B2 | c3 B3] and
    A_T = vstack(A1, A2, A3), so B_T A_T = sum_k c_k B_k A_k *exactly*.  The fit
    must then return c.  The constructed target has rank 3r while the corners
    have rank r, so this also pins the equal-rank guard in compute_gram: if the
    batched reshape were applied to a mixed-rank block it would not survive.
    """
    weights, blocks = _synthetic_factors(n=3, rank=4)
    corners = list(weights)
    c = np.array([0.25, 0.5, 0.25])
    target = {}
    for blk in blocks:
        target[blk] = {
            "A": np.vstack([weights[nm][blk]["A"] for nm in corners]),
            "B": np.hstack([c[i] * weights[nm][blk]["B"] for i, nm in enumerate(corners)]),
        }
    allw = dict(weights)
    allw["target"] = target

    gram = compute_gram(allw, scale=2.0, progress=False)
    fit = fit_interpolation(gram, "target", corners, c)
    err = np.abs(fit.ols - c).max()
    assert err < 1e-10, f"coefficients off by {err:.3e}: {fit.ols}"
    # residual_sq subtracts nearly-equal O(1) terms, so a *norm* that should be
    # zero bottoms out around sqrt(eps) ~ 1e-8 regardless of correctness. Assert
    # against that floor, not below it. (Irrelevant on real data, where the
    # residual is ~0.84 -- eight orders of magnitude above the floor.)
    assert fit.rel_ols < 1e-6, f"residual should vanish, got {fit.rel_ols:.3e}"
    assert abs(fit.gamma - 1.0) < 1e-10, f"gamma should be 1, got {fit.gamma}"
    return f"coeffs to {err:.2e}, rel residual {fit.rel_ols:.2e}"


@check("interpolation: the simplex fit is the constrained optimum")
def t_simplex_optimum():
    rng = np.random.default_rng(11)
    for trial in range(20):
        M = rng.normal(size=(3, 6))
        Gc = M @ M.T
        g = rng.normal(size=3)
        w, face = simplex_weights(Gc, g)
        assert np.all(w >= -1e-12), f"negative weight {w}"
        assert abs(w.sum() - 1.0) < 1e-10, f"weights sum to {w.sum()}"
        assert set(np.nonzero(w > 1e-12)[0]) <= set(face), "support outside the face"
        obj = -2.0 * (w @ g) + w @ Gc @ w
        # A dense sweep of the simplex must not beat it.
        steps = 40
        for a in range(steps + 1):
            for b in range(steps + 1 - a):
                p = np.array([a, b, steps - a - b], dtype=float) / steps
                assert -2.0 * (p @ g) + p @ Gc @ p >= obj - 1e-9, (
                    f"grid point {p} beats the claimed optimum at trial {trial}"
                )
    return "20 random Grams, optimal against a 40-step grid"


@check("interpolation: residual_sq matches an explicit norm")
def t_residual_sq():
    weights, blocks = _synthetic_factors(n=4)
    corners = list(weights)[:3]
    target = list(weights)[3]
    gram = compute_gram(weights, scale=2.0, progress=False)
    ci = [gram.index(c) for c in corners]
    G = gram.total
    Gc, g = G[np.ix_(ci, ci)], G[gram.index(target), ci]
    w = np.array([0.2, 0.3, 0.5])

    vec = lambda nm: np.concatenate(  # noqa: E731
        [(2.0 * weights[nm][b]["B"] @ weights[nm][b]["A"]).ravel() for b in blocks]
    )
    explicit = vec(target) - sum(w[i] * vec(c) for i, c in enumerate(corners))
    want = float(explicit @ explicit)
    got = residual_sq(float(G[gram.index(target), gram.index(target)]), g, Gc, w)
    rel = abs(got - want) / want
    assert rel < 1e-12, f"residual off by {rel:.3e}"
    return f"rel diff {rel:.2e}"


@check("loading: hybrid attention module keys parse and stay unique")
def t_key_parsing():
    real = [
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
        "base_model.model.model.layers.30.linear_attn.out_proj.lora_B.weight",
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.31.self_attn.o_proj.lora_B.weight",
    ]
    for key in real:
        m = KEY_RE.search(key)
        assert m is not None, f"did not match: {key}"
    m = KEY_RE.search(real[0])
    assert (int(m.group("layer")), m.group("path").rsplit(".", 1)[-1]) == (0, "in_proj_qkv")

    # The regex the rest of the repo uses would drop the linear_attn keys; this
    # is the silent-loss failure mode src/interp/loading.py exists to avoid.
    from src.notebook.lora_weights import _KEY_RE as REPO_RE

    dropped = [k for k in real if REPO_RE.search(k) is None]
    assert len(dropped) == 2, f"expected the 2 linear_attn keys to be dropped, got {dropped}"

    # And the uniqueness assertion must actually fire.
    from safetensors.numpy import save_file
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        save_file(
            {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                    np.zeros((2, 3), dtype=np.float32),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                    np.zeros((3, 2), dtype=np.float32),
                "base_model.model.model.layers.0.cross_attn.q_proj.lora_A.weight":
                    np.zeros((2, 3), dtype=np.float32),
                "base_model.model.model.layers.0.cross_attn.q_proj.lora_B.weight":
                    np.zeros((3, 2), dtype=np.float32),
            },
            str(Path(td) / "adapter_model.safetensors"),
        )
        try:
            load_adapter(td)
        except ValueError as e:
            assert "Ambiguous" in str(e), f"wrong error: {e}"
        else:
            raise AssertionError("colliding module names did not raise")
    return "4 real keys parse; 2 would be dropped by the repo regex; collision raises"


@check("loading: adapter names parse with every optional suffix")
def t_adapter_leaf():
    cases = {
        "yahoo_025g1_050g2_025g3_n1000_s00_r16": "yahoo_025g1_050g2_025g3_n1000_s00",
        "yahoo_025g1_050g2_025g3_n1000_s00_r16_i00": "yahoo_025g1_050g2_025g3_n1000_s00",
        "yahoo_025g1_050g2_025g3_n1000_s00_r16_i00_b5008":
            "yahoo_025g1_050g2_025g3_n1000_s00",
        "yahoo_025g1_050g2_025g3_n1000_s00_r16_i00_b5008_fea27ccee":
            "yahoo_025g1_050g2_025g3_n1000_s00",
    }
    for name, want in cases.items():
        got = parse_adapter_leaf(name)
        assert got is not None and got["name"] == want, f"{name} -> {got}"
    assert parse_adapter_leaf("no_rank_here") is None

    # The repo's own copies stop at _b and fail outright on the _f suffix, which
    # is why this package carries its own.
    from src.analysis.identity import _ADAPTER_DIR_RE

    longest = max(cases)
    assert _ADAPTER_DIR_RE.match(longest) is None, "repo regex unexpectedly matched"
    assert ADAPTER_LEAF_RE.match(longest) is not None
    return f"{len(cases)} name shapes; the _f suffix is handled"


@check("loading: mixture proportions normalize by their sum, not by 100")
def t_mixture_normalization():
    """The even three-way label sums to 99, not 100 -- only the label rounds."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "experiment_meta.json").write_text(
            json.dumps({"dataset_name": "yahoo_033g1_033g2_033g3_n1000_s00"})
        )
        labels, w = mixture_proportions(p)
    assert labels == ["g1", "g2", "g3"], labels
    assert np.allclose(w, 1 / 3, atol=0, rtol=1e-15), w
    assert abs(w.sum() - 1.0) < 1e-15
    err = abs(w[0] - 1 / 3)
    assert err == 0.0, f"dividing by 100 would give {33 / 100}; got {w[0]}"
    return "033/033/033 -> exactly 1/3 each"


@check("report: short labels drop only what every name shares")
def t_short_labels():
    names = [
        "yahoo_025g1_050g2_025g3_n1000_s00_r16_i00_b5008_fea27ccee",
        "yahoo_100g1_000g2_000g3_n1000_s00_r16_i00_b5008_fea27ccee",
        "yahoo_000g1_000g2_100g3_n1000_s00_r16_i00_b5008_fea27ccee",
    ]
    got = short_labels(names)
    assert got[names[0]] == "025g1_050g2_025g3", got[names[0]]
    assert len(set(got.values())) == len(names), "labels collided"
    assert short_labels([]) == {} and short_labels(["a"]) == {"a": "a"}
    # Collision must fall back to the full names rather than lose identity.
    assert short_labels(["x_1", "x_1"]) == {"x_1": "x_1"}
    return "trims on segment boundaries, keeps trailing g3"


# ── data-backed checks ────────────────────────────────────────────────────────

def _require_adapters():
    if not QWEN_ADAPTER_ROOT.is_dir():
        raise _Skip(f"no adapter cache at {QWEN_ADAPTER_ROOT}")
    try:
        return discover_adapters(QWEN_ADAPTER_ROOT, base_model_slug=QWEN_BASE_MODEL)
    except FileNotFoundError as e:
        raise _Skip(str(e)) from None


def _real_report():
    refs = _require_adapters()
    truth, vertices = {}, None
    for ref in refs:
        parsed = mixture_proportions(ref.path)
        if parsed is None:
            continue
        vertices, truth[ref.name] = parsed[0], parsed[1]
    gram = compute_gram(
        {ref.name: load_adapter(ref.path) for ref in refs},
        scale=2.0, base_model_id=QWEN_BASE_MODEL, rank=16, lora_alpha=32,
        progress=False,
    )
    return gram, build_report(gram, truth, vertices)


def _synthetic_coeffs(n=9, k=3, noise=0.02, seed=7):
    """A ground-truth simplex plus a mildly perturbed fit of it.

    Includes the k pure corners so the configuration spans the whole simplex,
    exactly as the real collection does.
    """
    rng = np.random.default_rng(seed)
    true = np.vstack([np.eye(k), rng.dirichlet(np.ones(k), n - k)])
    fit = np.clip(true + rng.normal(scale=noise, size=true.shape), 1e-9, None)
    fit /= fit.sum(1, keepdims=True)
    ids = [f"m{i}" for i in range(n)]
    is_corner = [i < k for i in range(n)]
    return true, fit, ids, [f"g{i + 1}" for i in range(k)], is_corner


@check("geometry: the barycentric embedding is an isometry iff the rows sum to 1")
def t_coefficient_dm_isometry():
    from scipy.spatial.distance import pdist, squareform

    true, fit, ids, vertices, _ = _synthetic_coeffs()

    # Sum-to-1 rows: the embedding rescales every distance by 1/sqrt(2) and
    # changes nothing else, which is what lets the coefficient matrix and the
    # ground-truth matrix be compared without any unit reconciliation.
    dm = coefficient_distance_matrix(fit, ids, vertices)
    expected = squareform(pdist(fit)) / np.sqrt(2.0)
    err = float(np.abs(dm.matrix - expected).max())
    # 1e-6, not 1e-12: simplex_geometry stores its coordinates as float32, so
    # every distance built on top of it — the truth's included — carries float32
    # precision. The distances are O(1), so a few ULPs of float32 is ~1e-7. That
    # is six orders of magnitude below the disparities being reported and does
    # not affect any result, but a tighter bound here would be measuring the
    # upstream dtype rather than the isometry.
    assert err < 1e-6, f"embedding is not an isometry on sum-to-1 rows: {err:.3e}"
    assert dm.taxonomy == "coefficients", dm.taxonomy

    # Rows with differing sums (the raw unnormalized ols): the embedding
    # projects the gamma direction away, so this is NOT the same distance
    # matrix. Pinned so a future change that starts scoring `ols` fails here
    # rather than silently reporting a different quantity.
    gamma = np.linspace(0.6, 1.0, len(fit))[:, None]
    scaled = fit * gamma
    dm_scaled = coefficient_distance_matrix(scaled, ids, vertices)
    gap = float(np.abs(dm_scaled.matrix - squareform(pdist(scaled)) / np.sqrt(2.0)).max())
    assert gap > 1e-3, (
        f"expected the embedding to distort non-sum-to-1 rows, but the gap is "
        f"only {gap:.3e}"
    )
    return f"isometry to {err:.1e}; gamma-varying rows distort by {gap:.3f}"


@check("geometry: the true coefficients score a perfect match against themselves")
def t_coefficient_geometry_perfect():
    true, _, ids, vertices, is_corner = _synthetic_coeffs()

    # PCA is classical MDS: deterministic, pure numpy, and exact for a distance
    # matrix that is genuinely Euclidean in k-1 dimensions. Using it keeps this
    # check runnable regardless of the installed scikit-learn.
    rep = compare_coefficients(
        {"ols_normalized": true, "simplex": true},
        true, ids, vertices, is_corner=is_corner,
        method="pca", n_permutations=199,
    )
    for kind, r in rep.results.items():
        assert r.procrustes.disparity < 1e-12, f"{kind}: {r.procrustes.disparity:.3e}"
        assert r.procrustes_direct < 1e-12, f"{kind}: {r.procrustes_direct:.3e}"
        assert abs(r.dcor.statistic - 1.0) < 1e-9, f"{kind}: dcor {r.dcor.statistic}"
        assert float(r.residuals.max()) < 1e-6, f"{kind}: residual {r.residuals.max():.3e}"
    return "disparity < 1e-12 and dcor = 1 for both kinds"


@check("geometry: a worse fit scores strictly worse, and the metrics agree on order")
def t_coefficient_geometry_ordering():
    true, close, ids, vertices, is_corner = _synthetic_coeffs(noise=0.01)
    _, far, _, _, _ = _synthetic_coeffs(noise=0.10)

    rep = compare_coefficients(
        {"ols_normalized": close, "simplex": far},
        true, ids, vertices, is_corner=is_corner,
        method="pca", n_permutations=199,
    )
    good, bad = rep.results["ols_normalized"], rep.results["simplex"]
    assert good.procrustes.disparity < bad.procrustes.disparity, (
        f"procrustes did not separate them: {good.procrustes.disparity:.5f} vs "
        f"{bad.procrustes.disparity:.5f}"
    )
    assert good.dcor.statistic > bad.dcor.statistic, (
        f"dcor disagrees with procrustes: {good.dcor.statistic:.4f} vs "
        f"{bad.dcor.statistic:.4f}"
    )
    head = rep.headline()
    assert set(head) == {"ols_normalized", "simplex"}, head
    return (f"disparity {good.procrustes.disparity:.4f} < {bad.procrustes.disparity:.4f}, "
            f"dcor {good.dcor.statistic:.4f} > {bad.dcor.statistic:.4f}")


@check("geometry: at n_components = k-1 the MDS step does not distort the shape")
def t_coefficient_mds_is_identity():
    true, fit, ids, vertices, is_corner = _synthetic_coeffs(noise=0.05)
    try:
        rep = compare_coefficients(
            {"ols_normalized": fit}, true, ids, vertices, is_corner=is_corner,
            kinds=("ols_normalized",), method="mds", n_permutations=199,
        )
    except TypeError as e:
        # MDSGeometry passes metric_mds=/metric="precomputed", which older
        # scikit-learn does not accept. Report the environment, do not fail.
        raise _Skip(f"scikit-learn too old for MDSGeometry ({e})") from e

    r = rep.results["ols_normalized"]
    gap = abs(r.procrustes.disparity - r.procrustes_direct)
    assert r.stress < 1e-3, f"stress {r.stress:.3e} — the distances are not realisable"
    assert gap < 1e-3, (
        f"MDS changed the disparity by {gap:.3e} ({r.procrustes.disparity:.5f} vs "
        f"{r.procrustes_direct:.5f} direct); the embedding is losing shape"
    )
    return f"stress {r.stress:.1e}, disparity gap {gap:.1e}"


@check("geometry: mixtures_only drops exactly the corners")
def t_coefficient_mixtures_only():
    true, fit, ids, vertices, is_corner = _synthetic_coeffs()
    fits = {
        "targets": ids, "labels": ids, "is_corner": is_corner,
        "vertices": vertices, "true": true,
        "ols_normalized": fit, "simplex": fit,
    }
    full = compare_from_fits(fits, method="pca", n_permutations=99)
    trimmed = compare_from_fits(fits, mixtures_only=True, method="pca", n_permutations=99)

    assert len(full.model_ids) == len(ids), full.model_ids
    assert trimmed.model_ids == [i for i, c in zip(ids, is_corner) if not c]
    assert not any(trimmed.is_corner), trimmed.is_corner
    # The corners sit on the vertices and fit themselves exactly, so removing
    # them has to move the disparity. Identical numbers would mean the flag was
    # silently ignored.
    a = full.results["ols_normalized"].procrustes.disparity
    b = trimmed.results["ols_normalized"].procrustes.disparity
    assert abs(a - b) > 1e-9, (
        f"dropping {sum(is_corner)} corners left the disparity at {a:.6f}; the "
        "flag did not take effect"
    )
    return f"{len(full.model_ids)} -> {len(trimmed.model_ids)} adapters, {a:.4f} -> {b:.4f}"


@check("report: fits round-trip through save/load_fits bit-for-bit")
def t_fits_roundtrip():
    import tempfile

    weights, blocks = _synthetic_factors(n=4)
    gram = compute_gram(weights, scale=2.0, rank=4, lora_alpha=8, progress=False)
    names = gram.adapter_names
    truth = {
        names[0]: np.array([1.0, 0.0, 0.0]),
        names[1]: np.array([0.0, 1.0, 0.0]),
        names[2]: np.array([0.0, 0.0, 1.0]),
        names[3]: np.array([0.25, 0.5, 0.25]),
    }
    report = build_report(gram, truth, ["g1", "g2", "g3"], corners=names[:3])

    with tempfile.TemporaryDirectory() as td:
        report.save(td, figures=False)
        loaded = load_fits(td)

    assert loaded["targets"] == [f.target for f in report.fits]
    assert loaded["vertices"] == report.vertices
    assert loaded["is_corner"] == [f.is_corner for f in report.fits]
    for kind in COEFF_KINDS + ("true", "ols"):
        want = np.stack([getattr(f, kind) for f in report.fits])
        assert np.array_equal(loaded[kind], want), f"{kind} did not round-trip"
    assert np.array_equal(
        loaded["gamma"], np.array([f.gamma for f in report.fits])
    ), "gamma did not round-trip"
    return f"{len(loaded['targets'])} fits, float64 preserved bit-for-bit"


@check("[data] loading: the Qwen3.5 hybrid adapter yields 104 blocks over 32 layers")
def t_qwen_blocks():
    refs = _require_adapters()
    blocks = load_adapter(refs[0].path)
    assert len(blocks) == 104, f"{len(blocks)} blocks, expected 104"
    lin = {l for l, m in blocks if m in {"in_proj_qkv", "in_proj_z", "out_proj"}}
    sa = {l for l, m in blocks if m in {"q_proj", "k_proj", "v_proj", "o_proj"}}
    assert lin.isdisjoint(sa), f"layer families overlap: {sorted(lin & sa)}"
    assert lin | sa == set(range(32)), f"layers covered: {sorted(lin | sa)}"
    assert len(lin) == 24 and len(sa) == 8, f"{len(lin)} linear_attn, {len(sa)} self_attn"
    return f"{len(refs)} adapters, 104 blocks, 24 linear_attn + 8 self_attn layers"


@check("[data] interpolation: corners fit as the identity with zero residual")
def t_corner_self_consistency():
    _, report = _real_report()
    corner_fits = [f for f in report.fits if f.is_corner]
    assert len(corner_fits) == 3, f"{len(corner_fits)} corner fits"
    for f in corner_fits:
        i = report.corners.index(f.target)
        want = np.eye(3)[i]
        assert np.abs(f.ols - want).max() < 1e-9, f"{f.target} fit {f.ols}"
        assert f.residual_ols < 1e-9, f"{f.target} residual {f.residual_ols:.3e}"
    return "all 3 corners return their own unit vector"


@check("[data] interpolation: the recorded coefficients and shrinkage reproduce")
def t_real_numbers():
    _, report = _real_report()
    s = report.summary()
    assert s["n_mixtures"] == 13, s["n_mixtures"]
    assert abs(s["gamma_mean"] - 0.7274) < 0.005, f"gamma {s['gamma_mean']:.4f}"
    assert s["gamma_std"] < 0.06, f"gamma std {s['gamma_std']:.4f}"
    assert 0.70 <= s["rel_ols_mean"] <= 0.95, f"rel_ols {s['rel_ols_mean']:.4f}"
    assert s["coeff_l2_mean"] < 0.08, f"coeff L2 {s['coeff_l2_mean']:.4f}"
    # The finding that decides which estimator is primary.
    assert s["coeff_l2_mean"] < s["coeff_l2_simplex_mean"], (
        "normalized fit should beat the simplex-constrained one"
    )
    assert s["coeff_l2_vs_baseline"] > 5.0, f"only {s['coeff_l2_vs_baseline']:.1f}x chance"
    return (
        f"gamma={s['gamma_mean']:.4f}+-{s['gamma_std']:.4f}, "
        f"rel_ols={s['rel_ols_mean']:.4f}, coeff L2={s['coeff_l2_mean']:.4f}"
    )


@check("[data] gram: the cache round-trips exactly")
def t_gram_cache():
    import tempfile

    from src.interp.gram import load_or_compute_gram

    refs = _require_adapters()[:4]
    with tempfile.TemporaryDirectory() as td:
        a, pa = load_or_compute_gram(refs, cache_dir=td, progress=False)
        b, pb = load_or_compute_gram(refs, cache_dir=td, progress=False)
        assert not pa["cache_hit"] and pb["cache_hit"], (pa, pb)
        assert np.array_equal(a.per_block, b.per_block), "cache altered the values"
        assert a.blocks == b.blocks and a.adapter_names == b.adapter_names
        assert a.scale == b.scale == 2.0
    return f"hash {pa['gram_hash']}, float64 preserved bit-for-bit"


@check("[data] geometry: the coefficient configuration recovers the true simplex")
def t_coefficient_geometry_real():
    run_dir = Path("results/lora_interpolation")
    if not (run_dir / "fits" / "fits.safetensors").exists():
        raise _Skip(f"no interpolation run at {run_dir}")

    fits = load_fits(run_dir)
    rep = compare_from_fits(fits, n_permutations=999)
    norm = rep.results["ols_normalized"]
    simp = rep.results["simplex"]

    # The headline finding, and the one that has to keep holding: the rescaled
    # unconstrained fit reproduces the simplex better than the constrained one,
    # the same ordering coeff_l2 reports per adapter (0.053 vs 0.117).
    assert norm.procrustes.disparity < simp.procrustes.disparity, (
        f"ols_normalized {norm.procrustes.disparity:.5f} did not beat simplex "
        f"{simp.procrustes.disparity:.5f}"
    )
    assert norm.procrustes.disparity < 0.02, norm.procrustes.disparity
    assert norm.dcor.statistic > 0.95 and simp.dcor.statistic > 0.95, (
        norm.dcor.statistic, simp.dcor.statistic
    )
    # The MDS is a near-identity at k-1 dimensions; if this drifts, the
    # narrative in coefficient_geometry.md is no longer true.
    for r in (norm, simp):
        assert r.stress < 1e-3, f"{r.kind} stress {r.stress:.3e}"
        assert abs(r.procrustes.disparity - r.procrustes_direct) < 1e-3, r.kind
    return (f"procrustes {norm.procrustes.disparity:.5f} < "
            f"{simp.procrustes.disparity:.5f}, dcor {norm.dcor.statistic:.4f} / "
            f"{simp.dcor.statistic:.4f} over {len(rep.model_ids)} adapters")


# ── runner ────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-k", default=None, help="only checks whose name contains this")
    args = ap.parse_args(argv)

    selected = _CHECKS
    if args.synthetic_only:
        selected = [c for c in selected if not c.check_name.startswith("[data]")]
    if args.k:
        selected = [c for c in selected if args.k.lower() in c.check_name.lower()]

    if args.list:
        for c in selected:
            print(c.check_name)
        return 0

    # Small GEMMs: unpinned BLAS oversubscribes badly on this cluster.
    try:
        from threadpoolctl import threadpool_limits

        ctx = threadpool_limits(1)
    except ImportError:
        from contextlib import nullcontext

        ctx = nullcontext()
    with ctx:
        for c in selected:
            c()

    width = max((len(n) for _, n, _ in _RESULTS), default=0)
    for status, name, note in _RESULTS:
        print(f"{status:4}  {name:<{width}}  {note}")
    counts = {s: sum(1 for r in _RESULTS if r[0] == s) for s in ("PASS", "SKIP", "FAIL")}
    print(f"\n{counts['PASS']} passed, {counts['SKIP']} skipped, {counts['FAIL']} failed")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
