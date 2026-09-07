#!/usr/bin/env python
"""How the number of models in a group moves each taxonomy level's score.

Every score this project reports -- dCor* against the simplex ground truth,
Procrustes disparity against it -- has been computed on a **16-model**
collection.  Nothing in that number says how much of it is the taxonomy level
and how much is the group size, and two properties of the estimators make the
question urgent rather than academic:

* **dCor\\*** (Szekely-Rizzo, bias-corrected) carries a ``1/(n(n-3))`` factor, so
  it is undefined at ``n=3`` and extremely noisy just above it.
* **Procrustes disparity** is MDS-mediated, and a configuration of few points is
  nearly degenerate under a similarity transform, so small groups can score
  *artificially well*.

Both are facts about the estimators.  What is not known is the size of the effect
on real taxonomy data, or where each level's score settles.  This script measures
it: read one pool of fine-tuned models, subsample groups of increasing size from
it, and score every group exactly the way the 16-model runs are scored.

Terminology
-----------
**surrogate**
    A representation view of a model at one taxonomy level -- one row of the
    figure grid.
**perspective**
    A surrogate together with a similarity metric -- one cell of the grid.
**canonical perspective**
    The single perspective designated as the default reading of a taxonomy
    level, so levels can be compared one-to-one without sweeping metrics.  The
    four are fixed in :data:`CANONICAL` and are the standing default recorded in
    ``docs/terminology.md``.
**variant A / variant B**
    Two ways of forming the matrices for a sampled group; see `The two variants`_.
**permute-and-partition**
    The subsampling scheme; see :func:`groups_for`.

The two variants
----------------
Each group is scored twice, because "include the reference models" has two
readings and they are not the same number.

**Variant B** builds the matrices over the ``n`` sampled models alone.

**Variant A** builds them over the ``n`` sampled models *plus* the three vertex
models and the even-mixture model, then scores only the ``n`` sampled rows.  The
four references inform the configuration without being scored.  What that means
concretely differs by estimator:

* Procrustes fits MDS to all ``n+4`` points and superimposes only the sampled
  ones on the truth, so the references pull the embedding.
* dCor* never embeds, so its only channel is the U-centring term, which is a mean
  over every row present.  Variant A is therefore
  :func:`~src.analysis.matrices.distance_correlation_restricted` -- centre over
  ``n+4``, sum over the sampled block -- while variant B is the ordinary
  :func:`~src.analysis.ground_truth.dcor_vs_truth`.

An earlier draft of this design claimed dCor* was *provably* identical between
the variants.  That holds only under the other reading, restrict-then-score:
cut both matrices to the sampled rows first and the extra models are genuinely
invisible, because a pairwise distance depends only on its two models.  Both
columns are carried here unconditionally so the difference is measured rather
than assumed; if they agree at every ``n`` that is a reportable fact about the
estimator, and if they do not, the "report it once" shortcut would have thrown
away a real effect.

What is deliberately not swept
------------------------------
One perspective per level, no metric sweep, no fleet transform.  The point of
this suite is the *size* axis, so anything else that moves would confound it.
Cost per group is milliseconds, and the pool is what was expensive.

Usage
-----
::

    python scripts/sweep_group_size.py \\
        --base-model allenai/OLMo-2-0425-1B-Instruct \\
        --outdir figures/simplex3_pool_olmo2_pool

    # the M=20 dry run, on whatever pool is already in the cache
    python scripts/sweep_group_size.py --n-grid 5,10,20 --replicates 20
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.bridge import fit_geometry  # noqa: E402
from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.discovery import scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.analysis.matrices import distance_correlation_restricted  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import mixture_weights, sort_by_mixture  # noqa: E402

#: The MDS seed every disparity here is fitted under.  Shared with the figure
#: suite so a score from this sweep and a score from ``crosslevel_scores.csv``
#: describe configurations fitted the same way.
MDS_SEED = suite.MDS_SEED


# ── the four canonical perspectives ───────────────────────────────────────────

def canonical_perspectives() -> dict[str, dict]:
    """``level -> {metric, and the selector keywords build_taxonomy_artifacts wants}``.

    Fixed rather than swept, and identical to the standing per-level defaults:
    the dataset level's mean embedding under euclidean, the structural and
    functional levels' last layer under cosine, and the behavioral level's
    per-query replicate mean under cosine.

    Read after ``suite.apply_architecture``: "the last layer" is a *position* in
    a stack, so it is derived from the checkpoint's layer count rather than
    written down.  Hidden states are indexed ``0..N_LAYERS`` with ``h0`` the
    embedding, so the final state is ``N_STATES - 1``; LoRA layers are indexed
    ``0..N_LAYERS-1``, so the final adapter layer is ``N_LAYERS - 1``.
    """
    return {
        "dataset_embedding": {
            "metric": "euclidean",
            "dataset_selector": {"n_samples": 1000, "seed": 0,
                                 "representation": "mean"},
            "embedder_hash": suite.DATASET_EMBEDDER,
        },
        "structural": {
            "metric": "cosine",
            "layers": [suite.N_LAYERS - 1],
            "projections": ["q", "k", "v", "o"],
        },
        "functional": {
            "metric": "cosine",
            "functional_selector": {
                "draw": suite.DRAW, "mode": "input", "pooling": "mean",
                "layers": [suite.N_STATES - 1], "view": "concat",
                "normalize": "layer", "max_new_tokens": None,
            },
        },
        "behavioral": {
            "metric": "cosine",
            "behavioral_selector": {
                "draw": suite.DRAW, "max_new_tokens": suite.MAX_NEW_TOKENS,
                "replicates": 16, "sampling_hash": suite.SAMP_SAMPLED,
                "embedder_hash": suite.EMBEDDER,
                "replicate_reduction": "mean", "view": "matrix",
                "normalize": "none", "representation": "matrix",
                "renormalize": True,
            },
        },
    }


# ── the pool ──────────────────────────────────────────────────────────────────

def reference_ids(ids: list[str], weights: np.ndarray) -> list[str]:
    """The models a variant-A matrix adds: the pure vertices and the even mixture.

    Identified from the weights rather than from their names, so this does not
    have to know how a mixture is spelled.  A vertex is a one-hot weight vector;
    the even mixture is the one closest to ``1/K`` in every coordinate, which is
    exact when ``K`` divides the grid and the rounded ``(33,33,33)`` label when it
    does not.

    Returns whatever subset is present, in vertex-then-even order.  A pool
    missing one of them is not an error here -- the dry run at M=20 may well not
    have every vertex -- but variant A then has fewer than four references, which
    is why the count is written into the CSV rather than assumed to be 4.
    """
    k = weights.shape[1]
    refs: list[str] = []
    for j in range(k):
        onehot = np.zeros(k)
        onehot[j] = 1.0
        hits = [m for m, w in zip(ids, weights) if np.allclose(w, onehot, atol=1e-9)]
        refs.extend(hits[:1])
    even = np.full(k, 1.0 / k)
    rest = [(float(np.abs(w - even).max()), m) for m, w in zip(ids, weights)
            if m not in refs]
    if rest:
        gap, m = min(rest)
        # 1/(2*grid) would be the exact tolerance; 0.01 is looser and still far
        # inside the nearest competitor on any grid this project uses.
        if gap <= 0.01:
            refs.append(m)
    return refs


def pool_matrices(index, ids, cache_root, levels, *, use_cache=True):
    """One M x M distance matrix per level, at that level's canonical perspective.

    Goes through :func:`~src.analysis.comparison.build_taxonomy_artifacts`, which
    computes with the plain ``_distances`` double loop and stores the assembled
    matrix in ``07_collections``.  That is the point: it never touches
    ``06_pairwise``.

    The pair store keys a single JSON dict on full absolute paths, read and
    written whole, so a 1004-model collection would put ~503,000 entries and
    ~176 MB through it *per perspective* -- a serialization wall rather than a
    compute one, and the compute it would save is 0.17-0.40 h of single-core
    metric calls.  The stored matrix is ~8 MB.
    """
    out = {}
    for level, spec in canonical_perspectives().items():
        if level not in levels:
            continue
        kwargs = {k: v for k, v in spec.items() if k != "metric"}
        t0 = time.time()
        dm, _ = build_taxonomy_artifacts(
            index, level, spec["metric"], cache_root=cache_root,
            n_components=(2,), use_cache=use_cache, id_scheme="model_id",
            label="group-size pool", **kwargs,
        )
        dm = dm.reindex(list(ids))
        print(f"  {level:<18} {dm.matrix.shape[0]:>5} models  "
              f"{spec['metric']:<10} {time.time() - t0:6.1f}s")
        out[level] = dm
    return out


# ── subsampling ───────────────────────────────────────────────────────────────

def groups_for(pool: list[str], n: int, n_replicates: int, seed: int):
    """``(group, shuffle_id, n_disjoint)`` triples, by permute-and-partition.

    Shuffle the pool, cut it into ``floor(M/n)`` **disjoint** groups, and repeat
    with a fresh shuffle until *n_replicates* have been collected.  Disjointness
    within a shuffle is what maximises the independence available at each size.

    **Not a bootstrap, and that is deliberate.**  Resampling *with* replacement
    would duplicate models, and a duplicated model is a coincident point: a zero
    off-diagonal that degenerates the MDS fit and inflates dCor*.  Drawing
    without replacement from a pool of i.i.d.-mixture models is marginally an
    exact i.i.d. sample of size ``n``, so group means stay unbiased.

    **The honest limit** is that replicates share models, so replicate spread is
    deflated by construction and reaches exactly zero at ``n = M``, where every
    replicate is the same group.  ``n_disjoint`` is returned, and written into
    the CSV, so a reader can see how much independence a given size actually had
    rather than inferring it from a small error bar.
    """
    if n > len(pool):
        raise ValueError(f"n={n} exceeds the pool of {len(pool)}")
    rng = np.random.default_rng(seed)
    n_disjoint = len(pool) // n
    out, shuffle = [], 0
    while len(out) < n_replicates:
        order = rng.permutation(len(pool))
        for c in range(n_disjoint):
            if len(out) == n_replicates:
                break
            block = order[c * n:(c + 1) * n]
            out.append(([pool[i] for i in block], shuffle, n_disjoint))
        shuffle += 1
    return out


def n_disjoint_of(pool, n: int) -> int:
    """How many disjoint groups of *n* one shuffle of *pool* yields."""
    return len(pool) // n


def hull_area(coords: np.ndarray) -> float:
    """Area of the convex hull of a group's ground-truth coordinates.

    A plain measurement of how much of the simplex a group covers, recorded so
    coverage can be conditioned on later without re-running anything.  Not used
    to select or weight groups here.

    ``nan`` when the hull is undefined -- fewer than three points, or points that
    are collinear, which ``QhullError`` reports rather than returning zero.
    """
    from scipy.spatial import ConvexHull, QhullError

    if coords.shape[0] < 3:
        return float("nan")
    try:
        return float(ConvexHull(coords).volume)  # 'volume' is area in 2-D
    except QhullError:
        return float("nan")


# ── scoring one group ─────────────────────────────────────────────────────────

def score_group(dm, group, refs, weights_of, vertex_names):
    """The four scores for one group, under both variants.

    *dm* is the pool matrix; *group* the sampled ids; *refs* the reference ids to
    add for variant A, already excluded from *group*.

    Variant B restricts the matrix first and scores what is left.  Variant A
    keeps the references in the matrix and removes them only from what is
    *scored*: the disparity fits MDS over ``n + len(refs)`` points and
    superimposes the sampled subset alone (``procrustes_compare`` reindexes onto
    the common ids), and the dCor* U-centres over the same enlarged set before
    summing over the sampled block.
    """
    def truth(ids):
        W = np.vstack([weights_of[m] for m in ids])
        return (simplex_distance_matrix(W, list(ids), vertex_names),
                simplex_geometry(W, list(ids), vertex_names))

    # -- variant B: the sampled models alone ----------------------------------
    dm_b = dm.reindex(list(group))
    tdm_b, tgeo_b = truth(group)
    dcor_b = dcor_vs_truth(dm_b, tdm_b)
    disparity_b = disparity_vs_truth(dm_b, tgeo_b, random_state=MDS_SEED,
                                     n_components=tgeo_b.coordinates.shape[1])

    # -- variant A: references in the matrix, out of the score ----------------
    wide = list(group) + [r for r in refs if r not in set(group)]
    dm_a = dm.reindex(wide)
    tdm_a, _ = truth(wide)
    dcor_a = distance_correlation_restricted(dm_a, tdm_a, list(group))
    geo_a = fit_geometry(dm_a, method="mds",
                         n_components=tgeo_b.coordinates.shape[1],
                         random_state=MDS_SEED)
    disparity_a = disparity_vs_truth(dm_a, tgeo_b, geometry=geo_a)

    return dcor_a, dcor_b, disparity_a, disparity_b, hull_area(
        np.asarray(tgeo_b.coordinates, dtype=np.float64))


# ── the sweep ─────────────────────────────────────────────────────────────────

FIELDS = ["level", "n", "replicate", "shuffle_id", "n_disjoint", "n_refs",
          "dcor_A", "dcor_B", "disparity_A", "disparity_B", "hull_area"]


def sweep(matrices, ids, weights, n_grid, n_replicates, seed, vertex_names):
    """One row per (level, n, replicate).  Returns the rows and the pool split."""
    weights_of = {m: w for m, w in zip(ids, weights)}
    refs = reference_ids(ids, weights)
    pool = [m for m in ids if m not in set(refs)]
    print(f"pool: {len(pool)} sampled + {len(refs)} reference "
          f"= {len(ids)} models")
    if len(refs) < weights.shape[1] + 1:
        print(f"  note: only {len(refs)} reference model(s) found; variant A "
              f"adds that many rather than {weights.shape[1] + 1}.")

    rows = []
    for n in n_grid:
        if n > len(pool):
            print(f"  n={n} skipped: larger than the {len(pool)}-model pool")
            continue
        drawn = groups_for(pool, n, n_replicates, seed)
        t0 = time.time()
        for level, dm in matrices.items():
            # Memoized on membership, because the scores are a function of the
            # *set* and an MDS fit is the expensive part -- ~34 s at n=1000.  It
            # pays for itself at the top of the grid, where there is only one
            # disjoint group and all 100 replicates are literally the same
            # models: without this they would be 100 identical 34-second fits.
            seen: dict[frozenset, tuple] = {}
            for r, (group, shuffle, n_disjoint) in enumerate(drawn):
                key = frozenset(group)
                try:
                    if key not in seen:
                        seen[key] = score_group(
                            dm, group, refs, weights_of, vertex_names)
                    a, b, da, db, area = seen[key]
                except Exception as exc:
                    print(f"    {level} n={n} rep={r} skipped -- "
                          f"{type(exc).__name__}: {exc}")
                    continue
                rows.append({
                    "level": level, "n": n, "replicate": r,
                    "shuffle_id": shuffle, "n_disjoint": n_disjoint,
                    "n_refs": len(refs),
                    "dcor_A": a, "dcor_B": b,
                    "disparity_A": da, "disparity_B": db, "hull_area": area,
                })
        distinct = len({frozenset(g) for g, _, _ in drawn})
        print(f"  n={n:<5} {len(drawn)} replicate(s) x {len(matrices)} level(s)"
              f"  ({distinct} distinct group(s), {n_disjoint_of(pool, n)} "
              f"disjoint per shuffle)  {time.time() - t0:6.1f}s")
    return rows


def variant_report(rows) -> str:
    """Which of the two readings the pipeline implements, measured not assumed.

    Verification step 3 of the plan: report ``dcor_A - dcor_B`` rather than
    asserting it is zero.  A non-zero difference means variant A is genuinely
    U-centring over the enlarged configuration and both columns must be carried
    into the write-up; a zero difference pins the pipeline to restrict-then-score
    and should be stated explicitly rather than left implicit.
    """
    if not rows:
        return "no rows scored, so no variant comparison"
    d_dcor = np.array([r["dcor_A"] - r["dcor_B"] for r in rows])
    d_disp = np.array([r["disparity_A"] - r["disparity_B"] for r in rows])
    finite = np.isfinite(d_dcor)
    worst = float(np.max(np.abs(d_dcor[finite]))) if finite.any() else float("nan")
    reading = ("centre-then-restrict (the references enter every centring term)"
               if worst > 1e-12 else
               "restrict-then-score (the references are invisible to dCor*)")
    return (f"variant sensitivity: max|dcor_A - dcor_B| = {worst:.3e}, "
            f"max|disparity_A - disparity_B| = "
            f"{float(np.nanmax(np.abs(d_disp))):.3e}\n"
            f"  dCor* under variant A is {reading}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--cache-root", default=None,
                    help=f"shared cache root (default: {suite.CACHE_ROOT})")
    ap.add_argument("--outdir", default="figures/simplex3_pool_olmo2_pool")
    ap.add_argument("--level", action="append", dest="levels",
                    choices=sorted(canonical_perspectives()),
                    help="restrict to one level; repeat for several")
    ap.add_argument("--dataset", action="append", dest="datasets",
                    default=None,
                    help="restrict the scan to a corpus, by recipe-name prefix "
                         "or dataset_id; repeat for several")
    ap.add_argument("--n-grid", default="5,10,20,50,100,200,500,1000",
                    help="comma-separated group sizes")
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the permute-and-partition shuffles")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the pool matrices, ignoring 07_collections "
                         "(still writes them back)")
    ap.add_argument("--draw-recipe-hash", default=suite.DRAW["recipe_hash"])
    ap.add_argument("--draw-n", type=int, default=suite.DRAW["n_samples"])
    ap.add_argument("--draw-seed", type=int, default=suite.DRAW["seed"])
    ap.add_argument("--draw-format-id", default=suite.DRAW["prompt_format_id"])
    args = ap.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve() \
        if args.cache_root else suite.CACHE_ROOT
    if not Path(cache_root).exists():
        raise SystemExit(
            f"no cache at {cache_root}. Pass --cache-root; from a git worktree "
            "the default is derived from the module's location and may not "
            "resolve to the checkout that holds the cache.")

    # Bind the architecture before reading the canonical perspectives: two of the
    # four name "the last layer", which is a position in a stack rather than a
    # number.
    suite.apply_architecture(suite.architecture(args.base_model))
    suite.DRAW = {"recipe_hash": args.draw_recipe_hash,
                  "n_samples": args.draw_n, "seed": args.draw_seed}
    if args.draw_format_id:
        suite.DRAW["prompt_format_id"] = args.draw_format_id
    print(f"model: {args.base_model}  ({suite.N_LAYERS} layers)")

    index = scan_cache(cache_root, base_model_id=args.base_model,
                       behavioral_draw=suite.DRAW,
                       functional_draw=suite.DRAW,
                       datasets=args.datasets)
    ids = sort_by_mixture([e.model_id for e in index.entries])
    if len(ids) < 5:
        raise SystemExit(
            f"only {len(ids)} model(s) in {cache_root} for {args.base_model}"
            + (f" restricted to {args.datasets}" if args.datasets else "")
            + ". The sweep needs at least 5 -- dCor* is undefined below 4.")
    weights = np.vstack([mixture_weights(m) for m in ids])
    vertex_names = [f"g{j + 1}" for j in range(weights.shape[1])]

    levels = args.levels or list(canonical_perspectives())
    print(f"building {len(levels)} pool matrix/matrices over {len(ids)} models")
    matrices = pool_matrices(index, ids, cache_root, levels,
                             use_cache=not args.no_cache)

    n_grid = [int(x) for x in args.n_grid.split(",") if x.strip()]
    rows = sweep(matrices, ids, weights, n_grid, args.replicates, args.seed,
                 vertex_names)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "group_size_scores.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} row(s) to {out}")
    print(variant_report(rows))


if __name__ == "__main__":
    main()
