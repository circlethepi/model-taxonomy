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
    A perspective designated as a default reading of a taxonomy level, so levels
    can be compared one-to-one without sweeping metrics.  The seven are fixed in
    :func:`canonical_perspectives` and are the standing default recorded in
    ``docs/terminology.md``.
**scope**
    How much of a model a surrogate reads -- which layers, which projections.  A
    level with more than one canonical perspective has one per scope.
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
The standing per-level defaults and nothing else: no metric sweep, no fleet
transform, no layer sweep.  The point of this suite is the *size* axis, so
anything else that moves would confound it.  Cost per group is milliseconds, and
the pool is what was expensive.

Seven perspectives over five levels, not one per level: the structural level is
read at three scopes and the functional level at two, because "all of the stack
versus the end of it" is a question this axis can answer and the scopes do not
have to move together as ``n`` grows.

Usage
-----
::

    # the scores: every default here is this suite's pool and grid
    python figures/simplex_collection_size/sweep_group_size.py --jobs 7

    # then the figure, which only reads group_size_scores.csv
    python figures/simplex_collection_size/make_figures.py

    # add a size the first run skipped, keeping every row already on disk
    python figures/simplex_collection_size/sweep_group_size.py \\
        --n-grid 999 --append

    # the dry run, on whatever pool is already in the cache
    python figures/simplex_collection_size/sweep_group_size.py \\
        --n-grid 5,10,20 --replicates 20
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.bridge import fit_geometry  # noqa: E402
from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.discovery import scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.analysis.matrices import distance_correlation_restricted  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import mixture_weights, sort_by_mixture  # noqa: E402

#: This suite's own directory: the default place for its scores and figures.
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

#: The canonical checkout's cache.  A worktree has no ``results/`` of its own and
#: ``suite.CACHE_ROOT`` is derived from this module's location, so a run from one
#: would otherwise resolve to a directory that does not exist.
CACHE_ROOT = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/shared_cache")

#: The twelve ``figures/simplex3_olmo2`` mixtures that are **not** this pool's
#: reference models: the quarter-step grid minus the three vertices and the
#: 33/33/33 centroid.
#:
#: They have to be named, because no scan argument separates them from the pool:
#: that suite trains them on the same checkpoint, the same corpus, the same
#: recipe and the same draw as the 999.  Four of its sixteen *are* this pool's
#: references and are kept; a pool that also carried the other twelve would mix
#: 999 uniform draws with twelve grid points and stop being a uniform sample of
#: the simplex.  Named rather than derived, so that adding a mixture to that
#: suite cannot silently add one here.
SIMPLEX3_EXTRA = [
    "yahoo_000g1_025g2_075g3", "yahoo_000g1_050g2_050g3",
    "yahoo_000g1_075g2_025g3", "yahoo_025g1_000g2_075g3",
    "yahoo_025g1_025g2_050g3", "yahoo_025g1_050g2_025g3",
    "yahoo_025g1_075g2_000g3", "yahoo_050g1_000g2_050g3",
    "yahoo_050g1_025g2_025g3", "yahoo_050g1_050g2_000g3",
    "yahoo_075g1_000g2_025g3", "yahoo_075g1_025g2_000g3",
]

#: Group sizes.  Roughly geometric, because the ``1/(n(n-3))`` factor in dCor*
#: and the similarity-transform degeneracy in Procrustes both live on a ratio
#: scale.  It stops at 500 rather than 1000: the sampled pool is 999 models, so
#: 1000 is not available at all, and 999 would be a single group -- one
#: measurement with no spread, drawn a hundred times.  To add it later::
#:
#:     python figures/simplex_collection_size/sweep_group_size.py \
#:         --n-grid 999 --append
#:
#: which keeps every row already on disk and appends that size alone.
DEFAULT_N_GRID = "5,10,20,50,100,200,500"

#: The MDS seed every disparity here is fitted under.  Shared with the figure
#: suite so a score from this sweep and a score from ``crosslevel_scores.csv``
#: describe configurations fitted the same way.
MDS_SEED = suite.MDS_SEED


# ── the canonical perspectives ────────────────────────────────────────────────

def canonical_perspectives() -> dict[str, dict]:
    """``perspective -> {taxonomy, metric, label, and the selector keywords}``.

    The standing per-level defaults, not a sweep: one row per *scope* a level is
    read at, and nothing else.  Seven rows over five levels, because two levels
    are read at more than one scope and the contrast between the scopes is the
    point.

    ================== ============================================= ==========
    perspective        surrogate                                     metric
    ================== ============================================= ==========
    dataset_embedding  mean embedding of the training draw           euclidean
    structural_all_o   every layer, output projections only          cosine
    structural_all_qkvo every layer, all four projections            cosine
    structural_last_o  the last decoder layer's output projection    cosine
    functional_all     every hidden state                            cosine
    functional_last    the final hidden state alone                  cosine
    behavioral         per-query replicate mean, R=16                cosine
    ================== ============================================= ==========

    Two things worth stating, because neither is visible in the table:

    * **The functional rows mirror the structural ones on the only axis the
      level has.**  Structural narrows by layer *and* by projection; the
      functional level reads hidden states, which have no projection axis, so
      its mirror of "all layers versus the last one" is exactly the two rows
      here.  Asking for a functional output projection would name something the
      cache does not hold.
    * **"The last layer" is a position, not a number.**  It is derived from the
      checkpoint's own layer count after ``suite.apply_architecture``, because a
      literal 31 names the final layer only on a 32-layer model.  Hidden states
      are indexed ``0..N_LAYERS`` with ``h0`` the embedding, so the final state
      is ``N_STATES - 1``; LoRA layers are indexed ``0..N_LAYERS-1``, so the
      final adapter layer is ``N_LAYERS - 1``.

    The dataset level is read under ``euclidean`` alone -- plain row-wise L2
    between the mean vectors, ``FrobeniusDistanceMetric(normalize=False)``.  Its
    normalised sibling ``frobenius`` is a legitimate second reading of "euclidean
    norm" and the simplex3 drivers carry both; this suite does not, because the
    axis it measures is group size and every extra row is another curve to read
    against it rather than another reading of the level.
    """
    fsel = {"draw": suite.DRAW, "mode": "input", "pooling": "mean",
            "view": "concat", "normalize": "layer", "max_new_tokens": None}

    # **The two levels read `layers=None` differently, and only one of them
    # means "all".**  `_functional_reps` treats None as every stored hidden
    # state, which is what `simplex_suite._fsel(None)` labels "all N states
    # (reference)".  `_structural_matrix` does not: it forwards
    # ``layer_indices=layers if layers is not None else "last"`` to
    # `load_lora_weights`, so a structural None loads the **last layer alone**
    # and the "all layers" rows would have been silent duplicates of the
    # last-layer one.  The tell was the clock -- 63 s for sixteen layers when one
    # layer took 58 -- not the numbers, which are perfectly plausible.  Hence the
    # explicit range here, matching `simplex_suite.structural_group_specs`, which
    # spells its all-layer selections `range(N_LAYERS)` for the same reason.
    all_layers = list(range(suite.N_LAYERS))
    return {
        "dataset_embedding": {
            "taxonomy": "dataset_embedding",
            "label": "Dataset embedding\nmean · euclidean",
            "metric": "euclidean",
            "dataset_selector": {"n_samples": 1000, "seed": 0,
                                 "representation": "mean"},
            "embedder_hash": suite.DATASET_EMBEDDER,
        },
        "structural_all_o": {
            "taxonomy": "structural",
            "label": f"Structural (a)\nall {suite.N_LAYERS} layers · o_proj · cosine",
            "metric": "cosine",
            "layers": all_layers,
            "projections": ["o"],
        },
        "structural_all_qkvo": {
            "taxonomy": "structural",
            "label": f"Structural (b)\nall {suite.N_LAYERS} layers · q,k,v,o · cosine",
            "metric": "cosine",
            "layers": all_layers,
            "projections": ["q", "k", "v", "o"],
        },
        "structural_last_o": {
            "taxonomy": "structural",
            "label": f"Structural (c)\nlayer {suite.N_LAYERS - 1} · o_proj · cosine",
            "metric": "cosine",
            "layers": [suite.N_LAYERS - 1],
            "projections": ["o"],
        },
        "functional_all": {
            "taxonomy": "functional",
            "label": f"Functional (a)\nall {suite.N_STATES} states · cosine",
            "metric": "cosine",
            "functional_selector": {**fsel, "layers": None},
        },
        "functional_last": {
            "taxonomy": "functional",
            "label": f"Functional (b)\nh{suite.N_STATES - 1} · final state · cosine",
            "metric": "cosine",
            "functional_selector": {**fsel, "layers": [suite.N_STATES - 1]},
        },
        "behavioral": {
            "taxonomy": "behavioral",
            "label": "Behavioral\nR=16 per query · cosine",
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
        # The deterministic control for the row above, and the *only* pair in
        # this suite that differs by nothing but decoding: same draw, same
        # queries, same embedder, same pooling.  It is here because the sampled
        # row is noise-limited rather than perspective-limited, and that claim
        # is only legible next to a run with no sampling noise in it at all.
        #
        # `replicates` MUST be 1: BehavioralTaxonomy refuses R > 1 under
        # ``do_sample: false`` rather than storing R copies of one greedy
        # continuation.  ``replicate_reduction`` is then a no-op, kept only so
        # the selector reads the same shape as its sibling.  Greedy nulls
        # temperature/top_p/top_k in its sampling hash, so it lands in its own
        # cache entry and cannot collide with the R=16 rows over the same draw.
        "behavioral_greedy": {
            "taxonomy": "behavioral",
            "label": "Behavioral (greedy)\nR=1 deterministic · cosine",
            "metric": "cosine",
            "behavioral_selector": {
                "draw": suite.DRAW, "max_new_tokens": suite.MAX_NEW_TOKENS,
                "replicates": 1, "sampling_hash": suite.SAMP_GREEDY,
                "embedder_hash": suite.EMBEDDER,
                "replicate_reduction": "mean", "view": "matrix",
                "normalize": "none", "representation": "matrix",
                "renormalize": True,
            },
        },
    }


#: Keys of :func:`canonical_perspectives` that are not selector keywords.
_META_KEYS = ("taxonomy", "metric", "label")


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


#: The corpus-and-mixture head of an adapter directory name, up to the training
#: draw that follows it (``..._n1000_s00_r16_...``).
_LABEL_RE = re.compile(r"^(.*?_\d+g\d+(?:_\d+g\d+)*)_n\d+_s\d+")


def pin_pool(index, *, train_n=None, train_seed=None, exclude=None):
    """Cut a scan down to the models this experiment's pool is made of.

    Three filters, each answering a way the cache has already been shown to
    return more than this suite asked for:

    * **``train_n`` / ``train_seed``** -- the training recipe.  The
      dataset-size sweep trains the *same* mixtures on the same base model at
      other row counts, so an unfiltered scan of this base model returns those
      too and the sweep would score a pool that mixes training sizes.
    * **``exclude``** -- named mixtures.  A corpus filter cannot separate two
      experiments that share a corpus; the simplex3 grid and this pool are both
      yahoo on the same checkpoint, so the grid's non-reference points are
      removed by name.

    Filtering here rather than at ``scan_cache`` keeps the exclusion a property
    of the *pool definition*, visible in ``run_config.json``, rather than a
    scan argument that leaves no trace in the output.
    """
    from src.analysis.discovery import CacheIndex

    entries = list(index.entries)
    if train_n:
        entries = [e for e in entries if e.n_samples == train_n]
    if train_seed is not None and train_seed >= 0:
        entries = [e for e in entries if e.seed == train_seed]
    if exclude:
        drop = set(exclude)
        entries = [e for e in entries if mixture_of(e.model_id) not in drop]
    n_before = len(index.entries)
    if len(entries) != n_before:
        print(f"pool pinned: {n_before} -> {len(entries)} model(s)"
              + (f" (train_n={train_n}" if train_n else " (")
              + (f", seed={train_seed}" if train_seed is not None
                 and train_seed >= 0 else "")
              + (f", {len(exclude)} mixture(s) excluded" if exclude else "")
              + ")")
    return CacheIndex(entries, index.cache_root)


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def merge_rows(path: Path, new, key=("perspective", "n"), fields=None):
    """Existing rows from *path* that *new* does not supersede, plus *new*.

    What makes the CSV editable rather than write-once.  A row already on disk
    survives unless this run recomputed its whole ``key`` cell -- so
    ``--n-grid 999 --append`` adds one group size to a finished sweep without
    touching the eight already there, and re-running one perspective replaces
    exactly that perspective.

    Returns ``(rows, n_kept)``.  Rows come back as strings from the CSV and as
    floats from the sweep; both are written through ``csv.DictWriter``, which
    formats either identically, so the round trip does not change a value.
    """
    with path.open() as fh:
        old = list(csv.DictReader(fh))
    superseded = {tuple(str(r[k]) for k in key) for r in new}
    kept = [r for r in old
            if tuple(str(r.get(k, "")) for k in key) not in superseded]
    fields = fields or FIELDS
    kept = [{f: r.get(f, "") for f in fields} for r in kept]
    return kept + list(new), len(kept)


def mixture_of(model_id: str) -> str:
    """``yahoo_050g1_025g2_025g3`` out of a full adapter path.

    Used for the membership record and for pinning the pool, so a group written
    today can be re-read after the cache has moved.  Falls back to the bare
    directory name on anything that does not carry a mixture, rather than
    raising: a pool with an unparseable member should still record what it drew.
    """
    name = Path(model_id).name
    m = _LABEL_RE.match(name)
    return m.group(1) if m else name


def pool_matrices(index, ids, cache_root, perspectives, *, use_cache=True):
    """One M x M distance matrix per **perspective**, in declaration order.

    Keyed on the perspective name rather than on the taxonomy level, because two
    levels are read at more than one scope: ``structural`` alone contributes
    three matrices, and keying on the level would have the last of them silently
    overwrite the first two.

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
    for name, spec in canonical_perspectives().items():
        if name not in perspectives:
            continue
        kwargs = {k: v for k, v in spec.items() if k not in _META_KEYS}
        t0 = time.time()
        dm, _ = build_taxonomy_artifacts(
            index, spec["taxonomy"], spec["metric"], cache_root=cache_root,
            n_components=(2,), use_cache=use_cache, id_scheme="model_id",
            label="group-size pool", **kwargs,
        )
        dm = dm.reindex(list(ids))
        print(f"  {name:<20} {dm.matrix.shape[0]:>5} models  "
              f"{spec['metric']:<10} {time.time() - t0:7.1f}s", flush=True)
        out[name] = dm
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

FIELDS = ["perspective", "taxonomy", "n", "replicate", "shuffle_id",
          "n_disjoint", "n_refs",
          "dcor_A", "dcor_B", "disparity_A", "disparity_B", "hull_area"]

#: ``group_members.csv``: which models each replicate actually drew.
#:
#: Written beside the scores because the shuffle seed alone reproduces a group
#: only against an *identical pool in an identical order* -- add one adapter to
#: the cache and every permutation downstream of it changes.  The membership
#: file is the record that does not depend on that, and it is what makes a later
#: ``--n-grid 999 --append`` comparable with what is already here rather than
#: merely similar.  Members are written as **mixture labels**, which are short,
#: readable, and stable across cache moves in a way absolute paths are not.
GROUP_FIELDS = ["n", "replicate", "shuffle_id", "n_disjoint", "members"]


def sweep(matrices, ids, weights, n_grid, n_replicates, seed, vertex_names,
          specs=None):
    """One row per (perspective, n, replicate).

    Returns ``(score_rows, group_rows)`` -- the scores, and the membership record
    that says which models each replicate drew.
    """
    specs = specs or canonical_perspectives()
    weights_of = {m: w for m, w in zip(ids, weights)}
    refs = reference_ids(ids, weights)
    pool = [m for m in ids if m not in set(refs)]
    print(f"pool: {len(pool)} sampled + {len(refs)} reference "
          f"= {len(ids)} models")
    if len(refs) < weights.shape[1] + 1:
        print(f"  note: only {len(refs)} reference model(s) found; variant A "
              f"adds that many rather than {weights.shape[1] + 1}.")

    rows, group_rows = [], []
    for n in n_grid:
        if n > len(pool):
            print(f"  n={n} skipped: larger than the {len(pool)}-model pool")
            continue
        drawn = groups_for(pool, n, n_replicates, seed)
        for r, (group, shuffle, n_disjoint) in enumerate(drawn):
            group_rows.append({
                "n": n, "replicate": r, "shuffle_id": shuffle,
                "n_disjoint": n_disjoint,
                "members": " ".join(mixture_of(m) for m in group),
            })
        t0 = time.time()
        for pname, dm in matrices.items():
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
                    print(f"    {pname} n={n} rep={r} skipped -- "
                          f"{type(exc).__name__}: {exc}")
                    continue
                rows.append({
                    "perspective": pname,
                    "taxonomy": specs[pname]["taxonomy"],
                    "n": n, "replicate": r,
                    "shuffle_id": shuffle, "n_disjoint": n_disjoint,
                    "n_refs": len(refs),
                    "dcor_A": a, "dcor_B": b,
                    "disparity_A": da, "disparity_B": db, "hull_area": area,
                })
        distinct = len({frozenset(g) for g, _, _ in drawn})
        print(f"  n={n:<5} {len(drawn)} replicate(s) x {len(matrices)} "
              f"perspective(s)  ({distinct} distinct group(s), "
              f"{n_disjoint_of(pool, n)} disjoint per shuffle)  "
              f"{time.time() - t0:7.1f}s", flush=True)
    return rows, group_rows


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


def check_matrices(matrices, ids) -> int:
    """Plan verification step 6: are the pool matrices fit to score?

    *Complete* means every model the cache scan found has a row -- a pool whose
    training or extraction half died partway still produces a perfectly valid
    smaller matrix, and nothing downstream would notice the models that are
    missing from it.  *Symmetric with a zero diagonal* means what came back is a
    distance matrix and not a partly filled buffer.

    Returns a process exit status, so a bad pool stops the pipeline instead of
    being scored with holes in it.
    """
    want = set(ids)
    bad = 0
    for level, dm in sorted(matrices.items()):
        a = np.asarray(dm.matrix, dtype=float)
        have = list(dm.model_ids)
        problems = []
        missing = [m for m in ids if m not in set(have)]
        if missing:
            problems.append(
                f"{len(missing)} missing row(s), e.g. {missing[:3]}")
        extra = [m for m in have if m not in want]
        if extra:
            problems.append(f"{len(extra)} unexpected row(s), e.g. {extra[:3]}")
        if not np.isfinite(a).all():
            problems.append(
                f"{int((~np.isfinite(a)).sum())} non-finite entr(ies)")
        else:
            asym = float(np.abs(a - a.T).max())
            diag = float(np.abs(np.diag(a)).max())
            if asym > 1e-9:
                problems.append(f"asymmetric by {asym:.3e}")
            if diag > 1e-9:
                problems.append(f"diagonal up to {diag:.3e}")
        if problems:
            bad += 1
            print(f"  FAIL  {level:<18}  " + "; ".join(problems))
        else:
            print(f"  ok    {level:<18}  {a.shape[0]}x{a.shape[1]}, "
                  "symmetric, zero diagonal")
    n = len(matrices)
    print(f"\n{n - bad}/{n} matrix/matrices complete over {len(ids)} model(s)")
    return 1 if bad else 0


def fan_out(argv: list[str], jobs: int, tmp: Path, outdir: Path) -> None:
    """Score one perspective per process, then merge into one CSV.

    Worth having rather than doing by hand, because the cost here is **MDS, and
    MDS does not parallelise inside one process**.  At ``n=500`` the pool yields
    a single disjoint group per shuffle, so the membership memo never hits and
    every replicate is its own fit: ~1400 fits at 500 points for the whole
    suite, hours of one core.  Perspectives are completely independent -- they
    share only the pool matrices, which are read from ``07_collections`` -- so
    splitting on them is exact rather than approximate.

    The scores are those of a serial run: the groups are a function of the pool
    and the shuffle seed, not of which process drew them.  Only the row *order*
    differs, since the merge concatenates one perspective at a time where a
    serial run interleaves them, and nothing downstream reads row order.  What
    the merge has to repair is ``run_config.json``, which each process writes
    listing only the perspective it was given.
    """
    import subprocess

    tmp.mkdir(parents=True, exist_ok=True)
    names = list(canonical_perspectives())
    running: list[tuple[str, subprocess.Popen]] = []

    def reap(limit: int) -> None:
        while len(running) >= limit:
            for i, (nm, p) in enumerate(list(running)):
                if p.poll() is not None:
                    running.pop(i)
                    if p.returncode != 0:
                        raise SystemExit(
                            f"{nm} failed with exit {p.returncode}; see "
                            f"{tmp / nm}.log")
                    print(f"  done: {nm}", flush=True)
                    break
            else:
                time.sleep(2)

    for nm in names:
        reap(jobs)
        (tmp / nm).mkdir(parents=True, exist_ok=True)
        log = (tmp / f"{nm}.log").open("w")
        cmd = [sys.executable, str(HERE / "sweep_group_size.py"), *argv,
               "--jobs", "1", "--outdir", str(tmp / nm), "--perspective", nm]
        print(f"  start: {nm}", flush=True)
        running.append((nm, subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)))
    reap(1)

    header, body, config = None, [], None
    for nm in names:
        lines = (tmp / nm / "group_size_scores.csv").read_text().splitlines()
        header = header or lines[0]
        body += lines[1:]
        cfg = json.loads((tmp / nm / "run_config.json").read_text())
        if config is None:
            config = cfg
        else:
            config["perspectives"].update(cfg["perspectives"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "group_size_scores.csv").write_text(
        "\n".join([header, *body]) + "\n")
    # Membership does not depend on the perspective, so any process's copy is
    # the whole record; taking the first is not a shortcut.
    shutil.copy(tmp / names[0] / "group_members.csv",
                outdir / "group_members.csv")
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"merged {len(body)} row(s) from {len(names)} perspective(s) "
          f"into {outdir / 'group_size_scores.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT),
                    help="shared cache root")
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--perspective", action="append", dest="perspectives",
                    help="restrict to one perspective; repeat for several "
                         "(default: all seven)")
    ap.add_argument("--dataset", action="append", dest="datasets",
                    default=["yahoo"],
                    help="restrict the scan to a corpus, by recipe-name prefix "
                         "or dataset_id; repeat for several")
    ap.add_argument("--train-n", type=int, default=1000,
                    help="keep only adapters trained on this many rows; the "
                         "pool's own training size (0 disables the filter)")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="keep only adapters at this data seed (-1 disables)")
    ap.add_argument("--exclude-mixture", action="append", dest="exclude",
                    default=list(SIMPLEX3_EXTRA),
                    help="drop a mixture label from the pool; repeat. Use for "
                         "models that share the pool's recipe but were trained "
                         "for another experiment (default: the twelve "
                         "simplex3_olmo2 grid points that are not this pool's "
                         "reference models)")
    ap.add_argument("--keep-simplex3", action="store_true",
                    help="do not drop the simplex3_olmo2 grid points, i.e. "
                         "clear the --exclude-mixture default")
    ap.add_argument("--n-grid", default=DEFAULT_N_GRID,
                    help="comma-separated group sizes")
    ap.add_argument("--jobs", type=int, default=1,
                    help="score this many perspectives at once, in separate "
                         "processes, then merge. Same scores as a serial run; "
                         "MDS is the cost and does not thread")
    ap.add_argument("--tmp", default=None,
                    help="scratch directory for --jobs (default: <outdir>/.fanout)")
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing group_size_scores.csv rather "
                         "than replacing it: rows for a (perspective, n) this "
                         "run computed are replaced, every other row is kept. "
                         "This is how a size the first run skipped is added "
                         "later without recomputing the rest")
    ap.add_argument("--replicates", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the permute-and-partition shuffles")
    ap.add_argument("--check-matrices", action="store_true",
                    help="build the pool matrices, verify they are complete, "
                         "symmetric and zero-diagonal, then exit without "
                         "scoring (plan verification step 6)")
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the pool matrices, ignoring 07_collections "
                         "(still writes them back)")
    ap.add_argument("--draw-recipe-hash", default=suite.DRAW["recipe_hash"])
    ap.add_argument("--draw-n", type=int, default=suite.DRAW["n_samples"])
    ap.add_argument("--draw-seed", type=int, default=suite.DRAW["seed"])
    ap.add_argument("--draw-format-id", default=suite.DRAW["prompt_format_id"])
    args = ap.parse_args()
    if args.keep_simplex3:
        args.exclude = None

    if args.jobs > 1 and not args.check_matrices:
        # Re-dispatch to one process per perspective.  argv is forwarded intact
        # apart from the flags fan_out sets itself, so every knob keeps meaning
        # what it means here.
        drop = {"--jobs", "--outdir", "--perspective", "--tmp"}
        argv, skip = [], False
        for tok in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if tok in drop:
                skip = True
                continue
            if any(tok.startswith(d + "=") for d in drop):
                continue
            argv.append(tok)
        outdir = Path(args.outdir)
        fan_out(argv, args.jobs,
                Path(args.tmp) if args.tmp else outdir / ".fanout", outdir)
        return

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
    index = pin_pool(index, train_n=args.train_n, train_seed=args.train_seed,
                     exclude=args.exclude)
    ids = sort_by_mixture([e.model_id for e in index.entries])
    if len(ids) < 5:
        raise SystemExit(
            f"only {len(ids)} model(s) in {cache_root} for {args.base_model}"
            + (f" restricted to {args.datasets}" if args.datasets else "")
            + ". The sweep needs at least 5 -- dCor* is undefined below 4.")
    weights = np.vstack([mixture_weights(m) for m in ids])
    vertex_names = [f"g{j + 1}" for j in range(weights.shape[1])]

    specs = canonical_perspectives()
    unknown = set(args.perspectives or ()) - set(specs)
    if unknown:
        raise SystemExit(f"unknown perspective(s) {sorted(unknown)}. "
                         f"Choose from {sorted(specs)}")
    names = args.perspectives or list(specs)
    print(f"building {len(names)} pool matrix/matrices over {len(ids)} models")
    matrices = pool_matrices(index, ids, cache_root, names,
                             use_cache=not args.no_cache)

    if args.check_matrices:
        raise SystemExit(check_matrices(matrices, ids))

    n_grid = [int(x) for x in args.n_grid.split(",") if x.strip()]
    rows, group_rows = sweep(matrices, ids, weights, n_grid, args.replicates,
                             args.seed, vertex_names, specs=specs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "group_size_scores.csv"
    #: The variant check is arithmetic on floats, and ``merge_rows`` hands back
    #: the kept rows as strings from the CSV (see its docstring).  Report on the
    #: rows this run actually scored: the kept ones were reported when they were
    #: computed, and re-checking them here would only re-derive that.
    scored = rows
    n_kept = 0
    if args.append and out.exists():
        rows, n_kept = merge_rows(out, rows)
    write_csv(out, FIELDS, rows)
    groups = outdir / "group_members.csv"
    if args.append and groups.exists():
        group_rows, _ = merge_rows(groups, group_rows,
                                   key=("n", "replicate"), fields=GROUP_FIELDS)
    write_csv(groups, GROUP_FIELDS, group_rows)

    record = {
        "base_model": args.base_model,
        "cache_root": str(cache_root),
        "datasets": args.datasets,
        "train_n": args.train_n, "train_seed": args.train_seed,
        "excluded_mixtures": sorted(args.exclude or []),
        "pool_size": len(ids),
        "n_grid": n_grid, "replicates": args.replicates,
        "shuffle_seed": args.seed,
        "mds_seed": MDS_SEED,
        "draw": suite.DRAW,
        "perspectives": {k: {"taxonomy": v["taxonomy"], "metric": v["metric"],
                             "label": v["label"].replace("\n", " · ")}
                         for k, v in specs.items() if k in names},
        "pool": [mixture_of(m) for m in ids],
    }
    (outdir / "run_config.json").write_text(json.dumps(record, indent=2) + "\n")

    print(f"\nwrote {len(rows)} row(s) to {out}"
          + (f" ({n_kept} kept from the previous run)" if args.append else ""))
    print(f"wrote {len(group_rows)} group(s) to {groups}")
    print(f"wrote {outdir / 'run_config.json'}")
    print(variant_report(scored))


if __name__ == "__main__":
    main()
