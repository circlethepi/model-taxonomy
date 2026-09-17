#!/usr/bin/env python
"""Score the initsweep: how much of a taxonomy score is LoRA initialisation luck.

**initsweep** -- a sweep over ``lora_init_seed``, the seed passed to
``torch.manual_seed()`` immediately before PEFT initialises the LoRA ``A`` and
``B`` matrices, holding the corpus, the mixture grid, the training draw, the
dataset seeds, the rank and every other optimizer setting fixed.  See
``docs/terminology.md`` and ``docs/notes/init_seed_sweep.md``.

**surrogate** -- a read-time view of one taxonomy level (a choice of layers,
projections, pooling and metric), as opposed to the level itself.  The canonical
surrogates are defined once in
``figures/simplex_collection_size/sweep_group_size.py`` and imported here rather
than copied, so this sweep stays comparable with the rank, group-size and nsweep
figures next door.

The collection is 160 adapters: the same 16-point 25% simplex over three yahoo
topic groups on OLMo-2-0425-1B-Instruct, at rank 16 on one 1000-row draw (seed
0), trained from **ten initialisations**, ``i00 .. i09``.  The ``i00`` sixteen
are ``simplex3_olmo2``'s own adapters -- content-addressed caching made that seed
a cache hit rather than a retrain -- so this sweep passes *through* the standing
experiment rather than beside it.

What is scored
--------------
Four surrogates, one per model-level read, and **no dataset row**:

=================== ==============================================  =========
perspective         surrogate                                       metric
=================== ==============================================  =========
structural_all_o    every layer, output projections only            cosine
functional_all      every hidden state                              cosine
behavioral          per-query mean over the R=16 replicates         cosine
behavioral_greedy   R=1 deterministic decoding                      cosine
=================== ==============================================  =========

The two behavioral rows are a **pair that differs by decoding alone** -- same
draw, same queries, same embedder, same pooling -- and they are never drawn on
one axes; every figure comes in a ``_sampled`` and a ``_greedy`` variant.  See
``make_initsweep_figures.py``.

``--perspective`` still reaches every canonical surrogate, ``dataset_embedding``
included.  Leaving the dataset row out of the default set also leaves out the
sweep's one **null control**: the initialisation seed is not in the draw path, so
all ten seeds read the same sixteen draws and that row's ten scores must come out
*exactly* equal -- a spread there would be a finding about pipeline determinism
rather than about initialisation.  Run
``--perspective dataset_embedding --skip-pool`` to take that reading; it is
seconds of CPU, since no adapter is touched.

This script writes the numbers for the three analyses of the design note; the
plotting lives in ``make_initsweep_figures.py`` and reads only these CSVs.

Analysis A -- ten replicates of the standing score  (``initsweep_scores.csv``)
-----------------------------------------------------------------------------
Score the 16-model simplex once per seed: ten independent dCor* and Procrustes
disparity values per surrogate, against one fixed ground truth (the mixture
simplex, which does not depend on the initialisation).  This is the analysis that
puts an error bar on numbers the project already reports, and it is the one with
no caveats -- within a single seed the sixteen adapters carry sixteen distinct
recipes, so the adapter-to-recipe mapping is one-to-one and every level joins
normally.

An eleventh row per surrogate carries ``init_seed = -1``, the **before-embedding
mean**: the ten per-seed distance matrices averaged elementwise and scored like
any other 16-model collection.  See the two means below.

Analysis B -- the full 160-model pool  (``initsweep_separation.csv``)
--------------------------------------------------------------------
One collection, all ten seeds, asking about separation: does a surrogate place a
model nearer its **seed-siblings** (same mixture, different initialisation) than
its mixture-neighbours?  Reported as three mean distances -- different mixture at
one initialisation, same mixture across two, different mixture across two -- and
**not** as a Procrustes score against
the truth, because on this pool the truth is degenerate: ``ground_truth_weights``
derives barycentric coordinates from the *recipe*, so all ten siblings get
byte-identical weights and their truth distance is exactly 0.  That is 720 of the
12,720 pairs, 5.7%, pinned at zero, and a Procrustes fit against it is dominated
by an unsatisfiable constraint.  dCor* would survive, since it never embeds, but
its truth matrix has a sixteen-block zero structure that would have to be
reported alongside the number, so this analysis reports neither.

A dataset row could not enter this pool even if asked for.  Its identity is the
recipe, which is ten-to-one across the seeds, and ``relabel()`` refuses to let
distinct models collide rather than silently averaging them; the level has
nothing to say about a pool whose models it cannot tell apart.  It is skipped
with a message rather than crashing the run.

The plain within/between ratio is kept in the CSV but is **not** the statistic to
read: the between set mixes pairs that share an initialisation with pairs that do
not, in a proportion fixed by how many seeds are in the pool, so it moves with
pool size while every distance in it stays put.  See :func:`separation_rows` for
the measurement that established this and for the three cells that replace it.

The statistic is written **per level and also per mixture**, because the two
answer different questions and the pool is read once either way: a seed cloud
might be tighter at a vertex, where the training signal is purest, than at the
centre.  Per-mixture rows carry the mixture in ``mixture`` and the per-level
summary carries ``mixture = ALL``.

Analysis C -- geometry over the full collection  (``initsweep_geometry.csv``)
----------------------------------------------------------------------------
Coordinates for the pictures A and B summarise -- whether the seed cloud is small
against the simplex, or large enough to swallow the 25% grid spacing.  Seven
kinds of row, all 2-D MDS.  Everything a figure needs is in this file, including
every alignment, so ``make_initsweep_figures.py`` reads coordinates and draws
them and does no analysis of its own.

Every kind is written in the **house orientation**: the pure-g1 vertex due north
of the configuration's centre and the pure-g2 vertex in positive x.  MDS leaves
rotation and reflection free, so without this two panels of the same simplex can
be mirror images and a reader comparing levels side by side has to re-derive
which way is which in each one.  Kinds that share a frame are turned together, by
one map computed from the sixteen mixture points in that frame, so the
superpositions above are not disturbed: ``pool160`` with ``mean_after``,
``mean_after_aligned`` with ``mean_before_aligned``, and ``seed_aligned`` with
``seed_mean`` and ``seed_reference``.  Scale is left alone.

``pool160``
    One MDS fit over all 160 models, per surrogate.  Every model is a point; the
    figure colours by mixture.

``mean_after``
    The **after-embedding mean**: each mixture's centroid over its ten
    ``pool160`` points.  In the pool's frame by construction, so it draws
    straight on top of the cloud it summarises.

``mean_before``
    The **before-embedding mean**: average the ten 16x16 per-seed distance
    matrices elementwise, then embed the result.  This is a genuine 16-model
    "average model" collection rather than a picture of one, which is why it is
    also *scored* against truth in analysis A and the after-embedding mean is
    not.

    The two disagree because MDS is not linear, and the design note left the
    choice open; both are carried, since the expensive part was the adapters and
    not either mean.  Row labels on the averaged matrix are seed ``i00``'s model
    ids, standing for their mixtures -- the mixture is parsed out of the id by
    everything downstream, and the ten siblings agree on it exactly.

``mean_after_aligned``
    ``mean_after`` as the superposition below leaves it: centred and scaled to
    unit norm.  This, not the raw ``mean_after``, is the half of the pair that
    goes on one axes with ``mean_before_aligned``; the raw one belongs with
    ``pool160``, whose frame it shares.

``mean_before_aligned``
    The same sixteen points, Procrustes-superimposed on ``mean_after`` so the two
    means can be drawn on one axes.  They are separate MDS fits of different
    objects and so start in different frames; without this, any gap between them
    would be mostly rotation.  The residual disparity of that superposition is
    printed by the run and is the number that says how much the choice between
    the two means actually matters for this surrogate.

``seed_aligned``
    The ten per-seed 16-point embeddings, overlaid.  **They are Procrustes-
    aligned to a common reference first**, because MDS fixes coordinates only up
    to rotation, reflection and scale: ten raw embeddings on one axes would show
    a spread that is mostly arbitrary orientation.  The reference is the
    ``mean_before`` embedding -- a consensus of the ten rather than one of them,
    so no seed is privileged by being the frame.  Models are matched across seeds
    by mixture, not by position.

    This alignment binds only the *picture*.  The scores in analysis A are
    orientation-invariant already and are unaffected by it.

``seed_mean``
    **The mean of the overlay**: each mixture's ten *aligned* positions averaged,
    sixteen points.  This is the centre of the seed cloud as the overlay draws
    it, and so the thing the overlay should be measured against.  It is a third
    mean over seeds and agrees with neither of the other two by construction --
    ``mean_after`` is a centroid inside a joint fit of all 160 adapters, and
    ``mean_before`` is an embedding of averaged distances.  This one averages ten
    separate fits after reconciling their frames.

``seed_reference``
    That consensus, in the frame the seeds were aligned into: sixteen points,
    one per mixture.  This is what ``seed_aligned`` should be drawn against, and
    ``mean_before`` is not, because the alignment scales both configurations to
    unit Frobenius norm and the raw mean is on the surrogate's own scale --
    order 1 for structural, order 1e-3 for functional.  Drawn against the raw
    mean, an overlay shows that ratio and nothing about the seeds.

A note on the behavioral rows
-----------------------------
The sampled behavioral level decodes at R=16 with ``do_sample=True``, so its
spread mixes initialisation with decode noise.  The confound is bounded but not
zero: ``BehavioralTaxonomy._seed_for_batch`` hashes the generation seed with the
batch start and depends on neither the adapter nor its initialisation, so all 160
models decode under the same RNG stream per batch -- the "different draw per
model" confound is absent, and what remains is that one stream applied to
different logits still yields different tokens.  The greedy control is
deterministic and is the clean read on this axis.  A large gap between the two
spreads is decode noise, not initialisation.

Usage::

    python figures/fig_structural_sweep/sweep_initsweep.py
    python figures/fig_structural_sweep/sweep_initsweep.py --perspective structural_all_o
    python figures/fig_structural_sweep/sweep_initsweep.py --skip-pool   # analysis A alone

    # the determinism null control the default set leaves out
    python figures/fig_structural_sweep/sweep_initsweep.py \
        --perspective dataset_embedding --skip-pool --out dataset_null.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.bridge import fit_geometry  # noqa: E402
from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.configurations import procrustes_compare  # noqa: E402
from src.analysis.discovery import CacheIndex, scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.core.distance import DistanceMatrix  # noqa: E402
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import (  # noqa: E402
    mixture_weights, raw_mixture_pcts, sort_by_mixture,
)

# Imported, never copied: these are the project's standing per-level defaults,
# not a property of any one experiment, and two divergent copies would be a
# silent way for four figures to stop being comparable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex_collection_size"))
from sweep_group_size import _META_KEYS, canonical_perspectives  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
CACHE_ROOT = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/shared_cache")

#: The surrogates this sweep scores by default: one canonical read per model
#: level, plus the greedy decoding control.  No dataset row -- see the module
#: docstring for what that costs and how to take the reading anyway.  Every
#: canonical perspective remains reachable through ``--perspective``.
DEFAULT_PERSPECTIVES = (
    "structural_all_o",
    "functional_all",
    "behavioral",
    "behavioral_greedy",
)

#: The training recipe every adapter in this sweep shares.  Pinning all three is
#: what separates the initsweep from the 1004-adapter group-size pool, the
#: nsweep's other draw sizes and the rsweep's other ranks -- all of which are
#: also yahoo, also this base model, and also in this one content-addressed
#: cache.  The rank pin is the one that is easy to forget: ``i00`` exists at
#: eight ranks, and without it seed 0 would hold 128 adapters and the other nine
#: would hold 16.
TRAIN_N, TRAIN_SEED, TRAIN_RANK = 1000, 0, 16

#: ``init_seed`` of the before-embedding-mean row in ``initsweep_scores.csv``.
#: Not a seed; a sentinel, so the column stays integer-typed.
MEAN_BEFORE = -1

#: One row per (perspective, init seed), scored against the requested simplex.
SCORE_FIELDS = [
    "perspective", "taxonomy", "metric", "init_seed", "n_models",
    "dcor", "disparity", "max_realized_deviation", "seconds",
]

#: One row per (perspective, mixture), plus a ``mixture = ALL`` summary.
SEPARATION_FIELDS = [
    "perspective", "taxonomy", "metric", "mixture", "n_models", "n_within_pairs",
    "n_between_pairs", "within_mean", "within_sd", "between_mean", "between_sd",
    "ratio", "n_between_same_seed_pairs", "between_same_seed_mean",
    "between_same_seed_sd", "n_between_diff_seed_pairs",
    "between_diff_seed_mean", "between_diff_seed_sd", "ratio_same_seed",
    "seconds",
]

#: One row per (perspective, kind, model), 2-D MDS coordinates.
GEOMETRY_FIELDS = [
    "perspective", "kind", "init_seed", "model_id", "mixture", "dim1", "dim2",
]

#: ``mixture = ALL`` in the separation CSV: the per-level summary row.
ALL_MIXTURES = "ALL"


def mixture_key(model_id: str) -> str:
    """The mixture an adapter was trained on, as the id spells it.

    The join key across seeds.  Ten seed-siblings differ in their ``_iNN_``
    field and in the content hash that follows it, and agree on everything
    before ``_r{rank}``; this returns that agreed head, e.g.
    ``yahoo_075g1_025g2_000g3_n1000_s00``.

    A model id in this project is the adapter's **full cache path**, so the
    directory is dropped first -- otherwise the key carries the cache root and
    every CSV cell is 90 characters of shared prefix.
    """
    head, sep, _ = Path(model_id).name.partition(f"_r{TRAIN_RANK}_i")
    if not sep:
        raise ValueError(f"{model_id!r} does not name rank {TRAIN_RANK} and an "
                         "init seed, so it cannot be joined across seeds")
    return head


def init_seed_key(model_id: str) -> str:
    """The initialisation an adapter was trained from, as the id spells it.

    The second join key.  ``_r{rank}_i`` is followed by the zero-padded seed and
    then the content hash, so the seed is the digits that open the tail, e.g.
    ``07``.  Needed because a *between* pair is not one population: two adapters
    of different mixtures may or may not share an initialisation, and those two
    cases sit at very different distances.
    """
    _, sep, tail = Path(model_id).name.partition(f"_r{TRAIN_RANK}_i")
    if not sep:
        raise ValueError(f"{model_id!r} does not name rank {TRAIN_RANK} and an "
                         "init seed, so its initialisation cannot be read")
    digits = tail[:len(tail) - len(tail.lstrip("0123456789"))]
    if not digits:
        raise ValueError(f"{model_id!r} names no init seed after "
                         f"'_r{TRAIN_RANK}_i'")
    return digits


def mixture_of(label: str) -> str:
    """:func:`mixture_key`, but a label that is *already* a key passes through.

    Both derived geometries -- the two means -- are labelled by mixture rather
    than by adapter, because neither is one adapter: a point of the
    after-embedding mean is ten models' centroid, and one of the
    before-embedding mean is a row of a matrix ten models contributed to.
    Demanding an ``_iNN_`` field of those labels would reject exactly the rows
    that have no seed to name.
    """
    try:
        return mixture_key(label)
    except ValueError:
        return Path(label).name


def slice_matrix(index, ids, cache_root, spec, *, label, use_cache=True):
    """``(distance matrix, 2-D MDS)`` for one perspective over one collection."""
    kwargs = {k: v for k, v in spec.items() if k not in _META_KEYS}
    dm, geos = build_taxonomy_artifacts(
        index, spec["taxonomy"], spec["metric"], cache_root=cache_root,
        n_components=(2,), use_cache=use_cache, id_scheme="model_id",
        label=label, mds_kwargs={"random_state": suite.MDS_SEED}, **kwargs,
    )
    return dm.reindex(list(ids)), geos["mds_2d"].reindex(list(ids))


def score_slice(dm, ids, vertex_names):
    """``(dcor, disparity)`` against the requested simplex, for one collection."""
    weights = suite.truth_weights(ids)
    tdm = simplex_distance_matrix(weights, list(ids), vertex_names)
    tgeo = simplex_geometry(weights, list(ids), vertex_names)
    return (
        dcor_vs_truth(dm, tdm),
        disparity_vs_truth(dm, tgeo, random_state=suite.MDS_SEED,
                           n_components=tgeo.coordinates.shape[1]),
    )


def mean_distance_matrix(per_seed, order):
    """The before-embedding mean: the ten 16x16 matrices averaged elementwise.

    *per_seed* maps init seed to that seed's :class:`DistanceMatrix`; *order* is
    the mixture-sorted id list of one reference seed, whose ids label the result.
    Every matrix is reindexed onto the mixture order first, so the average is of
    the same cell across seeds rather than of whatever row order each came in.
    """
    ref_seed = min(per_seed)
    keys = [mixture_key(m) for m in order]
    stack = []
    for seed in sorted(per_seed):
        dm = per_seed[seed]
        by_key = {mixture_key(m): m for m in dm.model_ids}
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(
                f"seed {seed} is missing mixture(s) {missing}; the ten "
                "collections must cover the same simplex for a mean over them "
                "to mean anything.")
        stack.append(np.asarray(dm.reindex([by_key[k] for k in keys]).matrix,
                                dtype=np.float64))
    mean = np.mean(stack, axis=0)
    return DistanceMatrix(matrix=mean, model_ids=list(order),
                          metric=per_seed[ref_seed].metric,
                          taxonomy=per_seed[ref_seed].taxonomy)


def separation_rows(dm, spec, name, seconds):
    """Within-mixture vs between-mixture distances, per mixture and overall.

    A *within* pair is two adapters of one mixture trained from different
    initialisations -- seed-siblings.  A *between* pair is two adapters of
    different mixtures.  The ratio is ``within_mean / between_mean``: near 0 the
    siblings sit on top of each other and the surrogate sees only the mixture,
    near 1 the initialisation moves a model as far as changing what it was
    trained on.

    ``between`` alone is **not a statistic of the surrogate**: it is a statistic
    of the surrogate *and* of how many seeds happen to be in the pool.  A
    between pair may share an initialisation or not, and those two cases sit far
    apart; the share that shares one is ``1/n_seeds``, so ``between_mean`` drifts
    with pool size even though every pair in it is unchanged.  Measured here on
    ``structural_all_o``, ``between_mean`` climbed 0.6021 -> 0.6603 -> 0.7079 ->
    0.7406 across pools of 2, 3, 5 and 10 seeds while ``within_mean`` held at
    0.617, and a two-cell fit to the first two rows predicts the last two to
    within 0.0012.  So the three cells below are reported separately and it is
    those, not ``ratio``, that carry the finding:

    ==============================  ============================================
    ``between_same_seed_mean``      different mixture, same initialisation
    ``within_mean``                 same mixture, different initialisation
    ``between_diff_seed_mean``      different mixture, different initialisation
    ==============================  ============================================

    ``ratio_same_seed`` is ``within_mean / between_same_seed_mean``, the
    comparison the pool size does not move: above 1 the initialisation moves an
    adapter further than changing what it was trained on.
    """
    ids = list(dm.model_ids)
    M = np.asarray(dm.matrix, dtype=np.float64)
    keys = np.array([mixture_key(m) for m in ids])
    seeds = np.array([init_seed_key(m) for m in ids])
    iu = np.triu_indices(len(ids), k=1)
    same = keys[iu[0]] == keys[iu[1]]
    same_seed = seeds[iu[0]] == seeds[iu[1]]
    d = M[iu]

    rows = []
    for mixture in [ALL_MIXTURES, *sorted(set(keys))]:
        if mixture == ALL_MIXTURES:
            touches = np.ones(d.shape, dtype=bool)
            within, between = d[same], d[~same]
            n_models = len(ids)
        else:
            # This mixture's own siblings, against every pair that joins one of
            # them to a model of another mixture. The between set is restricted
            # to pairs touching this mixture, so the row describes this cloud's
            # neighbourhood rather than the pool's.
            touches = (keys[iu[0]] == mixture) | (keys[iu[1]] == mixture)
            within, between = d[touches & same], d[touches & ~same]
            n_models = int((keys == mixture).sum())
        b_same = d[touches & ~same & same_seed]
        b_diff = d[touches & ~same & ~same_seed]
        wm, bm = float(np.mean(within)), float(np.mean(between))
        bsm = float(np.mean(b_same)) if b_same.size else float("nan")
        rows.append({
            "perspective": name, "taxonomy": spec["taxonomy"],
            "metric": spec["metric"], "mixture": mixture, "n_models": n_models,
            "n_within_pairs": int(within.size), "n_between_pairs": int(between.size),
            "within_mean": wm, "within_sd": float(np.std(within)),
            "between_mean": bm, "between_sd": float(np.std(between)),
            "ratio": wm / bm if bm else float("nan"),
            "n_between_same_seed_pairs": int(b_same.size),
            "between_same_seed_mean": bsm,
            "between_same_seed_sd": float(np.std(b_same)) if b_same.size else float("nan"),
            "n_between_diff_seed_pairs": int(b_diff.size),
            "between_diff_seed_mean": float(np.mean(b_diff)) if b_diff.size else float("nan"),
            "between_diff_seed_sd": float(np.std(b_diff)) if b_diff.size else float("nan"),
            "ratio_same_seed": wm / bsm if bsm else float("nan"),
            "seconds": round(seconds, 2),
        })
    return rows


def centroid_geometry(geo):
    """The **after-embedding mean**: each mixture's centroid over its ten points.

    A plain average of the ``pool160`` coordinates, labelled by mixture, so it
    lives in the pool's frame by construction and needs no alignment to be drawn
    on top of the cloud it summarises.  Contrast ``mean_distance_matrix``, which
    averages the *distances* and embeds afterwards; the two disagree because MDS
    is not linear, which is exactly why both are carried.
    """
    from src.core.geometry import GeometryResult

    coords = np.asarray(geo.coordinates, dtype=np.float64)
    by_key = defaultdict(list)
    for i, m in enumerate(geo.model_ids):
        by_key[mixture_key(m)].append(coords[i])
    keys = sorted(by_key)
    means = np.vstack([np.mean(by_key[k], axis=0) for k in keys])
    return GeometryResult(
        coordinates=means.astype(np.float32), model_ids=keys, method="mds",
        taxonomy=geo.taxonomy, n_components=means.shape[1], stress=0.0,
        metadata={"derived": "after-embedding mean over init seeds"},
    )


def _vertex_label(labels, group):
    """The label of the adapter trained on *group* alone, e.g. 100/0/0.

    Found by weight rather than by spelling, so the ``NNNgK`` field widths and
    the ``_nNNNN_sNN`` tail are not this function's problem.
    """
    for label in labels:
        weights = mixture_weights(label)
        if group < len(weights) and weights[group] > 0.99:
            return label
    raise ValueError(f"no pure group-{group + 1} vertex among {len(labels)} "
                     "labels, so the canonical orientation is undefined")


def canonical_frame(geo):
    """The centring and the 2x2 map that put a configuration in the house frame.

    MDS fixes coordinates only up to rotation, reflection and scale, so two
    panels of the same simplex can be mirror images of each other and nothing is
    wrong.  That is fine for a single picture and bad for a row of them: a reader
    comparing three levels side by side has to re-derive which way is which in
    every panel.  This pins the remaining freedom by the *content* rather than by
    the fit -- the pure-g1 vertex is rotated onto the positive y axis, and the
    pure-g2 vertex is reflected into positive x if it is not there already.
    Every panel then shows the simplex the same way up.

    Returns ``(centroid, M)``; apply with :func:`orient`.  Scale is deliberately
    untouched, so this composes with a Procrustes superposition without undoing
    its fitted scale.
    """
    coords = np.asarray(geo.coordinates, dtype=np.float64)[:, :2]
    labels = list(geo.model_ids)
    centroid = coords.mean(axis=0)
    at = {lab: coords[i] - centroid for i, lab in enumerate(labels)}

    x, y = at[_vertex_label(labels, 0)]
    # Rotate by the angle that carries this vertex from where it is to due north.
    alpha = np.pi / 2 - np.arctan2(y, x)
    c, s = np.cos(alpha), np.sin(alpha)
    M = np.array([[c, -s], [s, c]])
    if (M @ at[_vertex_label(labels, 1)])[0] < 0:
        # Mirror in the y axis. The g1 vertex is on that axis by construction and
        # so is fixed; only the handedness of the panel changes.
        M = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ M
    return centroid, M


def orient(geo, frame):
    """Apply a :func:`canonical_frame` to one geometry, keeping its labels."""
    from src.core.geometry import GeometryResult

    centroid, M = frame
    coords = np.asarray(geo.coordinates, dtype=np.float64)[:, :2]
    moved = (coords - centroid) @ M.T
    return GeometryResult(
        coordinates=moved.astype(np.float32), model_ids=list(geo.model_ids),
        method=geo.method, taxonomy=geo.taxonomy, n_components=2,
        stress=geo.stress, metadata={**(geo.metadata or {}),
                                     "oriented": "g1 vertex north, g2 vertex east"},
    )


def mean_of(geos):
    """Per-mixture centroid over a list of geometries that share a frame.

    The mean of the overlay: each mixture's ten *aligned* positions averaged.
    Distinct from both means over seeds in analysis C -- it is neither a centroid
    of a joint fit (``mean_after``, which embeds all 160 at once) nor an
    embedding of averaged distances (``mean_before``).  It is the centre of the
    seed cloud as the overlay actually draws it, which is what the overlay should
    be measured against.
    """
    from src.core.geometry import GeometryResult

    by_key = defaultdict(list)
    ref = geos[0]
    for geo in geos:
        coords = np.asarray(geo.coordinates, dtype=np.float64)[:, :2]
        for i, label in enumerate(geo.model_ids):
            by_key[mixture_of(label)].append(coords[i])
    keys = sorted(by_key)
    means = np.vstack([np.mean(by_key[k], axis=0) for k in keys])
    return GeometryResult(
        coordinates=means.astype(np.float32), model_ids=keys, method=ref.method,
        taxonomy=ref.taxonomy, n_components=2, stress=0.0,
        metadata={"derived": "mean over cross-seed-aligned embeddings"},
    )


def geometry_rows(geo, name, kind, init_seed, ids=None):
    """Coordinate rows for one embedding.

    *ids* replaces the geometry's own labels, in its row order.  The aligned
    output of ``procrustes_compare`` is labelled by the *match key* rather than
    by the model, which is what makes an overlay possible at all -- no two seeds
    share an id -- but would lose which adapter each point is.  Passing that
    seed's ids back in restores it, and ``mixture`` then holds the key.
    """
    coords = np.asarray(geo.coordinates, dtype=np.float64)
    labels = list(ids) if ids is not None else list(geo.model_ids)
    if len(labels) != len(geo.model_ids):
        raise ValueError(f"{len(labels)} ids for {len(geo.model_ids)} points")
    return [{
        "perspective": name, "kind": kind, "init_seed": init_seed,
        "model_id": m, "mixture": mixture_of(m),
        "dim1": float(coords[i, 0]), "dim2": float(coords[i, 1]),
    } for i, m in enumerate(labels)]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--perspective", action="append", dest="perspectives",
                    help="repeatable; default is the four model-level reads in "
                         "DEFAULT_PERSPECTIVES. Every canonical perspective is "
                         "reachable, dataset_embedding included")
    ap.add_argument("--spec", default="yahoo",
                    help="DataSimplexSpec key naming the mixture grid")
    ap.add_argument("--seed", action="append", type=int, dest="seeds",
                    help="repeatable; default is every init seed in the cache")
    ap.add_argument("--n-expected", type=int, default=16,
                    help="models per seed; a collection of any other size is fatal")
    ap.add_argument("--skip-pool", action="store_true",
                    help="analysis A only: skip the 160-model pool and its geometry")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="initsweep_scores.csv")
    ap.add_argument("--out-separation", default="initsweep_separation.csv")
    ap.add_argument("--out-geometry", default="initsweep_geometry.csv")
    args = ap.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    if not cache_root.exists():
        raise SystemExit(f"no cache at {cache_root}")

    # Bind the architecture before reading the perspectives: two of them name
    # "the last layer", which is a position in a stack rather than a number.
    suite.apply_architecture(suite.architecture(args.base_model))
    print(f"model: {args.base_model}  ({suite.N_LAYERS} layers, "
          f"{suite.N_STATES} states)", flush=True)

    spec = SPECS[args.spec]
    wanted = {tuple(m) for m in spec.mixture_pcts()}

    index = scan_cache(cache_root, base_model_id=args.base_model,
                       behavioral_draw=suite.DRAW, functional_draw=suite.DRAW,
                       datasets=["yahoo"])
    print(f"scanned {len(index)} yahoo adapters", flush=True)

    index = CacheIndex([e for e in index.entries
                        if raw_mixture_pcts(e.model_id) in wanted],
                       index.cache_root)
    index = index.filter(n_samples=TRAIN_N, seed=TRAIN_SEED, lora_rank=TRAIN_RANK)
    print(f"pinned to the {len(wanted)}-point grid at n={TRAIN_N}, "
          f"seed={TRAIN_SEED}, r={TRAIN_RANK}: {len(index)} adapters", flush=True)

    collections = index.slices(by=("lora_init_seed",))
    if args.seeds:
        missing = sorted(set(args.seeds) - {k[0] for k in collections})
        if missing:
            raise SystemExit(f"no adapters at init seed(s) {missing}")
        collections = {k: v for k, v in collections.items() if k[0] in args.seeds}
    seeds = [k[0] for k in sorted(collections)]
    print(f"{len(collections)} collections over init seeds {seeds}", flush=True)

    bad = {k[0]: len(v.model_ids) for k, v in collections.items()
           if len(v.model_ids) != args.n_expected}
    if bad:
        raise SystemExit(
            f"{len(bad)} seed(s) do not hold {args.n_expected} models: {bad}. "
            "The ground truth is only unambiguous at the full simplex, so this "
            "is fatal rather than skippable.")

    specs = canonical_perspectives()
    unknown = set(args.perspectives or ()) - set(specs)
    if unknown:
        raise SystemExit(f"unknown perspective(s) {sorted(unknown)}. "
                         f"Choose from {sorted(specs)}")
    names = args.perspectives or list(DEFAULT_PERSPECTIVES)

    pool = CacheIndex([e for c in collections.values() for e in c.entries],
                      index.cache_root)
    pool_ids = sorted(pool.model_ids, key=lambda m: (mixture_key(m), m))
    if not args.skip_pool:
        print(f"pool for analyses B and C: {len(pool_ids)} models", flush=True)

    score_rows, sep_rows, geo_rows = [], [], []
    t_start = time.time()
    for name in names:
        pspec = specs[name]

        # ── analysis A: one 16-model collection per seed ──────────────────
        per_seed_dm, per_seed_geo, order = {}, {}, None
        for seed in seeds:
            sub = collections[(seed,)]
            ids = sort_by_mixture(list(sub.model_ids))
            if order is None:
                order = ids
            vertex_names = [f"g{j + 1}"
                            for j in range(suite.truth_weights(ids).shape[1])]

            t0 = time.time()
            dm, geo = slice_matrix(sub, ids, cache_root, pspec,
                                   label="initsweep seed",
                                   use_cache=not args.no_cache)
            dcor, disparity = score_slice(dm, ids, vertex_names)
            dt = time.time() - t0
            per_seed_dm[seed], per_seed_geo[seed] = dm, geo
            score_rows.append({
                "perspective": name, "taxonomy": pspec["taxonomy"],
                "metric": pspec["metric"], "init_seed": seed,
                "n_models": len(ids), "dcor": dcor, "disparity": disparity,
                "max_realized_deviation": suite.check_realized_truth(ids, TRAIN_N),
                "seconds": round(dt, 2),
            })
            print(f"  {name:<20} i={seed:<3} dcor={dcor:.4f} "
                  f"disp={disparity:.4f} {dt:6.1f}s", flush=True)

        # ── the before-embedding mean: averaged first, then embedded ──────
        mean_dm = mean_distance_matrix(per_seed_dm, order)
        vertex_names = [f"g{j + 1}"
                        for j in range(suite.truth_weights(order).shape[1])]
        dcor, disparity = score_slice(mean_dm, order, vertex_names)
        score_rows.append({
            "perspective": name, "taxonomy": pspec["taxonomy"],
            "metric": pspec["metric"], "init_seed": MEAN_BEFORE,
            "n_models": len(order), "dcor": dcor, "disparity": disparity,
            "max_realized_deviation": suite.check_realized_truth(order, TRAIN_N),
            "seconds": 0.0,
        })
        print(f"  {name:<20} mean  dcor={dcor:.4f} disp={disparity:.4f}"
              "   (before-embedding mean)", flush=True)

        mean_geo = fit_geometry(mean_dm, method="mds", n_components=2,
                                random_state=suite.MDS_SEED)
        geo_rows += geometry_rows(orient(mean_geo, canonical_frame(mean_geo)),
                                  name, "mean_before", MEAN_BEFORE)

        # ── the overlay: ten embeddings in one frame ──────────────────────
        # Each seed's sixteen adapters are embedded against *each other* and
        # nobody else -- ten 16x16 matrices, ten MDS fits -- so the ten
        # configurations start in ten unrelated frames. MDS fixes coordinates
        # only up to rotation, reflection and scale, so they are superimposed
        # before they are plotted. Matched by mixture, since no two seeds share
        # an adapter id.
        aligned_seed = {}
        reference = None
        for seed in seeds:
            fit = procrustes_compare(mean_geo, per_seed_geo[seed],
                                     key=mixture_of)
            aligned_seed[seed] = fit.aligned_b
            if reference is None:
                # The reference the seeds were aligned *to*, in the frame they
                # were aligned into. procrustes_compare centres and scales both
                # configurations to unit Frobenius norm, so the raw
                # ``mean_before`` is not on this scale -- and is off by a factor
                # of the surrogate's distances, which run from ~1 for structural
                # to ~1e-3 for functional. Kept once: every seed is aligned to
                # the same reference, so this does not depend on which one
                # produced it.
                reference = fit.aligned_a

        # The mean of the overlay, computed here and not at plot time: the
        # centroid of each mixture's ten aligned positions. The whole overlay
        # frame -- seeds, reference and mean together -- is then turned the same
        # way up as every other panel.
        seed_mean = mean_of([aligned_seed[s] for s in seeds])
        frame = canonical_frame(seed_mean)
        geo_rows += geometry_rows(orient(seed_mean, frame), name,
                                  "seed_mean", MEAN_BEFORE)
        geo_rows += geometry_rows(orient(reference, frame), name,
                                  "seed_reference", MEAN_BEFORE)
        for seed in seeds:
            by_key = {mixture_key(m): m for m in per_seed_geo[seed].model_ids}
            moved = orient(aligned_seed[seed], frame)
            geo_rows += geometry_rows(
                moved, name, "seed_aligned", seed,
                ids=[by_key[k] for k in moved.model_ids])

        # ── analyses B and C: the 160-model pool ──────────────────────────
        if args.skip_pool:
            continue
        if pspec["taxonomy"] == "dataset_embedding":
            # The dataset level's identity is the recipe, which is ten-to-one
            # here. Not a gap in the measurement -- the level has nothing to say
            # about a pool whose models it cannot tell apart.
            print(f"  {name:<20} pool  skipped — the dataset level is "
                  "recipe-identified and its ten siblings collide", flush=True)
            continue

        t0 = time.time()
        pool_dm, pool_geo = slice_matrix(pool, pool_ids, cache_root, pspec,
                                         label="initsweep pool",
                                         use_cache=not args.no_cache)
        dt = time.time() - t0
        rows = separation_rows(pool_dm, pspec, name, dt)
        sep_rows += rows

        # Both means in one frame, so the figure can draw them together without
        # doing any analysis of its own. Note that this is the *superposition's*
        # frame and not the pool's: the fit centres and scales both
        # configurations to unit norm, so ``mean_after`` -- which is in the
        # pool's own units, and is what the pool figure draws its cloud against
        # -- is not on this scale. Both sides of the superposition are therefore
        # written, and a figure comparing the two means uses the aligned pair.
        after_geo = centroid_geometry(pool_geo)
        # The pool and its per-mixture centroids are one frame and are turned as
        # one, off the centroids: the 160 adapters carry ten labels per mixture
        # and no single vertex to orient by.
        pool_frame = canonical_frame(after_geo)
        geo_rows += geometry_rows(orient(pool_geo, pool_frame), name,
                                  "pool160", MEAN_BEFORE)
        geo_rows += geometry_rows(orient(after_geo, pool_frame), name,
                                  "mean_after", MEAN_BEFORE)
        in_frame = procrustes_compare(after_geo, mean_geo, key=mixture_of)
        means_frame = canonical_frame(in_frame.aligned_a)
        geo_rows += geometry_rows(orient(in_frame.aligned_a, means_frame), name,
                                  "mean_after_aligned", MEAN_BEFORE)
        geo_rows += geometry_rows(orient(in_frame.aligned_b, means_frame), name,
                                  "mean_before_aligned", MEAN_BEFORE)
        print(f"  {name:<20} means disparity between the two means = "
              f"{in_frame.disparity:.4f}", flush=True)
        summary = rows[0]
        print(f"  {name:<20} pool  same-seed/diff-mix="
              f"{summary['between_same_seed_mean']:.4f} "
              f"within={summary['within_mean']:.4f} "
              f"diff-seed/diff-mix={summary['between_diff_seed_mean']:.4f} "
              f"ratio*={summary['ratio_same_seed']:.4f} {dt:6.1f}s", flush=True)

    outdir = Path(args.outdir)
    write_csv(outdir / args.out, SCORE_FIELDS, score_rows)
    if sep_rows:
        write_csv(outdir / args.out_separation, SEPARATION_FIELDS, sep_rows)
    write_csv(outdir / args.out_geometry, GEOMETRY_FIELDS, geo_rows)
    print(f"({time.time() - t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
