#!/usr/bin/env python
"""Score every nsweep slice against the simplex it was trained on.

**nsweep** -- a sweep over ``n_samples``, the size of the training draw, holding
everything else fixed.  See ``docs/terminology.md``.

**Rung** -- one value of ``n_samples`` in this sweep, i.e. one of the nine
training draw sizes.  Note this is *not* the ``rung`` that ``docs/terminology.md``
retired in favour of ``surrogate``; that one meant a step on the ladder of
representations a level can be read at.  The two senses share a word and nothing
else, and the collision is unresolved -- see the note.

The question: how do dCor* and Procrustes agreement, between each taxonomy level
and the ground-truth mixture simplex, respond to the amount of data an adapter
was fine-tuned on?

The shape is nothing like the group-size sweep next door, and the difference is
worth stating because the two share most of their machinery.  There, a 1004-model
pool was *subsampled* into groups of size n, so a group was a random draw and the
interesting axis was how many models you had.  Here nothing is sampled: the cache
already holds 90 complete 16-model collections, one per ``(n_samples, seed)``
cell, and each is the same 25% simplex trained on a different amount of data.
So there are no replicates to average, no shuffle seed, and no variant A/B
distinction -- variant A adds the pure vertices and the even mixture to the
matrix, and here they are already *in* every collection by construction.

**Both ground truths are scored.**  Mixtures are allocated by largest remainder
(``src/datasets/mixed_dataset.py:_allocate_counts``), so a requested 25/75
realizes as 3/7 at N=10: at a small draw the adapter is trained on a mixture that
is not the one its name asks for, and because the allocation is deterministic the
error is identical across all ten seeds.  Scoring only against the requested
simplex would charge that discretisation to the taxonomy level and read as "small
N degrades recovery", indistinguishable from the effect being measured.  The gap
between the two curves is the decomposition of the N-effect into label
misspecification and representation noise.

Usage::

    python figures/simplex3_nsweep_olmo2_nsweep/sweep_nsweep.py
    python figures/simplex3_nsweep_olmo2_nsweep/sweep_nsweep.py --perspective behavioral
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.discovery import CacheIndex, scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import raw_mixture_pcts, sort_by_mixture  # noqa: E402

# The canonical perspectives are defined once, next to the group-size sweep, and
# imported rather than copied: they are the standing per-level defaults for this
# project, not a property of either experiment, and two divergent copies would be
# a silent way for the two figures to stop being comparable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex_collection_size"))
from sweep_group_size import _META_KEYS, canonical_perspectives  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
CACHE_ROOT = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/shared_cache")

#: One row per (perspective, n_samples, seed), scored under both truths.
FIELDS = [
    "perspective", "taxonomy", "metric", "n_samples", "seed", "n_models",
    "dcor_requested", "dcor_realized",
    "disparity_requested", "disparity_realized",
    "max_realized_deviation",
]


def slice_matrix(index, ids, cache_root, name, spec, *, use_cache=True):
    """The one distance matrix for *name* over the 16 models of one slice."""
    kwargs = {k: v for k, v in spec.items() if k not in _META_KEYS}
    dm, _ = build_taxonomy_artifacts(
        index, spec["taxonomy"], spec["metric"], cache_root=cache_root,
        n_components=(2,), use_cache=use_cache, id_scheme="model_id",
        label="nsweep slice", **kwargs,
    )
    return dm.reindex(list(ids))


def score_slice(dm, ids, n_samples, vertex_names):
    """dCor* and Procrustes disparity against both truths, for one slice.

    Returns ``(dcor_req, dcor_real, disp_req, disp_real, worst)`` where *worst*
    is the largest weight deviation the realized allocation introduced -- the
    number ``suite.check_realized_truth`` bounds by ``1 / n_samples``.
    """
    out = []
    for weights in (suite.truth_weights(ids),
                    suite.realized_truth_weights(ids, n_samples)):
        tdm = simplex_distance_matrix(weights, list(ids), vertex_names)
        tgeo = simplex_geometry(weights, list(ids), vertex_names)
        out.append((
            dcor_vs_truth(dm, tdm),
            disparity_vs_truth(dm, tgeo, random_state=suite.MDS_SEED,
                               n_components=tgeo.coordinates.shape[1]),
        ))
    (dcor_req, disp_req), (dcor_real, disp_real) = out
    worst = suite.check_realized_truth(ids, n_samples)
    return dcor_req, dcor_real, disp_req, disp_real, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--perspective", action="append", dest="perspectives",
                    help="repeatable; default is every canonical perspective")
    ap.add_argument("--spec", default="yahoo_nsweep",
                    help="DataSimplexSpec key naming the mixtures and the rungs")
    ap.add_argument("--n-expected", type=int, default=16,
                    help="models per slice; a slice of any other size is fatal")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit-slices", type=int, default=None,
                    help="score only the first N slices (a timing probe)")
    ap.add_argument("--out", default="nsweep_scores.csv")
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
    # The 1004-adapter group-size pool is also yahoo, also this base model and
    # also (n_samples=1000, seed=0), so the corpus filter alone leaves 1015
    # models. Only the mixture grid separates them.
    index = CacheIndex([e for e in index.entries
                        if raw_mixture_pcts(e.model_id) in wanted],
                       index.cache_root)

    slices = index.slices(by=("n_samples", "seed"))

    # Only the rungs that were actually bought. `extract_sizes` is the spec's
    # own record of that -- N=10000 stays in `train_sizes`, because a train
    # shard's index is its rung's position there and renumbering would orphan
    # already-submitted shards, but it was declined after the smoke shard, so
    # the cache holds 4 of its 160 adapters. Reading `train_sizes` here would
    # trip the completeness guard below on a rung nobody paid for.
    bought = set(spec.extract_sizes or spec.train_sizes)
    dropped = sorted({n for n, _ in slices} - bought)
    slices = {k: v for k, v in slices.items() if k[0] in bought}
    rungs = sorted({n for n, _ in slices})
    print(f"{len(slices)} slices over rungs {rungs}"
          + (f"; dropped unbought rung(s) {dropped}" if dropped else ""),
          flush=True)

    bad = {k: len(v.model_ids) for k, v in slices.items()
           if len(v.model_ids) != args.n_expected}
    if bad:
        raise SystemExit(
            f"{len(bad)} slice(s) do not hold {args.n_expected} models: "
            f"{dict(sorted(bad.items())[:8])}. The per-slice ground truth is "
            "only unambiguous at the full simplex, so this is fatal rather "
            "than skippable.")

    specs = canonical_perspectives()
    unknown = set(args.perspectives or ()) - set(specs)
    if unknown:
        raise SystemExit(f"unknown perspective(s) {sorted(unknown)}. "
                         f"Choose from {sorted(specs)}")
    names = args.perspectives or list(specs)

    keys = sorted(slices)
    if args.limit_slices:
        keys = keys[:args.limit_slices]

    rows = []
    t_start = time.time()
    for name in names:
        base_spec = specs[name]
        for (n_samples, seed) in keys:
            sub = slices[(n_samples, seed)]
            ids = sort_by_mixture(list(sub.model_ids))
            vertex_names = [f"g{j + 1}"
                            for j in range(suite.truth_weights(ids).shape[1])]

            # The dataset level is the one perspective whose surrogate IS the
            # training draw, so its selector moves with the slice. Every other
            # level reads the fixed 100-query test set and is slice-independent.
            s = dict(base_spec)
            if "dataset_selector" in s:
                s["dataset_selector"] = {**s["dataset_selector"],
                                         "n_samples": n_samples, "seed": seed}

            t0 = time.time()
            dm = slice_matrix(sub, ids, cache_root, name, s,
                              use_cache=not args.no_cache)
            dcor_req, dcor_real, disp_req, disp_real, worst = score_slice(
                dm, ids, n_samples, vertex_names)
            rows.append({
                "perspective": name, "taxonomy": s["taxonomy"],
                "metric": s["metric"], "n_samples": n_samples, "seed": seed,
                "n_models": len(ids),
                "dcor_requested": dcor_req, "dcor_realized": dcor_real,
                "disparity_requested": disp_req, "disparity_realized": disp_real,
                "max_realized_deviation": worst,
            })
            print(f"  {name:<20} n={n_samples:<6} s={seed:<3} "
                  f"dcor={dcor_req:.4f}/{dcor_real:.4f} "
                  f"disp={disp_req:.4f}/{disp_real:.4f} "
                  f"{time.time() - t0:6.1f}s", flush=True)

    out = Path(args.outdir) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}  "
          f"({time.time() - t_start:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
