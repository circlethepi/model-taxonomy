#!/usr/bin/env python
"""Score every rsweep collection against the simplex it was trained on.

**rsweep** -- a sweep over the LoRA rank, holding the corpus, the mixture grid,
the training draw, the test-query draw, the seeds and the optimizer fixed.  See
``docs/terminology.md``.

**surrogate** -- a read-time view of one taxonomy level (a choice of layers,
projections, pooling and metric), as opposed to the level itself.  The seven
canonical surrogates -- the standing per-level defaults for this project -- are
defined once in ``figures/simplex_collection_size/sweep_group_size.py`` and
imported here rather than copied, so this figure stays panel-for-panel
comparable with the group-size and nsweep figures next door.  Eight rows are
scored, not seven: the behavioral level carries a greedy control alongside its
R=16 read.

The question: how do Procrustes disparity and dCor*, between each taxonomy level
and the ground-truth mixture simplex, respond to the **capacity** of the adapter
that was fine-tuned?

Eight collections, one per rank in ``2**{0..7}`` = 1, 2, 4, 8, 16, 32, 64, 128.
Each is the same 16-point 25% simplex over three yahoo topic groups on
OLMo-2-0425-1B-Instruct, trained on the same 1000-row draw (seed 0, 5008 samples
seen) and read on the same 100-query 33/33/33 test draw.  The r=16 collection is
byte-identical to ``simplex3_olmo2``'s own sixteen adapters -- content-addressed
caching made it a cache hit, not a retrain -- so this sweep passes *through* the
existing experiment rather than beside it.

Four things about the design that the numbers do not show:

* **Rank varies capacity alone.**  ``gen_simplex3.py`` renders
  ``lora_alpha = 2 * rank``, so PEFT's ``alpha / rank`` scaling of ``B @ A`` is
  the constant 2 at every rank.  Had alpha been pinned instead, the update would
  have been scaled by ``1/rank`` and the axis would confound capacity with gain.
  The alpha is written into the CSV so this is checkable from the output.

* **The structural level is genuinely comparable across ranks, and this is not
  obvious.**  ``load_lora_weights`` returns the raw A (r x d_in) and B
  (d_out x r) factors, whose dimensions scale with rank, and the factorisation
  has a gauge freedom (A -> QA, B -> BQ^-1 leaves the model unchanged).  Neither
  reaches the score: ``structure.cosine_similarity_matrix`` computes cosines
  between the *products* ``B @ A``, via r x r traces rather than by forming the
  d x d matrix.  The represented object is therefore the weight delta, which has
  the same shape and the same gauge at every rank.

* **One ground truth, not two.**  Every adapter here trains at n_samples=1000,
  where largest-remainder allocation is exact on a 25% grid, so requested and
  realized mixtures coincide except at the centre point -- ``033g1_033g2_033g3``
  normalizes to exactly 1/3 and 1000 is not divisible by 3, a one-row, 1/1000
  discrepancy.  The nsweep settled that a deviation this size moves dCor* by
  exactly 0, so only the requested simplex is scored; the observed deviation is
  recorded per row as ``max_realized_deviation`` and bounded by
  ``suite.check_realized_truth``.

* **The dataset-embedding row is rank-invariant by construction**, and is scored
  anyway.  Its surrogate is the mean embedding of the *training draw*, which no
  adapter touches, so all eight ranks read one cached matrix and the panel is a
  flat line.  That line is the reference the other levels are climbing toward --
  the score a perfect read of this mixture geometry attains -- so dropping it
  would remove the only y-axis anchor the figure has.

There are no replicates: one seed per rank, so the panels carry a single curve
and no band.  The group-size sweep's variant A/B distinction does not arise
either -- the pure vertices and the even mixture are *in* every collection by
construction, so there is nothing to add to a matrix.

Usage::

    python figures/fig_structural_sweep/sweep_rsweep.py
    python figures/fig_structural_sweep/sweep_rsweep.py --perspective structural_all_o
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.discovery import CacheIndex, scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import raw_mixture_pcts, sort_by_mixture  # noqa: E402

# Imported, never copied: these are the project's standing per-level defaults,
# not a property of any one experiment, and two divergent copies would be a
# silent way for three figures to stop being comparable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex_collection_size"))
from sweep_group_size import _META_KEYS, canonical_perspectives  # noqa: E402

HERE = Path(__file__).resolve().parent
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
CACHE_ROOT = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/shared_cache")

#: The training recipe every adapter in this sweep shares. Pinning it is what
#: separates the rsweep's r=16 column from the 1004-adapter group-size pool and
#: the nsweep's other draw sizes, all of which are also yahoo, also this base
#: model, and also in this one content-addressed cache.
TRAIN_N, TRAIN_SEED = 1000, 0

#: ``lora_alpha = 2 * rank``, as ``gen_simplex3.lora_alpha`` renders it.
ALPHA_PER_RANK = 2

#: One row per (perspective, rank), scored against the requested simplex.
FIELDS = [
    "perspective", "taxonomy", "metric", "lora_rank", "lora_alpha", "n_models",
    "dcor", "disparity", "max_realized_deviation", "seconds",
]


def slice_matrix(index, ids, cache_root, spec, *, use_cache=True):
    """The one distance matrix for this perspective over the 16 models of one rank."""
    kwargs = {k: v for k, v in spec.items() if k not in _META_KEYS}
    dm, _ = build_taxonomy_artifacts(
        index, spec["taxonomy"], spec["metric"], cache_root=cache_root,
        n_components=(2,), use_cache=use_cache, id_scheme="model_id",
        label="rsweep rank", **kwargs,
    )
    return dm.reindex(list(ids))


def score_slice(dm, ids, vertex_names):
    """``(dcor, disparity)`` against the requested simplex, for one rank."""
    weights = suite.truth_weights(ids)
    tdm = simplex_distance_matrix(weights, list(ids), vertex_names)
    tgeo = simplex_geometry(weights, list(ids), vertex_names)
    return (
        dcor_vs_truth(dm, tdm),
        disparity_vs_truth(dm, tgeo, random_state=suite.MDS_SEED,
                           n_components=tgeo.coordinates.shape[1]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--perspective", action="append", dest="perspectives",
                    help="repeatable; default is every canonical perspective")
    ap.add_argument("--spec", default="yahoo",
                    help="DataSimplexSpec key naming the mixture grid")
    ap.add_argument("--rank", action="append", type=int, dest="ranks",
                    help="repeatable; default is every rank found in the cache")
    ap.add_argument("--n-expected", type=int, default=16,
                    help="models per rank; a collection of any other size is fatal")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="rsweep_scores.csv")
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

    # Two pins, because the corpus alone separates nothing here. The mixture
    # grid removes the group-size pool's 1%-grid points; the training recipe
    # removes the nsweep's other draw sizes and seeds. Measured: 2555 -> 1556 ->
    # 128, i.e. exactly 8 x 16.
    index = CacheIndex([e for e in index.entries
                        if raw_mixture_pcts(e.model_id) in wanted],
                       index.cache_root)
    index = index.filter(n_samples=TRAIN_N, seed=TRAIN_SEED)
    print(f"pinned to the {len(wanted)}-point grid at n={TRAIN_N}, "
          f"seed={TRAIN_SEED}: {len(index)} adapters", flush=True)

    collections = index.slices(by=("lora_rank",))
    if args.ranks:
        missing = sorted(set(args.ranks) - {k[0] for k in collections})
        if missing:
            raise SystemExit(f"no adapters at rank(s) {missing}")
        collections = {k: v for k, v in collections.items() if k[0] in args.ranks}
    ranks = [k[0] for k in sorted(collections)]
    print(f"{len(collections)} collections over ranks {ranks}", flush=True)

    bad = {k[0]: len(v.model_ids) for k, v in collections.items()
           if len(v.model_ids) != args.n_expected}
    if bad:
        raise SystemExit(
            f"{len(bad)} rank(s) do not hold {args.n_expected} models: {bad}. "
            "The ground truth is only unambiguous at the full simplex, so this "
            "is fatal rather than skippable.")

    specs = canonical_perspectives()
    unknown = set(args.perspectives or ()) - set(specs)
    if unknown:
        raise SystemExit(f"unknown perspective(s) {sorted(unknown)}. "
                         f"Choose from {sorted(specs)}")
    names = args.perspectives or list(specs)

    rows = []
    t_start = time.time()
    for name in names:
        pspec = specs[name]
        for rank in ranks:
            sub = collections[(rank,)]
            ids = sort_by_mixture(list(sub.model_ids))
            vertex_names = [f"g{j + 1}"
                            for j in range(suite.truth_weights(ids).shape[1])]

            t0 = time.time()
            dm = slice_matrix(sub, ids, cache_root, pspec,
                              use_cache=not args.no_cache)
            dcor, disparity = score_slice(dm, ids, vertex_names)
            dt = time.time() - t0
            rows.append({
                "perspective": name, "taxonomy": pspec["taxonomy"],
                "metric": pspec["metric"], "lora_rank": rank,
                "lora_alpha": ALPHA_PER_RANK * rank, "n_models": len(ids),
                "dcor": dcor, "disparity": disparity,
                "max_realized_deviation": suite.check_realized_truth(ids, TRAIN_N),
                "seconds": round(dt, 2),
            })
            print(f"  {name:<20} r={rank:<4} dcor={dcor:.4f} "
                  f"disp={disparity:.4f} {dt:6.1f}s", flush=True)

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
