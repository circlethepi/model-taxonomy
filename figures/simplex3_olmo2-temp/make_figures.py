#!/usr/bin/env python
"""The temperature sweep of the yahoo simplex on OLMo-2-0425-1B-Instruct.

The ``simplex3_olmo2`` suite decoded its 16 yahoo adapters at ten sampling
temperatures (T = 0.1 … 1.0, R=8 each) alongside the greedy run, and embedded all
of it. This driver draws that sweep, as ``figures/simplex3_qwen-05temp`` does for
the 4B — the same slices, the same scores, the same figure names — so the two can
be read against each other and the question "does decoding temperature matter to
what the behavioral level recovers?" gets an answer at two model sizes.

One perspective, not a grid
---------------------------
A **perspective** is a surrogate together with a metric — one cell of a grid. The
qwen directory surveys twelve of them per slice (three surrogates by four
metrics) because its job was to find out which one to trust. That question is
settled: the behavioral level's **canonical perspective** is ``per query`` ×
``cosine`` — a query's replicates averaged back to one row each, then cosine
distances — which is what ``figures/figure2`` and the standing per-level defaults
read it at. This driver plots only that, so every figure here carries one panel
per temperature and the temperature axis is the only thing varying along it.

That is also why there are no per-slice grid files: a 1x1 "grid" per slice would
be eleven files each repeating one panel of the cross-slice figure.

Greedy
------
Greedy has one replicate, so averaging a query's replicates is the identity — its
``per query`` and ``per generation`` reads are the same representation under two
names, and the suite builds only the latter. It is kept as the sweep's T=0
baseline rather than dropped, and its row is labelled ``(R=1: no per-query mean)``
so no table claims it was averaged over replicates it does not have.

Scan filters
------------
All three are load-bearing here, and none of them is optional:
``03_adapters/allenai--OLMo-2-0425-1B-Instruct`` holds 2513 adapters. The corpus
filter drops dolly and oasst1; the training draw drops the nsweep tree's 90
further draws of this same corpus; and the mixture filter drops the group-size
pool's 1004 adapters, which share the corpus, the base model *and* the training
draw, so nothing coarser separates them. They are the same three
``figures/simplex3_olmo2/make_figures.py`` and ``figures/figure2`` pass.

Outputs
-------
``fig_behavioral_temps_dm_grid.png`` and ``_mds_grid.png`` (eleven slices, one
column), ``fig_behavioral_canonical_mds.png`` (the MDS strip along the
temperature axis), ``fig_temperature_dcor.png`` and
``fig_temperature_procrustes.png`` (the two agreement curves),
``fig_ternary_legend.png`` (the colour key), plus ``temperature_scores.csv`` and
``temperature_agreement.md``.

Usage
-----
    python figures/simplex3_olmo2-temp/make_figures.py
    python figures/simplex3_olmo2-temp/make_figures.py --check-only
    python figures/simplex3_olmo2-temp/make_figures.py --curves-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Pinned before numpy loads its BLAS -- see the note in src/plots/simplex_suite.py.
os.environ.setdefault("MODEL_TAXONOMY_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.plots import simplex_temperature as T  # noqa: E402

BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

#: The three scan filters. See the module docstring for what each one drops.
DATASETS = ["yahoo"]
TRAIN_DRAW = (1000, 0)
MIXTURES = SPECS["yahoo"].mixture_pcts()

#: The question-only query draw both inference stages used, identical to the
#: qwen sweep's -- which is what makes the two comparable.
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}

#: The ten the sweep was generated at. Asserted rather than assumed, so a partial
#: sweep is an error here instead of a curve with silent gaps.
TEMPERATURES = [round(0.1 * i, 1) for i in range(1, 11)]

SCORES_CSV = HERE / "temperature_scores.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass pairwise reads; results are still written back")
    ap.add_argument("--check-only", action="store_true",
                    help="resolve and validate the slices, then stop before plotting")
    ap.add_argument("--curves-only", action="store_true",
                    help="redraw the two curve figures from the scores CSV on "
                         "disk, computing nothing")
    args = ap.parse_args()

    if args.curves_only:
        if not SCORES_CSV.exists():
            raise SystemExit(f"no {SCORES_CSV} — run without --curves-only first")
        for path in T.write_curves(args.outdir, SCORES_CSV):
            print(f"wrote {path}")
        return

    T.run_temperature_suite(
        base_model=BASE_MODEL,
        draw=DRAW,
        outdir=args.outdir,
        cache_root=args.cache_root,
        datasets=DATASETS,
        train_draw=TRAIN_DRAW,
        mixtures=MIXTURES,
        expected_temperatures=TEMPERATURES,
        no_cache=args.no_cache,
        check_only=args.check_only,
        # The canonical behavioral perspective, and only it.
        surrogates=(T.CANONICAL_SURROGATE,),
        metrics=(T.CANONICAL_METRIC,),
        # One perspective per slice, so a per-slice grid would be a single panel
        # already drawn in the cross-slice figure.
        per_slice_grids=False,
        strip_figure="fig_behavioral_canonical_mds.png",
        strip_title="Behavioral simplex recovery across sampling temperature — "
                    "OLMo-2-1B, per query · cosine",
        report_title="Behavioral level across the temperature sweep — OLMo-2-1B, "
                     "per query · cosine",
        source="figures/simplex3_olmo2-temp/make_figures.py",
    )


if __name__ == "__main__":
    main()
