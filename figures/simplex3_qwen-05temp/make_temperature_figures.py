#!/usr/bin/env python
"""Behavioral distance matrices and MDS embeddings across the Qwen temperature sweep.

The ``simplex3_qwen`` suite decoded its 16 adapters at ten sampling temperatures
(T = 0.1 … 1.0, R=8 each) alongside the greedy run, and embedded all of it. Every
behavioral figure in ``figures/simplex3_qwen_v*`` shows exactly two slices:
greedy, and T=1.0 at R=16.

This script draws the sweep: one grid per slice, three surrogates by four metrics
(cosine, frobenius, euclidean, CKA), scored against the ground-truth simplex by
distance correlation and Procrustes disparity.

All of the machinery is :mod:`src.plots.simplex_temperature`, which was this
file before a second suite needed it; what remains here is the run's own
coordinates. Read that module for why the sweep is read at a uniform R=8, why
greedy leads every figure, and what a slice, a surrogate and a perspective are.

The corpus filter is not optional even though this directory predates it:
``03_adapters/Qwen--Qwen3.5-4B`` now holds the dolly and oasst1 simplices too, so
an unfiltered scan returns 86 models where this figure wants its own 16.

Usage
-----
    python figures/simplex3_qwen-05temp/make_temperature_figures.py
    python figures/simplex3_qwen-05temp/make_temperature_figures.py --no-cache
    python figures/simplex3_qwen-05temp/make_temperature_figures.py --check-only
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

from src.plots import simplex_temperature as T  # noqa: E402

BASE_MODEL = "Qwen/Qwen3.5-4B"

#: Which corpus this driver plots. See the module docstring.
DATASETS = ["yahoo"]

#: The question-only query draw both inference stages used.
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}

#: The ten the sweep was generated at. Asserted rather than assumed, so a partial
#: sweep is an error here instead of a curve with silent gaps.
TEMPERATURES = [round(0.1 * i, 1) for i in range(1, 11)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass pairwise reads; results are still written back")
    ap.add_argument("--check-only", action="store_true",
                    help="resolve and validate the slices, then stop before plotting")
    args = ap.parse_args()

    T.run_temperature_suite(
        base_model=BASE_MODEL,
        draw=DRAW,
        outdir=args.outdir,
        cache_root=args.cache_root,
        datasets=DATASETS,
        expected_temperatures=TEMPERATURES,
        no_cache=args.no_cache,
        check_only=args.check_only,
        # The full grid: every surrogate by every metric, which is what this
        # directory exists to survey.
        surrogates=tuple(T.SURROGATES),
        metrics=T.METRICS,
        # The curves are `make_temperature_curves.py`, which reads the CSV this
        # run writes. Kept separate here because redrawing them is a second's
        # work and this run is not.
        curves=False,
        source="figures/simplex3_qwen-05temp/make_temperature_figures.py",
    )


if __name__ == "__main__":
    main()
