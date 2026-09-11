#!/usr/bin/env python
"""Agreement with the ground-truth simplex as a function of sampling temperature.

``make_temperature_figures.py`` scores every cell of the Qwen temperature sweep —
three surrogates by four metrics, at T = 0.1 … 1.0 — and writes the numbers to
``temperature_scores.csv``. Those scores reach the reader only as a markdown
table there, one table per temperature, which is eleven tables to hold in mind at
once. This script reads the same CSV back and draws them as curves instead:
temperature across the x axis, agreement up the y, one line per surrogate ×
metric cell.

The drawing is :func:`src.plots.simplex_temperature.write_curves`; this file is
the path to the CSV. Read that module for the encoding (colour is the pooling
method, dashes are the metric), for why greedy sits at T=0 under a labelled tick,
and for why some lines start at T=0.1 or are absent entirely.

Usage
-----
    python figures/simplex3_qwen-05temp/make_temperature_curves.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pinned before numpy loads its BLAS -- see the note in src/plots/simplex_suite.py.
os.environ.setdefault("MODEL_TAXONOMY_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plots import simplex_temperature as T  # noqa: E402

SCORES_CSV = HERE / "temperature_scores.csv"


def main() -> None:
    for path in T.write_curves(HERE, SCORES_CSV):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
