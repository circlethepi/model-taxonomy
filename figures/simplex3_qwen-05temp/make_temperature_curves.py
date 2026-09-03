#!/usr/bin/env python
"""Agreement with the ground-truth simplex as a function of sampling temperature.

``make_temperature_figures.py`` scores every cell of the Qwen temperature sweep —
three surrogates by four metrics, at T = 0.1 … 1.0 — and writes the numbers to
``temperature_scores.csv``. Those scores reach the reader only as a markdown
table there, one table per temperature, which is eleven tables to hold in mind
at once. This script reads the same CSV back and draws them as curves instead:
temperature across the x axis, agreement up the y, one line per surrogate ×
metric cell.

Terminology
-----------
surrogate
    What a model's generations are reduced to before any distance is taken —
    the CSV's ``surrogate`` column pairs it with the slice it was measured on
    (``T=0.4 · per query``). ``per generation`` keeps every replicate as its own
    row, ``per query`` averages a query's replicates back to one row each, and
    ``model mean`` collapses a model to a single row. This is the "pooling
    method": each surrogate pools a different amount of the sampling noise away.
dCor
    Distance correlation between the behavioral distance matrix and the
    ground-truth simplex distances. Higher is better agreement.
Procrustes disparity
    Residual after optimally rotating, scaling and translating one configuration
    onto the other. **Lower** is better agreement, so its figure reads inversely
    to the dCor one.

Two encodings, two figures
--------------------------
Colour carries the surrogate and linestyle carries the metric, so a reader
comparing pooling methods scans colours and a reader comparing metrics scans
dash patterns, without either question requiring the eleven-entry key that one
line-per-label legend would need. Both figures use the same encoding, so the
dCor and Procrustes figures can be read against each other directly.

Greedy is not on either x axis. ``do_sample=False`` makes the temperature
parameter inert, so the greedy slice has no temperature to be placed at; it is
the zero-noise baseline the sweep departs from rather than a point on the curve,
and ``make_temperature_figures.py`` already shows it beside the sweep. Its rows
in the CSV are skipped here.

``model mean`` has no CKA row at any temperature: the surrogate leaves one row
per model, and a CKA between two single rows is degenerate. That cell is absent
from the CSV rather than zero, and the corresponding line is simply not drawn.

Usage
-----
    python figures/simplex3_qwen-05temp/make_temperature_curves.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots import (  # noqa: E402
    PALETTE, encoding_legend, make_series, plot_lines, set_style,
)
from src.plots.figures import _get_fig_ax  # noqa: E402

OUTDIR = Path(__file__).resolve().parent
SCORES_CSV = OUTDIR / "temperature_scores.csv"

#: Surrogate → colour. Ordered coarsest-last, matching `make_temperature_figures`.
SURROGATE_COLORS = {
    "per generation": PALETTE[0],
    "per query": PALETTE[1],
    "model mean": PALETTE[2],
}

#: Metric → linestyle.
METRIC_STYLES = {
    "cosine": "-",
    "frobenius": "--",
    "euclidean": "-.",
    "cka": ":",
}

#: Score column → (axis label, figure filename). See the module docstring for
#: which direction each one reads.
SCORES = {
    "dcor": ("Distance correlation with the simplex", "fig_temperature_dcor.png"),
    "procrustes": ("Procrustes disparity (lower is better)",
                   "fig_temperature_procrustes.png"),
}

#: Where the two keys sit, in axes coordinates: outside the plotting area on the
#: right, stacked, so neither covers a curve.
LEGEND_LOCS = [(1.03, 0.50), (1.03, 0.00)]


def read_scores(path: Path) -> tuple[list[float], dict[tuple[str, str], dict[float, dict]]]:
    """``(temperatures, {(surrogate, metric): {temperature: row}})`` from the CSV.

    Rows whose slice carries no temperature — greedy — are dropped, and the
    temperature axis is whatever the remaining rows actually declare rather than
    a range spelled here, so a re-run over a different sweep needs no edit.
    """
    cells: dict[tuple[str, str], dict[float, dict]] = {}
    temps: set[float] = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            slice_label, surrogate = row["surrogate"].split(" · ")
            if not slice_label.startswith("T="):
                continue
            temp = float(slice_label.removeprefix("T="))
            temps.add(temp)
            cells.setdefault((surrogate, row["metric"]), {})[temp] = row
    return sorted(temps), cells


def curve_figure(temps, cells, column: str, ylabel: str, savepath: Path) -> Path:
    """One line per (surrogate, metric) cell, over the temperature axis.

    A cell missing at some temperature becomes NaN, which matplotlib leaves as a
    gap rather than interpolating across.
    """
    series = []
    for surrogate, color in SURROGATE_COLORS.items():
        for metric, linestyle in METRIC_STYLES.items():
            by_temp = cells.get((surrogate, metric))
            if not by_temp:
                continue
            ys = [float(by_temp[t][column]) if t in by_temp else np.nan for t in temps]
            # No label: the two keys below state the encoding instead.
            series.append(make_series(ys, color=color, linestyle=linestyle))

    fig, ax = _get_fig_ax(None)
    plot_lines(temps, series=series, ax=ax, xlabel="Sampling temperature",
               ylabel=ylabel, savepath=savepath)
    ax.set_xticks(temps)
    legends = encoding_legend(
        ax,
        ("Pooling", {s: {"color": c} for s, c in SURROGATE_COLORS.items()}),
        ("Metric", {m: {"color": "0.3", "linestyle": ls}
                    for m, ls in METRIC_STYLES.items()
                    if any((s, m) in cells for s in SURROGATE_COLORS)}),
        loc=LEGEND_LOCS,
    )
    # `save_figure` saves under the rcParams bbox, which crops the keys where
    # they hang off the axes; naming them as extra artists keeps them whole.
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)
    return savepath


def main() -> None:
    set_style("two_col", fig_width=8.0, fig_height=5.0)
    # Eleven lines distinguished partly by dash pattern: the preset's default
    # markers are wide enough to cover the pattern between two points.
    plt.rcParams["lines.markersize"] = 4
    temps, cells = read_scores(SCORES_CSV)
    for column, (ylabel, filename) in SCORES.items():
        path = curve_figure(temps, cells, column, ylabel, OUTDIR / filename)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
