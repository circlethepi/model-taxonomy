#!/usr/bin/env python
"""Recovery of the simplex against probe depth: number of queries, and R.

**qbig** (*big query set*) -- the axis of probe *depth*: the standing sixteen
OLMo-2-1B yahoo-simplex adapters, probed once at 1000 queries and R=64 sampled
replicates per query (plus greedy and the functional input-mode hidden states).
See ``docs/terminology.md``.

**R** -- the number of sampled generations per query that are averaged into the
query's embedding before distances are taken.  Greedy decoding has none.

Every number is read from ``qbig_subsample.csv``, which ``subsample_qbig.py``
writes by subsampling the one full-depth run: for each query count, ten query
sets drawn without replacement from the 1000; inside each, ten replicate sets per
R.  This file is purely plotting.

The figures (embedder ``v2moe`` unless ``--embedder`` says otherwise)
---------------------------------------------------------------------
``fig_qbig_depth_<embedder>.pdf``   **the main figure**
    Rows: Procrustes disparity, dCor*.  Columns: behavioral (sampled, one line
    per R), behavioral (greedy), functional (all hidden states), functional
    (final hidden state).  x is the number of queries.

``fig_qbig_*_<embedder>_linear.pdf``
    The same two figures with disparity on a linear 0-1 axis.

``fig_qbig_replicates_<embedder>.pdf``
    The sampled behavioral read alone, one column per R, so the R lines that
    share one panel in the main figure can be read without overplotting.

Reading the figures
-------------------
* **Disparity: down is better (0 = identical shape).  dCor*: up is better
  (1 = perfect).**  They are on separate rows and never share an axis.
* **A band is the 10th-90th percentile over trials** -- 100 per cell (ten
  replicate draws inside each of ten query draws), ten where R=64 (one replicate
  set exists), and none at 1000 queries with R=64, where exactly one subset exists
  and the marker *is* the measurement.  The line is the median.
* **The dotted vertical line is 100 queries**, the depth of the standing probe
  every other simplex figure reads; R=16 is its replicate count.  The qbig
  queries are a different draw (seed 2) from that probe (seed 1), so the point
  under the line is the standing depth, not the standing measurement.

Usage::

    python figures/simplex3_qbig_olmo2/make_qbig_figures.py
    python figures/simplex3_qbig_olmo2/make_qbig_figures.py --ext png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plots import save_figure  # noqa: E402
from src.plots.sweep_figures import (  # noqa: E402
    apply_style, embolden, place_caption, quantile_band,
)

HERE = Path(__file__).resolve().parent

# ── style: the qmix figures' values, so the two probe axes read alike ────────

#: Figure 2's colours for the two levels that read the query set -- the same
#: entries ``make_qmix_figures.LEVEL_COLORS`` carries.
LEVEL_COLORS = {"Functional": "#E69F00", "Behavioral": "#D55E00"}
#: Lighter than the qmix curves (2.1): the sampled panel carries six R lines.
SWEEP_LINEWIDTH = 1.6
SWEEP_MARKER = "o"
SWEEP_MARKER_SIZE = 3.2
#: Faint, so overlapping bands of neighbouring R do not read as a third colour.
BAND_ALPHA = 0.12
#: The percentiles a band spans.
BAND_Q = (10.0, 90.0)

#: One hue, light to dark, for R: magnitude, so sequential, and in the
#: behavioral level's own hue family so the sampled panel stays "behavioral".
#: The range stops short of the colormap's palest end, which vanishes on white.
R_CMAP = "Oranges"
R_CMAP_RANGE = (0.38, 1.0)

#: The standing probe's depth, marked on every panel.
STANDING_QUERIES = 100
QUERY_TICKS = [1, 10, 100, 1000]

#: ``(level, perspective in the CSV, panel title, level colour key)``.
PANELS = [
    ("behavioral", "sampled", "Behavioral · sampled, T=1", "Behavioral"),
    ("behavioral", "greedy", "Behavioral · greedy", "Behavioral"),
    ("functional", "all_states", "Functional · all states", "Functional"),
    ("functional", "final_state", "Functional · final state", "Functional"),
]

#: Disparity's y range per scale -- ``make_qmix_figures.YLIMS``, so the two
#: probe axes share a frame.  Log separates the functional values (~1e-2) from
#: the behavioral ones (~1e-1); linear runs the full 0-1 and is the honest read
#: of "how far from perfect".  Both are drawn; neither corrects the other.
DISPARITY_YLIMS = {"log": (2e-3, 1.4), "linear": (-0.03, 1.03)}

#: File-name suffix per disparity scale.  Log is the unsuffixed default.
YSCALE_SUFFIX = {"log": "", "linear": "_linear"}


def metrics(yscale):
    """``(column, y label, scale, limits)`` per row.  dCor* is always linear,
    where it can go negative; only disparity follows *yscale*."""
    return [
        ("disparity", "Procrustes disparity\n(0 = identical shape)", yscale,
         DISPARITY_YLIMS[yscale]),
        ("dcor", "dCor*\n(1 = perfect)", "linear", (None, 1.03)),
    ]

EMBEDDER_LABEL = {"v2moe": "nomic-embed-text-v2-moe",
                  "v15": "nomic-embed-text-v1.5"}

PAGE_WIDTH = 7.5


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no file at {path}. Run subsample_qbig.py first.")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    return rows


def select(rows, embedder, level, perspective, replicates=None):
    """The trials of one curve.  Functional rows carry embedder ``n/a``."""
    want_emb = "n/a" if level == "functional" else embedder
    return [r for r in rows
            if r["embedder"] == want_emb and r["level"] == level
            and r["perspective"] == perspective
            and (replicates is None or int(r["replicates"]) == replicates)]


def band(trials, column):
    by_n = defaultdict(list)
    for r in trials:
        by_n[int(r["n_queries"])].append(float(r[column]))
    return quantile_band(by_n, BAND_Q)


def r_colours(rs):
    cmap = plt.get_cmap(R_CMAP)
    return dict(zip(rs, cmap(np.linspace(*R_CMAP_RANGE, len(rs)))))


def draw(ax, trials, column, colour, label=None):
    xs, lo, mid, hi, _n = band(trials, column)
    if not xs:
        return
    if any(h > l for l, h in zip(lo, hi)):
        ax.fill_between(xs, lo, hi, color=colour, alpha=BAND_ALPHA, lw=0)
    ax.plot(xs, mid, "-", color=colour, lw=SWEEP_LINEWIDTH,
            marker=SWEEP_MARKER, ms=SWEEP_MARKER_SIZE, label=label)


def style_axes(ax, scale, lims):
    ax.set_xscale("log")
    ax.set_xticks(QUERY_TICKS)
    ax.set_xticklabels([str(t) for t in QUERY_TICKS])
    ax.set_xlim(0.7, 1450)
    ax.set_yscale(scale)
    lo, hi = lims
    ax.set_ylim(bottom=lo, top=hi)
    ax.axvline(STANDING_QUERIES, color="0.55", lw=0.7, ls=":", zorder=0)


def caption(embedder, what):
    return (
        f"{what} vs. probe depth · OLMo-2-1B-Instruct · ONE fixed 16-adapter "
        "yahoo simplex\n"
        "subsampled from one run at 1000 queries (seed 2) x R=64 · behavioral "
        f"embedder {EMBEDDER_LABEL.get(embedder, embedder)}\n"
        "line = median · band = 10-90% over query x replicate draws · "
        f"dotted = {STANDING_QUERIES} queries, the standing probe's depth"
    )


def fig_depth(rows, embedder, outdir, ext, yscale="log") -> Path:
    """The main figure: rows = metric, columns = level/perspective."""
    height = 5.6
    font = apply_style(PAGE_WIDTH)
    rows_m = metrics(yscale)
    fig, axes = plt.subplots(len(rows_m), len(PANELS),
                             figsize=(PAGE_WIDTH, height), sharex=True,
                             sharey="row", squeeze=False)
    rs = sorted({int(r["replicates"]) for r in
                 select(rows, embedder, "behavioral", "sampled")})
    colours = r_colours(rs)
    for mi, (col, ylabel, scale, lims) in enumerate(rows_m):
        for pi, (level, persp, title, ckey) in enumerate(PANELS):
            ax = axes[mi][pi]
            if persp == "sampled":
                for r in rs:
                    draw(ax, select(rows, embedder, level, persp, r), col,
                         colours[r], label=f"R={r}")
            else:
                draw(ax, select(rows, embedder, level, persp), col,
                     LEVEL_COLORS[ckey])
            style_axes(ax, scale, lims)
            if mi == 0:
                ax.set_title(title.replace(" · ", "\n"))
            if mi == len(rows_m) - 1:
                ax.set_xlabel("queries")
            if pi == 0:
                ax.set_ylabel(ylabel)
    handles, labs = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, 0.0),
                   title="sampled behavioral: replicates per query")
    place_caption(fig, caption(embedder, "Recovery of the mixture simplex"),
                  font, height, bottom=0.085)
    embolden(fig)
    return save_figure(fig, Path(outdir) / f"fig_qbig_depth_{embedder}{YSCALE_SUFFIX[yscale]}.{ext}")


def fig_replicates(rows, embedder, outdir, ext, yscale="log") -> Path:
    """Sampled behavioral alone, one column per R."""
    sampled = select(rows, embedder, "behavioral", "sampled")
    rs = sorted({int(r["replicates"]) for r in sampled})
    colours = r_colours(rs)
    height = 4.6
    font = apply_style(PAGE_WIDTH)
    rows_m = metrics(yscale)
    fig, axes = plt.subplots(len(rows_m), len(rs),
                             figsize=(PAGE_WIDTH, height), sharex=True,
                             sharey="row", squeeze=False)
    for mi, (col, ylabel, scale, lims) in enumerate(rows_m):
        for ri, r in enumerate(rs):
            ax = axes[mi][ri]
            draw(ax, [t for t in sampled if int(t["replicates"]) == r], col,
                 colours[r])
            style_axes(ax, scale, lims)
            ax.set_xticks([1, 100])
            ax.set_xticklabels(["1", "100"])
            if mi == 0:
                ax.set_title(f"R={r}")
            if mi == len(rows_m) - 1:
                ax.set_xlabel("queries")
            if ri == 0:
                ax.set_ylabel(ylabel)
    place_caption(fig, caption(embedder, "Sampled behavioral recovery"),
                  font, height)
    embolden(fig)
    return save_figure(fig,
                       Path(outdir) / f"fig_qbig_replicates_{embedder}"
                       f"{YSCALE_SUFFIX[yscale]}.{ext}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "qbig_subsample.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--embedder", default="v2moe",
                    choices=list(EMBEDDER_LABEL))
    ap.add_argument("--yscale", default="both",
                    choices=["both"] + list(DISPARITY_YLIMS),
                    help="disparity y scale; 'both' writes the log figures "
                         "and the _linear ones")
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    rows = read_rows(Path(args.scores))
    if not any(r["embedder"] == args.embedder for r in rows):
        raise SystemExit(f"{args.scores} has no rows for embedder "
                         f"{args.embedder!r}")
    scales = (list(DISPARITY_YLIMS) if args.yscale == "both"
              else [args.yscale])
    for yscale in scales:
        for fig in (fig_depth, fig_replicates):
            path = fig(rows, args.embedder, args.outdir, args.ext, yscale)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
