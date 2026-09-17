#!/usr/bin/env python
"""Procrustes disparity and dCor* against the composition of the query set.

**qmix** (*query mixture*) -- the axis of query-set composition: what fraction of
the 100-row probe is the corpus the adapters were trained on, the rest being a
diluting corpus they never saw.  One value of qmix is one ``(yahoo percentage,
diluent)`` pair.  See ``docs/terminology.md``.

One fixed fleet: the same sixteen OLMo-2-1B adapters of the yahoo 3-group 25%
simplex, at one rank, one init seed and one training draw, throughout.  Nothing
here was trained for this experiment; only the probe moves.

This file is purely plotting; every value is read from ``qmix_scores.csv``,
which ``sweep_qmix.py`` writes.

Reading the figure
------------------
The two estimators are on separate rows because they run in **opposite
directions**: a high dCor* means strong dependence on the ground truth, while a
Procrustes disparity of **0** means identical shape.  On one axis, one of the
curves would read backwards.

Five things the figure says out loud, each a property of the design rather than
of the data:

* **The ground truth never moves.**  Unlike every other sweep in this directory,
  the subjects are constant: one fleet, one set of training mixtures.  A curve
  that falls is the *level's recovery* of a fixed simplex degrading, with no
  confound from differently-trained adapters.

* **Two arms, two kinds of off-distribution.**  dolly is English instruction data
  -- same language, different task shape.  oasst1-zh is Chinese conversational
  data -- a different language entirely.  They are drawn as separate curves
  because there is no reason to expect one number to serve both.

* **The x axis is symlog below 1%,** because the interesting end is the sparse
  one: at 1% of a 100-row probe exactly one query is yahoo, and 0% is a real
  point (the fully-diluted floor) that a log axis cannot place.

* **The band is an interquartile range over ten draw seeds,** and it is widest at
  the low end by construction.  At 1% the identity of the single yahoo row is the
  dominant source of variance; that width is the measurement, not noise around
  it.

* **The undiluted point is the shared reference and is NOT the canonical probe.**
  The yahoo component of a qmix draw is unstratified -- all ten topics as one
  pool -- where the canonical probe draws the even g1/g2/g3 mixture.  So the 100%
  point here is a new draw with its own cached generations.  Read this figure
  against itself, across yahoo percentage.  Do not read its 100% point against
  the behavioral row of figure 2.

Usage
-----
::

    python figures/simplex3_qmix_olmo2/make_figures.py
    python figures/simplex3_qmix_olmo2/make_figures.py --arm dolly
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

from src.plots.config import set_style  # noqa: E402
from src.plots.figures import save_figure  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Column order and display names, one column per **perspective**.  A subset of
#: the list in ``figures/simplex_collection_size/make_figures.py``, in the same
#: order, so the panels that do appear line up with that figure's.  The
#: structural and dataset columns are absent because neither level reads the
#: query set: they are flat under this axis by construction, and a flat line in
#: a panel grid invites being read as a finding.
LEVELS = [
    ("functional_all", "Functional (a)\nall hidden states"),
    ("functional_last", "Functional (b)\nfinal hidden state"),
    ("behavioral", "Behavioral\nR=16 per query"),
    ("behavioral_greedy", "Behavioral (greedy)\nR=1 deterministic"),
]

#: Panels that differ from their neighbour by **decoding alone**.  Drawn with a
#: divider to their left, as in the collection-size and nsweep figures.
DECODING_PAIR = ("behavioral", "behavioral_greedy")

#: ``(column, axis label, whether high is good)`` per row of the grid.
#: Wrapped onto two lines: at four columns the panels are narrow, and the
#: one-line spellings the nsweep figure uses run over the y tick labels here.
SCORES = [
    ("dcor", "dCor*\n(1 = strong dependence)", True),
    ("disparity", "Procrustes disparity\n(0 = identical shape)", False),
]

#: The two diluent arms, as ``(csv value, label, colour, linestyle)``.  The
#: undiluted point belongs to neither and is drawn on both curves, because it is
#: the reference each arm departs from rather than a point of either.
ARMS = [
    ("dolly", "diluted with dolly (en, instructions)", "#2E5EAA", "-"),
    ("oasst1zh", "diluted with oasst1-zh (zh, conversation)", "#C1553B", "--"),
]

#: Where the linear part of the x axis gives way to the log part.  Below this the
#: axis is linear, so 0% -- a real point, and the fully-diluted floor -- has
#: somewhere to sit.
SYMLOG_THRESHOLD = 1.0

#: The yahoo percentage at which the probe holds exactly one yahoo row.
#: Annotated rather than dropped: it is the sharpest point of the experiment.
SINGLE_ROW_AT = 1


def read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def series(rows, perspective, column, arm):
    """``(pcts, lo, mid, hi)`` across draw seeds, for one panel and one arm.

    The undiluted point carries ``arm == "none"`` in the CSV and is folded into
    every arm here, so each curve runs the full width to its own reference rather
    than stopping at 50%.
    """
    by_pct = defaultdict(list)
    for r in rows:
        if r["perspective"] != perspective:
            continue
        if r["arm"] not in (arm, "none"):
            continue
        v = r.get(column)
        if v in (None, ""):
            continue
        by_pct[int(r["yahoo_pct"])].append(float(v))
    pcts = sorted(by_pct)
    if not pcts:
        return [], [], [], []
    lo = [float(np.percentile(by_pct[p], 25)) for p in pcts]
    mid = [float(np.median(by_pct[p])) for p in pcts]
    hi = [float(np.percentile(by_pct[p], 75)) for p in pcts]
    return pcts, lo, mid, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "qmix_scores.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--arm", choices=["both"] + [a[0] for a in ARMS],
                    default="both", help="draw one diluent arm instead of both")
    args = ap.parse_args()

    path = Path(args.scores)
    if not path.exists():
        raise SystemExit(f"no scores at {path}. Run sweep_qmix.py first.")
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"{path} is empty.")

    present = {r["perspective"] for r in rows}
    cols = [(k, lab) for k, lab in LEVELS if k in present]
    # Anything measured but not listed is appended rather than dropped, so
    # nothing scored goes unplotted.
    cols += [(k, k) for k in sorted(present - {k for k, _ in LEVELS})]
    if not cols:
        raise SystemExit(f"no known perspectives in {path}: found {sorted(present)}")

    arms = [a for a in ARMS if args.arm == "both" or a[0] == args.arm]

    set_style("two_col_full")
    fig, axes = plt.subplots(len(SCORES), len(cols),
                             figsize=(2.05 * len(cols), 5.2),
                             sharex=True, squeeze=False)

    for ri, (column, ylabel, _high_good) in enumerate(SCORES):
        for ci, (key, label) in enumerate(cols):
            ax = axes[ri][ci]
            drew = False
            for arm, alabel, colour, ls in arms:
                pcts, lo, mid, hi = series(rows, key, column, arm)
                if not pcts:
                    continue
                drew = True
                ax.fill_between(pcts, lo, hi, color=colour, alpha=0.18, lw=0)
                ax.plot(pcts, mid, ls, color=colour, lw=1.4, marker="o",
                        ms=2.8, label=alabel)
            ax.set_xscale("symlog", linthresh=SYMLOG_THRESHOLD)
            ax.set_xlim(-0.2, 130)
            ax.set_ylim(-0.03, 1.03)
            ax.axvline(SINGLE_ROW_AT, color="0.55", lw=0.7, ls=":", zorder=0)
            if not drew:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="0.5")
            if ri == 0:
                ax.set_title(label, fontsize=7.5)
            if ri == len(SCORES) - 1:
                ax.set_xlabel("yahoo % of the 100-row probe")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)
            else:
                ax.tick_params(labelleft=False)
            # The decoding pair is the only adjacent pair that differs by one
            # thing; the divider stops the eye grouping it with the rest.
            if key == DECODING_PAIR[1]:
                ax.spines["left"].set_linewidth(1.6)
                ax.spines["left"].set_color("0.35")

    handles, labs = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Taxonomy agreement with the mixture simplex vs. query-set composition\n"
        "OLMo-2-1B-Instruct · ONE fixed 16-adapter yahoo simplex · 100-row probe "
        "× 10 draw seeds · band = IQR over seeds\n"
        "yahoo component unstratified, so the 100% point is not the canonical probe",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))

    out = Path(args.outdir) / "fig_qmix_dataset_composition.png"
    save_figure(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
