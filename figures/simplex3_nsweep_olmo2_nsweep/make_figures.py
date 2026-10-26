#!/usr/bin/env python
"""Procrustes disparity and dCor* against the size of the training draw.

**nsweep** -- a sweep over ``n_samples``, the size of the training draw, holding
everything else fixed.  See ``docs/terminology.md``.

**nsamples_train** -- one value of ``n_samples``, i.e. one of the nine training
draw sizes this sweep varies.  Named rather than called a "rung" because that
word already carries two other senses in this project: the retired one
``docs/terminology.md`` replaced with ``surrogate`` (a step on the ladder of
representations a level is read at), and the scale ladder of base model sizes
(the "1B rung", the "12B rung").

Ninety collections: the same 16-point 25% simplex over three yahoo topic groups,
trained at nine draw sizes x ten dataset seeds on OLMo-2-0425-1B-Instruct.

This file is purely plotting; every value is read from ``nsweep_scores.csv``,
which ``sweep_nsweep.py`` writes.

Reading the figure
------------------
The two estimators are on separate rows because they run in **opposite
directions**: a high dCor* means strong dependence on the ground truth, while a
Procrustes disparity of **0** means identical shape.  On one axis, one of the
curves would read backwards.

Four things the figure says out loud, each a property of the design rather than
of the data:

* **The x axis is logarithmic** and the draw sizes are roughly geometric, so equal
  horizontal steps are equal *ratios* of training data.

* **The band is an interquartile range across the ten dataset seeds**, and here
  -- unlike the group-size sweep next door -- it is an honest one.  Those
  replicates were drawn without replacement from one shared pool and so
  overlapped in membership; these ten seeds are ten independent draws of the
  corpus, and no two collections share an adapter.

* **Two ground truths are drawn per panel.**  Mixtures are allocated by largest
  remainder, so a requested 25/75 realizes as 3/7 at N=10.  The *requested*
  curve scores against the mixture the name asks for, the *realized* curve
  against the one the draw actually contained.  Where they separate, that part
  of the N-effect is label misspecification rather than representation noise;
  where they coincide, the discretisation never mattered.  They converge as N
  grows, and must: no weight moves by more than ``1/N``.

* **N=10 is near-degenerate and is drawn anyway.**  The effective batch is 16 >
  10, so the whole draw is one batch and the run is ~4 optimizer steps over a
  single full-batch gradient; it also trains 6.4 epochs rather than 5, because
  the 5N budget rounds up to a step boundary.  The optimizer was held fixed at
  every ``nsamples_train`` deliberately -- lowering the batch at small N would
  have varied the optimizer along the same axis as the data, and a difference
  between draw sizes would no longer be attributable to N.  It is marked on the
  axis.

Usage
-----
::

    python figures/simplex3_nsweep_olmo2_nsweep/make_figures.py
    python figures/simplex3_nsweep_olmo2_nsweep/make_figures.py --truth realized
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

#: Column order and display names, one column per **perspective**, matching
#: ``figures/simplex_collection_size/make_figures.py`` exactly so the two
#: figures can be read against each other panel for panel.
LEVELS = [
    ("dataset_embedding", "Dataset embedding\nmean · euclidean"),
    ("structural_all_o", "Structural (a)\nall layers · o_proj"),
    ("structural_all_qkvo", "Structural (b)\nall layers · q,k,v,o"),
    ("structural_last_o", "Structural (c)\nlast layer · o_proj"),
    ("functional_all", "Functional (a)\nall hidden states"),
    ("functional_last", "Functional (b)\nfinal hidden state"),
    ("behavioral", "Behavioral\nR=16 per query"),
    ("behavioral_greedy", "Behavioral (greedy)\nR=1 deterministic"),
]

#: Panels that differ from their neighbour by **decoding alone**. Drawn with a
#: divider to their left, as in the collection-size figure.
DECODING_PAIR = ("behavioral", "behavioral_greedy")

#: ``(column stem, axis label, whether high is good)`` per row of the grid.
SCORES = [
    ("dcor", "dCor*  (1 = strong dependence)", True),
    ("disparity", "Procrustes disparity  (0 = identical shape)", False),
]

#: The two ground truths, as ``(csv suffix, label, colour, linestyle)``.
TRUTHS = [
    ("requested", "requested mixture", "#2E5EAA", "-"),
    ("realized", "realized mixture", "#C1553B", "--"),
]

#: The ``nsamples_train`` where the effective batch (16) exceeds the whole draw,
#: so the run is
#: a handful of steps on one full-batch gradient. Annotated, not dropped.
DEGENERATE_AT = 10


def read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def series(rows, perspective, stem, truth):
    """``(ns, lo, mid, hi)`` across seeds, for one panel and one truth."""
    by_n = defaultdict(list)
    for r in rows:
        if r["perspective"] != perspective:
            continue
        v = r.get(f"{stem}_{truth}")
        if v in (None, ""):
            continue
        by_n[int(r["n_samples"])].append(float(v))
    ns = sorted(by_n)
    if not ns:
        return [], [], [], []
    lo = [float(np.percentile(by_n[n], 25)) for n in ns]
    mid = [float(np.median(by_n[n])) for n in ns]
    hi = [float(np.percentile(by_n[n], 75)) for n in ns]
    return ns, lo, mid, hi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "nsweep_scores.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--truth", choices=["both", "requested", "realized"],
                    default="both",
                    help="draw one ground truth instead of both")
    args = ap.parse_args()

    path = Path(args.scores)
    if not path.exists():
        raise SystemExit(f"no scores at {path}. Run sweep_nsweep.py first.")
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"{path} is empty.")

    present = {r["perspective"] for r in rows}
    cols = [(k, lab) for k, lab in LEVELS]
    # Anything measured but not listed is appended rather than dropped, so
    # nothing scored goes unplotted.
    cols += [(k, k) for k in sorted(present - {k for k, _ in LEVELS})]

    truths = [t for t in TRUTHS
              if args.truth == "both" or t[0] == args.truth]

    set_style("two_col_full")
    fig, axes = plt.subplots(len(SCORES), len(cols),
                             figsize=(2.05 * len(cols), 5.2),
                             sharex=True, squeeze=False)

    for ri, (stem, ylabel, high_good) in enumerate(SCORES):
        for ci, (key, label) in enumerate(cols):
            ax = axes[ri][ci]
            drew = False
            for truth, tlabel, colour, ls in truths:
                ns, lo, mid, hi = series(rows, key, stem, truth)
                if not ns:
                    continue
                drew = True
                ax.fill_between(ns, lo, hi, color=colour, alpha=0.18, lw=0)
                ax.plot(ns, mid, ls, color=colour, lw=1.4, marker="o",
                        ms=2.8, label=tlabel)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            if DEGENERATE_AT is not None:
                ax.axvline(DEGENERATE_AT, color="0.55", lw=0.7, ls=":", zorder=0)
            if not drew:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="0.5")
            if ri == 0:
                ax.set_title(label, fontsize=7.5)
            if ri == len(SCORES) - 1:
                ax.set_xlabel("training draw size $N$")
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
        "Taxonomy agreement with the mixture simplex vs. training draw size\n"
        "OLMo-2-1B-Instruct · yahoo 3-group 25% simplex · 16 models × 10 seeds "
        "per draw size · band = IQR over seeds",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.99))

    out = Path(args.outdir) / "fig_nsweep_dataset_size.png"
    save_figure(fig, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
