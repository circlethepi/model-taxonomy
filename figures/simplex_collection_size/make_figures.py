#!/usr/bin/env python
"""
procrustes disparity and dcorr vs number of models in the collection

999 models trained on randomly selected mixings of group 1, group 2,
group 3 for the yahoo Q&A task.

this file is purely plotting, all values are read from
group_size_scores.csv. Scores calculated in sweep_group_size.py.

Reading the figure
------------------
The two estimators are on separate rows because they run in **opposite
directions**: a high dCor* means strong dependence on the ground truth, while a
Procrustes disparity of **0** means identical shape. On one axis, one of the
curves would read backwards.

Three things the figure says out loud, each a property of the design rather
than of the data:

* **The x axis is logarithmic** and the grid is roughly geometric, so equal
  horizontal steps are equal *ratios* of collection size. That is the scale the
  ``1/(n(n-3))`` factor in dCor* and the similarity-transform degeneracy in
  Procrustes both live on.
* **The band is an interquartile range across replicates, not a confidence
  interval.** Replicates are drawn without replacement from one shared pool, so
  they overlap in membership and the spread is deflated by construction.
* **Past ``n_disjoint = 1`` the band is not a measurement at all**: every
  replicate is the same collection, so the spread is exactly zero. Those points
  are drawn hollow, with the band dashed from the last size that had at least a
  handful of disjoint groups, so a reader cannot mistake "no independence left"
  for "very precise".

Usage
-----
::

    python figures/simplex_collection_size/make_figures.py
    python figures/simplex_collection_size/make_figures.py --variant B
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

#: Below this many disjoint groups per shuffle, the replicates share so many
#: models that their spread describes the pool rather than the sampling.  Four
#: is a judgement call and is drawn, not hidden: the band goes dashed here.
DEFLATED_BELOW = 4

#: Column order and display names, one column per **perspective**.  Fixed rather
#: than read off the CSV so two runs' figures put the same perspective in the
#: same column, and grouped by taxonomy level so the three structural scopes sit
#: side by side and can be read against each other.
#:
#: A perspective the CSV does not carry is drawn as an empty panel rather than
#: dropped, so a partial run is visibly partial instead of quietly narrower.
#: A perspective the CSV carries and this list does not is appended on the right
#: under its own name, so nothing measured goes unplotted.
LEVELS = [
    ("dataset_embedding", "Dataset embedding\nmean · euclidean"),
    ("structural_all_o", "Structural (a)\nall layers · o_proj"),
    ("structural_all_qkvo", "Structural (b)\nall layers · q,k,v,o"),
    ("structural_last_o", "Structural (c)\nlast layer · o_proj"),
    ("functional_all", "Functional (a)\nall hidden states"),
    ("functional_last", "Functional (b)\nfinal hidden state"),
    ("behavioral", "Behavioral\nR=16 per query"),
]

#: Column heading for the perspective key.  ``perspective`` since 2026-09-09,
#: when the suite grew to seven perspectives over five levels; ``level`` before
#: that, when there was exactly one perspective per level.  Both are read, so an
#: older CSV still plots.
KEY_COLUMNS = ("perspective", "level")

#: ``(column, axis label, whether high is good)`` per row of the grid.
SCORES = [
    ("dcor", "dCor*  (1 = strong dependence)", True),
    ("disparity", "Procrustes disparity  (0 = identical shape)", False),
]

#: Fixed y limits, per score row.  Both estimators are pinned to their full
#: ``[0, 1]`` range rather than auto-scaled, because both are bounded: an
#: autoscaled axis makes a level that never leaves 0.01 fill its panel exactly
#: like one that reaches 0.5, and the panels stop being comparable either to
#: each other or to any figure outside this one.
#:
#: The cost is real and worth stating: dCor* separates the flat perspectives by
#: ~0.04 (structural (a) at 0.984 against (c) at 0.940), and on a full-range
#: axis that difference is four percent of the panel height.  It is legible in
#: ``collection_size_summary.md``, which is the table to read for it.  The
#: figure's job here is the shape of each curve against ``n`` and the gap
#: between the levels, and both survive the fixed range.
YLIM = {"dcor": (0.0, 1.0), "disparity": (0.0, 1.0)}


def read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} has no rows")
    key = next((c for c in KEY_COLUMNS if c in rows[0]), None)
    if key is None:
        raise SystemExit(
            f"{path} has no {' or '.join(KEY_COLUMNS)} column; its header is "
            f"{sorted(rows[0])}")
    text = {key, "taxonomy"}
    out = []
    for r in rows:
        rec = {"level": r[key], "taxonomy": r.get("taxonomy", "")}
        for k, v in r.items():
            if k in text:
                continue
            try:
                rec[k] = float(v) if "." in v or "e" in v.lower() or v in ("nan", "") \
                    else int(v)
            except ValueError:
                rec[k] = float("nan")
        out.append(rec)
    return out


def columns_for(rows):
    """``(key, title)`` per panel: the declared order, then anything unexpected.

    Keeps a figure honest in both directions -- a perspective in :data:`LEVELS`
    that the CSV lacks still gets its (empty) panel, and a perspective the CSV
    has that nobody declared is plotted under its bare key rather than dropped.
    """
    present = {r["level"] for r in rows}
    known = {k for k, _ in LEVELS}
    return list(LEVELS) + [(k, k) for k in sorted(present - known)]


def summarise(rows, level, column):
    """``(n, median, q1, q3, n_disjoint)`` arrays for one level and one column."""
    by_n = defaultdict(list)
    disjoint = {}
    for r in rows:
        if r["level"] != level:
            continue
        v = r[column]
        if np.isfinite(v):
            by_n[r["n"]].append(v)
        disjoint[r["n"]] = r["n_disjoint"]
    ns = sorted(by_n)
    if not ns:
        return None
    med = np.array([np.median(by_n[n]) for n in ns])
    q1 = np.array([np.percentile(by_n[n], 25) for n in ns])
    q3 = np.array([np.percentile(by_n[n], 75) for n in ns])
    return (np.array(ns, dtype=float), med, q1, q3,
            np.array([disjoint[n] for n in ns]))


def draw(rows, variants, outdir: Path, suffix: str = "") -> Path:
    set_style("two_col_full")
    cols = columns_for(rows)
    fig, axes = plt.subplots(len(SCORES), len(cols),
                             figsize=(2.9 * len(cols), 3.4 * len(SCORES)),
                             # One y scale per estimator row.  With seven
                             # columns the question is which perspective sits
                             # higher, and independent scales would let a level
                             # that never leaves 0.2 look like one that reaches
                             # 0.9.
                             sharey="row", squeeze=False)
    colours = {"A": "#1f77b4", "B": "#d62728"}
    labels = {"A": "variant A (references in the matrix)",
              "B": "variant B (sampled models only)"}

    for col, (level, title) in enumerate(cols):
        for row, (score, ylabel, _high_good) in enumerate(SCORES):
            ax = axes[row][col]
            drew = False
            for variant in variants:
                got = summarise(rows, level, f"{score}_{variant}")
                if got is None:
                    continue
                ns, med, q1, q3, nd = got
                drew = True
                c = colours[variant]
                solid = nd >= DEFLATED_BELOW
                ax.fill_between(ns, q1, q3, color=c, alpha=0.18, linewidth=0)
                if (~solid).any():
                    # Redraw the deflated tail's edges dashed, so the band there
                    # reads as "no independence left" rather than as precision.
                    tail = ~solid
                    if solid.any():          # bridge the last solid point
                        tail[np.argmax(~solid) - 1] = True
                    ax.plot(ns[tail], q1[tail], color=c, ls="--", lw=0.8)
                    ax.plot(ns[tail], q3[tail], color=c, ls="--", lw=0.8)
                ax.plot(ns[solid], med[solid], color=c, marker="o", ms=4,
                        lw=1.6, label=labels[variant] if row == 0 and col == 0
                        else None)
                if (~solid).any():
                    ax.plot(ns[~solid], med[~solid], color=c, marker="o", ms=4,
                            mfc="white", lw=1.6, ls="--")
            if not drew:
                ax.text(0.5, 0.5, "no rows", ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
            ax.set_xscale("log")
            if score in YLIM:
                ax.set_ylim(*YLIM[score])
            ax.grid(True, which="both", alpha=0.25)
            if row == 0:
                ax.set_title(title, fontsize=8)
            if row == len(SCORES) - 1:
                ax.set_xlabel("models in the collection, $n$")
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)

    handles, labs = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=len(handles),
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Score against collection size, at each level's standing perspectives",
                 fontsize=11, y=0.995)
    fig.text(0.5, 0.945,
             "Band = interquartile range across replicates, not a confidence "
             "interval: replicates are drawn without replacement from one pool, "
             "so they share models and the spread is deflated.\n"
             f"Hollow markers and a dashed band mark sizes with fewer than "
             f"{DEFLATED_BELOW} disjoint groups per shuffle, where that "
             "deflation dominates.",
             ha="center", va="top", fontsize=7, color="0.35", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.905))
    return save_figure(fig, outdir / f"fig_collection_size{suffix}.png")


def write_summary(rows, variants, outdir: Path) -> Path:
    """A small markdown table beside the figure, for reading without the image."""
    lines = ["# Score against collection size", "",
             "Both variants; median over replicates.",
             "`n_disjoint` is how many disjoint groups one shuffle of the pool "
             "yields at that size — the honest ceiling on how independent the "
             "replicates are.", ""]
    for level, title in columns_for(rows):
        got = summarise(rows, level, f"dcor_{variants[0]}")
        if got is None:
            continue
        lines += [f"## {title.replace(chr(10), ' — ')}", "",
                  "| n | n_disjoint | "
                  + " | ".join(f"{s}_{v}" for s, _, _ in SCORES
                               for v in variants) + " |",
                  "|---" * (2 + len(SCORES) * len(variants)) + "|"]
        ns = got[0]
        for i, n in enumerate(ns):
            cells = []
            for score, _, _ in SCORES:
                for v in variants:
                    g = summarise(rows, level, f"{score}_{v}")
                    cells.append(f"{g[1][i]:.4f}" if g is not None else "—")
            lines.append(f"| {int(n)} | {int(got[4][i])} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    path = outdir / "collection_size_summary.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(HERE),
                    help="directory holding group_size_scores.csv, and where "
                         "the figure is written")
    ap.add_argument("--csv", default=None,
                    help="the scores CSV (default: <outdir>/group_size_scores.csv)")
    ap.add_argument("--variant", action="append", dest="variants",
                    choices=["A", "B"],
                    help="restrict to one variant; repeat for several "
                         "(default: both)")
    ap.add_argument("--suffix", default="",
                    help="appended to the figure filename")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csv_path = Path(args.csv) if args.csv else outdir / "group_size_scores.csv"
    if not csv_path.exists():
        raise SystemExit(f"no scores at {csv_path}. Run sweep_group_size.py "
                         "first.")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(csv_path)
    variants = args.variants or ["A", "B"]

    fig_path = draw(rows, variants, outdir, args.suffix)
    md_path = write_summary(rows, variants, outdir)
    print(f"wrote {fig_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
