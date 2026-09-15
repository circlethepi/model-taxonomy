#!/usr/bin/env python
"""Procrustes disparity against the LoRA rank, per canonical taxonomy level.

**rsweep** -- a sweep over the LoRA rank, holding the corpus, the mixture grid,
the training draw, the test-query draw, the seeds and the optimizer fixed.  See
``docs/terminology.md``.

**surrogate** -- a read-time view of one taxonomy level (a choice of layers,
projections, pooling and metric), as opposed to the level itself.  One panel per
canonical surrogate; the eight are the project's standing per-level defaults,
defined in ``figures/simplex_collection_size/sweep_group_size.py``.

Eight collections: the same 16-point 25% simplex over three yahoo topic groups,
trained at eight LoRA ranks (``2**{0..7}``) on OLMo-2-0425-1B-Instruct, all on
one 1000-row draw and read on one 100-query test draw.

This file is purely plotting; every value is read from ``rsweep_scores.csv``,
which ``sweep_rsweep.py`` writes.  Each run writes a panel grid, whose column
order matches the group-size and nsweep figures so the three can be read against
each other, and a single-axes overlay of the same curves, which is the quicker
read when the question is which level responds to rank at all.

Two axes choose which files come out, and every combination lands on its own
name, so a disparity figure and a dCor* figure of the same levels can sit side by
side instead of overwriting each other:

``--score {disparity,dcor,both}``
    Which estimator to draw.  A single score gives a one-row grid and the score's
    name in the filename; ``both`` draws two rows and keeps the unsuffixed name.

``--levels {primary,canonical,both}``
    Which perspectives to draw.  ``primary`` is the four **primary
    representations** and is the default; ``canonical`` is all eight; ``both``
    writes each.  Files of the primary set carry ``_primary``.

The primary representations
---------------------------
**These four are the set to plot by default in future figures**:
``dataset_embedding``, ``structural_all_o`` (all layers, o_proj),
``functional_all`` (all hidden states) and ``behavioral`` (R=16 per query) -- one
per taxonomy level, plus the dataset reference.  They are defined once, as
``sweep_group_size.PRIMARY_REPRESENTATIONS``, and imported here rather than
relisted, so this figure and its siblings cannot drift apart.

The other four canonical perspectives are **controls on the read, not levels**:
``structural_all_qkvo`` and ``structural_last_o`` move the structural scope,
``functional_last`` the functional scope, and ``behavioral_greedy`` moves decoding
alone.  They answer "does the scope matter on this axis", which is a separate
question from "do the levels agree with the simplex", and they crowd the headline
when both are asked at once.  All eight are still *scored* -- narrowing happens
at plot time, never at measure time, so no control ever needs a re-run.

Reading the figure
------------------
* **The x axis is log2 and the ticks are the ranks themselves**, so one
  horizontal step is one doubling of adapter capacity.

* **There is no band.**  One seed per rank, so each curve is eight single
  measurements rather than a median over replicates -- unlike the nsweep figure,
  whose ten dataset seeds give it an honest IQR.  Read the wobble as
  unquantified, not as absent.

* **Disparity is drawn on a log y axis and dCor* on a linear one.**  The
  disparities here span 0.006 to 0.66, two orders of magnitude, and half of them
  sit below 0.02; on a linear axis every level but the two worst would be a flat
  line on the floor.  dCor* is a correlation on [0, 1] and stays linear.

* **The two estimators run in opposite directions**, which is why they are never
  on one axis: high dCor* means strong dependence on the ground truth, while a
  Procrustes disparity of **0** means identical shape.

* **The dataset-embedding panel is flat by construction, and that is the
  check.**  Its surrogate is the mean embedding of the training draw, which no
  adapter touches, so all eight ranks read one cached matrix.  Measured: 0.0152
  disparity and 0.9797 dCor* at every rank, identical to four decimals.  It is
  the reference the adapter-derived levels climb toward, not a result.

* **r=16 is marked.**  It is the rank every earlier experiment in this project
  used, and its sixteen adapters are literally ``simplex3_olmo2``'s -- content-
  addressed caching made that column a cache hit rather than a retrain.  The
  marker says where the rest of the paper sits on this axis.

Usage
-----
::

    # the default: the four primary representations, disparity alone
    python figures/fig_structural_sweep/make_figures.py

    # dCor* alone, same four levels
    python figures/fig_structural_sweep/make_figures.py --score dcor

    # all eight canonical perspectives, both estimators
    python figures/fig_structural_sweep/make_figures.py --score both --levels canonical

    # everything, eight files
    python figures/fig_structural_sweep/make_figures.py --score both --levels both
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

from src.plots.config import set_style  # noqa: E402
from src.plots.figures import save_figure  # noqa: E402

# The primary set is defined at the same place as the canonical eight and
# imported, never relisted, so this figure cannot drift from its siblings.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex_collection_size"))
from sweep_group_size import PRIMARY_REPRESENTATIONS  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Column order and display names, one column per **perspective**, matching
#: ``figures/simplex3_nsweep_olmo2_nsweep/make_figures.py`` and
#: ``figures/simplex_collection_size/make_figures.py`` exactly so the three
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

#: The four **primary representations**: the default set to plot, one per
#: taxonomy level plus the dataset reference.  The remaining four canonical
#: perspectives are scope and decoding controls; see the module docstring.  Kept
#: in this file order rather than the tuple's so the primary figure's columns are
#: a subset of the canonical figure's in the same left-to-right order.
PRIMARY = tuple(PRIMARY_REPRESENTATIONS)

#: Perspective -> colour, following ``figures/figure2/make_figures.py``: one hue
#: per taxonomy family -- green data, blue structural, gold functional, red
#: behavioral -- with lightness separating the surrogates within a family, so the
#: overlay reads as four families before it reads as eight curves.
LEVEL_COLORS = {
    "dataset_embedding": "#009E73",
    "structural_all_o": "#08306B",
    "structural_all_qkvo": "#2171B5",
    "structural_last_o": "#6BAED6",
    "functional_all": "#B07800",
    "functional_last": "#E69F00",
    "behavioral": "#A03000",
    "behavioral_greedy": "#D55E00",
}

#: Panels that differ from their neighbour by **decoding alone**. Drawn with a
#: divider to their left, as in the two sibling figures.
DECODING_PAIR = ("behavioral", "behavioral_greedy")

#: ``(column stem, axis label, log y)`` per score. Disparity is logarithmic; see
#: the module docstring.
SCORES = {
    "disparity": ("disparity", "Procrustes disparity  (0 = identical shape)", True),
    "dcor": ("dcor", "dCor*  (1 = strong dependence)", False),
}

#: The rank every earlier experiment in this project used, and therefore the
#: column that is a cache hit against ``simplex3_olmo2`` rather than a retrain.
REFERENCE_RANK = 16

#: Floor for the log y axis. Disparities this small are numerically
#: indistinguishable from a perfect fit and should not set the axis.
DISPARITY_FLOOR = 4e-3


def read_rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def series(rows, perspective, stem):
    """``(ranks, values)`` for one panel, in ascending rank order."""
    by_rank = {}
    for r in rows:
        if r["perspective"] != perspective:
            continue
        v = r.get(stem)
        if v in (None, ""):
            continue
        by_rank[int(r["lora_rank"])] = float(v)
    ranks = sorted(by_rank)
    return ranks, [by_rank[k] for k in ranks]


def rank_axis(ax, ranks):
    """Log2 x axis ticked at the ranks themselves, not at powers of ten."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(ranks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=6.5)


def out_name(part, stems, primary):
    """One filename per (level set, score set), so no variant overwrites another.

    ``both`` estimators keep the unsuffixed name, which is the one the committed
    figures and the summary note already use.
    """
    stem = "fig_rsweep_lora_rank" + ("_primary" if primary else "")
    score = "" if len(stems) > 1 else f"_{stems[0]}"
    return f"{stem}{part}{score}.png"


def draw_grid(rows, cols, stems, outdir, primary=False):
    """One panel per perspective, one row per score."""
    set_style("two_col_full")
    fig, axes = plt.subplots(len(stems), len(cols),
                             figsize=(2.05 * len(cols), 2.7 * len(stems)),
                             sharex=True, squeeze=False)

    for ri, key_score in enumerate(stems):
        stem, ylabel, ylog = SCORES[key_score]
        vals = [v for k, _ in cols for v in series(rows, k, stem)[1]]
        lo = max(min(vals) * 0.6, DISPARITY_FLOOR) if ylog else -0.03
        hi = max(vals) * 1.6 if ylog else 1.03
        for ci, (key, label) in enumerate(cols):
            ax = axes[ri][ci]
            ranks, ys = series(rows, key, stem)
            if ranks:
                ax.plot(ranks, ys, "-", color=LEVEL_COLORS.get(key, "0.3"),
                        lw=1.4, marker="o", ms=3.0)
                rank_axis(ax, ranks)
            else:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="0.5")
            if ylog:
                ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            ax.axvline(REFERENCE_RANK, color="0.55", lw=0.7, ls=":", zorder=0)
            if ri == 0:
                ax.set_title(label, fontsize=7.5)
            if ri == len(stems) - 1:
                ax.set_xlabel("LoRA rank $r$")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)
            else:
                ax.tick_params(labelleft=False)
            # The decoding pair is the only adjacent pair that differs by one
            # thing; the divider stops the eye grouping it with the rest.
            if key == DECODING_PAIR[1]:
                ax.spines["left"].set_linewidth(1.6)
                ax.spines["left"].set_color("0.35")

    fig.suptitle(
        "Taxonomy agreement with the mixture simplex vs. LoRA rank"
        + (" — the four primary representations" if primary else "")
        + "\nOLMo-2-1B-Instruct · yahoo 3-group 25% simplex · 16 models per rank · "
        f"alpha = 2r · dotted line = r={REFERENCE_RANK}, the rank every other "
        "experiment uses",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out = Path(outdir) / out_name("", stems, primary)
    save_figure(fig, out)
    print(f"wrote {out}")


def draw_overlay(rows, cols, stems, outdir, primary=False):
    """Every level on one axes per score -- which levels respond to rank at all."""
    set_style("two_col_full")
    fig, axes = plt.subplots(1, len(stems), figsize=(4.2 * len(stems), 3.4),
                             squeeze=False)

    for ci, key_score in enumerate(stems):
        stem, ylabel, ylog = SCORES[key_score]
        ax = axes[0][ci]
        all_ranks: list[int] = []
        for key, label in cols:
            ranks, ys = series(rows, key, stem)
            if not ranks:
                continue
            all_ranks = ranks
            ax.plot(ranks, ys, "-", color=LEVEL_COLORS.get(key, "0.3"), lw=1.4,
                    marker="o", ms=3.0,
                    label=label.replace("\n", " · ") if ci == 0 else None)
        if all_ranks:
            rank_axis(ax, all_ranks)
        if ylog:
            ax.set_yscale("log")
        ax.axvline(REFERENCE_RANK, color="0.55", lw=0.7, ls=":", zorder=0)
        ax.set_xlabel("LoRA rank $r$")
        ax.set_ylabel(ylabel, fontsize=7.5)

    handles, labs = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=3, frameon=False,
                   fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        ("The four primary representations" if primary else "Every canonical level")
        + " against LoRA rank, one axes per estimator",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    out = Path(outdir) / out_name("_overlay", stems, primary)
    save_figure(fig, out)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "rsweep_scores.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--score", choices=["disparity", "dcor", "both"],
                    default="disparity",
                    help="which estimator to draw; disparity is the default. "
                         "A single score gives a one-row grid and puts its name "
                         "in the filename")
    ap.add_argument("--levels", choices=["primary", "canonical", "both"],
                    default="primary",
                    help="which perspectives to draw: the four primary "
                         "representations (the default), all eight canonical "
                         "perspectives, or one figure of each")
    args = ap.parse_args()

    path = Path(args.scores)
    if not path.exists():
        raise SystemExit(f"no scores at {path}. Run sweep_rsweep.py first.")
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"{path} is empty.")

    present = {r["perspective"] for r in rows}
    cols = [(k, lab) for k, lab in LEVELS if k in present]
    # Anything measured but not listed is appended rather than dropped, so
    # nothing scored goes unplotted.
    cols += [(k, k) for k in sorted(present - {k for k, _ in LEVELS})]

    stems = ["disparity", "dcor"] if args.score == "both" else [args.score]

    primary_cols = [(k, lab) for k, lab in cols if k in PRIMARY]
    missing = [k for k in PRIMARY if k not in present]
    if args.levels in ("primary", "both") and missing:
        # A primary row absent from the CSV is a scoring gap, not a plotting
        # choice: the headline figure would silently lose a level.
        raise SystemExit(
            f"{path} is missing primary representations {missing}; "
            "re-run sweep_rsweep.py for them or pass --levels canonical."
        )

    wanted = []
    if args.levels in ("primary", "both"):
        wanted.append((primary_cols, True))
    if args.levels in ("canonical", "both"):
        wanted.append((cols, False))

    for these, is_primary in wanted:
        draw_grid(rows, these, stems, args.outdir, primary=is_primary)
        draw_overlay(rows, these, stems, args.outdir, primary=is_primary)


if __name__ == "__main__":
    main()
