#!/usr/bin/env python
"""Procrustes disparity against query-set composition, and the geometry behind it.

**qmix** (*query mixture*) -- the axis of query-set *composition*: what fraction
of the 100-row probe is the corpus the adapters were trained on, the rest being a
diluting corpus they never saw.  One value of qmix is one ``(yahoo percentage,
diluent)`` pair.  See ``docs/terminology.md``.

**arm** -- one diluting corpus.  There are two: ``dolly`` (English instructions)
and ``oasst1zh`` (Chinese conversation).

**draw seed** -- which 100 rows a composition drew.  Ten per composition,
submitted as ten independent jobs, so a scoring run meets whatever has landed.

One fixed fleet throughout: the same sixteen OLMo-2-1B adapters of the yahoo
3-group 25% simplex, at one rank, one initialisation seed and one training draw.
Nothing here was trained for this experiment; only the probe moves.  So a curve
that rises is a level's *recovery of a fixed simplex* degrading, with no confound
from differently-trained adapters.

This file is purely plotting.  Every number and every coordinate is read from the
two CSVs ``sweep_qmix.py`` writes -- including every Procrustes alignment, which
happens there so that two builds of this figure cannot disagree about a frame.

Only the functional and behavioral levels appear
------------------------------------------------
The three structural rows and the dataset-embedding row do not read the query set
at all: structural reads LoRA weights, dataset reads the training draw.  Under
this axis they are flat by construction, and a flat line beside two that move
invites being read as a result.  They are already measured, at full strength, in
the ``simplex3_olmo2`` figures.

Two behavioral reads, never on one axes
---------------------------------------
Every figure here is written **twice**, ``_sampled`` (R=16, per-query mean over
replicates) and ``_greedy`` (R=1, deterministic).  The pair differs by *decoding
alone*, so overplotting them would invite reading the gap between them as a third
level.  The gap is decode noise, and it is read by putting the two files side by
side.  The functional line is identical in both, and is drawn in both so each
file stands alone.

The figures
-----------
``fig_qmix_disparity_arms_<variant>.pdf``   **the main figure**
    Procrustes disparity against yahoo percentage, one panel per arm, one line
    per level.  Within a panel the only comparison the eye makes is functional
    against behavioral, which is the comparison the experiment is about.

``fig_qmix_disparity_<variant>.pdf``
    The same four curves on one axes -- level by colour, arm by dash -- for
    reading the two arms against each other instead.

``fig_qmix_mds_<arm>_<variant>.pdf``
    The geometry the disparities summarise: rows are levels, columns are yahoo
    percentages.  Each panel carries three things at once, per :func:`draw_mds`:
    the **pooled** fit (distances averaged over seeds, then embedded once), the
    **mean location** of each adapter over the *aligned* per-seed fits, and those
    aligned per-seed fits themselves, overplotted.  At a single extracted seed
    all three coincide exactly, and the panel correctly shows one marker per
    adapter.

Reading the figures
-------------------
* **A disparity of 0 means identical shape**, so on these axes *down is better*
  and a rising curve is a level losing the simplex.  dCor* runs the other way
  and is in the CSV but not in this file; the two are never on one axis.

* **The x axis is symlog below 1%**, because the interesting end is the sparse
  one: at 1% of a 100-row probe exactly one query is yahoo, and 0% is a real
  point -- the fully-diluted floor -- that a log axis cannot place.

* **The undiluted point is the shared reference and is NOT the canonical probe.**
  The yahoo component of a qmix draw is unstratified -- all ten topics as one
  pool -- where the canonical probe draws the even g1/g2/g3 mixture.  So the 100%
  point here is a new draw with its own cached generations.  It is drawn on both
  arms' curves because it is the reference each arm departs from.  Read this
  figure against itself, across yahoo percentage; do not read its 100% point
  against the behavioral row of figure 2.

* **A band is the interquartile range over draw seeds**, and it appears only
  where more than one seed has been extracted.  Where a point rests on one seed
  there is no band and the marker is the measurement -- the figure does not
  invent a spread it has not measured, and :func:`seed_report` says in the
  caption how many seeds each curve actually rests on.

* **MDS axes carry no units and no origin**, so the geometry panels are drawn at
  equal aspect with no ticks.  Only relative distances mean anything.  Every
  panel is rotated onto one frame upstream (pure g1 north, pure g2 east), so a
  shape can be compared along a row without re-deriving which way is which.

Usage
-----
::

    python figures/simplex3_qmix_olmo2/make_qmix_figures.py
    python figures/simplex3_qmix_olmo2/make_qmix_figures.py --variant greedy
    python figures/simplex3_qmix_olmo2/make_qmix_figures.py --figure main
    python figures/simplex3_qmix_olmo2/make_qmix_figures.py --yscale linear
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

from src.plots import save_figure, set_style  # noqa: E402
from src.plots.config import bold_capable_family  # noqa: E402
from src.plots.simplex import (  # noqa: E402
    frame_labels, mixture_label, model_colors, n_groups, sort_by_mixture,
)

HERE = Path(__file__).resolve().parent

# ── style, taken from figure 2 rather than reinvented ────────────────────────
# These are figure 2's own values, copied deliberately: the two figures are read
# in one document, so a level that is gold there must be gold here. They are
# copied rather than imported because figure 2's driver runs a nine-panel build
# on import-time constants of its own, and importing it to borrow four colours
# would drag that in. The registry in `src/plots/simplex_runs.py` is NOT the
# place to put them -- editing it would silently recolour the tracked figures in
# `figures/tasks_aggregate` and `figures/simplex3_aggregate`.

#: Taxonomy level -> colour. Gold functional, red behavioral: figure 2's entries
#: for the two levels that read the query set.
LEVEL_COLORS = {"Functional": "#E69F00", "Behavioral": "#D55E00"}

#: One text size for everything, at one weight -- panel titles, axis labels,
#: tick labels, legends and the caption alike. Figure 2's value, and its rule:
#: several panels in one frame typed at different sizes reads as emphasis rather
#: than as provenance, so there is one size and no sub-scaling of it.
FONT_SIZE = 13

#: The width :data:`FONT_SIZE` is written for -- figure 2's own reference, so a
#: build of this file at a given width is typed the same as a build of that one.
#: 19 in is its nine-panel full build; these figures are narrower and so read
#: the size through :func:`apply_style`.
REFERENCE_WIDTH = 19.0

#: The smallest the text may get whatever the width asks for. Figure 2's value,
#: and its argument: scaling 13 pt down with the width would type a 7.5 in build
#: at 5.1 pt, which is a picture of a figure rather than a figure -- a reader
#: cannot read 5 pt in print, so the panels give up the room instead.
FONT_SIZE_FLOOR = 7.0

#: The curves carry the panel, so they are drawn heavier than a default line.
SWEEP_LINEWIDTH = 2.1
SWEEP_MARKER = "o"
SWEEP_MARKER_SIZE = 7

#: The IQR band is context for its curve, not a second series. Faint enough that
#: two overlapping bands do not read as a third colour.
BAND_ALPHA = 0.12

#: The shape a mixture point takes, and the shape the **frame points** take
#: instead -- the three pure mixtures and the even one, which are the four points
#: `frame_labels` names and the ones every panel is rotated onto. They are the
#: only points a reader looks up by position rather than by colour.
#:
#: A triangle of a given area reads smaller than a circle of it, so the frame
#: points are drawn at :data:`FRAME_MARKER_SCALE` times the area of the rest.
MIXTURE_MARKER_SHAPE = "o"
FRAME_MARKER_SHAPE = "^"
FRAME_MARKER_SCALE = 1.3

#: Figure 2's axis colour, so the two figures' frames match.
AXIS_COLOR = "0.35"

# ── this figure's own vocabulary ─────────────────────────────────────────────

#: ``(perspective in the CSV, level name)`` per decoding variant.  The functional
#: entry is the same in both: the level does not decode, and it is drawn in both
#: files so each stands alone.  The behavioral entry is the variant.
#:
#: Both are figure 2's pinned perspectives for their level -- functional as *all
#: hidden states* x cosine, behavioral x cosine -- so a curve here and a bar
#: there are the same measurement.
VARIANTS = {
    "sampled": [("functional_all", "Functional"), ("behavioral", "Behavioral")],
    "greedy": [("functional_all", "Functional"),
               ("behavioral_greedy", "Behavioral")],
}

#: How each variant names its behavioral read, for the caption.
VARIANT_BEHAVIORAL = {"sampled": "R=16, per-query mean over replicates",
                      "greedy": "greedy, R=1 deterministic"}

#: ``(csv value, label, linestyle)`` per arm.  The dash separates the arms in the
#: single-axes figure; in the two-panel figure the panel does, and both arms are
#: drawn solid there.
ARMS = [
    ("dolly", "dolly (en, instructions)", "-"),
    ("oasst1zh", "oasst1-zh (zh, conversation)", "--"),
]

#: The yahoo percentage whose panel belongs to neither arm. It appears in both
#: arms' grids, because it is the reference each departs from, and is fenced off
#: with a heavier left spine -- the divider idiom the collection-size and nsweep
#: figures already use for a column that differs from its neighbour in kind
#: rather than in degree.
SHARED_PCT = 100

#: The arm value the undiluted point carries in the CSV.  It belongs to neither
#: arm and is folded into both, because it is the reference each departs from.
SHARED_ARM = "none"

#: ``seed`` of a row that summarises the seeds rather than being one of them --
#: ``sweep_qmix.POOLED``.  It is not a seed, so it is excluded from every median
#: and every band; including it would weight the summary in with the sample it
#: summarises.
POOLED = -1

#: Where the linear part of the x axis gives way to the log part.  Below this the
#: axis is linear, so 0% -- a real point, and the fully-diluted floor -- has
#: somewhere to sit.
SYMLOG_THRESHOLD = 1.0

#: The yahoo percentage at which the probe holds exactly one yahoo row.
#: Annotated rather than dropped: it is the sharpest point of the experiment.
SINGLE_ROW_AT = 1

#: y range per scale.  Log is the default for the same reason figure 2's sweep
#: row uses it: the functional disparities here are ~1e-2 and the behavioral ones
#: ~1e-1, and on a linear 0-1 axis the functional curve is a flat line on the
#: floor.  Linear runs the full 0-1 and is the honest read of "how far from
#: perfect", so both are kept and neither is a correction of the other.
YLIMS = {"log": (2e-3, 1.4), "linear": (-0.03, 1.03)}

#: How much room to leave around the outermost point of a geometry grid, as a
#: fraction of the span it has to cover. Without it the pure-mixture markers sit
#: on the panel's own frame and are clipped by it -- and those are exactly the
#: points a reader looks up by position.
GEOMETRY_PAD = 0.12

#: How faint one seed's own fit is drawn in a geometry panel.  Low, because at
#: ten seeds there are 160 of these points and they are the texture of the panel
#: rather than its subject.
SEED_ALPHA = 0.30
SEED_MARKER_SIZE = 9

#: The spoke from a seed's position to its adapter's mean location.  Thin and
#: faint: it exists to say *which* mean a stray point belongs to, and a reader
#: who does not need that should not see it.
SPOKE_LW = 0.5
SPOKE_ALPHA = 0.35

#: The pooled fit's marker: an unfilled ring behind the filled mean, so the two
#: reads are distinguishable where they disagree and coalesce where they do not.
POOLED_MARKER_SIZE = 74
POOLED_MARKER_LW = 1.1
MEAN_MARKER_SIZE = 30


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no file at {path}. Run sweep_qmix.py first.")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    return rows


def apply_style(width: float) -> float:
    """The house style, then one text size over the top of it.

    Returns the font size actually used.  ``set_style`` scales tick labels and
    legends to three quarters of the base size and leaves titles at it, which is
    right for a panel that is a figure in its own right; here several panels sit
    in one frame and that difference reads as emphasis rather than as
    provenance, so it is flattened.

    The weight needs the family changed with it: ``set_style`` prefers Libre
    Franklin, a *variable* font matplotlib registers at exactly one weight --
    Thin -- so ``fontweight="bold"`` against it is a silent no-op that renders
    Thin.  ``bold_capable_family`` probes for the first family in the stack that
    actually ships more than one weight.
    """
    font = max(FONT_SIZE * min(1.0, width / REFERENCE_WIDTH),
               FONT_SIZE_FLOOR)
    set_style("two_col_full")
    matplotlib.rcParams.update({
        "font.size": font, "axes.titlesize": font, "axes.labelsize": font,
        "xtick.labelsize": font, "ytick.labelsize": font,
        "legend.fontsize": font, "legend.title_fontsize": font,
        "figure.titlesize": font,
        "font.sans-serif": [bold_capable_family(), "DejaVu Sans"],
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold", "axes.edgecolor": AXIS_COLOR,
    })
    return font


def embolden(fig) -> None:
    """Tick labels and legend entries to the same weight as everything else.

    ``font.weight`` reaches text matplotlib creates from the rc defaults, but
    tick labels and legend entries are not among them: there is no
    ``xtick.labelweight`` rc, and a legend builds its texts from the handles'
    labels.
    """
    family = bold_capable_family()
    for ax in fig.axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_fontfamily(family)
    for legend in [ax.get_legend() for ax in fig.axes] + fig.legends:
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontweight("bold")
                txt.set_fontfamily(family)


# ── the disparity curves ─────────────────────────────────────────────────────

def series(rows, perspective, arm, column="disparity"):
    """``(pcts, lo, mid, hi, n_seeds)`` for one curve.

    The undiluted point carries ``arm == "none"`` and is folded into every arm,
    so each curve runs the full width to its own reference rather than stopping
    at 50%.

    *lo* and *hi* are the interquartile range **over draw seeds**.  Where a point
    rests on one seed they equal *mid*, which draws as a band of zero width --
    the figure does not invent a spread it has not measured.  ``n_seeds`` is the
    per-point count, returned so the caption can say what the curve rests on.
    """
    by_pct = defaultdict(list)
    for r in rows:
        if r["perspective"] != perspective:
            continue
        if r["arm"] not in (arm, SHARED_ARM):
            continue
        if int(r["seed"]) == POOLED:
            continue
        v = r.get(column)
        if v in (None, ""):
            continue
        by_pct[int(r["yahoo_pct"])].append(float(v))
    pcts = sorted(by_pct)
    return (pcts,
            [float(np.percentile(by_pct[p], 25)) for p in pcts],
            [float(np.median(by_pct[p])) for p in pcts],
            [float(np.percentile(by_pct[p], 75)) for p in pcts],
            {p: len(by_pct[p]) for p in pcts})


def seed_report(counts: dict) -> str:
    """"1 seed" or "3-10 seeds" -- what the curves in this figure rest on.

    Written from the data rather than from the spec, because the ten draw seeds
    of a composition are ten independent jobs and a partly-run tree is the normal
    case rather than the exception.
    """
    ns = sorted({n for c in counts for n in c.values()})
    if not ns:
        return "no seeds"
    if len(ns) == 1:
        return f"{ns[0]} draw seed" + ("" if ns[0] == 1 else "s")
    return f"{ns[0]}-{ns[-1]} draw seeds"


def draw_curve(ax, rows, perspective, level, arm, *, linestyle, label):
    """One level's disparity curve for one arm.  Returns its per-point seed counts."""
    pcts, lo, mid, hi, counts = series(rows, perspective, arm)
    if not pcts:
        return {}
    colour = LEVEL_COLORS[level]
    if any(h > l for l, h in zip(lo, hi)):
        ax.fill_between(pcts, lo, hi, color=colour, alpha=BAND_ALPHA, lw=0)
    ax.plot(pcts, mid, linestyle, color=colour, lw=SWEEP_LINEWIDTH,
            marker=SWEEP_MARKER, ms=SWEEP_MARKER_SIZE ** 0.5 * 1.6,
            label=label)
    return counts


def style_curve_axes(ax, yscale) -> None:
    ax.set_xscale("symlog", linthresh=SYMLOG_THRESHOLD)
    ax.set_xlim(-0.2, 160)
    ax.set_yscale(yscale)
    ax.set_ylim(*YLIMS[yscale])
    ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
    # The one-yahoo-row mark. Behind the curves: it is a reference, not a series.
    ax.axvline(SINGLE_ROW_AT, color="0.55", lw=0.7, ls=":", zorder=0)


#: Lines in a caption, and the padding around it, in multiples of the text
#: size.  The caption is a fixed number of lines by construction -- see
#: :func:`caption` -- so the band it needs is arithmetic rather than a measured
#: bounding box.
CAPTION_LINES = 3.6


def place_caption(fig, text, font, height, *, bottom=0.0) -> None:
    """Lay the axes out, then put the caption in the band left for it.

    **The order matters and is the reason this is a function.**
    ``tight_layout`` accounts for a suptitle that already exists *in addition to*
    whatever ``rect`` reserves, so laying out first against a rect and adding the
    caption afterwards is the only way to reserve the band exactly once.  Doing
    it the other way round double-counts: the panels lose the caption's height
    twice and the figure grows a band of dead space between the two.
    """
    band = CAPTION_LINES * font / 72.0 / height
    fig.tight_layout(rect=(0, bottom, 1, 1.0 - band))
    fig.suptitle(text, fontsize=font, y=0.995, va="top")


def caption(variant: str, counts, extra: str = "") -> str:
    return (
        "Procrustes disparity against the mixture simplex vs. query-set "
        "composition\n"
        "OLMo-2-1B-Instruct · ONE fixed 16-adapter yahoo simplex · "
        f"100-row probe · behavioral: {VARIANT_BEHAVIORAL[variant]}\n"
        f"0 = identical shape · band = IQR over {seed_report(counts)} · "
        "yahoo component unstratified, so the 100% point is not the canonical "
        f"probe{extra}"
    )


#: The width of a standalone figure here: figure 2's "page" preset, which is a
#: two-column page's full measure. Every text size follows from it.
PAGE_WIDTH = 7.5

#: Heights of the two curve figures. The two-panel build is shorter per panel
#: because it has two of them side by side.
ARMS_HEIGHT = 3.0
ONE_AXES_HEIGHT = 3.4


def fig_disparity_arms(rows, variant, yscale, outdir, ext="pdf") -> Path:
    """The main figure: one panel per arm, one line per level."""
    levels = VARIANTS[variant]
    font = apply_style(PAGE_WIDTH)
    fig, axes = plt.subplots(1, len(ARMS), figsize=(PAGE_WIDTH, ARMS_HEIGHT),
                             sharey=True, squeeze=False)
    counts = []
    for ci, (arm, alabel, _ls) in enumerate(ARMS):
        ax = axes[0][ci]
        for perspective, level in levels:
            # Solid in both panels: the panel separates the arms here, so a dash
            # would encode the arm twice and the level not at all.
            counts.append(draw_curve(ax, rows, perspective, level, arm,
                                     linestyle="-", label=level))
        style_curve_axes(ax, yscale)
        ax.set_title(f"diluted with {alabel}")
        ax.set_xlabel("yahoo % of the 100-row probe")
        if ci == 0:
            ax.set_ylabel("Procrustes disparity\n(0 = identical shape)")

    handles, labs = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labs, loc="lower center", ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    place_caption(fig, caption(variant, counts), font, ARMS_HEIGHT,
                  bottom=0.07)
    embolden(fig)
    return save_figure(fig, Path(outdir) /
                       f"fig_qmix_disparity_arms_{variant}.{ext}")


def fig_disparity_one_axes(rows, variant, yscale, outdir,
                          ext="pdf") -> Path:
    """The same curves on one axes: level by colour, arm by dash."""
    levels = VARIANTS[variant]
    font = apply_style(PAGE_WIDTH)
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, ONE_AXES_HEIGHT))
    counts = []
    for arm, alabel, ls in ARMS:
        for perspective, level in levels:
            counts.append(draw_curve(ax, rows, perspective, level, arm,
                                     linestyle=ls,
                                     label=f"{level} · {alabel}"))
    style_curve_axes(ax, yscale)
    ax.set_xlabel("yahoo % of the 100-row probe")
    ax.set_ylabel("Procrustes disparity\n(0 = identical shape)")
    # Upper left, as figure 2's sweep panels do, and for the same reason: the
    # curves leave that corner free here, and the lower left is where the
    # functional pair runs.
    ax.legend(loc="upper left", frameon=False, fontsize=font)
    place_caption(fig, caption(variant, counts), font, ONE_AXES_HEIGHT)
    embolden(fig)
    return save_figure(fig, Path(outdir) /
                       f"fig_qmix_disparity_{variant}.{ext}")


# ── the geometry grid ────────────────────────────────────────────────────────

def geometry_index(geo_rows, perspective, arm):
    """``{pct: {kind: {model_id: (x, y)}}}`` for one panel row.

    The undiluted point is folded in from ``arm == "none"``, as in
    :func:`series`, so a row of panels runs the full width of the axis.
    """
    out = defaultdict(lambda: defaultdict(dict))
    for r in geo_rows:
        if r["perspective"] != perspective:
            continue
        if r["arm"] not in (arm, SHARED_ARM):
            continue
        kind = r["kind"] if r["kind"] != "seed_aligned" else "seed_aligned"
        pct = int(r["yahoo_pct"])
        key = (r["model_id"], int(r["seed"])) if kind == "seed_aligned" \
            else r["model_id"]
        out[pct][kind][key] = (float(r["dim1"]), float(r["dim2"]))
    return out


def grid_limits(per_level, pcts):
    """One square ``(lo, hi)`` covering every point the grid will draw.

    **Every panel gets the same limits, and this is not cosmetic.**  Each seed's
    fit was Procrustes-superimposed on its point's pooled fit upstream, and that
    superposition scales both configurations to unit Frobenius norm -- so every
    configuration in this grid is already on one scale.  Letting matplotlib
    autoscale each panel would undo that: a point whose recovered simplex is
    genuinely smaller would be blown up to fill its panel, and "compare the
    shapes along a row", which is the whole purpose of the grid, would be
    comparing eight different magnifications.

    Square rather than per-axis, because the panels are drawn at equal aspect and
    an MDS coordinate has no units -- the two axes are the same kind of thing and
    must not be scaled differently.
    """
    xs, ys = [], []
    for idx in per_level.values():
        for pct in pcts:
            for kind in idx.get(pct, {}).values():
                for x, y in kind.values():
                    xs.append(x)
                    ys.append(y)
    if not xs:
        return (-1.0, 1.0)
    half = max(max(map(abs, xs)), max(map(abs, ys))) * (1.0 + GEOMETRY_PAD)
    return (-half, half)


def draw_mds(ax, panel, ids, colours, frame, lim) -> None:
    """One geometry panel: the pooled fit, the mean locations, the seed cloud.

    Three objects, and they are three different things:

    * ``pooled`` -- the per-seed distance matrices averaged, then embedded once.
      Drawn as an unfilled ring.
    * ``mean`` -- each adapter's position averaged over the *aligned* per-seed
      fits.  Drawn filled.  It is not the same point as the pooled fit's, because
      MDS is not linear; both are carried rather than one chosen.
    * ``seed_aligned`` -- each seed's own fit, superimposed on the pooled one
      upstream.  Drawn faint, with a spoke to its adapter's mean, so what is left
      after alignment reads as a genuine disagreement between draws of the probe
      rather than as arbitrary orientation.

    At one extracted seed all three coincide exactly and the panel shows one
    marker per adapter, which is the honest picture of one measurement.
    """
    centre, up, right = frame
    pooled, mean = panel.get("pooled", {}), panel.get("mean", {})
    seeds = panel.get("seed_aligned", {})

    # The seed cloud and its spokes go down first: they are the texture of the
    # panel and must not sit on top of the markers they explain.
    for (model, _seed), (x, y) in seeds.items():
        if model in mean:
            mx, my = mean[model]
            ax.plot([mx, x], [my, y], "-", color=colours[model],
                    lw=SPOKE_LW, alpha=SPOKE_ALPHA, zorder=1)
    if seeds:
        ax.scatter([p[0] for p in seeds.values()],
                   [p[1] for p in seeds.values()],
                   c=[colours[m] for m, _ in seeds],
                   s=SEED_MARKER_SIZE, alpha=SEED_ALPHA, lw=0, zorder=2)

    for source, filled in (("pooled", False), ("mean", True)):
        pts = pooled if source == "pooled" else mean
        if not pts:
            continue
        for model, (x, y) in pts.items():
            is_frame = _mixture_of(model, ids) in (centre, up, right)
            shape = FRAME_MARKER_SHAPE if is_frame else MIXTURE_MARKER_SHAPE
            size = (MEAN_MARKER_SIZE if filled else POOLED_MARKER_SIZE)
            if is_frame:
                size *= FRAME_MARKER_SCALE
            ax.scatter([x], [y], marker=shape, s=size,
                       facecolors=colours[model] if filled else "none",
                       edgecolors=colours[model],
                       lw=0 if filled else POOLED_MARKER_LW,
                       zorder=4 if filled else 3)

    ax.set_aspect("equal")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([])
    ax.set_yticks([])


_LABELS: dict[str, str] = {}


def _mixture_of(model_id: str, ids) -> str:
    """The ``NN/NN/NN`` spelling of *model_id*'s mixture, memoised.

    Memoised because :func:`draw_mds` asks for it once per marker, and a grid of
    two levels by eight dilutions by sixteen adapters asks 256 times for sixteen
    distinct answers.
    """
    if model_id not in _LABELS:
        _LABELS[model_id] = mixture_label(model_id)
    return _LABELS[model_id]


def fig_mds(geo_rows, variant, arm, outdir, ext="pdf") -> Path | None:
    """Rows are levels, columns are yahoo percentages, for one arm."""
    levels = VARIANTS[variant]
    per_level = {p: geometry_index(geo_rows, p, arm) for p, _ in levels}
    pcts = sorted({p for idx in per_level.values() for p in idx})
    if not pcts:
        return None

    ids = sort_by_mixture(sorted({
        r["model_id"] for r in geo_rows
        if r["arm"] in (arm, SHARED_ARM) and r["kind"] == "mean"}))
    if not ids:
        return None
    colours = dict(zip(ids, model_colors(ids)))
    frame = frame_labels(n_groups(ids))

    lim = grid_limits(per_level, pcts)
    width = min(REFERENCE_WIDTH, 1.55 * len(pcts) + 1.0)
    font = apply_style(width)
    # The panels are square (equal aspect, equal limits), so the height follows
    # from the width rather than being chosen: anything else leaves the slack
    # that a grid of equal-aspect axes shows as a band of whitespace.
    cell = (width - 1.0) / len(pcts)
    height = cell * len(levels) + CAPTION_LINES * font / 72.0 + 0.45
    fig, axes = plt.subplots(len(levels), len(pcts),
                             figsize=(width, height), squeeze=False)
    n_seeds = set()
    for ri, (perspective, level) in enumerate(levels):
        for ci, pct in enumerate(pcts):
            ax = axes[ri][ci]
            panel = per_level[perspective].get(pct, {})
            if not panel:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        transform=ax.transAxes, fontsize=font * 0.6,
                        color="0.5")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                draw_mds(ax, panel, ids, colours, frame, lim)
                n_seeds.add(len({s for _, s in panel.get("seed_aligned", {})}))
            for spine in ax.spines.values():
                spine.set_color(AXIS_COLOR)
            # The undiluted column is the shared reference and belongs to
            # neither arm; the heavier spine stops the eye reading it as this
            # arm's last dilution step.
            if pct == SHARED_PCT and ci > 0:
                ax.spines["left"].set_linewidth(1.8)
                ax.spines["left"].set_color("0.2")
            if ri == 0:
                ax.set_title(f"{pct}%", fontsize=font)
            if ci == 0:
                ax.set_ylabel(level, color=LEVEL_COLORS[level], fontsize=font)

    alabel = dict((a, lab) for a, lab, _ in ARMS)[arm]
    seeds = sorted(n_seeds)
    seed_txt = (f"{seeds[0]}" if len(seeds) == 1
                else f"{seeds[0]}-{seeds[-1]}") if seeds else "0"
    place_caption(
        fig,
        f"Recovered geometry vs. query-set composition · diluted with "
        f"{alabel}\n"
        "columns are yahoo % of the 100-row probe · rings = pooled fit "
        "(distances averaged, then embedded) · filled = mean location over "
        "aligned per-seed fits\n"
        f"faint points = the {seed_txt} per-seed fit(s), Procrustes-aligned to "
        "the pooled one · triangles = the simplex frame (pure g1 north, "
        f"pure g2 east) · behavioral: {VARIANT_BEHAVIORAL[variant]}",
        font, height)
    embolden(fig)
    return save_figure(fig, Path(outdir) /
                       f"fig_qmix_mds_{arm}_{variant}.{ext}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "qmix_scores.csv"))
    ap.add_argument("--geometry", default=str(HERE / "qmix_geometry.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--variant", choices=["both"] + list(VARIANTS),
                    default="both",
                    help="draw one decoding variant instead of both")
    ap.add_argument("--figure", choices=["all", "main", "one-axes", "mds"],
                    default="all")
    ap.add_argument("--yscale", choices=["log", "linear"], default="log",
                    help="y range of the disparity curves; see YLIMS")
    # pdf is what goes in the paper; png is for looking at one on a terminal
    # that cannot open a pdf, and for a quick diff between two builds.
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    rows = read_rows(Path(args.scores))
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]

    written = []
    for variant in variants:
        present = {r["perspective"] for r in rows}
        wanted = {p for p, _ in VARIANTS[variant]}
        if not wanted & present:
            print(f"skipping {variant}: none of {sorted(wanted)} is in "
                  f"{args.scores} (found {sorted(present)})")
            continue
        if args.figure in ("all", "main"):
            written.append(fig_disparity_arms(rows, variant, args.yscale,
                                              args.outdir, args.ext))
        if args.figure in ("all", "one-axes"):
            written.append(fig_disparity_one_axes(
                rows, variant, args.yscale, args.outdir, args.ext))

    if args.figure in ("all", "mds"):
        geo_rows = read_rows(Path(args.geometry))
        for variant in variants:
            for arm, _lab, _ls in ARMS:
                out = fig_mds(geo_rows, variant, arm, args.outdir,
                              args.ext)
                if out is None:
                    print(f"skipping mds {arm}/{variant}: no coordinates")
                else:
                    written.append(out)

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
