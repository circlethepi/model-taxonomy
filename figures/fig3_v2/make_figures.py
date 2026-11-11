#!/usr/bin/env python
"""Figure 3 v2: recovery of the simplex against the probe, from the re-runs.

**qbig_v2** -- the standing sixteen OLMo-2-1B yahoo-simplex adapters, probed
once at 2000 uniformly drawn yahoo queries and R=64 sampled replicates per
query at T=1.  **qbig_v2_temp** -- the same probe decoded at T=0.1, ..., 0.9
and scored at R=16 (replicates 0-15).  **qmix_v2** -- query mixture: the yahoo
share of the probe varied, the rest a diluting corpus (an **arm**: dolly or
oasst1-cjk), every cell subsampled from one decoded pool per corpus.  **R** --
the number of sampled generations per query averaged into the query's
embedding.  See ``docs/terminology.md``.

This file is purely plotting.  Every number is read from CSVs copied into this
directory unchanged:

* ``qbig_v2_subsample.csv``      from ``simplex3_qbig_v2_olmo2/subsample_qbig_v2.py``
* ``qbig_v2_temp_subsample.csv`` from ``simplex3_qbig_v2_olmo2/subsample_qbig_v2_temp.py``
* ``qmix_v2_subsample.csv``      from ``simplex3_qmix_v2_olmo2/subsample_qmix_v2.py``
  (scoring job 1473695; collapsed trials are NaN)
* ``qbig_v2_replicates.csv``     from ``simplex3_qbig_v2_olmo2/subsample_qbig_v2_replicates.py``,
  a replicate-axis scoring of the qbig_v2 R=64 decode

All four are in place.  Should one go missing, its panel is drawn as a
**placeholder**: its real axes, title and key, no curves, and a grey "data
pending" note.  Once the file is copied back in, the panel fills itself.

Two variants, which differ only in how query mixture is shown
---------------------------------------------------------------
``qmix_dilution`` (1 x 4), v1's layout:

1. **Functional layers** -- thirds (early 1–5, middle 6–10, late 11–16) solid,
   single layers h1/h9/h16 dotted in their third's colour, all states
   (h0–h16) black with triangles; x = queries.
2. **Query mixture** -- sampled behavioral (T=1, R=16) and functional (all
   states), level by colour, corpus by dash; x = dilution % (100 - yahoo %),
   symlog in yahoo %, 100 on the left.  Read at dolly n=1000 and oasst1-cjk
   n=500 (the CJK pool's largest subsampled n); the undiluted 100%-yahoo
   point, shared by both arms, is at n=1000.
3. **Replicates** -- sampled behavioral at T=1, one line per R.
4. **Temperature** -- sampled behavioral at R=16, one line per T (no greedy).

``qmix_queries`` (1 x 5): panel 2 replaced by two panels on the queries axis,
one per level (sampled behavioral, functional all states); one line per
dilution %, coloured on a ramp in the level's colour; corpus by dash.

**Single-draw points.**  The full 2000-query probe (and, in qmix_v2, the full
859-row CJK pool) is scored once, so it has no band: it is drawn as a hollow
marker in its curve's colour, and the line is not extended to it.

Line = median; band = interquartile range over query draws (figure 2's
convention).  There is no figure title: the provenance belongs in the caption.

Usage::

    python figures/fig3_v2/make_figures.py
    python figures/fig3_v2/make_figures.py --ext png --yscale log --variant qmix_queries
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figures" / "simplex3_qbig_olmo2"))
sys.path.insert(0, str(REPO / "figures" / "simplex3_qmix_olmo2"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    LinearSegmentedColormap, ListedColormap, Normalize,
)

import make_qbig_figures as mq  # noqa: E402
import make_qbig_layer_figures as ml  # noqa: E402
import make_qmix_figures as mx  # noqa: E402
from src.plots import save_figure  # noqa: E402
from src.plots.sweep_figures import (  # noqa: E402
    apply_style, embolden, quantile_band,
)

HERE = Path(__file__).resolve().parent

# ── colours, built from figure 2's palette (unchanged from v1) ───────────────

#: Panel 1: one colour per third (early, middle, late) in the functional gold
#: family; single layers take their third's colour; ``all`` is black.
THIRD_COLORS = ("#ECA81E", "#BC7A0C", "#76500A")
REFERENCE_COLOR = ml.REFERENCE_COLOR
LAYER_LINE_ALPHA = 0.8
REFERENCE_MARKER = "^"
SINGLE_LS = (0, (1, 1))

#: Figure 2's level colours: functional gold, behavioral red.
LEVEL_COLORS = {"behavioral": "#D55E00", "functional": "#E69F00"}

#: Panel 3: light to dark through behavioral red, pinned so R=16 is the red.
R_RAMP = ["#F6C7A8", "#D55E00", "#9E3A00"]
R_ANCHOR = 16
#: Panel 4: cool blue-grey at T=0 (not drawn) to behavioral red at T=1, sampled
#: at each temperature's own value.
T_RAMP = ["#5A7598", "#D55E00"]

#: qmix_queries panels: one ramp per level over dilution %, light (undiluted)
#: to dark (fully diluted), through the level's figure 2 colour.
DILUTION_RAMPS = {"behavioral": ["#F6C7A8", "#D55E00", "#6E2800"],
                  "functional": ["#F5DB9A", "#E69F00", "#5E4008"]}

# ── content ──────────────────────────────────────────────────────────────────

FIXED_T = 1.0
FIXED_R = 16

#: Panel 1's reads from ``qbig_v2_subsample.csv``: ``(perspective, kind,
#: depth index or None)``.  Depth index 0/1/2 = early/middle/late.
LAYER_READS = [("layers_1_5", "group", 0), ("layers_6_10", "group", 1),
               ("layers_11_16", "group", 2),
               ("h1", "single", 0), ("h9", "single", 1), ("h16", "single", 2)]
REFERENCE_PERSPECTIVE = "all_states"
DEPTH_NAMES = ("early", "middle", "late")

#: qmix_v2's two levels: ``(level, perspective, legend name)``.
QMIX_LEVELS = [("behavioral", "sampled", "behavioral"),
               ("functional", "all_states", "functional")]
#: qmix_v2's diluting corpora, in key order, with their dash.
ARMS = [("oasst1cjk", "oasst1-cjk", "--"), ("dolly", "dolly", "-")]
ARM_LINESTYLES = {a: ls for a, _l, ls in ARMS}
#: The undiluted (100% yahoo) level, scored once and shared by both arms.
SHARED_ARM = "yahoo"
#: qmix_dilution: the query count each arm is read at.  CJK's subsampled grid
#: stops at 500; the shared 100%-yahoo point is read at dolly's n.
QMIX_N = {"dolly": 1000, "oasst1cjk": 500, SHARED_ARM: 1000}
#: qmix_v2's yahoo percentages (``subsample_qmix_v2.YAHOO_PCTS`` plus 100).
YAHOO_PCTS = [0, 1, 2, 5, 10, 20, 50, 100]
KEY_COLOR = "0.35"

#: Panel 4's colorbar labels only these temperatures (every cell is drawn).
T_TICK_LABELS = {0.1, 0.5, 1.0}
#: qmix_queries' dilution colorbar labels only these dilution %.
DILUTION_TICK_LABELS = {0, 80, 95, 100}

BAND_Q = (25.0, 75.0)
BAND_ALPHA = 0.25
LINEWIDTH = 2.1 * 7.0 / 13.0
MARKER_SIZE = 2.6
#: The hollow marker for a single-draw point (no band): a little larger than
#: the line markers so its open face reads.
SINGLE_DRAW_MARKER_SIZE = 3.6

#: The query axis now reaches the full 2000-query probe; the margins keep
#: v1's fraction of the axis width beyond the first and last points.
QUERY_DATA = (1, 2000)
_lo, _hi = np.log10((0.7, 1450)), np.log10((1, 1000))
_f = (_hi[0] - _lo[0]) / (_lo[1] - _lo[0]), (_lo[1] - _hi[1]) / (_lo[1] - _lo[0])
_span = np.log10(QUERY_DATA[1]) / (1 - _f[0] - _f[1])
QUERY_XLIM = (10 ** (-_f[0] * _span), 10 ** (np.log10(QUERY_DATA[1]) + _f[1] * _span))
QUERY_TICKS = [1, 10, 100, 1000]

QMIX_TICKS = [0, 1, 10, 100]
QMIX_LINSCALE = 0.5

EMBEDDER = "v2moe"
YLABEL = "Procrustes Disparity"
PENDING_TEXT = "data pending"

PAGE_WIDTH = 7.5
HEIGHT = {"log": 2.1, "linear": 2.4}
WSPACE_SCALE = 0.65
#: Legend placement per keyed panel (panels without an entry are keyed by an
#: inset colorbar, or not at all).
LEGEND_LOC = {"layers": "upper left", "layers_single": "upper left",
              "qmix_dilution": "upper center",
              "qmix_behavioral": "upper right", "qmix_functional": "upper right"}
HANDLE_LENGTH = {"layers": 1.6, "layers_single": 1.6, "qmix_dilution": 1.0,
                 "qmix_behavioral": 1.6, "qmix_functional": 1.6}
COLORBAR_BOX = {"log": [0.07, 0.18, 0.86, 0.05],
                "linear": [0.22, 0.84, 0.74, 0.05]}

VARIANTS = {
    "qmix_dilution": ["layers", "qmix_dilution", "replicates", "temperature"],
    "qmix_queries": ["layers", "qmix_behavioral", "qmix_functional",
                     "replicates", "temperature"],
    "qmix_dolly": ["layers_single", "qmix_dolly_both", "replicates",
                   "temperature_subset"],
    "qmix_oasst1cjk": ["layers_single", "qmix_oasst1cjk_both", "replicates",
                       "temperature_subset"],
}

#: qmix_dolly's merged panel: the two-row dilution bar's box per y scale.  On
#: the log scale the functional lines sit where the bottom box would be, so
#: the bar moves to the empty upper right.
STACKED_BAR_BOX = {"log": [0.42, 0.80, 0.53, 0.07],
                   "linear": [0.22, 0.80, 0.74, 0.07]}
#: qmix_oasst1cjk's log-scale bar: its behavioral lines sit higher at large n
#: than dolly's and cross the upper-right box, so the bar is thinner and
#: raised to clear them.
CJK_STACKED_BAR_BOX = {"log": [0.445, 0.835, 0.53, 0.05],
                       "linear": STACKED_BAR_BOX["linear"]}

#: qmix_dolly: the dolly arm only, at these yahoo % (dilution 100/99/90/50/0).
DOLLY_YAHOO_PCTS = [0, 1, 10, 50, 100]
#: qmix_oasst1cjk: as qmix_dolly, with the oasst1-cjk arm in place of dolly.
#: Same x axis; the CJK curves stop at its grid's n=500, plus the single-draw
#: full-pool point at 859.
#: qmix_dolly: the temperatures drawn (no greedy).
SUBSET_TS = [0.2, 0.4, 0.6, 0.8, 1.0]


# ── data ─────────────────────────────────────────────────────────────────────

def read_rows(path: Path) -> list[dict] | None:
    """The CSV's rows, or ``None`` if it has not been copied in yet."""
    if not path.exists():
        print(f"  {path.name}: not found -- placeholder")
        return None
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def multi(rows):
    """Rows from a curve's banded trials (``n_query_draws > 1``)."""
    return [r for r in rows if int(r["n_query_draws"]) > 1]


def single(rows):
    """Rows scored once (the full pool): drawn as hollow markers."""
    return [r for r in rows if int(r["n_query_draws"]) == 1]


def by_x(rows, x="n_queries", column="disparity"):
    """Trials by *x*.  A collapsed trial (all sixteen adapters coincide,
    scored NaN) is left out, so a median rests only on the trials that
    scored."""
    out = defaultdict(list)
    for r in rows:
        v = float(r[column])
        if not np.isnan(v):
            out[int(r[x])].append(v)
    return out


def pick(rows, **want):
    return [r for r in rows if all(r[k] == v for k, v in want.items())]


def sampled_at(rows, temperature=None, replicates=None):
    out = []
    for r in pick(rows, level="behavioral", perspective="sampled"):
        if temperature is not None and float(r["temperature"]) != temperature:
            continue
        if replicates is not None and int(r["replicates"]) != replicates:
            continue
        out.append(r)
    return out


# ── drawing ──────────────────────────────────────────────────────────────────

def ramp(colours, n):
    cmap = LinearSegmentedColormap.from_list("ramp", colours)
    return list(cmap(np.linspace(0.0, 1.0, n)))


def draw(ax, rows, colour, x="n_queries", ls="-", marker="o", zorder=2,
         alpha=1.0):
    """A banded median line over *rows*' multi-draw trials, and a hollow
    marker at each single-draw point (the line is not extended to it)."""
    xs, lo, mid, hi, _n = quantile_band(by_x(multi(rows), x), BAND_Q)
    if xs:
        if any(h > l for l, h in zip(lo, hi)):
            ax.fill_between(xs, lo, hi, color=colour, alpha=BAND_ALPHA, lw=0)
        ax.plot(xs, mid, linestyle=ls, color=colour, lw=LINEWIDTH,
                marker=marker, ms=MARKER_SIZE, zorder=zorder, alpha=alpha)
    for px, vals in by_x(single(rows), x).items():
        ax.plot([px], [float(np.median(vals))], linestyle="none",
                marker=marker, ms=SINGLE_DRAW_MARKER_SIZE, mfc="none",
                mec=colour, mew=LINEWIDTH * 0.6, zorder=zorder, alpha=alpha)


def pending(ax):
    ax.text(0.5, 0.5, PENDING_TEXT, transform=ax.transAxes, ha="center",
            va="center", color="0.55", style="italic")


def query_axes(ax, yscale):
    ax.set_xscale("log")
    ax.set_xticks(QUERY_TICKS)
    ax.set_xticklabels([str(t) for t in QUERY_TICKS])
    ax.set_xlim(*QUERY_XLIM)
    ax.set_yscale(yscale)
    ax.set_ylim(*mq.DISPARITY_YLIMS[yscale])
    ax.axvline(mq.STANDING_QUERIES, color="0.55", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel("Queries")


def matched_xlim(ax, lo, hi):
    """x limits giving *ax* the query panels' margins around ``[lo, hi]``."""
    (a, b), (c, d) = np.log10(QUERY_XLIM), np.log10(QUERY_DATA)
    f_lo, f_hi = (c - a) / (b - a), (b - d) / (b - a)
    t = ax.xaxis.get_transform()
    tlo, thi = t.transform([lo, hi])
    span = (thi - tlo) / (1 - f_lo - f_hi)
    return tuple(t.inverted().transform([tlo - f_lo * span,
                                         thi + f_hi * span]))


def inset_colorbar(ax, colours, labels, title, yscale):
    """A discrete horizontal colorbar inside *ax*: one cell per value; an
    empty label leaves that cell unticked."""
    cax = ax.inset_axes(COLORBAR_BOX[yscale])
    n = len(colours)
    cb = ax.figure.colorbar(
        ScalarMappable(Normalize(-0.5, n - 0.5), ListedColormap(colours)),
        cax=cax, orientation="horizontal")
    ticks = [i for i, lab in enumerate(labels) if lab]
    cb.set_ticks(ticks)
    cb.set_ticklabels([labels[i] for i in ticks])
    cb.ax.tick_params(length=2, pad=1)
    cb.outline.set_edgecolor(mq.GREEDY_COLOR)
    cax.set_title(title, pad=2)
    return cax


def inset_stacked_bar(ax, rows, labels, title, box):
    """A discrete two-row bar inside *ax*: ``rows`` is a list of colour lists
    (one per row, top first), each row half the bar's thickness; *labels*
    tick the columns, below the bottom row only."""
    cax = ax.inset_axes(box)
    n = len(labels)
    grid = np.array([[matplotlib.colors.to_rgba(c) for c in row]
                     for row in rows])
    cax.imshow(grid, aspect="auto", extent=(-0.5, n - 0.5, len(rows), 0),
               interpolation="nearest")
    for y in range(1, len(rows)):
        cax.axhline(y, color=mq.GREEDY_COLOR, lw=0.5)
    cax.set_xticks(range(n))
    cax.set_xticklabels(labels)
    cax.set_yticks([])
    cax.grid(False)
    cax.tick_params(length=2, pad=1)
    for s in cax.spines.values():
        s.set_edgecolor(mq.GREEDY_COLOR)
    cax.set_title(title, pad=2)
    return cax


def colorbar_frame(ax, cax, renderer):
    """A legend's frame behind *cax*, its title and its ticks (v1's)."""
    rc = matplotlib.rcParams
    pad = rc["legend.borderpad"] * rc["legend.fontsize"] / 72 * ax.figure.dpi
    box = cax.get_tightbbox(renderer).padded(pad)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(box.get_points())
    edge = rc["legend.edgecolor"]
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.02",
        transform=ax.transAxes, facecolor=ax.get_facecolor(),
        edgecolor=rc["axes.edgecolor"] if edge == "inherit" else edge,
        alpha=rc["legend.framealpha"], lw=rc["patch.linewidth"],
        zorder=4.5, clip_on=False))


# ── panels ───────────────────────────────────────────────────────────────────

def panel_layers(ax, data, yscale):
    rows = data["subsample"]
    query_axes(ax, yscale)
    ax.set_title("Functional layers")
    if rows is None:
        pending(ax)
    else:
        func = [r for r in rows if r["level"] == "functional"]
        draw(ax, pick(func, perspective=REFERENCE_PERSPECTIVE),
             REFERENCE_COLOR, marker=REFERENCE_MARKER, zorder=3,
             alpha=LAYER_LINE_ALPHA)
        for persp, kind, depth in LAYER_READS:
            draw(ax, pick(func, perspective=persp), THIRD_COLORS[depth],
                 ls="-" if kind == "group" else SINGLE_LS,
                 alpha=LAYER_LINE_ALPHA)
    for colour, name in zip(THIRD_COLORS, DEPTH_NAMES):
        ax.plot([], [], "-", color=colour, lw=LINEWIDTH, marker="o",
                ms=MARKER_SIZE, label=name, alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], "-", color=KEY_COLOR, lw=LINEWIDTH, label="group",
            alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], linestyle=SINGLE_LS, color=KEY_COLOR, lw=LINEWIDTH,
            label="single", alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], "-", color=REFERENCE_COLOR, lw=LINEWIDTH,
            marker=REFERENCE_MARKER, ms=MARKER_SIZE, label="all",
            alpha=LAYER_LINE_ALPHA)


def panel_layers_single(ax, data, yscale):
    """Panel 1 without the group (third) lines: single layers h1/h9/h16,
    dotted in their third's colour, and all states."""
    rows = data["subsample"]
    query_axes(ax, yscale)
    ax.set_title("Functional layers")
    if rows is None:
        pending(ax)
    else:
        func = [r for r in rows if r["level"] == "functional"]
        draw(ax, pick(func, perspective=REFERENCE_PERSPECTIVE),
             REFERENCE_COLOR, marker=REFERENCE_MARKER, zorder=3,
             alpha=LAYER_LINE_ALPHA)
        for persp, kind, depth in LAYER_READS:
            if kind == "single":
                draw(ax, pick(func, perspective=persp), THIRD_COLORS[depth],
                     ls=SINGLE_LS, alpha=LAYER_LINE_ALPHA)
    for colour, name in zip(THIRD_COLORS, DEPTH_NAMES):
        ax.plot([], [], linestyle=SINGLE_LS, color=colour, lw=LINEWIDTH,
                marker="o", ms=MARKER_SIZE, label=name,
                alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], "-", color=REFERENCE_COLOR, lw=LINEWIDTH,
            marker=REFERENCE_MARKER, ms=MARKER_SIZE, label="all",
            alpha=LAYER_LINE_ALPHA)


def panel_qmix_both(ax, data, yscale, arm="dolly",
                    yahoo_pcts=DOLLY_YAHOO_PCTS, bar_box=STACKED_BAR_BOX):
    """Sampled behavioral (T=1, R=16) and functional (all states) on one
    axes, one *arm*; dilution on each level's ramp, keyed by a two-row bar
    (behavioral on top, functional below)."""
    rows = data["qmix"]
    pcts = sorted(yahoo_pcts, reverse=True)            # undiluted first
    query_axes(ax, yscale)
    ax.set_title(f"Query mixture\n(T={FIXED_T:g}, R={FIXED_R})")
    bar_rows = []
    for level, persp, _name in QMIX_LEVELS:
        colours = dict(zip(pcts, ramp(DILUTION_RAMPS[level], len(pcts))))
        bar_rows.append([colours[p] for p in pcts])
        if rows is None:
            continue
        lvl = pick(rows, level=level, perspective=persp)
        draw(ax, pick(lvl, arm=SHARED_ARM), colours[100])
        for pct in pcts[1:]:
            draw(ax, [r for r in pick(lvl, arm=arm)
                      if int(r["yahoo_pct"]) == pct], colours[pct])
    if rows is None:
        pending(ax)
    inset_stacked_bar(ax, bar_rows, [str(100 - p) for p in pcts],
                      "Dilution %", bar_box[yscale])


def panel_qmix_dilution(ax, data, yscale):
    rows = data["qmix"]
    if rows is None:
        pending(ax)
    else:
        for level, persp, _name in QMIX_LEVELS:
            for arm, _l, ls in ARMS:
                cell = [r for r in pick(rows, level=level, perspective=persp)
                        if r["arm"] in (arm, SHARED_ARM)
                        and int(r["n_queries"]) == QMIX_N[r["arm"]]]
                draw(ax, cell, LEVEL_COLORS[level], x="yahoo_pct", ls=ls)
    for level, _p, name in QMIX_LEVELS:
        ax.plot([], [], "-", color=LEVEL_COLORS[level], lw=LINEWIDTH,
                marker="o", ms=MARKER_SIZE, label=name)
    for arm, label, ls in ARMS:
        ax.plot([], [], ls, color=KEY_COLOR, lw=LINEWIDTH, label=label)
    ax.set_yscale(yscale)
    ax.set_xscale("symlog", linthresh=mx.SYMLOG_THRESHOLD,
                  linscale=QMIX_LINSCALE)
    ax.set_xlim(*matched_xlim(ax, 0, 100))
    ax.set_ylim(*mq.DISPARITY_YLIMS[yscale])
    ax.set_xticks(QMIX_TICKS)
    ax.set_xticklabels([str(100 - t) for t in QMIX_TICKS])
    ax.set_xlabel("Dilution %")
    ax.set_title(f"Query mixture\n(M={QMIX_N['dolly']}/{QMIX_N['oasst1cjk']}, "
                 f"T={FIXED_T:g}, R={FIXED_R})")


def panel_qmix_queries(ax, data, yscale, level, arms=ARMS,
                       yahoo_pcts=YAHOO_PCTS, tick_labels=DILUTION_TICK_LABELS):
    rows = data["qmix"]
    persp, name = next((p, n) for lv, p, n in QMIX_LEVELS if lv == level)
    pcts = sorted(yahoo_pcts, reverse=True)            # undiluted first
    colours = dict(zip(pcts, ramp(DILUTION_RAMPS[level], len(pcts))))
    query_axes(ax, yscale)
    if rows is None:
        pending(ax)
    else:
        lvl = pick(rows, level=level, perspective=persp)
        draw(ax, pick(lvl, arm=SHARED_ARM), colours[100])
        for arm, _l, ls in arms:
            for pct in pcts[1:]:
                draw(ax, [r for r in pick(lvl, arm=arm)
                          if int(r["yahoo_pct"]) == pct], colours[pct], ls=ls)
    if len(arms) > 1:
        for arm, label, ls in arms:
            ax.plot([], [], ls, color=KEY_COLOR, lw=LINEWIDTH, label=label)
    labels = [str(100 - p) if 100 - p in tick_labels else ""
              for p in pcts]
    inset_colorbar(ax, [colours[p] for p in pcts], labels, "Dilution %",
                   yscale)
    detail = f"T={FIXED_T:g}, R={FIXED_R}" if level == "behavioral" \
        else "all states"
    ax.set_title(f"Query mixture\n({name}, {detail})")


def panel_replicates(ax, data, yscale):
    rows = data["replicates"]
    query_axes(ax, yscale)
    ax.set_title(f"Replicates\n(T={FIXED_T:g})")
    if rows is None:
        pending(ax)
        return
    sampled = sampled_at(rows, temperature=FIXED_T)
    rs = sorted({int(r["replicates"]) for r in sampled})
    at = rs.index(R_ANCHOR) / (len(rs) - 1) if R_ANCHOR in rs else 0.5
    cmap = LinearSegmentedColormap.from_list(
        "R", list(zip([0.0, at, 1.0], R_RAMP)))
    colours = [cmap(i / max(len(rs) - 1, 1)) for i in range(len(rs))]
    for r, colour in zip(rs, colours):
        draw(ax, [t for t in sampled if int(t["replicates"]) == r], colour)
    inset_colorbar(ax, colours, [str(r) for r in rs], "Replicates (R)",
                   yscale)


def panel_temperature(ax, data, yscale):
    rows = data["temp"]
    query_axes(ax, yscale)
    ax.set_title(f"Temperature\n(R={FIXED_R})")
    if rows is None:
        pending(ax)
        return
    sampled = sampled_at(rows, replicates=FIXED_R)
    ts = sorted({float(r["temperature"]) for r in sampled})
    cmap = LinearSegmentedColormap.from_list("T", T_RAMP)
    colours = [cmap(t) for t in ts]
    for t, colour in zip(ts, colours):
        draw(ax, [r for r in sampled if float(r["temperature"]) == t], colour)
    labels = [f"{t:g}" if t in T_TICK_LABELS else "" for t in ts]
    inset_colorbar(ax, colours, labels, "Temperature (T)", yscale)


def panel_temperature_subset(ax, data, yscale):
    """Sampled behavioral at R=16 for ``SUBSET_TS`` only (no greedy)."""
    rows = data["temp"]
    query_axes(ax, yscale)
    ax.set_title(f"Temperature\n(R={FIXED_R})")
    if rows is None:
        pending(ax)
        return
    cmap = LinearSegmentedColormap.from_list("T", T_RAMP)
    sampled = sampled_at(rows, replicates=FIXED_R)
    colours = [cmap(t) for t in SUBSET_TS]
    for t, colour in zip(SUBSET_TS, colours):
        draw(ax, [r for r in sampled if float(r["temperature"]) == t], colour)
    labels = [f"{t:g}" for t in SUBSET_TS]
    inset_colorbar(ax, colours, labels, "Temperature (T)", yscale)


DOLLY_ARMS = [a for a in ARMS if a[0] == "dolly"]

PANELS = {
    "layers": panel_layers,
    "layers_single": panel_layers_single,
    "qmix_dolly_both": panel_qmix_both,
    "qmix_oasst1cjk_both": lambda ax, d, y: panel_qmix_both(
        ax, d, y, arm="oasst1cjk", bar_box=CJK_STACKED_BAR_BOX),
    "qmix_dilution": panel_qmix_dilution,
    "qmix_behavioral": lambda ax, d, y: panel_qmix_queries(ax, d, y, "behavioral"),
    "qmix_functional": lambda ax, d, y: panel_qmix_queries(ax, d, y, "functional"),
    "qmix_behavioral_dolly": lambda ax, d, y: panel_qmix_queries(
        ax, d, y, "behavioral", DOLLY_ARMS, DOLLY_YAHOO_PCTS,
        {100 - p for p in DOLLY_YAHOO_PCTS}),
    "qmix_functional_dolly": lambda ax, d, y: panel_qmix_queries(
        ax, d, y, "functional", DOLLY_ARMS, DOLLY_YAHOO_PCTS,
        {100 - p for p in DOLLY_YAHOO_PCTS}),
    "replicates": panel_replicates,
    "temperature": panel_temperature,
    "temperature_subset": panel_temperature_subset,
}


def build(data, variant, outdir, ext, yscale) -> Path:
    apply_style(PAGE_WIDTH)
    keys = VARIANTS[variant]
    fig, axes = plt.subplots(1, len(keys), figsize=(PAGE_WIDTH, HEIGHT[yscale]),
                             sharey=True)
    for ax, key in zip(axes, keys):
        PANELS[key](ax, data, yscale)
    axes[0].set_ylabel(YLABEL)
    for ax, key in zip(axes, keys):
        if key not in LEGEND_LOC:
            continue
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=LEGEND_LOC[key],
                  ncol=2 if len(handles) > 2 else 1,
                  handlelength=HANDLE_LENGTH[key], handletextpad=0.3,
                  columnspacing=0.5, borderaxespad=0.3).set_in_layout(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=fig.subplotpars.wspace * WSPACE_SCALE)
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        for cax in ax.child_axes:
            colorbar_frame(ax, cax, renderer)
    embolden(fig)
    return save_figure(fig, Path(outdir) /
                       f"fig_figure3_{variant}{mq.YSCALE_SUFFIX[yscale]}.{ext}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsample", default=str(HERE / "qbig_v2_subsample.csv"))
    ap.add_argument("--temp", default=str(HERE / "qbig_v2_temp_subsample.csv"))
    ap.add_argument("--qmix", default=str(HERE / "qmix_v2_subsample.csv"))
    ap.add_argument("--replicates", default=str(HERE / "qbig_v2_replicates.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--variant", default="both", choices=["both"] + list(VARIANTS))
    ap.add_argument("--yscale", default="both",
                    choices=["both"] + list(mq.DISPARITY_YLIMS))
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    data = {k: read_rows(Path(getattr(args, k)))
            for k in ("subsample", "temp", "qmix", "replicates")}
    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    scales = (list(mq.DISPARITY_YLIMS) if args.yscale == "both"
              else [args.yscale])
    for variant in variants:
        for yscale in scales:
            path = build(data, variant, args.outdir, args.ext, yscale)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
