#!/usr/bin/env python
"""Figure 3: recovery of the simplex against the probe -- layers, composition,
replicates, temperature.

**qbig** (*big query set*) -- the standing sixteen OLMo-2-1B yahoo-simplex
adapters, probed once at 1000 queries and R=64 sampled replicates per query.
**qmix** (*query mixture*) -- the same sixteen adapters, probed with 100-row
query sets in which a fraction of the yahoo rows is replaced by a diluting
corpus (an **arm**: dolly or oasst1-zh).  **R** -- the number of sampled
generations per query averaged into the query's embedding.  See
``docs/terminology.md``.

This file is purely plotting.  Every number is read from three CSVs copied into
this directory unchanged:

* ``qbig_layers.csv``    from ``simplex3_qbig_olmo2/layers_qbig.py``
* ``qbig_subsample.csv`` from ``simplex3_qbig_olmo2/subsample_qbig.py``
* ``qmix_scores.csv``    from ``simplex3_qmix_olmo2/sweep_qmix.py``

The four panels, one row, y = Procrustes disparity on one shared axis
----------------------------------------------------------------------
1. **Functional layers** -- the contiguous layer subsets (thirds: early 1–5,
   middle 6–10, late 11–16) solid, the single layers h1, h9 and h16 dotted in
   their third's colour, and all of h1–h16 as the reference (black,
   triangles); x = number of queries.
2. **Query mixture** -- sampled behavioral (T=1, R=16) and functional (all
   hidden states), 100-row probe, level by colour (figure 2's), corpus by
   dash (dolly solid, oasst1-zh dashed);
   x = dilution % (100 - yahoo %), symlog in yahoo %, 100 on the left.  The 0% point is
   shared by both arms.
3. **Replicates** -- sampled behavioral at T=1, one line per R; x = queries.
4. **Temperature** -- sampled behavioral at R=16, one line per T; x = queries.

Line = median; band = interquartile range over draws (figure 2's convention).
There is no figure title: the provenance belongs in the paper's caption.

Usage::

    python figures/fig3_v1/make_figures.py
    python figures/fig3_v1/make_figures.py --ext png --yscale log
"""

from __future__ import annotations

import argparse
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

# ── colours, built from figure 2's palette ───────────────────────────────────
# figure 2: Functional gold, Behavioral red, models a blue ramp, corpora greens.

#: Panel 1: depth, light to dark, through figure 2's functional gold.  The dark
#: end stays ochre rather than brown, so it is not read as panel 4's red.
DEPTH_RAMP = ["#EDB84A", "#E69F00", "#7A5400"]
REFERENCE_COLOR = ml.REFERENCE_COLOR
#: Panel 1's curves (and their key entries) are drawn at this opacity, so the
#: overlapping layer curves show through one another.
LAYER_LINE_ALPHA = 0.8
#: ``all (1–16)`` is marked with a triangle, apart from the layer curves' dots.
REFERENCE_MARKER = "^"
#: Single layers dotted, so they do not read as subsets on the shared ramp.
#: Tighter than matplotlib's ":" (1 on, 1.65 off), so the short legend handle
#: still shows several dots either side of the marker.
SINGLE_LS = (0, (1, 1))
SINGLE_LAYERS = [1, 9, 16]

#: Panel 2: figure 2's level colours, functional gold and behavioral red; the
#: diluting corpus is told apart by linestyle (``ARM_LINESTYLES``).
LEVEL_COLORS = {"behavioral": "#D55E00", "functional": "#E69F00"}

#: Panel 3: light to dark through figure 2's behavioral red, over the R
#: values, with the red pinned at ``R_ANCHOR`` (the standing R=16, which panel
#: 4 holds fixed).  The dark end stops at a deep red-orange, short of the
#: brown that would read as panel 1's.
R_RAMP = ["#F6C7A8", "#D55E00", "#9E3A00"]
R_ANCHOR = 16
#: Panel 4: a cool blue-grey at T=0 (greedy, which is not drawn) to figure
#: 2's behavioral red at T=1, sampled at each temperature's own value -- so
#: T=0.25 is a greyish red a quarter of the way along.  The grey is cool
#: because a neutral one (``#404040``) mixes with the red into brown.
#: (Earlier tries: blue ``#3B5BDB`` or ``#2171B5``, purple ``#8E4FA0`` midway.)
T_RAMP = ["#5A7598", "#D55E00"]

# ── content ──────────────────────────────────────────────────────────────────

#: Panel 3 holds T fixed; panel 4 holds R fixed.  Panel 2 is qmix, which was
#: decoded only at T=1, R=16, so these match it.
FIXED_T = 1.0
FIXED_R = 16
#: ``(perspective in qmix_scores.csv, legend name)``: figure 2's pinned
#: perspectives for the two levels that read the query set.
QMIX_LEVELS = [("behavioral", "behavioral"),
               ("functional_all", "functional")]
#: Panel 2's linestyle per diluting corpus.
ARM_LINESTYLES = {"dolly": "-", "oasst1zh": "--"}
#: The order the corpora are keyed in panel 2's legend.
KEY_ARM_ORDER = ["oasst1zh", "dolly"]
#: The neutral colour of the line-kind entries in panels 1 and 2's keys
#: (group / single; dolly / oasst1-zh).
KEY_COLOR = "0.35"
#: Short arm names for the in-axes legend.
ARM_LABELS = {"dolly": "dolly", "oasst1zh": "oasst1-zh"}

#: Figure 2's band: the interquartile range.
BAND_Q = (25.0, 75.0)
#: Band opacity: raised from figure 2's 0.15 so the IQR reads at this size.
BAND_ALPHA = 0.25
#: Figure 2's sweep line width at the page width: its ``SWEEP_LINEWIDTH``
#: (2.1) scaled by the text ratio, 7 pt over its 13 pt reference.
LINEWIDTH = 2.1 * 7.0 / 13.0
MARKER_SIZE = 2.6

#: Panel 2 is plotted at yahoo % on qmix's symlog axis and labelled as
#: dilution % (= 100 - yahoo %), so the fully diluted probe sits on the left.
#: Only these yahoo % are labelled, so the labels do not run together.
QMIX_TICKS = [0, 1, 10, 100]
#: Width of symlog's linear 0-1% yahoo stretch, in decades (matplotlib's
#: default, and qmix's, is 1): narrowed so 100% dilution takes less room.
QMIX_LINSCALE = 0.5
#: The query panels' x limits (``mq.style_axes``) leave this fraction of the
#: axis width beyond the first and last points; panel 2 is padded to match.
QUERY_XLIM = (0.7, 1450)
QUERY_DATA = (1, 1000)

EMBEDDER = "v2moe"
#: Figure 2's y label.
YLABEL = "Procrustes Disparity"

PAGE_WIDTH = 7.5
#: Figure height per y scale.  Linear runs taller: its colorbar keys sit at
#: the top, where the behavioral curves start; 2.4 in is the shortest that
#: keeps panel 3's box off the R=2 curve and its markers (2.1 touches it).
HEIGHT = {"log": 2.1, "linear": 2.4}
#: In-axes legend placement per panel and y scale, chosen where the curves
#: leave room: on log the behavioral curves' plateau leaves the lower left
#: free; on linear it hugs the floor and the upper right is free instead.
LEGEND_LOC = {"layers": "upper left", "qmix": "upper center"}
#: Legend handle length per panel: panel 1's is just long enough to show the
#: single layers' dots on both sides of the marker.
HANDLE_LENGTH = {"layers": 1.6, "qmix": 1.0}
#: Panel 1's key names the thirds by depth, in order ``1–5``, ``6–10``,
#: ``11–16``; their colours also mark the single layers that fall in them.
DEPTH_NAMES = ("early", "middle", "late")
#: The gap between panels, as a fraction of what ``tight_layout`` leaves.
#: 0.5 runs the end tick labels together ("1000" into the next panel's "100");
#: 0.65 keeps about two thirds of a character between them.
WSPACE_SCALE = 0.65

#: Panels 3 and 4 are keyed by a discrete colorbar inset into the axes, one
#: tick per value: ``[x0, y0, width, height]`` in axes fractions, per y scale.
#: Like the legends it sits where the curves leave room -- lower left on log,
#: upper right on linear.  The bar's label sits above it, the ticks below.
COLORBAR_BOX = {"log": [0.07, 0.13, 0.86, 0.05],
                "linear": [0.22, 0.84, 0.74, 0.05]}


def ramp(colours, n):
    cmap = LinearSegmentedColormap.from_list("ramp", colours)
    return list(cmap(np.linspace(0.0, 1.0, n)))


def draw(ax, by_x, colour, label=None, ls="-", marker="o", zorder=2,
         alpha=1.0):
    xs, lo, mid, hi, _n = quantile_band(by_x, BAND_Q)
    if not xs:
        return
    if any(h > l for l, h in zip(lo, hi)):
        ax.fill_between(xs, lo, hi, color=colour, alpha=BAND_ALPHA, lw=0)
    ax.plot(xs, mid, linestyle=ls, color=colour, lw=LINEWIDTH, marker=marker,
            ms=MARKER_SIZE, label=label, zorder=zorder, alpha=alpha)


def by_queries(trials, column="disparity"):
    out = defaultdict(list)
    for r in trials:
        out[int(r["n_queries"])].append(float(r[column]))
    return out


def query_axes(ax, yscale):
    mq.style_axes(ax, yscale, mq.DISPARITY_YLIMS[yscale])
    ax.set_xlabel("Queries")


# ── panels ───────────────────────────────────────────────────────────────────

def panel_layers(ax, rows, yscale):
    cs = ml.curves(rows)
    top = max(ls[0] for _l, s, ls, _t in cs.values() if s == "single")
    dcol = dict(zip(range(1, top + 1), ramp(DEPTH_RAMP, top)))
    # Only the contiguous subsets: the evenly spaced ones are left out.
    contiguous = [v for v in cs.values() if v[1] == "subset"
                  and v[2] == list(range(v[2][0], v[2][-1] + 1))]
    scol = {v[0]: dcol[v[2][len(v[2]) // 2]] for v in contiguous}
    ref = next(v for v in cs.values() if v[1] == "all")

    # Drawn above the layer curves, so its triangles are not buried under them.
    draw(ax, by_queries(ref[3]), REFERENCE_COLOR,
         marker=REFERENCE_MARKER, zorder=3, alpha=LAYER_LINE_ALPHA)
    for label, _s, _ls, trials in contiguous:
        draw(ax, by_queries(trials), scol[label], alpha=LAYER_LINE_ALPHA)
    for label, s, layers, trials in cs.values():
        if s == "single" and layers[0] in SINGLE_LAYERS:
            # A single layer takes the colour of the third it falls in, so the
            # key's early / middle / late colours name both line kinds.
            third = next(v for v in contiguous if layers[0] in v[2])
            draw(ax, by_queries(trials), scol[third[0]], ls=SINGLE_LS,
                 alpha=LAYER_LINE_ALPHA)

    # The key: colour (column 1), then line kind (column 2).
    for (label, _s, _ls, _t), name in zip(contiguous, DEPTH_NAMES):
        ax.plot([], [], "-", color=scol[label], lw=LINEWIDTH, marker="o",
                ms=MARKER_SIZE, label=name, alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], "-", color=KEY_COLOR, lw=LINEWIDTH, label="group",
            alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], linestyle=SINGLE_LS, color=KEY_COLOR, lw=LINEWIDTH,
            label="single", alpha=LAYER_LINE_ALPHA)
    ax.plot([], [], "-", color=REFERENCE_COLOR, lw=LINEWIDTH,
            marker=REFERENCE_MARKER, ms=MARKER_SIZE, label="all",
            alpha=LAYER_LINE_ALPHA)
    query_axes(ax, yscale)
    ax.set_title("Functional layers")


def panel_qmix(ax, rows, yscale):
    for perspective, level in QMIX_LEVELS:
        for arm, _alabel, _ls in mx.ARMS:
            by_pct = defaultdict(list)
            for r in rows:
                if (r["perspective"] == perspective
                        and r["arm"] in (arm, mx.SHARED_ARM)
                        and int(r["seed"]) != mx.POOLED):
                    by_pct[int(r["yahoo_pct"])].append(float(r["disparity"]))
            draw(ax, by_pct, LEVEL_COLORS[level], ls=ARM_LINESTYLES[arm])
    # The key names level and corpus separately -- colour, then dash -- so its
    # entries stay short enough for a quarter-page panel.
    for _p, level in QMIX_LEVELS:
        ax.plot([], [], "-", color=LEVEL_COLORS[level], lw=LINEWIDTH,
                marker="o", ms=MARKER_SIZE, label=level)
    for arm in KEY_ARM_ORDER:
        ax.plot([], [], ARM_LINESTYLES[arm], color=KEY_COLOR,
                lw=LINEWIDTH, label=ARM_LABELS[arm])
    # qmix's axes without its one-yahoo-row (99% dilution) dotted line.
    ax.set_yscale(yscale)
    ax.set_xscale("symlog", linthresh=mx.SYMLOG_THRESHOLD,
                  linscale=QMIX_LINSCALE)
    ax.set_xlim(*matched_xlim(ax, 0, 100))
    ax.set_ylim(*mq.DISPARITY_YLIMS[yscale])
    ax.set_xticks(QMIX_TICKS)
    ax.set_xticklabels([str(100 - t) for t in QMIX_TICKS])
    ax.set_xlabel("Dilution %")
    ax.set_title("Query mixture")


def matched_xlim(ax, lo, hi):
    """x limits giving *ax* the query panels' margins around ``[lo, hi]``.

    The query panels pad their log axis by ``QUERY_XLIM`` around
    ``QUERY_DATA``; the same fraction of the axis width is put beyond *lo*
    and *hi* on *ax*'s own scale (set before this is called).
    """
    (a, b), (c, d) = np.log10(QUERY_XLIM), np.log10(QUERY_DATA)
    f_lo, f_hi = (c - a) / (b - a), (b - d) / (b - a)
    t = ax.xaxis.get_transform()
    tlo, thi = t.transform([lo, hi])
    span = (thi - tlo) / (1 - f_lo - f_hi)
    return tuple(t.inverted().transform([tlo - f_lo * span,
                                         thi + f_hi * span]))


def panel_replicates(ax, rows, yscale):
    sampled = mq.select(rows, EMBEDDER, "behavioral", "sampled",
                        temperature=FIXED_T)
    rs = sorted({int(r["replicates"]) for r in sampled})
    # Evenly spaced by index (R doubles each step), with the red stop moved
    # to R_ANCHOR's place in the sequence.
    at = rs.index(R_ANCHOR) / (len(rs) - 1)
    cmap = LinearSegmentedColormap.from_list(
        "R", list(zip([0.0, at, 1.0], R_RAMP)))
    colours = [cmap(i / (len(rs) - 1)) for i in range(len(rs))]
    for r, colour in zip(rs, colours):
        draw(ax, by_queries([t for t in sampled if int(t["replicates"]) == r]),
             colour)
    query_axes(ax, yscale)
    inset_colorbar(ax, rs, colours, [str(r) for r in rs], "replicates", yscale)
    ax.set_title(f"Replicates (T={FIXED_T:g})")


def panel_temperature(ax, rows, yscale):
    sampled = mq.select(rows, EMBEDDER, "behavioral", "sampled", FIXED_R,
                        temperature=None)
    ts = sorted({float(r["temperature"]) for r in sampled})
    cmap = LinearSegmentedColormap.from_list("T", T_RAMP)
    colours = [cmap(t) for t in ts]
    for t, colour in zip(ts, colours):
        draw(ax, by_queries([r for r in sampled
                             if float(r["temperature"]) == t]),
             colour)
    query_axes(ax, yscale)
    inset_colorbar(ax, ts, colours, [f"{t:g}" for t in ts], "temperature", yscale)
    ax.set_title(f"Temperature (R={FIXED_R})")


def inset_colorbar(ax, values, colours, labels, title, yscale):
    """A discrete horizontal colorbar inside *ax*: one cell per value."""
    cax = ax.inset_axes(COLORBAR_BOX[yscale])
    n = len(values)
    cb = ax.figure.colorbar(
        ScalarMappable(Normalize(-0.5, n - 0.5), ListedColormap(colours)),
        cax=cax, orientation="horizontal")
    cb.set_ticks(range(n))
    cb.set_ticklabels(labels)
    cb.ax.tick_params(length=2, pad=1)
    cb.outline.set_edgecolor(mq.GREEDY_COLOR)
    cax.set_title(title, pad=2)
    return cax


def colorbar_frame(ax, cax, renderer):
    """A legend's frame behind *cax*, its title and its ticks.

    Drawn from the house legend rc (rounded, ``legend.edgecolor``, the
    axes' face at ``legend.framealpha``) with the legend's own border pad, so
    the colorbar keys read as the same kind of object as panels 1-2's.
    Called after layout, when the colorbar's extent is final.
    """
    rc = matplotlib.rcParams
    pad = rc["legend.borderpad"] * rc["legend.fontsize"] / 72 * \
        ax.figure.dpi
    box = cax.get_tightbbox(renderer).padded(pad)
    (x0, y0), (x1, y1) = ax.transAxes.inverted().transform(box.get_points())
    edge = rc["legend.edgecolor"]
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.02",
        transform=ax.transAxes, facecolor=ax.get_facecolor(),
        edgecolor=rc["axes.edgecolor"] if edge == "inherit" else edge,
        alpha=rc["legend.framealpha"], lw=rc["patch.linewidth"],
        zorder=4.5, clip_on=False))


def build(layers, subsample, qmix, outdir, ext, yscale) -> Path:
    font = apply_style(PAGE_WIDTH)
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_WIDTH, HEIGHT[yscale]), sharey=True)
    panel_layers(axes[0], layers, yscale)
    panel_qmix(axes[1], qmix, yscale)
    panel_replicates(axes[2], subsample, yscale)
    panel_temperature(axes[3], subsample, yscale)
    axes[0].set_ylabel(YLABEL)

    for ax, (key, loc) in zip(axes, LEGEND_LOC.items()):
        # Figure 2's legend style (the house rc), in two columns filled
        # column-first: colour entries, then line-kind entries -- panel 1's
        # early/middle/late then group/single/all; panel 2's levels then its
        # corpora.  Only the handle and column gaps are
        # trimmed, so two columns fit a quarter-page panel.  Kept out of the
        # layout so the four panels stay evenly spaced.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=loc, ncol=2,
                  handlelength=HANDLE_LENGTH[key], handletextpad=0.3,
                  columnspacing=0.5, borderaxespad=0.3).set_in_layout(False)
    fig.tight_layout()
    fig.subplots_adjust(wspace=fig.subplotpars.wspace * WSPACE_SCALE)
    renderer = fig.canvas.get_renderer()
    for ax in axes[2:]:
        colorbar_frame(ax, ax.child_axes[0], renderer)
    embolden(fig)
    return save_figure(fig, Path(outdir) /
                       f"fig_figure3{mq.YSCALE_SUFFIX[yscale]}.{ext}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default=str(HERE / "qbig_layers.csv"))
    ap.add_argument("--subsample", default=str(HERE / "qbig_subsample.csv"))
    ap.add_argument("--qmix", default=str(HERE / "qmix_scores.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--yscale", default="both",
                    choices=["both"] + list(mq.DISPARITY_YLIMS))
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    layers = mq.read_rows(Path(args.layers))
    subsample = mq.read_rows(Path(args.subsample))
    qmix = mx.read_rows(Path(args.qmix))
    scales = (list(mq.DISPARITY_YLIMS) if args.yscale == "both"
              else [args.yscale])
    for yscale in scales:
        path = build(layers, subsample, qmix, args.outdir, args.ext, yscale)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
