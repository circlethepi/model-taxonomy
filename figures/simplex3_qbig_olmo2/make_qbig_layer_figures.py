#!/usr/bin/env python
"""Functional recovery of the simplex, layer by layer, against probe depth.

**qbig** (*big query set*) -- the standing sixteen OLMo-2-1B yahoo-simplex
adapters, probed once at 1000 queries.  See ``docs/terminology.md``.

Every number is read from ``qbig_layers.csv``, which ``layers_qbig.py`` writes:
the functional level (input mode, mean pooling, layer-normalized concatenation,
cosine) read through one hidden state or one combination of them, for ten query
sets per query count.  This file is purely plotting.

``fig_qbig_layers.pdf`` (and ``_linear``)
    Rows: Procrustes disparity, dCor*.  Columns:

    * **single layers** -- h1 … h16, one line each, light to dark by depth.
      ``h0`` (the embeddings, identical across all sixteen models) has no
      score and is not in the CSV.
    * **thirds** -- early (h1-h5), middle (h6-h10), late (h11-h16), each joined,
      coloured at their place on the same depth ramp.
    * **every Nth layer** -- one layer from each evenly spaced step of h1-h16,
      joined: every 15th (h1, h16), 5th, 3rd, and every layer (h1-h16).

    **All hidden states** (h0-h16), the standing functional row, is drawn in
    black in every panel as the reference.  x is the number of queries.

Usage::

    python figures/simplex3_qbig_olmo2/make_qbig_layer_figures.py
    python figures/simplex3_qbig_olmo2/make_qbig_layer_figures.py --ext png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402

import make_qbig_figures as mq  # noqa: E402
from src.plots import save_figure  # noqa: E402
from src.plots.sweep_figures import apply_style, embolden, place_caption  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Depth, light to dark, in the functional level's hue family.  The range stops
#: short of the palest end, which vanishes on white.
DEPTH_CMAP = "YlOrBr"
DEPTH_CMAP_RANGE = (0.3, 1.0)
#: "Every Nth layer" is a density, not a depth, so it takes a ramp of its own:
#: light = fewest layers kept.
DENSITY_CMAP = "PuBu"
DENSITY_CMAP_RANGE = (0.4, 1.0)
REFERENCE_COLOR = "black"
#: Figure fraction kept below the axes for the three column keys.
KEY_STRIP = 0.17

#: ``(layer_set in the CSV, panel title)`` per column.
COLUMNS = [("single", "Single layers"),
           ("third", "Thirds"),
           ("every_nth", "Every Nth layer")]


def depth_colours(top):
    """``{layer: colour}`` for h1..h(top)."""
    cmap = plt.get_cmap(DEPTH_CMAP)
    return dict(zip(range(1, top + 1),
                    cmap(np.linspace(*DEPTH_CMAP_RANGE, top))))


def curves(rows):
    """``{perspective: (label, layer_set, layers, trials)}`` in CSV order."""
    out = {}
    for r in rows:
        if r["perspective"] not in out:
            out[r["perspective"]] = (r["label"], r["layer_set"],
                                     [int(x) for x in r["layers"].split()], [])
        out[r["perspective"]][3].append(r)
    return out


def caption():
    return (
        "Functional recovery of the mixture simplex, by hidden state, vs. "
        "probe depth · OLMo-2-1B-Instruct · ONE fixed 16-adapter yahoo simplex\n"
        "subsampled from one run at 1000 queries (seed 2) · input mode, mean "
        "pooling, layer-normalized concatenation, cosine · black = all hidden "
        "states (h0–h16)\n"
        "line = median · band = 10-90% over query draws · "
        f"dotted = {mq.STANDING_QUERIES} queries, the standing probe's depth"
    )


def fig_layers(rows, outdir, ext, yscale="log") -> Path:
    cs = curves(rows)
    top = max(ls[0] for _l, s, ls, _t in cs.values() if s == "single")
    dcol = depth_colours(top)
    nth = sorted((v for v in cs.values() if v[1] == "every_nth"),
                 key=lambda v: len(v[2]))
    ncol = dict(zip((v[0] for v in nth), plt.get_cmap(DENSITY_CMAP)(
        np.linspace(*DENSITY_CMAP_RANGE, len(nth)))))
    ref = next(v for v in cs.values() if v[1] == "all")

    height = 6.0
    font = apply_style(mq.PAGE_WIDTH)
    rows_m = mq.metrics(yscale)
    fig, axes = plt.subplots(len(rows_m), len(COLUMNS),
                             figsize=(mq.PAGE_WIDTH, height), sharex=True,
                             sharey="row", squeeze=False)
    for mi, (col, ylabel, scale, lims) in enumerate(rows_m):
        for ci, (layer_set, title) in enumerate(COLUMNS):
            ax = axes[mi][ci]
            mq.draw(ax, ref[3], col, REFERENCE_COLOR,
                    label="all hidden states" if ci == 0 else None)
            if layer_set == "single":
                for label, s, layers, trials in cs.values():
                    if s == "single":
                        mq.draw(ax, trials, col, dcol[layers[0]])
            elif layer_set == "third":
                for label, s, layers, trials in cs.values():
                    if s == "third":
                        mid = layers[len(layers) // 2]
                        mq.draw(ax, trials, col, dcol[mid],
                                label=f"{label} (h{layers[0]}–h{layers[-1]})")
            else:
                for label, _s, _ls, trials in nth:
                    mq.draw(ax, trials, col, ncol[label], label=label)
            mq.style_axes(ax, scale, lims)
            if mi == 0:
                ax.set_title(title)
            if mi == len(rows_m) - 1:
                ax.set_xlabel("queries")
            if ci == 0:
                ax.set_ylabel(ylabel)

    # One key per column, under it: a depth colorbar for the singles, legends
    # for the other two.  The reference sits under the singles' bar.
    # place_caption runs tight_layout, so it reserves the keys' strip first and
    # the keys are placed against the axes' final positions.
    place_caption(fig, caption(), font, height, bottom=KEY_STRIP)
    for ci, (layer_set, _t) in enumerate(COLUMNS):
        box = axes[-1][ci].get_position()
        if layer_set == "single":
            cax = fig.add_axes([box.x0, KEY_STRIP - 0.045, box.width, 0.014])
            cmap = ListedColormap([dcol[k] for k in range(1, top + 1)])
            cb = fig.colorbar(ScalarMappable(Normalize(0.5, top + 0.5), cmap),
                              cax=cax, orientation="horizontal")
            ticks = [1, top // 4, top // 2, 3 * top // 4, top]
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"h{t}" for t in ticks])
            cb.ax.tick_params(labelsize=font * 0.8, length=2, pad=1)
            h, l = axes[0][0].get_legend_handles_labels()
            fig.legend(h, l, loc="upper left", frameon=False,
                       bbox_to_anchor=(box.x0 - 0.01, KEY_STRIP - 0.075),
                       fontsize=font * 0.8)
        else:
            # The reference is unlabelled here: it is keyed once, under singles.
            h, l = axes[0][ci].get_legend_handles_labels()
            fig.legend(h, l, loc="upper left", frameon=False,
                       bbox_to_anchor=(box.x0 - 0.01, KEY_STRIP - 0.025),
                       fontsize=font * 0.75, handlelength=1.4)
    embolden(fig)
    return save_figure(fig, Path(outdir) /
                       f"fig_qbig_layers{mq.YSCALE_SUFFIX[yscale]}.{ext}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "qbig_layers.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--yscale", default="both",
                    choices=["both"] + list(mq.DISPARITY_YLIMS))
    ap.add_argument("--ext", default="pdf", choices=["pdf", "png", "svg"])
    args = ap.parse_args()

    rows = mq.read_rows(Path(args.scores))
    scales = (list(mq.DISPARITY_YLIMS) if args.yscale == "both"
              else [args.yscale])
    for yscale in scales:
        print(f"wrote {fig_layers(rows, args.outdir, args.ext, yscale)}")


if __name__ == "__main__":
    main()
