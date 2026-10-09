"""House style for the sweep figures: one text size, one weight, a caption band.

The sweep drivers (``figures/simplex3_qmix_olmo2/make_qmix_figures.py``,
``figures/simplex3_qbig_olmo2/make_qbig_figures.py``) draw several panels in one
frame and are read in the same document as figure 2, so they share figure 2's
typographic rules.  Those rules lived first in the qmix driver; they are here so a
second driver imports them instead of carrying a copy that can drift.

Three pieces:

:func:`apply_style`
    ``set_style`` plus one text size and a bold-capable family over the top.

:func:`embolden`
    The texts ``font.weight`` does not reach -- tick labels and legend entries.

:func:`place_caption`
    Lays the axes out, *then* adds the caption, so its band is reserved once.

and one data helper, :func:`quantile_band`, which turns ``{x: [values]}`` into the
``(xs, lo, mid, hi, n)`` a banded curve is drawn from.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib
import numpy as np

from src.plots.config import bold_capable_family, set_style

__all__ = [
    "FONT_SIZE", "REFERENCE_WIDTH", "FONT_SIZE_FLOOR", "AXIS_COLOR",
    "CAPTION_LINES", "apply_style", "embolden", "place_caption",
    "quantile_band",
]

#: One text size for everything, at one weight -- panel titles, axis labels,
#: tick labels, legends and the caption alike. Figure 2's value, and its rule:
#: several panels in one frame typed at different sizes reads as emphasis rather
#: than as provenance, so there is one size and no sub-scaling of it.
FONT_SIZE = 13

#: The width :data:`FONT_SIZE` is written for -- figure 2's own reference, so a
#: build at a given width is typed the same as a build of figure 2. 19 in is its
#: nine-panel full build; narrower figures read the size through
#: :func:`apply_style`.
REFERENCE_WIDTH = 19.0

#: The smallest the text may get whatever the width asks for. Figure 2's value,
#: and its argument: scaling 13 pt down with the width would type a 7.5 in build
#: at 5.1 pt, which is a picture of a figure rather than a figure -- a reader
#: cannot read 5 pt in print, so the panels give up the room instead.
FONT_SIZE_FLOOR = 7.0

#: Figure 2's axis colour, so the figures' frames match.
AXIS_COLOR = "0.35"

#: Lines in a caption, and the padding around it, in multiples of the text
#: size.  A caption here is a fixed number of lines by construction, so the band
#: it needs is arithmetic rather than a measured bounding box.
CAPTION_LINES = 3.6


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


def place_caption(fig, text, font, height, *, bottom=0.0,
                  lines=CAPTION_LINES) -> None:
    """Lay the axes out, then put the caption in the band left for it.

    **The order matters and is the reason this is a function.**
    ``tight_layout`` accounts for a suptitle that already exists *in addition to*
    whatever ``rect`` reserves, so laying out first against a rect and adding the
    caption afterwards is the only way to reserve the band exactly once.  Doing
    it the other way round double-counts: the panels lose the caption's height
    twice and the figure grows a band of dead space between the two.

    *lines* is the band's height in multiples of the text size; *bottom* is a
    fraction of the figure left free below the axes, for a figure legend.
    """
    band = lines * font / 72.0 / height
    fig.tight_layout(rect=(0, bottom, 1, 1.0 - band))
    fig.suptitle(text, fontsize=font, y=0.995, va="top")


def quantile_band(
    values_by_x: Mapping[float, Sequence[float]],
    q: tuple[float, float] = (25.0, 75.0),
):
    """``(xs, lo, mid, hi, n)`` for one banded curve.

    *values_by_x* maps each x position to the trials measured there; *mid* is
    their median and *lo*/*hi* the percentiles *q*.  ``n`` maps each x to its
    trial count, so a caption can say what the curve rests on.  Where a position
    holds one trial, *lo* and *hi* equal *mid* and the band draws at zero width
    -- the figure does not invent a spread it has not measured.  Positions with
    no trials are dropped rather than drawn as gaps.
    """
    xs = sorted(x for x, v in values_by_x.items() if len(v))
    arr = {x: np.asarray(values_by_x[x], dtype=float) for x in xs}
    return (xs,
            [float(np.percentile(arr[x], q[0])) for x in xs],
            [float(np.median(arr[x])) for x in xs],
            [float(np.percentile(arr[x], q[1])) for x in xs],
            {x: int(arr[x].size) for x in xs})
