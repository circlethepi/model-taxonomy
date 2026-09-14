#!/usr/bin/env python
"""Figure 2: the cross-level simplex over one run, and the four ways it moves.

Nine panels in two rows, and the two rows answer different questions.

**Top row** — the simplex3 cross-level closer for OLMo-2-0425-1B-Instruct on
yahoo: the mixture key, then one MDS panel per taxonomy level, in the reading
order of the taxonomy — **Data, Structural, Functional, Behavioral**, the same
order the bar panels underneath group their bars in, so a top-row column sits
above its own bar group. It is the same picture
``figures/simplex3_olmo2/fig_crosslevel_mds.png`` draws, with three deliberate
differences. The panel titles carry the **taxonomy name and nothing else** — no
dCor, no Procrustes residual, no stress — there is no figure title, no corpus
subtitle and no title over the mixture key, and the axes carry no ticks. Those
numbers have not gone missing; they are what the bottom row is made of, and
repeating them above it would say the same thing twice. The ticks are gone
because an MDS coordinate has no units and no origin: the configuration is
defined only up to rotation, reflection and scale, which is why it is scored by
Procrustes in the first place.

**Bottom row** — four readings of the Procrustes disparity against the
ground-truth mixture geometry, on one shared y axis:

1. **Base Model** — every base model on yahoo, x = taxonomy level;
2. **Training Data Source** — every corpus on OLMo-2-1B, x = taxonomy level;
3. **Collection Size** — the collection-size sweep, x = models scored together;
4. **Training Set Size** — the nsweep, x = training examples per adapter.

Panels 1-2 are grouped bars, 3-4 are lines with an interquartile band, and all
four are the *same score in the same direction*: a raw disparity, 1 → 0, **lower
is better**. The axis label says the score and not the direction, because at one
shared text size a longer label is clipped; the direction is here and in the
caption.

``--yscale`` picks between the two ranges, and neither is a correction of the
other. **log** (the default, 4e-3 to 1.2) is the only scale on which all four
levels are legible at once — the row spans more than two orders of magnitude,
0.007 for the functional level on the yahoo simplex against 0.8 for the
behavioral level at a ten-example training draw. **linear** (a flat 0 to 1, the
score's own full range) is the only scale on which the differences are literal,
at the price of the data, structural and functional values collapsing onto the
axis.

Perspectives
------------
A **perspective** is a surrogate together with a metric — one cell of a level's
grid. Every panel here is pinned to one perspective per level, so a bar in panel
1 and a curve in panel 3 are the same measurement read on different axes:

=============  ==============================  ===========
level          surrogate                       metric
=============  ==============================  ===========
Data           dataset text · mean             euclidean
Structural     output projections              cosine
Functional     all layers (reference)          cosine
Behavioral     R=16 · per query *or* greedy    cosine
=============  ==============================  ===========

The behavioral row is the one that is not settled, and the figure says so by
drawing it both ways. Its default (``--behavioral mixed``) reads **R=16 · per
query** in the top row and in panels 1-2, and **greedy · per generation** in
panels 3-4. That split is not cosmetic: on the 999-model collection-size pool
the R=16 disparity sits at a flat ~0.47 at every size while greedy runs at
~0.13, so the sampled read has no signal left to show a size effect with, and a
panel drawn from it would report a noise floor as a result. ``--behavioral
greedy`` draws every panel, top row included, from greedy decoding instead, so
the two can be compared without the split.

Surrogate labels differ across architectures — a 16-layer OLMo names its
functional reference row ``all 17 layers (reference)`` where a 40-layer Nemo
names it ``all 41``, and hybrid-attention Qwen writes ``output projections
(d_in 4096)`` — so panels 1-2 match by substring against each run's own labels,
via :class:`src.plots.simplex_runs.Perspective`.

Which Procrustes disparity
--------------------------
Panel 2 puts three corpora on one axis and they do not share an embedding
dimension: yahoo mixes three groups so its truth lives in ``d2``, dolly and
oasst1 mix four and theirs lives in ``d3``. Each run is read at **its own
``d = K-1``**, which is that run's honest disparity. The alternative — every run
at ``d2`` — is one measurement throughout but asks a tetrahedral truth to fit in
a plane, which inflates dolly and oasst1 to 0.34-0.59 against yahoo's 0.007-0.08
and would make panel 1 unreadable beside it. Panels 1, 3 and 4 are yahoo only,
so ``d2`` and ``d = K-1`` are the same number there.

Panels 3 and 4 each carry two defensible readings in their CSVs, and this draws
one of each:

* panel 3, **variant B** — matrices built over the ``n`` sampled models alone,
  with no reference models informing the embedding;
* panel 4, the **requested** truth — scored against the mixture the recipe asks
  for, rather than the one largest-remainder allocation actually realized.

Bands
-----
Both sweep panels draw a faint interquartile fill around the median. In panel 4
that is an honest spread: ten independent dataset seeds, no two collections
sharing an adapter. In panel 3 it is **not** — the replicates are drawn without
replacement from one shared pool, so they overlap in membership and the spread
is deflated by construction, and past ``n = 500`` every replicate is the same
collection and the band is exactly zero width. The per-panel figures in
``figures/simplex_collection_size`` mark that boundary properly; this one is an
aggregate and does not.

Nothing in the bottom row is recomputed. Panels 1-2 read the twelve suites'
``crosslevel_scores.csv``, panel 3 reads ``group_size_scores.csv`` and panel 4
``nsweep_scores.csv``. The top row is the one part that needs the cache, because
an MDS panel needs the distance matrix itself and not a score of it; it calls
:func:`src.plots.simplex_suite.run_suite` with one perspective per level and
``closer=False``, so nothing is written and the four cells come back from
``06_pairwise``.

Width
-----
The figure is drawn at one of the named widths in :data:`PRESETS`, or at any
width ``--width`` is given as a number. **full** (19 in) is the figure as it has
always been and what the tracked PNGs are; **page** (7.5 in) is a letter page
inside 1 in margins.

A narrow build is not a photograph of the wide one, and the difference is the
point. Two things scale, by different amounts: the panels scale with the width,
and the text scales with the width only until it reaches
:data:`FONT_SIZE_FLOOR`, below which it stops. 7.5 in is 0.39 of the full width,
which would set the body text at 5.1 pt — a size nothing in print can be read
at — so the type stops at 7 pt and the panels give up the room instead. Three
things follow from that and are handled rather than left to collide:

* the figure comes out **taller than a proportional scaling** would make it,
  because the band of title, label and ticks around each panel shrinks only as
  fast as the text does. It is not made taller than the panels can use: the top
  row's panels are square by construction, so height past their own width is
  whitespace. ``--height`` overrides the result for fitting a fixed space.
* the **key thins**, below :data:`KEY_THIN_BELOW_WIDTH`, from all sixteen
  mixtures to the seven landmarks — vertices, centre, edge midpoints. Sixteen
  labels will not fit around a 1.3 in triangle at any size that can also be
  read. The thinned key also puts the centre's label to one side
  (:data:`KEY_CENTRE_LABEL_SIDE`) rather than letting the collision pass place
  it: straight up and straight down from the centre is another sampled point,
  and the band to either side of it holds none. The four MDS panels are
  untouched: they have always labelled the vertices and the centre alone, and
  that set is already inside this one. ``--no-mds-labels`` drops those four
  where the panel area is worth more than naming the same mixtures twice.
* the **wording changes**, by preset. "Training examples per adapter" under a
  1.8 in panel runs into its neighbour, and by then the text is at its floor, so
  the fix is shorter words rather than smaller ones. Every title and axis label
  in the figure is a slot in :data:`DEFAULT_TEXT`; a preset overrides the slots
  it needs, and ``--label slot=text`` overrides one from the command line. The
  four taxonomy levels have two sets of slots -- ``level_*`` for the top row's
  titles and the sweep legend, ``tick_*`` for the group under each set of bars
  -- and the tick slots default to the titles, so they only need setting when
  the two should read differently.

Typography
----------
Every piece of text in the figure is one size and one weight — see
:data:`FONT_SIZE`, :func:`layout_for` and :func:`_apply_style`. The three sources this figure is
assembled from do not agree on either by default, because each was written for a
figure it was the whole of, and side by side in one frame that difference reads
as emphasis rather than as provenance. The single exception is the labels that
name a mixture — the four annotated inside each MDS panel and the key's sixteen
— which are set together at :data:`MIXTURE_LABEL_SIZE`, small enough for the
densest panel that carries them.

Usage
-----
::

    python figures/figure2/make_figures.py
    python figures/figure2/make_figures.py --yscale linear
    python figures/figure2/make_figures.py --row top
    python figures/figure2/make_figures.py --row bottom --yscale linear
    python figures/figure2/make_figures.py --behavioral greedy
    python figures/figure2/make_figures.py --width page
    python figures/figure2/make_figures.py --width 11 --height 5
    python figures/figure2/make_figures.py --width page --label ylabel="Disparity ↓"
    python figures/figure2/make_figures.py --width page --no-mds-labels

The full-size build keeps the filenames it has; every other shape says its shape
in its name, so the nine tracked figures cannot be quietly replaced by a build
at another width.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Pinned before numpy loads its BLAS -- see the note in src/plots/simplex_suite.py.
os.environ.setdefault("MODEL_TAXONOMY_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.plots import make_series, plot_grouped_bars, save_figure, set_style  # noqa: E402
from src.plots.config import bold_capable_family  # noqa: E402
from src.plots import simplex_runs as runs  # noqa: E402
from src.plots.simplex import crosslevel_panel, ternary_legend  # noqa: E402
from src.plots.simplex import group_display, mixture_label  # noqa: E402
from src.plots.simplex import CENTRE_LABEL, VERTEX_LABELS  # noqa: E402
from src.plots.simplex import EDGE_MIDPOINT_LABELS  # noqa: E402

FIGURES_ROOT = REPO_ROOT / "figures"

# ── The palette ───────────────────────────────────────────────────────────────
#   Eleven encodings — four taxonomy levels, four base models, three corpora —
#   and no hue is used for two of them. Okabe-Ito wherever it reaches, and three
#   Tol additions where it does not; every entry stays separable under the common
#   colour-vision deficiencies.
#
#   Deliberately local to this driver rather than pushed into
#   `src/plots/simplex_runs.py`: that registry's model and corpus colours are
#   what `figures/tasks_aggregate` and `figures/simplex3_aggregate` already drew
#   their tracked figures under, and recolouring it here would silently redraw
#   those the next time they are run.
#
#   The three families never share a panel, so the near pairs across families —
#   sky against structural blue, teal against data green — are never seen side by
#   side. Within a panel the four (or three) entries are maximally separated,
#   which is the constraint that matters.

#: Taxonomy level -> colour. Green data, blue structural, gold functional, red
#: behavioral, in the reading order of the taxonomy: from what the model was
#: trained on, through what its weights became and what its activations do, to
#: what it says.
LEVEL_COLORS = {
    "Data": "#009E73",
    "Structural": "#0072B2",
    "Functional": "#E69F00",
    "Behavioral": "#D55E00",
}

#: Base model -> colour, in ascending parameter count. Size is the one axis
#: these four can be put on that is not arbitrary, so panel 1's bars can be read
#: left to right within a group.
#:
#: A single-hue blue ramp rather than four hues: lightness carries the ordering,
#: which no reader loses to colour vision, and one hue for the panel says the
#: four bars in a group differ in degree rather than in kind.
MODEL_COLORS = {
    # "allenai/OLMo-2-0425-1B-Instruct":      "#C6DBEF", # initial colors
    # "Qwen/Qwen3.5-4B":                      "#6BAED6",
    # "meta-llama/Llama-3.1-8B-Instruct":     "#2171B5",
    # "mistralai/Mistral-Nemo-Instruct-2407": "#08306B",
    "allenai/OLMo-2-0425-1B-Instruct":      "#A6C6E1",
    "Qwen/Qwen3.5-4B":                      "#649CCB",
    "meta-llama/Llama-3.1-8B-Instruct":     "#2171B5", # base color + tint
    "mistralai/Mistral-Nemo-Instruct-2407": "#123E64",
}

#: Corpus -> colour, in order of increasing distance from a topic mixture:
#: yahoo mixes topics, dolly mixes instruction tasks, oasst1 mixes languages.
#: Teal, green and turquoise: one neighbourhood of the wheel, like the model
#: ramp beside it, but three hues at comparable weight rather than a ramp --
#: nothing orders three corpora, so nothing here should look ordered.
CORPUS_COLORS = {
    "yahoo" :   "#1E5513",
    "dolly" :   "#379A22",
    "oasst1":   "#AFD7A7"
}
# {"yahoo": "#17807A", "dolly": "#4C9A2A", "oasst1": "#5FD3D0"}

#: One line style for every level in panels 3-4. Hue alone separates the four
#: curves; they are far enough apart vertically that a dash pattern per level
#: added texture without adding information.
SWEEP_LINESTYLE = "-"

#: The curves carry the panel, so they are drawn heavier than a default line.
SWEEP_LINEWIDTH = 2.1
SWEEP_MARKER_SIZE = 7

#: The shape every mixture point is drawn with, in the four MDS panels and in
#: the key alike, and the shape the **frame points** take instead -- the three
#: pure mixtures and the even one, which are the four points
#: :func:`~src.plots.simplex.frame_labels` names and the ones every panel is
#: rotated onto. They are the only points a reader looks up by position rather
#: than by colour, so they are the only ones given a shape of their own.
#:
#: A triangle of a given area reads smaller than a circle of it -- half the
#: bounding box is empty -- so the frame points are drawn at
#: :data:`FRAME_MARKER_SCALE` times the area of the rest, which is what makes
#: the two read as one size.
MIXTURE_MARKER_SHAPE = "o"
FRAME_MARKER_SHAPE = "^"
FRAME_MARKER_SCALE = 1.3

#: The IQR band is context for its curve, not a second series. Faint enough that
#: two overlapping bands do not read as a third colour.
BAND_ALPHA = 0.12

#: Where the sweep row's one legend sits. It names the four levels for both
#: panels, so it goes in whichever corner of the collection panel the curves
#: leave free: they rise to the right there, so the upper left is the empty one.
SWEEP_LEGEND_LOC = "upper left"

#: The marker is the same for every level. It marks where a point was measured
#: — the seven collection sizes, the nine draw sizes — which is a property of
#: the sweep and not of the level, so varying it by level would encode the
#: level twice and the sampling not at all. Colour and dash carry the level.
SWEEP_MARKER = "o"

#: Every piece of text in the figure, in points. One size throughout — panel
#: titles, axis labels, tick labels, legends, the annotated simplex points and
#: the mixture key's own labels.
#:
#: The pieces this figure is assembled from do not agree on sizes by default:
#: ``set_style("two_col_full")`` types a bar panel's ticks at 6 pt while
#: :func:`~src.plots.simplex.crosslevel_panel` types an MDS panel's title at 13,
#: because each was written for a figure it was the whole of. Side by side in one
#: figure that difference reads as emphasis rather than as provenance, so it is
#: flattened here rather than in either source.
#:
#: Every size in this file is written for :data:`REFERENCE_WIDTH`, and
#: :func:`layout_for` is what a narrower figure reads them through.
FONT_SIZE = 13

#: The smallest this figure's text is allowed to get, whatever the width asks
#: for. Scaling 13 pt down with the width would type the 7.5 in build at 5.1 pt,
#: which is a picture of the figure rather than the figure: a reader cannot read
#: 5 pt in print, so the panels give up the room instead. Below this floor the
#: text stops shrinking and the panels keep shrinking around it, which is why a
#: narrow build is not a scaled photograph of the wide one.
FONT_SIZE_FLOOR = 7.0

#: The one exception: every label that names a mixture, in the key and in the
#: four MDS panels alike. These sit among the points they name rather than
#: outside an axes, and sixteen of them share the key's triangle, so at FONT_SIZE
#: they crowd the configuration they are there to explain. One value for all of
#: them -- the same text saying the same thing in five panels -- and it is the
#: size the densest of those panels can carry.
MIXTURE_LABEL_SIZE = 8

#: Their own floor, a step under :data:`FONT_SIZE_FLOOR`. These labels keep the
#: same relation to the body text at every width -- a step under it -- and 5 pt
#: is readable here where it would not be for a tick, because a reader looks one
#: of these up deliberately rather than scanning them.
MIXTURE_LABEL_FLOOR = 5.0

#: The *triangle* is drawn smaller than the panels beside it -- it is a legend,
#: not a fifth measurement -- by widening the margin around it inside an axes
#: cell the row's grid fixes. Nothing else in the key shrinks with it: the
#: labels, the vertex names and the markers are text and marks, sized against
#: the rest of the figure rather than against the triangle they sit on.
KEY_PAD = 0.3

#: T seven are still pushed outward: the vertices and
#: the edge midpoints all sit on the boundary and label outward from it. The
#: seventh is the centre, and it does not fan at all -- see the ``fixed_points``
#: argument in :func:`draw_the same margin for a key that has thinned its labels. Seven labels need less
#: room to fan into than sixteen, so the margin gives a little back to the
#: triangle -- the one thing in the key that cannot afford to shrink twice, once
#: with the figure and once with the margin around it. It gives back only a
#: little because six of theop_row`.
KEY_PAD_THINNED = 0.28

#: How far a mixture label may be pushed to clear a neighbour, in points at
#: :data:`REFERENCE_WIDTH`. It is a distance in points on a triangle measured in
#: inches, so it travels with the text: a third-size key given the full 60 pt of
#: travel puts a label outside the frame to clear a neighbour it is only just
#: touching.
KEY_LABEL_MAX_EXTRA = 60.0

#: Which side the centre mixture's label is put on when the key thins, or
#: ``None`` to leave it to the collision pass. The centre is the one sampled
#: mixture with no radius to fan along, and the two directions it would
#: otherwise take are the two that are occupied: the mixtures sit in rows of
#: constant first weight, so straight up and straight down from the centre is
#: another sampled point. The band to either side of it, between two rows,
#: holds none. It is the *upper* right because the band is not level: the row
#: above the centre is the wider one, so leaning the label into it puts more
#: clear space between the label and the points it is threading between than
#: running straight out does.
KEY_CENTRE_LABEL_SIDE = "upper right"

#: The labels are pushed further out along their radii than the default 9.5 pt,
#: because the triangle they sit on is smaller while they are not: the sixteen
#: points are closer together in inches, so the labels need more room to fan
#: into. The vertex names clear this on their own -- see ``ternary_legend``.
KEY_LABEL_OFFSET = 19.0

#: How far above its point an MDS panel's label sits, in points. It travels with
#: the label size rather than staying put: at 5 pt the default 11 leaves a gap
#: the text no longer needs, and the label drifts toward its neighbour above.
POINT_LABEL_PAD = 11.0

#: and its floor. Below about 7 pt the label sits on the marker it names rather
#: than above it: the marker keeps a radius of its own however small the text
#: gets, so this distance cannot scale all the way down with the type.
POINT_LABEL_PAD_FLOOR = 7.0

#: The gap between a panel title and the panel, in points.
TITLE_PAD = 10.0

#: One marker area for every mixture point in the top row, key included. The
#: key and the four MDS panels draw the same sixteen models, and a reader who
#: looks up a point in the key and then finds it in a panel should be looking at
#: the same mark; the two sources default to 26 and 130, which reads as two
#: different kinds of thing.
MIXTURE_MARKER_SIZE = 150

#: The width of every axis line, and the thinnest it is allowed to become. Below
#: about half a point a spine drops out at screen resolution and prints grey.
AXIS_LINEWIDTH = 1.0
AXIS_LINEWIDTH_FLOOR = 0.6

#: Which mixtures an MDS panel names, and whether it names any. The four are
#: the frame: the three vertices and the centre are what ``align_to_simplex``
#: fixes each panel's rotation and reflection by, so they are the points a
#: reader checks a panel against. ``MDS_LABELS`` turns all four off for a build
#: that would rather have the panel area -- the key beside them names the same
#: mixtures, and in a narrow figure four labels inside a 1.8 in panel cost more
#: room than they return. ``--mds-labels`` / ``--no-mds-labels`` overrides it
#: per build, and a preset may set its own.
MDS_LABEL_POINTS = VERTEX_LABELS + (CENTRE_LABEL,)
MDS_LABELS = True

#: The colour of every axis line in the figure -- the MDS crosshairs, the panel
#: spines, and the key's triangle, which is that key's only frame. One value, so
#: the nine panels sit in one box rather than in nine.
AXIS_COLOR = "0.55"


# ── The wording ───────────────────────────────────────────────────────────────
#   Every title and axis label in the figure, in one table, because the right
#   words depend on how wide the figure is drawn. "Training examples per adapter"
#   is a 29-character axis label; under a 1.8 in panel it runs past the axes and
#   into its neighbour, and the fix is a shorter label rather than a smaller one
#   -- the text is already at its floor by then.
#
#   A slot is named for where it appears, not for what it says, so a preset can
#   change the words without either of them having to know about the other.

#: Slot -> the words at :data:`REFERENCE_WIDTH`. This is the figure as it has
#: always read; a preset overrides the slots it needs and inherits the rest.
DEFAULT_TEXT: dict[str, str | None] = {
    "key_title":         "Dataset Mixture",
    "mds_x":             "MDS 1",
    "mds_y":             "MDS 2",
    "ylabel":            "Procrustes Disparity",
    "models_title":      "Base Model",
    "corpora_title":     "Training Data Source",
    "collection_title":  "Collection Size",
    "collection_xlabel": "Models in the collection",
    "nsweep_title":      "Training Set Size",
    "nsweep_xlabel":     "Training examples per adapter",
    # The four taxonomy levels as a *reader* sees them: the top row's panel
    # titles, the bar panels' group labels and the sweep panels' legend. The
    # level's own ``label`` stays the key everything else is addressed by --
    # :data:`LEVEL_COLORS`, the cells of the top row -- so renaming here
    # restyles the figure without renaming the measurement.
    "level_data":        "Data",
    "level_structural":  "Structural",
    "level_functional":  "Functional",
    "level_behavioral":  "Behavioral",
    # The same four levels where they are a *tick* -- the group under each set
    # of bars in panels 1 and 2. ``None`` means "whatever ``level_*`` says", so
    # a preset that shortens the names shortens these with them and only a
    # preset that wants the two to differ has to say so. A tick sits under a
    # third of a panel while a title has a whole one over it, so this is where
    # the two part company first.
    "tick_data":         None,
    "tick_structural":   None,
    "tick_functional":   None,
    "tick_behavioral":   None,
}


@dataclass(frozen=True)
class Preset:
    """A named width, and the wording that width can carry.

    *height* is the figure height in inches, or ``None`` to let
    :func:`layout_for` derive it. *text* overrides :data:`DEFAULT_TEXT` slot by
    slot; every slot left out keeps the default wording.
    """

    width: float
    height: float | None = None
    text: dict = field(default_factory=dict)
    #: Whether the MDS panels name their four frame points, or ``None`` to take
    #: :data:`MDS_LABELS`.
    mds_labels: bool | None = None
    #: The gap between the two rows, or ``None`` to take :data:`ROW_GAP`. In the
    #: same units as that constant -- inches at :data:`REFERENCE_WIDTH`, scaled
    #: down with the text for a narrower build -- so the two numbers are
    #: comparable and a preset's value reads as "wider/tighter than standard"
    #: rather than as a measurement of this preset's own printed figure.
    row_gap: float | None = None


#: The named widths. ``--width`` also takes a bare number, so these are the two
#: that have been looked at rather than the only two that work.
#:
#: **full** is the figure as it stands: a 19 in row of nine panels, for a slide
#: or a poster, and the one the tracked PNGs are drawn at.
#:
#: **page** is 7.5 in -- a letter page inside 1 in margins, so a figure spanning
#: the text block of a two-column paper or the full width of a one-column one.
#: Its wording is the short form throughout: at 7 pt under a 1.8 in panel the
#: default labels collide, and the y label in particular has 77 pt of panel
#: height to sit in, which "Procrustes Disparity" needs 80 for. The score's name
#: is in the caption, as its direction already is. Its MDS panels are unlabelled
#: for the same reason: at this size the four frame-point labels take more of the
#: panel than the mixtures they name, and the key beside them already says which
#: mixture each colour is.
PRESETS: dict[str, Preset] = {
    "full": Preset(width=19.0),
    "page": Preset(width=7.5, 
                    mds_labels=False, 
                    row_gap=0.5,
                    text={
                        # "ylabel":            "Disparity",
                        "corpora_title":     "Data Source",
                        "collection_xlabel": "Models in collection",
                        "nsweep_xlabel":     "Examples per adapter",
                        # "level_structural":  "Struct",
                        # "level_functional":  "Funct",
                        # "level_behavioral":  "Behav",
                        "tick_data":         None,
                        "tick_structural":   "Struct.",
                        "tick_functional":   "Func.",
                        "tick_behavioral":   "Behav.",
                    }),
}

#: The width every size in this file is written for, in inches. A build at any
#: other width is these numbers read through :func:`layout_for`, so there is one
#: place a size is stated and one rule for how it moves.
REFERENCE_WIDTH = PRESETS["full"].width

#: The two rows' heights at :data:`REFERENCE_WIDTH`, each split into the part
#: that is panel and the part that is text. They scale by different amounts and
#: that is the whole reason for the split: the panels shrink with the width,
#: while the band of title, axis label and ticks shrinks only as fast as the
#: text does -- which, at the floor, is not at all.
#:
#: The top row's panels are square by construction (``crosslevel_panel`` sets an
#: equal aspect), so height beyond what the panel width asks for is whitespace,
#: not a bigger panel. That is why the rule is derived rather than free, and why
#: a narrow build comes out *taller* than a proportional scaling would make it
#: without being taller than it can use. ``--height`` overrides the result.
TOP_PANEL_H, TOP_TEXT_H = 3.2, 0.9
BOTTOM_PANEL_H, BOTTOM_TEXT_H = 2.7, 0.9

#: Below this width the key stops naming all sixteen mixtures and names the
#: seven landmarks instead -- the three vertices, the centre and the three edge
#: midpoints, which are the points whose position a reader can state without
#: counting grid steps. Sixteen labels around a 1.3 in triangle do not fit at
#: any size that can also be read, and a key that cannot be read is worse than a
#: key that names less. The four MDS panels are unaffected: they have labelled
#: the vertices and the centre alone at every width, and that set is already a
#: subset of this one.
KEY_THIN_BELOW_WIDTH = 12.0
THINNED_KEY_LABELS = VERTEX_LABELS + (CENTRE_LABEL,) + EDGE_MIDPOINT_LABELS

#: The bar panels' legend columns, wide and narrow. Four model names in two
#: columns need about 1.8 in, which is the whole panel at 7.5 in wide, so the
#: narrow build stacks them into one column and spends the height instead --
#: there is room above the bars on a log axis and none beside them.
LEGEND_NCOL = {"models": (2, 1), "corpora": (3, 1)}


def text_for(preset: str | None, overrides: dict | None = None) -> dict:
    """:data:`DEFAULT_TEXT`, with *preset*'s wording and then *overrides* on top.

    *preset* is a key of :data:`PRESETS` or ``None`` for a bare ``--width``,
    which takes the default wording -- an arbitrary width has no table of its
    own, and inheriting the full-size words is the honest default: they are the
    words the figure means, and a shorter form is a concession to a width
    someone has actually looked at.
    """
    words = dict(DEFAULT_TEXT)
    if preset is not None:
        words.update(PRESETS[preset].text)
    for slot, value in (overrides or {}).items():
        if slot not in DEFAULT_TEXT:
            raise SystemExit(
                f"unknown text slot {slot!r}. The figure names "
                f"{len(DEFAULT_TEXT)} pieces of text: "
                + ", ".join(sorted(DEFAULT_TEXT))
            )
        words[slot] = value
    return words


@dataclass(frozen=True)
class Layout:
    """Every size the figure draws at, for one width.

    Built by :func:`layout_for` rather than written down: there is one set of
    sizes in this file -- the ones at :data:`REFERENCE_WIDTH` -- and this is
    what a build at any other width reads them through.
    """

    width: float
    height: float
    top_h: float
    bottom_h: float
    row_gap: float
    font: float
    mixture_label: float
    key_label_offset: float
    point_label_pad: float
    title_pad: float
    marker_size: float
    key_pad: float
    key_vertex_scale: float
    key_label_max_extra: float
    key_top_room: bool
    mds_labels: bool
    sweep_lw: float
    sweep_ms: float
    axis_lw: float
    #: Which mixtures the key names, ``()`` for all of them, and which of those
    #: keep the position they are given instead of being pushed clear of their
    #: neighbours -- the second a label -> side mapping, so a pinned label is
    #: put somewhere rather than only held still.
    key_labels: tuple = ()
    key_fixed_labels: dict = field(default_factory=dict)

    @property
    def mds_label_points(self) -> tuple:
        """Which mixtures an MDS panel names -- none, when labels are off."""
        return MDS_LABEL_POINTS if self.mds_labels else ()

    @property
    def narrow(self) -> bool:
        """Whether this build is past the point where the key has to thin."""
        return self.width < KEY_THIN_BELOW_WIDTH

    def ncol(self, panel: str) -> int:
        """Legend columns for a bar panel, wide or narrow."""
        return LEGEND_NCOL[panel][1 if self.narrow else 0]


def layout_for(width: float, row: str = "both",
               height: float | None = None,
               mds_labels: bool = MDS_LABELS,
               row_gap: float | None = None) -> Layout:
    """The sizes for a figure *width* inches wide drawing *row*.

    Two scalings, not one, and the difference between them is the whole point.
    **Geometry** -- the panels themselves -- scales with the width. **Text**, and
    everything that has to stay in proportion to text rather than to a panel,
    scales with the font, which stops at :data:`FONT_SIZE_FLOOR`. At the full
    width the two are the same number and this is an identity; below it they
    diverge, and the figure trades panel area for legible type.

    Marker area goes as the *square* of the font scale, because an area and a
    point size are not the same kind of number: a marker meant to read as
    roughly a capital letter's worth of ink has to follow the letter's area.

    *height* overrides the derived total, keeping the two rows' proportions.
    *mds_labels* is carried through rather than derived: whether a panel names
    its frame points is a choice about what the figure says, not about how much
    room it has. *row_gap* replaces :data:`ROW_GAP` and is read in the same
    units -- inches at :data:`REFERENCE_WIDTH`, scaled with the text here -- so
    that a value written next to a preset's width means the same thing as the
    constant it overrides.
    """
    scale = width / REFERENCE_WIDTH
    font = max(FONT_SIZE * scale, FONT_SIZE_FLOOR)
    text = font / FONT_SIZE
    narrow = width < KEY_THIN_BELOW_WIDTH

    top_h = TOP_PANEL_H * scale + TOP_TEXT_H * text
    bottom_h = BOTTOM_PANEL_H * scale + BOTTOM_TEXT_H * text
    gap = ((ROW_GAP if row_gap is None else row_gap) * text
           if row == "both" else 0.0)
    drawn = [h for h, want in ((top_h, row in ("top", "both")),
                               (gap, row == "both"),
                               (bottom_h, row in ("bottom", "both"))) if want]
    if height is not None:
        if height <= 0:
            raise SystemExit(f"--height must be positive, got {height}")
        k = height / sum(drawn)
        top_h, bottom_h, gap = top_h * k, bottom_h * k, gap * k
        drawn = [h * k for h in drawn]

    return Layout(
        width=width,
        height=sum(drawn),
        top_h=top_h,
        bottom_h=bottom_h,
        row_gap=gap,
        font=font,
        mixture_label=max(MIXTURE_LABEL_SIZE * scale, MIXTURE_LABEL_FLOOR),
        key_label_offset=KEY_LABEL_OFFSET * text,
        point_label_pad=max(POINT_LABEL_PAD * text, POINT_LABEL_PAD_FLOOR),
        title_pad=TITLE_PAD * text,
        marker_size=MIXTURE_MARKER_SIZE * text ** 2,
        key_pad=(KEY_PAD_THINNED if width < KEY_THIN_BELOW_WIDTH else KEY_PAD),
        key_vertex_scale=text,
        # The key's cell is much taller than the triangle at the full width, so
        # "Group 1" sits in slack the title never reaches. At a near-square cell
        # there is no slack and the name runs past the top of the axes -- so the
        # axes grows to hold it, which keeps the key's title level with the four
        # beside it instead of raising it over the name.
        key_top_room=narrow,
        mds_labels=mds_labels,
        key_label_max_extra=KEY_LABEL_MAX_EXTRA * text * (1.4 if narrow else 1.0),
        # The centre is the one sampled mixture with no radius to fan along, so
        # it borrows the emptiest direction it can find -- which, on a small
        # triangle, is the one the nearest edge midpoint's label is already
        # travelling down. Both then run to the cap and stack on each other.
        # Straight up is no better: it is the next row's own point. So it is put
        # to the side, into the band between two rows, and held there.
        key_fixed_labels=({CENTRE_LABEL: KEY_CENTRE_LABEL_SIDE}
                          if narrow and KEY_CENTRE_LABEL_SIDE else {}),
        sweep_lw=SWEEP_LINEWIDTH * text,
        sweep_ms=SWEEP_MARKER_SIZE * text,
        axis_lw=max(AXIS_LINEWIDTH * text, AXIS_LINEWIDTH_FLOOR),
        key_labels=(THINNED_KEY_LABELS if width < KEY_THIN_BELOW_WIDTH else ()),
    )


def comma_label(model_id: str) -> str:
    """``'25,50,25'`` -- a mixture written as the vector it is.

    The project's usual form is ``25/50/25``. Commas are this figure's, for both
    the MDS point labels and the key: nine panels in, a reader meets these trios
    beside axis values and bar heights, where a slash reads as a fraction.
    """
    return ",".join(mixture_label(model_id).split("/"))


# ── The four levels, in every form the nine panels need them ──────────────────

@dataclass(frozen=True)
class Level:
    """One taxonomy level, pinned to one perspective, named three ways.

    The same measurement is addressed differently by each of the three data
    sources this figure reads, and carrying all three names in one place is what
    keeps a bar in panel 1 and a curve in panel 3 the same thing:

    *suite_level* / *surrogate* / *metric*
        the exact cell ``run_suite`` is asked to build for the top row, and the
        substring ``simplex_runs.Perspective`` matches for panels 1-2. Exact
        for the top row because that run is one known model; substring for
        panels 1-2 because those twelve runs name their surrogates after their
        own architectures.
    *sweep_key*
        the ``perspective`` column of ``group_size_scores.csv`` and
        ``nsweep_scores.csv``, which panels 3-4 read.
    """

    label: str
    suite_level: str
    surrogate: str
    metric: str
    sweep_key: str

    @property
    def color(self) -> str:
        return LEVEL_COLORS[self.label]

    def perspective(self) -> runs.Perspective:
        """This level as a :mod:`src.plots.simplex_runs` perspective."""
        return runs.Perspective(self.label, self.suite_level, self.metric,
                                surrogate=self.surrogate, color=self.color)


#: The behavioral level read two ways. See the module docstring: the sampled
#: read has no signal left on the dense collection-size pool, so the default
#: figure uses greedy for the two sweep panels and R=16 everywhere else, and
#: ``--behavioral greedy`` uses greedy throughout.
_BEHAVIORAL = {
    "sampled": Level("Behavioral", "behavioral", "R=16 · per query",
                     "cosine", "behavioral"),
    "greedy": Level("Behavioral", "behavioral", "greedy · per generation",
                    "cosine", "behavioral_greedy"),
}

#: The three levels whose perspective does not depend on the decoding choice.
_FIXED = [
    Level("Data", "dataset_embedding", "dataset text · mean",
          "euclidean", "dataset_embedding"),
    Level("Structural", "structural", "output projections",
          "cosine", "structural_all_o"),
    Level("Functional", "functional", "layers (reference)",
          "cosine", "functional_all"),
]


def level_name(level: Level, txt: dict) -> str:
    """*level* as the reader sees it, which a preset may have shortened.

    ``Level.label`` stays the name everything is addressed *by* -- the colour
    table, the cells of the top row, the bar groups' order -- so a preset that
    types "Behav" under a 1.8 in panel renames the words on the page and nothing
    underneath them.
    """
    return txt[f"level_{level.label.lower()}"]


def level_tick(level: Level, txt: dict) -> str:
    """*level* as a bar panel's x tick, which a preset may name separately.

    Falls back to :func:`level_name`: the tick slots exist so a preset *can*
    split the two, not so it has to keep them in step.
    """
    return txt[f"tick_{level.label.lower()}"] or level_name(level, txt)


def levels_for(decoding: str) -> list[Level]:
    """The four levels with the behavioral one read under *decoding*."""
    return _FIXED + [_BEHAVIORAL[decoding]]


#: ``--behavioral`` -> (decoding for the top row and panels 1-2, decoding for
#: the sweep panels 3-4).
DECODINGS = {"mixed": ("sampled", "greedy"), "greedy": ("greedy", "greedy")}


# ── Top row: the cross-level MDS panels ───────────────────────────────────────

#: The run the top row is drawn from, copied from
#: ``figures/simplex3_olmo2/make_figures.py`` -- the same base model, the same
#: query draw and the same corpus and mixture filters, so the panels here are
#: that suite's panels and not a second run of a similar experiment. See that
#: driver for why each filter is not optional.
TOP_BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
TOP_DATASETS = ["yahoo"]
TOP_DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
            "prompt_format_id": "ea27ccee"}

#: The training draw the simplex3 adapters were fine-tuned on. Not optional, and
#: not a filter ``figures/simplex3_olmo2/make_figures.py`` needed when it was
#: written: the nsweep tree has since put 90 further draws of this corpus under
#: this base model, so a scan filtered only by corpus and mixture returns 1444
#: models and ``n_expected`` trips. Note this is the *training* draw, a
#: different thing from :data:`TOP_DRAW`, which is the query draw both inference
#: stages read.
TOP_TRAIN_DRAW = (1000, 0)

#: The LoRA rank the simplex3 adapters were trained at. Not optional, and not a
#: filter this driver needed when it was written: the rank sweep has since
#: trained the same 16 mixtures of this corpus at eight ranks under this base
#: model and this training draw, so a scan filtered only by corpus, draw and
#: mixture returns 128 models and ``n_expected`` trips. 16 is the rank every
#: suite that predates the sweep ran at -- ``Suite.lora_rank``'s default -- so
#: pinning it here is what keeps this panel the same measurement it was.
TOP_LORA_RANK = 16

#: Decoder layers of :data:`TOP_BASE_MODEL`, from the checkpoint's own config.
#: The functional reference surrogate names ``N_LAYERS + 1`` rows, so the exact
#: label the suite must be asked for is derived from this rather than written
#: out -- a wrong count then fails loudly in ``compact_selection`` instead of
#: quietly selecting a different row.
TOP_N_LAYERS = 16

#: Left to right after the mixture key: the reading order of the taxonomy, from
#: what the model was trained on, through what its weights became and what its
#: activations do, to what it says. It is also the order panels 1-2 group their
#: bars in, so a column of the top row sits above its own bar group.
PANEL_ORDER = ["Data", "Structural", "Functional", "Behavioral"]


def _exact_surrogate(level: Level) -> str:
    """*level*'s surrogate as :data:`TOP_BASE_MODEL`'s own grid names it.

    ``run_suite``'s ``select`` is an exact match, not a substring, so the two
    labels that carry a number this model decides have to be completed here.
    """
    if level.label == "Functional":
        return f"all {TOP_N_LAYERS + 1} layers (reference)"
    if level.label == "Data":
        return "dataset text · mean · n1000_s00"
    return level.surrogate


def top_row_cells(levels: list[Level], cache_root=None, no_cache=False):
    """``({label: DistanceMatrix}, model_ids)`` for the top row's four panels.

    Runs the suite with one perspective per level and the closer off, so nothing
    is written and every cell comes back from ``06_pairwise`` if it is there.
    """
    from src.experiments.data_simplex_spec import SPECS
    from src.plots import simplex_suite as suite

    select = {lv.suite_level: [(_exact_surrogate(lv), lv.metric)] for lv in levels}
    per_level, ids = suite.run_suite(
        base_model=TOP_BASE_MODEL,
        draw=TOP_DRAW,
        outdir=str(HERE),
        cache_root=cache_root,
        levels=list(select),
        no_cache=no_cache,
        surrogates=False,
        select=select,
        crosslevel_only=True,
        closer=False,
        datasets=TOP_DATASETS,
        train_draw=TOP_TRAIN_DRAW,
        mixtures=SPECS["yahoo"].mixture_pcts(),
        lora_rank=TOP_LORA_RANK,
        source="figures/figure2/make_figures.py",
    )
    cells = {}
    for lv in levels:
        grid = per_level[lv.suite_level]
        got = [c for c in grid.values() if c is not None and not isinstance(c, str)]
        if len(got) != 1:
            raise SystemExit(
                f"{lv.label}: asked for one perspective, the suite built "
                f"{len(got)}. A panel drawn from the wrong one would look "
                f"right and mean something else."
            )
        cells[lv.label] = got[0]
    return cells, ids


def draw_top_row(fig, gs, levels: list[Level], cells, ids,
                 sz: Layout, txt: dict) -> list:
    """The mixture key and the four MDS panels, into *gs* (1 x 5).

    Returns the five axes, in row order, for :func:`align_titles`.
    """
    by_label = {lv.label: lv for lv in levels}
    kax = fig.add_subplot(gs[0, 0])
    ternary_legend(kax, ids, label_models=True,
                   vertex_names=group_display(3), show_topics=False,
                   label_size=sz.mixture_label, vertex_size=sz.font,
                   marker_size=sz.marker_size, label_fmt=comma_label,
                   fill_points=True, pad=sz.key_pad, outline_color=AXIS_COLOR,
                   label_offset=sz.key_label_offset, avoid_collisions=True,
                   fontweight="bold", fontfamily=bold_capable_family(),
                   label_points=sz.key_labels or None,
                   marker=MIXTURE_MARKER_SHAPE,
                   frame_marker=FRAME_MARKER_SHAPE,
                   frame_marker_scale=FRAME_MARKER_SCALE,
                   vertex_scale=sz.key_vertex_scale,
                   label_max_extra=sz.key_label_max_extra,
                   fixed_points=sz.key_fixed_labels,
                   top_room=sz.key_top_room)
    kax.set_title(txt["key_title"], fontsize=sz.font, pad=sz.title_pad,
                  fontweight="bold", fontfamily=bold_capable_family())
    panels = [kax]
    for k, label in enumerate(PANEL_ORDER):
        ax = crosslevel_panel(fig.add_subplot(gs[0, k + 1]),
                              level_name(by_label[label], txt),
                              cells[label], font_size=sz.font,
                              point_label_size=sz.mixture_label, bold=True,
                              label_fmt=comma_label, axis_color=AXIS_COLOR,
                              marker_size=sz.marker_size,
                              point_label_pad=sz.point_label_pad,
                              title_pad=sz.title_pad,
                              label_points=sz.mds_label_points,
                              marker=MIXTURE_MARKER_SHAPE,
                              frame_marker=FRAME_MARKER_SHAPE,
                              frame_marker_scale=FRAME_MARKER_SCALE)
        # No ticks. An MDS coordinate has no units and no origin a reader can
        # use -- the configuration is only defined up to rotation, reflection
        # and scale, which is exactly why it is scored by Procrustes -- so the
        # numbers on these axes invite a comparison that is not there to make.
        # The axes are still named, because which plane this is remains worth
        # saying.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(txt["mds_x"])
        # Only the leftmost panel is labelled on y: the four share a meaning,
        # not a scale (they differ by an order of magnitude in absolute MDS
        # size), and repeating the name four times says nothing the first does
        # not.
        if k == 0:
            ax.set_ylabel(txt["mds_y"])
        panels.append(ax)

    return panels


def align_titles(fig, axes, passes: int = 3) -> None:
    """Put every title in *axes* on one line, at the height of the highest.

    The five titles of the top row name five panels of one row and should read
    as one line of text. Left to themselves they do not: a title sits a fixed
    pad above its own axes, and these five axes do not share a top edge --
    each is square by its own aspect, so each gives up whatever its cell has
    spare, and the four MDS panels give up more of it than the key because they
    carry an x label the key does not.

    ``Figure.align_titles`` is the obvious tool and does not do this: it aligns
    a title against the *axes* boxes of its group, which for an aspect-fixed
    panel is not where the title ended up. So the titles are measured as drawn
    and moved onto the highest of them. Measuring means drawing, and moving a
    title moves the layout under it, so it settles over a few passes.
    """
    for _ in range(passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tops = [ax.title.get_window_extent(renderer).y0 for ax in axes]
        top = max(tops)
        if top - min(tops) < 0.5:
            return
        for ax, y0 in zip(axes, tops):
            x, y = ax.title.get_position()
            ax.title.set_position(
                (x, y + (top - y0) / ax.get_window_extent().height))
            # Keep matplotlib from putting the title back where it was: its own
            # automatic placement is per axes, which is the thing being undone.
            ax._autotitlepos = False


# ── Bottom row, panels 1 and 2: the bar charts ────────────────────────────────

def _bar_values(run: runs.Run, levels: list[Level]) -> list[float]:
    """One run's disparity per level, at that corpus's own ``d = K-1``."""
    rows = runs.read_run(FIGURES_ROOT, run)
    scores = runs.level_scores(rows, "procrustes", "dK1",
                               [lv.perspective() for lv in levels])
    return [scores[lv.label] for lv in levels]


def draw_models_bars(ax, levels: list[Level], sz: Layout, txt: dict) -> None:
    """Panel 1 — the four base models on yahoo, one bar per model."""
    ax.set_title(txt["models_title"])
    series = [
        make_series(_bar_values(r, levels), label=r.model_label,
                    color=MODEL_COLORS[r.base_model])
        for r in runs.runs_for_corpus("yahoo")
    ]
    plot_grouped_bars(series, [level_tick(lv, txt) for lv in levels], ax=ax,
                      legend=True,
                      legend_kwargs={"loc": "upper left", "ncol": sz.ncol("models")},
                      savefig=False)


def draw_corpora_bars(ax, levels: list[Level], sz: Layout, txt: dict) -> None:
    """Panel 2 — the three corpora on OLMo-2-1B, one bar per corpus."""
    ax.set_title(txt["corpora_title"])
    series = [
        make_series(_bar_values(r, levels), label=r.corpus,
                    color=CORPUS_COLORS[r.corpus])
        for r in runs.runs_for_model(TOP_BASE_MODEL)
    ]
    plot_grouped_bars(series, [level_tick(lv, txt) for lv in levels], ax=ax,
                      legend=True,
                      legend_kwargs={"loc": "upper left", "ncol": sz.ncol("corpora")},
                      savefig=False)


# ── Bottom row, panels 3 and 4: the two sweeps ────────────────────────────────

#: Panel -> ``(csv, x column, y column, x cap)``. The y column is the disparity
#: reading chosen for that sweep; see the module docstring for why. The cap
#: drops the sweep's largest draw sizes from the panel; ``None`` keeps every x in
#: the file. The nsweep ran one value past 2000 -- 5000 -- and it is cut here
#: rather than in the CSV, so the sweep's own figure keeps it.
#:
#: The panel's title and axis label are **not** here: they are wording, and
#: wording depends on the width, so they live in :data:`DEFAULT_TEXT` under the
#: slots ``{key}_title`` and ``{key}_xlabel``. The title still names what the
#: panel varies and the label the unit it varies it in -- neither is derived
#: from the other, they are just both text now.
SWEEPS = {
    "collection": (FIGURES_ROOT / "simplex_collection_size" / "group_size_scores.csv",
                   "n", "disparity_B", None),
    "nsweep": (FIGURES_ROOT / "simplex3_nsweep_olmo2_nsweep" / "nsweep_scores.csv",
               "n_samples", "disparity_requested", 2_000),
}


def sweep_series(path: Path, xcol: str, ycol: str, sweep_key: str,
                 xmax: int | None = None):
    """``(xs, median, q1, q3)`` over the replicates at each x, for one level."""
    by_x: dict[int, list[float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["perspective"] != sweep_key:
                continue
            x = int(row[xcol])
            if xmax is not None and x > xmax:
                continue
            by_x.setdefault(x, []).append(float(row[ycol]))
    if not by_x:
        raise SystemExit(f"{path.name} has no rows for perspective {sweep_key!r}")
    xs = np.array(sorted(by_x))
    vals = [np.asarray(by_x[x], dtype=float) for x in xs]
    return (xs,
            np.array([np.median(v) for v in vals]),
            np.array([np.percentile(v, 25) for v in vals]),
            np.array([np.percentile(v, 75) for v in vals]))


def draw_sweep(ax, which: str, levels: list[Level], legend: bool,
               sz: Layout, txt: dict) -> None:
    """Panels 3 and 4 — all four levels on one axes, median plus IQR band."""
    path, xcol, ycol, xmax = SWEEPS[which]
    for lv in levels:
        xs, med, q1, q3 = sweep_series(path, xcol, ycol, lv.sweep_key, xmax)
        ax.fill_between(xs, q1, q3, color=lv.color, alpha=BAND_ALPHA, lw=0,
                        zorder=2)
        ax.plot(xs, med, color=lv.color, ls=SWEEP_LINESTYLE,
                marker=SWEEP_MARKER, ms=sz.sweep_ms, lw=sz.sweep_lw,
                label=level_name(lv, txt), zorder=3)
    ax.set_xscale("log")
    ax.set_title(txt[f"{which}_title"])
    ax.set_xlabel(txt[f"{which}_xlabel"])
    if legend:
        ax.legend(loc=SWEEP_LEGEND_LOC, ncol=1 if sz.narrow else 2)


# ── Assembly ──────────────────────────────────────────────────────────────────

#: The gap between the two rows, **in inches**. The rows read as separate
#: statements -- one figure repeated at four levels, then four ways that figure
#: moves -- so they get far more air between them than a default layout leaves.
#: This is the one knob for it, at :data:`REFERENCE_WIDTH`; a narrower build
#: scales it with the text rather than with the width, because what it separates
#: is two statements and the reader's sense of that gap is set by the type.
#:
#: It is spent as a blank spacer row in the outer grid, not as the constrained
#: layout's ``hspace``: each row here is a *nested* gridspec, and the engine's
#: ``hspace`` is inert across a nesting boundary -- setting it moves nothing.
ROW_GAP = 0.4

#: Shared y range for the whole bottom row, per scale. Fixed rather than fitted
#: so any two builds of this figure — the ``--behavioral`` and ``--yscale``
#: variants especially — can be laid side by side. Both cover every value all
#: four panels draw under either ``--behavioral``: the functional level on the
#: yahoo simplex at the bottom, the behavioral level at a ten-example training
#: draw at the top.
#:
#: The two scales answer different questions and neither is a correction of the
#: other. **Log** is the only one on which all four levels are legible at once:
#: the row spans more than two orders of magnitude, and on a linear axis the
#: data, structural and functional values all sit inside the bottom twentieth of
#: it. **Linear** is the only one on which the *differences* are what they look
#: like — a bar twice as tall is twice the disparity — which is what a reader
#: comparing behavioral against everything else actually wants to see, at the
#: price of the other three levels collapsing onto the axis.
#: The linear range is a flat 0 to 1: the score's own full range, so a bar's
#: height is readable as a fraction of the worst possible disparity without
#: consulting the axis. The log range cannot start at 0 — a log axis has no zero
#: — so it starts just under the smallest value any panel draws and ends just
#: over 1.
YLIMS = {"log": (4e-3, 1.2), "linear": (0.0, 1.0)}


def draw_bottom_row(fig, gs, bar_levels: list[Level],
                    sweep_levels: list[Level], sz: Layout, txt: dict,
                    yscale: str = "log") -> list:
    """The four score panels, into *gs* (1 x 4), sharing one y axis."""
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2, 3)]

    draw_models_bars(axes[0], bar_levels, sz, txt)
    draw_corpora_bars(axes[1], bar_levels, sz, txt)
    draw_sweep(axes[2], "collection", sweep_levels, True, sz, txt)
    draw_sweep(axes[3], "nsweep", sweep_levels, False, sz, txt)

    for k, ax in enumerate(axes):
        ax.set_yscale(yscale)
        ax.set_ylim(*YLIMS[yscale])
        if k == 0:
            ax.set_ylabel(txt["ylabel"])
        else:
            ax.tick_params(labelleft=False)
    # The taxonomy names sit flat. Four of them across a panel this wide fit
    # horizontally with room to spare, and an angle costs a reader a head-tilt
    # for nothing -- it is worth paying only when the labels would otherwise
    # collide, which at this width they do not.
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=0)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("center")
    return axes


def _apply_style(sz: Layout) -> None:
    """The house style, then one text size over the top of it.

    ``set_style`` scales tick labels and legends to three quarters of the base
    size and leaves titles at it, which is right for a panel that is a figure in
    its own right. Here nine panels from three sources sit in one frame, and a
    size difference between them reads as emphasis rather than as provenance --
    so every piece of text is set to one size, at one weight -- *sz*\'s, which
    is :data:`FONT_SIZE` at the full width and no smaller than
    :data:`FONT_SIZE_FLOOR` at any other.

    The weight needs the family changed with it. ``set_style`` prefers Libre
    Franklin, which is a *variable* font that matplotlib registers at exactly one
    weight -- Thin -- so ``fontweight="bold"`` against it is a silent no-op that
    renders Thin. :func:`~src.plots.config.bold_capable_family` probes the font
    manager for the first family in the stack that actually ships more than one
    weight, and that is the one named here. Drop a static
    ``LibreFranklin-Bold.ttf`` beside the variable file and this starts
    returning Libre Franklin with no change on this side.
    """
    set_style("two_col_full")
    matplotlib.rcParams.update({
        "font.size": sz.font, "axes.titlesize": sz.font,
        "axes.labelsize": sz.font, "xtick.labelsize": sz.font,
        "ytick.labelsize": sz.font, "legend.fontsize": sz.font,
        "legend.title_fontsize": sz.font, "figure.titlesize": sz.font,
        "font.sans-serif": [bold_capable_family(), "DejaVu Sans"],
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": sz.axis_lw,
    })


def _embolden(fig) -> None:
    """Tick labels and legend entries to the same weight as everything else.

    ``font.weight`` reaches text matplotlib creates from the rc defaults, but
    tick labels and legend entries are not among them: there is no
    ``xtick.labelweight`` rc, and a legend builds its texts from the handles'
    labels. Both are therefore set here, after every axes is drawn, rather than
    left as the one unbolded thing in a figure that is otherwise one weight.
    """
    family = bold_capable_family()
    for ax in fig.axes:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_fontfamily(family)
        legend = ax.get_legend()
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontweight("bold")
                txt.set_fontfamily(family)


def build(row: str, decoding: str, yscale: str = "log",
          cache_root=None, no_cache=False, width: float | None = None,
          height: float | None = None, preset: str | None = "full",
          text_overrides: dict | None = None,
          mds_labels: bool | None = None,
          row_gap: float | None = None) -> Path:
    """Draw the requested row(s) and save. Returns the path written.

    *preset* names a :data:`PRESETS` entry, which supplies both the width and
    the wording; *width* overrides its inches and *height* the derived total.
    A bare width with no preset (``preset=None``) draws at the default wording.
    *mds_labels* overrides whether the MDS panels name their frame points, and
    *row_gap* the space between the two rows; each falls back to the preset's
    value and then to the module constant.
    """
    if preset is not None and preset not in PRESETS:
        raise SystemExit(f"unknown preset {preset!r}; "
                         f"choose from {list(PRESETS)} or pass a width")
    if width is None:
        width = PRESETS[preset].width if preset else REFERENCE_WIDTH
    if height is None and preset is not None:
        height = PRESETS[preset].height
    if mds_labels is None:
        chosen = PRESETS[preset].mds_labels if preset is not None else None
        mds_labels = MDS_LABELS if chosen is None else chosen
    if row_gap is None and preset is not None:
        row_gap = PRESETS[preset].row_gap
    sz = layout_for(width, row=row, height=height, mds_labels=mds_labels,
                    row_gap=row_gap)
    txt = text_for(preset, text_overrides)

    bar_dec, sweep_dec = DECODINGS[decoding]
    bar_levels = levels_for(bar_dec)
    sweep_levels = levels_for(sweep_dec)
    by_label = {lv.label: lv for lv in bar_levels}
    top_levels = [by_label[label] for label in PANEL_ORDER]

    want_top = row in ("top", "both")
    want_bottom = row in ("bottom", "both")

    # Fetched **before** the style is applied, not after: run_suite calls
    # set_style itself, and set_style begins with rcdefaults(). Fetching second
    # would silently reset the rcParams below, and the bottom row would draw at
    # the preset's 8 pt while the top row drew at the size it was passed
    # explicitly -- the exact split this is here to remove.
    cells = ids = None
    if want_top:
        cells, ids = top_row_cells(top_levels, cache_root, no_cache)

    _apply_style(sz)

    #: The key column is wider than a panel — it carries the mixture labels and
    #: three vertex names — and the top row's panels are square, so the two rows
    #: are given different heights rather than one shared one. Both heights, and
    #: the gap, come from :func:`layout_for`.
    both = want_top and want_bottom
    heights = [h for h, want in [(sz.top_h, want_top), (sz.row_gap, both),
                                 (sz.bottom_h, want_bottom)] if want]
    fig = plt.figure(figsize=(sz.width, sum(heights)), layout="constrained")
    outer = fig.add_gridspec(len(heights), 1, height_ratios=heights)

    r = 0
    top_axes = None
    if want_top:
        top_axes = draw_top_row(fig, outer[r].subgridspec(1, 5, width_ratios=[1.28] + [1.0] * 4),
                     top_levels, cells, ids, sz, txt)
        r += 2 if both else 1
    if want_bottom:
        draw_bottom_row(fig, outer[r].subgridspec(1, 4), bar_levels,
                        sweep_levels, sz, txt, yscale)

    _embolden(fig)
    if top_axes:
        align_titles(fig, top_axes)

    stem = f"fig_figure2_{row}" if row != "both" else "fig_figure2"
    if decoding != "mixed":
        stem += f"_{decoding}"
    if yscale != "log":
        stem += f"_{yscale}"
    # The full-size build keeps the names it has: those nine PNGs are tracked,
    # and a suffix on them would read as nine new figures beside nine stale
    # ones. Every other shape says its shape in its name -- the preset's, or
    # the width itself for one that has no name.
    if preset not in (None, "full") or sz.width != REFERENCE_WIDTH:
        named = preset if (preset not in (None, "full")
                           and sz.width == PRESETS[preset].width) else None
        stem += f"_{named}" if named else f"_w{sz.width:g}"
    if height is not None:
        stem += f"h{sz.height:g}"
    out = HERE / f"{stem}.png"
    save_figure(fig, str(out))
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", choices=["both", "top", "bottom"], default="both",
                    help="which row to draw; each is a standalone figure")
    ap.add_argument("--behavioral", choices=list(DECODINGS), default="mixed",
                    help="'mixed' reads R=16 · per query in the top row and the "
                         "two bar panels and greedy in the two sweep panels; "
                         "'greedy' reads greedy everywhere")
    ap.add_argument("--yscale", choices=list(YLIMS), default="log",
                    help="the bottom row's shared y axis. 'log' keeps all four "
                         "levels legible across two orders of magnitude; "
                         "'linear' makes the differences literal at the cost of "
                         "collapsing everything but the behavioral level")
    ap.add_argument("--width", default="full",
                    help="a preset name (" + " | ".join(PRESETS) + ") or a "
                         "width in inches. A preset carries both a width and "
                         "the wording that width can carry; a bare number "
                         "takes the default wording. Sizes scale with the "
                         f"width down to a {FONT_SIZE_FLOOR:g} pt floor, below "
                         "which the panels shrink and the type does not")
    ap.add_argument("--height", type=float, default=None,
                    help="figure height in inches, overriding the derived one. "
                         "The rows keep their proportions; the derived height "
                         "is what the panels can use, so this is for fitting a "
                         "space rather than for making a panel bigger")
    ap.add_argument("--mds-labels", dest="mds_labels", default=None,
                    action=argparse.BooleanOptionalAction,
                    help="whether the four MDS panels name their frame points "
                         "-- the three vertices and the centre. On by default; "
                         "--no-mds-labels drops them, which buys back the room "
                         "they take inside a narrow panel and leaves the key "
                         "beside them to name the mixtures")
    ap.add_argument("--row-gap", dest="row_gap", type=float, default=None,
                    help="the space between the two rows, overriding the "
                         f"preset's and the standard {ROW_GAP:g} in. Given in "
                         f"inches at the {REFERENCE_WIDTH:g} in reference "
                         "width and scaled down with the type, like the "
                         "constant it replaces -- so a narrower build draws "
                         "less than the number says. Ignored by --row top and "
                         "--row bottom, which have no second row to clear")
    ap.add_argument("--label", action="append", default=[], metavar="SLOT=TEXT",
                    help="override one title or axis label, repeatable. Slots: "
                         + ", ".join(sorted(DEFAULT_TEXT)))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the top row's matrices, ignoring stored "
                         "results (still writes them back)")
    args = ap.parse_args()

    # A preset name or a number, and the difference decides the wording as well
    # as the width -- see text_for.
    if args.width in PRESETS:
        preset, width = args.width, None
    else:
        try:
            preset, width = None, float(args.width)
        except ValueError:
            raise SystemExit(
                f"--width takes a preset name ({', '.join(PRESETS)}) or a "
                f"number of inches, not {args.width!r}")
        if width <= 0:
            raise SystemExit(f"--width must be positive, got {width}")

    overrides = {}
    for item in args.label:
        slot, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--label takes SLOT=TEXT, got {item!r}")
        overrides[slot.strip()] = value

    print(f"wrote {build(args.row, args.behavioral, args.yscale, args.cache_root, args.no_cache, width, args.height, preset, overrides, args.mds_labels, args.row_gap)}")


if __name__ == "__main__":
    main()
