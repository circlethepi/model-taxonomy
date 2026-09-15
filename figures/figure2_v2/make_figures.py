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

Typography
----------
Every piece of text in the figure is one size and one weight — see
:data:`FONT_SIZE` and :func:`_apply_style`. The three sources this figure is
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
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
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
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.plots import make_series, plot_grouped_bars, save_figure, set_style  # noqa: E402
from src.plots.config import bold_capable_family  # noqa: E402
from src.plots import simplex_runs as runs  # noqa: E402
from src.plots.simplex import crosslevel_panel, ternary_legend  # noqa: E402
from src.plots.simplex import group_display, mixture_label  # noqa: E402
from src.analysis.baselines import (band_quantiles,  # noqa: E402
                                    disparity_null_analytic,
                                    disparity_null_direct,
                                    truth_singular_values, uninformed_band)
from src.analysis.ground_truth import simplex_vertices  # noqa: E402
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.plots.figures import (PERMUTATION_COLOUR, PERMUTATION_LABEL,  # noqa: E402
                               UNINFORMED_COLOUR, UNINFORMED_LABEL,
                               draw_permutation_band,
                               draw_uninformed_band, load_baseline_table)

FIGURES_ROOT = REPO_ROOT / "figures"

# ── The two reference bands ───────────────────────────────────────────────────
#   Defined here on first use, per project convention; the full definitions live
#   in `docs/terminology.md` and `docs/notes/chance_baselines.md`.
#
#   **Uninformed band** — the 5-95 interval of the **structure null**: replace
#   the taxonomy with a structureless configuration drawn from a generator, keep
#   the truth, and score. It answers "what would any arrangement of this many
#   points have scored?".
#
#   **Permutation band** — the 5-95 interval of the **label null**: keep both
#   real geometries and permute which model is which. It answers "what would
#   *these two* geometries have scored had the correspondence been shuffled?".
#   The note spells this one "label null"; the figure says "permutation". They
#   are one concept under two names, not two nulls.
#
#   Drawing both is the point. The structure null is scored directly as a
#   configuration and never passes through MDS, so on this figure's
#   MDS-mediated disparity it is mildly optimistic; the label null carries the
#   real geometries through the real scoring path and has no such gap. Where the
#   two agree, that gap is small — which is itself the finding.

#: Which :data:`SPECS` grid each corpus was trained on. The uninformed band
#: needs the truth configuration, and for these suites it is known *exactly*:
#: they are complete symmetric grids on a simplex, so nothing is estimated.
SPEC_FOR_CORPUS = {"yahoo": "yahoo", "dolly": "dolly", "oasst1": "oasst1"}

#: The collection-size sweep's simplex. Named rather than inferred: a wrong
#: inference silently selects the wrong entry of the baseline table, which is a
#: band that looks entirely plausible and is about a different experiment.
COLLECTION_K = 3
COLLECTION_D = COLLECTION_K - 1

#: Monte Carlo draws for an exact band. The analytic (gaussian) path is cheap
#: enough to oversample; the dirichlet path runs the real scoring loop, so it
#: gets fewer — the quantiles converge long before the band is a pixel wide
#: either way.
_BAND_MC = {"gaussian": 40_000, "dirichlet": 4_000}

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
    "allenai/OLMo-2-0425-1B-Instruct":      "#C6DBEF",
    "Qwen/Qwen3.5-4B":                      "#6BAED6",
    "meta-llama/Llama-3.1-8B-Instruct":     "#2171B5",
    "mistralai/Mistral-Nemo-Instruct-2407": "#08306B",
}

#: Corpus -> colour, in order of increasing distance from a topic mixture:
#: yahoo mixes topics, dolly mixes instruction tasks, oasst1 mixes languages.
#: Teal, green and turquoise: one neighbourhood of the wheel, like the model
#: ramp beside it, but three hues at comparable weight rather than a ramp --
#: nothing orders three corpora, so nothing here should look ordered.
CORPUS_COLORS = {"yahoo": "#17807A", "dolly": "#4C9A2A", "oasst1": "#5FD3D0"}

#: One line style for every level in panels 3-4. Hue alone separates the four
#: curves; they are far enough apart vertically that a dash pattern per level
#: added texture without adding information.
SWEEP_LINESTYLE = "-"

#: The curves carry the panel, so they are drawn heavier than a default line.
SWEEP_LINEWIDTH = 2.1
SWEEP_MARKER_SIZE = 7

#: The IQR band is context for its curve, not a second series. Faint enough that
#: two overlapping bands do not read as a third colour.
BAND_ALPHA = 0.09

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
FONT_SIZE = 13

#: The one exception: every label that names a mixture, in the key and in the
#: four MDS panels alike. These sit among the points they name rather than
#: outside an axes, and sixteen of them share the key's triangle, so at FONT_SIZE
#: they crowd the configuration they are there to explain. One value for all of
#: them -- the same text saying the same thing in five panels -- and it is the
#: size the densest of those panels can carry.
MIXTURE_LABEL_SIZE = 8

#: The names of the two axes every top-row panel is drawn in.
MDS_AXES = ("MDS 1", "MDS 2")

#: The *triangle* is drawn smaller than the panels beside it -- it is a legend,
#: not a fifth measurement -- by widening the margin around it inside an axes
#: cell the row's grid fixes. Nothing else in the key shrinks with it: the
#: labels, the vertex names and the markers are text and marks, sized against
#: the rest of the figure rather than against the triangle they sit on.
KEY_PAD = 0.3
KEY_VERTEX_SIZE = FONT_SIZE

#: The labels are pushed further out along their radii than the default 9.5 pt,
#: because the triangle they sit on is smaller while they are not: the sixteen
#: points are closer together in inches, so the labels need more room to fan
#: into. The vertex names clear this on their own -- see ``ternary_legend``.
KEY_LABEL_OFFSET = 19.0
KEY_TITLE = "Dataset Mixture"

#: One marker area for every mixture point in the top row, key included. The
#: key and the four MDS panels draw the same sixteen models, and a reader who
#: looks up a point in the key and then finds it in a panel should be looking at
#: the same mark; the two sources default to 26 and 130, which reads as two
#: different kinds of thing.
MIXTURE_MARKER_SIZE = 150

#: The colour of every axis line in the figure -- the MDS crosshairs, the panel
#: spines, and the key's triangle, which is that key's only frame. One value, so
#: the nine panels sit in one box rather than in nine.
AXIS_COLOR = "0.55"


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

#: Decoder layers of :data:`TOP_BASE_MODEL`, from the checkpoint's own config.
#: The functional reference surrogate names ``N_LAYERS + 1`` rows, so the exact
#: label the suite must be asked for is derived from this rather than written
#: out -- a wrong count then fails loudly in ``compact_selection`` instead of
#: quietly selecting a different row.
TOP_N_LAYERS = 16

#: The LoRA rank the simplex3 adapters were trained at. Not a filter this figure
#: needed when it was written: the rank sweep has since trained the same 16
#: mixtures of this corpus, under this base model and this training draw, at
#: eight ranks — so a scan filtered only by corpus, draw and mixture returns 128
#: models and ``n_expected`` trips. 16 is the rank every suite that predates the
#: sweep ran at, so pinning it keeps this panel the same measurement it was.
#:
#: ``main`` carries the same constant on ``figures/figure2``; it is repeated
#: here rather than inherited because this branch is not merged with it. See
#: :func:`_rank_filter` for why it cannot simply be passed to ``run_suite``.
TOP_LORA_RANK = 16


@contextmanager
def _rank_filter(lora_rank):
    """Apply ``lora_rank`` to ``run_suite``'s scan on a branch that lacks it.

    ``run_suite`` gained a ``lora_rank`` parameter with the LoRA-rank sweep,
    which is newer than this branch. ``CacheIndex.filter(lora_rank=...)`` and
    ``CacheEntry.lora_rank`` are both already here, so only the plumbing is
    missing, and wrapping the scan applies exactly what upstream applies — in
    the same position, before the mixture filter.

    Adding the parameter to ``src/plots/simplex_suite.py`` here instead would
    put a second, independent version of it in a shared module that already has
    one upstream, which is how two branches quietly stop agreeing. This becomes
    dead code the moment ``run_suite`` accepts the argument, because then it is
    never called with a rank. The same shim, for the same reason, is in
    ``scripts/make_permutation_null.py``.
    """
    import inspect

    from src.plots import simplex_suite as _suite

    if lora_rank is None or "lora_rank" in inspect.signature(
            _suite.run_suite).parameters:
        yield
        return
    original = _suite.scan_cache

    def _scanned(*a, **kw):
        return original(*a, **kw).filter(lora_rank=lora_rank)

    _suite.scan_cache = _scanned
    try:
        yield
    finally:
        _suite.scan_cache = original

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
    with _rank_filter(TOP_LORA_RANK):
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
            source="figures/figure2_v2/make_figures.py",
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


def draw_top_row(fig, gs, levels: list[Level], cells, ids) -> None:
    """The mixture key and the four MDS panels, into *gs* (1 x 5)."""
    by_label = {lv.label: lv for lv in levels}
    kax = fig.add_subplot(gs[0, 0])
    ternary_legend(kax, ids, label_models=True,
                   vertex_names=group_display(3), show_topics=False,
                   label_size=MIXTURE_LABEL_SIZE, vertex_size=KEY_VERTEX_SIZE,
                   marker_size=MIXTURE_MARKER_SIZE, label_fmt=comma_label,
                   fill_points=True, pad=KEY_PAD, outline_color=AXIS_COLOR,
                   label_offset=KEY_LABEL_OFFSET, avoid_collisions=True,
                   fontweight="bold", fontfamily=bold_capable_family())
    kax.set_title(KEY_TITLE, fontsize=FONT_SIZE, pad=10,
                  fontweight="bold", fontfamily=bold_capable_family())
    for k, label in enumerate(PANEL_ORDER):
        ax = crosslevel_panel(fig.add_subplot(gs[0, k + 1]), by_label[label].label,
                              cells[label], font_size=FONT_SIZE,
                              point_label_size=MIXTURE_LABEL_SIZE, bold=True,
                              label_fmt=comma_label, axis_color=AXIS_COLOR,
                              marker_size=MIXTURE_MARKER_SIZE)
        # No ticks. An MDS coordinate has no units and no origin a reader can
        # use -- the configuration is only defined up to rotation, reflection
        # and scale, which is exactly why it is scored by Procrustes -- so the
        # numbers on these axes invite a comparison that is not there to make.
        # The axes are still named, because which plane this is remains worth
        # saying.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(MDS_AXES[0])
        # Only the leftmost panel is labelled on y: the four share a meaning,
        # not a scale (they differ by an order of magnitude in absolute MDS
        # size), and repeating the name four times says nothing the first does
        # not.
        if k == 0:
            ax.set_ylabel(MDS_AXES[1])


# ── Bottom row, panels 1 and 2: the bar charts ────────────────────────────────

def exact_band(spec_name: str, generator: str, _cache={}) -> tuple:
    """The uninformed band for a suite whose truth grid is known exactly.

    These suites are *complete symmetric grids on a simplex*, so the truth
    configuration is not sampled and Lambda is not estimated — it comes out
    analytically isotropic, ``1/sqrt(d)`` in every component. That removes both
    approximations the tabulated route carries: the pooled Lambda
    ``scripts/make_baselines.py`` averages over replicates (which for K=4 falls
    back to a synthetic Dirichlet draw, because the members file encodes only
    3-group mixtures), and the log-n interpolation the table needs at an n that
    is not a grid rung — at n=35 that alone is the difference between 0.933 and
    0.9426.

    It also means these panels need no ``constants.json`` and draw in a fresh
    clone. Memoised because ``disparity_null_direct`` runs the real scoring
    loop, and the figure asks for at most three distinct specs.
    """
    key = (spec_name, generator)
    if key in _cache:
        return _cache[key]
    W = np.array(SPECS[spec_name].mixture_pcts(), dtype=float) / 100.0
    coords = W @ simplex_vertices(W.shape[1])
    n_mc = _BAND_MC[generator]
    if generator == "gaussian":
        sample = disparity_null_analytic(len(W), truth_singular_values(coords),
                                         n_mc=n_mc)
    else:
        sample = disparity_null_direct(len(W), coords, generator=generator,
                                       n_mc=n_mc)
    _cache[key] = band_quantiles(sample)
    return _cache[key]


def collection_band(generator: str, xs) -> tuple:
    """The uninformed band over *xs* for panel 3, from the tabulated table.

    The one panel that cannot use :func:`exact_band`: its points are random
    subgroups of the 1004-model pool rather than a grid, so their Lambda varies
    with the draw and is genuinely pooled. Fails by naming the script that
    builds the table rather than with a ``KeyError`` from inside a draw call.
    """
    try:
        triples = [uninformed_band(x, table=load_baseline_table(),
                                   score="disparity", generator=generator,
                                   k=COLLECTION_K, d=COLLECTION_D) for x in xs]
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"cannot draw the collection baseline: {exc}") from None
    lo, mid, hi = (np.array(v) for v in zip(*triples))
    return lo, mid, hi


def load_permutation_digest(path=None) -> dict:
    """Parse ``results/figure2_v2/permutation_null.json``, or explain it.

    Generated and gitignored, like the baseline table, so a fresh clone
    genuinely does not have it — and the failure names the command and the
    cluster job that write it rather than a bare ``FileNotFoundError``.
    """
    import json

    if path is None:
        path = REPO_ROOT / "results/figure2_v2/permutation_null.json"
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"no permutation nulls at {path}.\n"
            "Run `python scripts/make_permutation_null.py` (or submit "
            "`jobs/figure2_v2_permutation_null.sh`); the file is generated, "
            "not tracked.")
    with path.open() as fh:
        return json.load(fh)


def _band_of(digest: dict, key: str):
    """``(lo, mid, hi)`` for one digest key, or ``None`` when it was not run."""
    entry = (digest.get("entries") or {}).get(key)
    if entry is None:
        return None
    return entry["lo"], entry["mid"], entry["hi"]


#: ``plot_grouped_bars`` centres each group's cluster on an integer tick and
#: gives the cluster ``bar_width`` of the slot, so these two reproduce its
#: geometry rather than guessing at it. Kept next to each other because they
#: must agree with that function and with each other.
BAR_WIDTH = 0.8


def group_span(g: int, pad: float = 0.0) -> tuple[float, float]:
    """The x extent of one group's whole cluster of bars."""
    return g - BAR_WIDTH / 2 - pad, g + BAR_WIDTH / 2 + pad


def bar_span(g: int, i: int, n_series: int) -> tuple[float, float]:
    """The x extent of the *i*-th series' bar within group *g*."""
    width = BAR_WIDTH / n_series
    off = (i - (n_series - 1) / 2) * width
    return g + off - width / 2, g + off + width / 2


class Bands:
    """Draws the two reference bands, and owns what must be decided once.

    Two things belong here rather than in the panel functions. First, **one
    legend entry per band type across the whole figure**: a band is a level, not
    a series, and four panels each contributing an entry would say there are
    eight of them. Second, the **grain**, which is a display choice and nothing
    more — the sidecar computes and stores a null for every individual cell
    because doing so costs about fifteen seconds, so switching between a band
    per level group and a band per bar never recomputes anything.
    """

    def __init__(self, generator: str | None = None, digest: dict | None = None,
                 grain: str = "level") -> None:
        self.generator = generator
        self.digest = digest
        self.grain = grain
        self._drew_uninformed = False
        self._drew_permutation = False

    # -- one legend entry each ------------------------------------------------
    #
    # The bands are never labelled *into a panel legend*. A bar panel's only
    # free space is the strip above its tallest bar, which is exactly where the
    # bands are, so a legend large enough to name them lands on top of them --
    # and widening the y axis far enough to seat a six-entry legend above a
    # band at 0.98 would compress the decade the bars actually live in. The
    # panel legends therefore keep naming only their own series, and the bands
    # are named once for the whole row by ``legend_handles`` below.
    def _u(self, ax, **kw):
        if self.generator is None:
            return
        draw_uninformed_band(ax, generator=self.generator, label=False, **kw)
        self._drew_uninformed = True

    def _p(self, ax, band, **kw):
        if band is None:
            return
        draw_permutation_band(ax, band=band, label=False, **kw)
        self._drew_permutation = True

    def legend_handles(self) -> tuple[list, list[str]]:
        """Proxy handles for whichever bands were actually drawn.

        Proxies rather than the real artists because a band is drawn many times
        -- once per level group, or once per bar -- and every copy is the same
        level. Each handle pairs the shaded interval with the dashed median the
        band draws, so the legend key looks like the thing it names.
        """
        handles: list = []
        labels: list[str] = []
        pairs = [
            (self._drew_uninformed, UNINFORMED_COLOUR,
             UNINFORMED_LABEL.format(generator=self.generator)),
            (self._drew_permutation, PERMUTATION_COLOUR, PERMUTATION_LABEL),
        ]
        for drew, colour, text in pairs:
            if not drew:
                continue
            handles.append((
                # Darker than the band itself (0.13). The band is faint by
                # design -- it is a backdrop -- but a key is a tenth the size
                # of the thing it names, and at 0.13 over a legend-sized patch
                # it reads as an empty box.
                Patch(facecolor=colour, alpha=0.38, linewidth=0),
                Line2D([], [], color=colour, ls="--", lw=1.0),
            ))
            labels.append(text)
        return handles, labels

    # -- the bar panels -------------------------------------------------------
    def uninformed_whole(self, ax, spec_name: str) -> None:
        """One level across the panel, for a panel whose bars share n and K."""
        if self.generator is None:
            return
        self._u(ax, mode="horizontal",
                band=exact_band(spec_name, self.generator))

    def uninformed_per_bar(self, ax, levels, panel_runs) -> None:
        """One level per bar, for a panel whose bars differ in n or K."""
        if self.generator is None:
            return
        for g in range(len(levels)):
            for i, run in enumerate(panel_runs):
                spec = SPEC_FOR_CORPUS.get(run.corpus)
                if spec is None:
                    continue
                self._u(ax, mode="span",
                        band=exact_band(spec, self.generator),
                        x=bar_span(g, i, len(panel_runs)))

    def permutation_groups(self, ax, levels, panel_runs, panel: str) -> None:
        """The permutation band on a bar panel, at the configured grain."""
        if self.digest is None:
            return
        for g, lv in enumerate(levels):
            if self.grain == "bar":
                for i, run in enumerate(panel_runs):
                    band = _band_of(self.digest,
                                    f"bars|{run.figure_dir}|{lv.suite_level}")
                    self._p(ax, band, mode="span",
                            x=bar_span(g, i, len(panel_runs)))
            else:
                band = _band_of(self.digest,
                                f"barlevel|{panel}|{lv.suite_level}")
                self._p(ax, band, mode="span", x=group_span(g))

    # -- the sweep panels -----------------------------------------------------
    def sweep(self, ax, which: str, xmax) -> None:
        """Panel 3's band is a curve in n; panel 4's is a level at fixed n=16.

        Both pool the four level curves into one band rather than drawing four.
        The panels already carry four medians and four IQR fills; a fifth,
        sixth, seventh and eighth shaded region behind them would be unreadable,
        and the band is meant to read as the backdrop the curves are measured
        against.
        """
        if which == "nsweep":
            # The x axis varies training examples at a fixed 16 models, so the
            # level does not move with x.
            self.uninformed_whole(ax, "yahoo_nsweep")
            self._p(ax, _band_of(self.digest or {}, "nsweep|ALL"),
                    mode="horizontal")
            return

        ns = sorted(self._collection_ns())
        if self.generator is not None:
            # A dense curve rather than the sweep's seven rungs, so the band
            # reads as a level and not as an eighth series. It spans exactly the
            # data's range, which keeps it inside the tabulated grid.
            lo_n, hi_n = (min(ns), max(ns)) if ns else (5, 500)
            if xmax is not None:
                hi_n = min(hi_n, xmax)
            xs = np.geomspace(lo_n, hi_n, 120)
            self._u(ax, mode="curve", band=collection_band(self.generator, xs),
                    x=xs)
        if self.digest is not None and ns:
            xs = np.array([n for n in ns if xmax is None or n <= xmax])
            triples = [_band_of(self.digest, f"collection|ALL|{n}") for n in xs]
            keep = [(x, t) for x, t in zip(xs, triples) if t is not None]
            if keep:
                xs = np.array([x for x, _ in keep])
                lo, mid, hi = (np.array(v) for v in zip(*[t for _, t in keep]))
                self._p(ax, (lo, mid, hi), mode="curve", x=xs)

    def _collection_ns(self) -> list[int]:
        out = []
        for key in (self.digest or {}).get("entries", {}):
            parts = key.split("|")
            if len(parts) == 3 and parts[0] == "collection" and parts[1] == "ALL":
                out.append(int(parts[2]))
        return out


def _bar_legend_loc(bands: Bands | None) -> str:
    """Where a bar panel's series legend goes. Always ``upper left``.

    Kept as a function because the answer was contested. Both reference levels
    sit near 0.9 on an axis whose bars top out near 0.3, so an upper-left
    legend risks landing on the very thing the bands were added to show. Two
    other placements were tried and are worse: a bar panel on a log axis has no
    free space *below* a bar -- bars are drawn from the axis floor, so every
    lower placement is on top of data -- and the mid-panel gap between the
    tallest bar and the bands is too short to seat two rows.

    What resolves it instead is headroom plus eviction: ``YLIMS_BANDED`` lifts
    the log top to 8, and the band keys move out of the panel entirely (see
    :func:`draw_band_legend`). That leaves the series legend four entries in
    two rows, which clears the bands, so the placement never has to change.
    """
    del bands  # kept in the signature for callers; the answer no longer varies
    return "upper left"


def _bar_values(run: runs.Run, levels: list[Level]) -> list[float]:
    """One run's disparity per level, at that corpus's own ``d = K-1``."""
    rows = runs.read_run(FIGURES_ROOT, run)
    scores = runs.level_scores(rows, "procrustes", "dK1",
                               [lv.perspective() for lv in levels])
    return [scores[lv.label] for lv in levels]


def draw_models_bars(ax, levels: list[Level], bands: Bands | None = None) -> None:
    """Panel 1 — the four base models on yahoo, one bar per model."""
    ax.set_title("Base Model")
    panel_runs = runs.runs_for_corpus("yahoo")
    series = [
        make_series(_bar_values(r, levels), label=r.model_label,
                    color=MODEL_COLORS[r.base_model])
        for r in panel_runs
    ]
    if bands is not None:
        # Every bar here is the same 16-model K=3 grid, so the uninformed level
        # is one level for the whole panel — it depends on n and K, not on the
        # taxonomy. The permutation null does depend on the taxonomy, so it is
        # drawn per level group.
        bands.uninformed_whole(ax, "yahoo")
        bands.permutation_groups(ax, levels, panel_runs, "models")
    plot_grouped_bars(series, [lv.label for lv in levels], ax=ax,
                      legend=True,
                      legend_kwargs={"loc": _bar_legend_loc(bands), "ncol": 2},
                      savefig=False)


def draw_corpora_bars(ax, levels: list[Level], bands: Bands | None = None) -> None:
    """Panel 2 — the three corpora on OLMo-2-1B, one bar per corpus."""
    ax.set_title("Training Data Source")
    panel_runs = runs.runs_for_model(TOP_BASE_MODEL)
    series = [
        make_series(_bar_values(r, levels), label=r.corpus,
                    color=CORPUS_COLORS[r.corpus])
        for r in panel_runs
    ]
    if bands is not None:
        # The one panel whose bars do not share a baseline: yahoo is 16 models
        # on a 3-vertex simplex, dolly and oasst1 are 35 on a 4-vertex one, and
        # this driver already reads each at its own d = K-1. One level across
        # the panel would apply yahoo's to all three.
        bands.uninformed_per_bar(ax, levels, panel_runs)
        bands.permutation_groups(ax, levels, panel_runs, "corpora")
    plot_grouped_bars(series, [lv.label for lv in levels], ax=ax,
                      legend=True,
                      legend_kwargs={"loc": _bar_legend_loc(bands), "ncol": 3},
                      savefig=False)


# ── Bottom row, panels 3 and 4: the two sweeps ────────────────────────────────

#: Panel -> (csv, x column, disparity column, panel title, x axis label). The
#: disparity column is the reading chosen for that sweep; see the module
#: docstring for why. The title names what the panel varies and the axis label
#: names the unit it varies it in, which is why both are carried and neither is
#: derived from the other.
#: ``(csv, x column, y column, title, x label, x cap)``. The cap drops the
#: sweep's largest draw sizes from the panel; ``None`` keeps every x in the
#: file. The nsweep ran one value past 2000 -- 5000 -- and it is cut here
#: rather than in the CSV, so the sweep's own figure keeps it.
SWEEPS = {
    "collection": (FIGURES_ROOT / "simplex_collection_size" / "group_size_scores.csv",
                   "n", "disparity_B", "Collection Size",
                   "Models in the collection", None),
    "nsweep": (FIGURES_ROOT / "simplex3_nsweep_olmo2_nsweep" / "nsweep_scores.csv",
               "n_samples", "disparity_requested", "Training Set Size",
               "Training examples per adapter", 2_000),
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
               bands: Bands | None = None) -> None:
    """Panels 3 and 4 — all four levels on one axes, median plus IQR band."""
    path, xcol, ycol, title, xlabel, xmax = SWEEPS[which]
    if bands is not None:
        bands.sweep(ax, which, xmax)
    for lv in levels:
        xs, med, q1, q3 = sweep_series(path, xcol, ycol, lv.sweep_key, xmax)
        ax.fill_between(xs, q1, q3, color=lv.color, alpha=BAND_ALPHA, lw=0,
                        zorder=2)
        ax.plot(xs, med, color=lv.color, ls=SWEEP_LINESTYLE,
                marker=SWEEP_MARKER, ms=SWEEP_MARKER_SIZE, lw=SWEEP_LINEWIDTH,
                label=lv.label, zorder=3)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if legend:
        ax.legend(loc="upper right", ncol=2)


# ── Assembly ──────────────────────────────────────────────────────────────────

#: The score alone. It runs 1 -> 0, **lower is better**, which is the opposite
#: of every other axis in this project and is not in the label -- the module
#: docstring and the caption carry it, because a longer label is clipped at the
#: top of a panel this tall once every piece of text is at one size.
YLABEL = "Procrustes Disparity"

#: The gap between the two rows, **in inches**. The rows read as separate
#: statements -- one figure repeated at four levels, then four ways that figure
#: moves -- so they get far more air between them than a default layout leaves.
#: This is the one knob for it.
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

#: The same ranges with headroom above the reference levels.
#:
#: Both bands sit near 0.9 on this score and the plain range stops just above
#: it, so with bands drawn a bar panel's series legend has nowhere to go that
#: is not on the bands or on the bars. Widening the top buys that room, and it
#: costs a reader nothing: the axis is a disparity, everything above 1 is
#: unreachable, and the added space is empty by construction. 8 rather than a
#: smaller number because it is what measurably clears a two-row legend — at 4
#: the legend's lower edge still cut the permutation median.
YLIMS_BANDED = {"log": (4e-3, 8.0), "linear": (0.0, 1.25)}


def ylim_for(yscale: str, bands: "Bands | None") -> tuple[float, float]:
    """The shared y range, widened when reference levels are drawn."""
    return (YLIMS_BANDED if bands is not None else YLIMS)[yscale]


#: How far below a bar panel's x axis the band legend sits, in axes-height
#: units. Enough to clear the tick labels -- the four taxonomy names -- and no
#: more, so the strip reads as belonging to the row above it.
BAND_LEGEND_DROP = -0.13


def draw_band_legend(ax, bands: "Bands") -> None:
    """Name the reference bands once, in a strip beneath the two bar panels.

    Not inside a panel, and not once per panel. A band is a *level* rather than
    a series: it is the same level on all four panels, so naming it four times
    would say there are four of them, and naming it inside a bar panel would
    put the key on top of the band, which is the only free space a bar panel
    has. Anchoring it to panel 1 and letting two columns run rightwards puts it
    under panels 1 and 2, in the margin the row already owns.

    Attached with ``add_artist`` rather than ``ax.legend`` so that panel 1 keeps
    the series legend it already has -- ``ax.legend`` would replace it.
    """
    handles, labels = bands.legend_handles()
    if not handles:
        return
    legend = Legend(
        ax, handles, labels,
        loc="upper left", bbox_to_anchor=(0.0, BAND_LEGEND_DROP),
        bbox_transform=ax.transAxes, ncol=len(handles),
        frameon=False, borderaxespad=0.0,
        # Both proxies are a shaded patch with the band's dashed median over
        # it; ndivide=None overlays the pair in one key rather than shrinking
        # them into two half-width keys side by side.
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    ax.add_artist(legend)


def draw_bottom_row(fig, gs, bar_levels: list[Level],
                    sweep_levels: list[Level], yscale: str = "log",
                    bands: Bands | None = None) -> list:
    """The four score panels, into *gs* (1 x 4), sharing one y axis."""
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2, 3)]

    draw_models_bars(axes[0], bar_levels, bands)
    draw_corpora_bars(axes[1], bar_levels, bands)
    draw_sweep(axes[2], "collection", sweep_levels, True, bands)
    draw_sweep(axes[3], "nsweep", sweep_levels, False, bands)

    for k, ax in enumerate(axes):
        ax.set_yscale(yscale)
        ax.set_ylim(*ylim_for(yscale, bands))
        if k == 0:
            ax.set_ylabel(YLABEL)
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
    if bands is not None:
        draw_band_legend(axes[0], bands)
    return axes


def _apply_style() -> None:
    """The house style, then one text size over the top of it.

    ``set_style`` scales tick labels and legends to three quarters of the base
    size and leaves titles at it, which is right for a panel that is a figure in
    its own right. Here nine panels from three sources sit in one frame, and a
    size difference between them reads as emphasis rather than as provenance --
    so every piece of text is set to :data:`FONT_SIZE`, at one weight.

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
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE, "figure.titlesize": FONT_SIZE,
        "font.sans-serif": [bold_capable_family(), "DejaVu Sans"],
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.edgecolor": AXIS_COLOR,
        "axes.linewidth": 1.0,
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
        # ``get_legend`` returns only the axes' *primary* legend. The band
        # legend is attached as a plain artist so it does not displace the
        # series legend beside it, so it has to be collected separately or it
        # would be the one unbolded thing on the figure.
        legends = [lg for lg in [ax.get_legend()] if lg is not None]
        legends += [a for a in ax.artists if isinstance(a, Legend)]
        for legend in legends:
            for txt in legend.get_texts():
                txt.set_fontweight("bold")
                txt.set_fontfamily(family)


def build(row: str, decoding: str, yscale: str = "log",
          cache_root=None, no_cache=False, baseline: str | None = None,
          permutation: bool = False, perm_grain: str = "level",
          suffix: str = "", outdir=None) -> Path:
    """Draw the requested row(s) and save. Returns the path written."""
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

    _apply_style()

    #: The key column is wider than a panel — it carries sixteen mixture labels
    #: and three vertex names — and the top row's panels are square, so the two
    #: rows are given different heights rather than one shared one.
    top_h, bottom_h = 4.1, 3.6
    width = 19.0
    both = want_top and want_bottom
    gap = ROW_GAP if both else 0.0
    heights = [h for h, want in [(top_h, want_top), (gap, both),
                                 (bottom_h, want_bottom)] if want]
    height = sum(heights)
    fig = plt.figure(figsize=(width, height), layout="constrained")
    outer = fig.add_gridspec(len(heights), 1, height_ratios=heights)

    r = 0
    if want_top:
        draw_top_row(fig, outer[r].subgridspec(1, 5, width_ratios=[1.28] + [1.0] * 4),
                     top_levels, cells, ids)
        r += 2 if both else 1
    if want_bottom:
        bands = None
        if baseline is not None or permutation:
            bands = Bands(
                generator=baseline,
                digest=load_permutation_digest() if permutation else None,
                grain=perm_grain)
        draw_bottom_row(fig, outer[r].subgridspec(1, 4), bar_levels,
                        sweep_levels, yscale, bands)

    _embolden(fig)

    stem = f"fig_figure2_{row}" if row != "both" else "fig_figure2"
    if decoding != "mixed":
        stem += f"_{decoding}"
    if yscale != "log":
        stem += f"_{yscale}"
    # A band token goes on the filename for the same reason the decoding does:
    # a figure with a reference level drawn on it is a different figure, and
    # overwriting the plain one with it would lose the comparison. An explicit
    # --suffix replaces both tokens rather than adding to them.
    if suffix:
        stem += suffix
    else:
        if baseline is not None:
            stem += f"_uninformed-{baseline}"
        if permutation:
            stem += "_perm" + ("-bar" if perm_grain == "bar" else "")
    out = (Path(outdir) if outdir else HERE) / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the top row's matrices, ignoring stored "
                         "results (still writes them back)")
    ap.add_argument("--baseline", nargs="?", const="dirichlet", default=None,
                    choices=["dirichlet", "gaussian"],
                    help="draw the uninformed band — the level a taxonomy that "
                         "learned nothing would score (default off; a bare "
                         "--baseline means dirichlet). Panels 1, 2 and 4 "
                         "compute it exactly from the suite's own mixture grid; "
                         "panel 3 needs results/baselines/constants.json, which "
                         "scripts/make_baselines.py writes")
    ap.add_argument("--permutation", action="store_true",
                    help="draw the permutation band — the level reached by "
                         "shuffling which model is which. Needs "
                         "results/figure2_v2/permutation_null.json, which "
                         "scripts/make_permutation_null.py writes")
    ap.add_argument("--perm-grain", choices=["level", "bar"], default="level",
                    help="on the bar panels, one permutation band per taxonomy "
                         "level (default) or one per bar. Purely a display "
                         "choice: every cell's null is already stored")
    ap.add_argument("--suffix", default="",
                    help="filename token, replacing the band tokens")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    print(f"wrote {build(args.row, args.behavioral, args.yscale, args.cache_root, args.no_cache, args.baseline, args.permutation, args.perm_grain, args.suffix, args.outdir)}")


if __name__ == "__main__":
    main()
