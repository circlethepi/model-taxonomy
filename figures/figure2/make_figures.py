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
as emphasis rather than as provenance. The single exception is the vertex and
centre labels annotated *inside* the MDS panels, which take one step down (see
:data:`POINT_LABEL_SIZE`) because they sit among the points they name.

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

from src.plots import make_series, plot_grouped_bars, save_figure, set_style  # noqa: E402
from src.plots.config import bold_capable_family  # noqa: E402
from src.plots import simplex_runs as runs  # noqa: E402
from src.plots.simplex import crosslevel_panel, ternary_legend  # noqa: E402
from src.plots.simplex import group_display  # noqa: E402

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
MODEL_COLORS = {
    "allenai/OLMo-2-0425-1B-Instruct":      "#000000",
    "Qwen/Qwen3.5-4B":                      "#56B4E9",
    "meta-llama/Llama-3.1-8B-Instruct":     "#CC79A7",
    "mistralai/Mistral-Nemo-Instruct-2407": "#882255",
}

#: Corpus -> colour, in order of increasing distance from a topic mixture:
#: yahoo mixes topics, dolly mixes instruction tasks, oasst1 mixes languages.
CORPUS_COLORS = {"yahoo": "#44AA99", "dolly": "#AA4499", "oasst1": "#999999"}

#: One dash pattern per level, so the four curves in panels 3-4 survive a
#: greyscale print and a reader who cannot separate the hues. It travels with
#: the level, not with its position.
LEVEL_DASHES = {"Data": "-", "Structural": "--",
                "Functional": "-.", "Behavioral": ":"}

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

#: The one exception, for the vertex and centre labels annotated inside the top
#: row's MDS panels. They sit among the points they name rather than outside the
#: axes, so at the full size they crowd the configuration they are there to
#: explain. A step down, not a different register.
POINT_LABEL_SIZE = FONT_SIZE - 2

#: The names of the two axes every top-row panel is drawn in.
MDS_AXES = ("MDS 1", "MDS 2")


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


def draw_top_row(fig, gs, levels: list[Level], cells, ids) -> None:
    """The mixture key and the four MDS panels, into *gs* (1 x 5)."""
    by_label = {lv.label: lv for lv in levels}
    kax = fig.add_subplot(gs[0, 0])
    ternary_legend(kax, ids, label_models=True,
                   vertex_names=group_display(3), show_topics=False,
                   label_size=FONT_SIZE, vertex_size=FONT_SIZE, marker_size=34,
                   fontweight="bold", fontfamily=bold_capable_family())
    for k, label in enumerate(PANEL_ORDER):
        ax = crosslevel_panel(fig.add_subplot(gs[0, k + 1]), by_label[label].label,
                              cells[label], font_size=FONT_SIZE,
                              point_label_size=POINT_LABEL_SIZE, bold=True)
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

def _bar_values(run: runs.Run, levels: list[Level]) -> list[float]:
    """One run's disparity per level, at that corpus's own ``d = K-1``."""
    rows = runs.read_run(FIGURES_ROOT, run)
    scores = runs.level_scores(rows, "procrustes", "dK1",
                               [lv.perspective() for lv in levels])
    return [scores[lv.label] for lv in levels]


def draw_models_bars(ax, levels: list[Level]) -> None:
    """Panel 1 — the four base models on yahoo, one bar per model."""
    ax.set_title("Base Model")
    series = [
        make_series(_bar_values(r, levels), label=r.model_label,
                    color=MODEL_COLORS[r.base_model])
        for r in runs.runs_for_corpus("yahoo")
    ]
    plot_grouped_bars(series, [lv.label for lv in levels], ax=ax,
                      legend=True, legend_kwargs={"loc": "upper left", "ncol": 2},
                      savefig=False)


def draw_corpora_bars(ax, levels: list[Level]) -> None:
    """Panel 2 — the three corpora on OLMo-2-1B, one bar per corpus."""
    ax.set_title("Training Data Source")
    series = [
        make_series(_bar_values(r, levels), label=r.corpus,
                    color=CORPUS_COLORS[r.corpus])
        for r in runs.runs_for_model(TOP_BASE_MODEL)
    ]
    plot_grouped_bars(series, [lv.label for lv in levels], ax=ax,
                      legend=True, legend_kwargs={"loc": "upper left", "ncol": 3},
                      savefig=False)


# ── Bottom row, panels 3 and 4: the two sweeps ────────────────────────────────

#: Panel -> (csv, x column, disparity column, panel title, x axis label). The
#: disparity column is the reading chosen for that sweep; see the module
#: docstring for why. The title names what the panel varies and the axis label
#: names the unit it varies it in, which is why both are carried and neither is
#: derived from the other.
SWEEPS = {
    "collection": (FIGURES_ROOT / "simplex_collection_size" / "group_size_scores.csv",
                   "n", "disparity_B", "Collection Size",
                   "Models in the collection"),
    "nsweep": (FIGURES_ROOT / "simplex3_nsweep_olmo2_nsweep" / "nsweep_scores.csv",
               "n_samples", "disparity_requested", "Training Set Size",
               "Training examples per adapter"),
}


def sweep_series(path: Path, xcol: str, ycol: str, sweep_key: str):
    """``(xs, median, q1, q3)`` over the replicates at each x, for one level."""
    by_x: dict[int, list[float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["perspective"] != sweep_key:
                continue
            by_x.setdefault(int(row[xcol]), []).append(float(row[ycol]))
    if not by_x:
        raise SystemExit(f"{path.name} has no rows for perspective {sweep_key!r}")
    xs = np.array(sorted(by_x))
    vals = [np.asarray(by_x[x], dtype=float) for x in xs]
    return (xs,
            np.array([np.median(v) for v in vals]),
            np.array([np.percentile(v, 25) for v in vals]),
            np.array([np.percentile(v, 75) for v in vals]))


def draw_sweep(ax, which: str, levels: list[Level], legend: bool) -> None:
    """Panels 3 and 4 — all four levels on one axes, median plus IQR band."""
    path, xcol, ycol, title, xlabel = SWEEPS[which]
    for lv in levels:
        xs, med, q1, q3 = sweep_series(path, xcol, ycol, lv.sweep_key)
        ax.fill_between(xs, q1, q3, color=lv.color, alpha=0.15, lw=0, zorder=2)
        ax.plot(xs, med, color=lv.color, ls=LEVEL_DASHES[lv.label],
                marker=SWEEP_MARKER, ms=4, lw=1.4, label=lv.label, zorder=3)
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
                    sweep_levels: list[Level], yscale: str = "log") -> list:
    """The four score panels, into *gs* (1 x 4), sharing one y axis."""
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, k], sharey=axes[0]) for k in (1, 2, 3)]

    draw_models_bars(axes[0], bar_levels)
    draw_corpora_bars(axes[1], bar_levels)
    draw_sweep(axes[2], "collection", sweep_levels, legend=True)
    draw_sweep(axes[3], "nsweep", sweep_levels, legend=False)

    for k, ax in enumerate(axes):
        ax.set_yscale(yscale)
        ax.set_ylim(*YLIMS[yscale])
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
          cache_root=None, no_cache=False) -> Path:
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
    height = (top_h if want_top else 0) + (bottom_h if want_bottom else 0)
    fig = plt.figure(figsize=(width, height), layout="constrained")
    outer = fig.add_gridspec(
        sum([want_top, want_bottom]), 1,
        height_ratios=[h for h, want in [(top_h, want_top), (bottom_h, want_bottom)]
                       if want])

    r = 0
    if want_top:
        draw_top_row(fig, outer[r].subgridspec(1, 5, width_ratios=[1.28] + [1.0] * 4),
                     top_levels, cells, ids)
        r += 1
    if want_bottom:
        draw_bottom_row(fig, outer[r].subgridspec(1, 4), bar_levels,
                        sweep_levels, yscale)

    _embolden(fig)

    stem = f"fig_figure2_{row}" if row != "both" else "fig_figure2"
    if decoding != "mixed":
        stem += f"_{decoding}"
    if yscale != "log":
        stem += f"_{yscale}"
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
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute the top row's matrices, ignoring stored "
                         "results (still writes them back)")
    args = ap.parse_args()
    print(f"wrote {build(args.row, args.behavioral, args.yscale, args.cache_root, args.no_cache)}")


if __name__ == "__main__":
    main()
