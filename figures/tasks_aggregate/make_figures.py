#!/usr/bin/env python
"""Every simplex3 run on one set of axes: four base models across three corpora.

``figures/simplex3_aggregate`` puts the four base models on shared axes **one
corpus at a time**, because a yahoo score and a dolly score are not the same
measurement. This driver asks the question that separation cannot: read across
all twelve ``(base model, corpus)`` pairs at once, does a taxonomy level's
standing hold — is the structural level always the one that recovers the mixture,
or does the answer move with the corpus, with the model, or with both?

Nothing is recomputed. Every number is read from the twelve suites'
``crosslevel_scores.csv`` files, so this runs in seconds on a login node and
needs no cache, no GPU and no adapters. The runs, their colours and the
per-level perspectives all come from :mod:`src.plots.simplex_runs`, which is the
single table both cross-run drivers read.

Four figure families
--------------------
1. **Radar** — one spoke per ``(model, corpus)`` pair, twelve of them, grouped
   into four arcs of three so each base model's corpora sit together and the
   model is named once. One polygon per taxonomy level. This is the figure that
   answers "does a level hold its rank all the way round".
2. **Bars by base model**, one figure per corpus — x is the taxonomy level, one
   bar per model. Reads down a single corpus: within this data, which models
   agree?
3. **Bars by corpus**, one figure per base model — x is the taxonomy level, one
   bar per corpus. Reads across corpora for one model: within this architecture,
   which data agree?
4. **Bars, everything at once** — x is the taxonomy level, twelve bars per
   group. Bar colour is the model's, tinted by corpus, so the same encoding
   carries both categories without a second palette.

Levels
------
The six rows of the standing per-level default set, from what the model was
trained on to what it says: the dataset mean embedding; the structural level at
three scopes (all projections, output projections, and the **last layer's**
output projection alone); the all-layer functional state; and the per-query
behavioral representation. See :data:`src.plots.simplex_runs.LEVELS`.

The Data bar is a property of the corpus alone — the dataset embedding never
touches the base model — so it is identical for all four models within a corpus.
That is not a bug in the figure; it is the reference the other five levels are
read against, and seeing it flat across a model group is the point.

Scores, and the two Procrustes conventions
------------------------------------------
``dcor`` is a distance correlation with the ground-truth mixture geometry, 0 → 1,
higher is better. ``procrustes`` is a disparity against the same geometry, 1 → 0,
**lower** is better; ``1-procrustes`` is that disparity flipped so it reads on
dcor's "taller/further out is better" convention. All three are drawn.

A disparity is reported at an embedding dimension, and the corpora do not share
one: yahoo mixes three groups (truth in ``d2``), dolly and oasst1 mix four
(truth in ``d3``). Both defensible conventions are emitted, filename-tagged:

``_d2``
    every run at ``procrustes_d2``. One measurement across the whole axis, at
    the price of asking a four-vertex truth to fit in a plane it cannot, which
    inflates the dolly and oasst1 numbers.
``_dK1``
    every run at its own ``K-1``. Each corpus's honest disparity, but a yahoo
    bar and a dolly bar are then not the same measurement.

The dcor figures have no such tag: a distance correlation is computed on the
distance matrices, not on a fitted embedding, so there is no dimension to pick.

Axis ranges are fixed per version across every figure of that version, so any
two of these figures can be laid side by side. dcor and ``1-procrustes`` use a
full 0–1; the raw disparities are all small, so their range is set from the
largest value in the whole twelve-run table rather than wasting nine tenths of
the axis.

Usage
-----
    python figures/tasks_aggregate/make_figures.py
    python figures/tasks_aggregate/make_figures.py --version dcor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Pinned before numpy loads its BLAS -- see the note in src/plots/simplex_suite.py.
os.environ.setdefault("MODEL_TAXONOMY_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.plots import (  # noqa: E402
    make_series, plot_grouped_bars, plot_radar, set_style, shade,
)
from src.plots import simplex_runs as runs  # noqa: E402

#: Figure version -> (score, procrustes convention, value transform, axis label).
#:
#: The transform is applied to the column as read; ``1 - procrustes`` is the only
#: one that is not the identity, and it exists purely so a disparity can be read
#: on the same "higher is better" convention as dcor. ``dims`` is ``None`` for
#: dcor, which has no fitted embedding and so no dimension to choose.
VERSIONS = {
    "dcor": ("dcor", None, lambda v: v,
             "Distance correlation with data mixture"),
    "procrustes_d2": ("procrustes", "d2", lambda v: v,
                      "Procrustes disparity vs. data mixture (d=2)"),
    "procrustes_dK1": ("procrustes", "dK1", lambda v: v,
                       "Procrustes disparity vs. data mixture (d=K−1)"),
    "1-procrustes_d2": ("procrustes", "d2", lambda v: 1.0 - v,
                        "1 − Procrustes disparity vs. data mixture (d=2)"),
    "1-procrustes_dK1": ("procrustes", "dK1", lambda v: 1.0 - v,
                         "1 − Procrustes disparity vs. data mixture (d=K−1)"),
}

#: How far each corpus's tint moves its model's colour towards the page, in the
#: figure where one colour has to carry both categories. In :data:`CORPORA`
#: order, so the run order and the tint order cannot drift apart.
CORPUS_TINTS = [-0.15, 0.25, 0.55]


def collect(figures_root: Path, version: str) -> dict[str, dict[str, float]]:
    """``{run label: {level label: value}}`` for one figure version.

    Every one of the twelve suites must have run: a partial table would draw a
    radar with a gap in it and a bar chart missing a series, neither of which
    says which run is absent. The ``FileNotFoundError`` is left to reach the
    caller, which names the missing file.
    """
    score, dims, transform, _ = VERSIONS[version]
    out = {}
    for run in runs.RUNS:
        rows = runs.read_run(figures_root, run)
        scores = runs.level_scores(rows, score, dims or "dK1")
        out[run.label] = {k: transform(v) for k, v in scores.items()}
    return out


def limits(table: dict[str, dict[str, float]], version: str) -> tuple[float, float]:
    """The y/radial range every figure of this version shares.

    dcor and the flipped disparity are read against a full 0–1: both are bounded
    there by construction and both put "good" at the top, so the empty part of
    the axis is information about how far from the truth the worst level is. A
    raw disparity is bounded there too but never comes close to filling it — the
    largest in the whole table is well under a half — so its range is set from
    the table with a little headroom, and set once for all figures of the version
    rather than per figure, which is what keeps them comparable.
    """
    # if not version.startswith("procrustes"):
    return (0.0, 1.0)
    # top = max(v for row in table.values() for v in row.values())
    # return (0.0, top * 1.15)


def _levels() -> list[str]:
    return [p.label for p in runs.LEVELS]


# ── The four figure families ──────────────────────────────────────────────────

def draw_radar(table, version: str, outdir: Path) -> Path:
    """Twelve spokes, one per (model, corpus); one polygon per taxonomy level."""
    _, _, _, axis_label = VERSIONS[version]
    set_style("one_col", fig_width=7.2, fig_height=6.0)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    #   The spoke says the corpus; the arc label above it says the model. With
    #   twelve spokes the full pair name on each would collide long before it
    #   was read.
    spokes = [r.corpus for r in runs.RUNS]
    groups = [(runs.MODELS[m][0], len(runs.runs_for_model(m))) for m in runs.MODELS]
    series = [
        make_series([table[r.label][p.label] for r in runs.RUNS],
                    label=p.label.replace("\n", " "), color=p.color,
                    marker="none", linestyle="-")
        for p in runs.LEVELS
    ]
    plot_radar(
        series,
        axis_labels=spokes,
        ax=ax,
        start_angle=90.0,      # the first model's first corpus at the top ...
        direction="cw",        # ... then clockwise, which is the reading order
        rlim=limits(table, version),
        #   Six polygons over a narrow band stack into one wash at any alpha.
        fill_alpha=0.0,
        label_pad=8,
        #   Straight ticked axes out to each spoke rather than concentric rings:
        #   the spokes are twelve separate runs that happen to share a scale, not
        #   a web of one thing.
        spoke_axes=True,
        rlabel_spoke=0,
        spoke_groups=groups,
        legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.10),
                       "ncol": 3, "frameon": False},
        savefig=False,
    )
    #   The score name goes on the figure rather than on the axes: a polar
    #   title sits inside the ring's bounding box and would land on the top
    #   group's label.
    fig.suptitle(axis_label, y=0.97)
    ax.set_position([0.19, 0.17, 0.62, 0.70])
    path = outdir / f"fig_tasks_radar_{version}.png"
    fig.savefig(path, pad_inches=0.3)
    plt.close(fig)
    return path


def draw_bars_by_model(table, version: str, outdir: Path) -> list[Path]:
    """One figure per corpus: taxonomy level on x, one bar per base model."""
    _, _, _, axis_label = VERSIONS[version]
    levels = _levels()
    paths = []
    for corpus in runs.CORPORA:
        set_style("one_col", fig_width=7.0, fig_height=3.8)
        fig, ax = plt.subplots()
        series = [
            make_series([table[r.label][lvl] for lvl in levels],
                        label=r.model_label, color=r.color)
            for r in runs.runs_for_corpus(corpus)
        ]
        plot_grouped_bars(
            series, group_labels=levels, ax=ax,
            ylabel=axis_label, ylim=limits(table, version),
            annotate=True, annot_fmt="{:.3f}", annot_size=5.5,
            annot_rotation=90, annot_inside="auto",
            title=f"{corpus} — {axis_label}",
            legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.16),
                           "ncol": 4, "frameon": False},
            savefig=False,
        )
        path = outdir / f"fig_tasks_bars_by_model_{corpus}_{version}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def draw_bars_by_corpus(table, version: str, outdir: Path) -> list[Path]:
    """One figure per base model: taxonomy level on x, one bar per corpus."""
    _, _, _, axis_label = VERSIONS[version]
    levels = _levels()
    paths = []
    for base_model, (model_label, *_) in runs.MODELS.items():
        set_style("one_col", fig_width=7.0, fig_height=3.8)
        fig, ax = plt.subplots()
        series = [
            make_series([table[r.label][lvl] for lvl in levels],
                        label=r.corpus, color=r.corpus_color)
            for r in runs.runs_for_model(base_model)
        ]
        plot_grouped_bars(
            series, group_labels=levels, ax=ax,
            ylabel=axis_label, ylim=limits(table, version),
            annotate=True, annot_fmt="{:.3f}", annot_size=5.5,
            annot_rotation=90, annot_inside="auto",
            title=f"{model_label} — {axis_label}",
            legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.16),
                           "ncol": 3, "frameon": False},
            savefig=False,
        )
        slug = model_label.replace(".", "").replace("/", "-")
        path = outdir / f"fig_tasks_bars_by_corpus_{slug}_{version}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def draw_bars_all(table, version: str, outdir: Path) -> Path:
    """One figure, twelve bars per taxonomy level: every (model, corpus) pair.

    Colour carries both categories at once — the hue is the model's and the tint
    is the corpus's — because twelve unrelated hues would be unreadable and a
    hue-plus-hatch scheme reads as noise at this bar width. The runs are
    model-major, so each model's three corpora stand together inside the group.
    """
    _, _, _, axis_label = VERSIONS[version]
    levels = _levels()
    set_style("one_col", fig_width=11.0, fig_height=4.4)
    fig, ax = plt.subplots()
    tints = dict(zip(runs.CORPORA, CORPUS_TINTS))
    series = [
        make_series([table[r.label][lvl] for lvl in levels],
                    label=r.label, color=shade(r.color, tints[r.corpus]))
        for r in runs.RUNS
    ]
    plot_grouped_bars(
        series, group_labels=levels, ax=ax, bar_width=0.88,
        ylabel=axis_label, ylim=limits(table, version),
        title=axis_label,
        legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.14),
                       "ncol": 4, "frameon": False},
        savefig=False,
    )
    path = outdir / f"fig_tasks_bars_all_{version}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures-root", type=Path, default=REPO_ROOT / "figures",
                    help="directory holding the twelve simplex3 figure dirs")
    ap.add_argument("--outdir", type=Path, default=HERE)
    ap.add_argument("--version", action="append", choices=list(VERSIONS),
                    help="restrict to one or more score versions (default: all)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for version in (args.version or list(VERSIONS)):
        try:
            table = collect(args.figures_root, version)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc.filename}: not written yet. All twelve simplex3 suites "
                "have to have run before these figures mean anything; run the "
                "per-model drivers first."
            ) from exc
        print(draw_radar(table, version, args.outdir))
        for p in draw_bars_by_model(table, version, args.outdir):
            print(p)
        for p in draw_bars_by_corpus(table, version, args.outdir):
            print(p)
        print(draw_bars_all(table, version, args.outdir))


if __name__ == "__main__":
    main()
