#!/usr/bin/env python
"""Cross-model comparison of the four yahoo simplex3 runs.

Every simplex3 suite scores each taxonomy level against the ground-truth
mixture geometry and writes the result to ``crosslevel_scores.csv``. Each of
those files describes one base model in isolation. This driver reads all four
back and puts them on shared axes, so the question stops being "how well does
this level recover the mixture for Llama?" and becomes "does the *same* level
recover it for every base model, or is the answer architecture-dependent?".

Nothing is recomputed here. The scores are read straight out of the four CSVs
the suites already wrote, which is why this script runs in a second on a login
node and needs no cache, no GPU and no adapters.

Perspectives
------------
One perspective is fixed per level, so a spoke means the same measurement for
all four models. Structural and functional labels differ between the uniform
models and hybrid-attention qwen, so they are matched by substring (see
``select_score``):

=============  =========================  ===========
level          surrogate                   metric
=============  =========================  ===========
Data           dataset text · mean         euclidean
Structural     output projections          cosine
Functional     final hidden state          cosine
Behavioral     R=16 · per query            cosine
=============  =========================  ===========

Scores
------
Two scores are drawn, one figure set each, and they run in **opposite**
directions:

* ``dcor``       distance correlation with the ground-truth geometry, 0 → 1,
                 higher is better.
* ``procrustes`` Procrustes disparity against the same geometry, 1 → 0,
                 **lower** is better.

The procrustes figures plot the raw disparity, so on those a *small* polygon
and *short* bars are the good result. Both use a fixed 0–1 radial/y range so
the four models are compared on one scale.

Usage
-----
    python figures/simplex3_aggregate/make_figures.py
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

from src.plots import make_series, plot_grouped_bars, plot_radar, set_style  # noqa: E402
from src.plots.simplex_suite import read_scores_csv, select_score  # noqa: E402

#: Source run per base model, in legend order: (base model, figure dir, colour).
#: The base model is the ``BASE_MODEL`` of that directory's own make_figures.py,
#: written out in full so the legend name is never guessed -- OLMo here is the
#: **1B** instruct model, not the 7B. The directory is the suite whose
#: crosslevel_scores.csv is authoritative for that model -- qwen_v4 rather than
#: qwen_v3 because v4 is the restricted run the llama figures are read against.
#:
#: Colours are Okabe-Ito, which is designed to stay separable under the common
#: colour-vision deficiencies.
MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct",   "simplex3_llama3i", "#0072B2"),  # blue
    ("mistralai/Mistral-Nemo-Instruct-2407", "simplex3_nemo",  "#D55E00"),  # vermillion
    ("allenai/OLMo-2-0425-1B-Instruct",    "simplex3_olmo2",   "#E69F00"),  # gold
    ("Qwen/Qwen3.5-4B",                    "simplex3_qwen_v4", "#CC79A7"),  # purple
]


def label_of(base_model: str) -> str:
    """Legend name for a base model: the HF repo id without its org prefix."""
    return base_model.split("/")[-1]

#: Marker and dash per model, carried alongside colour so the series stay
#: separable in greyscale and for readers who cannot rely on hue alone.
ENCODINGS = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]

#: (spoke label, level, surrogate substring, metric). The order is the order the
#: spokes are drawn in, counter-clockwise from the top: Data at the top,
#: Structural left, Behavioral at the bottom, Functional right.
PERSPECTIVES = [
    ("Data",       "dataset_embedding", None,                  "euclidean"),
    ("Structural", "structural",        "output projections",  "cosine"),
    ("Behavioral", "behavioral",        "R=16 · per query",    "cosine"),
    ("Functional", "functional",        "final hidden state",  "cosine"),
]

#: Left-to-right bar order: the reading order of the taxonomy, from what the
#: model was trained on to what it says. Not the radar's spoke order, which is
#: a placement, so the bars are indexed into PERSPECTIVES by label.
BAR_ORDER = ["Data", "Structural", "Functional", "Behavioral"]

#: Figure version -> (CSV column, transform, axis label, better direction).
#: The transform is applied to the column as read; ``1 - procrustes`` is the
#: only one that is not the identity, and it exists purely so a disparity can be
#: read on the same "further out is better" convention as dcor.
SCORES = {
    "dcor": ("dcor", lambda v: v,
             "Distance correlation with ground truth", "higher is better"),
    "procrustes": ("procrustes", lambda v: v,
                   "Procrustes disparity vs. ground truth", "lower is better"),
    "1-procrustes": ("procrustes", lambda v: 1.0 - v,
                     "1 - Procrustes disparity vs. ground truth",
                     "higher is better"),
}


def collect(figures_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """``{model: {version: {perspective: score}}}`` for every model in MODELS."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for base_model, subdir, _ in MODELS:
        rows = read_scores_csv(figures_root / subdir / "crosslevel_scores.csv")
        out[label_of(base_model)] = {
            version: {
                label: transform(
                    select_score(rows, level, metric, surrogate, field=column)
                )
                for label, level, surrogate, metric in PERSPECTIVES
            }
            for version, (column, transform, _, _) in SCORES.items()
        }
    return out


def _series(scores, version, labels):
    """One PlotSeries per model, values ordered to match *labels*."""
    return [
        make_series(
            [scores[label_of(base_model)][version][lbl] for lbl in labels],
            label=label_of(base_model), color=color, marker=marker, linestyle=ls,
        )
        for (base_model, _, color), (marker, ls) in zip(MODELS, ENCODINGS)
    ]


def draw_radar(scores, version: str, outdir: Path) -> Path:
    _, _, axis_label, direction = SCORES[version]
    labels = [p[0] for p in PERSPECTIVES]
    set_style("one_col", fig_width=6.0, fig_height=5.0)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(
        _series(scores, version, labels),
        axis_labels=labels,
        ax=ax,
        start_angle=90.0,      # Data at the top ...
        direction="ccw",       # ... then Structural left, Behavioral, Functional
        rlim=(0.0, 1.0),
        rticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        #   Four translucent polygons over a small range stack into one muddy
        #   wash, so the fill is faint enough to read as shading, not as ink.
        fill_alpha=0.05,
        #   Data and 1 - procrustes both sit at ~0.98, so their markers land on
        #   the outer ring; without the pad they cover the spoke labels.
        label_pad=12,
        title=f"{axis_label}\n({direction})",
        savefig=False,
    )
    #   The spoke labels sit outside the axes, and the left/right ones are the
    #   long words, so the axes box is narrowed to leave them room rather than
    #   relying on the tight bbox to find them.
    ax.set_position([0.16, 0.06, 0.62, 0.82])
    path = outdir / f"fig_taxonomy_radar_{version}.png"
    fig.savefig(path, pad_inches=0.25)
    plt.close(fig)
    return path


def draw_bars(scores, version: str, outdir: Path) -> Path:
    _, _, axis_label, direction = SCORES[version]
    set_style("one_col", fig_width=6.5, fig_height=3.6)
    fig, ax = plt.subplots()
    plot_grouped_bars(
        _series(scores, version, BAR_ORDER),
        group_labels=BAR_ORDER,
        ax=ax,
        ylabel=axis_label,
        ylim=(0.0, 1.0),
        annotate=True,
        annot_fmt="{:.3f}",
        #   Four bars per group leave no room for horizontal labels at three
        #   decimals; upright and small, they sit in the bar's own column.
        annot_size=6,
        annot_rotation=90,
        #   A fixed 0-1 axis leaves no headroom above a 0.98 bar, so tall bars
        #   take the label inside and short ones keep it above.
        annot_inside="auto",
        title=f"{axis_label} ({direction})",
        legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.12),
                       "ncol": 4, "frameon": False},
        savefig=False,
    )
    path = outdir / f"fig_taxonomy_bars_{version}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures-root", type=Path, default=REPO_ROOT / "figures",
                    help="directory holding the per-model simplex3 figure dirs")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    scores = collect(args.figures_root)
    for version in SCORES:
        print(draw_radar(scores, version, args.outdir))
        print(draw_bars(scores, version, args.outdir))


if __name__ == "__main__":
    main()
