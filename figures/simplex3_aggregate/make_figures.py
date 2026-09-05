#!/usr/bin/env python
"""Cross-model comparison of the simplex3 runs, one figure set per corpus.

Every simplex3 suite scores each taxonomy level against the ground-truth
mixture geometry and writes the result to ``crosslevel_scores.csv``. Each of
those files describes one base model in isolation. This driver reads them back
and puts each corpus's models on shared axes, so the question stops being "how
well does this level recover the mixture for Llama?" and becomes "does the
*same* level recover it for every base model, or is the answer
architecture-dependent?".

**One figure set per corpus, not one across all twelve.** Four models on a radar
is already crowded; twelve is unreadable. More to the point the corpora are not
on one axis: yahoo mixes three topic groups and dolly and oasst1 mix four, so
their ground truths live in different dimensions and their disparities are not
the same measurement. The cross-corpus question — does a level that recovers a
topic simplex also recover a task or language one — is answered by reading the
three figure sets side by side, each internally comparable.

A corpus whose suites have not run yet is **skipped with a note** rather than
being an error, so this driver is runnable from the moment the first corpus
finishes.

Nothing is recomputed here. The scores are read straight out of the CSVs the
suites already wrote, which is why this script runs in a second on a login node
and needs no cache, no GPU and no adapters.

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
                 **lower** is better. This is the disparity at the truth's own
                 dimension, ``d = K-1``: ``d2`` for yahoo's three vertices,
                 ``d3`` for the four-vertex corpora. The axis label says which,
                 because a d2 and a d3 disparity are not comparable numbers —
                 which is a second reason the corpora get separate figures.

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

#: Source run per base model: (base model, figure dir, colour, marker, dash).
#: The base model is the ``BASE_MODEL`` of that directory's own make_figures.py,
#: written out in full so the legend name is never guessed -- OLMo here is the
#: **1B** instruct model, not the 7B. The directory is the suite whose
#: crosslevel_scores.csv is authoritative for that model -- qwen_v4 rather than
#: qwen_v3 because v4 is the restricted run the llama figures are read against.
#:
#: The order is **ascending parameter count**, and it is the order of the bars
#: within each group and of the legend on every figure. Size is the one axis
#: these four models can be put on that is not arbitrary, so a trend across a
#: group can be read left to right; the count is in the model name in each case.
#:
#: Colours are Okabe-Ito, which is designed to stay separable under the common
#: colour-vision deficiencies. Marker and dash travel with the model rather than
#: with its position, so a reordering here cannot silently reassign them; the
#: radar overrides both (see draw_radar), so they are the model's assigned
#: encodings held for whatever draws them, not a promise that every figure does.
MODELS = [
    ("allenai/OLMo-2-0425-1B-Instruct",      "simplex3_olmo2",   "#E69F00", "^", "-."),   # 1B,  gold
    ("Qwen/Qwen3.5-4B",                      "simplex3_qwen_v4", "#CC79A7", "D", ":"),    # 4B,  purple
    ("meta-llama/Llama-3.1-8B-Instruct",     "simplex3_llama3i", "#0072B2", "o", "-"),    # 8B,  blue
    ("mistralai/Mistral-Nemo-Instruct-2407", "simplex3_nemo",    "#D55E00", "s", "--"),   # 12B, vermillion
]

#: The same four models over each corpus, and the directory each one's scores
#: are in. Built from MODELS rather than restated, so the colour, marker and
#: parameter-count ordering above stay the single source of all three — a model
#: keeps its encoding across every corpus, which is what lets the figure sets be
#: read side by side.
#:
#: ``file_token`` is appended to the output filenames. Yahoo's is empty, so its
#: six figures keep the names they are tracked under.
#: base model -> the suite tag its trees and drivers are named with. Yahoo's
#: directories predate the tag scheme and are named individually in MODELS, so
#: this is used only for the corpora added later.
SUITE_TAG = {
    "allenai/OLMo-2-0425-1B-Instruct": "olmo2",
    "Qwen/Qwen3.5-4B": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3i",
    "mistralai/Mistral-Nemo-Instruct-2407": "nemo",
}

CORPORA = {
    "yahoo": ("", MODELS),
    "dolly": ("_dolly", [(m, f"simplex3_dolly_{SUITE_TAG[m]}", c, k, l)
                         for m, _, c, k, l in MODELS]),
    "oasst1": ("_oasst1", [(m, f"simplex3_oasst1_{SUITE_TAG[m]}", c, k, l)
                           for m, _, c, k, l in MODELS]),
}


def label_of(base_model: str) -> str:
    """Legend name for a base model: the HF repo id without its org prefix."""
    return base_model.split("/")[-1]

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

#: Figure version -> (CSV column, transform, axis label).
#: The transform is applied to the column as read; ``1 - procrustes`` is the
#: only one that is not the identity, and it exists purely so a disparity can be
#: read on the same "further out is better" convention as dcor.
SCORES = {
    "dcor": ("dcor", lambda v: v, "Distance correlation with data mixture"),
    "procrustes": ("procrustes", lambda v: v,
                   "Procrustes disparity vs. data mixture"),
    "1-procrustes": ("procrustes", lambda v: 1.0 - v,
                     "1 - Procrustes disparity vs. data mixture"),
}


def collect(figures_root: Path, models) -> dict[str, dict[str, dict[str, float]]]:
    """``{model: {version: {perspective: score}}}`` for every model in *models*.

    Raises ``FileNotFoundError`` if any of the corpus's suites has not run;
    :func:`main` turns that into a skip, since a corpus mid-flight is the normal
    state of this directory rather than an error.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for base_model, subdir, *_ in models:
        rows = read_scores_csv(figures_root / subdir / "crosslevel_scores.csv")
        out[label_of(base_model)] = {
            version: {
                label: transform(
                    select_score(rows, level, metric, surrogate, field=column)
                )
                for label, level, surrogate, metric in PERSPECTIVES
            }
            for version, (column, transform, _) in SCORES.items()
        }
    return out


def _series(scores, models, version, labels, marker=None, linestyle=None):
    """One PlotSeries per model, values ordered to match *labels*.

    *marker* and *linestyle* override the per-model encodings in MODELS for
    every series at once, for a figure that wants one uniform stroke.
    """
    return [
        make_series(
            [scores[label_of(base_model)][version][lbl] for lbl in labels],
            label=label_of(base_model), color=color,
            marker=marker if marker is not None else mkr,
            linestyle=linestyle if linestyle is not None else ls,
        )
        for base_model, _, color, mkr, ls in models
    ]


def draw_radar(scores, models, version: str, outdir: Path, token: str,
               axis_note: str) -> Path:
    _, _, axis_label = SCORES[version]
    axis_label += axis_note
    labels = [p[0] for p in PERSPECTIVES]
    set_style("one_col", fig_width=4.4, fig_height=3.7)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plot_radar(
        #   Solid unmarked outlines: with four polygons crowded into the outer
        #   fifth of the range, markers and dashes read as texture rather than as
        #   identity, and colour alone separates the four cleanly enough.
        _series(scores, models, version, labels, marker="none", linestyle="-"),
        axis_labels=labels,
        ax=ax,
        start_angle=90.0,      # Data at the top ...
        direction="ccw",       # ... then Structural left, Behavioral, Functional
        rlim=(0.0, 1.0),
        rticks=[0.2, 0.4, 0.6, 0.8, 1.0],
        #   Four translucent polygons over a small range stack into one muddy
        #   wash whichever alpha is picked, so there is no fill at all.
        fill_alpha=0.0,
        #   Data and 1 - procrustes both sit at ~0.98, so their outlines run
        #   along the outer ring; without the pad they touch the spoke labels.
        label_pad=12,
        #   Straight ticked axes out to each spoke instead of concentric rings,
        #   numbered once on the Data spoke -- all four share the one scale.
        spoke_axes=True,
        rlabel_spoke=labels.index("Data"),
        title=axis_label,
        savefig=False,
    )
    #   The spoke labels sit outside the axes, and the left/right ones are the
    #   long words, so the axes box is narrowed to leave them room rather than
    #   relying on the tight bbox to find them.
    ax.set_position([0.16, 0.06, 0.62, 0.82])
    path = outdir / f"fig_taxonomy_radar_{version}{token}.png"
    fig.savefig(path, pad_inches=0.25)
    plt.close(fig)
    return path


def draw_bars(scores, models, version: str, outdir: Path, token: str,
              axis_note: str) -> Path:
    _, _, axis_label = SCORES[version]
    axis_label += axis_note
    set_style("one_col", fig_width=6.5, fig_height=3.6)
    fig, ax = plt.subplots()
    plot_grouped_bars(
        _series(scores, models, version, BAR_ORDER),
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
        title=axis_label,
        legend_kwargs={"loc": "upper center", "bbox_to_anchor": (0.5, -0.12),
                       "ncol": 4, "frameon": False},
        savefig=False,
    )
    path = outdir / f"fig_taxonomy_bars_{version}{token}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures-root", type=Path, default=REPO_ROOT / "figures",
                    help="directory holding the per-model simplex3 figure dirs")
    ap.add_argument("--outdir", type=Path, default=HERE)
    ap.add_argument("--corpus", action="append", choices=list(CORPORA),
                    help="restrict to one or more corpora (default: all that "
                         "have run)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    wanted = args.corpus or list(CORPORA)
    drawn = 0
    for corpus in wanted:
        token, models = CORPORA[corpus]
        try:
            scores = collect(args.figures_root, models)
        except FileNotFoundError as exc:
            # The normal state of a corpus whose suites are still queued. Named,
            # so a missing figure set is never mistaken for an empty result.
            print(f"{corpus}: skipped — {Path(exc.filename or '?').parent.name}"
                  f"/crosslevel_scores.csv not written yet")
            continue
        # The disparity is reported at the truth's own dimension, K-1, and a d2
        # number is not comparable with a d3 one. Read it off the CSV rather than
        # inferring it from the corpus name.
        rows = read_scores_csv(args.figures_root / models[0][1]
                               / "crosslevel_scores.csv")
        dims = sorted(int(k.split("_d")[1]) for k in rows[0]
                      if k.startswith("procrustes_d"))
        note = f" (d={dims[-1]})" if dims and "procrustes" in "".join(SCORES) else ""
        for version in SCORES:
            axis_note = note if "procrustes" in version else ""
            print(draw_radar(scores, models, version, args.outdir, token, axis_note))
            print(draw_bars(scores, models, version, args.outdir, token, axis_note))
        drawn += 1
    if not drawn:
        raise SystemExit(
            "no corpus has a complete set of crosslevel_scores.csv files; "
            "run the per-model drivers first.")


if __name__ == "__main__":
    main()
