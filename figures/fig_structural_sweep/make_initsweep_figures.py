#!/usr/bin/env python
"""How much of a taxonomy score is LoRA initialisation luck: the initsweep figures.

**initsweep** -- a sweep over ``lora_init_seed``, the seed passed to
``torch.manual_seed()`` immediately before PEFT initialises the LoRA ``A`` and
``B`` matrices, holding the corpus, the mixture grid, the training draw, the
dataset seeds, the rank and every other optimizer setting fixed.  See
``docs/terminology.md`` and ``docs/notes/init_seed_sweep.md``.

**surrogate** -- a read-time view of one taxonomy level (a choice of layers,
projections, pooling and metric), as opposed to the level itself.

**seed-siblings** -- the ten adapters trained on one mixture from ten
initialisations.  A *seed cloud* is where a surrogate puts them.

160 adapters: the 16-point 25% simplex over three yahoo topic groups on
OLMo-2-0425-1B-Instruct, at rank 16 on one 1000-row draw, trained from ten
initialisations.  This file is purely plotting -- every coordinate and every
alignment comes out of the three CSVs ``sweep_initsweep.py`` writes.

Three surrogates per figure
---------------------------
One canonical read per model level: ``structural_all_o`` (all layers, output
projections), ``functional_all`` (all hidden states) and one behavioral.  There
is no dataset row: the level is initialisation-free by construction, and on the
160-model pool its recipe-based identity cannot even tell the seed-siblings
apart.

**The two behavioral reads are never drawn on one axes.**  Every figure is
written twice, ``_sampled`` (per-query mean over the R=16 replicates) and
``_greedy`` (R=1 deterministic), because the pair differs by decoding alone and
overplotting them would invite reading the gap between them as a third level.
The gap is decode noise, and it is read by putting the two files side by side.

The figures
-----------
``fig_initsweep_replicates_<variant>.pdf``  (analysis A)
    Ten replicate scores per surrogate, one point per initialisation, against the
    fixed mixture simplex.  The band is the interquartile range and the solid
    line the median; the dashed line is the **before-embedding mean**, the score
    of the ten distance matrices averaged and scored as one collection, which is
    not a seed and so is not in the band.  Each panel is annotated with the full
    range across seeds -- the error bar the project's standing numbers have never
    carried.

``fig_initsweep_separation_<variant>.pdf``  (analysis B)
    Left: mean within-mixture distance against mean between-mixture distance over
    the 160-model pool, per surrogate, with standard-deviation whiskers.  A
    *within* pair is two seed-siblings; a *between* pair joins two mixtures.
    Right: their ratio per mixture, so a cloud that is tighter at a vertex than
    at the centre is visible rather than averaged away.  **The ratio is the
    figure's headline**: at 0 initialisation moves a model not at all, at 1 it
    moves it as far as changing what it was trained on.

``fig_initsweep_pool_<variant>.pdf``  (analysis C)
    One MDS fit over all 160 models, coloured by mixture, with a spoke from each
    model to its mixture's after-embedding mean.  This is the picture the two
    numbers summarise: whether the seed cloud is small against the simplex, or
    large enough to swallow the 25% grid spacing.

``fig_initsweep_means_<variant>.pdf``  (analysis C)
    The two means on one axes, joined per mixture.  The **after-embedding mean**
    is the centroid of a mixture's ten points in the 160-model fit; the
    **before-embedding mean** averages the ten distance matrices and embeds the
    result, which is a genuine 16-model "average model" collection and is scored
    in analysis A.  They disagree because MDS is not linear, and both are carried
    rather than one chosen.  The before-embedding fit is superimposed on the
    other upstream, so the gaps drawn here are shape and not rotation.

``fig_initsweep_overlay_<variant>.pdf``  (analysis C)
    The ten per-seed 16-point embeddings overlaid, coloured by mixture, with the
    before-embedding mean marked.  Each seed's embedding was Procrustes-aligned
    to that mean upstream: MDS fixes coordinates only up to rotation, reflection
    and scale, so ten raw embeddings on one axes would show a spread that is
    mostly arbitrary orientation.  **What is left after alignment is a genuine
    disagreement between initialisations**, and this is the only figure here that
    shows it per model rather than as a summary.

Reading the figures
-------------------
* **The two estimators run in opposite directions.**  A Procrustes disparity of 0
  means identical shape, while a dCor* of 1 means strong dependence on the
  ground truth.  They are never on one axis.

* **The spread is the result, not the level.**  A panel's height is the
  initialisation error bar on a number the project reports as a point estimate.
  Two surrogates whose bands overlap are not separated at n=16 by that estimator
  -- which is not the same as their measuring the same thing.

* **MDS axes are not interpretable** and carry no units, so the geometry panels
  are drawn with an equal aspect and without tick labels.  Only relative
  distances mean anything.

* **Ten seeds bound the variance of a score, not of a conclusion**, and this is
  n=16 on one base model, one corpus and one rank.  The group-size sweep showed
  both estimators are strongly size-dependent there.

Usage
-----
::

    # both behavioral variants, every figure
    python figures/fig_structural_sweep/make_initsweep_figures.py

    # one variant
    python figures/fig_structural_sweep/make_initsweep_figures.py --variant greedy

    # one figure, both variants
    python figures/fig_structural_sweep/make_initsweep_figures.py --figure overlay
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.plots.config import set_style  # noqa: E402
from src.plots.figures import save_figure  # noqa: E402
from src.plots.simplex import barycentric_color, mixture_weights  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Output format.  PDF rather than PNG: these are vector figures of at most 160
#: points, and the geometry panels are read by zooming into a seed cloud.
SUFFIX = ".pdf"

#: Opacity of one initialisation's point in the overlay. The per-mixture mean is
#: always drawn fully opaque, so this is what sets the contrast between the two;
#: ``--seed-alpha`` overrides it. Chosen so that ten coincident structural points
#: still read as one mark rather than as a dark blot.
SEED_ALPHA = 0.55

#: The two levels every variant shares, in the column order the rank,
#: group-size and nsweep figures use, so the four can be read against each other.
SHARED_LEVELS = [
    ("structural_all_o", "Structural\nall layers · o_proj"),
    ("functional_all", "Functional\nall hidden states"),
]

#: variant -> (perspective, label).  One behavioral read per figure, never both;
#: see the module docstring.
VARIANTS = {
    "sampled": ("behavioral", "Behavioral\nR=16 per query"),
    "greedy": ("behavioral_greedy", "Behavioral (greedy)\nR=1 deterministic"),
}

#: Perspective -> colour, following ``make_figures.py`` next door: one hue per
#: taxonomy family -- blue structural, gold functional, red behavioral.
LEVEL_COLORS = {
    "structural_all_o": "#08306B",
    "functional_all": "#B07800",
    "behavioral": "#A03000",
    "behavioral_greedy": "#D55E00",
}

#: ``(column stem, axis label)`` per estimator.  Disparity is drawn linearly
#: here, unlike the rank figure: this is one rank's worth of values rather than
#: two orders of magnitude, and a log axis would exaggerate a spread that is the
#: whole point of the figure.
SCORES = {
    "disparity": ("disparity", "Procrustes disparity\n(0 = identical shape)"),
    "dcor": ("dcor", "dCor*\n(1 = strong dependence)"),
}

#: ``init_seed`` of the before-embedding-mean row: a sentinel, not a seed.
#: Mirrors ``sweep_initsweep.MEAN_BEFORE``.
MEAN_BEFORE = -1

#: The ``mixture`` value of the per-level summary row in the separation CSV.
ALL_MIXTURES = "ALL"

FIGURES = ("replicates", "separation", "pool", "means", "overlay")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no input at {path}. Run sweep_initsweep.py first.")
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    return rows


def levels_for(variant: str) -> list[tuple[str, str]]:
    return [*SHARED_LEVELS, VARIANTS[variant]]


def short_mixture(mixture: str) -> str:
    """``yahoo_075g1_025g2_000g3_n1000_s00`` -> ``75/25/0``."""
    parts = [p for p in mixture.split("_") if p[:-2].isdigit() and p[-2] == "g"]
    return "/".join(str(int(p[:-2])) for p in parts)


def mixture_color(mixture: str):
    """The barycentric colour of a mixture: a pure vertex is its own anchor."""
    return barycentric_color(mixture_weights(mixture))


def out_path(outdir, stem, variant):
    return Path(outdir) / f"fig_initsweep_{stem}_{variant}{SUFFIX}"


def geometry_panel(ax, title):
    """MDS axes carry no units, so nothing is labelled and the aspect is equal."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.5)


def mixture_legend(fig, mixtures):
    """One swatch per mixture, in the order the simplex sorts them."""
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4,
                          color=mixture_color(m), label=short_mixture(m))
               for m in mixtures]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=8, frameon=False, fontsize=6,
               title="mixture  g1/g2/g3  (%)", title_fontsize=6.5,
               bbox_to_anchor=(0.5, -0.02))


# ── analysis A ────────────────────────────────────────────────────────────────

def draw_replicates(scores, variant, outdir):
    """Ten replicate scores per surrogate: the initialisation error bar."""
    levels = levels_for(variant)
    stems = list(SCORES)
    set_style("two_col_full")
    fig, axes = plt.subplots(len(stems), len(levels),
                             figsize=(2.3 * len(levels), 2.8 * len(stems)),
                             sharex=True, squeeze=False)

    for ri, key_score in enumerate(stems):
        stem, ylabel = SCORES[key_score]
        for ci, (key, label) in enumerate(levels):
            ax = axes[ri][ci]
            rows = [r for r in scores if r["perspective"] == key]
            per_seed = {int(r["init_seed"]): float(r[stem]) for r in rows
                        if int(r["init_seed"]) != MEAN_BEFORE}
            mean_rows = [float(r[stem]) for r in rows
                         if int(r["init_seed"]) == MEAN_BEFORE]
            if not per_seed:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="0.5")
                continue

            seeds = sorted(per_seed)
            vals = np.array([per_seed[s] for s in seeds])
            color = LEVEL_COLORS.get(key, "0.3")
            q1, med, q3 = np.percentile(vals, [25, 50, 75])

            ax.axhspan(q1, q3, color=color, alpha=0.13, lw=0, zorder=0)
            ax.axhline(med, color=color, lw=1.2, zorder=1)
            ax.plot(seeds, vals, ls="", marker="o", ms=4, color=color,
                    mec="white", mew=0.5, zorder=3)
            if mean_rows:
                # Not one of the ten, so it is a line rather than a point and
                # sits outside the band by construction.
                ax.axhline(mean_rows[0], color="0.25", lw=1.0, ls="--", zorder=2)

            spread = float(vals.max() - vals.min())
            ax.annotate(f"range {spread:.4f}", xy=(0.5, 0.97),
                        xycoords="axes fraction", ha="center", va="top",
                        fontsize=6.5, color="0.3")
            pad = max(spread, 1e-6) * 0.9
            ax.set_ylim(vals.min() - pad, vals.max() + pad)
            ax.set_xticks(seeds)
            ax.tick_params(axis="x", labelsize=6.5)
            if ri == 0:
                ax.set_title(label, fontsize=7.5)
            if ri == len(stems) - 1:
                ax.set_xlabel("LoRA initialisation seed $i$")
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=7.5)

    legend = [
        plt.Line2D([], [], marker="o", ls="", ms=4, color="0.3",
                   label="one initialisation"),
        plt.Line2D([], [], ls="-", lw=1.2, color="0.3", label="median · IQR band"),
        plt.Line2D([], [], ls="--", lw=1.0, color="0.25",
                   label="before-embedding mean (scored as one collection)"),
    ]
    fig.legend(legend, [h.get_label() for h in legend], loc="lower center",
               ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Agreement with the mixture simplex across ten LoRA initialisations\n"
        "OLMo-2-1B-Instruct · yahoo 3-group 25% simplex · 16 models per seed · "
        f"r=16 · {VARIANTS[variant][1].splitlines()[0].lower()} read",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out = out_path(outdir, "replicates", variant)
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


# ── analysis B ────────────────────────────────────────────────────────────────

def draw_separation(separation, variant, outdir):
    """Seed-siblings against mixture-neighbours, per level and per mixture.

    Three cells, not two.  A *between* pair -- two adapters of different
    mixtures -- may share an initialisation or not, and those sit far apart, so
    pooling them gives a mean that drifts with the number of seeds in the pool
    rather than describing the surrogate (see ``sweep_initsweep.py``,
    ``separation_rows``).  The left panel therefore draws the three cells the
    pool actually contains, and the ratio annotated on it and plotted on the
    right is ``within / between_same_seed``: same two mixtures' worth of
    distance, one initialisation against two.
    """
    levels = levels_for(variant)
    set_style("two_col_full")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.7]})

    # Left: the three cells per level, low to high.
    ax = axes[0]
    cells = [
        ("between_same_seed_mean", "between_same_seed_sd", "s", "white",
         "different mixture, one initialisation"),
        ("within_mean", "within_sd", "o", None,
         "same mixture, two initialisations (seed-siblings)"),
        ("between_diff_seed_mean", "between_diff_seed_sd", "D", None,
         "different mixture, two initialisations"),
    ]
    for i, (key, label) in enumerate(levels):
        row = next((r for r in separation if r["perspective"] == key
                    and r["mixture"] == ALL_MIXTURES), None)
        if row is None:
            continue
        color = LEVEL_COLORS.get(key, "0.3")
        ys = [float(row[m]) for m, _, _, _, _ in cells]
        ax.plot([i, i], [min(ys), max(ys)], color=color, lw=1.0, zorder=1)
        for (mkey, skey, marker, mfc, _), y in zip(cells, ys):
            ax.errorbar(i, y, yerr=float(row[skey]), fmt=marker, ms=5,
                        color=color, mfc=mfc or color, mec="white", mew=0.5,
                        capsize=2, lw=0.8, zorder=3)
        ax.annotate(f"ratio* {float(row['ratio_same_seed']):.2f}",
                    xy=(i, max(ys)), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=6.5, color="0.3")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([lab for _, lab in levels], fontsize=6.5)
    # Log, because the three levels are not on one distance scale: the
    # functional cells sit near 1e-3 where the structural ones sit near 1. A
    # linear axis would show only the structural level and three dots on zero.
    ax.set_yscale("log")
    ax.set_ylabel("pool distance  (log)", fontsize=7.5)
    ax.set_title("What moves an adapter further:\nits mixture, or its seed?",
                 fontsize=7.5)
    handles = [
        plt.Line2D([], [], marker=m, ls="", ms=5, color="0.3",
                   mfc=mfc or "0.3", label=lab)
        for m, mfc, lab in [(c[2], c[3], c[4]) for c in cells]
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=5.5)

    # Right: the ratio per mixture.
    ax = axes[1]
    mixtures = sorted({r["mixture"] for r in separation
                       if r["mixture"] != ALL_MIXTURES},
                      key=lambda m: tuple(-w for w in mixture_weights(m)))
    x = np.arange(len(mixtures))
    for key, label in levels:
        by_mix = {r["mixture"]: float(r["ratio_same_seed"]) for r in separation
                  if r["perspective"] == key and r["mixture"] != ALL_MIXTURES}
        if not by_mix:
            continue
        ys = [by_mix.get(m, np.nan) for m in mixtures]
        ax.plot(x, ys, "-o", ms=3, lw=1.2, color=LEVEL_COLORS.get(key, "0.3"),
                label=label.replace("\n", " · "))
    ax.axhline(1.0, color="0.55", lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([short_mixture(m) for m in mixtures], rotation=60,
                       ha="right", fontsize=6)
    ax.set_xlabel("mixture  g1/g2/g3  (%)", fontsize=7.5)
    ax.set_ylabel("ratio*   seed-siblings / one-seed", fontsize=7.5)
    ax.set_title("Is the seed cloud tighter at some mixtures?", fontsize=7.5)
    # The band between the functional and behavioral tracks is empty the whole
    # width of the panel, and no track crosses it.
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.42), frameon=False,
              fontsize=6)

    fig.suptitle(
        "How far initialisation moves a model, against how far its mixture does"
        "\n160 adapters · 16 mixtures x 10 initialisations · "
        "dotted line = re-seeding moves an adapter exactly as far as re-mixing it",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out_path(outdir, "separation", variant)
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


# ── analysis C ────────────────────────────────────────────────────────────────

def coords_of(geometry, perspective, kind, seed=None):
    """``{model or mixture -> (x, y)}`` and the mixture of each, for one panel."""
    pts, mix = {}, {}
    for r in geometry:
        if r["perspective"] != perspective or r["kind"] != kind:
            continue
        if seed is not None and int(r["init_seed"]) != seed:
            continue
        pts[r["model_id"]] = (float(r["dim1"]), float(r["dim2"]))
        mix[r["model_id"]] = r["mixture"]
    return pts, mix


def sorted_mixtures(geometry, perspective):
    """The 16 mixtures, vertices first, as the simplex figures order them."""
    found = {r["mixture"] for r in geometry if r["perspective"] == perspective}
    return sorted(found, key=lambda m: tuple(-w for w in mixture_weights(m)))


def draw_pool(geometry, variant, outdir):
    """All 160 models in one MDS fit, with a spoke to each mixture's mean."""
    levels = levels_for(variant)
    set_style("two_col_full")
    fig, axes = plt.subplots(1, len(levels), figsize=(2.6 * len(levels), 3.2),
                             squeeze=False)

    drew = False
    for ci, (key, label) in enumerate(levels):
        ax = axes[0][ci]
        pts, mix = coords_of(geometry, key, "pool160")
        means, _ = coords_of(geometry, key, "mean_after")
        geometry_panel(ax, label)
        if not pts:
            ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="0.5")
            continue
        drew = True
        for model, (x, y) in pts.items():
            color = mixture_color(mix[model])
            mean = means.get(mix[model])
            if mean is not None:
                ax.plot([x, mean[0]], [y, mean[1]], "-", color=color, lw=0.4,
                        alpha=0.5, zorder=1)
            ax.plot(x, y, "o", ms=2.6, color=color, alpha=0.85, zorder=2)
        for m, (x, y) in means.items():
            ax.plot(x, y, "o", ms=6, color=mixture_color(m), mec="0.15",
                    mew=0.7, zorder=3)

    if drew:
        mixture_legend(fig, sorted_mixtures(geometry, levels[0][0]))
    fig.suptitle(
        "The seed cloud against the simplex: 160 models in one MDS fit\n"
        "small point = one initialisation · ringed point = that mixture's "
        "after-embedding mean · spoke = the distance between them",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    out = out_path(outdir, "pool", variant)
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def draw_means(geometry, variant, outdir):
    """The two means on one axes, joined per mixture."""
    levels = levels_for(variant)
    set_style("two_col_full")
    fig, axes = plt.subplots(1, len(levels), figsize=(2.6 * len(levels), 3.2),
                             squeeze=False)

    drew = False
    for ci, (key, label) in enumerate(levels):
        ax = axes[0][ci]
        # Both sides of the superposition, not the raw ``mean_after``: the fit
        # scales both configurations to unit norm, so the raw after-mean is on
        # the pool's own scale and would not be comparable with the aligned
        # before-mean. That is the pool figure's frame, not this one's.
        after, _ = coords_of(geometry, key, "mean_after_aligned")
        before, bmix = coords_of(geometry, key, "mean_before_aligned")
        geometry_panel(ax, label)
        if not after or not before:
            ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="0.5")
            continue
        drew = True
        by_mixture = defaultdict(dict)
        for m, xy in after.items():
            by_mixture[m]["after"] = xy
        for model, xy in before.items():
            by_mixture[bmix[model]]["before"] = xy
        for m, pair in by_mixture.items():
            if "after" not in pair or "before" not in pair:
                continue
            color = mixture_color(m)
            (ax_, ay), (bx, by) = pair["after"], pair["before"]
            ax.plot([ax_, bx], [ay, by], "-", color=color, lw=0.8, alpha=0.8,
                    zorder=1)
            ax.plot(ax_, ay, "o", ms=5, color=color, mec="0.15", mew=0.7,
                    zorder=3)
            ax.plot(bx, by, "^", ms=5, color=color, mec="0.15", mew=0.7,
                    zorder=3)

    if drew:
        mixture_legend(fig, sorted_mixtures(geometry, levels[0][0]))
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=5, color="0.3", mec="0.15",
                   label="after-embedding mean (centroid of the ten points)"),
        plt.Line2D([], [], marker="^", ls="", ms=5, color="0.3", mec="0.15",
                   label="before-embedding mean (the ten matrices averaged, "
                         "then embedded)"),
    ]
    # Figure-level, not on the first panel: the panels are filled edge to edge
    # by construction (equal aspect, no ticks, points scaled to fill), so there
    # is no corner inside one that a legend does not cover.
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=6,
               ncol=2, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle(
        "The two means over initialisations, superimposed and joined per mixture"
        "\nthey differ because MDS is not linear; the rotation between the two "
        "fits is already removed",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.88))
    out = out_path(outdir, "means", variant)
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def draw_overlay(geometry, variant, outdir, seed_alpha=SEED_ALPHA):
    """The ten per-seed embeddings on one axes, against their per-mixture mean.

    Each seed's sixteen adapters are embedded against *each other* and nobody
    else -- ten 16x16 matrices, ten MDS fits -- and the ten configurations are
    Procrustes-superimposed before they are drawn, because MDS fixes coordinates
    only up to rotation, reflection and scale.  The mean marker is the centroid
    of each mixture's ten aligned points, so it is the centre of exactly the
    cloud drawn around it.

    *seed_alpha* fades the individual seeds against that mean.  It is a knob
    because the right value depends on the level: at 0.85 the structural panel
    reads as a single crisp point per mixture and the sampled behavioral panel is
    a solid mass, and no one setting serves both.
    """
    levels = levels_for(variant)
    set_style("two_col_full")
    fig, axes = plt.subplots(1, len(levels), figsize=(2.6 * len(levels), 3.2),
                             squeeze=False)

    drew = False
    for ci, (key, label) in enumerate(levels):
        ax = axes[0][ci]
        pts, mix = coords_of(geometry, key, "seed_aligned")
        # ``seed_mean``, not ``mean_before``: the alignment scaled every
        # configuration to unit norm, and the raw mean is on the surrogate's own
        # scale, which differs from it by three orders of magnitude on the
        # functional row. See ``sweep_initsweep.py`` for the full note.
        ref, refmix = coords_of(geometry, key, "seed_mean")
        geometry_panel(ax, label)
        if not pts:
            ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="0.5")
            continue
        drew = True
        anchor = {refmix[m]: xy for m, xy in ref.items()}
        for model, (x, y) in pts.items():
            color = mixture_color(mix[model])
            base = anchor.get(mix[model])
            if base is not None:
                ax.plot([x, base[0]], [y, base[1]], "-", color=color, lw=0.35,
                        alpha=seed_alpha * 0.55, zorder=1)
            ax.plot(x, y, "o", ms=2.6, color=color, alpha=seed_alpha, zorder=3)
        # Hollow, and drawn under the points: on the structural and functional
        # rows every seed lands inside the marker, and a filled mean would hide
        # the ten points it is meant to be compared against -- an empty panel and
        # a perfectly reproducible one would look identical.
        for m, (x, y) in anchor.items():
            ax.plot(x, y, "D", ms=7.0, mfc="none", mec=mixture_color(m),
                    mew=1.3, alpha=1.0, zorder=4)

    if drew:
        mixture_legend(fig, sorted_mixtures(geometry, levels[0][0]))
    fig.suptitle(
        "Ten per-seed embeddings overlaid, each Procrustes-aligned to the others"
        "\nsmall point = one initialisation's placement of one mixture "
        "· diamond = that mixture's mean over the ten aligned points",
        fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    out = out_path(outdir, "overlay", variant)
    save_figure(fig, out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(HERE / "initsweep_scores.csv"))
    ap.add_argument("--separation", default=str(HERE / "initsweep_separation.csv"))
    ap.add_argument("--geometry", default=str(HERE / "initsweep_geometry.csv"))
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--variant", choices=[*VARIANTS, "both"], default="both",
                    help="which behavioral read to draw. The two are never on "
                         "one axes; 'both' writes a file of each")
    ap.add_argument("--figure", action="append", dest="figures",
                    choices=FIGURES,
                    help="repeatable; default is every figure")
    ap.add_argument("--seed-alpha", type=float, default=SEED_ALPHA,
                    help="opacity of one initialisation's point in the overlay; "
                         "the per-mixture mean is always opaque "
                         f"(default {SEED_ALPHA})")
    args = ap.parse_args()
    if not 0.0 < args.seed_alpha <= 1.0:
        raise SystemExit("--seed-alpha must be in (0, 1]")

    variants = list(VARIANTS) if args.variant == "both" else [args.variant]
    wanted = args.figures or list(FIGURES)

    scores = read_rows(Path(args.scores))
    geometry = read_rows(Path(args.geometry)) if {"pool", "means", "overlay"} \
        & set(wanted) else []
    separation = read_rows(Path(args.separation)) if "separation" in wanted else []

    for variant in variants:
        present = {r["perspective"] for r in scores}
        missing = [k for k, _ in levels_for(variant) if k not in present]
        if missing:
            # A missing level would silently drop a column from the headline
            # figure, which is a scoring gap rather than a plotting choice.
            raise SystemExit(
                f"{args.scores} is missing {missing} for the {variant} variant; "
                f"re-run sweep_initsweep.py, or pass --variant "
                f"{'greedy' if variant == 'sampled' else 'sampled'}.")
        if "replicates" in wanted:
            draw_replicates(scores, variant, args.outdir)
        if "separation" in wanted:
            draw_separation(separation, variant, args.outdir)
        if "pool" in wanted:
            draw_pool(geometry, variant, args.outdir)
        if "means" in wanted:
            draw_means(geometry, variant, args.outdir)
        if "overlay" in wanted:
            draw_overlay(geometry, variant, args.outdir,
                         seed_alpha=args.seed_alpha)


if __name__ == "__main__":
    main()
