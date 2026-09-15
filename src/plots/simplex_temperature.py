#!/usr/bin/env python
"""The behavioral level read across a sampling-temperature sweep, for any suite.

A ``simplex3`` suite that carries ``temperature_sweep`` decodes its sixteen
adapters at ten sampling temperatures (T = 0.1 … 1.0) alongside the greedy run
and embeds all of it.  This module turns that into figures: per-slice grids, a
cross-temperature overview, one MDS strip along the temperature axis, the two
agreement curves, and the score tables behind them.

It is the sweep counterpart of :mod:`src.plots.simplex_suite`, and it is a
consumer of that module rather than a parallel implementation — the selector
schema, the per-row metric loop, both grid figures, the two scores and the
``06_pairwise`` read-through cache all come from there.  What lives here is the
part that is specific to a *sweep*: a temperature→hash lookup, the slice and
surrogate tables, the markdown report, and the curve figures.

Terminology
-----------
slice
    One decoding point of the sweep — the greedy run, or one temperature at one
    replicate count.  Eleven of them in a full sweep.
surrogate
    What a model's generations are reduced to before any distance is taken.
    ``per generation`` keeps every replicate as its own row, ``per query``
    averages a query's replicates back to one row each, and ``model mean``
    collapses a model to a single row.  Each pools a different amount of the
    sampling noise away.
perspective
    A surrogate together with a metric — one cell of a grid.  The same word
    :mod:`src.plots.simplex_suite` uses.
canonical perspective
    The one perspective a level is read at when a figure wants a single number
    rather than a grid.  At the behavioral level that is ``per query`` ×
    ``cosine``, which is what ``figures/figure2`` draws.

What is parameterized
---------------------
Everything that differs between two suites: the base model and the query draw,
the corpus / training-draw / mixture filters the scan needs, the replicate count
the sweep ran at, and — the parameter this module exists for — *which
perspectives to build*.  A driver that wants the full grid passes every
surrogate and every metric; a driver that wants the canonical read passes one of
each and gets one cell per temperature.

Why one replicate count throughout a sweep
------------------------------------------
R=16 exists at T=1.0 and nowhere else, because that is the main behavioral run.
Reading it there and R=8 at the other nine would vary the replicate count
alongside the temperature, and the sampling noise in a per-query mean scales with
the replicate count — so one point of the curve would have half the noise of the
others, and a dip or rise at T=1.0 could not be attributed to temperature.  The
sweeps were generated at a uniform R for this reason, including a deliberate
second T=1.0 run at the sweep's R beside the existing R=16 one.
:func:`temperature_hashes` reads that one and skips the R=16 entry.

Greedy carries no temperature — ``do_sample=False`` makes the parameter inert —
so it is one slice among eleven rather than a per-temperature repeat, and it
leads every figure as the zero-noise baseline the sweep departs from.

Orientation
-----------
Every MDS panel is drawn in the simplex's own frame — centre mixture at the
origin, pure g1 straight up, pure g2 to the right, the orientation
``ternary_legend`` uses.  That is ``mds_grid``'s default; see
:func:`src.plots.simplex.align_to_simplex`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `simplex_suite` pins the BLAS thread count to 1 before it imports numpy, and
# that is load-bearing rather than tidy: unpinned, the behavioral level did not
# finish inside 50 minutes on this host.  Importing numpy — or anything that
# imports numpy — above this line would let the default thread count win.  So
# this import comes first, and everything numeric comes through it.
import src.plots.simplex_suite as S  # noqa: E402

# `temp_token` lives in the generator that named the sweep's jobs, which is a
# script rather than a package module.  Importing it here is what keeps a
# figure's filename pointing at the job that produced the text it is drawn from.
sys.path.insert(0, str(S.REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from src.plots import (  # noqa: E402
    PALETTE, encoding_legend, make_series, plot_lines, set_style,
)
from src.plots.figures import _get_fig_ax  # noqa: E402

#: Every metric a sweep grid can carry, in the order the columns are drawn.
#: A driver narrows this; nothing here assumes all four are present.
METRICS = ("cosine", "frobenius", "euclidean", "cka")

#: The replicate count a sweep slice is read at unless a driver says otherwise.
#: See the module docstring for why it is uniform across the sweep.
SWEEP_REPLICATES = 8

#: Surrogate → the selector overrides that build it, on top of the base
#: behavioral selector.  Ordered coarsest-last, which is also least-to-most
#: aggregated.
SURROGATES = {
    "per generation": {},
    "per query": {"replicate_reduction": "mean", "representation": "matrix",
                  "renormalize": True},
    "model mean": {"replicate_reduction": "mean", "representation": "mean",
                   "renormalize": True},
}

#: The behavioral level's **canonical perspective** — the project's default
#: reading of this level, and what a single-cell run of the sweep uses.  It is
#: the row ``docs/terminology.md`` designates and ``canonical_perspectives()`` in
#: ``figures/simplex_collection_size/sweep_group_size.py`` fixes: a per-query
#: replicate mean, compared by cosine.  Named here rather than in a driver so the
#: sweep and ``figures/figure2``'s Behavioral panel cannot drift apart.
#:
#: The designated perspective says R=16; a sweep reads it at the sweep's own
#: uniform R, for the reason in this module's docstring.  That is a difference in
#: how much sampling noise is averaged away, not in what is being measured, and
#: it is why a sweep point is not directly comparable to a main run's number —
#: see the R=8 / R=16 gap at T=1.0 on the OLMo sweep.
CANONICAL_SURROGATE = "per query"
CANONICAL_METRIC = "cosine"

#: Greedy has one replicate, so averaging a query's replicates is the identity
#: and ``per query`` would be a byte-identical copy of ``per generation`` under a
#: label implying otherwise.  ``S.behavioral_cells`` drops it for the same reason.
#:
#: A single-cell run over the canonical ``per query`` perspective is the one case
#: where dropping it would cost something real: it would delete the T=0 baseline
#: from every figure rather than de-duplicate a row.  :func:`surrogates_for`
#: substitutes ``per generation`` there — the identical representation — and
#: :func:`slice_row_label` says so in the label, so no figure claims greedy was
#: averaged over replicates it does not have.
GREEDY_SURROGATES = ("per generation", "model mean")

#: Surrogate → curve colour, and metric → curve linestyle.  Colour carries the
#: pooling method and dashes carry the metric, so a reader comparing pooling
#: scans colours and a reader comparing metrics scans dash patterns, without
#: either question needing a legend with one entry per line.
SURROGATE_COLORS = {
    "per generation": PALETTE[0],
    "per query": PALETTE[1],
    "model mean": PALETTE[2],
}
METRIC_STYLES = {
    "cosine": "-",
    "frobenius": "--",
    "euclidean": "-.",
    "cka": ":",
}

#: Score column → (axis label, figure filename).  dCor runs 0→1 better and the
#: Procrustes residual 1→0 better, so the two figures read inversely.
SCORES = {
    "dcor": ("Distance correlation with the simplex", "fig_temperature_dcor.png"),
    "procrustes": ("Procrustes disparity (lower is better)",
                   "fig_temperature_procrustes.png"),
}

#: Where the two curve keys sit, in axes coordinates: outside the plotting area
#: on the right, stacked, so neither covers a curve.
LEGEND_LOCS = [(1.03, 0.50), (1.03, 0.00)]

#: Where the greedy slice is drawn on the temperature axis, and what its tick
#: says instead of the number.  ``do_sample=False`` makes the temperature
#: parameter inert, so the greedy slice carries no temperature of its own; zero
#: is where it belongs anyway, as the limit the sampled runs approach — a
#: temperature low enough always picks the argmax token, which is what greedy
#: decoding does.  The tick is labelled rather than numbered so the figure does
#: not claim the sweep was run at T=0.
GREEDY_TEMPERATURE = 0.0
GREEDY_TICK_LABEL = "greedy"

#: The tag :func:`build_slices` gives the greedy slice.  Spelled once because
#: three functions test for it.
GREEDY_TAG = "greedy"


# ── Slices ────────────────────────────────────────────────────────────────────

def temperature_hashes(cache_root, base_slug, draw, replicates=SWEEP_REPLICATES,
                       model_ids=None) -> dict[float, str]:
    """``{temperature: sampling_hash}`` for the sweep, read off one adapter.

    The hashes are not spelled anywhere in this repository on purpose.  A
    ``sampling_hash`` digests ``{do_sample, temperature, top_p, top_k,
    generation_seed}``, so ten literal digests in a driver would be ten chances
    to transcribe a slice wrong and no way to notice — the figure would render,
    under the right title, from the wrong generations.  Reading them back from
    the runs that declare their own temperature makes that failure impossible.

    Any adapter of the sweep will do: the identical sampling grid ran on all
    sixteen, and :func:`preflight` checks that every adapter resolves before
    anything is computed.  *model_ids* says which adapters are the sweep's, and
    is not optional once a base model carries more than one experiment —
    ``05_generated/<base_slug>`` holds every generation for that checkpoint, so
    an unfiltered glob can land on a pool or nsweep adapter that was never swept
    and report no temperatures at all.  ``None`` takes the first directory that
    has runs, which is what a single-experiment cache always gave.
    """
    import json

    from src.cache._draw import draw_name

    root = Path(cache_root) / "05_generated" / base_slug
    # `draw_name` is the one place that knows a prompt format enters the path as
    # `_f{id}`.  Spelling the stem out here instead would look in a directory
    # that does not exist — which is exactly what it did on the first run.
    stem = (f"{draw['recipe_hash']}/"
            + draw_name(draw["n_samples"], draw["seed"], draw["prompt_format_id"]))
    if model_ids:
        # A model id is the adapter's absolute path under `03_adapters`; the
        # generated text for it sits under the same *basename* in `05_generated`.
        run_dirs = [p for p in (root / Path(m).name / stem / "runs"
                                for m in model_ids) if p.is_dir()]
    else:
        run_dirs = sorted(root.glob(f"*/{stem}/runs"))
    if not run_dirs:
        raise SystemExit(f"no behavioral runs under {root}/*/{stem}/runs")

    found: dict[float, str] = {}
    for path in sorted(run_dirs[0].glob("*.json")):
        run = json.loads(path.read_text())
        sampling = run.get("sampling") or {}
        if not sampling.get("do_sample"):
            continue                      # the greedy run, handled separately
        if int(run["replicates"]) != replicates:
            continue                      # the R=16 T=1.0 run, deliberately skipped
        temperature = round(float(sampling["temperature"]), 4)
        found[temperature] = run["sampling_hash"]
    return dict(sorted(found.items()))


def build_slices(hashes: dict[float, str],
                 replicates=SWEEP_REPLICATES) -> dict[str, tuple[str, int, str]]:
    """``{tag: (label, replicates, sampling_hash)}`` — greedy first, then by T.

    Greedy leads because it is the zero-noise end of the axis the sweep runs
    along, not because it is one of the ten.  ``temp_token`` supplies the sampled
    tags; it is the same function that named the sweep's experiment YAMLs and
    SLURM jobs, so a figure's filename points at the job that generated the text
    it is drawn from.
    """
    from gen_simplex3 import temp_token

    slices = {GREEDY_TAG: (GREEDY_TAG, 1, S.SAMP_GREEDY)}
    for temperature, sampling in hashes.items():
        slices[temp_token(temperature)] = (
            f"T={temperature:.1f}", replicates, sampling)
    return slices


def selector(replicates: int, sampling: str, surrogate: str) -> dict:
    """The behavioral selector for one (slice, surrogate) pair.

    The base is the dict ``S.behavioral_cells`` builds; the surrogate overrides
    only the reduction and the representation.  ``S.DRAW`` and ``S.EMBEDDER`` are
    read at call time rather than captured, because
    :func:`run_temperature_suite` sets them from its arguments first.
    """
    base = {"draw": S.DRAW, "max_new_tokens": S.MAX_NEW_TOKENS,
            "replicates": replicates, "sampling_hash": sampling,
            "embedder_hash": S.EMBEDDER, "replicate_reduction": "all",
            "view": "matrix", "normalize": "none", "representation": "matrix"}
    return dict(base, **SURROGATES[surrogate])


def surrogates_for(tag: str, surrogates) -> tuple[str, ...]:
    """Which surrogate rows slice *tag* carries, out of the requested *surrogates*.

    Greedy drops ``per query`` for the reason :data:`GREEDY_SURROGATES` records,
    unless that is the only surrogate asked for — then it is substituted by the
    representation it is identical to, rather than leaving the sweep without its
    T=0 baseline.
    """
    if tag != GREEDY_TAG:
        return tuple(surrogates)
    kept = tuple(s for s in surrogates if s in GREEDY_SURROGATES)
    if kept:
        return kept
    return ("per generation",)


def slice_row_label(tag: str, label: str, surrogate: str, requested) -> str:
    """A row's name in the tables and figures: ``T=0.4 · per query``.

    When greedy stands in ``per generation`` for a ``per query`` that does not
    exist, the label says so, so no reader takes the row for a mean over
    replicates greedy never had.
    """
    if tag == GREEDY_TAG and surrogate not in requested:
        return f"{label} · {surrogate} (R=1: no per-query mean)"
    return f"{label} · {surrogate}"


# ── Score tables ──────────────────────────────────────────────────────────────

def _bold_row(values: dict[str, float], best) -> dict[str, str]:
    """Format one row of a metric table, bolding the cell *best* picks out.

    The comparison is at the printed precision, not on the raw float.
    ``euclidean`` is ``frobenius`` without the normalization, so the two are a
    positive rescaling of each other and every score here is scale-invariant —
    they are equal in exact arithmetic and differ only in the last bits.  Bolding
    on ``==`` would then mark one of two visibly identical cells and leave the
    reader looking for a difference that is not there.  Ties are printed as ties.
    """
    if not values:
        return {}
    if len(values) == 1:
        # One metric column: "this row's best metric" is the only metric, and
        # bolding every cell in the table says nothing.
        return {m: f"{v:.4f}" for m, v in values.items()}
    target = round(best(values.values()), 4)
    return {m: S._bold_if(v, round(v, 4) == target) for m, v in values.items()}


def _score_table(by_row, metrics, attr: str, best, heading: str,
                 note: str) -> list[str]:
    """One markdown table: rows x metrics, with the best cell of each row bold."""
    lines = [f"## {heading}", "", note, "",
             "| slice · surrogate | " + " | ".join(metrics) + " |",
             "|" + "---|" * (1 + len(metrics))]
    for row, cells in by_row.items():
        values = {m: getattr(cells[m], attr) for m in metrics if m in cells}
        formatted = _bold_row(values, best)
        lines.append(f"| {row} | "
                     + " | ".join(formatted.get(m, "—") for m in metrics) + " |")
    lines.append("")
    return lines


def write_agreement_md(ranked, row_order, winners, metrics, path: Path,
                       strip_heading: str, strip_note: str, title: str) -> None:
    """The scores as separate dCor and Procrustes tables, in slice order.

    ``write_scores_csv`` is for diffing and keeps ``rank_surrogates``' dCor
    ordering.  This is for reading, and the two scores run in opposite directions
    — dCor 0→1 better, the Procrustes residual 1→0 better — so putting them in
    one table means every row has to be read in two directions at once, and the
    bolding would mean opposite things in adjacent columns.  Split, each table
    has one direction and the bold cell is unambiguously that row's best metric.

    Both tables and the csv are written from the same ``SurrogateScore`` objects,
    so they cannot disagree about a number.
    """
    scores: dict[str, dict[str, object]] = {}
    for s in ranked:
        scores.setdefault(s.row, {})[s.col] = s
    by_row = {row: scores[row] for row in row_order if row in scores}

    lines = [f"# {title}", ""]
    # The bold cell marks a row's best *metric*, so the sentence promising one is
    # only true with more than one metric column to choose between.
    bold = ("; the bold cell is each row's best metric" if len(metrics) > 1
            else "")
    lines += _score_table(
        by_row, metrics, "dcor", max,
        "Distance correlation vs the ground-truth simplex",
        f"Higher is better{bold}. dCor scores "
        "the distance matrix directly and never embeds, so it is unaffected by "
        "the MDS fit.")
    lines += _score_table(
        by_row, metrics, "procrustes", min,
        "Procrustes residual vs the ground-truth simplex",
        f"**Lower** is better{bold}. The "
        "residual scores the 2-D MDS configuration each panel draws, so unlike "
        "dCor it inherits the distortion `stress` reports below.")
    lines += _score_table(
        by_row, metrics, "stress", min, "Kruskal stress of the MDS fit",
        "Lower is better. This is the fit each Procrustes residual above "
        "describes — a high stress means that row's residual is scoring a "
        "configuration that represents its distance matrix poorly.")

    # What the MDS strip actually shows, as numbers.  Without this the figure
    # asserts a cell per slice and nothing says which one it was.
    lines += [f"## {strip_heading}", "", strip_note, "",
              "| slice | surrogate | metric | dCor | Procrustes | stress |",
              "|---|---|---|---|---|---|"]
    for label, s in winners:
        lines.append(f"| {label} | {s.row.split(' · ', 1)[1]} | {s.col} | "
                     f"{s.dcor:.4f} | {s.procrustes:.4f} | {s.stress:.4f} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


# ── Curves ────────────────────────────────────────────────────────────────────

def read_scores(path: Path):
    """``(temperatures, {(surrogate, metric): {temperature: row}})`` from the CSV.

    The greedy slice is placed at :data:`GREEDY_TEMPERATURE`.  Every other x
    position is whatever the rows themselves declare rather than a range spelled
    here, so a re-run over a different sweep needs no edit.

    A greedy row whose surrogate carries the substituted-label suffix is read
    under its bare surrogate name, so it lands on the curve it belongs to rather
    than starting a line of its own with one point.

    The file is parsed by :func:`src.plots.simplex_suite.read_scores_csv` rather
    than by a plain ``DictReader``, and that is load-bearing: the disparity
    columns are written per MDS dimension (``procrustes_d2`` …) and only that
    reader back-fills the unsuffixed ``procrustes`` and ``stress`` names these
    curves ask for, from the truth's own dimension. Reading the file directly
    worked until the dimension sweep landed and then raised ``KeyError``.
    """
    cells: dict[tuple[str, str], dict[float, dict]] = {}
    temps: set[float] = set()
    for row in S.read_scores_csv(path):
        slice_label, surrogate = row["surrogate"].split(" · ", 1)
        surrogate = surrogate.split(" (", 1)[0]
        if slice_label.startswith("T="):
            temp = float(slice_label.removeprefix("T="))
        elif slice_label == GREEDY_TAG:
            temp = GREEDY_TEMPERATURE
        else:
            continue
        temps.add(temp)
        cells.setdefault((surrogate, row["metric"]), {})[temp] = row
    return sorted(temps), cells


def curve_figure(temps, cells, column: str, ylabel: str, savepath: Path,
                 title: str | None = None) -> Path:
    """One line per (surrogate, metric) cell, over the temperature axis.

    A cell missing at some temperature becomes NaN, which matplotlib leaves as a
    gap rather than interpolating across.  ``model mean`` has no CKA row at any
    temperature — the surrogate leaves one row per model, and a CKA between two
    single rows is degenerate — so that cell is absent from the CSV rather than
    zero, and its line is simply not drawn.

    With a single cell the two encoding keys would each carry one entry and
    state nothing; the cell is named in the title instead.
    """
    series, drawn = [], []
    for surrogate, color in SURROGATE_COLORS.items():
        for metric, linestyle in METRIC_STYLES.items():
            by_temp = cells.get((surrogate, metric))
            if not by_temp:
                continue
            ys = [float(by_temp[t][column]) if t in by_temp else np.nan
                  for t in temps]
            # No label: the two keys below state the encoding instead.
            series.append(make_series(ys, color=color, linestyle=linestyle))
            drawn.append((surrogate, metric))

    fig, ax = _get_fig_ax(None)
    plot_lines(temps, series=series, ax=ax, xlabel="Sampling temperature",
               ylabel=ylabel, savepath=savepath)
    ax.set_xticks(temps)
    ax.set_xticklabels([GREEDY_TICK_LABEL if t == GREEDY_TEMPERATURE else f"{t:g}"
                        for t in temps])
    legends = []
    if len(drawn) > 1:
        legends = encoding_legend(
            ax,
            ("Pooling", {s: {"color": c} for s, c in SURROGATE_COLORS.items()
                         if any(s == d for d, _ in drawn)}),
            ("Metric", {m: {"color": "0.3", "linestyle": ls}
                        for m, ls in METRIC_STYLES.items()
                        if any(m == d for _, d in drawn)}),
            loc=LEGEND_LOCS,
        )
    if title is not None:
        ax.set_title(title)
    elif len(drawn) == 1:
        ax.set_title(f"{drawn[0][0]} · {drawn[0][1]}")
    # `save_figure` saves under the rcParams bbox, which crops the keys where
    # they hang off the axes; naming them as extra artists keeps them whole.
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, bbox_inches="tight", bbox_extra_artists=legends)
    S.plt.close(fig)
    return savepath


def write_curves(outdir: Path, scores_csv: Path, title: str | None = None):
    """Both agreement-vs-temperature figures, from a scores CSV on disk.

    Reading the CSV back rather than taking the scores in memory is deliberate:
    it keeps the curves reproducible from a run that has already happened, which
    is what makes ``--curves-only`` a second's work instead of a recompute.
    """
    set_style("two_col", fig_width=8.0, fig_height=5.0)
    # Lines distinguished partly by dash pattern: the preset's default markers
    # are wide enough to cover the pattern between two points.
    S.plt.rcParams["lines.markersize"] = 4
    temps, cells = read_scores(scores_csv)
    written = []
    for column, (ylabel, filename) in SCORES.items():
        written.append(curve_figure(temps, cells, column, ylabel,
                                    Path(outdir) / filename, title=title))
    return written


# ── Pre-flight ────────────────────────────────────────────────────────────────

def preflight(idx, ids, slices, surrogates, replicates) -> None:
    """Resolve one sampled row and assert the shape the suite assumes.

    A per-query mean must reduce ``replicates × n_queries`` rows to
    ``n_queries``.  If ``replicates`` were wrong the selector would still resolve
    — to a different, real slice — and the error would surface as a figure that
    looks plausible and is not the one it claims to be.  Checking the shape once
    here turns that into an exception.

    The check runs on ``per query`` when it was asked for, because that is the
    surrogate whose row count states whether the reduction happened at all.  With
    only ``per generation`` requested there is no reduction to check and the
    expected row count is the full stack.
    """
    tag = next(t for t in slices if t != GREEDY_TAG)
    label, slice_replicates, sampling = slices[tag]
    surrogate = "per query" if "per query" in surrogates else list(surrogates)[0]
    sel = selector(slice_replicates, sampling, surrogate)
    reps, _order = S.resolve_ordered(idx, "behavioral", ids,
                                     behavioral_selector=sel)[:2]
    # `resolve_ordered` hands back `ModelRepresentation` objects, not bare
    # arrays; `.matrix` is the (n_queries, d) array underneath.
    shapes = {tuple(np.asarray(r.matrix).shape) for r in reps}
    if len(reps) != len(ids) or len(shapes) != 1:
        raise SystemExit(
            f"pre-flight: {label} resolved to {len(reps)} representations with "
            f"shapes {sorted(shapes)} — expected {len(ids)} of one shape")
    (n_rows, dim), = shapes
    expected = S.DRAW["n_samples"] if surrogate != "per generation" \
        else S.DRAW["n_samples"] * replicates
    if surrogate == "model mean":
        expected = 1
    if n_rows != expected:
        raise SystemExit(
            f"pre-flight: {label} {surrogate} has {n_rows} rows, expected "
            f"{expected} — a {surrogate} read over {replicates} replicates of "
            f"{S.DRAW['n_samples']} queries should give {expected}, so "
            f"{n_rows} means the replicate reduction did not happen as assumed")
    if not all(np.isfinite(np.asarray(r.matrix)).all() for r in reps):
        raise SystemExit(f"pre-flight: {label} {surrogate} has non-finite values")
    print(f"pre-flight: {label} · {surrogate} → {len(reps)} × ({n_rows}, {dim}), "
          "all finite")


# ── The run ───────────────────────────────────────────────────────────────────

def run_temperature_suite(
    *,
    base_model,
    draw,
    outdir,
    cache_root=None,
    datasets=None,
    train_draw=None,
    mixtures=None,
    embedder=None,
    sweep_replicates=SWEEP_REPLICATES,
    surrogates=tuple(SURROGATES),
    metrics=METRICS,
    expected_temperatures=None,
    n_expected=16,
    no_cache=False,
    check_only=False,
    per_slice_grids=True,
    ternary_key=True,
    curves=True,
    strip_figure="fig_behavioral_best_dcor_mds.png",
    strip_title="Behavioral simplex recovery across sampling temperature — "
                "best cell per slice",
    report_title="Behavioral level across the temperature sweep",
    scores_name="temperature_scores.csv",
    agreement_name="temperature_agreement.md",
    source=None,
):
    """Build the temperature-sweep figure set for one run.

    *datasets*, *train_draw* and *mixtures* are the same three scan filters
    :func:`src.plots.simplex_suite.run_suite` takes, and are needed for the same
    reason: ``03_adapters/<base_slug>`` holds every adapter ever trained on that
    checkpoint, whatever corpus, however much training and at whatever mixture
    resolution, so an unfiltered scan of a cache that has grown a second
    experiment returns far more than sixteen models and *n_expected* trips.

    *surrogates* and *metrics* choose the perspectives.  Passing one of each is
    the canonical single-cell read: every figure then carries one panel per
    temperature instead of a grid, which is why *per_slice_grids* exists — a
    1x1 "grid" per slice is eleven files that each repeat one panel of the
    cross-slice figure.

    Returns ``(ranked, ids)``: every scored cell, dCor-descending, and the model
    ids in mixture order.
    """
    surrogates = tuple(surrogates)
    metrics = tuple(metrics)
    unknown = set(surrogates) - set(SURROGATES)
    if unknown:
        raise SystemExit(f"unknown surrogate(s) {sorted(unknown)}; "
                         f"expected a subset of {list(SURROGATES)}")
    unknown = set(metrics) - set(S.METRICS)
    if unknown:
        raise SystemExit(f"unknown metric(s) {sorted(unknown)}; "
                         f"expected a subset of {list(S.METRICS)}")

    # Restricting the module-level metric table is what narrows the *whole*
    # pipeline: `metric_row`, `emit` and `_blocked` all read `S.METRIC_COLS`, so
    # there is no per-call column list to keep in sync and no way for one of them
    # to drift onto the seven-metric grid.
    S.METRICS = {k: S.METRICS[k] for k in metrics}
    S.METRIC_COLS = list(S.METRICS)

    S.BASE_MODEL = base_model
    S.BASE_SLUG = base_model.replace("/", "--")
    S.DRAW = dict(draw)
    if embedder is not None:
        S.EMBEDDER = embedder
    if source is not None:
        S.SOURCE = source

    cache_root = (Path(cache_root).expanduser().resolve()
                  if cache_root else S.CACHE_ROOT)
    if not cache_root.exists():
        raise SystemExit(f"no cache at {cache_root} — pass --cache-root")
    S.CACHE_ROOT = cache_root
    S.ADAPTER_ROOT = cache_root / "03_adapters"
    # Off until set: the module initialises `SUITE_CACHE` disabled so that
    # importing it never writes to a cache.
    S.SUITE_CACHE = S.SuiteCache(cache_root, read=not no_cache)

    # The architecture is deliberately *not* applied.  It decides the structural
    # and functional grids' row names, and this module builds neither — the
    # behavioral level names a sampling regime and a reduction, which no
    # checkpoint config can change.
    print(f"cache: {cache_root}")
    print(f"model: {base_model}")

    idx = S.scan_cache(str(cache_root), base_model_id=base_model,
                       behavioral_draw=S.DRAW, functional_draw=S.DRAW,
                       datasets=datasets)
    if train_draw is not None:
        n_samples, seed = train_draw
        idx = idx.filter(n_samples=n_samples, seed=seed)
    if mixtures is not None:
        from src.plots.simplex import raw_mixture_pcts
        wanted = {tuple(m) for m in mixtures}
        idx = S.CacheIndex([e for e in idx.entries
                            if raw_mixture_pcts(e.model_id) in wanted],
                           idx.cache_root)
    if len(idx.model_ids) != n_expected:
        raise SystemExit(
            f"expected {n_expected} models, found {len(idx.model_ids)} — pass "
            f"datasets=/train_draw=/mixtures= to say which experiment under "
            f"{base_model} this figure is about, or fix the cache")
    ids = S.sort_by_mixture(idx.model_ids)
    print(f"models: {len(ids)}")

    hashes = temperature_hashes(cache_root, S.BASE_SLUG, S.DRAW,
                                replicates=sweep_replicates, model_ids=ids)
    if expected_temperatures is not None:
        if list(hashes) != list(expected_temperatures):
            raise SystemExit(
                f"expected the sweep temperatures {list(expected_temperatures)} "
                f"at R={sweep_replicates}, found {list(hashes)} — sweep incomplete?")
    if not hashes:
        raise SystemExit(f"no sampled runs at R={sweep_replicates} — sweep not run?")
    print("temperatures: " + ", ".join(f"{t:.1f}→{h}" for t, h in hashes.items()))

    slices = build_slices(hashes, replicates=sweep_replicates)
    preflight(idx, ids, slices, surrogates, sweep_replicates)
    if check_only:
        print("check-only: slices resolve, stopping before plotting")
        return None, ids

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S.set_style("two_col_full")

    if ternary_key:
        fig, ax = S.plt.subplots(figsize=(4.2, 4.0))
        S.ternary_legend(ax, ids, label_models=True)
        ax.set_title("simplex3 mixtures — barycentric colour key", fontsize=9)
        S.save_figure(fig, str(outdir / "fig_ternary_legend.png"))
        S.plt.close("all")

    #: (slice tag, surrogate, metric) -> DistanceMatrix, computed once and then
    #: viewed three ways: per-slice grids, the cross-slice overview, and scoring.
    dms: dict[tuple[str, str, str], object] = {}
    for tag, (label, replicates, sampling) in slices.items():
        for surrogate in surrogates_for(tag, surrogates):
            row = slice_row_label(tag, label, surrogate, surrogates)
            print(f"{row} …")
            sel = selector(replicates, sampling, surrogate)
            blocked = S._blocked(sel["representation"] == "mean", None)
            got = S.metric_row(idx, "behavioral", ids, None, blocked,
                               label=row, behavioral_selector=sel)
            for col, dm in got.items():
                dms[(tag, surrogate, col)] = dm

    # One grid per slice: surrogates down, metrics across.
    if per_slice_grids:
        for tag, (label, _replicates, _sampling) in slices.items():
            rows = list(surrogates_for(tag, surrogates))
            cells = {(surrogate, col): dms[(tag, surrogate, col)]
                     for surrogate in rows for col in metrics}
            S.emit(f"behavioral_{tag}", rows, cells, outdir,
                   f"Behavioral level · {label}"
                   + ("" if tag == GREEDY_TAG else f" · R={sweep_replicates}"))

    # The sweep at a glance: one row per slice at a single surrogate, so the
    # eleven slices can be compared without opening eleven files.  The canonical
    # surrogate when it was asked for, since that is the row every other figure
    # of the project reads the behavioral level at; otherwise the finest one
    # present.  Greedy shows whatever `surrogates_for` left it.
    overview_surrogate = (CANONICAL_SURROGATE if CANONICAL_SURROGATE in surrogates
                          else surrogates[0])
    overview_rows, overview = [], {}
    for tag, (label, _replicates, _sampling) in slices.items():
        surrogate = (overview_surrogate if overview_surrogate in surrogates_for(tag, surrogates)
                     else surrogates_for(tag, surrogates)[0])
        row = slice_row_label(tag, label, surrogate, surrogates)
        overview_rows.append(row)
        overview.update({(row, col): dms[(tag, surrogate, col)] for col in metrics})
    S.emit("behavioral_temps", overview_rows, overview, outdir,
           "Behavioral level across sampling temperature")

    print("scoring …")
    labelled = {(slice_row_label(tag, slices[tag][0], surrogate, surrogates), col): dm
                for (tag, surrogate, col), dm in dms.items()}
    ranked = S.rank_surrogates(labelled, ids)
    S.write_scores_csv({"behavioral": ranked}, outdir / scores_name)

    # The MDS strip: one panel per slice, greedy first.  With several
    # perspectives per slice the panel is that slice's best-scoring cell; with
    # one it is the only cell, and `max` over a single candidate is that cell.
    by_key = {(s.row, s.col): s for s in ranked}
    single_cell = len(surrogates) == 1 and len(metrics) == 1
    winners = []
    for tag, (label, _replicates, _sampling) in slices.items():
        candidates = [
            by_key[(slice_row_label(tag, label, surrogate, surrogates), col)]
            for surrogate in surrogates_for(tag, surrogates) for col in metrics
            if (slice_row_label(tag, label, surrogate, surrogates), col) in by_key
        ]
        if not candidates:
            print(f"    {label}: no scorable cell — omitted")
            continue
        winners.append((label, max(candidates, key=lambda s: s.dcor)))

    # `crosslevel_mds`'s defaults are tuned for the four panels it was written
    # for.  Eleven in a row need a taller panel: each title here runs to four
    # lines (a two-line name over two score lines) and the figure height is
    # `panel_h + 1.15`, so at the default 3.5 the titles ran into the suptitle.
    # The *width* stays at the default: the score line's 12 pt is tuned to a
    # 3.5" panel, and narrowing to 2.9 to keep the strip shorter pushed
    # "dCor ... · Procrustes ..." into the neighbouring panel's title.
    S.crosslevel_mds(
        [(f"{label}\n{s.row.split(' · ', 1)[1]} · {s.col}", s.dm, s.dcor, s.procrustes)
         for label, s in winners],
        strip_title,
        subtitle=f"greedy, then T={min(hashes):.1f}…{max(hashes):.1f} "
                 f"at R={sweep_replicates}",
        savepath=outdir / strip_figure,
        panel_h=5.0,
        random_state=S.MDS_SEED,
    )
    S.plt.close("all")

    row_order = [slice_row_label(tag, slices[tag][0], surrogate, surrogates)
                 for tag in slices
                 for surrogate in surrogates_for(tag, surrogates)]
    write_agreement_md(
        ranked, row_order, winners, metrics, outdir / agreement_name,
        strip_heading="Cell per slice" if single_cell else "Best cell per slice",
        strip_note=(f"The panels of `{strip_figure}`: the single perspective "
                    f"this run reads the behavioral level at, per slice."
                    if single_cell else
                    f"The panels of `{strip_figure}`, chosen by dCor across "
                    f"every surrogate and metric at that slice."),
        title=report_title)

    if curves:
        write_curves(outdir, outdir / scores_name)

    n_png = len(list(outdir.glob("*.png")))
    print(f"\nwrote {n_png} figures and {len(ranked)} scored cells to {outdir}")
    print(S.SUITE_CACHE.report())
    return ranked, ids
