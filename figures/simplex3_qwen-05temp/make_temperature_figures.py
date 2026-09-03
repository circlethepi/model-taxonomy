#!/usr/bin/env python
"""Behavioral distance matrices and MDS embeddings across the Qwen temperature sweep.

The ``simplex3_qwen`` suite decoded its 16 adapters at ten sampling temperatures
(T = 0.1 … 1.0, R=8 each) alongside the greedy run, and embedded all of it — but
until now no figure read any of it. Every behavioral figure in
``figures/simplex3_qwen_v*`` shows exactly two slices: greedy, and T=1.0 at R=16.

This script draws the sweep: one grid per slice, three surrogates by four metrics
(cosine, frobenius, euclidean, CKA), scored against the ground-truth simplex by
distance correlation and Procrustes disparity.

Why R=8 everywhere
------------------
R=16 exists at T=1.0 and nowhere else. Reading it there and R=8 at the other nine
would vary the replicate count alongside the temperature, and the sampling noise
in a per-query mean scales with the replicate count — so one point of the curve
would have half the noise of the others, and the dip or rise at T=1.0 could not
be attributed to temperature. The sweep was generated at R=8 uniformly for this
reason (``docs/CHANGELOG.md``), including a deliberate second T=1.0 run at R=8
beside the existing R=16 one. This script reads that one. The R=16 slice keeps
its own figures in ``figures/simplex3_qwen_v3``.

Greedy carries no temperature — ``do_sample=False`` makes the parameter inert —
so it is one slice among eleven rather than a per-temperature repeat, and it
leads the cross-temperature figure as the zero-noise baseline the sweep departs
from.

Reuse
-----
Everything here comes from ``src/plots/simplex_suite.py``: the selector
schema, the per-row metric loop that resolves each row's tensors once, both grid
figures, the two scores, and the ``06_pairwise`` read-through cache. This module
contributes a temperature→hash lookup, the slice/surrogate tables, and the
markdown report. Nothing is reimplemented.

Orientation
-----------
Every MDS panel is drawn in the simplex's own frame — centre mixture at the
origin, pure g1 straight up, pure g2 to the right, the orientation
``ternary_legend`` uses. That is now ``mds_grid``'s default rather than anything
this script does; see :func:`src.plots.simplex.align_to_simplex`.

Usage
-----
    python figures/simplex3_qwen-05temp/make_temperature_figures.py
    python figures/simplex3_qwen-05temp/make_temperature_figures.py --no-cache
    python figures/simplex3_qwen-05temp/make_temperature_figures.py --check-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# `simplex_suite` pins the BLAS thread count to 1 before it imports numpy, and
# that is load-bearing rather than tidy: unpinned, the behavioral level did not
# finish inside 50 minutes on this host. Importing numpy — or anything that
# imports numpy — above this line would let the default thread count win. So
# this import comes first, and everything numeric comes from it.
#
# The suite machinery used to live in `scripts/make_simplex3_figures.py`, which
# is now a thin Qwen driver over this module. Importing the driver would still
# work, but the module-level assignments below (`S.METRIC_COLS`, `S.SUITE_CACHE`)
# have to land on the module the builders actually read, so import that module.
import src.plots.simplex_suite as S  # noqa: E402
from gen_simplex3 import temp_token  # noqa: E402

import numpy as np  # noqa: E402

#: The four metrics this figure set is about. Setting the module-level names is
#: what restricts the *whole* pipeline: `metric_row`, `emit` and `_blocked` all
#: read `S.METRIC_COLS`, so there is no per-call column list to keep in sync and
#: no way for one of them to drift onto the seven-metric grid.
METRICS = ("cosine", "frobenius", "euclidean", "cka")

#: The replicate count every sampled slice is read at. See the module docstring.
SWEEP_REPLICATES = 8

#: The grid rows: what a model's generations are reduced to before distancing.
#: `per generation` keeps every replicate as its own row, `per query` averages a
#: query's replicates back to one row each, `model mean` collapses to a single
#: row per model. They are ordered coarsest-last, which is also least-to-most
#: aggregated.
SURROGATES = {
    "per generation": {},
    "per query": {"replicate_reduction": "mean", "representation": "matrix",
                  "renormalize": True},
    "model mean": {"replicate_reduction": "mean", "representation": "mean",
                   "renormalize": True},
}

#: Greedy has one replicate, so averaging a query's replicates is the identity
#: and `per query` would be a byte-identical copy of `per generation` under a
#: label implying otherwise. `behavioral_cells` drops it for the same reason.
GREEDY_SURROGATES = ("per generation", "model mean")

OUTDIR = Path(__file__).resolve().parent


def temperature_hashes(cache_root: Path, base_slug: str, draw: dict,
                       replicates: int = SWEEP_REPLICATES) -> dict[float, str]:
    """``{temperature: sampling_hash}`` for the sweep, read off one adapter.

    The hashes are not spelled here on purpose. A ``sampling_hash`` digests
    ``{do_sample, temperature, top_p, top_k, generation_seed}``, so ten literal
    digests in this file would be ten chances to transcribe a slice wrong and no
    way to notice — the figure would render, under the right title, from the
    wrong generations. Reading them back from the runs that declare their own
    temperature makes that failure impossible.

    Any adapter will do: the sweep ran the identical sampling grid on all 16, and
    :func:`main` checks that every adapter resolves before anything is computed.
    """
    from src.cache._draw import draw_name

    root = (cache_root / "05_generated" / base_slug)
    # `draw_name` is the one place that knows a prompt format enters the path as
    # `_f{id}`. Spelling the stem out here instead would look in a directory that
    # does not exist — which is exactly what it did on the first run.
    stem = (f"{draw['recipe_hash']}/"
            + draw_name(draw["n_samples"], draw["seed"], draw["prompt_format_id"]))
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


def build_slices(hashes: dict[float, str]) -> dict[str, tuple[str, int, str]]:
    """``{tag: (label, replicates, sampling_hash)}`` — greedy first, then by T.

    Greedy leads because it is the zero-noise end of the axis the sweep runs
    along, not because it is one of the ten. ``temp_token`` supplies the sampled
    tags; it is the same function that named the sweep's experiment YAMLs and
    SLURM jobs, so a figure's filename points at the job that generated the text
    it is drawn from.
    """
    slices = {"greedy": ("greedy", 1, S.SAMP_GREEDY)}
    for temperature, sampling in hashes.items():
        slices[temp_token(temperature)] = (
            f"T={temperature:.1f}", SWEEP_REPLICATES, sampling)
    return slices


def selector(replicates: int, sampling: str, surrogate: str) -> dict:
    """The behavioral selector for one (slice, surrogate) pair.

    The base is the dict ``S.behavioral_cells`` builds; the surrogate overrides
    only the reduction and the representation.
    """
    base = {"draw": S.DRAW, "max_new_tokens": S.MAX_NEW_TOKENS,
            "replicates": replicates, "sampling_hash": sampling,
            "embedder_hash": S.EMBEDDER, "replicate_reduction": "all",
            "view": "matrix", "normalize": "none", "representation": "matrix"}
    return dict(base, **SURROGATES[surrogate])


def surrogates_for(tag: str) -> tuple[str, ...]:
    """Which surrogate rows a slice's grid carries."""
    return GREEDY_SURROGATES if tag == "greedy" else tuple(SURROGATES)


def _bold_row(values: dict[str, float], best) -> dict[str, str]:
    """Format one row of a metric table, bolding the cell *best* picks out.

    The comparison is at the printed precision, not on the raw float. `euclidean`
    is `frobenius` without the normalization, so the two are a positive rescaling
    of each other and every score here is scale-invariant — they are equal in
    exact arithmetic and differ only in the last bits. Bolding on `==` would then
    mark one of two visibly identical cells and leave the reader looking for a
    difference that is not there. Ties are printed as ties.
    """
    if not values:
        return {}
    target = round(best(values.values()), 4)
    return {m: S._bold_if(v, round(v, 4) == target) for m, v in values.items()}


def _score_table(by_row, attr: str, best, heading: str, note: str) -> list[str]:
    """One markdown table: rows x metrics, with the best cell of each row bold."""
    lines = [f"## {heading}", "", note, "",
             "| slice · surrogate | " + " | ".join(METRICS) + " |",
             "|" + "---|" * (1 + len(METRICS))]
    for row, cells in by_row.items():
        values = {m: getattr(cells[m], attr) for m in METRICS if m in cells}
        formatted = _bold_row(values, best)
        lines.append(f"| {row} | "
                     + " | ".join(formatted.get(m, "—") for m in METRICS) + " |")
    lines.append("")
    return lines


def write_agreement_md(ranked, row_order, winners, path: Path) -> None:
    """The scores as separate dCor and Procrustes tables, in slice order.

    ``write_scores_csv`` is for diffing and keeps ``rank_surrogates``' dCor
    ordering. This is for reading, and the two scores run in opposite directions
    — dCor 0→1 better, the Procrustes residual 1→0 better — so putting them in
    one table means every row has to be read in two directions at once, and the
    bolding would mean opposite things in adjacent columns. Split, each table has
    one direction and the bold cell is unambiguously that row's best metric.

    Both tables and the csv are written from the same `SurrogateScore` objects,
    so they cannot disagree about a number.
    """
    scores: dict[str, dict[str, object]] = {}
    for s in ranked:
        scores.setdefault(s.row, {})[s.col] = s
    by_row = {row: scores[row] for row in row_order if row in scores}

    lines = ["# Behavioral level across the temperature sweep", ""]
    lines += _score_table(
        by_row, "dcor", max, "Distance correlation vs the ground-truth simplex",
        "Higher is better; the bold cell is each row's best metric. dCor scores "
        "the distance matrix directly and never embeds, so it is unaffected by "
        "the MDS fit.")
    lines += _score_table(
        by_row, "procrustes", min, "Procrustes residual vs the ground-truth simplex",
        "**Lower** is better; the bold cell is each row's best metric. The "
        "residual scores the 2-D MDS configuration each panel draws, so unlike "
        "dCor it inherits the distortion `stress` reports below.")
    lines += _score_table(
        by_row, "stress", min, "Kruskal stress of the MDS fit",
        "Lower is better. This is the fit each Procrustes residual above "
        "describes — a high stress means that row's residual is scoring a "
        "configuration that represents its distance matrix poorly.")

    # What the cross-temperature figure actually shows, as numbers. Without this
    # the figure asserts a winner per slice and nothing says which cell it was.
    lines += ["## Best cell per slice", "",
              "The panels of `fig_behavioral_best_dcor_mds.png`, chosen by dCor "
              "across every surrogate and metric at that slice.", "",
              "| slice | surrogate | metric | dCor | Procrustes | stress |",
              "|---|---|---|---|---|---|"]
    for label, s in winners:
        lines.append(f"| {label} | {s.row.split(' · ', 1)[1]} | {s.col} | "
                     f"{s.dcor:.4f} | {s.procrustes:.4f} | {s.stress:.4f} |")
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def preflight(idx, ids, slices) -> None:
    """Resolve one sampled per-query row and assert the shape the suite assumes.

    The R=8 per-query mean must reduce 800 rows to 100. If ``replicates`` were
    wrong the selector would still resolve — to a different, real slice — and the
    error would surface as a figure that looks plausible and is not the one it
    claims to be. Checking the shape once here turns that into an exception.
    """
    tag = next(t for t in slices if t != "greedy")
    label, replicates, sampling = slices[tag]
    sel = selector(replicates, sampling, "per query")
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
    if n_rows != S.DRAW["n_samples"]:
        raise SystemExit(
            f"pre-flight: {label} per query has {n_rows} rows, expected "
            f"{S.DRAW['n_samples']} — a per-query mean over "
            f"{SWEEP_REPLICATES} replicates should give one row per query, so "
            f"{n_rows} means the replicate reduction did not happen")
    if not all(np.isfinite(np.asarray(r.matrix)).all() for r in reps):
        raise SystemExit(f"pre-flight: {label} per query has non-finite values")
    print(f"pre-flight: {label} · per query → {len(reps)} × ({n_rows}, {dim}), "
          "all finite")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(OUTDIR))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass pairwise reads; results are still written back")
    ap.add_argument("--check-only", action="store_true",
                    help="resolve and validate the slices, then stop before plotting")
    args = ap.parse_args()

    S.METRICS = {k: S.METRICS[k] for k in METRICS}
    S.METRIC_COLS = list(S.METRICS)

    cache_root = (Path(args.cache_root).expanduser().resolve()
                  if args.cache_root else S.CACHE_ROOT)
    if not cache_root.exists():
        raise SystemExit(f"no cache at {cache_root} — pass --cache-root")
    # Off until set: the module initialises `SUITE_CACHE` disabled so that
    # importing it never writes to a cache.
    S.SUITE_CACHE = S.SuiteCache(cache_root, read=not args.no_cache)

    hashes = temperature_hashes(cache_root, S.BASE_SLUG, S.DRAW)
    expected = [round(0.1 * i, 1) for i in range(1, 11)]
    if list(hashes) != expected:
        raise SystemExit(
            f"expected the ten sweep temperatures {expected} at "
            f"R={SWEEP_REPLICATES}, found {list(hashes)} — sweep incomplete?")
    print(f"cache: {cache_root}")
    print("temperatures: " + ", ".join(f"{t:.1f}→{h}" for t, h in hashes.items()))

    idx = S.scan_cache(str(cache_root), base_model_id=S.BASE_MODEL,
                       behavioral_draw=S.DRAW, functional_draw=S.DRAW)
    ids = S.sort_by_mixture(idx.model_ids)
    if len(ids) != 16:
        raise SystemExit(f"expected 16 models, found {len(ids)} — cache incomplete?")
    print(f"models: {len(ids)}")

    slices = build_slices(hashes)
    preflight(idx, ids, slices)
    if args.check_only:
        print("check-only: slices resolve, stopping before plotting")
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    S.set_style("two_col_full")

    fig, ax = S.plt.subplots(figsize=(4.2, 4.0))
    S.ternary_legend(ax, ids, label_models=True)
    ax.set_title("simplex3 mixtures — barycentric colour key", fontsize=9)
    S.save_figure(fig, str(outdir / "fig_ternary_legend.png"))
    S.plt.close("all")

    #: (slice tag, surrogate, metric) -> DistanceMatrix, computed once and then
    #: viewed three ways: per-slice grids, the cross-slice overview, and scoring.
    dms: dict[tuple[str, str, str], object] = {}
    for tag, (label, replicates, sampling) in slices.items():
        for surrogate in surrogates_for(tag):
            print(f"{label} · {surrogate} …")
            sel = selector(replicates, sampling, surrogate)
            blocked = S._blocked(sel["representation"] == "mean", None)
            got = S.metric_row(idx, "behavioral", ids, None, blocked,
                               label=f"{label} · {surrogate}",
                               behavioral_selector=sel)
            for col, dm in got.items():
                dms[(tag, surrogate, col)] = dm

    # One grid per slice: surrogates down, metrics across.
    for tag, (label, _replicates, _sampling) in slices.items():
        rows = list(surrogates_for(tag))
        cells = {(surrogate, col): dms[(tag, surrogate, col)]
                 for surrogate in rows for col in METRICS}
        S.emit(f"behavioral_{tag}", rows, cells, outdir,
               f"Behavioral level · {label}"
               + ("" if tag == "greedy" else f" · R={SWEEP_REPLICATES}"))

    # The sweep at a glance: one row per slice at a single surrogate, so the
    # eleven slices can be compared without opening eleven files. Greedy shows
    # `per generation`, which is the representation its `per query` would be.
    overview_rows, overview = [], {}
    for tag, (label, _replicates, _sampling) in slices.items():
        surrogate = "per generation" if tag == "greedy" else "per query"
        row = f"{label} · {surrogate}"
        overview_rows.append(row)
        overview.update({(row, col): dms[(tag, surrogate, col)] for col in METRICS})
    S.emit("behavioral_temps", overview_rows, overview, outdir,
           "Behavioral level across sampling temperature")

    print("scoring …")
    scored = {(tag, surrogate, col): dm for (tag, surrogate, col), dm in dms.items()}
    labelled = {(f"{slices[tag][0]} · {surrogate}", col): dm
                for (tag, surrogate, col), dm in scored.items()}
    ranked = S.rank_surrogates(labelled, ids)
    S.write_scores_csv({"behavioral": ranked}, outdir / "temperature_scores.csv")

    # The cross-slice figure: each slice's best-scoring cell, greedy first.
    by_key = {(s.row, s.col): s for s in ranked}
    winners = []
    for tag, (label, _replicates, _sampling) in slices.items():
        candidates = [by_key[(f"{label} · {surrogate}", col)]
                      for surrogate in surrogates_for(tag) for col in METRICS
                      if (f"{label} · {surrogate}", col) in by_key]
        if not candidates:
            print(f"    {label}: no scorable cell — omitted")
            continue
        winners.append((label, max(candidates, key=lambda s: s.dcor)))

    # `crosslevel_mds`'s defaults are tuned for the four panels it was written
    # for. Eleven in a row need a taller panel: each title here runs to four
    # lines (a two-line name over two score lines) and the figure height is
    # `panel_h + 1.15`, so at the default 3.5 the titles ran into the suptitle.
    # The *width* stays at the default: the score line's 12 pt is tuned to a
    # 3.5" panel, and narrowing to 2.9 to keep the strip shorter pushed
    # "dCor ... · Procrustes ..." into the neighbouring panel's title.
    S.crosslevel_mds(
        [(f"{label}\n{s.row.split(' · ', 1)[1]} · {s.col}", s.dm, s.dcor, s.procrustes)
         for label, s in winners],
        "Behavioral simplex recovery across sampling temperature — best cell per slice",
        subtitle=f"greedy, then T=0.1…1.0 at R={SWEEP_REPLICATES}",
        savepath=outdir / "fig_behavioral_best_dcor_mds.png",
        panel_h=5.0,
        random_state=S.MDS_SEED,
    )
    S.plt.close("all")

    row_order = [f"{slices[tag][0]} · {surrogate}"
                 for tag in slices for surrogate in surrogates_for(tag)]
    write_agreement_md(ranked, row_order, winners,
                       outdir / "temperature_agreement.md")

    n_png = len(list(outdir.glob("*.png")))
    print(f"\nwrote {n_png} figures and {len(ranked)} scored cells to {outdir}")
    print(S.SUITE_CACHE.report())


if __name__ == "__main__":
    main()
