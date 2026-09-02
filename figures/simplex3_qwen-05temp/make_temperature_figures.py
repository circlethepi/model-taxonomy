#!/usr/bin/env python
"""Behavioral distance matrices and MDS embeddings across the Qwen temperature sweep.

The ``simplex3_qwen`` suite decoded its 16 adapters at ten sampling temperatures
(T = 0.1 … 1.0, R=8 each) alongside the greedy run, and embedded all of it — but
until now no figure read any of it. Every behavioral figure in
``figures/simplex3_qwen_v*`` shows exactly two slices: greedy, and T=1.0 at R=16.

This script draws the sweep. One row per temperature plus greedy, four metrics
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
so it is one row and one figure set, not ten identical copies.

Reuse
-----
Everything here comes from ``scripts/make_simplex3_figures.py``: the selector
schema, the per-row metric loop that resolves each row's tensors once, both grid
figures, the annotated per-metric panels, the two scores, and the ``06_pairwise``
read-through cache. This module contributes a temperature→hash lookup, the row
table, and a markdown table. Nothing is reimplemented.

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

# `make_simplex3_figures` pins the BLAS thread count to 1 before it imports
# numpy, and that is load-bearing rather than tidy: unpinned, the behavioral
# level did not finish inside 50 minutes on this host. Importing numpy — or
# anything that imports numpy — above this line would let the default thread
# count win. So this import comes first, and everything numeric comes from it.
import make_simplex3_figures as S  # noqa: E402
from gen_simplex3 import temp_token  # noqa: E402

import numpy as np  # noqa: E402

#: The four metrics this figure set is about. Setting the module-level names is
#: what restricts the *whole* pipeline: `metric_row`, `emit`, `emit_detail` and
#: `_blocked` all read `S.METRIC_COLS`, so there is no per-call column list to
#: keep in sync and no way for one of them to drift onto the seven-metric grid.
METRICS = ("cosine", "frobenius", "euclidean", "cka")

#: The replicate count every sampled row is read at. See the module docstring.
SWEEP_REPLICATES = 8

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


def build_rows(hashes: dict[float, str]) -> dict[str, dict]:
    """``{row label: behavioral selector}``, greedy first then ascending T.

    The base selector is the one ``S.behavioral_cells`` builds; ``per_query``
    averages each query's replicates back to one row per query, which is the
    representation the request asked for and the only one at which all four
    metrics — CKA included — have more than one row to work with.

    Labels are display prose. They are never part of a cache key, so they can
    name the whole slice rather than a shorthand.
    """
    def sel(replicates: int, sampling: str, **over) -> dict:
        base = {"draw": S.DRAW, "max_new_tokens": S.MAX_NEW_TOKENS,
                "replicates": replicates, "sampling_hash": sampling,
                "embedder_hash": S.EMBEDDER, "replicate_reduction": "all",
                "view": "matrix", "normalize": "none", "representation": "matrix"}
        return dict(base, **over)

    per_query = {"replicate_reduction": "mean", "representation": "matrix",
                 "renormalize": True}

    # Greedy is `per generation`, not `per query`: at R=1 averaging a query's
    # replicates is the identity, so the two name one representation and only one
    # of them should appear. `behavioral_cells` omits the duplicate for the same
    # reason.
    rows = {"greedy · per generation": sel(1, S.SAMP_GREEDY)}
    for temperature, sampling in hashes.items():
        rows[f"T={temperature:.1f} · R={SWEEP_REPLICATES} · per query"] = sel(
            SWEEP_REPLICATES, sampling, **per_query)
    return rows


def slice_tag(label: str, hashes: dict[float, str]) -> str:
    """The filename tag for a row: ``greedy``, or ``t05`` for T=0.5.

    ``temp_token`` is the same function that named the sweep's experiment YAMLs
    and SLURM jobs, so a figure's filename points at the job that generated the
    text it is drawn from.
    """
    if label.startswith("greedy"):
        return "greedy"
    return temp_token(float(label.split("=", 1)[1].split(" ", 1)[0]))


def write_agreement_md(ranked, row_order, path: Path) -> None:
    """The scores as a table, in *row_order* — greedy, then ascending temperature.

    ``write_scores_csv`` is for diffing and keeps ``rank_surrogates``' dCor
    ordering; this is for reading, and the question it answers — does recovery
    change with temperature — is only legible if the rows ascend in temperature.
    Ranking them by score instead interleaves the ten temperatures and hides the
    trend the figure set exists to show. Both files are written from the same
    `SurrogateScore` objects, so they cannot disagree about a number.
    """
    scores: dict[str, dict[str, object]] = {}
    for s in ranked:
        scores.setdefault(s.row, {})[s.col] = s
    by_row = {row: scores[row] for row in row_order if row in scores}

    lines = [
        "# Behavioral level across the temperature sweep",
        "",
        "dCor runs 0→1, better higher: it scores the distance matrix and never "
        "embeds. The Procrustes residual runs 1→0, better lower: it scores the "
        "2-D MDS configuration each panel draws, so it inherits the distortion "
        "`stress` reports. Both are against the ground-truth simplex.",
        "",
        "| slice | " + " | ".join(f"{m} dCor" for m in METRICS)
        + " | " + " | ".join(f"{m} Procr." for m in METRICS) + " |",
        "|" + "---|" * (1 + 2 * len(METRICS)),
    ]
    for row in by_row:
        cells = by_row[row]
        dcor = [f"{cells[m].dcor:.4f}" if m in cells else "—" for m in METRICS]
        proc = [f"{cells[m].procrustes:.4f}" if m in cells else "—" for m in METRICS]
        lines.append(f"| {row} | " + " | ".join(dcor) + " | " + " | ".join(proc) + " |")

    lines += ["", "## Kruskal stress of the MDS fit each Procrustes residual describes", "",
              "| slice | " + " | ".join(METRICS) + " |",
              "|" + "---|" * (1 + len(METRICS))]
    for row in by_row:
        cells = by_row[row]
        stress = [f"{cells[m].stress:.4f}" if m in cells else "—" for m in METRICS]
        lines.append(f"| {row} | " + " | ".join(stress) + " |")

    path.write_text("\n".join(lines) + "\n")


def preflight(idx, ids, rows) -> None:
    """Resolve one sampled row and assert the shape the whole suite assumes.

    The R=8 per-query mean must reduce 800 rows to 100. If ``replicates`` were
    wrong the selector would still resolve — to a different, real slice — and the
    error would surface as a figure that looks plausible and is not the one it
    claims to be. Checking the shape once here turns that into an exception.
    """
    label = next(r for r in rows if not r.startswith("greedy"))
    reps, _order = S.resolve_ordered(idx, "behavioral", ids,
                                     behavioral_selector=rows[label])[:2]
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
            f"pre-flight: {label} has {n_rows} rows, expected "
            f"{S.DRAW['n_samples']} — a per-query mean over "
            f"{SWEEP_REPLICATES} replicates should give one row per query, so "
            f"{n_rows} means the replicate reduction did not happen")
    if not all(np.isfinite(np.asarray(r.matrix)).all() for r in reps):
        raise SystemExit(f"pre-flight: {label} has non-finite values")
    print(f"pre-flight: {label} → {len(reps)} × ({n_rows}, {dim}), all finite")


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

    rows = build_rows(hashes)
    preflight(idx, ids, rows)
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

    cells: dict[tuple[str, str], object] = {}
    for label, selector in rows.items():
        print(f"{label} …")
        got = S.metric_row(idx, "behavioral", ids, None, label=label,
                           behavioral_selector=selector)
        cells.update({(label, col): dm for col, dm in got.items()})

    # The overview: every slice against every metric, one figure each.
    S.emit("behavioral_temps", list(rows), cells, outdir,
           f"Behavioral level across sampling temperature (R={SWEEP_REPLICATES})")

    # Then one annotated panel per (slice, metric). `emit_detail` names its files
    # `fig_{level}_dm_{col}.png`, so passing the slice tag as the level is what
    # produces `fig_behavioral_t05_dm_cka.png` with no new plotting code.
    for label in rows:
        tag = slice_tag(label, hashes)
        S.emit_detail(f"behavioral_{tag}", label, cells, outdir,
                      f"Behavioral · {label}")

    print("scoring …")
    ranked = S.rank_surrogates(cells, ids)
    S.write_scores_csv({"behavioral": ranked}, outdir / "temperature_scores.csv")
    write_agreement_md(ranked, list(rows), outdir / "temperature_agreement.md")

    n_png = len(list(outdir.glob("*.png")))
    print(f"\nwrote {n_png} figures and {len(ranked)} scored cells to {outdir}")
    print(S.SUITE_CACHE.report())


if __name__ == "__main__":
    main()
